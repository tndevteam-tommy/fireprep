"""
Firestarter - Bow Drill Friction Physics & Coal Formation
Implements DESIGN Section 3.1:

    P = mu * F_N * omega * r_spindle

* Frictional power heats a tiny socket region (~0.45 g of wood at the spindle tip).  Heat
  leaks into the board, spindle and air through an effective conductance (~0.018 W/K), so
  the socket climbs toward T_amb + P/G with a ~25-40 s time constant - good, steady bowing
  for 30-90 s makes a coal, exactly like the real thing.
* Damp wood must first be dried: energy goes into evaporation at ~100 C (2260 kJ/kg).
* Abraded wood dust collects in the 45-degree V-notch (no notch -> dust scatters, no coal).
* Above ~220 C the dust chars; when the socket exceeds 380 C with >= ~0.08 g of charred dust
  in the notch, the dust oxidises into a self-sustaining 1-3 mm ember.
* Cordage wears with load and speed and eventually snaps.
"""

import math
from typing import Optional, Dict, Any

from firestarter.config import (
    TEMP_EMBER_FORMATION, CP_DRY_WOOD, H_VAP_WATER,
    BOW_DRILL_FRICTION_MU_RANGE, BOW_DRILL_FORCE_RANGE,
    BOW_DRILL_OMEGA_RANGE, BOW_DRILL_SPINDLE_RADIUS
)

SOCKET_MASS_KG = 0.00045
SOCKET_LOSS_W_PER_K = 0.018
DUST_FOR_EMBER_G = 0.08


class BowDrillSystem:
    def __init__(
        self,
        hearthboard_dry_mass: float = 0.12,
        hearthboard_moisture: float = 0.08,
        spindle_radius: float = BOW_DRILL_SPINDLE_RADIUS,
        spindle_moisture: float = 0.07,
        has_v_notch: bool = False,
        cordage_condition: float = 1.0,  # 0.0 to 1.0
        ambient_temp: float = 20.0,
    ):
        # 5 physical components: 1. Bow, 2. Cordage, 3. Spindle, 4. Hearthboard, 5. Bearing Block
        self.spindle_radius = spindle_radius
        self.hearthboard_moisture = hearthboard_moisture
        self.spindle_moisture = spindle_moisture
        self.has_v_notch = has_v_notch
        self.cordage_condition = cordage_condition
        self.ambient_temp = ambient_temp

        self.socket_temp = ambient_temp
        self.socket_water_g = self.hearthboard_moisture * SOCKET_MASS_KG * 1000.0
        self.wood_dust_mass = 0.0           # grams collected in the notch
        self.dust_char_fraction = 0.0       # 0..1
        self.has_ember = False
        self.ember_mass = 0.0               # grams
        self.ember_temp = ambient_temp
        self.total_strokes = 0
        self.is_smoking = False
        self.smoke_intensity = 0.0
        self.last_power = 0.0
        self.last_omega = 0.0
        self.notch_depth = 0.0              # visual: 0..1 burned socket depth
        self.sockets_used = 0

    def carve_v_notch(self) -> bool:
        """Player carves a 45-degree wedge into the hearthboard edge with the pocket knife."""
        self.has_v_notch = True
        return True

    def calculate_friction_power(self, normal_force_n: float, angular_velocity_rad_s: float,
                                 friction_mu: float = 0.55) -> float:
        """P = mu * F_N * omega * r_spindle  (damp wood lubricates and lowers mu)."""
        mu = max(BOW_DRILL_FRICTION_MU_RANGE[0], min(BOW_DRILL_FRICTION_MU_RANGE[1], friction_mu))
        f_n = max(0.0, normal_force_n)
        omega = max(0.0, angular_velocity_rad_s)
        moisture_penalty = max(0.4, 1.0 - max(0.0, self.hearthboard_moisture + self.spindle_moisture - 0.16) * 1.5)
        return mu * f_n * omega * self.spindle_radius * moisture_penalty

    def stroke(self, stroke_length_m: float, stroke_speed_m_s: float, downward_force_n: float,
               dt: float) -> Dict[str, Any]:
        """
        Bow stroke: the cord wrapped on the spindle converts bow speed v into spindle spin
        omega = v / r  (clamped to the DESIGN's 10-25 rad/s sustained range).
        """
        if self.cordage_condition <= 0.0:
            return {"success": False, "reason": "Cordage snapped! Twist new bark cordage.",
                    "socket_temp": self.socket_temp, "has_ember": self.has_ember}
        if stroke_speed_m_s <= 0.01:
            self.idle(dt)
            return self._result(0.0)

        omega = stroke_speed_m_s / max(0.005, self.spindle_radius)
        omega = min(BOW_DRILL_OMEGA_RANGE[1], max(BOW_DRILL_OMEGA_RANGE[0] * 0.4, omega))
        f_n = min(BOW_DRILL_FORCE_RANGE[1], max(BOW_DRILL_FORCE_RANGE[0] * 0.5, downward_force_n))
        cordage_wear = 0.0009 * (f_n / 50.0) * (omega / 15.0) * dt
        self.cordage_condition = max(0.0, self.cordage_condition - cordage_wear)
        self.total_strokes += 1
        return self.spin(omega, f_n, dt)

    def spin(self, omega: float, normal_force_n: float, dt: float, radius: Optional[float] = None) -> Dict[str, Any]:
        """Any spindle turning in the socket (bow drill or hand drill) at angular speed omega."""
        r0 = self.spindle_radius
        if radius is not None:
            self.spindle_radius = radius
        power = self.calculate_friction_power(normal_force_n, omega)
        self.spindle_radius = r0
        self.last_power = power
        self.last_omega = omega
        self._heat(power, dt)
        # abraded dust scales with the sliding distance and the load (Archard wear)
        dust = 0.0032 * (normal_force_n / 50.0) * (omega * (radius or r0) / 0.15) * dt
        if self.has_v_notch:
            self.wood_dust_mass = min(0.4, self.wood_dust_mass + dust)
        self.notch_depth = min(1.0, self.notch_depth + dust * 0.6)
        self._char_and_ember(dt)
        return self._result(power)

    def _heat(self, power: float, dt: float):
        c = SOCKET_MASS_KG * (CP_DRY_WOOD + 4184.0 * self.socket_water_g / (SOCKET_MASS_KG * 1000.0))
        loss = SOCKET_LOSS_W_PER_K * (self.socket_temp - self.ambient_temp)
        e = (power - loss) * dt
        new_t = self.socket_temp + e / c
        if new_t > 100.0 and self.socket_water_g > 1e-6:
            excess = (new_t - 100.0) * c
            evap_g = min(self.socket_water_g, excess / H_VAP_WATER * 1000.0)
            self.socket_water_g -= evap_g
            new_t = 100.0 + (excess - evap_g / 1000.0 * H_VAP_WATER) / c
            # the board keeps wicking moisture into the hot socket
            self.socket_water_g += self.hearthboard_moisture * 0.0006 * dt
        self.socket_temp = max(self.ambient_temp - 1.0, new_t)

    def _char_and_ember(self, dt: float):
        self.is_smoking = self.socket_temp >= 200.0
        self.smoke_intensity = max(0.0, min(1.0, (self.socket_temp - 180.0) / 250.0))
        if self.socket_temp >= 220.0 and self.wood_dust_mass > 0.0:
            rate = 0.12 * ((self.socket_temp - 200.0) / 180.0)
            self.dust_char_fraction = min(1.0, self.dust_char_fraction + rate * dt)
        if (self.has_v_notch and not self.has_ember and self.socket_temp >= TEMP_EMBER_FORMATION
                and self.dust_char_fraction >= 0.70 and self.wood_dust_mass >= DUST_FOR_EMBER_G):
            self.has_ember = True
            self.ember_mass = min(0.25, self.wood_dust_mass)
            self.ember_temp = self.socket_temp

    def idle(self, dt: float):
        """Not bowing: the socket cools; a formed ember smoulders on its own for a while."""
        c = SOCKET_MASS_KG * CP_DRY_WOOD
        self.socket_temp += -SOCKET_LOSS_W_PER_K * (self.socket_temp - self.ambient_temp) * dt / c
        self.is_smoking = self.socket_temp > 200.0 or self.has_ember
        self.smoke_intensity = max(0.0, min(1.0, (self.socket_temp - 180.0) / 250.0))
        if self.has_ember:
            # an unattended coal slowly cools: ~2-3 minutes of working time
            self.ember_temp = max(self.ambient_temp, self.ember_temp - 1.6 * dt)
            if self.ember_temp < 260.0:
                self.has_ember = False
                self.wood_dust_mass *= 0.3
        self.socket_water_g = min(self.hearthboard_moisture * SOCKET_MASS_KG * 1000.0,
                                  self.socket_water_g + 0.00002 * dt)

    def _result(self, power: float) -> Dict[str, Any]:
        return {
            "success": True,
            "power_watts": power,
            "socket_temp": self.socket_temp,
            "dust_mass_g": self.wood_dust_mass,
            "char_level": self.dust_char_fraction,
            "has_ember": self.has_ember,
            "ember_temp": self.ember_temp if self.has_ember else 0.0,
            "cordage_condition": self.cordage_condition,
            "is_smoking": self.is_smoking,
        }

    def extract_ember(self) -> Optional[float]:
        """Tips the glowing coal out of the V-notch (into a tinder nest)."""
        if not self.has_ember:
            return None
        ember_t = self.ember_temp
        self.has_ember = False
        self.wood_dust_mass = 0.0
        self.dust_char_fraction = 0.0
        self.socket_temp = max(self.ambient_temp, self.socket_temp * 0.6)
        self.sockets_used += 1
        return ember_t

    def replace_cordage(self):
        self.cordage_condition = 1.0
