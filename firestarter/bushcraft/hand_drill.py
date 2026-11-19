"""
Firestarter - Hand Drill & Stone Bearing
Implements DESIGN Section 3.2: a long (40-60 cm) spindle spun directly between the palms.

* Rubbing the palms back and forth at v ~ 0.6-1.2 m/s spins a 1.8 cm spindle at
  omega = v / r ~ 70-130 rad/s at the peak of each stroke (the direction reverses every
  stroke, so the mean speed is ~2/pi of the peak).
* The hands travel down the spindle while spinning and must be reset to the top, which
  costs a moment of cooling.  A smooth river stone used as a palm bearing lets the
  driller press harder (~45 N instead of ~32 N) and slows the slide down the spindle.
* The spindle drills into the same hearthboard socket / V-notch as the bow drill, so the
  socket heat, the char dust and the ember all share the BowDrillSystem socket model
  (P = mu F_N omega r).  Hand drilling is exhausting: stamina drains several times faster.
"""

from typing import Optional, Dict, Any

from firestarter.bushcraft.bow_drill import BowDrillSystem

MEAN_OF_OSCILLATION = 0.64      # 2/pi: mean |sin| of the back-and-forth palm stroke


class HandDrillSystem:
    def __init__(
        self,
        spindle_length_cm: float = 50.0,
        spindle_diameter_cm: float = 1.8,
        hearthboard_moisture: float = 0.08,
        has_stone_handle: bool = False,
        ambient_temp: float = 20.0,
        socket: Optional[BowDrillSystem] = None,
    ):
        self.spindle_length_cm = spindle_length_cm
        self.spindle_radius_m = (spindle_diameter_cm / 2.0) / 100.0
        self.hearthboard_moisture = hearthboard_moisture
        self.has_stone_handle = has_stone_handle
        self.socket = socket if socket is not None else BowDrillSystem(hearthboard_moisture=hearthboard_moisture,
                                                                       has_v_notch=True, ambient_temp=ambient_temp)
        self.hand_height_cm = spindle_length_cm * 0.9
        self.stamina_cost_rate = 2.4   # stamina / s of vigorous spinning (bow drilling: ~0.9 / s)
        self.resets = 0
        self.last_omega = 0.0

    # ---- shared socket state (exposed for the UI / tests) -----------------------------
    @property
    def ambient_temp(self) -> float:
        return self.socket.ambient_temp

    @ambient_temp.setter
    def ambient_temp(self, v: float):
        self.socket.ambient_temp = v

    @property
    def socket_temp(self) -> float:
        return self.socket.socket_temp

    @property
    def wood_dust_mass(self) -> float:
        return self.socket.wood_dust_mass

    @property
    def char_fraction(self) -> float:
        return self.socket.dust_char_fraction

    @property
    def has_ember(self) -> bool:
        return self.socket.has_ember

    @property
    def ember_temp(self) -> float:
        return self.socket.ember_temp

    def drill_step(self, hand_spin_speed_rad_s: float, player_stamina: float, dt: float,
                   stone: Optional[bool] = None) -> Dict[str, Any]:
        """One slice of spinning.  hand_spin_speed_rad_s is the peak spindle speed produced by
        the palms (v / r); stone overrides has_stone_handle for this step."""
        if player_stamina <= 0.0:
            return {"success": False, "reason": "Exhausted! Rest to recover stamina.",
                    "socket_temp": self.socket_temp, "has_ember": self.has_ember}
        use_stone = self.has_stone_handle if stone is None else stone
        downward_force = 45.0 if use_stone else 32.0
        omega = max(0.0, min(140.0, hand_spin_speed_rad_s)) * MEAN_OF_OSCILLATION
        self.last_omega = omega
        travel_speed = 1.2 if use_stone else 4.0       # cm/s down the spindle
        if omega > 1.0:
            self.hand_height_cm -= travel_speed * dt
        reset_required = False
        if self.hand_height_cm < 8.0:
            # float the hands back to the top of the spindle: a moment without pressure
            self.hand_height_cm = self.spindle_length_cm * 0.9
            self.resets += 1
            reset_required = True
            omega *= 0.2
        self.socket.hearthboard_moisture = self.hearthboard_moisture
        res = self.socket.spin(omega, downward_force, dt, radius=self.spindle_radius_m)
        res.update({
            "hand_height_cm": self.hand_height_cm,
            "reset_required": reset_required,
            "stamina_spent": self.stamina_cost_rate * (1.0 if omega > 1.0 else 0.0) * dt,
            "char_fraction": self.socket.dust_char_fraction,
            "dust_mass": self.socket.wood_dust_mass,
        })
        return res

    def idle(self, dt: float):
        self.socket.idle(dt)

    def extract_ember(self) -> Optional[float]:
        return self.socket.extract_ember()
