"""
Firestarter - Tinder Nest & Breathing Dynamics
Implements DESIGN Section 3.3: the breathing mechanic ('F' key).

An ember (from a bow drill, hand drill or smouldering fungus) is folded into a nest of dry
fine tinder.  Each breath forces air over the coal:

* Too weak (< 0.8 m/s):  convective cooling outruns oxidation (dQ/dt < 0).
* Optimal (~1.5-4 m/s):  boundary-layer oxygen supply multiplies char oxidation; the glowing
                         core spreads into the surrounding fibres.
* Too strong (> 6 m/s):  hot char dust is physically blown out of the nest.

Between breaths the coal keeps smouldering slowly.  Once the glowing core exceeds 550 C the
volatile gases from the fibres flash into open flame.
"""

import math
from typing import Dict, Any

from firestarter.config import (
    TEMP_TINDER_IGNITION, BREATH_O2_FRACTION,
    BREATH_OPTIMAL_VELOCITY, BREATH_MAX_VELOCITY, BREATH_MIN_VELOCITY
)
from firestarter.thermodynamics.airflow_draft import AirflowDraftSolver

TINDER_MATERIALS = {
    # material: (display name, moisture of extinction, spread factor)
    "cedar_shavings": ("Cedar shavings", 0.30, 1.0),
    "birch_bark": ("Birch bark curls", 0.45, 1.15),   # betulin burns even when damp
    "pine_needles": ("Dry pine needles", 0.28, 0.9),
    "dry_grass": ("Dry grass", 0.25, 1.25),
    "cattail_fluff": ("Cattail fluff", 0.22, 1.4),
    "feather_stick": ("Feather-stick shavings", 0.30, 1.0),
    "dry_leaves": ("Crumbled dry leaves", 0.22, 0.95),   # autumn litter: burns fast, hates damp
}


class TinderNest:
    def __init__(
        self,
        material: str = "cedar_shavings",
        dry_mass_g: float = 25.0,
        moisture: float = 0.08,
        has_ember: bool = False,
        ember_temp_c: float = 0.0,
        ambient_temp: float = 20.0,
    ):
        self.material = material if material in TINDER_MATERIALS else "cedar_shavings"
        self.dry_mass_g = dry_mass_g
        self.moisture = moisture
        self.has_ember = has_ember
        self.ambient_temp = ambient_temp
        self.core_temp = ember_temp_c if has_ember else ambient_temp
        self.glow_area_cm2 = 0.3 if has_ember else 0.0   # glowing region grows as you blow

        self.is_flaming = False
        self.is_destroyed = False
        self.flame_height = 0.0
        self.smoke_density = 0.0
        self.glow_intensity = 0.0
        self.breath_count = 0
        self.last_breath_status = ""
        self.time_since_breath = 99.0

    @property
    def display_name(self) -> str:
        return TINDER_MATERIALS[self.material][0]

    def insert_ember(self, ember_temp: float):
        """Folds a glowing coal into the centre of the nest."""
        self.has_ember = True
        self.core_temp = max(380.0, ember_temp)
        self.glow_area_cm2 = 0.3
        self.glow_intensity = 0.4
        self.is_destroyed = False

    def update(self, dt: float, ambient_temp: float = 20.0):
        self.ambient_temp = ambient_temp
        self.time_since_breath += dt
        if not self.has_ember or self.is_destroyed:
            self.glow_intensity = 0.0
            self.smoke_density = max(0.0, self.smoke_density - dt)
            return

        if self.is_flaming:
            self.flame_height = min(0.35, self.flame_height + 0.12 * dt)
            self.core_temp = min(850.0, self.core_temp + 60.0 * dt)
            self.dry_mass_g = max(0.0, self.dry_mass_g - 1.8 * dt)
            self.glow_intensity = 1.0
            if self.dry_mass_g <= 1.0:
                self.is_flaming = False
                self.has_ember = False
                self.is_destroyed = True
            return

        # Smouldering between breaths: natural convection supplies a little oxygen; damp fibres
        # sap heat.  A fresh coal survives ~1.5-3 minutes unattended in dry tinder.
        gain = 1.2 * (1.0 - min(1.0, self.moisture * 2.5)) * (1.0 + 0.15 * self.glow_area_cm2)
        loss = 0.006 * (self.core_temp - ambient_temp) * (1.0 + 3.0 * self.moisture)
        self.core_temp = max(ambient_temp, self.core_temp + (gain - loss) * dt)
        if self.core_temp < 250.0:
            self.has_ember = False
            self.glow_area_cm2 = 0.0
        self.glow_intensity = max(0.0, min(1.0, (self.core_temp - 250.0) / 450.0))
        self.smoke_density = max(0.05, min(1.0, (self.core_temp - 150.0) / 350.0)) * (0.5 + self.glow_area_cm2 / 6.0)

    def blow(self, breath_velocity: float, dt: float = 0.6) -> Dict[str, Any]:
        """One breath of duration dt at the given air velocity (m/s)."""
        if not self.has_ember or self.is_destroyed:
            return {"status": "no_ember", "is_flaming": False, "core_temp": self.core_temp}
        self.breath_count += 1
        self.time_since_breath = 0.0

        if breath_velocity > BREATH_MAX_VELOCITY:
            self.has_ember = False
            self.is_destroyed = True
            self.glow_intensity = 0.0
            self.last_breath_status = "blown_out"
            return {"status": "blown_out", "reason": "Breath too strong! The ember scattered.",
                    "is_flaming": False, "core_temp": self.core_temp}

        # fine char dust + fibres: ~8x the reactive surface of solid charcoal
        net_w, o2, _ = AirflowDraftSolver.calculate_breath_effect(breath_velocity, self.core_temp,
                                                                  self.ambient_temp, reactivity=8.0)
        # the glowing area and fibre moisture scale the effect
        area_scale = 0.6 + self.glow_area_cm2 / 2.0
        damp = 1.0 - min(0.9, self.moisture * 2.2)
        heat_capacity = 0.35 + 0.05 * self.glow_area_cm2      # J/K of the glowing fibres
        if breath_velocity < BREATH_MIN_VELOCITY:
            status = "too_weak"
            self.core_temp = max(self.ambient_temp, self.core_temp - 25.0 * dt)
        elif breath_velocity <= BREATH_OPTIMAL_VELOCITY + 1.5:
            status = "optimal"
            dT = max(0.0, net_w) * area_scale * damp * dt / heat_capacity
            self.core_temp += min(140.0, dT)
            spread = TINDER_MATERIALS[self.material][2]
            self.glow_area_cm2 = min(8.0, self.glow_area_cm2 + 0.8 * spread * damp * dt)
        else:
            status = "too_strong"
            self.core_temp += max(-40.0, min(20.0, net_w * 0.2 * dt))
        self.glow_intensity = max(0.0, min(1.0, (self.core_temp - 250.0) / 450.0))
        self.smoke_density = min(1.0, 0.3 + self.glow_area_cm2 / 8.0)

        m_ext = TINDER_MATERIALS[self.material][1]
        if self.core_temp >= TEMP_TINDER_IGNITION and self.moisture < m_ext and not self.is_flaming:
            self.is_flaming = True
            self.flame_height = 0.15
            status = "ignited"
        self.last_breath_status = status
        return {
            "status": status,
            "core_temp": self.core_temp,
            "glow_intensity": self.glow_intensity,
            "is_flaming": self.is_flaming,
            "smoke_density": self.smoke_density,
            "glow_area_cm2": self.glow_area_cm2,
        }
