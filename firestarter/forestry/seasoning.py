"""
Firestarter - Forestry Seasoning & Raised Platform Physics
Implements DESIGN Section 4.2.

Moisture relaxes toward the local equilibrium moisture content (EMC, Hailwood-Horrobin):

    dM/dt = -(M - EMC) / tau,     tau ~ r^2 / D_eff(T, wind)

* Core drying time scales with radius^2 (diffusion) - splitting a log into firewood seasons
  it ~4x faster; temperature enters through an Arrhenius factor (E ~ 30 kJ/mol), so winter
  drying is ~5x slower than summer; wind speeds up surface evaporation.
* Raised platforms eliminate ground capillary transfer (J_ground = 0).
* A waterproof tarp blocks rain (P_rain = 0) while the open sides keep the airflow.
* Calibration: a green (M_core = 1.10) 12 cm log on a tarped summer rack reaches the ideal
  ~12 % EMC after ~60 days (DESIGN 4.2).
* Logs left on bare soil in autumn / winter wick water up to ~95 % and - while wet and above
  freezing - accumulate fungal decay until they become spongy punk wood.
"""

import math
from typing import List, Dict, Any, Optional

import numpy as np

from firestarter.thermodynamics.wood_properties import WoodSpeciesType, equilibrium_moisture_content

E_DRY = 30.0e3
R_GAS = 8.314
TAU_CORE_REF_DAYS = 18.5       # 6 cm radius log, 29 C, 2 m/s wind, sheltered
RADIUS_REF = 0.06


def _radius(item) -> float:
    if hasattr(item, "initial_radius"):
        return float(item.initial_radius)
    return float(getattr(item, "radius", 0.05))


def _set_moisture(item, surface: float, core: float):
    if hasattr(item, "set_moisture"):
        item.set_moisture(surface=surface, core=core)
    else:
        item.m_surface = surface
        item.m_core = core


def drying_time_constants(radius: float, temp_c: float, wind: float) -> (float, float):
    """Returns (tau_surface_days, tau_core_days)."""
    temp_f = math.exp(-E_DRY / R_GAS * (1.0 / (max(-30.0, temp_c) + 273.15) - 1.0 / 302.15))
    wind_f = (1.0 + 0.15 * max(0.0, wind)) / 1.3
    tau_core = TAU_CORE_REF_DAYS * (max(0.004, radius) / RADIUS_REF) ** 2 / (temp_f * (0.6 + 0.4 * wind_f))
    tau_surf = max(0.5, tau_core * 0.12 / wind_f)
    return tau_surf, max(0.6, tau_core)


def step_moisture(item, days: float, temp_c: float, rh: float, wind: float,
                  rain: float = 0.0, on_ground: bool = False, covered: bool = False,
                  soil_moisture: float = 0.7, sub_steps: int = 4):
    r = _radius(item)
    m_s, m_c = float(item.m_surface), float(item.m_core)
    emc = equilibrium_moisture_content(rh, temp_c)
    tau_s, tau_c = drying_time_constants(r, temp_c, wind)
    is_sponge = getattr(item, "species", getattr(item, "species_type", None)) == WoodSpeciesType.PUNK_WOOD
    sponge = 3.0 if is_sponge else 1.0
    h = days / sub_steps
    for _ in range(sub_steps):
        target_s = emc
        wet_s = 0.0
        if not covered and rain > 0.0:
            wet_s += 0.25 * rain * sponge        # rain soaks the shell (per day at full rain)
        if on_ground:
            wet_s += 0.06 * soil_moisture * sponge  # capillary wicking from the soil
        # exact solution of dM/dt = -(M - EMC)/tau + wetting over the sub-step
        m_eq = target_s + wet_s * tau_s
        m_s = m_eq + (m_s - m_eq) * math.exp(-h / tau_s)
        # the core exchanges with the shell
        m_c += (m_s - m_c) * (1.0 - math.exp(-h / tau_c))
        cap = 0.95 if not is_sponge else 1.4
        m_s = max(0.03, min(cap + 0.1, m_s))
        m_c = max(0.03, min(cap, m_c))
    _set_moisture(item, m_s, m_c)


class RaisedDryingRack:
    def __init__(self, position: np.ndarray, has_tarp: bool = False, capacity: int = 10):
        self.position = np.array(position, dtype=float)
        self.has_tarp = has_tarp
        self.capacity = capacity
        self.logs: List[Any] = []
        self.history: List[Dict[str, float]] = []

    def add_log(self, log) -> bool:
        if len(self.logs) >= self.capacity:
            return False
        if hasattr(log, "is_on_ground"):
            log.is_on_ground = False
            log.is_tarp_covered = self.has_tarp
        self.logs.append(log)
        return True

    def remove_log(self, log):
        if log in self.logs:
            self.logs.remove(log)
            if hasattr(log, "is_on_ground"):
                log.is_on_ground = True
                log.is_tarp_covered = False

    def take_driest(self):
        if not self.logs:
            return None
        it = min(self.logs, key=lambda l: 0.3 * l.m_surface + 0.7 * l.m_core)
        self.remove_log(it)
        return it

    def install_tarp(self):
        self.has_tarp = True
        for log in self.logs:
            if hasattr(log, "is_tarp_covered"):
                log.is_tarp_covered = True

    def mean_moisture(self) -> Optional[float]:
        if not self.logs:
            return None
        return float(np.mean([0.3 * l.m_surface + 0.7 * l.m_core for l in self.logs]))

    def update_seasoning(self, days_passed: float, season: str = "Summer", rain_intensity: float = 0.0,
                         ground_humidity: float = 0.6, wind_speed: float = 2.0,
                         temp_c: Optional[float] = None, rh: Optional[float] = None, rain_fraction: Optional[float] = None):
        """Advance seasoning.  Raised: no ground contact.  Tarp: no rain."""
        if temp_c is None:
            temp_c = {"Summer": 28.0, "Autumn": 11.0, "Winter": -7.0, "Spring": 14.0}.get(season, 15.0)
        if rh is None:
            rh = {"Summer": 0.45, "Autumn": 0.78, "Winter": 0.84, "Spring": 0.72}.get(season, 0.6)
        rain = rain_intensity if rain_fraction is None else rain_fraction
        for log in self.logs:
            step_moisture(log, days_passed, temp_c, rh, wind_speed, rain=rain,
                          on_ground=False, covered=self.has_tarp)
        if self.logs:
            self.history.append({"days": days_passed, "mean": self.mean_moisture()})


class GroundSeasoningSimulator:
    """
    Unshielded logs lying directly on forest soil: capillary wicking + rain soaking.  While the
    wood is wet (> ~35 %) and above freezing, fungi colonise it; enough accumulated decay turns
    sound wood into soaked, spongy punk wood.
    """

    DECAY_DAYS_TO_PUNK = 55.0

    @staticmethod
    def step_ground_log(log, days_passed: float, season: str, rain_intensity: float,
                        soil_moisture: float = 0.75, temp_c: Optional[float] = None, rh: Optional[float] = None,
                        wind: float = 1.5):
        if hasattr(log, "is_on_ground") and not log.is_on_ground:
            return
        if temp_c is None:
            temp_c = {"Summer": 27.0, "Autumn": 10.0, "Winter": -7.0, "Spring": 13.0}.get(season, 12.0)
        if rh is None:
            rh = {"Summer": 0.5, "Autumn": 0.82, "Winter": 0.86, "Spring": 0.78}.get(season, 0.7)
        wet_season = season in ("Autumn", "Winter", "Spring")
        soil = soil_moisture if wet_season else soil_moisture * 0.45
        step_moisture(log, days_passed, temp_c, rh, wind * 0.4, rain=rain_intensity,
                      on_ground=True, covered=False, soil_moisture=soil)
        # fungal decay accumulates while damp (winter slows it; freezing stops it)
        m = 0.3 * log.m_surface + 0.7 * log.m_core
        activity = max(0.0, min(1.0, (m - 0.30) / 0.25)) * max(0.0, min(1.0, (temp_c + 2.0) / 15.0))
        # big stems rot far more slowly than a small log (fungi work inward from the surface)
        decay = getattr(log, "decay_days", 0.0) + activity * days_passed * (1.4 if wet_season else 1.0) * getattr(log, "decay_rate", 1.0)
        setattr(log, "decay_days", decay)
        if wet_season and log.m_core > 0.55:
            # sustained soaking in the cold, wet season drives towards ~95 %
            _set_moisture(log, min(0.98, log.m_surface + 0.004 * days_passed), min(0.95, log.m_core + 0.004 * days_passed))
        if decay >= GroundSeasoningSimulator.DECAY_DAYS_TO_PUNK and not getattr(log, "is_decayed", False):
            GroundSeasoningSimulator.make_punk(log)

    @staticmethod
    def make_punk(log):
        if hasattr(log, "convert_to_punk"):
            log.convert_to_punk()
        else:
            log.species = WoodSpeciesType.PUNK_WOOD
            log.is_punk = True
            if getattr(log, "dry_mass", None):
                log.dry_mass *= 0.4
        if hasattr(log, "species_type"):
            log.species_type = WoodSpeciesType.PUNK_WOOD
        log.is_decayed = True
