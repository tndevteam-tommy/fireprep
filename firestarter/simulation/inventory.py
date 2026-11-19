"""
Firestarter - Player Inventory

Fuel is carried as physical ``FuelItem`` records (species, size and moisture profile), so a
log you season on the rack stays seasoned in your pack and burns accordingly once placed.
Tools, consumables and bushcraft materials are tracked alongside.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from firestarter.thermodynamics.wood_properties import WoodSpeciesType, equilibrium_moisture_content
from firestarter.thermodynamics.combustion_body import CombustionBody, DEFAULT_DIMENSIONS, TINDER_DEFAULT_MASS
from firestarter.bushcraft.tinder_nest import TINDER_MATERIALS
from firestarter.bushcraft.fungi import SmolderingFungus

FUEL_ORDER = ["tinder", "kindling", "firewood", "log"]
FUEL_LABELS = {"tinder": "Tinder", "kindling": "Kindling", "firewood": "Split Firewood", "log": "Log"}

TINDER_SPECIES = {
    "cedar_shavings": WoodSpeciesType.RED_CEDAR,
    "birch_bark": WoodSpeciesType.PAPER_BIRCH,
    "pine_needles": WoodSpeciesType.WHITE_PINE,
    "dry_grass": WoodSpeciesType.WHITE_PINE,
    "cattail_fluff": WoodSpeciesType.WHITE_PINE,
    "feather_stick": WoodSpeciesType.WHITE_PINE,
    "dry_leaves": WoodSpeciesType.WHITE_OAK,
}
TINDER_MASS = {
    "cedar_shavings": 0.035, "birch_bark": 0.030, "pine_needles": 0.040,
    "dry_grass": 0.025, "cattail_fluff": 0.015, "feather_stick": 0.030, "dry_leaves": 0.030,
}


@dataclass
class FuelItem:
    kind: str
    species: WoodSpeciesType
    m_surface: float
    m_core: float
    length: float = 0.0
    radius: float = 0.0
    material: str = ""
    source: str = ""
    dry_mass: Optional[float] = None
    is_punk: bool = False

    def __post_init__(self):
        if self.length <= 0 or self.radius <= 0:
            L, r = DEFAULT_DIMENSIONS.get(self.kind, (0.4, 0.05))
            self.length = self.length or L
            self.radius = self.radius or r
        if self.dry_mass is None:
            if self.kind == "tinder":
                self.dry_mass = TINDER_MASS.get(self.material, TINDER_DEFAULT_MASS)
            else:
                from firestarter.thermodynamics.wood_properties import get_wood_properties
                rho = get_wood_properties(self.species).bulk_density
                self.dry_mass = rho * math.pi * self.radius ** 2 * self.length

    @property
    def name(self) -> str:
        if self.kind == "tinder":
            return TINDER_MATERIALS.get(self.material, ("Tinder",))[0]
        sp = "Punk wood" if self.is_punk or self.species == WoodSpeciesType.PUNK_WOOD else self.species.value
        return f"{sp} {FUEL_LABELS.get(self.kind, self.kind).lower()}"

    @property
    def moisture(self) -> float:
        return 0.3 * self.m_surface + 0.7 * self.m_core if self.kind not in ("tinder", "kindling") else self.m_surface

    @property
    def mass_kg(self) -> float:
        return float(self.dry_mass * (1.0 + self.moisture))

    def moisture_label(self) -> str:
        m = self.moisture
        if m < 0.16:
            return "seasoned"
        if m < 0.25:
            return "dry-ish"
        if m < 0.45:
            return "damp"
        return "green / soaked"

    def to_body(self, position: np.ndarray, orientation: np.ndarray, temp_c: float = 20.0) -> CombustionBody:
        geom = self.kind if self.kind != "punk" else "firewood"
        b = CombustionBody(
            self.species, dry_mass=self.dry_mass,
            initial_m_surface=self.m_surface, initial_m_core=self.m_core,
            initial_temp=temp_c, geometry_type=geom,
            position=np.array(position, dtype=float), orientation=np.array(orientation, dtype=float),
            dimensions=np.array([self.length, self.radius]),
        )
        b.source_item = self
        return b

    @staticmethod
    def from_body(body: CombustionBody, source: str = "") -> "FuelItem":
        return FuelItem(kind=body.geometry_type, species=body.species_type,
                        m_surface=body.m_surface, m_core=body.m_core,
                        length=body.length, radius=body.initial_radius,
                        material=getattr(getattr(body, "source_item", None), "material", ""),
                        source=source or getattr(getattr(body, "source_item", None), "source", ""), dry_mass=body.dry_mass,
                        is_punk=body.species_type == WoodSpeciesType.PUNK_WOOD)

    def equilibrate(self, hours: float, rh: float, temp_c: float, wetting: float = 0.0):
        """Moisture drifts toward the equilibrium moisture content (slow for big pieces)."""
        emc = equilibrium_moisture_content(rh, temp_c)
        # characteristic drying time grows with radius^2 (diffusion) - days for logs
        tau_h = max(2.0, (self.radius / 0.01) ** 2 * 6.0)
        k_s = 1.0 - math.exp(-hours / max(1.0, tau_h * 0.15))
        k_c = 1.0 - math.exp(-hours / tau_h)
        self.m_surface += (emc - self.m_surface) * k_s + wetting
        self.m_core += (emc - self.m_core) * k_c + wetting * 0.3
        self.m_surface = max(0.03, min(1.6, self.m_surface))
        self.m_core = max(0.03, min(1.6, self.m_core))


def make_tinder(material: str, moisture: float = 0.08, source: str = "") -> FuelItem:
    sp = TINDER_SPECIES.get(material, WoodSpeciesType.RED_CEDAR)
    return FuelItem("tinder", sp, moisture, moisture, material=material, source=source)


def make_kindling(species: WoodSpeciesType = WoodSpeciesType.WHITE_PINE, moisture: float = 0.10,
                  source: str = "", rng: Optional[np.random.Generator] = None) -> FuelItem:
    rng = rng or np.random.default_rng()
    return FuelItem("kindling", species, moisture, moisture + 0.02,
                    length=float(rng.uniform(0.26, 0.34)), radius=float(rng.uniform(0.006, 0.011)), source=source)


def make_firewood(species: WoodSpeciesType, m_surface: float = 0.12, m_core: float = 0.15,
                  source: str = "", rng: Optional[np.random.Generator] = None) -> FuelItem:
    rng = rng or np.random.default_rng()
    return FuelItem("firewood", species, m_surface, m_core,
                    length=float(rng.uniform(0.34, 0.42)), radius=float(rng.uniform(0.025, 0.035)), source=source)


def make_log(species: WoodSpeciesType, m_surface: float = 0.15, m_core: float = 0.18,
             source: str = "", rng: Optional[np.random.Generator] = None) -> FuelItem:
    rng = rng or np.random.default_rng()
    return FuelItem("log", species, m_surface, m_core,
                    length=float(rng.uniform(0.42, 0.50)), radius=float(rng.uniform(0.05, 0.065)), source=source)


class Inventory:
    MAX_CARRY_KG = 32.0

    def __init__(self):
        self.fuel: Dict[str, List[FuelItem]] = {k: [] for k in FUEL_ORDER}
        self.lighter_starts = 4           # README: a small lighter holds 4 fire starts
        self.lighter_burn_left = 0.0      # seconds of flame left on the current strike
        self.knife = "pocket"             # "pocket" or "steel"
        self.axe = "stone"                # "stone" or "steel"
        self.cordage = 1
        self.bark_strips = 2
        self.tarps = 0
        self.gasoline_kg = 0.0
        self.lighter_fluid_kg = 0.0
        self.fungi: List[SmolderingFungus] = []
        self.reference_editions = False
        self.paracord = 0

    # ---------------------------------------------------------------- fuel
    def add_fuel(self, item: FuelItem) -> bool:
        if self.total_mass() + item.mass_kg > self.MAX_CARRY_KG:
            return False
        self.fuel.setdefault(item.kind, []).append(item)
        return True

    def count(self, kind: str) -> int:
        return len(self.fuel.get(kind, []))

    def peek(self, kind: str) -> Optional[FuelItem]:
        lst = self.fuel.get(kind, [])
        if not lst:
            return None
        # hand the driest piece first - that is what a careful fire builder does
        return min(lst, key=lambda it: it.moisture)

    def take(self, kind: str) -> Optional[FuelItem]:
        it = self.peek(kind)
        if it is not None:
            self.fuel[kind].remove(it)
        return it

    def total_mass(self) -> float:
        return float(sum(it.mass_kg for lst in self.fuel.values() for it in lst) + self.gasoline_kg)

    def mean_moisture(self, kind: str) -> Optional[float]:
        lst = self.fuel.get(kind, [])
        if not lst:
            return None
        return float(np.mean([it.moisture for it in lst]))

    def equilibrate_all(self, hours: float, rh: float, temp_c: float):
        for lst in self.fuel.values():
            for it in lst:
                # the pack shelters items from rain; they slowly approach the air's EMC
                it.equilibrate(hours * 0.35, rh, temp_c)

    def summary(self) -> Dict[str, Any]:
        return {
            "fuel": {k: len(v) for k, v in self.fuel.items()},
            "lighter_starts": self.lighter_starts,
            "knife": self.knife, "axe": self.axe,
            "cordage": self.cordage, "bark_strips": self.bark_strips,
            "tarps": self.tarps, "gasoline_kg": round(self.gasoline_kg, 2),
            "fungi": len(self.fungi), "mass_kg": round(self.total_mass(), 1),
        }
