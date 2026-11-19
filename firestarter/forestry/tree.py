"""
Firestarter - Tree Anatomy & Ecological Growth
Implements DESIGN Sections 4.1 & 7.2: bark / sapwood / heartwood layers, living moisture,
growth stages, felling state and fungal fruiting bodies on dead timber.
"""

from enum import Enum
import math
from typing import Optional

import numpy as np

from firestarter.thermodynamics.wood_properties import WoodSpeciesType


class TreeGrowthStage(Enum):
    SAPLING = "Sapling"
    MATURE = "Mature Living"
    STANDING_DEAD = "Standing Dead"
    FALLEN_DECAYING = "Fallen Decaying"
    STUMP = "Stump"


SPECIES_SHAPE = {
    # species: (kind, height range m, radius range m)
    WoodSpeciesType.WHITE_PINE: ("conifer", (13.0, 21.0), (0.18, 0.30)),
    WoodSpeciesType.RED_CEDAR: ("conifer", (9.0, 15.0), (0.16, 0.26)),
    WoodSpeciesType.PAPER_BIRCH: ("deciduous", (9.0, 15.0), (0.10, 0.17)),
    WoodSpeciesType.WHITE_OAK: ("deciduous", (11.0, 17.0), (0.22, 0.36)),
}


class Tree:
    _ids = 0

    def __init__(
        self,
        species: WoodSpeciesType,
        position: np.ndarray,
        height_m: float = 12.0,
        trunk_radius_m: float = 0.22,
        stage: TreeGrowthStage = TreeGrowthStage.MATURE,
        yaw: float = 0.0,
        variant: int = 0,
    ):
        Tree._ids += 1
        self.id = Tree._ids
        self.species = species
        self.position = np.array(position, dtype=float)
        self.height_m = height_m
        self.trunk_radius_m = trunk_radius_m
        self.stage = stage
        self.yaw = yaw
        self.variant = variant
        self.age_years = 40.0 if stage != TreeGrowthStage.SAPLING else 1.0

        # Anatomy layers (radial fractions of trunk): bark 8 %, sapwood 40 %, heartwood 52 %
        self.bark_thickness_m = trunk_radius_m * 0.08
        self.sapwood_thickness_m = trunk_radius_m * 0.40
        self.heartwood_thickness_m = trunk_radius_m * 0.52

        if stage == TreeGrowthStage.MATURE:
            self.living_moisture_surface = 0.80
            self.living_moisture_core = 1.10          # 110 % dry-basis sap saturation
        elif stage == TreeGrowthStage.STANDING_DEAD:
            self.living_moisture_surface = 0.22
            self.living_moisture_core = 0.35
        elif stage == TreeGrowthStage.FALLEN_DECAYING:
            self.living_moisture_surface = 0.70
            self.living_moisture_core = 0.90
        else:
            self.living_moisture_surface = 0.85
            self.living_moisture_core = 1.20

        # Felling progress
        self.notch_depth_m = 0.0
        self.backcut_depth_m = 0.0
        self.fall_angle = 0.0          # degrees from vertical
        self.fall_velocity = 0.0       # deg/s
        self.fall_direction = 0.0      # radians (XZ heading of the fall)
        self.is_falling = False
        self.is_fallen = (stage == TreeGrowthStage.FALLEN_DECAYING)
        self.chop_shake = 0.0          # visual trunk shiver after a hit

        # Fungi fruiting on dead / fallen timber (bracket fungi / birch polypore)
        self.has_fungi = stage in (TreeGrowthStage.STANDING_DEAD, TreeGrowthStage.FALLEN_DECAYING)
        self.fungi_count = 3 if self.has_fungi else 0
        self.fungi_type = "Tinder Fungus" if species != WoodSpeciesType.PAPER_BIRCH else "Birch Polypore"
        self.bark_strips_left = 3 if stage == TreeGrowthStage.MATURE else 0
        self.bark_peeled = 0.0         # visual scar 0..1

    @property
    def kind(self) -> str:
        return SPECIES_SHAPE.get(self.species, ("conifer",))[0]

    @property
    def is_standing(self) -> bool:
        return self.stage in (TreeGrowthStage.MATURE, TreeGrowthStage.STANDING_DEAD, TreeGrowthStage.SAPLING) and not self.is_falling

    @property
    def collision_radius(self) -> float:
        if self.stage == TreeGrowthStage.SAPLING:
            return 0.0
        if self.is_fallen and self.stage != TreeGrowthStage.STUMP:
            return 0.0
        return self.trunk_radius_m + 0.12

    @property
    def total_trunk_mass_kg(self) -> float:
        """Green mass of the trunk (cylinder x density x living moisture)."""
        from firestarter.thermodynamics.wood_properties import get_wood_properties
        volume = math.pi * (self.trunk_radius_m ** 2) * self.height_m * 0.45   # taper factor
        dry = volume * get_wood_properties(self.species).bulk_density
        return dry * (1.0 + 0.5 * (self.living_moisture_surface + self.living_moisture_core))

    def describe(self) -> str:
        s = f"{self.species.value} ({self.stage.value.lower()})"
        if self.has_fungi and self.fungi_count > 0:
            s += f" - {self.fungi_count} {self.fungi_type.lower()} brackets"
        return s
