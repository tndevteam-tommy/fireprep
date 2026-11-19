"""
Firestarter - Forest Ecosystem & Lifecycle Dynamics
Implements DESIGN Section 7.2: procedural forest (species by micro-climate zone), seed
dispersal along seasonal wind vectors, ash-fertilised sapling growth, maturation, death,
and fungal fruiting on standing dead timber.
"""

import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from firestarter.forestry.tree import Tree, TreeGrowthStage, SPECIES_SHAPE
from firestarter.thermodynamics.wood_properties import WoodSpeciesType
from firestarter.simulation import terrain


class AshDepositZone:
    def __init__(self, position: np.ndarray, ash_mass_kg: float = 1.0):
        self.position = np.array(position, dtype=float)
        self.ash_mass_kg = ash_mass_kg
        # Potassium / calcium carbonates and phosphate raise soil fertility
        self.fertility_boost = min(3.0, 1.0 + ash_mass_kg * 0.5)


def _zone_species_weights(x: float, z: float) -> Dict[WoodSpeciesType, float]:
    W = WoodSpeciesType
    w = {W.WHITE_PINE: 1.0, W.RED_CEDAR: 0.6, W.PAPER_BIRCH: 0.7, W.WHITE_OAK: 0.7}
    if abs(z - float(terrain.creek_centerline_z(x))) < 22.0:      # creek: cedar & birch love wet feet
        w[W.RED_CEDAR] += 2.2
        w[W.PAPER_BIRCH] += 1.2
        w[W.WHITE_OAK] *= 0.4
    if z > 32.0:                                                   # sun-baked ridge: pine
        w[W.WHITE_PINE] += 2.5
        w[W.RED_CEDAR] *= 0.3
    if math.hypot(x + 40.0, z - 10.0) < 30.0:                      # hardwood basin
        w[W.WHITE_OAK] += 2.5
        w[W.PAPER_BIRCH] += 1.5
        w[W.WHITE_PINE] *= 0.5
    return w


class ForestEcosystem:
    def __init__(self, area_size_m: float = 150.0, seed: int = 42, n_trees: int = 230):
        self.area_size_m = area_size_m
        self.rng = np.random.default_rng(seed)
        self.trees: List[Tree] = []
        self.ash_zones: List[AshDepositZone] = []
        self.fallen_decay_logs: List[Dict[str, Any]] = []
        self.version = 0          # bumped whenever trees change (renderer cache key)
        self._populate_forest(n_trees)

    def _populate_forest(self, n_trees: int):
        rng = self.rng
        half = self.area_size_m * 0.5
        placed: List[Tuple[float, float, float]] = []
        attempts = 0
        while len(self.trees) < n_trees and attempts < n_trees * 40:
            attempts += 1
            x = float(rng.uniform(-half, half))
            z = float(rng.uniform(-half, half))
            r0 = math.hypot(x, z)
            if r0 < 13.0:                                      # camp clearing
                continue
            if abs(z - float(terrain.creek_centerline_z(x))) < 3.5:   # not in the water
                continue
            if terrain.slope_degrees(x, z) > 30.0:
                continue
            # keep the view from camp toward the north-west open-ish (sunset vista)
            if 13.0 < r0 < 28.0 and -2.7 < math.atan2(z, x) < -1.7 and rng.random() < 0.75:
                continue
            w = _zone_species_weights(x, z)
            sp_list = list(w.keys())
            p = np.array([w[s] for s in sp_list])
            sp = sp_list[int(rng.choice(len(sp_list), p=p / p.sum()))]
            _, hr, rr = SPECIES_SHAPE[sp]
            t = float(rng.random())
            h = hr[0] + (hr[1] - hr[0]) * t
            rad = rr[0] + (rr[1] - rr[0]) * (0.6 * t + 0.4 * float(rng.random()))
            if any((x - px) ** 2 + (z - pz) ** 2 < (pr + rad + 1.2) ** 2 for px, pz, pr in placed[-400:]):
                continue
            stage = TreeGrowthStage.MATURE
            roll = float(rng.random())
            if roll < 0.09:
                stage = TreeGrowthStage.STANDING_DEAD
                h *= 0.75
            y = float(terrain.height(x, z))
            tree = Tree(sp, np.array([x, y, z]), height_m=h, trunk_radius_m=rad, stage=stage,
                        yaw=float(rng.uniform(0, 360)), variant=int(rng.integers(0, 3)))
            self.trees.append(tree)
            placed.append((x, z, rad))
        # a few hand-placed trees that frame the camp (closest points of interest)
        self._add_landmark(WoodSpeciesType.PAPER_BIRCH, 7.4, -8.6, 12.5, 0.15, TreeGrowthStage.MATURE)
        self._add_landmark(WoodSpeciesType.PAPER_BIRCH, 9.2, -7.4, 11.0, 0.13, TreeGrowthStage.MATURE)
        self._add_landmark(WoodSpeciesType.WHITE_PINE, -11.0, -6.5, 19.0, 0.28, TreeGrowthStage.MATURE)
        self._add_landmark(WoodSpeciesType.RED_CEDAR, 11.5, 4.5, 12.0, 0.2, TreeGrowthStage.MATURE)
        self._add_landmark(WoodSpeciesType.PAPER_BIRCH, -10.8, 7.2, 11.0, 0.13, TreeGrowthStage.STANDING_DEAD)
        self._add_landmark(WoodSpeciesType.WHITE_OAK, -12.5, -13.0, 14.0, 0.3, TreeGrowthStage.MATURE)
        self._add_landmark(WoodSpeciesType.WHITE_PINE, 13.5, -10.0, 9.0, 0.2, TreeGrowthStage.STANDING_DEAD)
        self.version += 1

    def _add_landmark(self, sp, x, z, h, r, stage):
        y = float(terrain.height(x, z))
        t = Tree(sp, np.array([x, y, z]), height_m=h, trunk_radius_m=r, stage=stage,
                 yaw=float(self.rng.uniform(0, 360)), variant=int(self.rng.integers(0, 3)))
        t.is_landmark = True
        self.trees.append(t)

    # ------------------------------------------------------------------ queries
    def nearest_tree(self, pos: np.ndarray, max_dist: float = 3.0, standing_only: bool = True) -> Optional[Tree]:
        best, bd = None, max_dist
        for t in self.trees:
            if standing_only and not t.is_standing:
                continue
            d = math.hypot(t.position[0] - pos[0], t.position[2] - pos[2]) - t.trunk_radius_m
            if d < bd:
                best, bd = t, d
        return best

    def add_campfire_ash(self, position: np.ndarray, ash_kg: float):
        """Deposits potassium/calcium ash from a campfire, boosting local fertility."""
        for z in self.ash_zones:
            if np.linalg.norm(z.position - position) < 1.0:
                z.ash_mass_kg += ash_kg
                z.fertility_boost = min(3.0, 1.0 + z.ash_mass_kg * 0.5)
                return
        self.ash_zones.append(AshDepositZone(np.array(position, dtype=float), ash_kg))

    def fertility_at(self, pos: np.ndarray) -> float:
        f = 1.0
        for z in self.ash_zones:
            if np.linalg.norm(z.position[[0, 2]] - pos[[0, 2]]) < 6.0:
                f = max(f, z.fertility_boost)
        return f

    # ------------------------------------------------------------------ lifecycle
    def step_ecosystem(self, days_passed: float, wind_vector: np.ndarray, season: str = "Summer") -> Dict[str, int]:
        """
        Seasonal step: seed dispersal (wind-carried), ash-fertilised sapling growth, maturation,
        senescence and fungal fruiting on dead wood.
        """
        rng = self.rng
        stats = {"new_saplings": 0, "matured": 0, "died": 0, "fungi_fruited": 0}
        wind_strength = float(np.linalg.norm(wind_vector))
        # 1. Seeds: cones and acorns drop mostly in late summer / autumn
        seed_season = 1.0 if season in ("Autumn", "Summer") else 0.25
        n_events = int(rng.poisson(max(0.0, 0.08 * days_passed * seed_season)))
        mature = [t for t in self.trees if t.stage == TreeGrowthStage.MATURE]
        for _ in range(min(n_events, 12)):
            if not mature or len(self.trees) >= 320:
                break
            parent = mature[int(rng.integers(len(mature)))]
            carry = wind_vector / max(1e-6, wind_strength) * float(rng.uniform(2.0, 6.0 + 2.0 * wind_strength)) if wind_strength > 0.1 else np.zeros(3)
            jitter = np.array([rng.normal(0, 3.0), 0.0, rng.normal(0, 3.0)])
            p = parent.position + carry + jitter
            x, z = float(p[0]), float(p[2])
            if math.hypot(x, z) < 8.0 or abs(x) > 75 or abs(z) > 75 or terrain.is_in_water(x, z):
                continue
            fert = self.fertility_at(np.array([x, 0.0, z]))
            if rng.random() > 0.45 * fert:
                continue
            sap = Tree(parent.species, np.array([x, float(terrain.height(x, z)), z]), height_m=0.6,
                       trunk_radius_m=0.02, stage=TreeGrowthStage.SAPLING, yaw=float(rng.uniform(0, 360)),
                       variant=int(rng.integers(0, 3)))
            self.trees.append(sap)
            stats["new_saplings"] += 1
        # 2. Growth (fertility from campfire ash speeds it up)
        growth_season = {"Spring": 1.3, "Summer": 1.0, "Autumn": 0.3, "Winter": 0.0}.get(season, 0.8)
        for t in self.trees:
            if t.stage == TreeGrowthStage.SAPLING:
                g = 0.012 * days_passed * growth_season * self.fertility_at(t.position)
                t.height_m += g
                t.trunk_radius_m += g * 0.012
                t.age_years += days_passed / 365.0
                if t.height_m >= 6.0:
                    t.stage = TreeGrowthStage.MATURE
                    stats["matured"] += 1
        # 3. Senescence and fungal fruiting
        for t in self.trees:
            if t.stage == TreeGrowthStage.MATURE and rng.random() < 0.00004 * days_passed:
                t.stage = TreeGrowthStage.STANDING_DEAD
                t.living_moisture_surface, t.living_moisture_core = 0.25, 0.40
                stats["died"] += 1
            if t.stage == TreeGrowthStage.STANDING_DEAD and t.fungi_count < 4 and season in ("Autumn", "Spring", "Summer"):
                if rng.random() < 0.02 * days_passed:
                    t.has_fungi = True
                    t.fungi_count += 1
                    stats["fungi_fruited"] += 1
        if any(stats.values()):
            self.version += 1
        return stats
