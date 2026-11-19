"""
Firestarter - Scene asset construction: terrain meshes with material masks, the creek,
vegetation instance lists (trees, bushes, ferns, grass, rocks) and seasonal tints.
"""

import math
from typing import Dict, List, Tuple, Any

import numpy as np

from firestarter.simulation import terrain
from firestarter.rendering import mesh_generator as mg
from firestarter.thermodynamics.wood_properties import WoodSpeciesType
from firestarter.forestry.tree import TreeGrowthStage

NOMINAL_CONIFER_H = 15.0
NOMINAL_DECID_H = 14.0
NOMINAL_DEAD_H = 10.0


def terrain_mask(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Per-vertex material mask: r forest density, g camp clearing, b creek wetness, a ridge dryness."""
    r = np.sqrt(X * X + Z * Z)
    n = terrain.fbm(X * 0.05, Z * 0.05, 3) * 0.5 + 0.5
    forest = np.clip((r - 11.0) / 7.0, 0.0, 1.0) * (0.35 + 0.65 * n)
    clearing = 1.0 - np.clip((r - 6.0) / 6.0, 0.0, 1.0)
    d_creek = np.abs(Z - terrain.creek_centerline_z(X))
    wet = np.exp(-(d_creek / 7.0) ** 2)
    ridge = np.exp(-((Z - 52.0) / 18.0) ** 2)
    return np.stack([forest, clearing, wet, ridge], axis=-1).astype(np.float32)


def build_terrain() -> List[mg.MeshData]:
    inner = mg.create_terrain_mesh(size=128.0, resolution=257, elevation_fn=terrain.height, mask_fn=terrain_mask)
    outer = mg.create_terrain_mesh(size=520.0, resolution=131, elevation_fn=terrain.height, mask_fn=terrain_mask,
                                   hole=62.0, drop=0.03)
    return [inner, outer]


def build_creek() -> mg.MeshData:
    xs = np.linspace(-140.0, 140.0, 281)
    zc = terrain.creek_centerline_z(xs)
    lvl = np.asarray(terrain.creek_water_level(xs))
    width = 4.2 + 1.2 * np.sin(xs * 0.07)
    left = np.stack([xs, lvl, zc - width * 0.5], axis=1)
    right = np.stack([xs, lvl, zc + width * 0.5], axis=1)
    P = np.empty((len(xs) * 2, 3))
    P[0::2] = left
    P[1::2] = right
    N = np.tile((0.0, 1.0, 0.0), (len(P), 1))
    UV = np.zeros((len(P), 2))
    C = np.tile((1.0, 1.0, 1.0, 1.0), (len(P), 1))
    inds = []
    for i in range(len(xs) - 1):
        a = i * 2
        inds += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
    return mg.MeshData(mg._pack(P, N, UV, C), np.array(inds))


SEASON_FOLIAGE = {
    # species -> season -> tint multiplier applied to the base leaf colour
    "oak": {"Summer": (1.0, 1.0, 1.0), "Autumn": (2.7, 1.05, 0.6), "Winter": None, "Spring": (1.15, 1.25, 1.05)},
    "birch": {"Summer": (1.0, 1.0, 1.0), "Autumn": (2.6, 1.55, 0.45), "Winter": None, "Spring": (1.2, 1.3, 1.1)},
    "pine": {"Summer": (1.0, 1.0, 1.0), "Autumn": (1.0, 0.97, 0.95), "Winter": (0.85, 0.9, 0.95), "Spring": (1.05, 1.12, 1.0)},
    "cedar": {"Summer": (1.0, 1.0, 1.0), "Autumn": (1.05, 0.95, 0.9), "Winter": (0.8, 0.85, 0.9), "Spring": (1.05, 1.1, 1.0)},
}

SPECIES_KEY = {
    WoodSpeciesType.WHITE_PINE: "pine",
    WoodSpeciesType.RED_CEDAR: "cedar",
    WoodSpeciesType.WHITE_OAK: "oak",
    WoodSpeciesType.PAPER_BIRCH: "birch",
}


def build_tree_meshes() -> Dict[str, Any]:
    meshes = {}
    for v in range(3):
        meshes[("pine", v)] = mg.create_conifer("pine", seed=v, height=NOMINAL_CONIFER_H)
        meshes[("cedar", v)] = mg.create_conifer("cedar", seed=v + 10, height=NOMINAL_CONIFER_H)
        meshes[("oak", v)] = mg.create_deciduous("oak", seed=v, height=NOMINAL_DECID_H)
        meshes[("birch", v)] = mg.create_deciduous("birch", seed=v + 20, height=NOMINAL_DECID_H)
        meshes[("dead", v)] = (mg.create_dead_tree(seed=v, height=NOMINAL_DEAD_H), None)
    return meshes


def build_far_foliage_meshes() -> Dict[Tuple[str, int], Any]:
    """Lightweight card canopies for the distant tree line (LOD 1)."""
    out = {}
    if not mg.CARD_FOLIAGE:
        return out
    for v in range(3):
        out[("pine", v)] = mg.create_conifer_cards("pine", v, NOMINAL_CONIFER_H, lod=1)
        out[("cedar", v)] = mg.create_conifer_cards("cedar", v + 10, NOMINAL_CONIFER_H, lod=1)
        out[("oak", v)] = mg.create_broadleaf_cards("oak", v, NOMINAL_DECID_H, lod=1)
        out[("birch", v)] = mg.create_broadleaf_cards("birch", v + 20, NOMINAL_DECID_H, lod=1)
    return out


def tree_instances(trees, season: str) -> Dict[Tuple[str, str, int], np.ndarray]:
    """Instance arrays keyed by (species_key, part, variant). part in trunk / foliage / dead."""
    groups: Dict[Tuple[str, str, int], List[List[float]]] = {}
    for t in trees:
        if t.is_falling or (t.is_fallen and t.stage == TreeGrowthStage.STUMP):
            continue
        if t.is_fallen:
            continue
        key = SPECIES_KEY.get(t.species, "pine")
        rng = (t.id * 9301 + 49297) % 233280 / 233280.0
        yaw = math.radians(t.yaw)
        if t.stage == TreeGrowthStage.STANDING_DEAD:
            s = t.height_m / NOMINAL_DEAD_H
            sxz = max(0.35, t.trunk_radius_m / 0.24)
            tint = (0.95, 0.95, 0.95) if key != "birch" else (1.9, 1.85, 1.75)
            groups.setdefault(("dead", "dead", t.variant), []).append(
                [t.position[0], t.position[1], t.position[2], yaw, sxz, s, sxz, rng, *tint, 0.0])
            continue
        nominal = NOMINAL_CONIFER_H if key in ("pine", "cedar") else NOMINAL_DECID_H
        s = t.height_m / nominal
        r_nom = 0.25 if key in ("pine", "cedar") else (0.32 if key == "oak" else 0.17)
        sxz_trunk = max(0.25, min(2.0, t.trunk_radius_m / r_nom / max(0.3, s)))
        trunk_tint = (1.0 + (rng - 0.5) * 0.2,) * 3
        groups.setdefault((key, "trunk", t.variant), []).append(
            [t.position[0], t.position[1], t.position[2], yaw, s * sxz_trunk, s, s * sxz_trunk, rng, *trunk_tint, 0.0])
        tint = SEASON_FOLIAGE[key].get(season, (1.0, 1.0, 1.0))
        if tint is None:
            continue
        jitter = 0.88 + 0.24 * rng
        col = (tint[0] * jitter, tint[1] * (0.94 + 0.12 * rng), tint[2] * jitter)
        if key in ("oak", "birch") and season == "Autumn":
            # mixed autumn canopy: some crowns still green-gold, some deep red
            mix = (t.id * 7 % 10) / 10.0
            col = (col[0] * (0.75 + 0.5 * mix), col[1] * (1.15 - 0.45 * mix), col[2])
        groups.setdefault((key, "foliage", t.variant), []).append(
            [t.position[0], t.position[1], t.position[2], yaw, s, s, s, rng, *col, 0.0])
    return {k: np.array(v, dtype=np.float32) for k, v in groups.items()}


def fungus_instances(trees) -> np.ndarray:
    out = []
    for t in trees:
        if not (t.has_fungi and t.fungi_count > 0 and t.is_standing):
            continue
        for i in range(t.fungi_count):
            a = t.id * 1.7 + i * 2.1
            y = 0.6 + i * 0.45
            r = t.trunk_radius_m
            px = t.position[0] + math.cos(a) * r * 0.95
            pz = t.position[2] + math.sin(a) * r * 0.95
            s = 0.09 + 0.02 * (i % 2)
            out.append([px, t.position[1] + y, pz, -a, s, s, s, 0.3, 1.0, 1.0, 1.0, 0.0])
    return np.array(out, dtype=np.float32).reshape(-1, 12)


def scatter_vegetation(seed: int = 5, grass_density: float = 1.0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[str, List[List[float]]] = {"bush": [], "fern": [], "grass": [], "rock": [], "dead_log": []}

    def ok(x, z, clear_r):
        if math.hypot(x, z) < clear_r:
            return False
        if terrain.is_in_water(x, z):
            return False
        return terrain.slope_degrees(x, z) < 32.0

    for _ in range(320):
        x, z = rng.uniform(-70, 70, 2)
        if not ok(x, z, 9.0):
            continue
        s = rng.uniform(0.7, 1.5)
        out["bush"].append([x, terrain.height(x, z) - 0.05, z, rng.uniform(0, 6.28), s, s * rng.uniform(0.8, 1.2), s, rng.random(),
                            rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1), 1.0, 0.0])
    for _ in range(650):
        x, z = rng.uniform(-60, 60, 2)
        creek_bonus = math.exp(-(terrain.creek_distance(x, z) / 14.0) ** 2)
        if rng.random() > 0.35 + 0.65 * creek_bonus or not ok(x, z, 6.5):
            continue
        s = rng.uniform(0.6, 1.3)
        out["fern"].append([x, terrain.height(x, z), z, rng.uniform(0, 6.28), s, s, s, rng.random(), 1.0, 1.0, 1.0, 0.0])
    n_grass = int(9000 * grass_density)
    for _ in range(n_grass):
        # short and sparse inside the trampled campsite, lush around its edges and in sunny gaps
        r = abs(rng.normal(0, 16.0)) + 1.4
        a = rng.uniform(0, 2 * math.pi)
        x, z = r * math.cos(a), r * math.sin(a)
        if abs(x) > 60 or abs(z) > 60 or not ok(x, z, 1.3):
            continue
        if r < 9.0 and rng.random() < 0.65:
            continue
        if r < 3.6 and rng.random() < 0.8:
            continue     # trampled ground around the fire
        s = rng.uniform(0.7, 1.4) * (0.45 + 0.55 * min(1.0, max(0.0, (r - 3.0) / 8.0)))
        out["grass"].append([x, terrain.height(x, z) - 0.01, z, rng.uniform(0, 6.28), s, s * rng.uniform(0.8, 1.3), s, rng.random(),
                             rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1), rng.uniform(0.8, 1.0), 0.0])
    # grass in the forest gaps and thick along the creek banks
    for _ in range(int(6000 * grass_density)):
        x, z = rng.uniform(-60, 60, 2)
        creek_bonus = math.exp(-(terrain.creek_distance(x, z) / 9.0) ** 2)
        if rng.random() > 0.22 + 0.7 * creek_bonus or not ok(x, z, 12.0):
            continue
        s = rng.uniform(0.8, 1.6)
        out["grass"].append([x, terrain.height(x, z) - 0.01, z, rng.uniform(0, 6.28), s, s * rng.uniform(0.9, 1.4), s, rng.random(),
                             rng.uniform(0.85, 1.15), rng.uniform(0.9, 1.1), rng.uniform(0.8, 1.0), 0.0])
    for _ in range(260):
        x, z = rng.uniform(-75, 75, 2)
        if not ok(x, z, 7.0):
            continue
        s = rng.uniform(0.15, 0.7) * (1.8 if rng.random() < 0.1 else 1.0)
        out["rock"].append([x, terrain.height(x, z) - s * 0.25, z, rng.uniform(0, 6.28), s * rng.uniform(0.9, 1.4), s, s, rng.random(),
                            rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1), 0.0])
    # a few boulders around the camp for composition
    for (x, z, s) in ((-5.4, -2.6, 0.55), (4.8, 3.6, 0.42), (-3.2, 5.5, 0.7), (6.8, -4.2, 0.8), (-7.5, -8.0, 1.1)):
        out["rock"].append([x, terrain.height(x, z) - s * 0.25, z, x * 3.1, s * 1.3, s, s * 1.1, 0.5, 1.0, 1.0, 1.0, 0.0])
    for _ in range(40):
        x, z = rng.uniform(-65, 65, 2)
        if not ok(x, z, 12.0):
            continue
        s = rng.uniform(0.18, 0.32)
        out["dead_log"].append([x, terrain.height(x, z) + s * 0.5, z, rng.uniform(0, 6.28), s, rng.uniform(2.0, 4.5), s, rng.random(),
                                1.0, 1.0, 1.0, 0.0])
    return {k: np.array(v, dtype=np.float32).reshape(-1, 12) for k, v in out.items()}


def grass_season_colors(season: str, snow: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    # linear-space albedos (grass reflects ~10-20 % in the green, far less in red / blue)
    if season == "Summer":
        return (0.15, 0.23, 0.065), (0.36, 0.31, 0.14)
    if season == "Autumn":
        return (0.26, 0.23, 0.10), (0.40, 0.29, 0.14)
    if season == "Winter":
        return (0.28, 0.25, 0.16), (0.36, 0.31, 0.21)
    return (0.13, 0.26, 0.075), (0.24, 0.29, 0.11)


def far_forest_instances(seed: int = 11, count: int = 1400) -> Dict[Tuple[str, str, int], np.ndarray]:
    """Distant tree line on the hills beyond the walkable area (no collision, no shadows)."""
    rng = np.random.default_rng(seed)
    groups: Dict[Tuple[str, str, int], List[List[float]]] = {}
    placed = 0
    tries = 0
    while placed < count and tries < count * 6:
        tries += 1
        r = rng.uniform(78.0, 235.0)
        a = rng.uniform(0, 2 * math.pi)
        x, z = r * math.cos(a), r * math.sin(a)
        if terrain.is_in_water(x, z) or terrain.slope_degrees(x, z) > 34.0:
            continue
        dens = 0.55 + 0.45 * float(terrain.fbm(x * 0.02, z * 0.02, 3))
        if rng.random() > dens:
            continue
        y = float(terrain.height(x, z))
        roll = rng.random()
        sp = "pine" if roll < 0.5 else ("cedar" if roll < 0.75 else ("oak" if roll < 0.88 else "birch"))
        v = int(rng.integers(0, 3))
        h = rng.uniform(12.0, 22.0)
        s = h / (NOMINAL_CONIFER_H if sp in ("pine", "cedar") else NOMINAL_DECID_H)
        yaw = rng.uniform(0, 6.28)
        shade = rng.uniform(0.85, 1.1)
        groups.setdefault((sp, "trunk", v), []).append([x, y, z, yaw, s, s, s, rng.random(), 1.0, 1.0, 1.0, 0.0])
        groups.setdefault((sp, "foliage", v), []).append([x, y, z, yaw, s, s, s, rng.random(), shade, shade, shade, 0.0])
        placed += 1
    return {k: np.array(vv, dtype=np.float32) for k, vv in groups.items()}
