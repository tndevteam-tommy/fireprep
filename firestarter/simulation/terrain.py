"""
Firestarter - Procedural Terrain (single source of truth)

Every system that needs ground height (renderer, player movement, topo map,
fuel placement, tree placement, seasoning ground contact) samples the same
deterministic height field defined here.

World axes:  +X = East, -Z = North, +Y = Up.   Units: metres.

Landmarks (match the microclimate zones in economy/topomap.py):
  * Camp clearing at the origin (flattened, dry forest floor with the fire pit)
  * "Marshy Creek Bed"   - a meandering creek valley around z = -40
  * "Sun-Baked Ridge"    - a south-facing ridge around z = +50
  * "Hardwood Basin"     - a gentle bowl around x = -40, z = +10
  * Far mountains        - beyond the walkable area to give the horizon depth
"""

import math
from typing import Tuple, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

WALKABLE_HALF_EXTENT = 60.0      # player is clamped to +/- this many metres
TERRAIN_HALF_EXTENT = 260.0      # rendered terrain extends to the horizon
CLEARING_RADIUS = 10.0           # fully flat camp clearing
CLEARING_BLEND = 9.0             # blend distance back to natural terrain
CREEK_WATER_DEPTH = 0.45         # water surface sits this far above the creek bed


def _hash2(ix: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Deterministic integer hash -> [0, 1)."""
    h = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0x1000000)


def value_noise(x: ArrayLike, z: ArrayLike) -> np.ndarray:
    """Smooth 2D value noise in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x0 = np.floor(x)
    z0 = np.floor(z)
    fx = x - x0
    fz = z - z0
    ux = fx * fx * (3.0 - 2.0 * fx)
    uz = fz * fz * (3.0 - 2.0 * fz)
    a = _hash2(x0, z0)
    b = _hash2(x0 + 1, z0)
    c = _hash2(x0, z0 + 1)
    d = _hash2(x0 + 1, z0 + 1)
    v = a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz
    return v * 2.0 - 1.0


def fbm(x: ArrayLike, z: ArrayLike, octaves: int = 4) -> np.ndarray:
    total = np.zeros_like(np.asarray(x, dtype=np.float64))
    amp = 0.5
    freq = 1.0
    norm = 0.0
    for i in range(octaves):
        total = total + amp * value_noise(np.asarray(x) * freq + i * 17.3, np.asarray(z) * freq - i * 9.1)
        norm += amp
        amp *= 0.5
        freq *= 2.03
    return total / norm


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def creek_centerline_z(x: ArrayLike) -> np.ndarray:
    """The creek meanders east-west through the northern valley."""
    x = np.asarray(x, dtype=np.float64)
    return -40.0 + 6.0 * np.sin(x * 0.045) + 2.5 * np.sin(x * 0.11 + 1.3)


def _natural_height(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    # Rolling forest hills
    h = 3.2 * fbm(x * 0.018, z * 0.018, 4)
    h = h + 0.9 * fbm(x * 0.07 + 40.0, z * 0.07 - 12.0, 3)
    # Sun-baked ridge to the north-east (remember: -Z is north, ridge is at +z = south side)
    h = h + 11.0 * np.exp(-((z - 52.0) / 20.0) ** 2) * (0.75 + 0.25 * np.sin(x * 0.035 + 0.6))
    # Hardwood basin - a shallow bowl
    h = h - 2.2 * np.exp(-(((x + 40.0) / 22.0) ** 2 + ((z - 10.0) / 18.0) ** 2))
    # Creek valley with a carved channel
    d = np.abs(z - creek_centerline_z(x))
    h = h - 4.2 * np.exp(-(d / 13.0) ** 2)
    h = h - 1.1 * _smoothstep(3.2, 1.0, d)
    # Far mountains ring the horizon (outside the walkable area)
    r = np.sqrt(x * x + z * z)
    mount = _smoothstep(85.0, 210.0, r)
    h = h + mount * (28.0 + 22.0 * fbm(x * 0.012 + 7.0, z * 0.012 - 3.0, 4))
    return h


_ORIGIN_OFFSET = float(_natural_height(np.array([0.0]), np.array([0.0]))[0])


def height(x: ArrayLike, z: ArrayLike) -> Union[float, np.ndarray]:
    """Ground elevation (metres).  Vectorised; returns float for scalar input."""
    scalar = np.isscalar(x) and np.isscalar(z)
    xa = np.asarray(x, dtype=np.float64)
    za = np.asarray(z, dtype=np.float64)
    natural = _natural_height(xa, za) - _ORIGIN_OFFSET
    r = np.sqrt(xa * xa + za * za)
    blend = _smoothstep(CLEARING_RADIUS, CLEARING_RADIUS + CLEARING_BLEND, r)
    # A very gentle micro-relief inside the clearing so it never looks like a table top
    # (faded out around the fire pit so the rendered ground matches the physical pit floor)
    micro = 0.04 * value_noise(xa * 0.6, za * 0.6) * _smoothstep(1.6, 3.2, r)
    # The fire pit sits in a shallow hand-dug bowl (wide enough for the terrain mesh to resolve)
    pit = -0.06 * (1.0 - _smoothstep(0.0, 1.15, r)) ** 1.5
    h = (micro + pit) * (1.0 - blend) + natural * blend
    if scalar:
        return float(h)
    return h


def normal(x: float, z: float, eps: float = 0.35) -> np.ndarray:
    hl = height(x - eps, z)
    hr = height(x + eps, z)
    hd = height(x, z - eps)
    hu = height(x, z + eps)
    n = np.array([hl - hr, 2.0 * eps, hd - hu], dtype=np.float64)
    return n / np.linalg.norm(n)


def slope_degrees(x: float, z: float) -> float:
    n = normal(x, z)
    return math.degrees(math.acos(max(-1.0, min(1.0, n[1]))))


def creek_distance(x: float, z: float) -> float:
    return abs(z - float(creek_centerline_z(x)))


def creek_water_level(x: ArrayLike) -> Union[float, np.ndarray]:
    """Water surface elevation along the creek (just above the carved bed)."""
    xa = np.asarray(x, dtype=np.float64)
    bed = height(xa, creek_centerline_z(xa))
    lvl = np.asarray(bed) + CREEK_WATER_DEPTH
    if np.isscalar(x):
        return float(lvl)
    return lvl


def is_in_water(x: float, z: float) -> bool:
    return height(x, z) < creek_water_level(x) - 0.02


def ground_moisture(x: float, z: float) -> float:
    """Relative soil moisture 0..1 used for ground-contact seasoning."""
    d = creek_distance(x, z)
    wet = math.exp(-(d / 16.0) ** 2)
    ridge_dry = math.exp(-((z - 52.0) / 22.0) ** 2)
    return float(max(0.05, min(1.0, 0.45 + 0.5 * wet - 0.3 * ridge_dry)))


def height_grid(half_extent: float, resolution: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-half_extent, half_extent, resolution)
    zs = np.linspace(-half_extent, half_extent, resolution)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    return X, Z, np.asarray(height(X, Z))
