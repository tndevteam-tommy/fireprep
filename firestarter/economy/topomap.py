"""
Firestarter - Smartphone Topographical Map App
Implements DESIGN Section 5.1 App 2: a GPS topo map with 5 m contour lines (marching squares
over the shared terrain height field), micro-climate humidity zones and waypoints (hearth,
drying rack, fungal stands, felled timber, the player).
"""

import math
from typing import List, Dict, Tuple, Any

import numpy as np

from firestarter.simulation import terrain


class MapWaypoint:
    def __init__(self, name: str, category: str, position: np.ndarray, symbol: str):
        self.name = name
        self.category = category   # 'fire_pit', 'drying_rack', 'fungi_stand', 'player', 'log', 'woodpile'
        self.position = np.array(position, dtype=float)
        self.symbol = symbol


class TopographicalMap:
    def __init__(self, map_size_m: float = 200.0, contour_interval_m: float = 5.0):
        self.map_size_m = map_size_m
        self.contour_interval_m = contour_interval_m
        self.waypoints: List[MapWaypoint] = []
        self.zoom = 1.0
        self.microclimates = [
            {"name": "Marshy Creek Bed", "center": np.array([0.0, 0.0, -40.0]), "radius": 24.0, "humidity": 0.88,
             "desc": "Saturated soil, damp mossy timber", "color": (70, 130, 200)},
            {"name": "Sun-Baked Ridge", "center": np.array([20.0, 18.0, 50.0]), "radius": 30.0, "humidity": 0.22,
             "desc": "Arid, rapid drying, brittle tinder", "color": (220, 170, 70)},
            {"name": "Hardwood Basin", "center": np.array([-40.0, 5.0, 10.0]), "radius": 24.0, "humidity": 0.55,
             "desc": "Temperate mixed oak & birch canopy", "color": (120, 190, 110)},
        ]
        self._contour_cache: Dict[Tuple, List[Tuple[float, float, float, float, float]]] = {}

    def add_waypoint(self, name: str, category: str, position: np.ndarray, symbol: str = "o"):
        self.waypoints.append(MapWaypoint(name, category, np.array(position, dtype=float), symbol))

    def clear_waypoints(self):
        self.waypoints = []

    def get_elevation_at(self, x: float, z: float) -> float:
        """Terrain elevation (shared with the renderer and the player)."""
        return float(terrain.height(x, z))

    def get_microclimate_at(self, x: float, z: float) -> Tuple[str, float]:
        """Returns (zone_name, relative_humidity)."""
        p = np.array([x, 0.0, z])
        for zi in self.microclimates:
            c = zi["center"].copy()
            c[1] = 0.0
            if np.linalg.norm(p - c) < zi["radius"]:
                return zi["name"], zi["humidity"]
        return "Forest Floor", 0.50

    def generate_contour_slices(self, resolution: int = 24) -> Dict[str, Any]:
        half = self.map_size_m / 2.0
        xs = np.linspace(-half, half, resolution)
        zs = np.linspace(-half, half, resolution)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        elevations = np.asarray(terrain.height(X, Z))
        return {"xs": xs, "zs": zs, "elevations": elevations,
                "min_elev": float(np.min(elevations)), "max_elev": float(np.max(elevations)),
                "contour_interval": self.contour_interval_m}

    def contour_segments(self, cx: float, cz: float, half_extent: float, resolution: int = 72,
                         interval: float = None) -> List[Tuple[float, float, float, float, float]]:
        """Marching-squares iso-lines.  Returns (x1, z1, x2, z2, level) in world metres."""
        interval = interval or self.contour_interval_m
        key = (round(cx / 4.0), round(cz / 4.0), round(half_extent), resolution, interval)
        if key in self._contour_cache:
            return self._contour_cache[key]
        cx, cz = key[0] * 4.0, key[1] * 4.0
        xs = np.linspace(cx - half_extent, cx + half_extent, resolution)
        zs = np.linspace(cz - half_extent, cz + half_extent, resolution)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        H = np.asarray(terrain.height(X, Z))
        lo = math.floor(H.min() / interval) * interval
        hi = math.ceil(H.max() / interval) * interval
        segs = []
        for level in np.arange(lo, hi + 1e-6, interval):
            above = H > level
            for i in range(resolution - 1):
                for j in range(resolution - 1):
                    c = (int(above[i, j]) << 3) | (int(above[i + 1, j]) << 2) | (int(above[i + 1, j + 1]) << 1) | int(above[i, j + 1])
                    if c == 0 or c == 15:
                        continue
                    pts = []
                    corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                    for e in range(4):
                        a = corners[e]
                        b = corners[(e + 1) % 4]
                        ha, hb = H[a], H[b]
                        if (ha > level) != (hb > level):
                            t = (level - ha) / (hb - ha)
                            pts.append((xs[a[0]] + (xs[b[0]] - xs[a[0]]) * t, zs[a[1]] + (zs[b[1]] - zs[a[1]]) * t))
                    if len(pts) >= 2:
                        segs.append((pts[0][0], pts[0][1], pts[1][0], pts[1][1], float(level)))
                    if len(pts) == 4:
                        segs.append((pts[2][0], pts[2][1], pts[3][0], pts[3][1], float(level)))
        if len(self._contour_cache) > 16:
            self._contour_cache.clear()
        self._contour_cache[key] = segs
        return segs
