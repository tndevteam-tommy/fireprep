"""Setting a level's local geometry onto its terrain.

Level data is authored with every floor at y = 0. ``Placer`` lifts it onto the ground:

* anything on a **pad** (a building's floor, a levelled square) moves up by the pad's height,
  rigidly - the building stands exactly as drawn;
* a **wall in the open** (the town's curtain wall, a fence) is cut into short segments along its
  length, each reaching from just below the lowest ground under it to its height above the
  highest - it follows the slope in steps, never floating, never sinking;
* a **prop in the open** stands on the lowest ground under its footprint (``anchor="low"``, sunk a
  touch into the rest) or level with the highest (``"high"``: a bridge resting on both banks);
* a **point** (an NPC's spot, a light, a stair's waypoint) is lifted by the ground under it.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

from ..core.math3d import BoundingBox3D, Vector3D
from .terrain import Terrain

WALL_SEGMENT = 3.0           # longest piece a wall in the open is cut into
WALL_FOOTING = 0.4           # how far such a wall reaches below the lowest ground under it
SINK = 0.04                  # how far a prop in the open is sunk below its lowest ground


class Placer:
    def __init__(self, terrain: Terrain):
        self.terrain = terrain

    @property
    def flat(self) -> bool:
        return self.terrain.flat

    def ground(self, x: float, z: float) -> float:
        return self.terrain.height(x, z)

    def point(self, v: Vector3D) -> Vector3D:
        return Vector3D(v.x, v.y + self.ground(v.x, v.z), v.z)

    def _pad_under(self, box: BoundingBox3D):
        cx, cz = (box.min_point.x + box.max_point.x) / 2.0, (box.min_point.z + box.max_point.z) / 2.0
        return self.terrain.pad_at(cx, cz)

    def box(self, box: BoundingBox3D, anchor: str = "low") -> BoundingBox3D:
        """Lift ``box`` in place (see the module docstring); returns it."""
        if self.flat:
            return box
        pad = self._pad_under(box)
        if pad is not None:
            lift = pad.height
        else:
            lo, hi = self.terrain.range_over(box.min_point.x, box.min_point.z, box.max_point.x, box.max_point.z)
            lift = hi if anchor == "high" else lo - SINK
        box.min_point.y += lift
        box.max_point.y += lift
        return box

    def wall(self, wall: BoundingBox3D, rebuild: Callable[[BoundingBox3D, Vector3D, Vector3D], BoundingBox3D],
             segment: Optional[float] = None) -> List[BoundingBox3D]:
        """A wall on a pad moves up with it; one in the open becomes segments (``segment`` long at
        most, ``WALL_SEGMENT`` by default) that follow the ground."""
        if self.flat:
            return [wall]
        if self._pad_under(wall) is not None:
            lifted = self.box(wall)
            if getattr(wall, "reach_surface", False):
                return self._up_to_the_surface(lifted, rebuild)
            return [lifted]
        mn, mx = wall.min_point, wall.max_point
        rest_on = getattr(wall, "rest_on", None)
        if rest_on is not None:
            _, hi = self.terrain.range_over(*rest_on)
            return [rebuild(wall, Vector3D(mn.x, mn.y + hi, mn.z), Vector3D(mx.x, mx.y + hi, mx.z))]
        along_x = (mx.x - mn.x) >= (mx.z - mn.z)
        length = (mx.x - mn.x) if along_x else (mx.z - mn.z)
        count = max(1, int(math.ceil(length / (segment or WALL_SEGMENT))))
        pieces = []
        for i in range(count):
            a, b = i / count, (i + 1) / count
            if along_x:
                x1, x2, z1, z2 = mn.x + (mx.x - mn.x) * a, mn.x + (mx.x - mn.x) * b, mn.z, mx.z
            else:
                x1, x2, z1, z2 = mn.x, mx.x, mn.z + (mx.z - mn.z) * a, mn.z + (mx.z - mn.z) * b
            lo, hi = self.terrain.range_over(x1, z1, x2, z2, step=0.5)
            bottom = mn.y + lo - WALL_FOOTING if mn.y <= 1e-6 else mn.y + lo
            pieces.append(rebuild(wall, Vector3D(x1, bottom, z1), Vector3D(x2, mx.y + hi, z2)))
        return pieces

    SURFACE_PIECE = 0.5         # the rock framing a cave's mouth is cut into columns this wide ...
    SURFACE_PROUD = 0.05        # ... each standing this much over the highest ground on it

    def _up_to_the_surface(self, wall: BoundingBox3D, rebuild) -> List[BoundingBox3D]:
        """A wall that must meet the hillside over a cave's mouth: cut into columns (both ways),
        each standing up to the drawn ground over it - so its top follows the slope of the rock
        face, never jutting out of it by more than a sliver (the ground is not drawn over the
        mouth itself: this *is* the face there)."""
        mn, mx = wall.min_point, wall.max_point
        nx = max(1, int(math.ceil((mx.x - mn.x) / self.SURFACE_PIECE - 1e-9)))
        nz = max(1, int(math.ceil((mx.z - mn.z) / self.SURFACE_PIECE - 1e-9)))
        pieces = []
        for i in range(nx):
            x1, x2 = mn.x + (mx.x - mn.x) * i / nx, mn.x + (mx.x - mn.x) * (i + 1) / nx
            for k in range(nz):
                z1, z2 = mn.z + (mx.z - mn.z) * k / nz, mn.z + (mx.z - mn.z) * (k + 1) / nz
                samples = [self.terrain.surface(x, z) for x in (x1, x2) for z in (z1, z2)]
                top = max(mx.y, max(samples) + self.SURFACE_PROUD)
                pieces.append(rebuild(wall, Vector3D(x1, mn.y, z1), Vector3D(x2, top, z2)))
        return pieces

    def at_pad_or_ground(self, x: float, z: float) -> float:
        return self.ground(x, z)

    def optional_point(self, v: Optional[Vector3D]) -> Optional[Vector3D]:
        return None if v is None else self.point(v)
