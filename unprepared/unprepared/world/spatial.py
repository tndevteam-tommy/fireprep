"""A coarse grid over the level's static boxes, so the questions asked every frame - what might
the player bump into, what might block this line of sight - look at the few boxes nearby
instead of every wall, tree and barrel in the realm.

:class:`BoxIndex` buckets boxes by ``cell``-sized squares of the ground plan; a box lands in
every cell its footprint touches. :class:`SightBlockers` is what ``World.sight_blockers``
hands out: iterable like the plain list it replaced (everything, in the same order), with
``along(a, b)`` giving just the candidates for one ray.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from ..core.math3d import BoundingBox3D, Vector3D

Cell = Tuple[int, int]


class BoxIndex:
    def __init__(self, boxes: Iterable[BoundingBox3D], cell: float = 8.0):
        self.cell = cell
        self.boxes: List[BoundingBox3D] = list(boxes)
        self._cells: Dict[Cell, List[int]] = {}
        for i, box in enumerate(self.boxes):
            for key in self._keys(box.min_point.x, box.min_point.z, box.max_point.x, box.max_point.z):
                self._cells.setdefault(key, []).append(i)

    def _keys(self, x1: float, z1: float, x2: float, z2: float) -> Iterator[Cell]:
        c = self.cell
        for cx in range(int(math.floor(x1 / c)), int(math.floor(x2 / c)) + 1):
            for cz in range(int(math.floor(z1 / c)), int(math.floor(z2 / c)) + 1):
                yield cx, cz

    def in_rect(self, x1: float, z1: float, x2: float, z2: float) -> List[BoundingBox3D]:
        """Every box whose footprint may overlap the rectangle (in the order they were given)."""
        found = set()
        for key in self._keys(x1, z1, x2, z2):
            found.update(self._cells.get(key, ()))
        return [self.boxes[i] for i in sorted(found)]

    def near(self, x: float, z: float, radius: float) -> List[BoundingBox3D]:
        return self.in_rect(x - radius, z - radius, x + radius, z + radius)

    def along(self, a: Vector3D, b: Vector3D, pad: float = 0.25) -> List[BoundingBox3D]:
        """Candidates for blocking the segment a-b (those in the cells its bounding box covers)."""
        return self.in_rect(min(a.x, b.x) - pad, min(a.z, b.z) - pad, max(a.x, b.x) + pad, max(a.z, b.z) + pad)

    def __len__(self) -> int:
        return len(self.boxes)

    def __iter__(self) -> Iterator[BoundingBox3D]:
        return iter(self.boxes)


def _overlaps_xz(box: BoundingBox3D, x1: float, z1: float, x2: float, z2: float) -> bool:
    return box.min_point.x <= x2 and box.max_point.x >= x1 and box.min_point.z <= z2 and box.max_point.z >= z1


class SightBlockers(Sequence):
    """The static index plus this moment's closed door leaves: a sequence of every blocker, and
    ``along(a, b)`` for the ones that can matter to one line of sight."""

    def __init__(self, index: BoxIndex, dynamic: Sequence[BoundingBox3D] = ()):
        self.index = index
        self.dynamic = list(dynamic)

    def along(self, a: Vector3D, b: Vector3D) -> List[BoundingBox3D]:
        x1, z1, x2, z2 = min(a.x, b.x) - 0.25, min(a.z, b.z) - 0.25, max(a.x, b.x) + 0.25, max(a.z, b.z) + 0.25
        return self.index.in_rect(x1, z1, x2, z2) + [d for d in self.dynamic if _overlaps_xz(d, x1, z1, x2, z2)]

    def __len__(self) -> int:
        return len(self.index) + len(self.dynamic)

    def __getitem__(self, i):
        n = len(self.index)
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        return self.index.boxes[i] if i < n else self.dynamic[i - n]

    def __iter__(self) -> Iterator[BoundingBox3D]:
        yield from self.index.boxes
        yield from self.dynamic


class PointIndex:
    """Light fixtures bucketed by position: ``lights(p)`` - is ``p`` within reach of any of them."""

    def __init__(self, fixtures: Sequence, cell: float = 8.0):
        self.cell = cell
        self.fixtures = list(fixtures)
        self.reach = max((f.range for f in self.fixtures), default=0.0)
        self._cells: Dict[Cell, List] = {}
        for f in self.fixtures:
            x, _, z = f.position
            self._cells.setdefault((int(math.floor(x / cell)), int(math.floor(z / cell))), []).append(f)

    def near(self, position: Vector3D, radius: float) -> List:
        c = self.cell
        out = []
        for cx in range(int(math.floor((position.x - radius) / c)), int(math.floor((position.x + radius) / c)) + 1):
            for cz in range(int(math.floor((position.z - radius) / c)), int(math.floor((position.z + radius) / c)) + 1):
                out.extend(self._cells.get((cx, cz), ()))
        return out

    def lights(self, position: Vector3D) -> bool:
        return any(Vector3D(*f.position).distance_to(position) <= f.range for f in self.near(position, self.reach))
