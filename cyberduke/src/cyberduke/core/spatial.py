"""Uniform-grid spatial hash over AABBs so collision queries only touch nearby boxes.

The level has ~40 colliders and every movement step tests all of them; the traversability
tests make hundreds of thousands of such steps. Bucketing by a coarse cell size keeps each
query to the handful of boxes that can actually matter.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from cyberduke.core.geometry import AABB


class SpatialHash:
    def __init__(self, boxes: Iterable[AABB], cell_size: float = 2.0) -> None:
        self.cell_size = cell_size
        self.boxes: list[AABB] = list(boxes)
        self._cells: dict[tuple[int, int], list[AABB]] = defaultdict(list)
        for box in self.boxes:
            for key in self._keys(box.min_x, box.max_x, box.min_z, box.max_z):
                self._cells[key].append(box)

    def _keys(self, min_x: float, max_x: float, min_z: float, max_z: float):
        c = self.cell_size
        for gx in range(math.floor(min_x / c), math.floor(max_x / c) + 1):
            for gz in range(math.floor(min_z / c), math.floor(max_z / c) + 1):
                yield (gx, gz)

    def near(self, x: float, z: float, radius: float) -> Sequence[AABB]:
        """Boxes whose footprint could overlap a circle of ``radius`` at (x, z)."""
        found: dict[int, AABB] = {}
        for key in self._keys(x - radius, x + radius, z - radius, z + radius):
            for box in self._cells.get(key, ()):
                found[id(box)] = box
        return list(found.values())

    def __len__(self) -> int:
        return len(self.boxes)
