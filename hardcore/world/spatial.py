"""Uniform-grid broadphase for static obstacles.

`ObstacleIndex` behaves like the plain obstacle list (iteration, len, indexing)
so every existing call site keeps working, and adds `query()` which returns only
the obstacles whose footprint touches a given XZ range.  Huge obstacles (the
ground slab) live in a global list instead of being stamped into every cell.
"""
import numpy as np


class ObstacleIndex:
    def __init__(self, obstacles, cell=8.0, max_cells=64):
        self.items = list(obstacles)
        self.cell = float(cell)
        self.grid = {}
        self.global_items = []
        for o in self.items:
            b = o.bounds
            ix0, iz0 = int(np.floor(b.min[0] / self.cell)), int(np.floor(b.min[2] / self.cell))
            ix1, iz1 = int(np.floor(b.max[0] / self.cell)), int(np.floor(b.max[2] / self.cell))
            if (ix1 - ix0 + 1) * (iz1 - iz0 + 1) > max_cells:
                self.global_items.append(o)
                continue
            for ix in range(ix0, ix1 + 1):
                for iz in range(iz0, iz1 + 1):
                    self.grid.setdefault((ix, iz), []).append(o)

    # ----------------------------------------------------------- list facade
    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    # ----------------------------------------------------------------- query
    def query(self, min_pt, max_pt):
        """Obstacles whose grid cells overlap the XZ range [min_pt, max_pt]."""
        ix0, iz0 = int(np.floor(min_pt[0] / self.cell)), int(np.floor(min_pt[2] / self.cell))
        ix1, iz1 = int(np.floor(max_pt[0] / self.cell)), int(np.floor(max_pt[2] / self.cell))
        seen = set()
        out = list(self.global_items)
        for ix in range(ix0, ix1 + 1):
            for iz in range(iz0, iz1 + 1):
                for o in self.grid.get((ix, iz), ()):
                    if id(o) not in seen:
                        seen.add(id(o))
                        out.append(o)
        return out

    def query_box(self, box, pad=0.0):
        return self.query(box.min - pad, box.max + pad)


def candidates(obstacles, min_pt, max_pt):
    """Broadphase helper: narrow `obstacles` to the XZ range when it is an ObstacleIndex."""
    if hasattr(obstacles, "query"):
        return obstacles.query(min_pt, max_pt)
    return obstacles
