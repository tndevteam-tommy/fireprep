"""What a bot can see: line of sight over the level, as fast as a bot needs it.

`HeightField` is a 2.5D picture of the level: for every 1 m cell the top of the
tallest thing standing in it.  A sight line is blocked where it passes below the
top of a cell it crosses.  Things that overhang (an awning, a bus-shelter roof, a
skybridge) count as solid down to the ground - a little pessimistic, never
optimistic: a bot never sees through something a player could not see through.

The city is boxes, so this is exact for buildings, walls, vehicles and crates,
which is what hides an emblem on a roof from someone down in the street - just
as it is hidden from a player (emblems glow, but walls hide the glow).
"""
import math

import numpy as np


class HeightField:
    CELL = 1.0
    STEP = 0.5                      # sample spacing along a sight line (m)

    def __init__(self, obstacles, extent=None):
        items = list(obstacles)
        ground = [o for o in items if o.bounds.size[0] > 100.0 and o.bounds.size[2] > 100.0]
        if extent is None:
            if ground:
                b = max(ground, key=lambda o: o.bounds.size[0] * o.bounds.size[2]).bounds
                lo, hi = b.min, b.max
            else:
                mins = np.min([o.bounds.min for o in items], axis=0)
                maxs = np.max([o.bounds.max for o in items], axis=0)
                lo, hi = mins, maxs
        else:
            lo, hi = extent
        self.x0 = float(lo[0])
        self.z0 = float(lo[2])
        self.nx = int(math.ceil((float(hi[0]) - self.x0) / self.CELL)) + 1
        self.nz = int(math.ceil((float(hi[2]) - self.z0) / self.CELL)) + 1
        self.h = np.zeros((self.nx, self.nz), dtype=np.float32)
        for o in items:
            b = o.bounds
            top = float(b.max[1])
            if top <= 0.0:
                continue
            if o in ground:
                continue
            i0, k0 = self.cell_of(float(b.min[0]), float(b.min[2]))
            i1, k1 = self.cell_of(float(b.max[0]) - 1e-4, float(b.max[2]) - 1e-4)
            i0, i1 = max(0, i0), min(self.nx - 1, i1)
            k0, k1 = max(0, k0), min(self.nz - 1, k1)
            if i1 < i0 or k1 < k0:
                continue
            view = self.h[i0:i1 + 1, k0:k1 + 1]
            np.maximum(view, top, out=view)

    # ------------------------------------------------------------------ cells
    def cell_of(self, x, z):
        return int(math.floor((x - self.x0) / self.CELL)), int(math.floor((z - self.z0) / self.CELL))

    def height_at(self, x, z):
        i, k = self.cell_of(x, z)
        if 0 <= i < self.nx and 0 <= k < self.nz:
            return float(self.h[i, k])
        return 0.0

    # ------------------------------------------------------------------ sight
    def visible(self, eye, point, clearance=0.05):
        """True when nothing stands between the eye and the point (both world positions)."""
        return bool(self.visible_many(eye, np.asarray(point, dtype=np.float32)[None, :], clearance)[0])

    def visible_many(self, eye, points, clearance=0.05):
        """Line of sight from one eye to many points at once (a bool per point)."""
        eye = np.asarray(eye, dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(pts) == 0:
            return np.zeros(0, dtype=bool)
        d = pts - eye[None, :]
        length = np.hypot(d[:, 0], d[:, 2])
        n = int(min(600, max(2, math.ceil(float(length.max()) / self.STEP))))
        # samples strictly between the eye and the point (the surface under the point is not in the way)
        t = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[1:-1]
        if len(t) == 0:
            return np.ones(len(pts), dtype=bool)
        # skip the last 0.6 m before the point and the first 0.3 m from the eye
        near = np.clip(0.3 / np.maximum(length, 1e-3), 0.0, 1.0)
        far = np.clip(1.0 - 0.6 / np.maximum(length, 1e-3), 0.0, 1.0)
        xs = eye[0] + d[:, 0:1] * t[None, :]
        zs = eye[2] + d[:, 2:3] * t[None, :]
        ys = eye[1] + d[:, 1:2] * t[None, :]
        ii = np.floor((xs - self.x0) / self.CELL).astype(np.int64)
        kk = np.floor((zs - self.z0) / self.CELL).astype(np.int64)
        inside = (ii >= 0) & (ii < self.nx) & (kk >= 0) & (kk < self.nz)
        hh = np.where(inside, self.h[np.clip(ii, 0, self.nx - 1), np.clip(kk, 0, self.nz - 1)], 0.0)
        active = (t[None, :] >= near[:, None]) & (t[None, :] <= far[:, None])
        blocked = active & (hh > ys + clearance)
        return ~np.any(blocked, axis=1)
