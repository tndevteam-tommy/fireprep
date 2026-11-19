"""Ground navigation for the bot: a coarse occupancy grid of what it cannot simply run over, and A*.

Cells are blocked by anything taller than a running step-up with a footprint that
matters (buildings, walls, vehicles, containers, decks, shelters) - not by curbs or
litter.  A path runs through the streets to the foot of the target; climbing is
the bot's business (game/bot.py).  Paths are smoothed by line of sight so the bot
runs straight lines between corners.
"""
import heapq
import math

import numpy as np

import config


class NavGrid:
    CELL = 1.5
    BLOCK_HEIGHT = 1.05          # taller than a running step-up (AUTO_STEP_MAX_HEIGHT)
    MARGIN = 0.35                # keep the body clear of walls
    HEURISTIC_WEIGHT = 1.4       # weighted A*: routes within a few % of the shortest, found many times faster

    def __init__(self, obstacles, extent=None):
        obstacles = list(obstacles)
        if extent is None:
            big = [o.bounds for o in obstacles if o.bounds.size[0] > 200]
            if big:
                lo, hi = big[0].min, big[0].max
            else:
                lo = np.min([o.bounds.min for o in obstacles], axis=0) - 10.0
                hi = np.max([o.bounds.max for o in obstacles], axis=0) + 10.0
            extent = (float(lo[0]), float(lo[2]), float(hi[0]), float(hi[2]))
        self.x0, self.z0, x1, z1 = extent
        self.nx = int(math.ceil((x1 - self.x0) / self.CELL))
        self.nz = int(math.ceil((z1 - self.z0) / self.CELL))
        self.blocked = np.zeros((self.nx, self.nz), dtype=bool)
        for o in obstacles:
            b = o.bounds
            if b.size[0] > 200 or float(b.max[1]) < self.BLOCK_HEIGHT or float(b.min[1]) > 2.0:
                continue                           # the ground, low things, things overhead
            i0, k0 = self.cell_of(b.min[0] - self.MARGIN, b.min[2] - self.MARGIN)
            i1, k1 = self.cell_of(b.max[0] + self.MARGIN, b.max[2] + self.MARGIN)
            self.blocked[max(0, i0):min(self.nx, i1 + 1), max(0, k0):min(self.nz, k1 + 1)] = True
        self._refresh()

    def _refresh(self):
        """Flat free-cell lookup (a bytearray indexed i * nz + k): far cheaper than numpy indexing in
        the search's inner loop.  Call after changing `blocked`."""
        self._free = bytearray((~self.blocked).astype(np.uint8).tobytes())

    # ------------------------------------------------------------------ cells
    def cell_of(self, x, z):
        return int(math.floor((x - self.x0) / self.CELL)), int(math.floor((z - self.z0) / self.CELL))

    def center(self, i, k):
        return (self.x0 + (i + 0.5) * self.CELL, self.z0 + (k + 0.5) * self.CELL)

    def inside(self, i, k):
        return 0 <= i < self.nx and 0 <= k < self.nz

    def free(self, i, k):
        return 0 <= i < self.nx and 0 <= k < self.nz and self._free[i * self.nz + k] == 1

    def nearest_free(self, x, z, radius=12):
        ci, ck = self.cell_of(x, z)
        if self.free(ci, ck):
            return ci, ck
        best, best_d = None, None
        for r in range(1, radius + 1):
            for i in range(ci - r, ci + r + 1):
                for k in (ck - r, ck + r) if abs(i - ci) != r else range(ck - r, ck + r + 1):
                    if self.free(i, k):
                        d = (i - ci) ** 2 + (k - ck) ** 2
                        if best is None or d < best_d:
                            best, best_d = (i, k), d
            if best is not None:
                return best
        return None

    def cells_around_box(self, bmin, bmax, pad=1.2):
        """Free cells along the outside of a footprint (where you stand to climb it)."""
        i0, k0 = self.cell_of(bmin[0] - pad, bmin[2] - pad)
        i1, k1 = self.cell_of(bmax[0] + pad, bmax[2] + pad)
        out = set()
        for i in range(i0, i1 + 1):
            for k in range(k0, k1 + 1):
                if self.free(i, k):
                    out.add((i, k))
        return out

    # ------------------------------------------------------------------ search
    def line_clear(self, a, b):
        """Grid line of sight between two cells (sampled at quarter cells)."""
        (i0, k0), (i1, k1) = a, b
        n = int(max(abs(i1 - i0), abs(k1 - k0)) * 4) + 1
        nx, nz, free = self.nx, self.nz, self._free
        fi, fk = i0 + 0.5, k0 + 0.5
        di, dk = (i1 - i0) / n, (k1 - k0) / n
        for s in range(n + 1):
            i = int(math.floor(fi + di * s))
            k = int(math.floor(fk + dk * s))
            if not (0 <= i < nx and 0 <= k < nz) or free[i * nz + k] != 1:
                return False
        return True

    def path(self, start_xz, goal_cells, max_nodes=60000):
        """A* from a world point to the nearest of `goal_cells`.  Returns [(x, z), ...] (smoothed,
        the start excluded) or None."""
        start = self.nearest_free(*start_xz)
        goals = {g for g in goal_cells if self.free(*g)}
        if start is None or not goals:
            return None
        if start in goals:
            return [self.center(*start)]
        # in plain sight: no search needed
        nearest = min(goals, key=lambda g: (g[0] - start[0]) ** 2 + (g[1] - start[1]) ** 2)
        if self.line_clear(start, nearest):
            return [self.center(*nearest)]
        # admissible heuristic: distance to the goals' centroid less their spread
        cx = sum(g[0] for g in goals) / len(goals)
        ck = sum(g[1] for g in goals) / len(goals)
        spread = max(math.hypot(g[0] - cx, g[1] - ck) for g in goals)
        hypot = math.hypot
        nx, nz, free = self.nx, self.nz, self._free

        weight = self.HEURISTIC_WEIGHT

        def h(i, k):
            d = hypot(i - cx, k - ck) - spread
            return d * weight if d > 0.0 else 0.0

        start_key = start[0] * nz + start[1]
        goal_keys = {g[0] * nz + g[1] for g in goals}
        open_heap = [(h(*start), 0.0, start_key)]
        came = {start_key: -1}
        cost = {start_key: 0.0}
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
        expanded = 0
        end = None
        pop, push = heapq.heappop, heapq.heappush
        while open_heap:
            _, g, key = pop(open_heap)
            if g > cost.get(key, 1e18):
                continue
            if key in goal_keys:
                end = key
                break
            expanded += 1
            if expanded > max_nodes:
                return None
            ci, ck_ = divmod(key, nz)
            for di, dk, w in steps:
                ni, nk = ci + di, ck_ + dk
                if not (0 <= ni < nx and 0 <= nk < nz) or free[ni * nz + nk] != 1:
                    continue
                if di and dk and (free[(ci + di) * nz + ck_] != 1 or free[ci * nz + nk] != 1):
                    continue                       # no cutting blocked corners
                nkey = ni * nz + nk
                ng = g + w
                if ng < cost.get(nkey, 1e18):
                    cost[nkey] = ng
                    came[nkey] = key
                    push(open_heap, (ng + h(ni, nk), ng, nkey))
        if end is None:
            return None
        cells = []
        key = end
        while key != -1:
            cells.append(divmod(key, nz))
            key = came[key]
        cells.reverse()
        # string-pull: keep a corner only where the straight line would cut a blocked cell
        smooth = [cells[0]]
        for idx in range(1, len(cells)):
            if not self.line_clear(smooth[-1], cells[idx]):
                smooth.append(cells[idx - 1])
        smooth.append(cells[-1])
        out = []
        for c in smooth[1:]:
            p = self.center(*c)
            if not out or out[-1] != p:
                out.append(p)
        return out
