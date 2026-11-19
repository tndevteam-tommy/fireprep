"""Grid-based navigation for NPCs.

The level's static geometry (walls, props) is rasterised once into a boolean
occupancy grid on the X/Z plane, inflated by the agent radius so paths keep
clear of corners.  Doors are overlaid per query: a doorway is passable for an
agent that can open it (or when it already stands open).  Paths are found
with A* on 8-connected cells (no corner cutting) and then simplified with a
line-of-sight string-pull so NPCs walk straight where they can.
"""
from __future__ import annotations

import heapq
import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import NavigationSettings
from ..core.math3d import BoundingBox3D, Vector3D
from ..entities import Door

Cell = Tuple[int, int]
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


HEAD_HEIGHT = 1.9      # obstacles starting above this are walked under


class NavGrid:
    ROUTE_CACHE_SIZE = 1024
    def __init__(self, obstacles: Iterable[BoundingBox3D], settings: NavigationSettings = NavigationSettings(),
                 walls: Iterable[BoundingBox3D] = (), blocked_discs: Sequence[Tuple[float, float, float]] = (),
                 blocked_ellipses: Sequence[Tuple[float, float, float, float]] = (),
                 clear_boxes: Iterable[BoundingBox3D] = (), ground: Optional[Callable[[float, float], float]] = None,
                 blocked_where: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                 fences: Iterable[BoundingBox3D] = ()):
        """``obstacles`` block movement; ``walls`` (a subset) are additionally treated as
        impassable when snapping a start/goal that lies inside furniture to free space.
        ``blocked_discs`` (x, z, r) and ``blocked_ellipses`` (x, z, rx, rz) are water nobody plans
        a walk through; ``blocked_where(X, Z)`` marks any other ground nobody walks (a cliff, the
        open sea) given the cell centres; ``clear_boxes`` (bridges) are walkable whatever lies beneath,
        except for ``fences`` (a bridge's rails), blocked last."""
        self.settings = settings
        self.ground = ground or (lambda x, z: 0.0)
        self.cell = settings.cell_size
        self.min_x, self.min_z, self.max_x, self.max_z = settings.bounds
        self.cols = int(math.ceil((self.max_x - self.min_x) / self.cell))
        self.rows = int(math.ceil((self.max_z - self.min_z) / self.cell))
        self.static_blocked = np.zeros((self.cols, self.rows), dtype=bool)
        self.wall_mask = np.zeros((self.cols, self.rows), dtype=bool)
        for box in obstacles:
            if box.min_point.y - self._floor_under(box) < HEAD_HEIGHT:   # a lintel or a window piece up a wall is no obstacle
                self._rasterise(self.static_blocked, box, inflate=settings.agent_radius)
        for box in walls:
            if box.min_point.y - self._floor_under(box) < HEAD_HEIGHT:
                self._rasterise(self.wall_mask, box, inflate=settings.agent_radius)
        if blocked_discs or blocked_ellipses:
            xs = self.min_x + (np.arange(self.cols) + 0.5) * self.cell
            zs = self.min_z + (np.arange(self.rows) + 0.5) * self.cell
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            for x, z, r in blocked_discs:
                c1, r1 = self.to_cell(x - r, z - r)
                c2, r2 = self.to_cell(x + r, z + r)
                sub = (X[c1:c2 + 1, r1:r2 + 1] - x) ** 2 + (Z[c1:c2 + 1, r1:r2 + 1] - z) ** 2 <= r * r
                self.static_blocked[c1:c2 + 1, r1:r2 + 1] |= sub
            for x, z, rx, rz in blocked_ellipses:
                self.static_blocked |= ((X - x) / rx) ** 2 + ((Z - z) / rz) ** 2 <= 1.0
        if blocked_where is not None:
            xs = self.min_x + (np.arange(self.cols) + 0.5) * self.cell
            zs = self.min_z + (np.arange(self.rows) + 0.5) * self.cell
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            self.static_blocked |= np.asarray(blocked_where(X, Z), dtype=bool)
        for box in clear_boxes:
            x0, z0 = self.to_cell(box.min_point.x + 0.3, box.min_point.z + 0.3)
            x1, z1 = self.to_cell(box.max_point.x - 0.3, box.max_point.z - 0.3)
            self.static_blocked[x0:x1 + 1, z0:z1 + 1] = False
        for box in fences:
            self._rasterise(self.static_blocked, box, inflate=settings.agent_radius)
        self._static_rows: List[List[bool]] = self.static_blocked.tolist()      # the search reads plain lists
        self._door_cells = {}                                                  # door id -> its cells
        self._route_cache: dict = {}                                           # (start, goal, doors) -> string-pulled cells

    def _floor_under(self, box: BoundingBox3D) -> float:
        """Obstacles are lifted onto the terrain: what counts is how high above the ground under
        them they start (a lintel over a doorway is walked under, a wall on a hillside is not)."""
        return self.ground((box.min_point.x + box.max_point.x) / 2.0, (box.min_point.z + box.max_point.z) / 2.0)

    # ------------------------------------------------------------------ grid <-> world
    def to_cell(self, x: float, z: float) -> Cell:
        ix = int((x - self.min_x) / self.cell)
        iz = int((z - self.min_z) / self.cell)
        return (min(max(ix, 0), self.cols - 1), min(max(iz, 0), self.rows - 1))

    def to_world(self, cell: Cell, y: float = 0.0) -> Vector3D:
        return Vector3D(self.min_x + (cell[0] + 0.5) * self.cell, y, self.min_z + (cell[1] + 0.5) * self.cell)

    def _rasterise(self, grid: np.ndarray, box: BoundingBox3D, inflate: float = 0.0) -> None:
        x0, z0 = self.to_cell(box.min_point.x - inflate, box.min_point.z - inflate)
        x1, z1 = self.to_cell(box.max_point.x + inflate, box.max_point.z + inflate)
        grid[x0:x1 + 1, z0:z1 + 1] = True

    def door_cells(self, door: Door) -> frozenset:
        """The cells a door's doorway covers (cached per door)."""
        key = (door.door_id, door.hinge_position.x, door.hinge_position.z)
        cells = self._door_cells.get(key)
        if cells is None:
            box = door.get_bounding_box()
            x0, z0 = self.to_cell(box.min_point.x, box.min_point.z)
            x1, z1 = self.to_cell(box.max_point.x, box.max_point.z)
            cells = frozenset((i, j) for i in range(x0, x1 + 1) for j in range(z0, z1 + 1))
            self._door_cells[key] = cells
        return cells

    def closed_cells(self, doors: Sequence[Door] = (), can_open: Optional[Callable[[Door], bool]] = None) -> set:
        """Every cell of every door this agent cannot pass."""
        out = set()
        for door in doors:
            if not (door.is_open or (can_open is not None and can_open(door))):
                out |= self.door_cells(door)
        return out

    def occupancy(self, doors: Sequence[Door] = (), can_open: Optional[Callable[[Door], bool]] = None) -> np.ndarray:
        """Static obstacles plus every door this agent cannot pass (as an array)."""
        grid = self.static_blocked.copy()
        for i, j in self.closed_cells(doors, can_open):
            grid[i, j] = True
        return grid

    def is_blocked(self, x: float, z: float, grid: Optional[np.ndarray] = None) -> bool:
        """Is the cell under (x, z) occupied (on ``grid``, or the static geometry)?"""
        ix, iz = self.to_cell(x, z)
        return bool((self.static_blocked if grid is None else grid)[ix, iz])

    def nearest_free(self, cell: Cell, grid: np.ndarray, max_radius: int = 8) -> Optional[Cell]:
        """Closest free cell to ``cell``, reached by crossing furniture (a bed the NPC lies on)
        but never by going through a wall - so a bed against a wall snaps to the room, not outside."""
        blocked = grid.at if isinstance(grid, _Blocked) else (lambda c: bool(grid[c]))
        if not blocked(cell):
            return cell
        frontier: List[Tuple[float, int, Cell]] = [(0.0, 0, cell)]
        seen = {cell}
        counter = 1
        while frontier:
            cost, _, current = heapq.heappop(frontier)
            if not blocked(current):
                return current
            if cost > max_radius * 2:
                break
            for dx, dz in _NEIGHBOURS:
                nb = (current[0] + dx, current[1] + dz)
                if nb in seen or not (0 <= nb[0] < self.cols and 0 <= nb[1] < self.rows) or self.wall_mask[nb]:
                    continue
                seen.add(nb)
                heapq.heappush(frontier, (cost + (math.sqrt(2) if dx and dz else 1.0), counter, nb))
                counter += 1
        return None

    # ------------------------------------------------------------------ rooms
    def _rooms(self, doors: Sequence[Door]):
        """The walkable grid cut into *rooms* - regions connected without passing a doorway - and,
        per door, the rooms it joins. Computed once per set of doors: a search can then tell in
        a moment whether a goal is reachable at all through the doors an agent may use, instead
        of flooding the whole map to find out it is not (on a map this size that took seconds)."""
        key = tuple(sorted((d.door_id, d.hinge_position.x, d.hinge_position.z) for d in doors))
        cached = getattr(self, "_rooms_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        free = ~self.static_blocked
        door_cells = {}
        for d in doors:
            cells = self.door_cells(d)
            door_cells[d.door_id] = cells
            for c in cells:
                free[c] = False
        labels = _label_4(free)
        links = {}
        for door_id, cells in door_cells.items():
            touching = set()
            for cx, cz in cells:
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, nz = cx + dx, cz + dz
                    if 0 <= nx < self.cols and 0 <= nz < self.rows and labels[nx, nz] >= 0:
                        touching.add(int(labels[nx, nz]))
            links[door_id] = touching
        self._rooms_cache = (key, labels, links)
        return labels, links

    def reachable(self, start: Cell, goal: Cell, doors: Sequence[Door] = (),
                  can_open: Optional[Callable[[Door], bool]] = None) -> bool:
        """Can ``goal`` be reached from ``start`` through doors that stand open or that the agent
        may open? (Cells in a doorway belong to the rooms on both sides of it.)"""
        if not doors:
            return True
        labels, links = self._rooms(doors)
        passable = [d for d in doors if d.is_open or (can_open is not None and can_open(d))]

        def rooms_of(cell: Cell) -> set:
            label = int(labels[cell])
            if label >= 0:
                return {label}
            found = set()
            for d in doors:
                if cell in self.door_cells(d):
                    found |= links.get(d.door_id, set())
            return found

        start_rooms, goal_rooms = rooms_of(start), rooms_of(goal)
        if not start_rooms or not goal_rooms:
            return True                                 # (somewhere the rooms do not describe: let the search decide)
        seen = set(start_rooms)
        frontier = list(start_rooms)
        while frontier:
            room = frontier.pop()
            if room in goal_rooms:
                return True
            for d in passable:
                joined = links.get(d.door_id, ())
                if room in joined:
                    for other in joined:
                        if other not in seen:
                            seen.add(other)
                            frontier.append(other)
        return bool(seen & goal_rooms)

    # ------------------------------------------------------------------ search
    def find_path(self, start: Vector3D, goal: Vector3D, doors: Sequence[Door] = (),
                  can_open: Optional[Callable[[Door], bool]] = None) -> List[Vector3D]:
        """World-space waypoints from ``start`` to ``goal`` (excluding start, ending exactly
        at ``goal``). Empty if no route exists."""
        passable = tuple(bool(d.is_open or (can_open is not None and can_open(d))) for d in doors)
        blocked = _Blocked(self._static_rows, self.closed_cells(doors, can_open))
        s = self.nearest_free(self.to_cell(start.x, start.z), blocked)
        g = self.nearest_free(self.to_cell(goal.x, goal.z), blocked)
        if s is None or g is None:
            return []
        # the town's routines walk the same ways day after day: a route found once is kept
        key = (s, g, passable, tuple(d.door_id for d in doors))
        cells = self._route_cache.get(key)
        if cells is None:
            if not self.reachable(s, g, doors, can_open):
                return []                               # behind a door this agent cannot pass: no need to search
            cells = self._astar(s, g, blocked)
            if cells is None:
                return []
            cells = self._string_pull(cells, blocked)
            if len(self._route_cache) >= self.ROUTE_CACHE_SIZE:
                self._route_cache.pop(next(iter(self._route_cache)))      # (oldest first)
            self._route_cache[key] = cells
        waypoints = [self.to_world(c, goal.y) for c in cells[1:]]
        if waypoints:
            waypoints[-1] = Vector3D(goal.x, goal.y, goal.z)
        else:
            waypoints = [Vector3D(goal.x, goal.y, goal.z)]
        return waypoints

    def route_exists(self, start: Vector3D, goal: Vector3D, doors: Sequence[Door] = (),
                     can_open: Optional[Callable[[Door], bool]] = None) -> bool:
        """Whether ``find_path`` would find a way - by the rooms and the doors between them, without
        searching the cells (what a quick hour, with everybody put where they are going, needs)."""
        blocked = _Blocked(self._static_rows, self.closed_cells(doors, can_open))
        s = self.nearest_free(self.to_cell(start.x, start.z), blocked)
        g = self.nearest_free(self.to_cell(goal.x, goal.z), blocked)
        return s is not None and g is not None and self.reachable(s, g, doors, can_open)

    def _flat(self) -> bytearray:
        """The static grid as one flat byte per cell (index ``x * rows + z``), its outermost ring of
        cells blocked so the search never has to check the map's edges."""
        flat = getattr(self, "_flat_cache", None)
        if flat is None:
            grid = self.static_blocked.copy()
            grid[0, :] = grid[-1, :] = True
            grid[:, 0] = grid[:, -1] = True
            flat = bytearray(grid.astype(np.uint8).tobytes())      # C order: x major, z minor
            self._flat_cache = flat
        return flat

    def _astar(self, start: Cell, goal: Cell, grid: "_Blocked") -> Optional[List[Cell]]:
        """8-connected A* over the grid (static cells plus ``grid.closed``), never cutting a blocked
        corner. Cells are flat integers (``x * rows + z``) in the inner loop, the closed doorways
        are written into a copy of the flat grid, and the scores live in lists kept from one search
        to the next (only the cells a search touched are reset): on a map this size the search is
        what every long walk costs."""
        rows = self.rows
        flat = self._flat()
        s_idx, g_idx = start[0] * rows + start[1], goal[0] * rows + goal[1]
        gx, gz = goal
        if s_idx == g_idx:
            return [start]
        if grid.closed or (flat[g_idx] and not grid.rows[goal[0]][goal[1]]):
            flat = bytearray(flat)
            for cx_, cz_ in grid.closed:
                flat[cx_ * rows + cz_] = 1
            if not grid.rows[goal[0]][goal[1]] and (goal not in grid.closed):
                flat[g_idx] = 0                                # (the goal may sit on the blocked border ring)
        scores = getattr(self, "_scores", None)
        if scores is None or len(scores) != len(flat):
            scores = self._scores = [1e18] * len(flat)
            self._came = [-1] * len(flat)
        came_from = self._came
        touched = [s_idx]
        diag = math.sqrt(2)
        k = diag - 2.0
        straight = ((1, 0, rows), (-1, 0, -rows), (0, 1, 1), (0, -1, -1))
        diagonal = ((1, 1, rows + 1, rows, 1), (1, -1, rows - 1, rows, -1), (-1, 1, -rows + 1, -rows, 1),
                    (-1, -1, -rows - 1, -rows, -1))
        dx0, dz0 = abs(start[0] - gx), abs(start[1] - gz)
        open_heap = [((dx0 + dz0) + k * min(dx0, dz0), 0, s_idx, start[0], start[1], 0.0)]
        scores[s_idx] = 0.0
        came_from[s_idx] = -1
        counter = 1
        push, pop = heapq.heappush, heapq.heappop
        found = False
        try:
            while open_heap:
                _, _, cur, cx, cz, g = pop(open_heap)
                if g > scores[cur]:
                    continue    # a stale entry: the cell was queued again, cheaper, and expanded from that already
                if cur == g_idx:
                    found = True
                    break
                for dx, dz, d in straight:
                    nb = cur + d
                    if flat[nb]:
                        continue
                    t = g + 1.0
                    if t < scores[nb]:
                        if scores[nb] == 1e18:
                            touched.append(nb)
                        scores[nb] = t
                        came_from[nb] = cur
                        nx, nz = cx + dx, cz + dz
                        ax, az = abs(nx - gx), abs(nz - gz)
                        push(open_heap, (t + (ax + az) + k * (ax if ax < az else az), counter, nb, nx, nz, t))
                        counter += 1
                for dx, dz, d, side_x, side_z in diagonal:
                    nb = cur + d
                    if flat[nb] or flat[cur + side_x] or flat[cur + side_z]:
                        continue    # no cutting corners through blocked cells
                    t = g + diag
                    if t < scores[nb]:
                        if scores[nb] == 1e18:
                            touched.append(nb)
                        scores[nb] = t
                        came_from[nb] = cur
                        nx, nz = cx + dx, cz + dz
                        ax, az = abs(nx - gx), abs(nz - gz)
                        push(open_heap, (t + (ax + az) + k * (ax if ax < az else az), counter, nb, nx, nz, t))
                        counter += 1
            if not found:
                return None
            path = []
            cur = g_idx
            while cur != -1:
                path.append((cur // rows, cur % rows))
                cur = came_from[cur] if cur != s_idx else -1
            return path[::-1]
        finally:
            for idx in touched:                          # (ready for the next search)
                scores[idx] = 1e18
                came_from[idx] = -1

    def _line_clear(self, a: Cell, b: Cell, grid: "_Blocked") -> bool:
        """True if a straight walk between the centres of ``a`` and ``b`` touches only free cells,
        and passes no corner of a blocked cell closely (the squeeze between two blocked cells
        counts as blocked). Walks exactly the cells the line crosses (a grid traversal in integer
        arithmetic: from the centre of ``a`` the next x boundary is at t = (2i + 1) / 2|dx|, the
        next z one at (2j + 1) / 2|dz|) - where sampling the line four times a cell did many times
        the work for the same answer."""
        rows, closed = grid.rows, grid.closed
        x, z = a
        dx, dz = b[0] - x, b[1] - z
        sx = 1 if dx > 0 else -1
        sz = 1 if dz > 0 else -1
        adx, adz = abs(dx), abs(dz)
        if rows[x][z] or (closed and (x, z) in closed):
            return False
        i = j = 0
        near = adx * adz / (2.0 * max(adx, adz, 1))       # (crossings this close together: past a corner)
        while i < adx or j < adz:
            lhs = (2 * i + 1) * adz
            rhs = (2 * j + 1) * adx
            if adx and adz and i < adx and j < adz and abs(lhs - rhs) < near:
                if rows[x + sx][z] or rows[x][z + sz] or (closed and ((x + sx, z) in closed or (x, z + sz) in closed)):
                    return False
            if adz == 0 or (adx and lhs < rhs):
                x += sx
                i += 1
            elif adx == 0 or lhs > rhs:
                z += sz
                j += 1
            else:                                          # exactly through a corner
                x += sx
                z += sz
                i += 1
                j += 1
            if rows[x][z] or (closed and (x, z) in closed):
                return False
        return True

    def _string_pull(self, cells: List[Cell], grid: "_Blocked") -> List[Cell]:
        if len(cells) <= 2:
            return cells
        result = [cells[0]]
        anchor = 0
        i = 2
        while i < len(cells):
            if not self._line_clear(cells[anchor], cells[i], grid):
                result.append(cells[i - 1])
                anchor = i - 1
            i += 1
        result.append(cells[-1])
        return result


class _Blocked:
    """The static grid (nested lists) with a set of closed doorway cells on top: ``b[x][z]``."""
    __slots__ = ("rows", "closed")

    def __init__(self, rows: List[List[bool]], closed: set):
        self.rows, self.closed = rows, closed

    def at(self, cell: Cell) -> bool:
        return self.rows[cell[0]][cell[1]] or cell in self.closed


def _label_4(free: np.ndarray) -> np.ndarray:
    """Connected regions of ``free`` cells (4-neighbourhood - which is what 8-connected moves that
    never cut a corner can reach), as an array of labels (-1 for blocked cells). Runs of free cells
    down each column are joined with the overlapping runs of the column before (union-find), so
    the work is per run, not per cell."""
    cols, rows = free.shape
    labels = np.full(free.shape, -1, dtype=np.int32)
    parent: List[int] = []

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    prev_runs: List[Tuple[int, int, int]] = []
    for x in range(cols):
        column = free[x]
        padded = np.concatenate(([False], column, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs = []
        for a, b in zip(edges[::2], edges[1::2]):
            run_id = len(parent)
            parent.append(run_id)
            runs.append((int(a), int(b), run_id))
        i = j = 0
        while i < len(runs) and j < len(prev_runs):
            a1, b1, r1 = runs[i]
            a2, b2, r2 = prev_runs[j]
            if a1 < b2 and a2 < b1:                   # the runs share a row: the same region
                ra, rb = find(r1), find(r2)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
            if b1 < b2:
                i += 1
            else:
                j += 1
        for a, b, run_id in runs:
            labels[x, a:b] = run_id
        prev_runs = runs
    if parent:
        roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
        mask = labels >= 0
        labels[mask] = roots[labels[mask]]
    return labels
