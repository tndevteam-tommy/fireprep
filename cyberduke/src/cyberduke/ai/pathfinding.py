"""Grid A* pathfinding with line-of-sight string-pulling, for ground-walking agents.

The search runs on a regular grid inflated by the agent's radius, bounded by the level's
``nav_bounds``. A search budget proportional to the grid size (not a fixed constant) lets a
path cross the whole level, and per-search memoisation of blocked cells keeps that cheap.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from cyberduke.config import NPC
from cyberduke.core.geometry import AABB, any_cylinder_overlap, segment_clear_2d
from cyberduke.core.spatial import SpatialHash

Bounds = tuple[float, float, float, float]  # (min_x, max_x, min_z, max_z)
Point = tuple[float, float]
Cell = tuple[int, int]

_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, 1.414),
    (1, -1, 1.414),
    (-1, 1, 1.414),
    (-1, -1, 1.414),
)


@dataclass(frozen=True)
class Path:
    """Result of a search. ``waypoints`` always ends at the point the agent should walk to:
    the requested goal, or - when that goal sits inside an obstacle - the nearest free
    cell (``adjusted``). ``found`` is False when no route exists within the search budget,
    in which case the waypoints degrade to a straight line toward the goal."""

    waypoints: list[Point] = field(default_factory=list)
    found: bool = True
    adjusted: bool = False
    expanded: int = 0  # A* nodes expanded (0 for a direct line)

    @property
    def destination(self) -> Point:
        return self.waypoints[-1]


class Pathfinder:
    """2D A* pathfinder with line-of-sight path smoothing (string-pulling)."""

    GOAL_SNAP_RADIUS_CELLS = 3  # how far to look first for a free cell around a blocked goal
    GOAL_SNAP_MAX_RADIUS_CELLS = 12  # ... and how far the search may widen (4.8 m)

    def __init__(
        self,
        grid_size: float = 0.4,
        agent_radius: float = NPC.collision_radius,
        agent_height: float = NPC.body_height,
        max_expansions: int | None = None,
    ):
        self.grid_size = grid_size
        self.agent_radius = agent_radius
        self.agent_height = agent_height
        self.max_expansions = max_expansions  # None: derived from the bounds (one full grid)

    # ------------------------------------------------------------------ grid helpers
    def _to_grid(self, x: float, z: float) -> Cell:
        return int(round(x / self.grid_size)), int(round(z / self.grid_size))

    def _to_world(self, gx: int, gz: int) -> Point:
        return gx * self.grid_size, gz * self.grid_size

    def search_budget(self, bounds: Bounds) -> int:
        """Node-expansion budget: every cell of the bounded grid once (plus a little slack)."""
        if self.max_expansions is not None:
            return self.max_expansions
        cells_x = int((bounds[1] - bounds[0]) / self.grid_size) + 1
        cells_z = int((bounds[3] - bounds[2]) / self.grid_size) + 1
        return cells_x * cells_z + 64

    # ------------------------------------------------------------------ collision queries
    def is_blocked(self, x: float, z: float, obstacles: list[AABB], agent_y: float = 0.0) -> bool:
        return any_cylinder_overlap(
            obstacles, x, z, self.agent_radius, agent_y, agent_y + self.agent_height
        )

    def line_clear(
        self,
        x1: float,
        z1: float,
        x2: float,
        z2: float,
        obstacles: list[AABB],
        agent_y: float = 0.0,
    ) -> bool:
        feet, top = agent_y, agent_y + self.agent_height
        relevant = (o for o in obstacles if o.overlaps_vertical_span(feet, top))
        return segment_clear_2d(x1, z1, x2, z2, relevant, radius=self.agent_radius)

    # ------------------------------------------------------------------ search
    def find_path(
        self,
        start_x: float,
        start_z: float,
        goal_x: float,
        goal_z: float,
        obstacles: list[AABB],
        bounds: Bounds,
        agent_y: float = 0.0,
    ) -> Path:
        """Finds a collision-free smoothed path from (start_x, start_z) to (goal_x, goal_z)."""
        if not obstacles or self.line_clear(start_x, start_z, goal_x, goal_z, obstacles, agent_y):
            return Path([(goal_x, goal_z)])

        # Only obstacles at the agent's height matter for this whole search, and a cell
        # only needs to be tested against the boxes near it (the spatial hash keeps big
        # levels - a hundred-plus colliders - from turning every cell test into a scan).
        feet, top = agent_y, agent_y + self.agent_height
        relevant = [o for o in obstacles if o.overlaps_vertical_span(feet, top)]
        index = SpatialHash(relevant, cell_size=max(2.0, self.grid_size * 5))
        reach = self.agent_radius + self.grid_size
        blocked_cache: dict[Cell, bool] = {}

        def blocked(cell: Cell) -> bool:
            hit = blocked_cache.get(cell)
            if hit is None:
                wx, wz = self._to_world(*cell)
                near = index.near(wx, wz, reach)
                hit = any_cylinder_overlap(near, wx, wz, self.agent_radius, feet, top)
                blocked_cache[cell] = hit
            return hit

        start = self._to_grid(start_x, start_z)
        goal_cell = self._to_grid(goal_x, goal_z)
        min_gx, min_gz = self._to_grid(bounds[0], bounds[2])
        max_gx, max_gz = self._to_grid(bounds[1], bounds[3])

        # The search targets a free cell near the goal. The route still ends at the exact
        # goal point when that point itself is free (a cell centre can be blocked while the
        # goal, a few centimetres away, is not); otherwise it ends at the nearest free cell.
        target = goal_cell
        if blocked(goal_cell):
            target = self._nearest_free_cell(goal_cell, (goal_x, goal_z), blocked)
        adjusted = any_cylinder_overlap(relevant, goal_x, goal_z, self.agent_radius, feet, top)
        destination: Point = self._to_world(*target) if adjusted else (goal_x, goal_z)

        open_set: list[tuple[float, Cell]] = [(0.0, start)]
        came_from: dict[Cell, Cell] = {}
        g_score: dict[Cell, float] = {start: 0.0}
        budget = self.search_budget(bounds)
        expanded = 0
        found = False

        while open_set and expanded < budget:
            expanded += 1
            _, current = heapq.heappop(open_set)
            if current == target:
                found = True
                break
            for dx, dz, cost in _NEIGHBOURS:
                nx, nz = current[0] + dx, current[1] + dz
                if not (min_gx <= nx <= max_gx and min_gz <= nz <= max_gz):
                    continue
                neighbour = (nx, nz)
                if blocked(neighbour):
                    continue
                # No corner cutting: a diagonal step is only allowed when both orthogonal
                # neighbours are free, which (for boxes wider than a cell) guarantees the
                # segment between the two cell centres is clear as well.
                if (
                    dx
                    and dz
                    and (
                        blocked((current[0] + dx, current[1]))
                        or blocked((current[0], current[1] + dz))
                    )
                ):
                    continue
                tentative = g_score[current] + cost * self.grid_size
                if tentative < g_score.get(neighbour, math.inf):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    h = math.hypot(nx - target[0], nz - target[1]) * self.grid_size
                    heapq.heappush(open_set, (tentative + h, neighbour))

        if not found:
            return Path([(goal_x, goal_z)], found=False, adjusted=adjusted, expanded=expanded)

        raw: list[Point] = []
        cell = target
        while cell in came_from:
            raw.append(self._to_world(*cell))
            cell = came_from[cell]
        raw.reverse()
        if not raw or raw[-1] != destination:
            raw.append(destination)

        return Path(
            self._string_pull(start_x, start_z, raw, obstacles, agent_y),
            found=True,
            adjusted=adjusted,
            expanded=expanded,
        )

    def _nearest_free_cell(self, goal_cell: Cell, goal: Point, blocked) -> Cell:
        """The free cell nearest a blocked goal. Searches the usual snap radius first and,
        if the goal is buried deeper than that (a noise in the middle of a staircase),
        keeps widening ring by ring up to ``GOAL_SNAP_MAX_RADIUS_CELLS`` - otherwise the
        search would target a blocked cell, never find a route, and leave the agent
        pressing against the obstacle forever."""
        best, best_d = goal_cell, math.inf
        r = self.GOAL_SNAP_RADIUS_CELLS
        searched = -1
        while r <= self.GOAL_SNAP_MAX_RADIUS_CELLS:
            for dx in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dz)) <= searched:
                        continue  # inner rings were already checked
                    cell = (goal_cell[0] + dx, goal_cell[1] + dz)
                    if blocked(cell):
                        continue
                    wx, wz = self._to_world(*cell)
                    d = math.hypot(wx - goal[0], wz - goal[1])
                    if d < best_d:
                        best, best_d = cell, d
            # Unsearched cells lie at least r + 1 cells from the goal cell, i.e. at least
            # r + 0.5 cells from the goal point: stop once nothing further out can win.
            if best_d <= (r + 0.5) * self.grid_size:
                break
            searched = r
            r += 1
        return best

    def _string_pull(
        self, x: float, z: float, raw: list[Point], obstacles: list[AABB], agent_y: float
    ) -> list[Point]:
        """Greedy line-of-sight smoothing: from each point, skip to the furthest visible one."""
        smoothed: list[Point] = []
        i = 0
        while i < len(raw):
            furthest = i
            for j in range(len(raw) - 1, i, -1):
                if self.line_clear(x, z, raw[j][0], raw[j][1], obstacles, agent_y):
                    furthest = j
                    break
            smoothed.append(raw[furthest])
            x, z = raw[furthest]
            i = furthest + 1
        return smoothed
