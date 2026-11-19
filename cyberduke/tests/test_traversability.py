"""Can everything that fits actually get everywhere it fits, and can it always get back?

These tests drive the *real* movement code (PlayerController sliding + elevation, and the
NPC pathfinder) over the whole level:

* Player reachability: flood-fill from the spawn using real controller steps. Every cell
  where a player-sized body fits (including the catwalk level) must be reached, unless the
  level declares it a ``NoGoZone``. This is what would have caught the unreachable catwalk.
* Traps: every reachable position must be able to get back to the spawn. A position you
  can enter but not leave is a wedge - this is what would have caught the ramp/rail gap.
* Rail sweeps: walking the full length of every handrail while pushing into it must never
  stall, on both the stair and deck levels.
* NPC reachability: a guard-sized agent must be able to path to every cell it fits in,
  except declared no-go zones (e.g. the stairs, which NPCs cannot climb yet).
"""

from __future__ import annotations

import math
from collections import deque

import pytest

from cyberduke.ai.pathfinding import Pathfinder
from cyberduke.config import NPC, PLAYER
from cyberduke.core.geometry import any_cylinder_overlap
from cyberduke.core.spatial import SpatialHash
from cyberduke.entities.player import Player, PlayerController
from cyberduke.simulation import Simulation
from cyberduke.world import Room
from cyberduke.world.surfaces import ElevationState

CELL = 0.2  # flood-fill resolution (m); the player is 0.64 m wide
SUBSTEP = 0.1
DT = 1 / 60
DIRECTIONS = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 45)]


# ----------------------------------------------------------------------------- fixtures
def open_simulation() -> Simulation:
    sim = Simulation()
    for door in sim.room.doors.values():
        door.open_amount = door.travel
    sim.obstacles = sim._collect_obstacles()
    return sim


def key_of(x: float, z: float, up: bool) -> tuple[int, int, bool]:
    return (round(x / CELL), round(z / CELL), up)


class Walker:
    """Drives a PlayerController with nearby-obstacle lookups from a spatial hash."""

    def __init__(self, sim: Simulation):
        self.room = sim.room
        self.player = Player(spawn=sim.level.spawn)
        self.ctrl = PlayerController(self.player, self.room)
        self.index = SpatialHash(sim.obstacles)
        self.reach = PLAYER.collision_radius + SUBSTEP * 2 + 0.05

    def place(self, x: float, z: float, up: bool) -> None:
        p = self.player
        p.x, p.z = x, z
        p.elevation = ElevationState(on_catwalk=up)
        p.floor_y, p.elevation = self.room.floor_height(x, z, p.elevation)
        p.y = p.floor_y + p.eye_height

    def step(self, dx: float, dz: float) -> tuple[float, float, bool]:
        p = self.player
        near = self.index.near(p.x, p.z, self.reach)
        self.ctrl.move_by(dx, dz, DT, near)
        return p.x, p.z, p.elevation.on_catwalk

    def try_move(self, x: float, z: float, up: bool, ux: float, uz: float, distance: float):
        """From (x, z, up), push ``distance`` along (ux, uz) in sub-steps with sliding."""
        self.place(x, z, up)
        steps = max(1, round(distance / SUBSTEP))
        for _ in range(steps):
            self.step(ux * distance / steps, uz * distance / steps)
        p = self.player
        return p.x, p.z, p.elevation.on_catwalk


def flood_fill_player(sim: Simulation):
    """Returns (nodes, edges): nodes map a cell key to a real position; edges are the
    forward moves that were actually possible."""
    walker = Walker(sim)
    sx, sz, _ = sim.level.spawn
    start = key_of(sx, sz, False)
    nodes = {start: (sx, sz, False)}
    edges: dict[tuple, set[tuple]] = {start: set()}
    queue = deque([start])
    while queue:
        k = queue.popleft()
        x, z, up = nodes[k]
        for ux, uz in DIRECTIONS:
            nx, nz, nup = walker.try_move(x, z, up, ux, uz, CELL)
            if math.hypot(nx - x, nz - z) < 1e-4:
                continue
            nk = key_of(nx, nz, nup)
            edges[k].add(nk)
            if nk not in nodes:
                nodes[nk] = (nx, nz, nup)
                edges[nk] = set()
                queue.append(nk)
    return nodes, edges


def cells_where_player_fits(sim: Simulation):
    """Every (cell, layer) where a standing player-sized body is valid and collision-free."""
    room: Room = sim.room
    index = SpatialHash(sim.obstacles)
    r, h = PLAYER.collision_radius, PLAYER.body_height_standing
    b = sim.level.nav_bounds
    fits = {}
    gx0, gx1 = round(b[0] / CELL), round(b[1] / CELL)
    gz0, gz1 = round(b[2] / CELL), round(b[3] / CELL)
    for gx in range(gx0, gx1 + 1):
        for gz in range(gz0, gz1 + 1):
            x, z = gx * CELL, gz * CELL
            if not room.is_valid_position(x, z, r):
                continue
            near = index.near(x, z, r + 0.05)
            for up in (False, True):
                feet, state = room.floor_height(x, z, ElevationState(on_catwalk=up))
                if up and not state.on_catwalk:
                    continue  # the "up" layer only exists on the deck and the top steps
                if any_cylinder_overlap(near, x, z, r, feet, feet + h):
                    continue
                fits[(gx, gz, up)] = (x, z, up)
    return fits


def in_no_go(level, x: float, z: float, *, player: bool) -> bool:
    return any(
        (zone.for_player if player else zone.for_npc) and zone.box.contains_xz(x, z)
        for zone in level.no_go_zones
    )


@pytest.fixture(scope="module")
def sim():
    return open_simulation()


@pytest.fixture(scope="module")
def player_graph(sim):
    return flood_fill_player(sim)


# ----------------------------------------------------------------------------- player
def test_player_can_reach_every_cell_it_fits_in(sim, player_graph):
    nodes, _ = player_graph
    fits = cells_where_player_fits(sim)
    missing = [
        (x, z, up)
        for k, (x, z, up) in fits.items()
        if k not in nodes and not in_no_go(sim.level, x, z, player=True)
    ]
    sample = sorted(missing)[:15]
    assert not missing, (
        f"{len(missing)} cells fit a player but are unreachable from the spawn "
        f"(declare a NoGoZone if intentional). First: {sample}"
    )


def test_player_reaches_the_catwalk_on_both_sides(player_graph):
    nodes, _ = player_graph
    up_cells = {k for k in nodes if k[2]}
    assert any(k[0] > 0 for k in up_cells), "right catwalk unreachable"
    assert any(k[0] < 0 for k in up_cells), "left catwalk unreachable"
    assert any(k[1] * CELL > 40.0 for k in up_cells), "skybridge unreachable"


def test_no_position_is_a_trap(sim, player_graph):
    """Every reachable position must be able to get back to the spawn (strongly connected)."""
    nodes, edges = player_graph
    reverse: dict[tuple, set[tuple]] = {k: set() for k in nodes}
    for src, dsts in edges.items():
        for dst in dsts:
            reverse[dst].add(src)
    sx, sz, _ = sim.level.spawn
    start = key_of(sx, sz, False)
    can_return = {start}
    queue = deque([start])
    while queue:
        k = queue.popleft()
        for prev in reverse[k]:
            if prev not in can_return:
                can_return.add(prev)
                queue.append(prev)
    traps = [nodes[k] for k in nodes if k not in can_return]
    assert not traps, (
        f"{len(traps)} positions can be entered but not left. First: {sorted(traps)[:15]}"
    )


def test_every_reachable_position_can_move_somewhere(sim, player_graph):
    nodes, edges = player_graph
    dead_ends = [nodes[k] for k, out in edges.items() if not out]
    assert not dead_ends, f"wedged positions (no movement possible): {sorted(dead_ends)[:15]}"


@pytest.mark.parametrize("side", [-1, 1])
def test_sweeping_along_the_stair_and_deck_rails_never_stalls(sim, side):
    """Walk up the stairs and along the deck while pushing diagonally into the inner rail -
    exactly how the original ramp/catwalk junction wedged the player."""
    walker = Walker(sim)
    cw = sim.room.catwalk
    rail_outer_edge = 6.18  # collider edge of the inner handrail
    x = side * (rail_outer_edge + PLAYER.collision_radius + 0.02)  # hugging the rail
    walker.place(x, cw.stairs_z_start + 0.1, False)  # standing on the first step
    into_rail = -side * 0.7  # push toward the rail (toward x=0)
    progress = []
    for _ in range(400):
        _, z, _ = walker.step(into_rail * SUBSTEP, 0.7 * SUBSTEP)
        progress.append(z)
        if z > cw.bridge_z_start - 1.0:
            break
    assert progress[-1] > cw.bridge_z_start - 1.0, f"stalled at z={progress[-1]:.2f}"
    assert walker.player.elevation.on_catwalk and walker.player.floor_y == pytest.approx(cw.height)
    # and back down the same way
    for _ in range(400):
        _, z, _ = walker.step(into_rail * SUBSTEP, -0.7 * SUBSTEP)
        if z < cw.stairs_z_start + 0.1:
            break
    assert z < cw.stairs_z_start + 0.1, f"stalled on the way down at z={z:.2f}"
    assert not walker.player.elevation.on_catwalk


def test_closed_door_seals_off_everything_beyond_it():
    sim = Simulation()  # door shut
    nodes, _ = flood_fill_player(sim)
    assert all(z < 10.0 for _, z, _ in nodes.values()), "reached past the shut blast door"


def test_closed_vault_bulkhead_seals_off_the_vault():
    sim = Simulation()
    blast = sim.room.doors["blast_door"]
    blast.open_amount = blast.travel  # only the bulkhead is shut
    sim.obstacles = sim._collect_obstacles()
    nodes, _ = flood_fill_player(sim)
    zs = [z for _, z, _ in nodes.values()]
    assert max(zs) > 17.0, "the corridor up to the bulkhead is reachable"
    assert all(z < 18.0 for z in zs), "reached past the shut vault bulkhead"


# ----------------------------------------------------------------------------- NPCs
def npc_flood_fill(room: Room, nav_obstacles, start):
    pf = Pathfinder()
    index = SpatialHash(nav_obstacles)
    g = pf.grid_size
    b = room.nav_bounds
    gx0, gx1 = round(b[0] / g), round(b[1] / g)
    gz0, gz1 = round(b[2] / g), round(b[3] / g)
    start_cell = (round(start[0] / g), round(start[1] / g))
    seen = {start_cell}
    queue = deque([start_cell])

    def blocked(c):
        x, z = c[0] * g, c[1] * g
        return not room.is_valid_position(x, z, NPC.collision_radius) or pf.is_blocked(
            x, z, index.near(x, z, NPC.collision_radius + 0.05)
        )

    while queue:
        cx, cz = queue.popleft()
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cx + dx, cz + dz)
            if n in seen or not (gx0 <= n[0] <= gx1 and gz0 <= n[1] <= gz1) or blocked(n):
                continue
            seen.add(n)
            queue.append(n)
    return seen, blocked, (gx0, gx1, gz0, gz1)


def test_npc_can_reach_every_cell_it_fits_in(sim):
    nav = sim.room.get_nav_obstacles()
    g0 = sim.level.guards[0]
    reached, blocked, (gx0, gx1, gz0, gz1) = npc_flood_fill(sim.room, nav, (g0.x, g0.z))
    pf = Pathfinder()
    missing = []
    for gx in range(gx0, gx1 + 1):
        for gz in range(gz0, gz1 + 1):
            if (gx, gz) in reached or blocked((gx, gz)):
                continue
            x, z = gx * pf.grid_size, gz * pf.grid_size
            if not in_no_go(sim.level, x, z, player=False):
                missing.append((x, z))
    assert not missing, f"{len(missing)} NPC-sized cells are unreachable: {sorted(missing)[:15]}"


def test_npcs_can_route_to_their_posts_and_to_the_distraction_point(sim):
    nav = sim.room.get_nav_obstacles()
    pf = Pathfinder()
    for g in sim.level.guards:
        targets = list(g.patrol_points) + ([g.wall_pos] if g.wall_pos is not None else [])
        for target in targets:
            path = pf.find_path(g.x, g.z, *target, nav, sim.level.nav_bounds)
            assert path.found and not path.adjusted, f"guard at ({g.x},{g.z}) cannot reach {target}"
    vm = sim.level.vending_machines[0]
    g2 = sim.level.guards[1]
    path = pf.find_path(g2.x, g2.z, vm.x, vm.z + 1.2, nav, sim.level.nav_bounds)
    assert path.found and not path.adjusted


def test_npcs_never_enter_the_stairs(sim):
    """NPC navigation is ground-level; the stair footprints must be no-go for them."""
    nav = sim.room.get_nav_obstacles()
    pf = Pathfinder()
    cw = sim.room.catwalk
    for side in (-1, 1):
        x = side * (cw.inner_x + cw.outer_x) / 2.0
        assert pf.is_blocked(x, (cw.stairs_z_start + cw.stairs_z_end) / 2.0, nav)
