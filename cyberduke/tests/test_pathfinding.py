import math
import time

from cyberduke.ai.npc import BaseNPC, HumanoidSpec
from cyberduke.ai.pathfinding import Path, Pathfinder
from cyberduke.core.geometry import AABB
from cyberduke.levels import build_facility
from cyberduke.world import Room

BOUNDS = (-10.0, 10.0, -10.0, 10.0)


def _walkable(path, obstacles, pf, start):
    """Every smoothed leg must be clear for the agent radius."""
    x, z = start
    for wx, wz in path.waypoints:
        assert pf.line_clear(x, z, wx, wz, obstacles), f"leg ({x},{z})->({wx},{wz}) is blocked"
        x, z = wx, wz


def test_direct_line_when_nothing_is_in_the_way():
    pf = Pathfinder()
    assert pf.find_path(0, 0, 5, 5, [], bounds=BOUNDS) == Path([(5, 5)])
    far_box = [AABB(8, 9, 8, 9)]
    assert pf.find_path(0, 0, 5, 0, far_box, bounds=BOUNDS).waypoints == [(5, 0)]


def test_routes_around_a_pillar_and_every_leg_is_clear():
    pf = Pathfinder()
    pillar = [AABB(-0.5, 0.5, 1.5, 2.5)]
    path = pf.find_path(0.0, 0.0, 0.0, 4.0, pillar, bounds=BOUNDS)
    assert path.found and not path.adjusted and path.expanded > 0
    assert len(path.waypoints) >= 2, "must add at least one detour waypoint"
    assert path.destination == (0.0, 4.0)
    _walkable(path, pillar, pf, (0.0, 0.0))


def test_goal_inside_an_obstacle_ends_at_the_nearest_free_cell():
    pf = Pathfinder()
    crate = [AABB(2.0, 3.0, 2.0, 3.0)]
    path = pf.find_path(0.0, 0.0, 2.5, 2.5, crate, bounds=BOUNDS)
    assert path.found and path.adjusted
    dx, dz = path.destination
    assert not pf.is_blocked(dx, dz, crate)
    assert abs(dx - 2.5) < 1.2 and abs(dz - 2.5) < 1.2, "as close as it can get"
    _walkable(path, crate, pf, (0.0, 0.0))


def test_obstacles_above_the_agent_are_ignored():
    pf = Pathfinder()
    catwalk_deck = [AABB(-5, 5, 1, 3, 2.75, 2.85)]
    assert pf.find_path(0, 0, 0, 5, catwalk_deck, bounds=BOUNDS).waypoints == [(0, 5)]


def test_agent_on_catwalk_ignores_ground_level_boxes():
    pf = Pathfinder()
    crate = [AABB(-1, 1, 1, 3, 0.0, 1.65)]
    assert pf.find_path(0, 0, 0, 5, crate, agent_y=2.75, bounds=BOUNDS).waypoints == [(0, 5)]


def test_unreachable_goal_reports_not_found_and_falls_back_to_a_straight_line():
    pf = Pathfinder()
    walls = [AABB(1, 5, 1, 1.2), AABB(1, 5, 4.8, 5), AABB(1, 1.2, 1, 5), AABB(4.8, 5, 1, 5)]
    path = pf.find_path(0, 0, 3, 3, walls, bounds=BOUNDS)
    assert not path.found and path.waypoints == [(3, 3)]


def test_search_budget_scales_with_the_bounds():
    pf = Pathfinder(grid_size=0.4)
    small = pf.search_budget((-2.0, 2.0, -2.0, 2.0))
    facility = pf.search_budget(build_facility().nav_bounds)
    assert small < 200 < 800 < facility
    assert Pathfinder(max_expansions=123).search_budget(BOUNDS) == 123


def test_cross_sector_route_through_the_open_doors():
    """Regression for the original fixed 800-node budget, which exhausted on this route."""
    level = build_facility()
    room = Room(level)
    for door in room.doors.values():  # blast door and vault bulkhead
        door.open_amount = door.travel
    obstacles = room.get_obstacles()
    pf = Pathfinder()
    t0 = time.perf_counter()
    path = pf.find_path(0.0, -6.0, 0.0, 36.0, obstacles, bounds=level.nav_bounds)
    elapsed = time.perf_counter() - t0
    assert path.found, f"search exhausted after {path.expanded} expansions"
    assert len(path.waypoints) >= 2
    assert path.destination == (0.0, 36.0)
    _walkable(path, obstacles, pf, (0.0, -6.0))
    assert elapsed < 0.5, f"cross-level search took {elapsed:.2f}s"


def test_shut_door_makes_the_vault_unreachable_from_sector_one():
    level = build_facility()
    room = Room(level)
    path = Pathfinder().find_path(
        0.0, -6.0, 0.0, 36.0, room.get_obstacles(), bounds=level.nav_bounds
    )
    assert not path.found


# ----------------------------------------------------------------------------- navigation
def make_npc(x, z):
    return BaseNPC(x, 0.0, z, spec=HumanoidSpec(), nav_bounds=BOUNDS)


def test_navigate_towards_arrives_at_the_nearest_reachable_point_for_a_blocked_goal():
    """Regression: the README courier walked into a pillar forever because arrival was only
    ever checked against the unreachable original goal (and replanned every frame)."""
    npc = make_npc(0.0, 0.0)
    pillar = [AABB(2.1, 3.1, 0.7, 1.7)]
    arrived = False
    for _ in range(600):
        if npc.navigate_towards(
            3.0, 2.0, 1 / 60, obstacles=pillar
        ):  # goal within the agent radius of the pillar
            arrived = True
            break
    assert arrived
    assert not npc.pathfinder.is_blocked(npc.x, npc.z, pillar)
    assert abs(npc.x - 3.0) < 1.0 and abs(npc.z - 2.0) < 1.0


def test_navigate_towards_does_not_replan_every_frame_once_arrived():
    npc = make_npc(0.0, 0.0)
    pillar = [AABB(2.1, 3.1, 0.7, 1.7)]
    while not npc.navigate_towards(3.0, 2.0, 1 / 60, obstacles=pillar):
        pass
    calls = []
    original = npc.pathfinder.find_path
    npc.pathfinder.find_path = lambda *a, **k: calls.append(1) or original(*a, **k)
    for _ in range(10):
        assert npc.navigate_towards(3.0, 2.0, 1 / 60, obstacles=pillar)
    assert calls == []


def test_navigate_towards_without_obstacles_walks_straight():
    npc = make_npc(0.0, 0.0)
    for _ in range(600):
        if npc.navigate_towards(0.0, 3.0, 1 / 60, obstacles=[]):
            break
    assert abs(npc.z - 3.0) < 0.2 and abs(npc.x) < 1e-6


def test_spatial_hash_search_matches_a_brute_force_blocked_test_on_the_full_level():
    """The hashed cell test must agree with testing every obstacle (no box may be missed
    because it straddles hash cells), and a maze route must stay fast."""
    from cyberduke.core.geometry import any_cylinder_overlap
    from cyberduke.core.spatial import SpatialHash

    level = build_facility()
    room = Room(level)
    for door in room.doors.values():
        door.open_amount = door.travel
    obstacles = room.get_obstacles()
    pf = Pathfinder()
    index = SpatialHash(obstacles, cell_size=2.0)
    reach = pf.agent_radius + pf.grid_size
    for gx in range(-40, 41, 3):
        for gz in range(-25, 200, 3):
            x, z = gx * pf.grid_size, gz * pf.grid_size
            brute = any_cylinder_overlap(obstacles, x, z, pf.agent_radius, 0.0, pf.agent_height)
            hashed = any_cylinder_overlap(
                index.near(x, z, reach), x, z, pf.agent_radius, 0.0, pf.agent_height
            )
            assert brute == hashed, (x, z)
    t0 = time.perf_counter()
    path = pf.find_path(0.0, 49.8, 0.0, 73.6, obstacles, bounds=level.nav_bounds)
    elapsed = time.perf_counter() - t0
    assert path.found and elapsed < 0.6, f"maze route took {elapsed:.2f}s"
    _walkable(path, obstacles, pf, (0.0, 49.8))


def test_a_goal_buried_deep_in_an_obstacle_still_resolves_to_the_nearest_free_point():
    """Regression: the free-cell search around a blocked goal stopped after three cells, so a
    goal in the middle of a staircase-sized block targeted a blocked cell, no route was ever
    found and the agent pressed against the block forever."""
    pf = Pathfinder()
    block = [AABB(-1.75, 1.75, -1.6, 1.6, 0.0, 3.0)]  # the goal below is 1.6+ m from any edge
    path = pf.find_path(0.0, -5.0, 0.0, 0.0, block, BOUNDS)
    assert path.found and path.adjusted
    dx, dz = path.destination
    assert not pf.is_blocked(dx, dz, block) and math.hypot(dx, dz) < 2.5


def test_an_agent_pinned_against_an_obstacle_gives_up_and_counts_as_arrived():
    """A goal the grid cannot reach (route not found: a straight line into the wall) must
    not leave the agent walking on the spot forever."""
    npc = make_npc(0.0, 0.0)
    sealed = [AABB(2.0, 2.4, -20.0, 20.0), AABB(5.0, 5.4, -20.0, 20.0), AABB(2.0, 5.4, 2.0, 2.4),
              AABB(2.0, 5.4, -2.4, -2.0)]  # fmt: skip
    arrived_at = None
    for frame in range(int(10 / (1 / 60))):
        if npc.navigate_towards(3.7, 0.0, 1 / 60, obstacles=sealed):  # inside a sealed box
            arrived_at = frame / 60
            break
    assert arrived_at is not None and arrived_at < 8.0
    assert npc.x < 2.0 - npc.collision_radius + 0.05, "it stopped at the wall, not in it"
    assert npc.navigate_towards(3.7, 0.0, 1 / 60, obstacles=sealed), "and stays arrived"


def _frames_until(step, seconds: float) -> float:
    """Runs ``step`` (at 60 fps) until it returns True; how long that took (inf: never)."""
    for frame in range(int(seconds * 60)):
        if step():
            return frame / 60
    return math.inf


def test_a_walker_stops_short_of_a_spot_somebody_is_standing_on():
    """Regression: heading for a spot another person was standing on (the queue's place,
    the end of the counter), the walker pressed into them and walked on the spot until the
    walk timed out. Held up right by it, the spot counts as taken: as near as it gets."""
    npc = make_npc(0.0, 0.0)
    other = (3.0, 0.0, 0.35)  # standing right where it is going
    took = _frames_until(
        lambda: npc.navigate_towards(3.0, 0.0, 1 / 60, obstacles=[], stop_dist=0.3, avoid=[other]),
        8.0,
    )
    assert npc.arrival == "taken" and took < 6.0
    assert 0.55 < math.hypot(npc.x - 3.0, npc.z) < 1.6


def test_two_walkers_going_opposite_ways_get_past_each_other():
    """Two people walking at each other along the same line used to mirror each other's
    sidestep and dance on the spot; one of them gives way (steps back and waits) if they
    get nowhere for a moment."""
    a, b = make_npc(-3.0, 0.0), make_npc(3.0, 0.0)
    done = {}
    for frame in range(int(20 / (1 / 60))):
        for me, other, goal in ((a, b, (3.0, 0.0)), (b, a, (-3.0, 0.0))):
            if me in done:
                continue
            body = (other.x, other.z, other.collision_radius)
            if me.navigate_towards(*goal, 1 / 60, obstacles=[], stop_dist=0.3, avoid=[body]):
                done[me] = frame / 60
        assert math.hypot(a.x - b.x, a.z - b.z) > 0.5, "never through each other"
        if len(done) == 2:
            break
    assert len(done) == 2 and max(done.values()) < 15.0, done
    assert a.arrival == b.arrival == "there"


def test_put_inside_an_obstacle_a_walker_finds_its_way_out():
    """Regression: a scripted step (off a stool) left a body overlapping the stool's
    collider; per-axis sliding refused every step (each one still overlapped) and it was
    stuck there for good - its walks all 'arriving' where it stood."""
    stool = [AABB(-0.25, 0.25, -0.25, 0.25, 0.0, 0.6)]
    npc = make_npc(0.5, 0.0)  # 0.05 m inside the stool's square footprint
    assert npc._body_collides(npc.x, npc.z, stool)
    took = _frames_until(
        lambda: npc.navigate_towards(4.0, 3.0, 1 / 60, obstacles=stool, stop_dist=0.3), 10.0
    )
    assert npc.arrival == "there" and math.hypot(npc.x - 4.0, npc.z - 3.0) <= 0.3
    assert took < 8.0


def test_a_generous_stop_distance_does_not_cut_the_corners_of_the_route():
    """Regression: every corner of the route counted as turned from ``stop_dist`` away, so
    a walk with a generous stop distance cut the corner straight into whatever the route
    went round - and stuck against it."""
    wall = [AABB(-1.0, 1.0, 1.0, 1.6, 0.0, 2.0)]  # in the way between start and goal
    npc = make_npc(0.0, 0.0)
    _frames_until(
        lambda: npc.navigate_towards(0.0, 4.5, 1 / 60, obstacles=wall, stop_dist=2.0), 15.0
    )
    assert npc.arrival == "there" and math.hypot(npc.x, npc.z - 4.5) <= 2.0
    assert not npc._body_collides(npc.x, npc.z, wall)
