"""The watchtower: a hollow tower with a winding stair, windows and a look-out platform that the
player climbs step by step and NPCs climb by waypoints; nothing falls through floors or walks
through the parapet."""
import math

import pytest

from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState
from unprepared.systems import MoveIntent
from unprepared.world.tower import STEP_CLEARANCE, STEP_RISE, STEP_UP, build_tower
from tests.helpers import spot
from unprepared.world.levels.town import GUARD_BEDS, TOWER_PLATFORM, TOWER_TOP, TOWER_X1, TOWER_X2, TOWER_Z1, TOWER_Z2


def park_everyone(game):
    for n in game.world.npcs:
        n.schedule, n.current_task, n.path = [], None, []
        n.position = spot(game.world, 110.0, 10.0)           # far out in the country
        n.state = AIState.IDLE


def base(w) -> float:
    """The tower's floor: it shares the guard house's pad, up on the knoll."""
    return w.building_elevation(next(b for b in w.level.buildings if b.name == "watchtower"))


def T(w, x, y, z) -> Vector3D:
    """A spot in the tower given relative to its north-west corner (x, z) and its floor (y)."""
    return Vector3D(TOWER_X1 + x, base(w) + y, TOWER_Z1 + z)


def walk_to(game, target, max_frames=400):
    """Face ``target`` and walk forward until within reach (the way a player would)."""
    p = game.player
    for _ in range(max_frames):
        dx, dz = target.x - p.position.x, target.z - p.position.z
        if math.hypot(dx, dz) < 0.15:
            return True
        p.rotation_y = math.degrees(math.atan2(dx, -dz)) % 360.0
        game.update(0.05, MoveIntent(forward=True))
    return False


class TestGeometry:
    def test_build_tower_makes_a_climbable_spiral(self):
        tower = build_tower("t", 0.0, 0.0, 6.0, 6.0, 13.0, 2.0)
        steps = [p for p in tower.props if p.kind == "step"]
        assert len(steps) == 40
        tops = [p.max_point[1] for p in steps]
        assert tops == sorted(tops) and all(b - a <= STEP_UP for a, b in zip(tops, tops[1:]))
        assert tops[0] <= STEP_UP and abs(tops[-1] - 13.0) < 1e-6
        assert all(not p.solid for p in steps)                               # floors, not obstacles
        # every step stays inside the walls and butts against one of them (a hair short of it: no
        # face of a step in the plane of the wall's)
        for p in steps:
            assert 0.5 < p.min_point[0] and p.max_point[0] < 5.5 and 0.5 < p.min_point[2] and p.max_point[2] < 5.5
            gap = min(p.min_point[0] - 0.5, 5.5 - p.max_point[0], p.min_point[2] - 0.5, 5.5 - p.max_point[2])
            assert gap == pytest.approx(STEP_CLEARANCE)

    def test_the_platform_leaves_head_room_over_the_top_steps(self):
        tower = build_tower("t", 0.0, 0.0, 6.0, 6.0, 13.0, 2.0)
        steps = [p for p in tower.props if p.kind == "step"]
        platform = [p for p in tower.props if p.kind == "floor"]
        assert platform and all(abs(p.max_point[1] - 13.0) < 1e-6 for p in platform)
        for step in steps:
            for slab in platform:
                overlap = (step.min_point[0] < slab.max_point[0] and step.max_point[0] > slab.min_point[0]
                           and step.min_point[2] < slab.max_point[2] and step.max_point[2] > slab.min_point[2])
                if overlap:
                    assert step.max_point[1] + 1.75 <= slab.min_point[1] + 1e-6, step.name
        assert tower.stair.waypoints[-1] == tower.stair.top
        assert all(b[1] >= a[1] for a, b in zip(tower.stair.waypoints, tower.stair.waypoints[1:]))

    def test_windows_and_the_door_are_cut_into_the_walls(self):
        tower = build_tower("t", 0.0, 0.0, 6.0, 6.0, 13.0, 2.0)
        walls = tower.walls
        assert any(w.min_point[1] > 3.0 for w in walls)                      # pieces that start above the ground: openings below them
        south = [w for w in walls if w.min_point[2] >= 5.5]
        assert any(w.max_point[0] <= 2.0 for w in south) and any(w.min_point[0] >= 4.0 for w in south)
        assert any(w.min_point[0] == 2.0 and w.min_point[1] == 3.5 for w in south)   # the lintel over the door
        assert tower.wall_top > 13.0                                         # a parapet above the platform

    def test_the_town_has_one_behind_the_guard_house(self, world):
        assert [s.name for s in world.stairs] == ["watchtower", "ashgrove_stair", "vane_stair"]   # (and the manors' stairs)
        stair = world.stairs[0]
        assert stair.region == (TOWER_X1, TOWER_Z1, TOWER_X2, TOWER_Z2) and Vector3D(*stair.top) == world.spot(*TOWER_TOP)
        assert world.area_of(world.spot(*TOWER_TOP)) == "the watchtower"
        assert world.is_indoors(T(world, 3.0, 0.0, 4.0))
        tower = next(b for b in world.level.buildings if b.name == "watchtower")
        assert tower.open_top and tower.wall_height > 13.0
        assert world.find_door("tower_door").is_locked and world.find_door("tower_door").lock_difficulty == 2
        assert len([f for f in world.floors if f.kind == "step" and f.name.startswith("watchtower")]) == 40
        assert any(f.kind == "floor" for f in world.floors)


class TestPlayerClimbs:
    def test_up_the_stairs_to_the_platform_and_down_again(self, game):
        park_everyone(game)
        w, p = game.world, game.player
        stair = w.stairs[0]
        p.position = Vector3D(*stair.bottom)
        heights = []
        for wp in stair.waypoints:
            assert walk_to(game, Vector3D(*wp)), wp
            heights.append(p.position.y)
            assert abs(p.position.y - wp[1]) < 1e-6
        assert heights == sorted(heights) and p.position.y == pytest.approx(base(w) + TOWER_PLATFORM)
        assert w.floor_height_at(p.position.x, p.position.z, p.position.y) == pytest.approx(base(w) + TOWER_PLATFORM)
        for wp in reversed(stair.waypoints[:-1]):
            assert walk_to(game, Vector3D(*wp)), wp
        assert p.position.y == pytest.approx(base(w) + STEP_RISE)            # back on the first step
        assert walk_to(game, Vector3D(*stair.bottom)) and p.position.y == pytest.approx(base(w))

    def test_a_step_is_never_more_than_a_stride_up_and_cannot_be_jumped(self, game):
        park_everyone(game)
        w, p = game.world, game.player
        y0 = base(w)
        p.position = T(w, 1.5, 0.0, 4.5)
        first, fourth = T(w, 1.3, 0.0, 3.6), T(w, 1.3, 0.0, 1.3)
        assert w.floor_height_at(first.x, first.z, y0) == pytest.approx(y0 + STEP_RISE)   # the first step: reachable
        assert w.floor_height_at(fourth.x, fourth.z, y0) == pytest.approx(y0)             # the fourth (1.3 up): not from the ground
        blocking = [c for c in w.colliders() if c.min_point.z <= fourth.z <= c.max_point.z and c.min_point.x <= fourth.x <= c.max_point.x]
        assert blocking                                                                # ... it is solid from down here

    def test_the_parapet_keeps_you_on_the_platform(self, game):
        park_everyone(game)
        p = game.player
        top = base(game.world) + TOWER_PLATFORM
        p.position = T(game.world, 4.5, TOWER_PLATFORM, 3.2)
        p.rotation_y = 90.0                                                            # east, towards the wall
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True))
        assert p.position.x < TOWER_X2 - 0.5 and p.position.y == pytest.approx(top)
        p.rotation_y = 180.0                                                           # south
        for _ in range(80):
            game.update(0.05, MoveIntent(forward=True))
        assert p.position.z < TOWER_Z2 - 0.5 and p.position.y == pytest.approx(top)

    def test_stepping_off_the_platform_edge_lands_on_the_top_steps(self, game):
        park_everyone(game)
        p = game.player
        y0 = base(game.world)
        p.position = T(game.world, 4.7, TOWER_PLATFORM, 2.7)                           # the platform's edge over the stair
        assert walk_to(game, T(game.world, 4.7, 0.0, 1.3))                             # onto the last step
        assert p.position.y == pytest.approx(y0 + 13.0) or p.position.y > y0 + 12.0
        assert walk_to(game, T(game.world, 3.0, 0.0, 1.3))                             # down a couple more
        assert y0 + 11.5 < p.position.y < y0 + 13.0

    def test_the_floor_under_the_street_is_the_ground(self, game):
        w = game.world
        ground = w.ground_height(0.0, 4.0)
        assert w.floor_height_at(0.0, 4.0, 0.0) == ground and w.floor_height_at(0.0, 4.0, ground + 5.0) == ground


class TestGuardsClimb:
    def test_the_tower_watch_goes_up_at_dawn_and_down_to_eat(self, game):
        w = game.world
        aldric = w.find_npc("Guard Aldric")
        y0 = base(w)
        assert aldric.is_guard and aldric.position == w.spot(*TOWER_TOP)               # 07:00: already on watch
        assert any(i.id == "tower_key" for i in aldric.inventory)
        w.wait_hours(7)                                                                # 14:00: down for a meal
        assert aldric.state is AIState.TRAVELING and aldric.path and aldric.path[1].y < y0 + TOWER_PLATFORM
        ys = []
        for _ in range(120):
            game.update(0.25, MoveIntent())
            ys.append(aldric.position.y)
            if aldric.state is AIState.EATING:
                break
        assert aldric.state is AIState.EATING and aldric.position.y == pytest.approx(y0)
        assert all(b <= a + 1e-6 for a, b in zip(ys, ys[1:]))                          # down, never up, step by step
        door = w.find_door("tower_door")
        assert door.is_locked and not door.is_open                                      # locked behind him
        w.wait_hours(1)                                                                # 15:00: back up
        for _ in range(120):
            game.update(0.25, MoveIntent())
            if aldric.state is AIState.GUARDING:
                break
        assert aldric.state is AIState.GUARDING and aldric.position.y == pytest.approx(y0 + TOWER_PLATFORM)

    def test_find_path_splices_the_stair_in(self, game):
        w = game.world
        y0 = base(w)
        aldric = w.find_npc("Guard Aldric")
        down = w.find_path(aldric, T(w, 3.0, 0.0, 14.0))                              # into the guard house
        assert down and down[2].y < y0 + TOWER_PLATFORM and down[-1].y == pytest.approx(y0)
        assert [p.y for p in down] == sorted((p.y for p in down), reverse=True)
        vance = w.find_npc("Guard Captain Vance")
        vance.position = T(w, 0.5, 0.0, 12.0)
        up = w.find_path(vance, w.spot(*TOWER_TOP))
        assert up and up[-1].y == pytest.approx(y0 + TOWER_PLATFORM) and any(p.y == pytest.approx(y0) for p in up)
        assert [p.y for p in up] == sorted(p.y for p in up)
        assert w.find_path(w.find_npc("Pell"), w.spot(*TOWER_TOP)) == []              # no key to the tower door

    def test_a_guard_up_the_tower_is_no_obstacle_below_and_cannot_see_through_the_floor(self, game):
        w = game.world
        aldric = w.find_npc("Guard Aldric")
        game.player.position = Vector3D(aldric.position.x, base(w), aldric.position.z)  # right under him
        assert not any(abs(b[0] - aldric.position.x) < 0.1 for b in w.bodies_near(aldric))
        assert not aldric.has_line_of_sight(game.player.position, w.sight_blockers(), 1.35)
        for c in w.colliders():
            assert not game.player.collision_box().intersects(c)

    def test_the_barracks_sleeps_the_whole_watch(self, world):
        assert len(GUARD_BEDS) == 6
        for i, bed in enumerate(GUARD_BEDS):
            assert world.area_of(world.spot(*bed)) == "the guard house"
            assert world.prop_under(world.spot(*bed)).name == f"guard_bed_{i + 1}"
        aldric = world.find_npc("Guard Aldric")
        assert aldric.task_for(3).target_pos == world.spot(*GUARD_BEDS[3])
        assert len(world.guards()) == 7
        sleepers = {t.target_pos.copy().x * 1000 + t.target_pos.z for g in world.guards() if g.name != "Jailer Hesk"
                    for t in g.schedule if t.target_state is AIState.SLEEPING}
        assert len(sleepers) == 6                                                       # a bed each
