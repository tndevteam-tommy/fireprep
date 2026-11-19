import numpy as np
import pytest

from unprepared.config import NavigationSettings
from unprepared.core.math3d import BoundingBox3D, Vector3D
from unprepared.entities import AIState, Door
from unprepared.world import NavGrid


@pytest.fixture
def small_settings():
    return NavigationSettings(cell_size=0.5, agent_radius=0.4, bounds=(-10.0, -10.0, 10.0, 10.0))


def test_grid_shape_and_mapping(small_settings):
    grid = NavGrid([], small_settings)
    assert grid.cols == 40 and grid.rows == 40
    assert grid.to_cell(-10.0, -10.0) == (0, 0)
    assert grid.to_cell(9.99, 9.99) == (39, 39)
    assert grid.to_cell(100, -100) == (39, 0)       # clamped
    c = grid.to_world((0, 0))
    assert (c.x, c.z) == (-9.75, -9.75)
    assert not grid.static_blocked.any()


def test_obstacles_are_inflated(small_settings):
    wall = BoundingBox3D(Vector3D(0, 0, -5), Vector3D(0.5, 4, 5))
    grid = NavGrid([wall], small_settings)
    assert grid.is_blocked(0.25, 0.0)
    assert grid.is_blocked(-0.3, 0.0)        # inside the 0.4 inflation
    assert not grid.is_blocked(-1.0, 0.0)
    assert isinstance(grid.static_blocked, np.ndarray)


def test_straight_path_is_a_single_segment(small_settings):
    grid = NavGrid([], small_settings)
    path = grid.find_path(Vector3D(-5, 0, 0), Vector3D(5, 0, 0))
    assert path == [Vector3D(5, 0, 0)]


def test_path_goes_around_a_wall(small_settings):
    wall = BoundingBox3D(Vector3D(0, 0, -5), Vector3D(0.5, 4, 3))     # gap south of z=3
    grid = NavGrid([wall], small_settings)
    path = grid.find_path(Vector3D(-5, 0, 0), Vector3D(5, 0, 0))
    assert path and path[-1] == Vector3D(5, 0, 0)
    assert max(p.z for p in path) > 3.0                 # detoured through the gap
    occ = grid.occupancy()
    for a, b in zip([Vector3D(-5, 0, 0)] + path, path):
        for t in np.linspace(0, 1, 40):
            assert not grid.is_blocked(a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t, occ)


def test_no_path_when_enclosed(small_settings):
    box = [BoundingBox3D(Vector3D(-3, 0, -3), Vector3D(3, 4, -2.5)), BoundingBox3D(Vector3D(-3, 0, 2.5), Vector3D(3, 4, 3)),
           BoundingBox3D(Vector3D(-3, 0, -3), Vector3D(-2.5, 4, 3)), BoundingBox3D(Vector3D(2.5, 0, -3), Vector3D(3, 4, 3))]
    grid = NavGrid(box, small_settings)
    assert grid.find_path(Vector3D(0, 0, 0), Vector3D(8, 0, 8)) == []


def test_goal_inside_obstacle_snaps_to_nearest_free(small_settings):
    wall = BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 4, 1))
    grid = NavGrid([wall], small_settings)
    path = grid.find_path(Vector3D(-5, 0, 0.5), Vector3D(0.5, 0, 0.5))
    assert path and path[-1] == Vector3D(0.5, 0, 0.5)


class TestDoors:
    # a room x -5..5, z -6.5..0.5 with a doorway in the north wall between x -1.5 and 1.5
    walls = [BoundingBox3D(Vector3D(-5, 0, 0), Vector3D(-1.5, 4, 0.5)), BoundingBox3D(Vector3D(1.5, 0, 0), Vector3D(5, 4, 0.5)),
             BoundingBox3D(Vector3D(-5, 0, -6.5), Vector3D(-4.5, 4, 0.5)), BoundingBox3D(Vector3D(4.5, 0, -6.5), Vector3D(5, 4, 0.5)),
             BoundingBox3D(Vector3D(-5, 0, -6.5), Vector3D(5, 4, -6))]
    door = Door("d", Vector3D(-1.5, 0.0, 0.0), is_locked=True)

    def test_locked_door_blocks_strangers(self, small_settings):
        grid = NavGrid(self.walls, small_settings)
        assert grid.find_path(Vector3D(0, 0, 3), Vector3D(0, 0, -3), [self.door], can_open=lambda d: False) == []

    def test_owner_can_path_through_locked_door(self, small_settings):
        grid = NavGrid(self.walls, small_settings)
        path = grid.find_path(Vector3D(0, 0, 3), Vector3D(0, 0, -3), [self.door], can_open=lambda d: True)
        assert path and path[-1] == Vector3D(0, 0, -3)

    def test_open_door_is_passable_for_everyone(self, small_settings):
        door = Door("d", Vector3D(-1.5, 0.0, 0.0), is_locked=False, is_open=True)
        grid = NavGrid(self.walls, small_settings)
        assert grid.find_path(Vector3D(0, 0, 3), Vector3D(0, 0, -3), [door], can_open=lambda d: False)


def test_town_routes_exist(world):
    """Every scheduled location of every NPC is reachable from every other one once the doors
    are open (a shopper's route into a shop only exists while the shopkeeper has unlocked it)."""
    for door in world.doors:
        door.is_locked = False
    for npc in world.npcs:
        spots = [t.target_pos for t in npc.schedule]
        for a in spots:
            for b in spots:
                if a != b:
                    npc.position = a.copy()
                    assert world.find_path(npc, b), (npc.name, a, b)


def test_everyone_can_reach_their_own_rooms_with_their_keys(world):
    """With doors as they are at 07:00, every NPC can get from its bed to each of its *own*
    places (shops and homes it has keys to); visits to other people's shops wait for opening."""
    employees = {"Dorran", "Lys"}                    # work in somebody else's shop: they wait for it to open
    for npc in world.npcs:
        home = npc.task_for(3)
        for task in npc.schedule:
            if task.target_state in (AIState.SHOPPING, AIState.DRINKING, AIState.STROLLING):
                continue
            if npc.name in employees and task.target_state is AIState.WORKING:
                continue
            npc.position = home.target_pos.copy()
            assert world.find_path(npc, task.target_pos), (npc.name, task.action_text)
