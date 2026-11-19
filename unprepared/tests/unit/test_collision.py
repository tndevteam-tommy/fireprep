import pytest

from unprepared.config import PlayerSettings
from unprepared.core.math3d import BoundingBox3D, Vector3D
from unprepared.entities import Door, Player
from unprepared.systems import move_with_collision


def make_player(x=0.0, z=0.0):
    return Player(Vector3D(x, 0.0, z), settings=PlayerSettings())


WALL = BoundingBox3D(Vector3D(2.0, 0, -5), Vector3D(2.5, 4, 5))   # wall along z at x 2..2.5


def test_free_movement_without_colliders():
    p = make_player()
    assert move_with_collision(p, 1.0, -2.0, []) == (1.0, -2.0)
    assert p.position == Vector3D(1.0, 0.0, -2.0)


def test_blocked_by_wall_after_advancing_to_contact():
    p = make_player()
    dx, dz = move_with_collision(p, 3.0, 0.0, [WALL])
    assert dz == 0.0 and 1.4 <= dx <= 1.7            # wall face at 2.0 minus half-width 0.3, in 0.2 sub-steps
    assert p.position.x == pytest.approx(dx)
    assert move_with_collision(p, 3.0, 0.0, [WALL]) == (0.0, 0.0)   # now truly stuck


def test_slides_along_wall():
    p = make_player(x=1.6)
    dx, dz = move_with_collision(p, 0.5, 1.0, [WALL])   # diagonal into the wall
    assert dx == 0.0 and dz == 1.0
    assert p.position == Vector3D(1.6, 0.0, 1.0)


def test_can_approach_up_to_half_width():
    p = make_player(x=1.0)
    move_with_collision(p, 0.69, 0.0, [WALL])      # box would reach 1.99 < 2.0: fine
    assert p.position.x == pytest.approx(1.69)
    move_with_collision(p, 0.02, 0.0, [WALL])      # box would touch 2.01: blocked
    assert p.position.x == pytest.approx(1.69)


def test_low_obstacles_still_block_feet():
    anvil = BoundingBox3D(Vector3D(-0.5, 0, -2.4), Vector3D(0.5, 0.8, -1.6))   # below the eye, at foot level
    p = make_player()
    move_with_collision(p, 0.0, -1.5, [anvil])
    assert -1.3 <= p.position.z < -0.9              # stopped in front of the anvil face at z=-1.6


def test_tall_objects_above_head_do_not_block():
    beam = BoundingBox3D(Vector3D(-5, 2.5, -2.4), Vector3D(5, 3, -1.6))   # head reaches y=2.0
    p = make_player()
    move_with_collision(p, 0.0, -1.5, [beam])
    assert p.position.z == -1.5


def test_closed_and_open_door_leaves():
    door = Door("d", Vector3D(-1.5, 0.0, -2.0), is_locked=False)   # spans x -1.5..1.5 at z=-2
    p = make_player()
    move_with_collision(p, 0.0, -1.8, [door.collider()])
    assert p.position.z > -1.6                      # closed: stopped at the leaf (z -2.1..-1.9 minus half-width)
    door.is_open = True
    p = make_player()
    move_with_collision(p, 0.0, -1.8, [door.collider()])
    assert p.position.z == pytest.approx(-1.8)      # open: the doorway is free at x=0
    p2 = make_player(x=-1.5)                        # ... but the folded leaf at the hinge blocks
    move_with_collision(p2, 0.0, -1.8, [door.collider()])
    assert p2.position.z > -1.6


def test_player_cannot_leave_the_town_or_enter_buildings_through_walls(world):
    player = world.spawn_player()
    colliders = world.colliders(include_npcs=False)
    # the north town wall's inner face is at z -61; start in the open at (-8, -30)
    player.position = world.spot(-8.0, 0.0, -30.0)
    for _ in range(500):
        move_with_collision(player, 0.0, -0.1, colliders)
        player.position.y = world.floor_height_at(player.position.x, player.position.z, player.position.y)   # (uphill)
    assert player.position.z >= -61 + 0.3 - 1e-9
    # the general store's west wall at x -18..-17.5 (z -16..-6): walk east from (-21, -9)
    player.position = world.spot(-21.0, 0.0, -9.0)
    for _ in range(100):
        move_with_collision(player, 0.1, 0.0, colliders)
        player.position.y = world.floor_height_at(player.position.x, player.position.z, player.position.y)
    assert player.position.x <= -18.0 - 0.3 + 1e-9
