"""PlayerController: movement, collision sliding, elevation, look clamping."""

import math

import pytest

from cyberduke.config import PLAYER
from cyberduke.core.geometry import AABB
from cyberduke.entities.player import Player, PlayerController
from cyberduke.levels import build_facility
from cyberduke.world import Room

DT = 1 / 60


def make():
    level = build_facility()
    room = Room(level)
    player = Player(spawn=level.spawn)
    return player, PlayerController(player, room), room


def test_spawn_faces_minus_z_and_forward_moves_that_way():
    p, c, room = make()
    z0 = p.z
    for _ in range(30):
        c.move(1.0, 0.0, DT, room.get_obstacles())
    assert p.z < z0 - 1.0 and p.x == pytest.approx(0.0, abs=1e-6)


def test_diagonal_speed_is_normalised():
    p, c, room = make()
    x0, z0 = p.x, p.z
    c.move(1.0, 1.0, 1.0, [])
    assert math.hypot(p.x - x0, p.z - z0) == pytest.approx(PLAYER.walk_speed, rel=1e-6)


def test_crouching_is_slower():
    p, c, room = make()
    z0 = p.z
    c.toggle_crouch()
    c.move(1.0, 0.0, 1.0, [])
    assert abs(p.z - z0) == pytest.approx(PLAYER.crouch_speed, rel=1e-6)


def test_blocked_by_an_obstacle_but_slides_along_it():
    p, c, room = make()
    wall = AABB(-3.0, 3.0, 3.0, 3.2, 0.0, 3.0)  # in front of the player moving -Z
    for _ in range(120):
        c.move(1.0, 0.3, DT, [wall])  # pushing forward and strafing
    assert p.z >= 3.2 + PLAYER.collision_radius - 1e-6, "must not pass through the wall"
    assert p.x < -0.5, "must still slide sideways along the wall (A = -X when facing -Z)"


def test_cannot_leave_the_facility_boundary():
    p, c, room = make()
    p.yaw = 0.0  # +X
    for _ in range(600):
        c.move(1.0, 0.0, DT, [])
    assert p.x <= 5.6 - PLAYER.collision_radius + 1e-6  # staging bay walk region


def test_look_clamps_pitch_and_wraps_nothing_else():
    p, c, _ = make()
    c.look(0.0, -10_000.0)
    assert p.pitch == PLAYER.max_pitch
    c.look(0.0, 10_000.0)
    assert p.pitch == -PLAYER.max_pitch
    c.look(100.0, 0.0)
    assert p.yaw == pytest.approx(p.spawn[2] + 100.0 * PLAYER.mouse_sensitivity)


@pytest.mark.parametrize("dt", [1 / 240, 1 / 60, 0.1, 0.5, 5.0])
def test_eye_height_eases_toward_target_and_never_overshoots_for_any_dt(dt):
    """Regression for B4: the old linear smoother flipped below the floor for dt >= 0.2 s."""
    p, c, _ = make()
    c.toggle_crouch()
    p.update_height(dt, 0.0)
    assert PLAYER.eye_height_crouched <= p.y < PLAYER.eye_height_standing
    for _ in range(400):
        p.update_height(dt, 0.0)
    assert p.y == pytest.approx(PLAYER.eye_height_crouched, abs=1e-3)


def test_climbing_the_stairs_raises_the_floor_and_reaches_the_deck():
    p, c, room = make()
    door = room.doors["blast_door"]
    door.open_amount = door.travel
    p.x, p.z, p.yaw = 7.75, 20.0, 90.0  # in front of the right staircase, facing +Z
    for _ in range(240):
        c.move(1.0, 0.0, DT, room.get_obstacles())
    assert p.floor_y == pytest.approx(room.catwalk.height)
    assert p.elevation.on_catwalk and p.z > room.catwalk.stairs_z_end


def test_regression_cannot_enter_the_stairs_from_underneath_the_deck():
    """Original defect: the ramp had no underside, so a player under the catwalk could
    walk into its footprint and be lifted onto the deck."""
    p, c, room = make()
    p.x, p.z, p.yaw = (
        7.75,
        26.0,
        -90.0,
    )  # on the ground under the deck, facing -Z (toward the stairs' back)
    for _ in range(240):
        c.move(1.0, 0.0, DT, room.get_obstacles())
    assert p.floor_y == 0.0 and not p.elevation.on_catwalk
    assert p.z >= room.catwalk.stairs_z_end + PLAYER.collision_radius - 1e-6
