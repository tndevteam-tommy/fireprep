import math

import pytest

from unprepared.config import PlayerSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import Player
from unprepared.systems import MoveIntent, apply_movement

S = PlayerSettings()


@pytest.fixture
def player():
    return Player(Vector3D(0.0, 0.0, 0.0))


def test_idle(player):
    r = apply_movement(player, MoveIntent(), 0.5, S)
    assert not r.is_moving and not r.is_sprinting
    assert player.position == Vector3D(0.0, 0.0, 0.0)


def test_forward_moves_along_negative_z(player):
    r = apply_movement(player, MoveIntent(forward=True), 0.5, S)
    assert r.is_moving and not r.is_sprinting
    assert player.position.x == pytest.approx(0.0) and player.position.z == pytest.approx(-2.0)
    assert player.position.y == 0.0


def test_backward_and_strafe(player):
    apply_movement(player, MoveIntent(backward=True), 0.25, S)
    assert player.position.z == pytest.approx(1.0)
    apply_movement(player, MoveIntent(right=True), 0.25, S)
    assert player.position.x == pytest.approx(1.0)
    apply_movement(player, MoveIntent(left=True), 0.5, S)
    assert player.position.x == pytest.approx(-1.0)


def test_opposite_keys_cancel(player):
    apply_movement(player, MoveIntent(forward=True, backward=True), 1.0, S)
    assert player.position.z == pytest.approx(0.0)


def test_sprint_multiplier(player):
    r = apply_movement(player, MoveIntent(forward=True, sprint=True), 1.0, S)
    assert r.is_sprinting and player.position.z == pytest.approx(-7.2)


def test_sprint_flag_requires_movement(player):
    r = apply_movement(player, MoveIntent(sprint=True), 1.0, S)
    assert not r.is_sprinting and not r.is_moving


def test_movement_respects_colliders(player):
    from unprepared.core.math3d import BoundingBox3D
    wall = BoundingBox3D(Vector3D(-5, 0, -2.0), Vector3D(5, 4, -1.5))
    r = apply_movement(player, MoveIntent(forward=True), 1.0, S, colliders=[wall])
    assert r.is_moving
    # a 4-unit step is sub-stepped and stops short of the wall face at z=-1.5 (half-width 0.3)
    assert -1.2 - 1e-9 <= player.position.z <= -1.0
    before = player.position.z
    apply_movement(player, MoveIntent(forward=True), 0.05, S, colliders=[wall])
    assert player.position.z == before                       # stays blocked


def test_follows_yaw(player):
    player.rotation_y = 90.0
    apply_movement(player, MoveIntent(forward=True), 1.0, S)
    assert player.position.x == pytest.approx(4.0) and abs(player.position.z) < 1e-9


def test_diagonal_is_faster_than_single_axis(player):
    # No normalisation of diagonal movement, exactly like the original.
    apply_movement(player, MoveIntent(forward=True, right=True), 1.0, S)
    assert player.position.distance_to(Vector3D(0, 0, 0)) == pytest.approx(4.0 * math.sqrt(2))


def test_move_intent_any():
    assert not MoveIntent(sprint=True).any
    assert MoveIntent(left=True).any
