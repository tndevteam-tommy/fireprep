"""Swept AABB collision: no tunnelling, correct contact reporting, kill plane."""
import numpy as np
from harness import *  # noqa
from physics.collision import CollisionSystem
from physics.aabb import AABB


def test_no_tunnelling_through_thin_wall_at_sprint_speed():
    ls, obs = build_custom([(-3.0, 0.0, -6.0, 6.0, 3.0, 0.3)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 6.0))
    sim.hold("W", "SHIFT")
    sim.run(4.0)
    assert sim.pos[2] > -5.7, "player passed through a 0.3 m wall"


def test_wall_contact_reports_normal_and_approach_speed():
    ls, obs = build_custom([(-3.0, 0.0, -6.0, 6.0, 3.0, 1.0)])
    pos = np.array([0.0, 0.9, -4.65], dtype=np.float32)
    vel = np.array([0.0, 0.0, -6.0], dtype=np.float32)
    new_pos, new_vel, contact = CollisionSystem.resolve_movement(pos, BodyModel.SIZE, vel, 1.0 / 60.0, obs)
    assert contact.wall_normal is not None and contact.wall_normal[2] > 0.5
    assert abs(contact.approach_speed - 6.0) < 1e-5
    assert new_vel[2] == 0.0
    assert new_pos[2] >= -5.0 + 0.3


def test_landing_reports_ground():
    ls, obs = build_custom([])
    pos = np.array([0.0, 1.2, 0.0], dtype=np.float32)
    vel = np.array([0.0, -5.0, 0.0], dtype=np.float32)
    for _ in range(30):
        pos, vel, contact = CollisionSystem.resolve_movement(pos, BodyModel.SIZE, vel, 1.0 / 60.0, obs)
        vel[1] += config.GRAVITY / 60.0
    assert contact.on_ground
    assert abs(BodyModel.feet_y(pos)) < 1e-3


def test_ground_height_probe():
    ls, obs = build_custom([(-1.0, 0.0, -1.0, 2.0, 1.5, 2.0)])
    g = CollisionSystem.ground_height_below(np.array([0.0, 3.0, 0.0]), BodyModel.SIZE, obs)
    assert abs(g - 1.5) < 1e-6
    g2 = CollisionSystem.ground_height_below(np.array([5.0, 3.0, 0.0]), BodyModel.SIZE, obs)
    assert abs(g2 - 0.0) < 1e-6


def test_kill_plane_respawns_player():
    ls, obs = build_custom([], floor=False)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0))
    sim.player.on_ground = False
    assert sim.run(6.0, until=lambda s: s.player.respawn_count >= 1)
    assert np.allclose(sim.pos, sim.player.spawn_pos)


def test_aabb_intersection_is_strict():
    a = AABB([0, 0, 0], [1, 1, 1])
    b = AABB([1, 0, 0], [2, 1, 1])
    assert not a.intersects(b), "touching boxes are not overlapping"


def test_stairs_and_curbs_are_walked_over_without_a_vault():
    # a 0.15 m curb, then a staircase of 0.2 m risers up to 1.0 m
    blocks = [(-3.0, 0.0, -4.0, 6.0, 0.15, 1.0)]
    for i in range(5):
        blocks.append((-3.0, 0.0, -8.0 - i * 0.4, 6.0, 0.2 * (i + 1), 0.4))
    ls, obs = build_custom(blocks)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -1.0))
    sim.hold("W")
    assert sim.run(4.0, until=lambda s: s.feet_y > 0.95), "did not walk up the stairs"
    assert not sim.had_event("VAULT"), "steps must not trigger vault animations"
    assert ParkourState.MANTLE_VAULT not in sim.states
    assert sim.max_step < 0.5, "stepping up must be a small lift, not a teleport"


def test_step_offset_does_not_lift_over_tall_blocks_or_while_jumping():
    ls, obs = build_custom([(-3.0, 0.0, -4.0, 6.0, 0.6, 1.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -1.0))
    sim.hold("W")
    sim.run(1.5)
    assert sim.feet_y < 0.05 or sim.had_event("VAULT"), "0.6 m is not a step"
