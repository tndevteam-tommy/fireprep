"""Acrobatics: a flip on a second Space press in the air, a roll after a fall of more than 7 ft."""
import math

import numpy as np

from harness import *  # noqa
from parkour import acrobatics
from parkour.acrobatics import Spin
from core.gamepad_input import Haptics

SEVEN_FEET = 7 * 0.3048


def _world(*blocks):
    return build_custom(list(blocks))


def _flat_jump_and_flip(sim, hold_ticks=12, gap_ticks=2, extra=()):
    """Jump (Space held for a high leap), let go, press Space again."""
    sim.hold("SPACE")
    sim.step(hold_ticks)
    sim.release("SPACE")
    sim.step(gap_ticks)
    sim.hold(*extra)
    sim.tap("SPACE")


def _run_until_landed(sim, timeout=3.0):
    return sim.run(timeout, until=lambda s: s.player.on_ground and s.state != ParkourState.AIRBORNE)


def _upright_at_landing(sim):
    """Step until the feet are down; returns the spin angle left (0 = upright) at touch-down."""
    while not sim.player.on_ground or sim.state == ParkourState.AIRBORNE:
        sim.step()
        assert sim.time < 30.0
    spin = sim.fsm.spin
    return 0.0 if spin is None else min(spin.angle(), 360.0 - spin.angle())


# ------------------------------------------------------------------------ flips
def test_a_second_space_press_after_a_jump_flips_the_body_once_round():
    ls, obs = _world()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 10.0), yaw=0.0)
    sim.hold("W")
    sim.run(0.6)
    _flat_jump_and_flip(sim)
    assert sim.had_event("FLIP") and sim.fsm.spin is not None and sim.fsm.spin.kind == acrobatics.FRONT
    angles = []
    while sim.fsm.spin is not None and not sim.fsm.spin.is_roll:
        angles.append(sim.fsm.spin.angle())
        sim.step()
        assert sim.time < 5.0
    assert angles == sorted(angles), "the body turns one way, steadily"
    assert angles[-1] > 330.0 and sim.fsm.spin is None, "all the way round"
    assert max(b - a for a, b in zip(angles, angles[1:])) < 40.0, "no jumps in the rotation (smooth at 60 Hz)"
    # ... and it was over by the time the feet came down (timed to the airtime)
    assert sim.fsm.flip_count == 1
    _run_until_landed(sim)
    assert not sim.had_event("ROLL"), "a flat jump is no fall"


def test_the_flip_is_timed_to_finish_before_touch_down():
    ls, obs = _world((-3.0, 0.0, -3.0, 6.0, 3.0, 6.0, False))
    for hold, gap in ((12, 2), (15, 4), (6, 1)):
        sim = Sim(ls, obs, spawn=(0.0, 3.9, 2.0), yaw=180.0)      # on a 3 m block, facing off its edge
        sim.hold("W")
        sim.run(0.25)
        _flat_jump_and_flip(sim, hold, gap)
        sim.release("W")
        assert sim.had_event("FLIP")
        assert config.FLIP_MIN_TIME <= sim.fsm.spin.duration <= config.FLIP_MAX_TIME
        left = _upright_at_landing(sim)
        assert left < 5.0, f"still {left:.0f} deg from upright at touch-down (hold {hold}, gap {gap})"


def test_one_flip_per_jump_and_only_after_a_jump():
    ls, obs = _world((-3.0, 0.0, -3.0, 6.0, 4.0, 6.0, False))
    sim = Sim(ls, obs, spawn=(0.0, 4.9, 2.0), yaw=180.0)
    sim.hold("W")
    sim.run(0.2)
    _flat_jump_and_flip(sim)
    sim.step(20)
    sim.tap("SPACE")
    sim.step(4)
    sim.tap("SPACE")
    assert sim.fsm.flip_count == 1, "one flip per flight"
    # walking off an edge (no jump): Space is a reach, not a flip
    sim = Sim(ls, obs, spawn=(0.0, 4.9, 2.0), yaw=180.0)
    sim.hold("W")
    sim.run(0.8, until=lambda s: s.state == ParkourState.AIRBORNE)
    assert sim.state == ParkourState.AIRBORNE
    sim.step(8)
    sim.tap("SPACE")
    sim.step(3)
    assert sim.fsm.flip_count == 0 and not sim.had_event("FLIP")


def test_the_keys_held_choose_the_flip():
    ls, obs = _world()
    cases = [(("W",), acrobatics.FRONT), (("S",), acrobatics.BACK), (("A",), acrobatics.LEFT),
             (("D",), acrobatics.RIGHT), ((), acrobatics.BACK)]
    for keys, kind in cases:
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 10.0), yaw=0.0)
        sim.hold(*keys)
        _flat_jump_and_flip(sim)
        assert sim.fsm.spin is not None and sim.fsm.spin.kind == kind, (keys, sim.fsm.spin and sim.fsm.spin.kind)
        sim.release_all()
    assert acrobatics.flip_kind(True, False, False, False, 0.0) == acrobatics.FRONT
    assert acrobatics.flip_kind(False, False, False, False, 6.0) == acrobatics.FRONT, "running: front flip"


def test_no_flip_without_the_airtime_for_it():
    ls, obs = _world()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 10.0), yaw=0.0)
    sim.tap("SPACE")                                  # a low hop (~0.43 s in the air)
    sim.run(0.25)
    assert sim.state == ParkourState.AIRBORNE
    sim.tap("SPACE")
    assert not sim.had_event("FLIP"), "no time left to get round: no flip"


def test_a_press_near_a_hold_is_a_reach_not_a_flip():
    # a wall with crevasses straight ahead: jump at it and press again - the hands catch
    ls, obs = _world((-5.0, 0.0, -1.6, 10.0, 5.0, 1.0))
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.2), yaw=0.0)
    sim.look_at((0.0, 2.2, -1.6))
    sim.hold("W")
    _flat_jump_and_flip(sim, hold_ticks=10, gap_ticks=2)
    sim.hold("SPACE")
    sim.run(1.0, until=lambda s: s.state in (ParkourState.LEDGE_HANG, ParkourState.MANTLE_VAULT))
    assert not sim.had_event("FLIP")
    assert sim.had_event("GRAB") or sim.state == ParkourState.MANTLE_VAULT


def test_looking_at_a_hold_you_fly_toward_makes_the_press_a_reach():
    # a 3 m block, a wall 5 m on with holds at 2.0 / 3.8 / 4.4 m: looking at them, the press reaches
    ls, obs = _world((-3.0, 0.0, -3.0, 6.0, 3.0, 6.0, False), (-5.0, 0.0, -8.0, 10.0, 4.4, 1.0))
    sim = Sim(ls, obs, spawn=(0.0, 3.9, -2.5), yaw=0.0)
    sim.hold("W")
    sim.run(0.2)
    sim.look_at((0.0, 4.4, -7.0))
    _flat_jump_and_flip(sim, 12, 2)
    assert not sim.had_event("FLIP"), "the reticle is on a hold ahead: a reach"
    # the same jump looking up at the sky: a flip
    sim = Sim(ls, obs, spawn=(0.0, 3.9, -2.5), yaw=0.0, pitch=50.0)
    sim.hold("W")
    sim.run(0.2)
    _flat_jump_and_flip(sim, 12, 2)
    assert sim.had_event("FLIP")


def test_no_reaching_during_a_flip():
    ls, obs = _world((-3.0, 0.0, -3.0, 6.0, 3.0, 6.0, False), (-5.0, 0.0, -8.0, 10.0, 4.4, 1.0))
    sim = Sim(ls, obs, spawn=(0.0, 3.9, -2.5), yaw=0.0, pitch=50.0)
    sim.hold("W")
    sim.run(0.2)
    _flat_jump_and_flip(sim, 12, 2)
    assert sim.had_event("FLIP")
    sim.set_pitch(0.0)                              # (looking at the wall now makes no difference)
    sim.hold("SPACE", "E")                          # held through the flip: no catch while tumbling
    while sim.fsm.spin is not None and not sim.fsm.spin.is_roll:
        sim.step()
        assert sim.state not in (ParkourState.LEDGE_HANG, ParkourState.MANTLE_VAULT)


# ------------------------------------------------------------------------ rolls
def _drop_off(height, run=True, extra_x=0.0):
    """Run off the edge of a block `height` tall; returns the sim after touch-down + 2 s."""
    ls, obs = _world((-3.0, 0.0, -20.0, 6.0, height, 6.0, False))
    sim = Sim(ls, obs, spawn=(extra_x, height + 0.9, -18.0), yaw=180.0)
    if run:
        sim.hold("CAPS")
    sim.hold("W")
    trace = []
    for _ in range(int(3.5 / DT)):
        sim.step()
        trace.append((sim.time, sim.state, float(np.hypot(sim.vel[0], sim.vel[2])), sim.pos.copy()))
    return sim, trace


def test_a_fall_of_more_than_seven_feet_ends_in_a_roll_that_keeps_the_momentum():
    sim, trace = _drop_off(4.0)
    assert sim.had_event("ROLL") and sim.had_event("ROLL_DONE")
    assert sim.fsm.last_drop > SEVEN_FEET
    i_land = next(i for i, t in enumerate(trace) if t[1] == ParkourState.ROLL)
    landing_speed = trace[i_land - 1][2]
    rolling = [t for t in trace if t[1] == ParkourState.ROLL]
    assert 0.6 <= len(rolling) * DT <= 0.8, "a roll takes about three quarters of a second"
    assert min(t[2] for t in rolling) >= 0.85 * landing_speed, "the roll carries the speed through"
    # it goes on into the run, with no stop
    after = [t for t in trace[i_land + len(rolling):]]
    assert after[0][1] in (ParkourState.SPRINT, ParkourState.GROUND)
    assert min(t[2] for t in after[:30]) >= 0.85 * landing_speed, "straight on into the run"
    assert any(t[1] == ParkourState.SPRINT for t in after)
    # travelled along the run (+z here), never backwards
    zs = [t[3][2] for t in trace[i_land:i_land + len(rolling) + 30]]
    assert all(b >= a - 1e-4 for a, b in zip(zs, zs[1:]))


def test_lower_falls_do_not_roll_and_the_threshold_is_seven_feet():
    for height, rolls in ((1.5, False), (2.0, False), (2.4, True)):
        sim, _ = _drop_off(height, run=False)
        assert sim.had_event("ROLL") == rolls, (height, sim.fsm.last_drop)
    assert abs(config.ROLL_MIN_DROP - SEVEN_FEET) < 1e-3


def test_the_roll_turns_the_body_over_and_ends_upright():
    ls, obs = _world((-3.0, 0.0, -20.0, 6.0, 4.0, 6.0, False))
    sim = Sim(ls, obs, spawn=(0.0, 4.9, -18.0), yaw=180.0)
    sim.hold("W")
    sim.run(3.0, until=lambda s: s.had_event("ROLL"))
    spin = sim.fsm.spin
    assert spin is not None and spin.is_roll and not spin.backward
    angles, crouch, eye_y = [], [], []
    while sim.fsm.spin is spin:
        angles.append(spin.angle())
        crouch.append(spin.crouch())
        T = acrobatics.body_transform(spin, sim.pos)
        eye_y.append(float(acrobatics.apply(T, sim.player.camera_pos)[1]) - sim.feet_y)
        sim.step()
    assert angles[0] < 5.0 and angles[-1] > 355.0
    assert max(crouch) > 0.7 and crouch[-1] < 0.05, "sinks into the roll and stands up out of it"
    assert min(eye_y) > 0.12, "the head never goes through the ground"
    assert sim.state in (ParkourState.GROUND, ParkourState.SPRINT)


def test_letting_go_high_up_rolls_backward_away_from_the_wall():
    ls, obs = _world((-5.0, 0.0, -1.6, 10.0, 6.0, 1.0))
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.2), yaw=0.0)
    # hang from the wall's top edge (6 m) and let go
    top = [r for r in ls.rails if abs(r.height - 6.0) < 1e-3 and r.normal[2] > 0.9][0]
    s = top.clamp_s(top.project_s(np.array([0.0, 6.0, -0.6], dtype=np.float32)))
    sim.teleport(BodyModel.hang_center(top, s))
    sim.fsm.state = ParkourState.LEDGE_HANG
    sim.fsm.active_rail, sim.fsm.rail_s = top, s
    sim.fsm.grip_point = top.get_point(s)
    sim.fsm.settle_time = config.GRAB_SETTLE_TIME
    sim.player.on_ground = False
    sim.step()
    sim.tap("CTRL")
    sim.run(3.0, until=lambda s: s.had_event("ROLL"))
    assert sim.had_event("ROLL") and sim.fsm.spin is not None and sim.fsm.spin.backward
    z0 = float(sim.pos[2])
    sim.run(1.0)
    assert float(sim.pos[2]) > z0 + 0.8, "rolled away from the wall"
    assert BodyModel.pose_is_free(sim.pos, obs)


def test_a_roll_never_carries_you_off_what_you_landed_on():
    # a 10 m tower, and 5 m out a 3 m high platform only 2.6 m long: land on it and roll
    ls, obs = _world((-3.0, 0.0, -20.0, 6.0, 10.0, 6.0, False), (-3.0, 0.0, -10.0, 6.0, 3.0, 2.6, False))
    sim = Sim(ls, obs, spawn=(0.0, 10.9, -16.0), yaw=180.0)
    sim.hold("W")
    sim.run(4.0, until=lambda s: s.had_event("ROLL"))
    assert sim.had_event("ROLL") and sim.feet_y > 2.9, "landed on the platform"
    sim.release_all()
    sim.run(1.5)
    assert sim.feet_y > 2.9 and sim.player.on_ground, "still on the platform after the roll"


def test_a_flip_off_a_height_comes_round_then_rolls():
    ls, obs = _world((-3.0, 0.0, -20.0, 6.0, 5.0, 6.0, False))
    sim = Sim(ls, obs, spawn=(0.0, 5.9, -18.0), yaw=180.0)
    sim.hold("W")
    sim.run(0.3)
    _flat_jump_and_flip(sim, 10, 2)
    assert sim.had_event("FLIP")
    kinds = []
    for _ in range(int(3.0 / DT)):
        sim.step()
        if sim.fsm.spin is not None and (not kinds or kinds[-1] != sim.fsm.spin.kind):
            kinds.append(sim.fsm.spin.kind)
    assert sim.had_event("ROLL") and kinds[-1] == acrobatics.ROLL and kinds[0] == acrobatics.FRONT


def test_you_can_spring_out_of_the_end_of_a_roll():
    ls, obs = _world((-3.0, 0.0, -20.0, 6.0, 4.0, 6.0, False))
    sim = Sim(ls, obs, spawn=(0.0, 4.9, -18.0), yaw=180.0)
    sim.hold("CAPS", "W")
    sim.run(3.0, until=lambda s: s.had_event("ROLL"))
    t_roll = sim.time
    sim.run(config.ROLL_TIME * 0.1)
    sim.tap("SPACE")
    sim.run(config.INPUT_BUFFER_TIME + 0.02)      # (let that press run out of the input buffer)
    assert not sim.had_event("JUMP"), "too early in the roll"
    assert sim.time - t_roll < config.ROLL_TIME * config.ROLL_JUMP_FROM
    sim.run(config.ROLL_TIME * config.ROLL_JUMP_FROM - (sim.time - t_roll) + 0.02)
    n = len(sim.events)
    sim.tap("SPACE")
    sim.step(2)
    assert any(e == "JUMP" for _, e in sim.events[n:]) and sim.state == ParkourState.AIRBORNE


def test_a_jump_pressed_just_before_the_spring_out_window_is_buffered_into_it():
    ls, obs = _world((-3.0, 0.0, -20.0, 6.0, 4.0, 6.0, False))
    sim = Sim(ls, obs, spawn=(0.0, 4.9, -18.0), yaw=180.0)
    sim.hold("CAPS", "W")
    sim.run(3.0, until=lambda s: s.had_event("ROLL"))
    t_roll = sim.time
    sim.run(config.ROLL_TIME * config.ROLL_JUMP_FROM - 0.08)
    sim.tap("SPACE")                              # a little early ...
    assert sim.run(0.2, until=lambda s: s.had_event("JUMP")), "... still springs out as soon as it may"
    assert sim.time - t_roll >= config.ROLL_TIME * config.ROLL_JUMP_FROM - 0.02


# ------------------------------------------------------------------------ body
def test_body_stays_anatomical_through_flips_and_rolls():
    ls, obs = _world((-3.0, 0.0, -20.0, 6.0, 4.0, 6.0, False))
    sim = Sim(ls, obs, spawn=(0.0, 4.9, -18.0), yaw=180.0, with_arms=True)
    sim.hold("W")
    sim.run(0.3)
    _flat_jump_and_flip(sim, 10, 2)
    tucked = 0
    for _ in range(int(2.5 / DT)):
        sim.step()
        L = sim.arm_rig.last
        for side in ("left", "right"):
            assert abs(np.linalg.norm(L[side + "_elbow"] - L[side + "_shoulder"]) - config.UPPER_ARM_LEN) < 1e-3
            assert abs(np.linalg.norm(L[side + "_wrist"] - L[side + "_elbow"]) - config.FOREARM_LEN) < 1e-3
            assert config.ELBOW_MIN_ANGLE_DEG - 0.01 <= L[side + "_elbow_angle"] <= config.ELBOW_MAX_ANGLE_DEG + 0.01
        assert max(L["wrist_angles"]) <= config.WRIST_MAX_ANGLE_DEG + 1e-3
        G = sim.arm_rig.legs.last
        for i in range(2):
            assert abs(np.linalg.norm(G["knees"][i] - G["hips"][i]) - config.THIGH_LEN) < 1e-3
            assert abs(np.linalg.norm(G["ankles"][i] - G["knees"][i]) - config.SHIN_LEN) < 1e-3
            assert config.KNEE_MIN_ANGLE_DEG - 0.01 <= G["knee_angles"][i] <= config.KNEE_MAX_ANGLE_DEG + 0.01
        spin = sim.fsm.spin
        if spin is not None and spin.tuck() > 0.9:
            tucked += 1
            # knees up toward the chest, hands at the shins
            hip_y = float(G["hips"][0][1])
            assert min(float(G["knees"][0][1]), float(G["knees"][1][1])) > hip_y - 0.15
            for side, i in (("left", 0), ("right", 1)):
                assert np.linalg.norm(L[side + "_wrist"] - G["knees"][i]) < 0.3
    assert tucked >= 10 and sim.had_event("ROLL")


def test_spin_transform_is_rigid_and_round_trips():
    h = np.array([0.6, 0.0, -0.8], dtype=np.float32)
    c = np.array([1.0, 0.9, 2.0], dtype=np.float32)
    for kind in acrobatics.KINDS:
        for ang in (0.0, 90.0, 180.0, 270.0, 360.0):
            T = acrobatics.transform_for(kind, ang, 0.0, h, c)
            R = T[:3, :3]
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-5) and np.linalg.det(R) > 0.999
            pivot = Spin(kind, 1.0, h).pivot(c)
            assert np.allclose(acrobatics.apply(T, pivot), pivot, atol=1e-5), "turns about the pivot"
        assert np.allclose(acrobatics.transform_for(kind, 360.0, 0.0, h, c), np.eye(4), atol=1e-5)
    # a front flip takes the eyes forward and down first; a back flip up and back
    eye = c + np.array([0.0, 0.75, 0.0], dtype=np.float32)
    f = acrobatics.apply(acrobatics.transform_for(acrobatics.FRONT, 60.0, 0.0, h, c), eye)
    b = acrobatics.apply(acrobatics.transform_for(acrobatics.BACK, 60.0, 0.0, h, c), eye)
    assert float(np.dot(f - eye, h)) > 0.2 and float(np.dot(b - eye, h)) < -0.2
    # a right side flip takes the head to the right
    r = acrobatics.apply(acrobatics.transform_for(acrobatics.RIGHT, 60.0, 0.0, h, c), eye)
    right = np.cross(h, [0.0, 1.0, 0.0])
    assert float(np.dot(r - eye, right)) > 0.2
    assert abs(acrobatics.time_to_land(2.0, 0.0, 0.0) - math.sqrt(2 * 2.0 / -config.GRAVITY)) < 1e-6


def test_pads_feel_the_flip_and_the_roll():
    h = Haptics()
    assert h.on_tick("FLIP", False, False, 0.0) is not None
    assert h.on_tick("ROLL", False, True, -9.0) is not None


def test_a_climb_ends_the_flight_so_running_off_an_edge_later_is_a_coyote_jump_not_a_flip():
    """Regression: the 'this flight began with a jump' flag survived a catch and top-out, so after
    jump -> catch -> mantle, a Space pressed just after running off the far edge of the roof flipped
    (and fell) instead of taking the coyote jump."""
    ls, obs = _world((-3.0, 0.0, -14.0, 6.0, 2.2, 10.0, True))      # a 2.2 m block, 10 m deep
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 6.0), yaw=0.0)
    sim.hold("W", "SHIFT")
    sim.run(3.0, until=lambda s: s.pos[2] < -4.0 + 0.3 + 1.6)
    sim.hold("SPACE")
    assert sim.run(3.0, until=lambda s: s.had_event("MANTLE_DONE")), "jumped, caught the edge and topped out"
    sim.release("SPACE")
    assert sim.run(3.0, until=lambda s: not s.player.on_ground and s.state == ParkourState.AIRBORNE), "ran off the far edge"
    assert sim.feet_y > 1.9
    n = len(sim.events)
    sim.tap("SPACE")                              # within coyote time
    sim.step(2)
    after = [e for _, e in sim.events[n:]]
    assert "JUMP" in after and "FLIP" not in after, after
