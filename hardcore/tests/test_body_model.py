"""Human body model: anatomy and reach envelope of a 1.80 m parkour athlete."""
import numpy as np
from harness import *  # noqa
from parkour.body import BodyModel, GrabResult
from parkour.ledge_rail import LedgeRail, RailType


def test_arm_segment_lengths_are_human():
    # Anthropometric data for a 1.80 m adult: upper arm 0.30-0.36 m, forearm 0.25-0.30 m
    assert 0.30 <= config.UPPER_ARM_LEN <= 0.36
    assert 0.25 <= config.FOREARM_LEN <= 0.30
    assert 0.62 <= config.ARM_REACH <= 0.75


def test_standing_overhead_reach_is_human():
    # Flat-footed overhead fingertip reach ~1.25-1.3 x height (2.25-2.35 m); the old code allowed 3.45 m
    assert 2.15 <= config.STAND_REACH <= 2.35
    assert config.STAND_REACH == config.SHOULDER_HEIGHT + config.MAX_GRAB_REACH


def test_shoulder_and_eye_heights_consistent():
    assert config.SHOULDER_HEIGHT < config.PLAYER_EYE_HEIGHT < config.PLAYER_HEIGHT
    assert 1.40 <= config.SHOULDER_HEIGHT <= 1.52


def test_hang_pose_uses_bent_arms_not_stretched():
    rail = LedgeRail([-2, 2.0, 0], [2, 2.0, 0], [0, 0, 1], RailType.HEAD_CREVASSE)
    center = BodyModel.hang_center(rail, 0.5)
    shoulder = BodyModel.shoulder_center(center, -rail.normal)
    grip = rail.get_point(0.5)
    d = float(np.linalg.norm(grip - shoulder))
    # Must be reachable with the hand hooked over the lip, and not a hyper-extended straight arm
    assert d <= config.UPPER_ARM_LEN + config.FOREARM_LEN + config.HAND_GRIP_LEN * 0.5
    assert d >= 0.40
    # Feet hang roughly 2 m below the hands (1.8 m body + bent arms)
    feet = BodyModel.feet_y(center)
    assert 1.85 <= grip[1] - feet <= 2.10


def test_hang_body_does_not_clip_into_wall():
    rail = LedgeRail([-2, 3.0, 0], [2, 3.0, 0], [0, 0, 1], RailType.HEAD_CREVASSE)
    center = BodyModel.hang_center(rail, 0.5)
    # wall face is at z = 0, body extends half-width toward it
    assert center[2] - config.PLAYER_WIDTH * 0.5 >= 0.0


def _wall_scene(height, depth=4.0):
    """Block whose south face (normal +Z) is always at z = -1."""
    return build_custom([(-2.0, 0.0, -1.0 - depth, 4.0, height, depth)])


def _top_rail(ls):
    return [r for r in ls.rails if r.rail_type == RailType.MANTLE_TOP and r.normal[2] > 0.5][0]


def _grab(ls, obs, rail, center, on_ground=True, vel=None):
    facing = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return BodyModel.evaluate_grab(rail, np.array(center, dtype=np.float32), facing, facing, on_ground, obs, vel)


def test_standing_grab_height_limits():
    near = (0.0, 0.9, -0.65)      # body against the wall face (z=-1)
    for h, expect in ((2.0, True), (2.2, True), (2.3, False), (2.6, False), (3.4, False)):
        ls, obs = _wall_scene(h)
        res = _grab(ls, obs, _top_rail(ls), near)
        assert (res is not None) == expect, f"top edge at {h} m from flat ground: grabbable={res is not None}"


def test_cannot_grab_through_or_behind_wall():
    ls, obs = _wall_scene(2.2)
    rail = _top_rail(ls)
    facing_away = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert BodyModel.evaluate_grab(rail, np.array([0.0, 0.9, -0.65]), facing_away, facing_away, True, obs) is None
    # Standing inside the footprint behind the face (other side of the wall)
    assert _grab(ls, obs, rail, (0.0, 0.9, -1.5)) is None


def test_low_top_edge_is_a_vault_not_a_hang():
    """Hanging from a 1.7 m ledge would put the feet 0.3 m into the floor (old bug)."""
    for h in (0.8, 1.2, 1.7):
        ls, obs = _wall_scene(h, depth=0.5)
        res = _grab(ls, obs, _top_rail(ls), (0.0, 0.9, -0.65))
        assert res is not None, f"{h} m wall must be vaultable"
        assert res.mode == GrabResult.MANTLE, f"{h} m wall must vault, got {res.mode}"


def test_free_hang_only_when_body_fits():
    ls, obs = _wall_scene(2.2)
    res = _grab(ls, obs, _top_rail(ls), (0.0, 0.9, -0.65))
    assert res.mode == GrabResult.HANG
    assert BodyModel.feet_y(res.hang_center) >= -1e-3
    assert BodyModel.pose_is_free(res.hang_center, obs)


def test_airborne_cannot_catch_hold_below_chest():
    ls, obs = _wall_scene(2.2)
    rail = _top_rail(ls)
    # body centre at 2.6 -> shoulders at ~3.15, the 2.2 m edge is a metre below the chest
    assert _grab(ls, obs, rail, (0.0, 2.6, -0.65), on_ground=False, vel=np.zeros(3)) is None
    # shoulders just above the edge: a chest-height catch is fine
    assert _grab(ls, obs, rail, (0.0, 1.9, -0.65), on_ground=False, vel=np.zeros(3)) is not None


def test_rising_fast_skips_face_height_holds():
    ls, obs = _wall_scene(5.4)
    head = [r for r in ls.rails if r.rail_type == RailType.HEAD_CREVASSE and r.normal[2] > 0.5][0]
    center = (0.0, 1.2, -0.6)   # eyes at 1.95, hold at 2.0
    assert _grab(ls, obs, head, center, on_ground=False, vel=np.array([0, 6.0, 0])) is None
    assert _grab(ls, obs, head, center, on_ground=False, vel=np.array([0, 0.5, 0])) is not None


def test_hold_gap_rule_from_lip():
    assert BodyModel.hold_above_reachable_from_lip(2.0, 3.8)
    assert BodyModel.hold_above_reachable_from_lip(3.8, 5.4)
    assert BodyModel.hold_above_reachable_from_lip(3.8, 6.0) == (6.0 - 3.8 <= config.CLIMB_HOLD_GAP_MAX)
    assert not BodyModel.hold_above_reachable_from_lip(2.0, 4.6)
    assert config.CLIMB_HOLD_GAP_MAX < config.STAND_REACH


def test_find_hold_above_picks_highest_reachable_on_same_face():
    ls, obs = _wall_scene(5.4)
    head = [r for r in ls.rails if r.rail_type == RailType.HEAD_CREVASSE and r.normal[2] > 0.5][0]
    jump = [r for r in ls.rails if r.rail_type == RailType.JUMP_CREVASSE and r.normal[2] > 0.5][0]
    top = _top_rail(ls)
    nxt = BodyModel.find_hold_above(ls.rails, head, 0.5, obs)
    assert nxt is not None and nxt[0] is jump, "from the 2.0 lip the 3.8 crevasse is the hold (top at 5.4 is out of reach)"
    nxt2 = BodyModel.find_hold_above(ls.rails, jump, 0.5, obs)
    assert nxt2 is not None and nxt2[0] is top


def test_every_playground_hang_pose_is_collision_free_or_rejected():
    ls, obs = build_playground()
    for rail in ls.rails:
        for s in (rail.s_margin(), 0.5, 1.0 - rail.s_margin()):
            center = BodyModel.hang_center(rail, s)
            if not BodyModel.pose_is_free(center, obs):
                # Then it must not be offered as a free hang from directly in front
                body = center.copy()
                body[1] = 0.9
                res = BodyModel.evaluate_grab(rail, body, -rail.normal, -rail.normal, True, obs)
                if res is None or res.mode == GrabResult.MANTLE or res.supported:
                    continue
                # ... unless the collision buffer found a clear spot a few cm along the same edge:
                # then the hang offered is collision-free and within the buffer of the asked spot
                assert BodyModel.pose_is_free(res.hang_center, obs, shrink=0.001), res
                assert abs(res.s - s) * rail.length <= config.EDGE_COLLISION_BUFFER + 1e-4, res
