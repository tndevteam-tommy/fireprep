"""Arm rig: rigid bones, human elbow limits, no stretching, hands on the actual grip."""
import numpy as np
from harness import *  # noqa
from animation.two_bone_ik import solve_two_bone_ik, reach_limits
from animation.arm_rig import FirstPersonArmRig
from animation.skeleton import BoneID
from animation.finger_solver import HandPoseType

L1, L2 = config.UPPER_ARM_LEN, config.FOREARM_LEN
RNG = np.random.default_rng(7)


def _random_targets(n=400, scale=2.5):
    return RNG.uniform(-scale, scale, size=(n, 3)).astype(np.float32)


def test_ik_preserves_bone_lengths_for_any_target():
    shoulder = np.array([0.2, 1.45, 0.0], dtype=np.float32)
    pole = np.array([0.4, -1.0, -0.3], dtype=np.float32)
    for t in _random_targets():
        res = solve_two_bone_ik(shoulder, shoulder + t, L1, L2, pole)
        assert abs(np.linalg.norm(res.elbow - shoulder) - L1) < 1e-4
        assert abs(np.linalg.norm(res.wrist - res.elbow) - L2) < 1e-4


def test_ik_never_exceeds_arm_length_or_hyperflexes():
    shoulder = np.zeros(3, dtype=np.float32)
    pole = np.array([0.4, -1.0, -0.3], dtype=np.float32)
    d_min, d_max = reach_limits(L1, L2)
    for t in _random_targets():
        res = solve_two_bone_ik(shoulder, t, L1, L2, pole)
        d = float(np.linalg.norm(res.wrist - shoulder))
        assert d <= d_max + 1e-4, "arm stretched beyond its length"
        assert d >= d_min - 1e-4, "elbow folded past the human flexion limit"
        assert config.ELBOW_MIN_ANGLE_DEG - 1e-3 <= res.elbow_angle_deg <= config.ELBOW_MAX_ANGLE_DEG + 1e-3


def test_ik_out_of_reach_target_pulls_wrist_back_along_line():
    shoulder = np.zeros(3, dtype=np.float32)
    target = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    res = solve_two_bone_ik(shoulder, target, L1, L2, np.array([0.3, -1.0, -0.3]))
    assert res.clamped
    assert abs(res.wrist[1] - np.linalg.norm(res.wrist)) < 1e-4   # still straight up
    assert np.linalg.norm(res.wrist) < L1 + L2


def test_ik_elbow_sits_on_pole_side():
    shoulder = np.zeros(3, dtype=np.float32)
    pole = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    target = np.array([0.0, 0.3, -0.4], dtype=np.float32)
    res = solve_two_bone_ik(shoulder, target, L1, L2, pole)
    mid = (shoulder + res.wrist) * 0.5
    assert float(np.dot(res.elbow - mid, pole)) > 0.0, "elbow flipped to the wrong side of the arm"


def test_ik_is_independent_of_world_position():
    """Regression: the pole used to be a world position, so elbows changed with where you stood."""
    pole = np.array([0.4, -1.0, -0.3], dtype=np.float32)
    base_sh = np.array([0.2, 1.45, 0.0], dtype=np.float32)
    base_tg = np.array([0.3, 2.0, -0.4], dtype=np.float32)
    ref = solve_two_bone_ik(base_sh, base_tg, L1, L2, pole).elbow - base_sh
    for off in ([20, 0, 20], [-25, 0, 35], [3, 7, -40]):
        off = np.array(off, dtype=np.float32)
        res = solve_two_bone_ik(base_sh + off, base_tg + off, L1, L2, pole)
        assert np.allclose(res.elbow - (base_sh + off), ref, atol=1e-4)


def test_hand_frame_is_right_handed_orthonormal():
    m = FirstPersonArmRig._hand_frame([0, 0, -1], [0, -1, 0])
    R = m[0:3, 0:3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)
    assert np.linalg.det(R) > 0.99
    assert np.allclose(R[:, 1], [0, 0, -1], atol=1e-6)     # fingers
    assert np.allclose(R[:, 2], [0, -1, 0], atol=1e-6)     # palm


def _hang_sim(spawn_x, wall=(-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)):
    ls, obs = build_custom([wall])
    sim = Sim(ls, obs, spawn=(spawn_x, 0.9, 1.0), yaw=0.0, with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.3)
    return sim


def test_hands_grip_where_the_player_actually_hangs():
    """Regression: hands were drawn at the rail midpoint, metres away from the body on long rails."""
    sim = _hang_sim(spawn_x=-8.0)
    L = sim.arm_rig.last
    grip = sim.fsm.grip_point
    assert abs(sim.fsm.rail_s - 0.5) > 0.3, "test needs an off-centre grip"
    for side in ("left", "right"):
        wrist = L[side + "_wrist"]
        assert np.linalg.norm(wrist - grip) < 0.35, f"{side} hand {np.linalg.norm(wrist - grip):.2f} m from the grip"
        assert np.linalg.norm(wrist - sim.player.eye_pos) < 0.8
    # Hands are spaced about shoulder width apart on the rail
    spacing = np.linalg.norm(L["left_wrist"] - L["right_wrist"])
    assert 0.30 <= spacing <= 0.50


def test_hang_arms_bent_and_hands_reach_target():
    sim = _hang_sim(spawn_x=0.0)
    L = sim.arm_rig.last
    assert sim.arm_rig.pose_type == HandPoseType.CRIMP
    for side in ("left", "right"):
        assert np.linalg.norm(L[side + "_wrist"] - L[side + "_target"]) < 0.01, "hands must reach the hold"
        assert 90.0 <= L[side + "_elbow_angle"] <= 170.0, "active hang: elbows bent, not locked"
        # elbows flare outward (not crossing to the other side of the body) and stay off the wall
        sh, el = L[side + "_shoulder"], L[side + "_elbow"]
        lateral = float(np.dot(el - sh, L["right"]))
        assert (lateral < 0) if side == "left" else (lateral > 0)
        assert el[2] > -1.0 + 0.02, "elbow inside the wall"


def test_hang_hand_orientation_hooks_over_the_lip():
    """A hang is a finger hook: the hand stays nearly in line with the forearm (slight wrist
    extension toward the wall), the palm faces the wall and the curled fingers end up over / in
    the edge, where the wall hides them."""
    sim = _hang_sim(spawn_x=0.0)
    pal = sim.arm_rig.get_palette()
    rail = sim.fsm.active_rail
    L = sim.arm_rig.last
    assert max(L["wrist_angles"]) <= 30.0, "hanging on a thin edge should bend the wrist only slightly"
    for wrist_bone, tip_bone, tip_len in ((3, 9, 0.018), (22, 28, 0.018)):
        R = pal[wrist_bone][0:3, 0:3]
        palm = R[:, 2]
        assert float(np.dot(palm, -rail.normal)) > 0.7, "palm must face the wall"
        assert R[1, 1] > 0.8, "metacarpals point up the wall, not horizontally over it"
        tip = (pal[tip_bone] @ np.array([0.0, tip_len, 0.0, 1.0], dtype=np.float32))[:3]
        assert rail.owner.bounds.contains_point(tip), "curled fingertips go into the edge (hidden by the wall)"


def test_arm_rig_never_stretches_during_full_climb():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.2), yaw=0.0, with_arms=True)
    sim.hold("W", "E")
    seen = set()
    while sim.time < 6.0:
        sim.top_out_if_needed()
        sim.step()
        seen.add(sim.state)
        L = sim.arm_rig.last
        for side in ("left", "right"):
            sh, el, wr = L[side + "_shoulder"], L[side + "_elbow"], L[side + "_wrist"]
            assert abs(np.linalg.norm(el - sh) - L1) < 1e-3
            assert abs(np.linalg.norm(wr - el) - L2) < 1e-3
            assert np.linalg.norm(wr - sh) <= L1 + L2 + 1e-3
            assert config.ELBOW_MIN_ANGLE_DEG - 0.01 <= L[side + "_elbow_angle"] <= config.ELBOW_MAX_ANGLE_DEG + 0.01
    assert {ParkourState.LEDGE_HANG, ParkourState.CLIMB_UP, ParkourState.MANTLE_VAULT} <= seen


def test_pre_reach_never_extends_past_arm_length():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 2.2, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 4.0), yaw=0.0, with_arms=True)
    sim.hold("W")
    sim.run(3.0, until=lambda s: s.pos[2] < -1.0 + 0.3 + 1.3)
    sim.hold("SPACE")               # jump at the wall without pressing E: hands should still anticipate
    reached_pre = False
    t0 = sim.time
    while sim.time < t0 + 1.5 and (not sim.player.on_ground or sim.time < t0 + 0.1):
        sim.step()
        L = sim.arm_rig.last
        if sim.arm_rig.pose_type == HandPoseType.PRE_REACH:
            reached_pre = True
            for side in ("left", "right"):
                assert np.linalg.norm(L[side + "_wrist"] - L[side + "_shoulder"]) <= L1 + L2 + 1e-3
                # the *target* itself is reach-limited, so the arm never visibly stretches toward the hold
                assert np.linalg.norm(L[side + "_target"] - L[side + "_shoulder"]) <= L1 + L2 + 0.1
    assert reached_pre, "airborne near a ledge the hands should anticipate the grab"


def test_standing_still_the_arms_hang_at_the_sides_and_do_not_move():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), yaw=0.0, with_arms=True)
    sim.run(1.0)
    L0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sim.arm_rig.last.items()}
    right = sim.player.right_h
    fwd = sim.player.forward_h
    for side, sign in (("left", -1.0), ("right", 1.0)):
        sh, el, wr = L0[side + "_shoulder"], L0[side + "_elbow"], L0[side + "_wrist"]
        rel = wr - sh
        assert rel[1] < -0.50, "hands hang well below the shoulders"
        assert abs(float(np.dot(rel, fwd))) < 0.12, "beside the body, not out in front"
        assert float(np.dot(rel, right)) * sign > 0.0, "outside the thigh, on its own side"
        assert 150.0 <= L0[side + "_elbow_angle"] <= 175.0, "a slight bend at the elbow, never locked"
        assert float(np.dot(el - sh, fwd)) < 0.02, "the elbow points back, not forward"
        palm = L0["hand_frames"][0 if side == "left" else 1][0:3, 2]
        assert float(np.dot(palm, -right * sign)) > 0.7, "palms face the thighs"
        fingers = L0["hand_frames"][0 if side == "left" else 1][0:3, 1]
        assert fingers[1] < -0.8, "fingers point down"
    sim.run(1.0)
    L1 = sim.arm_rig.last
    for side in ("left", "right"):
        assert float(np.linalg.norm(L1[side + "_wrist"] - L0[side + "_wrist"])) < 1e-3, "arms are heavy: no idle sway"
    # while sprinting the hands are up in front and pump into view
    sim.hold("W", "SHIFT")
    sim.run(config.RUN_BUILD_TIME + 0.5)
    half_fov_v = np.radians(config.BASE_FOV * 0.5)
    seen = {"left": False, "right": False}
    t0 = sim.time
    while sim.time < t0 + 1.0:
        sim.step()
        L2 = sim.arm_rig.last
        cam = sim.player.camera_pos
        fwd3 = sim.player.forward_3d
        for side in ("left", "right"):
            rel = L2[side + "_wrist"] - cam
            depth = float(np.dot(rel, fwd3))
            if depth > 0.2 and np.arctan2(-float(rel[1]), depth) < half_fov_v:
                seen[side] = True
    assert all(seen.values()), "the pumping fists come up into the bottom of the view"


def test_arms_enter_the_view_from_its_lower_corners():
    """The camera sits ahead of the shoulder line and the shoulders are a real shoulder-width
    apart, so the arms come into the picture wide and low, as your own do."""
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), yaw=0.0, with_arms=True)
    sim.hold("W", "SHIFT")
    sim.run(2.0)
    L = sim.arm_rig.last
    cam = sim.player.camera_pos
    right = sim.player.right_h
    fwd = sim.player.forward_h
    for side, sign in (("left", -1.0), ("right", 1.0)):
        sh = L[side + "_shoulder"]
        rel = sh - cam
        assert float(np.dot(rel, fwd)) < 0.0, "shoulders are behind the camera"
        assert abs(float(np.dot(rel, right))) >= 0.20, "a full shoulder-width apart"
        el = L[side + "_elbow"] - cam
        depth = max(float(np.dot(el, fwd)), 1e-3)
        lateral = float(np.dot(el, right)) * sign
        assert lateral > 0.15, "elbows stay out to the sides"
        assert np.degrees(np.arctan2(lateral, depth)) > 30.0, "arms come in from the sides of the view, not its middle"


def test_climb_up_hands_stay_on_lip_then_reach_next_hold():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0), yaw=0.0, with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.CLIMB_UP)
    lip = sim.fsm.grip_point.copy()
    nxt_rail, nxt_s = sim.fsm.climb_next
    nxt = nxt_rail.get_point(nxt_s)
    pull_frac = config.CLIMB_PULL_TIME / config.CLIMB_TOTAL_TIME
    while sim.state == ParkourState.CLIMB_UP:
        sim.step()
        L = sim.arm_rig.last
        if sim.fsm.climb_phase < pull_frac * 0.9:
            for side in ("left", "right"):
                assert abs(L[side + "_wrist"][1] - lip[1]) < 0.13, "hands must stay on the hold during the pull-up"
        wall_plane = float(np.dot(L["left_target"], nxt_rail.normal))
        # hands stay at the wall while the body stands up (a hopping hand comes out just enough to clear the edge)
        assert -0.05 <= wall_plane - float(np.dot(lip, nxt_rail.normal)) <= config.WRIST_OUT_CRIMP + config.HOP_OUT + 0.03
    L = sim.arm_rig.last
    hooked = [abs(L[side + "_target"][1] - (nxt[1] - config.WRIST_DROP_CRIMP)) < 0.05 for side in ("left", "right")]
    assert any(hooked), "the lead hand has caught the next hold as the body arrives"
    assert sim.state == ParkourState.LEDGE_HANG
    sim.run(config.HANDOVER_TIME + 0.1)
    L = sim.arm_rig.last
    for side in ("left", "right"):
        assert abs(L[side + "_target"][1] - (nxt[1] - config.WRIST_DROP_CRIMP)) < 0.05, "both hands end hooked on the next hold"
    assert L["hand_rails"] == (nxt_rail, nxt_rail)


# ---------------------------------------------------------------- wrist & mesh
def test_wrist_never_bends_past_human_range_during_climbs_and_traverses():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.2), yaw=0.0, with_arms=True)
    sim.hold("W", "E")
    worst = 0.0
    while sim.time < 4.0:
        sim.top_out_if_needed()
        if sim.state == ParkourState.LEDGE_HANG and sim.time > 2.0:
            sim.hold("D")
        sim.step()
        worst = max(worst, max(sim.arm_rig.last["wrist_angles"]))
    assert worst <= config.WRIST_MAX_ANGLE_DEG + 1e-3
    assert config.WRIST_MAX_ANGLE_DEG <= 85.0, "human wrist extension is ~70-80 degrees"


def test_clamped_wrist_keeps_hand_attached_and_frame_orthonormal():
    from animation.skeleton import ArmSkeleton
    forearm = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # a hand asked to point straight back down the forearm (180 degrees) is limited to the cone
    rot = FirstPersonArmRig._hand_frame([0.0, -1.0, 0.0], [0.0, 0.0, 1.0])
    clamped, angle = ArmSkeleton.clamp_wrist(forearm, rot)
    assert abs(angle - config.WRIST_MAX_ANGLE_DEG) < 1e-3
    R = clamped[0:3, 0:3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5) and np.linalg.det(R) > 0.99
    assert np.degrees(np.arccos(np.dot(R[:, 1], forearm))) <= config.WRIST_MAX_ANGLE_DEG + 1e-3
    # within range nothing changes
    rot2 = FirstPersonArmRig._hand_frame([0.0, 0.9, 0.3], [0.0, 0.0, 1.0])
    same, angle2 = ArmSkeleton.clamp_wrist(forearm, rot2)
    assert same is rot2 and angle2 < 30.0


def test_forearm_and_hand_meshes_meet_at_the_wrist():
    """The forearm bone ends exactly at the wrist joint and the wrist cap sits on the wrist bone."""
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.2), with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.run(0.3)
    pal = sim.arm_rig.get_palette()
    L = sim.arm_rig.last
    for off, side in ((0, "left"), (19, "right")):
        forearm_tip = pal[off + 2] @ np.array([0.0, config.FOREARM_LEN, 0.0, 1.0], dtype=np.float32)
        wrist_origin = pal[off + 3][0:3, 3]
        assert np.linalg.norm(forearm_tip[:3] - wrist_origin) < 1e-4, f"{side} forearm detached from the hand"
        assert np.linalg.norm(wrist_origin - L[side + "_wrist"]) < 1e-4
        # forearm roll follows the palm: local +Z of the forearm is (nearly) the palm normal projected off the axis
        fz = pal[off + 2][0:3, 2]
        palm = pal[off + 3][0:3, 2]
        axis = pal[off + 2][0:3, 1]
        palm_perp = palm - axis * np.dot(palm, axis)
        palm_perp /= np.linalg.norm(palm_perp) + 1e-7
        assert np.dot(fz, palm_perp) > 0.95, "forearm should pronate with the hand"


def test_arm_mesh_is_rounded_and_well_formed():
    rig = FirstPersonArmRig()
    assert rig.vertices.shape[0] > 1500, "tapered prisms and joint caps, not 6-face boxes"
    assert rig.indices.max() < rig.vertices.shape[0]
    ids = rig.vertices[:, 8]
    assert ids.min() >= 0 and ids.max() <= BoneID.TOTAL_BONES - 1
    normals = rig.vertices[:, 3:6]
    assert np.all(np.isfinite(normals)) and np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)


def test_free_hands_move_smoothly_and_follow_the_torso_with_lag():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), with_arms=True)
    sim.hold("W", "SHIFT")
    sim.run(1.5)
    L0 = sim.arm_rig.last
    sim.player.yaw += 90.0            # an instant quarter turn of the head
    sim.step()
    L1 = sim.arm_rig.last
    jump = np.linalg.norm(L1["left_wrist"] - L0["left_wrist"])
    assert jump < 0.25, "hands must not snap to the new direction in one frame"
    # ... but they do get there
    sim.run(0.8)
    L2 = sim.arm_rig.last
    rel = L2["left_wrist"] - sim.player.camera_pos
    assert np.dot(rel, sim.player.forward_h) > 0.15, "running hands end up in front of the new view direction"
    # and while running straight the hands stay in front (no world-space lag)
    sim.run(1.0)
    L3 = sim.arm_rig.last
    rel3 = L3["left_wrist"] - sim.player.camera_pos
    assert np.dot(rel3, sim.player.forward_h) > 0.15


def test_hands_lift_toward_a_hold_being_looked_at_on_the_ground():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 2.3), yaw=0.0, with_arms=True)   # 3.3 m from the face: out of reach
    sim.run(0.5)
    rest_y = sim.arm_rig.last["left_wrist"][1]
    sim.look_at(np.array([0.0, 2.0, -1.0]))        # look at the head-height crevasse
    sim.hold("W")
    sim.run(0.6, until=lambda s: s.pos[2] < 0.9)   # walk to ~1.6 m from the face
    sim.release("W")
    sim.run(0.5)
    assert sim.arm_rig.last["left_wrist"][1] > rest_y + 0.05, "hands should rise toward the hold in anticipation"
    assert np.linalg.norm(sim.arm_rig.last["left_wrist"] - sim.arm_rig.last["left_shoulder"]) <= L1 + L2 + 1e-3
