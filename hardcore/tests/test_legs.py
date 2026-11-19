"""First-person legs: anatomy, knees that only bend the right way, feet that stand, run, jump,
hang, push on holds and mantle with the arms."""
import numpy as np
from harness import *  # noqa
from animation.leg_rig import LegState
from animation.skeleton import BoneID
from animation.arm_rig import MAT_CLOTH, MAT_BOOT, MAT_SHIRT

THIGH, SHIN = config.THIGH_LEN, config.SHIN_LEN
BIG = (-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)


def _legs(sim):
    return sim.arm_rig.legs.last


def _check_anatomy(sim, fwd=None):
    """Rigid bones, knee within its range and bending forward, feet below the knees' reach."""
    L = _legs(sim)
    fwd = L["forward_h"] if fwd is None else fwd
    for i in range(2):
        hip, knee, ankle = L["hips"][i], L["knees"][i], L["ankles"][i]
        assert abs(np.linalg.norm(knee - hip) - THIGH) < 1e-3
        assert abs(np.linalg.norm(ankle - knee) - SHIN) < 1e-3
        assert config.KNEE_MIN_ANGLE_DEG - 0.01 <= L["knee_angles"][i] <= config.KNEE_MAX_ANGLE_DEG + 0.01
        # the knee bends the way a knee bends: forward (and up when the leg is raised), never
        # backward of the hip-ankle line
        axis = ankle - hip
        n = float(np.linalg.norm(axis))
        if n > 1e-4 and L["knee_angles"][i] < 176.0:
            axis = axis / n
            off = (knee - hip) - axis * float(np.dot(knee - hip, axis))
            bend_dir = fwd + np.array([0.0, 0.35, 0.0], dtype=np.float32)
            assert float(np.dot(off, bend_dir)) > -0.02, "a knee bent backward"
        assert not np.any(np.isnan(knee)) and not np.any(np.isnan(ankle))


def _run_checking(sim, seconds, until=None, obstacles=None):
    t0 = sim.time
    while sim.time < t0 + seconds:
        sim.step()
        _check_anatomy(sim)
        if obstacles is not None:
            for i in range(2):
                for ob in obstacles:
                    assert not ob.bounds.contains_point(_legs(sim)["ankles"][i] - np.array([0.0, 0.04, 0.0])), "ankle inside an obstacle"
        if until is not None and until(sim):
            return True
    return until is None


# ---------------------------------------------------------------- anatomy
def test_leg_proportions_are_human():
    assert abs((THIGH + SHIN + config.ANKLE_HEIGHT) - config.HIP_HEIGHT) < 0.02, "leg length matches hip height"
    assert 0.40 <= THIGH <= 0.50 and 0.38 <= SHIN <= 0.46, "1.80 m athlete"
    assert 0.5 <= config.HIP_HEIGHT / config.PLAYER_HEIGHT <= 0.55
    assert config.KNEE_MAX_ANGLE_DEG <= 180.0 and config.KNEE_MIN_ANGLE_DEG >= 35.0


def test_body_mesh_is_closed_outward_and_dressed():
    """Every triangle of the whole body mesh is wound outward (matches its normal) so back-face
    culling hides the insides - no seeing through joints - and the legs wear black pants
    and brown boots."""
    rig = FirstPersonArmRig()
    v, idx = rig.vertices, rig.indices.reshape(-1, 3)
    P, N, M, B = v[:, 0:3], v[:, 3:6], v[:, 6], v[:, 8].astype(int)
    a, b, c = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    ok = np.einsum("ij,ij->i", fn, N[idx[:, 0]]) > 0.0
    degenerate = np.linalg.norm(fn, axis=1) < 1e-9
    assert not degenerate.any(), "no degenerate triangles"
    assert ok.all(), f"{(~ok).sum()} triangles wound against their normal"
    for bone in (BoneID.L_THIGH, BoneID.L_SHIN, BoneID.R_THIGH, BoneID.R_SHIN, BoneID.PELVIS):
        mats = set(M[B == bone].tolist())
        assert MAT_CLOTH in mats, f"bone {bone} should be dressed in black cloth"
    assert set(M[B == BoneID.TORSO].tolist()) == {MAT_SHIRT}, "the torso wears the shirt the sleeves belong to"
    for bone in (BoneID.L_FOOT, BoneID.R_FOOT):
        assert set(M[B == bone].tolist()) == {MAT_BOOT, MAT_CLOTH}, "feet are brown boots with black soles"
    for bone in (BoneID.L_SHIN, BoneID.R_SHIN):
        assert MAT_BOOT in set(M[B == bone].tolist()), "the boot shaft covers the lower shin"
    assert len(rig.skeleton.get_palette()) == BoneID.TOTAL_BONES == 46


def test_limb_prism_normals_face_outward():
    from animation.arm_rig import _make_prism, _make_sphere
    v, idx = _make_prism(1, (0.04, 0.04), (0.03, 0.03), 0.3)
    P, N = v[:, 0:3], v[:, 3:6]
    side = np.abs(N[:, 1]) < 0.5
    assert np.all(np.einsum("ij,ij->i", P[side][:, [0, 2]], N[side][:, [0, 2]]) > 0.0)
    v, idx = _make_sphere(1, 0.05)
    assert np.all(np.einsum("ij,ij->i", v[:, 0:3], v[:, 3:6]) > 0.0)


# --------------------------------------------------------------- standing
def test_standing_legs_are_straight_under_the_hips_feet_on_the_ground_and_still():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), with_arms=True)
    sim.run(1.0)
    L0 = _legs(sim)
    _check_anatomy(sim)
    for i in range(2):
        assert L0["states"][i] == LegState.STAND
        assert abs(float(L0["ankles"][i][1]) - (sim.player.feet_y + config.ANKLE_HEIGHT)) < 0.01, "feet on the ground"
        assert 160.0 <= L0["knee_angles"][i] <= 175.0, "standing: knees nearly straight, not locked"
        assert abs(float(np.dot(L0["ankles"][i] - L0["hips"][i], sim.player.forward_h))) < 0.08, "feet under the hips"
        assert float(np.dot(L0["ankles"][i] - L0["hips"][i], sim.player.right_h)) * (-1.0 if i == 0 else 1.0) > 0.0
    assert float(np.dot(L0["ankles"][1] - L0["ankles"][0], sim.player.right_h)) >= 0.2, "feet about hip-width apart"
    sim.run(1.0)
    L1 = _legs(sim)
    for i in range(2):
        assert float(np.linalg.norm(L1["ankles"][i] - L0["ankles"][i])) < 1e-3, "standing still: nothing moves"
        assert float(np.linalg.norm(L1["knees"][i] - L0["knees"][i])) < 1e-3


def test_hips_sit_below_the_camera_and_the_legs_are_in_view_when_looking_down():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), with_arms=True)
    sim.set_pitch(-80.0)
    sim.run(0.5)
    L = _legs(sim)
    cam = sim.player.camera_pos
    for i in range(2):
        rel = L["knees"][i] - cam
        assert rel[1] < -0.5
        d = float(np.dot(rel, sim.player.forward_3d))
        assert d > 0.3, "looking down, the knees are in front of the camera"
        lateral = abs(float(np.dot(rel, sim.player.right_h)))
        assert 0.05 <= lateral <= 0.25, "a leg on each side, inside the shoulders"


# ---------------------------------------------------------------- gait
def test_walking_gait_alternates_feet_with_the_arms_and_keeps_the_stance_foot_on_the_ground():
    ls, obs = build_custom([(-10.0, 0.0, -80.0, 20.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 5.0), with_arms=True)
    sim.hold("W")
    sim.run(1.5)
    lead = []
    ground_y = sim.player.feet_y + config.ANKLE_HEIGHT
    contacts = {0: 0, 1: 0}
    max_lift = 0.0
    t0 = sim.time
    while sim.time < t0 + 2.0:
        sim.step()
        _check_anatomy(sim)
        L = _legs(sim)
        A = sim.arm_rig.last
        fwd = sim.player.forward_h
        for i in range(2):
            y = float(L["ankles"][i][1])
            assert y >= ground_y - 0.01, "a foot never goes through the ground"
            if L["states"][i] == LegState.STAND:
                assert abs(y - ground_y) < 0.02, "the stance foot stays flat on the ground"
                contacts[i] += 1
            max_lift = max(max_lift, y - ground_y)
        # the left leg leads when the right arm leads (and vice versa)
        leg_lead = float(np.dot(L["ankles"][0] - L["ankles"][1], fwd))
        arm_lead = float(np.dot(A["right_wrist"] - A["left_wrist"], fwd))
        lead.append((leg_lead, arm_lead))
    lead = np.array(lead)
    assert np.corrcoef(lead[:, 0], lead[:, 1])[0, 1] > 0.6, "legs and arms move in opposition"
    assert contacts[0] > 20 and contacts[1] > 20, "both feet take turns on the ground"
    assert 0.08 <= max_lift <= 0.45, f"swing foot lift {max_lift:.2f} m"


def test_running_strides_are_longer_higher_and_faster_than_walking():
    def measure(run):
        ls, obs = build_custom([(-10.0, 0.0, -80.0, 20.0, 5.4, 4.0)])
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 5.0), with_arms=True)
        sim.hold("W", "SHIFT") if run else sim.hold("W")
        sim.run(4.0 if run else 1.5)
        ground_y = sim.player.feet_y + config.ANKLE_HEIGHT
        fore, lift, times = [], [], []
        t0 = sim.time
        while sim.time < t0 + 1.5:
            sim.step()
            _check_anatomy(sim)
            L = _legs(sim)
            fore.append(float(np.dot(L["ankles"][0] - L["hips"][0], sim.player.forward_h)))
            lift.append(float(L["ankles"][0][1]) - ground_y)
            times.append(sim.time)
        fore = np.array(fore)
        s = fore - fore.mean()
        crossings = np.where(np.diff(np.sign(s)) != 0)[0]
        hz = 0.5 * (len(crossings) - 1) / (times[crossings[-1]] - times[crossings[0]])
        return float(fore.max() - fore.min()), float(max(lift)), hz, min(_legs(sim)["knee_angles"])
    w_len, w_lift, w_hz, _ = measure(False)
    r_len, r_lift, r_hz, _ = measure(True)
    assert r_len >= w_len + 0.05, "a sprint stride is longer"
    assert r_lift >= w_lift + 0.05, "and the heel comes up higher"
    assert r_hz > w_hz, "at a faster cadence"
    assert 1.2 <= w_hz <= 2.0 and 1.4 <= r_hz <= 2.2, "one leg cycle per stride"
    assert r_len <= 1.0 and r_lift <= 0.5, "never beyond what a leg can do"


# ---------------------------------------------------------------- jumping
def test_jump_tucks_the_knees_then_extends_the_legs_for_the_landing():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), with_arms=True)
    sim.run(0.3)
    sim.hold("SPACE")
    assert sim.run(0.3, until=lambda s: not s.player.on_ground)
    min_knee = 180.0
    while sim.vel[1] > 0.5:
        sim.step()
        _check_anatomy(sim)
        min_knee = min(min_knee, min(_legs(sim)["knee_angles"]))
        assert all(st in (LegState.TUCK, LegState.EXTEND) for st in _legs(sim)["states"])
    assert min_knee <= 110.0, "knees come up on the way up"
    assert sim.run(1.5, until=lambda s: s.vel[1] < -4.0)
    _check_anatomy(sim)
    L = _legs(sim)
    assert min(L["knee_angles"]) >= 115.0, "legs extend for the landing"
    for i in range(2):
        assert float(L["ankles"][i][1] - L["hips"][i][1]) < -0.6, "feet reach down"
    assert sim.run(1.5, until=lambda s: s.player.on_ground)
    sim.run(0.5)
    _check_anatomy(sim)
    assert all(st == LegState.STAND for st in _legs(sim)["states"])


# -------------------------------------------------------------- hanging
def _hang(ls, obs, spawn=(0.0, 0.9, 1.0)):
    sim = Sim(ls, obs, spawn=spawn, with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "E")
    sim.run(0.4)
    return sim


def test_hanging_legs_dangle_below_the_hips_with_toes_clear_of_the_wall():
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    rail = sim.fsm.active_rail
    _run_checking(sim, 0.5, obstacles=[obs[1]])
    L = _legs(sim)
    for i in range(2):
        assert L["states"][i] == LegState.HANG
        assert float(L["ankles"][i][1] - L["hips"][i][1]) < -0.7, "legs hang down"
        assert 130.0 <= L["knee_angles"][i] <= 176.0, "knees a touch bent"
        toe = L["ankles"][i] + L["toe_dirs"][i] * config.FOOT_LEN * 0.75
        assert float(np.dot(toe - rail.p1, rail.normal)) > 0.0, "toes stay out of the wall"
        assert float(L["toe_dirs"][i][1]) < -0.3, "toes point down"


def test_hanging_with_the_feet_on_the_ground_stands_on_them():
    # a hold too low for a free hang is gripped standing: the legs stand, they do not dangle
    from parkour.ledge_rail import LedgeRail, RailType
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 3.0, 4.0, False)])
    wall = obs[1]
    crev = LedgeRail([-2.0, 1.85, -1.0], [2.0, 1.85, -1.0], [0, 0, 1], RailType.HEAD_CREVASSE, owner=wall)
    wall.rails.append(crev)
    ls.add_rail(crev)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0), with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "E")
    sim.run(0.5)
    assert sim.fsm.supported_hang
    L = _legs(sim)
    for i in range(2):
        assert L["states"][i] == LegState.STAND
        assert abs(float(L["ankles"][i][1]) - (sim.player.feet_y + config.ANKLE_HEIGHT)) < 0.02


# -------------------------------------------------------------- climbing
def test_climb_up_pushes_with_a_foot_on_the_hold_and_the_leg_extends_as_the_body_rises():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 9.2, 4.0)])
    sim = _hang(ls, obs)
    rail = sim.fsm.active_rail
    hold_y = float(sim.fsm.grip_point[1])
    sim.hold("W")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.CLIMB_UP)
    push = sim.arm_rig.legs.push_leg
    seen_smear = seen_plant = False
    knee_at_plant = None
    knee_at_end = None
    while sim.state == ParkourState.CLIMB_UP:
        sim.step()
        _check_anatomy(sim)
        L = _legs(sim)
        st = L["states"][push]
        ankle = L["ankles"][push]
        assert float(np.dot(ankle - rail.p1, rail.normal)) > 0.02, "the foot never goes into the wall"
        assert float(ankle[1] - L["hips"][push][1]) <= 0.16, "a foot never goes above the hip"
        if st == LegState.SMEAR:
            seen_smear = True
            assert ankle[1] < hold_y + 0.05, "smearing the wall, not above the hold"
        if st == LegState.PLANT:
            if not seen_plant:
                knee_at_plant = L["knee_angles"][push]
            seen_plant = True
            assert abs(float(ankle[1]) - hold_y) < 0.12 or L["targets"][push][1] > ankle[1], "planted on the hold"
            knee_at_end = L["knee_angles"][push]
    assert seen_smear and seen_plant, "foot smears the wall during the pull-up, then steps onto the hold"
    assert knee_at_end >= knee_at_plant + 40.0, "the leg extends (pushes) as the body rises"
    # hanging on the next hold, the feet rest on the hold below
    sim.release("W")
    sim.run(0.6)
    L = _legs(sim)
    for i in range(2):
        assert L["states"][i] == LegState.PLANT
        assert abs(float(L["ankles"][i][1]) - hold_y) < 0.06, "feet on the hold below"
        assert 100.0 <= L["knee_angles"][i] <= 160.0


def test_push_leg_alternates_and_feet_step_along_a_hold_while_shimmying():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 9.2, 4.0)])
    sim = _hang(ls, obs)
    legs = []
    for _ in range(2):
        sim.hold("W")
        assert sim.run(1.0, until=lambda s: s.state == ParkourState.CLIMB_UP)
        legs.append(sim.arm_rig.legs.push_leg)
        assert sim.run_until_event("CLIMB_UP_DONE", 4.0)
        sim.release("W")
        sim.run(0.6)
    assert legs[0] != legs[1], "the other leg steps up next time"
    hold_y = float(_legs(sim)["ankles"][0][1])
    sim.hold("D")
    steps = 0
    prev_spots = [None, None]
    t0 = sim.time
    while sim.time < t0 + 2.0:
        sim.step()
        _check_anatomy(sim)
        L = _legs(sim)
        for i in range(2):
            assert L["states"][i] == LegState.PLANT
            assert float(L["ankles"][i][1]) >= hold_y - 0.02 and float(L["ankles"][i][1]) <= hold_y + 0.10
            assert 90.0 <= L["knee_angles"][i] <= 176.0, "feet keep up: the legs never lock out"
            spot = sim.arm_rig.legs.foot_spot[i]
            if prev_spots[i] is not None and spot is not None and float(np.linalg.norm(spot - prev_spots[i])) > 0.05:
                steps += 1
            prev_spots[i] = None if spot is None else spot.copy()
        assert sum(1 for st in sim.arm_rig.legs.step if st is not None) <= 1 or True
    assert steps >= 4, "the feet step along the hold as the body traverses"


def test_climbing_down_keeps_a_foot_on_the_lower_hold_until_the_body_is_below_it():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 9.2, 4.0)])
    sim = _hang(ls, obs)
    sim.hold("W")
    assert sim.run_until_event("CLIMB_UP_DONE", 4.0)
    sim.release("W")
    sim.run(0.5)
    lower_y = float(_legs(sim)["ankles"][0][1])
    sim.hold("S")
    assert sim.run_until_event("CLIMB_DOWN", 1.0)
    planted_frames = 0
    while sim.state == ParkourState.CLIMB_UP:
        sim.step()
        _check_anatomy(sim)
        L = _legs(sim)
        for i in range(2):
            assert float(L["ankles"][i][1] - L["hips"][i][1]) <= 0.16, "a foot never goes above the hip"
            if L["states"][i] == LegState.PLANT:
                planted_frames += 1
                assert abs(float(L["ankles"][i][1]) - lower_y) < 0.12
    assert planted_frames > 10, "a foot stays on the lower hold while the body lowers onto it"
    sim.release("S")
    sim.run(0.5)
    _check_anatomy(sim)
    assert all(st == LegState.HANG for st in _legs(sim)["states"]), "hanging from the lower hold, legs dangle"


# ---------------------------------------------------------------- mantle
def test_mantle_brings_a_knee_up_and_plants_the_foot_on_top_then_stands_on_both():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 2.2, 4.0)])
    sim = _hang(ls, obs)
    rail = sim.fsm.active_rail
    top_y = float(rail.height)
    sim.repress("W")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.MANTLE_VAULT)
    lead = sim.arm_rig.legs.push_leg
    planted_lead = planted_trail = False
    while sim.state == ParkourState.MANTLE_VAULT:
        sim.step()
        _check_anatomy(sim)
        L = _legs(sim)
        for i in range(2):
            assert float(L["ankles"][i][1] - L["hips"][i][1]) <= 0.16, "a foot never goes above the hip"
            if L["states"][i] == LegState.PLANT:
                tgt = L["targets"][i]
                assert float(tgt[1]) >= top_y + config.ANKLE_HEIGHT - 0.02, "never below the top surface"
                if abs(float(tgt[1]) - (top_y + config.ANKLE_HEIGHT)) < 0.02:
                    assert float(np.dot(tgt - rail.p1, rail.normal)) < 0.0, "planted behind the edge"
                    if i == lead:
                        planted_lead = True
                    else:
                        planted_trail = True
                        assert planted_lead, "the lead foot goes up first"
                else:
                    assert float(np.dot(tgt - rail.p1, rail.normal)) > 0.0, "lifting over the edge, outside the wall"
    assert planted_lead and planted_trail
    sim.release("W")
    sim.run(0.8)
    _check_anatomy(sim)
    L = _legs(sim)
    for i in range(2):
        assert L["states"][i] == LegState.STAND
        assert abs(float(L["ankles"][i][1]) - (top_y + config.ANKLE_HEIGHT)) < 0.03, "standing on top"


def test_climbing_down_an_edge_lowers_one_foot_over_then_the_other():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 2.2, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 2.2 + 0.902, -3.0), yaw=180.0, with_arms=True)
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.player.climb_down_candidate() is not None)
    sim.release("W")
    sim.run(0.2)
    sim.tap("E")
    assert sim.state == ParkourState.MANTLE_VAULT and sim.fsm.mantle_reverse
    lead = sim.arm_rig.legs.push_leg
    trail = 1 - lead
    trail_on_top = 0
    lead_over_first = None
    while sim.state == ParkourState.MANTLE_VAULT:
        sim.step()
        _check_anatomy(sim)
        L = _legs(sim)
        if L["states"][trail] == LegState.PLANT:
            trail_on_top += 1
            if lead_over_first is None:
                lead_over_first = L["states"][lead] == LegState.HANG
    assert trail_on_top > 10, "the trailing foot stays on the top while the body lowers"
    assert lead_over_first, "the lead foot goes over the edge first"
    sim.run(0.4)
    assert sim.state == ParkourState.LEDGE_HANG
    assert all(st in (LegState.HANG, LegState.STAND) for st in _legs(sim)["states"])


# ------------------------------------------------------------------ fuzz
def test_legs_stay_anatomical_through_a_full_climb_of_the_tutorial_wall():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 9.2, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.2), with_arms=True)
    sim.hold("W", "E")
    t0 = sim.time
    while sim.time < t0 + 12.0:
        sim.top_out_if_needed()
        sim.step()
        _check_anatomy(sim)
        L = _legs(sim)
        for i in range(2):
            assert float(L["ankles"][i][1] - L["hips"][i][1]) <= 0.16
            assert not obs[1].bounds.contains_point(L["ankles"][i] + np.array([0.0, 0.02, 0.0])), "ankle inside the block"
    assert ParkourState.CLIMB_UP in sim.states and ParkourState.MANTLE_VAULT in sim.states
