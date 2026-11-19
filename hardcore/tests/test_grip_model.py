"""Wrist and finger behaviour on every kind of edge and in every climbing phase.

Real climbing: fingers do the holding and the wrist stays nearly straight; the
wrist only bends a lot while propping the body above the hands (top of a
pull-up, mantling, lowering over an edge, pushing off).  Fingers over an edge
are hidden by the edge.
"""
import numpy as np
from harness import *  # noqa
from animation.grip import grip_kind, GripKind, hang_extension, HandSpec, resolve_hand
from animation.finger_solver import HandPoseType, POSES
from parkour.ledge_rail import LedgeRail, RailType

BIG = (-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)


def _hang(ls, obs, spawn=(0.0, 0.9, 1.0)):
    sim = Sim(ls, obs, spawn=spawn, with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "E")
    sim.run(0.3)
    return sim


def _tips(sim, side):
    """Index and middle fingertip world positions."""
    pal = sim.arm_rig.get_palette()
    off = 0 if side == "left" else 19
    out = []
    for bone, length in ((off + 9, 0.018), (off + 12, 0.020)):
        out.append((pal[bone] @ np.array([0.0, length, 0.0, 1.0], dtype=np.float32))[:3])
    return out


# ------------------------------------------------------------------ edge kinds
def test_edge_kind_classification():
    ls, obs = build_custom([BIG, (4.0, 0.0, 2.0, 2.0, 1.5, 0.15)])   # a thin 15 cm slab edge
    crev = [r for r in ls.rails if r.is_crevasse][0]
    top = [r for r in ls.rails if r.rail_type == RailType.MANTLE_TOP and r.owner is obs[1]][0]
    thin = [r for r in ls.rails if r.owner is obs[2] and r.normal[2] > 0.5][0]
    assert grip_kind(crev) == GripKind.CRIMP
    assert grip_kind(top) == GripKind.LEDGE
    assert grip_kind(thin) == GripKind.CRIMP, "an edge with < 25 cm behind it is held like a thin edge"
    assert POSES[HandPoseType.CRIMP][1] > POSES[HandPoseType.HOOK][1] + 30, "crimp curls the PIP far more than a hook"


def test_hang_extension_is_a_function_of_body_height():
    # hanging well below the hold: nearly straight; hands at chest level: propped
    assert hang_extension(GripKind.CRIMP, 2.0, 1.5) == config.WRIST_EXT_HANG_CRIMP
    assert hang_extension(GripKind.LEDGE, 2.0, 1.5) == config.WRIST_EXT_HANG_LEDGE
    assert hang_extension(GripKind.LEDGE, 1.5, 1.6) == config.WRIST_EXT_PROP
    mid = hang_extension(GripKind.LEDGE, 1.7, 1.5)
    assert config.WRIST_EXT_HANG_LEDGE < mid < config.WRIST_EXT_PROP
    assert config.WRIST_EXT_HANG_CRIMP <= 20 and config.WRIST_EXT_HANG_LEDGE <= 30, "hanging is a finger hook, not a bent wrist"


def test_resolved_hand_frame_bends_the_designed_angle_toward_the_wall():
    rail = LedgeRail([-1, 2, 0], [1, 2, 0], [0, 0, 1], RailType.HEAD_CREVASSE)
    forearm = np.array([0.1, 1.0, 0.0])
    spec = HandSpec(HandSpec.HANG, rail, 15.0, POSES[HandPoseType.CRIMP])
    rot, angles = resolve_hand(spec, forearm)
    f = forearm / np.linalg.norm(forearm)
    bend = np.degrees(np.arccos(np.clip(np.dot(rot[0:3, 1], f), -1, 1)))
    assert abs(bend - 15.0) < 0.5
    assert np.dot(rot[0:3, 1], -rail.normal) > 0.2, "bent toward the wall"
    assert np.dot(rot[0:3, 2], -rail.normal) > 0.9, "palm faces the wall"
    press, _ = resolve_hand(HandSpec.press(rail), forearm)
    assert np.allclose(press[0:3, 2], [0, -1, 0], atol=1e-5), "press: palm down on the surface"


# ------------------------------------------------------- phases of a climb
def test_wrist_angles_through_a_full_climb_are_natural_and_continuous():
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    L = sim.arm_rig.last
    assert max(L["wrist_angles"]) <= 25.0, "hanging on a crevasse: slight wrist bend"
    assert L["finger_angles"][0][:3] == POSES[HandPoseType.CRIMP][:3]

    prev = max(L["wrist_angles"])
    pull_max = 0.0
    pull_frac = config.CLIMB_PULL_TIME / config.CLIMB_TOTAL_TIME
    sim.hold("W")
    while not sim.had_event("CLIMB_UP_DONE"):
        sim.step()
        L = sim.arm_rig.last
        cur = max(L["wrist_angles"])
        assert abs(cur - prev) <= 16.0, f"wrist snapped {prev:.0f} -> {cur:.0f} deg in one frame"
        prev = cur
        if sim.state == ParkourState.CLIMB_UP and sim.fsm.climb_phase < pull_frac:
            pull_max = max(pull_max, cur)
        assert cur <= config.WRIST_MAX_ANGLE_DEG + 1e-3
    assert pull_max >= 45.0, "the wrist extends as the chest is pulled up to the hold"
    sim.release("W")
    sim.run(0.4)
    assert max(sim.arm_rig.last["wrist_angles"]) <= 25.0, "back to a finger hook on the new hold"


def test_hopping_hand_opens_with_a_straight_wrist():
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    sim.hold("D")
    seen = False
    t0 = sim.time
    while sim.time < t0 + 2.0:
        sim.step()
        L = sim.arm_rig.last
        m = L["moving_hand"]
        if m is not None:
            assert L["hand_modes"][m] == HandSpec.OPEN
            assert L["hand_modes"][1 - m] == HandSpec.HANG, "the planted hand keeps holding"
            if 0.45 < sim.arm_rig.planner.hands[m].t < 0.55:      # apex of the hop: fully open
                seen = True
                # the open hand is not hooked: it only flexes back enough to skim the wall
                assert L["wrist_angles"][m] <= config.WRIST_FLEX_CLEAR_DEG + 1e-3
                rail = sim.fsm.active_rail
                for tip in _tips(sim, "left" if m == 0 else "right"):
                    assert float(np.dot(tip - rail.p1, rail.normal)) > -0.005, "fingertips stay off the wall"
                assert L["finger_angles"][m][:3] == POSES[HandPoseType.OPEN][:3]
    assert seen


def test_ledge_hook_then_press_on_mantle_then_release():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 2.2, 4.0)])
    sim = _hang(ls, obs)
    L = sim.arm_rig.last
    assert 18.0 <= max(L["wrist_angles"]) <= 35.0, "hooked over a deep ledge: slight extension"
    assert L["finger_angles"][0][:3] == POSES[HandPoseType.HOOK][:3]
    rail = sim.fsm.active_rail
    sim.repress("W")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.MANTLE_VAULT)
    press_seen = False
    prev = max(sim.arm_rig.last["wrist_angles"])
    while sim.state == ParkourState.MANTLE_VAULT:
        sim.step()
        L = sim.arm_rig.last
        cur = max(L["wrist_angles"])
        assert abs(cur - prev) <= 16.0
        prev = cur
        ph = sim.fsm.climb_phase
        if 0.50 <= ph <= 0.60:
            press_seen = True
            assert min(L["wrist_angles"]) >= 65.0, "propping on the palms bends the wrist a lot"
            assert max(abs(a - b) for a, b in zip(L["finger_angles"][0][:3], POSES[HandPoseType.MANTLE_PALM][:3])) < 1.0
            for side in ("left", "right"):
                w = L[side + "_wrist"]
                assert float(np.dot(w - rail.p1, rail.normal)) < 0.0, "palms are on the surface behind the edge"
                assert w[1] > rail.height, "and above it"
    assert press_seen
    sim.run(0.5)
    assert all(m in (HandSpec.FREE, HandSpec.OPEN) for m in sim.arm_rig.last["hand_modes"]), "hands released after the step-through"
    assert sim.arm_rig.last["hand_rails"] == (None, None)


def test_lowering_over_an_edge_goes_from_press_to_hook():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 2.2, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 2.2 + 0.902, -3.0), yaw=180.0, with_arms=True)
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.player.climb_down_candidate() is not None)
    sim.release("W")
    sim.run(0.2)
    sim.tap("E")
    angles = []
    while sim.state == ParkourState.MANTLE_VAULT:
        sim.step()
        angles.append(max(sim.arm_rig.last["wrist_angles"]))
    assert angles[0] >= 70.0, "starts propped on the palms"
    assert angles[-1] <= 40.0, "ends hooked on the edge"
    assert all(b - a <= 3.0 for a, b in zip(angles[:-1], angles[1:])), "extension only decreases as the body lowers"
    sim.run(0.3)
    assert max(sim.arm_rig.last["wrist_angles"]) <= 35.0


def test_thin_edge_keeps_fingers_hooked_while_a_ledge_gets_a_palm_press_during_the_push_up():
    # thin edge (crevasse): the hands stay hooked, wrist extension from body height only
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    sim.hold("W")
    modes = set()
    while not sim.had_event("CLIMB_UP_DONE"):
        sim.step()
        modes.update(sim.arm_rig.last["hand_modes"])
    assert HandSpec.PRESS not in modes
    # deep edge (window head on the building): palms prop on the lip while the legs push
    ls, obs = build_playground()
    sim = Sim(ls, obs, spawn=(-24.2, 0.9, 38.8), with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert grip_kind(sim.fsm.active_rail) == GripKind.LEDGE
    modes = set()
    while not sim.had_event("CLIMB_UP_DONE"):
        sim.step()
        modes.update(sim.arm_rig.last["hand_modes"])
    assert HandSpec.PRESS in modes


# ---------------------------------------------------------------- occlusion
def test_fingers_over_the_edge_are_inside_the_wall_where_it_hides_them():
    for block in (BIG, (-10.0, 0.0, -5.0, 20.0, 2.2, 4.0)):
        ls, obs = build_custom([block])
        sim = _hang(ls, obs)
        rail = sim.fsm.active_rail
        for side in ("left", "right"):
            for tip in _tips(sim, side):
                assert rail.owner.bounds.contains_point(tip), f"{side} fingertip outside the edge it holds"
        # and the wrists themselves are outside, below the edge (visible)
        for side in ("left", "right"):
            w = sim.arm_rig.last[side + "_wrist"]
            assert float(np.dot(w - rail.p1, rail.normal)) > 0.0 and w[1] < rail.height


def test_hopping_hand_fingers_come_out_into_view():
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    rail = sim.fsm.active_rail
    sim.hold("D")
    seen = False
    t0 = sim.time
    while sim.time < t0 + 2.0:
        sim.step()
        m = sim.arm_rig.last["moving_hand"]
        if m is not None and 0.45 < sim.arm_rig.planner.hands[m].t < 0.55:
            seen = True
            tips = _tips(sim, "left" if m == 0 else "right")
            assert not any(rail.owner.bounds.contains_point(t) for t in tips), "an open hopping hand is out in the air"
    assert seen


def test_arms_are_depth_tested_against_the_world():
    """The renderer must not clear the depth buffer before drawing the arms (that made hands
    show through ledges)."""
    src = open(os.path.join(ROOT, "render", "renderer.py")).read()
    world_pass = src.index("PASS 1")
    arm_pass = src.index("PASS 3")
    assert "glClear(GL_DEPTH_BUFFER_BIT)" not in src[world_pass:arm_pass + 600]
