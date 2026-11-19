"""Arm animation: running pump, hand-over-hand climbing, corner reaches.

What a climber's hands actually do:
  * running: the arms pump in opposition, harder and faster the faster you go; they go
    quiet in the air
  * climbing: a hand only lets go of an edge when the next hold is within reach, hops to it
    once (no ping-pong between edges), the lead hand alternates from one hand-over to the
    next and the swing travels with the body instead of being dragged through it
  * corners: the lead hand reaches around only as far as the arm allows and the trailing
    hand follows
"""
import numpy as np
import pytest
from harness import *  # noqa
from animation.grip import HandSpec
from tests.test_traverse_corners import _hang_on

BIG = (-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)
REACH = config.UPPER_ARM_LEN + config.FOREARM_LEN


def _hang(ls, obs, spawn=(0.0, 0.9, 1.0)):
    sim = Sim(ls, obs, spawn=spawn, with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "E")
    sim.run(0.3)
    return sim


def _fore_aft(sim, side):
    """Hand position along the body's forward axis, relative to its shoulder."""
    L = sim.arm_rig.last
    return float(np.dot(L[side + "_wrist"] - L[side + "_shoulder"], L["forward_h"]))


def _record_swing(sim, seconds):
    left, right, times = [], [], []
    t0 = sim.time
    while sim.time < t0 + seconds:
        sim.step()
        left.append(_fore_aft(sim, "left"))
        right.append(_fore_aft(sim, "right"))
        times.append(sim.time)
    return np.array(left), np.array(right), np.array(times)


def _frequency_hz(signal, times):
    """Dominant frequency from zero crossings of the de-meaned signal."""
    s = signal - signal.mean()
    crossings = np.where(np.diff(np.sign(s)) != 0)[0]
    if len(crossings) < 2:
        return 0.0
    return 0.5 * (len(crossings) - 1) / (times[crossings[-1]] - times[crossings[0]])


# ------------------------------------------------------------------ running pump
def _height_rel(sim, side):
    L = sim.arm_rig.last
    return float(L[side + "_wrist"][1] - L[side + "_shoulder"][1])


def test_arms_hang_and_swing_at_a_walk_and_pump_only_when_running():
    ls, obs = build_custom([(-10.0, 0.0, -80.0, 20.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 5.0), with_arms=True)
    sim.hold("W")
    sim.run(1.0)
    wl, wr, wt = _record_swing(sim, 2.0)
    walk_amp = float(wl.max() - wl.min())
    walk_hz = _frequency_hz(wl, wt)
    assert 0.25 <= walk_amp <= 0.50, f"walking arm swing {walk_amp:.2f} m (a pendulum from the shoulder)"
    assert 1.0 <= walk_hz <= 2.0, f"walking arm cycle {walk_hz:.2f} Hz (one per stride)"
    assert np.corrcoef(wl - wl.mean(), wr - wr.mean())[0, 1] < -0.8, "arms swing in opposition"
    assert _height_rel(sim, "left") < -0.45, "walking: the arms hang, hands down by the thighs"
    L = sim.arm_rig.last
    assert 150.0 <= L["left_elbow_angle"] <= 175.0, "hanging arm: slight bend at the elbow"

    sim.hold("SHIFT")
    sim.run(3.5)                       # up to full sprint
    sl, sr, st = _record_swing(sim, 2.0)
    sprint_amp = float(sl.max() - sl.min())
    sprint_hz = _frequency_hz(sl, st)
    assert 0.35 <= sprint_amp <= 0.60, f"sprint pump {sprint_amp:.2f} m"
    assert sprint_hz >= walk_hz + 0.2, "a sprint is a faster cycle"
    assert sprint_hz <= 2.2, "but no faster than a sprinter's stride"
    assert np.corrcoef(sl - sl.mean(), sr - sr.mean())[0, 1] < -0.8
    heights = []
    t0 = sim.time
    while sim.time < t0 + 0.8:
        sim.step()
        heights.append(_height_rel(sim, "left"))
    assert max(heights) > -0.12, "running: the fists come up to the chest at the front of the swing"
    assert min(heights) < -0.35, "and drop to the hip at the back"
    assert float(sl.mean()) > 0.15, "and the fists are out in front"
    assert max(sim.arm_rig.last["wrist_angles"]) <= 25.0, "fists stay in line with the forearms while pumping"
    assert 60.0 <= sim.arm_rig.last["left_elbow_angle"] <= 130.0, "pumping: elbows bent"


def test_arms_come_up_as_the_run_starts_and_drop_back_when_it_stops():
    ls, obs = build_custom([(-10.0, 0.0, -80.0, 20.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 5.0), with_arms=True)
    sim.hold("W")
    sim.run(1.0)
    assert sim.arm_rig.arms_up < 0.05
    sim.hold("SHIFT")
    sim.run(1.0)
    assert sim.arm_rig.arms_up > 0.9, "a jog brings the arms up"
    sim.release("SHIFT")
    sim.run(1.5)
    assert sim.arm_rig.arms_up < 0.1, "back to a walk: the arms drop again"


def test_arms_go_quiet_in_the_air_and_at_rest():
    ls, obs = build_custom([(-10.0, 0.0, -80.0, 20.0, 5.4, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 5.0), with_arms=True)
    sim.hold("W", "SHIFT")
    sim.run(2.5)
    sim.hold("SPACE")
    assert sim.run(0.3, until=lambda s: not s.player.on_ground)
    sim.run(0.2)
    al, ar, at = _record_swing(sim, 0.25)
    assert not sim.player.on_ground
    sim.release("SPACE")
    assert float(al.max() - al.min()) < 0.05 and _frequency_hz(al, at) < 1.0, "no pumping in the air"
    sim.release("W", "SHIFT")
    sim.run(1.5)
    rl, rr, rt = _record_swing(sim, 0.5)
    assert float(rl.max() - rl.min()) < 0.01, "hands settle when standing still"


# ------------------------------------------------------- hand-over-hand climbing
def _climb_hops(sim, climbs):
    """Run `climbs` climb-ups with W held; returns the per-climb list of hand-over hops."""
    out = []
    for _ in range(climbs):
        sim.hold("W")
        n0 = len(sim.arm_rig.planner.hops)
        assert sim.run_until_event("CLIMB_UP_DONE", 4.0)
        sim.release("W")
        sim.run(config.HANDOVER_TIME * 2 + 0.2)
        out.append(sim.arm_rig.planner.hops[n0:])
    return out


def test_lead_hand_alternates_between_hand_overs():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 9.2, 4.0)])
    sim = _hang(ls, obs)
    hops = _climb_hops(sim, 3)
    leads = []
    for climb in hops:
        handovers = [h for h in climb if h[2] is not None]
        assert handovers, "every climb is a hand-over to the next hold"
        leads.append(handovers[0][1])
    assert leads in ([0, 1, 0], [1, 0, 1]), f"lead hand alternates: {leads}"


def test_hands_hop_to_the_next_hold_once_and_never_back():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 9.2, 4.0)])
    sim = _hang(ls, obs)
    for climb in _climb_hops(sim, 2):
        rails = [h[2] for h in climb]
        per_hand = {0: [], 1: []}
        for _, i, rail, _, _ in climb:
            per_hand[i].append(rail)
        for i in (0, 1):
            assert len(per_hand[i]) == 1, f"hand {i} made {len(per_hand[i])} hops in one climb (ping-pong)"
        assert rails[0] is rails[1], "both hands go to the same new edge"


def test_climbing_hands_never_slide_back_down_the_wall():
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    sim.hold("W")
    prev = {s: float(sim.arm_rig.last[s + "_target"][1]) for s in ("left", "right")}
    while not sim.had_event("CLIMB_UP_DONE"):        # (first climb of this sim)
        sim.step()
        for side in ("left", "right"):
            y = float(sim.arm_rig.last[side + "_target"][1])
            # hop arcs land HOP_LIFT lower than their apex; nothing else ever goes down
            assert y >= prev[side] - config.HOP_LIFT - 0.01, f"{side} hand dropped {prev[side] - y:.2f} m during a climb-up"
            prev[side] = y


@pytest.mark.parametrize("keys", [("W",), ("W", "SHIFT")], ids=["climb", "fast-climb"])
def test_hand_over_swings_clear_of_the_shoulder_and_lands_as_the_body_arrives(keys):
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    nxt_rail, nxt_s = None, None
    sim.hold(*keys)
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.CLIMB_UP)
    nxt_rail, nxt_s = sim.fsm.climb_next
    min_dist = 10.0
    while sim.state == ParkourState.CLIMB_UP:
        sim.step()
        L = sim.arm_rig.last
        m = L["moving_hand"]
        if m is not None and 0.2 <= sim.arm_rig.planner.hands[m].t <= 0.8:
            # (a hop may *start* close in - hands at the chest at the top of a pull-up - but the
            # swing itself clears the shoulder)
            side = "left" if m == 0 else "right"
            min_dist = min(min_dist, float(np.linalg.norm(L[side + "_target"] - L[side + "_shoulder"])))
    assert min_dist >= config.HOP_MIN_SHOULDER_DIST - 0.02, "a reaching hand is never dragged through its own shoulder"
    rails = sim.arm_rig.last["hand_rails"]
    assert nxt_rail in rails, "the lead hand has caught the next hold when the body gets there"


def test_a_freed_hand_opens_gradually_and_follows_the_body():
    ls, obs = build_custom([BIG])
    sim = _hang(ls, obs)
    sim.hold("W")
    prev_angles = None
    prev_rel = None
    while not sim.had_event("CLIMB_UP_DONE"):
        sim.step()
        L = sim.arm_rig.last
        angles = L["wrist_angles"]
        if prev_angles is not None:
            for a, b in zip(prev_angles, angles):
                assert abs(a - b) <= 16.0, "wrist never snaps"
        prev_angles = angles
        rel = L["left_target"] - L["left_shoulder"]
        if prev_rel is not None and L["hand_modes"][0] == HandSpec.OPEN and L["moving_hand"] != 0:
            # a freed (not hopping) hand slides at a human speed relative to the shoulder
            assert float(np.linalg.norm(rel - prev_rel)) <= (config.REACH_SLIDE_SPEED + 0.5) * DT + 1e-3
        prev_rel = rel


def test_climbing_down_alternates_and_never_stretches():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 9.2, 4.0)])
    sim = _hang(ls, obs)
    _climb_hops(sim, 2)
    down_leads = []
    for _ in range(2):
        n0 = len(sim.arm_rig.planner.hops)
        sim.hold("S")
        assert sim.run_until_event("CLIMB_DOWN", 1.0)
        assert sim.run_until_event("CLIMB_UP_DONE", 4.0)
        sim.release("S")
        sim.run(config.HANDOVER_TIME * 2 + 0.2)
        handovers = [h for h in sim.arm_rig.planner.hops[n0:] if h[2] is not None]
        assert handovers
        down_leads.append(handovers[0][1])
        for tr in sim.trace[-60:]:
            assert tr["state"] in (ParkourState.LEDGE_HANG, ParkourState.CLIMB_UP, ParkourState.SHIMMY)
    assert down_leads[0] != down_leads[1], "down-climbs alternate the lead hand too"
    L = sim.arm_rig.last
    for side in ("left", "right"):
        assert float(np.linalg.norm(L[side + "_wrist"] - L[side + "_shoulder"])) <= REACH + 1e-3


# --------------------------------------------------------------------- corners
def test_corner_reach_stays_within_arms_length_and_trailing_hand_follows():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (1.0, 0.9, 1.6))
    sim.hold("D")
    assert sim.run(4.0, until=lambda s: s.had_event("CORNER_CONVEX"))
    q = sim.fsm.corner_link.rail
    lead_landed_at = None
    trail_launched_at = None
    while sim.state == ParkourState.CORNER_TURN:
        sim.step()
        L = sim.arm_rig.last
        for side in ("left", "right"):
            d = float(np.linalg.norm(L[side + "_target"] - L[side + "_shoulder"]))
            assert d <= REACH * config.HANDOVER_REACH_FRACTION + 0.02, f"{side} hand target beyond reach ({d:.2f} m)"
        rails = L["hand_rails"]
        if rails[1] is q and lead_landed_at is None:
            lead_landed_at = sim.time
        trail = sim.arm_rig.planner.hands[0]
        if trail.moving and trail.to_rail is q and trail_launched_at is None:
            trail_launched_at = sim.time
    assert lead_landed_at is not None
    assert trail_launched_at is not None and trail_launched_at >= lead_landed_at - 1e-6, "the trailing hand waits for the lead hand"
    sim.run(0.4)
    assert sim.arm_rig.last["hand_rails"] == (q, q)
