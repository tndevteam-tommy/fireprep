"""Hand-over-hand shimmy animation, corner turning and climbing down over an edge."""
import numpy as np
from harness import *  # noqa
from parkour.ledge_rail import RailLink


def _hang_on(ls, obs, spawn, yaw=0.0, with_arms=True):
    sim = Sim(ls, obs, spawn=spawn, yaw=yaw, with_arms=with_arms)
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.player.grab_candidate() is not None)
    sim.hold("SPACE")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "SPACE")
    sim.run(0.3)
    return sim


# ----------------------------------------------------------------- shimmy hands
def test_shimmy_is_hand_over_hand_one_hand_at_a_time():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (0.0, 0.9, 1.6))
    sim.hold("D")
    order = []
    prev = None
    t0 = sim.time
    while sim.time < t0 + 4.0:
        sim.step()
        L = sim.arm_rig.last
        assert sum(h.moving for h in sim.arm_rig.planner.hands) <= 1, "both hands left the rail at once"
        m = L["moving_hand"]
        if m is not None and m != prev:
            order.append(m)
        prev = m
    assert len(order) >= 6, "a 4 s traverse should need several hand hops"
    assert all(a != b for a, b in zip(order[:-1], order[1:])), f"hands must alternate: {order}"
    assert order[0] == 1, "traversing right, the right hand leads"
    # hands stay near the body and spaced like shoulders
    L = sim.arm_rig.last
    d_grip = sim.fsm.active_rail.distance_along(sim.fsm.grip_point)
    assert abs(L["hand_d"][0] - d_grip) < 0.5 and abs(L["hand_d"][1] - d_grip) < 0.5
    assert 0.2 <= L["hand_d"][1] - L["hand_d"][0] <= 0.8


def test_hands_hop_in_an_arc_and_never_stretch():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (0.0, 0.9, 1.6))
    sim.hold("D")
    lifted = False
    t0 = sim.time
    while sim.time < t0 + 2.0:
        sim.step()
        L = sim.arm_rig.last
        for side in ("left", "right"):
            assert np.linalg.norm(L[side + "_wrist"] - L[side + "_target"]) < 0.01, "hand could not reach its spot"
        m = L["moving_hand"]
        if m is not None:
            wrist = L["left_wrist" if m == 0 else "right_wrist"]
            if wrist[1] > 2.0 - config.WRIST_DROP_CRIMP + 0.04:
                lifted = True
    assert lifted, "a hopping hand should lift off the rail"


def test_leading_hand_stops_at_the_end_of_the_edge():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (6.0, 0.9, 1.6))
    rail = sim.fsm.active_rail
    rail.links.pop("right", None)            # the edge ends here (nothing to turn onto)
    sim.hold("D")
    assert sim.run(6.0, until=lambda s: s.had_event("EDGE_END"))
    sim.run(0.6)
    L = sim.arm_rig.last
    lead = L["hand_d"][1]
    assert lead <= rail.length - config.HAND_RAIL_END_MARGIN + 1e-3, "hand planted past the end of the edge"
    assert lead >= rail.length - config.HAND_RAIL_END_MARGIN - 0.08, "leading hand should reach the end of the edge"
    assert rail.distance_along(sim.fsm.grip_point) <= rail.length - config.RAIL_END_MARGIN + 1e-3
    assert sim.state == ParkourState.SHIMMY and sim.fsm.active_rail is rail


# ---------------------------------------------------------------- corner turns
def test_rails_of_a_block_link_around_every_corner():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    for r in ls.rails:
        assert r.links["right"].kind == RailLink.CONVEX and r.links["left"].kind == RailLink.CONVEX
        assert abs(r.links["right"].rail.height - r.height) < 1e-6
        assert np.linalg.norm(r.links["right"].rail.p1 - r.p2) < 1e-3


def test_l_shape_produces_inside_corners():
    ls, obs = build_custom([(-12.0, 0.0, 38.0, 8.0, 12.0, 8.0), (-18.0, 0.0, 42.0, 6.0, 6.0, 4.0)])
    kinds = [l.kind for r in ls.rails for l in r.links.values()]
    assert RailLink.CONCAVE in kinds and RailLink.CONVEX in kinds
    # the annex covers part of the main block's west face: that rail is clipped at the corner
    west = [r for r in ls.rails if r.normal[0] < -0.5 and abs(r.height - 2.0) < 1e-3 and r.owner.bounds.size[1] > 10]
    assert len(west) == 1 and abs(west[0].length - 4.0) < 1e-3


def test_shimmy_turns_every_outside_corner_and_loops_the_block():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (0.0, 0.9, 1.6), with_arms=False)
    yaw0 = sim.player.yaw
    start = sim.pos.copy()
    sim.hold("D")
    t0 = sim.time
    assert sim.run(25.0, until=lambda s: s.time - t0 > 3.0 and np.linalg.norm(s.pos - start) < 0.3
                   and s.state == ParkourState.SHIMMY)
    corners = [e for _, e in sim.events if e == "CORNER_CONVEX"]
    assert len(corners) == 4
    assert abs(((sim.player.yaw - yaw0) % 360.0)) < 1.0 or abs(((sim.player.yaw - yaw0) % 360.0) - 360.0) < 1.0
    assert sim.max_root_step < 0.15, "corner swing teleported"
    for tr in sim.trace:
        if tr["state"] in (ParkourState.SHIMMY, ParkourState.LEDGE_HANG, ParkourState.CORNER_TURN):
            assert BodyModel.pose_is_free(tr["pos"], obs), "body clipped the block while turning the corner"


def test_corner_turn_duration_and_camera_turn():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (1.2, 0.9, 1.6), with_arms=False)
    yaw0 = sim.player.yaw
    sim.hold("D")
    assert sim.run(4.0, until=lambda s: s.had_event("CORNER_CONVEX"))
    t0 = sim.time
    assert sim.run_until_event("CORNER_DONE", 2.0)
    assert 0.5 <= sim.time - t0 <= 0.9
    sim.run(0.2)
    d = (sim.player.yaw - yaw0 + 540.0) % 360.0 - 180.0
    assert abs(abs(d) - 90.0) < 1.0, f"camera should have turned 90 degrees, got {d:.1f}"
    assert sim.fsm.active_rail.normal[0] > 0.5, "now hanging on the east face"


def test_inside_corner_turn_both_ways():
    ls, obs = build_custom([(-12.0, 0.0, 38.0, 8.0, 12.0, 8.0), (-18.0, 0.0, 42.0, 6.0, 6.0, 4.0)])
    sim = _hang_on(ls, obs, (-13.6, 0.9, 40.0), yaw=-90.0, with_arms=True)
    assert sim.fsm.active_rail.normal[0] < -0.5
    sim.hold("D")
    assert sim.run(6.0, until=lambda s: s.had_event("CORNER_CONCAVE"))
    assert sim.run_until_event("CORNER_DONE", 2.0)
    assert sim.fsm.active_rail.normal[2] < -0.5, "now on the annex wall that was beside/behind the player"
    assert abs(((sim.player.yaw + 180.0) % 360.0)) < 1.5 or abs(((sim.player.yaw + 180.0) % 360.0) - 360.0) < 1.5
    x_after = float(sim.pos[0])
    sim.run(1.0)
    assert float(sim.pos[0]) < x_after - 0.5, "holding the same key keeps traversing along the new wall"
    sim.release("D")
    sim.hold("A")
    assert sim.run_until_event("CORNER_CONCAVE", 6.0)
    assert sim.run_until_event("CORNER_DONE", 2.0)
    assert sim.fsm.active_rail.normal[0] < -0.5, "back on the original wall"
    for tr in sim.trace:
        if tr["state"] in (ParkourState.SHIMMY, ParkourState.LEDGE_HANG, ParkourState.CORNER_TURN):
            assert BodyModel.pose_is_free(tr["pos"], obs)


def test_corner_turn_hands_lead_hand_goes_first():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (1.0, 0.9, 1.6))
    sim.hold("D")
    assert sim.run(4.0, until=lambda s: s.had_event("CORNER_CONVEX"))
    q = sim.fsm.corner_link.rail
    first_on_q = None
    while sim.state == ParkourState.CORNER_TURN:
        sim.step()
        rails = sim.arm_rig.last["hand_rails"]
        for i in (1, 0):
            if rails[i] is q and first_on_q is None:
                first_on_q = i
        assert sum(h.moving for h in sim.arm_rig.planner.hands) <= 1
    assert first_on_q == 1, "turning right, the right hand reaches around the corner first"
    sim.run(0.3)
    rails = sim.arm_rig.last["hand_rails"]
    assert rails[0] is q and rails[1] is q, "both hands end on the new edge"


def test_corner_turn_with_the_other_hand_when_turning_left():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 5.4, 4.0)])
    sim = _hang_on(ls, obs, (-1.0, 0.9, 1.6))
    sim.hold("A")
    assert sim.run(4.0, until=lambda s: s.had_event("CORNER_CONVEX"))
    q = sim.fsm.corner_link.rail
    first_on_q = None
    while sim.state == ParkourState.CORNER_TURN:
        sim.step()
        rails = sim.arm_rig.last["hand_rails"]
        for i in (0, 1):
            if rails[i] is q and first_on_q is None:
                first_on_q = i
    assert first_on_q == 0


# -------------------------------------------------------------- climb-down edge
def _on_top(ls, obs, spawn, yaw):
    sim = Sim(ls, obs, spawn=spawn, yaw=yaw, with_arms=True)
    sim.hold("W")
    ok = sim.run(3.0, until=lambda s: s.player.climb_down_candidate() is not None)
    sim.release("W")
    sim.run(0.2)
    return sim, ok


def test_climb_down_option_appears_only_at_an_edge_facing_out():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 2.2, 4.0)])
    sim, ok = _on_top(ls, obs, (0.0, 2.2 + 0.902, -3.0), 180.0)
    assert ok and sim.player.climb_down_candidate() is not None
    sim.player.yaw = 0.0                     # turn away from the edge
    assert sim.player.climb_down_candidate() is None
    sim.teleport((0.0, 2.2 + 0.902, -3.0), yaw=180.0)
    assert sim.player.climb_down_candidate() is None, "too far from the edge"


def test_e_at_an_edge_reverses_the_mantle_into_a_hang():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 2.2, 4.0)])
    sim, ok = _on_top(ls, obs, (0.0, 2.2 + 0.902, -3.0), 180.0)
    assert ok
    yaw0 = sim.player.yaw
    sim.tap("E")
    assert sim.events[-1][1] == "CLIMB_DOWN_EDGE"
    t0 = sim.time
    rail = sim.fsm.active_rail
    first_wrist = None
    while sim.state == ParkourState.MANTLE_VAULT:
        sim.step()
        if first_wrist is None:
            first_wrist = sim.arm_rig.last["left_wrist"].copy()
    last_wrist = sim.arm_rig.last["left_wrist"]
    assert float(np.dot(first_wrist - rail.p1, rail.normal)) < -0.05, "hands start pressed on the top surface behind the edge"
    assert float(np.dot(last_wrist - rail.p1, rail.normal)) > 0.0, "and end hooked over the edge"
    assert 0.8 <= sim.time - t0 <= 1.3, "a controlled lower, not a drop"
    assert sim.state == ParkourState.LEDGE_HANG
    assert abs(sim.fsm.active_rail.height - 2.2) < 1e-3
    sim.run(0.3)
    d = (sim.player.yaw - yaw0 + 540.0) % 360.0 - 180.0
    assert abs(abs(d) - 180.0) < 1.5, "ends facing the wall"
    assert sim.max_root_step < 0.15
    # and the climber can go straight back up
    sim.hold("W")
    assert sim.run_until_event("MANTLE_DONE", 3.0)


def test_climb_down_edge_needs_a_fresh_press_not_a_held_e():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 2.2, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 2.2 + 0.902, -3.0), yaw=180.0)
    sim.hold("W", "E")
    sim.run(1.5)
    assert not sim.had_event("CLIMB_DOWN_EDGE"), "running to an edge holding E must not lower you"
