"""The control scheme: one parkour button, push-to-climb, Caps Lock run toggle.

These tests cover the forgiving-but-fallible feel of the scheme; the rest of the
suite exercises the same scheme through the full climbing mechanics.
"""
import numpy as np
from harness import *  # noqa
from core.controls import make_scheme, FlowScheme, ControlSchemeName


def _tower(h=5.4):
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, h, 4.0)])
    return ls, obs, obs[1]


def _flow_hang(ls, obs, spawn=(0.0, 0.9, 1.6)):
    """Walk up to the wall, hold Space once a hold is in reach -> hang on the 2.0 m crevasse."""
    sim = Sim(ls, obs, spawn=spawn, scheme="flow")
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.player.grab_candidate() is not None)
    sim.hold("SPACE")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "SPACE")
    sim.run(0.3)
    return sim


def test_default_scheme_is_flow():
    assert config.CONTROL_SCHEME == ControlSchemeName.FLOW
    assert isinstance(make_scheme(None), FlowScheme)


def test_caps_lock_toggles_run_and_shift_is_momentary():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, scheme="flow")
    sim.hold("W")
    sim.run(1.5)
    assert abs(np.hypot(sim.vel[0], sim.vel[2]) - config.WALK_SPEED) < 0.05, "walk by default"
    sim.tap("CAPS")
    assert sim.player.scheme.run_mode
    sim.run(1.0)
    assert np.hypot(sim.vel[0], sim.vel[2]) >= config.JOG_SPEED - 0.05, "caps on = run (a jog at once)"
    sim.run(config.RUN_BUILD_TIME + 0.5)
    assert abs(np.hypot(sim.vel[0], sim.vel[2]) - config.SPRINT_SPEED) < 0.05, "a full sprint after the build-up"
    sim.tap("CAPS")
    assert not sim.player.scheme.run_mode
    sim.run(1.5)
    assert abs(np.hypot(sim.vel[0], sim.vel[2]) - config.WALK_SPEED) < 0.05, "caps off = walk"
    sim.teleport((0.0, 0.9, 0.0))         # back to the middle of the floor for the long sprint
    sim.hold("SHIFT")
    sim.run(config.RUN_BUILD_TIME + 1.0)
    assert abs(np.hypot(sim.vel[0], sim.vel[2]) - config.SPRINT_SPEED) < 0.05, "shift = momentary run"


def test_space_is_jump_on_open_ground_and_climb_at_a_wall():
    ls, obs, tower = _tower(2.2)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 6.0), scheme="flow")
    sim.tap("SPACE")
    assert sim.had_event("JUMP")
    sim.run(1.5)
    sim.hold("W")
    sim.run(3.0, until=lambda s: s.player.grab_candidate() is not None)
    sim.tap("SPACE")
    assert sim.events[-1][1] in ("GRAB", "VAULT"), "at a wall Space climbs instead of jumping"
    assert sim.run(4.0, until=lambda s: s.standing_on(tower), top_out=True)


def test_hold_space_to_catch_release_to_miss():
    """Letting go of the parkour button mid-air is a real (possibly fatal) miss."""
    ls, obs, tower = _tower(5.4)
    # Drop past the 3.8 crevasse with Space held: caught
    sim = Sim(ls, obs, spawn=(0.0, 3.9, -0.64), scheme="flow")
    sim.player.on_ground = False
    sim.hold("SPACE")
    assert sim.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert abs(sim.fsm.active_rail.height - 3.8) < 1e-3
    # Same fall without the button: nothing is caught, the player hits the ground
    sim = Sim(ls, obs, spawn=(0.0, 3.9, -0.64), scheme="flow")
    sim.player.on_ground = False
    sim.run(2.0, until=lambda s: s.player.on_ground)
    assert sim.player.on_ground and sim.state != ParkourState.LEDGE_HANG
    assert abs(sim.feet_y) < 0.01


def test_tap_only_jump_at_wall_neither_kicks_nor_grabs():
    ls, obs, tower = _tower(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 14.0), scheme="flow")
    sim.tap("CAPS")
    sim.hold("W")
    sim.run(5.0, until=lambda s: s.pos[2] < -1.0 + 0.3 + 1.8)
    sim.tap("SPACE")
    sim.run(2.0)
    evs = [e for _, e in sim.events]
    assert evs == ["JUMP"], evs
    assert sim.player.on_ground


def test_held_space_fast_route_kick_catch_climb():
    ls, obs, tower = _tower(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 14.0), scheme="flow")
    sim.tap("CAPS")
    sim.hold("W")
    sim.run(5.0, until=lambda s: s.pos[2] < -1.0 + 0.3 + 1.8)
    sim.hold("SPACE")
    t0 = sim.time
    assert sim.run(8.0, until=lambda s: s.standing_on(tower), top_out=True)
    evs = [e for _, e in sim.events]
    for e in ("JUMP", "WALL_KICK", "GRAB", "CLIMB_UP_FAST", "MANTLE_FAST", "MANTLE_DONE"):
        assert e in evs, evs
    assert sim.time - t0 < 5.5


def test_stale_held_space_does_not_dyno_on_catch():
    ls, obs, tower = _tower(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 3.9, -0.64), scheme="flow")
    sim.player.on_ground = False
    sim.hold("SPACE")
    assert sim.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.run(0.5)
    assert sim.state == ParkourState.LEDGE_HANG and not sim.had_event("DYNO_UP")


def test_hanging_push_w_climbs_without_modifier():
    ls, obs, tower = _tower(5.4)
    sim = _flow_hang(ls, obs)
    sim.hold("W")
    assert sim.run_until_event("CLIMB_UP_DONE", 3.0)
    assert abs(sim.fsm.active_rail.height - 3.8) < 1e-3
    assert sim.run(5.0, until=lambda s: s.standing_on(tower), top_out=True)


def test_hanging_shimmy_without_modifier():
    ls, obs, tower = _tower(5.4)
    sim = _flow_hang(ls, obs)
    x0 = float(sim.pos[0])
    sim.hold("D")
    sim.run(0.5)
    assert sim.state == ParkourState.SHIMMY and float(sim.pos[0]) - x0 > 0.4


def test_hanging_space_dynos_in_held_direction():
    ls, obs, tower = _tower(5.4)
    sim = _flow_hang(ls, obs)
    sim.tap("SPACE")
    assert sim.events[-1][1] == "DYNO_UP"

    sim = _flow_hang(ls, obs)
    sim.hold("D")
    sim.tap("SPACE")
    assert sim.events[-1][1] == "DYNO_LATERAL" and float(sim.vel[0]) > 0

    sim = _flow_hang(ls, obs)
    sim.hold("A")
    sim.tap("SPACE")
    assert sim.events[-1][1] == "DYNO_LATERAL" and float(sim.vel[0]) < 0

    sim = _flow_hang(ls, obs)
    sim.hold("S")
    sim.tap("SPACE")
    assert sim.events[-1][1] == "WALL_EJECT"


def test_hanging_s_climbs_down_or_drops_and_ctrl_lets_go():
    ls, obs, tower = _tower(5.4)
    sim = _flow_hang(ls, obs)
    sim.hold("W")
    assert sim.run_until_event("CLIMB_UP_DONE", 3.0)
    sim.release("W")
    sim.run(0.3)
    sim.tap("S")
    assert sim.events[-1][1] == "CLIMB_DOWN"
    assert sim.run_until_event("CLIMB_UP_DONE", 3.0)
    sim.run(0.3)
    assert abs(sim.fsm.active_rail.height - 2.0) < 1e-3
    sim.tap("CTRL")
    assert sim.events[-1][1] == "DROP"
    sim.run(1.0)
    assert sim.player.on_ground


def test_coyote_time_allows_late_jump_off_an_edge():
    ls, obs = build_custom([(-3.0, 0.0, -6.0, 6.0, 3.0, 6.0)])   # platform top at y=3, south edge z=0
    for delay, expect in ((0.05, True), (0.30, False)):
        sim = Sim(ls, obs, spawn=(0.0, 3.9, -1.0), scheme="flow")
        sim.hold("W")     # walking toward +Z? forward is -Z at yaw 0 -> face +Z
        sim.face(np.array([0.0, 0.0, 1.0]))
        sim.run(3.0, until=lambda s: not s.player.on_ground)
        sim.run(delay)
        sim.tap("SPACE")
        jumped = sim.had_event("JUMP")
        assert jumped == expect, f"delay {delay}: jumped={jumped}"


def test_walking_only_steps_knee_high_running_flows_over_hip_high():
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 0.8, 0.6)])
    block = obs[1]
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -2.5), scheme="flow")
    sim.hold("W")
    sim.run(2.0)
    assert not sim.standing_on(block), "walking into a hip-high block should stop you"
    sim.tap("CAPS")
    assert sim.run(3.0, until=lambda s: s.standing_on(block)), "running flows over it"

    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 0.5, 0.6)])
    low = obs[1]
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -2.5), scheme="flow")
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.standing_on(low)), "a knee-high step is taken while walking"


def test_every_tutorial_object_is_climbable_with_the_parkour_button():
    """Same as the E-based climbability test, holding Space (reach) instead of E."""
    from test_tutorial_climbability import FACES, _slow_climb, _climbable_obstacles
    ls, obs = build_playground()
    failures = []
    for o in _climbable_obstacles(obs):
        ok = False
        for n in FACES:
            res = _slow_climb(ls, obs, o, n, timeout=16.0, keys=("W", "SPACE"))
            if res is None:
                continue
            if res:
                ok = True
                break
        if not ok:
            failures.append((tuple(o.bounds.min.round(1)), tuple(o.bounds.size.round(1))))
    assert not failures, f"unclimbable with the parkour button: {failures}"


def test_top_edge_is_climbed_by_holding_w_and_hung_from_when_w_is_released():
    """Catching a top edge ends in a hang; W held carries on over it after the settle, W released
    leaves the player hanging until W is pressed again."""
    ls, obs, tower = _tower(2.2)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 8.0), scheme="flow")
    sim.tap("CAPS")
    sim.hold("W")
    sim.run(4.0, until=lambda s: s.pos[2] < -1 + 0.3 + 1.6)
    sim.hold("SPACE")                      # W and Space both held through the catch
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert sim.run(1.0, until=lambda s: s.had_event("MANTLE_FAST")), "W held: over the top"
    assert sim.run(3.0, until=lambda s: s.standing_on(tower))
    # and with W released at the catch the player hangs
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 8.0), scheme="flow")
    sim.tap("CAPS")
    sim.hold("W")
    sim.run(4.0, until=lambda s: s.pos[2] < -1 + 0.3 + 1.6)
    sim.hold("SPACE")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "SPACE")
    sim.run(1.5)
    assert sim.state == ParkourState.LEDGE_HANG and not sim.had_event("MANTLE") and not sim.had_event("MANTLE_FAST")
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.standing_on(tower))
