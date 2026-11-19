"""Climbing with the legs: crevasse stand-ups, mantles, dynos and ejects."""
import numpy as np
from harness import *  # noqa
from parkour.ledge_rail import RailType, LedgeRail


def _tower(h=5.4):
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, h, 4.0)])
    return ls, obs, obs[1]


def _hang_on_head_crevasse(ls, obs, sprint=False):
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.3)
    assert sim.fsm.active_rail.rail_type == RailType.HEAD_CREVASSE
    return sim


def test_crevasse_to_crevasse_uses_legs_and_takes_a_quick_but_real_time():
    ls, obs, tower = _tower()
    sim = _hang_on_head_crevasse(ls, obs)
    t0 = sim.time
    y0 = float(sim.pos[1])
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.had_event("CLIMB_UP_DONE"))
    sim.release("W")
    dt = sim.time - t0
    assert ParkourState.CLIMB_UP in sim.states
    # (a brisk athletic climb - quicker than the original 'human time' tuning, never a snap)
    assert 0.7 <= dt <= 1.4, f"2.0 -> 3.8 m climb took {dt:.2f} s"
    assert sim.fsm.active_rail.rail_type == RailType.JUMP_CREVASSE
    assert abs((float(sim.pos[1]) - y0) - 1.8) < 0.05
    # Legs did the work: after the pull-up (feet ~0.4 m) the body keeps rising with the
    # foot on the 2.0 m lip until the 3.8 m hold is within one arm's reach
    peak_feet = max(BodyModel.feet_y(tr["pos"]) for tr in sim.trace if tr["state"] == ParkourState.CLIMB_UP)
    assert peak_feet >= 1.4, f"climb never pushed up from the lip (peak feet {peak_feet:.2f})"
    assert sim.max_step < 0.15, "root motion teleported"


def test_climb_is_blocked_when_next_hold_is_beyond_reach():
    """A single crevasse at 2.0 m under a 4.6 m top: 2.6 m gap > what a lip stand-up can reach."""
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 4.6, 4.0, False)])
    wall = obs[1]
    crev = LedgeRail([-2.0, 2.0, -1.0], [2.0, 2.0, -1.0], [0, 0, 1], RailType.HEAD_CREVASSE, owner=wall)
    wall.rails.append(crev)
    ls.add_rail(crev)
    sim = _hang_on_head_crevasse(ls, obs)
    sim.hold("W")
    sim.run(2.0)
    assert sim.state == ParkourState.LEDGE_HANG
    assert sim.had_event("CLIMB_BLOCKED")
    assert not sim.had_event("CLIMB_UP")


def test_crevasse_below_top_reaches_the_edge_then_w_held_carries_on_over_it():
    ls, obs, wall = _tower(3.2)   # crevasse at 2.0, top at 3.2
    sim = _hang_on_head_crevasse(ls, obs)
    sim.hold("W")
    assert sim.run_until_event("CLIMB_UP_DONE", 4.0)
    t_edge = sim.time
    assert sim.fsm.active_rail.rail_type == RailType.MANTLE_TOP, "the climb ends hanging from the top edge"
    assert sim.run_until_event("MANTLE", 1.0), "W still held: straight over the top, no second press"
    assert sim.time - t_edge <= config.GRAB_SETTLE_TIME + 0.05, "after the same short settle every hold gets"
    assert sim.run(4.0, until=lambda s: s.standing_on(wall))
    evs = [e for _, e in sim.events]
    assert evs.index("CLIMB_UP") < evs.index("MANTLE")


def test_releasing_w_at_the_top_edge_hangs_there_until_w_is_pressed_again():
    ls, obs, wall = _tower(3.2)
    sim = _hang_on_head_crevasse(ls, obs)
    sim.hold("W")
    assert sim.run_until_event("CLIMB_UP", 1.0)
    sim.release("W")                       # let go of W during the climb: it completes to the edge ...
    assert sim.run_until_event("CLIMB_UP_DONE", 4.0)
    sim.run(1.0)
    assert sim.state == ParkourState.LEDGE_HANG and not sim.had_event("MANTLE"), "... and hangs there"
    sim.hold("W")
    assert sim.run(4.0, until=lambda s: s.standing_on(wall))


def test_mantle_from_dead_hang_is_slow_and_running_mantle_is_fast():
    ls, obs, wall = _tower(2.2)
    # Slow: walk up, grab, hang, then a deliberate W press
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.5)
    sim.hold("W")
    t0 = sim.time
    assert sim.run(3.0, until=lambda s: s.had_event("MANTLE_DONE"))
    slow = sim.time - t0
    assert 0.65 <= slow <= 1.2, f"dead-hang mantle {slow:.2f} s"
    assert sim.standing_on(wall)

    # Fast: sprint + jump at the wall with Space held -> catch the edge (hang), then W while running tops out quickly
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 8.0))
    sim.hold("W", "SHIFT")
    sim.run(4.0, until=lambda s: s.pos[2] < -1 + 0.3 + 1.6)
    sim.hold("SPACE")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG), "catching the edge ends in a hang"
    t_catch = sim.time
    assert sim.run(1.0, until=lambda s: s.had_event("MANTLE_FAST")), "W still held: straight over after the settle"
    assert sim.time - t_catch >= config.GRAB_SETTLE_TIME - 1e-6, "the catch settles first"
    t0 = sim.time
    assert sim.run(3.0, until=lambda s: s.had_event("MANTLE_DONE"))
    fast = sim.time - t0
    assert 0.3 <= fast <= 0.6, f"running mantle {fast:.2f} s"
    assert fast < slow
    assert sim.standing_on(wall) or (sim.player.on_ground and sim.feet_y > 2.1)


def test_mantle_lands_standing_on_top_surface_with_forward_flow():
    ls, obs, wall = _tower(2.2)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(5.0, until=lambda s: s.had_event("MANTLE_DONE"), top_out=True)
    assert abs(sim.feet_y - 2.2) < 0.05
    assert float(sim.vel[2]) < -0.5, "mantle should exit with forward momentum onto the surface"
    sim.run(0.5)
    assert sim.player.on_ground and sim.standing_on(wall)


def test_mantle_is_refused_when_top_is_blocked():
    # A slab sits right on top of the wall's edge: no room to stand on the 2.2 m top, so the
    # edge is only a hold - the climber stands on its lip and reaches the slab's top instead.
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 2.2, 4.0), (-2.0, 2.2, -5.0, 4.0, 2.0, 4.0, False)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    sim.run(6.0, until=lambda s: s.had_event("MANTLE_DONE"), top_out=True)
    evs = [e for _, e in sim.events]
    assert "VAULT" not in evs
    if "MANTLE" in evs:
        assert "CLIMB_UP" in evs and evs.index("CLIMB_UP") < evs.index("MANTLE")
        assert abs(sim.feet_y - 4.2) < 0.05, "must end on the slab, never inside it"
    for tr in sim.trace:
        if tr["state"] in (ParkourState.GROUND, ParkourState.SPRINT):
            assert not (2.15 < BodyModel.feet_y(tr["pos"]) < 4.15), "stood where the slab is"


def test_dyno_up_rise_is_a_leg_push_not_a_rocket():
    ls, obs, tower = _tower()
    sim = _hang_on_head_crevasse(ls, obs)
    y0 = float(sim.pos[1])
    sim.release_all()
    sim.hold("W", "E")
    sim.tap("SPACE")
    assert sim.had_event("DYNO_UP")
    sim.release_all()
    sim.run(1.5, until=lambda s: s.player.on_ground)
    rise = sim.max_height_reached() - y0
    assert 0.75 <= rise <= 1.05, f"dyno body rise {rise:.2f} m"
    # The hands could reach at most ~1.2 m above the old hold: not enough for a 1.8 m gap without feet
    assert rise + config.MAX_GRAB_REACH - (config.HANG_EYE_OFFSET + (config.PLAYER_EYE_HEIGHT - config.SHOULDER_HEIGHT)) < 1.8


def test_dyno_lockout_prevents_regrabbing_same_rail_instantly():
    ls, obs, tower = _tower()
    sim = _hang_on_head_crevasse(ls, obs)
    sim.release_all()
    sim.hold("W", "E")
    sim.tap("SPACE")
    sim.run(config.GRAB_LOCKOUT_TIME * 0.8)
    assert sim.state != ParkourState.LEDGE_HANG


def test_wall_eject_speed_and_turn():
    ls, obs, tower = _tower()
    sim = _hang_on_head_crevasse(ls, obs)
    yaw0 = sim.player.yaw
    sim.release_all()
    sim.hold("S", "E")
    sim.tap("SPACE")
    assert sim.had_event("WALL_EJECT")
    v = sim.vel.copy()
    horiz = float(np.hypot(v[0], v[2]))
    assert horiz <= 5.0, f"wall eject {horiz:.1f} m/s horizontal exceeds a leg push from a hang"
    assert v[1] <= 3.5
    same_height_range = 2.0 * v[1] * horiz / abs(config.GRAVITY)
    assert same_height_range <= 2.0
    sim.release_all()
    sim.run(config.WALL_EJECT_TURN_TIME + 0.1)
    assert abs(((sim.player.yaw - yaw0 + 180) % 360) - 180) > 170, "eject must turn the player to face the landing"


def test_lateral_dyno_is_a_human_cat_leap():
    ls, obs, tower = _tower()
    sim = _hang_on_head_crevasse(ls, obs)
    sim.release_all()
    sim.hold("D", "E")
    sim.tap("SPACE")
    assert sim.had_event("DYNO_LATERAL")
    v = sim.vel.copy()
    lateral = abs(float(np.dot(v, sim.fsm.ledge_system.rails[0].tangent))) if False else float(np.hypot(v[0], v[2]))
    assert lateral <= 5.5
    same_height_range = 2.0 * v[1] * lateral / abs(config.GRAVITY)
    assert same_height_range <= 2.6, "same-height lateral leap beyond a real wall-to-wall cat leap"
    assert abs(sim.fsm.dutch_tilt) > 1.0, "camera should roll into the leap"


def test_root_motion_velocity_is_bounded():
    ls, obs, tower = _tower()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(8.0, until=lambda s: s.standing_on(tower), top_out=True)
    for tr in sim.trace:
        if tr["state"] in (ParkourState.CLIMB_UP, ParkourState.MANTLE_VAULT, ParkourState.LEDGE_HANG):
            assert tr["step"] / config.FIXED_DT <= 6.0, "body moved faster than 6 m/s during a climb"


def test_climb_down_steps_to_lower_hold_and_drop_only_at_bottom():
    ls, obs, tower = _tower()
    sim = _hang_on_head_crevasse(ls, obs)
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.had_event("CLIMB_UP_DONE"))
    sim.release("W")
    sim.run(0.3)
    assert abs(sim.fsm.active_rail.height - 3.8) < 1e-3
    y_top = float(sim.pos[1])
    sim.hold("E")
    sim.hold("S")
    sim.step()
    sim.release("S")
    assert sim.had_event("CLIMB_DOWN")
    assert sim.run_until_event("CLIMB_UP_DONE", 3.0)
    assert abs(sim.fsm.active_rail.height - 2.0) < 1e-3
    assert abs((y_top - float(sim.pos[1])) - 1.8) < 0.05
    # feet took the weight on the way down (controlled, leg-driven descent: no free fall)
    feet = [BodyModel.feet_y(tr["pos"]) for tr in sim.trace if tr["state"] == ParkourState.CLIMB_UP]
    assert max(feet) >= 1.4
    assert sim.max_step < 0.15
    # from the lowest hold S+E is a plain drop
    sim.run(0.3)
    sim.hold("S")
    sim.step()
    assert sim.had_event("DROP")
