"""Locomotion & jumping calibrated to an elite parkour athlete."""
import numpy as np
from harness import *  # noqa
from parkour.state_machine import ParkourStateMachine


def _flat():
    return build_custom([])


def test_tap_jump_rise_is_a_low_hop():
    ls, obs = _flat()
    sim = Sim(ls, obs)
    sim.run(0.2)
    sim.tap("SPACE")
    sim.run(1.5, until=lambda s: s.player.on_ground and s.time > 0.4)
    rise = sim.max_height_reached() - 0.9
    assert 0.35 <= rise <= 0.60, f"tap hop rise {rise:.2f} m"


def test_held_jump_rise_matches_elite_standing_vertical():
    ls, obs = _flat()
    sim = Sim(ls, obs)
    sim.run(0.2)
    sim.hold("SPACE")
    sim.run(1.5, until=lambda s: s.player.on_ground and s.time > 0.4)
    rise = sim.max_height_reached() - 0.9
    # Elite standing vertical jump (centre of mass) is ~0.8-1.0 m; the old tuning produced 1.49 m
    assert 0.80 <= rise <= 1.05, f"held jump rise {rise:.2f} m"


def test_sprint_acceleration_is_human():
    ls, obs = _flat()
    sim = Sim(ls, obs)
    sim.hold("W", "SHIFT")
    t_walk = None
    t_jog = None
    t_sprint = None
    while sim.time < 6.0:
        sim.step()
        sp = float(np.hypot(sim.vel[0], sim.vel[2]))
        if t_walk is None and sp >= config.WALK_SPEED * 0.95:
            t_walk = sim.time
        if t_jog is None and sp >= config.JOG_SPEED * 0.98:
            t_jog = sim.time
        if t_sprint is None and sp >= config.SPRINT_SPEED * 0.98:
            t_sprint = sim.time
            break
    assert t_walk is not None and t_walk <= 0.5
    assert t_jog is not None and t_jog <= 0.8, "run mode is a jog almost at once"
    # Old code reached 9.5 m/s in 6 frames (~10 g). Top speed takes a few seconds of hard running.
    assert t_sprint is not None and config.RUN_BUILD_TIME * 0.8 <= t_sprint <= config.RUN_BUILD_TIME + 1.0, f"0 -> sprint in {t_sprint}"
    assert config.SPRINT_SPEED <= 9.0, "top speed beyond a trained athlete on a short run-up"


def test_top_speed_never_exceeds_sprint_speed():
    ls, obs = _flat()
    sim = Sim(ls, obs)
    sim.hold("W", "SHIFT")
    sim.run(3.0)
    sp = float(np.hypot(sim.vel[0], sim.vel[2]))
    assert sp <= config.SPRINT_SPEED + 0.05


def test_running_long_jump_distance_is_plausible():
    ls, obs = _flat()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 20.0))
    sim.hold("W", "SHIFT")
    sim.run(3.0)
    z0 = float(sim.pos[2])
    sim.hold("SPACE")
    sim.run(0.05)
    sim.run(3.0, until=lambda s: s.player.on_ground)
    dist = z0 - float(sim.pos[2])
    # A running jump of 4-6 m is the range for elite parkour practitioners (long-jump WR is ~9 m);
    # forward leaps carry JUMP_FORWARD_BOOST (+15 %) more momentum for a game feel a little above that
    boost = config.JUMP_FORWARD_BOOST
    assert 3.5 * boost <= dist <= 6.5 * boost, f"running long jump {dist:.2f} m"


def test_standing_long_jump_distance_is_plausible():
    ls, obs = _flat()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 20.0))
    sim.hold("W")
    sim.run(0.05)                       # a single step, no run-up
    z0 = float(sim.pos[2])
    sim.hold("SPACE")
    sim.run(3.0, until=lambda s: s.player.on_ground and s.time > 0.3)
    dist = z0 - float(sim.pos[2])
    assert dist <= 3.8, f"standing jump {dist:.2f} m exceeds the human standing long jump record"


def test_air_control_is_limited():
    ls, obs = _flat()
    sim = Sim(ls, obs)
    sim.run(0.2)
    sim.hold("SPACE")
    sim.step(2)
    sim.release("SPACE")
    sim.hold("D")
    sim.run(2.0, until=lambda s: s.player.on_ground and s.time > 0.3)
    # side-stepping in mid-air from a standing jump should move well under a metre
    assert abs(float(sim.pos[0])) < 1.0


def _kick_rise(keys, run_up):
    ls, obs = build_custom([(-3.0, 0.0, -6.0, 6.0, 8.0, 2.0, False)])   # bare wall: nothing to catch
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 12.0))
    sim.hold(*keys)
    sim.run(5.0, until=lambda s: s.pos[2] < -4.0 + 0.3 + run_up)
    approach = _speed(sim)
    sim.hold("SPACE")
    sim.run(2.5, until=lambda s: s.player.on_ground and s.time > 0.5)
    kicks = [e for _, e in sim.events if e == "WALL_KICK"]
    assert len(kicks) == 1, "one plant per jump"
    return sim.max_height_reached() - 0.9, approach


def test_wall_kick_adds_height_once_per_jump():
    # From a walking leap the plant is the classic wall climb-up: hands around a 4 m wall
    rise, v = _kick_rise(("W",), 1.2)
    assert abs(v - config.WALK_SPEED) < 0.1
    reach = rise + config.STAND_REACH
    assert 1.3 <= rise <= 1.8, f"walking wall kick body rise {rise:.2f} m"
    assert 3.5 <= reach <= 4.1, f"walking wall kick hand reach {reach:.2f} m"


def test_a_sprint_into_a_wall_runs_up_it():
    """Wall run: the faster you hit the wall, the more of the run-up goes into height.  A full
    sprint reaches the first-storey holds (3.8 m crevasse, 4.0 m window course) with margin."""
    walk, _ = _kick_rise(("W",), 1.2)
    sprint, v = _kick_rise(("W", "SHIFT"), 1.8)
    assert v > config.SPRINT_SPEED - 0.6, f"run-up reached {v:.2f} m/s"
    reach = sprint + config.STAND_REACH
    assert sprint >= walk + 0.5, (walk, sprint)
    assert 4.4 <= reach <= 5.3, f"sprint wall run hand reach {reach:.2f} m"
    extra = config.WALL_KICK_MAX_VY + config.WALL_RUN_EXTRA_VY
    assert config.rise_height(extra) <= 2.5, "the wall run is a strong pop, not a flight"


def test_no_wall_kick_when_falling_fast_or_walking():
    ls, obs = build_custom([(-3.0, 0.0, -6.0, 6.0, 8.0, 2.0)])
    # Walking into the wall: no kick
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -2.0))
    sim.hold("W")
    sim.run(2.0)
    assert not sim.had_event("WALL_KICK")
    # Falling onto the wall from above while descending fast: no kick
    sim = Sim(ls, obs, spawn=(0.0, 7.0, 0.0))
    sim.player.on_ground = False
    sim.vel[:] = [0.0, -4.0, -6.0]
    sim.hold("W")
    sim.run(1.5)
    assert not sim.had_event("WALL_KICK")


def test_friction_stops_player_quickly():
    ls, obs = _flat()
    sim = Sim(ls, obs)
    sim.hold("W")
    sim.run(1.0)
    sim.release("W")
    sim.run(0.6)
    assert float(np.hypot(sim.vel[0], sim.vel[2])) < 0.05


# ------------------------------------------------------------ run momentum
def _speed(sim):
    return float(np.hypot(sim.vel[0], sim.vel[2]))


def test_run_mode_is_a_jog_at_once_and_a_sprint_only_after_a_build_up():
    ls, obs = _flat()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 40.0))
    sim.hold("W", "SHIFT")
    sim.run(0.5)
    assert config.JOG_SPEED - 0.1 <= _speed(sim) <= config.JOG_SPEED + 0.4, "a jog almost at once"
    sim.run(config.RUN_BUILD_TIME * 0.5)
    mid = _speed(sim)
    assert config.JOG_SPEED + 0.3 < mid < config.SPRINT_SPEED - 0.3, "half-way through the build-up"
    sim.run(config.RUN_BUILD_TIME * 0.5 + 0.5)
    assert abs(_speed(sim) - config.SPRINT_SPEED) < 0.05, "full sprint after RUN_BUILD_TIME of running"


def test_sprint_build_up_is_lost_on_a_sharp_turn_and_drains_when_easing_off():
    ls, obs = _flat()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 40.0))
    sim.hold("W", "SHIFT")
    sim.run(config.RUN_BUILD_TIME + 0.5)
    assert abs(_speed(sim) - config.SPRINT_SPEED) < 0.05
    sim.player.yaw += 120.0                  # cut hard across the line of travel
    sim.run(0.3)
    assert sim.fsm.run_time < 0.35, "a sharp change of direction throws the build-up away"
    assert _speed(sim) < config.SPRINT_SPEED - 0.5
    sim.run(config.RUN_BUILD_TIME + 0.5)
    assert abs(_speed(sim) - config.SPRINT_SPEED) < 0.05, "and it builds again along the new line"
    sim.release("SHIFT")                     # ease off to a walk ...
    sim.run(1.0)
    assert abs(_speed(sim) - config.WALK_SPEED) < 0.05
    assert sim.fsm.run_time < config.RUN_BUILD_TIME - 1.5, "... and the build-up drains"
    sim.hold("SHIFT")
    sim.run(0.5)
    assert _speed(sim) < config.SPRINT_SPEED - 0.8, "picking the run back up starts from a jog again"


def test_hanging_resets_the_sprint_build_up():
    ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 2.2, 4.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 30.0))
    sim.hold("W", "SHIFT")
    sim.run(config.RUN_BUILD_TIME + 0.3)
    assert sim.fsm.run_time > config.RUN_BUILD_TIME * 0.8
    sim.run(6.0, until=lambda s: s.pos[2] < -1.0 + 0.3 + 1.6)
    sim.release("W")                     # catch the edge and hang there (no top-out)
    sim.hold("SPACE")
    assert sim.run(4.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.run(0.2)
    assert sim.fsm.run_time == 0.0, "hanging on a wall is not running"


def test_view_widens_with_actual_speed_not_a_key():
    ls, obs = _flat()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 40.0))
    sim.hold("W")
    sim.run(1.0)
    assert abs(sim.fsm.target_fov - config.BASE_FOV) < 0.5, "walking: base view"
    sim.hold("SHIFT")
    sim.run(0.8)
    jog_fov = sim.fsm.target_fov
    assert config.BASE_FOV + 2.0 < jog_fov < config.SPRINT_FOV - 2.0, "jog: part-way"
    sim.run(config.RUN_BUILD_TIME + 0.5)
    assert abs(sim.fsm.target_fov - config.SPRINT_FOV) < 0.5, "sprint: fully widened"
    sim.hold("SPACE")
    sim.run(0.3)
    assert not sim.player.on_ground
    assert sim.fsm.target_fov > config.SPRINT_FOV - 3.0, "momentum carries the wide view through the leap"


# ------------------------------------------------------------ jump momentum
def _leap(prep, hold=True):
    ls, obs = _flat()
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 30.0))
    prep(sim)
    start = sim.pos.copy()
    v0 = _speed(sim)
    if hold:
        sim.hold("SPACE")
    else:
        sim.tap("SPACE")
    sim.run(0.05)
    sim.run(3.0, until=lambda s: s.player.on_ground and s.time > 0.3)
    d = sim.pos - start
    return float(np.hypot(d[0], d[2])), v0, sim


def test_standing_long_jump_is_a_real_leap_not_a_hop():
    dist, v0, sim = _leap(lambda s: (s.run(0.2), s.hold("W")))
    assert v0 < 0.1
    assert 2.2 <= dist <= 3.2, f"standing long jump {dist:.2f} m"
    assert float(sim.pos[2]) < 30.0, "forward, the held direction"
    dist_tap, _, _ = _leap(lambda s: (s.run(0.2), s.hold("W")), hold=False)
    assert 1.2 <= dist_tap < dist - 0.5, f"a tapped hop is shorter ({dist_tap:.2f} m)"


def test_jump_distance_grows_with_run_up_momentum():
    stand, _, _ = _leap(lambda s: (s.run(0.2), s.hold("W")))
    step, _, _ = _leap(lambda s: (s.hold("W"), s.run(0.05)))
    walk, _, _ = _leap(lambda s: (s.hold("W"), s.run(1.0)))
    jog, v_jog, _ = _leap(lambda s: (s.hold("W", "SHIFT"), s.run(0.6)))
    sprint, v_sprint, _ = _leap(lambda s: (s.hold("W", "SHIFT"), s.run(config.RUN_BUILD_TIME + 0.5)))
    assert abs(v_jog - config.JOG_SPEED) < 0.3 and abs(v_sprint - config.SPRINT_SPEED) < 0.1
    assert stand <= step <= walk <= jog <= sprint, (stand, step, walk, jog, sprint)
    boost = config.JUMP_FORWARD_BOOST            # forward leaps: +15 % momentum out of the take-off
    assert 3.5 * boost <= walk <= 4.6 * boost, f"walking leap {walk:.2f} m"
    assert 4.0 * boost <= jog <= 5.3 * boost, f"jogging leap {jog:.2f} m"
    assert 5.5 * boost <= sprint <= 6.5 * boost, f"full-sprint leap {sprint:.2f} m (elite running long jump range)"
    assert sprint - jog < 1.5, "at speed the legs add little: the leap is carried by momentum"


def test_backward_and_sideways_standing_leaps():
    back, _, sim = _leap(lambda s: (s.run(0.2), s.hold("S")))
    assert back >= 1.5, f"backward standing leap {back:.2f} m"
    assert float(sim.pos[2]) > 30.0, "away from what you are facing"
    side, _, sim = _leap(lambda s: (s.run(0.2), s.hold("D")))
    assert side >= 1.5 and float(sim.pos[0]) > 0.5
    none, _, _ = _leap(lambda s: s.run(0.2))
    assert none < 0.05, "no direction held: a straight vertical jump"


def test_take_off_push_never_beats_sprint_momentum():
    best = 0.0
    for prep in (lambda s: (s.run(0.2), s.hold("W")),
                 lambda s: (s.hold("W", "SHIFT"), s.run(config.RUN_BUILD_TIME + 0.5))):
        ls, obs = _flat()
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 30.0))
        prep(sim)
        sim.hold("SPACE")
        sim.step()
        best = max(best, _speed(sim))
    # (the forward boost scales sprint momentum and push alike: the legs still never beat momentum)
    assert best <= (config.SPRINT_SPEED + config.JUMP_PUSH_SPRINT) * config.JUMP_FORWARD_BOOST + 0.05


def test_leap_off_a_roof_and_backward_away_from_a_wall():
    # Standing at the edge of a 4 m roof: W + Space clears a 1.5 m gap
    ls, obs = build_custom([(-4.0, 0.0, -8.0, 8.0, 4.0, 8.0), (-4.0, 0.0, -12.0, 8.0, 4.0, 2.5)])
    sim = Sim(ls, obs, spawn=(0.0, 4.9, -7.6), yaw=0.0)
    sim.run(0.3)
    assert sim.player.on_ground
    sim.hold("W", "SPACE")
    sim.run(0.05)
    assert sim.run(2.0, until=lambda s: s.player.on_ground and s.time > 0.6)
    sim.release("W", "SPACE")
    sim.run(0.2)
    assert sim.standing_on(obs[2]), "landed on the far roof"
    # Facing a wall: S + Space leaps backward away from it
    ls, obs = build_custom([(-4.0, 0.0, -4.0, 8.0, 4.0, 2.0)])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -1.3), yaw=0.0)
    sim.run(0.3)
    sim.hold("S", "SPACE")
    sim.run(0.05)
    sim.run(2.0, until=lambda s: s.player.on_ground and s.time > 0.6)
    assert float(sim.pos[2]) >= -1.3 + 1.5


def test_forward_leaps_carry_the_forward_boost_and_nothing_else_changes():
    """Forward jumps leave the ground with JUMP_FORWARD_BOOST x the horizontal momentum; backward /
    sideways leaps, straight-up jumps and the jump height are untouched."""
    def take_off(prep, keys=("SPACE",)):
        ls, obs = _flat()
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 30.0))
        prep(sim)
        before = _speed(sim)
        sim.hold(*keys)
        sim.step()
        return before, _speed(sim), sim
    boost = config.JUMP_FORWARD_BOOST
    for prep in (lambda s: (s.hold("W"), s.run(1.0)),
                 lambda s: (s.hold("W", "SHIFT"), s.run(config.RUN_BUILD_TIME + 0.5))):
        v0, v1, sim = take_off(prep)
        push = ParkourStateMachine.jump_push(v0)
        assert abs(v1 - (v0 + push) * boost) < 0.05, (v0, v1, push)
        assert abs(float(sim.vel[1]) - config.JUMP_IMPULSE) < 0.3, "height unchanged"
    # backward leap: no boost
    v0, v1, _ = take_off(lambda s: (s.run(0.2), s.hold("S")))
    assert abs(v1 - ParkourStateMachine.jump_push(0.0)) < 0.05, "backward leaps are not boosted"
    # straight up: still straight up
    v0, v1, _ = take_off(lambda s: s.run(0.2))
    assert v1 < 1e-3
