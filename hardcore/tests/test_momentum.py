"""Momentum carried through the moves that used to stop a run dead.

* Wall run: a sprint into a wall converts the run-up into height (the first-storey holds are caught).
* Climb flow: the speed you meet a wall with (and the rhythm of chaining holds with W) makes the
  climb quicker; topping out of a flowing climb sends you off at a run.
* Speed vaults: a vault / step-up taken at a run keeps the run going.
* Root motion flows through its keyframes (no dead stop at each) and the arms keep time with it.
"""
import numpy as np
import pytest
from harness import *  # noqa
from parkour.state_machine import ParkourStateMachine

L1, L2 = config.UPPER_ARM_LEN, config.FOREARM_LEN


def _speed(sim):
    return float(np.hypot(sim.vel[0], sim.vel[2]))


def _tower(h):
    ls, obs = build_custom([(-4.0, 0.0, -5.0, 8.0, h, 6.0)])     # face at z = 1
    return ls, obs, obs[1]


def _run_at_wall(ls, obs, mode, with_arms=False, run_up=25.0):
    """Approach the z = 1 wall face walking / jogging / sprinting and take off with Space held."""
    sim = Sim(ls, obs, spawn=(0.0, 0.9, run_up), with_arms=with_arms)
    sim.set_pitch(10.0)
    if mode in ("jog", "sprint"):
        sim.hold("CAPS")
    sim.hold("W")
    if mode == "jog":
        sim.run(0.6)
        sim.player.pos[2] = 4.0                    # (no time to build the sprint)
    sim.run(10.0, until=lambda s: s.pos[2] < 1.0 + 0.3 + 1.6)
    approach = _speed(sim)
    sim.repress("SPACE")
    return sim, approach


def _climb_time(h, mode):
    ls, obs, tower = _tower(h)
    sim, approach = _run_at_wall(ls, obs, mode)
    t0 = sim.time
    assert sim.run(30.0, until=lambda s: s.had_event("MANTLE_DONE")), f"{mode}: never topped out"
    return sim.time - t0, sim, t0, approach


# ------------------------------------------------------------------ wall run + climb flow
def test_a_sprint_into_a_building_climbs_it_faster_than_a_jog_or_a_walk():
    walk, _, _, v_walk = _climb_time(12.0, "walk")
    jog, _, _, v_jog = _climb_time(12.0, "jog")
    sprint, sim, t0, v_sprint = _climb_time(12.0, "sprint")
    assert v_walk < v_jog < v_sprint
    assert sprint < jog < walk, (walk, jog, sprint)
    assert sprint < 3.2, f"a 12 m wall from a full sprint took {sprint:.2f} s"
    assert walk > 1.5 * sprint, "the run-up is worth a lot"
    # the wall run catches the first-storey hold straight away - no falling back to the 2 m one
    grabs = [t for t, e in sim.events if e == "GRAB" and t >= t0]
    assert grabs and grabs[0] - t0 < 0.45
    assert sim.max_root_step < 0.2, "quick, never a snap"


def test_the_wall_run_catches_the_first_storey_hold_from_a_sprint():
    ls, obs, tower = _tower(12.0)
    sim, _ = _run_at_wall(ls, obs, "sprint")
    sim.step()                                     # (take off with W held: a forward leap)
    sim.release("W")                               # catch and hang there
    assert sim.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert abs(sim.fsm.active_rail.height - 3.8) < 0.01, f"caught the {sim.fsm.active_rail.height:.1f} m hold"
    assert sim.fsm.climb_flow > 0.9, "the sprint is carried into the climb"


def test_a_walking_leap_at_the_wall_still_catches_only_what_a_kick_reaches():
    ls, obs, tower = _tower(12.0)
    sim, _ = _run_at_wall(ls, obs, "walk")
    sim.step()
    sim.release("W")
    assert sim.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert sim.fsm.active_rail.height < 3.9
    assert sim.fsm.climb_flow < 0.4


def test_topping_out_of_a_flowing_climb_leaves_at_a_run():
    _, sim, _, _ = _climb_time(5.4, "sprint")
    assert _speed(sim) > config.JOG_SPEED - 1.0, f"left the edge at {_speed(sim):.2f} m/s"
    assert sim.fsm.run_time > 0.5 * config.RUN_BUILD_TIME, "the sprint build-up is restored"
    sim.run(0.6)
    assert sim.player.on_ground and sim.standing_on(sim.obstacles[1])
    assert _speed(sim) > config.JOG_SPEED, "and the run goes on across the roof"


def test_a_dead_hang_top_out_still_steps_off_slowly():
    ls, obs, wall = _tower(2.2)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 2.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "E")
    sim.run(0.5)
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.had_event("MANTLE_DONE"))
    assert _speed(sim) <= config.MANTLE_EXIT_SPEED * 0.5 + 0.05
    assert sim.fsm.run_time == 0.0


def test_climb_flow_drains_while_just_hanging():
    ls, obs, tower = _tower(12.0)
    sim, _ = _run_at_wall(ls, obs, "sprint")
    sim.step()
    sim.release("W")
    assert sim.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert sim.fsm.climb_flow > 0.9
    sim.run(1.0)
    assert sim.fsm.climb_flow == 0.0, "hanging about: the momentum of the run is gone"


def test_chaining_holds_at_a_run_builds_a_rhythm():
    ls, obs, tower = _tower(12.0)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 2.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W", "E")
    sim.run(1.0)
    assert sim.fsm.climb_flow == 0.0
    sim.hold("CAPS", "W")
    durations = []
    start = None
    for _ in range(int(8.0 / DT)):
        sim.step()
        if sim.fsm.last_event == "CLIMB_UP_FAST":
            start = sim.time
        elif sim.fsm.last_event == "CLIMB_UP_DONE" and start is not None:
            durations.append(sim.time - start)
            start = None
        if len(durations) >= 4:
            break
    assert len(durations) >= 4
    assert all(b <= a + 1e-6 for a, b in zip(durations, durations[1:])), durations
    assert durations[-1] < durations[0] * 0.9, durations
    assert sim.fsm.climb_flow <= config.CLIMB_FLOW_CHAIN_MAX + 1e-6, "rhythm alone never equals a sprint"


def test_flow_from_speed_spans_a_jog_to_a_sprint():
    lo, hi = config.CLIMB_FLOW_SPEED_RANGE
    assert ParkourStateMachine.flow_from_speed(0.0) == 0.0
    assert ParkourStateMachine.flow_from_speed(lo) == 0.0
    assert ParkourStateMachine.flow_from_speed(hi) == 1.0
    assert ParkourStateMachine.flow_from_speed(50.0) == 1.0
    assert 0.0 < ParkourStateMachine.flow_from_speed((lo + hi) * 0.5) < 1.0


def test_zombies_climb_slower_even_with_flow():
    ls, obs, tower = _tower(12.0)
    times = []
    for scale in (1.0, config.ZOMBIE_CLIMB_SCALE):
        sim, _ = _run_at_wall(ls, obs, "sprint")
        sim.player.set_speed_scales(1.0, scale)
        t0 = sim.time
        assert sim.run(30.0, until=lambda s: s.had_event("MANTLE_DONE"))
        times.append(sim.time - t0)
    assert times[1] > times[0] * 1.05, times


# ------------------------------------------------------------------ speed vaults
def _vault_run(height, keys, run_up=20.0):
    ls, obs = build_custom([(-3.0, 0.0, -2.0, 6.0, height, 1.2)])     # a car-sized block, face at z = -0.8
    sim = Sim(ls, obs, spawn=(0.0, 0.9, run_up))
    sim.hold(*keys)
    assert sim.run(8.0, until=lambda s: s.state == ParkourState.MANTLE_VAULT)
    before = float(sim.fsm.vault_speed)
    run_before = sim.fsm.run_time
    assert sim.run(2.0, until=lambda s: s.had_event("MANTLE_DONE"))
    return sim, before, run_before, obs


def test_a_sprint_over_a_car_keeps_the_run():
    sim, before, run_before, obs = _vault_run(1.0, ("W", "SHIFT"))
    assert before > config.SPRINT_SPEED - 0.6
    assert _speed(sim) >= config.VAULT_SPEED_KEEP * before - 0.05, f"came off the vault at {_speed(sim):.2f} m/s"
    assert sim.fsm.run_time >= run_before - 1e-6, "the sprint build-up is kept through the vault"
    assert sim.vel[2] < -config.JOG_SPEED, "still going the way the run was going"


def test_a_speed_vault_costs_little_time_in_a_chase():
    def time_to(z_end, blocks):
        ls, obs = build_custom(blocks)
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 30.0))
        sim.hold("W", "SHIFT")
        assert sim.run(12.0, until=lambda s: s.pos[2] < z_end)
        return sim.time
    clear = time_to(-15.0, [])
    vaulted = time_to(-15.0, [(-3.0, 0.0, -2.0, 6.0, 1.0, 1.2)])
    assert vaulted - clear < 0.35, f"the car cost {vaulted - clear:.2f} s"


def test_a_walking_step_up_is_not_faster_than_before():
    sim, before, _, _ = _vault_run(0.55, ("W",))
    assert before <= config.WALK_SPEED + 0.05
    assert _speed(sim) <= config.WALK_SPEED


# ------------------------------------------------------------------ root motion
def _fsm_with_keys(keys, start):
    ls, obs = build_custom([])
    fsm = ParkourStateMachine(ls, obs)
    fsm.motion_start = np.asarray(start, dtype=np.float32)
    fsm._start_root_motion(keys)
    return fsm


def _sample(fsm, t):
    fsm.motion_time = t
    return fsm._eval_root_motion(fsm.motion_start)


def test_root_motion_passes_every_key_and_starts_and_ends_at_rest():
    start = np.array([0.0, 0.0, 0.0])
    keys = [(0.15, [0.0, 0.4, 0.3]), (0.30, [0.0, 1.5, 0.3]), (0.10, [0.0, 1.4, 0.2])]
    fsm = _fsm_with_keys(keys, start)
    t = 0.0
    for dur, p in keys:
        t += dur
        assert np.allclose(_sample(fsm, t), p, atol=1e-5)
    eps = 1e-4
    assert np.linalg.norm(_sample(fsm, eps) - start) < 1e-5, "starts at rest"
    assert np.linalg.norm(_sample(fsm, fsm.motion_total - eps) - np.array(keys[-1][1])) < 1e-5, "ends at rest"


def test_root_motion_flows_through_interior_keys_without_overshoot():
    start = np.array([0.0, 0.0, 0.0])
    keys = [(0.15, [0.0, 0.4, 0.3]), (0.30, [0.0, 1.5, 0.3]), (0.10, [0.0, 1.4, 0.2])]
    fsm = _fsm_with_keys(keys, start)
    # continuous velocity through the first interior key, and it does not stop there (y keeps rising)
    h = 1e-3
    t1 = 0.15
    v_before = (_sample(fsm, t1) - _sample(fsm, t1 - h)) / h
    v_after = (_sample(fsm, t1 + h) - _sample(fsm, t1)) / h
    assert np.allclose(v_before, v_after, atol=0.05)
    assert v_after[1] > 1.0, "no dead stop at the end of the pull-up"
    # monotone: within each segment every axis stays between its two keys
    pts = [start] + [np.array(p) for _, p in keys]
    times = np.cumsum([0.0] + [d for d, _ in keys])
    for i in range(len(keys)):
        lo, hi = np.minimum(pts[i], pts[i + 1]) - 1e-5, np.maximum(pts[i], pts[i + 1]) + 1e-5
        for t in np.linspace(times[i], times[i + 1], 25):
            p = _sample(fsm, t)
            assert np.all(p >= lo) and np.all(p <= hi), (t, p)


def test_a_single_segment_move_is_a_plain_smoothstep():
    fsm = _fsm_with_keys([(0.5, [1.0, 2.0, 0.0])], [0.0, 0.0, 0.0])
    for u in (0.1, 0.3, 0.5, 0.8):
        k = u * u * (3.0 - 2.0 * u)
        assert np.allclose(_sample(fsm, 0.5 * u), np.array([1.0, 2.0, 0.0]) * k, atol=1e-5)


# ------------------------------------------------------------------ the arms keep up
def test_the_arms_keep_time_with_a_flowing_climb():
    ls, obs, tower = _tower(9.2)
    sim, _ = _run_at_wall(ls, obs, "sprint", with_arms=True)
    rates = []
    done = 0
    while sim.time < 12.0 and not sim.had_event("MANTLE_DONE"):
        sim.step()
        rates.append(sim.fsm.move_rate)
        L = sim.arm_rig.last
        for side in ("left", "right"):
            sh, el, wr = L[side + "_shoulder"], L[side + "_elbow"], L[side + "_wrist"]
            assert abs(np.linalg.norm(el - sh) - L1) < 1e-3
            assert abs(np.linalg.norm(wr - el) - L2) < 1e-3
            assert np.linalg.norm(wr - sh) <= L1 + L2 + 1e-3
            assert config.ELBOW_MIN_ANGLE_DEG - 0.01 <= L[side + "_elbow_angle"] <= config.ELBOW_MAX_ANGLE_DEG + 0.01
        if sim.fsm.last_event == "CLIMB_UP_DONE":
            done += 1
            rail = sim.fsm.active_rail
            assert any(r is rail for r in L["hand_rails"]), "a hand is on the new hold as the body arrives"
    assert sim.had_event("MANTLE_DONE") and done >= 2
    assert max(rates) > 1.8, "a flowing climb runs well above the nominal pace"


def test_the_legs_stay_anatomical_through_a_flowing_climb():
    from test_legs import _check_anatomy, _legs
    ls, obs, tower = _tower(9.2)
    sim, _ = _run_at_wall(ls, obs, "sprint", with_arms=True)
    while sim.time < 12.0 and not sim.had_event("MANTLE_DONE"):
        sim.step()
        _check_anatomy(sim)
        for i in range(2):
            assert not tower.bounds.contains_point(_legs(sim)["ankles"][i] + np.array([0.0, 0.02, 0.0])), "ankle inside the wall"
    assert sim.had_event("MANTLE_DONE") and ParkourState.CLIMB_UP in sim.states


def test_move_rate_is_nominal_on_the_ground_and_reflects_the_climb():
    ls, obs, tower = _tower(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 6.0))
    assert sim.fsm.move_rate == 1.0
    sim.hold("W", "E")
    assert sim.run(4.0, until=lambda s: s.state == ParkourState.CLIMB_UP)
    assert abs(sim.fsm.move_rate - 1.0 / sim.fsm.motion_scale) < 1e-6
    assert sim.fsm.move_rate >= 1.0


# ------------------------------------------------------------------ render interpolation
def test_frames_are_drawn_between_physics_ticks_and_teleports_are_not_blended():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 10.0))
    sim.hold("W", "SHIFT")
    sim.run(1.0)
    p = sim.player
    prev, cur = p.prev_pos.copy(), p.pos.copy()
    assert float(np.linalg.norm(cur - prev)) > 0.05, "running: the body moved this tick"
    assert np.allclose(p.render_pos(0.0), prev) and np.allclose(p.render_pos(1.0), cur)
    assert np.allclose(p.render_pos(0.5), (prev + cur) * 0.5, atol=1e-5)
    assert np.allclose(p.render_pos(7.0), cur), "never drawn ahead of the simulation"
    p.spawn_pos = np.array([40.0, 0.9, 40.0], dtype=np.float32)
    p.respawn()
    assert np.allclose(p.render_pos(0.3), p.pos), "a respawn is not blended across the map"
    sim.step()
    p.pos = p.pos + np.float32(5.0)                 # (a teleport inside a tick)
    assert np.allclose(p.render_pos(0.5), p.pos)


def _frame_speeds(frame_period, jitter, interpolate, seconds=3.0, seed=5):
    """The game loop's fixed-step accumulator (main.py) driven by frames `frame_period` apart with up
    to `jitter` of noise; returns the speed the camera is drawn moving at, frame to frame."""
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 60.0))
    sim.hold("W", "SHIFT")
    sim.run(config.RUN_BUILD_TIME + 0.5)                  # at full sprint
    rng = np.random.default_rng(seed)
    acc, shown, stamps, t = 0.0, [], [], 0.0
    while t < seconds:
        dt = frame_period + float(rng.uniform(-jitter, jitter))
        t += dt
        acc += dt
        while acc >= config.FIXED_DT:
            sim.step()
            acc -= config.FIXED_DT
        p = sim.player.render_pos(acc / config.FIXED_DT) if interpolate else sim.player.pos.copy()
        shown.append(float(p[2]))
        stamps.append(t)
    return np.abs(np.diff(shown)) / np.diff(stamps)


@pytest.mark.parametrize("period, jitter", [(1.0 / 60.0, 0.0015), (1.0 / 144.0, 0.0005)], ids=["60Hz", "144Hz"])
def test_a_sprint_is_drawn_at_an_even_speed_whatever_the_frame_timing(period, jitter):
    even = _frame_speeds(period, jitter, interpolate=True)
    raw = _frame_speeds(period, jitter, interpolate=False)
    assert abs(float(np.median(even)) - config.SPRINT_SPEED) < 0.2
    assert float(np.std(even)) < 0.15 * config.SPRINT_SPEED, f"interpolated: std {np.std(even):.2f} m/s"
    # (without it a frame shows two ticks of motion or none: the judder the interpolation removes)
    assert float(np.std(raw)) > 2.0 * float(np.std(even))
