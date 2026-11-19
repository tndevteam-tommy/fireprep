"""CyberGuard state machine: transitions, timings, distraction, perception modes."""

from dataclasses import replace

import pytest

from cyberduke.ai.guard import (
    CyberGuard,
    Inspecting,
    Investigating,
    Leaning,
    Pacing,
    Suspicious,
    WalkingToPatrol,
    WalkingToWall,
)
from cyberduke.config import GUARD
from cyberduke.core.geometry import AABB
from cyberduke.entities.player import Player
from cyberduke.systems.detection import DetectionSystem

DT = 1 / 60
NAV = (-10.0, 10.0, -10.0, 10.0)


def far_player():
    p = Player()
    p.x, p.z = 50.0, 50.0
    return p


def make_guard(**kw):
    cfg = replace(GUARD, **kw)
    return CyberGuard(
        0.0, 0.0, -6.0, NAV, patrol_z_min=-8.5, patrol_z_max=-3.0, wall_pos=(-5.5, -6.0), config=cfg
    )


def run(g, seconds, player=None, obstacles=None, detection=None):
    player = player or far_player()
    for _ in range(int(seconds / DT)):
        g.update(player, DT, obstacles=obstacles or [], detection=detection)


def test_full_patrol_cycle_visits_every_state_in_order():
    g = make_guard(pacing_duration=1.0, lean_duration=1.0)
    seen = [g.state]
    for _ in range(int(20 / DT)):
        g.update(far_player(), DT, obstacles=[])
        if g.state != seen[-1]:
            seen.append(g.state)
        if seen.count("PACING") == 2:
            break
    assert seen == ["PACING", "WALKING_TO_WALL", "LEANING", "WALKING_TO_PATROL", "PACING"]


def test_pacing_reverses_at_patrol_bounds_and_stays_in_lane():
    g = make_guard(pacing_duration=1e9)
    zs = []
    for _ in range(int(12 / DT)):
        g.update(far_player(), DT, obstacles=[])
        zs.append(g.z)
        assert g.x == 0.0
    assert min(zs) >= -8.5 - 0.05 and max(zs) <= -3.0 + 0.05
    assert max(zs) - min(zs) > 4.0, "actually walked the lane"


def test_pacing_reverses_when_an_obstacle_blocks_the_lane():
    g = make_guard(pacing_duration=1e9)
    crate = AABB(-0.5, 0.5, -5.0, -4.5)
    run(g, 3.0, obstacles=[crate])
    assert g.z < -5.0 - 0.35, "never entered the crate"
    assert g.patrol_dir == -1.0


def test_lean_lasts_the_configured_duration_and_ends_holstered():
    g = make_guard(pacing_duration=0.1, lean_duration=2.0)
    run(g, 0.2)
    run(g, 5.0)  # walk to the wall (0.35 m away at 1.2 m/s) and lean
    assert g.state in {"LEANING", "WALKING_TO_PATROL", "PACING"}
    # Find the last LEANING frame and check the prop is holstered there
    g = make_guard(pacing_duration=0.1, lean_duration=2.0)
    last_lean_pose = None
    for _ in range(int(8 / DT)):
        g.update(far_player(), DT, obstacles=[])
        if g.state == "LEANING":
            last_lean_pose = g.pose
    assert last_lean_pose is not None
    assert last_lean_pose.prop_pos == pytest.approx([-0.28, 0.98, 0.0], abs=0.02)


def test_distract_walks_to_the_point_inspects_then_returns():
    g = make_guard(investigate_duration=1.0)
    g.distract(2.0, -6.0, source=(2.0, -7.2))
    assert g.state == "INVESTIGATING" and g.investigate_target == (2.0, -6.0)
    run(g, 4.0)
    assert g.state in {"WALKING_TO_PATROL", "PACING"}
    assert g.investigate_target is None and g.investigate_source is None


def test_inspecting_faces_the_source_and_sways_around_it():
    g = make_guard(investigate_duration=10.0)
    g.fsm.transition_to(Inspecting(source=(0.0, -10.0)))  # source directly in -Z -> yaw 180
    assert g.target_yaw == pytest.approx(180.0)
    run(g, 1.5)  # let the initial turn toward the source finish
    deviations = []
    for _ in range(int(5 / DT)):
        g.update(far_player(), DT)
        deviations.append(abs(abs(((g.yaw + 180.0) % 360.0) - 180.0) - 180.0))
    assert max(deviations) <= GUARD.scan_amplitude_deg + 1.0
    assert max(deviations) > 20.0, "visibly swaying"


def test_reset_returns_to_pacing_from_anywhere():
    g = make_guard()
    g.fsm.transition_to(Leaning())
    g.x, g.z = 3.0, 3.0
    g.reset()
    assert g.state == "PACING" and (g.x, g.z) == (0.0, -6.0)
    assert g.fsm.time_in_state == 0.0 and g.patrol_dir == 1.0


def test_vision_narrows_while_engrossed_in_the_weapon():
    g = make_guard()
    g.yaw = 0.0
    p = Player()
    p.x, p.z = 0.0, -6.0 + 4.0  # 4 m ahead, in the alert cone but beyond the engrossed range
    g.fsm.transition_to(Pacing())
    assert g.can_see_player(p)
    g.fsm.transition_to(Leaning())  # turns to face the wall (+X, wall_yaw=90)
    g.fsm.update(2.5)  # mid-inspection: head down
    assert g.pose.head_pitch > GUARD.engrossed_head_pitch
    p.x, p.z = 4.0, -6.0  # 4 m ahead along the new facing: outside the engrossed range
    assert not g.can_see_player(p)
    p.x = 1.0  # but a player 1 m away is still noticed
    assert g.can_see_player(p)


def test_update_reports_to_the_detection_system_only_when_seen():
    g = make_guard()
    d = DetectionSystem()
    p = Player()
    p.x, p.z = 0.0, -3.5
    d.begin_frame()
    assert g.update(p, DT, obstacles=[], detection=d) is True
    d.end_frame(DT)
    assert d.level > 0
    d.begin_frame()
    assert g.update(far_player(), DT, obstacles=[], detection=d) is False
    d.end_frame(DT)
    assert d.level < 45 * DT


def test_state_classes_are_independently_instantiable():
    for cls in (Pacing, WalkingToWall, Leaning, WalkingToPatrol):
        assert cls().name
    assert Investigating((1.0, 2.0), None).name == "INVESTIGATING"


# =============================================================================== vision range
def test_standing_player_is_seen_at_realistic_distances():
    g = make_guard()
    g.yaw = 0.0  # facing +Z
    p = Player()
    for dist in (4.0, 8.0, 11.0):
        p.x, p.z = 0.0, -6.0 + dist
        assert g.can_see_player(p), f"not seen at {dist} m"
    p.z = -6.0 + 13.0
    assert not g.can_see_player(p), "beyond the alert range"


def test_detection_rate_falls_off_with_distance_and_for_crouching():
    g = make_guard()
    p = Player()
    rates = []
    for dist in (1.0, 3.0, 6.0, 9.0, GUARD.alert_range):
        p.x, p.z = 0.0, -6.0 + dist
        rates.append(g.sighting_rate(p))
    assert rates[0] == rates[1] == 45.0, "full rate up close"
    assert rates == sorted(rates, reverse=True), "monotonically weaker with distance"
    assert rates[-1] == pytest.approx(45.0 * GUARD.far_rate_fraction)
    p.crouching = True
    assert g.sighting_rate(p) == pytest.approx(12.0 * GUARD.far_rate_fraction)


# =============================================================================== suspicion
def player_in_front(dist=6.0):
    p = Player()
    p.x, p.z = 0.0, -6.0 + dist
    return p


def test_sighting_interrupts_the_routine_stops_the_guard_and_faces_the_player():
    g = make_guard(pacing_duration=1e9)
    g.yaw = 0.0
    p = player_in_front(8.0)
    assert g.update(p, DT, obstacles=[]) is True
    assert g.state == "ALERTED"
    x, z = g.x, g.z
    run(g, 1.0, player=p)
    assert (g.x, g.z) == (x, z), "stands still while alerted"
    assert g.target_yaw == pytest.approx(0.0) and g.last_seen == (p.x, p.z)
    assert g.pose.head_pitch <= GUARD.engrossed_head_pitch, "attentive, not engrossed"


def test_alert_escalates_to_suspicious_past_the_threshold_and_the_guard_creeps_closer():
    g = make_guard()
    g.yaw = 0.0
    p = player_in_front(8.0)
    d = DetectionSystem()
    states = []
    start_z = g.z
    for _ in range(int(6.0 / DT)):
        d.begin_frame()
        g.update(p, DT, obstacles=[], detection=d)
        d.end_frame(DT)
        states.append(g.state)
        if g.state == "SUSPICIOUS" and g.z - start_z > 0.5:
            break
    assert "ALERTED" in states and "SUSPICIOUS" in states
    assert states.index("SUSPICIOUS") > states.index("ALERTED")
    assert d.level >= GUARD.suspicion_threshold
    assert g.z > start_z, "moved toward the player"
    assert g.speed_factor == pytest.approx(GUARD.suspicious_speed_factor)


def test_suspicious_guard_moves_at_the_reduced_speed():
    g = make_guard()
    g.yaw = 0.0
    g.last_seen = (0.0, 4.0)
    g.time_since_seen = 0.0
    g.fsm.transition_to(Suspicious(WalkingToPatrol))
    z0 = g.z
    for _ in range(60):  # 1 s, player kept "visible" by pinning time_since_seen
        g.fsm.update(DT)
        g.time_since_seen = 0.0
    assert g.z - z0 == pytest.approx(g.walk_speed * GUARD.suspicious_speed_factor, rel=0.05)


def test_losing_sight_while_alerted_resumes_the_routine():
    g = make_guard(pacing_duration=1e9)
    g.yaw = 0.0
    p = player_in_front(8.0)
    g.update(p, DT, obstacles=[])
    assert g.state == "ALERTED"
    run(g, GUARD.alert_grace + 0.2)  # far_player(): out of sight
    assert g.state in {"WALKING_TO_PATROL", "PACING"}


def test_player_hiding_while_suspicious_makes_the_guard_give_up_and_resume():
    g = make_guard()
    g.yaw = 0.0
    g.last_seen = (0.0, 2.0)
    g.time_since_seen = 0.0
    g.fsm.transition_to(Suspicious(WalkingToPatrol))
    run(g, GUARD.suspicion_grace + 0.3)  # nobody in sight the whole time
    assert g.state in {"WALKING_TO_PATROL", "PACING"}
    assert g.speed_factor == 1.0


def test_arriving_at_the_last_seen_spot_without_a_sighting_looks_around_then_resumes():
    g = make_guard()
    g.yaw = 0.0
    g.last_seen = (0.0, -5.0)  # 1 m ahead
    g.time_since_seen = 0.0
    g.fsm.transition_to(Suspicious(WalkingToPatrol))
    seen = []
    for _ in range(int(6.0 / DT)):
        g.update(far_player(), DT, obstacles=[])
        g.time_since_seen = 0.0  # keep the suspicion alive until arrival
        seen.append(g.state)
        if g.state == "LOOKING_AROUND":
            break
    assert "LOOKING_AROUND" in seen
    run(g, GUARD.look_around_duration + 0.2)
    assert g.state in {"WALKING_TO_PATROL", "PACING"}


def test_resumes_the_wall_trip_if_that_was_interrupted():
    g = make_guard(pacing_duration=0.1)
    run(g, 0.2)
    assert g.state == "WALKING_TO_WALL"
    g.yaw = 0.0
    g.update(player_in_front(5.0), DT, obstacles=[])
    assert g.state == "ALERTED"
    run(g, GUARD.alert_grace + 0.2)
    assert g.state in {"WALKING_TO_WALL", "LEANING"}


def test_reset_clears_sighting_memory():
    g = make_guard()
    g.yaw = 0.0
    g.update(player_in_front(5.0), DT, obstacles=[])
    g.reset()
    assert g.last_seen is None and g.state == "PACING" and not g.is_reacting


# =============================================================================== end to end
def test_scenario_guard_investigates_then_gives_up_when_the_player_hides_behind_a_crate():
    from cyberduke.simulation import InputState, Simulation

    sim = Simulation()
    g = sim.guards[0]
    g.x, g.z, g.yaw = 0.0, -6.0, 0.0
    g.patrol_dir = 1.0
    p = sim.player
    p.x, p.z = 0.0, 2.0  # standing in the open, 8 m in front of the guard
    seen = []
    for _ in range(int(8.0 / (1 / 60))):
        sim.update(1 / 60, InputState())
        seen.append(g.state)
        if g.state == "SUSPICIOUS" and g.z > -5.7:
            break
    assert "ALERTED" in seen and g.state == "SUSPICIOUS"
    assert 0 < sim.detection.level < 100
    # Duck behind the left crate stack (1.6 m tall, at x -5.2..-3.8, z 2.3..3.7)
    p.x, p.z, p.crouching = -4.5, 4.6, True
    for _ in range(int(5.0 / (1 / 60))):
        sim.update(1 / 60, InputState())
        if not g.is_reacting:
            break
    assert not g.is_reacting, f"guard still {g.state}"
    assert g.state in {"WALKING_TO_PATROL", "PACING", "LOOKING_AROUND"} or not g.is_reacting
    assert sim.state.value == "PLAYING"


# =============================================================================== catwalk cover
def _on_catwalk(sim, x, z, crouching):
    from cyberduke.world.surfaces import ElevationState

    p = sim.player
    p.x, p.z, p.crouching = x, z, crouching
    p.elevation = ElevationState(on_catwalk=True)
    p.floor_y = sim.room.catwalk.height
    return p


def test_regression_crouching_behind_the_catwalk_rail_hides_the_player_from_below():
    """Guard 2 could see a crouched player on the catwalk: the rail panel was flagged
    see-through and the deck had no sight representation at all."""
    from cyberduke.simulation import Simulation

    sim = Simulation()
    g = sim.guards[1]
    g.x, g.z, g.yaw = 2.1, 30.0, 90.0  # on its lane, looking straight at the right catwalk
    g.frame_sight_blockers = sim.sight_blockers
    p = _on_catwalk(sim, 7.75, 30.0, crouching=True)
    assert not g.can_see_player(p)


def test_standing_on_the_catwalk_is_visible_over_the_rail_from_across_the_room():
    from cyberduke.simulation import Simulation

    sim = Simulation()
    g = sim.guards[1]
    g.x, g.z, g.yaw = -3.0, 30.0, 90.0  # far side of the vault
    g.frame_sight_blockers = sim.sight_blockers
    p = _on_catwalk(sim, 7.75, 30.0, crouching=False)
    assert g.can_see_player(p), "upper body shows above the 3.7 m rail top from a distance"
    p.crouching = True
    assert not g.can_see_player(p)


def test_the_deck_itself_hides_a_player_directly_above_the_guard():
    from cyberduke.simulation import Simulation

    sim = Simulation()
    g = sim.guards[1]
    g.x, g.z, g.yaw = 7.0, 27.0, 0.0  # under the right catwalk, facing +Z
    g.frame_sight_blockers = sim.sight_blockers
    p = _on_catwalk(sim, 7.75, 31.0, crouching=False)
    assert not g.can_see_player(p)


def test_simulation_passes_sight_blockers_to_guards_and_cameras():
    from cyberduke.simulation import InputState, Simulation

    sim = Simulation()
    g = sim.guards[1]
    g.x, g.z, g.yaw = 2.1, 30.0, 90.0
    g.fsm.transition_to(Inspecting(source=(7.75, 30.0)))
    _on_catwalk(sim, 7.75, 30.0, crouching=True)
    sim.update(1 / 60, InputState())
    assert sim.detection.level == 0.0
    assert g.frame_sight_blockers is sim.sight_blockers


# =============================================================================== quickhack states
def test_blind_stops_the_guard_seeing_for_the_duration_then_it_resumes_its_routine():
    g = make_guard(pacing_duration=1e9)
    g.yaw = 0.0
    p = player_in_front(5.0)
    assert g.can_see_player(p)
    assert g.blind(2.0)
    assert g.state == "BLINDED" and g.is_blinded and g.is_incapacitated
    assert not g.can_see_player(p), "optics jammed"
    assert g.blind_time_left == pytest.approx(2.0)
    x, z = g.x, g.z
    run(g, 1.0, player=p)
    assert (g.x, g.z) == (x, z), "stands still"
    assert g.blind_time_left == pytest.approx(1.0, abs=0.05)
    assert g.pose.arm_l[0] < -90 and g.pose.arm_r[0] < -90, "hands up at the visor"
    run(g, 1.2, player=far_player())
    assert not g.is_blinded
    assert g.state in {"WALKING_TO_PATROL", "PACING"}
    assert g.can_see_player(p)


def test_re_blinding_restarts_the_timer_and_keeps_the_original_resume_point():
    g = make_guard(pacing_duration=0.1)
    run(g, 0.2)
    assert g.state == "WALKING_TO_WALL"
    g.blind(1.0)
    run(g, 0.8)
    g.blind(1.0)
    assert g.blind_time_left == pytest.approx(1.0)
    run(g, 1.1)
    assert g.state in {"WALKING_TO_WALL", "LEANING"}, "back to the interrupted wall trip"


def test_knock_out_is_terminal_until_reset():
    g = make_guard()
    assert g.knock_out()
    assert g.state == "KNOCKED_OUT" and g.is_down and g.is_incapacitated
    assert not g.knock_out(), "already down"
    assert not g.blind(), "nothing to blind"
    assert g.shock_timer > 0.0
    run(g, 5.0, player=player_in_front(2.0))
    assert g.state == "KNOCKED_OUT" and g.fall_angle >= 75.0, "lying on the floor"
    assert g.shock_timer == 0.0 and not g.can_see_player(player_in_front(2.0))
    assert g.hack_position[1] < 0.75, "the marker comes down with the body"
    g.reset()
    assert g.state == "PACING" and not g.is_down and g.fall_angle == 0.0
    assert g.body is None, "back on its feet"


@pytest.mark.parametrize("direction", ["back", "front", "left", "right"])
def test_knocked_out_guard_convulses_on_its_feet_then_goes_down_that_way(direction):
    from cyberduke.config import HACKING

    g = make_guard()
    g.knock_out(direction)
    angles, poses, hips = [], [], []
    for _ in range(int(4.5 / DT)):
        g.update(far_player(), DT)
        angles.append(g.fall_angle)
        poses.append(g.pose.head_yaw)
        hips.append(g.body.hip[1] if g.body else 0.9)
    shock_frames = int(HACKING.knockout_shock_duration / DT)
    assert all(a == 0.0 for a in angles[:shock_frames]), "upright while the current runs"
    assert max(poses[:shock_frames]) - min(poses[:shock_frames]) > 15.0, "twitching"
    falling = angles[shock_frames:]
    assert falling == sorted(falling), "then goes over, never back up"
    assert 0.0 < falling[10] < 70.0 and falling[-1] >= 75.0
    assert hips[-1] < 0.45, "down on the floor"


def test_knocked_out_guard_never_stays_propped_against_a_wall():
    """A wall 0.45 m behind a guard falling backward: it hits the wall, slides down it and
    topples off to the side - it ends lying on the floor, not leaning on the wall."""
    wall = AABB(-5.0, 5.0, -6.75, -6.45, 0.0, 3.0)
    g = make_guard()
    g.yaw = 0.0
    g.knock_out("back")
    tilts = []
    for _ in range(int(6.0 / DT)):
        g.update(far_player(), DT, obstacles=[wall])
        if g.body is not None:
            tilts.append(g.fall_angle)
    assert any(10.0 < a < 60.0 for a in tilts), "it met the wall on the way down"
    assert tilts[-1] >= 75.0, "and ended on the floor"
    head = g.body.point(0.0, 2.0, 0.0)
    assert head[2] > -6.45, "nothing of it inside the wall"


def test_incapacitated_guards_ignore_noises():
    for method in ("blind", "knock_out"):
        g = make_guard()
        getattr(g, method)()
        g.distract(2.0, -6.0)
        assert g.state != "INVESTIGATING", method


def test_sighting_never_interrupts_blindness():
    g = make_guard()
    g.yaw = 0.0
    g.blind(3.0)
    assert g.update(player_in_front(3.0), DT, obstacles=[]) is False
    assert g.state == "BLINDED"


def test_essential_guard_gets_the_elite_look_and_label():
    from cyberduke.ai.guard import guard_spec

    plain = CyberGuard(0.0, 0.0, 0.0, NAV, guard_id="door_guard")
    elite = CyberGuard(0.0, 0.0, 0.0, NAV, guard_id="boss", essential=True)
    assert plain.spec == guard_spec("door_guard") and plain.display_name == "DOOR GUARD"
    assert elite.spec == guard_spec("boss", elite=True) and elite.display_name == "BOSS (ELITE)"
    assert elite.spec.visor_color == (1.0, 0.75, 0.1), "the elite's visor is gold"
    assert plain.spec.visor_color[0] == 1.0 and plain.spec.visor_color[1] < 0.5, "red"
    assert elite.target_kind == "guard" and elite.hack_id == "boss" and not elite.secured


# =============================================================================== distractions
def test_noise_decides_how_long_it_is_inspected_and_the_default_is_the_config():
    from cyberduke.core.events import EventBus, NoiseEmitted

    bus = EventBus()
    g = make_guard(investigate_duration=2.0)
    g.bind(bus)
    bus.publish(NoiseEmitted(0.0, -5.0, 10.0, 0.0, -5.0, inspect_duration=6.0))
    run(g, 3.0)  # arrive (1 m) and inspect
    assert g.state == "INSPECTING" and g.fsm.state.duration == 6.0
    run(g, 4.0)
    assert g.state in {"WALKING_TO_PATROL", "PACING"}
    g.distract(0.0, -5.0)
    run(g, 1.5)
    assert g.state == "INSPECTING" and g.fsm.state.duration == 2.0, "config default"


def test_bang_gesture_hits_the_machine_first_then_just_stares_at_it():
    g = make_guard(bang_duration=2.0)
    g.fsm.transition_to(Inspecting(source=(0.0, -8.0), duration=5.0, gesture="bang"))
    poses = []
    for _ in range(int(1.9 / DT)):
        g.update(far_player(), DT)
        poses.append(g.pose.arm_r[0])
        assert g.is_banging
    assert max(poses) - min(poses) > 40.0, "the right arm swings up and down"
    assert min(poses) < -100.0, "raised high"
    run(g, 0.3)
    assert not g.is_banging and g.state == "INSPECTING", "banging stops, watching goes on"
    run(g, 3.0)
    assert g.state != "INSPECTING"


def test_look_gesture_never_bangs():
    g = make_guard()
    g.fsm.transition_to(Inspecting(source=(0.0, -8.0), duration=5.0))
    run(g, 1.0)
    assert not g.is_banging and g.pose.arm_r[0] > -30.0


# =============================================================================== routes
def route_guard(**kw):
    cfg = replace(GUARD, **kw)
    return CyberGuard(
        0.0, 0.0, 0.0, NAV, wall_pos=None, config=cfg,
        patrol_route=((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)),
    )  # fmt: skip


def test_route_guard_walks_its_waypoints_in_order_and_loops():
    g = route_guard(route_pause=0.2)
    assert g.state == "PATROL_ROUTE" and g.wall_pos is None
    visited = []
    for _ in range(int(25 / DT)):
        g.update(far_player(), DT, obstacles=[])
        if g.route_index != (visited[-1] if visited else -1):
            visited.append(g.route_index)
        if len(visited) >= 6:
            break
    assert visited[:6] == [0, 1, 2, 3, 0, 1]
    assert g.state == "PATROL_ROUTE", "never goes to lean on a wall it does not have"


def test_route_guard_lingers_and_looks_around_at_each_waypoint():
    g = route_guard(route_pause=1.0)
    for _ in range(int(6 / DT)):
        g.update(far_player(), DT, obstacles=[])
        if g.fsm.state.linger > 0.0:
            break
    assert g.fsm.state.linger > 0.0
    x, z = g.x, g.z
    run(g, 0.5)
    assert (g.x, g.z) == (x, z) and g.pose.head_yaw != 0.0


def test_route_guard_resumes_the_route_at_the_nearest_waypoint_after_a_reaction():
    g = route_guard(pacing_duration=1e9)
    g.x, g.z, g.yaw = 3.8, 0.4, 0.0
    p = Player()
    p.x, p.z = 3.8, 4.0  # 3.6 m ahead
    g.update(p, DT, obstacles=[])
    assert g.state == "ALERTED"
    run(g, GUARD.alert_grace + 0.2)
    assert g.state == "PATROL_ROUTE" and g.route_index == 1, "picked up at (4, 0)"


def test_route_guard_investigates_noises_and_comes_back_to_the_loop():
    g = route_guard(investigate_duration=0.5)
    g.distract(2.0, 2.0)
    run(g, 4.0)
    assert g.state in {"PATROL_ROUTE", "WALKING_TO_PATROL"}


def test_reset_restarts_the_route_from_the_first_waypoint():
    g = route_guard()
    g.route_index = 2
    g.reset()
    assert g.route_index == 0 and g.state == "PATROL_ROUTE"


def test_lane_guard_without_a_wall_spot_paces_forever():
    g = CyberGuard(0.0, 0.0, -6.0, NAV, patrol_z_min=-8.5, patrol_z_max=-3.0, wall_pos=None,
                   config=replace(GUARD, pacing_duration=0.5))  # fmt: skip
    run(g, 5.0)
    assert g.state == "PACING"


def test_guards_wear_one_uniform_but_are_each_their_own():
    """Every guard: the armour, the visor, the sidearm - and their own hair, rank
    stripes and visor red, so that no two are the same."""
    from cyberduke.ai.guard import guard_spec

    specs = [guard_spec(f"guard_{k}") for k in range(8)]
    for spec in specs:
        assert spec.has_visor and spec.has_holster and spec.prop == "laser_blaster"
        assert spec.look.top == "armor" and spec.look.eyes == "visor"
    looks = {(s.look, s.hair_color, s.visor_color, s.skin_color) for s in specs}
    assert len(looks) == len(specs), "no two guards alike"
    assert len({s.look.hair for s in specs}) >= 3, "their own haircuts"


def test_a_guards_face_shows_what_it_is_doing():
    """Calm on the beat; a frown when something is up; teeth bared closing in."""
    g = make_guard()
    run(g, 0.5)
    assert g.pose.expression is None, "the resting face on patrol"
    g.yaw = 0.0
    g.update(player_in_front(3.0), DT, obstacles=[])
    g.update(player_in_front(3.0), DT, obstacles=[])
    assert g.state == "ALERTED" and g.pose.expression == "frown"
    g.blind(3.0)
    g.update(player_in_front(3.0), DT, obstacles=[])
    assert g.pose.expression == "shout"
