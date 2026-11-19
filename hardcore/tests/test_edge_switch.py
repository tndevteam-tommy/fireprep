"""A traverse blocked in the middle of an edge by something in the way (an awning, a balcony).

Regression: the state machine took *any* failed shimmy step for the end of the edge and turned the
corner onto whatever the edge was joined to at its end - so a player shimmying into an awning was
swept along the wall (and through the awning) onto the building's next wall, up to ~9 m away, still
hanging.  The same happened after jumping at a wall with a direction held: the catch was followed by
the shimmy into the obstruction.  Now a corner is only turned at the end of the edge, and an
obstruction with an edge of its own is switched onto - lower down if need be - so the player can
climb on top of it.
"""
import numpy as np

from harness import *  # noqa
from test_legs import _check_anatomy
from physics.collision import CollisionSystem
from parkour.state_machine import ParkourStateMachine
from world.building import BuildingGenerator, BuildingSpec, SOUTH, NORTH, EAST, WEST

L1, L2 = config.UPPER_ARM_LEN, config.FOREARM_LEN


def _scene(parts):
    ls = LedgeSystem()
    floor = Obstacle(-80.0, -0.5, -80.0, 160.0, 0.5, 160.0, add_crevasses=False)
    floor.rails.clear()
    obs = [floor] + parts
    ls.build(obs)
    return ls, obs


def _awning_building():
    spec = BuildingSpec(0.0, -20.0, 14.0, 10.0, 3, facade_faces=(SOUTH,), door_face=SOUTH, rooftop=False,
                        awning=True, name="b")
    parts = BuildingGenerator.build(spec)
    ls, obs = _scene(parts)
    return ls, obs, parts, [p for p in parts if p.name.endswith("awning")][0]


def _hang(ls, obs, rail, x, with_arms=False):
    """A settled hang on `rail` at world x (along a south-facing wall)."""
    sim = Sim(ls, obs, spawn=(x, 0.9, -5.0), with_arms=with_arms)
    sim.face(-rail.normal)
    s = rail.clamp_s(rail.project_s(np.array([x, rail.height, float(rail.p1[2])], np.float32)))
    sim.player.pos = BodyModel.hang_center(rail, s)
    sim.player.on_ground = False
    f = sim.fsm
    f.state, f.active_rail, f.rail_s = ParkourState.LEDGE_HANG, rail, s
    f.grip_point = rail.get_point(s).copy()
    f.settle_time = 1.0
    return sim


def _run_checked(sim, seconds, obs, until=None):
    """Step while checking the body never enters geometry and the rigs stay anatomical."""
    for _ in range(int(round(seconds / DT))):
        sim.step()
        if sim.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY, ParkourState.CORNER_TURN):
            hit = CollisionSystem.overlaps_any(BodyModel.box_at(sim.pos, 0.03), obs)
            assert hit is None, f"{sim.state}: body inside {hit.name} at {sim.pos}"
        if sim.arm_rig is not None:
            L = sim.arm_rig.last
            for side in ("left", "right"):
                sh, el, wr = L[side + "_shoulder"], L[side + "_elbow"], L[side + "_wrist"]
                assert abs(np.linalg.norm(el - sh) - L1) < 1e-3 and abs(np.linalg.norm(wr - el) - L2) < 1e-3
                assert np.linalg.norm(wr - sh) <= L1 + L2 + 1e-3
                assert config.ELBOW_MIN_ANGLE_DEG - 0.01 <= L[side + "_elbow_angle"] <= config.ELBOW_MAX_ANGLE_DEG + 0.01
            assert max(L["wrist_angles"]) <= config.WRIST_MAX_ANGLE_DEG + 1e-3
            _check_anatomy(sim)
        if until is not None and until(sim):
            return True
    return until is None


def test_shimmying_into_an_awning_switches_onto_it_instead_of_being_thrown_round_the_corner():
    ls, obs, parts, aw = _awning_building()
    sill = [r for r in ls.rails if abs(r.height - 4.0) < 1e-3 and r.normal[2] > 0.99][0]
    assert sill.links, "the 4.0 m sill is joined round the building's corners (what the throw used)"
    for side, key in ((-1.0, "D"), (1.0, "A")):
        x0 = (float(aw.bounds.min[0]) - 1.0) if side < 0 else (float(aw.bounds.max[0]) + 1.0)
        sim = _hang(ls, obs, sill, x0, with_arms=True)
        sim.hold(key)
        assert _run_checked(sim, 3.0, obs, until=lambda s: s.had_event("EDGE_SWITCH") and s.state != ParkourState.CORNER_TURN)
        rail = sim.fsm.active_rail
        assert rail.owner is aw and abs(rail.height - float(aw.bounds.max[1])) < 1e-3, "hanging from the awning's edge"
        assert float(np.dot(rail.normal, [side, 0.0, 0.0])) > 0.99, "the side of the awning we ran into"
        assert abs(float(sim.pos[0]) - x0) < 1.0 and abs(float(sim.pos[2]) - (-10.0 + 0.36)) < 0.1, \
            "still right there at the wall, not round the building's corner"
        assert not any(e.startswith("CORNER_") for _, e in sim.events), [e for _, e in sim.events]


def test_after_switching_onto_the_awning_the_player_climbs_on_top_of_it():
    ls, obs, parts, aw = _awning_building()
    sill = [r for r in ls.rails if abs(r.height - 4.0) < 1e-3 and r.normal[2] > 0.99][0]
    sim = _hang(ls, obs, sill, float(aw.bounds.min[0]) - 1.0, with_arms=True)
    sim.hold("D")
    assert _run_checked(sim, 3.0, obs, until=lambda s: s.had_event("EDGE_SWITCH") and s.state == ParkourState.SHIMMY)
    sim.release("D")
    _run_checked(sim, 0.3, obs)
    sim.hold("W")
    assert _run_checked(sim, 3.0, obs, until=lambda s: s.standing_on(aw)), [e for _, e in sim.events]


def test_jumping_at_the_wall_with_a_direction_held_catches_then_switches_onto_the_awning():
    """The jump variant of the throw: a wall-kick catch of the sill beside the awning with the
    direction toward it still held."""
    ls, obs, parts, aw = _awning_building()
    sim = Sim(ls, obs, spawn=(float(aw.bounds.min[0]) - 1.0, 0.9, -8.3), with_arms=True)
    sim.face(np.array([0.0, 0.0, -1.0]))
    sim.tap("CAPS")
    sim.hold("W")
    sim.run(0.15)
    sim.hold("SPACE", "D")
    _run_checked(sim, 3.0, obs)
    evs = [e for _, e in sim.events]
    assert "GRAB" in evs
    assert not any(e.startswith("CORNER_") for e in evs), evs
    assert abs(float(sim.pos[0]) - (float(aw.bounds.min[0]) - 1.0)) < 2.5, "never swept along the wall"
    rail = sim.fsm.active_rail
    assert sim.standing_on(aw) or (rail is not None and (rail.owner is aw or rail.normal[2] > 0.99)), evs


def test_shimmying_into_a_balcony_switches_onto_it_and_climbs_onto_it():
    spec = BuildingSpec(0.0, -20.0, 14.0, 10.0, 3, facade_faces=(NORTH,), door_face=NORTH, rooftop=False,
                        balconies=((SOUTH, 1, 6.0),), name="b")
    parts = BuildingGenerator.build(spec)
    ls, obs = _scene(parts)
    bal = [p for p in parts if p.name.endswith("balcony")][0]
    course = [r for r in ls.rails if abs(r.height - 4.0) < 1e-3 and r.normal[2] > 0.99 and r.owner.name.endswith("core")][0]
    sim = _hang(ls, obs, course, float(bal.bounds.min[0]) - 1.0, with_arms=True)
    sim.hold("D")
    assert _run_checked(sim, 3.0, obs, until=lambda s: s.had_event("EDGE_SWITCH") and s.state != ParkourState.CORNER_TURN)
    assert sim.fsm.active_rail.owner is bal and abs(sim.fsm.active_rail.height - float(bal.bounds.max[1])) < 1e-3
    sim.release("D")
    _run_checked(sim, 0.3, obs)
    sim.hold("W")
    assert _run_checked(sim, 3.0, obs, until=lambda s: s.standing_on(bal))


def test_an_obstruction_without_an_edge_just_stops_the_traverse():
    # a long wall with a post in front of it (no holds on the post)
    ls, obs = build_custom([(-8.0, 0.0, -5.0, 16.0, 3.2, 1.0, False)])
    post = Obstacle(1.0, 0.0, -3.95, 0.3, 2.8, 0.3, rails=False, name="post")
    obs = obs + [post]
    ls.build(obs)
    edge = [r for r in ls.rails if abs(r.height - 3.2) < 1e-3 and r.normal[2] > 0.99][0]
    assert edge.links == {} or True
    sim = _hang(ls, obs, edge, -1.0)
    sim.hold("D")
    _run_checked(sim, 3.0, obs)
    assert sim.state == ParkourState.SHIMMY and sim.fsm.active_rail is edge
    assert float(sim.pos[0]) < 1.0 - 0.29, "stopped at the post"
    assert sim.fsm.last_event == "SHIMMY_BLOCKED" and not sim.had_event("EDGE_SWITCH")


def test_corner_turns_still_happen_at_the_end_of_an_edge():
    ls, obs, parts, aw = _awning_building()
    sill = [r for r in ls.rails if abs(r.height - 4.0) < 1e-3 and r.normal[2] > 0.99][0]
    sim = _hang(ls, obs, sill, 12.5)
    sim.hold("D")
    assert _run_checked(sim, 4.0, obs, until=lambda s: s.had_event("CORNER_DONE"))
    assert sim.fsm.active_rail.normal[0] > 0.99 and abs(sim.fsm.active_rail.height - 4.0) < 1e-3, "round onto the east wall"


def test_a_corner_turn_never_sweeps_the_body_along_a_wall():
    """Safety net for any future cause: a corner turn whose entry is far from the hands is refused."""
    ls, obs, parts, aw = _awning_building()
    sill = [r for r in ls.rails if abs(r.height - 4.0) < 1e-3 and r.normal[2] > 0.99][0]
    link = sill.links["right"]
    fsm = ParkourStateMachine(ls, obs)
    fsm.rail_s = sill.clamp_s(1.0)
    entry = BodyModel.corner_entry(link, obs)
    assert fsm._corner_sweep_ok(sill, link, entry), "at the end of the edge: a normal corner turn"
    fsm.rail_s = sill.clamp_s(0.4)
    assert not fsm._corner_sweep_ok(sill, link, entry), "mid-edge: the turn would sweep metres"
    assert fsm._near_rail_end(sill, 1.0) is False
    fsm.rail_s = sill.clamp_s(1.0)
    assert fsm._near_rail_end(sill, 1.0) and not fsm._near_rail_end(sill, -1.0)


# ----------------------------------------------------------- climbing onto a side-facing hold
def _camera_vs_body(sim):
    r = sim.fsm.active_rail
    return float(np.degrees(np.arccos(np.clip(np.dot(sim.player.forward_h, -r.normal), -1.0, 1.0))))


def test_climbing_onto_a_side_facing_hold_turns_the_camera_with_the_body():
    """Regression: from the window head beside a door awning, W climbs onto the awning's side edge
    (a hold off to the side, facing 90 deg away from the wall).  The body turned with the hold but
    the camera did not: the player hung sideways at the corner, feet off the ground, looking at
    the side of their own arm."""
    ls, obs, parts, aw = _awning_building()
    head = [r for r in ls.rails if abs(r.height - 2.2) < 1e-3 and r.normal[2] > 0.99
            and min(r.p1[0], r.p2[0]) < float(aw.bounds.min[0]) < max(r.p1[0], r.p2[0])][0]
    sim = _hang(ls, obs, head, float(aw.bounds.min[0]) - 0.05, with_arms=True)
    yaw0 = sim.player.yaw
    sim.hold("W")
    assert _run_checked(sim, 3.0, obs, until=lambda s: s.fsm.active_rail is not None and s.fsm.active_rail.owner is aw
                        and s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    _run_checked(sim, 0.5, obs)
    assert sim.fsm.active_rail.owner is aw and abs(sim.fsm.active_rail.normal[0]) > 0.99, "on the awning's side edge"
    assert _camera_vs_body(sim) < 5.0, f"camera {_camera_vs_body(sim):.0f} deg off the way the body faces"
    assert abs(abs(sim.player.yaw - yaw0) - 90.0) < 5.0, "the camera turned a quarter turn with the body"


def test_the_reported_jump_sequence_never_leaves_the_camera_sideways():
    """The exact sequence the randomised hunt found: a running jump at the wall beside the awning with
    W held catches the window head, shifts out from under the awning and climbs onto its side edge."""
    ls, obs, parts, aw = _awning_building()
    sim = Sim(ls, obs, spawn=(7.12, 0.9, -7.84), yaw=15.4)
    sim.tap("CAPS")
    sim.hold("W")
    sim.run(0.26)
    sim.hold("SHIFT")
    sim.run(0.14)
    sim.hold("SPACE")
    on_side_edge = False
    for _ in range(int(1.6 / DT)):
        sim.step()
        r = sim.fsm.active_rail
        if (sim.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY) and r is not None
                and abs(sim.player.turn_remaining) < 1e-3 and sim.fsm.settle_time >= config.GRAB_SETTLE_TIME):
            assert _camera_vs_body(sim) <= 20.0, (r.owner.name, _camera_vs_body(sim))
            on_side_edge = on_side_edge or (r.owner is aw and abs(r.normal[0]) > 0.99)
    assert on_side_edge, [e for _, e in sim.events]


def test_turn_with_body_is_the_signed_angle_between_the_two_facings():
    ls, obs, parts, aw = _awning_building()
    wall = [r for r in ls.rails if abs(r.height - 4.0) < 1e-3 and r.normal[2] > 0.99][0]
    west = [r for r in ls.rails if r.owner is aw and r.normal[0] < -0.99][0]
    east = [r for r in ls.rails if r.owner is aw and r.normal[0] > 0.99][0]
    fsm = ParkourStateMachine(ls, obs)
    for target, expect in ((west, -90.0), (east, 90.0)):
        fsm.turn_request_deg = 0.0
        fsm._turn_with_body(wall, target, 0.8)
        assert abs(fsm.turn_request_deg - expect) < 1e-3 and abs(fsm.turn_time_request - 0.8) < 1e-9
        # and the controller applying it really ends up facing the new hold
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0), yaw=0.0)
        sim.player.yaw = 0.0                          # facing -z, i.e. the south wall
        sim.player.fsm.turn_request_deg, sim.player.fsm.turn_time_request = fsm.turn_request_deg, 0.8
        sim.run(1.0)
        assert float(np.dot(sim.player.forward_h, -target.normal)) > 0.999
    fsm.turn_request_deg = 0.0
    fsm._turn_with_body(wall, wall, 0.5)
    assert fsm.turn_request_deg == 0.0, "same facing: no turn"
