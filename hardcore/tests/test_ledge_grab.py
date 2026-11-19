"""Grabbing, hanging, shimmying and dropping in the live simulation."""
import numpy as np
from harness import *  # noqa
from parkour.ledge_rail import RailType, LedgeRail


def _wall(h, depth=4.0, width=4.0):
    """Block whose south face (normal +Z) is at z = -1; the player approaches from +Z."""
    ls, obs = build_custom([(-width / 2, 0.0, -1.0 - depth, width, h, depth)])
    return ls, obs, obs[1]


def test_walk_up_and_grab_2m2_wall_then_hang_properly():
    ls, obs, wall = _wall(2.2)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.3)
    grip = sim.fsm.grip_point
    assert abs(grip[1] - 2.2) < 1e-3
    eye = sim.player.eye_pos
    assert grip[1] - eye[1] > 0.2, "eyes must be below the hands in a hang"
    assert sim.feet_y >= -1e-3, "feet below the floor while hanging"
    assert BodyModel.pose_is_free(sim.pos, obs)
    assert sim.max_step < 0.2, "grab must blend into the hang, not teleport"


def test_low_walls_vault_instead_of_hanging():
    for h in (1.2, 1.7):
        ls, obs, wall = _wall(h, depth=0.6)
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
        sim.hold("W", "E")
        assert sim.run(4.0, until=lambda s: s.standing_on(wall)), f"{h} m wall not vaulted"
        assert ParkourState.LEDGE_HANG not in sim.states
        assert min(tr["pos"][1] for tr in sim.trace) >= 0.9 - 1e-3, "body sank into the floor"


def test_auto_step_heights_walking_vs_running():
    ls, obs, low = _wall(0.5, depth=0.6)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.standing_on(low)), "knee-high block is stepped onto while walking"

    ls, obs, hip = _wall(0.8, depth=0.6)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W")
    sim.run(2.0)
    assert not sim.standing_on(hip), "walking into a hip-high block stops you"
    sim.hold("SHIFT")
    assert sim.run(3.0, until=lambda s: s.standing_on(hip)), "running flows over a hip-high block"

    ls, obs, mid = _wall(1.2, depth=0.6)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "SHIFT")
    sim.run(2.0)
    assert not sim.standing_on(mid), "1.2 m needs a deliberate vault (Space / E)"
    sim.tap("SPACE")
    assert sim.run(3.0, until=lambda s: s.standing_on(mid))


def test_cannot_grab_ledge_above_standing_reach():
    """Regression: a 3.4 m edge used to be grabbable flat-footed (reach was measured 1.8 m from the eyes)."""
    ls, obs, wall = _wall(3.4, depth=4.0)
    wall.rails = [r for r in wall.rails if r.rail_type == RailType.MANTLE_TOP]
    ls.rails[:] = [r for r in ls.rails if r.rail_type == RailType.MANTLE_TOP]
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    sim.run(2.5)
    assert sim.state in (ParkourState.GROUND, ParkourState.SPRINT)
    assert sim.fsm.active_rail is None


def test_grab_head_crevasse_from_ground_and_shimmy_within_bounds():
    ls, obs, wall = _wall(5.4, width=6.0)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    rail = sim.fsm.active_rail
    assert rail.rail_type == RailType.HEAD_CREVASSE
    # Shimmy right for 1 s: speed about SHIMMY_SPEED, hands stay on the rail
    x0 = float(sim.pos[0])
    sim.hold("D")
    sim.run(1.0)
    moved = float(sim.pos[0]) - x0
    assert sim.state == ParkourState.SHIMMY
    assert 0.8 * config.SHIMMY_SPEED <= moved <= 1.05 * config.SHIMMY_SPEED, f"shimmy moved {moved:.2f} m/s"
    assert config.SHIMMY_SPEED <= 1.5, "hand-over-hand traverse faster than a human"
    sim.run(6.0)
    assert sim.fsm.rail_s <= 1.0 - rail.s_margin() + 1e-6
    grip = sim.fsm.grip_point
    assert grip[0] <= rail.p2[0] - config.RAIL_END_MARGIN + 1e-3, "outer hand hangs past the corner"
    for tr in sim.trace:
        if tr["state"] in (ParkourState.SHIMMY, ParkourState.LEDGE_HANG):
            assert BodyModel.pose_is_free(tr["pos"], obs)


def test_drop_releases_and_lockout_prevents_instant_regrab():
    ls, obs, wall = _wall(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.3)
    sim.hold("S")
    sim.step()
    assert sim.had_event("DROP")
    assert sim.state in (ParkourState.AIRBORNE, ParkourState.GROUND)
    # E is still held; must not re-grab during the lockout
    sim.run(0.2)
    assert sim.state != ParkourState.LEDGE_HANG
    sim.run(1.0)
    assert sim.player.on_ground


def test_held_s_and_e_chord_does_not_auto_drop_on_grab():
    """Trigger semantics: the drop needs a fresh S press, otherwise a held S+E would drop instantly."""
    ls, obs, wall = _wall(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 4.0))
    sim.hold("S", "E")
    sim.run(0.3)                       # the S press is now older than the input buffer window
    sim.teleport((0.0, 0.9, -0.6))     # arrive under the 2.0 m crevasse with the chord still held
    sim.run(0.5)
    assert sim.state == ParkourState.LEDGE_HANG, "should grab and stay hanging"
    assert not sim.had_event("DROP")
    # A fresh S press while hanging does drop
    sim.release("S")
    sim.step()
    sim.hold("S")
    sim.step()
    assert sim.had_event("DROP")


def test_airborne_catch_of_ledge_while_falling_past_it():
    ls, obs, wall = _wall(4.0)
    # Drop from above next to the wall face, facing it, with E held: catch the 3.8 crevasse on the way down
    sim = Sim(ls, obs, spawn=(0.0, 6.0, -0.6))
    sim.player.on_ground = False
    sim.hold("E")
    sim.run(2.0, until=lambda s: s.state == ParkourState.LEDGE_HANG or s.player.on_ground)
    assert sim.state == ParkourState.LEDGE_HANG
    assert sim.fsm.active_rail.height <= 4.0 + 1e-3


def test_grab_requires_facing_the_wall():
    ls, obs, wall = _wall(2.2)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -0.65), yaw=180.0)   # back to the wall
    sim.hold("E")
    sim.run(0.5)
    assert sim.state != ParkourState.LEDGE_HANG


def test_hanging_zeroes_velocity_and_ignores_gravity():
    ls, obs, wall = _wall(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.3)          # settle blend into the hang pose
    y = float(sim.pos[1])
    sim.run(1.0)
    assert abs(float(sim.pos[1]) - y) < 1e-3
    assert np.allclose(sim.vel, 0.0)


def test_buffered_grab_press_survives_until_ledge_is_in_reach():
    """A tap of E up to 150 ms before the hands reach the ledge must still grab (input buffer)."""
    ls, obs, wall = _wall(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 2.0))
    sim.hold("W")
    sim.run(2.0, until=lambda s: s.pos[2] < 0.15)      # walking at 5 m/s, ~0.5 m (0.1 s) short of reach
    assert sim.player.grab_candidate() is None
    sim.tap("E")
    assert not sim.input.is_down(Action.GRAB)
    assert sim.run(0.5, until=lambda s: s.state == ParkourState.LEDGE_HANG)


def test_hang_arms_follow_wall_not_head_yaw():
    ls, obs, wall = _wall(5.4)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0), with_arms=True)
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.3)
    wrists_before = (sim.arm_rig.last["left_wrist"].copy(), sim.arm_rig.last["right_wrist"].copy())
    sim.player.yaw += 80.0          # look far to the side while hanging
    sim.run(0.1)
    L = sim.arm_rig.last
    assert np.allclose(L["left_wrist"], wrists_before[0], atol=1e-4)
    assert np.allclose(L["right_wrist"], wrists_before[1], atol=1e-4)
    for side in ("left", "right"):
        assert np.linalg.norm(L[side + "_wrist"] - L[side + "_target"]) < 0.01


def test_low_crevasse_is_held_with_feet_on_the_ground():
    """A hold that is too low for a free hang is gripped standing (feet never sink into the floor)."""
    ls, obs = build_custom([(-2.0, 0.0, -5.0, 4.0, 3.0, 4.0, False)])
    wall = obs[1]
    crev = LedgeRail([-2.0, 1.85, -1.0], [2.0, 1.85, -1.0], [0, 0, 1], RailType.HEAD_CREVASSE, owner=wall)
    wall.rails.append(crev)
    ls.add_rail(crev)
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.0))
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    sim.run(0.3)
    assert sim.fsm.supported_hang
    assert abs(sim.feet_y) < 0.01
    sim.hold("D")
    sim.run(1.0)
    assert abs(sim.feet_y) < 0.01, "feet must stay on the ground while traversing a low hold"
    sim.release("D")
    sim.hold("W")
    assert sim.run(4.0, until=lambda s: s.standing_on(wall), top_out=True)
