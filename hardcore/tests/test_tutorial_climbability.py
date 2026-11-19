"""Every object in the tutorial playground can be climbed - slowly and at speed."""
import numpy as np
from harness import *  # noqa
from physics.collision import CollisionSystem

FACES = [np.array(n, dtype=np.float32) for n in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0))]


def _face_center(o, n):
    b = o.bounds
    c = b.center.copy()
    c[1] = float(b.min[1])
    # point on the face
    for i in (0, 2):
        if n[i] > 0.5:
            c[i] = float(b.max[i])
        elif n[i] < -0.5:
            c[i] = float(b.min[i])
    return c


def _spawn_points(o, n, obstacles, dist=1.6):
    """Ground spawn points in front of face `n` of `o`: centred, then shifted sideways
    (an attached ledge such as a balcony is reached from the wall beside it)."""
    fc = _face_center(o, n)
    half = 0.5 * float(np.dot(np.abs(np.cross(n, [0.0, 1.0, 0.0])), o.bounds.size))
    out = []
    for lateral in (0.0, 1.0, -1.0, 2.0, -2.0, half + 0.8, -(half + 0.8)):
        t = np.cross(n, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        p = fc + n * dist + t * lateral
        center = np.array([p[0], 0.9, p[2]], dtype=np.float32)
        if not BodyModel.pose_is_free(center, obstacles):
            continue
        ground = CollisionSystem.ground_height_below(center, BodyModel.SIZE, obstacles, max_drop=0.2)
        if ground is None:
            continue
        if not any(np.linalg.norm(center - c) < 0.5 for c in out):
            out.append(center)
    return out


def _spawn_in_front(o, n, obstacles, dist=1.6):
    pts = _spawn_points(o, n, obstacles, dist)
    return pts[0] if pts else None


def _climbable_obstacles(obstacles):
    out = []
    for o in obstacles:
        if not o.rails or not o.climb_target:
            continue          # the floor / structural sub-parts of a building
        out.append(o)
    return out


def _slow_climb(ls, obs, o, n, timeout=14.0, keys=("W", "E")):
    """Walk into face `n` of `o` holding `keys`, looking at the top while on a rail.
    Returns the Sim on success, False if every spawn in front of that face failed,
    None if there is nowhere to stand in front of that face."""
    spawns = _spawn_points(o, n, obs)
    if not spawns:
        return None
    b = o.bounds
    for spawn in spawns:
        sim = Sim(ls, obs, spawn=tuple(spawn))
        sim.face(-n)
        sim.hold(*keys)

        def tick(s):
            # a climber looks at where they want to go (the nearest bit of the top surface);
            # holds in the look direction are preferred
            if s.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY):
                aim = np.array([np.clip(s.pos[0], b.min[0] + 0.3, b.max[0] - 0.3), b.max[1],
                                np.clip(s.pos[2], b.min[2] + 0.3, b.max[2] - 0.3)], dtype=np.float32)
                s.look_at(aim)
            return s.standing_on(o)
        if sim.run(timeout, until=tick, top_out=True):
            return sim
    return False


def test_every_tutorial_object_is_climbable_slowly():
    ls, obs = build_playground()
    failures = []
    for o in _climbable_obstacles(obs):
        ok = False
        tried = 0
        for n in FACES:
            res = _slow_climb(ls, obs, o, n)
            if res is None:
                continue
            tried += 1
            if res:
                ok = True
                assert res.max_root_step < 0.2, "teleport during climb"
                break
        if not ok:
            failures.append((tuple(o.bounds.min.round(1)), tuple(o.bounds.size.round(1)), tried))
    assert not failures, f"unclimbable obstacles: {failures}"


def test_every_tall_object_climb_uses_crevasses_and_legs():
    """Anything taller than standing reach must be climbed hold by hold (no lucky jumps)."""
    ls, obs = build_playground()
    for o in _climbable_obstacles(obs):
        if float(o.bounds.size[1]) <= config.STAND_REACH:
            continue
        for n in FACES:
            res = _slow_climb(ls, obs, o, n)
            if res:
                evs = [e for _, e in res.events]
                assert "CLIMB_UP" in evs or "CLIMB_UP_FAST" in evs, f"{o.bounds.size} climbed without a lip stand-up"
                assert evs[-1] == "MANTLE_DONE"
                break
        else:
            raise AssertionError(f"no face of {o.bounds.size} could be climbed")


def _fast_route(ls, obs, o, n, run_up=12.0, takeoff=1.6):
    fc = _face_center(o, n)
    start = fc + n * run_up
    center = np.array([start[0], 0.9, start[2]], dtype=np.float32)
    if not BodyModel.pose_is_free(center, obs):
        return None
    sim = Sim(ls, obs, spawn=tuple(center))
    sim.face(-n)
    sim.hold("W", "SHIFT")
    face_plane = float(np.dot(fc, n))
    sim.run(6.0, until=lambda s: float(np.dot(s.pos, n)) - face_plane < 0.3 + takeoff)
    t_jump = sim.time
    sim.hold("SPACE", "E")
    ok = sim.run(7.0, until=lambda s: s.standing_on(o), top_out=True)
    return sim, ok, sim.time - t_jump


def test_tower_fast_route_wall_kick_to_jump_crevasse():
    ls, obs = build_playground()
    tower = obstacle_by_size(obs, 4.0, 5.4, 4.0)
    res = _fast_route(ls, obs, tower, FACES[0])   # south face, approached from +Z
    assert res is not None
    sim, ok, elapsed = res
    assert ok, "sprinting route did not reach the top"
    evs = [e for _, e in sim.events]
    assert "WALL_KICK" in evs, "at speed the leg plant on the wall should be used"
    assert "CLIMB_UP_FAST" in evs, "sprint held: climbs are done at speed"
    assert elapsed < 5.0, f"fast route took {elapsed:.1f} s"


def test_fast_route_is_faster_than_slow_route_on_pillar():
    ls, obs = build_playground()
    pillar = [o for o in obs if abs(o.bounds.min[0] - 6.0) < 1e-3 and abs(o.bounds.min[2] - 22.0) < 1e-3][0]
    n = FACES[1]   # north face approached from -Z (open runway)
    slow = _slow_climb(ls, obs, pillar, n)
    assert slow
    slow_t = slow.time
    fast = _fast_route(ls, obs, pillar, n)
    assert fast is not None and fast[1]
    assert fast[2] < slow_t


def test_traverse_wall_shimmy_then_climb():
    ls, obs = build_playground()
    wall = obstacle_by_size(obs, 0.8, 3.2, 18.0)
    sim = Sim(ls, obs, spawn=(-17.6, 0.9, 10.0), yaw=90.0)   # facing -X, the long east face
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("W")
    z0 = float(sim.pos[2])
    sim.hold("D")
    sim.run(5.0)
    sim.release("D")
    assert abs(float(sim.pos[2]) - z0) > 4.0, "shimmy along the long face"
    sim.hold("W")
    assert sim.run(5.0, until=lambda s: s.standing_on(wall), top_out=True)


def test_chasm_wall_eject_lands_on_opposite_tower():
    ls, obs = build_playground()
    t1, t2 = sorted([o for o in obs if abs(o.bounds.size[1] - 6.0) < 1e-3 and abs(o.bounds.size[0] - 4.0) < 1e-3],
                    key=lambda o: o.bounds.min[0])
    gap_x0 = float(t1.bounds.max[0])
    sim = Sim(ls, obs, spawn=(gap_x0 + 1.6, 0.9, 2.5), yaw=-90.0)   # facing +X: tower 2's west face
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert sim.run(4.0, until=lambda s: s.had_event("CLIMB_UP_DONE"))
    sim.release("W")
    sim.run(0.3)
    assert abs(sim.fsm.active_rail.height - 3.8) < 1e-3
    sim.hold("S")
    sim.tap("SPACE")
    sim.release("S")
    assert sim.had_event("WALL_EJECT")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    rail = sim.fsm.active_rail
    assert rail.owner is t1, "should catch the opposite tower"
    assert rail.height < 3.8, "a backward cat leap across 3.2 m must lose height"
    # and from there the tower can be topped out
    sim.hold("W", "E")
    assert sim.run(8.0, until=lambda s: s.standing_on(t1), top_out=True)


def test_pillar_lateral_dyno_reaches_next_pillar():
    ls, obs = build_playground()
    A = [o for o in obs if abs(o.bounds.min[0] - 6.0) < 1e-3 and abs(o.bounds.min[2] - 22.0) < 1e-3][0]
    B = [o for o in obs if abs(o.bounds.min[0] - 10.4) < 1e-3 and abs(o.bounds.min[2] - 25.5) < 1e-3][0]
    sim = Sim(ls, obs, spawn=(9.6, 0.9, 23.0), yaw=90.0)   # facing -X: pillar A's east face
    sim.hold("W", "E")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    assert sim.fsm.active_rail.owner is A
    assert sim.run(4.0, until=lambda s: s.had_event("CLIMB_UP_DONE"))
    sim.release("W")
    sim.run(0.3)
    sim.hold("A")                                # shimmy toward the end of the face nearest pillar B ...
    sim.run(0.5)
    sim.tap("SPACE")                             # ... then leap left along the face (+Z)
    assert sim.had_event("DYNO_LATERAL")
    sim.release("A")
    sim.face(np.array([1.0, 0.0, 0.0]))          # turn toward pillar B's west face mid-air
    sim.hold("W")
    sim.run(0.2)
    sim.release("W")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.LEDGE_HANG), f"missed: {sim.pos}"
    assert sim.fsm.active_rail.height < 3.8, "a lateral cat leap across ~2.8 m lands lower than it started"
    assert sim.fsm.active_rail.owner is B
    sim.hold("W", "E")
    assert sim.run(8.0, until=lambda s: s.standing_on(B), top_out=True)


def test_hurdle_runway_can_be_run_at_speed():
    """Zone 1: sprint straight down the runway; low hurdles are stepped, taller ones vaulted with E,
    the 2.2 m ledge at the end is caught and topped out with a W press."""
    ls, obs = build_playground()
    sim = Sim(ls, obs, spawn=(-6.0, 0.9, -24.0), yaw=180.0)   # runway runs toward +Z
    sim.hold("W", "SHIFT", "E")
    ok = sim.run(12.0, until=lambda s: s.pos[2] > -2.5 and s.player.on_ground, top_out=True)
    assert ok, f"runway not completed, stuck at z={sim.pos[2]:.1f} state={sim.state}"
    assert sim.player.respawn_count == 0
