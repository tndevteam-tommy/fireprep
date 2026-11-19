"""Building holds: every wall climbs to its roof on its own, every course of holds wraps the whole
building, attachments never interpenetrate, and small collisions near a hold are buffered."""
import numpy as np

from harness import *  # noqa
from physics.collision import CollisionSystem
from parkour.ledge_rail import RailType
from world.building import BuildingGenerator, BuildingSpec, SOUTH, NORTH, EAST, WEST
from world.spatial import ObstacleIndex

FACES = {SOUTH: (0, 0, 1), NORTH: (0, 0, -1), EAST: (1, 0, 0), WEST: (-1, 0, 0)}


def _scene(parts):
    ls = LedgeSystem()
    floor = Obstacle(-80.0, -0.5, -80.0, 160.0, 0.5, 160.0, add_crevasses=False)
    floor.rails.clear()
    obs = [floor] + parts
    ls.build(obs)
    return ls, obs


def _ground_starts(rail, obs):
    """Spots on `rail` a player standing on the ground in front of it can grab and hang from."""
    out = []
    n = max(1, int(rail.length / 0.5))
    for i in range(n + 1):
        s = rail.clamp_s(i / n)
        g = rail.get_point(s)
        for back in (0.31, 0.45, 0.6):
            c = g + rail.normal * back
            ground = CollisionSystem.ground_height_below(c + np.array([0, 2.5, 0], np.float32), BodyModel.SIZE, obs, max_drop=4.0)
            if ground is None:
                continue
            c[1] = ground + config.PLAYER_HALF_HEIGHT + 0.001
            if not BodyModel.pose_is_free(c, obs):
                continue
            res = BodyModel.evaluate_grab(rail, c, -rail.normal, -rail.normal, True, obs)
            if res is not None and res.rail is rail and res.mode == "HANG":
                out.append(res.s)
                break
    return out


def face_reaches_roof(ls, obs, spec, parts, face):
    """Climbing one wall with no corner turns: grab from the ground, shimmy along holds, climb to
    any hold above (aiming at it), mantle onto platforms and reach up from standing on them."""
    ids = set(id(p) for p in parts)
    fn = np.array(FACES[face], np.float32)
    mine = [r for r in ls.rails if id(r.owner) in ids]
    on_face = [r for r in mine if float(np.dot(r.normal, fn)) > 0.99]
    roofs = [r for r in on_face if r.rail_type == RailType.MANTLE_TOP and abs(r.height - spec.height) < 1e-3]
    frontier = [(r, s) for r in on_face if r.height <= config.STAND_REACH + 0.01 for s in _ground_starts(r, obs)]
    seen = set()
    while frontier and len(seen) < 3000:
        r, s = frontier.pop()
        key = (id(r), round(s * r.length / 0.5))
        if key in seen:
            continue
        seen.add(key)
        if r in roofs and BodyModel.mantle_is_valid(r, s, obs):
            return True
        for ds in (-0.5, 0.5):
            s2 = s + ds / max(r.length, 1e-6)
            if r.s_margin() <= s2 <= 1 - r.s_margin() and BodyModel.pose_is_free(BodyModel.hang_center(r, s2), obs):
                frontier.append((r, s2))
        eye = BodyModel.eye_pos(BodyModel.hang_center(r, s))
        grip = r.get_point(s)
        looks = [None]
        for ci in BodyModel._holds_near(ls.rails, grip, r.height + 0.3, r.height + config.CLIMB_HOLD_GAP_MAX, radius=2.0):
            c = ls.rails[ci]
            if id(c.owner) in ids:
                g = c.get_point(c.clamp_s(c.project_s(grip)))
                looks.append((g - eye) / (np.linalg.norm(g - eye) + 1e-7))
        for look in looks:
            nxt = BodyModel.find_hold_above(ls.rails, r, s, obs, look)
            if nxt is not None and id(nxt[0].owner) in ids:
                frontier.append((nxt[0], nxt[1]))
        if r.rail_type == RailType.MANTLE_TOP and BodyModel.mantle_is_valid(r, s, obs):
            stand = BodyModel.mantle_target_center(r, s, obs)
            for ci in BodyModel._holds_near(ls.rails, stand, r.height + 0.3, r.height + config.STAND_REACH, radius=2.5):
                c = ls.rails[ci]
                if id(c.owner) not in ids or c is r:
                    continue
                s0 = c.clamp_s(c.project_s(stand))
                for back in (0.31, 0.45, 0.6, 0.8):
                    p = c.get_point(s0) + c.normal * back
                    p[1] = r.height + config.PLAYER_HALF_HEIGHT + 0.001
                    if not BodyModel.pose_is_free(p, obs):
                        continue
                    gy = CollisionSystem.ground_height_below(p + np.array([0, 0.05, 0], np.float32), BodyModel.SIZE, obs, max_drop=0.3)
                    if gy is None or abs(gy - r.height) > 0.05:
                        continue
                    res = BodyModel.evaluate_grab(c, p, -c.normal, -c.normal, True, obs)
                    if res is not None and res.rail is c:
                        frontier.append((c, res.s))
                        break
    return False


def _specs():
    """A building of every height with plain walls next to facades, and the attachments."""
    out = []
    for floors in range(2, 7):
        out.append(BuildingSpec(0.0, -20.0, 12.0, 10.0, floors, facade_faces=(WEST, SOUTH), door_face=SOUTH,
                                awning=True, rooftop=False, name=f"f{floors}"))
    out.append(BuildingSpec(0.0, -20.0, 14.0, 10.0, 4, facade_faces=(EAST,), door_face=EAST,
                            scaffold=(SOUTH, 0.5, 13.0, 3), balconies=((NORTH, 1, 2.0), (NORTH, 2, 7.0)),
                            fire_escape=(WEST, 3.0), rooftop=True, name="attached"))
    out.append(BuildingSpec(0.0, -20.0, 12.0, 10.0, 3, facade_faces=(), rooftop=False, name="plain"))
    return out


# ------------------------------------------------------------------ reaching the roof
def test_every_wall_of_every_building_climbs_to_its_roof_on_its_own():
    """Regression: plain walls of 2- and 5-storey buildings dead-ended 2.2 m under the roof
    (3.8 / 12.8 m), and a scaffold's top deck had no hold in reach above it."""
    failed = []
    for spec in _specs():
        parts = BuildingGenerator.build(spec)
        ls, obs = _scene(parts)
        for face in (SOUTH, NORTH, EAST, WEST):
            if not face_reaches_roof(ls, obs, spec, parts, face):
                failed.append((spec.name, face))
    assert not failed, f"walls that do not lead to the roof: {failed}"


def test_plain_wall_courses_leave_no_gap_beyond_a_climb():
    for floor_h in (3.0, 3.5, 4.0, 5.0):
        spec = BuildingSpec(0.0, 0.0, 10.0, 10.0, 4, floor_height=floor_h, facade_faces=(SOUTH,))
        ys = BuildingGenerator.course_heights(spec)
        chain = ys + [spec.height]
        assert ys[0] <= config.STAND_REACH, ys
        for lo, hi in zip(chain[:-1], chain[1:]):
            assert 0.3 <= hi - lo <= config.CLIMB_HOLD_GAP_MAX + 1e-9, (floor_h, lo, hi)
    spec = BuildingSpec(0.0, 0.0, 10.0, 10.0, 3)
    assert BuildingGenerator.course_heights(spec) == [1.0, 2.2, 4.0, 5.2, 7.0, 8.2], \
        "standard storeys: the courses are the facades' sill and window-head lines"


def test_plain_wall_of_a_two_storey_building_climbs_straight_up_with_w():
    spec = BuildingSpec(0.0, -20.0, 12.0, 10.0, 2, facade_faces=(WEST, SOUTH), door_face=SOUTH, rooftop=False, name="b")
    parts = BuildingGenerator.build(spec)
    ls, obs = _scene(parts)
    roof = [p for p in parts if p.name.endswith("cornice")][0]
    sim = Sim(ls, obs, spawn=(13.2, 0.9, -15.0))
    sim.face(np.array([-1.0, 0.0, 0.0]))
    sim.hold("W")
    assert sim.run(1.0, until=lambda s: s.player.grab_candidate() is not None)
    sim.hold("E")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("E")
    assert sim.run(12.0, until=lambda s: s.standing_on(roof), top_out=True), [e for _, e in sim.events]
    assert ParkourState.CORNER_TURN not in sim.states, "straight up the plain wall"


def test_a_scaffold_top_deck_always_has_a_hold_within_reach_above_it():
    deck = 6.0
    lo, hi = deck + BuildingGenerator.PLATFORM_HOLD_MIN, deck + BuildingGenerator.PLATFORM_HOLD_MAX
    # 3 m storeys: the 8.2 m course is already in reach of the 6 m deck - nothing is added
    spec = BuildingSpec(0.0, -20.0, 12.0, 10.0, 4, facade_faces=(SOUTH,), door_face=SOUTH,
                        scaffold=(NORTH, 0.5, 11.0, 3), rooftop=False, name="std")
    core = [p for p in BuildingGenerator.build(spec) if p.name.endswith("core")][0]
    north = sorted(round(r.height, 3) for r in core.rails if r.normal[2] < -0.99 and r.is_crevasse)
    assert north == [round(y, 3) for y in BuildingGenerator.course_heights(spec)], north
    # 4.5 m storeys: no course lands 1.65 - 2.2 m above the deck (6.7 / 8.6 m), so one is added there
    spec = BuildingSpec(0.0, -20.0, 12.0, 10.0, 3, floor_height=4.5, facade_faces=(SOUTH,), door_face=SOUTH,
                        scaffold=(NORTH, 0.5, 11.0, 3), rooftop=False, name="tall")
    assert not any(lo <= y <= hi for y in BuildingGenerator.course_heights(spec))
    parts = BuildingGenerator.build(spec)
    core = [p for p in parts if p.name.endswith("core")][0]
    added = [r for r in core.rails if r.normal[2] < -0.99 and r.is_crevasse and lo <= r.height <= hi]
    assert len(added) == 1, sorted(r.height for r in core.rails if r.normal[2] < -0.99)
    x0, x1 = sorted((float(added[0].p1[0]), float(added[0].p2[0])))
    assert abs(x0 - spec.x) < 1e-6 and abs(x1 - (spec.x + spec.width)) < 1e-6, "corner to corner like every course"
    ls, obs = _scene(parts)
    assert face_reaches_roof(ls, obs, spec, parts, NORTH)


# ----------------------------------------------------------------------- wrapping
def _wall_rows(ls, spec, parts):
    """Hold rows on the building's outer walls (and its roof edge), by height."""
    ids = set(id(p) for p in parts)
    x0, z0, x1, z1 = spec.x, spec.z, spec.x + spec.width, spec.z + spec.depth
    planes = {(0, 0, 1): (2, z1), (0, 0, -1): (2, z0), (1, 0, 0): (0, x1), (-1, 0, 0): (0, x0)}
    rows = {}
    for r in ls.rails:
        if id(r.owner) not in ids or abs(r.normal[1]) > 0.1:
            continue
        if abs(r.height - spec.height) < 1e-3:
            ok = r.owner.name.endswith("cornice")
        else:
            axis, v = planes[tuple(int(round(c)) for c in r.normal)]
            ok = abs(r.p1[axis] - v) < 0.02 and not any(k in r.owner.name for k in
                                                        ("balcony", "escape", "scaffold", "awning", "rooftop"))
        if ok:
            rows.setdefault(round(r.height, 3), []).append(r)
    return rows


def _corners_turned(rows_at_h, spec, is_roof):
    x0, z0, x1, z1 = spec.x, spec.z, spec.x + spec.width, spec.z + spec.depth
    tol = 0.45 if is_roof else 0.05
    turned = 0
    for cx, cz in ((x0, z1), (x1, z1), (x1, z0), (x0, z0)):
        if any(abs(end[0] - cx) < tol and abs(end[2] - cz) < tol and side in r.links and r.links[side].kind == "convex"
               for r in rows_at_h for end, side in ((r.p2, "right"), (r.p1, "left"))):
            turned += 1
    return turned


def test_every_course_wraps_the_whole_building():
    """Regression: plain walls (2.0 / 3.8 / 5.6 ... m) and facades (1.0 / 2.2 / 4.0 ... m) never met, and the
    plain courses stopped 0.5 m short of the corners - no row of holds went round a building."""
    for spec in _specs():
        parts = BuildingGenerator.build(spec)
        ls, _ = _scene(parts)
        for h, rails in _wall_rows(ls, spec, parts).items():
            assert _corners_turned(rails, spec, abs(h - spec.height) < 1e-3) == 4, (spec.name, h)


def test_a_traverse_goes_all_the_way_round_a_building():
    spec = BuildingSpec(0.0, -20.0, 12.0, 10.0, 2, facade_faces=(WEST, SOUTH), door_face=SOUTH, rooftop=False, name="b")
    parts = BuildingGenerator.build(spec)
    ls, obs = _scene(parts)
    sim = Sim(ls, obs, spawn=(13.2, 0.9, -15.0))
    sim.face(np.array([-1.0, 0.0, 0.0]))
    sim.hold("W")
    sim.run(1.0, until=lambda s: s.player.grab_candidate() is not None)
    sim.hold("E")
    sim.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("E")
    assert sim.run_until_event("CLIMB_UP_DONE", 4.0)
    sim.release("W")
    sim.run(0.3)
    h = sim.fsm.active_rail.height
    sim.hold("D")
    faces = []
    for _ in range(int(50.0 / DT)):
        before = sim.state
        sim.step()
        if before == ParkourState.CORNER_TURN and sim.state != ParkourState.CORNER_TURN:
            faces.append(tuple(int(round(c)) for c in sim.fsm.active_rail.normal))
            assert abs(sim.fsm.active_rail.height - h) < 1e-3, "the course stays level all the way round"
        if len(faces) >= 4:
            break
    assert len(set(faces)) == 4, f"turned onto {faces}: the traverse must visit all four walls"


# --------------------------------------------------------------------- the city
def _city():
    from test_city import city
    return city()


def test_every_city_course_wraps_its_building():
    c = _city()
    for spec, parts in c["gen"].buildings:
        for h, rails in _wall_rows(c["ls"], spec, parts).items():
            assert _corners_turned(rails, spec, abs(h - spec.height) < 1e-3) == 4, (spec.name, h)


def test_city_plain_walls_have_climbable_courses():
    c = _city()
    for spec, parts in c["gen"].buildings:
        core = [p for p in parts if p.name.endswith("core")][0]
        for face, n in FACES.items():
            if face in spec.facade_faces:
                continue
            ys = sorted(set(round(r.height, 3) for r in core.rails if np.allclose(r.normal, n)))
            chain = ys + [spec.height]
            assert ys and ys[0] <= config.STAND_REACH, (spec.name, face)
            assert all(b - a <= config.CLIMB_HOLD_GAP_MAX + 1e-6 for a, b in zip(chain[:-1], chain[1:])), (spec.name, face, ys)


def test_nothing_in_the_city_interpenetrates_another_structure():
    """Regression: billboard / lamp posts stood through awnings and skybridges ran through balconies
    and fire-escape landings (a slab 10 cm above the bridge's edge, overlapping it)."""
    c = _city()
    obs = c["obs"]
    index = ObstacleIndex(obs)

    def group(o):
        n = o.name or "block"
        parts = n.split("-")
        return parts[0] + ("-" + parts[1] if n.startswith("bld-") else "")
    skip = ("", "sidewalk", "city-wall")
    bad = []
    for a in obs:
        if a.name in skip or a.bounds.size[0] > 200:
            continue
        for b in index.query(a.bounds.min - 0.01, a.bounds.max + 0.01):
            if b is a or id(b) < id(a) or b.name in skip or b.bounds.size[0] > 200 or group(a) == group(b):
                continue
            ov = np.minimum(a.bounds.max, b.bounds.max) - np.maximum(a.bounds.min, b.bounds.min)
            if (ov > 0.005).all():
                bad.append((a.name, b.name))
    assert not bad, bad
    assert sum(1 for o in obs if o.name == "skybridge") >= 3, "bridges were moved clear, not dropped"


# --------------------------------------------------------------- collision buffers
def _awning_building():
    spec = BuildingSpec(0.0, -20.0, 12.0, 10.0, 3, facade_faces=(SOUTH,), door_face=SOUTH, rooftop=False,
                        awning=True, name="b")
    parts = BuildingGenerator.build(spec)
    ls, obs = _scene(parts)
    return spec, parts, ls, obs, [p for p in parts if p.name.endswith("awning")][0]


def test_a_climb_blocked_by_an_awning_shifts_along_the_hold_and_climbs():
    """Hanging from the window head beside the door, right under the awning's overhang: the climb
    used to be blocked for good; now the climber shifts out from under it and climbs."""
    spec, parts, ls, obs, aw = _awning_building()
    x = float(aw.bounds.min[0]) + 0.2                         # under the awning's overlap of the pier
    sim = Sim(ls, obs, spawn=(x, 0.9, -8.5))
    sim.face(np.array([0.0, 0.0, -1.0]))
    sim.hold("W")
    sim.run(2.0, until=lambda s: s.player.grab_candidate() is not None)
    sim.release("W")
    sim.hold("E")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("E")
    sim.run(0.3)
    hold = sim.fsm.active_rail
    assert abs(hold.height - 2.2) < 1e-3 and float(sim.pos[0]) > float(aw.bounds.min[0]), "hanging under the awning"
    assert BodyModel.find_hold_above(ls.rails, hold, sim.fsm.rail_s, obs) is None, "straight up is blocked here"
    x_before = float(sim.pos[0])
    sim.hold("W")
    assert sim.run(3.0, until=lambda s: s.state == ParkourState.CLIMB_UP), [e for _, e in sim.events][-6:]
    shift = abs(float(sim.pos[0]) - x_before)
    assert 0.05 < shift <= config.CLIMB_SHIFT_MAX + 0.05, f"shifted {shift:.2f} m along the hold"
    assert ParkourState.CORNER_TURN not in sim.states
    # ... on up the wall, or onto the awning itself (its side edge is the nearest hold once clear of it)
    assert sim.run(4.0, until=lambda s: s.standing_on(aw) or
                   (s.state == ParkourState.LEDGE_HANG and s.fsm.active_rail.height > 3.0)), [e for _, e in sim.events][-6:]


def test_climb_shift_never_kicks_in_without_a_clear_spot_in_range_and_stops_on_release():
    # a window sill (no room to mantle) on a wall with nothing above: W just reports it is blocked
    ls, obs = build_custom([(-3.0, 0.0, -4.0, 6.0, 2.0, 1.0, False)])
    wall_top = [r for r in ls.rails if abs(r.height - 2.0) < 1e-3 and r.normal[2] > 0.5][0]
    # a slab right over the whole wall top: no mantle, no hold above, anywhere along it
    ls, obs = build_custom([(-3.0, 0.0, -4.0, 6.0, 2.0, 1.0, False), (-4.0, 2.3, -4.0, 8.0, 0.3, 2.2, False)])
    wall_top = [r for r in ls.rails if abs(r.height - 2.0) < 1e-3 and r.normal[2] > 0.5][0]
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -2.6))
    sim.face(np.array([0.0, 0.0, -1.0]))
    sim.hold("E")
    assert sim.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("E")
    sim.run(0.3)
    x0 = float(sim.pos[0])
    sim.hold("W")
    sim.run(1.0)
    assert sim.state == ParkourState.LEDGE_HANG and abs(float(sim.pos[0]) - x0) < 1e-3, "nothing in range: stay put"
    assert sim.fsm.last_event in ("CLIMB_BLOCKED", "MANTLE_BLOCKED")
    # with a clear spot in range the shift starts - and releasing W stops it where it is
    spec, parts, ls, obs, aw = _awning_building()
    sim = Sim(ls, obs, spawn=(float(aw.bounds.min[0]) + 0.2, 0.9, -8.5))
    sim.face(np.array([0.0, 0.0, -1.0]))
    sim.hold("W")
    sim.run(2.0, until=lambda s: s.player.grab_candidate() is not None)
    sim.release("W")
    sim.hold("E")
    sim.run(1.0, until=lambda s: s.state == ParkourState.LEDGE_HANG)
    sim.release("E")
    sim.run(0.3)
    sim.hold("W")
    sim.step(4)
    assert sim.fsm.climb_shift is not None and sim.state == ParkourState.SHIMMY
    sim.release("W")
    sim.step()
    x1 = float(sim.pos[0])
    sim.run(0.5)
    assert sim.fsm.climb_shift is None and sim.state == ParkourState.LEDGE_HANG and abs(float(sim.pos[0]) - x1) < 1e-3


def test_a_catch_slides_clear_of_a_post_a_few_centimetres_away_but_not_through_a_wall():
    # a hold whose hang pose clips a thin post by a few cm: the hands shift a hand-width and it holds
    ls, obs = build_custom([(-3.0, 0.0, -4.0, 6.0, 3.2, 1.0, False), (0.26, 0.0, -3.0, 0.2, 2.0, 0.2, False)])
    edge = [r for r in ls.rails if abs(r.height - 3.2) < 1e-3 and r.normal[2] > 0.5][0]
    body = np.array([0.0, 3.2 - 1.45 + config.PLAYER_HALF_HEIGHT - 0.1, -2.55], np.float32)
    assert not BodyModel.pose_is_free(BodyModel.hang_center(edge, edge.clamp_s(edge.project_s(body))), obs)
    res = BodyModel.evaluate_grab(edge, body, np.array([0, 0, -1.0]), np.array([0, 0, -1.0]), False, obs, np.zeros(3))
    assert res is not None and res.mode == "HANG"
    assert BodyModel.pose_is_free(res.hang_center, obs, shrink=0.001)
    moved = abs(float(res.grip[0]) - 0.0)
    assert 0.0 < moved <= config.EDGE_COLLISION_BUFFER + 1e-3, moved
    # a post that blocks the pose by more than the buffer is still a real obstruction
    ls, obs = build_custom([(-3.0, 0.0, -4.0, 6.0, 3.2, 1.0, False), (-0.2, 0.0, -3.0, 0.6, 2.0, 0.3, False)])
    edge = [r for r in ls.rails if abs(r.height - 3.2) < 1e-3 and r.normal[2] > 0.5][0]
    res = BodyModel.evaluate_grab(edge, body, np.array([0, 0, -1.0]), np.array([0, 0, -1.0]), False, obs, np.zeros(3))
    assert res is None or res.mode == "MANTLE", "never offered as a hang through the post (a vault over the top is fine)"
