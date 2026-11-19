"""The procedural city: layout sanity, climbability of every building, vehicles and props."""
import time
import numpy as np
from harness import *  # noqa
from world import props
from world.city import CityGenerator, Rect
from world.spatial import ObstacleIndex
from parkour.ledge_rail import RailType

_CITY = {}


def city():
    """Build the city once per test session (it takes a few seconds)."""
    if "ls" not in _CITY:
        t = time.time()
        ls = LedgeSystem()
        obstacles, verts, idx, gen = LevelGenerator.build_city(ls, seed=config.CITY_SEED)
        _CITY.update(ls=ls, obs=obstacles, gen=gen, build_time=time.time() - t, verts=verts, idx=idx)
    return _CITY


def _scene(parts):
    ls = LedgeSystem()
    floor = Obstacle(-60.0, -0.5, -60.0, 120.0, 0.5, 120.0, add_crevasses=False, rails=False)
    obs = [floor] + parts
    ls.build(obs)
    return ls, obs


# ------------------------------------------------------------------ layout
def test_city_is_much_larger_than_the_park_and_builds_quickly():
    c = city()
    ext = CityGenerator.EXTENT
    assert ext.w * ext.d >= 18 * (60 * 80), "the map should be more than an order of magnitude larger"
    assert c["build_time"] < 30.0
    st = c["gen"].stats
    assert st["buildings"] >= 60 and st["vehicles"] >= 200 and st["props"] >= 100 and st["setpieces"] >= 3
    assert len(c["ls"].rails) > 5000
    assert c["idx"].max() < len(c["verts"])


def test_city_is_deterministic_for_a_seed():
    ls1 = LedgeSystem()
    obs1, _, _, g1 = LevelGenerator.build_city(ls1, seed=3)
    ls2 = LedgeSystem()
    obs2, _, _, g2 = LevelGenerator.build_city(ls2, seed=3)
    assert len(obs1) == len(obs2) and g1.stats == g2.stats
    assert all(np.allclose(a.bounds.min, b.bounds.min) for a, b in zip(obs1[:500], obs2[:500]))
    ls3 = LedgeSystem()
    obs3, _, _, g3 = LevelGenerator.build_city(ls3, seed=4)
    assert g3.stats != g1.stats or len(obs3) != len(obs1)


def test_buildings_do_not_overlap_and_stay_inside_the_walls():
    c = city()
    rects = []
    for spec, parts in c["gen"].buildings:
        rects.append(Rect(spec.x, spec.z, spec.x + spec.width, spec.z + spec.depth))
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.overlaps(b), f"buildings overlap: {a} {b}"
    ext = CityGenerator.EXTENT
    for o in c["obs"]:
        if o.name == "city-wall" or o.bounds.size[0] > 200:
            continue
        assert ext.x0 - 0.01 <= o.bounds.min[0] and o.bounds.max[0] <= ext.x1 + 0.01, o
        assert ext.z0 - 0.01 <= o.bounds.min[2] and o.bounds.max[2] <= ext.z1 + 0.01, o


def test_training_park_is_preserved_inside_the_city():
    c = city()
    names = [o.name for o in c["obs"]]
    assert "building-a-cornice" in names and "lblock-annex" in names
    sizes = [tuple(o.bounds.size.round(1)) for o in c["obs"]]
    assert (4.0, 5.4, 4.0) in sizes, "the double-crevasse tower"
    assert (0.8, 3.2, 18.0) in sizes, "the traverse wall"
    # nothing from the city intrudes into the park rectangle
    park = CityGenerator.PARK
    for o in c["obs"]:
        if o.name.startswith(("bld-", "car", "van", "bus", "truck", "sidewalk", "deck", "containers", "frame")):
            r = Rect(o.bounds.min[0], o.bounds.min[2], o.bounds.max[0], o.bounds.max[2])
            assert not r.overlaps(park, pad=-0.5), f"{o} intrudes into the park"


def test_city_has_the_promised_variety():
    c = city()
    names = set(o.name.split("-")[-1] if o.name.startswith("bld") else o.name for o in c["obs"])
    for needed in ("awning", "balcony", "escape", "scaffold", "skybridge", "deck", "ramp", "containers",
                   "car", "van", "bus", "truck", "shelter", "billboard", "bench", "tree", "boulder", "fountain", "hill"):
        assert any(needed in n for n in names), f"no {needed} in the city"


# ------------------------------------------------------------ climbability
def _roof_reachable(ls, index, spec, parts):
    """Hold-graph search from ground-reachable holds to a mantle-able roof edge."""
    ids = set(id(p) for p in parts)
    mine = [r for r in ls.rails if id(r.owner) in ids]
    frontier = []
    for r in mine:
        if r.height <= config.STAND_REACH + 1e-3:
            hc = BodyModel.hang_center(r, 0.5)
            if BodyModel.pose_is_free(hc, index) and BodyModel.feet_y(hc) >= -0.05:
                frontier.append((r, 0.5))
    seen = set()
    steps = 0
    while frontier and steps < 300:
        r, s = frontier.pop()
        if id(r) in seen:
            continue
        seen.add(id(r))
        steps += 1
        if abs(r.height - spec.height) < 1e-3 and r.rail_type == RailType.MANTLE_TOP and BodyModel.mantle_is_valid(r, s, index):
            return True
        nxt = BodyModel.find_hold_above(ls.rails, r, s, index, None)
        if nxt is not None:
            frontier.append((nxt[0], nxt[1]))
        for link in r.links.values():
            e = BodyModel.corner_entry(link, index)
            if e is not None:
                frontier.append((link.rail, e))
    return False


def test_every_city_building_roof_is_reachable_by_climbing():
    c = city()
    index = ObstacleIndex(c["obs"])
    failed = [spec.name for spec, parts in c["gen"].buildings if not _roof_reachable(c["ls"], index, spec, parts)]
    assert not failed, f"unclimbable buildings: {failed}"


def test_a_city_building_is_climbed_by_the_scripted_climber():
    """Full simulation on one random city building: walk up, climb, top out."""
    c = city()
    spec, parts = c["gen"].buildings[5]
    roof = [p for p in parts if p.name.endswith("cornice")][0]
    # approach the door facade from the street
    face = spec.door_face
    if face == "south":
        spawn, n = (spec.x + spec.width * 0.5 + 1.1, 0.9, spec.z + spec.depth + 1.8), (0.0, 0.0, 1.0)
    elif face == "north":
        spawn, n = (spec.x + spec.width * 0.5 + 1.1, 0.9, spec.z - 1.8), (0.0, 0.0, -1.0)
    elif face == "east":
        spawn, n = (spec.x + spec.width + 1.8, 0.9, spec.z + spec.depth * 0.5 + 1.1), (1.0, 0.0, 0.0)
    else:
        spawn, n = (spec.x - 1.8, 0.9, spec.z + spec.depth * 0.5 + 1.1), (-1.0, 0.0, 0.0)
    ok = False
    from physics.collision import CollisionSystem
    for lateral in (0.0, 1.1, -1.1, 2.2, -2.2):
        t = np.cross(np.array(n), [0.0, 1.0, 0.0])
        sp = np.array(spawn) + t * lateral
        ground = CollisionSystem.ground_height_below(sp + np.array([0.0, 3.0, 0.0]), BodyModel.SIZE, c["obs"], max_drop=5.0)
        if ground is None:
            continue
        sp[1] = ground + config.PLAYER_HALF_HEIGHT + 0.002        # stand on the sidewalk, not in it
        if not BodyModel.pose_is_free(sp, c["obs"]):
            continue
        sim = Sim(c["ls"], c["obs"], spawn=tuple(sp))
        sim.face(-np.array(n))
        if sim.climb_to(roof, timeout=20.0, keys=("W", "E")):
            ok = True
            break
    assert ok, f"could not climb {spec.name} ({spec.floors} floors)"


# ---------------------------------------------------------------- vehicles
def test_cars_are_run_over_lengthwise_and_vaulted_from_the_side():
    ls, obs = _scene(props.car(0.0, -6.0, "z"))
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0))
    sim.tap("CAPS")
    sim.hold("W")
    sim.run(4.0)
    assert sim.pos[2] < -9.0, "should have run over the car"
    assert max(BodyModel.feet_y(t["pos"]) for t in sim.trace) >= 1.4, "over the roof"
    ls, obs = _scene(props.car(0.0, -6.0, "x"))
    cabin = obs[-1]
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 0.0))
    assert sim.climb_to(cabin, timeout=5.0, keys=("W", "SPACE"))


def test_van_bus_and_truck_roofs():
    ls, obs = _scene(props.van(0.0, -6.0, "x"))
    assert Sim(ls, obs, spawn=(0.0, 0.9, -1.0)).climb_to(obs[-1], 8.0, keys=("W", "SPACE"))
    ls, obs = _scene(props.bus(0.0, -8.0, "x"))
    assert Sim(ls, obs, spawn=(0.0, 0.9, -1.0)).climb_to(obs[-1], 8.0, keys=("W", "SPACE"))
    parts = props.truck(0.0, -8.0, "x")
    ls, obs = _scene(parts)
    cab = [p for p in parts if "cab" in p.name][0]
    trailer = [p for p in parts if "trailer" in p.name][0]
    sim = Sim(ls, obs, spawn=(-4.6, 0.9, -3.0), yaw=0.0)
    assert sim.climb_to(cab, 8.0, keys=("W", "SPACE")), "cab roof"
    sim.release_all()
    sim.run(0.3)
    sim.face(np.array([1.0, 0.0, 0.0]))
    assert sim.climb_to(trailer, 8.0, keys=("W", "E")), "trailer from the cab"
    assert abs(sim.feet_y - 4.0) < 0.05


# ------------------------------------------------------------------- props
def test_props_are_climbable():
    cases = [
        ("container stack", props.container_stack(0.0, -8.0, "x"), -1, (0.0, 0.9, -1.0)),
        ("bus shelter", props.bus_shelter(0.0, -6.0, "x"), "roof", (0.0, 0.9, -1.0)),
        ("boulder", props.boulder(0.0, -6.0), -1, (0.0, 0.9, -1.0)),
        ("terraced hill", props.terrace_hill(0.0, -12.0), -1, (0.0, 0.9, -1.0)),
        ("fountain", props.fountain(0.0, -8.0), -1, (0.0, 0.9, -1.0)),
        ("dumpster", props.dumpster(0.0, -5.0), 0, (0.0, 0.9, -1.0)),
        ("tree", props.tree(0.0, -6.0), "canopy", (2.0, 0.9, -2.0)),
    ]
    for name, parts, tgt, spawn in cases:
        ls, obs = _scene(parts)
        target = parts[tgt] if isinstance(tgt, int) else [p for p in parts if tgt in p.name][0]
        sim = Sim(ls, obs, spawn=spawn)
        assert sim.climb_to(target, 14.0, keys=("W", "SPACE")), f"{name} not climbable: {[e for _, e in sim.events][:8]}"


def test_parking_deck_is_reached_by_walking_up_the_ramp():
    parts = props.parking_deck(-8.0, -20.0)
    ls, obs = _scene(parts)
    deck = parts[0]
    sim = Sim(ls, obs, spawn=(10.0, 0.9, -24.0), yaw=180.0)
    sim.tap("CAPS")
    sim.hold("W")
    assert sim.run(6.0, until=lambda s: s.feet_y > 2.9), "ramp not climbed"
    assert not sim.had_event("VAULT") and ParkourState.MANTLE_VAULT not in sim.states, "a ramp is walked, not vaulted"
    sim.release("W")
    sim.run(0.2)
    sim.face(np.array([-1.0, 0.0, 0.0]))
    sim.hold("W")
    assert sim.run(4.0, until=lambda s: s.standing_on(deck))


def test_scaffold_is_a_ladder_of_platforms():
    from world.building import BuildingGenerator, BuildingSpec, NORTH, SOUTH
    spec = BuildingSpec(-5.0, -16.0, 10.0, 8.0, floors=3, facade_faces=(SOUTH,), scaffold=(NORTH, 1.0, 6.0, 3), rooftop=False, name="b")
    parts = BuildingGenerator.build(spec)
    ls, obs = _scene(parts)
    top_platform = [p for p in parts if p.name.endswith("scaffold")][-1]
    assert abs(top_platform.bounds.max[1] - 6.0) < 1e-3
    sim = Sim(ls, obs, spawn=(0.0, 0.9, -19.5), yaw=180.0)
    assert sim.climb_to(top_platform, 14.0, keys=("W", "E")), [e for _, e in sim.events][:10]
