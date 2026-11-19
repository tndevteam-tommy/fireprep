"""Vehicles on their tyres with windows, doors and lights; corrugated containers with doors,
firmly on the ground; boots with rounded toes."""
import numpy as np
from harness import *  # noqa
from world import props
from world.obstacle import (VERTEX_FLOATS, MAT_CAR_BODY, MAT_CAR_GLASS, MAT_VAN, MAT_BUS, MAT_TRUCK_CAB,
                            MAT_CONTAINER, MAT_WHEEL_SIDE, MAT_WHEEL_TREAD, ROLE_SIDE, ROLE_FRONT, ROLE_BACK,
                            ROLE_TOP)
from animation.arm_rig import FirstPersonArmRig, MAT_BOOT
from animation.skeleton import BoneID

WHEEL_SIDES = 14


def _verts(parts):
    return np.concatenate([p.vertices for p in parts], axis=0)


def _codes(v):
    """material * 10 + role (the variant stripped off)."""
    return (v[:, 11] + 0.5).astype(int) % 1000


def _wheels(parts):
    """Tyres are drawn as WHEEL_SIDES tread quads each."""
    v = _verts(parts)
    codes = _codes(v)
    tread = v[codes // 10 == MAT_WHEEL_TREAD]
    return tread, len(tread) // (4 * WHEEL_SIDES)


def _check_vehicle(parts, expected_wheels, body_material):
    v = _verts(parts)
    assert v.shape[1] == VERTEX_FLOATS
    tread, n = _wheels(parts)
    assert n == expected_wheels, f"{n} tyres, expected {expected_wheels}"
    assert abs(float(tread[:, 1].min()) - 0.0) < 0.02, "tyres sit on the ground"
    body = parts[0]
    gap = float(body.bounds.min[1])
    assert 0.25 <= gap <= 0.95, "the body is lifted off the ground"
    assert float(tread[:, 1].max()) > gap + 0.05, "the tyres reach up into the wheel arches"
    # the tyres' outer faces stand a few centimetres proud of the body sides (no painted arches)
    b = body.bounds
    lat = 0 if abs(float(b.size[2])) > abs(float(b.size[0])) else 2
    along = 2 - lat
    jut_lo = float(b.min[lat] - tread[:, lat].min())
    jut_hi = float(tread[:, lat].max() - b.max[lat])
    assert 0.02 <= jut_lo <= 0.08 and 0.02 <= jut_hi <= 0.08, f"tyres jut {jut_lo:.3f} / {jut_hi:.3f} m"
    assert tread[:, along].min() >= b.min[along] - 0.02 and tread[:, along].max() <= b.max[along] + 0.02
    codes = _codes(v)
    mats = set((codes // 10).tolist())
    assert body_material in mats and MAT_WHEEL_SIDE in mats and MAT_WHEEL_TREAD in mats
    roles = set((codes[codes // 10 == body_material] % 10).tolist())
    assert {ROLE_SIDE, ROLE_FRONT, ROLE_BACK, ROLE_TOP} <= roles, "the body knows its front, back and sides"
    # metric uv: the side face's u spans the body length, v its height
    side = v[(codes // 10 == body_material) & (codes % 10 == ROLE_SIDE)]
    assert abs(float(side[:, 6].max() - side[:, 6].min()) - max(b.size[0], b.size[2])) < 1e-3
    assert abs(float(side[:, 7].max() - side[:, 7].min()) - float(b.size[1])) < 1e-3


def test_car_has_four_tyres_a_detailed_body_and_a_glass_cabin():
    for axis in ("x", "z"):
        parts = props.car(3.0, -2.0, axis)
        _check_vehicle(parts, 4, MAT_CAR_BODY)
        assert parts[1].material == MAT_CAR_GLASS
        assert parts[0].facing is not None and parts[1].facing is not None
        assert abs(float(parts[0].bounds.max[1]) - 1.0) < 1e-6, "hood still at 1.0 m: run over it"


def test_van_and_bus_tyres():
    _check_vehicle(props.van(0.0, 0.0, "z"), 4, MAT_VAN)
    _check_vehicle(props.bus(0.0, 0.0, "x"), 6, MAT_BUS)          # single steer tyres, dual rear tyres


def test_semi_truck_is_an_eighteen_wheeler_carrying_a_container():
    for axis in ("x", "z"):
        parts = props.truck(0.0, 0.0, axis)
        cab, trailer = parts
        assert cab.material == MAT_TRUCK_CAB and trailer.material == MAT_CONTAINER
        tread, n = _wheels(parts)
        assert n == 18, f"{n} tyres on the semi"
        assert abs(float(tread[:, 1].min())) < 0.02
        # the cab sits clear in front of the container (not inside it), with a small gap
        ax = 0 if axis == "x" else 2
        assert not cab.bounds.intersects(trailer.bounds), "the cab is not submerged in the container"
        gap = float(trailer.bounds.min[ax] - cab.bounds.max[ax])
        assert 0.2 <= gap <= 0.6, f"cab-to-container gap {gap:.2f} m"
        # every tyre is under the cab or the container
        for o in (cab, trailer):
            v = o.vertices
            codes = _codes(v)
            t = v[codes // 10 == MAT_WHEEL_TREAD]
            assert t[:, ax].min() >= min(cab.bounds.min[ax], trailer.bounds.min[ax]) - 0.05
            assert t[:, ax].max() <= max(cab.bounds.max[ax], trailer.bounds.max[ax]) + 0.05
        # the container's doors (back role) are at the end away from the cab
        v = trailer.vertices
        codes = _codes(v)
        back = v[(codes // 10 == MAT_CONTAINER) & (codes % 10 == ROLE_BACK)]
        assert len(back) >= 4
        cab_c = float(cab.bounds.center[ax])
        assert abs(float(back[:, ax].mean()) - cab_c) > 6.0, "doors at the far end from the cab"
        # the cab faces away from the container
        assert float(cab.facing[ax]) < 0.0


def test_containers_sit_on_the_ground_and_are_corrugated_with_doors_at_one_end():
    parts = props.container(2.0, 3.0, 0.0, "x", props.COL_CONTAINER[1])
    c = parts[0]
    assert abs(float(c.bounds.min[1])) < 1e-6, "firmly on the ground"
    assert c.material == MAT_CONTAINER and c.facing is not None
    codes = _codes(c.vertices)
    roles = codes % 10
    assert (roles == ROLE_BACK).sum() == 4 and (roles == ROLE_FRONT).sum() == 4, "exactly one door end"
    stack = props.container_stack(0.0, 0.0, "z")
    assert all(p.material == MAT_CONTAINER for p in stack)
    assert sorted(round(float(p.bounds.min[1]), 3) for p in stack) == [0.0, 0.0, 2.6]
    # the truck's container is the same material (same grooves and doors)
    assert props.truck(0.0, 0.0, "z")[1].material == MAT_CONTAINER


_CITY = {}


def _city():
    if "c" not in _CITY:
        from world.generator import LevelGenerator
        from parkour.ledge_system import LedgeSystem
        ls = LedgeSystem()
        _CITY["c"] = LevelGenerator.build_city(ls, seed=config.CITY_SEED)
    return _CITY["c"]


def test_city_vehicles_all_stand_on_tyres():
    obstacles, verts, idx, city = _city()
    assert verts.shape[1] == VERTEX_FLOATS and idx.max() < len(verts)
    codes = _codes(verts)
    n_tyres = int((codes // 10 == MAT_WHEEL_TREAD).sum()) // (4 * WHEEL_SIDES)
    assert n_tyres >= 4 * city.stats["vehicles"], "every vehicle has at least four tyres"
    assert set((codes // 10).tolist()) >= {MAT_CAR_BODY, MAT_VAN, MAT_BUS, MAT_TRUCK_CAB, MAT_CONTAINER}


def test_city_vehicles_come_in_many_colours_and_bus_liveries():
    obstacles, verts, idx, city = _city()
    cabs = [o for o in obstacles if o.name.endswith("-cab")]
    trailers = [o for o in obstacles if o.name.endswith("-trailer")]
    vans = [o for o in obstacles if o.name == "van"]
    buses = [o for o in obstacles if o.name == "bus"]
    assert len(set(tuple(np.round(o.color, 2)) for o in cabs)) >= 3, "truck cabs in several colours"
    assert len(set(tuple(np.round(o.color, 2)) for o in trailers)) >= 3, "the containers they carry too"
    assert len(set(tuple(np.round(o.color, 2)) for o in vans)) >= 3, "vans too"
    school = [b for b in buses if b.variant == 0]
    metro = [b for b in buses if b.variant >= 1]
    assert 1 <= len(school) < len(metro), "a couple of yellow school buses, mostly metro buses"
    assert len(set(b.variant for b in metro)) >= 3, "metro buses in several transit colours"
    assert all(tuple(np.round(b.color, 2)) == tuple(np.round(props.COL_BUS, 2)) for b in school)


def test_city_character_billboards_lights_walls_trees_and_litter():
    from world.obstacle import MAT_BILLBOARD, MAT_LIGHT, MAT_WALL, MAT_WINDOW_WALL, MAT_BARK, MAT_LEAVES, MAT_ASPHALT, MAT_PAVEMENT, MAT_TRASH
    from world.building import WALL_PLASTER, WALL_BRICK, WALL_PANELS, DECOR_GRAFFITI, DECOR_MURAL, DECOR_GRIME
    obstacles, verts, idx, city = _city()
    boards = [o for o in obstacles if o.name == "billboard"]
    assert len(boards) >= 4 and len(set(o.variant for o in boards)) >= 4, "different advertisements"
    assert all(o.material == MAT_BILLBOARD for o in boards)
    lamps = [o for o in obstacles if o.name == "lamp"]
    assert lamps
    for lamp in lamps[:10]:
        codes = _codes(lamp.vertices)
        assert MAT_LIGHT in set((codes // 10).tolist()), "every lamp post carries a light"
        light = lamp.vertices[codes // 10 == MAT_LIGHT]
        assert light[:, 1].min() > 4.5, "up on the arm"
        assert np.hypot(float(light[:, 0].mean() - lamp.bounds.center[0]), float(light[:, 2].mean() - lamp.bounds.center[2])) > 1.2, "reaching out over the road"
    cores = [o for o in obstacles if o.name.endswith("-core")]
    styles = set(o.variant % 3 for o in cores)
    decors = set(o.variant // 3 for o in cores)
    assert styles >= {WALL_PLASTER, WALL_BRICK, WALL_PANELS}, "plaster, brick and panel buildings"
    assert decors >= {DECOR_GRAFFITI, DECOR_GRIME} and (DECOR_MURAL in decors), "graffiti, water damage and murals around the city"
    assert all(o.material == MAT_WALL for o in cores)
    assert any(MAT_WINDOW_WALL in o.face_materials.values() for o in cores), "glass behind the facade windows"
    assert len(set(tuple(np.round(o.color, 2)) for o in cores)) >= 6, "many wall colours"
    trunks = [o for o in obstacles if o.name.endswith("-trunk")]
    assert trunks and all(o.material == MAT_BARK for o in trunks)
    canopies = [o for o in obstacles if o.name.endswith("-canopy")]
    assert canopies and all(o.material == MAT_LEAVES for o in canopies)
    assert all(len(o.vertices) > 24 * 5 for o in canopies), "leaf clumps around the canopy box"
    assert all(len(o.vertices) > 24 * 6 for o in trunks), "root flare, twigs and a rounded trunk"
    litter = [o for o in obstacles if o.name == "trash"]
    assert len(litter) >= 5 and all(MAT_TRASH in set((_codes(o.vertices) // 10).tolist()) for o in litter)
    mats = set((_codes(verts) // 10).tolist())
    assert {MAT_ASPHALT, MAT_PAVEMENT} <= mats, "worn asphalt streets and paved sidewalks"


def test_boots_have_rounded_toes_and_heels():
    rig = FirstPersonArmRig()
    v = rig.vertices
    foot = v[(v[:, 8].astype(int) == BoneID.R_FOOT) & (v[:, 6] == MAT_BOOT)]
    y = foot[:, 1]                     # along the toe direction
    tip = float(y.max())
    assert 0.18 <= tip <= 0.24, "boot length"
    mid = foot[np.abs(y - 0.08) < 0.07]
    mid_w = float(np.abs(mid[:, 0]).max())
    assert mid_w >= 0.035, "a boot's width"
    heel = float(y.min())
    tip_w = float(np.abs(foot[y > tip - 0.006][:, 0]).max())
    heel_w = float(np.abs(foot[y < heel + 0.006][:, 0]).max())
    assert tip_w < 0.01, "the toe closes to a rounded tip, not a square end"
    assert heel_w < 0.01, "rounded heel"
    near_tip_w = float(np.abs(foot[(y > tip - 0.04) & (y <= tip - 0.006)][:, 0]).max())
    assert tip_w < near_tip_w < mid_w, "the width tapers smoothly into the tip"
    # the toe cap is a dome: the tip sits below the boot's upper surface
    assert float(foot[y > tip - 0.006][:, 2].max()) < float(mid[:, 2].max())
