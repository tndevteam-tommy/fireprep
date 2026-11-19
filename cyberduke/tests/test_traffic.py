"""The traffic on the main street (``entities.traffic``, ``world.vehicles``,
``render.traffic``): cars, taxis and vans come out of the tunnel at one end, drive down
their lane keeping their distance - slowing behind a slower one - and go off into the
tunnel at the other end, their wheels turning with the road."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cyberduke.core.clock import WorldClock
from cyberduke.entities.traffic import Car, Traffic
from cyberduke.levels import build_night_market, main_street
from cyberduke.world.level import LaneData
from cyberduke.world.vehicles import MODELS, PAINTS, Paint, wheel_turn

DT = 1 / 30
LEVEL = build_night_market()


def _clock(hour: float) -> WorldClock:
    return WorldClock(hour * 60.0, 0.0)


# ----------------------------------------------------------------------------- the road
def test_two_lanes_one_each_way_between_the_tunnels():
    west, east = sorted(LEVEL.lanes, key=lambda ln: ln.heading)
    assert west.heading < 0 < east.heading
    assert west.z < east.z, "keeping to the right: westbound on the near side"
    for lane in LEVEL.lanes:
        assert min(lane.x_from, lane.x_to) < -main_street.TUNNEL_X - 20.0
        assert max(lane.x_from, lane.x_to) > main_street.TUNNEL_X + 20.0, "tunnel to tunnel"
        assert lane.y == main_street.ROAD_Y
    widest = max(m.width for m in MODELS.values())
    assert (
        west.z - widest / 2.0 > main_street.PARKING_Z
        and west.z + widest / 2.0 < main_street.MEDIAN[0]
    )
    assert (
        main_street.MEDIAN[1] < east.z - widest / 2.0
        and east.z + widest / 2.0 < main_street.ROAD_Z1
    )
    # the parked coupe is clear of the westbound lane's traffic
    assert main_street.CAR_AT[1] + 1.05 < west.z - widest / 2.0


def test_the_road_runs_into_tunnels_at_both_ends():
    labels = [p.label for p in LEVEL.props]
    assert labels.count("Tunnel portal wall") == 12, "six pieces round two mouths, each end"
    assert "Road tunnel roof" in labels and "Monorail tunnel roof" in labels
    assert labels.count("Monorail pillar") == len(main_street.PILLAR_XS)


# ----------------------------------------------------------------------------- the traffic
def test_the_street_has_traffic_already_and_it_moves():
    from cyberduke.simulation import InputState, Simulation

    sim = Simulation(LEVEL)
    assert sim.traffic is not None and sim.traffic.cars, "cars about as soon as you get there"
    before = {c.car_id: c.x for c in sim.traffic.cars}
    for _ in range(30):
        sim.update(DT, InputState())
    moved = [c for c in sim.traffic.cars if c.car_id in before and c.x != before[c.car_id]]
    assert moved, "they drive"
    for car in moved:
        assert (car.x - before[car.car_id]) * car.heading > 0, "forwards, the way of the lane"


def test_vehicles_only_come_and_go_in_the_tunnels():
    traffic = Traffic(LEVEL.lanes, seed=5, clock=_clock(18.0))
    seen: dict[int, float] = {c.car_id: c.x for c in traffic.cars}
    last: dict[int, float] = dict(seen)
    for _ in range(int(240 / 0.1)):
        traffic.update(0.1)
        now = {c.car_id: c.x for c in traffic.cars}
        for car_id, x in now.items():
            if car_id not in seen:
                seen[car_id] = x
                assert abs(x) > main_street.TUNNEL_X + 20.0, "out of the distance, in a tunnel"
        for car_id in set(last) - set(now):
            assert abs(last[car_id]) > main_street.TUNNEL_X + 20.0, "off into the tunnel"
        last = now
    assert len(seen) > 30, "four minutes of the evening rush"
    models = {c.model.name for c in traffic.cars}
    assert models, "still cars about"


def test_nobody_runs_into_the_car_in_front():
    traffic = Traffic(LEVEL.lanes, seed=11, clock=_clock(18.5))
    for _ in range(int(300 / 0.05)):
        traffic.update(0.05)
        for lane in traffic.lanes:
            cars = traffic.in_lane(lane)
            for ahead, car in zip(cars, cars[1:], strict=False):
                assert traffic.gap(car, ahead) > 0.4


def test_a_fast_car_slows_behind_a_slow_one_and_follows_it():
    lane = LaneData("test", 0.0, 0.0, 1000.0)
    traffic = Traffic([lane], seed=1)
    traffic.cars = []
    traffic.due = {"test": 1e9}  # (nobody else)
    van = Car(1, MODELS["van"], Paint((1.0, 1.0, 1.0)), lane, cruise=7.0, speed=7.0, x=60.0)
    coupe = Car(2, MODELS["coupe"], Paint((1.0, 0.0, 0.0)), lane, cruise=16.0, speed=16.0, x=10.0)
    traffic.cars = [van, coupe]
    braked = False
    for _ in range(int(30 / DT)):
        traffic.update(DT)
        braked |= coupe.braking
    assert braked, "the brake lights came on"
    assert coupe.speed == pytest.approx(van.speed, abs=0.3), "it follows at the van's speed"
    gap = traffic.gap(coupe, van)
    assert traffic.MIN_GAP < gap < traffic.MIN_GAP + van.speed * traffic.HEADWAY * 1.6
    # the van turns off (gone): the coupe gets back up to its own speed
    traffic.cars = [coupe]
    for _ in range(int(20 / DT)):
        traffic.update(DT)
    assert coupe.speed == pytest.approx(coupe.cruise, abs=0.5)


def test_a_car_stopped_in_front_stops_the_one_behind():
    lane = LaneData("test", 0.0, 0.0, 1000.0)
    traffic = Traffic([lane], seed=1)
    traffic.due = {"test": 1e9}
    stopped = Car(1, MODELS["taxi"], Paint((1.0, 1.0, 0.0)), lane, cruise=0.0, speed=0.0, x=80.0)
    car = Car(2, MODELS["coupe"], Paint((0.0, 0.0, 1.0)), lane, cruise=14.0, speed=14.0, x=20.0)
    traffic.cars = [stopped, car]
    for _ in range(int(20 / DT)):
        traffic.update(DT)
    assert car.speed < 0.2 and 0.4 < traffic.gap(car, stopped) < traffic.MIN_GAP + 1.0


def test_the_wheels_turn_with_the_distance_driven():
    lane = LaneData("test", 0.0, 0.0, 1000.0)
    traffic = Traffic([lane], seed=1)
    traffic.due = {"test": 1e9}
    car = Car(1, MODELS["coupe"], Paint((1.0, 1.0, 1.0)), lane, cruise=10.0, speed=10.0, x=0.0)
    traffic.cars = [car]
    traffic.update(0.1)
    r = MODELS["coupe"].wheel_radius
    assert car.spin == pytest.approx(math.degrees(car.x / r) % 360.0, abs=1e-6)
    assert wheel_turn(2 * math.pi * r, r) == pytest.approx(360.0)


def test_busier_in_the_evening_rush_than_in_the_small_hours():
    def count(hour: float) -> int:
        traffic = Traffic(LEVEL.lanes, seed=3, clock=_clock(hour))
        ids = {c.car_id for c in traffic.cars}
        for _ in range(int(300 / 0.25)):
            traffic.update(0.25)
            ids |= {c.car_id for c in traffic.cars}
        return len(ids)

    assert count(18.0) > 1.6 * count(3.0)


def test_the_traffic_is_the_same_every_time():
    a = Traffic(LEVEL.lanes, seed=9, clock=_clock(20.0))
    b = Traffic(LEVEL.lanes, seed=9, clock=_clock(20.0))
    for _ in range(100):
        a.update(0.1)
        b.update(0.1)
    assert [(c.model.name, c.paint, round(c.x, 6)) for c in a.cars] == [
        (c.model.name, c.paint, round(c.x, 6)) for c in b.cars
    ]


# ----------------------------------------------------------------------------- the models
@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_is_built_without_z_fighting(name):
    from cyberduke.render.audit import GeometryRecorder, find_zfights
    from cyberduke.render.props import draw_prop

    model = MODELS[name]
    for body in PAINTS[name][:2]:
        ctx = GeometryRecorder()
        for p in model.parts(Paint(body)):
            ctx.label = p.label
            draw_prop(ctx, p, {})
            ctx.alpha(1.0)
        assert find_zfights(ctx.faces, ctx.solids) == []


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_stands_on_its_wheels_within_its_size(name):
    model = MODELS[name]
    boxes = [p.footprint() for p in model.parts(Paint((0.5, 0.5, 0.5)))]
    lo_x, hi_x = min(b.min_x for b in boxes), max(b.max_x for b in boxes)
    lo_z, hi_z = min(b.min_z for b in boxes), max(b.max_z for b in boxes)
    lo_y, hi_y = min(b.min_y for b in boxes), max(b.max_y for b in boxes)
    assert lo_y == pytest.approx(0.0, abs=1e-6), "the tyres on the road"
    assert hi_x - lo_x <= model.length + 0.05 and hi_z - lo_z <= model.width + 0.05
    assert hi_y <= model.height + 0.05
    for x, z, _ in model.wheels():
        assert lo_x < x < hi_x and abs(z) < hi_z


# ----------------------------------------------------------------------------- the drawing
class _BatchRecorder:
    """Enough of a context to draw the traffic with batches: every batch's vertices in
    world space."""

    batches = True
    texture_tiles: dict = {}

    def __init__(self):
        from tests.test_visual_regressions import MatrixRecorder

        self.inner = MatrixRecorder()
        self.drawn = []  # (world vertices (n, 3))

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def draw_batch(self, vertices, indices, ranges):
        m = self.inner.m
        pts = np.asarray(vertices[:, :3], dtype=float) @ m[:3, :3].T + m[:3, 3]
        self.drawn.append(pts)


def test_vehicles_are_drawn_where_they_are_and_their_wheels_roll():
    from cyberduke.render.traffic import TrafficRenderer

    lane = LEVEL.lanes[0]
    car = Car(1, MODELS["coupe"], Paint((0.2, 0.3, 0.9)), lane, 12.0, 12.0, x=5.0)
    renderer = TrafficRenderer()
    ctx = _BatchRecorder()
    renderer.draw(ctx, [car], {})
    body, *wheels = ctx.drawn
    assert len(wheels) == 4
    assert body[:, 0].min() > 5.0 - 2.6 and body[:, 0].max() < 5.0 + 2.6
    assert abs(body[:, 2].mean() - lane.z) < 0.2 and body[:, 1].min() >= lane.y - 1e-6
    car.spin = 90.0
    ctx2 = _BatchRecorder()
    renderer.draw(ctx2, [car], {})
    assert np.allclose(ctx2.drawn[0], body), "the body the same"
    moved = np.abs(ctx2.drawn[1] - wheels[0]).max()
    assert moved > 0.2, "the wheel has turned a quarter"
    hub = wheels[0].mean(axis=0)
    assert np.allclose(ctx2.drawn[1].mean(axis=0), hub, atol=0.02), "about its hub"
    far = Car(2, MODELS["van"], Paint((1.0, 1.0, 1.0)), lane, 9.0, 9.0, x=90.0)
    ctx3 = _BatchRecorder()
    renderer.draw(ctx3, [far], {}, eye=(0.0, 1.7, 44.0), view_distance=40.0)
    assert ctx3.drawn == [], "out of sight down the tunnel: not drawn"
