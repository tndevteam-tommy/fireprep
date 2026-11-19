"""The city's weather (``systems.weather``, drawn by ``render.weather``): a drizzle that
fades in slowly, spells of steady rain, downpours and thunderstorms one after another,
puddles that fill while it rains and dry out after, lightning only in a storm, fog that
comes and goes over hours and limits how far anything can be seen, and cars throwing up
spray driving through the puddles on the road."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cyberduke.core.clock import WorldClock
from cyberduke.levels import build_night_market
from cyberduke.levels.loader import level_from_dict, level_to_dict
from cyberduke.render.scene import MIN_SIGHT, weather_light
from cyberduke.render.weather import WeatherRenderer
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems.weather import (
    FOG_RATE,
    SPELLS,
    STORM_RAIN,
    Weather,
    open_sky_at,
    puddle_sites,
    under_shelter,
    visibility,
)

LEVEL = build_night_market()
NIGHT = 22 * 60.0


def _run(w: Weather, seconds: float, clock: WorldClock, dt: float = 1.0) -> None:
    for _ in range(int(seconds / dt)):
        clock.advance(dt)
        w.update(dt, clock)


def test_the_weather_is_the_same_every_time_for_the_same_seed():
    a, b = Weather(seed=5), Weather(seed=5)
    ca, cb = WorldClock(NIGHT, 0.5), WorldClock(NIGHT, 0.5)
    _run(a, 3 * 3600, ca, 2.0)
    _run(b, 3 * 3600, cb, 2.0)
    assert a.to_dict() == b.to_dict() and a.strikes == b.strikes
    c = Weather(seed=6)
    _run(c, 3 * 3600, WorldClock(NIGHT, 0.5), 2.0)
    assert c.to_dict() != a.to_dict(), "another seed, other weather"


def test_a_drizzle_fades_in_slowly():
    w = Weather(seed=1, rain=0.0)
    w.set_spell("drizzle", minutes=200.0)
    clock = WorldClock(NIGHT, 0.5)
    _run(w, 2.0, clock, 0.5)
    assert 0.0 < w.rain < 0.02, "it starts spitting, barely"
    _run(w, 18.0, clock, 0.5)  # (ten game minutes)
    assert w.rain < SPELLS["drizzle"].rain * 0.75, "still coming in a quarter of a minute on"
    _run(w, 60.0, clock, 0.5)
    assert w.rain == pytest.approx(SPELLS["drizzle"].rain), "a steady drizzle in the end"


def test_the_rain_goes_through_spells_and_never_jumps():
    w = Weather(seed=3)
    clock = WorldClock(NIGHT, 0.5)
    seen, last = set(), w.rain
    biggest = 0.0
    for _ in range(int(24 * 3600 / 5.0)):  # (a day of real time: 12 game days)
        clock.advance(5.0)
        w.update(5.0, clock)
        seen.add(w.spell)
        biggest = max(biggest, abs(w.rain - last))
        last = w.rain
    assert seen == set(SPELLS), "dry spells, drizzle, rain, downpours and storms"
    fastest = max(s.arrive for s in SPELLS.values())
    assert biggest <= fastest * 2.5 + 1e-9, "it eases from one to the next"


def test_the_puddles_fill_while_it_rains_and_dry_out_after():
    w = Weather(seed=2, wet=0.0)
    w.set_spell("rain", minutes=60.0)
    clock = WorldClock(NIGHT, 0.5)
    _run(w, 60.0, clock)  # (half an hour of rain)
    filling = w.wet
    assert filling > 0.1
    _run(w, 60.0, clock)
    assert w.wet > filling, "still filling"
    wet = w.wet
    w.set_spell("dry", minutes=600.0)
    _run(w, 240.0, clock)
    assert w.wet < wet, "drying out when it stops"
    # ... faster by day
    night = Weather(seed=2, wet=0.8, rain=0.0)
    day = Weather(seed=2, wet=0.8, rain=0.0)
    night.set_spell("dry", 600.0)
    day.set_spell("dry", 600.0)
    _run(night, 120.0, WorldClock(NIGHT, 0.5))
    _run(day, 120.0, WorldClock(12 * 60.0, 0.5))
    assert day.wet < night.wet


def test_puddles_gather_on_open_ground_clear_of_everything_standing_there():
    sites = puddle_sites(LEVEL)
    assert len(sites) > 40 and sites == puddle_sites(LEVEL), "plenty, the same every time"
    colliders = [c for s in LEVEL.sectors for c in s.colliders]
    for p in sites:
        region = open_sky_at(p.x, p.z, LEVEL.open_sky)
        assert region is not None and region.puddles, "out under the sky, on the ground"
        assert p.y == pytest.approx(region.ground + 0.004)
        reach = max(p.rx, p.rz)
        assert region.min_x + reach <= p.x <= region.max_x - reach, "clear of the walls"
        assert region.min_z + reach <= p.z <= region.max_z - reach
        assert not under_shelter(p.x, p.z, LEVEL.shelters)
        for c in colliders:
            if c.min_y <= region.ground + 0.3 and c.max_y > region.ground:
                inside = c.min_x - reach < p.x < c.max_x + reach and (
                    c.min_z - reach < p.z < c.max_z + reach
                )
                assert not inside, "not under a bin, a stall, a doorstep or a car"
    assert any(p.road for p in sites) and any(not p.road for p in sites)
    # the deepest dips fill first; a puddle grows with the wet
    p = min(sites, key=lambda s: s.fills)
    q = max(sites, key=lambda s: s.fills)
    wet = (p.fills + q.fills) / 2.0
    assert p.size(wet) > 0.0 and q.size(wet) == 0.0
    assert p.size(p.fills + 0.05) < p.size(p.fills + 0.2) <= 1.0


def test_lightning_only_in_a_thunderstorm():
    clock = WorldClock(NIGHT, 0.5)
    for spell in ("drizzle", "rain", "downpour"):
        w = Weather(seed=4, rain=SPELLS[spell].rain)
        w.set_spell(spell, minutes=1000.0)
        flashes = 0.0
        for _ in range(1800):
            clock.advance(1 / 15)
            w.update(1 / 15, clock)
            flashes = max(flashes, w.flash)
        assert w.strikes == 0 and flashes == 0.0, spell
    storm = Weather(seed=4, rain=1.0)
    storm.set_spell("storm", minutes=1000.0)
    brightest = 0.0
    for _ in range(1800):  # (two minutes)
        clock.advance(1 / 15)
        storm.update(1 / 15, clock)
        brightest = max(brightest, storm.flash)
    assert storm.storm and storm.strikes >= 3 and brightest > 0.5
    # a flash is a flicker: gone within a second
    storm.strike_age = 1.5
    assert storm.flash == 0.0
    # ... and not while a storm spell's rain is still coming in
    early = Weather(seed=4, rain=STORM_RAIN - 0.2)
    early.set_spell("storm", minutes=1000.0)
    assert not early.storm


def test_fog_comes_in_over_hours_and_limits_how_far_can_be_seen():
    clear = Weather(fog=0.0, rain=0.0)
    assert clear.visibility == math.inf
    thick = Weather(fog=0.85, rain=0.0)
    assert thick.visibility < 20.0
    assert Weather(fog=0.3).visibility > thick.visibility
    assert Weather(fog=0.0, rain=1.0).visibility < 100.0, "a downpour hides the far end"
    w = Weather(seed=9, fog=0.0, fog_spell="fog", fog_left=1000.0)
    clock = WorldClock(NIGHT, 0.5)
    _run(w, 120.0, clock)  # (an hour)
    assert 0.2 < w.fog < 0.85 * 0.5, "coming in slowly: not there yet after an hour"
    assert w.fog == pytest.approx(FOG_RATE * 60.0, rel=0.05)
    # the scene: fog pulls the far plane in, thinner off the street, never to nothing
    light = weather_light((0.25, 0.3, 0.4), 0.0, True, thick, 1.0, 75.0)
    assert light.density > 0.0 and light.far == pytest.approx(visibility(light.density), abs=0.1)
    assert MIN_SIGHT <= light.far < 20.0
    inside = weather_light((0.25, 0.3, 0.4), 0.0, True, thick, 0.3, 75.0)
    assert inside.density < light.density and inside.far > light.far
    clear_light = weather_light((0.25, 0.3, 0.4), 0.0, True, clear, 1.0, 75.0)
    assert clear_light.far == 75.0 and clear_light.density == 0.0
    indoors = weather_light((0.25, 0.3, 0.4), 0.0, False, thick, 1.0, 75.0)
    assert indoors.density == 0.0, "no weather in the facility"


def test_a_lightning_flash_lights_up_the_street_and_the_sky():
    storm = Weather(rain=1.0, fog=0.0)
    storm.set_spell("storm", 100.0)
    dark = weather_light((0.25, 0.3, 0.4), 0.0, True, storm)
    storm.strokes, storm.strike_age = ((0.0, 1.0),), 0.0
    lit = weather_light((0.25, 0.3, 0.4), 0.0, True, storm)
    assert all(b > a + 0.3 for a, b in zip(dark.ambient, lit.ambient, strict=True))
    assert all(b > a + 0.3 for a, b in zip(dark.sky, lit.sky, strict=True))


def test_cars_throw_up_spray_driving_through_puddles_on_the_road():
    sim = Simulation(LEVEL)
    sim.clock.rate = 0.0
    road = [p for p in sim.puddles if p.road]
    assert road
    lane = {ln.lane_id: ln for ln in LEVEL.lanes}

    def drive(wet: float, speed: float) -> int:
        sim.weather.wet, sim.weather.drops = wet, []
        thrown = 0
        traffic = sim.traffic
        traffic.cars = []
        # a car along the lane that runs through a road puddle
        p = min(road, key=lambda s: min(abs(s.z - ln.z) for ln in LEVEL.lanes))
        ln = min(lane.values(), key=lambda ln: abs(ln.z - p.z))
        from cyberduke.entities.traffic import Car
        from cyberduke.world.vehicles import random_vehicle

        model, paint = random_vehicle(__import__("random").Random(3))
        car = Car(99, model, paint, ln, speed, speed, x=p.x - ln.heading * 6.0)
        for _ in range(40):
            car.x += ln.heading * car.speed / 30.0
            sim.weather.spray([car], sim.puddles, 1 / 30)
            thrown = max(thrown, len(sim.weather.drops))
        return thrown

    assert drive(1.0, 13.0) > 10, "a splash off the wheels"
    assert drive(0.0, 13.0) == 0, "a dry road: nothing"
    assert drive(1.0, 1.0) == 0, "crawling through it: nothing"
    # the drops fly up and fall back down (gone when they land)
    drive(1.0, 13.0)
    for _ in range(60):
        sim.weather.spray([], sim.puddles, 1 / 30)
    assert sim.weather.drops == []


def test_the_rain_falls_on_open_ground_never_under_a_roof():
    w = Weather(rain=1.0)
    w.set_spell("downpour", 100.0)
    r = WeatherRenderer()
    # under the food stall's awning, in the alley
    streaks, crowns = r.rain_segments(w, LEVEL, (-3.0, 1.6, 14.0), 3.0, 1600)
    assert len(streaks) > 300 and len(crowns) > 0
    for bottom, _top in streaks:
        assert open_sky_at(bottom[0], bottom[2], LEVEL.open_sky) is not None
        assert not under_shelter(bottom[0], bottom[2], LEVEL.shelters, bottom[1])
    # standing in a flat, the rain only shows outside (out of the window, the door)
    inside = (-9.5, 1.6, 38.0)
    assert open_sky_at(inside[0], inside[2], LEVEL.open_sky) is None
    streaks, _ = r.rain_segments(w, LEVEL, inside, 3.0, 1600)
    assert all(open_sky_at(b[0], b[2], LEVEL.open_sky) for b, _ in streaks)
    # no rain: nothing; the streaks stay put in the world as the eye moves
    assert len(r.rain_segments(Weather(rain=0.0), LEVEL, inside, 3.0, 1600)[0]) == 0
    a, _ = r.rain_segments(w, LEVEL, (0.0, 1.6, 20.0), 3.0, 1600)
    b, _ = r.rain_segments(w, LEVEL, (0.3, 1.6, 20.0), 3.0, 1600)
    common = {tuple(np.round(s[0], 4)) for s in a} & {tuple(np.round(s[0], 4)) for s in b}
    assert len(common) > 0.8 * min(len(a), len(b))


def test_the_weather_goes_on_from_level_to_level_and_is_saved():
    from cyberduke.systems.savegame import restore, snapshot

    sim = Simulation(LEVEL)
    sim.weather.set_spell("storm", 30.0)
    sim.weather.rain, sim.weather.wet, sim.weather.fog = 0.9, 0.7, 0.4
    for _ in range(10):
        sim.update(1 / 30, InputState())
    data = snapshot(sim)
    other = Simulation(LEVEL)
    restore(other, data)
    assert other.weather.to_dict() == sim.weather.to_dict()
    # the street's open sky and shelters survive being written out and read back
    again = level_from_dict(level_to_dict(LEVEL))
    assert again.open_sky == LEVEL.open_sky and again.shelters == LEVEL.shelters


def test_the_puddles_are_drawn_as_one_mesh_each_step_of_wet():
    from tests.test_render_backends import RecordingContext

    from cyberduke.render.weather import PUDDLE_SLICES

    sites = puddle_sites(LEVEL)
    r = WeatherRenderer()
    ctx = RecordingContext()
    assert r._puddle_meshes(ctx, sites, 0.0) is None, "dry: nothing to draw"
    rim_key, rim, sheens, water = r._puddle_meshes(ctx, sites, 1.0)
    shown = [p for p in sites if p.size(1.0) > 0.0]
    assert len(rim) == len(shown) * PUDDLE_SLICES * 9
    assert rim[:, 3].min() >= 0.0 and rim[:, 3].max() <= 1.0
    assert sum(len(t) for _, _, t in sheens) == len(shown) * PUDDLE_SLICES * 9
    vertices, indices, ranges = water
    assert len(vertices) == len(shown) * (PUDDLE_SLICES + 1)
    levels = np.array(sorted({p.y + 0.0005 for p in shown}))
    assert np.abs(vertices[:, 1][:, None] - levels[None, :]).min(axis=1).max() < 1e-5
    assert indices.max() < len(vertices) and ranges[0][2] == len(indices)
    # the same wet again: the same meshes (built once); the next step: new ones
    assert r._puddle_meshes(ctx, sites, 1.0)[0] == rim_key
    assert r._puddle_meshes(ctx, sites, 0.9)[0] != rim_key
