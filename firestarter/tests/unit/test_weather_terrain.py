"""DESIGN Section 7 (seasons / weather / time) and the shared terrain height field."""

import math

import numpy as np
import pytest

from firestarter.simulation.weather import WeatherSystem, Season, CLIMATE, SEASON_ORDER
from firestarter.simulation import terrain


# ------------------------------------------------------------------------ solar geometry
def test_summer_solstice_sun_path_at_45_north():
    w = WeatherSystem(start_hour=12.0, season=Season.SUMMER)
    # noon elevation = 90 - lat + declination = 68.4 deg on June 21 at 45 N
    assert w.sun_elevation_deg() == pytest.approx(68.4, abs=1.0)
    rise, sset = w.sunrise_sunset_hours()
    assert 15.2 < sset - rise < 15.8         # ~15.5 h of daylight
    w.set_time(0.0)
    assert w.is_night() and w.sun_elevation_deg() < -15


def test_winter_solstice_is_short_and_low():
    w = WeatherSystem(start_hour=12.0, season=Season.WINTER)
    assert w.sun_elevation_deg() == pytest.approx(21.6, abs=1.0)
    rise, sset = w.sunrise_sunset_hours()
    assert 8.4 < sset - rise < 9.2          # ~8.6 h geometric day length


def test_sun_rises_in_the_east_and_sets_in_the_west():
    w = WeatherSystem(start_hour=7.0, season=Season.AUTUMN)
    d_morning = w.get_sun_direction()
    w.set_time(17.0)
    d_evening = w.get_sun_direction()
    assert d_morning[0] > 0.3 and d_evening[0] < -0.3      # +X is east in world space
    assert np.linalg.norm(d_morning) == pytest.approx(1.0, abs=1e-6)


def test_daylight_factor_is_smooth_through_twilight():
    w = WeatherSystem(season=Season.SUMMER)
    vals = []
    for h in np.arange(19.0, 22.5, 0.25):
        w.set_time(h)
        vals.append(w.daylight_factor())
    assert vals[0] == pytest.approx(1.0) and vals[-1] == pytest.approx(0.0)
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
    assert max(abs(b - a) for a, b in zip(vals, vals[1:])) < 0.5


# ------------------------------------------------------------------------ climate
def test_diurnal_temperature_cycle_peaks_in_afternoon():
    w = WeatherSystem(season=Season.AUTUMN)
    w.forced_precip = 0.0
    temps = {}
    for h in (5.0, 15.0):
        w.set_time(h)
        temps[h] = w.ambient_temp_c
    assert temps[15.0] > temps[5.0] + 5.0


@pytest.mark.parametrize("season", list(Season))
def test_seasonal_temperature_matches_climatology(season):
    w = WeatherSystem(seed=3, season=season)
    w.forced_precip = 0.0
    temps = []
    for h in range(24):
        w.set_time(float(h))
        temps.append(w.ambient_temp_c)
    assert np.mean(temps) == pytest.approx(CLIMATE[season]["base_temp"], abs=3.0)


def test_long_run_wet_fraction_matches_season():
    for season in (Season.SUMMER, Season.AUTUMN):
        w = WeatherSystem(seed=11, season=season)
        wet = 0
        n = 0
        for _ in range(24 * 80):
            w._step_weather(1.0, 3600.0)
            w._update_parameters(3600.0)
            w.current_season = season
            n += 1
            wet += (w.rain_intensity + w.snow_intensity) > 0.05
        assert wet / n == pytest.approx(CLIMATE[season]["wet_fraction"], abs=0.12)


def test_winter_precipitation_falls_as_snow_and_accumulates():
    w = WeatherSystem(season=Season.WINTER, start_hour=3.0)
    w.force_precipitation(0.8)
    depth0 = w.snow_accumulation_m
    w.advance_time(6.0)
    assert w.snow_intensity > 0.3 and w.rain_intensity == 0.0
    assert w.snow_accumulation_m > depth0
    assert w.describe() in ("Snowfall", "Blizzard")


def test_rain_wets_the_ground_and_sun_dries_it():
    w = WeatherSystem(season=Season.SUMMER, start_hour=9.0)
    w.force_precipitation(1.0)
    w.advance_time(3.0)
    wet = w.ground_wetness
    assert wet > 0.8 and w.relative_humidity > 0.8
    w.force_precipitation(0.0)
    w.advance_time(8.0)
    assert w.ground_wetness < wet - 0.3


def test_season_rolls_over_after_90_days():
    w = WeatherSystem(season=Season.SUMMER)
    stats = w.advance_time(24 * 91)
    assert w.current_season == Season.AUTUMN and stats["season_changes"] == 1
    assert SEASON_ORDER.index(Season.AUTUMN) == 1


def test_wind_gusts_vary_but_stay_bounded():
    w = WeatherSystem(season=Season.AUTUMN)
    speeds = []
    for _ in range(600):
        w.update(0.5)
        speeds.append(w.wind_speed_m_s)
    assert 0.2 <= min(speeds) and max(speeds) <= 12.0
    assert np.std(speeds) > 0.2


def test_clear_night_sky_is_radiatively_cold():
    w = WeatherSystem(season=Season.SUMMER, start_hour=1.0)
    w.forced_cloud = 0.0
    w.cloud_cover = 0.0
    assert w.sky_temperature_c() < w.ambient_temp_c - 15


def test_forecast_covers_next_12_hours():
    fc = WeatherSystem().forecast(12)
    assert len(fc) == 5 and all({"hour", "temp", "precip", "snow"} <= set(f) for f in fc)


# ------------------------------------------------------------------------ terrain
def test_camp_clearing_is_flat_and_pit_is_a_shallow_bowl():
    assert terrain.height(0.0, 0.0) == pytest.approx(-0.06, abs=0.005)
    assert -0.01 < terrain.height(1.3, 0.0) - terrain.height(3.0, 0.0) < 0.06
    for x, z in ((4, 0), (0, 6), (-5, -5), (7, 2)):
        assert terrain.slope_degrees(x, z) < 4.0


def test_terrain_is_continuous_and_vectorised():
    X, Z, H = terrain.height_grid(60.0, 121)
    assert H.shape == (121, 121) and np.isfinite(H).all()
    assert np.abs(np.diff(H, axis=0)).max() < 1.5       # < 1.5 m rise per metre anywhere walkable
    assert float(terrain.height(np.array([3.0]), np.array([4.0]))[0]) == pytest.approx(terrain.height(3.0, 4.0))


def test_creek_runs_north_and_ridge_south():
    x = 0.0
    zc = float(terrain.creek_centerline_z(x))
    assert -50 < zc < -30
    assert terrain.is_in_water(x, zc)
    assert not terrain.is_in_water(0.0, 0.0)
    assert terrain.height(0.0, 52.0) > terrain.height(0.0, zc) + 8.0
    assert terrain.ground_moisture(0.0, zc) > terrain.ground_moisture(0.0, 52.0) + 0.3


def test_far_mountains_frame_the_horizon():
    assert terrain.height(0.0, 220.0) > 25.0 and terrain.height(-220.0, 0.0) > 20.0
