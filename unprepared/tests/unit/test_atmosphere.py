"""Sun/moon motion and the time-of-day light/fog model."""
import math

import pytest

from unprepared.config import LightingSettings
from unprepared.rendering.atmosphere import SUNRISE, SUNSET, compute_atmosphere, moon_direction, sun_direction

S = LightingSettings()


def test_sun_rises_east_peaks_at_midday_sets_west():
    east = sun_direction(SUNRISE)
    assert east[0] > 0.9 and abs(east[1]) < 1e-6
    midday = sun_direction((SUNRISE + SUNSET) / 2)
    assert midday[1] > 0.95 and abs(midday[0]) < 1e-6
    west = sun_direction(SUNSET)
    assert west[0] < -0.9 and abs(west[1]) < 1e-6
    assert sun_direction((SUNRISE + SUNSET) / 2 + 12.0)[1] < -0.9      # the middle of the night: straight down
    assert (SUNRISE, SUNSET) == (6.0, 18.0)


def test_moon_is_opposite_the_sun():
    s, m = sun_direction(3.0), moon_direction(3.0)
    assert m[0] == pytest.approx(-s[0]) and m[1] == pytest.approx(-s[1])


def test_motion_is_smooth():
    prev = sun_direction(6.0)
    for i in range(1, 500):
        cur = sun_direction(6.0 + i * 0.024)
        assert math.dist(prev, cur) < 0.01
        prev = cur


def test_noon_is_bright_midnight_is_dark_and_foggy():
    noon, midnight = compute_atmosphere(12.0, S), compute_atmosphere(0.0, S)
    assert noon.daylight == 1.0 and midnight.daylight == 0.0
    assert sum(noon.sun_color) > 2.5 and sum(midnight.sun_color) == 0.0
    assert sum(midnight.moon_color) > 0 and sum(noon.moon_color) == 0.0
    assert sum(noon.ambient) > sum(midnight.ambient) * 4
    assert noon.sky_color == S.sky_day and midnight.sky_color == S.sky_night
    assert midnight.fog_density > noon.fog_density * 5


def test_dusk_is_orange_and_in_between():
    dusk = compute_atmosphere(18.1, S)
    assert dusk.sky_color[0] > dusk.sky_color[2]                       # red over blue
    assert 0.0 < dusk.daylight < 1.0
    assert dusk.sun_color[0] > dusk.sun_color[2]                       # warm low sun


def test_the_evening_fades_as_the_torches_come_out():
    from unprepared.core.daynight import DayNightCycle
    late_afternoon = compute_atmosphere(17.0, S)
    assert late_afternoon.daylight == 1.0 and not DayNightCycle.is_torch_time(17.0)
    dusk = compute_atmosphere(18.0, S)
    assert 0.0 < dusk.daylight < 1.0 and DayNightCycle.is_torch_time(18.0)      # getting dim: torches lit
    assert compute_atmosphere(20.0, S).daylight < 0.1                            # dark by the night rules' hour


def test_values_change_gradually_through_the_day():
    prev = compute_atmosphere(0.0, S)
    for i in range(1, 24 * 20):
        cur = compute_atmosphere(i / 20.0, S)
        assert math.dist(prev.sky_color, cur.sky_color) < 0.08
        assert abs(prev.fog_density - cur.fog_density) < 0.01
        prev = cur
