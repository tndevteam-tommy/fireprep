
import pytest

from unprepared.core.daynight import DayNightCycle, DayPhase


@pytest.fixture
def cycle():
    return DayNightCycle(ambient_day=1.0, ambient_night=0.1)


@pytest.mark.parametrize("hour,expected", [(0, 0.1), (5, 0.1), (6, 1.0), (12, 1.0), (19, 1.0), (20, 0.1), (21, 0.1), (23, 0.1)])
def test_ambient_light(cycle, hour, expected):
    assert cycle.ambient_light(hour) == expected


@pytest.mark.parametrize("hour,expected", [(5, False), (6, True), (19, True), (20, False), (23, False)])
def test_is_daytime(hour, expected):
    assert DayNightCycle.is_daytime(hour) is expected


@pytest.mark.parametrize("hour,expected", [(17, False), (17.25, False), (17.6, True), (18, True), (19, True), (20, True),
                                           (23, True), (0, True), (5, True), (6, True), (6.25, True), (6.6, False), (7, False)])
def test_torch_time_follows_the_sun_not_the_clock(hour, expected):
    """NPCs light up once the sun is low (about 17:30) and put out at about 06:30."""
    assert DayNightCycle.is_torch_time(hour) is expected


def test_sun_rises_at_six_and_sets_at_six():
    from unprepared.core.daynight import SUNRISE, SUNSET, sun_direction
    assert (SUNRISE, SUNSET) == (6.0, 18.0)
    assert abs(sun_direction(6.0)[1]) < 1e-9 and abs(sun_direction(18.0)[1]) < 1e-9
    assert sun_direction(12.0)[1] > 0.95 and sun_direction(0.0)[1] < -0.95


@pytest.mark.parametrize("hour,phase", [
    (0, DayPhase.NIGHT), (4, DayPhase.NIGHT), (5, DayPhase.DUSK), (6, DayPhase.DAY), (17, DayPhase.DAY),
    (18, DayPhase.DUSK), (19, DayPhase.DUSK), (20, DayPhase.NIGHT), (23, DayPhase.NIGHT),
])
def test_phase(hour, phase):
    assert DayNightCycle.phase(hour) is phase


def test_night_boundaries_are_consistent():
    """Ambient light and 'daytime' (the rules) switch at 20:00 and 06:00; torches come out earlier."""
    for hour in range(24):
        night = DayNightCycle.is_night(hour)
        assert DayNightCycle.is_daytime(hour) is (not night)
        assert (DayNightCycle(1.0, 0.1).ambient_light(hour) == 0.1) is night
        if night:
            assert DayNightCycle.is_torch_time(hour)

