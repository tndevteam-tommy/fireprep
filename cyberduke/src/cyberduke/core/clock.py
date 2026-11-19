"""Frame timing helpers, and the city's clock (the time of day)."""

from __future__ import annotations


def clamp_dt(raw_dt: float, max_dt: float) -> float:
    """Caps a frame delta so a hitch (window drag, GC pause, breakpoint) cannot make an
    agent tunnel through a wall or overshoot an exponential smoother (bug B4)."""
    if raw_dt < 0.0:
        return 0.0
    return min(raw_dt, max_dt)


# ============================================================================ the city's day
MINUTES_PER_DAY = 24 * 60


def hhmm(hour: float, minute: float = 0.0) -> float:
    """Minutes since midnight of ``hour``:``minute``."""
    return hour * 60.0 + minute


def within(hour: float, start: float, end: float) -> bool:
    """Whether ``hour`` (0..24) falls in [start, end) on the clock - wrapping past
    midnight when ``end`` is earlier than ``start`` (22 -> 6 is the night)."""
    hour %= 24.0
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _ramp(x: float, a: float, b: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    u = (x - a) / (b - a)
    return u * u * (3.0 - 2.0 * u)


class WorldClock:
    """The time of day in the city: ``minutes`` since midnight. It runs at ``rate`` game
    minutes per real second (config ``CLOCK``) - a day goes by in under an hour - and
    keeps running whichever level the netrunner is in. Only the time of day matters
    (schedules, daylight): the clock does not count the days that have gone by."""

    def __init__(self, minutes: float = 0.0, rate: float = 1.0) -> None:
        self.minutes = minutes % MINUTES_PER_DAY
        self.rate = rate

    def advance(self, dt: float) -> None:
        self.minutes = (self.minutes + max(0.0, dt) * self.rate) % MINUTES_PER_DAY

    def set(self, minutes: float) -> None:
        self.minutes = minutes % MINUTES_PER_DAY

    @property
    def hour(self) -> float:
        """Hours since midnight, fractional (21.5 is half past nine at night)."""
        return self.minutes / 60.0

    @property
    def text(self) -> str:
        """The time as a 24-hour digital clock shows it: "21:07"."""
        m = int(self.minutes)
        return f"{m // 60:02d}:{m % 60:02d}"

    def is_between(self, start: float, end: float) -> bool:
        return within(self.hour, start, end)

    def hours_until(self, hour: float) -> float:
        """How long (in game hours) until the clock next reads ``hour``."""
        return (hour - self.hour) % 24.0

    @property
    def daylight(self) -> float:
        """0 at night, 1 in the (smog-brown) day: dawn 05:30 - 08:00, dusk 17:30 - 20:00."""
        h = self.hour
        return _ramp(h, 5.5, 8.0) * (1.0 - _ramp(h, 17.5, 20.0))

    def seconds_for(self, game_hours: float) -> float:
        """Real seconds that ``game_hours`` take at this clock's rate."""
        return game_hours * 60.0 / max(self.rate, 1e-9)
