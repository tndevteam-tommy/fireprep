"""When the people of the street are where: the hours of their days.

Everybody with a home (``levels.flats``, number 16) keeps hours by the city's clock
(``core.clock.WorldClock``): when they are out and about (their story in ``ai.alley``
plays), when they are at home (pottering about the flat) and, within that, asleep in
bed - and for the office worker, when he is at work (from home: at the computer in his
front room). Nobody goes out of sight: every stretch of the day is spent somewhere you
can walk up to - the street, or behind their own front door.

* the **sushi chef** works long shifts: the stall is open from 14:00 to 02:00;
* the **junk dealer** opens up at 10:00 and heads home at midnight;
* the **office worker** works from home 07:30 - 17:00 and comes by after it (the evening
  rush starts at five), home by 23:00;
* the **tinkerer** and the **stroller** are late risers, the **courier** works the day;
* **Nix** is almost always out by his wall (ready to talk business) - home only for a
  nap before dawn and a late lunch, an hour or two in all;
* **Dell** (the resident) keeps to his flat and goes out on errands at any hour he is
  awake - for sushi while the stall is open, to the vending machine when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from cyberduke.core.clock import within

Span = tuple[float, float]  # (from, to) in hours on the clock (wrapping past midnight)


@dataclass(frozen=True)
class Schedule:
    home: tuple[Span, ...] = ()  # at home (in the flat)
    sleep: Span | None = None  # in bed (within a stretch at home)
    work: tuple[Span, ...] = ()  # working - from home, at the computer in the front room

    def at_home(self, hour: float) -> bool:
        return any(within(hour, a, b) for a, b in self.home)

    def asleep(self, hour: float) -> bool:
        return self.sleep is not None and within(hour, *self.sleep)

    def working(self, hour: float) -> bool:
        return any(within(hour, a, b) for a, b in self.work)

    def indoors(self, hour: float) -> bool:
        """Behind their own front door: at home, or working there."""
        return self.at_home(hour) or self.working(hour)

    def out(self, hour: float) -> bool:
        """Out and about on the street: neither at home nor at work."""
        return not self.indoors(hour)


SCHEDULES: dict[str, Schedule] = {
    "sushi_chef": Schedule(home=((2.0, 14.0),), sleep=(3.0, 12.5)),
    "junk_dealer": Schedule(home=((0.0, 10.0),), sleep=(1.0, 9.0)),
    "office_worker": Schedule(home=((23.0, 7.5),), sleep=(23.5, 7.0), work=((7.5, 17.0),)),
    "tinkerer": Schedule(home=((23.5, 11.0),), sleep=(0.5, 10.0)),
    "courier": Schedule(home=((22.0, 9.0),), sleep=(23.0, 8.0)),
    "stroller": Schedule(home=((2.5, 12.0),), sleep=(3.5, 11.0)),
    # Nix: out nearly all day and night - a nap before dawn, a late lunch
    "smoker": Schedule(home=((5.0, 6.5), (13.5, 14.25)), sleep=(5.1, 6.4)),
    # Dell: home unless he is out on an errand (his story decides); asleep in the small hours
    "resident": Schedule(sleep=(2.0, 9.5)),
}

# The stalls' opening hours (their merchants' hours out)
OPENING = {"sushi": "sushi_chef", "junk": "junk_dealer"}


def schedule(role: str) -> Schedule:
    return SCHEDULES.get(role, Schedule())


def stall_hours_open(stall: str, hour: float) -> bool:
    """Whether ``stall`` ("sushi", "junk") keeps these hours open (the merchant decides
    when exactly: a customer is served before closing)."""
    role = OPENING.get(stall)
    return role is None or schedule(role).out(hour)


__all__ = ["OPENING", "SCHEDULES", "Schedule", "schedule", "stall_hours_open"]
