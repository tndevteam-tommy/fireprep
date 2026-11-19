"""The metro: a two-car monorail train up and down the line over the main street.

Every few minutes (``Metro.interval``: less often in the small hours) a train comes out of
the tunnel at one end, runs along the beam, slows and stops at the station's platform
(``world.level.MetroData``), opens its doors for a while, closes them and pulls away into
the tunnel at the other end; the next one comes the other way. Whoever is waiting on the
platform gets on - walks through a door and on into the dark of the car - and
whoever was aboard for this stop gets off (``ai.alley``: the commuters). A train holds its
doors for anybody still on their way to them.

The station's ticket gates (``MetroData.gates``) open for a moment for whoever has a pass
- and for whoever walks up to one a netrunner has forced (its lock is a door on the level:
``MetroData.gate_doors``; ``gate_doors`` here are those doors, for the flaps to show).

Pure simulation (``render.metro`` draws the train, its doors and the gates).
"""

from __future__ import annotations

import math

from cyberduke.core.clock import WorldClock, within
from cyberduke.world.level import MetroData
from cyberduke.world.vehicles import car_middles, train_doors


class Metro:
    CRUISE = 16.0  # m/s along the line
    BRAKE = 1.1  # m/s^2 slowing into the station
    ACCEL = 1.0  # m/s^2 pulling away
    DWELL = 14.0  # seconds at the platform (doors open and shut included)
    DOORS = 1.6  # seconds the doors take to open (and to close)
    HOLD_MAX = 20.0  # seconds a train will wait for passengers on their way to it
    GATE_OPEN = 1.8  # seconds a ticket gate stays open for somebody with a pass
    SETTLE = 0.8  # seconds between the doors shutting and the train moving off

    def __init__(self, data: MetroData, clock: WorldClock | None = None) -> None:
        self.data = data
        self.clock = clock
        self.gate_doors: dict[int, object] = {}  # lane -> the door that is its flaps' lock
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        self.phase = "away"  # away (in a tunnel) -> arriving -> stopped -> departing
        self.heading = 1.0  # the way the next train comes (+1: from the west)
        self.x = self.data.x_west
        self.speed = 0.0
        self.wait = 8.0  # seconds until the next train comes out of a tunnel
        self.dwell = 0.0
        self.held = 0.0
        self.boarding: set[object] = set()  # on their way into a door
        self.riders: list[object] = []  # aboard
        self.gates: dict[int, float] = {}  # lane -> seconds left open
        self.trains = 0  # how many have stopped here

    # ------------------------------------------------------------------ the timetable
    def interval(self) -> float:
        """Seconds between one train leaving and the next coming in: every couple of
        minutes, half as often in the small hours."""
        hour = self.clock.hour if self.clock is not None else 20.0
        return 100.0 if within(hour, 1.0, 5.0) else 45.0

    # ------------------------------------------------------------------ where it is
    @property
    def on_the_line(self) -> bool:
        """A train is out of the tunnels (coming in, standing, or pulling out)."""
        return self.phase != "away"

    @property
    def stopped(self) -> bool:
        return self.phase == "stopped"

    @property
    def doors(self) -> float:
        """How far open its doors are (0 shut .. 1 open)."""
        if self.phase != "stopped":
            return 0.0
        closing_at = self.closing_at
        if self.dwell < self.DOORS:
            return self.dwell / self.DOORS
        if self.dwell > closing_at:
            return max(0.0, 1.0 - (self.dwell - closing_at) / self.DOORS)
        return 1.0

    @property
    def closing_at(self) -> float:
        """When in the stop the doors start to close (later, while somebody is boarding)."""
        return self.DWELL - self.DOORS + self.held

    @property
    def doors_open(self) -> bool:
        return self.doors >= 1.0

    @property
    def boarding_time(self) -> bool:
        """The doors are open and not about to close: time to get on (or off)."""
        return self.doors_open and self.dwell < self.closing_at - 1.0

    def door_xs(self) -> tuple[float, ...]:
        """The middles of the train's doors (x), where it stands now."""
        return train_doors(self.x)

    def car_xs(self) -> tuple[float, ...]:
        return car_middles(self.x)

    # ------------------------------------------------------------------ passengers
    def start_boarding(self, who) -> None:
        """``who`` is on the way into a door: the doors wait for them."""
        self.boarding.add(who)

    def board(self, who) -> None:
        """Through the door into the dark of the car: aboard (out of sight)."""
        self.boarding.discard(who)
        if who not in self.riders:
            self.riders.append(who)

    def alight(self, who) -> None:
        if who in self.riders:
            self.riders.remove(who)

    def open_gate(self, lane: int) -> None:
        self.gates[lane] = self.GATE_OPEN

    def gate(self, lane: int) -> float:
        """How far open ticket gate ``lane`` is (0 .. 1): quick to open, quick to shut -
        for a pass, or forced (its door open)."""
        door = self.gate_doors.get(lane)
        forced = door.open_amount / door.travel if door is not None else 0.0  # type: ignore[attr-defined]
        left = self.gates.get(lane, 0.0)
        if left <= 0.0:
            return forced
        opened = self.GATE_OPEN - left
        return max(forced, min(1.0, opened / 0.3, left / 0.3))

    # ------------------------------------------------------------------ per frame
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return
        self.time += dt
        for lane in list(self.gates):
            self.gates[lane] -= dt
            if self.gates[lane] <= 0.0:
                del self.gates[lane]
        d = self.data
        if self.phase == "away":
            self.wait -= dt
            if self.wait <= 0.0:
                self.phase = "arriving"
                self.x = d.x_west if self.heading > 0 else d.x_east
                self.speed = self.CRUISE
        elif self.phase == "arriving":
            left = (d.stop_x - self.x) * self.heading
            self.speed = min(self.CRUISE, math.sqrt(2.0 * self.BRAKE * max(left, 0.0)) + 0.25)
            step = self.speed * dt
            if step >= left:
                self.x = d.stop_x
                self.speed = 0.0
                self.phase = "stopped"
                self.dwell = 0.0
                self.held = 0.0
                self.trains += 1
            else:
                self.x += step * self.heading
        elif self.phase == "stopped":
            self.dwell += dt
            # somebody still on their way into a door (the doors open): it waits for them
            if self.boarding and self.held < self.HOLD_MAX and self.dwell >= self.closing_at - 1.0:
                self.held += dt
            if self.dwell >= self.closing_at + self.DOORS + self.SETTLE:
                self.boarding.clear()
                self.phase = "departing"
                self.speed = 0.0
        elif self.phase == "departing":
            gone = (self.x - d.stop_x) * self.heading
            self.speed = min(self.CRUISE, math.sqrt(2.0 * self.ACCEL * max(gone, 0.0)) + 0.4)
            self.x += self.speed * dt * self.heading
            end = d.x_east if self.heading > 0 else d.x_west
            if (end - self.x) * self.heading <= 0.0:
                self.phase = "away"
                self.heading = -self.heading
                self.wait = self.interval()
                self.speed = 0.0

    def due_in(self) -> float:
        """Seconds until the next train stops here (roughly: the departures board)."""
        d = self.data
        run = abs((d.x_east if self.heading < 0 else d.x_west) - d.stop_x) / (self.CRUISE * 0.8)
        if self.phase == "away":
            return self.wait + run
        if self.phase == "arriving":
            return abs(d.stop_x - self.x) / max(self.speed, 1.0)
        return 0.0


__all__ = ["Metro"]
