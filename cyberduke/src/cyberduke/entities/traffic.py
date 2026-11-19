"""The traffic on the main street: cars, taxis and vans driving by.

Each lane (``world.level.LaneData``) runs from one tunnel to the other. Now and then a
vehicle comes out of the distance in one of them - a model and a paint job of its own
(``world.vehicles``), a speed it likes to drive at - and drives down the lane and off into
the other tunnel, its wheels turning with the road.

Nobody overtakes: a car that comes up behind a slower one slows down and follows it at a
safe distance (the *intelligent driver model* - it eases off as the gap closes, brakes
harder the faster it is closing, and gets going again when the gap opens), its brake
lights on while it brakes. A new vehicle only comes out of a tunnel when the lane is
clear there. The road is busier in the evening rush than in the dead of night.

Pure simulation (no drawing: ``render.traffic`` draws the vehicles where they are).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from cyberduke.core.clock import WorldClock, within
from cyberduke.world.level import LaneData
from cyberduke.world.vehicles import Paint, VehicleModel, random_vehicle, wheel_turn

RGB = tuple[float, float, float]


class Car:
    """One vehicle on the road."""

    def __init__(
        self, car_id: int, model: VehicleModel, paint: Paint, lane: LaneData, cruise: float,
        speed: float, x: float | None = None,
    ) -> None:  # fmt: skip
        self.car_id = car_id
        self.model = model
        self.paint = paint
        self.lane = lane
        self.cruise = cruise  # the speed it likes (m/s)
        self.speed = speed
        self.x = lane.x_from if x is None else x
        self.accel = 0.0
        self.spin = 0.0  # how far its wheels have turned (degrees)

    # ------------------------------------------------------------------ where it is
    @property
    def y(self) -> float:
        return self.lane.y

    @property
    def z(self) -> float:
        return self.lane.z

    @property
    def heading(self) -> float:
        return self.lane.heading

    @property
    def yaw(self) -> float:
        """The model's nose is at its -x: facing -x unturned, turned round to face +x."""
        return 0.0 if self.heading < 0 else 180.0

    @property
    def progress(self) -> float:
        """How far along its lane it has come (metres from where it came in)."""
        return (self.x - self.lane.x_from) * self.heading

    @property
    def braking(self) -> bool:
        return self.accel < -0.8

    @property
    def length(self) -> float:
        return self.model.length

    def front(self) -> float:
        return self.progress + self.length / 2.0

    def back(self) -> float:
        return self.progress - self.length / 2.0


class Traffic:
    """The vehicles on the level's lanes (``LevelData.lanes``)."""

    # the intelligent driver model
    ACCEL = 2.2  # m/s^2 pulling away
    COMFORT_BRAKE = 3.5  # m/s^2 braking it is happy with
    MAX_BRAKE = 9.0  # m/s^2 at most
    MIN_GAP = 3.0  # metres kept to the car in front, standing
    HEADWAY = 1.3  # seconds kept to the car in front, moving
    ENTRY_CLEAR = 9.0  # metres clear behind the last car before another comes out
    WARM_UP = 60.0  # seconds of traffic run before the street is first seen

    def __init__(
        self, lanes: Sequence[LaneData], seed: int = 2077, clock: WorldClock | None = None
    ) -> None:
        self.lanes = tuple(lanes)
        self.seed = seed
        self.clock = clock
        self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self.cars: list[Car] = []
        self.next_id = 1
        self.time = 0.0
        self.due = {lane.lane_id: self.rng.uniform(0.0, 4.0) for lane in self.lanes}
        for _ in range(int(self.WARM_UP / 0.25)):  # (the road is busy already when seen)
            self.update(0.25)

    # ------------------------------------------------------------------ how busy
    def interval(self) -> tuple[float, float]:
        """Seconds between vehicles coming out of a tunnel, by the time of day: the
        evening rush is busiest, the small hours quietest."""
        hour = self.clock.hour if self.clock is not None else 20.0
        if within(hour, 17.0, 21.0) or within(hour, 7.0, 9.5):
            return 3.0, 8.0
        if within(hour, 1.0, 5.5):
            return 9.0, 22.0
        return 4.5, 12.0

    # ------------------------------------------------------------------ per frame
    def update(self, dt: float) -> None:
        if dt <= 0.0:
            return
        self.time += dt
        for lane in self.lanes:
            cars = self.in_lane(lane)
            for k, car in enumerate(cars):
                ahead = cars[k - 1] if k > 0 else None
                self._drive(car, ahead, dt)
            self.cars = [c for c in self.cars if c.lane is not lane or c.back() < lane.length]
            self.due[lane.lane_id] -= dt
            if self.due[lane.lane_id] <= 0.0:
                self._come_out(lane)

    def in_lane(self, lane: LaneData) -> list[Car]:
        """The lane's vehicles, the one furthest along first."""
        return sorted((c for c in self.cars if c.lane is lane), key=lambda c: -c.progress)

    def ahead_of(self, car: Car) -> Car | None:
        """The vehicle in front of ``car`` in its lane (None: the road ahead is clear)."""
        best = None
        for other in self.cars:
            if other is car or other.lane is not car.lane or other.progress <= car.progress:
                continue
            if best is None or other.progress < best.progress:
                best = other
        return best

    def gap(self, car: Car, ahead: Car) -> float:
        """Bumper to bumper, metres."""
        return ahead.back() - car.front()

    def _drive(self, car: Car, ahead: Car | None, dt: float) -> None:
        v, v0 = car.speed, max(car.cruise, 0.1)
        free = 1.0 - (v / v0) ** 4
        interaction = 0.0
        if ahead is not None:
            s = max(self.gap(car, ahead), 0.1)
            closing = v - ahead.speed
            wanted = self.MIN_GAP + max(
                0.0,
                v * self.HEADWAY + v * closing / (2.0 * math.sqrt(self.ACCEL * self.COMFORT_BRAKE)),
            )
            interaction = (wanted / s) ** 2
        a = self.ACCEL * (free - interaction)
        a = max(-self.MAX_BRAKE, a)
        car.accel = a
        new_v = max(0.0, v + a * dt)
        distance = (v + new_v) / 2.0 * dt
        if ahead is not None:  # never into the back of it, whatever the maths says
            distance = min(distance, max(0.0, self.gap(car, ahead) - 0.5))
        car.speed = new_v
        car.x += distance * car.heading
        car.spin = (car.spin + wheel_turn(distance, car.model.wheel_radius)) % 360.0

    def _come_out(self, lane: LaneData) -> None:
        """A vehicle comes out of the tunnel into ``lane`` - if there is room for it."""
        last = min((c for c in self.cars if c.lane is lane), key=lambda c: c.progress, default=None)
        if last is not None and last.back() < self.ENTRY_CLEAR:
            return  # (next frame, then)
        model, paint = random_vehicle(self.rng)
        cruise = self.rng.uniform(*model.cruise)
        speed = cruise if last is None else min(cruise, last.speed)
        self.cars.append(Car(self.next_id, model, paint, lane, cruise, speed))
        self.next_id += 1
        self.due[lane.lane_id] = self.rng.uniform(*self.interval())


__all__ = ["Car", "Traffic"]
