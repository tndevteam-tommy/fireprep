"""The netrunner's scanner: a ping that reveals nearby electronics.

A pulse reveals every hack target within ``scan_range`` of the viewpoint - through walls,
because it is a network ping rather than a sight check - and keeps it revealed for
``reveal_duration`` seconds. What a revealed target *is* (its options, whether it can be
hacked from here) is the ``HackingSystem``'s business; the scanner only answers "has the
player pinged this recently?".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from cyberduke.config import HACKING, HackingConfig
from cyberduke.core.geometry import Vec3, distance_3d


class Pingable(Protocol):
    """The little a target needs for the scanner to find it."""

    @property
    def hack_id(self) -> str: ...

    @property
    def hack_position(self) -> Vec3: ...


T = TypeVar("T", bound=Pingable)


@dataclass(frozen=True)
class ScanPulse:
    """A pulse in flight, for the renderer (expanding ring) and the HUD."""

    origin: Vec3
    age: float
    max_radius: float
    duration: float

    @property
    def radius(self) -> float:
        return self.max_radius * min(1.0, self.age / self.duration)

    @property
    def done(self) -> bool:
        return self.age >= self.duration


class Scanner:
    def __init__(self, config: HackingConfig = HACKING) -> None:
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._revealed: dict[str, float] = {}  # hack id -> seconds of reveal left
        self.cooldown = 0.0
        self.pulse: ScanPulse | None = None
        self.pulses_fired = 0

    # ------------------------------------------------------------------ commands
    @property
    def ready(self) -> bool:
        return self.cooldown <= 0.0

    def pulse_from(self, origin: Vec3, targets: Iterable[T], range_scale: float = 1.0) -> list[T]:
        """Fires a pulse from ``origin``. Returns the targets it revealed (in range), or an
        empty list if the scanner is still recharging. ``range_scale`` stretches the ping
        (a ping amplifier mod)."""
        if not self.ready:
            return []
        reach = self.cfg.scan_range * range_scale
        self.cooldown = self.cfg.scan_cooldown
        self.pulses_fired += 1
        self.pulse = ScanPulse(origin, 0.0, reach, self.cfg.pulse_visual_duration)
        hits = [t for t in targets if distance_3d(origin, t.hack_position) <= reach]
        for t in hits:
            self._revealed[t.hack_id] = self.cfg.reveal_duration
        return hits

    def reveal(self, hack_id: str, seconds: float | None = None) -> None:
        """Marks a target revealed without a pulse (e.g. something you just hacked)."""
        self._revealed[hack_id] = self.cfg.reveal_duration if seconds is None else seconds

    # ------------------------------------------------------------------ queries
    def is_revealed(self, hack_id: str) -> bool:
        return self._revealed.get(hack_id, 0.0) > 0.0

    def time_left(self, hack_id: str) -> float:
        return max(0.0, self._revealed.get(hack_id, 0.0))

    @property
    def revealed_ids(self) -> list[str]:
        return [k for k, v in self._revealed.items() if v > 0.0]

    # ------------------------------------------------------------------ per frame
    def update(self, dt: float) -> None:
        if self.cooldown > 0.0:
            self.cooldown = max(0.0, self.cooldown - dt)
        if self.pulse is not None:
            self.pulse = ScanPulse(
                self.pulse.origin, self.pulse.age + dt, self.pulse.max_radius, self.pulse.duration
            )
            if self.pulse.done:
                self.pulse = None
        for key in list(self._revealed):
            self._revealed[key] -= dt
            if self._revealed[key] <= 0.0:
                del self._revealed[key]
