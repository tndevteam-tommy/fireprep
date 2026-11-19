"""Single owner of the player's detection ("stealth") meter.

Observers (guards, cameras) *report* sightings with a gain rate; once per frame the system
applies the summed gain or, if nobody saw the player, the decay. The original code let every
observer and the main loop each apply decay, so the meter drained 3x too fast and depended
on update order (bug B1).
"""

from __future__ import annotations

from typing import Protocol

from cyberduke.config import DETECTION


class SightingSink(Protocol):
    @property
    def level(self) -> float: ...

    def report_sighting(self, rate: float) -> None: ...


class DetectionSystem:
    def __init__(
        self, decay_rate: float = DETECTION.decay_rate, max_level: float = DETECTION.max_level
    ):
        self.decay_rate = decay_rate
        self.max_level = max_level
        # Scales every sighting's gain (the netrunner's cyberware: a signal damper makes
        # them slower to be sure of) - set each frame by the simulation.
        self.gain_scale = 1.0
        self.reset()

    def reset(self) -> None:
        self.level: float = 0.0
        self._gain = 0.0
        self._spotted = False
        self.spotted_last_frame = False

    # ------------------------------------------------------------------ per-frame protocol
    def begin_frame(self) -> None:
        self._gain = 0.0
        self._spotted = False

    def report_sighting(self, rate: float) -> None:
        """Called by any observer that currently has line of sight. Rates are additive."""
        self._gain += rate
        self._spotted = True

    def end_frame(self, dt: float) -> None:
        if self._spotted:
            self.level = min(self.max_level, self.level + self._gain * self.gain_scale * dt)
        else:
            self.level = max(0.0, self.level - self.decay_rate * dt)
        self.spotted_last_frame = self._spotted

    # ------------------------------------------------------------------ queries
    @property
    def is_compromised(self) -> bool:
        return self.level >= self.max_level

    @property
    def fraction(self) -> float:
        return self.level / self.max_level
