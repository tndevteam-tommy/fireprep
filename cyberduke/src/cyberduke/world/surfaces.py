"""Floor elevation for multi-level navigation (ground, stairs, elevated catwalks).

``floor_height`` is a pure function: it returns the new elevation *and* the new tracking
state instead of mutating the player.

The stairs are treated as a stepped surface: an agent inside a stair footprint stands on
the top of the step beneath it. Nothing can be *under* the stairs - the level's colliders
close the back and sides of each staircase (see ``tests/test_traversability.py``), so the
footprint alone decides the elevation. That is what fixes the original "walk through the
underside of the ramp onto the catwalk" defect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ElevationState:
    """Whether an agent is currently on the elevated level (deck or upper stairs) - the
    facility's catwalk, or (``deck_y``) the raised floor a flight of stairs climbs to (the
    metro station's landing, footbridge and platform: ``Flight``)."""

    on_catwalk: bool = False
    deck_y: float | None = None


@dataclass(frozen=True)
class Flight:
    """A straight flight of stairs (``LevelData.flights``): ``steps`` treads climbing
    from ``bottom`` to ``top`` across its footprint the way ``axis`` says ("+x": up
    towards +x). Whoever is on it stands on the tread under them; near the top (past
    ``commit_fraction`` of the way up) they count as up on the raised floor it leads to
    (``LevelData.decks`` at ``top``), and walk on along it instead of under it. The
    level closes its sides and its underside (colliders), so the footprint decides."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float
    bottom: float
    top: float
    steps: int
    axis: str = "+x"
    commit_fraction: float = 0.95

    def contains(self, x: float, z: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_z <= z <= self.max_z

    def fraction(self, x: float, z: float) -> float:
        """How far up it (x, z) is: 0 at its foot .. 1 at its top."""
        if self.axis in ("+x", "-x"):
            t = (x - self.min_x) / (self.max_x - self.min_x)
        else:
            t = (z - self.min_z) / (self.max_z - self.min_z)
        return t if self.axis.startswith("+") else 1.0 - t

    def height(self, x: float, z: float) -> float:
        """The top of the tread under (x, z) (the first is one rise above its foot)."""
        index = min(self.steps - 1, max(0, math.floor(self.fraction(x, z) * self.steps)))
        return self.bottom + (index + 1) * (self.top - self.bottom) / self.steps


@dataclass(frozen=True)
class Catwalk:
    """Symmetric pair of catwalks at |x| in [inner_x, outer_x] joined by a rear skybridge,
    each reached by a staircase rising along +Z from ``stairs_z_start`` to ``stairs_z_end``."""

    inner_x: float = 6.0
    outer_x: float = 9.6
    stairs_z_start: float = 21.0
    stairs_z_end: float = 24.2
    deck_z_end: float = 42.0
    bridge_z_start: float = 39.5
    height: float = 2.75
    steps: int = 8
    commit_fraction: float = 0.95  # past this point on the stairs you count as "up"

    @property
    def stairs_length(self) -> float:
        return self.stairs_z_end - self.stairs_z_start

    @property
    def step_run(self) -> float:
        return self.stairs_length / self.steps

    @property
    def step_rise(self) -> float:
        return self.height / self.steps

    def on_stairs(self, x: float, z: float) -> bool:
        return (
            self.inner_x <= abs(x) <= self.outer_x and self.stairs_z_start <= z <= self.stairs_z_end
        )

    def on_deck_footprint(self, x: float, z: float) -> bool:
        side = self.inner_x <= abs(x) <= self.outer_x and self.stairs_z_end < z <= self.deck_z_end
        bridge = abs(x) <= self.outer_x and self.bridge_z_start <= z <= self.deck_z_end
        return side or bridge

    def stairs_fraction(self, z: float) -> float:
        return (z - self.stairs_z_start) / self.stairs_length

    def step_height(self, z: float) -> float:
        """Top surface of the step under ``z`` (the first step is one rise above the floor)."""
        index = min(self.steps - 1, max(0, math.floor(self.stairs_fraction(z) * self.steps)))
        return (index + 1) * self.step_rise


def floor_height(
    x: float, z: float, state: ElevationState, catwalk: Catwalk
) -> tuple[float, ElevationState]:
    """Returns ``(floor_y, new_state)`` for an agent at (x, z)."""
    if catwalk.on_stairs(x, z):
        committed = catwalk.stairs_fraction(z) >= catwalk.commit_fraction
        return catwalk.step_height(z), replace(state, on_catwalk=committed)

    if catwalk.on_deck_footprint(x, z):
        # Either up on the deck, or walking on the ground underneath it.
        return (catwalk.height if state.on_catwalk else 0.0), state

    return 0.0, replace(state, on_catwalk=False)


def on_flights_and_decks(
    x: float, z: float, y: float, state: ElevationState, flights, decks
) -> tuple[float, ElevationState]:
    """The floor under (x, z) with any flights of stairs and raised floors (decks: the
    floor at each box's ``max_y``) taken into account: on a flight, its tread (and up on
    the deck it climbs to once near the top); off it, still on the deck while (x, z) is
    over one at that height - down on the ground (``y``) once off every one (the level's
    colliders and walk regions keep anybody from stepping off a deck's edge)."""
    for flight in flights:
        if flight.contains(x, z):
            up = flight.fraction(x, z) >= flight.commit_fraction
            return max(y, flight.height(x, z)), replace(state, deck_y=flight.top if up else None)
    if state.deck_y is not None:
        level = state.deck_y
        if any(d.contains_xz(x, z) and abs(d.max_y - level) < 1e-6 for d in decks):
            return max(y, level), state
        return y, replace(state, deck_y=None)
    return y, state
