"""Top-level game state."""

from __future__ import annotations

from enum import Enum


class GameState(str, Enum):
    """There is no winning any more - the run goes on. Caught, the netrunner sees a red
    "you were caught" screen that fades to black, then pays for it (a new game in the
    facility; thrown out of somebody's home in the city)."""

    PLAYING = "PLAYING"
    CAUGHT = "CAUGHT"
