"""Where the game keeps the player's files: ``<home>/Unprepared``.

* ``settings.json`` - display mode (fullscreen or a window) and graphics preset, read at
  start-up so the game comes up the way it was left;
* ``Saves/`` - save games, one JSON file per save, named by the date and time it was made
  (``save_2031-04-02_23-15-07.json``): a new save never overwrites an old one.

``CYBERDUKE_USER_DIR`` overrides the location (the tests point it at a temporary folder).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

APP_DIR_NAME = "Unprepared"
SAVES_DIR_NAME = "Saves"
SETTINGS_FILE = "settings.json"
GRAPHICS_CHOICES = ("LOW", "MEDIUM", "HIGH")


def user_dir() -> Path:
    """``<home>/Unprepared`` (or ``$CYBERDUKE_USER_DIR``)."""
    override = os.environ.get("CYBERDUKE_USER_DIR")
    return Path(override) if override else Path.home() / APP_DIR_NAME


def saves_dir() -> Path:
    return user_dir() / SAVES_DIR_NAME


@dataclass
class Settings:
    fullscreen: bool = False
    graphics: str | None = None  # None: detect from the GPU on start-up

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Settings:
        graphics = d.get("graphics")
        if graphics is not None:
            graphics = str(graphics).upper()
            if graphics not in GRAPHICS_CHOICES:
                graphics = None
        return cls(fullscreen=bool(d.get("fullscreen", False)), graphics=graphics)


def load_settings() -> Settings:
    """The saved settings, or the defaults if there are none (or they are unreadable)."""
    path = user_dir() / SETTINGS_FILE
    try:
        return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, AttributeError):
        return Settings()


def save_settings(settings: Settings) -> Path:
    path = user_dir() / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
