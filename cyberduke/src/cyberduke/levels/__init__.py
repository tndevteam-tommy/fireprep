"""Shipped levels as typed Python data, plus a JSON/TOML loader for external level files."""

from cyberduke.levels.facility import FACILITY, build_facility
from cyberduke.levels.loader import (
    LevelFormatError,
    level_from_dict,
    level_to_dict,
    load_level,
    save_level,
)
from cyberduke.levels.street import NIGHT_MARKET, build_night_market

LEVELS = {
    "facility": build_facility,
    "night_market": build_night_market,
}
"""Level name -> builder, for ``LevelData.exit_to`` transitions and ``--level``."""

__all__ = [
    "FACILITY",
    "LEVELS",
    "NIGHT_MARKET",
    "LevelFormatError",
    "build_facility",
    "build_night_market",
    "level_from_dict",
    "level_to_dict",
    "load_level",
    "save_level",
]
