"""
Firestarter Simulation Module

Public names are resolved lazily (PEP 562) so sub-modules can import each other without
circular-import problems.
"""

import importlib

_EXPORTS = {
    "WeatherSystem": "weather",
    "Season": "weather",
    "TimeEngine": "time_engine",
    "TIME_SKIP_OPTIONS": "time_engine",
    "Player": "player",
    "WorldManager": "world_manager",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f"firestarter.simulation.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'firestarter.simulation' has no attribute {name!r}")
