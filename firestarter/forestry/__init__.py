"""
Firestarter Forestry Module

Public names are resolved lazily (PEP 562) so sub-modules can import each other without
circular-import problems.
"""

import importlib

_EXPORTS = {
    "Tree": "tree",
    "TreeGrowthStage": "tree",
    "TreeFellingSystem": "felling",
    "RaisedDryingRack": "seasoning",
    "GroundSeasoningSimulator": "seasoning",
    "ForestEcosystem": "ecosystem",
    "AshDepositZone": "ecosystem",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f"firestarter.forestry.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'firestarter.forestry' has no attribute {name!r}")
