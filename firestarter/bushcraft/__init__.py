"""
Firestarter Bushcraft Module

Public names are resolved lazily (PEP 562) so sub-modules can import each other without
circular-import problems.
"""

import importlib

_EXPORTS = {
    "BowDrillSystem": "bow_drill",
    "HandDrillSystem": "hand_drill",
    "TinderNest": "tinder_nest",
    "SmolderingFungus": "fungi",
    "FungusType": "fungi",
    "BushcraftCraftingController": "crafting_controller",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f"firestarter.bushcraft.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'firestarter.bushcraft' has no attribute {name!r}")
