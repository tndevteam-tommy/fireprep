"""
Firestarter Recipes Module

Public names are resolved lazily (PEP 562) so sub-modules can import each other without
circular-import problems.
"""

import importlib

_EXPORTS = {
    "FireBlueprint": "blueprints",
    "BlueprintSlot": "blueprints",
    "ALL_BLUEPRINTS": "blueprints",
    "create_teepee_blueprint": "blueprints",
    "create_log_cabin_blueprint": "blueprints",
    "create_lean_to_blueprint": "blueprints",
    "create_swedish_torch_blueprint": "blueprints",
    "create_star_fire_blueprint": "blueprints",
    "ReferenceBook": "reference_book",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f"firestarter.recipes.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'firestarter.recipes' has no attribute {name!r}")
