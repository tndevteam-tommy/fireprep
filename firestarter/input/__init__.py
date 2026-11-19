"""
Firestarter Input Module

Public names are resolved lazily (PEP 562) so sub-modules can import each other without
circular-import problems.
"""

import importlib

_EXPORTS = {
    "HandController": "hand_physics_rig",
    "HandJointState": "hand_physics_rig",
    "InputManager": "input_manager",
    "Settings": "input_manager",
    "Tutorial": "tutorial",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f"firestarter.input.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'firestarter.input' has no attribute {name!r}")
