"""
Firestarter Economy Module

Public names are resolved lazily (PEP 562) so sub-modules can import each other without
circular-import problems.
"""

import importlib

_EXPORTS = {
    "CreditSystem": "credits",
    "StoreApp": "store",
    "StoreItem": "store",
    "DEFAULT_CATALOG": "store",
    "TopographicalMap": "topomap",
    "MapWaypoint": "topomap",
    "SmartphoneOS": "smartphone",
    "PhoneAppType": "smartphone",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f"firestarter.economy.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'firestarter.economy' has no attribute {name!r}")
