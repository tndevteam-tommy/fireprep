"""
Firestarter Thermodynamics Module

Public names are resolved lazily (PEP 562) so sub-modules can import each other without
circular-import problems.
"""

import importlib

_EXPORTS = {
    "WoodSpeciesType": "wood_properties",
    "WoodProperties": "wood_properties",
    "get_wood_properties": "wood_properties",
    "CombustionBody": "combustion_body",
    "CombustionStage": "combustion_body",
    "MoistureDiffusionEngine": "moisture_diffusion",
    "AirflowDraftSolver": "airflow_draft",
    "ThermodynamicGrid": "thermal_grid",
    "ThermalConvectionSolver": "thermal_grid",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        mod = importlib.import_module(f"firestarter.thermodynamics.{_EXPORTS[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'firestarter.thermodynamics' has no attribute {name!r}")
