"""DESIGN Section 2.4 - chimney / stack effect, base vacuum and choke detection."""

import math

import numpy as np
import pytest

from firestarter.thermodynamics.airflow_draft import AirflowDraftSolver as A, StructureAnalysis, STRUCTURE_CD
from firestarter.config import GRAVITY, CHOKE_GAP_METERS


def test_stack_effect_formula():
    u = A.calculate_draft_velocity(0.7, 0.5, 600.0, 20.0)
    Ti, Ta = 873.15, 293.15
    assert u == pytest.approx(0.7 * math.sqrt(2 * GRAVITY * 0.5 * (Ti - Ta) / Ti))
    assert A.calculate_draft_velocity(0.7, 0.5, 10.0, 20.0) == 0.0           # no draft when colder than air


def test_taller_hotter_structures_draw_harder():
    assert A.calculate_draft_velocity(0.8, 0.6, 700, 20) > A.calculate_draft_velocity(0.8, 0.3, 700, 20)
    assert A.calculate_draft_velocity(0.8, 0.5, 900, 20) > A.calculate_draft_velocity(0.8, 0.5, 400, 20)


def test_base_vacuum_is_bernoulli():
    p, vac = A.calculate_base_vacuum(101325.0, 2.0)
    assert vac == pytest.approx(0.5 * 1.204 * 4.0, rel=0.01) and p == pytest.approx(101325.0 - vac)


def _horizontal(xs, y=0.03, r=0.03, L=0.4):
    centers = [np.array([x, y, 0.0]) for x in xs]
    axes = [np.array([0.0, 0.0, 1.0])] * len(xs)
    return centers, axes, [L] * len(xs), [r] * len(xs), ["firewood"] * len(xs)


def test_edge_to_edge_logs_are_choked_and_spaced_logs_are_not():
    tight = A.analyze_pieces(*_horizontal([-0.061, 0.0, 0.061]))
    loose = A.analyze_pieces(*_horizontal([-0.12, 0.0, 0.12]))
    assert tight.min_gap < CHOKE_GAP_METERS and tight.choke_fraction > 0.6 and tight.c_d < 0.2
    assert loose.choke_fraction == 0.0 and loose.c_d > 0.35


def test_structure_discharge_coefficients():
    assert STRUCTURE_CD["teepee"] > STRUCTURE_CD["log_cabin"] > STRUCTURE_CD["lean_to"]
    an = A.analyze_pieces(*_horizontal([-0.12, 0.12]), structure_type="teepee")
    assert an.c_d == pytest.approx(STRUCTURE_CD["teepee"])


def test_oxygen_balance_open_vs_choked():
    open_an = StructureAnalysis(height=0.4, diameter=0.4, choke_fraction=0.0, packing_ratio=0.05, intake_area=0.03)
    choked = StructureAnalysis(height=0.15, diameter=0.3, choke_fraction=0.9, packing_ratio=0.5, intake_area=0.003)
    o_open = A.oxygen_balance(open_an, 1.5, 20e3, 0.0005, 1.0)
    o_choked = A.oxygen_balance(choked, 0.2, 20e3, 0.0005, 1.0)
    assert o_open > 0.16 and o_choked < 0.10                                    # < 10 %: flames die (DESIGN 2.4)


def test_legacy_structure_evaluation():
    cd, u, o2, vac = A.evaluate_structure_airflow([(0, 0.03, -0.061), (0, 0.03, 0.0), (0, 0.03, 0.061)], [0.03] * 3, 600, 20)
    assert cd < 0.2 and o2 < 0.12
