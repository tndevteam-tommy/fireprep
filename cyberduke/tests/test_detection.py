"""DetectionSystem: the one place the stealth meter is accumulated and decayed."""

import pytest

from cyberduke.systems.detection import DetectionSystem

DT = 1 / 60


def frame(system, *rates, dt=DT):
    system.begin_frame()
    for r in rates:
        system.report_sighting(r)
    system.end_frame(dt)


def test_unseen_decays_at_exactly_the_configured_rate_regardless_of_observer_count():
    """Regression for B1: decay used to be applied once per guard plus once globally."""
    s = DetectionSystem(decay_rate=30.0)
    s.level = 50.0
    frame(s)  # zero observers reporting
    assert s.level == pytest.approx(50.0 - 30.0 * DT)
    # Observers that do NOT see the player simply don't report; same decay
    s.level = 50.0
    frame(s)
    assert s.level == pytest.approx(50.0 - 30.0 * DT)


def test_sightings_are_additive_and_suppress_decay():
    s = DetectionSystem()
    frame(s, 45.0, 65.0)
    assert s.level == pytest.approx((45.0 + 65.0) * DT)
    assert s.spotted_last_frame


def test_result_is_independent_of_observer_order():
    a, b = DetectionSystem(), DetectionSystem()
    a.level = b.level = 40.0
    frame(a, 45.0, 0.0)
    frame(b, 0.0, 45.0)
    assert a.level == b.level


def test_clamped_to_bounds_and_compromised_flag():
    s = DetectionSystem(max_level=100.0)
    for _ in range(1000):
        frame(s, 65.0)
    assert s.level == 100.0 and s.is_compromised
    for _ in range(1000):
        frame(s)
    assert s.level == 0.0 and not s.is_compromised


def test_reset():
    s = DetectionSystem()
    frame(s, 45.0)
    s.reset()
    assert s.level == 0.0 and not s.spotted_last_frame


def test_fraction():
    s = DetectionSystem(max_level=200.0)
    s.level = 50.0
    assert s.fraction == 0.25
