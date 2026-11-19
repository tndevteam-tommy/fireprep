"""Animation clips: the weapon-inspection routine must fit the caller's duration (B5)."""

import pytest

from cyberduke.animation import (
    INSPECT_WEAPON_DURATION,
    HumanoidPose,
    inspect_weapon_animation,
    investigate_animation,
    walk_animation,
)

HOLSTER_POS = [-0.28, 0.98, 0.0]
CHEST_POS = [-0.02, 1.28, 0.35]


@pytest.mark.parametrize("duration", [6.0, 12.0, INSPECT_WEAPON_DURATION, 20.0])
def test_regression_b5_weapon_is_holstered_exactly_when_the_routine_ends(duration):
    end = inspect_weapon_animation(duration, duration)
    assert end.prop_pos == pytest.approx(HOLSTER_POS, abs=1e-6)
    assert end.head_pitch == pytest.approx(0.0, abs=1e-6)
    assert end.arm_r[3] == pytest.approx(-15.0, abs=1e-6), "arm back at rest"


@pytest.mark.parametrize("duration", [6.0, 12.0, INSPECT_WEAPON_DURATION, 20.0])
def test_weapon_is_at_the_chest_midway_for_any_duration(duration):
    mid = inspect_weapon_animation(duration * 0.3, duration)  # the inspection phase
    assert mid.prop_pos == pytest.approx(CHEST_POS, abs=1e-6)
    assert mid.head_pitch > 20.0, "head down at the weapon"


def test_default_duration_matches_the_authored_clip():
    for t in (0.0, 3.3, 9.9, 14.0):
        assert inspect_weapon_animation(t) == inspect_weapon_animation(t, INSPECT_WEAPON_DURATION)


def test_original_12s_lean_used_to_cut_the_holster_phase_short():
    """Documents the bug: at t=12 of an unscaled 15 s clip the weapon is still at the chest."""
    unscaled_at_12 = inspect_weapon_animation(12.0, INSPECT_WEAPON_DURATION)
    assert unscaled_at_12.prop_pos == pytest.approx(CHEST_POS, abs=1e-6)
    scaled_at_12 = inspect_weapon_animation(12.0, 12.0)
    assert scaled_at_12.prop_pos == pytest.approx(HOLSTER_POS, abs=1e-6)


def test_walk_cycle_alternates_legs_and_keeps_prop_holstered():
    a = walk_animation(0.5)
    b = walk_animation(0.5 + 3.14159265)
    assert a.leg_l[0] == pytest.approx(b.leg_r[0], abs=1e-4)
    assert a.has_prop and a.prop_pos == HOLSTER_POS
    assert all(p.leg_l[2] >= 0.0 for p in (a, b)), "knees never hyper-extend"


def test_investigate_clip_keeps_head_up_enough_to_see():
    from cyberduke.config import GUARD

    for t in (0.0, 0.7, 1.5, 4.0):
        pose = investigate_animation(t)
        assert isinstance(pose, HumanoidPose)
        assert 6.0 <= pose.head_pitch <= 18.0 < GUARD.engrossed_head_pitch
