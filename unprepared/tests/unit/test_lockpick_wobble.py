"""Lockpicking wobble: an unsteady pick wanders, so timing [SPACE] matters."""

from unprepared.config import LockpickSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import Door, Player
from unprepared.systems import LockpickMinigame, LockpickOutcome


def test_steady_hands_do_not_wobble():
    g = LockpickMinigame(LockpickSettings())
    g.begin(Door("d", Vector3D(0, 0, 0)), sweet_spot=0.0, wobble=0.0)
    g.update(0.37)
    assert g.wobble == 0.0 and g.effective_angle == 0.0 and g.is_aligned


def test_wobble_moves_the_pick_over_time_and_timing_decides():
    g = LockpickMinigame(LockpickSettings(success_tolerance_degrees=5.0))
    g.begin(Door("d", Vector3D(0, 0, 0)), sweet_spot=0.0, tolerance=5.0, wobble=14.0)
    seen = set()
    aligned_moments, total = 0, 0
    for _ in range(300):
        g.update(0.01)
        seen.add(round(g.wobble, 2))
        total += 1
        aligned_moments += g.is_aligned
    assert len(seen) > 50                                  # it really wanders ...
    assert max(abs(w) for w in seen) <= 14.0 + 1e-6        # ... within the amplitude
    assert 0 < aligned_moments < total                     # sometimes on the spot, often not


def test_space_uses_the_wobbled_angle():
    player = Player(Vector3D(0, 0, 0))
    door = Door("d", Vector3D(0, 0, 0), is_locked=True)
    g = LockpickMinigame(LockpickSettings(consume_pick_on_snap=False))
    g.begin(door, sweet_spot=0.0, tolerance=3.0, wobble=20.0)
    # find a moment where the wobble carries the pick off the spot
    for _ in range(200):
        g.update(0.013)
        if abs(g.wobble) > 6.0:
            break
    assert not g.is_aligned
    assert g.attempt(player) is LockpickOutcome.SNAPPED and door.is_locked
    g.begin(door, sweet_spot=0.0, tolerance=3.0, wobble=20.0)       # back at t=0 the pick is steady on 0
    assert g.is_aligned and g.attempt(player) is LockpickOutcome.SUCCESS
