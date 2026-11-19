"""Headless parity with the prototype for the rules that were kept verbatim:
the player's noise & visibility model and the segment/box ray cast."""
from __future__ import annotations

import random

import pytest

from unprepared.core.math3d import BoundingBox3D, Vector3D
from unprepared.entities import Item, Player

pytestmark = pytest.mark.parity


# NOTE: the prototype's suspicion model was applied per *frame* (an alarm in ten frames);
# the game now uses per-second rates, a field of view and eye-height line of sight, so only
# the noise/visibility formulas and the ray cast are still compared with the prototype.


def test_player_noise_and_visibility_identical(baseline):
    LM = baseline.models
    rng = random.Random(2)
    for _ in range(500):
        mods = [rng.uniform(-0.3, 0.3) for _ in range(rng.randrange(4))]
        lp = LM.Player(LM.Vector3D(0, 0, 0))
        lp.inventory = [LM.Item(f"i{i}", "x", noise_modifier=m) for i, m in enumerate(mods)]
        np_ = Player(Vector3D(0, 0, 0), inventory=[Item(f"i{i}", "x", noise_modifier=m) for i, m in enumerate(mods)])
        lp.is_crouching = np_.is_crouching = rng.random() < 0.5
        lp.has_lit_torch = np_.has_lit_torch = rng.random() < 0.5
        for moving in (False, True):
            for sprint in (False, True):
                assert lp.get_noise_level(moving, sprint) == np_.get_noise_level(moving, sprint)
        for ambient in (0.1, 1.0):
            assert lp.get_visibility(ambient) == np_.get_visibility(ambient)


def test_ray_cast_identical(baseline):
    from unprepared.core.math3d import ray_intersects_box
    L = baseline.models
    rng = random.Random(1234)
    for _ in range(2000):
        a = Vector3D(rng.uniform(-30, 30), rng.uniform(-1, 6), rng.uniform(-30, 30))
        b = Vector3D(rng.uniform(-30, 30), rng.uniform(-1, 6), rng.uniform(-30, 30))
        bmin = Vector3D(rng.uniform(-25, 20), 0, rng.uniform(-25, 20))
        box = BoundingBox3D(bmin, Vector3D(bmin.x + rng.uniform(0.5, 10), 4, bmin.z + rng.uniform(0.5, 10)))
        lbox = L.BoundingBox3D(L.Vector3D(bmin.x, 0, bmin.z), L.Vector3D(box.max_point.x, 4, box.max_point.z))
        assert ray_intersects_box(a, b, box) == L.ray_intersects_box(L.Vector3D(a.x, a.y, a.z), L.Vector3D(b.x, b.y, b.z), lbox)
