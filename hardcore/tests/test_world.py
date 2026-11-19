"""Procedural world: every face has holds and every hold ladder is climbable."""
import numpy as np
from harness import *  # noqa
from world.obstacle import VERTEX_FLOATS


def _faces(obstacle):
    return [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0)]


def test_crevasses_exist_on_all_four_faces():
    """Regression: the 18 m traverse wall only had crevasses on its 0.8 m end caps."""
    ls, obs = build_playground()
    for o in obs:
        crev = [r for r in o.rails if r.is_crevasse]
        if not crev or o.crevasse_faces is not None:
            continue
        for n in _faces(o):
            on_face = [r for r in crev if np.allclose(r.normal, n)]
            assert on_face, f"obstacle {o.bounds.min} {o.bounds.size} lacks crevasses on face {n}"


def test_hold_ladder_gaps_are_climbable_on_every_face():
    ls, obs = build_playground()
    for o in obs:
        if not o.rails or not o.climb_target or o.crevasse_faces is not None:
            continue
        base = float(o.bounds.min[1])
        top = float(o.bounds.max[1])
        if top - base <= config.STAND_REACH:
            continue   # top reachable from the ground
        for n in _faces(o):
            ys = sorted(r.height for r in o.rails if r.is_crevasse and np.allclose(r.normal, n))
            assert ys, f"tall obstacle face {n} without crevasses"
            assert ys[0] - base <= config.STAND_REACH, "first hold out of standing reach"
            chain = [base] + ys + [top]
            for lo, hi in zip(chain[:-1], chain[1:]):
                if lo == base:
                    continue
                assert hi - lo <= config.CLIMB_HOLD_GAP_MAX, f"hold gap {lo}->{hi} too large for a lip stand-up"
            assert top - ys[-1] >= 0.25, "crevasse too close to the top to be useful"


def test_crevasse_heights_generator_rules():
    assert Obstacle.crevasse_heights(2.2) == []
    assert Obstacle.crevasse_heights(3.2) == [2.0]
    assert Obstacle.crevasse_heights(5.4) == [2.0, 3.8]
    hs = Obstacle.crevasse_heights(8.0)
    assert hs[:2] == [2.0, 3.8] and len(hs) >= 4
    for h in (3.0, 4.2, 5.0, 6.0, 7.5, 9.9, 14.0):
        ys = Obstacle.crevasse_heights(h)
        chain = ys + [h]
        assert ys[0] <= config.STAND_REACH
        for lo, hi in zip(chain[:-1], chain[1:]):
            assert 0.3 <= hi - lo <= config.CLIMB_HOLD_GAP_MAX


def test_rail_tangent_is_hanging_players_right_and_owner_set():
    ls, obs = build_playground()
    up = np.array([0.0, 1.0, 0.0])
    for r in ls.rails:
        assert r.owner is not None
        right = np.cross(-r.normal, up)
        assert float(np.dot(right, r.tangent)) > 0.99


def test_rail_end_margins_keep_both_hands_on_the_rail():
    ls, obs = build_playground()
    for r in ls.rails:
        m = r.s_margin()
        if r.length > 2 * config.RAIL_END_MARGIN:
            assert abs(m * r.length - config.RAIL_END_MARGIN) < 1e-5
        else:
            assert m == 0.5


def test_chasm_gap_is_a_human_cat_leap():
    ls, obs = build_playground()
    towers = [o for o in obs if abs(o.bounds.size[1] - 6.0) < 1e-3 and abs(o.bounds.size[0] - 4.0) < 1e-3]
    assert len(towers) == 2
    a, b = sorted(towers, key=lambda o: o.bounds.min[0])
    gap = float(b.bounds.min[0] - a.bounds.max[0])
    assert 2.5 <= gap <= 3.5, f"chasm gap {gap} m is not a realistic wall-to-wall cat leap"


def test_playground_mesh_batches_consistently():
    ls = LedgeSystem()
    obstacles, verts, idx = LevelGenerator.build_playground(ls)
    assert verts.shape[1] == VERTEX_FLOATS
    assert idx.max() < len(verts)
    owners = set(id(o) for o in obstacles)
    assert all(id(r.owner) in owners for r in ls.rails)
    assert all(r.length >= config.MIN_RAIL_LENGTH - 1e-6 for r in ls.rails)
