"""Distant "streaking" (z-fighting): two faces drawn in the same plane, facing the same way, that
the depth buffer cannot order - at a distance they flicker into stripes.

The causes found and removed: walls authored overlapping at the corners (now made disjoint),
lintels over gaps a wall already filled, the tower's column flush with its platform, fascia
boards flush with the gable ends, the layers of a figure's clothing (a hood over a cap, a
cloak's clasp) sharing planes, footwear over the hose, the well's crank. These tests keep every
one of them from coming back: no two boxes of the static world, and no two boxes of any
figure in any gear, share a face plane over any area. (Faces flat on the ground at the foot of a
figure are excluded: they face down into the ground and are never seen.)
"""
from __future__ import annotations

import pytest

from unprepared.config import GameSettings
from unprepared.core.math3d import BoundingBox3D
from unprepared.entities import Item
from unprepared.entities.equipment import GEAR_SLOTS, Figure
from unprepared.rendering import gear
from unprepared.world import World, get_level
from unprepared.world.geometry import coplanar_overlaps, disjoint_boxes, subtract_box
from unprepared.core.math3d import Vector3D


@pytest.fixture(scope="module")
def w():
    return World(get_level("town"), GameSettings(echo_messages_to_stdout=False))


def _visible(pairs):
    """Drop pairs of faces lying flat on y = 0 facing down (under a figure's feet)."""
    return [(a, b, axis, d) for a, b, axis, d in pairs if not (axis == "y" and d < 0 and abs(a.min_point.y) < 1e-9)]


class TestTheStaticWorld:
    def test_no_two_walls_or_lintels_share_a_face(self, w):
        pairs = coplanar_overlaps([*w.walls, *w.lintels])
        assert pairs == [], [(a.min_point, a.max_point, b.min_point, b.max_point) for a, b, *_ in pairs[:5]]

    def test_walls_are_disjoint(self, w):
        walls = w.walls
        for i, a in enumerate(walls):
            for b in walls[i + 1:]:
                overlap = all(min(getattr(a.max_point, k), getattr(b.max_point, k)) - max(getattr(a.min_point, k),
                                                                                         getattr(b.min_point, k)) > 1e-6
                              for k in "xyz")
                assert not overlap, (a, b)

    def test_disjoint_boxes_keeps_the_union(self):
        boxes = [BoundingBox3D(Vector3D(0, 0, 0), Vector3D(4, 2, 1)), BoundingBox3D(Vector3D(3, 0, 0), Vector3D(4, 2, 5)),
                 BoundingBox3D(Vector3D(-1, 1, -1), Vector3D(1, 3, 1))]
        pieces = disjoint_boxes(boxes, lambda original, mn, mx: BoundingBox3D(mn, mx))
        volume = lambda b: (b.max_point.x - b.min_point.x) * (b.max_point.y - b.min_point.y) * (b.max_point.z - b.min_point.z)  # noqa
        assert sum(volume(p) for p in pieces) == pytest.approx(8 + 10 + 8 - 2 - 1)
        assert subtract_box(boxes[0], BoundingBox3D(Vector3D(10, 10, 10), Vector3D(11, 11, 11))) == \
            [(boxes[0].min_point, boxes[0].max_point)]


class TestFigures:
    @pytest.fixture
    def recorder(self, monkeypatch):
        boxes = []
        monkeypatch.setattr(gear, "draw_textured_box", lambda mn, mx, tex, tile=None: boxes.append(BoundingBox3D(mn.copy(), mx.copy())))

        class AnyTexture(dict):
            def __getitem__(self, key):
                return 0

        def figure(equipment):
            boxes.clear()
            gear.draw_figure(Figure(), equipment, AnyTexture(), lambda **kw: None)
            return list(boxes)
        return figure

    def test_every_piece_of_gear_alone(self, recorder):
        for kind, slot in GEAR_SLOTS.items():
            pairs = _visible(coplanar_overlaps(recorder({slot: Item("x", "x", gear=kind)})))
            assert pairs == [], kind

    def test_everybody_in_town_as_dressed(self, recorder, w):
        for npc in w.npcs:
            assert _visible(coplanar_overlaps(recorder(npc.equipment))) == [], npc.name

    def test_any_two_pieces_worn_together(self, recorder):
        """Every pair of gear kinds in different slots (a hood with a cloak, a sling with tall
        boots, a sword under a cloak): the pieces of one never share a plane with the other's."""
        from itertools import combinations
        by_slot = {}
        for kind, slot in GEAR_SLOTS.items():
            by_slot.setdefault(slot, []).append(kind)
        for (slot_a, kinds_a), (slot_b, kinds_b) in combinations(by_slot.items(), 2):
            for a in kinds_a:
                for b in kinds_b:
                    outfit = {slot_a: Item("a", "a", gear=a), slot_b: Item("b", "b", gear=b)}
                    assert _visible(coplanar_overlaps(recorder(outfit))) == [], (a, b)

    def test_a_full_outfit_of_each_kind(self, recorder):
        by_slot = {}
        for kind, slot in GEAR_SLOTS.items():
            by_slot.setdefault(slot, []).append(kind)
        longest = max(len(k) for k in by_slot.values())
        for i in range(longest):
            outfit = {slot: Item("x", "x", gear=kinds[i % len(kinds)]) for slot, kinds in by_slot.items()}
            assert _visible(coplanar_overlaps(recorder(outfit))) == [], {s.value: it.gear for s, it in outfit.items()}
