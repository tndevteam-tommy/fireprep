"""Roof geometry (``rendering/buildings.py``) checked without a GL context: the GL entry points the
module imported are swapped for recorders, so every quad it emits can be inspected."""
from __future__ import annotations

import math

import pytest

from unprepared.rendering import buildings
from unprepared.world.level import BuildingDef


class Recorder:
    """Stand-ins for glBegin / glNormal3f / glVertex3f ...: collects (mode, normal, vertices) primitives."""

    def __init__(self):
        self.primitives = []
        self.normal = None
        self.mode = None

    def begin(self, mode):
        self.mode = mode
        self.current = []

    def end(self):
        self.primitives.append((self.mode, self.current))

    def set_normal(self, x, y, z):
        self.normal = (x, y, z)

    def vertex(self, x, y, z):
        self.current.append((self.normal, (x, y, z)))

    def quads(self):
        """(normal, four corners) for every quad drawn."""
        out = []
        for mode, verts in self.primitives:
            if mode != buildings.GL_QUADS:
                continue
            for i in range(0, len(verts), 4):
                corners = verts[i:i + 4]
                out.append((corners[0][0], [v for _, v in corners]))
        return out


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    for name, fn in (("glBegin", rec.begin), ("glEnd", rec.end), ("glNormal3f", rec.set_normal),
                     ("glVertex3f", rec.vertex), ("glTexCoord2f", lambda u, v: None), ("glEnable", lambda *a: None),
                     ("glDisable", lambda *a: None), ("glBindTexture", lambda *a: None)):
        monkeypatch.setattr(buildings, name, fn)
    return rec


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _unit(v):
    m = math.sqrt(_dot(v, v))
    return tuple(x / m for x in v)


@pytest.mark.parametrize("footprint", [
    ((-26.0, -22.0), (-8.0, -6.0)),        # the inn: wider than deep, the ridge runs along x
    ((30.0, -16.0), (42.0, -6.0)),         # the guard house: likewise
    ((0.0, 0.0), (6.0, 14.0)),             # deeper than wide: the ridge runs along z
])
def test_roof_slopes_are_lit_by_normals_at_right_angles_to_them(recorder, footprint):
    """Each sloped plane's normal is perpendicular to both of its edges, points up and away from
    the ridge. (The long-ridge case used to swap two components, tipping the normals towards the
    horizon: the north slopes of most roofs in town went near-black at noon.)"""
    (x1, z1), (x2, z2) = footprint
    building = BuildingDef("b", (x1, z1), (x2, z2), wall_height=4.0, roof_height=1.6)
    buildings.draw_roof(building, texture_id=1)
    slopes = recorder.quads()
    assert len(slopes) == 2
    centre = ((x1 + x2) / 2.0, building.wall_height, (z1 + z2) / 2.0)
    for normal, corners in slopes:
        n = _unit(normal)
        for a, b in ((corners[0], corners[1]), (corners[1], corners[2])):
            assert _dot(n, _unit(_sub(b, a))) == pytest.approx(0.0, abs=1e-9)
        middle = tuple(sum(c[i] for c in corners) / 4.0 for i in range(3))
        assert n[1] > 0.0                                                     # a roof faces the sky ...
        assert _dot(n, (middle[0] - centre[0], 0.0, middle[2] - centre[2])) > 0.0   # ... and away from the ridge
        # a shallow roof (1.6 rise over a 5-9 unit half-span) is lit mostly from above
        assert n[1] > 0.9


def test_a_shallow_roof_is_lit_almost_as_well_as_flat_ground_at_noon(recorder):
    """Both slopes of the inn's roof catch most of the midday sun (the old normals left the north
    slope at under a tenth of it)."""
    from unprepared.core.daynight import sun_direction
    building = BuildingDef("inn", (-26.0, -22.0), (-8.0, -6.0), wall_height=4.0, roof_height=1.6)
    buildings.draw_roof(building, texture_id=1)
    sun = sun_direction(12.0)
    for normal, _ in recorder.quads():
        assert _dot(_unit(normal), sun) > 0.8
