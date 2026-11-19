"""The fog and the steam stay where they are as the viewer walks about.

Translucent layers are drawn back to front, and the order changes as the viewer moves;
each fog layer drifts slowly about where the level put it. The drift was keyed to the
layer's place in the *drawing order* - so whenever two layers swapped places in it (every
few steps, walking through the biodome or past the sushi bar's steam) they swapped drifts,
jumping up to 1.8 m from one frame to the next: the fog blinked in and out. The drift is
now keyed to the layer's own place in its sector (``render.sectors.fog_drift``)."""

from __future__ import annotations

import math

from tests.test_render_backends import RecordingContext

from cyberduke.levels import build_facility, build_night_market
from cyberduke.render import sectors
from cyberduke.render.sectors import FOG_TAG, draw_translucent_props, fog_drift


def drawn_fog(level, eye, time, only=None, monkeypatch=None):
    """World position each fog layer is drawn at, by identity (``monkeypatch`` unused: the
    recorder is put in for the one draw and taken out again - left in, every later draw
    went through every earlier recorder too, writing into their results)."""
    out = {}
    ctx = RecordingContext()
    real = sectors.draw_prop

    def record(c, prop, textures, *a, **k):
        if FOG_TAG in prop.tags:
            moves = [call for call in c.calls if call[0] == "translate"]
            dx, dy, dz = moves[-1][1:] if moves else (0.0, 0.0, 0.0)
            out[id(prop)] = (prop.x + dx, prop.y + dy, prop.z + dz)
        return real(c, prop, textures, *a, **k)

    sectors.draw_prop = record
    try:
        secs = [s for s in level.sectors if only is None or s.name in only]
        draw_translucent_props(ctx, secs, {}, time, eye)
    finally:
        sectors.draw_prop = real
    return out


def test_fog_layers_do_not_move_when_the_viewer_does(monkeypatch):
    """Walking up the biodome and past the wok: the draw order changes - the layers do not."""
    dome = [(0.0, 1.6, 55.4 + 0.25 * k) for k in range(72)] + [
        (-11.0 + 0.25 * k, 1.6, 55.6) for k in range(36)
    ]
    dome += [(-6.0 + 0.25 * k, 1.6, 72.5) for k in range(48)]
    alley = [(x, 1.6, -1.0 + 0.2 * k) for x in (-1.9, 1.5) for k in range(215)]
    for level, path in ((build_facility(), dome), (build_night_market(), alley)):
        first = None
        for eye in path:
            where = drawn_fog(level, eye, 100.0, monkeypatch=monkeypatch)
            assert where, "there is fog here"
            if first is None:
                first = where
            assert where == first, "a layer moved because the viewer did"


def test_fog_layers_stay_put_whichever_other_sectors_are_in_view(monkeypatch):
    level = build_facility()
    greenhouse = [s.name for s in level.sectors if any(FOG_TAG in p.tags for p in s.props)]
    alone = drawn_fog(level, (0.0, 1.6, 60.0), 50.0, only=greenhouse, monkeypatch=monkeypatch)
    everything = drawn_fog(level, (0.0, 1.6, 60.0), 50.0, monkeypatch=monkeypatch)
    assert alone == {k: v for k, v in everything.items() if k in alone}


def test_the_drift_is_slow_and_smooth():
    for k in range(12):
        for t in (0.0, 33.3, 500.0):
            a, b = fog_drift(k, t), fog_drift(k, t + 1 / 30)
            assert math.dist(a, b) < 0.01, "no jumps from one frame to the next"
            assert math.hypot(a[0], a[2]) <= math.sqrt(2.0) * 0.9 + 1e-9
