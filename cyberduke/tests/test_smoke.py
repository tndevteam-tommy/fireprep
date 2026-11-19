"""Cigarette smoke: a thin trail off the lit end that stops while the cigarette is drawn
on and starts again after; a puff blown out of the mouth after the drag; the cherry dull
between drags and flaring on one."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from tests.test_render_backends import RecordingContext

from cyberduke.animation import street as clips
from cyberduke.render.items import cherry_glow, draw_item
from cyberduke.render.smoke import SmokeRenderer, cigarette_tip, mouth

DT = 1 / 30


def smoker(t: float = 0.0):
    pose = clips.smoke(t, 0.0)
    pose.hand_r = "cigarette"
    return SimpleNamespace(
        x=1.0, y=0.0, z=2.0, yaw=-90.0, hidden=False, spec=SimpleNamespace(height=1.0), pose=pose
    )


# ----------------------------------------------------------------------------- the clip
def test_a_puff_is_a_drag_then_the_smoke_blown_out():
    d0, d1 = clips.DRAG
    e0, e1 = clips.EXHALE
    before = clips.smoke_state(0.3)
    during = clips.smoke_state((d0 + d1) / 2.0)
    after = clips.smoke_state((e0 + e1) / 2.0 - 0.2)
    later = clips.smoke_state(5.5)
    assert before == (0.0, False, 0.0), "smouldering by the hip"
    assert during[0] > 0.95 and during[1] and during[2] == 0.0, "drawn on: the cherry flares"
    assert not after[1] and after[2] > 0.5, "then blown out"
    assert later[0] < 0.05 and not later[1] and later[2] == 0.0
    assert d1 <= e0, "the smoke comes out after the drag, not during it"


def test_the_cherry_is_dull_between_drags_and_bright_on_one():
    dull, hot = cherry_glow(0.0), cherry_glow(1.0)
    assert sum(hot) > 4.0 * sum(dull) and sum(dull) > 0.0
    emitted = []

    class Ctx(RecordingContext):
        def emission(self, rgb):
            emitted.append(tuple(rgb) if rgb is not None else None)
            super().emission(rgb)

    ctx = Ctx()
    draw_item(ctx, "cigarette", {}, ember=0.0)
    draw_item(ctx, "cigarette", {}, ember=1.0)
    glows = [e for e in emitted if e and sum(e) > 0.0]
    assert glows[0] == pytest.approx(dull) and glows[-1] == pytest.approx(hot)


# ----------------------------------------------------------------------------- the smoke
def test_the_trail_stops_while_the_cigarette_is_drawn_on_and_a_puff_follows():
    smoke = SmokeRenderer(seed=1)
    npc = smoker()
    born: list[tuple[float, str, bool, float]] = []  # (t, kind, inhaling, exhale)
    seen = set()
    t = 0.0
    while t < clips.SMOKE_CYCLE * 2:
        npc.pose = clips.smoke(t, 0.0)
        npc.pose.hand_r = "cigarette"
        smoke.update([npc], DT)
        for p in smoke.particles:
            if id(p) not in seen:
                seen.add(id(p))
                born.append((t, p.kind, npc.pose.inhaling, npc.pose.exhale))
        t += DT
    trail = [b for b in born if b[1] == "trail"]
    puffs = [b for b in born if b[1] == "puff"]
    assert trail and puffs
    assert not [b for b in trail if b[2]], "no smoke off the tip while it is drawn on"
    assert all(b[3] > 0.0 for b in puffs), "a puff only while breathing out"
    d0, d1 = clips.DRAG
    resumed = [b for b in trail if d1 < (b[0] % clips.SMOKE_CYCLE) < clips.SMOKE_CYCLE]
    assert resumed, "and the trail starts again after the drag"


def test_the_trail_rises_from_the_tip_and_the_puff_comes_out_of_the_mouth():
    smoke = SmokeRenderer(seed=2)
    npc = smoker(0.2)  # smouldering by the hip
    tip = cigarette_tip(npc)
    assert tip is not None and 0.8 < tip[1] < 1.4, "the lit end, at hip height"
    smoke.update([npc], 0.5)
    first = smoke.particles[0]
    assert math.dist((first.x, first.y, first.z), tip) < 0.1
    y0 = first.y
    for _ in range(30):
        smoke.update([], DT)
    assert first.y > y0 + 0.05, "it rises"
    # a puff: out of the mouth and away from the face
    npc = smoker(sum(clips.EXHALE) / 2.0)
    at, forward = mouth(npc)
    assert 1.6 < at[1] < 2.0
    smoke = SmokeRenderer(seed=3)
    smoke.update([npc], 0.3)
    puffs = [p for p in smoke.particles if p.kind == "puff"]
    assert puffs and all(math.dist((p.x, p.y, p.z), at) < 0.3 for p in puffs)
    assert all(p.vx * forward[0] + p.vz * forward[2] > 0.0 for p in puffs), "blown forwards"


def test_smoke_is_drawn_as_soft_discs_facing_the_viewer_and_hangs_still_when_paused():
    smoke = SmokeRenderer(seed=4)
    npc = smoker(0.2)
    smoke.update([npc], 1.0)
    positions = [(p.x, p.y, p.z) for p in smoke.particles]
    smoke.update([npc], 0.0)
    assert [(p.x, p.y, p.z) for p in smoke.particles] == positions, "paused: nothing moves"
    ctx = RecordingContext()
    smoke.draw(ctx, (0.0, 1.6, 0.0))
    discs = [c for c in ctx.calls if c[0] == "soft_disc"]
    assert len(discs) == len(smoke.particles) + len(smoke.embers), (
        "every puff, and the cherry's halo"
    )
    assert smoke.embers and smoke.embers[0][1] == pytest.approx(npc.pose.ember)
