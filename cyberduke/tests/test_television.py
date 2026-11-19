"""Television: every home has a set (``ai.alley.TvSet``) its owner switches on from the sofa
and flicks through with the remote - the snow of the dead channels between the stations -
until the news, a cartoon or the music channel holds them (``ai.alley.WatchTv``); the
renderer shows what is on, frame by frame (``render.television``, the channels' frames in
``assets.homes``): snow that crawls, a moment of it as the set retunes, dark glass when it
is off. And the window views follow the time of day."""

from __future__ import annotations

import random

import numpy as np
import pytest
from tests.test_render_backends import RecordingContext

from cyberduke.ai.alley import CHANNEL_LINEUP, WATCHABLE, TvSet, Wait, WatchTv
from cyberduke.animation import street as clips
from cyberduke.assets.homes import TV_CHANNELS, tv_frame
from cyberduke.assets.textures import available_textures, generate_texture
from cyberduke.levels import build_night_market
from cyberduke.render.sectors import is_static
from cyberduke.render.television import (
    RETUNE,
    draw_screens,
    is_screen,
    screen_of,
    tv_picture,
    view_picture,
)
from cyberduke.simulation import InputState, Simulation

DT = 1 / 30
LEVEL = build_night_market()


def run(sim, seconds, until=None) -> bool:
    for _ in range(int(seconds / DT)):
        sim.player.x, sim.player.z = 0.0, -1.2
        sim.update(DT, InputState())
        if until is not None and until():
            return True
    return until is None


# ----------------------------------------------------------------------------- the pictures
def test_every_channel_is_a_loop_of_frames_and_the_lineup_has_stations_between_the_snow():
    names = set(available_textures())
    for channel, (frames, fps) in TV_CHANNELS.items():
        assert frames >= 6 and fps > 1.0, channel
        assert {f"tv_{channel}_{k}" for k in range(frames)} <= names
        assert tv_frame(channel, frames + 1) == tv_frame(channel, 1), "it loops"
    assert "tv_off" in names
    assert set(CHANNEL_LINEUP) <= set(TV_CHANNELS)
    assert "static" in CHANNEL_LINEUP and set(CHANNEL_LINEUP) >= WATCHABLE
    assert CHANNEL_LINEUP.count("static") >= 2, "dead channels between the stations"


def test_a_dead_channel_moves_like_snow():
    """Not one still picture of fuzz: every frame's snow is fresh, and the dark bar rolls
    down the picture from one frame to the next."""
    frames = [generate_texture(tv_frame("static", k)).astype(float) for k in range(8)]
    for a, b in zip(frames, frames[1:], strict=False):
        assert np.abs(a - b).mean() > 20.0, "fresh snow every frame"
    rows = np.array([f.mean(axis=(1, 2)) for f in frames])
    rows -= rows.mean(axis=0)  # (what every frame has in common: the tube's dark corners)
    kernel = np.ones(15) / 15.0
    smooth = [np.convolve(np.concatenate([r[-7:], r, r[:7]]), kernel, "valid") for r in rows]
    darkest = [int(np.argmin(r)) for r in smooth]
    assert len(set(darkest)) >= 6, f"the bar rolls: {darkest}"


def test_the_stations_are_pictures_that_move_too():
    for channel in ("news", "cartoon", "music", "ads"):
        a = generate_texture(tv_frame(channel, 0)).astype(float)
        b = generate_texture(tv_frame(channel, 3)).astype(float)
        assert np.abs(a - b).mean() > 0.5, f"{channel}: something changes from frame to frame"
        assert a.std() > 25.0, f"{channel}: a picture, not a flat colour"


def test_what_a_set_shows_off_retuning_and_on_a_channel():
    tv = TvSet()
    assert tv_picture(tv, 10.0) == "tv_off" and tv_picture(None, 10.0) == "tv_off"
    tv.switch(True, 10.0)
    assert tv.showing == CHANNEL_LINEUP[0]
    assert tv_picture(tv, 10.1).startswith("tv_static_"), "a moment of snow as it warms up"
    shown = {tv_picture(tv, 10.0 + RETUNE + 0.01 + k * 0.2) for k in range(10)}
    assert all(s.startswith(f"tv_{tv.showing}_") for s in shown) and len(shown) > 1, "it moves"
    tv.next_channel(20.0)
    assert tv_picture(tv, 20.05).startswith("tv_static_"), "retuning"
    at = [tv_picture(tv, 20.0 + RETUNE + 0.01 + k / 18.0) for k in range(5)]
    assert len(set(at)) >= 4, "the snow crawls at 18 frames a second"
    tv.switch(False, 30.0)
    assert tv_picture(tv, 30.5) == "tv_off"


def test_the_view_out_of_a_window_follows_the_day():
    assert view_picture("city", 0.0) == "view_city_night"
    assert view_picture("city", 1.0) == "view_city_day"
    assert view_picture("alley", 0.2) == "view_alley_night"


# ----------------------------------------------------------------------------- the sets
def test_every_home_has_one_television_facing_its_sofa():
    screens = [p for p in LEVEL.props if is_screen(p) and screen_of(p)[0] == "tv"]
    # (and one up on the wall of the Neon Moth, for its customers: tests/test_nightlife.py)
    assert [screen_of(p)[1] for p in screens].count("bar") == 1
    screens = [p for p in screens if screen_of(p)[1] != "bar"]
    homes = {a.area_id: a for a in LEVEL.private_areas}
    assert sorted(screen_of(p)[1] for p in screens) == sorted(homes), "one set in every home"
    sofas = {p.name: p for p in LEVEL.places if p.name.endswith(".sofa")}
    for p in screens:
        area = homes[screen_of(p)[1]]
        assert area.contains(p.x, p.z), "in its own home"
        seat = next(s for s in sofas.values() if area.contains(s.x, s.z))
        fp = p.footprint()
        thin = "x" if fp.max_x - fp.min_x < fp.max_z - fp.min_z else "z"
        # the screen is thin along the line to the sofa (it faces it)
        d = (abs(seat.x - p.x), abs(seat.z - p.z))
        assert (d[0] > d[1]) == (thin == "x"), (p.label, screen_of(p))
        assert not is_static(p), "drawn live, not baked into the sector"


def test_windows_look_out_on_the_city_or_the_alley():
    views = [p for p in LEVEL.props if is_screen(p) and screen_of(p)[0] == "view"]
    assert len(views) == len(LEVEL.private_areas), "a window in every home"
    assert {screen_of(p)[1] for p in views} == {"city", "alley"}


def test_the_renderer_draws_each_set_with_the_frame_on_it():
    sim = Simulation(LEVEL)
    textures = {name: k + 1 for k, name in enumerate(available_textures())}
    sim.life.tv("flat32").switch(True, 0.0)
    sim.life.tv("flat32").channel = CHANNEL_LINEUP.index("news")
    ctx = RecordingContext()
    draw_screens(ctx, LEVEL.sectors, textures, sim.life.tvs, now=5.0, daylight=0.0)
    drawn = [c[1] for c in ctx.calls if c[0] == "cube"]
    names = {v: k for k, v in textures.items()}
    shown = [names[t] for t in drawn if t in names]
    assert tv_frame("news", int(5.0 * TV_CHANNELS["news"][1])) in shown
    # the rest are off (and the bar's, which nobody has switched on)
    assert shown.count("tv_off") == len(LEVEL.private_areas) - 1 + 1, "the rest are off"
    assert shown.count("view_city_night") >= 6


# ----------------------------------------------------------------------------- watching
class _Viewer:
    """Just enough of a person for an act: where they are, their pose, a dice."""

    def __init__(self) -> None:
        self.rng = random.Random(7)
        self.seed = 3.0
        self.pose = clips.idle(0.0)


def test_watching_tv_the_remote_flicks_through_the_snow_to_a_station():
    tv = TvSet()
    now = [0.0]
    act = WatchTv(tv, lambda: now[0], 0.5, lambda: False, timeout=60.0)
    me = _Viewer()
    act.begin(me)
    seen, labels, raised = [], set(), 0.0
    for _ in range(int(62.0 / DT)):
        now[0] += DT
        if act.step(me, DT):
            break
        seen.append(tv.showing)
        labels.add(act.label)
        raised = max(raised, act.arm())
        assert me.pose.leg_l[0] < -50.0, "sitting"
    assert act.result is False and tv.on, "the time ran out with the set on"
    changes = [b for a, b in zip(seen, seen[1:], strict=False) if a != b]
    assert len(changes) >= 4, "flicking through the channels"
    assert "static" in changes and set(changes) & WATCHABLE, "past the snow, onto stations"
    assert "flicking through the channels" in labels
    assert labels & {"watching the news", "watching cartoons", "watching the music channel"}
    assert raised > 0.99, "the remote comes up, pointed at the set"


def test_somebody_at_home_switches_the_set_on_and_off_again_when_they_go_out():
    """Nix, home for a while: sat on the sofa, the remote in his hand, the set comes on;
    when he goes out again it goes off."""
    from cyberduke.ai.alley import stay_home

    sim = Simulation(LEVEL)
    nix = sim.npc_with_role("smoker")
    home = sim.life.homes["smoker"]
    life = sim.life
    tv = life.tv(home.area_id)

    def home_till_the_tv_is_on(me):
        yield from stay_home(me, life, lambda: tv.on)  # (and out of the door again)
        while True:
            yield Wait(lambda: False, lambda t: clips.idle(t, me.seed), label="done")

    nix.storyteller.story, nix.storyteller.act = home_till_the_tv_is_on(nix), None
    nix.last_result = None
    remote = []
    assert run(sim, 300.0, until=lambda: tv.on and not remote.append(nix.hands["r"]))
    assert remote == ["remote"], "switched on with the remote"
    assert life.inside(home.area_id, nix.x, nix.z) and nix.activity.startswith(
        ("switching", "watching", "flicking")
    )
    assert run(sim, 120.0, until=lambda: nix.activity == "done")
    assert not tv.on, "off on the way out"
    assert nix.hands["r"] is None, "the remote left on the cushion"
    assert not life.inside(home.area_id, nix.x, nix.z)


@pytest.mark.parametrize("key", ["static", "news"])
def test_a_tvset_round_trips_through_a_save(key):
    from cyberduke.systems import savegame

    sim = Simulation(LEVEL)
    tv = sim.life.tv("flat35")
    tv.switch(True, sim.life.time)
    tv.channel = CHANNEL_LINEUP.index(key)
    other = Simulation(LEVEL)
    savegame.restore(other, savegame.snapshot(sim))
    again = other.life.tv("flat35")
    assert again.on and again.showing == key
