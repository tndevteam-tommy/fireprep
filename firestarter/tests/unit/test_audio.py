"""
DESIGN Section 9 - procedural ASMR audio: every sound is synthesised (no samples), loops are
seamless, each sound has the spectral character of the real thing, and the live mixer follows
the simulation.
"""

import io
import math
import os
import wave

import numpy as np
import pytest

from firestarter.audio import procedural_audio as pa
from firestarter.audio.procedural_audio import ProceduralAudioEngine, LOOPS, SHOTS, SR


@pytest.fixture(scope="module")
def sounds():
    return {k: fn() for k, fn in ProceduralAudioEngine.all_sounds().items()}


def band_fraction(x, lo, hi):
    m = x.mean(axis=1) if x.ndim == 2 else x
    spec = np.abs(np.fft.rfft(m * np.hanning(len(m)))) ** 2
    f = np.fft.rfftfreq(len(m), 1.0 / SR)
    return float(spec[(f >= lo) & (f < hi)].sum() / max(1e-20, spec.sum()))


def centroid(x):
    m = x.mean(axis=1) if x.ndim == 2 else x
    spec = np.abs(np.fft.rfft(m)) ** 2
    f = np.fft.rfftfreq(len(m), 1.0 / SR)
    return float((spec * f).sum() / max(1e-20, spec.sum()))


def test_sound_bank_is_complete(sounds):
    assert len(sounds) >= 55
    for needed in ("loop_fire_roar", "loop_fire_crackle", "loop_steam_hiss", "loop_embers", "loop_wind", "loop_rain",
                   "loop_rain_tarp", "loop_creek", "loop_forest_day", "loop_night", "lighter_strike", "friction_squeak_0",
                   "breath_med", "whittle_0", "chop_0", "tree_fall", "resin_pop_0", "step_snow_0", "purchase"):
        assert needed in sounds


def test_every_sound_is_clean_stereo(sounds):
    for name, x in sounds.items():
        assert x.ndim == 2 and x.shape[1] == 2, name
        assert np.isfinite(x).all(), name
        peak = float(np.max(np.abs(x)))
        assert 0.05 < peak <= 1.0, (name, peak)                           # audible, never clipping
        rms = float(np.sqrt(np.mean(x ** 2)))
        assert rms > 0.004, (name, rms)
        assert abs(float(np.mean(x))) < 0.02, name                         # no DC offset
        assert 0.03 <= len(x) / SR < 20.0, name


def test_loops_are_seamless(sounds):
    """The wrap-around sample step must look like any other step (no click at the loop point)."""
    for name in LOOPS:
        x = sounds[name].mean(axis=1)
        d = np.abs(np.diff(x))
        wrap = abs(x[0] - x[-1])
        assert wrap <= np.percentile(d, 99.9) + 1e-3, (name, wrap, np.percentile(d, 99.9))
        # loudness continuity across the seam (first vs last 100 ms)
        n = SR // 10
        a, b = np.sqrt(np.mean(x[:n] ** 2)), np.sqrt(np.mean(x[-n:] ** 2))
        assert 0.33 < (a + 1e-6) / (b + 1e-6) < 3.0, (name, a, b)


def test_spectral_character(sounds):
    # a fire's roar is low-frequency turbulence; steam hiss and rain are bright broadband noise
    assert band_fraction(sounds["loop_fire_roar"], 0, 500) > 0.6
    assert centroid(sounds["loop_steam_hiss"]) > 2500
    assert centroid(sounds["loop_steam_hiss"]) > 4 * centroid(sounds["loop_fire_roar"])
    assert centroid(sounds["loop_rain"]) > 1200
    # snow muffles wind
    assert centroid(sounds["loop_wind_snow"]) < centroid(sounds["loop_wind"])
    # harder breaths are louder
    rms = lambda k: float(np.sqrt(np.mean(sounds[k] ** 2)))
    assert rms("breath_soft") < rms("breath_med") < rms("breath_hard")


def test_one_shots_are_transient(sounds):
    for name in ("resin_pop_0", "chop_0", "wood_knock_0", "lighter_strike", "ui_click"):
        x = np.abs(sounds[name]).mean(axis=1)
        env = np.convolve(x, np.ones(256) / 256, mode="same")
        i_peak = int(np.argmax(env))
        assert i_peak < 0.5 * len(env), name                              # attack early
        assert env[-512:].mean() < 0.2 * env.max(), name                   # decays away


def test_wav_encoding(sounds):
    data = pa.to_wav_bytes(sounds["chop_0"])
    with wave.open(io.BytesIO(data)) as wf:
        assert wf.getnchannels() == 2 and wf.getsampwidth() == 2 and wf.getframerate() == 44100
        assert wf.getnframes() == len(sounds["chop_0"])


def test_asset_generation_and_version_stamp(tmp_path):
    d = str(tmp_path / "audio")
    assert not ProceduralAudioEngine.assets_up_to_date(d)
    n = ProceduralAudioEngine.generate_and_save_all_assets(d)
    assert n == len(LOOPS) + len(SHOTS)
    assert ProceduralAudioEngine.assets_up_to_date(d)
    with open(os.path.join(d, ".version"), "w") as f:
        f.write("0")
    assert not ProceduralAudioEngine.assets_up_to_date(d)


# ------------------------------------------------------------------------ live mixer
@pytest.fixture(scope="module")
def mixer_assets(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("mix_audio"))
    ProceduralAudioEngine.generate_and_save_all_assets(d)
    return d


@pytest.fixture
def foley(mixer_assets):
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    from firestarter.audio.asmr_foley import ASMRFoleyManager
    f = ASMRFoleyManager(asset_dir=mixer_assets)
    if not f.is_pygame_audio_available:
        pytest.skip("pygame mixer unavailable")
    yield f
    f.shutdown()


def test_mixer_follows_the_fire_and_weather(state, foley):
    from firestarter.automation.staging import build_log_fire
    w = state.world
    for _ in range(60):
        foley.update_world(w, 1 / 30)
    assert foley.levels["loop_fire_roar"] < 0.01
    build_log_fire(w)
    for _ in range(300):
        state.step(1 / 30, move=False)
        foley.update_world(w, 1 / 30)
    near = foley.levels["loop_fire_roar"]
    assert near > 0.1 and foley.levels["loop_fire_crackle"] > 0.1
    assert any(p.startswith("resin_pop") for p in foley.played)
    # walk away: inverse-distance falloff
    w.player.position[2] = 25.0
    for _ in range(30):
        foley.update_world(w, 1 / 30)
    assert foley.levels["loop_fire_roar"] < 0.35 * near
    # rain bed
    w.weather.force_precipitation(1.0)
    for _ in range(30):
        foley.update_world(w, 1 / 30)
    assert foley.levels["loop_rain"] > 0.5


def test_fire_is_panned_to_the_side_it_is_on(state, foley):
    w = state.world
    p = w.player
    p.position[:] = [0.0, 1.7, 3.0]
    p.yaw = 0.0            # looking +X: the pit (at -Z relative) is on the left
    pan, gain, dist = foley._pan_and_gain(p.eye, p.yaw, np.array([0.0, 0.0, 0.0]))
    assert pan < -0.5 and 2.5 < dist < 3.6
    p.yaw = 180.0
    pan2, _, _ = foley._pan_and_gain(p.eye, p.yaw, np.array([0.0, 0.0, 0.0]))
    assert pan2 > 0.5


def test_game_events_trigger_one_shots(state, foley):
    w = state.world
    evs = [("lighter", None), ("chop", np.zeros(3)), ("breath", {"velocity": 2.5}), ("purchase", "x"), ("page_turn", None),
           ("tree_fall", np.zeros(3)), ("pickup", None), ("tarp", np.zeros(3))]
    foley.handle_events(evs, w)
    for s in ("lighter_strike", "chop", "breath_med", "purchase", "page_turn", "tree_fall", "rustle", "tarp"):
        assert s in foley.played
