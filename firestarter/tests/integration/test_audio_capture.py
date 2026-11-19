"""
End-to-end audio: the live mixer really outputs sound through SDL.

SDL's ``disk`` audio driver writes the final mixed stream to a raw PCM file, so we can listen
to what a player would hear: a quiet summer evening, then the same place with a roaring fire.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = r'''
import os, sys, time, types, json
ROOT = sys.argv[1]; OUT = sys.argv[2]; MODE = sys.argv[3]
sys.path.insert(0, ROOT)
p = types.ModuleType("firestarter"); p.__path__ = [ROOT]; sys.modules["firestarter"] = p
import numpy as np
from firestarter.audio.procedural_audio import ProceduralAudioEngine
from firestarter.audio.asmr_foley import ASMRFoleyManager
from firestarter.game import GameState, AUDIO_DIR
from firestarter.automation.staging import build_log_fire
if not ProceduralAudioEngine.assets_up_to_date(AUDIO_DIR):
    ProceduralAudioEngine.generate_and_save_all_assets(AUDIO_DIR)
st = GameState(tutorial=False)
w = st.world
if MODE in ("fire", "pause"):
    build_log_fire(w)
    for _ in range(600):
        st.step(1 / 30, move=False)
import pygame
foley = ASMRFoleyManager(asset_dir=AUDIO_DIR)
assert foley.is_pygame_audio_available
t0 = time.time()
frames = 0
marks = {}
while time.time() - t0 < 4.0:
    paused = MODE == "pause" and time.time() - t0 > 2.0
    if paused and "pause" not in marks:
        marks["pause"] = os.path.getsize(OUT)
    foley.set_paused(paused)
    if not paused:
        st.step(1 / 30, move=False)
        foley.handle_events(w.pop_events(), w)
        foley.update_world(w, 1 / 30)
    frames += 1
    time.sleep(1 / 60)
marks["end"] = os.path.getsize(OUT)
json.dump({"levels": foley.levels, "played": foley.played[-50:], "frames": frames, "marks": marks}, open(OUT + ".json", "w"))
foley.shutdown()
pygame.mixer.quit()
'''


def capture(tmp_path, mode):
    raw = str(tmp_path / f"{mode}.raw")
    env = dict(os.environ, SDL_AUDIODRIVER="disk", SDL_DISKAUDIOFILE=raw, SDL_DISKAUDIODELAY="0")
    r = subprocess.run([sys.executable, "-c", SCRIPT, ROOT, raw, mode], env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 and "disk" in (r.stderr or "").lower():
        pytest.skip("SDL disk audio driver unavailable")
    assert r.returncode == 0, r.stderr[-2000:]
    pcm = np.fromfile(raw, dtype="<i2").astype(np.float32) / 32768.0
    meta = json.load(open(raw + ".json"))
    return pcm.reshape(-1, 2), meta


def test_fire_is_audible_over_the_evening_ambience(tmp_path):
    quiet, mq = capture(tmp_path, "quiet")
    fire, mf = capture(tmp_path, "fire")
    assert len(quiet) > 44100 and len(fire) > 44100
    rms_q = float(np.sqrt(np.mean(quiet[44100 // 2:] ** 2)))
    rms_f = float(np.sqrt(np.mean(fire[44100 // 2:] ** 2)))
    assert rms_q > 1e-4                                     # birds / crickets / wind are playing
    assert rms_f > 1.5 * rms_q                              # the fire dominates the soundscape
    assert float(np.max(np.abs(fire))) < 0.999              # no clipping in the final mix
    # crackle: fire audio is impulsive (heavy-tailed sample distribution)
    m = fire.mean(axis=1)
    d = np.diff(m)
    kurt = float(np.mean(d ** 4) / (np.mean(d ** 2) ** 2 + 1e-20))
    assert kurt > 4.0
    assert any(p.startswith("resin_pop") for p in mf["played"])
    assert mf["levels"]["loop_fire_roar"] > 0.1


def test_pause_menu_silences_the_world(tmp_path):
    """Esc pauses *everything*: once paused, the fire, birds and wind fall silent."""
    pcm, meta = capture(tmp_path, "pause")
    m = pcm.mean(axis=1)
    p = meta["marks"]["pause"] // 4                     # bytes -> stereo 16-bit frames
    before = m[max(0, p - 44100):p]
    after = m[p + 44100 // 4:]                           # skip the device buffer latency
    assert len(before) > 20000 and len(after) > 20000
    rms_b = float(np.sqrt(np.mean(before ** 2)))
    rms_a = float(np.sqrt(np.mean(after ** 2)))
    assert rms_b > 0.003 and rms_a < 0.02 * rms_b, (rms_b, rms_a)
