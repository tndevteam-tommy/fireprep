"""
Firestarter - Procedural Audio & ASMR Sensory Design Engine
Implements DESIGN Section 9.  Every sound in the game is synthesised from first principles
(filtered noise, modal resonators, stick-slip friction pulse trains, FM birdsong...) at
44.1 kHz stereo, and written once to ``assets/audio`` on first launch.

Loops are made seamless with an equal-power crossfade of their tail into their head.
"""

import io
import math
import os
import wave
from typing import Dict, Callable, Tuple

import numpy as np
from scipy import signal as sps

SR = 44100
AUDIO_VERSION = "3"


# ------------------------------------------------------------------------------------ DSP helpers
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def white(n: int, rng) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, n)


def pink(n: int, rng) -> np.ndarray:
    # Paul Kellet's economy pink filter
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return sps.lfilter(b, a, white(n, rng)) * 3.5


def brown(n: int, rng) -> np.ndarray:
    x = np.cumsum(white(n, rng))
    x = sps.lfilter([1, -1], [1, -0.997], x)
    return x / (np.max(np.abs(x)) + 1e-9)


def bandpass(x, lo, hi, order=2):
    sos = sps.butter(order, [lo / (SR / 2), min(0.99, hi / (SR / 2))], btype="band", output="sos")
    return sps.sosfilt(sos, x)


def lowpass(x, f, order=2):
    sos = sps.butter(order, min(0.99, f / (SR / 2)), btype="low", output="sos")
    return sps.sosfilt(sos, x)


def highpass(x, f, order=2):
    sos = sps.butter(order, min(0.99, f / (SR / 2)), btype="high", output="sos")
    return sps.sosfilt(sos, x)


def smooth_noise(n: int, rate_hz: float, rng) -> np.ndarray:
    """Slowly varying random control signal in [-1, 1]."""
    k = max(2, int(n * rate_hz / SR) + 3)
    pts = rng.uniform(-1, 1, k)
    xs = np.linspace(0, k - 1, n)
    return np.interp(xs, np.arange(k), pts)


def resonator(impulse: np.ndarray, freqs, decays, gains) -> np.ndarray:
    out = np.zeros_like(impulse)
    for f, d, g in zip(freqs, decays, gains):
        r = math.exp(-1.0 / (d * SR))
        w = 2 * math.pi * f / SR
        out += g * sps.lfilter([1 - r], [1, -2 * r * math.cos(w), r * r], impulse)
    return out


def env_ad(n: int, attack: float, decay: float, curve: float = 1.0) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    d = np.exp(-np.maximum(0, t - attack) / max(decay, 1e-4))
    return (a * d) ** curve


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-9
    return x / m * peak


def stereo(l: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.stack([l, r], axis=1)


def mono_to_stereo(x: np.ndarray, width: float = 0.0, pan: float = 0.0) -> np.ndarray:
    d = int(SR * 0.00035 * width)
    r = np.roll(x, d) if d else x
    gl = math.cos((pan + 1) * math.pi / 4)
    gr = math.sin((pan + 1) * math.pi / 4)
    return stereo(x * gl * 1.414, r * gr * 1.414)


def make_loop(x: np.ndarray, xfade_s: float = 0.6) -> np.ndarray:
    """Equal-power crossfade of the tail into the head -> seamless loop."""
    n = int(xfade_s * SR)
    body = x[:-n].copy()
    tail = x[-n:]
    t = np.linspace(0, 1, n)
    fi = np.sin(t * math.pi / 2)
    fo = np.cos(t * math.pi / 2)
    if x.ndim == 2:
        fi, fo = fi[:, None], fo[:, None]
    body[:n] = body[:n] * fi + tail * fo
    return body


def to_wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    if x.ndim == 1:
        x = mono_to_stereo(x)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _dur(s: float) -> int:
    return int(s * SR)


# ------------------------------------------------------------------------------------ loops
def loop_fire_roar(seed=1, secs=9.0):
    rng = _rng(seed)
    n = _dur(secs)
    chans = []
    for c in range(2):
        b = brown(n, rng)
        low = lowpass(b, 260) * 1.6
        whoosh = bandpass(pink(n, rng), 180, 1400) * 0.35
        mod = 0.65 + 0.25 * smooth_noise(n, 1.3, rng) + 0.1 * smooth_noise(n, 6.0, rng)
        flutter = 0.8 + 0.2 * smooth_noise(n, 11.0, rng)
        chans.append((low * mod + whoosh * mod * flutter))
    x = stereo(*chans)
    return normalize(make_loop(x, 0.8), 0.85)


def loop_fire_crackle(seed=2, secs=8.0):
    rng = _rng(seed)
    n = _dur(secs)
    out = np.zeros((n, 2))
    count = int(secs * 38)
    for _ in range(count):
        i = int(rng.integers(0, n - 2000))
        a = rng.exponential(0.25)
        L = int(rng.uniform(60, 600))
        t = np.arange(L) / SR
        click = white(L, rng) * np.exp(-t * rng.uniform(900, 4000)) * a
        pan = rng.uniform(-0.8, 0.8)
        out[i:i + L, 0] += click * (1 - pan) * 0.7
        out[i:i + L, 1] += click * (1 + pan) * 0.7
    for c in range(2):
        out[:, c] = highpass(out[:, c], 700)
    return normalize(make_loop(out, 0.3), 0.8)


def loop_steam_hiss(seed=3, secs=6.0):
    rng = _rng(seed)
    n = _dur(secs)
    ch = []
    for c in range(2):
        h = bandpass(white(n, rng), 2800, 9500)
        mod = 0.55 + 0.3 * smooth_noise(n, 4.5, rng) + 0.15 * np.sin(2 * np.pi * 9.0 * np.arange(n) / SR + c)
        sputter = np.zeros(n)
        for _ in range(int(secs * 6)):
            i = int(rng.integers(0, n - 3000))
            L = int(rng.uniform(800, 2600))
            sputter[i:i + L] += np.hanning(L) * rng.uniform(0.3, 0.8)
        ch.append(h * (mod + sputter * 0.6))
    return normalize(make_loop(stereo(*ch), 0.4), 0.6)


def loop_embers(seed=4, secs=10.0):
    rng = _rng(seed)
    n = _dur(secs)
    base = lowpass(pink(n, rng), 500) * 0.12
    out = np.stack([base, np.roll(base, 400)], axis=1)
    for _ in range(int(secs * 3.5)):
        i = int(rng.integers(0, n - 20000))
        L = 16000
        imp = np.zeros(L)
        imp[0] = 1.0
        tink = resonator(imp, [rng.uniform(3000, 7000), rng.uniform(5000, 9500)], [0.05, 0.03], [0.6, 0.3])
        pan = rng.uniform(-0.7, 0.7)
        out[i:i + L, 0] += tink * (1 - pan) * 0.25
        out[i:i + L, 1] += tink * (1 + pan) * 0.25
    return normalize(make_loop(out, 0.5), 0.5)


def loop_wind(seed=5, secs=12.0, muffled=False):
    rng = _rng(seed)
    n = _dur(secs)
    gust = 0.55 + 0.45 * smooth_noise(n, 0.35, rng)
    ch = []
    for c in range(2):
        b = brown(n, rng) * 0.8 + pink(n, rng) * 0.05
        body = bandpass(b, 60, 700 if not muffled else 300) * 2.5
        # narrow whistling resonances through branches, sweeping with the gusts
        whistle = np.zeros(n)
        for k in range(2):
            f0 = rng.uniform(420, 900)
            seg = bandpass(white(n, rng), f0 * 0.97, f0 * 1.03, order=2) * (0.25 if not muffled else 0.05)
            whistle += seg * np.clip(gust - 0.4, 0, 1) * 2
        ch.append((body + whistle) * gust)
    return normalize(make_loop(stereo(*ch), 1.0), 0.75)


def _bird_phrase(rng) -> np.ndarray:
    kind = rng.integers(0, 3)
    parts = []
    if kind == 0:      # descending whistle notes
        for k in range(int(rng.integers(3, 6))):
            L = _dur(rng.uniform(0.08, 0.16))
            t = np.arange(L) / SR
            f0 = rng.uniform(3200, 4200) - k * rng.uniform(80, 200)
            f = f0 + 500 * np.sin(np.pi * t / t[-1])
            ph = 2 * np.pi * np.cumsum(f) / SR
            parts.append(np.sin(ph) * np.hanning(L))
            parts.append(np.zeros(_dur(rng.uniform(0.04, 0.09))))
    elif kind == 1:    # trill
        L = _dur(rng.uniform(0.6, 1.1))
        t = np.arange(L) / SR
        f = rng.uniform(4200, 5600) + 700 * np.sign(np.sin(2 * np.pi * rng.uniform(18, 26) * t))
        f = lowpass(f, 80)
        ph = 2 * np.pi * np.cumsum(f) / SR
        parts.append(np.sin(ph) * np.hanning(L) * (0.6 + 0.4 * np.sin(2 * np.pi * 22 * t)))
    else:              # two-note "chick-a-dee" like call
        for f0 in (rng.uniform(3600, 4000), rng.uniform(2900, 3300), rng.uniform(2900, 3300)):
            L = _dur(0.12)
            t = np.arange(L) / SR
            ph = 2 * np.pi * np.cumsum(f0 - 300 * t / t[-1]) / SR
            parts.append((np.sin(ph) + 0.3 * np.sin(2 * ph)) * np.hanning(L))
            parts.append(np.zeros(_dur(0.05)))
    return np.concatenate(parts)


def loop_forest_day(seed=6, secs=16.0):
    rng = _rng(seed)
    n = _dur(secs)
    rustle = bandpass(pink(n, rng), 1500, 7000) * (0.35 + 0.35 * np.clip(smooth_noise(n, 0.5, rng), 0, 1)) * 0.25
    out = np.stack([rustle, np.roll(rustle, 2000)], axis=1)
    t = 0.4
    while t < secs - 2.0:
        ph = _bird_phrase(rng) * rng.uniform(0.12, 0.35)
        i = _dur(t)
        L = min(len(ph), n - i)
        pan = rng.uniform(-0.9, 0.9)
        # distance: a little high-cut and room tail
        ph = lowpass(ph, rng.uniform(5000, 9000))
        out[i:i + L, 0] += ph[:L] * (1 - pan)
        out[i:i + L, 1] += ph[:L] * (1 + pan)
        t += rng.uniform(1.2, 3.5)
    return normalize(make_loop(out, 0.8), 0.55)


def loop_night(seed=7, secs=16.0):
    rng = _rng(seed)
    n = _dur(secs)
    tt = np.arange(n) / SR
    out = np.zeros((n, 2))
    for k in range(3):      # individual crickets
        f = rng.uniform(4300, 4900)
        rate = rng.uniform(1.6, 2.6)     # chirps per second
        pulses = rng.integers(3, 5)
        carrier = np.sin(2 * np.pi * f * tt)
        gate = np.zeros(n)
        period = 1.0 / rate
        start = rng.uniform(0, period)
        while start < secs:
            for p in range(pulses):
                s0 = _dur(start + p * 0.022)
                L = _dur(0.014)
                if s0 + L < n:
                    gate[s0:s0 + L] += np.hanning(L)
            start += period * rng.uniform(0.9, 1.1)
        c = carrier * gate * rng.uniform(0.08, 0.16)
        pan = rng.uniform(-0.9, 0.9)
        out[:, 0] += c * (1 - pan)
        out[:, 1] += c * (1 + pan)
    # distant owl
    for t0 in (4.0, 11.5):
        L = _dur(0.5)
        t = np.arange(L) / SR
        f = 380 + 15 * np.sin(2 * np.pi * 6 * t)
        hoot = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(L) * 0.18
        for rep, dt_ in ((0, 0.0), (1, 0.65)):
            i = _dur(t0 + dt_)
            out[i:i + L, 0] += lowpass(hoot, 1200) * 0.7
            out[i:i + L, 1] += lowpass(hoot, 1200) * 0.4
    air = lowpass(pink(n, rng), 400) * 0.05
    out[:, 0] += air
    out[:, 1] += np.roll(air, 300)
    return normalize(make_loop(out, 0.8), 0.5)


def loop_rain(seed=8, secs=10.0):
    rng = _rng(seed)
    n = _dur(secs)
    out = np.zeros((n, 2))
    for c in range(2):
        drops = np.zeros(n)
        idx = rng.integers(0, n, int(secs * 2500))
        drops[idx] = rng.exponential(0.3, len(idx)) * np.sign(rng.uniform(-1, 1, len(idx)))
        drops = bandpass(drops, 1500, 9000)
        hiss = bandpass(pink(n, rng), 400, 6000) * 0.4
        out[:, c] = drops + hiss
    return normalize(make_loop(out, 0.5), 0.7)


def loop_rain_tarp(seed=9, secs=8.0):
    rng = _rng(seed)
    n = _dur(secs)
    imp = np.zeros(n)
    idx = rng.integers(0, n, int(secs * 180))
    imp[idx] = rng.uniform(0.3, 1.0, len(idx))
    taps = resonator(imp, [180, 330, 520], [0.03, 0.02, 0.012], [1.0, 0.6, 0.35])
    taps = taps + highpass(imp, 2000) * 0.3
    return normalize(make_loop(mono_to_stereo(taps, 1.0), 0.4), 0.7)


def loop_creek(seed=10, secs=12.0):
    rng = _rng(seed)
    n = _dur(secs)
    out = np.zeros((n, 2))
    base = bandpass(pink(n, rng), 200, 3000) * 0.35
    out[:, 0] += base
    out[:, 1] += np.roll(base, 900)
    for _ in range(int(secs * 45)):
        L = _dur(rng.uniform(0.01, 0.05))
        t = np.arange(L) / SR
        f0 = rng.uniform(300, 1100)
        f = f0 * (1 + 1.5 * t / t[-1])          # rising bubble resonance
        b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * rng.uniform(40, 120))
        i = int(rng.integers(0, n - L))
        pan = rng.uniform(-0.8, 0.8)
        a = rng.uniform(0.05, 0.25)
        out[i:i + L, 0] += b * a * (1 - pan)
        out[i:i + L, 1] += b * a * (1 + pan)
    return normalize(make_loop(out, 0.6), 0.6)


# ------------------------------------------------------------------------------------ one-shots
def shot_resin_pop(seed=0):
    rng = _rng(100 + seed)
    n = _dur(0.18)
    t = np.arange(n) / SR
    click = white(n, rng) * np.exp(-t * 900)
    tone = np.sin(2 * np.pi * rng.uniform(1800, 3200) * t) * np.exp(-t * 160)
    hiss = bandpass(white(n, rng), 3000, 9000) * np.exp(-t * 30) * 0.15
    x = click * 0.9 + tone * 0.35 + hiss
    return mono_to_stereo(normalize(x, 0.9), 0.5)


def shot_lighter(seed=0, ignite=True):
    rng = _rng(200 + seed)
    n = _dur(0.55)
    t = np.arange(n) / SR
    scrape = bandpass(white(n, rng), 2200, 7500) * np.where(t < 0.09, np.sin(np.pi * t / 0.09), 0) * 0.8
    grit = np.zeros(n)
    for _ in range(14):
        i = int(rng.integers(0, _dur(0.09)))
        grit[i] = rng.uniform(0.5, 1.0)
    grit = highpass(grit, 3000)
    x = scrape + grit * 0.6
    if ignite:
        w = lowpass(white(n, rng), 900) * np.clip((t - 0.07) * 40, 0, 1) * np.exp(-np.maximum(0, t - 0.1) * 9) * 1.6
        x = x + w
    return mono_to_stereo(normalize(x, 0.8), 0.3)


def shot_whittle(seed=0):
    rng = _rng(300 + seed)
    n = _dur(0.32)
    t = np.arange(n) / SR
    stickslip = 0.7 + 0.3 * np.sign(np.sin(2 * np.pi * rng.uniform(90, 160) * t))
    x = bandpass(white(n, rng), 1800, 7000) * stickslip * np.sin(np.pi * t / t[-1]) ** 1.5
    return mono_to_stereo(normalize(x, 0.7), 0.4)


def shot_chop(seed=0):
    rng = _rng(400 + seed)
    n = _dur(0.5)
    imp = np.zeros(n)
    imp[0] = 1.0
    body = resonator(imp, [rng.uniform(160, 200), rng.uniform(380, 440), rng.uniform(720, 800), 1300], [0.09, 0.05, 0.03, 0.015], [1.0, 0.7, 0.45, 0.2])
    t = np.arange(n) / SR
    crunch = bandpass(white(n, rng), 800, 5000) * np.exp(-t * 45) * 0.5
    x = body * 2.0 + crunch
    return mono_to_stereo(normalize(x, 0.95), 0.3)


def shot_tree_creak(seed=0):
    rng = _rng(500 + seed)
    n = _dur(2.4)
    t = np.arange(n) / SR
    f = 30 + 25 * np.sin(2 * np.pi * 0.4 * t) + 10 * smooth_noise(n, 3.0, rng)
    ph = np.cumsum(f) / SR
    pulses = (np.diff(np.floor(ph), prepend=0) > 0).astype(float)
    x = resonator(pulses, [240, 520, 870], [0.02, 0.015, 0.01], [1.0, 0.6, 0.4])
    x *= np.sin(np.pi * t / t[-1]) ** 0.5
    return mono_to_stereo(normalize(x, 0.8), 0.6)


def shot_tree_fall(seed=0):
    rng = _rng(600 + seed)
    n = _dur(4.0)
    t = np.arange(n) / SR
    swoosh = bandpass(pink(n, rng), 500, 4000) * np.clip(t / 1.4, 0, 1) ** 2 * (t < 1.6) * 0.6
    crash = np.zeros(n)
    i0 = _dur(1.55)
    for _ in range(90):
        i = i0 + int(rng.exponential(_dur(0.25)))
        if i < n - 800:
            L = int(rng.uniform(150, 800))
            crash[i:i + L] += white(L, rng) * np.exp(-np.arange(L) / (L * 0.2)) * rng.uniform(0.3, 1.0)
    crash = highpass(crash, 600)
    thud = np.zeros(n)
    thud[i0] = 1.0
    thud = resonator(thud, [55, 85, 130], [0.35, 0.25, 0.15], [1.0, 0.7, 0.4]) * 3.0
    rumble = lowpass(brown(n, rng), 150) * np.exp(-np.maximum(0, t - 1.55) * 2.0) * (t > 1.55) * 0.8
    x = swoosh + crash * 0.8 + thud + rumble
    return mono_to_stereo(normalize(x, 0.95), 1.0)


def shot_squeak(seed=0):
    rng = _rng(700 + seed)
    n = _dur(0.4)
    t = np.arange(n) / SR
    f = rng.uniform(380, 520) + 160 * np.sin(np.pi * t / t[-1])
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)) * (0.7 + 0.3 * np.sign(np.sin(2 * np.pi * 60 * t)))
    x = x * np.sin(np.pi * t / t[-1]) ** 1.3 * 0.5 + bandpass(white(n, rng), 1500, 5000) * 0.15 * np.sin(np.pi * t / t[-1])
    return mono_to_stereo(normalize(x, 0.55), 0.3)


def shot_breath(seed=0, intensity=1.0):
    rng = _rng(800 + seed)
    dur = 0.45 + 0.35 * intensity
    n = _dur(dur)
    t = np.arange(n) / SR
    noise = white(n, rng)
    x = bandpass(noise, 500, 2600) * 0.8 + bandpass(noise, 2600, 6000) * 0.35 * intensity
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** (2.0 - intensity)
    return mono_to_stereo(normalize(x * e, 0.5 + 0.3 * intensity), 0.2)


def shot_rustle(seed=0):
    rng = _rng(900 + seed)
    n = _dur(0.45)
    t = np.arange(n) / SR
    grains = np.zeros(n)
    for _ in range(60):
        i = int(rng.integers(0, n - 400))
        L = int(rng.uniform(80, 400))
        grains[i:i + L] += white(L, rng) * rng.uniform(0.2, 1.0) * np.hanning(L)
    x = bandpass(grains, 1200, 8000) * np.sin(np.pi * t / t[-1])
    return mono_to_stereo(normalize(x, 0.6), 0.5)


def shot_wood_knock(seed=0):
    rng = _rng(1000 + seed)
    n = _dur(0.35)
    imp = np.zeros(n)
    imp[0] = 1.0
    imp[int(rng.integers(300, 2000))] = 0.5
    x = resonator(imp, [rng.uniform(300, 380), rng.uniform(650, 800), 1450], [0.05, 0.03, 0.015], [1.0, 0.6, 0.3])
    return mono_to_stereo(normalize(x, 0.7), 0.3)


def shot_footstep(seed=0, surface="leaves"):
    rng = _rng(1100 + seed)
    n = _dur(0.28)
    t = np.arange(n) / SR
    if surface == "snow":
        grains = np.zeros(n)
        for _ in range(120):
            i = int(rng.integers(0, n - 200))
            grains[i] = rng.uniform(0.3, 1.0)
        x = bandpass(grains, 700, 5000) * np.sin(np.pi * t / t[-1])
    elif surface == "water":
        x = bandpass(white(n, rng), 300, 3000) * np.exp(-t * 10)
        for _ in range(8):
            L = _dur(0.03)
            i = int(rng.integers(0, n - L))
            tt = np.arange(L) / SR
            x[i:i + L] += np.sin(2 * np.pi * rng.uniform(500, 1400) * (1 + tt * 30) * tt) * np.exp(-tt * 80) * 0.5
    else:
        thud = lowpass(white(n, rng), 300) * np.exp(-t * 40) * 0.8
        crunch = np.zeros(n)
        for _ in range(35):
            i = int(rng.integers(0, n // 2))
            L = int(rng.uniform(60, 300))
            crunch[i:i + L] += white(L, rng) * np.hanning(L) * rng.uniform(0.2, 1.0)
        x = thud + bandpass(crunch, 1500, 8000) * 0.7
    return mono_to_stereo(normalize(x, 0.5), 0.2)


def shot_collapse(seed=0):
    rng = _rng(1200 + seed)
    n = _dur(1.1)
    t = np.arange(n) / SR
    thud = np.zeros(n)
    thud[0] = 1.0
    x = resonator(thud, [110, 260, 540], [0.12, 0.06, 0.03], [1.0, 0.5, 0.3]) * 1.2
    crack = np.zeros(n)
    for _ in range(50):
        i = int(rng.exponential(_dur(0.18)))
        if i < n - 400:
            L = int(rng.uniform(60, 400))
            crack[i:i + L] += white(L, rng) * np.exp(-np.arange(L) / (L * 0.25)) * rng.exponential(0.4)
    whoosh = lowpass(white(n, rng), 1200) * np.exp(-t * 4) * 0.4
    return mono_to_stereo(normalize(x + highpass(crack, 800) + whoosh, 0.85), 0.7)


def shot_gas_whoosh(seed=0):
    rng = _rng(1300 + seed)
    n = _dur(2.2)
    t = np.arange(n) / SR
    whump = np.zeros(n)
    whump[0] = 1.0
    whump = resonator(whump, [45, 70], [0.25, 0.15], [1.0, 0.6]) * 3.0
    roar = lowpass(brown(n, rng), 900) * np.clip(t * 20, 0, 1) * np.exp(-t * 1.6) * 2.0
    hiss = bandpass(white(n, rng), 800, 6000) * np.exp(-t * 2.5) * 0.6
    return mono_to_stereo(normalize(whump + roar + hiss, 0.98), 1.0)


def shot_page_turn(seed=0):
    rng = _rng(1400 + seed)
    n = _dur(0.45)
    t = np.arange(n) / SR
    sweep = np.zeros(n)
    f = 1500 + 4000 * t / t[-1]
    noise = white(n, rng)
    for k in range(4):
        s = slice(k * n // 4, (k + 1) * n // 4)
        sweep[s] = bandpass(noise, 1000 + 1200 * k, 2500 + 1800 * k)[s]
    flap = np.zeros(n)
    flap[int(0.34 * SR)] = 1.0
    flap = resonator(flap, [420, 900], [0.02, 0.01], [1.0, 0.4])
    x = sweep * np.sin(np.pi * t / t[-1]) ** 2 * 0.7 + flap * 0.4
    return mono_to_stereo(normalize(x, 0.5), 0.6)


def shot_chime(seed=0, notes=(880.0, 1318.5), spacing=0.09, dur=0.9):
    n = _dur(dur + spacing * len(notes))
    t = np.arange(n) / SR
    x = np.zeros(n)
    for k, f in enumerate(notes):
        t0 = k * spacing
        tt = np.clip(t - t0, 0, None)
        on = t >= t0
        x += on * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2.76 * tt) * np.exp(-tt * 8)) * np.exp(-tt * 4.5)
    return mono_to_stereo(normalize(x, 0.35), 0.3)


def shot_click(seed=0):
    n = _dur(0.04)
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 220)
    return mono_to_stereo(normalize(x, 0.25))


def shot_error(seed=0):
    n = _dur(0.22)
    t = np.arange(n) / SR
    x = np.sign(np.sin(2 * np.pi * 160 * t)) * np.exp(-t * 14) * 0.5
    return mono_to_stereo(normalize(lowpass(x, 1500), 0.25))


def shot_snap(seed=0):
    rng = _rng(1500 + seed)
    n = _dur(0.25)
    t = np.arange(n) / SR
    x = white(n, rng) * np.exp(-t * 120) + np.sin(2 * np.pi * 1800 * t) * np.exp(-t * 60) * 0.3
    return mono_to_stereo(normalize(highpass(x, 500), 0.8))


def shot_tarp(seed=0):
    rng = _rng(1600 + seed)
    n = _dur(0.9)
    t = np.arange(n) / SR
    crinkle = np.zeros(n)
    for _ in range(150):
        i = int(rng.integers(0, n - 300))
        L = int(rng.uniform(40, 300))
        crinkle[i:i + L] += white(L, rng) * np.hanning(L) * rng.exponential(0.5)
    x = bandpass(crinkle, 2000, 10000) * np.sin(np.pi * t / t[-1])
    return mono_to_stereo(normalize(x, 0.6), 0.8)


def shot_pour(seed=0):
    rng = _rng(1700 + seed)
    n = _dur(1.0)
    x = bandpass(white(n, rng), 300, 2500) * 0.3
    for _ in range(12):
        L = _dur(0.06)
        i = int(rng.integers(0, n - L))
        tt = np.arange(L) / SR
        x[i:i + L] += np.sin(2 * np.pi * rng.uniform(180, 320) * (1 + tt * 8) * tt) * np.exp(-tt * 40)
    return mono_to_stereo(normalize(x, 0.6), 0.3)


def shot_harvest(seed=0):
    rng = _rng(1800 + seed)
    n = _dur(0.6)
    t = np.arange(n) / SR
    tear = np.zeros(n)
    for _ in range(80):
        i = int(rng.integers(0, n - 200))
        tear[i] = rng.uniform(0.3, 1.0)
    x = bandpass(tear, 800, 6000) * np.sin(np.pi * t / t[-1]) + lowpass(white(n, rng), 600) * 0.2 * np.exp(-t * 6)
    return mono_to_stereo(normalize(x, 0.6), 0.5)


def shot_ember(seed=0):
    rng = _rng(1900 + seed)
    n = _dur(0.8)
    imp = np.zeros(n)
    imp[0] = 1.0
    x = resonator(imp, [5200, 7400], [0.08, 0.05], [0.6, 0.3])
    crackle = shot_resin_pop(seed)[:, 0]
    x[:len(crackle)] += crackle * 0.15
    return mono_to_stereo(normalize(x, 0.35), 0.3)


def shot_whoosh(seed=0):
    rng = _rng(2000 + seed)
    n = _dur(1.2)
    t = np.arange(n) / SR
    x = bandpass(pink(n, rng), 300, 3000) * np.sin(np.pi * t / t[-1]) ** 2
    return mono_to_stereo(normalize(x, 0.5), 1.0)


LOOPS: Dict[str, Callable[[], np.ndarray]] = {
    "loop_fire_roar": loop_fire_roar,
    "loop_fire_crackle": loop_fire_crackle,
    "loop_steam_hiss": loop_steam_hiss,
    "loop_embers": loop_embers,
    "loop_wind": loop_wind,
    "loop_wind_snow": lambda: loop_wind(seed=55, muffled=True),
    "loop_forest_day": loop_forest_day,
    "loop_night": loop_night,
    "loop_rain": loop_rain,
    "loop_rain_tarp": loop_rain_tarp,
    "loop_creek": loop_creek,
}

SHOTS: Dict[str, Callable[[], np.ndarray]] = {
    **{f"resin_pop_{i}": (lambda i=i: shot_resin_pop(i)) for i in range(4)},
    "lighter_strike": lambda: shot_lighter(0, True),
    "lighter_fail": lambda: shot_lighter(1, False),
    **{f"whittle_{i}": (lambda i=i: shot_whittle(i)) for i in range(3)},
    **{f"chop_{i}": (lambda i=i: shot_chop(i)) for i in range(3)},
    "tree_creak": shot_tree_creak,
    "tree_fall": shot_tree_fall,
    **{f"friction_squeak_{i}": (lambda i=i: shot_squeak(i)) for i in range(3)},
    "breath_soft": lambda: shot_breath(0, 0.4),
    "breath_med": lambda: shot_breath(1, 0.8),
    "breath_hard": lambda: shot_breath(2, 1.2),
    **{f"rustle_{i}": (lambda i=i: shot_rustle(i)) for i in range(3)},
    **{f"wood_knock_{i}": (lambda i=i: shot_wood_knock(i)) for i in range(3)},
    **{f"step_leaves_{i}": (lambda i=i: shot_footstep(i, "leaves")) for i in range(4)},
    **{f"step_snow_{i}": (lambda i=i: shot_footstep(i, "snow")) for i in range(3)},
    **{f"step_water_{i}": (lambda i=i: shot_footstep(i, "water")) for i in range(2)},
    **{f"collapse_{i}": (lambda i=i: shot_collapse(i)) for i in range(2)},
    "gas_whoosh": shot_gas_whoosh,
    "page_turn": shot_page_turn,
    "ui_click": shot_click,
    "ui_error": shot_error,
    "phone_open": lambda: shot_chime(0, (1046.5, 1568.0)),
    "purchase": lambda: shot_chime(1, (784.0, 987.8, 1318.5), 0.08, 1.1),
    "credits": lambda: shot_chime(2, (1318.5, 1760.0, 2093.0), 0.06, 0.8),
    "cord_snap": shot_snap,
    "tarp": shot_tarp,
    "pour": shot_pour,
    "harvest": shot_harvest,
    "ember": shot_ember,
    "whoosh": shot_whoosh,
}


class ProceduralAudioEngine:
    """Synthesis front-end (kept with the original class name)."""

    @staticmethod
    def synth_resin_pop(duration: float = 0.08, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(shot_resin_pop(0))

    @staticmethod
    def synth_steam_hiss(duration: float = 1.2, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(loop_steam_hiss(secs=max(1.0, duration + 0.4)))

    @staticmethod
    def synth_combustion_roar(duration: float = 2.0, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(loop_fire_roar(secs=max(1.5, duration + 0.8)))

    @staticmethod
    def synth_whittling(duration: float = 0.28, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(shot_whittle(0))

    @staticmethod
    def synth_friction_squeak(duration: float = 0.35, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(shot_squeak(0))

    @staticmethod
    def synth_breath_blowing(duration: float = 0.75, intensity: float = 1.0, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(shot_breath(0, intensity))

    @staticmethod
    def synth_wind_ambient(duration: float = 2.5, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(loop_wind(secs=max(2.0, duration + 1.0)))

    @staticmethod
    def synth_rain_on_tarp(duration: float = 2.0, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(loop_rain_tarp(secs=max(1.5, duration + 0.4)))

    @staticmethod
    def _to_wav_bytes(sig: np.ndarray, sample_rate: int = SR) -> bytes:
        return to_wav_bytes(sig, sample_rate)

    @staticmethod
    def all_sounds() -> Dict[str, Callable[[], np.ndarray]]:
        return {**LOOPS, **SHOTS}

    @staticmethod
    def assets_up_to_date(output_dir: str) -> bool:
        stamp = os.path.join(output_dir, ".version")
        if not os.path.exists(stamp):
            return False
        if open(stamp).read().strip() != AUDIO_VERSION:
            return False
        return all(os.path.exists(os.path.join(output_dir, f"{k}.wav")) for k in ProceduralAudioEngine.all_sounds())

    @staticmethod
    def generate_and_save_all_assets(output_dir: str, verbose: bool = False) -> int:
        """Synthesises every sound to WAV.  Returns the number of files written."""
        os.makedirs(output_dir, exist_ok=True)
        n = 0
        for name, fn in ProceduralAudioEngine.all_sounds().items():
            data = to_wav_bytes(fn())
            with open(os.path.join(output_dir, f"{name}.wav"), "wb") as f:
                f.write(data)
            n += 1
            if verbose:
                print(f"  synthesised {name}.wav ({len(data) // 1024} KiB)")
        with open(os.path.join(output_dir, ".version"), "w") as f:
            f.write(AUDIO_VERSION)
        return n
