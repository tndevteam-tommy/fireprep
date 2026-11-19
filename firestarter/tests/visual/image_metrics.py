"""
Objective image metrics used by the visual test-suite.

All functions take an (H, W, 3) uint8 sRGB array.  They are deliberately simple, robust
statistics (no reference images): exposure, contrast, colourfulness, sharpness, warm
"fire" pixels, clipping, region colours and frame-to-frame motion.
"""

from typing import Dict, Tuple

import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def luminance(img: np.ndarray) -> np.ndarray:
    f = to_float(img)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def colorfulness(img: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colourfulness metric (0 = grey; > 40 vivid)."""
    f = np.asarray(img, dtype=np.float32)
    rg = f[..., 0] - f[..., 1]
    yb = 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def sharpness(img: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian of the luminance (detail / focus measure)."""
    L = luminance(img) * 255.0
    lap = -4 * L[1:-1, 1:-1] + L[:-2, 1:-1] + L[2:, 1:-1] + L[1:-1, :-2] + L[1:-1, 2:]
    return float(lap.var())


def hsv(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = to_float(img)
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    h = np.zeros_like(mx)
    m = d > 1e-6
    rr = m & (mx == r)
    gg = m & (mx == g) & ~rr
    bb = m & ~rr & ~gg
    h[rr] = ((g - b)[rr] / d[rr]) % 6.0
    h[gg] = (b - r)[gg] / d[gg] + 2.0
    h[bb] = (r - g)[bb] / d[bb] + 4.0
    return h * 60.0, s, mx


def warm_fraction(img: np.ndarray, min_value: float = 0.55, min_sat: float = 0.35) -> float:
    """Fraction of bright, saturated orange-yellow pixels (flames, embers, firelight)."""
    h, s, v = hsv(img)
    warm = (h >= 5.0) & (h <= 60.0) & (s >= min_sat) & (v >= min_value)
    return float(warm.mean())


def hue_fraction(img: np.ndarray, lo: float, hi: float, min_sat: float = 0.3, min_value: float = 0.15) -> float:
    h, s, v = hsv(img)
    return float(((h >= lo) & (h <= hi) & (s >= min_sat) & (v >= min_value)).mean())


def clipped_fraction(img: np.ndarray) -> float:
    return float((np.asarray(img) >= 250).all(axis=2).mean())


def dark_fraction(img: np.ndarray, thr: float = 0.03) -> float:
    return float((luminance(img) < thr).mean())


def region(img: np.ndarray, y0: float, y1: float, x0: float = 0.0, x1: float = 1.0) -> np.ndarray:
    H, W = img.shape[:2]
    return img[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]


def mean_rgb(img: np.ndarray) -> np.ndarray:
    return to_float(img).reshape(-1, 3).mean(axis=0)


def bright_points(img: np.ndarray, thr: float = 0.55) -> int:
    """Count isolated bright pixels (stars, sparks): brighter than all 8 neighbours by a margin."""
    L = luminance(img)
    c = L[1:-1, 1:-1]
    neigh = np.max(np.stack([L[:-2, :-2], L[:-2, 1:-1], L[:-2, 2:], L[1:-1, :-2], L[1:-1, 2:], L[2:, :-2], L[2:, 1:-1], L[2:, 2:]]), axis=0)
    nmin = np.min(np.stack([L[:-2, 1:-1], L[2:, 1:-1], L[1:-1, :-2], L[1:-1, 2:]]), axis=0)
    return int(((c > thr) & (c >= neigh) & (c - nmin > 0.2)).sum())


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(to_float(a) - to_float(b)).mean())


def summary(img: np.ndarray) -> Dict[str, float]:
    L = luminance(img)
    return {
        "mean_luma": round(float(L.mean()), 4),
        "std_luma": round(float(L.std()), 4),
        "colorfulness": round(colorfulness(img), 2),
        "sharpness": round(sharpness(img), 1),
        "warm_fraction": round(warm_fraction(img), 5),
        "clipped_fraction": round(clipped_fraction(img), 5),
        "dark_fraction": round(dark_fraction(img), 4),
    }
