"""
Firestarter - procedural textures (no image files are shipped).

``foliage_atlas()`` paints a 2 x 2 atlas of leaf / needle clusters that the tree, bush and
distant-forest meshes use as alpha-cut "cards":

    cell 0  white-oak spray   - lobed leaves on a forked twig
    cell 1  paper-birch spray - small toothed ovate leaves hanging from fine twigs
    cell 2  white-pine spray  - soft 5-needle fascicles along a twig
    cell 3  red-cedar frond   - flat, fan-like sprays of scale leaves

RGB holds a brightness / hue multiplier (the species colour comes from the mesh), A the
coverage.  Painted at 2x and downsampled for smooth edges.  Deterministic (seeded).
"""

import math
import os
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CELL = 512
ATLAS = 2 * CELL
SS = 2                      # supersampling factor while painting


def cell_uv(cell: int) -> Tuple[float, float, float, float]:
    """(u0, v0, u1, v1) of an atlas cell (v = 0 at the bottom, OpenGL convention)."""
    cx, cy = cell % 2, cell // 2
    return cx * 0.5, 1.0 - (cy + 1) * 0.5, (cx + 1) * 0.5, 1.0 - cy * 0.5


def _leaf_polygon(length: float, width: float, lobes: int, tooth: float, rng) -> np.ndarray:
    """Leaf outline along +x from the petiole (0,0) to the tip (length, 0)."""
    n = 40
    t = np.linspace(0.0, 1.0, n)
    # base ovate profile, blunt at the base and pointed at the tip
    prof = np.sin(np.pi * np.clip(t, 0, 1) ** 0.75) * (1.0 - 0.25 * t)
    if lobes > 0:
        prof = prof * (0.72 + 0.28 * np.abs(np.sin(t * np.pi * lobes + 0.3)))
    if tooth > 0:
        prof = prof * (1.0 - tooth * (np.arange(n) % 2))
    upper = np.stack([t * length, prof * width * 0.5], axis=1)
    lower = np.stack([t[::-1] * length, -prof[::-1] * width * 0.5 * rng.uniform(0.85, 1.0)], axis=1)
    return np.concatenate([upper, lower])


def _draw_leaf(d: ImageDraw.ImageDraw, origin, angle, length, width, lobes, tooth, shade, rng):
    poly = _leaf_polygon(length, width, lobes, tooth, rng)
    c, s = math.cos(angle), math.sin(angle)
    pts = [(origin[0] + x * c - y * s, origin[1] + x * s + y * c) for x, y in poly]
    g = int(max(0, min(255, shade * 200)))
    hue = rng.uniform(-5, 5)
    d.polygon(pts, fill=(int(max(0, min(255, g + hue))), g, int(max(0, min(255, g - hue * 0.5))), 255))
    # midrib and a darker rim for a bit of form
    d.line([(origin[0], origin[1]), (origin[0] + length * 0.92 * c, origin[1] + length * 0.92 * s)],
           fill=(min(255, g + 30), min(255, g + 34), min(255, g + 18), 255), width=max(1, int(length * 0.03)))


def _paint_broadleaf(rng, small: bool) -> Image.Image:
    S = CELL * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    twig = (105, 100, 95, 255)
    # a main twig rising through the cell with alternate side shoots
    base = np.array([S * 0.5, S * 0.99])
    top = np.array([S * rng.uniform(0.42, 0.58), S * 0.08])
    d.line([tuple(base), tuple(top)], fill=twig, width=int(S * (0.007 if small else 0.011)))
    shoots = [(base + (top - base) * 0.05, top)]
    for k in range(7 if small else 5):
        t = 0.15 + 0.75 * k / (6 if small else 4)
        o = base + (top - base) * t
        side = 1 if k % 2 else -1
        ang = -math.pi / 2 + side * rng.uniform(0.85, 1.35)
        L = S * rng.uniform(0.34, 0.46) * (1.0 - 0.35 * t)
        e = o + L * np.array([math.cos(ang), math.sin(ang)])
        e = np.clip(e, S * 0.06, S * 0.94)
        d.line([tuple(o), tuple(e)], fill=twig, width=int(S * (0.005 if small else 0.008)))
        shoots.append((o, e))
    n_leaves = 120 if small else 62
    for i in range(n_leaves):
        b, e = shoots[i % len(shoots)]
        t = rng.uniform(0.15, 1.0)
        o = b + (e - b) * t
        side = 1 if i % 2 else -1
        ang = math.atan2(e[1] - b[1], e[0] - b[0]) + side * rng.uniform(0.35, 1.1)
        if small:
            ang += side * rng.uniform(0.1, 0.5) + 0.35   # birch leaves droop on slender petioles
            length = S * rng.uniform(0.075, 0.105)
            width = length * rng.uniform(0.62, 0.75)
            _draw_leaf(d, o, ang, length, width, 0, 0.06, rng.uniform(0.72, 1.12), rng)
        else:
            length = S * rng.uniform(0.13, 0.2)
            width = length * rng.uniform(0.45, 0.58)
            _draw_leaf(d, o, ang, length, width, 4, 0.0, rng.uniform(0.68, 1.1), rng)
    return img


def _paint_needles(rng) -> Image.Image:
    S = CELL * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    a = np.array([S * 0.5, S * 0.97])
    b = np.array([S * 0.5 + rng.uniform(-30, 30), S * 0.06])
    d.line([tuple(a), tuple(b)], fill=(95, 88, 80, 255), width=int(S * 0.01))
    for i in range(230):
        t = rng.uniform(0.04, 0.97)
        o = a + (b - a) * t
        base_ang = math.atan2(b[1] - a[1], b[0] - a[0])
        ang = base_ang + rng.choice([-1, 1]) * rng.uniform(0.3, 1.25)
        L = S * rng.uniform(0.2, 0.34) * (1.0 - 0.3 * t)
        g = rng.uniform(0.6, 1.1)
        col = (int(min(255, g * 185)), int(min(255, g * 200)), int(min(255, g * 180)), 255)
        bend = rng.uniform(-0.15, 0.15)
        mid = o + 0.5 * L * np.array([math.cos(ang + bend), math.sin(ang + bend)])
        end = o + L * np.array([math.cos(ang), math.sin(ang)])
        d.line([tuple(o), tuple(mid), tuple(end)], fill=col, width=max(2, int(S * 0.0045)))
    return img


def _paint_cedar(rng) -> Image.Image:
    S = CELL * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def frond(p0, ang, L, w, depth):
        p1 = p0 + L * np.array([math.cos(ang), math.sin(ang)])
        g = rng.uniform(0.7, 1.1)
        d.line([tuple(p0), tuple(p1)], fill=(int(g * 180), int(g * 188), int(g * 172), 255), width=max(2, int(w)))
        # scale leaves: short stubby nubs along the branchlet
        n = int(L / (w * 1.1)) + 1
        for k in range(n):
            q = p0 + (p1 - p0) * (k / max(1, n))
            for sgn in (-1, 1):
                a2 = ang + sgn * 0.9
                e = q + w * 0.9 * np.array([math.cos(a2), math.sin(a2)])
                d.line([tuple(q), tuple(e)], fill=(int(g * 172), int(g * 182), int(g * 165), 255), width=max(2, int(w * 0.9)))
        if depth > 0:
            for k in range(1, 5):
                q = p0 + (p1 - p0) * (k / 5.0)
                for sgn in (-1, 1):
                    frond(q, ang + sgn * rng.uniform(0.6, 0.95), L * rng.uniform(0.3, 0.45), w * 0.8, depth - 1)

    frond(np.array([S * 0.5, S * 0.98]), -math.pi / 2 + rng.uniform(-0.1, 0.1), S * 0.9, S * 0.016, 2)
    return img


def _finish(img: Image.Image) -> Image.Image:
    img = img.resize((CELL, CELL), Image.LANCZOS)
    # gently bleed colour into transparent texels so mip-mapping does not fringe to black
    rgb = img.convert("RGB").filter(ImageFilter.MaxFilter(5))
    alpha = img.getchannel("A")
    base = Image.composite(img.convert("RGB"), rgb, alpha)
    base.putalpha(alpha)
    return base


_ATLAS_CACHE = None


TEXTURE_VERSION = "1"
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures")


def _paint_atlas() -> Image.Image:
    rng = np.random.default_rng(2024)
    cells = [_paint_broadleaf(rng, small=False), _paint_broadleaf(rng, small=True), _paint_needles(rng), _paint_cedar(rng)]
    atlas = Image.new("RGBA", (ATLAS, ATLAS), (0, 0, 0, 0))
    for i, c in enumerate(cells):
        atlas.paste(_finish(c), ((i % 2) * CELL, (i // 2) * CELL))
    return atlas


def foliage_atlas(use_cache: bool = True) -> np.ndarray:
    """(ATLAS, ATLAS, 4) uint8 RGBA, row 0 = bottom (ready for glTexImage2D).
    Painted on first launch (~2 s) and cached in assets/textures/."""
    global _ATLAS_CACHE
    if _ATLAS_CACHE is not None:
        return _ATLAS_CACHE
    path = os.path.join(CACHE_DIR, f"foliage_atlas_v{TEXTURE_VERSION}.png")
    atlas = None
    if use_cache and os.path.exists(path):
        try:
            atlas = Image.open(path).convert("RGBA")
            if atlas.size != (ATLAS, ATLAS):
                atlas = None
        except Exception:
            atlas = None
    if atlas is None:
        atlas = _paint_atlas()
        if use_cache:
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                atlas.save(path)
            except OSError:
                pass
    arr = np.asarray(atlas, dtype=np.uint8)[::-1].copy()
    _ATLAS_CACHE = arr
    return arr
