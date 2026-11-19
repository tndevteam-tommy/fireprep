"""Neon signs with words on them: the Neon Moth's name over its door and in its window,
OPEN, the METRO over the station's doorway, the station's name on the platform, the
gates' NO PASS NO ENTRY.

A sign is drawn for the board it goes on: the board is wide and low (a 3.6 x 0.44 m strip
over a door), the texture square - so the words are laid out for the board's
``aspect`` (width over height) and stretched back to the square, and on the board they
come out the right shape. The letters are the homes' little pixel font
(``assets.homes``), each pixel a blob of tube with its halo of light round it, on a dark
panel in a tube border.

Registered through ``assets.textures.texture`` (imported at the end of that module).
"""

from __future__ import annotations

import numpy as np

from cyberduke.assets.homes import _GLYPHS
from cyberduke.assets.textures import SIZE, Image, _canvas, blur, finish, noise, texture

RGB = tuple[float, float, float]
PINK: RGB = (255, 40, 190)
CYAN: RGB = (40, 235, 255)
AMBER: RGB = (255, 180, 40)
RED: RGB = (255, 50, 50)
WHITE: RGB = (235, 240, 255)


def _letters(text: str) -> np.ndarray:
    """The words in the pixel font: 1 where a letter's pixel is (5 rows high)."""
    cols = max(1, len(text) * 4 - 1)
    out = np.zeros((5, cols))
    for k, ch in enumerate(text.upper()):
        for r, line in enumerate(_GLYPHS.get(ch, _GLYPHS[" "])):
            for c, bit in enumerate(line):
                if bit == "1":
                    out[r, k * 4 + c] = 1.0
    return out


def _place(mask: np.ndarray, canvas: np.ndarray, top: float, bottom: float, margin: float) -> None:
    """Scales ``mask`` (the words) into the rows ``top``..``bottom`` (fractions of the
    height) of ``canvas`` (the sign laid out at its board's shape), centred across it with
    ``margin`` (a fraction of the width) either side at least - keeping the letters'
    pixels square on the board."""
    h, w = canvas.shape
    rows = max(1, int((bottom - top) * h))
    px = rows / 5.0  # a font pixel, in canvas pixels
    width = mask.shape[1] * px
    room = w * (1.0 - 2.0 * margin)
    if width > room:  # too long for the board: smaller
        px = room / mask.shape[1]
        width = room
    y0 = int(top * h + ((bottom - top) * h - 5 * px) / 2.0)
    x0 = int((w - width) / 2.0)
    for r in range(5):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                a, b = int(y0 + r * px), int(y0 + (r + 1) * px)
                e, g = int(x0 + c * px), int(x0 + (c + 1) * px)
                # (a tube is thinner than its pixel: a gap between neighbouring blobs)
                inset = max(0, int(px * 0.12))
                canvas[
                    a + inset : max(a + inset + 1, b - inset),
                    e + inset : max(e + inset + 1, g - inset),
                ] = 1.0


def neon_sign(
    lines: list[tuple[str, RGB]], aspect: float, border: RGB | None = None, seed: int = 0,
    panel: RGB = (12, 8, 20),
) -> Image:  # fmt: skip
    """A dark panel with ``lines`` of words (each its own colour) in neon tube, laid out
    for a board ``aspect`` times as wide as it is high, inside a ``border`` tube."""
    h = SIZE
    w = max(SIZE, int(round(SIZE * aspect)))  # (laid out at the board's shape)
    n = len(lines)
    glow = np.zeros((h, w, 3))
    tube = np.zeros((h, w, 3))
    lit = np.zeros((h, w))
    pad = 0.16
    for k, (text, colour) in enumerate(lines):
        mask = np.zeros((h, w))
        top = pad + (1.0 - 2 * pad) * k / n
        bottom = pad + (1.0 - 2 * pad) * (k + 1) / n
        _place(_letters(text), mask, top + 0.02, bottom - 0.02, 0.06)
        c = np.array(colour, dtype=float)
        tube += mask[..., None] * (0.55 * c + 115)
        glow += mask[..., None] * c
        lit = np.maximum(lit, mask)
    if border is not None:
        mask = np.zeros((h, w))
        t = max(2, int(h * 0.045))
        e = max(2, int(h * 0.06))
        mask[e : e + t, e : w - e] = 1.0
        mask[h - e - t : h - e, e : w - e] = 1.0
        tb = max(2, int(t * w / h))  # (the sides: as thick once squeezed back to the square)
        mask[e : h - e, e : e + tb] = 1.0
        mask[e : h - e, w - e - tb : w - e] = 1.0
        c = np.array(border, dtype=float)
        tube = tube * (1 - mask[..., None]) + mask[..., None] * (0.55 * c + 115)
        glow += mask[..., None] * c
        lit = np.maximum(lit, mask)
    # back to the square texture (the board stretches it out again)
    cols = (np.arange(SIZE) * w / SIZE).astype(int)
    tube, glow, lit = tube[:, cols], glow[:, cols], lit[:, cols]
    img = _canvas(panel) * (0.8 + 0.2 * noise(601 + seed, beta=2.0, low=1))[..., None]
    halo = np.stack([blur(glow[..., i], 2.5) for i in range(3)], axis=-1)
    img += halo * 0.75
    img = img * (1 - lit[..., None]) + tube
    return finish(img)


@texture("sign_neon_moth")
def _sign_neon_moth() -> Image:
    """Over the bar's door: NEON MOTH in pink tube, a cyan border (a 3.6 x 0.44 m board)."""
    return neon_sign([("NEON MOTH", PINK)], aspect=3.6 / 0.44, border=CYAN, seed=1)


@texture("sign_moth_window")
def _sign_moth_window() -> Image:
    """In the bar's window: NEON over MOTH, pink, and BAR in amber under them."""
    return neon_sign(
        [("NEON", PINK), ("MOTH", PINK), ("BAR", AMBER)], aspect=1.2 / 0.75, seed=2,
        panel=(4, 3, 8),
    )  # fmt: skip


@texture("sign_open")
def _sign_open() -> Image:
    """OPEN in amber inside a cyan border."""
    return neon_sign([("OPEN", AMBER)], aspect=0.84 / 0.42, border=CYAN, seed=3, panel=(4, 3, 8))


@texture("sign_metro")
def _sign_metro() -> Image:
    """Over the station's doorway: METRO in cyan, a white border."""
    return neon_sign([("METRO", CYAN)], aspect=2.4 / 0.4, border=WHITE, seed=4)


@texture("sign_station")
def _sign_station() -> Image:
    """On the platform: the station's name - NEON DISTRICT - in white over the line's
    cyan, on the dark blue of the metro's signs."""
    return neon_sign(
        [("NEON DISTRICT", WHITE), ("LINE 3 EAST-WEST", CYAN)], aspect=3.0 / 0.5, seed=5,
        panel=(10, 18, 50),
    )  # fmt: skip


@texture("sign_no_pass")
def _sign_no_pass() -> Image:
    """Over the ticket gates: NO PASS NO ENTRY, red."""
    return neon_sign(
        [("NO PASS", RED), ("NO ENTRY", RED)], aspect=1.6 / 0.36, seed=6, panel=(20, 6, 8)
    )


__all__ = ["neon_sign"]
