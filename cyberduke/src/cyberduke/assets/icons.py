"""Inventory icons, drawn with NumPy: neon line-art on transparent black, the way the
HUD is drawn - bright strokes with a soft glow, a few flat fills.

``generate_icon(name)`` returns a ``ICON x ICON x 4`` uint8 RGBA image (row 0 = top); the
renderer uploads them (``render.icons``). Every item in ``core.inventory`` names one of
these (``ItemDef.icon``); an unknown name gets the generic chip.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ICON = 96

_ICONS: dict[str, Callable[[], np.ndarray]] = {}

CYAN = (40, 235, 255)
MAGENTA = (255, 50, 190)
YELLOW = (255, 215, 40)
GREEN = (60, 255, 140)
ORANGE = (255, 140, 40)
VIOLET = (170, 90, 255)
WHITE = (235, 240, 245)
STEEL = (120, 130, 145)
DARK = (26, 30, 38)


def icon(name: str):
    def register(fn):
        _ICONS[name] = fn
        return fn

    return register


def available_icons() -> list[str]:
    return sorted(_ICONS)


def generate_icon(name: str) -> np.ndarray:
    return _ICONS.get(name, _ICONS["chip"])()


# ============================================================================== toolkit
def _grid() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:ICON, 0:ICON].astype(float)
    return x + 0.5, y + 0.5


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, r, mode="constant")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, pad)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, out)


class Canvas:
    """An RGB canvas with coverage, drawn with signed distances (soft 1 px edges)."""

    def __init__(self) -> None:
        self.x, self.y = _grid()
        self.rgb = np.zeros((ICON, ICON, 3))
        self.a = np.zeros((ICON, ICON))
        self.glow = np.zeros((ICON, ICON, 3))

    # distance fields -------------------------------------------------------
    def d_box(self, cx, cy, w, h, r=0.0) -> np.ndarray:
        qx = np.abs(self.x - cx) - (w / 2 - r)
        qy = np.abs(self.y - cy) - (h / 2 - r)
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        return outside + np.minimum(np.maximum(qx, qy), 0) - r

    def d_circle(self, cx, cy, r) -> np.ndarray:
        return np.hypot(self.x - cx, self.y - cy) - r

    def d_segment(self, x0, y0, x1, y1) -> np.ndarray:
        px, py = self.x - x0, self.y - y0
        dx, dy = x1 - x0, y1 - y0
        t = np.clip((px * dx + py * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
        return np.hypot(px - t * dx, py - t * dy)

    # painting ---------------------------------------------------------------
    def _paint(self, cover: np.ndarray, color, glow: float = 0.0) -> None:
        c = np.array(color, dtype=float)
        self.rgb = self.rgb * (1 - cover[..., None]) + c * cover[..., None]
        self.a = np.maximum(self.a, cover)
        if glow > 0:
            self.glow += _blur(cover, 3.0)[..., None] * c * glow

    def fill(self, d: np.ndarray, color, glow: float = 0.0) -> None:
        self._paint(np.clip(0.5 - d, 0, 1), color, glow)

    def stroke(self, d: np.ndarray, width: float, color, glow: float = 0.6) -> None:
        self._paint(np.clip(width / 2 + 0.5 - np.abs(d), 0, 1), color, glow)

    def line(self, x0, y0, x1, y1, width, color, glow: float = 0.6) -> None:
        self._paint(np.clip(width / 2 + 0.5 - self.d_segment(x0, y0, x1, y1), 0, 1), color, glow)

    def polyline(self, pts, width, color, glow: float = 0.6) -> None:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            self.line(x0, y0, x1, y1, width, color, glow)

    def finish(self) -> np.ndarray:
        rgb = self.rgb + self.glow * (1 - self.a[..., None] * 0.6)
        glow_a = np.clip(self.glow.max(axis=2) / 255.0, 0, 1) * 0.9
        a = np.clip(np.maximum(self.a, glow_a), 0, 1)
        out = np.zeros((ICON, ICON, 4), dtype=np.uint8)
        safe = np.maximum(a, 1e-6)[..., None]
        out[..., :3] = np.clip(np.where(a[..., None] > 0, rgb / np.maximum(safe, 1.0), 0), 0, 255)
        out[..., :3] = np.clip(rgb, 0, 255)
        out[..., 3] = np.clip(a * 255, 0, 255)
        return out


# ============================================================================== the icons
@icon("chip")
def _chip() -> np.ndarray:
    c = Canvas()
    c.fill(c.d_box(48, 48, 44, 44, 5), DARK)
    c.stroke(c.d_box(48, 48, 44, 44, 5), 2.5, CYAN)
    for k in range(5):
        for side in (-1, 1):
            y = 32 + k * 8
            c.line(48 + side * 22, y, 48 + side * 31, y, 2.2, CYAN, 0.3)
            c.line(y, 48 + side * 22, y, 48 + side * 31, 2.2, CYAN, 0.3)
    c.fill(c.d_box(48, 48, 18, 18, 2), CYAN, 0.8)
    return c.finish()


@icon("mod_ping")
def _mod_ping() -> np.ndarray:
    """The ping amplifier: a chip broadcasting rings."""
    c = Canvas()
    c.fill(c.d_box(48, 64, 34, 22, 4), DARK)
    c.stroke(c.d_box(48, 64, 34, 22, 4), 2.2, CYAN)
    for k in range(4):
        c.line(37 + k * 7, 75, 37 + k * 7, 83, 2.0, CYAN, 0.3)
    c.fill(c.d_circle(48, 60, 4), CYAN, 1.0)
    for r, alpha in ((14, 1.0), (24, 0.8), (34, 0.6)):
        ring = c.d_circle(48, 60, r)
        arc = (c.y < 56) & (np.abs(c.x - 48) < (c.y * 0 + r * 0.95))
        c._paint(np.clip(1.5 - np.abs(ring), 0, 1) * arc * alpha, CYAN, 0.9)
    return c.finish()


@icon("mod_quickjack")
def _mod_quickjack() -> np.ndarray:
    """The QuickJack: three fingertip jacks with cables running into a plug."""
    c = Canvas()
    c.fill(c.d_box(48, 74, 36, 14, 4), DARK)
    c.stroke(c.d_box(48, 74, 36, 14, 4), 2.2, MAGENTA)
    for k, x in enumerate((30, 48, 66)):
        top = 18 + (k % 2) * 6
        c.fill(c.d_box(x, top + 14, 10, 26, 4), DARK)
        c.stroke(c.d_box(x, top + 14, 10, 26, 4), 2.0, MAGENTA)
        c.fill(c.d_circle(x, top + 4, 2.6), YELLOW, 1.2)
        c.polyline([(x, top + 27), (x, 60), (48 + (x - 48) * 0.4, 67)], 1.8, MAGENTA, 0.5)
    for x in (38, 44, 50, 56):
        c.line(x, 81, x, 88, 2.0, YELLOW, 0.4)
    return c.finish()


@icon("mod_wraith")
def _mod_wraith() -> np.ndarray:
    """The Wraith damper: a hooded ghost half dissolved into static."""
    c = Canvas()
    body = np.minimum(c.d_circle(48, 40, 20), c.d_box(48, 60, 40, 36, 3))
    wave = (c.y > 72) & (np.sin(c.x * 0.5) * 4 + 74 < c.y)
    cover = np.clip(0.5 - body, 0, 1) * (~wave)
    c._paint(cover * 0.85, (40, 26, 70), 0.0)
    c.stroke(body, 2.2, VIOLET)
    rng = np.random.default_rng(7)
    static = (rng.random((ICON, ICON)) > 0.82) & (cover > 0.5) & (c.y > 52)
    c._paint(static.astype(float) * 0.8, (200, 160, 255), 0.2)
    for sx in (40, 56):
        c.fill(c.d_box(sx, 40, 7, 4, 1.5), CYAN, 1.3)
    return c.finish()


@icon("datastick")
def _datastick() -> np.ndarray:
    """A scuffed data stick with a hand-written label and its LED lit."""
    c = Canvas()
    body = c.d_box(44, 52, 28, 56, 6)
    c.fill(body, (34, 38, 48))
    c.stroke(body, 2.4, YELLOW)
    c.fill(c.d_box(44, 18, 18, 14, 1), STEEL)
    c.stroke(c.d_box(44, 18, 18, 14, 1), 1.5, WHITE, 0.2)
    for x in (39, 49):
        c.fill(c.d_box(x, 17, 4, 5, 0.5), DARK)
    c.fill(c.d_box(44, 58, 20, 22, 2), (210, 205, 180))
    for k in range(3):
        c.line(37, 52 + k * 6, 51 - k * 3, 52 + k * 6, 1.4, (60, 60, 90), 0.0)
    c.fill(c.d_circle(44, 76, 3.2), GREEN, 1.5)
    c.line(66, 30, 74, 22, 2.0, YELLOW, 0.8)
    c.line(70, 40, 80, 36, 2.0, YELLOW, 0.8)
    return c.finish()


@icon("data_shard")
def _data_shard() -> np.ndarray:
    """A crystal data shard."""
    c = Canvas()
    pts = [(48, 10), (70, 40), (60, 86), (36, 86), (26, 40), (48, 10)]
    inside = np.ones((ICON, ICON), dtype=bool)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        inside &= (x1 - x0) * (c.y - y0) - (y1 - y0) * (c.x - x0) >= 0
    c._paint(inside.astype(float) * 0.55, (20, 90, 60))
    c.polyline(pts, 2.4, GREEN, 0.9)
    c.polyline([(26, 40), (48, 52), (70, 40)], 1.6, GREEN, 0.5)
    c.line(48, 52, 48, 86, 1.6, GREEN, 0.5)
    c.fill(c.d_circle(48, 32, 3), WHITE, 1.0)
    return c.finish()


@icon("eddies")
def _eddies() -> np.ndarray:
    """A cred chip with the eddie sign."""
    c = Canvas()
    card = c.d_box(48, 50, 64, 42, 6)
    c.fill(card, (40, 34, 10))
    c.stroke(card, 2.4, YELLOW)
    c.fill(c.d_box(30, 44, 14, 11, 2), (200, 170, 60))
    c.stroke(c.d_circle(60, 52, 11), 2.4, YELLOW, 0.9)
    c.line(52, 48, 66, 48, 2.0, YELLOW, 0.6)
    c.line(52, 55, 66, 55, 2.0, YELLOW, 0.6)
    return c.finish()


@icon("cyberdeck")
def _cyberdeck() -> np.ndarray:
    c = Canvas()
    deck = c.d_box(48, 56, 72, 40, 5)
    c.fill(deck, (22, 26, 34))
    c.stroke(deck, 2.4, CYAN)
    c.fill(c.d_box(48, 44, 48, 12, 2), (10, 60, 70), 0.4)
    c.line(28, 44, 60, 44, 1.6, CYAN, 0.6)
    for k in range(7):
        c.fill(c.d_box(24 + k * 8, 62, 6, 5, 1), STEEL)
        c.fill(c.d_box(24 + k * 8, 70, 6, 5, 1), STEEL)
    c.polyline([(80, 48), (88, 40), (88, 22)], 2.0, MAGENTA, 0.7)
    c.fill(c.d_circle(88, 20, 3), MAGENTA, 1.0)
    return c.finish()


@icon("badge")
def _badge() -> np.ndarray:
    c = Canvas()
    card = c.d_box(48, 50, 44, 58, 5)
    c.fill(card, (30, 40, 50))
    c.stroke(card, 2.2, CYAN)
    c.fill(c.d_circle(48, 40, 9), STEEL)
    c.fill(c.d_box(48, 58, 22, 9, 4), STEEL)
    c.fill(c.d_box(48, 72, 32, 4, 1), ORANGE, 0.6)
    return c.finish()


@icon("cigarettes")
def _cigarettes() -> np.ndarray:
    c = Canvas()
    pack = c.d_box(48, 56, 40, 52, 3)
    c.fill(pack, (120, 30, 30))
    c.stroke(pack, 2.0, WHITE, 0.2)
    for x in (38, 48, 58):
        c.fill(c.d_box(x, 26, 7, 16, 2), (240, 235, 220))
        c.fill(c.d_box(x, 33, 7, 5, 1), ORANGE)
    c.fill(c.d_circle(48, 58, 8), (240, 235, 220))
    return c.finish()


@icon("phone")
def _phone() -> np.ndarray:
    c = Canvas()
    body = c.d_box(48, 48, 36, 66, 6)
    c.fill(body, DARK)
    c.stroke(body, 2.2, STEEL, 0.2)
    c.fill(c.d_box(48, 46, 28, 48, 2), (20, 70, 110), 0.5)
    c.line(34, 30, 62, 62, 1.2, WHITE, 0.1)  # the crack
    return c.finish()


@icon("keys")
def _keys() -> np.ndarray:
    c = Canvas()
    c.stroke(c.d_circle(38, 38, 12), 3.0, STEEL, 0.2)
    c.line(46, 46, 74, 74, 5.0, YELLOW, 0.3)
    c.line(66, 66, 72, 60, 3.0, YELLOW, 0.3)
    c.line(72, 72, 78, 66, 3.0, YELLOW, 0.3)
    return c.finish()


@icon("parcel")
def _parcel() -> np.ndarray:
    c = Canvas()
    box = c.d_box(48, 52, 60, 46, 2)
    c.fill(box, (150, 110, 70))
    c.stroke(box, 2.0, (220, 190, 140), 0.2)
    c.line(48, 29, 48, 75, 2.5, (240, 230, 200), 0.0)
    c.line(18, 52, 78, 52, 2.5, (240, 230, 200), 0.0)
    return c.finish()


@icon("knife")
def _knife() -> np.ndarray:
    c = Canvas()
    c.fill(c.d_box(40, 42, 56, 20, 2), STEEL, 0.3)
    c.fill(c.d_box(76, 48, 18, 8, 2), (60, 40, 30))
    return c.finish()


@icon("stim")
def _stim() -> np.ndarray:
    c = Canvas()
    c.fill(c.d_box(48, 48, 16, 50, 4), (40, 20, 50))
    c.stroke(c.d_box(48, 48, 16, 50, 4), 2.0, MAGENTA)
    c.fill(c.d_box(48, 54, 10, 26, 2), MAGENTA, 0.9)
    c.line(48, 22, 48, 10, 2.0, STEEL, 0.2)
    return c.finish()
