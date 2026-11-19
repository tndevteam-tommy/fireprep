"""The faces: painted on the front of the head, where ``animation.anatomy`` puts the eyes,
the nose and the mouth.

A face is painted for a skin tone (the skin is that colour; the whites of the eyes, the
teeth and the ink are not), a mark (``animation.look.MARKS``: circuit lines, a tattoo,
neon eye paint, a scar) and an expression - the mouth and the brows of the moment:

* ``neutral`` (lips closed), ``smile``, ``grimace`` (teeth gritted), ``frown``;
* ``talk`` and ``talk_o`` (the mouth open on a vowel: talking cycles through them),
  ``mumble`` (lips parted, lopsided: muttering to oneself), ``shout``;
* ``eat`` (open wide for a bite), ``pursed`` (the lips pushed out: a drag on a cigarette,
  a sip from a can).

They are shaded like faces - cheeks with some colour in them, shadow in the eye sockets
and under the nose, light on the forehead, pores - rather than flat. Faces are made on
demand (``face_name`` -> ``face_image``; ``render.textures`` uploads the ones that are
seen): a handful of tones x marks x expressions is hundreds. The plain tone's neutral,
smiling and grimacing faces are registered textures too ("skin", "skin_smile",
"skin_grimace").
"""

from __future__ import annotations

# (before the import of ``assets.textures``, which imports these back from here)
FACES = ("skin", "skin_smile", "skin_grimace")  # the plain tone's: neutral, smiling, gritted
MOUTH_RGB = (120, 56, 56)  # the line of the lips

from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402

from cyberduke.animation import anatomy  # noqa: E402
from cyberduke.assets.textures import SIZE, Image, blur, coords, finish, noise, texture  # noqa: E402

RGB = tuple[float, float, float]

EXPRESSIONS = (
    "neutral", "smile", "grimace", "frown", "talk", "talk_o", "mumble", "shout", "eat", "pursed",
)  # fmt: skip
FACE_PROUD = 1.012  # (``animation.rig``: the face's box is this much bigger than the head's)
FW = anatomy.HEAD_SIZE[0] * FACE_PROUD  # the width of the picture (metres)
FH = anatomy.HEAD_SIZE[1] * FACE_PROUD  # ... and its height
SKIN: RGB = (206, 168, 144)  # (the flesh texture's colour: faces match necks and hands)
TEETH = (238, 234, 224)
SCLERA = (236, 232, 226)
K = 1.3  # the mouth a little bigger than life (so that it reads at a distance)

_Y, _X = coords()
XM = (_X + 0.5) / SIZE * FW - FW / 2.0  # metres across the face (its left is +x)
YM = FH / 2.0 - (_Y + 0.5) / SIZE * FH  # metres up from the middle of the head
PX = FW / SIZE  # metres a pixel


def col(x: float) -> float:
    """The picture's column of ``x`` metres across the face."""
    return (x / FW + 0.5) * SIZE - 0.5


def row(y: float) -> float:
    """The picture's row of ``y`` metres up from the middle of the head."""
    return (0.5 - y / FH) * SIZE - 0.5


class Canvas:
    """A window of the face being painted: the picture and its tint mask (1 where the
    skin's tone applies), and the metre coordinates of its pixels (``x``, ``y``)."""

    def __init__(self, img: np.ndarray, tint: np.ndarray, x0=-1.0, x1=1.0, y0=-1.0, y1=1.0):
        c0, c1 = max(0, int(col(x0)) - 1), min(SIZE, int(col(x1)) + 2)
        r0, r1 = max(0, int(row(y1)) - 1), min(SIZE, int(row(y0)) + 2)
        self.img, self.tint = img[r0:r1, c0:c1], tint[r0:r1, c0:c1]
        self.x, self.y = XM[r0:r1, c0:c1], YM[r0:r1, c0:c1]

    def blob(self, x: float, y: float, rx: float, ry: float) -> np.ndarray:
        """A soft round blob (1 in the middle, 0 outside) at (x, y) metres."""
        d = ((self.x - x) / rx) ** 2 + ((self.y - y) / ry) ** 2
        return np.clip(1.0 - d, 0.0, 1.0) ** 1.5

    def ellipse(self, x: float, y: float, rx: float, ry: float) -> np.ndarray:
        return ((self.x - x) / rx) ** 2 + ((self.y - y) / ry) ** 2 < 1.0

    def line(self, points, width: float) -> np.ndarray:
        """An anti-aliased polyline through metre points, ``width`` metres wide."""
        out = np.zeros(self.x.shape)
        for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
            dx, dy = x1 - x0, y1 - y0
            t = ((self.x - x0) * dx + (self.y - y0) * dy) / max(dx * dx + dy * dy, 1e-12)
            t = np.clip(t, 0.0, 1.0)
            d = np.hypot(self.x - (x0 + t * dx), self.y - (y0 + t * dy))
            out = np.maximum(out, np.clip((width / 2.0 - d) / PX + 0.5, 0.0, 1.0))
        return out

    def paint(self, mask, rgb, alpha: float = 1.0, skin: bool = False) -> None:
        """Paints ``rgb`` where ``mask`` (0..1, or bool) says; ``skin``: it takes the tone."""
        if mask.dtype == bool:
            if not mask.any():
                return
            if alpha >= 1.0:
                self.img[mask] = rgb
                self.tint[mask] = 1.0 if skin else 0.0
                return
            m = mask * alpha
        else:
            m = mask * alpha
        self.img[:] = self.img * (1.0 - m[..., None]) + np.asarray(rgb, dtype=float) * m[..., None]
        self.tint[:] = self.tint * (1.0 - m) + (1.0 if skin else 0.0) * m


# ----------------------------------------------------------------------------- the parts
@lru_cache(maxsize=1)
def plain_skin() -> np.ndarray:
    """The skin with nothing on it: the flesh of the rest of the body (``flesh``: the
    head, the ears, the nose, the neck, the hands) - the same colours, the same mottling."""
    from cyberduke.assets.textures import generate_texture

    img = generate_texture("flesh").astype(float)
    img.flags.writeable = False
    return img


FACE_EDGE = (0.2, 0.36)  # (unit z of the head's surface): the face's features fade out ..


def _feature_weight(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1 over the face proper, fading to 0 towards where the face meets the sides, the
    top and the bottom of the head (by how far forward the head's surface is there:
    ``shapes.FACE_FRONT`` is where the face stops)."""
    half_w = np.zeros(x.shape)
    out = np.zeros(x.shape)
    for idx in np.ndindex(x.shape):
        out[idx] = anatomy.surface_z(float(x[idx]), float(y[idx])) / anatomy.HEAD_SIZE[2]
        t = min(1.0, max(0.0, float(y[idx]) / anatomy.HEAD_SIZE[1] + 0.5))
        half_w[idx] = 0.5 * anatomy.width(t) * anatomy.rounding(t) * anatomy.HEAD_SIZE[0]
    out[np.abs(x) >= half_w] = -1.0  # (off the side of the head)
    lo, hi = FACE_EDGE
    u = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _skin(c: Canvas) -> None:
    """The skin, shaded: pores, colour in the cheeks and the nose, light on the forehead
    and the chin, shadow in the eye sockets, under the nose and the lower lip, and the
    face turning away at its sides."""
    img = c.img
    img[:] = plain_skin()
    plain = img.copy()
    img *= (0.975 + 0.05 * noise(7002, beta=0.4, low=28))[..., None]  # pores
    warm = np.array((225, 140, 130), dtype=float)

    def flush(blob: np.ndarray, k: float) -> None:
        img[:] = img * (1.0 - k * blob[..., None]) + warm * k * blob[..., None]

    for side in (-1.0, 1.0):
        flush(c.blob(side * 0.082, anatomy.MOUTH_LINE + 0.065, 0.035, 0.03), 0.28)
        img *= (1.0 - 0.12 * c.blob(side * anatomy.EYE_X, anatomy.EYE_LINE + 0.004, 0.045, 0.03))[
            ..., None
        ]
    flush(c.blob(0.0, anatomy.NOSE_Y - 0.02, 0.02, 0.02), 0.18)
    img *= (1.0 + 0.07 * c.blob(0.0, anatomy.HAIRLINE_FRONT - 0.04, 0.07, 0.045))[..., None]
    img *= (1.0 + 0.05 * c.blob(0.0, anatomy.MOUTH_LINE - 0.06, 0.03, 0.02))[..., None]
    img *= (1.0 - 0.14 * c.blob(0.0, anatomy.NOSE_Y - 0.05, 0.022, 0.008))[..., None]
    img *= (1.0 - 0.12 * c.blob(0.0, anatomy.MOUTH_LINE - 0.022, 0.018, 0.008))[..., None]
    for side in (-1.0, 1.0):  # the lines from the nose to the corners of the mouth
        fold = c.line(
            [(side * 0.022, anatomy.NOSE_Y - 0.03), (side * 0.036, anatomy.MOUTH_LINE - 0.006)],
            0.004,
        )
        img *= (1.0 - 0.07 * blur(fold, 1.0))[..., None]
    # Towards the edges of the face (where the face meets the rest of the head) all of
    # that fades back to the plain skin - the flesh of the head, the ears, the neck and
    # the hands - so there is no line where the face stops (the lights turn the face
    # away at its sides, not the paint: it used to be darkened there, and a seam showed)
    w = _feature_weight(c.x, c.y)[..., None]
    img[:] = plain + (img - plain) * w
    c.tint[:] = 1.0
    for side in (-1.0, 1.0):  # nostrils under the modelled nose
        nostril = c.ellipse(side * 0.011, anatomy.NOSE_Y - 0.038, 0.0055, 0.0035)
        c.paint(blur(nostril.astype(float), 0.8), (110, 66, 58), 0.85, skin=True)


def _mark(c: Canvas, mark: str) -> None:
    e, ex = anatomy.EYE_LINE, anatomy.EYE_X
    if mark == "circuit":  # implanted traces under the skin of the left cheek and temple
        traces = [
            [(0.05, e - 0.04), (0.08, e - 0.04), (0.095, e - 0.055), (0.095, e - 0.09)],
            [(0.07, e - 0.05), (0.07, e - 0.075), (0.085, e - 0.09)],
            [(0.1, e + 0.03), (0.115, e + 0.045), (0.115, e + 0.08)],
        ]
        lines = np.zeros(c.x.shape)
        for t in traces:
            lines = np.maximum(lines, c.line(t, 0.0028))
            for x, y in (t[0], t[-1]):
                lines = np.maximum(lines, c.ellipse(x, y, 0.004, 0.004).astype(float))
        c.paint(blur(lines, 1.0), (40, 180, 220), 0.5)
        c.paint(lines, (90, 235, 255))
    elif mark == "tattoo":  # three ink bars under the right eye and a drop
        for k in range(3):
            x = -ex - 0.012 + 0.012 * k
            c.paint(c.line([(x, e - 0.03), (x, e - 0.055)], 0.004), (25, 25, 35))
        c.paint(c.ellipse(-ex - 0.022, e - 0.07, 0.004, 0.006), (25, 25, 35))
    elif mark == "paint":  # a neon band across both eyes, temple to temple
        band = (np.abs(c.y - (e + 0.004)) < 0.014) & (np.abs(c.x) < 0.108)
        c.paint(blur(band.astype(float), 1.2), (230, 40, 170), 0.85)
    elif mark == "scar":
        scar = c.line([(ex + 0.02, e + 0.06), (ex - 0.005, e + 0.01), (ex - 0.02, e - 0.05)], 0.004)
        c.paint(scar, (230, 190, 175), 0.9, skin=True)


@lru_cache(maxsize=8)
def _base(mark: str) -> tuple[np.ndarray, np.ndarray]:
    """The skin with its mark, before the expression (the same for every expression)."""
    img, tint = np.zeros((SIZE, SIZE, 3)), np.ones((SIZE, SIZE))
    c = Canvas(img, tint)
    _skin(c)
    _mark(c, mark)
    img.flags.writeable = tint.flags.writeable = False
    return img, tint


def _eyes(img: np.ndarray, tint: np.ndarray, expression: str, iris: RGB) -> None:
    """Almond eyes: the whites, a ringed iris with its pupil and a catch light, the lids
    (the upper one heavy with lashes), a crease above - squinting in a smile, narrowed in
    a frown."""
    e, ex = anatomy.EYE_LINE, anatomy.EYE_X
    half_w = 0.027
    up, down = 0.012, 0.009
    if expression == "smile":
        down = 0.005  # the cheeks push the lower lids up
    if expression in ("frown", "grimace", "shout"):
        up = 0.009
    for side in (-1.0, 1.0):
        cx = side * ex
        c = Canvas(img, tint, cx - 0.034, cx + 0.034, e - 0.018, e + 0.026)
        u = np.clip(np.abs(c.x - cx) / half_w, 0.0, 1.0)
        skew = 0.002 * side * (c.x - cx) / half_w  # (the outer corner a touch higher)
        top = e + up * (1.0 - u**2) ** 0.8 + skew
        bottom = e - down * (1.0 - u**2) ** 0.9 + skew
        inside = (np.abs(c.x - cx) < half_w) & (c.y < top) & (c.y > bottom)
        c.paint(inside, SCLERA)
        ring = (c.x - cx) ** 2 + (c.y - e) ** 2
        c.paint(inside & (ring < 0.0115**2), tuple(v * 0.7 for v in iris))
        c.paint(inside & (ring < 0.0095**2), iris)
        c.paint(inside & (ring < 0.0045**2), (12, 10, 14))
        shine = ((c.x - cx - 0.003) ** 2 + (c.y - e - 0.004) ** 2) < 0.0022**2
        c.paint(inside & shine, (255, 255, 255))
        c.paint((np.abs(c.y - top) < 0.0022) & (np.abs(c.x - cx) < half_w * 1.05), (35, 22, 20))
        low = (np.abs(c.y - bottom) < 0.0009) & (np.abs(c.x - cx) < half_w * 0.9)
        c.paint(low, (120, 80, 70), 0.7)
        crease = (np.abs(c.y - (top + 0.006)) < 0.0012) & (np.abs(c.x - cx) < half_w * 0.85)
        c.paint(crease, (150, 105, 90), 0.4, skin=True)


def _brows(img: np.ndarray, tint: np.ndarray, expression: str) -> None:
    """Brows of hair strokes, tapering outwards: arched, raised in a smile, knotted down
    at the nose in a frown, one up in a mumble."""
    b, ex = anatomy.BROW_LINE, anatomy.EYE_X
    grain = _grain()
    for side in (-1.0, 1.0):
        inner, outer = side * (ex - 0.028), side * (ex + 0.032)
        c = Canvas(
            img, tint, min(inner, outer) - 0.01, max(inner, outer) + 0.01, b - 0.02, b + 0.02
        )
        lift_in, lift_out, arch = 0.0, -0.004, 0.006
        if expression == "smile":
            lift_in, lift_out = 0.004, 0.0
        elif expression in ("frown", "grimace", "shout"):
            lift_in, lift_out, arch = -0.011, 0.001, 0.003
        elif expression == "mumble" and side > 0:
            lift_in, lift_out = 0.006, 0.004
        mid = ((inner + outer) / 2.0, b + arch + (lift_in + lift_out) / 2.0)
        brow = np.maximum(
            c.line([(inner, b + lift_in), mid], 0.0075), c.line([mid, (outer, b + lift_out)], 0.005)
        )
        c.paint(brow * (0.75 + 0.25 * grain[: brow.shape[0], : brow.shape[1]]), (58, 40, 30), 0.95)


@lru_cache(maxsize=1)
def _grain() -> np.ndarray:
    return noise(7010, beta=0.2, low=30)


def _mouth(img: np.ndarray, tint: np.ndarray, expression: str) -> None:
    """The lips (rose, the lower fuller and lit), closed along a line of ``MOUTH_RGB`` or
    open on the dark of the mouth, teeth and a tongue."""
    m = anatomy.MOUTH_LINE
    c = Canvas(img, tint, -0.06, 0.06, m - 0.045, m + 0.03)
    dx, yy = c.x, c.y
    upper, lower = (176, 98, 94), (198, 118, 110)
    inside = (70, 24, 30)

    def closed(half: float, curve: float) -> None:
        line_y = m + curve * (dx / half) ** 2
        span = np.abs(dx) < half
        w = np.clip(1 - (dx / half) ** 2, 0.0, 1.0)
        c.paint(span & (line_y < yy) & (line_y + 0.0045 * w + 0.001 > yy), upper, skin=True)
        lo = span & (line_y > yy) & (line_y - 0.0075 * w - 0.0005 < yy)
        c.paint(lo, lower, skin=True)
        c.paint(lo & (np.abs(yy - (line_y - 0.004)) < 0.0012), (225, 160, 150), 0.6, skin=True)
        c.paint(span & (np.abs(yy - line_y) < 0.0011), MOUTH_RGB)

    def opened(half: float, h: float, cx: float = 0.0, teeth: bool = True, tongue: bool = True,
               drop: float = 0.0) -> None:  # fmt: skip
        u = np.clip(np.abs(dx - cx) / half, 0.0, 1.0)
        top = m + 0.3 * h * (1 - u**2) ** 0.5
        bottom = m - (0.7 * h + drop) * (1 - u**2) ** 0.6
        hole = (np.abs(dx - cx) < half) & (top > yy) & (bottom < yy)
        lips_out = (np.abs(dx - cx) < half * 1.12) & (top + 0.0045 > yy) & (bottom - 0.007 < yy)
        c.paint(lips_out & (m <= yy), upper, skin=True)
        c.paint(lips_out & (m > yy), lower, skin=True)
        c.paint(hole, inside)
        if teeth:
            c.paint(hole & (top - 0.0035 < yy), TEETH)
        if tongue:
            c.paint(hole & (bottom + 0.0035 > yy) & (np.abs(dx - cx) < half * 0.6), (170, 70, 80))
        c.paint((np.abs(dx - cx) < half) & (np.abs(yy - top) < 0.0009), MOUTH_RGB)
        c.paint((np.abs(dx - cx) < half) & (np.abs(yy - bottom) < 0.0009), MOUTH_RGB)

    if expression == "smile":
        half = 0.03 * K
        line = m + 0.008 * (dx / half) ** 2 - 0.002
        grin = (np.abs(dx) < half) & (line > yy)
        grin &= m - 0.009 * np.clip(1 - (dx / half) ** 2, 0.0, 1.0) ** 0.7 < yy
        c.paint(
            (np.abs(dx) < half * 1.08) & (line + 0.004 > yy) & (m - 0.015 < yy), lower, skin=True
        )
        c.paint(grin, inside)
        c.paint(grin & (line - 0.004 < yy), TEETH)
        c.paint((np.abs(dx) < half) & (np.abs(yy - line) < 0.0011), MOUTH_RGB)
        for side in (-1.0, 1.0):  # dimples
            c.paint(c.ellipse(side * (half + 0.004), m + 0.007, 0.0016, 0.003), MOUTH_RGB, 0.7)
    elif expression == "grimace":
        half = 0.03 * K
        top = m + 0.004 + 0.004 * (dx / half) ** 2
        bottom = m - 0.005 - 0.005 * (dx / half) ** 2
        teeth = (np.abs(dx) < half) & (top > yy) & (bottom < yy)
        c.paint(teeth, TEETH)
        gaps = (np.abs(((dx + 0.0025) % 0.005) - 0.0025) < 0.0004) | (
            np.abs(yy - (top + bottom) / 2) < 0.0005
        )
        c.paint(teeth & gaps, (150, 130, 120))
        edges = (np.abs(yy - top) < 0.0012) | (np.abs(yy - bottom) < 0.0012)
        c.paint((np.abs(dx) < half) & edges, MOUTH_RGB)
    elif expression == "frown":
        closed(0.025 * K, -0.006)
    elif expression == "talk":
        opened(0.02 * K, 0.014 * K)
    elif expression == "talk_o":
        opened(0.011 * K, 0.014 * K, teeth=False)
    elif expression == "mumble":
        opened(0.016 * K, 0.0045 * K, cx=0.004, teeth=False, tongue=False)
    elif expression == "shout":
        opened(0.026 * K, 0.02 * K, drop=0.006)
    elif expression == "eat":
        opened(0.021 * K, 0.018 * K, drop=0.005)
    elif expression == "pursed":
        rx, ry = 0.0135 * K, 0.0095 * K
        pout = c.ellipse(0.0, m, rx, ry)
        c.paint(pout, upper, skin=True)
        c.paint(pout & (m > yy), lower, skin=True)
        c.paint(c.ellipse(0.0, m, 0.0035, 0.0022), inside)
        ridges = pout & (np.abs(np.sin(np.arctan2(yy - m, dx) * 6.0)) < 0.25)
        c.paint(ridges, MOUTH_RGB, 0.5)
        c.paint(np.abs(((dx / rx) ** 2 + ((yy - m) / ry) ** 2) - 1.0) < 0.12, MOUTH_RGB)
    else:  # neutral
        closed(0.026 * K, -0.0012)


@lru_cache(maxsize=64)
def _expression(mark: str, expression: str) -> tuple[np.ndarray, np.ndarray]:
    """The untoned face for a mark and an expression."""
    base_img, base_tint = _base(mark)
    img, tint = base_img.copy(), base_tint.copy()
    _brows(img, tint, expression)
    _eyes(img, tint, expression, (92, 62, 42))
    _mouth(img, tint, expression)
    img.flags.writeable = tint.flags.writeable = False
    return img, tint


def face_image(tone: RGB = (1.0, 1.0, 1.0), mark: str = "", expression: str = "neutral") -> Image:
    """The face for a skin ``tone`` (a multiplier of the plain skin), a ``mark`` and an
    ``expression`` (``EXPRESSIONS``)."""
    if expression not in EXPRESSIONS:
        expression = "neutral"
    img, tint = _expression(mark, expression)
    t = np.array(tone, dtype=float)
    return finish(img * (tint[..., None] * t + (1.0 - tint[..., None])))


def flesh_name(tone: RGB) -> str:
    """The name of the flesh texture in a skin ``tone`` (``render.textures`` makes it on
    first use): the head, neck and hands are drawn with it the way the face is drawn with
    its own - the tone in the picture, lit alike - so the two match wherever they meet."""
    r, g, b = (round(float(c), 2) for c in tone)
    return f"flesh|{r:.2f},{g:.2f},{b:.2f}"


def flesh_from_name(name: str) -> Image:
    _, tone = name.split("|")
    t = np.array([float(c) for c in tone.split(",")])
    return finish(plain_skin() * t)


def face_name(tone: RGB, mark: str = "", expression: str = "neutral") -> str:
    """The name the renderer knows a face by (``render.textures`` makes it on first use)."""
    r, g, b = (round(float(c), 2) for c in tone)
    return f"face|{r:.2f},{g:.2f},{b:.2f}|{mark}|{expression}"


def face_from_name(name: str) -> Image:
    _, tone, mark, expression = name.split("|")
    r, g, b = (float(c) for c in tone.split(","))
    return face_image((r, g, b), mark, expression)


@texture("skin")
def _skin_neutral() -> Image:
    return face_image(expression="neutral")


@texture("skin_smile")
def _skin_smile() -> Image:
    return face_image(expression="smile")


@texture("skin_grimace")
def _skin_grimace() -> Image:
    return face_image(expression="grimace")


# =============================================================================== hair
def _waves(seed: int, k0: int, k1: int, slope: float, x: np.ndarray) -> np.ndarray:
    """A periodic 1D pattern along ``x`` (0 .. SIZE): random waves of frequencies
    ``k0`` .. ``k1`` (times round the texture), amplitudes falling as k^-``slope``;
    0 .. 1."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x, dtype=float)
    for k in range(k0, k1 + 1):
        out = out + k**-slope * np.cos(2.0 * np.pi * k * x / SIZE + rng.uniform(0, 2 * np.pi))
    out -= out.min()
    return out / max(out.max(), 1e-9)


@texture("hair_strands")
def _hair_strands() -> Image:
    """Hair (takes its colour from the person): strands running the way it grows - up
    the texture (``v``) - gathered into locks with dark partings between them, each
    strand a little lighter or darker than the next, wavering a little as it goes."""
    y, x = coords()
    phase = 2.0 * np.pi * y / SIZE
    sway = 0.9 * np.sin(phase * 2.0) + 0.4 * np.sin(phase * 5.0 + 1.3)
    xs = (x + sway) % SIZE
    strands = _waves(7101, 18, 64, 0.4, xs)  # fine: every strand its own shade
    clumps = _waves(7102, 4, 12, 1.0, xs)  # coarse: the locks the hair falls in
    partings = np.clip((clumps - 0.18) / 0.3, 0.0, 1.0) ** 0.7  # (dark between the locks)
    img = 60.0 + 150.0 * partings * (0.55 + 0.45 * strands)
    # (no band of sheen painted in: the light's own highlight moves over the hair)
    img = img * (0.93 + 0.07 * _waves(7103, 30, 64, 0.2, xs))
    return finish(np.repeat(np.clip(img, 0.0, 255.0)[..., None], 3, axis=2))


@texture("stubble")
def _stubble() -> Image:
    """A scalp shaved to stubble: the skin, peppered finely with the hair's cut ends (takes
    the hair's colour mixed with the skin's)."""
    rng = np.random.default_rng(7111)
    ends = rng.random((SIZE, SIZE))
    dots = np.clip((ends - 0.55) / 0.25, 0.0, 1.0)
    dots = 0.6 * dots + 0.4 * blur(dots, 0.6)
    img = 225.0 - 95.0 * dots
    img *= 0.96 + 0.04 * noise(7112, beta=2.0, low=1)
    return finish(np.repeat(img[..., None], 3, axis=2))


__all__ = [
    "EXPRESSIONS", "FACES", "MOUTH_RGB", "face_from_name", "face_image", "face_name",
    "flesh_from_name", "flesh_name",
]  # fmt: skip
