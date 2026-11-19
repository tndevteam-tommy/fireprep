"""Procedural RGB textures (128 x 128).

Each generator returns a ``(SIZE, SIZE, 3)`` uint8 array in image orientation (row 0 is the
*top* of the picture; the uploader flips it for OpenGL). Uploading is the job of
``cyberduke.render.textures``; keeping the two apart lets the generators be unit-tested
headlessly.

Two kinds of texture:

* **Tiled surface materials** (``@texture(name, tile=metres)``) - walls, floors, metal,
  grating, moss... The renderer repeats them across each surface at ``tile`` metres per
  copy, so they must be 100% seamless. Everything in them is built from periodic parts:
  noise filtered in the frequency domain, blurs and relief lighting that wrap around the
  edges, and patterns whose period divides the texture size (``tests/test_textures.py``
  checks every edge).
* **Pictures** (no ``tile``) - screens, signs, the crate and can artwork, the face: one
  copy per face of whatever they are on.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SIZE = 128
Image = np.ndarray
RGB = tuple[float, float, float]

_GENERATORS: dict[str, Callable[[], Image]] = {}
_TILES: dict[str, float] = {}


def texture(
    name: str, tile: float | None = None
) -> Callable[[Callable[[], Image]], Callable[[], Image]]:
    """Registers a generator. ``tile`` (metres) marks a seamless surface material."""

    def register(fn: Callable[[], Image]) -> Callable[[], Image]:
        _GENERATORS[name] = fn
        if tile is not None:
            _TILES[name] = tile
        return fn

    return register


def available_textures() -> list[str]:
    return sorted(_GENERATORS)


def tile_size(name: str) -> float | None:
    """World size (metres) of one copy of a tiled material; ``None`` for pictures."""
    return _TILES.get(name)


def tiled_textures() -> list[str]:
    return sorted(_TILES)


def generate_texture(name: str) -> Image:
    try:
        return _GENERATORS[name]()
    except KeyError:
        raise KeyError(f"unknown texture style {name!r}; known: {available_textures()}") from None


# =============================================================================== toolkit
# Every helper here is periodic: what leaves one edge comes back in at the opposite one.
def _freq_radius(size: int = SIZE) -> np.ndarray:
    f = np.fft.fftfreq(size) * size
    return np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


def noise(seed: int, beta: float = 2.0, low: float = 1.0, high: float | None = None,
          stretch: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:  # fmt: skip
    """Seamless fractal noise in [0, 1]: white noise shaped by a 1/f^beta spectrum between
    the ``low`` and ``high`` cycles-per-tile. ``stretch`` > 1 along an axis makes streaks
    along it (brushed metal, grain)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((SIZE, SIZE))
    fy = np.fft.fftfreq(SIZE)[:, None] * SIZE * stretch[1]
    fx = np.fft.fftfreq(SIZE)[None, :] * SIZE * stretch[0]
    k = np.sqrt(fx**2 + fy**2)
    amp = np.where(k >= low, 1.0 / np.maximum(k, 1e-6) ** (beta / 2.0), 0.0)
    if high is not None:
        amp = np.where(k <= high, amp, 0.0)
    out = np.real(np.fft.ifft2(np.fft.fft2(white) * amp))
    out -= out.min()
    return out / max(out.max(), 1e-9)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic Gaussian blur (per channel for colour images)."""
    k = _freq_radius()
    gauss = np.exp(-2.0 * (np.pi * sigma * k / SIZE) ** 2)
    if a.ndim == 3:
        return np.stack([blur(a[..., c], sigma) for c in range(a.shape[2])], axis=-1)
    return np.real(np.fft.ifft2(np.fft.fft2(a) * gauss))


def relief(height: np.ndarray, strength: float = 1.0, light=(-0.6, -0.7)) -> np.ndarray:
    """Lighting multiplier (around 1) from a height map lit from the top left, wrapping at
    the edges: raised parts catch light on their upper-left sides, grooves shadow."""
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    gy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    return 1.0 + strength * (gx * light[0] + gy * light[1])


def mix(a: RGB, b: RGB, t: np.ndarray) -> np.ndarray:
    """Per-pixel blend between two colours (0..255) by ``t`` in [0, 1]."""
    a_ = np.array(a, dtype=float)
    b_ = np.array(b, dtype=float)
    return a_ + (b_ - a_) * np.clip(t, 0.0, 1.0)[..., None]


def finish(img: np.ndarray) -> Image:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def grid_lines(period: int, width: int = 1, offset: int = 0, axis: int = 0) -> np.ndarray:
    """1 on rows (axis 0) / columns (axis 1) ``offset``, ``offset + period``, ..."""
    idx = (np.arange(SIZE) - offset) % period < width
    m = np.zeros((SIZE, SIZE))
    if axis == 0:
        m[idx, :] = 1.0
    else:
        m[:, idx] = 1.0
    return m


def coords() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    return y.astype(float), x.astype(float)


def wrapped_dist(a: np.ndarray, centre: float) -> np.ndarray:
    d = np.abs(a - centre) % SIZE
    return np.minimum(d, SIZE - d)


def spots(seed: int, count: int, radius: tuple[float, float]) -> np.ndarray:
    """Soft round blotches (stains, rivet heads...) scattered with wrap-around."""
    rng = np.random.default_rng(seed)
    y, x = coords()
    out = np.zeros((SIZE, SIZE))
    for _ in range(count):
        cy, cx = rng.uniform(0, SIZE, 2)
        r = rng.uniform(*radius)
        d = np.sqrt(wrapped_dist(y, cy) ** 2 + wrapped_dist(x, cx) ** 2)
        out = np.maximum(out, np.clip(1.0 - d / r, 0.0, 1.0))
    return out


def panel_seams(cells: int, groove: float = 1.5) -> np.ndarray:
    """Height map of panels: 1 on the plates, falling to 0 in grooves on the cell borders
    (``cells`` panels across the tile, so the grooves line up across tiles)."""
    period = SIZE / cells
    y, x = coords()
    dy = np.minimum(y % period, period - (y % period))
    dx = np.minimum(x % period, period - (x % period))
    d = np.minimum(dx, dy)
    return np.clip((d - 0.5) / groove, 0.0, 1.0)


def rivets(cells: int, inset: float, radius: float = 1.8) -> np.ndarray:
    """Dome-shaped rivet heads near each panel corner."""
    period = SIZE / cells
    y, x = coords()
    out = np.zeros((SIZE, SIZE))
    for oy in (inset, period - inset):
        for ox in (inset, period - inset):
            dy = wrapped_dist(y % period, oy)
            dx = wrapped_dist(x % period, ox)
            d = np.sqrt(dx**2 + dy**2)
            out = np.maximum(out, np.sqrt(np.clip(1.0 - (d / radius) ** 2, 0.0, 1.0)))
    return out


def streaks(seed: int, count: int, length: tuple[int, int], width: float = 1.2) -> np.ndarray:
    """Vertical grime drips running down from random points (wrapping top to bottom)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((SIZE, SIZE))
    y, x = coords()
    for _ in range(count):
        cx = rng.uniform(0, SIZE)
        top = rng.uniform(0, SIZE)
        ln = rng.integers(*length)
        along = (y - top) % SIZE
        fade = np.clip(1.0 - along / ln, 0.0, 1.0) * (along < ln)
        out = np.maximum(out, fade * np.clip(1.0 - wrapped_dist(x, cx) / width, 0.0, 1.0))
    return blur(out, 0.6)


def scratches(seed: int, count: int, length: tuple[float, float]) -> np.ndarray:
    """Thin bright scratch lines at random angles (wrapping)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((SIZE, SIZE))
    for _ in range(count):
        y0, x0 = rng.uniform(0, SIZE, 2)
        ang = rng.uniform(0, np.pi)
        ln = rng.uniform(*length)
        for t in np.linspace(0.0, ln, int(ln * 2)):
            yy = int(y0 + np.sin(ang) * t) % SIZE
            xx = int(x0 + np.cos(ang) * t) % SIZE
            out[yy, xx] = max(out[yy, xx], 1.0 - t / ln * 0.5)
    return out


def _canvas(rgb: RGB) -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3))
    img[:] = rgb
    return img


def stroke(points, width: float) -> np.ndarray:
    """Anti-aliased polyline (a neon tube, a crack): 1 inside, fading over a pixel."""
    y, x = coords()
    out = np.zeros((SIZE, SIZE))
    for (y0, x0), (y1, x1) in zip(points, points[1:], strict=False):
        dy, dx = y1 - y0, x1 - x0
        t = np.clip(((y - y0) * dy + (x - x0) * dx) / max(dy * dy + dx * dx, 1e-9), 0.0, 1.0)
        d = np.hypot(y - (y0 + t * dy), x - (x0 + t * dx))
        out = np.maximum(out, np.clip(width / 2.0 + 0.5 - d, 0.0, 1.0))
    return out


def cracks(seed: int, count: int, steps: int = 14) -> np.ndarray:
    """Hairline cracks: jagged random walks (drawn into a periodic canvas)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((SIZE, SIZE))
    for _ in range(count):
        y, x = rng.uniform(0, SIZE, 2)
        heading = rng.uniform(0, 2 * np.pi)
        for _ in range(steps):
            heading += rng.normal(0.0, 0.6)
            ln = rng.uniform(2.0, 6.0)
            ny, nx = y + np.sin(heading) * ln, x + np.cos(heading) * ln
            for t in np.linspace(0.0, 1.0, 8):
                out[int(y + (ny - y) * t) % SIZE, int(x + (nx - x) * t) % SIZE] = 1.0
            y, x = ny, nx
    return out


# =============================================================================== surfaces
@texture("wall", tile=2.0)
def _wall() -> Image:
    """Industrial wall cladding: four 1 m panels per tile with bevelled seams and rivets,
    mottled grime and drips running down from the joints."""
    plates = panel_seams(2, groove=2.0)
    studs = rivets(2, inset=6.0, radius=2.2)
    height = plates * 0.8 + studs * 0.6 + noise(41, beta=2.5, low=2) * 0.15
    grime = noise(42, beta=2.2, low=1)
    base = mix((17, 20, 27), (31, 35, 45), grime)
    base *= relief(height, 1.6)[..., None]
    base = base * (0.55 + 0.45 * plates)[..., None]  # grooves stay dark
    base += studs[..., None] * np.array((28, 30, 36))
    base *= (1.0 - 0.35 * streaks(43, 14, (20, 60)))[..., None]
    base += noise(44, beta=1.0, low=16)[..., None] * 6.0 - 3.0  # fine grain
    return finish(base)


@texture("floor", tile=2.0)
def _floor() -> Image:
    """Steel floor plates (four per tile) with a worn diamond tread, scuffs, seams and the
    odd oil stain."""
    y, x = coords()
    plates = panel_seams(2, groove=1.5)
    period = 16.0  # tread diamonds: 8 per plate across
    u = ((x + y) % period) / period - 0.5
    v = ((x - y) % period) / period - 0.5
    tread = np.clip(1.0 - (np.abs(u) * 2.0 + np.abs(v) * 6.0), 0.0, 1.0)
    wear = noise(51, beta=2.0, low=1)
    tread *= np.clip(1.4 - wear * 1.6, 0.0, 1.0)  # worn flat where people walk
    height = plates * 0.9 + tread * 0.5
    base = mix((22, 24, 30), (37, 40, 48), noise(52, beta=2.2, low=1))
    base *= relief(height, 1.3)[..., None]
    base *= (0.5 + 0.5 * plates)[..., None]
    stains = blur(spots(53, 5, (6, 16)), 2.0)
    base *= (1.0 - 0.35 * stains)[..., None]
    base += scratches(54, 18, (6, 18))[..., None] * np.array((14, 15, 17))
    return finish(base)


@texture("metal", tile=1.0)
def _metal() -> Image:
    """Brushed steel: fine streaks along one direction, soft mottling and light scratches."""
    brush = noise(61, beta=1.2, low=4, stretch=(0.08, 1.0))
    mottle = noise(62, beta=2.5, low=1)
    base = mix((36, 39, 45), (56, 60, 68), 0.55 * brush + 0.45 * mottle)
    base += scratches(63, 22, (5, 20))[..., None] * np.array((22, 23, 26))
    base *= (1.0 - 0.25 * blur(spots(64, 4, (8, 20)), 3.0))[..., None]
    return finish(base)


@texture("hazard", tile=1.0)
def _hazard() -> Image:
    """Worn yellow and black safety stripes at 45 degrees, the paint chipped at the edges
    and dirty (four stripe pairs per tile, so the diagonals meet across tiles)."""
    y, x = coords()
    period = SIZE / 4
    phase = ((x + y) % period) / period
    edge = (
        np.minimum(np.abs(phase - 0.5), np.minimum(phase, 1.0 - phase)) * period
    )  # px to a stripe edge
    yellow = phase < 0.5
    chip = noise(71, beta=1.4, low=6)
    chipped = (edge < 3.5) & (chip > 0.6)
    dirt = noise(72, beta=2.3, low=1)
    paint = np.where(
        yellow[..., None],
        mix((178, 140, 8), (236, 190, 24), dirt),
        mix((14, 14, 16), (32, 31, 30), dirt),
    )
    bare = mix((60, 62, 66), (90, 92, 96), noise(73, beta=1.5, low=4))
    img = np.where(chipped[..., None], bare, paint)
    img *= (1.0 - 0.3 * streaks(74, 10, (15, 45)))[..., None]
    height = yellow * 0.3 + blur(yellow.astype(float), 0.8) * 0.2
    img *= relief(height, 0.8)[..., None]
    return finish(img)


@texture("grate", tile=0.5)
def _grate() -> Image:
    """Heavy bar grating: bearing bars every 8 px, cross bars every 32, dark drainage below."""
    y, x = coords()
    bearing = np.clip(1.0 - wrapped_dist(x % 8, 0.0) / 1.6, 0.0, 1.0)
    cross = np.clip(1.0 - wrapped_dist(y % 32, 0.0) / 1.8, 0.0, 1.0)
    bars = np.maximum(bearing, cross)
    below = mix((4, 5, 6), (12, 14, 16), noise(81, beta=2.0, low=2))
    steel = mix((46, 50, 58), (70, 76, 86), noise(82, beta=1.8, low=2))
    img = below + (steel - below) * bars[..., None]
    img *= relief(blur(bars, 0.7), 1.6)[..., None]
    img *= (1.0 - 0.3 * blur(spots(83, 3, (10, 20)), 3.0))[..., None]
    return finish(img)


@texture("concrete", tile=2.0)
def _concrete() -> Image:
    """Poured concrete: soft mottling, pores and a few hairline cracks."""
    base = mix((70, 71, 74), (104, 104, 106), noise(91, beta=2.4, low=1))
    base *= (0.88 + 0.12 * noise(92, beta=1.0, low=10))[..., None]
    pores = (noise(93, beta=0.3, low=20) > 0.82).astype(float)
    base *= (1.0 - 0.35 * pores)[..., None]
    base *= (1.0 - 0.4 * blur(cracks(94, 4), 0.45))[..., None]
    base *= (1.0 - 0.25 * streaks(95, 8, (30, 80), width=2.0))[..., None]
    return finish(base)


@texture("plaster", tile=1.5)
def _plaster() -> Image:
    """A tenement flat's wall: cheap paint over plaster, yellowed and stained - damp
    blooms, water streaks from the ceiling, hairline cracks, a patch flaked off."""
    base = mix((112, 104, 84), (138, 130, 106), noise(301, beta=2.3, low=1))
    damp = blur(spots(302, 5, (10, 26)), 4.0)
    base = base * (1.0 - 0.35 * damp)[..., None] + damp[..., None] * np.array((18, 22, 8))
    base *= (1.0 - 0.3 * streaks(303, 10, (40, 110), width=2.5))[..., None]
    base *= (1.0 - 0.5 * blur(cracks(304, 5), 0.5))[..., None]
    flake = noise(305, beta=0.8, low=8) > 0.86
    base[flake] = base[flake] * 0.55 + np.array((60, 58, 56)) * 0.45
    return finish(base)


@texture("linoleum", tile=1.2)
def _linoleum() -> Image:
    """Cracked checkerboard lino, worn through in places, grimy in the seams."""
    y, x = coords()
    check = ((x // 32 + y // 32) % 2).astype(float)
    base = mix((46, 52, 50), (120, 118, 104), check)
    base *= (0.85 + 0.15 * noise(311, beta=2.2, low=1))[..., None]
    seams = np.maximum(grid_lines(32, 1, 0, 0), grid_lines(32, 1, 0, 1))
    base *= (1.0 - 0.45 * seams)[..., None]
    worn = blur(spots(312, 4, (12, 28)), 5.0)
    base = base * (1.0 - 0.3 * worn)[..., None] + worn[..., None] * np.array((40, 34, 24))
    base *= (1.0 - 0.55 * blur(cracks(313, 6), 0.4))[..., None]
    return finish(base)


@texture("tile", tile=0.6)
def _tile() -> Image:
    """Small bathroom tiles gone grey-green, black mould in the grout, a few cracked."""
    y, x = coords()
    grout = np.maximum(grid_lines(16, 2, 0, 0), grid_lines(16, 2, 0, 1))
    base = mix((150, 164, 156), (180, 190, 182), noise(321, beta=1.4, low=6))
    base *= (0.9 + 0.1 * noise(322, beta=2.0, low=1))[..., None]
    base = base * (1.0 - grout[..., None]) + grout[..., None] * np.array((52, 56, 48))
    mould = blur(spots(323, 6, (6, 18)), 3.0) * (0.4 + 0.6 * grout)
    base *= (1.0 - 0.5 * mould)[..., None]
    base *= (1.0 - 0.6 * blur(cracks(324, 3), 0.4))[..., None]
    return finish(base)


@texture("static")
def _static() -> Image:
    """A dead channel: snow, rolling bars."""
    rng = np.random.default_rng(331)
    snow = rng.uniform(0.0, 1.0, (SIZE, SIZE))
    y, _ = coords()
    bars = 0.75 + 0.25 * np.sin(y / SIZE * 2 * np.pi * 3)
    img = (snow * bars)[..., None] * np.array((200, 210, 230)) + 20
    return finish(img)


@texture("asphalt", tile=2.0)
def _asphalt() -> Image:
    """Wet road surface: dark aggregate with lighter stones and shallow glossy patches."""
    base = mix((26, 27, 30), (44, 45, 48), noise(101, beta=1.6, low=2))
    stones = (noise(102, beta=0.2, low=24) > 0.74).astype(float)
    base += stones[..., None] * np.array((26, 26, 27))
    base *= (0.85 + 0.15 * noise(103, beta=2.4, low=1))[..., None]
    return finish(base)


@texture("brick", tile=1.0)
def _brick() -> Image:
    """Old tenement brick: 8 courses per tile, staggered, each brick a little different,
    recessed mortar and soot."""
    y, x = coords()
    course = 16
    row = (y // course).astype(int)
    shifted = (x + (row % 2) * 16) % SIZE
    brick_w = 32
    mortar_y = np.minimum(y % course, course - (y % course))
    mortar_x = np.minimum(shifted % brick_w, brick_w - (shifted % brick_w))
    face = np.clip((np.minimum(mortar_x, mortar_y) - 0.6) / 1.4, 0.0, 1.0)
    brick_id = row * 4 + (shifted // brick_w).astype(int)
    rng = np.random.default_rng(111)
    tones = rng.uniform(0.75, 1.15, size=(SIZE // course) * (SIZE // brick_w) + 8)
    tone = tones[brick_id % len(tones)]
    clay = mix((120, 62, 48), (170, 96, 74), noise(112, beta=1.8, low=3)) * tone[..., None]
    mortar = mix((70, 66, 62), (96, 92, 86), noise(113, beta=1.5, low=4))
    img = mortar + (clay - mortar) * face[..., None]
    img *= relief(face * 0.8 + noise(114, beta=1.2, low=8) * 0.1, 1.4)[..., None]
    img *= (1.0 - 0.45 * noise(115, beta=2.6, low=1) ** 2)[..., None]  # soot
    return finish(img)


@texture("wood", tile=0.8)
def _wood() -> Image:
    """Planks of worn timber: three boards per tile, long grain with knots."""
    y, x = coords()
    grain = noise(121, beta=1.4, low=2, stretch=(0.05, 1.0))
    rings = 0.5 + 0.5 * np.sin((x / SIZE * 4 + grain * 3.0) * 2 * np.pi)
    board = np.minimum(y % (SIZE / 3), SIZE / 3 - (y % (SIZE / 3)))
    gap = np.clip((board - 0.5) / 1.2, 0.0, 1.0)
    wood = mix((92, 62, 38), (150, 108, 68), 0.6 * rings + 0.4 * grain)
    knots = blur(spots(122, 3, (3, 6)), 1.0)
    wood *= (1.0 - 0.45 * knots)[..., None]
    img = wood * (0.35 + 0.65 * gap)[..., None]
    img *= relief(gap * 0.6, 1.2)[..., None]
    return finish(img)


@texture("cardboard", tile=0.6)
def _cardboard() -> Image:
    """Brown corrugated board: fluting ridges, fibre, scuffs and a strip of packing tape."""
    y, x = coords()
    flutes = 0.5 + 0.5 * np.cos(x / SIZE * 2 * np.pi * 32)
    board = mix((142, 108, 72), (182, 144, 100), noise(131, beta=1.6, low=2))
    board *= (0.94 + 0.06 * flutes)[..., None]
    tape = np.clip(1.0 - wrapped_dist(y, 64.0) / 12.0, 0.0, 1.0) > 0.0
    board = np.where(tape[..., None], board * 1.08 + 10, board)
    board *= (1.0 - 0.25 * blur(spots(132, 4, (6, 14)), 2.0))[..., None]
    return finish(board)


# =============================================================================== screens
def _screen_base(bg: RGB, glow: RGB, seed: int) -> np.ndarray:
    """A monitor face: dark glass with a faint glow towards the middle and very soft
    scanlines (strong alternating lines alias into black bands at a distance)."""
    y, x = coords()
    vignette = 1.0 - 0.45 * (((x - 64) / 64) ** 2 + ((y - 64) / 64) ** 2)
    img = _canvas(bg) * vignette[..., None]
    img += np.array(glow) * 0.08 * noise(seed, beta=2.0, low=1)[..., None]
    img *= (1.0 - 0.12 * (np.arange(SIZE) % 4 < 2))[:, None, None]
    return img


def _text_rows(img, rows, x0, color, seed, dim=0.45):
    """Rows of 'text': short runs of lit cells like a terminal dump."""
    rng = np.random.default_rng(seed)
    for r in rows:
        x = x0
        while x < SIZE - 10:
            w = int(rng.integers(3, 14))
            if rng.random() < 0.8:
                shade = 1.0 if rng.random() < 0.3 else dim + rng.random() * 0.3
                img[r : r + 3, x : min(x + w, SIZE - 10)] = np.array(color) * shade
            x += w + int(rng.integers(2, 5))


def _glow(
    img: np.ndarray, lit: np.ndarray, color: RGB, radius: float, strength: float
) -> np.ndarray:
    return img + blur(lit, radius)[..., None] * np.array(color) * strength


@texture("terminal_locked")
def _terminal_locked() -> Image:
    """Crimson 'ACCESS DENIED' screen: title bar, padlock, hex dump and a warning bar."""
    img = _screen_base((34, 7, 9), (255, 40, 40), 141)
    lit = np.zeros((SIZE, SIZE))
    img[8:20, 8:120] = (150, 20, 24)
    img[11:17, 14:100] = (255, 190, 190)
    # padlock
    img[44:84, 44:84] = (190, 28, 30)
    img[50:78, 50:78] = (70, 12, 14)
    img[58:70, 58:70] = (255, 70, 70)
    img[30:46, 52:56] = (190, 28, 30)
    img[30:46, 72:76] = (190, 28, 30)
    img[30:34, 52:76] = (190, 28, 30)
    lit[44:84, 44:84] = 1.0
    _text_rows(img, range(94, 118, 6), 12, (255, 90, 90), 142)
    img[120:124, 8:120] = (255, 60, 60)
    return finish(_glow(img, lit, (255, 40, 40), 4.0, 0.25))


@texture("terminal_unlocked")
def _terminal_unlocked() -> Image:
    """Green 'SYSTEM OVERRIDE' screen: title bar, decrypted code streams, a progress bar."""
    img = _screen_base((6, 26, 14), (40, 255, 120), 151)
    img[8:20, 8:120] = (0, 130, 60)
    img[11:17, 14:104] = (200, 255, 220)
    _text_rows(img, range(28, 100, 6), 10, (60, 255, 140), 152)
    img[108:118, 10:118] = (0, 60, 30)
    img[110:116, 12:116] = (90, 255, 150)
    return finish(img)


@texture("screen_net")
def _screen_net() -> Image:
    """Blue network topology: faint grid, linked nodes, a status strip."""
    img = _screen_base((6, 16, 30), (0, 160, 255), 161)
    grid = np.maximum(grid_lines(16, 1, axis=0), grid_lines(16, 1, axis=1))
    img += grid[..., None] * np.array((0, 26, 40))
    nodes = ((30, 28), (30, 96), (64, 62), (98, 28), (98, 96), (64, 108))
    lit = np.zeros((SIZE, SIZE))
    links = ((0, 2), (1, 2), (2, 3), (2, 4), (4, 5), (1, 5))
    for (ay, ax), (by, bx) in ((nodes[a], nodes[b]) for a, b in links):
        for t in np.linspace(0, 1, 120):
            yy, xx = int(ay + (by - ay) * t), int(ax + (bx - ax) * t)
            img[yy : yy + 2, xx : xx + 2] = (0, 150, 210)
    for ny, nx in nodes:
        img[ny - 6 : ny + 7, nx - 6 : nx + 7] = (0, 200, 255)
        img[ny - 3 : ny + 4, nx - 3 : nx + 4] = (235, 250, 255)
        lit[ny - 6 : ny + 7, nx - 6 : nx + 7] = 1.0
    img[116:122, 8:120] = (0, 90, 140)
    return finish(_glow(img, lit, (0, 180, 255), 3.0, 0.35))


@texture("screen_radar")
def _screen_radar() -> Image:
    """Amber radar: range rings, crosshair, a fading sweep wedge and two blips."""
    img = _screen_base((28, 18, 6), (255, 170, 0), 171)
    y, x = coords()
    dist = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)
    for r, c in ((50, (255, 190, 0)), (34, (190, 130, 0)), (18, (150, 100, 0))):
        img[np.abs(dist - r) < 1.2] = c
    img[63:65, 12:116] = (220, 150, 0)
    img[12:116, 63:65] = (220, 150, 0)
    ang = (np.degrees(np.arctan2(y - 64, x - 64)) + 360.0) % 360.0
    sweep = np.clip(1.0 - ((ang - 300.0) % 360.0) / 70.0, 0.0, 1.0) * (dist < 50)
    img += sweep[..., None] * np.array((110, 70, 0))
    lit = np.zeros((SIZE, SIZE))
    for by, bx in ((40, 86), (80, 44)):
        img[by - 3 : by + 3, bx - 3 : bx + 3] = (255, 255, 190)
        lit[by - 3 : by + 3, bx - 3 : bx + 3] = 1.0
    return finish(_glow(img, lit, (255, 200, 80), 3.0, 0.6))


@texture("server_rack", tile=0.75)
def _server_rack() -> Image:
    """Rack-mounted servers, four 2U units per tile: brushed faceplates, vent slots, drive
    bays, handles and rows of status LEDs (green, cyan, amber)."""
    y, x = coords()
    unit = 32
    in_unit = y % unit
    img = mix((16, 17, 21), (28, 30, 36), noise(181, beta=1.2, low=4, stretch=(0.1, 1.0)))
    seam = (in_unit < 2) | (in_unit > unit - 2)
    img[seam] = (6, 7, 9)
    img[(in_unit == 2)] = (46, 50, 58)
    # vent slots on the left, drive bays in the middle, handles at the sides
    vents = (x > 10) & (x < 44) & (in_unit > 8) & (in_unit < 24) & ((x % 3) == 0)
    img[vents] = (8, 9, 11)
    bays = (x > 50) & (x < 96) & (in_unit > 7) & (in_unit < 25) & ((x - 50) % 12 < 10)
    img[bays] = img[bays] * 0.6 + np.array((22, 24, 30))
    handle = ((x < 6) | (x > 121)) & (in_unit > 6) & (in_unit < 26)
    img[handle] = (70, 74, 82)
    rng = np.random.default_rng(182)
    colours = ((0, 255, 120), (0, 220, 255), (255, 180, 0), (0, 255, 120))
    for u in range(SIZE // unit):
        for k in range(4):
            if rng.random() < 0.85:
                c = colours[int(rng.integers(0, len(colours)))]
                yy = u * unit + 10 + (k // 2) * 8
                xx = 102 + (k % 2) * 8
                img[yy : yy + 3, xx : xx + 4] = c
    img *= relief(seam.astype(float) * -0.5, 1.0)[..., None]
    return finish(img)


# =============================================================================== props
@texture("crate")
def _crate() -> Image:
    """One face of a reinforced alloy cargo crate: bevelled frame, corner plates with
    bolts, a stencilled hazard band and scuffs."""
    y, x = coords()
    frame = ((x < 10) | (x > 117) | (y < 10) | (y > 117)).astype(float)
    img = mix((44, 48, 56), (62, 67, 78), noise(191, beta=2.0, low=1))
    img = np.where(frame[..., None], img * 1.25 + 6, img)
    height = frame * 0.8
    for cy in (0, 104):
        for cx in (0, 104):
            plate = (y >= cy) & (y < cy + 24) & (x >= cx) & (x < cx + 24)
            img[plate] = (96, 104, 120)
            height = np.where(plate, 1.2, height)
            for by, bx in (
                (cy + 6, cx + 6),
                (cy + 17, cx + 17),
                (cy + 6, cx + 17),
                (cy + 17, cx + 6),
            ):
                bolt = (y - by) ** 2 + (x - bx) ** 2 < 5
                img[bolt] = (150, 158, 172)
    band = (y >= 52) & (y < 76) & (x >= 24) & (x < 104)
    stripes = ((x + y) // 8) % 2 == 0
    img[band & stripes] = (226, 178, 10)
    img[band & ~stripes] = (24, 24, 26)
    img *= relief(blur(height, 1.0), 1.2)[..., None]
    img += scratches(192, 14, (6, 20))[..., None] * np.array((30, 32, 36))
    return finish(img)


@texture("soda_can")
def _soda_can() -> Image:
    """The NEON COLD energy-drink wrapper: cyan/magenta chevrons, a bright logo band and a
    barcode (wraps around the can: the stripes meet at the seam)."""
    y, x = coords()
    phase = (x + y * 0.5) % 32
    img = _canvas((190, 196, 206))
    img[phase < 13] = (0, 210, 250)
    img[(phase >= 13) & (phase < 20)] = (250, 0, 130)
    img[40:62, :] = (245, 245, 248)
    img[44:58, 20:108] = (20, 24, 40)
    img[47:55, 26:102:4] = (250, 0, 130)
    img[96:116, 70:110] = (240, 240, 240)
    img[98:114, 72:108:3] = (16, 16, 16)
    img[98:114, 73:108:5] = (16, 16, 16)
    img *= (0.8 + 0.2 * np.cos((x / SIZE) * 2 * np.pi))[..., None]  # a highlight round the can
    return finish(img)


@texture("neon_sign")
def _neon_sign() -> Image:
    """A dark panel with the katakana サイバー (CYBER) in cyan neon tubes inside a magenta
    neon border, each tube with its halo of light on the panel."""
    img = _canvas((10, 8, 18)) * (0.8 + 0.2 * noise(201, beta=2.0, low=1))[..., None]
    magenta, cyan = np.array((255, 40, 190)), np.array((40, 235, 255))
    border = stroke([(7, 7), (7, 120), (120, 120), (120, 7), (7, 7)], 3.0)
    glyphs = [
        # サ: a bar, two uprights, the right one sweeping down to the left
        [(46, 14), (46, 40)], [(36, 21), (56, 21)], [(36, 33), (60, 33), (74, 24)],
        # イ: a stroke slashing down-left, and the upright hanging from it
        [(36, 64), (58, 46)], [(48, 56), (82, 56)],
        # バ: ハ (two strokes splaying apart) and the two dakuten ticks
        [(48, 78), (78, 70)], [(46, 88), (78, 98)], [(34, 96), (40, 99)], [(32, 102), (38, 105)],
        # ー: the long bar
        [(96, 22), (96, 106)],
    ]  # fmt: skip
    tubes = np.zeros((SIZE, SIZE))
    for g in glyphs:
        tubes = np.maximum(tubes, stroke(g, 3.2))
    img += blur(tubes, 3.0)[..., None] * cyan * 0.8 + blur(border, 3.0)[..., None] * magenta * 0.8
    img = img * (1 - tubes[..., None]) + tubes[..., None] * (0.55 * cyan + 115)
    img = img * (1 - border[..., None]) + border[..., None] * (0.55 * magenta + 115)
    return finish(img)


# =============================================================================== people
# (the faces are ``assets.faces``: painted per skin tone, mark and expression)


@texture("flesh", tile=0.5)
def _flesh() -> Image:
    """Plain skin for arms, hands and necks: soft mottling, no features."""
    img = mix((198, 160, 136), (220, 182, 158), noise(221, beta=2.2, low=1))
    img *= (0.96 + 0.04 * noise(222, beta=0.8, low=12))[..., None]
    return finish(img)


def _weave(seed: int, dark: RGB, light: RGB, period: int = 4) -> np.ndarray:
    """A woven fabric: alternating over/under threads (period divides SIZE)."""
    y, x = coords()
    warp = np.cos(x / period * np.pi) ** 2
    weft = np.cos(y / period * np.pi) ** 2
    over = (x // period + y // period) % 2 == 0
    thread = np.where(over, warp, weft)
    img = mix(dark, light, 0.6 * thread + 0.4 * noise(seed, beta=2.0, low=1))
    return img


@texture("cloth", tile=0.3)
def _cloth() -> Image:
    """Light woven cotton - a civilian's shirt, tinted by the wearer's colour."""
    img = _weave(231, (168, 168, 164), (226, 226, 220))
    return finish(img)


@texture("fabric", tile=0.3)
def _fabric() -> Image:
    """Dark technical fabric (guard under-armour, awnings, curtains)."""
    img = _weave(241, (24, 28, 36), (48, 54, 66))
    return finish(img)


@texture("pants", tile=0.3)
def _pants() -> Image:
    """Dark twill: fine diagonal ribs."""
    y, x = coords()
    rib = 0.5 + 0.5 * np.cos(((x + y) % 8) / 8 * 2 * np.pi)
    img = mix((14, 16, 22), (32, 35, 46), 0.5 * rib + 0.5 * noise(251, beta=2.0, low=1))
    return finish(img)


@texture("leather", tile=0.4)
def _leather() -> Image:
    """Dark pebbled leather with worn, lighter patches."""
    grain = noise(261, beta=0.6, low=16)
    wear = noise(262, beta=2.4, low=1)
    img = mix((20, 16, 13), (46, 36, 28), 0.5 * grain + 0.5 * wear)
    img *= relief(grain * 0.5, 1.0)[..., None]
    return finish(img)


@texture("armor", tile=0.4)
def _armor() -> Image:
    """Ballistic plating: four 20 cm plates per tile, each chamfered at the edges with a
    raised spine across the middle, bolted at the corners, the paint scuffed."""
    y, x = coords()
    plates = panel_seams(2, groove=4.0)
    py, px = y % 64, x % 64
    spine = np.clip(1.0 - np.abs(py - 32) / 3.0, 0.0, 1.0) * (np.abs(px - 32) < 24)
    height = plates * 0.8 + blur(spine, 0.8) * 0.5
    img = mix((34, 38, 46), (60, 66, 78), noise(271, beta=2.2, low=1))
    img *= (0.45 + 0.55 * plates)[..., None]
    img *= relief(blur(height, 0.7), 1.6)[..., None]
    bolts = rivets(2, inset=8.0, radius=2.4)
    img = img * (1.0 - bolts[..., None]) + bolts[..., None] * np.array((104, 110, 124))
    img += scratches(272, 12, (4, 12))[..., None] * np.array((22, 23, 26))
    return finish(img)


# =============================================================================== greenhouse
@texture("moss", tile=1.5)
def _moss() -> Image:
    """Damp moss over drainage plates: clumpy greens with darker seams showing through."""
    clumps = noise(281, beta=1.6, low=3)
    big = noise(282, beta=2.6, low=1)
    img = mix((24, 40, 26), (64, 96, 52), 0.6 * clumps + 0.4 * big)
    seams = 1.0 - panel_seams(2, groove=2.5)
    img *= (1.0 - 0.35 * seams * (clumps < 0.55))[..., None]
    img *= relief(clumps * 0.6, 1.4)[..., None]
    return finish(img)


@texture("soil", tile=1.0)
def _soil() -> Image:
    """Dark substrate with crumbs, pale mycelium threads and a little moss."""
    img = mix((34, 25, 19), (68, 52, 38), noise(291, beta=1.4, low=3))
    crumbs = (noise(292, beta=0.3, low=20) > 0.78).astype(float)
    img *= (1.0 + 0.25 * crumbs)[..., None]
    threads = blur(scratches(293, 26, (10, 30)), 0.5)
    img += np.clip(threads, 0, 1)[..., None] * np.array((80, 78, 64))
    return finish(img)


@texture("fungus", tile=0.3)
def _fungus() -> Image:
    """Pale ribbed fungal flesh: cream with soft growth rings."""
    y, _ = coords()
    rings = 0.5 + 0.5 * np.cos(y / 16 * 2 * np.pi)
    img = mix((172, 162, 134), (214, 204, 176), 0.5 * rings + 0.5 * noise(301, beta=1.6, low=2))
    return finish(img)


@texture("spore")
def _spore() -> Image:
    """Mushroom cap skin (wraps around the cone): pale blue-white with radiating gills
    running up it and speckles, so the cap's emission colour is what glows."""
    y, x = coords()
    gills = 0.5 + 0.5 * np.cos(x / SIZE * 2 * np.pi * 24)
    img = mix((140, 174, 216), (196, 222, 248), 0.5 * gills + 0.5 * noise(311, beta=1.8, low=2))
    speck = noise(312, beta=0.2, low=24) > 0.85
    img[speck] = (236, 252, 230)
    return finish(img)


# =============================================================================== flat neon
@texture("neon_cyan")
def _neon_cyan() -> Image:
    y, _ = coords()
    core = 1.0 - 0.18 * ((y - 64) / 64) ** 2  # brightest along the middle of the tube
    return finish(_canvas((30, 240, 255)) * core[..., None])


@texture("neon_magenta")
def _neon_magenta() -> Image:
    y, _ = coords()
    core = 1.0 - 0.18 * ((y - 64) / 64) ** 2
    return finish(_canvas((255, 30, 185)) * core[..., None])


# =============================================================================== food
# The fish's skin wraps round ``render.shapes.fish_body`` (u round the body: 0.25 the
# belly, 0.75 the back, 0 and 0.5 its flanks; v along it: 0 the tail, 1 the nose).
FISH_EYE_V = 0.86  # how far along the body the eye is (and the gill cover behind it)
FISH_GILL_V = 0.74


@texture("fish_skin")
def _fish_skin() -> Image:
    """A salmon's skin: a dark blue-grey back speckled black, silver flanks, a pale belly;
    rows of overlapping scales, the lateral line down each flank, the curve of the gill
    cover, an eye with a gold ring and a dark mouth line at the nose."""
    y, x = coords()
    u = x / SIZE
    v = 1.0 - y / SIZE  # (row 0 is the top of the picture: the nose end)
    up = np.cos(2.0 * np.pi * (u - 0.75))  # 1 along the back, -1 along the belly
    back = np.clip(up, 0.0, 1.0) ** 0.8
    belly = np.clip(-up, 0.0, 1.0) ** 0.7
    img = mix((176, 186, 194), (52, 70, 92), back)
    img = (
        img * (1.0 - belly[..., None])
        + mix((200, 200, 204), (236, 226, 222), belly) * belly[..., None]
    )
    img *= (0.93 + 0.1 * noise(911, beta=1.6, low=6))[..., None]
    # scales: overlapping arcs, row on staggered row (none on the head)
    cell = 6.0
    row = np.floor(y / cell)
    sx = (x + (row % 2) * cell / 2.0) % cell - cell / 2.0
    sy = y % cell
    arc = np.abs(np.hypot(sx, sy * 1.3) - cell * 0.62)
    scale = np.clip(1.0 - arc / 0.9, 0.0, 1.0) * (v < FISH_GILL_V)
    img *= (1.0 - 0.22 * scale)[..., None]
    img += (24.0 * np.clip(1.0 - np.hypot(sx, sy - cell * 0.75) / 2.0, 0, 1) * (v < FISH_GILL_V))[
        ..., None
    ]
    # black speckles along the back
    speck = spots(912, 60, (0.8, 1.6)) * (back > 0.35)
    img *= (1.0 - 0.75 * np.clip(speck, 0.0, 1.0))[..., None]
    # the lateral line down each flank (a little above the middle of it)
    for side in (0.97, 0.53):
        line = np.clip(1.0 - wrapped_dist(x, side * SIZE) / 1.2, 0.0, 1.0) * (v < FISH_GILL_V)
        img = img * (1.0 - 0.35 * line[..., None])
    # the gill cover's curved edge, and the head smoother and darker above it
    for side in (0.0, 0.5):
        du = wrapped_dist(x, side * SIZE) / SIZE
        edge = FISH_GILL_V - 0.05 * (du / 0.25) ** 2
        gill = np.clip(1.0 - np.abs(v - edge) * SIZE / 1.4, 0.0, 1.0) * (du < 0.24)
        img *= (1.0 - 0.55 * gill)[..., None]
    # the eyes (one on each flank, a little towards the back) and the mouth line
    for side, toward in ((1.0, -1.0), (0.5, 1.0)):
        cu = (side + 0.06 * toward) * SIZE
        r = np.hypot(wrapped_dist(x, cu), (v - FISH_EYE_V) * SIZE)
        ring = np.clip(4.6 - r, 0.0, 1.0)
        img = img * (1.0 - ring[..., None]) + np.array((196, 160, 70)) * ring[..., None]
        pupil = np.clip(3.0 - r, 0.0, 1.0)
        img = img * (1.0 - pupil[..., None]) + np.array((8, 8, 10)) * pupil[..., None]
        glint = np.clip(
            1.2 - np.hypot(wrapped_dist(x, cu - 1.0), (v - FISH_EYE_V) * SIZE - 1.0), 0, 1
        )
        img += (200.0 * glint)[..., None]
        mouth_u = (side - 0.05 * toward) * SIZE
        mouth = np.clip(1.0 - wrapped_dist(x, mouth_u) / 1.1, 0.0, 1.0) * (v > 0.955)
        img *= (1.0 - 0.7 * mouth)[..., None]
    return finish(img)


@texture("fin")
def _fin() -> Image:
    """A fin: thin grey-blue membrane between darker rays (the rays along v)."""
    y, x = coords()
    ray = 0.5 + 0.5 * np.cos(x / SIZE * 2.0 * np.pi * 12.0)
    img = mix((118, 130, 146), (66, 78, 96), 0.7 * ray**3 + 0.3 * noise(921, beta=2.0, low=2))
    tip = np.clip((SIZE * 0.25 - y) / (SIZE * 0.25), 0.0, 1.0)  # darker towards the edge
    img *= (1.0 - 0.3 * tip)[..., None]
    return finish(img)


@texture("salmon")
def _salmon() -> Image:
    """Raw salmon: deep orange flesh in bands with pale stripes of fat between them."""
    y, x = coords()
    wave = 3.0 * np.sin(x / SIZE * 2.0 * np.pi)
    band = ((y + x * 0.5 + wave) % 32) / 32.0
    fat = np.clip(1.0 - np.abs(band - 0.5) / 0.06, 0.0, 1.0)
    img = mix((236, 104, 56), (250, 140, 84), noise(931, beta=2.2, low=1))
    img = img * (1.0 - fat[..., None]) + np.array((252, 214, 190)) * fat[..., None]
    return finish(img)


# The homes' surfaces, pictures, window views and television channels register themselves
# with ``texture`` above (a module of their own: ``assets.homes``).
from cyberduke.assets import homes as _homes  # noqa: E402, F401
from cyberduke.assets import signs as _signs  # noqa: E402, F401
from cyberduke.assets.faces import FACES, MOUTH_RGB  # noqa: E402, F401
