"""Procedural textures for the people's homes (``levels.flats``, ``levels.apartment``):
what makes one flat somebody's and not a copy of the next.

* **Surfaces** (tiled, seamless like every tiled material - ``tests/test_textures.py``):
  wallpapers (faded stripes, damask, a seventies geometric), wood panelling, painted
  breeze block, worn carpet, chequered lino, parquet, hexagon and subway tiles.
* **Pictures** (one copy per face): posters, framed prints and photos, a calendar, a
  certificate, a tool board, rugs, book spines, a cereal box, fridge notes - and the view
  out of a window, by night and by day.
* **Television**: every channel a short loop of frames (``TV_CHANNELS``) the renderer
  steps through - snow that crawls and rolls on a dead channel, the news (an anchor,
  a picture in the picture, a ticker running along the bottom), a cartoon, a music
  channel, the adverts - and the dark glass of a set that is switched off.

Registered through ``assets.textures.texture`` (imported at the end of that module), so
they are generated, tested and uploaded with every other texture.
"""

from __future__ import annotations

import numpy as np

from cyberduke.assets.textures import (
    SIZE,
    Image,
    _canvas,
    blur,
    coords,
    cracks,
    finish,
    grid_lines,
    mix,
    noise,
    relief,
    spots,
    streaks,
    stroke,
    texture,
    wrapped_dist,
)

RGB = tuple[float, float, float]

# =============================================================================== lettering
# A 3 x 5 pixel font: enough for a ticker, a logo, a price
_GLYPHS = {
    "A": ("010", "101", "111", "101", "101"), "B": ("110", "101", "110", "101", "110"),
    "C": ("011", "100", "100", "100", "011"), "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"), "F": ("111", "100", "110", "100", "100"),
    "G": ("011", "100", "101", "101", "011"), "H": ("101", "101", "111", "101", "101"),
    "I": ("111", "010", "010", "010", "111"), "J": ("001", "001", "001", "101", "010"),
    "K": ("101", "101", "110", "101", "101"), "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"), "N": ("110", "101", "101", "101", "101"),
    "O": ("010", "101", "101", "101", "010"), "P": ("110", "101", "110", "100", "100"),
    "Q": ("010", "101", "101", "110", "011"), "R": ("110", "101", "110", "101", "101"),
    "S": ("011", "100", "010", "001", "110"), "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"), "V": ("101", "101", "101", "101", "010"),
    "W": ("101", "101", "111", "111", "101"), "X": ("101", "101", "010", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"), "Z": ("111", "001", "010", "100", "111"),
    "0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "010", "100", "111"), "3": ("110", "001", "010", "001", "110"),
    "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "110", "001", "110"),
    "6": ("011", "100", "111", "101", "111"), "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "110"),
    " ": ("000", "000", "000", "000", "000"), "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"), "!": ("010", "010", "010", "000", "010"),
    ":": ("000", "010", "000", "010", "000"), "+": ("000", "010", "111", "010", "000"),
    "%": ("101", "001", "010", "100", "101"), "$": ("011", "110", "010", "011", "110"),
    "/": ("001", "001", "010", "100", "100"), "'": ("010", "010", "000", "000", "000"),
}  # fmt: skip


def text_width(text: str, scale: int = 1) -> int:
    return len(text) * 4 * scale - scale


def write(
    img: np.ndarray, text: str, x: int, y: int, color: RGB, scale: int = 1, wrap: bool = False
) -> None:
    """Lettering into ``img`` (rows, columns) with its top left corner at (x, y); letters
    off the edge are clipped (or come back in at the other side, ``wrap``)."""
    h, w = img.shape[:2]
    for k, ch in enumerate(text.upper()):
        rows = _GLYPHS.get(ch, _GLYPHS[" "])
        for r, line in enumerate(rows):
            for c, bit in enumerate(line):
                if bit != "1":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        px = x + (k * 4 + c) * scale + dx
                        py = y + r * scale + dy
                        if wrap:
                            px %= w
                        if 0 <= px < w and 0 <= py < h:
                            img[py, px] = color


def _disc(cy: float, cx: float, r: float) -> np.ndarray:
    y, x = coords()
    return ((y - cy) ** 2 + (x - cx) ** 2) < r * r


def _ellipse(cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    y, x = coords()
    return ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1.0


def _rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((SIZE, SIZE), dtype=bool)
    m[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = True
    return m


def _paint(img: np.ndarray, mask: np.ndarray, color: RGB, alpha: float = 1.0) -> None:
    img[mask] = img[mask] * (1.0 - alpha) + np.array(color, dtype=float) * alpha


def _aged(img: np.ndarray, seed: int, amount: float = 0.25) -> np.ndarray:
    """Paper gone yellow and grubby at the edges, a crease or two (a poster, a photo)."""
    y, x = coords()
    edge = np.minimum(np.minimum(x, SIZE - 1 - x), np.minimum(y, SIZE - 1 - y))
    vignette = 1.0 - amount * np.clip(1.0 - edge / 18.0, 0.0, 1.0)
    img = img * vignette[..., None]
    img = img * (0.9 + 0.1 * noise(seed, beta=2.0, low=1))[..., None]
    img *= (1.0 - 0.3 * blur(cracks(seed + 1, 2, steps=10), 0.5))[..., None]
    return img


# =============================================================================== surfaces
@texture("wallpaper_stripes", tile=0.64)
def _wallpaper_stripes() -> Image:
    """Faded regency stripes - two widths, a thin gold pinstripe - gone patchy with damp
    and brown where the water came down the wall."""
    _, x = coords()
    wide = (x % 32) < 12
    pin = np.abs((x % 32) - 20.0) < 1.0
    base = mix((176, 168, 146), (190, 184, 160), noise(1101, beta=2.0, low=1))
    img = np.where(wide[..., None], base * np.array((0.78, 0.86, 0.84)), base)
    img = np.where(pin[..., None], img * 0.7 + np.array((150, 120, 40)) * 0.3, img)
    damp = blur(spots(1102, 3, (14, 30)), 4.0)
    img *= (1.0 - 0.28 * damp)[..., None]
    img *= (1.0 - 0.22 * streaks(1103, 6, (30, 90)))[..., None]
    return finish(img)


@texture("wallpaper_damask", tile=0.5)
def _wallpaper_damask() -> Image:
    """An old damask: a flock medallion repeated on a half drop, the ground faded and the
    raised pattern rubbed bare in places."""
    y, x = coords()
    img = mix((104, 78, 44), (120, 90, 52), noise(1111, beta=2.0, low=1))
    motif = np.zeros((SIZE, SIZE))
    for cy, cx in ((32, 32), (96, 96), (32, 96), (96, 32)):
        small = (cy, cx) in ((32, 96), (96, 32))
        k = 0.55 if small else 1.0
        dy, dx = wrapped_dist(y, cy) / (28.0 * k), wrapped_dist(x, cx) / (20.0 * k)
        r = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(y - cy, x - cx + 1e-9)
        petal = np.cos(angle * 4.0) * 0.22 + 0.7
        motif = np.maximum(motif, np.clip((petal - r) * 8.0, 0.0, 1.0))
        motif = np.where(np.abs(r - 0.4) < 0.06, 0.0, motif)  # a ring cut through it
    wear = noise(1112, beta=1.6, low=3) > 0.2
    flock = motif * wear
    img = img * (1.0 - 0.7 * flock[..., None]) + flock[..., None] * np.array((196, 158, 90)) * 0.7
    img *= relief(flock * 0.8, 0.8)[..., None]
    img *= (1.0 - 0.2 * blur(spots(1113, 3, (10, 24)), 4.0))[..., None]
    return finish(img)


@texture("wallpaper_geo", tile=0.5)
def _wallpaper_geo() -> Image:
    """A seventies print: interlocking orange and brown arcs on cream, sun-bleached."""
    y, x = coords()
    cell = 32.0
    cy, cx = (y % cell) - cell / 2, (x % cell) - cell / 2
    r = np.sqrt(cx**2 + cy**2)
    ring = (np.abs(r - 12.0) < 3.0) | (np.abs(r - 5.0) < 2.0)
    quadrant = ((y // cell + x // cell) % 2).astype(bool)
    img = mix((214, 196, 160), (226, 210, 176), noise(1121, beta=2.0, low=1))
    _paint(img, ring & quadrant, (196, 98, 36), 0.85)
    _paint(img, ring & ~quadrant, (112, 70, 40), 0.85)
    img *= (0.86 + 0.14 * noise(1122, beta=2.4, low=1))[..., None]
    return finish(img)


@texture("wood_panel", tile=1.0)
def _wood_panel() -> Image:
    """Cheap wood-effect wall panelling: vertical boards with dark grooves, printed grain."""
    y, x = coords()
    grain = noise(1131, beta=1.4, low=2, stretch=(1.0, 0.06))
    board = np.minimum(x % 32, 32 - (x % 32))
    groove = np.clip((board - 0.6) / 1.2, 0.0, 1.0)
    tone = np.array([1.0, 0.9, 1.08, 0.95])[(x // 32).astype(int) % 4]
    img = mix((96, 60, 34), (140, 94, 56), 0.6 * grain + 0.4 * noise(1132, beta=2.0, low=1))
    img *= tone[..., None]
    img = img * (0.3 + 0.7 * groove)[..., None]
    return finish(img)


@texture("breeze_block", tile=1.2)
def _breeze_block() -> Image:
    """Painted concrete blocks: four courses a tile, the paint thick in the joints, chipped."""
    y, x = coords()
    row = (y // 32).astype(int)
    sx = (x + (row % 2) * 32) % SIZE
    jy = np.minimum(y % 32, 32 - (y % 32))
    jx = np.minimum(sx % 64, 64 - (sx % 64))
    face = np.clip((np.minimum(jx, jy) - 0.8) / 1.6, 0.0, 1.0)
    img = mix((150, 162, 170), (168, 180, 188), noise(1141, beta=1.8, low=2))
    img *= relief(face * 0.7 + noise(1142, beta=1.0, low=10) * 0.08, 1.2)[..., None]
    img *= (0.75 + 0.25 * face)[..., None]
    chip = (noise(1143, beta=1.2, low=6) > 0.78) & (face > 0.5)
    _paint(img, chip, (104, 104, 100), 0.7)
    return finish(img)


@texture("carpet", tile=0.5)
def _carpet() -> Image:
    """Worn loop-pile carpet: a fine flecked weave, paler and flattened where people walk."""
    y, x = coords()
    fleck = noise(1151, beta=0.3, low=30)
    weave = 0.5 + 0.5 * np.sin(x * np.pi / 2.0) * np.sin(y * np.pi / 2.0)
    img = mix((92, 96, 104), (126, 130, 138), 0.6 * fleck + 0.4 * weave)
    img *= (0.85 + 0.3 * noise(1152, beta=2.2, low=1))[..., None]
    img *= (1.0 - 0.3 * blur(spots(1153, 3, (8, 18)), 3.0))[..., None]
    return finish(img)


@texture("checker", tile=0.6)
def _checker() -> Image:
    """Black and white diner lino in 15 cm squares, yellowed, scuffed, a square or two
    lifting at the corner."""
    y, x = coords()
    dark = ((y // 32 + x // 32) % 2).astype(bool)
    light = mix((206, 200, 182), (226, 220, 200), noise(1161, beta=1.8, low=2))
    black = mix((26, 26, 28), (44, 44, 46), noise(1162, beta=1.8, low=2))
    img = np.where(dark[..., None], black, light)
    seam = np.maximum(grid_lines(32, 1, 0, 0), grid_lines(32, 1, 0, 1))
    img *= (1.0 - 0.25 * seam)[..., None]
    img *= (1.0 - 0.25 * blur(spots(1163, 4, (6, 16)), 2.5))[..., None]
    return finish(img)


@texture("parquet", tile=0.6)
def _parquet() -> Image:
    """Herringbone parquet: short oak blocks at right angles, each its own shade."""
    y, x = coords()
    block = 16
    cx, cy = x // block, y // block
    along = ((cx + cy) % 2).astype(bool)
    grain = np.where(
        along,
        noise(1171, beta=1.4, low=2, stretch=(0.06, 1.0)),
        noise(1172, beta=1.4, low=2, stretch=(1.0, 0.06)),
    )
    rng = np.random.default_rng(1173)
    shades = rng.uniform(0.8, 1.12, (SIZE // block, SIZE // block))
    img = (
        mix((120, 80, 44), (166, 116, 68), grain)
        * shades[cy.astype(int), cx.astype(int)][..., None]
    )
    joint = np.maximum(grid_lines(block, 1, 0, 0), grid_lines(block, 1, 0, 1))
    img *= (1.0 - 0.4 * joint)[..., None]
    return finish(img)


@texture("tile_hex", tile=0.3)
def _tile_hex() -> Image:
    """Little white hexagon tiles with grey grout gone dark by the drain."""
    y, x = coords()
    # hexagons on a grid that repeats every 32 px across and 64 px down (tileable)
    best = np.full((SIZE, SIZE), 1e9)
    second = np.full((SIZE, SIZE), 1e9)
    for oy in range(-64, SIZE + 64, 32):
        for ox in range(-32, SIZE + 32, 32):
            shift = 16 if (oy // 32) % 2 else 0
            d = np.sqrt((x - (ox + shift)) ** 2 + ((y - oy) * 1.15) ** 2)
            second = np.where(d < best, best, np.minimum(second, d))
            best = np.minimum(best, d)
    grout = (second - best) < 2.0
    img = mix((214, 218, 214), (232, 234, 230), noise(1181, beta=1.6, low=4))
    _paint(img, grout, (110, 114, 110), 1.0)
    img *= (1.0 - 0.35 * blur(spots(1182, 3, (8, 18)), 3.0))[..., None]
    return finish(img)


@texture("tile_subway", tile=0.6)
def _tile_subway() -> Image:
    """White subway tiles laid brick fashion, glossy, the grout grey and a few cracked."""
    y, x = coords()
    row = (y // 16).astype(int)
    sx = (x + (row % 2) * 16) % SIZE
    gy = np.minimum(y % 16, 16 - (y % 16))
    gx = np.minimum(sx % 32, 32 - (sx % 32))
    face = np.clip((np.minimum(gx, gy) - 0.5) / 1.0, 0.0, 1.0)
    img = mix((206, 214, 216), (232, 238, 240), noise(1191, beta=1.6, low=4))
    img = img * face[..., None] + (1.0 - face[..., None]) * np.array((104, 108, 110))
    img *= relief(face * 0.5, 1.0)[..., None]
    img *= (1.0 - 0.5 * blur(cracks(1192, 2, 8), 0.4))[..., None]
    return finish(img)


# =============================================================================== pictures
def _poster(seed: int, top: RGB, bottom: RGB) -> np.ndarray:
    y, _ = coords()
    return mix(top, bottom, y / SIZE) * (0.92 + 0.08 * noise(seed, beta=2.0, low=1))[..., None]


@texture("poster_band")
def _poster_band() -> Image:
    """A gig poster: CHROME WOLVES live at the Neon Moth - a singer's silhouette against a
    magenta sun, the band's name in fat block capitals."""
    img = _poster(1201, (40, 10, 60), (230, 40, 140))
    _paint(img, _disc(62, 64, 30), (255, 150, 60), 0.85)
    for k in range(5):  # the sun's bands
        _paint(img, _rect(66 + k * 5, 68 + k * 5, 30, 98), (230, 40, 140), 0.9)
    body = _ellipse(92, 64, 22, 12) | _disc(64, 64, 8) | _rect(76, 128, 58, 70)
    mic = _rect(56, 86, 76, 78)
    _paint(img, body | mic, (12, 4, 18))
    write(img, "CHROME", 16, 8, (250, 240, 120), scale=4)
    write(img, "WOLVES", 16, 30, (250, 240, 120), scale=4)
    write(img, "LIVE NEON MOTH", 9, 116, (255, 255, 255), scale=2)
    return finish(_aged(img, 1202))


@texture("poster_film")
def _poster_film() -> Image:
    """A film poster: a face in profile split by a red katana, BLADE CITY 2 in steel."""
    img = _poster(1211, (8, 12, 24), (30, 40, 70))
    face = _ellipse(58, 56, 34, 24) & ~_ellipse(62, 82, 16, 8)
    _paint(img, face, (60, 120, 160), 0.9)
    _paint(img, _ellipse(52, 66, 3, 5), (255, 240, 200))
    img = np.maximum(img, stroke([(100, 12), (20, 116)], 3.0)[..., None] * np.array((255, 40, 50)))
    write(img, "BLADE", 24, 96, (220, 226, 236), scale=4)
    write(img, "CITY 2", 30, 114, (255, 60, 60), scale=2)
    return finish(_aged(img, 1212, 0.3))


@texture("poster_idol")
def _poster_idol() -> Image:
    """A pop idol: big eyes, cyan hair, hearts - RIN, signed."""
    img = _poster(1221, (250, 200, 230), (160, 210, 250))
    _paint(img, _ellipse(58, 64, 40, 34), (40, 220, 230))  # hair
    _paint(img, _ellipse(66, 64, 28, 22), (250, 220, 200))  # face
    for cx in (54, 74):
        _paint(img, _ellipse(64, cx, 7, 5), (30, 30, 60))
        _paint(img, _disc(61, cx - 1, 2), (255, 255, 255))
    _paint(img, _ellipse(80, 64, 2, 6), (220, 80, 110))
    for cy, cx in ((20, 20), (30, 108), (100, 16), (110, 110)):
        _paint(img, _disc(cy, cx - 3, 4) | _disc(cy, cx + 3, 4), (255, 80, 140))
    write(img, "RIN", 46, 104, (255, 40, 130), scale=4)
    signature = stroke([(118, 84), (112, 96), (120, 104), (114, 116)], 1.5)
    img -= signature[..., None] * np.array((120, 160, 60))  # signed in marker
    return finish(_aged(img, 1222))


@texture("poster_corp")
def _poster_corp() -> Image:
    """ARASAKA - FOR A BETTER TOMORROW: a corporate poster, the smiling family on it
    defaced - horns, a moustache, a slogan sprayed across it in red."""
    img = _poster(1231, (230, 232, 236), (180, 186, 196))
    _paint(img, _rect(0, 22, 0, 128), (200, 20, 30))
    write(img, "ARASAKA", 36, 6, (255, 255, 255), scale=2)
    for cx, h in ((42, 44), (64, 52), (86, 40)):
        top = 100 - h
        _paint(img, _disc(top, cx, 8), (230, 190, 160))
        _paint(img, _rect(top + 8, 100, cx - 9, cx + 9), (40, 70, 130))
    write(img, "FOR A BETTER", 18, 104, (40, 40, 60), scale=2)
    write(img, "TOMORROW", 34, 116, (40, 40, 60), scale=1)
    spray = np.zeros((SIZE, SIZE))
    for pts in (
        [(50, 36), (42, 32)],
        [(50, 48), (42, 52)],  # horns
        [(58, 58), (62, 64), (58, 70)],  # a moustache
        [(88, 10), (76, 40), (90, 70), (74, 100), (86, 122)],  # a scrawl
    ):
        spray = np.maximum(spray, stroke(pts, 2.5))
    img = img * (1.0 - spray[..., None]) + spray[..., None] * np.array((220, 20, 40))
    write(img, "LIARS", 44, 80, (20, 20, 20), scale=3)
    return finish(_aged(img, 1232, 0.35))


@texture("mirror_cracked")
def _mirror_cracked() -> Image:
    """A bathroom mirror with a crack across it from a corner, gone cloudy at the edges."""
    y, x = coords()
    img = mix((120, 136, 150), (170, 186, 198), np.clip((x + y) / 256.0, 0.0, 1.0))
    edge = np.minimum(np.minimum(x, 127 - x), np.minimum(y, 127 - y))
    img *= (0.7 + 0.3 * np.clip(edge / 20.0, 0.0, 1.0))[..., None]
    crack = np.zeros((SIZE, SIZE))
    lines = (
        [(0, 118), (40, 80), (58, 70), (90, 40), (128, 30)],
        [(58, 70), (70, 100), (100, 128)],
        [(40, 80), (20, 60)],
    )
    for pts in lines:
        crack = np.maximum(crack, stroke(pts, 2.2))
    halo = blur(crack, 1.5)  # (the silvering lifting along it)
    img = img * (1.0 - 0.35 * halo[..., None]) + halo[..., None] * 60
    img = img * (1.0 - 0.9 * crack[..., None]) + crack[..., None] * 25
    return finish(img)


@texture("poster_fish")
def _poster_fish() -> Image:
    """A fishmonger's chart: the fish of the bay in blue ink on yellowed paper."""
    img = _canvas((226, 214, 176)).astype(float)
    ink = np.array((40, 60, 110))
    for k, (cy, size) in enumerate(((26, 1.0), (62, 0.8), (98, 1.1))):
        body = _ellipse(cy, 64, 10 * size, 34 * size)
        tail = _ellipse(cy, 64 + 40 * size, 9 * size, 6 * size)
        _paint(img, body | tail, ink, 0.75 - 0.1 * k)
        _paint(img, _disc(cy - 2 * size, 64 - 24 * size, 2.2), (240, 240, 230))
    write(img, "SAKANA", 40, 116, tuple(ink), scale=2)
    return finish(_aged(img, 1231, 0.35))


@texture("poster_sports")
def _poster_sports() -> Image:
    """A skyball team poster: the Harbour Sharks' player mid-leap, the fixture list."""
    img = _poster(1241, (0, 90, 140), (0, 30, 60))
    _paint(img, _disc(30, 90, 12), (250, 150, 20))  # the ball
    player = _ellipse(64, 56, 26, 10) | _disc(34, 60, 8) | _rect(84, 116, 44, 52)
    _paint(img, player, (240, 240, 240))
    _paint(img, _rect(56, 72, 46, 66), (240, 60, 40))  # jersey
    write(img, "SHARKS", 28, 6, (255, 255, 255), scale=3)
    write(img, "HOME 12 AWAY 9", 8, 120, (250, 200, 40))
    return finish(_aged(img, 1242))


@texture("city_map")
def _city_map() -> Image:
    """A street map of the district, pinned with routes in marker (the courier's)."""
    y, x = coords()
    img = _canvas((226, 222, 206)).astype(float)
    blocks = (np.minimum(x % 24, 24 - x % 24) > 3) & (np.minimum(y % 20, 20 - y % 20) > 3)
    _paint(img, blocks, (200, 196, 172))
    _paint(img, _rect(60, 68, 0, 128), (250, 220, 120))  # the main road
    _paint(img, _ellipse(100, 30, 20, 26), (140, 190, 220))  # the bay
    route = stroke([(10, 14), (10, 58), (64, 58), (64, 100), (112, 100)], 2.0) > 0.5
    _paint(img, route, (220, 30, 40))
    for cy, cx in ((10, 14), (64, 100), (112, 100)):
        _paint(img, _disc(cy, cx, 4), (30, 30, 200))
    return finish(_aged(img, 1251, 0.2))


@texture("art_synthwave")
def _art_synthwave() -> Image:
    """A print: a striped sun setting behind wireframe mountains over a neon grid."""
    y, x = coords()
    img = mix((20, 0, 40), (200, 40, 120), np.clip(y / 70.0, 0.0, 1.0))
    sun = _disc(52, 64, 26) & (y < 64)
    gaps = sun & ((y % 8) > 5) & (y > 40)
    _paint(img, sun & ~gaps, (255, 200, 60))
    ridge = 64 - np.abs(((x * 0.9) % 40) - 20) * 0.9
    _paint(img, (y > ridge) & (y < 66), (40, 0, 60))
    ground = y >= 66
    img[ground] = (16, 0, 30)
    depth = np.clip((y - 66) / 62.0, 0.01, 1.0)
    lines = ground & (
        (np.abs(((x - 64) / (0.2 + depth)) % 16) < 1.4)
        | (np.clip(y - 66, 0, None) ** 0.7 % 5 < 0.8)
    )
    _paint(img, lines, (0, 230, 255))
    return finish(img)


@texture("art_abstract")
def _art_abstract() -> Image:
    """A gallery print: flat blocks of colour and a black grid, a little sun-faded."""
    img = _canvas((236, 232, 220)).astype(float)
    _paint(img, _rect(8, 60, 8, 70), (210, 40, 40))
    _paint(img, _rect(70, 120, 84, 120), (30, 70, 180))
    _paint(img, _rect(96, 120, 8, 40), (240, 200, 30))
    for r in (64, 90):
        _paint(img, _rect(r, r + 5, 0, 128), (20, 20, 20))
    for c in (74, 44):
        _paint(img, _rect(0, 128, c, c + 5), (20, 20, 20))
    return finish(_aged(img, 1261, 0.15))


@texture("photo_family")
def _photo_family() -> Image:
    """A snapshot: three people on a beach at sunset, a dog, the sea behind."""
    y, _ = coords()
    img = mix((250, 170, 90), (120, 80, 140), np.clip(y / 70.0, 0.0, 1.0))
    _paint(img, y > 70, (40, 70, 110))
    _paint(img, y > 96, (200, 170, 120))
    for cx, h, shirt in (
        (40, 40, (200, 60, 60)),
        (64, 46, (60, 120, 200)),
        (88, 30, (240, 200, 60)),
    ):
        top = 108 - h
        _paint(img, _disc(top, cx, 6), (90, 60, 50))
        _paint(img, _rect(top + 6, top + 26, cx - 7, cx + 7), shirt)
        _paint(img, _rect(top + 26, 110, cx - 6, cx + 6), (50, 50, 60))
    _paint(img, _ellipse(110, 108, 5, 9), (120, 80, 40))
    return finish(_aged(img, 1271, 0.35))


@texture("photo_city")
def _photo_city() -> Image:
    """A print of the harbour skyline in the rain."""
    y, x = coords()
    img = mix((30, 40, 70), (90, 90, 120), y / SIZE)
    rng = np.random.default_rng(1281)
    for k in range(12):
        left = int(k * 11 + rng.integers(0, 4))
        top = int(rng.integers(20, 80))
        tower = _rect(top, 100, left, left + 9)
        _paint(img, tower, (20, 24, 40))
        lit = tower & ((y % 6) < 2) & ((x % 4) < 2) & (noise(1282 + k, beta=0.5, low=20) > 0.55)
        _paint(img, lit, (250, 220, 140))
    _paint(img, y >= 100, (20, 30, 50))
    img *= (1.0 - 0.25 * streaks(1283, 30, (10, 40)))[..., None]
    return finish(_aged(img, 1284, 0.2))


@texture("certificate")
def _certificate() -> Image:
    """A framed certificate: EMPLOYEE OF THE MONTH, a gold seal, a signature."""
    img = _canvas((240, 236, 222)).astype(float)
    _paint(img, _rect(4, 124, 4, 124) & ~_rect(8, 120, 8, 120), (150, 120, 40))
    write(img, "ARASAKA", 36, 16, (40, 40, 60), scale=2)
    write(img, "EMPLOYEE OF", 22, 42, (60, 60, 80), scale=2)
    write(img, "THE MONTH", 28, 56, (60, 60, 80), scale=2)
    _paint(img, _disc(96, 36, 13), (210, 170, 40))
    _paint(img, _disc(96, 36, 8), (240, 200, 70))
    img = img - stroke([(100, 64), (94, 74), (102, 84), (96, 96), (100, 110)], 1.4)[..., None] * 160
    return finish(img)


@texture("calendar")
def _calendar() -> Image:
    """A wall calendar: a picture of a beach up top, the month's grid below with days
    crossed off in red."""
    y, x = coords()
    img = _canvas((244, 242, 236)).astype(float)
    top = y < 56
    img[top] = mix((120, 200, 240), (240, 220, 160), y / 56.0)[top]
    _paint(img, top & (y > 40), (230, 210, 150))
    _paint(img, _disc(16, 100, 8), (255, 240, 160))
    write(img, "MARCH 2077", 24, 60, (200, 30, 40), scale=2)
    for r in range(5):
        for c in range(7):
            cy, cx = 76 + r * 10, 6 + c * 17
            img[cy : cy + 9, cx : cx + 16] = (226, 226, 220)
            if r * 7 + c < 17:  # the days gone by, crossed off
                cross = stroke([(cy, cx + 2), (cy + 8, cx + 14)], 1.2) > 0.5
                _paint(img, cross, (210, 30, 30))
    return finish(img)


@texture("pegboard")
def _pegboard() -> Image:
    """A tinkerer's pegboard: holes on a grid, the outlines of tools painted where they
    hang (and most of the tools hanging in them)."""
    y, x = coords()
    img = mix((170, 140, 100), (186, 154, 112), noise(1291, beta=2.0, low=1))
    holes = (wrapped_dist(y % 8, 4) < 1.2) & (wrapped_dist(x % 8, 4) < 1.2)
    _paint(img, holes, (60, 44, 30))
    tools = [
        ([(14, 16), (70, 16)], 5.0, (200, 40, 40)),  # screwdriver
        ([(14, 40), (60, 40)], 4.0, (60, 60, 70)),  # spanner
        ([(14, 64), (40, 64), (52, 74)], 5.0, (230, 200, 40)),  # soldering iron
        ([(80, 20), (112, 44)], 6.0, (40, 120, 200)),  # pliers
        ([(84, 80), (84, 116)], 9.0, (80, 80, 90)),  # a coil of wire
    ]
    for pts, width, color in tools:
        _paint(img, stroke(pts, width + 3) > 0.5, (240, 240, 235), 0.8)
        _paint(img, stroke(pts, width) > 0.5, color)
    return finish(img)


@texture("notes")
def _notes() -> Image:
    """Fridge door clutter: sticky notes, a takeaway menu, a photo under a magnet."""
    img = _canvas((210, 212, 208)).astype(float)
    for y0, x0, col in (
        (8, 10, (250, 240, 120)),
        (40, 60, (250, 160, 200)),
        (84, 20, (160, 230, 160)),
    ):
        _paint(img, _rect(y0, y0 + 28, x0, x0 + 30), col)
        for k in range(3):
            img[y0 + 6 + k * 6, x0 + 4 : x0 + 24] = (60, 60, 90)
    _paint(img, _rect(10, 60, 80, 120), (255, 255, 250))  # the menu
    write(img, "WOK", 88, 14, (200, 30, 30), scale=2)
    for k in range(5):
        img[30 + k * 5, 84:116] = (90, 90, 90)
    _paint(img, _rect(78, 118, 66, 108), (40, 60, 90))  # a photo
    _paint(img, _disc(98, 86, 9), (230, 190, 150))
    for cy, cx, col in ((10, 24, (255, 40, 40)), (42, 74, (40, 90, 255)), (80, 86, (40, 200, 60))):
        _paint(img, _disc(cy, cx, 4), col)
    return finish(img)


@texture("books")
def _books() -> Image:
    """A row of books on a shelf, seen from the front: spines of every colour and height,
    titles as pale bars, a gap or two, one leaning over."""
    y, x = coords()
    rng = np.random.default_rng(1301)
    img = _canvas((16, 12, 10)).astype(float)
    px = 0
    palette = ((150, 30, 30), (30, 60, 120), (40, 100, 60), (200, 170, 60), (90, 40, 100),
               (200, 200, 190), (30, 30, 34), (180, 90, 30), (60, 120, 130))  # fmt: skip
    while px < SIZE:
        w = int(rng.integers(5, 12))
        if rng.random() < 0.07:
            px += w  # a gap
            continue
        top = int(rng.integers(0, 36))
        col = np.array(palette[int(rng.integers(len(palette)))], dtype=float)
        col *= rng.uniform(0.75, 1.1)
        spine = _rect(top, SIZE, px, min(px + w - 1, SIZE))
        _paint(img, spine, tuple(col))
        for band in (top + 8, SIZE - 14):
            _paint(img, _rect(band, band + 2, px + 1, min(px + w - 2, SIZE)), (230, 210, 120), 0.7)
        _paint(img, _rect(top + 16, top + 16 + int(rng.integers(14, 40)), px + w // 2 - 1,
                          px + w // 2 + 1), (235, 230, 215), 0.8)  # fmt: skip
        px += w
    img *= relief(blur((img.sum(axis=2) > 60).astype(float), 0.6) * 0.6, 1.0)[..., None]
    return finish(img)


@texture("cereal")
def _cereal() -> Image:
    """The front of a cereal box: CHOCO BOOM, a bowl, a grinning mascot."""
    img = _poster(1311, (250, 190, 20), (240, 120, 20))
    write(img, "CHOCO", 24, 10, (120, 40, 10), scale=4)
    write(img, "BOOM!", 24, 32, (200, 20, 30), scale=4)
    _paint(img, _ellipse(98, 64, 16, 40) & (coords()[0] > 94), (250, 250, 250))
    _paint(img, _ellipse(94, 64, 6, 36), (110, 60, 30))
    _paint(img, _disc(66, 100, 12), (120, 200, 60))
    _paint(img, _disc(62, 96, 3) | _disc(62, 104, 3), (255, 255, 255))
    return finish(img)


@texture("rug_persian")
def _rug_persian() -> Image:
    """A threadbare red rug with a medallion and a patterned border."""
    y, x = coords()
    img = mix((120, 24, 30), (150, 34, 40), noise(1321, beta=1.6, low=3))
    border = np.minimum(np.minimum(x, 127 - x), np.minimum(y, 127 - y)) < 14
    _paint(img, border, (40, 40, 80), 0.8)
    _paint(img, border & ((x + y) % 8 < 3), (220, 180, 90), 0.7)
    dx, dy = np.abs(x - 63.5) / 40.0, np.abs(y - 63.5) / 40.0
    medallion = dx + dy < 1.0
    _paint(img, medallion, (220, 180, 90), 0.7)
    _paint(img, dx + dy < 0.6, (40, 40, 80), 0.8)
    _paint(img, dx + dy < 0.25, (150, 34, 40), 0.9)
    img *= (0.8 + 0.2 * noise(1322, beta=0.4, low=20))[..., None]  # the pile worn thin
    return finish(img)


@texture("rug_stripes")
def _rug_stripes() -> Image:
    """A cheap woven rag rug in bands of colour."""
    y, x = coords()
    band = (y // 10).astype(int) % 5
    colors = np.array(
        ((200, 90, 40), (240, 220, 170), (60, 110, 130), (240, 220, 170), (130, 50, 70))
    )
    img = colors[band].astype(float)
    img *= (0.8 + 0.2 * (np.sin(x * 1.3) * 0.5 + 0.5))[..., None]
    img *= (0.85 + 0.15 * noise(1331, beta=1.0, low=8))[..., None]
    return finish(img)


@texture("rug_shag")
def _rug_shag() -> Image:
    """A fluffy shag rug gone matted - lime green, of course."""
    img = mix((110, 150, 40), (170, 210, 70), noise(1341, beta=0.2, low=30))
    img *= (0.8 + 0.2 * noise(1342, beta=2.0, low=1))[..., None]
    return finish(img)


# =============================================================================== windows
def _sky(night: bool) -> np.ndarray:
    y, _ = coords()
    if night:
        return mix((10, 8, 30), (60, 20, 70), np.clip(y / 90.0, 0.0, 1.0))
    return mix((150, 140, 110), (200, 170, 120), np.clip(y / 90.0, 0.0, 1.0))  # smog


def _towers(img: np.ndarray, seed: int, night: bool, base: int = 128) -> None:
    y, x = coords()
    rng = np.random.default_rng(seed)
    left = -4
    while left < SIZE:
        w = int(rng.integers(12, 26))
        top = int(rng.integers(10, 70))
        tower = _rect(top, base, left, left + w)
        shade = rng.uniform(0.7, 1.2)
        _paint(img, tower, tuple(np.array((18, 18, 30) if night else (90, 92, 100)) * shade))
        lit = tower & ((y - top) % 7 < 3) & ((x - left) % 5 < 3)
        lit &= noise(seed + left % 97 + 3, beta=0.3, low=20) > (0.45 if night else 0.8)
        _paint(img, lit, (250, 210, 130) if night else (60, 60, 70))
        if rng.random() < 0.5:  # a sign up the side
            sy = top + int(rng.integers(6, 30))
            col = ((255, 40, 180), (40, 230, 255), (255, 170, 40))[int(rng.integers(3))]
            _paint(img, _rect(sy, sy + 24, left + 2, left + 6), col if night else (150, 150, 160))
        left += w + int(rng.integers(0, 5))


def _window_view(seed: int, night: bool, alley: bool) -> np.ndarray:
    img = _sky(night)
    if alley:
        # the alley: the tenement across the way fills the view, a fire escape, a lantern
        y, x = coords()
        wall = mix((60, 34, 30), (80, 46, 40), noise(seed, beta=1.8, low=2))
        if night:
            wall *= 0.45
        img = wall.copy()
        for cx in (20, 64, 108):
            win = _rect(20, 56, cx - 12, cx + 12)
            _paint(img, win, (240, 190, 110) if night and cx != 64 else (40, 44, 56))
        _paint(img, _rect(70, 74, 0, 128) | ((x % 12 < 2) & (y > 74) & (y < 96)), (30, 30, 34))
        if night:
            glow = blur(_disc(110, 40, 6).astype(float), 6.0)
            img += glow[..., None] * np.array((255, 60, 160)) * 2.0
            _paint(img, _disc(110, 40, 6), (255, 120, 200))
    else:
        _towers(img, seed, night)
    # rain on the glass
    img *= (1.0 - 0.2 * streaks(seed + 7, 40, (6, 26)))[..., None]
    img += streaks(seed + 8, 30, (4, 14))[..., None] * (40.0 if night else 20.0)
    return img


@texture("view_city_night")
def _view_city_night() -> Image:
    """Out of a street window after dark: the towers across the road lit up, signs glowing,
    rain running down the glass."""
    return finish(_window_view(1401, night=True, alley=False))


@texture("view_city_day")
def _view_city_day() -> Image:
    """...and by day: the same towers grey under a smog-brown sky."""
    return finish(_window_view(1401, night=False, alley=False))


@texture("view_alley_night")
def _view_alley_night() -> Image:
    """Out of an alley window: the brick of the building across, lit windows, a fire
    escape, a pink lantern."""
    return finish(_window_view(1411, night=True, alley=True))


@texture("view_alley_day")
def _view_alley_day() -> Image:
    return finish(_window_view(1411, night=False, alley=True))


# =============================================================================== television
TV_CHANNELS: dict[str, tuple[int, float]] = {
    # channel -> (frames in its loop, frames a second)
    "static": (8, 18.0),
    "news": (16, 6.0),
    "cartoon": (8, 7.0),
    "music": (6, 5.0),
    "ads": (6, 2.5),
}
"""Every channel's loop of frames: ``tv_<channel>_<k>`` for k in 0 .. frames - 1."""


def tv_frame(channel: str, k: int) -> str:
    """The texture name of frame ``k`` (any integer: it loops) of ``channel``."""
    frames, _ = TV_CHANNELS[channel]
    return f"tv_{channel}_{k % frames}"


def _crt(img: np.ndarray) -> np.ndarray:
    """A picture on a tube: soft scanlines, darker corners."""
    y, x = coords()
    vignette = 1.0 - 0.35 * (((x - 64) / 70) ** 2 + ((y - 64) / 70) ** 2)
    img = img * vignette[..., None]
    return img * (1.0 - 0.1 * (np.arange(SIZE) % 3 == 0))[:, None, None]


def _static_frame(k: int) -> Image:
    """A dead channel, frame ``k``: fresh snow every frame, a dark bar rolling down the
    picture and a bright tear sliding sideways - so that, stepped through, it crawls."""
    rng = np.random.default_rng(1500 + k)
    y, x = coords()
    snow = rng.uniform(0.0, 1.0, (SIZE, SIZE))
    snow = 0.6 * snow + 0.4 * rng.uniform(0.0, 1.0, (SIZE, 1))  # streaky, line by line
    roll = ((y - k * 16) % SIZE) / SIZE  # the bar rolls down 16 rows a frame
    bars = 0.6 + 0.4 * np.clip(np.abs(roll - 0.5) * 3.0, 0.0, 1.0)
    tear = np.abs(y - ((k * 37) % SIZE)) < 3
    snow = np.where(tear, np.roll(snow, 9, axis=1) * 1.3, snow)
    img = (snow * bars)[..., None] * np.array((210, 218, 236)) + 14
    return finish(_crt(img))


def _news_frame(k: int) -> Image:
    """NCN news: the anchor at her desk (talking - the mouth moves), a picture in the
    picture of smoke over the arcology, the channel's bug, and a ticker running along
    the bottom (16 frames: two letters a frame)."""
    y, x = coords()
    img = mix((10, 40, 90), (20, 80, 150), y / SIZE)
    _paint(img, ((x + y // 2) % 24) < 2, (40, 110, 180), 0.5)
    # the anchor
    _paint(img, _ellipse(86, 44, 26, 22), (40, 40, 50))  # suit
    _paint(img, _rect(66, 80, 40, 48), (240, 240, 240))  # collar
    _paint(img, _ellipse(50, 44, 15, 11), (220, 180, 150))  # face
    _paint(img, _ellipse(40, 44, 10, 13) & (y < 44), (40, 20, 10))  # hair
    for cx in (40, 49):
        _paint(img, _disc(49, cx, 1.6), (20, 20, 30))
    open_ = 1.8 if k % 3 == 1 else (1.0 if k % 3 == 2 else 0.4)
    _paint(img, _ellipse(57, 45, open_, 3), (140, 40, 50))
    _paint(img, _rect(92, 104, 0, 128), (180, 190, 200))  # the desk
    # the picture in the picture: the arcology burning
    _paint(img, _rect(20, 62, 70, 120), (60, 50, 60))
    _paint(img, _rect(36, 62, 84, 100), (30, 30, 40))
    flame = _ellipse(36, 92, 8 + (k % 4), 6) | _ellipse(30, 96, 5, 4)
    _paint(img, flame, (250, 140 + 20 * (k % 3), 40))
    write(img, "LIVE", 73, 23, (255, 255, 255))
    _paint(img, _rect(23, 28, 88, 93), (230, 30, 30) if k % 2 == 0 else (90, 20, 20))
    # the bug and the ticker
    _paint(img, _rect(4, 16, 4, 34), (200, 20, 30))
    write(img, "NCN", 8, 6, (255, 255, 255), scale=2)
    _paint(img, _rect(106, 124, 0, 128), (230, 230, 235))
    _paint(img, _rect(106, 124, 0, 22), (200, 20, 30))
    write(img, "NEWS", 3, 112, (255, 255, 255))
    band = np.zeros((5, SIZE, 3))
    msg = "ARCOLOGY FIRE DAY 3 -- NIKKEI 4410 +2% -- RAIN ALL WEEK -- "
    write(band, msg, 128 - k * 8, 0, (20, 20, 40), wrap=False)
    lit = band.sum(axis=2) > 0
    region = img[113:118, 24:128]
    region[lit[:, 24:128]] = (20, 20, 40)
    return finish(_crt(img))


def _cartoon_frame(k: int) -> Image:
    """Saturday-morning cartoons: a yellow duck in a cape bounces across a hillside chased
    by a grumpy robot, clouds drifting behind."""
    y, x = coords()
    img = mix((120, 200, 255), (200, 235, 255), y / 80.0)
    hills = y > 88 + 8 * np.sin(x / 18.0 + k * 0.2)
    _paint(img, hills, (90, 200, 80))
    for cx in (20, 80):
        ccx = (cx + k * 4) % 150 - 10
        _paint(img, _ellipse(24, ccx, 7, 14) | _ellipse(20, ccx + 8, 7, 10), (255, 255, 255))
    hop = abs((k % 8) - 4) * 5  # up and down, four frames each way
    dx = 20 + k * 8
    body = _ellipse(78 - hop, dx, 14, 16)
    head = _disc(58 - hop, dx + 10, 10)
    _paint(img, _ellipse(80 - hop, dx - 14, 10, 8), (230, 30, 40))  # the cape
    _paint(img, body | head, (255, 220, 40))
    _paint(img, _ellipse(60 - hop, dx + 22, 4, 7), (255, 130, 20))  # the beak
    _paint(img, _disc(55 - hop, dx + 13, 3), (255, 255, 255))
    _paint(img, _disc(55 - hop, dx + 14, 1.5), (0, 0, 0))
    rx = dx - 46
    _paint(img, _rect(70, 96, rx - 10, rx + 10), (150, 150, 170))  # the robot
    _paint(img, _rect(60, 70, rx - 7, rx + 7), (120, 120, 140))
    _paint(img, _rect(63, 66, rx - 4, rx + 4), (255, 40, 40) if k % 2 else (120, 0, 0))
    return finish(_crt(img))


def _music_frame(k: int) -> Image:
    """A music channel: a singer silhouetted against pulsing neon, the equaliser jumping."""
    y, x = coords()
    rng = np.random.default_rng(1600 + k)
    hue = ((255, 40, 180), (40, 220, 255), (180, 60, 255))[k % 3]
    img = mix((10, 0, 20), tuple(c * 0.5 for c in hue), np.clip(1.0 - np.abs(x - 64) / 64.0, 0, 1))
    for r in range(3):
        ring = np.abs(np.sqrt((y - 50) ** 2 + (x - 64) ** 2) - (14 + r * 16 + k * 3)) < 2
        _paint(img, ring, hue, 0.6)
    singer = _disc(40, 64, 9) | _ellipse(78, 64, 26, 13) | _rect(58, 64, 70, 74)
    _paint(img, singer, (6, 0, 12))
    for b in range(16):
        h = int(rng.integers(4, 26))
        _paint(img, _rect(124 - h, 124, 2 + b * 8, 8 + b * 8), hue)
    return finish(_crt(img))


def _ads_frame(k: int) -> Image:
    """The adverts: BUY CHROME - a flashing price, a spinning cyber-arm, a phone number."""
    y, x = coords()
    flash = k % 2 == 0
    img = _canvas((250, 220, 0) if flash else (240, 20, 120)).astype(float)
    _paint(img, ((x + y + k * 6) % 20) < 4, (255, 255, 255), 0.25)
    write(img, "BUY", 10, 8, (20, 20, 20) if flash else (255, 255, 255), scale=4)
    write(img, "CHROME", 10, 30, (20, 20, 20) if flash else (255, 255, 255), scale=3)
    width = 10 + 8 * abs((k % 6) - 3)  # the arm turning
    _paint(img, _rect(56, 100, 84 - width // 2, 84 + width // 2), (170, 180, 200))
    _paint(img, _rect(50, 58, 84 - width // 2 - 3, 84 + width // 2 + 3), (40, 230, 255))
    write(img, "$99", 10, 70, (200, 0, 40) if flash else (255, 240, 0), scale=4)
    write(img, "CALL 555-0199", 12, 114, (20, 20, 20) if flash else (255, 255, 255))
    return finish(_crt(img))


def _register_channel(channel: str, make) -> None:
    frames, _ = TV_CHANNELS[channel]
    for k in range(frames):
        texture(f"tv_{channel}_{k}")(lambda k=k: make(k))


_register_channel("static", _static_frame)
_register_channel("news", _news_frame)
_register_channel("cartoon", _cartoon_frame)
_register_channel("music", _music_frame)
_register_channel("ads", _ads_frame)


@texture("display_0000")
def _display_0000() -> Image:
    """A microwave's clock that was never set: 00:00 in green segments on black glass."""
    img = _canvas((6, 14, 8)).astype(float)
    write(img, "00:00", 14, 44, (90, 255, 140), scale=6)
    return finish(blur(img, 0.6) * 0.5 + img * 0.6)


@texture("tv_off")
def _tv_off() -> Image:
    """A set that is switched off: dark glass with the room's light lying on it."""
    y, x = coords()
    img = mix((8, 10, 12), (18, 20, 24), np.clip((x + y) / 256.0, 0, 1))
    sheen = np.clip(1.0 - np.abs((x - y * 0.6) - 40) / 10.0, 0.0, 1.0) * (y < 70)
    img += sheen[..., None] * 18.0
    return finish(img)


__all__ = ["TV_CHANNELS", "text_width", "tv_frame", "write"]
