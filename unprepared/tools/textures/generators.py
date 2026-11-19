"""Procedural texture generators for Unprepared (tooling, not part of the game package).

Every texture is a 128x128 tileable PNG built from noise and simple shapes so
the art can be regenerated or tweaked without an image editor:

    python tools/textures/gen_textures.py            # writes assets/textures/*.png
    python tools/textures/gen_textures.py --seed 3   # a different variation

The game itself only *loads* PNGs (``unprepared.rendering.textures``); if one is
missing it uses a magenta placeholder so the problem is obvious.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict

import pygame

SIZE = 128
TEXTURE_KEYS = ("ground", "building", "wall", "guard", "npc", "wood", "bed", "shelf", "door", "ore", "skin",
                "flame", "sun", "sun_glow", "moon", "stone", "torch", "roof", "roof_slate", "iron", "lockbox",
                "lockbox_side", "wax", "cobble", "cobble_edge", "cloth", "leather", "water", "coals",
                "sign_smith", "sign_store", "sign_tailor", "sign_inn",
                "bark", "leaves", "pine", "barrel", "log_end", "crate", "hay", "thatch", "cobble_cracked",
                "cobble_cracked_edge", "wall_cracked", "planks_rot", "crack_decal", "stain_decal", "dirt", "dirt_edge",
                "rock", "grate", "marble", "carpet", "plaster", "plaster_timber", "field", "field_edge",
                "stream", "lake", "sand", "snow", "cliff", "cave_rock", "cave_floor", "mushroom",
                "bone", "notice", "sign_arm", "plaque", "parchment", "banner_greyhold", "banner_deep",
                "ashlar", "manor_plaster", "window_manor", "window_cottage", "window_lancet", "painting_portrait",
                "painting_landscape", "rug", "rug_blue", "tile_floor", "brick", "books", "clock_face", "wallpaper",
                "flowers", "gilt", "sign_bakery")


# ----------------------------------------------------------------------------- helpers
def _noise(rng: random.Random, amount: int) -> int:
    return rng.randint(-amount, amount)


def _clamp(v: float) -> int:
    return max(0, min(255, int(v)))


def _fill_noise(surf: pygame.Surface, base, amount: int, rng: random.Random) -> None:
    w, h = surf.get_size()
    for y in range(h):
        for x in range(w):
            n = _noise(rng, amount)
            surf.set_at((x, y), (_clamp(base[0] + n), _clamp(base[1] + n), _clamp(base[2] + n)))


def _value_noise(rng: random.Random, w: int, h: int, cells: int) -> list:
    """Smooth tileable value noise in [0, 1] (bilinear over a cells x cells lattice)."""
    lattice = [[rng.random() for _ in range(cells)] for _ in range(cells)]
    out = []
    for y in range(h):
        fy = y / h * cells
        iy, ty = int(fy), fy - int(fy)
        row = []
        for x in range(w):
            fx = x / w * cells
            ix, tx = int(fx), fx - int(fx)
            a = lattice[iy % cells][ix % cells]
            b = lattice[iy % cells][(ix + 1) % cells]
            c = lattice[(iy + 1) % cells][ix % cells]
            d = lattice[(iy + 1) % cells][(ix + 1) % cells]
            sx, sy = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
            row.append((a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy)
        out.append(row)
    return out


# ----------------------------------------------------------------------------- generators
def gen_ground(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    patches = _value_noise(rng, SIZE, SIZE, 6)
    for y in range(SIZE):
        for x in range(SIZE):
            p = patches[y][x]
            g = 95 + p * 55 + _noise(rng, 14)
            r = 38 + p * 30 + _noise(rng, 10)
            surf.set_at((x, y), (_clamp(r), _clamp(g), _clamp(28 + p * 12)))
    for _ in range(260):                                   # grass blades / clover specks
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        shade = rng.choice([(70, 140, 45), (120, 175, 70), (55, 105, 40)])
        surf.set_at((x, y), shade)
        surf.set_at((x, (y + 1) % SIZE), shade)
    for _ in range(18):                                    # small stones / earth
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, (110, 100, 80), (x, y), rng.randint(1, 2))
    return surf


def gen_stone_blocks(rng: random.Random, base=(128, 126, 120), mortar=(70, 66, 60), rows=4) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, base, 12, rng)
    shade = _value_noise(rng, SIZE, SIZE, 8)
    for y in range(SIZE):
        for x in range(SIZE):
            c = surf.get_at((x, y))
            k = 0.8 + shade[y][x] * 0.4
            surf.set_at((x, y), (_clamp(c.r * k), _clamp(c.g * k), _clamp(c.b * k)))
    block_h = SIZE // rows
    block_w = SIZE // 2
    for r in range(rows):
        y = r * block_h
        pygame.draw.line(surf, mortar, (0, y), (SIZE, y), 3)
        offset = (block_w // 2) if r % 2 else 0
        for bx in range(offset, SIZE + block_w, block_w):
            pygame.draw.line(surf, mortar, (bx % SIZE, y), (bx % SIZE, y + block_h), 3)
            # bevel highlight on the top-left of each block
            pygame.draw.line(surf, tuple(_clamp(c + 25) for c in base), ((bx + 3) % SIZE, y + 3), ((bx + block_w - 4) % SIZE, y + 3), 1)
    pygame.draw.line(surf, mortar, (0, SIZE - 1), (SIZE, SIZE - 1), 2)
    return surf


def gen_planks(rng: random.Random, base=(142, 92, 48), gap=(62, 38, 18), planks=5, vertical=False) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, base, 10, rng)
    plank = SIZE // planks
    for p in range(planks):
        start = p * plank
        tone = _noise(rng, 18)
        for i in range(plank):
            for j in range(SIZE):
                x, y = (start + i, j) if vertical else (j, start + i)
                c = surf.get_at((x, y))
                grain = 12 * math.sin(j / 7.0 + p * 2.1) + _noise(rng, 6)
                surf.set_at((x, y), (_clamp(c.r + tone + grain), _clamp(c.g + tone * 0.8 + grain * 0.7), _clamp(c.b + tone * 0.5 + grain * 0.4)))
        if vertical:
            pygame.draw.line(surf, gap, (start, 0), (start, SIZE), 2)
        else:
            pygame.draw.line(surf, gap, (0, start), (SIZE, start), 2)
        for _ in range(2):                                  # nails
            nx, ny = rng.randrange(4, SIZE - 4), start + plank // 2
            if vertical:
                nx, ny = start + plank // 2, rng.randrange(4, SIZE - 4)
            pygame.draw.circle(surf, (40, 38, 36), (nx, ny), 1)
    return surf


def gen_door(rng: random.Random) -> pygame.Surface:
    surf = gen_planks(rng, base=(120, 78, 40), gap=(50, 30, 14), planks=4, vertical=True)
    band = (58, 60, 66)
    for y in (SIZE // 5, SIZE * 4 // 5):
        pygame.draw.rect(surf, band, (0, y - 5, SIZE, 10))
        for bx in range(8, SIZE, 16):
            pygame.draw.circle(surf, (110, 112, 118), (bx, y), 2)
    pygame.draw.circle(surf, (190, 160, 60), (SIZE - 22, SIZE // 2), 5)    # handle
    pygame.draw.circle(surf, (60, 50, 20), (SIZE - 22, SIZE // 2), 2)
    return surf


def gen_shelf(rng: random.Random) -> pygame.Surface:
    surf = gen_planks(rng, base=(112, 74, 38), gap=(48, 30, 14), planks=3, vertical=False)
    for y in range(SIZE // 3 - 6, SIZE, SIZE // 3):
        pygame.draw.rect(surf, (40, 26, 12), (0, y, SIZE, 6))                  # shelf boards (shadow)
        for _ in range(4):                                                        # goods on the shelf
            x = rng.randrange(6, SIZE - 16)
            col = rng.choice([(160, 40, 40), (60, 90, 150), (200, 170, 60), (90, 140, 70), (200, 200, 210)])
            pygame.draw.rect(surf, col, (x, y - rng.randint(10, 16), rng.randint(6, 12), rng.randint(8, 14)))
    return surf


def gen_bed(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (150, 40, 40), 8, rng)                                     # blanket
    for y in range(0, SIZE, 12):
        pygame.draw.line(surf, (120, 30, 30), (0, y), (SIZE, y), 1)
    for x in range(0, SIZE, 12):
        pygame.draw.line(surf, (120, 30, 30), (x, 0), (x, SIZE), 1)
    pygame.draw.rect(surf, (230, 225, 205), (0, 0, SIZE, 30))                    # pillow / sheet
    pygame.draw.rect(surf, (200, 195, 175), (12, 6, SIZE - 24, 18), 2)
    return surf


def gen_wood(rng: random.Random) -> pygame.Surface:
    return gen_planks(rng, base=(96, 62, 30), gap=(40, 24, 10), planks=4, vertical=False)


def gen_ore(rng: random.Random) -> pygame.Surface:
    """Grey-brown rock with layered strata, rust-coloured iron veins and a few metallic glints."""
    surf = pygame.Surface((SIZE, SIZE))
    coarse = _value_noise(rng, SIZE, SIZE, 4)
    fine = _value_noise(rng, SIZE, SIZE, 16)
    for y in range(SIZE):
        for x in range(SIZE):
            strata = 0.5 + 0.5 * math.sin((y + 18 * coarse[y][x]) / 9.0)            # bands of sediment
            v = 48 + coarse[y][x] * 50 + fine[y][x] * 30 + strata * 18 + _noise(rng, 6)
            surf.set_at((x, y), (_clamp(v * 1.02), _clamp(v * 0.96), _clamp(v * 0.9)))
    for _ in range(7):                                                            # rusty iron veins
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        for _step in range(rng.randint(12, 30)):
            pygame.draw.circle(surf, rng.choice([(146, 78, 34), (168, 94, 40), (120, 62, 28)]), (x % SIZE, y % SIZE), rng.randint(1, 2))
            x += rng.randint(-3, 3)
            y += rng.randint(-2, 3)
    for _ in range(22):                                                           # metallic glints
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, (215, 205, 190), (x, y), 1)
    for _ in range(10):                                                           # dark cracks
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.line(surf, (22, 20, 18), (x, y), (x + rng.randint(-16, 16), y + rng.randint(-16, 16)), 1)
    return surf


def gen_roof(rng: random.Random, base=(128, 58, 36), dark=(78, 34, 20)) -> pygame.Surface:
    """Overlapping rows of shingles."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, base, 10, rng)
    rows, cols = 8, 8
    rh, cw = SIZE // rows, SIZE // cols
    for r in range(rows):
        y = r * rh
        offset = (cw // 2) if r % 2 else 0
        pygame.draw.line(surf, dark, (0, y), (SIZE, y), 2)
        for c in range(-1, cols + 1):
            x = (c * cw + offset) % SIZE
            pygame.draw.line(surf, dark, (x, y), (x, y + rh - 2), 1)
            tone = _noise(rng, 16)
            pygame.draw.rect(surf, tuple(_clamp(v + tone) for v in base), (x + 1, y + 2, cw - 2, rh - 4))
    return surf


def gen_roof_slate(rng: random.Random) -> pygame.Surface:
    return gen_roof(rng, base=(78, 84, 96), dark=(40, 44, 52))


def gen_guard(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (120, 122, 130), 8, rng)                                   # steel
    for y in range(0, SIZE, 10):                                                  # plate rows
        pygame.draw.line(surf, (80, 82, 90), (0, y), (SIZE, y), 2)
    pygame.draw.rect(surf, (150, 30, 30), (SIZE // 2 - 16, 0, 32, SIZE))         # red tabard stripe
    pygame.draw.rect(surf, (200, 170, 50), (SIZE // 2 - 6, 18, 12, 12))          # crest
    pygame.draw.rect(surf, (60, 40, 25), (0, SIZE * 3 // 4, SIZE, 6))            # belt
    return surf


def gen_npc(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (90, 70, 120), 10, rng)                                    # tunic
    for y in range(0, SIZE, 6):
        pygame.draw.line(surf, (78, 60, 105), (0, y), (SIZE, y), 1)              # weave
    pygame.draw.rect(surf, (70, 45, 25), (0, SIZE * 5 // 8, SIZE, 8))             # belt
    pygame.draw.rect(surf, (190, 160, 60), (SIZE // 2 - 5, SIZE * 5 // 8 - 1, 10, 10))   # buckle
    pygame.draw.rect(surf, (60, 50, 45), (0, SIZE * 5 // 8 + 8, SIZE, SIZE))     # trousers
    return surf


def gen_skin(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (215, 170, 135), 6, rng)
    pygame.draw.rect(surf, (80, 50, 25), (0, 0, SIZE, 26))                       # hair
    return surf


def gen_flame(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    for y in range(SIZE):
        for x in range(SIZE):
            t = 1.0 - y / SIZE
            flicker = _noise(rng, 10)
            surf.set_at((x, y), (_clamp(255), _clamp(120 + t * 120 + flicker), _clamp(20 + t * 60)))
    return surf


def gen_torch(rng: random.Random) -> pygame.Surface:
    surf = gen_planks(rng, base=(70, 45, 25), gap=(40, 24, 10), planks=2, vertical=True)
    pygame.draw.rect(surf, (40, 30, 20), (0, 0, SIZE, SIZE // 4))
    return surf


def gen_disc(color_center, color_edge) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    c = SIZE / 2
    for y in range(SIZE):
        for x in range(SIZE):
            d = math.hypot(x - c + 0.5, y - c + 0.5) / c
            if d <= 1.0:
                t = d * d
                col = tuple(_clamp(color_center[i] + (color_edge[i] - color_center[i]) * t) for i in range(3))
                alpha = 255 if d < 0.85 else _clamp(255 * (1.0 - (d - 0.85) / 0.15))
                surf.set_at((x, y), (*col, alpha))
    return surf


def gen_sun_glow(rng: random.Random) -> pygame.Surface:
    """Soft radial halo drawn (additively) behind the sun disc: bright core fading to nothing."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    c = SIZE / 2
    for y in range(SIZE):
        for x in range(SIZE):
            d = math.hypot(x - c + 0.5, y - c + 0.5) / c
            if d < 1.0:
                t = (1.0 - d) ** 2.2
                surf.set_at((x, y), (255, _clamp(235 + 20 * t), _clamp(150 + 90 * t), _clamp(215 * t)))
            else:
                surf.set_at((x, y), (255, 240, 180, 0))
    return surf


def gen_iron(rng: random.Random) -> pygame.Surface:
    """Dark wrought iron for cell bars and fittings."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (52, 54, 60), 9, rng)
    for _ in range(40):                                                           # pitting and rust specks
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, rng.choice([(40, 40, 44), (84, 60, 40), (70, 72, 78)]), (x, y), rng.randint(1, 2))
    return surf


def gen_cobble(rng: random.Random, alpha_edge: bool = False) -> pygame.Surface:
    """Rounded grey-brown cobbles in dark mortar; the edge variant fades to transparent towards
    the top of the image so the paving blends into the grass along the road's sides."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    _fill_noise(surf, (58, 52, 44), 6, rng)                                      # mortar / packed earth
    stones = []
    step = 16
    for row in range(SIZE // step + 1):
        for col in range(SIZE // step + 1):
            x = col * step + (step // 2 if row % 2 else 0) + rng.randint(-2, 2)
            y = row * step + rng.randint(-2, 2)
            stones.append((x % SIZE, y % SIZE, rng.randint(5, 7), rng.randint(4, 6), rng.randint(-12, 12)))
    for x, y, rx, ry, tone in stones:
        base = (118 + tone, 110 + tone, 98 + tone)
        for ox in (-SIZE, 0, SIZE):                                              # tileable: wrap the stones
            for oy in (-SIZE, 0, SIZE):
                pygame.draw.ellipse(surf, base, (x + ox - rx, y + oy - ry, 2 * rx, 2 * ry))
                pygame.draw.ellipse(surf, (_clamp(base[0] + 22), _clamp(base[1] + 22), _clamp(base[2] + 18)),
                                    (x + ox - rx + 2, y + oy - ry + 1, rx, ry))  # highlight
    if alpha_edge:
        for y in range(SIZE):
            t = y / (SIZE - 1)                                                   # 0 at the top (outer) -> 1 inner
            a = _clamp(255 * (t * t))
            for x in range(SIZE):
                c = surf.get_at((x, y))
                surf.set_at((x, y), (c.r, c.g, c.b, a))
    return surf


def gen_wax(rng: random.Random) -> pygame.Surface:
    """Candle wax: warm off-white with faint drips."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (236, 228, 205), 6, rng)
    for _ in range(10):
        x = rng.randrange(SIZE)
        pygame.draw.line(surf, (220, 210, 185), (x, 0), (x, rng.randrange(20, SIZE)), 2)
    return surf


def gen_lockbox_side(rng: random.Random) -> pygame.Surface:
    """The plain faces of a strongbox: dark oak planks with riveted iron bands (no lock)."""
    surf = gen_planks(rng, base=(92, 60, 32), gap=(40, 24, 10), planks=4, vertical=False)
    band = (58, 60, 66)
    rivet = (120, 122, 128)
    for x in (SIZE // 6, SIZE * 5 // 6):                                          # vertical iron bands
        pygame.draw.rect(surf, band, (x - 7, 0, 14, SIZE))
        for ry in range(10, SIZE, 18):
            pygame.draw.circle(surf, rivet, (x, ry), 2)
    pygame.draw.rect(surf, band, (0, 0, SIZE, 9))                                  # lid edge
    pygame.draw.rect(surf, band, (0, SIZE - 9, SIZE, 9))
    return surf


def gen_lockbox(rng: random.Random) -> pygame.Surface:
    """The front of a strongbox: the banded planks plus a hasp and a hanging brass padlock -
    the standard look for lockboxes. Drawn in *world* orientation (image top = box top):
    the curved shackle sits on top, the lock body hangs below it."""
    surf = gen_lockbox_side(rng)
    cx, cy = SIZE // 2, SIZE // 2
    pygame.draw.rect(surf, (70, 72, 78), (cx - 16, cy - 16, 32, 30))               # hasp plate
    pygame.draw.rect(surf, (150, 125, 40), (cx - 7, cy - 14, 14, 14), 3)           # shackle (curved bar, on top)
    pygame.draw.rect(surf, (190, 160, 60), (cx - 9, cy - 4, 18, 16))               # brass padlock body, hanging below
    pygame.draw.circle(surf, (40, 30, 10), (cx, cy + 5), 2)                        # keyhole
    return surf


def gen_cloth(rng: random.Random) -> pygame.Surface:
    """Undyed woven cloth (light, so the renderer can tint it any colour)."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (205, 200, 192), 7, rng)
    for y in range(0, SIZE, 4):
        pygame.draw.line(surf, (188, 183, 175), (0, y), (SIZE, y), 1)          # weft
    for x in range(0, SIZE, 4):
        pygame.draw.line(surf, (196, 191, 183), (x, 0), (x, SIZE), 1)          # warp
    return surf


def gen_leather(rng: random.Random) -> pygame.Surface:
    """Worn brown leather with a few scuffs and stitched edges."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (96, 62, 36), 12, rng)
    for _ in range(40):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.line(surf, (78, 50, 28), (x, y), (x + rng.randint(-6, 6), y + rng.randint(-3, 3)), 1)
    for x in range(2, SIZE, 6):                                                  # stitching along the top and bottom
        pygame.draw.line(surf, (170, 140, 90), (x, 3), (x + 3, 3), 1)
        pygame.draw.line(surf, (170, 140, 90), (x, SIZE - 4), (x + 3, SIZE - 4), 1)
    return surf


def gen_water(rng: random.Random) -> pygame.Surface:
    """Dark well water with faint ripples and a few glints."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (22, 48, 66), 6, rng)
    for i in range(6):
        r = 10 + i * 9
        pygame.draw.circle(surf, (34, 66, 88), (SIZE // 2, SIZE // 2), r, 1)
    for _ in range(30):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.line(surf, (90, 130, 150), (x, y), (x + rng.randint(2, 6), y), 1)
    return surf


def _noise_grid(rng: random.Random, cells_x: int, cells_y: int) -> list:
    """Tileable value noise over a ``cells_x`` x ``cells_y`` lattice, sampled on the SIZE grid:
    more cells one way than the other stretches its blobs the other way."""
    lattice = [[rng.random() for _ in range(cells_x)] for _ in range(cells_y)]
    out = []
    for y in range(SIZE):
        fy = y / SIZE * cells_y
        iy, ty = int(fy), fy - int(fy)
        sy = ty * ty * (3 - 2 * ty)
        row = []
        for x in range(SIZE):
            fx = x / SIZE * cells_x
            ix, tx = int(fx), fx - int(fx)
            sx = tx * tx * (3 - 2 * tx)
            a = lattice[iy % cells_y][ix % cells_x]
            b = lattice[iy % cells_y][(ix + 1) % cells_x]
            c = lattice[(iy + 1) % cells_y][ix % cells_x]
            d = lattice[(iy + 1) % cells_y][(ix + 1) % cells_x]
            row.append((a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy)
        out.append(row)
    return out


def _water(rng: random.Random, waves, deep, shallow, light=(0.6, -0.8), glint: float = 0.7,
           contrast: float = 2.2) -> pygame.Surface:
    """Moving water: a sum of sine waves (``waves``: (kx, ky, amplitude) - whole numbers of
    wavelengths across the tile, so it tiles) with random phases, shaded as if lit from ``light``:
    wave faces towards it bright, away from it dark, a glint where a crest catches the sky."""
    phases = [rng.uniform(0.0, 2.0 * math.pi) for _ in waves]
    tau = 2.0 * math.pi / SIZE
    lx, ly = light
    norm = sum(a * math.hypot(kx, ky) for kx, ky, a in waves) * tau
    surf = pygame.Surface((SIZE, SIZE))
    for y in range(SIZE):
        for x in range(SIZE):
            h = gx = gy = 0.0
            for (kx, ky, a), ph in zip(waves, phases):
                arg = tau * (kx * x + ky * y) + ph
                h += a * math.sin(arg)
                c = a * math.cos(arg) * tau
                gx += c * kx
                gy += c * ky
            shade = (gx * lx + gy * ly) / norm * contrast                  # -1 .. 1, roughly
            shade = max(-1.0, min(1.0, shade))
            t = max(0.0, min(1.0, 0.5 + 0.45 * shade))
            col = [deep[i] + (shallow[i] - deep[i]) * t for i in range(3)]
            if shade > glint:                                            # a glint on the crest
                g = (shade - glint) / (1.0 - glint)
                col = [c + (230 - c) * 0.8 * g for c in col]
            surf.set_at((x, y), tuple(_clamp(c + _noise(rng, 2)) for c in col))
    return surf


def gen_stream(rng: random.Random) -> pygame.Surface:
    """Running water: ripple fronts across the flow (the texture's y runs downstream), bent and
    broken by a few slanting waves, and long faint streaks along it."""
    waves = [(0, 3, 1.0), (1, 5, 0.8), (-1, 7, 0.6), (2, 9, 0.45), (-2, 11, 0.35), (3, 4, 0.3), (-3, 6, 0.3), (1, 14, 0.2),
             (5, 1, 0.18), (-4, 2, 0.16)]
    return _water(rng, waves, deep=(26, 60, 74), shallow=(82, 134, 144), light=(0.35, -0.94), glint=0.62)


def gen_lake(rng: random.Random) -> pygame.Surface:
    """Still water stirred by the wind: many small wavelets, every which way (wave numbers drawn
    from a ring, so no one direction or size dominates), a few brighter ruffles."""
    waves = []
    while len(waves) < 26:
        kx, ky = rng.randint(-13, 13), rng.randint(-13, 13)
        k = math.hypot(kx, ky)
        if 3.0 <= k <= 13.5 and (kx, ky) not in [(w[0], w[1]) for w in waves] and (-kx, -ky) not in [(w[0], w[1]) for w in waves]:
            waves.append((kx, ky, rng.uniform(0.6, 1.0) / k ** 0.8))
    return _water(rng, waves, deep=(20, 50, 72), shallow=(80, 128, 150), light=(0.5, -0.87), glint=0.55, contrast=4.5)


def gen_coals(rng: random.Random) -> pygame.Surface:
    """A bed of coals: black lumps with orange-white heat glowing between them."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (235, 110, 30), 20, rng)                                   # the glow underneath
    for _ in range(160):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        w, h = rng.randint(6, 14), rng.randint(5, 11)
        shade = rng.randint(18, 45)
        pygame.draw.ellipse(surf, (shade, shade - 4, shade - 6), (x - w // 2, y - h // 2, w, h))
        if rng.random() < 0.35:
            pygame.draw.ellipse(surf, (shade + 30, shade + 12, shade), (x - w // 2 + 2, y - h // 2 + 1, w - 4, h - 3), 1)
    for _ in range(25):                                                          # white-hot specks
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, (255, 220, 150), (x, y), rng.randint(1, 2))
    return surf


def _sign_board(rng: random.Random) -> pygame.Surface:
    """A hanging shop sign: weathered planks with an iron rim, ready for a painted device."""
    surf = gen_planks(rng, base=(120, 82, 46), gap=(70, 44, 22), planks=4)
    pygame.draw.rect(surf, (50, 48, 50), (0, 0, SIZE, SIZE), 6)
    for x, y in ((10, 10), (SIZE - 11, 10), (10, SIZE - 11), (SIZE - 11, SIZE - 11)):
        pygame.draw.circle(surf, (80, 78, 82), (x, y), 3)                       # nail heads
    return surf


def gen_sign_smith(rng: random.Random) -> pygame.Surface:
    """An anvil with a hammer raised over it."""
    surf = _sign_board(rng)
    ink = (30, 26, 28)
    pygame.draw.polygon(surf, ink, [(26, 70), (102, 70), (96, 82), (74, 82), (74, 96), (54, 96), (54, 82), (32, 82)])   # anvil
    pygame.draw.polygon(surf, ink, [(20, 64), (40, 60), (44, 70), (26, 72)])                                          # the horn
    pygame.draw.rect(surf, ink, (48, 98, 32, 8))                                                                       # the block
    pygame.draw.line(surf, (90, 60, 35), (86, 22), (64, 58), 5)                                                        # hammer haft
    pygame.draw.rect(surf, (70, 70, 78), (78, 14, 24, 14))                                                             # hammer head
    return surf


def gen_sign_store(rng: random.Random) -> pygame.Surface:
    """A merchant's balance: a post, a beam and two hanging pans."""
    surf = _sign_board(rng)
    gold = (214, 176, 60)
    pygame.draw.rect(surf, gold, (61, 34, 6, 66))                                                     # post
    pygame.draw.rect(surf, gold, (44, 96, 40, 7))                                                     # foot
    pygame.draw.line(surf, gold, (22, 40), (106, 40), 5)                                              # beam
    for cx in (30, 98):
        pygame.draw.line(surf, gold, (cx, 42), (cx - 12, 66), 2)
        pygame.draw.line(surf, gold, (cx, 42), (cx + 12, 66), 2)
        pygame.draw.ellipse(surf, gold, (cx - 16, 62, 32, 10))                                       # pan
    pygame.draw.circle(surf, gold, (64, 32), 5)
    return surf


def gen_sign_tailor(rng: random.Random) -> pygame.Surface:
    """A spool of thread with a needle through it."""
    surf = _sign_board(rng)
    pygame.draw.rect(surf, (150, 110, 70), (40, 36, 48, 60))                                         # spool body
    pygame.draw.rect(surf, (110, 78, 48), (36, 30, 56, 10))                                          # flanges
    pygame.draw.rect(surf, (110, 78, 48), (36, 92, 56, 10))
    for y in range(42, 92, 4):
        pygame.draw.line(surf, (200, 60, 70), (40, y), (88, y + 2), 2)                               # red thread
    pygame.draw.line(surf, (220, 220, 230), (22, 104), (106, 26), 4)                                 # needle
    pygame.draw.circle(surf, (60, 60, 70), (102, 30), 4, 2)                                          # the eye
    pygame.draw.line(surf, (200, 60, 70), (102, 30), (112, 20), 2)                                   # thread through it
    return surf


def gen_sign_inn(rng: random.Random) -> pygame.Surface:
    """A foaming tankard."""
    surf = _sign_board(rng)
    pygame.draw.rect(surf, (170, 120, 60), (38, 44, 44, 58))                                         # the tankard
    pygame.draw.rect(surf, (130, 90, 45), (38, 44, 44, 58), 3)
    pygame.draw.line(surf, (130, 90, 45), (38, 62), (82, 62), 2)                                     # bands
    pygame.draw.line(surf, (130, 90, 45), (38, 84), (82, 84), 2)
    pygame.draw.ellipse(surf, (130, 90, 45), (80, 52, 22, 34), 5)                                    # handle
    for cx, cy, r in ((46, 40, 9), (60, 34, 11), (76, 40, 9), (66, 44, 7)):
        pygame.draw.circle(surf, (245, 240, 225), (cx, cy), r)                                       # foam
    return surf


def gen_sign_bakery(rng: random.Random) -> pygame.Surface:
    """A loaf and a sheaf of wheat."""
    surf = _sign_board(rng)
    pygame.draw.ellipse(surf, (196, 138, 66), (24, 58, 80, 40))                     # the loaf
    pygame.draw.ellipse(surf, (150, 96, 40), (24, 58, 80, 40), 3)
    for x in (44, 60, 76):
        pygame.draw.line(surf, (120, 74, 30), (x, 64), (x + 8, 90), 3)               # its slashes
    for k in range(5):                                                              # the wheat
        x = 38 + k * 12
        pygame.draw.line(surf, (230, 200, 90), (x, 54), (64, 20), 2)
        pygame.draw.ellipse(surf, (236, 206, 96), (x - 4, 14 + abs(k - 2) * 3, 8, 16))
    return surf


def gen_moon(rng: random.Random) -> pygame.Surface:
    surf = gen_disc((235, 235, 225), (170, 175, 190))
    for _ in range(9):                                                            # craters
        x, y = rng.randrange(24, SIZE - 24), rng.randrange(24, SIZE - 24)
        pygame.draw.circle(surf, (150, 155, 170, 255), (x, y), rng.randint(3, 8))
    return surf


# ----------------------------------------------------------------------------- the wider world
def gen_bark(rng: random.Random) -> pygame.Surface:
    """Rough brown bark: vertical furrows and ridges (tiles vertically up a trunk)."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (86, 64, 44), 10, rng)
    for _ in range(22):
        x = rng.randrange(SIZE)
        w = rng.randint(1, 3)
        shade = rng.choice([(52, 38, 26), (62, 46, 32), (108, 84, 60)])
        y = rng.randrange(SIZE)
        for k in range(SIZE):
            xx = (x + int(2 * math.sin((y + k) / 9.0))) % SIZE
            for dx in range(w):
                surf.set_at(((xx + dx) % SIZE, (y + k) % SIZE), shade)
    return surf


def _blotches(rng, base, shades, count, rmin, rmax, alpha=False):
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA if alpha else 0)
    _fill_noise(surf, base, 10, rng)
    for _ in range(count):
        x, y, r = rng.randrange(SIZE), rng.randrange(SIZE), rng.randint(rmin, rmax)
        colour = rng.choice(shades)
        for ox in (-SIZE, 0, SIZE):
            for oy in (-SIZE, 0, SIZE):
                pygame.draw.circle(surf, colour, (x + ox, y + oy), r)
    return surf


def gen_leaves(rng: random.Random) -> pygame.Surface:
    """A canopy of broad leaves: greens in overlapping clumps, a few dark gaps."""
    return _blotches(rng, (52, 98, 40), [(40, 82, 32), (70, 122, 52), (88, 140, 60), (30, 62, 26)], 190, 3, 7)


def gen_pine(rng: random.Random) -> pygame.Surface:
    """Pine needles: dark blue-green in fine diagonal strokes."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (34, 66, 44), 8, rng)
    for _ in range(900):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        colour = rng.choice([(26, 52, 36), (48, 86, 58), (60, 100, 66)])
        d = rng.choice((-1, 1))
        for k in range(4):
            surf.set_at(((x + k * d) % SIZE, (y + k) % SIZE), colour)
    return surf


def gen_barrel(rng: random.Random) -> pygame.Surface:
    """Barrel staves running up the image, two dark iron hoops across them."""
    surf = gen_planks(rng, base=(132, 90, 52), gap=(58, 36, 18), planks=8, vertical=True)
    for y in (SIZE // 6, SIZE * 5 // 6):
        pygame.draw.rect(surf, (56, 56, 60), (0, y - 6, SIZE, 12))
        pygame.draw.line(surf, (96, 96, 102), (0, y - 5), (SIZE, y - 5), 1)
    return surf


def gen_log_end(rng: random.Random) -> pygame.Surface:
    """The sawn end of a log: pale wood with growth rings, bark round the rim."""
    surf = pygame.Surface((SIZE, SIZE))
    surf.fill((70, 50, 34))
    c = SIZE // 2
    for r in range(c - 4, 0, -5):
        shade = 176 - (r % 10) * 4 + _noise(rng, 6)
        pygame.draw.circle(surf, (_clamp(shade), _clamp(shade * 0.8), _clamp(shade * 0.55)), (c, c), r)
        pygame.draw.circle(surf, (130, 96, 62), (c, c), r, 1)
    for _ in range(3):                                                            # a crack or two
        a = rng.uniform(0, 2 * math.pi)
        pygame.draw.line(surf, (84, 60, 38), (c, c), (c + int(math.cos(a) * (c - 6)), c + int(math.sin(a) * (c - 6))), 1)
    return surf


def gen_crate(rng: random.Random) -> pygame.Surface:
    surf = gen_planks(rng, base=(150, 112, 70), gap=(80, 58, 34), planks=4)
    frame = (104, 76, 44)
    pygame.draw.rect(surf, frame, (0, 0, SIZE, SIZE), 10)
    pygame.draw.line(surf, frame, (8, 8), (SIZE - 8, SIZE - 8), 10)
    return surf


def gen_hay(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (188, 160, 88), 12, rng)
    for _ in range(700):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        length = rng.randint(4, 10)
        dx = rng.choice((-1, 0, 1))
        colour = rng.choice([(214, 186, 110), (160, 130, 66), (232, 208, 132)])
        pygame.draw.line(surf, colour, (x, y), (x + dx * length, y + length // 3), 1)
    return surf


def gen_thatch(rng: random.Random) -> pygame.Surface:
    """A thatched roof: straw laid in overlapping courses down the slope."""
    surf = gen_hay(rng)
    for y in range(0, SIZE, 16):
        pygame.draw.line(surf, (118, 94, 50), (0, y), (SIZE, y), 2)
    return surf


def _cracks(surf, rng, count, colour, width=1, alpha=None):
    """Branching cracks. Each segment is drawn where it lies and shifted by a tile either way, so
    a crack running over an edge carries on from the opposite one (the texture stays tileable)."""
    ink = colour if alpha is None else (*colour, alpha)

    def seg(a, b, w):
        for ox in (-SIZE, 0, SIZE):
            for oy in (-SIZE, 0, SIZE):
                pygame.draw.line(surf, ink, (a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy), w)

    for _ in range(count):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        a = rng.uniform(0, 2 * math.pi)
        for _step in range(rng.randint(6, 14)):
            a += rng.uniform(-0.7, 0.7)
            nx, ny = x + math.cos(a) * rng.randint(3, 7), y + math.sin(a) * rng.randint(3, 7)
            seg((x, y), (nx, ny), width)
            x, y = nx, ny
            if rng.random() < 0.2:                                                # a branch
                seg((x, y), (x + math.cos(a + 1.2) * 6, y + math.sin(a + 1.2) * 6), 1)


def gen_cobble_cracked(rng: random.Random, alpha_edge: bool = False) -> pygame.Surface:
    """Old paving: the cobbles cracked, some missing (bare earth), weeds in the gaps."""
    surf = gen_cobble(rng, alpha_edge=alpha_edge)
    for _ in range(5):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.ellipse(surf, (74, 62, 46, 255), (x - 7, y - 5, 14, 10))                # a lifted stone
    _cracks(surf, rng, 6, (34, 30, 26), alpha=255 if alpha_edge else None)
    for _ in range(30):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, (68, 104, 44, 255) if alpha_edge else (68, 104, 44), (x, y), 1)
    if alpha_edge:                                                                          # keep the fade
        for y in range(SIZE):
            t = y / (SIZE - 1)
            for x in range(SIZE):
                c = surf.get_at((x, y))
                surf.set_at((x, y), (c.r, c.g, c.b, min(c.a, _clamp(255 * t * t))))
    return surf


def gen_wall_cracked(rng: random.Random) -> pygame.Surface:
    """Weathered curtain-wall stone: cracked blocks, moss along the joints, rain-streaks."""
    surf = gen_stone_blocks(rng)
    _cracks(surf, rng, 7, (44, 40, 36), width=2)
    for _ in range(60):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, rng.choice([(70, 96, 50), (84, 108, 60)]), (x, y), rng.randint(1, 2))
    for _ in range(6):                                                                      # water streaks
        x = rng.randrange(SIZE)
        for y in range(rng.randrange(SIZE // 2), SIZE):
            c = surf.get_at((x, y))
            surf.set_at((x, y), (_clamp(c.r * 0.82), _clamp(c.g * 0.85), _clamp(c.b * 0.85)))
    return surf


def gen_planks_rot(rng: random.Random) -> pygame.Surface:
    """Old boards: grey weathering, dark rot spreading from the bottom, a green bloom of mould."""
    surf = gen_planks(rng, base=(120, 92, 66), gap=(50, 36, 22))
    rot = _value_noise(rng, SIZE, SIZE, 5)
    for y in range(SIZE):
        for x in range(SIZE):
            c = surf.get_at((x, y))
            k = rot[y][x] * (y / SIZE) ** 1.5                                              # worse near the ground
            grey = (c.r + c.g + c.b) / 3
            r, g, b = c.r * 0.6 + grey * 0.4, c.g * 0.6 + grey * 0.4, c.b * 0.6 + grey * 0.4
            surf.set_at((x, y), (_clamp(r * (1 - 0.55 * k)), _clamp(g * (1 - 0.45 * k) + 10 * k), _clamp(b * (1 - 0.5 * k))))
    return surf


def gen_crack_decal(rng: random.Random) -> pygame.Surface:
    """Transparent but for dark cracks: laid over paving."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    _cracks(surf, rng, 4, (28, 24, 20), width=2, alpha=230)
    return surf


def gen_stain_decal(rng: random.Random) -> pygame.Surface:
    """A water stain running down a wall: darker at the top where it seeps, dripping streaks."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    for _ in range(14):
        x = rng.randrange(16, SIZE - 16)
        top = rng.randrange(0, 20)
        length = rng.randrange(SIZE // 2, SIZE)
        w = rng.randint(2, 6)
        for y in range(top, min(SIZE, top + length)):
            a = int(150 * (1 - (y - top) / length))
            for dx in range(-w, w + 1):
                if 0 <= x + dx < SIZE:
                    old = surf.get_at((x + dx, y)).a
                    surf.set_at((x + dx, y), (30, 34, 28, max(old, a - abs(dx) * 12)))
    return surf


def gen_dirt(rng: random.Random, alpha_edge: bool = False) -> pygame.Surface:
    """A trodden earth path: brown with pebbles and ruts."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    patches = _value_noise(rng, SIZE, SIZE, 5)
    for y in range(SIZE):
        for x in range(SIZE):
            p = patches[y][x]
            base = 96 + p * 30 + _noise(rng, 10)
            a = 255 if not alpha_edge else _clamp(255 * (y / (SIZE - 1)) ** 2)
            surf.set_at((x, y), (_clamp(base), _clamp(base * 0.8), _clamp(base * 0.56), a))
    for _ in range(70):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        a = 255 if not alpha_edge else _clamp(255 * (y / (SIZE - 1)) ** 2)
        pygame.draw.circle(surf, (*rng.choice([(130, 120, 104), (80, 66, 50)]), a), (x, y), rng.randint(1, 2))
    return surf


def gen_rock(rng: random.Random) -> pygame.Surface:
    """Cave and cliff rock: grey strata, fractures and damp dark patches."""
    surf = pygame.Surface((SIZE, SIZE))
    strata = _value_noise(rng, SIZE, SIZE, 4)
    for y in range(SIZE):
        for x in range(SIZE):
            band = 0.5 + 0.5 * math.sin(y / 6.0 + strata[y][x] * 4)
            v = 82 + band * 34 + _noise(rng, 10)
            surf.set_at((x, y), (_clamp(v), _clamp(v * 0.97), _clamp(v * 0.92)))
    _cracks(surf, rng, 6, (48, 46, 44), width=2)
    return surf


def gen_sand(rng: random.Random) -> pygame.Surface:
    """Beach sand: pale and fine, rippled by the wind, a few shells and dark grains."""
    surf = pygame.Surface((SIZE, SIZE))
    patches = _value_noise(rng, SIZE, SIZE, 4)
    for y in range(SIZE):
        for x in range(SIZE):
            ripple = 0.5 + 0.5 * math.sin((x * 0.35 + y * 0.9) * 2 * math.pi / 16 + patches[y][x] * 3.0)
            v = 196 + patches[y][x] * 26 + ripple * 10 + _noise(rng, 8)
            surf.set_at((x, y), (_clamp(v), _clamp(v * 0.9), _clamp(v * 0.7)))
    for _ in range(90):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        surf.set_at((x, y), rng.choice([(120, 104, 84), (236, 228, 210), (150, 132, 100)]))
    return surf


def gen_snow(rng: random.Random) -> pygame.Surface:
    """Old snow on the peaks: white going blue in the hollows, a glitter of ice."""
    surf = pygame.Surface((SIZE, SIZE))
    patches = _value_noise(rng, SIZE, SIZE, 5)
    for y in range(SIZE):
        for x in range(SIZE):
            v = 214 + patches[y][x] * 34 + _noise(rng, 5)
            surf.set_at((x, y), (_clamp(v - 10), _clamp(v - 4), _clamp(v + 6)))
    for _ in range(40):
        surf.set_at((rng.randrange(SIZE), rng.randrange(SIZE)), (255, 255, 255))
    return surf


def gen_cliff(rng: random.Random) -> pygame.Surface:
    """Bare mountainside: grey-brown rock in broken ledges - strata of uneven thickness whose
    edges wander, split into blocks by leaning joints, each ledge lit along its lip and shadowed
    under its overhang - with lichen and moss on the ledges. (The terrain shows it side-on on a
    crag: v is up the face.)"""
    surf = pygame.Surface((SIZE, SIZE))
    lumps = _value_noise(rng, SIZE, SIZE, 8)
    grain = _value_noise(rng, SIZE, SIZE, 16)
    moss = _value_noise(rng, SIZE, SIZE, 6)
    warp = _value_noise(rng, SIZE, SIZE, 5)
    # the strata: bands of 8 - 24 px whose thicknesses add up to the tile (so it tiles up the face)
    bands, y = [], 0
    while y < SIZE:
        h = rng.randint(8, 24)
        if SIZE - (y + h) < 8:
            h = SIZE - y
        joints = sorted(rng.randrange(SIZE) for _ in range(rng.randint(1, 3)))
        bands.append((y, h, joints, [rng.uniform(-16.0, 16.0) for _ in joints], rng.uniform(-0.6, 0.6)))
        y += h
    band_of = {}
    for b in bands:
        for dy in range(b[1]):
            band_of[b[0] + dy] = b
    for y in range(SIZE):
        for x in range(SIZE):
            yw = int(y + (warp[0][x] - 0.5) * 10.0 + (warp[y][x] - 0.5) * 4.0) % SIZE     # wandering ledges
            y0, h, joints, tones, lean = band_of[yw]
            dy = (yw - y0) % SIZE
            xs = (x - lean * dy) % SIZE                                   # leaning joints
            block = sum(1 for j in joints if j <= xs) - 1
            lip = 14.0 * max(0.0, 1.0 - dy / 2.5)                         # the lit top of the ledge
            under = -38.0 * max(0.0, 1.0 - (h - 1 - dy) / 2.5)             # the shadow under the one above
            v = 92.0 + tones[block] + lumps[y][x] * 40.0 + grain[y][x] * 16.0 + lip + under + _noise(rng, 6)
            if any(min(abs(xs - j), SIZE - abs(xs - j)) < 0.8 for j in joints) and dy > 1:
                v *= 0.6                                                   # a joint between blocks
            col = [v * 1.0, v * 0.95, v * 0.87]
            m = moss[y][x]
            if m > 0.6 and dy < 5:                                         # lichen and moss on the ledges
                k = min(1.0, (m - 0.6) / 0.25) * (1.0 - dy / 5.0)
                col = [col[0] * (1 - 0.4 * k), col[1] * (1 - 0.05 * k), col[2] * (1 - 0.5 * k)]
            surf.set_at((x, y), tuple(_clamp(c) for c in col))
    _cracks(surf, rng, 3, (62, 59, 54), width=1)
    return surf


def gen_cave_rock(rng: random.Random) -> pygame.Surface:
    """The walls of a cave: dark, damp rock in lumps and seams, glistening here and there."""
    surf = pygame.Surface((SIZE, SIZE))
    lumps = _value_noise(rng, SIZE, SIZE, 6)
    seams = _value_noise(rng, SIZE, SIZE, 3)
    for y in range(SIZE):
        for x in range(SIZE):
            v = 54 + lumps[y][x] * 46 + _noise(rng, 8)
            if abs(seams[y][x] - 0.5) < 0.018:
                v *= 0.8
            surf.set_at((x, y), (_clamp(v * 0.96), _clamp(v * 0.93), _clamp(v * 0.9)))
    for _ in range(30):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.line(surf, (128, 132, 136), (x, y), (x, y + rng.randint(1, 3)), 1)
    _cracks(surf, rng, 3, (36, 34, 32), width=1)
    return surf


def gen_cave_floor(rng: random.Random) -> pygame.Surface:
    """A cave's floor: packed earth over rock, grit and flat stones."""
    surf = pygame.Surface((SIZE, SIZE))
    patches = _value_noise(rng, SIZE, SIZE, 5)
    for y in range(SIZE):
        for x in range(SIZE):
            v = 62 + patches[y][x] * 34 + _noise(rng, 9)
            surf.set_at((x, y), (_clamp(v * 1.02), _clamp(v * 0.92), _clamp(v * 0.8)))
    for _ in range(26):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        r = rng.randint(2, 6)
        c = rng.randint(70, 100)
        pygame.draw.ellipse(surf, (c, c - 4, c - 10), (x - r, y - r // 2, 2 * r, r))
    return surf


def gen_mushroom(rng: random.Random) -> pygame.Surface:
    """A cave mushroom's cap: pale blue-green flesh, darker gills at the rim and bright freckles."""
    surf = pygame.Surface((SIZE, SIZE))
    patches = _value_noise(rng, SIZE, SIZE, 4)
    for y in range(SIZE):
        for x in range(SIZE):
            v = patches[y][x]
            surf.set_at((x, y), (_clamp(150 + v * 50 + _noise(rng, 6)), _clamp(200 + v * 40 + _noise(rng, 6)),
                                 _clamp(175 + v * 45 + _noise(rng, 6))))
    for _ in range(40):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, (235, 255, 240), (x, y), rng.randint(1, 3))
    return surf


def gen_bone(rng: random.Random) -> pygame.Surface:
    """Old bone: yellowed ivory, stained brown in the grain."""
    surf = pygame.Surface((SIZE, SIZE))
    streaks = _value_noise(rng, SIZE, SIZE, 6)
    for y in range(SIZE):
        for x in range(SIZE):
            v = streaks[y][x]
            surf.set_at((x, y), (_clamp(214 - v * 40 + _noise(rng, 5)), _clamp(200 - v * 44 + _noise(rng, 5)),
                                 _clamp(160 - v * 50 + _noise(rng, 5))))
    return surf


def gen_parchment(rng: random.Random) -> pygame.Surface:
    """A page: cream parchment, darker at the edges, lines of faded brown writing."""
    surf = pygame.Surface((SIZE, SIZE))
    blot = _value_noise(rng, SIZE, SIZE, 4)
    for y in range(SIZE):
        for x in range(SIZE):
            edge = min(x, y, SIZE - 1 - x, SIZE - 1 - y) / 20.0
            v = 222 + blot[y][x] * 18 - max(0.0, 1.0 - edge) * 40
            surf.set_at((x, y), (_clamp(v), _clamp(v * 0.92), _clamp(v * 0.74)))
    for row in range(14, SIZE - 12, 9):
        x = 12
        while x < SIZE - 14:
            w = rng.randint(4, 14)
            pygame.draw.line(surf, (96, 70, 46), (x, row), (min(SIZE - 12, x + w), row), 1)
            x += w + rng.randint(3, 5)
    return surf


def gen_notice(rng: random.Random) -> pygame.Surface:
    """A notice board: weathered planks with papers pinned all over it."""
    surf = gen_planks(rng, base=(112, 80, 50), gap=(64, 44, 26), planks=5, vertical=True)
    page = gen_parchment(rng)
    for _ in range(5):
        w, h = rng.randint(34, 52), rng.randint(40, 60)
        x, y = rng.randint(4, SIZE - w - 4), rng.randint(4, SIZE - h - 4)
        surf.blit(pygame.transform.smoothscale(page, (w, h)), (x, y))
        pygame.draw.circle(surf, (150, 40, 30), (x + w // 2, y + 4), 2)        # the pin
    pygame.draw.rect(surf, (54, 38, 24), (0, 0, SIZE, SIZE), 5)
    return surf


def gen_sign_arm(rng: random.Random) -> pygame.Surface:
    """A fingerpost's arm: a painted board with a line of lettering (too small to read from the road)."""
    surf = gen_planks(rng, base=(190, 172, 140), gap=(150, 130, 100), planks=1)
    pygame.draw.rect(surf, (60, 44, 30), (0, 0, SIZE, SIZE), 8)
    for i in range(7):
        x = 18 + i * 14
        h = rng.randint(26, 44)
        pygame.draw.rect(surf, (40, 32, 28), (x, 64 - h // 2, 8, h))
    return surf


def gen_plaque(rng: random.Random) -> pygame.Surface:
    """A lettered board: dark stain, a gilded border and rows of cut letters."""
    surf = gen_planks(rng, base=(70, 46, 30), gap=(50, 32, 20), planks=2)
    pygame.draw.rect(surf, (190, 150, 60), (4, 4, SIZE - 8, SIZE - 8), 4)
    for row in range(26, SIZE - 20, 20):
        x = 16
        while x < SIZE - 20:
            w = rng.randint(6, 10)
            pygame.draw.rect(surf, (220, 190, 110), (x, row, w, 10))
            x += w + 4
    return surf


def _banner(rng: random.Random, field, trim, device) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    weave = _value_noise(rng, SIZE, SIZE, 8)
    for y in range(SIZE):
        for x in range(SIZE):
            k = 0.9 + 0.12 * weave[y][x]
            surf.set_at((x, y), tuple(_clamp(c * k) for c in field))
    pygame.draw.rect(surf, trim, (0, 0, SIZE, SIZE), 8)
    device(surf)
    for x in range(0, SIZE, 16):                                  # the fringe along the foot
        pygame.draw.polygon(surf, trim, [(x, SIZE - 1), (x + 16, SIZE - 1), (x + 8, SIZE - 11)])
    return surf


def gen_banner_greyhold(rng: random.Random) -> pygame.Surface:
    """Greyhold's colours: a white tower on grey, under a silver band."""
    def tower(surf):
        c = (235, 235, 240)
        pygame.draw.rect(surf, c, (48, 50, 32, 56))
        for x in range(44, 84, 10):
            pygame.draw.rect(surf, c, (x, 40, 7, 10))
        pygame.draw.rect(surf, (90, 92, 100), (58, 60, 12, 18))
        pygame.draw.rect(surf, (200, 200, 210), (8, 14, SIZE - 16, 6))
    return _banner(rng, (100, 104, 114), (196, 198, 206), tower)


def gen_banner_deep(rng: random.Random) -> pygame.Surface:
    """The Howling Deep's rag: a white wolf's head howling on blood red."""
    def wolf(surf):
        c = (230, 226, 214)
        pygame.draw.polygon(surf, c, [(40, 30), (64, 50), (88, 30), (84, 70), (70, 92), (58, 92), (44, 70)])
        pygame.draw.polygon(surf, c, [(40, 30), (36, 56), (48, 52)])
        pygame.draw.polygon(surf, c, [(88, 30), (92, 56), (80, 52)])
        pygame.draw.circle(surf, (20, 10, 10), (54, 62), 4)
        pygame.draw.circle(surf, (20, 10, 10), (74, 62), 4)
        pygame.draw.polygon(surf, (60, 10, 10), [(60, 80), (68, 80), (64, 100)])
    return _banner(rng, (120, 22, 20), (40, 30, 26), wolf)


def gen_ashlar(rng: random.Random) -> pygame.Surface:
    """Dressed stone for a fine house: long honey-coloured blocks laid in even courses, thin mortar."""
    surf = pygame.Surface((SIZE, SIZE))
    shade = _value_noise(rng, SIZE, SIZE, 6)
    for y in range(SIZE):
        for x in range(SIZE):
            v = shade[y][x]
            surf.set_at((x, y), (_clamp(196 + v * 30 + _noise(rng, 6)), _clamp(170 + v * 26 + _noise(rng, 6)),
                                 _clamp(126 + v * 22 + _noise(rng, 6))))
    rows, per = 4, 2
    bh = SIZE // rows
    for r in range(rows):
        y = r * bh
        tone = rng.randint(-12, 12)
        off = (SIZE // per // 2) if r % 2 else 0
        for bx in range(-SIZE, SIZE * 2, SIZE // per):
            x0 = bx + off
            for yy in range(y + 1, y + bh - 1):
                for xx in range(max(0, x0 + 1), min(SIZE, x0 + SIZE // per - 1)):
                    c = surf.get_at((xx, yy))
                    surf.set_at((xx, yy), (_clamp(c.r + tone), _clamp(c.g + tone), _clamp(c.b + tone * 0.8)))
            if 0 <= x0 < SIZE:
                pygame.draw.line(surf, (150, 126, 92), (x0, y), (x0, y + bh), 2)
        pygame.draw.line(surf, (150, 126, 92), (0, y), (SIZE, y), 2)
        pygame.draw.line(surf, (226, 204, 160), (0, y + 2), (SIZE, y + 2), 1)
    return surf


def gen_manor_plaster(rng: random.Random) -> pygame.Surface:
    """Smooth limewash render over a fine house's upper storey, warm cream, faint lines of rustication."""
    surf = pygame.Surface((SIZE, SIZE))
    shade = _value_noise(rng, SIZE, SIZE, 5)
    for y in range(SIZE):
        for x in range(SIZE):
            v = shade[y][x]
            surf.set_at((x, y), (_clamp(232 + v * 16 + _noise(rng, 3)), _clamp(220 + v * 14 + _noise(rng, 3)),
                                 _clamp(190 + v * 12 + _noise(rng, 3))))
    for y in (SIZE // 2 - 1, SIZE - 2):
        pygame.draw.line(surf, (206, 192, 160), (0, y), (SIZE, y), 2)
    return surf


def _glass(surf, rect, rng, dark=(34, 48, 62), light=(120, 150, 170)):
    x, y, w, h = rect
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            t = ((xx - x) / max(w, 1) + (yy - y) / max(h, 1)) * 0.5
            glint = 1.0 if 0.32 < t < 0.4 or 0.55 < t < 0.58 else 0.0
            c = [dark[i] + (light[i] - dark[i]) * (0.25 * t + 0.6 * glint) for i in range(3)]
            surf.set_at((xx, yy), tuple(_clamp(v + _noise(rng, 3)) for v in c))


def gen_window_manor(rng: random.Random) -> pygame.Surface:
    """A tall sash window for a fine house: a dressed stone surround and keystone, glazing bars in a
    dark frame, green shutters folded back either side and a sill underneath. RGBA: the wall shows
    round it."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    stone, frame, shutter = (222, 206, 172), (60, 44, 34), (52, 92, 64)
    pygame.draw.rect(surf, stone, (30, 4, 68, 118))                       # the surround
    pygame.draw.polygon(surf, (236, 222, 190), [(56, 4), (72, 4), (68, 14), (60, 14)])   # keystone
    pygame.draw.rect(surf, stone, (24, 118, 80, 8))                       # the sill
    pygame.draw.rect(surf, frame, (38, 14, 52, 102))
    for row in range(3):
        for col in range(2):
            _glass(surf, (42 + col * 24, 18 + row * 32, 20, 28), rng)
    for sx in (4, 100):                                                   # the shutters
        pygame.draw.rect(surf, shutter, (sx, 12, 24, 104))
        for yy in range(16, 112, 6):
            pygame.draw.line(surf, (36, 66, 46), (sx + 2, yy), (sx + 21, yy + 2), 2)
        pygame.draw.rect(surf, (30, 50, 36), (sx, 12, 24, 104), 2)
    return surf


def gen_window_cottage(rng: random.Random) -> pygame.Surface:
    """A cottage window: a wooden frame of four small panes, a plank sill and a box of flowers."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    frame = (92, 62, 38)
    pygame.draw.rect(surf, frame, (22, 10, 84, 84))
    for row in range(2):
        for col in range(2):
            _glass(surf, (28 + col * 38, 16 + row * 38, 34, 34), rng, dark=(40, 46, 52), light=(150, 160, 150))
    pygame.draw.rect(surf, (110, 76, 44), (14, 94, 100, 10))
    pygame.draw.rect(surf, (96, 60, 36), (20, 104, 88, 16))
    for _ in range(26):
        x, y = rng.randint(22, 104), rng.randint(90, 104)
        pygame.draw.circle(surf, rng.choice([(200, 40, 50), (240, 200, 60), (230, 230, 240), (180, 70, 170)]), (x, y), 3)
        pygame.draw.circle(surf, (50, 110, 40), (x + 2, y + 4), 2)
    return surf


def gen_window_lancet(rng: random.Random) -> pygame.Surface:
    """A chapel's lancet: a tall pointed window of stained glass in lead cames, a stone frame."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    stone = (196, 190, 178)
    pts = [(40, 126), (40, 40), (64, 4), (88, 40), (88, 126)]
    pygame.draw.polygon(surf, stone, pts)
    inner = [(46, 122), (46, 42), (64, 14), (82, 42), (82, 122)]
    pygame.draw.polygon(surf, (20, 20, 30), inner)
    colours = [(170, 30, 40), (40, 70, 170), (220, 170, 40), (40, 130, 70), (120, 40, 130)]
    for y in range(16, 122, 9):
        for x in range(46, 82, 9):
            if surf.get_at((x + 4, y + 4))[:3] == (20, 20, 30):
                pygame.draw.rect(surf, rng.choice(colours), (x + 1, y + 1, 7, 7))
    pygame.draw.line(surf, stone, (64, 14), (64, 122), 3)
    pygame.draw.circle(surf, (230, 200, 90), (64, 46), 8)
    return surf


def _framed(rng, inner_draw) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    surf.fill((150, 110, 40))
    for i in range(10):
        c = 150 + 8 * math.sin(i * 1.3) + i * 5
        pygame.draw.rect(surf, (_clamp(c + 40), _clamp(c), _clamp(c * 0.4)), (i, i, SIZE - 2 * i, SIZE - 2 * i), 1)
    inner = pygame.Surface((SIZE - 24, SIZE - 24))
    inner_draw(inner)
    surf.blit(inner, (12, 12))
    return surf


def gen_painting_portrait(rng: random.Random) -> pygame.Surface:
    """An ancestor in oils: a stern face over a dark coat, a gilt frame."""
    def paint(s):
        s.fill((42, 34, 30))
        w, h = s.get_size()
        pygame.draw.ellipse(s, (70, 30, 40), (w // 2 - 34, h // 2, 68, 70))          # shoulders, a red coat
        pygame.draw.ellipse(s, (214, 176, 140), (w // 2 - 16, h // 2 - 30, 32, 40))   # the face
        pygame.draw.ellipse(s, (90, 70, 50), (w // 2 - 18, h // 2 - 36, 36, 18))      # the hair
        pygame.draw.line(s, (60, 40, 30), (w // 2 - 8, h // 2 - 14), (w // 2 - 3, h // 2 - 14), 2)
        pygame.draw.line(s, (60, 40, 30), (w // 2 + 3, h // 2 - 14), (w // 2 + 8, h // 2 - 14), 2)
        pygame.draw.line(s, (240, 230, 210), (w // 2 - 6, h // 2 + 10), (w // 2 + 6, h // 2 + 10), 3)   # a ruff
    return _framed(rng, paint)


def gen_painting_landscape(rng: random.Random) -> pygame.Surface:
    """A landscape in oils: hills, a sky going gold, a keep on a rise."""
    def paint(s):
        w, h = s.get_size()
        for y in range(h):
            t = y / h
            pygame.draw.line(s, (_clamp(120 + 100 * t), _clamp(150 + 50 * t), _clamp(200 - 80 * t)), (0, y), (w, y))
        pygame.draw.polygon(s, (60, 90, 50), [(0, h), (0, h * 0.6), (w * 0.3, h * 0.45), (w * 0.6, h * 0.62), (w, h * 0.5), (w, h)])
        pygame.draw.rect(s, (110, 110, 120), (w * 0.62, h * 0.36, 12, 26))
        pygame.draw.polygon(s, (40, 70, 40), [(0, h), (w * 0.2, h * 0.75), (w * 0.5, h * 0.85), (w, h * 0.72), (w, h)])
    return _framed(rng, paint)


def _rug(rng, field, border, motif) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, field, 6, rng)
    pygame.draw.rect(surf, border, (0, 0, SIZE, SIZE), 14)
    pygame.draw.rect(surf, motif, (18, 18, SIZE - 36, SIZE - 36), 3)
    c = SIZE // 2
    pygame.draw.polygon(surf, motif, [(c, 30), (SIZE - 30, c), (c, SIZE - 30), (30, c)], 4)
    pygame.draw.circle(surf, border, (c, c), 12)
    for k in range(4, SIZE, 8):                                  # the fringe's knots
        pygame.draw.line(surf, (230, 220, 190), (k, 0), (k, 4), 2)
        pygame.draw.line(surf, (230, 220, 190), (k, SIZE - 5), (k, SIZE), 2)
    return surf


def gen_rug(rng: random.Random) -> pygame.Surface:
    return _rug(rng, (140, 34, 36), (40, 30, 60), (220, 180, 90))


def gen_rug_blue(rng: random.Random) -> pygame.Surface:
    return _rug(rng, (40, 60, 120), (140, 40, 40), (230, 210, 150))


def gen_tile_floor(rng: random.Random) -> pygame.Surface:
    """A kitchen's floor: red and black quarry tiles in a chequer, worn."""
    surf = pygame.Surface((SIZE, SIZE))
    n = 4
    t = SIZE // n
    for i in range(n):
        for j in range(n):
            base = (150, 60, 40) if (i + j) % 2 else (46, 40, 38)
            pygame.draw.rect(surf, base, (i * t, j * t, t, t))
    for y in range(SIZE):
        for x in range(SIZE):
            c = surf.get_at((x, y))
            k = 1.0 + _noise(rng, 8) / 100.0
            surf.set_at((x, y), (_clamp(c.r * k), _clamp(c.g * k), _clamp(c.b * k)))
    for i in range(n + 1):
        pygame.draw.line(surf, (90, 80, 70), (i * t, 0), (i * t, SIZE), 2)
        pygame.draw.line(surf, (90, 80, 70), (0, i * t), (SIZE, i * t), 2)
    return surf


def gen_brick(rng: random.Random) -> pygame.Surface:
    surf = gen_stone_blocks(rng, base=(150, 72, 52), mortar=(196, 184, 168), rows=8)
    return surf


def gen_books(rng: random.Random) -> pygame.Surface:
    """A bookcase's face: shelves of leather spines in many colours, gilt lines on some."""
    surf = pygame.Surface((SIZE, SIZE))
    surf.fill((70, 46, 28))
    shelves = 4
    sh = SIZE // shelves
    for s_ in range(shelves):
        y0 = s_ * sh + 4
        x = 4
        while x < SIZE - 6:
            w = rng.randint(5, 10)
            h = rng.randint(sh - 14, sh - 6)
            col = rng.choice([(110, 30, 30), (40, 60, 100), (40, 80, 50), (120, 90, 40), (80, 40, 80), (60, 40, 30)])
            pygame.draw.rect(surf, col, (x, y0 + (sh - 6 - h), w, h))
            if rng.random() < 0.5:
                pygame.draw.line(surf, (220, 190, 90), (x, y0 + (sh - 6 - h) + 4), (x + w - 1, y0 + (sh - 6 - h) + 4), 1)
            x += w + 1
        pygame.draw.rect(surf, (96, 64, 38), (0, s_ * sh + sh - 4, SIZE, 5))
    pygame.draw.rect(surf, (96, 64, 38), (0, 0, SIZE, SIZE), 4)
    return surf


def gen_clock_face(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    surf.fill((80, 50, 30))
    pygame.draw.circle(surf, (230, 220, 190), (64, 64), 54)
    pygame.draw.circle(surf, (180, 140, 60), (64, 64), 54, 4)
    for k in range(12):
        a = k * math.pi / 6
        pygame.draw.line(surf, (40, 30, 20), (64 + math.cos(a) * 44, 64 + math.sin(a) * 44),
                         (64 + math.cos(a) * 50, 64 + math.sin(a) * 50), 3)
    pygame.draw.line(surf, (30, 20, 10), (64, 64), (64, 30), 4)
    pygame.draw.line(surf, (30, 20, 10), (64, 64), (88, 70), 3)
    return surf


def gen_wallpaper(rng: random.Random) -> pygame.Surface:
    """Damask on a rich green ground, for the inside of a fine house."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (46, 78, 60), 4, rng)
    for cx in (32, 96):
        for cy in (32, 96):
            ox, oy = (cx, cy) if (cx + cy) % 128 == 0 else (cx, cy)
            pygame.draw.ellipse(surf, (92, 124, 90), (ox - 14, oy - 22, 28, 44), 2)
            pygame.draw.line(surf, (92, 124, 90), (ox, oy - 26), (ox, oy + 26), 2)
            pygame.draw.circle(surf, (150, 140, 80), (ox, oy), 4)
    return surf


def gen_flowers(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (54, 96, 44), 14, rng)
    for _ in range(140):
        x, y = rng.randrange(SIZE), rng.randrange(SIZE)
        pygame.draw.circle(surf, rng.choice([(210, 40, 60), (240, 210, 70), (240, 240, 250), (170, 80, 190), (240, 140, 40)]),
                           (x, y), rng.randint(2, 4))
    return surf


def gen_gilt(rng: random.Random) -> pygame.Surface:
    surf = pygame.Surface((SIZE, SIZE))
    shade = _value_noise(rng, SIZE, SIZE, 6)
    for y in range(SIZE):
        for x in range(SIZE):
            v = shade[y][x]
            surf.set_at((x, y), (_clamp(200 + v * 50), _clamp(160 + v * 40), _clamp(60 + v * 30)))
    return surf


def gen_grate(rng: random.Random) -> pygame.Surface:
    """An iron grate: bars with transparent gaps (a culvert under the wall)."""
    surf = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    for x in range(0, SIZE, 16):
        pygame.draw.rect(surf, (60, 62, 68, 255), (x, 0, 5, SIZE))
    for y in (4, SIZE // 2, SIZE - 8):
        pygame.draw.rect(surf, (50, 52, 58, 255), (0, y, SIZE, 5))
    return surf


def gen_marble(rng: random.Random) -> pygame.Surface:
    """A polished marble floor: cream tiles with grey veins."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (226, 220, 206), 5, rng)
    _cracks(surf, rng, 5, (170, 166, 160))
    for i in range(0, SIZE, SIZE // 2):
        pygame.draw.line(surf, (186, 180, 168), (i, 0), (i, SIZE), 1)
        pygame.draw.line(surf, (186, 180, 168), (0, i), (SIZE, i), 1)
    return surf


def gen_carpet(rng: random.Random) -> pygame.Surface:
    """A rich carpet: deep red with a gold border and a repeating lozenge."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (128, 24, 32), 6, rng)
    pygame.draw.rect(surf, (196, 156, 60), (0, 0, SIZE, SIZE), 8)
    pygame.draw.rect(surf, (92, 16, 24), (10, 10, SIZE - 20, SIZE - 20), 2)
    c = SIZE // 2
    pygame.draw.polygon(surf, (196, 156, 60), [(c, 24), (SIZE - 24, c), (c, SIZE - 24), (24, c)], 3)
    pygame.draw.polygon(surf, (60, 44, 96), [(c, 40), (SIZE - 40, c), (c, SIZE - 40), (40, c)])
    return surf


def gen_plaster(rng: random.Random, timber: bool = False) -> pygame.Surface:
    """Lime plaster (cream, a little stained); with ``timber`` the dark beams of a half-timbered house."""
    surf = pygame.Surface((SIZE, SIZE))
    _fill_noise(surf, (214, 204, 180), 8, rng)
    shade = _value_noise(rng, SIZE, SIZE, 4)
    for y in range(SIZE):
        for x in range(SIZE):
            c = surf.get_at((x, y))
            k = 0.9 + shade[y][x] * 0.16
            surf.set_at((x, y), (_clamp(c.r * k), _clamp(c.g * k), _clamp(c.b * k)))
    if timber:
        beam = (70, 48, 30)
        for x in (0, SIZE // 2):
            pygame.draw.rect(surf, beam, (x, 0, 9, SIZE))
        pygame.draw.rect(surf, beam, (0, 0, SIZE, 9))
        pygame.draw.rect(surf, beam, (0, SIZE // 2 - 4, SIZE, 9))
        pygame.draw.line(surf, beam, (9, SIZE // 2), (SIZE // 2, SIZE), 8)
    return surf


def gen_field(rng: random.Random, alpha_edge: bool = False) -> pygame.Surface:
    """A tilled field: furrows of dark earth and rows of green shoots (``alpha_edge``: the same,
    fading out towards the bottom - the strip where a field's edge meets the grass)."""
    surf = gen_dirt(rng)
    for y in range(0, SIZE, 16):
        pygame.draw.line(surf, (70, 54, 36), (0, y), (SIZE, y), 4)
        for x in range(rng.randrange(4), SIZE, 6):
            pygame.draw.line(surf, rng.choice([(76, 128, 50), (98, 150, 60)]), (x, y + 8), (x + 1, y + 3), 2)
    if alpha_edge:
        faded = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
        for y in range(SIZE):
            a = _clamp(255 * (y / (SIZE - 1)) ** 2)
            for x in range(SIZE):
                c = surf.get_at((x, y))
                faded.set_at((x, y), (c.r, c.g, c.b, a))
        return faded
    return surf


GENERATORS: Dict[str, Callable[[random.Random], pygame.Surface]] = {
    "ground": gen_ground,
    "wall": lambda rng: gen_stone_blocks(rng),
    "stone": lambda rng: gen_stone_blocks(rng, base=(105, 100, 95), rows=5),
    "building": lambda rng: gen_planks(rng),
    "wood": gen_wood,
    "door": gen_door,
    "shelf": gen_shelf,
    "bed": gen_bed,
    "ore": gen_ore,
    "guard": gen_guard,
    "npc": gen_npc,
    "skin": gen_skin,
    "flame": gen_flame,
    "torch": gen_torch,
    "sun": lambda rng: gen_disc((255, 250, 220), (255, 210, 120)),
    "sun_glow": gen_sun_glow,
    "moon": gen_moon,
    "iron": gen_iron,
    "lockbox": gen_lockbox,
    "lockbox_side": gen_lockbox_side,
    "wax": gen_wax,
    "cobble": gen_cobble,
    "cobble_edge": lambda rng: gen_cobble(rng, alpha_edge=True),
    "roof": gen_roof,
    "roof_slate": gen_roof_slate,
    "cloth": gen_cloth,
    "leather": gen_leather,
    "water": gen_water,
    "coals": gen_coals,
    "sign_smith": gen_sign_smith,
    "sign_store": gen_sign_store,
    "sign_tailor": gen_sign_tailor,
    "sign_inn": gen_sign_inn,
    "bark": gen_bark,
    "leaves": gen_leaves,
    "pine": gen_pine,
    "barrel": gen_barrel,
    "log_end": gen_log_end,
    "crate": gen_crate,
    "hay": gen_hay,
    "thatch": gen_thatch,
    "cobble_cracked": gen_cobble_cracked,
    "cobble_cracked_edge": lambda rng: gen_cobble_cracked(rng, alpha_edge=True),
    "wall_cracked": gen_wall_cracked,
    "planks_rot": gen_planks_rot,
    "crack_decal": gen_crack_decal,
    "stain_decal": gen_stain_decal,
    "dirt": gen_dirt,
    "dirt_edge": lambda rng: gen_dirt(rng, alpha_edge=True),
    "rock": gen_rock,
    "grate": gen_grate,
    "marble": gen_marble,
    "carpet": gen_carpet,
    "plaster": gen_plaster,
    "plaster_timber": lambda rng: gen_plaster(rng, timber=True),
    "field": gen_field,
    "field_edge": lambda rng: gen_field(rng, alpha_edge=True),
    "stream": gen_stream,
    "lake": gen_lake,
    "sand": gen_sand,
    "snow": gen_snow,
    "cliff": gen_cliff,
    "cave_rock": gen_cave_rock,
    "cave_floor": gen_cave_floor,
    "mushroom": gen_mushroom,
    "bone": gen_bone,
    "notice": gen_notice,
    "sign_arm": gen_sign_arm,
    "plaque": gen_plaque,
    "parchment": gen_parchment,
    "banner_greyhold": gen_banner_greyhold,
    "banner_deep": gen_banner_deep,
    "ashlar": gen_ashlar,
    "manor_plaster": gen_manor_plaster,
    "window_manor": gen_window_manor,
    "window_cottage": gen_window_cottage,
    "window_lancet": gen_window_lancet,
    "painting_portrait": gen_painting_portrait,
    "painting_landscape": gen_painting_landscape,
    "rug": gen_rug,
    "rug_blue": gen_rug_blue,
    "tile_floor": gen_tile_floor,
    "brick": gen_brick,
    "books": gen_books,
    "clock_face": gen_clock_face,
    "wallpaper": gen_wallpaper,
    "flowers": gen_flowers,
    "gilt": gen_gilt,
    "sign_bakery": gen_sign_bakery,
}


def generate_texture(key: str, seed: int = 7) -> pygame.Surface:
    """Procedural texture for ``key`` (deterministic for a given seed); magenta if unknown."""
    gen = GENERATORS.get(key)
    if gen is None:
        surf = pygame.Surface((SIZE, SIZE))
        surf.fill((255, 0, 255))
        return surf
    return gen(random.Random(f"{key}:{seed}"))


# ----------------------------------------------------------------------------- artwork (not tiles)
LOGO_SIZE = (900, 440)


def _font(names, size, bold=False):
    for name in names:
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def gen_logo(rng: random.Random) -> pygame.Surface:
    """The title: UNPREPARED over a moonlit roofline - a town's gables and chimneys, the watchtower
    with its battlements, and on the tower's top a hooded figure crouched against the moon - a
    dagger laid under the name and the tagline beneath. Transparent round about (it is drawn over
    the live town on the start screen)."""
    w, h = LOGO_SIZE
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    # the moon and its glow, up and to the right of the middle
    mx, my, mr = 612, 150, 92
    for r in range(mr + 60, mr, -3):
        a = int(70 * (1.0 - (r - mr) / 60.0) ** 2)
        pygame.draw.circle(surf, (210, 220, 255, a), (mx, my), r)
    pygame.draw.circle(surf, (238, 236, 214, 255), (mx, my), mr)
    for _ in range(9):                                          # (its seas)
        cx, cy = mx + rng.randint(-60, 60), my + rng.randint(-60, 60)
        if (cx - mx) ** 2 + (cy - my) ** 2 < (mr - 18) ** 2:
            pygame.draw.circle(surf, (214, 212, 190, 255), (cx, cy), rng.randint(6, 18))
    # the roofline: a ridge of gables across the bottom of the moon, the tower on the right
    ink = (14, 14, 24, 255)
    base = 232
    x = 40
    roofs = []
    while x < w - 40:
        width = rng.randint(46, 86)
        eave = base - rng.randint(4, 30)
        roofs.append((x, width, eave))
        x += width + rng.randint(-6, 4)
    for x0, width, eave in roofs:
        apex = eave - width * 0.42
        pygame.draw.polygon(surf, ink, [(x0, base + 2), (x0, eave), (x0 + width / 2, apex), (x0 + width, eave), (x0 + width, base + 2)])
        if rng.random() < 0.5:                                   # a chimney
            cx = x0 + width * rng.uniform(0.2, 0.7)
            cy = apex + (eave - apex) * 0.4
            pygame.draw.rect(surf, ink, (cx, cy - 18, 9, 22))
        if rng.random() < 0.6:                                   # a lit window under the eaves
            pygame.draw.rect(surf, (240, 190, 90, 255), (x0 + width * 0.4, eave + 8, 7, 9))
    tx1, tx2, top = 640, 700, 70                                # the watchtower against the moon
    pygame.draw.rect(surf, ink, (tx1, top + 14, tx2 - tx1, base - top - 12))
    for k in range(4):
        pygame.draw.rect(surf, ink, (tx1 + k * 17, top, 10, 16))
    pygame.draw.rect(surf, (240, 190, 90, 255), (tx1 + 26, top + 60, 8, 14))
    # the thief: crouched on the tower's top, hood up, a cloak hanging behind
    fx, fy = tx1 + 22, top - 2
    pygame.draw.polygon(surf, ink, [(fx - 14, fy), (fx - 6, fy - 26), (fx + 4, fy - 34), (fx + 14, fy - 26), (fx + 18, fy - 6),
                                    (fx + 26, fy)])
    pygame.draw.circle(surf, ink, (fx + 6, fy - 36), 9)
    pygame.draw.polygon(surf, ink, [(fx - 2, fy - 42), (fx + 8, fy - 50), (fx + 14, fy - 38)])      # the hood's peak
    pygame.draw.line(surf, ink, (fx + 18, fy - 18), (fx + 34, fy - 28), 3)                          # a grapnel's rope
    # the name: big, moon-silver, edged in black, a soft shadow under it
    title_font = _font(("texgyretermes", "liberationserif", "dejavuserif", "georgia", "timesnewroman"), 118, bold=True)
    text = "UNPREPARED"
    tw, th = title_font.size(text)
    tx, ty = (w - tw) // 2, base - 38
    shadow = title_font.render(text, True, (0, 0, 0))
    for dx, dy in ((4, 6), (2, 4)):
        shadow.set_alpha(120)
        surf.blit(shadow, (tx + dx, ty + dy))
    outline = title_font.render(text, True, (8, 8, 14))
    for dx in (-3, 0, 3):
        for dy in (-3, 0, 3):
            if dx or dy:
                surf.blit(outline, (tx + dx, ty + dy))
    face = title_font.render(text, True, (255, 255, 255))
    shade = pygame.Surface((tw, th), pygame.SRCALPHA)          # a silver-to-gold sheen, top to bottom
    for row in range(th):
        t = row / max(1, th - 1)
        shade.fill((int(232 - 40 * t), int(232 - 70 * t), int(240 - 150 * t), 255), (0, row, tw, 1))
    face.blit(shade, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(face, (tx, ty))
    # the dagger under it: blade, crossguard, grip, pommel
    y = ty + th - 2
    bx1, bx2 = tx + 40, tx + tw - 110
    pygame.draw.polygon(surf, (10, 10, 16, 255), [(bx1 - 3, y), (bx2, y - 9), (bx2, y + 9)])      # (an edge to it)
    pygame.draw.polygon(surf, (205, 210, 222, 255), [(bx1, y), (bx2, y - 6), (bx2, y + 6)])
    pygame.draw.line(surf, (120, 124, 140, 255), (bx1 + 10, y), (bx2, y), 2)                       # the fuller
    pygame.draw.rect(surf, (150, 120, 60, 255), (bx2, y - 13, 7, 26))
    pygame.draw.rect(surf, (70, 40, 24, 255), (bx2 + 7, y - 4, 44, 8))
    pygame.draw.circle(surf, (150, 120, 60, 255), (bx2 + 55, y), 7)
    # the tagline
    tag_font = _font(("texgyretermes", "liberationserif", "dejavuserif"), 24)
    tag = "   ".join("A TALE OF STEALTH AND THEFT".split(" "))
    tagsurf = tag_font.render(tag, True, (200, 190, 160))
    surf.blit(tagsurf, ((w - tagsurf.get_width()) // 2, y + 18))
    return surf


ARTWORK = {"logo": gen_logo}


def write_textures(out_dir: str, seed: int = 7, keys=TEXTURE_KEYS, overwrite: bool = True) -> list:
    """Write the tiles ``keys`` (and any artwork among them - the start screen's logo) as PNGs."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for key in keys:
        path = os.path.join(out_dir, f"{key}.png")
        if overwrite or not os.path.exists(path):
            image = ARTWORK[key](random.Random(f"{key}:{seed}")) if key in ARTWORK else generate_texture(key, seed)
            pygame.image.save(image, path)
            written.append(path)
    return written
