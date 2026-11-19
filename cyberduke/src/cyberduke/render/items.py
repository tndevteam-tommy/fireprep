"""Small things people handle: a fish, a butcher knife, sushi, takeaway boxes, parcels,
a phone, a cigarette, a cred chip... Each is a few boxes drawn around its own origin (the
point it is held by, or the middle of its underside when it lies on something), +y up and
+z away from whoever holds it.

People hold them (``HumanoidPose.hand_l`` / ``hand_r``: ``render.humanoid`` draws them in
the hand) and they lie about the world (``sim.items``: a knife on a chopping board, a box
waiting on a counter, a parcel left on a doorstep; ``draw_loose_items``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from cyberduke.animation.hands import outward
from cyberduke.animation.street import (
    CHOPSTICK_LENGTH,
    NOODLE_BITES,
    SUSHI_PIECES,
    TWO_HANDED_GRIP,
    noodles_top,
    sushi_piece_at,
)
from cyberduke.render.context import DrawContext
from cyberduke.render.textures import TextureMap

Drawer = Callable[[DrawContext, TextureMap], None]


def _box(ctx, x, y, z, sx, sy, sz, color, tex=None, emission=None) -> None:
    ctx.color(*color)
    if emission is not None:
        ctx.emission(emission)
    ctx.box(x, y, z, sx, sy, sz, tex)
    if emission is not None:
        ctx.emission(None)


FISH_LENGTH = 0.25  # the body, nose to the root of the tail (the tail fin adds 7 cm)
FISH_HEIGHT = 0.09  # back to belly
FISH_THICK = 0.048  # flank to flank
FISH_MID = 0.03  # how far ahead of the origin the middle of the body is


def _fin_at(ctx: DrawContext, t: TextureMap, root, direction, length, width, thick=0.004) -> None:
    """A fin (``shapes.fin``) with its root at ``root`` sticking out along ``direction``
    (in the fish's frame), flat across the body's side."""
    dx, dy, dz = direction
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / n, dy / n, dz / n
    # the turn taking the fin's +y onto the direction: about y x direction
    ax, az = dz, -dx
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dy))))
    x, y, z = root
    with ctx.transformed(translate=(x + dx * length / 2, y + dy * length / 2, z + dz * length / 2)):
        if math.hypot(ax, az) > 1e-6:
            ctx.rotate(angle, ax, 0.0, az)
        elif dy < 0.0:
            ctx.rotate(180.0, 1.0, 0.0, 0.0)
        ctx.scale(thick, length, width)
        ctx.shape("fin", t.get("fin"))


def _fish_upright(ctx: DrawContext, t: TextureMap) -> None:
    """The fish the way it swims (its middle at the origin, nose to +z, back up)."""
    length, high, thick = FISH_LENGTH, FISH_HEIGHT, FISH_THICK
    zc = FISH_MID
    ctx.color(1.0, 1.0, 1.0)
    ctx.specular((0.55, 0.6, 0.65), 40.0)  # wet and slippery: it shines
    with ctx.transformed((0.0, 0.0, zc), [(90.0, 1.0, 0.0, 0.0)], (thick, length, high)):
        ctx.shape("fish_body", t.get("fish_skin"))
    ctx.specular(None)
    ctx.color(0.85, 0.88, 0.95)
    tail = zc - length / 2.0 + 0.012
    for up in (1.0, -1.0):  # the forked tail: two lobes, up and back, down and back
        _fin_at(ctx, t, (0.0, 0.004 * up, tail), (0.0, 0.62 * up, -0.78), 0.075, 0.04, 0.005)
    _fin_at(ctx, t, (0.0, high * 0.4, zc + 0.01), (0.0, 0.8, -0.6), 0.042, 0.055)  # dorsal
    _fin_at(ctx, t, (0.0, high * 0.22, zc - 0.085), (0.0, 0.7, -0.7), 0.016, 0.014)  # adipose
    _fin_at(ctx, t, (0.0, -high * 0.3, zc - 0.07), (0.0, -0.7, -0.7), 0.03, 0.03)  # anal
    for side in (1.0, -1.0):
        # pectoral fins just behind the gill covers, pelvic fins under the belly
        _fin_at(
            ctx,
            t,
            (side * thick * 0.36, -high * 0.18, zc + 0.05),
            (side * 0.22, -0.35, -0.9),
            0.045,
            0.022,
        )
        _fin_at(
            ctx,
            t,
            (side * thick * 0.22, -high * 0.36, zc - 0.02),
            (side * 0.15, -0.6, -0.78),
            0.03,
            0.016,
        )
    # the eyes: glossy black domes in the gold rings painted on the skin
    ctx.color(0.03, 0.03, 0.04)
    ctx.specular((1.0, 1.0, 1.0), 90.0)
    for side in (1.0, -1.0):
        with ctx.transformed(
            (side * thick * 0.42, high * 0.17, zc + length * 0.36), scale=(0.006, 0.009, 0.009)
        ):
            ctx.shape("sphere")
    ctx.specular(None)


def _fish(ctx: DrawContext, t: TextureMap) -> None:
    """A whole salmon lying on its side (on a board, on the flat of a hand): a tapering
    body with a rounded head, silver scales, a dark speckled back and a pale belly, a
    forked tail, the fins, and the eye looking up."""
    with ctx.transformed((0.0, FISH_THICK / 2.0 + 0.003, 0.0), [(90.0, 0.0, 0.0, 1.0)]):
        _fish_upright(ctx, t)


def _sushi(ctx: DrawContext, t: TextureMap) -> None:
    """The fish cut up on the board: the pieces that go into the box (``SUSHI_PIECES`` -
    the same nigiri, as many as the box is packed with), in a row down the board."""
    for k in range(SUSHI_PIECES):
        z = (k - (SUSHI_PIECES - 1) / 2.0) * 0.045
        with ctx.transformed((0.0, 0.0, z), [(90.0, 0.0, 1.0, 0.0)]):
            _sushi_piece(ctx, 0.0, 0.0, 0.0, t)


KNIFE_HANDLE = (0.025, 0.03, 0.11)  # the handle's width, height and length
KNIFE_BLADE = (0.008, 0.07, 0.22)


def _knife(ctx: DrawContext, t: TextureMap) -> None:
    """A cleaver-ish butcher's knife held by its handle (the origin): a broad blade on
    forward, its back level with the top of the handle and its edge down - below the
    grip, the way a knife is held to cut (it used to be the other way up)."""
    hw, hh, hl = KNIFE_HANDLE
    bw, bh, bl = KNIFE_BLADE
    _box(ctx, 0.0, 0.0, -0.02, hw, hh, hl, (0.12, 0.08, 0.06))
    _box(ctx, 0.0, 0.0, 0.04, hw + 0.004, hh + 0.004, 0.012, (0.7, 0.7, 0.72))  # the bolster
    ctx.specular((0.9, 0.9, 0.95), 80.0)
    # bright steel: it catches whatever light there is
    spine = hh / 2.0 - 0.002
    _box(
        ctx,
        0.0,
        spine - bh / 2.0,
        0.046 + bl / 2.0,
        bw,
        bh,
        bl,
        (0.85, 0.87, 0.9),
        t.get("metal"),
        emission=(0.25, 0.26, 0.3),
    )
    ctx.specular(None)


def _knife_lying(ctx: DrawContext, t: TextureMap) -> None:
    """The knife put down on the board: on its side, flat, the edge towards whoever put
    it there (its origin is where the handle's middle is when it is held: the handle's
    underside 1.5 cm below it)."""
    ctx.push()
    ctx.translate(0.0, -KNIFE_HANDLE[1] / 2.0 + KNIFE_HANDLE[0] / 2.0, 0.0)
    ctx.rotate(90.0, 0.0, 0.0, 1.0)
    hw, hh, hl = KNIFE_HANDLE
    bw, bh, bl = KNIFE_BLADE
    _box(ctx, 0.0, 0.0, -0.02, hw, hh, hl, (0.12, 0.08, 0.06))
    _box(ctx, 0.0, 0.0, 0.04, hw + 0.004, hh + 0.004, 0.012, (0.7, 0.7, 0.72))
    ctx.specular((0.9, 0.9, 0.95), 80.0)
    # (on its side the blade lies on the board too: 1 mm off it, its flat face down)
    _box(
        ctx,
        hw / 2.0 - bw / 2.0 - 0.001,
        hh / 2.0 - 0.002 - bh / 2.0,
        0.046 + bl / 2.0,
        bw,
        bh,
        bl,
        (0.85, 0.87, 0.9),
        t.get("metal"),
        emission=(0.25, 0.26, 0.3),
    )
    ctx.specular(None)
    ctx.pop()


def _takeaway(ctx: DrawContext, t: TextureMap) -> None:
    """A closed kraft takeaway box with a red band."""
    _box(ctx, 0.0, 0.035, 0.0, 0.2, 0.07, 0.14, (0.86, 0.74, 0.55), t.get("cardboard"))
    _box(ctx, 0.0, 0.035, 0.0, 0.205, 0.074, 0.03, (0.8, 0.15, 0.12))


def _takeaway_open(ctx: DrawContext, t: TextureMap) -> None:
    _box(ctx, 0.0, 0.025, 0.0, 0.2, 0.05, 0.14, (0.86, 0.74, 0.55), t.get("cardboard"))
    _box(ctx, 0.0, 0.056, -0.075, 0.2, 0.012, 0.13, (0.86, 0.74, 0.55), t.get("cardboard"))


def _sushi_piece(
    ctx: DrawContext, x: float, y: float, z: float, t: TextureMap | None = None
) -> None:
    """One nigiri: a rounded block of rice with a slab of salmon draped over it, its
    stripes of fat showing (``y`` its underside)."""
    ctx.color(0.95, 0.95, 0.92)
    with ctx.transformed((x, y + 0.0125, z), scale=(0.03, 0.025, 0.05)):
        ctx.shape("rbox")
    ctx.color(1.0, 1.0, 1.0)
    with ctx.transformed((x, y + 0.029, z), scale=(0.033, 0.011, 0.056)):
        ctx.shape("plate", t.get("salmon") if t is not None else None)


def _sushi_box(pieces: int) -> Drawer:
    """The open box with ``pieces`` of sushi left in it (being packed, or being eaten
    one piece at a time - from the right-hand end)."""

    def draw(ctx: DrawContext, t: TextureMap) -> None:
        _takeaway_open(ctx, t)
        for j in range(pieces):
            x, y, z = sushi_piece_at(j)
            _sushi_piece(ctx, x, y - 0.035, z, t)

    return draw


def _noodle_box(bites: int) -> Drawer:
    """A noodle box with ``bites`` chopstickfuls of chow mein left in it (the full box
    has it peeking out of the top; an empty one is just the box)."""

    def draw(ctx: DrawContext, t: TextureMap) -> None:
        _box(ctx, 0.0, 0.06, 0.0, 0.12, 0.12, 0.12, (0.95, 0.95, 0.92))
        _box(ctx, 0.0, 0.06, 0.0, 0.125, 0.04, 0.125, (0.8, 0.15, 0.12))
        if bites:
            top = noodles_top(bites)
            _box(ctx, 0.0, top - 0.005, 0.0, 0.1, 0.01, 0.1, (0.9, 0.7, 0.3))  # chow mein

    return draw


CHOPSTICKS = CHOPSTICK_LENGTH  # from where they are held to the tips


def _chopsticks(ctx: DrawContext, t: TextureMap) -> None:
    for x in (-0.008, 0.008):
        _box(
            ctx, x, 0.0, CHOPSTICKS / 2.0 - 0.01, 0.007, 0.007, CHOPSTICKS + 0.02, (0.75, 0.6, 0.4)
        )


def _chopsticks_sushi(ctx: DrawContext, t: TextureMap) -> None:
    """Chopsticks with a piece of sushi held at the tips."""
    _chopsticks(ctx, t)
    _sushi_piece(ctx, 0.0, -0.02, CHOPSTICKS - 0.015, t)


def _chopsticks_noodles(ctx: DrawContext, t: TextureMap) -> None:
    """Chopsticks with a tangle of chow mein hanging from the tips."""
    _chopsticks(ctx, t)
    _box(ctx, 0.0, -0.02, CHOPSTICKS - 0.01, 0.03, 0.05, 0.025, (0.9, 0.7, 0.3))


def _parcel(ctx: DrawContext, t: TextureMap) -> None:
    """A parcel in brown paper tied with string."""
    _box(ctx, 0.0, 0.1, 0.0, 0.34, 0.2, 0.26, (0.72, 0.58, 0.4), t.get("cardboard"))
    _box(ctx, 0.0, 0.1, 0.0, 0.345, 0.205, 0.015, (0.85, 0.82, 0.7))
    _box(ctx, 0.0, 0.1, 0.0, 0.015, 0.205, 0.265, (0.85, 0.82, 0.7))
    _box(ctx, 0.08, 0.203, 0.06, 0.09, 0.004, 0.06, (0.95, 0.95, 0.95))  # a label


def _gift(ctx: DrawContext, t: TextureMap) -> None:
    """A small thing wrapped up at the counter: glossy paper, a bright tape cross."""
    ctx.specular((0.6, 0.6, 0.7), 60.0)
    _box(ctx, 0.0, 0.04, 0.0, 0.16, 0.08, 0.12, (0.15, 0.45, 0.7))
    ctx.specular(None)
    _box(ctx, 0.0, 0.04, 0.0, 0.165, 0.084, 0.02, (1.0, 0.75, 0.1))
    _box(ctx, 0.0, 0.04, 0.0, 0.02, 0.084, 0.125, (1.0, 0.75, 0.1))


def _cyberdeck(ctx: DrawContext, t: TextureMap) -> None:
    _box(ctx, 0.0, 0.02, 0.0, 0.18, 0.04, 0.11, (0.12, 0.12, 0.14), t.get("metal"))
    _box(ctx, 0.0, 0.042, 0.02, 0.12, 0.004, 0.05, (0.1, 0.6, 0.9), emission=(0.2, 1.2, 1.8))


def _chip(ctx: DrawContext, t: TextureMap) -> None:
    _box(ctx, 0.0, 0.006, 0.0, 0.05, 0.012, 0.07, (1.0, 0.7, 0.0), emission=(1.6, 1.0, 0.0))


def _remote(ctx: DrawContext, t: TextureMap) -> None:
    """A TV remote held by its butt end (the origin), pointing on out of the fist at the
    set: a worn black bar, a red power button, rows of rubber buttons, the IR window."""
    _box(ctx, 0.0, 0.0, 0.035, 0.042, 0.02, 0.17, (0.06, 0.06, 0.07))
    _box(ctx, 0.0, 0.0105, 0.095, 0.012, 0.003, 0.012, (0.9, 0.1, 0.1), emission=(0.4, 0.0, 0.0))
    for k in range(3):
        for side in (-1.0, 1.0):
            _box(ctx, 0.01 * side, 0.0105, 0.07 - 0.018 * k, 0.01, 0.002, 0.01, (0.35, 0.35, 0.4))
    _box(ctx, 0.0, 0.0, 0.1205, 0.03, 0.012, 0.002, (0.25, 0.05, 0.08))


def _phone(ctx: DrawContext, t: TextureMap) -> None:
    _box(ctx, 0.0, 0.0, 0.0, 0.07, 0.012, 0.14, (0.08, 0.08, 0.1))
    _box(ctx, 0.0, 0.007, 0.0, 0.06, 0.002, 0.12, (0.4, 0.8, 1.0), emission=(0.5, 1.1, 1.5))


CHERRY_AT = (0.0, 0.0, 0.062)  # the lit end of a cigarette, in its own frame
CHERRY_DULL = (0.35, 0.07, 0.01)  # smouldering between drags: a dull red glow under ash
CHERRY_HOT = (3.2, 1.1, 0.2)  # drawn on: it flares orange-white


def cherry_glow(ember: float) -> tuple[float, float, float]:
    """The emission of a cigarette's lit end, ``ember`` 0 (smouldering) .. 1 (a drag)."""
    e = max(0.0, min(1.0, ember))
    return tuple(a + (b - a) * e for a, b in zip(CHERRY_DULL, CHERRY_HOT, strict=True))  # type: ignore[return-value]


def _cigarette(ctx: DrawContext, t: TextureMap, ember: float = 0.0) -> None:
    """Held between the fingers 2.5 cm from the filter: the filter end behind the hand
    (at the lips, on a drag), the lit end out in front of it - grey with ash and glowing
    dully, flaring up while it is drawn on (``ember``)."""
    _box(ctx, 0.0, 0.0, -0.025, 0.0085, 0.0085, 0.02, (0.85, 0.6, 0.3))  # the filter
    _box(ctx, 0.0, 0.0, 0.02, 0.008, 0.008, 0.07, (0.95, 0.93, 0.88))
    e = max(0.0, min(1.0, ember))
    ash = (0.45 + 0.55 * e, 0.3 + 0.15 * e, 0.25 - 0.15 * e)  # ash grey -> glowing orange
    _box(ctx, 0.0, 0.0, 0.058, 0.009, 0.009, 0.008, ash, emission=cherry_glow(e))


def _soda(ctx: DrawContext, t: TextureMap) -> None:
    """A can of something from the vending machine, held upright."""
    ctx.color(1.0, 1.0, 1.0)
    ctx.push()
    ctx.translate(0.0, -0.04, 0.0)
    ctx.cylinder(0.033, 0.033, 0.12, slices=10, tex_id=t.get("soda_can"))
    ctx.pop()


BEER = (0.86, 0.55, 0.12)
FOAM = (0.95, 0.93, 0.85)
GLASS = (0.72, 0.82, 0.88)


def _pint_on(ctx: DrawContext, full: bool) -> None:
    """A pint glass standing on its base (the origin): amber to the brim under a head of
    foam - or empty, a ring of froth left down the inside."""
    ctx.specular((1.0, 1.0, 1.0), 90.0)
    if full:
        ctx.color(*BEER)
        ctx.emission((0.18, 0.1, 0.0))
        ctx.cylinder(0.032, 0.038, 0.12, slices=12)
        ctx.emission(None)
        ctx.push()
        ctx.translate(0.0, 0.12, 0.0)
        ctx.color(*FOAM)
        ctx.cylinder(0.038, 0.04, 0.028, slices=12)
        ctx.pop()
    else:
        ctx.color(*GLASS)
        ctx.cylinder(0.032, 0.04, 0.148, slices=12)
        ctx.push()
        ctx.translate(0.0, 0.09, 0.0)
        ctx.color(*FOAM)
        ctx.cylinder(0.0375, 0.0385, 0.012, slices=12)
        ctx.pop()
    ctx.specular(None)


def _beer(ctx: DrawContext, t: TextureMap) -> None:
    """A pint of beer, held round its middle."""
    ctx.push()
    ctx.translate(0.0, -0.055, 0.0)
    _pint_on(ctx, True)
    ctx.pop()


def _glass(ctx: DrawContext, t: TextureMap) -> None:
    """An empty pint glass, held round its middle."""
    ctx.push()
    ctx.translate(0.0, -0.055, 0.0)
    _pint_on(ctx, False)
    ctx.pop()


def _closed_sign(ctx: DrawContext, t: TextureMap) -> None:
    """A little A-frame sign stood on a counter: CLOSED (a red panel, a white bar) on both
    faces - its two panels leaning in to meet at the top (an A, not a V), one face to the
    front (+z) and one to the back."""
    height, lean = 0.26, 16.0
    for side in (-1.0, 1.0):
        ctx.push()
        # each panel's foot out on its side, its top leaning in over the middle
        ctx.translate(0.0, 0.0, side * height * math.sin(math.radians(lean)))
        ctx.rotate(-side * lean, 1.0, 0.0, 0.0)
        _box(
            ctx,
            0.0,
            height / 2.0,
            0.0,
            0.34,
            height,
            0.012,
            (0.7, 0.08, 0.08),
            emission=(0.5, 0.04, 0.04),
        )
        _box(ctx, 0.0, height / 2.0 + 0.02, side * 0.0085, 0.26, 0.05, 0.005, (0.95, 0.95, 0.9))
        ctx.pop()
    _box(
        ctx,
        0.0,
        height * math.cos(math.radians(lean)) + 0.004,
        0.0,
        0.36,
        0.014,
        0.02,
        (0.25, 0.25, 0.27),
    )  # the hinge


def _stall_cover(ctx: DrawContext, t: TextureMap) -> None:
    """A tarp thrown over a trestle table for the night, hanging down the sides (drawn
    from the middle of the table's top)."""
    tarp = t.get("fabric")
    col = (0.22, 0.26, 0.4)
    _box(ctx, 0.0, 0.09, 0.0, 0.72, 0.03, 3.56, col, tarp)
    for side in (-1.0, 1.0):
        _box(ctx, side * 0.35, -0.15, 0.0, 0.03, 0.5, 3.5, col, tarp)
        _box(ctx, 0.0, -0.15, side * 1.77, 0.68, 0.5, 0.03, col, tarp)


def _cred(ctx: DrawContext, t: TextureMap) -> None:
    _box(ctx, 0.0, 0.0, 0.0, 0.055, 0.004, 0.085, (0.9, 0.8, 0.3), emission=(0.4, 0.3, 0.05))


def _bag(ctx: DrawContext, t: TextureMap) -> None:
    """A paper grocery bag hanging from the hand, greens sticking out of the top."""
    _box(ctx, 0.0, -0.2, 0.0, 0.26, 0.3, 0.16, (0.7, 0.58, 0.42), t.get("cardboard"))
    _box(ctx, 0.05, -0.02, 0.02, 0.04, 0.1, 0.04, (0.3, 0.7, 0.25))
    _box(ctx, -0.05, -0.03, -0.02, 0.05, 0.08, 0.05, (0.85, 0.3, 0.2))


def _datastick(ctx: DrawContext, t: TextureMap) -> None:
    """A scuffed data stick lying flat: yellow casing, a steel plug, a taped-on label and
    a green LED (so it catches the eye in a dark room)."""
    _box(ctx, 0.0, 0.008, 0.0, 0.026, 0.016, 0.07, (0.85, 0.7, 0.15))
    _box(ctx, 0.0, 0.008, 0.044, 0.018, 0.008, 0.02, (0.7, 0.72, 0.76), t.get("metal"))
    _box(ctx, 0.0, 0.0165, -0.006, 0.02, 0.002, 0.032, (0.9, 0.88, 0.8))
    _box(ctx, 0.0, 0.0165, -0.03, 0.006, 0.003, 0.006, (0.2, 1.0, 0.4), emission=(0.4, 2.2, 0.8))


def _rag(ctx: DrawContext, t: TextureMap) -> None:
    _box(ctx, 0.0, 0.006, 0.0, 0.16, 0.012, 0.12, (0.7, 0.72, 0.75), t.get("fabric"))


def _home_door(ctx: DrawContext, t: TextureMap) -> None:
    """A tenement front door, drawn from its hinge (the origin) along +z: the leaf, its
    handle on both faces and a kick plate."""
    _box(ctx, 0.0, 1.1, 0.5, 0.1, 2.2, 1.0, (0.35, 0.22, 0.14), t.get("leather"))
    _box(ctx, 0.0, 0.12, 0.5, 0.11, 0.2, 0.96, (0.3, 0.31, 0.34), t.get("metal"))
    for side in (-1.0, 1.0):
        _box(ctx, 0.07 * side, 1.0, 0.86, 0.04, 0.04, 0.12, (0.75, 0.7, 0.5))


# name -> (drawer, how it sits in a hand: "level" stays upright, "along" points down the
# forearm, "outward" sticks out from between the fingers - across the forearm, away from
# the body - and "hang" hangs below the hand). Anything but a level thing points where
# the pose aims it, if it does (``HumanoidPose.aim_l`` / ``aim_r``).
ITEMS: dict[str, tuple[Drawer, str]] = {
    "fish": (_fish, "level"),
    "sushi": (_sushi, "level"),
    "knife": (_knife, "along"),
    "takeaway": (_takeaway, "level"),
    "takeaway_open": (_takeaway_open, "level"),
    "sushi_open": (_sushi_box(SUSHI_PIECES), "level"),
    **{f"sushi_open{n}": (_sushi_box(n), "level") for n in range(1, SUSHI_PIECES)},
    "noodles": (_noodle_box(NOODLE_BITES), "level"),
    **{f"noodles{n}": (_noodle_box(n), "level") for n in range(NOODLE_BITES)},
    "chopsticks": (_chopsticks, "along"),
    "chopsticks_sushi": (_chopsticks_sushi, "along"),
    "chopsticks_noodles": (_chopsticks_noodles, "along"),
    "parcel": (_parcel, "level"),
    "gift": (_gift, "level"),
    "cyberdeck": (_cyberdeck, "level"),
    "chip": (_chip, "level"),
    "phone": (_phone, "level"),
    "remote": (_remote, "along"),
    "cigarette": (_cigarette, "outward"),
    "soda": (_soda, "level"),
    "beer": (_beer, "level"),
    "glass": (_glass, "level"),
    "closed_sign": (_closed_sign, "level"),
    "stall_cover": (_stall_cover, "level"),
    "cred": (_cred, "level"),
    "bag": (_bag, "hang"),
    "rag": (_rag, "level"),
    "datastick": (_datastick, "level"),
    "home_door": (_home_door, "level"),
}


def draw_item(ctx: DrawContext, name: str, textures: TextureMap, ember: float = 0.0) -> None:
    """Draws item ``name`` at the current origin (unknown names draw nothing). ``ember``:
    how hot a cigarette's tip is (``HumanoidPose.ember``)."""
    found = ITEMS.get(name)
    if found is None:
        return
    if name == "cigarette":
        _cigarette(ctx, textures, ember)
    else:
        found[0](ctx, textures)
    ctx.emission(None)
    ctx.color(1.0, 1.0, 1.0)


def held_transform(
    name: str,
    hand: tuple[float, float, float],
    forearm,
    both: bool = False,
    aim=None,
    tilt: float = 0.0,
    side: str = "r",
    axis=None,
) -> tuple:
    """(position, yaw, pitch) to draw a held item at (body frame; the item's +z is turned
    by the yaw about +y, then the pitch about +x). Level items sit upright at the hand
    (in ``both`` hands: between them, held by the sides). A knife or chopsticks point on
    down the forearm - or at ``aim``, if the pose aims them. A cigarette sticks out across
    the forearm, forwards and away from the body, turned ``tilt`` degrees about the
    forearm (rolled in the fingers)."""
    how = ITEMS.get(name, (None, "level"))[1]
    x, y, z = hand
    if how in ("level", "hang"):
        grip = TWO_HANDED_GRIP.get(name, 0.02) if both else 0.02
        if both and how == "level" and tilt:
            # (held up in both hands, tipped towards the face by the pose: a phone - resting
            # ``-grip`` off the hands along the way its face looks, up and back)
            a = math.radians(tilt)
            up, back = math.cos(a), math.sin(a)
            return (x, y - grip * up, z + grip * back), 0.0, -tilt
        if axis is not None and not both:
            # held round the middle and tipped with the hand (a can at the lips): its up
            # along ``axis``, its middle at the hand (its origin is 2 cm under that, upright)
            ax, ay, az = (float(c) for c in axis)
            pitch = math.degrees(math.acos(max(-1.0, min(1.0, ay))))
            yaw = math.degrees(math.atan2(ax, az)) if pitch > 1e-3 else 0.0
            s = math.sin(math.radians(pitch))
            k = 0.02
            dx, dy, dz = (
                math.sin(math.radians(yaw)) * s * k,
                ay * k,
                math.cos(math.radians(yaw)) * s * k,
            )
            return (x - dx, y - dy, z - dz), yaw, pitch
        return (x, y - grip, z), 0.0, 0.0
    fx, fy, fz = (float(c) for c in forearm)
    if aim is not None:
        dx, dy, dz = aim[0] - x, aim[1] - y, aim[2] - z
    elif how == "outward":
        dx, dy, dz = outward(side, (fx, fy, fz), tilt)
    else:
        dx, dy, dz = fx, fy, fz
    yaw = math.degrees(math.atan2(dx, dz))
    pitch = -math.degrees(math.atan2(dy, math.hypot(dx, dz)))
    return (x, y, z), yaw, pitch


# Things that lie differently from how they are held (a knife on its side)
LYING: dict[str, Drawer] = {
    "knife": _knife_lying,
    # (a glass stands on its base: a held one is drawn from its middle)
    "beer": lambda ctx, t: _pint_on(ctx, True),
    "glass": lambda ctx, t: _pint_on(ctx, False),
}


def draw_loose_items(ctx: DrawContext, items: Iterable, textures: TextureMap) -> None:
    """Things lying about (``x, y, z`` the middle of the underside, ``yaw`` about +y)."""
    for item in items:
        ctx.push()
        ctx.translate(item.x, item.y, item.z)
        if item.yaw:
            ctx.rotate(item.yaw, 0.0, 1.0, 0.0)
        lying = LYING.get(item.name)
        if lying is not None:
            lying(ctx, textures)
            ctx.emission(None)
            ctx.color(1.0, 1.0, 1.0)
        else:
            draw_item(ctx, item.name, textures)
        ctx.pop()
