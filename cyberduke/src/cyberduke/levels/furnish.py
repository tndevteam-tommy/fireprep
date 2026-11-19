"""Furnishings for the people's homes: the things that make a flat somebody's.

Number 16 (``levels.apartment``) and every ``FlatPlan`` flat (``levels.flats``) furnish
themselves with these: a window with a view (and blinds or curtains), pictures and
posters in frames, shelves with books and jars and whatever their owner collects, kitchen
cabinets with doors and handles, a worktop with a real sink let into it (a basin you can
see down into, a tap), a toilet with a bowl, a seat and a lid, a basin on a pedestal,
plants, rugs, lamps - and food: on the counter, on the shelves, on the coffee table.

Everything is built through a ``Builder`` in a frame of its own: a ``FlatPlan``'s (``u``
along the building's face, ``v`` into it) or the world's (``WorldFrame``: ``u`` is x,
``v`` is z - number 16 is laid out in world coordinates). Every box stays axis-aligned in
the world; a ``Wall`` places things on a wall face by how far along it they are (``s``),
how high (``y``) and how far out from it (``d``).

The rules the levels are held to hold here too: nothing shares a plane with anything
facing the same way (``render.audit`` - decals stand a few millimetres proud, the pieces
of a thing meet edge to edge or face to face), whatever is solid is inside a collider,
and the decorations are ``transparent`` (they block nobody's view).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from cyberduke.core.geometry import AABB
from cyberduke.world.props import Material, Prop

DECAL = frozenset({"transparent"})
RGB = tuple[float, float, float]

WHITE = Material(None, (0.86, 0.86, 0.82), specular=(0.5, 0.5, 0.5), shininess=40.0)
PORCELAIN = Material(None, (0.9, 0.9, 0.88), specular=(0.9, 0.9, 0.9), shininess=70.0)
CHROME = Material("metal", (0.75, 0.77, 0.8), specular=(1.0, 1.0, 1.0), shininess=90.0)
STEEL = Material("metal", (0.55, 0.57, 0.6), specular=(0.8, 0.8, 0.8), shininess=60.0)
DARK = Material("metal", (0.12, 0.12, 0.14))
BLACK = Material(None, (0.05, 0.05, 0.06))
WATER = Material(None, (0.18, 0.24, 0.28), specular=(1.0, 1.0, 1.0), shininess=100.0)
SOIL = Material("soil", (0.5, 0.4, 0.3))
LEAF = Material(None, (0.22, 0.5, 0.2))
LEAF_DARK = Material(None, (0.12, 0.34, 0.14))
GLASS = Material(None, (0.35, 0.45, 0.5), specular=(1.0, 1.0, 1.0), shininess=100.0)


def lamp_glow(rgb: RGB = (1.0, 0.8, 0.5), strength: float = 1.2) -> Material:
    return Material(None, rgb, emission=tuple(c * strength for c in rgb))


# ============================================================================ frames
class Frame(Protocol):
    def world(self, u: float, v: float) -> tuple[float, float]: ...

    def box(self, u0: float, u1: float, v0: float, v1: float, y0: float, y1: float) -> AABB: ...


class WorldFrame:
    """``u`` is x, ``v`` is z."""

    def world(self, u: float, v: float) -> tuple[float, float]:
        return u, v

    def box(self, u0: float, u1: float, v0: float, v1: float, y0: float, y1: float) -> AABB:
        return AABB(min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1), y0, y1)


class Builder:
    """Collects the props and colliders of a home, laid out in ``frame``."""

    def __init__(self, frame: Frame) -> None:
        self.plan = frame
        self.props: list[Prop] = []
        self.colliders: list[AABB] = []

    def add(
        self, u0, u1, v0, v1, y0, y1, mat: Material, label: str, solid=False, tags=frozenset()
    ) -> None:
        """A box from (u0, v0, y0) to (u1, v1, y1); ``solid`` also makes it a collider."""
        b = self.plan.box(u0, u1, v0, v1, y0, y1)
        self.props.append(
            Prop.box(
                (b.min_x + b.max_x) / 2.0, (y0 + y1) / 2.0, (b.min_z + b.max_z) / 2.0,
                b.max_x - b.min_x, y1 - y0, b.max_z - b.min_z, mat, label=label, tags=tags,
                solid=solid,
            )
        )  # fmt: skip
        if solid:
            self.colliders.append(b)

    def collide(self, u0, u1, v0, v1, y0, y1) -> None:
        self.colliders.append(self.plan.box(u0, u1, v0, v1, y0, y1))

    def cylinder(
        self, u, v, y, r_bot, r_top, h, mat: Material, label: str, slices=10, tags=frozenset()
    ) -> None:
        """Upright, rising from ``y``."""
        x, z = self.plan.world(u, v)
        self.props.append(
            Prop.cylinder(x, y, z, r_bot, r_top, h, mat, slices=slices, label=label, tags=tags)
        )

    def lying(
        self, u0, u1, v0, v1, y, r, mat: Material, label: str, along: str = "u", slices=10,
        tags=frozenset(),
    ) -> None:  # fmt: skip
        """A cylinder of radius ``r`` on its side (a roll of paper, a bottle), its underside
        at ``y``, lying ``along`` the frame's "u" or "v" across the box it fills."""
        b = self.plan.box(u0, u1, v0, v1, y, y + 2 * r)
        cx, cz = (b.min_x + b.max_x) / 2.0, (b.min_z + b.max_z) / 2.0
        (x0, z0), (x1, z1) = self.plan.world(0.0, 0.0), self.plan.world(
            *((1.0, 0.0) if along == "u" else (0.0, 1.0))
        )
        if abs(x1 - x0) > abs(z1 - z0):  # along x: turned about z, rising to +x
            p = Prop.cylinder(b.min_x, y + r, cz, r, r, b.max_x - b.min_x, mat, slices=slices,
                              rotate=((-90.0, 0.0, 0.0, 1.0),), label=label, tags=tags)  # fmt: skip
        else:  # along z: turned about x, rising to +z
            p = Prop.cylinder(cx, y + r, b.min_z, r, r, b.max_z - b.min_z, mat, slices=slices,
                              rotate=((90.0, 1.0, 0.0, 0.0),), label=label, tags=tags)  # fmt: skip
        self.props.append(p)

    def prop(self, p: Prop) -> None:
        self.props.append(p)

    def wall(self, axis: str, at: float, out: float) -> Wall:
        return Wall(self, axis, at, out)


@dataclass(frozen=True)
class Wall:
    """A wall face in a builder's frame: the plane ``at`` - a ``u`` for ``axis`` "u" (the
    wall runs along v), a ``v`` for "v" (it runs along u) - and ``out``, +1 or -1: the way
    into the room. Things on it are placed by ``s`` (along it), ``y`` (up) and ``d`` (out
    from its face)."""

    b: Builder
    axis: str
    at: float
    out: float

    def _n(self, d0: float, d1: float) -> tuple[float, float]:
        a, c = self.at + self.out * d0, self.at + self.out * d1
        return min(a, c), max(a, c)

    def box(self, s0, s1, y0, y1, d0, d1, mat, label, solid=False, tags=DECAL) -> None:
        n0, n1 = self._n(d0, d1)
        if self.axis == "u":
            self.b.add(n0, n1, s0, s1, y0, y1, mat, label, solid=solid, tags=tags)
        else:
            self.b.add(s0, s1, n0, n1, y0, y1, mat, label, solid=solid, tags=tags)

    def collide(self, s0, s1, y0, y1, d0, d1) -> None:
        n0, n1 = self._n(d0, d1)
        if self.axis == "u":
            self.b.collide(n0, n1, s0, s1, y0, y1)
        else:
            self.b.collide(s0, s1, n0, n1, y0, y1)

    def uv(self, s: float, d: float) -> tuple[float, float]:
        n = self.at + self.out * d
        return (n, s) if self.axis == "u" else (s, n)

    def cylinder(self, s, d, y, r_bot, r_top, h, mat, label, slices=10, tags=DECAL) -> None:
        u, v = self.uv(s, d)
        self.b.cylinder(u, v, y, r_bot, r_top, h, mat, label, slices=slices, tags=tags)

    def disc(self, s, y, r, d0, d1, mat, label, slices=16, tags=DECAL) -> None:
        """A disc flat against the wall (a dartboard, a clock): its middle at (``s``,
        ``y``), ``d0``..``d1`` out from the wall."""
        n0, n1 = self._n(d0, d1)
        if self.axis == "u":
            self.b.lying(n0, n1, s - r, s + r, y - r, r, mat, label, along="u", slices=slices, tags=tags)
        else:
            self.b.lying(s - r, s + r, n0, n1, y - r, r, mat, label, along="v", slices=slices, tags=tags)

    def lying(self, s0, s1, d0, d1, y, r, mat, label, slices=10, tags=DECAL) -> None:
        """A cylinder on its side along the wall (s0..s1), ``d0``..``d1`` out from it."""
        n0, n1 = self._n(d0, d1)
        if self.axis == "u":  # (the wall runs along v)
            self.b.lying(n0, n1, s0, s1, y, r, mat, label, along="v", slices=slices, tags=tags)
        else:
            self.b.lying(s0, s1, n0, n1, y, r, mat, label, along="u", slices=slices, tags=tags)


# ============================================================================ on the walls
FRAME_WOOD = Material("wood", (0.35, 0.24, 0.16))
FRAME_BLACK = Material(None, (0.08, 0.08, 0.09), specular=(0.3, 0.3, 0.3), shininess=30.0)
FRAME_GOLD = Material("metal", (0.8, 0.62, 0.25), specular=(1.0, 0.9, 0.6), shininess=60.0)


def picture(
    w: Wall, s: float, y: float, width: float, height: float, texture: str, label: str,
    frame: Material | None = FRAME_BLACK, d: float = 0.0,
) -> None:  # fmt: skip
    """A picture on the wall, its middle at (``s``, ``y``): framed (the bars 2.2 cm proud
    of the wall, the print set back inside them) or, with no ``frame``, a poster taped
    straight onto it (3 mm proud) - ``d`` further out if it hangs on something."""
    art = Material(texture, (1.0, 1.0, 1.0))
    s0, s1, y0, y1 = s - width / 2.0, s + width / 2.0, y - height / 2.0, y + height / 2.0
    if frame is None:
        w.box(s0, s1, y0, y1, d + 0.002, d + 0.005, art, label)
        for ps, py in ((s0 + 0.03, y1 - 0.03), (s1 - 0.03, y1 - 0.03)):
            w.box(ps - 0.012, ps + 0.012, py - 0.012, py + 0.012, d + 0.005, d + 0.007,
                  Material(None, (0.85, 0.82, 0.7)), f"Tape on the {label.lower()}")  # fmt: skip
        return
    t = min(0.035, width / 8.0)
    for a, c in ((y0, y0 + t), (y1 - t, y1)):  # bottom and top bars: the full width
        w.box(s0, s1, a, c, d + 0.004, d + 0.026, frame, f"{label} frame")
    for a, c in ((s0, s0 + t), (s1 - t, s1)):  # the sides between them
        w.box(a, c, y0 + t, y1 - t, d + 0.004, d + 0.026, frame, f"{label} frame")
    w.box(s0 + t, s1 - t, y0 + t, y1 - t, d + 0.004, d + 0.012, art, label)


def window(
    w: Wall, s: float, y0: float, y1: float, width: float, view: str, frame: Material,
    cover: str = "blinds", cover_color: RGB = (0.85, 0.82, 0.75), drawn: float = 0.35,
    label: str = "Window",
) -> None:  # fmt: skip
    """A window: a frame and a cross bar, the glass showing the view outside (``view``:
    "city" or "alley" - a screen the renderer keeps up to the time of day), a sill, and
    venetian ``blinds`` let down ``drawn`` of the way, ``curtains`` either side, or a
    ``roller`` blind."""
    s0, s1 = s - width / 2.0, s + width / 2.0
    t = 0.05
    pane = Material(f"view_{view}_night", (1.0, 1.0, 1.0), emission=(0.62, 0.62, 0.66))
    w.box(s0 + t, s1 - t, y0 + t, y1 - t, 0.002, 0.006, pane, f"{label}: the view outside",
          tags=frozenset({"transparent", "screen", f"view:{view}"}))  # fmt: skip
    for a, c in ((y0, y0 + t), (y1 - t, y1)):
        w.box(s0, s1, a, c, 0.002, 0.045, frame, f"{label} frame")
    for a, c in ((s0, s0 + t), (s1 - t, s1)):
        w.box(a, c, y0 + t, y1 - t, 0.002, 0.045, frame, f"{label} frame")
    mid = (y0 + y1) / 2.0
    w.box(s0 + t, s1 - t, mid - 0.015, mid + 0.015, 0.006, 0.03, frame, f"{label} transom")
    w.box(s - 0.015, s + 0.015, y0 + t, mid - 0.015, 0.006, 0.03, frame, f"{label} mullion")
    w.box(s - 0.015, s + 0.015, mid + 0.015, y1 - t, 0.006, 0.03, frame, f"{label} mullion")
    w.box(s0 - 0.06, s1 + 0.06, y0 - 0.045, y0, 0.0, 0.16, frame, f"{label} sill")
    cloth = Material("cloth", cover_color)
    if cover == "blinds":  # slats from the top, a hand's breadth apart, down to ``drawn``
        bottom = y1 - t - (y1 - y0 - 2 * t) * drawn
        w.box(s0 + 0.02, s1 - 0.02, y1 - t - 0.04, y1 - t, 0.05, 0.09,
              Material("metal", cover_color), f"{label} blind head")  # fmt: skip
        y = y1 - t - 0.08
        while y > bottom:
            w.box(s0 + 0.04, s1 - 0.04, y - 0.012, y, 0.055, 0.085,
                  Material(None, cover_color), f"{label} blind slat")  # fmt: skip
            y -= 0.045
        for cs in (s0 + 0.15, s1 - 0.15):
            w.box(cs - 0.003, cs + 0.003, y, y1 - t - 0.04, 0.068, 0.072, DARK,
                  f"{label} blind cord")  # fmt: skip
    elif cover == "roller":
        bottom = y1 - t - (y1 - y0 - 2 * t) * drawn
        w.box(s0 + 0.02, s1 - 0.02, bottom, y1 - t, 0.05, 0.056, cloth, f"{label} roller blind")
        w.box(s0, s1, y1 - t, y1 + 0.03, 0.046, 0.1, Material(None, cover_color),
              f"{label} blind roll")  # fmt: skip
    else:  # curtains, either side, on a rail
        w.box(s0 - 0.2, s1 + 0.2, y1 + 0.08, y1 + 0.11, 0.05, 0.08, DARK, f"{label} curtain rail")
        for a, c in ((s0 - 0.18, s0 + 0.12), (s1 - 0.12, s1 + 0.18)):
            w.box(a, c, y0 - 0.12, y1 + 0.08, 0.06, 0.1, cloth, f"{label} curtain")


def shelf(
    w: Wall, s0: float, s1: float, y: float, mat: Material, depth: float = 0.24,
    label: str = "Wall shelf", brackets: Material = DARK,
) -> None:  # fmt: skip
    """A board on two brackets, its top ``y`` up (things stand on it at ``y``)."""
    w.box(s0, s1, y - 0.025, y, 0.0, depth, mat, label)
    for bs in (s0 + 0.08, s1 - 0.08):
        w.box(bs - 0.012, bs + 0.012, y - 0.2, y - 0.025, 0.0, depth * 0.8, brackets,
              f"{label} bracket")  # fmt: skip


BOOK_COLOURS: tuple[RGB, ...] = (
    (0.55, 0.12, 0.12), (0.12, 0.22, 0.45), (0.16, 0.38, 0.2), (0.7, 0.6, 0.22),
    (0.35, 0.15, 0.4), (0.78, 0.76, 0.7), (0.1, 0.1, 0.12), (0.66, 0.34, 0.12),
    (0.22, 0.45, 0.5), (0.85, 0.3, 0.5),
)  # fmt: skip


def books(
    w: Wall, s0: float, s1: float, y: float, rng: random.Random, depth: float = 0.2,
    label: str = "Book", colours: Sequence[RGB] = BOOK_COLOURS, tall: float = 0.26,
) -> None:  # fmt: skip
    """A row of books standing on a shelf (``y``), spine out, from ``s0`` to ``s1`` - every
    one its own height, thickness and colour, a gap here and there."""
    s = s0
    while True:
        width = rng.uniform(0.025, 0.055)
        if s + width > s1:
            break
        if rng.random() < 0.06:
            s += width * 1.5
            continue
        height = tall * rng.uniform(0.68, 1.0)
        deep = depth * rng.uniform(0.78, 1.0)
        col = rng.choice(colours)
        tone = rng.uniform(0.8, 1.1)
        w.box(s, s + width - 0.002, y, y + height, 0.01, 0.01 + deep,
              Material(None, tuple(min(1.0, c * tone) for c in col)), label)  # fmt: skip
        s += width


def jar(w: Wall, s: float, d: float, y: float, r: float, h: float, rgb: RGB, label: str) -> None:
    """A jar (or a tin): the body, a lid."""
    w.cylinder(s, d, y, r, r, h, Material(None, rgb, specular=(0.6, 0.6, 0.6), shininess=50.0),
               label, slices=10)  # fmt: skip
    w.cylinder(s, d, y + h, r * 1.04, r * 1.04, 0.015, Material(None, (0.7, 0.7, 0.72)),
               f"{label} lid", slices=10)  # fmt: skip


def bottle(w: Wall, s: float, d: float, y: float, rgb: RGB, label: str, h: float = 0.26) -> None:
    glass = Material(None, rgb, specular=(1.0, 1.0, 1.0), shininess=90.0)
    w.cylinder(s, d, y, 0.035, 0.035, h * 0.68, glass, label, slices=10)
    w.cylinder(s, d, y + h * 0.68, 0.035, 0.012, h * 0.14, glass, f"{label} shoulder", slices=10)
    w.cylinder(s, d, y + h * 0.82, 0.012, 0.012, h * 0.18, glass, f"{label} neck", slices=8)


def can(w: Wall, s: float, d: float, y: float, label: str = "Can") -> None:
    w.cylinder(s, d, y, 0.033, 0.033, 0.12, Material("soda_can", (1.0, 1.0, 1.0)), label, 10)


def box_of(
    w: Wall, s: float, d: float, y: float, size: tuple[float, float, float], mat: Material,
    label: str,
) -> None:  # fmt: skip
    """A box standing on a surface: ``size`` is (along the wall, up, out)."""
    sw, sh, sd = size
    w.box(s - sw / 2.0, s + sw / 2.0, y, y + sh, d - sd / 2.0, d + sd / 2.0, mat, label)


def plant(
    w: Wall, s: float, d: float, y: float, size: float, rng: random.Random,
    pot: RGB = (0.72, 0.38, 0.24), label: str = "Potted plant",
) -> None:  # fmt: skip
    """A plant in a pot standing at (``s``, ``d``) on whatever is at ``y``: a tapering
    pot, soil, and a head of leaves - a stack of leafy cones, or (now and then) a fan of
    long blades."""
    r = 0.09 * size
    w.cylinder(s, d, y, r * 0.8, r, 0.16 * size, Material(None, pot), f"{label} pot", slices=10)
    w.cylinder(s, d, y + 0.16 * size, r * 0.94, r * 0.94, 0.006, SOIL, f"{label} soil", slices=10)
    top = y + 0.166 * size
    if rng.random() < 0.35:  # a snake plant: upright blades
        for k in range(5):
            a = k * 2.0 * math.pi / 5 + rng.uniform(-0.3, 0.3)
            du, dd = math.cos(a) * r * 0.45, math.sin(a) * r * 0.45
            u, v = w.uv(s + du, d + dd)
            x, z = w.b.plan.world(u, v)
            h = size * rng.uniform(0.28, 0.45)
            w.b.prop(Prop.box(x, top + h / 2.0, z, 0.03 * size, h, 0.008, LEAF_DARK,
                              rotate=((math.degrees(a), 0.0, 1.0, 0.0), (rng.uniform(-10, 10), 1.0, 0.0, 0.0)),
                              label=f"{label} leaf", tags=DECAL))  # fmt: skip
        return
    for k in range(3):
        rr = r * (1.9 - 0.45 * k) * rng.uniform(0.9, 1.1)
        w.cylinder(s, d, top + k * 0.09 * size, rr, rr * 0.25, 0.16 * size,
                   LEAF if k % 2 == 0 else LEAF_DARK, f"{label} leaves", slices=7)  # fmt: skip


def rug(b: Builder, u0, u1, v0, v1, texture: str, rgb: RGB = (1.0, 1.0, 1.0)) -> None:
    """A rug on the floor (its top 6 mm up: above the floor, never level with it)."""
    b.add(u0, u1, v0, v1, 0.002, 0.008, Material(texture, rgb), "Rug", tags=DECAL)


def hanging_lamp(b: Builder, u: float, v: float, ceiling: float, shade: RGB, drop: float = 0.55) -> None:
    """A pendant shade on a flex, glowing underneath."""
    b.add(u - 0.006, u + 0.006, v - 0.006, v + 0.006, ceiling - drop + 0.14, ceiling - 0.001,
          DARK, "Lamp flex", tags=DECAL)  # fmt: skip
    b.cylinder(u, v, ceiling - drop, 0.2, 0.07, 0.14, Material(None, shade), "Lamp shade", 12, DECAL)
    b.cylinder(u, v, ceiling - drop - 0.004, 0.15, 0.15, 0.004, lamp_glow(), "Lamp glow", 12, DECAL)


def floor_lamp(b: Builder, u: float, v: float, shade: RGB) -> None:
    """A standard lamp: a base, a pole, a glowing shade."""
    b.cylinder(u, v, 0.0, 0.14, 0.14, 0.03, DARK, "Floor lamp base", 12, DECAL)
    b.cylinder(u, v, 0.03, 0.014, 0.014, 1.37, DARK, "Floor lamp pole", 8, DECAL)
    b.cylinder(u, v, 1.4, 0.2, 0.13, 0.24, lamp_glow(shade, 0.9), "Floor lamp shade", 12, DECAL)


def fairy_lights(w: Wall, s0: float, s1: float, y: float, sag: float = 0.12) -> None:
    """A string of little lights looped along the wall."""
    n = max(4, int((s1 - s0) / 0.12))
    colours = ((1.0, 0.85, 0.5), (1.0, 0.4, 0.7), (0.4, 0.9, 1.0))
    for k in range(n + 1):
        t = k / n
        s = s0 + (s1 - s0) * t
        yy = y - sag * 4.0 * t * (1.0 - t)
        c = colours[k % 3]
        w.box(s - 0.01, s + 0.01, yy - 0.01, yy + 0.01, 0.01, 0.03, lamp_glow(c, 1.6),
              "Fairy light")  # fmt: skip


# ============================================================================ kitchens
def cabinet_fronts(
    w: Wall, s0: float, s1: float, y0: float, y1: float, face: Material, handle: Material,
    drawers: bool = True, label: str = "Cabinet",
) -> None:  # fmt: skip
    """Doors (and a row of drawers above them) on the front of a counter or a wall
    cabinet whose face is ``w`` (the doors 1.2 cm proud of it), each with its handle."""
    width = s1 - s0
    n = max(1, round(width / 0.48))
    dw = width / n
    top = y1
    if drawers:
        top = y1 - 0.16
        for k in range(n):
            a = s0 + k * dw
            w.box(a + 0.006, a + dw - 0.006, top + 0.006, y1 - 0.006, 0.0, 0.012, face,
                  f"{label} drawer")  # fmt: skip
            w.box(a + dw / 2.0 - 0.07, a + dw / 2.0 + 0.07, y1 - 0.06, y1 - 0.045, 0.012, 0.03,
                  handle, f"{label} drawer handle")  # fmt: skip
    for k in range(n):
        a = s0 + k * dw
        w.box(a + 0.006, a + dw - 0.006, y0 + 0.006, top - 0.006, 0.0, 0.012, face,
              f"{label} door")  # fmt: skip
        hs = a + dw - 0.05 if k % 2 == 0 else a + 0.05
        hy = _handle_height(y0, top)
        w.box(hs - 0.008, hs + 0.008, hy - 0.06, hy + 0.06, 0.012, 0.032, handle,
              f"{label} door handle")  # fmt: skip


def _handle_height(y0: float, top: float) -> float:
    """Where a cupboard door's handle goes: half way up a tall door (a wardrobe's), near
    the top of a low one, near the bottom of one up on the wall."""
    if top - y0 > 1.2:
        return (y0 + top) / 2.0
    return top - 0.09 if y0 < 0.5 else y0 + 0.08


def wall_cabinet(
    w: Wall, s0: float, s1: float, y0: float, y1: float, body: Material, face: Material,
    handle: Material, depth: float = 0.32, label: str = "Wall cabinet",
) -> None:  # fmt: skip
    """A cupboard on the wall (over a counter): its carcass ``depth`` deep and the doors on
    its front."""
    w.box(s0, s1, y0, y1, 0.0, depth, body, label)
    front = Wall(w.b, w.axis, w.at + w.out * depth, w.out)
    cabinet_fronts(front, s0, s1, y0, y1, face, handle, drawers=False, label=label)


def worktop_with_sink(
    b: Builder, u0: float, u1: float, v0: float, v1: float, y0: float, y1: float,
    hole: tuple[float, float, float, float], top: Material, label: str = "Counter top",
) -> None:  # fmt: skip
    """A worktop slab (``y0``..``y1``) with a sink let into it: the slab in four pieces
    round the ``hole`` (u0, u1, v0, v1), a stainless basin lining it - the rim 5 mm
    proud of the top, the bottom most of the slab's depth down - and a drain."""
    hu0, hu1, hv0, hv1 = hole
    b.add(u0, u1, v0, hv0, y0, y1, top, label)
    b.add(u0, u1, hv1, v1, y0, y1, top, label)
    b.add(u0, hu0, hv0, hv1, y0, y1, top, label)
    b.add(hu1, u1, hv0, hv1, y0, y1, top, label)
    t = 0.006
    lip = y1 + 0.005
    floor = y0 + 0.008
    b.add(hu0, hu0 + t, hv0, hv1, floor, lip, STEEL, "Sink basin")
    b.add(hu1 - t, hu1, hv0, hv1, floor, lip, STEEL, "Sink basin")
    b.add(hu0 + t, hu1 - t, hv0, hv0 + t, floor, lip, STEEL, "Sink basin")
    b.add(hu0 + t, hu1 - t, hv1 - t, hv1, floor, lip, STEEL, "Sink basin")
    b.add(hu0 + t, hu1 - t, hv0 + t, hv1 - t, y0 + 0.002, floor,
          Material("metal", (0.42, 0.44, 0.47), specular=(0.9, 0.9, 0.9), shininess=70.0),
          "Sink basin (bottom)")  # fmt: skip
    b.cylinder((hu0 + hu1) / 2.0, (hv0 + hv1) / 2.0, floor, 0.025, 0.025, 0.002, BLACK,
               "Sink drain", slices=10)  # fmt: skip


def tap(w: Wall, s: float, d: float, y: float, reach: float = 0.16, label: str = "Tap") -> None:
    """A mixer tap on the worktop (``y``) behind a sink, its spout reaching ``reach`` out
    over the basin, a lever on top."""
    w.cylinder(s, d, y, 0.022, 0.02, 0.24, CHROME, label, slices=10)
    w.box(s - 0.012, s + 0.012, y + 0.2, y + 0.226, d, d + reach, CHROME, f"{label} spout")
    w.cylinder(s, d + reach - 0.012, y + 0.178, 0.011, 0.011, 0.022, CHROME, f"{label} nozzle", 8)
    w.box(s - 0.008, s + 0.008, y + 0.24, y + 0.252, d - 0.01, d + 0.07, CHROME, f"{label} lever")


def extractor_hood(w: Wall, s0: float, s1: float, y0: float, depth: float = 0.45) -> None:
    """A steel hood over the hob, its duct going up to the ceiling."""
    w.box(s0, s1, y0, y0 + 0.14, 0.0, depth, STEEL, "Extractor hood")
    w.box(s0 + 0.015, s1 - 0.015, y0 - 0.004, y0, 0.02, depth - 0.02,
          Material(None, (0.2, 0.2, 0.22)), "Extractor hood filter")  # fmt: skip
    mid = (s0 + s1) / 2.0
    w.box(mid - 0.12, mid + 0.12, y0 + 0.14, y0 + 0.62, 0.0, 0.22, STEEL, "Extractor duct")


def fridge_details(w: Wall, s0: float, s1: float, height: float, notes: bool = True) -> None:
    """On a fridge's front (``w`` is its face): the freezer's seam, two handles and - on the
    door - the notes, menus and photos everybody sticks there."""
    w.box(s0 + 0.01, s1 - 0.01, height * 0.64, height * 0.64 + 0.008, 0.0, 0.003, DARK,
          "Fridge door seam")  # fmt: skip
    for y0, y1 in ((height * 0.7, height * 0.92), (height * 0.38, height * 0.6)):
        w.box(s1 - 0.07, s1 - 0.045, y0, y1, 0.0, 0.035, CHROME, "Fridge handle")
    if notes:
        w.box(s0 + 0.05, s1 - 0.12, height * 0.22, height * 0.58, 0.003, 0.005,
              Material("notes", (1.0, 1.0, 1.0)), "Notes on the fridge")  # fmt: skip


def fridge(
    face: Wall, s0: float, s1: float, height: float, depth: float, enamel: Material,
    label: str = "Fridge", notes: bool = True, rust: bool = False,
    magnets: Sequence[RGB] = ((0.9, 0.2, 0.2), (0.2, 0.5, 0.9), (0.95, 0.8, 0.1)),
) -> None:  # fmt: skip
    """A fridge-freezer whose front is ``face`` (its body behind it, ``depth`` deep): the
    cabinet on a dark plinth, a freezer door over the fridge door (3 cm proud, a seam
    between them), long chrome bar handles on stand-offs, a badge, magnets and notes -
    and, ``rust``, rust coming through the enamel at its foot."""
    face.box(s0, s1, 0.07, height, -depth, 0.0, enamel, label, tags=frozenset())
    face.box(s0 + 0.03, s1 - 0.03, 0.0, 0.07, -depth + 0.02, -0.03, DARK, f"{label} plinth",
             tags=frozenset())  # fmt: skip
    split = 0.07 + (height - 0.07) * 0.66
    doors = ((0.08, split - 0.004, "door"), (split + 0.004, height - 0.01, "freezer door"))
    for y0, y1, what in doors:
        face.box(s0 + 0.008, s1 - 0.008, y0, y1, 0.0, 0.03, enamel, f"{label} {what}",
                 tags=frozenset())  # fmt: skip
    grip = s1 - 0.075  # the handles on the side away from the hinges
    for y0, y1 in ((0.55, split - 0.1), (split + 0.08, height - 0.12)):
        face.box(grip - 0.014, grip + 0.014, y0, y1, 0.06, 0.085, CHROME, f"{label} handle")
        for y in (y0 + 0.03, y1 - 0.03):
            face.box(grip - 0.01, grip + 0.01, y - 0.015, y + 0.015, 0.03, 0.06, CHROME,
                     f"{label} handle stand-off")  # fmt: skip
    face.box(s0 + 0.06, s0 + 0.2, height - 0.1, height - 0.07, 0.03, 0.034, CHROME, f"{label} badge")
    if notes:
        face.box(s0 + 0.05, grip - 0.06, split * 0.42, split * 0.9, 0.03, 0.033,
                 Material("notes", (1.0, 1.0, 1.0)), f"Notes on the {label.lower()}")  # fmt: skip
    for k, rgb in enumerate(magnets):
        face.disc(s0 + 0.08 + 0.07 * k, split + 0.12 + 0.05 * (k % 2), 0.018, 0.03, 0.042,
                  Material(None, rgb), "Fridge magnet", slices=10)  # fmt: skip
    if rust:
        face.box(s0 + 0.02, s0 + 0.22, 0.09, 0.2, 0.03, 0.0315, Material(None, (0.42, 0.22, 0.1)),
                 f"Rust on the {label.lower()}")  # fmt: skip


def microwave(
    w: Wall, s: float, y: float, body: Material, width: float = 0.48, depth: float = 0.34,
    height: float = 0.28, label: str = "Microwave", display: str = "00:00",
) -> None:  # fmt: skip
    """A microwave on a worktop (``y``) against ``w``, its middle ``s`` along it: the
    cabinet on four little feet, the door with its dark window (and the light's sheen on
    it) and a handle, the control strip with a lit clock, a dial and buttons."""
    s0, s1 = s - width / 2.0, s + width / 2.0
    front = depth + 0.02
    y0 = y + 0.012
    for fs in (s0 + 0.04, s1 - 0.04):
        for fd in (0.08, depth - 0.02):
            w.box(fs - 0.015, fs + 0.015, y, y0, fd - 0.015, fd + 0.015, DARK, f"{label} foot",
                  tags=frozenset())  # fmt: skip
    w.box(s0, s1, y0, y0 + height, 0.02, front, body, label, tags=frozenset())
    panel = s1 - width * 0.28  # the control strip on the right, the door on the left
    w.box(s0 + 0.012, panel - 0.006, y0 + 0.015, y0 + height - 0.015, front, front + 0.012, body,
          f"{label} door", tags=frozenset())  # fmt: skip
    w.box(s0 + 0.04, panel - 0.05, y0 + 0.045, y0 + height - 0.045, front + 0.012, front + 0.015,
          Material(None, (0.05, 0.06, 0.07), specular=(1.0, 1.0, 1.0), shininess=100.0),
          f"{label} window", tags=frozenset())  # fmt: skip
    w.box(s0 + 0.07, s0 + 0.1, y0 + 0.07, y0 + height - 0.07, front + 0.015, front + 0.0185,
          Material(None, (0.25, 0.28, 0.3)), f"{label} window sheen", tags=frozenset())  # fmt: skip
    w.box(panel - 0.035, panel - 0.015, y0 + 0.05, y0 + height - 0.05, front + 0.012, front + 0.035,
          DARK, f"{label} door handle", tags=frozenset())  # fmt: skip
    w.box(panel + 0.004, s1 - 0.012, y0 + 0.015, y0 + height - 0.015, front, front + 0.008,
          Material(None, (0.1, 0.1, 0.11)), f"{label} control panel", tags=frozenset())  # fmt: skip
    mid = (panel + s1) / 2.0
    w.box(mid - 0.045, mid + 0.045, y0 + height - 0.07, y0 + height - 0.035, front + 0.008,
          front + 0.01, Material(f"display_{display.replace(':', '')}", (1.0, 1.0, 1.0),
          emission=(0.2, 1.2, 0.4)), f"{label} clock: {display}")  # fmt: skip
    w.disc(mid, y0 + 0.1, 0.03, front + 0.008, front + 0.03, Material("metal", (0.7, 0.72, 0.75)),
           f"{label} dial", slices=12)  # fmt: skip
    for k in range(2):
        for j in range(2):
            bs, by = mid - 0.025 + 0.05 * j, y0 + height - 0.1 - 0.03 * k
            w.box(bs - 0.012, bs + 0.012, by - 0.008, by + 0.008, front + 0.008, front + 0.012,
                  Material(None, (0.45, 0.45, 0.48)), f"{label} button")  # fmt: skip
    for k in range(5):  # vent slots on the top
        vs = s0 + 0.08 + 0.03 * k
        w.box(vs - 0.008, vs + 0.008, y0 + height, y0 + height + 0.003, 0.1, 0.24,
              Material(None, (0.15, 0.15, 0.17)), f"{label} vent", tags=frozenset())  # fmt: skip


# ============================================================================ bathrooms
def toilet(
    b: Builder, wall: Wall, s: float, seat: RGB = (0.9, 0.9, 0.88), lid_up: bool = True,
    label: str = "Toilet",
) -> None:  # fmt: skip
    """A toilet against ``wall`` at ``s``: a cistern on the wall with its flush button, a
    flared pedestal, the bowl - its rim and seat, the dark water down in it - and the lid
    up against the cistern (or down). Collider included."""
    seat_mat = Material(None, seat, specular=(0.6, 0.6, 0.6), shininess=50.0)
    wall.box(s - 0.2, s + 0.2, 0.4, 0.82, 0.0, 0.17, PORCELAIN, f"{label} cistern", tags=frozenset())
    wall.box(s - 0.21, s + 0.21, 0.82, 0.845, 0.0, 0.185, PORCELAIN, f"{label} cistern lid",
             tags=frozenset())  # fmt: skip
    wall.cylinder(s, 0.09, 0.845, 0.025, 0.025, 0.008, CHROME, f"{label} flush button",
                  slices=10, tags=frozenset())  # fmt: skip
    bowl = 0.36  # the middle of the bowl, out from the wall
    wall.box(s - 0.06, s + 0.06, 0.28, 0.4, 0.16, 0.24, PORCELAIN, f"{label} neck", tags=frozenset())
    wall.cylinder(s, bowl, 0.0, 0.13, 0.1, 0.26, PORCELAIN, f"{label} pedestal", slices=14,
                  tags=frozenset())  # fmt: skip
    wall.cylinder(s, bowl, 0.26, 0.12, 0.19, 0.13, PORCELAIN, f"{label} bowl", slices=16,
                  tags=frozenset())  # fmt: skip
    wall.cylinder(s, bowl, 0.39, 0.2, 0.2, 0.022, seat_mat, f"{label} seat", slices=16,
                  tags=frozenset())  # fmt: skip
    wall.cylinder(s, bowl, 0.412, 0.13, 0.13, 0.003, WATER, f"{label} water (down in the bowl)",
                  slices=16, tags=frozenset())  # fmt: skip
    if lid_up:
        wall.box(s - 0.19, s + 0.19, 0.415, 0.82, 0.185, 0.205, seat_mat, f"{label} lid (up)",
                 tags=frozenset())  # fmt: skip
    else:
        wall.cylinder(s, bowl, 0.415, 0.2, 0.2, 0.018, seat_mat, f"{label} lid", slices=16,
                      tags=frozenset())  # fmt: skip
    wall.collide(s - 0.23, s + 0.23, 0.0, 0.85, 0.0, bowl + 0.22)


MIRROR = Material(None, (0.55, 0.6, 0.65), specular=(1.0, 1.0, 1.0), shininess=100.0)


def pedestal_basin(
    b: Builder, wall: Wall, s: float, label: str = "Wash basin", mirror: bool = True,
    cabinet: Material | None = None, glass: Material = MIRROR, mirror_label: str = "Mirror",
) -> None:  # fmt: skip
    """A basin on a pedestal against ``wall``: the bowl (its inside a shade darker), a tap
    at the back, a mirror over it - on a little cabinet, if ``cabinet`` - and a shelf of
    toiletries. Collider included."""
    out = 0.24
    wall.cylinder(s, out, 0.0, 0.1, 0.075, 0.7, PORCELAIN, f"{label} pedestal", slices=12,
                  tags=frozenset())  # fmt: skip
    wall.cylinder(s, out, 0.7, 0.12, 0.23, 0.14, PORCELAIN, f"{label}", slices=16, tags=frozenset())
    wall.cylinder(s, out, 0.84, 0.185, 0.185, 0.003,
                  Material(None, (0.72, 0.74, 0.74), specular=(0.9, 0.9, 0.9), shininess=80.0),
                  f"{label} (the bowl)", slices=16, tags=frozenset())  # fmt: skip
    tap(wall, s, 0.03, 0.84, reach=0.13, label=f"{label} tap")  # (behind the bowl's inside)
    if mirror:
        if cabinet is not None:
            wall.box(s - 0.3, s + 0.3, 1.25, 1.85, 0.0, 0.12, cabinet, "Mirror cabinet")
            wall.box(s - 0.27, s + 0.27, 1.28, 1.82, 0.12, 0.125, glass,
                     f"{mirror_label} (the cabinet door)")  # fmt: skip
        else:
            wall.box(s - 0.24, s + 0.24, 1.25, 1.78, 0.008, 0.018, glass, mirror_label)
        wall.box(s - 0.28, s + 0.28, 1.1, 1.12, 0.0, 0.1, Material(None, (0.85, 0.85, 0.85)),
                 "Glass shelf under the mirror")  # fmt: skip
        bottle(wall, s - 0.18, 0.05, 1.12, (0.3, 0.6, 0.9), "Shampoo bottle", h=0.16)
        wall.cylinder(s + 0.02, 0.05, 1.12, 0.025, 0.028, 0.09, Material(None, (0.9, 0.9, 0.9)),
                      "Tooth mug", slices=10)  # fmt: skip
        wall.box(s + 0.01, s + 0.03, 1.15, 1.3, 0.04, 0.06, Material(None, (0.2, 0.6, 0.9)), "Toothbrush")
        bottle(wall, s + 0.18, 0.05, 1.12, (0.9, 0.4, 0.5), "Lotion bottle", h=0.14)
    wall.collide(s - 0.25, s + 0.25, 0.0, 0.9, 0.0, out + 0.23)


def towel_rail(w: Wall, s0: float, s1: float, y: float, towel: RGB) -> None:
    """A rail on the wall with a towel folded over it."""
    w.box(s0, s1, y - 0.012, y + 0.012, 0.06, 0.08, CHROME, "Towel rail")
    for bs in (s0 + 0.02, s1 - 0.02):
        w.box(bs - 0.01, bs + 0.01, y - 0.02, y + 0.02, 0.0, 0.06, CHROME, "Towel rail bracket")
    w.box(s0 + 0.06, s1 - 0.06, y - 0.36, y + 0.016, 0.04, 0.1, Material("cloth", towel), "Towel")


def toilet_roll(w: Wall, s: float, y: float) -> None:
    w.box(s - 0.07, s + 0.07, y + 0.06, y + 0.075, 0.0, 0.1, CHROME, "Toilet roll holder")
    w.lying(s - 0.05, s + 0.05, 0.015, 0.125, y - 0.05, 0.055, PORCELAIN, "Toilet roll", slices=10)


# ============================================================================ food
def fruit_bowl(w: Wall, s: float, d: float, y: float, rng: random.Random) -> None:
    w.cylinder(s, d, y, 0.07, 0.13, 0.07, Material(None, (0.7, 0.55, 0.35)), "Fruit bowl", 12)
    fruit = ((0.85, 0.15, 0.1), (0.95, 0.75, 0.1), (0.95, 0.55, 0.1), (0.4, 0.75, 0.2))
    turn = rng.uniform(0.0, math.pi / 2.0)
    for k in range(4):  # (each a little higher than the last: heaped, no two level)
        a = k * math.pi / 2 + turn
        w.cylinder(s + math.cos(a) * 0.055, d + math.sin(a) * 0.055, y + 0.04 + 0.006 * k, 0.036,
                   0.03, 0.06, Material(None, fruit[k]), "Fruit", slices=8)  # fmt: skip


def counter_food(w: Wall, s0: float, s1: float, d: float, y: float, rng: random.Random) -> None:
    """Things on a kitchen worktop (``w`` is the wall behind it, ``d`` how far out along it
    they stand): picked from a cereal box, a loaf, a kettle, a bottle of oil, a jar of
    pickles, instant noodles, a bowl of fruit, a pile of takeaway boxes."""
    choices = ["cereal", "bread", "kettle", "oil", "jar", "noodles", "fruit", "takeaway"]
    rng.shuffle(choices)
    s = s0 + 0.05
    for name in choices:
        if s > s1 - 0.12:
            break
        if name == "cereal":
            box_of(w, s + 0.1, d, y, (0.2, 0.3, 0.07), Material("cereal", (1.0, 1.0, 1.0)), "Cereal box")
            s += 0.26
        elif name == "bread":
            box_of(w, s + 0.12, d, y, (0.24, 0.1, 0.12), Material(None, (0.78, 0.58, 0.32)), "Loaf of bread")
            s += 0.3
        elif name == "kettle":
            w.cylinder(s + 0.09, d, y, 0.09, 0.075, 0.2, Material("metal", (0.8, 0.8, 0.82),
                       specular=(1.0, 1.0, 1.0), shininess=80.0), "Kettle", slices=12)  # fmt: skip
            w.box(s + 0.07, s + 0.11, y + 0.12, y + 0.24, d - 0.1, d - 0.08, DARK, "Kettle handle")
            s += 0.22
        elif name == "oil":
            bottle(w, s + 0.05, d, y, (0.75, 0.6, 0.15), "Bottle of oil", h=0.3)
            bottle(w, s + 0.13, d + 0.02, y, (0.25, 0.15, 0.08), "Bottle of soy sauce", h=0.22)
            s += 0.2
        elif name == "jar":
            jar(w, s + 0.05, d, y, 0.045, 0.14, (0.5, 0.6, 0.25), "Jar of pickles")
            jar(w, s + 0.15, d - 0.02, y, 0.04, 0.1, (0.85, 0.5, 0.1), "Jar of peanut butter")
            s += 0.22
        elif name == "noodles":
            for k in range(3):
                w.cylinder(s + 0.05, d, y + k * 0.095, 0.045, 0.05, 0.095,
                           Material(None, (0.95, 0.92, 0.85)), "Instant noodle cup", slices=10)  # fmt: skip
            w.cylinder(s + 0.05, d, y + 0.285, 0.052, 0.052, 0.004, Material(None, (0.9, 0.2, 0.1)),
                       "Instant noodle lid", slices=10)  # fmt: skip
            s += 0.14
        elif name == "fruit":
            fruit_bowl(w, s + 0.14, d, y, rng)
            s += 0.3
        elif name == "takeaway":
            for k in range(2):
                box_of(w, s + 0.07, d, y + k * 0.07, (0.12, 0.07, 0.1),
                       Material(None, (0.92, 0.9, 0.85)), "Takeaway box")  # fmt: skip
            s += 0.18


def pantry_shelf(w: Wall, s0: float, s1: float, y: float, rng: random.Random) -> None:
    """A shelf of food: tins, jars, a bag of rice, a bottle or two."""
    s = s0 + 0.05
    while s < s1 - 0.1:
        roll = rng.random()
        if roll < 0.35:
            jar(w, s + 0.04, 0.12, y, 0.04, rng.uniform(0.08, 0.14),
                rng.choice(((0.8, 0.2, 0.15), (0.3, 0.5, 0.2), (0.85, 0.6, 0.2), (0.6, 0.6, 0.65))),
                "Tin of food")  # fmt: skip
            s += 0.1
        elif roll < 0.55:
            box_of(w, s + 0.08, 0.12, y, (0.16, 0.22, 0.1), Material(None, (0.92, 0.9, 0.82)), "Bag of rice")
            s += 0.19
        elif roll < 0.75:
            bottle(w, s + 0.04, 0.12, y, rng.choice(((0.25, 0.15, 0.08), (0.2, 0.5, 0.25))),
                   "Bottle", h=rng.uniform(0.2, 0.28))  # fmt: skip
            s += 0.1
        else:
            can(w, s + 0.04, 0.12, y, "Can of soda")
            can(w, s + 0.11, 0.12, y, "Can of soda")
            s += 0.16


__all__ = [
    "Builder",
    "Wall",
    "WorldFrame",
    "books",
    "cabinet_fronts",
    "counter_food",
    "picture",
    "shelf",
    "toilet",
    "window",
    "worktop_with_sink",
]
