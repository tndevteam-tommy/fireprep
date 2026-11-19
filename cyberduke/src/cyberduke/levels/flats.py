"""The homes of the people of the street: small ground-floor flats built from a plan.

Number 16 (``levels.apartment``) is drawn by hand; everybody else's home - Nix's number
17 across the alley from where he smokes, and six more round the corner on the main
street - is a ``FlatPlan`` built here. What number 16 taught is built in:

* every flat is a shell of its own behind the building's face: a thin front wall with
  the tenement's brick outside and plaster inside, a real hinged front door in it (its
  owner unlocks it, goes in and locks it behind them; from inside a latch opens it),
  party walls, a back wall, floors and a ceiling;
* **no two surfaces share a plane** (``render.audit``): the floors stop at the front
  wall (the pavement outside is a slab of its own), the ceiling stops inside the brick
  and under the storey line, decals stand a few millimetres proud of what they are on;
* a front room with a kitchenette along one wall - cupboards with doors and drawers, a
  sink let into the worktop, a hood over the hob, food about - a sofa facing a TV (a
  real one: ``render.television`` shows what is on), a coffee table, a desk with a
  computer under a window that looks out on the street (and that the street sees lit);
  behind a partition a bedroom and a tiled bathroom (a toilet with its bowl, seat and
  lid, a shower, a basin on a pedestal under a mirror);
* no two flats alike: each has a ``Style`` of its own (wallpaper or panelling, the floor,
  the tiles, the fabrics, the TV, blinds or curtains) and the things of whoever lives
  there (``_LIVES``: the junk dealer's shelves of salvage, the tinkerer's tool board, the
  stroller's plants and prints, the chef's knives and pans...);
* the owner's places (``Place``: "h31.entry", "h31.kitchen", "h31.bed"...) and straight
  routes between them through the doorways (``LevelData.routes``), walk regions for the
  netrunner, and a ``PrivateArea`` - the owner minds who is in there.

A flat is laid out in its own frame: ``u`` along the building's face, ``v`` from the face
into the building. ``facing`` ("+x" or "+z") is the way its front door faces out; the
frame is turned into the world to match, so every box stays axis-aligned.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from cyberduke.core.geometry import AABB
from cyberduke.levels import furnish as f
from cyberduke.levels.furnish import Builder, Wall
from cyberduke.world.level import DoorData, PanelData, Place, PrivateArea, WalkRegion
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop

FRONT = 0.3  # the front wall's thickness (brick outside, plaster inside)
SIDE = 0.2  # party walls and the back wall
CEILING_Y = 2.7
STOREY = 3.0  # the ground floor's top: above it the building's own face
DOOR_W, DOOR_H = 1.0, 2.2
LIVING_DEPTH = 3.1  # from the inside of the front wall to the partition
PARTITION = 0.12
BED_LENGTH = 2.2  # head board to foot (a tall sleeper's heels stay on it)
DOORWAY_H = 2.05  # the inner doorways (no doors, long gone)
TILE_D = 0.006  # bathroom and splashback tiles stand this far proud of the wall
DESK_TOP = 0.76  # the computer desk (``ai.alley.FLAT_DESK_TOP``)
DESK_DEPTH = 0.6
CHAIR_AT = 0.86  # the chair's middle, out from the front wall (a seated typist's hips)
CHAIR_SEAT = 0.48  # ``ai.alley.CHAIR_SEAT``
WINDOW_Y = (1.05, 2.05)  # the window in the front wall, over the desk

BRICK = Material("brick", (0.12, 0.12, 0.13))
DARK = Material("metal", (0.12, 0.12, 0.14))
WHITE = Material(None, (0.8, 0.8, 0.76))
CLUTTER = frozenset({"clutter", "transparent"})
DECAL = frozenset({"transparent"})
TILE = Material("tile", (0.78, 0.84, 0.82))
BULB = Material(None, (1.0, 0.9, 0.6), emission=(2.2, 1.8, 1.0))
WINDOW_LIT = Material(None, (0.9, 0.7, 0.4), emission=(0.55, 0.4, 0.2))  # (seen from the street)

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Style:
    """How a flat is done up: no two alike. Surfaces are (texture, tint) pairs."""

    walls: tuple[str, RGB]  # the front room and the bedroom
    floor: tuple[str, RGB]
    bath_tiles: tuple[str, RGB]
    bath_floor: tuple[str, RGB]
    sofa: tuple[str, RGB]
    cover: RGB  # the bed cover, cushions, the coat on the hook
    door: RGB
    accent: RGB  # a neon strip, the fridge magnets - somebody lives here
    kitchen: tuple[str, RGB]  # the cupboard doors
    worktop: tuple[str, RGB]
    window: str  # "blinds", "roller" or "curtains"
    window_cloth: RGB
    tv: str  # "flat" (a flat screen on a stand) or "crt" (a deep old tube set)
    rug: str | None  # a rug's texture under the coffee table (or none)
    shade: RGB  # the lampshade over the ceiling bulb
    towel: RGB


STYLES = (
    # 0: sage stripes, parquet, white subway tiles, a mustard sofa (the stroller's)
    Style(("wallpaper_stripes", (0.86, 0.95, 0.88)), ("parquet", (1.0, 1.0, 1.0)),
          ("tile_subway", (1.0, 1.0, 1.0)), ("tile_hex", (0.9, 0.95, 0.9)),
          ("cloth", (0.78, 0.58, 0.22)), (0.3, 0.5, 0.4), (0.2, 0.35, 0.25), (0.6, 1.0, 0.4),
          ("wood", (0.95, 0.9, 0.8)), ("wood", (0.8, 0.65, 0.45)), "curtains", (0.92, 0.88, 0.76),
          "flat", "rug_persian", (0.95, 0.85, 0.6), (0.8, 0.4, 0.3)),
    # 1: blue-grey breeze block, grey lino, old green tiles, black leather (Nix's)
    Style(("breeze_block", (0.62, 0.66, 0.78)), ("concrete", (0.75, 0.75, 0.78)),
          ("tile", (0.7, 0.85, 0.72)), ("linoleum", (0.45, 0.5, 0.48)),
          ("leather", (0.1, 0.1, 0.11)), (0.45, 0.1, 0.12), (0.25, 0.3, 0.38), (1.0, 0.2, 0.25),
          ("plaster", (0.55, 0.6, 0.68)), ("metal", (0.5, 0.52, 0.5)), "blinds", (0.22, 0.22, 0.25),
          "crt", None, (0.8, 0.2, 0.2), (0.15, 0.15, 0.18)),
    # 2: wood panelling, brown carpet, beige tiles, an olive sofa (the junk dealer's)
    Style(("wood_panel", (0.95, 0.88, 0.8)), ("carpet", (0.85, 0.6, 0.42)),
          ("tile", (0.9, 0.82, 0.7)), ("tile", (0.75, 0.7, 0.6)),
          ("cloth", (0.52, 0.55, 0.28)), (0.6, 0.5, 0.3), (0.45, 0.18, 0.15), (1.0, 0.7, 0.2),
          ("wood", (0.7, 0.5, 0.32)), ("wood", (0.55, 0.42, 0.3)), "curtains", (0.85, 0.45, 0.15),
          "crt", "rug_shag", (0.95, 0.6, 0.2), (0.9, 0.6, 0.2)),
    # 3: lilac plaster, black-and-white lino, hexagon tiles, a tan sofa (the tinkerer's)
    Style(("plaster", (0.66, 0.6, 0.7)), ("checker", (1.0, 1.0, 1.0)),
          ("tile_hex", (0.85, 0.9, 1.0)), ("checker", (0.9, 0.9, 0.9)),
          ("leather", (0.55, 0.36, 0.2)), (0.3, 0.25, 0.5), (0.3, 0.3, 0.3), (0.4, 1.0, 0.6),
          ("plaster", (0.35, 0.75, 0.68)), ("metal", (0.45, 0.46, 0.48)), "blinds", (0.9, 0.9, 0.88),
          "flat", "rug_stripes", (0.5, 0.9, 0.7), (0.3, 0.7, 0.6)),
    # 4: off-white plaster, pale parquet, white tiles, a grey sofa - tidy (the office worker's)
    Style(("plaster", (0.9, 0.9, 0.86)), ("parquet", (1.1, 1.05, 1.0)),
          ("tile_subway", (0.95, 0.97, 1.0)), ("tile", (0.85, 0.86, 0.88)),
          ("cloth", (0.5, 0.52, 0.56)), (0.25, 0.3, 0.45), (0.2, 0.25, 0.35), (0.2, 0.9, 1.0),
          ("plaster", (0.95, 0.95, 0.95)), ("metal", (0.62, 0.64, 0.66)), "blinds", (0.85, 0.86, 0.88),
          "flat", None, (0.92, 0.92, 0.9), (0.9, 0.9, 0.92)),
    # 5: seventies geometric paper, blue lino, blue tiles, a red sofa (the courier's)
    Style(("wallpaper_geo", (1.0, 1.0, 1.0)), ("linoleum", (0.55, 0.62, 0.75)),
          ("tile", (0.6, 0.75, 0.95)), ("tile_hex", (0.7, 0.8, 1.0)),
          ("cloth", (0.7, 0.16, 0.14)), (0.2, 0.3, 0.6), (0.5, 0.32, 0.2), (1.0, 0.3, 0.8),
          ("wood", (0.85, 0.55, 0.3)), ("wood", (0.75, 0.6, 0.4)), "roller", (0.3, 0.45, 0.7),
          "crt", "rug_stripes", (1.0, 0.55, 0.3), (0.9, 0.3, 0.3)),
    # 6: old damask, dark boards, hexagon tiles, oxblood leather (the sushi chef's)
    Style(("wallpaper_damask", (1.0, 1.0, 1.0)), ("wood", (0.55, 0.42, 0.32)),
          ("tile_hex", (1.0, 1.0, 1.0)), ("tile", (0.6, 0.6, 0.58)),
          ("leather", (0.42, 0.09, 0.08)), (0.55, 0.15, 0.15), (0.5, 0.12, 0.1), (1.0, 0.55, 0.15),
          ("wood", (0.45, 0.3, 0.2)), ("wood", (0.35, 0.25, 0.18)), "curtains", (0.6, 0.1, 0.12),
          "flat", "rug_persian", (1.0, 0.5, 0.25), (0.95, 0.95, 0.9)),
)  # fmt: skip


def mat(surface: tuple[str, RGB], **kw) -> Material:
    """A style's surface as a material. "leather" (the texture is a jacket's, near black)
    is a smooth tinted finish with a sheen, so a tan or an oxblood sofa looks it."""
    texture, rgb = surface
    if texture == "leather":
        return Material(None, rgb, specular=(0.45, 0.42, 0.4), shininess=28.0)
    return Material(texture, rgb, **kw)


@dataclass(frozen=True)
class FlatPlan:
    """One flat. ``u0``..``u1``: its inside, along the building's face (the party walls
    are outside that); ``door_u``: the middle of the front door along the face;
    ``face``: the building face's plane (x for a "+x" flat, z for a "+z" one)."""

    number: str
    owner_role: str
    facing: str  # "+x" or "+z": the way out of the front door
    face: float
    u0: float
    u1: float
    door_u: float
    depth: float = 6.8  # from the face to the back of the back wall
    style: int = 0

    @property
    def prefix(self) -> str:
        return f"h{self.number}"

    @property
    def door_id(self) -> str:
        return f"door{self.number}"

    @property
    def area_id(self) -> str:
        return f"flat{self.number}"

    @property
    def back(self) -> float:
        """v of the inside face of the back wall."""
        return self.depth - SIDE

    @property
    def split(self) -> float:
        """u of the bedroom | bathroom partition."""
        return self.u0 + 0.6 * (self.u1 - self.u0)

    # ------------------------------------------------------------------ frame
    def world(self, u: float, v: float) -> tuple[float, float]:
        """(x, z) of a point of the flat's frame."""
        if self.facing == "+x":
            return self.face - v, u
        return u, self.face - v

    def box(self, u0: float, u1: float, v0: float, v1: float, y0: float, y1: float) -> AABB:
        (xa, za), (xb, zb) = self.world(u0, v0), self.world(u1, v1)
        return AABB(min(xa, xb), max(xa, xb), min(za, zb), max(za, zb), y0, y1)

    def yaw(self, local: float) -> float:
        """World yaw (NPC convention: 0 faces +z) of a local heading - 0 into the flat
        (+v), 90 along +u."""
        du, dv = math.sin(math.radians(local)), math.cos(math.radians(local))
        (x0, z0), (x1, z1) = self.world(0.0, 0.0), self.world(du, dv)
        return math.degrees(math.atan2(x1 - x0, z1 - z0))

    @property
    def inside(self) -> str:
        return "-x" if self.facing == "+x" else "-z"




def _prop(
    plan: FlatPlan, u0, u1, v0, v1, y0, y1, mat: Material, label: str, tags=frozenset(), solid=False
) -> Prop:
    b = plan.box(u0, u1, v0, v1, y0, y1)
    return Prop.box(
        (b.min_x + b.max_x) / 2.0, (y0 + y1) / 2.0, (b.min_z + b.max_z) / 2.0,
        b.max_x - b.min_x, y1 - y0, b.max_z - b.min_z, mat, label=label, tags=tags, solid=solid,
    )  # fmt: skip


def style_of(plan: FlatPlan) -> Style:
    return STYLES[plan.style % len(STYLES)]


def _rng(plan: FlatPlan) -> random.Random:
    return random.Random(sum(ord(c) * 31**k for k, c in enumerate(plan.number)))


# ============================================================================ the layout
def sofa_span(plan: FlatPlan) -> tuple[float, float]:
    """Where the sofa stands along the partition (its arms clear of the bedroom door)."""
    return plan.u0 + 0.15, plan.u0 + 1.75


def tv_u(plan: FlatPlan) -> float:
    """The middle of the TV (facing the sofa)."""
    return sum(sofa_span(plan)) / 2.0


def desk_span(plan: FlatPlan) -> tuple[float, float] | None:
    """Where the desk stands against the front wall - between the TV cabinet and the
    swing of the front door, up to 1.2 m wide - with the window over it (None if there
    is no room for one)."""
    left = tv_u(plan) + 0.7
    right = plan.door_u - DOOR_W / 2.0 - 0.15
    width = min(1.2, right - left)
    if width < 0.6:
        return None
    mid = (left + right) / 2.0
    return mid - width / 2.0, mid + width / 2.0


def window_width(desk: tuple[float, float]) -> float:
    """How wide the window over a desk is."""
    return min(1.1, desk[1] - desk[0] + 0.1)


@dataclass
class Rooms:
    """The flat's walls (``furnish.Wall``: things go on them) and where things stand."""

    b: Builder
    plan: FlatPlan
    style: Style
    rng: random.Random
    front: Wall  # the inside of the front wall (facing into the front room)
    left: Wall  # the front room's u0 wall
    right: Wall  # the front room's u1 wall (the kitchenette)
    partition: Wall  # the partition, front room side
    bed_partition: Wall  # ...and bedroom side
    bed_back: Wall  # the bedroom's back wall
    bed_left: Wall
    bath_left: Wall  # the bathroom's walls, at the face of their tiles
    bath_right: Wall
    bath_back: Wall
    bath_partition: Wall
    desk: tuple[float, float] | None

    @property
    def split(self) -> float:
        return self.plan.split


class _Builder(Builder):
    """(a ``furnish.Builder`` in a flat's frame)"""

    def __init__(self, plan: FlatPlan) -> None:
        super().__init__(plan)


# ============================================================================ the build
def build(plan: FlatPlan) -> tuple[list[Prop], list[AABB]]:
    """Every prop and collider of one flat."""
    b = _Builder(plan)
    f0 = FRONT
    v0 = FRONT + LIVING_DEPTH
    v1 = v0 + PARTITION
    s0 = plan.split + PARTITION
    rooms = Rooms(
        b, plan, style_of(plan), _rng(plan),
        front=b.wall("v", f0, 1.0), left=b.wall("u", plan.u0, 1.0),
        right=b.wall("u", plan.u1, -1.0), partition=b.wall("v", v0, -1.0),
        bed_partition=b.wall("v", v1, 1.0), bed_back=b.wall("v", plan.back, -1.0),
        bed_left=b.wall("u", plan.u0, 1.0),
        bath_left=b.wall("u", s0 + TILE_D, 1.0), bath_right=b.wall("u", plan.u1 - TILE_D, -1.0),
        bath_back=b.wall("v", plan.back - TILE_D, -1.0),
        bath_partition=b.wall("v", v1 + TILE_D, 1.0),
        desk=desk_span(plan),
    )  # fmt: skip
    _shell(rooms)
    _kitchen(rooms)
    _front_room(rooms)
    _bedroom(rooms)
    _bathroom(rooms)
    _LIVES.get(plan.owner_role, _a_life)(rooms)
    return b.props, b.colliders


def _shell(r: Rooms) -> None:
    b, p, st = r.b, r.plan, r.style
    name = f"number {p.number}"
    walls = mat(st.walls)
    u0, u1, back = p.u0, p.u1, p.back
    d0, d1 = p.door_u - DOOR_W / 2.0, p.door_u + DOOR_W / 2.0
    # the front wall: brick outside, plaster (or paper) inside (4 mm back, so their faces
    # never meet), the doorway and a lintel over it
    for a, c in ((u0 - SIDE, d0), (d1, u1 + SIDE)):
        b.add(a, c, 0.0, FRONT - 0.01, 0.0, STOREY, BRICK, f"Front wall of {name}", solid=True)
        b.add(max(a, u0), min(c, u1), FRONT - 0.01, FRONT, 0.0, CEILING_Y, walls, f"Plaster inside the front wall of {name}")
    b.add(d0, d1, 0.0, FRONT, DOOR_H, STOREY, BRICK, f"Lintel over the door of {name}", solid=True)
    # the door frame (outside) and the number beside it
    for e in (d0, d1):  # (the posts stop under the head: overlapping it, they z-fought)
        b.add(e - 0.06, e + 0.06, -0.1, 0.02, 0.0, DOOR_H + 0.04, DARK, "Door frame post")
    b.add(d0 - 0.06, d1 + 0.06, -0.1, 0.02, DOOR_H + 0.04, DOOR_H + 0.16, DARK, "Door frame head")
    b.add(d1 + 0.25, d1 + 0.45, -0.015, 0.0, 1.85, 1.97, Material(None, (0.8, 0.75, 0.5), emission=(0.3, 0.28, 0.15)), f"Door number {p.number}", tags=DECAL)
    # party walls, back wall
    for a, c in ((u0 - SIDE, u0), (u1, u1 + SIDE)):
        b.add(a, c, FRONT, back + SIDE, 0.0, CEILING_Y, walls, f"Party wall of {name}", solid=True)
    b.add(u0, u1, back, back + SIDE, 0.0, CEILING_Y, walls, f"Back wall of {name}", solid=True)
    # the partition between the front room and the back rooms, with two doorways
    v0, v1 = FRONT + LIVING_DEPTH, FRONT + LIVING_DEPTH + PARTITION
    split = p.split
    bed_door = (split - 1.05, split - 0.2)
    bath_door = (split + PARTITION + 0.3, split + PARTITION + 1.1)
    edges = [u0, bed_door[0], bed_door[1], bath_door[0], bath_door[1], u1]
    for a, c in zip(edges[0::2], edges[1::2], strict=True):
        b.add(a, c, v0, v1, 0.0, CEILING_Y, walls, f"Partition wall of {name}", solid=True)
    for a, c in (bed_door, bath_door):
        b.add(a, c, v0, v1, DOORWAY_H, CEILING_Y, walls, "Partition over a doorway", solid=True)
        for e in (a, c):  # a frame, no door
            b.add(e - 0.03, e + 0.03, v0 - 0.03, v1 + 0.03, 0.0, DOORWAY_H, Material("wood", (0.5, 0.4, 0.3)), "Doorway frame")
    # bedroom | bathroom
    b.add(split, split + PARTITION, v1, back, 0.0, CEILING_Y, walls, "Bathroom wall", solid=True)
    # floors (their tops 2 mm above the street's: never level with the pavement) and the
    # ceiling (inside the brick, under the storey line)
    b.add(u0, u1, FRONT, v0, -0.02, 0.002, mat(st.floor), f"Floor of {name}")
    b.add(u0, split, v0, back, -0.02, 0.002, mat(st.floor), f"Bedroom floor of {name}")
    b.add(split + PARTITION, u1, v0, back, -0.02, 0.002, mat(st.bath_floor), f"Bathroom floor of {name}")
    # the threshold: the doorway's floor, a step of 4 mm (level with the pavement, or
    # with the floor inside, it would share their plane)
    b.add(d0, d1, 0.0, FRONT - 0.005, -0.05, 0.008, Material("concrete", (0.2, 0.2, 0.22)), "Threshold")
    b.add(u0 - SIDE + 0.005, u1 + SIDE - 0.005, FRONT - 0.2, back + SIDE, CEILING_Y, STOREY - 0.01, Material("plaster", (0.36, 0.34, 0.3)), f"Ceiling of {name}")
    # the light: a shade on a flex over the middle of the front room
    f.hanging_lamp(b, (u0 + u1) / 2.0, FRONT + 1.6, CEILING_Y, st.shade)
    # the window over the desk - and the same window seen from the street, lit
    if r.desk is not None:
        a, c = r.desk
        mid, width = (a + c) / 2.0, window_width(r.desk)
        view = "alley" if p.facing == "+x" else "city"
        frame = Material("metal", (0.82, 0.82, 0.8)) if p.style % 2 else Material("wood", (0.55, 0.4, 0.28))
        f.window(r.front, mid, WINDOW_Y[0], WINDOW_Y[1], width, view, frame, cover=st.window,
                 cover_color=st.window_cloth, drawn=0.2 + 0.4 * r.rng.random(), label="Window")  # fmt: skip
        b.add(mid - width / 2.0, mid + width / 2.0, -0.05, 0.0, 1.1, 2.0, WINDOW_LIT, f"Window of number {p.number}")
        b.add(mid - width / 2.0 - 0.075, mid + width / 2.0 + 0.075, -0.16, 0.0, 1.025, 1.075, Material("concrete", (0.1, 0.1, 0.115)), "Window sill (outside)")
        for k in range(5):  # security bars (the lit window shows between them)
            bu = mid - width / 2.0 + width * (k + 0.5) / 5.0
            b.add(bu - 0.01, bu + 0.01, -0.09, -0.07, 1.075, 2.0, DARK, "Window bar", tags=DECAL)


def _kitchen(r: Rooms) -> None:
    """Along the far side wall: a counter of cupboards and drawers with a sink let into
    its worktop and a tap, the hob and a hood over it, cupboards on the wall, a tiled
    splashback, the fridge in the corner with everybody's notes on it; food about."""
    b, p, st, w = r.b, r.plan, r.style, r.right
    u1, fr = p.u1, FRONT
    carcass = Material("wood", (0.3, 0.27, 0.24))
    face = mat(st.kitchen, specular=(0.3, 0.3, 0.3), shininess=20.0)
    handle = f.CHROME if p.style % 2 else Material("metal", (0.15, 0.15, 0.17))
    s0, s1 = fr + 1.05, fr + 2.45  # the counter, along the wall
    b.add(u1 - 0.6, u1, s0, s1, 0.0, 0.8, carcass, "Kitchen counter", solid=True)
    f.worktop_with_sink(b, u1 - 0.62, u1, s0 - 0.02, s1 + 0.02, 0.8, 0.92, (u1 - 0.48, u1 - 0.12, fr + 1.97, fr + 2.33), mat(st.worktop), label="Counter top")
    f.cabinet_fronts(b.wall("u", u1 - 0.6, -1.0), s0, s1, 0.02, 0.78, face, handle, label="Kitchen cupboard")
    b.add(u1 - 0.5, u1 - 0.1, fr + 1.2, fr + 1.65, 0.92, 0.95, DARK, "Hob")
    for dv in (1.31, 1.54):
        b.cylinder(u1 - 0.3, fr + dv, 0.95, 0.07, 0.07, 0.003, Material(None, (0.3, 0.08, 0.05)), "Hob ring", slices=12)
    b.cylinder(u1 - 0.3, fr + 1.31, 0.953, 0.12, 0.12, 0.13, Material("metal", (0.45, 0.46, 0.5)), "Pot on the hob")
    f.tap(w, fr + 2.15, 0.05, 0.92, reach=0.16, label="Kitchen tap")
    w.box(s0, s1, 0.925, 1.495, 0.0, TILE_D, mat(st.bath_tiles), "Splashback tiles")
    f.extractor_hood(w, fr + 1.12, fr + 1.72, 1.62)
    f.wall_cabinet(w, fr + 1.76, s1, 1.5, 2.15, carcass, face, handle, label="Wall cupboard")
    f.counter_food(w, fr + 1.67, fr + 1.96, 0.42, 0.92, r.rng)
    # the fridge in the corner by the partition, the door facing the room
    fridge = Material(None, (0.8, 0.81, 0.78), specular=(0.6, 0.6, 0.6), shininess=40.0)  # (enamel)
    b.add(u1 - 0.7, u1, fr + 2.5, fr + LIVING_DEPTH - 0.02, 0.0, 1.75, fridge, "Fridge", solid=True)
    f.fridge_details(b.wall("u", u1 - 0.7, -1.0), fr + 2.5, fr + LIVING_DEPTH - 0.02, 1.75)
    f.box_of(w, fr + 2.62, 0.35, 1.75, (0.2, 0.3, 0.07), Material("cereal", (1.0, 1.0, 1.0)), "Cereal box")
    f.jar(w, fr + 2.86, 0.3, 1.75, 0.05, 0.16, (0.85, 0.85, 0.8), "Biscuit tin")
    f.shelf(w, fr + 2.52, fr + 3.06, 2.12, carcass, depth=0.26, label="Pantry shelf")
    f.pantry_shelf(w, fr + 2.52, fr + 3.06, 2.12, r.rng)
    b.colliders.append(p.box(u1 - 0.65, u1, fr + 1.0, fr + LIVING_DEPTH, 0.0, 1.75))  # counter, fridge


def _front_room(r: Rooms) -> None:
    b, p, st = r.b, r.plan, r.style
    fr = FRONT
    wood = Material("wood", (0.5, 0.4, 0.3))
    sofa = mat(st.sofa)
    cover = Material("cloth", st.cover)
    # the sofa against the partition, facing the front wall; cushions, a throw
    su0, su1 = sofa_span(p)
    sv1 = fr + LIVING_DEPTH - 0.02
    b.add(su0, su1, sv1 - 0.8, sv1, 0.0, 0.42, sofa, "Sofa", solid=True)
    b.add(su0, su1, sv1 - 0.25, sv1, 0.42, 0.88, sofa, "Sofa back", solid=True)
    for a, c in ((su0 - 0.14, su0), (su1, su1 + 0.14)):
        b.add(a, c, sv1 - 0.8, sv1, 0.0, 0.6, sofa, "Sofa arm", solid=True)
    b.add(su0 + 0.2, su0 + 0.9, sv1 - 0.72, sv1 - 0.3, 0.42, 0.5, cover, "Cushion")
    b.add(su1 - 0.42, su1 - 0.06, sv1 - 0.32, sv1 - 0.25, 0.46, 0.8, cover, "Cushion (against the back)")
    b.add(su0 + 1.0, su0 + 1.5, sv1 - 0.24, sv1 - 0.02, 0.885, 0.9, Material("cloth", st.window_cloth), "Throw over the sofa back")
    # the TV against the front wall, facing the sofa: a flat screen on a stand, or a tube
    tu = tv_u(p)
    b.add(tu - 0.5, tu + 0.5, fr + 0.02, fr + 0.45, 0.0, 0.45, wood, "TV cabinet", solid=True)
    b.add(tu - 0.42, tu - 0.05, fr + 0.38, fr + 0.452, 0.06, 0.4, Material(None, (0.2, 0.17, 0.14)), "TV cabinet door")
    b.add(tu + 0.05, tu + 0.42, fr + 0.38, fr + 0.452, 0.06, 0.4, Material(None, (0.2, 0.17, 0.14)), "TV cabinet door")
    screen_tags = frozenset({"transparent", "screen", f"tv:{p.area_id}"})
    glass = Material("tv_off", (1.0, 1.0, 1.0), emission=(0.85, 0.85, 0.88))
    if st.tv == "crt":
        b.add(tu - 0.33, tu + 0.33, fr + 0.06, fr + 0.42, 0.45, 0.95, Material(None, (0.18, 0.17, 0.16)), "TV (an old tube set)", solid=True)
        b.add(tu - 0.26, tu + 0.26, fr + 0.42, fr + 0.424, 0.5, 0.88, glass, "TV screen", tags=screen_tags)
        for k in range(2):
            b.add(tu + 0.27, tu + 0.31, fr + 0.42, fr + 0.44, 0.8 - 0.08 * k, 0.84 - 0.08 * k, Material(None, (0.6, 0.6, 0.6)), "TV knob")
        for lean in (-1.0, 1.0):
            b.add(tu + lean * 0.06 - 0.005, tu + lean * 0.06 + 0.005, fr + 0.2, fr + 0.21, 0.95, 1.3, f.CHROME, "TV aerial")
    else:
        b.add(tu - 0.08, tu + 0.08, fr + 0.12, fr + 0.28, 0.45, 0.47, DARK, "TV stand")
        b.add(tu - 0.03, tu + 0.03, fr + 0.14, fr + 0.17, 0.47, 0.55, DARK, "TV stand neck")
        b.add(tu - 0.45, tu + 0.45, fr + 0.1, fr + 0.16, 0.55, 1.07, DARK, "TV", solid=True)
        b.add(tu - 0.42, tu + 0.42, fr + 0.16, fr + 0.164, 0.58, 1.04, glass, "TV screen", tags=screen_tags)
    b.add(tu + 0.2, tu + 0.45, fr + 0.1, fr + 0.3, 0.45, 0.5, Material(None, (0.1, 0.1, 0.11)), "Games console")
    b.add(tu + 0.4, tu + 0.42, fr + 0.3, fr + 0.303, 0.462, 0.47, Material(None, (0.2, 0.9, 0.4), emission=(0.2, 1.2, 0.4)), "Console light")
    # the coffee table
    b.add(tu - 0.45, tu + 0.45, fr + 1.0, fr + 1.45, 0.36, 0.4, wood, "Coffee table", solid=True)
    for du, dv in ((-0.4, 1.05), (0.4, 1.05), (-0.4, 1.4), (0.4, 1.4)):
        b.add(tu + du - 0.025, tu + du + 0.025, fr + dv - 0.025, fr + dv + 0.025, 0.0, 0.36, DARK, "Coffee table leg")
    if st.rug is not None:
        f.rug(b, tu - 0.8, tu + 0.8, fr + 0.85, fr + 1.95, st.rug)
    # the neon strip over the sofa (somebody lives here)
    b.add(su0, su1, fr + LIVING_DEPTH - 0.03, fr + LIVING_DEPTH - 0.006, 2.1, 2.14, Material(None, st.accent, emission=tuple(c * 1.2 for c in st.accent)), "Neon strip", tags=DECAL)
    # a coat on the hook by the door
    b.add(p.door_u + DOOR_W / 2.0 + 0.15, p.door_u + DOOR_W / 2.0 + 0.19, fr + 0.05, fr + 0.09, 1.2, 1.7, DARK, "Coat hooks")
    b.add(p.door_u + DOOR_W / 2.0 + 0.08, p.door_u + DOOR_W / 2.0 + 0.28, fr + 0.1, fr + 0.16, 0.9, 1.45, Material("cloth", st.cover), "Coat on a hook")
    # a standard lamp in the corner by the sofa
    f.floor_lamp(b, p.u0 + 0.25, fr + 2.12, st.shade)
    # the desk under the window, its chair and its computer
    if r.desk is not None:
        _desk(r)
    b.colliders += [
        p.box(su0 - 0.16, su1 + 0.16, sv1 - 0.82, fr + LIVING_DEPTH, 0.0, 0.9),  # sofa
        p.box(tu - 0.52, tu + 0.52, fr, fr + 0.47, 0.0, 1.07 if st.tv == "flat" else 0.95),  # TV
        p.box(tu - 0.47, tu + 0.47, fr + 0.98, fr + 1.47, 0.0, 0.4),  # coffee table
    ]


def _desk(r: Rooms) -> None:
    b, p = r.b, r.plan
    a, c = r.desk  # type: ignore[misc]
    mid = (a + c) / 2.0
    fr = FRONT
    top = Material("wood", (0.4, 0.32, 0.25)) if p.style % 3 else Material("metal", (0.3, 0.31, 0.33))
    b.add(a, c, fr + 0.01, fr + DESK_DEPTH, DESK_TOP - 0.035, DESK_TOP, top, "Computer desk", solid=True)
    for lu in (a + 0.03, c - 0.03):
        for lv in (fr + 0.06, fr + DESK_DEPTH - 0.04):
            b.add(lu - 0.02, lu + 0.02, lv - 0.02, lv + 0.02, 0.0, DESK_TOP - 0.035, DARK, "Computer desk leg")
    b.colliders.append(p.box(a, c, fr, fr + DESK_DEPTH, 0.0, DESK_TOP))
    # the computer: a laptop, or a monitor (two, if there is room), a keyboard, a mouse,
    # a mug
    if p.owner_role in LAPTOP_PEOPLE:
        _laptop(r, mid - 0.1)
        screens: tuple[float, ...] = ()
    else:
        screens = (mid,) if c - a < 1.0 else (mid - 0.27, mid + 0.27)
    for k, su in enumerate(screens):
        b.add(su - 0.06, su + 0.06, fr + 0.12, fr + 0.22, DESK_TOP, DESK_TOP + 0.015, DARK, "Monitor stand")
        b.add(su - 0.02, su + 0.02, fr + 0.15, fr + 0.18, DESK_TOP + 0.015, DESK_TOP + 0.12, DARK, "Monitor stand neck")
        b.add(su - 0.25, su + 0.25, fr + 0.18, fr + 0.21, DESK_TOP + 0.12, DESK_TOP + 0.43, DARK, "Monitor")
        texture = ("screen_net", "terminal_unlocked", "screen_radar")[(k + p.style) % 3]
        b.add(su - 0.23, su + 0.23, fr + 0.21, fr + 0.213, DESK_TOP + 0.14, DESK_TOP + 0.41, Material(texture, (1.0, 1.0, 1.0), emission=(0.6, 0.6, 0.6)), "Monitor screen", tags=DECAL)
    if screens:
        b.add(mid - 0.22, mid + 0.22, fr + 0.38, fr + 0.52, DESK_TOP, DESK_TOP + 0.02, Material(None, (0.15, 0.15, 0.16)), "Keyboard")
    b.add(mid + 0.3, mid + 0.36, fr + 0.42, fr + 0.52, DESK_TOP, DESK_TOP + 0.02, Material(None, (0.2, 0.2, 0.22)), "Mouse")
    b.cylinder(c - 0.1, fr + 0.3, DESK_TOP, 0.04, 0.04, 0.1, Material(None, (0.85, 0.25, 0.2)), "Mug", slices=10)
    # the chair, pulled out a little (the owner sits on it: ``ai.alley`` "desk")
    cv = fr + CHAIR_AT
    seat = Material("cloth", (0.12, 0.12, 0.14))
    b.cylinder(mid, cv, 0.0, 0.26, 0.26, 0.03, DARK, "Desk chair base", slices=5)
    b.cylinder(mid, cv, 0.03, 0.025, 0.025, CHAIR_SEAT - 0.07, f.CHROME, "Desk chair post", slices=8)
    b.add(mid - 0.23, mid + 0.23, cv - 0.22, cv + 0.2, CHAIR_SEAT - 0.04, CHAIR_SEAT, seat, "Desk chair seat")
    b.add(mid - 0.2, mid + 0.2, cv + 0.2, cv + 0.25, CHAIR_SEAT + 0.05, CHAIR_SEAT + 0.5, seat, "Desk chair back")
    b.add(mid - 0.02, mid + 0.02, cv + 0.2, cv + 0.24, CHAIR_SEAT, CHAIR_SEAT + 0.05, DARK, "Desk chair back post")
    b.colliders.append(p.box(mid - 0.27, mid + 0.27, cv - 0.25, cv + 0.27, 0.0, CHAIR_SEAT + 0.5))


LAPTOP_PEOPLE = frozenset({"stroller", "sushi_chef", "courier"})


def _laptop(r: Rooms, s: float) -> None:
    """A laptop open on the desk (its middle ``s`` along the front wall), the screen lit."""
    w = r.front
    shell = Material("metal", (0.7, 0.72, 0.76), specular=(0.8, 0.8, 0.8), shininess=60.0)
    w.box(s - 0.17, s + 0.17, DESK_TOP, DESK_TOP + 0.018, 0.3, 0.54, shell, "Laptop")
    w.box(s - 0.15, s + 0.15, DESK_TOP + 0.018, DESK_TOP + 0.02, 0.36, 0.48, Material(None, (0.12, 0.12, 0.13)), "Laptop keyboard")
    w.box(s - 0.17, s + 0.17, DESK_TOP + 0.018, DESK_TOP + 0.25, 0.285, 0.3, shell, "Laptop lid")
    w.box(s - 0.155, s + 0.155, DESK_TOP + 0.03, DESK_TOP + 0.235, 0.3, 0.303, Material(("screen_net", "terminal_unlocked", "screen_radar")[r.plan.style % 3], (1.0, 1.0, 1.0), emission=(0.6, 0.6, 0.6)), "Laptop screen")


def _bedroom(r: Rooms) -> None:
    b, p, st = r.b, r.plan, r.style
    u0, back = p.u0, p.back
    wood = Material("wood", (0.45, 0.36, 0.27))
    # the bed against the back wall, head to it
    bu0, bu1 = u0 + 0.1, u0 + 1.5
    bv0 = back - BED_LENGTH
    b.add(bu0, bu1, bv0, back - 0.02, 0.0, 0.3, wood, "Bed frame", solid=True)
    b.add(bu0 + 0.03, bu1 - 0.03, bv0 + 0.03, back - 0.05, 0.3, 0.48, Material("cloth", (0.8, 0.78, 0.74)), "Mattress", solid=True)
    b.add(bu0 + 0.1, bu1 - 0.1, back - 0.5, back - 0.1, 0.48, 0.58, WHITE, "Pillow")
    b.add(bu0 + 0.05, bu1 - 0.05, bv0 + 0.1, back - 0.6, 0.48, 0.53, Material("cloth", st.cover), "Bed cover")
    b.add(bu0 + 0.08, bu1 - 0.08, bv0 + 0.1, bv0 + 0.32, 0.53, 0.55, Material("cloth", st.window_cloth), "Blanket folded at the foot")
    b.add(bu1 + 0.08, bu1 + 0.5, back - 0.5, back - 0.05, 0.0, 0.52, wood, "Nightstand", solid=True)
    b.add(bu1 + 0.1, bu1 + 0.48, back - 0.52, back - 0.5, 0.3, 0.46, Material("wood", (0.35, 0.28, 0.2)), "Nightstand drawer")
    b.cylinder(bu1 + 0.29, back - 0.28, 0.52, 0.05, 0.05, 0.2, DARK, "Bedside lamp", slices=8)
    b.cylinder(bu1 + 0.29, back - 0.28, 0.72, 0.1, 0.06, 0.1, Material(None, (1.0, 0.8, 0.5), emission=(1.1, 0.8, 0.4)), "Bedside lamp shade", slices=10)
    b.add(bu1 + 0.12, bu1 + 0.22, back - 0.46, back - 0.4, 0.52, 0.58, Material(None, (0.08, 0.08, 0.09)), "Alarm clock")
    b.add(bu1 + 0.13, bu1 + 0.21, back - 0.465, back - 0.46, 0.53, 0.57, Material(None, (1.0, 0.2, 0.15), emission=(1.4, 0.2, 0.1)), "Alarm clock face", tags=DECAL)
    # a wardrobe in the corner by the back wall (clear of the doorway), its doors
    wu1 = p.split - 0.02
    b.add(wu1 - 0.6, wu1, back - 0.95, back - 0.02, 0.0, 1.9, wood, "Wardrobe", solid=True)
    f.cabinet_fronts(b.wall("v", back - 0.95, -1.0), wu1 - 0.6, wu1, 0.04, 1.86, Material("wood", (0.5, 0.4, 0.3)), DARK, drawers=False, label="Wardrobe")
    # clothes on the floor, a picture over the bed, a shelf, a rug by the bed
    b.add(bu0 + 0.4, bu0 + 0.85, bv0 - 0.55, bv0 - 0.15, 0.0, 0.04, Material("cloth", st.cover), "Clothes on the floor", tags=CLUTTER)
    f.rug(b, bu1 + 0.06, bu1 + 0.9, back - 1.9, back - 0.62, "rug_stripes" if p.style % 2 else "carpet", (0.9, 0.85, 0.8))
    f.shelf(r.bed_left, back - 1.7, back - 0.9, 1.45, wood, depth=0.2, label="Shelf over the bed")
    f.books(r.bed_left, back - 1.66, back - 1.2, 1.45, r.rng, depth=0.16, tall=0.22)
    f.plant(r.bed_left, back - 1.05, 0.1, 1.45, 0.7, r.rng)
    b.colliders += [
        p.box(bu0 - 0.02, bu1 + 0.52, bv0 - 0.02, back, 0.0, 0.6),
        p.box(wu1 - 0.62, wu1, back - 0.97, back, 0.0, 1.9),
    ]


def _bathroom(r: Rooms) -> None:
    b, p, st = r.b, r.plan, r.style
    u1, back = p.u1, p.back
    v1 = FRONT + LIVING_DEPTH + PARTITION
    s0 = p.split + PARTITION
    tiles = mat(st.bath_tiles)
    bath_door = (s0 + 0.3, s0 + 1.1)
    # tiled to shoulder height (their faces ``TILE_D`` off the walls)
    b.add(s0, s0 + TILE_D, v1, back - TILE_D, 0.0, 1.6, tiles, "Bathroom tiles", tags=DECAL)
    b.add(u1 - TILE_D, u1, v1, back - TILE_D, 0.0, 1.6, tiles, "Bathroom tiles", tags=DECAL)
    b.add(s0, u1, back - TILE_D, back, 0.0, 1.6, tiles, "Bathroom tiles", tags=DECAL)
    for a, c in ((s0 + TILE_D, bath_door[0]), (bath_door[1], u1 - TILE_D)):
        if c - a > 0.02:
            b.add(a, c, v1, v1 + TILE_D, 0.0, 1.6, tiles, "Bathroom tiles", tags=DECAL)
    # the shower in the back corner: a tray, a curtain on a rail, the head on its pipe
    b.add(u1 - 0.85, u1 - 0.02, back - 0.85, back - 0.02, 0.0, 0.08, WHITE, "Shower tray")
    b.cylinder(u1 - 0.43, back - 0.43, 0.08, 0.035, 0.035, 0.002, f.BLACK, "Shower drain", slices=10)
    b.add(u1 - 0.87, u1 - 0.85, back - 0.85, back - 0.3, 0.4, 2.0, Material("cloth", (0.55, 0.7, 0.65)), "Shower curtain")
    b.add(u1 - 0.875, u1 - 0.845, back - 0.87, back - 0.02, 2.02, 2.045, f.CHROME, "Shower curtain rail")
    b.add(u1 - 0.46, u1 - 0.4, back - 0.12, back - 0.02, 1.9, 2.0, Material("metal", (0.45, 0.46, 0.5)), "Shower head")
    r.bath_back.box(u1 - 0.445, u1 - 0.415, 1.2, 1.9, 0.0, 0.02, f.CHROME, "Shower pipe")
    f.bottle(r.bath_back, u1 - 0.12, 0.05, 0.08, (0.9, 0.5, 0.2), "Shampoo bottle", h=0.2)
    # the toilet along the side wall, the basin on a pedestal under a mirror
    f.toilet(b, r.bath_left, back - 0.72, seat=(0.9, 0.9, 0.88) if p.style % 2 else (0.3, 0.22, 0.16), lid_up=p.style % 3 != 0)
    f.toilet_roll(r.bath_left, back - 0.3, 0.62)
    f.pedestal_basin(b, r.bath_right, v1 + 0.42, cabinet=Material("plaster", (0.9, 0.9, 0.88)) if p.style % 2 else None)
    b.add(u1 - 0.92, u1 - 0.5, v1 + 0.15, v1 + 0.7, 0.002, 0.008, Material("carpet", st.towel), "Bath mat", tags=DECAL)
    # a towel on a rail, on the back wall between the toilet and the shower (or by the door)
    ra, rc = s0 + 0.66, u1 - 0.92
    if rc - ra >= 0.3:
        f.towel_rail(r.bath_back, ra, min(rc, ra + 0.45), 1.35, st.towel)
    else:
        f.towel_rail(r.bath_partition, bath_door[1] + 0.04, bath_door[1] + 0.34, 1.35, st.towel)
    f.shelf(r.bath_left, back - 0.95, back - 0.5, 1.3, Material("plaster", (0.9, 0.9, 0.88)), depth=0.16, label="Bathroom shelf")
    r.bath_left.lying(back - 0.92, back - 0.8, 0.02, 0.13, 1.3, 0.055, f.PORCELAIN, "Spare toilet roll")
    f.bottle(r.bath_left, back - 0.6, 0.08, 1.3, (0.3, 0.7, 0.9), "Bottle of bleach", h=0.24)
    r.bath_right.box(v1 + 0.22, v1 + 0.62, 1.92, 1.96, 0.0, 0.04, Material(None, (1.0, 0.95, 0.85), emission=(1.4, 1.3, 1.1)), "Mirror light")
    b.colliders += [p.box(u1 - 0.87, u1, back - 0.87, back, 0.0, 2.0)]


# ============================================================================ their lives
def _tv_wall_top(r: Rooms) -> float:
    """The lowest a picture may hang over the TV (a tube set's aerial reaches higher)."""
    return 1.42 if r.style.tv == "crt" else 1.2


def _between_doorways(r: Rooms) -> tuple[float, float]:
    """The stretch of partition (front room side) between the bedroom and bathroom doors."""
    split = r.plan.split
    return split - 0.2, split + PARTITION + 0.3


def bookcase(r: Rooms, s0: float, s1: float, rows: str = "books", height: float = 1.85) -> None:
    """A bookcase against the front room's u0 wall (``s`` along it): sides, a back, four
    shelves, and on them ``rows`` of books (or ``salvage``, ``records``). Collider included."""
    w = r.left
    case = Material("wood", (0.42, 0.3, 0.2)) if rows == "books" else Material("metal", (0.35, 0.36, 0.38))
    depth = 0.32
    w.box(s0, s1, 0.0, height, 0.0, 0.008, case, "Bookcase back")
    for a, c in ((s0, s0 + 0.02), (s1 - 0.02, s1)):
        w.box(a, c, 0.0, height, 0.008, depth, case, "Bookcase side")
    w.box(s0 + 0.02, s1 - 0.02, height - 0.02, height, 0.008, depth, case, "Bookcase top")
    levels = [0.06 + k * (height - 0.08) / 4.0 for k in range(4)]
    for y in levels:
        w.box(s0 + 0.02, s1 - 0.02, y - 0.02, y, 0.008, depth - 0.01, case, "Bookcase shelf")
    for k, y in enumerate(levels):
        if rows == "books" or (rows == "salvage" and k == 3):
            f.books(w, s0 + 0.03, s1 - 0.03, y, r.rng, depth=0.22, tall=min(0.3, (height - 0.08) / 4.0 - 0.08))
        elif rows == "salvage":
            _salvage(r, w, s0 + 0.04, s1 - 0.04, y)
    w.collide(s0, s1, 0.0, height, 0.0, depth)


def _salvage(r: Rooms, w: Wall, s0: float, s1: float, y: float) -> None:
    """A shelf of the junk dealer's stock: dead decks, boxes with a light still on, coils
    of cable, a monitor with its back off."""
    s = s0
    while s < s1 - 0.12:
        roll = r.rng.random()
        if roll < 0.3:
            f.box_of(w, s + 0.1, 0.14, y, (0.18, 0.08, 0.2), Material("metal", (0.2, 0.21, 0.24)), "Dead cyberdeck")
            w.box(s + 0.03, s + 0.06, y + 0.05, y + 0.065, 0.24, 0.242, f.lamp_glow((0.2, 1.0, 0.6), 1.4), "Deck light")
            s += 0.21
        elif roll < 0.55:
            w.cylinder(s + 0.07, 0.14, y, 0.07, 0.07, 0.05, Material(None, (0.1, 0.1, 0.1)), "Coil of cable", slices=12)
            w.cylinder(s + 0.07, 0.14, y + 0.05, 0.07, 0.07, 0.04, Material(None, (0.7, 0.2, 0.1)), "Coil of cable", slices=12)
            s += 0.16
        elif roll < 0.8:
            f.box_of(w, s + 0.12, 0.14, y, (0.24, 0.18, 0.24), Material("crate", (0.8, 0.8, 0.8)), "Parts box")
            s += 0.26
        else:
            f.box_of(w, s + 0.13, 0.13, y, (0.26, 0.22, 0.18), Material(None, (0.7, 0.68, 0.6)), "Old monitor")
            w.box(s + 0.03, s + 0.23, y + 0.03, y + 0.19, 0.22, 0.222, Material("static", (0.5, 0.5, 0.5)), "Old monitor screen")
            s += 0.28


def _wall_shelves(r: Rooms, kind: str = "books") -> None:
    """Two shelves on the front room's u0 wall (where there is no bookcase), with books,
    a plant and a jar or two - or records, or salvage."""
    w, fr = r.left, FRONT
    board = Material("wood", (0.5, 0.38, 0.26))
    for y in (1.35, 1.75):
        f.shelf(w, fr + 0.7, fr + 1.9, y, board, depth=0.24)
    f.books(w, fr + 0.74, fr + 1.4, 1.35, r.rng, depth=0.18, tall=0.24)
    f.plant(w, fr + 1.62, 0.12, 1.35, 0.8, r.rng)
    if kind == "books":
        f.books(w, fr + 0.74, fr + 1.2, 1.75, r.rng, depth=0.18, tall=0.22)
        f.jar(w, fr + 1.4, 0.12, 1.75, 0.05, 0.12, (0.4, 0.6, 0.7), "Jar of loose change")
        f.box_of(w, fr + 1.7, 0.12, 1.75, (0.22, 0.14, 0.16), Material("cardboard", (0.9, 0.8, 0.7)), "Shoebox of photos")
    else:
        _salvage(r, w, fr + 0.74, fr + 1.86, 1.75)


def _dartboard(w: Wall, s: float, y: float) -> None:
    w.disc(s, y, 0.23, 0.0, 0.03, Material(None, (0.1, 0.1, 0.1)), "Dartboard")
    w.disc(s, y, 0.18, 0.03, 0.034, Material(None, (0.75, 0.12, 0.1)), "Dartboard ring")
    w.disc(s, y, 0.14, 0.034, 0.038, Material(None, (0.9, 0.85, 0.7)), "Dartboard ring")
    w.disc(s, y, 0.07, 0.038, 0.042, Material(None, (0.1, 0.5, 0.2)), "Dartboard ring")
    w.disc(s, y, 0.02, 0.042, 0.046, Material(None, (0.8, 0.1, 0.1)), "Dartboard bull")
    for ds, dy in ((0.05, 0.03), (-0.09, -0.06), (0.02, 0.11)):
        w.box(s + ds - 0.004, s + ds + 0.004, y + dy - 0.004, y + dy + 0.004, 0.038, 0.14, Material(None, (0.3, 0.6, 0.9)), "Dart")


def _ashtray(b: Builder, u: float, v: float, y: float) -> None:
    b.cylinder(u, v, y, 0.07, 0.06, 0.03, Material("metal", (0.3, 0.3, 0.32)), "Overflowing ashtray", slices=10)
    b.cylinder(u, v, y + 0.03, 0.05, 0.035, 0.02, Material(None, (0.45, 0.43, 0.4)), "Ash and butts", slices=10)


def _framed_photo_standing(b: Builder, w: Wall, s: float, d: float, y: float, texture: str) -> None:
    """A little photo frame standing on a desk or a shelf, facing out from ``w``."""
    w.box(s - 0.08, s + 0.08, y, y + 0.12, d, d + 0.015, f.FRAME_BLACK, "Photo frame standing on the desk")
    w.box(s - 0.065, s + 0.065, y + 0.015, y + 0.105, d + 0.015, d + 0.018, Material(texture, (1.0, 1.0, 1.0)), "Photo")


def _a_life(r: Rooms) -> None:
    """Anybody's: a print over the TV, shelves with books on the wall, a picture between
    the doorways, a plant on the sill."""
    f.picture(r.front, tv_u(r.plan), _tv_wall_top(r) + 0.35, 0.6, 0.45, "art_abstract", "Print over the TV")
    _wall_shelves(r)
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.55, min(0.4, c - a - 0.1), 0.5, "photo_city", "Picture", frame=f.FRAME_WOOD)
    f.picture(r.bed_back, r.plan.u0 + 0.8, 1.5, 0.7, 0.45, "art_synthwave", "Picture over the bed")
    if r.desk is not None:
        f.plant(r.front, (r.desk[0] + r.desk[1]) / 2.0 + 0.25, 0.09, WINDOW_Y[0], 0.6, r.rng)


def _smoker(r: Rooms) -> None:
    """Nix's: a band poster, a dartboard, ashtrays everywhere, cans, cartons of smokes,
    a stack of records."""
    b, fr, tu = r.b, FRONT, tv_u(r.plan)
    f.picture(r.front, tu, _tv_wall_top(r) + 0.45, 0.55, 0.7, "poster_band", "Poster: CHROME WOLVES", frame=None)
    _dartboard(r.left, fr + 1.35, 1.6)
    _ashtray(b, tu - 0.2, fr + 1.2, 0.4)
    _ashtray(b, r.plan.u0 + 1.9, r.plan.back - 0.14, 0.52)  # (on the nightstand, by the lamp)
    for du, dv in ((0.1, 1.3), (0.22, 1.15), (0.3, 1.32)):
        b.cylinder(tu + du, fr + dv, 0.4, 0.033, 0.033, 0.12, Material("soda_can", (1.0, 1.0, 1.0)), "Empty can", slices=8)
    w = r.bed_partition
    f.shelf(w, r.plan.u0 + 0.25, r.plan.u0 + 1.4, 1.6, Material("metal", (0.3, 0.3, 0.32)), depth=0.24)
    for k in range(3):
        f.box_of(w, r.plan.u0 + 0.42 + k * 0.27, 0.12, 1.6, (0.24, 0.06, 0.1), Material(None, (0.85, 0.85, 0.8) if k % 2 else (0.8, 0.15, 0.12)), "Carton of cigarettes")
    for k in range(6):
        w.box(r.plan.u0 + 1.05, r.plan.u0 + 1.35, 1.6 + k * 0.006, 1.6 + (k + 1) * 0.006 - 0.001, 0.02, 0.22, Material(None, (0.05, 0.05, 0.05)), "Record")
    f.picture(r.bed_back, r.plan.u0 + 0.8, 1.45, 0.5, 0.65, "poster_film", "Poster: BLADE CITY 2", frame=None)
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.5, min(0.4, c - a - 0.1), 0.5, "calendar", "Calendar", frame=None)


def _junk_dealer(r: Rooms) -> None:
    """The junk dealer's: a steel shelving unit crammed with salvage, more of it on the
    wall shelves, a film poster, a calendar - and his stock in a crate by the sofa."""
    fr, tu = FRONT, tv_u(r.plan)
    bookcase(r, fr + 0.62, fr + 1.72, rows="salvage")
    f.picture(r.front, tu, _tv_wall_top(r) + 0.4, 0.5, 0.65, "poster_film", "Poster: BLADE CITY 2", frame=None)
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.5, min(0.4, c - a - 0.1), 0.5, "calendar", "Calendar", frame=None)
    f.picture(r.bed_back, r.plan.u0 + 0.8, 1.5, 0.7, 0.45, "photo_city", "Picture over the bed", frame=f.FRAME_WOOD)
    f.shelf(r.bed_partition, r.plan.u0 + 0.25, r.plan.u0 + 1.4, 1.6, Material("metal", (0.3, 0.3, 0.32)), depth=0.24)
    _salvage(r, r.bed_partition, r.plan.u0 + 0.3, r.plan.u0 + 1.36, 1.6)


def _tinkerer(r: Rooms) -> None:
    """The tinkerer's: a tool board on the wall, parts bins, a soldering station on the
    desk, a little robot of his own making, prints."""
    fr, tu = FRONT, tv_u(r.plan)
    w = r.left
    w.box(fr + 0.65, fr + 1.95, 1.05, 1.95, 0.0, 0.015, Material("pegboard", (1.0, 1.0, 1.0)), "Tool board")
    f.shelf(w, fr + 0.7, fr + 1.9, 0.95, Material("metal", (0.35, 0.36, 0.38)), depth=0.2, label="Parts shelf")
    for k in range(6):
        colour = ((0.85, 0.2, 0.15), (0.2, 0.5, 0.85), (0.9, 0.75, 0.1))[k % 3]
        f.box_of(w, fr + 0.8 + k * 0.18, 0.1, 0.95, (0.15, 0.1, 0.16), Material(None, colour), "Parts bin")
    f.picture(r.front, tu, _tv_wall_top(r) + 0.35, 0.6, 0.45, "art_abstract", "Print over the TV")
    f.picture(r.bed_back, r.plan.u0 + 0.8, 1.5, 0.7, 0.45, "photo_city", "Picture over the bed", frame=f.FRAME_BLACK)
    if r.desk is not None:
        a, c = r.desk
        top = DESK_TOP
        fw = r.front
        f.box_of(fw, a + 0.15, 0.45, top, (0.16, 0.06, 0.12), Material(None, (0.8, 0.3, 0.1)), "Soldering station")
        fw.box(a + 0.05, a + 0.08, top + 0.06, top + 0.08, 0.3, 0.5, f.CHROME, "Soldering iron")
        # the robot: a body, a head with two lit eyes, arms
        rs = c - 0.14
        f.box_of(fw, rs, 0.45, top, (0.1, 0.12, 0.08), Material("metal", (0.6, 0.62, 0.66)), "Little robot")
        f.box_of(fw, rs, 0.45, top + 0.12, (0.08, 0.07, 0.07), Material("metal", (0.6, 0.62, 0.66)), "Little robot's head")
        for ds in (-0.02, 0.02):
            fw.box(rs + ds - 0.008, rs + ds + 0.008, top + 0.16, top + 0.172, 0.485, 0.487, f.lamp_glow((0.3, 1.0, 1.0), 1.6), "Little robot's eye")
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.55, min(0.4, c - a - 0.1), 0.5, "art_synthwave", "Print", frame=f.FRAME_BLACK)


def _stroller(r: Rooms) -> None:
    """The stroller's: a bookcase full of books, plants everywhere, prints, fairy lights."""
    fr, tu = FRONT, tv_u(r.plan)
    bookcase(r, fr + 0.62, fr + 1.72, rows="books")
    f.picture(r.front, tu, _tv_wall_top(r) + 0.4, 0.75, 0.5, "art_synthwave", "Print over the TV", frame=f.FRAME_WOOD)
    f.fairy_lights(r.front, tu - 0.6, tu + 0.6, 2.45)
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.55, min(0.4, c - a - 0.1), 0.5, "art_abstract", "Print", frame=f.FRAME_WOOD)
    f.picture(r.bed_back, r.plan.u0 + 0.8, 1.5, 0.7, 0.45, "photo_family", "Photo over the bed", frame=f.FRAME_WOOD)
    if r.desk is not None:
        mid = (r.desk[0] + r.desk[1]) / 2.0
        for k, ds in enumerate((-0.25, 0.25)):
            f.plant(r.front, mid + ds, 0.09, WINDOW_Y[0], 0.6 + 0.15 * k, r.rng)
    # a big plant in the corner behind the TV, another in the bedroom by the wardrobe
    f.plant(r.left, fr + 0.3, 0.23, 0.0, 1.3, r.rng, pot=(0.3, 0.3, 0.32), label="Big plant")
    f.plant(r.bed_back, r.plan.split - 0.85, 0.25, 0.0, 1.4, r.rng, label="Plant")


def _office_worker(r: Rooms) -> None:
    """The office worker's: tidy. A certificate, a wall calendar, binders on a shelf, a
    photo of the family on the desk, a plant."""
    fr, tu = FRONT, tv_u(r.plan)
    f.picture(r.front, tu, _tv_wall_top(r) + 0.35, 0.45, 0.5, "certificate", "Certificate", frame=f.FRAME_GOLD)
    w = r.left
    f.shelf(w, fr + 0.7, fr + 1.9, 1.5, Material("plaster", (0.95, 0.95, 0.95)), depth=0.28)
    for k in range(9):
        w.box(fr + 0.74 + k * 0.075, fr + 0.74 + k * 0.075 + 0.065, 1.5, 1.8, 0.02, 0.27, Material(None, (0.1, 0.25, 0.5) if k < 6 else (0.75, 0.1, 0.1)), "Binder")
    f.plant(w, fr + 1.65, 0.14, 1.5, 0.9, r.rng, pot=(0.95, 0.95, 0.95))
    f.picture(w, fr + 1.3, 1.05, 0.42, 0.5, "calendar", "Calendar", frame=None)
    if r.desk is not None:
        _framed_photo_standing(r.b, r.front, r.desk[0] + 0.12, 0.25, DESK_TOP, "photo_family")
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.55, min(0.4, c - a - 0.1), 0.5, "photo_city", "Picture", frame=f.FRAME_BLACK)
    f.picture(r.bed_back, r.plan.u0 + 0.8, 1.5, 0.7, 0.45, "art_abstract", "Print over the bed", frame=f.FRAME_BLACK)


def _courier(r: Rooms) -> None:
    """The courier's: the district's map with the routes marked, a skyball poster,
    parcels stacked by the door, a helmet and energy drinks on the shelf."""
    b, p, fr, tu = r.b, r.plan, FRONT, tv_u(r.plan)
    f.picture(r.left, fr + 1.3, 1.5, 1.1, 0.8, "city_map", "Map of the district", frame=None)
    f.picture(r.front, tu, _tv_wall_top(r) + 0.45, 0.5, 0.65, "poster_sports", "Poster: SHARKS", frame=None)
    # parcels in the corner by the door
    corner = p.u1 - 0.5
    card = Material("cardboard", (0.9, 0.8, 0.65))
    b.add(corner, p.u1 - 0.02, fr + 0.1, fr + 0.6, 0.0, 0.4, card, "Parcel", solid=True)
    b.add(corner + 0.05, p.u1 - 0.06, fr + 0.15, fr + 0.5, 0.4, 0.66, card, "Parcel", solid=True)
    b.add(corner + 0.1, p.u1 - 0.1, fr + 0.62, fr + 0.9, 0.0, 0.3, card, "Parcel", solid=True)
    w = r.bed_partition
    f.shelf(w, p.u0 + 0.25, p.u0 + 1.4, 1.6, Material("wood", (0.5, 0.38, 0.26)), depth=0.26)
    w.cylinder(p.u0 + 0.45, 0.13, 1.6, 0.12, 0.1, 0.14, Material(None, (0.95, 0.6, 0.1), specular=(1.0, 1.0, 1.0), shininess=80.0), "Cycling helmet", slices=12)
    for k in range(4):
        f.can(w, p.u0 + 0.75 + k * 0.08, 0.12, 1.6, "Energy drink")
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.55, min(0.4, c - a - 0.1), 0.5, "photo_family", "Photo", frame=f.FRAME_WOOD)
    f.picture(r.bed_back, p.u0 + 0.8, 1.5, 0.7, 0.45, "art_synthwave", "Print over the bed", frame=f.FRAME_BLACK)


def _sushi_chef(r: Rooms) -> None:
    """The sushi chef's: the fish chart, cookbooks, a lucky cat waving on the shelf, knives
    on a magnetic strip, a photo of home."""
    fr, tu = FRONT, tv_u(r.plan)
    f.picture(r.front, tu, _tv_wall_top(r) + 0.4, 0.55, 0.55, "poster_fish", "Chart of the fish of the bay", frame=f.FRAME_WOOD)
    w = r.left
    board = Material("wood", (0.35, 0.25, 0.18))
    for y in (1.35, 1.75):
        f.shelf(w, fr + 0.7, fr + 1.9, y, board, depth=0.24)
    f.books(w, fr + 0.74, fr + 1.35, 1.35, r.rng, depth=0.2, tall=0.28, colours=((0.75, 0.2, 0.1), (0.9, 0.85, 0.7), (0.2, 0.3, 0.25)))
    # the lucky cat: a white body, a head, a red collar, one paw up
    cat_s, cat_d, y = fr + 1.6, 0.12, 1.35
    white = Material(None, (0.95, 0.94, 0.9))
    f.box_of(w, cat_s, cat_d, y, (0.12, 0.12, 0.1), white, "Lucky cat")
    f.box_of(w, cat_s, cat_d, y + 0.12, (0.13, 0.1, 0.1), white, "Lucky cat's head")
    w.box(cat_s - 0.062, cat_s + 0.062, y + 0.1, y + 0.116, cat_d - 0.052, cat_d + 0.052, Material(None, (0.85, 0.1, 0.1)), "Lucky cat's collar")
    w.box(cat_s + 0.062, cat_s + 0.092, y + 0.1, y + 0.2, cat_d - 0.015, cat_d + 0.015, white, "Lucky cat's paw")
    f.jar(w, fr + 0.8, 0.12, 1.75, 0.06, 0.18, (0.9, 0.9, 0.85), "Jar of rice")
    f.bottle(w, fr + 1.0, 0.12, 1.75, (0.9, 0.92, 0.9), "Bottle of sake", h=0.3)
    f.bottle(w, fr + 1.12, 0.12, 1.75, (0.25, 0.15, 0.08), "Bottle of soy sauce", h=0.22)
    # knives on a strip by the front door end of the kitchen wall
    k = r.right
    k.box(fr + 0.35, fr + 0.85, 1.4, 1.45, 0.0, 0.015, Material("metal", (0.2, 0.2, 0.22)), "Knife strip")
    for n in range(4):
        ks = fr + 0.42 + n * 0.11
        k.box(ks - 0.012, ks + 0.012, 1.3, 1.4, 0.015, 0.03, Material(None, (0.15, 0.1, 0.08)), "Knife handle")
        k.box(ks - 0.02, ks + 0.02, 1.45, 1.68 - 0.03 * n, 0.015, 0.02, f.CHROME, "Knife blade")
    a, c = _between_doorways(r)
    f.picture(r.partition, (a + c) / 2.0, 1.55, min(0.4, c - a - 0.1), 0.5, "photo_family", "Photo of home", frame=f.FRAME_WOOD)
    f.picture(r.bed_back, r.plan.u0 + 0.8, 1.5, 0.7, 0.45, "photo_city", "Picture over the bed", frame=f.FRAME_BLACK)


_LIVES: dict[str, Callable[[Rooms], None]] = {
    "smoker": _smoker,
    "junk_dealer": _junk_dealer,
    "tinkerer": _tinkerer,
    "stroller": _stroller,
    "office_worker": _office_worker,
    "courier": _courier,
    "sushi_chef": _sushi_chef,
}


# ============================================================================ the rest
def door(plan: FlatPlan) -> DoorData:
    """The front door: hinged on its inner edge, swinging in; locked; a latch inside."""
    d0, d1 = plan.door_u - DOOR_W / 2.0, plan.door_u + DOOR_W / 2.0
    st = STYLES[plan.style % len(STYLES)]
    leaf_mat = Material("leather", st.door)
    # the leaf (its outer face 5 cm inside the building's face) and its handles
    leaf = [
        _prop(plan, d0 + 0.01, d1 - 0.01, 0.05, 0.15, 0.0, DOOR_H - 0.01, leaf_mat, f"Front door of number {plan.number}"),
        _prop(plan, d1 - 0.2, d1 - 0.12, 0.015, 0.05, 0.98, 1.02, Material(None, (0.75, 0.7, 0.5)), "Door handle (outside)"),
        _prop(plan, d1 - 0.2, d1 - 0.12, 0.15, 0.185, 0.98, 1.02, Material(None, (0.75, 0.7, 0.5)), "Door handle (inside)"),
    ]
    hinge = plan.world(d0 + 0.01, 0.1)
    passage = plan.box(d0 + 0.05, d1 - 0.05, -0.9, FRONT + 0.3, 0.0, DOOR_H)
    inset = frozenset({"min_z", "max_z"}) if plan.facing == "+x" else frozenset({"min_x", "max_x"})
    # (the swing: into the flat, whichever way round its frame is turned)
    swing = -85.0 if plan.facing == "+x" else 85.0
    return DoorData(
        door_id=plan.door_id,
        obstacle=plan.box(d0, d1, 0.0, FRONT, 0.0, DOOR_H),
        unlocked_by=None,
        travel=1.0,
        speed=1.4,
        passable_at=0.7,
        passage=WalkRegion(passage.min_x, passage.max_x, passage.min_z, passage.max_z, inset=inset),
        approach_range=1.15,
        inside=plan.inside,
        leaf=tuple(leaf),
        hinge=hinge,
        swing=swing,
        faction=TENEMENT_LOCKS,
        hackable=False,  # (hacked through its access panel: ``panel``)
    )


TENEMENT_LOCKS = "tenement"  # the flats' door panels: one cheap building-wide lock system
PANEL_OUT = 0.047  # the panel's origin out from the wall (its body reaches 4.5 cm back)


def panel(plan: FlatPlan) -> PanelData:
    """The access panel beside the front door (on its latch side, under the door number):
    the tenement's cheap lock system (``TENEMENT_LOCKS``) - ping it, BYPASS LOCK, and the
    door is open to the netrunner, as number 16's is."""
    u = plan.door_u + DOOR_W / 2.0 + 0.32
    x, z = plan.world(u, -PANEL_OUT)
    return PanelData(
        panel_id=f"panel{plan.number}", x=round(x, 4), y=1.3, z=round(z, 4),
        door_id=plan.door_id, yaw=90.0 if plan.facing == "+x" else 0.0,
        faction=TENEMENT_LOCKS, name=f"NO.{plan.number} ACCESS PANEL",
    )  # fmt: skip


def walk_regions(plan: FlatPlan) -> tuple[WalkRegion, ...]:
    """The flat's floor, for the netrunner: the rooms and the doorways between them."""
    v0 = FRONT + LIVING_DEPTH
    v1 = v0 + PARTITION
    split = plan.split
    bed_door = (split - 1.05, split - 0.2)
    bath_door = (split + PARTITION + 0.3, split + PARTITION + 1.1)

    def region(u0, u1, va, vb, open_edges_v=False):
        box = plan.box(u0, u1, va, vb, 0.0, 1.0)
        if open_edges_v:  # a doorway: open at its two ends (into the rooms), walls at its sides
            inset = frozenset({"min_z", "max_z"}) if plan.facing == "+x" else frozenset({"min_x", "max_x"})
        else:
            inset = frozenset({"min_x", "max_x", "min_z", "max_z"})
        return WalkRegion(box.min_x, box.max_x, box.min_z, box.max_z, inset=inset)

    return (
        region(plan.u0, plan.u1, FRONT, v0),
        region(plan.u0, split, v1, plan.back),
        region(split + PARTITION, plan.u1, v1, plan.back),
        region(*bed_door, v0 - 0.3, v1 + 0.3, open_edges_v=True),
        region(*bath_door, v0 - 0.3, v1 + 0.3, open_edges_v=True),
    )


def places(plan: FlatPlan) -> tuple[tuple[Place, ...], tuple[tuple[str, str], ...]]:
    """Where the owner goes about the flat (and stands outside it), and the routes."""
    h = plan.prefix
    fr = FRONT
    u0, u1 = plan.u0, plan.u1
    split = plan.split
    tu = tv_u(plan)
    v0 = fr + LIVING_DEPTH
    back = plan.back

    def at(name, u, v, yaw_local=0.0, y=0.0):
        x, z = plan.world(u, v)
        return Place(f"{h}.{name}", round(x, 3), round(z, 3), yaw=round(plan.yaw(yaw_local), 2), y=y)

    bed_u = (u0 + 0.1 + u0 + 1.5) / 2.0
    out = [
        at("front", plan.door_u, -1.0, 0.0),  # on the pavement outside the door, facing it
        at("door", plan.door_u, -0.45, 0.0),  # at the door, key in hand
        at("entry", plan.door_u, fr + 0.6, 0.0),
        at("living", plan.door_u - 0.7, fr + 1.95, 180.0),
        at("kitchen", u1 - 0.95, fr + 1.45, 90.0),  # at the hob
        at("sink", u1 - 0.95, fr + 2.15, 90.0),
        at("by_sofa", tu, fr + 1.95, 0.0),  # in front of the sofa
        at("sofa", tu, fr + 2.7, 180.0),  # on it, facing the TV
        at("bed_gap", split - 0.62, v0 + 0.06, 0.0),
        at("bedroom", split - 0.75, back - 2.15, 0.0),
        at("bed", bed_u, back - 0.14, 0.0, y=0.53),  # the top of the head on the pillow
        at("bedside", bed_u + 0.95, back - 1.1, -90.0),
        # (at the wardrobe a little to the left of its middle: it stands in the corner by
        # the bathroom wall, and a reaching right arm used to go through that wall)
        at("wardrobe", split - 0.5, back - 1.4, 0.0),
        at("bath_gap", split + PARTITION + 0.7, v0 + 0.06, 0.0),
        at("bath", u1 - 0.85, v0 + PARTITION + 0.42, 90.0),  # at the basin, facing it
    ]
    routes = [
        (f"{h}.entry", f"{h}.living"),
        (f"{h}.living", f"{h}.kitchen"),
        (f"{h}.kitchen", f"{h}.sink"),
        (f"{h}.living", f"{h}.by_sofa"),
        (f"{h}.living", f"{h}.bed_gap"),
        (f"{h}.bed_gap", f"{h}.bedroom"),
        (f"{h}.bedroom", f"{h}.bedside"),
        (f"{h}.bedroom", f"{h}.wardrobe"),
        (f"{h}.living", f"{h}.bath_gap"),
        (f"{h}.bath_gap", f"{h}.bath"),
    ]
    desk = desk_span(plan)
    if desk is not None:
        mid = (desk[0] + desk[1]) / 2.0
        # on the chair at the desk (facing it, the front wall), and behind the chair
        out.append(at("desk", mid, fr + CHAIR_AT, 180.0))
        out.append(at("by_desk", mid, fr + CHAIR_AT + 0.62, 180.0))
        routes.append((f"{h}.living", f"{h}.by_desk"))
    return tuple(out), tuple(routes)


def area(plan: FlatPlan) -> PrivateArea:
    inside = plan.box(plan.u0 - SIDE, plan.u1 + SIDE, 0.0, plan.depth, 0.0, CEILING_Y)
    ex, ez = plan.world(plan.door_u, -1.3)
    out_yaw = plan.yaw(180.0)  # (facing away from the door, out into the street)
    return PrivateArea(
        area_id=plan.area_id,
        name=f"NUMBER {plan.number}",
        boxes=(inside,),
        exit=(ex, ez, out_yaw),
        door_id=plan.door_id,
        confiscate=(),
        owner_role=plan.owner_role,
    )


def light(plan: FlatPlan) -> Light:
    x, z = plan.world((plan.u0 + plan.u1) / 2.0, FRONT + 1.6)
    return Light(
        x, 2.15, z,
        ambient=(0.08, 0.068, 0.045), diffuse=(1.35, 1.1, 0.7), specular=(0.7, 0.6, 0.45),
        linear_attenuation=0.3, quadratic_attenuation=0.12, fixture="none",
    )  # fmt: skip
