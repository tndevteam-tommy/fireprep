"""The Neon Moth: a bar on the main street, just round the corner from the alley.

Behind the west block's face between number 31 and the alley's wall, north of Dell's
flat: a long bar down the right-hand side (a steel top, a magenta neon strip under its
lip, a brass foot rail, beer taps and a till), four stools along it, the back bar
behind it (a cabinet with a glass-washer sink, lit glass shelves of bottles), two little
round tables with two chairs each on the left, a jukebox in the corner, a television up
on the back wall, a neon moth on the left-hand wall - and big windows onto the street,
real glass (you can see in), the bar's name in pink neon in one and OPEN in the other.

The bartender (``ai.alley.bartender``) comes in to open up at five and locks up again
after three in the morning; while it is open the door opens for anybody who walks up to
it (``DOOR``), and the people of the street drop in for a drink (``ai.alley``).

Layout (x along the street, z into the block; the door faces +z onto the pavement):

    z=42.5 ══[window: NEON MOTH]══[door]══[window: OPEN]══  (the pavement beyond)
           │  jukebox?   table 2 ·· chair        ░ pass ░░░░ │
           │ moth neon               stool 4 ║ counter ║ back │
           │             table 1     stool 3 ║  taps   ║ bar  │
           │                          stool 2 ║ till    ║ shelves
           │ jukebox                  stool 1 ║         ║      │
    z=36   └──────────────[TV]─────────────────────────────────┘  (Dell's flat behind)
         x=-11.2                                           x=-5.7
"""

from __future__ import annotations

import random

from cyberduke.core.geometry import AABB
from cyberduke.levels import furnish as f
from cyberduke.levels.furnish import Builder, Wall, WorldFrame
from cyberduke.world.level import DoorData, Place, WalkRegion
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop

DECAL = frozenset({"transparent"})

# ----------------------------------------------------------------------------- the shell
XW, XE = -11.2, -5.7  # the inside faces of the side walls
ZS = 36.0  # the inside face of the back wall (Dell's flat is behind it)
FACE = 42.5  # the street face (the block's)
FRONT = 0.3  # the front wall's thickness
ZF = FACE - FRONT  # the inside face of the front wall
CEILING_Y = 2.7
STOREY = 3.0
DOOR_X = (-9.05, -8.05)  # the front door (1 m), hinged on its west edge
DOOR_H = 2.1
WINDOWS = ((-10.9, -9.4), (-7.7, -6.0))  # the front windows (x from, to)
WINDOW_Y = (0.85, 2.3)
FACTION = "neon_moth"  # (no network: a plain keyed door)
DOOR_ID = "bar_door"
AREA = "bar"  # (the television's id, the places' prefix)

# ----------------------------------------------------------------------------- the bar
COUNTER_X = (-7.5, -7.05)  # the counter's body (customers at -x, the bartender at +x)
COUNTER_Z = (ZS, 40.4)  # ... from the back wall up to the pass by the door
COUNTER_TOP = 1.06  # (``ai.alley.BAR_TOP``)
BACKBAR_X = (-6.25, XE - 0.01)
BACKBAR_TOP = 0.94
STOOL_X = -7.95
STOOL_ZS = (36.75, 37.65, 38.55, 39.45)
STOOL_TOP = 0.78  # (``ai.alley.BAR_STOOL_TOP``)
TAP_X = -7.3  # the beer taps' tower on the counter
GLASS_X = -7.35  # where a drink is put down on the counter for whoever is on a stool
TABLES = ((-10.2, 37.6), (-10.2, 40.0))
TABLE_TOP = 0.74
CHAIR_DZ = 0.66  # each table's two chairs, either side of it along z
CHAIR_SEAT = 0.47

BRICK = Material("brick", (0.12, 0.12, 0.13))
PLASTER = Material("plaster", (0.26, 0.17, 0.3))  # a deep plum
PANELLING = Material("wood_panel", (0.32, 0.18, 0.14))
FLOOR = Material("checker", (0.55, 0.5, 0.55), specular=(0.4, 0.4, 0.45), shininess=40.0)
CEILING = Material("plaster", (0.12, 0.1, 0.14))
COUNTER_WOOD = Material("wood_panel", (0.42, 0.13, 0.16))
STEEL_TOP = Material("metal", (0.55, 0.56, 0.6), specular=(1.0, 1.0, 1.0), shininess=80.0)
BRASS = Material("metal", (0.8, 0.6, 0.25), specular=(1.0, 0.9, 0.6), shininess=70.0)
RED_LEATHER = Material("leather", (0.62, 0.08, 0.12), specular=(0.5, 0.4, 0.4), shininess=40.0)
BLACK = Material(None, (0.05, 0.05, 0.06))
DARK = Material("metal", (0.12, 0.12, 0.14))
GLASS_PANE = Material(None, (0.35, 0.45, 0.55), specular=(1.0, 1.0, 1.0), shininess=120.0, alpha=0.22)
NEON_PINK = Material(None, (1.0, 0.3, 0.8), emission=(2.4, 0.55, 1.8))
NEON_CYAN = Material(None, (0.2, 0.9, 1.0), emission=(0.5, 2.2, 2.6))
NEON_AMBER = Material(None, (1.0, 0.7, 0.2), emission=(2.1, 1.3, 0.3))
SHELF_GLOW = Material(None, (0.3, 0.8, 1.0), emission=(0.35, 1.0, 1.3))
BOTTLES = (
    (0.55, 0.3, 0.08), (0.15, 0.45, 0.18), (0.75, 0.7, 0.55), (0.5, 0.1, 0.15),
    (0.2, 0.35, 0.6), (0.85, 0.55, 0.15), (0.3, 0.12, 0.4), (0.9, 0.9, 0.85),
)  # fmt: skip


def _shell(b: Builder) -> None:
    """The front wall (brick, its windows and the doorway), the back wall, the side walls'
    plaster and panelling, the floor, the ceiling."""
    # the front wall: brick outside, plaster inside (4 mm back), the windows and the doorway
    (w0, w1), (e0, e1) = WINDOWS
    d0, d1 = DOOR_X
    solid = ((XW, w0), (w1, d0), (d1, e0), (e1, XE))
    for a, c in solid:
        b.add(a, c, ZF + 0.01, FACE, 0.0, STOREY, BRICK, "Front wall of the Neon Moth", solid=True)
        b.add(a, c, ZF, ZF + 0.01, 0.0, CEILING_Y, PLASTER, "Plaster inside the front wall of the bar")
    for a, c in WINDOWS:
        b.add(a, c, ZF + 0.01, FACE, 0.0, WINDOW_Y[0], BRICK, "Front wall under a bar window", solid=True)
        b.add(a, c, ZF + 0.01, FACE, WINDOW_Y[1], STOREY, BRICK, "Front wall over a bar window", solid=True)
    b.add(d0, d1, ZF + 0.01, FACE, DOOR_H, STOREY, BRICK, "Lintel over the bar door", solid=True)
    b.add(d0, d1, ZF + 0.005, FACE - 0.005, -0.05, 0.008, Material("concrete", (0.2, 0.2, 0.22)), "Bar threshold")
    # the back wall (against Dell's), the plaster on the side walls (number 31's party wall,
    # the alley's wall), wood panelling to the dado rail
    b.add(XW, XE, ZS - 0.2, ZS, 0.0, STOREY, PLASTER, "Back wall of the bar", solid=True)
    b.add(XW, XW + 0.01, ZS, ZF, 0.0, CEILING_Y, PLASTER, "Plaster on the bar's west wall")
    b.add(XE - 0.01, XE, ZS, ZF, 0.0, CEILING_Y, PLASTER, "Plaster on the bar's east wall")
    west, back = b.wall("u", XW + 0.01, 1.0), b.wall("v", ZS, 1.0)
    west.box(ZS + 0.005, ZF - 0.005, 0.0, 1.1, 0.0, 0.02, PANELLING, "Panelling (bar)", tags=frozenset())
    west.box(ZS + 0.005, ZF - 0.005, 1.1, 1.14, 0.0, 0.035, BRASS, "Dado rail (bar)")
    back.box(XW + 0.03, COUNTER_X[0] - 0.002, 0.0, 1.1, 0.0, 0.02, PANELLING, "Panelling (bar)", tags=frozenset())
    back.box(XW + 0.03, COUNTER_X[0] - 0.002, 1.1, 1.14, 0.0, 0.035, BRASS, "Dado rail (bar)")
    # the floor (2 mm above the street's), the ceiling
    b.add(XW, XE, ZS, ZF, -0.02, 0.002, FLOOR, "Floor of the bar")
    b.add(XW + 0.005, XE - 0.005, ZS - 0.19, FACE - 0.2, CEILING_Y, STOREY - 0.01, CEILING, "Ceiling of the bar")
    # neon along the tops of the walls
    west.box(ZS + 0.1, ZF - 0.1, 2.56, 2.6, 0.0, 0.03, NEON_PINK, "Neon strip (bar)")
    back.box(XW + 0.1, XE - 0.1, 2.56, 2.6, 0.0, 0.03, NEON_CYAN, "Neon strip (bar)")


def _windows(b: Builder) -> None:
    """The two front windows: a steel frame, the glass (you can see in), the bar's name in
    pink neon hanging in one and OPEN in cyan in the other."""
    inside = Wall(b, "v", ZF, -1.0)  # (its "out" is into the room: -z)
    frame = Material("metal", (0.16, 0.16, 0.18), specular=(0.6, 0.6, 0.6), shininess=50.0)
    y0, y1 = WINDOW_Y
    for a, c in WINDOWS:
        b.add(a + 0.01, c - 0.01, FACE - 0.166, FACE - 0.154, y0 + 0.045, y1 - 0.045, GLASS_PANE, "Bar window glass", tags=DECAL)
        for z0, z1 in ((FACE - 0.2, FACE - 0.17), (FACE - 0.15, FACE - 0.12)):
            b.add(a, c, z0, z1, y0, y0 + 0.04, frame, "Bar window frame")
            b.add(a, c, z0, z1, y1 - 0.04, y1, frame, "Bar window frame")
            b.add(a, a + 0.04, z0, z1, y0 + 0.04, y1 - 0.04, frame, "Bar window frame")
            b.add(c - 0.04, c, z0, z1, y0 + 0.04, y1 - 0.04, frame, "Bar window frame")
        b.add(a - 0.03, c + 0.03, FACE - 0.005, FACE + 0.12, y0 - 0.05, y0 + 0.01, Material("concrete", (0.22, 0.22, 0.24)), "Bar window sill")
        b.add(a, c, ZF - 0.2, ZF, y0 - 0.04, y0, Material("wood", (0.35, 0.2, 0.14)), "Bar window ledge (inside)")
    (w0, w1), (e0, e1) = WINDOWS
    # the bar's name in pink tube, hanging in the west window on chains; OPEN in the east
    for (a, c), tex, width, height, y in (
        ((w0, w1), "sign_moth_window", 1.2, 0.75, 1.75),
        ((e0, e1), "sign_open", 0.84, 0.42, 1.72),
    ):
        mid = (a + c) / 2.0
        is_open = tex == "sign_open"  # (lit while the bar is open: ``render.television``)
        inside.box(mid - width / 2.0, mid + width / 2.0, y - height / 2.0, y + height / 2.0,
                   0.1, 0.12, Material(tex, (1.0, 1.0, 1.0), emission=(1.3, 1.3, 1.3)),
                   f"{'OPEN' if is_open else 'NEON MOTH'} sign in the window",
                   tags=frozenset({"transparent", "screen", f"open:{AREA}"}) if is_open else DECAL)  # fmt: skip
        for cs in (mid - width / 2.0 + 0.08, mid + width / 2.0 - 0.08):
            inside.box(cs - 0.004, cs + 0.004, y + height / 2.0, WINDOW_Y[1], 0.107, 0.113, DARK, "Sign chain")


def _front(b: Builder) -> None:
    """Outside: the door's frame, the sign over it, the drinks board."""
    d0, d1 = DOOR_X
    for e in (d0, d1):
        b.add(e - 0.06, e + 0.06, FACE - 0.02, FACE + 0.1, 0.0, DOOR_H + 0.04, DARK, "Bar door frame post")
    b.add(d0 - 0.06, d1 + 0.06, FACE - 0.02, FACE + 0.1, DOOR_H + 0.04, DOOR_H + 0.16, DARK, "Bar door frame head")
    # the sign over the door, a moth in neon either side of the name
    b.add(-10.4, -6.6, FACE, FACE + 0.1, 2.42, 2.86, Material("sign_neon_moth", (1.0, 1.0, 1.0), emission=(1.3, 1.3, 1.3)), "NEON MOTH sign", tags=DECAL)
    # a chalk board on the pavement wall: tonight's drinks
    b.add(-11.1, -10.95, FACE, FACE + 0.03, 0.3, 1.1, Material("wood", (0.3, 0.2, 0.12)), "Drinks board", tags=DECAL)


def _counter(b: Builder, rng: random.Random) -> None:
    """The bar itself: a panelled counter with a steel top, the neon under its lip, the
    foot rail, the taps, the till, things on it."""
    x0, x1 = COUNTER_X
    z0, z1 = COUNTER_Z
    b.add(x0, x1, z0, z1, 0.0, COUNTER_TOP - 0.06, COUNTER_WOOD, "Bar counter", solid=True)
    b.add(x0 - 0.1, x1 + 0.04, z0, z1 + 0.05, COUNTER_TOP - 0.06, COUNTER_TOP, STEEL_TOP, "Bar counter top")
    b.collide(x0 - 0.1, x1 + 0.04, z0, z1 + 0.05, 0.0, COUNTER_TOP)
    b.add(x0 - 0.012, x0, z0 + 0.05, z1 - 0.05, COUNTER_TOP - 0.14, COUNTER_TOP - 0.1, NEON_PINK, "Neon under the bar", tags=DECAL)
    b.add(x0 - 0.03, x0, z0, z1, 0.0, 0.1, BLACK, "Bar kick plate")
    front = Wall(b, "u", x0 - 0.03, -1.0)
    front.lying(z0 + 0.1, z1 - 0.1, 0.12, 0.17, 0.17, 0.024, BRASS, "Brass foot rail", slices=8)
    for z in (z0 + 0.3, (z0 + z1) / 2.0, z1 - 0.3):
        front.box(z - 0.015, z + 0.015, 0.0, 0.2, 0.0, 0.15, BRASS, "Foot rail bracket")
    top = COUNTER_TOP
    # the beer taps: a tower on the counter, the spouts and handles on the bartender's side
    b.cylinder(TAP_X, 39.6, top, 0.045, 0.045, 0.36, f.CHROME, "Beer tap tower", slices=12)
    b.add(TAP_X - 0.04, TAP_X + 0.04, 39.3, 39.9, top + 0.32, top + 0.38, f.CHROME, "Beer tap bar")
    for k, rgb in enumerate(((0.9, 0.6, 0.1), (0.1, 0.1, 0.1), (0.9, 0.2, 0.5))):
        z = 39.38 + k * 0.22
        b.add(TAP_X + 0.04, TAP_X + 0.08, z - 0.015, z + 0.015, top + 0.22, top + 0.36, f.CHROME, "Beer tap spout")
        b.add(TAP_X + 0.04, TAP_X + 0.08, z - 0.02, z + 0.02, top + 0.38, top + 0.52, Material(None, rgb), "Beer tap handle")
    b.add(TAP_X + 0.0, TAP_X + 0.14, 39.25, 39.95, top, top + 0.01, Material("grate", (0.3, 0.3, 0.32)), "Drip tray")
    # the till (its display to the bartender)
    b.add(-7.45, -7.2, 40.0, 40.3, top, top + 0.16, DARK, "Till")
    b.add(-7.2, -7.19, 40.04, 40.26, top + 0.05, top + 0.15, Material(None, (0.2, 0.9, 0.4), emission=(0.3, 1.4, 0.5)), "Till display", tags=DECAL)
    # a bowl of bar snacks, beer mats, a stack of glasses
    b.cylinder(-7.35, 37.2, top, 0.07, 0.09, 0.05, Material(None, (0.2, 0.2, 0.25)), "Snack bowl", slices=12)
    b.cylinder(-7.35, 37.2, top + 0.035, 0.075, 0.075, 0.012, Material(None, (0.8, 0.55, 0.2)), "Bar snacks", slices=12)
    for z in STOOL_ZS:
        b.add(GLASS_X - 0.06, GLASS_X + 0.06, z - 0.06, z + 0.06, top, top + 0.004, Material(None, (0.9, 0.85, 0.6)), "Beer mat", tags=DECAL)
    for k in range(3):
        b.cylinder(-7.14, 38.0 + k * 0.09, top, 0.035, 0.04, 0.13, Material(None, (0.7, 0.8, 0.85), specular=(1, 1, 1), shininess=90.0), "Clean glass", slices=10)


def _back_bar(b: Builder, rng: random.Random) -> None:
    """Against the east wall: a cabinet (doors, a glass-washer sink in its top), glass
    shelves lit from behind, rows of bottles, a mirror strip, a fridge of bottled beer."""
    x0, x1 = BACKBAR_X
    z0, z1 = ZS + 0.005, 40.3
    top = BACKBAR_TOP
    b.add(x0, x1, z0, z1, 0.0, top - 0.04, Material("wood_panel", (0.22, 0.12, 0.1)), "Back bar cabinet", solid=True)
    f.worktop_with_sink(b, x0 - 0.03, x1, z0, z1, top - 0.04, top, (x0 + 0.08, x1 - 0.1, 36.6, 37.1), STEEL_TOP, "Back bar top")
    face = Wall(b, "u", x0, -1.0)
    f.cabinet_fronts(face, 37.3, z1 - 0.05, 0.08, top - 0.1, Material("wood", (0.3, 0.16, 0.12)), BRASS, drawers=False, label="Back bar cupboard")
    # the bottle fridge under the sink end: a glass door, lit inside
    face.box(z0 + 0.05, 37.25, 0.1, top - 0.1, 0.0, 0.012, Material(None, (0.2, 0.5, 0.6), emission=(0.15, 0.45, 0.55)), "Bottle fridge door (lit)")
    f.tap(Wall(b, "u", x1 - 0.01, -1.0), 36.85, 0.05, top, reach=0.16, label="Back bar tap")
    # the wall behind: a mirror strip, glass shelves lit from behind, bottles
    wall = Wall(b, "u", XE - 0.01, -1.0)
    wall.box(36.4, 40.1, 1.2, 2.3, 0.0, 0.006, SHELF_GLOW, "Back bar light panel", tags=DECAL)
    wall.box(36.3, 40.2, 2.3, 2.5, 0.0, 0.008, Material(None, (0.3, 0.32, 0.36), specular=(1, 1, 1), shininess=120.0), "Back bar mirror strip")
    for y in (1.25, 1.65, 2.05):
        wall.box(36.4, 40.1, y - 0.02, y, 0.006, 0.28, Material(None, (0.5, 0.7, 0.75), specular=(1, 1, 1), shininess=100.0), "Glass shelf")
        for bz in (36.6, 39.9):
            wall.box(bz - 0.01, bz + 0.01, y - 0.06, y - 0.02, 0.006, 0.22, f.CHROME, "Glass shelf bracket")
        z = 36.52
        while z < 40.0:
            rgb = rng.choice(BOTTLES)
            h = rng.uniform(0.24, 0.34)
            f.bottle(wall, z, 0.14, y, rgb, "Bottle on the back bar", h=h)
            z += rng.uniform(0.1, 0.16)
    # the television, up on the back wall over the end of the bar, for the customers
    back = Wall(b, "v", ZS, 1.0)
    tv_x = -8.9
    back.box(tv_x - 0.52, tv_x + 0.52, 1.78, 2.42, 0.03, 0.09, DARK, "Bar TV", tags=frozenset())
    back.box(tv_x - 0.06, tv_x + 0.06, 2.0, 2.2, 0.0, 0.03, DARK, "Bar TV bracket")
    back.box(tv_x - 0.48, tv_x + 0.48, 1.81, 2.39, 0.09, 0.094, Material("tv_off", (1.0, 1.0, 1.0), emission=(0.85, 0.85, 0.9)), "Bar TV screen", tags=frozenset({"transparent", "screen", f"tv:{AREA}"}))
    b.collide(tv_x - 0.52, tv_x + 0.52, ZS, ZS + 0.1, 1.78, 2.42)


def _stool(b: Builder, x: float, z: float, top: float = STOOL_TOP) -> None:
    b.cylinder(x, z, 0.0, 0.2, 0.2, 0.03, DARK, "Bar stool base", slices=12)
    b.cylinder(x, z, 0.03, 0.03, 0.03, top - 0.1, f.CHROME, "Bar stool pole", slices=8)
    b.cylinder(x, z, 0.3, 0.15, 0.15, 0.02, f.CHROME, "Bar stool foot ring", slices=12)
    b.cylinder(x, z, top - 0.07, 0.2, 0.19, 0.07, RED_LEATHER, "Bar stool seat", slices=14)
    b.collide(x - 0.2, x + 0.2, z - 0.2, z + 0.2, 0.0, top)


def _tables(b: Builder) -> None:
    """Two small round tables, a chair either side of each (along z: facing each other),
    and the jukebox in the corner."""
    for x, z in TABLES:
        b.cylinder(x, z, 0.0, 0.25, 0.25, 0.03, DARK, "Bar table base", slices=12)
        b.cylinder(x, z, 0.03, 0.04, 0.04, TABLE_TOP - 0.06, f.CHROME, "Bar table leg", slices=8)
        b.cylinder(x, z, TABLE_TOP - 0.03, 0.38, 0.38, 0.03, Material("wood", (0.25, 0.12, 0.1), specular=(0.5, 0.5, 0.5), shininess=40.0), "Bar table top", slices=18)
        b.collide(x - 0.38, x + 0.38, z - 0.38, z + 0.38, 0.0, TABLE_TOP)
        b.cylinder(x + 0.12, z - 0.05, TABLE_TOP, 0.03, 0.03, 0.07, Material(None, (0.9, 0.5, 0.2), emission=(1.2, 0.5, 0.1)), "Table candle (lit)", slices=8)
        for side in (-1.0, 1.0):  # the chairs: seat, back on the far side, four legs
            cz = z + side * CHAIR_DZ
            b.add(x - 0.21, x + 0.21, cz - 0.2, cz + 0.2, CHAIR_SEAT - 0.05, CHAIR_SEAT, RED_LEATHER, "Bar chair seat", solid=True)
            bz = cz + side * 0.18
            b.add(x - 0.21, x + 0.21, bz - 0.02, bz + 0.02, CHAIR_SEAT, CHAIR_SEAT + 0.45, RED_LEATHER, "Bar chair back")
            for lx in (x - 0.18, x + 0.18):
                for lz in (cz - 0.17, cz + 0.17):
                    b.add(lx - 0.015, lx + 0.015, lz - 0.015, lz + 0.015, 0.0, CHAIR_SEAT - 0.05, DARK, "Bar chair leg")
            b.collide(x - 0.21, x + 0.21, cz - 0.2, cz + 0.2, 0.0, CHAIR_SEAT)
    # the jukebox in the back corner: a rounded top of light, the records behind glass
    x0, x1, z0, z1 = XW + 0.03, XW + 0.72, ZS + 0.005, ZS + 0.5
    b.add(x0, x1, z0, z1, 0.0, 1.25, Material(None, (0.35, 0.12, 0.3), specular=(0.8, 0.8, 0.8), shininess=60.0), "Jukebox", solid=True)
    b.add(x0 + 0.02, x1 - 0.02, z1, z1 + 0.012, 0.55, 1.05, Material(None, (0.3, 0.9, 1.0), emission=(0.4, 1.4, 1.7)), "Jukebox window (lit)", tags=DECAL)
    b.add(x0 - 0.01, x1 + 0.01, z0, z1 + 0.02, 1.25, 1.45, Material(None, (1.0, 0.5, 0.2), emission=(1.8, 0.8, 0.25)), "Jukebox arch (lit)")
    b.add(x0 + 0.1, x1 - 0.1, z1, z1 + 0.012, 0.2, 0.4, Material("grate", (0.6, 0.5, 0.4)), "Jukebox speaker grille", tags=DECAL)


def _walls_dressed(b: Builder) -> None:
    """The moth in neon on the west wall, posters, a dartboard, a clock."""
    west = Wall(b, "u", XW + 0.03, 1.0)
    plaster = Wall(b, "u", XW + 0.01, 1.0)
    mz, my = 38.8, 1.95
    west.box(mz - 0.025, mz + 0.025, my - 0.32, my + 0.22, 0.02, 0.05, NEON_AMBER, "Neon moth (body)")
    for side in (-1.0, 1.0):
        lo, hi = sorted((mz + side * 0.05, mz + side * 0.46))
        west.box(lo, hi, my + 0.05, my + 0.09, 0.02, 0.05, NEON_PINK, "Neon moth (wing)")
        west.box(lo, hi, my - 0.27, my - 0.23, 0.02, 0.05, NEON_PINK, "Neon moth (wing)")
        lo, hi = sorted((mz + side * 0.42, mz + side * 0.46))
        west.box(lo, hi, my - 0.23, my + 0.05, 0.02, 0.05, NEON_PINK, "Neon moth (wing)")
        west.box(mz + side * 0.03, mz + side * 0.15, my + 0.28, my + 0.3, 0.02, 0.05, NEON_CYAN, "Neon moth (feeler)")
    f.picture(plaster, 37.0, 1.8, 0.5, 0.7, "poster_band", "Band poster (bar)", frame=None)
    f.picture(plaster, 40.6, 1.75, 0.55, 0.75, "poster_film", "Film poster (bar)", frame=f.FRAME_GOLD)
    back = Wall(b, "v", ZS, 1.0)
    back.disc(-10.1, 1.75, 0.22, 0.02, 0.05, Material(None, (0.15, 0.15, 0.12)), "Dartboard", slices=16)
    back.disc(-10.1, 1.75, 0.14, 0.05, 0.055, Material(None, (0.7, 0.15, 0.12)), "Dartboard ring", slices=16)
    f.picture(back, -7.95, 1.85, 0.45, 0.6, "poster_idol", "Idol poster (bar)", frame=None)


def build() -> tuple[list[Prop], list[AABB]]:
    b = Builder(WorldFrame())
    rng = random.Random(1313)
    _shell(b)
    _windows(b)
    _front(b)
    _counter(b, rng)
    _back_bar(b, rng)
    for z in STOOL_ZS:
        _stool(b, STOOL_X, z)
    _tables(b)
    _walls_dressed(b)
    f.hanging_lamp(b, -9.6, 38.8, CEILING_Y, (0.9, 0.3, 0.6))
    f.hanging_lamp(b, -7.3, 37.2, CEILING_Y, (0.9, 0.55, 0.2), drop=0.7)
    f.hanging_lamp(b, -7.3, 39.4, CEILING_Y, (0.9, 0.55, 0.2), drop=0.7)
    return b.props, b.colliders


# ----------------------------------------------------------------------------- the door
def door() -> DoorData:
    """The front door: hinged on its west edge, swinging in; a porthole; locked when the
    bar is shut (the latch opens it from inside), open to whoever walks up while it is
    open - the bartender unlocks it at opening time (``ai.alley.bartender``)."""
    d0, d1 = DOOR_X
    leaf_mat = Material("leather", (0.5, 0.08, 0.2))
    leaf = (
        Prop.box((d0 + d1) / 2.0, (DOOR_H - 0.01) / 2.0, FACE - 0.1, d1 - d0 - 0.02, DOOR_H - 0.01, 0.1, leaf_mat, label="Bar door"),
        Prop.box((d0 + d1) / 2.0, 1.5, FACE - 0.045, 0.32, 0.32, 0.012, Material(None, (0.95, 0.5, 0.8), emission=(0.9, 0.3, 0.6)), label="Bar door porthole (lit from inside)"),
        Prop.box(d1 - 0.16, 1.0, FACE - 0.035, 0.08, 0.04, 0.03, BRASS, label="Bar door handle (outside)"),
        Prop.box(d1 - 0.16, 1.0, FACE - 0.165, 0.08, 0.04, 0.03, BRASS, label="Bar door handle (inside)"),
    )
    return DoorData(
        door_id=DOOR_ID,
        obstacle=AABB(d0, d1, ZF, FACE, 0.0, DOOR_H),
        unlocked_by=None,
        travel=1.0,
        speed=1.4,
        passable_at=0.7,
        passage=WalkRegion(d0 + 0.05, d1 - 0.05, ZF - 0.3, FACE + 0.9, inset=frozenset({"min_x", "max_x"})),
        approach_range=1.3,
        inside="-z",
        leaf=leaf,
        hinge=(d0 + 0.01, FACE - 0.1),
        swing=85.0,
        faction=FACTION,
        hackable=False,
    )


# ----------------------------------------------------------------------------- for people
FLOOR_REGION = WalkRegion(XW, XE, ZS, ZF, inset=frozenset({"min_x", "max_x", "min_z", "max_z"}))
WALK_REGIONS = (FLOOR_REGION,)

LIGHTS = (
    Light(
        -9.0, 2.35, 39.0,
        ambient=(0.1, 0.05, 0.08), diffuse=(1.5, 0.75, 1.05), specular=(0.8, 0.5, 0.6),
        linear_attenuation=0.32, quadratic_attenuation=0.12, fixture="none",
    ),
    Light(
        -6.6, 2.3, 38.2,
        ambient=(0.05, 0.04, 0.03), diffuse=(1.2, 0.85, 0.5), specular=(0.6, 0.5, 0.35),
        linear_attenuation=0.4, quadratic_attenuation=0.18, fixture="none",
    ),
)  # fmt: skip

SERVE_X = -6.78  # where the bartender stands behind the counter
BACK_X = -6.6  # ... and at the back bar (the sink, the shelves)
BY_STOOL_X = -8.6  # standing behind a stool (before sitting on it)


def places() -> tuple[tuple[Place, ...], tuple[tuple[str, str], ...]]:
    """Where people go in the bar (``ai.alley``), and the straight walks between them."""
    p = AREA
    mid = (DOOR_X[0] + DOOR_X[1]) / 2.0
    out = [
        Place(f"{p}.front", mid, FACE + 0.9, yaw=180.0),  # on the pavement, facing the door
        Place(f"{p}.door", mid, FACE + 0.45, yaw=180.0),
        Place(f"{p}.entry", mid, ZF - 0.6, yaw=180.0),
        Place(f"{p}.floor", -8.6, 40.9, yaw=180.0),
        Place(f"{p}.aisle", -9.1, 38.6, yaw=180.0),
        # behind the bar: in through the pass at the door end, along the counter
        Place(f"{p}.pass", -7.2, 41.2, yaw=90.0),
        Place(f"{p}.behind", SERVE_X, 40.7, yaw=180.0),
        Place(f"{p}.tap", SERVE_X, 39.6, yaw=-90.0),  # at the taps, facing them over the counter
        Place(f"{p}.till", SERVE_X, 40.15, yaw=-90.0),
        Place(f"{p}.sink", BACK_X, 36.85, yaw=90.0),  # at the glass-washer, facing the back bar
        Place(f"{p}.shelves", BACK_X, 38.3, yaw=90.0),  # facing the bottles
    ]
    routes = [
        (f"{p}.entry", f"{p}.floor"),
        (f"{p}.floor", f"{p}.aisle"),
        (f"{p}.floor", f"{p}.pass"),
        (f"{p}.pass", f"{p}.behind"),
        (f"{p}.behind", f"{p}.till"),
        (f"{p}.till", f"{p}.tap"),
        (f"{p}.entry", f"{p}.door"),
    ]
    # the stools: behind each one, and on it (facing the counter)
    prev = f"{p}.floor"
    for k, z in reversed(list(enumerate(STOOL_ZS, start=1))):
        out.append(Place(f"{p}.by_stool{k}", BY_STOOL_X, z, yaw=90.0))
        out.append(Place(f"{p}.stool{k}", STOOL_X + 0.08, z, yaw=90.0))
        out.append(Place(f"{p}.stool{k}.glass", GLASS_X, z, y=COUNTER_TOP))  # (its drink)
        out.append(Place(f"{p}.serve{k}", SERVE_X, z, yaw=-90.0))  # behind the bar, facing it
        routes.append((prev, f"{p}.by_stool{k}"))
        prev = f"{p}.by_stool{k}"
    # along behind the bar: the serving spots, the shelves, the sink
    chain = [f"{p}.tap", f"{p}.serve4", f"{p}.serve3", f"{p}.shelves", f"{p}.serve2", f"{p}.serve1", f"{p}.sink"]
    routes += list(zip(chain, chain[1:], strict=False))
    # the tables: beside each chair, and on it (facing the table)
    for k, (x, z) in enumerate(TABLES, start=1):
        out.append(Place(f"{p}.by_table{k}", x + 0.75, z, yaw=-90.0))
        for side, tag in ((-1.0, "a"), (1.0, "b")):
            cz = z + side * CHAIR_DZ
            out.append(Place(f"{p}.chair{k}{tag}", x, cz + side * 0.04, yaw=0.0 if side < 0 else 180.0))
            out.append(Place(f"{p}.chair{k}{tag}.glass", x, z + side * 0.2, y=TABLE_TOP))
            out.append(Place(f"{p}.by_chair{k}{tag}", x + 0.55, cz, yaw=-90.0))
            routes.append((f"{p}.by_table{k}", f"{p}.by_chair{k}{tag}"))
        routes.append((f"{p}.aisle", f"{p}.by_table{k}"))
    return tuple(out), tuple(routes)


__all__ = ["DOOR_ID", "LIGHTS", "WALK_REGIONS", "build", "door", "places"]
