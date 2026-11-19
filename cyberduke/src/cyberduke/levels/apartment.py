"""Number 16: Dell's flat, behind the alley's west wall - somewhere to break into.

A ground-floor one-bedroom in a building nobody has maintained in thirty years: the front
door opens off the alley into the living room (a sagging sofa facing a TV stuck on a dead
channel, a coffee table of noodle cups, the cheap little computer desk by the door), a
galley kitchen across the back (hotplate, a sink full of dishes, a dented fridge), a
bathroom with a cracked mirror and a mouldy shower, and a small bedroom - a mattress on
pallets, a crate for a nightstand, and on it the data stick Nix wants.

Layout (x across, z along the alley; the alley is to the +x side, x > -4.5):

      z=35.6 ┌───────────────────────────────────────────┐
             │ fridge   KITCHEN (hotplate, sink)  counter │
      z=32.7 ├──────────────┬─────┐    opening    ┌─────┤
             │   BEDROOM    │ TV  │               │desk │
             │  mattress    │     LIVING ROOM     │     │  front door (z 29.9..31.1)
      z=29.1 ├────nightstand┤bed door                   ║  ── the alley, the step,
             │   BATHROOM   │bath door    coffee table  │     the access panel
             │ shower toilet│ camera   SOFA             │
      z=26.8 └──────────────┴──────────────────────────┘
           x=-10.8        x=-8.2                      x=-4.8

Everything here is data the street level folds in (``levels.street``): the props and
colliders, the floor the player may walk (``WALK_REGIONS``), the door with its latch, the
access panel by it, the home camera, the computer, the data stick, the private area the
owner and the camera guard, the places the resident lives their life between, and two
lights.
"""

from __future__ import annotations

import random

from cyberduke.core.geometry import AABB
from cyberduke.levels import furnish
from cyberduke.levels.furnish import Builder, WorldFrame
from cyberduke.world.level import (
    CameraData,
    DoorData,
    PanelData,
    PickupData,
    Place,
    PrivateArea,
    TerminalData,
    WalkRegion,
)
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop

FACTION = "dells_home"  # the flat's own little network: the camera, the panel, the computer

# ----------------------------------------------------------------------------- dimensions
FRONT_FACE_X = -4.5  # the alley face of the building
FRONT_BACK_X = -4.8  # the inside face of the front wall
BACK_X = -10.8
SOUTH_Z, NORTH_Z = 26.8, 35.6
PARTITION_X = (-8.32, -8.2)  # living room | bathroom & bedroom
BATH_BED_Z = (29.0, 29.12)
KITCHEN_WALL_Z = (32.6, 32.72)
KITCHEN_GAP_X = (-7.2, -5.6)  # the opening from the living room into the kitchen
BATH_DOOR_Z = (27.5, 28.4)
BED_DOOR_Z = (30.4, 31.3)
DOOR_Z = (29.9, 31.1)  # the front door (1.2 m)
DOOR_HEIGHT = 2.2
CEILING_Y = 2.7
SHELL_TOP = 3.0  # the ground floor ends here; the facade above is the building's

# ----------------------------------------------------------------------------- materials
PLASTER = Material("plaster", (0.72, 0.68, 0.6))
PLASTER_DARK = Material("plaster", (0.5, 0.48, 0.44))
BRICK = Material("brick", (0.12, 0.12, 0.13))  # the alley face (the tenement's brick)
LINO = Material("linoleum", (0.8, 0.8, 0.78))
TILE = Material("tile", (0.8, 0.85, 0.82))
CEILING = Material("plaster", (0.36, 0.34, 0.3))
SOFA = Material("leather", (0.42, 0.16, 0.13))
FABRIC = Material("fabric", (0.5, 0.48, 0.52))
WOOD = Material("wood", (0.55, 0.42, 0.3))
METAL = Material("metal", (0.45, 0.46, 0.5))
DARK = Material("metal", (0.12, 0.12, 0.14))
WHITE = Material(None, (0.78, 0.78, 0.74))
CRATE = Material("crate", (0.8, 0.8, 0.8))
CARDBOARD = Material("cardboard", (0.85, 0.75, 0.6))
BAG = Material("leather", (0.16, 0.16, 0.18), specular=(0.4, 0.4, 0.45), shininess=30.0)
TV_GLASS = Material("tv_off", (1.0, 1.0, 1.0), emission=(0.85, 0.85, 0.9))
TV_TAGS = frozenset({"transparent", "screen", "tv:flat16"})
BULB = Material(None, (1.0, 0.9, 0.6), emission=(2.2, 1.8, 1.0))
TUBE = Material(None, (0.85, 1.0, 0.9), emission=(1.2, 1.6, 1.3))
NEON_PINK = Material(None, (1.0, 0.3, 0.7), emission=(2.2, 0.5, 1.4))
NEON_TEAL = Material(None, (0.2, 1.0, 0.9), emission=(0.4, 2.0, 1.8))
HEATER = Material(None, (1.0, 0.35, 0.1), emission=(1.8, 0.5, 0.1))
DOOR_MAT = Material("leather", (0.55, 0.34, 0.22))  # scuffed oxblood paint
CLUTTER = frozenset({"clutter", "transparent"})
DECAL = frozenset({"transparent"})


def _box(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    mat: Material,
    label: str,
    solid: bool = False,
    tags: frozenset[str] = frozenset(),
    rotate: tuple = (),
) -> Prop:
    return Prop.box(
        (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0, x1 - x0, y1 - y0, z1 - z0, mat,
        solid=solid, label=label, tags=tags, rotate=rotate,
    )  # fmt: skip


def _wall(x0, x1, z0, z1, mat=PLASTER, label="Flat wall", y0=0.0, y1=CEILING_Y):
    """A wall segment (prop and collider)."""
    return _box(x0, x1, y0, y1, z0, z1, mat, label, solid=True), AABB(x0, x1, z0, z1, y0, y1)


def shell() -> tuple[list[Prop], list[AABB]]:
    """Walls, floors, ceilings and the front wall with its doorway."""
    props: list[Prop] = []
    colliders: list[AABB] = []

    def wall(*a, **k):
        p, c = _wall(*a, **k)
        props.append(p)
        colliders.append(c)

    # the front wall (thin here: the flat sits right behind it) - the tenement's brick on
    # the alley side, plaster inside - with the doorway
    for z0, z1 in ((SOUTH_Z - 0.2, DOOR_Z[0]), (DOOR_Z[1], NORTH_Z + 0.2)):
        props.append(_box(FRONT_BACK_X + 0.01, FRONT_FACE_X, 0.0, SHELL_TOP, z0, z1, BRICK, "Front wall of number 16", solid=True))
        props.append(_box(FRONT_BACK_X, FRONT_BACK_X + 0.01, 0.0, CEILING_Y, max(z0, SOUTH_Z), min(z1, NORTH_Z), PLASTER_DARK, "Plaster inside the front wall"))
        colliders.append(AABB(FRONT_BACK_X - 0.02, FRONT_FACE_X + 0.2, z0, z1, 0.0, SHELL_TOP))
    props.append(_box(FRONT_BACK_X, FRONT_FACE_X, DOOR_HEIGHT, SHELL_TOP, DOOR_Z[0], DOOR_Z[1], BRICK, "Lintel over the door of number 16", solid=True))
    colliders.append(AABB(FRONT_BACK_X, FRONT_FACE_X, DOOR_Z[0], DOOR_Z[1], DOOR_HEIGHT, SHELL_TOP))
    # outer walls: the neighbours' flats either side, the back of the building
    wall(BACK_X - 0.2, FRONT_BACK_X, SOUTH_Z - 0.2, SOUTH_Z, label="Party wall")
    wall(BACK_X - 0.2, FRONT_BACK_X, NORTH_Z, NORTH_Z + 0.2, label="Party wall")
    wall(BACK_X - 0.2, BACK_X, SOUTH_Z, NORTH_Z, label="Back wall of the flat")
    # the partition: living room | bathroom, bedroom (two doorways)
    px0, px1 = PARTITION_X
    for z0, z1 in ((SOUTH_Z, BATH_DOOR_Z[0]), (BATH_DOOR_Z[1], BED_DOOR_Z[0]), (BED_DOOR_Z[1], KITCHEN_WALL_Z[0])):
        wall(px0, px1, z0, z1, label="Partition wall")
    for z0, z1 in (BATH_DOOR_Z, BED_DOOR_Z):
        props.append(_box(px0, px1, 2.05, CEILING_Y, z0, z1, PLASTER, "Partition over a doorway", solid=True))
        colliders.append(AABB(px0, px1, z0, z1, 2.05, CEILING_Y))
        # a door frame, no door (long gone)
        for z in (z0, z1):
            props.append(_box(px0 - 0.03, px1 + 0.03, 0.0, 2.05, z - 0.03, z + 0.03, WOOD, "Door frame"))
    wall(BACK_X, px0, BATH_BED_Z[0], BATH_BED_Z[1], TILE, label="Bathroom wall")
    # the kitchen wall with its opening
    kz0, kz1 = KITCHEN_WALL_Z
    wall(BACK_X, KITCHEN_GAP_X[0], kz0, kz1, label="Kitchen wall")
    wall(KITCHEN_GAP_X[1], FRONT_BACK_X, kz0, kz1, label="Kitchen wall")
    props.append(_box(KITCHEN_GAP_X[0], KITCHEN_GAP_X[1], 2.25, CEILING_Y, kz0, kz1, PLASTER, "Soffit over the kitchen opening", solid=True))
    colliders.append(AABB(KITCHEN_GAP_X[0], KITCHEN_GAP_X[1], kz0, kz1, 2.25, CEILING_Y))
    # floors and ceiling
    props.append(_box(PARTITION_X[1], FRONT_BACK_X, -0.02, 0.0, SOUTH_Z, KITCHEN_WALL_Z[0], LINO, "Living room floor (cracked lino)"))
    props.append(_box(BACK_X, FRONT_BACK_X, -0.02, 0.0, KITCHEN_WALL_Z[1], NORTH_Z, LINO, "Kitchen floor"))
    props.append(_box(BACK_X, PARTITION_X[0], -0.02, 0.0, SOUTH_Z, BATH_BED_Z[0], TILE, "Bathroom floor"))
    props.append(_box(BACK_X, PARTITION_X[0], -0.02, 0.0, BATH_BED_Z[1], KITCHEN_WALL_Z[0], Material("wood", (0.32, 0.25, 0.2)), "Bedroom floorboards"))
    # (it stops inside the front wall's brick and under the top of the shell: run out to the
    # wall's outer face, its edge z-fought with the brick - the flat's ceiling showing
    # through the outside of the building)
    props.append(_box(BACK_X - 0.2, FRONT_BACK_X + 0.1, CEILING_Y, SHELL_TOP - 0.01, SOUTH_Z - 0.2, NORTH_Z + 0.2, CEILING, "Ceiling of number 16"))
    # the bathroom and kitchen walls tiled up to shoulder height
    props.append(_box(BACK_X + 0.005, BACK_X + 0.02, 0.0, 1.495, SOUTH_Z, BATH_BED_Z[0], TILE, "Bathroom tiles"))
    props.append(_box(BACK_X, PARTITION_X[0], 0.0, 1.5, SOUTH_Z + 0.005, SOUTH_Z + 0.02, TILE, "Bathroom tiles"))
    props.append(_box(-9.8, -4.9, 0.92, 1.5, NORTH_Z - 0.02, NORTH_Z - 0.005, TILE, "Splashback tiles"))
    return props, colliders


def living_room() -> tuple[list[Prop], list[AABB]]:
    props: list[Prop] = []
    colliders: list[AABB] = []
    # the sofa against the south wall, facing the TV: sagging, one cushion split
    props += [
        _box(-7.35, -5.35, 0.0, 0.42, SOUTH_Z + 0.02, 27.72, SOFA, "Sofa", solid=True),
        _box(-7.35, -5.35, 0.42, 0.9, SOUTH_Z + 0.02, 27.05, SOFA, "Sofa back", solid=True),
        _box(-7.5, -7.35, 0.0, 0.62, SOUTH_Z + 0.02, 27.72, SOFA, "Sofa arm", solid=True),
        _box(-5.35, -5.2, 0.0, 0.62, SOUTH_Z + 0.02, 27.72, SOFA, "Sofa arm", solid=True),
        _box(-6.95, -6.35, 0.42, 0.5, 27.1, 27.68, FABRIC, "Sofa cushion (split)"),
        _box(-6.8, -5.9, 0.5, 0.56, 27.0, 27.5, Material("fabric", (0.25, 0.35, 0.5)), "Blanket thrown on the sofa", rotate=((8.0, 0.0, 1.0, 0.0),)),
    ]
    colliders.append(AABB(-7.5, -5.2, SOUTH_Z, 27.75, 0.0, 0.9))
    # the coffee table: noodle cups, cans, an overflowing ashtray
    props += [
        _box(-6.95, -5.95, 0.38, 0.42, 28.25, 28.8, WOOD, "Coffee table", solid=True),
        _box(-6.9, -6.85, 0.0, 0.38, 28.3, 28.35, DARK, "Coffee table leg"),
        _box(-6.05, -6.0, 0.0, 0.38, 28.3, 28.35, DARK, "Coffee table leg"),
        _box(-6.9, -6.85, 0.0, 0.38, 28.7, 28.75, DARK, "Coffee table leg"),
        _box(-6.05, -6.0, 0.0, 0.38, 28.7, 28.75, DARK, "Coffee table leg"),
    ]
    colliders.append(AABB(-6.95, -5.95, 28.25, 28.8, 0.0, 0.42))
    for k, (x, z) in enumerate(((-6.7, 28.45), (-6.45, 28.62), (-6.2, 28.4))):
        props.append(Prop.cylinder(x, 0.42, z, 0.045, 0.05, 0.1, Material(None, (0.95, 0.92, 0.85)), slices=8, label="Empty noodle cup", tags=CLUTTER))
        if k == 1:
            props.append(Prop.cylinder(x + 0.18, 0.42, z - 0.12, 0.033, 0.033, 0.12, Material("soda_can", (1.0, 1.0, 1.0)), slices=8, label="Crushed can", tags=CLUTTER))
    props.append(Prop.cylinder(-6.05, 0.42, 28.68, 0.07, 0.06, 0.03, DARK, slices=10, label="Overflowing ashtray", tags=CLUTTER))
    # the TV on a cabinet by the kitchen wall, stuck on a dead channel
    props += [
        _box(-8.15, -7.35, 0.0, 0.5, 32.1, 32.58, WOOD, "TV cabinet", solid=True),
        _box(-8.05, -7.45, 0.5, 1.0, 32.38, 32.56, DARK, "Old flatscreen TV", solid=True),
        # (what is on it is up to Dell and his remote: ``render.television``)
        _box(-8.0, -7.5, 0.54, 0.96, 32.37, 32.38, TV_GLASS, "TV screen", tags=TV_TAGS),
    ]
    colliders.append(AABB(-8.2, -7.3, 32.05, KITCHEN_WALL_Z[0], 0.0, 1.0))
    # the space heater, glowing; a trash bag by the door; posters
    props += [
        _box(-5.0, -4.85, 0.0, 0.45, 28.55, 28.95, DARK, "Space heater", solid=True),
        _box(-5.01, -5.0, 0.08, 0.4, 28.6, 28.9, HEATER, "Space heater element", tags=DECAL),
    ]
    colliders.append(AABB(-5.05, FRONT_BACK_X, 28.5, 29.0, 0.0, 0.45))
    props.append(Prop.box(-5.15, 0.25, 32.3, 0.5, 0.5, 0.45, BAG, rotate=((20.0, 0.0, 1.0, 0.0),), label="Trash bag", tags=CLUTTER))
    props.append(_box(-8.19, -8.18, 1.25, 1.85, 28.9, 29.75, Material("poster_idol", (0.95, 0.9, 0.85)), "Poster: NEON IDOL RIN (curling at the corners)", tags=DECAL))
    props.append(_box(-6.75, -6.05, 1.35, 1.9, SOUTH_Z + 0.005, SOUTH_Z + 0.01, Material("poster_corp", (0.95, 0.95, 0.9)), "Poster: ARASAKA FOR A BETTER TOMORROW (defaced)", tags=DECAL))
    # a bare bulb on a cord; cables and a pipe along the ceiling
    props += [
        _box(-6.52, -6.48, 2.3, CEILING_Y, 29.78, 29.82, DARK, "Light cord"),
        Prop.box(-6.5, 2.25, 29.8, 0.08, 0.1, 0.08, BULB, label="Bare bulb"),
        _box(-8.1, -4.9, 2.6, 2.66, 32.45, 32.51, Material("metal", (0.35, 0.3, 0.25)), "Water pipe along the ceiling"),
        _box(-8.1, -4.9, 2.62, 2.64, 27.0, 27.02, DARK, "Cable along the ceiling"),
    ]
    return props, colliders


def kitchen() -> tuple[list[Prop], list[AABB]]:
    props: list[Prop] = []
    colliders: list[AABB] = []
    kz = KITCHEN_WALL_Z[1]
    # the counter along the back wall: cupboards (a door or two hanging off), a sink let
    # into the worktop with the dishes piled in it and a tap, the hotplate and a pot
    b = Builder(WorldFrame())
    front = b.wall("v", 35.0, -1.0)  # the counter's front (facing into the kitchen)
    back = b.wall("v", 35.58, -1.0)  # the splashback's face
    props += [
        _box(-9.6, -4.82, 0.0, 0.8, 35.0, NORTH_Z - 0.01, Material("wood", (0.45, 0.4, 0.32)), "Kitchen counter", solid=True),
    ]
    furnish.worktop_with_sink(b, -9.62, -4.8, 34.98, NORTH_Z, 0.8, 0.94, (-6.55, -5.95, 35.12, 35.48), Material("metal", (0.5, 0.52, 0.5)), label="Counter top")
    furnish.cabinet_fronts(front, -9.6, -4.82, 0.02, 0.78, Material("wood", (0.55, 0.48, 0.36)), Material("metal", (0.25, 0.25, 0.27)), label="Kitchen cupboard")
    furnish.tap(back, -6.25, 0.05, 0.94, reach=0.15, label="Kitchen tap")
    for k in range(5):  # the dishes piled in the sink: plates, a pot, a mug
        b.cylinder(-6.38, 35.3, 0.81 + 0.013 * k, 0.11, 0.11, 0.012, WHITE, "Dishes piled in the sink", slices=12)
    b.cylinder(-6.12, 35.26, 0.81, 0.09, 0.09, 0.11, METAL, "Pot in the sink (soaking)", slices=10)
    b.cylinder(-6.12, 35.26, 0.92, 0.085, 0.085, 0.003, furnish.WATER, "Grey water in the pot", slices=10)
    b.cylinder(-6.05, 35.42, 0.81, 0.04, 0.04, 0.1, Material(None, (0.3, 0.5, 0.8)), "Mug in the sink", slices=8)
    # cupboards on the wall, one either side of the hotplate
    for x0, x1 in ((-9.6, -8.9), (-7.5, -6.8)):
        furnish.wall_cabinet(back, x0, x1, 1.55, 2.2, Material("wood", (0.4, 0.36, 0.28)), Material("wood", (0.55, 0.48, 0.36)), Material("metal", (0.25, 0.25, 0.27)), label="Wall cupboard")
    # his supper: noodle cups, hot sauce, takeaway boxes, a bag of crisps
    furnish.counter_food(back, -7.65, -6.65, 0.3, 0.94, random.Random(16))
    furnish.bottle(back, -5.7, 0.12, 0.94, (0.75, 0.1, 0.05), "Bottle of hot sauce", h=0.18)
    for k in range(2):
        furnish.box_of(back, -5.35, 0.3, 0.94 + 0.07 * k, (0.12, 0.07, 0.1), Material(None, (0.92, 0.9, 0.85)), "Takeaway box")
    furnish.box_of(back, -5.05, 0.22, 0.94, (0.18, 0.22, 0.06), Material(None, (0.9, 0.8, 0.1)), "Bag of crisps")
    props += b.props
    props += [
        _box(-8.4, -7.7, 0.94, 0.99, 35.1, 35.5, DARK, "Hotplate"),
        Prop.cylinder(-8.05, 0.99, 35.3, 0.14, 0.14, 0.14, METAL, slices=10, label="Pot on the hotplate"),

    ]
    colliders.append(AABB(-9.65, -4.8, 34.95, NORTH_Z, 0.0, 0.94))
    colliders.append(AABB(-9.5, -8.95, 35.1, NORTH_Z, 0.94, 1.25))  # the microwave
    # the microwave (its clock never set), and the fridge in the corner: yellowed, rusting
    # at the foot, a sticker, magnets holding the bills
    b = Builder(WorldFrame())
    furnish.microwave(b.wall("v", 35.58, -1.0), -9.22, 0.94, Material(None, (0.75, 0.74, 0.68), specular=(0.4, 0.4, 0.4), shininess=30.0), width=0.46, depth=0.33, height=0.27, label="Microwave")
    furnish.fridge(b.wall("v", 34.88, -1.0), -10.75, -10.05, 1.8, 0.64, Material(None, (0.7, 0.68, 0.58), specular=(0.4, 0.4, 0.4), shininess=30.0), label="Dented fridge", rust=True)
    props += b.props
    props += [
        _box(-10.6, -10.35, 1.42, 1.62, 34.846, 34.849, Material(None, (0.9, 0.9, 0.2)), "Sticker on the fridge: NUKA-KOLA", tags=DECAL),
    ]
    colliders.append(AABB(-10.8, -10.0, 34.85, NORTH_Z, 0.0, 1.8))
    # a little table and a stool
    props += [
        _box(-9.9, -9.1, 0.7, 0.75, 33.2, 33.9, Material("metal", (0.4, 0.4, 0.42)), "Kitchen table", solid=True),
        Prop.cylinder(-9.5, 0.0, 33.55, 0.04, 0.04, 0.7, DARK, slices=8, label="Kitchen table leg"),
        Prop.cylinder(-9.5, 0.0, 34.3, 0.03, 0.03, 0.45, DARK, slices=8, label="Kitchen stool"),
        Prop.cylinder(-9.5, 0.45, 34.3, 0.16, 0.16, 0.04, Material("leather", (0.3, 0.25, 0.2)), slices=10, label="Kitchen stool seat"),
        _box(-9.8, -9.5, 0.75, 0.81, 33.3, 33.5, CARDBOARD, "Pizza box", tags=CLUTTER),
    ]
    colliders.append(AABB(-9.95, -9.05, 33.15, 33.95, 0.0, 0.75))
    colliders.append(AABB(-9.7, -9.3, 34.1, 34.5, 0.0, 0.5))
    # trash, the flickering tube
    props.append(Prop.box(-5.3, 0.25, kz + 0.45, 0.5, 0.5, 0.45, BAG, rotate=((-25.0, 0.0, 1.0, 0.0),), label="Trash bag", tags=CLUTTER))
    props.append(Prop.box(-5.75, 0.2, kz + 0.55, 0.45, 0.4, 0.4, BAG, rotate=((40.0, 0.0, 1.0, 0.0),), label="Trash bag", tags=CLUTTER))
    props.append(_box(-9.0, -6.0, 2.62, 2.66, 34.2, 34.3, TUBE, "Fluorescent tube"))
    return props, colliders


def bathroom() -> tuple[list[Prop], list[AABB]]:
    props: list[Prop] = []
    colliders: list[AABB] = []
    b = Builder(WorldFrame())
    back = b.wall("u", BACK_X + 0.02, 1.0)  # the back wall, at the face of its tiles
    south = b.wall("v", SOUTH_Z + 0.02, 1.0)  # the south wall, likewise
    # the toilet against the back wall (a seat gone yellow, the lid up), the paper by it
    furnish.toilet(b, back, 27.25, seat=(0.85, 0.8, 0.6), lid_up=True)
    furnish.toilet_roll(back, 27.78, 0.62)
    # the basin on its pedestal, the cracked mirror over it, a shelf of this and that
    furnish.pedestal_basin(
        b, south, -9.3, glass=Material("mirror_cracked", (1.0, 1.0, 1.0), specular=(1.0, 1.0, 1.0), shininess=90.0),
        mirror_label="Cracked mirror",
    )
    # a grey towel over a rail on the partition, by the door
    furnish.towel_rail(b.wall("u", PARTITION_X[0], -1.0), 28.45, 28.9, 1.3, (0.55, 0.55, 0.52))
    props += b.props
    colliders += b.colliders
    props += [
        # the shower: a tray, a curtain half drawn, mould
        _box(-10.78, -9.9, 0.0, 0.08, 28.05, 28.98, WHITE, "Shower tray"),
        _box(-9.92, -9.9, 0.4, 2.0, 28.05, 28.6, Material("cloth", (0.55, 0.7, 0.65)), "Shower curtain (mouldy)"),
        _box(-9.925, -9.895, 2.02, 2.045, 28.05, 28.97, furnish.CHROME, "Shower curtain rail"),
        _box(-10.6, -10.55, 1.9, 2.0, 28.45, 28.55, METAL, "Shower head"),
        _box(-10.78, -10.76, 1.1, 1.9, 28.485, 28.515, METAL, "Shower pipe"),
        Prop.cylinder(-8.75, 0.0, 28.7, 0.14, 0.16, 0.3, Material("metal", (0.3, 0.4, 0.5)), slices=10, label="Bucket", tags=CLUTTER),
    ]
    colliders += [
        AABB(-10.8, -9.88, 28.0, BATH_BED_Z[0], 0.0, 2.0),
    ]
    return props, colliders


NIGHTSTAND_TOP = 0.5
DATASTICK_AT = (-10.48, NIGHTSTAND_TOP, 30.72)


def bedroom() -> tuple[list[Prop], list[AABB]]:
    props: list[Prop] = []
    colliders: list[AABB] = []
    z0 = BATH_BED_Z[1]
    props += [
        # a mattress on pallets, sheets kicked off, a pillow
        _box(-10.75, -9.25, 0.0, 0.14, 31.0, 32.55, WOOD, "Pallets", solid=True),
        _box(-10.72, -9.28, 0.14, 0.32, 31.03, 32.52, Material("cloth", (0.75, 0.72, 0.68)), "Mattress (stained)", solid=True),
        _box(-10.6, -9.6, 0.32, 0.4, 32.05, 32.45, WHITE, "Pillow"),
        _box(-10.4, -9.35, 0.32, 0.38, 31.1, 31.9, Material("cloth", (0.3, 0.2, 0.45)), "Blanket, kicked off", rotate=((-12.0, 0.0, 1.0, 0.0),)),
        # the nightstand: a crate on end, a lamp, and the stick
        _box(-10.75, -10.25, 0.0, NIGHTSTAND_TOP, 30.4, 30.95, CRATE, "Crate used as a nightstand", solid=True),
        Prop.cylinder(-10.58, NIGHTSTAND_TOP, 30.5, 0.05, 0.05, 0.22, DARK, slices=8, label="Bedside lamp"),
        Prop.cylinder(-10.58, NIGHTSTAND_TOP + 0.2, 30.5, 0.1, 0.06, 0.1, Material(None, (1.0, 0.8, 0.5), emission=(1.3, 0.9, 0.45)), slices=10, label="Bedside lamp shade"),
        # clothes on the floor, a rail, a boarded window with the neon bleeding through
        _box(-9.2, -8.6, 0.0, 0.14, 29.3, 29.8, Material("cloth", (0.3, 0.3, 0.4)), "Pile of clothes", tags=CLUTTER),
        _box(-10.7, -10.66, 0.9, 2.0, 29.3, 30.2, DARK, "Clothes rail"),
        _box(-10.66, -10.5, 1.0, 1.8, 29.35, 30.15, Material("cloth", (0.25, 0.45, 0.35)), "Coats on the rail"),
        _box(BACK_X + 0.01, BACK_X + 0.03, 1.2, 2.0, 31.4, 32.4, Material(None, (0.06, 0.05, 0.07)), "Boarded window"),
    ]
    for k in range(4):  # light through the boards
        mat = NEON_PINK if k % 2 == 0 else NEON_TEAL
        props.append(_box(BACK_X + 0.03, BACK_X + 0.04, 1.32 + 0.18 * k, 1.36 + 0.18 * k, 31.45, 32.35, mat, "Neon through the window boards", tags=DECAL))
    colliders += [
        AABB(-10.8, -9.2, 30.95, KITCHEN_WALL_Z[0], 0.0, 0.4),
        AABB(-10.8, -10.2, 30.35, 31.0, 0.0, 0.72),
        AABB(-10.8, -10.45, 29.25, 30.25, 0.0, 2.0),
    ]
    del z0
    return props, colliders


WINDOW_Z = (27.17, 28.07)  # the living room's window onto the alley (z from, to)
GAME_BOXES = ((0.8, 0.1, 0.1), (0.1, 0.3, 0.8), (0.9, 0.8, 0.1), (0.1, 0.6, 0.3), (0.6, 0.2, 0.7))


def furnishings() -> tuple[list[Prop], list[AABB]]:
    """What makes it Dell's: a grimy window on the alley with its blinds bent (and the same
    window from outside, lit, barred), shelves of old games and junk, a calendar nobody
    has turned for a year, pizza boxes, a photo of somebody by the bed, a rug."""
    rng = random.Random(1616)
    b = Builder(WorldFrame())
    front = b.wall("u", FRONT_BACK_X, -1.0)  # the inside of the front wall
    furnish.window(front, 27.62, 1.15, 2.0, 0.95, "alley", Material("metal", (0.3, 0.3, 0.32)),
                   cover="blinds", cover_color=(0.62, 0.6, 0.55), drawn=0.55, label="Window")  # fmt: skip
    x0, x1 = FRONT_FACE_X, FRONT_FACE_X + 0.05  # (the alley side)
    b.add(x0, x1, *WINDOW_Z, 1.2, 1.95, Material(None, (0.8, 0.6, 0.35), emission=(0.42, 0.3, 0.15)), "Window of number 16")
    b.add(x0, x0 + 0.16, 27.1, 28.14, 1.13, 1.17, CEILING, "Window sill (outside)")
    for k in range(5):  # security bars
        z = 27.25 + k * 0.185
        b.add(x1 + 0.04, x1 + 0.06, z - 0.01, z + 0.01, 1.17, 2.0, DARK, "Window bar")
    furnish.picture(front, 29.38, 1.55, 0.34, 0.42, "calendar", "Calendar (last year's)", frame=None)
    # shelves on the partition by the TV: old games, a figure, cans, a snow globe
    part = b.wall("u", PARTITION_X[1], 1.0)
    for y in (1.45, 1.85):
        furnish.shelf(part, 31.4, 32.02, y, WOOD, depth=0.22)
    furnish.books(part, 31.44, 31.8, 1.45, rng, depth=0.14, tall=0.17, colours=GAME_BOXES, label="Game box")
    furnish.can(part, 31.9, 0.1, 1.45, "Empty can")
    furnish.box_of(part, 31.55, 0.1, 1.85, (0.08, 0.16, 0.06), Material(None, (0.9, 0.3, 0.1)), "Action figure")
    part.cylinder(31.78, 0.1, 1.85, 0.05, 0.05, 0.02, DARK, "Snow globe base", slices=10)
    part.cylinder(31.78, 0.1, 1.87, 0.045, 0.045, 0.08, Material(None, (0.7, 0.85, 0.95), specular=(1.0, 1.0, 1.0), shininess=100.0), "Snow globe", slices=10)
    # over the TV, a shelf of old films
    kitchen_wall = b.wall("v", KITCHEN_WALL_Z[0], -1.0)
    furnish.shelf(kitchen_wall, -8.1, -7.3, 1.45, WOOD, depth=0.2)
    furnish.books(kitchen_wall, -8.06, -7.6, 1.45, rng, depth=0.14, tall=0.19, label="Old film box")
    furnish.plant(kitchen_wall, -7.45, 0.1, 1.45, 0.55, rng, label="Dead plant")
    # pizza boxes on the floor by the sofa
    for k in range(3):
        b.add(-7.48 + 0.01 * k, -7.08 + 0.01 * k, 0.004 + 0.045 * k, 0.044 + 0.045 * k, 27.92, 28.32, CARDBOARD, "Pizza box", tags=CLUTTER)
    # the bedroom: a photo over the crate, a shelf of odds and ends, a rug by the mattress
    bed_back = b.wall("u", BACK_X, 1.0)
    furnish.picture(bed_back, 30.68, 1.45, 0.3, 0.25, "photo_family", "Photo (faded)", frame=furnish.FRAME_WOOD)
    bed_part = b.wall("u", PARTITION_X[0], -1.0)
    furnish.shelf(bed_part, 31.45, 32.45, 1.55, WOOD, depth=0.22)
    furnish.books(bed_part, 31.5, 31.95, 1.55, rng, depth=0.16, tall=0.22)
    furnish.bottle(bed_part, 32.1, 0.1, 1.55, (0.6, 0.4, 0.1), "Empty bottle of whisky", h=0.3)
    furnish.box_of(bed_part, 32.3, 0.1, 1.55, (0.16, 0.08, 0.12), Material(None, (0.12, 0.12, 0.13)), "Old radio")
    furnish.rug(b, -9.22, -8.5, 30.7, 31.9, "rug_stripes", (0.6, 0.55, 0.5))
    return b.props, b.colliders


def build() -> tuple[list[Prop], list[AABB]]:
    """Every prop and collider of the flat."""
    props: list[Prop] = []
    colliders: list[AABB] = []
    for part in (shell, living_room, kitchen, bathroom, bedroom, furnishings):
        p, c = part()
        props += p
        colliders += c
    return props, colliders


# ----------------------------------------------------------------------------- floor
WALK_REGIONS: tuple[WalkRegion, ...] = (
    WalkRegion(PARTITION_X[1], FRONT_BACK_X, SOUTH_Z, KITCHEN_WALL_Z[0]),  # living room
    WalkRegion(BACK_X, FRONT_BACK_X, KITCHEN_WALL_Z[1], NORTH_Z),  # kitchen
    WalkRegion(BACK_X, PARTITION_X[0], SOUTH_Z, BATH_BED_Z[0]),  # bathroom
    WalkRegion(BACK_X, PARTITION_X[0], BATH_BED_Z[1], KITCHEN_WALL_Z[0]),  # bedroom
    # the doorways between them
    WalkRegion(KITCHEN_GAP_X[0], KITCHEN_GAP_X[1], 32.1, 33.2, inset=frozenset({"min_x", "max_x"})),
    WalkRegion(-8.9, -7.6, *BATH_DOOR_Z, inset=frozenset({"min_z", "max_z"})),
    WalkRegion(-8.9, -7.6, *BED_DOOR_Z, inset=frozenset({"min_z", "max_z"})),
)

# ----------------------------------------------------------------------------- the door
HINGE = (FRONT_FACE_X + 0.05, DOOR_Z[0])  # the door swings in about its z = 29.9 edge
DOOR16 = DoorData(
    door_id="door16",
    obstacle=AABB(FRONT_BACK_X, FRONT_FACE_X + 0.1, DOOR_Z[0], DOOR_Z[1], 0.0, DOOR_HEIGHT),
    unlocked_by=None,
    travel=1.0,  # open fraction: 1 = swung right in
    speed=1.4,
    passable_at=0.7,
    # the doorway and the step outside it, walkable while the door is open
    passage=WalkRegion(-5.3, -4.2, DOOR_Z[0], DOOR_Z[1], inset=frozenset({"min_z", "max_z"})),
    faction=FACTION,
    approach_range=1.15,  # unlocked, it opens as somebody walks up to it
    inside="-x",  # and from the inside it always does (a latch)
    leaf=(
        Prop.box(FRONT_FACE_X + 0.05, DOOR_HEIGHT / 2.0, sum(DOOR_Z) / 2.0, 0.1, DOOR_HEIGHT, DOOR_Z[1] - DOOR_Z[0], DOOR_MAT, label="Front door of number 16"),
        Prop.box(FRONT_FACE_X + 0.05, 0.12, sum(DOOR_Z) / 2.0, 0.11, 0.2, DOOR_Z[1] - DOOR_Z[0] - 0.04, Material("metal", (0.3, 0.31, 0.34)), label="Kick plate"),
        Prop.box(FRONT_FACE_X + 0.12, 1.0, DOOR_Z[1] - 0.14, 0.04, 0.04, 0.12, Material(None, (0.75, 0.7, 0.5)), label="Door handle (outside)"),
        Prop.box(FRONT_FACE_X - 0.02, 1.0, DOOR_Z[1] - 0.14, 0.04, 0.04, 0.12, Material(None, (0.75, 0.7, 0.5)), label="Door handle (inside)"),
        Prop.box(FRONT_FACE_X + 0.105, 1.55, sum(DOOR_Z) / 2.0, 0.012, 0.04, 0.04, DARK, label="Peephole"),
    ),
    hinge=HINGE,
    swing=-85.0,
)  # fmt: skip

PANEL = PanelData(
    panel_id="panel16", x=FRONT_FACE_X + 0.25, y=1.3, z=31.62, door_id="door16", yaw=90.0,
    faction=FACTION, name="NO.16 ACCESS PANEL",
)  # fmt: skip

CAMERA = CameraData(
    -7.95, 2.5, 27.05, mount_yaw=31.0, sweep_angle=32.0, sweep_speed=0.7,
    camera_id="home_camera_16", faction=FACTION, watch_area="flat16",
)  # fmt: skip

TERMINAL = TerminalData(
    "dells_computer", -5.15, 0.0, 31.9, yaw=-90.0, faction=FACTION, style="home",
    notes="NOTE TO SELF: the stick's not safe plugged into this thing - anyone on the "
    "net could read it. Moved it to the bedroom. Nightstand. -D",
)  # fmt: skip

DATASTICK = PickupData("datastick16", "datastick", *DATASTICK_AT, yaw=25.0)

FLAT16 = PrivateArea(
    area_id="flat16",
    name="NUMBER 16",
    boxes=(AABB(BACK_X, FRONT_FACE_X, SOUTH_Z, NORTH_Z, 0.0, CEILING_Y),),
    exit=(-2.6, 30.5, 0.0),  # out on the alley, by the step (the player's yaw: +x... ahead)
    door_id="door16",
    confiscate=("datastick",),
    owner_role="resident",
)

# The resident's places inside (see ``ai.alley.resident``) and how they connect: straight
# walks that keep clear of the furniture and go through the doorways.
PLACES = (
    Place("apt.entry", -5.4, 30.5, yaw=-90.0),
    Place("apt.living", -6.6, 30.4, yaw=-90.0),
    Place("apt.by_sofa", -5.5, 29.3, yaw=180.0),
    Place("apt.sofa", -5.8, 27.4, yaw=0.0),  # on the seat (in front: z + 0.7)
    Place("apt.desk", -5.95, 31.9, yaw=90.0),
    Place("apt.kitchen_gap", -6.4, 32.66, yaw=0.0),
    Place("apt.kitchen", -8.05, 34.45, yaw=0.0),
    Place("apt.sink", -6.25, 34.45, yaw=0.0),
    Place("apt.bed_gap", -8.26, 30.85, yaw=-90.0),
    Place("apt.bedroom", -9.1, 30.75, yaw=-90.0),
    Place("apt.rail", -9.9, 29.8, yaw=-90.0),
    Place("apt.bedside", -8.95, 31.7, yaw=-90.0),  # beside the mattress
    # (lying on it corner to corner, head on the pillow: the mattress is a short one)
    Place("apt.bed", -10.55, 32.42, yaw=-42.0, y=0.4),  # the top of his head on the pillow
    Place("apt.bath_gap", -8.26, 27.95, yaw=-90.0),
    Place("apt.bath", -9.3, 27.75, yaw=180.0),
)
ROUTES: tuple[tuple[str, str], ...] = (
    ("apt.entry", "apt.living"),
    ("apt.living", "apt.by_sofa"),  # (onto the sofa itself is a scripted sit-down)
    ("apt.living", "apt.desk"),
    ("apt.living", "apt.kitchen_gap"),
    ("apt.kitchen_gap", "apt.kitchen"),
    ("apt.kitchen_gap", "apt.sink"),
    ("apt.living", "apt.bed_gap"),
    ("apt.bed_gap", "apt.bedroom"),
    ("apt.bedroom", "apt.rail"),
    ("apt.bedroom", "apt.bedside"),
    ("apt.living", "apt.bath_gap"),
    ("apt.bath_gap", "apt.bath"),
)

LIGHTS = (
    Light(  # the bare bulb in the living room (it reaches the kitchen through the opening)
        -6.5, 2.2, 30.6,
        ambient=(0.09, 0.075, 0.05), diffuse=(1.5, 1.2, 0.75), specular=(0.8, 0.7, 0.5),
        linear_attenuation=0.3, quadratic_attenuation=0.12,
        fixture="none",
    ),
    Light(  # the bedside lamp (the bathroom gets what comes round the door)
        -10.2, 1.0, 30.6,
        ambient=(0.07, 0.05, 0.03), diffuse=(1.3, 0.95, 0.55), specular=(0.6, 0.5, 0.35),
        linear_attenuation=0.45, quadratic_attenuation=0.2,
        fixture="none",
    ),
)  # fmt: skip
