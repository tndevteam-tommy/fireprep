"""The second level: the night-market alley outside the facility.

Out of the Halcyon building's side door and into a narrow alley between two tenements -
wet asphalt, overhead cables strung with paper lanterns, a sushi-and-wok bar doing its
evening trade, a junk dealer selling cheap cyberware, doors and lit windows of the homes
above, dumpsters, a parked bike, and people going about their business: the chef and the
dealer at their stalls, and an office worker, a tinkerer, a courier, the resident of
number 16, a smoker and a stroller, each with a story of their own that brings them to
the stalls, the doors and each other (see ``ai.alley`` and the ``places`` below). They
see and step around one another (and the player). The far end opens onto the main
street (``levels.main_street``): a pavement along the blocks' faces with everybody else's
front doors on it, a car at the kerb, and barricades at both ends - there is no winning,
and no walking off into a city that is not built yet.

Layout (x across the alley, z along it; walking up the alley from the door, +x is on
your left and -x on your right):

      z=52   [main-street backdrop: towers, signs, shopfronts]
      z=42.5 the alley mouth: the main street's pavement, left and right
      z=36   x>0: home door 21 (stoop)           x<0: dumpster
      z=31   x>0: parked bike                    x<0: home door 16 (stoop, lamp)
      z=25   x>0: CYBERWARE junk stall (the dealer)
      z=17.6 x>0: NEON COLD vending machine
      z=14   x<0: SUSHI & WOK food stall (the chef)
      z=6    x>0: dumpster                       x<0: home door 14 (stoop, lamp)
      z=1.6                                      x<0: number 17, Nix's (lamp)
      z=-2.5 the facility's outer wall with the yellow door you came out of
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

from cyberduke.core.geometry import AABB
from cyberduke.levels import apartment, bar, flats, main_street, metro
from cyberduke.levels.doorway import (
    ease_doorways,
    staff_door_alarm,
    staff_door_leaf,
    staff_door_trim,
)
from cyberduke.levels.dressing import Placer, scatter_trash
from cyberduke.world.level import (
    CivilianData,
    FactionData,
    ItemData,
    LevelData,
    MerchantData,
    OpenSky,
    Place,
    SectorData,
    VendingMachineData,
    WalkRegion,
)
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop
from cyberduke.world.surfaces import Catwalk

# ----------------------------------------------------------------------------- materials
ASPHALT = Material("asphalt", (0.27, 0.29, 0.35), specular=(0.5, 0.55, 0.65), shininess=40.0)
BRICK_L = Material("brick", (0.12, 0.12, 0.13))
BRICK_R = Material("brick", (0.09, 0.11, 0.14))
CONCRETE = Material("concrete", (0.1, 0.1, 0.115))
STEEL = Material("metal", (0.3, 0.31, 0.34))
DARK_STEEL = Material("metal", (0.12, 0.12, 0.14))
RUST = Material("metal", (0.4, 0.25, 0.16))
WINDOW_DARK = Material(None, (0.08, 0.09, 0.12))
WINDOW_WARM = Material(None, (0.9, 0.7, 0.4), emission=(0.55, 0.4, 0.2))
WINDOW_COOL = Material(None, (0.5, 0.75, 1.0), emission=(0.25, 0.4, 0.6))
WINDOW_TV = Material(None, (0.55, 0.7, 1.0), emission=(0.3, 0.45, 0.7))
CURTAIN = Material("fabric", (0.6, 0.5, 0.4))
DOOR = Material("leather", (0.55, 0.36, 0.24))
DOOR_FRAME = Material("metal", (0.3, 0.3, 0.33))
LAMP = Material(None, (1.0, 0.85, 0.55), emission=(1.2, 0.9, 0.5))
LAMP_GLOW = Material(None, (1.0, 0.8, 0.45), alpha=0.35)
LANTERN_RED = Material(None, (1.0, 0.35, 0.2), emission=(1.6, 0.4, 0.2))
LANTERN_GLOW = Material(None, (1.0, 0.45, 0.25), alpha=0.3)
CABLE = Material(None, (0.06, 0.06, 0.07))
AWNING_A = Material("hazard", (0.95, 0.35, 0.3))
AWNING_B = Material("fabric", (0.2, 0.35, 0.6))
COUNTER = Material("metal", (0.55, 0.55, 0.58), specular=(0.6, 0.6, 0.65), shininess=50.0)
WOOD = Material("wood", (0.8, 0.65, 0.45))
NEON_MAGENTA = Material("neon_magenta", (1.0, 0.3, 0.8), emission=(2.4, 0.5, 1.8))
NEON_CYAN = Material("neon_cyan", (0.2, 0.9, 1.0), emission=(0.5, 2.2, 2.6))
NEON_AMBER = Material("hazard", (1.0, 0.75, 0.2), emission=(2.0, 1.3, 0.3))
NEON_GREEN = Material(None, (0.12, 0.45, 0.15), emission=(0.3, 2.2, 0.45))  # dark diffuse: lit, it stays green
STEAM = Material(None, (0.85, 0.88, 0.9), alpha=0.18)
DUMPSTER = Material("metal", (0.16, 0.3, 0.22))
TARP = Material("fabric", (0.25, 0.3, 0.5))
TIRE = Material(None, (0.08, 0.08, 0.09))
BIKE = Material("metal", (0.6, 0.15, 0.15), specular=(0.7, 0.7, 0.7), shininess=60.0)
DRAIN = Material("grate", (0.25, 0.26, 0.3))
PUDDLE = Material(None, (0.14, 0.2, 0.3), alpha=0.5)
TRASH_BAG = Material("leather", (0.3, 0.3, 0.34), specular=(0.4, 0.4, 0.45), shininess=30.0)
CLUTTER = frozenset({"clutter", "transparent"})
DECAL = frozenset({"transparent"})

ALLEY_HALF = 4.5  # building faces at x = +-4.5
ALLEY_END = 42.0  # where it opens onto the main street
STREET_FAR = main_street.ROAD_Z1 + 1.0  # the far pavement of the main street (a backdrop)
TUNNEL_END = main_street.TUNNEL_X + main_street.TUNNEL_DEPTH + 2.0  # (the tunnels' far ends)
WALL_H = 12.0
FACILITY_WALL_FACE_Z = -2.5  # the alley face of the Halcyon building's outer wall


# ----------------------------------------------------------------------------- buildings
def _windows(face_x: float, side: int, seed: int) -> list[Prop]:
    """Rows of windows up a tenement face: lit warm, lit cool, a TV flicker, or dark; a few
    with curtains, a few with an air conditioner hung under them."""
    props: list[Prop] = []
    rng = random.Random(seed)
    x = face_x - side * 0.02  # just proud of the wall face, on the alley side
    for floor in range(3):
        y = 3.4 + floor * 2.8
        for k in range(12):
            z = 0.8 + k * 3.4 + (floor % 2) * 1.1
            if z > ALLEY_END + 2.0:
                continue
            roll = rng.random()
            mat = WINDOW_DARK if roll < 0.45 else (WINDOW_WARM if roll < 0.75 else (WINDOW_COOL if roll < 0.9 else WINDOW_TV))
            props.append(Prop.box(x, y, z, 0.05, 1.1, 0.8, mat, label="Tenement window"))
            props.append(Prop.box(face_x - side * 0.04, y + 0.62, z, 0.12, 0.06, 0.95, DARK_STEEL, label="Window lintel"))
            props.append(Prop.box(face_x - side * 0.06, y - 0.6, z, 0.16, 0.05, 0.95, CONCRETE, label="Window sill"))
            if mat is not WINDOW_DARK and rng.random() < 0.35:
                props.append(Prop.box(x - side * 0.01, y, z + 0.25, 0.04, 1.0, 0.29, CURTAIN, label="Curtain"))
            if rng.random() < 0.3:
                props.append(Prop.box(face_x - side * 0.3, y - 0.95, z - 0.1, 0.55, 0.4, 0.5, STEEL, label="Air conditioner unit"))
    return props


def _facade(
    side: int, doors: tuple[tuple[float, float], ...] = (), flats: tuple[tuple[float, float], ...] = ()
) -> tuple[list[Prop], list[AABB]]:
    """One side of the alley: the tenement's wall, windows, pipes, a drain; the concrete
    skirting along its foot stops either side of each door (``doors``: (z, half the gap)) -
    it used to run straight across them, hiding all but the top of every front door.
    ``flats`` ((z from, z to), ...): ground-floor flats behind the wall there (number 16,
    ``levels.apartment``; number 17, ``levels.flats``) - the wall's lower storey is each
    flat's own front wall."""
    face_x = side * ALLEY_HALF
    brick = BRICK_L if side < 0 else BRICK_R
    skirting = []
    z = -2.5
    cuts = tuple(doors) + tuple(c for f0, f1 in flats for c in ((f0, 0.0), (f1, 0.0)))
    spans = sorted(flats)
    for dz, half in (*sorted(cuts), (ALLEY_END + 0.5 + 0.75, 0.75)):
        end = dz - half
        if end - z > 0.05:
            skirting.append(Prop.box(face_x - side * 0.1, 0.9, (z + end) / 2.0, 0.2, 1.8, end - z, CONCRETE, solid=True, label="Wall base skirting"))
        z = dz + half
    wall_x, wall_z0, wall_z1 = face_x + side * 0.6, -3.0, ALLEY_END + 0.5
    label = f"Tenement wall (x {'<' if side < 0 else '>'} 0)"
    top = apartment.SHELL_TOP
    walls = []
    z = wall_z0
    for f0, f1 in (*spans, (wall_z1, wall_z1)):
        if f0 - z > 0.01:
            walls.append(Prop.box(wall_x, WALL_H / 2.0, (z + f0) / 2.0, 1.2, WALL_H, f0 - z, brick, solid=True, label=label))
        if f1 > f0:  # the upper storeys over the flat (its own front wall is the storey below)
            walls.append(Prop.box(wall_x, (top + WALL_H) / 2.0, (f0 + f1) / 2.0, 1.2, WALL_H - top, f1 - f0, brick, solid=True, label=label))
        z = max(z, f1)
    props: list[Prop] = [
        *walls,
        *skirting,
        Prop.cylinder(face_x - side * 0.16, 0.0, 3.0 if side < 0 else 21.0, 0.07, 0.07, WALL_H - 0.5, RUST, slices=8, label="Downpipe", tags=DECAL),
        Prop.cylinder(face_x - side * 0.16, 0.0, 38.5 if side < 0 else 12.0, 0.07, 0.07, WALL_H - 0.5, RUST, slices=8, label="Downpipe", tags=DECAL),
        Prop.box(face_x - side * 0.4, 0.006, (-1.0 + ALLEY_END + 0.4) / 2.0, 0.48, 0.01, ALLEY_END + 1.4, DRAIN, label="Gutter drain grate"),
    ]
    props.extend(_windows(face_x, side, seed=7 if side < 0 else 13))
    near, far = face_x - side * 0.2, face_x + side * 1.3  # skirting face .. back of the wall
    lo, hi = min(near, far), max(near, far)
    colliders = []
    z = -4.0
    for f0, f1 in (*spans, (ALLEY_END + 0.5, ALLEY_END + 0.5)):
        if f0 > z:
            colliders.append(AABB(lo, hi, z, f0, 0.0, WALL_H))
        if f1 > f0:
            colliders.append(AABB(lo, hi, f0, f1, top, WALL_H))  # (below: the flat's front wall)
        z = max(z, f1)
    return props, colliders


STOOP_DEPTH = 0.9  # room to stand on a doorstep (and to put a parcel down beside the door)


def _home_door(
    side: int, z: float, lamp: bool, number: str, width: float = 1.0, real: bool = False
) -> tuple[list[Prop], list[AABB], list[AABB], list[AABB]]:
    """A front door in the wall with a frame, a stoop, a number plate and (sometimes) a
    lamp over it: (props, colliders, NPC-only no-go boxes, platforms). A ``real`` door
    leads somewhere (number 16 - ``levels.apartment``): its leaf is a door of the level
    that swings open (``DoorData.hinge``), and its stoop is a platform the player can step
    up on (NPCs are scripted onto it, so for them it is a no-go box); any other is a prop
    on a solid stoop."""
    face_x = side * ALLEY_HALF
    half = width / 2.0
    step_half = half + 0.25
    props = [
        Prop.box(face_x - side * 0.1, 2.28, z, 0.2, 0.16, width + 0.24, DOOR_FRAME, label="Door frame head"),
        Prop.box(face_x - side * 0.1, 1.1, z - half - 0.06, 0.2, 2.2, 0.12, DOOR_FRAME, label="Door frame post"),
        Prop.box(face_x - side * 0.1, 1.1, z + half + 0.06, 0.2, 2.2, 0.12, DOOR_FRAME, label="Door frame post"),
        Prop.box(face_x - side * STOOP_DEPTH / 2.0, 0.08, z, STOOP_DEPTH, 0.16, 2.0 * step_half, CONCRETE, solid=not real, label=f"Stoop of door {number}"),
        # the number on the wall beside the door (on the leaf it swung away with it)
        Prop.box(face_x - side * 0.01, 1.95, z + half + 0.35, 0.02, 0.12, 0.2, Material(None, (0.8, 0.75, 0.5), emission=(0.3, 0.28, 0.15)), label=f"Door number {number}"),
    ]
    if not real:
        props.insert(0, Prop.box(face_x - side * 0.05, 1.1, z, 0.1, 2.2, width, DOOR, label=f"Front door {number}"))
    if lamp:
        props.append(Prop.box(face_x - side * 0.2, 2.55, z, 0.3, 0.12, 0.3, LAMP, label="Door lamp", tags=DECAL))
        props.append(Prop.disc(face_x - side * 0.55, 0.004, z, 1.1, 1.3, 0.3, LAMP_GLOW, slices=18, label="Door lamp light on the pavement", tags=frozenset({"glow", "transparent"})))
    edge = face_x - side * STOOP_DEPTH
    stoop = AABB(min(face_x, edge), max(face_x, edge), z - step_half, z + step_half, 0.0, 0.16)
    if real:
        return props, [], [stoop], [stoop]
    return props, [stoop], [], []


# ----------------------------------------------------------------------------- stalls
def _food_stall() -> tuple[list[Prop], list[AABB]]:
    """SUSHI & WOK: a narrow counter along the x<0 wall with the chef's lane behind it and
    a bench against the wall - the wok on its burner, a prep bench with the rice cooker and
    a fish cooler. On the counter the chopping board with its knife and a stack of takeaway
    boxes; two stools on the customers' side; lanterns, a menu board, two neon signs."""
    z0, z1 = 12.0, 16.0
    zc = (z0 + z1) / 2.0
    props: list[Prop] = [
        Prop.box(-3.0, 0.48, zc, 0.5, 0.96, z1 - z0, COUNTER, solid=True, label="Sushi counter"),
        Prop.box(-3.0, 0.985, zc, 0.6, 0.05, z1 - z0 + 0.1, STEEL, label="Counter top"),
        Prop.box(-2.74, 0.45, zc, 0.03, 0.8, z1 - z0 - 0.2, Material("hazard", (0.9, 0.3, 0.25)), solid=True, label="Counter front panel (red stripes)"),
        # awning on two poles, striped, tilted down toward the alley
        Prop.box(-2.95, 2.65, zc, 2.4, 0.06, z1 - z0 + 0.8, AWNING_A, rotate=((0.0, 0.0, 1.0, 0.0), (-14.0, 0.0, 0.0, 1.0)), label="Food stall awning"),
        Prop.cylinder(-2.0, 0.0, z0 - 0.2, 0.04, 0.04, 2.6, STEEL, slices=8, solid=True, label="Awning pole"),
        Prop.cylinder(-2.0, 0.0, z1 + 0.2, 0.04, 0.04, 2.6, STEEL, slices=8, solid=True, label="Awning pole"),
        # the bench against the wall, behind the chef's lane (x -3.95 .. -3.3)
        Prop.box(-4.22, 0.45, 13.0, 0.55, 0.9, 1.1, DARK_STEEL, solid=True, label="Burner cabinet"),
        # the wok: a burner ring on the cabinet, the bowl on it - narrow at the bottom,
        # flaring out to the rim (it used to stand on its rim, upside down) - the chow mein
        # heaped in it and a long handle out towards the chef
        Prop.cylinder(-4.22, 0.9, 13.0, 0.16, 0.16, 0.03, Material("metal", (0.08, 0.08, 0.09)), slices=12, label="Wok burner"),
        Prop.cylinder(-4.22, 0.93, 13.0, 0.09, 0.2, 0.06, DARK_STEEL, slices=14, label="Wok"),
        Prop.cylinder(-4.22, 0.99, 13.0, 0.2, 0.29, 0.07, DARK_STEEL, slices=14, label="Wok (the flare to the rim)"),
        Prop.cylinder(-4.22, 1.06, 13.0, 0.2, 0.12, 0.035, Material(None, (0.9, 0.6, 0.2), emission=(0.9, 0.5, 0.15)), slices=12, label="Noodles in the wok"),
        Prop.box(-3.84, 1.08, 13.0, 0.22, 0.025, 0.035, Material("wood", (0.3, 0.2, 0.12)), rotate=((0.0, 0.0, 1.0, 0.0), (12.0, 0.0, 0.0, 1.0)), label="Wok handle"),
        Prop.box(-4.22, 0.45, 14.05, 0.55, 0.9, 0.9, STEEL, solid=True, label="Prep bench"),
        Prop.cylinder(-4.25, 0.9, 14.05, 0.14, 0.13, 0.22, Material(None, (0.9, 0.9, 0.88)), slices=12, label="Rice cooker"),
        Prop.box(-4.22, 0.375, 15.05, 0.55, 0.75, 1.0, Material("metal", (0.75, 0.8, 0.86)), solid=True, label="Fish cooler"),
        Prop.box(-3.945, 0.55, 15.05, 0.01, 0.1, 0.8, Material(None, (0.3, 0.7, 1.0), emission=(0.2, 0.5, 0.8)), label="Fish cooler ICE stripe", tags=DECAL),
        Prop.box(-4.3, 1.7, zc, 0.25, 0.06, 3.6, STEEL, label="Back shelf"),
        Prop.box(-4.3, 2.3, zc, 0.25, 0.06, 3.6, STEEL, label="Back shelf"),
        # on the counter: the chopping board (the knife lies on it - see the level's
        # items) and a stack of takeaway boxes, both at the back edge by the chef
        Prop.box(-3.1, 1.025, 14.2, 0.3, 0.03, 0.45, WOOD, label="Chopping board"),
        Prop.box(-3.12, 1.045, 14.72, 0.22, 0.07, 0.16, Material("cardboard", (0.9, 0.8, 0.62)), label="Takeaway boxes"),
        Prop.box(-3.12, 1.115, 14.72, 0.21, 0.07, 0.15, Material("cardboard", (0.9, 0.8, 0.62)), rotate=((8.0, 0.0, 1.0, 0.0),), label="Takeaway boxes"),
        Prop.box(-2.95, 1.05, z0 + 0.4, 0.2, 0.08, 0.3, Material(None, (0.85, 0.8, 0.3)), label="Napkin box"),
        # menu board and signs on the wall
        Prop.box(-4.44, 2.0, zc + 1.0, 0.04, 0.9, 1.4, Material(None, (0.12, 0.12, 0.14)), label="Menu board"),
        Prop.box(-4.42, 2.15, zc + 1.0, 0.01, 0.08, 1.1, NEON_AMBER, label="Menu line: NIGIRI 6 ED", tags=DECAL),
        Prop.box(-4.42, 1.98, zc + 1.0, 0.01, 0.08, 0.9, NEON_GREEN, label="Menu line: CHOW MEIN 4 ED", tags=DECAL),
        Prop.box(-4.42, 1.81, zc + 1.0, 0.01, 0.08, 1.0, NEON_CYAN, label="Menu line: DUMPLINGS 6/5 ED", tags=DECAL),
        Prop.box(-4.4, 3.4, zc, 0.08, 0.55, 3.2, NEON_MAGENTA, label="SUSHI & WOK neon sign", tags=DECAL),
        Prop.box(-4.38, 2.95, zc - 1.2, 0.06, 0.3, 1.0, NEON_AMBER, label="NIGIRI neon", tags=DECAL),
        # lanterns under the awning, and the light they throw
        Prop.box(-2.4, 2.2, z0 + 0.6, 0.22, 0.3, 0.22, LANTERN_RED, label="Paper lantern", tags=DECAL),
        Prop.box(-2.4, 2.2, zc, 0.22, 0.3, 0.22, LANTERN_RED, label="Paper lantern", tags=DECAL),
        Prop.box(-2.4, 2.2, z1 - 0.6, 0.22, 0.3, 0.22, LANTERN_RED, label="Paper lantern", tags=DECAL),
        Prop.disc(-2.2, 0.004, zc, 2.2, 2.8, 0.25, LANTERN_GLOW, slices=20, label="Lantern light on the pavement", tags=frozenset({"glow", "transparent"})),
        # jars on the shelf
        Prop.cylinder(-4.3, 1.73, zc - 1.2, 0.08, 0.08, 0.2, Material(None, (0.8, 0.2, 0.1)), slices=8, label="Chilli oil jar"),
        Prop.cylinder(-4.3, 1.73, zc - 0.8, 0.07, 0.07, 0.18, Material(None, (0.2, 0.6, 0.2)), slices=8, label="Pickled ginger jar"),
        Prop.cylinder(-4.3, 1.73, zc + 0.4, 0.06, 0.06, 0.16, Material(None, (0.2, 0.15, 0.12)), slices=8, label="Soy sauce bottle"),
        # stools at the counter
        Prop.cylinder(-2.3, 0.0, z0 + 0.5, 0.05, 0.05, 0.55, DARK_STEEL, slices=8, solid=True, label="Stool"),
        Prop.cylinder(-2.3, 0.55, z0 + 0.5, 0.2, 0.2, 0.06, RUST, slices=10, label="Stool seat"),
        Prop.cylinder(-2.3, 0.0, z1 - 0.6, 0.05, 0.05, 0.55, DARK_STEEL, slices=8, solid=True, label="Stool"),
        Prop.cylinder(-2.3, 0.55, z1 - 0.6, 0.2, 0.2, 0.06, RUST, slices=10, label="Stool seat"),
    ]
    # steam off the wok: puffs that fade at the edges, stacked up and out
    for k in range(4):
        props.append(Prop.disc(-4.22 + 0.1 * k, 1.15 + 0.3 * k, 13.0 + 0.05 * k, 0.35 + 0.12 * k, 0.3 + 0.1 * k, 0.2, STEAM, slices=14, label="Steam off the wok", tags=frozenset({"fog", "transparent"})))
    colliders = [
        AABB(-4.5, -2.7, z0 - 0.4, z1 + 0.4, 0.0, 2.5),  # counter, the chef's lane and the bench
        AABB(-2.55, -2.05, z0 + 0.25, z0 + 0.75, 0.0, 0.6),  # stools
        AABB(-2.55, -2.05, z1 - 0.85, z1 - 0.35, 0.0, 0.6),
        AABB(-2.1, -1.9, z0 - 0.3, z0 - 0.1, 0.0, 2.6),  # awning poles
        AABB(-2.1, -1.9, z1 + 0.1, z1 + 0.3, 0.0, 2.6),
    ]
    return props, colliders


def _junk_stall() -> tuple[list[Prop], list[AABB]]:
    """CYBERWARE CHEAP: a narrow trestle table under a tarp on the x>0 wall, heaped with
    chips, optics, decks and cables, the dealer's lane behind it and crates of stock against
    the wall."""
    z0, z1 = 23.8, 27.2
    zc = (z0 + z1) / 2.0
    props: list[Prop] = [
        Prop.box(3.0, 0.86, zc, 0.6, 0.06, z1 - z0, WOOD, label="Trestle table top"),
        Prop.box(2.75, 0.42, z0 + 0.3, 0.06, 0.84, 0.06, DARK_STEEL, label="Trestle leg"),
        Prop.box(3.25, 0.42, z0 + 0.3, 0.06, 0.84, 0.06, DARK_STEEL, label="Trestle leg"),
        Prop.box(2.75, 0.42, z1 - 0.3, 0.06, 0.84, 0.06, DARK_STEEL, label="Trestle leg"),
        Prop.box(3.25, 0.42, z1 - 0.3, 0.06, 0.84, 0.06, DARK_STEEL, label="Trestle leg"),
        Prop.box(3.0, 0.86, zc, 0.56, 0.07, z1 - z0 - 0.1, Material("fabric", (0.3, 0.1, 0.12)), label="Velvet cloth on the table"),
        Prop.box(3.2, 2.5, zc, 2.8, 0.05, z1 - z0 + 1.0, TARP, rotate=((9.0, 0.0, 0.0, 1.0),), label="Tarp canopy"),
        Prop.cylinder(1.95, 0.0, z0 - 0.3, 0.035, 0.035, 2.4, STEEL, slices=8, solid=True, label="Tarp pole"),
        Prop.cylinder(1.95, 0.0, z1 + 0.3, 0.035, 0.035, 2.4, STEEL, slices=8, solid=True, label="Tarp pole"),
        Prop.box(4.4, 3.05, zc, 0.08, 0.4, 2.6, NEON_CYAN, label="CYBERWARE neon sign", tags=DECAL),
        Prop.box(4.4, 2.55, zc + 0.5, 0.06, 0.22, 1.2, NEON_GREEN, label="CHEAP neon", tags=DECAL),
        Prop.box(4.15, 0.5, zc - 1.0, 0.6, 1.0, 0.8, Material("cardboard", (0.7, 0.62, 0.5)), solid=True, label="Stock crate behind the stall"),
        Prop.box(4.2, 1.3, zc - 1.0, 0.5, 0.6, 0.6, Material("crate", (0.85, 0.85, 0.85)), solid=True, label="Stock crate, stacked"),
        Prop.box(4.15, 0.35, zc + 0.9, 0.6, 0.7, 0.6, Material("crate", (0.85, 0.85, 0.85)), solid=True, label="Stock crate behind the stall"),
        Prop.disc(2.6, 0.004, zc, 1.8, 2.4, 0.3, Material(None, (0.3, 0.9, 1.0), alpha=0.3), slices=20, label="Neon light on the pavement", tags=frozenset({"glow", "transparent"})),
    ]
    rng = random.Random(99)
    for k in range(16):  # the merchandise
        tx = 2.8 + rng.random() * 0.4
        tz = z0 + 0.3 + rng.random() * (z1 - z0 - 0.6)
        kind = k % 4
        if kind == 0:
            props.append(Prop.box(tx, 0.9, tz, 0.12, 0.015, 0.08, Material(None, (1.0, 0.7, 0.0), emission=(2.0, 1.2, 0.0)), rotate=((rng.uniform(0, 360), 0.0, 1.0, 0.0),), label="Datachip for sale"))
        elif kind == 1:
            props.append(Prop.cylinder(tx, 0.895, tz, 0.05, 0.05, 0.04, Material(None, (0.9, 0.2, 0.2), emission=(0.9, 0.1, 0.1)), slices=8, label="Optic implant for sale"))
        elif kind == 2:
            props.append(Prop.box(tx, 0.92, tz, 0.16, 0.05, 0.1, DARK_STEEL, rotate=((rng.uniform(0, 360), 0.0, 1.0, 0.0),), label="Cyberdeck for sale"))
        else:
            props.append(Prop.box(tx, 0.905, tz, 0.03, 0.02, 0.3, CABLE, rotate=((rng.uniform(0, 360), 0.0, 1.0, 0.0),), label="Coiled cable for sale"))
    colliders = [
        AABB(2.65, 4.5, z0 - 0.3, z1 + 0.3, 0.0, 2.0),  # table, the dealer's lane and the stock
        AABB(1.85, 2.05, z0 - 0.4, z0 - 0.2, 0.0, 2.4),
        AABB(1.85, 2.05, z1 + 0.2, z1 + 0.4, 0.0, 2.4),
    ]
    return props, colliders


# ----------------------------------------------------------------------------- street furniture
def _local(x: float, z: float, yaw: float):
    """Maps a point in an object's own frame to the world, turning it with the object (the
    same handedness as the props' glRotatef-style yaw about +Y)."""
    rad = math.radians(yaw)
    cos_y, sin_y = math.cos(rad), math.sin(rad)

    def at(lx: float, lz: float) -> tuple[float, float]:
        return x + lx * cos_y + lz * sin_y, z - lx * sin_y + lz * cos_y

    return at


def _dumpster(x: float, z: float, yaw: float, lid_open: bool) -> tuple[list[Prop], list[AABB]]:
    at = _local(x, z, yaw)
    props = [
        Prop.box(x, 0.65, z, 1.6, 1.3, 1.1, DUMPSTER, rotate=((yaw, 0.0, 1.0, 0.0),), solid=True, label="Dumpster"),
        Prop.box(x, 0.08, z, 1.4, 0.16, 0.95, DARK_STEEL, rotate=((yaw, 0.0, 1.0, 0.0),), label="Dumpster wheels bar"),
    ]
    if lid_open:
        # Hinged along the back top edge (local z = -0.56, y = 1.33) and swung up 55 degrees:
        # the lid used to be tilted about its own middle, so its front edge sank into the bin.
        angle, half = 55.0, 0.55
        lx, lz = at(0.0, -0.56 + half * math.cos(math.radians(angle)))
        props.append(Prop.box(lx, 1.33 + half * math.sin(math.radians(angle)), lz, 1.55, 0.06, 2 * half, DUMPSTER, rotate=((yaw, 0.0, 1.0, 0.0), (-angle, 1.0, 0.0, 0.0)), solid=True, label="Dumpster lid, propped open"))
        props.append(Prop.box(x, 1.38, z, 1.3, 0.2, 0.8, TRASH_BAG, rotate=((yaw, 0.0, 1.0, 0.0),), label="Overflowing rubbish"))
    else:
        props.append(Prop.box(x, 1.33, z, 1.62, 0.06, 1.12, DUMPSTER, rotate=((yaw, 0.0, 1.0, 0.0),), label="Dumpster lid"))
    r = 1.0
    return props, [AABB(x - r, x + r, z - 0.75, z + 0.75, 0.0, 1.4)]


def _bike(x: float, z: float, yaw: float) -> tuple[list[Prop], list[AABB]]:
    """A parked bike along local z (front at +z). Every part is placed in the bike's own
    frame and turned with it - they used to be turned in place, about their own centres,
    which left the frame skewed off the line of the wheels."""
    at = _local(x, z, yaw)
    turn = (yaw, 0.0, 1.0, 0.0)
    wheel_r, tyre = 0.34, 0.08
    props: list[Prop] = []
    for lz in (-0.6, 0.6):
        # the wheel's cylinder extends from its base along local -x: centre it on the frame
        wx, wz = at(tyre / 2.0, lz)
        props.append(Prop.cylinder(wx, wheel_r, wz, wheel_r, wheel_r, tyre, TIRE, slices=12, rotate=(turn, (90.0, 0.0, 0.0, 1.0)), label="Bike wheel"))
    fx, fz = at(0.0, 0.0)
    props.append(Prop.box(fx, 0.62, fz, 0.12, 0.3, 1.1, BIKE, rotate=(turn, (-18.0, 1.0, 0.0, 0.0)), label="Bike frame"))
    sx, sz = at(0.0, -0.15)
    props.append(Prop.box(sx, 0.86, sz, 0.28, 0.1, 0.42, Material("leather", (0.2, 0.18, 0.16)), rotate=(turn,), label="Bike seat"))
    hx, hz = at(0.0, 0.5)
    props.append(Prop.box(hx, 0.95, hz, 0.6, 0.04, 0.04, STEEL, rotate=(turn,), label="Handlebars"))
    # at the back, on a little rack above the rear wheel (it used to sit by the handlebars)
    tx, tz = at(0.0, -0.66)
    props.append(Prop.box(tx, 0.74, tz, 0.1, 0.08, 0.12, Material(None, (1.0, 0.3, 0.2), emission=(0.8, 0.2, 0.1)), rotate=(turn,), label="Bike tail light"))
    return props, [AABB(x - 0.45, x + 0.45, z - 1.0, z + 1.0, 0.0, 1.0)]


def _overhead() -> list[Prop]:
    """Cables strung across the alley with paper lanterns and their pools of light, and
    the odd piece of laundry."""
    props: list[Prop] = []
    for k, z in enumerate((3.5, 9.0, 17.5, 21.0, 29.0, 34.5, 39.5)):
        y = 5.6 + 0.6 * (k % 3)
        props.append(Prop.box(0.0, y, z, ALLEY_HALF * 2.0 + 0.2, 0.03, 0.03, CABLE, rotate=((2.5 * ((k % 2) * 2 - 1), 0.0, 0.0, 1.0),), label="Cable across the alley"))
        if k % 2 == 0:
            lx = -1.5 + (k % 3) * 1.4
            props.append(Prop.box(lx, y - 0.45, z, 0.03, 0.6, 0.03, CABLE, label="Lantern cord"))
            props.append(Prop.box(lx, y - 0.95, z, 0.28, 0.36, 0.28, LANTERN_RED, label="Paper lantern on the cable", tags=DECAL))
            props.append(Prop.disc(lx, 0.005, z, 1.8, 1.8, 0.2, LANTERN_GLOW, slices=18, label="Lantern light on the pavement", tags=frozenset({"glow", "transparent"})))
        else:
            props.append(Prop.box(-1.0 + (k % 3), y - 0.4, z, 0.5, 0.7, 0.02, Material("fabric", (0.7, 0.7, 0.75)), rotate=((6.0, 0.0, 0.0, 1.0),), label="Laundry on the line"))
    return props


def _main_street() -> list[Prop]:
    """The far side of the main street (a backdrop - the near side is
    ``levels.main_street``): a monorail beam over the road, the far pavement, a row of lit
    shopfronts under towers of windows and neon."""
    far = STREET_FAR
    props: list[Prop] = [
        Prop.box(0.0, 9.0, far + 1.0, 2.0 * main_street.TUNNEL_X, 18.0, 1.0, Material("concrete", (0.045, 0.05, 0.07)), solid=True, label="Main street towers (backdrop)"),
        Prop.box(0.0, -0.06, (main_street.ROAD_Z1 + 0.3 + far + 0.5) / 2.0, 2.0 * main_street.TUNNEL_X, 0.12, far + 0.5 - main_street.ROAD_Z1 - 0.3, Material("concrete", (0.2, 0.2, 0.22)), label="Far pavement"),
        # big signs up the towers
        Prop.box(-9.0, 6.0, far + 0.4, 5.0, 1.2, 0.2, NEON_MAGENTA, label="Big neon sign across the street"),
        Prop.box(9.5, 8.5, far + 0.4, 4.0, 1.0, 0.2, NEON_CYAN, label="Big neon sign across the street"),
        Prop.box(1.0, 11.5, far + 0.4, 6.0, 0.8, 0.2, NEON_AMBER, label="Billboard across the street"),
        Prop.box(-14.5, 9.0, far + 0.4, 0.8, 6.0, 0.2, NEON_CYAN, label="Vertical neon down the tower"),
        Prop.box(15.5, 7.5, far + 0.4, 0.8, 5.0, 0.2, NEON_MAGENTA, label="Vertical neon down the tower"),
        Prop.box(-26.0, 7.05, far + 0.4, 4.5, 1.1, 0.2, NEON_AMBER, label="Big neon sign across the street"),
        Prop.box(26.5, 9.5, far + 0.4, 0.8, 5.5, 0.2, NEON_CYAN, label="Vertical neon down the tower"),
    ]
    # shopfronts along the far pavement: each its own colour and sign, pillars between
    fronts = (
        (-34.0, (0.9, 0.3, 0.2), "PAWN"), (-29.5, (0.3, 0.9, 1.0), "BIOWARE"), (-25.0, (1.0, 0.6, 0.1), "DUMPLINGS"),
        (-20.5, (0.6, 0.4, 1.0), "ARCADE"),
        (-16.0, (1.0, 0.5, 0.2), "RAMEN"), (-11.5, (0.3, 0.9, 1.0), "PHARMA"), (-7.0, (0.9, 0.2, 0.6), "BAR"),
        (-2.5, (1.0, 0.85, 0.4), "24H MART"), (2.0, (0.4, 1.0, 0.5), "CLINIC"), (6.5, (0.9, 0.3, 0.2), "PAWN"),
        (11.0, (0.5, 0.6, 1.0), "NET CAFE"), (15.5, (1.0, 0.6, 0.1), "NOODLES"),
        (20.0, (1.0, 0.3, 0.6), "KARAOKE"), (24.5, (0.4, 1.0, 0.8), "LAUNDRY"), (29.0, (1.0, 0.85, 0.4), "CHROME"),
        (33.5, (0.5, 0.6, 1.0), "HOSTEL"), (38.0, (0.9, 0.2, 0.6), "TATTOO"),
        (-38.5, (0.4, 1.0, 0.5), "VAPES"),
    )  # fmt: skip
    for fx, col, name in fronts:
        glass = Material(None, tuple(c * 0.45 for c in col), emission=tuple(c * 0.22 for c in col))
        props.append(Prop.box(fx, 1.3, far + 0.45, 3.6, 2.6, 0.1, glass, label=f"Shopfront: {name}"))
        props.append(Prop.box(fx, 0.5, far + 0.4, 3.2, 1.0, 0.06, Material(None, tuple(c * 0.18 for c in col)), label=f"Shopfront counter: {name}"))
        for mx in (-1.2, 0.0, 1.2):  # window frames
            props.append(Prop.box(fx + mx, 1.3, far + 0.36, 0.08, 2.6, 0.06, DARK_STEEL, label="Shopfront mullion"))
        props.append(Prop.box(fx, 1.7, far + 0.36, 3.6, 0.08, 0.05, DARK_STEEL, label="Shopfront transom"))
        props.append(Prop.box(fx, 2.95, far + 0.3, 3.0, 0.4, 0.12, Material(None, tuple(c * 0.9 for c in col), emission=tuple(c * 0.6 for c in col)), label=f"Shop sign: {name}"))
        props.append(Prop.box(fx, 2.62, far + 0.0, 3.4, 0.06, 0.8, Material("fabric", tuple(c * 0.35 for c in col)), rotate=((12.0, 1.0, 0.0, 0.0),), label=f"Awning: {name}"))
        props.append(Prop.box(fx + 2.25, 1.6, far + 0.4, 0.5, 3.2, 0.3, CONCRETE, label="Pillar between shopfronts"))
    rng = random.Random(21)
    for k in range(185):
        wx, wy = -41.65 + (k % 37) * 2.3, 4.6 + (k // 37) * 2.5
        mat = WINDOW_WARM if rng.random() < 0.45 else (WINDOW_COOL if rng.random() < 0.5 else WINDOW_DARK)
        props.append(Prop.box(wx, wy, far + 0.45, 1.1, 1.3, 0.05, mat, label="Tower window"))
    return props


def _ends() -> tuple[list[Prop], list[AABB]]:
    """The facility's outer wall behind the player (with the yellow door they came out
    of) and, at the far end, the glow and towers of the main street."""
    props: list[Prop] = [
        Prop.box(0.0, (WALL_H - 0.02) / 2.0, -3.1, ALLEY_HALF * 2.0 + 2.36, WALL_H - 0.02, 1.2, CONCRETE, solid=True, label="Halcyon facility outer wall"),  # (inside the tenements' ends)
        # the yellow door you came out of - the same door as the greenhouse's street exit
        replace(staff_door_leaf(0.0, FACILITY_WALL_FACE_Z, +1), label="The yellow door you came out of"),
        *staff_door_trim(0.0, FACILITY_WALL_FACE_Z, +1),
        staff_door_alarm(0.0, FACILITY_WALL_FACE_Z, +1),
    ]
    props += _main_street()
    p, c = main_street.build()
    props += p
    colliders = [
        AABB(-6.0, 6.0, -4.0, -2.5, 0.0, WALL_H),
        AABB(-main_street.TUNNEL_X, main_street.TUNNEL_X, STREET_FAR - 0.5, STREET_FAR + 2.0, 0.0, 18.0),  # across the street
        *c,
    ]
    return props, colliders


# ----------------------------------------------------------------------------- assembly
HOME_DOORS_AT = ((-1, 6.0, True, "14"), (-1, 30.5, True, "16"), (1, 36.0, False, "21"))  # side, z, lamp, number
DOOR_WIDTH = {"16": apartment.DOOR_Z[1] - apartment.DOOR_Z[0]}  # number 16's is a real door
FLAT_Z = (apartment.SOUTH_Z - 0.2, apartment.NORTH_Z + 0.2)  # number 16's flat, behind the wall
_F17 = main_street.FLAT17
FLAT17_Z = (_F17.u0 - 0.2, _F17.u1 + 0.2)  # number 17's, across the alley from Nix's wall
# (its frame stands out from the wall; its access panel is beside it on the flat's own wall)
_P17 = flats.panel(_F17)
DOOR17_CUT = ((_F17.door_u - 0.62 + _P17.z + 0.16) / 2.0, (_P17.z + 0.16 - (_F17.door_u - 0.62)) / 2.0)
# the windows of numbers 16 and 17 onto the alley: the skirting stops either side of them
# (run across them it hid all but the top of each window)
_W17 = flats.desk_span(_F17)
WINDOW_CUTS: tuple[tuple[float, float], ...] = (
    ((apartment.WINDOW_Z[0] + apartment.WINDOW_Z[1]) / 2.0, (apartment.WINDOW_Z[1] - apartment.WINDOW_Z[0]) / 2.0 + 0.1),
)
if _W17 is not None:
    WINDOW_CUTS += (((_W17[0] + _W17[1]) / 2.0, flats.window_width(_W17) / 2.0 + 0.1),)

METRO_DOORWAY = WalkRegion(
    metro.DOORWAY[0], metro.DOORWAY[1], metro.GATES_Z[1], main_street.FACE_Z + 0.3,
    inset=frozenset({"min_x", "max_x", "min_z"}),
)  # fmt: skip


def _build_sector() -> SectorData:
    props: list[Prop] = [
        # (it ends under the flats' front walls at the sides and at the pavement at the
        # mouth: run on under them, the street showed through their floors)
        Prop.box(0.0, -0.05, (-2.5 + main_street.FACE_Z) / 2.0, ALLEY_HALF * 2.0 + 0.6, 0.1, main_street.FACE_Z + 2.5, ASPHALT, label="Alley asphalt, wet"),
    ]
    colliders: list[AABB] = []
    nav_blockers: list[AABB] = []
    for side in (-1, 1):
        doors = tuple(
            (z, DOOR_WIDTH.get(n, 1.0) / 2.0 + 0.25) for s_, z, _, n in HOME_DOORS_AT if s_ == side
        )
        if side < 0:
            doors += (DOOR17_CUT, *WINDOW_CUTS)
        p, c = _facade(side, doors, flats=(FLAT_Z, FLAT17_Z) if side < 0 else ())
        props += p
        colliders += c
    for side, z, lamp, number in HOME_DOORS_AT:
        real = number == "16"
        p, c, n, _ = _home_door(side, z, lamp, number, DOOR_WIDTH.get(number, 1.0), real=real)
        props += p
        colliders += c
        nav_blockers += n
    p, c = apartment.build()
    props += p
    colliders += c
    p, c, *_ = main_street.flat_parts()
    props += p
    colliders += c
    p, c = bar.build()  # the Neon Moth, round the corner
    props += p
    colliders += c
    p, c = metro.build()  # the metro station at the east end
    props += p
    colliders += c
    p, c = _vending_spot()
    props += p
    colliders += c
    props += _door17_lamp()
    for builder in (_food_stall, _junk_stall, _ends):
        p, c = builder()
        props += p
        colliders += c
    for x, z, yaw, open_ in ((3.5, 5.5, 8.0, True), (-3.5, 34.0, -6.0, False)):
        p, c = _dumpster(x, z, yaw, open_)
        props += p
        colliders += c
    p, c = _bike(3.7, 31.0, 12.0)
    props += p
    colliders += c
    props += _overhead()
    props += _dress(colliders)
    return SectorData(
        name="NightMarketAlley",
        bounds=(-TUNNEL_END, TUNNEL_END, -4.0, STREET_FAR + 2.0),
        props=tuple(props),
        colliders=tuple(colliders),
        nav_blockers=(*nav_blockers, *metro.NAV_BLOCKERS),
        lights=ALLEY_LIGHTS + apartment.LIGHTS + main_street.street_lights() + FLAT_LIGHTS + bar.LIGHTS + metro.LIGHTS,
        walk_regions=(
            # the alley, out of its mouth onto the pavement of the main street (and along
            # it as far as the barricades at either end of the built city)
            WalkRegion(-4.2, 4.2, -2.0, main_street.FACE_Z + 0.3, inset=frozenset({"min_x", "max_x", "min_z"})),
            main_street.PAVEMENT,
            STOOP_16,  # number 16's step
            *apartment.WALK_REGIONS,
            *FLAT_REGIONS,
            *bar.WALK_REGIONS,
            METRO_DOORWAY,  # (as far as the ticket gates)
            *metro.WALK_REGIONS,  # (through them, up the stair, out onto the platform)
        ),
    )


_FLATS = main_street.flat_parts()
FLAT_DOORS, FLAT_REGIONS, FLAT_PLACES, FLAT_ROUTES, FLAT_AREAS = _FLATS[2:7]
FLAT_PANELS = tuple(_FLATS[8])
FLAT_LIGHTS = tuple(_FLATS[7])

# The vending machine against the east wall, between the dumpster and the junk stall: a
# can of something for whoever is passing - and, when the stalls are shut, Dell's supper.
VENDING_AT = (3.84, 0.0, 17.6)  # (its back to the skirting, facing the alley: yaw -90)


def _door17_lamp() -> list[Prop]:
    """A lamp over number 17's door, like number 16's."""
    z = _F17.door_u
    return [
        Prop.box(-ALLEY_HALF + 0.2, 2.55, z, 0.3, 0.12, 0.3, LAMP, label="Door lamp", tags=DECAL),
        Prop.disc(-ALLEY_HALF + 0.55, 0.004, z, 1.1, 1.3, 0.3, LAMP_GLOW, slices=18, label="Door lamp light on the pavement", tags=frozenset({"glow", "transparent"})),
    ]


def _vending_spot() -> tuple[list[Prop], list[AABB]]:
    """The pool of light under the machine (the machine itself is an entity)."""
    x, _, z = VENDING_AT
    props = [
        Prop.disc(x - 0.9, 0.008, z, 1.2, 1.4, 0.3, Material(None, (0.3, 0.8, 1.0), alpha=0.25), slices=16, label="Vending machine glow on the pavement", tags=frozenset({"glow", "transparent"})),
    ]
    return props, [AABB(x - 0.55, x + 0.55, z - 0.65, z + 0.65, 0.0, 2.3)]
# the step of number 16 (up to the wall; the door and the wall stop the player there)
STOOP_16 = WalkRegion(
    -ALLEY_HALF, -ALLEY_HALF + STOOP_DEPTH + 0.3, 30.5 - 0.85, 30.5 + 0.85,
    inset=frozenset({"min_z", "max_z"}),
)  # fmt: skip
STOOP_16_TOP = AABB(-ALLEY_HALF, -ALLEY_HALF + STOOP_DEPTH, 30.5 - 0.85, 30.5 + 0.85, 0.0, 0.16)


def _dress(colliders: list[AABB]) -> list[Prop]:
    """Trash along the walls, puddles down the middle, bags by the dumpsters."""
    keep_out = [AABB(x - 0.6, x + 0.6, z - 0.6, z + 0.6, 0.0, 2.0) for x, z in ALLEY_WAYPOINTS]
    keep_out += [AABB(m.x - 0.6, m.x + 0.6, m.z - 0.6, m.z + 0.6, 0.0, 2.0) for m in ALLEY_MERCHANTS]
    # (and of every spot the people of the street go to: the stalls' back ways, the
    # vending machine, everybody's front door)
    keep_out += [
        AABB(pl.x - 0.7, pl.x + 0.7, pl.z - 0.7, pl.z + 0.7, 0.0, 2.0)
        for pl in ALLEY_PLACES
        if pl.name.split(".")[0] in ("sushi", "junk", "vending", "mouth")
        or pl.name.endswith((".front", ".door"))
    ]
    placer = Placer(seed=808, blockers=colliders + keep_out)
    props: list[Prop] = []
    props += scatter_trash(placer, AABB(-4.3, -2.3, -1.5, 41.0), 16, ("can", "paper", "box", "cable", "bottle", "rag", "chip", "monitor"))
    props += scatter_trash(placer, AABB(2.3, 4.3, -1.5, 41.0), 16, ("can", "paper", "box", "cable", "bottle", "rag", "chip"))
    props += scatter_trash(placer, AABB(-2.3, 2.3, -1.5, 41.0), 10, ("puddle", "puddle", "paper", "can", "chip"))
    # the pavement of the main street: a little litter along the foot of the blocks
    props += scatter_trash(placer, AABB(-32.0, -6.5, 42.7, 43.4), 12, ("can", "paper", "bottle", "chip", "rag"))
    props += scatter_trash(placer, AABB(6.5, 30.0, 42.7, 43.4), 12, ("can", "paper", "bottle", "chip", "box"))
    for x, z, yaw, h in ((2.4, 6.6, 20.0, 0.5), (2.9, 7.2, -30.0, 0.4), (-2.15, 33.0, 70.0, 0.45), (-2.5, 35.3, 15.0, 0.5)):
        props.append(Prop.box(x, h / 2.0, z, 0.5, h, 0.45, TRASH_BAG, rotate=((yaw, 0.0, 1.0, 0.0),), label="Trash bag", tags=CLUTTER))
    return props


# A little stronger than the facility's lights: the renderer's overall light exposure
# (render.scene.LIGHT_EXPOSURE) would otherwise leave the alley - already the darkest place
# in the game - too dim to read.
ALLEY_LIGHTS = (
    Light(
        -2.6, 3.2, 14.0,
        ambient=(0.202, 0.09, 0.034), diffuse=(2.912, 1.568, 0.56), specular=(2.24, 1.344, 0.56),
        linear_attenuation=0.12, quadratic_attenuation=0.03,
        fixture="none",
    ),
    Light(
        -4.0, 3.4, 14.0,
        ambient=(0.134, 0.0, 0.101), diffuse=(2.688, 0.336, 2.016), specular=(2.24, 0.448, 1.792),
        linear_attenuation=0.15, quadratic_attenuation=0.04,
        fixture="none",
    ),
    Light(
        3.3, 3.0, 25.5,
        ambient=(0.0, 0.112, 0.134), diffuse=(0.448, 2.24, 2.688), specular=(0.336, 2.016, 2.464),
        linear_attenuation=0.15, quadratic_attenuation=0.04,
        fixture="none",
    ),
    Light(
        -3.6, 2.65, 6.0,  # the lamp over number 14 (out from the wall, onto the door)
        ambient=(0.112, 0.09, 0.045), diffuse=(2.1, 1.6, 0.95), specular=(1.344, 1.008, 0.56),
        linear_attenuation=0.25, quadratic_attenuation=0.08,
        fixture="none",
    ),
    Light(
        # the lamp over number 16 (the courier's, the resident's - and the quest's - door):
        # a little out from the wall, so it lights the door and its panel, not just the stoop
        -3.6, 2.65, 30.5,
        ambient=(0.112, 0.09, 0.045), diffuse=(2.1, 1.6, 0.95), specular=(1.344, 1.008, 0.56),
        linear_attenuation=0.25, quadratic_attenuation=0.08,
        fixture="none",
    ),
    Light(
        0.0, 5.2, 21.0,
        ambient=(0.067, 0.034, 0.022), diffuse=(1.568, 0.56, 0.336), specular=(1.12, 0.448, 0.224),
        linear_attenuation=0.2, quadratic_attenuation=0.05,
        fixture="none",
    ),
    Light(
        0.0, 6.0, 47.0,
        ambient=(0.134, 0.134, 0.202), diffuse=(2.464, 2.24, 3.136), specular=(1.68, 1.68, 2.24),
        linear_attenuation=0.06, quadratic_attenuation=0.01,
        fixture="none",
    ),
    Light(  # the lamp over number 17 (Nix's)
        -3.6, 2.65, 1.6,
        ambient=(0.112, 0.09, 0.045), diffuse=(2.0, 1.5, 0.9), specular=(1.3, 1.0, 0.55),
        linear_attenuation=0.25, quadratic_attenuation=0.08,
        fixture="none", night_only=True,
    ),
    Light(  # the vending machine's glow
        VENDING_AT[0] - 0.8, 1.3, VENDING_AT[2],
        ambient=(0.0, 0.04, 0.06), diffuse=(0.5, 1.3, 1.7), specular=(0.3, 0.8, 1.0),
        linear_attenuation=0.35, quadratic_attenuation=0.25,
        fixture="none",
    ),
)

# Where people wander: along the alley, with stops at both stalls and the doors.
ALLEY_WAYPOINTS: tuple[tuple[float, float], ...] = (
    (-2.4, 2.5), (2.0, 3.0), (-1.4, 9.5), (1.8, 9.0), (-1.2, 14.0), (1.6, 15.5), (-2.2, 19.5),
    (2.2, 19.0), (0.6, 25.5), (-2.0, 27.0), (-1.6, 30.0), (1.2, 33.5), (-0.8, 37.5), (2.0, 39.5),
    (0.0, 41.0),
)  # fmt: skip

# Where everybody goes about their business (see ai.alley): the sushi bar's kitchen and
# counter, the junk table, the step of number 16, the bin, the wall by the Halcyon door.
BAR_PLACES, BAR_ROUTES = bar.places()
ALLEY_PLACES = (
    # the sushi bar: the chef's spots behind the counter, what is on it, the customers' side
    Place("sushi.chef", -3.6, 14.2, yaw=90.0),
    Place("sushi.board", -3.1, 14.2, y=1.04),
    Place("sushi.boxes", -3.12, 14.72, y=1.12),
    Place("sushi.cooler", -3.6, 15.05, yaw=-90.0),
    Place("sushi.wok", -3.6, 13.0, yaw=-90.0),
    Place("sushi.serve", -3.6, 13.55, yaw=90.0),
    Place("sushi.pass", -2.98, 13.55, y=1.01),
    Place("sushi.customer", -2.38, 13.55, yaw=-90.0),
    Place("sushi.queue", -1.3, 13.9, yaw=-90.0),
    Place("sushi.menu", -4.44, 15.0, y=2.0),
    Place("sushi.stool", -2.22, 12.5, yaw=-90.0),
    Place("sushi.lean", -1.6, 10.5, yaw=-20.0),  # out of the queue's way, to eat standing
    # the junk table
    Place("junk.dealer", 3.6, 25.2, yaw=-90.0),
    Place("junk.stock", 3.6, 26.3, yaw=90.0),
    Place("junk.customer", 2.3, 25.2, yaw=90.0),
    Place("junk.browse", 2.3, 26.3, yaw=90.0),
    Place("junk.table", 3.0, 25.5, y=0.9),  # (the middle of its top: the cover goes there)
    # number 16, its step and where a parcel gets left on it (and inside: apartment.PLACES)
    Place("door16.stoop", -4.0, 30.5, yaw=-90.0, y=0.16),
    Place("door16.beside", -3.86, 30.5, yaw=0.0, y=0.16),  # on the step, facing along it
    Place("door16.front", -3.15, 30.5, yaw=-90.0),  # at the foot of the step
    # the corner of the step by the door's latch side: clear of the door, its frame and
    # of anybody standing on the step
    Place("door16.parcel", -4.08, 30.95, yaw=0.0, y=0.16),
    # the open dumpster, the alley mouth, the smoker's wall and where people stop to chat
    Place("bin", 2.1, 5.5, yaw=90.0),
    # off round the corner and away: the METRO steps and the UNDERPASS at the ends of the
    # main street ("mouth", "mouth.left" - see levels.main_street)
    *main_street.PLACES,
    # the vending machine: where to stand at it (facing it), and what it is
    Place("vending", VENDING_AT[0] - 1.05, VENDING_AT[2], yaw=90.0),
    Place("vending.slot", VENDING_AT[0] - 0.42, VENDING_AT[2] + 0.3, y=0.35),
    # the stalls' back ways out, for closing time
    Place("sushi.exit", -3.55, 16.95, yaw=180.0),
    Place("junk.exit", 3.5, 27.95, yaw=0.0),
    Place("smoke", 3.85, 1.6, yaw=-90.0),
    Place("chat", 2.6, 2.3, yaw=90.0),
    *apartment.PLACES,
    *FLAT_PLACES,
    *BAR_PLACES,  # the Neon Moth (``levels.bar``)
    *metro.places(),  # the way up to the metro's platform (``levels.metro``)
)

ALLEY_ITEMS = (
    ItemData("knife", -3.0, 1.055, 13.98, yaw=0.0, item_id="knife"),  # on the chopping board
)

ALLEY_CIVILIANS = tuple(
    CivilianData(x, 0.0, z, ALLEY_WAYPOINTS, seed=seed, civilian_id=cid, role=role)
    for (x, z), seed, cid, role in (
        ((0.8, 40.0), 11, "civilian_office_worker", "office_worker"),
        ((-1.0, 36.0), 12, "civilian_courier", "courier"),
        ((1.5, 20.0), 13, "civilian_tinkerer", "tinkerer"),
        ((3.85, 1.6), 16, "civilian_smoker", "smoker"),
        ((-0.5, 38.0), 14, "civilian_resident", "resident"),
        ((-1.4, 9.5), 15, "civilian_stroller", "stroller"),
    )
) + (
    # the Neon Moth's bartender (behind the bar: it is open when the night begins) and a
    # regular of his, on the metro's platform (off home before coming back for a drink)
    CivilianData(bar.SERVE_X, 0.0, 40.7, ALLEY_WAYPOINTS, seed=17, civilian_id="civilian_bartender", role="bartender"),
    CivilianData(metro.WAIT_SPOTS[2][0], metro.FLOOR_Y, metro.WAIT_SPOTS[2][1], ALLEY_WAYPOINTS, seed=18, civilian_id="civilian_regular", role="regular"),
)

ALLEY_MERCHANTS = (
    MerchantData(-3.6, 0.0, 14.2, face_yaw=90.0, style="cook", seed=201, merchant_id="sushi_and_wok", role="sushi_chef"),
    MerchantData(3.6, 0.0, 25.2, face_yaw=-90.0, style="hawk", seed=202, merchant_id="cyberware_cheap", role="junk_dealer"),
)

NIGHT_MARKET = LevelData(
    name="Night Market Alley",
    brief="Out. The city's poorest sector: keep your head down - and your eyes open.",
    sectors=(_build_sector(),),
    doors=(apartment.DOOR16, *FLAT_DOORS, bar.door(), *metro.GATE_DOORS),
    catwalk=Catwalk(inner_x=90.0, outer_x=91.0, stairs_z_start=900.0, stairs_z_end=901.0),  # none here
    nav_bounds=(-35.5, 33.5, -4.0, 46.0),
    spawn=(0.0, -1.2, 90.0),
    factions=(
        FactionData(apartment.FACTION, "DELL'S HOME", (1.0, 0.45, 0.75)),
        FactionData("neon_cold", "NEON COLD VENDING", (0.3, 0.85, 1.0)),
        FactionData(flats.TENEMENT_LOCKS, "TENEMENT DOORLOCKS", (0.95, 0.75, 0.3)),
        FactionData(metro.GATE_FACTION, "METRO TRANSIT", (0.45, 1.0, 0.6)),
    ),
    civilians=ALLEY_CIVILIANS,
    merchants=ALLEY_MERCHANTS,
    places=ALLEY_PLACES,
    items=ALLEY_ITEMS,
    # caught in here (only number 16's camera and owner watch for the netrunner) means
    # being thrown out of it
    on_caught="evict",
    outdoor=True,
    terminals=(apartment.TERMINAL,),
    cameras=(apartment.CAMERA,),
    panels=(apartment.PANEL, *FLAT_PANELS),
    pickups=(apartment.DATASTICK,),
    private_areas=(apartment.FLAT16, *FLAT_AREAS),
    platforms=(STOOP_16_TOP,),
    routes=apartment.ROUTES + tuple(FLAT_ROUTES) + BAR_ROUTES,
    vending_machines=(VendingMachineData(*VENDING_AT, yaw=-90.0, machine_id="street_vendor", faction="neon_cold"),),
    lanes=main_street.LANES,  # the traffic on the main street
    metro=metro.DATA,  # the trains along the monorail, and the station
    flights=(metro.FLIGHT,),  # the station's stair (anybody can climb it) ...
    decks=metro.DECKS,  # ... to the landing, the footbridge and the platform
)


# What keeps the rain off (``LevelData.shelters``): the awnings and canopies, the metro's
# platform and footbridge, the parked car's roof - each prop's box, down to the ground
SHELTER_LABELS = (
    "Awning:", "Food stall awning", "Tarp canopy", "Platform canopy", "Platform deck",
    "Footbridge roof", "Car roof", "Car canopy",
)  # fmt: skip


def _shelters(level: LevelData) -> tuple[AABB, ...]:
    out = []
    for p in level.props:
        if p.label.startswith(SHELTER_LABELS):
            f = p.footprint()
            out.append(AABB(f.min_x, f.max_x, f.min_z, f.max_z, main_street.ROAD_Y, f.max_y))
    return tuple(out)


ALLEY_SKY = OpenSky(-ALLEY_HALF, ALLEY_HALF, -2.5, main_street.FACE_Z, 0.0)
NIGHT_MARKET = replace(
    NIGHT_MARKET, open_sky=(ALLEY_SKY, *main_street.OPEN_SKY), shelters=_shelters(NIGHT_MARKET)
)
# every front door a little easier to walk through than it looks (``doorway.ease_doorways``)
NIGHT_MARKET = ease_doorways(NIGHT_MARKET)


def build_night_market() -> LevelData:
    return NIGHT_MARKET


__all__ = ["NIGHT_MARKET", "build_night_market"]
