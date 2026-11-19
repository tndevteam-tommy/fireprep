"""Building blocks for level data: walls with doorways, furniture, a lived-in town's clutter,
wall torches, straight stairs to an upper floor and the schedule helpers every resident uses.

Everything here returns plain level dataclasses (``WallDef``, ``PropDef``, ``TaskDef`` ...)
authored with its floor at y = 0; ``World`` lifts it onto the terrain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..entities import AIState
from .level import ContainerDef, ItemDef, LightDef, PropDef, ReadableDef, TaskDef, WallDef
from .tower import STEP_RISE, STEP_THICKNESS, StairDef

Vec = Tuple[float, float, float]

WALL = 0.5                       # a building wall's thickness
STOREY = 4.0                     # floor to floor
FLOOR_THICKNESS = 0.3            # an upper floor's boards (the prop's box)
BED_LENGTH, BED_WIDTH = 2.0, 1.1
DOUBLE_BED_WIDTH = 1.7
TORCH_HEIGHT = 2.7               # a wall torch's ring above the ground at the wall's foot
TORCH_OFF_WALL = 0.3             # the flame stands this far out from the wall face
TORCH_RANGE = 8.5


def key_item(door_id: str, label: str) -> ItemDef:
    return ItemDef(f"{door_id}_key", label, value=1.0, is_key=True, key_door_id=door_id, for_sale=False, category="key")


def coins(n: int) -> ItemDef:
    """A pile of coins in a chest: taken, they go straight into the purse (see ``ContainerPolicy``)."""
    return ItemDef("coins", "Gold Coins", value=1.0, stackable=True, quantity=n, category="coins")


# --- walls ------------------------------------------------------------------------------------
def house_walls(x1: float, z1: float, x2: float, z2: float, door_x: float, door_side: str, texture: str = "building",
                door_width: float = 3.0, height: float = STOREY, door_x2: Optional[float] = None,
                door_width2: float = 3.0) -> List[WallDef]:
    """Four walls of a rectangular building with a door gap in the ``door_side`` ("north" = the
    z1 wall, "south" = the z2 wall starting at x ``door_x``; "west" = the x1 wall, "east" = the x2
    wall, starting at z ``door_x``); ``door_x2`` cuts a second doorway in the opposite wall (the guard
    house's way into the tower). The side walls run the full depth and the front and back walls fit
    between them (``World`` makes any overlap disjoint anyway)."""
    if door_side in ("west", "east"):
        walls = [
            WallDef((x1, 0, z1), (x2, height, z1 + WALL), texture=texture),        # north
            WallDef((x1, 0, z2 - WALL), (x2, height, z2), texture=texture),        # south
        ]
        for side, xa, xb in (("west", x1, x1 + WALL), ("east", x2 - WALL, x2)):
            cut = door_x if side == door_side else door_x2
            width = door_width if side == door_side else door_width2
            if cut is None:
                walls.append(WallDef((xa, 0, z1 + WALL), (xb, height, z2 - WALL), texture=texture))
            else:
                walls.append(WallDef((xa, 0, z1 + WALL), (xb, height, cut), texture=texture))
                walls.append(WallDef((xa, 0, cut + width), (xb, height, z2 - WALL), texture=texture))
        return walls
    walls = [
        WallDef((x1, 0, z1), (x1 + WALL, height, z2), texture=texture),            # west
        WallDef((x2 - WALL, 0, z1), (x2, height, z2), texture=texture),            # east
    ]
    for side, za, zb in (("north", z1, z1 + WALL), ("south", z2 - WALL, z2)):
        if side == door_side:
            walls.append(WallDef((x1 + WALL, 0, za), (door_x, height, zb), texture=texture))
            walls.append(WallDef((door_x + door_width, 0, za), (x2 - WALL, height, zb), texture=texture))
        elif door_x2 is not None:
            walls.append(WallDef((x1 + WALL, 0, za), (door_x2, height, zb), texture=texture))
            walls.append(WallDef((door_x2 + door_width2, 0, za), (x2 - WALL, height, zb), texture=texture))
        else:
            walls.append(WallDef((x1 + WALL, 0, za), (x2 - WALL, height, zb), texture=texture))
    return walls


def partition_x(x: float, z1: float, z2: float, height: float = STOREY, door: Optional[Tuple[float, float]] = None,
                y: float = 0.0, texture: str = "building", thickness: float = 0.3) -> List[WallDef]:
    """An inner wall along z at ``x`` (from z1 to z2), with an open doorway (z_a, z_b) if given."""
    if door is None:
        return [WallDef((x, y, z1), (x + thickness, y + height, z2), texture=texture)]
    a, b = door
    return [WallDef((x, y, z1), (x + thickness, y + height, a), texture=texture),
            WallDef((x, y, b), (x + thickness, y + height, z2), texture=texture)]


def partition_z(z: float, x1: float, x2: float, height: float = STOREY, door: Optional[Tuple[float, float]] = None,
                y: float = 0.0, texture: str = "building", thickness: float = 0.3) -> List[WallDef]:
    """An inner wall along x at ``z`` (from x1 to x2), with a doorway (x_a, x_b) if given."""
    if door is None:
        return [WallDef((x1, y, z), (x2, y + height, z + thickness), texture=texture)]
    a, b = door
    return [WallDef((x1, y, z), (a, y + height, z + thickness), texture=texture),
            WallDef((b, y, z), (x2, y + height, z + thickness), texture=texture)]


# --- furniture ----------------------------------------------------------------------------------
def bed(name: str, centre: Vec, facing: float = 0.0, owner: Optional[str] = None, double: bool = False) -> PropDef:
    """A bed centred on ``centre`` (the sleeper's spot) with its head towards ``facing`` (0 = -z,
    90 = +x, 180 = +z, 270 = -x): the footprint is always ``BED_LENGTH`` along the head-foot line
    and the bed's width across it, so every bed in a room looks the same size whichever way it
    is turned. ``centre``'s y is the floor it stands on (an upper storey's)."""
    cx, cy, cz = centre
    half_len, half_w = BED_LENGTH / 2.0, (DOUBLE_BED_WIDTH if double else BED_WIDTH) / 2.0
    along_z = int(round(facing / 90.0)) % 2 == 0
    hx, hz = (half_w, half_len) if along_z else (half_len, half_w)
    return PropDef(name, (cx - hx, cy, cz - hz), (cx + hx, cy + 0.6, cz + hz), texture="bed", kind="bed", facing=facing,
                   owner=owner)


def table(name: str, label: str, centre: Vec, items=(), owner=None, size=(0.75, 0.5)) -> ContainerDef:
    cx, cy, cz = centre
    hx, hz = size
    return ContainerDef(name, label, (cx - hx, cy, cz - hz), (cx + hx, cy + 0.8, cz + hz), items=tuple(items), owner=owner,
                        texture="wood", kind="table")


def chest(name: str, label: str, centre: Vec, items=(), owner=None, along_x: bool = True, locked: bool = False,
          lock_difficulty: int = 1, key_id: Optional[str] = None, strongbox: bool = False, crime_name: str = "Theft") -> ContainerDef:
    cx, cy, cz = centre
    hx, hz = (0.55, 0.35) if along_x else (0.35, 0.55)
    return ContainerDef(name, label, (cx - hx, cy, cz - hz), (cx + hx, cy + 0.7, cz + hz), items=tuple(items), owner=owner,
                        texture="wood", kind="strongbox" if strongbox else "chest", locked=locked,
                        lock_difficulty=lock_difficulty, key_id=key_id, crime_name=crime_name)


def pedestal(name: str, label: str, centre: Vec, items=(), owner=None, locked: bool = True, lock_difficulty: int = 5,
             key_id: Optional[str] = None) -> ContainerDef:
    """A treasure on show on a marble plinth (locked in a cage of wards: ``lock_difficulty``)."""
    cx, cy, cz = centre
    return ContainerDef(name, label, (cx - 0.4, cy, cz - 0.4), (cx + 0.4, cy + 1.1, cz + 0.4), items=tuple(items), owner=owner,
                        texture="marble", kind="pedestal", locked=locked, lock_difficulty=lock_difficulty, key_id=key_id)


# --- a lived-in town ----------------------------------------------------------------------------
def barrel(name: str, x: float, z: float, r: float = 0.42, h: float = 1.0, y: float = 0.0) -> PropDef:
    return PropDef(name, (x - r, y, z - r), (x + r, y + h, z + r), texture="barrel", kind="barrel")


def barrels(prefix: str, spots: Sequence[Tuple[float, float]], y: float = 0.0) -> List[PropDef]:
    return [barrel(f"{prefix}_{i}", x, z, y=y) for i, (x, z) in enumerate(spots)]


def woodpile(name: str, x1: float, z1: float, x2: float, z2: float, h: float = 1.1, y: float = 0.0) -> PropDef:
    """Split logs stacked along the footprint's longer side (the logs run across it)."""
    along_x = (x2 - x1) >= (z2 - z1)
    return PropDef(name, (x1, y, z1), (x2, y + h, z2), texture="bark", kind="woodpile", facing=90.0 if along_x else 0.0)


def oak(name: str, x: float, z: float, height: float = 3.4, trunk: float = 0.32) -> PropDef:
    return PropDef(name, (x - trunk, 0, z - trunk), (x + trunk, height, z + trunk), texture="bark", kind="oak")


def pine(name: str, x: float, z: float, height: float = 4.2, trunk: float = 0.22) -> PropDef:
    return PropDef(name, (x - trunk, 0, z - trunk), (x + trunk, height, z + trunk), texture="bark", kind="pine")


def bush(name: str, x: float, z: float, r: float = 0.8, h: float = 1.1) -> PropDef:
    return PropDef(name, (x - r, 0, z - r), (x + r, h, z + r), texture="leaves", kind="bush")


def boulder(name: str, x: float, z: float, r: float = 1.0, h: float = 1.2, texture: str = "rock") -> PropDef:
    return PropDef(name, (x - r, 0, z - r), (x + r, h, z + r), texture=texture, kind="boulder")


def haystack(name: str, x: float, z: float, r: float = 1.1, h: float = 1.6) -> PropDef:
    return PropDef(name, (x - r, 0, z - r), (x + r, h, z + r), texture="hay", kind="haystack")


def stall(name: str, x: float, z: float, facing: float = 0.0, w: float = 2.6, d: float = 1.6, h: float = 2.3) -> PropDef:
    along_x = int(round(facing / 90.0)) % 2 == 0
    hx, hz = (w / 2, d / 2) if along_x else (d / 2, w / 2)
    return PropDef(name, (x - hx, 0, z - hz), (x + hx, h, z + hz), texture="wood", kind="stall", facing=facing)


def fence(name: str, x1: float, z1: float, x2: float, z2: float, h: float = 1.1) -> PropDef:
    """A post-and-rail fence along the longer side of the (thin) footprint."""
    return PropDef(name, (x1, 0, z1), (x2, h, z2), texture="wood", kind="fence")


def crack(name: str, x: float, z: float, w: float = 1.6, d: float = 1.2, y: float = 0.0) -> PropDef:
    """Cracked paving: a decal on a level square (never on a slope - it would sink into the rise)."""
    return PropDef(name, (x - w / 2, y, z - d / 2), (x + w / 2, y + 0.02, z + d / 2), texture="crack_decal", kind="crack",
                   solid=False)


def stain(name: str, face_x1: float, face_z1: float, face_x2: float, face_z2: float, facing: float, y1: float = 0.4,
          y2: float = 3.2) -> PropDef:
    """Water damage seeping down the outside of a wall: a decal on the wall face from (x1, z1) to
    (x2, z2) (a line along the face), ``facing`` the way that face looks."""
    return PropDef(name, (face_x1, y1, face_z1), (face_x2, y2, face_z2), texture="stain_decal", kind="stain", solid=False)


def grate(name: str, x1: float, z1: float, x2: float, z2: float, h: float = 2.2) -> PropDef:
    """Iron bars across a culvert where water runs under a wall: solid, see-through."""
    return PropDef(name, (x1, -1.2, z1), (x2, h, z2), texture="grate", kind="grate")


def bridge(name: str, x1: float, z1: float, x2: float, z2: float, span: str = "x") -> PropDef:
    """A plank bridge: a walkable deck level with the higher bank (``anchor="high"``). ``span`` is
    the way it crosses the water - the way the road runs over it ("x" or "z"): its rails run
    that way too, along the edges that look down on the water, never across the road."""
    if span not in ("x", "z"):
        raise ValueError(f"a bridge spans along 'x' or 'z', not {span!r}")
    return PropDef(name, (x1, 0.0, z1), (x2, 0.12, z2), texture="wood", kind="bridge", solid=False, anchor="high",
                   facing=90.0 if span == "x" else 0.0)


def wall_torch(x: float, z: float, facing: float, room: str = "", y: float = TORCH_HEIGHT) -> LightDef:
    """A torch in an iron sconce on a wall, ``facing`` the way the wall face looks (0 = -z, 90 = +x
    ...); (x, z) is on that face - the flame is set out from it by ``TORCH_OFF_WALL``. Torches burn
    from dusk to dawn."""
    a = math.radians(facing)
    return LightDef("torch", (x + math.sin(a) * TORCH_OFF_WALL, y, z - math.cos(a) * TORCH_OFF_WALL), facing=facing,
                    room=room, range=TORCH_RANGE)


WINDOW_MARGIN = 0.15       # a window keeps this far from a door's jamb, a torch's bracket, the wall's end


def wall_windows(name: str, lo: float, hi: float, plane: float, facing: float, keep_clear=(), texture: str = "window_cottage",
                 width: float = 1.4, min_width: float = 0.9, y: float = 1.2, height: float = 1.3) -> List[PropDef]:
    """Windows (wall decals) along a wall running ``lo`` .. ``hi`` whose outer face is at ``plane``
    and looks ``facing``: one in the middle of every stretch left free between the (a, b) spans in
    ``keep_clear`` (a doorway, a torch, a sign) that has room for one at least ``min_width`` wide -
    none overlaps a door or hangs off the end of the wall."""
    free = [(lo + WINDOW_MARGIN, hi - WINDOW_MARGIN)]
    for a, b in keep_clear:
        a, b = a - WINDOW_MARGIN, b + WINDOW_MARGIN
        free = [piece for s, e in free for piece in ((s, min(e, a)), (max(s, b), e)) if piece[1] - piece[0] > 1e-6]
    out = []
    along_x = int(round(facing / 90.0)) % 2 == 0
    for s, e in free:
        w = min(width, e - s)
        if w < min_width:
            continue
        c = (s + e) / 2.0
        if along_x:
            z1, z2 = (plane - 0.02, plane) if int(round(facing / 90.0)) % 4 == 0 else (plane, plane + 0.02)
            box = ((c - w / 2, y, z1), (c + w / 2, y + height, z2))
        else:
            x1, x2 = (plane, plane + 0.02) if int(round(facing / 90.0)) % 4 == 1 else (plane - 0.02, plane)
            box = ((x1, y, c - w / 2), (x2, y + height, c + w / 2))
        out.append(PropDef(f"{name}_{len(out)}", *box, texture=texture, kind="window", facing=facing, solid=False))
    return out


def sconce(x: float, z: float, facing: float, room: str = "", y: float = TORCH_HEIGHT) -> LightDef:
    """A torch in a sconce that is kept burning day and night (down in a cave, in a windowless hall)."""
    torch = wall_torch(x, z, facing, room, y)
    return LightDef("sconce", torch.position, facing=facing, room=room, range=TORCH_RANGE)


def torches_beside_door(door_x1: float, door_x2: float, wall_z: float, facing: float, room: str = "") -> List[LightDef]:
    """A pair of torches either side of a doorway in a wall along x."""
    return [wall_torch(door_x1 - 0.6, wall_z, facing, room), wall_torch(door_x2 + 0.6, wall_z, facing, room)]


# --- camps, caves and the country --------------------------------------------------------------
def bedroll(name: str, centre: Vec, facing: float = 0.0, owner: Optional[str] = None) -> PropDef:
    """A bedroll on the floor, the head towards ``facing`` (a bed as far as the sleepers go)."""
    cx, cy, cz = centre
    along_z = int(round(facing / 90.0)) % 2 == 0
    hx, hz = (0.45, 1.0) if along_z else (1.0, 0.45)
    return PropDef(name, (cx - hx, cy, cz - hz), (cx + hx, cy + 0.3, cz + hz), texture="cloth", kind="bedroll", facing=facing,
                   owner=owner, solid=True)


def rock_steps(name: str, x1: float, x2: float, z_start: float, z_dir: float, top: float, bottom: float,
               depth: float = 0.6, texture: str = "cave_rock") -> List[PropDef]:
    """Steps cut in the rock from a floor at ``top`` down to one at ``bottom`` (heights above the
    lower floor: ``bottom`` is usually 0), running from ``z_start`` in the direction ``z_dir`` (+1/-1)
    between x1 and x2 - the whole width of the way through, so no gap beside them looks down past
    the floors. Walkable (``kind="step"``), each no more than a step's rise; the first is flush with
    the upper floor (its front closes the drop under that floor's edge, and a walker steps off it
    onto the floor on the level)."""
    rise = top - bottom
    count = max(1, int(math.ceil(rise / STEP_RISE)))
    steps = []
    for k in range(count):
        y = top - k * rise / count
        za = z_start + z_dir * k * depth
        zb = za + z_dir * depth
        steps.append(PropDef(f"{name}_{k}", (x1, bottom - 0.3, min(za, zb)), (x2, y, max(za, zb)), texture=texture, kind="step",
                             solid=False))
    return steps


def skeleton(name: str, label: str, centre: Vec, items=(), along_x: bool = False, owner: Optional[str] = None) -> ContainerDef:
    """Somebody's bones where they fell, something still about them (searched like a chest)."""
    cx, cy, cz = centre
    hx, hz = (1.0, 0.45) if along_x else (0.45, 1.0)
    return ContainerDef(name, label, (cx - hx, cy, cz - hz), (cx + hx, cy + 0.3, cz + hz), items=tuple(items), owner=owner,
                        texture="bone", kind="skeleton")


def stash(name: str, label: str, centre: Vec, items=(), owner: Optional[str] = None, locked: bool = False,
          lock_difficulty: int = 1) -> ContainerDef:
    """A hiding place - loose stones over a hollow, a gap behind a beam: found only by somebody
    crouched close and searching (``hidden``)."""
    cx, cy, cz = centre
    return ContainerDef(name, label, (cx - 0.4, cy, cz - 0.4), (cx + 0.4, cy + 0.45, cz + 0.4), items=tuple(items), owner=owner,
                        texture="cave_rock", kind="stash", hidden=True, locked=locked, lock_difficulty=lock_difficulty)


def sacks(name: str, label: str, x1: float, z1: float, x2: float, z2: float, items=(), owner=None, y: float = 0.0,
          crime_name: str = "Theft") -> ContainerDef:
    return ContainerDef(name, label, (x1, y, z1), (x2, y + 1.0, z2), items=tuple(items), owner=owner, texture="hay", kind="sacks",
                        crime_name=crime_name)


def crates(name: str, x1: float, z1: float, x2: float, z2: float, h: float = 1.4, y: float = 0.0) -> PropDef:
    return PropDef(name, (x1, y, z1), (x2, y + h, z2), texture="crate", kind="crates")


def signpost(name: str, x: float, z: float, lines: Sequence[str], label: str = "Signpost") -> ReadableDef:
    """A fingerpost where ways part: what lies down each of them."""
    return ReadableDef(name, label, (x - 0.2, 0.0, z - 0.2), (x + 0.2, 2.6, z + 0.2), tuple(lines), kind="signpost")


def notice_board(name: str, label: str, x: float, z: float, lines: Sequence[str], facing: float = 0.0) -> ReadableDef:
    """A board on two posts with notices pinned to it (its face looks the way of ``facing``)."""
    along_x = int(round(facing / 90.0)) % 2 == 0
    hx, hz = (0.9, 0.2) if along_x else (0.2, 0.9)
    return ReadableDef(name, label, (x - hx, 0.0, z - hz), (x + hx, 2.3, z + hz), tuple(lines), kind="notice", facing=facing)


def plaque(name: str, label: str, x: float, z: float, facing: float, lines: Sequence[str], y: float = 1.6,
           width: float = 0.8) -> ReadableDef:
    """A lettered board on a wall face at (x, z), ``facing`` the way the face looks."""
    along_x = int(round(facing / 90.0)) % 2 == 0
    hx, hz = (width / 2, 0.05) if along_x else (0.05, width / 2)
    a = math.radians(facing)
    x, z = x + math.sin(a) * 0.06, z - math.cos(a) * 0.06
    return ReadableDef(name, label, (x - hx, y, z - hz), (x + hx, y + 0.5, z + hz), tuple(lines), kind="plaque", facing=facing,
                       solid=False)


def journal(name: str, label: str, x: float, y: float, z: float, lines: Sequence[str], facing: float = 0.0) -> ReadableDef:
    """An open book or a few pages left lying at (x, y, z) - a table top, a ledge, a dead man's hand."""
    return ReadableDef(name, label, (x - 0.25, y, z - 0.2), (x + 0.25, y + 0.06, z + 0.2), tuple(lines), kind="book",
                       facing=facing, solid=False)


def dummy(name: str, x: float, z: float, facing: float = 0.0) -> PropDef:
    return PropDef(name, (x - 0.35, 0.0, z - 0.35), (x + 0.35, 1.9, z + 0.35), texture="hay", kind="dummy", facing=facing)


def rowboat(name: str, x: float, z: float, facing: float = 0.0) -> PropDef:
    along_z = int(round(facing / 90.0)) % 2 == 0
    hx, hz = (0.75, 2.1) if along_z else (2.1, 0.75)
    return PropDef(name, (x - hx, 0.0, z - hz), (x + hx, 0.75, z + hz), texture="building", kind="rowboat", facing=facing)


def banner(name: str, x: float, z: float, facing: float, device: str, y: float = 2.0, height: float = 2.2,
           width: float = 0.9) -> PropDef:
    """A banner hanging against a wall face at (x, z) (its face looks the way of ``facing``)."""
    along_x = int(round(facing / 90.0)) % 2 == 0
    a = math.radians(facing)
    x, z = x + math.sin(a) * 0.15, z - math.cos(a) * 0.15
    hx, hz = (width / 2, 0.06) if along_x else (0.06, width / 2)
    return PropDef(name, (x - hx, y, z - hz), (x + hx, y + height, z + hz), texture=device, kind="banner", facing=facing,
                   solid=False)


def pool(name: str, x: float, z: float, r: float = 1.4, y: float = 0.0) -> PropDef:
    return PropDef(name, (x - r, y, z - r), (x + r, y + 0.3, z + r), texture="lake", kind="pool", solid=False)


def timber(name: str, x: float, z: float, span: float, height: float, along_x: bool = True) -> PropDef:
    """Pit props: two posts and a cap across a gallery ``span`` wide (along x, or along z)."""
    hx, hz = (span / 2, 0.15) if along_x else (0.15, span / 2)
    return PropDef(name, (x - hx, 0.0, z - hz), (x + hx, height, z + hz), texture="wood", kind="timber",
                   facing=0.0 if along_x else 90.0, solid=False)


def minecart(name: str, x: float, z: float, facing: float = 0.0) -> PropDef:
    along_z = int(round(facing / 90.0)) % 2 == 0
    hx, hz = (0.5, 0.75) if along_z else (0.75, 0.5)
    return PropDef(name, (x - hx, 0.0, z - hz), (x + hx, 1.0, z + hz), texture="wood", kind="minecart", facing=facing)


def throne(name: str, x: float, z: float, facing: float = 0.0) -> PropDef:
    return PropDef(name, (x - 0.55, 0.0, z - 0.5), (x + 0.55, 1.9, z + 0.5), texture="wood", kind="throne", facing=facing)


# --- stairs ------------------------------------------------------------------------------------
@dataclass
class Flight:
    props: List[PropDef]
    stair: StairDef
    hole: Tuple[float, float, float, float]        # the stairwell in the floor above (x1, z1, x2, z2)
    landing: Vec


def straight_stair(name: str, x1: float, x2: float, z_foot: float, top_y: float, region: Tuple[float, float, float, float],
                   depth: float = 0.55, landing_offset: Tuple[float, float] = (1.4, -0.9)) -> Flight:
    """A straight flight between x1 and x2 climbing north (towards -z) from ``z_foot`` up to an
    upper floor at ``top_y``. Returns the steps (walkable ``step`` props), the ``StairDef`` NPCs
    follow (``region`` is the building: above its floor there, a walker is upstairs), the
    stairwell to leave open in the floor above and the landing spot beside the top step."""
    count = max(2, int(round(top_y / STEP_RISE)))
    rise = top_y / count
    props: List[PropDef] = []
    waypoints: List[Vec] = []
    cx = (x1 + x2) / 2.0
    for k in range(count):
        z2 = z_foot - k * depth
        z1 = z2 - depth
        y = rise * (k + 1)
        props.append(PropDef(f"{name}_step_{k}", (x1, y - STEP_THICKNESS, z1), (x2, y, z2), texture="wood", kind="step",
                             solid=False))
        waypoints.append((cx, y, (z1 + z2) / 2.0))
    z_top = z_foot - count * depth
    landing = (x2 + landing_offset[0] - (x2 - x1) / 2.0, top_y, z_top + landing_offset[1])
    waypoints.append(landing)
    bottom = (cx, 0.0, z_foot + 0.7)
    hole = (x1, z_top, x2, z_foot)
    return Flight(props, StairDef(name, bottom, landing, tuple(waypoints), region), hole, landing)


def upper_floor(name: str, x1: float, z1: float, x2: float, z2: float, y: float, holes: Sequence[Tuple[float, float, float, float]],
                texture: str = "wood") -> List[PropDef]:
    """The boards of an upper storey over the interior (x1, z1)-(x2, z2), less the stairwells."""
    from .tower import _subtract
    boxes = [(x1, z1, x2, z2)]
    for hole in holes:
        boxes = [piece for box in boxes for piece in _subtract(box, hole)]
    return [PropDef(f"{name}_{i}", (bx1, y - FLOOR_THICKNESS, bz1), (bx2, y, bz2), texture=texture, kind="floor", solid=False)
            for i, (bx1, bz1, bx2, bz2) in enumerate(boxes)]


# --- schedules ---------------------------------------------------------------------------------
def span(rng, state, text, pos, **kw) -> List[TaskDef]:
    """Tasks for a (start, end) hour range that may wrap past midnight."""
    start, end = rng
    if start < end:
        return [TaskDef(start, end, pos, state, text, **kw)]
    return [TaskDef(start, 24, pos, state, text, **kw), TaskDef(0, end, pos, state, text, **kw)]


def resident_tasks(name: str, home_door: Optional[str], bed_spot, table_seat, food: str, day: List[TaskDef],
                   wake: int = 7, bedtime: int = 21, facing: float = 0.0,
                   supper: Optional[Tuple[int, int]] = None) -> Tuple[TaskDef, ...]:
    """Frame a resident's day: breakfast at home (unlocking the door), the ``day`` tasks, supper at
    home (the hour before ``bedtime``, or the ``supper`` hours - two of them for somebody whose
    evening ends at the far end of town) and bed (locking up; nobody who stays up until midnight
    has a bedtime task: the night's sleep from 0 carries them). Everyone who is not a guard lives
    like this (``home_door`` None: somebody with no door to lock - a stable boy in the hay)."""
    door = (home_door,) if home_door else ()
    start, end = supper if supper is not None else (bedtime - 1, bedtime)
    tasks = [
        TaskDef(0, wake, bed_spot, AIState.SLEEPING, f"{name} is asleep at home.", lock_doors=door),
        TaskDef(wake, wake + 1, table_seat, AIState.EATING, f"{name} is having breakfast.", unlock_doors=door,
                facing=facing, food_container=food),
    ]
    tasks += day
    tasks.append(TaskDef(start, end, table_seat, AIState.EATING, f"{name} is having supper at home.",
                         unlock_doors=door, facing=facing, food_container=food))
    if end < 24:
        tasks.append(TaskDef(end, 24, bed_spot, AIState.SLEEPING, f"{name} {'locked up and went' if door else 'went'} to bed.",
                             lock_doors=door))
    return tuple(tasks)


def stroll(start: int, end: int, name: str, route, text: str = None) -> TaskDef:
    return TaskDef(start, end, route[0], AIState.STROLLING, text or f"{name} is out for a stroll.", patrol_points=route)


def visit(start: int, end: int, spot, state: AIState, text: str, facing: float = 0.0, food: str = None) -> TaskDef:
    return TaskDef(start, end, spot, state, text, facing=facing, food_container=food)
