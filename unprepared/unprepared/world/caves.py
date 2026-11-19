"""Caves dug into the hillside: chambers under the ground, a cutting down to the way in.

A cave is a set of *chambers* - rectangles, each an underground ``BuildingDef`` with its floor
at an absolute height (``Terrain`` keeps the hillside over it clear of its ceiling) - opening
into one another through gaps in their walls, and a *mouth*: a gap in an outer wall at the end
of a trail graded down into the slope (``TrailDef.end_height`` = the floor), where the ground
is not drawn (``TerrainDef.holes``) and rock stands up to the hillside over the way in
(``WallDef.reach_surface``).

:func:`dig` turns that outline into level data. Nothing about it is square: the walls are cut
into short lengths set back by random amounts, boulders lie half sunk against them, columns of
rock grow up from the floor and hang from the ceiling - seeded, so a cave is the same every
game. What lives in it (fires, bedrolls, ore, the dead) the level adds itself.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from .level import BuildingDef, LightDef, PropDef, WallDef

Rect = Tuple[float, float, float, float]        # x1, z1, x2, z2

WALL_T = 0.7               # the rock wall's least thickness inside a chamber's outline
WALL_JITTER = 0.25         # ... and how much further in a length of it may stand
MOUTH_HEIGHT = 3.4         # the way in (and every gap between chambers) is this tall
PASSAGE_HEIGHT = 3.2
MOUTH_TEXTURE = "cliff"    # the rock round the way in is the hillside's own
JAMB = 1.5                 # the length of wall either side of the way in that stands up to the hillside
#                            (the trail's cutting is that much wider than the way in: its banks are
#                            left out of the drawn ground over the mouth, and the jambs stand in for them)


@dataclass(frozen=True)
class Chamber:
    key: str                   # its building's name
    rect: Rect
    floor: float               # absolute height of its floor
    height: float = 4.5        # floor to the rock overhead


@dataclass(frozen=True)
class Mouth:
    """The way in: a gap ``(a, b)`` along one ``side`` ("north", "south", "east", "west") of a chamber."""
    chamber: str
    side: str
    a: float
    b: float


@dataclass
class Dug:
    walls: List[WallDef] = field(default_factory=list)
    buildings: List[BuildingDef] = field(default_factory=list)
    props: List[PropDef] = field(default_factory=list)
    lights: List[LightDef] = field(default_factory=list)
    holes: List[Rect] = field(default_factory=list)


def _overlaps(a1: float, a2: float, b1: float, b2: float) -> bool:
    return a1 < b2 and b1 < a2


def _touches(rect: Rect, gap: Rect) -> bool:
    return _overlaps(rect[0], rect[2], gap[0], gap[2]) and _overlaps(rect[1], rect[3], gap[1], gap[3])


def _side_band(rect: Rect, side: str, thickness: float) -> Rect:
    x1, z1, x2, z2 = rect
    return {"north": (x1, z1, x2, z1 + thickness), "south": (x1, z2 - thickness, x2, z2),
            "west": (x1, z1, x1 + thickness, z2), "east": (x2 - thickness, z1, x2, z2)}[side]


def dig(prefix: str, chambers: Sequence[Chamber], passages: Sequence[Rect], mouth: Mouth, seed: int,
        texture: str = "cave_rock", floor_texture: str = "cave_floor", dressing: float = 1.0) -> Dug:
    """Level data for a cave (see the module docstring). ``passages`` are rectangles where no wall
    stands - the gaps between chambers that touch; ``mouth`` is the way in from the outside."""
    rng = random.Random(seed)
    out = Dug()
    # a chamber lower than one it opens into must reach up past the top of the way between them
    # (else the gap over its ceiling would open on nothing)
    from dataclasses import replace
    chambers = list(chambers)
    for i, c in enumerate(chambers):
        for p in passages:
            if not _touches(c.rect, p):
                continue
            highest = max(o.floor for o in chambers if _touches(o.rect, p))
            need = highest + PASSAGE_HEIGHT + 0.4 - c.floor
            if c.height < need:
                c = chambers[i] = replace(c, height=need)
    by_key = {c.key: c for c in chambers}
    gaps: List[Tuple[str, Rect]] = [("", p) for p in passages]
    m = by_key[mouth.chamber]
    mx1, mz1, mx2, mz2 = m.rect
    if mouth.side in ("north", "south"):
        z_edge = mz1 if mouth.side == "north" else mz2
        mouth_rect = (mouth.a, z_edge - 1.2, mouth.b, z_edge + 1.2)
        a, b = max(mx1, mouth.a - JAMB), min(mx2, mouth.b + JAMB)          # (no further than the chamber's face)
        hole = (a, z_edge - 1.0, b, z_edge) if mouth.side == "south" else (a, z_edge, b, z_edge + 1.0)
    else:
        x_edge = mx1 if mouth.side == "west" else mx2
        mouth_rect = (x_edge - 1.2, mouth.a, x_edge + 1.2, mouth.b)
        a, b = max(mz1, mouth.a - JAMB), min(mz2, mouth.b + JAMB)
        hole = (x_edge - 1.0, a, x_edge, b) if mouth.side == "east" else (x_edge, a, x_edge + 1.0, b)
    gaps.append(("mouth", mouth_rect))
    out.holes.append(hole)

    for c in chambers:
        x1, z1, x2, z2 = c.rect
        out.buildings.append(BuildingDef(c.key, (x1, z1), (x2, z2), wall_height=c.height, roof_style="flat", roof_texture=texture,
                                         floor_texture=floor_texture, lit=False, underground=True, elevation=c.floor))
        for side in ("north", "south", "west", "east"):
            out.walls += _wall_run(c, side, gaps, rng, texture, reach=(c.key == mouth.chamber and side == mouth.side),
                                   face_texture=MOUTH_TEXTURE)
        # a lintel over every gap in this chamber's walls: rock down to the top of the way through
        # (a way between chambers on two levels is as tall as the higher floor needs)
        for kind, (gx1, gz1, gx2, gz2) in gaps:
            joined = [o.floor for o in chambers if _touches(o.rect, (gx1, gz1, gx2, gz2))]
            for side in ("north", "south", "west", "east"):
                # (over the mouth the rock reaches back a whole cell of the ground: the hillside's
                # face is not drawn over the way in, and this stands in for it)
                depth = 1.0 if (kind == "mouth" and c.key == mouth.chamber and side == mouth.side) else WALL_T
                bx1, bz1, bx2, bz2 = _side_band(c.rect, side, depth)
                if not (_overlaps(gx1, gx2, bx1, bx2) and _overlaps(gz1, gz2, bz1, bz2)):
                    continue
                top = MOUTH_HEIGHT if kind == "mouth" else max(joined + [c.floor]) + PASSAGE_HEIGHT - c.floor
                lx1, lz1, lx2, lz2 = max(gx1, bx1), max(gz1, bz1), min(gx2, bx2), min(gz2, bz2)
                if top < c.height:
                    at_mouth = kind == "mouth" and c.key == mouth.chamber
                    out.walls.append(WallDef((lx1, top, lz1), (lx2, c.height, lz2), texture=MOUTH_TEXTURE if at_mouth else texture,
                                             reach_surface=at_mouth))
        if dressing > 0.0:
            out.props += _dress(prefix, c, gaps, rng, texture, dressing)
    out.props += _mouth_rocks(prefix, m, mouth, rng)
    return out


def _mouth_rocks(prefix: str, c: Chamber, mouth: Mouth, rng: random.Random) -> List[PropDef]:
    """Crags heaped either side of the way in and a brow of rock over it: the mouth of a cave is
    a hole in a hillside, not a doorway in a wall."""
    x1, z1, x2, z2 = c.rect
    a, b = mouth.a, mouth.b
    out = 1.0 if mouth.side in ("south", "east") else -1.0          # which way is outside
    spots = [(a - 1.4, 0.7, 1.9, 3.2), (b + 1.4, 0.6, 1.8, 2.8), (a - 3.4, 1.4, 1.4, 2.0), (b + 3.5, 1.2, 1.5, 2.2),
             (a - 2.2, 2.6, 0.9, 1.1), (b + 2.6, 2.4, 1.0, 1.2)]
    props = []
    for i, (along, off, r, h) in enumerate(spots):
        if mouth.side in ("north", "south"):
            edge = z2 if mouth.side == "south" else z1
            x, z = along, edge + out * off
        else:
            edge = x2 if mouth.side == "east" else x1
            x, z = edge + out * off, along
        r *= rng.uniform(0.9, 1.1)
        # (wholly outside the chamber: a rock half in it would be half in the cave's shade, cut
        # dark along the line of the hillside's face)
        push = max(0.0, r + 0.02 - off)
        if mouth.side in ("north", "south"):
            z += out * push
        else:
            x += out * push
        props.append(PropDef(f"{prefix}_mouth_crag_{i}", (x - r, 0.0, z - r), (x + r, h, z + r), texture=MOUTH_TEXTURE,
                             kind="boulder"))
    mid = (a + b) / 2.0
    if mouth.side in ("north", "south"):
        edge = z2 if mouth.side == "south" else z1
        bx, bz, hx, hz = mid, edge + out * 0.92, (b - a) / 2.0 + 1.2, 0.9
    else:
        edge = x2 if mouth.side == "east" else x1
        bx, bz, hx, hz = edge + out * 0.92, mid, 0.9, (b - a) / 2.0 + 1.2
    props.append(PropDef(f"{prefix}_mouth_brow", (bx - hx, MOUTH_HEIGHT - 0.35, bz - hz), (bx + hx, MOUTH_HEIGHT + 1.5, bz + hz),
                         texture=MOUTH_TEXTURE, kind="boulder", solid=False))
    return props


def _wall_run(c: Chamber, side: str, gaps, rng: random.Random, texture: str, reach: bool,
              face_texture: str = "cliff") -> List[WallDef]:
    """One side of a chamber: lengths of rock set back by random amounts, broken where a gap is."""
    x1, z1, x2, z2 = c.rect
    along_x = side in ("north", "south")
    a0, a1 = (x1, x2) if along_x else (z1, z2)
    band = _side_band(c.rect, side, WALL_T + WALL_JITTER)
    cuts = []
    for _, (gx1, gz1, gx2, gz2) in gaps:
        if _overlaps(gx1, gx2, band[0], band[2]) and _overlaps(gz1, gz2, band[1], band[3]):
            cuts.append((gx1, gx2) if along_x else (gz1, gz2))
    walls = []
    a = a0
    while a < a1 - 1e-6:
        b = min(a1, a + rng.uniform(1.4, 2.8))
        if a1 - b < 0.8:
            b = a1
        pieces = [(a, b)]
        for g1, g2 in cuts:
            pieces = [q for p in pieces for q in ((p[0], min(p[1], g1)), (max(p[0], g2), p[1])) if q[1] - q[0] > 0.05]
        depth = WALL_T + rng.uniform(0.0, WALL_JITTER)
        near_mouth = reach and any(_overlaps(p0 - 1.5, p1 + 1.5, g1, g2) for p0, p1 in pieces for g1, g2 in cuts)
        tex = face_texture if near_mouth else texture
        for p0, p1 in pieces:
            # beside the way in, a jamb of rock stands up to the hillside (the ground is not drawn
            # over the mouth: this is the side of the notch); the rest stays under the ground
            spans = [(p0, p1, False)]
            if near_mouth:
                spans = []
                lo, hi = p0, p1
                if any(abs(p0 - g2) < 1e-6 for _, g2 in cuts) and p1 - p0 > JAMB:
                    spans.append((p0, p0 + JAMB, True))
                    lo = p0 + JAMB
                if any(abs(p1 - g1) < 1e-6 for g1, _ in cuts) and hi - lo > JAMB:
                    spans.append((p1 - JAMB, p1, True))
                    hi = p1 - JAMB
                touches = any(abs(p0 - g2) < 1e-6 or abs(p1 - g1) < 1e-6 for g1, g2 in cuts)
                spans.append((lo, hi, touches and hi - lo <= JAMB + 1e-6))
            for s0, s1, up in spans:
                if along_x:
                    za, zb = (z1, z1 + depth) if side == "north" else (z2 - depth, z2)
                    walls.append(WallDef((s0, 0.0, za), (s1, c.height, zb), texture=tex, reach_surface=up))
                else:
                    xa, xb = (x1, x1 + depth) if side == "west" else (x2 - depth, x2)
                    walls.append(WallDef((xa, 0.0, s0), (xb, c.height, s1), texture=tex, reach_surface=up))
        a = b
    return walls


def _in_gap(x: float, z: float, gaps, margin: float = 1.2) -> bool:
    return any(gx1 - margin <= x <= gx2 + margin and gz1 - margin <= z <= gz2 + margin for _, (gx1, gz1, gx2, gz2) in gaps)


def _lanes(c: Chamber, gaps) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """The ways across a chamber that must stay clear: from the middle of every gap in its walls
    to the middle of the chamber (nobody picks their way round a boulder in the doorway)."""
    x1, z1, x2, z2 = c.rect
    mid = ((x1 + x2) / 2.0, (z1 + z2) / 2.0)
    out = []
    for _, (gx1, gz1, gx2, gz2) in gaps:
        if _touches(c.rect, (gx1, gz1, gx2, gz2)):
            g = (min(max((gx1 + gx2) / 2.0, x1), x2), min(max((gz1 + gz2) / 2.0, z1), z2))
            out.append((g, mid))
    return out


def _near_lane(x: float, z: float, lanes, reach: float) -> bool:
    for (ax, az), (bx, bz) in lanes:
        sx, sz = bx - ax, bz - az
        t = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / max(sx * sx + sz * sz, 1e-9)))
        if ((x - ax - t * sx) ** 2 + (z - az - t * sz) ** 2) ** 0.5 < reach:
            return True
    return False


NARROW = 7.0               # a chamber narrower than this is a tunnel: nothing on its floor to trip over


def _dress(prefix: str, c: Chamber, gaps, rng: random.Random, texture: str, amount: float) -> List[PropDef]:
    """Rock against the walls and growing from floor and ceiling, clear of every way through
    (and of the floor of a tunnel altogether)."""
    x1, z1, x2, z2 = c.rect
    props: List[PropDef] = []
    tunnel = min(x2 - x1, z2 - z1) < NARROW
    lanes = _lanes(c, gaps)
    inner = (x1 + WALL_T + WALL_JITTER, z1 + WALL_T + WALL_JITTER, x2 - WALL_T - WALL_JITTER, z2 - WALL_T - WALL_JITTER)
    perimeter = 2 * ((x2 - x1) + (z2 - z1))
    n_boulders = int(perimeter / 3.2 * amount)
    for i in range(n_boulders):
        side = rng.choice(("north", "south", "west", "east"))
        if side in ("north", "south"):
            x = rng.uniform(inner[0], inner[2])
            z = inner[1] + 0.15 if side == "north" else inner[3] - 0.15
        else:
            z = rng.uniform(inner[1], inner[3])
            x = inner[0] + 0.15 if side == "west" else inner[2] - 0.15
        if tunnel or _in_gap(x, z, gaps) or _near_lane(x, z, lanes, 2.4):
            continue
        r = rng.uniform(0.5, 1.05)
        props.append(PropDef(f"{prefix}_{c.key}_rock_{i}", (x - r, 0.0, z - r), (x + r, rng.uniform(0.6, 1.5), z + r),
                             texture=texture, kind="boulder"))
    # stalagmites near the walls, stalactites over them and here and there across the roof
    n_mites = int((x2 - x1) * (z2 - z1) / 28.0 * amount)
    for i in range(n_mites):
        x = rng.uniform(inner[0] + 0.3, inner[2] - 0.3)
        z = rng.uniform(inner[1] + 0.3, inner[3] - 0.3)
        edge = min(x - inner[0], inner[2] - x, z - inner[1], inner[3] - z)
        if _in_gap(x, z, gaps, 1.6):
            continue
        if edge < 1.6 and rng.random() < 0.75 and not tunnel and not _near_lane(x, z, lanes, 2.4):
            r = rng.uniform(0.18, 0.38)
            props.append(PropDef(f"{prefix}_{c.key}_mite_{i}", (x - r, 0.0, z - r), (x + r, rng.uniform(0.7, 2.0), z + r),
                                 texture=texture, kind="stalagmite"))
        r = rng.uniform(0.15, 0.32)
        length = rng.uniform(0.5, 1.3) if edge > 1.6 else rng.uniform(0.7, 1.8)
        props.append(PropDef(f"{prefix}_{c.key}_tite_{i}", (x - r, c.height - length, z - r), (x + r, c.height, z + r),
                             texture=texture, kind="stalactite", solid=False))
    return props


DRESSING_KINDS = ("boulder", "stalagmite")


def is_dressing(prop: PropDef) -> bool:
    """A rock or stalagmite ``dig`` strewed over a chamber's floor (not one set there on purpose)."""
    return prop.kind in DRESSING_KINDS and ("_rock_" in prop.name or "_mite_" in prop.name)


def _overlap_3d(a, b) -> bool:
    return all(a.min_point[i] < b.max_point[i] and b.min_point[i] < a.max_point[i] for i in range(3))


def clear_the_floor(props: Sequence[PropDef], things: Sequence, spots: Sequence[Tuple[float, float]] = (),
                    margin: float = 0.25, reach: float = 0.9) -> List[PropDef]:
    """``props`` less any dressing that would stand in something furnished afterwards - a chest,
    a bedroll, a seam, a table (``things``: anything with ``min_point``/``max_point``) - or on a
    spot somebody sleeps, works or stands on (``spots``), and any stalactite hanging into one."""
    keep = []
    boxes = [(t.min_point[0] - margin, t.min_point[2] - margin, t.max_point[0] + margin, t.max_point[2] + margin)
             for t in things if not (isinstance(t, PropDef) and is_dressing(t))]
    solids = [t for t in things if not (isinstance(t, PropDef) and (is_dressing(t) or t.kind == "stalactite"))]
    for prop in props:
        if prop.kind == "stalactite":
            # one hanging into a pit prop's beam, a banner, a lantern's bracket: gone
            if any(_overlap_3d(prop, t) for t in solids):
                continue
        elif is_dressing(prop):
            x1, z1, x2, z2 = prop.min_point[0], prop.min_point[2], prop.max_point[0], prop.max_point[2]
            if any(x1 < bx2 and bx1 < x2 and z1 < bz2 and bz1 < z2 for bx1, bz1, bx2, bz2 in boxes):
                continue
            if any(x1 - reach < sx < x2 + reach and z1 - reach < sz < z2 + reach for sx, sz in spots):
                continue
        keep.append(prop)
    return keep


def mushrooms(prefix: str, spots: Sequence[Tuple[float, float]], y: float = 0.0, room: str = "",
              glow: bool = True) -> Tuple[List[PropDef], List[LightDef]]:
    """Clusters of pale cave mushrooms, their caps glowing a cold green (a faint light of their own)."""
    props, lights = [], []
    for i, (x, z) in enumerate(spots):
        props.append(PropDef(f"{prefix}_{i}", (x - 0.45, y, z - 0.45), (x + 0.45, y + 0.5, z + 0.45), texture="mushroom",
                             kind="mushroom", solid=False))
        if glow:
            lights.append(LightDef("glow", (x, y + 0.45, z), room=room, range=3.2))
    return props, lights
