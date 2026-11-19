"""Set dressing: the fungus, the trash and the small wreckage that make the facility look
lived in and neglected - placed by a seeded generator with rejection sampling, so the
result is deterministic, nothing intersects a wall, a crate, a desk or another prop, and
nothing solid ever lands in a walkway (everything here is walk-over clutter).

``Placer`` owns the rules: a candidate footprint is accepted only if it is clear of every
blocker (colliders, door obstacles, entity footprints) by a margin and clear of everything
placed before it. ``tests/test_level_data.py`` re-checks the output independently.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from cyberduke.core.geometry import AABB
from cyberduke.world.props import Material, Prop

RGB = tuple[float, float, float]
CLUTTER_TAGS = frozenset({"clutter", "transparent"})

# ----------------------------------------------------------------------------- materials
FUNGUS = Material("fungus", (0.45, 0.43, 0.38))
GLOW_MATERIALS: dict[str, Material] = {
    # the pool of light a cap throws on the ground: an unlit soft disc, full in the middle,
    # fading to nothing at the rim
    "blue": Material(None, (0.3, 0.6, 1.0), alpha=0.5),
    "azure": Material(None, (0.3, 0.85, 1.0), alpha=0.5),
    "violet": Material(None, (0.65, 0.4, 1.0), alpha=0.5),
}
CAP_MATERIALS: dict[str, Material] = {
    # bright bioluminescent blues, a cyan and a rarer violet; the renderer pulses emission
    "blue": Material("spore", (0.35, 0.6, 1.0), emission=(0.25, 1.1, 3.4)),
    "azure": Material("spore", (0.3, 0.8, 1.0), emission=(0.2, 1.8, 3.0)),
    "violet": Material("spore", (0.6, 0.45, 1.0), emission=(1.4, 0.5, 3.2)),
}
SODA = Material("soda_can")
PAPER = Material(None, (0.72, 0.7, 0.62))
CARDBOARD = Material("cardboard", (0.8, 0.72, 0.6))
CABLE = Material(None, (0.08, 0.08, 0.1))
PUDDLE = Material(None, (0.12, 0.2, 0.3), alpha=0.5)  # water: translucent, soft-edged
DEAD_SCREEN = Material(None, (0.05, 0.05, 0.06), specular=(0.6, 0.6, 0.7), shininess=60.0)
CASING = Material("metal", (0.16, 0.17, 0.2))
BOTTLE = Material(None, (0.25, 0.45, 0.35), specular=(0.8, 0.9, 0.85), shininess=90.0)
CHIP = Material(None, (1.0, 0.7, 0.0), emission=(3.0, 1.8, 0.0))
RAG = Material("fabric", (0.4, 0.36, 0.32))


# ----------------------------------------------------------------------------- placement
@dataclass
class Placer:
    """Rejection-sampling placer. ``blockers`` are the things nothing may intersect;
    ``placed`` grows with every accepted footprint so clutter never overlaps clutter."""

    seed: int
    blockers: Sequence[AABB]
    margin: float = 0.06
    placed: list[AABB] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def clear(self, box: AABB, margin: float | None = None) -> bool:
        m = self.margin if margin is None else margin
        for other in self.blockers:
            if _overlap_xz(box, other, m) and _overlap_y(box, other):
                return False
        return all(not (_overlap_xz(box, p, m) and _overlap_y(box, p)) for p in self.placed)

    def sample(
        self, zone: AABB, half_w: float, half_d: float, height: float, floor_y: float = 0.0, tries: int = 40
    ) -> tuple[float, float] | None:
        """A position inside ``zone`` where a ``2*half_w x 2*half_d`` footprint fits."""
        for _ in range(tries):
            x = self.rng.uniform(zone.min_x + half_w, zone.max_x - half_w)
            z = self.rng.uniform(zone.min_z + half_d, zone.max_z - half_d)
            box = AABB(x - half_w, x + half_w, z - half_d, z + half_d, floor_y, floor_y + height)
            if self.clear(box):
                return x, z
        return None

    def commit(self, box: AABB) -> None:
        self.placed.append(box)


def _overlap_xz(a: AABB, b: AABB, margin: float) -> bool:
    return (
        a.min_x - margin < b.max_x
        and b.min_x < a.max_x + margin
        and a.min_z - margin < b.max_z
        and b.min_z < a.max_z + margin
    )


def _overlap_y(a: AABB, b: AABB, eps: float = 0.002) -> bool:
    return a.min_y + eps < b.max_y and b.min_y + eps < a.max_y


# ----------------------------------------------------------------------------- mushrooms
@dataclass(frozen=True)
class MushroomSpec:
    x: float
    y: float
    z: float
    scale: float
    tint: str
    phase: int

    @property
    def cap_radius(self) -> float:
        return 0.26 * self.scale

    @property
    def stalk_height(self) -> float:
        return 0.32 * self.scale

    @property
    def footprint(self) -> AABB:
        r = self.cap_radius
        top = self.y + self.stalk_height + 0.16 * self.scale
        return AABB(self.x - r, self.x + r, self.z - r, self.z + r, self.y, top)


def mushroom_props(m: MushroomSpec) -> tuple[Prop, Prop, Prop]:
    """A bioluminescent mushroom: a pale stalk, a pulsing cap (cone) and the pool of light
    it throws on the ground (a translucent disc the renderer draws in the late pass)."""
    stalk_r = 0.09 * m.scale
    cap = CAP_MATERIALS[m.tint]
    glow_r = m.cap_radius * 2.4
    return (
        Prop.cylinder(m.x, m.y, m.z, stalk_r, stalk_r * 0.75, m.stalk_height, FUNGUS, slices=8, label="Mushroom stalk", tags=CLUTTER_TAGS),
        Prop.cylinder(m.x, m.y + m.stalk_height - 0.02 * m.scale, m.z, m.cap_radius, 0.04 * m.scale, 0.16 * m.scale, cap, slices=10, label="Mushroom cap",
                      tags=CLUTTER_TAGS | {"pulse", f"phase{m.phase}"}),
        Prop.disc(m.x, m.y + 0.006, m.z, glow_r, glow_r, 0.18, GLOW_MATERIALS[m.tint], slices=16, label="Mushroom glow",
                  tags=CLUTTER_TAGS | {"glow"}),
    )


# ----------------------------------------------------------------------------- fungus light
SPLASH_TAGS = CLUTTER_TAGS | {"glow"}
SPLASH_OFF = 0.018  # a splash lies this far in front of the face it lights: clear of a face
# plate riding on it (the racks' are 1.1 cm proud), so the two never share a depth


@dataclass(frozen=True)
class _Face:
    """A vertical face of a box: centre (x, z), outward normal, half width along the
    face, and the height range it covers."""

    x: float
    z: float
    nx: float
    nz: float
    half: float
    y0: float
    y1: float


def _yaw_of(prop: Prop) -> float | None:
    """The prop's rotation about +Y (degrees), or None if it is turned any other way."""
    yaw = 0.0
    for angle, ax, ay, az in prop.rotate:
        if abs(ax) > 1e-9 or abs(az) > 1e-9:
            return None
        yaw += angle if ay > 0 else -angle
    return yaw


def _faces(box: Prop) -> list[_Face]:
    yaw = _yaw_of(box)
    if yaw is None or box.kind != "box":
        return []
    sx, sy, sz = box.size
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    ax, az = (c, -s), (s, c)  # the box's local x and z axes in the world
    y0, y1 = box.y - sy / 2.0, box.y + sy / 2.0
    out = []
    for (nx, nz), depth, half in (
        (ax, sx / 2.0, sz / 2.0), ((-ax[0], -ax[1]), sx / 2.0, sz / 2.0),
        (az, sz / 2.0, sx / 2.0), ((-az[0], -az[1]), sz / 2.0, sx / 2.0),
    ):  # fmt: skip
        out.append(_Face(box.x + nx * depth, box.z + nz * depth, nx, nz, half, y0, y1))
    return out


def _segment_hits(box: AABB, a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """Does the segment a-b pass through the box (slab test, ends excluded)?"""
    t0, t1 = 0.02, 0.98
    for lo, hi, p, q in ((box.min_x, box.max_x, a[0], b[0]), (box.min_y, box.max_y, a[1], b[1]),
                         (box.min_z, box.max_z, a[2], b[2])):  # fmt: skip
        d = q - p
        if abs(d) < 1e-12:
            if p <= lo or p >= hi:
                return False
            continue
        u0, u1 = (lo - p) / d, (hi - p) / d
        t0, t1 = max(t0, min(u0, u1)), min(t1, max(u0, u1))
        if t0 >= t1:
            return False
    return True


def light_fungus(props: Sequence[Prop], surfaces: Sequence[Prop]) -> list[Prop]:
    """Lays each mushroom's light where it really falls. ``props`` are mushroom props as
    ``mushroom_props`` makes them (stalk, cap, pool); ``surfaces`` the boxes around them.

    * The pool lies on top of whatever the mushroom grows on - above the moss on a planter,
      not under it (where it could not be seen) - and stops at that surface's edges, so it
      never hangs in the air past the rim of a planter.
    * Every wall face near enough to the cap, and facing it, gets a splash of the same
      light: a soft disc on the face, centred level with the cap, as wide as the light
      reaches at that distance, fainter the further away the face is, cut to the face and
      to what is above the ground the mushroom stands on.
    """
    boxes = [
        p for p in surfaces
        if p.kind == "box" and not (p.tags & {"clutter", "fog", "transparent"}) and _yaw_of(p) is not None
    ]  # fmt: skip
    feet = [(p, p.footprint()) for p in boxes]
    out: list[Prop] = []
    cap: Prop | None = None
    for prop in props:
        if prop.label == "Mushroom cap":
            cap = prop
        if prop.label != "Mushroom glow" or cap is None:
            out.append(prop)
            continue
        out.extend(_pool_and_splashes(prop, cap, feet))
    return out


def _pool_and_splashes(pool: Prop, cap: Prop, feet: list[tuple[Prop, AABB]]) -> list[Prop]:
    radius = pool.size[0]
    ground = pool.y - 0.006  # the mushroom's own floor (as mushroom_props placed the pool)
    x, z = pool.x, pool.z
    # what it grows on: boxes under the foot whose top is (about) the ground
    under = [
        (p, f) for p, f in feet
        if f.min_x <= x <= f.max_x and f.min_z <= z <= f.max_z and abs(f.max_y - ground) <= 0.06
    ]  # fmt: skip
    out: list[Prop] = []
    if under:
        top = max(f.max_y for _, f in under)
        _, support = max(under, key=lambda pf: (pf[1].max_x - pf[1].min_x) * (pf[1].max_z - pf[1].min_z))
        clip = (
            (support.min_x - x) / radius, (support.max_x - x) / radius,
            (support.min_z - z) / radius, (support.max_z - z) / radius,
        )  # fmt: skip
        if clip[0] <= -1.0 and clip[1] >= 1.0 and clip[2] <= -1.0 and clip[3] >= 1.0:
            clip = None  # the whole disc fits: no need to cut it
        out.append(replace(pool, y=top + 0.008, clip=clip))
    else:
        top = ground
        out.append(pool)
    # splashes up the faces around it
    light = (x, cap.y + cap.size[2] * 0.5, z)
    base_alpha = pool.material.alpha
    supports = {id(p) for p, _ in under}
    for box, foot in feet:
        if id(box) in supports or foot.max_y <= top + 0.05 or box.size[1] < 0.12:
            continue
        if foot.min_x > x + radius or foot.max_x < x - radius or foot.min_z > z + radius or foot.max_z < z - radius:
            continue
        for face in _faces(box):
            d = (x - face.x) * face.nx + (z - face.z) * face.nz  # in front of the face by
            if not 0.02 < d < radius * 0.92:
                continue
            tx, tz = -face.nz, face.nx  # along the face
            t = (x - face.x) * tx + (z - face.z) * tz
            reach = math.sqrt(radius * radius - d * d)
            if abs(t) - reach >= face.half:
                continue
            foot_pt = (face.x + tx * t + face.nx * SPLASH_OFF, light[1], face.z + tz * t + face.nz * SPLASH_OFF)
            if any(
                other is not box and _segment_hits(f, light, foot_pt) for other, f in feet
                if f.min_x <= max(x, foot_pt[0]) and f.max_x >= min(x, foot_pt[0])
                and f.min_z <= max(z, foot_pt[2]) and f.max_z >= min(z, foot_pt[2])
            ):  # fmt: skip
                continue  # something between the cap and the face
            y0, y1 = max(face.y0, top), face.y1
            # the disc's u axis runs against the face's t axis (see the rotation below)
            clip = ((t - face.half) / reach, (t + face.half) / reach, (y0 - light[1]) / reach, (y1 - light[1]) / reach)
            if clip[1] <= -1.0 or clip[0] >= 1.0 or clip[3] <= -1.0 or clip[2] >= 1.0:
                continue
            strength = (1.0 - d / radius) ** 1.5
            material = replace(pool.material, alpha=round(base_alpha * 0.85 * strength, 3))
            yaw = math.degrees(math.atan2(face.nx, face.nz))
            out.append(Prop.disc(
                foot_pt[0], foot_pt[1], foot_pt[2], reach, reach, 0.12, material, slices=16,
                rotate=((yaw, 0.0, 1.0, 0.0), (-90.0, 1.0, 0.0, 0.0)), clip=clip,
                label="Mushroom glow on the wall", tags=SPLASH_TAGS,
            ))  # fmt: skip
    return out


def mushroom_cluster(
    placer: Placer,
    zone: AABB,
    count: int,
    scale_range: tuple[float, float],
    floor_y: float = 0.0,
    spread: float | None = None,
) -> list[Prop]:
    """A clump of ``count`` mushrooms of varying size around one spot in ``zone``. Fungus
    never grows in rows: sizes, offsets and tints all come from the seed, and every cap is
    checked against walls, props and the other caps so nothing clips."""
    rng = placer.rng
    big = scale_range[1]
    cx_cz = placer.sample(zone, 0.26 * big, 0.26 * big, 0.5 * big, floor_y)
    if cx_cz is None:
        return []
    cx, cz = cx_cz
    spread = spread if spread is not None else 0.25 + 0.18 * count
    props: list[Prop] = []
    placed_here = 0
    for _ in range(count * 6):
        if placed_here >= count:
            break
        scale = rng.uniform(*scale_range)
        if placed_here == 0:
            x, z = cx, cz
        else:
            a = rng.uniform(0.0, math.tau)
            d = rng.uniform(0.2, spread)
            x, z = cx + math.cos(a) * d, cz + math.sin(a) * d
        spec = MushroomSpec(x, floor_y, z, scale, _tint(rng), rng.randrange(7))
        box = spec.footprint
        inside = zone.min_x <= box.min_x and box.max_x <= zone.max_x and zone.min_z <= box.min_z and box.max_z <= zone.max_z
        if not inside or not placer.clear(box, 0.03):
            continue
        placer.commit(box)
        props.extend(mushroom_props(spec))
        placed_here += 1
    return props


def _tint(rng: random.Random) -> str:
    roll = rng.random()
    return "violet" if roll < 0.12 else ("azure" if roll < 0.45 else "blue")


# ----------------------------------------------------------------------------- trash
def _yaw(rng: random.Random) -> float:
    return rng.uniform(0.0, 360.0)


def scatter_trash(
    placer: Placer, zone: AABB, count: int, kinds: Sequence[str], floor_y: float = 0.0
) -> list[Prop]:
    """``count`` pieces of junk from ``kinds`` dropped across ``zone`` - cans on their side,
    crumpled paper, flattened boxes, loose cable, puddles, datachips, a bottle, a rag.
    All walk-over clutter: nothing here has a collider or blocks a route."""
    rng = placer.rng
    props: list[Prop] = []
    for _ in range(count):
        kind = rng.choice(list(kinds))
        maker = _TRASH[kind]
        props.extend(maker(placer, zone, floor_y))
    return props


def _can(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    pos = placer.sample(zone, 0.08, 0.08, 0.08, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - 0.08, x + 0.08, z - 0.08, z + 0.08, y, y + 0.08))
    return [Prop.cylinder(x, y + 0.035, z, 0.035, 0.035, 0.12, SODA, slices=8,
                          rotate=((_yaw(placer.rng), 0.0, 1.0, 0.0), (90.0, 1.0, 0.0, 0.0)),
                          label="Dropped soda can", tags=CLUTTER_TAGS)]


def _paper(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    pos = placer.sample(zone, 0.18, 0.18, 0.01, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - 0.18, x + 0.18, z - 0.18, z + 0.18, y, y + 0.01))
    return [Prop.box(x, y + 0.003, z, 0.21, 0.004, 0.3, PAPER, rotate=((_yaw(placer.rng), 0.0, 1.0, 0.0),),
                     label="Crumpled printout", tags=CLUTTER_TAGS)]


def _box(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    rng = placer.rng
    w, d = rng.uniform(0.35, 0.55), rng.uniform(0.3, 0.45)
    h = rng.choice([0.12, 0.28, 0.36])  # flattened, or still a box
    half = max(w, d) * 0.75
    pos = placer.sample(zone, half, half, h, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - half, x + half, z - half, z + half, y, y + h))
    return [Prop.box(x, y + h / 2.0, z, w, h, d, CARDBOARD, rotate=((_yaw(rng), 0.0, 1.0, 0.0),),
                     label="Discarded cardboard box", tags=CLUTTER_TAGS)]


def _cable(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    length = placer.rng.uniform(0.8, 1.6)
    half = length * 0.5
    pos = placer.sample(zone, half, half, 0.03, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - half, x + half, z - half, z + half, y, y + 0.03))
    return [Prop.box(x, y + 0.012, z, 0.03, 0.024, length, CABLE, rotate=((_yaw(placer.rng), 0.0, 1.0, 0.0),),
                     label="Loose cable", tags=CLUTTER_TAGS)]


def _puddle(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    r = placer.rng.uniform(0.3, 0.45)
    pos = placer.sample(zone, r, r, 0.004, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - r, x + r, z - r, z + r, y, y + 0.004))
    rz = r * placer.rng.uniform(0.55, 0.9)
    return [Prop.disc(x, y + 0.003, z, r, rz, 0.4, PUDDLE, slices=18, rotate=((_yaw(placer.rng), 0.0, 1.0, 0.0),), label="Leak puddle", tags=CLUTTER_TAGS)]


def _chip(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    pos = placer.sample(zone, 0.08, 0.08, 0.02, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - 0.08, x + 0.08, z - 0.08, z + 0.08, y, y + 0.02))
    return [Prop.box(x, y + 0.008, z, 0.12, 0.015, 0.08, CHIP, rotate=((_yaw(placer.rng), 0.0, 1.0, 0.0),),
                     label="Discarded datachip", tags=CLUTTER_TAGS)]


def _bottle(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    pos = placer.sample(zone, 0.14, 0.14, 0.1, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - 0.14, x + 0.14, z - 0.14, z + 0.14, y, y + 0.1))
    return [Prop.cylinder(x, y + 0.04, z, 0.04, 0.025, 0.26, BOTTLE, slices=8,
                          rotate=((_yaw(placer.rng), 0.0, 1.0, 0.0), (90.0, 1.0, 0.0, 0.0)),
                          label="Empty bottle on its side", tags=CLUTTER_TAGS)]


def _rag(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    pos = placer.sample(zone, 0.25, 0.25, 0.05, y)
    if pos is None:
        return []
    x, z = pos
    placer.commit(AABB(x - 0.25, x + 0.25, z - 0.25, z + 0.25, y, y + 0.05))
    return [Prop.box(x, y + 0.02, z, 0.38, 0.04, 0.3, RAG, rotate=((_yaw(placer.rng), 0.0, 1.0, 0.0),),
                     label="Oily rag", tags=CLUTTER_TAGS)]


def _broken_monitor(placer: Placer, zone: AABB, y: float) -> list[Prop]:
    """A monitor that came off a desk: face up on the floor, propped on its own bezel,
    screen dead and cracked (a thin emissive line)."""
    rng = placer.rng
    pos = placer.sample(zone, 0.34, 0.34, 0.2, y)
    if pos is None:
        return []
    x, z = pos
    yaw = _yaw(rng)
    placer.commit(AABB(x - 0.34, x + 0.34, z - 0.34, z + 0.34, y, y + 0.2))
    tilt = rng.uniform(8.0, 18.0)
    return [
        Prop.box(x, y + 0.08, z, 0.56, 0.09, 0.4, CASING, rotate=((yaw, 0.0, 1.0, 0.0), (tilt, 1.0, 0.0, 0.0)), label="Broken monitor on the floor", tags=CLUTTER_TAGS),
        Prop.box(x, y + 0.135, z, 0.5, 0.012, 0.34, DEAD_SCREEN, rotate=((yaw, 0.0, 1.0, 0.0), (tilt, 1.0, 0.0, 0.0)), label="Broken monitor dead screen", tags=CLUTTER_TAGS),
        Prop.box(x, y + 0.15, z, 0.3, 0.006, 0.012, Material(None, (0.9, 0.9, 1.0), emission=(0.4, 0.5, 0.7)), rotate=((yaw + 20.0, 0.0, 1.0, 0.0), (tilt, 1.0, 0.0, 0.0)), label="Crack across the dead screen", tags=CLUTTER_TAGS),
    ]


_TRASH = {
    "can": _can,
    "paper": _paper,
    "box": _box,
    "cable": _cable,
    "puddle": _puddle,
    "chip": _chip,
    "bottle": _bottle,
    "rag": _rag,
    "monitor": _broken_monitor,
}
TRASH_KINDS: tuple[str, ...] = tuple(_TRASH)


def all_clutter_boxes(props: Iterable[Prop]) -> list[AABB]:
    return [p.footprint() for p in props if "clutter" in p.tags]
