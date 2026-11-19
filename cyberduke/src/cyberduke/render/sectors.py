"""Draws a ``Room``: every sector's props and light fixtures (cached as static geometry by
the backend), then the doors (dynamic, since the leaf moves)."""

from __future__ import annotations

import math
from dataclasses import replace

from cyberduke.core.geometry import AABB
from cyberduke.render.context import DrawContext
from cyberduke.render.glow import pool_triangles
from cyberduke.render.lighting import draw_light_fixture
from cyberduke.render.props import draw_prop, draw_props
from cyberduke.render.textures import TextureMap
from cyberduke.world import Material, Prop, Room, SectorData, SlidingDoor

Vec3 = tuple[float, float, float]

PULSE_TAG = "pulse"
"""Props with this tag are drawn every frame with their emission breathing over time (the
greenhouse's bioluminescent caps); their phase tag (``phase0``..``phase6``) staggers them."""


def is_pulsing(prop: Prop) -> bool:
    return PULSE_TAG in prop.tags


FOG_TAG = "fog"
"""Translucent props (the haze over the greenhouse): never cached, drawn last with depth
writes off, sorted back to front, drifting slowly with time."""
GLOW_TAG = "glow"
"""Light pools on surfaces: drawn in the late pass too, but baked per sector and colour."""


def is_translucent(prop: Prop) -> bool:
    return FOG_TAG in prop.tags or prop.kind == "disc" or prop.material.alpha < 1.0


def is_pool(prop: Prop) -> bool:
    """A pool of light lying on a surface (or splashed up a wall): baked, never sorted."""
    return prop.kind == "disc" and GLOW_TAG in prop.tags and FOG_TAG not in prop.tags


_POOLS: dict[int, tuple[SectorData, dict[tuple[float, ...], list[Prop]]]] = {}


def light_pools(sector: SectorData) -> dict[tuple[float, ...], list[Prop]]:
    """The sector's light pools grouped by colour (remembered per sector)."""
    found = _POOLS.get(id(sector))
    if found is None or found[0] is not sector:
        groups: dict[tuple[float, ...], list[Prop]] = {}
        for prop in sector.props:
            if is_pool(prop):
                groups.setdefault(tuple(prop.material.color), []).append(prop)
        found = _POOLS[id(sector)] = (sector, groups)
    return found[1]


SCREEN_TAG = "screen"
"""Televisions and window views: drawn live every frame (``render.television``)."""


def is_static(prop: Prop) -> bool:
    return not is_pulsing(prop) and not is_translucent(prop) and SCREEN_TAG not in prop.tags


def pulse_material(prop: Prop, time: float) -> Material:
    phase = 0
    for tag in prop.tags:
        if tag.startswith("phase") and tag[5:].isdigit():
            phase = int(tag[5:])
    base = prop.material
    if base.emission is None:
        return base
    glow = 0.55 + 0.45 * math.sin(time * 1.6 + phase * 0.9 + prop.x * 0.3 + prop.z * 0.17)
    return replace(base, emission=tuple(c * glow for c in base.emission))


def draw_sector(ctx: DrawContext, sector: SectorData, textures: TextureMap) -> None:
    """The static part of a sector: everything except the pulsing props, the translucent
    ones and the screens."""
    draw_props(ctx, (p for p in sector.props if is_static(p)), textures)
    for light in sector.lights:
        draw_light_fixture(ctx, light, textures)
    ctx.color(1.0, 1.0, 1.0)


def draw_pulsing_props(
    ctx: DrawContext,
    sector: SectorData,
    textures: TextureMap,
    time: float,
    eye: Vec3 | None = None,
    draw_distance: float = math.inf,
) -> None:
    limit = draw_distance * draw_distance
    for prop in sector.props:
        if not is_pulsing(prop):
            continue
        if eye is not None and (prop.x - eye[0]) ** 2 + (prop.z - eye[2]) ** 2 > limit:
            continue
        draw_prop(ctx, prop, textures, material=pulse_material(prop, time))
    ctx.emission(None)
    ctx.color(1.0, 1.0, 1.0)


def slid_leaf_piece(prop: Prop, open_amount: float, lo: float, hi: float) -> Prop | None:
    """The part of a door-leaf prop that is still inside the doorway ``[lo, hi]`` (along X)
    once it has slid ``open_amount`` metres toward -X, or ``None`` when it has gone into
    the wall entirely.

    Door leaves are thicker than the walls they slide into (a 0.22 m blast door in a 0.1 m
    wall), so drawing the whole slid leaf painted it onto both faces of the wall beside the
    doorway - and a maze gate rose out of the hedge it slid into. Only axis-aligned boxes
    can be trimmed; anything else is returned slid but untrimmed."""
    if prop.kind != "box" or prop.rotate:
        return replace(prop, x=prop.x - open_amount)
    half = prop.size[0] / 2.0
    left = max(prop.x - half - open_amount, lo)
    right = min(prop.x + half - open_amount, hi)
    if right - left <= 1e-4:
        return None
    return replace(prop, x=(left + right) / 2.0, size=(right - left, prop.size[1], prop.size[2]))


def door_leaf_boxes(door: SlidingDoor) -> list[AABB]:
    """Bounds of whatever of the door's leaf is still in the doorway (for clipping beams)."""
    if door.data is None or door.data.hinge is not None:
        return [door.obstacle] if not door.passable else []
    lo, hi = door.obstacle.min_x, door.obstacle.max_x
    pieces = (slid_leaf_piece(p, door.open_amount, lo, hi) for p in door.data.leaf)
    return [p.footprint() for p in pieces if p is not None]


def draw_door(ctx: DrawContext, door: SlidingDoor, textures: TextureMap) -> None:
    """The leaf (and its status strips) slide along -X by the door's open amount, into the
    wall: only what is still inside the doorway (the door's obstacle box) is drawn."""
    data = door.data
    if data is None:
        return
    if data.hinge is not None:
        # a hinged door swings about its hinge instead (``DoorData.hinge``)
        hx, hz = data.hinge
        angle = data.swing * max(0.0, min(1.0, door.open_amount / max(door.travel, 1e-6)))
        with ctx.pushed():
            ctx.translate(hx, 0.0, hz)
            if angle:
                ctx.rotate(angle, 0.0, 1.0, 0.0)
            ctx.translate(-hx, 0.0, -hz)
            for prop in data.leaf:
                draw_prop(ctx, prop, textures)
        ctx.emission(None)
        ctx.color(1.0, 1.0, 1.0)
        return
    lo, hi = door.obstacle.min_x, door.obstacle.max_x
    for prop in data.leaf:
        piece = slid_leaf_piece(prop, door.open_amount, lo, hi)
        if piece is not None:
            draw_prop(ctx, piece, textures)
    glow = data.indicator_open if door.passable else data.indicator_closed
    for prop in data.indicators:
        piece = slid_leaf_piece(prop, door.open_amount, lo, hi)
        if piece is not None:
            draw_prop(ctx, piece, textures, material=glow)
    # a lamp shows the lock: green as soon as the door is unlocked, open or shut
    lamp = data.indicator_open if door.unlocked else data.indicator_closed
    for prop in data.lamps:
        draw_prop(ctx, prop, textures, material=lamp)
    ctx.emission(None)
    ctx.color(1.0, 1.0, 1.0)


FOG_DRIFT = 0.9  # metres a fog layer wanders from where the level put it


def fog_drift(k: int, time: float) -> Vec3:
    """How far fog prop number ``k`` (its place in the level's props) has drifted at
    ``time``: slowly, smoothly, each on its own."""
    return (
        math.sin(time * 0.05 + k * 1.7) * FOG_DRIFT,
        0.0,
        math.cos(time * 0.04 + k * 0.9) * FOG_DRIFT,
    )


def fog_position(prop: Prop, k: int, time: float) -> Vec3:
    """Where a translucent prop is drawn (a fog layer where it has drifted to)."""
    if FOG_TAG not in prop.tags:
        return (prop.x, prop.y, prop.z)
    dx, dy, dz = fog_drift(k, time)
    return (prop.x + dx, prop.y + dy, prop.z + dz)


def _distance2(a: Vec3, b: Vec3) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def draw_translucent_props(
    ctx: DrawContext,
    sectors: list[SectorData],
    textures: TextureMap,
    time: float,
    eye: Vec3 | None = None,
) -> None:
    """The late pass, without writing depth so nothing translucent punches holes in
    anything else: first the light pools (discs tagged ``glow`` - under the mushrooms, the
    lanterns, the neon - each sector's baked into one mesh per colour, ``render.glow``),
    then fog layers and anything else see-through, back to front so the blending is right."""
    ctx.depth_mask(False)
    for sector in sectors:
        for color, pools in light_pools(sector).items():
            ctx.glow_mesh(
                ("pools", sector.name, color),
                (*color, 1.0),
                lambda pools=pools: pool_triangles(pools),  # each pool's alpha baked in
            )
    ctx.depth_mask(True)
    # Each one keeps its own number (its place in its sector), whatever order it is drawn
    # in and whichever other sectors are in view: the drift of a fog layer is a function
    # of that and the time - never of where it comes in the back-to-front order, which
    # changes as the viewer walks about (when it did, the layers swapped drifts - jumping
    # up to 1.8 m - and the fog blinked).
    props = [
        (k, p)
        for s in sectors
        for k, p in enumerate(s.props)
        if is_translucent(p) and not is_pool(p)
    ]
    if not props:
        ctx.alpha(1.0)
        ctx.color(1.0, 1.0, 1.0)
        return
    if eye is not None:
        props.sort(key=lambda kp: -_distance2(fog_position(kp[1], kp[0], time), eye))
    ctx.depth_mask(False)
    for k, prop in props:
        if FOG_TAG in prop.tags:  # haze drifts; light pools stay put
            with ctx.transformed(fog_drift(k, time)):
                draw_prop(ctx, prop, textures)
        else:
            draw_prop(ctx, prop, textures)
    ctx.depth_mask(True)
    ctx.alpha(1.0)
    ctx.emission(None)
    ctx.color(1.0, 1.0, 1.0)


def visible_sectors(room: Room, eye: Vec3, view_distance: float) -> list[SectorData]:
    """Sectors with any part inside the view distance.

    Culling can only ever drop geometry that is beyond the far plane anyway: a sector's
    props include the walls it shares with its neighbours (the vault owns the partition
    wall around the airlock door), so hiding a sector because a door between here and
    there is shut would punch holes in the level and let the player see through walls.
    """
    out = []
    for sector in room.sectors:
        min_x, max_x, min_z, max_z = sector.bounds
        dx = max(min_x - eye[0], 0.0, eye[0] - max_x)
        dz = max(min_z - eye[2], 0.0, eye[2] - max_z)
        if math.hypot(dx, dz) <= view_distance:
            out.append(sector)
    return out


def draw_room(
    ctx: DrawContext,
    room: Room,
    textures: TextureMap,
    eye: Vec3 | None = None,
    view_distance: float = math.inf,
    effect_distance: float = math.inf,
) -> None:
    """Draws the sectors within ``view_distance`` of ``eye`` (all of them when no eye is
    given), their pulsing props within ``effect_distance``, and every door."""
    sectors = visible_sectors(room, eye, view_distance) if eye is not None else list(room.sectors)
    for sector in sectors:
        ctx.cached(("sector", sector.name), lambda s=sector: draw_sector(ctx, s, textures))
        draw_pulsing_props(ctx, sector, textures, room.anim_time, eye, effect_distance)
    for door in room.doors.values():
        draw_door(ctx, door, textures)
