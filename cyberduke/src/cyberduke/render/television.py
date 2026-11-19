"""The screens that change while you look at them: every home's television and the view
out of its windows.

They are props like any other (``levels.flats``, ``levels.apartment``), tagged ``screen``
and either ``tv:<home>`` (the set in that home - ``ai.alley.TvSet``: on or off, which
channel) or ``view:<city|alley>`` (what a window looks out on). They are left out of the
sectors' cached geometry (``render.sectors.is_static``) and drawn every frame here, with
the texture of the moment: the channel's frame (``assets.homes.TV_CHANNELS``: snow that
crawls, the news, cartoons...), a flash of snow as the set retunes after a click of the
remote, dark glass when it is off; the street by night or by day through the window.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import replace

from cyberduke.assets.homes import TV_CHANNELS, tv_frame
from cyberduke.render.context import DrawContext
from cyberduke.render.props import draw_prop
from cyberduke.render.sectors import SCREEN_TAG
from cyberduke.render.textures import TextureMap
from cyberduke.world import Prop, SectorData

RETUNE = 0.35  # seconds of snow after the channel changes (or the set comes on)
UNLIT = (0.3, 0.3, 0.32)  # an OPEN sign switched off: dark glass tubes
TV_OFF = "tv_off"

Vec3 = tuple[float, float, float]


def is_screen(prop: Prop) -> bool:
    return SCREEN_TAG in prop.tags


def screen_of(prop: Prop) -> tuple[str, str]:
    """("tv", the home's area id), ("view", "city" / "alley") or ("open", a business - an
    OPEN sign, lit while it is open); ("", "") for none of them."""
    for tag in prop.tags:
        kind, sep, name = tag.partition(":")
        if sep and kind in ("tv", "view", "open"):
            return kind, name
    return "", ""


def tv_picture(tv, now: float) -> str:
    """The texture a television shows at ``now`` (the street's clock, so a paused game
    freezes the picture): dark glass when it is off (or nobody's), a moment of snow after
    every change of channel, and otherwise its channel's loop of frames."""
    if tv is None or not tv.on:
        return TV_OFF
    channel = tv.showing
    if now - tv.changed < RETUNE or channel not in TV_CHANNELS:
        channel = "static"
    _, fps = TV_CHANNELS[channel]
    return tv_frame(channel, int(math.floor(now * fps)))


def view_picture(kind: str, daylight: float) -> str:
    """The view out of a window (``kind``: "city" or "alley"), by day or by night."""
    return f"view_{kind}_{'day' if daylight >= 0.5 else 'night'}"


_SCREENS: dict[int, tuple[SectorData, list[Prop]]] = {}


def screens_in(sector: SectorData) -> list[Prop]:
    """The sector's screen props (found once per sector)."""
    found = _SCREENS.get(id(sector))
    if found is None or found[0] is not sector:
        found = (sector, [p for p in sector.props if is_screen(p)])
        _SCREENS[id(sector)] = found
    return found[1]


def draw_screens(
    ctx: DrawContext,
    sectors: Iterable[SectorData],
    textures: TextureMap,
    tvs: Mapping[str, object],
    now: float,
    daylight: float = 0.0,
    eye: Vec3 | None = None,
    draw_distance: float = math.inf,
    opens: Mapping[str, bool] | None = None,
) -> None:
    """Draws every television and window view in ``sectors`` (within ``draw_distance`` of
    ``eye``) as it looks this moment. ``tvs``: the sets by home (``AlleyLife.tvs``);
    ``opens``: which businesses are open (``AlleyLife.open``: their OPEN signs lit)."""
    limit = draw_distance * draw_distance
    for sector in sectors:
        for prop in screens_in(sector):
            if eye is not None and (prop.x - eye[0]) ** 2 + (prop.z - eye[2]) ** 2 > limit:
                continue
            kind, name = screen_of(prop)
            if kind == "tv":
                picture = tv_picture(tvs.get(name), now)
                glow = None if picture == TV_OFF else prop.material.emission
                material = replace(prop.material, texture=picture, emission=glow)
            elif kind == "view":
                material = replace(prop.material, texture=view_picture(name, daylight))
            elif kind == "open" and not (opens or {}).get(name, False):
                material = replace(prop.material, color=UNLIT, emission=None)  # (switched off)
            else:
                material = prop.material
            draw_prop(ctx, prop, textures, material=material)
    ctx.emission(None)
    ctx.color(1.0, 1.0, 1.0)


__all__ = [
    "RETUNE",
    "SCREEN_TAG",
    "draw_screens",
    "is_screen",
    "screen_of",
    "screens_in",
    "tv_picture",
    "view_picture",
]
