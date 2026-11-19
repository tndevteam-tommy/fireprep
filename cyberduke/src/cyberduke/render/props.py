"""Draws ``Prop`` records (the level's static geometry) through a DrawContext."""

from __future__ import annotations

from collections.abc import Iterable

from cyberduke.render.context import DrawContext
from cyberduke.render.glow import disc_triangles
from cyberduke.render.textures import TextureMap
from cyberduke.world.props import Material, Prop


def apply_material(ctx: DrawContext, m: Material) -> None:
    if m.alpha != 1.0:
        ctx.alpha(m.alpha)
    ctx.color(*m.color)
    ctx.emission(m.emission)
    if m.specular:
        ctx.specular(m.specular, m.shininess)


def restore_material(ctx: DrawContext, m: Material) -> None:
    if m.specular:
        ctx.specular(None)
    if m.alpha != 1.0:
        ctx.alpha(1.0)


def draw_prop(
    ctx: DrawContext, prop: Prop, textures: TextureMap, material: Material | None = None
) -> None:
    """Draws one prop. ``material`` overrides the prop's own (used for state-driven glow)."""
    m = material or prop.material
    if prop.kind == "disc" and prop.clip is not None:  # a cut pool: world-space triangles
        ctx.glow_mesh(("prop", prop, m), (*m.color, m.alpha), lambda: disc_triangles(prop))
        return
    tex = textures.get(m.texture) if m.texture else None
    apply_material(ctx, m)
    with ctx.transformed((prop.x, prop.y, prop.z), rotate=prop.rotate):
        if prop.kind == "box":
            ctx.scale(*prop.size)
            ctx.cube(tex)
        elif prop.kind == "disc":
            rx, rz, inner = prop.size
            ctx.scale(rx, 1.0, rz)
            ctx.soft_disc((*m.color, m.alpha), slices=prop.slices, inner=inner)
        else:
            r_bot, r_top, h = prop.size
            ctx.cylinder(r_bot, r_top, h, slices=prop.slices, tex_id=tex)
    restore_material(ctx, m)


def draw_props(ctx: DrawContext, props: Iterable[Prop], textures: TextureMap) -> None:
    for prop in props:
        draw_prop(ctx, prop, textures)
    ctx.emission(None)
    ctx.color(1.0, 1.0, 1.0)
