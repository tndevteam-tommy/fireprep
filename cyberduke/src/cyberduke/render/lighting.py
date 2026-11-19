"""Hardware lights and their visible fixtures, both driven by ``world.lights.Light``."""

from __future__ import annotations

from collections.abc import Sequence

from OpenGL.GL import *

from cyberduke.render.context import DrawContext
from cyberduke.render.textures import TextureMap
from cyberduke.world.lights import Light

MAX_HARDWARE_LIGHTS = 8
AMBIENT_BASE = (0.25, 0.31, 0.44, 1.0)  # lifts the scene out of near-blackness (the HIGH preset)


def setup_lighting(lights: Sequence[Light], ambient: Sequence[float] = AMBIENT_BASE) -> None:
    """Enables fixed-function lighting and uploads up to 8 positional lights.

    ``ambient`` is the scene's global ambient (the quality preset's level). Light
    *positions* set here are only provisional: fixed-function GL stores GL_POSITION in eye
    space using the modelview matrix current at the time of the call, so the renderer must
    call ``position_lights`` again after loading each frame's view matrix - otherwise the
    lights stay glued to the camera instead of hanging where the level put them.
    """
    if len(lights) > MAX_HARDWARE_LIGHTS:
        raise ValueError(
            f"fixed-function GL supports {MAX_HARDWARE_LIGHTS} lights, got {len(lights)}"
        )

    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_NORMALIZE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (*tuple(ambient)[:3], 1.0))

    for i in range(len(lights), MAX_HARDWARE_LIGHTS):
        glDisable(GL_LIGHT0 + i)  # a smaller light budget must switch the rest off
    for i, light in enumerate(lights):
        gl_light = GL_LIGHT0 + i
        glEnable(gl_light)
        glLightfv(gl_light, GL_POSITION, (*light.position, 1.0))
        glLightfv(gl_light, GL_AMBIENT, (*light.ambient, 1.0))
        glLightfv(gl_light, GL_DIFFUSE, (*light.diffuse, 1.0))
        glLightfv(gl_light, GL_SPECULAR, (*light.specular, 1.0))
        glLightf(gl_light, GL_CONSTANT_ATTENUATION, 1.0)
        glLightf(gl_light, GL_LINEAR_ATTENUATION, light.linear_attenuation)
        glLightf(gl_light, GL_QUADRATIC_ATTENUATION, light.quadratic_attenuation)


def position_lights(lights: Sequence[Light]) -> None:
    """Re-specifies every active light's world position under the current modelview matrix
    (call right after the camera's view transform has been loaded for the frame)."""
    for i, light in enumerate(lights[:MAX_HARDWARE_LIGHTS]):
        glLightfv(GL_LIGHT0 + i, GL_POSITION, (*light.position, 1.0))


# ----------------------------------------------------------------------------- fixtures
def draw_light_fixture(ctx: DrawContext, light: Light, textures: TextureMap) -> None:
    drawer = _FIXTURES.get(light.fixture)
    if drawer is not None:
        drawer(ctx, light, textures)


def _hanging_cage(ctx: DrawContext, light: Light, textures: TextureMap) -> None:
    """Industrial neon tube in a caged frame hung from the ceiling on two cables."""
    x, y, z = light.x, light.y + light.fixture_offset_y, light.z
    tube_tex = textures.get(light.fixture_texture)
    frame_tex = textures.get("metal")
    ceiling_y = light.ceiling_y

    ctx.color(0.15, 0.15, 0.18)
    for offset_z in (-0.85, 0.85):
        ctx.box(x, (y + ceiling_y) * 0.5, z + offset_z, 0.02, ceiling_y - y, 0.02)

    ctx.color(0.22, 0.24, 0.28)
    ctx.box(x, y + 0.08, z, 0.28, 0.10, 1.90, frame_tex)

    with ctx.emissive(light.fixture_emission, light.fixture_color):
        ctx.box(x, y, z, 0.16, 0.08, 1.76, tube_tex)

    ctx.color(0.12, 0.13, 0.16)
    ctx.box(x, y - 0.05, z, 0.24, 0.02, 1.84)


def _panel(
    ctx: DrawContext, light: Light, textures: TextureMap, size: tuple[float, float, float]
) -> None:
    with ctx.emissive(light.fixture_emission, light.fixture_color):
        ctx.box(
            light.x,
            light.y + light.fixture_offset_y,
            light.z,
            *size,
            textures.get(light.fixture_texture),
        )


def _caution_lamp(ctx: DrawContext, light: Light, textures: TextureMap) -> None:
    """Flush ceiling-mounted amber hazard lamp."""
    _panel(ctx, light, textures, (0.24, 0.10, 1.4))


def _halo(ctx: DrawContext, light: Light, textures: TextureMap) -> None:
    """Large square emissive ceiling panel."""
    _panel(ctx, light, textures, (3.0, 0.10, 3.0))


def _flood_bar(ctx: DrawContext, light: Light, textures: TextureMap) -> None:
    """Narrow emissive floodlight bar."""
    _panel(ctx, light, textures, (0.30, 0.12, 2.0))


_FIXTURES = {
    "hanging_cage": _hanging_cage,
    "caution_lamp": _caution_lamp,
    "halo": _halo,
    "flood_bar": _flood_bar,
}
