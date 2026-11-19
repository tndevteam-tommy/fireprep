"""Composite models: NPC bodies with heads, eyes and torches; the player's held torch."""
from __future__ import annotations

import math
import zlib
from typing import Callable, Optional, Tuple

from OpenGL.GL import glPopMatrix, glPushMatrix, glRotatef, glTranslatef

from ..core.math3d import BoundingBox3D, Vector3D
from .primitives import draw_box, draw_sphere, draw_textured_box

Material = Callable[..., None]   # material(color=(r,g,b), emission=(r,g,b), textured=bool, fog=float)

from .gear import BODY_HALF_X, EYE_Y, HEAD_HALF, draw_figure

EYE_RADIUS, PUPIL_RADIUS = 0.065, 0.035
EYE_SPACING = 0.095


def npc_texture_key(npc) -> str:
    """The body texture: the watch's steel, or undyed cloth that is tinted by the outfit."""
    return "guard" if npc.clothes.armor else "cloth"


def draw_outfit(npc, textures, material: Material) -> None:
    """The clothes on an upright model (local space: feet at the origin, front along -z): the
    NPC's figure dressed in its equipment (``gear.draw_figure``), exactly as the player is."""
    draw_figure(npc.figure, npc.equipment, textures, material)


def draw_face(material, asleep: bool = False) -> None:
    """Eyes on the front face of the head (local -z), pupils a touch further forward."""
    material(color=(1.0, 1.0, 1.0), textured=False)
    for sx in (-EYE_SPACING, EYE_SPACING):
        draw_sphere((sx, EYE_Y, -HEAD_HALF - 0.01), EYE_RADIUS)
    material(color=(0.05, 0.05, 0.07), textured=False)
    for sx in (-EYE_SPACING, EYE_SPACING):
        draw_sphere((sx, EYE_Y, -HEAD_HALF - 0.01 - EYE_RADIUS * 0.75), PUPIL_RADIUS)


def draw_npc(npc, textures, material: Material, body_box: BoundingBox3D, time: float,
             outfit: Optional[Callable[[Material], None]] = None, head: str = "") -> None:
    """Upright NPCs get a dressed body, a head with eyes looking along ``facing_yaw`` and (at
    night) a torch; sleeping NPCs are drawn lying flat on their bed. ``outfit`` may supply a
    pre-recorded drawing of the clothes (``draw_outfit`` otherwise)."""
    if npc.is_asleep:
        draw_sleeping(npc, body_box, textures, material, head)
        return
    p = npc.position
    glPushMatrix()
    glTranslatef(p.x, p.y, p.z)
    glRotatef(-npc.facing_yaw, 0.0, 1.0, 0.0)        # local -z is the NPC's front
    if outfit is not None:
        outfit(material)
    else:
        draw_outfit(npc, textures, material)
    draw_face(material)
    if npc.has_torch:
        draw_torch((BODY_HALF_X + 0.12, 0.95, -0.05), textures, material, time, phase=zlib.crc32(npc.name.encode("utf-8")) % 7)
    glPopMatrix()


def draw_sleeping(npc, body_box: BoundingBox3D, textures, material: Material, head_end: str = "") -> None:
    """Lying pose: the body box from the world (in the tunic's colour) plus a head at the end the
    bed's headboard is (``head_end`` "-x" / "+x" / "-z" / "+z"; default the box's low end)."""
    o = npc.clothes
    b = body_box
    along_x = (b.max_point.x - b.min_point.x) >= (b.max_point.z - b.min_point.z)
    head_end = head_end or ("-x" if along_x else "-z")
    head = 0.42
    mn, mx = b.min_point, b.max_point
    if head_end == "-x":
        body = BoundingBox3D(Vector3D(mn.x + head, mn.y, mn.z), mx)
        head_box = BoundingBox3D(Vector3D(mn.x, mn.y, mn.z + 0.05), Vector3D(mn.x + head, mx.y, mx.z - 0.05))
    elif head_end == "+x":
        body = BoundingBox3D(mn, Vector3D(mx.x - head, mx.y, mx.z))
        head_box = BoundingBox3D(Vector3D(mx.x - head, mn.y, mn.z + 0.05), Vector3D(mx.x, mx.y, mx.z - 0.05))
    elif head_end == "+z":
        body = BoundingBox3D(mn, Vector3D(mx.x, mx.y, mx.z - head))
        head_box = BoundingBox3D(Vector3D(mn.x + 0.05, mn.y, mx.z - head), Vector3D(mx.x - 0.05, mx.y, mx.z))
    else:
        body = BoundingBox3D(Vector3D(mn.x, mn.y, mn.z + head), mx)
        head_box = BoundingBox3D(Vector3D(mn.x + 0.05, mn.y, mn.z), Vector3D(mx.x - 0.05, mx.y, mn.z + head))
    material(color=(1.0, 1.0, 1.0) if o.armor else o.tunic, textured=True)
    draw_box(body, textures[npc_texture_key(npc)], tile=None)
    material(color=(1.0, 1.0, 1.0), textured=True)
    draw_box(head_box, textures["skin"], tile=None)


def flame_scale(time: float, phase: float = 0.0) -> float:
    return 1.0 + 0.12 * math.sin(time * 17.0 + phase) + 0.06 * math.sin(time * 29.0 + phase * 1.3)


def draw_torch(origin: Tuple[float, float, float], textures, material: Material, time: float, phase: float = 0.0,
               stick_height: float = 0.45) -> None:
    """A wooden stick with a glowing flame on top (local coordinates)."""
    x, y, z = origin
    material(textured=True)
    draw_textured_box(Vector3D(x - 0.04, y, z - 0.04), Vector3D(x + 0.04, y + stick_height, z + 0.04), textures["torch"], tile=None)
    k = flame_scale(time, phase)
    fh, fw = 0.22 * k, 0.11
    # the flame is emissive and punches through the fog so distant torches show as points of light
    material(emission=(1.0, 0.75, 0.35), textured=True, fog=0.25)
    draw_textured_box(Vector3D(x - fw, y + stick_height, z - fw), Vector3D(x + fw, y + stick_height + fh, z + fw), textures["flame"], tile=None)
    material(textured=True)


def draw_fixture_body(fixture, textures, material: Material) -> None:
    """The unchanging part of an indoor light: a wall lantern's iron frame on its bracket, or
    a candle's stub of wax. (A "fire" fixture is a light only: the forge / oven it burns in
    draws its own coals and flames.)"""
    x, y, z = fixture.position
    if fixture.kind in ("fire", "glow"):
        return
    material(textured=True)
    if fixture.kind in ("torch", "sconce"):
        # a wall sconce: an iron bracket out of the wall and a pitch-soaked stick in its ring
        glPushMatrix()
        glTranslatef(x, y, z)
        glRotatef(-fixture.facing, 0.0, 1.0, 0.0)      # local -z away from the wall
        iron = textures["iron"]
        draw_textured_box(Vector3D(-0.03, -0.45, 0.02), Vector3D(0.03, -0.39, 0.3), iron, tile=None)    # arm into the wall
        draw_textured_box(Vector3D(-0.06, -0.42, -0.04), Vector3D(0.06, -0.38, 0.06), iron, tile=None)  # the ring
        draw_textured_box(Vector3D(-0.035, -0.62, -0.02), Vector3D(0.035, 0.0, 0.04), textures["torch"], tile=None)   # the stick
        glPopMatrix()
        return
    if fixture.kind == "candle":
        draw_textured_box(Vector3D(x - 0.035, y, z - 0.035), Vector3D(x + 0.035, y + 0.16, z + 0.035), textures["wax"], tile=None)
        return
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(-fixture.facing, 0.0, 1.0, 0.0)          # local -z points into the room; the wall is behind (+z)
    iron = textures["iron"]
    draw_textured_box(Vector3D(-0.03, -0.02, 0.0), Vector3D(0.03, 0.02, 0.35), iron, tile=None)       # bracket to the wall
    draw_textured_box(Vector3D(-0.13, -0.22, -0.13), Vector3D(0.13, -0.18, 0.13), iron, tile=None)    # base plate
    draw_textured_box(Vector3D(-0.13, 0.16, -0.13), Vector3D(0.13, 0.2, 0.13), iron, tile=None)       # cap
    for sx in (-0.12, 0.1):
        for sz in (-0.12, 0.1):
            draw_textured_box(Vector3D(sx, -0.18, sz), Vector3D(sx + 0.02, 0.16, sz + 0.02), iron, tile=None)   # corner posts
    glPopMatrix()


def draw_fixture_flame(fixture, textures, material: Material, time: float) -> None:
    """The flickering flame of a lantern or candle (drawn live every frame)."""
    x, y, z = fixture.position
    if fixture.kind in ("fire", "glow"):
        return
    phase = (zlib.crc32(fixture.room.encode("utf-8")) % 11) * 0.6      # per-room phase (stable across runs)
    k = flame_scale(time, phase + x * 0.37)
    if fixture.kind in ("torch", "sconce"):
        material(emission=(1.0, 0.7, 0.32), textured=True, fog=0.25)
        fh, fw = 0.3 * k, 0.08
        draw_textured_box(Vector3D(x - fw, y, z - fw), Vector3D(x + fw, y + fh, z + fw), textures["flame"], tile=None)
        material(textured=True)
        return
    if fixture.kind == "candle":
        material(emission=(1.0, 0.75, 0.35), textured=True, fog=0.25)
        fh, fw = 0.1 * k, 0.05
        draw_textured_box(Vector3D(x - fw, y + 0.16, z - fw), Vector3D(x + fw, y + 0.16 + fh, z + fw), textures["flame"], tile=None)
    else:
        material(emission=(1.0, 0.78, 0.4), textured=True, fog=0.25)
        fh, fw = 0.2 * k, 0.06
        draw_textured_box(Vector3D(x - fw, y - 0.16, z - fw), Vector3D(x + fw, y - 0.16 + fh, z + fw), textures["flame"], tile=None)
    material(textured=True)


def draw_held_torch(textures, material: Material, time: float) -> None:
    """The player's torch in the lower right of the view (call with an identity modelview)."""
    glPushMatrix()
    glTranslatef(0.42, -0.55, -0.9)
    glRotatef(-18.0, 0.0, 0.0, 1.0)
    glRotatef(12.0, 1.0, 0.0, 0.0)
    draw_torch((0.0, 0.0, 0.0), textures, material, time, phase=0.0, stick_height=0.55)
    glPopMatrix()
