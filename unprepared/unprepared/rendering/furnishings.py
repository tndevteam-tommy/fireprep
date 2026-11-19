"""More modelled things: the furnishings of caves, camps and the country, and the readable signs.

Like :mod:`structures`, every ``draw_*`` fills the footprint of the prop (or container, or
readable) it is given, turned by its ``facing`` (0 = its front looks along -z, 90 = +x ...):
the level says where a thing stands and how big it is, the modelling lives here.
"""
from __future__ import annotations

import math
import random
from typing import Callable

from OpenGL.GL import (GL_QUADS, GL_TEXTURE_2D, GL_TRIANGLE_FAN, GL_TRIANGLES, glBegin, glBindTexture, glDisable, glEnable,
                       glEnd, glNormal3f, glPopMatrix, glPushMatrix, glRotatef, glScalef, glTexCoord2f, glTranslatef, glVertex3f)

from ..core.math3d import Vector3D
from .primitives import draw_board, draw_textured_box
from .structures import _decal_quad, _frame, draw_cylinder, draw_log, draw_rock

Material = Callable[..., None]


def _local(prop):
    """Push a matrix to the prop's centre on the ground, turned by its facing; returns (hx, hz, height)
    - half its width and depth along its own axes."""
    cx, cz, hx, hz, base, height = _frame(prop)
    glPushMatrix()
    glTranslatef(cx, base, cz)
    glRotatef(-getattr(prop, "facing", 0.0), 0.0, 1.0, 0.0)
    return hx, hz, height


def _box(x1, y1, z1, x2, y2, z2, tex, tile=None):
    draw_textured_box(Vector3D(x1, y1, z1), Vector3D(x2, y2, z2), tex, tile=tile)


# ----------------------------------------------------------------------------- sleeping rough
def draw_bedroll(prop, textures, material: Material, time: float) -> None:
    """A bedroll on the floor: a blanket over a straw pallet, a rolled-up cloak for a pillow at the
    head end (``facing`` is the way the head lies, as for a bed)."""
    hx, hz, height = _local(prop)
    length, width = max(hx, hz), min(hx, hz)
    # local: the head towards -z
    material(color=(0.75, 0.68, 0.5), textured=True)
    _box(-width, 0.0, -length, width, 0.1, length, textures["hay"], tile=0.8)
    material(color=(0.52, 0.42, 0.34), textured=True)
    _box(-width + 0.05, 0.1, -length + 0.55, width - 0.05, 0.2, length - 0.04, textures["cloth"], tile=0.6)
    material(color=(0.45, 0.4, 0.36), textured=True)
    glPushMatrix()
    glTranslatef(0.0, 0.0, -length + 0.3)
    glRotatef(90.0, 0.0, 1.0, 0.0)
    draw_log(-width + 0.08, width - 0.08, 0.17, 0.0, 0.14, textures["cloth"], textures["cloth"], segments=8)
    glPopMatrix()
    material(textured=True)
    glPopMatrix()


# ----------------------------------------------------------------------------- the dead and their hoards
def draw_skeleton(container, textures, material: Material, time: float) -> None:
    """The bones of somebody who never got out: a skull, ribs, the long bones of arms and legs,
    lying as they fell along the container's longer side, a rotten satchel by the hand."""
    hx, hz, height = _local(container)
    length = max(hx, hz)
    glPushMatrix()
    if hx > hz:
        glRotatef(90.0, 0.0, 1.0, 0.0)             # (lying along local z)
    bone = textures["bone"]
    material(color=(0.92, 0.88, 0.78), textured=True)
    head_z = -length + 0.2
    draw_rock(0.05, 0.12, head_z, 0.13, bone, seed=3, squash=0.85, lumpiness=0.08, slices=8, stacks=6, tile=0.3)   # the skull
    _box(-0.03, 0.02, head_z + 0.14, 0.03, 0.06, head_z + 0.3, bone)                                               # neck
    spine_z1, spine_z2 = head_z + 0.28, head_z + 0.95
    _box(-0.025, 0.02, spine_z1, 0.025, 0.07, spine_z2, bone)                                                      # spine
    for k in range(6):                                                                                             # ribs
        z = spine_z1 + 0.08 + k * 0.09
        w = 0.2 - abs(k - 2) * 0.02
        _box(-w, 0.03, z, w, 0.06, z + 0.035, bone)
    _box(-0.16, 0.02, spine_z2 - 0.02, 0.16, 0.09, spine_z2 + 0.1, bone)                                           # pelvis
    for side in (-1.0, 1.0):
        _box(side * 0.32 - 0.03, 0.02, spine_z1, side * 0.32 + 0.03, 0.06, spine_z1 + 0.5, bone)                   # arms
        _box(side * 0.36 - 0.03, 0.02, spine_z1 + 0.5, side * 0.36 + 0.03, 0.06, spine_z1 + 0.9, bone)
        _box(side * 0.1 - 0.035, 0.02, spine_z2 + 0.1, side * 0.14 + 0.035, 0.07, spine_z2 + 0.62, bone)           # legs
        _box(side * 0.14 - 0.03, 0.02, spine_z2 + 0.62, side * 0.17 + 0.03, 0.06, min(length, spine_z2 + 1.1), bone)
    material(color=(0.45, 0.33, 0.22), textured=True)
    _box(0.42, 0.0, spine_z1 + 0.55, 0.72, 0.2, spine_z1 + 0.85, textures["leather"])                             # the satchel
    material(textured=True)
    glPopMatrix()
    glPopMatrix()


def draw_stash(container, textures, material: Material, time: float) -> None:
    """Loose stones heaped into a hollow in the rock - look closer and something was hidden
    behind them."""
    cx, cz, hx, hz, base, height = _frame(container)
    rng = random.Random(int(abs(cx * 5.3 + cz * 7.1)) % 1000)
    material(color=(0.88, 0.86, 0.82), textured=True)
    tex = textures["cave_rock"]
    for k in range(6):
        r = rng.uniform(0.14, 0.26)
        x = cx + rng.uniform(-hx + r, hx - r)
        z = cz + rng.uniform(-hz + r, hz - r)
        draw_rock(x, base + r * 0.6, z, r, tex, seed=k + 40, squash=0.8, lumpiness=0.2, slices=7, stacks=5, tile=0.6)
    material(textured=True)


# ----------------------------------------------------------------------------- things to read
def draw_notice(readable, textures, material: Material, time: float) -> None:
    """A notice board on two posts under a little roof, papers pinned to its face (local -z)."""
    hx, hz, height = _local(readable)
    wood = textures["wood"]
    material(textured=True)
    for x in (-hx + 0.06, hx - 0.18):
        _box(x, -0.1, -0.06, x + 0.12, height, 0.06, wood, tile=1.0)
    board_y1, board_y2 = height * 0.42, height * 0.86
    glPushMatrix()
    draw_board(Vector3D(-hx + 0.02, board_y1, -0.05), Vector3D(hx - 0.02, board_y2, 0.03), textures["notice"], wood)
    glPopMatrix()
    material(color=(0.8, 0.7, 0.6), textured=True)
    roof = textures["roof_slate"]
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, roof)
    glBegin(GL_QUADS)
    top = height + 0.25
    for side in (-1.0, 1.0):
        glNormal3f(0.0, 0.7, side * 0.7)
        pts = ((-hx - 0.15, height - 0.05, side * 0.35), (hx + 0.15, height - 0.05, side * 0.35), (hx + 0.15, top, 0.0),
               (-hx - 0.15, top, 0.0))
        for (u, v), p in zip(((0, 0), (2, 0), (2, 1), (0, 1)), pts):
            glTexCoord2f(u, v)
            glVertex3f(*p)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    material(textured=True)
    glPopMatrix()


def draw_signpost(readable, textures, material: Material, time: float) -> None:
    """A fingerpost at a fork in the way: a tall post, a cap, and an arm pointing each way it
    names (three arms, turned to catch the eye from the road)."""
    hx, hz, height = _local(readable)
    wood = textures["wood"]
    material(color=(0.85, 0.78, 0.68), textured=True)
    _box(-0.07, -0.1, -0.07, 0.07, height, 0.07, wood, tile=1.0)
    _box(-0.1, height, -0.1, 0.1, height + 0.08, 0.1, wood)
    rng = random.Random(int(abs(readable.center.x * 3 + readable.center.z)) % 97)
    for k, (angle, y) in enumerate(((0.0, height - 0.35), (120.0 + rng.uniform(-20, 20), height - 0.7),
                                    (235.0 + rng.uniform(-20, 20), height - 1.05))):
        glPushMatrix()
        glRotatef(angle, 0.0, 1.0, 0.0)
        draw_board(Vector3D(0.07, y - 0.13, -0.03), Vector3D(0.95, y + 0.13, 0.03), textures["sign_arm"], wood)
        glPopMatrix()
    material(textured=True)
    glPopMatrix()


def draw_plaque(readable, textures, material: Material, time: float) -> None:
    """A board fixed to a wall, lettered (its face looks the way of ``facing``)."""
    hx, hz, height = _local(readable)
    material(textured=True)
    draw_board(Vector3D(-hx, 0.0, -0.04), Vector3D(hx, height, 0.04), textures["plaque"], textures["wood"])
    glPopMatrix()


def draw_book(readable, textures, material: Material, time: float) -> None:
    """An open book (or a journal, a ledger) lying on whatever is under it."""
    hx, hz, height = _local(readable)
    material(color=(0.45, 0.25, 0.18), textured=True)
    _box(-0.24, 0.0, -0.17, 0.24, 0.03, 0.17, textures["leather"])
    material(textured=True)
    for side in (-1.0, 1.0):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, textures["parchment"])
        glBegin(GL_QUADS)
        glNormal3f(-side * 0.12, 1.0, 0.0)
        for (u, v), (x, y, z) in zip(((0, 0), (1, 0), (1, 1), (0, 1)),
                                     ((0.0, 0.035, -0.15), (side * 0.22, 0.05, -0.15), (side * 0.22, 0.05, 0.15), (0.0, 0.035, 0.15))):
            glTexCoord2f(u, v)
            glVertex3f(x, y, z)
        glEnd()
        glDisable(GL_TEXTURE_2D)
    glPopMatrix()


# ----------------------------------------------------------------------------- camps and shores
def draw_dummy(prop, textures, material: Material, time: float) -> None:
    """A practice dummy: a post, a cross-bar for arms and a straw-stuffed sack of a body and head."""
    hx, hz, height = _local(prop)
    wood = textures["wood"]
    material(textured=True)
    _box(-0.06, 0.0, -0.06, 0.06, height * 0.95, 0.06, wood, tile=1.0)
    _box(-0.5, height * 0.68, -0.05, 0.5, height * 0.74, 0.05, wood, tile=1.0)
    material(color=(0.9, 0.8, 0.6), textured=True)
    draw_rock(0.0, height * 0.6, 0.0, 0.26, textures["hay"], seed=11, squash=1.35, lumpiness=0.1, slices=8, stacks=6, tile=0.5)
    draw_rock(0.0, height * 0.9, 0.0, 0.14, textures["hay"], seed=12, squash=1.0, lumpiness=0.1, slices=7, stacks=5, tile=0.4)
    material(textured=True)
    glPopMatrix()


def draw_rowboat(prop, textures, material: Material, time: float) -> None:
    """A clinker-built rowing boat drawn up on the sand, bow to ``facing``, oars shipped."""
    hx, hz, height = _local(prop)
    length, beam = max(hx, hz), min(hx, hz)
    glPushMatrix()
    if hx > hz:
        glRotatef(90.0, 0.0, 1.0, 0.0)
    wood = textures["building"]
    material(color=(0.75, 0.62, 0.5), textured=True)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, wood)
    stations = 9
    prev = None
    for i in range(stations + 1):
        t = i / stations
        z = -length + 2 * length * t
        w = beam * math.sin(math.pi * min(max(t, 0.04), 0.96)) ** 0.7 * (1.0 if t < 0.85 else 0.9)
        keel = 0.02 + 0.12 * (abs(t - 0.5) * 2) ** 2
        gun = height * (0.85 + 0.25 * (abs(t - 0.5) * 2) ** 2)
        section = ((-w, gun, z), (-w * 0.55, keel + 0.05, z), (0.0, keel, z), (w * 0.55, keel + 0.05, z), (w, gun, z))
        if prev is not None:
            glBegin(GL_QUADS)
            for a in range(4):
                p0, p1, q1, q0 = prev[a], prev[a + 1], section[a + 1], section[a]
                nx = -1.0 if a < 2 else 1.0
                for (u, v), p in zip(((0, 0), (1, 0), (1, 1), (0, 1)), (p0, p1, q1, q0)):
                    glNormal3f(nx * 0.8, 0.4, 0.0)
                    glTexCoord2f(u, v)
                    glVertex3f(*p)
                for (u, v), p in zip(((0, 0), (1, 0), (1, 1), (0, 1)), (q0, q1, p1, p0)):    # the inside
                    glNormal3f(-nx * 0.8, 0.4, 0.0)
                    glTexCoord2f(u, v)
                    glVertex3f(*p)
            glEnd()
        prev = section
    glDisable(GL_TEXTURE_2D)
    material(color=(0.7, 0.58, 0.45), textured=True)
    for z in (-length * 0.3, length * 0.15):                        # thwarts
        _box(-beam * 0.8, height * 0.55, z - 0.1, beam * 0.8, height * 0.62, z + 0.1, textures["wood"], tile=1.0)
    for side in (-1.0, 1.0):                                        # the oars, shipped along the thwarts
        _box(side * beam * 0.5 - 0.04, height * 0.66, -length * 0.7, side * beam * 0.5 + 0.04, height * 0.72, length * 0.6,
             textures["wood"], tile=1.0)
    material(textured=True)
    glPopMatrix()
    glPopMatrix()


def draw_banner(prop, textures, material: Material, time: float) -> None:
    """A long banner hanging from a pole set out from a wall, its cloth stirring (``texture`` is the device)."""
    hx, hz, height = _local(prop)
    material(textured=True)
    _box(-hx - 0.05, height - 0.06, -0.05, hx + 0.05, height, 0.05, textures["wood"], tile=1.0)
    material(color=(1.0, 1.0, 1.0), textured=True)
    cloth = textures[prop.texture or "cloth"]
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, cloth)
    glBegin(GL_QUADS)
    rows = 6
    for i in range(rows):
        y0, y1 = height - 0.06 - height * i / rows, height - 0.06 - height * (i + 1) / rows
        z0 = 0.04 * math.sin(i * 0.9 + time * 1.3)
        z1 = 0.04 * math.sin((i + 1) * 0.9 + time * 1.3)
        for nz in (-1.0, 1.0):
            glNormal3f(0.0, 0.0, nz)
            pts = ((-hx, y1, z1), (hx, y1, z1), (hx, y0, z0), (-hx, y0, z0))
            uvs = ((0.0, 1.0 - (i + 1) / rows), (1.0, 1.0 - (i + 1) / rows), (1.0, 1.0 - i / rows), (0.0, 1.0 - i / rows))
            order = range(4) if nz > 0 else range(3, -1, -1)
            for k in order:
                glTexCoord2f(*uvs[k])
                glVertex3f(*pts[k])
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()


def draw_pool(prop, textures, material: Material, time: float) -> None:
    """Still water in a hollow of the rock: a dark disc ringed with wet stones (drawn live: it glints)."""
    cx, cz, hx, hz, base, height = _frame(prop)
    rng = random.Random(int(abs(cx * 3.7 + cz * 9.1)) % 1000)
    material(color=(0.85, 0.85, 0.85), textured=True)
    for k in range(9):
        a = 2 * math.pi * k / 9 + rng.uniform(-0.2, 0.2)
        r = rng.uniform(0.18, 0.32)
        draw_rock(cx + math.cos(a) * hx * 0.95, base + r * 0.4, cz + math.sin(a) * hz * 0.95, r, textures["cave_rock"],
                  seed=k + 70, squash=0.7, lumpiness=0.2, slices=7, stacks=5, tile=0.6)
    material(color=(0.45, 0.6, 0.7), emission=(0.02, 0.06, 0.07), textured=True)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textures["lake"])
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0.0, 1.0, 0.0)
    drift = time * 0.02
    glTexCoord2f(0.5 + drift, 0.5)
    glVertex3f(cx, base + 0.06, cz)
    for i in range(25):
        a = 2 * math.pi * i / 24
        glTexCoord2f(0.5 + 0.5 * math.cos(a) + drift, 0.5 + 0.5 * math.sin(a))
        glVertex3f(cx + math.cos(a) * hx * 0.9, base + 0.06, cz + math.sin(a) * hz * 0.9)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    material(textured=True)


def draw_timber(prop, textures, material: Material, time: float) -> None:
    """Pit props holding up a mine's roof: two posts and a cap-piece across the gallery."""
    hx, hz, height = _local(prop)
    material(color=(0.7, 0.6, 0.48), textured=True)
    wood = textures["wood"]
    for x in (-hx, hx - 0.22):
        _box(x, 0.0, -0.11, x + 0.22, height, 0.11, wood, tile=1.0)
    _box(-hx - 0.1, height - 0.24, -0.13, hx + 0.1, height, 0.13, wood, tile=1.0)
    material(textured=True)
    glPopMatrix()


def draw_minecart(prop, textures, material: Material, time: float) -> None:
    """An ore cart left on a short run of rail: an iron-bound box on four wheels, ore heaped in it."""
    hx, hz, height = _local(prop)
    material(color=(0.62, 0.62, 0.66), textured=True)
    iron, wood = textures["iron"], textures["wood"]
    for x in (-hx * 0.55, hx * 0.55):                       # the rails
        _box(x - 0.04, 0.0, -hz - 0.8, x + 0.04, 0.06, hz + 0.8, iron, tile=1.0)
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            glPushMatrix()
            glTranslatef(sx * hx * 0.62, 0.18, sz * hz * 0.6)
            glRotatef(90.0, 0.0, 0.0, 1.0)
            draw_cylinder(0.0, 0.0, 0.17, -0.05, 0.05, iron, iron, segments=10)
            glPopMatrix()
    material(color=(0.7, 0.6, 0.5), textured=True)
    _box(-hx, 0.25, -hz, hx, height, hz, wood, tile=0.8)
    material(textured=True)
    draw_rock(0.0, height, 0.0, min(hx, hz) * 0.75, textures["ore"], seed=5, squash=0.45, lumpiness=0.25, slices=8, stacks=5)
    glPopMatrix()


def draw_throne(prop, textures, material: Material, time: float) -> None:
    """A bandit king's seat: a heavy chair of dark wood with a high back, furs thrown over it."""
    hx, hz, height = _local(prop)
    material(color=(0.55, 0.38, 0.3), textured=True)
    wood = textures["wood"]
    _box(-hx, 0.0, -hz, hx, height * 0.32, hz, wood, tile=0.8)                 # the seat's block
    _box(-hx, 0.0, hz - 0.18, hx, height, hz, wood, tile=0.8)                  # the back (behind: +z local)
    for x in (-hx, hx - 0.16):
        _box(x, height * 0.32, -hz, x + 0.16, height * 0.52, hz, wood, tile=0.8)   # arms
    material(color=(0.6, 0.52, 0.42), textured=True)
    _box(-hx + 0.1, height * 0.32, -hz + 0.05, hx - 0.1, height * 0.36, hz - 0.18, textures["leather"])   # the fur
    material(textured=True)
    glPopMatrix()


def draw_crates(prop, textures, material: Material, time: float) -> None:
    """Crates stacked as they were carried in: two or three, a smaller one on top."""
    cx, cz, hx, hz, base, height = _frame(prop)
    rng = random.Random(int(abs(cx * 2.3 + cz * 4.1)) % 1000)
    material(textured=True)
    crate = textures["crate"]
    if hx > 0.8 or hz > 0.8:
        along_x = hx >= hz
        half = (hx if along_x else hz)
        for k, off in enumerate((-half / 2.0, half / 2.0)):
            w = half / 2.0 - 0.03
            h = height * rng.uniform(0.45, 0.55)
            if along_x:
                _box(cx + off - w, base, cz - hz, cx + off + w, base + h, cz + hz, crate)
            else:
                _box(cx - hx, base, cz + off - w, cx + hx, base + h, cz + off + w, crate)
        top = min(hx, hz) * 0.8
        _box(cx - top, base + height * 0.52, cz - top, cx + top, base + height, cz + top, crate)
    else:
        _box(cx - hx, base, cz - hz, cx + hx, base + height, cz + hz, crate)


def draw_sacks(container, textures, material: Material, time: float) -> None:
    """Sacks piled up, bulging (pepper, grain, contraband)."""
    cx, cz, hx, hz, base, height = _frame(container)
    rng = random.Random(int(abs(cx * 6.1 + cz * 2.9)) % 1000)
    material(color=(0.86, 0.78, 0.6), textured=True)
    for k in range(4):
        r = min(hx, hz) * rng.uniform(0.5, 0.7)
        x = cx + rng.uniform(-hx + r, hx - r) if hx > r else cx
        z = cz + rng.uniform(-hz + r, hz - r) if hz > r else cz
        y = base + r * 0.7 + (0.25 * height if k == 3 else 0.0)
        draw_rock(x, y, z, r, textures["hay"], seed=k + 90, squash=0.75, lumpiness=0.12, slices=8, stacks=5, tile=0.5)
    material(textured=True)


def draw_brazier(prop, textures, material: Material, time: float) -> None:
    """An iron fire-basket on three legs, coals glowing in it and flames licking up (drawn live)."""
    from .models import flame_scale
    cx, cz, hx, hz, base, height = _frame(prop)
    r = min(hx, hz)
    iron = textures["iron"]
    material(color=(0.55, 0.55, 0.6), textured=True)
    for k in range(3):
        a = 2 * math.pi * k / 3
        x, z = cx + math.cos(a) * r * 0.7, cz + math.sin(a) * r * 0.7
        _box(x - 0.04, base, z - 0.04, x + 0.04, base + height * 0.72, z + 0.04, iron)
    draw_cylinder(cx, cz, r, base + height * 0.62, base + height * 0.78, iron, textures["coals"], segments=10)
    material(color=(1.0, 0.6, 0.3), emission=(0.9, 0.4, 0.1), textured=True, fog=0.3)
    _box(cx - r * 0.8, base + height * 0.78, cz - r * 0.8, cx + r * 0.8, base + height * 0.8, cz + r * 0.8, textures["coals"])
    material(color=(1.0, 0.78, 0.5), emission=(0.9, 0.55, 0.2), textured=True, fog=0.25)
    for i, (fx, fz) in enumerate(((0.0, 0.0), (0.12, 0.06), (-0.1, 0.08), (0.04, -0.12))):
        k = flame_scale(time, i * 1.9 + cx)
        fh = (0.5 - 0.08 * i) * k
        _box(cx + fx - 0.07, base + height * 0.8, cz + fz - 0.07, cx + fx + 0.07, base + height * 0.8 + fh, cz + fz + 0.07,
             textures["flame"])
    material(textured=True)


# ----------------------------------------------------------------------------- the finer houses
def _face_quad(prop, out: float = 0.03):
    """The four corners of the face of a thin prop lying against a wall, its ``facing`` the way the
    wall's face looks, set ``out`` from it; and that face's normal."""
    mn, mx = prop.min_point, prop.max_point
    face = int(round(getattr(prop, "facing", 0.0) / 90.0)) % 4
    if face == 0:
        z = mn.z - out
        return ((mx.x, mn.y, z), (mn.x, mn.y, z), (mn.x, mx.y, z), (mx.x, mx.y, z)), (0.0, 0.0, -1.0)
    if face == 2:
        z = mx.z + out
        return ((mn.x, mn.y, z), (mx.x, mn.y, z), (mx.x, mx.y, z), (mn.x, mx.y, z)), (0.0, 0.0, 1.0)
    if face == 1:
        x = mx.x + out
        return ((x, mn.y, mx.z), (x, mn.y, mn.z), (x, mx.y, mn.z), (x, mx.y, mx.z)), (1.0, 0.0, 0.0)
    x = mn.x - out
    return ((x, mn.y, mn.z), (x, mn.y, mx.z), (x, mx.y, mx.z), (x, mx.y, mn.z)), (-1.0, 0.0, 0.0)


def draw_window(prop, textures, material: Material, time: float) -> None:
    """A window (or a painting, a tapestry) on a wall: its picture on the wall's face, a hair out
    from it - the wall round it shows through the picture's clear parts."""
    corners, normal = _face_quad(prop)
    material(textured=True)
    _decal_quad(corners, textures[prop.texture], normal)


def draw_rug(prop, textures, material: Material, time: float) -> None:
    """A rug on the floor (a decal over the boards)."""
    mn, mx = prop.min_point, prop.max_point
    y = mn.y + 0.07
    material(textured=True)
    _decal_quad(((mn.x, y, mx.z), (mx.x, y, mx.z), (mx.x, y, mn.z), (mn.x, y, mn.z)), textures[prop.texture], (0.0, 1.0, 0.0))


def draw_column(prop, textures, material: Material, time: float) -> None:
    """A column: a square base, a round shaft, a capital."""
    cx, cz, hx, hz, base, height = _frame(prop)
    r = min(hx, hz)
    tex = textures[prop.texture]
    material(color=(0.95, 0.93, 0.9), textured=True)
    _box(cx - r, base, cz - r, cx + r, base + 0.25, cz + r, tex)
    draw_cylinder(cx, cz, r * 0.72, base + 0.25, base + height - 0.3, tex, tex, segments=12, tile=1.0)
    _box(cx - r, base + height - 0.3, cz - r, cx + r, base + height, cz + r, tex)
    material(textured=True)


def draw_portico(prop, textures, material: Material, time: float) -> None:
    """A porch before a grand door: two columns, the beam they carry, a little pitched roof and its
    pediment - the front of the house puts on airs. ``facing``: the way it faces out."""
    hx, hz, height = _local(prop)
    stone = textures["marble"]
    material(color=(0.96, 0.94, 0.9), textured=True)
    roof_base = height * 0.72
    for x in (-hx + 0.3, hx - 0.3):
        _box(x - 0.32, 0.0, -hz + 0.05, x + 0.32, 0.25, -hz + 0.69, stone)
        draw_cylinder(x, -hz + 0.37, 0.22, 0.25, roof_base - 0.3, stone, stone, segments=12, tile=1.0)
        _box(x - 0.3, roof_base - 0.3, -hz + 0.07, x + 0.3, roof_base, -hz + 0.67, stone)
    _box(-hx, roof_base, -hz, hx, roof_base + 0.35, hz, stone)                   # the entablature
    material(color=(0.92, 0.9, 0.88), textured=True)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, stone)
    top = height
    base_y = roof_base + 0.35
    glBegin(GL_TRIANGLES)                                                        # the pediment, front and back
    for z, nz in ((-hz, -1.0), (hz, 1.0)):
        glNormal3f(0.0, 0.0, nz)
        pts = ((-hx - 0.1, base_y, z), (hx + 0.1, base_y, z), (0.0, top, z))
        if nz > 0:
            pts = pts[::-1]
        for (u, v), p in zip(((0, 0), (2, 0), (1, 1)), pts):
            glTexCoord2f(u, v)
            glVertex3f(*p)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, textures["roof_slate"])
    glBegin(GL_QUADS)
    for side in (-1.0, 1.0):
        glNormal3f(side * 0.6, 0.8, 0.0)
        pts = ((0.0, top + 0.05, -hz - 0.1), (0.0, top + 0.05, hz + 0.1), (side * (hx + 0.25), base_y - 0.05, hz + 0.1),
               (side * (hx + 0.25), base_y - 0.05, -hz - 0.1))
        if side < 0:
            pts = pts[::-1]
        for (u, v), p in zip(((0, 0), (1, 0), (1, 1), (0, 1)), pts):
            glTexCoord2f(u, v)
            glVertex3f(*p)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    material(textured=True)
    glPopMatrix()


def draw_chimney(prop, textures, material: Material, time: float) -> None:
    """A brick chimney stack with a stone cap and two pots."""
    mn, mx = prop.min_point, prop.max_point
    material(textured=True)
    _box(mn.x, mn.y, mn.z, mx.x, mx.y - 0.2, mx.z, textures["brick"], tile=1.0)
    _box(mn.x - 0.08, mx.y - 0.2, mn.z - 0.08, mx.x + 0.08, mx.y, mx.z + 0.08, textures["stone"], tile=1.0)
    cx, cz = (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0
    along_x = (mx.x - mn.x) >= (mx.z - mn.z)
    for k in (-1.0, 1.0):
        px, pz = (cx + k * (mx.x - mn.x) / 4.0, cz) if along_x else (cx, cz + k * (mx.z - mn.z) / 4.0)
        material(color=(0.8, 0.5, 0.4), textured=True)
        draw_cylinder(px, pz, 0.13, mx.y, mx.y + 0.45, textures["brick"], textures["coals"], segments=8)
    material(textured=True)


def draw_wardrobe(prop, textures, material: Material, time: float) -> None:
    """A tall wardrobe: a carved cornice, two doors with brass pulls, on little feet (front: ``facing``)."""
    hx, hz, height = _local(prop)
    wood = textures["wood"]
    material(color=(0.62, 0.42, 0.3), textured=True)
    _box(-hx, 0.12, -hz, hx, height - 0.15, hz, wood, tile=0.8)
    _box(-hx - 0.05, height - 0.15, -hz - 0.05, hx + 0.05, height, hz + 0.05, wood, tile=0.8)
    for x in (-hx + 0.08, hx - 0.14):
        for z in (-hz + 0.06, hz - 0.12):
            _box(x, 0.0, z, x + 0.06, 0.12, z + 0.06, wood)
    material(color=(0.55, 0.36, 0.26), textured=True)
    for x1, x2 in ((-hx + 0.06, -0.02), (0.02, hx - 0.06)):
        _box(x1, 0.3, -hz - 0.03, x2, height - 0.3, -hz, textures["door"], tile=None)
    material(color=(1.0, 0.85, 0.4), emission=(0.2, 0.15, 0.05), textured=True)
    for x in (-0.08, 0.05):
        _box(x, height * 0.5, -hz - 0.06, x + 0.03, height * 0.5 + 0.12, -hz - 0.02, textures["gilt"])
    material(textured=True)
    glPopMatrix()


def draw_bookcase(prop, textures, material: Material, time: float) -> None:
    """Shelves of books to the ceiling, their spines to the room (front: ``facing``)."""
    hx, hz, height = _local(prop)
    material(textured=True)
    _box(-hx, 0.0, -hz + 0.04, hx, height, hz, textures["wood"], tile=0.8)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textures["books"])
    glBegin(GL_QUADS)
    glNormal3f(0.0, 0.0, -1.0)
    reps = max(1.0, round(height / (2.0 * hx) * 1.2))
    for (u, v), p in zip(((0, 0), (1, 0), (1, reps), (0, reps)),
                         ((hx - 0.05, 0.05, -hz + 0.03), (-hx + 0.05, 0.05, -hz + 0.03), (-hx + 0.05, height - 0.05, -hz + 0.03),
                          (hx - 0.05, height - 0.05, -hz + 0.03))):
        glTexCoord2f(u, v)
        glVertex3f(*p)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()


def draw_fireplace(prop, textures, material: Material, time: float) -> None:
    """A marble chimneypiece against the wall (front: ``facing``): the surround, the mantel shelf, a
    black hearth and a fire of logs in it (drawn live: it burns)."""
    from .models import flame_scale
    hx, hz, height = _local(prop)
    marble = textures["marble"]
    material(color=(0.95, 0.93, 0.9), textured=True)
    _box(-hx, 0.0, -hz * 0.2, -hx + 0.35, height * 0.85, hz, marble)
    _box(hx - 0.35, 0.0, -hz * 0.2, hx, height * 0.85, hz, marble)
    _box(-hx, height * 0.62, -hz * 0.2, hx, height * 0.85, hz, marble)
    _box(-hx - 0.1, height * 0.85, -hz * 0.45, hx + 0.1, height * 0.9, hz, marble)               # the mantel
    material(color=(0.25, 0.22, 0.2), textured=True)
    _box(-hx + 0.35, 0.0, 0.1, hx - 0.35, height * 0.62, hz, textures["stone"])                 # the back of the hearth
    _box(-hx - 0.2, 0.0, -hz - 0.2, hx + 0.2, 0.06, -hz * 0.2, textures["stone"])                # the hearth stone
    material(textured=True)
    for i in range(3):
        glPushMatrix()
        glTranslatef(0.0, 0.08 + i * 0.05, 0.25)
        glRotatef(90.0 + (i - 1) * 25.0, 0.0, 1.0, 0.0)
        draw_log(-0.3, 0.3, 0.0, 0.0, 0.06, textures["bark"], textures["log_end"], segments=6)
        glPopMatrix()
    material(color=(1.0, 0.78, 0.5), emission=(0.9, 0.55, 0.2), textured=True, fog=0.25)
    for i, fx in enumerate((-0.15, 0.0, 0.14)):
        k = flame_scale(time, i * 2.1)
        _box(fx - 0.07, 0.12, 0.2, fx + 0.07, 0.12 + (0.35 - 0.06 * abs(i - 1)) * k, 0.34, textures["flame"])
    material(textured=True)
    glPopMatrix()


def draw_harpsichord(prop, textures, material: Material, time: float) -> None:
    """A harpsichord: a long wing-shaped case on turned legs, its lid propped, keys along the front."""
    hx, hz, height = _local(prop)
    wood = textures["wood"]
    material(color=(0.5, 0.3, 0.22), textured=True)
    for x, z in ((-hx + 0.1, -hz + 0.1), (hx - 0.2, -hz + 0.1), (-hx + 0.1, hz - 0.2), (hx * 0.3, hz - 0.2)):
        _box(x, 0.0, z, x + 0.08, height * 0.65, z + 0.08, wood)
    _box(-hx, height * 0.65, -hz, hx, height * 0.85, hz, wood, tile=0.8)
    material(color=(0.95, 0.93, 0.88), textured=True)
    _box(-hx + 0.1, height * 0.85, -hz + 0.02, hx - 0.1, height * 0.88, -hz + 0.3, textures["marble"])
    material(color=(0.15, 0.12, 0.12), textured=True)
    for k in range(12):
        x = -hx + 0.18 + k * (2 * hx - 0.36) / 12
        _box(x, height * 0.88, -hz + 0.18, x + 0.04, height * 0.9, -hz + 0.3, textures["iron"])
    material(color=(0.5, 0.3, 0.22), textured=True)
    glPushMatrix()
    glTranslatef(0.0, height * 0.85, hz)
    glRotatef(-40.0, 1.0, 0.0, 0.0)
    _box(-hx, 0.0, -2 * hz + 0.3, hx, 0.04, 0.0, wood, tile=0.8)
    glPopMatrix()
    material(textured=True)
    glPopMatrix()


def draw_globe(prop, textures, material: Material, time: float) -> None:
    cx, cz, hx, hz, base, height = _frame(prop)
    wood = textures["wood"]
    material(color=(0.6, 0.42, 0.3), textured=True)
    draw_cylinder(cx, cz, 0.05, base, base + height * 0.55, wood, wood, segments=6)
    draw_cylinder(cx, cz, min(hx, hz) * 0.8, base, base + 0.06, wood, wood, segments=10)
    material(color=(0.7, 0.8, 0.7), textured=True)
    draw_rock(cx, base + height * 0.72, cz, height * 0.22, textures["painting_landscape"], seed=1, squash=1.0, lumpiness=0.0,
              slices=12, stacks=8, tile=0.6)
    material(textured=True)


def draw_clock(prop, textures, material: Material, time: float) -> None:
    """A long-case clock, its face towards ``facing`` (and it ticks, for all the player can tell)."""
    hx, hz, height = _local(prop)
    wood = textures["wood"]
    material(color=(0.5, 0.32, 0.22), textured=True)
    _box(-hx, 0.0, -hz, hx, height * 0.15, hz, wood)
    _box(-hx * 0.8, height * 0.15, -hz * 0.8, hx * 0.8, height * 0.72, hz * 0.8, wood)
    _box(-hx, height * 0.72, -hz, hx, height, hz, wood)
    material(textured=True)
    _box(-hx * 0.75, height * 0.76, -hz - 0.01, hx * 0.75, height * 0.96, -hz, textures["clock_face"], tile=None)
    glPopMatrix()


def draw_bust(prop, textures, material: Material, time: float) -> None:
    """A marble bust of some great-grandfather on a plinth."""
    cx, cz, hx, hz, base, height = _frame(prop)
    marble = textures["marble"]
    material(color=(0.96, 0.95, 0.93), textured=True)
    _box(cx - hx, base, cz - hz, cx + hx, base + height * 0.6, cz + hz, marble)
    draw_rock(cx, base + height * 0.68, cz, min(hx, hz) * 0.9, marble, seed=4, squash=0.5, lumpiness=0.05, slices=9, stacks=5)
    draw_rock(cx, base + height * 0.86, cz, min(hx, hz) * 0.55, marble, seed=5, squash=1.2, lumpiness=0.05, slices=9, stacks=6)
    material(textured=True)


def draw_bathtub(prop, textures, material: Material, time: float) -> None:
    """A copper bath on claw feet."""
    cx, cz, hx, hz, base, height = _frame(prop)
    material(color=(0.85, 0.55, 0.35), emission=(0.05, 0.03, 0.01), textured=True)
    along_x = hx >= hz
    glPushMatrix()
    glTranslatef(cx, base + height * 0.55, cz)
    if along_x:
        glScalef(hx / hz, 1.0, 1.0)
    else:
        glScalef(1.0, 1.0, hz / hx)
    draw_rock(0.0, 0.0, 0.0, min(hx, hz), textures["gilt"], seed=9, squash=0.5, lumpiness=0.0, slices=14, stacks=7, tile=0.8)
    glPopMatrix()
    material(color=(0.7, 0.85, 0.9), emission=(0.05, 0.08, 0.1), textured=True)
    _box(cx - hx * 0.8, base + height * 0.78, cz - hz * 0.8, cx + hx * 0.8, base + height * 0.8, cz + hz * 0.8, textures["lake"])
    material(textured=True)


def draw_winerack(prop, textures, material: Material, time: float) -> None:
    """A rack of wine bottles, their ends to the room (front: ``facing``)."""
    hx, hz, height = _local(prop)
    material(color=(0.6, 0.45, 0.32), textured=True)
    _box(-hx, 0.0, -hz + 0.05, hx, height, hz, textures["wood"], tile=0.8)
    material(color=(0.2, 0.35, 0.2), emission=(0.02, 0.05, 0.02), textured=True)
    rows, cols = max(2, int(height / 0.3)), max(2, int(2 * hx / 0.28))
    for i in range(rows):
        for j in range(cols):
            x = -hx + 0.14 + j * (2 * hx - 0.28) / max(cols - 1, 1)
            y = 0.16 + i * (height - 0.32) / max(rows - 1, 1)
            _box(x - 0.06, y - 0.06, -hz, x + 0.06, y + 0.06, -hz + 0.08, textures["iron"])
    material(textured=True)
    glPopMatrix()


def draw_stove(prop, textures, material: Material, time: float) -> None:
    """A kitchen range: an iron stove under a brick hood, pots and a kettle on it (front: ``facing``)."""
    hx, hz, height = _local(prop)
    iron = textures["iron"]
    material(color=(0.45, 0.45, 0.48), textured=True)
    _box(-hx, 0.0, -hz, hx, height * 0.45, hz, iron)
    material(color=(1.0, 0.55, 0.25), emission=(0.6, 0.25, 0.05), textured=True)
    _box(-hx * 0.5, 0.12, -hz - 0.02, hx * 0.5, height * 0.32, -hz, textures["coals"], tile=None)
    material(textured=True)
    _box(-hx, height * 0.45, hz * 0.2, hx, height, hz, textures["brick"], tile=1.0)
    material(color=(0.55, 0.5, 0.48), textured=True)
    draw_cylinder(-hx * 0.45, -hz * 0.2, 0.2, height * 0.45, height * 0.45 + 0.28, iron, iron, segments=10)
    draw_cylinder(hx * 0.4, -hz * 0.25, 0.14, height * 0.45, height * 0.45 + 0.2, textures["gilt"], iron, segments=8)
    material(textured=True)
    glPopMatrix()


def draw_chandelier(prop, textures, material: Material, time: float) -> None:
    """A brass ring of candles hung from the ceiling on a chain (the light is a fixture of its own)."""
    cx, cz, hx, hz, base, height = _frame(prop)
    r = min(hx, hz)
    material(color=(1.0, 0.85, 0.45), emission=(0.15, 0.12, 0.04), textured=True)
    _box(cx - 0.02, base + 0.15, cz - 0.02, cx + 0.02, base + height, cz + 0.02, textures["iron"])
    for k in range(8):
        a = 2 * math.pi * k / 8
        x, z = cx + math.cos(a) * r, cz + math.sin(a) * r
        _box(min(cx, x) - 0.015, base + 0.1, min(cz, z) - 0.015, max(cx, x) + 0.015, base + 0.14, max(cz, z) + 0.015,
             textures["gilt"])
        material(color=(0.95, 0.92, 0.85), textured=True)
        _box(x - 0.025, base + 0.14, z - 0.025, x + 0.025, base + 0.3, z + 0.025, textures["wax"])
        material(color=(1.0, 0.85, 0.45), emission=(1.0, 0.75, 0.35), textured=True, fog=0.3)
        _box(x - 0.02, base + 0.3, z - 0.02, x + 0.02, base + 0.37, z + 0.02, textures["flame"])
        material(color=(1.0, 0.85, 0.45), emission=(0.15, 0.12, 0.04), textured=True)
    material(textured=True)


def draw_sofa(prop, textures, material: Material, time: float) -> None:
    """A settee upholstered in velvet (its seat faces ``facing``)."""
    hx, hz, height = _local(prop)
    material(color=(0.55, 0.12, 0.18) if prop.texture != "rug_blue" else (0.2, 0.28, 0.55), textured=True)
    cloth = textures["cloth"]
    _box(-hx, 0.15, -hz, hx, height * 0.5, hz, cloth, tile=0.6)
    _box(-hx, height * 0.5, hz - 0.25, hx, height, hz, cloth, tile=0.6)
    for x in (-hx, hx - 0.2):
        _box(x, height * 0.5, -hz, x + 0.2, height * 0.72, hz, cloth, tile=0.6)
    material(color=(0.5, 0.32, 0.2), textured=True)
    for x in (-hx + 0.05, hx - 0.12):
        for z in (-hz + 0.05, hz - 0.12):
            _box(x, 0.0, z, x + 0.07, 0.15, z + 0.07, textures["wood"])
    material(textured=True)
    glPopMatrix()


def draw_planter(prop, textures, material: Material, time: float) -> None:
    """A box of flowers (under a window, along a path)."""
    mn, mx = prop.min_point, prop.max_point
    material(color=(0.6, 0.45, 0.32), textured=True)
    _box(mn.x, mn.y, mn.z, mx.x, mn.y + (mx.y - mn.y) * 0.6, mx.z, textures["wood"])
    material(textured=True)
    _box(mn.x + 0.04, mn.y + (mx.y - mn.y) * 0.6, mn.z + 0.04, mx.x - 0.04, mx.y, mx.z - 0.04, textures["flowers"], tile=0.6)


def draw_fountain(prop, textures, material: Material, time: float) -> None:
    """A stone basin with a pillar and a bowl in the middle, water spilling (drawn live)."""
    from .structures import draw_ring
    cx, cz, hx, hz, base, height = _frame(prop)
    r = min(hx, hz)
    stone = textures["marble"]
    material(color=(0.92, 0.9, 0.86), textured=True)
    draw_ring(cx, cz, r - 0.3, r, base, base + 0.6, stone, stone, tile=1.0)
    draw_cylinder(cx, cz, 0.25, base, base + height * 0.75, stone, stone, segments=10)
    draw_cylinder(cx, cz, 0.8, base + height * 0.75, base + height * 0.8, stone, stone, segments=14)
    material(color=(0.55, 0.72, 0.85), emission=(0.02, 0.05, 0.07), textured=True)
    drift = time * 0.05
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textures["lake"])
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0.0, 1.0, 0.0)
    glTexCoord2f(0.5 + drift, 0.5)
    glVertex3f(cx, base + 0.45, cz)
    for i in range(21):
        a = 2 * math.pi * i / 20
        glTexCoord2f(0.5 + 0.5 * math.cos(a) + drift, 0.5 + 0.5 * math.sin(a))
        glVertex3f(cx + math.cos(a) * (r - 0.3), base + 0.45, cz + math.sin(a) * (r - 0.3))
    glEnd()
    glDisable(GL_TEXTURE_2D)
    material(color=(0.7, 0.85, 0.95), emission=(0.1, 0.15, 0.2), textured=True, fog=0.5)
    k = 0.9 + 0.1 * math.sin(time * 6.0)
    draw_cylinder(cx, cz, 0.07 * k, base + height * 0.8, base + height, textures["lake"], textures["lake"], segments=6)
    material(textured=True)


def draw_topiary(prop, textures, material: Material, time: float) -> None:
    """A bay tree clipped into a ball on a stem, in a square tub."""
    cx, cz, hx, hz, base, height = _frame(prop)
    r = min(hx, hz)
    material(color=(0.6, 0.45, 0.32), textured=True)
    _box(cx - r * 0.6, base, cz - r * 0.6, cx + r * 0.6, base + 0.5, cz + r * 0.6, textures["wood"])
    draw_cylinder(cx, cz, 0.06, base + 0.5, base + height * 0.55, textures["bark"], textures["bark"], segments=6)
    material(color=(0.75, 0.9, 0.7), textured=True)
    draw_rock(cx, base + height * 0.72, cz, r, textures["leaves"], seed=int(abs(cx * 7 + cz)) % 97, squash=1.0, lumpiness=0.06,
              slices=10, stacks=7, tile=0.5)
    material(textured=True)


def draw_railing(prop, textures, material: Material, time: float) -> None:
    """Iron railings along the footprint's longer side: spear-headed bars between two rails."""
    mn, mx = prop.min_point, prop.max_point
    along_x = (mx.x - mn.x) >= (mx.z - mn.z)
    iron = textures["iron"]
    material(color=(0.25, 0.25, 0.28), textured=True)
    length = (mx.x - mn.x) if along_x else (mx.z - mn.z)
    h = mx.y - mn.y
    for y in (mn.y + 0.15, mn.y + h - 0.2):
        if along_x:
            _box(mn.x, y, mn.z, mx.x, y + 0.05, mx.z, iron, tile=1.0)
        else:
            _box(mn.x, y, mn.z, mx.x, y + 0.05, mx.z, iron, tile=1.0)
    n = max(2, int(length / 0.18))
    for i in range(n + 1):
        t = length * i / n
        if along_x:
            _box(mn.x + t - 0.02, mn.y, mn.z, mn.x + t + 0.02, mx.y, mx.z, iron)
        else:
            _box(mn.x, mn.y, mn.z + t - 0.02, mx.x, mx.y, mn.z + t + 0.02, iron)
    material(textured=True)


def draw_bench(prop, textures, material: Material, time: float) -> None:
    """A plank bench on two trestles (long side along the longer footprint)."""
    mn, mx = prop.min_point, prop.max_point
    wood = textures["wood"]
    material(color=(0.78, 0.66, 0.5), textured=True)
    h = mx.y - mn.y
    _box(mn.x, mn.y + h * 0.8, mn.z, mx.x, mx.y, mx.z, wood, tile=1.0)
    along_x = (mx.x - mn.x) >= (mx.z - mn.z)
    for k in (0.12, 0.88):
        if along_x:
            x = mn.x + (mx.x - mn.x) * k
            _box(x - 0.06, mn.y, mn.z + 0.05, x + 0.06, mn.y + h * 0.8, mx.z - 0.05, wood)
        else:
            z = mn.z + (mx.z - mn.z) * k
            _box(mn.x + 0.05, mn.y, z - 0.06, mx.x - 0.05, mn.y + h * 0.8, z + 0.06, wood)
    material(textured=True)


def draw_bell(prop, textures, material: Material, time: float) -> None:
    """A bronze bell hung in a belfry's frame."""
    cx, cz, hx, hz, base, height = _frame(prop)
    r = min(hx, hz)
    material(color=(0.75, 0.55, 0.3), emission=(0.05, 0.03, 0.0), textured=True)
    draw_rock(cx, base + height * 0.45, cz, r, textures["gilt"], seed=2, squash=1.1, lumpiness=0.0, slices=12, stacks=8, tile=0.8)
    material(color=(0.4, 0.3, 0.2), textured=True)
    _box(cx - r - 0.1, base + height - 0.1, cz - 0.06, cx + r + 0.1, base + height, cz + 0.06, textures["wood"])
    material(textured=True)


HEADSTONE_THICK = 0.16


def draw_headstone(prop, textures, material: Material, time: float) -> None:
    """A grave, within the prop's footprint: the low mound, and at its head (the far end, looking
    the way the prop ``facing`` - over the mound, at whoever stands at the grave's foot) the
    headstone, leaning back a little."""
    hx, hz, height = _local(prop)
    material(color=(0.75, 0.74, 0.72), textured=True)
    glPushMatrix()
    glTranslatef(0.0, 0.0, hz - HEADSTONE_THICK)       # the head of the grave (local +z: behind the stone's face)
    glRotatef(6.0, 1.0, 0.0, 0.0)                      # (leaning back, away from the mound)
    _box(-hx, 0.0, 0.0, hx, height - hx, HEADSTONE_THICK, textures["stone"])
    glPopMatrix()
    material(color=(0.7, 0.9, 0.7), textured=True)
    _box(-hx + 0.05, 0.0, -hz + 0.05, hx - 0.05, 0.18, hz - HEADSTONE_THICK - 0.12, textures["leaves"], tile=0.8)
    material(textured=True)
    glPopMatrix()


def draw_washing_line(prop, textures, material: Material, time: float) -> None:
    """A line between two posts, washing pegged out along it and stirring in the wind (drawn live)."""
    mn, mx = prop.min_point, prop.max_point
    along_x = (mx.x - mn.x) >= (mx.z - mn.z)
    wood = textures["wood"]
    material(textured=True)
    h = mx.y - mn.y
    cz, cx = (mn.z + mx.z) / 2.0, (mn.x + mx.x) / 2.0
    ends = ((mn.x, cz), (mx.x, cz)) if along_x else ((cx, mn.z), (cx, mx.z))
    for x, z in ends:
        _box(x - 0.05, mn.y, z - 0.05, x + 0.05, mx.y, z + 0.05, wood)
    length = (mx.x - mn.x) if along_x else (mx.z - mn.z)
    cloths = ((0.9, 0.9, 0.95), (0.7, 0.2, 0.2), (0.3, 0.4, 0.7), (0.9, 0.85, 0.6), (0.95, 0.95, 0.95))
    n = max(2, int(length / 0.9))
    for i in range(n):
        t = (i + 0.5) / n
        sway = 0.06 * math.sin(time * 2.0 + i * 1.3)
        material(color=cloths[i % len(cloths)], textured=True)
        w = 0.3
        if along_x:
            x = mn.x + length * t
            _box(x - w, mn.y + h - 0.75, cz - 0.01 + sway, x + w, mn.y + h - 0.05, cz + 0.01 + sway, textures["cloth"])
        else:
            z = mn.z + length * t
            _box(cx - 0.01 + sway, mn.y + h - 0.75, z - w, cx + 0.01 + sway, mn.y + h - 0.05, z + w, textures["cloth"])
    material(textured=True)


DRAWERS = {"brazier": draw_brazier, "bedroll": draw_bedroll, "notice": draw_notice, "dummy": draw_dummy, "rowboat": draw_rowboat,
           "banner": draw_banner, "pool": draw_pool, "timber": draw_timber, "minecart": draw_minecart, "throne": draw_throne,
           "crates": draw_crates, "window": draw_window, "painting": draw_window, "rug": draw_rug, "column": draw_column,
           "portico": draw_portico, "chimney": draw_chimney, "wardrobe": draw_wardrobe, "bookcase": draw_bookcase,
           "fireplace": draw_fireplace, "harpsichord": draw_harpsichord, "globe": draw_globe, "clock": draw_clock, "bust": draw_bust,
           "bathtub": draw_bathtub, "winerack": draw_winerack, "stove": draw_stove, "chandelier": draw_chandelier,
           "sofa": draw_sofa, "planter": draw_planter, "fountain": draw_fountain, "topiary": draw_topiary,
           "railing": draw_railing, "bench": draw_bench, "bell": draw_bell, "headstone": draw_headstone,
           "washing_line": draw_washing_line}
ANIMATED = ("banner", "pool", "brazier", "fireplace", "fountain", "washing_line")
def draw_barrel_store(container, textures, material: Material, time: float) -> None:
    """A barrel that is a hiding place (a rain barrel, a false head): drawn as any barrel."""
    from .structures import draw_barrel
    draw_barrel(container, textures, material, time)


CONTAINER_DRAWERS = {"skeleton": draw_skeleton, "stash": draw_stash, "sacks": draw_sacks, "barrel_store": draw_barrel_store}
READABLE_DRAWERS = {"notice": draw_notice, "signpost": draw_signpost, "plaque": draw_plaque, "book": draw_book}
