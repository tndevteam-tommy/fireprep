"""Modelled props: things that are more than a textured box - the town well, the
smith's forge, the inn's bread oven and the signs hung beside shop doors.

Each ``draw_*`` fills the footprint of the prop it is given (``Prop.kind`` picks
the function in :mod:`scene`), so the level only states *where* a structure stands
and how big it is; the modelling lives here.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence

from OpenGL.GL import (GL_BLEND, GL_ONE_MINUS_SRC_ALPHA, GL_POLYGON_OFFSET_FILL, GL_QUADS, GL_SRC_ALPHA, GL_TEXTURE_2D, GL_TRIANGLES,
                       GL_TRIANGLE_FAN, glBegin, glBindTexture, glBlendFunc, glDepthMask, glDisable, glEnable, glEnd,
                       glNormal3f, glPolygonOffset, glPopMatrix, glPushMatrix, glRotatef, glScalef, glTexCoord2f, glTranslatef, glVertex3f)

from ..core.math3d import Vector3D
from .models import flame_scale
from .primitives import draw_board, draw_textured_box

Material = Callable[..., None]
SEGMENTS = 18


# ----------------------------------------------------------------------------- round things
def draw_ring(cx: float, cz: float, r_in: float, r_out: float, y0: float, y1: float, side_tex: int, top_tex: int,
              inner_tex: int = None, tile: float = 1.0, segments: int = SEGMENTS) -> None:
    """A hollow cylinder (a well's wall): outer face, inner face and the flat rim on top."""
    inner_tex = side_tex if inner_tex is None else inner_tex
    glEnable(GL_TEXTURE_2D)
    step = 2.0 * math.pi / segments
    for radius, tex, outward in ((r_out, side_tex, 1.0), (r_in, inner_tex, -1.0)):
        glBindTexture(GL_TEXTURE_2D, tex)
        glBegin(GL_QUADS)
        for i in range(segments):
            a0, a1 = i * step, (i + 1) * step
            x0, z0, x1, z1 = cx + radius * math.cos(a0), cz + radius * math.sin(a0), cx + radius * math.cos(a1), cz + radius * math.sin(a1)
            u0, u1 = (a0 * radius) / tile, (a1 * radius) / tile
            nx, nz = outward * math.cos((a0 + a1) / 2), outward * math.sin((a0 + a1) / 2)
            glNormal3f(nx, 0.0, nz)
            order = ((x0, z0, u0), (x1, z1, u1)) if outward > 0 else ((x1, z1, u1), (x0, z0, u0))
            (ax, az, au), (bx, bz, bu) = order
            glTexCoord2f(au, y0 / tile); glVertex3f(ax, y0, az)
            glTexCoord2f(bu, y0 / tile); glVertex3f(bx, y0, bz)
            glTexCoord2f(bu, y1 / tile); glVertex3f(bx, y1, bz)
            glTexCoord2f(au, y1 / tile); glVertex3f(ax, y1, az)
        glEnd()
    glBindTexture(GL_TEXTURE_2D, top_tex)
    glBegin(GL_QUADS)
    glNormal3f(0.0, 1.0, 0.0)
    for i in range(segments):
        a0, a1 = i * step, (i + 1) * step
        for (ra, a) in ((r_in, a0), (r_out, a0), (r_out, a1), (r_in, a1)):
            x, z = cx + ra * math.cos(a), cz + ra * math.sin(a)
            glTexCoord2f((x - cx) / tile, (z - cz) / tile); glVertex3f(x, y1, z)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_disc(cx: float, y: float, cz: float, radius: float, tex: int, tile: float = 1.0, segments: int = SEGMENTS) -> None:
    """A flat textured disc facing up (the water in the well)."""
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0.0, 1.0, 0.0)
    glTexCoord2f(0.5, 0.5); glVertex3f(cx, y, cz)
    for i in range(segments + 1):
        a = i * 2.0 * math.pi / segments
        x, z = cx + radius * math.cos(a), cz + radius * math.sin(a)
        glTexCoord2f(0.5 + 0.5 * math.cos(a) * radius / tile, 0.5 + 0.5 * math.sin(a) * radius / tile)
        glVertex3f(x, y, z)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_gable_roof(cx: float, cz: float, half_x: float, half_z: float, eave_y: float, ridge_y: float, tex: int,
                    tile: float = 1.0, thickness: float = 0.0, edge_tex: int = None) -> None:
    """A small pitched roof with its ridge running along x: two slopes and, given a
    ``thickness``, a matching underside plus the boards' edges along the eaves, the ridge
    and the two gable ends (``edge_tex``, default the roof texture) - a slab, not a sheet."""
    edge_tex = tex if edge_tex is None else edge_tex
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glBegin(GL_QUADS)
    uv = ((0.0, 0.0), (2 * half_x / tile, 0.0), (2 * half_x / tile, 1.0), (0.0, 1.0))
    for side in (-1.0, 1.0):
        z_eave = cz + side * half_z
        ny, nz = half_z, side * (ridge_y - eave_y)
        m = math.hypot(ny, nz) or 1.0
        glNormal3f(0.0, ny / m, nz / m)
        corners = ((cx - half_x, eave_y, z_eave), (cx + half_x, eave_y, z_eave), (cx + half_x, ridge_y, cz), (cx - half_x, ridge_y, cz))
        if side < 0:
            corners = corners[::-1]
        for (x, y, z), (u, v) in zip(corners, uv):
            glTexCoord2f(u, v); glVertex3f(x, y, z)
        if thickness > 0.0:                                   # the underside, facing down into the shelter
            glNormal3f(0.0, -ny / m, -nz / m)
            for (x, y, z), (u, v) in zip(corners[::-1], uv):
                glTexCoord2f(u, v); glVertex3f(x, y - thickness, z)
    glEnd()
    if thickness > 0.0:
        glBindTexture(GL_TEXTURE_2D, edge_tex)
        glBegin(GL_QUADS)
        for side in (-1.0, 1.0):                              # the eave edges
            z_eave = cz + side * half_z
            glNormal3f(0.0, 0.0, side)
            pts = ((cx - half_x, eave_y - thickness), (cx + half_x, eave_y - thickness), (cx + half_x, eave_y), (cx - half_x, eave_y))
            if side < 0:
                pts = pts[::-1]
            for (x, y), (u, v) in zip(pts, ((0, 0), (1, 0), (1, 1), (0, 1))):
                glTexCoord2f(u, v); glVertex3f(x, y, z_eave)
        for gx, nx in ((cx - half_x, -1.0), (cx + half_x, 1.0)):    # the gable ends: the slab's cut profile
            glNormal3f(nx, 0.0, 0.0)
            profile = ((eave_y, cz - half_z), (ridge_y, cz), (eave_y, cz + half_z))
            for i in range(2):
                (ya, za), (yb, zb) = profile[i], profile[i + 1]
                quad = ((ya - thickness, za), (yb - thickness, zb), (yb, zb), (ya, za))
                if nx < 0:
                    quad = quad[::-1]
                for (y, z), (u, v) in zip(quad, ((0, 0), (1, 0), (1, 1), (0, 1))):
                    glTexCoord2f(u, v); glVertex3f(gx, y, z)
        glEnd()
    glDisable(GL_TEXTURE_2D)


_LATTICES: dict = {}


def _sphere_lattice(slices: int, stacks: int):
    """(unit normals, (u, v) per unit radius, lattice index) for every triangle corner of a
    ``slices`` x ``stacks`` sphere - each quad (i, j) as (i, j) (i, j+1) (i+1, j+1) and
    (i, j) (i+1, j+1) (i+1, j) - computed once per size."""
    import numpy as np
    key = (slices, stacks)
    if key not in _LATTICES:
        lat = math.pi * (-0.5 + np.arange(stacks + 1) / stacks)
        lng = 2.0 * math.pi * np.arange(slices + 1) / slices
        LAT, LNG = np.meshgrid(lat, lng, indexing="ij")
        n = np.stack([np.cos(LAT) * np.cos(LNG), np.sin(LAT), np.cos(LAT) * np.sin(LNG)], axis=-1).reshape(-1, 3)
        uv = np.stack([LNG / math.pi, LAT / math.pi + 0.5], axis=-1).reshape(-1, 2)
        I, J = np.meshgrid(np.arange(stacks), np.arange(slices), indexing="ij")
        I, J = I.ravel(), J.ravel()
        ti = np.stack([I, I, I + 1, I, I + 1, I + 1], axis=-1).ravel()
        tj = np.stack([J, J + 1, J + 1, J, J + 1, J], axis=-1).ravel()
        tri = ti * (slices + 1) + tj
        _LATTICES[key] = (n[tri], uv[tri], tri)
    return _LATTICES[key]


def draw_rock(cx: float, cy: float, cz: float, radius: float, tex: int, seed: int, squash: float = 0.7,
              lumpiness: float = 0.22, slices: int = 10, stacks: int = 7, tile: float = 1.0) -> None:
    """A lumpy boulder: a sphere whose vertices are pushed in and out by a seeded amount
    (``lumpiness`` x radius), flattened to ``squash`` of its radius vertically so it sits
    on the ground like a rock rather than a ball. Textured around and over. (Handed to GL as
    arrays: the woods are thousands of these.)"""
    import numpy as np
    from .primitives import draw_arrays
    rng = random.Random(seed)
    bumps = np.ones((stacks + 1, slices + 1))
    inner = np.array([[1.0 + lumpiness * rng.uniform(-1.0, 1.0) for _ in range(slices)] for _ in range(stacks - 1)])
    if stacks > 1:
        bumps[1:stacks, :slices] = inner
        bumps[1:stacks, slices] = bumps[1:stacks, 0]           # seam: the same bump as the first vertex
    normals, lat_lng, tri = _sphere_lattice(slices, stacks)
    b = bumps.ravel()[tri]
    verts = normals * (radius * b)[:, None]
    verts[:, 1] *= squash
    verts += (cx, cy, cz)
    texcoords = lat_lng * (radius / tile)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    draw_arrays(GL_TRIANGLES, verts, normals, texcoords)
    glDisable(GL_TEXTURE_2D)


# ----------------------------------------------------------------------------- the structures
def draw_ore_deposit(station, textures, material: Material, time: float) -> None:
    """An outcrop of iron ore filling the station's footprint: a few large lumpy boulders,
    half sunk into the ground, with smaller stones scattered about their feet."""
    mn, mx = station.min_point, station.max_point
    cx, cz = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2
    hx, hz, height = (mx.x - mn.x) / 2, (mx.z - mn.z) / 2, mx.y - mn.y
    ore = textures[station.texture]
    material(textured=True)
    # (offset x, offset z, radius, how deep it sits) as fractions of the footprint
    boulders = ((0.0, 0.0, 0.62, 0.35), (-0.55, 0.35, 0.42, 0.3), (0.5, -0.4, 0.46, 0.3), (0.45, 0.5, 0.34, 0.3),
                (-0.5, -0.55, 0.3, 0.3), (-0.1, 0.7, 0.22, 0.2), (0.15, -0.75, 0.2, 0.2), (0.75, 0.05, 0.18, 0.2))
    for k, (ox, oz, rf, sink) in enumerate(boulders):
        radius = rf * min(hx, hz)
        draw_rock(cx + ox * hx, mn.y + radius * (1.0 - sink) * 0.75, cz + oz * hz, min(radius, height / (1.6 * 0.75)),
                  ore, seed=k + 17, squash=0.75, lumpiness=0.25 if radius > 0.4 else 0.35)


def draw_well(prop, textures, material: Material, time: float) -> None:
    """A round stone well: a hollow ring you can look down into (water at the bottom), two
    posts carrying a windlass with its rope and bucket, and a little wooden roof over it all."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2
    r_out = min(mx.x - mn.x, mx.z - mn.z) / 2 * 0.8
    r_in = r_out - 0.32
    rim_y = mx.y
    material(textured=True)
    draw_ring(cx, cz, r_in, r_out, mn.y, rim_y, textures["wall"], textures["stone"], textures["stone"], tile=1.2)
    material(color=(0.55, 0.55, 0.6), textured=True)
    draw_ring(cx, cz, r_in - 0.01, r_in, mn.y - 0.02, mn.y + 0.02, textures["stone"], textures["stone"])   # shaft lip
    material(color=(0.7, 0.8, 0.9), emission=(0.02, 0.04, 0.06), textured=True)
    draw_disc(cx, mn.y + 0.5, cz, r_in, textures["water"], tile=2.0 * r_in)            # the water, half way down the shaft
    # posts, crossbeam, windlass and roof
    material(textured=True)
    post_x = r_out + 0.1
    post_top = rim_y + 1.5
    for sx in (-1.0, 1.0):
        draw_textured_box(Vector3D(cx + sx * post_x - 0.08, mn.y, cz - 0.08), Vector3D(cx + sx * post_x + 0.08, post_top, cz + 0.08),
                          textures["wood"], tile=None)
    draw_textured_box(Vector3D(cx - post_x - 0.1, post_top - 0.16, cz - 0.1), Vector3D(cx + post_x + 0.1, post_top, cz + 0.1),
                      textures["wood"], tile=None)                                              # crossbeam
    axle_y = rim_y + 0.95
    draw_textured_box(Vector3D(cx - post_x, axle_y - 0.1, cz - 0.1), Vector3D(cx + post_x, axle_y + 0.1, cz + 0.1),
                      textures["torch"], tile=None)                                             # the windlass drum
    draw_textured_box(Vector3D(cx + post_x + 0.08, axle_y - 0.04, cz - 0.04), Vector3D(cx + post_x + 0.5, axle_y + 0.04, cz + 0.04),
                      textures["wood"], tile=None)                                              # crank
    draw_textured_box(Vector3D(cx + post_x + 0.42, axle_y - 0.04, cz - 0.045), Vector3D(cx + post_x + 0.505, axle_y + 0.4, cz + 0.045),
                      textures["wood"], tile=None)                                              # its handle (proud of the arm)
    material(color=(0.8, 0.72, 0.55), textured=True)
    draw_textured_box(Vector3D(cx - 0.015, rim_y + 0.25, cz - 0.015), Vector3D(cx + 0.015, axle_y, cz + 0.015),
                      textures["cloth"], tile=None)                                             # rope
    material(textured=True)
    # the bucket: a round wooden pail (staves, a hoop and a bottom) hanging under the rope
    bucket_y = rim_y + 0.02
    draw_ring(cx, cz, 0.13, 0.165, bucket_y, bucket_y + 0.26, textures["wood"], textures["wood"], tile=0.5, segments=12)
    draw_disc(cx, bucket_y + 0.03, cz, 0.135, textures["wood"], tile=0.5, segments=12)           # the bottom, seen from inside
    material(color=(0.35, 0.35, 0.4), textured=True)
    draw_ring(cx, cz, 0.165, 0.18, bucket_y + 0.17, bucket_y + 0.21, textures["iron"], textures["iron"], segments=12)   # hoop
    material(textured=True)
    # the roof: a wooden ceiling board the posts and crossbeam carry, with a slab of a roof on top
    roof_half = r_out + 0.45
    half_x, half_z = roof_half + 0.1, roof_half * 0.8
    draw_textured_box(Vector3D(cx - half_x, post_top - 0.08, cz - half_z), Vector3D(cx + half_x, post_top, cz + half_z),
                      textures["wood"], tile=1.0)                                               # the ceiling
    draw_gable_roof(cx, cz, half_x, half_z, post_top + 0.12, post_top + 0.77, textures["roof"], tile=2.0,
                    thickness=0.12, edge_tex=textures["wood"])
    material(color=(1.0, 1.0, 1.0), textured=True)


def draw_forge(prop, textures, material: Material, time: float) -> None:
    """The smith's forge: a waist-high stone hearth with a shallow bowl sunk into its top, full
    of glowing coals with several small flames licking up, and a hood/chimney at the back."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2
    top = mn.y + min(mx.y - mn.y, 1.0)
    glPushMatrix()
    glTranslatef(cx, 0.0, cz)
    glRotatef(-prop.facing, 0.0, 1.0, 0.0)
    hx, hz = (mx.x - mn.x) / 2, (mx.z - mn.z) / 2
    material(textured=True)
    # the hearth: four stone walls around the bowl and a stone base, so the bowl is sunk in
    lip = 0.16
    bowl_depth = 0.14
    draw_textured_box(Vector3D(-hx, mn.y, -hz), Vector3D(hx, top - bowl_depth, hz), textures["wall"], tile=1.0)
    for x1, x2, z1, z2 in ((-hx, -hx + lip, -hz, hz), (hx - lip, hx, -hz, hz), (-hx, hx, -hz, -hz + lip), (-hx, hx, hz - lip, hz)):
        draw_textured_box(Vector3D(x1, top - bowl_depth, z1), Vector3D(x2, top, z2), textures["stone"], tile=1.0)
    # the bed of coals in the bowl
    material(color=(1.0, 0.9, 0.85), emission=(0.6, 0.22, 0.06), textured=True, fog=0.5)
    draw_textured_box(Vector3D(-hx + lip, top - bowl_depth, -hz + lip), Vector3D(hx - lip, top - bowl_depth + 0.1, hz - lip),
                      textures["coals"], tile=None)
    # several small flames, each flickering on its own
    material(color=(1.0, 0.78, 0.5), emission=(0.9, 0.55, 0.2), textured=True, fog=0.25)
    for i, (fx, fz) in enumerate(((-0.4, -0.15), (0.05, 0.12), (0.42, -0.08), (-0.15, -0.3), (0.25, 0.28), (-0.3, 0.25), (0.2, -0.3))):
        k = flame_scale(time, i * 1.9)
        fh, fw = 0.16 * k + 0.04 * (i % 3), 0.055
        base = top - bowl_depth + 0.06
        draw_textured_box(Vector3D(fx - fw, base, fz - fw), Vector3D(fx + fw, base + fh, fz + fw), textures["flame"], tile=None)
    # the hood at the back (+z is the back: the front faces -z), narrowing into a chimney
    material(textured=True)
    draw_textured_box(Vector3D(-hx, top, hz - 0.25), Vector3D(hx, top + 0.9, hz), textures["wall"], tile=1.0)
    draw_textured_box(Vector3D(-hx * 0.6, top + 0.9, hz - 0.25), Vector3D(hx * 0.6, top + 1.7, hz), textures["wall"], tile=1.0)
    draw_textured_box(Vector3D(-hx * 0.35, top + 1.7, hz - 0.22), Vector3D(hx * 0.35, top + 2.6, hz - 0.03), textures["stone"], tile=1.0)
    # the bellows: a leather wedge at the side, its nozzle into the coals
    material(color=(1.0, 1.0, 1.0), textured=True)
    draw_textured_box(Vector3D(hx, top - 0.45, -0.25), Vector3D(hx + 0.55, top - 0.15, 0.25), textures["leather"], tile=None)
    draw_textured_box(Vector3D(hx + 0.5, top - 0.4, -0.05), Vector3D(hx + 0.9, top - 0.2, 0.05), textures["wood"], tile=None)
    glPopMatrix()
    material(color=(1.0, 1.0, 1.0), textured=True)


def draw_oven(prop, textures, material: Material, time: float) -> None:
    """The inn's bread oven: a stone box with an arched mouth in the front, the fire set well
    back inside it and a shelf in front of the mouth where pots and loaves are set to warm."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2
    hx, hz, top = (mx.x - mn.x) / 2, (mx.z - mn.z) / 2, mx.y
    glPushMatrix()
    glTranslatef(cx, 0.0, cz)
    glRotatef(-prop.facing, 0.0, 1.0, 0.0)
    material(textured=True)
    mouth_w, mouth_h, sill = 0.55, 0.6, 0.45
    front = -hz
    # the body: a floor slab, two cheeks either side of the mouth, the lintel above it and the back
    draw_textured_box(Vector3D(-hx, mn.y, front), Vector3D(hx, mn.y + sill, hz), textures["wall"], tile=1.0)        # hearth slab
    draw_textured_box(Vector3D(-hx, mn.y + sill, front), Vector3D(-mouth_w, top, hz), textures["wall"], tile=1.0)   # left cheek
    draw_textured_box(Vector3D(mouth_w, mn.y + sill, front), Vector3D(hx, top, hz), textures["wall"], tile=1.0)     # right cheek
    draw_textured_box(Vector3D(-mouth_w, mn.y + sill + mouth_h, front), Vector3D(mouth_w, top, hz), textures["wall"], tile=1.0)  # lintel
    draw_textured_box(Vector3D(-mouth_w, mn.y + sill, front + 1.0), Vector3D(mouth_w, top, hz), textures["stone"], tile=1.0)      # the back
    # inside the mouth: a dark floor, the fire at the back of the chamber
    material(color=(0.45, 0.42, 0.4), textured=True)
    draw_textured_box(Vector3D(-mouth_w, mn.y + sill - 0.02, front + 0.02), Vector3D(mouth_w, mn.y + sill + 0.02, front + 1.0),
                      textures["stone"], tile=1.0)
    material(color=(1.0, 0.9, 0.85), emission=(0.6, 0.22, 0.06), textured=True, fog=0.5)
    draw_textured_box(Vector3D(-mouth_w + 0.08, mn.y + sill + 0.02, front + 0.6), Vector3D(mouth_w - 0.08, mn.y + sill + 0.08, front + 0.98),
                      textures["coals"], tile=None)
    material(color=(1.0, 0.78, 0.5), emission=(0.9, 0.55, 0.2), textured=True, fog=0.25)
    for i, fx in enumerate((-0.25, 0.0, 0.25)):
        k = flame_scale(time, 0.7 + i * 2.1)
        fh, fw = 0.3 * k, 0.1
        draw_textured_box(Vector3D(fx - fw, mn.y + sill + 0.06, front + 0.72), Vector3D(fx + fw, mn.y + sill + 0.06 + fh, front + 0.9),
                          textures["flame"], tile=None)
    # a stone shelf in front of the mouth with a pot and a couple of loaves warming on it
    material(color=(1.0, 1.0, 1.0), textured=True)
    draw_textured_box(Vector3D(-hx, mn.y + sill - 0.08, front - 0.45), Vector3D(hx, mn.y + sill, front + 0.02), textures["stone"], tile=1.0)
    material(color=(0.25, 0.25, 0.28), textured=True)
    draw_textured_box(Vector3D(-0.55, mn.y + sill, front - 0.38), Vector3D(-0.25, mn.y + sill + 0.22, front - 0.08), textures["stone"], tile=None)  # pot
    material(color=(0.78, 0.6, 0.35), textured=True)
    for lx in (0.05, 0.35):
        draw_textured_box(Vector3D(lx, mn.y + sill, front - 0.35), Vector3D(lx + 0.24, mn.y + sill + 0.12, front - 0.12), textures["cloth"], tile=None)  # loaves
    # the flue on top
    material(color=(1.0, 1.0, 1.0), textured=True)
    draw_textured_box(Vector3D(-0.25, top, hz - 0.6), Vector3D(0.25, top + 0.7, hz - 0.1), textures["stone"], tile=1.0)
    glPopMatrix()
    material(color=(1.0, 1.0, 1.0), textured=True)


def draw_sign(prop, textures, material: Material, time: float) -> None:
    """A shop sign beside a door: a board carrying the trade's device (``prop.texture``:
    sign_smith, sign_store, ...) on its two faces, plain wood on the narrow edges."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2, (mn.z + mx.z) / 2
    hx = (mx.x - mn.x) / 2
    glPushMatrix()
    glTranslatef(cx, 0.0, cz)
    glRotatef(-prop.facing, 0.0, 1.0, 0.0)          # the board's face looks along -z
    material(textured=True)
    draw_board(Vector3D(-hx, mn.y, -0.04), Vector3D(hx, mx.y, 0.04), textures[prop.texture], textures["wood"])
    glPopMatrix()


BED_FRAME_TOP = 0.30        # the rails the mattress rests on
BED_MATTRESS_TOP = 0.48
BED_BLANKET = 0.035         # how thick the blanket lies on the mattress (and how far it hangs over its sides)
SLEEPER_ABOVE_TOP = 0.06    # how far below the prop's top a sleeper lies (on the blanket, not floating over it)


def bed_local_extent(prop):
    """(length along the head direction, width across it) of a bed prop turned to ``prop.facing``."""
    sx, sz = prop.max_point.x - prop.min_point.x, prop.max_point.z - prop.min_point.z
    along_z = int(round(prop.facing / 90.0)) % 2 == 0          # facing 0 / 180: the head is at -z / +z
    return (sz, sx) if along_z else (sx, sz)


def draw_bed(prop, textures, material: Material, time: float) -> None:
    """A bed filling the prop's footprint, its head towards ``prop.facing`` (0 = -z, 90 = +x ...):
    a wooden frame on four legs with a headboard and a lower footboard, a linen mattress, a
    checked blanket over the foot three quarters of it - hanging a little over the sides, whose
    faces are plain cloth, not the picture on top - and a pillow at the head."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0
    length, width = bed_local_extent(prop)
    hl, hw, base = length / 2.0, width / 2.0, mn.y
    glPushMatrix()
    glTranslatef(cx, base, cz)
    glRotatef(-prop.facing, 0.0, 1.0, 0.0)          # local -z is the head end
    wood, cloth = textures["wood"], textures["cloth"]
    material(color=(0.78, 0.62, 0.46), textured=True)
    leg = 0.07
    for lx in (-hw, hw - leg):
        for lz in (-hl, hl - leg):
            draw_textured_box(Vector3D(lx, 0.0, lz), Vector3D(lx + leg, BED_FRAME_TOP, lz + leg), wood, tile=None)
    draw_textured_box(Vector3D(-hw, BED_FRAME_TOP - 0.1, -hl + leg), Vector3D(-hw + leg, BED_FRAME_TOP, hl - leg), wood, tile=None)
    draw_textured_box(Vector3D(hw - leg, BED_FRAME_TOP - 0.1, -hl + leg), Vector3D(hw, BED_FRAME_TOP, hl - leg), wood, tile=None)
    proud = 0.012                                                   # (the boards stand clear of the legs' faces)
    draw_textured_box(Vector3D(-hw - proud, 0.02, -hl - 0.02), Vector3D(hw + proud, 0.95, -hl + 0.05), wood, tile=None)   # headboard
    draw_textured_box(Vector3D(-hw - proud, 0.02, hl - 0.05), Vector3D(hw + proud, 0.66, hl + 0.02), wood, tile=None)     # footboard
    inset = 0.05
    material(color=(0.93, 0.9, 0.82), textured=True)                             # the mattress (linen ticking)
    draw_textured_box(Vector3D(-hw + inset, BED_FRAME_TOP - 0.1 + 0.011, -hl + 0.05), Vector3D(hw - inset, BED_MATTRESS_TOP, hl - 0.05),
                      cloth, tile=None)
    # the blanket: a checked top (the bed picture's red part only - never its pillow) and plain red sides
    over = BED_BLANKET
    bx1, bx2 = -hw + inset - over, hw - inset + over
    bz1, bz2 = -hl + 0.05 + (length - 0.1) * 0.28, hl - 0.05 + over
    top, low = BED_MATTRESS_TOP + BED_BLANKET, BED_MATTRESS_TOP - 0.13
    material(color=(0.62, 0.2, 0.2), textured=True)
    draw_textured_box(Vector3D(bx1, low, bz1), Vector3D(bx2, top - 0.006, bz2), cloth, tile=None)
    material(color=(1.0, 1.0, 1.0), textured=True)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textures["bed"])
    glBegin(GL_QUADS)
    glNormal3f(0.0, 1.0, 0.0)
    for (u, v), (x, z) in (((0.0, 0.0), (bx1, bz2)), ((1.0, 0.0), (bx2, bz2)), ((1.0, 0.7), (bx2, bz1)), ((0.0, 0.7), (bx1, bz1))):
        glTexCoord2f(u, v); glVertex3f(x, top, z)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    material(color=(0.97, 0.96, 0.92), textured=True)                            # the pillow
    draw_textured_box(Vector3D(-hw + inset + 0.08, BED_MATTRESS_TOP, -hl + 0.1), Vector3D(hw - inset - 0.08, BED_MATTRESS_TOP + 0.1, bz1 - 0.04),
                      cloth, tile=None)
    material(color=(1.0, 1.0, 1.0), textured=True)
    glPopMatrix()


# ----------------------------------------------------------------------------- a lived-in town
def draw_cylinder(cx: float, cz: float, radius: float, y0: float, y1: float, side_tex: int, cap_tex: int = None,
                  segments: int = 12, tile: float = 1.0) -> None:
    """An upright cylinder (a barrel's staves, a post) with a flat top cap."""
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, side_tex)
    glBegin(GL_QUADS)
    step = 2.0 * math.pi / segments
    for i in range(segments):
        a0, a1 = i * step, (i + 1) * step
        glNormal3f(math.cos((a0 + a1) / 2), 0.0, math.sin((a0 + a1) / 2))
        for a, u, y, v in ((a0, i / segments, y0, 0.0), (a0, i / segments, y1, 1.0), (a1, (i + 1) / segments, y1, 1.0),
                           (a1, (i + 1) / segments, y0, 0.0)):
            glTexCoord2f(u * tile, v)
            glVertex3f(cx + math.cos(a) * radius, y, cz + math.sin(a) * radius)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, cap_tex if cap_tex is not None else side_tex)
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0.0, 1.0, 0.0)
    glTexCoord2f(0.5, 0.5); glVertex3f(cx, y1, cz)
    for i in range(segments, -1, -1):
        a = i * step
        glTexCoord2f(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))
        glVertex3f(cx + math.cos(a) * radius, y1, cz + math.sin(a) * radius)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_log(x0: float, x1: float, y: float, z: float, radius: float, bark: int, end: int, segments: int = 8) -> None:
    """A log lying along x from x0 to x1, its axis at (y, z): bark round it, sawn rings at both ends."""
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, bark)
    glBegin(GL_QUADS)
    step = 2.0 * math.pi / segments
    for i in range(segments):
        a0, a1 = i * step, (i + 1) * step
        glNormal3f(0.0, math.sin((a0 + a1) / 2), math.cos((a0 + a1) / 2))
        for a, x, u in ((a0, x0, 0.0), (a1, x0, 0.0), (a1, x1, 1.0), (a0, x1, 1.0)):
            glTexCoord2f(u, a / (2 * math.pi))
            glVertex3f(x, y + math.sin(a) * radius, z + math.cos(a) * radius)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, end)
    for x, nx in ((x0, -1.0), (x1, 1.0)):
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(nx, 0.0, 0.0)
        glTexCoord2f(0.5, 0.5); glVertex3f(x, y, z)
        order = range(segments + 1) if nx > 0 else range(segments, -1, -1)
        for i in order:
            a = i * step
            glTexCoord2f(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))
            glVertex3f(x, y + math.sin(a) * radius, z + math.cos(a) * radius)
        glEnd()
    glDisable(GL_TEXTURE_2D)


def _frame(prop):
    """(centre x, centre z, half width along local x, half depth along local z, base y, height) of a prop
    turned by its facing (90 / 270 swap the footprint's axes)."""
    mn, mx = prop.min_point, prop.max_point
    sx, sz = (mx.x - mn.x) / 2.0, (mx.z - mn.z) / 2.0
    if int(round(getattr(prop, "facing", 0.0) / 90.0)) % 2:
        sx, sz = sz, sx
    return (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0, sx, sz, mn.y, mx.y - mn.y


def draw_barrel(prop, textures, material: Material, time: float) -> None:
    cx, cz, hx, hz, base, height = _frame(prop)
    r = min(hx, hz)
    material(textured=True)
    draw_cylinder(cx, cz, r * 0.92, base, base + height * 0.08, textures["barrel"], textures["log_end"])      # the chime
    draw_cylinder(cx, cz, r, base + height * 0.08, base + height * 0.92, textures["barrel"], textures["log_end"])
    draw_cylinder(cx, cz, r * 0.92, base + height * 0.92, base + height, textures["barrel"], textures["log_end"])


def draw_woodpile(prop, textures, material: Material, time: float) -> None:
    """Split firewood stacked against a wall: rows of logs, each row a little shorter on top."""
    cx, cz, hx, hz, base, height = _frame(prop)
    glPushMatrix()
    glTranslatef(cx, base, cz)
    glRotatef(-prop.facing, 0.0, 1.0, 0.0)
    material(textured=True)
    radius = 0.13
    rows = max(1, int(height / (radius * 1.8)))
    per_row = max(1, int(2 * hz / (radius * 2.05)))
    for row in range(rows):
        count = max(1, per_row - row // 2)
        y = radius + row * radius * 1.75
        width = count * radius * 2.05
        for i in range(count):
            z = -width / 2.0 + radius + i * radius * 2.05 + (row % 2) * radius * 0.5
            if abs(z) + radius > hz + 0.01:
                continue
            jitter = ((i * 7 + row * 3) % 5 - 2) * 0.02
            draw_log(-hx + jitter, hx - jitter, y, z, radius * (0.9 + 0.1 * ((i + row) % 3) / 2), textures["bark"], textures["log_end"])
    glPopMatrix()


def draw_oak(prop, textures, material: Material, time: float) -> None:
    """A broad-leaved tree: a stout trunk (the prop's solid box) and a crown of leafy clumps."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0
    trunk_r = (mx.x - mn.x) / 2.0
    height = mx.y - mn.y
    material(color=(0.9, 0.85, 0.8), textured=True)
    draw_cylinder(cx, cz, trunk_r, mn.y - 0.2, mx.y, textures["bark"], textures["log_end"], segments=9, tile=2.0)
    seed = int(abs(cx * 13.7 + cz * 7.3)) % 1000
    crown = height * 0.55
    material(color=(0.85 + 0.15 * (seed % 3) / 2, 0.95, 0.8), textured=True)
    for k, (ox, oy, oz, rf) in enumerate(((0.0, 0.15, 0.0, 1.0), (0.55, -0.1, 0.2, 0.7), (-0.5, -0.05, -0.3, 0.72),
                                          (0.1, 0.45, -0.45, 0.62), (-0.2, 0.2, 0.55, 0.66))):
        draw_rock(cx + ox * crown, mx.y + oy * crown, cz + oz * crown, crown * rf, textures["leaves"], seed=seed + k,
                  squash=0.85, lumpiness=0.25, slices=8, stacks=6, tile=0.6)
    material(textured=True)


def draw_pine(prop, textures, material: Material, time: float) -> None:
    """A conifer: a tall thin trunk and tiers of drooping needled cones."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0
    trunk_r = (mx.x - mn.x) / 2.0
    height = mx.y - mn.y
    material(color=(0.85, 0.8, 0.78), textured=True)
    draw_cylinder(cx, cz, trunk_r, mn.y - 0.2, mn.y + height * 1.4, textures["bark"], textures["log_end"], segments=7, tile=2.0)
    material(textured=True)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textures["pine"])
    tiers = 4
    segments = 10
    # (each tree's needles turned its own way: two neighbours' cones of a height, set the same way,
    # had facets in one plane where their crowns met - and flickered there)
    turn = (abs(cx * 12.9898 + cz * 78.233) % 1.0) * 2.0 * math.pi / segments
    for t in range(tiers):
        y0 = mn.y + height * (0.45 + t * 0.32)
        r = height * (0.62 - t * 0.12)
        apex = y0 + height * 0.55
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0.0, 1.0, 0.0)
        glTexCoord2f(0.5, 1.0); glVertex3f(cx, apex, cz)
        for i in range(segments, -1, -1):
            a = 2.0 * math.pi * i / segments + turn
            nx, nz = math.cos(a), math.sin(a)
            glNormal3f(nx * 0.7, 0.6, nz * 0.7)
            glTexCoord2f(i / segments * 2.0, 0.0)
            glVertex3f(cx + nx * r, y0 - 0.25 * (i % 2), cz + nz * r)
        glEnd()
        glBegin(GL_TRIANGLE_FAN)                       # the underside, so the tier is not hollow from below
        glNormal3f(0.0, -1.0, 0.0)
        glTexCoord2f(0.5, 0.5); glVertex3f(cx, y0 + 0.3, cz)
        for i in range(segments + 1):
            a = 2.0 * math.pi * i / segments + turn
            glTexCoord2f(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))
            glVertex3f(cx + math.cos(a) * r, y0 - 0.25 * (i % 2), cz + math.sin(a) * r)
        glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_bush(prop, textures, material: Material, time: float) -> None:
    cx, cz, hx, hz, base, height = _frame(prop)
    seed = int(abs(cx * 11.3 + cz * 5.1)) % 1000
    material(color=(0.85, 0.95, 0.8), textured=True)
    for k, (ox, oz, rf) in enumerate(((0.0, 0.0, 1.0), (0.45, 0.2, 0.7), (-0.4, -0.25, 0.75))):
        r = min(hx, hz) * rf
        draw_rock(cx + ox * hx, base + r * 0.55, cz + oz * hz, r, textures["leaves"], seed=seed + k, squash=0.75,
                  lumpiness=0.3, slices=7, stacks=5, tile=0.7)
    material(textured=True)


def draw_boulder(prop, textures, material: Material, time: float) -> None:
    """A lumpy rock filling the footprint - stretched along the longer side when the footprint is
    not square (a ledge of rock, the brow over a cave's mouth)."""
    mn, mx = prop.min_point, prop.max_point
    cx, cz = (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0
    hx, hz, base, height = (mx.x - mn.x) / 2.0, (mx.z - mn.z) / 2.0, mn.y, mx.y - mn.y
    seed = int(abs(cx * 3.1 + cz * 17.9)) % 1000
    material(textured=True)
    r = min(hx, hz)
    tex = textures[prop.texture if prop.texture in ("rock", "stone", "ore", "cave_rock", "cliff") else "rock"]
    if abs(hx - hz) < 0.05:
        draw_rock(cx, base + height * 0.35, cz, r * 1.05, tex, seed=seed, squash=max(0.4, height / (2 * r)), lumpiness=0.22,
                  slices=9, stacks=6, tile=1.5)
        return
    glPushMatrix()
    glTranslatef(cx, base + height * 0.35, cz)
    glScalef(hx / r, 1.0, hz / r)
    draw_rock(0.0, 0.0, 0.0, r * 1.05, tex, seed=seed, squash=max(0.4, height / (2 * r)), lumpiness=0.22, slices=11, stacks=6,
              tile=1.5)
    glPopMatrix()


def draw_spike(cx: float, cz: float, y_base: float, y_tip: float, radius: float, tex: int, seed: int, segments: int = 7,
               rings: int = 4, tile: float = 1.0) -> None:
    """A lumpy tapering spike of rock from ``y_base`` to a point at ``y_tip`` (upwards or down):
    rings narrowing to the tip, each pushed about a little, so it looks grown, not turned."""
    rng = random.Random(seed)
    down = y_tip < y_base
    lean_x, lean_z = rng.uniform(-0.25, 0.25) * radius, rng.uniform(-0.25, 0.25) * radius
    ring_pts = []
    for k in range(rings + 1):
        t = k / rings
        r = radius * (1.0 - t) ** 1.3 * (1.0 + 0.18 * rng.uniform(-1.0, 1.0)) if k < rings else 0.0
        y = y_base + (y_tip - y_base) * t
        ox, oz = lean_x * t, lean_z * t
        ring = []
        for i in range(segments + 1):
            a = 2.0 * math.pi * i / segments
            wobble = 1.0 + (0.2 * rng.uniform(-1.0, 1.0) if i < segments and k < rings else 0.0)
            if i == segments:
                wobble = ring[0][3]
            ring.append((cx + ox + math.cos(a) * r * wobble, y, cz + oz + math.sin(a) * r * wobble, wobble, a))
        ring_pts.append(ring)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glBegin(GL_QUADS)
    slant = 0.45 if not down else -0.45
    for k in range(rings):
        for i in range(segments):
            quad = (ring_pts[k][i], ring_pts[k][i + 1], ring_pts[k + 1][i + 1], ring_pts[k + 1][i])
            if down:
                quad = quad[::-1]
            for x, y, z, _, a in quad:
                glNormal3f(math.cos(a), slant, math.sin(a))
                glTexCoord2f(a * radius / tile, (y - y_base) / tile)
                glVertex3f(x, y, z)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_stalagmite(prop, textures, material: Material, time: float) -> None:
    """A column of rock grown up from a cave floor (drips from the roof over ages)."""
    cx, cz, hx, hz, base, height = _frame(prop)
    seed = int(abs(cx * 31.7 + cz * 13.1)) % 10000
    material(color=(0.92, 0.9, 0.86), textured=True)
    tex = textures[prop.texture] if prop.texture in ("cave_rock", "rock", "cliff") else textures["cave_rock"]
    draw_spike(cx, cz, base - 0.1, base + height, min(hx, hz), tex, seed)
    material(textured=True)


def draw_stalactite(prop, textures, material: Material, time: float) -> None:
    """A spike of rock hanging from a cave's roof."""
    cx, cz, hx, hz, base, height = _frame(prop)
    seed = int(abs(cx * 17.3 + cz * 29.9)) % 10000
    material(color=(0.9, 0.9, 0.88), textured=True)
    tex = textures[prop.texture] if prop.texture in ("cave_rock", "rock", "cliff") else textures["cave_rock"]
    draw_spike(cx, cz, base + height + 0.05, base, min(hx, hz), tex, seed, segments=6, rings=3)
    material(textured=True)


def draw_mushroom(prop, textures, material: Material, time: float) -> None:
    """A clump of pale cave mushrooms, their caps aglow with a cold light of their own."""
    cx, cz, hx, hz, base, height = _frame(prop)
    rng = random.Random(int(abs(cx * 7.7 + cz * 3.3)) % 10000)
    for k in range(rng.randint(4, 7)):
        a = rng.uniform(0.0, 2.0 * math.pi)
        d = rng.uniform(0.0, 0.8) * min(hx, hz)
        x, z = cx + math.cos(a) * d, cz + math.sin(a) * d
        stem_h = height * rng.uniform(0.35, 0.95)
        cap_r = rng.uniform(0.07, 0.16) * (1.0 + stem_h)
        material(color=(0.85, 0.84, 0.78), textured=True)
        draw_cylinder(x, z, 0.025 + cap_r * 0.18, base - 0.05, base + stem_h, textures["wax"], textures["wax"], segments=5)
        material(color=(0.75, 1.0, 0.9), emission=(0.22, 0.75, 0.55), textured=True, fog=0.5)
        draw_rock(x, base + stem_h, z, cap_r, textures["mushroom"], seed=k + int(cx * 10), squash=0.45, lumpiness=0.12,
                  slices=7, stacks=4, tile=0.4)
    material(textured=True)


def draw_haystack(prop, textures, material: Material, time: float) -> None:
    cx, cz, hx, hz, base, height = _frame(prop)
    material(textured=True)
    draw_rock(cx, base + height * 0.35, cz, min(hx, hz) * 1.05, textures["hay"], seed=int(abs(cx + cz * 3)) % 100,
              squash=height / (2.2 * min(hx, hz)), lumpiness=0.12, slices=10, stacks=6, tile=0.6)


def draw_stall(prop, textures, material: Material, time: float) -> None:
    """A market stall: a counter, four posts and a striped awning, goods laid out on the board."""
    cx, cz, hx, hz, base, height = _frame(prop)
    glPushMatrix()
    glTranslatef(cx, base, cz)
    glRotatef(-prop.facing, 0.0, 1.0, 0.0)              # the customer side faces local -z
    wood = textures["wood"]
    material(textured=True)
    for px in (-hx + 0.05, hx - 0.15):
        for pz in (-hz + 0.05, hz - 0.15):
            draw_textured_box(Vector3D(px, 0.0, pz), Vector3D(px + 0.1, height, pz + 0.1), wood, tile=None)
    draw_textured_box(Vector3D(-hx + 0.15, 0.0, -hz + 0.2), Vector3D(hx - 0.15, 0.95, -hz + 0.75), wood, tile=1.0)   # counter
    seed = int(abs(cx * 5 + cz)) % 7
    tint = ((0.75, 0.2, 0.18), (0.2, 0.35, 0.7), (0.2, 0.55, 0.3), (0.7, 0.55, 0.2))[seed % 4]
    stripes = 6
    for i in range(stripes):                            # the awning, sloping down to the front
        x1 = -hx - 0.1 + (2 * hx + 0.2) * i / stripes
        x2 = -hx - 0.1 + (2 * hx + 0.2) * (i + 1) / stripes
        material(color=tint if i % 2 == 0 else (0.92, 0.9, 0.85), textured=True)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, textures["cloth"])
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.9, -0.43)
        glTexCoord2f(0, 0); glVertex3f(x1, height - 0.45, -hz - 0.35)
        glTexCoord2f(1, 0); glVertex3f(x2, height - 0.45, -hz - 0.35)
        glTexCoord2f(1, 1); glVertex3f(x2, height + 0.05, hz + 0.1)
        glTexCoord2f(0, 1); glVertex3f(x1, height + 0.05, hz + 0.1)
        glEnd()
        glDisable(GL_TEXTURE_2D)
    goods = ((0.85, 0.2, 0.15), (0.95, 0.75, 0.2), (0.45, 0.65, 0.25), (0.6, 0.4, 0.25))
    for i in range(5):                                  # wares on the counter
        material(color=goods[(i + seed) % 4], textured=True)
        x = -hx + 0.35 + i * (2 * hx - 0.7) / 4.0
        draw_textured_box(Vector3D(x - 0.1, 0.95, -hz + 0.3), Vector3D(x + 0.1, 1.1 + 0.04 * (i % 2), -hz + 0.55), textures["cloth"], tile=None)
    material(textured=True)
    glPopMatrix()


def draw_bridge(prop, textures, material: Material, time: float) -> None:
    """A plank bridge over the stream: the deck (the prop's walkable top), a beam under each edge
    and a rail along each side on posts - the sides over the water, parallel to the way across
    (``facing`` 90/270: the span, and the road over it, run along x; 0/180: along z)."""
    mn, mx = prop.min_point, prop.max_point
    along_x = int(round(prop.facing / 90.0)) % 2 == 1      # facing 90/270: the span runs along x
    wood = textures["wood"]
    material(color=(0.85, 0.75, 0.62), textured=True)
    draw_textured_box(Vector3D(mn.x, mx.y - 0.16, mn.z), Vector3D(mx.x, mx.y, mx.z), textures["building"], tile=1.0)   # deck
    rail_h = 1.0
    if along_x:
        for z in (mn.z, mx.z - 0.12):
            draw_textured_box(Vector3D(mn.x - 0.1, mx.y - 0.5, z), Vector3D(mx.x + 0.1, mx.y - 0.16, z + 0.12), wood, tile=1.0)
            draw_textured_box(Vector3D(mn.x, mx.y + rail_h - 0.1, z + 0.02), Vector3D(mx.x, mx.y + rail_h, z + 0.1), wood, tile=1.0)
            for i in range(4):
                x = mn.x + (mx.x - mn.x - 0.12) * i / 3.0
                draw_textured_box(Vector3D(x, mx.y - 0.12, z), Vector3D(x + 0.12, mx.y + rail_h - 0.1, z + 0.12), wood, tile=None)
    else:
        for x in (mn.x, mx.x - 0.12):
            draw_textured_box(Vector3D(x, mx.y - 0.5, mn.z - 0.1), Vector3D(x + 0.12, mx.y - 0.16, mx.z + 0.1), wood, tile=1.0)
            draw_textured_box(Vector3D(x + 0.02, mx.y + rail_h - 0.1, mn.z), Vector3D(x + 0.1, mx.y + rail_h, mx.z), wood, tile=1.0)
            for i in range(4):
                z = mn.z + (mx.z - mn.z - 0.12) * i / 3.0
                draw_textured_box(Vector3D(x, mx.y - 0.12, z), Vector3D(x + 0.12, mx.y + rail_h - 0.1, z + 0.12), wood, tile=None)
    material(textured=True)


def draw_fence(prop, textures, material: Material, time: float) -> None:
    """A post-and-rail fence along the prop's long side."""
    mn, mx = prop.min_point, prop.max_point
    along_x = (mx.x - mn.x) >= (mx.z - mn.z)
    wood = textures["wood"]
    material(color=(0.8, 0.72, 0.6), textured=True)
    length = (mx.x - mn.x) if along_x else (mx.z - mn.z)
    posts = max(2, int(length / 2.0) + 1)
    for i in range(posts):
        t = (length - 0.12) * i / (posts - 1)
        if along_x:
            draw_textured_box(Vector3D(mn.x + t, mn.y, mn.z), Vector3D(mn.x + t + 0.12, mx.y, mx.z), wood, tile=None)
        else:
            draw_textured_box(Vector3D(mn.x, mn.y, mn.z + t), Vector3D(mx.x, mx.y, mn.z + t + 0.12), wood, tile=None)
    for y in (0.35, 0.8):
        y0 = mn.y + (mx.y - mn.y) * y
        if along_x:
            draw_textured_box(Vector3D(mn.x, y0, mn.z + 0.02), Vector3D(mx.x, y0 + 0.08, mx.z - 0.02), wood, tile=1.0)
        else:
            draw_textured_box(Vector3D(mn.x + 0.02, y0, mn.z), Vector3D(mx.x - 0.02, y0 + 0.08, mx.z), wood, tile=1.0)
    material(textured=True)


def _decal_quad(corners, texture_id, normal) -> None:
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(-1.0, -4.0)                      # a decal always wins against the surface it lies on
    glDepthMask(False)
    glBegin(GL_QUADS)
    glNormal3f(*normal)
    for (u, v), p in zip(((0, 0), (1, 0), (1, 1), (0, 1)), corners):
        glTexCoord2f(u, v); glVertex3f(*p)
    glEnd()
    glDepthMask(True)
    glDisable(GL_POLYGON_OFFSET_FILL)
    glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)


def draw_crack(prop, textures, material: Material, time: float) -> None:
    """Cracks in the paving (a decal a hair above the ground the prop's footprint covers)."""
    mn, mx = prop.min_point, prop.max_point
    y = mn.y + 0.06
    material(textured=True)
    _decal_quad(((mn.x, y, mx.z), (mx.x, y, mx.z), (mx.x, y, mn.z), (mn.x, y, mn.z)), textures["crack_decal"], (0.0, 1.0, 0.0))


def draw_stain(prop, textures, material: Material, time: float) -> None:
    """Water damage down a wall: a decal on the face of the wall the prop's thin box lies against;
    ``facing`` is the way that face looks (0 = -z, 90 = +x, 180 = +z, 270 = -x)."""
    mn, mx = prop.min_point, prop.max_point
    face = int(round(prop.facing / 90.0)) % 4
    material(textured=True)
    if face == 0:
        z = mn.z
        corners = ((mx.x, mn.y, z), (mn.x, mn.y, z), (mn.x, mx.y, z), (mx.x, mx.y, z))
        normal = (0.0, 0.0, -1.0)
    elif face == 2:
        z = mx.z
        corners = ((mn.x, mn.y, z), (mx.x, mn.y, z), (mx.x, mx.y, z), (mn.x, mx.y, z))
        normal = (0.0, 0.0, 1.0)
    elif face == 1:
        x = mx.x
        corners = ((x, mn.y, mx.z), (x, mn.y, mn.z), (x, mx.y, mn.z), (x, mx.y, mx.z))
        normal = (1.0, 0.0, 0.0)
    else:
        x = mn.x
        corners = ((x, mn.y, mn.z), (x, mn.y, mx.z), (x, mx.y, mx.z), (x, mx.y, mn.z))
        normal = (-1.0, 0.0, 0.0)
    # the stain texture seeps from the top: v = 1 at the top of the quad
    _decal_quad(corners, textures["stain_decal"], normal)


def draw_grate(prop, textures, material: Material, time: float) -> None:
    """An iron grate across a culvert (see-through bars; the prop's box is the solid barrier)."""
    mn, mx = prop.min_point, prop.max_point
    along_x = (mx.x - mn.x) >= (mx.z - mn.z)
    material(color=(0.8, 0.8, 0.85), textured=True)
    cz = (mn.z + mx.z) / 2.0
    cx = (mn.x + mx.x) / 2.0
    if along_x:
        corners = ((mn.x, mn.y, cz), (mx.x, mn.y, cz), (mx.x, mx.y, cz), (mn.x, mx.y, cz))
        normal = (0.0, 0.0, 1.0)
    else:
        corners = ((cx, mn.y, mn.z), (cx, mn.y, mx.z), (cx, mx.y, mx.z), (cx, mx.y, mn.z))
        normal = (1.0, 0.0, 0.0)
    _decal_quad(corners, textures["grate"], normal)


def draw_campfire(prop, textures, material: Material, time: float) -> None:
    """A ring of stones, a few logs and flames (drawn live: the flames move)."""
    cx, cz, hx, hz, base, height = _frame(prop)
    material(textured=True)
    for i in range(8):
        a = 2 * math.pi * i / 8
        draw_textured_box(Vector3D(cx + math.cos(a) * 0.55 - 0.1, base, cz + math.sin(a) * 0.55 - 0.1),
                          Vector3D(cx + math.cos(a) * 0.55 + 0.1, base + 0.16, cz + math.sin(a) * 0.55 + 0.1), textures["stone"], tile=None)
    glPushMatrix()
    glTranslatef(cx, base, cz)
    for rot in (20.0, 80.0, 140.0):
        glPushMatrix()
        glRotatef(rot, 0.0, 1.0, 0.0)
        draw_log(-0.4, 0.4, 0.08, 0.0, 0.07, textures["bark"], textures["log_end"], segments=6)
        glPopMatrix()
    glPopMatrix()
    material(color=(1.0, 0.78, 0.5), emission=(0.9, 0.55, 0.2), textured=True, fog=0.25)
    for i, (fx, fz) in enumerate(((0.0, 0.0), (0.15, 0.08), (-0.12, 0.1), (0.05, -0.14))):
        k = flame_scale(time, i * 1.7)
        fh = (0.55 - 0.1 * i) * k
        draw_textured_box(Vector3D(cx + fx - 0.08, base + 0.1, cz + fz - 0.08), Vector3D(cx + fx + 0.08, base + 0.1 + fh, cz + fz + 0.08),
                          textures["flame"], tile=None)
    material(textured=True)


def draw_chest(container, textures, material: Material, time: float) -> None:
    """A wooden chest bound in iron (a strongbox for a ``lockbox`` kind: iron all over)."""
    mn, mx = container.min_point, container.max_point
    iron_box = getattr(container, "kind", "chest") == "strongbox"
    body = textures["iron"] if iron_box else textures["wood"]
    material(color=(0.75, 0.75, 0.8) if iron_box else (1.0, 0.92, 0.82), textured=True)
    lid = mn.y + (mx.y - mn.y) * 0.68
    draw_textured_box(Vector3D(mn.x, mn.y, mn.z), Vector3D(mx.x, lid, mx.z), body, tile=None)
    draw_textured_box(Vector3D(mn.x - 0.02, lid, mn.z - 0.02), Vector3D(mx.x + 0.02, mx.y, mx.z + 0.02), body, tile=None)
    material(color=(0.35, 0.35, 0.4), textured=True)
    along_x = (mx.x - mn.x) >= (mx.z - mn.z)
    for t in (0.18, 0.82):                                  # iron bands over the lid
        if along_x:
            x = mn.x + (mx.x - mn.x) * t
            draw_textured_box(Vector3D(x - 0.04, mn.y - 0.004, mn.z - 0.034), Vector3D(x + 0.04, mx.y + 0.014, mx.z + 0.034), textures["iron"], tile=None)
        else:
            z = mn.z + (mx.z - mn.z) * t
            draw_textured_box(Vector3D(mn.x - 0.034, mn.y - 0.004, z - 0.04), Vector3D(mx.x + 0.034, mx.y + 0.014, z + 0.04), textures["iron"], tile=None)
    material(color=(1.0, 0.85, 0.3), textured=True)          # the lock plate
    cx, cz = (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0
    if along_x:
        draw_textured_box(Vector3D(cx - 0.07, lid - 0.12, mn.z - 0.05), Vector3D(cx + 0.07, lid + 0.02, mn.z - 0.02), textures["iron"], tile=None)
    else:
        draw_textured_box(Vector3D(mn.x - 0.05, lid - 0.12, cz - 0.07), Vector3D(mn.x - 0.02, lid + 0.02, cz + 0.07), textures["iron"], tile=None)
    material(textured=True)


def draw_pedestal(container, textures, material: Material, time: float) -> None:
    """A treasure on show: a carved stone plinth, and on it the piece itself, shining (live: it pulses)."""
    mn, mx = container.min_point, container.max_point
    cx, cz = (mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0
    hx, hz = (mx.x - mn.x) / 2.0, (mx.z - mn.z) / 2.0
    top = mx.y
    material(textured=True)
    draw_textured_box(Vector3D(mn.x, mn.y, mn.z), Vector3D(mx.x, mn.y + 0.18, mx.z), textures["marble"], tile=None)
    draw_textured_box(Vector3D(cx - hx * 0.6, mn.y + 0.18, cz - hz * 0.6), Vector3D(cx + hx * 0.6, top - 0.12, cz + hz * 0.6),
                      textures["marble"], tile=None)
    draw_textured_box(Vector3D(mn.x + 0.02, top - 0.12, mn.z + 0.02), Vector3D(mx.x - 0.02, top, mx.z - 0.02), textures["marble"], tile=None)
    pulse = 0.75 + 0.25 * math.sin(time * 2.2)
    glow = getattr(container, "glow", (0.5, 0.3, 1.0))
    material(color=glow, emission=(glow[0] * pulse, glow[1] * pulse, glow[2] * pulse), textured=True, fog=0.2)
    bob = 0.04 * math.sin(time * 1.6)
    draw_textured_box(Vector3D(cx - 0.12, top + 0.08 + bob, cz - 0.12), Vector3D(cx + 0.12, top + 0.32 + bob, cz + 0.12),
                      textures["iron"], tile=None)
    material(textured=True)


DRAWERS = {"stalagmite": draw_stalagmite, "stalactite": draw_stalactite, "mushroom": draw_mushroom,
           "well": draw_well, "forge": draw_forge, "oven": draw_oven, "sign": draw_sign, "ore": draw_ore_deposit,
           "bed": draw_bed, "barrel": draw_barrel, "woodpile": draw_woodpile, "oak": draw_oak, "pine": draw_pine,
           "bush": draw_bush, "boulder": draw_boulder, "haystack": draw_haystack, "stall": draw_stall, "bridge": draw_bridge,
           "fence": draw_fence, "crack": draw_crack, "stain": draw_stain, "grate": draw_grate, "campfire": draw_campfire}
CONTAINER_DRAWERS = {"chest": draw_chest, "strongbox": draw_chest, "pedestal": draw_pedestal}


__all__: Sequence[str] = ("DRAWERS", "draw_well", "draw_forge", "draw_oven", "draw_sign", "draw_ore_deposit", "draw_rock",
                          "draw_ring", "draw_disc", "draw_gable_roof")
