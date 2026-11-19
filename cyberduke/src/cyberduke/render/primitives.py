"""Unit primitives. Every mesh in the game is built from these two.

After ``compile_primitives()`` has run (once a GL context exists) the vertex streams are
replayed from display lists: a cube costs one ``glCallList`` instead of ~54 immediate-mode
calls. Without it the functions fall back to immediate mode, so tests and tools that
record GL calls still see the full geometry.

Tiled textures: ``uv_repeat`` gives how many times the texture repeats across the shape -
(along x, along y, along z) for a cube, (around, along, across the caps) for a cylinder -
so a 20 m wall shows its panels at their real size instead of one stretched image. Each
distinct repeat (rounded to 1/100) is compiled once and replayed like the plain shapes.
"""

from __future__ import annotations

import math

from OpenGL.GL import *

from cyberduke.render.display_lists import compile_list, is_compiling

Repeat = tuple[float, float, float]

_cube_list: int | None = None
_cylinder_lists: dict[int, int] = {}  # slices -> unit cylinder (r=1, h=1) list
_cone_lists: dict[tuple[int, float], int] = {}  # (slices, r_top/r_bot) -> unit cone list
_tiled_cube_lists: dict[Repeat, int] = {}
_tiled_cylinder_lists: dict[tuple, int] = {}
_MAX_CONE_LISTS = 64
_MAX_TILED_LISTS = 1024


def compile_primitives(cylinder_slices: tuple[int, ...] = (8, 10)) -> None:
    """Compiles the unit cube and unit cylinders. Call once after creating the context."""
    global _cube_list
    if _cube_list is None:
        _cube_list = compile_list(_emit_cube)
    for slices in cylinder_slices:
        if slices not in _cylinder_lists:
            _cylinder_lists[slices] = compile_list(
                lambda s=slices: _emit_cylinder(1.0, 1.0, 1.0, s)
            )


def reset_primitives() -> None:
    """Forgets compiled lists (e.g. after the GL context is destroyed)."""
    global _cube_list
    _cube_list = None
    _cylinder_lists.clear()
    _cone_lists.clear()
    _tiled_cube_lists.clear()
    _tiled_cylinder_lists.clear()


def _bind(tex_id: int | None) -> None:
    if tex_id:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
    else:
        glDisable(GL_TEXTURE_2D)


def _round_repeat(rep: Repeat) -> Repeat:
    return tuple(round(max(r, 0.01), 2) for r in rep)  # type: ignore[return-value]


def _cached(cache: dict, key, emit) -> int | None:
    """The list for ``key``, compiling it on first use (never while another list is being
    compiled, and never beyond the cache limit - the caller then draws immediately)."""
    found = cache.get(key)
    if found is None and len(cache) < _MAX_TILED_LISTS and not is_compiling():
        found = cache[key] = compile_list(emit)
    return found


def draw_cube(tex_id: int | None = None, uv_repeat: Repeat | None = None) -> None:
    """Unit cube centred on the origin (extent -0.5..0.5 on every axis)."""
    _bind(tex_id)
    if uv_repeat is not None and tex_id:
        rep = _round_repeat(uv_repeat)
        unit = _cached(_tiled_cube_lists, rep, lambda r=rep: _emit_cube(r))
        if unit is not None:
            glCallList(unit)
        else:
            _emit_cube(rep)
    elif _cube_list is not None:
        glCallList(_cube_list)
    else:
        _emit_cube()
    glDisable(GL_TEXTURE_2D)


# (normal, 4 corners, which axes the face's u and v run along) - uv (0,0) (1,0) (1,1) (0,1)
_N, _P = -0.5, 0.5
_CUBE_FACES = (
    ((0.0, 0.0, 1.0), ((_N, _N, _P), (_P, _N, _P), (_P, _P, _P), (_N, _P, _P)), (0, 1)),
    ((0.0, 0.0, -1.0), ((_P, _N, _N), (_N, _N, _N), (_N, _P, _N), (_P, _P, _N)), (0, 1)),
    ((0.0, 1.0, 0.0), ((_N, _P, _N), (_P, _P, _N), (_P, _P, _P), (_N, _P, _P)), (0, 2)),
    ((0.0, -1.0, 0.0), ((_N, _N, _P), (_P, _N, _P), (_P, _N, _N), (_N, _N, _N)), (0, 2)),
    ((1.0, 0.0, 0.0), ((_P, _N, _P), (_P, _N, _N), (_P, _P, _N), (_P, _P, _P)), (2, 1)),
    ((-1.0, 0.0, 0.0), ((_N, _N, _N), (_N, _N, _P), (_N, _P, _P), (_N, _P, _N)), (2, 1)),
)
_QUAD_UV = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def _emit_cube(rep: Repeat = (1.0, 1.0, 1.0)) -> None:
    glBegin(GL_QUADS)
    for normal, corners, (ua, va) in _CUBE_FACES:
        glNormal3f(*normal)
        su, sv = rep[ua], rep[va]
        for (u, v), corner in zip(_QUAD_UV, corners, strict=True):
            glTexCoord2f(u * su, v * sv)
            glVertex3f(*corner)
    glEnd()


def draw_box(
    cx: float, cy: float, cz: float, sx: float, sy: float, sz: float, tex_id: int | None = None
) -> None:
    """Axis-aligned box centred at (cx, cy, cz) with full extents (sx, sy, sz)."""
    glPushMatrix()
    glTranslatef(cx, cy, cz)
    glScalef(sx, sy, sz)
    draw_cube(tex_id)
    glPopMatrix()


def draw_cylinder(
    r_bot: float,
    r_top: float,
    h: float,
    slices: int = 10,
    tex_id: int | None = None,
    uv_repeat: Repeat | None = None,
) -> None:
    """Vertical cylinder or truncated cone from y=0 to y=h, centred on the Y axis."""
    _bind(tex_id)
    radius = max(r_bot, r_top)
    rep = _round_repeat(uv_repeat) if uv_repeat is not None and tex_id else None
    unit = None
    if radius > 0.0:
        ratio_bot, ratio_top = round(r_bot / radius, 3), round(r_top / radius, 3)
        if rep is not None:
            key = (slices, ratio_bot, ratio_top, rep)
            unit = _cached(
                _tiled_cylinder_lists,
                key,
                lambda k=key: _emit_cylinder(k[1], k[2], 1.0, k[0], k[3]),
            )
        elif r_bot == r_top:
            unit = _cylinder_lists.get(slices)
        else:
            # Truncated cones (mushroom caps, by the hundred) are compiled on first use per
            # (slices, taper) and then scaled like cylinders.
            key = (slices, ratio_bot, ratio_top)
            unit = _cone_lists.get(key)
            if unit is None and len(_cone_lists) < _MAX_CONE_LISTS and not is_compiling():
                unit = _cone_lists[key] = compile_list(
                    lambda k=key: _emit_cylinder(k[1], k[2], 1.0, k[0])
                )
    if unit is not None:
        # Non-uniform scale of the unit shape; GL_NORMALIZE (enabled in setup_lighting)
        # re-normalises the scaled normals.
        glPushMatrix()
        glScalef(radius, h, radius)
        glCallList(unit)
        glPopMatrix()
    else:
        _emit_cylinder(r_bot, r_top, h, slices, rep or (1.0, 1.0, 1.0))
    glDisable(GL_TEXTURE_2D)


def _emit_cylinder(
    r_bot: float, r_top: float, h: float, slices: int, rep: Repeat = (1.0, 1.0, 1.0)
) -> None:
    around, along, cap = rep
    glBegin(GL_QUADS)
    for i in range(slices):
        u1 = i / slices
        u2 = (i + 1) / slices
        th1 = 2.0 * math.pi * u1
        th2 = 2.0 * math.pi * u2
        c1, s1 = math.cos(th1), math.sin(th1)
        c2, s2 = math.cos(th2), math.sin(th2)
        glNormal3f(c1, 0.0, s1)
        glTexCoord2f(u1 * around, 0.0)
        glVertex3f(r_bot * c1, 0.0, r_bot * s1)
        glNormal3f(c2, 0.0, s2)
        glTexCoord2f(u2 * around, 0.0)
        glVertex3f(r_bot * c2, 0.0, r_bot * s2)
        glTexCoord2f(u2 * around, along)
        glVertex3f(r_top * c2, h, r_top * s2)
        glNormal3f(c1, 0.0, s1)
        glTexCoord2f(u1 * around, along)
        glVertex3f(r_top * c1, h, r_top * s1)
    glEnd()

    for y, r, ny, sign in ((0.0, r_bot, -1.0, -1.0), (h, r_top, 1.0, 1.0)):
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0.0, ny, 0.0)
        glTexCoord2f(0.5, 0.5)
        glVertex3f(0.0, y, 0.0)
        for i in range(slices + 1):
            th = sign * 2.0 * math.pi * i / slices
            c, s = math.cos(th), math.sin(th)
            glTexCoord2f(0.5 + 0.5 * c * cap, 0.5 + 0.5 * s * cap)
            glVertex3f(r * c, y, r * s)
        glEnd()
