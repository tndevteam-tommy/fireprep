"""Fixed-function OpenGL 1.x backend: the original look, with display lists for static geometry."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from OpenGL.GL import *

from cyberduke.render.context import NO_EMISSION, RGBA, DrawContext, Vec3
from cyberduke.render.display_lists import ListCache
from cyberduke.render.lighting import AMBIENT_BASE, position_lights, setup_lighting
from cyberduke.render.matrices import apply_look_at, apply_perspective
from cyberduke.render.primitives import compile_primitives, draw_cube, draw_cylinder
from cyberduke.world.lights import Light


def _rotation3(angle_deg: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    t = 1.0 - c
    return np.array(
        (
            (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
        )
    )


class LegacyContext(DrawContext):
    name = "legacy"
    batches = True  # people are drawn as one batch each (``render.batch``)

    def __init__(
        self,
        width: int,
        height: int,
        fov_y: float,
        near: float,
        far: float,
        lights: Sequence[Light],
    ):
        self.width, self.height = width, height
        self.fov_y, self.near, self.far = fov_y, near, far
        glViewport(0, 0, width, height)
        self._apply_projection()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._lights: list[Light] = list(lights)
        self._ambient: tuple[float, ...] = tuple(AMBIENT_BASE[:3])
        setup_lighting(self._lights, self._ambient)
        compile_primitives()
        self._cache = ListCache()
        self._alpha = 1.0
        # The linear part of the model transform, mirrored here (GL keeps the real one):
        # tiled textures need each primitive's world size to know how often to repeat.
        self._lin = np.identity(3)
        self._lin_stack: list[np.ndarray] = []

    def _apply_projection(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        apply_perspective(self.fov_y, self.width / self.height, self.near, self.far)
        glMatrixMode(GL_MODELVIEW)

    # ------------------------------------------------------------------ quality
    def set_view_distance(self, far: float) -> None:
        if far != self.far:
            self.far = far
            self._apply_projection()

    def set_lights(self, lights: Sequence[Light]) -> None:
        self._lights = list(lights)
        setup_lighting(self._lights, self._ambient)  # keeps the preset's ambient level

    def set_sky(self, rgb) -> None:
        glClearColor(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

    def set_global_ambient(self, rgb) -> None:
        self._ambient = tuple(rgb)[:3]
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (*self._ambient, 1.0))

    # ------------------------------------------------------------------ frame
    def begin_frame(self, eye: Vec3, target: Vec3) -> None:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self._lin, self._lin_stack = np.identity(3), []
        apply_look_at(eye, target)
        # Light positions are stored in eye space at the time they are set: give them the
        # world positions again under this frame's view, or they would follow the camera.
        position_lights(self._lights)

    def end_frame(self) -> None:
        pass

    # ------------------------------------------------------------------ matrix stack
    def push(self) -> None:
        glPushMatrix()
        self._lin_stack.append(self._lin)

    def pop(self) -> None:
        glPopMatrix()
        self._lin = self._lin_stack.pop()

    def translate(self, x: float, y: float, z: float) -> None:
        glTranslatef(x, y, z)

    def rotate(self, angle_deg: float, x: float, y: float, z: float) -> None:
        glRotatef(angle_deg, x, y, z)
        self._lin = self._lin @ _rotation3(angle_deg, x, y, z)

    def scale(self, x: float, y: float, z: float) -> None:
        glScalef(x, y, z)
        self._lin = self._lin * np.array((x, y, z))  # scales the columns

    def _axis_lengths(self) -> np.ndarray:
        return np.sqrt((self._lin * self._lin).sum(axis=0))

    # ------------------------------------------------------------------ material
    def color(self, r: float, g: float, b: float) -> None:
        glColor4f(r, g, b, self._alpha)

    def alpha(self, a: float) -> None:
        self._alpha = a

    def depth_mask(self, enabled: bool) -> None:
        glDepthMask(GL_TRUE if enabled else GL_FALSE)

    def emission(self, rgb: Sequence[float] | None) -> None:
        e = NO_EMISSION if rgb is None else tuple(rgb[:3])
        glMaterialfv(GL_FRONT, GL_EMISSION, (*e, 1.0))

    def specular(self, rgb: Sequence[float] | None, shininess: float = 0.0) -> None:
        s = NO_EMISSION if rgb is None else tuple(rgb[:3])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (*s, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess if rgb is not None else 0.0)

    # ------------------------------------------------------------------ primitives
    def cube(self, tex_id: int | None = None) -> None:
        tile = self.texture_tiles.get(tex_id) if tex_id else None
        if tile:
            draw_cube(tex_id, self.cube_repeat(self._axis_lengths(), tile))
        else:
            draw_cube(tex_id)

    def cylinder(
        self, r_bot: float, r_top: float, h: float, slices: int = 10, tex_id: int | None = None
    ) -> None:
        tile = self.texture_tiles.get(tex_id) if tex_id else None
        rep = None
        if tile:
            rep = self.cylinder_repeat(r_bot, r_top, h, self._axis_lengths(), tile)
        draw_cylinder(r_bot, r_top, h, slices, tex_id, rep)

    def shape(self, name: str, tex_id: int | None = None) -> None:
        lists = self.__dict__.setdefault("_shape_lists", {})
        found = lists.get(name)
        if found is None:
            from cyberduke.render.display_lists import compile_list
            from cyberduke.render.shapes import mesh

            base, idx = mesh(name)

            def emit(base=base, idx=idx) -> None:
                glBegin(GL_TRIANGLES)
                for k in idx:
                    x, y, z, nx, ny, nz, u, v = (float(c) for c in base[k])
                    glNormal3f(nx, ny, nz)
                    glTexCoord2f(u, v)
                    glVertex3f(x, y, z)
                glEnd()

            found = lists[name] = compile_list(emit)
        if tex_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, tex_id)
        else:
            glDisable(GL_TEXTURE_2D)
        glCallList(found)
        glDisable(GL_TEXTURE_2D)

    def draw_batch(self, vertices: np.ndarray, indices: np.ndarray, ranges: Sequence[Any]) -> None:
        if not len(vertices):
            return
        v = np.ascontiguousarray(vertices, dtype=np.float32)
        pos = np.ascontiguousarray(v[:, 0:3])
        nrm = np.ascontiguousarray(v[:, 3:6])
        uv = np.ascontiguousarray(v[:, 6:8])
        alpha = np.full((len(v), 1), self._alpha, dtype=np.float32)
        col = np.ascontiguousarray(np.hstack((v[:, 8:11], alpha)))
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, pos)
        glNormalPointer(GL_FLOAT, 0, nrm)
        glTexCoordPointer(2, GL_FLOAT, 0, uv)
        glColorPointer(4, GL_FLOAT, 0, col)
        idx = np.ascontiguousarray(indices, dtype=np.uint32)
        # (fixed function has no per-vertex emission or specular: within a texture's range
        # the triangles are drawn in runs that share them)
        material = v[idx, 11:16]
        try:
            for tex, first, count in ranges:
                if tex:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, tex)
                else:
                    glDisable(GL_TEXTURE_2D)
                tri_mat = material[first : first + count : 3]
                change = np.flatnonzero(np.any(tri_mat[1:] != tri_mat[:-1], axis=1)) + 1
                starts = [0, *change.tolist(), len(tri_mat)]
                for a, b in zip(starts[:-1], starts[1:], strict=True):
                    e = tri_mat[a, 0:3]
                    s = float(tri_mat[a, 3])
                    glMaterialfv(GL_FRONT, GL_EMISSION, (*e, 1.0))
                    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (s, s, s, 1.0))
                    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, float(tri_mat[a, 4]))
                    lo, hi = first + 3 * a, first + 3 * b
                    glDrawElements(GL_TRIANGLES, hi - lo, GL_UNSIGNED_INT, idx[lo:hi])
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)
            glDisable(GL_TEXTURE_2D)
            glMaterialfv(GL_FRONT, GL_EMISSION, (0.0, 0.0, 0.0, 1.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0.0)
            glColor4f(1.0, 1.0, 1.0, self._alpha)

    def lines(self, segments: Sequence[tuple[Vec3, Vec3]], rgba: RGBA, width: float = 1.0) -> None:
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glLineWidth(width)
        glColor4f(*rgba)
        glBegin(GL_LINES)
        for a, b in segments:
            glVertex3f(*a)
            glVertex3f(*b)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)

    def soft_disc(self, rgba: RGBA, slices: int = 24, inner: float = 0.0) -> None:
        r, g, b, a = rgba
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        ring = [
            (math.cos(2.0 * math.pi * i / slices), math.sin(2.0 * math.pi * i / slices))
            for i in range(slices + 1)
        ]
        # the plateau: a fan from the centre out to ``inner`` at full alpha
        plateau = inner if inner > 0.0 else 0.0
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(r, g, b, a)
        glVertex3f(0.0, 0.0, 0.0)
        if plateau > 0.0:
            for cx, cz in ring:
                glVertex3f(cx * plateau, 0.0, cz * plateau)
        else:
            glColor4f(r, g, b, 0.0)
            for cx, cz in ring:
                glVertex3f(cx, 0.0, cz)
        glEnd()
        if plateau > 0.0:  # the fade: a strip from the plateau rim (alpha a) to the edge (0)
            glBegin(GL_TRIANGLE_STRIP)
            for cx, cz in ring:
                glColor4f(r, g, b, a)
                glVertex3f(cx * plateau, 0.0, cz * plateau)
                glColor4f(r, g, b, 0.0)
                glVertex3f(cx, 0.0, cz)
            glEnd()
        glEnable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 1.0, self._alpha)

    # ------------------------------------------------------------------ static caching
    def cached(self, key: Any, build: Callable[[], None]) -> None:
        self._cache.draw(key, build)

    def glow_mesh(self, key: Any, rgba: RGBA, build: Callable[[], Any]) -> None:
        def emit() -> None:
            tris = np.asarray(build(), dtype=float).reshape(-1, 4)
            r, g, b, a = rgba
            glBegin(GL_TRIANGLES)
            for x, y, z, f in tris:
                glColor4f(r, g, b, a * f)
                glVertex3f(x, y, z)
            glEnd()

        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        self._cache.draw(("glow", key), emit)
        glEnable(GL_LIGHTING)
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def invalidate(self, key: Any = None) -> None:
        self._cache.invalidate(key)
        if key is not None:  # (a glow mesh is kept under its own key, like the modern one)
            self._cache.invalidate(("glow", key))

    # ------------------------------------------------------------------ 2D overlay
    def clip(self, x: float, y: float, w: float, h: float) -> None:
        height = getattr(self, "_overlay_height", self.height)
        glEnable(GL_SCISSOR_TEST)
        glScissor(int(x), int(height - (y + h)), max(0, int(w)), max(0, int(h)))

    def unclip(self) -> None:
        glDisable(GL_SCISSOR_TEST)

    def begin_overlay(self, width: int, height: int) -> None:
        self._overlay_height = height
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def end_overlay(self) -> None:
        glLineWidth(1.0)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)

    def rect(self, x: float, y: float, w: float, h: float, rgba: RGBA) -> None:
        glDisable(GL_TEXTURE_2D)
        glColor4f(*rgba)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def rect_outline(
        self, x: float, y: float, w: float, h: float, rgba: RGBA, width: float = 2.0
    ) -> None:
        glDisable(GL_TEXTURE_2D)
        glLineWidth(width)
        glColor4f(*rgba)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def image(self, tex_id: int, x: float, y: float, w: float, h: float) -> None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x, y + h)
        glTexCoord2f(1, 0)
        glVertex2f(x + w, y + h)
        glTexCoord2f(1, 1)
        glVertex2f(x + w, y)
        glTexCoord2f(0, 1)
        glVertex2f(x, y)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    # ------------------------------------------------------------------ resources
    def upload_texture_rgba(self, data: bytes, w: int, h: int) -> int:
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return int(tex)

    def delete_texture(self, tex_id: int) -> None:
        glDeleteTextures([tex_id])
