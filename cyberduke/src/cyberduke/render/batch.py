"""One draw for a whole person: geometry collected on the CPU, handed to the backend once.

A person is some sixty parts - a skull, a jaw, ten finger bones a hand - and drawing each
as its own primitive costs a state change and a draw call apiece (about 0.1 ms each on a
software renderer: a crowd of ten would be the whole frame). ``BatchContext`` is a
``DrawContext`` that draws nothing itself: it keeps the matrix stack and the current
material like a backend, turns every ``cube`` / ``cylinder`` / ``shape`` into transformed
triangles, groups them by texture and material, and on ``flush`` hands the lot to the real
context's ``draw_batch`` - one upload, a handful of draws.

It is pure NumPy (no OpenGL), so what it collects can be checked in tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from cyberduke.render.context import RGBA, DrawContext, Vec3
from cyberduke.render.shapes import mesh


def _rotation(angle_deg: float, x: float, y: float, z: float) -> np.ndarray:
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


_UNIT: dict[Any, tuple[np.ndarray, np.ndarray]] = {}


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    found = _UNIT.get("cube")
    if found is None:
        from cyberduke.render.backends.modern import quad_indices, unit_cube_vertices

        found = _UNIT["cube"] = (unit_cube_vertices().astype(np.float64), quad_indices(6))
    return found


def _unit_cylinder(slices: int, bottom: float, top: float) -> tuple[np.ndarray, np.ndarray]:
    key = ("cyl", slices, round(bottom, 3), round(top, 3))
    found = _UNIT.get(key)
    if found is None:
        from cyberduke.render.backends.modern import unit_cylinder_vertices

        base, idx = unit_cylinder_vertices(slices)
        base = base.astype(np.float64)
        k = np.where(base[:, 1] > 0.5, top, bottom)
        base[:, 0] *= k
        base[:, 2] *= k
        found = _UNIT[key] = (base, idx)
    return found


def _cofactor(r: np.ndarray) -> np.ndarray:
    """Normals' transform for a batch of linear maps (K, 3, 3): the cofactor matrices -
    the inverse transpose times the determinant (lengths do not matter: normals are
    normalised when lit). Written out: ``np.cross`` costs more than the sums."""
    a = r[:, :, 0]
    b = r[:, :, 1]
    c = r[:, :, 2]
    out = np.empty_like(r)
    # columns: b x c, c x a, a x b
    for col, (u, v) in enumerate(((b, c), (c, a), (a, b))):
        out[:, 0, col] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        out[:, 1, col] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        out[:, 2, col] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return out


_LAYOUTS: dict[tuple, tuple] = {}  # the parts' meshes laid out in a batch, by the parts
_MAX_LAYOUTS = 512


def _layout(meshes: list[tuple[np.ndarray, np.ndarray]], textures: list[int]) -> tuple:
    """(all the vertices one after another, each vertex's part, the triangles' indices,
    (texture, first index, count) ranges) for parts with these meshes and textures."""
    counts = np.array([len(m[0]) for m in meshes])
    base = np.concatenate([m[0] for m in meshes]).astype(np.float32)
    which = np.repeat(np.arange(len(meshes)), counts)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    tri_counts = np.array([len(m[1]) for m in meshes])
    indices = (np.concatenate([m[1] for m in meshes]) + np.repeat(starts, tri_counts)).astype(
        np.uint32
    )
    ranges: list[list] = []
    first = 0
    for tex, n in zip(textures, tri_counts.tolist(), strict=True):
        if ranges and ranges[-1][0] == tex:
            ranges[-1][2] += n
        else:
            ranges.append([tex, first, n])
        first += n
    return base, which, indices, [tuple(r) for r in ranges]


class BatchContext(DrawContext):
    """Collects geometry for one ``draw_batch`` on ``target`` (whose current transform
    places it). ``world_scale``: the scale ``target`` already has (a tall person), for
    tiled textures.

    Drawing calls only record an instance (which mesh, its transform and material); the
    transforming is done at ``flush``, every instance of a mesh at once."""

    name = "batch"

    def __init__(self, target: DrawContext, world_scale: float = 1.0, lo: bool = False) -> None:
        self.target = target
        self.lo = lo  # the coarser meshes (somebody further off)
        self.texture_tiles = getattr(target, "texture_tiles", {})
        self.world_scale = world_scale
        # the current transform as plain numbers (a 3x3, row by row, and a translation):
        # a person is a few hundred small matrix products - cheaper without NumPy
        self.r: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.t: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._stack: list[tuple] = []
        self._color = (1.0, 1.0, 1.0)
        self._emission = (0.0, 0.0, 0.0)
        self._spec = (0.0, 0.0)
        # (mesh key, rotation-scale 3x3 (9 numbers), translation, colour + emission +
        # specular, texture)
        self.instances: list[tuple] = []
        self.meshes: dict[Any, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ transform
    def push(self) -> None:
        self._stack.append((self.r, self.t))

    def pop(self) -> None:
        self.r, self.t = self._stack.pop()

    def translate(self, x: float, y: float, z: float) -> None:
        r, t = self.r, self.t
        self.t = (
            t[0] + r[0] * x + r[1] * y + r[2] * z,
            t[1] + r[3] * x + r[4] * y + r[5] * z,
            t[2] + r[6] * x + r[7] * y + r[8] * z,
        )

    def rotate(self, angle_deg: float, x: float, y: float, z: float) -> None:
        if not angle_deg:
            return
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        if x == 1.0 and y == 0.0 and z == 0.0:
            m = (1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
        elif y == 1.0 and x == 0.0 and z == 0.0:
            m = (c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c)
        elif z == 1.0 and x == 0.0 and y == 0.0:
            m = (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
        else:
            m = tuple(_rotation(angle_deg, x, y, z).ravel())
        r = self.r
        self.r = (
            r[0] * m[0] + r[1] * m[3] + r[2] * m[6],
            r[0] * m[1] + r[1] * m[4] + r[2] * m[7],
            r[0] * m[2] + r[1] * m[5] + r[2] * m[8],
            r[3] * m[0] + r[4] * m[3] + r[5] * m[6],
            r[3] * m[1] + r[4] * m[4] + r[5] * m[7],
            r[3] * m[2] + r[4] * m[5] + r[5] * m[8],
            r[6] * m[0] + r[7] * m[3] + r[8] * m[6],
            r[6] * m[1] + r[7] * m[4] + r[8] * m[7],
            r[6] * m[2] + r[7] * m[5] + r[8] * m[8],
        )

    def scale(self, x: float, y: float, z: float) -> None:
        r = self.r
        self.r = (
            r[0] * x,
            r[1] * y,
            r[2] * z,
            r[3] * x,
            r[4] * y,
            r[5] * z,
            r[6] * x,
            r[7] * y,
            r[8] * z,
        )

    # ------------------------------------------------------------------ material
    def color(self, r: float, g: float, b: float) -> None:
        self._color = (float(r), float(g), float(b))

    def emission(self, rgb: Sequence[float] | None) -> None:
        self._emission = (0.0, 0.0, 0.0) if rgb is None else tuple(float(c) for c in rgb[:3])  # type: ignore[assignment]

    def specular(self, rgb: Sequence[float] | None, shininess: float = 0.0) -> None:
        self._spec = (0.0, 0.0) if rgb is None else (float(sum(rgb[:3]) / 3.0), float(shininess))

    # ------------------------------------------------------------------ geometry
    def _record(self, key: Any, tex_id: int | None, r: tuple | None = None) -> None:
        self.instances.append(
            (
                key,
                self.r if r is None else r,
                self.t,
                (*self._color, *self._emission, *self._spec),
                int(tex_id or 0),
            )
        )

    def _axis_lengths(self) -> np.ndarray:
        r = self.r
        return (
            np.array(
                (
                    math.sqrt(r[0] * r[0] + r[3] * r[3] + r[6] * r[6]),
                    math.sqrt(r[1] * r[1] + r[4] * r[4] + r[7] * r[7]),
                    math.sqrt(r[2] * r[2] + r[5] * r[5] + r[8] * r[8]),
                )
            )
            * self.world_scale
        )

    def cube(self, tex_id: int | None = None) -> None:
        tile = self.texture_tiles.get(tex_id) if tex_id else None
        if tile:
            from cyberduke.render.backends.modern import tiled_uvs

            rep = tuple(round(x, 2) for x in self.cube_repeat(self._axis_lengths(), tile))
            key = ("cube", rep)
            if key not in self.meshes:
                base, idx = _unit_cube()
                tiled = base.copy()
                tiled[:, 6:8] = tiled_uvs(base, 1, rep)
                self.meshes[key] = (tiled, idx)
            self._record(key, tex_id)
        else:
            if "cube" not in self.meshes:
                self.meshes["cube"] = _unit_cube()
            self._record("cube", tex_id)

    def cylinder(
        self, r_bot: float, r_top: float, h: float, slices: int = 10, tex_id: int | None = None
    ) -> None:
        radius = max(r_bot, r_top)
        if radius <= 0.0:
            return
        key: Any = ("cyl", slices, round(r_bot / radius, 3), round(r_top / radius, 3))
        base, idx = _unit_cylinder(*key[1:])
        tile = self.texture_tiles.get(tex_id) if tex_id else None
        if tile:
            from cyberduke.render.backends.modern import tiled_uvs

            rep = tuple(
                round(x, 2)
                for x in self.cylinder_repeat(r_bot, r_top, h, self._axis_lengths(), tile)
            )
            key = (*key, rep)
            if key not in self.meshes:
                tiled = base.copy()
                tiled[:, 6:8] = tiled_uvs(base, 2, rep)
                self.meshes[key] = (tiled, idx)
        elif key not in self.meshes:
            self.meshes[key] = (base, idx)
        r = self.r
        self._record(
            key,
            tex_id,
            (
                r[0] * radius,
                r[1] * h,
                r[2] * radius,
                r[3] * radius,
                r[4] * h,
                r[5] * radius,
                r[6] * radius,
                r[7] * h,
                r[8] * radius,
            ),
        )

    def add_part(
        self,
        name: str,
        size: tuple[float, float, float],
        color: tuple[float, float, float],
        tex_id: int | None,
        emission: tuple[float, float, float] | None = None,
        shininess: float = 0.0,
    ) -> None:
        """A shape scaled to ``size`` in a colour (and glow, and shine) - what ``push``,
        ``scale``, ``color``, ``emission``, ``specular``, ``shape`` and ``pop`` would do,
        in one call (a person is sixty of them)."""
        if name not in self.meshes:
            self.meshes[name] = mesh(name, self.lo)
        r = self.r
        x, y, z = size
        e = emission or (0.0, 0.0, 0.0)
        self.instances.append(
            (
                name,
                (
                    r[0] * x,
                    r[1] * y,
                    r[2] * z,
                    r[3] * x,
                    r[4] * y,
                    r[5] * z,
                    r[6] * x,
                    r[7] * y,
                    r[8] * z,
                ),
                self.t,
                (
                    color[0], color[1], color[2], e[0], e[1], e[2],
                    *((0.5, shininess) if shininess else self._spec),
                ),
                int(tex_id or 0),
            )
        )  # fmt: skip

    def shape(self, name: str, tex_id: int | None = None) -> None:
        if name not in self.meshes:
            self.meshes[name] = mesh(name, self.lo)
        self._record(name, tex_id)

    # ------------------------------------------------------------------ flushing
    def flush(self) -> int:
        """Transforms everything recorded and hands it to the target in one
        ``draw_batch`` (vertex rows: position, normal, uv, colour, emission, specular;
        one range per texture); returns the number of vertices. Every vertex of every
        part is transformed in one go (each by its part's matrix); how the parts' meshes
        are laid out in the batch is worked out once per set of parts (``_LAYOUTS``)."""
        if not self.instances:
            return 0
        # in texture order, so each texture's triangles are one range
        inst = sorted(self.instances, key=lambda x: x[4])
        layout_key = tuple((x[0], x[4]) for x in inst)
        layout = _LAYOUTS.get(layout_key)
        if layout is None:
            layout = _layout([self.meshes[x[0]] for x in inst], [x[4] for x in inst])
            if len(_LAYOUTS) < _MAX_LAYOUTS:
                _LAYOUTS[layout_key] = layout
        base, which, indices, ranges = layout
        # per part: its 3x3 (9), translation (3), the normals' 3x3 (9), its material (8)
        part = np.array([(*x[1], *x[2], *x[3]) for x in inst], dtype=np.float32)
        per = np.empty((len(inst), 29), dtype=np.float32)
        per[:, 0:12] = part[:, 0:12]
        per[:, 12:21] = _cofactor(part[:, 0:9].reshape(-1, 3, 3)).reshape(-1, 9)
        per[:, 21:29] = part[:, 12:20]
        v = per[which]  # (one row per vertex: its part's numbers)
        x, y, z = base[:, 0:1], base[:, 1:2], base[:, 2:3]
        nx, ny, nz = base[:, 3:4], base[:, 4:5], base[:, 5:6]
        rows = np.empty((len(base), 16), dtype=np.float32)
        # each vertex by its own part's matrix (rows of 3: x * column 0 + ... )
        rows[:, 0:3] = v[:, [0, 3, 6]] * x + v[:, [1, 4, 7]] * y + v[:, [2, 5, 8]] * z + v[:, 9:12]
        rows[:, 3:6] = v[:, [12, 15, 18]] * nx + v[:, [13, 16, 19]] * ny + v[:, [14, 17, 20]] * nz
        rows[:, 6:8] = base[:, 6:8]
        rows[:, 8:16] = v[:, 21:29]
        self.target.draw_batch(rows, indices, ranges)
        self.instances = []
        return len(rows)

    # ------------------------------------------------------------------ not batched
    def begin_frame(self, eye: Vec3, target: Vec3) -> None:  # pragma: no cover - not used
        raise RuntimeError("a batch is not a frame")

    def end_frame(self) -> None:  # pragma: no cover - not used
        pass

    def lines(self, segments: Sequence[tuple[Vec3, Vec3]], rgba: RGBA, width: float = 1.0) -> None:
        pass

    def soft_disc(self, rgba: RGBA, slices: int = 24, inner: float = 0.0) -> None:
        pass

    def cached(self, key: Any, build: Callable[[], None]) -> None:
        build()

    def invalidate(self, key: Any = None) -> None:
        pass

    def begin_overlay(self, width: int, height: int) -> None:  # pragma: no cover
        raise RuntimeError("no overlay in a batch")

    def end_overlay(self) -> None:  # pragma: no cover
        pass

    def rect(self, x: float, y: float, w: float, h: float, rgba: RGBA) -> None:  # pragma: no cover
        pass

    def rect_outline(self, x, y, w, h, rgba, width: float = 2.0) -> None:  # pragma: no cover
        pass

    def image(
        self, tex_id: int, x: float, y: float, w: float, h: float
    ) -> None:  # pragma: no cover
        pass

    def upload_texture_rgba(self, data: bytes, w: int, h: int) -> int:  # pragma: no cover
        return self.target.upload_texture_rgba(data, w, h)

    def delete_texture(self, tex_id: int) -> None:  # pragma: no cover
        self.target.delete_texture(tex_id)


__all__ = ["BatchContext"]
