"""Clipping the security cameras' laser fans against what is actually drawn.

The beam is a purely visual effect, so it has to stop where the player *sees* a surface.
Gameplay colliders are deliberately generous (the airlock's pipe collider reaches 0.3 m
out from the pipes, the greenhouse rim is one 5 m block over 0.56 m planter beds), so rays
clipped against them ended in mid-air in front of walls and hung their "drop seam" lines
there. This module keeps the level's visible solid geometry as exact oriented boxes and
clips every ray of a fan in 3D with one vectorised slab test.

Pure NumPy (no OpenGL), so it is testable headlessly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from cyberduke.core.geometry import AABB
from cyberduke.world.props import Prop

Vec3 = tuple[float, float, float]

# Props that are not surfaces a beam should stop on: walk-over clutter, light pools, fog,
# pulsing caps, decals and anything translucent.
_SKIP_TAGS = frozenset({"clutter", "fog", "glow", "pulse", "transparent"})
_MIN_EXTENT = 0.05  # thinner than this in every direction: trim, not a surface


def blocks_beam(prop: Prop) -> bool:
    if prop.kind not in ("box", "cylinder") or prop.material.alpha < 1.0:
        return False
    if prop.tags & _SKIP_TAGS:
        return False
    if prop.kind == "box":
        return max(prop.size) >= _MIN_EXTENT
    r_bot, r_top, h = prop.size
    return max(r_bot, r_top, h) >= _MIN_EXTENT


def _rotation_matrix(rotations: Sequence[tuple[float, float, float, float]]) -> np.ndarray:
    """Local-to-world rotation of glRotatef calls applied in order (M = R1 R2 ... Rn)."""
    m = np.identity(3)
    for angle, ax, ay, az in rotations:
        n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        x, y, z = ax / n, ay / n, az / n
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        t = 1.0 - c
        r = np.array(
            (
                (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
                (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
                (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
            )
        )
        m = m @ r
    return m


@dataclass
class BoxSet:
    """Oriented boxes: centres (N, 3), world-to-local rotations (N, 3, 3), half extents (N, 3)."""

    centers: np.ndarray
    to_local: np.ndarray
    halves: np.ndarray

    @classmethod
    def empty(cls) -> BoxSet:
        return cls(np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)))

    @classmethod
    def from_aabbs(cls, boxes: Iterable[AABB]) -> BoxSet:
        boxes = list(boxes)
        if not boxes:
            return cls.empty()
        lo = np.array([(b.min_x, b.min_y, b.min_z) for b in boxes], dtype=float)
        hi = np.array([(b.max_x, b.max_y, b.max_z) for b in boxes], dtype=float)
        eye = np.broadcast_to(np.identity(3), (len(boxes), 3, 3)).copy()
        return cls((lo + hi) / 2.0, eye, (hi - lo) / 2.0)

    @classmethod
    def from_props(cls, props: Iterable[Prop]) -> BoxSet:
        centers, rots, halves = [], [], []
        for p in props:
            if p.kind == "box":
                centers.append((p.x, p.y, p.z))
                rots.append(_rotation_matrix(p.rotate).T)
                halves.append(tuple(v / 2.0 for v in p.size))
            else:  # cylinders: their exact bounds are close enough for a beam
                fp = p.footprint()
                lo, hi = (fp.min_x, fp.min_y, fp.min_z), (fp.max_x, fp.max_y, fp.max_z)
                centers.append(tuple((a + b) / 2.0 for a, b in zip(lo, hi, strict=True)))
                rots.append(np.identity(3))
                halves.append(tuple((b - a) / 2.0 for a, b in zip(lo, hi, strict=True)))
        if not centers:
            return cls.empty()
        return cls(
            np.array(centers, dtype=float),
            np.array(rots, dtype=float),
            np.array(halves, dtype=float),
        )

    def __len__(self) -> int:
        return len(self.centers)

    def subset(self, mask: np.ndarray) -> BoxSet:
        return BoxSet(self.centers[mask], self.to_local[mask], self.halves[mask])

    def concat(self, other: BoxSet) -> BoxSet:
        if not len(other):
            return self
        if not len(self):
            return other
        return BoxSet(
            np.concatenate([self.centers, other.centers]),
            np.concatenate([self.to_local, other.to_local]),
            np.concatenate([self.halves, other.halves]),
        )

    def near(self, x: float, z: float, reach: float) -> np.ndarray:
        """Mask of boxes whose bounding sphere comes within ``reach`` of (x, z) on the ground."""
        if not len(self):
            return np.zeros(0, dtype=bool)
        radius = np.linalg.norm(self.halves, axis=1)
        d = np.hypot(self.centers[:, 0] - x, self.centers[:, 2] - z)
        return d <= reach + radius


def clip_rays(origin: Vec3, ends: Sequence[Vec3], boxes: BoxSet) -> list[tuple[Vec3, bool]]:
    """Clips the segments ``origin -> end`` against ``boxes``. Returns, per segment, the
    first point hit (or the end itself) and whether it landed on an upright face above the
    floor (where the beam draws a vertical "drop seam" down to the ground). Boxes that
    contain the origin are ignored."""
    o = np.asarray(origin, dtype=float)
    e = np.asarray(ends, dtype=float).reshape(-1, 3)
    out: list[tuple[Vec3, bool]] = []
    if not len(boxes) or not len(e):
        return [(tuple(p), False) for p in e.tolist()]  # type: ignore[misc]
    d = e - o  # (R, 3)
    o_l = np.einsum("nij,nj->ni", boxes.to_local, o - boxes.centers)  # (N, 3)
    d_l = np.einsum("nij,rj->rni", boxes.to_local, d)  # (R, N, 3)
    h = boxes.halves[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d_l
        t1 = (-h - o_l[None]) * inv
        t2 = (h - o_l[None]) * inv
    parallel = np.abs(d_l) < 1e-12
    inside = np.abs(o_l)[None].repeat(len(e), axis=0) <= h
    t_lo = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    t_hi = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    t_enter = t_lo.max(axis=2)  # (R, N)
    t_exit = t_hi.min(axis=2)
    hit = (t_enter <= t_exit) & (t_enter > 1e-3) & (t_enter <= 1.0)
    t_first = np.where(hit, t_enter, np.inf)
    best = t_first.argmin(axis=1)  # (R,)
    for r in range(len(e)):
        n = best[r]
        t = t_first[r, n]
        if not np.isfinite(t):
            out.append((tuple(e[r].tolist()), False))  # type: ignore[arg-type]
            continue
        p = o + d[r] * t
        axis = int(t_lo[r, n].argmax())
        # Entered through the box's local y face while that axis points up: a top, not a wall.
        on_top = axis == 1 and abs(boxes.to_local[n, 1, 1]) > 0.9
        out.append(((float(p[0]), float(p[1]), float(p[2])), not on_top and p[1] > 0.05))
    return out


class BeamOccluders:
    """The static visible geometry of a level, for clipping camera beams (built once per
    level; each camera keeps the subset within its reach)."""

    def __init__(self, props: Iterable[Prop]) -> None:
        self.boxes = BoxSet.from_props(p for p in props if blocks_beam(p))
        self._near: dict[tuple[float, float, float], BoxSet] = {}

    def around(self, x: float, z: float, reach: float) -> BoxSet:
        key = (round(x, 3), round(z, 3), round(reach, 3))
        found = self._near.get(key)
        if found is None:
            found = self.boxes.subset(self.boxes.near(x, z, reach))
            self._near[key] = found
        return found

    def height_above(self, x: float, y: float, z: float, limit: float = 8.0) -> float | None:
        """Height of the first drawn surface straight above (x, y, z) within ``limit``
        metres - the ceiling, beam or dome panel a ceiling-mounted device hangs from."""
        boxes = self.around(x, z, 0.5)
        ((point, _),) = clip_rays((x, y, z), [(x, y + limit, z)], boxes)
        return point[1] if point[1] < y + limit - 1e-6 else None
