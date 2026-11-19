"""Static level geometry as data.

A ``Prop`` is one box or cylinder with a material. Sectors are lists of props; the renderer
draws them (and can compile them into display lists), tests can reason about them, and a
future loader can read them from a file. Props carry no OpenGL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cyberduke.core.geometry import AABB

RGB = tuple[float, float, float]
Rotation = tuple[float, float, float, float]  # (angle_deg, axis_x, axis_y, axis_z)


@dataclass(frozen=True)
class Material:
    texture: str | None = None  # procedural texture style name, or None for flat colour
    color: RGB = (1.0, 1.0, 1.0)
    emission: RGB | None = None  # self-illumination (values may exceed 1.0 for neon)
    specular: RGB | None = None
    shininess: float = 0.0
    alpha: float = 1.0  # < 1: translucent (drawn in the late pass, no depth writes: fog)


DEFAULT_MATERIAL = Material()


@dataclass(frozen=True)
class Prop:
    """``kind`` is "box" (unit cube scaled by ``size``), "cylinder" (``size`` is
    (radius_bottom, radius_top, height) with ``slices`` segments; rises from y=0) or "disc"
    (a flat, unlit, soft-edged translucent disc in the XZ plane: ``size`` is (radius_x,
    radius_z, inner) where ``inner`` is the fraction of the radius at full opacity before
    the fade to the edge; the material's colour and alpha are what you see).

    ``rotate`` lists glRotatef-style rotations applied after translation, in order.
    ``solid`` marks props that must lie inside a collider (see tests/test_level_data.py).
    ``clip`` (discs only) keeps just the part of the disc inside a rectangle given in the
    disc's own unit coordinates, ``(u0, u1, v0, v1)`` with u along its x radius and v along
    its z radius (-1..1 spans the whole disc): a pool of light that stops at the edge of the
    planter it lies on, or the half of a splash that is above the floor.
    """

    kind: str
    x: float
    y: float
    z: float
    size: tuple[float, float, float]
    material: Material = DEFAULT_MATERIAL
    rotate: tuple[Rotation, ...] = ()
    slices: int = 10
    solid: bool = False
    label: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)
    clip: tuple[float, float, float, float] | None = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def box(
        cls,
        x: float,
        y: float,
        z: float,
        sx: float,
        sy: float,
        sz: float,
        material: Material = DEFAULT_MATERIAL,
        *,
        rotate: tuple[Rotation, ...] = (),
        solid: bool = False,
        label: str = "",
        tags: frozenset[str] = frozenset(),
    ) -> Prop:
        return cls("box", x, y, z, (sx, sy, sz), material, rotate, 0, solid, label, tags)

    @classmethod
    def cylinder(
        cls,
        x: float,
        y: float,
        z: float,
        r_bot: float,
        r_top: float,
        h: float,
        material: Material = DEFAULT_MATERIAL,
        *,
        slices: int = 10,
        rotate: tuple[Rotation, ...] = (),
        solid: bool = False,
        label: str = "",
        tags: frozenset[str] = frozenset(),
    ) -> Prop:
        return cls(
            "cylinder", x, y, z, (r_bot, r_top, h), material, rotate, slices, solid, label, tags
        )

    @classmethod
    def disc(
        cls,
        x: float,
        y: float,
        z: float,
        rx: float,
        rz: float,
        inner: float,
        material: Material = DEFAULT_MATERIAL,
        *,
        slices: int = 24,
        rotate: tuple[Rotation, ...] = (),
        label: str = "",
        tags: frozenset[str] = frozenset(),
        clip: tuple[float, float, float, float] | None = None,
    ) -> Prop:
        return cls(
            "disc", x, y, z, (rx, rz, inner), material, rotate, slices, False, label, tags, clip
        )

    # ------------------------------------------------------------------ geometry
    def local_to_world(self, px: float, py: float, pz: float) -> tuple[float, float, float]:
        """World position of a point given in the prop's local (pre-rotation) frame."""
        rx, ry, rz = _rotate_point((px, py, pz), self.rotate)
        return (self.x + rx, self.y + ry, self.z + rz)

    def footprint(self) -> AABB:
        """Exact axis-aligned bounds of the (possibly rotated) prop."""
        if self.kind == "box":
            hx, hy, hz = (v / 2.0 for v in self.size)
            corners = [
                (sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ]
        elif self.kind == "disc":
            rx, rz, _ = self.size
            u0, u1, v0, v1 = self.clip or (-1.0, 1.0, -1.0, 1.0)
            us = (max(-1.0, u0), min(1.0, u1))
            vs = (max(-1.0, v0), min(1.0, v1))
            corners = [(u * rx, 0.0, v * rz) for u in us for v in vs]
        else:
            r_bot, r_top, h = self.size
            r = max(r_bot, r_top)
            corners = [(sx * r, y, sz * r) for sx in (-1, 1) for sz in (-1, 1) for y in (0.0, h)]
        pts = [_rotate_point(c, self.rotate) for c in corners]
        xs = [self.x + px for px, _, _ in pts]
        ys = [self.y + py for _, py, _ in pts]
        zs = [self.z + pz for _, _, pz in pts]
        return AABB(min(xs), max(xs), min(zs), max(zs), min(ys), max(ys))


def _rotate_point(p: tuple[float, float, float], rotations: tuple[Rotation, ...]):
    """Applies glRotatef-style rotations. GL composes matrices right-to-left, so a point is
    transformed by the *last* rotation first."""
    x, y, z = p
    for angle, ax, ay, az in reversed(rotations):
        n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ux, uy, uz = ax / n, ay / n, az / n
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        dot = ux * x + uy * y + uz * z
        cx, cy, cz = uy * z - uz * y, uz * x - ux * z, ux * y - uy * x
        x, y, z = (
            x * c + cx * s + ux * dot * (1 - c),
            y * c + cy * s + uy * dot * (1 - c),
            z * c + cz * s + uz * dot * (1 - c),
        )
    return x, y, z
