"""Axis-aligned bounding boxes and segment/ray intersection tests.

World convention (unchanged from the original code): X/Z is the ground plane, Y is up.
Yaw 0 deg faces +Z, yaw 90 deg faces +X.

All collision and line-of-sight code in the project must go through this module so that
there is exactly one definition of "does this segment hit this box" and "does this body
overlap this box".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

DEFAULT_OBSTACLE_HEIGHT = 2.5
"""Vertical extent assumed for obstacles authored without an explicit Y range."""

VERTICAL_OVERLAP_EPSILON = 0.05
"""Slack used when deciding whether a body's vertical span overlaps a box (lets an agent
stand exactly on top of a 2.65 m column or walk under a 2.75 m catwalk)."""

_EPS = 1e-6

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class AABB:
    """Axis-aligned box. ``blocks_sight=False`` marks things like handrails that stop
    movement but that guards and cameras can see through."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float
    min_y: float = 0.0
    max_y: float = DEFAULT_OBSTACLE_HEIGHT
    blocks_sight: bool = True

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_tuple(cls, box: Sequence[float], *, blocks_sight: bool = True) -> AABB:
        """Accepts the legacy ``(min_x, max_x, min_z, max_z[, min_y, max_y])`` layout."""
        if isinstance(box, AABB):
            return box
        if len(box) >= 6:
            return cls(box[0], box[1], box[2], box[3], box[4], box[5], blocks_sight)
        return cls(box[0], box[1], box[2], box[3], blocks_sight=blocks_sight)

    @classmethod
    def from_center(
        cls,
        cx: float,
        cz: float,
        width: float,
        depth: float,
        min_y: float = 0.0,
        max_y: float = DEFAULT_OBSTACLE_HEIGHT,
        *,
        blocks_sight: bool = True,
    ) -> AABB:
        return cls(
            cx - width / 2.0,
            cx + width / 2.0,
            cz - depth / 2.0,
            cz + depth / 2.0,
            min_y,
            max_y,
            blocks_sight,
        )

    def inflated(self, radius: float) -> AABB:
        """Expands the horizontal footprint by ``radius`` on every side (Minkowski sum
        with an agent's collision circle)."""
        return replace(
            self,
            min_x=self.min_x - radius,
            max_x=self.max_x + radius,
            min_z=self.min_z - radius,
            max_z=self.max_z + radius,
        )

    # ------------------------------------------------------------------ queries
    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) * 0.5

    @property
    def center_z(self) -> float:
        return (self.min_z + self.max_z) * 0.5

    def contains_xz(self, x: float, z: float, margin: float = 0.0) -> bool:
        return (self.min_x - margin <= x <= self.max_x + margin) and (
            self.min_z - margin <= z <= self.max_z + margin
        )

    def overlaps_vertical_span(self, feet_y: float, top_y: float) -> bool:
        """True when a body occupying [feet_y, top_y] shares vertical space with the box."""
        eps = VERTICAL_OVERLAP_EPSILON
        return not (feet_y >= self.max_y - eps or top_y <= self.min_y + eps)

    def overlaps_cylinder(
        self, x: float, z: float, radius: float, feet_y: float, top_y: float
    ) -> bool:
        """Height-aware test of a vertical cylinder (agent body) against this box."""
        return self.overlaps_vertical_span(feet_y, top_y) and self.contains_xz(x, z, radius)


# ---------------------------------------------------------------------------- segments
def segment_hits_aabb_2d(x1: float, z1: float, x2: float, z2: float, box: AABB) -> float | None:
    """Slab test of the ground-plane segment (x1,z1)->(x2,z2) against the box footprint.

    Returns the entry parameter ``t`` in [0, 1] along the segment, or ``None`` on a miss.
    A segment starting inside the box returns ``t == 0.0``.
    """
    tmin, tmax = 0.0, 1.0
    for p, d, lo, hi in ((x1, x2 - x1, box.min_x, box.max_x), (z1, z2 - z1, box.min_z, box.max_z)):
        if abs(d) < _EPS:
            if p < lo or p > hi:
                return None
            continue
        t_lo = (lo - p) / d
        t_hi = (hi - p) / d
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        tmin = max(tmin, t_lo)
        tmax = min(tmax, t_hi)
        if tmin > tmax:
            return None
    return tmin


def segment_intersects_aabb_2d(x1: float, z1: float, x2: float, z2: float, box: AABB) -> bool:
    return segment_hits_aabb_2d(x1, z1, x2, z2, box) is not None


def segment_hits_aabb_3d(p1: Vec3, p2: Vec3, box: AABB) -> float | None:
    """Slab test of a 3D segment against the full box. Returns entry ``t`` or ``None``."""
    tmin, tmax = 0.0, 1.0
    spans = (
        (p1[0], p2[0] - p1[0], box.min_x, box.max_x),
        (p1[1], p2[1] - p1[1], box.min_y, box.max_y),
        (p1[2], p2[2] - p1[2], box.min_z, box.max_z),
    )
    for p, d, lo, hi in spans:
        if abs(d) < _EPS:
            if p < lo or p > hi:
                return None
            continue
        t_lo = (lo - p) / d
        t_hi = (hi - p) / d
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        tmin = max(tmin, t_lo)
        tmax = min(tmax, t_hi)
        if tmin > tmax:
            return None
    return tmin


def segment_intersects_aabb_3d(p1: Vec3, p2: Vec3, box: AABB) -> bool:
    return segment_hits_aabb_3d(p1, p2, box) is not None


# ---------------------------------------------------------------------------- bulk helpers
def any_cylinder_overlap(
    boxes: Iterable[AABB], x: float, z: float, radius: float, feet_y: float, top_y: float
) -> bool:
    return any(b.overlaps_cylinder(x, z, radius, feet_y, top_y) for b in boxes)


def segment_clear_2d(
    x1: float, z1: float, x2: float, z2: float, boxes: Iterable[AABB], radius: float = 0.0
) -> bool:
    """True when the ground segment touches none of the (optionally inflated) boxes.

    Hot path (every navigating NPC calls it every frame), so the inflation and the slab
    test are done inline on the numbers rather than by building inflated boxes, and boxes
    entirely outside the segment's bounding rectangle are skipped up front.
    """
    lo_x, hi_x = (x1, x2) if x1 <= x2 else (x2, x1)
    lo_z, hi_z = (z1, z2) if z1 <= z2 else (z2, z1)
    dx, dz = x2 - x1, z2 - z1
    for b in boxes:
        min_x, max_x = b.min_x - radius, b.max_x + radius
        min_z, max_z = b.min_z - radius, b.max_z + radius
        if max_x < lo_x or min_x > hi_x or max_z < lo_z or min_z > hi_z:
            continue
        tmin, tmax = 0.0, 1.0
        if abs(dx) < _EPS:
            if x1 < min_x or x1 > max_x:
                continue
        else:
            t_lo, t_hi = (min_x - x1) / dx, (max_x - x1) / dx
            if t_lo > t_hi:
                t_lo, t_hi = t_hi, t_lo
            tmin, tmax = max(tmin, t_lo), min(tmax, t_hi)
            if tmin > tmax:
                continue
        if abs(dz) < _EPS:
            if z1 < min_z or z1 > max_z:
                continue
        else:
            t_lo, t_hi = (min_z - z1) / dz, (max_z - z1) / dz
            if t_lo > t_hi:
                t_lo, t_hi = t_hi, t_lo
            tmin, tmax = max(tmin, t_lo), min(tmax, t_hi)
            if tmin > tmax:
                continue
        return False
    return True


def line_of_sight_clear(p1: Vec3, p2: Vec3, boxes: Iterable[AABB]) -> bool:
    """3D visibility test that ignores boxes flagged ``blocks_sight=False``."""
    return not any(b.blocks_sight and segment_intersects_aabb_3d(p1, p2, b) for b in boxes)


def distance_xz(x1: float, z1: float, x2: float, z2: float) -> float:
    return math.hypot(x2 - x1, z2 - z1)


def distance_3d(a: Vec3, b: Vec3) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


# ---------------------------------------------------------------------------- viewpoints
@dataclass(frozen=True)
class Viewpoint:
    """An eye position plus a unit forward vector: where something is looking from and
    toward. Built from the player's camera or a hijacked security camera, so code that
    needs "what is in front of the netrunner" does not care which one it is."""

    x: float
    y: float
    z: float
    fx: float
    fy: float
    fz: float
    source: str = "player"  # "player" or the id of the device providing the view

    @property
    def eye(self) -> Vec3:
        return (self.x, self.y, self.z)

    @property
    def forward(self) -> Vec3:
        return (self.fx, self.fy, self.fz)

    @property
    def target(self) -> Vec3:
        """A point one metre ahead (what a look-at camera wants)."""
        return (self.x + self.fx, self.y + self.fy, self.z + self.fz)

    def distance_to(self, point: Vec3) -> float:
        return distance_3d(self.eye, point)

    def angle_to(self, point: Vec3) -> float:
        """Degrees between the forward vector and the direction to ``point`` (0 = dead centre)."""
        dx, dy, dz = point[0] - self.x, point[1] - self.y, point[2] - self.z
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        if n < _EPS:
            return 0.0
        dot = (dx * self.fx + dy * self.fy + dz * self.fz) / n
        return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def forward_vector(yaw_deg: float, pitch_deg: float, *, yaw_zero_is_x: bool) -> Vec3:
    """Unit forward vector for a yaw/pitch pair. The two conventions in the code base:
    the player camera (``yaw_zero_is_x=True``: 0 = +X, 90 = +Z) and NPCs/devices
    (``yaw_zero_is_x=False``: 0 = +Z, 90 = +X). Pitch is positive upward."""
    cp = math.cos(math.radians(pitch_deg))
    sp = math.sin(math.radians(pitch_deg))
    rad = math.radians(yaw_deg)
    if yaw_zero_is_x:
        return (cp * math.cos(rad), sp, cp * math.sin(rad))
    return (cp * math.sin(rad), sp, cp * math.cos(rad))


def yaw_towards(from_x: float, from_z: float, to_x: float, to_z: float) -> float:
    """Heading in degrees from one point to another (yaw 0 = +Z, 90 = +X)."""
    return math.degrees(math.atan2(to_x - from_x, to_z - from_z))
