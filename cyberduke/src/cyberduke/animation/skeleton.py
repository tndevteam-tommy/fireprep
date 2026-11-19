"""The humanoid's body as boxes, for collision - the same parts ``render.humanoid`` draws.

``body_boxes(pose, ...)`` walks the body (``animation.rig``: the one description of it
both use) and returns every part as a 4x4 matrix that maps the unit cube (-0.5..0.5) onto
the box the part fills, in the body's own frame: feet at the origin, facing +z, the body's
left along +x. The renderer draws a rounded shape inside each of those boxes.
``tests/test_skeleton.py`` draws a pose through the renderer and checks that the two agree
box for box, so they cannot drift apart.

The knockout rag doll (``animation.ragdoll``) uses it to test a falling body against the
walls around it, and to find how far above the floor its lowest point is.
"""

from __future__ import annotations

import math

import numpy as np

from cyberduke.animation import rig
from cyberduke.animation.pose import HumanoidPose
from cyberduke.animation.rig import (
    ANKLE_DROP,
    BOOT_DROP,
    BOOT_HEIGHT,
    EYE_LINE,
    GROUPS,
    GUARD_KIT,
    HIP_HEIGHT,
    BodyKit,
)


def rotation3(angle_deg: float, x: float, y: float, z: float) -> np.ndarray:
    """3x3 rotation about an axis, the glRotatef way (right-handed, degrees)."""
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array(
        (
            (c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s),
            (y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s),
            (z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)),
        )
    )


def rotation(angle_deg: float, x: float, y: float, z: float) -> np.ndarray:
    """4x4 rotation about an axis, the glRotatef way (right-handed, degrees)."""
    m = np.identity(4)
    m[:3, :3] = rotation3(angle_deg, x, y, z)
    return m


def _axis_rotation(angle_deg: float, axis: int) -> np.ndarray:
    """The same about a coordinate axis (0 = x, 1 = y, 2 = z), without the general case."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    if axis == 0:
        return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
    if axis == 1:
        return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


class _Stack:
    """A matrix stack kept as (rotation 3x3, translation) - rigid moves only until a box
    is emitted with its size - collecting the parts ``animation.rig`` emits as boxes."""

    def __init__(self, base: np.ndarray | None = None) -> None:
        if base is None:
            self.r, self.p = np.identity(3), np.zeros(3)
        else:
            self.r, self.p = base[:3, :3].copy(), base[:3, 3].copy()
        self.saved: list[tuple[np.ndarray, np.ndarray]] = []
        self.boxes: list[tuple[str, np.ndarray]] = []

    def push(self) -> None:
        self.saved.append((self.r, self.p))

    def pop(self) -> None:
        self.r, self.p = self.saved.pop()

    def translate(self, x: float, y: float, z: float) -> None:
        r = self.r
        self.p = self.p + r[:, 0] * x + r[:, 1] * y + r[:, 2] * z

    def rotate(self, a: float, x: float, y: float, z: float) -> None:
        if not a:
            return
        if x == 1 and not y and not z:
            self.r = self.r @ _axis_rotation(a, 0)
        elif y == 1 and not x and not z:
            self.r = self.r @ _axis_rotation(a, 1)
        elif z == 1 and not x and not y:
            self.r = self.r @ _axis_rotation(a, 2)
        else:
            self.r = self.r @ rotation3(a, x, y, z)

    def box(self, name: str, sx: float, sy: float, sz: float) -> None:
        m = np.zeros((4, 4))
        m[:3, :3] = self.r * (sx, sy, sz)
        m[:3, 3] = self.p
        m[3, 3] = 1.0
        self.boxes.append((name, m))

    def part(self, name: str, size, shape: str, material: str) -> None:
        if name.startswith(rig.DECOR):  # (hair, chrome, neon: drawn, not collided with)
            return
        self.box(name, *size)


def body_boxes(
    pose: HumanoidPose,
    kit: BodyKit = GUARD_KIT,
    base: np.ndarray | None = None,
    parts: tuple[str, ...] = GROUPS,
) -> list[tuple[str, np.ndarray]]:
    """Every box of the posed body as (part name, unit-cube matrix) - all of them in the
    renderer's draw order, or just the ``parts`` groups. ``base`` is the body's placement
    (default: its own frame)."""
    whole = tuple(parts) == GROUPS
    boxes: list[tuple[str, np.ndarray]] = []
    for group in GROUPS:
        if group not in parts:
            continue
        st = _Stack(base)
        rig.build_group(st, pose, group, kit, whole)
        boxes.extend(st.boxes)
    return boxes


__all__ = [
    "ANKLE_DROP",
    "BOOT_DROP",
    "BOOT_HEIGHT",
    "EYE_LINE",
    "GROUPS",
    "GUARD_KIT",
    "HIP_HEIGHT",
    "BodyKit",
    "body_boxes",
    "box_points",
    "rotation",
    "rotation3",
]


# ----------------------------------------------------------------------------- sampling
def _surface_lattice(size: tuple[float, float, float], spacing: float) -> np.ndarray:
    """Unit-cube points on the surface of a box of ``size``, no more than ``spacing``
    apart on it (corners, edges and faces), plus the centre."""
    ticks = [np.linspace(-0.5, 0.5, max(2, math.ceil(s / spacing) + 1)) for s in size]
    gx, gy, gz = np.meshgrid(*ticks, indexing="ij")
    pts = np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1)
    on_surface = (np.abs(pts) >= 0.5 - 1e-9).any(axis=1)
    return np.concatenate((pts[on_surface], np.zeros((1, 3))))


_LATTICES: dict[tuple[str, float], np.ndarray] = {}


def box_points(boxes: list[tuple[str, np.ndarray]], spacing: float = 0.11) -> np.ndarray:
    """Surface sample points (N, 3) of all the boxes, for collision tests."""
    out = []
    for name, m in boxes:
        key = (name, spacing)
        lattice = _LATTICES.get(key)
        if lattice is None:
            size = tuple(float(v) for v in np.linalg.norm(m[:3, :3], axis=0))
            lattice = _LATTICES[key] = _surface_lattice(size, spacing)  # type: ignore[arg-type]
        out.append(lattice @ m[:3, :3].T + m[:3, 3])
    return np.concatenate(out)
