"""Finds z-fighting in a level before anybody sees it flicker.

Z-fighting is what happens when two surfaces lie in the same plane, facing the same way,
over the same patch: the depth buffer cannot say which is in front, so from one frame (and
one pixel) to the next it picks either - the ceiling's texture showing through the outside
wall of a building in stripes, the street's asphalt through a flat's lino, a mirror
"glitching" in and out of the tiles it is stuck to. It is never one prop's fault alone:
it is two pieces of geometry given exactly the same coordinate for a face (one box's
outer face flush with another's, two slabs at ``y = 0``), and nothing a renderer can fix.

The audit draws a level the way the game does - every prop of every sector, the light
fixtures, the doors, the computers, cameras, vending machines, access panels and things
lying about, the fog and the pools of light - into a ``GeometryRecorder``: a
``DrawContext`` that keeps the model matrix like a backend and turns every unit cube,
cylinder, soft disc and baked light-pool triangle into world-space faces instead of
pixels. Then it looks for pairs of faces from *different* draws that are

* coplanar - the same plane to within ``gap`` (1 mm by default: far closer than any
  depth buffer can tell apart at a distance), facing the same way (or either of them a
  see-through, two-sided surface: fog, a pool of light);
* overlapping - by more than ``min_area`` (a square centimetre);
* and visible - the patch is not buried inside some other solid box, and is not the
  underside of something lying on the floor.

Run it on the shipped levels with ``python -m cyberduke.render.audit`` (``--level`` for
one of them, ``--gap`` to be stricter); ``tests/test_zfighting.py`` runs it on every
level and fails on any hit. No OpenGL is touched: it is NumPy over the draw calls.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cyberduke.render.context import RGB, RGBA, DrawContext, Vec3

GAP = 0.001  # metres: closer than this, two same-facing faces z-fight
MIN_AREA = 1e-4  # square metres of overlap worth reporting (1 cm^2)
FLOOR_LEVEL = 0.05  # faces facing down below this are undersides on the floor: unseen
BURY_PROBE = 0.004  # metres in front of a patch tested for being inside something solid


# ============================================================================ the recorder
@dataclass
class Face:
    """One flat convex polygon that was drawn, in world space."""

    corners: np.ndarray  # (n, 3)
    normal: np.ndarray  # unit, pointing out of the front
    draw: int  # which primitive it came from (faces of one primitive never z-fight)
    label: str  # what drew it
    translucent: bool = False  # blended (fog, a light pool): seen from both sides
    texture: int | None = None

    @property
    def centre(self) -> np.ndarray:
        return self.corners.mean(axis=0)


@dataclass
class Solid:
    """An opaque box that was drawn: the world -> unit-cube transform, to tell whether a
    point is buried inside it."""

    inverse: np.ndarray  # (4, 4)
    draw: int
    label: str


def _rotation(angle_deg: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    t = 1.0 - c
    m = np.identity(4)
    m[:3, :3] = (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )
    return m


_CUBE_FACES = (  # (normal axis, sign): the unit cube's six faces
    (0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0),
)  # fmt: skip


def _cube_face(axis: int, sign: float) -> np.ndarray:
    """Corners (4, 3) of one face of the unit cube (-0.5..0.5), in order round it."""
    u, v = [a for a in range(3) if a != axis]
    pts = []
    for a, b in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        p = [0.0, 0.0, 0.0]
        p[axis] = 0.5 * sign
        p[u], p[v] = a, b
        pts.append(p)
    return np.array(pts)


class GeometryRecorder(DrawContext):
    """A ``DrawContext`` that draws nothing: it records every face (``faces``) and every
    opaque box (``solids``) in world space. ``label`` is stamped on what is drawn next."""

    def __init__(self) -> None:
        self.m = np.identity(4)
        self._stack: list[np.ndarray] = []
        self.faces: list[Face] = []
        self.solids: list[Solid] = []
        self.label = ""
        self._alpha = 1.0
        self._depth_writes = True
        self._draws = 0

    # ------------------------------------------------------------------ frame / matrices
    def begin_frame(self, eye: Vec3, target: Vec3) -> None:
        pass

    def end_frame(self) -> None:
        pass

    def push(self) -> None:
        self._stack.append(self.m.copy())

    def pop(self) -> None:
        self.m = self._stack.pop()

    def translate(self, x: float, y: float, z: float) -> None:
        t = np.identity(4)
        t[:3, 3] = (x, y, z)
        self.m = self.m @ t

    def rotate(self, angle_deg: float, x: float, y: float, z: float) -> None:
        self.m = self.m @ _rotation(angle_deg, x, y, z)

    def scale(self, x: float, y: float, z: float) -> None:
        self.m = self.m @ np.diag((x, y, z, 1.0))

    # ------------------------------------------------------------------ material
    def color(self, r: float, g: float, b: float) -> None:
        pass

    def emission(self, rgb: Sequence[float] | None) -> None:
        pass

    def specular(self, rgb: Sequence[float] | None, shininess: float = 0.0) -> None:
        pass

    def alpha(self, a: float) -> None:
        self._alpha = a

    def depth_mask(self, enabled: bool) -> None:
        self._depth_writes = enabled

    @property
    def _see_through(self) -> bool:
        return self._alpha < 1.0 or not self._depth_writes

    # ------------------------------------------------------------------ primitives
    def _world(self, pts: np.ndarray) -> np.ndarray:
        h = np.hstack((pts, np.ones((len(pts), 1))))
        return (self.m @ h.T).T[:, :3]

    def _add(self, corners: np.ndarray, inside: np.ndarray | None, translucent: bool, tex) -> None:
        """A face; its normal points away from ``inside`` (a point inside the solid it
        bounds) - or, for a lone surface, along the right-handed winding."""
        a, b = corners[1] - corners[0], corners[2] - corners[0]
        n = np.cross(a, b)
        length = float(np.linalg.norm(n))
        if length < 1e-12:
            return
        n = n / length
        if inside is not None and float(np.dot(n, corners.mean(axis=0) - inside)) < 0.0:
            n = -n
        self.faces.append(Face(corners, n, self._draws, self.label, translucent, tex))

    def cube(self, tex_id: int | None = None) -> None:
        self._draws += 1
        centre = self._world(np.zeros((1, 3)))[0]
        for axis, sign in _CUBE_FACES:
            self._add(self._world(_cube_face(axis, sign)), centre, self._see_through, tex_id)
        if not self._see_through and abs(np.linalg.det(self.m[:3, :3])) > 1e-12:
            self.solids.append(Solid(np.linalg.inv(self.m), self._draws, self.label))

    def cylinder(
        self, r_bot: float, r_top: float, h: float, slices: int = 10, tex_id: int | None = None
    ) -> None:
        self._draws += 1
        centre = self._world(np.array([(0.0, h / 2.0, 0.0)]))[0]
        ring = [
            (math.cos(2.0 * math.pi * k / slices), math.sin(2.0 * math.pi * k / slices))
            for k in range(slices)
        ]
        see = self._see_through
        for y, r in ((0.0, r_bot), (h, r_top)):
            if r > 1e-6:
                cap = np.array([(r * c, y, r * s) for c, s in ring])
                self._add(self._world(cap), centre, see, tex_id)
        for k in range(slices):
            (c0, s0), (c1, s1) = ring[k], ring[(k + 1) % slices]
            quad = np.array(
                [(r_bot * c0, 0.0, r_bot * s0), (r_bot * c1, 0.0, r_bot * s1),
                 (r_top * c1, h, r_top * s1), (r_top * c0, h, r_top * s0)]
            )  # fmt: skip
            if r_bot < 1e-6 and r_top < 1e-6:
                continue
            self._add(self._world(quad if r_bot > 1e-6 else quad[1:]), centre, see, tex_id)

    def lines(self, segments: Sequence[tuple[Vec3, Vec3]], rgba: RGBA, width: float = 1.0) -> None:
        pass  # (lines have no area)

    def soft_disc(self, rgba: RGBA, slices: int = 24, inner: float = 0.0) -> None:
        self._draws += 1
        ring = np.array(
            [
                (math.cos(2.0 * math.pi * k / slices), 0.0, math.sin(2.0 * math.pi * k / slices))
                for k in range(slices)
            ]
        )
        self._add(self._world(ring), None, True, None)

    def glow_mesh(self, key: Any, rgba: RGBA, build: Callable[[], Any]) -> None:
        tris = np.asarray(build(), dtype=float)
        for k in range(0, len(tris) - 2, 3):
            self._draws += 1
            self._add(tris[k : k + 3, :3], None, True, None)

    def cached(self, key: Any, build: Callable[[], None]) -> None:
        build()

    def invalidate(self, key: Any = None) -> None:
        pass

    # ------------------------------------------------------------------ (no overlay)
    def begin_overlay(self, width: int, height: int) -> None:
        pass

    def end_overlay(self) -> None:
        pass

    def rect(self, x: float, y: float, w: float, h: float, rgba: RGBA) -> None:
        pass

    def rect_outline(
        self, x: float, y: float, w: float, h: float, rgba: RGBA, width: float = 2.0
    ) -> None:
        pass

    def image(self, tex_id: int, x: float, y: float, w: float, h: float) -> None:
        pass

    def upload_texture_rgba(self, data: bytes, w: int, h: int) -> int:
        return 0

    def delete_texture(self, tex_id: int) -> None:
        pass

    @property
    def name(self) -> str:
        return "audit"

    def set_global_ambient(self, rgb: RGB) -> None:
        pass


# ============================================================================ the search
@dataclass(frozen=True)
class ZFight:
    """Two faces that fight over the same patch."""

    a: str
    b: str
    point: tuple[float, float, float]  # the middle of the shared patch
    normal: tuple[float, float, float]
    area: float  # square metres
    gap: float  # how far apart the planes are

    def __str__(self) -> str:
        x, y, z = self.point
        nx, ny, nz = self.normal
        return (
            f"z-fight at ({x:.3f}, {y:.3f}, {z:.3f}) facing ({nx:+.2f}, {ny:+.2f}, {nz:+.2f}): "
            f"{self.a!r} / {self.b!r} - {self.area * 1e4:.1f} cm^2, {self.gap * 1000:.2f} mm apart"
        )


def _basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array((1.0, 0.0, 0.0)) if abs(n[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _ccw(poly: np.ndarray) -> np.ndarray:
    area = 0.5 * float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    )
    return poly if area >= 0.0 else poly[::-1]


def _area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return abs(
        0.5
        * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
    )


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: the part of convex polygon ``subject`` inside convex ``clipper``
    (both 2D, counter-clockwise)."""
    out = subject
    for k in range(len(clipper)):
        if len(out) == 0:
            break
        a, b = clipper[k], clipper[(k + 1) % len(clipper)]
        edge = b - a

        def inside(p, a=a, edge=edge):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12

        src, out = out, []
        for i in range(len(src)):
            cur, prev = src[i], src[i - 1]
            cin, pin = inside(cur), inside(prev)
            if cin != pin:
                d = cur - prev
                denom = edge[0] * d[1] - edge[1] * d[0]
                if abs(denom) > 1e-15:
                    t = (edge[0] * (prev[1] - a[1]) - edge[1] * (prev[0] - a[0])) / -denom
                    out.append(prev + d * t)
            if cin:
                out.append(cur)
        out = np.array(out) if out else np.zeros((0, 2))
    return out


def _buried(point: np.ndarray, solids: list[Solid], skip: set[int]) -> bool:
    """Whether ``point`` is inside one of the solid boxes (other than ``skip``)."""
    if not solids:
        return False
    h = np.append(point, 1.0)
    inv = np.stack([s.inverse for s in solids])
    local = inv @ h
    inside = np.all(np.abs(local[:, :3]) < 0.5 - 1e-6, axis=1)
    return any(inside[k] and solids[k].draw not in skip for k in np.flatnonzero(inside))


def find_zfights(
    faces: list[Face],
    solids: Iterable[Solid] = (),
    gap: float = GAP,
    min_area: float = MIN_AREA,
) -> list[ZFight]:
    """Every pair of faces (from different draws) that share a plane and a patch and can
    be seen (see the module doc)."""
    solids = list(solids)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    planes = []
    for i, f in enumerate(faces):
        n = f.normal
        # one bucket per direction (a see-through face goes in with either facing)
        for sign in (1.0, -1.0) if f.translucent else (1.0,):
            key = tuple(int(round(c * 64.0)) for c in n * sign)
            buckets.setdefault(key, []).append(i)  # type: ignore[arg-type]
        planes.append(float(np.dot(n, f.corners[0])))
    found: dict[tuple[int, int], ZFight] = {}
    for members in buckets.values():
        if len(members) < 2:
            continue
        ref = faces[members[0]].normal
        ds = sorted((float(np.dot(ref, faces[i].corners[0])), i) for i in members)
        for a in range(len(ds)):
            da, i = ds[a]
            for b in range(a + 1, len(ds)):
                db, j = ds[b]
                if db - da > gap:
                    break
                fi, fj = faces[i], faces[j]
                if fi.draw == fj.draw or (fi.translucent and fj.translucent):
                    continue
                key = (min(i, j), max(i, j))
                if key in found:
                    continue
                if abs(float(np.dot(fi.normal, fj.normal))) < 0.9999:
                    continue
                hit = _overlap(fi, fj, min_area)
                if hit is None:
                    continue
                centre, area = hit
                n = fi.normal if not fi.translucent else fj.normal
                if n[1] < -0.9 and centre[1] < FLOOR_LEVEL:
                    continue  # the underside of something on the floor
                probe = centre + n * BURY_PROBE
                if _buried(probe, solids, {fi.draw, fj.draw}):
                    continue
                found[key] = ZFight(
                    fi.label or "?", fj.label or "?", tuple(round(float(c), 4) for c in centre),
                    tuple(float(c) for c in n), area, abs(db - da),
                )  # fmt: skip
    return sorted(found.values(), key=lambda z: (-z.area, z.point))


def _overlap(fi: Face, fj: Face, min_area: float) -> tuple[np.ndarray, float] | None:
    """(centre, area) of the patch two coplanar faces share, if bigger than ``min_area``."""
    lo = np.maximum(fi.corners.min(axis=0), fj.corners.min(axis=0))
    hi = np.minimum(fi.corners.max(axis=0), fj.corners.max(axis=0))
    if np.any(hi - lo < -GAP * 2):
        return None
    u, v = _basis(fi.normal)
    pi = _ccw(np.stack((fi.corners @ u, fi.corners @ v), axis=1))
    pj = _ccw(np.stack((fj.corners @ u, fj.corners @ v), axis=1))
    inter = _clip(pi, pj)
    area = _area(inter)
    if area <= min_area:
        return None
    c2 = inter.mean(axis=0)
    plane = float(np.dot(fi.normal, fi.corners[0]))
    centre = u * c2[0] + v * c2[1] + fi.normal * plane
    return centre, area


# ============================================================================ a level
def record_level(level) -> GeometryRecorder:
    """Draws everything static in ``level`` (a ``LevelData``) into a recorder, each draw
    labelled with what it is: the props (by their labels), light fixtures, doors (shut),
    computers, cameras, vending machines, access panels, things to pick up and things
    lying about."""
    from cyberduke.assets.textures import available_textures
    from cyberduke.render.entity_renderers import (
        draw_access_panel,
        draw_pickup,
        draw_security_camera,
        draw_terminal,
        draw_vending_machine,
    )
    from cyberduke.render.items import draw_loose_items
    from cyberduke.render.lighting import draw_light_fixture
    from cyberduke.render.props import draw_prop
    from cyberduke.render.sectors import draw_door
    from cyberduke.simulation import Simulation

    textures = {name: k + 1 for k, name in enumerate(available_textures())}
    for slot in ("shirt", "pants", "flesh"):
        textures.setdefault(slot, len(textures) + 1)
    sim = Simulation(level)
    ctx = GeometryRecorder()
    for sector in sim.room.sectors:
        for prop in sector.props:
            ctx.label = prop.label or f"{prop.kind} in {sector.name}"
            draw_prop(ctx, prop, textures)
            ctx.alpha(1.0)
        for light in sector.lights:
            ctx.label = f"light fixture at ({light.x:.1f}, {light.y:.1f}, {light.z:.1f})"
            draw_light_fixture(ctx, light, textures)
    for door_id, door in sim.room.doors.items():
        ctx.label = f"door {door_id}"
        draw_door(ctx, door, textures)
    for t in sim.terminals:
        ctx.label = f"terminal {t.terminal_id}"
        draw_terminal(ctx, t, textures)
    for cam in sim.cameras:
        ctx.label = f"camera {cam.camera_id}"
        draw_security_camera(ctx, cam, [])
    for vm in sim.vending_machines:
        ctx.label = f"vending machine {getattr(vm, 'machine_id', '')}"
        draw_vending_machine(ctx, vm, textures)
    for panel in getattr(sim, "panels", ()):
        ctx.label = f"access panel {panel.panel_id}"
        draw_access_panel(ctx, panel, textures)
    for pickup in getattr(sim, "pickups", ()):
        ctx.label = f"pickup {pickup.item_id}"
        draw_pickup(ctx, pickup, textures)
    for item in sim.items:
        ctx.label = f"{item.name} lying about"
        draw_loose_items(ctx, [item], textures)
    ctx.depth_mask(True)
    return ctx


def audit_level(level, gap: float = GAP, min_area: float = MIN_AREA) -> list[ZFight]:
    ctx = record_level(level)
    return find_zfights(ctx.faces, ctx.solids, gap, min_area)


def main(argv: list[str] | None = None) -> int:
    from cyberduke.levels import LEVELS

    parser = argparse.ArgumentParser(
        prog="python -m cyberduke.render.audit",
        description="Looks for z-fighting (coplanar, overlapping, visible faces) in levels.",
    )
    parser.add_argument("--level", choices=sorted(LEVELS), action="append")
    parser.add_argument("--gap", type=float, default=GAP, help="plane tolerance in metres")
    parser.add_argument("--min-area", type=float, default=MIN_AREA, help="square metres")
    args = parser.parse_args(argv)
    total = 0
    for key in args.level or sorted(LEVELS):
        level = LEVELS[key]()
        hits = audit_level(level, args.gap, args.min_area)
        total += len(hits)
        print(f"{key} ({level.name}): {len(hits)} z-fighting patch(es)")
        for hit in hits:
            print("  " + str(hit))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
