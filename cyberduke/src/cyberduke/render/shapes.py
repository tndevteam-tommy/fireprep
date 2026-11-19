"""Rounded unit shapes for the people: heads, limbs, hands, boots.

Every shape fills the unit cube (-0.5 .. 0.5 on each axis) the way a box part of the body
(``animation.rig``) does - the renderer scales it to the part's size exactly as it used
to scale a cube - so the collision body (``animation.skeleton``, a box per part) still
bounds what is drawn. What changes is what is inside the box: an arm is a tapering,
round-ended limb, a head is a skull with a narrower jaw, a boot has a flat sole, a heel
and a rounded toe cap.

A shape is (vertices, indices): ``vertices`` rows are x, y, z, normal x, y, z, u, v (the
layout both backends take), ``indices`` triangles (counter-clockwise from outside).
NumPy only - they are built once and cached, and tests can look at them.

Most of them are *lofts*: rings of a superellipse cross-section (``p`` = 2 a circle, higher
squarer) stacked along y, each ring scaled by a width profile, the ends rounded off
(``cap``: how much of the length each rounded end takes, ``q``: how round it is).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from cyberduke.animation import anatomy, hairdo

Mesh = tuple[np.ndarray, np.ndarray]
Profile = Callable[[float], float]


def _one(_: float) -> float:
    return 1.0


def _taper(bottom: float, top: float) -> Profile:
    return lambda t: bottom + (top - bottom) * t


def _superellipse(theta: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = np.cos(theta), np.sin(theta)
    e = 2.0 / p
    return np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e


def _smooth_normals(
    pos: np.ndarray, tris: np.ndarray, seam: dict[int, int] | None = None
) -> np.ndarray:
    """Vertex normals: the area-weighted sum of the normals of the triangles round each
    vertex (vertices duplicated along a texture seam share theirs)."""
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    face = np.cross(b - a, c - a)
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, tris[:, k], face)
    if seam:
        for i, j in seam.items():
            total = n[i] + n[j]
            n[i] = n[j] = total
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    return n / length


def _grid_triangles(rings: int, around: int) -> np.ndarray:
    """Two triangles per quad of a (rings x around+1) vertex grid, outward for a loft that
    goes up in y and round counter-clockwise seen from above."""
    tris = []
    cols = around + 1
    for j in range(rings - 1):
        for i in range(around):
            a = j * cols + i
            b = a + 1
            c = a + cols
            d = c + 1
            tris.append((a, c, b))
            tris.append((b, c, d))
    return np.array(tris, dtype=np.int64)


def _section(
    t: float,
    width: Profile,
    depth: Profile,
    cap: tuple[float, float],
    q: tuple[float, float],
    z_shift: Profile | None,
) -> tuple[float, float, float]:
    """(x half-width, z half-depth, z shift) of a loft's cross-section at height t."""
    s = 1.0
    c0, c1 = cap
    if c0 > 0.0 and t < c0:
        d = (c0 - t) / c0
        s = (max(0.0, 1.0 - d ** q[0])) ** (1.0 / q[0])
    elif c1 > 0.0 and t > 1.0 - c1:
        d = (t - (1.0 - c1)) / c1
        s = (max(0.0, 1.0 - d ** q[1])) ** (1.0 / q[1])
    shift = z_shift(t) if z_shift is not None else 0.0
    return 0.5 * width(t) * s, 0.5 * depth(t) * s, shift * s


def _finish(
    pos: np.ndarray, uv: np.ndarray, tris: np.ndarray, rings: int, around: int, closed=True
) -> np.ndarray:
    """Vertex rows (position, smooth normal, uv) of a grid of ``rings`` x ``around``+1
    (``closed``: all the way round, its first and last columns one seam)."""
    cols = around + 1
    seam = {j * cols: j * cols + around for j in range(rings)} if closed else None
    normals = _smooth_normals(pos, tris, seam)
    # the poles (rings shrunk to a point) share one normal, straight along the axis
    for j in (0, rings - 1):
        row = slice(j * cols, (j + 1) * cols)
        if np.ptp(pos[row, 0]) < 1e-9 and np.ptp(pos[row, 2]) < 1e-9:
            normals[row] = (0.0, -1.0 if j == 0 else 1.0, 0.0)
    out = np.zeros((len(pos), 8))
    out[:, :3] = pos
    out[:, 3:6] = normals
    out[:, 6:8] = uv
    return out


def loft(
    around: int = 12,
    rings: int = 10,
    p: float = 2.0,
    width: Profile = _one,
    depth: Profile | None = None,
    cap: tuple[float, float] = (0.15, 0.15),
    q: tuple[float, float] = (2.0, 2.0),
    z_shift: Profile | None = None,
    uv_around: float = 1.0,
) -> Mesh:
    """A closed, round-ended shape along y. ``width``/``depth``: the x / z scale of the
    cross-section along the length (t = 0 at the bottom, 1 at the top); ``cap`` the share
    of the length each end's rounding takes (bottom, top); ``q`` how round each end is
    (2 a dome, bigger flatter); ``z_shift`` moves the section forwards (a jaw, a calf)."""
    depth = depth or width
    around, rings = _res(around, rings)
    # the rings' heights: denser towards the rounded ends
    ts = [0.5 - 0.5 * math.cos(math.pi * j / (rings - 1)) for j in range(rings)]
    theta = np.linspace(0.0, 2.0 * math.pi, around + 1)
    cx, cz = _superellipse(theta, p)
    pos, uv = [], []
    for t in ts:
        wx, wz, shift = _section(t, width, depth, cap, q, z_shift)
        for k in range(around + 1):
            pos.append((cx[k] * wx, t - 0.5, cz[k] * wz + shift))
            uv.append((theta[k] / (2 * math.pi) * uv_around, t))
    tris = _grid_triangles(rings, around)
    out = _finish(np.array(pos), np.array(uv), tris, rings, around)
    return _fit(out), tris.astype(np.uint32)


_RESOLUTION = [1.0]  # (set while a coarser version of a shape is built: ``mesh(lo=True)``)


def _res(around: int, rings: int) -> tuple[int, int]:
    k = _RESOLUTION[0]
    if k >= 1.0:
        return around, rings
    return max(6, round(around * k)), max(4, round(rings * k))


def _fit(v: np.ndarray) -> np.ndarray:
    """Never outside the unit cube (rounding error, a shifted section)."""
    v[:, :3] = np.clip(v[:, :3], -0.5, 0.5)
    return v


def _compact(v: np.ndarray, tris: np.ndarray) -> Mesh:
    used = np.unique(tris)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[tris].astype(np.uint32)


# ----------------------------------------------------------------------------- the head
# (the head's shape is ``animation.anatomy``'s: the one description the rig and the face
# painter use too)
def head(around: int = 20, rings: int = 15) -> Mesh:
    """The skull: a jaw narrowing to the chin, a good high forehead, rounder over the face
    than at the back of the head (flatter there, as real heads are)."""
    around, rings = _res(around, rings)
    ts = [0.5 - 0.5 * math.cos(math.pi * j / (rings - 1)) for j in range(rings)]
    theta = np.linspace(0.0, 2.0 * math.pi, around + 1)
    pos, uv = [], []
    for t in ts:
        for th in theta:
            pos.append(anatomy.point(t, float(th)))
            uv.append((th / (2 * math.pi), t))
    tris = _grid_triangles(rings, around)
    out = _finish(np.array(pos), np.array(uv), tris, rings, around)
    return _fit(out), tris.astype(np.uint32)


FACE_FRONT = 0.18  # the face covers the head where its surface is this far forward (unit z)


def face() -> Mesh:
    """The front of the head (drawn in a box ``animation.rig.FACE_PROUD`` times the
    head's, so a hair's breadth proud of it), with the face texture laid on it flat from
    the front (u across, v up - as on the front of the old head cube)."""
    v, tris = head()
    front = v[:, 2] > FACE_FRONT
    tris = tris[front[tris].all(axis=1)]
    v, tris = _compact(v.copy(), tris)
    # (the face part's box is the head's, ``rig.FACE_PROUD`` times bigger: the same surface in
    # its unit cube stands that much proud of the head's)
    v[:, 6] = np.clip(v[:, 0] + 0.5, 0.0, 1.0)
    v[:, 7] = np.clip(v[:, 1] + 0.5, 0.0, 1.0)
    return _fit(v), tris


def hairline(forward: float) -> float:
    """Where the hair stops (unit y of the head) round the head (``anatomy.hairline``)."""
    return anatomy.hairline(forward)


def _head_point(t: float, theta: float) -> tuple[float, float, float]:
    return anatomy.point(t, theta)


# ----------------------------------------------------------------------------- hair
def _welded_normals(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Smooth normals, vertices in one place (a seam, a pole, a rim) sharing theirs."""
    key = np.round(pos / 1e-6).astype(np.int64)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    a, b, c = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
    face = np.cross(b - a, c - a)
    n = np.zeros((int(inverse.max()) + 1, 3))
    for k in range(3):
        np.add.at(n, inverse[tris[:, k]], face)
    n = n[inverse]
    length = np.linalg.norm(n, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    return n / length


def _in_box(pos: np.ndarray, uv: np.ndarray, tris: np.ndarray, mid, size) -> Mesh:
    """A mesh in metres (the head's frame) as a unit shape in the box (``mid``, ``size``)
    the rig draws it in: positions squeezed into the box, normals the other way."""
    mid, size = np.asarray(mid), np.asarray(size)
    normals = _welded_normals(pos, tris)
    v = np.zeros((len(pos), 8))
    v[:, :3] = (pos - mid) / size
    v[:, 3:6] = normals * size
    v[:, 3:6] /= np.maximum(np.linalg.norm(v[:, 3:6], axis=1, keepdims=True), 1e-12)
    v[:, 6:8] = uv
    return _fit(v), tris.astype(np.uint32)


def hair_piece(name: str) -> Mesh:
    """A piece of a hairstyle (``animation.hairdo``) in its own box (``hairdo.box``)."""
    pos, uv, tris = hairdo.raw(name)
    return _in_box(pos, uv, tris, *hairdo.box(name))


def stubble_cap(around: int = 28, rings: int = 9) -> Mesh:
    """Hair cropped to nothing (stubble, or a bare scalp): a skin a few millimetres over
    the skull from the natural hairline up, with an inner skin and a rim along the
    hairline - drawn in the crop's box (every hairstyle's cap is: ``hairdo.CAP``)."""
    around, rings = _res(around, rings)
    theta = np.linspace(0.0, 2.0 * math.pi, around + 1)
    hx, hy, hz = anatomy.HEAD_SIZE
    outer, inner, uv = [], [], []
    for j in range(rings):
        u = j / (rings - 1)
        for th in theta:
            t0 = hairline(math.sin(th)) + 0.5
            t = t0 + (1.0 - t0) * (1.0 - (1.0 - u) ** 1.6)
            x, y, z = _head_point(t, float(th))
            outer.append((x * hx * 1.012, y * hy * 1.006 + 0.0015, z * hz * 1.012))
            inner.append((x * hx * 0.99, y * hy * 0.99, z * hz * 0.99))
            uv.append((th / (2 * math.pi) * 3.0, u))
    n = len(outer)
    tris = _grid_triangles(rings, around)
    rim = []
    for k in range(around):
        rim.append((k, k + 1, k + 1 + n))
        rim.append((k, k + 1 + n, k + n))
    pos = np.concatenate((np.array(outer), np.array(inner)))
    alltris = np.concatenate((tris, tris[:, ::-1] + n, np.array(rim, dtype=np.int64)))
    uvs = np.concatenate((np.array(uv), np.array(uv)))
    return _in_box(pos, uvs, alltris, *hairdo.box(hairdo.CAP))


def visor_bounds() -> tuple[np.ndarray, np.ndarray]:
    """(middle, size) of the visor band, in the head's unit cube (``animation.rig`` puts the
    visor part's box there: ``VISOR_BOX``)."""
    v, _ = _visor_band()
    lo, hi = v[:, :3].min(axis=0), v[:, :3].max(axis=0)
    return (lo + hi) / 2.0, hi - lo


def visor() -> Mesh:
    """The visor band fitted to its own box (``visor_bounds``)."""
    v, tris = _visor_band()
    middle, size = visor_bounds()
    v = v.copy()
    v[:, :3] = (v[:, :3] - middle) / size
    v[:, 3:6] = v[:, 3:6] * size  # (normals: the inverse of the squeeze)
    v[:, 3:6] /= np.linalg.norm(v[:, 3:6], axis=1, keepdims=True)
    return _fit(v), tris


def _visor_band(around: int = 16, rings: int = 3) -> Mesh:
    """A wraparound visor across the eyes: a band of the head's own shape, a little proud
    of it, round the front half of the head - closed (an inner skin inside the head, rims
    top and bottom and at the ends). In the head's unit cube."""
    around, rings = _res(around, rings)
    h = anatomy.HEAD_SIZE[1]
    band = 0.13 * 0.38 / h  # (4.9 cm either side of the eyes)
    lo, hi = 0.5 + anatomy.EYE_LINE / h - band, 0.5 + anatomy.EYE_LINE / h + band
    theta = np.linspace(math.radians(-15.0), math.radians(195.0), around + 1)
    outer, inner, uv = [], [], []
    for j in range(rings):
        t = lo + (hi - lo) * j / (rings - 1)
        for th in theta:
            x, y, z = _head_point(t, th)
            outer.append((x * 1.06, y, z * 1.06))
            inner.append((x * 0.97, y, z * 0.97))
            uv.append((0.0, 0.0))
    n = len(outer)
    tris = _grid_triangles(rings, around)
    v_out = _finish(np.array(outer), np.array(uv), tris, rings, around)
    v_in = _finish(np.array(inner), np.array(uv), tris, rings, around)
    v_in[:, 3:6] *= -1.0
    cols = around + 1
    rim = []
    for row in (0, rings - 1):  # top and bottom edges
        for k in range(around):
            a, b = row * cols + k, row * cols + k + 1
            if row == 0:
                rim += [(a, b + n, b), (a, a + n, b + n)]
            else:
                rim += [(a, b, b + n), (a, b + n, a + n)]
    for k in (0, around):  # the ends
        for j in range(rings - 1):
            a, b = j * cols + k, (j + 1) * cols + k
            rim += (
                [(a, b, b + n), (a, b + n, a + n)] if k == 0 else [(a, b + n, b), (a, a + n, b + n)]
            )
    allv = np.concatenate((v_out, v_in))
    alltris = np.concatenate((tris, tris[:, ::-1] + n, np.array(rim, dtype=np.int64)))
    return allv, alltris.astype(np.uint32)


# ----------------------------------------------------------------------------- the body
def torso(around: int = 14, rings: int = 10) -> Mesh:
    """Shoulders square-ish and rounded on top, the chest the widest, a narrower waist."""

    def width(t: float) -> float:
        if t < 0.6:
            return 0.86 + 0.14 * math.sin(0.5 * math.pi * t / 0.6)
        return 1.0

    def depth(t: float) -> float:
        return 0.86 + 0.14 * math.sin(math.pi * min(1.0, 0.25 + t * 0.75))

    return loft(around, rings, p=2.7, width=width, depth=depth, cap=(0.06, 0.16), q=(4.0, 2.2))


def pelvis() -> Mesh:
    return loft(12, 6, p=2.6, width=_taper(0.9, 1.0), cap=(0.35, 0.2), q=(2.2, 3.0))


def rbox(p: float = 4.0, cap: float = 0.2) -> Mesh:
    """A box with its edges rounded off."""
    return loft(12, 6, p=p, cap=(cap, cap), q=(4.0, 4.0))


def limb(
    bottom: float = 0.86, bulge: float = 0.0, cap: float = 0.16, p: float = 2.2, around: int = 10
) -> Mesh:
    """An arm or a leg segment: round-ended, tapering from the top (the joint nearer the
    body) to the bottom; ``bulge`` swells it a little above the middle (a calf, a
    forearm's muscle)."""

    def width(t: float) -> float:
        base = bottom + (1.0 - bottom) * t
        swell = bulge * math.sin(math.pi * min(1.0, max(0.0, (t - 0.3) / 0.65)))
        return min(1.0, base + swell)

    return loft(around, 8, p=p, width=width, cap=(cap, cap), q=(2.0, 2.0))


def sphere(around: int = 10, rings: int = 7) -> Mesh:
    return loft(around, rings, p=2.0, cap=(0.5, 0.5), q=(2.0, 2.0))


def finger() -> Mesh:
    """A finger segment: a little capsule, thinner at the tip end (bottom)."""
    return loft(6, 5, p=2.0, width=_taper(0.84, 1.0), cap=(0.3, 0.3), q=(2.0, 2.0))


def palm() -> Mesh:
    """The palm: a thick, rounded slab (thickness along x, width along z, the fingers at
    the bottom), a touch narrower at the wrist (top)."""
    return loft(
        10, 6, p=3.2, width=_one, depth=lambda t: 1.0 - 0.12 * t, cap=(0.16, 0.2), q=(2.6, 2.4)
    )


def nose() -> Mesh:
    """Narrow at the bridge (top), fuller at the nostrils."""
    return loft(
        8, 6, p=2.2, width=_taper(1.0, 0.6), depth=_taper(1.0, 0.7), cap=(0.3, 0.25), q=(2.0, 2.0)
    )


def ear() -> Mesh:
    return loft(8, 6, p=2.0, width=_one, depth=_one, cap=(0.5, 0.5), q=(2.0, 2.0))


def neck() -> Mesh:
    return loft(
        10,
        4,
        p=2.0,
        width=_taper(1.0, 0.8),
        depth=_taper(1.0, 0.86),
        cap=(0.08, 0.08),
        q=(3.0, 3.0),
    )


# ----------------------------------------------------------------------------- the boot
def boot(around: int = 14, rings: int = 8) -> Mesh:
    """A boot: a flat sole, the heel and the shaft at the back (the full height of the
    box), the upper sloping down over the instep to a low, rounded toe cap at the front
    (+z), the top edges rounded off."""
    around, rings = _res(around, rings)
    theta = np.linspace(0.0, 2.0 * math.pi, around + 1)
    ox, oz = _superellipse(theta, 2.6)
    ox = ox * 0.5
    oz = oz * 0.5
    # a touch narrower at the heel than across the ball of the foot
    ox = ox * (0.9 + 0.1 * np.clip((oz + 0.5) / 0.7, 0.0, 1.0))

    def top(z: np.ndarray) -> np.ndarray:
        u = np.clip((z + 0.08) / 0.58, 0.0, 1.0)  # 0 behind the instep, 1 at the toe
        return 0.5 - 0.62 * (u * u * (3.0 - 2.0 * u))

    verts = []
    ts = (
        [0.0, 0.03, 0.3, 0.6, 0.78, 0.9, 0.97, 1.0]
        if rings >= 8
        else [0.0, 0.04, 0.5, 0.82, 0.95, 1.0]
    )
    for t in ts:
        # the sole's edge is bevelled a little; the top rounds over
        if t < 0.06:
            s = 0.96 + 0.04 * (t / 0.06)
        elif t > 0.75:
            d = (t - 0.75) / 0.25
            s = (max(0.0, 1.0 - d**2.2)) ** (1.0 / 2.2)
        else:
            s = 1.0
        yt = top(oz * s)
        for k in range(around + 1):
            y = -0.5 + t * (yt[k] + 0.5)
            verts.append((ox[k] * s, y, oz[k] * s, theta[k] / (2 * math.pi), t))
    data = np.array(verts)
    pos = data[:, :3]
    tris = _grid_triangles(len(ts), around)
    cols = around + 1
    # the sole: a fan across the bottom ring
    centre = len(pos)
    pos = np.vstack((pos, (0.0, -0.5, 0.0)))
    sole = [(centre, i + 1, i) for i in range(around)]
    tris = np.concatenate((tris, np.array(sole, dtype=np.int64)))
    seam = {j * cols: j * cols + around for j in range(len(ts))}
    normals = _smooth_normals(pos, tris, seam)
    # the sole is flat: its rim vertices keep their side normals, the centre points down
    normals[centre] = (0.0, -1.0, 0.0)
    out = np.zeros((len(pos), 8))
    out[:, :3] = pos
    out[:, 3:6] = normals
    out[: len(data), 6:8] = data[:, 3:5]
    out[centre, 6:8] = (0.5, 0.0)
    # the top ring has shrunk to (nearly) a point: close it with its own fan
    last = (len(ts) - 1) * cols
    top_centre = len(out)
    tc = out[last : last + around + 1, :3].mean(axis=0)
    out = np.vstack((out, (tc[0], tc[1], tc[2], 0.0, 1.0, 0.0, 0.5, 1.0)))
    lid = [(top_centre, last + i, last + i + 1) for i in range(around)]
    tris = np.concatenate((tris, np.array(lid, dtype=np.int64)))
    return _fit(out), tris.astype(np.uint32)


# ----------------------------------------------------------------------------- a fish
def fish_body(around: int = 16, rings: int = 14) -> Mesh:
    """A fish's body along y (the tail end at the bottom, the nose at the top; its height
    along z, its thickness along x): slim at the root of the tail, deepest a little ahead
    of the middle, a rounded head tapering to a blunt nose, the belly sagging a touch
    (towards +z). Textured round (``fish_skin``: u 0.25 the belly, 0.75 the back)."""

    def thick(t: float) -> float:
        return (
            0.3
            + 0.7 * math.sin(math.pi * min(1.0, (t / 0.62) ** 0.8) / 2.0)
            - 0.25 * max(0.0, t - 0.7)
        )

    def tall(t: float) -> float:
        rise = math.sin(math.pi * min(1.0, (t / 0.55) ** 0.75) / 2.0)
        return (0.26 + 0.64 * rise) * (1.0 - 0.3 * max(0.0, t - 0.72) / 0.28)

    return loft(
        around,
        rings,
        p=2.0,
        width=thick,
        depth=tall,
        cap=(0.03, 0.2),
        q=(3.0, 1.8),
        z_shift=lambda t: 0.05 * math.sin(math.pi * t),
    )


def fin() -> Mesh:
    """A fin: a flat paddle along y, narrow at its root (the bottom), broadening to a
    rounded edge (its thickness along x: drawn thin)."""
    return loft(8, 6, p=2.0, width=_taper(0.5, 1.0), depth=_taper(0.3, 1.0), cap=(0.05, 0.3))


# ----------------------------------------------------------------------------- registry
BUILDERS: dict[str, Callable[[], Mesh]] = {
    "head": head,
    "face": face,
    "stubble_cap": stubble_cap,
    **{name: (lambda name=name: hair_piece(name)) for name in hairdo.PIECES},
    "visor": visor,
    "nose": nose,
    "ear": ear,
    "neck": neck,
    "torso": torso,
    "pelvis": pelvis,
    "belt": lambda: rbox(3.4, 0.3),
    "rbox": rbox,
    "plate": lambda: rbox(5.0, 0.12),
    "sphere": sphere,
    "upper_arm": lambda: limb(0.84, 0.05, 0.16),
    "forearm": lambda: limb(0.7, 0.1, 0.15),
    "thigh": lambda: limb(0.8, 0.04, 0.14, p=2.1),
    "shin": lambda: limb(0.66, 0.16, 0.12, p=2.1),
    "boot": boot,
    "palm": palm,
    "finger": finger,
    "fish_body": fish_body,
    "fin": fin,
}
SHAPES = frozenset(BUILDERS)

_CACHE: dict[tuple[str, bool], Mesh] = {}
LO_RESOLUTION = 0.55  # the coarser meshes for people further off


def mesh(name: str, lo: bool = False) -> Mesh:
    """The (vertices, indices) of a named unit shape, built on first use - or its coarser
    version (``lo``) for somebody further off."""
    key = (name, lo)
    found = _CACHE.get(key)
    if found is None:
        _RESOLUTION[0] = LO_RESOLUTION if lo else 1.0
        try:
            v, i = BUILDERS[name]()
        finally:
            _RESOLUTION[0] = 1.0
        found = _CACHE[key] = (
            np.ascontiguousarray(v, dtype=np.float32),
            np.ascontiguousarray(i, dtype=np.uint32).reshape(-1),
        )
    return found


__all__ = ["BUILDERS", "SHAPES", "mesh"]
