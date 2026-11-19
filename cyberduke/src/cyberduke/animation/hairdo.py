"""Hairstyles as geometry: the shapes of the street's haircuts, built on the skull.

Every hairstyle (``animation.look.HAIRSTYLES``) is made of a few meshes grown on the
head's own surface (``animation.anatomy``), in metres from the middle of the head - the
frame the rig draws the head's decoration in:

* a **crop** (``cap``): the hair close over the skull from a natural hairline, a little
  fuller over the crown and the front;
* a **mohawk**: the sides shaved to stubble, a strip of hair down the middle of the
  skull and, rising from it, a row of fat gelled liberty spikes fanning back from the
  forehead to the nape - their points dyed;
* **spikes**: clumps of hair gelled into flat, tapering, curving blades all over the top
  of the head, a few hanging over the forehead as bangs - their tips dyed;
* an **undercut** (a side shave): one side shaved to stubble, the rest falling over to the
  jaw on the other side, a fringe swept across the forehead - dyed underneath;
* a **bob**: a helmet of straight hair falling from the crown to the jaw, cut level
  all round, a blunt fringe over the brows - its underside dyed a neon colour that shows
  as it moves;
* **long** hair: the same blunt fringe, locks framing the face to the chin and the rest
  falling straight to the shoulders (a hime cut), dyed underneath;
* a **top knot** (``bun``): the sides shaved, the top slicked back and tied at the crown
  in a short tail, a neon band round it;
* **twin tails**: the hair tied high either side of the head into two tails that hang
  down past the jaw, flaring and tapering, their ends dyed, neon ties at the roots;
* a **buzz** cut and a **shaved** head are the skull's own cap (stubble, bare scalp).

A mesh here is (positions (n, 3) in metres, uv (n, 2), triangles (m, 3)): texture ``u``
across the strands and ``v`` along them - the way the hair grows, which is the way the
strands of the hair texture run (``assets.faces``: "hair_strands"). ``render.shapes``
fits each into its unit box (``box``: where the rig puts it) and works out its normals.
NumPy only; every mesh is built once.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache

import numpy as np

from cyberduke.animation import anatomy

Raw = tuple[np.ndarray, np.ndarray, np.ndarray]  # positions (metres), uv, triangles
HX, HY, HZ = anatomy.HEAD_SIZE


# ============================================================================ the skull
def skull(t: float, theta: float) -> np.ndarray:
    """A point of the skull's surface in metres (``anatomy.point``: ``t`` 0 the chin ..
    1 the crown; ``theta`` 0 the left, 90 degrees the face, -90 the back)."""
    x, y, z = anatomy.point(t, theta)
    return np.array((x * HX, y * HY, z * HZ))


def normal(t: float, theta: float) -> np.ndarray:
    """The skull's outward normal there (straight up at the crown)."""
    if t > 0.995:
        return np.array((0.0, 1.0, 0.0))
    e = 1e-3
    dt = skull(min(1.0, t + e), theta) - skull(max(0.0, t - e), theta)
    dth = skull(t, theta + e) - skull(t, theta - e)
    n = np.cross(dth, dt)
    length = np.linalg.norm(n)
    if length < 1e-12:
        return np.array((0.0, 1.0, 0.0))
    n = n / length
    p = skull(t, theta)
    if float(np.dot(n, p - np.array((0.0, -0.02, 0.0)))) < 0.0:  # (outwards, always)
        n = -n
    return n


def _t_at_height(y: float, theta: float) -> float:
    """The skull's ``t`` at height ``y`` (metres) round ``theta``, on its upper half."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if skull(mid, theta)[1] < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _smooth(a: float, b: float, x: float) -> float:
    u = min(1.0, max(0.0, (x - a) / (b - a)))
    return u * u * (3.0 - 2.0 * u)


# ============================================================================ assembling
class _Mesh:
    """Collects grids of vertices into one mesh."""

    def __init__(self) -> None:
        self.pos: list[np.ndarray] = []
        self.uv: list[np.ndarray] = []
        self.tris: list[np.ndarray] = []
        self.count = 0

    def grid(self, pos: np.ndarray, uv: np.ndarray, away: np.ndarray) -> int:
        """A (rows, cols) grid of points, two triangles a quad, each turned to face away
        from ``away`` (a point per vertex: what the surface faces away from - the skull
        under the hair, a blade's spine). Returns its first index."""
        rows, cols = pos.shape[:2]
        base = self.count
        idx = np.arange(rows * cols).reshape(rows, cols)
        a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        tris = np.concatenate((np.stack((a, c, b), 1), np.stack((b, c, d), 1)))
        flat = pos.reshape(-1, 3)
        back = away.reshape(-1, 3)
        p0, p1, p2 = flat[tris[:, 0]], flat[tris[:, 1]], flat[tris[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        out = (p0 + p1 + p2) / 3.0 - (back[tris[:, 0]] + back[tris[:, 1]] + back[tris[:, 2]]) / 3.0
        wrong = np.einsum("ij,ij->i", n, out) < 0.0
        tris[wrong] = tris[wrong][:, ::-1]
        self.pos.append(flat)
        self.uv.append(uv.reshape(-1, 2))
        self.tris.append(tris + base)
        self.count += rows * cols
        return base

    def strip(self, a: np.ndarray, b: np.ndarray, uv_a, uv_b, away: np.ndarray) -> None:
        """A band of quads joining two rows of points (a rim between two skins), facing
        away from ``away`` (a point per point of the rows)."""
        self.grid(np.stack((a, b)), np.stack((uv_a, uv_b)), np.stack((away, away)))

    def build(self) -> Raw:
        return np.concatenate(self.pos), np.concatenate(self.uv), np.concatenate(self.tris)


def _uv_grid(rows: int, cols: int, u_repeat: float, v0: float = 0.0, v1: float = 1.0):
    u = np.linspace(0.0, u_repeat, cols)
    v = np.linspace(v0, v1, rows)
    return np.stack(np.meshgrid(u, v), axis=-1)


# ============================================================================ shells
def _two_skins(
    outer: np.ndarray,
    inner: np.ndarray,
    base: np.ndarray,
    u_repeat: float,
    rim_row: int,
    uv: np.ndarray | None = None,
) -> tuple[Raw, Raw]:
    """(the outside of the hair and its cut edge; its underside) as two meshes - the
    underside can be dyed another colour. ``base``: the skull points the hair grows from
    (the outside faces away from them, the underside towards them); ``rim_row``: which
    row is the cut edge (0 or -1)."""
    rows, cols = outer.shape[:2]
    if uv is None:
        uv = _uv_grid(rows, cols, u_repeat)
    top = _Mesh()
    top.grid(outer, uv, base)
    under = _Mesh()
    far = base + (outer - base) * 40.0  # (a point well outside: the underside faces in)
    under.grid(inner, uv, far)
    v = 1.0 if rim_row == -1 else 0.0
    uv_e = np.stack((np.linspace(0.0, u_repeat, cols), np.full(cols, v)), 1)
    uv_e2 = uv_e.copy()
    uv_e2[:, 1] = v + (0.04 if v == 0.0 else -0.04)
    # the cut edge (the hair's own colour), facing on, away from the row before it
    before = outer[1] if rim_row == 0 else outer[-2]
    top.strip(outer[rim_row], inner[rim_row], uv_e, uv_e2, before)
    return top.build(), under.build()


# ----------------------------------------------------------------------------- a crop
def crop(
    hairline: Callable[[float], float] | None = None,
    volume: Callable[[float, float], float] | None = None,
    sweep: float = 0.0,
    around: int = 28,
    rings: int = 10,
    combed: float | None = None,
) -> tuple[Raw, Raw]:
    """Hair over the skull from a hairline (``hairline(theta)``: its ``t``; the natural
    one by default) up to the crown, ``volume(u, theta)`` metres thick (``u``: 0 at the
    hairline, 1 at the crown), pushed ``sweep`` metres over towards +x at the crown (a
    side-swept top). The strands run from the hairline up to the crown - or, ``combed``,
    straight back over the head from the forehead, turned ``combed`` degrees towards +x
    (slicked back, swept over to one side)."""
    if hairline is None:

        def hairline(theta: float) -> float:
            return anatomy.hairline(math.sin(theta)) + 0.5

    if volume is None:

        def volume(u: float, theta: float) -> float:
            front = max(0.0, math.sin(theta))
            # thin at the hairline (it grows out of the skin), fuller over the crown and
            # at the front
            return 0.0018 + 0.017 * u**0.55 + 0.01 * front * u**0.5 * (1.0 - u)

    thetas = np.linspace(0.0, 2.0 * math.pi, around + 1)
    ts = np.zeros((rings, around + 1))
    for j, th in enumerate(thetas):
        t0 = hairline(float(th))
        for i in range(rings):
            u = i / (rings - 1)
            ts[i, j] = t0 + (1.0 - t0) * (1.0 - (1.0 - u) ** 1.6)

    outer = np.zeros((rings, around + 1, 3))
    inner = np.zeros((rings, around + 1, 3))
    base = np.zeros((rings, around + 1, 3))
    for i in range(rings):
        u = i / (rings - 1)
        shift = np.array((sweep * math.sin(math.pi * 0.5 * u), 0.0, 0.0))
        for j, th in enumerate(thetas):
            t = float(ts[i, j])
            p, n = skull(t, float(th)), normal(t, float(th))
            base[i, j] = p
            outer[i, j] = p + n * volume(u, float(th)) + shift
            inner[i, j] = p + n * 0.001
    for a in (outer, inner, base):
        a[:, -1] = a[:, 0]  # (the seam)
    uv = None
    if combed is not None:  # (the strands laid from the front to the back, seen from above)
        c, sn = math.cos(math.radians(combed)), math.sin(math.radians(combed))
        across = outer[..., 0] * c + outer[..., 2] * sn
        along = -outer[..., 2] * c + outer[..., 0] * sn
        uv = np.stack((across / 0.06, along / 0.32 + 0.5), axis=-1)
    return _two_skins(outer, inner, base, 6.0, rim_row=0, uv=uv)


# ----------------------------------------------------------------------------- a curtain
def curtain(
    cut: Callable[[float], float],
    thickness: Callable[[float, float], float],
    flare: float = 0.08,
    around: int = 40,
    rings: int = 16,
) -> tuple[Raw, Raw]:
    """Straight hair falling from the crown, over the skull down to its widest, then
    straight down to where it is cut (``cut(theta)``: the height of the cut, metres from
    the middle of the head) - a bob, a fringe, long hair. ``thickness(s, theta)``: how
    thick it is along the way (``s``: 0 the crown, 1 the cut); ``flare``: how far out it
    swings per metre of fall. The strands run down from the crown."""
    thetas = np.linspace(0.0, 2.0 * math.pi, around + 1)
    t_wide = 0.6  # (the skull's widest, round the temples)
    outer = np.zeros((rings, around + 1, 3))
    inner = np.zeros((rings, around + 1, 3))
    base = np.zeros((rings, around + 1, 3))
    for j, th in enumerate(thetas):
        th = float(th)
        y_cut = cut(th)
        # the way down this side of the head: over the skull, then hanging
        path, norms = [], []
        y_wide = float(skull(t_wide, th)[1])
        t_end = _t_at_height(y_cut, th) if y_cut > y_wide else t_wide
        for k in range(24):
            t = 1.0 - (1.0 - t_end) * k / 23.0
            path.append(skull(t, th))
            norms.append(normal(t, th))
        if y_cut < y_wide:
            wide = skull(t_wide, th)
            side = np.array((wide[0], 0.0, wide[2]))
            side = side / (np.linalg.norm(side) or 1.0)
            n_hang = side
            for k in range(1, 17):
                drop = (y_wide - y_cut) * k / 16.0
                curl = 0.014 * _smooth(0.8, 1.0, k / 16.0)  # (the ends turned under)
                path.append(wide + np.array((0.0, -drop, 0.0)) + side * (flare * drop - curl))
                norms.append(n_hang)
        path_a = np.array(path)
        norms_a = np.array(norms)
        # resample evenly along the way
        seg = np.linalg.norm(np.diff(path_a, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(seg)))
        total = arc[-1] or 1.0
        for i in range(rings):
            s = i / (rings - 1)
            a = s * total
            k = min(len(arc) - 2, int(np.searchsorted(arc, a, side="right") - 1))
            f = (a - arc[k]) / max(arc[k + 1] - arc[k], 1e-9)
            p = path_a[k] * (1 - f) + path_a[k + 1] * f
            n = norms_a[k] * (1 - f) + norms_a[k + 1] * f
            n = n / (np.linalg.norm(n) or 1.0)
            base[i, j] = p
            outer[i, j] = p + n * thickness(s, th)
            inner[i, j] = p + n * 0.005
    for a in (outer, inner, base):
        a[:, -1] = a[:, 0]  # (the seam)
    return _two_skins(outer, inner, base, 4.0, rim_row=-1)


BROW = anatomy.BROW_LINE + 0.014  # where a fringe is cut (just clear of the brows)


def _cut_bob(theta: float) -> float:
    """A bob: the fringe at the brows, the sides and back level with the jaw."""
    ahead = math.sin(theta)
    return BROW + (-0.1 - BROW) * (1.0 - _smooth(0.62, 0.8, ahead))


def _cut_long(theta: float) -> float:
    """A hime cut: the fringe at the brows, locks to the chin framing the face, the rest
    to the shoulders."""
    ahead = math.sin(theta)
    chin, shoulders = -0.15, -0.26
    y = shoulders + (chin - shoulders) * _smooth(0.1, 0.35, ahead)
    return y + (BROW - y) * _smooth(0.62, 0.8, ahead)


def _cut_side_shave(theta: float, towards: float) -> float:
    """A side shave: the hair falls to the jaw on the ``towards`` side (+1: +x), is shaved
    off on the other (cut just over the temple), slants across the back between them, and
    a fringe sweeps across the forehead from the short side down to the brow."""
    side = math.cos(theta) * towards  # (+1 the long side, -1 the shaved one)
    ahead = math.sin(theta)
    shaved, long_ = 0.135, -0.11
    y = shaved + (long_ - shaved) * _smooth(-0.45, 0.3, side)
    fringe = 0.145 + (BROW - 0.145) * _smooth(-0.6, 0.5, side)
    return y + (fringe - y) * _smooth(0.62, 0.82, ahead) if ahead > 0.0 else y


def _thick(crown: float, ears: float) -> Callable[[float, float], float]:
    def thickness(s: float, theta: float) -> float:
        side = 1.0 - abs(math.sin(theta))  # (over the ears: clear of them)
        return crown + (ears - crown) * _smooth(0.3, 0.6, s) * (0.5 + 0.5 * side) - 0.004 * s

    return thickness


# ============================================================================ blades
def _blade(
    m: _Mesh,
    spine: Callable[[float], np.ndarray],
    side: np.ndarray,
    width: Callable[[float], float],
    depth: Callable[[float], float],
    s0: float = 0.0,
    s1: float = 1.0,
    rings: int = 7,
    around: int = 8,
    u_repeat: float = 1.0,
) -> None:
    """A tapering bundle of hair along a curved ``spine(s)`` - a spike, a tail, a lock -
    its cross-section ``width(s)`` across ``side`` and ``depth(s)`` the other way, from
    ``s0`` to ``s1`` of its length (a tip dyed another colour is the same blade from
    further along). Closes to a point where it tapers to nothing."""
    pos = np.zeros((rings, around + 1, 3))
    uv = np.zeros((rings, around + 1, 2))
    middle = np.zeros((rings, around + 1, 3))
    for i in range(rings):
        s = s0 + (s1 - s0) * i / (rings - 1)
        p = spine(s)
        middle[i, :] = p
        e = 1e-3
        tangent = spine(min(1.0, s + e)) - spine(max(0.0, s - e))
        tangent = tangent / (np.linalg.norm(tangent) or 1.0)
        b = side - np.dot(side, tangent) * tangent
        b = b / (np.linalg.norm(b) or 1.0)
        n = np.cross(tangent, b)
        w, d = width(s) / 2.0, depth(s) / 2.0
        for k in range(around + 1):
            a = 2.0 * math.pi * k / around
            pos[i, k] = p + b * w * math.cos(a) + n * d * math.sin(a)
            uv[i, k] = (k / around * u_repeat, s)
    m.grid(pos, uv, middle)


# ----------------------------------------------------------------------------- a mohawk
def _strip_line(half_width: float) -> Callable[[float], float]:
    """A hairline for a strip of hair ``half_width`` either side of the middle of the
    skull (a mohawk's): for each way round the head, the ``t`` where the strip's edge is."""

    def hairline(theta: float) -> float:
        natural = anatomy.hairline(math.sin(theta)) + 0.5
        if abs(math.cos(theta)) < 1e-6:
            return natural
        lo, hi = natural, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if abs(float(skull(mid, theta)[0])) > half_width:
                lo = mid
            else:
                hi = mid
        return max(natural, (lo + hi) / 2.0)

    return hairline


# the mohawk's spikes: (how far along the middle of the skull they stand - by the length
# of it, 0 the front hairline, 1 the nape -, their length, how far they lean back). They
# used to be spaced by the skull's height rather than the length over it: four bunched
# over the forehead, three down the back and a bare strip over the crown between them,
# room for two more. Now nine stand evenly spaced from the forehead to the nape, the
# tallest over the crown (and every one as slender for its length as the others: the
# short one over the forehead came out a blunt stub, not a point like the rest).
CREST_FROM, CREST_TO, CREST_SPIKES = 0.02, 0.88, 9
CREST = tuple(
    (
        CREST_FROM + (CREST_TO - CREST_FROM) * i / (CREST_SPIKES - 1),
        0.15 + 0.05 * math.sin(math.pi * i / (CREST_SPIKES - 1)),
        -0.15 + 0.75 * i / (CREST_SPIKES - 1),
    )
    for i in range(CREST_SPIKES)
)


def midline_along(u: float) -> tuple[float, float]:
    """(y, z) of the middle of the top of the skull ``u`` of the way along it by length
    (``anatomy.midline`` goes by height, and so crosses the crown in a jump)."""
    ks, lengths = _MIDLINE
    return anatomy.midline(float(np.interp(u * lengths[-1], lengths, ks)))


def _midline_lengths() -> tuple[np.ndarray, np.ndarray]:
    ks = np.linspace(0.0, 1.0, 801)
    pts = np.array([anatomy.midline(float(k)) for k in ks])
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return ks, np.concatenate(([0.0], np.cumsum(steps)))


_MIDLINE = _midline_lengths()
CREST_SLENDER = (0.36, 0.33)  # a spike's width and depth at its root, for its length (round)


def _crest(part: str) -> Raw:
    """A mohawk: a strip of hair down the middle of the skull ("strip" - its outside;
    "strip_under"), and rising from it a row of fat gelled spikes fanning back from the
    forehead to the nape ("crest"), their points dyed ("crest_tips")."""
    if part in ("strip", "strip_under"):
        top, under = crop(hairline=_strip_line(0.035), volume=lambda u, th: 0.002 + 0.012 * u**0.5)
        return top if part == "strip" else under
    m = _Mesh()
    s0, s1 = (0.62, 1.0) if part == "crest_tips" else (0.0, 0.62)
    for u, length, lean in CREST:
        y, z = midline_along(u)
        foot = np.array((0.0, y, z))
        out = foot - np.array((0.0, -0.03, 0.0))
        out = out / np.linalg.norm(out)
        back = np.array((0.0, -0.2, -1.0))
        d = out + back * lean * 0.5
        d = d / np.linalg.norm(d)
        along = np.cross(np.array((1.0, 0.0, 0.0)), out)  # (along the middle of the skull)

        def spine(s, foot=foot - out * 0.01, d=d, n=length):
            return foot + d * n * s + np.array((0.0, -0.015, -0.02)) * s * s

        w0, d0 = (k_ * length for k_ in CREST_SLENDER)
        _blade(
            m, spine, along, lambda s, w0=w0: w0 * (1.0 - s) ** 1.1 + 0.002,
            lambda s, d0=d0: d0 * (1.0 - s) ** 0.9 + 0.002, s0, s1, rings=8, around=10,
        )  # fmt: skip
    return m.build()


# ----------------------------------------------------------------------------- spikes
# (t, theta in degrees, length, lean back, curl) of each clump of gelled spikes
SPIKES = (
    (0.99, 90.0, 0.13, 0.6, 0.3), (0.93, 60.0, 0.12, 0.4, 0.3), (0.93, 120.0, 0.12, 0.4, 0.3),
    (0.86, 30.0, 0.11, 0.2, 0.35), (0.86, 150.0, 0.11, 0.2, 0.35), (0.78, 5.0, 0.1, 0.1, 0.4),
    (0.78, 175.0, 0.1, 0.1, 0.4), (0.9, -40.0, 0.13, 0.8, 0.3), (0.9, -140.0, 0.13, 0.8, 0.3),
    (0.94, -90.0, 0.14, 1.0, 0.25), (0.78, -70.0, 0.12, 1.0, 0.3), (0.78, -110.0, 0.12, 1.0, 0.3),
    (0.72, -25.0, 0.1, 0.5, 0.4), (0.72, -155.0, 0.1, 0.5, 0.4), (0.97, -10.0, 0.12, 0.5, 0.3),
    (0.97, -170.0, 0.12, 0.5, 0.3), (0.7, 45.0, 0.09, 0.2, 0.4), (0.7, 135.0, 0.09, 0.2, 0.4),
    (0.66, -50.0, 0.1, 0.6, 0.4), (0.66, -130.0, 0.1, 0.6, 0.4),
)  # fmt: skip
BANGS = ((-0.07, 0.1), (-0.02, 0.115), (0.035, 0.11), (0.08, 0.095))  # (x, length): over the brow


def _spikes(tips: bool, flip: bool = False) -> Raw:
    """Clumps of gelled hair: fat at the root, flattened, tapering and curving to a point
    (the points: ``tips``) - all over the top of the head, and locks hanging over the
    forehead."""
    m = _Mesh()
    s0, s1 = (0.62, 1.0) if tips else (0.0, 0.62)
    k = -1.0 if flip else 1.0
    for t, theta_deg, length, lean, curl in SPIKES:
        th = math.radians(theta_deg)
        foot = skull(t, th)
        n = normal(t, th)
        up = n + np.array((0.0, 0.5, 0.0))
        up = up / np.linalg.norm(up)
        back = np.array((0.0, -0.2, -1.0)) * lean * 0.4
        bend = (n * 0.2 + np.array((0.0, -0.6, -0.3)) + np.array((0.2 * k, 0.0, 0.0))) * curl
        direction = up + back
        direction = direction / np.linalg.norm(direction)
        around_dir = np.array((-math.sin(th), 0.0, math.cos(th)))  # (round the head)

        def spine(s, foot=foot - n * 0.006, d=direction, b=bend, n_=length):
            return foot + d * n_ * s + b * n_ * s * s

        _blade(m, spine, around_dir, lambda s: 0.1 * (1.0 - s) ** 1.15 + 0.002,
               lambda s: 0.045 * (1.0 - s) ** 0.9 + 0.002, s0, s1, rings=8, around=10)  # fmt: skip
    for x, length in BANGS:  # locks over the forehead, pointing down and forward
        y = anatomy.HAIRLINE_FRONT + 0.02
        foot = np.array((x * 0.9, y, anatomy.surface_z(x * 0.9, y) - 0.008))
        d = np.array((x * 2.2 * k, -1.0, 0.45))
        d = d / np.linalg.norm(d)

        def spine(s, foot=foot, d=d, n_=length):
            return foot + d * n_ * s + np.array((0.0, 0.02, 0.03)) * s * s

        _blade(m, spine, np.array((1.0, 0.0, 0.0)), lambda s: 0.06 * (1.0 - s) ** 1.1 + 0.002,
               lambda s: 0.02 * (1.0 - s) ** 0.8 + 0.002, s0, s1, rings=7, around=10)  # fmt: skip
    return m.build()


# ----------------------------------------------------------------------------- tails
def _tail_spine(root: np.ndarray, out: np.ndarray, length: float):
    """A tail: out from its tie, then hanging - curving down under its own weight."""

    def spine(s: float) -> np.ndarray:
        fall = s * s
        return (
            root
            + out * length * 0.35 * math.sin(min(1.0, s * 1.6) * math.pi / 2.0)
            + np.array((0.0, -length * fall, -0.04 * s))
        )

    return spine


def _tail_width(s: float) -> float:
    """Tied tight at the root, full a third of the way down, tapering to the ends."""
    full = math.sin(math.pi * min(1.0, s * 1.5)) ** 0.6 * (1.0 - s) ** 0.55
    tied = 1.0 - 0.5 * math.exp(-(((s - 0.09) / 0.05) ** 2))
    return (0.035 + 0.065 * full) * tied + 0.002


def _twin_tails(part: str, flip: bool = False) -> Raw:
    """``part``: "tails" (the tails down to their dyed ends), "ends", or "ties"."""
    m = _Mesh()
    for sgn in (-1.0, 1.0):
        root = np.array((sgn * 0.165, 0.1, -0.05))
        out = np.array((sgn * 0.9, 0.25, -0.35))
        out = out / np.linalg.norm(out)
        spine = _tail_spine(root, out, 0.36)
        side = np.array((0.0, 0.0, 1.0))
        if part == "ties":
            _blade(m, spine, side, lambda s: 0.042, lambda s: 0.038, 0.06, 0.12, rings=3, around=12)
            continue
        s0, s1 = (0.0, 0.72) if part == "tails" else (0.72, 1.0)
        _blade(m, spine, side, _tail_width, lambda s: _tail_width(s) * 0.85, s0, s1,
               rings=9, around=10, u_repeat=2.0)  # fmt: skip
    del flip
    return m.build()


def _top_knot(part: str) -> Raw:
    """The hair drawn up to the crown and tied, the end standing up and curving back
    (``part``: "knot", or its "tie" - a neon band round it)."""
    m = _Mesh()
    root = skull(0.97, -math.pi / 2.0) + np.array((0.0, -0.005, 0.0))
    d = np.array((0.0, 0.85, -0.45))
    d = d / np.linalg.norm(d)

    def spine(s: float) -> np.ndarray:
        return root + d * 0.18 * s + np.array((0.0, -0.08, -0.05)) * s * s

    side = np.array((1.0, 0.0, 0.0))
    if part == "tie":
        _blade(m, spine, side, lambda s: 0.062, lambda s: 0.058, 0.1, 0.19, rings=3, around=12)
    else:

        def width(s: float) -> float:  # (gathered at the tie, fanning out, to a point)
            gather = 1.0 - 0.45 * math.exp(-(((s - 0.15) / 0.08) ** 2))
            full = math.sin(math.pi * min(1.0, s * 1.2))
            return (0.07 + 0.04 * full) * (1.0 - s) ** 0.7 * gather + 0.002

        _blade(m, spine, side, width, lambda s: width(s) * 0.85, 0.0, 1.0, rings=12, around=12,
               u_repeat=2.0)  # fmt: skip
    return m.build()


# ============================================================================ the styles
def _shaved_sides(theta: float) -> float:
    """Where a shaved-sides cut's top starts (``t``): at the natural hairline in front,
    high over the temples and the ears, across the back of the crown."""
    ahead = math.sin(theta)
    front = anatomy.hairline(1.0) + 0.5
    return front + (0.8 - front) * (1.0 - _smooth(0.55, 0.95, ahead))


@cache
def raw(name: str) -> Raw:
    """The mesh of one piece of a hairstyle (``PIECES``), in metres from the middle of
    the head."""
    if name in ("crop", "crop_under"):
        top, under = crop()
        return top if name == "crop" else under
    if name.startswith("undercut"):  # (a side shave)
        towards = 1.0 if "_l" in name else -1.0
        top, under = curtain(
            lambda th, w=towards: _cut_side_shave(th, w), _thick(0.028, 0.04), flare=0.06
        )
        return under if name.endswith("_under") else top
    if name.startswith("slick"):
        top, under = crop(
            hairline=_shaved_sides, volume=lambda u, th: 0.0018 + 0.011 * u**0.5, combed=0.0
        )
        return under if name.endswith("_under") else top
    if name.startswith("bob"):
        top, under = curtain(_cut_bob, _thick(0.026, 0.04), flare=0.05)
        return under if name.endswith("_under") else top
    if name.startswith("long"):
        top, under = curtain(_cut_long, _thick(0.026, 0.04), flare=0.1, rings=20)
        return under if name.endswith("_under") else top
    if name in ("crest", "crest_tips", "strip", "strip_under"):
        return _crest(name)
    if name.startswith("spikes"):
        return _spikes(tips="tips" in name, flip="_b" in name)
    if name.startswith("twintails"):
        return _twin_tails(name.split("_")[1] if "_" in name else "tails")
    if name == "knot":
        return _top_knot("knot")
    if name == "knot_tie":
        return _top_knot("tie")
    raise KeyError(name)


PIECES = (
    "crop", "crop_under", "undercut_l", "undercut_l_under", "undercut_r", "undercut_r_under",
    "slick", "slick_under", "bob", "bob_under", "long", "long_under", "strip", "strip_under",
    "crest", "crest_tips",
    "spikes_a", "spikes_a_tips", "spikes_b", "spikes_b_tips", "twintails", "twintails_ends",
    "twintails_ties", "knot", "knot_tie",
)  # fmt: skip


@cache
def box(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(middle, size) of a piece's box, metres from the middle of the head: where the rig
    puts it, the box its unit mesh is scaled into."""
    pos, _, _ = raw(name)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    mid = (lo + hi) / 2.0
    size = np.maximum(hi - lo, 1e-4)
    return (
        (float(mid[0]), float(mid[1]), float(mid[2])),
        (float(size[0]), float(size[1]), float(size[2])),
    )


# Each hairstyle: the skull's own cap (``CAP``: the shape and material of the part of the
# head the collision body knows as "hair" - a crop of hair, stubble, a bare scalp) and the
# pieces over it, each with its material - "hair" (the colour), "hair2" (the dyed
# colour), "neon" (a tie)
CAP = "crop"  # (the cap's box: the crop's - every style's cap is drawn in it)
STYLES: dict[str, tuple[tuple[str, str], tuple[tuple[str, str], ...]]] = {
    "cap": (("crop", "hair"), ()),
    "buzz": (("stubble_cap", "stubble"), ()),
    "shaved": (("stubble_cap", "scalp"), ()),
    "mohawk": (
        ("stubble_cap", "stubble"),
        (("strip", "hair"), ("strip_under", "hair"), ("crest", "hair"), ("crest_tips", "hair2")),
    ),
    "spikes": (("crop", "hair"), (("spikes_{v}", "hair"), ("spikes_{v}_tips", "hair2"))),
    "undercut": (
        ("stubble_cap", "stubble"),
        (("undercut_{s}", "hair"), ("undercut_{s}_under", "hair2")),
    ),
    "bob": (("stubble_cap", "hair"), (("bob", "hair"), ("bob_under", "hair2"))),
    "long": (("stubble_cap", "hair"), (("long", "hair"), ("long_under", "hair2"))),
    "bun": (
        ("stubble_cap", "stubble"),
        (("slick", "hair"), ("slick_under", "hair"), ("knot", "hair2"), ("knot_tie", "neon")),
    ),  # fmt: skip
    "twintails": (
        ("crop", "hair"),
        (("twintails", "hair"), ("twintails_ends", "hair2"), ("twintails_ties", "neon")),
    ),  # fmt: skip
}


def pieces(style: str, seed: int = 0) -> tuple[tuple[str, str], tuple[tuple[str, str], ...]]:
    """((the cap's shape, its material), (the pieces over it and their materials)) of a
    hairstyle - the variant for ``seed`` (which way an undercut is swept, which way the
    spikes curl)."""
    cap, parts = STYLES.get(style, STYLES["cap"])
    s = "l" if seed % 2 else "r"
    v = "a" if (seed // 2) % 2 else "b"
    return cap, tuple((name.format(s=s, v=v), mat) for name, mat in parts)


__all__ = ["CAP", "PIECES", "STYLES", "box", "pieces", "raw"]
