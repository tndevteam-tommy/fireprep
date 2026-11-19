"""
Firestarter - Procedural Low-Poly Mesh Generators
Stylised, painterly low-poly geometry (DESIGN 1.2: "avoids blocky/voxel designs").

Vertex layout (12 floats): position(3) normal(3) uv(2) colour(4).
The colour alpha channel carries per-vertex metadata:
  * vegetation: 1 - wind-sway weight (1 = rigid trunk base, 0 = free branch tip)
  * fuel pieces: 0 on end-grain caps (shader draws growth rings), 1 on bark
"""

import math
from typing import Dict, Tuple, List, Callable, Optional, Sequence

import numpy as np

Color = Tuple[float, float, float, float]
CARD_FOLIAGE = True          # leaf / needle spray cards (False: the old solid low-poly clumps)


class MeshData:
    def __init__(self, vertices: np.ndarray, indices: np.ndarray):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1)
        self.indices = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
        self.vertex_count = len(self.vertices) // 12
        self.index_count = len(self.indices)

    def as_array(self) -> np.ndarray:
        return self.vertices.reshape(-1, 12)

    def transformed(self, m: np.ndarray, color_mul: Optional[Sequence[float]] = None) -> "MeshData":
        v = self.as_array().copy()
        p = np.c_[v[:, 0:3], np.ones(len(v))] @ m.T
        nm = np.linalg.inv(m[:3, :3]).T
        n = v[:, 3:6] @ nm.T
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        v[:, 0:3] = p[:, :3]
        v[:, 3:6] = n
        if color_mul is not None:
            v[:, 8:12] *= np.array(color_mul, dtype=np.float32)
        return MeshData(v, self.indices.copy())


def _pack(pos, nrm, uv, col) -> np.ndarray:
    n = len(pos)
    out = np.zeros((n, 12), dtype=np.float32)
    out[:, 0:3] = pos
    out[:, 3:6] = nrm
    out[:, 6:8] = uv
    out[:, 8:12] = col
    return out


def drop_degenerate(m: MeshData, eps: float = 1e-12) -> MeshData:
    """Removes zero-area triangles (collapsed cone tips, pointed leaf ends)."""
    if m.index_count == 0:
        return m
    v = m.as_array()[:, 0:3]
    tri = m.indices.reshape(-1, 3)
    area = np.linalg.norm(np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]]), axis=1)
    keep = area > eps
    if keep.all():
        return m
    return MeshData(m.as_array(), tri[keep].reshape(-1))


def fix_winding(m: MeshData) -> MeshData:
    """Makes every triangle counter-clockwise with respect to its vertex normals."""
    v = m.as_array()
    tri = m.indices.reshape(-1, 3).copy()
    P, N = v[:, 0:3], v[:, 3:6]
    g = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    flip = np.einsum("ij,ij->i", g, N[tri].sum(axis=1)) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return MeshData(v, tri.reshape(-1))


def merge(meshes: List[MeshData]) -> MeshData:
    verts, inds, base = [], [], 0
    for m in meshes:
        if m is None or m.vertex_count == 0:
            continue
        verts.append(m.as_array())
        inds.append(m.indices + base)
        base += m.vertex_count
    if not verts:
        return MeshData(np.zeros((0, 12)), np.zeros(0))
    return MeshData(np.concatenate(verts), np.concatenate(inds))


def _m_translate(x, y, z):
    m = np.identity(4)
    m[:3, 3] = (x, y, z)
    return m


def _m_scale(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])


def _m_rot_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def _m_rot_x(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def _m_rot_z(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def _m_align(d):
    d = np.asarray(d, dtype=float)
    d = d / max(1e-9, np.linalg.norm(d))
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = float(np.dot(y, d))
    m = np.identity(4)
    if np.linalg.norm(v) < 1e-8:
        if c < 0:
            m[1, 1] = -1
            m[0, 0] = -1
        return m
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    m[:3, :3] = np.identity(3) + vx + vx @ vx / (1.0 + c)
    return m


# ------------------------------------------------------------------------------------ primitives
def create_box_mesh(dimensions=(1.0, 1.0, 1.0), color: Color = (0.55, 0.42, 0.30, 1.0)) -> MeshData:
    dx, dy, dz = [d * 0.5 for d in dimensions]
    faces = [
        ((0, 0, 1), [(-dx, -dy, dz), (dx, -dy, dz), (dx, dy, dz), (-dx, dy, dz)]),
        ((0, 0, -1), [(dx, -dy, -dz), (-dx, -dy, -dz), (-dx, dy, -dz), (dx, dy, -dz)]),
        ((-1, 0, 0), [(-dx, -dy, -dz), (-dx, -dy, dz), (-dx, dy, dz), (-dx, dy, -dz)]),
        ((1, 0, 0), [(dx, -dy, dz), (dx, -dy, -dz), (dx, dy, -dz), (dx, dy, dz)]),
        ((0, 1, 0), [(-dx, dy, dz), (dx, dy, dz), (dx, dy, -dz), (-dx, dy, -dz)]),
        ((0, -1, 0), [(-dx, -dy, -dz), (dx, -dy, -dz), (dx, -dy, dz), (-dx, -dy, dz)]),
    ]
    pos, nrm, uv, inds = [], [], [], []
    for fi, (n, corners) in enumerate(faces):
        for c, t in zip(corners, [(0, 0), (1, 0), (1, 1), (0, 1)]):
            pos.append(c)
            nrm.append(n)
            uv.append(t)
        b = fi * 4
        inds += [b, b + 1, b + 2, b, b + 2, b + 3]
    col = np.tile(color, (len(pos), 1))
    return MeshData(_pack(np.array(pos), np.array(nrm), np.array(uv), col), np.array(inds))


def create_frustum(r_bottom: float, r_top: float, height: float, segments: int = 10, rings: int = 1,
                   color: Color = (0.5, 0.4, 0.3, 1.0), color_top: Optional[Color] = None,
                   caps: bool = True, cap_alpha: Optional[float] = None, centered: bool = False,
                   alpha_bottom: float = None, alpha_top: float = None, jitter: float = 0.0,
                   rng: Optional[np.random.Generator] = None) -> MeshData:
    """Tapered cylinder along +Y (0..height, or centred).  UV.y runs 0..1 along the length."""
    rng = rng or np.random.default_rng(1)
    y0 = -height * 0.5 if centered else 0.0
    color_top = color_top or color
    th = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    pos, nrm, uv, col = [], [], [], []
    slope = (r_bottom - r_top) / max(1e-6, height)
    for ri in range(rings + 1):
        t = ri / rings
        r = r_bottom + (r_top - r_bottom) * t
        y = y0 + height * t
        c = np.array(color) * (1 - t) + np.array(color_top) * t
        if alpha_bottom is not None:
            c[3] = alpha_bottom + (alpha_top - alpha_bottom) * t
        jit = rng.uniform(-jitter, jitter, segments + 1) if jitter > 0 else np.zeros(segments + 1)
        jit[-1] = jit[0]
        for si in range(segments + 1):
            rr = r * (1.0 + jit[si])
            pos.append((rr * math.cos(th[si]), y, rr * math.sin(th[si])))
            n = np.array([math.cos(th[si]), slope, math.sin(th[si])])
            nrm.append(n / np.linalg.norm(n))
            uv.append((si / segments, t))
            col.append(c)
    inds = []
    row = segments + 1
    for ri in range(rings):
        for si in range(segments):
            a = ri * row + si
            b = a + row
            inds += [a, b, a + 1, a + 1, b, b + 1]
    mesh = MeshData(_pack(np.array(pos), np.array(nrm), np.array(uv), np.array(col)), np.array(inds))
    if min(r_bottom, r_top) <= 1e-6:
        mesh = drop_degenerate(mesh)
    if not caps:
        return mesh
    parts = [mesh]
    ca = cap_alpha if cap_alpha is not None else color[3]
    for end, r, y, n in ((0, r_bottom, y0, -1.0), (1, r_top, y0 + height, 1.0)):
        if r <= 1e-5:
            continue
        cc = list(color if end == 0 else color_top)
        cc[3] = ca
        p = [(0.0, y, 0.0)] + [(r * math.cos(a), y, r * math.sin(a)) for a in th[:-1]]
        q = [(0.5, float(end))] + [(0.5 + 0.5 * math.cos(a), float(end)) for a in th[:-1]]
        ci = []
        for si in range(segments):
            a, b = 1 + si, 1 + (si + 1) % segments
            ci += [0, b, a] if n > 0 else [0, a, b]
        parts.append(MeshData(_pack(np.array(p), np.tile((0, n, 0), (len(p), 1)), np.array(q), np.tile(cc, (len(p), 1))), np.array(ci)))
    return merge(parts)


def create_cylinder_mesh(radius: float = 0.05, length: float = 0.5, segments: int = 12,
                         color: Color = (0.55, 0.40, 0.25, 1.0)) -> MeshData:
    return create_frustum(radius, radius, length, segments, 1, color, centered=True)


def create_cone_mesh(radius: float = 0.5, height: float = 1.0, segments: int = 10,
                     color: Color = (0.18, 0.38, 0.18, 1.0)) -> MeshData:
    return create_frustum(radius, 0.0, height, segments, 1, color)


def vertex_normals(P: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of an indexed triangle mesh."""
    fn = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, tris[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def create_stone(seed: int = 0, color: Color = (0.40, 0.39, 0.37, 1.0), flatten: float = 0.62,
                 roughness: float = 0.22, faceted: bool = True) -> MeshData:
    """A weathered field stone: multi-octave displaced icosphere with a flat-ish bed, true
    (geometric) normals and optional faceting so it reads as rock rather than an egg."""
    base = create_icosphere(2, 1.0, color)
    arr = base.as_array().copy()
    V = arr[:, 0:3].astype(float)
    rng = np.random.default_rng(seed + 4242)
    disp = np.zeros(len(V))
    amp = 1.0
    for octave in range(4):
        for _ in range(3):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            disp += np.sin(V @ d * rng.uniform(1.5, 3.0) * (1.9 ** octave) + rng.uniform(0, 6.28)) * amp
        amp *= 0.5
    disp /= 4.5
    # a few planar "cleavage" cuts make it look split from bedrock
    for _ in range(3):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        cut = rng.uniform(0.55, 0.8)
        proj = V @ d
        disp = np.where(proj > cut, disp - (proj - cut) * 1.2, disp)
    P = V * (1.0 + roughness * disp)[:, None]
    P[:, 1] *= flatten
    P[:, 1] = np.where(P[:, 1] < -0.25 * flatten, -0.25 * flatten + (P[:, 1] + 0.25 * flatten) * 0.3, P[:, 1])
    # colour variation: darker underside, mottled minerals
    shade = 0.86 + 0.14 * np.clip(P[:, 1] / flatten + 0.5, 0.0, 1.0) + rng.uniform(-0.05, 0.05, len(P))
    arr[:, 8:11] = np.array(color[:3])[None, :] * shade[:, None]
    tris = base.indices.reshape(-1, 3)
    if faceted:
        Pf = P[tris].reshape(-1, 3)
        fn = np.cross(P[tris[:, 1]] - P[tris[:, 0]], P[tris[:, 2]] - P[tris[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        # soften the facets a little toward the smooth normal
        sm = vertex_normals(P, tris)[tris].reshape(-1, 3)
        Nf = np.repeat(fn, 3, axis=0) * 0.65 + sm * 0.35
        Nf /= np.linalg.norm(Nf, axis=1, keepdims=True)
        out = np.repeat(arr[tris.reshape(-1)][:, None, :], 1, axis=1).reshape(-1, 12).copy()
        out[:, 0:3] = Pf
        out[:, 3:6] = Nf
        return MeshData(out, np.arange(len(out)))
    arr[:, 0:3] = P
    arr[:, 3:6] = vertex_normals(P, tris)
    return MeshData(arr, base.indices)


def create_icosphere(subdiv: int = 1, radius: float = 1.0, color: Color = (0.5, 0.5, 0.5, 1.0),
                     displace: float = 0.0, seed: int = 0, flatten: float = 1.0,
                     normal_bias: Optional[np.ndarray] = None) -> MeshData:
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    cache = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            p = verts[a] + verts[b]
            verts.append(p / np.linalg.norm(p))
            cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(subdiv):
        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    V = np.array(verts)
    rng = np.random.default_rng(seed)
    if displace > 0:
        # smooth-ish noise: combine a few random directional waves
        disp = np.zeros(len(V))
        for _ in range(5):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            disp += np.sin(V @ d * rng.uniform(2.0, 5.0) + rng.uniform(0, 6.28)) * rng.uniform(0.3, 1.0)
        disp = disp / 5.0
        V = V * (1.0 + displace * disp)[:, None]
    N = V / np.linalg.norm(V, axis=1, keepdims=True)
    if normal_bias is not None:
        N = N + normal_bias[None, :]
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    P = V * radius
    P[:, 1] *= flatten
    uv = np.c_[np.arctan2(N[:, 2], N[:, 0]) / (2 * np.pi) + 0.5, N[:, 1] * 0.5 + 0.5]
    col = np.tile(color, (len(P), 1))
    return MeshData(_pack(P, N, uv, col), np.array(f).reshape(-1))


def create_quad_mesh():
    """Fullscreen quad (legacy helper)."""
    verts = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1], dtype=np.float32)
    return verts, np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)


# ------------------------------------------------------------------------------------ terrain
def create_terrain_mesh(size: float = 130.0, resolution: int = 40,
                        elevation_fn: Optional[Callable] = None, mask_fn: Optional[Callable] = None,
                        hole: float = 0.0, drop: float = 0.0) -> MeshData:
    half = size * 0.5
    xs = np.linspace(-half, half, resolution)
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    if elevation_fn is None:
        H = np.zeros_like(X)
    else:
        H = np.asarray(elevation_fn(X, Z), dtype=float)
    H = H - drop
    dx = xs[1] - xs[0]
    Hp = np.pad(H, 1, mode="edge")
    nx = Hp[:-2, 1:-1] - Hp[2:, 1:-1]
    nz = Hp[1:-1, :-2] - Hp[1:-1, 2:]
    N = np.stack([nx, np.full_like(H, 2.0 * dx), nz], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    P = np.stack([X, H, Z], axis=-1).reshape(-1, 3)
    UV = np.stack([X / size + 0.5, Z / size + 0.5], axis=-1).reshape(-1, 2)
    if mask_fn is not None:
        C = mask_fn(X, Z).reshape(-1, 4)
    else:
        C = np.tile((0.5, 0.0, 0.0, 0.0), (len(P), 1))
    i = np.arange(resolution - 1)
    I, J = np.meshgrid(i, i, indexing="ij")
    a = (I * resolution + J).ravel()
    b = ((I + 1) * resolution + J).ravel()
    c = (I * resolution + J + 1).ravel()
    d = ((I + 1) * resolution + J + 1).ravel()
    if hole > 0:
        cx = (X[:-1, :-1] + X[1:, 1:]).ravel() * 0.5
        cz = (Z[:-1, :-1] + Z[1:, 1:]).ravel() * 0.5
        keep = (np.abs(cx) > hole) | (np.abs(cz) > hole)
        a, b, c, d = a[keep], b[keep], c[keep], d[keep]
    inds = np.stack([a, c, b, b, c, d], axis=1).ravel()
    return MeshData(_pack(P, N.reshape(-1, 3), UV, C), inds)


# ------------------------------------------------------------------------------------ vegetation
def _tier_skirt(y_top, y_edge, radius, n, color, rng, droop=0.15, underside=0.35, ao_top=1.0):
    """A jagged conifer branch tier: upper cone + flatter underside (closed)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 1)
    r = radius * (1.0 + rng.uniform(-0.22, 0.12, n))
    tips = np.stack([r * np.cos(th), y_edge - droop * radius * rng.uniform(0.6, 1.4, n), r * np.sin(th)], axis=1)
    inner = np.stack([0.35 * r * np.cos(th + np.pi / n), np.full(n, y_edge + (y_top - y_edge) * 0.25), 0.35 * r * np.sin(th + np.pi / n)], axis=1)
    top = np.array([[0.0, y_top, 0.0]])
    under = np.array([[0.0, y_edge + (y_top - y_edge) * underside - radius * 0.1, 0.0]])
    P = np.concatenate([top, tips, inner, under])
    C = np.array([0.0, (y_top + y_edge) * 0.5 + radius * 0.3, 0.0])
    Nrm = P - C
    Nrm[:, 1] += radius * 0.6
    Nrm /= np.linalg.norm(Nrm, axis=1, keepdims=True)
    col = np.tile(color, (len(P), 1)).astype(float)
    # baked AO: inner / underside darker, tips brighter; alpha = 1 - sway weight
    col[0, :3] *= 0.85
    col[1:n + 1, :3] *= 1.12
    col[n + 1:2 * n + 1, :3] *= 0.7
    col[-1, :3] *= 0.45
    col[:, 3] = 0.6
    col[1:n + 1, 3] = 0.15
    inds = []
    for i in range(n):
        j = (i + 1) % n
        ti, tj = 1 + i, 1 + j
        mi = 1 + n + i
        inds += [0, tj, mi, 0, mi, ti, ti, mi, tj]
        inds += [2 * n + 1, ti, tj]
    uv = np.zeros((len(P), 2))
    return MeshData(_pack(P, Nrm, uv, col), np.array(inds))


# ------------------------------------------------------------------------------------ foliage cards
ATLAS_CELLS = {"oak": 0, "birch": 1, "pine": 2, "cedar": 3}


def _atlas_uv(cell: int):
    cx, cy = cell % 2, cell // 2
    u0, u1 = cx * 0.5 + 0.004, (cx + 1) * 0.5 - 0.004
    v0, v1 = 1.0 - (cy + 1) * 0.5 + 0.004, 1.0 - cy * 0.5 - 0.004
    return u0, v0, u1, v1


class CardBuilder:
    """Accumulates alpha-cut foliage cards (quads textured with a leaf-spray atlas cell).

    Each card's normals blend the outward direction of the crown with the card's own normal,
    so thousands of flat cards shade like one soft, volumetric canopy.  Vertex alpha holds
    the sway weight convention of OBJECT_VS (alpha = rigidity)."""

    def __init__(self):
        self.pos: List[np.ndarray] = []
        self.nrm: List[np.ndarray] = []
        self.uv: List[Tuple[float, float]] = []
        self.col: List[np.ndarray] = []
        self.idx: List[int] = []

    def card(self, base: np.ndarray, up: np.ndarray, normal: np.ndarray, w: float, h: float, cell: int,
             color, outward: np.ndarray, rigidity: float = 0.15, blend: float = 0.65, centered: bool = False):
        up = up / max(1e-9, np.linalg.norm(up))
        n = normal - up * float(np.dot(normal, up))
        if np.linalg.norm(n) < 1e-6:
            n = np.cross(up, np.array([0.3, 0.1, 0.9]))
        n = n / np.linalg.norm(n)
        right = np.cross(up, n)
        u0, v0, u1, v1 = _atlas_uv(cell)
        b = base - up * (h * 0.5 if centered else 0.0)
        corners = [(b - right * w * 0.5, (u0, v0)), (b + right * w * 0.5, (u1, v0)),
                   (b + right * w * 0.5 + up * h, (u1, v1)), (b - right * w * 0.5 + up * h, (u0, v1))]
        o = outward / max(1e-9, np.linalg.norm(outward))
        shade_n = o * blend + n * (1.0 - blend) * (1.0 if np.dot(n, o) >= 0 else -1.0)
        shade_n = shade_n / max(1e-9, np.linalg.norm(shade_n))
        k = len(self.pos)
        c = np.array(color, dtype=float)
        for j, (p, t) in enumerate(corners):
            self.pos.append(p)
            self.nrm.append(shade_n)
            self.uv.append(t)
            cc = c.copy()
            cc[3] = rigidity if j >= 2 else min(1.0, rigidity + 0.25)
            self.col.append(cc)
        self.idx += [k, k + 1, k + 2, k, k + 2, k + 3]

    def mesh(self) -> MeshData:
        if not self.pos:
            return MeshData(np.zeros((0, 12)), np.zeros(0))
        return MeshData(_pack(np.array(self.pos), np.array(self.nrm), np.array(self.uv), np.array(self.col)), np.array(self.idx))


def _rand_unit(rng) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def create_conifer_cards(kind: str = "pine", seed: int = 0, height: float = 15.0, lod: int = 0) -> MeshData:
    """Whorled branches of needle sprays (crossed card pairs) - lod 1 = distant, fewer cards."""
    rng = np.random.default_rng(seed + (0 if lod == 0 else 5000))
    cb = CardBuilder()
    if kind == "pine":
        cell, start, max_r, whorls, per = ATLAS_CELLS["pine"], 0.36, 0.27, 13, 5
        leaf = np.array([0.055, 0.105, 0.05, 1.0])
    else:
        cell, start, max_r, whorls, per = ATLAS_CELLS["cedar"], 0.1, 0.2, 17, 6
        leaf = np.array([0.05, 0.10, 0.055, 1.0])
    if lod:
        whorls, per = max(5, whorls // 2), max(3, per - 2)
    for wi in range(whorls):
        t = wi / (whorls - 1)
        y = height * (start + (0.95 - start) * t)
        rad = height * max_r * (1.0 - t) ** (0.85 if kind == "pine" else 1.0) + 0.35
        yaw0 = rng.uniform(0, 2 * math.pi)
        for bi in range(per):
            a = yaw0 + bi * 2 * math.pi / per + rng.uniform(-0.3, 0.3)
            out = np.array([math.cos(a), 0.0, math.sin(a)])
            droop = rng.uniform(-0.25, 0.1) if kind == "pine" else rng.uniform(-0.45, -0.1)
            d = out + np.array([0.0, droop + 0.25 * t, 0.0])
            d /= np.linalg.norm(d)
            L = rad * rng.uniform(0.85, 1.1)
            n_spray = (4 if L > 1.6 else (3 if L > 0.9 else 2)) if not lod else 1
            for k in range(n_spray):
                f = (k + 0.6) / (n_spray + 0.1) if not lod else 0.45
                p = np.array([0.0, y, 0.0]) + d * L * f * 0.8
                seg_len = L * (0.9 if not lod else 1.3) / max(1, n_spray) + 0.55
                w = seg_len * (0.95 if kind == "pine" else 1.05)
                ao = 0.55 + 0.45 * min(1.0, f + 0.2) * (0.7 + 0.3 * t)
                col = leaf.copy()
                col[:3] *= ao * rng.uniform(0.9, 1.1)
                outward = out * 0.8 + np.array([0.0, 0.55, 0.0])
                flat_n = np.array([0.0, 1.0, 0.0]) + _rand_unit(rng) * 0.35
                cb.card(p - d * seg_len * 0.2, d, flat_n, w, seg_len, cell, col, outward, 0.12, centered=False)
                # crossed card so the spray reads from the side as well
                side_n = np.cross(d, np.array([0.0, 1.0, 0.0]))
                cb.card(p - d * seg_len * 0.2, d, side_n + _rand_unit(rng) * 0.2, w * 0.8, seg_len, cell, col, outward, 0.12)
    # leader tuft
    for k in range(3 if not lod else 2):
        a = k * math.pi / 3
        col = leaf.copy()
        col[:3] *= 1.05
        cb.card(np.array([0.0, height * 0.9, 0.0]), np.array([0.0, 1.0, 0.0]),
                np.array([math.cos(a), 0.0, math.sin(a)]), 1.0, height * 0.12, cell, col, np.array([0.0, 1.0, 0.0]), 0.1)
    return cb.mesh()


def create_broadleaf_cards(kind: str = "oak", seed: int = 0, height: float = 14.0, branch_tips=None,
                           lod: int = 0) -> MeshData:
    """A rounded deciduous crown of leaf sprays clustered around the branch tips."""
    rng = np.random.default_rng(seed + 900 + (0 if lod == 0 else 5000))
    cb = CardBuilder()
    cell = ATLAS_CELLS["oak" if kind == "oak" else "birch"]
    if kind == "oak":
        leaf = np.array([0.075, 0.12, 0.035, 1.0])
        crown_c, rx, ry, n_cards, size = np.array([0.0, height * 0.68, 0.0]), height * 0.34, height * 0.25, 230, 2.5
    else:
        leaf = np.array([0.10, 0.15, 0.04, 1.0])
        crown_c, rx, ry, n_cards, size = np.array([0.0, height * 0.70, 0.0]), height * 0.24, height * 0.29, 200, 1.9
    if lod:
        n_cards, size = n_cards // 3, size * 1.5
    anchors = [np.asarray(b, dtype=float) for b in (branch_tips or [])]
    for i in range(n_cards):
        if anchors and i % 3 == 0:
            c = anchors[i % len(anchors)] + rng.normal(0, 1.0, 3) * np.array([rx, ry, rx]) * 0.18
        else:
            # stratified points in an ellipsoidal shell (leaves live on the outside of a crown)
            u = _rand_unit(rng)
            r = rng.uniform(0.55, 1.0) ** 0.5
            c = crown_c + u * np.array([rx, ry, rx]) * r
            c[1] = max(c[1], crown_c[1] - ry * 0.75)
        rel = (c - crown_c) / np.array([rx, ry, rx])
        shell = min(1.0, float(np.linalg.norm(rel)))
        outward = rel / max(1e-6, np.linalg.norm(rel)) + np.array([0.0, 0.25, 0.0])
        up = outward * 0.6 + np.array([0.0, 0.5, 0.0]) + _rand_unit(rng) * 0.6
        normal = _rand_unit(rng) + outward * 0.5
        s = size * rng.uniform(0.8, 1.2) * (height / 14.0)
        ao = (0.45 + 0.55 * shell) * (0.8 + 0.2 * max(-1.0, min(1.0, rel[1])))
        col = leaf.copy()
        col[:3] *= ao * rng.uniform(0.88, 1.12)
        cb.card(c, up, normal, s, s, cell, col, outward, 0.15, blend=0.7, centered=True)
    return cb.mesh()


def create_bush_cards(seed: int = 0, color: Color = (0.07, 0.12, 0.04, 1.0)) -> MeshData:
    rng = np.random.default_rng(seed + 300)
    cb = CardBuilder()
    center = np.array([0.0, 0.45, 0.0])
    for i in range(22):
        u = _rand_unit(rng)
        u[1] = abs(u[1]) * 0.8 + 0.1
        c = center + u * np.array([0.75, 0.42, 0.75]) * rng.uniform(0.4, 1.0)
        c[1] = max(0.15, c[1])
        col = np.array(color, dtype=float)
        col[:3] *= (0.6 + 0.4 * min(1.0, c[1] / 0.8)) * rng.uniform(0.85, 1.15)
        s = rng.uniform(0.55, 0.85)
        cb.card(c, u * 0.5 + np.array([0.0, 0.6, 0.0]), _rand_unit(rng), s, s, ATLAS_CELLS["oak"], col,
                c - np.array([0.0, 0.2, 0.0]), 0.4, blend=0.7, centered=True)
    return cb.mesh()


def create_limb_foliage(kind: str = "pine", seed: int = 0) -> MeshData:
    """Foliage of one branch of a felled tree, in branch space: the branch runs along +Y
    from 0 (where it leaves the trunk) to 1 (its tip); scale uniformly by the limb length.
    Conifers carry flat needle sprays along the whole outer branch, broadleaves a loose
    cluster of leaf sprays around the outer half."""
    rng = np.random.default_rng(seed + 4400)
    cb = CardBuilder()
    axis = np.array([0.0, 1.0, 0.0])
    if kind in ("pine", "cedar"):
        cell = ATLAS_CELLS[kind]
        leaf = np.array([0.055, 0.105, 0.05, 1.0]) if kind == "pine" else np.array([0.05, 0.10, 0.055, 1.0])
        for k in range(13):
            f = 0.15 + 0.85 * (k + rng.uniform(0.0, 0.8)) / 13.0
            side = 1.0 if k % 2 else -1.0
            # side shoots leave the branch at ~45 degrees, in a roughly flat spray
            d = axis * 0.75 + np.array([side * 0.65, 0.0, rng.uniform(-0.25, 0.25)])
            d /= np.linalg.norm(d)
            L = 0.42 * (1.15 - 0.55 * f) * rng.uniform(0.8, 1.15)
            p0 = axis * f
            col = leaf.copy()
            col[:3] *= rng.uniform(0.85, 1.12) * (0.75 + 0.25 * f)
            flat_n = np.array([0.0, 0.0, 1.0]) + _rand_unit(rng) * 0.3
            cb.card(p0, d, flat_n, L * 0.85, L, cell, col, flat_n + d * 0.3, 0.3)
            cb.card(p0, d, np.cross(d, flat_n), L * 0.6, L, cell, col, flat_n + d * 0.3, 0.3)
        # the tip spray
        col = leaf.copy()
        cb.card(axis * 0.9, axis, np.array([0.0, 0.0, 1.0]), 0.22, 0.3, cell, col, axis, 0.3)
        cb.card(axis * 0.9, axis, np.array([1.0, 0.0, 0.0]), 0.18, 0.3, cell, col, axis, 0.3)
    else:
        cell = ATLAS_CELLS["oak" if kind == "oak" else "birch"]
        leaf = np.array([0.075, 0.12, 0.035, 1.0]) if kind == "oak" else np.array([0.10, 0.15, 0.04, 1.0])
        for k in range(30):
            f = rng.uniform(0.3, 1.05)
            c = axis * f + _rand_unit(rng) * np.array([0.26, 0.16, 0.26]) * (0.4 + 0.6 * f)
            out = c - axis * f * 0.7
            col = leaf.copy()
            col[:3] *= rng.uniform(0.8, 1.12) * (0.7 + 0.3 * f)
            s = rng.uniform(0.26, 0.4)
            cb.card(c, out * 0.6 + axis * 0.4 + _rand_unit(rng) * 0.4, _rand_unit(rng) + out, s, s, cell, col, out, 0.3,
                    blend=0.6, centered=True)
    return cb.mesh()


def create_bole_mesh(runs, total_len: float, segments: int = 14) -> MeshData:
    """The stem of a felled tree in its own frame (Y along the stem, +Z up, metres).

    `runs` is a list of contiguous pieces, each ``{"rings": [(y, r, notch), ...], "cap0": bool,
    "cap1": bool}``; `notch` > 0 clamps the top of that ring down by the kerf depth so a
    half-finished axe cut shows as a V-notch of bright wood.  Vertex alpha 0 marks wood
    (end grain / fresh cut faces) for the fuel shader, 1 bark."""
    th = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    pos, nrm, uv, col, idx = [], [], [], [], []
    for run in runs:
        rings = run["rings"]
        base = len(pos)
        for j, (y, r, notch) in enumerate(rings):
            if j + 1 < len(rings):
                slope = (r - rings[j + 1][1]) / max(1e-6, rings[j + 1][0] - y)
            else:
                slope = (rings[j - 1][1] - r) / max(1e-6, y - rings[j - 1][0]) if j else 0.0
            for a in th:
                x, z = r * math.cos(a), r * math.sin(a)
                n = np.array([math.cos(a), slope, math.sin(a)])
                wood = 0.0
                if notch > 0.0 and z > r - notch:
                    z = r - notch
                    n = np.array([0.0, 0.3, 1.0])
                    wood = 1.0
                pos.append((x, y, z))
                nrm.append(n / np.linalg.norm(n))
                uv.append((a / (2 * np.pi), y / max(total_len, 1e-6)))
                col.append((1.0, 1.0, 1.0, 0.0 if wood else 1.0))
        row = segments + 1
        for j in range(len(rings) - 1):
            for i in range(segments):
                a0 = base + j * row + i
                b0 = a0 + row
                idx += [a0, b0, a0 + 1, a0 + 1, b0, b0 + 1]
        for end, flag in ((0, run.get("cap0", True)), (len(rings) - 1, run.get("cap1", True))):
            if not flag:
                continue
            y, r, notch = rings[end]
            sgn = -1.0 if end == 0 else 1.0
            c = len(pos)
            pos.append((0.0, y, 0.0)); nrm.append((0.0, sgn, 0.0)); uv.append((0.5, y / max(total_len, 1e-6)))
            col.append((1.0, 1.0, 1.0, 0.0))
            for a in th:
                pos.append((r * math.cos(a), y, min(r * math.sin(a), r - notch if notch > 0 else 9.0)))
                nrm.append((0.0, sgn, 0.0)); uv.append((a / (2 * np.pi), y / max(total_len, 1e-6)))
                col.append((1.0, 1.0, 1.0, 0.0))
            for i in range(segments):
                if end == 0:
                    idx += [c, c + 1 + i, c + 2 + i]
                else:
                    idx += [c, c + 2 + i, c + 1 + i]
    if not pos:
        return MeshData(np.zeros((0, 12)), np.zeros(0))
    return fix_winding(MeshData(_pack(np.array(pos), np.array(nrm), np.array(uv), np.array(col)), np.array(idx)))


def create_conifer(kind: str = "pine", seed: int = 0, height: float = 15.0) -> Tuple[MeshData, MeshData]:
    """Returns (trunk, foliage) meshes for a pine or cedar (nominal height)."""
    rng = np.random.default_rng(seed)
    if kind == "pine":
        bark = (0.33, 0.23, 0.16, 1.0)
        leaf = np.array([0.13, 0.25, 0.12, 1.0]) * np.array([rng.uniform(0.9, 1.1), rng.uniform(0.92, 1.08), 1, 1])
        tiers, start, max_r = 7, 0.38, 0.30
    else:
        bark = (0.42, 0.24, 0.16, 1.0)
        leaf = np.array([0.10, 0.22, 0.13, 1.0]) * np.array([1, rng.uniform(0.95, 1.05), 1, 1])
        tiers, start, max_r = 11, 0.10, 0.22
    trunk = create_frustum(0.25, 0.03, height * 0.97, 9, 4, bark, alpha_bottom=1.0, alpha_top=0.4, caps=False, jitter=0.06, rng=rng)
    roots = []
    for i in range(4):
        a = i * math.pi / 2 + rng.uniform(-0.3, 0.3)
        root = create_frustum(0.12, 0.02, 0.9, 6, 1, bark, caps=False).transformed(
            _m_translate(math.cos(a) * 0.12, 0.25, math.sin(a) * 0.12) @ _m_rot_y(-a) @ _m_rot_z(math.radians(-72)))
        roots.append(root)
    trunk = merge([trunk] + roots)
    if CARD_FOLIAGE:
        # bare branch stubs inside the crown
        stubs = []
        for i in range(tiers * 2):
            t = i / (tiers * 2 - 1)
            y = height * (start + (0.9 - start) * t)
            a = rng.uniform(0, 2 * math.pi)
            L = height * max_r * (1.0 - t) * 0.8 + 0.3
            d = np.array([math.cos(a), -0.1, math.sin(a)])
            stubs.append(create_frustum(0.05 * (1.0 - t) + 0.015, 0.01, L, 5, 1, bark, alpha_bottom=0.5, alpha_top=0.2,
                                        caps=False).transformed(_m_translate(0, y, 0) @ _m_align(d)))
        trunk = merge([trunk] + stubs)
        return trunk, create_conifer_cards(kind, seed, height)
    parts = []
    for i in range(tiers):
        t = i / (tiers - 1)
        y_edge = height * (start + (0.93 - start) * t)
        if kind == "pine":
            rad = height * max_r * (1.0 - t) ** 0.85 + 0.35
            y_top = y_edge + height * 0.10 + rad * 0.35
        else:
            rad = height * max_r * (1.0 - t) ** 1.05 + 0.25
            y_top = y_edge + height * 0.07 + rad * 0.5
        shade_ = 0.78 + 0.3 * t
        c = leaf.copy()
        c[:3] *= shade_
        parts.append(_tier_skirt(y_top, y_edge, rad, 11 if kind == "pine" else 9, c, rng, droop=0.22 if kind == "pine" else 0.28))
    top = create_frustum(0.35, 0.0, height * 0.08, 7, 1, tuple(leaf * np.array([1.1, 1.1, 1.1, 1])), caps=False
                         ).transformed(_m_translate(0, height * 0.93, 0))
    parts.append(top)
    foliage = merge(parts)
    return trunk, foliage


def create_deciduous(kind: str = "oak", seed: int = 0, height: float = 14.0) -> Tuple[MeshData, MeshData]:
    """Returns (trunk+branches, canopy clumps) for oak or birch (nominal height)."""
    rng = np.random.default_rng(seed + 100)
    if kind == "oak":
        bark = (0.30, 0.25, 0.20, 1.0)
        r0, crown_y, crown_r, n_br, n_clumps = 0.32, 0.55, 0.36, 5, 11
    else:
        bark = (0.88, 0.86, 0.80, 1.0)
        r0, crown_y, crown_r, n_br, n_clumps = 0.17, 0.5, 0.22, 4, 8
    trunk_h = height * (crown_y + 0.1)
    parts = [create_frustum(r0, r0 * 0.55, trunk_h, 9, 3, bark, alpha_bottom=1.0, alpha_top=0.6, caps=False, jitter=0.05, rng=rng)]
    ends = []
    for i in range(n_br):
        a = i * 2 * math.pi / n_br + rng.uniform(-0.4, 0.4)
        tilt = math.radians(rng.uniform(30, 55))
        L = height * rng.uniform(0.28, 0.42)
        base = np.array([0.0, trunk_h * rng.uniform(0.62, 0.95), 0.0])
        d = np.array([math.cos(a) * math.sin(tilt), math.cos(tilt), math.sin(a) * math.sin(tilt)])
        br = create_frustum(r0 * 0.45, r0 * 0.12, L, 6, 2, bark, alpha_bottom=0.55, alpha_top=0.2, caps=False)
        parts.append(br.transformed(_m_translate(*base) @ _m_align(d)))
        ends.append(base + d * L)
    # leader
    ends.append(np.array([0.0, trunk_h + height * 0.25, 0.0]))
    parts.append(create_frustum(r0 * 0.5, r0 * 0.15, height * 0.3, 6, 1, bark, alpha_bottom=0.6, alpha_top=0.25, caps=False
                                ).transformed(_m_translate(0, trunk_h * 0.95, 0)))
    # secondary branches fan out of each primary toward the crown surface
    tips = []
    for e in ends[:-1]:
        for k in range(2):
            d2 = _rand_unit(rng) * 0.6 + (e / max(1e-6, np.linalg.norm(e))) * 0.8 + np.array([0.0, 0.4, 0.0])
            d2 /= np.linalg.norm(d2)
            L2 = height * rng.uniform(0.1, 0.16)
            parts.append(create_frustum(r0 * 0.12, r0 * 0.04, L2, 5, 1, bark, alpha_bottom=0.2, alpha_top=0.1,
                                        caps=False).transformed(_m_translate(*e) @ _m_align(d2)))
            tips.append(e + d2 * L2)
    tips += ends
    trunk = merge(parts)
    if CARD_FOLIAGE:
        return trunk, create_broadleaf_cards(kind, seed, height, tips)
    crown_c = np.array([0.0, height * 0.70, 0.0])
    clumps = []
    leaf = np.array([0.20, 0.36, 0.14, 1.0]) if kind == "oak" else np.array([0.32, 0.46, 0.18, 1.0])
    rx = height * crown_r * (1.15 if kind == "oak" else 0.95)
    ry = height * (0.24 if kind == "oak" else 0.30)
    n_shell = 26 if kind == "oak" else 20
    centers = []
    # clumps on an ellipsoidal crown shell (golden-spiral), plus a few at branch tips
    for i in range(n_shell):
        t = (i + 0.5) / n_shell
        yy = 1.0 - 2.0 * t * 0.92
        rr = math.sqrt(max(0.0, 1.0 - yy * yy))
        ang = i * 2.399963 + rng.uniform(-0.3, 0.3)
        p = crown_c + np.array([math.cos(ang) * rr * rx, yy * ry, math.sin(ang) * rr * rx]) * rng.uniform(0.78, 1.0)
        centers.append((p, rng.uniform(0.55, 0.8)))
    for e in ends[:-1]:
        centers.append((e + np.array([0.0, height * 0.02, 0.0]), rng.uniform(0.6, 0.85)))
    for i in range(5):
        centers.append((crown_c + rng.normal(0, 1, 3) * np.array([rx * 0.3, ry * 0.3, rx * 0.3]), 0.9))
    base_s = height * (0.095 if kind == "oak" else 0.085)
    for c, sc in centers:
        s = base_s * sc * 1.6
        bias = (c - crown_c) / np.array([rx, ry, rx])
        nb = float(np.linalg.norm(bias))
        bias = bias / max(1e-6, nb) * 0.9
        col = leaf.copy()
        # sunlit top of the crown slightly lighter, underside darker (baked)
        top = (c[1] - crown_c[1]) / ry
        col[:3] *= rng.uniform(0.88, 1.1) * (0.85 + 0.25 * max(-1.0, min(1.0, top)))
        col[3] = 0.15
        m = create_icosphere(1, s, tuple(col), displace=0.38, seed=int(rng.integers(1e6)), flatten=0.72, normal_bias=bias)
        arr = m.as_array()
        low = arr[:, 1] < 0
        arr[low, 8:11] *= 0.72
        clumps.append(MeshData(arr, m.indices).transformed(_m_translate(*c)))
    return trunk, merge(clumps)


def create_dead_tree(seed: int = 0, height: float = 10.0) -> MeshData:
    rng = np.random.default_rng(seed + 7)
    bark = (0.42, 0.40, 0.37, 1.0)
    parts = [create_frustum(0.24, 0.1, height, 8, 3, bark, alpha_bottom=1.0, alpha_top=0.8, caps=True, jitter=0.08, rng=rng)]
    for i in range(5):
        a = rng.uniform(0, 2 * math.pi)
        y = height * rng.uniform(0.35, 0.9)
        L = rng.uniform(0.6, 1.6)
        d = np.array([math.cos(a), rng.uniform(0.1, 0.6), math.sin(a)])
        parts.append(create_frustum(0.05, 0.01, L, 5, 1, bark, caps=False).transformed(_m_translate(0, y, 0) @ _m_align(d)))
    return merge(parts)


def create_fungus_bracket(color: Color = (0.62, 0.52, 0.38, 1.0)) -> MeshData:
    """A half-disc shelf fungus (amadou / birch polypore), pointing +X from the trunk."""
    m = create_icosphere(1, 1.0, color, displace=0.08, seed=3, flatten=0.35)
    arr = m.as_array()
    arr[:, 0] = np.maximum(arr[:, 0], -0.05)   # flat against the bark
    arr[arr[:, 1] < 0, 8:11] *= 0.75          # pale pored underside
    arr[arr[:, 1] < 0, 8:11] += 0.15
    return MeshData(arr, m.indices)


def create_stump(color: Color = (0.35, 0.27, 0.2, 1.0)) -> MeshData:
    m = create_frustum(1.0, 0.92, 1.0, 10, 1, color, caps=True, jitter=0.05)
    arr = m.as_array()
    top = arr[:, 4] > 0.9
    arr[top, 8:11] = (0.78, 0.62, 0.42)
    return MeshData(arr, m.indices)


def create_bush(seed: int = 0, color: Color = (0.18, 0.32, 0.13, 1.0)) -> MeshData:
    if CARD_FOLIAGE:
        return create_bush_cards(seed)
    rng = np.random.default_rng(seed + 300)
    parts = []
    for i in range(5):
        c = np.array([rng.normal(0, 0.35), rng.uniform(0.25, 0.6), rng.normal(0, 0.35)])
        s = rng.uniform(0.35, 0.55)
        col = np.array(color)
        col[:3] *= rng.uniform(0.85, 1.15)
        col[3] = 0.4
        m = create_icosphere(1, s, tuple(col), displace=0.3, seed=int(rng.integers(1e6)), flatten=0.8,
                             normal_bias=c / max(1e-6, np.linalg.norm(c)) * 0.6)
        arr = m.as_array()
        arr[arr[:, 1] < 0, 8:11] *= 0.7
        parts.append(MeshData(arr, m.indices).transformed(_m_translate(*c)))
    return merge(parts)


def create_fern(seed: int = 0, color: Color = (0.22, 0.40, 0.14, 1.0)) -> MeshData:
    rng = np.random.default_rng(seed + 500)
    pos, nrm, col, inds = [], [], [], []
    n_fronds = 8
    for f in range(n_fronds):
        a = f * 2 * math.pi / n_fronds + rng.uniform(-0.2, 0.2)
        L = rng.uniform(0.6, 0.95)
        segs = 6
        base = len(pos)
        for s in range(segs + 1):
            t = s / segs
            r = L * t
            y = 0.5 * L * math.sin(t * math.pi * 0.55) - 0.15 * t * t
            w = 0.11 * math.sin(math.pi * min(1.0, t * 1.1 + 0.05))
            cx, cz = r * math.cos(a), r * math.sin(a)
            sx, sz = -math.sin(a) * w, math.cos(a) * w
            c = np.array(color) * np.array([1.0 + 0.2 * t, 1.0 + 0.15 * t, 1.0, 1.0])
            c[3] = 1.0 - t
            for side in (-1, 1):
                pos.append((cx + sx * side, y, cz + sz * side))
                nrm.append((0.0, 1.0, 0.0))
                col.append(c)
        for s in range(segs):
            a0 = base + s * 2
            inds += [a0, a0 + 1, a0 + 2, a0 + 1, a0 + 3, a0 + 2]
    return drop_degenerate(MeshData(_pack(np.array(pos), np.array(nrm), np.zeros((len(pos), 2)), np.array(col)), np.array(inds)))


def create_grass_tuft(seed: int = 0, color: Color = (0.30, 0.42, 0.16, 1.0), blades: int = 7, height: float = 0.35) -> MeshData:
    rng = np.random.default_rng(seed + 900)
    pos, nrm, col, inds = [], [], [], []
    for b in range(blades):
        a = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.1, 0.5)
        h = height * rng.uniform(0.6, 1.3)
        w = rng.uniform(0.012, 0.022)
        ox, oz = rng.normal(0, 0.04, 2)
        dx, dz = math.cos(a), math.sin(a)
        base = len(pos)
        for s in range(4):
            t = s / 3
            cx = ox + dx * lean * h * t * t
            cz = oz + dz * lean * h * t * t
            y = h * t
            ww = w * (1.0 - t)
            c = np.array(color, dtype=float).copy()
            c[3] = 1.0 - t
            pos += [(cx - dz * ww, y, cz + dx * ww), (cx + dz * ww, y, cz - dx * ww)]
            nrm += [(dx * 0.3162, 0.9487, dz * 0.3162)] * 2      # unit normal tipped along the lean
            col += [c, c]
        for s in range(3):
            a0 = base + s * 2
            inds += [a0, a0 + 1, a0 + 2, a0 + 1, a0 + 3, a0 + 2]
    return drop_degenerate(MeshData(_pack(np.array(pos), np.array(nrm), np.zeros((len(pos), 2)), np.array(col)), np.array(inds)))


def create_rock(seed: int = 0, color: Color = (0.42, 0.41, 0.39, 1.0), flatten: float = 0.6) -> MeshData:
    return create_stone(seed + 1200, color, flatten, roughness=0.3)


# ------------------------------------------------------------------------------------ camp props
def create_hearth_ring_mesh(inner_r: float = 0.45, outer_r: float = 0.65, stone_count: int = 13,
                            ground_fn: Optional[Callable] = None) -> MeshData:
    """Stone ring in pit-local coordinates; ground_fn(x, z) gives the local ground height."""
    rng = np.random.default_rng(42)
    parts = []
    for i in range(stone_count):
        a = 2 * math.pi * i / stone_count + rng.uniform(-0.06, 0.06)
        r = 0.5 * (inner_r + outer_r) + rng.uniform(-0.02, 0.02)
        s = rng.uniform(0.085, 0.12)
        g = rng.uniform(0.27, 0.38)
        warm = rng.uniform(-0.02, 0.03)
        stone = create_stone(i + 77, (g + warm, g, g * 0.93 - warm * 0.5, 1.0), flatten=rng.uniform(0.55, 0.7))
        arr = stone.as_array()
        # soot on the inner faces
        inner_face = (arr[:, 0] * -math.cos(a) + arr[:, 2] * -math.sin(a)) > 0.2
        arr[inner_face, 8:11] *= 0.45
        stone = MeshData(arr, stone.indices)
        gx, gz = r * math.cos(a), r * math.sin(a)
        gy = ground_fn(gx, gz) if ground_fn is not None else 0.0
        parts.append(stone.transformed(_m_translate(gx, gy + s * 0.42, gz) @ _m_rot_y(-a) @ _m_scale(s * 1.15, s, s * 0.95)))
    return merge(parts)


def create_log_mesh(segments: int = 12, rings: int = 6) -> MeshData:
    """Unit fuel cylinder along Y (-0.5..0.5), radius 1.  Caps carry alpha 0 (end grain)."""
    return create_frustum(1.0, 1.0, 1.0, segments, rings, (1.0, 1.0, 1.0, 1.0), caps=True, cap_alpha=0.0, centered=True)


SPLIT_SECTOR_DEG = 100.0


def split_fuel_centroid(sector_deg: float = SPLIT_SECTOR_DEG) -> float:
    """Distance from the pith (apex) to the centroid of the unit-area split wedge."""
    th = math.radians(sector_deg)
    R = math.sqrt(2.0 * math.pi / th)
    return 2.0 * R * math.sin(th / 2.0) / (1.5 * th)


def create_split_fuel_mesh(sector_deg: float = SPLIT_SECTOR_DEG, segments: int = 8, seed: int = 5) -> MeshData:
    """A split firewood wedge for the fuel shader, along Y (-0.5..0.5), pith (apex) at the origin
    and the bark arc towards +X.  Its cross-section has the area of the unit circle, so scaling by
    a piece's equivalent radius keeps its volume; vertex alpha 1 = bark, 0 = wood (the two riven
    faces and the end grain)."""
    rng = np.random.default_rng(seed)
    th = math.radians(sector_deg)
    R = math.sqrt(2.0 * math.pi / th)
    angs = np.linspace(-th / 2, th / 2, segments + 1)
    jag = 1.0 + rng.uniform(-0.035, 0.035, segments + 1)
    prof = [(R * jag[i] * math.cos(a), R * jag[i] * math.sin(a)) for i, a in enumerate(angs)]
    apex = (0.06 * R, 0.0)
    pos, nrm, uv, col, idx = [], [], [], [], []

    def vert(p, n, v, alpha):
        pos.append(p); nrm.append(n); uv.append((0.5, v)); col.append((1.0, 1.0, 1.0, alpha))
        return len(pos) - 1

    # bark arc: rings along the length so the shader's segment gradients interpolate
    rings = 4
    for j in range(rings + 1):
        y = -0.5 + j / rings
        for i, ((x, z), a) in enumerate(zip(prof, angs)):
            vert((x, y, z), (math.cos(a), 0.0, math.sin(a)), j / rings, 1.0)
    row = segments + 1
    for j in range(rings):
        for i in range(segments):
            a0 = j * row + i
            b0 = a0 + row
            idx += [a0, b0, a0 + 1, a0 + 1, b0, b0 + 1]
    # the two riven faces: slightly wavy, as split wood follows the grain
    for (x, z), a, sgn in ((prof[0], angs[0], -1.0), (prof[-1], angs[-1], 1.0)):
        n = (-math.sin(a) * sgn, 0.0, math.cos(a) * sgn)
        k = len(pos)
        for j in range(rings + 1):
            y = -0.5 + j / rings
            w = 0.04 * math.sin(j * 2.1 + sgn)
            vert((apex[0] + w * n[0], y, apex[1] + w * n[2]), n, j / rings, 0.0)
            vert((x + w * n[0], y, z + w * n[2]), n, j / rings, 0.0)
        for j in range(rings):
            a0 = k + j * 2
            idx += [a0, a0 + 2, a0 + 1, a0 + 1, a0 + 2, a0 + 3]
    # end grain
    for y, ny in ((-0.5, -1.0), (0.5, 1.0)):
        c = vert((apex[0], y, apex[1]), (0.0, ny, 0.0), 0.0 if ny < 0 else 1.0, 0.0)
        ring = [vert((x, y, z), (0.0, ny, 0.0), 0.0 if ny < 0 else 1.0, 0.0) for x, z in prof]
        for i in range(segments):
            idx += [c, ring[i], ring[i + 1]]
    return fix_winding(MeshData(_pack(np.array(pos), np.array(nrm), np.array(uv), np.array(col)), np.array(idx)))


def create_split_log_mesh() -> MeshData:
    """Half-round split firewood piece (unit), for the woodpile."""
    segs = 8
    th = np.linspace(0, math.pi, segs + 1)
    pos, nrm, col, inds = [], [], [], []
    for y in (-0.5, 0.5):
        for a in th:
            pos.append((math.cos(a), y, math.sin(a)))
            nrm.append((math.cos(a), 0, math.sin(a)))
            col.append((0.24, 0.17, 0.12, 1.0))
    row = segs + 1
    for i in range(segs):
        inds += [i, row + i, i + 1, i + 1, row + i, row + i + 1]
    base = len(pos)
    for y in (-0.5, 0.5):
        for x in (-1.0, 1.0):
            pos.append((x, y, 0.0))
            nrm.append((0, 0, -1))
            col.append((0.66, 0.50, 0.32, 1.0))
    inds += [base, base + 2, base + 1, base + 1, base + 2, base + 3]
    mesh = MeshData(_pack(np.array(pos), np.array(nrm), np.zeros((len(pos), 2)), np.array(col)), np.array(inds))
    return mesh


def create_tinder_nest_mesh(seed: int = 0) -> MeshData:
    """A bird's-nest bundle of fibres (unit radius, y up to ~1): a dense core with loose
    strands wound around a cupped hollow where the ember goes."""
    rng = np.random.default_rng(seed + 40)
    core = create_icosphere(2, 1.0, (1.0, 1.0, 1.0, 1.0), displace=0.25, seed=seed + 41, flatten=0.5)
    arr = core.as_array().copy()
    arr[:, 0:3] *= 0.82
    top = arr[:, 1] > 0.15
    arr[top, 1] -= 0.22
    arr[:, 7] = 0.5
    arr[:, 11] = 1.0
    parts = [MeshData(arr, core.indices)]
    for k in range(70):
        r0 = rng.uniform(0.55, 1.08)
        y0 = rng.uniform(-0.25, 0.35)
        a0 = rng.uniform(0, 2 * math.pi)
        sweep = rng.uniform(1.2, 3.2) * rng.choice([-1, 1])
        n = 7
        pts = []
        for i in range(n + 1):
            t = i / n
            a = a0 + sweep * t
            rr = r0 * (1.0 + 0.1 * math.sin(t * 7.0 + k))
            pts.append(np.array([rr * math.cos(a), y0 + 0.25 * math.sin(t * math.pi + k) * rng.uniform(0.2, 1.0), rr * math.sin(a)]))
        thick = rng.uniform(0.025, 0.05)
        for i in range(n):
            d = pts[i + 1] - pts[i]
            L = float(np.linalg.norm(d))
            if L < 1e-5:
                continue
            seg = create_frustum(thick, thick * 0.9, L, 4, 1, (1, 1, 1, 1), caps=False).transformed(_m_translate(*pts[i]) @ _m_align(d))
            sa = seg.as_array().copy()
            sa[:, 7] = (i + 0.5) / n
            sa[:, 8:11] *= rng.uniform(0.75, 1.15)
            sa[:, 11] = 1.0
            parts.append(MeshData(sa, seg.indices))
    return merge(parts)


def create_needle_pile(color: Color = (0.46, 0.30, 0.16, 1.0)) -> MeshData:
    return create_icosphere(2, 1.0, color, displace=0.12, seed=5, flatten=0.28)


def create_birch_curls() -> MeshData:
    rng = np.random.default_rng(11)
    parts = []
    for i in range(7):
        c = create_frustum(0.035, 0.035, rng.uniform(0.08, 0.14), 8, 1, (0.93, 0.90, 0.84, 1.0), caps=False, centered=True)
        arr = c.as_array()
        # open the curl (C-shape) and colour the inner face tan
        ang = np.arctan2(arr[:, 2], arr[:, 0])
        arr[ang > 2.2, 8:11] = (0.78, 0.55, 0.40)
        c = MeshData(arr, c.indices)
        parts.append(c.transformed(_m_translate(rng.normal(0, 0.06), 0.03, rng.normal(0, 0.06)) @ _m_rot_y(rng.uniform(0, 6)) @ _m_rot_z(math.pi / 2 + rng.uniform(-0.3, 0.3))))
    return merge(parts)


def create_cattails() -> MeshData:
    rng = np.random.default_rng(21)
    parts = []
    for i in range(6):
        h = rng.uniform(0.8, 1.25)
        x, z = rng.normal(0, 0.12, 2)
        lean = rng.uniform(-0.08, 0.08)
        stalk = create_frustum(0.008, 0.004, h, 5, 1, (0.45, 0.50, 0.26, 0.9), caps=False, alpha_bottom=1.0, alpha_top=0.1)
        head = create_frustum(0.022, 0.02, 0.14, 7, 1, (0.36, 0.22, 0.12, 0.2), caps=True)
        parts.append(stalk.transformed(_m_translate(x, 0, z) @ _m_rot_z(lean)))
        parts.append(head.transformed(_m_translate(x - lean * h * 0.8, h * 0.78, z) @ _m_rot_z(lean)))
    return merge(parts)


def create_kindling_bundle(color: Color = (0.40, 0.30, 0.20, 1.0), n: int = 11) -> MeshData:
    rng = np.random.default_rng(31)
    parts = []
    for i in range(n):
        L = rng.uniform(0.3, 0.42)
        r = rng.uniform(0.006, 0.011)
        st = create_frustum(r, r * 0.8, L, 5, 1, color, centered=True)
        d = np.array([1.0, rng.uniform(-0.05, 0.08), rng.normal(0, 0.2)])
        parts.append(st.transformed(_m_translate(rng.normal(0, 0.03), 0.012 + (i % 3) * 0.014, rng.normal(0, 0.05)) @ _m_align(d)))
    return merge(parts)


def create_split_piece(radius: float, length: float, sector_deg: float, rng, bark=(0.22, 0.16, 0.11),
                       wood=(0.62, 0.47, 0.30)) -> MeshData:
    """A split firewood wedge along +Z (centred): bark on the outer arc, pale split faces, and
    end grain with growth rings.  Vertex colours carry the materials."""
    seg = 7
    a0 = -math.radians(sector_deg) / 2
    th = np.linspace(a0, -a0, seg + 1)
    jag = 1.0 + rng.uniform(-0.04, 0.04, seg + 1)
    prof = [(radius * jag[i] * math.cos(t), radius * jag[i] * math.sin(t)) for i, t in enumerate(th)]
    apex = (radius * 0.08, 0.0)
    pos, nrm, col, inds = [], [], [], []
    wood_c = np.array(wood) * rng.uniform(0.88, 1.08)
    bark_c = np.array(bark) * rng.uniform(0.85, 1.15)
    z0, z1 = -length / 2, length / 2

    def quad(p0, p1, p2, p3, n, c):
        k = len(pos)
        for q in (p0, p1, p2, p3):
            pos.append(q)
            nrm.append(n)
            col.append((*c, 1.0))
        inds.extend([k, k + 1, k + 2, k, k + 2, k + 3])

    # bark arc
    for i in range(seg):
        (x0, y0), (x1, y1) = prof[i], prof[i + 1]
        n = np.array([x0 + x1, y0 + y1, 0.0])
        n /= np.linalg.norm(n)
        quad((x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1), n, bark_c * rng.uniform(0.85, 1.1))
    # the two split faces (slightly uneven tone); outward normals point away from the wedge
    for (x, y), a in ((prof[0], th[0]), (prof[-1], th[-1])):
        n = np.array([math.sin(a), -math.cos(a), 0.0]) if a < 0 else np.array([-math.sin(a), math.cos(a), 0.0])
        quad((apex[0], apex[1], z0), (x, y, z0), (x, y, z1), (apex[0], apex[1], z1), n, wood_c * rng.uniform(0.9, 1.05))
    # end grain: fan with darker rings toward the bark
    for z, nz in ((z0, -1.0), (z1, 1.0)):
        k0 = len(pos)
        pos.append((apex[0], apex[1], z))
        nrm.append((0.0, 0.0, nz))
        col.append((*(wood_c * 1.08), 1.0))
        for i, (x, y) in enumerate(prof):
            pos.append((x, y, z))
            nrm.append((0.0, 0.0, nz))
            col.append((*(bark_c * 1.3), 1.0))
        for i in range(seg):
            a, b = k0 + 1 + i, k0 + 2 + i
            inds.extend([k0, a, b] if nz > 0 else [k0, b, a])
    m = MeshData(_pack(np.array(pos, dtype=float), np.array(nrm, dtype=float), np.zeros((len(pos), 2)), np.array(col)),
                 np.array(inds))
    return fix_winding(m)


def create_woodpile_mesh(seed: int = 3) -> MeshData:
    """A tidy rick of split firewood on two ground rails, held by stakes."""
    rng = np.random.default_rng(seed)
    parts = []
    for z in (-0.16, 0.16):
        parts.append(create_frustum(0.035, 0.035, 1.0, 7, 1, (0.24, 0.19, 0.14, 1.0), centered=True).transformed(
            _m_translate(0, 0.035, z) @ _m_rot_z(math.pi / 2)))
    for x in (-0.46, 0.46):
        parts.append(create_frustum(0.025, 0.02, 0.62, 6, 1, (0.26, 0.2, 0.15, 1.0)).transformed(_m_translate(x, 0.0, 0.0)))
    y = 0.07
    for row in range(4):
        n = 6 - (row % 2)
        r_row = 0.0
        for i in range(n):
            r = rng.uniform(0.07, 0.085)
            L = rng.uniform(0.38, 0.44)
            sector = rng.uniform(95, 135)
            piece = create_split_piece(r, L, sector, rng)
            x = (i - (n - 1) / 2) * 0.15 + rng.uniform(-0.01, 0.01)
            roll = rng.choice([math.pi / 2, -math.pi / 2]) + rng.uniform(-0.5, 0.5)
            parts.append(piece.transformed(_m_translate(x, y + r * 0.55, rng.uniform(-0.025, 0.025)) @ _m_rot_y(rng.uniform(-0.08, 0.08))
                                           @ _m_rot_z(roll)))
            r_row = max(r_row, r)
        y += r_row * 0.95
    return merge(parts)


def create_backpack_mesh() -> MeshData:
    """Waxed-canvas rucksack with a lid, front pocket, side straps and a rolled sleeping mat."""
    canvas = (0.20, 0.24, 0.15, 1.0)
    dark = (0.13, 0.15, 0.10, 1.0)
    leather = (0.30, 0.19, 0.11, 1.0)
    parts = [create_rounded_box(0.16, 0.23, 0.10, 0.3, 3, canvas).transformed(_m_translate(0, 0.24, 0))]
    parts.append(create_rounded_box(0.165, 0.05, 0.11, 0.35, 2, dark).transformed(_m_translate(0, 0.46, 0.005)))
    parts.append(create_rounded_box(0.12, 0.09, 0.035, 0.35, 2, canvas).transformed(_m_translate(0, 0.2, 0.11)))
    for x in (-0.08, 0.08):
        parts.append(create_box_mesh((0.022, 0.3, 0.012), leather).transformed(_m_translate(x, 0.33, 0.125)))
        parts.append(create_box_mesh((0.026, 0.018, 0.016), (0.55, 0.52, 0.45, 1.0)).transformed(_m_translate(x, 0.2, 0.132)))
    mat = create_capsule(0.065, 0.065, 0.3, 14, 3, (0.42, 0.18, 0.12, 1.0))
    parts.append(mat.transformed(_m_translate(-0.15, 0.565, 0.0) @ _m_rot_y(math.pi / 2)))
    for x in (-0.09, 0.09):
        parts.append(create_frustum(0.068, 0.068, 0.018, 14, 1, leather, centered=True).transformed(
            _m_translate(x, 0.565, 0.0) @ _m_rot_z(math.pi / 2)))
    return merge(parts)


def create_drying_rack_mesh() -> MeshData:
    wood = (0.42, 0.32, 0.22, 1.0)
    parts = []
    for ox, oz in ((-0.65, -0.32), (0.65, -0.32), (-0.65, 0.32), (0.65, 0.32)):
        parts.append(create_frustum(0.04, 0.035, 1.25, 7, 1, wood).transformed(_m_translate(ox, -0.1, oz)))
    for oz in (-0.32, 0.32):
        parts.append(create_frustum(0.032, 0.032, 1.5, 7, 1, wood, centered=True).transformed(_m_translate(0, 0.55, oz) @ _m_rot_z(math.pi / 2)))
        parts.append(create_frustum(0.03, 0.03, 1.5, 7, 1, wood, centered=True).transformed(_m_translate(0, 1.08, oz) @ _m_rot_z(math.pi / 2)))
    for ox in (-0.5, -0.25, 0.0, 0.25, 0.5):
        parts.append(create_frustum(0.018, 0.018, 0.76, 5, 1, wood, centered=True).transformed(_m_translate(ox, 0.59, 0) @ _m_rot_x(math.pi / 2)))
    # ridge pole for the tarp
    parts.append(create_frustum(0.03, 0.03, 1.6, 7, 1, wood, centered=True).transformed(_m_translate(0, 1.3, 0) @ _m_rot_z(math.pi / 2)))
    for ox in (-0.72, 0.72):
        parts.append(create_frustum(0.03, 0.028, 1.45, 7, 1, wood).transformed(_m_translate(ox, -0.1, 0)))
    return merge(parts)


def create_tarp_mesh(width: float = 1.75, depth: float = 1.15, sag: float = 0.07, res: int = 14) -> MeshData:
    """A-frame polyethylene tarp draped over the ridge pole (y up, ridge along X)."""
    xs = np.linspace(-width / 2, width / 2, res)
    zs = np.linspace(-depth / 2, depth / 2, res)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    Y = 1.33 - np.abs(Z) * 0.75 - sag * np.sin(np.pi * (X / width + 0.5)) * (np.abs(Z) / (depth / 2))
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    Yp = np.pad(Y, 1, mode="edge")
    dx = xs[1] - xs[0]
    nx = (Yp[:-2, 1:-1] - Yp[2:, 1:-1])
    nz = (Yp[1:-1, :-2] - Yp[1:-1, 2:])
    N = np.stack([nx, np.full_like(Y, 2 * dx), nz], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    UV = np.stack([X / width + 0.5, Z / depth + 0.5], -1).reshape(-1, 2)
    C = np.tile((0.14, 0.32, 0.58, 1.0), (len(P), 1))
    i = np.arange(res - 1)
    I, J = np.meshgrid(i, i, indexing="ij")
    a = (I * res + J).ravel()
    b = ((I + 1) * res + J).ravel()
    c = (I * res + J + 1).ravel()
    d = ((I + 1) * res + J + 1).ravel()
    inds = np.stack([a, c, b, b, c, d], 1).ravel()
    return MeshData(_pack(P, N.reshape(-1, 3), UV, C), inds)


def create_hearthboard_kit(notched: bool = False, notch_depth: float = 0.0) -> MeshData:
    wood = (0.70, 0.56, 0.38, 1.0)
    parts = [create_box_mesh((0.38, 0.018, 0.07), wood).transformed(_m_translate(0, 0.009, 0))]
    # burned socket & notch
    char = (0.05, 0.04, 0.03, 1.0)
    if notch_depth > 0:
        parts.append(create_frustum(0.012, 0.012, 0.004, 10, 1, char).transformed(_m_translate(0.05, 0.017, 0.02)))
    if notched:
        parts.append(create_box_mesh((0.016, 0.02, 0.03), (0.55, 0.4, 0.25, 1.0)).transformed(_m_translate(0.05, 0.009, 0.036) @ _m_rot_y(math.pi / 4)))
    return merge(parts)


def create_bow_mesh(length: float = 0.62) -> MeshData:
    wood = (0.40, 0.30, 0.20, 1.0)
    parts = []
    n = 6
    pts = []
    for i in range(n + 1):
        t = i / n
        x = (t - 0.5) * length
        y = 0.07 * math.sin(math.pi * t)
        pts.append(np.array([x, y, 0.0]))
    for i in range(n):
        d = pts[i + 1] - pts[i]
        L = np.linalg.norm(d)
        parts.append(create_frustum(0.011, 0.01, L, 6, 1, wood, caps=False).transformed(_m_translate(*pts[i]) @ _m_align(d)))
    # the cord (thin, straight between the tips)
    cord = (0.72, 0.64, 0.48, 1.0)
    d = pts[-1] - pts[0]
    parts.append(create_frustum(0.0022, 0.0022, np.linalg.norm(d), 4, 1, cord, caps=False).transformed(_m_translate(*pts[0]) @ _m_align(d)))
    return merge(parts)


# ------------------------------------------------------------------------------------ hands & tools
def create_capsule(r0: float, r1: float, length: float, segments: int = 12, cap_rings: int = 4,
                   color: Color = (1, 1, 1, 1)) -> MeshData:
    """Tapered capsule along +Z from 0..length with hemispherical ends (metric, no scaling)."""
    rows = []
    # back cap (z <= 0), body, front cap (z >= length)
    for k in range(cap_rings, 0, -1):
        a = (k / cap_rings) * (math.pi / 2)
        rows.append((-r0 * math.sin(a), r0 * math.cos(a), -math.sin(a)))
    rows.append((0.0, r0, 0.0))
    rows.append((length, r1, 0.0))
    for k in range(1, cap_rings + 1):
        a = (k / cap_rings) * (math.pi / 2)
        rows.append((length + r1 * math.sin(a), r1 * math.cos(a), math.sin(a)))
    slope = (r0 - r1) / max(length, 1e-6)
    pos, nrm, uv = [], [], []
    th = np.linspace(0, 2 * math.pi, segments + 1)
    for ri, (z, r, nz) in enumerate(rows):
        for t in th:
            c, s_ = math.cos(t), math.sin(t)
            pos.append((r * c, r * s_, z))
            if abs(nz) < 1e-9:
                n = np.array([c, s_, slope])
            else:
                n = np.array([c * math.sqrt(max(0.0, 1 - nz * nz)), s_ * math.sqrt(max(0.0, 1 - nz * nz)), nz])
            nrm.append(n / max(1e-9, np.linalg.norm(n)))
            uv.append((t / (2 * math.pi), ri / (len(rows) - 1)))
    inds = []
    row = segments + 1
    for ri in range(len(rows) - 1):
        for si in range(segments):
            a = ri * row + si
            b = a + row
            inds += [a, a + 1, b, a + 1, b + 1, b]
    m = MeshData(_pack(np.array(pos), np.array(nrm), np.array(uv), np.tile(color, (len(pos), 1))), np.array(inds))
    return drop_degenerate(m)


def create_rounded_box(hx: float, hy: float, hz: float, exponent: float = 0.35, subdiv: int = 3,
                       color: Color = (1, 1, 1, 1)) -> MeshData:
    """Superellipsoid 'rounded box' (palms, pads, boots)."""
    base = create_icosphere(subdiv, 1.0, color)
    arr = base.as_array().copy()
    d = arr[:, 0:3].astype(float)
    P = np.sign(d) * np.abs(d) ** exponent * np.array([hx, hy, hz])
    tris = base.indices.reshape(-1, 3)
    arr[:, 0:3] = P
    arr[:, 3:6] = vertex_normals(P, tris)
    return MeshData(arr, base.indices)


# Glove anatomy (metres): knuckle x/z offsets, phalanx lengths and radii per finger
FINGERS = [
    # x,      z,      (proximal, middle, distal) lengths,  radius
    (-0.0285, 0.040, (0.039, 0.024, 0.019), 0.0092),   # index (thumb side, -x on the right hand)
    (-0.0095, 0.045, (0.043, 0.027, 0.020), 0.0095),   # middle
    (0.0095, 0.043, (0.040, 0.025, 0.019), 0.0090),    # ring
    (0.0275, 0.036, (0.031, 0.019, 0.017), 0.0080),    # little
]
THUMB = ((0.030, 0.026), 0.0105)


def create_glove_parts() -> Dict[str, MeshData]:
    """Leather work-glove pieces: palm, phalanges (metric), thumb, gauntlet cuff."""
    parts: Dict[str, MeshData] = {}
    palm = create_rounded_box(0.041, 0.015, 0.046, 0.4)
    arr = palm.as_array().copy()
    # cup the palm a little and thicken the heel of the hand
    arr[:, 1] -= 0.004 * (arr[:, 0] / 0.041) ** 2
    arr[:, 1] += np.where(arr[:, 2] < -0.02, 0.002, 0.0)
    parts["palm"] = MeshData(arr, palm.indices).transformed(_m_translate(0.0, 0.0, 0.0))
    for fi, (_, _, lengths, rad) in enumerate(FINGERS):
        for j, L in enumerate(lengths):
            r0 = rad * (1.0 - 0.07 * j)
            r1 = rad * (1.0 - 0.07 * (j + 1)) * (0.92 if j == 2 else 1.0)
            parts[f"f{fi}_{j}"] = create_capsule(r0, r1, L, 12, 4)
    for j, L in enumerate(THUMB[0]):
        parts[f"t_{j}"] = create_capsule(THUMB[1] * (1.0 - 0.06 * j), THUMB[1] * (0.94 - 0.06 * j), L, 12, 4)
    parts["thumb_base"] = create_rounded_box(0.014, 0.012, 0.022, 0.5, 2)
    return parts


def create_finger_segment(tip: bool = False) -> MeshData:
    """Unit phalanx along +Z (0..1): a slightly tapered tube; the last one gets a rounded tip."""
    body = create_frustum(1.0, 0.9, 1.0, 10, 1, (1, 1, 1, 1), caps=False).transformed(_m_rot_x(math.pi / 2))
    joint = create_icosphere(1, 1.0, (1, 1, 1, 1)).transformed(_m_scale(1.0, 1.0, 0.5))
    parts = [body, joint]
    if tip:
        parts.append(create_icosphere(1, 0.9, (1, 1, 1, 1)).transformed(_m_translate(0, 0, 1.0) @ _m_scale(1, 1, 0.75)))
    return merge(parts)


def create_palm() -> MeshData:
    return create_icosphere(2, 1.0, (1, 1, 1, 1), flatten=0.42)


def create_hand_mesh(is_right_hand: bool = True) -> MeshData:
    """Static approximation (kept for compatibility): palm + four fingers + thumb."""
    skin = (0.78, 0.60, 0.48, 1.0)
    parts = [create_palm().transformed(_m_scale(0.045, 0.045, 0.052), color_mul=skin)]
    seg = create_finger_segment()
    for i in range(4):
        x = (i - 1.5) * 0.019
        parts.append(seg.transformed(_m_translate(x, 0, 0.045) @ _m_scale(0.0085, 0.0085, 0.05), color_mul=skin))
    sign = 1.0 if is_right_hand else -1.0
    parts.append(seg.transformed(_m_translate(-0.04 * sign, 0, 0.0) @ _m_rot_y(-0.7 * sign) @ _m_scale(0.01, 0.01, 0.045), color_mul=skin))
    return merge(parts)


def create_lighter_mesh() -> MeshData:
    """Disposable flint lighter: translucent-red rounded body, steel hood, spark wheel, gas lever."""
    body = create_rounded_box(0.0125, 0.031, 0.0065, 0.28, 3, (0.78, 0.14, 0.08, 1.0))
    parts = [body.transformed(_m_translate(0, -0.009, 0))]
    hood = create_rounded_box(0.0118, 0.0085, 0.0066, 0.2, 2, (0.72, 0.72, 0.74, 1.0))
    parts.append(hood.transformed(_m_translate(0, 0.030, 0)))
    wheel = create_frustum(0.0045, 0.0045, 0.007, 12, 1, (0.35, 0.35, 0.37, 1.0), centered=True)
    parts.append(wheel.transformed(_m_translate(0.004, 0.0405, 0) @ _m_rot_x(math.pi / 2)))
    lever = create_rounded_box(0.0055, 0.0025, 0.005, 0.3, 1, (0.12, 0.12, 0.13, 1.0))
    parts.append(lever.transformed(_m_translate(-0.006, 0.0395, 0)))
    return merge(parts)


def create_knife_mesh(steel: bool = False) -> MeshData:
    handle_col = (0.22, 0.14, 0.08, 1.0) if not steel else (0.12, 0.12, 0.13, 1.0)
    handle = create_box_mesh((0.018, 0.10, 0.014), handle_col)
    blade = create_box_mesh((0.003, 0.085, 0.02), (0.78, 0.80, 0.84, 1.0))
    barr = blade.as_array()
    tip = barr[:, 1] > 0.03
    barr[tip, 2] *= 0.3
    return merge([handle.transformed(_m_translate(0, -0.02, 0)), MeshData(barr, blade.indices).transformed(_m_translate(0, 0.07, 0.004))])


def _loft_blade(zs, y_bot, y_top, thick, color, edge_color=None, edge_band=0.0) -> MeshData:
    """Closed, flat-shaded solid lofted along +z (the cutting direction) from cross-sections
    that are thin rectangles in x (thickness) by y (height).  Used for axe heads."""
    rings = []
    for z, yb, yt, t in zip(zs, y_bot, y_top, thick):
        h = max(t, 1e-4) * 0.5
        rings.append([(h, yt, z), (-h, yt, z), (-h, yb, z), (h, yb, z)])
    rings = np.array(rings, dtype=float)
    tris = []
    n = len(rings)
    # rings run counter-clockwise seen from +z, so these windings all face outward
    for i in range(n - 1):
        for k in range(4):
            a, b = rings[i][k], rings[i][(k + 1) % 4]
            c, d = rings[i + 1][(k + 1) % 4], rings[i + 1][k]
            tris += [(a, b, c), (a, c, d)]
    r0, r1 = rings[0], rings[-1]
    tris += [(r0[0], r0[3], r0[2]), (r0[0], r0[2], r0[1])]      # poll, facing -z
    tris += [(r1[0], r1[1], r1[2]), (r1[0], r1[2], r1[3])]      # edge, facing +z
    tri = np.array(tris, dtype=float)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    P = tri.reshape(-1, 3)
    N = np.repeat(fn, 3, axis=0)
    col = np.tile(np.array(color, dtype=float), (len(P), 1))
    if edge_color is not None:
        zmax = float(np.max(zs))
        bright = P[:, 2] > zmax - edge_band
        col[bright, :3] = np.array(edge_color[:3])
    m = MeshData(_pack(P, N, np.zeros((len(P), 2)), col), np.arange(len(P)))
    return drop_degenerate(m)


def create_axe_mesh(steel: bool = False) -> MeshData:
    """A felling axe: ~62 cm hickory haft along +y with a head at the top whose cutting edge
    faces +z.  Steel: a drop-forged head with a broad, slightly bearded 13 cm bit and a
    polished edge bevel.  Stone: a ground greenstone celt lashed into a split haft."""
    wood = (0.55, 0.40, 0.24, 1.0)
    haft = create_frustum(0.0145, 0.012, 0.60, 9, 2, wood, centered=True)
    knob = create_frustum(0.017, 0.0145, 0.03, 9, 1, wood).transformed(_m_translate(0, -0.3, 0))
    parts = [haft, knob]
    top = 0.27
    if steel:
        zs = np.array([-0.035, -0.02, 0.0, 0.02, 0.045, 0.07, 0.095, 0.115, 0.122])
        yb = np.array([-0.024, -0.031, -0.032, -0.031, -0.026, -0.036, -0.056, -0.072, -0.070])
        yt = np.array([0.024, 0.031, 0.032, 0.031, 0.027, 0.035, 0.048, 0.058, 0.056])
        th = np.array([0.030, 0.034, 0.036, 0.032, 0.018, 0.012, 0.007, 0.003, 0.0008])
        head = _loft_blade(zs, yb, yt, th, (0.26, 0.27, 0.29, 1.0), (0.78, 0.80, 0.84, 1.0), 0.016)
    else:
        zs = np.array([-0.03, -0.01, 0.015, 0.045, 0.075, 0.095, 0.104])
        yb = np.array([-0.022, -0.03, -0.033, -0.034, -0.038, -0.042, -0.040])
        yt = np.array([0.022, 0.03, 0.033, 0.034, 0.037, 0.040, 0.038])
        th = np.array([0.030, 0.040, 0.044, 0.036, 0.022, 0.008, 0.0015])
        head = _loft_blade(zs, yb, yt, th, (0.30, 0.33, 0.29, 1.0), (0.42, 0.45, 0.40, 1.0), 0.012)
        lash = create_frustum(0.02, 0.02, 0.06, 10, 1, (0.50, 0.40, 0.27, 1.0), centered=True)
        parts.append(lash.transformed(_m_translate(0, top, 0)))
    parts.append(head.transformed(_m_translate(0, top, 0)))
    return merge(parts)


def create_gas_can_mesh() -> MeshData:
    can = create_box_mesh((0.10, 0.14, 0.06), (0.78, 0.12, 0.08, 1.0))
    spout = create_frustum(0.008, 0.006, 0.07, 6, 1, (0.15, 0.15, 0.15, 1.0))
    handle = create_box_mesh((0.06, 0.012, 0.02), (0.6, 0.1, 0.07, 1.0))
    return merge([can, spout.transformed(_m_translate(0.03, 0.07, 0) @ _m_rot_z(-0.6)), handle.transformed(_m_translate(-0.01, 0.078, 0))])
