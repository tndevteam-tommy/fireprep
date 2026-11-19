"""Soft pools of light as triangle meshes.

A light pool (``Prop.disc``: a mushroom's glow on the moss, a lantern's on the asphalt,
the splash of the same glow up a wall) is a disc whose opacity is full out to ``inner`` of
its radius and fades to nothing at the rim. Drawn one by one as fans they cost a draw call
each and could not stop at the edge of what they lie on; here each disc becomes a small
grid of triangles in world space - cut to its ``clip`` rectangle, so a pool ends exactly at
the edge of its planter - with the fade as a per-vertex alpha, and every pool of one
colour in a sector is baked into one mesh the backend keeps (``DrawContext.glow_mesh``).
"""

from __future__ import annotations

import math

import numpy as np

from cyberduke.world.props import Prop

GRID = 12  # cells across a whole disc (a clipped one keeps the same cell size)


def falloff(r: np.ndarray, inner: float) -> np.ndarray:
    """Opacity factor at ``r`` (in units of the radius): 1 out to ``inner``, then a linear
    fade to 0 at the rim - the same profile as ``DrawContext.soft_disc``."""
    if inner >= 1.0:
        return (r <= 1.0).astype(float)
    return np.clip((1.0 - r) / (1.0 - inner), 0.0, 1.0)


def _rotation_matrix(rotations) -> np.ndarray:
    """The 3x3 matrix of glRotatef-style rotations applied in order (as the renderer does:
    the last one acts on the point first)."""
    m = np.identity(3)
    for angle, ax, ay, az in rotations:
        n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        x, y, z = ax / n, ay / n, az / n
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        r = np.array(
            (
                (c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s),
                (y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s),
                (z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)),
            )
        )
        m = m @ r
    return m


def disc_triangles(prop: Prop) -> np.ndarray:
    """World-space triangles of a (possibly clipped) light-pool disc: an (N, 4) array of
    x, y, z and the opacity factor, three rows per triangle. Cells wholly outside the disc
    are dropped."""
    rx, rz, inner = prop.size
    u0, u1, v0, v1 = prop.clip or (-1.0, 1.0, -1.0, 1.0)
    u0, u1, v0, v1 = max(-1.0, u0), min(1.0, u1), max(-1.0, v0), min(1.0, v1)
    if u1 <= u0 or v1 <= v0:
        return np.zeros((0, 4))
    nu = max(1, math.ceil((u1 - u0) / (2.0 / GRID)))
    nv = max(1, math.ceil((v1 - v0) / (2.0 / GRID)))
    us = np.linspace(u0, u1, nu + 1)
    vs = np.linspace(v0, v1, nv + 1)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    alpha = falloff(np.hypot(uu, vv), inner)
    local = np.stack((uu * rx, np.zeros_like(uu), vv * rz), axis=-1)
    world = local @ _rotation_matrix(prop.rotate).T + (prop.x, prop.y, prop.z)
    pts = np.concatenate((world, alpha[..., None]), axis=-1)  # (nu+1, nv+1, 4)
    a, b = pts[:-1, :-1], pts[1:, :-1]
    c, d = pts[1:, 1:], pts[:-1, 1:]
    lit = (a[..., 3] + b[..., 3] + c[..., 3] + d[..., 3]) > 0.0
    quads = [q[lit] for q in (a, b, c, d)]
    tris = np.stack((quads[0], quads[1], quads[2], quads[0], quads[2], quads[3]), axis=1)
    return tris.reshape(-1, 4)


def pool_triangles(props) -> np.ndarray:
    """Every pool's triangles as one mesh, each pool's own opacity (its material's alpha)
    multiplied into its vertices - one mesh serves every pool of a colour."""
    parts = []
    for p in props:
        tris = disc_triangles(p)
        tris[:, 3] *= p.material.alpha
        parts.append(tris)
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.zeros((0, 4))
