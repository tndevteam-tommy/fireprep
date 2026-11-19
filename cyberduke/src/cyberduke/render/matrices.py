"""Projection and view matrices built with plain math (replaces the two GLU calls, removing
the dependency on the system libGLU which PyOpenGL does not bundle)."""

from __future__ import annotations

import math

from OpenGL.GL import glFrustum, glMultMatrixf, glTranslatef

from cyberduke.core.geometry import forward_vector


def apply_perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> None:
    """Equivalent of gluPerspective; multiplies the current (projection) matrix."""
    top = near * math.tan(math.radians(fov_y_deg) * 0.5)
    right = top * aspect
    glFrustum(-right, right, -top, top, near, far)


def apply_look_at(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> None:
    """Equivalent of gluLookAt; multiplies the current (modelview) matrix."""
    fx, fy, fz = target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]
    n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / n, fy / n, fz / n

    ux, uy, uz = up
    sx, sy, sz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
    n = math.sqrt(sx * sx + sy * sy + sz * sz) or 1.0
    sx, sy, sz = sx / n, sy / n, sz / n

    # recomputed up = s x f
    ux, uy, uz = sy * fz - sz * fy, sz * fx - sx * fz, sx * fy - sy * fx

    # column-major 4x4 as OpenGL expects
    m = (
        sx, ux, -fx, 0.0,
        sy, uy, -fy, 0.0,
        sz, uz, -fz, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )  # fmt: skip
    glMultMatrixf(m)
    glTranslatef(-eye[0], -eye[1], -eye[2])


def look_direction(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    """Unit forward vector for the *player camera* convention: yaw 0 = +X, yaw 90 = +Z.

    NOTE: NPCs use a different convention (yaw 0 = +Z, yaw 90 = +X, see core.geometry).
    Both live in ``core.geometry.forward_vector``; this wrapper is kept for callers.
    """
    return forward_vector(yaw_deg, pitch_deg, yaw_zero_is_x=True)
