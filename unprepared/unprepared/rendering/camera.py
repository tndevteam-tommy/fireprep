"""Projection / view helpers implemented in plain OpenGL (no GLU dependency).

GLU is deprecated and frequently missing on modern Linux and macOS installs;
these functions build the exact same matrices Mesa's ``gluPerspective`` and
``gluLookAt`` do.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from OpenGL.GL import glMultMatrixd, glMultMatrixf, glTranslated


def perspective_matrix(fovy_degrees: float, aspect: float, z_near: float, z_far: float) -> List[float]:
    """Column-major 4x4 perspective matrix (identical maths to Mesa GLU)."""
    radians = fovy_degrees / 2.0 * math.pi / 180.0
    delta_z = z_far - z_near
    sine = math.sin(radians)
    if delta_z == 0 or sine == 0 or aspect == 0:
        raise ValueError("degenerate perspective parameters")
    cot = math.cos(radians) / sine
    m = [0.0] * 16
    m[0] = cot / aspect
    m[5] = cot
    m[10] = -(z_far + z_near) / delta_z
    m[11] = -1.0
    m[14] = -2.0 * z_near * z_far / delta_z
    return m


def look_at_matrix(eye: Tuple[float, float, float], center: Tuple[float, float, float],
                   up: Tuple[float, float, float]) -> List[float]:
    """Column-major rotation part of a look-at view matrix (translation applied separately)."""
    fx, fy, fz = center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]
    mag = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / mag, fy / mag, fz / mag
    ux, uy, uz = up
    sx, sy, sz = fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux
    mag = math.sqrt(sx * sx + sy * sy + sz * sz)
    sx, sy, sz = sx / mag, sy / mag, sz / mag
    upx, upy, upz = sy * fz - sz * fy, sz * fx - sx * fz, sx * fy - sy * fx
    return [sx, upx, -fx, 0.0,
            sy, upy, -fy, 0.0,
            sz, upz, -fz, 0.0,
            0.0, 0.0, 0.0, 1.0]


def project(point, eye, center, fovy_degrees: float, aspect: float, width: int, height: int,
            up=(0.0, 1.0, 0.0)):
    """World point -> (pixel x, pixel y, eye-space depth) for the camera looking from ``eye`` at
    ``center``; None when the point is behind the camera. Pixel y grows downwards."""
    m = look_at_matrix(eye, center, up)
    dx, dy, dz = point[0] - eye[0], point[1] - eye[1], point[2] - eye[2]
    ex = m[0] * dx + m[4] * dy + m[8] * dz
    ey = m[1] * dx + m[5] * dy + m[9] * dz
    ez = m[2] * dx + m[6] * dy + m[10] * dz
    if ez > -0.05:
        return None
    cot = 1.0 / math.tan(math.radians(fovy_degrees) / 2.0)
    ndc_x = (cot / aspect) * ex / -ez
    ndc_y = cot * ey / -ez
    return ((ndc_x + 1.0) * 0.5 * width, (1.0 - ndc_y) * 0.5 * height, -ez)


def apply_perspective(fovy_degrees: float, aspect: float, z_near: float, z_far: float) -> None:
    glMultMatrixd(perspective_matrix(fovy_degrees, aspect, z_near, z_far))


def apply_look_at(eye, center, up=(0.0, 1.0, 0.0)) -> None:
    glMultMatrixf(look_at_matrix(eye, center, up))
    glTranslated(-eye[0], -eye[1], -eye[2])


class Frustum:
    """What the camera can see, as planes through the eye - left, right, top, bottom and the near
    side - facing inwards (world space). ``sees_box`` is the usual "positive vertex" test: a box
    is out of view if it lies wholly behind any one plane. ``margin`` degrees widen the view all
    round, so nothing pops at the edges (a crown of leaves wider than the trunk it is filed under,
    a roof's overhang)."""

    def __init__(self, eye, center, fovy_degrees: float, aspect: float, margin: float = 6.0, up=(0.0, 1.0, 0.0)):
        m = look_at_matrix(eye, center, up)
        s = (m[0], m[4], m[8])
        u = (m[1], m[5], m[9])
        f = (-m[2], -m[6], -m[10])
        half_v = math.radians(min(89.0, fovy_degrees / 2.0 + margin))
        half_h = min(math.radians(89.0), math.atan(math.tan(math.radians(fovy_degrees) / 2.0) * aspect) + math.radians(margin))
        ch, sh, cv, sv = math.cos(half_h), math.sin(half_h), math.cos(half_v), math.sin(half_v)
        self.eye = tuple(eye)
        self.planes = [
            tuple(f[i] * sh + s[i] * ch for i in range(3)),      # left
            tuple(f[i] * sh - s[i] * ch for i in range(3)),      # right
            tuple(f[i] * sv + u[i] * cv for i in range(3)),      # bottom
            tuple(f[i] * sv - u[i] * cv for i in range(3)),      # top
        ]

    def sees_box(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> bool:
        ex, ey, ez = self.eye
        if x1 <= ex <= x2 and y1 <= ey <= y2 and z1 <= ez <= z2:
            return True
        for nx, ny, nz in self.planes:
            px = x2 if nx >= 0.0 else x1
            py = y2 if ny >= 0.0 else y1
            pz = z2 if nz >= 0.0 else z1
            if nx * (px - ex) + ny * (py - ey) + nz * (pz - ez) < 0.0:
                return False
        return True

    def sees_sphere(self, x: float, y: float, z: float, radius: float) -> bool:
        ex, ey, ez = self.eye
        dx, dy, dz = x - ex, y - ey, z - ez
        for nx, ny, nz in self.planes:
            if nx * dx + ny * dy + nz * dz < -radius:
                return False
        return True
