"""
Firestarter - Rendering math & GL helpers (column-vector convention, numpy float32).
Matrices are uploaded transposed (row-major numpy -> column-major GLSL).
"""

import math
from typing import Tuple

import numpy as np


def perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(left: float, right: float, bottom: float, top: float, near: float, far: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    f = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    f /= max(1e-9, np.linalg.norm(f))
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    s = np.cross(f, up)
    s /= max(1e-9, np.linalg.norm(s))
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def scale(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.identity(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


def rot_x(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = np.identity(4, dtype=np.float32)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def rot_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = np.identity(4, dtype=np.float32)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_z(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = np.identity(4, dtype=np.float32)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def align_y_to(direction: np.ndarray) -> np.ndarray:
    """Rotation that maps +Y onto `direction` (unit)."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(1e-9, np.linalg.norm(d))
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = float(np.dot(y, d))
    m = np.identity(4, dtype=np.float32)
    if np.linalg.norm(v) < 1e-8:
        if c < 0:
            m[1, 1] = -1.0
            m[0, 0] = -1.0
        return m
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.identity(3) + vx + vx @ vx * (1.0 / (1.0 + c))
    m[:3, :3] = r
    return m


def orientation_model(position: np.ndarray, axis: np.ndarray, roll_deg: float = 0.0,
                      sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> np.ndarray:
    """Model matrix for a Y-aligned primitive whose long axis should follow `axis`."""
    return translate(*position) @ align_y_to(axis) @ rot_y(roll_deg) @ scale(sx, sy, sz)


def to_gl(m: np.ndarray) -> bytes:
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


def view_basis(view: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    right = np.array(view[0, :3], dtype=np.float64)
    up = np.array(view[1, :3], dtype=np.float64)
    fwd = -np.array(view[2, :3], dtype=np.float64)
    return right, up, fwd


def blackbody_rgb(temp_c: float) -> np.ndarray:
    """Approximate incandescent colour (linear RGB, normalised) of a body at temp_c."""
    t = (temp_c + 273.15) / 100.0
    if t <= 66:
        r = 1.0
        g = max(0.0, min(1.0, (99.4708025861 * math.log(max(t, 1.0)) - 161.1195681661) / 255.0))
    else:
        r = max(0.0, min(1.0, 329.698727446 * ((t - 60) ** -0.1332047592) / 255.0))
        g = max(0.0, min(1.0, 288.1221695283 * ((t - 60) ** -0.0755148492) / 255.0))
    if t >= 66:
        b = 1.0
    elif t <= 19:
        b = 0.0
    else:
        b = max(0.0, min(1.0, (138.5177312231 * math.log(t - 10) - 305.0447927307) / 255.0))
    return np.array([r, g, b]) ** 2.2
