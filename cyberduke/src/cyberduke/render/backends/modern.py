"""Modern backend: vertex buffer objects + GLSL shaders (OpenGL 2.1 / GLSL 1.20 baseline).

How it maps the ``DrawContext`` API:

* Matrix stack   - a NumPy 4x4 stack; the current model matrix is a uniform for dynamic
                   draws and is baked into vertex positions for cached static geometry.
* Materials      - colour / emission / specular are per-vertex attributes in static batches
                   and uniforms for dynamic draws (the shader combines both).
* Lighting       - the fixed-function model (global ambient + up to 8 attenuated point lights
                   with ambient/diffuse/specular terms, then modulated by the texture),
                   evaluated per pixel instead of per vertex.
* ``cached``     - the build function is run once in *capture* mode; every cube/cylinder it
                   emits is appended to a per-texture batch which becomes one VBO each.
                   Replaying a sector is then one draw call per texture.
* Lines/overlay  - small dedicated shaders with streamed vertex data.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders

from cyberduke.render.context import RGBA, DrawContext, Vec3
from cyberduke.world.lights import Light

MAX_LIGHTS = 8
MAX_CACHED_CONES = 256  # distinct cone shapes kept as VBOs (a level uses a handful)
GLOBAL_AMBIENT = (0.25, 0.31, 0.44)  # same base as the fixed-function path
FLOAT = np.float32

# Interleaved vertex layout shared by static batches and the unit primitives
# pos(3) normal(3) uv(2) color(3) emission(3) spec(2: strength, shininess)
STRIDE_FLOATS = 16
STRIDE_BYTES = STRIDE_FLOATS * 4
ATTRIBS = (  # name, size, offset (floats)
    ("a_pos", 3, 0),
    ("a_normal", 3, 3),
    ("a_uv", 2, 6),
    ("a_color", 3, 8),
    ("a_emission", 3, 11),
    ("a_spec", 2, 14),
)

# ----------------------------------------------------------------------------- shaders
LIT_VERT = """
#version 120
uniform mat4 u_proj;
uniform mat4 u_view;
uniform mat4 u_model;
uniform mat3 u_normal;      // inverse-transpose of (view * model)
uniform vec3 u_color;
uniform vec3 u_emission;
uniform vec2 u_spec;
uniform float u_alpha;
// Tiled textures on dynamic draws: repeats along the shape's local x, y, z (a cube) or
// around / along / across the caps (a cylinder); w = 0 none, 1 cube, 2 cylinder.
uniform vec4 u_uv_repeat;
attribute vec3 a_pos;
attribute vec3 a_normal;
attribute vec2 a_uv;
attribute vec3 a_color;
attribute vec3 a_emission;
attribute vec2 a_spec;
varying vec3 v_pos;
varying vec3 v_normal;
varying vec2 v_uv;
varying vec3 v_color;
varying vec3 v_emission;
varying vec2 v_spec;
varying float v_alpha;
varying float v_depth;
void main() {
    v_alpha = u_alpha;
    vec4 pe = u_view * u_model * vec4(a_pos, 1.0);
    v_pos = pe.xyz;
    v_normal = u_normal * a_normal;
    vec2 uv = a_uv;
    if (u_uv_repeat.w > 1.5) {
        if (abs(a_normal.y) > 0.5) uv = (uv - 0.5) * u_uv_repeat.z + 0.5;
        else uv *= u_uv_repeat.xy;
    } else if (u_uv_repeat.w > 0.5) {
        vec3 n = abs(a_normal);
        if (n.z > 0.5) uv *= u_uv_repeat.xy;
        else if (n.y > 0.5) uv *= u_uv_repeat.xz;
        else uv *= u_uv_repeat.zy;
    }
    v_uv = uv;
    v_color = a_color * u_color;
    v_emission = a_emission + u_emission;
    v_spec = a_spec + u_spec;
    gl_Position = u_proj * pe;
    v_depth = gl_Position.w;
}
"""

LIT_FRAG = """
#version 120
uniform sampler2D u_texture;
uniform vec3 u_global_ambient;
uniform int u_light_count;
uniform vec3 u_light_pos[8];       // eye space
uniform vec3 u_light_ambient[8];
uniform vec3 u_light_diffuse[8];
uniform vec3 u_light_specular[8];
uniform vec3 u_light_atten[8];     // constant, linear, quadratic
varying vec3 v_pos;
varying vec3 v_normal;
varying vec2 v_uv;
varying vec3 v_color;
varying vec3 v_emission;
varying vec2 v_spec;
varying float v_alpha;
varying float v_depth;
uniform float u_log_depth;
uniform vec4 u_fog;  // rgb: the fog's colour, a: its density (per metre; 0 none)
void main() {
    // Logarithmic depth: with the usual hyperbolic depth a 16-bit depth buffer (what SDL
    // asks for unless told otherwise) cannot separate a neon sign from the panel 2 cm
    // behind it 20 m away, and the panel shows through in flickering black bands.
    gl_FragDepth = log2(max(1e-6, 1.0 + v_depth)) * u_log_depth;
    vec3 n = normalize(v_normal);
    vec3 v = normalize(-v_pos);
    vec3 lit = v_emission + u_global_ambient * v_color;
    for (int i = 0; i < 8; i++) {
        if (i >= u_light_count) break;
        vec3 to_light = u_light_pos[i] - v_pos;
        float d = length(to_light);
        vec3 l = to_light / max(d, 1e-4);
        vec3 k = u_light_atten[i];
        float att = 1.0 / (k.x + k.y * d + k.z * d * d);
        float ndotl = max(dot(n, l), 0.0);
        float spec = 0.0;
        if (ndotl > 0.0 && v_spec.x > 0.0) {
            vec3 h = normalize(l + v);
            spec = pow(max(dot(n, h), 0.0), max(v_spec.y, 1.0)) * v_spec.x;
        }
        lit += att * (u_light_ambient[i] * v_color + u_light_diffuse[i] * v_color * ndotl
                      + u_light_specular[i] * spec);
    }
    lit = clamp(lit, 0.0, 1.0);
    vec4 tex = texture2D(u_texture, v_uv);
    vec3 rgb = lit * tex.rgb;
    if (u_fog.a > 0.0) {
        float k = u_fog.a * length(v_pos);
        float f = exp(-k * k);
        // what glows of itself (neon, a screen) shows through further
        float glow = clamp(max(v_emission.r, max(v_emission.g, v_emission.b)), 0.0, 1.0);
        f = mix(f, sqrt(f), glow);
        rgb = mix(u_fog.rgb, rgb, f);
    }
    gl_FragColor = vec4(rgb, v_alpha);
}
"""

FLAT_VERT = """
#version 120
uniform mat4 u_mvp;
attribute vec3 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
varying float v_depth;
void main() {
    v_uv = a_uv;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_depth = gl_Position.w;
}
"""

FLAT_FRAG = """
#version 120
uniform sampler2D u_texture;
uniform vec4 u_color;
uniform float u_use_texture;
uniform float u_log_depth;
uniform float u_fog_density;  // (the lit pass's fog: thins what is drawn out in it)
varying vec2 v_uv;
varying float v_depth;
void main() {
    gl_FragDepth = log2(max(1e-6, 1.0 + v_depth)) * u_log_depth;  // same depth as the lit pass
    vec4 t = texture2D(u_texture, v_uv);
    gl_FragColor = mix(u_color, t * vec4(1.0, 1.0, 1.0, u_color.a), u_use_texture);
    float k = u_fog_density * v_depth;
    gl_FragColor.a *= exp(-k * k);
}
"""


GLOW_FRAG = """
#version 120
uniform vec4 u_color;
uniform float u_log_depth;
uniform float u_fog_density;
varying vec2 v_uv;
varying float v_depth;
void main() {
    gl_FragDepth = log2(max(1e-6, 1.0 + v_depth)) * u_log_depth;
    float k = u_fog_density * v_depth;
    gl_FragColor = vec4(u_color.rgb, u_color.a * v_uv.x * exp(-k * k));
}
"""


def log_depth_coefficient(far: float) -> float:
    """Scale that maps view depth 0..far onto 0..1 in the shaders' logarithmic depth."""
    return 1.0 / math.log2(far + 1.0)


# ----------------------------------------------------------------------------- maths
def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=FLOAT)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 1.0, 0.0)) -> np.ndarray:
    e = np.array(eye, dtype=np.float64)
    f = np.array(target, dtype=np.float64) - e
    f /= np.linalg.norm(f) or 1.0
    s = np.cross(f, np.array(up, dtype=np.float64))
    s /= np.linalg.norm(s) or 1.0
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ e
    return m.astype(FLOAT)


def ortho(width: float, height: float) -> np.ndarray:
    """Pixel space with y down (matches glOrtho(0, w, h, 0, -1, 1))."""
    m = np.identity(4, dtype=FLOAT)
    m[0, 0] = 2.0 / width
    m[1, 1] = -2.0 / height
    m[0, 3] = -1.0
    m[1, 3] = 1.0
    return m


def translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.identity(4, dtype=FLOAT)
    m[:3, 3] = (x, y, z)
    return m


def scaling(x: float, y: float, z: float) -> np.ndarray:
    return np.diag((x, y, z, 1.0)).astype(FLOAT)


def rotation(angle_deg: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    t = 1.0 - c
    m = np.identity(4, dtype=FLOAT)
    m[:3, :3] = (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )
    return m


def normal_matrix(m: np.ndarray) -> np.ndarray:
    """Inverse-transpose of the upper 3x3 (correct normals under non-uniform scale)."""
    a = m[:3, :3].astype(np.float64)
    try:
        return np.linalg.inv(a).T.astype(FLOAT)
    except np.linalg.LinAlgError:
        return np.identity(3, dtype=FLOAT)


# ----------------------------------------------------------------------------- geometry
def unit_cube_vertices() -> np.ndarray:
    """24 vertices (4 per face) with the same winding/uv layout as the legacy primitive."""
    faces = [
        # normal, 4 corners (uv 00, 10, 11, 01)
        ((0, 0, 1), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0, 0, -1), [(0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)]),
        ((0, 1, 0), [(-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
        ((0, -1, 0), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5)]),
        ((1, 0, 0), [(0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]),
        ((-1, 0, 0), [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)]),
    ]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    out = np.zeros((24, 8), dtype=FLOAT)
    i = 0
    for normal, corners in faces:
        for corner, uv in zip(corners, uvs, strict=True):
            out[i, 0:3] = corner
            out[i, 3:6] = normal
            out[i, 6:8] = uv
            i += 1
    return out


def quad_indices(n_quads: int, base: int = 0) -> np.ndarray:
    q = np.arange(n_quads, dtype=np.uint32) * 4 + base
    return np.stack([q, q + 1, q + 2, q, q + 2, q + 3], axis=1).reshape(-1)


def unit_cylinder_vertices(slices: int) -> tuple[np.ndarray, np.ndarray]:
    """Radius 1, height 1 (y from 0 to 1): side quads plus two caps. Returns (verts, indices)."""
    verts = []
    idx = []
    for i in range(slices):
        u1, u2 = i / slices, (i + 1) / slices
        th1, th2 = 2 * math.pi * u1, 2 * math.pi * u2
        c1, s1, c2, s2 = math.cos(th1), math.sin(th1), math.cos(th2), math.sin(th2)
        base = len(verts)
        verts += [
            (c1, 0.0, s1, c1, 0.0, s1, u1, 0.0),
            (c2, 0.0, s2, c2, 0.0, s2, u2, 0.0),
            (c2, 1.0, s2, c2, 0.0, s2, u2, 1.0),
            (c1, 1.0, s1, c1, 0.0, s1, u1, 1.0),
        ]
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    for y, ny, sign in ((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)):
        centre = len(verts)
        verts.append((0.0, y, 0.0, 0.0, ny, 0.0, 0.5, 0.5))
        ring = []
        for i in range(slices + 1):
            th = sign * 2 * math.pi * i / slices
            c, s = math.cos(th), math.sin(th)
            ring.append(len(verts))
            verts.append((c, y, s, 0.0, ny, 0.0, 0.5 + 0.5 * c, 0.5 + 0.5 * s))
        for i in range(slices):
            idx += [centre, ring[i], ring[i + 1]]
    return np.array(verts, dtype=FLOAT), np.array(idx, dtype=np.uint32)


def _soft_disc_vertices(slices: int, inner: float) -> np.ndarray:
    """Triangles for a unit disc in XZ with the alpha factor in uv.x: 1 in the centre and
    out to ``inner``, 0 at the rim (8 floats per vertex, the flat-shader layout)."""
    tris: list[tuple[float, float, float, float]] = []  # x, z, alpha
    rings = [(0.0, 1.0), (1.0, 0.0)] if inner <= 0.0 else [(0.0, 1.0), (inner, 1.0), (1.0, 0.0)]
    for (r0, a0), (r1, a1) in zip(rings, rings[1:], strict=False):
        for i in range(slices):
            t0, t1 = 2 * math.pi * i / slices, 2 * math.pi * (i + 1) / slices
            p00 = (math.cos(t0) * r0, math.sin(t0) * r0, a0)
            p01 = (math.cos(t1) * r0, math.sin(t1) * r0, a0)
            p10 = (math.cos(t0) * r1, math.sin(t0) * r1, a1)
            p11 = (math.cos(t1) * r1, math.sin(t1) * r1, a1)
            if r0 == 0.0:
                tris += [p00, p10, p11]
            else:
                tris += [p00, p10, p11, p00, p11, p01]
    out = np.zeros((len(tris), 8), dtype=FLOAT)
    for i, (x, z, a) in enumerate(tris):
        out[i, 0], out[i, 2], out[i, 6] = x, z, a
    return out


def tiled_uvs(base: np.ndarray, mode: int, rep) -> np.ndarray:
    """The uv columns of ``base`` scaled for a tiled texture, exactly as the vertex shader
    does it for dynamic draws: mode 1 = cube (repeats along local x, y, z, picked per face
    from the normal), mode 2 = cylinder (around / along the side, across the caps)."""
    uv = base[:, 6:8].astype(FLOAT, copy=True)
    if mode == 0:
        return uv
    n = np.abs(base[:, 3:6])
    if mode == 1:
        su = np.where(n[:, 2] > 0.5, rep[0], np.where(n[:, 1] > 0.5, rep[0], rep[2]))
        sv = np.where(n[:, 2] > 0.5, rep[1], np.where(n[:, 1] > 0.5, rep[2], rep[1]))
        uv[:, 0] *= su
        uv[:, 1] *= sv
        return uv
    cap = n[:, 1] > 0.5
    uv[cap] = (uv[cap] - 0.5) * rep[2] + 0.5
    uv[~cap, 0] *= rep[0]
    uv[~cap, 1] *= rep[1]
    return uv


def with_material(base: np.ndarray, color, emission, spec) -> np.ndarray:
    """Expands pos/normal/uv rows (8 floats) to the full 16-float layout."""
    out = np.zeros((len(base), STRIDE_FLOATS), dtype=FLOAT)
    out[:, :8] = base
    out[:, 8:11] = color
    out[:, 11:14] = emission
    out[:, 14:16] = spec
    return out


# ----------------------------------------------------------------------------- GPU objects
class Program:
    def __init__(self, vert: str, frag: str, attribs: Sequence[str]):
        vs = shaders.compileShader(vert, GL_VERTEX_SHADER)
        fs = shaders.compileShader(frag, GL_FRAGMENT_SHADER)
        self.id = int(glCreateProgram())
        glAttachShader(self.id, vs)
        glAttachShader(self.id, fs)
        for loc, name in enumerate(attribs):
            glBindAttribLocation(self.id, loc, name)
        glLinkProgram(self.id)
        if glGetProgramiv(self.id, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.id))
        glDeleteShader(vs)
        glDeleteShader(fs)
        self._uniforms: dict[str, int] = {}

    def use(self) -> None:
        glUseProgram(self.id)

    def loc(self, name: str) -> int:
        if name not in self._uniforms:
            self._uniforms[name] = glGetUniformLocation(self.id, name)
        return self._uniforms[name]

    def mat4(self, name: str, m: np.ndarray) -> None:
        glUniformMatrix4fv(self.loc(name), 1, GL_TRUE, np.ascontiguousarray(m, dtype=FLOAT))

    def mat3(self, name: str, m: np.ndarray) -> None:
        glUniformMatrix3fv(self.loc(name), 1, GL_TRUE, np.ascontiguousarray(m, dtype=FLOAT))

    def vec3(self, name: str, v) -> None:
        glUniform3f(self.loc(name), *v)

    def vec2(self, name: str, v) -> None:
        glUniform2f(self.loc(name), *v)

    def vec4(self, name: str, v) -> None:
        glUniform4f(self.loc(name), *v)

    def float(self, name: str, v: float) -> None:
        glUniform1f(self.loc(name), v)

    def int(self, name: str, v: int) -> None:
        glUniform1i(self.loc(name), v)


class VertexBuffer:
    """Interleaved VBO (+ index buffer) with the shared attribute layout.

    Attribute pointers are (re)specified only when a different buffer is bound - the
    context tracks the current one - which removes six GL calls from most draws.
    """

    bound: VertexBuffer | None = None  # class-level: the buffer whose pointers are set up

    def __init__(self, vertices: np.ndarray, indices: np.ndarray, attrib_count: int = len(ATTRIBS)):
        self.vbo = int(glGenBuffers(1))
        self.ibo = int(glGenBuffers(1))
        self.count = int(len(indices))
        self.attrib_count = attrib_count
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        v = np.ascontiguousarray(vertices, dtype=FLOAT)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        i = np.ascontiguousarray(indices, dtype=np.uint32)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i.nbytes, i, GL_STATIC_DRAW)
        VertexBuffer.bound = None

    def bind(self) -> None:
        if VertexBuffer.bound is self:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        for loc, (_, size, offset) in enumerate(ATTRIBS[: self.attrib_count]):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(
                loc, size, GL_FLOAT, GL_FALSE, STRIDE_BYTES, ctypes.c_void_p(offset * 4)
            )
        for loc in range(self.attrib_count, len(ATTRIBS)):
            glDisableVertexAttribArray(loc)
        VertexBuffer.bound = self

    def draw(self, mode: int = GL_TRIANGLES, first: int = 0, count: int | None = None) -> None:
        self.bind()
        n = self.count - first if count is None else count
        glDrawElements(mode, n, GL_UNSIGNED_INT, ctypes.c_void_p(first * 4))

    def delete(self) -> None:
        if VertexBuffer.bound is self:
            VertexBuffer.bound = None
        glDeleteBuffers(2, [self.vbo, self.ibo])


class StreamBuffer(VertexBuffer):
    """A vertex + index buffer refilled every time it is drawn (a person's batch)."""

    def __init__(self) -> None:
        self.vbo = int(glGenBuffers(1))
        self.ibo = int(glGenBuffers(1))
        self.count = 0
        self.attrib_count = len(ATTRIBS)

    def update(self, vertices: np.ndarray, indices: np.ndarray) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        v = np.ascontiguousarray(vertices, dtype=FLOAT)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v, GL_STREAM_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        i = np.ascontiguousarray(indices, dtype=np.uint32)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i.nbytes, i, GL_STREAM_DRAW)
        self.count = int(len(i))
        VertexBuffer.bound = None


Mesh = tuple[np.ndarray, np.ndarray, VertexBuffer]  # base vertices, indices, GPU buffer


class StaticBatch:
    """All of one cached key's geometry in a single VBO, drawn as one index range per texture."""

    def __init__(self, parts: dict[int, list[tuple[np.ndarray, np.ndarray]]]):
        verts, idx, base = [], [], 0
        self.ranges: list[tuple[int, int, int]] = []  # (texture, first index, index count)
        for tex, pieces in parts.items():
            first = sum(len(i) for i in idx)
            for v, i in pieces:
                verts.append(v)
                idx.append(i + base)
                base += len(v)
            self.ranges.append((tex, first, sum(len(i) for i in idx) - first))
        self.vbo = VertexBuffer(np.concatenate(verts), np.concatenate(idx)) if verts else None

    def draw(self) -> None:
        if self.vbo is None:
            return
        for tex, first, count in self.ranges:
            glBindTexture(GL_TEXTURE_2D, tex)
            self.vbo.draw(GL_TRIANGLES, first, count)

    def delete(self) -> None:
        if self.vbo is not None:
            self.vbo.delete()


class BatchBuilder:
    """Accumulates world-space vertices per texture while a static build runs in capture mode."""

    def __init__(self) -> None:
        self.parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

    def add(self, tex: int, vertices: np.ndarray, indices: np.ndarray) -> None:
        self.parts.setdefault(tex, []).append((vertices, indices))

    def build(self) -> StaticBatch:
        return StaticBatch(self.parts)


# ----------------------------------------------------------------------------- context
class ModernContext(DrawContext):
    name = "modern"
    batches = True  # people are drawn as one batch each (``render.batch``)

    def __init__(
        self,
        width: int,
        height: int,
        fov_y: float,
        near: float,
        far: float,
        lights: Sequence[Light],
    ):
        if len(lights) > MAX_LIGHTS:
            raise ValueError(f"modern backend supports {MAX_LIGHTS} lights, got {len(lights)}")
        self.width, self.height = width, height
        glViewport(0, 0, width, height)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.lit = Program(LIT_VERT, LIT_FRAG, [a[0] for a in ATTRIBS])
        self.flat = Program(FLAT_VERT, FLAT_FRAG, ["a_pos", "a_uv"])
        self.glow = Program(FLAT_VERT, GLOW_FRAG, ["a_pos", "a_uv"])
        self.fov_y, self.near, self.far = fov_y, near, far
        self.proj = perspective(fov_y, width / height, near, far)
        self.view = np.identity(4, dtype=FLOAT)
        self.lights = list(lights)
        self.global_ambient = GLOBAL_AMBIENT
        self.fog = (0.0, 0.0, 0.0, 0.0)

        # Resources
        self.white_tex = self._white_texture()
        cube = unit_cube_vertices()
        self.cube_vbo = VertexBuffer(
            with_material(cube, (1, 1, 1), (0, 0, 0), (0, 0)), quad_indices(6)
        )
        self._cylinders: dict[int, VertexBuffer] = {}
        self._cones: dict[tuple[int, float, float], Mesh] = {}
        self._discs: dict[tuple[int, float], tuple[int, int]] = {}  # key -> (buffer, vertices)
        self._glows: dict[Any, tuple[int, int]] = {}  # baked light pools: (buffer, vertices)
        self._cylinder_base: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._cube_base = cube
        self._cube_indices = quad_indices(6)
        self._line_vbo = int(glGenBuffers(1))
        self._shapes: dict[str, Mesh] = {}
        self._stream: StreamBuffer | None = None

        # State
        self._stack: list[np.ndarray] = []
        self.model = np.identity(4, dtype=FLOAT)
        self._color = (1.0, 1.0, 1.0)
        self._emission = (0.0, 0.0, 0.0)
        self._spec = (0.0, 0.0)
        self._alpha = 1.0
        self._capture: BatchBuilder | None = None
        self._static: dict[Any, StaticBatch] = {}
        self._overlay: np.ndarray | None = None

        self._upload_static_uniforms()

    # ------------------------------------------------------------------ setup
    def _white_texture(self) -> int:
        tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, bytes((255, 255, 255))
        )
        return tex

    # ------------------------------------------------------------------ quality
    def set_view_distance(self, far: float) -> None:
        if far == self.far:
            return
        self.far = far
        self.proj = perspective(self.fov_y, self.width / self.height, self.near, far)
        self.lit.use()
        self.lit.mat4("u_proj", self.proj)
        self._upload_log_depth()

    def _upload_log_depth(self) -> None:
        coefficient = log_depth_coefficient(self.far)
        for program in (self.lit, self.flat, self.glow):
            program.use()
            program.float("u_log_depth", coefficient)

    def set_lights(self, lights: Sequence[Light]) -> None:
        if len(lights) > MAX_LIGHTS:
            raise ValueError(f"modern backend supports {MAX_LIGHTS} lights, got {len(lights)}")
        self.lights = list(lights)
        self._upload_static_uniforms()

    def set_sky(self, rgb) -> None:
        glClearColor(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)

    def set_global_ambient(self, rgb) -> None:
        self.global_ambient = tuple(rgb)
        self.lit.use()
        self.lit.vec3("u_global_ambient", self.global_ambient)

    fog_capable = True  # (per pixel, in the shaders)

    def set_fog(self, rgb, density: float) -> None:
        fog = (float(rgb[0]), float(rgb[1]), float(rgb[2]), max(0.0, float(density)))
        if fog == self.fog:
            return
        self.fog = fog
        self._upload_fog()

    def _upload_fog(self, overlay: bool = False) -> None:
        self.lit.use()
        self.lit.vec4("u_fog", self.fog)
        for program in (self.flat, self.glow):
            program.use()
            program.float("u_fog_density", 0.0 if overlay else self.fog[3])

    def _upload_static_uniforms(self) -> None:
        p = self.lit
        p.use()
        p.mat4("u_proj", self.proj)
        p.vec3("u_global_ambient", self.global_ambient)
        p.int("u_light_count", len(self.lights))
        p.int("u_texture", 0)
        for i, light in enumerate(self.lights):
            p.vec3(f"u_light_ambient[{i}]", light.ambient)
            p.vec3(f"u_light_diffuse[{i}]", light.diffuse)
            p.vec3(f"u_light_specular[{i}]", light.specular)
            p.vec3(
                f"u_light_atten[{i}]", (1.0, light.linear_attenuation, light.quadratic_attenuation)
            )
        self.flat.use()
        self.flat.int("u_texture", 0)
        self._upload_log_depth()
        self._upload_fog()

    # ------------------------------------------------------------------ frame
    def begin_frame(self, eye: Vec3, target: Vec3) -> None:
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.view = look_at(eye, target)
        self._stack.clear()
        self.model = np.identity(4, dtype=FLOAT)
        self.lit.use()
        self.lit.mat4("u_view", self.view)
        for i, light in enumerate(self.lights):
            pos_eye = self.view @ np.array((light.x, light.y, light.z, 1.0), dtype=FLOAT)
            self.lit.vec3(f"u_light_pos[{i}]", pos_eye[:3])

    def end_frame(self) -> None:
        pass

    # ------------------------------------------------------------------ matrix stack
    def push(self) -> None:
        self._stack.append(self.model)

    def pop(self) -> None:
        self.model = self._stack.pop()

    def translate(self, x: float, y: float, z: float) -> None:
        self.model = self.model @ translation(x, y, z)

    def rotate(self, angle_deg: float, x: float, y: float, z: float) -> None:
        self.model = self.model @ rotation(angle_deg, x, y, z)

    def scale(self, x: float, y: float, z: float) -> None:
        self.model = self.model @ scaling(x, y, z)

    # ------------------------------------------------------------------ material
    def color(self, r: float, g: float, b: float) -> None:
        self._color = (r, g, b)

    def emission(self, rgb: Sequence[float] | None) -> None:
        self._emission = (0.0, 0.0, 0.0) if rgb is None else tuple(float(c) for c in rgb[:3])

    def alpha(self, a: float) -> None:
        self._alpha = float(a)

    def depth_mask(self, enabled: bool) -> None:
        glDepthMask(GL_TRUE if enabled else GL_FALSE)

    def specular(self, rgb: Sequence[float] | None, shininess: float = 0.0) -> None:
        if rgb is None:
            self._spec = (0.0, 0.0)
        else:
            # fixed-function used a specular *colour*; we keep its mean as a strength
            self._spec = (float(sum(rgb[:3]) / 3.0), float(shininess))

    # ------------------------------------------------------------------ primitives
    def _emit(
        self,
        base: np.ndarray,
        indices: np.ndarray,
        vbo: VertexBuffer | None,
        tex_id: int | None,
        uv_mode: int = 0,
        uv_rep: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        tex = tex_id or self.white_tex
        if self._capture is not None:
            # Bake the transform into world space so the whole batch shares one draw call.
            pts = base[:, :3] @ self.model[:3, :3].T + self.model[:3, 3]
            nrm = base[:, 3:6] @ normal_matrix(self.model).T
            uvs = tiled_uvs(base, uv_mode, uv_rep)
            verts = np.concatenate([pts, nrm, uvs], axis=1)
            self._capture.add(
                tex, with_material(verts, self._color, self._emission, self._spec), indices
            )
            return
        assert vbo is not None
        p = self.lit
        p.use()
        p.mat4("u_model", self.model)
        p.mat3("u_normal", normal_matrix(self.view @ self.model))
        p.vec3("u_color", self._color)
        p.vec3("u_emission", self._emission)
        p.vec2("u_spec", self._spec)
        p.float("u_alpha", self._alpha)
        p.vec4("u_uv_repeat", (*uv_rep, float(uv_mode)))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        vbo.draw()

    def _axis_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.model[:3, :3], axis=0)

    def cube(self, tex_id: int | None = None) -> None:
        tile = self.texture_tiles.get(tex_id) if tex_id else None
        if tile:
            rep = self.cube_repeat(self._axis_lengths(), tile)
            self._emit(self._cube_base, self._cube_indices, self.cube_vbo, tex_id, 1, rep)
        else:
            self._emit(self._cube_base, self._cube_indices, self.cube_vbo, tex_id)

    def _cylinder(self, slices: int) -> tuple[np.ndarray, np.ndarray, VertexBuffer]:
        if slices not in self._cylinders:
            base, idx = unit_cylinder_vertices(slices)
            self._cylinder_base[slices] = (base, idx)
            self._cylinders[slices] = VertexBuffer(
                with_material(base, (1, 1, 1), (0, 0, 0), (0, 0)), idx
            )
        base, idx = self._cylinder_base[slices]
        return base, idx, self._cylinders[slices]

    def cylinder(
        self, r_bot: float, r_top: float, h: float, slices: int = 10, tex_id: int | None = None
    ) -> None:
        tile = self.texture_tiles.get(tex_id) if tex_id else None
        mode, rep = 0, (1.0, 1.0, 1.0)
        if tile:
            mode, rep = 2, self.cylinder_repeat(r_bot, r_top, h, self._axis_lengths(), tile)
        if r_bot != r_top:  # truncated cone (the mushroom caps, by the hundred)
            radius = max(r_bot, r_top)
            if radius <= 0.0:
                return
            base, idx, vbo, temporary = self._cone(slices, r_bot / radius, r_top / radius)
            with self.pushed():
                self.scale(radius, h, radius)
                self._emit(base, idx, vbo, tex_id, mode, rep)
            if temporary:
                vbo.delete()
            return
        base, idx, vbo = self._cylinder(slices)
        with self.pushed():
            self.scale(r_bot, h, r_bot)
            self._emit(base, idx, vbo, tex_id, mode, rep)

    def shape(self, name: str, tex_id: int | None = None) -> None:
        found = self._shapes.get(name)
        if found is None:
            from cyberduke.render.shapes import mesh

            base, idx = mesh(name)
            base = np.ascontiguousarray(base, dtype=FLOAT)
            vbo = VertexBuffer(with_material(base, (1, 1, 1), (0, 0, 0), (0, 0)), idx)
            found = self._shapes[name] = (base, idx, vbo)
        base, idx, vbo = found
        self._emit(base, idx, vbo, tex_id)

    def draw_batch(self, vertices: np.ndarray, indices: np.ndarray, ranges: Sequence[Any]) -> None:
        if not len(vertices):
            return
        data = np.ascontiguousarray(vertices, dtype=FLOAT)  # (already the 16-float layout)
        if self._capture is not None:  # (inside a cached build: bake it in like the rest)
            part = data.copy()
            part[:, :3] = data[:, :3] @ self.model[:3, :3].T + self.model[:3, 3]
            part[:, 3:6] = data[:, 3:6] @ normal_matrix(self.model).T
            for tex, first, count in ranges:
                self._capture.add(tex or self.white_tex, part, indices[first : first + count])
            return
        if self._stream is None:
            self._stream = StreamBuffer()
        stream = self._stream
        stream.update(data, indices)
        p = self.lit
        p.use()
        p.mat4("u_model", self.model)
        p.mat3("u_normal", normal_matrix(self.view @ self.model))
        p.vec3("u_color", (1.0, 1.0, 1.0))
        p.vec3("u_emission", (0.0, 0.0, 0.0))
        p.vec2("u_spec", (0.0, 0.0))
        p.float("u_alpha", self._alpha)
        p.vec4("u_uv_repeat", (1.0, 1.0, 1.0, 0.0))  # (the batch has its tiling baked in)
        glActiveTexture(GL_TEXTURE0)
        for tex, first, count in ranges:
            glBindTexture(GL_TEXTURE_2D, tex or self.white_tex)
            stream.draw(GL_TRIANGLES, first, count)

    def _cone(
        self, slices: int, bottom: float, top: float
    ) -> tuple[np.ndarray, np.ndarray, VertexBuffer, bool]:
        """A unit-height cone with radii ``bottom`` / ``top`` (the larger one is 1), cached
        per shape like the unit cylinders and scaled into place - building a throwaway VBO
        for every cone on every frame cost ~0.5 ms each, a third of a greenhouse frame.
        Returns (vertices, indices, vbo, temporary); a temporary VBO (the cache is full)
        must be deleted after drawing."""
        key = (slices, round(bottom, 3), round(top, 3))
        cached = self._cones.get(key)
        if cached is not None:
            return (*cached, False)
        base, idx = unit_cylinder_vertices(slices)
        cone = base.copy()
        scale = np.where(cone[:, 1] > 0.5, key[2], key[1])
        cone[:, 0] *= scale
        cone[:, 2] *= scale
        vbo = VertexBuffer(with_material(cone, (1, 1, 1), (0, 0, 0), (0, 0)), idx)
        if len(self._cones) >= MAX_CACHED_CONES:
            return cone, idx, vbo, True
        self._cones[key] = (cone, idx, vbo)
        return cone, idx, vbo, False

    def lines(self, segments: Sequence[tuple[Vec3, Vec3]], rgba: RGBA, width: float = 1.0) -> None:
        if len(segments) == 0:
            return
        if isinstance(segments, np.ndarray):  # (n, 2, 3): the rain, in one go
            pts = segments.reshape(-1, 3).astype(FLOAT)
        else:
            pts = np.array([p for seg in segments for p in seg], dtype=FLOAT)
        data = np.zeros((len(pts), 8), dtype=FLOAT)
        data[:, :3] = pts
        self._stream_flat(
            data,
            GL_LINES,
            rgba,
            textured=False,
            mvp=self.proj @ self.view @ self.model,
            width=width,
        )

    def soft_disc(self, rgba: RGBA, slices: int = 24, inner: float = 0.0) -> None:
        # The disc's triangles never change: each shape lives in its own static buffer
        # (hundreds of light pools used to be re-uploaded with glBufferData every frame).
        key = (slices, round(inner, 3))
        cached = self._discs.get(key)
        if cached is None:
            data = np.ascontiguousarray(_soft_disc_vertices(slices, inner))
            buffer = int(glGenBuffers(1))
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            cached = self._discs[key] = (buffer, len(data))
        buffer, count = cached
        p = self.glow
        p.use()
        p.mat4("u_mvp", self.proj @ self.view @ self.model)
        p.vec4("u_color", rgba)
        VertexBuffer.bound = None  # we are about to change buffer bindings and pointers
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        glDrawArrays(GL_TRIANGLES, 0, count)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    def _stream_flat(
        self,
        data: np.ndarray,
        mode: int,
        rgba: RGBA,
        textured: bool,
        mvp: np.ndarray,
        width: float = 1.0,
        program: Program | None = None,
    ) -> None:
        p = program or self.flat
        p.use()
        p.mat4("u_mvp", mvp)
        p.vec4("u_color", rgba)
        if p is self.flat:
            p.float("u_use_texture", 1.0 if textured else 0.0)
        VertexBuffer.bound = None  # we are about to change buffer bindings and pointers
        glBindBuffer(GL_ARRAY_BUFFER, self._line_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, np.ascontiguousarray(data), GL_STREAM_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        glLineWidth(width)
        glDrawArrays(mode, 0, len(data))
        glLineWidth(1.0)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    def glow_mesh(self, key: Any, rgba: RGBA, build: Callable[[], Any]) -> None:
        cached = self._glows.get(key)
        if cached is None:
            tris = np.asarray(build(), dtype=FLOAT).reshape(-1, 4)
            data = np.zeros((len(tris), 8), dtype=FLOAT)  # the glow shader's layout
            data[:, 0:3] = tris[:, 0:3]
            data[:, 6] = tris[:, 3]
            buffer = 0
            if len(data):
                buffer = int(glGenBuffers(1))
                glBindBuffer(GL_ARRAY_BUFFER, buffer)
                data = np.ascontiguousarray(data)
                glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            cached = self._glows[key] = (buffer, len(data))
        buffer, count = cached
        if not count:
            return
        p = self.glow
        p.use()
        p.mat4("u_mvp", self.proj @ self.view)
        p.vec4("u_color", rgba)
        VertexBuffer.bound = None
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
        glDrawArrays(GL_TRIANGLES, 0, count)
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)

    # ------------------------------------------------------------------ static caching
    def cached(self, key: Any, build: Callable[[], None]) -> None:
        batch = self._static.get(key)
        if batch is None:
            if self._capture is not None:
                raise RuntimeError(
                    "static geometry cannot be captured while another capture is running"
                )
            self._capture = BatchBuilder()
            saved = (self.model.copy(), self._color, self._emission, self._spec)
            try:
                build()
            finally:
                builder, self._capture = self._capture, None
                self.model, self._color, self._emission, self._spec = saved
            batch = builder.build()
            self._static[key] = batch
        p = self.lit
        p.use()
        p.mat4("u_model", np.identity(4, dtype=FLOAT))
        p.mat3("u_normal", normal_matrix(self.view))
        p.vec3("u_color", (1.0, 1.0, 1.0))
        p.vec3("u_emission", (0.0, 0.0, 0.0))
        p.vec2("u_spec", (0.0, 0.0))
        p.float("u_alpha", 1.0)
        p.vec4("u_uv_repeat", (1.0, 1.0, 1.0, 0.0))  # the batch has its tiling baked in
        glActiveTexture(GL_TEXTURE0)
        batch.draw()

    def invalidate(self, key: Any = None) -> None:
        keys = [key] if key is not None else list(self._static)
        for k in keys:
            batch = self._static.pop(k, None)
            if batch is not None:
                batch.delete()
        for k in [key] if key is not None else list(self._glows):
            glow = self._glows.pop(k, None)
            if glow is not None and glow[0]:
                glDeleteBuffers(1, [glow[0]])

    @property
    def static_batch_count(self) -> int:
        """Number of (VBO, texture-range) draw calls used by all cached static geometry."""
        return sum(len(b.ranges) for b in self._static.values())

    # ------------------------------------------------------------------ 2D overlay
    def clip(self, x: float, y: float, w: float, h: float) -> None:
        height = getattr(self, "_overlay_height", self.height)
        glEnable(GL_SCISSOR_TEST)
        glScissor(int(x), int(height - (y + h)), max(0, int(w)), max(0, int(h)))

    def unclip(self) -> None:
        glDisable(GL_SCISSOR_TEST)

    def begin_overlay(self, width: int, height: int) -> None:
        self._overlay_height = height
        glDisable(GL_DEPTH_TEST)
        self._overlay = ortho(width, height)
        if self.fog[3] > 0.0:
            self._upload_fog(overlay=True)  # (no fog on the HUD)

    def end_overlay(self) -> None:
        self._overlay = None
        glEnable(GL_DEPTH_TEST)
        if self.fog[3] > 0.0:
            self._upload_fog()

    def _quad_data(self, x: float, y: float, w: float, h: float) -> np.ndarray:
        d = np.zeros((4, 8), dtype=FLOAT)
        d[:, :3] = ((x, y + h, 0), (x + w, y + h, 0), (x + w, y, 0), (x, y, 0))
        d[:, 6:8] = ((0, 0), (1, 0), (1, 1), (0, 1))
        return d

    def rect(self, x: float, y: float, w: float, h: float, rgba: RGBA) -> None:
        assert self._overlay is not None
        d = self._quad_data(x, y, w, h)[[0, 1, 2, 0, 2, 3]]
        self._stream_flat(d, GL_TRIANGLES, rgba, textured=False, mvp=self._overlay)

    def rect_outline(
        self, x: float, y: float, w: float, h: float, rgba: RGBA, width: float = 2.0
    ) -> None:
        assert self._overlay is not None
        d = self._quad_data(x, y, w, h)
        self._stream_flat(d, GL_LINE_LOOP, rgba, textured=False, mvp=self._overlay, width=width)

    def image(self, tex_id: int, x: float, y: float, w: float, h: float) -> None:
        assert self._overlay is not None
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        d = self._quad_data(x, y, w, h)[[0, 1, 2, 0, 2, 3]]
        self._stream_flat(d, GL_TRIANGLES, (1.0, 1.0, 1.0, 1.0), textured=True, mvp=self._overlay)

    # ------------------------------------------------------------------ resources
    def upload_texture_rgba(self, data: bytes, w: int, h: int) -> int:
        tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return tex

    def delete_texture(self, tex_id: int) -> None:
        glDeleteTextures([tex_id])
