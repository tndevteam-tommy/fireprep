"""GLSL 1.20 per-pixel lighting used for the 3D pass (compatibility profile, so the
immediate-mode geometry and matrix stack keep working unchanged)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

from OpenGL.GL import (GL_COMPILE_STATUS, GL_FRAGMENT_SHADER, GL_LINK_STATUS, GL_VERTEX_SHADER, glAttachShader,
                       glCompileShader, glCreateProgram, glCreateShader, glGetProgramInfoLog, glGetProgramiv,
                       glGetShaderInfoLog, glGetShaderiv, glGetUniformLocation, glLinkProgram, glShaderSource,
                       glUniform1f, glUniform1i, glUniform2fv, glUniform3f, glUniform3fv, glUniform1fv, glUniform4fv,
                       glUniformMatrix4fv, glUseProgram)

MAX_POINT_LIGHTS = 8
MAX_SHADE_ZONES = 16

VERTEX_SHADER = """
#version 120
varying vec3 vEyePos;
varying vec3 vNormal;
varying vec2 vTexCoord;
varying float vAlpha;
void main() {
    vec4 eye = gl_ModelViewMatrix * gl_Vertex;
    vEyePos = eye.xyz;
    vNormal = normalize(gl_NormalMatrix * gl_Normal);
    vTexCoord = gl_MultiTexCoord0.xy;
    vAlpha = gl_Color.a;
    gl_Position = gl_ProjectionMatrix * eye;
}
"""

FRAGMENT_SHADER_TEMPLATE = """
#version 120
#define MAX_LIGHTS %d
#define MAX_SHADE %d
#define KNEE 0.75
uniform sampler2D uTexture;
uniform int uUseTexture;
uniform vec3 uColor;
uniform vec3 uEmission;
uniform float uFogStrength;
uniform float uContrast;
uniform float uAlpha;
uniform int uVertexAlpha;               // 1: the vertex colour's alpha fades the surface (ground blended by what it is made of)
uniform vec3 uSunDir;       // eye space, towards the sun
uniform vec3 uSunColor;
uniform vec3 uMoonDir;
uniform vec3 uMoonColor;
uniform vec3 uAmbient;
uniform int uNumLights;
uniform vec3 uLightPos[MAX_LIGHTS];     // eye space
uniform vec3 uLightColor[MAX_LIGHTS];
uniform float uLightRange[MAX_LIGHTS];
uniform vec4 uLightBounds[MAX_LIGHTS];  // world-space (minX, minZ, maxX, maxZ) the light is confined to
uniform float uLightBounded[MAX_LIGHTS];// 1.0 = apply the bounds (an indoor light never leaks through the walls)
uniform vec4 uLightPatch[MAX_LIGHTS];   // a second rectangle a bounded light may reach: the open doorway it spills through
uniform float uLightPatched[MAX_LIGHTS];// 1.0 = the patch is set
uniform mat4 uInvView;                  // eye space -> world space
uniform int uNumShade;                  // underground: boxes the sky's light does not reach into
uniform vec4 uShadeRects[MAX_SHADE];    // world-space (minX, minZ, maxX, maxZ)
uniform vec2 uShadeY[MAX_SHADE];        // (floor, ceiling) heights of each
uniform float uShadeLevel;              // the sky light left down there (a glimmer from the way in)
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform float uFogFar;                  // distance fog: fully fogged at the draw distance (0 = off)
varying vec3 vEyePos;
varying vec3 vNormal;
varying vec2 vTexCoord;
varying float vAlpha;

void main() {
    vec3 n = normalize(vNormal);
    vec3 base = uColor;
    float alpha = 1.0;
    if (uUseTexture == 1) {
        vec4 texel = texture2D(uTexture, vTexCoord);
        base *= texel.rgb;
        alpha = texel.a;
    }
    if (uVertexAlpha == 1) {
        alpha *= vAlpha;
    }

    vec3 world = (uInvView * vec4(vEyePos, 1.0)).xyz;
    float sky = 1.0;
    for (int i = 0; i < MAX_SHADE; i++) {
        if (i >= uNumShade) break;
        vec4 r = uShadeRects[i];
        // (strictly inside: the rock faces a cave's mouth turns to the daylight lie on its outline, and are lit)
        if (world.x > r.x && world.z > r.y && world.x < r.z && world.z < r.w &&
            world.y >= uShadeY[i].x && world.y <= uShadeY[i].y) {
            sky = uShadeLevel;
        }
    }
    vec3 light = uAmbient * sky;
    light += uSunColor * max(dot(n, normalize(uSunDir)), 0.0) * sky;
    light += uMoonColor * max(dot(n, normalize(uMoonDir)), 0.0) * sky;
    for (int i = 0; i < MAX_LIGHTS; i++) {
        if (i >= uNumLights) break;
        if (uLightBounded[i] > 0.5) {
            vec4 b = uLightBounds[i];
            bool outside = world.x < b.x || world.z < b.y || world.x > b.z || world.z > b.w;
            if (outside && uLightPatched[i] > 0.5) {
                vec4 d = uLightPatch[i];
                outside = world.x < d.x || world.z < d.y || world.x > d.z || world.z > d.w;
            }
            if (outside) continue;
        }
        vec3 toLight = uLightPos[i] - vEyePos;
        float dist = length(toLight);
        float atten = clamp(1.0 - dist / uLightRange[i], 0.0, 1.0);
        atten = atten * (2.0 - atten);            // bright near the flame, soft fall-off
        float ndl = max(dot(n, toLight / max(dist, 0.001)), 0.0);
        light += uLightColor[i] * (ndl * 0.8 + 0.2) * atten;
    }
    vec3 color = base * light + uEmission * base;
    // Soft knee: linear up to KNEE, then compressed towards (never reaching) white - so a lantern,
    // a candle and a held torch together brighten a room without bleaching its walls and beds.
    vec3 over = max(color - vec3(KNEE), vec3(0.0));
    color = min(color, vec3(KNEE)) + (1.0 - KNEE) * (vec3(1.0) - exp(-over / (1.0 - KNEE)));
    color = (color - vec3(0.5)) * uContrast + vec3(0.5);

    float dist = length(vEyePos);
    float f = exp(-(uFogDensity * dist) * (uFogDensity * dist));
    if (uFogFar > 0.0) {
        // whatever is not drawn beyond the draw distance has faded into the fog by then
        f = min(f, clamp((uFogFar - dist) / (uFogFar * 0.45), 0.0, 1.0));
    }
    f = clamp(f, 0.0, 1.0);
    f = mix(1.0, f, uFogStrength);
    // under the ground the haze is no sky-coloured mist: it is the dark
    vec3 fogColor = uFogColor * sky;
    gl_FragColor = vec4(mix(fogColor, color, f), alpha * uAlpha);
}
"""


def fragment_shader(max_lights: int = MAX_POINT_LIGHTS) -> str:
    """The fragment shader's source for ``max_lights`` point lights."""
    return FRAGMENT_SHADER_TEMPLATE % (max_lights, MAX_SHADE_ZONES)


FRAGMENT_SHADER = fragment_shader()


class ShaderError(RuntimeError):
    pass


def _compile(kind, source: str):
    shader = glCreateShader(kind)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise ShaderError(glGetShaderInfoLog(shader))
    return shader


class LightingShader:
    """Wraps the lighting program and caches uniform locations. ``max_lights``: how many point
    lights it is built to add up (the graphics settings' torch lights, at least the default)."""

    def __init__(self, max_lights: int = MAX_POINT_LIGHTS):
        self.max_lights = max(1, int(max_lights))
        self.program = glCreateProgram()
        glAttachShader(self.program, _compile(GL_VERTEX_SHADER, VERTEX_SHADER))
        glAttachShader(self.program, _compile(GL_FRAGMENT_SHADER, fragment_shader(self.max_lights)))
        glLinkProgram(self.program)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            raise ShaderError(glGetProgramInfoLog(self.program))
        self._loc: Dict[str, int] = {}
        self.active = False
        self._material = (None, None, None, None, None)     # the material uniforms as last sent (``set_material``)

    def loc(self, name: str) -> int:
        if name not in self._loc:
            self._loc[name] = glGetUniformLocation(self.program, name)
        return self._loc[name]

    def use(self) -> None:
        glUseProgram(self.program)
        self.active = True
        glUniform1i(self.loc("uTexture"), 0)
        glUniform1f(self.loc("uContrast"), 1.0)
        glUniform1f(self.loc("uAlpha"), 1.0)

    def set_contrast(self, contrast: float) -> None:
        glUniform1f(self.loc("uContrast"), contrast)

    def set_alpha(self, alpha: float) -> None:
        glUniform1f(self.loc("uAlpha"), alpha)

    def stop(self) -> None:
        glUseProgram(0)
        self.active = False

    # --- per-frame ---------------------------------------------------------------
    def set_atmosphere(self, sun_dir_eye, sun_color, moon_dir_eye, moon_color, ambient, fog_color, fog_density) -> None:
        glUniform3f(self.loc("uSunDir"), *sun_dir_eye)
        glUniform3f(self.loc("uSunColor"), *sun_color)
        glUniform3f(self.loc("uMoonDir"), *moon_dir_eye)
        glUniform3f(self.loc("uMoonColor"), *moon_color)
        glUniform3f(self.loc("uAmbient"), *ambient)
        glUniform3f(self.loc("uFogColor"), *fog_color)
        glUniform1f(self.loc("uFogDensity"), fog_density)

    def set_fog_far(self, far: float) -> None:
        """Distance fog that reaches full strength at ``far`` (the object draw distance); 0 disables it."""
        glUniform1f(self.loc("uFogFar"), far)

    def set_shade_zones(self, zones: Sequence, level: float = 0.1) -> None:
        """``zones``: (min_x, min_z, max_x, max_z, floor, ceiling) boxes - caves - where only
        ``level`` of the sky's light (sun, moon and ambient) reaches: down there it is dark but
        for the fires and torches."""
        zones = list(zones)[:MAX_SHADE_ZONES]
        glUniform1i(self.loc("uNumShade"), len(zones))
        glUniform1f(self.loc("uShadeLevel"), level)
        if zones:
            glUniform4fv(self.loc("uShadeRects"), len(zones), [c for z in zones for c in z[:4]])
            glUniform2fv(self.loc("uShadeY"), len(zones), [c for z in zones for c in z[4:6]])

    def set_inverse_view(self, matrix: Sequence[float]) -> None:
        """Column-major eye->world matrix, so the fragment shader can confine indoor lights to their building."""
        glUniformMatrix4fv(self.loc("uInvView"), 1, False, list(matrix))

    def set_point_lights(self, lights: Sequence) -> None:
        """``lights``: sequence of (eye_pos(3), color(3), range[, bounds[, patch]]) where ``bounds``
        is an optional world-space (min_x, min_z, max_x, max_z) the light may not leave and
        ``patch`` an optional second rectangle (an open doorway) it may reach all the same."""
        lights = list(lights)[:self.max_lights]
        glUniform1i(self.loc("uNumLights"), len(lights))
        if lights:
            def rect(l, index):
                return l[index] if len(l) > index and l[index] is not None else None
            pos = [c for l in lights for c in l[0]]
            col = [c for l in lights for c in l[1]]
            rng = [l[2] for l in lights]
            bounds = [c for l in lights for c in (rect(l, 3) or (0.0, 0.0, 0.0, 0.0))]
            bounded = [1.0 if rect(l, 3) is not None else 0.0 for l in lights]
            patch = [c for l in lights for c in (rect(l, 4) or (0.0, 0.0, 0.0, 0.0))]
            patched = [1.0 if rect(l, 4) is not None else 0.0 for l in lights]
            glUniform3fv(self.loc("uLightPos"), len(lights), pos)
            glUniform3fv(self.loc("uLightColor"), len(lights), col)
            glUniform1fv(self.loc("uLightRange"), len(lights), rng)
            glUniform4fv(self.loc("uLightBounds"), len(lights), bounds)
            glUniform1fv(self.loc("uLightBounded"), len(lights), bounded)
            glUniform4fv(self.loc("uLightPatch"), len(lights), patch)
            glUniform1fv(self.loc("uLightPatched"), len(lights), patched)

    # --- per-object ----------------------------------------------------------------
    def set_material(self, color=(1.0, 1.0, 1.0), emission=(0.0, 0.0, 0.0), textured: bool = True,
                     fog_strength: float = 1.0, vertex_alpha: bool = False) -> None:
        """The per-object uniforms - each sent only when it differs from what the program already
        holds (a frame sets materials thousands of times, most of them to what was set before:
        uniforms keep their values in the program, whoever else draws in between)."""
        last = self._material
        color, emission = tuple(color), tuple(emission)
        if last[0] != color:
            glUniform3f(self.loc("uColor"), *color)
        if last[1] != emission:
            glUniform3f(self.loc("uEmission"), *emission)
        if last[2] != textured:
            glUniform1i(self.loc("uUseTexture"), 1 if textured else 0)
        if last[3] != fog_strength:
            glUniform1f(self.loc("uFogStrength"), fog_strength)
        if last[4] != vertex_alpha:
            glUniform1i(self.loc("uVertexAlpha"), 1 if vertex_alpha else 0)
        self._material = (color, emission, textured, fog_strength, vertex_alpha)


def try_create_lighting_shader(max_lights: int = MAX_POINT_LIGHTS) -> Optional[LightingShader]:
    try:
        return LightingShader(max(MAX_POINT_LIGHTS, max_lights))
    except Exception:   # no GLSL support: the renderer falls back to fixed-function lighting
        return None
