"""
Firestarter - GLSL shaders (OpenGL 3.3 core).

Lighting model ("stylised realism", DESIGN 1.2): a directional sun / moon with PCF shadow
maps and dusk rim light, hemispherical sky / ground ambient, a flickering inverse-square
fire light plus a small secondary light (lighter flame, nest glow, gasoline flash), wrap
lighting for foliage, exponential height fog with sun in-scattering, seasonal snow cover
and rain wetness.  Incandescence uses a blackbody ramp that starts at the Draper point
(~525 C), the temperature at which solids first visibly glow.
"""

COMMON = """
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
uniform float u_time;

uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_sky_amb;
uniform vec3 u_ground_amb;
uniform vec3 u_fire_pos;
uniform vec3 u_fire_color;
uniform float u_fire_intensity;
uniform float u_fire_range;
uniform vec3 u_light2_pos;
uniform vec3 u_light2_color;
uniform float u_light2_intensity;
uniform vec3 u_fog_color;
uniform vec3 u_fog_sun_color;
uniform float u_fog_density;
uniform float u_fog_height;
uniform float u_snow;
uniform float u_wetness;
uniform vec3 u_wind;
uniform mat4 u_light_vp;
uniform sampler2DShadow u_shadow_map;
uniform int u_shadows;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float vnoise3(vec3 p) {
    vec3 i = floor(p); vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float a = hash13(i), b = hash13(i + vec3(1,0,0)), c = hash13(i + vec3(0,1,0)), d = hash13(i + vec3(1,1,0));
    float e = hash13(i + vec3(0,0,1)), f1 = hash13(i + vec3(1,0,1)), g = hash13(i + vec3(0,1,1)), h = hash13(i + vec3(1,1,1));
    return mix(mix(mix(a,b,u.x), mix(c,d,u.x), u.y), mix(mix(e,f1,u.x), mix(g,h,u.x), u.y), u.z);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(17.1, 9.2); a *= 0.5; }
    return s;
}
float fbm3(vec3 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { s += a * vnoise3(p); p = p * 2.07 + vec3(5.1, 13.7, 2.3); a *= 0.5; }
    return s;
}
// rotated-octave fbm: no axis-aligned "blocky" artefacts
float rfbm(vec2 p) {
    const mat2 R = mat2(0.8, -0.6, 0.6, 0.8);
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { s += a * vnoise(p); p = R * p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return s / 0.97;
}
float rnoise(vec2 p) {
    const mat2 R = mat2(0.8, -0.6, 0.6, 0.8);
    return 0.5 * (vnoise(p) + vnoise(R * p * 1.13 + 3.7));
}
// cellular (Worley) noise: distance to the nearest feature point, and the cell id hash
vec2 cellular(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    float d1 = 8.0; float id = 0.0;
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec2 g = vec2(x, y);
        vec2 o = vec2(hash12(i + g), hash12(i + g + 17.3));
        float d = length(g + o - f);
        if (d < d1) { d1 = d; id = hash12(i + g + 5.1); }
    }
    return vec2(d1, id);
}
// bump mapping without tangents (Mikkelsen 2010): perturb N by the screen-space gradient of h
vec3 bump_normal(vec3 N, vec3 P, float h, float strength) {
    vec3 dpdx = dFdx(P), dpdy = dFdy(P);
    float dhx = dFdx(h) * strength, dhy = dFdy(h) * strength;
    vec3 r1 = cross(dpdy, N), r2 = cross(N, dpdx);
    float det = dot(dpdx, r1);
    vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
    return normalize(abs(det) * N - grad);
}
// Incandescent colour * radiance for a surface at t_c (Celsius).  Zero below ~480 C.
vec3 blackbody(float t_c) {
    float k = clamp((t_c - 480.0) / 720.0, 0.0, 1.0);
    vec3 dull = vec3(0.35, 0.02, 0.0);
    vec3 cherry = vec3(1.0, 0.12, 0.01);
    vec3 orange = vec3(1.0, 0.42, 0.06);
    vec3 yellow = vec3(1.0, 0.78, 0.38);
    vec3 c = k < 0.33 ? mix(dull, cherry, k / 0.33) : (k < 0.66 ? mix(cherry, orange, (k - 0.33) / 0.33) : mix(orange, yellow, (k - 0.66) / 0.34));
    float radiance = pow(max(0.0, (t_c + 273.15) / 1100.0), 4.0) * smoothstep(480.0, 620.0, t_c);
    return c * radiance * 3.2;
}
float shadow_factor(vec3 P, vec3 N) {
    if (u_shadows == 0) return 1.0;
    vec4 lp = u_light_vp * vec4(P + N * 0.05, 1.0);
    vec3 sc = lp.xyz / lp.w * 0.5 + 0.5;
    if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0) return 1.0;
    float bias = 0.0012;
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    // rotated Poisson-disk PCF: soft penumbrae without blocky texel steps
    const vec2 disk[12] = vec2[](vec2(-0.326, -0.406), vec2(-0.840, -0.074), vec2(-0.696, 0.457), vec2(-0.203, 0.621),
                                 vec2(0.962, -0.195), vec2(0.473, -0.480), vec2(0.519, 0.767), vec2(0.185, -0.893),
                                 vec2(0.507, 0.064), vec2(0.896, 0.412), vec2(-0.322, -0.933), vec2(-0.792, -0.598));
    float ang = hash12(gl_FragCoord.xy) * 6.2831853;
    mat2 rot = mat2(cos(ang), -sin(ang), sin(ang), cos(ang));
    float s = 0.0;
    for (int i = 0; i < 12; i++)
        s += texture(u_shadow_map, vec3(sc.xy + rot * disk[i] * texel * 2.4, sc.z - bias));
    return s / 12.0;
}
vec3 fire_light(vec3 P, vec3 N, float wrap) {
    vec3 Lv = u_fire_pos - P;
    float d = length(Lv);
    vec3 L = Lv / max(d, 1e-3);
    float ndl = clamp((dot(N, L) + wrap) / (1.0 + wrap), 0.0, 1.0);
    float att = u_fire_intensity / (1.0 + 0.9 * d * d) * smoothstep(u_fire_range, u_fire_range * 0.35, d);
    vec3 c = u_fire_color * ndl * att;
    vec3 Lv2 = u_light2_pos - P;
    float d2 = length(Lv2);
    float ndl2 = clamp((dot(N, Lv2 / max(d2, 1e-3)) + wrap) / (1.0 + wrap), 0.0, 1.0);
    c += u_light2_color * ndl2 * u_light2_intensity / (1.0 + 12.0 * d2 * d2);
    return c;
}
vec3 apply_fog(vec3 col, vec3 P) {
    vec3 V = P - u_cam_pos;
    float dist = length(V);
    vec3 dir = V / max(dist, 1e-3);
    float h = max(0.0, P.y + 2.0);
    float density = u_fog_density * exp(-h * u_fog_height);
    float f = 1.0 - exp(-dist * density);
    float sun_amt = pow(max(dot(dir, u_sun_dir), 0.0), 6.0);
    vec3 fogc = mix(u_fog_color, u_fog_sun_color, sun_amt);
    return mix(col, fogc, clamp(f, 0.0, 1.0));
}
vec3 shade(vec3 albedo, vec3 N, vec3 P, float rough, float spec, float wrap, float ao) {
    vec3 V = normalize(u_cam_pos - P);
    vec3 L = normalize(u_sun_dir);
    float sh = shadow_factor(P, N);
    float ndl = clamp((dot(N, L) + wrap) / (1.0 + wrap), 0.0, 1.0);
    vec3 sun = u_sun_color * ndl * sh;
    // dusk rim light: low sun grazing the silhouettes
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * max(dot(-V, L), 0.0) * (1.0 - smoothstep(0.0, 0.5, L.y));
    vec3 amb = mix(u_ground_amb, u_sky_amb, N.y * 0.5 + 0.5) * ao;
    vec3 H = normalize(L + V);
    float sp = pow(max(dot(N, H), 0.0), mix(8.0, 96.0, 1.0 - rough)) * spec * sh;
    vec3 fire = fire_light(P, N, wrap * 0.5 + 0.2);
    return albedo * (sun + amb + fire) + u_sun_color * sp + u_sun_color * rim * 0.25 * sh * albedo * 2.0;
}
"""

# ------------------------------------------------------------------------------------ sky
SKY_VS = """
#version 330 core
out vec2 v_ndc;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    v_ndc = p;
    gl_Position = vec4(p, 0.9999, 1.0);
}
"""

SKY_FS = """
#version 330 core
in vec2 v_ndc;
out vec4 frag;
uniform mat4 u_inv_vp;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform vec3 u_ground_col;
uniform vec3 u_sun_dir2;
uniform vec3 u_moon_dir;
uniform vec3 u_sun_disc_col;
uniform float u_cloud_cover;
uniform float u_night;
uniform float u_time2;
uniform vec2 u_cloud_offset;
uniform vec3 u_cloud_lit;
uniform vec3 u_cloud_shadow;
uniform float u_moon_intensity;
""" + COMMON.replace("uniform float u_time;", "") + """
void main() {
    vec4 w = u_inv_vp * vec4(v_ndc, 1.0, 1.0);
    vec3 dir = normalize(w.xyz / w.w - u_cam_pos);
    float y = dir.y;
    float hz = pow(1.0 - max(y, 0.0), 4.0);
    vec3 col = mix(u_zenith, u_horizon, hz);
    if (y < 0.0) col = mix(u_horizon, u_ground_col, clamp(-y * 6.0, 0.0, 1.0));
    // sun disc & Mie glow
    float sd = max(dot(dir, u_sun_dir2), 0.0);
    col += u_sun_disc_col * (pow(sd, 6.0) * 0.35 + pow(sd, 64.0) * 0.9);
    col += u_sun_disc_col * smoothstep(0.9996, 0.99985, sd) * 30.0;
    // stars
    if (u_night > 0.01 && y > 0.0) {
        // star field: each 3D cell may hold one star; the angular distance to the star's
        // direction is measured exactly, so stars are round points wherever the sphere cuts
        float stars = 0.0;
        vec3 starc = vec3(0.0);
        for (int layer = 0; layer < 2; layer++) {
            float scale = layer == 0 ? 160.0 : 330.0;
            vec3 sp = dir * scale;
            vec3 cell = floor(sp);
            for (int k = 0; k < 8; k++) {
                vec3 c = cell + vec3(float(k & 1), float((k >> 1) & 1), float((k >> 2) & 1)) - 0.5 * vec3(1.0);
                float h = hash13(c + float(layer) * 31.7);
                if (h < (layer == 0 ? 0.992 : 0.986)) continue;
                vec3 sd = normalize(c + 0.5 + (vec3(hash13(c + 1.3), hash13(c + 2.9), hash13(c + 4.1)) - 0.5) * 0.8);
                float ang = length(cross(dir, sd)) * scale;
                float mag = pow(hash13(c + 7.7), 8.0) * (layer == 0 ? 3.0 : 1.0) + 0.12;
                float tw = 0.75 + 0.25 * sin(u_time2 * (3.0 + h * 7.0) + h * 60.0);
                float st = exp(-ang * ang * 9.0) * mag * tw;
                stars += st;
                starc += st * mix(vec3(1.0, 0.82, 0.62), vec3(0.75, 0.85, 1.0), hash13(c + 9.9));
            }
        }
        col += starc * u_night * (1.0 - u_cloud_cover) * 2.4 * smoothstep(0.0, 0.12, y);
        // faint milky way band
        float band = exp(-pow(dot(dir, normalize(vec3(0.35, 0.6, -0.72))) * 4.0, 2.0));
        col += vec3(0.16, 0.18, 0.27) * band * smoothstep(0.35, 0.8, fbm(dir.xz * 7.0 + dir.y * 3.0)) * u_night * 0.35 * (1.0 - u_cloud_cover);
    }
    // moon
    float md = max(dot(dir, u_moon_dir), 0.0);
    col += vec3(0.8, 0.85, 1.0) * (smoothstep(0.99955, 0.99975, md) * 2.6 + pow(md, 180.0) * 0.18) * u_moon_intensity;
    // clouds (painterly fbm deck)
    if (y > 0.0) {
        vec2 uv = dir.xz / (y + 0.08) * 0.9 + u_cloud_offset;
        float n = fbm(uv * 0.55) * 0.65 + fbm(uv * 1.7 + 3.1) * 0.35;
        float cov = mix(0.66, 0.30, u_cloud_cover);
        float c = smoothstep(cov, cov + 0.22, n) * smoothstep(0.0, 0.2, y);
        float lit = clamp(0.55 + 0.45 * dot(normalize(vec3(u_sun_dir2.x, 0.3, u_sun_dir2.z)), normalize(vec3(dir.x, 0.2, dir.z))), 0.0, 1.0);
        vec3 cc = mix(u_cloud_shadow, u_cloud_lit, lit * (1.0 - 0.5 * n));
        cc += u_sun_disc_col * pow(sd, 12.0) * 0.6 * (1.0 - c);
        col = mix(col, cc, c * 0.92);
    }
    frag = vec4(col, 1.0);
}
"""

# ------------------------------------------------------------------------------------ terrain
TERRAIN_VS = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec4 in_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_pos;
out vec3 v_nrm;
out vec4 v_col;
void main() {
    v_pos = in_position;
    v_nrm = in_normal;
    v_col = in_color;
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
}
"""

TERRAIN_FS = """
#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec4 v_col;
out vec4 frag;
uniform vec3 u_grass_col;
uniform vec3 u_grass_dry;
uniform vec3 u_pit_center;
uniform float u_ash_amount;
""" + COMMON + """
void main() {
    vec3 N = normalize(v_nrm);
    vec2 xz = v_pos.xz;
    float dist = length(v_pos - u_cam_pos);
    float detail = 1.0 - smoothstep(18.0, 60.0, dist);      // fade micro detail with distance
    vec2 warp = vec2(rfbm(xz * 0.21), rfbm(xz * 0.21 + 11.0)) - 0.5;
    float n1 = rfbm(xz * 0.09 + warp * 0.8);                 // large meadows / dry patches
    float n2 = rfbm(xz * 0.55 + warp * 2.0);                 // clumps, dirt patches
    float nf = rnoise(xz * 9.0) * 0.55 + rnoise(xz * 27.0) * 0.45;   // blades & litter texture
    float nm = rnoise(xz * 3.1 + 4.0);
    float slope = 1.0 - N.y;
    // --- materials
    vec3 grass = mix(u_grass_col, u_grass_dry, smoothstep(0.38, 0.72, n1));
    grass *= mix(vec3(0.92, 1.0, 1.06), vec3(1.08, 1.0, 0.88), nm);         // hue jitter: blue-green <-> yellow-green
    grass *= mix(1.0, 0.62 + 0.55 * nf, detail);                             // dark gaps between blades
    vec3 litter = mix(vec3(0.085, 0.064, 0.045), vec3(0.17, 0.12, 0.075), n2);   // leaf litter & humus
    vec2 cl = cellular(xz * 11.0 + warp * 3.0);
    float leafy = smoothstep(0.42, 0.15, cl.x + (rnoise(xz * 23.0) - 0.5) * 0.35) * step(0.35, cl.y);
    litter = mix(litter, mix(vec3(0.30, 0.18, 0.08), vec3(0.20, 0.16, 0.09), cl.y), leafy * 0.3 * detail);
    litter *= mix(1.0, 0.75 + 0.4 * nf, detail);
    vec3 needles = vec3(0.33, 0.21, 0.11) * mix(1.0, 0.75 + 0.45 * nf, detail);
    vec3 dirt = mix(vec3(0.27, 0.22, 0.16), vec3(0.36, 0.29, 0.21), n2) * mix(1.0, 0.8 + 0.3 * nf, detail);
    vec2 peb = cellular(xz * 11.0);
    dirt = mix(dirt, vec3(0.25, 0.23, 0.2) * (0.7 + 0.4 * peb.y), smoothstep(0.16, 0.08, peb.x) * step(0.82, peb.y) * detail * 0.7);
    vec3 rock = mix(vec3(0.33, 0.32, 0.30), vec3(0.46, 0.44, 0.40), nf) * (0.8 + 0.3 * n1);
    vec3 mud = vec3(0.16, 0.13, 0.10);
    // mask: v_col.r = forest density, v_col.g = clearing, v_col.b = creek wetness, v_col.a = ridge dryness
    float forest = smoothstep(0.3, 0.7, v_col.r + (n1 - 0.5) * 0.6);
    // moss and low herbs carpet the damper hollows of the forest floor
    float moss = smoothstep(0.48, 0.7, rfbm(xz * 0.33 + 7.0) + v_col.b * 0.35) * (1.0 - v_col.a);
    litter = mix(litter, u_grass_col * vec3(0.75, 0.85, 0.7) * mix(1.0, 0.7 + 0.5 * nf, detail), moss * 0.8);
    vec3 col = mix(grass, litter, forest);
    col = mix(col, needles, smoothstep(0.55, 0.9, v_col.r) * smoothstep(0.45, 0.7, n2) * 0.6);
    // trampled camp clearing: worn paths and bare earth between the grass
    float bare = smoothstep(0.62, 0.74, n2 + 0.12 * nm) * v_col.g;
    col = mix(col, dirt, bare);
    col = mix(col, mud, v_col.b * 0.8);
    col = mix(col, mix(col, vec3(0.55, 0.47, 0.30), 0.5), v_col.a * 0.6);
    float rocky = smoothstep(0.28, 0.5, slope + (n2 - 0.5) * 0.2);
    col = mix(col, rock, rocky);
    // --- the fire pit: trampled earth, soot and ash
    float pr = length(v_pos.xz - u_pit_center.xz);
    col = mix(col, dirt * 0.8, smoothstep(1.7, 0.85, pr + (n2 - 0.5) * 0.4) * 0.75);
    float pitm = smoothstep(0.62, 0.46, pr);
    vec3 scorched = mix(vec3(0.15, 0.115, 0.085), vec3(0.06, 0.052, 0.047), smoothstep(0.62, 0.25, pr + (nf - 0.5) * 0.12));
    vec2 ch = cellular(xz * 30.0);
    float chips = smoothstep(0.35, 0.15, ch.x) * step(0.45, ch.y);
    scorched = mix(scorched, vec3(0.018, 0.016, 0.015), chips * 0.85);
    col = mix(col, scorched, pitm);
    float ash = smoothstep(0.44, 0.08, pr) * u_ash_amount * smoothstep(0.3, 0.7, rfbm(xz * 9.0) + 0.25);
    col = mix(col, vec3(0.55, 0.53, 0.50) * (0.78 + 0.3 * rnoise(xz * 40.0)), ash);
    // --- wetness darkens and adds sheen
    float wet = clamp(u_wetness + v_col.b * 0.5, 0.0, 1.0);
    col *= 1.0 - 0.35 * wet;
    // --- snow cover on flat ground
    float snow = u_snow * smoothstep(0.55, 0.85, N.y) * smoothstep(0.25, 0.55, n2 * 0.6 + u_snow * 0.7);
    snow *= smoothstep(0.55, 0.75, pr + 0.1 * u_snow * 4.0);
    col = mix(col, vec3(0.86, 0.89, 0.95), clamp(snow, 0.0, 1.0));
    // --- micro relief: clumps, pebbles and ruts catch the low sun
    float h = (n2 * 0.6 + nf * 0.25 * detail) * (1.0 - snow * 0.7) - bare * 0.15 + rocky * nf * 0.4;
    vec3 Nb = bump_normal(N, v_pos, h, 0.06 + 0.1 * rocky);
    float ao = mix(1.0, 0.8 + 0.25 * nf, detail * (1.0 - bare));
    // under a closed canopy little direct sun reaches the floor beyond the shadow-map range
    float canopy = smoothstep(0.45, 0.9, v_col.r) * smoothstep(25.0, 45.0, dist);
    col *= 1.0 - 0.35 * canopy;
    float rough = mix(0.9, 0.35, wet);
    float spec = mix(0.03, 0.35, wet) + snow * 0.15;
    vec3 lit = shade(col, Nb, v_pos, rough, spec, 0.1, ao);
    frag = vec4(apply_fog(lit, v_pos), 1.0);
}
"""

# ------------------------------------------------------------------------------------ generic objects
OBJECT_VS = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec4 in_color;
#ifdef INSTANCED
layout (location = 4) in vec4 in_i0;   // xyz position, w yaw (radians)
layout (location = 5) in vec4 in_i1;   // xyz scale, w seed
layout (location = 6) in vec4 in_i2;   // rgb tint, w param (fall angle / misc)
#endif
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;
uniform float u_time;
uniform vec3 u_wind;
uniform float u_sway;
out vec3 v_pos;
out vec3 v_nrm;
out vec2 v_uv;
out vec4 v_col;
out vec3 v_tint;
out float v_seed;
out vec3 v_local;
void main() {
    vec3 p = in_position;
    vec3 n = in_normal;
    vec3 tint = vec3(1.0);
    float seed = 0.0;
    mat4 model = u_model;
#ifdef INSTANCED
    float c = cos(in_i0.w), s = sin(in_i0.w);
    mat3 R = mat3(c, 0.0, -s,  0.0, 1.0, 0.0,  s, 0.0, c);
    p = R * (p * in_i1.xyz);
    n = normalize(R * (n / in_i1.xyz));
    p += in_i0.xyz;
    tint = in_i2.rgb;
    seed = in_i1.w;
    model = mat4(1.0);
#endif
    vec4 wp = model * vec4(p, 1.0);
    vec3 wn = normalize(mat3(model) * n);
    // wind sway weighted by vertex colour alpha (0 = rigid trunk base, 1 = branch tips)
    float sway = (1.0 - in_color.a) * u_sway;
    if (sway > 0.0) {
        float h = max(0.0, in_position.y);
        float ph = u_time * 1.7 + seed * 6.28 + wp.x * 0.15 + wp.z * 0.11;
        vec2 w2 = u_wind.xz * 0.035 + vec2(0.02);
        vec2 off = w2 * (sin(ph) * 0.6 + sin(ph * 2.7) * 0.25 + 0.5) * h * h * 0.03 * sway;
        off += vec2(sin(ph * 4.1 + h), cos(ph * 3.3 + h)) * 0.015 * sway * h * 0.1;
        wp.xz += off;
    }
    v_pos = wp.xyz;
    v_nrm = wn;
    v_uv = in_uv;
    v_col = in_color;
    v_tint = tint;
    v_seed = seed;
    v_local = in_position;
    gl_Position = u_proj * u_view * wp;
}
"""

OBJECT_FS = """
#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_col;
in vec3 v_tint;
in float v_seed;
in vec3 v_local;
out vec4 frag;
uniform int u_material;   // 0 generic, 1 bark, 2 foliage, 3 rock, 4 skin, 5 cloth, 6 metal, 7 tarp, 8 birch bark, 9 fungus, 10 emissive, 11 grass, 12 leather, 13 leaf card
uniform vec4 u_color;
uniform float u_emissive;
uniform vec3 u_emit_color;
uniform float u_snow_cap;
uniform float u_highlight;
uniform sampler2D u_leaf_tex;
""" + COMMON + """
void main() {
    vec3 N = normalize(v_nrm);
    vec3 base = v_col.rgb * u_color.rgb * v_tint;
    if (u_material == 13) {
        // alpha-cut leaf / needle spray card; the normal is the canopy's (not the card's), so
        // it is never flipped - the whole crown shades like a soft volume
        vec4 lt = texture(u_leaf_tex, v_uv);
        float a = (lt.a - 0.42) / max(fwidth(lt.a), 1e-4) + 0.5;
        if (a < 0.5) discard;
        base *= lt.rgb * 1.45;
        vec3 V = normalize(u_cam_pos - v_pos);
        vec3 L = normalize(u_sun_dir);
        float sh = shadow_factor(v_pos, N);
        // light filtering through the leaves when looking toward the sun
        float trans = pow(max(dot(-V, L), 0.0), 4.0) * 0.9 + 0.12;
        // snow only settles on the upper faces of the sprays; the needles stay green beneath
        float snowl = u_snow * u_snow_cap * smoothstep(0.45, 0.9, N.y) * smoothstep(0.35, 0.65, lt.r);
        base = mix(base, vec3(0.82, 0.85, 0.9), snowl * 0.6);
        base *= 1.0 - 0.25 * u_wetness * (1.0 - snowl);
        vec3 col = shade(base, N, v_pos, 0.8, 0.04, 0.5, 1.0);
        col += base * vec3(1.0, 1.05, 0.7) * u_sun_color * trans * sh * (1.0 - snowl) * 0.55;
        col += vec3(1.0, 0.85, 0.55) * u_highlight * 0.2;
        frag = vec4(apply_fog(col, v_pos), 1.0);
        return;
    }
    if (!gl_FrontFacing) N = -N;
    float rough = 0.85, spec = 0.05, wrap = 0.1, ao = 1.0;
    if (u_material == 1) {            // bark: vertical furrows
        float f = vnoise(vec2(atan(v_local.x, v_local.z) * 6.0, v_local.y * 1.4 + v_seed * 10.0));
        float f2 = vnoise(vec2(atan(v_local.x, v_local.z) * 18.0, v_local.y * 5.0));
        base *= 0.65 + 0.45 * f + 0.15 * f2;
        rough = 0.95;
    } else if (u_material == 8) {     // paper birch: white bark with dark lenticels
        float band = step(0.82, vnoise(vec2(atan(v_local.x, v_local.z) * 3.0, v_local.y * 9.0 + v_seed * 7.0)));
        float patch = smoothstep(0.62, 0.7, vnoise(vec2(atan(v_local.x, v_local.z) * 1.5, v_local.y * 1.2)));
        base = mix(base, vec3(0.08, 0.07, 0.06), max(band * 0.85, patch * 0.7));
        rough = 0.8;
    } else if (u_material == 2) {     // foliage clumps: painterly light/dark variation
        float n = vnoise(v_pos.xz * 1.3 + v_pos.y * 0.7);
        base *= 0.72 + 0.5 * n;
        wrap = 0.55;
        spec = 0.02;
    } else if (u_material == 3) {     // rock with lichen
        float n = fbm(v_pos.xz * 3.0 + v_pos.y);
        base *= 0.75 + 0.45 * n;
        base = mix(base, vec3(0.45, 0.48, 0.30), smoothstep(0.62, 0.75, n) * 0.5 * N.y);
        rough = 0.92;
    } else if (u_material == 4) {     // skin
        rough = 0.6; spec = 0.08; wrap = 0.35;
    } else if (u_material == 12) {    // leather work glove: fine grain, scuffed lighter patches
        float g = vnoise(v_local.xz * 900.0 + v_local.y * 700.0);
        float wear = vnoise(v_local.xz * 70.0 + v_local.y * 40.0);
        base *= 0.86 + 0.2 * g;
        base = mix(base, base * 1.3 + vec3(0.02), smoothstep(0.62, 0.9, wear) * 0.6);
        rough = 0.6; spec = 0.1; wrap = 0.3;
    } else if (u_material == 5) {     // waxed canvas: fine weave and soft creases
        float weave = vnoise(v_uv * vec2(160.0, 90.0));
        float crease = vnoise(vec2(v_uv.x * 5.0, v_uv.y * 14.0));
        base *= 0.85 + 0.12 * weave - 0.18 * smoothstep(0.55, 0.9, crease);
        rough = 0.8; spec = 0.08; wrap = 0.25;
    } else if (u_material == 6) {     // metal
        rough = 0.25; spec = 0.9;
    } else if (u_material == 7) {     // polyethylene tarp
        float wrin = vnoise(v_uv * vec2(18.0, 7.0));
        base *= 0.8 + 0.25 * wrin;
        rough = 0.45; spec = 0.25 + 0.4 * u_wetness; wrap = 0.4;
    } else if (u_material == 11) {    // grass blades: root-to-tip gradient with back-lit translucency
        float tip = 1.0 - v_col.a;
        base = u_color.rgb * v_tint * mix(0.45, 1.15, tip);
        vec3 L2 = normalize(u_sun_dir);
        vec3 V2 = normalize(u_cam_pos - v_pos);
        float back = pow(max(dot(-V2, L2), 0.0), 3.0) * tip;
        vec3 col2 = shade(base, normalize(vec3(N.x * 0.3, 1.0, N.z * 0.3)), v_pos, 0.9, 0.02, 0.6, 1.0);
        col2 += base * u_sun_color * back * 0.8 * shadow_factor(v_pos, vec3(0, 1, 0));
        frag = vec4(apply_fog(col2, v_pos), 1.0);
        return;
    } else if (u_material == 9) {     // bracket fungus: concentric bands
        float r = length(v_local.xz) * 40.0;
        base *= 0.75 + 0.3 * sin(r);
        rough = 0.8;
    }
    // snow sitting on upward faces
    float snow = u_snow * u_snow_cap * smoothstep(0.45, 0.85, N.y);
    base = mix(base, vec3(0.86, 0.89, 0.95), snow);
    base *= 1.0 - 0.25 * u_wetness * (1.0 - snow);
    vec3 col = shade(base, N, v_pos, rough, spec, wrap, ao);
    col += u_emit_color * u_emissive;
    col += vec3(1.0, 0.85, 0.55) * u_highlight * (0.25 + 0.25 * sin(u_time * 5.0));
    frag = vec4(apply_fog(col, v_pos), u_color.a);
}
"""

# ------------------------------------------------------------------------------------ fuel (burning wood)
FUEL_FS = """
#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_col;
in vec3 v_tint;
in float v_seed;
in vec3 v_local;
out vec4 frag;
uniform vec3 u_bark_col;
uniform vec3 u_wood_col;
uniform float u_seg_T[3];
uniform float u_seg_char[3];
uniform float u_seg_phi[3];
uniform float u_seg_ash[3];
uniform float u_seg_wet[3];
uniform float u_tinder;
uniform float u_highlight;
uniform float u_bark_style;    // 0 generic, 1 birch, 2 pine, 3 oak, 4 cedar
uniform mat4 u_model;
""" + COMMON + """
float seg_lerp(float a[3], float v) {
    float x = clamp(v * 3.0 - 0.5, 0.0, 2.0);
    int i = int(floor(x));
    float f = fract(x);
    if (i >= 2) return a[2];
    return mix(a[i], a[i + 1], f);
}
float voronoi_edge(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec2 g = vec2(x, y);
        vec2 o = vec2(hash12(i + g), hash12(i + g + 17.3));
        float d = length(g + o - f);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
    return d2 - d1;
}
void main() {
    vec3 N = normalize(v_nrm);
    float v = v_uv.y;                         // 0..1 along the piece
    float T = seg_lerp(u_seg_T, v);
    float chr = seg_lerp(u_seg_char, v);
    float phi = seg_lerp(u_seg_phi, v);
    float ash = seg_lerp(u_seg_ash, v);
    float wet = seg_lerp(u_seg_wet, v);
    bool cap = v_col.a < 0.5;                 // end-grain caps
    float ang = atan(v_local.x, v_local.z);
    vec3 bark = u_bark_col * (0.7 + 0.4 * vnoise(vec2(ang * 5.0, v_local.y * 18.0 + v_seed * 9.0)));
    // species bark in real units: metres along the piece (y) and around it (c)
    float sy = length(u_model[1].xyz);
    float sx = length(u_model[0].xyz);
    float ym = v_local.y * sy + v_seed * 3.0;
    float cm = ang * max(0.02, length(v_local.xz) * sx);
    int style = int(u_bark_style + 0.5);
    if (style == 1) {            // paper birch: chalky white, dark horizontal lenticels, black scars
        float len = vnoise(vec2(cm * 7.0, ym * 75.0));
        float dash = smoothstep(0.74, 0.84, len) * smoothstep(0.3, 0.6, vnoise(vec2(cm * 2.0, ym * 9.0)));
        float scar = smoothstep(0.78, 0.86, vnoise(vec2(cm * 2.4, ym * 2.2) + 7.0));
        vec3 paper = u_bark_col * (0.9 + 0.12 * vnoise(vec2(cm * 3.0, ym * 6.0))) * vec3(1.0, 0.985, 0.95);
        bark = mix(paper, vec3(0.09, 0.08, 0.07), max(dash * 0.85, scar * 0.9));
    } else if (style == 2) {     // pine: red-brown plates split by dark fissures
        float e = voronoi_edge(vec2(cm * 9.0, ym * 3.5));
        float fiss = 1.0 - smoothstep(0.03, 0.16, e);
        bark = u_bark_col * vec3(1.12, 0.95, 0.85) * (0.85 + 0.3 * vnoise(vec2(cm * 5.0, ym * 5.0)));
        bark = mix(bark, u_bark_col * 0.35, fiss);
    } else if (style == 3) {     // oak: deep vertical ridges and furrows, grey-brown
        float r = vnoise(vec2(cm * 16.0, ym * 1.6)) * 0.7 + vnoise(vec2(cm * 40.0, ym * 4.0)) * 0.3;
        float ridge = smoothstep(0.35, 0.7, r);
        bark = u_bark_col * mix(0.45, 1.15, ridge) * (0.9 + 0.15 * vnoise(vec2(cm * 3.0, ym * 8.0)));
    } else if (style == 4) {     // cedar: long fibrous strips
        float f = vnoise(vec2(cm * 30.0, ym * 0.9)) * 0.6 + vnoise(vec2(cm * 70.0, ym * 2.5)) * 0.4;
        bark = u_bark_col * (0.6 + 0.6 * f);
    }
    vec3 base = bark;
    if (cap) {
        float r = length(v_local.xz) / max(1e-4, max(abs(v_local.x), 1e-4) + 0.0001);
        float rings = sin(length(v_local.xz) * 420.0 + vnoise(v_local.xz * 60.0) * 3.0);
        base = u_wood_col * (0.82 + 0.14 * rings);
    }
    if (u_tinder > 0.5) {
        float f = vnoise(v_local.xz * 90.0 + v_local.y * 40.0);
        base = u_bark_col * (0.6 + 0.6 * f);
    }
    // char: black, cracked "alligator" pattern with glowing fissures
    vec2 cuv = vec2(ang * 2.2, v_local.y * 14.0) + v_seed * 3.0;
    if (cap) cuv = v_local.xz * 40.0;
    float edge = voronoi_edge(cuv * (1.0 + 0.6 * chr));
    float crack = 1.0 - smoothstep(0.02, 0.12, edge);
    vec3 charcol = vec3(0.035, 0.03, 0.028) * (0.8 + 0.6 * vnoise(cuv * 3.0));
    base = mix(base, charcol, smoothstep(0.05, 0.55, chr));
    // wet wood is darker
    base *= 1.0 - 0.35 * clamp(wet * 1.2, 0.0, 1.0);
    // ash blanket: pale grey-white flecks
    float ashmask = smoothstep(0.1, 0.9, ash) * smoothstep(0.35, 0.6, vnoise(cuv * 1.7 + 5.0));
    base = mix(base, vec3(0.62, 0.60, 0.57), ashmask);
    vec3 col = shade(base, N, v_pos, 0.9, 0.04 + 0.3 * wet, 0.15, 1.0);
    // incandescence: fissures glow first, whole surface glows when very hot
    float flick = 0.85 + 0.15 * sin(u_time * 7.0 + v_seed * 11.0 + v * 9.0) * sin(u_time * 3.1 + v * 4.0);
    vec3 glow = blackbody(T) * (crack * 1.6 + 0.35 * smoothstep(650.0, 950.0, T)) * smoothstep(0.2, 0.6, chr) * (1.0 - 0.8 * ashmask);
    col += glow * flick;
    // under-flame orange bounce on flaming parts
    col += vec3(1.0, 0.45, 0.12) * phi * 0.25;
    col += vec3(1.0, 0.85, 0.55) * u_highlight * (0.2 + 0.15 * sin(u_time * 5.0));
    frag = vec4(apply_fog(col, v_pos), 1.0);
}
"""

# ------------------------------------------------------------------------------------ water
WATER_FS = """
#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_col;
in vec3 v_tint;
in float v_seed;
in vec3 v_local;
out vec4 frag;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform float u_rain;
""" + COMMON + """
void main() {
    vec2 flow = vec2(u_time * 0.35, 0.0);
    float h1 = fbm(v_pos.xz * 1.3 + flow);
    float h2 = fbm(v_pos.xz * 3.1 - flow * 1.7);
    vec3 N = normalize(vec3((h1 - 0.5) * 0.35 + (h2 - 0.5) * 0.2, 1.0, (h2 - 0.5) * 0.35));
    if (u_rain > 0.05) {
        vec2 cell = floor(v_pos.xz * 3.0);
        float ph = fract(u_time * 1.3 + hash12(cell));
        float r = length(fract(v_pos.xz * 3.0) - 0.5);
        N.xz += (sin((r - ph * 0.5) * 60.0) * (1.0 - ph) * smoothstep(0.5, 0.0, r)) * 0.25 * u_rain;
        N = normalize(N);
    }
    vec3 V = normalize(u_cam_pos - v_pos);
    vec3 R = reflect(-V, N);
    float fres = 0.03 + 0.97 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    vec3 sky = mix(u_horizon, u_zenith, clamp(R.y * 1.5, 0.0, 1.0));
    vec3 deep = vec3(0.03, 0.07, 0.07);
    vec3 col = mix(deep * (u_sky_amb * 2.0 + 0.2), sky, fres);
    vec3 L = normalize(u_sun_dir);
    col += u_sun_color * pow(max(dot(R, L), 0.0), 200.0) * 2.0;
    col += fire_light(v_pos, N, 0.0) * 0.3;
    frag = vec4(apply_fog(col, v_pos), 0.88);
}
"""

# ------------------------------------------------------------------------------------ shadow
SHADOW_VS = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in vec4 in_color;
out vec2 v_uv;
#ifdef INSTANCED
layout (location = 4) in vec4 in_i0;
layout (location = 5) in vec4 in_i1;
#endif
uniform mat4 u_light_vp;
uniform mat4 u_model;
void main() {
    vec3 p = in_position;
    mat4 model = u_model;
#ifdef INSTANCED
    float c = cos(in_i0.w), s = sin(in_i0.w);
    mat3 R = mat3(c, 0.0, -s,  0.0, 1.0, 0.0,  s, 0.0, c);
    p = R * (p * in_i1.xyz) + in_i0.xyz;
    model = mat4(1.0);
#endif
    v_uv = in_uv;
    gl_Position = u_light_vp * model * vec4(p, 1.0);
}
"""

SHADOW_FS = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_leaf_tex;
uniform int u_alpha_test;
void main() {
    if (u_alpha_test == 1 && texture(u_leaf_tex, v_uv).a < 0.42) discard;
}
"""

# ------------------------------------------------------------------------------------ flames
FLAME_VS = """
#version 330 core
layout (location = 0) in vec2 in_corner;      // (-1..1, 0..1)
layout (location = 1) in vec4 in_f0;          // base position xyz, width
layout (location = 2) in vec4 in_f1;          // height, seed, intensity, kind (0 flame, 1 blue CO, 2 lighter)
layout (location = 3) in vec4 in_f2;          // tilt xz, temperature, unused
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
out vec2 v_uv;
out float v_seed;
out float v_int;
out float v_kind;
out float v_temp;
out vec3 v_wpos;
void main() {
    vec3 base = in_f0.xyz;
    float w = in_f0.w;
    float h = in_f1.x;
    vec3 to_cam = u_cam_pos - base;
    vec3 d = normalize(to_cam);
    vec3 flat_fwd = normalize(vec3(to_cam.x, 0.0, to_cam.z) + vec3(1e-4, 0.0, 0.0));
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), flat_fwd));
    // seen from above, an upright sheet would collapse to a sliver: lean it toward the eye
    float lean = smoothstep(0.35, 0.9, d.y) * 0.8;
    vec3 up = normalize(mix(vec3(0.0, 1.0, 0.0), normalize(cross(d, right)), lean));
    float y = in_corner.y;
    vec3 p = base + right * in_corner.x * w * 0.5 + up * (y * h);
    p.xz += in_f2.xy * y * y * h;        // wind bends the upper flame
    v_uv = vec2(in_corner.x * 0.5 + 0.5, y);
    v_seed = in_f1.y;
    v_int = in_f1.z;
    v_kind = in_f1.w;
    v_temp = in_f2.z;
    v_wpos = p;
    gl_Position = u_proj * u_view * vec4(p, 1.0);
}
"""

FLAME_FS = """
#version 330 core
in vec2 v_uv;
in float v_seed;
in float v_int;
in float v_kind;
in float v_temp;
in vec3 v_wpos;
out vec4 frag;
uniform float u_time;
uniform float u_gain;          // exposure-compensated emission (flames are what the eye adapts to)
uniform sampler2D u_depth;
uniform vec2 u_screen;
uniform float u_near;
uniform float u_far;
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec2(3.1, 1.7); a *= 0.5; }
    return s / 0.9375;
}
float lin_depth(float d) {
    float z = d * 2.0 - 1.0;
    return (2.0 * u_near * u_far) / (u_far + u_near - z * (u_far - u_near));
}
// Flame colour from a normalised soot temperature: luminous soot radiates like a ~1000-1400 C
// blackbody - deep red at the cooling tips, orange body, yellow core.
vec3 flame_ramp(float k) {
    vec3 c0 = vec3(0.40, 0.035, 0.004);
    vec3 c1 = vec3(0.95, 0.20, 0.018);
    vec3 c2 = vec3(1.00, 0.45, 0.07);
    vec3 c3 = vec3(1.00, 0.70, 0.26);
    vec3 c4 = vec3(1.00, 0.86, 0.55);
    if (k < 0.25) return mix(c0, c1, k / 0.25);
    if (k < 0.5) return mix(c1, c2, (k - 0.25) / 0.25);
    if (k < 0.8) return mix(c2, c3, (k - 0.5) / 0.3);
    return mix(c3, c4, clamp((k - 0.8) / 0.2, 0.0, 1.0));
}
void main() {
    vec2 uv = v_uv;
    float y = uv.y;
    float x = (uv.x - 0.5) * 2.0;
    float t = u_time + v_seed * 37.0;
    // buoyant eddies rise through the flame (~1.3 flame lengths per second) and puff
    vec2 p = vec2(x * 1.15 + v_seed * 7.3, y * 1.7 - t * 2.1);
    float w = fbm(p * 0.9 + vec2(0.0, t * 0.35));
    float sway = (w - 0.5) * 0.9 * y + sin(t * 2.3 + v_seed * 9.0) * 0.10 * y * y;
    float xs = x - sway;
    float n = fbm(vec2(xs * 1.6, y * 2.3 - t * 2.6) + w * 1.3 + v_seed * 3.0);
    float n2 = vnoise(vec2(xs * 5.0, y * 6.0 - t * 5.5) + v_seed * 11.0);
    // teardrop envelope: rounded base, wide body, tapering to licking tips
    float base = smoothstep(0.0, 0.10, y);
    float prof = mix(1.0, 0.08, pow(clamp(y, 0.0, 1.0), 0.9)) * (0.55 + 0.45 * base);
    float env = 1.0 - abs(xs) / max(prof, 0.02);
    // density field: the envelope eroded by turbulence; higher up only the strongest
    // eddies survive as detached tongues
    float d = env * 0.95 + (n - 0.5) * 1.25 + (n2 - 0.5) * 0.25 - y * 0.75 + 0.12;
    d *= base;
    float body = smoothstep(0.0, 0.22, d);
    if (body < 0.004) discard;
    float k = clamp(d * 1.05 * (1.05 - 0.35 * y), 0.0, 1.0);
    vec3 col;
    float alpha;
    if (v_kind > 0.5 && v_kind < 1.5) {
        // carbon-monoxide flames dancing over white-hot coals: faint, blue-violet
        col = vec3(0.22, 0.38, 1.0) * body * 0.55;
        alpha = body * 0.15;
    } else if (v_kind > 1.5) {
        // lighter: clean butane flame, blue root and a small yellow tip
        col = mix(vec3(0.25, 0.45, 1.0) * 0.8, flame_ramp(0.85), smoothstep(0.15, 0.45, y)) * body;
        alpha = body * 0.5;
    } else {
        col = flame_ramp(k) * body * (0.3 + 0.85 * k);
        // the faint blue root where fresh air meets the volatiles
        col = mix(col, vec3(0.15, 0.25, 0.9) * body * 0.5, smoothstep(0.10, 0.0, y) * 0.55);
        alpha = body * (0.22 + 0.25 * k);
    }
    float scene = lin_depth(texture(u_depth, gl_FragCoord.xy / u_screen).r);
    float me = lin_depth(gl_FragCoord.z);
    float soft = clamp((scene - me) * 14.0, 0.0, 1.0);
    // premultiplied: emission + partial coverage
    frag = vec4(col * v_int * u_gain * soft, alpha * soft);
}
"""

# ------------------------------------------------------------------------------------ particles
PARTICLE_VS = """
#version 330 core
layout (location = 0) in vec2 in_corner;
layout (location = 1) in vec4 in_p0;      // position xyz, size
layout (location = 2) in vec4 in_p1;      // rgb, alpha
layout (location = 3) in vec4 in_p2;      // velocity xyz (for streaks), kind
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_cam_pos;
out vec2 v_uv;
out vec4 v_col;
out float v_kind;
out vec3 v_wpos;
out float v_seed;
void main() {
    vec3 pos = in_p0.xyz;
    float size = in_p0.w;
    float kind = in_p2.w;
    vec3 right = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 up = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);
    vec3 p;
    if (kind > 1.5 && kind < 3.5) {
        // velocity-aligned streaks (sparks = 2, rain = 3)
        vec3 vel = in_p2.xyz;
        float sp = length(vel);
        vec3 dir = sp > 1e-4 ? vel / sp : up;
        vec3 to_cam = normalize(u_cam_pos - pos);
        vec3 side = normalize(cross(dir, to_cam));
        float len = kind < 2.5 ? size * 0.5 + sp * 0.012 : size * 4.0 + sp * 0.017;
        p = pos + side * in_corner.x * size * 0.5 + dir * in_corner.y * len;
    } else {
        float rot = fract(sin(dot(pos.xz, vec2(12.9898, 78.233))) * 43758.5) * 6.28;
        float c = cos(rot), s = sin(rot);
        vec2 cr = vec2(c * in_corner.x - s * in_corner.y, s * in_corner.x + c * in_corner.y);
        p = pos + (right * cr.x + up * cr.y) * size * 0.5;
    }
    v_uv = in_corner * 0.5 + 0.5;
    v_col = in_p1;
    v_kind = kind;
    v_wpos = pos;
    v_seed = fract(sin(dot(pos.xy, vec2(3.7, 9.1))) * 1000.0);
    gl_Position = u_proj * u_view * vec4(p, 1.0);
}
"""

PARTICLE_FS = """
#version 330 core
in vec2 v_uv;
in vec4 v_col;
in float v_kind;
in vec3 v_wpos;
in float v_seed;
out vec4 frag;
uniform sampler2D u_depth;
uniform vec2 u_screen;
uniform float u_near;
uniform float u_far;
uniform int u_additive;
uniform float u_emit_gain;
""" + COMMON + """
float lin_depth(float d) {
    float z = d * 2.0 - 1.0;
    return (2.0 * u_near * u_far) / (u_far + u_near - z * (u_far - u_near));
}
void main() {
    vec2 c = v_uv - 0.5;
    float r = length(c) * 2.0;
    float scene = lin_depth(texture(u_depth, gl_FragCoord.xy / u_screen).r);
    float me = lin_depth(gl_FragCoord.z);
    float kind = v_kind;
    if (u_additive == 1) {
        // sparks / embers / fireflies (emissive)
        if (kind > 1.5 && kind < 2.5) {
            float a = smoothstep(1.0, 0.0, abs(c.x) * 2.0) * smoothstep(1.0, 0.2, abs(c.y) * 2.0);
            frag = vec4(v_col.rgb * a * v_col.a * u_emit_gain, 1.0);
        } else {
            float a = smoothstep(1.0, 0.0, r);
            a *= a;
            frag = vec4(v_col.rgb * a * v_col.a * u_emit_gain, 1.0);
        }
        if (scene < me - 0.05) discard;
        return;
    }
    float alpha;
    vec3 col = v_col.rgb;
    float cam_d = length(v_wpos - u_cam_pos);
    if (kind > 2.5 && kind < 3.5) {          // rain streak (catches sky light and the firelight)
        alpha = smoothstep(1.0, 0.0, abs(c.x) * 2.0) * v_col.a * smoothstep(0.9, 2.2, cam_d);
        vec3 lit = col * (u_sky_amb * 2.5 + u_sun_color * 0.35 + 0.12) + fire_light(v_wpos, vec3(0, 1, 0), 1.0) * 1.6;
        frag = vec4(lit * alpha, alpha);
        return;
    }
    if (kind > 3.5 && kind < 4.5) {          // snowflake
        alpha = smoothstep(1.0, 0.3, r) * v_col.a * smoothstep(0.35, 1.0, cam_d);
        vec3 lit = col * (u_sky_amb * 1.4 + u_sun_color * 0.5 + 0.06) + fire_light(v_wpos, vec3(0, 1, 0), 1.0) * 0.6;
        frag = vec4(lit * alpha, alpha);
        return;
    }
    if (kind > 4.5 && kind < 5.5) {          // falling leaf
        float leaf = smoothstep(1.0, 0.6, length(c * vec2(1.0, 2.2)) * 2.0);
        alpha = leaf * v_col.a;
        vec3 lit = shade(col, vec3(0, 1, 0), v_wpos, 0.9, 0.0, 0.8, 1.0);
        frag = vec4(lit * alpha, alpha);
        return;
    }
    // smoke / steam puffs: soft billowing noise, lit by sun, sky and the fire below
    float n = fbm(c * 3.0 + v_seed * 10.0 + u_time * 0.15);
    alpha = smoothstep(1.0, 0.15, r + (n - 0.5) * 0.55) * v_col.a;
    float soft = clamp((scene - me) * 3.0, 0.0, 1.0);
    alpha *= soft;
    if (alpha < 0.003) discard;
    vec3 L = normalize(u_sun_dir);
    float scatter = 0.55 + 0.45 * pow(max(dot(normalize(v_wpos - u_cam_pos), L), 0.0), 3.0);
    vec3 lit = col * (u_sky_amb * 1.8 + u_sun_color * scatter * 0.7 + 0.02);
    // smoke is lit from below by the flames: only single scattering of a fraction of the light
    lit += col * min(fire_light(v_wpos, vec3(0.0, -1.0, 0.0), 1.0), vec3(1.2)) * 0.35;
    frag = vec4(apply_fog(lit, v_wpos) * alpha, alpha);
}
"""

# ------------------------------------------------------------------------------------ ember bed
EMBER_FS = """
#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_col;
in vec3 v_tint;
in float v_seed;
in vec3 v_local;
out vec4 frag;
uniform float u_bed_T;
uniform float u_bed_amount;
uniform float u_ash;
uniform float u_bed_radius;
uniform float u_breath;
""" + COMMON + """
float voronoi(vec2 p, out float edge) {
    vec2 i = floor(p); vec2 f = fract(p);
    float d1 = 8.0, d2 = 8.0;
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        vec2 g = vec2(x, y);
        vec2 o = vec2(hash12(i + g), hash12(i + g + 5.7));
        float d = length(g + o - f);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
    }
    edge = d2 - d1;
    return d1;
}
void main() {
    vec2 p = v_local.xz;
    float r = length(p) / max(0.02, u_bed_radius);
    if (r > 1.0) discard;
    float edge;
    vec2 cp = p * 38.0;
    float d = voronoi(cp, edge);
    vec2 cell = floor(cp + 0.5);
    float lump = smoothstep(0.0, 0.25, edge);
    float cellT = u_bed_T * (0.75 + 0.35 * hash12(floor(cp))) * (1.0 - 0.35 * r * r);
    float pulse = 0.8 + 0.2 * sin(u_time * (1.2 + hash12(floor(cp)) * 2.0) + hash12(floor(cp) + 3.0) * 6.28);
    pulse += u_breath * 0.5;
    vec3 N = normalize(vec3(0.0, 1.0, 0.0) + vec3(cos(d * 6.0), 0.0, sin(d * 6.0)) * 0.25 * lump);
    vec3 ashc = vec3(0.5, 0.49, 0.47) * (0.7 + 0.3 * vnoise(cp * 0.7));
    vec3 charc = vec3(0.03, 0.028, 0.026);
    float ashm = smoothstep(0.2, 1.0, u_ash) * smoothstep(0.3, 0.7, vnoise(cp * 0.25 + 2.0));
    vec3 base = mix(charc, ashc, ashm);
    vec3 col = shade(base, N, v_pos, 0.95, 0.02, 0.2, 1.0);
    vec3 glow = blackbody(cellT * pulse) * (0.25 + 1.2 * (1.0 - lump)) * (1.0 - 0.85 * ashm);
    col += glow;
    float a = smoothstep(1.0, 0.82, r) * clamp(u_bed_amount * 4.0, 0.0, 1.0);
    frag = vec4(apply_fog(col, v_pos), a);
}
"""

# ------------------------------------------------------------------------------------ hologram ghost
GHOST_FS = """
#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_col;
in vec3 v_tint;
in float v_seed;
in vec3 v_local;
out vec4 frag;
uniform vec4 u_color;
uniform float u_time;
uniform vec3 u_cam_pos;
void main() {
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(u_cam_pos - v_pos);
    float fres = pow(1.0 - abs(dot(N, V)), 2.0);
    float scan = 0.6 + 0.4 * sin(v_pos.y * 140.0 - u_time * 6.0);
    float a = (0.18 + 0.8 * fres) * scan * u_color.a;
    frag = vec4(u_color.rgb * a * 1.6, a);
}
"""

# ------------------------------------------------------------------------------------ post-processing
FULLSCREEN_VS = """
#version 330 core
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

BLOOM_DOWN_FS = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform int u_first;
uniform float u_threshold;
vec3 prefilter(vec3 c) {
    float br = max(c.r, max(c.g, c.b));
    float soft = clamp(br - u_threshold + 0.5, 0.0, 1.0);
    soft = soft * soft * 0.5;
    float contrib = max(soft, br - u_threshold) / max(br, 1e-4);
    return c * contrib;
}
void main() {
    vec2 t = u_texel;
    vec3 a = texture(u_src, v_uv + t * vec2(-2, 2)).rgb;
    vec3 b = texture(u_src, v_uv + t * vec2(0, 2)).rgb;
    vec3 c = texture(u_src, v_uv + t * vec2(2, 2)).rgb;
    vec3 d = texture(u_src, v_uv + t * vec2(-2, 0)).rgb;
    vec3 e = texture(u_src, v_uv).rgb;
    vec3 f = texture(u_src, v_uv + t * vec2(2, 0)).rgb;
    vec3 g = texture(u_src, v_uv + t * vec2(-2, -2)).rgb;
    vec3 h = texture(u_src, v_uv + t * vec2(0, -2)).rgb;
    vec3 i = texture(u_src, v_uv + t * vec2(2, -2)).rgb;
    vec3 j = texture(u_src, v_uv + t * vec2(-1, 1)).rgb;
    vec3 k = texture(u_src, v_uv + t * vec2(1, 1)).rgb;
    vec3 l = texture(u_src, v_uv + t * vec2(-1, -1)).rgb;
    vec3 m = texture(u_src, v_uv + t * vec2(1, -1)).rgb;
    vec3 col = e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
    if (u_first == 1) col = prefilter(min(col, vec3(60.0)));
    frag = vec4(col, 1.0);
}
"""

BLOOM_UP_FS = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_radius;
void main() {
    vec2 t = u_texel * u_radius;
    vec3 c = texture(u_src, v_uv + t * vec2(-1, 1)).rgb + texture(u_src, v_uv + t * vec2(0, 1)).rgb * 2.0
           + texture(u_src, v_uv + t * vec2(1, 1)).rgb + texture(u_src, v_uv + t * vec2(-1, 0)).rgb * 2.0
           + texture(u_src, v_uv).rgb * 4.0 + texture(u_src, v_uv + t * vec2(1, 0)).rgb * 2.0
           + texture(u_src, v_uv + t * vec2(-1, -1)).rgb + texture(u_src, v_uv + t * vec2(0, -1)).rgb * 2.0
           + texture(u_src, v_uv + t * vec2(1, -1)).rgb;
    frag = vec4(c / 16.0, 1.0);
}
"""

COMPOSITE_FS = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_depth;
uniform float u_exposure;
uniform float u_bloom_strength;
uniform float u_time;
uniform vec4 u_heat;         // screen xy of the fire base, radius, strength
uniform float u_heat_top;
uniform float u_fire_depth;
uniform float u_near;
uniform float u_far;
uniform vec3 u_grade_shadow;
uniform vec3 u_grade_high;
uniform float u_saturation;
uniform float u_vignette;
uniform float u_flash;
uniform float u_night_contrast;
uniform vec2 u_aspect;
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
}
float lin_depth(float d) {
    float z = d * 2.0 - 1.0;
    return (2.0 * u_near * u_far) / (u_far + u_near - z * (u_far - u_near));
}
vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main() {
    vec2 uv = v_uv;
    // heat shimmer above the flames (only on pixels behind the fire)
    if (u_heat.w > 0.001) {
        vec2 d = (uv - u_heat.xy) * u_aspect;
        float above = smoothstep(-0.02, 0.05, uv.y - u_heat.y) * smoothstep(u_heat_top + 0.15, u_heat_top - 0.05, uv.y);
        float side = smoothstep(u_heat.z * 1.2, u_heat.z * 0.3, abs(d.x));
        float mask = above * side;
        float sd = lin_depth(texture(u_depth, uv).r);
        mask *= smoothstep(u_fire_depth - 0.4, u_fire_depth + 0.2, sd);
        if (mask > 0.001) {
            vec2 n = vec2(vnoise(uv * vec2(40.0, 22.0) + vec2(0.0, -u_time * 3.5)),
                          vnoise(uv * vec2(38.0, 20.0) + vec2(5.0, -u_time * 3.1))) - 0.5;
            uv += n * 0.008 * u_heat.w * mask;
        }
    }
    vec3 col = texture(u_scene, uv).rgb;
    vec3 bloom = texture(u_bloom, uv).rgb;
    col += bloom * u_bloom_strength;
    col *= u_exposure;
    col += vec3(1.0, 0.7, 0.4) * u_flash;
    col = aces(col);
    // night: deeper shadows (a campfire scene is a pool of light in the dark, as a camera sees it)
    col = mix(col, pow(col, vec3(1.3)), u_night_contrast);
    // colour grade: split-tone shadows / highlights, saturation
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(col * u_grade_shadow, col * u_grade_high, smoothstep(0.05, 0.7, l));
    col = mix(vec3(l), col, u_saturation);
    // vignette
    vec2 vc = (v_uv - 0.5) * vec2(1.0, 0.8);
    col *= 1.0 - u_vignette * smoothstep(0.25, 0.85, length(vc) * 1.35);
    // gamma + dither (kills banding in the night sky)
    col = pow(max(col, 0.0), vec3(1.0 / 2.2));
    col += (hash12(gl_FragCoord.xy + u_time) - 0.5) / 255.0;
    frag = vec4(col, 1.0);
}
"""

FXAA_FS = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform int u_enabled;
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
void main() {
    vec3 c = texture(u_src, v_uv).rgb;
    if (u_enabled == 0) { frag = vec4(c, 1.0); return; }
    float lM = luma(c);
    float lN = luma(texture(u_src, v_uv + vec2(0, u_texel.y)).rgb);
    float lS = luma(texture(u_src, v_uv - vec2(0, u_texel.y)).rgb);
    float lE = luma(texture(u_src, v_uv + vec2(u_texel.x, 0)).rgb);
    float lW = luma(texture(u_src, v_uv - vec2(u_texel.x, 0)).rgb);
    float mn = min(lM, min(min(lN, lS), min(lE, lW)));
    float mx = max(lM, max(max(lN, lS), max(lE, lW)));
    float range = mx - mn;
    if (range < max(0.04, mx * 0.125)) { frag = vec4(c, 1.0); return; }
    vec2 dir = vec2(-((lN + lS) - (lE + lW)), (lN + lE) - (lS + lW));
    dir = normalize(dir + 1e-5) * u_texel;
    vec3 a = 0.5 * (texture(u_src, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb + texture(u_src, v_uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 b = a * 0.5 + 0.25 * (texture(u_src, v_uv + dir * -0.5).rgb + texture(u_src, v_uv + dir * 0.5).rgb);
    float lb = luma(b);
    frag = vec4((lb < mn || lb > mx) ? a : b, 1.0);
}
"""

UI_VS = """
#version 330 core
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = vec2(p.x, 1.0 - p.y);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

UI_FS = """
#version 330 core
in vec2 v_uv;
out vec4 frag;
uniform sampler2D u_texture;
void main() {
    frag = texture(u_texture, v_uv);
}
"""

# Legacy names kept for compatibility with older imports
SCENE_VERTEX_SHADER = OBJECT_VS
SCENE_FRAGMENT_SHADER = OBJECT_FS
PARTICLE_VERTEX_SHADER = PARTICLE_VS
PARTICLE_FRAGMENT_SHADER = PARTICLE_FS
UI_VERTEX_SHADER = UI_VS
UI_FRAGMENT_SHADER = UI_FS


def with_defines(src: str, *defines: str) -> str:
    lines = src.strip("\n").split("\n")
    head = lines[0]
    extra = "\n".join(f"#define {d}" for d in defines)
    return head + "\n" + extra + "\n" + "\n".join(lines[1:])
