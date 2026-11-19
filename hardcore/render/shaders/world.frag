#version 330 core
in vec3 v_FragPos;
in vec3 v_Normal;
in vec2 v_UV;        // metres across the face: u along it (back -> front on a vehicle's sides), v up
in vec3 v_Color;
flat in vec3 v_Extra;  // variant * 1000 + material * 10 + role, face width, face height

uniform vec3 u_LightDir;
uniform vec3 u_CamPos;

out vec4 FragColor;

// materials (world/obstacle.py)
const int MAT_PLAIN = 0, MAT_CAR_BODY = 1, MAT_CAR_GLASS = 2, MAT_VAN = 3, MAT_BUS = 4, MAT_TRUCK_CAB = 5,
          MAT_CONTAINER = 6, MAT_WHEEL_SIDE = 7, MAT_WHEEL_TREAD = 8, MAT_CHASSIS = 9, MAT_WALL = 10,
          MAT_WINDOW_WALL = 11, MAT_BILLBOARD = 12, MAT_LIGHT = 13, MAT_BARK = 14, MAT_LEAVES = 15,
          MAT_ASPHALT = 16, MAT_PAVEMENT = 17, MAT_TRASH = 18;
const int ROLE_SIDE = 0, ROLE_FRONT = 1, ROLE_BACK = 2, ROLE_TOP = 3, ROLE_BOTTOM = 4;

const vec3 GLASS = vec3(0.16, 0.22, 0.28);
const vec3 DARK = vec3(0.06, 0.06, 0.07);
const vec3 CHROME = vec3(0.70, 0.72, 0.74);
const vec3 HEADLIGHT = vec3(0.95, 0.95, 0.85);
const vec3 TAILLIGHT = vec3(0.85, 0.10, 0.08);
const vec3 AMBER = vec3(0.95, 0.60, 0.10);

// ------------------------------------------------------------------ noise
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0)), c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    return 0.5 * vnoise(p) + 0.25 * vnoise(p * 2.03 + 7.1) + 0.125 * vnoise(p * 4.11 + 3.7) + 0.0625 * vnoise(p * 8.3 + 1.3);
}
// a stable 2D seed for this surface point (so detail never repeats across faces)
vec2 worldSeed() { return v_FragPos.xz * 0.73 + v_FragPos.yy * vec2(1.31, 0.57); }

// ----------------------------------------------------------------- shapes
float box(vec2 p, vec2 a, vec2 b) { vec2 d = step(a, p) * step(p, b); return d.x * d.y; }
float vline(float u, float at, float w) { return step(abs(u - at), w * 0.5); }
float hline(float v, float at, float w) { return step(abs(v - at), w * 0.5); }
float disc(vec2 p, vec2 c, float r) { return step(length(p - c), r); }
float glassPane(vec2 uv, vec2 a, vec2 b, float frame) { return box(uv, a + vec2(frame), b - vec2(frame)); }

vec3 palette(int k) {
    // transit / mural / graffiti colours
    if (k == 0) return vec3(0.85, 0.15, 0.12);
    if (k == 1) return vec3(0.10, 0.45, 0.80);
    if (k == 2) return vec3(0.95, 0.70, 0.10);
    if (k == 3) return vec3(0.15, 0.62, 0.35);
    if (k == 4) return vec3(0.85, 0.35, 0.65);
    if (k == 5) return vec3(0.95, 0.45, 0.10);
    if (k == 6) return vec3(0.35, 0.20, 0.65);
    if (k == 7) return vec3(0.10, 0.70, 0.70);
    if (k == 8) return vec3(0.92, 0.92, 0.90);
    return vec3(0.12, 0.12, 0.14);
}

// cracked glass: a few bright fracture lines radiating from a point
float cracks(vec2 uv, vec2 c, float extent) {
    float d = length(uv - c);
    if (d > extent) return 0.0;
    float ang = atan(uv.y - c.y, uv.x - c.x);
    float rays = step(0.92, fract(ang * 1.75 + 0.2)) + step(0.95, fract(ang * 2.9 + 0.61));
    float rings = step(0.93, fract(d * 6.0 + fbm(uv * 9.0) * 0.6));
    return clamp(rays * step(0.02, d) + rings * step(d, extent * 0.8), 0.0, 1.0) * (1.0 - d / extent);
}

// weathering: water streaks from the top, general grime low down, hairline cracks
vec3 weather(vec3 color, vec2 uv, float W, float H, float amount) {
    vec2 sd = worldSeed();
    float streak = vnoise(vec2(uv.x * 3.0 + sd.x, 0.5)) * vnoise(vec2(uv.x * 11.0 + sd.y, 1.7));
    float fromTop = smoothstep(0.0, 2.5, H - uv.y);
    float fromTopFade = 1.0 - smoothstep(0.0, 5.0, H - uv.y);
    float runs = smoothstep(0.45, 0.8, streak) * fromTopFade * fromTop;
    float grime = smoothstep(0.3, 0.75, fbm(uv * 0.9 + sd)) * (1.0 - smoothstep(0.0, 2.2, uv.y));
    float crackN = fbm(uv * 1.7 + sd * 0.3);
    float hairline = step(abs(crackN - 0.5), 0.004) * step(0.55, fbm(uv * 0.35 + sd));
    color *= 1.0 - amount * (0.35 * runs + 0.3 * grime);
    color = mix(color, color * 0.45, hairline * amount);
    return color;
}

// street-level tags: overlapping rounded colour shapes with a dark outline
vec3 graffiti(vec3 color, vec2 uv, float W, float H, int variant) {
    vec2 sd = worldSeed();
    // tags are sprayed from the street: only within reach of the ground (world height)
    float band = step(0.35, v_FragPos.y) * step(v_FragPos.y, 2.3);
    if (band < 0.5) return color;
    // tags sit at a few spots along the wall
    float spots = 0.0;
    vec3 tagCol = color;
    for (int i = 0; i < 3; i++) {
        float fi = float(i);
        float cu = 1.5 + fract(hash21(vec2(variant, fi + 3.0) + floor(sd * 0.01)) * 7.0) * max(1.0, W - 3.0);
        if (cu > W - 1.0) continue;
        vec2 c = vec2(cu, 1.2 + 0.4 * sin(fi * 2.1));
        vec2 q = vec2(uv.x, v_FragPos.y);
        vec2 d = (q - c) / vec2(1.4, 0.55);
        float blob = 1.0 - smoothstep(0.85, 1.0, length(d) + 0.25 * fbm(q * 4.0 + fi));
        float inner = 1.0 - smoothstep(0.45, 0.6, length(d) + 0.25 * fbm(q * 4.0 + fi));
        vec3 c1 = palette(int(mod(float(variant) + fi * 2.0, 8.0)));
        vec3 c2 = palette(int(mod(float(variant) + fi * 2.0 + 3.0, 8.0)));
        float outline = blob * (1.0 - smoothstep(0.0, 0.12, blob));
        tagCol = mix(tagCol, c1, blob);
        tagCol = mix(tagCol, c2, inner);
        tagCol = mix(tagCol, DARK, clamp(outline * 4.0, 0.0, 1.0) * blob);
        spots = max(spots, blob);
    }
    return mix(color, tagCol, clamp(spots, 0.0, 1.0) * 0.9);
}

// a mural: big geometric shapes across the whole wall
vec3 mural(vec3 color, vec2 uv, float W, float H, int variant) {
    if (W < 5.0 || H < 5.0) return color;
    vec2 p = uv / vec2(W, H);
    vec3 bg = palette(variant % 8) * 0.9;
    vec3 a = palette((variant + 3) % 8);
    vec3 b = palette((variant + 5) % 8);
    vec3 m = bg;
    float sun = 1.0 - smoothstep(0.28, 0.30, length((p - vec2(0.62, 0.62)) * vec2(W / H, 1.0) * 0.7));
    float hills = step(p.y, 0.28 + 0.12 * sin(p.x * 9.0 + float(variant)) + 0.08 * sin(p.x * 23.0));
    float stripes = step(0.5, fract((p.x + p.y) * 6.0));
    m = mix(m, a, sun);
    m = mix(m, b * (0.8 + 0.2 * stripes), hills);
    float frame = step(0.04, p.x) * step(p.x, 0.96) * step(0.05, p.y) * step(p.y, 0.95);
    m = mix(vec3(0.12, 0.12, 0.13), m, frame);
    // faded and chipped
    float chip = smoothstep(0.55, 0.8, fbm(uv * 1.3 + worldSeed()));
    return mix(m * 0.92, color, chip * 0.8);
}

void main() {
    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightDir);
    vec3 viewDir = normalize(u_CamPos - v_FragPos);
    int code = int(v_Extra.x + 0.5);
    int variant = code / 1000;
    int mat = (code - variant * 1000) / 10;
    int role = code - variant * 1000 - mat * 10;
    float W = v_Extra.y;
    float H = v_Extra.z;
    vec2 uv = v_UV;
    vec3 base = v_Color;
    vec3 color = base;
    float gloss = 0.0;          // specular strength
    float emissive = 0.0;
    vec3 n = norm;
    vec2 sd = worldSeed();

    // ------------------------------------------------------------- materials
    if (mat == MAT_CAR_BODY) {
        gloss = 0.35;
        if (role == ROLE_SIDE) {
            // two doors with seams and handles, a sill, a trim line
            float seam = vline(uv.x, W * 0.30, 0.025) + vline(uv.x, W * 0.52, 0.025) + vline(uv.x, W * 0.74, 0.025);
            seam *= step(0.10, uv.y);
            float handle = box(uv, vec2(W * 0.52 - 0.22, 0.46), vec2(W * 0.52 - 0.06, 0.52))
                         + box(uv, vec2(W * 0.74 - 0.22, 0.46), vec2(W * 0.74 - 0.06, 0.52));
            float sill = step(uv.y, 0.07);
            float trim = hline(uv.y, H - 0.02, 0.03);
            color = mix(color, color * 0.55, clamp(seam, 0.0, 1.0));
            color = mix(color, DARK, clamp(handle, 0.0, 1.0));
            color = mix(color, color * 0.6, sill);
            color = mix(color, color * 0.8, trim);
            if ((variant & 1) == 1) {
                // a dent in the rear door: a shaded hollow with scuffed paint
                vec2 dc = vec2(W * 0.62, 0.35);
                float dd = length((uv - dc) / vec2(0.32, 0.2));
                float dent = 1.0 - smoothstep(0.6, 1.0, dd);
                color = mix(color, color * 0.6, dent * 0.7);
                color = mix(color, vec3(0.55, 0.55, 0.56), dent * step(0.6, fbm(uv * 14.0 + sd)) * 0.7);
                n = normalize(norm + vec3(0.0, 0.35, 0.0) * dent * sign(uv.y - dc.y));
            }
        } else if (role == ROLE_FRONT) {
            float light = box(uv, vec2(0.12, H * 0.55), vec2(0.52, H * 0.85)) + box(uv, vec2(W - 0.52, H * 0.55), vec2(W - 0.12, H * 0.85));
            float grille = box(uv, vec2(0.62, H * 0.50), vec2(W - 0.62, H * 0.80));
            float slats = step(0.5, fract(uv.y * 12.0));
            float bumper = step(uv.y, H * 0.30);
            float plate = box(uv, vec2(W * 0.5 - 0.26, H * 0.08), vec2(W * 0.5 + 0.26, H * 0.24));
            color = mix(color, color * 0.75, bumper);
            color = mix(color, mix(DARK, color * 0.5, slats), grille);
            color = mix(color, HEADLIGHT, light);
            color = mix(color, vec3(0.85, 0.85, 0.80), plate);
            if (light > 0.5) gloss = 0.8;
        } else if (role == ROLE_BACK) {
            float light = box(uv, vec2(0.10, H * 0.55), vec2(0.60, H * 0.85)) + box(uv, vec2(W - 0.60, H * 0.55), vec2(W - 0.10, H * 0.85));
            float indicator = box(uv, vec2(0.10, H * 0.55), vec2(0.25, H * 0.85)) + box(uv, vec2(W - 0.25, H * 0.55), vec2(W - 0.10, H * 0.85));
            float bumper = step(uv.y, H * 0.30);
            float plate = box(uv, vec2(W * 0.5 - 0.26, H * 0.36), vec2(W * 0.5 + 0.26, H * 0.52));
            float seam = hline(uv.y, H * 0.92, 0.025);
            color = mix(color, color * 0.75, bumper);
            color = mix(color, TAILLIGHT, light);
            color = mix(color, AMBER, indicator);
            color = mix(color, vec3(0.85, 0.85, 0.80), plate);
            color = mix(color, color * 0.6, seam);
            if (light > 0.5) gloss = 0.8;
        } else if (role == ROLE_TOP) {
            float seam = vline(uv.x, 0.9, 0.02) + vline(uv.x, W - 0.9, 0.02);
            color = mix(color, color * 0.7, clamp(seam, 0.0, 1.0));
        }
    } else if (mat == MAT_CAR_GLASS) {
        gloss = 0.6;
        float pane = 0.0;
        if (role == ROLE_SIDE) {
            pane = glassPane(uv, vec2(0.0, 0.0), vec2(W * 0.5, H), 0.07) + glassPane(uv, vec2(W * 0.5, 0.0), vec2(W, H), 0.07);
            pane *= step(0.06, uv.y);
        } else if (role == ROLE_FRONT || role == ROLE_BACK) {
            pane = glassPane(uv, vec2(0.0, 0.0), vec2(W, H), 0.10) * step(0.05, uv.y);
        }
        color = mix(base, GLASS, pane);
        if ((variant & 2) == 2 && pane > 0.5 && (role == ROLE_BACK || role == ROLE_SIDE)) {
            float cr = cracks(uv, vec2(W * 0.3, H * 0.55), 0.5);
            color = mix(color, vec3(0.9, 0.93, 0.95), cr);
        }
        if (pane < 0.5) gloss = 0.3;
    } else if (mat == MAT_VAN || mat == MAT_TRUCK_CAB || mat == MAT_BUS) {
        gloss = 0.3;
        bool school = (mat == MAT_BUS && variant == 0);
        vec3 band = palette(variant - 1);
        float winLo = (mat == MAT_BUS) ? 1.15 : (mat == MAT_VAN ? 1.05 : 1.45);
        float winHi = H - 0.22;
        if (role == ROLE_SIDE) {
            if (mat == MAT_VAN) {
                float cabWin = glassPane(uv, vec2(W - 1.75, winLo), vec2(W - 0.25, winHi), 0.05);
                float doorSeam = vline(uv.x, W - 1.75, 0.025) + vline(uv.x, W - 3.35, 0.025);
                doorSeam *= step(0.12, uv.y);
                float handle = box(uv, vec2(W - 1.75 - 0.22, 0.95), vec2(W - 1.75 - 0.06, 1.01))
                             + box(uv, vec2(W - 3.35 + 0.06, 0.95), vec2(W - 3.35 + 0.22, 1.01));
                float stripe = hline(uv.y, 0.75, 0.06);
                color = mix(color, color * 0.8, stripe);
                color = mix(color, color * 0.55, clamp(doorSeam, 0.0, 1.0));
                color = mix(color, DARK, clamp(handle, 0.0, 1.0));
                color = mix(color, GLASS, cabWin);
                if (cabWin > 0.5) gloss = 0.6;
            } else if (mat == MAT_BUS) {
                // a band of windows with pillars every 1.5 m, a front door and a middle door
                float inBand = step(winLo, uv.y) * step(uv.y, winHi);
                float pillar = step(fract((uv.x - 0.3) / 1.5), 0.08 / 1.5);
                float win = inBand * (1.0 - pillar) * step(0.3, uv.x) * step(uv.x, W - 0.3);
                float door1 = box(uv, vec2(W - 1.9, 0.12), vec2(W - 0.7, winHi));
                float door2 = box(uv, vec2(W * 0.5 - 0.6, 0.12), vec2(W * 0.5 + 0.6, winHi));
                float doorSeam = vline(uv.x, W - 1.3, 0.03) + vline(uv.x, W * 0.5, 0.03) + vline(uv.x, W - 1.9, 0.03) + vline(uv.x, W - 0.7, 0.03)
                               + vline(uv.x, W * 0.5 - 0.6, 0.03) + vline(uv.x, W * 0.5 + 0.6, 0.03);
                float doorGlass = (door1 + door2) * step(1.3, uv.y) * (1.0 - doorSeam);
                float skirt = step(uv.y, 0.35);
                if (school) {
                    // black rub rails along the yellow body
                    float rails = hline(uv.y, 0.62, 0.10) + hline(uv.y, 0.98, 0.10) + hline(uv.y, H - 0.12, 0.08);
                    color = mix(color, DARK, clamp(rails, 0.0, 1.0));
                    color = mix(color, DARK, skirt);
                } else {
                    // metro livery: white body, a coloured belt under the windows and a roof line
                    float belt = step(0.75, uv.y) * step(uv.y, winLo - 0.02);
                    float roofline = step(winHi + 0.03, uv.y);
                    float swoosh = step(uv.y, 0.75 + 0.35 * smoothstep(W - 3.5, W - 0.6, uv.x)) * step(0.35, uv.y) * step(W - 3.5, uv.x);
                    color = mix(color, band, clamp(belt + swoosh, 0.0, 1.0));
                    color = mix(color, band * 0.7, roofline);
                    color = mix(color, vec3(0.30, 0.30, 0.32), skirt);
                    // a route number box by the front door
                    float routeBox = box(uv, vec2(W - 0.65, winHi - 0.42), vec2(W - 0.3, winHi - 0.12));
                    color = mix(color, vec3(0.1, 0.1, 0.1), routeBox);
                    color = mix(color, AMBER, routeBox * step(0.5, fract(uv.x * 9.0)) * step(0.4, fract(uv.y * 7.0)));
                }
                color = mix(color, GLASS, clamp(win, 0.0, 1.0));
                color = mix(color, color * 0.5, clamp(doorSeam * (door1 + door2), 0.0, 1.0));
                color = mix(color, GLASS, clamp(doorGlass, 0.0, 1.0));
                if (win > 0.5 || doorGlass > 0.5) gloss = 0.6;
            } else {
                // truck cab: a tall door with a window, a step and a grille band
                float win = glassPane(uv, vec2(0.3, winLo), vec2(W - 0.3, winHi), 0.06);
                float seam = vline(uv.x, 0.3, 0.03) * step(0.3, uv.y);
                float handle = box(uv, vec2(0.36, 1.2), vec2(0.52, 1.26));
                float stepBar = box(uv, vec2(0.3, 0.0), vec2(W - 0.3, 0.10));
                float grilleBand = step(uv.y, 0.7) * step(0.5, fract(uv.y * 10.0));
                color = mix(color, color * 0.75, grilleBand);
                color = mix(color, color * 0.5, seam);
                color = mix(color, DARK, handle);
                color = mix(color, CHROME, stepBar);
                color = mix(color, GLASS, win);
                if (win > 0.5) gloss = 0.6;
            }
            if ((variant & 1) == 1 && mat != MAT_BUS) {      // (a bus's variant is its livery)
                // a scuffed dent low on the body side
                vec2 dc = vec2(W * 0.35, 0.55);
                float dd = length((uv - dc) / vec2(0.4, 0.25));
                float dent = 1.0 - smoothstep(0.6, 1.0, dd);
                color = mix(color, color * 0.6, dent * 0.7 * step(uv.y, winLo - 0.05));
            }
        } else if (role == ROLE_FRONT) {
            float wsLo = (mat == MAT_BUS) ? 1.15 : (mat == MAT_VAN ? 1.05 : 1.45);
            float windshield = glassPane(uv, vec2(0.15, wsLo), vec2(W - 0.15, H - 0.18), 0.05);
            float light = box(uv, vec2(0.15, 0.55), vec2(0.60, 0.85)) + box(uv, vec2(W - 0.60, 0.55), vec2(W - 0.15, 0.85));
            float grille = box(uv, vec2(0.75, 0.45), vec2(W - 0.75, 0.95)) * step(0.5, fract(uv.y * 10.0));
            float bumper = step(uv.y, 0.32);
            float plate = box(uv, vec2(W * 0.5 - 0.26, 0.10), vec2(W * 0.5 + 0.26, 0.26));
            color = mix(color, school ? DARK : color * 0.7, bumper);
            color = mix(color, DARK, grille);
            color = mix(color, HEADLIGHT, light);
            color = mix(color, vec3(0.85, 0.85, 0.80), plate);
            if (mat == MAT_BUS && !school) {
                float dest = box(uv, vec2(0.3, H - 0.55), vec2(W - 0.3, H - 0.2));
                color = mix(color, vec3(0.08, 0.08, 0.08), dest);
                color = mix(color, AMBER, dest * step(0.55, fract(uv.x * 8.0)) * step(0.35, fract(uv.y * 12.0)) * step(0.5, uv.x) * step(uv.x, W - 0.5));
                windshield *= step(uv.y, H - 0.6);
            }
            color = mix(color, GLASS, windshield);
            if (windshield > 0.5 || light > 0.5) gloss = 0.7;
        } else if (role == ROLE_BACK) {
            float light = box(uv, vec2(0.12, 0.55), vec2(0.45, 1.05)) + box(uv, vec2(W - 0.45, 0.55), vec2(W - 0.12, 1.05));
            float bumper = step(uv.y, 0.32);
            float doorSeam = vline(uv.x, W * 0.5, 0.03) * step(0.32, uv.y) * step(uv.y, H - 0.15);
            float doorWin = (mat == MAT_BUS)
                ? glassPane(uv, vec2(0.3, 1.3), vec2(W - 0.3, H - 0.2), 0.06)
                : (glassPane(uv, vec2(0.15, H - 0.85), vec2(W * 0.5, H - 0.18), 0.06) + glassPane(uv, vec2(W * 0.5, H - 0.85), vec2(W - 0.15, H - 0.18), 0.06));
            if (mat == MAT_TRUCK_CAB) doorWin = 0.0;
            float handle = box(uv, vec2(W * 0.5 - 0.2, 1.0), vec2(W * 0.5 - 0.05, 1.06)) + box(uv, vec2(W * 0.5 + 0.05, 1.0), vec2(W * 0.5 + 0.2, 1.06));
            float plate = box(uv, vec2(W * 0.5 - 0.26, 0.40), vec2(W * 0.5 + 0.26, 0.56));
            color = mix(color, school ? DARK : color * 0.7, bumper);
            color = mix(color, color * 0.5, doorSeam);
            color = mix(color, DARK, clamp(handle, 0.0, 1.0));
            color = mix(color, TAILLIGHT, light);
            color = mix(color, vec3(0.85, 0.85, 0.80), plate);
            color = mix(color, GLASS, clamp(doorWin, 0.0, 1.0));
            if (light > 0.5 || doorWin > 0.5) gloss = 0.7;
        } else if (role == ROLE_TOP && mat == MAT_BUS) {
            float hatch = box(uv, vec2(W * 0.3 - 0.4, H * 0.5 - 0.4), vec2(W * 0.3 + 0.4, H * 0.5 + 0.4))
                        + box(uv, vec2(W * 0.7 - 0.4, H * 0.5 - 0.4), vec2(W * 0.7 + 0.4, H * 0.5 + 0.4));
            color = mix(color, color * 0.8, clamp(hatch, 0.0, 1.0));
        }
    } else if (mat == MAT_CONTAINER) {
        // corrugated steel: vertical grooves every 25 cm (across the top), corner castings, and a
        // pair of doors with lock rods and hinges on the back end; rust where the paint has gone
        float groove = abs(fract(uv.x / 0.25) - 0.5) * 2.0;
        float shade = mix(1.0, 0.72, smoothstep(0.35, 0.85, groove));
        float corner = (step(uv.x, 0.18) + step(W - 0.18, uv.x)) * (step(uv.y, 0.25) + step(H - 0.25, uv.y));
        float rail = hline(uv.y, 0.10, 0.06) + hline(uv.y, H - 0.08, 0.05);
        float rust = smoothstep(0.62, 0.85, fbm(uv * 1.6 + sd)) * (0.5 + 0.5 * (1.0 - smoothstep(0.0, 1.2, uv.y)));
        if (role == ROLE_BACK) {
            shade = mix(1.0, 0.85, smoothstep(0.35, 0.85, groove) * 0.5);
            float seam = vline(uv.x, W * 0.5, 0.04);
            float rods = vline(uv.x, W * 0.22, 0.05) + vline(uv.x, W * 0.36, 0.05) + vline(uv.x, W * 0.64, 0.05) + vline(uv.x, W * 0.78, 0.05);
            rods *= step(0.25, uv.y) * step(uv.y, H - 0.25);
            float handles = (box(uv, vec2(W * 0.22 - 0.03, 1.05), vec2(W * 0.36 + 0.03, 1.12))
                           + box(uv, vec2(W * 0.64 - 0.03, 1.05), vec2(W * 0.78 + 0.03, 1.12)));
            float hinges = (vline(uv.x, 0.12, 0.10) + vline(uv.x, W - 0.12, 0.10)) * step(0.5, fract(uv.y / 0.6 + 0.25)) * step(0.3, uv.y) * step(uv.y, H - 0.3);
            color = color * shade;
            color = mix(color, vec3(0.40, 0.22, 0.12), rust * 0.7);
            color = mix(color, DARK, seam);
            color = mix(color, CHROME * 0.8, clamp(rods, 0.0, 1.0));
            color = mix(color, CHROME, clamp(handles, 0.0, 1.0));
            color = mix(color, color * 0.6, clamp(hinges, 0.0, 1.0));
            gloss = 0.15;
        } else {
            color = color * shade;
            color = mix(color, vec3(0.40, 0.22, 0.12), rust * 0.7);
            color = mix(color, color * 0.7, clamp(rail, 0.0, 1.0) * float(role == ROLE_SIDE || role == ROLE_FRONT));
            gloss = 0.1;
        }
        color = mix(color, DARK * 1.3, clamp(corner, 0.0, 1.0) * float(role != ROLE_TOP && role != ROLE_BOTTOM));
        vec3 across = (role == ROLE_TOP || role == ROLE_BOTTOM) ? vec3(1.0, 0.0, 0.0) : normalize(cross(norm, vec3(0.0, 1.0, 0.0)));
        float slope = (fract(uv.x / 0.25) < 0.5 ? 1.0 : -1.0) * 0.35;
        if (role != ROLE_BACK) n = normalize(norm + across * slope);
    } else if (mat == MAT_WHEEL_SIDE) {
        float r = length(uv);
        float rim = step(r, 0.62);
        float hub = step(r, 0.16);
        float ang = atan(uv.y, uv.x);
        float lug = 0.0;
        for (int i = 0; i < 5; i++) {
            float a = 6.2831853 * float(i) / 5.0;
            lug += disc(uv, vec2(cos(a), sin(a)) * 0.36, 0.06);
        }
        float spoke = step(0.55, fract(ang * 5.0 / 6.2831853)) * step(0.2, r) * step(r, 0.55);
        color = mix(base, CHROME * 0.9, rim);
        color = mix(color, CHROME * 0.6, spoke * rim);
        color = mix(color, CHROME * 0.75, hub);
        color = mix(color, DARK, clamp(lug, 0.0, 1.0));
        float sidewall = step(0.62, r) * step(r, 0.70);
        color = mix(color, base * 1.25, sidewall);
        gloss = rim * 0.5;
    } else if (mat == MAT_WHEEL_TREAD) {
        float tread = step(0.5, fract(uv.x / 0.06));
        color = base * mix(0.85, 1.05, tread);
    } else if (mat == MAT_WALL) {
        int style = variant - 3 * (variant / 3);
        int decor = variant / 3;
        if (style == 1) {
            // brick: running bond 0.24 x 0.075 with mortar, each brick its own shade, the odd
            // broken or darker one
            float row = floor(uv.y / 0.075);
            float ux = uv.x + (mod(row, 2.0) < 0.5 ? 0.0 : 0.12);
            vec2 cell = vec2(floor(ux / 0.24), row);
            vec2 f = vec2(fract(ux / 0.24), fract(uv.y / 0.075));
            float mortar = step(f.x, 0.05) + step(f.y, 0.14);
            float tone = 0.78 + 0.44 * hash21(cell + sd * 0.001);
            float dark = step(0.93, hash21(cell * 1.7 + 0.3));
            float chipped = step(0.96, hash21(cell * 2.3 + 0.7)) * step(0.5, f.x);
            vec3 brick = base * tone;
            brick = mix(brick, brick * 0.55, dark);
            brick = mix(brick, vec3(0.62, 0.60, 0.56), chipped);
            color = mix(brick, vec3(0.66, 0.64, 0.60), clamp(mortar, 0.0, 1.0));
        } else if (style == 2) {
            // cladding panels 1.2 x 1.0 with seams, a few replaced (lighter) panels and rust at
            // the seams
            vec2 cell = floor(uv / vec2(1.2, 1.0));
            vec2 f = fract(uv / vec2(1.2, 1.0));
            float seam = step(f.x, 0.025) + step(f.y, 0.03);
            float tone = 0.9 + 0.2 * hash21(cell + sd * 0.001);
            float replaced = step(0.9, hash21(cell * 3.1 + 1.1));
            vec3 panel = base * tone;
            panel = mix(panel, panel * 1.25 + 0.05, replaced);
            float rustStreak = smoothstep(0.6, 0.9, vnoise(vec2(uv.x * 9.0, uv.y * 0.7) + sd)) * (1.0 - smoothstep(0.0, 0.5, f.y));
            panel = mix(panel, vec3(0.42, 0.24, 0.12), rustStreak * 0.6);
            color = mix(panel, panel * 0.45, clamp(seam, 0.0, 1.0));
            gloss = 0.15;
        } else {
            // plaster / painted concrete: mottled, with patches of older paint showing through
            float mottle = 0.92 + 0.16 * fbm(uv * 2.5 + sd);
            float patch = smoothstep(0.58, 0.7, fbm(uv * 0.6 + sd * 1.7));
            color = base * mottle;
            color = mix(color, base * vec3(0.9, 0.85, 0.8), patch * 0.5);
        }
        float amount = (decor == 3) ? 1.0 : 0.45;
        color = weather(color, uv, W, H, amount);
        if (decor == 1 && role != ROLE_TOP) color = graffiti(color, uv, W, H, variant);
        if (decor == 2 && role != ROLE_TOP) color = mural(color, uv, W, H, variant);
    } else if (mat == MAT_WINDOW_WALL) {
        // glass behind the window recesses of a facade: floors are 3 m, the pane spans sill 1.0
        // to lintel 2.2, a mullion at mid height; some panes are boarded up or broken
        float fl = floor(uv.y / 3.0);
        float vy = uv.y - fl * 3.0;
        vec2 cell = vec2(floor(uv.x / 2.2), fl);
        float h = hash21(cell + floor(sd * 0.01));
        float mullion = hline(vy, 1.6, 0.05) + vline(fract(uv.x / 2.2) * 2.2, 1.1, 0.05);
        vec3 glass = GLASS * (0.8 + 0.4 * h);
        glass += vec3(0.08) * smoothstep(1.0, 2.2, vy);          // a little sky in the upper half
        // the odd lit room
        float lit = step(0.8, h);
        glass = mix(glass, vec3(0.85, 0.75, 0.5), lit * 0.6);
        color = glass;
        if (h < 0.08) {
            // boarded up
            color = vec3(0.50, 0.40, 0.28) * (0.9 + 0.2 * fract(uv.y * 5.0));
            color = mix(color, color * 0.6, hline(fract(vy / 0.25), 0.0, 0.08));
        } else if (h < 0.2) {
            float cr = cracks(vec2(fract(uv.x / 2.2) * 2.2, vy), vec2(0.9 + 0.6 * h, 1.5 + 0.5 * h), 0.7);
            color = mix(color, vec3(0.92, 0.94, 0.96), cr);
            float hole = 1.0 - smoothstep(0.12, 0.22, length(vec2(fract(uv.x / 2.2) * 2.2, vy) - vec2(0.9 + 0.6 * h, 1.5 + 0.5 * h)));
            color = mix(color, vec3(0.03, 0.03, 0.04), hole);
        }
        color = mix(color, vec3(0.25, 0.25, 0.27), clamp(mullion, 0.0, 1.0));
        gloss = 0.5;
    } else if (mat == MAT_BILLBOARD) {
        if (role == ROLE_FRONT || role == ROLE_BACK) {
            vec2 p = uv / vec2(W, H);
            vec3 ad = vec3(0.9);
            int k = variant;
            if (k == 0) {            // bold logo: a big disc on a two-tone field
                ad = mix(palette(1), palette(8), step(0.5, p.x));
                ad = mix(ad, palette(2), disc(p * vec2(2.0, 1.0), vec2(1.0, 0.5), 0.36));
                ad = mix(ad, DARK, box(p, vec2(0.62, 0.3), vec2(0.94, 0.42)));
            } else if (k == 1) {     // diagonal stripes and a product bar
                float d = step(0.5, fract((p.x * 2.0 + p.y) * 3.0));
                ad = mix(palette(0), palette(2), d);
                ad = mix(ad, palette(8), box(p, vec2(0.1, 0.15), vec2(0.55, 0.45)));
                ad = mix(ad, DARK, box(p, vec2(0.14, 0.22), vec2(0.5, 0.26)) + box(p, vec2(0.14, 0.32), vec2(0.4, 0.36)));
            } else if (k == 2) {     // sunset with a skyline
                ad = mix(palette(5), palette(6), p.y);
                ad = mix(ad, palette(2), disc(p * vec2(2.0, 1.0), vec2(1.3, 0.55), 0.2));
                float sky = step(p.y, 0.25 + 0.15 * step(0.5, fract(p.x * 7.0)) + 0.1 * step(0.5, fract(p.x * 3.0)));
                ad = mix(ad, DARK, sky);
            } else if (k == 3) {     // checkerboard promo
                float c = step(0.5, fract(p.x * 6.0)) == step(0.5, fract(p.y * 3.0)) ? 1.0 : 0.0;
                ad = mix(palette(8), palette(9), c);
                ad = mix(ad, palette(0), box(p, vec2(0.25, 0.35), vec2(0.75, 0.65)));
            } else if (k == 4) {     // target rings
                float r = length((p - 0.5) * vec2(2.0, 1.0));
                ad = mix(palette(8), palette(0), step(0.5, fract(r * 5.0)));
                ad = mix(ad, palette(1), box(p, vec2(0.05, 0.8), vec2(0.95, 0.95)));
            } else if (k == 5) {     // dark poster with text blocks
                ad = vec3(0.1, 0.1, 0.12);
                ad = mix(ad, palette(8), box(p, vec2(0.08, 0.55), vec2(0.7, 0.7)) + box(p, vec2(0.08, 0.4), vec2(0.5, 0.48)));
                ad = mix(ad, palette(2), box(p, vec2(0.08, 0.15), vec2(0.3, 0.3)));
                ad = mix(ad, palette(7), disc(p * vec2(2.0, 1.0), vec2(1.6, 0.5), 0.28));
            } else if (k == 6) {     // a big arrow
                ad = palette(3);
                float arrow = step(abs(p.y - 0.5), 0.12) * step(0.1, p.x) * step(p.x, 0.6)
                            + step(abs(p.y - 0.5), (0.9 - p.x) * 0.9) * step(0.6, p.x) * step(p.x, 0.9);
                ad = mix(ad, palette(8), clamp(arrow, 0.0, 1.0));
                ad = mix(ad, DARK, box(p, vec2(0.1, 0.1), vec2(0.55, 0.2)));
            } else if (k == 7) {     // polka dots
                vec2 g = fract(p * vec2(8.0, 4.0)) - 0.5;
                ad = mix(palette(4), palette(8), step(length(g), 0.3));
                ad = mix(ad, DARK, box(p, vec2(0.3, 0.42), vec2(0.7, 0.58)));
            } else if (k == 8) {     // colour bars
                ad = palette(int(floor(p.x * 8.0)));
                ad = mix(ad, DARK, box(p, vec2(0.0, 0.0), vec2(1.0, 0.18)));
            } else {                 // an old torn poster
                ad = mix(palette(2), palette(0), step(0.5, p.y));
                ad = mix(ad, palette(8), box(p, vec2(0.2, 0.3), vec2(0.8, 0.7)));
                float torn = step(0.58, fbm(uv * 1.5 + sd));
                ad = mix(ad, vec3(0.78, 0.76, 0.70), torn);
            }
            // sun-faded and grimy
            float fade = 0.85 + 0.15 * fbm(uv * 0.8 + sd);
            color = ad * fade;
            color = weather(color, uv, W, H, 0.35);
        } else {
            color = vec3(0.22, 0.22, 0.24);
        }
    } else if (mat == MAT_LIGHT) {
        color = base;
        emissive = 1.0;
    } else if (mat == MAT_BARK) {
        // bark: long vertical fissures, ridges and knots
        vec2 bp = vec2(uv.x * 9.0, uv.y * 1.2) + sd * 0.2;
        float ridge = fbm(bp);
        float fissure = smoothstep(0.42, 0.55, ridge);
        float knot = 1.0 - smoothstep(0.12, 0.2, length(fract(uv * vec2(1.0, 0.5) + sd * 0.01) - 0.5));
        color = base * (0.6 + 0.6 * fissure);
        color = mix(color, base * 0.5, knot * step(0.6, hash21(floor(uv * vec2(1.0, 0.5) + sd * 0.01))));
        vec3 across = normalize(cross(norm, vec3(0.0, 1.0, 0.0)));
        n = normalize(norm + across * (fissure - 0.5) * 0.8);
    } else if (mat == MAT_LEAVES) {
        // foliage: clusters of leaves, light and dark greens, gaps between them and a bit of
        // yellow on the sunny side
        vec2 lp = uv * 7.0 + sd * 1.3;
        float cl = fbm(lp);
        float cl2 = vnoise(lp * 3.0 + 2.0);
        float gap = smoothstep(0.28, 0.36, cl);
        vec3 light = base * vec3(1.25, 1.3, 0.9);
        vec3 dark = base * 0.45;
        color = mix(dark, base, gap);
        color = mix(color, light, smoothstep(0.55, 0.75, cl) * (0.5 + 0.5 * cl2));
        color = mix(color, vec3(0.62, 0.60, 0.22), smoothstep(0.78, 0.92, cl2) * 0.5);
        n = normalize(norm + vec3(cl2 - 0.5, cl - 0.5, cl2 - cl) * 0.9);
    } else if (mat == MAT_ASPHALT) {
        // worn asphalt: grain, patches, cracks and lane paint
        vec2 p = v_FragPos.xz;
        float grain = 0.9 + 0.2 * vnoise(p * 7.0);
        float patch = smoothstep(0.55, 0.7, fbm(p * 0.15 + 3.1));
        float crack = step(abs(fbm(p * 0.9) - 0.5), 0.004) * step(0.58, fbm(p * 0.07 + 9.0));
        float pothole = 1.0 - smoothstep(0.2, 0.5, length(fract(p * 0.05 + 0.3) - 0.5) - 0.25 * fbm(p * 0.4)) ;
        pothole *= step(0.93, hash21(floor(p * 0.05 + 0.3)));
        color = base * grain;
        color = mix(color, base * 1.25, patch * 0.6);
        color = mix(color, base * 0.6, crack);
        color = mix(color, base * 0.55, pothole);
    } else if (mat == MAT_PAVEMENT) {
        // paving slabs 1 m with joints, settled and cracked here and there, chewing gum stains
        if (role == ROLE_TOP) {
            vec2 p = v_FragPos.xz;
            vec2 f = fract(p);
            vec2 cell = floor(p);
            float joint = step(f.x, 0.03) + step(f.y, 0.03);
            float tone = 0.88 + 0.2 * hash21(cell);
            float crack = step(abs(fbm(p * 1.6 + cell * 0.01) - 0.5), 0.005) * step(0.8, hash21(cell * 1.3));
            float gum = step(0.97, hash21(cell * 2.1 + 0.5)) * (1.0 - smoothstep(0.05, 0.08, length(f - vec2(hash21(cell), hash21(cell + 1.0)))));
            color = base * tone;
            color = mix(color, base * 0.6, clamp(joint, 0.0, 1.0));
            color = mix(color, base * 0.5, crack);
            color = mix(color, base * 0.35, gum);
            color = weather(color, vec2(p.x, 0.0), 100.0, 1.0, 0.2);
        } else {
            color = base * 0.85;
        }
    } else if (mat == MAT_TRASH) {
        // bin bags: glossy black plastic with creases
        float crease = fbm(uv * 12.0 + sd);
        color = base * (0.6 + 0.9 * crease);
        gloss = 0.6;
        n = normalize(norm + vec3(crease - 0.5) * 0.8);
    } else if (mat == MAT_CHASSIS) {
        color = base;
    }

    // ---------------------------------------------------------------- lighting
    float ambientStrength = 0.35;
    vec3 ambient = ambientStrength * vec3(0.85, 0.90, 0.95);
    float diff = max(dot(n, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 0.98, 0.92);

    // Procedural grid pattern on plain horizontal surfaces (the training park floor, rooftops)
    float grid = 1.0;
    if (mat == MAT_PLAIN && abs(norm.y) > 0.7) {
        vec2 coord = v_FragPos.xz;
        vec2 gridVec = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
        float line = min(gridVec.x, gridVec.y);
        grid = 0.85 + 0.15 * min(line, 1.0);
    }

    vec3 result = (ambient + diffuse) * color * grid;
    if (gloss > 0.0) {
        vec3 halfDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(n, halfDir), 0.0), 24.0);
        result += gloss * spec * vec3(1.0, 0.98, 0.95);
    }
    result = mix(result, color * 1.15, emissive);
    FragColor = vec4(result, 1.0);
}
