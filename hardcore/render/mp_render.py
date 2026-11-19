"""Multiplayer visuals: other players (and the bots), emblems, flags and bases, the HUD.

* Avatars are built from boxes posed by `avatar_parts()` (pure, testable): a running
  gait that grows with speed, a tuck in the air, arms overhead facing the wall while
  hanging or climbing, flips and rolls (the same whole-body rotation the local player
  sees, parkour/acrobatics.py), a flag on the back of whoever carries one.  Shirts are
  the team's colour in a teams game.  A name tag floats above each head.
* Emblems are spinning, gently bobbing diamonds with a pulsing blue aura (additive
  camera-facing glow), drawn depth-tested so walls hide them.
* Capture the flag: a flag on a pole at each base inside a glowing ring, and a
  column of light over it you can see across the city; the free-for-all flag and its
  drop zone the same in gold.
* The HUD: the game and the score (team scores in a teams game), the time left, the
  scoreboard, a banner while you carry a flag, and markers on the screen edge
  pointing at the flags / your base.
"""
import math

import numpy as np

import config
from game import teams as T
from math3d.transform import translate, rotate_x, rotate_y, rotate_z, scale, ortho
from parkour import acrobatics

try:
    from OpenGL.GL import *
except ImportError:        # pragma: no cover
    pass

FEET_TO_CENTER = config.PLAYER_HALF_HEIGHT
SHIRTS = [(0.85, 0.25, 0.25), (0.25, 0.55, 0.9), (0.3, 0.75, 0.35), (0.85, 0.75, 0.25), (0.65, 0.35, 0.8),
          (0.25, 0.75, 0.75), (0.9, 0.5, 0.2), (0.9, 0.45, 0.65), (0.55, 0.6, 0.25), (0.45, 0.45, 0.85),
          (0.8, 0.8, 0.8), (0.5, 0.3, 0.2), (0.2, 0.4, 0.3), (0.7, 0.2, 0.45), (0.35, 0.65, 0.55)]
BOT_SHIRT = (1.0, 0.55, 0.1)
PANTS = (0.12, 0.13, 0.16)
SKIN = (0.78, 0.58, 0.46)
EMBLEM_COLOR = (0.35, 0.72, 1.0)
ZOMBIE_SKIN = (0.42, 0.62, 0.30)
ZOMBIE_PANTS = (0.17, 0.19, 0.13)
ZOMBIE_EYES = (1.0, 0.85, 0.25)
LIVING_GLOW = (1.0, 0.86, 0.45)
NEUTRAL_FLAG = (1.0, 0.82, 0.25)
POLE = (0.82, 0.84, 0.88)
FLAG_COLORS = {T.RED: T.SHIRTS[T.RED], T.BLUE: T.SHIRTS[T.BLUE], None: NEUTRAL_FLAG}
HANG_STATES = ("LEDGE_HANG", "SHIMMY", "CORNER_TURN")
UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)
WAVE_OFFSET = {T.RED: 0.0, T.BLUE: 2.1, "neutral": 4.2}


def shirt_color(pid, bot=False, team=None):
    if T.valid(team):
        base = np.array(T.SHIRTS[team], dtype=np.float32)
        shade = 0.82 + 0.18 * ((int(pid) * 37) % 7) / 6.0          # teammates differ a little
        return tuple(float(c) for c in np.clip(base * shade, 0.0, 1.0))
    return BOT_SHIRT if bot else SHIRTS[(int(pid) - 1) % len(SHIRTS)]


def flag_color(key):
    return FLAG_COLORS.get(key if key in T.TEAMS else None, NEUTRAL_FLAG)


def facing_yaw(pose):
    """The way the body faces: the wall while on it, else where the player looks."""
    n = pose.get("n")
    if pose.get("st") in HANG_STATES + ("CLIMB_UP",) and n is not None:
        return math.degrees(math.atan2(n[0], n[1]))
    return float(pose.get("yaw", 0.0))


def zombie_shirt(shirt):
    """A zombie's clothes: the shirt faded, dirtied toward grey-green."""
    c = np.asarray(shirt, dtype=np.float32)
    grey = float(c.mean())
    rag = np.array([0.30, 0.34, 0.25], dtype=np.float32)
    return tuple(float(x) for x in np.clip((c * 0.3 + grey * 0.2) * 0.6 + rag * 0.6, 0.0, 1.0))


def avatar_parts(pose, t, pid=1, bot=False, team=None, carrying=None, zombie=False):
    """Boxes making up a posed avatar: [(center xyz, size xyz, 3x3 rotation, color, emissive)].
    `carrying`: the key of the flag on this player's back (or None).  `zombie`: green skin, rags,
    arms out in front and a lurching, limping shamble."""
    st = pose.get("st", "GROUND")
    spd = float(pose.get("spd", 0.0))
    p = np.asarray(pose["p"], dtype=np.float32)
    feet = p - np.array([0.0, FEET_TO_CENTER, 0.0], dtype=np.float32)
    yaw = math.radians(facing_yaw(pose))
    R = rotate_y(yaw)[:3, :3]
    shirt = shirt_color(pid, bot, team)
    skin, pants = SKIN, PANTS
    if zombie:
        shirt, skin, pants = zombie_shirt(shirt), ZOMBIE_SKIN, ZOMBIE_PANTS
    ac = pose.get("ac")

    run = min(1.0, spd / config.SPRINT_SPEED)
    phase = t * (5.0 + 6.0 * run) + pid * 1.3
    lean = 0.0
    sway = 0.0                                           # sideways lurch (a zombie's shamble)
    head_tilt = 0.0
    if zombie:
        head_tilt = math.radians(16.0 + 4.0 * math.sin(t * 1.3 + pid))
    if ac is not None:
        # tucked: knees to the chest, hands on the shins (the whole body turns below)
        sh, el, th, kn = (math.radians(55), math.radians(55)), (1.5, 1.5), (math.radians(115), math.radians(115)), (2.35, 2.35)
    elif st in HANG_STATES:
        sh, el, th, kn = (math.radians(168), math.radians(168)), (0.2, 0.2), (0.15, -0.05), (0.35, 0.25)
    elif st == "CLIMB_UP":
        k = 0.5 + 0.5 * math.sin(t * 6.0)
        sh, el = (math.radians(150), math.radians(130)), (0.6, 0.9)
        th, kn = (math.radians(20 + 60 * k), math.radians(10)), (math.radians(30 + 70 * k), 0.3)
    elif st == "MANTLE_VAULT":
        sh, el, th, kn = (math.radians(70), math.radians(70)), (0.3, 0.3), (math.radians(80), 0.2), (1.2, 0.4)
    elif st in ("AIRBORNE", "DYNO_LAUNCH", "WALL_EJECT"):
        sh, el, th, kn = (math.radians(110), math.radians(95)), (0.5, 0.5), (math.radians(45), math.radians(30)), (1.1, 0.9)
    elif zombie:
        # the shamble: a slow, uneven gait - one leg drags - the body lurching side to side and
        # forward, arms held out in front reaching for you
        phase = t * (3.2 + 4.0 * run) + pid * 1.3
        s = math.sin(phase)
        amp = math.radians(10 + 30 * run) if spd > 0.3 else math.radians(3)
        th = (amp * s, -amp * s * 0.55)
        kn = (max(0.0, -s) * amp * 1.3, max(0.0, s) * amp * 0.5 + 0.12)
        reach = math.radians(62 + 6 * math.sin(phase * 0.5 + 1.0))      # (+ the forward lean: level arms)
        sh = (reach + 0.08 * s, reach - 0.08 * s + math.radians(6))
        el = (0.12, 0.22)
        lean = math.radians(14 + 5 * abs(s)) + math.radians(8) * run
        sway = math.radians(9) * s
    else:
        amp = math.radians(8 + 40 * run) if spd > 0.3 else 0.0
        s = math.sin(phase)
        th = (amp * s, -amp * s)
        kn = (max(0.0, -s) * amp * 1.6, max(0.0, s) * amp * 1.6)
        sh = (-th[0] * 0.9, -th[1] * 0.9)
        el = (0.3 + 1.0 * run, 0.3 + 1.0 * run)
        lean = math.radians(12) * run
    parts = []

    def box(local_center, size, local_rot, color, emissive=0.0):
        c = feet + R @ np.asarray(local_center, dtype=np.float32)
        parts.append((c, np.asarray(size, dtype=np.float32), R @ local_rot, color, emissive))

    Lean = rotate_z(sway)[:3, :3] @ rotate_x(-lean)[:3, :3]        # (forward is -z: a lean tips the top forward)
    hip_y = 0.92
    box((0.0, hip_y + 0.05, 0.0), (0.36, 0.2, 0.22), Lean, pants)
    torso_c = Lean @ np.array([0.0, 0.33, 0.0]) + np.array([0.0, hip_y, 0.0])
    box(torso_c, (0.42, 0.56, 0.24), Lean, shirt)
    head_c = Lean @ np.array([0.0, 0.74, 0.0]) + np.array([0.0, hip_y, 0.0])
    Head = Lean @ rotate_z(head_tilt)[:3, :3]
    box(head_c, (0.22, 0.25, 0.23), Head, skin)
    if zombie:                                           # dull glowing eyes
        for ex in (-0.05, 0.05):
            box(head_c + Head @ np.array([ex, 0.03, -0.118]), (0.045, 0.03, 0.015), Head, ZOMBIE_EYES, 0.9)
    elif bot:                                            # a glowing visor so a bot reads as a bot
        box(head_c + Lean @ np.array([0.0, 0.02, -0.115]), (0.2, 0.06, 0.02), Lean, (0.3, 0.9, 1.0), 1.0)
    if carrying is not None:
        # the flag rides on the back: a short pole over the shoulder, the cloth streaming behind
        col = flag_color(carrying)
        back = Lean @ np.array([0.1, 0.55, 0.16]) + np.array([0.0, hip_y, 0.0])
        box(back, (0.04, 1.3, 0.04), Lean, POLE)
        box(back + Lean @ np.array([0.0, 0.42, 0.26]), (0.03, 0.34, 0.5), Lean, col, 0.35)

    def limb(joint, angle, bend, l1, l2, w1, w2, c1, c2, knee=False):
        # bone 1 swings about x from pointing down; bone 2 bends (knees backward, elbows forward)
        r1 = rotate_x(angle)[:3, :3]
        d1 = r1 @ np.array([0.0, -1.0, 0.0])
        box(np.asarray(joint) + d1 * l1 * 0.5, (w1, l1, w1), r1, c1)
        j2 = np.asarray(joint) + d1 * l1
        r2 = rotate_x(angle - bend if knee else angle + bend)[:3, :3]
        d2 = r2 @ np.array([0.0, -1.0, 0.0])
        box(j2 + d2 * l2 * 0.5, (w2, l2, w2), r2, c2)

    for side, i in ((-1.0, 0), (1.0, 1)):
        limb((0.11 * side, hip_y, 0.0), th[i], kn[i], 0.46, 0.46, 0.15, 0.13, pants, pants, knee=True)
        shoulder = Lean @ np.array([0.25 * side, 0.52, 0.0]) + np.array([0.0, hip_y, 0.0])
        limb(shoulder, sh[i] + lean, el[i], 0.3, 0.3, 0.11, 0.1, shirt, skin)

    if ac is not None:
        # flips and rolls: the whole body turns about its centre of mass (as in first person)
        try:
            kind, angle, crouch, hx, hz = ac
            Tm = acrobatics.transform_for(kind, float(angle), float(crouch), (float(hx), 0.0, float(hz)), p)
            Rt = Tm[:3, :3]
            parts = [(acrobatics.apply(Tm, c), size, Rt @ rot, col, em) for c, size, rot, col, em in parts]
        except (TypeError, ValueError):
            pass
    return parts


def _cube_mesh():
    v, idx = [], []
    faces = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
             ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
             ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]
    for n, a, b in faces:
        n, a, b = (np.array(x, dtype=np.float32) for x in (n, a, b))
        c = n * 0.5
        base = len(v)
        for sa, sb in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            v.append(list(c + a * sa + b * sb) + list(n))
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(v, dtype=np.float32), np.array(idx, dtype=np.uint32)


def _octahedron():
    pts = [(0, 1, 0), (1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1), (0, -1, 0)]
    tris = [(0, 1, 4), (0, 4, 3), (0, 3, 2), (0, 2, 1), (5, 4, 1), (5, 3, 4), (5, 2, 3), (5, 1, 2)]
    v, idx = [], []
    for a, b, c in tris:
        pa, pb, pc = (np.array(pts[i], dtype=np.float32) for i in (a, b, c))
        n = np.cross(pb - pa, pc - pa)
        n = n / np.linalg.norm(n)
        base = len(v)
        for q in (pa, pb, pc):
            v.append(list(q) + list(n))
        idx += [base, base + 1, base + 2]
    return np.array(v, dtype=np.float32), np.array(idx, dtype=np.uint32)


def emblem_transform(pos, t, eid=0):
    """Model matrix of an emblem core: spinning about Y and bobbing gently."""
    bob = 0.12 * math.sin(t * 2.4 + eid * 1.7)
    return (translate(float(pos[0]), float(pos[1]) + bob, float(pos[2])) @ rotate_y(t * 1.8 + eid) @
            scale(0.24, 0.36, 0.24))


def flag_stand_parts(ground, key, t):
    """A flag on its stand: base plate, pole, waving cloth.  [(center, size, rot, color, emissive)]"""
    g = np.asarray(ground, dtype=np.float32)
    col = flag_color(key)
    out = [(g + UP * 0.04, np.array([0.9, 0.08, 0.9], np.float32), np.eye(3, dtype=np.float32), (0.25, 0.26, 0.3), 0.0),
           (g + UP * 1.35, np.array([0.07, 2.7, 0.07], np.float32), np.eye(3, dtype=np.float32), POLE, 0.0)]
    wave = rotate_y(0.35 * math.sin(t * 2.6))[:3, :3]
    cloth_c = g + UP * 2.35 + wave @ np.array([0.48, 0.0, 0.0], np.float32)
    out.append((cloth_c, np.array([0.9, 0.56, 0.03], np.float32), wave, col, 0.45))
    return out


def zone_ring_parts(center, radius, key, segments=28):
    """A glowing ring on the ground marking a base / the drop zone."""
    c = np.asarray(center, dtype=np.float32)
    col = flag_color(key)
    seg_len = 2.0 * math.pi * radius / segments * 1.02
    out = []
    for i in range(segments):
        a = (i + 0.5) * 2.0 * math.pi / segments
        pos = c + np.array([math.cos(a) * radius, 0.05, math.sin(a) * radius], np.float32)
        rot = rotate_y(-a)[:3, :3]                    # box x along the ring's tangent
        out.append((pos, np.array([0.12, 0.06, seg_len], np.float32), rot, col, 0.9))
    return out


class MultiplayerRenderer:
    """Draws the remote players, emblems, flags and the multiplayer HUD (needs a GL context)."""

    def __init__(self, menu):
        from render.shader import Shader
        from render.buffer import MeshBuffer
        self.menu = menu
        self.shader = Shader("render/shaders/avatar.vert", "render/shaders/avatar.frag")
        self.aura = Shader("render/shaders/aura.vert", "render/shaders/aura.frag")
        layout = [(0, 3, GL_FLOAT, False), (1, 3, GL_FLOAT, False)]
        self.cube = MeshBuffer(*_cube_mesh(), layout)
        self.gem = MeshBuffer(*_octahedron(), layout)
        quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
        self.quad = MeshBuffer(quad, np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32), [(0, 2, GL_FLOAT, False)])
        self.labels = {}
        self.time = 0.0
        # what to draw this frame (set by the game loop)
        self.players = []          # [(RemotePlayer, pose)]
        self.emblems = []
        self.hud = None            # dict for the HUD (see game/session.py hud_state)
        self.cam_pos = np.zeros(3, dtype=np.float32)

    def set_frame(self, players, emblems, hud):
        self.players, self.emblems, self.hud = players, emblems, hud

    def clear(self):
        self.players, self.emblems, self.hud = [], [], None

    # ------------------------------------------------------------------ helpers
    def _objectives(self):
        hud = self.hud or {}
        obj = hud.get("objectives") or {}
        return obj.get("flags", []), obj.get("zones", [])

    def _carriers(self):
        """{pid: flag key} for flags being carried."""
        flags, _ = self._objectives()
        return {f.get("carrier"): f.get("key") for f in flags if f.get("carrier") is not None}

    def _living(self):
        """Zombie tag: the living player's id (None in other games)."""
        z = (self.hud or {}).get("zombie")
        return z.get("living") if z else None

    def is_zombie(self, pid):
        hud = self.hud or {}
        return hud.get("mode") == "zombie" and hud.get("zombie") is not None and pid != self._living()

    def _draw_boxes(self, sh, parts):
        for c, size, rot, color, emissive in parts:
            m = np.eye(4, dtype=np.float32)
            m[:3, :3] = rot * size[None, :]            # rotate a box of this size ...
            m[:3, 3] = c                               # ... placed here
            sh.set_mat4("u_Model", m)
            sh.set_vec3("u_Color", color)
            sh.set_float("u_Emissive", emissive)
            self.cube.draw()

    # ------------------------------------------------------------------ 3D
    def draw_world(self, view, proj, cam_pos, light_dir, dt):
        self.time += dt
        self.cam_pos = np.asarray(cam_pos, dtype=np.float32)
        flags, zones = self._objectives()
        living = self._living()
        if not self.players and not self.emblems and not flags:
            return
        sh = self.shader
        sh.use()
        sh.set_mat4("u_View", view)
        sh.set_mat4("u_Proj", proj)
        sh.set_vec3("u_LightDir", light_dir)
        sh.set_vec3("u_CamPos", cam_pos)
        fwd = -np.linalg.inv(view)[:3, 2]
        carriers = self._carriers()
        for rp, pose in self.players:
            rel = np.asarray(pose["p"], dtype=np.float32) - np.asarray(cam_pos, dtype=np.float32)
            dist = float(np.linalg.norm(rel))
            if dist > 150.0 or (dist > 3.0 and float(np.dot(rel, fwd)) < -1.5):
                continue                                   # far away or behind the camera
            self._draw_boxes(sh, avatar_parts(pose, self.time, rp.pid, rp.bot, getattr(rp, "team", None),
                                              carriers.get(rp.pid), zombie=self.is_zombie(rp.pid)))
        for e in self.emblems:
            sh.set_mat4("u_Model", emblem_transform(e["pos"], self.time, e["id"]))
            sh.set_vec3("u_Color", EMBLEM_COLOR)
            sh.set_float("u_Emissive", 0.85)
            self.gem.draw()
        # capture the flag: flags on their stands (a carried flag rides on its carrier), base rings
        me = (self.hud or {}).get("me")
        remote_ids = {rp.pid for rp, _ in self.players}
        for f in flags:
            if f.get("carrier") is None:
                self._draw_boxes(sh, flag_stand_parts(f["home"], f["key"], self.time + WAVE_OFFSET.get(f["key"], 0.0)))
            elif f["carrier"] != me and f["carrier"] not in remote_ids:
                self._draw_boxes(sh, flag_stand_parts(np.asarray(f["pos"]) - UP * FEET_TO_CENTER, f["key"], self.time))
            if f.get("team") is not None or f.get("carrier") is None:
                self._draw_boxes(sh, [(np.asarray(f["home"], np.float32) + UP * 0.03,
                                       np.array([0.5, 0.04, 0.5], np.float32), np.eye(3, dtype=np.float32),
                                       flag_color(f["key"]), 0.6)])
        for z in zones:
            self._draw_boxes(sh, zone_ring_parts(z["pos"], z.get("r", config.MP_BASE_RADIUS), z.get("team")))
        # auras: additive glow, depth-tested (walls hide them) without writing depth
        living_pose = None
        if living is not None:
            living_pose = next((q for rp, q in self.players if rp.pid == living), None)
        if self.emblems or flags or zones or living_pose is not None:
            inv = np.linalg.inv(view)
            right, up = inv[:3, 0], inv[:3, 1]
            a = self.aura
            a.use()
            a.set_mat4("u_View", view)
            a.set_mat4("u_Proj", proj)
            glDepthMask(GL_FALSE)
            glDisable(GL_CULL_FACE)
            glBlendFunc(GL_ONE, GL_ONE)
            a.set_vec3("u_Right", right)
            a.set_vec3("u_Up", up)
            a.set_vec2("u_Stretch", (1.0, 1.0))
            a.set_float("u_Beam", 0.0)
            a.set_vec3("u_Color", EMBLEM_COLOR)
            for e in self.emblems:
                bob = 0.12 * math.sin(self.time * 2.4 + e["id"] * 1.7)
                c = np.array([e["pos"][0], e["pos"][1] + bob, e["pos"][2]], dtype=np.float32)
                pulse = 0.5 + 0.5 * math.sin(self.time * 3.0 + e["id"])
                a.set_vec3("u_Center", c)
                for size, alpha in ((1.5 + 0.15 * pulse, 0.65 + 0.25 * pulse), (0.6, 1.0)):
                    a.set_float("u_Size", size)
                    a.set_float("u_Alpha", alpha)
                    self.quad.draw()
            # light columns over the flags (where they are now) and the zones: seen across the city
            flat_right = np.array([right[0], 0.0, right[2]], dtype=np.float32)
            flat_right /= (np.linalg.norm(flat_right) + 1e-6)
            a.set_vec3("u_Right", flat_right)
            a.set_vec3("u_Up", UP)
            a.set_float("u_Beam", 1.0)
            pulse = 0.5 + 0.5 * math.sin(self.time * 2.0)
            for f in flags:
                if f.get("carrier") == me:
                    continue
                base = np.asarray(f["pos"] if f.get("carrier") is not None else f["home"], dtype=np.float32)
                if f.get("carrier") is not None:
                    base = base - UP * FEET_TO_CENTER
                a.set_vec3("u_Color", flag_color(f["key"]))
                a.set_vec3("u_Center", base + UP * 22.0)
                a.set_float("u_Size", 0.9)
                a.set_vec2("u_Stretch", (1.0, 24.0))
                a.set_float("u_Alpha", 0.55 + 0.2 * pulse)
                self.quad.draw()
            if living_pose is not None:
                # zombie tag: a gold column of light over the living player - the zombies' quarry
                base = np.asarray(living_pose["p"], dtype=np.float32) - UP * FEET_TO_CENTER
                a.set_vec3("u_Color", LIVING_GLOW)
                a.set_vec3("u_Center", base + UP * 16.0)
                a.set_float("u_Size", 0.55)
                a.set_vec2("u_Stretch", (1.0, 30.0))
                a.set_float("u_Alpha", 0.45 + 0.2 * pulse)
                self.quad.draw()
            if not flags or any(f.get("team") is None for f in flags):
                for z in zones:
                    if z.get("team") is not None:
                        continue
                    a.set_vec3("u_Color", flag_color(None))
                    a.set_vec3("u_Center", np.asarray(z["pos"], np.float32) + UP * 15.0)
                    a.set_float("u_Size", 1.6)
                    a.set_vec2("u_Stretch", (1.0, 9.0))
                    a.set_float("u_Alpha", 0.35)
                    self.quad.draw()
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_CULL_FACE)
            glDepthMask(GL_TRUE)

    # ------------------------------------------------------------------- HUD
    def _label(self, text, size, color=(240, 242, 246)):
        from render.menu import TextLabel
        key = (text, size, tuple(color))
        lab = self.labels.get(key)
        if lab is None:
            if len(self.labels) > 200:
                for old in self.labels.values():
                    old.release()
                self.labels.clear()
            lab = self.labels[key] = TextLabel(text, size, tuple(color))
        return lab

    def markers(self, view, proj, screen_w, screen_h):
        """Screen markers for the objectives: [(x, y, text, color, on_screen)] (pure but for the matrices)."""
        hud = self.hud or {}
        flags, zones = self._objectives()
        chase = hud.get("chase") or []
        if not flags and not chase:
            return []
        me, team, teams_game = hud.get("me"), hud.get("team"), hud.get("teams")
        carrying = hud.get("carrying")
        want = []
        for c in chase:
            # zombie tag: the living player (for zombies), the zombies closing in (for the living)
            pos = np.asarray(c["pos"], dtype=np.float32) + UP * 0.9
            want.append((pos, c["label"], LIVING_GLOW if c.get("kind") == "living" else ZOMBIE_SKIN))
        for f in flags:
            if f.get("carrier") == me:
                continue
            pos = np.asarray(f["pos"] if f.get("carrier") is not None else f["home"], dtype=np.float32)
            if f.get("carrier") is None:
                pos = pos + UP * 1.2
            if teams_game and f["key"] == team:
                label = "Your flag" if f.get("carrier") is None else "Your flag - TAKEN"
            elif f["key"] in T.TEAMS:
                label = f"{T.name(f['key'])} flag"
            else:
                label = "Flag" if f.get("carrier") is None else "Flag carrier"
            want.append((pos, label, flag_color(f["key"])))
        if carrying is not None:
            for z in zones:
                if (teams_game and z.get("team") == team) or (not teams_game and z.get("team") is None):
                    want.append((np.asarray(z["pos"], np.float32) + UP * 1.0,
                                 "Your base" if teams_game else "Drop zone", flag_color(z.get("team"))))
        out = []
        vp = proj @ view
        margin = 60.0
        cx, cy = screen_w * 0.5, screen_h * 0.5
        for pos, label, color in want:
            dist = float(np.linalg.norm(pos - self.cam_pos))
            p4 = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
            cam = view @ p4                                  # camera space: x right, y up, -z ahead
            on = False
            if cam[2] < -0.1:
                clip = vp @ p4
                ndc = clip[:2] / clip[3]
                x, y = (ndc[0] + 1.0) * 0.5 * screen_w, (1.0 - ndc[1]) * 0.5 * screen_h
                on = margin * 0.5 <= x <= screen_w - margin * 0.5 and margin * 0.5 <= y <= screen_h - margin * 0.5
                dx, dy = x - cx, y - cy
            else:
                # beside or behind you: point the way along the screen edge (behind: along the bottom)
                dx = float(cam[0])
                dy = max(abs(float(cam[1])), 0.5 * abs(float(cam[2]))) if cam[2] >= -0.1 else -float(cam[1])
            if not on:
                s_edge = min((cx - margin) / (abs(dx) + 1e-6), (cy - margin) / (abs(dy) + 1e-6))
                x, y = cx + dx * s_edge, cy + dy * s_edge
            out.append((float(x), float(y), f"{label}  {int(dist)} m", color, bool(on)))
        return out

    def draw_hud(self, text_shader, view, proj, screen_w, screen_h, ui_scale):
        if self.hud is None:
            return
        m = self.menu
        k = ui_scale
        hud = self.hud
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        text_shader.use()
        text_shader.set_mat4("u_Proj", ortho(0.0, screen_w, screen_h, 0.0, -1.0, 1.0))
        text_shader.set_int("u_Tex", 0)
        teams_game = hud.get("teams")
        # name tags over the heads
        vp = proj @ view
        carriers = self._carriers()
        for rp, pose in self.players:
            head = np.array([pose["p"][0], pose["p"][1] + 1.05, pose["p"][2], 1.0], dtype=np.float32)
            clip = vp @ head
            if clip[3] <= 0.1:
                continue
            ndc = clip[:3] / clip[3]
            if abs(ndc[0]) > 1.1 or abs(ndc[1]) > 1.1:
                continue
            dist = float(clip[3])
            if dist > 90.0:
                continue
            team = getattr(rp, "team", None)
            color = T.UI[team] if teams_game and T.valid(team) else ((255, 200, 120) if rp.bot else (235, 238, 245))
            flag = "  [FLAG]" if rp.pid in carriers else ""
            if hud.get("mode") == "zombie" and hud.get("zombie"):
                if rp.pid == self._living():
                    color, flag = (255, 220, 120), "  ALIVE"
                else:
                    color = (150, 220, 110)
            lab = self._label(f"{rp.name}  {rp.score}{flag}", max(12, int(22 * k)), color)
            x = (ndc[0] + 1.0) * 0.5 * screen_w - lab.width * 0.5
            y = (1.0 - ndc[1]) * 0.5 * screen_h - lab.height
            m._draw_label(text_shader, m._blank, x - 6 * k, y - 2 * k, (0.0, 0.0, 0.0, 0.45), lab.width + 12 * k, lab.height + 4 * k)
            m._draw_label(text_shader, lab, x, y)
        # objective markers (capture the flag)
        for x, y, text, color, on in self.markers(view, proj, screen_w, screen_h):
            col255 = tuple(int(255 * min(1.0, c * 1.15 + 0.1)) for c in color)
            lab = self._label(text, max(11, int(20 * k)), col255)
            size = 12 * k
            m._draw_label(text_shader, m._blank, x - size * 0.5, y - size * 0.5, tuple(color) + (0.95,), size, size)
            lx = min(max(6 * k, x - lab.width * 0.5), screen_w - lab.width - 6 * k)
            ly = y + size * 0.7 if y < screen_h * 0.5 else y - size * 0.7 - lab.height
            m._draw_label(text_shader, m._blank, lx - 5 * k, ly - 2 * k, (0.0, 0.0, 0.0, 0.45), lab.width + 10 * k, lab.height + 4 * k)
            m._draw_label(text_shader, lab, lx, ly)
        # the game, the score and the clock (top right)
        x1 = screen_w - 24 * k
        y = 20 * k
        big = max(12, int(28 * k))
        zinfo = hud.get("zombie") if hud.get("mode") == "zombie" else None
        if hud.get("mode") == "ctf":
            head = self._label("CAPTURE THE FLAG", big, (255, 214, 120))
        elif hud.get("mode") == "zombie":
            head = self._label("ZOMBIE TAG", big, (150, 225, 110))
        else:
            head = self._label(f"EMBLEMS LEFT  {hud.get('remaining', 0)}", big, (130, 200, 255))
        m._draw_label(text_shader, head, x1 - head.width, y)
        y += head.height + 4 * k
        tleft = hud.get("time_left")
        limit = hud.get("capture_limit", 0) if hud.get("mode") == "ctf" else 0
        if zinfo:
            limit = zinfo.get("goal", 0)
        goal = (f"   first to {limit}" + (" points" if zinfo else "")) if limit else ""
        if tleft is None:
            clock = self._label("no time limit" + goal, max(11, int(22 * k)), (210, 214, 222))
        else:
            tl = max(0, int(tleft))
            clock = self._label(f"{tl // 60}:{tl % 60:02d}" + goal,
                                max(11, int(22 * k)), (210, 214, 222) if tl > 30 else (255, 150, 120))
        m._draw_label(text_shader, clock, x1 - clock.width, y)
        y += clock.height + 6 * k
        if teams_game:
            ts = hud.get("team_scores", {})
            red = self._label(f"RED {ts.get(T.RED, 0)}", big, T.UI[T.RED])
            blue = self._label(f"{ts.get(T.BLUE, 0)} BLUE", big, T.UI[T.BLUE])
            dash = self._label("  -  ", big, (220, 220, 226))
            w = red.width + dash.width + blue.width
            m._draw_label(text_shader, m._blank, x1 - w - 10 * k, y - 2 * k, (0.0, 0.0, 0.0, 0.4), w + 20 * k, red.height + 4 * k)
            m._draw_label(text_shader, red, x1 - w, y)
            m._draw_label(text_shader, dash, x1 - w + red.width, y)
            m._draw_label(text_shader, blue, x1 - blue.width, y)
            y += red.height + 8 * k
        living = self._living() if zinfo else None
        for row in hud.get("scores", [])[:16]:
            score, name, me, bot, team = row[:5]
            pid = row[5] if len(row) > 5 else None
            tag = ""
            if teams_game and T.valid(team):
                color = tuple(min(255, c + 40) for c in T.UI[team]) if me else T.UI[team]
            elif zinfo:
                alive = pid is not None and pid == living
                color = (255, 215, 110) if alive else ((190, 240, 150) if me else (140, 200, 110))
                tag = "  ALIVE" if alive else ""
            else:
                color = (255, 190, 90) if me else ((255, 200, 120) if bot else (225, 228, 234))
            lab = self._label(f"{name}{'  (you)' if me else ''}{tag}   {score}", max(11, int(20 * k)), color)
            m._draw_label(text_shader, m._blank, x1 - lab.width - 10 * k, y - 2 * k, (0.0, 0.0, 0.0, 0.35),
                          lab.width + 20 * k, lab.height + 4 * k)
            m._draw_label(text_shader, lab, x1 - lab.width, y)
            y += lab.height + 5 * k
        # carrying a flag: say so, and where to take it
        if hud.get("carrying") is not None:
            pulse = 0.5 + 0.5 * math.sin(self.time * 5.0)
            text = ("YOU HAVE THE FLAG - bring it to your base!" if teams_game
                    else "YOU HAVE THE FLAG - run it to the drop zone!")
            col = tuple(int(c) for c in (255, 200 + 40 * pulse, 90 + 60 * pulse))
            lab = self._label(text, max(14, int(30 * k)), col)
            bx = screen_w * 0.5 - lab.width * 0.5
            by = screen_h * 0.235
            m._draw_label(text_shader, m._blank, bx - 14 * k, by - 6 * k, (0.0, 0.0, 0.0, 0.5), lab.width + 28 * k, lab.height + 12 * k)
            m._draw_label(text_shader, lab, bx, by)
        # zombie tag: who you are and what to do
        if zinfo and zinfo.get("living") is not None:
            pulse = 0.5 + 0.5 * math.sin(self.time * 5.0)
            if zinfo.get("alive"):
                text = f"YOU'RE ALIVE - RUN!   {int(zinfo.get('points', 0))} / {zinfo.get('goal', 0)} points"
                col = (255, int(200 + 40 * pulse), int(90 + 60 * pulse))
            else:
                text = f"YOU'RE A ZOMBIE - catch {zinfo.get('living_name', '?')}!"
                col = (int(140 + 40 * pulse), 235, 120)
            lab = self._label(text, max(14, int(30 * k)), col)
            bx = screen_w * 0.5 - lab.width * 0.5
            by = screen_h * 0.235
            m._draw_label(text_shader, m._blank, bx - 14 * k, by - 6 * k, (0.0, 0.0, 0.0, 0.5), lab.width + 28 * k, lab.height + 12 * k)
            m._draw_label(text_shader, lab, bx, by)
            note = None
            if zinfo.get("grace", 0.0) > 0.05:
                note = f"{'Head start' if zinfo.get('alive') else 'No catching for'}  {zinfo['grace']:.1f} s"
            elif not zinfo.get("alive") and zinfo.get("no_tagback", 0.0) > 0.05:
                note = f"No tag-backs: you can catch {zinfo.get('living_name', '?')} in {zinfo['no_tagback']:.0f} s"
            if note:
                g = self._label(note, max(12, int(22 * k)), (220, 224, 230))
                m._draw_label(text_shader, g, screen_w * 0.5 - g.width * 0.5, by + lab.height + 14 * k)
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
