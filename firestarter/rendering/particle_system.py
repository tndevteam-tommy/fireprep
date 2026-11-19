"""
Firestarter - Vectorised Particle System
README "Visual & Atmospheric Philosophy": smoke changes from white steam (moisture
evaporation) to thick yellow-brown smoke (pyrolysis volatiles) to thin clean haze (carbon
oxidation); resin pops throw sparks; rain, snow, autumn leaves, summer fireflies and
cold-weather breath vapour bring the ambience.

All particles live in flat numpy arrays and are advected by gravity / buoyancy, drag,
wind, the fire plume and a cheap curl-like turbulence field.
"""

import math
from typing import Tuple, Optional

import numpy as np

K_SMOKE, K_STEAM, K_SPARK, K_RAIN, K_SNOW, K_LEAF, K_EMBER, K_FIREFLY, K_BREATH, K_ASH = range(10)
ADDITIVE_KINDS = (K_SPARK, K_EMBER, K_FIREFLY)
# shader kind codes (see PARTICLE_FS): 0 puff, 2 streak (additive), 3 rain, 4 snow, 5 leaf, 6 glow point
SHADER_KIND = {K_SMOKE: 0.0, K_STEAM: 0.0, K_SPARK: 2.0, K_RAIN: 3.0, K_SNOW: 4.0, K_LEAF: 5.0,
               K_EMBER: 6.0, K_FIREFLY: 6.0, K_BREATH: 0.0, K_ASH: 4.0}


class Particle:
    """Kept for API compatibility (single particle record)."""
    __slots__ = ('pos', 'vel', 'color', 'size', 'lifetime', 'max_lifetime', 'gravity', 'growth_rate')

    def __init__(self, pos, vel, color, size, lifetime, gravity=0.0, growth_rate=0.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = np.array(color, dtype=float)
        self.size = size
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.gravity = gravity
        self.growth_rate = growth_rate


class ParticleSystem:
    def __init__(self, max_particles: int = 7000, seed: int = 3):
        self.max_particles = max_particles
        self.rng = np.random.default_rng(seed)
        n = max_particles
        self.pos = np.zeros((n, 3), np.float32)
        self.vel = np.zeros((n, 3), np.float32)
        self.col = np.zeros((n, 4), np.float32)
        self.size = np.zeros(n, np.float32)
        self.life = np.zeros(n, np.float32)
        self.max_life = np.ones(n, np.float32)
        self.kind = np.zeros(n, np.int32)
        self.grow = np.zeros(n, np.float32)
        self.drag = np.zeros(n, np.float32)
        self.buoy = np.zeros(n, np.float32)
        self.alpha0 = np.zeros(n, np.float32)
        self.temp = np.zeros(n, np.float32)
        self.alive = np.zeros(n, bool)
        self.quality = 1.0
        self.plume_pos = np.zeros(3)
        self.plume_q_kw = 0.0
        self.plume_radius = 0.2
        self.time = 0.0
        self._spawn_accum = {}

    # ------------------------------------------------------------------ bookkeeping
    @property
    def particles(self):
        """Compatibility: number-of-live-particles proxy list."""
        return [None] * int(self.alive.sum())

    def count(self, kind: Optional[int] = None) -> int:
        if kind is None:
            return int(self.alive.sum())
        return int((self.alive & (self.kind == kind)).sum())

    def _alloc(self, n: int) -> np.ndarray:
        free = np.nonzero(~self.alive)[0]
        if len(free) < n:
            # recycle the oldest-looking particles of low priority (rain/snow first)
            n = len(free)
        return free[:n]

    def _accum(self, key: str, rate: float, dt: float) -> int:
        a = self._spawn_accum.get(key, 0.0) + rate * dt * self.quality
        n = int(a)
        self._spawn_accum[key] = a - n
        return n

    def _emit(self, n, pos, vel, col, size, life, kind, grow=0.0, drag=0.5, buoy=0.0, temp=0.0):
        if n <= 0:
            return
        idx = self._alloc(n)
        m = len(idx)
        if m == 0:
            return
        self.pos[idx] = pos[:m]
        self.vel[idx] = vel[:m]
        self.col[idx] = col[:m] if np.ndim(col) == 2 else col
        self.alpha0[idx] = self.col[idx, 3]
        self.size[idx] = size[:m] if np.ndim(size) else size
        self.life[idx] = life[:m] if np.ndim(life) else life
        self.max_life[idx] = self.life[idx]
        self.kind[idx] = kind
        self.grow[idx] = grow
        self.drag[idx] = drag
        self.buoy[idx] = buoy
        self.temp[idx] = temp[:m] if np.ndim(temp) else temp
        self.alive[idx] = True

    # ------------------------------------------------------------------ spawners
    def spawn_smoke(self, pos: np.ndarray, rate: float, dt: float, color=(0.55, 0.50, 0.42), alpha=0.32,
                    spread: float = 0.06, size=(0.12, 0.22), life=(2.8, 5.0), key: str = "smoke",
                    wind_vec: Optional[np.ndarray] = None, kind: int = K_SMOKE, rise: float = 0.45,
                    grow: Optional[float] = None):
        n = self._accum(key + str(int(pos[0] * 100)) + str(int(pos[2] * 100)), rate, dt)
        if n <= 0:
            return
        r = self.rng
        p = np.asarray(pos)[None, :] + r.normal(0, spread, (n, 3)) * np.array([1, 0.3, 1])
        v = np.c_[r.normal(0, 0.05, n), r.uniform(rise * 0.7, rise * 1.3, n), r.normal(0, 0.05, n)]
        c = np.tile((*color, alpha), (n, 1)) * np.c_[r.uniform(0.9, 1.1, (n, 3)), np.ones(n)]
        g = grow if grow is not None else (0.32 if kind == K_SMOKE else 0.42)
        self._emit(n, p, v, c, r.uniform(*size, n), r.uniform(*life, n), kind, grow=g,
                   drag=0.6, buoy=0.25 if kind == K_SMOKE else 0.45)

    def spawn_steam(self, pos: np.ndarray, rate_kg_s: float = 0.01, dt: float = 0.016, wind_vec=None):
        rate = min(30.0, 2.0 + rate_kg_s * 8000.0)
        self.spawn_smoke(pos, rate, dt, color=(0.95, 0.96, 0.98), alpha=0.38, spread=0.06, size=(0.10, 0.18),
                         life=(1.0, 2.0), key="steam", kind=K_STEAM, rise=0.7)

    def spawn_pyrolysis_smoke(self, pos: np.ndarray, rate_g_s: float = 0.5, dt: float = 0.016, wind_vec=None):
        rate = min(45.0, 2.0 + rate_g_s * 12.0)
        self.spawn_smoke(pos, rate, dt, key="pyro")

    def spawn_sparks(self, pos: np.ndarray, n: int, energy: float = 1.0, spread: float = 0.05):
        n = int(n * self.quality)
        if n <= 0:
            return
        r = self.rng
        p = np.asarray(pos)[None, :] + r.normal(0, spread, (n, 3))
        v = np.c_[r.normal(0, 0.9, n), r.uniform(1.0, 3.2, n), r.normal(0, 0.9, n)] * energy
        temp = r.uniform(1000.0, 1350.0, n)
        c = np.tile((1.0, 0.7, 0.3, 1.0), (n, 1))
        self._emit(n, p, v, c, r.uniform(0.006, 0.012, n), r.uniform(0.5, 1.6, n), K_SPARK, drag=0.35, buoy=1.6, temp=temp)

    def spawn_resin_sparks(self, pos: np.ndarray, count: int = 6):
        self.spawn_sparks(pos, count, 1.2)

    def spawn_embers(self, pos: np.ndarray, rate: float, dt: float, radius: float = 0.15):
        n = self._accum("embers", rate, dt)
        if n <= 0:
            return
        r = self.rng
        p = np.asarray(pos)[None, :] + np.c_[r.normal(0, radius, n), r.uniform(0, 0.2, n), r.normal(0, radius, n)]
        v = np.c_[r.normal(0, 0.25, n), r.uniform(0.6, 1.8, n), r.normal(0, 0.25, n)]
        c = np.tile((1.0, 0.55, 0.18, 1.0), (n, 1))
        self._emit(n, p, v, c, r.uniform(0.012, 0.022, n), r.uniform(1.2, 3.2, n), K_EMBER, drag=0.8, buoy=0.9,
                   temp=r.uniform(950.0, 1150.0, n))

    def spawn_flame(self, pos: np.ndarray, count: int = 2, height: float = 0.35, draft_vel: float = 0.5):
        """Legacy API: small licking embers above a flame."""
        self.spawn_sparks(pos + np.array([0, height * 0.5, 0]), max(1, count // 2), 0.5)

    def spawn_weather(self, cam_pos: np.ndarray, rain_intensity: float = 0.0, snow_accumulation: float = 0.0,
                      wind_vec: Optional[np.ndarray] = None, dt: float = 0.016, snow_intensity: float = 0.0,
                      leaves: float = 0.0, fireflies: float = 0.0):
        r = self.rng
        wind = np.zeros(3) if wind_vec is None else np.asarray(wind_vec, dtype=float)
        if rain_intensity > 0.03:
            n = self._accum("rain", 3200.0 * rain_intensity, dt)
            if n > 0:
                p = cam_pos[None, :] + np.c_[r.uniform(-9, 9, n), r.uniform(1.0, 7.0, n), r.uniform(-9, 9, n)] + wind[None, :] * 0.3
                v = np.tile(np.array([0.0, -9.0, 0.0]) + wind * 0.35, (n, 1)) + r.normal(0, 0.2, (n, 3))
                c = np.tile((0.75, 0.80, 0.88, 0.42 + 0.25 * rain_intensity), (n, 1))
                self._emit(n, p, v, c, r.uniform(0.009, 0.014, n), r.uniform(0.8, 1.1, n), K_RAIN, drag=0.0)
        if snow_intensity > 0.03:
            n = self._accum("snow", 500.0 * snow_intensity, dt)
            if n > 0:
                p = cam_pos[None, :] + np.c_[r.uniform(-11, 11, n), r.uniform(1.0, 7.0, n), r.uniform(-11, 11, n)]
                v = np.tile(np.array([0.0, -1.0, 0.0]) + wind * 0.25, (n, 1)) + r.normal(0, 0.15, (n, 3))
                c = np.tile((0.97, 0.98, 1.0, 0.85), (n, 1))
                self._emit(n, p, v, c, r.uniform(0.02, 0.045, n), r.uniform(4.0, 7.0, n), K_SNOW, drag=0.9)
        if leaves > 0.0:
            n = self._accum("leaves", 6.0 * leaves, dt)
            if n > 0:
                p = cam_pos[None, :] + np.c_[r.uniform(-9, 9, n), r.uniform(3.0, 8.0, n), r.uniform(-9, 9, n)]
                v = np.c_[r.normal(0, 0.3, n), r.uniform(-0.7, -0.4, n), r.normal(0, 0.3, n)] + wind[None, :] * 0.3
                pal = np.array([[0.75, 0.36, 0.10], [0.82, 0.62, 0.16], [0.62, 0.20, 0.08], [0.55, 0.45, 0.15]])
                c = np.c_[pal[r.integers(0, 4, n)], np.full(n, 0.95)]
                self._emit(n, p, v, c, r.uniform(0.05, 0.08, n), r.uniform(6.0, 10.0, n), K_LEAF, drag=1.2)
        if fireflies > 0.0 and self.count(K_FIREFLY) < 60 * fireflies:
            n = self._accum("ff", 8.0 * fireflies, dt)
            if n > 0:
                ang = r.uniform(0, 2 * np.pi, n)
                dist = r.uniform(4.0, 18.0, n)
                p = cam_pos[None, :] + np.c_[np.cos(ang) * dist, r.uniform(-1.2, 0.2, n), np.sin(ang) * dist]
                v = r.normal(0, 0.25, (n, 3))
                c = np.tile((0.75, 1.0, 0.35, 1.0), (n, 1))
                self._emit(n, p, v, c, r.uniform(0.03, 0.05, n), r.uniform(5.0, 9.0, n), K_FIREFLY, drag=0.4)

    def spawn_breath(self, pos: np.ndarray, direction: np.ndarray, strength: float = 1.0):
        r = self.rng
        n = int(14 * self.quality)
        p = np.asarray(pos)[None, :] + r.normal(0, 0.01, (n, 3))
        v = np.asarray(direction)[None, :] * r.uniform(0.8, 1.6, (n, 1)) * strength + r.normal(0, 0.1, (n, 3))
        c = np.tile((0.9, 0.92, 0.95, 0.16), (n, 1))
        self._emit(n, p, v, c, r.uniform(0.05, 0.09, n), r.uniform(0.8, 1.4, n), K_BREATH, grow=0.25, drag=1.6, buoy=0.1)

    def spawn_chips(self, pos: np.ndarray, n: int):
        r = self.rng
        p = np.asarray(pos)[None, :] + r.normal(0, 0.05, (n, 3))
        v = np.c_[r.normal(0, 1.2, n), r.uniform(0.5, 2.2, n), r.normal(0, 1.2, n)]
        c = np.tile((0.78, 0.62, 0.42, 1.0), (n, 1)) * np.c_[r.uniform(0.8, 1.1, (n, 3)), np.ones(n)]
        self._emit(n, p, v, c, r.uniform(0.015, 0.03, n), r.uniform(0.8, 1.4, n), K_LEAF, drag=0.5)

    def spawn_ash(self, pos: np.ndarray, n: int):
        r = self.rng
        p = np.asarray(pos)[None, :] + r.normal(0, 0.1, (n, 3))
        v = np.c_[r.normal(0, 0.6, n), r.uniform(0.5, 1.5, n), r.normal(0, 0.6, n)]
        c = np.tile((0.7, 0.69, 0.66, 0.8), (n, 1))
        self._emit(n, p, v, c, r.uniform(0.01, 0.02, n), r.uniform(2.0, 4.0, n), K_ASH, drag=1.4, buoy=0.2)

    def clear(self, kind: Optional[int] = None):
        if kind is None:
            self.alive[:] = False
        else:
            self.alive[self.kind == kind] = False

    # ------------------------------------------------------------------ update
    def set_plume(self, pos: np.ndarray, q_kw: float, radius: float):
        self.plume_pos = np.asarray(pos, dtype=float)
        self.plume_q_kw = q_kw
        self.plume_radius = max(0.1, radius)

    def update(self, dt: float, wind_vec: Optional[np.ndarray] = None, ground_fn=None):
        self.time += dt
        a = self.alive
        if not np.any(a):
            return
        idx = np.nonzero(a)[0]
        wind = np.zeros(3) if wind_vec is None else np.asarray(wind_vec, dtype=np.float32)
        self.life[idx] -= dt
        dead = self.life[idx] <= 0
        self.alive[idx[dead]] = False
        idx = idx[~dead]
        if len(idx) == 0:
            return
        p = self.pos[idx]
        v = self.vel[idx]
        k = self.kind[idx]
        t = self.time
        # turbulence (curl-like sines)
        turb = np.c_[np.sin(p[:, 1] * 3.1 + t * 1.7 + p[:, 2] * 2.0),
                     np.sin(p[:, 0] * 2.3 + t * 1.1) * 0.3,
                     np.cos(p[:, 1] * 2.7 - t * 1.3 + p[:, 0] * 1.6)]
        tscale = np.where((k == K_SMOKE) | (k == K_STEAM) | (k == K_BREATH), 0.35,
                          np.where(k == K_FIREFLY, 0.8, np.where(k == K_LEAF, 1.2, 0.15)))
        # fire plume updraft
        q = self.plume_q_kw
        if q > 0.05:
            d = p - self.plume_pos[None, :]
            h = np.maximum(0.0, d[:, 1])
            rr = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
            b = self.plume_radius + 0.15 * h
            u = 1.9 * q ** 0.2 * np.exp(-(rr / b) ** 2) * np.exp(-h / 4.0)
            plume = np.c_[np.zeros(len(p)), u, np.zeros(len(p))]
        else:
            plume = 0.0
        wind_factor = np.where(k == K_RAIN, 0.0, np.where(k == K_SPARK, 0.5, 1.0))[:, None]
        target_v = plume + wind[None, :] * wind_factor * np.minimum(1.0, p[:, 1:2] * 0.3 + 0.4)
        drag = self.drag[idx][:, None]
        acc = (target_v - v) * drag + turb * tscale[:, None]
        acc[:, 1] += self.buoy[idx] * np.where(k == K_SPARK, np.clip((self.temp[idx] - 400.0) / 800.0, 0.0, 1.0) - 0.55, 1.0) * (k != K_RAIN)
        grav = np.where(k == K_RAIN, -9.8, np.where(k == K_SPARK, -2.0, np.where(k == K_SNOW, -0.6, np.where(k == K_LEAF, -0.4, 0.0))))
        acc[:, 1] += grav
        v = v + acc * dt
        if np.any(k == K_FIREFLY):
            ff = k == K_FIREFLY
            v[ff] = v[ff] * 0.97 + np.c_[np.sin(t * 2.0 + idx[ff]), np.cos(t * 1.3 + idx[ff] * 0.7) * 0.5, np.cos(t * 1.7 + idx[ff])][:] * 0.05
        p = p + v * dt
        self.vel[idx] = v
        self.pos[idx] = p
        self.size[idx] += self.grow[idx] * dt
        life_ratio = np.clip(self.life[idx] / np.maximum(self.max_life[idx], 1e-4), 0.0, 1.0)
        fade_in = np.clip((1.0 - life_ratio) * 6.0, 0.0, 1.0)
        alpha = self.alpha0[idx] * np.where((k == K_SMOKE) | (k == K_STEAM) | (k == K_BREATH),
                                            fade_in * life_ratio ** 1.2, np.minimum(1.0, life_ratio * 3.0))
        self.col[idx, 3] = alpha
        # sparks and embers cool: yellow -> orange -> dull red
        hot = (k == K_SPARK) | (k == K_EMBER)
        if np.any(hot):
            hi = idx[hot]
            self.temp[hi] -= dt * np.where(self.kind[hi] == K_SPARK, 420.0, 160.0)
            T = self.temp[hi]
            x = np.clip((T - 550.0) / 700.0, 0.0, 1.0)
            self.col[hi, 0] = 1.0
            self.col[hi, 1] = 0.08 + 0.7 * x ** 1.3
            self.col[hi, 2] = 0.01 + 0.35 * x ** 3
            bright = 2.5 * np.clip((T - 520.0) / 500.0, 0.0, 1.5) ** 2
            self.col[hi, 3] = bright * np.minimum(1.0, life_ratio[hot] * 4.0)
            self.alive[hi[T < 520.0]] = False
        if np.any(k == K_FIREFLY):
            fi = idx[k == K_FIREFLY]
            blink = np.clip(np.sin(t * 2.3 + fi * 1.7) * 3.0 - 1.5, 0.0, 1.0)
            self.col[fi, 3] = blink * 1.6 * np.minimum(1.0, self.life[fi])
        # rain / snow / leaves / ash die on the ground
        if ground_fn is not None:
            low = np.isin(k, (K_RAIN, K_SNOW, K_LEAF, K_ASH, K_SPARK))
            if np.any(low):
                li = idx[low]
                gy = ground_fn(self.pos[li, 0], self.pos[li, 2])
                hit = self.pos[li, 1] < gy
                self.alive[li[hit]] = False

    # ------------------------------------------------------------------ render data
    def get_instances(self, cam_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (alpha_instances, additive_instances), each (n, 12) float32, alpha ones sorted back-to-front."""
        idx = np.nonzero(self.alive)[0]
        if len(idx) == 0:
            z = np.zeros((0, 12), np.float32)
            return z, z
        k = self.kind[idx]
        shader_kind = np.array([SHADER_KIND[int(x)] for x in range(10)], np.float32)[k]
        inst = np.zeros((len(idx), 12), np.float32)
        inst[:, 0:3] = self.pos[idx]
        inst[:, 3] = self.size[idx]
        inst[:, 4:8] = self.col[idx]
        inst[:, 8:11] = self.vel[idx]
        inst[:, 11] = shader_kind
        add = np.isin(k, ADDITIVE_KINDS)
        alpha = inst[~add]
        if len(alpha):
            d = np.sum((alpha[:, 0:3] - cam_pos[None, :].astype(np.float32)) ** 2, axis=1)
            alpha = alpha[np.argsort(-d)]
        return alpha, inst[add]

    def get_render_data(self, cam_right: np.ndarray, cam_up: np.ndarray):
        """Legacy API: returns (vertices, indices) of simple quads."""
        a, b = self.get_instances(np.zeros(3))
        inst = np.concatenate([a, b]) if len(a) + len(b) else np.zeros((0, 12), np.float32)
        n = len(inst)
        verts = np.zeros(n * 4 * 9, np.float32)
        inds = np.zeros(n * 6, np.uint32)
        for i in range(n):
            pos = inst[i, 0:3]
            s = inst[i, 3] * 0.5
            corners = [pos - cam_right * s - cam_up * s, pos + cam_right * s - cam_up * s,
                       pos + cam_right * s + cam_up * s, pos - cam_right * s + cam_up * s]
            for j, (c, uv) in enumerate(zip(corners, [(0, 0), (1, 0), (1, 1), (0, 1)])):
                verts[(i * 4 + j) * 9:(i * 4 + j + 1) * 9] = [c[0], c[1], c[2], uv[0], uv[1], *inst[i, 4:8]]
            inds[i * 6:i * 6 + 6] = [i * 4, i * 4 + 1, i * 4 + 2, i * 4, i * 4 + 2, i * 4 + 3]
        return verts, inds
