"""Drawing the weather (``systems.weather``): the rain, the puddles, the spray, the fog
and the lightning.

* **Rain**: streaks falling through a column of air round the eye - as many as the
  quality preset allows (``GraphicsSettings.rain_drops``) times how hard it is raining -
  each one anchored in the world (walk through the rain and it stays where it is: the
  column wraps round the eye), slanting a little in the wind, only over open ground
  (``LevelData.open_sky``) and never under a roof (``shelters``). Where they land, little
  crowns of splash on the ground.
* **Puddles**: dark, glossy water on the ground (a soft-edged disc) with a sheen of the
  neon in it, each as big as the wet lets it be (``PuddleSite.size``); rings spreading
  across them while it rains.
* **Spray**: the drops a car throws up driving through a puddle (``Weather.drops``).
* **Fog and lightning** are the scene's (``scene.SceneRenderer._apply_weather``: the
  fog's colour and thickness, the far plane pulled in to what can be seen, the flash
  lighting up the sky and everything under it).

Purely visual; the numbers are all the simulation's.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cyberduke.render.context import DrawContext
from cyberduke.systems.weather import PuddleSite, Weather

RAIN_REACH = 8.5  # the column of rain round the eye: this far each way ..
RAIN_ABOVE, RAIN_BELOW = 6.0, 3.0  # .. and from this far over the eye to this far under it
RAIN_SPEED = (6.5, 10.0)  # m/s: a drizzle drifts down, a downpour comes straight down
RAIN_RGB = (0.62, 0.7, 0.82)
SPLASHES = 90  # crowns of splash round the eye in the heaviest rain
SPLASH_REACH = 7.0
PUDDLE_DRAW = 40.0  # metres: puddles further off than this are not drawn
RIPPLE_DRAW = 14.0
PUDDLE_RGB = (0.03, 0.045, 0.07)  # the water: dark, it takes the colour of the night
SHEENS = ((0.2, 0.75, 1.0), (1.0, 0.25, 0.8), (1.0, 0.65, 0.25), (0.55, 0.6, 0.75))
SPRAY_RGB = (0.75, 0.8, 0.88)
PUDDLE_STEP = 0.01  # the puddles' meshes are rebuilt each time the wet moves on this much
PUDDLE_SLICES = 20


def _hash(n: np.ndarray, k: float) -> np.ndarray:
    """Cheap, stable pseudo-random numbers in [0, 1) for integer ``n`` (per ``k``)."""
    return np.modf(np.sin(n * 12.9898 + k * 78.233) * 43758.5453)[0] % 1.0


class WeatherRenderer:
    """Draws the weather around the eye (see the module's notes)."""

    def __init__(self) -> None:
        self._ids = np.arange(4096, dtype=np.float64)
        self._puddle_key: tuple | None = None
        self._puddle_cache = None
        self._puddle_glows: list = []

    def clear(self, ctx: DrawContext | None = None) -> None:
        """Forgets the puddles' meshes (a new level)."""
        if ctx is not None:
            for old in self._puddle_glows:
                ctx.invalidate(old)
        self._puddle_key, self._puddle_cache, self._puddle_glows = None, None, []

    # ------------------------------------------------------------------ rain
    def rain_segments(
        self, weather: Weather, level, eye, time: float, budget: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(streaks (n, 2, 3), splash crowns (m, 2, 3)) to draw this frame."""
        empty = np.zeros((0, 2, 3))
        sky = getattr(level, "open_sky", ())
        if not weather.raining or not sky or budget <= 0:
            return empty, empty
        rain = weather.rain
        n = min(len(self._ids), int(budget * min(1.0, rain * 1.15)))
        ids = self._ids[:n]
        ex, ey, ez = eye
        span = 2.0 * RAIN_REACH
        height = RAIN_ABOVE + RAIN_BELOW
        speed = RAIN_SPEED[0] + (RAIN_SPEED[1] - RAIN_SPEED[0]) * rain
        # where each streak is: fixed in the world (the column wraps round the eye)
        hx = _hash(ids, 1.0) * span
        hz = _hash(ids, 2.0) * span
        x = ex - RAIN_REACH + np.mod(hx - (ex - RAIN_REACH), span)
        z = ez - RAIN_REACH + np.mod(hz - (ez - RAIN_REACH), span)
        fall = np.mod(
            time * speed * (0.9 + 0.2 * _hash(ids, 3.0)) + _hash(ids, 4.0) * height, height
        )
        y = ey + RAIN_ABOVE - fall
        ground, keep = _ground_under(x, z, sky)
        keep &= y > ground
        keep &= ~_sheltered(x, z, y, getattr(level, "shelters", ()))
        length = 0.18 + 0.04 * speed * (0.6 + 0.8 * rain)  # (motion blur: longer when harder)
        wind = 0.12 * math.sin(time * 0.07) + 0.05
        top = np.stack((x + wind * length, y + length, z + 0.3 * wind * length), axis=1)
        bottom = np.stack((x, y, z), axis=1)
        streaks = np.stack((bottom, top), axis=1)[keep]
        # splash crowns where drops are landing, round the eye (each lasts a moment)
        m = int(SPLASHES * rain)
        if m <= 0:
            return streaks, empty
        sid = self._ids[:m]
        rate = 7.0
        cycle = np.floor(time * rate + _hash(sid, 5.0))
        age = np.mod(time * rate + _hash(sid, 5.0), 1.0)
        sx = ex + (_hash(sid * 31.0 + cycle, 6.0) * 2.0 - 1.0) * SPLASH_REACH
        sz = ez + (_hash(sid * 17.0 + cycle, 7.0) * 2.0 - 1.0) * SPLASH_REACH
        sground, sk = _ground_under(sx, sz, sky)
        sk &= ~_sheltered(sx, sz, sground + 0.01, getattr(level, "shelters", ()))
        sk &= age < 0.4  # (a crown lasts a moment: most of the cycle there is none)
        sx, sz, sground, age = sx[sk], sz[sk], sground[sk], age[sk] / 0.4
        crowns = []
        rise = 0.07 * np.sin(np.pi * age)
        spread = 0.025 + 0.05 * age
        for k in range(3):
            a = 2.0 * math.pi * k / 3.0
            dx, dz = math.cos(a), math.sin(a)
            base = np.stack(
                (sx + dx * spread * 0.3, sground + 0.005, sz + dz * spread * 0.3), axis=1
            )
            tip = np.stack((sx + dx * spread, sground + 0.005 + rise, sz + dz * spread), axis=1)
            crowns.append(np.stack((base, tip), axis=1))
        return streaks, np.concatenate(crowns) if crowns else empty

    def draw_rain(
        self, ctx: DrawContext, weather: Weather, level, eye, time: float, budget: int, light: float
    ) -> None:
        """The rain (after everything else: the streaks are see-through). ``light``: how
        lit the air is (the day, a lightning flash) - the streaks catch it."""
        streaks, crowns = self.rain_segments(weather, level, eye, time, budget)
        lift = min(1.0, 0.35 + 0.6 * light)
        rgb = tuple(c * (0.55 + 0.45 * lift) for c in RAIN_RGB)
        if len(streaks):
            ctx.lines(streaks, (*rgb, 0.3 + 0.25 * weather.rain + 0.2 * light), 1.0)
        if len(crowns):
            ctx.lines(crowns, (*rgb, 0.3), 1.0)

    # ------------------------------------------------------------------ puddles
    def draw_puddles(
        self, ctx: DrawContext, puddles: Sequence[PuddleSite], weather: Weather, eye, time: float
    ) -> None:
        """The water standing on the ground (in the translucent pass: it writes no
        depth): a soft dark stain of wet round each puddle, the water itself - black
        glass that catches the lamps and glows faintly with the neon it mirrors - and a
        sheen of that neon across it; while it rains, rings spreading over it. All the
        puddles are one mesh of each (rebuilt as the wet changes, a step at a time)."""
        wet = weather.wet
        if wet <= 0.0 or not puddles:
            return
        meshes = self._puddle_meshes(ctx, puddles, wet)
        if meshes is None:
            return
        rim_key, rim, sheens, water = meshes
        ctx.depth_mask(False)
        ctx.glow_mesh(rim_key, (*PUDDLE_RGB, 1.0), lambda: rim)
        if water is not None:
            ctx.alpha(0.8)
            ctx.draw_batch(*water)
            ctx.alpha(1.0)
        for key, rgb, tris in sheens:
            ctx.glow_mesh(key, (*rgb, 1.0), lambda tris=tris: tris)
        ctx.depth_mask(True)
        ctx.color(1.0, 1.0, 1.0)
        if not weather.raining:
            return
        ex, _, ez = eye
        ripples: list[list] = [[], [], []]
        for k, p in enumerate(puddles):
            if abs(p.x - ex) > RIPPLE_DRAW or abs(p.z - ez) > RIPPLE_DRAW:
                continue
            size = p.size(wet)
            if size > 0.0:
                self._ripples(p, size, k, weather.rain, time, ripples)
        for level, segments in enumerate(ripples):
            if segments:
                ctx.lines(segments, (0.7, 0.78, 0.9, (0.06, 0.12, 0.2)[level]), 1.0)

    def _puddle_meshes(self, ctx: DrawContext, puddles: Sequence[PuddleSite], wet: float):
        """(rim glow key, its triangles, [(sheen glow key, colour, triangles)], water batch)
        for this wetness - built
        (and the last step's thrown away) when the wet has moved on a step."""
        step = round(wet / PUDDLE_STEP)
        key = (id(puddles), step)
        if key == self._puddle_key:
            return self._puddle_cache
        for old in self._puddle_glows:
            ctx.invalidate(old)
        self._puddle_key = key
        sizes = np.array([p.size(step * PUDDLE_STEP) for p in puddles])
        shown = [(k, p, s) for k, (p, s) in enumerate(zip(puddles, sizes, strict=True)) if s > 0]
        if not shown:
            self._puddle_glows = []
            self._puddle_cache = None
            return None
        arrays = _puddle_arrays(shown)
        alpha = np.minimum(1.0, arrays["size"] * 2.5)
        rim = _soft_ellipses(arrays, 1.25, 0.45 * alpha, 0.5)
        rim_key = ("puddle rims", key)
        sheens = []
        which = np.array([k % len(SHEENS) for k, _, _ in shown])
        for c, rgb in enumerate(SHEENS):
            pick = which == c
            if pick.any():
                part = {name: value[pick] for name, value in arrays.items()}
                tris = _soft_ellipses(part, 0.55, 0.16 * alpha[pick], 0.15, (0.15, -0.1), 0.8)
                sheens.append((("puddle sheen", key, c), rgb, tris))
        water = _water_batch(shown)
        self._puddle_glows = [rim_key] + [k for k, _, _ in sheens]
        self._puddle_cache = (rim_key, rim, sheens, water)
        return self._puddle_cache

    @staticmethod
    def _ripples(p: PuddleSite, s: float, k: int, rain: float, time: float, out) -> None:
        """Rings spreading from where the drops hit the puddle (a few at once, more in
        harder rain), each fading as it grows."""
        c, n = math.cos(math.radians(p.yaw)), math.sin(math.radians(p.yaw))
        count = 1 + int(rain * 4)
        for j in range(count):
            phase = time * (1.3 + 0.4 * j) + (k * 0.37 + j * 0.61)
            cycle = math.floor(phase)
            age = phase - cycle
            h1 = math.sin((k * 13.1 + j * 7.7 + cycle) * 12.9898) * 43758.5453 % 1.0
            h2 = math.sin((k * 5.3 + j * 3.1 + cycle) * 78.233) * 43758.5453 % 1.0
            u, v = (h1 * 2.0 - 1.0) * 0.6 * p.rx * s, (h2 * 2.0 - 1.0) * 0.6 * p.rz * s
            x = p.x + u * c + v * n
            z = p.z - u * n + v * c
            r = 0.03 + 0.17 * age
            level = min(2, int((1.0 - age) * 3.0))
            y = p.y + 0.002
            pts = [
                (x + r * math.cos(2 * math.pi * i / 10), y, z + r * math.sin(2 * math.pi * i / 10))
                for i in range(11)
            ]
            out[level].extend(zip(pts[:-1], pts[1:], strict=True))

    # ------------------------------------------------------------------ spray
    @staticmethod
    def draw_spray(ctx: DrawContext, weather: Weather, eye) -> None:
        """The drops a car throws up, each a short streak along the way it is flying."""
        drops = weather.drops
        if not drops:
            return
        segs = []
        for d in drops:
            k = 0.025
            segs.append(((d.x, d.y, d.z), (d.x - d.vx * k, d.y - d.vy * k, d.z - d.vz * k)))
        ctx.lines(segs, (*SPRAY_RGB, 0.5), 1.5)


def _puddle_arrays(shown) -> dict[str, np.ndarray]:
    """The shown puddles' numbers as arrays (one row each): middle, half-axes (already
    at their size), the turn, the height of the water."""
    return {
        "x": np.array([p.x for _, p, _ in shown]),
        "y": np.array([p.y for _, p, _ in shown]),
        "z": np.array([p.z for _, p, _ in shown]),
        "rx": np.array([p.rx * s for _, p, s in shown]),
        "rz": np.array([p.rz * s for _, p, s in shown]),
        "c": np.cos(np.radians([p.yaw for _, p, _ in shown])),
        "n": np.sin(np.radians([p.yaw for _, p, _ in shown])),
        "size": np.array([s for _, _, s in shown]),
    }


def _ellipses(a: dict, scale: float, shift=(0.0, 0.0), squash: float = 1.0):
    """The rims of every puddle at ``scale`` of its size: (n, slices + 1, 2) world x, z
    (closed), and their middles (n, 2) (``shift``: moved by that share of the half-axes
    over 0.55, ``squash``: rz narrowed by that)."""
    theta = np.linspace(0.0, 2.0 * math.pi, PUDDLE_SLICES + 1)
    rx, rz = a["rx"][:, None] * scale, a["rz"][:, None] * scale
    mu, mv = shift[0] * rx / 0.55, shift[1] * rz / 0.55
    u = mu + np.cos(theta)[None, :] * rx
    v = mv + np.sin(theta)[None, :] * rz * squash
    c, n = a["c"][:, None], a["n"][:, None]
    ring = np.stack((a["x"][:, None] + u * c + v * n, a["z"][:, None] - u * n + v * c), axis=2)
    middle = np.stack((a["x"] + (mu * c + mv * n)[:, 0], a["z"] + (-mu * n + mv * c)[:, 0]), axis=1)
    return ring, middle


def _soft_ellipses(
    a: dict, scale: float, alpha: np.ndarray, inner: float, shift=(0.0, 0.0), squash=1.0
) -> np.ndarray:
    """Triangles (x, y, z, opacity) of a soft-edged ellipse on every puddle: ``alpha`` (one
    each) out to ``inner`` of its size, fading to nothing at its rim - one mesh."""
    ring, mid = _ellipses(a, scale, shift, squash)
    y = a["y"] + (0.001 if shift != (0.0, 0.0) else 0.0)
    count, slices = len(y), PUDDLE_SLICES
    edge = np.concatenate(
        (ring[:, :, 0:1], np.repeat(y[:, None, None], slices + 1, 1), ring[:, :, 1:2]), 2
    )
    m3 = np.stack((mid[:, 0], y, mid[:, 1]), axis=1)[:, None, :]
    plateau = m3 + (edge - m3) * inner
    al = np.repeat(alpha[:, None], slices, 1)[..., None]
    zero = np.zeros_like(al)

    def vert(p: np.ndarray, o: np.ndarray) -> np.ndarray:  # (n, slices, 4)
        return np.concatenate((p, o), axis=2)

    mid_v = vert(np.repeat(m3, slices, 1), al)
    pa, pb = vert(plateau[:, :-1], al), vert(plateau[:, 1:], al)
    ea, eb = vert(edge[:, :-1], zero), vert(edge[:, 1:], zero)
    tris = np.stack((mid_v, pa, pb, pa, ea, eb, pa, eb, pb), axis=2)  # (n, slices, 9, 4)
    return tris.reshape(count * slices * 9, 4)


def _water_batch(shown) -> tuple[np.ndarray, np.ndarray, list] | None:
    """The puddles' water as one lit batch (``DrawContext.draw_batch``): a fan per puddle,
    facing up, dark, glossy (the lamps shine in it) and faintly lit with its neon."""
    if not shown:
        return None
    a = _puddle_arrays(shown)
    ring, mid = _ellipses(a, 1.0)
    count, slices = len(shown), PUDDLE_SLICES
    per = slices + 1  # (the middle, then the rim)
    v = np.zeros((count, per, 16), dtype=np.float32)
    v[:, 0, 0], v[:, 0, 2] = mid[:, 0], mid[:, 1]
    v[:, 1:, 0], v[:, 1:, 2] = ring[:, :-1, 0], ring[:, :-1, 1]
    v[:, :, 1] = (a["y"] + 0.0005)[:, None]
    v[:, :, 4] = 1.0  # (facing up)
    v[:, :, 6:8] = 0.5
    v[:, :, 8:11] = PUDDLE_RGB
    glow = np.array([[c * 0.05 for c in SHEENS[k % len(SHEENS)]] for k, _, _ in shown])
    v[:, :, 11:14] = glow[:, None, :]
    v[:, :, 14], v[:, :, 15] = 0.7, 110.0
    base = (np.arange(count) * per)[:, None]
    i = np.arange(slices)[None, :]
    idx = np.stack((base + 0 * i, base + 1 + (i + 1) % slices, base + 1 + i), axis=2)
    return v.reshape(count * per, 16), idx.reshape(-1).astype(np.uint32), [(None, 0, idx.size)]


def _ground_under(x: np.ndarray, z: np.ndarray, sky) -> tuple[np.ndarray, np.ndarray]:
    """(the ground's height, whether there is open sky) at each (x, z)."""
    ground = np.zeros_like(x)
    found = np.zeros(x.shape, dtype=bool)
    for r in sky:
        inside = (~found) & (x >= r.min_x) & (x <= r.max_x) & (z >= r.min_z) & (z <= r.max_z)
        ground[inside] = r.ground
        found |= inside
    return ground, found


def _sheltered(x: np.ndarray, z: np.ndarray, y: np.ndarray, shelters) -> np.ndarray:
    out = np.zeros(x.shape, dtype=bool)
    for s in shelters:
        out |= (x >= s.min_x) & (x <= s.max_x) & (z >= s.min_z) & (z <= s.max_z) & (y < s.max_y)
    return out


def fog_colour(daylight: float, flash: float, sky: tuple[float, float, float]) -> tuple:
    """The fog's colour: the night's blue-grey murk lit by the neon, the day's smog-brown,
    white-blue in a lightning flash."""
    night = (0.085, 0.09, 0.11)
    day = tuple(min(1.0, c * 1.6 + 0.04) for c in sky)
    base = tuple(n + (d - n) * daylight for n, d in zip(night, day, strict=True))
    return tuple(min(1.0, b + flash * w) for b, w in zip(base, (0.55, 0.6, 0.75), strict=True))
