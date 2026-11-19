"""The city's weather: rain, puddles, thunderstorms and fog.

It is the city's, not a level's - like the clock it keeps going whichever level the
netrunner is in, and only shows out of doors (``LevelData.outdoor``; the rain falls on
the level's ``open_sky``, not under its ``shelters``). Everything moves in game time
(``core.clock.WorldClock``: half a game minute a second), slowly:

* **Rain** goes through spells (``SPELLS``): dry, a drizzle, steady rain, a downpour, a
  thunderstorm - each lasting a while (game minutes) and followed by another picked at
  random (``FOLLOWS``: a Markov chain - a drizzle often thickens into rain, a storm
  blows over into rain or a downpour, a dry spell usually ends in a drizzle). The rain
  never jumps: it eases towards what the spell brings (``rain``), a drizzle fading in
  over a quarter of an hour, a storm front coming in faster.
* **Wet**: the puddles fill while it rains (harder rain, faster) and dry out slowly
  after - faster by day. Each puddle (``puddle_sites``: the level's ground under the
  open sky, clear of anything standing on it) has its own depth: the deepest dips fill
  first and the last to dry.
* **Thunderstorms**: lightning (``flash``: a sharp flicker of light over the whole city,
  two or three strokes in a strike) every few seconds to half a minute - only in a
  storm, and only while the rain is heavy.
* **Fog** comes and goes on its own spells (``FOG_SPELLS``): clear, a haze, a mist, a
  real fog - likelier at night and towards dawn, and after rain - fading in and out
  over hours, not minutes. Fog (and heavy rain) limit how far anything can be seen
  (``fog_density``, ``visibility``): the renderer fades the world into it and draws no
  further than that.
* **Spray**: a car driving through a puddle on the road throws water up from its wheels
  (``Weather.spray``).

Deterministic: everything random comes from the weather's own seeded generators.
Pure simulation (``render.weather`` draws it).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from cyberduke.core.geometry import AABB
from cyberduke.world.level import OpenSky

# ----------------------------------------------------------------------------- rain spells


@dataclass(frozen=True)
class Spell:
    """A spell of weather: how hard it rains (0..1), how long it lasts (game minutes),
    how fast the rain comes up to it (per game minute), and whether it thunders."""

    rain: float
    minutes: tuple[float, float]
    arrive: float
    storm: bool = False


SPELLS: dict[str, Spell] = {
    "dry": Spell(0.0, (70.0, 260.0), 0.0),
    "drizzle": Spell(0.22, (40.0, 150.0), 0.012),  # fades in over a quarter of an hour
    "rain": Spell(0.55, (30.0, 120.0), 0.02),
    "downpour": Spell(0.85, (15.0, 50.0), 0.04),
    "storm": Spell(1.0, (20.0, 60.0), 0.05, storm=True),
}
# What a spell turns into when it is over (weights)
FOLLOWS: dict[str, tuple[tuple[str, float], ...]] = {
    "dry": (("dry", 0.3), ("drizzle", 0.52), ("rain", 0.12), ("storm", 0.06)),
    "drizzle": (("dry", 0.35), ("drizzle", 0.15), ("rain", 0.4), ("downpour", 0.1)),
    "rain": (("drizzle", 0.35), ("downpour", 0.3), ("storm", 0.15), ("dry", 0.2)),
    "downpour": (("rain", 0.5), ("storm", 0.3), ("drizzle", 0.2)),
    "storm": (("downpour", 0.35), ("rain", 0.45), ("drizzle", 0.2)),
}
RAIN_EASE = 0.025  # how fast the rain dies away (per game minute)
STORM_RAIN = 0.7  # lightning only while the rain is at least this heavy

# ----------------------------------------------------------------------------- the wet
WET_RISE = 0.022  # per game minute, at full rain (a downpour fills the puddles in an hour)
WET_DRY = 0.0035  # per game minute, at night (dry by day twice as fast)

# ----------------------------------------------------------------------------- fog
FOG_SPELLS: dict[str, float] = {"clear": 0.0, "haze": 0.25, "mist": 0.5, "fog": 0.85}
FOG_MINUTES = (60.0, 300.0)
FOG_RATE = 0.004  # per game minute: a fog takes a few game hours to come in or lift
FOG_DENSITY = 0.15  # per metre, at fog 1 (about 15 m of sight)
RAIN_HAZE = 0.025  # per metre, at rain 1 (a downpour hides the far end of the street)
SIGHT = 2.2  # density x distance beyond which nothing shows (exp2 fog: < 1 % left)

# ----------------------------------------------------------------------------- lightning
STRIKE_EVERY = (5.0, 28.0)  # real seconds between strikes in a storm
STRIKE_LENGTH = 0.9  # seconds a strike's strokes take to flicker out
STROKE_DECAY = 0.07  # seconds for a stroke's light to fall by e

# ----------------------------------------------------------------------------- spray
SPRAY_MAX = 500  # drops in the air at once
SPRAY_SPEED = 3.0  # m/s: slower cars do not throw any
GRAVITY = 9.8


def fog_density(fog: float, rain: float = 0.0) -> float:
    """How thick the air is (per metre, for exp2 fog: what is left of a thing ``d``
    metres away is exp(-(density * d)^2))."""
    return FOG_DENSITY * max(0.0, fog) + RAIN_HAZE * max(0.0, rain)


def visibility(density: float) -> float:
    """How far anything can be seen through air that thick (metres; inf in clear air)."""
    return math.inf if density <= 1e-6 else SIGHT / density


def _pick(rng: random.Random, options: Sequence[tuple[str, float]]) -> str:
    total = sum(w for _, w in options)
    x = rng.uniform(0.0, total)
    for name, w in options:
        x -= w
        if x <= 0.0:
            return name
    return options[-1][0]


def _toward(value: float, target: float, step: float) -> float:
    if value < target:
        return min(target, value + step)
    return max(target, value - step)


# ----------------------------------------------------------------------------- puddles
@dataclass(frozen=True)
class PuddleSite:
    """Where a puddle gathers: its middle (``y`` the ground's surface), its full size
    (half-axes ``rx``, ``rz`` before its ``yaw``), and ``fills``: how wet the ground has
    to be before any water stands there (0 the deepest dip .. 1 never)."""

    x: float
    y: float
    z: float
    rx: float
    rz: float
    yaw: float
    fills: float
    road: bool = False  # on the road (the cars drive through it)

    def size(self, wet: float) -> float:
        """How much of its full size the puddle is at this wetness (0 none)."""
        if wet <= self.fills:
            return 0.0
        return min(1.0, (wet - self.fills) / max(0.05, (1.0 - self.fills) * 0.6))

    def covers(self, x: float, z: float, wet: float) -> bool:
        s = self.size(wet)
        if s <= 0.0:
            return False
        c, n = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        dx, dz = x - self.x, z - self.z
        u, v = dx * c - dz * n, dx * n + dz * c
        return (u / (self.rx * s)) ** 2 + (v / (self.rz * s)) ** 2 <= 1.0


PUDDLE_SPACING = 2.2  # metres between candidate spots
PUDDLE_CLEARANCE = 0.25  # kept this far from anything standing on the ground
PUDDLE_LIFT = 0.004  # the water's surface this far over the ground (over the road paint)


def under_shelter(x: float, z: float, shelters: Iterable[AABB], y: float | None = None) -> bool:
    """Whether (``x``, ``z``) - at height ``y``, if given - is under one of ``shelters``
    (a roof over open ground: nothing falls below its ``max_y``)."""
    for s in shelters:
        if s.min_x <= x <= s.max_x and s.min_z <= z <= s.max_z and (y is None or y < s.max_y):
            return True
    return False


def open_sky_at(x: float, z: float, sky: Iterable[OpenSky]) -> OpenSky | None:
    """The open-sky region over (``x``, ``z``), if any."""
    for r in sky:
        if r.min_x <= x <= r.max_x and r.min_z <= z <= r.max_z:
            return r
    return None


def puddle_sites(level, seed: int = 1701) -> list[PuddleSite]:
    """Where the puddles gather on ``level``'s open ground (``LevelData.open_sky``):
    spots on a jittered grid, each with a size and a depth of its own, clear of whatever
    stands on the ground there (its colliders), of the shelters, of the region's edges
    (the walls) and of each other - a spot too tight for a big one gets a smaller one.
    Deterministic."""
    sky: Sequence[OpenSky] = getattr(level, "open_sky", ())
    if not sky:
        return []
    shelters: Sequence[AABB] = getattr(level, "shelters", ())
    rng = random.Random(seed)
    colliders = [c for s in level.sectors for c in s.colliders]
    lanes = getattr(level, "lanes", ())
    road_y = min((lane.y for lane in lanes), default=None)
    sites: list[PuddleSite] = []

    def clear(region: OpenSky, x: float, z: float, reach: float) -> bool:
        if not (
            region.min_x + reach <= x <= region.max_x - reach
            and region.min_z + reach <= z <= region.max_z - reach
        ):
            return False
        ground = region.ground
        for c in colliders:
            if (
                c.min_y <= ground + 0.3
                and c.max_y > ground
                and c.min_x - reach < x < c.max_x + reach
                and c.min_z - reach < z < c.max_z + reach
            ):
                return False
        if under_shelter(x, z, shelters):
            return False
        return all(math.hypot(x - s.x, z - s.z) >= reach + max(s.rx, s.rz) for s in sites)

    for region in sky:
        if not region.puddles:
            continue
        x0, x1, z0, z1 = region.min_x, region.max_x, region.min_z, region.max_z
        nx = max(1, round((x1 - x0) / PUDDLE_SPACING))
        nz = max(1, round((z1 - z0) / PUDDLE_SPACING))
        for i in range(nx):
            for j in range(nz):
                skip = rng.random() < 0.3  # (not everywhere: patches of it)
                x = x0 + (i + rng.uniform(0.15, 0.85)) * (x1 - x0) / nx
                z = z0 + (j + rng.uniform(0.15, 0.85)) * (z1 - z0) / nz
                rx, rz = rng.uniform(0.4, 1.3), rng.uniform(0.3, 0.8)
                yaw = rng.uniform(0.0, 180.0)
                fills = rng.uniform(0.05, 0.85)
                if skip:
                    continue
                for shrink in (1.0, 0.7, 0.45):
                    reach = max(rx, rz) * shrink + PUDDLE_CLEARANCE
                    if clear(region, x, z, reach):
                        on_road = road_y is not None and abs(region.ground - road_y) < 0.01
                        sites.append(
                            PuddleSite(
                                x,
                                region.ground + PUDDLE_LIFT,
                                z,
                                rx * shrink,
                                rz * shrink,
                                yaw,
                                fills,
                                road=on_road,
                            )  # fmt: skip
                        )
                        break
    return sites


# ----------------------------------------------------------------------------- spray
@dataclass
class Drop:
    """A drop of water thrown up by a wheel (flying: position, velocity, age, life)."""

    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    age: float = 0.0
    life: float = 0.8
    ground: float = 0.0


# ----------------------------------------------------------------------------- weather
@dataclass
class Weather:
    """The city's weather now (see the module's notes). ``update`` moves it on."""

    seed: int = 2077
    spell: str = "dry"
    spell_left: float = 25.0  # game minutes
    rain: float = 0.0  # how hard it is raining now (0..1)
    wet: float = 0.35  # how wet the ground is (0 dry .. 1 flooded): the street starts damp
    fog_spell: str = "haze"
    fog_left: float = 150.0
    fog: float = 0.2
    strike_in: float = 8.0  # real seconds to the next strike (in a storm)
    strike_age: float = math.inf  # seconds since the last strike began
    strokes: tuple[tuple[float, float], ...] = ()  # the strike's (delay, strength)
    strikes: int = 0  # how many strikes there have been
    drops: list[Drop] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._spray_rng = random.Random(self.seed + 1)

    # ------------------------------------------------------------------ state
    @property
    def storm(self) -> bool:
        """Thundering: a storm spell with the rain heavy enough."""
        return SPELLS[self.spell].storm and self.rain >= STORM_RAIN

    @property
    def raining(self) -> bool:
        return self.rain > 0.02

    @property
    def flash(self) -> float:
        """How bright the lightning is this instant (0 none .. ~1 a close stroke)."""
        if self.strike_age > STRIKE_LENGTH:
            return 0.0
        out = 0.0
        for delay, strength in self.strokes:
            t = self.strike_age - delay
            if t >= 0.0:
                out = max(out, strength * math.exp(-t / STROKE_DECAY))
        return out

    @property
    def density(self) -> float:
        """How thick the air is (``fog_density``)."""
        return fog_density(self.fog, self.rain)

    @property
    def visibility(self) -> float:
        return visibility(self.density)

    @property
    def summary(self) -> str:
        """A word or two for it ("drizzle", "thunderstorm, mist")."""
        words = {
            "dry": "dry" if self.rain < 0.05 else "rain easing",
            "drizzle": "drizzle",
            "rain": "rain",
            "downpour": "downpour",
            "storm": "thunderstorm",
        }[self.spell]
        if self.fog >= 0.4:
            words += ", fog" if self.fog >= 0.65 else ", mist"
        return words

    # ------------------------------------------------------------------ moving on
    def update(self, dt: float, clock=None) -> None:
        """Moves the weather on by ``dt`` real seconds (``clock``: the city's; the rain,
        the puddles and the fog move in its game minutes, the lightning in seconds)."""
        if dt <= 0.0:
            return
        rate = clock.rate if clock is not None else 0.5
        hour = clock.hour if clock is not None else 21.0
        daylight = clock.daylight if clock is not None else 0.0
        minutes = dt * rate
        if minutes > 0.0:
            self._advance(minutes, hour, daylight)
        self._lightning(dt)

    def _advance(self, minutes: float, hour: float, daylight: float) -> None:
        # the rain's spells
        self.spell_left -= minutes
        if self.spell_left <= 0.0:
            self.spell = _pick(self._rng, FOLLOWS[self.spell])
            lo, hi = SPELLS[self.spell].minutes
            self.spell_left = self._rng.uniform(lo, hi)
        spell = SPELLS[self.spell]
        rising = spell.rain > self.rain
        self.rain = _toward(
            self.rain, spell.rain, (spell.arrive if rising else RAIN_EASE) * minutes
        )
        # the wet: filling while it rains, drying when it stops
        if self.rain > 0.05:
            self.wet = min(1.0, self.wet + WET_RISE * self.rain * minutes)
        else:
            self.wet = max(0.0, self.wet - WET_DRY * (1.0 + daylight) * minutes)
        # the fog's spells
        self.fog_left -= minutes
        if self.fog_left <= 0.0:
            night = hour >= 21.0 or hour < 9.0
            weights = {
                "clear": 0.45 if not night else 0.25,
                "haze": 0.35,
                "mist": (0.15 if not night else 0.27) + 0.2 * self.wet,
                "fog": (0.05 if not night else 0.16) + 0.08 * self.wet,
            }
            self.fog_spell = _pick(self._rng, tuple(weights.items()))
            self.fog_left = self._rng.uniform(*FOG_MINUTES)
        self.fog = _toward(self.fog, FOG_SPELLS[self.fog_spell], FOG_RATE * minutes)

    def _lightning(self, dt: float) -> None:
        self.strike_age += dt
        if not self.storm:
            self.strike_in = max(self.strike_in, STRIKE_EVERY[0])
            return
        self.strike_in -= dt
        if self.strike_in <= 0.0:
            rng = self._rng
            strokes = [(0.0, rng.uniform(0.6, 1.0))]
            for _ in range(rng.randint(1, 3)):
                strokes.append((strokes[-1][0] + rng.uniform(0.07, 0.25), rng.uniform(0.3, 0.9)))
            self.strokes = tuple(strokes)
            self.strike_age = 0.0
            self.strikes += 1
            self.strike_in = rng.uniform(*STRIKE_EVERY)

    # ------------------------------------------------------------------ spray
    def spray(self, cars: Iterable, puddles: Sequence[PuddleSite], dt: float) -> None:
        """Cars driving through puddles on the road throw water up from their wheels
        (the faster, the more); the drops fly, fall and are gone when they land."""
        if dt <= 0.0:
            return
        kept = []
        for d in self.drops:
            d.age += dt
            d.vy -= GRAVITY * dt
            d.x += d.vx * dt
            d.y += d.vy * dt
            d.z += d.vz * dt
            if d.age < d.life and d.y > d.ground:
                kept.append(d)
        self.drops = kept
        road = [p for p in puddles if p.road and p.size(self.wet) > 0.0]
        if not road:
            return
        rng = self._spray_rng
        for car in cars:
            if car.speed < SPRAY_SPEED:
                continue
            heading = car.heading
            for wx, wz, side in car.model.wheels():
                x = car.x - heading * wx
                z = car.z - heading * wz
                for p in road:
                    if abs(p.x - x) > 1.2 or abs(p.z - z) > 1.2 or not p.covers(x, z, self.wet):
                        continue
                    n = int(car.speed * 0.9 * dt * 30.0 * (0.4 + 0.6 * p.size(self.wet)))
                    out = -heading * side  # (which way the wheel's outer wall faces, in z)
                    for _ in range(n):
                        if len(self.drops) >= SPRAY_MAX:
                            break
                        self.drops.append(
                            Drop(
                                x + rng.uniform(-0.1, 0.1),
                                p.y + 0.03,
                                z + rng.uniform(-0.08, 0.08),
                                heading * car.speed * rng.uniform(0.05, 0.3),
                                rng.uniform(1.0, 3.2) * min(1.5, car.speed / 10.0),
                                out * rng.uniform(0.6, 2.6),
                                life=rng.uniform(0.4, 0.9),
                                ground=p.y,
                            )
                        )
                    break

    # ------------------------------------------------------------------ saving
    def to_dict(self) -> dict:
        return {
            "spell": self.spell,
            "spell_left": round(self.spell_left, 3),
            "rain": round(self.rain, 4),
            "wet": round(self.wet, 4),
            "fog_spell": self.fog_spell,
            "fog_left": round(self.fog_left, 3),
            "fog": round(self.fog, 4),
        }

    def load(self, data: dict) -> None:
        """Takes the weather a save was made in (anything missing or unknown stays as
        it is)."""
        if data.get("spell") in SPELLS:
            self.spell = data["spell"]
        if data.get("fog_spell") in FOG_SPELLS:
            self.fog_spell = data["fog_spell"]
        for key in ("spell_left", "rain", "wet", "fog_left", "fog"):
            if key in data:
                setattr(self, key, float(data[key]))
        self.rain = min(1.0, max(0.0, self.rain))
        self.wet = min(1.0, max(0.0, self.wet))
        self.fog = min(1.0, max(0.0, self.fog))
        self.strike_age = math.inf
        self.drops = []

    def set_spell(self, spell: str, minutes: float | None = None) -> None:
        """Starts ``spell`` now (for ``minutes``, or a length of its own)."""
        self.spell = spell
        lo, hi = SPELLS[spell].minutes
        self.spell_left = minutes if minutes is not None else self._rng.uniform(lo, hi)
