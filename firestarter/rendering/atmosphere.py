"""
Firestarter - Atmosphere & lighting state derived from the weather simulation:
sky palette through day, golden hour, twilight and night; sun / moon light; ambient; fog;
exposure; and the flickering fire light from the thermodynamic grid's heat release.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


def _lerp(a, b, t):
    return np.asarray(a, dtype=float) * (1.0 - t) + np.asarray(b, dtype=float) * t


def _smooth(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class LightingState:
    sun_dir: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.8, 0.2]))
    light_dir: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.8, 0.2]))
    light_color: np.ndarray = field(default_factory=lambda: np.ones(3))
    sky_amb: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.25, 0.3]))
    ground_amb: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.09, 0.07]))
    zenith: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.4, 0.8]))
    horizon: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.7, 0.85]))
    ground_col: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.1, 0.1]))
    sun_disc: np.ndarray = field(default_factory=lambda: np.ones(3))
    moon_dir: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    moon_intensity: float = 0.0
    fog_color: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.7, 0.8]))
    fog_sun: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.8, 0.6]))
    fog_density: float = 0.004
    night: float = 0.0
    cloud_lit: np.ndarray = field(default_factory=lambda: np.ones(3))
    cloud_shadow: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.62, 0.66]))
    exposure: float = 1.0
    grade_shadow: np.ndarray = field(default_factory=lambda: np.ones(3))
    grade_high: np.ndarray = field(default_factory=lambda: np.ones(3))
    saturation: float = 1.05


class Atmosphere:
    def __init__(self):
        self.exposure = 1.0
        self._adapted = False        # first frame snaps straight to the target exposure
        self.fire_flicker = 1.0
        self._t = 0.0
        self.fire_intensity = 0.0
        self.fire_color = np.array([1.0, 0.5, 0.2])
        self.fire_pos = np.zeros(3)
        self.fire_range = 6.0

    def compute(self, weather: Any, dt: float, fire_kw: float = 0.0, cam_fire_dist: float = 99.0) -> LightingState:
        s = LightingState()
        sun = weather.get_sun_direction()
        el = math.degrees(math.asin(max(-1.0, min(1.0, sun[1]))))
        cloud = float(weather.cloud_cover)
        precip = float(weather.rain_intensity + weather.snow_intensity)
        s.sun_dir = sun
        day = _smooth(-6.0, 8.0, el)          # 0 night .. 1 day
        golden = _smooth(-2.0, 4.0, el) * (1.0 - _smooth(8.0, 24.0, el))
        twilight = _smooth(-12.0, -1.0, el) * (1.0 - _smooth(-1.0, 6.0, el))
        night = 1.0 - _smooth(-12.0, -4.0, el)
        s.night = night

        # --- sky palette
        z_day, h_day = np.array([0.16, 0.36, 0.78]), np.array([0.60, 0.74, 0.90])
        z_gold, h_gold = np.array([0.22, 0.32, 0.58]), np.array([1.00, 0.60, 0.32])
        z_twi, h_twi = np.array([0.06, 0.08, 0.20]), np.array([0.60, 0.30, 0.28])
        z_night, h_night = np.array([0.0025, 0.004, 0.011]), np.array([0.012, 0.017, 0.034])
        zen = _lerp(z_night, z_day, day)
        hor = _lerp(h_night, h_day, day)
        zen = _lerp(zen, z_gold, golden * 0.8)
        hor = _lerp(hor, h_gold, golden)
        zen = _lerp(zen, z_twi, twilight * 0.7)
        hor = _lerp(hor, h_twi, twilight * 0.8)
        grey = np.array([0.55, 0.58, 0.62]) * (0.15 + 0.85 * day)
        overcast = min(1.0, cloud * 0.9 + precip * 0.5)
        s.zenith = _lerp(zen, grey * 0.9, overcast * 0.75)
        s.horizon = _lerp(hor, grey, overcast * 0.7)
        s.ground_col = s.horizon * 0.35
        # --- sun light
        sun_int = 3.1 * _smooth(-3.0, 10.0, el) * (1.0 - 0.78 * overcast)
        warm = 1.0 - _smooth(5.0, 35.0, el)
        sun_col = _lerp(np.array([1.0, 0.96, 0.90]), np.array([1.0, 0.58, 0.28]), warm)
        s.sun_disc = sun_col * (1.2 + 2.5 * warm) * (1.0 - overcast * 0.85) * _smooth(-3.0, 1.0, el)
        # --- moon light at night (cool, dim)
        moon = weather.get_moon_direction()
        s.moon_dir = moon
        moon_up = _smooth(-2.0, 10.0, math.degrees(math.asin(max(-1, min(1, moon[1])))))
        s.moon_intensity = night * moon_up * (1.0 - overcast * 0.8)
        if sun_int > 0.05 or s.moon_intensity < 0.01:
            s.light_dir = sun if sun[1] > -0.05 else np.array([sun[0], 0.05, sun[2]])
            s.light_color = sun_col * sun_int
        else:
            s.light_dir = moon
            s.light_color = np.array([0.45, 0.55, 0.85]) * 0.11 * s.moon_intensity
        # --- ambient
        dome = s.zenith * 0.9 + s.horizon * 0.35
        # hemispherical sky light is bluer than sunlight, but far less saturated than the zenith
        dome = _lerp(dome, np.full(3, float(np.dot(dome, [0.2126, 0.7152, 0.0722]))), 0.45)
        s.sky_amb = _lerp(dome, np.array([0.5, 0.52, 0.56]) * (0.25 + 0.6 * day), overcast * 0.5)
        s.sky_amb = s.sky_amb * (0.9 + 0.3 * day) + np.array([0.006, 0.009, 0.02]) * night
        s.ground_amb = np.array([0.14, 0.12, 0.08]) * (0.1 + 0.8 * day) + np.array([0.006, 0.007, 0.01]) * night
        if weather.snow_accumulation_m > 0.05:
            s.ground_amb = s.ground_amb * 1.8 + np.array([0.05, 0.055, 0.06]) * day
        # --- fog & haze
        s.fog_color = _lerp(s.horizon, grey, overcast * 0.5)
        s.fog_sun = _lerp(s.horizon, sun_col * 1.4, _smooth(-4.0, 4.0, el) * (1.0 - overcast))
        s.fog_density = 0.0016 + 0.045 * float(weather.fog_density) ** 1.5 + 0.008 * precip
        # --- clouds
        s.cloud_lit = _lerp(np.array([0.04, 0.05, 0.08]), np.array([1.0, 0.98, 0.95]), day)
        s.cloud_lit = _lerp(s.cloud_lit, np.array([1.0, 0.62, 0.42]), golden * 0.85)
        s.cloud_shadow = _lerp(np.array([0.02, 0.025, 0.04]), np.array([0.55, 0.58, 0.66]), day)
        s.cloud_shadow = _lerp(s.cloud_shadow, np.array([0.42, 0.30, 0.38]), golden * 0.6)
        # --- exposure: eyes adapt to the dark; a nearby fire pulls them back
        scene_lum = 0.35 * sun_int + 0.6 * float(np.mean(s.sky_amb)) + 0.02
        fire_at_cam = 0.11 * math.sqrt(max(0.0, fire_kw)) / (1.0 + 0.35 * cam_fire_dist ** 2)
        target = 0.54 / (scene_lum + fire_at_cam * 0.8) ** 0.55
        # the eye adapts to darkness, but only so far: a moonlit forest stays dark and blue
        target = max(0.3, min(2.4, target))
        k = 1.0 - math.exp(-dt / 1.2) if (dt > 0 and self._adapted) else 1.0
        self._adapted = True
        self.exposure += (target - self.exposure) * k
        s.exposure = self.exposure
        # --- colour grade: warm highlights at golden hour, cool blue shadows at night
        s.grade_high = _lerp(np.array([1.02, 1.0, 0.97]), np.array([1.08, 0.98, 0.88]), golden)
        s.grade_shadow = _lerp(np.array([0.98, 1.0, 1.03]), np.array([0.92, 0.97, 1.12]), night)
        s.saturation = 1.16 - 0.3 * overcast
        return s

    def fire_light(self, grid: Any, time: float, dt: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """(position, colour, intensity, range) of the campfire point light with flicker."""
        flame_kw = grid.flame_hrr / 1000.0
        coal_kw = grid.char_hrr / 1000.0
        bed = grid.coal_bed
        glow_kw = coal_kw + (bed.char_kg * 8.0 if bed.T > 500 else 0.0)
        flick = (1.0 + 0.13 * math.sin(time * 13.1) + 0.09 * math.sin(time * 23.7 + 1.3)
                 + 0.07 * math.sin(time * 7.3 + 0.4) + 0.05 * math.sin(time * 41.0))
        # visible light is a small fraction of a wood fire's (mostly infrared) radiant output
        target = 0.3 * flame_kw ** 0.5 * flick + 0.18 * glow_kw ** 0.5 * (1.0 + 0.04 * math.sin(time * 2.1))
        if flame_kw < 0.02 and glow_kw < 0.02:
            target = 0.0
        self.fire_intensity += (target - self.fire_intensity) * min(1.0, dt * 14.0)
        frac_flame = flame_kw / max(1e-6, flame_kw + glow_kw)
        col = _lerp(np.array([1.0, 0.28, 0.06]), np.array([1.0, 0.47, 0.15]), frac_flame)
        h = min(0.5, 0.15 + 0.35 * grid.flame_height)
        c = np.array(grid.fire_centroid if grid.flame_hrr > 5 else grid.pit_center, dtype=float)
        pos = np.array([c[0] + 0.05 * math.sin(time * 9.0), grid.pit_center[1] + h, c[2] + 0.05 * math.cos(time * 7.0)])
        rng = 3.5 + 3.0 * math.sqrt(max(0.0, flame_kw + 0.5 * glow_kw))
        return pos, col, max(0.0, self.fire_intensity), min(26.0, rng)
