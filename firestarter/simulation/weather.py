"""
Firestarter - Weather, Seasons & Environmental Parameters
Implements DESIGN Section 7.1:

* Seasonal climatology (Summer 25-35 C / >900 W/m^2 sun, Autumn rain, Winter -15..-2 C
  with snow, Spring snow-melt & sap flow).
* Real solar geometry at 45 deg N: declination from day-of-year, hour angle from local
  solar time -> accurate sun elevation / azimuth ("accurate sun vectors").
* A continuous stochastic weather model (seeded, deterministic) with smooth rain-front
  transitions, cloud cover, wind gusts, humidity, snow accumulation / melt and ground
  wetness.  The same model is used both in real time and during fast-forward time skips.

World axes: +X = East, -Z = North, +Y = Up.
"""

from enum import Enum
import math
from typing import Dict, Any, Optional, List

import numpy as np

from firestarter.config import H_FUS_ICE  # noqa: F401  (re-exported for winter enthalpy docs)

LATITUDE_DEG = 45.0


class Season(Enum):
    SUMMER = "Summer"
    AUTUMN = "Autumn"
    WINTER = "Winter"
    SPRING = "Spring"


SEASON_ORDER: List[Season] = [Season.SUMMER, Season.AUTUMN, Season.WINTER, Season.SPRING]
DAYS_PER_SEASON = 90

# Day-of-year on which each in-game season begins (solstices / equinoxes)
SEASON_START_DOY = {
    Season.SUMMER: 172,   # June 21
    Season.AUTUMN: 264,   # Sept 21
    Season.WINTER: 355,   # Dec 21
    Season.SPRING: 80,    # March 21
}

# Climatology per season
#   base_temp / diurnal amplitude (C), mean wind (m/s), fraction of time with precipitation,
#   mean precipitation front duration (h), base relative humidity, peak clear-sky irradiance
CLIMATE = {
    Season.SUMMER: dict(base_temp=29.0, temp_amp=5.5, wind=2.0, wet_fraction=0.10, front_hours=3.0, rh=0.42, solar=1000.0),
    Season.AUTUMN: dict(base_temp=11.5, temp_amp=5.0, wind=3.6, wet_fraction=0.42, front_hours=9.0, rh=0.80, solar=620.0),
    Season.WINTER: dict(base_temp=-8.0, temp_amp=5.0, wind=4.2, wet_fraction=0.32, front_hours=12.0, rh=0.84, solar=420.0),
    Season.SPRING: dict(base_temp=14.0, temp_amp=5.5, wind=3.2, wet_fraction=0.30, front_hours=6.0, rh=0.74, solar=820.0),
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class WeatherSystem:
    def __init__(self, seed: int = 7, start_hour: float = 18.5, season: Season = Season.SUMMER):
        self.rng = np.random.default_rng(seed)
        self.current_season = season
        self.day_of_season = 1.0
        self.time_of_day_hours = start_hour
        self.total_hours = 0.0

        # Environmental state
        self.ambient_temp_c = 28.0
        self.solar_radiation_w_m2 = 0.0
        self.rain_intensity = 0.0           # 0..1 (1.0 ~ 10 mm/h heavy rain)
        self.snow_intensity = 0.0           # 0..1 snowfall rate
        self.snow_accumulation_m = 0.0
        self.cloud_cover = 0.15
        self.fog_density = 0.0
        self.relative_humidity = 0.45
        self.ground_wetness = 0.1           # 0..1 wet-look of terrain and ground capillary source
        self.wind_speed_m_s = 2.2
        self.wind_gust = 0.0
        self.wind_direction_rad = 0.45      # direction the wind blows TOWARD (radians in XZ-plane)
        self.wind_vector = np.array([math.cos(0.45), 0.0, math.sin(0.45)]) * 2.2
        self.falling_leaves = 0.0

        # Stochastic precipitation state
        self._precip_target = 0.0
        self._front_hours_left = float(self.rng.uniform(3.0, 10.0))
        self._cloud_target = 0.15
        self._gust_phase = float(self.rng.uniform(0, 100))
        self._wind_dir_target = 0.45

        # Forced overrides for scripted scenes / tests
        self.forced_precip: Optional[float] = None
        self.forced_cloud: Optional[float] = None
        self.forced_wind: Optional[float] = None
        self.forced_fog: Optional[float] = None

        self.rain_hours_since_skip = 0.0
        self._update_parameters(0.0)

    # ------------------------------------------------------------------ calendar
    @property
    def day_of_year(self) -> int:
        start = SEASON_START_DOY[self.current_season]
        return int((start + int(self.day_of_season) - 1 - 1) % 365) + 1

    @property
    def climate(self) -> Dict[str, float]:
        return CLIMATE[self.current_season]

    def season_progress(self) -> float:
        return max(0.0, min(1.0, (self.day_of_season - 1.0) / DAYS_PER_SEASON))

    def clock_string(self) -> str:
        h = int(self.time_of_day_hours) % 24
        m = int((self.time_of_day_hours - int(self.time_of_day_hours)) * 60) % 60
        return f"{h:02d}:{m:02d}"

    # ------------------------------------------------------------------ solar geometry
    def solar_declination_rad(self) -> float:
        n = self.day_of_year
        return math.radians(23.44) * math.sin(2.0 * math.pi * (284 + n) / 365.0)

    def sun_elevation_azimuth(self, hour: Optional[float] = None) -> (float, float):
        """Returns (elevation_rad, azimuth_rad from North, clockwise)."""
        hour = self.time_of_day_hours if hour is None else hour
        lat = math.radians(LATITUDE_DEG)
        dec = self.solar_declination_rad()
        ha = math.radians(15.0 * (hour - 12.0))
        sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
        el = math.asin(max(-1.0, min(1.0, sin_el)))
        cos_az = (math.sin(dec) - math.sin(el) * math.sin(lat)) / max(1e-6, math.cos(el) * math.cos(lat))
        az = math.acos(max(-1.0, min(1.0, cos_az)))
        if ha > 0:
            az = 2.0 * math.pi - az
        return el, az

    @staticmethod
    def _dir_from_el_az(el: float, az: float) -> np.ndarray:
        # North = -Z, East = +X
        x = math.cos(el) * math.sin(az)
        z = -math.cos(el) * math.cos(az)
        y = math.sin(el)
        v = np.array([x, y, z], dtype=float)
        return v / max(1e-9, np.linalg.norm(v))

    def get_sun_direction(self) -> np.ndarray:
        """Unit vector pointing from the ground TOWARD the sun (world space)."""
        el, az = self.sun_elevation_azimuth()
        return self._dir_from_el_az(el, az)

    def sun_elevation_deg(self) -> float:
        return math.degrees(self.sun_elevation_azimuth()[0])

    def get_moon_direction(self) -> np.ndarray:
        # Approximate full-ish moon roughly opposite the sun, riding low in summer / high in winter
        el, az = self.sun_elevation_azimuth(self.time_of_day_hours + 12.0)
        dec = -self.solar_declination_rad()
        lat = math.radians(LATITUDE_DEG)
        ha = math.radians(15.0 * (self.time_of_day_hours - 0.0))
        sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
        el = math.asin(max(-1.0, min(1.0, sin_el)))
        return self._dir_from_el_az(el, (az + 0.35) % (2 * math.pi))

    def sunrise_sunset_hours(self) -> (float, float):
        lat = math.radians(LATITUDE_DEG)
        dec = self.solar_declination_rad()
        c = -math.tan(lat) * math.tan(dec)
        c = max(-1.0, min(1.0, c))
        h0 = math.degrees(math.acos(c)) / 15.0
        return 12.0 - h0, 12.0 + h0

    def daylight_factor(self) -> float:
        """0 at night -> 1 in full day, smooth through civil twilight (-6 deg)."""
        el = self.sun_elevation_deg()
        return max(0.0, min(1.0, (el + 6.0) / 12.0))

    def is_night(self) -> bool:
        return self.sun_elevation_deg() < -6.0

    # ------------------------------------------------------------------ dynamics
    def update(self, dt_seconds: float, clock_scale: float = 1.0):
        """Real-time update (dt in real seconds; clock_scale accelerates the sky clock)."""
        hours = dt_seconds * clock_scale / 3600.0
        self._advance_clock(hours)
        self._step_weather(hours, dt_seconds)
        self._update_parameters(dt_seconds)

    def advance_time(self, hours: float) -> Dict[str, Any]:
        """Fast-forward (Time Passage Engine). Steps weather hour by hour and returns stats."""
        stats = {"rain_hours": 0.0, "snow_hours": 0.0, "mean_temp": 0.0, "samples": 0,
                 "season_changes": 0, "start_season": self.current_season.value}
        remaining = hours
        step = 1.0 if hours <= 24 * 14 else 3.0
        prev_season = self.current_season
        while remaining > 1e-9:
            h = min(step, remaining)
            self._advance_clock(h)
            self._step_weather(h, h * 3600.0)
            self._update_parameters(h * 3600.0)
            if self.rain_intensity > 0.05:
                stats["rain_hours"] += h
            if self.snow_intensity > 0.05:
                stats["snow_hours"] += h
            stats["mean_temp"] += self.ambient_temp_c
            stats["samples"] += 1
            if self.current_season != prev_season:
                stats["season_changes"] += 1
                prev_season = self.current_season
            remaining -= h
        if stats["samples"]:
            stats["mean_temp"] /= stats["samples"]
        stats["end_season"] = self.current_season.value
        return stats

    def _advance_clock(self, hours: float):
        self.total_hours += hours
        self.time_of_day_hours += hours
        while self.time_of_day_hours >= 24.0:
            self.time_of_day_hours -= 24.0
            self.day_of_season += 1.0
            if self.day_of_season > DAYS_PER_SEASON:
                idx = (SEASON_ORDER.index(self.current_season) + 1) % len(SEASON_ORDER)
                self.current_season = SEASON_ORDER[idx]
                self.day_of_season = 1.0

    def _step_weather(self, hours: float, dt_seconds: float):
        c = self.climate
        # --- precipitation fronts (two-state renewal process with seasonal duty cycle) ---
        self._front_hours_left -= hours
        if self._front_hours_left <= 0.0:
            # Alternating renewal process: wet fronts and dry spells alternate, so the long-run
            # wet fraction equals the seasonal climatology (wet_fraction).
            currently_wet = self._precip_target > 0.0
            if not currently_wet:
                self._precip_target = float(self.rng.uniform(0.35, 1.0))
                self._front_hours_left = float(self.rng.exponential(c["front_hours"])) + 0.25
                self._cloud_target = float(self.rng.uniform(0.8, 1.0))
            else:
                self._precip_target = 0.0
                dry_mean = c["front_hours"] * (1.0 - c["wet_fraction"]) / max(0.05, c["wet_fraction"])
                self._front_hours_left = float(self.rng.exponential(dry_mean)) + 0.25
                self._cloud_target = float(self.rng.uniform(0.0, 0.55))
        precip_target = self._precip_target if self.forced_precip is None else self.forced_precip
        cloud_target = self._cloud_target if self.forced_cloud is None else self.forced_cloud
        if self.forced_precip is not None and self.forced_cloud is None:
            cloud_target = max(cloud_target, 0.85 if self.forced_precip > 0.05 else 0.0)

        # Smooth transitions (~15 game minutes time constant)
        k = 1.0 - math.exp(-hours / 0.25)
        precip = _lerp(self.rain_intensity + self.snow_intensity, precip_target, k)
        if self.forced_precip is not None and hours == 0.0:
            precip = self.forced_precip
        self.cloud_cover = _lerp(self.cloud_cover, cloud_target, 1.0 - math.exp(-hours / 0.4))

        # Phase of precipitation depends on temperature
        if self.ambient_temp_c <= 0.5:
            self.snow_intensity = precip
            self.rain_intensity = 0.0
        else:
            self.rain_intensity = precip
            self.snow_intensity = 0.0

        # --- snow pack ---
        if self.snow_intensity > 0.02:
            self.snow_accumulation_m = min(1.2, self.snow_accumulation_m + 0.012 * self.snow_intensity * hours)
        if self.ambient_temp_c > 0.5 and self.snow_accumulation_m > 0.0:
            melt = 0.0025 * (self.ambient_temp_c - 0.5) * hours * (1.0 + self.solar_radiation_w_m2 / 600.0)
            self.snow_accumulation_m = max(0.0, self.snow_accumulation_m - melt)

        # --- ground wetness ---
        if self.rain_intensity > 0.02:
            self.ground_wetness = min(1.0, self.ground_wetness + 0.6 * self.rain_intensity * hours)
        else:
            dry_rate = 0.02 + 0.00012 * max(0.0, self.solar_radiation_w_m2) + 0.004 * max(0.0, self.ambient_temp_c)
            self.ground_wetness = max(0.05 if self.current_season != Season.SUMMER else 0.0,
                                      self.ground_wetness - dry_rate * hours)
        if self.snow_accumulation_m > 0.0 and self.ambient_temp_c > 0.5:
            self.ground_wetness = min(1.0, self.ground_wetness + 0.05 * hours)

        # --- wind with gusts (smooth noise) ---
        self._gust_phase += dt_seconds * 0.35
        g = (math.sin(self._gust_phase * 1.3) * 0.5 + math.sin(self._gust_phase * 2.9 + 1.7) * 0.3
             + math.sin(self._gust_phase * 0.47 + 0.3) * 0.2)
        self.wind_gust = g
        if self.rng.random() < min(1.0, hours / 3.0):
            self._wind_dir_target = (self._wind_dir_target + float(self.rng.normal(0.0, 0.5))) % (2 * math.pi)
        d = (self._wind_dir_target - self.wind_direction_rad + math.pi) % (2 * math.pi) - math.pi
        self.wind_direction_rad = (self.wind_direction_rad + d * (1.0 - math.exp(-hours / 1.5))) % (2 * math.pi)
        base = c["wind"] * (1.0 + 0.6 * (self.rain_intensity + self.snow_intensity))
        if self.forced_wind is not None:
            base = self.forced_wind
        self.wind_speed_m_s = max(0.2, min(12.0, base * (1.0 + 0.35 * g)))
        self.wind_vector = np.array([math.cos(self.wind_direction_rad), 0.0,
                                     math.sin(self.wind_direction_rad)]) * self.wind_speed_m_s

        # Autumn leaf fall (temporary dry tinder, visual particles)
        self.falling_leaves = 1.0 if self.current_season == Season.AUTUMN else 0.0

    def _update_parameters(self, dt_seconds: float):
        c = self.climate
        # Diurnal temperature: minimum ~05:00, maximum ~15:00
        diurnal = math.sin((self.time_of_day_hours - 9.0) * (2.0 * math.pi / 24.0))
        # Seasonal drift within the season toward the next one's climate
        nxt = CLIMATE[SEASON_ORDER[(SEASON_ORDER.index(self.current_season) + 1) % 4]]
        prog = self.season_progress()
        drift = max(0.0, prog - 0.7) / 0.3 * 0.5
        base = _lerp(c["base_temp"], nxt["base_temp"], drift)
        amp = c["temp_amp"] * (1.0 - 0.55 * self.cloud_cover)
        precip = self.rain_intensity + self.snow_intensity
        self.ambient_temp_c = base + diurnal * amp - 2.5 * precip

        el = self.sun_elevation_azimuth()[0]
        clear_sky = c["solar"] * max(0.0, math.sin(el)) ** 1.15
        self.solar_radiation_w_m2 = clear_sky * (1.0 - 0.75 * self.cloud_cover ** 3.0)

        rh = c["rh"] - 0.12 * diurnal
        # falling rain/snow saturates the air near the ground (RH ~ 90-98 % in steady rain)
        rh = _lerp(rh, 0.96, min(1.0, 1.2 * precip))
        self.relative_humidity = max(0.15, min(1.0, rh))

        # Morning mist / fog: humid, calm, cool mornings, plus rain haze
        morning = math.exp(-((self.time_of_day_hours - 6.0) / 1.8) ** 2)
        calm = max(0.0, 1.0 - self.wind_speed_m_s / 4.0)
        self.fog_density = max(0.0, min(1.0, 0.9 * morning * calm * max(0.0, self.relative_humidity - 0.55) / 0.45
                                        + 0.45 * precip))
        if self.forced_fog is not None:
            self.fog_density = self.forced_fog

    # ------------------------------------------------------------------ helpers / scripted control
    def set_time(self, hour: float, season: Optional[Season] = None, day: Optional[float] = None):
        self.time_of_day_hours = hour % 24.0
        if season is not None:
            self.current_season = season
        if day is not None:
            self.day_of_season = max(1.0, min(float(DAYS_PER_SEASON), day))
        if self.current_season == Season.WINTER and self.snow_accumulation_m < 0.05:
            self.snow_accumulation_m = 0.25
        if self.current_season == Season.SUMMER:
            self.snow_accumulation_m = 0.0
        self._update_parameters(0.0)

    def force_precipitation(self, intensity: Optional[float]):
        self.forced_precip = intensity
        if intensity is not None:
            if self.ambient_temp_c <= 0.5:
                self.snow_intensity, self.rain_intensity = intensity, 0.0
            else:
                self.rain_intensity, self.snow_intensity = intensity, 0.0
            self.cloud_cover = max(self.cloud_cover, 0.85 if intensity > 0.05 else 0.0)
        self._update_parameters(0.0)

    def is_freezing(self) -> bool:
        return self.ambient_temp_c < 0.0

    def sky_temperature_c(self) -> float:
        """Effective radiative sky temperature (clear nights radiate to a cold sky)."""
        return self.ambient_temp_c - 20.0 * (1.0 - self.cloud_cover) - 2.0

    def forecast(self, hours_ahead: int = 12) -> List[Dict[str, Any]]:
        """Cheap deterministic-ish forecast for the phone's home-screen widget."""
        out = []
        rain_left = self._front_hours_left
        wet = self._precip_target > 0.05
        for h in range(0, hours_ahead + 1, 3):
            hour = (self.time_of_day_hours + h) % 24.0
            diurnal = math.sin((hour - 9.0) * (2.0 * math.pi / 24.0))
            c = self.climate
            t = c["base_temp"] + diurnal * c["temp_amp"]
            state_wet = wet if h < rain_left else (not wet and c["wet_fraction"] > 0.35)
            out.append({"hour": hour, "temp": t, "precip": state_wet,
                        "snow": state_wet and t <= 0.5})
        return out

    def describe(self) -> str:
        if self.snow_intensity > 0.6:
            return "Blizzard"
        if self.snow_intensity > 0.05:
            return "Snowfall"
        if self.rain_intensity > 0.6:
            return "Heavy rain"
        if self.rain_intensity > 0.05:
            return "Light rain"
        if self.fog_density > 0.45:
            return "Mist"
        if self.cloud_cover > 0.7:
            return "Overcast"
        if self.cloud_cover > 0.35:
            return "Partly cloudy"
        return "Clear" if self.daylight_factor() < 0.5 else "Sunny"
