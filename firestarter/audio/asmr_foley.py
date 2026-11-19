"""
Firestarter - ASMR Foley Manager / live mixer (DESIGN Section 9)

* Ambience beds on reserved channels whose volumes track the simulation every frame:
  fire roar (heat release & draft), fine crackle, steam hiss (moisture boiling out),
  ember tinkle (glowing coals), wind (speed, muffled in snow), day birdsong / night crickets
  & owls, rain (and rain drumming on the tarp near the rack) and the creek (proximity).
* Fire sounds are spatialised: distance attenuation and stereo pan relative to where the
  player is looking.
* Discrete events (resin pops, lighter strikes, chops, footsteps, UI...) trigger one-shots.

If no audio device is available (CI, headless) everything degrades to a silent no-op.
"""

import math
import os
from typing import Dict, Any, Optional, List

import numpy as np

# ~5 dB of headroom on every voice: a roaring fire, crackles, rain and wind summed together
# must never clip the master bus
MIX_HEADROOM = 0.55

LOOP_CHANNELS = ["loop_fire_roar", "loop_fire_crackle", "loop_steam_hiss", "loop_embers", "loop_wind",
                 "loop_wind_snow", "loop_forest_day", "loop_night", "loop_rain", "loop_rain_tarp", "loop_creek"]


class ASMRFoleyManager:
    def __init__(self, asset_dir: str = "firestarter/assets/audio", num_channels: int = 40, seed: int = 5):
        self.asset_dir = asset_dir
        self.sounds: Dict[str, Any] = {}
        self.is_pygame_audio_available = False
        self.loop_channels: Dict[str, Any] = {}
        self.loop_levels: Dict[str, List[float]] = {}
        self.master = 0.8
        self.ambience = 0.8
        self.effects = 0.9
        self.rng = np.random.default_rng(seed)
        self.num_channels = num_channels
        self.played: List[str] = []           # log of one-shots (for tests)
        self.levels: Dict[str, float] = {}     # latest bed levels (for tests / HUD)
        self._crackle_accum = 0.0
        self.paused = False
        self._squeak_timer = 0.0
        self._init_audio()

    def _init_audio(self):
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(44100, -16, 2, 1024)
                pygame.mixer.init(44100, -16, 2, 1024)
            pygame.mixer.set_num_channels(self.num_channels)
            self._pg = pygame
            for fname in sorted(os.listdir(self.asset_dir)):
                if fname.endswith(".wav"):
                    self.sounds[fname[:-4]] = pygame.mixer.Sound(os.path.join(self.asset_dir, fname))
            pygame.mixer.set_reserved(len(LOOP_CHANNELS))
            for i, name in enumerate(LOOP_CHANNELS):
                if name in self.sounds:
                    ch = pygame.mixer.Channel(i)
                    ch.play(self.sounds[name], loops=-1)
                    ch.set_volume(0.0, 0.0)
                    self.loop_channels[name] = ch
                    self.loop_levels[name] = [0.0, 0.0]
            self.is_pygame_audio_available = True
        except Exception:
            self.is_pygame_audio_available = False

    UI_SOUNDS = ("ui_click", "ui_error", "page_turn", "phone_open")

    def set_paused(self, paused: bool):
        """Pause menu: freeze every playing sound (fire, weather, birds, foley).  Only the menu's
        own clicks are heard while paused; everything resumes exactly where it stopped."""
        if paused == self.paused:
            return
        self.paused = paused
        if not self.is_pygame_audio_available:
            return
        try:
            if paused:
                self._pg.mixer.pause()
            else:
                self._pg.mixer.unpause()
        except Exception:
            pass

    def set_volumes(self, master: float, ambience: float, effects: float):
        self.master, self.ambience, self.effects = master, ambience, effects

    # ------------------------------------------------------------------ one-shots
    def play(self, sound_name: str, volume: float = 1.0, pan: float = 0.0):
        if self.paused and not sound_name.startswith(self.UI_SOUNDS):
            return
        self.played.append(sound_name)
        self.played = self.played[-200:]
        if not self.is_pygame_audio_available:
            return
        name = sound_name
        if name not in self.sounds:
            variants = [k for k in self.sounds if k.startswith(name + "_")]
            if not variants:
                return
            name = variants[int(self.rng.integers(len(variants)))]
        snd = self.sounds[name]
        ch = self._pg.mixer.find_channel(True)
        if ch is None:
            return
        v = max(0.0, min(1.0, volume * self.effects * self.master * MIX_HEADROOM))
        l = v * math.sqrt(0.5 * (1.0 - pan))
        r = v * math.sqrt(0.5 * (1.0 + pan))
        ch.play(snd)
        ch.set_volume(min(1.0, l * 1.41), min(1.0, r * 1.41))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pan_and_gain(listener_pos, listener_yaw_deg, source_pos, ref: float = 1.5):
        d = np.asarray(source_pos, dtype=float) - np.asarray(listener_pos, dtype=float)
        dist = float(np.linalg.norm(d[[0, 2]])) + 0.3
        yaw = math.radians(listener_yaw_deg)
        right = np.array([-math.sin(yaw), math.cos(yaw)])
        pan = float(np.clip(np.dot(d[[0, 2]] / dist, right), -1.0, 1.0)) * 0.8
        gain = 1.0 / (1.0 + (dist / ref) ** 1.4)
        return pan, gain, dist

    def _set_loop(self, name: str, level: float, pan: float = 0.0, dt: float = 0.016, slew: float = 3.0):
        cur = self.loop_levels.get(name)
        if cur is None:
            return
        level = max(0.0, min(1.0, level))
        g = level * MIX_HEADROOM
        target_l = g * math.sqrt(0.5 * (1.0 - pan)) * 1.41
        target_r = g * math.sqrt(0.5 * (1.0 + pan)) * 1.41
        k = min(1.0, dt * slew)
        cur[0] += (target_l - cur[0]) * k
        cur[1] += (target_r - cur[1]) * k
        self.levels[name] = level
        ch = self.loop_channels.get(name)
        if ch is not None:
            ch.set_volume(min(1.0, cur[0]), min(1.0, cur[1]))

    # ------------------------------------------------------------------ per-frame mix
    def update_world(self, world: Any, dt: float):
        if self.paused:
            return
        g = world.thermal_grid
        p = world.player
        w = world.weather
        amb = self.ambience * self.master
        fx = self.effects * self.master
        lp = p.eye if hasattr(p, "eye") else p.position
        pan, gain, dist = self._pan_and_gain(lp, p.yaw, g.fire_centroid, 1.8)
        flame_kw = g.flame_hrr / 1000.0
        glow_kw = g.char_hrr / 1000.0
        roar = min(1.0, 0.18 * math.sqrt(max(0.0, flame_kw)) * (0.7 + 0.3 * min(2.0, g.draft_velocity)))
        crackle = min(1.0, 0.22 * math.sqrt(max(0.0, flame_kw)) + 0.05 * math.sqrt(max(0.0, glow_kw)))
        hiss = min(1.0, g.steam_rate * 900.0)
        embers = min(1.0, 0.25 * math.sqrt(max(0.0, glow_kw)) + (0.2 if g.coal_bed.T > 500 and g.coal_bed.char_kg > 0.02 else 0.0))
        self._set_loop("loop_fire_roar", roar * gain * fx, pan, dt)
        self._set_loop("loop_fire_crackle", crackle * gain * fx * 0.8, pan, dt)
        self._set_loop("loop_steam_hiss", hiss * gain * fx * 0.7, pan, dt, slew=6.0)
        self._set_loop("loop_embers", embers * gain * fx * 0.6, pan, dt)
        # discrete crackles: rate ~ flame size x resin
        rate = 1.2 * math.sqrt(max(0.0, flame_kw)) + 0.15 * math.sqrt(max(0.0, glow_kw))
        self._crackle_accum += rate * dt
        while self._crackle_accum >= 1.0:
            self._crackle_accum -= 1.0
            if self.rng.random() < 0.6:
                self.play("resin_pop", float(self.rng.uniform(0.15, 0.45)) * gain, pan + float(self.rng.uniform(-0.2, 0.2)))
        for pos in g.pop_events:
            pp, gg, _ = self._pan_and_gain(lp, p.yaw, pos, 1.8)
            self.play("resin_pop", float(self.rng.uniform(0.5, 0.9)) * gg, pp)
        # weather & nature beds
        snowy = w.snow_accumulation_m > 0.05 or w.snow_intensity > 0.05
        wind_l = min(1.0, 0.08 + w.wind_speed_m_s / 9.0)
        self._set_loop("loop_wind", wind_l * amb * (0.0 if snowy else 0.7), 0.0, dt, 0.8)
        self._set_loop("loop_wind_snow", wind_l * amb * (0.8 if snowy else 0.0), 0.0, dt, 0.8)
        day = w.daylight_factor()
        quiet = 1.0 - min(1.0, w.rain_intensity * 1.5 + w.snow_intensity)
        birds = day * quiet * (0.0 if w.current_season.value == "Winter" else 0.55)
        night = (1.0 - day) * quiet * {"Summer": 0.6, "Spring": 0.4, "Autumn": 0.25, "Winter": 0.0}[w.current_season.value]
        self._set_loop("loop_forest_day", birds * amb, 0.0, dt, 0.5)
        self._set_loop("loop_night", night * amb, 0.0, dt, 0.5)
        self._set_loop("loop_rain", min(1.0, w.rain_intensity * 0.9) * amb, 0.0, dt, 1.0)
        rp, rg, rd = self._pan_and_gain(lp, p.yaw, world.drying_rack.position, 2.5)
        tarp = w.rain_intensity * rg if world.drying_rack.has_tarp else 0.0
        self._set_loop("loop_rain_tarp", min(1.0, tarp) * amb, rp, dt, 1.0)
        from firestarter.simulation import terrain
        cd = terrain.creek_distance(lp[0], lp[2])
        creek = math.exp(-max(0.0, cd - 2.0) / 14.0) * 0.7
        self._set_loop("loop_creek", creek * amb, 0.0, dt, 1.0)
        # footsteps
        for ev in p.pop_step_events():
            surf = "water" if ev == "water" else ("snow" if snowy else "leaves")
            self.play(f"step_{surf}", 0.25 if not p.is_crouching else 0.12)
        # bow drill squeak rhythm
        if world.bow_active and world.bow_input_speed > 0.15:
            self._squeak_timer -= dt
            if self._squeak_timer <= 0:
                self._squeak_timer = 0.34
                self.play("friction_squeak", min(0.7, 0.2 + world.bow_input_speed * 0.3))

    def handle_events(self, events, world: Any = None):
        p = world.player if world is not None else None
        for name, data in events:
            pan, gain = 0.0, 1.0
            if p is not None and isinstance(data, np.ndarray) and data.shape == (3,):
                pan, gain, _ = self._pan_and_gain(p.eye, p.yaw, data, 3.0)
            if name == "lighter":
                self.play("lighter_strike", 0.8)
            elif name == "chop":
                self.play("chop", 0.9 * max(0.4, gain), pan)
            elif name == "tree_creak":
                self.play("tree_creak", 0.8 * max(0.3, gain), pan)
            elif name == "tree_fall":
                self.play("tree_fall", 1.0 * max(0.4, gain), pan)
            elif name == "collapse":
                self.play("collapse", 0.7 * gain, pan)
            elif name == "gas_flash":
                self.play("gas_whoosh", 1.0)
            elif name == "breath":
                v = data.get("velocity", 2.0) if isinstance(data, dict) else 2.0
                self.play("breath_soft" if v < 1.3 else ("breath_med" if v < 4.5 else "breath_hard"), 0.7)
            elif name in ("pickup",):
                self.play("rustle", 0.6)
            elif name in ("pickup_wood", "place_wood"):
                self.play("wood_knock", 0.6 * max(0.5, gain), pan)
            elif name == "place_tinder":
                self.play("rustle", 0.5)
            elif name == "knife":
                self.play("whittle", 0.7)
            elif name == "harvest":
                self.play("harvest", 0.7)
            elif name == "ember":
                self.play("ember", 0.8)
            elif name == "nest_flame":
                self.play("whoosh", 0.5)
                self.play("resin_pop", 0.5)
            elif name == "page_turn":
                self.play("page_turn", 0.6)
            elif name in ("ui_click", "equip"):
                self.play("ui_click", 0.5)
            elif name == "ui_error":
                self.play("ui_error", 0.6)
            elif name == "phone":
                if data:
                    self.play("phone_open", 0.5)
            elif name == "purchase":
                self.play("purchase", 0.6)
            elif name == "credits":
                self.play("credits", 0.6)
            elif name == "tarp":
                self.play("tarp", 0.7)
            elif name == "pour":
                self.play("pour", 0.7)
            elif name == "time_skip":
                self.play("whoosh", 0.6)
            elif name == "book":
                self.play("page_turn", 0.5)
            elif name == "swing":
                self.play("whoosh", 0.3)

    # ------------------------------------------------------------------ legacy API
    def update_continuous_ambience(self, core_temp_c: float, draft_velocity: float, steam_rate_kg_s: float,
                                   rain_intensity: float, wind_speed: float, dt: float = 0.016):
        roar = min(1.0, max(0.0, (core_temp_c - 150.0) / 600.0) * (0.5 + 0.5 * min(2.0, draft_velocity)))
        self._set_loop("loop_fire_roar", roar * self.effects * self.master, 0.0, dt)
        self._set_loop("loop_steam_hiss", min(1.0, steam_rate_kg_s * 900.0) * self.effects * self.master, 0.0, dt)
        self._set_loop("loop_rain", min(1.0, rain_intensity) * self.ambience * self.master, 0.0, dt)
        self._set_loop("loop_wind", min(1.0, wind_speed / 9.0) * self.ambience * self.master, 0.0, dt)

    def shutdown(self):
        if self.is_pygame_audio_available:
            try:
                self._pg.mixer.stop()
            except Exception:
                pass
