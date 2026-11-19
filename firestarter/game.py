"""
Firestarter - Game application (window, main loop, subsystems wiring).

Implements the DESIGN Section 8 architecture: main -> Game -> {RenderEngine, WorldManager,
InputManager} with the simulation subsystems underneath.  Physics runs at a fixed 60 Hz and
the thermal grid at 20 Hz inside WorldManager.update(); rendering runs as fast as the
display allows (vsync) or at a fixed step during scripted playback.
"""

import os
import sys
import time
import math
import json
from typing import Optional, Dict, Any, List

import numpy as np

from firestarter.config import WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, START_HOUR, DEFAULT_SEED
from firestarter.audio.procedural_audio import ProceduralAudioEngine
from firestarter.audio.asmr_foley import ASMRFoleyManager
from firestarter.simulation.world_manager import WorldManager
from firestarter.economy.smartphone import SmartphoneOS
from firestarter.recipes.reference_book import ReferenceBook
from firestarter.input.hand_physics_rig import HandController
from firestarter.input.input_manager import InputManager, Settings
from firestarter.input.tutorial import Tutorial

ROOT = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(ROOT, "assets", "audio")


def _window_icon(pygame):
    """A small procedural flame for the window / taskbar icon."""
    from PIL import Image, ImageDraw, ImageFilter
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([14, 50, 50, 60], fill=(60, 40, 30, 255))
    d.polygon([(32, 4), (46, 30), (48, 44), (40, 54), (24, 54), (16, 44), (18, 30), (26, 20)], fill=(235, 110, 30, 255))
    d.polygon([(33, 20), (41, 36), (40, 48), (32, 53), (25, 47), (25, 36)], fill=(255, 196, 70, 255))
    d.polygon([(32, 34), (36, 44), (32, 51), (28, 44)], fill=(255, 240, 190, 255))
    img = img.filter(ImageFilter.SMOOTH)
    return pygame.image.frombuffer(img.tobytes(), img.size, "RGBA")


class GameState:
    """Bundle of the live subsystems (handed to the tutorial, playback scripts and tests)."""

    def __init__(self, seed: int = DEFAULT_SEED, start_hour: float = START_HOUR, tutorial: bool = True):
        self.world = WorldManager(seed=seed, start_hour=start_hour)
        self.phone = SmartphoneOS()
        self.book = ReferenceBook()
        self.world.reference_book = self.book
        self.world.credits = self.phone.credits
        self.hands = HandController()
        self.settings = Settings()
        self.input = InputManager(self.world, self.phone, self.book, self.hands, self.settings)
        self.tutorial = Tutorial(self)
        if not tutorial:
            self.tutorial.skip()
        self.input.tutorial = self.tutorial
        self.time = 0.0

    def step(self, dt: float, move: bool = True):
        """Advance everything except rendering / audio by dt seconds."""
        w = self.world
        p = w.player
        if move:
            f, s = self.input.movement_axes()
            p.move(f, s, dt)
        w.update(dt)
        self.tutorial.update(dt, w.sim_time)
        self.phone.update(dt)
        g = w.thermal_grid
        self.phone.credits.update(dt, g.core_coal_temp, g.current_heat_flux, efficiency_ratio=g.efficiency_ratio)
        cf = w.crafting
        nest = cf.active_tinder_nest
        ctx = {
            "right_item": p.primary_hand,
            "left_item": "nest" if (nest is not None and nest.has_ember) else (
                w.current_offhand() if ((w.current_offhand() == "fungus" and w.inventory.fungi) or
                                        (w.current_offhand() != "fungus" and w.inventory.count(w.current_offhand()) > 0)) else "none"),
            "lighter_lit": cf.lighter_is_lit,
            "walk_phase": p.bob_phase,
            "walk_amount": p.bob_amount,
            "crouch": 1.0 if p.is_crouching else 0.0,
        }
        if w.lighter_point is not None:
            # reach target in camera space
            from firestarter.rendering import gl_utils as G
            view = G.look_at(p.eye, p.eye + p.get_forward_ray())
            rel = w.lighter_point - p.eye
            ctx["reach_target"] = np.array([np.dot(view[0, :3], rel), np.dot(view[1, :3], rel), -np.dot(view[2, :3], rel)])
        if w.bow_active:
            act = "hand_drilling" if w.friction_mode == "hand" else "bow_drilling"
            hd = cf.hand_drill
            ctx["hand_height"] = hd.hand_height_cm / max(1.0, hd.spindle_length_cm * 0.9)
            ctx["drill_speed"] = w.bow_input_speed
            # where the hands work above the notch, in view-model space: the palms slide down the
            # hand-drill spindle; the bow drill's bearing block caps a ~25 cm spindle
            from firestarter.rendering import gl_utils as G
            from firestarter.rendering.render_engine import FOV_DEGREES, VM_FOV
            view = G.look_at(p.eye, p.eye + p.get_forward_ray())
            k = math.tan(math.radians(FOV_DEGREES) * 0.5) / math.tan(math.radians(VM_FOV) * 0.5)
            h = (0.12 + 0.3 * ctx["hand_height"]) if w.friction_mode == "hand" else 0.27
            grip = w.hearthboard_pos + np.array([0.05, h, 0.03])
            rel = grip - p.eye
            cs = np.array([np.dot(view[0, :3], rel) / k, np.dot(view[1, :3], rel) / k, -np.dot(view[2, :3], rel)])
            dist = float(np.linalg.norm(cs))
            if dist > 0.52:
                cs = cs / dist * 0.52     # arms only reach so far
            ctx["drill_grip"] = cs
        else:
            act = cf.active_task
        self.hands.update(dt, act, ctx)
        self.time += dt

    def snapshot(self) -> Dict[str, Any]:
        w = self.world
        g = w.thermal_grid
        return {
            "time": round(self.time, 2),
            "clock": w.weather.clock_string(),
            "season": w.weather.current_season.value,
            "fire_stage": g.fire_stage(),
            "core_temp": round(g.core_coal_temp, 1),
            "hrr_kw": round(g.current_heat_flux / 1000.0, 2),
            "flame_height": round(g.flame_height, 3),
            "bodies": len(g.bodies),
            "credits": round(self.phone.credits.credits, 2),
            "pending": round(self.phone.credits.pending, 2),
            "tutorial_step": self.tutorial.index,
            "inventory": w.inventory.summary(),
            "structure": w.structure_key,
        }


class Game:
    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT, seed: int = DEFAULT_SEED,
                 preset: str = "High", fullscreen: bool = False, audio: bool = True, start_hour: float = START_HOUR,
                 tutorial: bool = True, fixed_dt: Optional[float] = None, vsync: bool = True):
        self.width, self.height = width, height
        self.fixed_dt = fixed_dt
        self.running = False
        self.frame = 0
        self.screenshots: List[str] = []
        self.recorder = None
        self.playback = None
        self.perf: List[float] = []

        if audio:
            try:
                if not ProceduralAudioEngine.assets_up_to_date(AUDIO_DIR):
                    print("[Audio] Synthesising procedural ASMR audio (first launch)...")
                    n = ProceduralAudioEngine.generate_and_save_all_assets(AUDIO_DIR)
                    print(f"[Audio] {n} sounds written to assets/audio")
            except Exception as e:   # pragma: no cover
                print(f"[Audio] Audio synthesis skipped: {e}")

        if not audio:
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        try:
            pygame.display.set_icon(_window_icon(pygame))
        except Exception:
            pass
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
        try:
            self.screen = pygame.display.set_mode((width, height), flags, vsync=1 if (vsync and fixed_dt is None) else 0)
        except TypeError:
            self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption(WINDOW_TITLE)
        self.width, self.height = pygame.display.get_surface().get_size()

        self.state = GameState(seed=seed, start_hour=start_hour, tutorial=tutorial)
        self.state.settings.graphics_preset = preset
        self.foley = ASMRFoleyManager(asset_dir=AUDIO_DIR) if audio else ASMRFoleyManager(asset_dir="/nonexistent")
        from firestarter.rendering.render_engine import RenderEngine
        self.renderer = RenderEngine(self.width, self.height, preset=preset)
        if not self.renderer.is_gl_available:
            raise RuntimeError(f"OpenGL 3.3 renderer failed to initialise:\n{self.renderer.init_error}")
        self.clock = pygame.time.Clock()
        self.mouse_captured = False
        self._set_capture(True)

    # ------------------------------------------------------------------ mouse capture
    def _set_capture(self, on: bool):
        pg = self.pg
        if on == self.mouse_captured:
            return
        self.mouse_captured = on
        try:
            pg.mouse.set_visible(not on)
            pg.event.set_grab(on)
            if hasattr(pg.mouse, "set_relative_mode"):
                pg.mouse.set_relative_mode(on)
        except Exception:
            pass

    # ------------------------------------------------------------------ events
    def handle_event(self, event):
        pg = self.pg
        inp = self.state.input
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == getattr(pg, "WINDOWFOCUSLOST", -1) and self.playback is None and self.recorder is None:
            # alt-tabbing away pauses the game (never during scripted playback)
            if not inp.in_menu:
                inp.handle_key_down("escape")
                inp.handle_key_up("escape")
        elif event.type == pg.VIDEORESIZE:
            self.width, self.height = max(320, event.w), max(200, event.h)
            self.renderer.handle_resize(self.width, self.height)
        elif event.type == pg.KEYDOWN:
            name = pg.key.name(event.key)
            if name == "f11":
                pg.display.toggle_fullscreen()
                return
            if name == "f12":
                self.save_screenshot(os.path.join(ROOT, "screenshots", f"shot_{int(time.time())}.png"))
                return
            inp.handle_key_down(name)
        elif event.type == pg.KEYUP:
            inp.handle_key_up(pg.key.name(event.key))
        elif event.type == pg.MOUSEBUTTONDOWN:
            inp.handle_mouse_button_down(event.button)
        elif event.type == pg.MOUSEBUTTONUP:
            inp.handle_mouse_button_up(event.button)
        elif event.type == pg.MOUSEWHEEL:
            # pygame 2 delivers wheel as its own event as well as buttons 4/5 on some platforms
            pass
        elif event.type == pg.MOUSEMOTION:
            if self.mouse_captured or getattr(event, "synthetic", False):
                dx, dy = event.rel
                inp.handle_mouse_motion(dx, dy)
        if self.recorder is not None:
            self.recorder.record(self.frame, event)

    # ------------------------------------------------------------------ loop
    def tick(self, dt: float, render: bool = True):
        st = self.state
        pg = self.pg
        for event in pg.event.get():
            self.handle_event(event)
        inp = st.input
        self._set_capture(not inp.in_menu)
        if inp.quit_requested:
            self.running = False
        # apply settings
        self.foley.set_volumes(st.settings.master_volume, st.settings.ambience_volume, st.settings.effects_volume)
        if st.settings.graphics_preset != self.renderer.preset_name:
            self.renderer.set_preset(st.settings.graphics_preset)
        st.world.clock_scale = st.settings.clock_speed
        paused = inp.is_paused
        # the pause menu freezes everything: simulation, clock, particles, animation and audio
        self.foley.set_paused(paused)
        if not paused:
            st.step(dt)
        events = st.world.pop_events()
        self.renderer.push_events(events)
        self.foley.handle_events(events, st.world)
        for cue in st.world.crafting.pop_audio():
            self.foley.play(cue)
        if not paused:
            self.foley.update_world(st.world, dt)
        if render:
            t0 = time.perf_counter()
            self.renderer.render(st.world, st.world.player, st.hands, st.phone, st.book, inp, dt=0.0 if paused else dt)
            self.perf.append((time.perf_counter() - t0) * 1000.0)
            self.perf = self.perf[-600:]
            if self.playback is not None:
                self.playback.after_render(self)
            pg.display.flip()
        self.frame += 1

    def run(self, max_frames: Optional[int] = None) -> int:
        self.running = True
        print("[Game] Firestarter running.  Press H for controls, Esc to pause.")
        while self.running:
            if self.fixed_dt is not None:
                dt = self.fixed_dt
                self.clock.tick(0)
            else:
                dt = min(0.1, self.clock.tick(120) / 1000.0)
            if self.playback is not None:
                if not self.playback.before_frame(self, dt):
                    self.running = False
                    break
            self.tick(dt, render=self.playback.render_now if self.playback is not None else True)
            if max_frames is not None and self.frame >= max_frames:
                break
        if self.recorder is not None:
            self.recorder.save()
        self.foley.shutdown()
        code = 0
        if self.playback is not None:
            code = self.playback.finish(self)
        self.pg.quit()
        return code

    # ------------------------------------------------------------------ capture
    def save_screenshot(self, path: str) -> str:
        from PIL import Image
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        img = self.renderer.read_pixels()
        Image.fromarray(img).save(path)
        self.screenshots.append(path)
        return path
