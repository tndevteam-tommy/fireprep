"""Pygame window, screens (start menu / playing / pause / graphics options) and the main loop.

Everything here is glue between pygame and ``Game``; the game never knows a
menu exists.
"""
from __future__ import annotations

import dataclasses
import os
import sys
from enum import Enum
from typing import Optional, Tuple

import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from .config import DEFAULT_SETTINGS, PROJECT_DIR, GameSettings
from .game import Game
from .graphics import AA_STEPS, ANISOTROPY_STEPS, DRAW_DISTANCE_RANGE, MAX_SHADER_LIGHTS, GraphicsSettings, graphics_settings_path
from .attract import TitleView
from .input import DEFAULT_BINDINGS, KeyBindings, PygameInputController
from .paths import migrate_legacy_saves
from .perf import FrameProfiler, recommend_quality
from .rendering import Renderer
from .rendering.hud import MenuRenderer
from .ui import Menu, MenuItem, slider_text
from .world import LevelDefinition


DEPTH_BITS = 24        # SDL asks for a 16-bit depth buffer unless told otherwise


def desktop_size() -> Tuple[int, int]:
    """The (first) screen's resolution - what a full-screen window takes."""
    try:
        sizes = pygame.display.get_desktop_sizes()
        if sizes:
            return tuple(sizes[0])
    except (AttributeError, pygame.error):
        pass
    info = pygame.display.Info()
    return (info.current_w, info.current_h)


def open_gl_window(width: int, height: int, fullscreen: bool = False, samples: int = 0, vsync: bool = False) -> int:
    """Open the OpenGL window with a 24-bit depth buffer; returns the depth bits obtained.

    SDL's default request is 16 bits, which many drivers honour. With the near plane at 0.1
    that resolves depth to about 1 cm at 5 units but to ~0.4 units at 35 and over a unit at 70:
    everything a character is made of (face, hair, eyes, helmet, belt) and every layered surface
    of a distant building then fights for the same depth values and shows as flickering streaks.
    24 bits make that 256 times finer. A driver that cannot provide 24 gets the default.

    ``fullscreen`` takes the whole screen (``width`` x ``height`` should be its resolution, see
    ``desktop_size``); ``samples`` asks for multisampled anti-aliasing and ``vsync`` for swaps in
    step with the screen - whatever the driver will not do is dropped and the window opened
    without it, never left unopened."""
    flags = DOUBLEBUF | OPENGL | (pygame.FULLSCREEN if fullscreen else 0)
    attempts = []
    for depth in (DEPTH_BITS, 16):
        for aa in ((samples, 0) if samples > 0 else (0,)):
            attempts.append((depth, aa))
    last_error = None
    for depth, aa in attempts:
        try:
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, depth)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1 if aa else 0)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, aa)
            try:
                pygame.display.set_mode((width, height), flags, vsync=1 if vsync else 0)
            except (TypeError, pygame.error):
                if not vsync:
                    raise
                pygame.display.set_mode((width, height), flags)     # (no vsync to be had: without it)
            break
        except pygame.error as error:
            last_error = error
    else:
        raise last_error
    try:
        return int(pygame.display.gl_get_attribute(pygame.GL_DEPTH_SIZE))
    except pygame.error:
        return 0


class Screen(Enum):
    START = "start"
    PLAYING = "playing"
    PAUSE = "pause"
    GRAPHICS = "graphics"
    SAVE = "save"          # pick a slot to save into
    LOAD = "load"          # pick a save to load
    CONFIRM = "confirm"    # a yes/no question (overwrite a save, exit to menu)


class PygameApp:
    BENCHMARK_FRAMES = 40

    def __init__(self, settings: GameSettings = DEFAULT_SETTINGS, level: Optional[LevelDefinition] = None,
                 bindings: KeyBindings = DEFAULT_BINDINGS, save_file: Optional[str] = None,
                 graphics: Optional[GraphicsSettings] = None, skip_menu: bool = False, auto_quality: bool = True):
        self.settings = settings
        self.level = level
        self.save_file = settings.resolved_save_file(save_file)
        if save_file is None and settings.save_file is None:
            # the default (per-user) location: bring along saves an older version kept next to main.py
            migrate_legacy_saves(PROJECT_DIR, os.path.dirname(self.save_file))
        pygame.init()
        pygame.font.init()
        self.graphics_path = graphics_settings_path(self.save_file)
        self.graphics = graphics or GraphicsSettings.load(self.graphics_path) or GraphicsSettings()
        self.windowed_size = (settings.window.width, settings.window.height)
        self._display_mode: Optional[tuple] = None       # (fullscreen, samples, vsync) the window was opened with
        self.settings = settings = self._open_window(settings)
        pygame.display.set_caption(settings.window.title)
        self.profiler = FrameProfiler()
        self._mouse_captured: Optional[bool] = None
        self.game = self._make_game(level=level, settings=settings)
        self.input = PygameInputController(bindings)
        self.renderer = Renderer(settings, bindings.controls_help(), graphics=self.graphics)
        self.menus = MenuRenderer(self.renderer.text, settings.hud_layout(), settings.hud_colors, self.renderer.textures)
        self.clock = pygame.time.Clock()
        self.screen = Screen.PLAYING if skip_menu else Screen.START
        self.in_session = skip_menu            # a game is underway (menus return to it instead of a fresh one)
        self.title: Optional[TitleView] = None  # the start screen's live picture of the town (no game underway)
        self.menu: Optional[Menu] = None
        self._load_return = Screen.START
        self._confirm_cancel = self.open_pause_menu
        self.quit_requested = False
        self.graphics.detected_renderer = self.renderer.renderer_name
        if auto_quality and self.graphics.auto_quality:
            self.auto_configure_quality()
        if not skip_menu:
            self.open_start_menu()
        self._sync_mouse_capture()

    # ------------------------------------------------------------------ quality detection
    def benchmark(self, frames: int = BENCHMARK_FRAMES) -> float:
        """Render the current scene ``frames`` times with the *high* preset; returns the average ms.
        ``self.graphics`` is restored *in place* afterwards: the renderer and an open graphics
        menu hold references to it and must keep seeing the live object."""
        self.show_busy("Measuring the graphics card")
        previous = self.graphics.to_dict()
        self.graphics.apply_preset("high")
        self.rebuild_renderer()
        self.renderer.prebake(self.game)
        profiler = FrameProfiler()
        for _ in range(frames):
            self.renderer.render_scene_only(self.game)
            pygame.display.flip()
            profiler.end_frame()
        self.graphics.update_from_dict(previous)
        self.rebuild_renderer()
        return profiler.average_frame_ms

    def auto_configure_quality(self) -> str:
        ms = self.benchmark()
        quality = recommend_quality(ms, self.renderer.renderer_name)
        self.graphics.apply_preset(quality)
        self.graphics.auto_quality = True
        self.graphics.benchmark_ms = round(ms, 2)
        if not self.apply_display():                      # (a preset's anti-aliasing needs the window opened anew)
            self.rebuild_renderer()
        self.graphics.save(self.graphics_path)
        return quality

    def rebuild_renderer(self) -> None:
        """Recreate GL resources after a change that needs it (shaders / mipmaps toggled)."""
        self.show_busy("Applying the graphics settings")
        self.renderer.release()
        self.renderer = Renderer(self.settings, self.input.bindings.controls_help(), graphics=self.graphics)
        self.menus = MenuRenderer(self.renderer.text, self.settings.hud_layout(), self.settings.hud_colors, self.renderer.textures)

    # ------------------------------------------------------------------ long work
    LONG_TASK_BUDGET = 0.03     # seconds of a frame a long task (hours passing in the cells) may take
    NPC_PLANNING_BUDGET = 0.004  # seconds of a frame for planning (and beginning) the townsfolk's walks

    def _make_game(self, level=None, settings=None, tutorial: bool = False) -> Game:
        """A new ``Game`` that runs its long tasks a frame's budget at a time (under the busy box)."""
        game = Game(level=level if level is not None else self.level, settings=settings or self.settings,
                    save_file=self.save_file, tutorial=tutorial)
        game.long_task_budget = self.LONG_TASK_BUDGET
        # (the walks people set off on at the stroke of the hour: planned ahead in the frames before
        # it and begun a few at a time, instead of a fifth of a second's stall on the hour)
        game.world.planning_budget = game.world.switch_budget = self.NPC_PLANNING_BUDGET
        return game

    def show_busy(self, label: str, detail: str = "") -> None:
        """Put the busy box on screen now, over the last picture, before something that keeps the
        frame from coming round for a while (the town built for a new game, a save read, the scene
        baked again for new graphics settings): the game is seen to be working, not hung."""
        if getattr(self, "menus", None) is None:
            return
        try:
            if not self.renderer.needs_baking(self.game):
                self.renderer.render_scene_only(self.game)      # (the picture as it is, under the box)
            else:
                from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, glClear, glClearColor
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.menus.draw_busy(label, None, detail)
            pygame.display.flip()
            pygame.event.pump()
        except Exception:                                  # (no window to draw on yet: nothing to show)
            pass

    # ------------------------------------------------------------------ the window
    def _wanted_display(self) -> tuple:
        g = self.graphics
        return (bool(g.fullscreen), int(getattr(g, "antialiasing", 0)), bool(getattr(g, "vsync", False)))

    def _open_window(self, settings: GameSettings) -> GameSettings:
        """(Re)open the window as the graphics settings ask - windowed at the configured size or
        full screen at the desktop's resolution, with the anti-aliasing and vsync asked for - and
        return the settings with the window's real size (the HUD lays itself out to it)."""
        fullscreen, samples, vsync = self._wanted_display()
        width, height = desktop_size() if fullscreen else self.windowed_size
        try:
            self.depth_bits = open_gl_window(width, height, fullscreen=fullscreen, samples=samples, vsync=vsync)
        except pygame.error:
            if not fullscreen:
                raise
            self.graphics.fullscreen = False                 # (no full screen to be had: a window after all)
            width, height = self.windowed_size
            self.depth_bits = open_gl_window(width, height, samples=samples, vsync=vsync)
        if fullscreen:
            self._settle_window((width, height))
        width, height = self._window_size() or (width, height)
        self._display_mode = self._wanted_display()
        return dataclasses.replace(settings, window=dataclasses.replace(settings.window, width=int(width), height=int(height)))

    @staticmethod
    def _window_size() -> Optional[Tuple[int, int]]:
        """The window's size as it is now (None if there is no window to ask)."""
        try:
            width, height = pygame.display.get_window_size()
        except (AttributeError, pygame.error):
            surface = pygame.display.get_surface()
            if surface is None:
                return None
            width, height = surface.get_size()
        return (int(width), int(height)) if width > 0 and height > 0 else None

    SETTLE_SECONDS = 0.6         # how long a new full-screen window is given to take the whole screen

    def _settle_window(self, wanted: Tuple[int, int]) -> None:
        """A window manager makes a full-screen window full screen a moment *after* it opens - it
        first appears in the desktop's work area (the screen less its panels) - so wait, handling
        the window's events, until it has the size asked for (or the moment has passed): the HUD
        must be laid out for the size the window ends up, not the one it started at."""
        import time as _time
        deadline = _time.perf_counter() + self.SETTLE_SECONDS
        while self._window_size() != tuple(wanted) and _time.perf_counter() < deadline:
            pygame.event.pump()
            pygame.time.wait(10)

    def _follow_window_size(self) -> bool:
        """Every frame: if the window is not the size the game is laid out for (the window manager
        resized it - a full-screen window settling late, a window kept off a panel), lay the view
        and the HUD out again for the size it is. Returns whether anything changed."""
        size = self._window_size()
        w = self.settings.window
        if size is None or size == (w.width, w.height):
            return False
        self.settings = dataclasses.replace(self.settings, window=dataclasses.replace(w, width=size[0], height=size[1]))
        self.renderer.resize(self.settings)
        self.menus.layout = self.settings.hud_layout()
        return True

    def apply_display(self, force: bool = False) -> bool:
        """Re-open the window if full screen, anti-aliasing or vsync changed (every GL resource is
        made again for the new context: the renderer and the menus). Returns whether it did."""
        if not force and self._wanted_display() == self._display_mode:
            return False
        self.show_busy("Opening the window")
        self.renderer.release()
        # a fresh display, not set_mode over the old one: SDL keeps an OpenGL window's old size (and
        # its pixel format - the multisampling) when the mode is merely set again
        pygame.display.quit()
        pygame.display.init()
        self.settings = self._open_window(self.settings)
        pygame.display.set_caption(self.settings.window.title)
        self.renderer = Renderer(self.settings, self.input.bindings.controls_help(), graphics=self.graphics)
        self.menus = MenuRenderer(self.renderer.text, self.settings.hud_layout(), self.settings.hud_colors, self.renderer.textures)
        self._mouse_captured = None                       # (grab the pointer again in the new window if playing)
        self._sync_mouse_capture()
        return True

    # ------------------------------------------------------------------ menus
    @property
    def saves(self):
        return self.game.saves

    def open_start_menu(self) -> None:
        self.screen = Screen.START
        if self.in_session or self.title is None or self.title.game is not self.game:
            # (no game underway now: the menu's town goes on living behind the start screen, the
            # view strolling its streets - a fresh one for a session just left)
            if self.in_session:
                self.game = self._make_game()
            self.title = TitleView(self.game)
        self.in_session = False
        items = []
        latest = self.saves.latest()
        if latest is not None:
            items.append(MenuItem("Continue", self.continue_game, value=lambda: latest.label))
        items += [
            MenuItem("New Game", self.new_game, value=lambda: "(with the tutorial)"),
            MenuItem("New Game - skip the tutorial", self.new_game_free),
            MenuItem("Load Game", self.open_load_menu, enabled=self.saves.has_saves),
            MenuItem("Graphics Settings", self.open_graphics_menu),
            MenuItem("Exit", self.request_quit),
        ]
        # (no subtitle of its own: the logo carries the tagline - a subtitle here is news, like a
        # save that would not load)
        self.menu = Menu("UNPREPARED", items, footer="[UP/DOWN] select   [ENTER] confirm")

    def open_pause_menu(self) -> None:
        self.screen = Screen.PAUSE
        self.menu = Menu("PAUSED", [
            MenuItem("Resume", self.resume),
            MenuItem("Save Game", self.open_save_menu, enabled=lambda: self.game.can_save,
                     value=lambda: "" if self.game.can_save else f"({self.game.save_refusal})"),
            MenuItem("Load Game", self.open_load_menu, enabled=self.saves.has_saves),
            MenuItem("Graphics Settings", self.open_graphics_menu),
            MenuItem("Exit to Menu", self.confirm_exit_to_menu),
        ], footer="[UP/DOWN] select   [ENTER] confirm   [ESC] resume")

    def _save_rows(self):
        return self.saves.list_saves()

    def open_save_menu(self) -> None:
        """Scrollable list of saves, newest first, defaulting to [New Save]; picking an old one asks first."""
        self.screen = Screen.SAVE
        items = [MenuItem("[New Save]", self.save_to_new_slot)]
        for info in self._save_rows():
            items.append(MenuItem(info.name, lambda info=info: self.confirm_overwrite(info), value=lambda info=info: info.label))
        self.menu = Menu("SAVE GAME", items, footer="[UP/DOWN] select   [ENTER] save here   [ESC] back")

    def open_load_menu(self) -> None:
        """Scrollable list of saves, newest first."""
        self._load_return = self.screen
        self.screen = Screen.LOAD
        items = [MenuItem(info.name, lambda info=info: self.load_from(info.path), value=lambda info=info: info.label)
                 for info in self._save_rows()]
        if not items:
            items.append(MenuItem("(no saves yet)", enabled=lambda: False))
        self.menu = Menu("LOAD GAME", items, footer="[UP/DOWN] select   [ENTER] load   [ESC] back")

    def open_confirm(self, title: str, question: str, on_yes, on_no, yes_label: str = "Yes", no_label: str = "No") -> None:
        self.screen = Screen.CONFIRM
        self._confirm_cancel = on_no
        self.menu = Menu(title, [MenuItem(no_label, on_no), MenuItem(yes_label, on_yes)], subtitle=question,
                         footer="[UP/DOWN] select   [ENTER] confirm   [ESC] back")

    def confirm_overwrite(self, info) -> None:
        self.open_confirm("OVERWRITE SAVE?", f"Replace \"{info.name}\" ({info.label})?",
                          on_yes=lambda: self.save_to(info.path), on_no=self.open_save_menu, yes_label="Yes, overwrite")

    def confirm_exit_to_menu(self) -> None:
        self.open_confirm("EXIT TO MENU?", "Unsaved progress will be lost.", on_yes=self.exit_to_menu,
                          on_no=self.open_pause_menu, yes_label="Exit without saving", no_label="Back")

    def open_graphics_menu(self) -> None:
        g = self.graphics
        self._graphics_return = self.screen
        self.screen = Screen.GRAPHICS

        def set_quality(delta):
            g.next_quality(delta)
            if not self.apply_display():
                self.rebuild_renderer()

        def cycle(attr, steps, rebuild=False, display=False):
            def f(delta=1):
                current = getattr(g, attr)
                idx = steps.index(current) if current in steps else 0
                setattr(g, attr, steps[(idx + delta) % len(steps)])
                g.auto_quality = False
                if display:
                    self.apply_display()
                elif rebuild:
                    self.rebuild_renderer()
            return f

        def toggle_display(attr):
            def f(delta=1):
                setattr(g, attr, not getattr(g, attr))
                self.apply_display()
            return f

        def set_lights(delta):
            g.torch_lights = max(2, min(MAX_SHADER_LIGHTS, g.torch_lights + delta * 2))
            g.auto_quality = False
            shader = self.renderer.shader
            if shader is not None and g.torch_lights > shader.max_lights:
                self.rebuild_renderer()                  # (the shader is built for so many lights: build a bigger one)

        def adjust(attr, low, high, step, rebuild=False):
            def f(delta):
                setattr(g, attr, round(max(low, min(high, getattr(g, attr) + delta * step)), 2))
                g.auto_quality = False
                if rebuild:
                    self.rebuild_renderer()
            return f

        def toggle(attr, rebuild=False):
            def f(delta=1):
                setattr(g, attr, not getattr(g, attr))
                g.auto_quality = False
                if rebuild:
                    self.rebuild_renderer()
            return f

        self.menu = Menu("GRAPHICS SETTINGS", [
            MenuItem("Quality preset", adjust=set_quality, value=lambda: f"< {g.quality.upper()} >" + ("  (auto)" if g.auto_quality else "")),
            MenuItem("Full screen", adjust=toggle_display("fullscreen"), action=lambda: toggle_display("fullscreen")(),
                     value=lambda: "ON" if g.fullscreen else "OFF (window)"),
            MenuItem("Draw distance", adjust=adjust("draw_distance", DRAW_DISTANCE_RANGE[0], DRAW_DISTANCE_RANGE[1], 20.0),
                     value=lambda: slider_text(g.draw_distance, DRAW_DISTANCE_RANGE[0], DRAW_DISTANCE_RANGE[1])),
            MenuItem("Field of view", adjust=adjust("fov", 50.0, 100.0, 2.0), value=lambda: slider_text(g.fov, 50, 100, suffix=" deg")),
            MenuItem("Object fade distance", adjust=adjust("object_fade", 0.3, 1.0, 0.05), value=lambda: slider_text(g.object_fade, 0.3, 1.0)),
            MenuItem("Character fade distance", adjust=adjust("character_fade", 0.3, 1.0, 0.05), value=lambda: slider_text(g.character_fade, 0.3, 1.0)),
            MenuItem("Contrast", adjust=adjust("contrast", 0.6, 1.6, 0.05), value=lambda: slider_text(g.contrast, 0.6, 1.6)),
            MenuItem("Distance haze", adjust=adjust("haze", 0.0, 1.5, 0.05), value=lambda: slider_text(g.haze, 0.0, 1.5)),
            MenuItem("Torch & lantern lights", adjust=set_lights, value=lambda: f"< {g.torch_lights} >"),
            MenuItem("Anti-aliasing", adjust=cycle("antialiasing", AA_STEPS, display=True),
                     value=lambda: f"{g.antialiasing}x MSAA" if g.antialiasing else "OFF"),
            MenuItem("Texture filtering", adjust=cycle("anisotropy", ANISOTROPY_STEPS, rebuild=True),
                     value=lambda: f"{g.anisotropy}x anisotropic" if g.anisotropy > 1 else "trilinear"),
            MenuItem("Vertical sync", adjust=toggle_display("vsync"), value=lambda: "ON" if g.vsync else "OFF"),
            MenuItem("Shader lighting", adjust=toggle("shaders", rebuild=True), value=lambda: "ON" if g.shaders else "OFF (fixed-function)"),
            MenuItem("Stars", adjust=toggle("stars"), value=lambda: "ON" if g.stars else "OFF"),
            MenuItem("Roofs & floors", adjust=toggle("roofs"), value=lambda: "ON" if g.roofs else "OFF"),
            MenuItem("Performance overlay", adjust=toggle("show_perf_overlay"), value=lambda: "ON" if g.show_perf_overlay else "OFF"),
            MenuItem("Re-detect best settings", self.auto_configure_quality, value=lambda: f"({g.detected_renderer[:28]}, {g.benchmark_ms} ms)"),
            MenuItem("Back", self.close_graphics_menu),
        ], subtitle=f"Renderer: {self.renderer.renderer_name[:48]}")

    def close_graphics_menu(self) -> None:
        self.graphics.save(self.graphics_path)
        if self._graphics_return is Screen.START:
            self.open_start_menu()
        else:
            self.open_pause_menu()

    def _back(self) -> None:
        """ESC inside a sub-menu: one level up."""
        if self.screen is Screen.GRAPHICS:
            self.close_graphics_menu()
        elif self.screen is Screen.SAVE:
            self.open_pause_menu()
        elif self.screen is Screen.LOAD:
            self.open_start_menu() if self._load_return is Screen.START else self.open_pause_menu()
        elif self.screen is Screen.CONFIRM:
            self._confirm_cancel()
        elif self.screen is Screen.PAUSE:
            self.resume()

    # ------------------------------------------------------------------ actions
    def new_game(self) -> None:
        """A new game starts with the tutorial: an urchin in Jae's debt."""
        self.show_busy("Preparing a new game")
        self.game = self._make_game(tutorial=True)
        self.resume()

    def new_game_free(self) -> None:
        """A new game without the tutorial: the town, the kit and fifty-odd gold."""
        self.show_busy("Preparing a new game")
        self.game = self._make_game()
        self.resume()

    def continue_game(self) -> None:
        latest = self.saves.latest()
        if latest is not None:
            self.load_from(latest.path)

    def load_game(self) -> None:
        """Load the newest save (kept for callers that do not pick a slot)."""
        self.continue_game()

    def load_from(self, path: str) -> None:
        """Load a save; a corrupt or unreadable one is reported on the menu and nothing changes."""
        # (from the start screen into a fresh game - the title's town stays behind the menu if the
        # file turns out to be no good)
        self.show_busy("Loading", os.path.basename(path))
        game = self.game if self.in_session else self._make_game()
        if game.load_game(path):
            self.game = game
            self.resume()
        elif self.menu is not None:
            self.menu.subtitle = f"Could not load {os.path.basename(path)}: the file is corrupt or not a save."

    def save_game(self) -> None:
        """Quick-save to the default file and resume."""
        self.save_to(None)

    def save_to(self, path) -> None:
        """Every save goes through ``Game.save_game`` so the same refusals apply as to F5 (mid-fade,
        on the way to the cells); a refused or failed save leaves the menu open with the reason."""
        if self.game.save_game(path):
            self.resume()
        elif self.menu is not None:
            self.menu.subtitle = self.game.messages.last(1)[0] if len(self.game.messages) else "Could not save."

    def save_to_new_slot(self) -> None:
        self.save_to(self.game.saves.new_save_path())

    def exit_to_menu(self) -> None:
        """Back to the start screen; the world keeps rendering behind it, nothing is saved."""
        self.open_start_menu()

    def resume(self) -> None:
        self.screen = Screen.PLAYING
        self.in_session = True
        self.title = None
        self.menu = None
        self.clock.tick()                      # drop the time spent in the menu
        self._sync_mouse_capture()

    def request_quit(self) -> None:
        self.quit_requested = True
        self.game.request_quit()

    # ------------------------------------------------------------------ loop
    @property
    def running(self) -> bool:
        return not self.quit_requested

    def _sync_mouse_capture(self) -> None:
        captured = self.screen is Screen.PLAYING and self.game.mouse_captured
        if captured != self._mouse_captured:
            pygame.mouse.set_visible(not captured)
            pygame.event.set_grab(captured)
            self._mouse_captured = captured

    def _process_menu_events(self) -> None:
        b = self.input.bindings
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.request_quit()
            elif event.type == pygame.KEYDOWN and self.menu is not None:
                if event.key == b.menu_up:
                    self.menu.move(-1)
                elif event.key == b.menu_down:
                    self.menu.move(+1)
                elif event.key == pygame.K_LEFT:
                    self.menu.adjust(-1)
                elif event.key == pygame.K_RIGHT:
                    self.menu.adjust(+1)
                elif event.key in (b.menu_confirm, pygame.K_KP_ENTER):
                    self.menu.activate()
                elif event.key == b.escape:
                    self._back()
                elif event.key == b.debug_overlay:
                    self.graphics.show_perf_overlay = not self.graphics.show_perf_overlay

    def _process_game_events(self) -> None:
        """Playing: ESC opens the pause menu, F3 toggles the overlay; everything else goes to the game."""
        b = self.input.bindings
        pending = []
        events = pygame.event.get()
        for index, event in enumerate(events):
            if (event.type == pygame.KEYDOWN and event.key == b.escape and self.game.in_explore_mode
                    and getattr(self.game, "reading", None) is None):      # (Esc puts down what is being read)
                self.open_pause_menu()
                for later in events[index + 1:]:          # the rest of the queue belongs to the menu now
                    pygame.event.post(later)
                break
            if event.type == pygame.KEYDOWN and event.key == b.debug_overlay:
                self.graphics.show_perf_overlay = not self.graphics.show_perf_overlay
                continue
            pending.append(event)
        self.input.process_events(self.game, pending)
        if not self.game.running:
            self.quit_requested = True

    def step(self) -> None:
        """Run exactly one frame: input -> simulation -> render -> present."""
        with self.profiler.phase("input"):
            delta_time = self.clock.tick(self.settings.window.target_fps) / 1000.0
            self._follow_window_size()
            if self.screen is Screen.PLAYING:
                self._process_game_events()
            else:
                self._process_menu_events()
            self._sync_mouse_capture()
        with self.profiler.phase("update"):
            if self.screen is Screen.PLAYING:
                self.game.update(delta_time, self.input.poll_movement())
            elif self.title is not None and not self.in_session:
                self.title.update(min(delta_time, 0.1))          # the start screen's town lives on behind the menus
        with self.profiler.phase("render3d"):
            if self.renderer.needs_baking(self.game):
                self.show_busy("Building the town")             # (the scene is baked before this frame is drawn)
            self.renderer.render_scene_only(self.game)
        with self.profiler.phase("hud"):
            if self.screen is Screen.PLAYING:
                self.renderer.render_hud(self.game)
                task = self.game.long_task
                if task is not None and task.worth_showing:
                    self.menus.draw_busy(task.label, task.progress, self._long_task_detail())
            if self.menu is not None:
                if self.screen is Screen.START and self.title is not None:
                    self.menus.draw_title(self.menu)
                else:
                    self.menus.draw_menu(self.menu, dim=True)
            if self.graphics.show_perf_overlay:
                scene = self.renderer.scene
                self.menus.draw_perf_overlay(self.profiler, [
                    f"quality {self.graphics.quality}  shaders {'on' if self.renderer.uses_shaders else 'off'}",
                    f"objects drawn {scene.drawn_objects}  culled {scene.culled_objects}  lights {len(self.renderer.lighting.point_lights)}",
                    f"renderer {self.renderer.renderer_name[:40]}  depth {self.depth_bits} bit",
                ])
        with self.profiler.phase("flip"):
            pygame.display.flip()
        self.profiler.end_frame()

    def _long_task_detail(self) -> str:
        """The line under a long task's name: the day and the hour it has reached."""
        world = self.game.world
        return f"Day {world.day_count + 1}, {world.clock.hour:02d}:00"

    def run(self) -> None:
        while self.running:
            self.step()
        self._sync_mouse_capture()

    def shutdown(self) -> None:
        self.graphics.save(self.graphics_path)
        pygame.quit()


def main() -> None:
    app = PygameApp()
    try:
        app.run()
    finally:
        app.shutdown()
    sys.exit()
