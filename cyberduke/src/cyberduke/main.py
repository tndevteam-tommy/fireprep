"""Application entry point: window, input mapping, frame loop.

Everything the game does is in ``cyberduke.game.Game`` (menus, pausing, the [Tab]
screen, saving and loading) and ``cyberduke.simulation.Simulation`` (the world); this
module owns what only pygame can do - the window (and switching it between a window and
fullscreen), the keyboard and mouse, the clock - and the user's settings file.
"""

from __future__ import annotations

import argparse
import sys

import pygame

from cyberduke.config import SIMULATION
from cyberduke.core.menu import MenuAction, PauseMenu
from cyberduke.core.userdata import Settings, load_settings, save_settings
from cyberduke.levels import load_level
from cyberduke.render.scene import (
    BACKENDS,
    SceneRenderer,
    ViewportConfig,
    close_window,
    open_window,
)
from cyberduke.simulation import InputState

WINDOW_TITLE = "CyberDuke: Tactical Infiltration"


def poll_input(show_cursor: bool = False) -> tuple[InputState, bool]:
    """Translates pygame events/keys into an InputState. Returns (input, keep_running)."""
    inp = InputState()
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_ESCAPE:
                inp.toggle_pause = True  # the pause menu (or back out of a screen)
            elif key == pygame.K_BACKSPACE:
                inp.menu_back = True
            elif key in (pygame.K_UP, pygame.K_w):
                inp.menu_move = -1
            elif key in (pygame.K_DOWN, pygame.K_s):
                inp.menu_move = 1
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                inp.menu_select = True
            elif key == pygame.K_c:
                inp.toggle_crouch = True
            elif key == pygame.K_TAB:
                inp.toggle_inventory = True
            elif key == pygame.K_f:
                inp.scan = True  # the scanner ping
            elif key == pygame.K_e:
                inp.interact_pressed = True
            elif key == pygame.K_q:
                inp.disconnect = True
            elif key == pygame.K_LEFTBRACKET:
                inp.tab = -1
            elif key == pygame.K_RIGHTBRACKET:
                inp.tab = 1
            elif key == pygame.K_PAGEUP:
                inp.scroll = -1
            elif key == pygame.K_PAGEDOWN:
                inp.scroll = 1
            elif key in (pygame.K_LEFT, pygame.K_a):
                if key == pygame.K_LEFT:
                    inp.cycle_camera = -1
                inp.menu_move_x = -1
            elif key in (pygame.K_RIGHT, pygame.K_d):
                if key == pygame.K_RIGHT:
                    inp.cycle_camera = 1
                inp.menu_move_x = 1
            elif pygame.K_1 <= key <= pygame.K_9:
                inp.hack_slot = key - pygame.K_1 + 1
            elif pygame.K_KP1 <= key <= pygame.K_KP9:
                inp.hack_slot = key - pygame.K_KP1 + 1
        elif event.type == pygame.MOUSEMOTION:
            dx, dy = event.rel
            inp.look_dx += dx
            inp.look_dy += dy
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            inp.click = True
        elif event.type == pygame.MOUSEWHEEL:
            inp.wheel += int(event.y)

    keys = pygame.key.get_pressed()
    inp.forward = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
    inp.strafe = float(keys[pygame.K_a]) - float(keys[pygame.K_d])
    inp.interact = bool(keys[pygame.K_e])
    if show_cursor:
        inp.mouse_x, inp.mouse_y = (float(c) for c in pygame.mouse.get_pos())
        inp.look_dx = inp.look_dy = 0.0  # the mouse is a pointer, not the head
    return inp, running


def drive_pause_menu(menu: PauseMenu, inp: InputState, apply_graphics) -> MenuAction:
    """Maps one frame of input onto a menu. Returns the action the player asked for;
    ``apply_graphics(preset_name)`` is called whenever the graphics row changes so the
    new preset takes effect immediately. (``Game`` does the same for the real menus.)"""
    if inp.menu_move:
        menu.move(inp.menu_move)
    action = MenuAction.NONE
    if inp.menu_move_x:
        action = menu.adjust(inp.menu_move_x)
    if inp.menu_select:
        action = menu.activate()
    if action is MenuAction.SETTING_CHANGED:
        apply_graphics(menu.graphics)
    return action


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cyberduke", description="Cyberduke: Tactical Infiltration"
    )
    parser.add_argument(
        "--level",
        metavar="FILE",
        help="load a .json/.toml level file instead of the built-in facility as the level a "
        "new game starts in (export one with: python -m cyberduke.levels export level.json)",
    )
    parser.add_argument(
        "--graphics",
        choices=["auto", "low", "medium", "high"],
        default=None,
        help="graphics quality preset (view distance, lights, ambient detail). 'auto' detects "
        "the GPU: software rasteriser = low, known discrete card = high. Default: the preset "
        "saved in ~/Unprepared/settings.json, else auto.",
    )
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="start fullscreen (or --no-fullscreen: in a window). Default: as last time "
        "(~/Unprepared/settings.json).",
    )
    parser.add_argument(
        "--renderer",
        choices=BACKENDS,
        default="auto",
        help="'modern' = VBO/GLSL pipeline with per-pixel lighting; 'legacy' = fixed-function "
        "OpenGL 1.x with display lists; 'auto' (default) = modern on a GPU, legacy on a "
        "software renderer. Modern falls back to legacy if shaders fail to compile.",
    )
    return parser.parse_args(argv)


class Display:
    """The window: a 1024 x 768 window, or fullscreen at the desktop's resolution. The
    renderer is built for one window, so switching builds a new one."""

    WINDOW = (1024, 768)

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.renderer: SceneRenderer | None = None
        self.viewport = ViewportConfig()

    def open(self, fullscreen: bool, room, graphics: str | None) -> SceneRenderer:
        if fullscreen:
            sizes = (
                pygame.display.get_desktop_sizes()
                if hasattr(pygame.display, "get_desktop_sizes")
                else []
            )
            w, h = sizes[0] if sizes else self.WINDOW
        else:
            w, h = self.WINDOW
        self.viewport = ViewportConfig(width=w, height=h)
        if pygame.display.get_surface() is not None:
            # SDL cannot reliably switch an OpenGL window between windowed and fullscreen
            # in place (the window changes size, the GL surface and viewport do not): the
            # window is closed and opened again, with a new context - and a new renderer.
            self.renderer = None
            close_window()
        open_window(self.viewport, fullscreen=fullscreen)
        pygame.display.set_caption(WINDOW_TITLE)
        self.renderer = SceneRenderer(
            self.viewport,
            room,
            backend=self.backend,
            graphics=graphics.lower() if graphics else None,
        )
        return self.renderer


def main(argv: list[str] | None = None) -> None:
    from cyberduke.game import Game, Mode

    args = parse_args(argv)
    level = load_level(args.level) if args.level else None
    settings: Settings = load_settings()
    if args.fullscreen is not None:
        settings.fullscreen = args.fullscreen
    if args.graphics is not None:
        settings.graphics = None if args.graphics == "auto" else args.graphics.upper()

    pygame.init()
    display = Display(args.renderer)
    game = Game(settings, graphics=settings.graphics or "MEDIUM", level=level)
    renderer = display.open(settings.fullscreen, game.sim.room, settings.graphics)
    if settings.graphics is None:  # detected: show (and remember) what was picked
        game.graphics = renderer.settings.name
        game.title_menu.graphics = game.pause_menu.graphics = renderer.settings.name
    pygame.display.set_caption(
        f"{WINDOW_TITLE}  [{renderer.backend} renderer, {renderer.settings.name} graphics]"
    )
    clock = pygame.time.Clock()
    cursor = None

    running = True
    while running:
        raw_dt = clock.tick(SIMULATION.target_fps) / 1000.0
        show = game.shows_cursor
        if show != cursor:  # a menu or screen: the mouse is a pointer; in game it is the head
            pygame.event.set_grab(not show)
            pygame.mouse.set_visible(show)
            cursor = show
        inp, running = poll_input(show)
        for request in game.update(raw_dt, inp):
            if request == "quit":
                running = False
            elif request == "graphics":
                renderer.apply_graphics(game.graphics.lower())
                save_settings(game.settings)
            elif request == "display":
                renderer = display.open(game.settings.fullscreen, game.sim.room, game.graphics)
                cursor = None
                save_settings(game.settings)
        frozen = game.mode in (Mode.PAUSED, Mode.SCREEN) or (
            game.mode is Mode.LOAD and game.load_return is Mode.PAUSED
        )
        # A paused frame advances nothing, not even the HUD's own animation.
        renderer.draw_frame(game.sim, 0.0 if frozen else raw_dt, game=game)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
