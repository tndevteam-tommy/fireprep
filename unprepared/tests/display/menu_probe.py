"""Subprocess body for the menu smoke test: drive the start / pause / save / load / graphics
menus with synthetic key events through the real ``PygameApp`` loop and report what happened."""
from __future__ import annotations

import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def main() -> int:
    workdir = sys.argv[1]
    import pygame
    from unprepared.app import PygameApp, Screen, desktop_size
    from unprepared.config import GameSettings, WindowSettings
    from unprepared.graphics import GraphicsSettings

    settings = GameSettings(window=WindowSettings(width=800, height=600, title="menu-probe"),
                            asset_dir=os.path.join(PROJECT_DIR, "assets", "textures"), echo_messages_to_stdout=False)
    save_file = os.path.join(workdir, "menu_save.json")
    app = PygameApp(settings, save_file=save_file, graphics=GraphicsSettings(), auto_quality=False)
    report = {"screens": [], "benchmark_quality": None}

    def key(k):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0))
        app.step()
        report["screens"].append(app.screen.name)

    def steps(n):
        for _ in range(n):
            app.step()

    def labels():
        return [item.label for item in app.menu.items]

    steps(2)
    report["title_view"] = app.title is not None
    if app.title is not None:
        eye_before = (app.game.player.position.x, app.game.player.position.z)
        steps(3)
        report["title_moves"] = (app.game.player.position.x, app.game.player.position.z) != eye_before
    report["start_items"] = labels()                           # no saves yet: no Continue
    report["load_enabled_without_save"] = app.menu.items[2].enabled()
    key(pygame.K_DOWN); key(pygame.K_UP)                      # cursor moves, nothing activates
    report["cursor_after_nav"] = app.menu.cursor
    key(pygame.K_RETURN)                                       # New Game (with the tutorial)
    report["playing_after_new_game"] = app.screen is Screen.PLAYING and app.menu is None
    report["tutorial_on"] = app.game.in_tutorial and app.game.tutorial_hint != ""
    report["mouse_grabbed_playing"] = pygame.event.get_grab()
    steps(3)
    gold_before = app.game.player.gold
    app.game.player.gold = 77.0                                # something to recognise in the save

    # --- pause -> Save Game -> [New Save] ------------------------------------------------
    key(pygame.K_ESCAPE)
    report["paused"] = app.screen is Screen.PAUSE
    report["mouse_grabbed_paused"] = pygame.event.get_grab()
    report["pause_items"] = labels()
    key(pygame.K_DOWN); key(pygame.K_RETURN)                   # Save Game -> slot list
    report["save_screen"] = app.screen is Screen.SAVE
    report["save_items_first_time"] = labels()                 # just [New Save]
    key(pygame.K_RETURN)                                       # new slot -> back to playing
    report["saved_new_slot"] = app.screen is Screen.PLAYING and len(app.saves.list_saves()) == 1
    first_save = app.saves.list_saves()[0]
    report["save_label_has_gold"] = "77g" in first_save.label and "Day 1" in first_save.label

    # --- quick save with F5 adds the quick-save file to the list ---------------------------
    key(pygame.K_F5)
    report["quick_save_listed"] = len(app.saves.list_saves()) == 2 and app.saves.list_saves()[0].quick

    # --- Save Game -> pick the old slot -> confirm overwrite -------------------------------
    app.game.player.gold = 99.0
    key(pygame.K_ESCAPE); key(pygame.K_DOWN); key(pygame.K_RETURN)
    report["save_items_second_time"] = labels()                # [New Save], Quick Save, save-...
    key(pygame.K_DOWN); key(pygame.K_DOWN); key(pygame.K_RETURN)   # the named slot -> confirm
    report["overwrite_confirm"] = app.screen is Screen.CONFIRM and labels() == ["No", "Yes, overwrite"]
    key(pygame.K_ESCAPE)                                       # back out -> save list again
    report["confirm_escape_returns_to_list"] = app.screen is Screen.SAVE
    key(pygame.K_DOWN); key(pygame.K_DOWN); key(pygame.K_RETURN)
    key(pygame.K_DOWN); key(pygame.K_RETURN)                   # Yes, overwrite
    report["overwritten"] = app.screen is Screen.PLAYING and "99g" in app.saves.describe(first_save.path)

    # --- Load Game from the pause menu restores the quick save (77g) -----------------------
    app.game.player.gold = 5.0
    key(pygame.K_ESCAPE); key(pygame.K_DOWN); key(pygame.K_DOWN); key(pygame.K_RETURN)
    report["load_screen"] = app.screen is Screen.LOAD
    report["load_items"] = labels()
    quick_index = next(i for i, item in enumerate(app.menu.items) if item.label == "Quick Save")
    for _ in range(quick_index):
        key(pygame.K_DOWN)
    key(pygame.K_RETURN)
    report["loaded_quick_save"] = app.screen is Screen.PLAYING and app.game.player.gold == 77.0

    # --- Graphics menu ------------------------------------------------------------------
    key(pygame.K_ESCAPE)
    key(pygame.K_DOWN); key(pygame.K_DOWN); key(pygame.K_DOWN); key(pygame.K_RETURN)   # Graphics Settings
    report["graphics_open"] = app.screen is Screen.GRAPHICS
    quality_before = app.graphics.quality
    key(pygame.K_RIGHT)                                        # next quality preset (rebuilds the renderer)
    report["quality_changed"] = app.graphics.quality != quality_before
    def goto(prefix):
        target = next(i for i, item in enumerate(app.menu.items) if item.label.startswith(prefix))
        while app.menu.cursor != target:
            key(pygame.K_DOWN)

    goto("Field of view")
    key(pygame.K_RIGHT); key(pygame.K_RIGHT)                   # FOV +4
    report["fov"] = app.graphics.fov
    report["auto_quality_off_after_manual_change"] = not app.graphics.auto_quality
    goto("Shader lighting")
    before = app.graphics.shaders
    key(pygame.K_RIGHT)
    report["shaders_toggled"] = app.graphics.shaders is (not before) and app.renderer.uses_shaders is (not before)
    key(pygame.K_RIGHT)
    report["shaders_restored"] = app.graphics.shaders is before and app.renderer.uses_shaders is before
    # the draw distance moves the far plane at once
    goto("Draw distance")
    far_before = app.graphics.far_plane
    key(pygame.K_RIGHT)
    report["draw_distance"] = (far_before, app.graphics.far_plane, app.renderer.graphics.far_plane)
    # full screen re-opens the window at the desktop's size (and back); the HUD follows it
    windowed = pygame.display.get_window_size()
    goto("Full screen")
    desktop = desktop_size()
    key(pygame.K_RETURN)
    # (a window manager may finish making it full screen a moment later: the game follows the
    # window's size from frame to frame, so give it a few frames - no more than three seconds)
    import time as _time
    deadline = _time.perf_counter() + 3.0
    while tuple(pygame.display.get_window_size()) != tuple(desktop) and _time.perf_counter() < deadline:
        steps(1)
    steps(1)
    report["fullscreen_on"] = (app.graphics.fullscreen, list(pygame.display.get_window_size()),
                               list(desktop), [app.settings.window.width, app.settings.window.height])
    key(pygame.K_RETURN)
    report["fullscreen_off"] = (app.graphics.fullscreen, list(pygame.display.get_window_size()), list(windowed))
    steps(3)
    report["hud_is_the_window"] = ([app.settings.window.width, app.settings.window.height],
                                   list(pygame.display.get_window_size()))
    # a window the window manager resizes behind the game's back: the view and the HUD follow it
    real_size = app._window_size
    app._window_size = lambda: (700, 500)
    steps(1)
    report["followed_resize"] = (app.settings.window.width, app.settings.window.height, app.renderer.hud.layout.width,
                                 app.renderer.hud.layout.height, app.menus.layout.width, app.renderer.scene.aspect)
    app._window_size = real_size
    steps(1)
    report["followed_back"] = [app.settings.window.width, app.settings.window.height]
    # anti-aliasing re-opens the window too (whatever the driver grants, it must keep running)
    goto("Anti-aliasing")
    aa_before = app.graphics.antialiasing
    key(pygame.K_RIGHT)
    steps(2)
    report["antialiasing"] = (aa_before, app.graphics.antialiasing, app.screen is Screen.GRAPHICS)
    key(pygame.K_LEFT)
    goto("Texture filtering")
    key(pygame.K_RIGHT)
    steps(1)
    report["anisotropy_ok"] = app.screen is Screen.GRAPHICS
    goto("Torch & lantern lights")
    key(pygame.K_RIGHT); key(pygame.K_RIGHT)
    steps(1)
    shader = app.renderer.shader
    report["lights"] = (app.graphics.torch_lights, shader.max_lights if shader is not None else None)
    # "Re-detect best settings" reruns the benchmark; the menu must keep editing the *live*
    # settings object afterwards (it used to be swapped out, orphaning the menu's closures)
    settings_obj = app.graphics
    redetect = next(i for i, item in enumerate(app.menu.items) if item.label.startswith("Re-detect"))
    while app.menu.cursor != redetect:
        key(pygame.K_DOWN)
    key(pygame.K_RETURN)
    report["redetect_keeps_settings_object"] = app.graphics is settings_obj
    goto("Field of view")
    fov_before = app.graphics.fov
    key(pygame.K_RIGHT)                                        # +2 deg after the re-detect
    report["fov_after_redetect_edit"] = (app.graphics.fov, fov_before + 2.0, app.menu.current.display_value())
    report["menu_shows_live_quality"] = app.graphics.quality.upper() in app.menu.items[0].display_value()
    key(pygame.K_ESCAPE)                                       # back to the pause menu (saves graphics json)
    report["back_to_pause"] = app.screen is Screen.PAUSE
    with open(app.graphics_path) as f:
        report["saved_fov_matches_menu"] = json.load(f)["fov"] == app.graphics.fov
    report["graphics_json_saved"] = os.path.exists(app.graphics_path)
    key(pygame.K_ESCAPE)                                       # resume
    report["resumed"] = app.screen is Screen.PLAYING

    # --- Exit to Menu asks first, then shows the start menu with Continue on top ---------------
    key(pygame.K_ESCAPE)
    key(pygame.K_UP); key(pygame.K_RETURN)                     # wraps to "Exit to Menu"
    report["exit_confirm"] = app.screen is Screen.CONFIRM and labels() == ["Back", "Exit without saving"]
    key(pygame.K_RETURN)                                       # "Back" (the default) -> pause menu
    report["exit_back"] = app.screen is Screen.PAUSE
    key(pygame.K_UP); key(pygame.K_RETURN); key(pygame.K_DOWN); key(pygame.K_RETURN)   # confirm the exit
    report["start_after_exit"] = app.screen is Screen.START
    report["start_items_with_saves"] = labels()
    report["continue_is_default"] = app.menu.cursor == 0 and labels()[0] == "Continue"
    # a corrupt file picked from the start screen: reported, and the town behind the menu goes on
    bad = os.path.join(workdir, "corrupt_save.json")
    with open(bad, "w") as f:
        f.write("{ not a save")
    app.load_from(bad)
    report["corrupt_load_from_start"] = (app.screen is Screen.START and app.title is not None and app.title.game is app.game
                                         and "Could not load" in (app.menu.subtitle or ""))
    app.game.player.gold = 1.0                                 # the abandoned game
    key(pygame.K_RETURN)                                       # Continue -> loads the newest save (the 99g overwrite)
    report["continued"] = app.screen is Screen.PLAYING and app.game.player.gold == 99.0

    report["benchmark_quality"] = app.auto_configure_quality()
    report["benchmark_ms"] = app.graphics.benchmark_ms
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.step()
    report["quit"] = not app.running
    report["gold_before"] = gold_before
    app.shutdown()
    print("REPORT " + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
