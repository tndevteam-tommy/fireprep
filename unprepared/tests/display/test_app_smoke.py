"""Launch the real pygame/OpenGL app in a subprocess and make sure it runs."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tests.parity.display import PROJECT_DIR, _command_prefix, display_available

pytestmark = [pytest.mark.display, pytest.mark.skipif(not display_available(), reason="needs a display or xvfb-run")]


ASSET_DIR = os.path.join(PROJECT_DIR, "assets", "textures")


def _probe(width, height, frames, asset_dir, workdir):
    cmd = _command_prefix() + [sys.executable, os.path.join(PROJECT_DIR, "tests", "display", "app_probe.py"),
                               str(width), str(height), str(frames), asset_dir, workdir]
    env = dict(os.environ, SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    if not os.environ.get("DISPLAY"):
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    proc = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"probe failed\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
    line = next(l for l in proc.stdout.splitlines() if l.startswith("REPORT "))
    return json.loads(line[len("REPORT "):])


def test_app_runs_default_resolution(tmp_path):
    report = _probe(1024, 768, 20, ASSET_DIR, str(tmp_path))
    assert report["frames"] >= 20 and report["window"] == [1024, 768]
    assert report["caption"] == "probe"
    assert report["pixel_bytes"] == 1024 * 768 * 3 and report["nonblack"]
    assert report["missing_textures"] == []          # every texture the scene asks for ships in assets/


def test_app_runs_custom_resolution_with_missing_assets(tmp_path):
    """An empty asset dir must not crash the app: missing textures become placeholders."""
    report = _probe(800, 600, 5, str(tmp_path), str(tmp_path))
    assert report["window"] == [800, 600] and report["pixel_bytes"] == 800 * 600 * 3
    assert {"ground", "wall", "guard", "npc", "roof"} <= set(report["missing_textures"])


def test_main_entry_point_starts_and_quits(tmp_path):
    """`python main.py` must start; we close it by injecting a QUIT via an env-triggered hook."""
    code = (
        "import sys, os, pygame\n"
        "sys.path.insert(0, %r)\n"
        "orig = pygame.event.get\n"
        "n = [0]\n"
        "def fake():\n"
        "    n[0] += 1\n"
        "    return orig() + ([pygame.event.Event(pygame.QUIT)] if n[0] >= 3 else [])\n"
        "pygame.event.get = fake\n"
        "import runpy\n"
        "try:\n"
        "    runpy.run_path(os.path.join(%r, 'main.py'), run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    print('EXIT', e.code)\n"
        "print('FRAMES', n[0])\n"
    ) % (PROJECT_DIR, PROJECT_DIR)
    cmd = _command_prefix() + [sys.executable, "-c", code]
    data_dir = tmp_path / "data"
    # the default save location is the per-user data directory: point it at the temp dir
    env = dict(os.environ, SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1", UNPREPARED_DATA_DIR=str(data_dir))
    if not os.environ.get("DISPLAY"):
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "EXIT None" in proc.stdout and "FRAMES 3" in proc.stdout
    assert "[HUD / DIALOGUE] Tutorial started" in proc.stdout
    assert (data_dir / "graphics.json").exists()          # settings went to the data directory, not the CWD
    assert not (tmp_path / "graphics.json").exists()


def test_menus_drive_the_app_with_synthetic_keys(tmp_path):
    cmd = _command_prefix() + [sys.executable, os.path.join(PROJECT_DIR, "tests", "display", "menu_probe.py"), str(tmp_path)]
    env = dict(os.environ, SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    if not os.environ.get("DISPLAY"):
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    proc = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"menu probe failed\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
    r = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("REPORT "))[len("REPORT "):])
    # start menu without saves
    assert r["title_view"] and r["title_moves"]               # the live town behind the start menu
    assert r["start_items"] == ["New Game", "New Game - skip the tutorial", "Load Game", "Graphics Settings", "Exit"]
    assert r["load_enabled_without_save"] is False and r["cursor_after_nav"] == 0
    assert r["playing_after_new_game"] and r["mouse_grabbed_playing"] is True and r["tutorial_on"]
    # pause / save / load
    assert r["paused"] and r["mouse_grabbed_paused"] is False
    assert r["pause_items"] == ["Resume", "Save Game", "Load Game", "Graphics Settings", "Exit to Menu"]
    assert r["save_screen"] and r["save_items_first_time"] == ["[New Save]"]
    assert r["saved_new_slot"] and r["save_label_has_gold"] and r["quick_save_listed"]
    assert r["save_items_second_time"][:2] == ["[New Save]", "Quick Save"] and len(r["save_items_second_time"]) == 3
    assert r["overwrite_confirm"] and r["confirm_escape_returns_to_list"] and r["overwritten"]
    assert r["load_screen"] and len(r["load_items"]) == 2 and r["loaded_quick_save"]
    # graphics
    assert r["graphics_open"] and r["quality_changed"]
    assert r["fov"] == 64.0 and r["auto_quality_off_after_manual_change"]
    assert r["shaders_toggled"] and r["shaders_restored"] and r["back_to_pause"] and r["graphics_json_saved"] and r["resumed"]
    far_before, far_after, far_renderer = r["draw_distance"]
    assert far_after == far_before + 20.0 and far_renderer == far_after
    on, size_on, desktop, hud_size = r["fullscreen_on"]
    assert on and size_on == desktop and hud_size == desktop           # the HUD lays itself out to the new size
    off, size_off, windowed = r["fullscreen_off"]
    assert not off and size_off == windowed
    hud, window = r["hud_is_the_window"]
    assert hud == window
    assert r["followed_resize"] == [700, 500, 700, 500, 700, pytest.approx(700 / 500)]
    assert r["followed_back"] == window
    aa_before, aa_after, still_in_menu = r["antialiasing"]
    assert aa_after != aa_before and still_in_menu
    assert r["anisotropy_ok"]
    lights, shader_lights = r["lights"]
    assert shader_lights is None or shader_lights >= lights             # the shader is rebuilt for more lights
    assert r["redetect_keeps_settings_object"] and r["menu_shows_live_quality"] and r["saved_fov_matches_menu"]
    fov_now, fov_expected, shown = r["fov_after_redetect_edit"]
    assert fov_now == fov_expected and f"{fov_now:g}" in shown
    # exit to menu + continue
    assert r["exit_confirm"] and r["exit_back"] and r["start_after_exit"]
    assert r["start_items_with_saves"] == ["Continue", "New Game", "New Game - skip the tutorial", "Load Game", "Graphics Settings", "Exit"]
    assert r["corrupt_load_from_start"]
    assert r["continue_is_default"] and r["continued"]
    assert r["benchmark_quality"] in ("high", "medium", "low") and r["benchmark_ms"] > 0
    assert r["quit"]


def test_benchmark_tool_runs(tmp_path):
    cmd = _command_prefix() + [sys.executable, os.path.join(PROJECT_DIR, "tools", "benchmark.py"),
                               "--frames", "3", "--presets", "low", "--json", str(tmp_path / "bench.json")]
    env = dict(os.environ, SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    if not os.environ.get("DISPLAY"):
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    data = json.loads((tmp_path / "bench.json").read_text())
    assert "low" in data["presets"] and data["presets"]["low"]["frames"] == 12   # 3 frames x 4 camera views
    assert "recommended" in data and data["renderer"]


def test_equipment_view_draws_the_dressed_figure(tmp_path):
    """The [Tab] view renders the player's figure into its frame, turns it, and redraws it when
    something is taken off."""
    cmd = _command_prefix() + [sys.executable, os.path.join(PROJECT_DIR, "tests", "display", "equipment_probe.py"), str(tmp_path)]
    env = dict(os.environ, SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    if not os.environ.get("DISPLAY"):
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    proc = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"equipment probe failed\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
    r = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("REPORT "))[len("REPORT "):])
    assert r["opened"] and r["mode"] == "equipment" and not r["mouse_captured"]
    assert r["worn"] == {"feet": "soft_shoes", "head": "iron_helm", "body": "plate_cuirass", "weapon": "short_sword", "neck": "wool_cloak"}
    assert 0.08 * r["stand_pixels"] < r["figure_pixels_front"] < 0.8 * r["stand_pixels"]     # a figure, not a blank or a flood
    assert 0.08 * r["stand_pixels"] < r["figure_pixels_side"] < 0.8 * r["stand_pixels"]
    assert r["turns"] and r["helm_off_changes_the_figure"] and r["closed"]


def test_fixed_function_lights_sit_where_the_camera_sees_them(tmp_path):
    """On the *low* preset (no shader) GL places its lights through the modelview current when they
    are set: the sun and the torches must be given with this frame's camera on every frame - also
    after frames that end drawing in view space (the held torch, the tutorial's arrow), which used
    to leave the sun and every torch swinging round with the player's head."""
    cmd = _command_prefix() + [sys.executable, os.path.join(PROJECT_DIR, "tests", "display", "lighting_probe.py"), str(tmp_path)]
    env = dict(os.environ, SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    if not os.environ.get("DISPLAY"):
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    proc = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"lighting probe failed\nSTDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
    r = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("REPORT "))[len("REPORT "):])
    assert r["shaders"] is False and r["waypoint"]
    assert len(r["frames"]) == 8 and any("torch" in f for f in r["frames"])
    for index, frame in enumerate(r["frames"]):
        assert frame["sun"] < 1e-3, (index, frame)
        assert frame.get("torch", 0.0) < 1e-3, (index, frame)
