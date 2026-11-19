"""Fullscreen support: the saved preference and command-line flags, and a UI that is laid out for
any screen size (the GL side is exercised by running the game; these are its pure parts)."""
import json
import os
import tempfile

from pad_helpers import *  # noqa
import config
from core.window import fullscreen_from_args, load_display_settings, save_display_settings
from core.gamepad import ControllerBindings, Family
from core.controller_menu import ControllerMenuModel, ControllerLayout


def test_command_line_flags_override_the_saved_choice():
    assert fullscreen_from_args(["--fullscreen"], False) is True
    assert fullscreen_from_args(["--windowed"], True) is False
    assert fullscreen_from_args([], True) is True and fullscreen_from_args([], False) is False
    assert fullscreen_from_args(["--something-else"], False) is False


def test_display_preference_is_saved_and_survives_bad_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cfg", "display.json")
        assert load_display_settings(path) == {"fullscreen": config.FULLSCREEN}, "no file: the default"
        assert save_display_settings({"fullscreen": True}, path)
        assert load_display_settings(path) == {"fullscreen": True}
        with open(path, "w") as f:
            f.write("not json")
        assert load_display_settings(path) == {"fullscreen": config.FULLSCREEN}
        with open(path, "w") as f:
            json.dump([1, 2], f)
        assert load_display_settings(path) == {"fullscreen": config.FULLSCREEN}
        blocker = os.path.join(tmp, "file")
        open(blocker, "w").close()
        assert save_display_settings({"fullscreen": True}, os.path.join(blocker, "display.json")) is False


def test_display_settings_live_next_to_the_controller_settings():
    from core.window import display_settings_path
    from core.gamepad import default_settings_path
    old = os.environ.get("HARDCORE_CONFIG_DIR")
    os.environ["HARDCORE_CONFIG_DIR"] = "/somewhere"
    try:
        assert os.path.dirname(display_settings_path()) == os.path.dirname(default_settings_path()) == "/somewhere"
    finally:
        if old is None:
            del os.environ["HARDCORE_CONFIG_DIR"]
        else:
            os.environ["HARDCORE_CONFIG_DIR"] = old


def test_ui_scales_uniformly_with_the_screen():
    from render.reticle import ui_scale
    assert ui_scale(1280, 720) == 1.0
    assert ui_scale(1920, 1080) == 1.5
    assert ui_scale(1920, 1200) == 1.5, "16:10: the limiting axis decides (no stretching)"
    assert ui_scale(3840, 2160) == 3.0
    assert ui_scale(2560, 1080) == 1.5, "ultrawide: limited by the height"
    assert ui_scale(1024, 768) == 0.8


def test_controller_screen_surface_is_capped_for_big_screens():
    from render.menu import controller_surface_size, CONTROLLER_SURFACE_MAX
    assert controller_surface_size(1280, 720) == (1280, 720)
    assert controller_surface_size(1920, 1080) == (1920, 1080)
    w, h = controller_surface_size(3840, 2160)
    assert w <= CONTROLLER_SURFACE_MAX[0] and h <= CONTROLLER_SURFACE_MAX[1]
    assert abs(w / h - 3840 / 2160) < 0.01, "same aspect: stretched back to the screen without distortion"


def test_controller_screen_lays_out_again_for_a_new_size_keeping_its_state():
    m = ControllerMenuModel(ControllerBindings(), 1280, 720)
    m.set_device("Pad", Family.PLAYSTATION)
    m.open()
    m.select(("control", "y"))
    for size in ((1920, 1080), (1920, 1200), (2560, 1080), (1024, 768)):
        m.resize(*size)
        assert m.waiting == "y", "a key wait survives going fullscreen"
        L = m.layout
        assert (L.screen_w, L.screen_h) == size
        # every label box is on screen and clickable at its new place
        for control, entry in L.callouts.items():
            x, y, w, h = entry["rect"]
            assert 0 <= x and x + w <= size[0] and 0 <= y and y + h <= size[1], (size, control)
            assert L.hit((x + w / 2, y + h / 2)) == ("control", control)
    big, small = ControllerLayout(Family.XBOX, 1920, 1080), ControllerLayout(Family.XBOX, 1280, 720)
    assert abs(big.controls["a"]["r"] / small.controls["a"]["r"] - 1.5) < 1e-6, "drawn 1.5x larger at 1080p"
