"""Graphics quality presets, GPU detection, and the pause menu that drives them."""

import pytest

from cyberduke.core.menu import MenuAction, PauseMenu
from cyberduke.render.quality import (
    PRESETS,
    QUALITY_ORDER,
    Quality,
    detect_quality,
    lights_for,
    settings_for,
)
from cyberduke.world.lights import Light


# ----------------------------------------------------------------------------- presets
def test_every_quality_has_a_preset_and_they_grow_with_the_quality():
    assert set(PRESETS) == set(Quality) == set(QUALITY_ORDER)
    low, medium, high = (PRESETS[q] for q in QUALITY_ORDER)
    assert low.view_distance < medium.view_distance < high.view_distance
    assert low.max_lights < medium.max_lights <= high.max_lights <= 8
    assert low.effect_distance < medium.effect_distance < high.effect_distance
    assert low.beam_slices < medium.beam_slices < high.beam_slices
    assert not low.sparks and medium.sparks and high.sparks
    assert not low.scan_ring and high.scan_ring
    assert not low.prefer_modern_backend and high.prefer_modern_backend
    assert low.global_ambient >= high.global_ambient, "fewer lights, so a flatter base level"
    assert settings_for("high") is PRESETS[Quality.HIGH]
    assert settings_for(Quality.LOW).name == "LOW"


def test_view_distance_covers_the_facility_at_high_quality():
    from cyberduke.levels import build_facility

    level = build_facility()
    depth = max(s.bounds[3] for s in level.sectors) - min(s.bounds[2] for s in level.sectors)
    assert settings_for(Quality.HIGH).view_distance >= depth * 0.8
    assert settings_for(Quality.LOW).view_distance >= 25.0, "still see across a room"


# ----------------------------------------------------------------------------- detection
@pytest.mark.parametrize(
    ("renderer", "vendor", "expected"),
    [
        ("llvmpipe (LLVM 15.0.7, 256 bits)", "Mesa/X.org", Quality.LOW),
        ("softpipe", "Mesa", Quality.LOW),
        ("SwiftShader", "Google SwRast", Quality.LOW),
        ("NVIDIA GeForce RTX 4070/PCIe/SSE2", "NVIDIA Corporation", Quality.HIGH),
        ("AMD Radeon RX 7900 XT (radeonsi)", "AMD", Quality.HIGH),
        ("Apple M2 Pro", "Apple", Quality.HIGH),
        ("Mesa Intel(R) UHD Graphics 620 (KBL GT2)", "Intel", Quality.MEDIUM),
        ("Intel Iris Plus Graphics", "Intel Inc.", Quality.MEDIUM),
        ("", "", Quality.MEDIUM),
        ("Some Unknown GPU 9000", "Nobody", Quality.MEDIUM),
    ],
)
def test_detect_quality_from_the_gl_strings(renderer, vendor, expected):
    assert detect_quality(renderer, vendor) is expected


def test_software_detection_wins_over_a_gpu_sounding_name():
    """Mesa's llvmpipe sometimes reports a device-ish name; the CPU tag must dominate."""
    assert detect_quality("llvmpipe, GeForce emulation", "Mesa") is Quality.LOW


# ----------------------------------------------------------------------------- lights
def _light(x, z):
    return Light(x, 3.0, z, ambient=(0, 0, 0), diffuse=(1, 1, 1), specular=(0, 0, 0))


def test_light_budget_keeps_the_nearest_lights_in_level_order():
    lights = [_light(0, 0), _light(0, 20), _light(0, 40), _light(0, 60)]
    high = lights_for(settings_for(Quality.HIGH), lights, (0.0, 1.6, 0.0))
    assert high == lights, "the whole set fits in the high budget"
    low = settings_for(Quality.LOW)
    near = lights_for(low, lights, (0.0, 1.6, 42.0))
    assert len(near) == low.max_lights
    assert near == [lights[2], lights[3]], "nearest two, still in level order"
    assert lights_for(low, lights, (0.0, 1.6, 0.0)) == [lights[0], lights[1]]


def test_light_budget_is_stable_when_the_eye_barely_moves():
    lights = [_light(0, 0), _light(0, 10)]
    low = settings_for(Quality.LOW)
    assert lights_for(low, lights, (0, 1.6, 4.9)) == lights_for(low, lights, (0, 1.6, 5.1))


def test_the_shipped_level_fits_the_high_preset_budget():
    from cyberduke.levels import build_facility

    assert len(build_facility().lights) <= settings_for(Quality.HIGH).max_lights


# ----------------------------------------------------------------------------- menus
PAUSE_ROWS = ["resume", "save", "load", "graphics", "display", "new", "main", "quit"]


def test_pause_menu_rows_and_wrapping():
    menu = PauseMenu(graphics="MEDIUM")
    assert [i.key for i in menu.items] == PAUSE_ROWS
    graphics = menu.items[PAUSE_ROWS.index("graphics")]
    assert menu.current.key == "resume" and graphics.value == "MEDIUM"
    menu.move(1)
    assert menu.current.key == "save"
    menu.move(-1)
    assert menu.current.key == "resume"
    menu.move(-1)
    assert menu.current.key == "quit", "wraps around the top"
    menu.move(1)
    assert menu.current.key == "resume", "and the bottom"


def test_pause_menu_actions():
    menu = PauseMenu()
    expected = {
        "resume": MenuAction.RESUME, "save": MenuAction.SAVE, "load": MenuAction.LOAD_MENU,
        "new": MenuAction.NEW_GAME, "main": MenuAction.MAIN_MENU, "quit": MenuAction.QUIT,
    }  # fmt: skip
    for k, key in enumerate(PAUSE_ROWS):
        menu.index = k
        if key in expected:
            assert menu.activate() is expected[key], key
    assert MenuAction.RESTART is MenuAction.NEW_GAME, "(the old name)"


def test_graphics_row_cycles_its_value_and_other_rows_ignore_left_right():
    menu = PauseMenu(graphics="LOW")
    assert menu.adjust(1) is MenuAction.NONE, "the cursor is on RESUME"
    menu.index = PAUSE_ROWS.index("graphics")
    assert menu.adjust(1) is MenuAction.SETTING_CHANGED and menu.graphics == "MEDIUM"
    assert menu.adjust(1) is MenuAction.SETTING_CHANGED and menu.graphics == "HIGH"
    assert menu.adjust(1) is MenuAction.SETTING_CHANGED and menu.graphics == "LOW", "wraps"
    assert menu.adjust(-1) is MenuAction.SETTING_CHANGED and menu.graphics == "HIGH"
    assert menu.activate() is MenuAction.SETTING_CHANGED and menu.graphics == "LOW"
    assert menu.adjust(0) is MenuAction.NONE


def test_display_row_toggles_fullscreen():
    menu = PauseMenu(fullscreen=False)
    menu.index = PAUSE_ROWS.index("display")
    assert menu.current.value == "WINDOWED"
    assert menu.adjust(1) is MenuAction.DISPLAY_CHANGED and menu.fullscreen
    assert menu.current.value == "FULLSCREEN"
    assert menu.activate() is MenuAction.DISPLAY_CHANGED and not menu.fullscreen


def test_title_menu_skips_what_is_not_there():
    from cyberduke.core.menu import TitleMenu

    menu = TitleMenu(has_saves=False)
    assert [i.key for i in menu.items] == ["continue", "new", "load", "graphics", "display", "quit"]
    assert menu.current.key == "new", "no saves: the cursor starts on NEW GAME"
    menu.move(1)
    assert menu.current.key == "graphics", "LOAD GAME is skipped without saves"
    menu.index = 0
    assert menu.activate() is MenuAction.NONE
    menu = TitleMenu(has_saves=True)
    assert menu.current.key == "continue" and menu.activate() is MenuAction.CONTINUE


def test_menu_reset_puts_the_cursor_back_on_resume():
    menu = PauseMenu()
    menu.move(2)
    menu.reset()
    assert menu.index == 0


# ----------------------------------------------------------------------------- app wiring
def test_drive_pause_menu_applies_graphics_immediately_and_reports_the_choice():
    from cyberduke.main import drive_pause_menu
    from cyberduke.simulation import InputState

    applied = []
    menu = PauseMenu(graphics="LOW")
    menu.index = PAUSE_ROWS.index("graphics") - 1
    assert drive_pause_menu(menu, InputState(menu_move=1), applied.append) is MenuAction.NONE
    assert menu.current.key == "graphics"
    assert (
        drive_pause_menu(menu, InputState(menu_move_x=1), applied.append)
        is MenuAction.SETTING_CHANGED
    )
    assert applied == ["MEDIUM"] and menu.graphics == "MEDIUM"
    menu.index = PAUSE_ROWS.index("new") - 1
    assert drive_pause_menu(menu, InputState(menu_move=1), applied.append) is MenuAction.NONE
    assert (
        drive_pause_menu(menu, InputState(menu_select=True), applied.append) is MenuAction.NEW_GAME
    )
    assert applied == ["MEDIUM"], "only a settings change re-applies graphics"
    assert drive_pause_menu(menu, InputState(), applied.append) is MenuAction.NONE
