from unprepared.config import DEFAULT_SETTINGS, GameSettings, HudLayout, WindowSettings


def test_defaults_match_prototype():
    s = DEFAULT_SETTINGS
    assert (s.window.width, s.window.height) == (1024, 768)
    assert s.player.walk_speed == 4.0 and s.player.sprint_multiplier == 1.8
    assert s.player.eye_height == 1.55           # the prototype's 1.6 sat on a position 1.0 up: a head taller than everyone
    assert s.time.time_scale == 2.5 and s.time.start_hour == 7
    assert s.interaction.door_range == 5.0 and s.interaction.npc_range == 6.0 and s.interaction.pickpocket_range == 2.5
    assert s.player.crouch_eye_height < s.player.eye_height
    assert s.lockpick.default_sweet_spot == 45.0 and s.lockpick.success_tolerance_degrees == 15.0
    assert s.skill_rewards.lockpicking == 45.0 and s.skill_rewards.pickpocket == 20.0 and s.skill_rewards.smithing == 40.0
    assert s.lighting.fog_density_night > s.lighting.fog_density_day and s.lighting.torch_range > 0
    assert s.crafting.lockpicks_min == 1 and s.crafting.lockpicks_max == 15
    # the one skill model: levels 1..100, every check on one logistic curve
    assert s.progression.max_level == 100 and s.progression.logistic_scale == 8.0
    assert list(s.stealth.lock_challenge) == sorted(s.stealth.lock_challenge)      # tiers get harder in order
    assert len(s.stealth.lock_challenge) == 6


def test_settings_are_immutable():
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_SETTINGS.player.walk_speed = 1.0


def test_hud_layout_default_is_prototype_layout():
    layout = GameSettings().hud_layout()
    assert layout == HudLayout()
    assert layout.center == (512, 384)
    assert layout.controls_panel == (720, 55, 1014, 275)
    assert layout.log_panel == (370, 560, 1014, 750)        # full width so long messages wrap, not truncate


def test_hud_layout_scales_with_window():
    layout = HudLayout.for_window(1280, 1024)
    assert layout.center == (640, 512)
    assert layout.controls_panel == (976, 55, 1270, 275)          # anchored right
    assert layout.inventory_panel == (10, 836, 350, 1006)         # anchored bottom
    assert layout.log_panel == (370, 816, 1270, 1006)
    assert layout.trade_panel == (340, 262, 940, 762)             # centred
    assert layout.lockpick_center == (640, 558)


def test_override_window_settings():
    s = GameSettings(window=WindowSettings(width=800, height=600, title="Test"))
    assert s.hud_layout().width == 800 and s.window.title == "Test"
