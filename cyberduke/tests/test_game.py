"""The game around the simulation (cyberduke.game): the start menu, pausing, the [Tab]
screen with its INVENTORY and QUESTS tabs, saving and loading from the menus, and the
settings that survive a restart (display mode and graphics, in <home>/Unprepared)."""

from __future__ import annotations

import json

import pytest

from cyberduke.core.screens import GRID_COLUMNS, Hotspot
from cyberduke.core.userdata import Settings, load_settings, save_settings, user_dir
from cyberduke.game import Game, Mode, title_view
from cyberduke.levels import build_night_market
from cyberduke.simulation import InputState

DT = 1 / 30


@pytest.fixture(autouse=True)
def user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CYBERDUKE_USER_DIR", str(tmp_path / "Unprepared"))
    return tmp_path / "Unprepared"


def step(game, n=1, **kw):
    out = []
    for _ in range(n):
        out += game.update(DT, InputState(**kw))
    return out


def playing() -> Game:
    game = Game()
    step(game, menu_select=True)  # NEW GAME (the cursor starts there with no saves)
    assert game.mode is Mode.PLAYING
    return game


# ----------------------------------------------------------------------------- the start menu
def test_the_game_opens_on_the_start_menu_over_the_living_street():
    game = Game()
    assert game.mode is Mode.TITLE and game.shows_cursor
    assert game.sim.level.name == "Night Market Alley", "the street is the backdrop"
    assert game.title_menu.current.key == "new", "no saves: CONTINUE is greyed out"
    people = [(c.x, c.z) for c in game.sim.civilians]
    step(game, 60)
    assert [(c.x, c.z) for c in game.sim.civilians] != people, "they go about their business"
    assert game.sim.viewpoint.source == "title", "seen from the drifting camera"
    assert title_view(0.0).eye != title_view(30.0).eye


def test_new_game_starts_in_the_facility():
    game = playing()
    assert game.sim.level.name == "Corporate Facility" and not game.shows_cursor
    assert game.sim.viewpoint.source == "player"


def test_continue_loads_the_newest_save_and_quit_asks_the_app_to_quit():
    game = playing()
    game.sim.player.x = 2.5
    step(game, toggle_pause=True)
    game.pause_menu.index = [i.key for i in game.pause_menu.items].index("save")
    step(game, menu_select=True)
    assert "GAME SAVED" in game.message
    game.pause_menu.index = [i.key for i in game.pause_menu.items].index("main")
    step(game, menu_select=True)
    assert game.mode is Mode.TITLE and game.title_menu.current.key == "continue"
    step(game, menu_select=True)
    assert game.mode is Mode.PLAYING and game.sim.level.name == "Corporate Facility"
    assert game.sim.player.x == pytest.approx(2.5)
    step(game, toggle_pause=True)
    game.pause_menu.index = [i.key for i in game.pause_menu.items].index("quit")
    assert "quit" in step(game, menu_select=True)


# ----------------------------------------------------------------------------- pause, save, load
def test_pause_freezes_the_world_and_esc_resumes():
    game = playing()
    step(game, 10, forward=1.0)
    step(game, toggle_pause=True)
    assert game.mode is Mode.PAUSED and game.shows_cursor
    before = (game.sim.time, game.sim.player.x, game.sim.player.z)
    step(game, 30, forward=1.0)
    assert (game.sim.time, game.sim.player.x, game.sim.player.z) == before
    step(game, toggle_pause=True)
    assert game.mode is Mode.PLAYING


def test_the_load_menu_lists_the_saves_and_loads_the_one_picked(user_home):
    game = playing()
    game.sim.player.x = 1.0
    game.save()
    game.sim.player.x = 2.0
    game.save()
    step(game, toggle_pause=True)
    game.pause_menu.index = [i.key for i in game.pause_menu.items].index("load")
    step(game, menu_select=True)
    assert game.mode is Mode.LOAD and len(game.load_menu.saves) == 2
    assert sorted(p.name for p in (user_home / "Saves").iterdir()) == sorted(
        s.path.name for s in game.load_menu.saves
    )
    step(game, menu_move=1)  # the older one
    step(game, menu_select=True)
    assert game.mode is Mode.PLAYING and game.sim.player.x == pytest.approx(1.0)
    # [Esc] in the load menu goes back where it came from
    step(game, toggle_pause=True)
    game.open_load_menu(Mode.PAUSED)
    step(game, toggle_pause=True)
    assert game.mode is Mode.PAUSED


def test_the_mouse_picks_menu_rows():
    game = Game()
    game.title_menu.hotspots = [
        (0, 100 + 40 * k, 300, 34) for k in range(len(game.title_menu.items))
    ]
    quit_row = [i.key for i in game.title_menu.items].index("quit")
    assert "quit" in step(game, mouse_x=10, mouse_y=100 + 40 * quit_row + 5, click=True)


def test_a_resting_pointer_does_not_steal_the_keyboards_choice():
    """Regression: the pointer resting over a row re-selected that row every frame, so
    [Down] [Enter] in the pause menu chose whatever the mouse happened to be over."""
    game = playing()
    step(game, toggle_pause=True)
    menu = game.pause_menu
    menu.hotspots = [(0, 100 + 40 * k, 300, 34) for k in range(len(menu.items))]
    quit_row = [i.key for i in menu.items].index("quit")
    over_quit = {"mouse_x": 10.0, "mouse_y": 100.0 + 40 * quit_row + 5}
    step(game, 3, **over_quit)  # the pointer shows up over QUIT and rests there
    step(game, menu_move=1, **over_quit)  # [Down]: RESUME -> SAVE GAME
    assert menu.current.key == "save"
    assert "quit" not in step(game, menu_select=True, **over_quit)
    assert game.message.startswith("GAME SAVED")
    # moving it over a row does pick the row
    step(game, mouse_x=12.0, mouse_y=over_quit["mouse_y"])
    assert menu.current.key == "quit"


def test_no_saving_your_way_out_of_being_caught(user_home):
    from cyberduke.core.state import GameState

    game = playing()
    game.sim.state = GameState.CAUGHT
    step(game, toggle_pause=True)
    assert game.save() is None and game.message.startswith("COULD NOT SAVE")
    assert not list(user_home.glob("Saves/*.json")), "nothing written"


# ----------------------------------------------------------------------------- settings
def test_settings_live_in_the_users_unprepared_folder(user_home):
    assert user_dir() == user_home
    assert load_settings() == Settings(), "defaults when there is no file"
    save_settings(Settings(fullscreen=True, graphics="HIGH"))
    assert json.loads((user_home / "settings.json").read_text()) == {
        "fullscreen": True,
        "graphics": "HIGH",
    }
    assert load_settings() == Settings(True, "HIGH")
    (user_home / "settings.json").write_text('{"fullscreen": "yes", "graphics": "ULTRA"}')
    assert load_settings().graphics is None, "an unknown preset: detect again"
    (user_home / "settings.json").write_text("garbage")
    assert load_settings() == Settings()


def test_display_and_graphics_changes_are_requested_of_the_app_and_kept():
    game = Game(Settings(fullscreen=False, graphics="LOW"), graphics="LOW")
    keys = [i.key for i in game.title_menu.items]
    game.title_menu.index = keys.index("display")
    assert step(game, menu_move_x=1) == ["display"]
    assert game.settings.fullscreen and game.pause_menu.fullscreen
    game.title_menu.index = keys.index("graphics")
    assert step(game, menu_move_x=1) == ["graphics"]
    assert game.graphics == "MEDIUM" == game.settings.graphics == game.pause_menu.graphics


# ----------------------------------------------------------------------------- the tab screen
def test_tab_opens_the_inventory_freezes_the_world_and_tab_closes_it():
    game = playing()
    step(game, 5)
    step(game, toggle_inventory=True)
    assert game.mode is Mode.SCREEN and game.screen.tab_name == "INVENTORY"
    t = game.sim.time
    step(game, 20, forward=1.0)
    assert game.sim.time == t, "the world waits"
    step(game, toggle_inventory=True)
    assert game.mode is Mode.PLAYING


def test_brackets_switch_tabs_and_the_last_tab_is_the_one_that_opens_next_time():
    game = playing()
    step(game, toggle_inventory=True)
    step(game, tab=1)
    assert game.screen.tab_name == "QUESTS"
    step(game, toggle_inventory=True)
    step(game, 3)
    step(game, toggle_inventory=True)
    assert game.screen.tab_name == "QUESTS", "preserved"
    step(game, tab=-1)
    assert game.screen.tab_name == "INVENTORY"
    step(game, tab=1)
    step(game, tab=1)
    assert game.screen.tab_name == "INVENTORY", "wraps"


def test_the_equip_button_installs_and_removes_a_mod():
    game = playing()
    inv = game.sim.player.inventory
    step(game, toggle_inventory=True)
    step(game, menu_move_x=1)  # the deck -> the ping amplifier
    assert inv.stacks[game.screen.slot].item_id == "mod_ping_amp"
    step(game, menu_select=True)
    assert inv.find("mod_ping_amp").equipped
    step(game, interact_pressed=True)
    assert not inv.find("mod_ping_amp").equipped, "[E] too"
    # by mouse: the renderer leaves hotspots; clicking the slot, then the button
    game.screen.hotspots = [
        Hotspot(0, 0, 50, 50, "slot", 2),
        Hotspot(100, 0, 80, 30, "equip", 2),
    ]
    step(game, mouse_x=10, mouse_y=10, click=True)
    assert inv.stacks[game.screen.slot].item_id == "mod_quickjack"
    step(game, mouse_x=120, mouse_y=10, click=True)
    assert inv.find("mod_quickjack").equipped
    step(game, menu_move_x=1)  # the money: not a mod, nothing to equip
    assert not game.toggle_equipped()


def test_the_inventory_grid_scrolls_and_keeps_the_selection_in_view():
    game = playing()
    inv = game.sim.player.inventory
    for _ in range(20):
        inv.add("phone")
    step(game, toggle_inventory=True)
    game.screen.grid_rows_visible = 2
    rows = (len(inv) + GRID_COLUMNS - 1) // GRID_COLUMNS
    step(game, wheel=-1)
    assert game.screen.grid_scroll == 1
    step(game, wheel=-10)
    assert game.screen.grid_scroll == rows - 2, "no further than the last row"
    step(game, wheel=10)
    assert game.screen.grid_scroll == 0
    for _ in range(4):
        step(game, menu_move=1)
    assert game.screen.slot == 4 * GRID_COLUMNS
    row = game.screen.slot // GRID_COLUMNS
    assert game.screen.grid_scroll <= row < game.screen.grid_scroll + 2, "scrolled to it"


def test_the_quest_tab_selects_quests_and_scrolls_their_details():
    game = playing()
    game.sim.load_level(build_night_market())
    game.sim.quests.start("number16")
    step(game, toggle_inventory=True)
    step(game, tab=1)
    screen = game.screen
    screen.detail_height, screen.detail_view = 900.0, 300.0
    step(game, wheel=-2)
    assert screen.detail_scroll == 80.0
    step(game, scroll=1)
    assert screen.detail_scroll == 200.0
    step(game, scroll=10)
    assert screen.detail_scroll == 600.0, "no further than the end"
    step(game, menu_move=1)
    assert screen.quest == 0, "one quest"
