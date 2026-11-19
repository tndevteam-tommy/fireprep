"""The game's screens drawn into a recording context (no window): the [Tab] screen's
INVENTORY grid and QUESTS tab, the start/pause/load menus and the caught overlay. Checks
what is drawn, that labels stay inside their boxes, and that the clickable hotspots the
renderer leaves behind drive the game."""

from __future__ import annotations

import pytest
from tests.test_render_backends import RecordingContext

from cyberduke.core.screens import GRID_COLUMNS
from cyberduke.game import Game, Mode
from cyberduke.render.hud import Hud
from cyberduke.render.scene import ViewportConfig
from cyberduke.simulation import InputState

DT = 1 / 30
VP = ViewportConfig()


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    import pygame

    pygame.font.init()
    monkeypatch.setenv("CYBERDUKE_USER_DIR", str(tmp_path / "Unprepared"))
    yield


def step(game, n=1, **kw):
    for _ in range(n):
        game.update(DT, InputState(**kw))


def playing() -> Game:
    game = Game()
    step(game, menu_select=True)
    assert game.mode is Mode.PLAYING
    return game


def recorded(hud):
    """Every label the HUD draws, as (text, x, y, w, h)."""
    drawn = []
    original = hud.text

    def text(s, x, y, size=14, color=(255, 255, 255)):
        _, w, h = hud.labels.get(s, size, color)
        drawn.append((s, x, y, w, h))
        original(s, x, y, size, color)

    hud.text = text
    return drawn


def draw(game, hud=None):
    hud = hud or Hud(RecordingContext(), VP.width, VP.height)
    drawn = recorded(hud)
    hud.draw(game.sim, None, DT, game=game)
    return hud, drawn


def click(game, spot):
    step(game, mouse_x=spot.x + spot.w / 2, mouse_y=spot.y + spot.h / 2, click=True)


# ----------------------------------------------------------------------------- inventory
def test_the_inventory_grid_shows_every_item_with_an_icon_and_a_label_that_fits_its_cell():
    game = playing()
    step(game, toggle_inventory=True)
    assert game.mode is Mode.SCREEN and game.screen.tab_name == "INVENTORY"
    hud, drawn = draw(game)
    slots = [s for s in game.screen.hotspots if s.action == "slot"]
    inv = game.sim.player.inventory
    assert len(slots) == min(len(inv), GRID_COLUMNS * game.screen.grid_rows_visible)
    images = [c for c in hud.ctx.calls if c[0] == "image"]
    assert len(images) >= len(slots), "an icon in every slot"
    cell = slots[0].w
    for spot in slots:
        under = [
            d
            for d in drawn
            if spot.x <= d[1] + d[3] / 2 < spot.x + spot.w
            and spot.y + cell <= d[2] < spot.y + spot.h
        ]
        assert under, "a name under the icon"
        for text, x, _y, w, _h in under:
            assert x >= spot.x - 1 and x + w <= spot.x + spot.w + 1, (
                f"{text!r} spills out of its cell"
            )
    texts = [d[0] for d in drawn]
    assert "INVENTORY" in texts and "QUESTS" in texts and "DETAILS" in texts


def test_the_equip_button_installs_and_removes_the_selected_mod():
    game = playing()
    step(game, toggle_inventory=True)
    inv = game.sim.player.inventory
    game.screen.slot = inv.stacks.index(inv.find("mod_ping_amp"))
    _, drawn = draw(game)
    button = [s for s in game.screen.hotspots if s.action == "equip"]
    assert button and any("EQUIP" in d[0] for d in drawn)
    click(game, button[0])
    assert inv.find("mod_ping_amp").equipped
    _, drawn = draw(game)
    assert any("UNEQUIP" in d[0] for d in drawn) and "INSTALLED - ACTIVE" in [d[0] for d in drawn]
    click(game, [s for s in game.screen.hotspots if s.action == "equip"][0])
    assert not inv.find("mod_ping_amp").equipped


def test_a_long_inventory_scrolls_and_only_visible_rows_are_clickable():
    game = playing()
    inv = game.sim.player.inventory
    for name in ("knife", "stim", "parcel", "phone", "cigarettes", "keys_16", "halcyon_badge") * 3:
        inv.add(name)
    step(game, toggle_inventory=True)
    draw(game)
    rows = (len(inv) + GRID_COLUMNS - 1) // GRID_COLUMNS
    assert rows > game.screen.grid_rows_visible, "more than fits"
    first = min(s.arg for s in game.screen.hotspots if s.action == "slot")
    step(game, wheel=-1)
    draw(game)
    assert game.screen.grid_scroll == 1
    assert min(s.arg for s in game.screen.hotspots if s.action == "slot") == first + GRID_COLUMNS


# ----------------------------------------------------------------------------- quests
def test_the_quests_tab_lists_quests_with_their_pictures_and_details():
    game = playing()
    game.sim.quests.start("number16")
    step(game, toggle_inventory=True)
    step(game, tab=1)
    assert game.screen.tab_name == "QUESTS"
    asked = []
    hud = Hud(RecordingContext(), VP.width, VP.height)
    hud.pictures = lambda pic: asked.append(pic.picture_id) or 7
    hud, drawn = draw(game, hud)
    texts = [d[0] for d in drawn]
    assert "SOMETHING SPECIAL" in texts and "CURRENT OBJECTIVE" in texts
    assert asked == ["door16", "datastick"], "a picture of the door and of the stick"
    assert [c for c in hud.ctx.calls if c[0] == "image" and c[1] == 7]
    assert [s for s in game.screen.hotspots if s.action == "quest"]
    assert [s for s in game.screen.hotspots if s.action == "detail"]
    # the last tab is kept when the screen is closed and opened again
    step(game, toggle_inventory=True)
    step(game, toggle_inventory=True)
    assert game.screen.tab_name == "QUESTS"


def test_no_quests_yet_says_so():
    game = playing()
    step(game, toggle_inventory=True)
    step(game, tab=1)
    _, drawn = draw(game)
    assert "NO QUESTS YET" in [d[0] for d in drawn]


# ----------------------------------------------------------------------------- menus
def test_the_start_menu_and_the_load_menu_draw_their_rows_as_hotspots():
    game = Game()
    _, drawn = draw(game)
    texts = [d[0] for d in drawn]
    assert "> NEW GAME" in texts and "  CONTINUE" in texts
    assert len(game.title_menu.hotspots) == len(game.title_menu.items)
    step(game)
    game.title_menu.index = [i.key for i in game.title_menu.items].index("new")
    game = playing()
    game.save()
    game.save()
    step(game, toggle_pause=True)
    assert game.mode is Mode.PAUSED
    game.open_load_menu(Mode.PAUSED)
    _, drawn = draw(game)
    assert len(game.load_menu.hotspots) == 2
    assert any(d[0] == "SAVE FILES  2" for d in drawn)


def test_the_caught_overlay_is_drawn_over_the_hud():
    from cyberduke.core.state import GameState

    game = playing()
    game.sim.state, game.sim.state_timer = GameState.CAUGHT, 0.2
    _, drawn = draw(game)
    assert "YOU WERE CAUGHT" in [d[0] for d in drawn]
