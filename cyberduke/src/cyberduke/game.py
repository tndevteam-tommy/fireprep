"""The game around the simulation: the start menu, pausing, the [Tab] screen, saving and
loading, and the settings. No pygame, no OpenGL - ``main.py`` turns keys into
``InputState``s, feeds them to ``Game.update`` and draws whatever ``Game.mode`` says is on
screen; the app-level things the game cannot do itself (go fullscreen, apply a graphics
preset, quit) come back as ``requests``.

Modes:

* ``TITLE`` - the start menu over the night market going about its business (the street
  runs in the background, seen from a slowly drifting camera);
* ``PLAYING`` - the simulation runs;
* ``PAUSED`` - [Esc]: the pause menu, the world frozen;
* ``SCREEN`` - [Tab]: the inventory / quests screen, the world frozen ([ and ] switch
  tabs; the tab last looked at is the one that opens next time);
* ``LOAD`` - the list of save files (from the start menu or the pause menu).
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path

from cyberduke.config import CLOCK
from cyberduke.core.geometry import Viewpoint
from cyberduke.core.menu import LoadMenu, MenuAction, PauseMenu, TitleMenu
from cyberduke.core.screens import GRID_COLUMNS, TabScreen
from cyberduke.core.userdata import Settings
from cyberduke.levels import build_night_market
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems import savegame

MESSAGE_SECONDS = 3.5
DETAIL_SCROLL_STEP = 40.0  # pixels the quest details move per wheel notch / key press


class Mode(str, Enum):
    TITLE = "TITLE"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    SCREEN = "SCREEN"
    LOAD = "LOAD"


def title_view(t: float) -> Viewpoint:
    """The start menu's camera: drifting slowly up the alley and back, a little above head
    height, looking on up it towards the stalls and the main road's glow."""
    u = 0.5 - 0.5 * math.cos(t * 2.0 * math.pi / 90.0)  # there and back in 90 s
    x = 1.6 + 0.6 * math.sin(t * 0.11)
    z = 1.5 + 22.0 * u
    eye = (x, 2.6 + 0.2 * math.sin(t * 0.07), z)
    target = (-1.2 + 0.8 * math.sin(t * 0.05), 1.7, z + 9.0)
    dx, dy, dz = (target[i] - eye[i] for i in range(3))
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    return Viewpoint(*eye, dx / n, dy / n, dz / n, source="title")


class Game:
    def __init__(
        self,
        settings: Settings | None = None,
        graphics: str = "MEDIUM",
        saves: Path | None = None,
        level=None,
    ) -> None:
        self.settings = settings or Settings()
        self.saves_dir = saves  # None: the user's Saves folder
        self.sim = Simulation(level)  # (its first level - the facility - is the new game)
        self.time = 0.0
        self.requests: list[str] = []  # "display", "graphics", "quit" for the app
        self.message = ""  # "GAME SAVED: ..." on the menus
        self.message_time = 0.0
        self.graphics = graphics
        self.title_menu = TitleMenu(graphics=graphics, fullscreen=self.settings.fullscreen)
        self.pause_menu = PauseMenu(graphics=graphics, fullscreen=self.settings.fullscreen)
        self.load_menu = LoadMenu()
        self.screen = TabScreen()
        self.load_return = Mode.TITLE  # where [Esc] leaves the load menu for
        # Where the pointer was last frame: only a pointer that *moves* over a menu row
        # selects it - one resting on a row must not keep stealing the keyboard's choice.
        self._last_mouse: tuple[float, float] | None = None
        self._mouse_moved = False
        self.mode = Mode.TITLE
        self.to_title()

    # ------------------------------------------------------------------ modes
    @property
    def menu(self):
        """The menu on screen (for the HUD), if any."""
        return {
            Mode.TITLE: self.title_menu,
            Mode.PAUSED: self.pause_menu,
            Mode.LOAD: self.load_menu,
        }.get(self.mode)

    @property
    def shows_cursor(self) -> bool:
        return self.mode is not Mode.PLAYING

    def to_title(self) -> None:
        """The start menu, with the night market as the backdrop."""
        self.mode = Mode.TITLE
        self.sim.load_level(build_night_market())
        self.sim.clock.set(CLOCK.title_at)  # (the street of the start menu: early evening)
        self.sim.camera_override = title_view(self.time)
        self.title_menu.has_saves = bool(self.saves())
        self.title_menu.reset()

    def start_new_game(self) -> None:
        self.sim.camera_override = None
        self.sim.new_game()
        self.screen = TabScreen()
        self.mode = Mode.PLAYING

    def saves(self) -> list[savegame.SaveInfo]:
        return savegame.list_saves(self.saves_dir)

    def save(self) -> Path | None:
        if self.sim.state.value != "PLAYING":
            # (saving on the red screen would be a way out of having been caught)
            self.say("COULD NOT SAVE: NOT WHILE YOU ARE BEING CAUGHT")
            return None
        try:
            path = savegame.save_game(self.sim, self.saves_dir)
        except (OSError, savegame.SaveError) as e:
            self.say(f"COULD NOT SAVE: {e}")
            return None
        self.say(f"GAME SAVED: {path.name}")
        return path

    def load(self, path: Path) -> bool:
        try:
            savegame.load_game(self.sim, path)
        except (OSError, savegame.SaveError, KeyError, ValueError) as e:
            self.say(f"COULD NOT LOAD: {e}")
            return False
        self.sim.camera_override = None
        self.screen = TabScreen()
        self.mode = Mode.PLAYING
        self.sim.toast(f"LOADED: {Path(path).name}")
        return True

    def open_load_menu(self, back_to: Mode) -> None:
        self.load_menu = LoadMenu(self.saves())
        self.load_return = back_to
        self.mode = Mode.LOAD

    def say(self, text: str) -> None:
        self.message = text
        self.message_time = MESSAGE_SECONDS

    # ------------------------------------------------------------------ per frame
    def update(self, dt: float, inp: InputState) -> list[str]:
        """One frame. Returns what the app must do (and clears the list next frame)."""
        self.requests = []
        self.time += dt
        mouse = (inp.mouse_x, inp.mouse_y) if inp.mouse_x >= 0 else None
        # (the first frame a pointer shows up is where it rests, not a move)
        self._mouse_moved = None not in (mouse, self._last_mouse) and mouse != self._last_mouse
        self._last_mouse = mouse
        if self.message_time > 0.0:
            self.message_time = max(0.0, self.message_time - dt)
        handler = {
            Mode.TITLE: self._title,
            Mode.PLAYING: self._playing,
            Mode.PAUSED: self._paused,
            Mode.SCREEN: self._screen,
            Mode.LOAD: self._load,
        }[self.mode]
        handler(dt, inp)
        return self.requests

    def _menu_input(self, menu, inp: InputState) -> MenuAction:
        if inp.menu_move:
            menu.move(inp.menu_move)
        action = MenuAction.NONE
        if inp.menu_move_x:
            action = menu.adjust(inp.menu_move_x)
        hot = getattr(menu, "hotspots", [])
        if inp.mouse_x >= 0 and (self._mouse_moved or inp.click):
            for k, (x, y, w, h) in enumerate(hot):
                if x <= inp.mouse_x < x + w and y <= inp.mouse_y < y + h:
                    menu.hover(k)
                    if inp.click:
                        action = menu.activate()
                    break
        if inp.menu_select:
            action = menu.activate()
        if action is MenuAction.SETTING_CHANGED:
            self.graphics = menu.graphics
            self.title_menu.graphics = self.pause_menu.graphics = menu.graphics
            self.settings.graphics = menu.graphics
            self.requests.append("graphics")
        elif action is MenuAction.DISPLAY_CHANGED:
            self.settings.fullscreen = menu.fullscreen
            self.title_menu.fullscreen = self.pause_menu.fullscreen = menu.fullscreen
            self.requests.append("display")
        return action

    def _title(self, dt: float, inp: InputState) -> None:
        sim = self.sim
        sim.camera_override = title_view(self.time)
        sim.player.x, sim.player.z = 0.0, -1.2  # out of everybody's way, by the staff door
        sim.update(dt, InputState())
        action = self._menu_input(self.title_menu, inp)
        if action is MenuAction.NEW_GAME:
            self.start_new_game()
        elif action is MenuAction.CONTINUE:
            saves = self.saves()
            if saves:
                self.load(saves[0].path)
        elif action is MenuAction.LOAD_MENU:
            self.open_load_menu(Mode.TITLE)
        elif action is MenuAction.QUIT:
            self.requests.append("quit")

    def _playing(self, dt: float, inp: InputState) -> None:
        if inp.toggle_pause:
            self.pause_menu.reset()
            self.mode = Mode.PAUSED
            return
        if inp.toggle_inventory and self.sim.state.value == "PLAYING":
            self.screen.open()
            self.mode = Mode.SCREEN
            return
        self.sim.update(dt, inp)

    def _paused(self, dt: float, inp: InputState) -> None:
        if inp.toggle_pause or inp.menu_back:
            self.mode = Mode.PLAYING
            return
        action = self._menu_input(self.pause_menu, inp)
        if action is MenuAction.RESUME:
            self.mode = Mode.PLAYING
        elif action is MenuAction.SAVE:
            self.save()
        elif action is MenuAction.LOAD_MENU:
            self.open_load_menu(Mode.PAUSED)
        elif action is MenuAction.NEW_GAME:
            self.start_new_game()
        elif action is MenuAction.MAIN_MENU:
            self.to_title()
        elif action is MenuAction.QUIT:
            self.requests.append("quit")

    def _load(self, dt: float, inp: InputState) -> None:
        menu = self.load_menu
        if self.load_return is Mode.TITLE:  # the street goes on behind the menu
            self.sim.camera_override = title_view(self.time)
            self.sim.update(dt, InputState())
        if inp.toggle_pause or inp.menu_back:
            self.mode = self.load_return
            return
        if inp.menu_move:
            menu.move(inp.menu_move)
        if inp.wheel:
            menu.scroll_by(-inp.wheel)
        action = MenuAction.NONE
        if inp.mouse_x >= 0 and (self._mouse_moved or inp.click):
            for k, (x, y, w, h) in enumerate(getattr(menu, "hotspots", [])):
                if x <= inp.mouse_x < x + w and y <= inp.mouse_y < y + h:
                    menu.hover(k + menu.scroll)
                    if inp.click:
                        action = menu.activate()
                    break
        if inp.menu_select:
            action = menu.activate()
        if action is MenuAction.LOAD_SAVE and menu.current is not None:
            self.load(menu.current.path)

    # ------------------------------------------------------------------ the tab screen
    def _screen(self, dt: float, inp: InputState) -> None:
        screen, sim = self.screen, self.sim
        inv = sim.player.inventory
        quests = sim.quests.quests
        screen.clamp(len(inv), len(quests))
        if inp.toggle_inventory or inp.toggle_pause or inp.menu_back:
            screen.close()
            self.mode = Mode.PLAYING
            return
        if inp.tab:
            screen.switch(inp.tab)
        hit = screen.hit(inp.mouse_x, inp.mouse_y) if inp.mouse_x >= 0 else None
        if inp.click and hit is not None:
            if hit.action == "tab":
                screen.tab = hit.arg
            elif hit.action == "slot":
                screen.slot = hit.arg
            elif hit.action == "equip":
                self.toggle_equipped()
            elif hit.action == "quest":
                screen.move_quest(hit.arg - screen.quest, len(quests))
        if screen.tab_name == "INVENTORY":
            if inp.menu_move or inp.menu_move_x:
                screen.move_slot(inp.menu_move_x, inp.menu_move, len(inv))
            if inp.wheel:
                screen.scroll_grid(-inp.wheel, len(inv))
            if inp.scroll:
                screen.scroll_grid(inp.scroll * 2, len(inv))
            if inp.menu_select or inp.interact_pressed:
                self.toggle_equipped()
        else:
            if inp.menu_move:
                screen.move_quest(inp.menu_move, len(quests))
            over_list = hit is not None and hit.action == "quest"
            if inp.wheel:
                if over_list:
                    screen.list_scroll = max(
                        0,
                        min(
                            max(0, len(quests) - screen.list_rows_visible),
                            screen.list_scroll - inp.wheel,
                        ),
                    )
                else:
                    screen.scroll_detail(-inp.wheel * DETAIL_SCROLL_STEP)
            if inp.scroll:
                screen.scroll_detail(inp.scroll * DETAIL_SCROLL_STEP * 3)
            if inp.menu_move_x:
                screen.scroll_detail(inp.menu_move_x * DETAIL_SCROLL_STEP)

    def toggle_equipped(self) -> bool:
        """The EQUIP / UNEQUIP button on the selected item (cyber mods only)."""
        inv = self.sim.player.inventory
        if not 0 <= self.screen.slot < len(inv):
            return False
        stack = inv.stacks[self.screen.slot]
        if not stack.item.equippable:
            return False
        on = inv.toggle_equipped(stack)
        self.sim.hacking.log.append(f"{'INSTALLED' if on else 'REMOVED'}: {stack.item.name}")
        return True

    @property
    def grid_columns(self) -> int:
        return GRID_COLUMNS
