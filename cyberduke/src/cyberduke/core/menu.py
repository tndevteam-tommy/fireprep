"""The game's menus as small state machines: rows, a cursor, and the action a key press
(or a click) asks for. No pygame, no OpenGL - the application maps keys onto them and the
HUD draws them, so the behaviour (what wraps, what a row does, which setting is shown) is
unit-testable.

* ``TitleMenu`` - the start menu: CONTINUE (the newest save), NEW GAME, LOAD GAME, the
  graphics preset, the display mode, QUIT.
* ``PauseMenu`` - [Esc] in game: RESUME, SAVE GAME, LOAD GAME, GRAPHICS, DISPLAY, NEW
  GAME, MAIN MENU, QUIT TO DESKTOP.
* ``LoadMenu`` - the save files, newest first (scrolls), to pick one to load.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

GRAPHICS_CHOICES = ("LOW", "MEDIUM", "HIGH")


class MenuAction(str, Enum):
    NONE = "none"
    RESUME = "resume"
    CONTINUE = "continue"
    NEW_GAME = "new"
    RESTART = "new"  # (the old name for starting over)
    LOAD_MENU = "load"
    SAVE = "save"
    MAIN_MENU = "main"
    QUIT = "quit"
    SETTING_CHANGED = "setting_changed"  # the graphics preset
    DISPLAY_CHANGED = "display_changed"  # windowed <-> fullscreen
    LOAD_SAVE = "load_save"  # a save file picked in the load menu
    BACK = "back"


@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    value: str = ""  # shown on the right (a setting's current value)
    enabled: bool = True
    note: str = ""  # a small line under the label


class Menu:
    """Rows with a cursor. Settings rows ("graphics", "display") change their value with
    [Left]/[Right] (and [Enter]); every other row asks for its action."""

    title = "MENU"

    def __init__(
        self,
        graphics_choices: Sequence[str] = GRAPHICS_CHOICES,
        graphics: str = "MEDIUM",
        fullscreen: bool = False,
    ) -> None:
        self.graphics_choices = list(graphics_choices)
        self.graphics = graphics
        self.fullscreen = fullscreen
        self.index = 0

    # ------------------------------------------------------------------ state
    @property
    def items(self) -> list[MenuItem]:  # pragma: no cover - overridden
        return []

    @property
    def current(self) -> MenuItem:
        return self.items[self.index]

    def reset(self) -> None:
        self.index = 0
        if not self.items[0].enabled:
            self.move(1)

    def _settings_rows(self) -> list[MenuItem]:
        return [
            MenuItem("graphics", "GRAPHICS", self.graphics),
            MenuItem("display", "DISPLAY", "FULLSCREEN" if self.fullscreen else "WINDOWED"),
        ]

    # ------------------------------------------------------------------ input
    def move(self, delta: int) -> None:
        """Up/down through the rows, wrapping at both ends and skipping disabled ones."""
        items = self.items
        count = len(items)
        step = 1 if delta > 0 else -1
        for _ in range(abs(delta)):
            for _ in range(count):
                self.index = (self.index + step) % count
                if items[self.index].enabled:
                    break

    def hover(self, index: int) -> None:
        """The mouse is over row ``index``."""
        if 0 <= index < len(self.items) and self.items[index].enabled:
            self.index = index

    def adjust(self, delta: int) -> MenuAction:
        """Left/right: changes the value of a settings row (nothing elsewhere)."""
        key = self.current.key
        if delta == 0:
            return MenuAction.NONE
        if key == "graphics" and self.graphics_choices:
            choices = self.graphics_choices
            i = choices.index(self.graphics) if self.graphics in choices else 0
            self.graphics = choices[(i + delta) % len(choices)]
            return MenuAction.SETTING_CHANGED
        if key == "display":
            self.fullscreen = not self.fullscreen
            return MenuAction.DISPLAY_CHANGED
        return MenuAction.NONE

    def activate(self) -> MenuAction:
        """Enter on the selected row. A settings row cycles its value instead of acting."""
        item = self.current
        if not item.enabled:
            return MenuAction.NONE
        if item.key in ("graphics", "display"):
            return self.adjust(1)
        return MenuAction(item.key)


class PauseMenu(Menu):
    title = "PAUSED"

    @property
    def items(self) -> list[MenuItem]:
        return [
            MenuItem("resume", "RESUME"),
            MenuItem("save", "SAVE GAME"),
            MenuItem("load", "LOAD GAME"),
            *self._settings_rows(),
            MenuItem("new", "NEW GAME", note="back to the facility - unsaved progress is lost"),
            MenuItem("main", "MAIN MENU", note="unsaved progress is lost"),
            MenuItem("quit", "QUIT TO DESKTOP"),
        ]


class TitleMenu(Menu):
    title = "CYBERDUKE"

    def __init__(self, *args, has_saves: bool = False, **kwargs) -> None:
        self.has_saves = has_saves
        super().__init__(*args, **kwargs)
        self.reset()

    @property
    def items(self) -> list[MenuItem]:
        return [
            MenuItem("continue", "CONTINUE", enabled=self.has_saves, note="the newest save"),
            MenuItem("new", "NEW GAME"),
            MenuItem("load", "LOAD GAME", enabled=self.has_saves),
            *self._settings_rows(),
            MenuItem("quit", "QUIT TO DESKTOP"),
        ]


class LoadMenu:
    """The save files to choose from (``systems.savegame.SaveInfo``s, newest first)."""

    title = "LOAD GAME"
    VISIBLE = 8  # rows on screen at once (the list scrolls)

    def __init__(self, saves: Sequence = ()) -> None:
        self.saves = list(saves)
        self.index = 0
        self.scroll = 0

    @property
    def current(self):
        return self.saves[self.index] if self.saves else None

    def move(self, delta: int) -> None:
        if not self.saves:
            return
        self.index = max(0, min(len(self.saves) - 1, self.index + delta))
        self._keep_in_view()

    def scroll_by(self, rows: int) -> None:
        self.scroll = max(0, min(max(0, len(self.saves) - self.VISIBLE), self.scroll + rows))

    def hover(self, index: int) -> None:
        if 0 <= index < len(self.saves):
            self.index = index

    def _keep_in_view(self) -> None:
        if self.index < self.scroll:
            self.scroll = self.index
        elif self.index >= self.scroll + self.VISIBLE:
            self.scroll = self.index - self.VISIBLE + 1

    def activate(self) -> MenuAction:
        return MenuAction.LOAD_SAVE if self.saves else MenuAction.NONE
