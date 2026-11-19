"""The [Tab] screen: the netrunner's INVENTORY and QUESTS, as tabs.

``TabScreen`` holds what the screen is showing - which tab (kept when the screen is
closed and opened again), the selected item and how far the grid is scrolled, the
selected quest and how far its details are scrolled - and turns input into changes of
that (keys, the mouse wheel, clicks). It knows nothing about pixels: the renderer lays the
screen out, tells it how much fits (``grid_rows_visible``, ``detail_view``), and leaves
a list of ``Hotspot``s - rectangles and what clicking them does - for the next frame's
clicks. No pygame, no OpenGL.
"""

from __future__ import annotations

from dataclasses import dataclass

TABS = ("INVENTORY", "QUESTS")
GRID_COLUMNS = 5


@dataclass(frozen=True)
class Hotspot:
    x: float
    y: float
    w: float
    h: float
    action: str  # "tab", "slot", "equip", "quest", "detail", "grid"
    arg: int = 0

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


class TabScreen:
    def __init__(self) -> None:
        self.is_open = False
        self.tab = 0  # kept between openings
        self.slot = 0  # the selected inventory slot
        self.grid_scroll = 0  # rows scrolled off the top of the grid
        self.grid_rows_visible = 3  # (set by the renderer)
        self.quest = 0  # the selected quest
        self.list_scroll = 0
        self.list_rows_visible = 6  # (set by the renderer)
        self.detail_scroll = 0.0  # pixels scrolled down the quest details
        self.detail_height = 0.0  # (set by the renderer) how tall the details are...
        self.detail_view = 1.0  # ... and how much of them shows
        self.hotspots: list[Hotspot] = []  # (set by the renderer)
        self.pending_equip = False  # the equip button was pressed (the game acts on it)

    @property
    def tab_name(self) -> str:
        return TABS[self.tab]

    # ------------------------------------------------------------------ open/close/tabs
    def open(self) -> None:
        self.is_open = True
        self.hotspots = []

    def close(self) -> None:
        self.is_open = False
        self.hotspots = []

    def switch(self, delta: int) -> None:
        self.tab = (self.tab + delta) % len(TABS)
        self.hotspots = []

    # ------------------------------------------------------------------ inventory
    def move_slot(self, dx: int, dy: int, count: int) -> None:
        """Arrow keys over the grid (``count`` items in it)."""
        if count <= 0:
            self.slot = 0
            return
        slot = self.slot + dx + dy * GRID_COLUMNS
        self.slot = max(0, min(count - 1, slot))
        self._keep_slot_in_view()

    def _keep_slot_in_view(self) -> None:
        row = self.slot // GRID_COLUMNS
        if row < self.grid_scroll:
            self.grid_scroll = row
        elif row >= self.grid_scroll + self.grid_rows_visible:
            self.grid_scroll = row - self.grid_rows_visible + 1

    def scroll_grid(self, rows: int, count: int) -> None:
        total_rows = (count + GRID_COLUMNS - 1) // GRID_COLUMNS
        most = max(0, total_rows - self.grid_rows_visible)
        self.grid_scroll = max(0, min(most, self.grid_scroll + rows))

    def clamp(self, items: int, quests: int) -> None:
        self.slot = max(0, min(max(0, items - 1), self.slot))
        self.quest = max(0, min(max(0, quests - 1), self.quest))
        self.scroll_grid(0, items)

    # ------------------------------------------------------------------ quests
    def move_quest(self, delta: int, count: int) -> None:
        if count <= 0:
            return
        quest = max(0, min(count - 1, self.quest + delta))
        if quest != self.quest:
            self.quest = quest
            self.detail_scroll = 0.0
        if self.quest < self.list_scroll:
            self.list_scroll = self.quest
        elif self.quest >= self.list_scroll + self.list_rows_visible:
            self.list_scroll = self.quest - self.list_rows_visible + 1

    def scroll_detail(self, pixels: float) -> None:
        most = max(0.0, self.detail_height - self.detail_view)
        self.detail_scroll = max(0.0, min(most, self.detail_scroll + pixels))

    # ------------------------------------------------------------------ the mouse
    def hit(self, x: float, y: float) -> Hotspot | None:
        """The topmost hotspot under the cursor (the last drawn wins)."""
        for spot in reversed(self.hotspots):
            if spot.contains(x, y):
                return spot
        return None
