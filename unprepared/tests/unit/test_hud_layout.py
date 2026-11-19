"""The HUD's pixel arithmetic that needs no window: word wrap, the clock, and where the tutorial
panel goes so that it covers nothing that matters."""
from types import SimpleNamespace

import pytest

from unprepared.config import HudColors, HudLayout
from unprepared.rendering.hud import HudRenderer, clock_text, wrap_text


class FakeText:
    """Measures seven pixels a character, fourteen tall - enough for layout arithmetic."""
    font = font_small = "font"

    def measure(self, text, font=None):
        return 7 * len(text), 14


def hud(width=1024, height=768):
    return HudRenderer(FakeText(), HudLayout.for_window(width, height), HudColors(), [])


def fake_game(hint, **flags):
    return SimpleNamespace(tutorial_hint=hint, in_transfer=flags.get("in_transfer", False),
                           in_lockpicking=flags.get("in_lockpicking", False))


def overlaps(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def rect(panel):
    x1, y1, w, h = panel[:4]
    return x1, y1, x1 + w, y1 + h


LONG_HINT = ("Cross the street to Barnaby's store, the door under the sign with the scales. The door is locked. "
             "Crouch with [Ctrl], look at the door and press [E] to pick the lock. Turn the pick with [LEFT] and "
             "[RIGHT] until the ring glows, then press [SPACE]. If you miss, the pick snaps, but you will be given "
             "another. Stay out of sight of the guards while you work.")


class TestWrapAndClock:
    def test_wrap_text_keeps_every_word_and_splits_the_unsplittable(self):
        measure = FakeText().measure
        lines = wrap_text("one two three four five six", 70, measure)       # ten characters a line
        assert lines == ["one two", "three four", "five six"]
        assert wrap_text("abcdefghijklmnop", 70, measure) == ["abcdefghij", "klmnop"]
        assert wrap_text("", 70, measure) == [""]

    def test_clock_text_shows_hours_and_the_minutes_accumulated(self):
        clock = SimpleNamespace(hour=7, minute_accumulator=0.0, minutes_per_hour=24.0)
        assert clock_text(clock) == "07:00"
        clock.minute_accumulator = 12.0
        assert clock_text(clock) == "07:30"
        clock.minute_accumulator = 24.0                                     # never 07:60
        assert clock_text(clock) == "07:59"


class TestTutorialPanelPlacement:
    def test_no_hint_no_panel(self):
        assert hud()._tutorial_panel(fake_game("")) is None

    def test_in_the_street_it_sits_under_the_controls_panel(self):
        h = hud()
        panel = h._tutorial_panel(fake_game("Light your torch."))
        cx1, cy1, cx2, cy2 = h.layout.controls_panel
        assert panel[0] == cx1 and panel[2] == cx2 - cx1 and panel[1] == cy2 + h.TUTORIAL_GAP
        assert rect(panel)[3] < h.layout.log_panel[1]

    def test_beside_a_shelf_or_a_deal_it_covers_neither_the_screen_nor_the_log(self):
        h = hud()
        panel = h._tutorial_panel(fake_game(LONG_HINT, in_transfer=True))
        assert not overlaps(rect(panel), h.layout.trade_panel)
        assert not overlaps(rect(panel), h.layout.log_panel)
        assert panel[1] == h.layout.trade_panel[1]                           # level with the screen's top
        assert rect(panel)[2] <= h.layout.width - 10

    def test_beside_the_lock_likewise(self):
        h = hud()
        panel = h._tutorial_panel(fake_game(LONG_HINT, in_lockpicking=True))
        assert not overlaps(rect(panel), h.layout.lockpick_panel)
        assert not overlaps(rect(panel), h.layout.log_panel)

    def test_in_a_small_window_it_takes_the_inventory_corner_behind_a_screen(self):
        h = hud(800, 600)
        panel = h._tutorial_panel(fake_game(LONG_HINT, in_transfer=True))
        ix1, iy1, ix2, iy2 = h.layout.inventory_panel
        assert panel[0] == ix1 and panel[2] == ix2 - ix1 and rect(panel)[3] == iy2
        tx1, ty1, tx2, ty2 = h.layout.trade_panel
        assert not overlaps(rect(panel), (tx1, ty1, tx2, ty1 + 140))         # the screen's title and headings stay clear
        assert not overlaps(rect(panel), h.layout.log_panel)

    @pytest.mark.parametrize("size", [(1024, 768), (1280, 720), (1920, 1080), (800, 600)])
    def test_it_never_leaves_the_window(self, size):
        h = hud(*size)
        for flags in ({}, {"in_transfer": True}, {"in_lockpicking": True}):
            panel = h._tutorial_panel(fake_game(LONG_HINT, **flags))
            x1, y1, x2, y2 = rect(panel)
            assert 0 <= x1 < x2 <= h.layout.width and 0 <= y1 < y2 <= h.layout.height


class TestTheStatusBar:
    ITEMS = [("Time: 07:00", "07:00", "w"), ("Gold: 60.0g", "60.0g", "g"), ("Torch [1]: OFF", "Torch: OFF", "w"),
             ("Stance [Ctrl]: Standing", "Standing", "w"), ("Events [~]: OFF", "Log [~]: OFF", "h")]

    def test_the_usual_places_in_a_wide_window(self):
        laid = hud(1024, 768).status_layout(self.ITEMS)
        assert [x for _, x, _ in laid] == list(HudRenderer.STATUS_X)
        assert [text for text, _, _ in laid] == [full for full, _, _ in self.ITEMS]

    @pytest.mark.parametrize("width", [520, 640, 800])
    def test_a_narrow_window_spreads_them_so_nothing_runs_off_or_overlaps(self, width):
        laid = hud(width, 600).status_layout(self.ITEMS)
        ends = [x + 7 * len(text) for text, x, _ in laid]
        assert all(end < x for end, (_, x, _) in zip(ends, laid[1:]))
        assert ends[-1] <= width - 10
        assert [color for _, _, color in laid] == [color for _, _, color in self.ITEMS]
        if width >= 800:
            assert laid[-1][0] == "Events [~]: OFF"                   # (still the long forms when they fit)
