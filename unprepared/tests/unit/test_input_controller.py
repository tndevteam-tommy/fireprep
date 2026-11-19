"""pygame event -> Game action mapping (no window needed: events are injected)."""
import pygame
import pytest

from unprepared.input import KeyBindings, PygameInputController
from unprepared.systems import MoveIntent


class RecordingGame:
    """Stand-in for Game that records which actions were invoked."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name,) + args)
        return record


def feed(monkeypatch, events):
    monkeypatch.setattr(pygame.event, "get", lambda: events)


@pytest.fixture
def controller():
    return PygameInputController()


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)


def test_quit_and_mouse(monkeypatch, controller):
    g = RecordingGame()
    feed(monkeypatch, [pygame.event.Event(pygame.QUIT), pygame.event.Event(pygame.MOUSEMOTION, rel=(3, -4), pos=(0, 0), buttons=(0, 0, 0))])
    controller.process_events(g)
    assert g.calls == [("request_quit",), ("look", 3, -4)]


@pytest.mark.parametrize("k,action", [
    (pygame.K_ESCAPE, ("escape",)), (pygame.K_F5, ("save_game",)), (pygame.K_F9, ("load_game",)),
    (pygame.K_e, ("interact",)), (pygame.K_1, ("toggle_torch",)), (pygame.K_LCTRL, ("toggle_crouch",)),
    (pygame.K_t, ("wait",)), (pygame.K_y, ("confirm_yes",)), (pygame.K_n, ("confirm_no",)),
    (pygame.K_UP, ("cursor_up",)), (pygame.K_DOWN, ("cursor_down",)),
    (pygame.K_RETURN, ("transfer_confirm",)), (pygame.K_TAB, ("tab",)), (pygame.K_x, ("loot_all",)),
    (pygame.K_LEFT, ("lockpick_rotate", -1)), (pygame.K_RIGHT, ("lockpick_rotate", 1)), (pygame.K_SPACE, ("lockpick_attempt",)),
])
def test_key_mapping(monkeypatch, controller, k, action):
    g = RecordingGame()
    feed(monkeypatch, [key(k)])
    controller.process_events(g)
    assert g.calls == [action]


def test_unbound_key_ignored(monkeypatch, controller):
    g = RecordingGame()
    feed(monkeypatch, [key(pygame.K_z), key(pygame.K_p), key(pygame.K_r), pygame.event.Event(pygame.KEYUP, key=pygame.K_e)])   # P and R are gone
    controller.process_events(g)
    assert g.calls == []


def test_rebinding(monkeypatch):
    g = RecordingGame()
    c = PygameInputController(KeyBindings(interact=pygame.K_f))
    feed(monkeypatch, [key(pygame.K_e), key(pygame.K_f)])
    c.process_events(g)
    assert g.calls == [("interact",)]


def test_poll_movement(monkeypatch, controller):
    held = {pygame.K_w, pygame.K_d, pygame.K_LSHIFT}

    class Keys:
        def __getitem__(self, k):
            return k in held

    monkeypatch.setattr(pygame.key, "get_pressed", lambda: Keys())
    assert controller.poll_movement() == MoveIntent(forward=True, right=True, sprint=True)


def test_controls_help_text():
    lines = KeyBindings().controls_help()
    assert lines[0] == "CONTROLS:" and len(lines) == 9 and any("[~] World Events Log" in line for line in lines)
    assert not any("Scroll Inventory" in line for line in lines)          # (there is no inventory panel to scroll)
    assert not any("[P]" in line for line in lines) and any("Crouch + [E]" in line for line in lines)
    # the obvious ones are left out to make room; save/load are the *quick* ones
    joined = " ".join(lines)
    assert "W/A/S/D" not in joined and "Mouse" not in joined and "ESC" not in joined
    assert "[F5] Quick Save" in joined and "[F9] Quick Load" in joined


def test_the_backquote_key_toggles_the_event_log(monkeypatch, controller):
    """[`] - labelled [~], its shifted face, which is how most players know the key."""
    g = RecordingGame()
    assert KeyBindings().toggle_log == pygame.K_BACKQUOTE and KeyBindings.LOG_KEY_LABEL == "~"
    feed(monkeypatch, [key(pygame.K_BACKQUOTE)])
    controller.process_events(g)
    assert ("toggle_event_log",) in g.calls


def test_controls_help_fits_the_panel():
    """Every line must fit inside the controls panel at the HUD's small font (no clipping)."""
    from unprepared.config import HudLayout, WindowSettings
    pygame.font.init()
    w = WindowSettings()
    font = pygame.font.SysFont(w.font_name, w.font_small_size)
    x1, y1, x2, y2 = HudLayout().controls_panel
    lines = KeyBindings().controls_help()
    for line in lines:
        assert font.size(line)[0] <= (x2 - x1) - 20, line
    assert 10 + 22 * len(lines) <= (y2 - y1)


def test_transfer_help_lines_fit_the_panel():
    """Every transfer policy's help line fits the modal panel at the HUD's small font."""
    from unprepared.config import HudLayout, WindowSettings
    from unprepared.systems.transfer import ContainerPolicy, PickpocketPolicy, TradePolicy, TransferPolicy
    pygame.font.init()
    w = WindowSettings()
    font = pygame.font.SysFont(w.font_name, w.font_small_size)
    x1, _, x2, _ = HudLayout().trade_panel
    for policy in (TransferPolicy, TradePolicy, PickpocketPolicy, ContainerPolicy):
        assert font.size(policy.help_text)[0] <= (x2 - x1) - 56, policy.help_text
