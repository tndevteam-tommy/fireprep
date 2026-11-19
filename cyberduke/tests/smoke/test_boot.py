"""Boots the actual game for a handful of frames. Run with `pytest --smoke` under a display."""

import os

import pytest

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def _dummy_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    # the player's own settings and saves are not the test's business (or to be touched)
    monkeypatch.setenv("CYBERDUKE_USER_DIR", str(tmp_path / "Unprepared"))


@pytest.mark.parametrize("backend", ["legacy", "modern"])
def test_game_boots_and_renders_frames(monkeypatch, backend):
    import pygame

    frames = {"n": 0}
    real_get = pygame.event.get

    def get_with_autoquit(*args, **kwargs):
        frames["n"] += 1
        events = real_get(*args, **kwargs)
        if frames["n"] >= int(os.environ.get("CYBERDUKE_SMOKE_FRAMES", "60")):
            events.append(pygame.event.Event(pygame.QUIT))
        return events

    monkeypatch.setattr(pygame.event, "get", get_with_autoquit)

    from cyberduke.main import main

    with pytest.raises(SystemExit):
        main(
            ["--renderer", backend]
        )  # explicit argv: pytest's own arguments must not reach argparse
    assert frames["n"] >= 60


def test_a_new_game_the_deck_screen_a_save_and_the_pause_menu(monkeypatch, tmp_path):
    """Through the real event loop: NEW GAME from the start menu, a few steps, [Tab] to the
    deck screen and [ ] to its QUESTS tab, back out, [Esc] to the pause menu, SAVE GAME,
    resume, quit. The save lands in the user folder."""
    import pygame

    def key(k):
        return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="", scancode=0)

    script = {
        5: [key(pygame.K_RETURN)],  # NEW GAME (no saves: the cursor starts on it)
        20: [key(pygame.K_TAB)],
        25: [key(pygame.K_RIGHTBRACKET)],
        30: [key(pygame.K_TAB)],
        35: [key(pygame.K_ESCAPE)],
        40: [key(pygame.K_DOWN)],  # RESUME -> SAVE GAME
        41: [key(pygame.K_RETURN)],
        45: [key(pygame.K_ESCAPE)],
        55: [pygame.event.Event(pygame.QUIT)],
    }
    frames = {"n": 0}
    real_get = pygame.event.get

    def scripted(*args, **kwargs):
        frames["n"] += 1
        return real_get(*args, **kwargs) + script.get(frames["n"], [])

    monkeypatch.setattr(pygame.event, "get", scripted)

    from cyberduke.main import main

    with pytest.raises(SystemExit):
        main(["--renderer", "legacy", "--graphics", "low"])
    saves = list((tmp_path / "Unprepared" / "Saves").glob("save_*.json"))
    assert len(saves) == 1
