"""Shared fixtures.

Environment variables honoured by the display/parity tests:
  UNPREPARED_TEST_DISPLAY=0   skip tests that need an OpenGL window
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from types import SimpleNamespace

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TESTS_DIR)
BASELINE_DIR = os.path.join(TESTS_DIR, "baseline")
ASSET_DIR = os.path.join(PROJECT_DIR, "assets", "textures")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def load_baseline() -> SimpleNamespace:
    """Import the frozen original modules (models, world, main) under their own names."""
    saved = {name: sys.modules.pop(name, None) for name in ("models", "world", "main")}
    sys.path.insert(0, BASELINE_DIR)
    try:
        models = importlib.import_module("models")
        world = importlib.import_module("world")
        main = importlib.import_module("main")
    finally:
        sys.path.remove(BASELINE_DIR)
        legacy = {name: sys.modules.pop(name, None) for name in ("models", "world", "main")}
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
    # keep legacy modules importable by each other via their attributes
    return SimpleNamespace(models=models, world=world, main=main, modules=legacy)


@pytest.fixture(scope="session")
def baseline() -> SimpleNamespace:
    return load_baseline()


@pytest.fixture
def settings():
    from unprepared.config import GameSettings
    return GameSettings(echo_messages_to_stdout=False, random_seed=1234, max_log_messages=60)


@pytest.fixture
def game(settings, tmp_path):
    from unprepared.game import Game
    return Game(settings=settings, save_file=str(tmp_path / "savegame.json"))


@pytest.fixture
def world(settings):
    from unprepared.world import World, get_level
    from unprepared.core.messages import MessageLog
    log = MessageLog(200)                     # (a town of sixty fills the game's six lines in an hour)
    w = World(get_level("town"), settings, notify_callback=log.add)
    w.log = log
    return w


@pytest.fixture
def asset_workdir(tmp_path):
    """Temp working directory containing copies of the texture assets (the legacy
    code loads textures from the CWD)."""
    for name in os.listdir(ASSET_DIR):
        shutil.copy(os.path.join(ASSET_DIR, name), tmp_path / name)
    return tmp_path


def said(messages, kind, **fmt):
    """True if one of the log lines quotes a phrasing of dialogue kind ``kind`` (lines vary)."""
    from unprepared.core import dialogue
    options = {text.format(**fmt) if fmt else text for text in dialogue.LINES[kind]}
    if not fmt:
        options |= {text for text in dialogue.LINES[kind]}
    for message in messages:
        if '"' in message:
            quoted = message.split('"', 1)[1].rsplit('"', 1)[0]
            if quoted in options:
                return True
    return False


def bubbled(npc, kind, **fmt) -> bool:
    """True if the NPC's current speech bubble is a phrasing of dialogue kind ``kind``
    (town chatter is shown above heads only, never logged)."""
    from unprepared.core import dialogue
    if npc.speech is None:
        return False
    options = {text.format(**fmt) if fmt else text for text in dialogue.LINES[kind]}
    return npc.speech.text in options
