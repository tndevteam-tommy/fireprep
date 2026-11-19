"""Shared helpers for the controller tests: skipping, a headless pygame, and a pad-driven Sim."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


class SkipTest(Exception):
    """Raised to skip a test when pytest is not installed (the fallback runner reports it)."""


def skip(reason):
    try:
        import pytest
    except ImportError:
        raise SkipTest(reason)
    pytest.skip(reason)


def headless_pygame():
    """pygame with the dummy video driver (events + fonts, no window), or skip."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    # The dummy window never has keyboard focus, and SDL drops pad input for unfocused apps
    # (the right thing in the game, where the window is focused).  Must be set before SDL starts.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    try:
        import core.gamepad_device  # noqa: F401  (sets the SDL controller hints before SDL starts)
        import pygame
    except ImportError:
        skip("pygame not installed")
    if not pygame.display.get_init():
        pygame.display.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    if not pygame.font.get_init():
        pygame.font.init()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    return pygame
