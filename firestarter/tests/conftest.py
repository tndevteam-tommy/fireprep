"""Shared pytest configuration: package alias, headless SDL, common fixtures."""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "firestarter" not in sys.modules:
    pkg = types.ModuleType("firestarter")
    pkg.__path__ = [ROOT]
    pkg.__file__ = os.path.join(ROOT, "__init__.py")
    sys.modules["firestarter"] = pkg
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

ARTIFACTS = os.environ.get("FIRESTARTER_ARTIFACTS", os.path.join(ROOT, "test_artifacts"))


def pytest_configure(config):
    config.addinivalue_line("markers", "gl: needs an OpenGL 3.3 context (a display or Xvfb)")
    config.addinivalue_line("markers", "slow: long-running simulation / playback test")


@pytest.fixture
def state():
    from firestarter.game import GameState
    return GameState(tutorial=False)


@pytest.fixture
def tutorial_state():
    from firestarter.game import GameState
    return GameState(tutorial=True)


@pytest.fixture(scope="session")
def artifacts_dir():
    os.makedirs(ARTIFACTS, exist_ok=True)
    return ARTIFACTS
