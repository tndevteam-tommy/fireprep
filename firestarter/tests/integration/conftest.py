import os
import shutil

import pytest


def have_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@pytest.fixture(scope="session")
def display():
    if not have_display():
        pytest.skip("no display: run under xvfb-run (run_tests.sh does this automatically)")
    return os.environ.get("DISPLAY")
