"""Helpers to run a display-backed subprocess (optionally under Xvfb)."""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
from typing import List, Optional

import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "tests", "parity", "_artifacts")


def display_available() -> bool:
    if os.environ.get("UNPREPARED_TEST_DISPLAY") == "0":
        return False
    return bool(os.environ.get("DISPLAY")) or shutil.which("xvfb-run") is not None


def _command_prefix() -> List[str]:
    if os.environ.get("DISPLAY"):
        return []
    return ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24"]


def run_scenario_subprocess(impl: str, scenario: str, workdir: str, timeout: int = 600, mask_rects=()):
    out = os.path.join(workdir, f"{impl}_{scenario}.pkl")
    cmd = _command_prefix() + [sys.executable, "-m", "tests.parity.run_scenario", "--impl", impl,
                               "--scenario", scenario, "--workdir", workdir, "--out", out]
    for rect in mask_rects:
        cmd += ["--mask", ",".join(str(int(v)) for v in rect)]
    env = dict(os.environ, SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1")
    if not os.environ.get("DISPLAY"):
        env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")     # deterministic Mesa software rendering
    proc = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(out):
        pytest.fail(f"{impl}/{scenario} runner failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}")
    with open(out, "rb") as f:
        return pickle.load(f)


def save_artifacts(scenario: str, frame: int, legacy_png: Optional[bytes], new_png: Optional[bytes]) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    paths = []
    for label, png in (("legacy", legacy_png), ("new", new_png)):
        if png:
            path = os.path.join(ARTIFACT_DIR, f"{scenario}_f{frame}_{label}.png")
            with open(path, "wb") as f:
                f.write(png)
            paths.append(path)
    return ", ".join(paths)
