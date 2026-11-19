"""Opt-in rendering regression: replay each scenario in the real OpenGL app and
compare the per-frame framebuffer hashes with goldens recorded on *this* machine.

Framebuffer hashes depend on the GPU/driver, so the goldens are local
(``tests/regression/pixel_goldens``, git-ignored). Record them once with

    UNPREPARED_UPDATE_GOLDENS=1 pytest tests/regression/test_pixel_goldens.py

and every later run will flag any change in what the renderer draws.
"""
from __future__ import annotations

import os
import shutil

import pytest

from tests.parity.display import display_available, run_scenario_subprocess, save_artifacts
from tests.parity.scenarios import SCENARIOS
from tests.regression.goldens_util import PIXEL_GOLDEN_DIR, load_json, save_json, updating

pytestmark = [pytest.mark.regression, pytest.mark.display, pytest.mark.slow,
              pytest.mark.skipif(not display_available(), reason="needs a display or xvfb-run")]

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "textures")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_rendering_matches_local_golden(name, tmp_path):
    path = os.path.join(PIXEL_GOLDEN_DIR, f"{name}.json")
    if not updating() and not os.path.exists(path):
        pytest.skip("no local pixel golden recorded (UNPREPARED_UPDATE_GOLDENS=1 to record)")
    workdir = tmp_path / "run"
    workdir.mkdir()
    for f in os.listdir(ASSET_DIR):
        shutil.copy(os.path.join(ASSET_DIR, f), workdir / f)
    result = run_scenario_subprocess("new", name, str(workdir))
    assert result.error is None, result.error
    hashes = [fr.pixel_hash for fr in result.frames]
    if updating():
        save_json(path, {"scenario": name, "frames": hashes})
        return
    golden = load_json(path)["frames"]
    bad = [i for i, (a, b) in enumerate(zip(golden, hashes)) if a != b]
    if len(golden) != len(hashes) or bad:
        first = bad[0] if bad else min(len(golden), len(hashes))
        png = next((fr.png for fr in result.frames if fr.index == first), None)
        where = save_artifacts(name, first, None, png)
        pytest.fail(f"{name}: rendering changed in {len(bad)} frame(s) (first: {first}; frame count {len(golden)} -> {len(hashes)}) {where}")
