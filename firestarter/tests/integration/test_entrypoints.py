"""The documented entry points work (README 'Running the Simulation')."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run(args, timeout=300, env=None):
    e = dict(os.environ, SDL_AUDIODRIVER="dummy")
    if env:
        e.update(env)
    return subprocess.run([sys.executable, os.path.join(ROOT, "main.py")] + args, cwd=ROOT, env=e,
                          capture_output=True, text=True, timeout=timeout)


def test_help_lists_every_option():
    r = _run(["--help"])
    assert r.returncode == 0
    for opt in ("--headless", "--playback", "--record", "--preset", "--fixed-dt", "--no-audio", "--start-hour"):
        assert opt in r.stdout


def test_headless_simulation_check():
    r = _run(["--headless"], timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Verification passed" in r.stdout


def test_package_imports_cleanly_in_any_order():
    code = ("import sys, types, os; sys.path.insert(0, %r); p = types.ModuleType('firestarter'); p.__path__ = [%r]; "
            "sys.modules['firestarter'] = p; import firestarter.forestry.tree, firestarter.simulation.world_manager, "
            "firestarter.input.input_manager, firestarter.audio, firestarter.economy, firestarter.recipes, firestarter.bushcraft; "
            "from firestarter.simulation import WorldManager; print('ok')") % (ROOT, ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
