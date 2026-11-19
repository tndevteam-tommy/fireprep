"""
Scripted play sessions (automation/scripts/*.json) driven by real keyboard/mouse events.

Each script asserts gameplay outcomes along the way and captures screenshots into
``test_artifacts/playback`` for the visual test-suite.  Select a subset with
``FIRESTARTER_SCRIPTS=01,05``.
"""

import glob
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = sorted(glob.glob(os.path.join(ROOT, "automation", "scripts", "*.json")))
pytestmark = [pytest.mark.gl, pytest.mark.slow]


def _selected():
    sel = os.environ.get("FIRESTARTER_SCRIPTS", "").strip()
    if not sel:
        return SCRIPTS
    keys = [s.strip() for s in sel.split(",") if s.strip()]
    return [s for s in SCRIPTS if any(os.path.basename(s).startswith(k) for k in keys)]


@pytest.mark.parametrize("script", _selected(), ids=lambda p: os.path.splitext(os.path.basename(p))[0])
def test_playback_script(script, display, artifacts_dir):
    out = os.path.join(artifacts_dir, "playback")
    os.makedirs(out, exist_ok=True)
    name = os.path.splitext(os.path.basename(script))[0]
    env = dict(os.environ, SDL_AUDIODRIVER=os.environ.get("SDL_AUDIODRIVER", "dummy"))
    log_path = os.path.join(out, f"{name}__stdout.log")
    with open(log_path, "w") as log:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--playback", script, "--out", out],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3000)
    report_path = os.path.join(out, f"{name}__report.json")
    assert os.path.exists(report_path), open(log_path).read()[-3000:]
    rep = json.load(open(report_path))
    failed = [x for x in rep["results"] if not x.get("passed")]
    msg = "\n".join(f"step {f.get('step')}: {f.get('type')} {f.get('expr', '')} -- {f.get('message', '')} {f.get('error', '')} "
                    f"value={f.get('value', '')}" for f in failed)
    assert r.returncode == 0 and rep["passed"], msg or (rep.get("fatal") or open(log_path).read()[-3000:])
    assert rep["steps_executed"] == rep["steps_total"]
    for shot in rep["screenshots"]:
        assert os.path.getsize(os.path.join(out, shot)) > 20_000, shot
