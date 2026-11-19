"""Headless golden-state regression: every scripted scenario is replayed through
the real input controller + ``Game`` and the per-frame state must match the
recorded golden exactly.

The golden stores a digest of *all* frames plus the full snapshots of the
scenario's capture frames (so a failure shows *what* changed), and is committed
to git so behaviour changes are explicit, reviewable diffs.

    UNPREPARED_UPDATE_GOLDENS=1 pytest tests/regression     # re-record after an intended change
"""
from __future__ import annotations

import os

import pytest

from tests.parity.harness import dry_run
from tests.parity.scenarios import SCENARIOS
from tests.regression.goldens_util import STATE_GOLDEN_DIR, describe_diff, digest, load_json, normalize, save_json, updating

pytestmark = pytest.mark.regression

# scenario name -> (frames played, still running after the last one). A replay is deterministic - the
# goldens depend on it - so the golden test's run also answers ``test_every_scenario_plays_to_the_end``
# instead of every scenario being played a second time.
_PLAYED: dict = {}


def play(name: str):
    """Replay a scenario headlessly; returns (scenario, per-frame snapshots, the game)."""
    scenario = SCENARIOS[name]()
    frames = []
    game = dry_run(scenario, frames)
    _PLAYED[name] = (len(frames), frames[-1]["running"] if frames else None)
    return scenario, frames, game


def record(name: str) -> dict:
    scenario, frames, game = play(name)
    captures = sorted(scenario.capture_frames | {len(frames) - 1})
    return {
        "scenario": name,
        "frames": len(frames),
        "digest": digest(frames),
        "captures": {str(i): normalize(frames[i]) for i in captures if i < len(frames)},
        "final_messages": game.messages.messages,
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_matches_golden(name):
    path = os.path.join(STATE_GOLDEN_DIR, f"{name}.json")
    actual = record(name)
    if updating() or not os.path.exists(path):
        save_json(path, actual)
        if not updating():
            pytest.fail(f"no golden existed for {name!r}; recorded {path} - re-run to verify")
        return
    golden = load_json(path)
    if golden["digest"] != actual["digest"] or golden["frames"] != actual["frames"]:
        diffs = describe_diff(golden["captures"], actual["captures"])
        diffs += describe_diff(golden["final_messages"], actual["final_messages"], "/final_messages")
        pytest.fail(f"{name}: state diverged from golden ({golden['frames']} -> {actual['frames']} frames)\n"
                    + "\n".join(diffs[:40]) + ("\n..." if len(diffs) > 40 else "")
                    + "\n(set UNPREPARED_UPDATE_GOLDENS=1 to accept an intended change)")


def test_every_scenario_plays_to_the_end():
    """Every frame of every script is played - except by a scenario that quits, which stops at the
    quitting frame (the headless replay never sees the trailing QUIT event a windowed run gets)."""
    for name in SCENARIOS:
        if name not in _PLAYED:
            play(name)
        played, running = _PLAYED[name]
        assert played >= 1, name
        assert played == len(SCENARIOS[name]()) or running is False, name
