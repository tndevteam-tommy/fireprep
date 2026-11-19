"""End-to-end parity with the frozen prototype.

The game has since diverged from the prototype on purpose (quirk fixes, NPC
routines, a bigger town), so what remains comparable is the player-facing loop
itself: for scenarios that touch none of the redesigned systems (idle, looking
around, input edge cases, quitting) the player state, clock, prompts and message
log must still match the prototype frame for frame. Rendering is no longer
compared (the town layout and HUD changed); see tests/regression for pixel goldens.

The prototype's crash on selling stolen goods to a merchant is also verified.
"""
from __future__ import annotations

import os
import shutil

import pytest

from tests.parity.display import display_available, run_scenario_subprocess
from tests.parity.scenarios import CRASHING_IN_PROTOTYPE, PROTOTYPE_IDENTICAL, SCENARIOS

pytestmark = [pytest.mark.parity, pytest.mark.display,
              pytest.mark.skipif(not display_available(), reason="needs a display or xvfb-run")]

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "textures")

# State the prototype and the refactor must still agree on in the identical scenarios.
PLAYER_VISIBLE_KEYS = ("running", "player", "hour", "minute_accumulator", "in_trade", "merchant", "trade_selection",
                       "in_lockpicking", "prompt", "messages", "moving", "mouse_visible", "mouse_grab")


def _workdir(tmp_path, label):
    d = tmp_path / label
    d.mkdir()
    for name in os.listdir(ASSET_DIR):
        shutil.copy(os.path.join(ASSET_DIR, name), d / name)   # the prototype loads textures from CWD
    return str(d)


PROTOTYPE_SKILLS = ("Sneak", "Lockpicking", "Pickpocket", "Smithing")     # the game added Stealing and Barter


def _visible(state: dict, origin=(0.0, 0.0, 0.0), starting_gold: float = 0.0) -> dict:
    """The player-facing state, with the position taken relative to ``origin`` (the spawn point:
    the town has been rebuilt around a different one, the player loop has not changed) and the
    purse relative to ``starting_gold`` (the game starts the player with more coin than the
    prototype did; what the loop does with it is what must match). Skills compare by level and
    experience: the cost of each level (``xp_needed``) follows the game's own 1..100 curve now."""
    out = {k: state[k] for k in PLAYER_VISIBLE_KEYS if k in state}
    pos = tuple(round(a - b, 6) for a, b in zip(state["player"]["pos"], origin))
    # (the eye height is a deliberate change: the prototype's camera stood a head taller than the townsfolk)
    out["player"] = {k: v for k, v in state["player"].items() if k not in ("inventory", "skills", "pos", "eye", "gold")} | \
        {"pos": pos, "gold": round(state["player"]["gold"] - starting_gold, 6),
         "inventory": [{"lockpick_1": "iron_lockpick", "stolen_goblet": "silver_goblet"}.get(i, i) for i in state["player"]["inventory"]],
         "skills": {k: {kk: vv for kk, vv in v.items() if kk != "xp_needed"}
                    for k, v in state["player"].get("skills", {}).items() if k in PROTOTYPE_SKILLS}}
    if not state["running"]:
        out.pop("mouse_visible", None); out.pop("mouse_grab", None)   # released differently after quitting
    return out


def _describe_diff(a: dict, b: dict) -> str:
    return "\n".join(f"  {k}: legacy={a.get(k)!r} new={b.get(k)!r}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k))


@pytest.mark.parametrize("name", PROTOTYPE_IDENTICAL)
def test_identical_scenarios_match_prototype_state(name, tmp_path):
    legacy = run_scenario_subprocess("legacy", name, _workdir(tmp_path, "legacy"))
    new = run_scenario_subprocess("new", name, _workdir(tmp_path, "new"))
    assert legacy.error is None, f"prototype crashed: {legacy.error}"
    assert new.error is None, f"refactor crashed: {new.error}"
    assert len(legacy.frames) == len(new.frames) >= 1

    state_mismatches = []
    legacy_spawn, new_spawn = legacy.frames[0].state["player"]["pos"], new.frames[0].state["player"]["pos"]
    legacy_gold, new_gold = legacy.frames[0].state["player"]["gold"], new.frames[0].state["player"]["gold"]
    for lf, nf in zip(legacy.frames, new.frames):
        ls, ns = _visible(lf.state, legacy_spawn, legacy_gold), _visible(nf.state, new_spawn, new_gold)
        if ls != ns:
            state_mismatches.append(f"frame {lf.index}:\n{_describe_diff(ls, ns)}")
    assert not state_mismatches, "STATE DIVERGED:\n" + "\n".join(state_mismatches[:5])
    assert legacy.frames[-1].index >= len(SCENARIOS[name]()) - 1


def test_escape_opens_pause_menu_instead_of_quitting(tmp_path):
    """Deliberate divergence: the prototype exits on ESC; the app opens a pause menu whose
    "Exit to Menu" entry asks first, returns to the start menu, and only its "Exit" quits.
    Both runs use the same script."""
    legacy = run_scenario_subprocess("legacy", "escape_quits", _workdir(tmp_path, "legacy"))
    new = run_scenario_subprocess("new", "escape_quits", _workdir(tmp_path, "new"))
    assert legacy.error is None and new.error is None
    assert legacy.frames[-1].index == 3 and legacy.frames[-1].state["running"] is False   # quit on the ESC frame
    by_index = {f.index: f.state for f in new.frames}
    assert by_index[3]["running"] is True and by_index[3]["screen"] == "PAUSE"
    assert by_index[3]["mouse_visible"] is True                                         # cursor released in menus
    assert by_index[7]["screen"] == "CONFIRM"                                           # "Exit to Menu?" asks first
    assert by_index[9]["screen"] == "START" and by_index[9]["running"] is True          # back at the start menu
    assert new.frames[-1].index == 11 and new.frames[-1].state["running"] is False       # Exit quits
    assert len(new.frames) == 12
    # the world did not advance while in the pause menu and its confirmation
    assert by_index[3]["hour"] == by_index[7]["hour"]
    assert by_index[3]["minute_accumulator"] == by_index[7]["minute_accumulator"]
    # the start menu it returns to has a fresh town behind it, living its afternoon (attract.TitleView)
    assert by_index[9]["hour"] == 14 and new.frames[-1].state["hour"] == 14


@pytest.mark.parametrize("name", CRASHING_IN_PROTOTYPE)
def test_prototype_crash_is_fixed(name, tmp_path):
    """The original crashed with NameError (``AIState`` not imported in main.py)
    when selling stolen goods to a non-fence merchant; the refactor carries on
    with the intended behaviour (merchant reports you, menu closes)."""
    legacy = run_scenario_subprocess("legacy", name, _workdir(tmp_path, "legacy"))
    assert legacy.error is not None and legacy.error.startswith("NameError"), legacy.error
    last = legacy.frames[-1].state
    assert last["in_trade"] and last["merchant"] == "Shopkeeper Barnaby"     # crashed on the sale itself

    # The refactor handles the same situation (headless, via the scripted scenario).
    from tests.parity.harness import dry_run
    frames = []
    game = dry_run(SCENARIOS["trade_merchant_refuses_stolen"](), frames)
    assert not game.in_trade_menu and game.mouse_captured
    from tests.conftest import said
    seen = {m for f in frames for m in f["messages"]}                  # (the log holds six lines: look at every frame)
    assert said(seen, "stolen_goods", item="Silver Goblet")
    # ... and his shout brought a guard who settled it: the goblet went back and, the bill being more than
    # the purse (30g fencing plus the stolen goblet's 120g), the cells
    assert any("Fencing stolen goods" in m for m in seen)
    assert game.world.jail.serving and not any(i.id == "silver_goblet" for i in game.player.inventory)
