"""Executes the NPC example code from README.md against the real API so the docs can't drift."""

import pathlib
import re

import pytest

from cyberduke.simulation import InputState, Simulation

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _code_block(marker: str) -> str:
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    for block in blocks:
        if marker in block:
            return block
    pytest.fail(f"no README python block containing {marker!r}")


def test_pose_clip_example_runs():
    ns: dict = {}
    exec(_code_block("def typing_on_datapad_animation"), ns)
    pose = ns["typing_on_datapad_animation"](1.0)
    assert pose.has_prop and pose.head_pitch == 28.0


def test_courier_example_navigates_in_the_real_simulation():
    ns: dict = {}
    exec(_code_block("class CorporateCourier"), ns)
    sim = Simulation()
    courier = ns["CorporateCourier"](-3.0, 0.0, 4.0, sim.room.nav_bounds)
    for _ in range(600):
        sim.update(1 / 60, InputState())
        courier.update(sim.player, 1 / 60, sim.obstacles, sim.detection)
    assert courier.state in {"READING", "PATROLLING_B"}, courier.state
    assert courier.spec.prop == "datapad"


def test_busker_story_example_plays_out_on_the_street(monkeypatch):
    from dataclasses import replace

    from cyberduke.ai import alley
    from cyberduke.animation import street as clips
    from cyberduke.levels import build_night_market
    from cyberduke.world.level import Place

    ns: dict = {"Walk": alley.Walk, "Turn": alley.Turn, "Play": alley.Play, "clips": clips}
    exec(_code_block("def busker"), ns)
    monkeypatch.setitem(alley.STORIES, "busker", ns["busker"])
    monkeypatch.setitem(alley.NEEDS, "busker", ("busk",))
    base = build_night_market()
    level = replace(
        base,
        places=(*base.places, Place("busk", 1.5, 8.0, yaw=-90.0)),
        civilians=(replace(base.civilians[0], x=1.5, z=11.0, role="busker"),),
    )
    sim = Simulation(level)
    seen = set()
    for _ in range(int(30 / (1 / 30))):
        sim.update(1 / 30, InputState())
        seen.add(sim.civilians[0].activity)
    assert {"finding a good spot", "singing for coins", "chatting up the crowd"} <= seen
