"""Save games (systems.savegame): everything a run has changed survives a save and a load,
save files go in <home>/Unprepared/Saves named by the date and time, and a new save never
overwrites an old one."""

from __future__ import annotations

import json
import math
from datetime import datetime

import pytest

from cyberduke.core.userdata import saves_dir
from cyberduke.levels import build_night_market
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems import savegame
from cyberduke.systems.hacking import HackKind

DT = 1 / 30


@pytest.fixture(autouse=True)
def user_home(tmp_path, monkeypatch):
    """Every test gets its own <home>/Unprepared."""
    monkeypatch.setenv("CYBERDUKE_USER_DIR", str(tmp_path / "Unprepared"))
    return tmp_path / "Unprepared"


def run(sim, seconds, inp=None):
    for _ in range(int(seconds / DT)):
        sim.update(DT, inp or InputState())


def _without_time(data):
    data = json.loads(json.dumps(data))
    data.pop("saved_at", None)
    for group in ("civilians", "merchants"):
        for d in data.get(group, {}).values():
            d.pop("activity", None)
    return data


def test_saves_go_in_the_users_unprepared_saves_folder_and_never_overwrite(user_home):
    sim = Simulation()
    when = datetime(2077, 4, 2, 23, 15, 7)
    a = savegame.save_game(sim, when=when)
    b = savegame.save_game(sim, when=when)  # the same second: a new file, not the old one
    assert a.parent == saves_dir() == user_home / "Saves"
    assert a.name == "save_2077-04-02_23-15-07.json" and b.name == "save_2077-04-02_23-15-07_2.json"
    assert a.exists() and b.exists()
    later = savegame.save_game(sim, when=datetime(2077, 4, 3, 1, 0, 0))
    listed = savegame.list_saves()
    assert [s.path for s in listed] == [later, b, a], "newest first"
    assert listed[0].level_name == "Corporate Facility" and "IN THE FACILITY" in listed[0].detail
    (user_home / "Saves" / "save_broken.json").write_text("{ not json")
    assert len(savegame.list_saves()) == 3, "a broken file is not offered"


def test_a_facility_run_survives_a_save_and_a_load():
    """Guards down, blinded, cameras dark (their timers running), doors open, a computer
    breached and copied, the player crouching somewhere with stolen data and a mod in."""
    sim = Simulation()
    run(sim, 1.0)
    h = sim.hacking
    assert h.apply(sim.terminals[0], HackKind.OVERRIDE_DOORS)  # the blast door, and a breach
    assert h.apply(sim.cameras[0], HackKind.SHUTDOWN)
    assert h.apply(sim.guards[0], HackKind.ELECTROCUTE)
    sim.terminals[1].steal_intel()
    run(sim, 6.0)  # the shocked guard goes down
    assert h.apply(sim.guards[2], HackKind.BLIND)
    p = sim.player
    p.x, p.z, p.yaw, p.crouching = 1.2, 4.0, 37.0, True
    p.inventory.equip(p.inventory.find("mod_ping_amp"))
    sim.update(DT, InputState(scan=True))
    run(sim, 0.3)
    path = savegame.save_game(sim)
    before = savegame.snapshot(sim)

    other = Simulation()
    run(other, 2.0)
    savegame.load_game(other, path)
    after = savegame.snapshot(other)
    assert _without_time(after) == _without_time(before)
    g0 = other.guards[0]
    assert g0.is_down and g0.body is not None, "down again, lying where it fell"
    assert other.guards[2].is_blinded and other.guards[2].blind_time_left > 0.5
    assert not other.cameras[0].is_online and other.cameras[0].shutdown_timer > 150.0
    assert other.room.doors["blast_door"].unlocked
    assert other.hacking.is_breached("halcyon_dynamics") or any(
        n.breached for n in other.hacking.networks.values()
    )
    assert other.player.crouching and other.player.stat("ping_range") > 1.5
    assert other.player.inventory.has("data_shard", "greenhouse door cipher")
    run(other, 1.0)  # and it plays on
    assert other.state.value == "PLAYING"


def test_a_street_run_with_the_quest_half_done_survives_a_save_and_a_load():
    sim = Simulation(build_night_market())
    run(sim, 20.0)
    sim.quests.start("number16", sim.time)
    sim.panels[0].bypass()
    sim.director.update()
    sim.pickups[0].taken = True
    sim.player.inventory.add("datastick")
    sim.player.x, sim.player.z = -6.0, 30.5  # inside the flat
    path = savegame.save_game(sim)
    before = savegame.snapshot(sim)

    other = Simulation()
    savegame.load_game(other, path)
    assert other.level.name == "Night Market Alley"
    after = savegame.snapshot(other)
    assert _without_time(after) == _without_time(before)
    state = other.quests.get("number16")
    assert state is not None and "lock_released" in state.flags
    assert other.pickups[0].taken and other.player.inventory.has("datastick")
    assert other.room.doors["door16"].unlocked
    info = savegame.list_saves()[0]
    assert info.where == "INSIDE NUMBER 16" and info.quests == 1
    # the people of the street pick their stories up again from where they stand
    run(other, 10.0)
    assert other.state.value in ("PLAYING", "CAUGHT")


def test_loading_puts_the_knife_back_if_the_chef_had_it_in_hand():
    sim = Simulation(build_night_market())
    knife = sim.life.item("knife")
    sim.life.remove(knife)  # in the chef's hand right now
    path = savegame.save_game(sim)
    other = Simulation()
    savegame.load_game(other, path)
    assert other.life.item("knife") is not None


def test_a_save_from_a_newer_game_is_refused():
    sim = Simulation()
    data = savegame.snapshot(sim)
    data["version"] = savegame.SAVE_VERSION + 1
    with pytest.raises(savegame.SaveError, match="newer"):
        savegame.restore(sim, data)
    with pytest.raises(savegame.SaveError, match="not a Cyberduke save"):
        savegame.restore(sim, {"format": "something else"})


def test_the_save_keeps_what_the_load_menu_shows(user_home):
    sim = Simulation(build_night_market())
    sim.time = 125.0
    path = savegame.save_game(sim)
    data = json.loads(path.read_text())
    assert data["format"] == "cyberduke-save" and data["level"] == "night_market"
    info = savegame.list_saves()[0]
    assert info.play_time == pytest.approx(125.0) and "02:05 PLAYED" in info.detail
    assert math.isclose(savegame.distance_moved(data, data), 0.0)


def test_the_time_of_day_the_stalls_and_who_was_asleep_are_saved():
    """The city's clock, which stalls were shut for the night (and the CLOSED sign on the
    counter), and who was in bed: a load at four in the morning is four in the morning."""
    sim = Simulation(build_night_market())
    sim.clock.set(4 * 60.0)
    sim.clock.rate = 0.0
    for _ in range(int(150.0 / (1 / 30))):
        sim.player.x, sim.player.z = 0.0, -1.2
        sim.update(1 / 30, InputState())
        if sim.npc_with_role("junk_dealer").asleep and sim.life.item("sushi_closed"):
            break
    assert not sim.life.open["sushi"] and sim.life.item("sushi_closed") is not None
    path = savegame.save_game(sim)
    data = json.loads(path.read_text())
    assert data["clock"] == {"minutes": 240.0}, "the time of day (no count of days)"
    assert data["merchants"]["cyberware_cheap"]["asleep"]

    other = Simulation(build_night_market())
    savegame.load_game(other, path)
    assert other.clock.text == "04:00"
    assert other.life.open == {"sushi": False, "junk": False, "bar": False}
    assert other.life.item("sushi_closed") is not None, "the sign is still on the counter"
    info = savegame.list_saves()[0]
    assert "04:00" in info.detail and "DAY" not in info.detail
    other.clock.rate = 0.0
    dealer = other.npc_with_role("junk_dealer")
    for _ in range(int(60.0 / (1 / 30))):
        other.player.x, other.player.z = 0.0, -1.2
        other.update(1 / 30, InputState())
        if dealer.asleep:
            break
    assert dealer.asleep, "back to bed (his story picks up from where he stands)"


def test_an_older_save_that_counted_days_still_loads():
    """Saves from before the day counter was dropped carry ``"day"`` in their clock: it is
    ignored, the time of day is read."""
    sim = Simulation(build_night_market())
    data = savegame.snapshot(sim)
    data["clock"] = {"minutes": 615.0, "day": 7}
    other = Simulation(build_night_market())
    savegame.restore(other, data)
    assert other.clock.text == "10:15"


def test_a_broken_vending_machine_is_still_broken_after_a_load():
    """Shorted, sparks over, nobody has kicked it yet: saved and loaded, it is still out
    of order (and comes back by itself no sooner than it would have)."""
    sim = Simulation(build_night_market())
    machine = sim.vending_machines[0]
    machine.trigger_distraction()
    for _ in range(int(25.0 / (1 / 30))):
        sim.player.x, sim.player.z = 0.0, -1.2
        sim.update(1 / 30, InputState())
    assert machine.broken and not machine.is_sparking
    other = Simulation(build_night_market())
    savegame.restore(other, savegame.snapshot(sim))
    again = other.vending_machines[0]
    assert again.broken and not again.is_sparking
    assert again.broken_timer == pytest.approx(machine.broken_timer, abs=1e-3)
