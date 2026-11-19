"""The first quest, "Something Special": Nix the smoker wants the data stick from number
16 - a shabby little flat behind the alley's west wall, with an access panel by its door,
a camera inside and an owner who does not like strangers in it. And what being caught
costs now: in the facility a new game; in somebody's home, being thrown out of it."""

from __future__ import annotations

import math

import pytest

from cyberduke.ai.alley import Away
from cyberduke.config import SIMULATION
from cyberduke.core.dialogue import Conversation, Line
from cyberduke.core.state import GameState
from cyberduke.levels import build_night_market
from cyberduke.levels.apartment import DATASTICK_AT, FLAT16
from cyberduke.simulation import InputState, Simulation

DT = 1 / 30
LEVEL = build_night_market()


def run(sim, seconds, **kw):
    for _ in range(int(seconds / DT)):
        sim.update(DT, InputState(**kw))


def street() -> Simulation:
    return Simulation(build_night_market())


def dell_goes_out(sim) -> None:
    """Dell off up the main road for good (an empty flat to test in)."""
    dell = sim.npc_with_role("resident")

    def away(me):
        while True:
            yield Away(1e9)

    dell.storyteller.story = away(dell)
    dell.storyteller.act = None


def walk_to(sim, x, z, seconds=8.0, **kw) -> bool:
    p = sim.player
    for _ in range(int(seconds / DT)):
        if math.hypot(x - p.x, z - p.z) < 0.3:
            return True
        p.yaw = math.degrees(math.atan2(z - p.z, x - p.x))
        p.pitch = 0.0
        sim.update(DT, InputState(forward=1.0, **kw))
    return False


def talk(sim) -> list[str]:
    """[E] at whoever is in front, then [E] through every line. Returns the lines."""
    sim.update(DT, InputState(interact_pressed=True, interact=True))
    lines = []
    while sim.conversation is not None:
        lines.append(sim.conversation.line.text)
        sim.update(DT, InputState(interact_pressed=True))
        run(sim, 0.05)
    return lines


def face_nix(sim) -> None:
    p = sim.player
    p.x, p.z, p.yaw = 2.3, 1.7, 0.0  # camera yaw 0 = +x: Nix leans on the wall at x 3.85


# ----------------------------------------------------------------------------- dialogue
def test_captions_type_out_skip_on_e_and_move_on_by_themselves():
    done = []
    conv = Conversation(
        [Line("NIX", "Hey."), Line("NIX", "Number 16.")], on_done=lambda: done.append(1)
    )
    conv.update(0.03)
    assert conv.shown_text == "H" and not conv.typed
    conv.advance()  # [E]: the rest of the line at once
    assert conv.shown_text == "Hey." and conv.index == 0
    conv.advance()  # [E] again: the next line
    assert conv.index == 1 and conv.shown_text == ""
    conv.update(conv.line.duration + 0.01)  # left alone it moves on
    assert conv.finished and done == [1]


# ----------------------------------------------------------------------------- the quest
def test_nix_gives_the_quest_and_it_shows_what_to_do_next():
    sim = street()
    face_nix(sim)
    run(sim, 0.2)
    assert sim.prompt == "[E] TALK TO NIX"
    lines = talk(sim)
    assert any("something special" in line for line in lines)
    assert any("data stick" in line for line in lines)
    state = sim.quests.get("number16")
    assert state is not None and state.status == "active"
    assert any("NEW QUEST" in t for t, _ in sim.toasts)
    assert "access panel" in sim.director.objective(state)
    assert sim.prompt == "", "a last [E] does not start him over"
    run(sim, 1.0)
    assert talk(sim) == ["Number 16. The stick. The panel by the door. Go."]
    assert len(sim.quests.quests) == 1
    quest = state.quest
    assert [p.picture_id for p in quest.pictures] == ["door16", "datastick"]
    assert quest.description and quest.reward == "mod_wraith"


def test_nix_turns_to_face_the_netrunner_and_waits_while_they_talk():
    sim = street()
    face_nix(sim)
    run(sim, 0.2)
    nix = sim.npc_with_role("smoker")
    sim.update(DT, InputState(interact_pressed=True, interact=True))
    where = (nix.x, nix.z)
    run(sim, 1.5)
    assert nix.talking_to is sim.player and (nix.x, nix.z) == where
    bearing = math.degrees(math.atan2(sim.player.x - nix.x, sim.player.z - nix.z))
    assert abs((nix.yaw - bearing + 180) % 360 - 180) < 5.0, "facing the netrunner"
    sim.player.x = 8.0  # walks off mid-conversation: it ends
    run(sim, 0.2)
    assert sim.conversation is None and nix.talking_to is None


def test_the_access_panel_is_scanned_and_hacked_to_release_the_lock():
    sim = street()
    door = sim.room.doors["door16"]
    panel = sim.panels[0]
    assert panel.door is door and not door.unlocked
    assert door not in sim.hack_targets and panel in sim.hack_targets, "the lock is the panel"
    p = sim.player
    p.x, p.z = -2.0, 29.4
    p.yaw = math.degrees(math.atan2(panel.z - p.z, panel.x - p.x))
    p.pitch = math.degrees(math.atan2(panel.y - 1.6, math.hypot(panel.x - p.x, panel.z - p.z)))
    sim.update(DT, InputState(scan=True))
    run(sim, 0.2)
    assert sim.hacking.focus is panel
    assert [o.spec.label for o in sim.hacking.focus_options()] == ["BYPASS LOCK"]
    sim.update(DT, InputState(hack_slot=1))
    run(sim, 2.8)
    assert door.unlocked and panel.lock_released
    assert sim.hacking.options(panel)[0].reason == "LOCK ALREADY RELEASED"


def test_the_door_is_locked_until_bypassed_opens_as_you_walk_up_and_from_inside_always():
    sim = street()
    dell_goes_out(sim)
    door = sim.room.doors["door16"]
    p = sim.player
    p.x, p.z = -2.5, 30.5
    walk_to(sim, -6.0, 30.5, seconds=3.0)
    assert p.x > -4.2, "a locked door stops you on the step"
    assert p.floor_y == pytest.approx(0.16), "up on the step"
    sim.panels[0].bypass()
    assert walk_to(sim, -6.0, 30.5)
    assert FLAT16.contains(p.x, p.z) and p.floor_y == 0.0
    run(sim, 3.0)
    assert door.open_amount == 0.0, "it swings shut behind you"
    door.lock()  # Dell comes back and locks up: you can still get out (the latch)
    assert walk_to(sim, -2.5, 30.5)


def test_every_room_of_the_flat_can_be_walked_into():
    sim = street()
    dell_goes_out(sim)
    sim.cameras[0].shut_down(999)
    sim.panels[0].bypass()
    p = sim.player
    p.x, p.z = -2.5, 30.5
    for x, z in ((-6.0, 30.5), (-6.4, 31.6), (-6.4, 33.6), (-8.6, 34.0), (-6.4, 31.6),
                 (-7.7, 30.85), (-9.0, 30.0), (-7.7, 30.85), (-7.5, 29.4), (-7.6, 27.95),
                 (-9.2, 27.9)):  # fmt: skip
        assert walk_to(sim, x, z), f"could not get to ({x}, {z}) from ({p.x:.2f}, {p.z:.2f})"


def test_dells_computer_says_where_the_stick_is_and_the_stick_can_be_taken():
    sim = street()
    dell_goes_out(sim)
    sim.cameras[0].shut_down(999)
    sim.quests.start("number16")
    sim.panels[0].bypass()
    p = sim.player
    p.x, p.z = -5.85, 31.9  # at the desk
    run(sim, 2.5, interact=True)
    assert sim.hacking.session is sim.terminals[0]
    assert "bedroom" in sim.terminals[0].notes
    state = sim.quests.get("number16")
    assert "read_note" in state.flags and "bedroom" in sim.director.objective(state)
    sim.update(DT, InputState(disconnect=True))
    p.x, p.z = -8.9, 30.7
    stick = sim.pickups[0]
    p.yaw = math.degrees(math.atan2(stick.z - p.z, stick.x - p.x))
    p.pitch = -30.0
    run(sim, 0.2)
    assert sim.prompt == "[E] TAKE DATA STICK (NO.16)"
    sim.update(DT, InputState(interact_pressed=True, interact=True))
    assert stick.taken and p.inventory.has("datastick")
    assert "Nix" in sim.director.objective(state)
    assert (stick.x, stick.y, stick.z) == DATASTICK_AT


def test_bringing_nix_the_stick_gets_something_special():
    sim = street()
    sim.quests.start("number16")
    sim.player.inventory.add("datastick")
    nix = sim.npc_with_role("smoker")
    assert nix.inventory.has("mod_wraith")
    face_nix(sim)
    run(sim, 0.2)
    lines = talk(sim)
    assert any("Militech" in line for line in lines)
    state = sim.quests.get("number16")
    assert state.done and not sim.player.inventory.has("datastick")
    assert sim.player.inventory.has("mod_wraith") and not nix.inventory.has("mod_wraith")
    assert "Done" in sim.director.objective(state)
    run(sim, 1.0)
    assert talk(sim) == ["We're square. Now get lost before somebody sees us together."]


# ----------------------------------------------------------------------------- who minds
def test_the_home_camera_and_dell_only_mind_the_netrunner_inside_the_flat():
    sim = street()
    dell = sim.npc_with_role("resident")
    cam = sim.cameras[0]
    assert cam.watch_area is FLAT16 and dell.watch_area is FLAT16
    # out in the alley, right in front of the open door with Dell looking out: nothing
    sim.player.x, sim.player.z = -2.8, 30.5
    dell.x, dell.z, dell.yaw = -5.2, 30.5, 90.0
    sim.room.doors["door16"].open_amount = 1.0
    run(sim, 1.0)
    assert sim.detection.level == 0.0


def test_dell_only_notices_an_intruder_once_the_meter_turns_red():
    """Seeing a stranger in his flat, Dell's eyes fill the detection meter from the first
    moment - but he only *notices* them (drops what he is doing, turns on them, shouts)
    once it has gone red; while it is cyan or yellow he carries on with his chore."""
    from cyberduke.ai.alley import Wait
    from cyberduke.animation import street as clips
    from cyberduke.config import DETECTION

    sim = street()
    run(sim, 15.0)  # Dell home, about the flat
    dell = sim.npc_with_role("resident")
    assert FLAT16.contains(dell.x, dell.z)
    sim.cameras[0].shut_down(999)  # (only Dell watching)
    dell.x, dell.z, dell.yaw = -6.6, 32.0, 180.0  # in the living room, facing the sofa

    def at_the_desk(me):  # (a chore that keeps him where he is, facing the way he is)
        while True:
            yield Wait(lambda: False, lambda t: clips.idle(t, me.seed), label="tidying up")

    dell.storyteller.story = at_the_desk(dell)
    dell.storyteller.act = None
    p = sim.player
    p.x, p.z = -5.7, 30.6  # in front of him and off to one side, in plain view
    first_bark = None
    for _ in range(int(3.0 / DT)):
        sim.update(DT, InputState())
        p.x, p.z = -5.7, 30.6
        if any(b.line.speaker == "DELL" for b in sim.barks):
            first_bark = sim.detection.level
            break
        assert dell.alarm == 0.0 and dell.activity == "tidying up", (
            f"noticed at {sim.detection.level:.0f}: still cyan or yellow"
        )
        assert dell.yaw == 180.0, "he has not turned on the netrunner"
    assert first_bark is not None, "he does notice in the end"
    assert first_bark > DETECTION.red_level, "...once the meter has gone red"
    assert dell.alarm > 0.0
    run(sim, 0.6)
    assert abs(dell.yaw - 180.0) > 15.0, "turning on the stranger"


def test_half_aware_dell_forgets_it_if_the_intruder_slips_out_of_sight_in_time():
    """Seen only for a moment (the meter yellow, not red), and then out of his sight: he
    never noticed - no shout, no alarm - and the meter drains again."""
    from cyberduke.ai.alley import Wait
    from cyberduke.animation import street as clips

    sim = street()
    run(sim, 15.0)
    dell = sim.npc_with_role("resident")
    sim.cameras[0].shut_down(999)
    dell.x, dell.z, dell.yaw = -6.6, 32.0, 180.0

    def stays(me):
        while True:
            yield Wait(lambda: False, lambda t: clips.idle(t, me.seed), label="tidying up")

    dell.storyteller.story = stays(dell)
    dell.storyteller.act = None
    p = sim.player
    for _ in range(int(0.4 / DT)):  # in view for a moment
        p.x, p.z = -6.5, 30.5
        sim.update(DT, InputState())
    assert 0.0 < sim.detection.level < 50.0
    for _ in range(int(3.0 / DT)):  # behind him, out of his view
        p.x, p.z = -6.5, 33.3
        sim.update(DT, InputState())
    assert sim.detection.level == 0.0 and dell.alarm == 0.0
    assert not any(b.line.speaker == "DELL" for b in sim.barks)


# ----------------------------------------------------------------------------- caught
def test_caught_in_the_flat_you_are_thrown_out_and_lose_the_stick():
    sim = street()
    dell_goes_out(sim)
    sim.quests.start("number16")
    sim.panels[0].bypass()
    stick = sim.pickups[0]
    stick.taken = True
    sim.player.inventory.add("datastick")
    cam = sim.cameras[0]
    p = sim.player
    p.x, p.z = -6.0, 30.4  # in the middle of the living room, standing, in the sweep
    assert cam.watch_area is FLAT16
    for _ in range(int(8.0 / DT)):
        sim.update(DT, InputState())
        if sim.state is GameState.CAUGHT:
            break
    assert sim.state is GameState.CAUGHT and sim.caught_area is FLAT16
    assert "HOME CAMERA 16" in sim.caught_by
    run(sim, SIMULATION.evict_hold + SIMULATION.caught_fade + 0.1)
    assert sim.state is GameState.PLAYING and sim.fade_in > 0.0
    assert (p.x, p.z) == FLAT16.exit[:2], "out on the alley"
    assert not p.inventory.has("datastick") and not stick.taken, (
        "the stick is back on the nightstand"
    )
    door = sim.room.doors["door16"]
    assert not door.unlocked and door.open_amount == 0.0, "and the door locked behind you"
    assert sim.level.name == "Night Market Alley", "not a new game"
    assert sim.detection.level == 0.0
    assert any("THROWN OUT OF NUMBER 16" in t for t, _ in sim.toasts)
    state = sim.quests.get("number16")
    assert "took the stick back" in state.journal[-1][1]
    assert "nightstand" in sim.director.objective(state) or "Find" in sim.director.objective(state)


def test_caught_in_the_facility_is_a_red_screen_a_fade_and_a_new_game():
    from cyberduke.ai.guard import Inspecting

    sim = Simulation()
    sim.player.inventory.equip(sim.player.inventory.find("mod_ping_amp"))
    sim.quests.start("number16")  # (as if - a new game forgets everything)
    g = sim.guards[0]
    g.x, g.z, g.yaw = 0.0, 2.0, 0.0
    g.fsm.transition_to(Inspecting(source=(0.0, 6.0), duration=1e9))
    sim.player.x = 1.0
    run(sim, 3.0)
    assert sim.state is GameState.CAUGHT and sim.caught_rule == "restart"
    run(sim, SIMULATION.caught_hold - 0.5)
    assert sim.state is GameState.CAUGHT, "the red screen stays up a few seconds"
    run(sim, SIMULATION.caught_fade + 0.6)
    assert sim.state is GameState.PLAYING and sim.fade_in > 0.0
    assert sim.level.name == "Corporate Facility"
    assert (sim.player.x, sim.player.z) == sim.level.spawn[:2]
    assert not sim.player.inventory.find("mod_ping_amp").equipped, (
        "a new game: fresh out of the box"
    )
    assert not sim.quests.quests


def test_a_camera_looks_down_over_low_furniture_but_not_through_a_tall_crate():
    """Cameras hang under the ceiling: their line of sight is a 3D ray to the chest, so a
    coffee table between the lens and a standing intruder hides nothing."""
    from cyberduke.core.geometry import AABB
    from cyberduke.entities.camera import SecurityCamera
    from cyberduke.entities.player import Player

    cam = SecurityCamera(0.0, 2.5, 0.0, mount_yaw=0.0)
    p = Player((0.0, 4.0, 0.0))
    table = AABB(-0.5, 0.5, 2.0, 2.6, 0.0, 0.42)
    assert cam.can_see(p, [table])
    crate = AABB(-0.5, 0.5, 3.3, 3.7, 0.0, 1.6)
    p.crouching = True
    assert not cam.can_see(p, [crate])
