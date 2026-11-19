import json

import pytest

from tests.helpers import spot
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState
from unprepared.world import SaveManager


def test_save_and_load_roundtrip(world, tmp_path):
    player = world.spawn_player()
    player.gold = 7.0
    player.position = Vector3D(1, 1, 1)
    player.rotation_y = 123.0
    player.inventory.pop()
    world.simulated_hour = 21
    world.clock.minute_accumulator = 12.5
    world.npcs[2].state = AIState.ALERT_GUARDS
    world.npcs[2].suspicion_meter = 100.0
    world.npcs[0].inventory.pop(0)                 # Barnaby lost his shop key
    world.doors[0].is_open = True
    world.doors[1].is_locked = False
    path = tmp_path / "save.json"
    mgr = SaveManager(str(path), notify_callback=world.log.add)
    mgr.save(world, player)
    assert world.log.messages[-1] == "Game state saved successfully!"

    world.containers[0].inventory[1].quantity = 1        # two bottles of wine gone
    player.inventory[0].quantity = 3                     # a stack of picks
    player.crimes_witnessed = 2
    world.npcs[2].crimes_witnessed = 2
    data = json.loads(path.read_text()) if False else None
    mgr.save(world, player)
    data = json.loads(path.read_text())
    assert set(data) == {"version", "meta", "simulated_hour", "minute_accumulator", "day_count", "elapsed", "player", "npcs",
                         "doors", "containers", "jail", "inn"}
    assert data["version"] == 9 and data["jail"] == {"sentence_end_day": None, "escaped": False, "grace_until": 0.0, "released": False}
    assert data["simulated_hour"] == 21 and data["minute_accumulator"] == 12.5
    assert [n["name"] for n in data["npcs"]][:3] == ["Shopkeeper Barnaby", "Hilde the Innkeeper", "Gareth the Smith"]
    assert [d["door_id"] for d in data["doors"]][:3] == ["smithy_door", "inn_door", "keeper_room_door"]
    assert [c["name"] for c in data["containers"]][:2] == ["shop_shelf_west", "shop_shelf_east"]

    # mutate and load back
    player.gold = 999
    world.simulated_hour = 3
    world.npcs[2].state = AIState.SLEEPING
    world.npcs[0].inventory.insert(0, world.npcs[0].inventory[0])
    world.doors[0].is_open = False
    world.doors[1].is_locked = True
    world.containers[0].inventory[1].quantity = 3
    player.inventory[0].quantity = 1
    assert mgr.load(world, player) is True
    assert player.gold == 7.0 and player.position == Vector3D(1, 1, 1) and len(player.inventory) == 2
    assert player.rotation_y == 123.0
    assert world.simulated_hour == 21 and world.clock.minute_accumulator == 12.5
    assert world.npcs[2].state is AIState.IDLE and world.npcs[2].suspicion_meter == 100.0   # alarms are not resumed from a save
    assert [i.id for i in world.npcs[0].inventory][0] == "home_key"
    assert world.doors[0].is_open and not world.doors[1].is_locked
    assert world.containers[0].inventory[1].quantity == 1 and player.inventory[0].quantity == 3
    assert player.crimes_witnessed == 2 and world.npcs[2].crimes_witnessed == 2
    assert world.log.messages[-1] == "Game state loaded successfully!"


def test_loaded_npcs_resume_their_routine_from_the_saved_spot(world, tmp_path):
    player = world.spawn_player()
    barnaby = world.find_npc("Shopkeeper Barnaby")
    world.simulated_hour = 10                       # shop hours, but save him standing in the square
    barnaby.position = spot(world, 0.0, 5.0)
    barnaby.state = AIState.PATROL
    mgr = SaveManager(str(tmp_path / "s.json"))
    mgr.save(world, player)
    barnaby.position = spot(world, 15.0, 15.0)
    assert mgr.load(world, player)
    assert barnaby.position == spot(world, 0.0, 5.0)
    assert barnaby.state is AIState.TRAVELING and barnaby.path and barnaby.path[-1] == barnaby.task_for(10).target_pos


def test_load_missing_file(world, tmp_path):
    mgr = SaveManager(str(tmp_path / "nope.json"), notify_callback=world.log.add)
    assert mgr.load(world, world.spawn_player()) is False
    assert world.log.messages[-1] == "No save file found!"


def test_loads_original_prototype_save_format(world, tmp_path):
    """A save written by the original prototype must still load."""
    legacy = {
        "simulated_hour": 9,
        "player": {"position": {"x": 1.0, "y": 1.0, "z": 2.0}, "gold": 10.0, "has_lit_torch": True,
                   "is_crouching": False, "inventory": [{"id": "a", "name": "A", "value": 1.0}],
                   "skills": {"Sneak": {"level": 3, "xp": 0.0, "xp_needed": 156.25}}},
        "npcs": [{"name": "Jae the Fence", "position": {"x": 0, "y": 0, "z": 0}, "suspicion_meter": 5.0, "state": 3}],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    player = world.spawn_player()
    assert SaveManager(str(path)).load(world, player)
    assert player.has_lit_torch and [i.id for i in player.inventory] == ["a"]
    assert player.skills.to_dict()["Sneak"]["level"] == 3
    jae = world.find_npc("Jae the Fence")
    # the town has been rebuilt since: the old coordinates point at nothing, so the player is put back
    # at the spawn and everybody at their post for the hour (what they felt is kept)
    assert player.position == world.spot(*world.level.player_spawn)
    assert jae.position == jae.task_for(9).target_pos and jae.suspicion_meter == 5.0
    assert jae.state is AIState.SHOPKEEPING
    assert [i.id for i in jae.inventory] == ["hideout_door_key", "lockpick_pack", "gold_ring", "black_hood", "dried_meat"]   # v1 saves had no NPC inventory
    assert world.find_door("shop_door").is_locked     # v1 saves had no door state: unchanged
    assert [(i.id, i.quantity) for i in world.containers[0].inventory][1] == ("wine", 3)   # no containers either


def test_loads_version_2_save_without_containers_or_stacks(world, tmp_path):
    v2 = {
        "version": 2, "simulated_hour": 9, "minute_accumulator": 3.0,
        "player": {"position": {"x": 1.0, "y": 1.0, "z": 2.0}, "rotation_y": 10.0, "rotation_x": 0.0, "gold": 10.0,
                   "has_lit_torch": False, "is_crouching": False,
                   "inventory": [{"id": "iron_lockpick", "name": "Iron Lockpick", "value": 5.0, "is_lockpick": True}],
                   "skills": {}},
        "npcs": [{"name": "Guard Captain Vance", "position": {"x": 0, "y": 0, "z": 0}, "suspicion_meter": 0.0, "state": 2,
                  "has_torch": False, "inventory": []}],
        "doors": [{"door_id": "shop_door", "is_locked": False, "is_open": True}],
    }
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(v2))
    player = world.spawn_player()
    assert SaveManager(str(path)).load(world, player)
    assert player.inventory[0].quantity == 1 and player.crimes_witnessed == 0
    assert world.find_npc("Guard Captain Vance").inventory == [] and world.find_npc("Guard Orrin").inventory   # untouched
    assert not world.find_door("shop_door").is_locked and world.find_door("guardhouse_door").is_locked


def test_door_state_is_persisted(world, tmp_path):
    player = world.spawn_player()
    door = world.find_door("hideout_door")
    door.is_open = True
    data = SaveManager.serialize(world, player)
    assert {"door_id": "hideout_door", "is_locked": False, "is_open": True, "sealed": False} in data["doors"]


def test_old_saves_put_the_player_back_on_their_feet(game, tmp_path):
    """Saves from before version 6 kept the player a unit off the ground."""
    import json
    assert game.save_game()
    path = game.saves.filename
    data = json.load(open(path))
    data["version"] = 5
    data["player"]["position"]["y"] = 1.0
    json.dump(data, open(path, "w"))
    spawn = game.world.spot(*game.world.level.player_spawn)
    assert game.load_game() and game.player.position == spawn          # (and, a pre-hillside save, back at the spawn)


def test_old_saves_get_item_owners_from_their_holders(game):
    """Saves from before version 7 had no owner on NPC and container items."""
    import json
    assert game.save_game()
    path = game.saves.filename
    data = json.load(open(path))
    data["version"] = 6
    for npc in data["npcs"]:
        for item in npc["inventory"]:
            item["owner"] = None
    for container in data["containers"]:
        for item in container["inventory"]:
            item["owner"] = None
    shelf = next(c for c in data["containers"] if c["name"] == "shop_shelf_west")
    shelf["inventory"][0]["is_stolen"] = True               # a stolen thing on a shelf stays of unknown origin
    json.dump(data, open(path, "w"))
    assert game.load_game()
    barnaby = game.world.find_npc("Shopkeeper Barnaby")
    assert all(i.owner == "Shopkeeper Barnaby" for i in barnaby.inventory)
    west = game.world.find_container("shop_shelf_west")
    assert west.inventory[0].owner is None and all(i.owner == "Shopkeeper Barnaby" for i in west.inventory[1:])
    table = game.world.find_container("guard_table")
    assert all(i.owner == "guards" for i in table.inventory)


# --- robustness: corrupt files, atomic writes, nothing leaks across a load ------------------------
def test_corrupt_save_is_reported_not_raised(game, tmp_path):
    game.saves.filename = str(tmp_path / "corrupt.json")
    (tmp_path / "corrupt.json").write_text('{"version": 7, "player": ')        # truncated by a crash mid-write
    gold = game.player.gold
    assert game.load_game() is False
    assert game.player.gold == gold and game.running
    assert any("corrupt" in m for m in game.messages.messages)
    assert game.saves.describe(game.saves.filename).endswith("(unreadable)")  # still listed, flagged
    (tmp_path / "corrupt.json").write_text('[1, 2, 3]')                         # valid JSON, not a save
    assert game.load_game() is False
    (tmp_path / "corrupt.json").write_text('{"version": 7}')                    # no player block
    assert game.load_game() is False


def test_a_save_that_breaks_half_way_through_loading_changes_nothing(game, tmp_path):
    """The player block applies cleanly, a later block is malformed: the load is refused and the
    game is put back exactly as it was - not left with the save's purse and position but the
    running game's doors."""
    w, p = game.world, game.player
    p.gold, p.position = 999.0, Vector3D(-30.0, 0.0, 0.0)
    path = tmp_path / "damaged.json"
    assert game.save_game(str(path))
    data = json.load(open(path))
    data["doors"] = [{"not_a_door_id": True}]                       # KeyError once the player is already loaded
    json.dump(data, open(path, "w"))
    p.gold, p.position = 5.0, Vector3D(2.0, 0.0, 3.0)
    w.find_door("hideout_door").is_open = True
    w.clock.hour = 15
    assert game.load_game(str(path)) is False
    assert p.gold == 5.0 and p.position == Vector3D(2.0, 0.0, 3.0)
    assert w.find_door("hideout_door").is_open and w.clock.hour == 15
    assert any("is damaged (KeyError)" in m and "nothing was loaded" in m for m in game.messages.messages)
    assert game.saves.last_loaded is None


@pytest.mark.parametrize("block, value", [("jail", None), ("inn", "full"), ("npcs", [["Jae"]]), ("containers", "x"),
                                          ("player", {"position": "here"})])
def test_a_block_of_the_wrong_shape_is_reported_not_raised(game, tmp_path, block, value):
    path = tmp_path / "odd.json"
    game.player.gold = 7.0
    assert game.save_game(str(path))
    data = json.load(open(path))
    data[block] = value
    json.dump(data, open(path, "w"))
    assert game.load_game(str(path)) is False
    assert game.player.gold == 7.0 and game.running
    assert any("is damaged" in m for m in game.messages.messages)


def test_saves_are_written_atomically(game, tmp_path, monkeypatch):
    """A save goes to a temp file and is renamed into place: a failure mid-write leaves the
    previous save untouched and no temp file behind."""
    path = tmp_path / "atomic.json"
    game.player.gold = 11.0
    game.save_game(str(path))
    good = path.read_text()
    game.player.gold = 22.0
    import json as _json

    def explode(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(_json, "dump", explode)
    assert game.save_game(str(path)) is False                       # reported, not raised
    assert any("Could not write the save file" in m for m in game.messages.messages)
    monkeypatch.undo()
    assert path.read_text() == good                                  # the old save survived
    assert not (tmp_path / "atomic.json.tmp").exists()
    assert _json.loads(good)["player"]["gold"] == 11.0


def test_saving_is_refused_while_being_taken_to_the_cells(game):
    """From the moment a sentence is passed until the player stands in the cell, no save can be
    made - a save in between would have the sentence but not the prisoner."""
    from unprepared.systems import justice
    assert game.can_save
    justice.imprison(game.world, game.player, 2, game.messages.add)
    assert game.world.jail.transfer_pending and not game.can_save
    assert game.save_game() is False and any("cannot save" in m for m in game.messages.messages)
    for _ in range(400):                                              # the delay, the fade to black, the move
        game.update(0.016)
        if not game.world.jail.transfer_pending:
            break
    assert game.world.area_of(game.player.position) == game.world.jail.cell_area
    assert not game.world.jail.transfer_pending and game.world.save_blocked is None
    game.player.freeze_timer = 0.0
    while game.in_transition:
        game.update(0.016)
    assert game.can_save and game.save_game()


def test_loading_during_a_fade_leaves_nothing_in_flight(game):
    """Quick-loading while an eviction is fading must not leave the old relocation pending: the
    justice system keys off ``World.relocating`` and would otherwise stay switched off for good."""
    w, p = game.world, game.player
    game.update(0.016)
    assert game.save_game()
    p.position = spot(game.world, -30.0, 10.0)
    w.throw_out(p, "Tamsin's cottage", by=w.find_npc("Tamsin"), line="out")
    assert game.in_transition and w.relocating
    game.update(0.1)                                                  # mid-fade, still not black
    assert game.load_game()
    assert game.fade is None and not w.relocating and w.relocations_in_flight == [] and w.wait_interruption is None
    # ... and a guard with a grudge can still charge the player afterwards
    vance = w.find_npc("Guard Captain Vance")
    vance.remember_crime("Pickpocketing", w.day_count, True, 25.0)
    vance.path, vance.state = [], AIState.PATROL
    p.position = spot(game.world, vance.position.x, vance.position.z + 1.5)
    vance.turn_towards(p.position)
    game.update(0.05)
    assert p.gold < 60.0 and vance.crime_memories == []               # fined on the spot


def test_a_load_drops_every_npcs_short_term_state(game):
    w = game.world
    game.update(0.016)
    assert game.save_game()
    jae, tamsin = w.find_npc("Jae the Fence"), w.find_npc("Tamsin")
    conv = w.start_conversation(jae, tamsin)
    jae.wait_for("curfew", game.player.position)
    jae.curfew_asks, jae.torch_warnings, jae.awake_for_business, jae.locked_out = 2, 2, True, True
    jae.last_scold["trespass"] = 5.0
    w.wait_interruption = "caught"
    assert w.conversations and jae.chat_pause and tamsin.conversation is conv
    assert game.load_game()
    assert w.conversations == [] and conv.finished
    for npc in w.npcs:
        assert npc.conversation is None and not npc.chat_pause and npc.speech is None
        assert npc.waiting_for is None and npc.curfew_asks == 0 and npc.torch_warnings == 0
        assert not npc.awake_for_business and not npc.locked_out and npc.last_scold == {}
        assert npc.pending_locks == [] and npc.goal is None and npc.report_to is None
    assert w.wait_interruption is None and jae.state is not AIState.WAITING


def test_a_flag_of_the_running_session_does_not_colour_the_state_loaded():
    """An NPC saved asleep loads asleep - even when, in the session being replaced, they had just
    been roused (a fence woken for a deal, an innkeeper for a welcome). The old session's
    ``awake_for_business`` used to make the loaded state look "busy" and be thrown away."""
    from unprepared.entities import NPC
    jae = NPC("Jae the Fence", False, True, True, Vector3D(-20.75, 0.0, 14.5))
    jae.state = AIState.SLEEPING
    data = jae.to_dict()
    jae.wake_for_business()
    assert jae.awake_for_business and jae.is_busy
    jae.load_dict(data)
    assert jae.state is AIState.SLEEPING and not jae.awake_for_business
    busy = NPC("Guard Orrin", True, False, False, Vector3D(0.0, 0.0, 0.0))
    busy.state = AIState.RESPONDING
    saved = busy.to_dict()
    busy.load_dict(saved)
    assert busy.state is AIState.IDLE                  # a saved interruption is still not resumed from a save
