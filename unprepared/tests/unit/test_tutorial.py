"""The tutorial: an urchin in Jae's debt is led through the trade - and kept to the job."""
import json

import pytest

from tests.helpers import spot
from tests.conftest import said
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState, Item
from unprepared.game import Game, GameMode
from unprepared.systems import NPCAction, Side, TransferOutcome, use_station
from unprepared.systems.targeting import InteractionTarget, TargetKind
from unprepared.tutorial import (DONE_MESSAGE, FENCE, KINDNESS_GOLD, LOCKED_STAGES, JAE_NOT_YET, REFUSALS, START_HOUR,
                                 TARGET_ITEM, WATCHMAN, WATCHMAN_LINE, Stage, Tutorial)
from unprepared.world.levels.town import STARTING_GOLD


@pytest.fixture
def tgame(settings, tmp_path):
    return Game(settings=settings, save_file=str(tmp_path / "savegame.json"), tutorial=True)


def settle(game, seconds, dt=0.1):
    for _ in range(int(seconds / dt)):
        game.update(dt)


def target(game, kind, obj, action=None):
    game.target = InteractionTarget(kind, obj, "", npc_action=action)
    game.interact()


def pick_the_shop_door(game):
    """Crouch in front of the store's door, pick it (a sure hand), stand up."""
    w, p = game.world, game.player
    p.position = spot(game.world, -13.5, -4.0)
    p.rotation_y, p.is_crouching = 0.0, True
    game.update(0.05)
    target(game, TargetKind.DOOR, w.find_door("shop_door"))
    assert game.in_lockpicking
    game.lockpicking.pick_angle, game.lockpicking.wobble_amplitude = game.lockpicking.sweet_spot, 0.0
    game.lockpicking.set_chance = 1.0                       # (the pins set: this is the lesson, not the roll)
    assert game.lockpick_attempt().value == "success"
    p.is_crouching = False
    game.update(0.05)


def take_the_candlestick(game):
    w, p = game.world, game.player
    p.position = spot(game.world, -16.6, -13.0)
    game.update(0.05)
    shelf = w.find_container("shop_shelf_west")
    target(game, TargetKind.CONTAINER, shelf)
    assert game.mode is GameMode.CONTAINER
    rows = [i.id for i in game.transfer.rows(Side.LEFT)]
    game.transfer.selection[Side.LEFT] = rows.index(TARGET_ITEM)
    result = game.transfer_confirm()
    game.close_transfer()
    return result


def until(game, predicate, seconds=60.0, dt=0.05):
    for _ in range(int(seconds / dt)):
        game.update(dt)
        if predicate():
            return True
    return False


def justice_bill(world, player) -> float:
    """What a guard would charge the player right now (crimes only, stolen goods aside)."""
    from unprepared.systems import justice
    return justice.outstanding_fine(world)


def serve_the_night(game):
    while game.in_transition or game.frozen:
        game.update(0.05)
    assert game.wait() and game.confirmation is not None
    game.confirm_yes()
    game.update(0.1)


# --- the start -----------------------------------------------------------------------------------
class TestTheStart:
    def test_a_new_game_begins_in_the_hideout_at_night_with_nothing(self, tgame):
        w, p, t = tgame.world, tgame.player, tgame.tutorial
        assert t.stage is Stage.INTRO and tgame.in_tutorial and t.locked
        assert w.clock.hour == START_HOUR and w.area_of(p.position) == "the hideout"
        assert p.gold == 0.0
        assert sorted((i.id, i.quantity, i.equipped) for i in p.inventory) == [("iron_lockpick", 3, False), ("soft_shoes", 1, False)]
        assert not any(i.is_stolen for i in p.inventory)                      # no goblet: an urchin owns nothing hot
        assert w.suppress_curfews                                             # Jae has taken you in
        assert tgame.tutorial_hint.startswith("Jae took you in") and "TUTORIAL" not in tgame.tutorial_hint

    def test_jae_explains_the_debt_and_then_you_may_go(self, tgame):
        w, t = tgame.world, tgame.tutorial
        jae = w.find_npc(FENCE)
        settle(tgame, 2.0)
        assert jae.speech is not None and "gutter" in jae.speech.text
        assert t.stage is Stage.INTRO
        settle(tgame, 12.0)
        assert t.stage is Stage.TORCH                                         # the three lines said: the torch law first
        assert len([m for m in tgame.messages.messages if m.startswith("(Jae")]) == 3
        assert "[1]" in tgame.tutorial_hint
        tgame.toggle_torch()
        tgame.update(0.1)
        assert tgame.player.has_lit_torch and t.stage is Stage.TO_THE_SHOP

    def test_leaving_the_hideout_early_moves_the_lesson_on(self, tgame):
        tgame.player.position = spot(tgame.world, -12.0, 3.0)
        tgame.update(0.1)
        assert tgame.tutorial.stage is Stage.TORCH

    def test_by_day_the_torch_law_is_explained_not_enforced(self, tgame):
        w, t = tgame.world, tgame.tutorial
        w.clock.hour = 12
        t._enter(Stage.TORCH)
        settle(tgame, 3.0)
        assert t.stage is Stage.TORCH
        settle(tgame, 6.0)
        assert t.stage is Stage.TO_THE_SHOP and not tgame.player.has_lit_torch

    def test_at_night_the_torch_stage_waits_for_the_torch(self, tgame):
        t = tgame.tutorial
        t._enter(Stage.TORCH)
        settle(tgame, 12.0)
        assert t.stage is Stage.TORCH                                         # 22:00: no torch, no moving on
        tgame.toggle_torch()
        tgame.update(0.1)
        assert t.stage is Stage.TO_THE_SHOP

    def test_jae_does_not_throw_you_out_at_curfew_tonight(self, tgame):
        w = tgame.world
        jae = w.find_npc(FENCE)
        settle(tgame, 30.0)                                                   # 22:0x in the hideout, well past the curfew hour
        assert jae.state is not AIState.WAITING and not said(tgame.messages.messages, "leave_late")

    def test_a_game_without_the_tutorial_is_the_usual_town(self, settings, tmp_path):
        game = Game(settings=settings, save_file=str(tmp_path / "s.json"))
        assert game.tutorial is None and not game.in_tutorial and game.tutorial_hint == ""
        assert game.world.clock.hour == 7 and game.player.gold == STARTING_GOLD and not game.world.suppress_curfews
        assert any(i.id == "silver_goblet" for i in game.player.inventory)


# --- kept to the job ------------------------------------------------------------------------------
class TestTheLeash:
    def test_only_the_stores_door_may_be_picked(self, tgame):
        w, p = tgame.world, tgame.player
        p.position, p.is_crouching = spot(tgame.world, -32.0, -4.5), True
        tgame.update(0.1)
        inn = w.find_door("inn_door")
        target(tgame, TargetKind.DOOR, inn)
        assert not inn.is_open and tgame.messages.messages[-1] == REFUSALS["door"]
        target(tgame, TargetKind.DOOR, w.find_door("tamsin_door"))
        assert not tgame.in_lockpicking
        tgame.toggle_torch()
        tgame.update(0.1)
        pick_the_shop_door(tgame)
        assert not w.find_door("shop_door").is_locked and tgame.tutorial.stage is Stage.IN_THE_SHOP

    def test_the_hideout_door_and_the_jail_are_allowed(self, tgame):
        w = tgame.world
        for door_id in ("hideout_door", "jail_door", "cell_door", "shop_door"):
            assert tgame.tutorial.refuses("door", w.find_door(door_id)) is None
        assert tgame.tutorial.refuses("door", w.find_door("smithy_door")) == REFUSALS["door"]

    def test_only_the_candlestick_leaves_the_shelf(self, tgame):
        w, p = tgame.world, tgame.player
        w.find_door("shop_door").is_locked = False
        tgame.update(0.1)
        p.position = spot(tgame.world, -16.6, -13.0)
        tgame.update(0.05)
        target(tgame, TargetKind.CONTAINER, w.find_container("shop_shelf_east"))
        assert tgame.mode is GameMode.EXPLORE and tgame.messages.messages[-1] == REFUSALS["container"]
        shelf = w.find_container("shop_shelf_west")
        target(tgame, TargetKind.CONTAINER, shelf)
        rows = [i.id for i in tgame.transfer.rows(Side.LEFT)]
        tgame.transfer.selection[Side.LEFT] = rows.index("wine")
        assert tgame.transfer_confirm().outcome is TransferOutcome.REFUSED and tgame.messages.messages[-1] == REFUSALS["item"]
        assert tgame.loot_all() is None and tgame.messages.messages[-1] == REFUSALS["loot_all"]
        assert sum(i.quantity for i in shelf.inventory) == 9                   # nothing moved

    def test_no_trading_no_pickpocketing_no_stations_before_the_job_is_done(self, tgame):
        w, p = tgame.world, tgame.player
        barnaby, jae = w.find_npc("Shopkeeper Barnaby"), w.find_npc(FENCE)
        target(tgame, TargetKind.NPC, barnaby, NPCAction.TRADE)
        assert tgame.mode is GameMode.EXPLORE and tgame.messages.messages[-1] == REFUSALS["trade"]
        target(tgame, TargetKind.NPC, jae, NPCAction.TRADE)
        assert tgame.mode is GameMode.EXPLORE and tgame.messages.messages[-1] == JAE_NOT_YET
        p.is_crouching = True
        target(tgame, TargetKind.NPC, jae, NPCAction.PICKPOCKET)
        assert tgame.mode is GameMode.EXPLORE and tgame.messages.messages[-1] == REFUSALS["pickpocket"]
        target(tgame, TargetKind.STATION, w.find_station("ore_deposit"))
        assert tgame.messages.messages[-1] == REFUSALS["station"]
        assert tgame.tutorial.refuses("talk", barnaby) is None                 # talking is fine

    def test_out_of_picks_jae_slips_you_another(self, tgame):
        p = tgame.player
        p.position = spot(tgame.world, -12.0, 3.0)
        tgame.update(0.1)
        p.inventory = [i for i in p.inventory if not i.is_lockpick]
        tgame.update(0.1)
        assert p.has_lockpick() and tgame.tutorial.picks_given == 1
        assert any("another lockpick" in m for m in tgame.messages.messages)


# --- the catch, the cells, and out again ------------------------------------------------------------
class TestCaughtAndPunished:
    def _to_the_cell(self, tgame):
        w, t = tgame.world, tgame.tutorial
        settle(tgame, 1.0)
        tgame.toggle_torch()
        pick_the_shop_door(tgame)
        assert take_the_candlestick(tgame).outcome is TransferOutcome.MOVED
        tgame.update(0.05)
        return w, t

    def test_the_theft_is_seen_charged_and_punished_with_a_single_night(self, tgame):
        npcs_before = len(tgame.world.npcs)
        w, t = self._to_the_cell(tgame)
        assert t.stage is Stage.CAUGHT and t.caught_guard == WATCHMAN
        guard = w.find_npc(WATCHMAN)                                          # a watchman who was not there a moment ago
        assert guard is not None and guard.is_guard and guard.is_alarmed and len(w.npcs) == npcs_before + 1
        assert w.building_at(guard.position).name == "shop"
        assert guard.position.distance_to(tgame.player.position) <= w.settings.justice.charge_reach
        assert guard.speech is not None and guard.speech.text == WATCHMAN_LINE
        assert any(m.startswith(f"({WATCHMAN} witnessed Shoplifting!)") and WATCHMAN_LINE in m for m in tgame.messages.messages)
        assert any("witnessed Shoplifting" in m for m in tgame.messages.messages)
        assert until(tgame, lambda: w.jail.serving, seconds=20.0)
        assert t.stage is Stage.JAIL and t.sentence_trimmed
        assert w.find_npc(WATCHMAN) is None and len(w.npcs) == npcs_before    # ... and is gone again
        assert t.caught_guard is None
        assert not any(n.crime_memories for n in w.npcs if n.is_guard)        # nothing left to charge twice
        assert w.jail.sentence_end_day == w.day_count + 1                     # the captain's pity: one night
        assert any("first offence" in m for m in tgame.messages.messages)
        assert any("Returned to their owners: Silver Candlestick" in m for m in tgame.messages.messages)
        shelf = w.find_container("shop_shelf_west")
        assert any(i.id == TARGET_ITEM for i in shelf.inventory)              # back on the shelf for the second try
        while tgame.in_transition or tgame.frozen:
            tgame.update(0.05)
        assert w.area_of(tgame.player.position) == w.jail.cell_area
        assert [i.id for i in tgame.player.inventory] == ["iron_lockpick"]    # everything but one pick confiscated

    def test_the_watchman_stays_at_your_elbow(self, tgame):
        w, t = self._to_the_cell(tgame)
        guard = w.find_npc(WATCHMAN)
        tgame.player.position = spot(tgame.world, -11.0, -13.0)                     # a dash across the store
        tgame.update(0.05)
        assert guard.position.distance_to(tgame.player.position) <= w.settings.justice.charge_reach
        assert until(tgame, lambda: w.jail.serving, seconds=20.0)

    def test_a_save_taken_mid_arrest_summons_the_watchman_again(self, tgame, settings, tmp_path):
        w, t = self._to_the_cell(tgame)
        tgame.player.freeze_timer = 0.0                                      # (the watchman holds the controls: pretend not)
        assert tgame.save_game()
        data = json.load(open(tgame.saves.filename))
        assert data["tutorial"]["stage"] == "caught" and data["tutorial"]["caught_guard"] == WATCHMAN
        fresh = Game(settings=settings, save_file=tgame.saves.filename)      # a new run of the program: no Guard Tobin in its town
        assert fresh.world.find_npc(WATCHMAN) is None
        assert fresh.load_game()
        assert fresh.tutorial.stage is Stage.CAUGHT
        assert until(fresh, lambda: fresh.world.jail.serving, seconds=20.0)    # ... until the tutorial calls him back
        assert any(m.startswith(f"({WATCHMAN}") and "days in the cells" in m for m in fresh.messages.messages)
        assert fresh.world.find_npc(WATCHMAN) is None and fresh.tutorial.stage is Stage.JAIL

    def test_loading_an_earlier_save_sends_the_watchman_away_with_his_grudge(self, tgame):
        """F9 back to before the theft, in the same session: the watchman who stepped in (and only
        exists for that arrest) is not left standing in the store holding a shoplifting that never
        happened - the guards' memories are what every later fine is totted up from."""
        w, t = tgame.world, tgame.tutorial
        settle(tgame, 1.0)
        tgame.toggle_torch()
        settle(tgame, 0.5)
        assert tgame.save_game()                                              # before the job
        residents = len(w.npcs)
        pick_the_shop_door(tgame)
        take_the_candlestick(tgame)
        tgame.update(0.05)
        assert t.stage is Stage.CAUGHT and w.find_npc(WATCHMAN) is not None
        tgame.player.freeze_timer = 0.0
        assert tgame.load_game()
        assert tgame.tutorial.stage is not Stage.CAUGHT
        assert w.find_npc(WATCHMAN) is None and len(w.npcs) == residents
        assert WATCHMAN not in w.unfrozen_guards
        assert not any(m.crime == "Shoplifting" for g in w.guards() for m in g.crime_memories)
        tgame.tutorial._finish()                                              # the town opens up ...
        tgame.update(0.1)
        assert justice_bill(w, tgame.player) == 0.0                           # ... with nothing against the player

    def test_caught_at_the_lock_is_the_same_lesson_sooner(self, tgame):
        from unprepared.systems import justice
        w, t = tgame.world, tgame.tutorial
        settle(tgame, 1.0)
        justice.imprison(w, tgame.player, 9)                                  # a patrol saw the picking: straight to the cells
        tgame.update(0.05)
        assert t.stage is Stage.JAIL and w.jail.sentence_end_day == w.day_count + 1

    def test_serving_the_night_then_the_lockbox_the_boots_the_torch(self, tgame):
        w, t = self._to_the_cell(tgame)
        assert until(tgame, lambda: w.jail.serving, seconds=20.0)
        serve_the_night(tgame)
        assert w.jail.released and w.clock.hour == 6 and t.stage is Stage.LOCKBOX
        p = tgame.player
        p.position = spot(tgame.world, 51.0, 14.0)                                # at the lockbox in the office
        tgame.update(0.1)
        box = w.find_container("jail_storage")
        assert not box.is_locked
        target(tgame, TargetKind.CONTAINER, box)
        assert tgame.mode is GameMode.CONTAINER
        assert all(r.outcome is TransferOutcome.MOVED for r in tgame.loot_all())
        tgame.close_transfer()
        tgame.update(0.1)
        assert t.stage is Stage.DRESS and any(i.id == "soft_shoes" for i in p.inventory)
        tgame.tab()
        boots = next(i for i in p.inventory if i.id == "soft_shoes")
        tgame.equipment_cursor = p.inventory.index(boots)
        tgame.equipment_toggle()
        assert boots.equipped
        tgame.tab()
        tgame.update(0.1)
        assert t.stage is Stage.WAIT_FOR_NIGHT and "[T]" in tgame.tutorial_hint
        settle(tgame, 10.0)
        assert t.stage is Stage.WAIT_FOR_NIGHT                                # 06:00: daylight, however long you stand there
        while w.clock.hour < 20:
            assert tgame.wait()
            tgame.update(0.1)
        assert t.stage is Stage.STEAL_AGAIN and t.waited                      # 20:00: night; back to the store

    def test_already_night_the_wait_stage_still_asks_for_one_wait(self, tgame):
        w, t = tgame.world, tgame.tutorial
        w.clock.hour = 23
        t._enter(Stage.WAIT_FOR_NIGHT)
        settle(tgame, 2.0)
        assert t.stage is Stage.WAIT_FOR_NIGHT
        assert tgame.wait()
        tgame.update(0.1)
        assert t.stage is Stage.STEAL_AGAIN

    def test_picks_are_replaced_after_the_cells_too(self, tgame):
        w, t = self._to_the_cell(tgame)
        assert until(tgame, lambda: w.jail.serving, seconds=20.0)
        serve_the_night(tgame)
        p = tgame.player
        assert t.stage is Stage.LOCKBOX
        p.inventory = [i for i in p.inventory if not i.is_lockpick]
        tgame.update(0.1)
        assert p.has_lockpick() and t.picks_given == 1

    def test_the_lock_picked_without_the_torch_lit_still_moves_the_lesson_on(self, tgame):
        t = tgame.tutorial
        settle(tgame, 1.0)
        pick_the_shop_door(tgame)                                             # the torch lesson ignored
        assert t.stage is Stage.IN_THE_SHOP
        assert take_the_candlestick(tgame).outcome is TransferOutcome.MOVED
        tgame.update(0.05)
        assert t.stage is Stage.CAUGHT

    def test_a_jailbreak_counts_as_out_too(self, tgame):
        w, t = self._to_the_cell(tgame)
        assert until(tgame, lambda: w.jail.serving, seconds=20.0)
        while tgame.in_transition or tgame.frozen:
            tgame.update(0.05)
        tgame.player.position = spot(tgame.world, -12.0, 3.0)                     # out through the bars and away
        tgame.update(0.1)
        assert w.jail.escaped and t.stage is Stage.LOCKBOX
        tgame.update(0.1)
        assert t.stage is Stage.DRESS                                         # out of the jail house: the lockbox can wait
        tgame.tab(); tgame.update(0.1); tgame.tab(); tgame.update(0.1)
        assert t.stage is Stage.WAIT_FOR_NIGHT                                # no boots to put on: the view seen is enough


# --- the job done properly, and the town opens up ----------------------------------------------------
class TestTheJobAndTheTown:
    def _free_again(self, tgame):
        w, t, p = tgame.world, tgame.tutorial, tgame.player
        t._enter(Stage.STEAL_AGAIN)
        w.clock.hour = 2
        w.resync_schedules(announce=False, teleport=True)
        p.position = spot(tgame.world, -12.0, 3.0)
        tgame.update(0.1)
        return w, t, p

    def test_the_candlestick_sold_to_jae_for_next_to_nothing_earns_his_kindness(self, tgame):
        w, t, p = self._free_again(tgame)
        w.find_door("shop_door").is_locked = False
        assert take_the_candlestick(tgame).outcome is TransferOutcome.MOVED
        tgame.update(0.1)
        assert t.stage is Stage.SELL
        jae = w.find_npc(FENCE)
        p.position = spot(tgame.world, -10.0, 10.5)
        tgame.update(0.1)
        target(tgame, TargetKind.NPC, jae, NPCAction.TRADE)
        assert tgame.mode is GameMode.TRADE                                   # now he deals - in the candlestick only
        assert tgame.transfer_confirm().outcome is TransferOutcome.REFUSED     # buying a pick: "the candlestick first"
        tgame.transfer_switch_side()
        rows = [i.id for i in tgame.transfer.rows(Side.RIGHT)]
        tgame.transfer.selection[Side.RIGHT] = rows.index("iron_lockpick")
        assert tgame.transfer_confirm().outcome is TransferOutcome.REFUSED
        tgame.transfer.selection[Side.RIGHT] = rows.index(TARGET_ITEM)
        result = tgame.transfer_confirm()
        assert result.outcome is TransferOutcome.MOVED and result.price < 10.0     # barely anything
        tgame.close_transfer()
        tgame.update(0.1)
        assert t.stage is Stage.TOOLS and not t.locked
        assert p.gold == pytest.approx(result.price + KINDNESS_GOLD)
        settle(tgame, 8.0)
        assert any("sixty gold" in m for m in tgame.messages.messages if m.startswith("(Jae"))
        # the town is open now
        assert t.refuses("door", w.find_door("inn_door")) is None
        assert t.refuses("trade", w.find_npc("Shopkeeper Barnaby")) is None
        assert t.refuses("pickpocket", jae) is None

    def test_tools_ore_picks_and_a_bed(self, tgame):
        w, t, p = self._free_again(tgame)
        t._enter(Stage.TOOLS)
        p.gold = KINDNESS_GOLD
        for item_id, name in (("pickaxe", "Pickaxe"), ("mallet", "Mallet")):
            p.add_item(Item(item_id, name, value=12.0, category="tool", owner="player"))
        tgame.update(0.1)
        assert t.stage is Stage.MINE
        ore = w.find_station("ore_deposit")
        assert use_station(p, ore, tgame.settings, tgame.messages.add, rng=tgame.rng) is not None
        tgame.update(0.1)
        assert t.stage is Stage.FORGE
        anvil = w.find_station("anvil")
        tgame.rng.random = lambda: 0.0                                        # a sure hand at the anvil
        assert use_station(p, anvil, tgame.settings, tgame.messages.add, rng=tgame.rng) is not None
        tgame.update(0.1)
        assert t.stage is Stage.INN and "Drowsy Dragon" in tgame.tutorial_hint
        assert not any(DONE_MESSAGE[:20] in m for m in tgame.messages.messages)    # the town is not declared open yet
        p.position = spot(tgame.world, -26.0, -12.0)                              # in at the inn's door
        assert w.building_at(p.position) is w.inn_building()
        tgame.update(0.1)
        assert t.stage is Stage.DONE and not tgame.in_tutorial
        assert tgame.tutorial_hint == DONE_MESSAGE                           # the closing words stay in the panel ...
        assert not w.suppress_curfews and not w.guards_frozen
        assert any(m.startswith("TUTORIAL COMPLETE") and DONE_MESSAGE in m for m in tgame.messages.messages)
        settle(tgame, 25.0)
        assert tgame.tutorial_hint == DONE_MESSAGE
        settle(tgame, 6.0)
        assert tgame.tutorial_hint == ""                                     # ... for half a minute

    def test_a_room_rented_counts_as_a_bed_found(self, tgame):
        w, t, p = self._free_again(tgame)
        t._enter(Stage.INN)
        w.inn.rented_room, w.inn.rent_end_day = 0, w.day_count + 1
        tgame.update(0.1)
        assert t.stage is Stage.DONE

    def test_losing_the_candlestick_again_means_stealing_it_again(self, tgame):
        w, t, p = self._free_again(tgame)
        t._enter(Stage.SELL)
        tgame.update(0.1)                                                     # (never had it: confiscated, say)
        assert t.stage is Stage.STEAL_AGAIN


# --- persistence and the record --------------------------------------------------------------------
class TestPersistence:
    def test_the_stage_and_the_leash_survive_a_save(self, tgame):
        w, t = tgame.world, tgame.tutorial
        t._enter(Stage.STEAL_AGAIN)
        t.tab_opened = True
        tgame.update(0.1)
        assert tgame.save_game()
        data = json.load(open(tgame.saves.filename))
        assert data["tutorial"]["stage"] == "steal_again" and data["tutorial"]["tab_opened"]
        tgame.tutorial = None
        w.suppress_curfews = False
        assert tgame.load_game()
        assert tgame.tutorial is not None and tgame.tutorial.stage is Stage.STEAL_AGAIN and tgame.tutorial.tab_opened
        assert w.suppress_curfews and tgame.tutorial.locked

    def test_a_save_without_the_tutorial_loads_without_one(self, tgame, settings, tmp_path):
        plain = Game(settings=settings, save_file=str(tmp_path / "plain.json"))
        plain.update(0.1)
        assert plain.save_game()
        assert tgame.load_game(str(tmp_path / "plain.json"))
        assert tgame.tutorial is None and not tgame.world.suppress_curfews

    def test_the_snapshot_names_the_stage(self, tgame, settings, tmp_path):
        assert tgame.snapshot()["tutorial"] == "intro"
        assert Game(settings=settings, save_file=str(tmp_path / "p.json")).snapshot()["tutorial"] is None

    def test_every_locked_stage_has_a_hint(self):
        from unprepared.tutorial import HINTS
        for stage in LOCKED_STAGES:
            assert HINTS[stage]
        assert Tutorial.__name__ == "Tutorial"

    def test_an_unknown_or_old_stage_in_a_save_is_not_fatal(self, tgame):
        assert Tutorial.from_dict(tgame, {"stage": "wait"}).stage is Stage.INN
        assert Tutorial.from_dict(tgame, {"stage": "no_such_stage"}).stage is Stage.INTRO


# --- the guards stand still ----------------------------------------------------------------------
class TestFrozenGuards:
    def test_every_guard_is_held_where_they_stand_for_the_whole_tutorial(self, tgame):
        w = tgame.world
        assert w.guards_frozen and all(w.is_frozen(g) for g in w.guards())
        before = {g.name: (g.position.copy(), g.state) for g in w.guards()}
        settle(tgame, 20.0)
        for g in w.guards():
            assert g.position.distance_to(before[g.name][0]) < 1e-6 and g.state is before[g.name][1]

    def test_frozen_guards_see_nothing_and_answer_no_cry(self, tgame):
        w, p = tgame.world, tgame.player
        mira = next(g for g in w.guards() if not g.is_asleep)
        p.position = spot(tgame.world, mira.position.x + 2.0, mira.position.z)   # unlit, crouching, right beside her
        mira.turn_towards(p.position)
        p.is_crouching = True
        settle(tgame, 5.0)
        assert mira.suspicion_meter == 0.0 and not mira.is_alarmed
        w.declare_crime("Shoplifting", p.position.copy(), witnessed=False, reporter="Shopkeeper Barnaby")
        tgame.update(0.1)
        assert not any(g.crime_memories for g in w.guards()) and not any(g.state is AIState.RESPONDING for g in w.guards())

    def test_the_hour_does_not_move_them_either(self, tgame):
        w = tgame.world
        before = {g.name: g.position.copy() for g in w.guards()}
        w.wait_hours(9)                                                       # 22:00 -> 07:00: the day shift would start
        assert all(g.position.distance_to(before[g.name]) < 1e-6 for g in w.guards())

    def test_the_tutorials_own_watchman_is_not_held(self, tgame):
        w = tgame.world
        settle(tgame, 1.0)
        tgame.toggle_torch()
        pick_the_shop_door(tgame)
        assert take_the_candlestick(tgame).outcome is TransferOutcome.MOVED
        tgame.update(0.05)
        tobin = w.find_npc(WATCHMAN)
        assert tobin is not None and not w.is_frozen(tobin) and w.is_frozen(w.find_npc("Guard Mira"))
        assert until(tgame, lambda: w.jail.serving, seconds=20.0)
        assert WATCHMAN not in w.unfrozen_guards

    def test_the_guards_walk_again_once_the_tutorial_is_done(self, tgame):
        w, t = tgame.world, tgame.tutorial
        w.clock.hour = 9
        t._enter(Stage.INN)
        tgame.player.position = spot(tgame.world, -26.0, -12.0)
        tgame.update(0.1)
        assert t.stage is Stage.DONE and not w.guards_frozen
        vance = w.find_npc("Guard Captain Vance")
        assert vance.current_task is not None and vance.current_task.start_hour <= 9 < vance.current_task.end_hour

    def test_a_loaded_save_freezes_or_frees_them_to_match(self, tgame, settings, tmp_path):
        w = tgame.world
        assert tgame.save_game()
        w.guards_frozen = False
        assert tgame.load_game()
        assert w.guards_frozen
        plain = Game(settings=settings, save_file=str(tmp_path / "plain.json"))
        plain.update(0.1)
        assert plain.save_game()
        assert tgame.load_game(str(tmp_path / "plain.json"))
        assert not w.guards_frozen


# --- the waypoint arrow -----------------------------------------------------------------------------
class TestWaypoint:
    def _door(self, w, door_id):
        d = w.find_door(door_id)
        return Vector3D(d.center.x, d.hinge_position.y, d.hinge_position.z)

    def test_from_the_hideout_the_arrow_points_at_its_door_then_at_the_store(self, tgame):
        w, t, p = tgame.world, tgame.tutorial, tgame.player
        assert t.stage is Stage.INTRO
        assert t.waypoint == self._door(w, "hideout_door")
        assert tgame.tutorial_waypoint == t.waypoint
        p.position = spot(tgame.world, -12.0, 3.0)                                # out in the street
        tgame.update(0.1)
        assert t.stage is Stage.TORCH and t.waypoint == self._door(w, "shop_door")
        t._enter(Stage.TO_THE_SHOP)
        assert t.waypoint == self._door(w, "shop_door")

    def test_inside_the_store_it_points_at_the_shelf(self, tgame):
        w, t, p = tgame.world, tgame.tutorial, tgame.player
        t._enter(Stage.IN_THE_SHOP)
        p.position = spot(tgame.world, -13.5, -9.0)
        shelf = w.find_container("shop_shelf_west")
        assert t.waypoint == Vector3D(shelf.center.x, 0.0, shelf.center.z)
        p.position = spot(tgame.world, -13.5, -3.0)                                 # still outside: the door first
        assert t.waypoint == self._door(w, "shop_door")

    def test_no_arrow_while_the_next_step_is_a_key(self, tgame):
        t = tgame.tutorial
        for stage in (Stage.CAUGHT, Stage.JAIL, Stage.DRESS, Stage.DONE):
            t._enter(stage)
            assert t.waypoint is None
        t._enter(Stage.DONE)
        assert tgame.tutorial_waypoint is None

    def test_the_lockbox_jae_the_tools_the_ore_the_anvil_the_inn(self, tgame):
        w, t, p = tgame.world, tgame.tutorial, tgame.player
        t._enter(Stage.LOCKBOX)
        p.position = spot(tgame.world, 55.0, 13.0)                                # in the cell: the lockbox is in the same house
        box = w.find_container("jail_storage")
        assert t.waypoint == Vector3D(box.center.x, 0.0, box.center.z)
        t._enter(Stage.SELL)
        p.position = spot(tgame.world, -12.0, 3.0)
        assert t.waypoint == self._door(w, "hideout_door")
        p.position = spot(tgame.world, -12.0, 11.0)
        assert t.waypoint == w.find_npc(FENCE).position
        t._enter(Stage.TOOLS)
        p.position = spot(tgame.world, -13.5, -9.0)
        east = w.find_container("shop_shelf_east")
        assert t.waypoint == Vector3D(east.center.x, 0.0, east.center.z)
        p.position = spot(tgame.world, -12.0, 3.0)
        t._enter(Stage.MINE)
        ore = w.find_station("ore_deposit")
        assert t.waypoint == Vector3D(ore.center.x, 0.0, ore.center.z)
        t._enter(Stage.FORGE)
        anvil = w.find_station("anvil")
        assert t.waypoint == Vector3D(anvil.center.x, 0.0, anvil.center.z)
        t._enter(Stage.INN)
        assert t.waypoint == self._door(w, "inn_door")
        t._enter(Stage.WAIT_FOR_NIGHT)
        assert t.waypoint == self._door(w, "shop_door")

    def test_indoors_elsewhere_the_arrow_leads_out_of_the_door_first(self, tgame):
        w, t, p = tgame.world, tgame.tutorial, tgame.player
        t._enter(Stage.MINE)
        p.position = spot(tgame.world, -26.0, -12.0)                              # in the inn: its door, not the ore through the wall
        assert t.waypoint == self._door(w, "inn_door")

    def test_relative_yaw_is_the_turn_to_face_the_target(self, tgame):
        from unprepared.rendering.waypoint import relative_yaw
        p = tgame.player
        p.position, p.rotation_y = spot(tgame.world, 0.0, 0.0), 0.0              # facing -z
        assert relative_yaw(p, spot(tgame.world, 0.0, -5.0)) == pytest.approx(0.0)
        assert relative_yaw(p, spot(tgame.world, 5.0, 0.0)) == pytest.approx(90.0)      # to the right
        assert relative_yaw(p, spot(tgame.world, -5.0, 0.0)) == pytest.approx(-90.0)    # to the left
        p.rotation_y = 270.0
        assert relative_yaw(p, spot(tgame.world, 0.0, -5.0)) == pytest.approx(90.0)
