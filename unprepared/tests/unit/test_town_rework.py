"""The bigger town: specialist merchants, the open-sided shed, paving and doorsteps, people
locked out of where they are going, getting out of bed, the player's attention being held,
household pantries, and a few movement rules that the wider streets exposed."""
import pytest

from tests.conftest import bubbled, said
from tests.helpers import spot
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState, Item
from unprepared.game import GameMode
from unprepared.systems import MoveIntent, Side, TransferOutcome
from unprepared.world.levels.town import (ANVIL_SPOT, INN_SEATS, FORGE_SPOT, ORE_DEPOSIT, PATROL_ROUTE, POST_DOOR,
                                          SHOP_COUNTER, SHOP_FLOOR, STROLL_EAST, STROLL_WEST, WELL)
from unprepared.world.levels.town import BARNABY_BED, STARTING_GOLD


def park(game, *names):
    """Take these NPCs off their routines and out of the way."""
    for name in names:
        npc = game.world.find_npc(name)
        npc.schedule, npc.current_task, npc.path = [], None, []
        npc.position = spot(game.world, 46.0, 28.0)
        npc.state = AIState.IDLE


def face(game, npc, dist=2.0):
    """Put the player ``dist`` south of ``npc`` looking at them (and them at the player)."""
    game.player.position = spot(game.world, npc.position.x, npc.position.z + dist, npc.position.y)
    game.player.rotation_y, game.player.rotation_x = 0.0, -25.0
    npc.turn_towards(game.player.position)
    game.update(0.016, MoveIntent())


# ----------------------------------------------------------------------------- specialists
class TestSpecialistMerchants:
    def test_the_smith_sells_only_weapons_armour_and_ore(self, game):
        gareth = game.world.find_npc("Gareth the Smith")
        assert gareth.trade_categories == ("weapon", "armor", "ore", "tool") and gareth.role_tag == "smith"
        assert gareth.buys(Item("x", "Dagger", category="weapon")) and gareth.buys(Item("x", "Ore", category="ore"))
        assert not gareth.buys(Item("x", "Goblet")) and not gareth.buys(Item("x", "Cloak", category="clothing"))
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        assert barnaby.trade_categories == () and barnaby.buys(Item("x", "Anything", category="clothing"))

    def test_the_smith_trades_while_working_at_the_anvil(self, game):
        gareth = game.world.find_npc("Gareth the Smith")
        gareth.state, gareth.position = AIState.WORKING, game.world.spot(*ANVIL_SPOT)
        gareth.schedule, gareth.current_task, gareth.path = [], None, []
        assert gareth.is_open_for_business
        face(game, gareth)
        assert game.interaction_prompt == "[E] Talk / Trade with Gareth the Smith"
        game.interact()
        assert game.mode is GameMode.TRADE
        assert said(game.messages.messages[-1:], "greet_smith")
        stock = [i.id for i in game.transfer.rows(Side.LEFT)]
        assert "iron_dagger" in stock and "iron_ore" in stock and "hammer" not in stock     # his own hammer is not for sale

    def test_selling_the_smith_something_outside_his_trade_is_refused_without_closing(self, game):
        gareth = game.world.find_npc("Gareth the Smith")
        gareth.state, gareth.position = AIState.WORKING, game.world.spot(*ANVIL_SPOT)
        gareth.schedule, gareth.current_task, gareth.path = [], None, []
        face(game, gareth)
        game.interact()
        game.transfer_switch_side()
        game.cursor_down()                                   # soft shoes (goods)
        game.player.unequip(game.transfer.selected_item())   # (worn: selling them would ask first)
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.NOT_THEIR_TRADE and not result.close_session
        assert game.mode is GameMode.TRADE and game.player.gold == STARTING_GOLD
        assert said(game.messages.messages, "wrong_trade", role="smith")
        game.player.inventory.append(Item("ore", "Iron Ore", value=6.0, category="ore", stackable=True))
        rows = [i.id for i in game.transfer.rows(Side.RIGHT)]
        while rows[game.transfer.selection[Side.RIGHT]] != "ore":
            game.cursor_down()
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.MOVED and game.player.gold > STARTING_GOLD

    def test_the_tailor_deals_in_clothing_and_tools(self, game):
        mirelle = game.world.find_npc("Mirelle the Tailor")
        assert mirelle.trade_categories == ("clothing", "tool")
        assert mirelle.buys(Item("x", "Lockpick", category="tool")) and not mirelle.buys(Item("x", "Sword", category="weapon"))

    def test_every_stocked_item_has_a_category_the_right_merchant_accepts(self, game):
        w = game.world
        for name in ("Gareth the Smith", "Mirelle the Tailor", "Shopkeeper Barnaby", "Hilde the Innkeeper"):
            npc = w.find_npc(name)
            for item in w.merchant_stock(npc):
                if item.for_sale:
                    assert item.category == "service" or npc.buys(item), (name, item.id)


# ----------------------------------------------------------------------------- the yard
class TestOpenShed:
    def test_a_roof_on_posts_is_not_indoors(self, game):
        w = game.world
        under = Vector3D(ANVIL_SPOT[0], 0.0, ANVIL_SPOT[2])
        assert w.building_at(under) is None and not w.is_indoors(under)
        assert w.building_at(under, include_open=True).name == "smithy_shed"
        assert w.building_at(spot(game.world, -70.0, -10.0)).name == "smithy"

    def test_the_yard_is_lit_by_the_forge_and_open_to_the_sky(self, game):
        w = game.world
        assert any(light.kind == "fire" and light.room == "the smith's yard" for light in w.level.lights)
        spot = Vector3D(FORGE_SPOT[0], 0.0, FORGE_SPOT[2])
        assert w.area_of(spot) == "the smith's yard"
        assert not w.is_dark_at(spot)
        w.clock.hour = 23
        assert w.is_dark_at(spot) and w.is_dark_at(game.world.spot(*WELL))    # outdoors at night is dark, fire or no fire

    def test_the_ore_path_runs_through_the_shed(self, game):
        w = game.world
        gareth = w.find_npc("Gareth the Smith")
        path = w.find_path(gareth, game.world.spot(*ORE_DEPOSIT))
        assert path and path[-1].distance_to(game.world.spot(*ORE_DEPOSIT)) < 1.0


class TestRoadsAndDoorsteps:
    def test_the_main_street_is_paved_and_every_door_gets_a_doorstep(self, game):
        w = game.world
        assert len(w.roads) > len(w.level.roads)
        street = w.level.roads[0]
        assert street.min_xz[1] == -6.0 and street.max_xz[1] == 6.0
        # a door standing back from the paving (Barnaby's, on the north wall at z = 7) gets a strip up to it
        steps = [r for r in w.roads if r not in w.level.roads]
        barnaby = next(r for r in steps if r.min_xz[0] <= -70.0 <= r.max_xz[0] and r.min_xz[1] >= 6.0)
        assert barnaby.min_xz[1] == 6.0 and barnaby.max_xz[1] == 7.0
        # doors that open straight onto the paving (the shop's, at z = -6) need none
        assert not any(r.min_xz[0] <= -13.5 <= r.max_xz[0] and -6.5 < r.min_xz[1] < -5.5 for r in steps)

    def test_the_well_stands_on_the_street_and_is_solid(self, game):
        w = game.world
        assert w.area_of(game.world.spot(*WELL)) == "the well"
        assert w.nav.is_blocked(WELL[0], WELL[2])
        game.player.position = spot(game.world, -2.0, 0.0)
        game.player.rotation_y = 90.0
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True))
        assert game.player.position.x < -0.8                            # stopped by the well


# ----------------------------------------------------------------------------- routines
class TestLockedOut:
    def test_somebody_bound_for_a_shut_shop_waits_on_its_doorstep_and_retries(self, game):
        w = game.world
        dorran = w.find_npc("Dorran")                                  # hauls crates for Barnaby from 08:00
        shop = w.find_door("shop_door")
        assert shop.is_locked
        park(game, "Shopkeeper Barnaby")                                # Barnaby never opens up
        w.wait_hours(1)                                                 # 08:00: Dorran sets off
        for _ in range(60):
            game.update(0.5, MoveIntent())
            if dorran.locked_out and dorran.state is AIState.IDLE:     # (on the doorstep)
                break
        # (a bubble, not a log line - said as he finds the door shut: a passer-by may chat with him later)
        assert bubbled(dorran, "locked_out") and not said(game.messages.messages, "locked_out")
        for _ in range(20):
            game.update(0.5, MoveIntent())
        assert dorran.locked_out and dorran.state is AIState.IDLE
        assert dorran.position.distance_to(w.exit_of("the shop")) < 2.5
        assert dorran.retry_timer > 0 and dorran.npc_settings.locked_out_retry == 15.0
        shop.is_locked = False                                          # somebody unlocks it
        for _ in range(40):
            game.update(0.5, MoveIntent())
        assert not dorran.locked_out and dorran.state in (AIState.TRAVELING, AIState.WORKING)
        for _ in range(40):
            game.update(0.5, MoveIntent())
        assert dorran.state is AIState.WORKING and w.area_of(dorran.position) in ("the shop", "behind the counter")

    def test_the_wait_is_abandoned_when_the_next_task_comes_round(self, game):
        w = game.world
        dorran = w.find_npc("Dorran")
        park(game, "Shopkeeper Barnaby")
        w.wait_hours(1)
        for _ in range(40):
            game.update(0.5, MoveIntent())
        assert dorran.locked_out
        w.wait_hours(4)                                                 # 12:00: home for his midday meal
        assert not dorran.locked_out and dorran.state is AIState.EATING
        assert w.area_of(dorran.position) == "Lys and Dorran's house"


class TestGettingUp:
    def test_a_woken_sleeper_steps_off_the_bed(self, game):
        w = game.world
        barnaby = w.find_npc("Shopkeeper Barnaby")
        barnaby.schedule, barnaby.current_task, barnaby.path = [], None, []
        barnaby.position = game.world.spot(*BARNABY_BED)                # on his bed
        barnaby.state = AIState.SLEEPING
        game.update(0.05, MoveIntent())
        assert w.prop_under(barnaby.position) is not None              # asleep: stays put
        barnaby.state = AIState.INVESTIGATING
        game.update(0.05, MoveIntent())
        assert w.prop_under(barnaby.position) is None                  # awake: off the bed ...
        assert barnaby.position.distance_to(game.world.spot(*BARNABY_BED)) < 2.5    # ... but right beside it
        assert w.area_of(barnaby.position) == "Barnaby's house"

    def test_a_walker_already_on_a_path_is_left_alone(self, game):
        w = game.world
        barnaby = w.find_npc("Shopkeeper Barnaby")
        barnaby.schedule, barnaby.current_task = [], None
        barnaby.position = game.world.spot(*BARNABY_BED)
        barnaby.state = AIState.TRAVELING
        barnaby.path = [spot(game.world, -69.0, 9.5)]
        w.update_npcs(0.0)
        assert barnaby.position == game.world.spot(*BARNABY_BED)


class TestAttention:
    def test_a_caught_player_is_frozen_and_turned_smoothly(self, game):
        p = game.player
        p.rotation_y, p.rotation_x = 0.0, 30.0
        p.hold_attention(Vector3D(p.position.x + 5.0, p.position.y, p.position.z), 1.0)     # somebody due east
        assert p.frozen and game.frozen
        p.update_turning(0.1)                                           # 15 degrees per tick
        assert p.rotation_y == pytest.approx(15.0) and p.rotation_x == pytest.approx(15.0)
        for _ in range(5):
            p.update_turning(0.1)
        assert p.rotation_y == pytest.approx(90.0) and p.rotation_x == 0.0 and p.turn_target is None
        for _ in range(5):
            p.update_turning(0.1)
        assert not p.frozen

    def test_from_a_crouch_the_view_tilts_up_to_their_face(self, game):
        p = game.player
        p.is_crouching = True
        p.current_eye_height = p.settings.crouch_eye_height
        p.rotation_y, p.rotation_x = 0.0, 0.0
        target = Vector3D(p.position.x, p.position.y, p.position.z - 2.0)     # somebody two paces north
        p.hold_attention(target, 1.0)
        for _ in range(10):
            p.update_turning(0.1)
        assert p.rotation_x == pytest.approx(p.pitch_towards(target)) and 10.0 < p.rotation_x < 25.0
        assert p.turn_target is None
        p.is_crouching, p.current_eye_height = False, p.settings.eye_height
        assert p.pitch_towards(target) == pytest.approx(0.0)                  # standing: level with their face

    def test_the_turn_takes_the_short_way_round(self, game):
        p = game.player
        p.rotation_y = 350.0
        p.hold_attention(Vector3D(p.position.x + 1.0, p.position.y, p.position.z - 5.0), 1.0)   # just east of north (~11 deg)
        p.update_turning(0.1)
        assert p.rotation_y == pytest.approx(5.0)

    def test_controls_are_ignored_while_frozen_but_time_passes(self, game):
        p = game.player
        before = p.position.copy()
        p.hold_attention(Vector3D(p.position.x, p.position.y, p.position.z + 5.0), 0.5)
        hour_before = game.world.clock.minute_accumulator
        game.update(0.1, MoveIntent(forward=True))
        assert p.position == before and game.world.clock.minute_accumulator > hour_before
        game.look(100, 0)
        assert p.rotation_y != 0.0 or p.turn_target is not None      # the view is being steered, not by the mouse
        for _ in range(10):
            game.update(0.1, MoveIntent())
        game.update(0.1, MoveIntent(forward=True))
        assert p.position != before

    def test_a_guards_scolding_holds_attention(self, game):
        w = game.world
        vance = w.find_npc("Guard Captain Vance")
        park(game, *(n.name for n in w.guards() if n is not vance))
        vance.schedule, vance.current_task, vance.path = [], None, []
        vance.position = spot(game.world, 3.0, -3.0)
        vance.state = AIState.GUARDING
        game.player.position = spot(game.world, 3.0, 1.0)
        game.player.rotation_y = 180.0                                  # back to him
        vance.turn_towards(game.player.position)
        game.toggle_crouch()
        game.rng.random = lambda: 0.99
        for _ in range(40):
            game.update(0.05, MoveIntent(left=True, right=True))
        assert said(game.messages.messages, "skulking")
        assert game.player.turn_target is not None or game.player.rotation_y != 180.0
        for _ in range(40):
            game.update(0.05, MoveIntent())
        assert game.player.turn_target is None                          # the swing is over ...
        off = (game.player.rotation_y - game.player.yaw_towards(vance.position) + 180.0) % 360.0 - 180.0
        assert abs(off) < 90.0                                          # ... and he is in front of us (he kept walking up)


class TestPantries:
    def test_households_eat_from_their_tables_and_buy_more_every_few_days(self, game):
        w = game.world
        assert game.settings.economy.pantry_restock_days == 3
        table = w.find_container("couple_table")
        start = table.inventory[0].quantity
        w.wait_hours(14)                                                # breakfast, lunch, supper for two
        assert table.inventory and table.inventory[0].quantity < start
        w.wait_hours(11)                                                # past midnight: no shopping trip yet
        assert table.inventory[0].quantity < start
        w.wait_hours(48)                                                # ... until the third night
        assert table.inventory[0].quantity == start
        assert w.pantries["couple_table"][0].id == "stew"

    def test_every_table_is_laid_for_three_days(self, game):
        """Nobody goes hungry between shopping trips."""
        w = game.world
        for _ in range(3 * 24):
            w.wait_hours(1)
            bad = [m for m in game.messages.messages if "nothing to eat" in m or "Who's had" in m or "crumb" in m]
            assert not bad, (w.day_count, w.clock.hour, bad)

    def test_a_cleared_table_is_laid_again(self, game):
        w = game.world
        table = w.find_container("tamsin_table")
        table.inventory.clear()
        w.day_count = 3
        w._daily_upkeep()
        assert [(i.id, i.quantity) for i in table.inventory] == [("stew", 14)]

    def test_shop_shelves_are_not_pantries(self, game):
        assert "shop_shelf_west" not in game.world.pantries and "guard_table" in game.world.pantries


# ----------------------------------------------------------------------------- movement rules
class TestLongFrames:
    def test_an_owner_carried_through_their_door_in_one_step_still_locks_it(self, game):
        """At one-second frames an NPC moves 3.6 units: the doorway is crossed within a single step."""
        w = game.world
        barnaby = w.find_npc("Shopkeeper Barnaby")
        door = w.find_door("home_door")
        w.wait_hours(1, step=1.0)                                       # 08:00: off to the shop, 1 s steps
        for _ in range(6):
            w.update_time(1.0)
            w.update_npcs(1.0)
        assert w.area_of(barnaby.position) != "Barnaby's house"
        assert door.is_locked and not door.is_open

    def test_an_investigating_guard_keeps_watching_a_suspect_in_view(self, game):
        w = game.world
        vance = w.find_npc("Guard Captain Vance")
        park(game, *(n.name for n in w.guards() if n is not vance))
        vance.schedule, vance.current_task, vance.path = [], None, []
        vance.position = spot(game.world, 3.0, -3.0)
        vance.state = AIState.PATROL
        game.player.position = spot(game.world, 3.0, 1.0)
        vance.turn_towards(game.player.position)
        game.toggle_crouch()
        game.rng.random = lambda: 0.99                                  # every sneak roll fails
        for _ in range(200):                                            # 10 s: he walks over and stares
            game.update(0.05, MoveIntent(left=True, right=True))
            if game.player.gold < STARTING_GOLD:
                break
        assert vance.faces(game.player.position) and vance.position.distance_to(game.player.position) < 2.0
        assert said(game.messages.messages, "halt") and game.player.gold < STARTING_GOLD     # the alarm, then the fine


class TestSchedulesOnTheStreet:
    def test_strollers_walk_their_loops_and_stop_at_the_well(self, game):
        w = game.world
        tamsin = w.find_npc("Tamsin")
        task = tamsin.task_for(9)
        assert task.target_state is AIState.STROLLING and task.patrol_points == tuple(w.spot(*p) for p in STROLL_WEST)
        assert any(p.distance_to(game.world.spot(*WELL)) < 4.0 for p in task.patrol_points)   # (the well is 2.6 across)
        assert w.find_npc("Pell").task_for(9).patrol_points == tuple(w.spot(*p) for p in STROLL_EAST)

    def test_the_patrol_covers_the_whole_street(self, game):
        """End to end of Market Street, up East Lane and along Upper Street past the manors, down
        North Lane to the square and down Gate Road to the gate."""
        on_market = [p for p in PATROL_ROUTE if -6.0 <= p[2] <= 6.0]
        xs = [p[0] for p in on_market]
        assert min(xs) <= -60 and max(xs) >= 30
        assert any(p[2] <= -36.0 for p in PATROL_ROUTE)                     # Upper Street
        assert any(p[2] >= 40.0 for p in PATROL_ROUTE)                      # down by the gate
        w = game.world
        for a, b in zip(PATROL_ROUTE, PATROL_ROUTE[1:] + PATROL_ROUTE[:1]):
            assert w.nav.find_path(w.spot(*a), w.spot(*b)), (a, b)          # every leg walkable

    def test_the_watch_rotates_through_the_day_post(self, game):
        w = game.world
        post = game.world.spot(*POST_DOOR)
        at_post = {h: [n.name for n in w.guards() if n.task_for(h) and n.task_for(h).target_pos == post
                       and n.task_for(h).target_state is AIState.GUARDING] for h in (8, 16, 23)}
        assert at_post[8] == ["Guard Orrin"] and at_post[16] == ["Guard Mira"] and at_post[23] == []
        gate = {h: sorted(n.name for n in w.guards() if n.task_for(h) and w.area_of(n.task_for(h).target_pos) == "the town gate")
                for h in (8, 12, 20, 3)}
        assert gate == {8: ["Guard Bram"], 12: ["Guard Bram"], 20: ["Guard Wenna"], 3: ["Guard Wenna"]}   # always manned

    def test_the_inn_has_a_seat_for_every_regular(self, game):
        w = game.world
        drinkers = [n for n in w.npcs if any(t.target_state is AIState.DRINKING for t in n.schedule)]
        assert len(drinkers) >= 5
        seats = {(round(t.target_pos.x, 2), round(t.target_pos.z, 2))
                 for n in drinkers for t in n.schedule if t.target_state is AIState.DRINKING and w.area_of(t.target_pos) == "the inn"}
        assert seats <= {(s[0], s[2]) for s in INN_SEATS} and len(seats) >= 5

    def test_shoppers_stand_on_the_shop_floor_not_behind_the_counter(self, game):
        w = game.world
        floor, counter = game.world.spot(*SHOP_FLOOR), game.world.spot(*SHOP_COUNTER)
        assert w.area_of(floor) == "the shop" and w.area_of(counter) == "behind the counter"
        for n in w.npcs:
            for t in n.schedule:
                if t.target_state is AIState.SHOPPING and w.area_of(t.target_pos) in ("the shop", "behind the counter"):
                    assert t.target_pos == floor


class TestTradeAvailability:
    """Who trades when: merchants at work, the innkeeper whenever awake, the fence always."""

    def stand_before(self, game, npc, dist=2.0, pitch=-15.0):
        from tests.unit.test_game import teleport
        teleport(game, npc.position.x, npc.position.z + dist, yaw=0.0, pitch=pitch)
        npc.turn_towards(game.player.position)
        game.update(0.0)

    def test_the_innkeeper_lets_rooms_while_cleaning_but_not_while_asleep_or_alarmed(self, game):
        w = game.world
        hilde = w.find_npc("Hilde the Innkeeper")
        assert hilde.trade_availability == "awake"
        hilde.path, hilde.current_task, hilde.schedule = [], None, []
        for state in (AIState.CLEANING, AIState.WORKING, AIState.EATING, AIState.SHOPKEEPING):
            hilde.state = state
            assert hilde.is_open_for_business, state
        self.stand_before(game, hilde)
        assert game.interaction_prompt == "[E] Talk / Trade with Hilde the Innkeeper"
        game.interact()
        assert game.mode is GameMode.TRADE and any(i.id == "rent_night" for i in game.transfer.rows(Side.LEFT))
        game.escape()
        hilde.state = AIState.SLEEPING
        assert not hilde.is_open_for_business
        hilde.state = AIState.REPORTING
        assert not hilde.is_open_for_business
        hilde.state = AIState.WAITING
        assert not hilde.is_open_for_business

    def test_ordinary_merchants_still_need_to_be_at_work(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        assert barnaby.trade_availability == "at_work"
        barnaby.state = AIState.EATING
        assert not barnaby.is_open_for_business
        barnaby.state = AIState.SHOPKEEPING
        assert barnaby.is_open_for_business

    def test_a_sleeping_fence_is_woken_for_business_and_goes_back_to_bed(self, game):
        w = game.world
        jae = w.find_npc("Jae the Fence")
        assert jae.trade_availability == "always"
        w.wait_until(2)                                               # 02:00: Jae asleep in his den
        assert jae.is_asleep and jae.is_open_for_business
        bed = jae.position.copy()
        from tests.helpers import teleport                             # (his bed's head is at the south wall: come from the north)
        teleport(game, jae.position.x, jae.position.z - 2.2, yaw=180.0, pitch=-35.0)   # he is lying down: look at the bed
        assert game.interaction_prompt == "[E] Talk / Trade with Jae the Fence"
        game.interact()
        assert game.mode is GameMode.TRADE and not jae.is_asleep and jae.awake_for_business and jae.is_busy
        assert said(game.messages.messages, "greet_fence_woken")
        game.transfer_switch_side()
        rows = game.transfer.rows(Side.RIGHT)
        game.transfer.selection[Side.RIGHT] = next(i for i, it in enumerate(rows) if it.id == "silver_goblet")
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED    # he buys the hot goblet at 02:00
        for _ in range(5):
            game.update(0.1)                                          # the hour's routine does not reclaim him mid-deal
        assert jae.state is AIState.SHOPKEEPING
        game.escape()                                                 # deal done
        assert not jae.awake_for_business
        for _ in range(5):
            game.update(0.1)
        assert jae.state is AIState.WAITING and said(game.messages.messages, "leave_late")   # the hideout's curfew: out you go
        game.player.position = spot(game.world, -12.0, 3.0)               # ... so we leave
        for _ in range(80):
            game.update(0.1)
        assert jae.is_asleep and jae.position.distance_to(bed) < 0.5    # and he is back in bed

    def test_an_awake_fence_is_not_marked_as_roused(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SHOPKEEPING
        assert jae.wake_for_business() is False and not jae.awake_for_business
