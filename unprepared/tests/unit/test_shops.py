"""Shopkeepers and their premises: closing time clears the shop (unless a sneak succeeds),
skulking customers are told to stand up or get out, and a wait never skips either."""
from __future__ import annotations

import pytest

from tests.helpers import spot
from tests.conftest import said
from unprepared.entities import AIState, SkillType
from unprepared.systems import MoveIntent
from unprepared.world.levels.town import (INN_BAR, SHOP_CLOSE_HOUR, SHOP_COUNTER, SHOP_FLOOR, TAILOR_CLOSE_HOUR,
                                          TAILOR_COUNTER, TAILOR_FLOOR)
from unprepared.world.levels.town import STARTING_GOLD

SHOPS = [  # keeper, a spot on the shop floor, the hour the keeper leaves the premises, the area, the keeper's counter
    ("Shopkeeper Barnaby", SHOP_FLOOR, SHOP_CLOSE_HOUR, "the shop", SHOP_COUNTER),
    ("Mirelle the Tailor", TAILOR_FLOOR, TAILOR_CLOSE_HOUR, "the tailor's", TAILOR_COUNTER),
    ("Hilde the Innkeeper", (-29.25, 0.0, -11.5), 0, "the inn", INN_BAR),
]


def at_closing(game, keeper_name, spot, close_hour):
    """Stand on the shop floor half an hour before the keeper leaves; returns the keeper."""
    w = game.world
    w.wait_until((close_hour - 1) % 24)
    w.wait_hours(0.5)
    game.player.position = game.world.spot(*spot)
    game.player.rotation_y = 0.0
    game.update(0.0)
    keeper = w.find_npc(keeper_name)
    assert w.player_on_premises_of(keeper) is not None, (keeper_name, w.area_of(game.player.position))
    return keeper


def settle(game, seconds=5.0):
    for _ in range(int(seconds / 0.1)):
        game.update(0.1)


class TestClosingTime:
    @pytest.mark.parametrize("keeper_name, spot, close_hour, area", [s[:4] for s in SHOPS[:2]])
    def test_standing_in_the_shop_at_closing_is_shown_out_and_the_wait_interrupted(self, game, keeper_name, spot, close_hour, area):
        w = game.world
        keeper = at_closing(game, keeper_name, spot, close_hour)
        assert game.wait()                                            # [T]: an hour - but closing time comes first
        assert w.wait_interruption == "evicted" and "Your wait is interrupted." in game.messages.messages
        assert w.clock.hour == close_hour and w.clock.minute_accumulator < 2.0
        assert said(game.messages.messages, "shop_closing") and f"You are shown out of {area}" in " ".join(game.messages.messages)
        assert game.frozen and game.in_transition                     # the usual eviction: attention, then the fade
        settle(game)
        assert w.building_at(game.player.position) is None            # out on the street
        assert w.player_on_premises_of(keeper) is None

    def test_a_crouching_player_whose_sneak_succeeds_is_locked_in_unnoticed(self, game):
        w = game.world
        keeper = at_closing(game, *SHOPS[0][:3])
        game.toggle_crouch()
        game.rng.random = lambda: 0.0                                 # every roll succeeds
        xp = game.player.skills.skills[SkillType.SNEAK]["xp"]
        assert game.wait()
        assert w.wait_interruption is None and not game.frozen
        assert "Barnaby locks up without noticing you." in game.messages.messages
        assert w.area_of(game.player.position) == "the shop" and w.find_door("shop_door").is_locked
        assert game.player.skills.skills[SkillType.SNEAK]["xp"] > xp   # a successful sneak teaches
        assert w.player_unwelcome_at(game.player.position) is not None   # ... but you are trespassing now
        assert keeper.state is not AIState.SHOPKEEPING

    def test_a_crouching_player_whose_sneak_fails_is_found_and_shown_out(self, game):
        w = game.world
        at_closing(game, *SHOPS[0][:3])
        game.toggle_crouch()
        game.rng.random = lambda: 0.99                                # every roll fails
        assert game.wait()
        assert w.wait_interruption == "evicted" and said(game.messages.messages, "shop_closing")
        settle(game)
        assert w.area_of(game.player.position) != "the shop"

    @pytest.mark.parametrize("guest", [True, False])
    def test_anybody_may_stay_in_the_inn_after_closing(self, game, guest):
        """The inn is a public house at every hour: the keeper going to bed clears nobody out -
        a paying guest or a visitor without a room alike."""
        w = game.world
        if guest:
            w.inn.rent(w, game.player, 1)
            game.close_screens()
        keeper = at_closing(game, *SHOPS[2][:3])
        assert game.wait()
        assert w.wait_interruption is None and w.area_of(game.player.position) == "the inn"
        assert not said(game.messages.messages, "shop_closing")
        assert keeper.state is not AIState.SHOPKEEPING

    def test_lunch_on_the_premises_is_not_closing_time(self, game):
        """Mirelle eats her lunch at the side table of her own shop (13:00, well inside its public
        hours): a customer on the shop floor is neither shown out nor cut short in a wait - as
        for Barnaby's lunch in his back room. Only leaving the building is closing up."""
        from unprepared.world.levels.town import TAILOR_TABLE_SEAT
        w = game.world
        keeper = at_closing(game, "Mirelle the Tailor", TAILOR_FLOOR, 13)
        assert w.find_area("the tailor's").is_open_to_public(13)
        assert game.wait()
        assert w.wait_interruption is None and w.clock.hour == 13
        assert not said(game.messages.messages, "shop_closing")
        assert w.area_of(game.player.position) == "the tailor's" and not game.in_transition
        assert keeper.current_task.target_pos == game.world.spot(*TAILOR_TABLE_SEAT)      # she did go to lunch

    def test_closing_happens_without_a_wait_too(self, game):
        w = game.world
        at_closing(game, *SHOPS[0][:3])
        for _ in range(14 * 4):                                       # half an hour of ordinary frames (1 s each)
            game.update(1.0)
            if w.wait_interruption == "evicted":
                break
        assert said(game.messages.messages, "shop_closing") and game.in_transition


class TestSkulkingInTheShop:
    QUIET_HOUR = {"Shopkeeper Barnaby": 18, "Mirelle the Tailor": 17}      # the keeper at the counter, no other customers

    def in_front_of(self, game, keeper_name, spot, counter):
        w = game.world
        w.wait_until(self.QUIET_HOUR[keeper_name])
        keeper = w.find_npc(keeper_name)
        assert keeper.state is AIState.SHOPKEEPING and keeper.position.distance_to(game.world.spot(*counter)) < 0.5
        assert not [n for n in w.npcs if n is not keeper and w.building_at(n.position) is w.building_at(keeper.position)]
        game.player.position = game.world.spot(*spot)
        keeper.turn_towards(game.player.position)
        game.toggle_crouch()
        game.rng.random = lambda: 0.99                                # every sneak roll fails
        return keeper

    def creep(self, game, seconds):
        for _ in range(int(seconds / 0.1)):
            game.update(0.1, MoveIntent(left=True, right=True))       # shifting about, crouched

    @pytest.mark.parametrize("keeper_name, spot, counter", [(s[0], s[1], s[4]) for s in SHOPS[:2]])
    def test_keeper_demands_you_stand_then_throws_you_out(self, game, keeper_name, spot, counter):
        w = game.world
        keeper = self.in_front_of(game, keeper_name, spot, counter)
        self.creep(game, 1.5)
        assert keeper.state is AIState.WAITING and keeper.waiting_for == "skulking"
        assert said(game.messages.messages, "shop_skulk") and not keeper.is_open_for_business
        interval = game.settings.justice.curfew_ask_interval
        self.creep(game, interval + 0.5)
        assert said(game.messages.messages, "shop_skulk_again") and keeper.state is AIState.WAITING
        assert game.player.gold == STARTING_GOLD and not keeper.is_alarmed     # no alarm, no guards: the keeper handles it
        self.creep(game, interval + 0.5)
        assert said(game.messages.messages, "shop_out") and "You are thrown out of the shop." in game.messages.messages
        assert game.frozen and game.in_transition
        settle(game)
        assert w.player_on_premises_of(keeper) is None
        assert keeper.state is not AIState.WAITING                    # back to business

    def test_standing_up_satisfies_the_keeper(self, game):
        keeper = self.in_front_of(game, SHOPS[0][0], SHOPS[0][1], SHOPS[0][4])
        self.creep(game, 1.5)
        assert keeper.state is AIState.WAITING
        game.toggle_crouch()
        for _ in range(3):
            game.update(0.1)
        assert keeper.state is AIState.SHOPKEEPING and keeper.is_open_for_business

    def test_leaving_the_shop_satisfies_the_keeper(self, game):
        keeper = self.in_front_of(game, SHOPS[0][0], SHOPS[0][1], SHOPS[0][4])
        self.creep(game, 1.5)
        assert keeper.state is AIState.WAITING
        game.player.position = spot(game.world, -13.5, -3.0)               # out on the street, still crouched
        for _ in range(3):
            game.update(0.1)
        assert keeper.state is AIState.SHOPKEEPING

    def test_a_successful_sneak_keeps_the_keeper_in_the_dark(self, game):
        keeper = self.in_front_of(game, SHOPS[0][0], SHOPS[0][1], SHOPS[0][4])
        game.rng.random = lambda: 0.0                                 # every roll succeeds
        self.creep(game, 4.0)
        assert keeper.state is AIState.SHOPKEEPING and not said(game.messages.messages, "shop_skulk")

    def test_the_smiths_open_yard_gets_the_ordinary_warning(self, game):
        from unprepared.world.levels.town import ANVIL_SPOT
        w = game.world
        gareth = w.find_npc("Gareth the Smith")
        gareth.path, gareth.current_task, gareth.schedule = [], None, []
        gareth.state = AIState.WORKING
        gareth.position = game.world.spot(*ANVIL_SPOT)
        game.player.position = spot(game.world, ANVIL_SPOT[0], ANVIL_SPOT[2] + 2.5)
        gareth.turn_towards(game.player.position)
        game.toggle_crouch()
        game.rng.random = lambda: 0.99
        self.creep(game, 1.5)
        assert w.player_on_premises_of(gareth) is None                # no walls: nothing to throw you out of
        assert gareth.state is AIState.WORKING and said(game.messages.messages, "skulking_citizen")


def test_a_wait_is_cut_short_when_a_guard_lays_hands_on_you(game):
    w = game.world
    vance = w.find_npc("Guard Captain Vance")
    vance.path, vance.current_task, vance.schedule = [], None, []
    vance.remember_crime("Pickpocketing", w.day_count, True, 25.0)
    game.player.position = spot(w, vance.position.x, vance.position.z + 1.5)
    vance.turn_towards(game.player.position)
    assert game.wait()
    assert w.wait_interruption == "caught" and w.clock.minute_accumulator < 5.0
    assert game.player.gold == STARTING_GOLD - 25.0 and "Your wait is interrupted." in game.messages.messages
