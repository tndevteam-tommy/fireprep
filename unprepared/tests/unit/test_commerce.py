"""Shopkeeping: composite stock, stock-taking, theft reports, restocks and confrontations."""
import pytest

from tests.helpers import spot
from unprepared.config import GameSettings
from unprepared.entities import AIState, Item, SkillType
from unprepared.game import Game
from unprepared.systems import commerce
from unprepared.world.levels.town import SHOP_COUNTER, SHOP_OPEN_HOUR
from unprepared.world.levels.town import STARTING_GOLD


def run_until(game, hour, extra_seconds=0.0, dt=0.25):
    for _ in range(200000):
        if game.world.simulated_hour == hour:
            break
        game.update(dt)
    for _ in range(int(extra_seconds / dt)):
        game.update(dt)


@pytest.fixture
def game(tmp_path):
    return Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=3, max_log_messages=40),
                save_file=str(tmp_path / "s.json"))


def barnaby(game):
    return game.world.find_npc("Shopkeeper Barnaby")


class TestMerchantStock:
    def test_union_of_pockets_and_shelves(self, game):
        stock = game.world.merchant_stock(barnaby(game))
        ids = [i.id for i in stock]
        assert ids[:5] == ["shop_key", "home_key", "bread", "apple_1", "torch_item"]
        assert "candlestick" in ids and "pickaxe" in ids and len(stock) == len(ids)
        assert game.world.containers[0].inventory[0] in stock

    def test_remove_and_append_hit_the_right_source(self, game):
        b = barnaby(game)
        stock = game.world.merchant_stock(b)
        candle = game.world.find_container("shop_shelf_west").inventory[0]
        stock.remove(candle)
        assert candle not in game.world.find_container("shop_shelf_west").inventory
        stock.append(Item("x", "X"))
        assert b.inventory[-1].id == "x"
        with pytest.raises(ValueError):
            stock.remove(candle)

    def test_count_stock_counts_for_sale_units(self, game):
        counts = commerce.count_stock(barnaby(game), game.world)
        assert counts["bread"] == 4 and counts["wine"] == 3 and "shop_key" not in counts


class TestStockTaking:
    def test_theft_is_reported_when_the_shop_opens(self, game):
        west = game.world.find_container("shop_shelf_west")
        west.inventory[0].quantity = 0
        stolen = west.inventory.pop(0)                         # the candlestick vanished overnight
        wine = west.inventory[0]
        wine.quantity -= 2
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)     # Barnaby opens up and counts
        reports = game.world.theft_reports
        assert len(reports) == 1
        report = reports[0]
        assert report.reporter == "Shopkeeper Barnaby" and report.area == "behind the counter"
        assert report.items == {stolen.id: 1, "wine": 2}
        assert any("Thieves!" in m and "Silver Candlestick x1" in m and "Bottle of Wine x2" in m for m in game.messages.messages)
        assert "behind the counter" in next(m for m in game.messages.messages if "Thieves!" in m)
        for guard in (n for n in game.world.npcs if n.is_guard):
            assert guard.known_thefts == [report]
        # counted once: no second report when he closes
        run_until(game, 20, extra_seconds=20)
        assert len(game.world.theft_reports) == 1

    def test_legitimate_sales_are_not_thefts(self, game):
        b = barnaby(game)
        b.position = game.world.spot(*SHOP_COUNTER)
        b.state = AIState.SHOPKEEPING
        game.open_trade(b)
        game.transfer_confirm()                                # buy bread
        game.close_transfer()
        assert b.expected_stock["bread"] == 3
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)
        assert game.world.theft_reports == []

    def test_nothing_missing_means_no_report(self, game):
        """Barnaby eating his own bread at lunch is not a theft either."""
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)
        run_until(game, 20, extra_seconds=20)
        assert game.world.theft_reports == []
        assert barnaby(game).inventory[2].quantity == 3                 # one loaf eaten at lunch


class TestRestock:
    def test_restock_due_every_three_days(self, game):
        b = barnaby(game)
        assert not commerce.restock_due(b, game.world)
        game.world.day_count = 3
        assert commerce.restock_due(b, game.world)

    def test_restock_tops_up_base_stock_only_at_level_one(self, game):
        b = barnaby(game)
        west = game.world.find_container("shop_shelf_west")
        west.inventory.pop(0)                                  # candlestick gone
        b.inventory[2].quantity = 1                            # bread nearly sold out
        game.world.day_count = 3
        added = commerce.restock(b, game.world, player_level=1)
        names = sorted(i.name for i in added)
        assert names == ["Fresh Bread", "Silver Candlestick"]
        assert b.inventory[2].quantity == 4 and any(i.id == "candlestick" for i in west.inventory)
        assert not any(i.id == "fine_lockpick" for i in game.world.merchant_stock(b))
        assert b.last_restock_day == 3 and b.expected_stock["bread"] == 4

    def test_higher_player_level_unlocks_better_pricier_goods(self, game):
        b = barnaby(game)
        game.world.day_count = 3
        added = commerce.restock(b, game.world, player_level=4)
        ids = {i.id for i in added}
        assert {"fine_lockpick", "jeweled_dagger"} <= ids and "ruby_necklace" not in ids
        assert "ruby_necklace" not in ids                     # level 6 tier
        east = game.world.find_container("shop_shelf_east")
        assert any(i.id == "fine_lockpick" for i in east.inventory)
        dagger = next(i for i in game.world.merchant_stock(b) if i.id == "jeweled_dagger")
        assert dagger.value > 100 and dagger.for_sale

    def test_restock_happens_on_opening_day(self, game):
        b = barnaby(game)
        game.world.day_count = 3
        game.player.skills.skills[SkillType.SNEAK]["level"] = 3      # player level 2
        assert game.player.level == 2
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)
        assert b.last_restock_day == 3
        assert any(i.id == "fine_lockpick" for i in game.world.merchant_stock(b))
        assert any("Fresh stock in!" in m for m in game.messages.messages)


class TestPlayerLevel:
    def test_level_from_skills(self, game):
        p = game.player
        assert p.level == 1
        p.skills.skills[SkillType.SNEAK]["level"] = 2
        assert p.level == 1
        p.skills.skills[SkillType.LOCKPICKING]["level"] = 2
        assert p.level == 2
        p.skills.skills[SkillType.PICKPOCKET]["level"] = 5
        assert p.level == 4


class TestConfrontation:
    def steal_candlestick(self, game):
        west = game.world.find_container("shop_shelf_west")
        candle = west.inventory.pop(0)
        candle.is_stolen = True
        game.player.add_item(candle)
        return candle

    def test_guard_who_saw_you_in_the_shop_confronts_you(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        vance.remember_sighting("behind the counter", game.world.day_count)   # he saw you skulking behind the counter
        candle = self.steal_candlestick(game)
        game.player.position = spot(game.world, -12.0, 12.0)             # lying low in the hideout meanwhile
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)             # theft reported at opening
        assert vance.pending_confrontations and vance.pending_confrontations[0].items == {"candlestick": 1}
        # stand in front of him, in plain view (on the open street: wherever his round had taken him)
        vance.path = []
        vance.position = spot(game.world, 5.0, -2.0)
        game.player.position = spot(game.world, vance.position.x, vance.position.z + 3.0)
        vance.turn_towards(game.player.position)
        game.update(0.05)
        assert vance.pending_confrontations == []
        assert candle not in game.player.inventory
        assert candle in game.world.find_container("shop_shelf_west").inventory     # back on Barnaby's shelf
        assert candle.owner == "Shopkeeper Barnaby" and not candle.is_stolen
        assert game.player.crimes_witnessed == 1
        assert any("Hand it over!" in m for m in game.messages.messages)
        # ... and the theft is settled on the spot: stealing inside a shop is 500g and five days - more
        # than a 50g purse can cover, so it is seven - plus two for the stolen goblet still on us
        assert vance.crime_memories == [] and game.player.gold == STARTING_GOLD
        assert game.world.jail.serving and game.world.jail.days_left(game.world.day_count) == 7 + 2

    def test_confrontation_takes_back_only_the_stolen_goods(self, game):
        """A bottle bought and paid for is the player's whatever it is called; the guard recovers
        only the reporter's stolen units, and no more of them than went missing."""
        from unprepared.systems.transfer import move_one
        w, p = game.world, game.player
        barnaby = w.find_npc("Shopkeeper Barnaby")
        shelf = w.find_container("shop_shelf_west")
        wine = next(i for i in shelf.inventory if i.id == "wine")             # a stack of 3 on the shelf
        bought = move_one(shelf.inventory, p.inventory, wine, lambda u: u.transfer_to(Item.PLAYER_OWNER))
        commerce.note_sale(barnaby, wine, -1)                                  # an honest sale
        move_one(shelf.inventory, p.inventory, wine, Item.steal)                # and one bottle stolen
        assert sorted((i.quantity, i.owner, i.is_stolen) for i in p.inventory if i.id == "wine") == \
            [(1, "Shopkeeper Barnaby", True), (1, Item.PLAYER_OWNER, False)]
        report = commerce.take_stock(barnaby, w)
        assert report.items == {"wine": 1}
        vance = w.find_npc("Guard Captain Vance")
        vance.remember_sighting(report.area, w.day_count)
        vance.learn_theft(report)
        w._confront(vance, p)
        mine = [i for i in p.inventory if i.id == "wine"]
        assert len(mine) == 1 and mine[0] is bought and mine[0].is_players and not mine[0].is_stolen
        assert sum(i.quantity for i in shelf.inventory if i.id == "wine") == 2   # the stolen one is back, the sold one is not
        lockbox = w.find_container("jail_storage")
        assert not any(i.id == "wine" for i in lockbox.inventory)               # nothing of the player's confiscated

    def test_confrontation_takes_no_more_than_went_missing(self, game):
        """Two bottles stolen from the shelf, one reported missing (one was eaten meanwhile, say):
        one is recovered; stolen goods of somebody else's are not the reporter's to claim."""
        from unprepared.systems.transfer import move_one
        w, p = game.world, game.player
        barnaby = w.find_npc("Shopkeeper Barnaby")
        shelf = w.find_container("shop_shelf_west")
        wine = next(i for i in shelf.inventory if i.id == "wine")
        move_one(shelf.inventory, p.inventory, wine, Item.steal)
        move_one(shelf.inventory, p.inventory, wine, Item.steal)
        others = Item("wine", "Bottle of Wine", value=12.0, stackable=True, is_stolen=True, owner="Hilde the Innkeeper")
        p.add_item(others)
        report = commerce.take_stock(barnaby, w)
        report.items["wine"] = 1                                                # only one is on the report
        vance = w.find_npc("Guard Captain Vance")
        vance.remember_sighting(report.area, w.day_count)
        vance.learn_theft(report)
        w._confront(vance, p)
        left = [(i.quantity, i.owner) for i in p.inventory if i.id == "wine"]
        assert sorted(left) == [(1, "Hilde the Innkeeper"), (1, "Shopkeeper Barnaby")]
        assert sum(i.quantity for i in shelf.inventory if i.id == "wine") == 2

    def test_guard_without_a_sighting_does_not_confront(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        candle = self.steal_candlestick(game)
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)
        assert vance.known_thefts and not vance.pending_confrontations
        vance.path = []
        game.player.position = spot(game.world, vance.position.x, vance.position.z + 3.0)
        vance.turn_towards(game.player.position)
        game.update(0.05)
        assert candle in game.player.inventory

    def test_warning_when_the_goods_are_already_gone(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        vance.remember_sighting("behind the counter", 0)
        self.steal_candlestick(game)
        game.player.position = spot(game.world, -12.0, 12.0)
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)
        game.player.inventory = [i for i in game.player.inventory if i.id != "candlestick"]   # fenced already
        vance.path = []
        vance.position = spot(game.world, 5.0, -2.0)
        game.player.position = spot(game.world, vance.position.x, vance.position.z + 3.0)
        vance.turn_towards(game.player.position)
        game.update(0.05)
        assert any("I've got my eye on you" in m for m in game.messages.messages)
        assert vance.state is not AIState.ALERT_GUARDS

    def test_suspicious_behaviour_in_view_of_a_guard_is_remembered(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        vance.path = []
        vance.current_task = None
        vance.schedule = []
        vance.suspicion_meter = 60.0
        game.player.position = spot(game.world, -13.0, -8.0)            # on the shop floor (open hours not yet: 07:00)
        vance.position = spot(game.world, -13.0, -9.5)
        vance.turn_towards(game.player.position)
        game.toggle_crouch()
        game.update(0.05)
        assert any(s.area == "the shop" for s in vance.sightings)
