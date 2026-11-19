"""Every item belongs to somebody; ownership only changes hands legitimately."""
from __future__ import annotations


from unprepared.entities import AIState, Item
from unprepared.game import GameMode
from unprepared.systems import Side, TransferOutcome
from unprepared.world.levels.town import SHOP_COUNTER


def test_everything_in_the_town_has_an_owner(game):
    w = game.world
    for npc in w.npcs:
        assert all(i.owner == npc.name for i in npc.inventory), npc.name
    for container in w.containers:
        assert all(i.owner == container.owner for i in container.inventory), container.name
    for item in game.player.inventory:
        assert item.owner == Item.PLAYER_OWNER or item.is_stolen
    for name, stock in w.pantries.items():
        assert all(i.owner == w.find_container(name).owner for i in stock)


def test_restocks_and_resets_keep_the_owner(game):
    from unprepared.systems import commerce
    w = game.world
    barnaby = w.find_npc("Shopkeeper Barnaby")
    barnaby.inventory = [i for i in barnaby.inventory if i.id != "apple_1"]
    west = w.find_container("shop_shelf_west")
    west.inventory.clear()
    w.day_count = 10
    added = commerce.restock(barnaby, w, player_level=1)
    assert added and all(i.owner == "Shopkeeper Barnaby" for i in added)
    assert all(i.owner == "Shopkeeper Barnaby" for i in west.inventory)
    vance = w.find_npc("Guard Captain Vance")
    vance.inventory.clear()
    assert all(i.owner == "Guard Captain Vance" for i in vance.reset_inventory(10))
    w.pantries["barnaby_table"][0].quantity = 8
    w.find_container("barnaby_table").inventory.clear()
    w._restock_pantries()
    assert all(i.owner == "Shopkeeper Barnaby" for i in w.find_container("barnaby_table").inventory)


def open_shop(game):
    from tests.unit.test_game import open_shop as _open
    return _open(game)


class TestChangesOfHands:
    def test_buying_makes_it_yours_and_selling_makes_it_theirs(self, game):
        from tests.unit.test_game import teleport
        open_shop(game)
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        teleport(game, SHOP_COUNTER[0], -8.5, yaw=0.0)
        game.interact()
        assert game.mode is GameMode.TRADE
        bought = game.transfer_confirm()
        assert bought.outcome is TransferOutcome.MOVED and game.player.inventory[-1].owner == Item.PLAYER_OWNER
        game.transfer_switch_side()
        rows = game.transfer.rows(Side.RIGHT)
        game.transfer.selection[Side.RIGHT] = next(i for i, it in enumerate(rows) if it.id == "soft_shoes")
        game.player.unequip(game.transfer.selected_item())       # (worn: selling them would ask first)
        sold = game.transfer_confirm()
        assert sold.outcome is TransferOutcome.MOVED
        shoes = next(i for i in barnaby.inventory if i.id == "soft_shoes")
        assert shoes.owner == "Shopkeeper Barnaby" and not shoes.is_stolen

    def test_a_fence_launders_what_it_buys(self, game):
        from tests.unit.test_game import teleport
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SHOPKEEPING
        goblet = next(i for i in game.player.inventory if i.id == "silver_goblet")
        goblet.owner = "Mirelle the Tailor"
        teleport(game, jae.position.x, jae.position.z + 2.0, yaw=0.0)
        jae.turn_towards(game.player.position)
        game.interact()
        assert game.mode is GameMode.TRADE
        game.transfer_switch_side()
        rows = game.transfer.rows(Side.RIGHT)
        game.transfer.selection[Side.RIGHT] = next(i for i, it in enumerate(rows) if it.id == "silver_goblet")
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED
        assert goblet in jae.inventory and goblet.owner == "Jae the Fence" and not goblet.is_stolen

    def test_theft_keeps_the_owner_and_returns_go_home(self, game):
        w = game.world
        west = w.find_container("shop_shelf_west")
        candle = next(i for i in west.inventory if i.id == "candlestick")
        game.open_container(west)
        game.transfer.selection[Side.LEFT] = west.inventory.index(candle)
        game.transfer_confirm()
        game.close_transfer()
        assert candle in game.player.inventory and candle.is_stolen and candle.owner == "Shopkeeper Barnaby"
        assert w.owner_inventory_for(candle) is west.inventory
        key = Item("guard_key", "Guard House Key", is_key=True, key_door_id="guardhouse_door", is_stolen=True, owner="Guard Orrin")
        assert w.owner_inventory_for(key) is w.find_npc("Guard Orrin").inventory
        ration = Item("ration", "Guard's Ration", is_stolen=True, owner="guards", stackable=True)
        assert any(w.owner_inventory_for(ration) is w.find_container(table).inventory        # (one of the watch's tables)
                   for table in ("guard_table", "gate_lodge_table"))
        assert w.owner_inventory_for(Item("x", "Unknown", is_stolen=True)) is w.find_container("jail_storage").inventory

    def test_planting_gives_clean_goods_away_but_stolen_ones_stay_stolen(self, game):
        from unprepared.systems import PickpocketPolicy, TransferSession
        from unprepared.config import SkillRewards
        fenwick = game.world.find_npc("Fenwick")
        mine = Item("pipe", "Clay Pipe", owner=Item.PLAYER_OWNER)
        hot = Item("plate", "Pewter Plate", is_stolen=True, owner="Shopkeeper Barnaby")
        game.player.inventory += [mine, hot]
        policy = PickpocketPolicy(fenwick, game.player, SkillRewards())
        s = TransferSession(fenwick.inventory, game.player.inventory, policy)
        s.switch_side()
        s.selection[Side.RIGHT] = game.player.inventory.index(mine)
        s.confirm()
        s.selection[Side.RIGHT] = game.player.inventory.index(hot)
        s.confirm()
        assert mine.owner == "Fenwick" and not mine.is_stolen
        assert hot.owner == "Shopkeeper Barnaby" and hot.is_stolen
        jae = game.world.find_npc("Jae the Fence")
        game.player.inventory.append(hot2 := Item("plate", "Pewter Plate", is_stolen=True, owner="Shopkeeper Barnaby"))
        s = TransferSession(jae.inventory, game.player.inventory, PickpocketPolicy(jae, game.player, SkillRewards()))
        s.switch_side()
        s.selection[Side.RIGHT] = game.player.inventory.index(hot2)
        s.confirm()
        assert hot2.owner == "Jae the Fence" and not hot2.is_stolen           # slipped to a fence: laundered

    def test_a_meal_on_the_house_is_yours(self, game):
        w = game.world
        w.inn.rent(w, game.player, 1)
        table = w.find_container("inn_table_3")
        game.open_container(table)
        r = game.transfer_confirm()
        assert r.outcome is TransferOutcome.MOVED
        bread = game.player.inventory[-1]
        assert bread.id == "bread" and bread.owner == Item.PLAYER_OWNER and not bread.is_stolen
        assert table.inventory[0].owner == "Hilde the Innkeeper"                 # the rest of the loaf stays hers
