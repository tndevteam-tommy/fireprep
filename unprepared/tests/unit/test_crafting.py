"""Stations: the ore deposit (needs a pickaxe) and the anvil (ore -> lockpicks, skill-scaled)."""
import random

import pytest

from unprepared.entities import Item, SkillType
from unprepared.systems import station_prompt, use_station
from unprepared.systems.crafting import smithing_chance, smithing_yield


def give_ore(player, n=1, mallet=True):
    player.add_item(Item("iron_ore", "Iron Ore", value=6.0, stackable=True, quantity=n))
    if mallet:
        player.add_item(Item("mallet", "Mallet", value=12.0, category="tool"))


def set_level(player, skill, level):
    player.skills.skills[skill]["level"] = level


def test_anvil_needs_ore(world, settings):
    player = world.spawn_player()
    anvil = world.find_station("anvil")
    assert station_prompt(anvil, player) == "[Anvil: needs 1 Iron Ore]"
    assert use_station(player, anvil, settings, world.log.add) is None
    assert world.log.messages[-1] == "You need 1 Iron Ore to use the anvil. Mine some with a pickaxe."
    assert player.inventory[0].quantity == 1


def test_forging_needs_a_mallet_as_well(world, settings):
    player = world.spawn_player()
    anvil = world.find_station("anvil")
    give_ore(player, 1, mallet=False)
    assert station_prompt(anvil, player, settings) == "[Anvil: needs a mallet]"
    assert use_station(player, anvil, settings, world.log.add) is None
    assert world.log.messages[-1] == "You need a mallet to work the anvil."
    assert [i.quantity for i in player.inventory if i.id == "iron_ore"] == [1]      # the ore is not wasted
    player.add_item(Item("mallet", "Mallet", value=12.0, category="tool"))
    assert station_prompt(anvil, player, settings).startswith("[E] Forge")


def test_the_smith_and_the_merchants_can_supply_a_mallet(world):
    smith = world.find_npc("Gareth the Smith")
    assert {i.id for i in smith.inventory} >= {"mallet", "pickaxe", "hammer"}
    assert not any(i.for_sale for i in smith.inventory if i.id in ("mallet", "pickaxe"))   # his own tools
    for_sale = {i.id for c in ("smithy_rack", "shop_shelf_east") for i in world.find_container(c).inventory}
    assert "mallet" in for_sale


def test_prompt_shows_the_yield_for_the_current_skill(world, settings):
    player = world.spawn_player()
    anvil = world.find_station("anvil")
    give_ore(player, 1)
    assert station_prompt(anvil, player, settings) == "[E] Forge Iron Lockpicks (1 Iron Ore; up to 1 at your skill)"
    set_level(player, SkillType.SMITHING, 50)
    assert station_prompt(anvil, player, settings) == "[E] Forge Iron Lockpicks (1 Iron Ore; up to 15 at your skill)"


def test_yield_and_odds_scale_with_smithing():
    from unprepared.config import GameSettings
    from unprepared.core.math3d import Vector3D
    from unprepared.entities import Player
    s = GameSettings()
    p = Player(Vector3D(0, 1, 0), inventory=[])
    yields, chances = [], []
    for level in (1, 8, 15, 25, 50, 70):
        set_level(p, SkillType.SMITHING, level)
        yields.append(smithing_yield(p, s))
        chances.append(smithing_chance(p, s))
    from unprepared.systems.progression import chance
    assert yields == [1, 3, 5, 7, 15, 15]
    assert chances[0] == pytest.approx(chance(1, s.stealth.forge_challenge))      # about a coin toss for a novice
    assert all(a < b for a, b in zip(chances, chances[1:]))
    assert chances[-2] > 0.99 and chances[-1] <= s.progression.chance_ceiling


def test_novice_often_ruins_the_ore_but_still_learns(world, settings):
    player = world.spawn_player()
    anvil = world.find_station("anvil")
    give_ore(player, 1)
    rng = random.Random(0)
    rng.random = lambda: 0.99                    # a roll the novice cannot pass
    assert use_station(player, anvil, settings, world.log.add, rng=rng) is None
    assert not any(i.id == "iron_ore" for i in player.inventory)        # the ore is gone
    assert anvil.uses == 1
    from unprepared.systems.progression import challenge_factor
    assert player.skills.skills[SkillType.SMITHING]["xp"] == pytest.approx(
        settings.skill_rewards.smithing_fail * challenge_factor(1, settings.stealth.forge_challenge))
    assert "the ore is ruined" in world.log.messages[-1]


def test_successful_forging_yields_picks_that_stack_with_the_players_own(world, settings):
    player = world.spawn_player()
    anvil = world.find_station("anvil")
    give_ore(player, 2)
    rng = random.Random(0)
    rng.random = lambda: 0.0                     # always succeeds
    set_level(player, SkillType.SMITHING, 8)     # -> 3 picks per ore
    batch = use_station(player, anvil, settings, world.log.add, rng=rng)
    assert batch.id == "iron_lockpick" and batch.owner == Item.PLAYER_OWNER
    picks = [i for i in player.inventory if i.is_lockpick]
    assert [(i.id, i.quantity) for i in picks] == [("iron_lockpick", 4)]          # 1 starting pick + 3 forged
    assert [i.quantity for i in player.inventory if i.id == "iron_ore"] == [1]
    assert world.log.messages[-1] == "Forged 3 Iron Lockpicks from 1 iron ore!"
    xp = player.skills.skills[SkillType.SMITHING]["xp"]
    assert 0 < xp < settings.skill_rewards.smithing * 1.2         # scaled down: the forge is easy for level 8
    use_station(player, anvil, settings, rng=rng)
    assert not any(i.id == "iron_ore" for i in player.inventory) and anvil.uses == 2


def test_mining_needs_a_pickaxe(world, settings):
    player = world.spawn_player()
    node = world.find_station("ore_deposit")
    assert station_prompt(node, player) == "[Ore Deposit: needs a pickaxe]"
    assert use_station(player, node, settings, world.log.add) is None
    assert world.log.messages[-1] == "You need a pickaxe to work the ore deposit."
    player.add_item(Item("pickaxe", "Pickaxe", value=35.0))
    assert station_prompt(node, player) == "[E] Mine Iron Ore from the Ore Deposit"
    unit = use_station(player, node, settings, world.log.add)
    assert unit.id == "iron_ore" and unit.owner == Item.PLAYER_OWNER and node.uses == 1
    use_station(player, node, settings)
    assert [i.quantity for i in player.inventory if i.id == "iron_ore"] == [2]
    from unprepared.systems.progression import challenge_factor
    assert player.skills.skills[SkillType.SMITHING]["xp"] == pytest.approx(
        2 * settings.skill_rewards.mining * challenge_factor(1, settings.stealth.mining_challenge))
