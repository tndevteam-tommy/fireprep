import pytest

from tests.conftest import said
from unprepared.config import EconomySettings, SkillRewards
from unprepared.core.math3d import BoundingBox3D, Vector3D
from unprepared.entities import AIState, Item, NPC, Player, SkillType
from unprepared.systems import ContainerPolicy, PickpocketPolicy, Side, TradePolicy, TransferOutcome, TransferSession
from unprepared.world import Container


@pytest.fixture
def player():
    return Player(Vector3D(0, 0, 0), gold=50.0, inventory=[
        Item("iron_lockpick", "Iron Lockpick", value=5.0, is_lockpick=True, stackable=True, quantity=2),
        Item("stolen_goblet", "Stolen Silver Goblet", value=120.0, is_stolen=True),
    ])


@pytest.fixture
def merchant():
    npc = NPC("Barnaby", False, False, True, Vector3D(0, 0, 0))
    npc.state = AIState.SHOPKEEPING
    npc.gold = 400.0
    npc.inventory = [Item("key", "Shop Key", value=50.0, is_key=True, key_door_id="shop", for_sale=False),
                     Item("bread", "Fresh Bread", value=15.0, stackable=True, quantity=3), Item("gem", "Gem", value=500.0)]
    return npc


class ScriptedRolls:
    """A stand-in RNG for the policies: ``random()`` answers from ``values`` in turn (cycling).
    0.0 passes any roll; 0.999 fails any (every chance lies within 0.5% .. 99.5%)."""

    def __init__(self, *values):
        self.values, self.i = list(values), 0

    def random(self):
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v


PASS, FAIL = 0.0, 0.999


def total_xp(player, skill):
    """XP earned in ``skill`` so far, including what went into levels already gained."""
    from unprepared.entities import SkillTree
    data = player.skills.skills[skill]
    return data["xp"] + sum(SkillTree.xp_needed_for(level) for level in range(1, int(data["level"])))


def factor(level, challenge):
    from unprepared.systems.progression import challenge_factor
    return challenge_factor(level, challenge)


def trade(merchant, player):
    msgs = []
    policy = TradePolicy(merchant, player, EconomySettings(), notify=msgs.append)
    return TransferSession(merchant.inventory, player.inventory, policy), msgs


class TestSessionNavigation:
    def test_starts_on_the_left(self, merchant, player):
        s, _ = trade(merchant, player)
        assert s.side is Side.LEFT and s.selected_item().id == "bread"
        assert [i.id for i in s.rows(Side.LEFT)] == ["bread", "gem"]          # key hidden
        assert [i.id for i in s.rows(Side.RIGHT)] == ["iron_lockpick", "stolen_goblet"]

    def test_tab_switches_sides_and_keeps_cursors(self, merchant, player):
        s, _ = trade(merchant, player)
        s.move_selection(1)
        assert s.switch_side() is Side.RIGHT and s.selected_item().id == "iron_lockpick"
        s.move_selection(1)
        assert s.selected_item().id == "stolen_goblet"
        assert s.switch_side() is Side.LEFT and s.selected_item().id == "gem"

    def test_wraps_within_a_side(self, merchant, player):
        s, _ = trade(merchant, player)
        assert s.move_selection(-1) == 1
        assert s.move_selection(1) == 0

    def test_empty_side(self, merchant):
        p = Player(Vector3D(0, 0, 0))
        s, _ = trade(merchant, p)
        s.switch_side()
        assert s.selected_item() is None and s.move_selection(1) == 0
        assert s.confirm().outcome is TransferOutcome.NOTHING

    def test_row_text(self, merchant, player):
        s, _ = trade(merchant, player)
        # the merchant asks worth + 25% and offers a quarter of the worth (half that for hot goods); the worth is always shown
        assert s.row_text(merchant.inventory[1], Side.LEFT) == "Fresh Bread (3) - 18.8g (worth 15g)"
        assert s.row_text(player.inventory[1], Side.RIGHT) == "Stolen Silver Goblet - 15g (worth 120g)"   # the (S) tag is drawn by the HUD
        assert s.row_text(player.inventory[0], Side.RIGHT) == "Iron Lockpick (2) - 1.2g (worth 5g)"
        assert s.policy.title == "TRADING WITH: Barnaby   (purse: 400g)"


class TestTrade:
    def test_buy_one_unit_of_a_stack(self, merchant, player):
        s, msgs = trade(merchant, player)
        r = s.confirm()
        assert r.outcome is TransferOutcome.MOVED and r.price == 18.8 and r.side is Side.LEFT   # 15g worth + 25% markup
        assert player.gold == pytest.approx(31.2)
        assert merchant.inventory[1].quantity == 2
        assert player.inventory[-1].id == "bread" and player.inventory[-1].quantity == 1
        s.confirm()
        assert player.inventory[-1].quantity == 2 and len(player.inventory) == 3      # merged into the stack
        assert msgs == ["Bought Fresh Bread for 18.8g.", "Bought Fresh Bread for 18.8g."]
        assert merchant.gold == pytest.approx(437.6)                                  # the till fills up
        assert player.inventory[-1].owner == Item.PLAYER_OWNER                        # bought and paid for: yours
        assert player.skills.skills[SkillType.BARTER]["xp"] == pytest.approx(2 * (5.0 + 9.4) * factor(1, 0.0))   # base + value XP (by price)

    def test_buy_last_unit_removes_the_row_and_clamps(self, merchant, player):
        merchant.inventory[1].quantity = 1
        s, _ = trade(merchant, player)
        s.move_selection(1)               # gem (too expensive) ... go back to bread
        s.move_selection(-1)
        s.confirm()
        assert [i.id for i in s.rows(Side.LEFT)] == ["gem"] and s.selected_item().id == "gem"

    def test_not_enough_gold(self, merchant, player):
        s, msgs = trade(merchant, player)
        s.move_selection(1)
        assert s.confirm().outcome is TransferOutcome.NOT_ENOUGH_GOLD
        assert player.gold == 50.0 and msgs == ["Not enough gold to buy this item!"]

    def test_sell(self, merchant, player):
        s, msgs = trade(merchant, player)
        s.switch_side()
        r = s.confirm()
        assert r.outcome is TransferOutcome.MOVED and r.price == 1.2 and r.side is Side.RIGHT   # a quarter of 5g
        assert player.gold == pytest.approx(51.2) and player.inventory[0].quantity == 1
        assert merchant.gold == pytest.approx(398.8)
        assert merchant.inventory[-1].id == "iron_lockpick" and merchant.inventory[-1].quantity == 1
        assert msgs == ["Sold Iron Lockpick for 1.2g."]
        assert player.skills.skills[SkillType.BARTER]["xp"] == pytest.approx((5.0 + 0.6) * factor(1, 0.0))

    def test_merchant_must_be_able_to_afford_it(self, merchant, player):
        merchant.gold = 1.0
        s, msgs = trade(merchant, player)
        s.switch_side()
        r = s.confirm()
        assert r.outcome is TransferOutcome.MERCHANT_BROKE and not r.close_session
        assert player.gold == 50.0 and player.inventory[0].quantity == 2 and merchant.gold == 1.0
        assert len(msgs) == 1 and msgs[0].startswith("(Barnaby): ") and said(msgs, "cant_afford", gold="1")

    def test_three_prices_worth_asking_and_offer(self, merchant, player):
        """The worth never changes; the asking price is worth + markup haggled down with Barter (never
        below the worth); the offer is a fraction of the worth growing with Barter (never above it)."""
        s, _ = trade(merchant, player)
        bread, gem, pick = merchant.inventory[1], merchant.inventory[2], player.inventory[0]
        assert s.policy.markup == pytest.approx(0.25) and s.policy.sell_fraction == pytest.approx(0.25)
        assert s.policy.buy_price(bread) == 18.8 and s.policy.buy_price(gem) == 625.0     # worth + 25%
        assert s.policy.sell_price(pick) == 1.2                                          # a quarter of the worth
        assert bread.value == 15.0 and gem.value == 500.0 and pick.value == 5.0           # the worth is untouched
        player.skills.skills[SkillType.BARTER]["level"] = 26                            # half the markup gone, offer 50%
        assert s.policy.markup == pytest.approx(0.125) and s.policy.sell_fraction == pytest.approx(0.5)
        assert s.policy.buy_price(gem) == 562.5 and s.policy.sell_price(pick) == 2.5
        assert s.row_text(gem, Side.LEFT) == "Gem - 562.5g (worth 500g)"
        player.skills.skills[SkillType.BARTER]["level"] = 51                            # the markup is gone ...
        assert s.policy.markup == 0.0 and s.policy.buy_price(gem) == 500.0
        assert s.row_text(gem, Side.LEFT) == "Gem - 500g"
        player.skills.skills[SkillType.BARTER]["level"] = 500                           # ... and the offer is capped at the worth
        assert s.policy.sell_fraction == 1.0 and s.policy.sell_price(pick) == 5.0 and s.policy.buy_price(gem) == 500.0
        player.skills.skills[SkillType.BARTER]["level"] = 1
        player.gold = 1000.0
        s.move_selection(1)                                             # the 500g gem at 625
        s.confirm()
        # 5 base + 250 (value XP is capped at 250), a trade being an even match at level 1: two levels (80 + 92)
        earned = 255.0 * factor(1, 0.0)
        assert player.skills.level(SkillType.BARTER) == 3
        assert player.skills.skills[SkillType.BARTER]["xp"] == pytest.approx(earned - 80.0 - 92.0)

    @pytest.mark.parametrize("level", [1, 10, 30, 51, 76, 200])
    def test_buying_and_selling_back_always_loses_money(self, merchant, player, level):
        """No Barter level turns a merchant into a money pump: the offer never exceeds the asking price."""
        s, _ = trade(merchant, player)
        player.skills.skills[SkillType.BARTER]["level"] = level
        for item in (merchant.inventory[1], merchant.inventory[2]):
            assert s.policy.sell_price(item) <= s.policy.buy_price(item)
            assert s.policy.sell_price(item) <= item.value <= s.policy.buy_price(item)

    def test_sell_stolen_to_merchant_is_refused_and_closes(self, merchant, player):
        s, msgs = trade(merchant, player)
        s.switch_side()
        s.move_selection(1)
        r = s.confirm()
        assert r.outcome is TransferOutcome.REFUSED_STOLEN and r.close_session
        assert merchant.state is AIState.REPORTING and player.gold == 50.0
        assert len(msgs) == 1 and said(msgs, "stolen_goods", item="Stolen Silver Goblet")

    def test_fence_buys_stolen_at_half_price(self, player):
        fence = NPC("Jae", False, True, True, Vector3D(0, 0, 0))
        s, msgs = trade(fence, player)
        s.switch_side()
        s.move_selection(1)
        fence.gold = 600.0
        r = s.confirm()
        # half the usual offer for hot goods: 120g worth -> a quarter is 30 -> 15 for stolen
        assert r.outcome is TransferOutcome.MOVED and r.price == 15.0 and player.gold == 65.0 and fence.gold == 585.0
        assert fence.inventory[0].id == "stolen_goblet" and fence.state is AIState.SLEEPING
        assert not fence.inventory[0].is_stolen            # laundered: no longer hot


class TestPickpocket:
    @pytest.fixture
    def victim(self):
        npc = NPC("Vance", True, False, False, Vector3D(0, 0, 0))
        npc.inventory = [Item("guard_key", "Guard Key", value=60.0, is_key=True, key_door_id="g", for_sale=False),
                         Item("ration", "Ration", value=3.0, stackable=True, quantity=2)]
        return npc

    def session(self, victim, player, chance=1.0, caught=None, unseen=0.0):
        """``chance`` 1/0: the victim never / always feels it; ``unseen`` 1/0: the bystanders never /
        always get to witness the dip. (Every lift rolls the bystanders first, then the victim.)"""
        msgs, crimes = [], []
        rolls = ScriptedRolls(PASS if unseen else FAIL, PASS if chance else FAIL)
        policy = PickpocketPolicy(victim, player, SkillRewards(), notify=msgs.append, report_crime=crimes.append,
                                  rng=rolls, on_caught=caught)
        return TransferSession(victim.inventory, player.inventory, policy), msgs, crimes

    @staticmethod
    def lift_xp(victim, item, level=1):
        """What lifting ``item`` from ``victim`` teaches: (20 base + value XP) x the challenge factor."""
        from unprepared.config import StealthSettings
        from unprepared.systems.progression import value_xp
        from unprepared.systems.stealth import pickpocket_challenge
        return (20.0 + value_xp(item.value)) * factor(level, pickpocket_challenge(victim, item, StealthSettings()))

    def test_keys_are_visible_and_stealable(self, victim, player):
        expected = self.lift_xp(victim, victim.inventory[0])
        s, msgs, crimes = self.session(victim, player)
        assert [i.id for i in s.rows(Side.LEFT)] == ["guard_key", "ration"]        # no purse: he is broke
        r = s.confirm()
        assert r.outcome is TransferOutcome.MOVED
        assert player.has_key_for("g") and player.inventory[-1].is_stolen
        assert crimes == ["Pickpocketing"]
        # (20 base + 30 value XP), a guard's vigilance less a sleeper's: the challenge sets the factor
        assert total_xp(player, SkillType.PICKPOCKET) == pytest.approx(expected)
        assert msgs == ["Pickpocketed Guard Key from Vance!"]

    def test_awake_victims_and_pricey_loot_teach_more_elites_learn_less(self, victim, player):
        asleep = self.lift_xp(victim, victim.inventory[0])
        victim.state = AIState.PATROL
        expected = self.lift_xp(victim, victim.inventory[0])
        s, _, _ = self.session(victim, player)
        s.confirm()
        awake = total_xp(player, SkillType.PICKPOCKET)
        assert awake == pytest.approx(expected) and awake > 2 * asleep       # an awake guard is a real test
        player.skills.skills[SkillType.PICKPOCKET] = {"level": 50, "xp": 0.0, "xp_needed": 9999.0}
        s.confirm()                                                              # the ration (3g)
        elite = player.skills.skills[SkillType.PICKPOCKET]["xp"]
        assert 0 < elite < awake / 10

    def test_coins_come_in_handfuls(self, victim, player):
        victim.gold = 35.0
        s, msgs, crimes = self.session(victim, player)
        rows = s.rows(Side.LEFT)
        assert rows[0].id == "gold_coins" and rows[0].quantity == 35
        r = s.confirm()
        assert r.outcome is TransferOutcome.MOVED and r.price == 20.0
        assert player.gold == 70.0 and victim.gold == 15.0
        s.confirm()
        assert player.gold == 85.0 and victim.gold == 0.0
        assert [i.id for i in s.rows(Side.LEFT)] == ["guard_key", "ration"]     # purse empty: row gone
        assert msgs[0] == "Lifted 20 gold from Vance's purse!" and crimes == ["Pickpocketing"] * 2

    def test_a_deft_dip_is_never_put_to_the_bystanders(self, victim, player):
        s, msgs, crimes = self.session(victim, player, unseen=1.0)
        assert s.confirm().outcome is TransferOutcome.MOVED
        assert s.confirm().outcome is TransferOutcome.MOVED
        assert crimes == [] and player.inventory[-1].is_stolen          # stolen all the same, but nobody asked the onlookers

    def test_bystander_roll_scales_with_the_pickpocket_skill(self, victim, player):
        from unprepared.config import StealthSettings
        from unprepared.systems import pickpocket_unseen_chance
        from unprepared.systems.progression import chance
        s = StealthSettings()
        novice = pickpocket_unseen_chance(player, s)
        assert novice == pytest.approx(chance(1, s.bystander_challenge))
        player.skills.skills[SkillType.PICKPOCKET]["level"] = 30
        assert 0.95 < pickpocket_unseen_chance(player, s) < 1.0

    def test_stolen_units_do_not_merge_with_clean_stacks(self, victim, player):
        player.inventory.append(Item("ration", "Ration", value=3.0, stackable=True, quantity=1))
        s, _, _ = self.session(victim, player)
        s.move_selection(1)
        s.confirm()
        rations = [i for i in player.inventory if i.id == "ration"]
        assert [(r.quantity, r.is_stolen) for r in rations] == [(1, False), (1, True)]
        assert victim.inventory[-1].quantity == 1

    def test_fumbled_pickpocket_is_caught(self, victim, player):
        expected = self.lift_xp(victim, victim.inventory[0]) * 0.3           # consolation XP
        caught = []
        s, msgs, crimes = self.session(victim, player, chance=0.0, caught=caught.append)
        r = s.confirm()
        assert r.outcome is TransferOutcome.CAUGHT and r.close_session
        assert caught == [victim] and crimes == ["Pickpocketing"]
        assert [i.id for i in victim.inventory] == ["guard_key", "ration"]      # nothing taken
        assert total_xp(player, SkillType.PICKPOCKET) == pytest.approx(expected)
        assert msgs == ["Fumbled! Vance catches your hand in their pocket."]

    def test_chance_scales_with_skill_and_victim(self, victim, player):
        from unprepared.systems import pickpocket_chance
        from unprepared.config import StealthSettings
        from unprepared.systems.progression import chance
        s = StealthSettings()
        guard = dict(s.vigilance)["guard"]
        victim.state = AIState.PATROL
        awake_guard = pickpocket_chance(player, victim, s)
        assert awake_guard == pytest.approx(chance(1, guard))
        victim.state = AIState.SLEEPING
        asleep_guard = pickpocket_chance(player, victim, s)
        assert asleep_guard == pytest.approx(chance(1, guard + s.asleep_challenge)) and asleep_guard > 3 * awake_guard
        legendary = Item("crown", "Crown", rarity="legendary")
        assert pickpocket_chance(player, victim, s, item=legendary) < 0.02            # a legendary piece, even from a sleeper
        player.skills.skills[SkillType.PICKPOCKET]["level"] = 3
        assert pickpocket_chance(player, victim, s) == pytest.approx(chance(3, guard + s.asleep_challenge))
        player.skills.skills[SkillType.PICKPOCKET]["level"] = 60
        victim.state = AIState.PATROL
        assert 0.99 < pickpocket_chance(player, victim, s) < 1.0        # an elite barely ever fumbles

    def test_plant_an_item(self, victim, player):
        s, msgs, crimes = self.session(victim, player)
        s.switch_side()
        s.move_selection(1)                          # the stolen goblet
        r = s.confirm()
        assert r.outcome is TransferOutcome.MOVED and r.side is Side.RIGHT
        assert victim.inventory[-1].id == "stolen_goblet" and crimes == []
        assert msgs == ["Planted Stolen Silver Goblet on Vance."]
        assert player.skills.skills[SkillType.PICKPOCKET]["xp"] == 0.0


class TestContainer:
    def make(self, owner):
        c = Container("shelf", "Shop Shelf", BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1)), owner=owner, crime_name="Shoplifting")
        c.inventory = [Item("wine", "Bottle of Wine", value=12.0, stackable=True, quantity=3), Item("plate", "Plate", value=18.0)]
        return c

    def session(self, container, player, unnoticed=False, witnesses=()):
        from unprepared.config import StealthSettings
        msgs, crimes = [], []

        def report(crime):
            crimes.append(crime)
            return list(witnesses)
        rng = ScriptedRolls(PASS if unnoticed else FAIL)
        policy = ContainerPolicy(container, player, notify=msgs.append, report_crime=report, stealth=StealthSettings(), rng=rng)
        return TransferSession(container.inventory, player.inventory, policy), msgs, crimes

    def test_taking_from_an_owned_container_is_theft(self, player):
        c = self.make("Barnaby")
        s, msgs, crimes = self.session(c, player)
        assert s.policy.is_theft and s.policy.title == "SHOP SHELF"
        s.confirm()
        assert crimes == ["Shoplifting"]
        assert player.inventory[-1].id == "wine" and player.inventory[-1].is_stolen and c.inventory[0].quantity == 2
        assert msgs == ["Stole Bottle of Wine from the shop shelf."]
        assert player.skills.skills[SkillType.STEALING]["xp"] == pytest.approx((10.0 + 6.0) * factor(1, 5.0))     # base + value XP

    def test_a_deft_thief_is_not_noticed(self, player):
        c = self.make("Barnaby")
        s, msgs, crimes = self.session(c, player, unnoticed=True)
        s.confirm()
        assert crimes == []                                           # nobody is told: sleight of hand
        assert player.inventory[-1].is_stolen                         # ... but the wine is still hot

    def test_getting_caught_still_teaches_a_little(self, player):
        c = self.make("Barnaby")
        s, msgs, crimes = self.session(c, player, witnesses=["Barnaby"])
        s.confirm()
        assert crimes == ["Shoplifting"]
        assert player.skills.skills[SkillType.STEALING]["xp"] == pytest.approx((10.0 + 6.0) * 0.4 * factor(1, 5.0))

    def test_taking_back_your_own_things_is_not_theft(self, player):
        c = self.make("guards")
        c.inventory.append(Item("boots", "Boots", value=30.0, owner=Item.PLAYER_OWNER))
        s, msgs, crimes = self.session(c, player)
        s.move_selection(2)
        s.confirm()
        assert crimes == [] and not player.inventory[-1].is_stolen and player.inventory[-1].owner == Item.PLAYER_OWNER
        assert msgs == ["Took back your Boots."]
        assert player.skills.skills[SkillType.STEALING]["xp"] == 0.0

    def test_unowned_container_is_free_loot(self, player):
        c = self.make(None)
        s, msgs, crimes = self.session(c, player)
        s.move_selection(1)
        s.confirm()
        assert crimes == [] and not player.inventory[-1].is_stolen and msgs == ["Took Plate."]

    def test_put_item_in(self, player):
        c = self.make("Barnaby")
        s, msgs, crimes = self.session(c, player)
        s.switch_side()
        s.confirm()
        assert c.inventory[-1].id == "iron_lockpick" and c.inventory[-1].quantity == 1
        assert player.inventory[0].quantity == 1 and crimes == []
        assert msgs == ["Put Iron Lockpick in the shop shelf."]

    def test_persistence(self):
        c = self.make("Barnaby")
        data = c.to_dict()
        other = self.make("Barnaby")
        other.inventory = []
        other.load_dict(data)
        assert [(i.id, i.quantity) for i in other.inventory] == [("wine", 3), ("plate", 1)]


class TestWornAndTheftTags:
    def test_worn_items_are_tagged_in_the_players_column(self, merchant, player):
        s, _ = trade(merchant, player)
        boots = Item("soft_shoes", "Soft Leather Boots", value=30.0, gear="boots", equipped=True)
        player.inventory.append(boots)
        assert s.row_tag(boots, Side.RIGHT) == ("worn", "worn")
        assert s.row_tag(player.inventory[0], Side.RIGHT) is None
        assert s.row_tag(merchant.inventory[1], Side.LEFT) is None
        s.switch_side()
        s.selection[Side.RIGHT] = player.inventory.index(boots)
        assert s.giving_away_worn() is boots
        s.selection[Side.RIGHT] = 0
        assert s.giving_away_worn() is None

    def test_container_rows_say_when_taking_is_theft(self, player):
        shelf = Container("shelf", "Shop Shelf", BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1)), owner="Barnaby")
        shelf.inventory = [Item("plate", "Pewter Plate", value=18.0, owner="Barnaby"),
                           Item("mine", "My Lockpick", value=5.0, owner=Item.PLAYER_OWNER),
                           Item("stew", "Stew", value=4.0, owner="Barnaby", is_food=True)]
        policy = ContainerPolicy(shelf, player, complimentary=lambda item: item.is_food)
        s = TransferSession(shelf.inventory, player.inventory, policy)
        assert s.row_tag(shelf.inventory[0], Side.LEFT) == ("THEFT", "theft")
        assert s.row_tag(shelf.inventory[1], Side.LEFT) is None                # yours: taking it back is no crime
        assert s.row_tag(shelf.inventory[2], Side.LEFT) == ("free", "free")    # on the house
        chest = Container("chest", "Chest", BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1)))   # nobody's
        s2 = TransferSession(chest.inventory, player.inventory, ContainerPolicy(chest, player))
        chest.inventory.append(Item("coin", "Coin"))
        assert s2.row_tag(chest.inventory[0], Side.LEFT) is None


class TestRareAndSecuredThings:
    """Epic and legendary pieces are chained, mounted or warded: working one free is a Stealing check
    of its own on the common curve - hopeless (not even attempted) far below its challenge."""

    def vault(self, rarity):
        c = Container("vault", "Strongbox", BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1)), owner="Lord Vane", crime_name="Theft")
        c.inventory = [Item("relic", "Relic", value=900.0, rarity=rarity)]
        return c

    def session(self, container, player, *rolls):
        from unprepared.config import StealthSettings
        msgs, crimes = [], []
        policy = ContainerPolicy(container, player, notify=msgs.append, report_crime=lambda c: crimes.append(c) or [],
                                 stealth=StealthSettings(), rng=ScriptedRolls(*rolls))
        return TransferSession(container.inventory, player.inventory, policy), msgs, crimes

    def test_a_legendary_piece_is_beyond_a_novice_entirely(self, player):
        c = self.vault("legendary")
        s, msgs, crimes = self.session(c, player, PASS)
        r = s.confirm()
        assert r.outcome is TransferOutcome.HOPELESS and c.inventory and crimes == []
        assert "cannot even see how it could be freed" in msgs[-1]

    def test_within_reach_it_is_a_roll_and_a_failed_attempt_may_be_heard(self, player):
        from unprepared.config import StealthSettings
        from unprepared.systems.stealth import secure_verdict
        player.skills.skills[SkillType.STEALING]["level"] = 30                # 48 - 30 = 18: within the gate
        c = self.vault("legendary")
        challenge, odds, possible = secure_verdict(player, c.inventory[0], StealthSettings())
        assert possible and challenge == 48.0 and 0.05 < odds < 0.15
        s, msgs, crimes = self.session(c, player, FAIL, FAIL)                 # won't come free; the rattle is noticed
        assert s.confirm().outcome is TransferOutcome.SECURED and c.inventory and crimes == ["Theft"]
        assert "will not come free" in msgs[-1]
        s, msgs, crimes = self.session(c, player, PASS, PASS)                 # free, and unnoticed
        assert s.confirm().outcome is TransferOutcome.MOVED and not c.inventory
        assert player.inventory[-1].id == "relic" and player.inventory[-1].is_stolen and crimes == []

    def test_gear_moves_the_gate(self, player):
        from unprepared.config import StealthSettings
        from unprepared.systems.stealth import secure_verdict
        relic = Item("relic", "Relic", rarity="legendary")
        player.skills.skills[SkillType.STEALING]["level"] = 10
        assert not secure_verdict(player, relic, StealthSettings())[2]
        cloak = Item("cloak", "Cloak", gear="legend_cloak", skill_bonuses=(("Stealing", 15),))
        player.add_item(cloak)
        player.equip(cloak)
        assert secure_verdict(player, relic, StealthSettings())[2]             # 25 effective: now worth a try

    def test_common_and_rare_things_simply_come_away(self, player):
        from unprepared.config import StealthSettings
        from unprepared.systems.stealth import secure_verdict
        for rarity in ("common", "uncommon", "rare"):
            assert secure_verdict(player, Item("x", "x", rarity=rarity), StealthSettings()) == (None, 1.0, True)

    def test_rarer_things_are_harder_to_take_unnoticed(self, player):
        from unprepared.config import StealthSettings
        from unprepared.systems.stealth import stealing_chance
        odds = [stealing_chance(player, StealthSettings(), item=Item("x", "x", rarity=r))
                for r in ("common", "uncommon", "rare", "epic", "legendary")]
        assert all(a > b for a, b in zip(odds, odds[1:]))

    def test_a_chest_of_coins_is_emptied_a_handful_at_a_time(self, player):
        from unprepared.systems.transfer import COIN_HANDFUL
        c = Container("chest", "Chest", BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1)), owner="Lady Ashgrove")
        c.inventory = [Item("coins", "Gold Coins", value=1.0, stackable=True, quantity=COIN_HANDFUL + 15, category="coins")]
        s, msgs, crimes = self.session(c, player, PASS)
        gold = player.gold
        r = s.confirm()
        assert r.outcome is TransferOutcome.MOVED and player.gold == gold + COIN_HANDFUL and c.inventory[0].quantity == 15
        assert not any(i.id == "coins" for i in player.inventory)             # straight into the purse
        results = s.take_all()
        assert player.gold == gold + COIN_HANDFUL + 15 and c.inventory == [] and len(results) == 1
        assert msgs[0] == f"Stole {COIN_HANDFUL} gold from the chest."
