"""Crimes and punishments, rule by rule (README "Crimes and punishments")."""
from __future__ import annotations

import pytest

from tests.conftest import said
from tests.helpers import spot
from unprepared.world.levels.town import STARTING_GOLD
from unprepared.config import GameSettings, Penalty
from unprepared.entities import AIState, Item
from unprepared.systems import MoveIntent, justice
from unprepared.world.levels.town import SHOP_FLOOR


@pytest.fixture
def game(tmp_path):
    from unprepared.game import Game
    g = Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=5, max_log_messages=400),
             save_file=str(tmp_path / "s.json"))
    g.player.inventory = [i for i in g.player.inventory if not i.is_stolen]      # start with clean pockets
    return g


def guard(game, name="Guard Captain Vance"):
    v = game.world.find_npc(name)
    v.path, v.current_task, v.schedule = [], None, []
    return v


def caught_by(game, v, at=None):
    """Put the player within the guard's reach (optionally at ``at``) and let one frame run."""
    if at is not None:
        v.position = spot(game.world, at[0], at[2] - 2.0)
        v.state = AIState.PATROL
        game.player.position = spot(game.world, at[0], at[2])
    else:
        game.player.position = spot(game.world, v.position.x, v.position.z + 2.0)
    v.turn_towards(game.player.position)
    game.update(0.05)


def charge(game, crime, indoors, gold, stolen=()):
    """Declare ``crime`` indoors or out, with ``gold`` in the purse and ``stolen`` items carried,
    and have Vance charge the player on the spot (no chase: the memory is already ripe)."""
    w = game.world
    v = guard(game)
    game.player.gold = gold
    for item in stolen:
        game.player.inventory.append(item)
    where = w.spot(*SHOP_FLOOR) if indoors else spot(game.world, -12.0, 3.0)
    assert w.is_indoors(where) is indoors
    w.declare_crime(crime, where, witnessed=True, reporter=v.name, witnesses=[v])
    v.charge_ready_at = 0.0
    caught_by(game, v, at=(where.x, 0.0, where.z))
    return w.jail, v


def days(game):
    return game.world.jail.days_left(game.world.day_count) if game.world.jail.serving else 0


class TestPenaltyTable:
    def test_theft_crimes_are_punished_by_where_they_happened(self, game):
        j = game.settings.justice
        for crime in ("Pickpocketing", "Lockpicking a door", "Lockpicking a lockbox", "Shoplifting", "Theft"):
            assert justice.penalty_for(crime, False, j) == Penalty(100.0, 2, 3)
            assert justice.penalty_for(crime, True, j) == Penalty(500.0, 5, 7)

    def test_minor_crimes_carry_a_fine_and_a_day_only_if_it_cannot_be_paid(self, game):
        j = game.settings.justice
        assert justice.penalty_for("Skulking about", True, j) == Penalty(20.0, 0, 1)
        assert justice.penalty_for("No torch after dark", False, j) == Penalty(5.0, 0, 1)
        assert justice.penalty_for("Trespassing", True, j) == Penalty(15.0, 0, 1)
        assert justice.penalty_for("Something new", False, j) == j.default_penalty == Penalty(20.0, 0, 1)

    def test_the_penalty_is_fixed_when_the_crime_is_declared(self, game):
        w = game.world
        v = guard(game)
        w.declare_crime("Pickpocketing", w.spot(*SHOP_FLOOR), witnessed=True, reporter=v.name, witnesses=[v])
        m = v.crime_memories[-1]
        assert (m.fine, m.jail_days, m.jail_days_unpaid, m.indoors) == (500.0, 5, 7, True)
        assert m.label == "Pickpocketing (indoors)"


class TestSkulkingAndTheTorchLaw:
    def test_skulking_is_a_20g_fine_with_no_jail(self, game):
        jail, _ = charge(game, "Skulking about", False, gold=50.0)
        assert game.player.gold == 30.0 and not jail.serving

    def test_skulking_unpaid_is_one_day(self, game):
        jail, _ = charge(game, "Skulking about", False, gold=10.0)
        assert jail.serving and days(game) == 1 and game.player.gold == 10.0

    def night_walk(self, game, seconds):
        """Walk about unlit in the dark in front of Orrin; stop as soon as the torch law is on the record."""
        w = game.world
        w.clock.hour = 23                                              # dark
        orrin = guard(game, "Guard Orrin")
        orrin.state = AIState.GUARDING
        orrin.position = spot(game.world, 30.0, -2.0)
        orrin.facing_yaw = 90.0                                        # looking east ...
        game.player.position = spot(game.world, 35.0, -2.0)               # ... at us, five units off
        game.player.has_lit_torch = False
        for _ in range(int(seconds / 0.1)):
            game.update(0.1, MoveIntent(left=True, right=True))      # shuffling about in the dark, ignoring him
            if any(m.crime == "No torch after dark" for m in orrin.crime_memories):
                break
        return orrin

    def test_ignoring_the_torch_law_is_a_5g_crime(self, game):
        w = game.world
        orrin = self.night_walk(game, 60.0)
        assert said(game.messages.messages, "torch_law")
        memory = next(m for m in orrin.crime_memories if m.crime == "No torch after dark")
        assert (memory.fine, memory.jail_days, memory.jail_days_unpaid) == (5.0, 0, 1)
        assert not any(m.crime == "Skulking about" for m in orrin.crime_memories)   # walking unlit is not skulking
        orrin.charge_ready_at = 0.0
        orrin.state = AIState.GUARDING
        game.player.position = spot(game.world, orrin.position.x + 1.5, orrin.position.z)     # within reach
        orrin.turn_towards(game.player.position)
        game.update(0.05)
        assert game.player.gold == STARTING_GOLD - 5.0 and not w.jail.serving

    def test_the_third_ignored_warning_is_a_crime_even_far_off(self, game):
        """Too far away for suspicion to build, the warnings alone make the crime."""
        orrin = guard(game, "Guard Orrin")
        orrin.torch_warnings = game.settings.perception.torch_warnings_before_crime - 1
        game.world.clock.hour = 23
        orrin.state = AIState.GUARDING
        orrin.position = spot(game.world, 30.0, -2.0)
        orrin.facing_yaw = 90.0
        orrin.has_torch = True
        game.player.position = spot(game.world, 42.0, -2.0)                 # twelve units: seen by torchlight, slow to alarm
        game.player.has_lit_torch = False
        for _ in range(30):
            game.update(0.1, MoveIntent(left=True, right=True))
            if orrin.crime_memories:
                break
        assert [m.crime for m in orrin.crime_memories] == ["No torch after dark"] and orrin.torch_warnings == 0

    def test_lighting_the_torch_resets_the_warnings(self, game):
        orrin = guard(game, "Guard Orrin")
        orrin.torch_warnings = 2
        game.world.clock.hour = 23
        game.player.has_lit_torch = True
        game.player.position = spot(game.world, orrin.position.x, orrin.position.z + 3.0)
        game.update(0.1)
        assert orrin.torch_warnings == 0


class TestTheftOutdoorsAndIndoors:
    @pytest.mark.parametrize("crime", ["Pickpocketing", "Lockpicking a door"])
    def test_outdoors_is_100g_and_two_days(self, game, crime):
        jail, _ = charge(game, crime, False, gold=150.0)
        assert game.player.gold == 50.0 and jail.serving and days(game) == 2

    def test_outdoors_unpaid_is_three_days(self, game):
        jail, _ = charge(game, "Pickpocketing", False, gold=60.0)
        assert game.player.gold == 60.0 and days(game) == 3            # nothing could be paid: the purse is left alone

    @pytest.mark.parametrize("crime", ["Pickpocketing", "Lockpicking a lockbox", "Shoplifting", "Theft"])
    def test_indoors_is_500g_and_five_days(self, game, crime):
        jail, _ = charge(game, crime, True, gold=600.0)
        assert game.player.gold == 100.0 and days(game) == 5

    def test_indoors_unpaid_is_seven_days(self, game):
        jail, _ = charge(game, "Shoplifting", True, gold=50.0)
        assert game.player.gold == 50.0 and days(game) == 7


class TestStolenGoodsOnYou:
    def test_each_stolen_item_adds_its_value_and_a_day(self, game):
        loot = [Item("plate", "Pewter Plate", value=18.0, is_stolen=True, owner="Shopkeeper Barnaby"),
                Item("wine", "Bottle of Wine", value=12.0, is_stolen=True, stackable=True, quantity=2, owner="Shopkeeper Barnaby")]
        jail, _ = charge(game, "Skulking about", False, gold=100.0, stolen=loot)
        assert game.player.gold == 100.0 - 20.0 - 18.0 - 12.0 * 2           # the fine, and every stolen unit's worth
        assert days(game) == 3                                              # a day per stolen unit (skulking itself: none)
        assert not any(i.is_stolen for i in game.player.inventory)          # ... and the goods went back
        assert any("3 stolen items on you" in m for m in game.messages.messages)

    def test_a_stolen_item_you_cannot_pay_for_is_two_days(self, game):
        loot = [Item("necklace", "Ruby Necklace", value=320.0, is_stolen=True, owner="Shopkeeper Barnaby")]
        jail, _ = charge(game, "Skulking about", False, gold=30.0, stolen=loot)
        assert game.player.gold == 10.0 and days(game) == 2                 # the fine paid, the necklace not: 1 + 1

    def test_everything_stacks(self, game):
        """Indoors pickpocketing (500g/5d) with 50g: the fine is beyond the purse (7 days) but the
        two stolen things on us are not - their worth is paid and each costs a day: 9 in all."""
        loot = [Item("lantern", "Brass Lantern", value=30.0, is_stolen=True, owner="Shopkeeper Barnaby"),
                Item("honey", "Jar of Honey", value=9.0, is_stolen=True, owner="Shopkeeper Barnaby")]
        jail, _ = charge(game, "Pickpocketing", True, gold=50.0, stolen=loot)
        assert days(game) == 7 + 1 + 1 and game.player.gold == 50.0 - 30.0 - 9.0
        loot = [Item("necklace", "Ruby Necklace", value=320.0, is_stolen=True, owner="Shopkeeper Barnaby")]
        jail, _ = charge(game, "Pickpocketing", True, gold=50.0, stolen=loot)
        assert days(game) == 7 + 2 and game.player.gold == 50.0              # ... an unaffordable one costs two

    def test_the_bill_is_paid_in_order_of_severity(self, game):
        """Two crimes and 110g: the 100g street theft is paid, the 20g skulking is not."""
        w = game.world
        v = guard(game)
        game.player.gold = 110.0
        street = spot(game.world, -12.0, 3.0)
        w.declare_crime("Pickpocketing", street, witnessed=True, reporter=v.name, witnesses=[v])
        w.declare_crime("Skulking about", street, witnessed=True, reporter=v.name, witnesses=[v])
        verdict = justice.judge(w, game.player)
        assert verdict.charges == ["Pickpocketing (in the street)", "Skulking about"]
        assert (verdict.fine, verdict.paid, verdict.days, verdict.unpaid) == (120.0, 100.0, 2 + 1, True)

    def test_the_same_crime_known_to_several_guards_is_charged_once(self, game):
        w = game.world
        v = guard(game)
        w.declare_crime("Skulking about", spot(game.world, -12.0, 3.0), witnessed=True, reporter=v.name, witnesses=[v])
        assert sum(1 for g in w.guards() if g.crime_memories) >= 2           # the cry carried to the others
        assert justice.judge(w, game.player).fine == 20.0 and justice.outstanding_fine(w) == 20.0
