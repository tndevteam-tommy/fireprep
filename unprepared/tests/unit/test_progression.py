"""The one skill model behind every check: effective level (base + gear) against a challenge on
one logistic curve; XP doubling with every 12 levels the deed stands above the base level."""
import math

import pytest

from unprepared.config import LOCK_TIER_NAMES, ProgressionSettings, StealthSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import Item, Player, SkillType
from unprepared.systems import progression
from unprepared.systems.progression import chance

S = ProgressionSettings()
LEGENDARY = StealthSettings().lock_challenge[LOCK_TIER_NAMES.index("Legendary")]


def player_with(level: int = 1, skill=SkillType.LOCKPICKING, bonus: int = 0) -> Player:
    p = Player(Vector3D(0, 0, 0), inventory=[])
    p.skills.skills[skill]["level"] = level
    if bonus:
        amulet = Item("amulet", "Amulet", gear="amulet", skill_bonuses=((skill.value, bonus),))
        p.add_item(amulet)
        assert p.equip(amulet)
    return p


class TestTheExamples:
    """The figures the design was drawn up around - a legendary lock and a +40 necklace."""

    def test_level_10_without_gear_is_all_but_hopeless(self):
        p = player_with(10)
        assert progression.effective_level(p, SkillType.LOCKPICKING) == 10
        assert progression.skill_chance(p, SkillType.LOCKPICKING, LEGENDARY) < 0.02

    def test_level_10_with_a_plus_40_necklace_is_a_fair_chance(self):
        p = player_with(10, bonus=40)
        assert progression.effective_level(p, SkillType.LOCKPICKING) == 50
        assert 0.55 <= progression.skill_chance(p, SkillType.LOCKPICKING, LEGENDARY) <= 0.65

    def test_level_60_with_the_necklace_finds_it_easy(self):
        p = player_with(60, bonus=40)
        assert progression.effective_level(p, SkillType.LOCKPICKING) == 100
        assert progression.skill_chance(p, SkillType.LOCKPICKING, LEGENDARY) > 0.99

    def test_the_necklace_counts_only_while_worn(self):
        p = player_with(10, bonus=40)
        amulet = next(i for i in p.inventory if i.id == "amulet")
        p.unequip(amulet)
        assert progression.effective_level(p, SkillType.LOCKPICKING) == 10
        assert progression.effective_level(p, SkillType.SNEAK) == 1          # (and only for its own skill)


class TestTheCurve:
    def test_an_even_match_is_a_coin_toss(self):
        for c in (-25.0, 0.0, 17.0, 47.0):
            assert chance(c, c) == pytest.approx(0.5)

    def test_it_is_the_logistic_of_the_margin(self):
        for e, c in ((10, 47), (50, 47), (1, -10), (30, 18)):
            expected = S.chance_floor + (S.chance_ceiling - S.chance_floor) / (1 + math.exp(-(e - c) / S.logistic_scale))
            assert chance(e, c) == pytest.approx(expected)

    def test_symmetric_skill_and_challenge_swap_the_odds(self):
        for e, c in ((10, 47), (33, 5), (80, 60)):
            assert chance(e, c) + chance(c, e) == pytest.approx(1.0)

    def test_monotone_in_skill_and_in_challenge(self):
        for c in StealthSettings().lock_challenge:
            odds = [chance(e, c) for e in range(1, 101)]
            assert all(a < b for a, b in zip(odds, odds[1:]))
        for e in (1, 25, 60):
            odds = [chance(e, c) for c in StealthSettings().lock_challenge]
            assert all(a > b for a, b in zip(odds, odds[1:]))

    def test_never_quite_certain_either_way(self):
        assert chance(1, 500) == pytest.approx(S.chance_floor, abs=1e-9)
        assert chance(500, 1) == pytest.approx(S.chance_ceiling, abs=1e-9)

    def test_gear_is_worth_exactly_its_levels(self):
        """+12 from gear is the same as 12 more levels of practice."""
        geared, practised = player_with(8, SkillType.SNEAK, bonus=12), player_with(20, SkillType.SNEAK)
        for c in (-5.0, 15.0, 40.0):
            assert progression.skill_chance(geared, SkillType.SNEAK, c) == progression.skill_chance(practised, SkillType.SNEAK, c)


class TestXP:
    def test_an_even_deed_pays_its_base(self):
        assert progression.xp_reward(20, 50.0, challenge=20.0) == pytest.approx(50.0)
        assert progression.xp_reward(1, 50.0) == 50.0                             # no challenge given: factor 1
        assert progression.xp_reward(1, 50.0, magnitude=2.0) == 100.0
        assert progression.xp_reward(1, -5.0) == 0.0

    def test_every_12_levels_above_the_base_doubles_it_below_halves_it(self):
        assert progression.challenge_factor(10, 22) == pytest.approx(2.0)
        assert progression.challenge_factor(10, 34) == pytest.approx(4.0)
        assert progression.challenge_factor(22, 10) == pytest.approx(0.5)
        assert progression.challenge_factor(1, 500) == S.xp_factor_max
        assert progression.challenge_factor(100, -25) == S.xp_factor_min

    def test_gear_that_lets_a_novice_do_hard_things_makes_them_learn_fast(self):
        """XP is reckoned against the *base* level: the necklace opens the legendary lock, and the
        level-10 picker is paid for it as the master feat it is for them."""
        novice, master = player_with(10, bonus=40), player_with(60, bonus=40)
        assert progression.grant(novice, SkillType.LOCKPICKING, 45.0, challenge=LEGENDARY) > \
            8 * progression.grant(master, SkillType.LOCKPICKING, 45.0, challenge=LEGENDARY)

    def test_grant_reaches_the_skill_tree(self):
        p = Player(Vector3D(0, 0, 0), inventory=[])
        assert progression.grant(p, SkillType.STEALING, 40.0, challenge=1.0) == pytest.approx(40.0)
        assert p.skills.skills[SkillType.STEALING]["xp"] == pytest.approx(40.0)
        assert progression.grant(p, SkillType.STEALING, 0.0) == 0.0

    def test_value_xp_is_capped_and_has_a_floor(self):
        assert progression.value_xp(0.0, S) == S.value_xp_floor
        assert progression.value_xp(120.0, S) == 60.0
        assert progression.value_xp(10000.0, S) == S.value_xp_cap


def test_the_quietest_steps_come_with_sneak_and_never_reach_silence():
    values = [progression.noise_multiplier(e) for e in (1, 10, 30, 50, 100, 200)]
    assert values[0] == pytest.approx(1.0)
    assert all(a > b for a, b in zip(values, values[1:]))
    assert values[-1] > S.noise_floor and values[3] < 0.5


def test_scaled_yield_ramps_linearly():
    assert progression.scaled_yield(1, 1, 15, 50) == 1
    assert progression.scaled_yield(50, 1, 15, 50) == 15
    assert progression.scaled_yield(99, 1, 15, 50) == 15
    assert progression.scaled_yield(25, 1, 15, 50) == 7
    assert progression.scaled_yield(3, 4, 4, 1) == 4
