import pytest

from unprepared.entities import SkillTree, SkillType


def test_initial_state():
    tree = SkillTree()
    for skill in SkillType:
        assert tree.skills[skill] == {"level": 1, "xp": 0.0, "xp_needed": 80.0}


def test_xp_accumulates_without_levelling():
    tree = SkillTree()
    assert tree.add_xp(SkillType.SNEAK, 50.0) == 0
    assert tree.skills[SkillType.SNEAK]["xp"] == 50.0
    assert tree.level(SkillType.SNEAK) == 1


def test_level_up_carries_over_and_grows_requirement_linearly():
    tree = SkillTree()
    messages = []
    assert tree.add_xp(SkillType.LOCKPICKING, 110.0, notify_callback=messages.append) == 1
    data = tree.skills[SkillType.LOCKPICKING]
    assert data["level"] == 2 and data["xp"] == 30.0 and data["xp_needed"] == 92.0        # 80 + 12 per level
    assert messages == ["SKILL UP! Your Lockpicking skill increased to Level 2!"]


def test_large_grant_levels_several_times():
    tree = SkillTree()
    msgs = []
    # 80 + 92 + 104 = 276 needed for three levels
    assert tree.add_xp(SkillType.SMITHING, 286.0, notify_callback=msgs.append) == 3
    data = tree.skills[SkillType.SMITHING]
    assert data["level"] == 4 and data["xp"] == pytest.approx(10.0) and data["xp_needed"] == pytest.approx(116.0)
    assert [m[-8:] for m in msgs] == ["Level 2!", "Level 3!", "Level 4!"]


def test_elite_levels_stay_reachable():
    """The curve is linear: level 50 needs 668 XP for the next level, not millions."""
    assert SkillTree.xp_needed_for(50) == 80.0 + 12.0 * 49
    assert SkillTree.xp_needed_for(1) == 80.0


def test_levels_stop_at_100():
    tree = SkillTree()
    assert SkillTree.MAX_LEVEL == 100
    gained = tree.add_xp(SkillType.SNEAK, 10_000_000.0)
    assert gained == 99 and tree.level(SkillType.SNEAK) == 100 and tree.skills[SkillType.SNEAK]["xp"] == 0.0
    assert tree.add_xp(SkillType.SNEAK, 500.0) == 0 and tree.level(SkillType.SNEAK) == 100
    other = SkillTree()
    other.load_dict({"Sneak": {"level": 250, "xp": 3.0}})                     # a hand-edited save: clamped
    assert other.level(SkillType.SNEAK) == 100


def test_negative_or_zero_xp_is_ignored():
    tree = SkillTree()
    assert tree.add_xp(SkillType.BARTER, 0.0) == 0 and tree.add_xp(SkillType.BARTER, -5.0) == 0
    assert tree.skills[SkillType.BARTER]["xp"] == 0.0


def test_progress_fraction():
    """Progress towards the next level is the XP banked over the XP that level needs - a quarter
    of the way after 20 of the first 80, and counted afresh (against the bigger requirement)
    once the level is gained."""
    tree = SkillTree()
    tree.add_xp(SkillType.STEALING, 20.0)
    data = tree.skills[SkillType.STEALING]
    assert data["xp"] / data["xp_needed"] == pytest.approx(0.25)
    tree.add_xp(SkillType.STEALING, 60.0 + 46.0)                        # level 2 (80 XP), then 46 of the next 92
    data = tree.skills[SkillType.STEALING]
    assert data["level"] == 2 and data["xp"] / data["xp_needed"] == pytest.approx(0.5)


def test_dict_roundtrip():
    tree = SkillTree()
    tree.add_xp(SkillType.PICKPOCKET, 75.0)
    other = SkillTree()
    other.load_dict(tree.to_dict())
    assert other.to_dict() == tree.to_dict()
    assert set(tree.to_dict()) == {"Sneak", "Lockpicking", "Pickpocket", "Stealing", "Barter", "Smithing"}


def test_load_dict_ignores_unknown_and_recomputes_the_requirement():
    tree = SkillTree()
    tree.load_dict({"Sneak": {"level": 5, "xp": 1.0, "xp_needed": 2.0}, "Bogus": {}})
    assert tree.level(SkillType.SNEAK) == 5
    assert tree.skills[SkillType.SNEAK]["xp_needed"] == SkillTree.xp_needed_for(5)     # old curve replaced
    assert tree.level(SkillType.SMITHING) == 1
