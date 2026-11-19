import random

import pytest

from unprepared.config import LockpickSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import Door, Item, Player, SkillType
from unprepared.systems import LockpickMinigame, LockpickOutcome


@pytest.fixture
def door():
    return Door("shop_door", Vector3D(0, 0, 0), is_locked=True)


@pytest.fixture
def player():
    return Player(Vector3D(0, 0, 0), inventory=[Item("lp1", "Pick", is_lockpick=True), Item("lp2", "Pick", is_lockpick=True)])


@pytest.fixture
def game():
    return LockpickMinigame(LockpickSettings(), rng=random.Random(42))


def test_initial_state(game):
    assert game.pick_angle == 0.0 and not game.active and game.target is None


def test_rotate_wraps(game):
    assert game.rotate(-1) == 350.0
    assert game.rotate(2) == 10.0
    assert game.rotate(35) == 0.0


def test_alignment_window_wraps_around_360():
    g = LockpickMinigame(LockpickSettings(random_sweet_spot=False, default_sweet_spot=5.0))
    g.pick_angle = 350.0
    assert g.angle_error == pytest.approx(15.0) and g.is_aligned
    g.pick_angle = 340.0
    assert not g.is_aligned


def test_each_attempt_gets_a_fresh_random_sweet_spot(game, door):
    game.begin(door)
    first = game.sweet_spot
    game.begin(door)
    assert 0.0 <= first < 360.0 and game.sweet_spot != first
    # deterministic for a given seed
    other = LockpickMinigame(LockpickSettings(), rng=random.Random(42))
    other.begin(door)
    assert other.sweet_spot == first


def test_pick_resets_each_attempt(game, door):
    game.begin(door)
    game.rotate(3)
    game.cancel()
    game.begin(door)
    assert game.pick_angle == 0.0


def test_fixed_sweet_spot_option(door):
    g = LockpickMinigame(LockpickSettings(random_sweet_spot=False))
    g.begin(door)
    assert g.sweet_spot == 45.0
    g.begin(door, sweet_spot=200.0)
    assert g.sweet_spot == 200.0


def test_snap_consumes_a_pick_and_ends_the_attempt(game, door, player):
    msgs = []
    game.begin(door, sweet_spot=180.0, fail_xp=15.0)
    assert game.attempt(player, msgs.append) is LockpickOutcome.SNAPPED
    assert not game.active and door.is_locked                 # try again from scratch
    assert [i.id for i in player.inventory] == ["lp2"]
    assert player.skills.skills[SkillType.LOCKPICKING]["xp"] == 15.0     # failing still teaches a little
    assert msgs == ["SNAP! Your lockpick broke against the mechanism. Try again."]


def lockable(tier):
    return Door(f"tier{tier}", Vector3D(0, 0, 0), is_locked=True, lock_difficulty=tier)


def test_tolerance_comes_from_skill_and_the_lock_tier(door, player):
    """The window widens with the margin of effective skill over the lock's challenge."""
    from unprepared.config import StealthSettings
    from unprepared.systems import lockpick_tolerance
    s = StealthSettings()
    window = [lockpick_tolerance(player, lockable(t), s) for t in range(6)]
    assert all(a > b for a, b in zip(window, window[1:]) if b > s.lockpick_min_tolerance)   # better locks, narrower
    assert window[-1] == s.lockpick_min_tolerance              # a legendary lock at level 1: a sliver
    before = lockpick_tolerance(player, lockable(3), s)
    player.skills.skills[SkillType.LOCKPICKING]["level"] = 40
    assert lockpick_tolerance(player, lockable(3), s) > before
    player.skills.skills[SkillType.LOCKPICKING]["level"] = 100
    assert lockpick_tolerance(player, lockable(0), s) == s.lockpick_max_tolerance   # capped: a master picks a latch almost blind
    g = LockpickMinigame(LockpickSettings())
    g.begin(door, sweet_spot=0.0, tolerance=5.0)
    g.pick_angle = 10.0
    assert not g.is_aligned
    g.pick_angle = 4.0
    assert g.is_aligned


def test_harder_locks_pay_more_xp():
    """A lock's challenge sets what picking it teaches (against the picker's base level)."""
    from unprepared.config import StealthSettings
    s = StealthSettings()
    xp = []
    for tier in (0, 3):
        p = Player(Vector3D(0, 0, 0), inventory=[Item("lp1", "Pick", is_lockpick=True)])
        g = LockpickMinigame(LockpickSettings())
        g.begin(lockable(tier), sweet_spot=0.0, challenge=s.lock_challenge[tier])
        assert g.attempt(p) is LockpickOutcome.SUCCESS
        xp.append(p.skills.skills[SkillType.LOCKPICKING]["xp"] + 80.0 * (p.skills.level(SkillType.LOCKPICKING) - 1))
    assert xp[1] > 4 * xp[0]


def test_on_the_spot_the_pins_set_with_the_locks_odds():
    """Good timing puts the pick on the spot; the pins then set with the lock's chance - so a
    perfectly timed press opens a lock exactly as often as the model says."""
    wins, trials = 0, 4000
    g = LockpickMinigame(LockpickSettings(consume_pick_on_snap=False), rng=random.Random(5))
    p = Player(Vector3D(0, 0, 0))
    for _ in range(trials):
        d = Door("d", Vector3D(0, 0, 0), is_locked=True)
        g.begin(d, sweet_spot=0.0, tolerance=5.0, wobble=0.0, set_chance=0.592)
        wins += g.attempt(p) is LockpickOutcome.SUCCESS
    assert wins / trials == pytest.approx(0.592, abs=0.025)


def test_slipping_pins_snap_the_pick_too(door, player):
    msgs = []
    g = LockpickMinigame(LockpickSettings(), rng=random.Random(1))
    g.begin(door, sweet_spot=0.0, set_chance=0.0)              # on the spot, hopeless lock
    assert g.is_aligned
    assert g.attempt(player, msgs.append) is LockpickOutcome.SNAPPED and door.is_locked
    assert msgs[0] == "SNAP! The pins slip and your lockpick breaks. Try again."


def test_last_pick_snapping_ends_the_minigame(game, door, player):
    msgs = []
    player.inventory.pop()
    game.begin(door, sweet_spot=180.0)
    assert game.attempt(player, msgs.append) is LockpickOutcome.OUT_OF_PICKS
    assert not game.active and not player.has_lockpick()
    assert msgs[-1] == "That was your last lockpick. Forge more at the anvil."


def test_success_unlocks_opens_and_rewards(game, door, player):
    msgs = []
    game.begin(door, sweet_spot=40.0)
    for _ in range(4):
        game.rotate(1)
    assert game.attempt(player, msgs.append) is LockpickOutcome.SUCCESS
    assert not game.active and not door.is_locked and door.is_open
    assert player.skills.skills[SkillType.LOCKPICKING]["xp"] == 45.0
    assert len(player.inventory) == 2           # success does not consume a pick
    assert msgs == ["SUCCESS! Lock successfully bypassed!"]


def test_every_random_sweet_spot_is_reachable_in_10_degree_steps(door):
    g = LockpickMinigame(LockpickSettings(), rng=random.Random(7))
    for _ in range(200):
        g.begin(door)
        assert any(abs(((k * 10.0 - g.sweet_spot) + 180) % 360 - 180) <= 15.0 for k in range(36))
