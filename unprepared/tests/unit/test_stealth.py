"""Sneak checks: rolling against watching NPCs, unseen sneaking XP, skill effects, failure consequences."""
import pytest

from tests.helpers import spot
from tests.conftest import said
from unprepared.config import GameSettings, StealthSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState, SkillType
from unprepared.game import Game
from unprepared.systems import MoveIntent, lockpick_wobble, noise_multiplier, sneak_chance
from unprepared.world.levels.town import STARTING_GOLD


@pytest.fixture
def game(tmp_path):
    return Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=7), save_file=str(tmp_path / "s.json"))


S = StealthSettings()


def vigil(role):
    return dict(S.vigilance)[role]


def factor(level, challenge):
    from unprepared.systems.progression import challenge_factor
    return challenge_factor(level, challenge)


def isolate(game, keep):
    """Park everyone but ``keep`` in the far corner so nobody else hears or sees anything."""
    for npc in game.world.npcs:
        if npc is keep:
            continue
        npc.path, npc.current_task, npc.schedule = [], None, []
        npc.position = spot(game.world, 46.0, 28.0)
        npc.state = AIState.IDLE


def watcher(game):
    """Vance standing still in the street, looking at the player 4 units south of him."""
    vance = game.world.find_npc("Guard Captain Vance")
    vance.path = []
    vance.current_task = None                       # no patrol re-planning during the test
    vance.schedule = []
    vance.position = spot(game.world, 3.0, -3.0)                  # beside the well, not behind it
    game.player.position = spot(game.world, vance.position.x, vance.position.z + 4.0)
    vance.turn_towards(game.player.position)
    return vance


def test_sneak_chance_scales_with_skill_guards_and_daylight(game):
    """The observer's vigilance plus how visible the player is, against effective Sneak."""
    from unprepared.systems.progression import chance
    vance = game.world.find_npc("Guard Captain Vance")
    jae = game.world.find_npc("Jae the Fence")
    s = StealthSettings()
    vig = dict(s.vigilance)
    assert sneak_chance(game.player, jae, s) == pytest.approx(chance(1, vig["fence"]))                 # level 1, dark
    assert sneak_chance(game.player, vance, s) == pytest.approx(chance(1, vig["guard"]))              # guards are sharper
    assert sneak_chance(game.player, vance, s, visibility=1.0) == pytest.approx(
        chance(1, vig["guard"] + s.sneak_visibility_challenge))                                            # ... in broad daylight
    assert sneak_chance(game.player, vance, s, visibility=1.0) < sneak_chance(game.player, vance, s) < sneak_chance(game.player, jae, s)
    game.player.skills.skills[SkillType.SNEAK]["level"] = 15
    mid = sneak_chance(game.player, vance, s, visibility=1.0)
    game.player.skills.skills[SkillType.SNEAK]["level"] = 60
    elite = sneak_chance(game.player, vance, s, visibility=1.0)
    assert chance(1, 30.0) < mid < elite
    assert elite > 0.95                      # an elite sneak is all but invisible at noon in front of a guard
    game.player.skills.skills[SkillType.SNEAK]["level"] = 100
    assert sneak_chance(game.player, vance, s, visibility=1.0) < 1.0                   # never certain


def test_skill_makes_the_player_quieter(game):
    from unprepared.systems.progression import noise_multiplier as curve
    s = StealthSettings()
    assert noise_multiplier(game.player, s) == 1.0
    game.player.skills.skills[SkillType.SNEAK]["level"] = 25
    assert noise_multiplier(game.player, s) == pytest.approx(curve(25))
    quiet = noise_multiplier(game.player, s)
    assert 0.55 < quiet < 0.75
    game.player.noise_skill_multiplier = quiet
    game.toggle_crouch()
    clothes = game.player.clothing_noise_factor()                          # soft leather boots: a touch quieter
    assert 0.9 < clothes < 1.0
    assert game.player.get_noise_level(True, False) == pytest.approx(1.5 * 0.3 * clothes * quiet)


def test_lockpick_wobble_shrinks_with_skill_and_grows_with_the_lock(game):
    from unprepared.world.level import Container
    from unprepared.core.math3d import BoundingBox3D
    s = StealthSettings()

    def lock(tier):
        return Container("box", "Box", BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1)), locked=True, lock_difficulty=tier)
    flimsy, legendary = lock(0), lock(5)
    assert lockpick_wobble(game.player, flimsy, s) < 2.0                      # a latch at level 1: steady
    assert lockpick_wobble(game.player, legendary, s) == s.lockpick_max_wobble  # far beyond: wild
    game.player.skills.skills[SkillType.LOCKPICKING]["level"] = 100
    assert lockpick_wobble(game.player, legendary, s) == 0.0


def total_sneak_xp(game):
    """XP earned so far including what went into levels already gained."""
    from unprepared.entities import SkillTree
    data = game.player.skills.skills[SkillType.SNEAK]
    return data["xp"] + sum(SkillTree.xp_needed_for(level) for level in range(1, int(data["level"])))


def sneak_xp_after(game, seconds=2.0, dt=0.05):
    before = total_sneak_xp(game)
    for _ in range(int(seconds / dt)):
        game.update(dt, MoveIntent(left=True, right=True))    # "moving" on the spot
    return total_sneak_xp(game) - before


def test_sneaking_alone_teaches_a_trickle(game):
    game.player.position = spot(game.world, -30.0, 30.0)          # nobody around
    game.toggle_crouch()
    assert sneak_xp_after(game) == pytest.approx(0.75 * factor(1, 0.0) * 2.0, rel=0.05)
    for _ in range(40):
        game.update(0.05)                                      # crouching but still: nothing
    assert sneak_xp_after(game, seconds=0.0) == 0.0


def test_grinding_sneak_around_a_sleeper(game):
    """Crouch-walking about a sleeping person's house is the intended way to train early on."""
    barnaby = game.world.find_npc("Shopkeeper Barnaby")
    isolate(game, barnaby)
    barnaby.state = AIState.SLEEPING
    barnaby.position = spot(game.world, -37.25, 13.25)            # his bed
    game.player.position = spot(game.world, -41.0, 9.0)           # across the room, behind furniture
    game.toggle_crouch()
    rate = game.world.sneak_xp_rate(game.player)
    expected = 3.0 * factor(1, vigil("merchant") + S.asleep_challenge)   # the "behind" rate, for a sleeper
    assert rate == pytest.approx(expected)                     # line of sight or not
    assert rate > 0.75 * factor(1, 0.0)                        # more than pottering about alone
    gained = sneak_xp_after(game, seconds=10.0)
    assert gained == pytest.approx(expected * 10.0, rel=0.1)
    game.player.position = spot(game.world, -30.0, 28.0)          # out of range: back to the trickle
    assert game.world.sneak_xp_rate(game.player) == pytest.approx(0.75 * factor(1, 0.0))


def test_sneaking_behind_an_npc_teaches_a_little_in_front_a_lot(game):
    barnaby = game.world.find_npc("Shopkeeper Barnaby")
    isolate(game, barnaby)
    barnaby.state = AIState.IDLE
    barnaby.position = spot(game.world, -45.0, -22.0)             # an empty corner of town
    barnaby.facing_yaw = 0.0                                   # looking -z
    game.toggle_crouch()
    game.rng.random = lambda: 0.0                              # never spotted
    game.player.position = spot(game.world, -45.0, -16.0)         # behind him, out of earshot of crouched steps
    behind = sneak_xp_after(game)
    game.player.position = spot(game.world, -45.0, -28.0)         # right in front of him
    in_front = sneak_xp_after(game)
    assert 0 < behind < in_front
    assert behind == pytest.approx(3.0 * factor(1, vigil("merchant")) * 2.0, rel=0.2)
    assert in_front > 2 * behind                               # the per-second stream plus the passed rolls


def test_guards_teach_more_and_elite_sneaks_learn_less(game):
    vance = watcher(game)
    isolate(game, vance)
    vance.position = spot(game.world, -45.0, -22.0)              # away from the other guards
    vance.facing_yaw = 0.0                                     # back turned (looking -z)
    game.player.position = spot(game.world, -45.0, -16.0)         # out of earshot, in range
    game.toggle_crouch()
    game.rng.random = lambda: 0.0
    novice = sneak_xp_after(game)
    game.player.skills.skills[SkillType.SNEAK]["level"] = 50
    elite = sneak_xp_after(game)
    assert novice == pytest.approx(3.0 * factor(1, vigil("guard")) * 2.0, rel=0.2)   # behind a guard
    assert novice > 3.0 * factor(1, vigil("merchant")) * 2.0                           # ... more than behind a shopkeeper
    assert 0 < elite < novice / 3                              # what is easy for you teaches little


def test_rolls_happen_once_per_second(game, monkeypatch):
    watcher(game)
    game.toggle_crouch()
    rolls = []
    import unprepared.world.world as world_module
    monkeypatch.setattr(world_module, "sneak_chance", lambda *a, **k: rolls.append(1) or 1.0)   # count; all succeed
    for _ in range(50):
        game.update(0.05, MoveIntent(left=True, right=True))   # 2.5 seconds
    assert len(rolls) == 2


def test_failed_sneak_in_front_of_a_guard_is_noticed(game):
    vance = watcher(game)
    game.toggle_crouch()
    game.rng.random = lambda: 0.99                             # every roll fails
    for _ in range(21):
        game.update(0.05, MoveIntent(left=True, right=True))   # first roll after one second
    assert vance.suspicion_meter >= 35.0
    assert said(game.messages.messages, "skulking")
    assert game.player.skills.skills[SkillType.SNEAK]["xp"] >= 1.0       # failing still teaches a little
    for _ in range(80):
        game.update(0.05, MoveIntent(left=True, right=True))
    assert game.player.gold == STARTING_GOLD                                       # his warning bought a few seconds
    for _ in range(80):
        game.update(0.05, MoveIntent(left=True, right=True))
    assert said(game.messages.messages, "halt")                           # the alarm went up ...
    for _ in range(int(game.settings.justice.charge_delay / 0.05) + 20):
        game.update(0.05, MoveIntent(left=True, right=True))
    assert game.player.gold < STARTING_GOLD                                         # ... and, caught up with, a fine


def test_successful_sneak_hides_you(game):
    vance = watcher(game)
    game.toggle_crouch()
    game.rng.random = lambda: 0.0                              # every roll succeeds
    for _ in range(60):
        game.update(0.05, MoveIntent(left=True, right=True))   # 3 seconds in plain view
    assert vance.state is not AIState.ALERT_GUARDS and vance.suspicion_meter < 50.0
    assert game.player.skills.skills[SkillType.SNEAK]["xp"] >= 8.0
    assert vance.sneak_unnoticed_until > 0


def test_high_skill_is_detected_less_often(tmp_path):
    """Statistically: a level-1 sneak is caught far more often than an elite one."""
    def alarms(level):
        caught = 0
        for seed in range(10):
            g = Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=seed), save_file=str(tmp_path / "s.json"))
            g.player.skills.skills[SkillType.SNEAK]["level"] = level
            vance = watcher(g)
            g.toggle_crouch()
            for _ in range(240):                                   # 12 s: past the warning grace
                g.update(0.05, MoveIntent(left=True, right=True))
            caught += (vance.is_alarmed or g.player.gold < STARTING_GOLD)      # alarmed, or already fined for it
        return caught
    low, high = alarms(1), alarms(45)
    assert low >= 8 and high <= 2 and low > high, (low, high)


def test_crouching_behind_low_cover_hides_you(game):
    """Sight is checked against the chest: standing you show over the well, crouched you do not."""
    vance = watcher(game)
    isolate(game, vance)
    vance.position = spot(game.world, 0.0, -3.0)                  # the well (1 unit high) between you
    game.player.position = spot(game.world, 0.0, 1.5)
    vance.turn_towards(game.player.position)
    blockers = game.world.sight_blockers()
    from unprepared.entities import NPC
    assert NPC.chest_height(game.player) == pytest.approx(1.35)
    assert vance.can_see(game.player.position, blockers, NPC.chest_height(game.player))
    game.toggle_crouch()
    for _ in range(30):
        game.update(0.05)
    assert NPC.chest_height(game.player) == pytest.approx(0.7)
    assert not vance.can_see(game.player.position, blockers, NPC.chest_height(game.player))
    game.player.position = spot(game.world, 3.0, 1.5)               # step out from behind it
    vance.turn_towards(game.player.position)
    assert vance.can_see(game.player.position, blockers, NPC.chest_height(game.player))
