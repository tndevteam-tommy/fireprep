"""Guard behaviour off the routine: investigating noises, tailing skulkers, guarding restricted areas."""
import pytest

from tests.helpers import spot
from tests.conftest import said
from unprepared.config import GameSettings
from unprepared.entities import AIState
from unprepared.game import Game
from unprepared.systems import MoveIntent, TransferOutcome
from unprepared.world.levels.town import STARTING_GOLD


@pytest.fixture
def game(tmp_path):
    return Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=9, max_log_messages=400),
                save_file=str(tmp_path / "s.json"))


def posted(game, name="Guard Orrin"):
    """A guard standing post in the street, looking east (away from the well)."""
    g = game.world.find_npc(name)
    g.path, g.current_task, g.schedule = [], None, []
    g.state = AIState.GUARDING
    g.position = spot(game.world, 20.0, 0.0)
    g.facing_yaw = 90.0
    for other in game.world.npcs:                                  # nobody else about to muddy the water
        if other is not g and not other.is_asleep:
            other.path, other.current_task, other.schedule = [], None, []
            other.position = spot(game.world, 46.0, 28.0)
            other.state = AIState.IDLE
    return g


def test_guard_investigates_a_noise_behind_him(game):
    orrin = posted(game)
    game.world.simulated_hour = 23                               # night: walking about unlit is suspicious
    game.player.position = spot(game.world, 12.0, 1.0)              # 8 units behind him, out in the open
    game.player.has_lit_torch = False
    for _ in range(10):
        game.update(0.1, MoveIntent(left=True, right=True))      # footsteps
    assert orrin.state is AIState.INVESTIGATING and orrin.goal is not None
    assert orrin.goal.distance_to(spot(game.world, 12.0, 0.0)) < 1.5
    assert said(game.messages.messages, "investigate")
    for _ in range(150):
        game.update(0.1)                                         # walks over, finds nothing, goes back to guarding
    assert orrin.position.distance_to(spot(game.world, 12.0, 1.0)) < 1.5 or orrin.state is AIState.GUARDING


def test_guard_turns_to_watch_and_follows_a_skulker(game):
    orrin = posted(game)
    orrin.facing_yaw = 270.0                                     # looking west, at the player
    game.player.position = spot(game.world, 15.0, 2.5)
    game.rng.random = lambda: 0.99                               # sneak rolls fail
    game.toggle_crouch()
    states, turned = [], False
    for _ in range(40):
        game.update(0.1, MoveIntent(left=True, right=True))
        states.append(orrin.state)
        turned = turned or (orrin.suspicion_meter >= 50.0 and orrin.faces(game.player.position))
    assert said(game.messages.messages, "skulking")
    assert turned                                                # turned to watch the skulker ...
    assert AIState.FOLLOWING in states                           # ... tailed them when they kept it up ...
    assert game.player.gold == STARTING_GOLD                              # ... his warning bought a few seconds' grace ...
    for _ in range(100):                                         # (the alarm, then the chase's own delay)
        game.update(0.1, MoveIntent(left=True, right=True))
    assert game.player.gold < STARTING_GOLD                               # ... and finally fined them for it


def test_guards_object_to_the_player_in_the_guard_house(game):
    orrin = posted(game)
    orrin.position = spot(game.world, 49.0, -47.0)                # inside the guard house
    orrin.facing_yaw = 180.0
    game.player.position = spot(game.world, 49.0, -44.5)             # so is the player, standing upright, by day
    for _ in range(15):
        game.update(0.1)
    assert said(game.messages.messages, "trespass")
    assert orrin.suspicion_meter > 0


def test_citizens_object_to_the_player_in_their_home(game):
    barnaby = game.world.find_npc("Shopkeeper Barnaby")         # at breakfast at home, facing north
    game.player.position = spot(game.world, -71.5, 9.0)             # in his house, across the table from him
    for _ in range(15):
        game.update(0.1)
    assert said(game.messages.messages, "trespass")
    assert barnaby.suspicion_meter > 0


def test_fence_watches_like_a_merchant_but_only_a_theft_from_him_brings_the_watch(game):
    """A fence is as wary as any shopkeeper: creeping about his den gets you told to stand up
    (and shown the door), but skulking never makes him call the guards - robbing him does."""
    jae = game.world.find_npc("Jae the Fence")
    game.player.position = spot(game.world, -10.0, 10.0)
    game.toggle_crouch()
    for _ in range(40):
        game.update(0.1, MoveIntent(left=True, right=True))
    assert jae.faces(game.player.position)                        # he has turned to keep an eye on the creeper
    assert jae.suspicion_meter > 0.0 and not jae.is_alarmed       # wary ...
    assert jae.state is AIState.WAITING and jae.waiting_for == "skulking"     # ... "stand up straight in my den"
    from tests.conftest import said
    assert said(game.messages.messages, "shop_skulk")
    assert not any(g.state is AIState.RESPONDING for g in game.world.guards())   # no cry for the watch
    game.toggle_crouch()                                          # stand up: he lets it go
    game.update(0.1)
    assert jae.state is not AIState.WAITING
    game.rng.random = lambda: 0.99                                # every roll fails: clumsy, and he feels it
    game.toggle_crouch()
    game.player.position = spot(game.world, -10.0, 10.5)
    game.player.rotation_y = 180.0
    game.player.rotation_x = -20.0
    game.update(0.0)
    game.interact()                                               # crouched: pick his pockets, in his face
    assert game.mode.value == "pickpocket"
    assert not jae.is_alarmed                                     # the victim is no bystander: nobody else saw the reach
    game.cursor_down()
    assert game.transfer_confirm().outcome is TransferOutcome.CAUGHT      # ... but he feels the hand in his pocket
    assert jae.state is AIState.REPORTING                         # theft from *him* is another matter


def test_every_fence_is_wary_not_just_jae(game):
    """The rule is the fence's trade, not his name: any fence watches and objects the same way."""
    from unprepared.entities import NPC
    w = game.world
    fence = NPC("Vekk the Fence", False, True, True, spot(game.world, -10.0, 12.0))
    fence.trade_availability, fence.role_tag, fence.state = "always", "fence", AIState.SHOPKEEPING
    jae = w.find_npc("Jae the Fence")
    jae.position = spot(game.world, -46.0, 28.0); jae.schedule, jae.current_task, jae.path, jae.state = [], None, [], AIState.IDLE
    w.npcs.append(fence)
    import dataclasses
    w.level = dataclasses.replace(w.level, areas=tuple(
        a if a.name != "the hideout" else dataclasses.replace(a, owner=fence.name, trusted_by=(fence.name,)) for a in w.level.areas))
    game.player.position = spot(game.world, -10.0, 10.0)
    game.toggle_crouch()
    for _ in range(40):
        game.update(0.1, MoveIntent(left=True, right=True))
    assert fence.suspicion_meter > 0.0 and fence.state is AIState.WAITING and fence.waiting_for == "skulking"
    assert not fence.is_alarmed


def test_a_guards_scolding_turns_the_player_to_face_him(game):
    orrin = posted(game)
    orrin.facing_yaw = 270.0                                     # looking west, at the player
    game.player.position = spot(game.world, 15.0, 2.5)
    game.player.rotation_y, game.player.rotation_x = 0.0, -25.0  # looking away, down
    game.rng.random = lambda: 0.99
    game.toggle_crouch()
    for _ in range(25):
        game.update(0.1, MoveIntent(left=True, right=True))
        if said(game.messages.messages, "skulking"):
            break
    assert said(game.messages.messages, "skulking")
    assert game.player.frozen                                                            # a moment's attention ...
    for _ in range(15):
        game.update(0.1)
    import math
    expected = math.degrees(math.atan2(20.0 - 15.0, -(0.0 - 2.5)))                        # ... swung round towards Orrin
    assert game.player.rotation_y == pytest.approx(expected)
    assert game.player.rotation_x == pytest.approx(game.player.pitch_towards(orrin.position)) and game.player.rotation_x > 0.0
    # (crouched: up at his face; standing, the view would be level)
