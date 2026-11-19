"""Headless tests of the full game flow through the ``Game`` action API."""
import pytest

from tests.conftest import said
from unprepared.config import GameSettings, LockpickSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState, Item, SkillType  # noqa: F401
from unprepared.game import Game, GameMode
from unprepared.systems import LockpickOutcome, MoveIntent, Side, TransferOutcome
from tests.helpers import spot, teleport
from unprepared.world.levels import town
from unprepared.world.levels.town import SHOP_COUNTER
from unprepared.world.levels.town import STARTING_GOLD

# where a test stands: in front of the store's (north-side) door, of the hideout's (south-side) one
SHOP_DOOR_FRONT = (-13.5, -3.0)          # three paces south of the shop door, looking north (-z) at it
HIDEOUT_DOOR_FRONT = (-11.0, 4.5)        # three paces north of the hideout door, looking south (+z) at it
WEST_SHELF_FRONT = (-16.0, -13.0)        # behind the counter, looking north at the west shelf
FAR_AWAY = (110.0, 10.0)                 # out in the country: where tests park the people in the way


# Talking works from ~5 units; pickpocketing needs the player within 2.5 (feet to feet).
TALK_DIST = 3.5
STEAL_DIST = 2.0
LOOK_DOWN = -20.0


def open_shop(game):
    """Put Barnaby behind the counter, shop open (as during business hours)."""
    barnaby = game.world.find_npc("Shopkeeper Barnaby")
    barnaby.position = game.world.spot(*SHOP_COUNTER)
    barnaby.state = AIState.SHOPKEEPING
    barnaby.facing_yaw = 180.0
    barnaby.path = []
    game.world.find_door("shop_door").unlock_and_open()
    return barnaby


def face_player(npc, game):
    npc.turn_towards(game.player.position)


def face_jae(game, dist=TALK_DIST):
    jae = game.world.find_npc("Jae the Fence")
    teleport(game, jae.position.x, jae.position.z - dist, yaw=180.0, pitch=LOOK_DOWN)
    return jae


def crouch(game):
    game.toggle_crouch()
    for _ in range(30):
        game.update(0.05)           # let the camera settle at crouch height
    return game


def send_guards_away(game):
    """Park every guard in the far corner so nobody sees what the test is up to."""
    for g in game.world.guards():
        g.position = spot(game.world, *FAR_AWAY)
        g.path = []
        g.current_task = None
        g.schedule = []
        g.facing_yaw = 0.0
    return game


class TestStartup:
    def test_initial_state(self, game):
        assert game.mode is GameMode.EXPLORE and game.running and game.in_explore_mode
        assert game.world.simulated_hour == 7
        assert game.player.position == game.world.spot(*town.PLAYER_SPAWN)
        assert game.messages.messages == ["Tutorial started: Explore merchants, trade wares, and open doors!"]
        assert game.mouse_captured and game.accepts_free_look
        assert game.target is None and game.interaction_prompt is None

    def test_snapshot_shape(self, game):
        snap = game.snapshot()
        assert snap["mode"] == "explore" and snap["hour"] == 7
        assert snap["player"]["inventory"] == [("iron_lockpick", 1, False), ("soft_shoes", 1, False), ("silver_goblet", 1, True)]
        assert [n["name"] for n in snap["npcs"]][:3] == ["Shopkeeper Barnaby", "Hilde the Innkeeper", "Gareth the Smith"]
        seat = game.world.spot(*town.BARNABY_TABLE_SEAT)
        assert snap["npcs"][0]["pos"] == pytest.approx((seat.x, seat.y, seat.z), abs=1e-6)

    def test_seeded_rng(self, tmp_path):
        a = Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=5), save_file=str(tmp_path / "a"))
        b = Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=5), save_file=str(tmp_path / "b"))
        assert a.rng.random() == b.rng.random()


class TestUpdate:
    def test_movement_and_time(self, game):
        game.update(0.5, MoveIntent(forward=True))
        assert game.player.position.z == pytest.approx(town.PLAYER_SPAWN[2] - 2.0)
        assert game.movement.is_moving
        assert game.world.clock.minute_accumulator == pytest.approx(1.25)

    def test_walls_block_the_player(self, game):
        teleport(game, 38.0, -4.0)                            # on Market Street, at the foot of East Lane
        for _ in range(400):
            game.update(0.1, MoveIntent(forward=True))       # up the hill to the north town wall (inner face z = -61)
        r = game.settings.player.collision_radius
        assert town.TOWN_Z1 + 1.0 + r - 1e-9 <= game.player.position.z < town.TOWN_Z1 + 1.0 + r + 0.2

    def test_hour_passes_and_npcs_move(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        for _ in range(24):
            game.update(1.0)
        assert game.world.simulated_hour == 8
        assert "Time Update: 08:00 hours" in game.messages.messages
        assert barnaby.state is AIState.TRAVELING

    def test_look(self, game):
        game.look(100, 0)
        assert game.player.rotation_y == pytest.approx(15.0)

    def test_escape_in_explore_is_the_apps_business(self, game):
        """Nothing to back out of: ESC returns False (the app opens its pause menu) and the game runs on."""
        assert game.escape() is False
        assert game.running and game.mouse_captured

    def test_request_quit(self, game):
        game.request_quit()
        assert not game.running


class TestToggles:
    def test_torch(self, game):
        assert game.toggle_torch() is True and game.player.has_lit_torch
        assert game.messages.messages[-1] == "Torch is now LIT."
        assert game.toggle_torch() is False
        assert game.messages.messages[-1] == "Torch is now EXTINGUISHED."

    def test_crouch(self, game):
        assert game.toggle_crouch() is True
        assert game.messages.messages[-1] == "Stance: Crouching."
        game.toggle_crouch()
        assert game.messages.messages[-1] == "Stance: Standing."

    def test_crouching_lowers_the_camera_smoothly(self, game):
        assert game.player.eye_position.y - game.player.position.y == pytest.approx(1.55)   # at the townsfolk's eye level (theirs: 1.6)
        game.toggle_crouch()
        game.update(0.05)
        assert 0.9 < game.player.current_eye_height < 1.55    # on its way down
        for _ in range(30):
            game.update(0.05)
        assert game.player.current_eye_height == pytest.approx(0.9)
        game.toggle_crouch()
        for _ in range(30):
            game.update(0.05)
        assert game.player.current_eye_height == pytest.approx(1.55)


class TestDoors:
    def test_open_and_close_hideout_door(self, game):
        teleport(game, *HIDEOUT_DOOR_FRONT, yaw=180.0)     # looking +z at the hideout door
        assert game.interaction_prompt == "[E] Open Door"
        count = len(game.messages.messages)
        game.interact()
        door = game.world.find_door("hideout_door")
        assert door.is_open and len(game.messages.messages) == count      # no door chatter in the log
        game.update(0.0)
        assert game.interaction_prompt == "[E] Close Door"
        game.interact()
        assert not door.is_open and len(game.messages.messages) == count

    def test_closed_door_blocks_open_door_lets_through(self, game):
        teleport(game, -12.0, 4.5, yaw=180.0)
        for _ in range(40):
            game.update(0.05, MoveIntent(forward=True))
        assert game.player.position.z < 7.5 - 0.3          # stopped at the closed leaf (z 7.4..7.6)
        teleport(game, -12.0, 4.5, yaw=180.0)
        assert game.interaction_prompt == "[E] Open Door"
        game.interact()
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True))
        assert game.player.position.z > 8.5                  # walked through into the hideout

    def test_locked_door_standing(self, game):
        teleport(game, *SHOP_DOOR_FRONT, yaw=0.0)        # looking -z at the shop door, with a lockpick, standing
        assert game.interaction_prompt == "[Locked]"
        game.interact()
        assert game.mode is GameMode.EXPLORE
        assert game.messages.messages[-1] == "Door is locked. Crouch to pick it, or find the key."

    def test_locked_door_crouched_without_tools(self, game):
        game.player.inventory = []
        crouch(game)
        teleport(game, *SHOP_DOOR_FRONT, yaw=0.0)
        assert game.interaction_prompt == "[Locked] (no lockpick)"
        game.interact()
        assert game.mode is GameMode.EXPLORE
        assert game.messages.messages[-1] == "Door is locked! You need a key or a lockpick."

    def test_unlock_with_key_then_open(self, game):
        game.player.inventory.append(Item("shop_key", "Key", is_key=True, key_door_id="shop_door"))
        teleport(game, *SHOP_DOOR_FRONT)
        assert game.interaction_prompt == "[E] Unlock with Key"
        game.interact()
        door = game.world.find_door("shop_door")
        assert not door.is_locked and not door.is_open            # unlocking does not open
        assert game.messages.messages[-1] == "Unlocked the door with your key."
        game.update(0.0)
        assert game.interaction_prompt == "[E] Open Door"
        game.interact()
        assert door.is_open

    def test_interact_with_no_target_is_noop(self, game):
        before = game.snapshot()
        game.interact()
        assert game.snapshot() == before


class TestLockpickingFlow:
    def test_full_flow(self, game):
        send_guards_away(game)
        game.player.inventory[0].quantity = 2
        crouch(game)
        teleport(game, *SHOP_DOOR_FRONT)
        assert game.interaction_prompt == "[E] Pick Lock"
        game.interact()
        assert game.mode is GameMode.LOCKPICK and game.in_lockpicking and not game.in_explore_mode
        assert "Started lockpicking the door (a Simple lock, 80% at your skill)." in game.messages.messages
        assert game.mouse_captured and not game.accepts_free_look
        assert game.lockpicking.pick_angle == 0.0
        assert game.lockpicking.tier_name == "Simple" and game.lockpicking.set_chance == pytest.approx(0.80, abs=0.01)

        # input is ignored while picking (look, move, interact, world actions)
        game.look(500, 0)
        game.update(1.0, MoveIntent(forward=True))
        game.interact()
        assert game.toggle_torch() is None and game.toggle_crouch() is None
        assert game.save_game() is False and game.load_game() is False and game.wait() is False
        assert game.player.rotation_y == 0.0 and game.player.position.z == SHOP_DOOR_FRONT[1]
        assert not game.movement.is_moving and game.in_lockpicking
        assert not [m for m in game.messages.messages if "witnessed" in m]   # no second crime report

        # miss: the pick snaps and the attempt ends; one pick left
        game.lockpicking.sweet_spot = (game.lockpicking.pick_angle + 180.0) % 360.0
        assert game.lockpick_attempt() is LockpickOutcome.SNAPPED
        assert not game.in_lockpicking and game.player.inventory[0].quantity == 1
        # a fumble teaches a little - less on a lock far below a master's (a Simple lock: challenge -10)
        assert game.player.skills.skills[SkillType.LOCKPICKING]["xp"] == pytest.approx(10.0 * 2 ** ((-10 - 1) / 12))

        # try again: rotate onto the sweet spot and open it
        game.update(0.0)
        game.interact()
        assert game.in_lockpicking
        steps = round(game.lockpicking.sweet_spot / 10.0) % 36
        game.lockpicking.sweet_spot = steps * 10.0
        for _ in range(steps):
            game.lockpick_rotate(+1)
        assert game.lockpicking.is_aligned
        assert game.lockpick_attempt() is LockpickOutcome.SUCCESS
        assert game.mode is GameMode.EXPLORE
        door = game.world.find_door("shop_door")
        assert not door.is_locked and door.is_open

    def test_losing_the_last_pick_ends_the_minigame(self, game):
        send_guards_away(game)                       # nobody around to fine us and spin us round
        crouch(game)
        teleport(game, *SHOP_DOOR_FRONT)
        game.interact()
        game.lockpicking.sweet_spot = 180.0
        assert game.lockpick_attempt() is LockpickOutcome.OUT_OF_PICKS
        assert game.mode is GameMode.EXPLORE and not game.player.has_lockpick()
        game.update(0.0)
        assert game.interaction_prompt == "[Locked] (no lockpick)"

    def test_each_attempt_is_different(self, game):
        crouch(game)
        teleport(game, *SHOP_DOOR_FRONT)
        game.interact()
        first = game.lockpicking.sweet_spot
        game.escape()
        game.interact()
        assert game.lockpicking.sweet_spot != first and game.lockpicking.pick_angle == 0.0

    def test_escape_cancels(self, game):
        crouch(game)
        teleport(game, *SHOP_DOOR_FRONT)
        game.interact()
        game.escape()
        assert game.mode is GameMode.EXPLORE and game.running

    def test_lockpicking_is_witnessed_only_from_the_front(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        vance.position = spot(game.world, -15.0, -1.0)
        vance.path = []
        vance.facing_yaw = 180.0                      # looking away (+z) from the door
        crouch(game)
        teleport(game, *SHOP_DOOR_FRONT)
        game.interact()
        assert vance.state is AIState.PATROL and game.player.crimes_witnessed == 0
        game.escape()
        face_player(vance, game)
        game.update(0.0)
        game.interact()
        assert vance.state is AIState.ALERT_GUARDS and game.player.crimes_witnessed == 1

    def test_lockpick_actions_ignored_outside_mode(self, game):
        game.lockpick_rotate(1)
        assert game.lockpicking.pick_angle == 0.0
        assert game.lockpick_attempt() is None


class TestTradeFlow:
    def test_open_and_close(self, game):
        jae = face_jae(game)
        assert game.interaction_prompt == "[E] Talk / Trade with Jae the Fence"
        game.interact()
        assert game.mode is GameMode.TRADE and game.in_trade_menu and game.active_merchant is jae
        assert not game.mouse_captured
        assert game.messages.messages[-1].startswith("(Jae the Fence): ") and said(game.messages.messages[-1:], "greet_fence")
        assert game.transfer.side is Side.LEFT and game.transfer.selection[Side.LEFT] == 0
        game.interact()       # [E] closes
        assert game.mode is GameMode.EXPLORE and game.mouse_captured and game.transfer is None
        game.interact()
        game.escape()
        assert game.mode is GameMode.EXPLORE and game.running

    def test_tab_then_sell_stolen_goods_to_fence(self, game):
        face_jae(game)
        game.interact()
        game.transfer_switch_side()                       # Tab: player's column
        game.cursor_up()                   # wraps to the goblet
        assert game.transfer.selected_item().id == "silver_goblet"
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.MOVED and result.price == 15.0    # a quarter of the worth, halved for hot goods
        assert game.player.gold == STARTING_GOLD + 15.0

    def test_buy_lockpick_stacks_with_existing(self, game):
        face_jae(game)
        game.interact()
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED
        assert game.player.gold == STARTING_GOLD - 25.0          # a 20g pick at the fence's asking price (+25%)
        picks = [i for i in game.player.inventory if i.is_lockpick]
        assert [(i.id, i.quantity) for i in picks] == [("iron_lockpick", 1), ("lockpick_pack", 1)]

    def test_trade_blocks_look_movement_and_world_actions(self, game):
        jae = face_jae(game)
        game.interact()
        game.look(100, 100)
        game.update(1.0, MoveIntent(forward=True, sprint=True))
        assert game.player.rotation_y == 180.0 and game.player.position.z == jae.position.z - TALK_DIST
        assert not game.movement.is_sprinting
        assert game.toggle_torch() is None and not game.player.has_lit_torch
        assert game.toggle_crouch() is None and game.wait() is False
        assert game.save_game() is False and game.load_game() is False
        assert game.mode is GameMode.TRADE

    def test_merchant_refuses_stolen_and_closes_menu(self, game):
        barnaby = open_shop(game)
        vance = game.world.find_npc("Guard Captain Vance")
        vance.position, vance.path = spot(game.world, -5.0, 0.0), []          # on his round, within earshot of the shop
        teleport(game, barnaby.position.x, barnaby.position.z + TALK_DIST, yaw=0.0, pitch=LOOK_DOWN)
        assert game.interaction_prompt == "[E] Talk / Trade with Shopkeeper Barnaby"
        game.interact()
        rows = [i.id for i in game.transfer.rows(Side.LEFT)]
        assert rows[:3] == ["bread", "apple_1", "torch_item"] and "shop_key" not in rows         # keys hidden
        assert "candlestick" in rows and "pickaxe" in rows                                         # shelves included
        game.transfer_switch_side()
        game.cursor_up()               # goblet
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.REFUSED_STOLEN
        assert game.mode is GameMode.EXPLORE and game.mouse_captured
        assert barnaby.state is AIState.REPORTING
        assert said(game.messages.messages, "stolen_goods", item="Silver Goblet")
        assert said(game.messages.messages[-1:], "cry_response")               # his shout brings the guards
        assert vance.state is AIState.RESPONDING and [m.crime for m in vance.crime_memories] == ["Fencing stolen goods"]

    def test_shop_closed_outside_business_hours(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")      # 07:00, eating breakfast at home
        teleport(game, barnaby.position.x, barnaby.position.z - 2.0, yaw=180.0, pitch=LOOK_DOWN)
        assert game.interaction_prompt == "[E] Talk to Shopkeeper Barnaby"
        game.interact()
        assert game.mode is GameMode.EXPLORE
        assert said(game.messages.messages[-1:], "greet_closed")

    def test_talking_to_guard_does_not_open_trade(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        teleport(game, vance.position.x, vance.position.z - TALK_DIST, yaw=180.0, pitch=LOOK_DOWN)
        assert game.interaction_prompt == "[E] Talk to Guard Captain Vance"
        game.interact()
        assert game.mode is GameMode.EXPLORE
        assert game.messages.messages[-1].startswith("(Guard Captain Vance): ") and said(game.messages.messages[-1:], "greet_guard")

    def test_transfer_actions_ignored_outside_mode(self, game):
        game.cursor_down()
        game.transfer_switch_side()
        assert game.transfer_confirm() is None


class TestPickpocketFlow:
    def sneak_up_on(self, game, npc, dist=STEAL_DIST):
        crouch(game)
        teleport(game, npc.position.x, npc.position.z - dist, yaw=180.0, pitch=LOOK_DOWN)

    def test_crouched_e_opens_the_pockets(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SLEEPING
        self.sneak_up_on(game, jae)
        game.rng.random = lambda: 0.0
        assert game.interaction_prompt == "[E] Pickpocket Jae the Fence"
        game.interact()
        assert game.mode is GameMode.PICKPOCKET and not game.mouse_captured
        assert game.transfer.policy.title == "PICKPOCKETING: Jae the Fence"
        # his purse is the first row, then his pockets
        assert [i.id for i in game.transfer.rows(Side.LEFT)] == ["gold_coins", "hideout_door_key", "lockpick_pack", "gold_ring", "black_hood", "dried_meat"]
        game.cursor_down()
        game.cursor_down()
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED
        assert game.player.inventory[-1].id == "lockpick_pack" and game.player.inventory[-1].is_stolen
        assert jae.inventory[1].id == "lockpick_pack" and jae.inventory[1].quantity == 2
        assert game.messages.messages[-1] == "Pickpocketed Master Lockpick from Jae the Fence!"

    def test_lifting_coins_from_a_purse(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SLEEPING
        self.sneak_up_on(game, jae)
        game.rng.random = lambda: 0.0
        game.interact()
        purse = game.transfer.rows(Side.LEFT)[0]
        assert purse.quantity == 600 and "grab up to 20" in game.transfer.row_text(purse, Side.LEFT)
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED
        assert game.player.gold == STARTING_GOLD + 20.0 and jae.gold == 580.0
        assert game.transfer.rows(Side.LEFT)[0].quantity == 580
        assert game.messages.messages[-1] == "Lifted 20 gold from Jae the Fence's purse!"
        jae.gold = 0.0
        assert [i.id for i in game.transfer.rows(Side.LEFT)][0] != "gold_coins"      # empty purse: no row

    def test_a_skilled_hand_takes_a_bigger_handful(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SLEEPING
        self.sneak_up_on(game, jae)
        game.rng.random = lambda: 0.0
        game.player.skills.skills[SkillType.PICKPOCKET]["level"] = 11                 # +2 coins per level above 1
        game.interact()
        purse = game.transfer.rows(Side.LEFT)[0]
        assert game.transfer.policy.purse_grab == 40 and "grab up to 40" in game.transfer.row_text(purse, Side.LEFT)
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED
        assert game.player.gold == STARTING_GOLD + 40.0 and jae.gold == 560.0

    def test_plant_an_item(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SLEEPING
        self.sneak_up_on(game, jae)
        game.interact()
        game.transfer_switch_side()
        game.cursor_up()              # the goblet
        game.transfer_confirm()                      # planting never fails
        assert jae.inventory[-1].id == "silver_goblet"
        assert game.messages.messages[-1] == "Planted Silver Goblet on Jae the Fence."
        game.escape()
        assert game.mode is GameMode.EXPLORE

    def test_too_far_to_pickpocket(self, game):
        jae = game.world.find_npc("Jae the Fence")
        self.sneak_up_on(game, jae, dist=3.5)
        assert game.interaction_prompt == "[Sneak closer to pickpocket Jae the Fence]"
        game.interact()
        assert game.mode is GameMode.EXPLORE
        assert game.messages.messages[-1] == "Sneak closer to pick Jae the Fence's pockets."

    def _alone_with(self, game, victim):
        """Nobody but ``victim`` is anywhere near the player (the other guards are parked far off)."""
        victim.path = []
        for g in game.world.guards():
            if g is not victim:
                g.position = spot(game.world, *FAR_AWAY); g.path, g.current_task, g.schedule = [], None, []
        self.sneak_up_on(game, victim)
        face_player(victim, game)

    def test_the_victim_is_not_a_bystander(self, game):
        """The onlookers' roll is made against the people *around*: with nobody else about, a clumsy
        reach into an awake guard's pockets is not a crime he witnesses - only the second roll, whether
        he feels the hand himself, can catch the player."""
        vance = game.world.find_npc("Guard Captain Vance")
        self._alone_with(game, vance)
        rolls = iter((0.99, 0.99, 0.0, 0.99, 0.0))    # reach: onlookers 0.99 (clumsy); lifts: onlookers 0.99, victim 0.0 (deft)
        game.rng.random = lambda: next(rolls)
        game.interact()
        assert game.mode is GameMode.PICKPOCKET and vance.state is not AIState.ALERT_GUARDS
        game.cursor_down()                            # skip his purse
        game.transfer_confirm()                       # steal the key
        game.transfer_confirm()                       # steal a ration
        assert vance.crimes_witnessed == 0 and game.player.crimes_witnessed == 0 and vance.crime_memories == []
        assert not any("witnessed" in m for m in game.messages.messages)
        assert game.player.has_key_for("guardhouse_door")

    def test_a_fumble_is_one_crime_declared_once(self, game):
        """A fumbled lift wakes the victim's wrath exactly once: one alarm line, one witnessing, one
        memory per guard - not one for the bystanders' roll, one for the victim and one for the cry."""
        vance = game.world.find_npc("Guard Captain Vance")
        self._alone_with(game, vance)
        rolls = iter((0.0, 0.0, 0.99))                # reach: deft; lift: onlookers deft, victim feels it (0.99 fumbles)
        game.rng.random = lambda: next(rolls)
        game.interact()
        game.cursor_down()
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.CAUGHT and game.mode is GameMode.EXPLORE
        assert vance.state is AIState.ALERT_GUARDS and vance.crimes_witnessed == 1 and game.player.crimes_witnessed == 1
        assert len(vance.crime_memories) == 1 and vance.crime_memories[0].witnessed
        assert vance.crime_memories[0].crime == "Pickpocketing"
        assert [m for m in game.messages.messages if "A thief! Hands where I can see them!" in m].__len__() == 1
        assert not any("witnessed Pickpocketing" in m for m in game.messages.messages)

    def test_an_onlooker_already_shouting_does_not_shout_again(self, game):
        """An onlooker who caught the reach keeps seeing the lifts but says his piece once; a second
        onlooker who only sees a later lift still raises his own alarm (legitimate reports survive)."""
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SLEEPING
        self._alone_with(game, jae)
        vance, orrin = game.world.find_npc("Guard Captain Vance"), game.world.find_npc("Guard Orrin")
        vance.position = spot(game.world, game.player.position.x + 2.0, game.player.position.z)
        face_player(vance, game)
        orrin.position = spot(game.world, game.player.position.x - 2.0, game.player.position.z)
        orrin.facing_yaw = (orrin.facing_yaw + 180.0) % 360.0           # back turned for now
        orrin.turn_towards(Vector3D(orrin.position.x - 5.0, 0.0, orrin.position.z))
        rolls = iter((0.99, 0.99, 0.0, 0.99, 0.0))    # reach: clumsy; lifts: clumsy, victim never feels it
        game.rng.random = lambda: next(rolls)
        game.interact()                               # Vance sees the clumsy reach
        assert vance.state is AIState.ALERT_GUARDS and vance.crimes_witnessed == 1
        assert len(vance.crime_memories) == 1
        game.cursor_down()
        game.transfer_confirm()                       # another clumsy lift: Vance sees it again but says nothing new
        assert vance.crimes_witnessed == 1 and len(vance.crime_memories) == 1 and game.player.crimes_witnessed == 1
        assert len([m for m in game.messages.messages if "Vance witnessed" in m]) == 1
        face_player(orrin, game)                      # Orrin turns round in time for the next one
        game.transfer_confirm()
        assert orrin.state is AIState.ALERT_GUARDS and orrin.crimes_witnessed == 1
        assert len(orrin.crime_memories) == 1 and orrin.crime_memories[0].witnessed
        assert game.player.crimes_witnessed == 2
        assert len([m for m in game.messages.messages if "Orrin witnessed" in m]) == 1

    def test_the_same_kind_of_crime_later_is_a_new_crime(self, game):
        """The de-duplication is about one deed, not one day: a guard who heard of a pickpocketing
        in the morning is sent running again for another in the afternoon."""
        vance = game.world.find_npc("Guard Captain Vance")
        w = game.world
        where = Vector3D(0.0, 0.0, 0.0)
        w.declare_crime("Pickpocketing", where, witnessed=False, reporter="Pell")
        assert len(vance.crime_memories) == 1 and vance.state is AIState.RESPONDING
        vance.resume_routine(w)
        w.declare_crime("Pickpocketing", where, witnessed=False, reporter="Pell")      # the same deed, moments later
        assert len(vance.crime_memories) == 1 and vance.state is not AIState.RESPONDING
        w.elapsed += 600.0
        w.declare_crime("Pickpocketing", where, witnessed=False, reporter="Tamsin")    # a new one, ten minutes on
        assert len(vance.crime_memories) == 2 and vance.state is AIState.RESPONDING

    def test_fumbled_pickpocket_wakes_victim_and_throws_you_out(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")      # asleep in his house? make it so
        barnaby.state = AIState.SLEEPING
        barnaby.position = game.world.spot(*town.BARNABY_BED)
        self.sneak_up_on(game, barnaby)
        assert game.world.area_of(game.player.position) == "Barnaby's house"
        game.rng.random = lambda: 0.999                            # force the roll to fail
        game.interact()
        result = game.transfer_confirm()
        assert result.outcome is TransferOutcome.CAUGHT and game.mode is GameMode.EXPLORE
        assert barnaby.state is AIState.REPORTING                  # woke up and raised the alarm
        msgs = game.messages.messages
        assert any("Get your hands off me" in m for m in msgs) and any("thrown out" in m for m in msgs)
        # the eviction plays out behind a fade to black: still inside, swinging round to face Barnaby
        assert game.in_transition and game.world.area_of(game.player.position) == "Barnaby's house"
        assert game.player.turn_target is not None and game.player.frozen
        for _ in range(10):
            game.update(0.1)
        assert game.world.area_of(game.player.position) != "Barnaby's house"   # thrown out
        assert game.player.position.z < 10.0 and game.fade_alpha > 0.0
        for _ in range(10):
            game.update(0.1)
        assert not game.in_transition and game.fade_alpha == 0.0
        assert 0 < game.player.skills.skills[SkillType.PICKPOCKET]["xp"] < 20.0       # a fumble still teaches a little
        assert game.player.has_key_for("shop_door") is False

    def test_keys_from_sleeping_barnaby(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        barnaby.state = AIState.SLEEPING
        barnaby.position = game.world.spot(*town.BARNABY_BED)       # in his bed
        self.sneak_up_on(game, barnaby)
        game.rng.random = lambda: 0.0                               # always succeed
        game.interact()
        game.cursor_down()                                 # skip his purse
        game.transfer_confirm()
        game.transfer_confirm()
        assert game.player.has_key_for("shop_door") and game.player.has_key_for("home_door")
        assert barnaby.crimes_witnessed == 0          # asleep


class TestContainers:
    def test_shoplifting_from_a_shelf(self, game):
        open_shop(game)
        teleport(game, *WEST_SHELF_FRONT, yaw=0.0)    # behind the counter, facing the west shelf on the north wall
        assert game.interaction_prompt == "[E] Search Shop Shelf"
        game.interact()
        assert game.mode is GameMode.CONTAINER and game.transfer.policy.title == "SHOP SHELF"
        game.cursor_down()                  # wine (stack of 3)
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED
        assert game.player.inventory[-1].id == "wine" and game.player.inventory[-1].is_stolen
        assert game.world.containers[0].inventory[1].quantity == 2
        assert game.messages.messages[-1] == "Stole Bottle of Wine from the shop shelf."
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        assert barnaby.state is AIState.REPORTING     # he saw it from the counter (facing the shop)
        assert any("witnessed Shoplifting" in m for m in game.messages.messages)

    def test_put_item_on_a_shelf(self, game):
        teleport(game, *WEST_SHELF_FRONT, yaw=0.0)
        game.interact()
        game.transfer_switch_side()
        game.cursor_down()                  # soft shoes - worn, so it asks first
        assert game.transfer_confirm() is None and game.confirmation is not None
        game.confirm_yes()
        assert game.world.containers[0].inventory[-1].id == "soft_shoes"
        assert game.messages.messages[-1] == "Put Soft Leather Boots in the shop shelf."


class TestLootAll:
    def test_x_empties_a_shelf_one_unit_at_a_time(self, game):
        """[X] sweeps the whole container across: every unit goes through the policy like a lift by
        hand (flagged stolen, rolled for, rewarded), the cursor stays valid, and the screen stays open."""
        open_shop(game)
        teleport(game, *WEST_SHELF_FRONT, yaw=0.0)
        game.interact()
        shelf = game.world.containers[0]
        units = sum(i.quantity for i in shelf.inventory)
        before = len(game.player.inventory)
        results = game.loot_all()
        assert results is not None and len(results) == units and all(r.outcome is TransferOutcome.MOVED for r in results)
        assert shelf.inventory == [] and game.mode is GameMode.CONTAINER
        assert game.transfer.selected_item() is None and game.transfer.rows(Side.LEFT) == []
        taken = game.player.inventory[before:]
        assert sum(i.quantity for i in taken) == units and all(i.is_stolen for i in taken)
        assert game.player.skills.skills[SkillType.STEALING]["xp"] > 0 or game.player.skills.level(SkillType.STEALING) > 1
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        assert barnaby.state is AIState.REPORTING
        assert len([m for m in game.messages.messages if "witnessed Shoplifting" in m]) <= 1   # one alarm, not one per unit

    def test_x_takes_back_your_own_things_from_the_lockbox(self, game):
        from unprepared.systems import justice
        justice.return_stolen_goods(game.world, game.player)          # as a charge does before the cell
        justice.imprison(game.world, game.player, 1)
        for _ in range(400):
            game.update(0.016)
            if not game.world.jail.transfer_pending:
                break
        justice.release(game.world, game.player)
        lockbox = game.world.find_container("jail_storage")
        game.transfer = None
        game.open_container(lockbox)
        mine = [i for i in lockbox.inventory if i.is_players]
        assert mine
        game.loot_all()
        assert all(i not in lockbox.inventory for i in mine)
        assert not any(i.is_stolen for i in game.player.inventory if i in mine)

    def test_x_does_nothing_in_pockets_or_trade(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SLEEPING
        TestPickpocketFlow.sneak_up_on(TestPickpocketFlow(), game, jae)
        game.rng.random = lambda: 0.0
        game.interact()
        assert game.mode is GameMode.PICKPOCKET
        pockets = list(jae.inventory)
        assert game.loot_all() is None and jae.inventory == pockets
        game.close_transfer()
        jae.state = AIState.SHOPKEEPING
        game.open_trade(jae)
        stock = [i.id for i in game.transfer.rows(Side.LEFT)]
        assert game.loot_all() is None and [i.id for i in game.transfer.rows(Side.LEFT)] == stock
        assert not game.transfer.policy.can_take_all

    def test_x_key_is_routed_by_the_controller(self, game):
        from unprepared.input import PygameInputController
        import pygame
        teleport(game, *WEST_SHELF_FRONT, yaw=0.0)
        game.interact()
        shelf = game.world.containers[0]
        PygameInputController().process_events(game, [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_x, mod=0, unicode="", scancode=0)])
        assert shelf.inventory == []


class TestCraft:
    def test_anvil_is_used_with_e_and_needs_ore_from_the_deposit(self, game):
        teleport(game, -62.0, -6.8, yaw=0.0)                # looking -z at the anvil under the smith's shed
        assert game.interaction_prompt == "[Anvil: needs 1 Iron Ore]"
        game.interact()
        assert game.messages.messages[-1].startswith("You need 1 Iron Ore")
        game.player.add_item(Item("pickaxe", "Pickaxe", value=35.0))
        game.player.add_item(Item("mallet", "Mallet", value=12.0, category="tool"))
        teleport(game, -59.5, -19.5, yaw=270.0)            # the deposit lies north of the shed
        assert game.interaction_prompt == "[E] Mine Iron Ore from the Ore Deposit"
        game.interact()
        game.interact()
        assert [i.quantity for i in game.player.inventory if i.id == "iron_ore"] == [2]
        teleport(game, -62.0, -6.8, yaw=0.0)
        assert game.interaction_prompt == "[E] Forge Iron Lockpicks (1 Iron Ore; up to 1 at your skill)"
        game.rng.random = lambda: 0.0                       # the forging succeeds ...
        game.interact()
        assert game.player.inventory[0].quantity == 2       # ... for a single pick at level 1
        game.rng.random = lambda: 0.99                      # ... and fails
        game.interact()
        assert game.player.inventory[0].quantity == 2
        assert not any(i.id == "iron_ore" for i in game.player.inventory)
        assert "ruined" in game.messages.messages[-1]

    def test_deposit_prompt_without_pickaxe(self, game):
        teleport(game, -59.5, -19.5, yaw=270.0)
        assert game.interaction_prompt == "[Ore Deposit: needs a pickaxe]"
        game.interact()
        assert game.messages.messages[-1] == "You need a pickaxe to work the ore deposit."


class TestSaveLoad:
    def test_roundtrip_through_game(self, game):
        game.toggle_torch()
        where = teleport(game, 3.0, 3.0, yaw=45.0)
        game.world.find_door("hideout_door").is_open = True
        assert game.save_game() is True
        assert game.messages.messages[-1] == "Game state saved successfully!"
        game.player.has_lit_torch = False
        game.player.position = Vector3D(0, 1, 3)
        game.player.rotation_y = 0.0
        game.world.find_door("hideout_door").is_open = False
        assert game.load_game() is True
        assert game.player.has_lit_torch and game.player.position == where
        assert game.player.rotation_y == 45.0 and game.world.find_door("hideout_door").is_open

    def test_load_without_save(self, game):
        assert game.load_game() is False
        assert game.messages.messages[-1] == "No save file found!"

    def test_loading_inside_a_screen_says_why_it_is_refused(self, game):
        """[F9] in a trade or the lock mini-game is refused, like [F5] - and, like it, says so."""
        assert game.save_game()
        game.open_container(game.world.find_container("jae_table"))
        assert game.load_game() is False and game.in_transfer
        assert game.messages.messages[-1] == "You cannot load right now. Close the screen first."


class TestFades:
    def test_a_move_no_fade_applied_gets_one_of_its_own(self, game):
        """Whatever fade is running when the world asks for a move, the move happens: one left
        pending when that fade has played out is given its own fade to black (a relocation must
        never stay in flight - the jail, curfews and charges all wait for it to land)."""
        from unprepared.game import ScreenFade
        w, p = game.world, game.player
        game.fade = ScreenFade(lambda: None, out_seconds=0.1, hold_seconds=0.0, in_seconds=0.1)   # not a relocation's fade
        w.relocate_player(Vector3D(-1.5, 0.0, 4.5), yaw=90.0, reason="evicted")
        assert game.pending_relocations and w.relocating
        for _ in range(200):
            game.update(0.02)
            if not w.relocating and game.fade is None:
                break
        assert not w.relocating and game.pending_relocations == [] and game.fade is None
        assert (p.position.x, p.position.z) == (-1.5, 4.5) and p.rotation_y == 90.0


class TestPerceptionIntegration:
    def test_sneaking_past_guard_at_night_raises_alarm(self, game):
        game.world.simulated_hour = 23
        vance = game.world.find_npc("Guard Captain Vance")
        vance.update_schedule(23, world=game.world)         # asleep in the guard house at 23:00 ...
        vance.state = AIState.PATROL                         # ... but pretend he is a night watchman
        vance.path = []
        vance.position = spot(game.world, 20.0, 0.0)
        teleport(game, 20.0, -2.0)
        vance.turn_towards(game.player.position)
        game.player.has_lit_torch = True                     # lit, so he can actually see the crouching figure
        game.rng.random = lambda: 0.99                       # sneak rolls fail
        game.toggle_crouch()
        for _ in range(400):
            game.update(1 / 60, MoveIntent(left=True, right=True))   # "moving" but staying put
        # his warning bought the player a few seconds of grace ...
        assert any("skulking" in m for m in game.messages.messages) and game.player.gold == STARTING_GOLD
        for _ in range(300):
            game.update(1 / 60, MoveIntent(left=True, right=True))
        # ... then the alarm goes up and, with the player right in front of him, he fines them on the spot
        assert game.player.gold == STARTING_GOLD - 20.0
        assert any("Skulking about" in m for m in game.messages.messages)


def test_custom_level_and_settings(tmp_path):
    from unprepared.config import TimeSettings
    from unprepared.world import LevelDefinition
    level = LevelDefinition("empty", walls=(), doors=(), npcs=(), player_spawn=(1, 1, 1),
                            player_gold=5.0, intro_message=None)
    g = Game(level=level, settings=GameSettings(echo_messages_to_stdout=False, time=TimeSettings(start_hour=3)),
             save_file=str(tmp_path / "s.json"))
    assert g.player.gold == 5.0 and g.world.simulated_hour == 3 and g.messages.messages == []
    g.update(1.0)
    assert g.target is None


def test_fixed_sweet_spot_setting(tmp_path):
    settings = GameSettings(echo_messages_to_stdout=False, lockpick=LockpickSettings(random_sweet_spot=False))
    g = Game(settings=settings, save_file=str(tmp_path / "s.json"))
    crouch(g)
    teleport(g, *SHOP_DOOR_FRONT)
    g.interact()
    assert g.lockpicking.sweet_spot == 45.0


class TestTheEventLog:
    """The World Events log is hidden at the start and toggled with [`]; while it is hidden, what
    concerns the player pops up for a few seconds - the town's routine news does not."""

    def test_hidden_at_the_start_and_toggled(self, game):
        assert not game.show_event_log
        assert game.toggle_event_log() is True and game.show_event_log
        assert game.toggle_event_log() is False and not game.show_event_log

    def test_what_concerns_the_player_pops_up_and_fades(self, game):
        game.update(0.1)
        assert [t for t, _ in game.toasts] == [game.world.level.intro_message]      # (the greeting pops up too)
        game.toggle_torch()
        game.update(0.1)
        assert [t for t, _ in game.toasts][-1] == "Torch is now LIT."
        game.world.news_callback("Barnaby is having lunch.")           # the town's news: kept, not popped up
        game.update(0.1)
        assert [t for t, _ in game.toasts][-1] == "Torch is now LIT." and game.messages.messages[-1] == "Barnaby is having lunch."
        for _ in range(60):
            game.update(0.1)
        assert game.toasts == []                                         # gone after a few seconds

    def test_no_pop_ups_while_the_log_is_shown(self, game):
        game.toggle_event_log()
        game.toggle_torch()
        game.update(0.1)
        assert game.toasts == []

    def test_the_cursor_keys_do_nothing_outside_a_screen(self, game):
        before = [i.id for i in game.player.inventory]
        game.cursor_down()
        game.cursor_up()
        assert game.in_explore_mode and [i.id for i in game.player.inventory] == before

    def test_cursor_keys_drive_the_menu_while_trading(self, game):
        for i in range(10):
            game.player.inventory.append(Item(f"junk{i}", f"Junk {i}", value=1.0))
        face_jae(game)
        game.interact()
        game.cursor_down()
        assert game.transfer.selection[Side.LEFT] == 1


class TestStolenGoods:
    def test_names_carry_no_stolen_prefix_and_fence_launders(self, game):
        goblet = game.player.inventory[2]
        assert goblet.name == "Silver Goblet" and goblet.is_stolen
        face_jae(game)
        game.interact()
        game.transfer_switch_side()
        game.cursor_up()
        game.transfer_confirm()
        jae = game.world.find_npc("Jae the Fence")
        sold = next(i for i in jae.inventory if i.id == "silver_goblet")
        assert not sold.is_stolen


class TestTransferScrolling:
    def test_long_columns_scroll_to_keep_the_cursor_visible(self, game):
        jae = game.world.find_npc("Jae the Fence")
        for i in range(20):
            jae.inventory.append(Item(f"loot{i}", f"Loot {i}", value=1.0))
        face_jae(game)
        game.interact()
        s = game.transfer
        s.visible_rows = 5
        first, rows = s.visible(Side.LEFT)
        assert first == 0 and len(rows) == 5
        for _ in range(7):
            game.cursor_down()
        first, rows = s.visible(Side.LEFT)
        assert first == 3 and rows[-1] is s.selected_item()
        for _ in range(7):
            game.cursor_up()
        assert s.visible(Side.LEFT)[0] == 0
        game.cursor_up()                                       # wrap to the bottom
        first, rows = s.visible(Side.LEFT)
        assert first == len(s.rows(Side.LEFT)) - 5 and rows[-1] is s.selected_item()


class TestMiscRules:
    def test_no_sprinting_while_crouched(self, game):
        crouch(game)
        start = game.player.position.z
        game.update(0.5, MoveIntent(forward=True, sprint=True))
        assert not game.movement.is_sprinting
        assert game.player.position.z == pytest.approx(start - 1.0)          # sneaking: half walking speed

    def test_wait_passes_an_hour_and_npcs_live_on(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        assert game.wait(2) is True
        assert game.world.simulated_hour == 9 and barnaby.state is AIState.SHOPKEEPING
        assert any("You wait for 2 hours" in m for m in game.messages.messages)

    def test_fence_goods_are_never_stolen(self, game):
        jae = game.world.find_npc("Jae the Fence")
        assert not any(i.is_stolen for i in jae.inventory)
        jae.state = AIState.SLEEPING
        crouch(game)
        teleport(game, jae.position.x, jae.position.z - STEAL_DIST, yaw=180.0, pitch=LOOK_DOWN)
        game.interact()
        game.transfer_switch_side()
        game.cursor_up()                        # the stolen goblet
        game.transfer_confirm()                                # planted on the fence
        planted = next(i for i in jae.inventory if i.id == "silver_goblet")
        assert not planted.is_stolen

    def test_door_never_traps_the_player(self, game):
        door = game.world.find_door("hideout_door")
        teleport(game, -12.0, 4.5, yaw=180.0)
        game.interact()                                        # open
        game.player.position = spot(game.world, door.hinge_position.x, 5.5)     # stand where the open leaf lies
        game.update(0.0)
        assert not game.player.collision_box().intersects(door.collider())


class TestSellingWhatYouWear:
    def test_selling_a_worn_item_asks_first(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SHOPKEEPING
        face_jae(game)
        game.interact()
        game.transfer_switch_side()
        boots = next(i for i in game.player.inventory if i.id == "soft_shoes")
        assert boots.equipped
        game.transfer.selection[Side.RIGHT] = game.player.inventory.index(boots)
        assert game.transfer_confirm() is None                                 # not sold yet ...
        assert game.mode is GameMode.CONFIRM and "Soft Leather Boots you are wearing" in game.confirmation.text
        assert boots in game.player.inventory and boots.equipped
        game.confirm_no()                                                     # ESC: keep them on
        assert boots in game.player.inventory and boots.equipped and game.mode is GameMode.TRADE
        game.transfer_confirm()
        game.confirm_yes()                                                    # ENTER: sell them after all
        assert boots not in game.player.inventory and not boots.equipped
        assert any(m.startswith("Sold Soft Leather Boots") for m in game.messages.messages)

    def test_unworn_items_sell_without_a_question(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SHOPKEEPING
        face_jae(game)
        game.interact()
        game.transfer_switch_side()
        pick = next(i for i in game.player.inventory if i.id == "iron_lockpick")
        game.transfer.selection[Side.RIGHT] = game.player.inventory.index(pick)
        assert game.transfer_confirm().outcome is TransferOutcome.MOVED and game.confirmation is None
