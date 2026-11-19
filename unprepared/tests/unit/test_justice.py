"""Fines, jail, confiscation, sentences and jailbreaks."""
import pytest

from tests.conftest import said
from tests.helpers import spot
from unprepared.config import GameSettings
from unprepared.entities import AIState, Item, SkillType
from unprepared.game import Game
from unprepared.systems import justice
from unprepared.world.levels.town import STARTING_GOLD


@pytest.fixture
def game(tmp_path):
    g = Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=5, max_log_messages=400),
             save_file=str(tmp_path / "s.json"))
    return g


def vance(game, clean_pockets=True):
    """Vance off duty (no routine to walk away on). ``clean_pockets`` drops the stolen goblet
    from the starting kit, so the tests below deal with exactly the crime they plant (being
    caught carrying stolen goods is a charge of its own - see TestPunishments)."""
    v = game.world.find_npc("Guard Captain Vance")
    v.path, v.current_task, v.schedule = [], None, []
    if clean_pockets:
        game.player.inventory = [i for i in game.player.inventory if not i.is_stolen]
    return v


def face_off(game, guard, dist=2.0):
    """Stand ``dist`` in front of the guard - within ``charge_reach`` by default, so a guard who
    already holds a (ready) memory against the player charges them on the first frame."""
    game.player.position = spot(game.world, guard.position.x, guard.position.z + dist)
    guard.turn_towards(game.player.position)


def settle(game, seconds=6.5, dt=0.1):
    """Let a caught-freeze, the delayed fade to black and the relocation play out."""
    for _ in range(int(seconds / dt)):
        game.update(dt)


class TestMemory:
    def test_witnessed_crimes_last_weeks_reported_ones_days(self, game):
        v = vance(game)
        seen = v.remember_crime("Pickpocketing", 10, witnessed=True, fine=25.0, rng=game.rng)
        heard = v.remember_crime("Shoplifting", 10, witnessed=False, fine=20.0, rng=game.rng)
        assert 21 <= seen.expires_day - 10 <= 35
        assert 3 <= heard.expires_day - 10 <= 5
        assert len(v.active_memories(10)) == 2
        assert [m.crime for m in v.active_memories(16)] == ["Pickpocketing"]
        assert v.active_memories(50) == []

    def test_outstanding_fine_counts_each_crime_once(self, game):
        for g in game.world.guards():
            g.remember_crime("Pickpocketing", 0, False, 25.0)
        game.world.find_npc("Guard Orrin").remember_crime("Theft", 0, True, 20.0)
        assert justice.outstanding_fine(game.world) == 45.0
        j = game.settings.justice
        assert justice.penalty_for("Jailbreak", False, j).fine == 100.0
        assert justice.penalty_for("Unknown", False, j) == j.default_penalty


class TestAlarmResponse:
    def test_a_cry_for_help_sends_every_guard_in_earshot_running_even_sleepers(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        mira = game.world.find_npc("Guard Mira")
        assert mira.is_asleep
        barnaby.path, barnaby.current_task, barnaby.schedule = [], None, []
        barnaby.position = spot(game.world, 30.0, -22.0)                # robbed in the street, out of the guards' sight
        assert game.world.can_hear_cry(mira, barnaby.position)           # (but the guard house is in earshot)
        barnaby.state = AIState.STROLLING
        game.player.position = spot(game.world, barnaby.position.x, barnaby.position.z - 2.0)
        barnaby.turn_towards(game.player.position)
        witnesses = game.report_crime("Pickpocketing")
        assert barnaby in witnesses and barnaby.state is AIState.REPORTING
        for guard in game.world.guards():
            if guard.name == "Jailer Hesk" or not game.world.can_hear_cry(guard, barnaby.position):
                continue                                     # (the look-out atop the tower, the wardens: out of earshot)
            assert guard.state is AIState.RESPONDING and guard.path, guard.name
            assert [m.crime for m in guard.crime_memories] == ["Pickpocketing"] and not guard.crime_memories[0].witnessed
        assert any("woke up" in m for m in game.messages.messages) and said(game.messages.messages, "cry_response")

    def test_responding_guard_gives_up_and_remembers_the_face(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        barnaby.path, barnaby.current_task, barnaby.schedule = [], None, []
        barnaby.position = spot(game.world, 10.0, 3.0)
        barnaby.state = AIState.STROLLING
        game.player.position = spot(game.world, barnaby.position.x, barnaby.position.z - 2.0)
        barnaby.turn_towards(game.player.position)
        game.report_crime("Pickpocketing")
        game.player.position = spot(game.world, -45.0, -28.0)          # run far away and hide
        for _ in range(int(60 / 0.25)):
            game.update(0.25)
        v = game.world.find_npc("Guard Captain Vance")
        assert v.state not in (AIState.RESPONDING, AIState.ALERT_GUARDS)
        assert v.active_memories(game.world.day_count)                 # still wanted ...
        assert said(game.messages.messages, "gone")
        game.world.day_count += 6                                      # ... until the reported crime fades
        assert not v.active_memories(game.world.day_count)

    def test_wanted_player_is_charged_on_sight(self, game):
        v = vance(game)
        v.remember_crime("Shoplifting", game.world.day_count, False, 20.0)
        game.player.inventory.append(Item("wine", "Bottle of Wine", value=12.0, is_stolen=True, stackable=True,
                                          owner="Shopkeeper Barnaby"))
        face_off(game, v)
        game.update(0.05)
        assert game.player.gold == STARTING_GOLD - 20.0 - 12.0                       # the fine, and the bottle's worth
        assert game.world.jail.serving and game.world.jail.days_left(game.world.day_count) == 1   # caught with stolen goods
        assert v.crime_memories == []
        assert not any(i.id == "wine" for i in game.player.inventory)      # returned to the shop shelf
        west = game.world.find_container("shop_shelf_west")
        assert next(i for i in west.inventory if i.id == "wine").quantity == 4
        assert any("1 stolen item on you" in m and "1 day in the cells" in m for m in game.messages.messages)

    def test_guard_facing_away_does_not_notice(self, game):
        v = vance(game)
        v.remember_crime("Shoplifting", game.world.day_count, False, 20.0)
        face_off(game, v)
        v.facing_yaw = 0.0                                             # looking away
        game.update(0.05)
        assert game.player.gold == STARTING_GOLD


class TestTheChase:
    """Declaring a crime starts a chase: "Halt!", the guard runs after the player, and the charge
    waits until ``charge_delay`` has passed *and* the guard is within ``charge_reach``."""

    def test_no_charge_before_the_delay_even_within_reach(self, game):
        v = vance(game)
        face_off(game, v, dist=2.0)
        game.world.declare_crime("Pickpocketing", game.player.position.copy(), witnessed=True, reporter=v.name, witnesses=[v])
        assert v.charge_ready_at == pytest.approx(game.world.elapsed + game.settings.justice.charge_delay)
        game.update(0.05)
        assert game.player.gold == STARTING_GOLD and v.state is AIState.ALERT_GUARDS
        for _ in range(int(game.settings.justice.charge_delay / 0.1) + 2):
            game.update(0.1)
        assert v.crime_memories == []                                            # the delay passed: charged -
        assert game.world.jail.serving and game.player.gold == STARTING_GOLD              # picking pockets in the street is 100g:
        assert game.world.jail.days_left(game.world.day_count) == 3              # unaffordable, so three days

    def test_guard_runs_after_a_distant_player_and_charges_on_arrival(self, game):
        v = vance(game)
        face_off(game, v, dist=8.0)
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)     # an old grudge: no delay left
        game.update(0.05)
        assert game.player.gold == STARTING_GOLD and v.state is AIState.ALERT_GUARDS       # "Halt!" - and the chase is on
        assert said(game.messages.messages, "halt") and v.path
        start = v.position.copy()
        for _ in range(20):
            game.update(0.1)
        assert v.position.distance_to(start) > 4.0                              # running, not walking
        for _ in range(40):
            game.update(0.1)
            if game.player.gold < STARTING_GOLD:
                break
        assert game.player.gold == STARTING_GOLD - 25.0
        assert v.position.distance_to(game.player.position) <= game.settings.justice.charge_reach + 0.5

    def test_a_sprinting_thief_gets_away_this_time(self, game):
        from unprepared.systems import MoveIntent
        v = vance(game)
        face_off(game, v, dist=6.0)
        game.player.rotation_y = 270.0                                          # facing -x: down the street, away from him
        game.world.declare_crime("Pickpocketing", game.player.position.copy(), witnessed=True, reporter=v.name, witnesses=[v])
        for _ in range(600):                                                    # up to a minute of running
            game.update(0.1, MoveIntent(forward=True, sprint=True))
            if v.state is not AIState.ALERT_GUARDS:
                break
        assert game.player.gold == STARTING_GOLD                                          # never caught up with
        assert v.state is not AIState.ALERT_GUARDS and said(game.messages.messages, "gone")
        assert [m.crime for m in v.crime_memories] == ["Pickpocketing"]          # ... but not forgotten


class TestJail:
    def test_unpaid_fine_means_jail(self, game):
        v = vance(game)
        game.player.gold = 5.0
        game.player.inventory[0].quantity = 3                          # three picks
        v.remember_crime("Lockpicking a door", game.world.day_count, True, 30.0)
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        jail = game.world.jail
        assert jail.serving and jail.sealed and jail.sentence_end_day == game.world.day_count + 2    # a day for each unpaid fine
        assert game.player.gold == 5.0                                # nothing could be paid; the purse is not confiscated
        # the player faces the guard while the screen fades; the move happens in the dark
        assert game.in_transition and game.world.area_of(game.player.position) == "main street"
        assert abs(game.player.rotation_y - 0.0) < 1e-6            # Vance is to the north (-z)
        settle(game)
        assert not game.in_transition
        assert game.world.area_of(game.player.position) == "the jail cell"
        assert [(i.id, i.quantity) for i in game.player.inventory] == [("iron_lockpick", 1)]   # one pick kept
        chest = game.world.find_container("jail_storage")
        assert chest.is_locked
        assert {i.id for i in chest.inventory} == {"iron_lockpick", "soft_shoes"}
        assert all(i.owner == Item.PLAYER_OWNER for i in chest.inventory)       # confiscated, but still theirs
        door = game.world.find_door("cell_door")
        assert door.is_locked and door.sealed and not door.is_open
        hesk = game.world.find_npc("Jailer Hesk")
        assert not hesk.can_open(door)         # sealed: not even the jailer opens the bars

    def test_stolen_goods_go_back_to_their_owners(self, game):
        v = vance(game)
        game.player.gold = 0.0
        game.player.inventory.append(Item("candlestick", "Silver Candlestick", value=45.0, is_stolen=True, owner="Shopkeeper Barnaby"))
        game.player.inventory.append(Item("guard_key", "Guard House Key", value=1.0, is_key=True, key_door_id="guardhouse_door",
                                          is_stolen=True, owner="Guard Captain Vance"))
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        west = game.world.find_container("shop_shelf_west")
        assert sum(1 for i in west.inventory if i.id == "candlestick") == 2       # back on Barnaby's shelf
        assert sum(1 for i in v.inventory if i.id == "guard_key") == 2                # back in Vance's own pockets
        assert game.world.find_door("cell_door").is_locked
        assert any("thrown in the cell" in m for m in game.messages.messages)

    def test_what_was_worn_comes_off_when_it_is_taken_away(self, game):
        """A stolen helm on the thief's head goes back to the rack, and their own boots into the
        lockbox - neither of them still "worn" by somebody who no longer has them."""
        v = vance(game)
        p = game.player
        p.gold = 0.0
        rack = game.world.find_container("smithy_rack")
        helm = next(i for i in rack.inventory if i.id == "iron_helm")
        rack.inventory.remove(helm)
        p.add_item(helm.steal())
        assert p.equip(helm) and helm in p.worn_items()
        boots = next(i for i in p.inventory if i.id == "soft_shoes")
        assert boots.equipped
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        assert game.world.jail.serving
        assert helm in rack.inventory and not helm.equipped and not helm.is_stolen        # back with Gareth, off the head
        chest = game.world.find_container("jail_storage")
        assert boots in chest.inventory and not boots.equipped                              # confiscated: off the feet
        assert not any(i.equipped for i in chest.inventory)
        assert p.equipment == {}

    def test_goods_of_unknown_origin_go_to_the_lockbox(self, game):
        v = vance(game)
        game.player.inventory.append(Item("loot", "Mystery Loot", value=5.0, is_stolen=True))       # no owner known
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        storage = game.world.find_container("jail_storage")
        loot = [i for i in storage.inventory if i.id == "loot"]
        assert len(loot) == 1 and not loot[0].is_stolen and loot[0].owner is None

    def test_sentence_is_served_and_released(self, game):
        v = vance(game)
        game.player.gold = 0.0
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        settle(game)
        assert game.world.jail.serving and game.world.jail.days_left(game.world.day_count) == 1
        game.world.wait_hours(17)                                     # 07:00 -> midnight: the release day ...
        assert game.world.jail.serving                                # ... but prisoners are let out at 06:00, not midnight
        game.world.wait_hours(6)
        assert not game.world.jail.serving and game.world.jail.released
        door = game.world.find_door("cell_door")
        assert not door.is_locked and door.is_open and not door.sealed
        assert not game.world.find_container("jail_storage").is_locked          # lockbox opened for the ex-prisoner
        assert any("time is served" in m for m in game.messages.messages)
        hesk = game.world.find_npc("Jailer Hesk")
        assert not game.world.is_trespassing(game.player, hesk)
        assert hesk.state is AIState.WAITING and hesk.waiting_for == "release"    # watching until they leave
        # walking out through the (never locked) jail door ends the release: lockbox locked, jailer back to work
        assert not game.world.find_door("jail_door").is_locked
        game.player.position = spot(game.world, 51.5, 4.0)
        game.update(0.1)
        assert not game.world.jail.released and game.world.find_container("jail_storage").is_locked
        assert hesk.state is not AIState.WAITING

    def test_the_jailer_waits_up_for_a_night_release(self, game):
        """Released while the jailer would normally sleep, he stays awake until the player has gone."""
        hesk = game.world.find_npc("Jailer Hesk")
        game.world.simulated_hour = 23
        game.world.resync_schedules(announce=False, teleport=True)
        assert hesk.is_asleep
        game.player.position = spot(game.world, 55.0, 13.0)
        game.world.jail.sentence_end_day = game.world.day_count
        justice.release(game.world, game.player)
        for _ in range(30):
            game.update(0.1)
        assert hesk.state is AIState.WAITING and not hesk.is_asleep
        game.world.update_time(24.0)                                  # the hour ticks: his routine says "sleep" ...
        assert hesk.state is AIState.WAITING                          # ... but he waits for the player to leave
        game.player.position = spot(game.world, 51.5, 4.0)
        game.update(0.1)
        assert hesk.state is AIState.TRAVELING                        # and only then heads for his bed
        for _ in range(100):
            game.update(0.1)
        assert hesk.state is AIState.SLEEPING

    def test_prisoner_waits_out_the_sentence_with_t(self, game):
        v = vance(game)
        game.player.gold = 0.0
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        settle(game)
        assert game.world.jail.sealed
        assert game.wait() and game.confirmation is not None            # asked first
        assert game.confirmation.text == "Wait out your sentence (1 day left)?"
        game.confirm_yes()
        assert game.confirmation is None and not game.world.jail.serving and game.world.jail.released
        assert game.world.simulated_hour == game.settings.justice.release_hour
        # [T] again inside the question just waits the usual hour
        game.world.jail.sentence_end_day = game.world.day_count + 1
        game.world.jail.released = False
        hour = game.world.simulated_hour
        game.wait()
        assert game.confirmation is not None
        game.wait()
        assert game.confirmation is None and game.world.simulated_hour == hour + 1

    def test_prisoner_in_the_cell_is_tolerated_but_a_breakout_is_a_crime(self, game):
        v = vance(game)
        game.player.gold = 0.0
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        settle(game)
        hesk = game.world.find_npc("Jailer Hesk")
        assert not game.world.is_trespassing(game.player, hesk)
        game.player.position = spot(game.world, -45.0, -28.0)           # out of the jail (and out of sight) before the release day
        game.update(0.05)                                             # noticed at once
        assert game.world.jail.escaped and not game.world.jail.serving
        assert not game.world.find_door("cell_door").sealed
        assert any("broken out" in m for m in game.messages.messages)
        heard = [g for g in game.world.guards() if any(m.crime == "Jailbreak" for m in g.crime_memories)]
        assert heard                                                  # the guards within earshot remember it

    def test_jail_state_is_saved(self, game):
        game.world.jail.sentence_end_day = 4
        game.player.position = game.world.jail.cell_spot.copy()      # serving it, in the cell
        game.save_game()
        game.world.jail.sentence_end_day = None
        game.load_game()
        assert game.world.jail.sentence_end_day == 4 and not game.world.jail.escaped

    def test_loading_checks_the_jail_at_once(self, game):
        """A save that has the prisoner outside the cell is judged on loading, not an hour later."""
        game.world.jail.sentence_end_day = 4
        game.save_game()                                              # (in the street)
        game.world.jail.sentence_end_day = None
        game.load_game()
        assert game.world.jail.escaped and not game.world.jail.serving
        assert any("broken out" in m for m in game.messages.messages)

    def test_the_jail_house_is_public_but_the_cell_is_not(self, game):
        hesk = game.world.find_npc("Jailer Hesk")
        game.player.position = spot(game.world, 51.0, 14.2)             # in the jail office by the lockbox
        assert not game.world.is_trespassing(game.player, hesk)
        assert not game.world.find_door("jail_door").is_locked and not game.world.find_door("jail_door").owners
        game.player.position = spot(game.world, 55.0, 13.0)             # inside the cell uninvited
        assert game.world.is_trespassing(game.player, hesk)
        justice.release(game.world, game.player)                      # a released prisoner may linger in the cell ...
        assert not game.world.is_trespassing(game.player, hesk)
        justice.end_release(game.world)
        game.world.elapsed += 200.0
        assert game.world.is_trespassing(game.player, hesk)           # ... not forever

    def test_nobody_can_get_stuck_in_the_jail_house(self, game):
        """Released, the player can always leave: the jail door is never locked and nobody locks it."""
        door = game.world.find_door("jail_door")
        hesk = game.world.find_npc("Jailer Hesk")
        for hour in range(24):
            game.world.simulated_hour = hour
            game.world.resync_schedules(announce=False, teleport=True)
            assert not door.is_locked
        assert not any(hesk.owns(d) and d.door_id == "jail_door" for d in game.world.doors)
        assert game.world.exit_of("the jail") is not None

    def test_own_items_in_the_lockbox_are_not_stolen_goods(self, game):
        """Picking the lockbox is a crime; taking your confiscated belongings back is not."""
        v = vance(game, clean_pockets=False)                            # the stolen goblet comes along (and costs days)
        game.player.gold = 0.0
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        settle(game)
        chest = game.world.find_container("jail_storage")
        chest.unlock_and_open()                                        # say the lock was picked
        game.player.position = spot(game.world, 51.2, 14.25)
        game.player.rotation_y, game.player.rotation_x = 270.0, -20.0
        game.update(0.0)
        assert game.interaction_prompt == "[E] Search Confiscation Lockbox"
        game.interact()
        before = game.player.skills.skills[SkillType.STEALING]["xp"]
        while game.transfer.rows(game.transfer.side):
            game.transfer_confirm()
        mine = [i for i in game.player.inventory if i.id != "silver_goblet"]
        assert mine and not any(i.is_stolen for i in mine)
        assert all(i.owner == Item.PLAYER_OWNER for i in mine)
        assert any("Took back your" in m for m in game.messages.messages)
        # the goblet was stolen property, confiscated for good: lifting it again *is* theft
        goblet = next(i for i in game.player.inventory if i.id == "silver_goblet")
        assert goblet.is_stolen and goblet.owner is None
        assert game.player.skills.skills[SkillType.STEALING]["xp"] > before
        assert not game.world.theft_reports

    def test_the_jailer_ignores_a_rattling_prisoner(self, game):
        """Noise from inside a sealed cell never makes the jailer open the bars to look."""
        v = vance(game)
        game.player.gold = 0.0
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v)
        game.update(0.05)
        settle(game)
        hesk = game.world.find_npc("Jailer Hesk")
        hesk.path, hesk.current_task, hesk.schedule = [], None, []
        hesk.state = AIState.GUARDING
        hesk.position = spot(game.world, 51.5, 12.0)
        hesk.facing_yaw = 90.0
        game.world.simulated_hour = 23
        from unprepared.systems import MoveIntent
        for _ in range(60):
            game.update(0.1, MoveIntent(left=True, right=True, sprint=True))
        door = game.world.find_door("cell_door")
        assert door.is_locked and not door.is_open and door.sealed
        assert hesk.state is not AIState.INVESTIGATING
        assert game.world.area_of(hesk.position) != "the jail cell"
        assert said(game.messages.messages, "cell_noise")

    def test_a_fine_inside_someones_house_also_gets_you_thrown_out(self, game):
        v = vance(game)
        v.position = spot(game.world, -70.0, 11.0)                       # a guard inside Barnaby's house
        v.state = AIState.PATROL
        game.player.gold = 50.0
        v.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        face_off(game, v, dist=2.0)
        assert game.world.area_of(game.player.position) == "Barnaby's house"
        game.update(0.05)
        assert game.player.gold == 25.0 and game.in_transition
        assert any("no business in Barnaby's house" in m for m in game.messages.messages)
        settle(game)
        assert game.world.area_of(game.player.position) != "Barnaby's house"


class TestTrespassAndTorchLaw:
    def test_shopkeeper_objects_to_customers_behind_the_counter(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        barnaby.position = spot(game.world, -13.0, -12.2)
        barnaby.state = AIState.SHOPKEEPING
        barnaby.path, barnaby.current_task, barnaby.schedule = [], None, []
        game.world.simulated_hour = 10
        send = game.world.guards()
        for g in send:
            g.position = spot(game.world, -45, 28); g.path, g.current_task, g.schedule = [], None, []
        game.player.position = spot(game.world, -11.0, -13.5)              # behind the counter, standing upright
        for _ in range(10):
            game.update(0.1)
        assert said(game.messages.messages, "trespass")
        assert barnaby.suspicion_meter > 0
        assert barnaby.faces(game.player.position)                    # he turned to watch

    def test_customers_on_the_shop_floor_are_fine(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        barnaby.position = spot(game.world, -13.0, -12.2)
        barnaby.state = AIState.SHOPKEEPING
        barnaby.path, barnaby.current_task, barnaby.schedule = [], None, []
        game.world.simulated_hour = 10
        game.player.position = spot(game.world, -13.0, -8.4)
        for _ in range(10):
            game.update(0.1)
        assert barnaby.suspicion_meter == 0.0 and barnaby.faces(game.player.position)

    def test_jae_trusts_the_player_in_the_hideout(self, game):
        jae = game.world.find_npc("Jae the Fence")
        game.player.position = spot(game.world, -16.0, 16.0)       # in the hideout, on the customers' side
        game.toggle_crouch()
        for _ in range(40):
            game.update(0.1)
        assert not jae.is_alarmed and jae.state is AIState.SHOPKEEPING     # wary, but no cry for the watch

    def test_guard_demands_a_torch_at_night(self, game):
        game.world.simulated_hour = 23
        v = vance(game)
        v.state = AIState.PATROL
        v.has_torch = True
        v.position = spot(game.world, 20, 0)
        face_off(game, v, dist=2.5)
        game.player.has_lit_torch = False
        from unprepared.systems import MoveIntent
        for _ in range(10):
            game.update(0.1, MoveIntent(left=True, right=True))
        assert said(game.messages.messages, "torch_law")
