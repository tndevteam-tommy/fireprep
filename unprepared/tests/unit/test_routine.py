"""The NPC daily-routine framework, exercised on Barnaby's day in the town."""
import math

import pytest

from unprepared.config import GameSettings
from unprepared.entities import AIState
from unprepared.game import Game
from tests.conftest import said
from tests.helpers import spot, teleport
from unprepared.core.dialogue import LINES
from unprepared.world.levels.town import (BARNABY_BED, BARNABY_TABLE_SEAT, GUARD_BEDS, GUARD_TABLE_SEATS, PATROL_ROUTE,
                                          POST_DOOR, POST_WEST, JAE_DEN, SHOP_BACKROOM, SHOP_CLOSE_HOUR, SHOP_COUNTER,
                                          SHOP_OPEN_HOUR)


def at(npc, spot, tol=0.05):
    """On the spot (a level's local one: its floor height is the world's business)."""
    return math.hypot(npc.position.x - spot[0], npc.position.z - spot[2]) <= tol


def run_until(game, hour, dt=0.25, extra_seconds=0.0):
    """Advance until the clock shows ``hour`` (then optionally a bit more)."""
    for _ in range(100000):
        if game.world.simulated_hour == hour:
            break
        game.update(dt)
    else:
        raise AssertionError("hour never reached")
    for _ in range(int(extra_seconds / dt)):
        game.update(dt)


@pytest.fixture
def game(tmp_path):
    return Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=1, max_log_messages=100),
                save_file=str(tmp_path / "s.json"))


def barnaby(game):
    return game.world.find_npc("Shopkeeper Barnaby")


class TestBarnabysDay:
    def test_breakfast_at_start(self, game):
        b = barnaby(game)
        assert b.state is AIState.EATING and at(b, BARNABY_TABLE_SEAT)
        assert game.world.find_door("shop_door").is_locked
        assert not game.world.find_door("home_door").is_locked      # unlocked while he is up and about at home

    def test_commute_to_the_shop(self, game):
        b = barnaby(game)
        shop, home = game.world.find_door("shop_door"), game.world.find_door("home_door")
        teleport(game, -40.0, 0.0)                  # on Market Street, between his house and the shop: in earshot of both
        run_until(game, SHOP_OPEN_HOUR)
        assert b.state is AIState.TRAVELING and b.path
        assert "(Shopkeeper Barnaby): \"Time to open up the shop.\"" in game.messages.messages
        # he walks out through his own front door ...
        seen_home_open = False
        for _ in range(800):
            game.update(0.05)
            seen_home_open = seen_home_open or home.is_open
            if b.state is AIState.SHOPKEEPING:
                break
        assert seen_home_open
        assert b.state is AIState.SHOPKEEPING and at(b, SHOP_COUNTER)
        # ... locks it behind him, and unlocks the shop on arrival (the door stays shut: customers open it)
        assert home.is_locked and not home.is_open
        assert not shop.is_locked and not shop.is_open
        assert game.messages.messages[-1] == "Barnaby opened the shop for business."     # narration, not speech
        assert b.is_open_for_business

    def test_walk_is_collision_free(self, game):
        b = barnaby(game)
        run_until(game, SHOP_OPEN_HOUR)
        obstacles = game.world.static_obstacles()
        for _ in range(400):
            game.update(0.05)
            p = b.position
            assert not any(o.min_point.x < p.x < o.max_point.x and o.min_point.z < p.z < o.max_point.z for o in obstacles), p
            if b.state is AIState.SHOPKEEPING:
                break

    def test_lunch_break_closes_nothing(self, game):
        b = barnaby(game)
        run_until(game, 13, extra_seconds=8)
        assert b.state is AIState.EATING and at(b, SHOP_BACKROOM)
        assert not b.is_open_for_business
        assert not game.world.find_door("shop_door").is_locked
        run_until(game, 14, extra_seconds=8)
        assert b.state is AIState.SHOPKEEPING and at(b, SHOP_COUNTER)

    def test_evening_lock_up_and_bed(self, game):
        b = barnaby(game)
        shop, home = game.world.find_door("shop_door"), game.world.find_door("home_door")
        run_until(game, SHOP_CLOSE_HOUR, extra_seconds=23)                  # (a long walk home, across the bridge)
        assert b.state is AIState.EATING and at(b, BARNABY_TABLE_SEAT)
        assert shop.is_locked and not shop.is_open
        assert not home.is_locked and not home.is_open          # unlocked for the evening, closed behind him
        run_until(game, SHOP_CLOSE_HOUR + 1, extra_seconds=12)
        assert b.state is AIState.EATING                        # (two hours for supper: the walk home takes a while)
        run_until(game, SHOP_CLOSE_HOUR + 2, extra_seconds=12)
        assert b.state is AIState.SLEEPING and at(b, BARNABY_BED)
        assert not home.is_open and home.is_locked              # locked up for the night with his house key
        assert not b.has_torch                                  # asleep, and his house is lit anyway
        assert game.world.prop_under(b.position).name == "barnaby_bed"

    def test_full_cycle_repeats(self, game):
        b = barnaby(game)
        run_until(game, 6)            # next morning
        assert b.state is AIState.SLEEPING
        run_until(game, 7, extra_seconds=12)
        assert b.state is AIState.EATING and at(b, BARNABY_TABLE_SEAT)
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)
        assert b.state is AIState.SHOPKEEPING and at(b, SHOP_COUNTER)

    def test_trading_hours(self, game):
        b = barnaby(game)
        assert not b.is_open_for_business and b.greeting() in LINES["greet_closed"]
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=23)
        assert b.is_open_for_business
        run_until(game, 22, extra_seconds=12)
        assert b.greeting() in LINES["greet_asleep"]


class TestGuardRotation:
    GUARDS = ("Guard Captain Vance", "Guard Orrin", "Guard Mira")

    def states(self, game):
        return {n: game.world.find_npc(n).state for n in self.GUARDS}

    def test_one_patrols_one_posts_one_sleeps_at_start(self, game):
        v, o, m = (game.world.find_npc(n) for n in self.GUARDS)
        assert v.state is AIState.PATROL and at(v, PATROL_ROUTE[0]) and v.is_patrolling
        assert o.state is AIState.GUARDING and at(o, POST_DOOR)
        assert m.state is AIState.SLEEPING and at(m, GUARD_BEDS[2])

    def test_shifts_rotate_through_the_day(self, game):
        """What each of the three is on at the hour (walking there is part of the shift: the guard
        house is up the hill, a long way from the west end of the street)."""
        expected = {
            10: ("PATROL", "GUARDING", "SLEEPING"),
            13: ("PATROL", "EATING", "SLEEPING"),
            16: ("GUARDING", "PATROL", "GUARDING"),
            21: ("GUARDING", "PATROL", "EATING"),
            22: ("SLEEPING", "SLEEPING", "PATROL"),
            5: ("EATING", "SLEEPING", "PATROL"),
        }
        for hour in (10, 13, 16, 21, 22, 5):
            run_until(game, hour, extra_seconds=2)
            got = tuple(game.world.find_npc(n).current_task.target_state.name for n in self.GUARDS)
            assert got == expected[hour], (hour, got)
        run_until(game, 16, extra_seconds=23)                     # and at a post, they get there
        assert at(game.world.find_npc("Guard Captain Vance"), POST_WEST) and at(game.world.find_npc("Guard Mira"), POST_DOOR)

    def test_everyone_gets_a_meal_and_a_bed(self, game):
        seen = {n: set() for n in self.GUARDS}
        for _ in range(int(24 * 24 / 0.25)):                          # a full day, sampling every frame
            game.update(0.25)
            for n, state in self.states(game).items():
                seen[n].add(state)
        for n in self.GUARDS:
            assert {AIState.PATROL, AIState.EATING, AIState.SLEEPING, AIState.GUARDING} <= seen[n], (n, seen[n])

    def test_each_guard_sleeps_in_their_own_bed(self, game):
        run_until(game, 2, extra_seconds=40)
        v, o = game.world.find_npc("Guard Captain Vance"), game.world.find_npc("Guard Orrin")
        assert at(v, GUARD_BEDS[0]) and at(o, GUARD_BEDS[1])
        bed = game.world.prop_under(v.position)
        assert bed.name == "guard_bed_1"
        assert game.world.npc_body_box(v).min_point.y == pytest.approx(bed.max_point.y - 0.06)   # lying on top of the bed

    def test_patrol_walks_the_route(self, game):
        v = game.world.find_npc("Guard Captain Vance")
        visited = set()
        for _ in range(2400):                                        # 120 s of patrol
            game.update(0.05)
            for idx, point in enumerate(PATROL_ROUTE):
                if at(v, point, 0.3):
                    visited.add(idx)
        assert len(visited) >= 4 and v.state is AIState.PATROL
        assert 0 in visited and 1 in visited                           # square and shop front

    def test_guard_house_door_relocks_behind_guards(self, game):
        door = game.world.find_door("guardhouse_door")
        assert door.is_locked and not door.is_open
        seen_open = False
        run_until(game, 13)                                           # Orrin heads in from the door for his meal
        for _ in range(300):                                          # 15 s
            game.update(0.05)
            seen_open = seen_open or door.is_open
        assert seen_open
        assert door.is_locked and not door.is_open                    # ... and it is locked again behind him
        assert at(game.world.find_npc("Guard Orrin"), GUARD_TABLE_SEATS[1])

    def test_tower_post(self, game):
        run_until(game, 15, extra_seconds=20)               # (the post at the west end of Market Street since 14:00)
        v = game.world.find_npc("Guard Captain Vance")
        assert v.state is AIState.GUARDING and at(v, POST_WEST)

    def test_jae_has_a_life_but_never_leaves_home(self, game):
        r = game.world.find_npc("Jae the Fence")
        assert r.state is AIState.SHOPKEEPING and at(r, JAE_DEN)
        seen = set()
        for _ in range(int(24 * 24 / 0.25)):
            game.update(0.25)
            seen.add(r.state)
            assert game.world.area_of(r.position) == "the hideout", r.position
        assert {AIState.SLEEPING, AIState.EATING, AIState.SHOPKEEPING} <= seen
        run_until(game, 2, extra_seconds=10)
        assert r.state is AIState.SLEEPING and game.world.prop_under(r.position).name == "jae_bed"

    def test_npcs_close_doors_behind_them(self, game):
        """Every door an NPC walks through is shut again once they are clear of it."""
        door = game.world.find_door("shop_door")
        run_until(game, SHOP_OPEN_HOUR)
        states = []
        for _ in range(600):
            game.update(0.05)
            states.append((door.is_open, door.is_locked))
        # walked through (open), shut and relocked behind him, then unlocked for business (still shut)
        assert (True, False) in states and (False, True) in states[states.index((True, False)):]
        assert states[-1] == (False, False)

    def test_guards_relock_a_door_the_player_unlocked(self, game):
        door = game.world.find_door("guardhouse_door")
        door.is_locked = False                              # the player picked it ...
        door.is_open = True                                 # ... and left it wide open
        run_until(game, 13, extra_seconds=12)               # Orrin walks in for his meal
        assert not door.is_open and door.is_locked

    def test_walking_past_an_open_door_does_not_close_it(self, game):
        shop = game.world.find_door("shop_door")
        shop.is_locked = False
        shop.is_open = True                                 # the player propped the shop door open
        orrin = game.world.find_npc("Guard Orrin")
        orrin.current_task = None
        orrin.schedule = []
        orrin.position = spot(game.world, -20.0, -3.0)
        orrin.go_to(game.world, spot(game.world, -5.0, -3.0), AIState.INVESTIGATING)    # walks right past the doorway
        for _ in range(200):
            game.update(0.05)
        assert shop.is_open and not shop.is_locked          # not his door, not his business

    def test_guards_do_not_open_locked_shops(self, game):
        """Without the shop key a guard's path cannot go through the shop door."""
        orrin = game.world.find_npc("Guard Orrin")
        assert not orrin.can_open(game.world.find_door("shop_door"))
        orrin.position = spot(game.world, -13.5, -3.0)
        assert game.world.find_path(orrin, spot(game.world, -13.5, -9.0)) == []


class TestInterruptions:
    def test_alarmed_npc_stops_then_resumes_its_routine(self, game):
        b = barnaby(game)
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=1.0)
        assert b.state is AIState.TRAVELING
        mid = b.position.copy()
        b.witness_crime("Pickpocketing", mid, [])            # sees the player red-handed
        assert b.state is AIState.REPORTING and b.path == []
        game.update(1.0)
        assert b.position == mid                             # frozen in place, shouting
        for _ in range(60):
            game.update(0.25)                                # ~15 s later he gives up and gets on with his day
        assert b.state in (AIState.TRAVELING, AIState.SHOPKEEPING)
        run_until(game, SHOP_OPEN_HOUR + 1, extra_seconds=12)
        assert b.state is AIState.SHOPKEEPING

    def test_player_closing_the_shop_door_does_not_trap_him(self, game):
        b = barnaby(game)
        shop = game.world.find_door("shop_door")
        run_until(game, SHOP_CLOSE_HOUR - 1)
        shop.is_open = False                                 # player shut it
        run_until(game, SHOP_CLOSE_HOUR, extra_seconds=23)
        assert b.state is AIState.EATING and at(b, BARNABY_TABLE_SEAT)
        assert shop.is_locked and not shop.is_open

    def test_without_his_keys_the_owner_is_locked_out_until_he_gets_new_ones(self, game):
        b = barnaby(game)
        shop = game.world.find_door("shop_door")
        b.inventory = [i for i in b.inventory if not i.is_key]      # the player lifted both keys overnight
        run_until(game, SHOP_OPEN_HOUR, extra_seconds=20)
        assert b.state is AIState.IDLE and shop.is_locked           # no key, no shop: he waits outside on the doorstep
        # (within a stride of the doorstep: neighbours waiting for the shop may be standing on it, chatting)
        assert b.locked_out and b.position.distance_to(game.world.exit_of("the shop")) < 2.5
        assert not said(game.messages.messages, "locked_out")       # said above his head (a bubble), never logged
        # three days later the shopkeeper has had new keys cut ...
        game.world.day_count += 3
        game.world._daily_upkeep()
        assert b.has_key_for(shop)
        run_until(game, SHOP_OPEN_HOUR + 1, extra_seconds=20)
        assert b.state is AIState.SHOPKEEPING and not shop.is_locked

    def test_guards_are_issued_new_keys_every_night(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        vance.inventory = []
        run_until(game, 0)
        assert any(i.id == "guard_key" for i in vance.inventory) and any(i.id == "ration" for i in vance.inventory)
        assert vance.last_inventory_reset_day == 1

    def test_meals_consume_food_from_the_table_first(self, game):
        table = game.world.find_container("guard_table")
        rations_before = table.inventory[0].quantity
        run_until(game, 14, extra_seconds=20)                        # Orrin (at 13) and Aldric (at 14) eat there
        assert table.inventory[0].quantity == rations_before - 2
        aldric = game.world.find_npc("Guard Aldric")
        assert aldric.state is AIState.EATING and not aldric.hungry

    def test_no_food_means_no_meal(self, game):
        table = game.world.find_container("guard_table")
        table.inventory.clear()
        orrin = game.world.find_npc("Guard Orrin")
        orrin.inventory = [i for i in orrin.inventory if not i.is_food]
        teleport(game, POST_DOOR[0] - 3.0, POST_DOOR[2] + 3.0)                # within earshot of the guard house
        run_until(game, 13, extra_seconds=20)
        assert orrin.state is AIState.IDLE and orrin.hungry
        assert said(game.messages.messages, "no_food")


class TestStartingMidDay:
    def make(self, hour, tmp_path):
        from unprepared.config import TimeSettings
        settings = GameSettings(echo_messages_to_stdout=False, random_seed=1, time=TimeSettings(start_hour=hour))
        return Game(settings=settings, save_file=str(tmp_path / "s.json"))

    def test_start_during_shop_hours(self, tmp_path):
        g = self.make(10, tmp_path)
        b = barnaby(g)
        assert b.state is AIState.SHOPKEEPING and at(b, SHOP_COUNTER)
        assert g.world.find_door("home_door").is_locked                 # he left for work
        assert not g.world.find_door("shop_door").is_open and not g.world.find_door("shop_door").is_locked

    def test_start_at_night(self, tmp_path):
        g = self.make(23, tmp_path)
        b = barnaby(g)
        assert b.state is AIState.SLEEPING and at(b, BARNABY_BED) and not b.has_torch     # asleep: no torch
        home, shop = g.world.find_door("home_door"), g.world.find_door("shop_door")
        assert not home.is_open and home.is_locked                     # locked for the night
        assert shop.is_locked and not shop.is_open
        assert g.world.find_npc("Guard Mira").state is AIState.PATROL  # night watch
        assert g.world.find_npc("Guard Captain Vance").state is AIState.SLEEPING

    def test_start_before_dawn(self, tmp_path):
        g = self.make(3, tmp_path)
        assert g.world.find_npc("Guard Captain Vance").state is AIState.SLEEPING
        assert g.world.find_door("home_door").is_locked


class TestScheduleCatchUp:
    """An NPC that could not switch tasks on the hour takes the hour's task up as soon as it is free."""

    def test_suspicious_npc_switches_once_calm(self, world):
        barnaby = world.find_npc("Shopkeeper Barnaby")
        world.clock.hour = 19
        barnaby.begin_task(barnaby.task_for(19), world)
        barnaby.suspicion_meter = 60.0                   # rattled by something at the tick
        world.clock.hour = 20
        barnaby.update_schedule(20, None, world=world)
        assert barnaby.current_task is barnaby.task_for(19)      # the hour tick did nothing
        world.update_npcs(0.1)
        assert barnaby.current_task is barnaby.task_for(19)      # still suspicious: still nothing
        barnaby.suspicion_meter = 0.0
        world.update_npcs(0.1)
        assert barnaby.current_task is barnaby.task_for(20)      # calm again: supper at once
        assert barnaby.state is AIState.EATING                   # (he was at his table already)

    def test_busy_npc_is_left_alone(self, world):
        vance = world.find_npc("Guard Captain Vance")
        world.clock.hour = 14
        vance.state = AIState.RESPONDING                 # running to a cry for help
        previous = vance.current_task
        assert vance.catch_up_schedule(14, None, world=world) is False
        assert vance.current_task is previous

    def test_nothing_to_do_when_on_the_right_task(self, world):
        for npc in world.npcs:
            assert npc.catch_up_schedule(world.clock.hour, None, world=world) is False
