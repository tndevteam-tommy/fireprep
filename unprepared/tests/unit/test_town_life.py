"""Indoors vs outdoors, solid people, barred gates, curfews, cries for help and purses."""
import pytest

from tests.conftest import said
from tests.helpers import spot

from unprepared.world.levels.town import STARTING_GOLD
from unprepared.config import GameSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState, DoorKind
from unprepared.game import Game
from unprepared.systems import MoveIntent


@pytest.fixture
def game(tmp_path):
    g = Game(settings=GameSettings(echo_messages_to_stdout=False, random_seed=11, max_log_messages=400),
             save_file=str(tmp_path / "s.json"))
    _WORLDS[:] = [g.world]
    return g


_WORLDS = []


def npc_world(npc):
    return next(w for w in _WORLDS if npc in w.npcs)


def park(npc, x, z, state=AIState.GUARDING, yaw=0.0):
    npc.path, npc.current_task, npc.schedule = [], None, []
    npc.state = state
    npc.position = spot(npc_world(npc), x, z)
    npc.facing_yaw = yaw
    return npc


class TestIndoorsOutdoors:
    def test_buildings_define_indoors(self, game):
        w = game.world
        assert w.is_indoors(spot(game.world, -13.0, -11.0)) and w.building_at(spot(game.world, -13.0, -11.0)).name == "shop"
        assert not w.is_indoors(spot(game.world, 0.0, 3.0))
        assert w.same_room(spot(game.world, 0.0, 3.0), spot(game.world, 20.0, 20.0))          # both outside
        assert w.same_room(spot(game.world, 47.0, -49.0), spot(game.world, 52.0, -45.0))       # both in the guard house
        assert not w.same_room(spot(game.world, 0.0, 3.0), spot(game.world, 47.0, -49.0))
        assert w.noise_attenuation(spot(game.world, 0.0, 3.0), spot(game.world, 47.0, -49.0)) == 0.25
        assert w.noise_attenuation(spot(game.world, 50.0, -45.0), spot(game.world, 47.0, -49.0)) == 1.0
        # a roof on posts is not indoors
        assert not w.is_indoors(spot(game.world, -62.0, -9.0)) and w.building_at(spot(game.world, -62.0, -9.0), include_open=True).name == "smithy_shed"

    def test_running_past_the_guard_house_does_not_wake_the_sleepers(self, game):
        mira = game.world.find_npc("Guard Mira")
        assert mira.is_asleep and game.world.is_indoors(mira.position)
        game.player.position = spot(game.world, 46.5, -41.0)          # in the street, 2 units from the wall, 4.5 from her bed
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True, backward=True, sprint=True))   # sprinting on the spot
        assert mira.is_asleep
        assert not any("woke up" in m for m in game.messages.messages)

    def test_running_about_inside_their_house_does_wake_them(self, game):
        mira = game.world.find_npc("Guard Mira")
        for g in game.world.guards():
            if g is not mira:
                park(g, -35.0, 35.0)                               # the others far away
        game.player.position = spot(game.world, 49.0, -46.0)          # inside the guard house (trespassing, and loud)
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True, backward=True, sprint=True))
        assert not mira.is_asleep
        assert any("woke up" in m for m in game.messages.messages)


class TestSolidPeople:
    def test_player_cannot_walk_through_an_npc(self, game):
        vance = park(game.world.find_npc("Guard Captain Vance"), 0.0, -3.0)
        game.player.position = spot(game.world, 0.0, 0.0)
        game.player.rotation_y = 0.0                               # facing -z, straight at him
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True))
        assert game.player.position.z > vance.position.z + 0.6    # stopped short of him, not through him
        assert game.player.position.z - vance.position.z == pytest.approx(0.7, abs=0.25)   # radii 0.3 + 0.4

    def test_sleepers_are_not_in_the_way_their_beds_are(self, game):
        mira = game.world.find_npc("Guard Mira")
        assert mira.is_asleep
        boxes = game.world.npc_colliders()
        assert len(boxes) == sum(1 for n in game.world.npcs if not n.is_asleep)

    def test_npcs_converging_on_one_spot_stand_around_it(self, game):
        """Every guard ordered to the same point ends up near it, none inside another."""
        target = spot(game.world, 20.0, 0.0)
        guards = game.world.guards()
        for i, g in enumerate(guards):
            park(g, 8.0 + 3.0 * i, -4.0 + (i % 3), state=AIState.PATROL)
            assert g.go_to(game.world, target, AIState.RESPONDING)
        game.player.position = spot(game.world, -30.0, -30.0)        # out of the way
        for _ in range(200):
            game.world.update_npcs(0.05)
        for g in guards:
            assert g.position.distance_to(target) < 3.0, g.name       # (seven of them now: a wider ring)
        for a in guards:
            for b in guards:
                if a is not b:
                    assert a.position.distance_to(b.position) >= 0.75, (a.name, b.name)

    def test_npc_steps_around_someone_in_the_way(self, game):
        vance = park(game.world.find_npc("Guard Captain Vance"), 20.0, -4.0, state=AIState.PATROL)
        game.player.position = spot(game.world, 20.0, 0.0)           # standing on his line
        vance.go_to(game.world, spot(game.world, 20.0, 4.0), AIState.INVESTIGATING)
        closest = 99.0
        for _ in range(120):
            game.world.update_npcs(0.05)
            closest = min(closest, vance.position.distance_to(spot(game.world, 20.0, 0.0)))
        assert vance.position.distance_to(spot(game.world, 20.0, 4.0)) < 0.5        # got there ...
        assert closest >= 0.69                                                    # ... without stepping into the player

    def test_a_hemmed_in_npc_gives_up_and_retries_on_the_hour(self, game):
        vance = park(game.world.find_npc("Guard Captain Vance"), 20.0, -4.0, state=AIState.PATROL)
        from unprepared.entities import ScheduleTask
        task = ScheduleTask(0, 24, spot(game.world, 20.0, 4.0), AIState.GUARDING, "post")
        vance.schedule = [task]
        world = game.world
        real = world.bodies_near
        import math
        world.bodies_near = lambda npc, radius=6.0: [                          # a tight ring of people around him
            (npc.position.x + math.cos(a), npc.position.z + math.sin(a), 0.55) for a in [i * math.pi / 4 for i in range(8)]]
        vance.begin_task(task, world)
        assert vance.state is AIState.TRAVELING
        for _ in range(100):                                       # 5 s: still patiently blocked
            world.update_npcs(0.05)
        assert vance.state is AIState.TRAVELING and vance.position.distance_to(spot(game.world, 20.0, -4.0)) < 0.1
        for _ in range(100):                                       # 10 s in all: patience runs out
            world.update_npcs(0.05)
        assert vance.state is AIState.IDLE
        world.bodies_near = real                                   # the crowd disperses
        vance.update_schedule(world.clock.hour, world=world)       # the hour tick re-plans
        for _ in range(200):
            world.update_npcs(0.05)
        assert vance.state is AIState.GUARDING and vance.position.distance_to(task.target_pos) < 0.5


class TestBarredGates:
    def test_the_cell_door_is_bars(self, game):
        door = game.world.find_door("cell_door")
        assert door.kind is DoorKind.BARS and door.is_bars and not door.auto_closes
        assert not door.blocks_sight                               # you can see through bars ...
        assert door.collider() is not None                         # ... but not walk through them
        assert door.collider() not in game.world.sight_blockers()

    def test_an_npc_walking_through_open_bars_leaves_them_open(self, game):
        hesk = park(game.world.find_npc("Jailer Hesk"), 51.5, 8.0, state=AIState.GUARDING)
        door = game.world.find_door("cell_door")
        door.is_locked = False
        door.is_open = True
        hesk.go_to(game.world, spot(game.world, 55.5, 13.0), AIState.INVESTIGATING)      # into the cell
        for _ in range(120):
            game.world.update_npcs(0.05)
        assert game.world.area_of(hesk.position) == "the jail cell"
        assert door.is_open and not door.is_locked                 # bars are never shut as a side effect
        hesk.go_to(game.world, spot(game.world, 51.5, 8.0), AIState.INVESTIGATING)       # and back out
        for _ in range(150):
            game.world.update_npcs(0.05)
        assert door.is_open and not door.is_locked

    def test_a_leaf_door_is_still_shut_behind_an_npc(self, game):
        """The ordinary door code is untouched: Hesk shuts the jail door behind himself."""
        hesk = park(game.world.find_npc("Jailer Hesk"), 51.5, 9.0, state=AIState.GUARDING)
        door = game.world.find_door("jail_door")
        assert not door.is_locked and not door.is_open
        hesk.go_to(game.world, spot(game.world, 53.0, 3.0), AIState.INVESTIGATING)
        opened = False
        for _ in range(150):
            game.world.update_npcs(0.05)
            opened = opened or door.is_open
        assert opened and not door.is_open and not door.is_locked   # opened to pass, closed again, never locked


class TestCurfew:
    def test_a_welcome_guest_is_asked_to_leave_late_and_finally_shown_the_door(self, game):
        jae = game.world.find_npc("Jae the Fence")
        game.player.position = spot(game.world, -12.0, 12.0)         # in the hideout, which Jae trusts us in
        game.world.simulated_hour = 21
        game.world.resync_schedules(announce=False, teleport=True)
        for _ in range(20):
            game.update(0.1)
        assert jae.state is AIState.SHOPKEEPING and not any("It's late" in m for m in game.messages.messages)
        game.world.update_time(24.0)                               # 22:00: Jae's supper and the curfew
        for _ in range(5):
            game.update(0.1)
        assert jae.state is AIState.WAITING and jae.waiting_for == "curfew"
        assert said(game.messages.messages, "leave_late")
        assert jae.faces(game.player.position)
        for _ in range(85):
            game.update(0.1)                                       # 8.5 s: he asks again
        assert said(game.messages.messages, "leave_again")
        assert not game.in_transition
        for _ in range(85):
            game.update(0.1)                                       # and then throws us out
        assert said(game.messages.messages, "leave_now")
        assert game.in_transition
        for _ in range(20):
            game.update(0.1)
        assert game.world.area_of(game.player.position) != "the hideout"
        assert jae.state is not AIState.WAITING                    # back to his evening

    def test_leaving_when_asked_ends_it(self, game):
        jae = game.world.find_npc("Jae the Fence")
        game.player.position = spot(game.world, -12.0, 12.0)
        game.world.simulated_hour = 22
        game.world.resync_schedules(announce=False, teleport=True)
        for _ in range(5):
            game.update(0.1)
        assert jae.state is AIState.WAITING
        game.player.position = spot(game.world, -12.0, 3.0)          # we go
        for _ in range(5):
            game.update(0.1)
        assert jae.state is AIState.EATING and jae.waiting_for is None
        assert sum(1 for m in game.messages.messages if said([m], "leave_late")) == 1

    def test_curfew_hours_wrap_midnight(self, game):
        area = game.world.find_area("the hideout")
        assert area.curfew_hours == (22, 6)
        assert area.curfew_active(23) and area.curfew_active(2) and not area.curfew_active(12)
        assert game.world.player_unwelcome_at(spot(game.world, -12.0, 12.0)) is None     # trusted by day ...
        game.world.simulated_hour = 23
        assert game.world.player_unwelcome_at(spot(game.world, -12.0, 12.0)) is area     # ... not after curfew


class TestCriesForHelp:
    def test_only_guards_within_earshot_come_running(self, game):
        far = park(game.world.find_npc("Guard Mira"), 56.0, -44.5, state=AIState.GUARDING)     # indoors, up the hill
        near = park(game.world.find_npc("Guard Orrin"), 5.0, 6.0, state=AIState.GUARDING)
        park(game.world.find_npc("Guard Captain Vance"), -38.0, 38.0)
        park(game.world.find_npc("Jailer Hesk"), -38.0, -38.0)
        where = spot(game.world, -12.0, 10.0)                                            # inside the hideout
        assert game.world.can_hear_cry(near, where)
        assert not game.world.can_hear_cry(far, where)
        game.world.declare_crime("Theft", where, witnessed=False, reporter="Jae the Fence")
        assert near.state is AIState.RESPONDING and near.crime_memories
        assert far.state is AIState.GUARDING and not far.crime_memories

    def test_a_shopkeepers_theft_report_brings_a_guard_to_take_it(self, game):
        from unprepared.entities import TheftReport
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        orrin = park(game.world.find_npc("Guard Orrin"), 6.0, 4.0, state=AIState.GUARDING)
        for g in game.world.guards():
            if g is not orrin:
                park(g, 110.0, 10.0)                                                    # far out in the country
        park(barnaby, -9.0, -4.0, state=AIState.SHOPKEEPING)
        orrin.remember_sighting("behind the counter", game.world.day_count)          # he saw us lurking there
        report = TheftReport(barnaby.name, "behind the counter", game.world.day_count, {"wine": 2}, {"wine": "Bottle of Wine"})
        game.world.report_theft(report, "opening up")
        assert orrin.state is AIState.RESPONDING and orrin.goal is not None and orrin.report_to is report
        assert orrin.goal.distance_to(barnaby.position) < 0.1
        assert orrin.pending_confrontations == [report]
        for _ in range(200):                                                           # he runs over and takes the report
            game.update(0.25)
            if orrin.conversation is not None:
                break
        assert orrin.conversation is not None and orrin.conversation.participants == (orrin, barnaby)
        assert orrin.faces(barnaby.position) and barnaby.faces(orrin.position)
        for _ in range(200):
            game.update(0.25)
            if orrin.conversation is None:
                break
        assert any("I saw someone skulking about here myself" in m or "who had no business there" in m
                   for m in game.messages.messages)


class TestPurses:
    def test_everyone_carries_coin(self, game):
        gold = {n.name: n.gold for n in game.world.npcs}
        assert gold["Shopkeeper Barnaby"] == 400.0 and gold["Jae the Fence"] == 600.0
        assert all(gold[n] == 15.0 for n in ("Guard Captain Vance", "Guard Orrin", "Guard Mira", "Jailer Hesk"))

    def test_purses_are_topped_up_every_restock_cycle_but_never_cut(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        jae = game.world.find_npc("Jae the Fence")
        barnaby.gold = 20.0                                        # cleaned out
        jae.gold = 900.0                                           # flush
        game.world.day_count = 2
        game.world._daily_upkeep()
        assert barnaby.gold == 20.0 and jae.gold == 900.0          # too soon (3-day cycle)
        game.world.day_count = 3
        game.world._daily_upkeep()
        assert barnaby.gold == 400.0 and jae.gold == 900.0         # restored, or kept if above the minimum
        assert barnaby.last_gold_restock_day == 3

    def test_purses_survive_a_save(self, game):
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        barnaby.gold = 123.0
        game.save_game()
        barnaby.gold = 0.0
        game.load_game()
        assert barnaby.gold == 123.0


class TestSavedDoorsStayAsSaved:
    def test_a_picked_lock_survives_a_save_and_load(self, game):
        """Loading must not let NPCs re-apply their task's lock/unlock side effects."""
        door = game.world.find_door("home_door")
        barnaby = game.world.find_npc("Shopkeeper Barnaby")
        game.world.simulated_hour = 23
        game.world.resync_schedules(announce=False, teleport=True)       # Barnaby asleep, house locked
        assert barnaby.is_asleep and door.is_locked
        door.unlock_and_open()                                           # picked and opened
        door.is_open = False
        game.player.position = spot(game.world, 22.0, 14.0)
        game.save_game()
        door.lock_and_close()
        assert game.load_game()
        assert not door.is_locked and barnaby.is_asleep                  # exactly as saved
        for _ in range(20):
            game.update(0.1)
        assert not door.is_locked

    def test_an_npc_arriving_after_the_load_still_locks_up(self, game):
        """The exemption is only for the moment of loading: routines carry on afterwards."""
        door = game.world.find_door("home_door")
        game.world.simulated_hour = 20
        game.world.resync_schedules(announce=False, teleport=True)       # dinner time: house unlocked
        assert not door.is_locked
        game.save_game()
        assert game.load_game()
        game.world.wait_hours(3)                                         # 23:00: he went to bed (at 22) and locked up
        assert door.is_locked


class TestFencesHouse:
    def test_guards_never_evict_the_player_from_the_fences_house_by_day(self, game):
        vance = park(game.world.find_npc("Guard Captain Vance"), -12.0, 11.0, state=AIState.PATROL)
        game.world.simulated_hour = 12
        game.player.inventory = [i for i in game.player.inventory if not i.is_stolen]   # (no stolen goods on us)
        game.player.position = spot(game.world, -12.0, 9.0)                # Jae's den
        vance.remember_crime("Pickpocketing", game.world.day_count, True, 25.0)
        vance.turn_towards(game.player.position)
        game.update(0.05)
        assert game.player.gold == STARTING_GOLD - 25.0                  # fined ...
        assert not game.in_transition and not game.world.relocating      # ... but not thrown out of a friend's house
        assert game.world.player_unwelcome_at(game.player.position) is None
        assert not any("no business" in m for m in game.messages.messages)

    def test_the_rule_is_the_owner_being_a_fence_not_the_trust_list(self, game):
        from unprepared.world.level import AreaDef
        area = AreaDef("den", (0, 0), (1, 1), owner="Jae the Fence", restricted=True)
        assert game.world.area_trusts_player(area)
        other = AreaDef("house", (0, 0), (1, 1), owner="Shopkeeper Barnaby", restricted=True)
        assert not game.world.area_trusts_player(other)


class TestIndoorLights:
    def test_a_lantern_per_room_and_a_candle_per_table(self, game):
        w = game.world
        fixtures = w.fixtures
        lanterns = {f.room for f in fixtures if f.kind == "lantern"}
        candles = {f.room for f in fixtures if f.kind == "candle"}
        assert {"the smithy", "the inn kitchen", "the keeper's room", "room one", "room two", "room three",
                "the inn", "behind the counter", "the shop", "the tailor's back room", "the tailor's",
                "the guard house", "Barnaby's house", "Tamsin's cottage", "the hideout", "Fenwick's cottage",
                "Lys and Dorran's house", "Pell's cottage", "the jail cell", "the jail", "the watchtower",
                "Vane Manor kitchen", "Vane Manor store", "Vane Manor study", "Vane Manor bedroom", "Ashgrove Manor kitchen",
                "Ashgrove Manor servant's room", "Ashgrove Manor dressing room", "Ashgrove Manor second bedroom", "the old granary",
                "the chapel", "Odo's farmhouse", "the keep hall", "the Greyhold treasury"} <= lanterns
        assert not {"the smugglers' cave", "the Howling Deep", "the Deep King's hall"} & lanterns    # caves: their fires only
        assert {"shop_table", "jae_table", "barnaby_table", "guard_table", "inn_table_1", "smith_table"} <= candles
        assert [f.room for f in fixtures if f.kind == "fire"][0] == "the smith's yard"        # the kiln, from the level
        assert not any(f.room == "smithy_shed" for f in fixtures)                             # no lantern for a roof on posts
        for f in fixtures:
            pos = Vector3D(*f.position)
            if f.kind in ("fire", "torch"):
                continue                                          # (fires burn in yards; torches on outside walls)
            assert w.is_indoors(pos), f
            if f.kind == "lantern":
                building = w.building_at(pos)
                storey = building.wall_height / max(1, building.floors)
                above = (f.position[1] - w.building_elevation(building) - 2.9) / storey   # (on an upper floor: a storey up)
                assert (above == pytest.approx(round(above)) or f.room == "the watchtower"), f     # (a stair's)
                assert not any(wl.min_point.x <= pos.x <= wl.max_point.x and wl.min_point.z <= pos.z <= wl.max_point.z
                               and wl.min_point.y <= pos.y <= wl.max_point.y
                               for wl in w.walls), f              # hangs inside the room, not inside a wall

    def test_wall_torches_burn_outside_by_night_only(self, game):
        w = game.world
        torches = [f for f in w.fixtures if f.kind == "torch"]
        assert len(torches) >= 40                                  # doors, the curtain wall, the gate, the keep
        for f in torches:
            assert not w.is_indoors(Vector3D(*f.position)), f
        w.simulated_hour = 12
        assert not w.is_dark_at(spot(w, 0.0, 3.0))
        w.simulated_hour = 23
        assert w.is_dark_at(spot(w, 0.0, 3.0))                     # the torches light the way, not the law

    def test_lanterns_avoid_the_door_wall(self, game):
        house = next(f for f in game.world.fixtures if f.room == "Barnaby's house" and f.kind == "lantern")
        assert house.position[2] > 14.0 and house.facing == 0.0          # south wall; the door is on the north wall
        hall = next(f for f in game.world.fixtures if f.room == "the inn" and f.kind == "lantern")
        assert hall.position[0] > -23.0                                  # the hall's lantern hangs on the east wall ...
        assert not any(r.contains(hall.position[0], hall.position[2]) for r in game.world.level.areas
                       if r.name in ("room one", "room two", "room three", "the inn kitchen", "the keeper's room"))   # ... not in a guest room

    def test_explicit_lights_and_the_rule_can_be_switched_off(self):
        import dataclasses
        from unprepared.world import World, get_level
        from unprepared.world.level import LightDef
        level = dataclasses.replace(get_level("town"), auto_lights=False,
                                    lights=(LightDef("candle", (0.0, 0.8, 0.0), room="square"),))
        world = World(level, GameSettings(echo_messages_to_stdout=False))
        assert [f.room for f in world.fixtures] == ["square"]

    def test_dark_is_night_outdoors_or_an_unlit_room(self, game):
        w = game.world
        w.simulated_hour = 12
        assert not w.is_dark_at(spot(game.world, 0.0, 3.0))
        assert not w.is_dark_at(spot(game.world, -13.0, -9.0))                # the shop: lit by its lanterns
        w.simulated_hour = 23
        assert w.is_dark_at(spot(game.world, 0.0, 3.0))
        assert not w.is_dark_at(spot(game.world, -13.0, -9.0))                # still lit indoors
        w.fixtures = []
        assert w.is_dark_at(spot(game.world, -13.0, -9.0))                    # an unlit room is dark even at noon
        w.simulated_hour = 12
        assert w.is_dark_at(spot(game.world, -13.0, -9.0))

    def test_npcs_light_torches_wherever_it_is_dark(self, game):
        vance = park(game.world.find_npc("Guard Captain Vance"), 0.0, 3.0, state=AIState.GUARDING)
        game.world.simulated_hour = 12
        game.update(0.05)
        assert not vance.has_torch                                       # noon, outdoors
        game.world.simulated_hour = 23
        game.update(0.05)
        assert vance.has_torch                                           # night, outdoors
        vance.position = spot(game.world, -13.0, -9.0)                        # steps into the lit shop
        game.update(0.05)
        assert not vance.has_torch
        game.world.fixtures = []                                         # an unlit building at noon
        game.world.simulated_hour = 12
        game.update(0.05)
        assert vance.has_torch
        vance.state = AIState.SLEEPING
        game.update(0.05)
        assert not vance.has_torch                                       # sleepers carry none
