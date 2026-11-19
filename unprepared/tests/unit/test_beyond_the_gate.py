"""The wider realm: the town gate (barred until the tutorial is done), Jae's job on the rich,
the two families on the hill, their two-storey houses, the outlaws and knights who keep their
own peace out in the country, and the legendary things kept deepest of all."""
from __future__ import annotations

import pytest

from tests.conftest import said
from tests.helpers import spot, stand_by_door, teleport
from unprepared.core import dialogue
from unprepared.entities import AIState, Item
from unprepared.game import Game
from unprepared.quests import ACTIVE, AVAILABLE, DONE, STICKING_IT_TO_THEM
from unprepared.systems import MoveIntent
from unprepared.tutorial import GATE_CLOSED_TEXT, Stage
from unprepared.world.levels import town, wilds


@pytest.fixture
def tgame(settings, tmp_path):
    return Game(settings=settings, save_file=str(tmp_path / "savegame.json"), tutorial=True)


def settle(game, seconds, dt=0.1, intent=MoveIntent()):
    for _ in range(int(seconds / dt)):
        game.update(dt, intent)


# ----------------------------------------------------------------------------- the town gate
class TestTheTownGate:
    def test_two_leaves_meet_where_the_main_road_reaches_the_south_wall(self, world):
        west, east = world.gate_leaves("town_gate")
        assert west.hinge_position.z == east.hinge_position.z == town.GATE_Z
        assert (west.x1, west.x2, east.x1, east.x2) == (-4.0, 0.0, 0.0, 4.0) and east.hinge_right
        gate_road = [r for r in world.level.roads if r.min_xz[0] == -3.0 and r.max_xz[0] == 3.0]
        assert max(r.max_xz[1] for r in gate_road) >= town.TOWN_Z2 - 1.0          # the main road runs up to it
        assert not west.auto_closes and not east.auto_closes                      # a gate stands as the watch leaves it

    def test_a_new_game_without_the_tutorial_finds_it_open(self, game):
        assert all(d.is_open and not d.is_locked for d in game.world.gate_leaves("town_gate"))

    def test_two_guards_keep_it_round_the_clock_changing_sides(self, world):
        bram, wenna = world.find_npc("Guard Bram"), world.find_npc("Guard Wenna")
        posts = {h: {n.name: n.task_for(h).target_pos for n in (bram, wenna) if n.task_for(h).target_state is AIState.GUARDING}
                 for h in range(24)}
        assert all(posts[h] for h in range(24))                                   # never unmanned
        sides = {round(p.x) for h in range(6, 18) for p in [posts[h]["Guard Bram"]]}
        assert sides == {-6, 6}                                                   # he changes sides through the day
        assert all(world.area_of(p) == "the town gate" for h in posts for p in posts[h].values())

    def test_during_the_tutorial_the_gate_is_barred_and_the_guards_turn_you_away(self, tgame):
        w, p = tgame.world, tgame.player
        assert all(d.is_locked and not d.is_open for d in w.gate_leaves("town_gate"))
        wenna = w.find_npc("Guard Wenna")                                         # on the night watch at the gate
        stand_by_door(tgame, "town_gate_west", side="inside", distance=2.0)
        tgame.interact()
        assert tgame.target is not None and tgame.target.obj.gate == "town_gate"
        last = tgame.messages.messages[-1]
        assert last.endswith(GATE_CLOSED_TEXT) and last.startswith("(Guard ")
        assert said([last], "gate_refusal")
        assert all(d.is_locked and not d.is_open for d in w.gate_leaves("town_gate"))
        # crouching to pick it is refused just the same (a legendary lock besides)
        p.is_crouching = True
        tgame.update(0.05)
        tgame.interact()
        assert not tgame.in_lockpicking and tgame.messages.messages[-1].endswith(GATE_CLOSED_TEXT)
        assert wenna.speech is not None or any("Guard" in m for m in tgame.messages.messages[-2:])

    def test_even_after_the_sale_the_gate_waits_for_the_end_of_the_tutorial(self, tgame):
        t = tgame.tutorial
        t._enter(Stage.TOOLS)                                                    # the candlestick is sold: the town opens ...
        assert not t.locked
        stand_by_door(tgame, "town_gate_west", side="inside", distance=2.0)
        tgame.interact()
        assert tgame.messages.messages[-1].endswith(GATE_CLOSED_TEXT)             # ... but not the gate
        t._finish()
        assert all(d.is_open and not d.is_locked for d in tgame.world.gate_leaves("town_gate"))
        assert "town gate stands open" in tgame.messages.messages[-1]

    def test_nobody_leaves_over_the_walls_or_down_the_millstream(self, world):
        """The grates over the culverts are solid; the walls are high."""
        grates = [p for p in world.props if p.kind == "grate"]
        assert len(grates) == 2 and all(p.solid for p in grates)
        for grate in grates:
            assert grate in world.static_obstacles()
        x1, z1, x2, z2 = world.level.town_area
        assert all(w.max_point.y - world.ground_height((w.min_point.x + w.max_point.x) / 2, (w.min_point.z + w.max_point.z) / 2)
                   >= 4.5 for w in world.walls if w.min_point.z == z1 and w.max_point.z == z1 + 1.0)

    def test_the_gate_state_is_saved(self, game):
        for door in game.world.gate_leaves("town_gate"):
            door.is_open, door.is_locked = False, True
        assert game.save_game()
        game.world.open_gate("town_gate")
        assert game.load_game()
        assert all(d.is_locked and not d.is_open for d in game.world.gate_leaves("town_gate"))


# ----------------------------------------------------------------------------- Jae's job
class TestStickingItToTheJerks:
    def test_no_job_while_the_tutorial_runs(self, tgame):
        assert tgame.quests.stage(STICKING_IT_TO_THEM) == AVAILABLE
        jae = tgame.world.find_npc("Jae the Fence")
        assert not tgame.quests.talk(jae) and tgame.quests.stage(STICKING_IT_TO_THEM) == AVAILABLE
        assert tgame.guide_title == "TUTORIAL"

    def test_jae_offers_it_when_spoken_to_and_the_guide_shows_the_way(self, game):
        jae = game.world.find_npc("Jae the Fence")
        teleport(game, jae.position.x, jae.position.z - 2.5, yaw=180.0, pitch=-20.0)
        game.interact()                                                           # [E]: the trade opens as ever ...
        assert game.in_trade_menu and game.quests.stage(STICKING_IT_TO_THEM) == ACTIVE
        assert any(m.startswith("QUEST: Sticking It to the Jerks") for m in game.messages.messages)
        assert game.guide_title == "QUEST: Sticking It to the Jerks" and "Ravenfeather Cloak" in game.guide_hint
        game.close_transfer()
        settle(game, 20.0)                                                         # ... and he tells the rest of it
        assert sum(1 for m in game.messages.messages if m.startswith("(Jae the Fence)")) >= 4
        door = game.world.find_door("vane_door")
        assert game.waypoint.x == door.center.x and game.waypoint.z == door.hinge_position.z
        teleport(game, 16.0, -50.0)                                                # inside Vane Manor: the wardrobe chest
        chest = game.world.find_container("vane_wardrobe")
        assert (game.waypoint.x, game.waypoint.z) == (chest.center.x, chest.center.z)

    def test_the_cloak_is_kept_upstairs_in_vane_manor(self, world):
        chest = world.find_container("vane_wardrobe")
        cloak = next(i for i in chest.inventory if i.id == "ravenfeather_cloak")
        assert cloak.rarity == "rare" and cloak.owner == "Lord Aurelian Vane" and cloak.bonus_for("Sneak") > 0
        assert chest.is_locked and world.building_at(chest.center).name == "vane"
        assert chest.min_point.y >= world.building_elevation(world.building_at(chest.center)) + 3.9      # on the upper floor

    def test_handing_it_in_pays_and_the_cloak_is_the_players_to_keep(self, game):
        q = game.quests
        q.stages[STICKING_IT_TO_THEM.id] = ACTIVE
        chest = game.world.find_container("vane_wardrobe")
        cloak = next(i for i in chest.inventory if i.id == "ravenfeather_cloak")
        chest.inventory.remove(cloak)
        cloak.steal()
        game.player.add_item(cloak)
        assert "back to Jae" in game.guide_hint
        gold = game.player.gold
        jae = game.world.find_npc("Jae the Fence")
        teleport(game, jae.position.x, jae.position.z - 2.5, yaw=180.0, pitch=-20.0)
        game.interact()
        assert not game.in_trade_menu                                              # the hand-in took the conversation
        assert q.stage(STICKING_IT_TO_THEM) == DONE and game.player.gold == gold + STICKING_IT_TO_THEM.reward_gold
        assert cloak in game.player.inventory and cloak.is_players and not cloak.is_stolen    # honestly the player's now
        assert any(m.startswith("QUEST COMPLETE") for m in game.messages.messages)
        settle(game, 6.0)
        assert any("just for sticking it to those jerks" in m for m in game.messages.messages)
        assert game.guide_hint == "" and game.waypoint is None

    def test_selling_it_to_him_over_the_counter_finishes_the_job_too(self, game):
        q = game.quests
        q.stages[STICKING_IT_TO_THEM.id] = ACTIVE
        jae = game.world.find_npc("Jae the Fence")
        jae.inventory.append(Item("ravenfeather_cloak", "Ravenfeather Cloak", value=450.0))
        gold = game.player.gold
        game.update(0.0)
        assert q.stage(STICKING_IT_TO_THEM) == DONE and game.player.gold == gold + STICKING_IT_TO_THEM.reward_gold

    def test_the_job_survives_a_save(self, game):
        game.quests.stages[STICKING_IT_TO_THEM.id] = ACTIVE
        assert game.save_game()
        game.quests.stages[STICKING_IT_TO_THEM.id] = AVAILABLE
        assert game.load_game() and game.quests.stage(STICKING_IT_TO_THEM) == ACTIVE


# ----------------------------------------------------------------------------- the rich
class TestTheFamiliesOnTheHill:
    def test_two_rich_pompous_neighbours_in_finery(self, world):
        for name, prefix in (("Lord Aurelian Vane", "vane"), ("Lady Seraphine Ashgrove", "ashgrove")):
            npc = world.find_npc(name)
            assert npc.role_tag == "rich" and npc.gold >= 200 and any(i.key_door_id == f"{prefix}_door" for i in npc.inventory)
            worn = {slot.value: item.gear for slot, item in npc.equipment.items()}
            assert worn["body"] in ("doublet", "gown") and worn["head"] in ("feathered_hat", "circlet")
            assert worn["neck"] in ("mantle", "choker")                               # jewellery and furs

    def test_they_talk_down_to_everyone_guards_included(self, world):
        import random
        vane, tamsin, orrin = (world.find_npc(n) for n in ("Lord Aurelian Vane", "Tamsin", "Guard Orrin"))
        rng = random.Random(3)
        for other, topic in ((tamsin, "rich"), (orrin, "rich_guard")):
            for a, b in ((vane, other), (other, vane)):
                conv = dialogue.build_conversation(a, b, 14, rng, exchanges=1)
                opener_speaker, opener = conv.lines[2]
                assert opener_speaker is vane and opener in {o.format(other=dialogue.short_name(other)) for o, _ in dialogue.TOPICS[topic]}
        vane.state = AIState.STROLLING                                            # (up and about)
        assert vane.greeting(12) in dialogue.LINES["greet_rich"]
        assert vane.greeting(23) in dialogue.LINES["greet_rich_night"]

    def test_their_houses_are_two_storeys_and_harder_to_get_into(self, world):
        for prefix in ("vane", "ashgrove"):
            house = next(b for b in world.level.buildings if b.name == prefix)
            assert house.floors == 2 and house.wall_height == 8.0
            door = world.find_door(f"{prefix}_door")
            assert door.lock_difficulty == town.MASTER and door.lock_difficulty > world.find_door("tailor_door").lock_difficulty
            box = world.find_container(f"{prefix}_strongbox")
            assert box.lock_difficulty == town.MASTER and any(i.category == "coins" for i in box.inventory)
            assert any(i.category == "coins" for i in world.find_container(f"{prefix}_coffer").inventory)

    def test_the_lord_climbs_to_his_bed_upstairs(self, game):
        w = game.world
        vane = w.find_npc("Lord Aurelian Vane")
        base = w.building_elevation(next(b for b in w.level.buildings if b.name == "vane"))
        w.clock.hour, w.clock.minute_accumulator = 21, 50.0
        w.resync_schedules(announce=False, teleport=True)
        for _ in range(int(60 / 0.1)):
            game.update(0.1)
            if vane.is_asleep:
                break
        assert vane.is_asleep and vane.position.y == pytest.approx(base + 4.0)
        assert w.prop_under(vane.position).name == "vane_bed"
        assert w.find_door("vane_door").is_locked                                  # locked up for the night

    def test_the_player_can_walk_up_the_stairs(self, game):
        w, p = game.world, game.player
        base = w.building_elevation(next(b for b in w.level.buildings if b.name == "vane"))
        for n in w.npcs:
            n.schedule, n.current_task, n.path = [], None, []
            n.position = spot(w, 110.0, 10.0)
        stair = next(s for s in w.level.stairs if s.name == "vane_stair")
        teleport(game, stair.bottom[0], stair.bottom[2], yaw=0.0)                   # at the foot of the stair
        settle(game, 4.0, dt=0.05, intent=MoveIntent(forward=True))
        assert p.position.y == pytest.approx(base + 4.0)                            # on the upper floor
        assert w.building_at(p.position).name == "vane"


# ----------------------------------------------------------------------------- wardens
class TestWardens:
    def _at_the_stash(self, game):
        w, p = game.world, game.player
        w.clock.hour = 20
        w.resync_schedules(announce=False, teleport=True)
        kell = w.find_npc("Kell the Smuggler")
        assert kell.warden_of == ("the smugglers' cave",) and not kell.is_guard
        return w, p, kell

    def test_outlaws_throw_you_out_take_back_their_goods_and_a_toll(self, game):
        w, p, kell = self._at_the_stash(game)
        p.gold = 100.0
        stash = w.find_container("smugglers_stash")
        hood = next(i for i in stash.inventory if i.id == "shadowsilk_hood")
        stash.inventory.remove(hood)
        hood.steal()
        p.add_item(hood)
        teleport(game, kell.position.x + 0.5, kell.position.z + 1.4, yaw=0.0)      # right in front of him, in his cave
        assert w.area_of(p.position) == "the smugglers' cave"
        settle(game, 3.0)
        assert said(game.messages.messages, "warden_warn_bandit")                # a warning first ...
        settle(game, 10.0)                                                         # ... then hands laid on
        assert hood not in p.inventory and any(i.id == "shadowsilk_hood" for i in stash.inventory)
        assert p.gold == pytest.approx(75.0)                                       # a quarter of the purse
        assert any("You are dragged out of the smugglers' cave" in m for m in game.messages.messages)
        settle(game, 3.0)
        assert w.area_of(p.position) != "the smugglers' cave"
        assert p.position.distance_to(w.ground_point(*wilds.SMUGGLERS_EXIT[::2])) < 1.0
        assert not any(g.crime_memories for g in w.guards())                        # the town watch knows nothing of it
        assert not w.jail.serving

    def test_a_warden_who_loses_sight_of_you_lets_it_go(self, game):
        w, p, kell = self._at_the_stash(game)
        kell.raise_alarm(None)
        kell.warden_seize_at = w.elapsed
        teleport(game, 0.0, 4.0)                                                    # far away, back in town
        settle(game, w.WARDEN_GIVE_UP + 2.0, dt=0.5)
        assert kell.warden_seize_at is None and not kell.is_alarmed

    def test_the_realm_keeps_its_legends_deepest(self, world):
        where = {}
        for c in world.containers:
            for i in c.inventory:
                if i.rarity == "legendary":
                    where[i.id] = c
        assert set(where) == {"amulet_open_door", "nightweave_cloak", "fingers_of_the_magpie"}
        assert world.area_of(where["amulet_open_door"].center) == "the Deep vault"                  # behind the bandit king
        assert world.area_of(where["nightweave_cloak"].center) == "the Greyhold treasury"           # in the old keep
        assert world.area_of(where["fingers_of_the_magpie"].center) == "Ashgrove Manor"             # a lady's strongbox
        vault = world.find_door("deep_vault_door")
        boss = world.find_npc("Grimmaw the Bandit King")
        assert vault.lock_difficulty == town.LEGENDARY and boss.role_tag == "boss" and boss.has_key_for(vault)
        # he stands in the hall before the vault door, and sleeps only three hours of the night
        assert world.area_of(boss.task_for(12).target_pos) == "the Deep King's hall"
        assert sum(1 for h in range(24) if boss.task_for(h).target_state is AIState.SLEEPING) == 3

    def test_a_legendary_piece_needs_a_master_thief(self, game):
        """A novice cannot even see how the amulet comes off its pedestal; a master with the best
        gear has a fair chance."""
        from unprepared.config import DEFAULT_SETTINGS
        from unprepared.systems.stealth import secure_verdict
        amulet = next(i for c in game.world.containers for i in c.inventory if i.id == "amulet_open_door")
        s, pr = DEFAULT_SETTINGS.stealth, DEFAULT_SETTINGS.progression
        _, odds, possible = secure_verdict(game.player, amulet, s, pr)
        assert not possible
        game.player.skills.skills[next(k for k in game.player.skills.skills if k.value == "Stealing")]["level"] = 60
        _, odds, possible = secure_verdict(game.player, amulet, s, pr)
        assert possible and 0.5 < odds < 1.0


# ----------------------------------------------------------------------------- the country
class TestTheCountry:
    def test_the_village_the_woods_the_caves_and_the_keep(self, world):
        names = {b.name for b in world.level.buildings}
        assert {"odo_house", "marta_house", "haskel_hut", "barn", "hunter_hut", "hrolf_hut", "greyhold_keep",
                "deep_hall", "deep_kings_hall", "deep_vault", "cove_den", "cove_store", "mine_gallery", "mine_sump"} <= names
        caves = [b for b in world.level.buildings if b.underground]
        assert {b.name.split("_")[0] for b in caves} == {"deep", "cove", "mine"}
        assert all(b.roof_style == "flat" and not b.lit for b in caves)
        assert all(world.find_door(d).lock_difficulty == town.FLIMSY for d in ("odo_door", "marta_door", "haskel_door"))
        trees = [p for p in world.props if p.kind in ("oak", "pine")]
        assert len(trees) > 150                                                    # a forest, not a copse
        x1, z1, x2, z2 = world.level.town_area
        in_town = [p for p in trees if x1 <= p.center.x <= x2 and z1 <= p.center.z <= z2]
        assert 20 <= len(in_town) < 60

    def test_paths_lead_from_the_gate_to_everything(self, world):
        """With every door open, the player could walk from the gate to each place out there - the
        village, the hermit smith, the lake, the shrine, the keep and into all three caves."""
        for door in world.doors:
            door.is_locked, door.is_open = False, True
        start = world.spot(*town.GATE_OUTSIDE)
        places = {"the village": (15.0, 73.0), "the hunter's hut": (wilds.HUNTER_SEAT[0], wilds.HUNTER_SEAT[2]),
                  "Hrolf's anvil": (wilds.HROLF_ANVIL[0], wilds.HROLF_ANVIL[2] + 1.5), "the shrine": (wilds.SHRINE[0], wilds.SHRINE[1] - 2.0),
                  "the keep": wilds.KEEP_TABLE_SEATS[0][::2], "the Deep's hall": wilds.DEEP_FIRE_SEATS[0],
                  "the Deep King's throne": wilds.BOSS_POST, "the vault": (wilds.VAULT_PEDESTAL[0], wilds.VAULT_PEDESTAL[1] + 1.5),
                  "the smugglers' den": wilds.COVE_FIRE_SEATS[0], "the smugglers' store": wilds.COVE_STORE_SPOT,
                  "the mine gallery": (100.0, -88.0), "the flooded sump": (92.0, -102.0), "the Deep's west cubby": (-170.0, -81.5)}
        for name, (x, z) in places.items():
            assert world.nav.find_path(start, world.spot(x, 0.0, z)), name

    def test_no_wall_fences_the_map_in_the_land_does(self, game):
        """Nothing built stands at the edge of the map: the mountain is too steep to climb to the
        north, the thicket too dense to the west and south, the sea too deep to the east."""
        w, p = game.world, game.player
        x1, z1, x2, z2 = wilds.BOUNDS
        assert not [wall for wall in w.walls if wall.min_point.x <= x1 + 2 or wall.max_point.x >= x2 - 2
                    or wall.min_point.z <= z1 + 2 or wall.max_point.z >= z2 - 2]
        for n in w.npcs:
            n.schedule, n.current_task, n.path = [], None, []
        for (x, z), yaw, stops in (((-120.0, -60.0), 0.0, lambda q: q.z > -85.0),          # up the Greyspine
                                   ((60.0, -62.0), 0.0, lambda q: q.z > -85.0),
                                   ((-185.0, 50.0), 270.0, lambda q: q.x > x1 + 2.0),     # into the thicket
                                   ((-20.0, 200.0), 180.0, lambda q: q.z < z2 - 2.0),
                                   ((140.0, 60.0), 90.0, lambda q: q.x < x2 - 10.0)):     # into the sea
            teleport(game, x, z, yaw=yaw)
            settle(game, 12.0, dt=0.05, intent=MoveIntent(forward=True))
            assert stops(p.position), ((x, z), p.position)
        assert w.terrain.water_level_at(170.0, 60.0) - w.terrain.height(170.0, 60.0) > 1.0   # out of your depth

    def test_the_ground_is_rock_on_the_steeps_sand_on_the_shore_snow_on_the_peaks(self, world):
        t = world.terrain
        assert t.rock is not None and t.rock.max() > 0.9
        assert t.sand is not None and t.sand.max() > 0.9
        assert t.snow is not None and t.snow.max() > 0.9
