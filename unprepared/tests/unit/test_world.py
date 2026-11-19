import math

import pytest

from unprepared.config import GameSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState
from unprepared.world import LevelDefinition, World, available_levels, get_level, register_level
from unprepared.world.world import WorldClock
from unprepared.world.levels import town, wilds
from unprepared.world.levels.town import BARNABY_TABLE_SEAT, GUARD_BEDS, PATROL_ROUTE, POST_DOOR, JAE_DEN
from unprepared.world.levels.town import STARTING_GOLD
from unprepared.world.spatial import BoxIndex


class TestLevelRegistry:
    def test_town_registered(self):
        assert "town" in available_levels()
        assert get_level("town").name == "town"

    def test_register_custom_level(self):
        tiny = LevelDefinition("tiny", walls=(), doors=(), npcs=(), player_spawn=(0, 1, 0))
        register_level("tiny", lambda: tiny)
        assert get_level("tiny") is tiny
        world = World(tiny, GameSettings())
        assert world.walls == [] and world.npcs == [] and world.doors == []
        assert world.spawn_player().inventory == []


class TestTownLevel:
    def test_geometry(self, world):
        x1, z1, x2, z2 = world.level.town_area
        north = [w for w in world.walls if w.texture in ("wall", "wall_cracked") and w.min_point.z == z1 and w.max_point.z == z1 + 1]
        assert north and min(w.min_point.x for w in north) == x1 and max(w.max_point.x for w in north) == x2   # the curtain wall
        assert world.level.bounds == wilds.BOUNDS and (x1, z1, x2, z2) == (-80.0, -62.0, 80.0, 54.0)
        bx1, bz1, bx2, bz2 = wilds.BOUNDS
        assert world.nav.cols == int((bx2 - bx1) / 0.5) and world.nav.rows == int((bz2 - bz1) / 0.5)
        assert {"well", "forge", "inn_bar", "inn_oven", "shop_counter", "tailor_counter"} <= {p.name for p in world.props}
        town_beds = [p for p in world.props if p.is_bed and world.in_town(p.center)]
        sleepers = [n for n in world.npcs if world.in_town(n.home_position)]
        # everyone in town has a bed, and there are some to spare: three guest rooms at the inn, a cell, the
        # manors' second bedrooms, the barracks' and the gate lodge's for the watch off duty
        assert len(sleepers) == 33 and len(town_beds) == 40
        for n in sleepers:
            for t in n.schedule:
                if t.target_state is AIState.SLEEPING:
                    p = t.target_pos
                    assert any(b.min_point.x - 0.3 <= p.x <= b.max_point.x + 0.3
                               and b.min_point.z - 0.3 <= p.z <= b.max_point.z + 0.3 for b in town_beds), n.name
        assert all(p.kind in ("bed", "bedroll") for p in world.props if p.is_bed)
        assert not any(p.is_bed for p in world.props if p.kind in ("rug", "painting", "planter"))   # (the bedroom's rug)
        assert {"shop_shelf_west", "shop_shelf_east", "smithy_rack", "tailor_rack", "jail_storage", "inn_kitchen_table"} <= \
            {c.name for c in world.containers}
        assert [s.name for s in world.stations] == ["ore_deposit", "anvil", "hrolf_anvil", "thornwood_ore", "quarry_ore",
                                                    "deep_silver_vein", "deep_iron_vein", "mine_ore_west", "mine_ore_east",
                                                    "mine_ore_stope", "mine_silver_vein"]
        solid = [p for p in world.props if p.solid]
        notices = [r for r in world.readables if r.solid]                                    # (a notice board, a signpost)
        assert notices and len(world.static_obstacles()) == (len(world.walls) + len(solid) + len(world.containers)
                                                             + len(world.stations) + len(notices))
        assert all(not p.solid for p in world.props if p.kind in ("step", "floor", "sign", "bridge", "crack", "stain"))
        high_floors = [f for f in world.floors if f.max_point.y > world.ground_height(f.center.x, f.center.z) + 0.5]
        everything = world.colliders(include_npcs=False)
        assert len(everything) >= len(world.static_obstacles()) + len(world.doors)
        awake = sum(1 for n in world.npcs if not n.is_asleep)
        assert len(world.colliders()) == len(everything) + awake                             # people are solid too
        names = [b.name for b in world.level.buildings]
        assert names[:3] == ["smithy", "smithy_shed", "inn"] and "jail" in names and "guard_house" in names
        tower = town.WATCHTOWER
        assert all(town.TOWER_X1 <= w.min_point[0] and w.max_point[0] <= town.TOWER_X2 and town.TOWER_Z1 <= w.min_point[2]
                   and w.max_point[2] <= town.TOWER_Z2 for w in tower.walls)
        gx1, gz1, gx2, gz2 = town.GUARD_HOUSE
        assert gx1 <= town.TOWER_X1 and town.TOWER_X2 <= gx2 and town.TOWER_Z2 == gz1      # behind the guard house
        assert next(b for b in world.level.buildings if b.name == "smithy_shed").open_sides
        assert high_floors                                                                   # the tower's and the manors' upper steps

    def test_street_and_doorsteps(self, world):
        main = [r for r in world.level.roads if r.min_xz[1] == -6.0 and r.max_xz[1] == 6.0]
        assert min(r.min_xz[0] for r in main) < -70 and max(r.max_xz[0] for r in main) > 70     # the long street
        steps = world.roads[len(world.level.roads):]
        assert steps, "south-side doors set back from the street get a doorstep"
        home = world.find_door("home_door")                                  # hinge z = 7: a one-unit step
        assert any(r.min_xz[0] < home.center.x < r.max_xz[0] and r.min_xz[1] == 6.0 and r.max_xz[1] == 7.0 for r in steps)
        shop = world.find_door("shop_door")
        assert not any(r.min_xz[1] < -6.0 < r.max_xz[1] and r.min_xz[0] < shop.center.x < r.max_xz[0] for r in steps)   # onto it

    def test_nothing_overlaps(self, world):
        """No two pieces of furniture, stations or walls-vs-furniture intersect (in three dimensions:
        a chest upstairs stands over the hall, not in it). What grows wild - trees and bushes in a
        thicket, rock heaped on rock, mushrooms in a crack - may crowd together and into the rock of
        a cave's walls (as may a seam of ore, or a pit prop), but never into anything furnished."""
        skip = ("step", "floor", "crack", "stain", "grate", "bridge", "rug",
                "window", "painting", "banner", "portico", "chimney")     # (decals, and what stands round its columns)
        wild = ("bush", "oak", "pine", "boulder", "stalagmite", "stalactite", "mushroom")
        in_the_rock = wild + ("ore", "timber")
        boxes = [(p.name, p) for p in world.props if p.kind not in skip]
        boxes += [(c.name, c) for c in world.containers] + [(s.name, s) for s in world.stations]
        kind = {id(b): getattr(b, "kind", "") for _, b in boxes}
        kind.update({id(s): "ore" for s in world.stations if "ore" in s.name or "vein" in s.name})

        def hit(a, b):
            return (a.min_point.x < b.max_point.x and a.max_point.x > b.min_point.x and a.min_point.z < b.max_point.z
                    and a.max_point.z > b.min_point.z and a.min_point.y < b.max_point.y and a.max_point.y > b.min_point.y)
        index = BoxIndex([b for _, b in boxes], cell=6.0)
        names = {id(b): n for n, b in boxes}
        for name_a, a in boxes:
            for b in index.near((a.min_point.x + a.max_point.x) / 2, (a.min_point.z + a.max_point.z) / 2, 4.0):
                if b is not a and not (kind[id(a)] in wild and kind[id(b)] in wild):
                    assert not hit(a, b), (name_a, names[id(b)])
            if kind[id(a)] in in_the_rock:
                continue
            for wall in world.walls_near(a.center, 6.0):
                assert not hit(a, wall), (name_a, wall)

    def test_doors(self, world):
        by_id = {d.door_id: d for d in world.doors}
        hideout, shop, home = by_id["hideout_door"], by_id["shop_door"], by_id["home_door"]
        assert (hideout.is_locked, hideout.key_id, hideout.owners) == (False, "hideout_door_key", ("Jae the Fence",))
        assert (shop.is_locked, shop.key_id, shop.owners) == (True, "shop_key", ("Shopkeeper Barnaby",))
        assert (home.key_id, home.owners) == ("home_key", ("Shopkeeper Barnaby",))
        assert (by_id["guardhouse_door"].key_id, by_id["guardhouse_door"].owners, by_id["guardhouse_door"].lock_difficulty) == \
            ("guard_key", ("guards",), town.STRONG)
        assert by_id["cell_door"].is_bars and by_id["cell_door"].is_locked and not by_id["jail_door"].is_locked
        assert (shop.hinge_position.x, shop.hinge_position.z) == (-15.0, -6.0)
        assert (home.hinge_position.x, home.hinge_position.z) == (-71.5, 7.0)
        assert shop.schedule is None                      # the shopkeeper runs the door now
        assert world.find_door("shop_door") is shop and world.find_door("nope") is None
        # every door with an owner has an owner (or one of the owning faction) who carries its key
        for door in world.doors:
            if not door.owners:
                continue
            owner = door.owners[0]
            keyholders = [n for n in world.npcs if (n.name == owner or n.faction == owner) and n.has_key_for(door)]
            assert keyholders, door.door_id
        rooms = [d for d in world.doors if d.door_id.startswith("inn_room")]
        assert len(rooms) == 3 and all(d.width == 1.6 and d.is_locked for d in rooms)
        leaves = world.gate_leaves("town_gate")
        assert [d.door_id for d in leaves] == ["town_gate_west", "town_gate_east"] and leaves[1].hinge_right
        assert leaves[0].x2 == leaves[1].x1 == 0.0                             # the two leaves meet in the middle

    def test_lock_tiers_follow_what_they_guard(self, world):
        tier = lambda door_id: world.find_door(door_id).lock_difficulty   # noqa: E731
        assert tier("odo_door") == town.FLIMSY                                  # a farmhouse latch
        assert tier("tamsin_door") == tier("fenwick_door") == tier("pell_door") == town.SIMPLE
        assert tier("tailor_door") == tier("smithy_door") == tier("inn_door") == town.STURDY      # (shops)
        assert tier("home_door") == tier("couple_door") == town.SIMPLE                             # (homes)
        assert tier("shop_door") == town.SIMPLE                                 # the urchin's first lock (the tutorial)
        assert tier("guardhouse_door") == tier("cell_door") == town.STRONG
        assert tier("vane_door") == tier("ashgrove_door") == tier("treasury_door") == town.MASTER
        assert tier("deep_vault_door") == town.LEGENDARY
        strongboxes = [c for c in world.containers if c.kind == "strongbox"]
        assert strongboxes and all(c.is_locked and c.lock_difficulty >= town.MASTER for c in strongboxes)

    def test_npcs(self, world):
        names = [n.name for n in world.npcs]
        assert names[:10] == ["Shopkeeper Barnaby", "Hilde the Innkeeper", "Gareth the Smith", "Mirelle the Tailor",
                              "Jae the Fence", "Tamsin", "Fenwick", "Lys", "Dorran", "Pell"]
        assert names[10:12] == ["Lord Aurelian Vane", "Lady Seraphine Ashgrove"]
        assert [n.name for n in world.guards()] == ["Guard Captain Vance", "Guard Aldric", "Guard Orrin", "Guard Mira",
                                                    "Guard Bram", "Guard Wenna", "Jailer Hesk"]
        assert {"Odo the Farmer", "Ysolde the Hunter", "Grimmaw the Bandit King", "Ser Caddock", "Dame Ilsa"} <= set(names)
        barnaby, jae = world.find_npc("Shopkeeper Barnaby"), world.find_npc("Jae the Fence")
        assert barnaby.can_trade and not barnaby.is_fence and not barnaby.is_guard
        assert [i.id for i in barnaby.inventory] == ["shop_key", "home_key", "bread", "apple_1", "torch_item"]
        assert [(i.id, i.quantity) for i in barnaby.stock] == [("bread", 4), ("apple_1", 6), ("torch_item", 1)]
        assert barnaby.is_shopkeeper and barnaby.shop_containers == ["shop_shelf_west", "shop_shelf_east"]
        assert [t.min_player_level for t in barnaby.restock_tiers] == [2, 4]
        assert jae.is_fence and jae.can_trade and not jae.is_shopkeeper
        assert [i.id for i in jae.inventory] == ["hideout_door_key", "lockpick_pack", "gold_ring", "black_hood", "dried_meat"]
        assert not any(i.is_stolen for i in jae.inventory)             # a fence's goods are never 'hot'
        for name in ("Guard Captain Vance", "Guard Orrin", "Guard Mira"):
            g = world.find_npc(name)
            assert g.is_guard and g.faction == "guards" and not g.can_trade
            assert [i.id for i in g.inventory] == ["guard_key", "tower_key", "ration"]
            assert g.stock == [i for i in g.inventory if i.id == "ration"]
        hesk = world.find_npc("Jailer Hesk")
        assert hesk.is_guard and any(i.id == "cell_key" for i in hesk.inventory)
        assert all(i.value == 1.0 for n in world.npcs for i in n.inventory if i.is_key)
        # specialists and their trades
        smith, tailor, hilde = (world.find_npc(n) for n in ("Gareth the Smith", "Mirelle the Tailor", "Hilde the Innkeeper"))
        assert smith.trade_categories == ("weapon", "armor", "ore", "tool") and smith.role_tag == "smith" and smith.can_trade
        assert tailor.trade_categories == ("clothing", "tool") and hilde.trade_categories == ()
        assert {i.id for i in hilde.stock} >= {"rent_night", "rent_week"} and not any(i.is_key and i.for_sale for i in hilde.inventory)
        # everyone who is not a guard has a bed they sleep in; the townsfolk a key to their own door
        for npc in world.npcs:
            if npc.is_guard:
                continue
            sleep = [t for t in npc.schedule if t.target_state is AIState.SLEEPING]
            assert sleep, npc.name
            bed = world.prop_under(sleep[0].target_pos)
            assert bed is not None and bed.is_bed, npc.name
            if not npc.warden_of and world.building_at(bed.center) is not None:    # (Hobb sleeps in the open stable)
                assert any(i.is_key for i in npc.inventory), npc.name
        for npc in world.npcs:                                          # every day is covered, hour by hour
            hours = sorted((t.start_hour, t.end_hour) for t in npc.schedule)
            assert hours[0][0] == 0 and hours[-1][1] == 24, npc.name
            assert all(a[1] == b[0] for a, b in zip(hours, hours[1:])), npc.name

    def test_shop_shelves_hold_loot(self, world):
        west, east = world.find_container("shop_shelf_west"), world.find_container("shop_shelf_east")
        assert west.owner == east.owner == "Shopkeeper Barnaby" and west.crime_name == "Shoplifting"
        assert [(i.id, i.quantity) for i in west.inventory] == [("candlestick", 1), ("wine", 3), ("thread", 5)]
        assert [(i.id, i.quantity) for i in east.inventory] == [("lantern", 1), ("honey", 2), ("plate", 1), ("pickaxe", 1), ("mallet", 1)]
        # shelves stand against the shop's north wall, behind the counter (a restricted area)
        shop = next(b for b in world.level.buildings if b.name == "shop")
        for shelf in (west, east):
            assert shop.min_xz[0] <= shelf.min_point.x and shelf.max_point.x <= shop.max_xz[0]
            assert shelf.min_point.z <= shop.min_xz[1] + 0.6
            assert world.area_of(shelf.center) == "behind the counter"
        rack = world.find_container("smithy_rack")
        assert {i.category for i in rack.inventory} == {"weapon", "armor", "ore", "tool"}
        assert all(i.category == "clothing" or i.id == "needles" for i in world.find_container("tailor_rack").inventory)

    def test_the_tailor_sells_no_dark_cloth(self, world):
        """Dark, quiet clothing is rare loot now - stolen from the rich, the smugglers and the keep -
        never stock on a rack."""
        tailor = world.find_npc("Mirelle the Tailor")
        for tier in tailor.restock_tiers:
            for item in tier.items:
                assert item.build().rarity in ("common", "uncommon"), item.id
        for c in world.containers:
            if c.name in ("tailor_rack", "smithy_rack", "shop_shelf_west", "shop_shelf_east"):
                assert all(i.rarity in ("common", "uncommon") for i in c.inventory), c.name

    def test_npcs_start_on_their_07_00_task(self, world):
        barnaby, jae = world.find_npc("Shopkeeper Barnaby"), world.find_npc("Jae the Fence")
        vance, orrin, mira, hesk = (world.find_npc(n) for n in ("Guard Captain Vance", "Guard Orrin", "Guard Mira", "Jailer Hesk"))
        assert barnaby.state is AIState.EATING and barnaby.position == world.spot(*BARNABY_TABLE_SEAT)
        assert jae.state is AIState.SHOPKEEPING and jae.position == world.spot(*JAE_DEN)
        assert vance.state is AIState.PATROL and vance.position == world.spot(*PATROL_ROUTE[0])
        assert orrin.state is AIState.GUARDING and orrin.position == world.spot(*POST_DOOR)
        assert mira.state is AIState.SLEEPING and mira.position == world.spot(*GUARD_BEDS[2])
        assert hesk.state is AIState.GUARDING and world.area_of(hesk.position) == "the jail"
        assert world.find_npc("Hilde the Innkeeper").state is AIState.SHOPKEEPING         # up since five
        bram = world.find_npc("Guard Bram")
        assert bram.state is AIState.GUARDING and world.area_of(bram.position) == "the town gate"
        assert world.log.messages == []                   # silent at start

    def test_jae_is_not_inside_a_wall(self, world):
        jae = world.find_npc("Jae the Fence")
        assert not any(w.min_point.x <= jae.position.x <= w.max_point.x and w.min_point.z <= jae.position.z <= w.max_point.z
                       for w in world.walls)
        assert not world.nav.is_blocked(jae.position.x, jae.position.z)

    def test_npc_spots_are_free_or_in_a_bed(self, world):
        for npc in world.npcs:
            for task in npc.schedule:
                pos = task.target_pos
                if task.target_state is AIState.SLEEPING:
                    bed = world.prop_under(pos)
                    assert bed is not None and bed.is_bed, (npc.name, pos)
                elif pos.y <= world.ground_height(pos.x, pos.z) + 0.05:            # (upstairs spots are off the ground grid)
                    assert not world.nav.is_blocked(pos.x, pos.z), (npc.name, task.action_text)
                for point in task.patrol_points:
                    if point.y <= world.ground_height(point.x, point.z) + 0.05:       # (upstairs: see test_manors)
                        assert not world.nav.is_blocked(point.x, point.z), (npc.name, point)

    def test_everybody_can_walk_their_day(self, world):
        """Every task's spot can be reached from the one before it (doors opened by whoever has the
        key - and a door they cannot open is one the day never asks them through): with every door
        open, only the streets and the stairs are left to find a way along."""
        for door in world.doors:
            door.is_locked, door.is_open = False, True
        for npc in world.npcs:
            tasks = sorted(npc.schedule, key=lambda t: t.start_hour)
            for prev, task in zip(tasks, tasks[1:] + tasks[:1]):
                if prev.target_pos.distance_to(task.target_pos) < 0.5:
                    continue
                npc.position = prev.target_pos.copy()
                path = world.find_path(npc, task.target_pos)
                assert path, (npc.name, prev.action_text, task.action_text)
                assert path[-1].distance_to(task.target_pos) < 0.01

    def test_areas_and_exits(self, world):
        at = lambda x, z: world.area_of(world.spot(x, 0.0, z))   # noqa: E731
        assert at(-13, -8) == "the shop" and at(-13, -13) == "behind the counter"
        assert at(50, -48) == "the guard house"
        assert at(55, 13) == "the jail cell" and at(50, 12) == "the jail"
        assert at(-45, 28) == "the streets"
        assert at(1, 1) == "the well" and at(30, 1) == "main street"
        assert at(-33, -20) == "room one" and at(-28, -9) == "the inn"
        assert at(0, 90) == "Thornbury" and at(-150, 50) == "the Thornwood" and at(100, 0) == "the countryside"
        assert at(16, -50) == "Vane Manor" and at(-151, -91) == "the Howling Deep" and at(-151, -112) == "the Deep King's hall"
        assert at(-151, -125) == "the Deep vault" and at(106, 24) == "the smugglers' cave" and at(100, -88) == "the old mine"
        # (in the cave's mouth, under the mountain, it is the Deep; on the hillside above it, the wood)
        assert at(-151, -70) == "the Howling Deep" and at(-151, -60) == "the Thornwood"
        shop_exit = world.exit_of("the shop")
        assert (shop_exit.x, shop_exit.z) == (-13.5, -4.0)                       # outside the street-side door
        assert world.exit_of("behind the counter") == shop_exit
        home_exit = world.exit_of("Barnaby's house")
        assert (home_exit.x, home_exit.z) == (-70.0, 5.0)
        room_exit = world.exit_of("room one")
        assert (room_exit.x, room_exit.z) == (-32.5, -4.0)                       # out of the inn altogether
        assert room_exit.y == pytest.approx(world.ground_height(-32.5, -4.0))
        assert world.exit_of("the streets") is None

    def test_ore_deposit_by_the_smithy_and_anvil_station(self, world):
        node = world.find_station("ore_deposit")
        assert node.definition.requires_item_id == "pickaxe" and node.definition.yields.id == "iron_ore" and node.is_mine
        anvil = world.find_station("anvil")
        assert anvil.definition.consumes == ("iron_ore", 1) and anvil.definition.produces.quantity == 3
        assert anvil.center.distance_to(node.center) < 14.0                 # the deposit is a short walk from the shed
        shed = next(b for b in world.level.buildings if b.name == "smithy_shed")
        assert shed.min_xz[0] <= anvil.center.x <= shed.max_xz[0] and shed.min_xz[1] <= anvil.center.z <= shed.max_xz[1]
        kiln = next(p for p in world.props if p.name == "forge")
        assert shed.min_xz[0] <= kiln.min_point.x and kiln.max_point.x <= shed.max_xz[0]
        assert any(f.kind == "fire" for f in world.fixtures)

    def test_trust(self, world):
        player = world.spawn_player()
        jae, barnaby, vance = world.find_npc("Jae the Fence"), world.find_npc("Shopkeeper Barnaby"), world.find_npc("Guard Captain Vance")
        put = lambda x, z: setattr(player, "position", world.spot(x, 0.0, z))   # noqa: E731
        put(-12.0, 12.0)                                                # the hideout: the fence's house welcomes the player
        assert not world.is_trespassing(player, jae) and not world.is_trespassing(player, barnaby)
        assert not world.is_trespassing(player, vance)                  # ... and a guard has no say in it by day
        world.simulated_hour = 23
        assert world.is_trespassing(player, vance) and not world.is_trespassing(player, jae)   # after curfew he does
        world.simulated_hour = 7
        put(-13.0, -8.0)                                                # shop floor before opening
        assert world.is_trespassing(player, barnaby) and world.is_trespassing(player, vance)
        world.simulated_hour = 10
        assert not world.is_trespassing(player, barnaby)                # open for business
        put(-13.0, -13.0)                                               # behind the counter
        assert world.is_trespassing(player, barnaby) and not world.is_trespassing(player, jae)
        put(50.0, -48.0)                                                # the guard house
        assert world.is_trespassing(player, vance) and not world.is_trespassing(player, barnaby)
        put(-28.0, -9.0)                                                # the inn's public room
        assert not world.is_trespassing(player, world.find_npc("Hilde the Innkeeper"))
        put(-33.0, -20.0)                                               # a guest room, unrented
        assert world.is_trespassing(player, world.find_npc("Hilde the Innkeeper"))
        put(0.0, 3.0)
        assert not world.is_trespassing(player, vance)
        put(16.0, -50.0)                                                # a manor: its lady or lord - and the watch - object
        assert world.is_trespassing(player, world.find_npc("Lord Aurelian Vane")) and world.is_trespassing(player, vance)

    def test_public_hours_may_run_past_midnight(self):
        """Like curfews: a public house open (18, 2) is open in the evening and the small hours."""
        from unprepared.world import AreaDef
        tavern = AreaDef("the tavern", (0.0, 0.0), (4.0, 4.0), owner="Hilde", restricted=True, public_hours=(18, 2))
        assert [h for h in range(24) if tavern.is_open_to_public(h)] == [0, 1, 18, 19, 20, 21, 22, 23]
        shop = AreaDef("a shop", (0.0, 0.0), (4.0, 4.0), owner="Barnaby", restricted=True, public_hours=(8, 20))
        assert [h for h in range(24) if shop.is_open_to_public(h)] == list(range(8, 20))
        never = AreaDef("shut", (0.0, 0.0), (4.0, 4.0), owner="Barnaby", restricted=True, public_hours=(9, 9))
        assert not any(never.is_open_to_public(h) for h in range(24))

    def test_an_area_nobody_owns_is_nobody_elses_to_guard(self, settings):
        """A restricted (or curfewed) area without an owner: the watch still objects to the player
        being there, but no townsperson takes it upon themselves - a missing owner is not "every
        NPC without a faction"."""
        import dataclasses
        from unprepared.world import AreaDef
        level = get_level("town")
        chapel = AreaDef("the old chapel", (22.0, -5.0), (28.0, 5.0), restricted=True, curfew_hours=(20, 6))
        w = World(dataclasses.replace(level, areas=(chapel, *level.areas)), settings)
        player = w.spawn_player()
        player.position = Vector3D(25.0, 0.0, 0.0)
        tamsin, vance = w.find_npc("Tamsin"), w.find_npc("Guard Captain Vance")
        assert not w.is_trespassing(player, tamsin) and w.is_trespassing(player, vance)
        w.simulated_hour = 22
        tamsin.path, tamsin.current_task, tamsin.schedule = [], None, []
        tamsin.state, tamsin.position = AIState.IDLE, Vector3D(26.0, 0.0, 1.0)
        w._tick_curfews(0.1)
        assert tamsin.state is AIState.IDLE and tamsin.waiting_for is None        # not her place to show anyone out

    def test_fractional_hour_advances_smoothly(self, world):
        assert world.clock.fractional_hour == 7.0
        world.update_time(12.0)                                               # 30 simulated minutes
        assert world.clock.fractional_hour == pytest.approx(7.5)

    def test_days_are_counted(self, world):
        world.simulated_hour = 23
        for _ in range(24):
            world.update_time(1.0)
        assert world.simulated_hour == 0 and world.day_count == 1

    def test_sleeping_npcs_lie_flat_on_their_beds(self, world):
        from unprepared.world.world import NPC_HEIGHT, NPC_LYING_THICKNESS, SLEEPER_ABOVE_TOP
        mira = world.find_npc("Guard Mira")
        bed = world.prop_under(mira.position)
        box = world.npc_body_box(mira)
        assert box.min_point.y == pytest.approx(bed.max_point.y - SLEEPER_ABOVE_TOP)          # on the blanket
        assert box.max_point.y - box.min_point.y == pytest.approx(NPC_LYING_THICKNESS)
        assert box.max_point.x - box.min_point.x == pytest.approx(NPC_HEIGHT)                 # along the bunk (heads to the wall)
        assert box.max_point.z - box.min_point.z == pytest.approx(0.8)
        assert world.sleep_head(mira) == "-x"                                                 # a west-wall bunk: head west
        vance = world.find_npc("Guard Captain Vance")
        up = world.npc_body_box(vance)
        assert up.max_point.y - up.min_point.y == pytest.approx(1.8) and up.min_point.y == pytest.approx(vance.position.y)
        mira.state = AIState.GUARDING
        assert world.npc_body_box(mira).max_point.y - mira.position.y == pytest.approx(1.8)

    def test_beds_are_all_one_size_whichever_way_they_face(self, world):
        from unprepared.world.builders import BED_LENGTH, BED_WIDTH, DOUBLE_BED_WIDTH
        from unprepared.rendering.structures import bed_local_extent
        for bed in world.beds:
            length, width = bed_local_extent(bed)
            assert length == pytest.approx(BED_LENGTH), bed.name
            if bed.kind == "bedroll":                                   # (an outlaw's roll of blankets: narrower)
                assert width < BED_WIDTH, bed.name
                continue
            assert width in (pytest.approx(BED_WIDTH), pytest.approx(DOUBLE_BED_WIDTH)), bed.name
        bunks = [b for b in world.beds if b.name.startswith("guard_bed")]
        assert len(bunks) == 6 and all(b.max_point.x - b.min_point.x == pytest.approx(BED_LENGTH) for b in bunks)   # one way

    def test_player_spawn(self, world):
        player = world.spawn_player()
        assert player.position == world.spot(*town.PLAYER_SPAWN)          # on the well square
        assert world.area_of(player.position) == "the well"
        assert player.gold == STARTING_GOLD
        assert [i.id for i in player.inventory] == ["iron_lockpick", "soft_shoes", "silver_goblet"]
        assert player.inventory[0].stackable and player.inventory[2].is_stolen and player.inventory[2].name == "Silver Goblet"
        assert world.player is player and world.player_level == 1



class TestClock:
    def test_advance(self):
        clock = WorldClock(start_hour=7, time_scale=2.5)
        assert clock.advance(23.9) is False
        assert clock.hour == 7
        assert clock.advance(0.1) is True
        assert clock.hour == 8 and clock.minute_accumulator == 0.0

    def test_wraps_midnight(self):
        clock = WorldClock(start_hour=23, time_scale=60.0)
        assert clock.advance(1.0) is True
        assert clock.hour == 0


class TestWorldTime:
    def test_hour_tick_triggers_schedules_and_messages(self, world):
        for _ in range(24):
            world.update_time(1.0)
        assert world.simulated_hour == 8
        barnaby = world.find_npc("Shopkeeper Barnaby")
        assert barnaby.state is AIState.TRAVELING and barnaby.path
        assert world.log.messages[0] == "Time Update: 08:00 hours"
        assert "(Shopkeeper Barnaby): \"Time to open up the shop.\"" in world.log.messages
        assert barnaby.current_speech(world.elapsed) is not None                 # said aloud, in a bubble

    def test_door_schedules_still_work_for_unowned_doors(self, world):
        from unprepared.entities import DoorSchedule
        door = world.find_door("hideout_door")
        door.schedule = DoorSchedule(8, 22, "hideout opens", "hideout closes")
        world.simulated_hour = 7
        for _ in range(24):
            world.update_time(1.0)
        assert door.is_open and world.log.messages[0] == "hideout opens"

    def test_ambient_light_property(self, world):
        assert world.ambient_light == 1.0
        world.simulated_hour = 23
        assert world.ambient_light == 0.1

    def test_report_crime_returns_witnesses(self, world):
        vance = world.find_npc("Guard Captain Vance")
        spot = Vector3D(vance.position.x, 0.0, vance.position.z + 5.0)
        vance.turn_towards(spot)
        witnesses = world.report_crime_to_witnesses("Pickpocketing", spot)
        assert vance in witnesses and vance.state is AIState.ALERT_GUARDS

    def test_update_perception_uses_ambient(self, world):
        player = world.spawn_player()
        player.is_crouching = True
        vance = world.find_npc("Guard Captain Vance")
        player.position = Vector3D(vance.position.x, 0.0, vance.position.z + 3.0)
        vance.turn_towards(player.position)
        world.update_perception(player, True, False, delta_time=0.5)
        assert vance.suspicion_meter > 0

    def test_door_leaf_pushes_the_player_clear(self, world):
        player = world.spawn_player()
        door = world.find_door("hideout_door")                        # hinge (-13.5, 7.5), leaf along x when closed
        h = door.hinge_position
        player.position = Vector3D(h.x + 1.5, h.y, 7.5)               # standing in the doorway
        door.is_open = False
        assert world.push_out_of_doors(player)
        assert player.position.z > 7.6 + 0.3 - 1e-9                   # shoved out to the +z side
        door.is_open = True                                            # leaf folds along z at the hinge's x
        player.position = Vector3D(h.x, h.y, 5.5)
        assert world.push_out_of_doors(player)
        assert player.position.x >= h.x + 0.1 + 0.3 - 1e-9
        player.position = Vector3D(h.x + 2.5, h.y, 3.0)
        assert not world.push_out_of_doors(player)

    def test_update_npcs_moves_travellers(self, world):
        vance = world.find_npc("Guard Captain Vance")
        world.clock.hour = 23                            # (an NPC on an off-hour task would be put back on the hour's)
        vance.begin_task(vance.task_for(23), world)      # walk to bed
        assert vance.state is AIState.TRAVELING
        start = vance.position.copy()
        world.update_npcs(0.5)
        assert math.hypot(vance.position.x - start.x, vance.position.z - start.z) == pytest.approx(1.8)   # 3.6 units/s


class TestStreetDoors:
    def test_door_of_a_building_is_the_one_onto_the_street(self, world):
        shop = next(b for b in world.level.buildings if b.name == "shop")
        assert world.door_of(shop).door_id == "shop_door"
        inn = world.inn_building()
        assert world.door_of(inn).door_id == "inn_door"
        jail = world.building_at(world.jail.cell_spot)
        assert world.door_of(jail).door_id == "jail_door"                    # not the barred cell door


class TestFrozenGuards:
    def test_freezing_stops_every_guard_and_unfreezing_puts_them_back_on_duty(self, world):
        guards = world.guards()
        world.freeze_guards(except_for=["Guard Mira"])
        assert all(world.is_frozen(g) for g in guards if g.name != "Guard Mira")
        assert not world.is_frozen(world.find_npc("Guard Mira"))
        assert not any(g.path for g in guards if world.is_frozen(g))
        world.unfreeze_guards()
        assert not world.guards_frozen and not any(world.is_frozen(g) for g in guards)
        assert all(g.current_task is not None for g in guards)
