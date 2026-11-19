"""Beyond the town gate: the lie of the land and everybody who lives out in it.

    north   the Greyspine - a wall of mountain behind the town; nobody climbs it. Two holes go
            into its foot: the Howling Deep in the north-west (a bandit king's hall and his
            vault, far under the rock) and the old Emberhollow mine in the north-east (worked
            out, fallen in, and not quite empty)
    west    the Thornwood: the hermit smith's forge, Ysolde the hunter's hut, the lake of
            Thornmere - and beyond them the wood grows too thick for anybody to get through
    south   Thornbury, the village outside the gate, its lane and fields; the road on south to
            the ruined shrine of Saint Wendel, where the Deepwood closes in
    east    the coast road and the sea: Greyhold keep on its rise by the shore, and under the
            bluff over the Saltcove, the smugglers' cave

Everything here is plain level data merged into the town's ``LevelDefinition`` (see
``town.py``). The outlaws and the knights are *wardens* (``NPCDef.warden``): they answer to no
town watch - somebody they catch in their domain is dragged out and dumped beyond it, the
goods taken back and a toll of coin with them.

The ground is laid out first (``BASE``: hills, noise and the sea) so that the caves can be dug
at the right depth under it (``LAND.natural_point``) and the trees kept off the steeps.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from ...entities import AIState, Outfit
from ..builders import (WALL, banner, barrel, barrels, bed, bedroll, boulder, bridge, bush, chest, coins, crates, dummy, fence,
                        haystack, house_walls, journal, key_item, minecart, notice_board, oak, partition_z, pedestal, pine,
                        plaque, pool, resident_tasks, rock_steps, rowboat, sacks, sconce, signpost, skeleton, span, stash,
                        table, throne, timber, wall_torch, woodpile)
from ..caves import Chamber, Mouth, clear_the_floor, dig, mushrooms
from ..level import (AreaDef, BuildingDef, ContainerDef, DoorDef, ItemDef, LightDef, NPCDef, PropDef, ReadableDef, RoadDef,
                     StationDef, TaskDef, WallDef)
from ..terrain import FlatDef, HillDef, LakeDef, SeaDef, Terrain, TerrainDef, TrailDef

BOUNDS = (-200.0, -135.0, 175.0, 212.0)          # the map: min_x, min_z, max_x, max_z (the caves run far under the mountain)
FLIMSY, SIMPLE, STURDY, STRONG, MASTER, LEGENDARY = range(6)
TERRAIN_SKIRT = 36.0


# =============================================================================== the land
def _mountains() -> List[HillDef]:
    """The Greyspine, in three ranks along the north: a broken rampart at the mountain's foot,
    steep enough that nobody climbs past it; behind it a row of craggy peaks, each a broad
    shoulder with a sharp summit on it; and the high peaks beyond, snow-capped. Gaussian hills
    all (their flanks fall away quickly: the town below is not lifted by them), roughened by
    ridged noise (``HillDef.crags``) into ribs, gullies and broken summits."""
    rng = random.Random(404)
    out = []
    x = -250.0
    while x <= 220.0:
        out.append(HillDef(x + rng.uniform(-3, 3), -101.0 + rng.uniform(-3, 3), rng.uniform(13, 17), rng.uniform(10, 12), crags=0.18))
        x += 14.0
    x = -240.0
    while x <= 220.0:
        px, pz, ph = x + rng.uniform(-10, 10), -118.0 + rng.uniform(-5, 5), rng.uniform(30, 45)
        out.append(HillDef(px, pz, ph * 0.55, rng.uniform(15, 18), crags=0.2))
        out.append(HillDef(px + rng.uniform(-3, 3), pz + rng.uniform(-3, 3), ph * 0.45, rng.uniform(6, 8), crags=0.2))
        x += rng.uniform(40, 55)
    x = -260.0
    while x <= 240.0:
        px, pz, ph = x + rng.uniform(-12, 12), -150.0 + rng.uniform(-5, 5), rng.uniform(55, 75)
        out.append(HillDef(px, pz, ph * 0.55, rng.uniform(22, 26), crags=0.15))
        out.append(HillDef(px + rng.uniform(-4, 4), pz + rng.uniform(-4, 4), ph * 0.45, rng.uniform(9, 12), crags=0.15))
        x += rng.uniform(45, 60)
    return out


HILLS = (
    HillDef(0.0, -175.0, 10.0, 95.0),            # the long rise the town sits on
    HillDef(52.0, -56.0, 5.5, 34.0),             # the knoll the watchtower crowns
    *_mountains(),
    HillDef(-150.0, -84.0, 13.0, 10.0),          # the crag over the Howling Deep
    HillDef(100.0, -84.0, 11.0, 10.0),           # the spur the old mine is driven into
    HillDef(112.0, 24.0, 17.0, 13.0),            # the bluff over the Saltcove
    HillDef(94.0, 116.0, 4.5, 24.0),             # Greyhold's rise
    HillDef(-120.0, 30.0, 4.0, 30.0),            # the wooded slopes west of the town
)
SEA = SeaDef(coast=((150.0, -140.0), (149.0, -60.0), (147.0, 0.0), (144.0, 40.0), (141.0, 80.0), (138.0, 120.0),
                    (128.0, 160.0), (110.0, 200.0), (84.0, 240.0), (60.0, 290.0)), level=-0.8, beach=14.0)
LAKES = (LakeDef("thornmere", (-92.0, 158.0), 30.0, 18.0, level=-0.6, depth=2.6),)
MILLSTREAM_OUTSIDE = ((-56.0, 62.0), (-60.0, 72.0), (-66.0, 86.0), (-74.0, 102.0), (-82.0, 118.0), (-88.0, 134.0),
                      (-92.0, 146.0))
BASE = TerrainDef(base=0.0, hills=HILLS, noise_amplitude=0.35, noise_scale=16.0, seed=11, pad_margin=4.0, sea=SEA,
                  skirt=TERRAIN_SKIRT, snow_line=50.0)
LAND = Terrain.landform(BASE, BOUNDS)


def ground(x: float, z: float) -> float:
    """The natural ground at (x, z): where a cave's mouth is, how deep to dig."""
    return LAND.natural_point(x, z)


# =============================================================================== named spots
ODO_BED, ODO_SEAT = (19.4, 0.0, 81.45), (14.5, 0.0, 80.9)
MARTA_BED, MARTA_SEAT = (-16.1, 0.0, 80.45), (-20.5, 0.0, 80.8)
HASKEL_BED, HASKEL_SEAT = (-37.1, 0.0, 79.45), (-40.5, 0.0, 79.6)
FIELD_EAST = ((14.0, 0.0, 104.0), (30.0, 0.0, 108.0), (22.0, 0.0, 114.0), (12.0, 0.0, 110.0))
FIELD_WEST = ((-34.0, 0.0, 94.0), (-16.0, 0.0, 98.0), (-24.0, 0.0, 104.0))
CHOPPING_SPOT = (-46.0, 0.0, 84.0)
FELLING_SPOT = (-60.0, 0.0, 95.0)                          # where the Thornwood begins, west of the village
MARKET_PITCH = (6.0, 0.0, 2.5)                             # Odo's pitch by the well on market day
CHAPEL_VISIT = (-24.0, 0.0, 40.2)                          # in the aisle of the chapel in town
HUNTER_BED, HUNTER_SEAT = (-116.1, 0.0, 100.45), (-120.4, 0.0, 100.7)
HUNTING_ROUTE = ((-110.0, 0.0, 110.0), (-140.0, 0.0, 96.0), (-150.0, 0.0, 70.0), (-128.0, 0.0, 62.0), (-108.0, 0.0, 84.0))
GREYHOLD_EXIT = (62.0, 0.0, 112.0)
KNIGHT_BEDS = ((87.55, 0.0, 106.6), (87.55, 0.0, 109.2), (87.55, 0.0, 111.8))
KEEP_TABLE_SEATS = ((95.0, 0.0, 109.6), (97.0, 0.0, 113.0), (99.0, 0.0, 109.6))
GATEWAY_POST = (75.5, 0.0, 115.0)
COURTYARD_ROUTE = ((78.0, 0.0, 100.0), (110.0, 0.0, 100.0), (111.0, 0.0, 131.0), (78.0, 0.0, 131.0))

# -- the Howling Deep: its mouth in the crag at the foot of the mountain, the halls under it
DEEP_MOUTH_Z = -66.0
DEEP_FLOOR = round(ground(-150.0, -64.5) - 2.5, 2)         # the way in is cut down into the rock ...
DEEP_HALL_FLOOR = round(DEEP_FLOOR - 1.3, 2)                # ... and steps lead lower still
DEEP_EXIT = (-150.0, 0.0, -46.0)
DEEP_FIRE = (-151.0, -91.0)
DEEP_TABLE = (-142.0, -86.5)
DEEP_SEATS = ((-143.6, -85.0), (-140.4, -85.0), (-143.6, -88.0), (-140.4, -88.0))
DEEP_FIRE_SEATS = ((-153.2, -90.0), (-148.8, -92.0), (-151.0, -88.6), (-151.0, -93.4))
DEEP_MOUTH_POST = (-150.0, -76.5)
DEEP_DUMMY_SPOT = (-160.0, -92.0)
DEEP_POOL = (-127.0, -91.0)
DEEP_RACK_SPOT = (-145.0, -99.3)
DEEP_BUNKS = ((-184.5, -96.0), (-184.5, -92.8), (-184.5, -89.6), (-184.5, -86.6), (-176.0, -95.9))
DEEP_ROUNDS = ((-150.0, -82.5), (-160.0, -95.0), (-172.0, -90.0), (-150.0, -98.0), (-130.0, -90.0), (-140.0, -84.0))
BOSS_POST = (-151.0, -115.3)                                # the throne, before the vault door
BOSS_BED = (-158.4, -116.5)
VAULT_PEDESTAL = (-151.0, -127.0)
VAULT_DOOR_X = -152.5

# -- the Saltcove: the smugglers' cave under the bluff, its mouth on the beach
COVE_MOUTH_X = 122.0
COVE_FLOOR = round(max(ground(131.0, 20.0) + 0.9, 0.6), 2)
SMUGGLERS_EXIT = (134.0, 0.0, 48.0)
COVE_FIRE = (106.0, 21.0)
COVE_TABLE = (104.0, 26.5)
COVE_SEATS = ((102.3, 26.5), (105.7, 26.5), (104.0, 25.1))
COVE_FIRE_SEATS = ((104.2, 21.0), (107.8, 21.0), (106.0, 19.2))
COVE_BEDS = ((101.0, 14.5), (101.0, 17.4), (101.0, 20.3), (108.6, 12.1), (111.4, 12.1))
COVE_STORE_SPOT = (108.0, 34.0)
COVE_MOUTH_POST = (116.5, 20.0)
COVE_BOAT = (137.0, 28.0)
COVE_LOOKOUT = (133.8, 36.0)
COVE_NETS = (132.0, 32.0)                # where Maggs guts the catch (the nets hang behind her)
COVE_HAUL = ((136.0, 0.0, 25.0), (126.0, 0.0, 20.0), (115.0, 0.0, 20.0), (108.0, 0.0, 34.0))

# -- the old mine: driven into the spur north-east of the town
MINE_MOUTH_Z = -66.0
MINE_FLOOR = round(ground(100.5, -64.5) - 2.0, 2)
MINE_SUMP_FLOOR = round(MINE_FLOOR - 1.3, 2)

# -- the hermit smith's forge in the Thornwood
HROLF_HUT = (-160.0, 24.0, -152.0, 32.0)
HROLF_BED, HROLF_SEAT = (-158.45, 0.0, 30.3), (-154.2, 0.0, 29.2)
HROLF_ANVIL, HROLF_FORGE = (-147.6, 0.0, 27.6), (-149.2, 0.0, 26.5)
HROLF_ORE = (-146.0, 0.0, 10.5)

SHRINE = (0.0, 184.0)


# =============================================================================== the ways
TRAILS = (
    TrailDef("thornwood_way", ((-103.0, 71.75), (-112.0, 66.0), (-122.0, 56.0), (-130.0, 44.0), (-136.0, 32.0), (-140.0, 22.0),
                               (-143.0, 10.0), (-146.0, -6.0), (-149.0, -24.0), (-150.0, -40.0), (-150.0, -54.0),
                               (-150.0, DEEP_MOUTH_Z)), width=2.8, end_height=DEEP_FLOOR, max_grade=0.16, shoulder=2.6),
    TrailDef("forge_spur", ((-139.0, 25.0), (-144.0, 22.4), (-151.0, 22.4), (-156.5, 22.6)), width=2.2, shoulder=2.0),
    TrailDef("hunters_path", ((-103.0, 73.5), (-106.0, 82.0), (-112.0, 89.0), (-118.5, 94.6)), width=2.2),
    TrailDef("lake_path", ((-101.0, 73.5), (-100.0, 92.0), (-99.0, 112.0), (-97.0, 126.0), (-94.0, 137.0)), width=2.2),
    TrailDef("shrine_road", ((0.0, 121.5), (1.0, 140.0), (-2.0, 156.0), (0.0, 170.0), (0.0, 178.5)), width=2.6),
    TrailDef("keep_road", ((46.5, 73.5), (47.0, 88.0), (51.0, 101.0), (60.0, 111.0), (72.5, 115.0)), width=2.8),
    TrailDef("coast_road", ((47.5, 71.75), (62.0, 69.0), (80.0, 64.0), (98.0, 59.0), (114.0, 55.0), (126.0, 52.0),
                            (133.0, 49.0)), width=2.8),
    TrailDef("cove_path", ((133.0, 49.0), (136.5, 40.0), (136.5, 30.0), (132.0, 22.0), (127.0, 20.0), (COVE_MOUTH_X, 20.0)),
             width=2.6, end_height=COVE_FLOOR, max_grade=0.2, shoulder=2.2),
    TrailDef("mine_trail", ((80.0, 64.0), (84.0, 48.0), (88.0, 30.0), (91.0, 10.0), (94.0, -10.0), (97.0, -30.0),
                            (99.5, -48.0), (100.5, -58.0), (100.5, MINE_MOUTH_Z)), width=2.4, end_height=MINE_FLOOR,
             max_grade=0.16, shoulder=2.4),
)


# =============================================================================== the parts
@dataclass
class Part:
    walls: List[WallDef] = field(default_factory=list)
    buildings: List[BuildingDef] = field(default_factory=list)
    doors: List[DoorDef] = field(default_factory=list)
    props: List[PropDef] = field(default_factory=list)
    containers: List[ContainerDef] = field(default_factory=list)
    lights: List[LightDef] = field(default_factory=list)
    areas: List[AreaDef] = field(default_factory=list)
    npcs: List[NPCDef] = field(default_factory=list)
    roads: List[RoadDef] = field(default_factory=list)
    stations: List[StationDef] = field(default_factory=list)
    readables: List[ReadableDef] = field(default_factory=list)
    holes: List[Tuple[float, float, float, float]] = field(default_factory=list)

    def clear_cave_floors(self) -> "Part":
        """Take up the dug rock that landed on anything furnished since, or where somebody sleeps,
        works or walks a round (see ``caves.clear_the_floor``)."""
        spots = [(n.home_pos[0], n.home_pos[2]) for n in self.npcs]
        for n in self.npcs:
            for t in n.schedule:
                spots += [(t.target_pos[0], t.target_pos[2]), *((q[0], q[2]) for q in t.patrol_points)]
        self.props = clear_the_floor(self.props, [*self.props, *self.containers, *self.stations, *self.readables], spots)
        return self


def at(spot2, y: float = 0.0) -> Tuple[float, float, float]:
    """A spot (x, z) as a level point on the floor under it (a cave's floor is its chamber's pad:
    ``World`` lifts everything authored at y = 0 onto it; ``y`` is above that floor)."""
    return (spot2[0], y, spot2[1])


def task(start, end, spot, state, text, **kw) -> List[TaskDef]:
    return span((start, end), state, text, spot, **kw)


def rounds(start, end, name, route, text) -> List[TaskDef]:
    """A walk round a route (``STROLLING``: a pause at every stop), wrapping past midnight if need be."""
    return span((start, end), AIState.STROLLING, text, route[0], patrol_points=tuple(route))


STEW = lambda n: ItemDef("stew", "Pot of Stew", value=4.0, stackable=True, quantity=n, is_food=True, category="food")  # noqa: E731
ALE = lambda n: ItemDef("ale", "Mug of Ale", value=3.0, stackable=True, quantity=n, is_food=True, category="food")    # noqa: E731
BREAD = lambda n: ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=n, is_food=True, category="food")  # noqa: E731
IRON_ORE = ItemDef("iron_ore", "Iron Ore", value=6.0, stackable=True, category="ore")
SILVER_ORE = ItemDef("silver_ore", "Silver Ore", value=22.0, stackable=True, category="ore")


def ore_seam(name: str, label: str, x: float, z: float, y: float = 0.0, yields: ItemDef = IRON_ORE, size: float = 1.5,
             texture: str = "ore") -> StationDef:
    """A seam of ore to be broken out with a pickaxe."""
    return StationDef(name, label, (x - size, y, z - size), (x + size, y + 1.7, z + size), texture=texture, kind="ore",
                      yields=yields, requires_item_id="pickaxe", tool_label="a pickaxe")


# ------------------------------------------------------------------------------- Thornbury
def village() -> Part:
    p = Part()
    p.walls += house_walls(12, 75, 21, 83, 15.0, "north", texture="plaster_timber")       # Odo's farmhouse
    p.walls += house_walls(-22, 75, -15, 82, -20.0, "north", texture="plaster")           # Marta's cottage
    p.walls += house_walls(-42, 75, -36, 81, -40.0, "north", texture="building")          # Haskel's hut
    p.walls += [WallDef((x, 0, z), (x + 0.3, 3.6, z + 0.3)) for x, z in ((24.0, 88.0), (35.7, 88.0), (24.0, 97.7), (35.7, 97.7))]
    p.buildings += [
        BuildingDef("odo_house", (12.0, 75.0), (21.0, 83.0), roof_texture="thatch"),
        BuildingDef("marta_house", (-22.0, 75.0), (-15.0, 82.0), roof_texture="thatch"),
        BuildingDef("haskel_hut", (-42.0, 75.0), (-36.0, 81.0), roof_texture="thatch"),
        BuildingDef("barn", (24.0, 88.0), (36.0, 98.0), wall_height=3.6, roof_height=2.2, roof_texture="thatch", open_sides=True),
    ]
    p.doors += [
        DoorDef("odo_door", (15.0, 0.0, 75.0), is_locked=True, key_id="odo_door_key", owners=("Odo the Farmer",),
                lock_difficulty=FLIMSY, label="Odo's Farmhouse"),
        DoorDef("marta_door", (-20.0, 0.0, 75.0), is_locked=True, key_id="marta_door_key", owners=("Marta",),
                lock_difficulty=FLIMSY, label="Marta's Cottage"),
        DoorDef("haskel_door", (-40.0, 0.0, 75.0), is_locked=True, key_id="haskel_door_key", owners=("Haskel the Woodcutter",),
                lock_difficulty=FLIMSY, label="Haskel's Hut"),
    ]
    p.props += [
        bed("odo_bed", ODO_BED, facing=180.0), bed("marta_bed", MARTA_BED, facing=180.0), bed("haskel_bed", HASKEL_BED, facing=180.0),
        PropDef("village_well", (-9.3, 0, 64.7), (-6.7, 1.0, 67.3), texture="wall", kind="well"),
        bridge("lane_bridge", -64.5, 69.3, -55.5, 74.2, span="x"),                  # the lane crosses the millstream east-west
        haystack("barn_hay_1", 27.0, 91.0), haystack("barn_hay_2", 32.5, 94.5, r=1.0, h=1.4), haystack("field_hay", 40.0, 104.0),
        woodpile("haskel_woodpile_1", -45.5, 75.0, -44.6, 80.5), woodpile("haskel_woodpile_2", -50.0, 82.0, -44.0, 82.9),
        PropDef("chopping_block", (-46.4, 0, 85.0), (-45.6, 0.55, 85.8), texture="log_end"),
        *barrels("village_barrels", [(21.6, 76.0), (21.6, 77.0), (-14.4, 76.0)]),
        fence("field_fence_1", 6.0, 99.0, 42.0, 99.2), fence("field_fence_2", 6.0, 99.2, 6.2, 120.0),
        fence("field_fence_3", -42.0, 87.0, -6.0, 87.2),
        *(oak(f"village_oak_{i}", x, z, height=4.0) for i, (x, z) in enumerate(
            [(-28.0, 66.0), (26.0, 66.0), (-52.0, 90.0), (8.0, 88.0), (52.0, 84.0), (-8.0, 84.0), (30.0, 124.0)])),
        *(bush(f"village_bush_{i}", x, z) for i, (x, z) in enumerate([(-12.0, 86.0), (24.0, 84.0), (-30.0, 82.0), (42.0, 76.0)])),
    ]
    p.containers += [
        table("odo_table", "Kitchen Table", (14.5, 0.0, 79.5), owner="Odo the Farmer", items=[BREAD(11)]),
        chest("odo_chest", "Farmhouse Chest", (13.2, 0.0, 81.8), owner="Odo the Farmer", along_x=False,
              items=[coins(18), ItemDef("wool_cloak", "Brown Wool Cloak", value=120.0, category="clothing")]),
        table("marta_table", "Table", (-20.5, 0.0, 79.5), owner="Marta", items=[ALE(4), BREAD(7)]),
        chest("marta_chest", "Linen Chest", (-20.9, 0.0, 76.3), owner="Marta",
              items=[coins(10), ItemDef("silk_gloves", "Silk Gloves", value=60.0, category="clothing")]),
        table("haskel_table", "Table", (-40.5, 0.0, 78.3), owner="Haskel the Woodcutter", size=(0.6, 0.45), items=[STEW(11)]),
        chest("haskel_chest", "Tool Chest", (-41.0, 0.0, 76.5), owner="Haskel the Woodcutter", along_x=False,
              items=[coins(8), ItemDef("hatchet", "Hatchet", value=14.0, category="tool")]),
        # Haskel buries what he does not drink under the woodpile: crouch by it and look
        stash("haskel_stash", "Hollow under the Woodpile", (-44.0, 0.0, 81.6), owner="Haskel the Woodcutter",
              items=[coins(35), ItemDef("silver_brooch", "Tarnished Silver Brooch", value=45.0)]),
    ]
    p.lights += [wall_torch(15.0 - 0.6, 75.0, 0.0, "Odo's farmhouse"), wall_torch(-20.0 - 0.6, 75.0, 0.0, "Marta's cottage")]
    p.areas += [
        AreaDef("Odo's farmhouse", (12.0, 75.0), (21.0, 83.0), owner="Odo the Farmer", restricted=True),
        AreaDef("Marta's cottage", (-22.0, 75.0), (-15.0, 82.0), owner="Marta", restricted=True),
        AreaDef("Haskel's hut", (-42.0, 75.0), (-36.0, 81.0), owner="Haskel the Woodcutter", restricted=True),
        AreaDef("the barn", (24.0, 88.0), (36.0, 98.0)),
    ]
    def villager(name, role, bed_spot, seat, door, food, day, lines=(), other_days=()):
        def plan(tasks):
            return resident_tasks(name.split()[0], door, bed_spot, seat, food, tasks, wake=5, bedtime=21)
        return NPCDef(name, bed_spot, role=role, gold=12.0, inventory_reset_days=3, lines=tuple(lines),
                      inventory=(key_item(door, "Cottage Key"),), schedule=plan(day), rota=tuple(plan(d) for d in other_days))
    p.npcs += [
        villager("Odo the Farmer", "farmer", ODO_BED, ODO_SEAT, "odo_door", "odo_table", [
            TaskDef(6, 12, FIELD_EAST[0], AIState.STROLLING, "Odo is working his field.", patrol_points=FIELD_EAST,
                    depart_text="The turnips won't hoe themselves."),
            TaskDef(12, 14, ODO_SEAT, AIState.EATING, "Odo is home for his dinner.", facing=0.0, food_container="odo_table",
                    unlock_doors=("odo_door",)),
            TaskDef(14, 18, FIELD_EAST[1], AIState.STROLLING, "Odo is back in his field.", patrol_points=FIELD_EAST[1:] + FIELD_EAST[:1]),
            TaskDef(18, 20, (28.0, 0.0, 93.0), AIState.WORKING, "Odo is forking hay in the barn.", facing=90.0),
        ], other_days=([                                    # market day: turnips up to the town square
            TaskDef(6, 12, MARKET_PITCH, AIState.WORKING, "Odo is selling turnips in the square. Nobody is buying.", facing=0.0,
                    depart_text="Market day. Bess, mind the field."),
            TaskDef(12, 14, ODO_SEAT, AIState.EATING, "Odo is home for his dinner.", facing=0.0, food_container="odo_table",
                    unlock_doors=("odo_door",)),
            TaskDef(14, 18, FIELD_EAST[1], AIState.STROLLING, "Odo is back in his field.", patrol_points=FIELD_EAST[1:] + FIELD_EAST[:1]),
            TaskDef(18, 20, (28.0, 0.0, 93.0), AIState.WORKING, "Odo is forking hay in the barn.", facing=90.0),
        ],), lines=("A tenth of every turnip I grow goes up to Greyhold for the King. A tenth! The King's never tasted a turnip.",
                  "The knights count the tithe like misers. Ser Caddock weighed my sack of barley twice.",
                  "Don't go near the Howling Deep. My cousin did. We buried what they sent back.")),
        villager("Marta", "villager", MARTA_BED, MARTA_SEAT, "marta_door", "marta_table", [
            TaskDef(6, 10, FIELD_WEST[0], AIState.STROLLING, "Marta is gathering in the west field.", patrol_points=FIELD_WEST),
            TaskDef(10, 13, (-8.0, 0.0, 68.9), AIState.WORKING, "Marta is drawing water at the green.", facing=0.0),
            TaskDef(13, 14, MARTA_SEAT, AIState.EATING, "Marta is having her dinner.", facing=0.0, food_container="marta_table",
                    unlock_doors=("marta_door",)),
            TaskDef(14, 20, FIELD_WEST[1], AIState.STROLLING, "Marta is back at the gleaning.", patrol_points=FIELD_WEST[1:] + FIELD_WEST[:1]),
        ], other_days=([                                    # chapel day in town
            TaskDef(6, 9, FIELD_WEST[0], AIState.STROLLING, "Marta is gathering in the west field.", patrol_points=FIELD_WEST),
            TaskDef(9, 12, CHAPEL_VISIT, AIState.WORKING, "Marta has walked up to the chapel in town to pray.", facing=180.0),
            TaskDef(12, 14, MARTA_SEAT, AIState.EATING, "Marta is having her dinner.", facing=0.0, food_container="marta_table",
                    unlock_doors=("marta_door",)),
            TaskDef(14, 20, FIELD_WEST[1], AIState.STROLLING, "Marta is back at the gleaning.", patrol_points=FIELD_WEST[1:] + FIELD_WEST[:1]),
        ],), lines=("Haskel drinks more than he chops, and he chops a great deal.",
                  "There's a hermit smith out in the Thornwood. They say he's Gareth's own brother - they don't speak.",
                  "Lights on the sea at night, by the Saltcove. Fishermen, the reeve says. Fishermen don't row without lanterns.")),
        villager("Haskel the Woodcutter", "villager", HASKEL_BED, HASKEL_SEAT, "haskel_door", "haskel_table", [
            TaskDef(6, 12, CHOPPING_SPOT, AIState.WORKING, "Haskel is splitting logs.", facing=180.0, depart_text="Wood to split."),
            TaskDef(12, 13, HASKEL_SEAT, AIState.EATING, "Haskel is eating.", facing=0.0, food_container="haskel_table",
                    unlock_doors=("haskel_door",)),
            TaskDef(13, 20, CHOPPING_SPOT, AIState.WORKING, "Haskel is back at the chopping block.", facing=180.0),
        ], other_days=([                                    # felling at the wood's edge, and a jug on the green after
            TaskDef(6, 12, FELLING_SPOT, AIState.WORKING, "Haskel is felling an oak at the edge of the Thornwood.", facing=270.0,
                    depart_text="A tree wants felling."),
            TaskDef(12, 13, HASKEL_SEAT, AIState.EATING, "Haskel is eating.", facing=0.0, food_container="haskel_table",
                    unlock_doors=("haskel_door",)),
            TaskDef(13, 17, CHOPPING_SPOT, AIState.WORKING, "Haskel is splitting the oak.", facing=180.0),
            TaskDef(17, 20, (-10.5, 0.0, 68.0), AIState.WORKING, "Haskel is sharing a jug on the green.", facing=90.0),
        ],), lines=("Banks? Never trusted a bank. Never trusted anything I couldn't dig up again myself.",
                  "The Thornwood gets so thick out west you'd need an axe and a year to get through.",
                  "Old Hrolf at the hermit's forge trades good steel for honest ore. Not much of either about.")),
    ]
    p.roads += [
        RoadDef((-3.0, 53.5), (3.0, 64.0)),                                             # out of the gate
        RoadDef((-2.5, 64.0), (2.5, 70.0), texture="dirt"),
        RoadDef((-104.0, 70.0), (48.0, 73.5), texture="dirt"),                          # the village lane
        RoadDef((-2.5, 73.5), (2.5, 122.0), texture="dirt"),                            # the road on south, past the fields
        RoadDef((8.0, 100.0), (40.0, 119.0), texture="field"), RoadDef((-40.0, 88.5), (-8.0, 108.0), texture="field"),
        RoadDef((-13.0, 65.0), (-3.0, 70.0), texture="dirt"),                           # the green by the well
    ]
    p.readables += [
        signpost("signpost_green", 4.6, 68.0, (
            "NORTH: Emberhollow - the town gate.",
            "WEST along the lane: the Thornwood - the hermit's forge, Ysolde the hunter, Thornmere lake. The Howling Deep beyond (turn back).",
            "EAST along the lane: Greyhold Keep; the coast road to the Saltcove and the old mine.",
            "SOUTH: the fields, and the old shrine of Saint Wendel.")),
        signpost("signpost_west_end", -106.5, 68.5, (
            "NORTH-WEST: the Thornwood way - Hrolf's forge, and at the mountain's foot the Howling Deep. Bandits. Do not go.",
            "SOUTH: Ysolde's hut. Thornmere.",
            "EAST: Thornbury.")),
        signpost("signpost_east_end", 49.8, 68.8, (
            "SOUTH-EAST: Greyhold Keep - the King's tithe-house. No admittance.",
            "EAST: the coast road - the Saltcove; the old Emberhollow mine (closed by order of the Reeve).",
            "WEST: Thornbury and the town.")),
        notice_board("thornbury_notices", "Thornbury Notices", -5.5, 63.2, (
            "TITHE DAY at Greyhold Keep is the last of the month. Bring a tenth of grain, fleece and coin. - Ser Caddock, Castellan.",
            "LOST: one brown cow, answers to Bess. Reward. - Odo.",
            "The OLD MINE is closed since the fall of the east stope. Do not enter. Whatever you hear down there, do not answer it. - The Reeve.",
            "WANTED for robbery on the King's road: GRIMMAW, called the Bandit King, once of Greyhold. 200 crowns."),
                     facing=180.0),
    ]
    return p


# ------------------------------------------------------------------------------- the Thornwood
def thornwood() -> Part:
    """Ysolde's hut and the hermit smith's forge."""
    p = Part()
    p.walls += house_walls(-122, 96, -115, 102, -120.0, "north", texture="building")      # the hunter's hut
    hx1, hz1, hx2, hz2 = HROLF_HUT
    p.walls += house_walls(hx1, hz1, hx2, hz2, -158.0, "north", texture="plaster_timber")  # the hermit smith's hut
    p.walls += [WallDef((x, 0, z), (x + 0.3, 3.6, z + 0.3)) for x, z in ((-151.0, 24.0), (-145.3, 24.0), (-151.0, 30.7),
                                                                         (-145.3, 30.7))]      # his forge shed's posts
    p.buildings += [
        BuildingDef("hunter_hut", (-122.0, 96.0), (-115.0, 102.0), roof_texture="thatch"),
        BuildingDef("hrolf_hut", (hx1, hz1), (hx2, hz2), roof_texture="thatch"),
        BuildingDef("hrolf_shed", (-151.0, 24.0), (-145.0, 31.0), wall_height=3.6, roof_texture="roof_slate", open_sides=True,
                    pad_with="hrolf_hut"),
    ]
    p.doors += [
        DoorDef("hunter_door", (-120.0, 0.0, 96.0), is_locked=True, key_id="hunter_door_key", owners=("Ysolde the Hunter",),
                lock_difficulty=FLIMSY, label="Ysolde's Hut"),
        DoorDef("hrolf_door", (-158.0, 0.0, hz1), is_locked=True, key_id="hrolf_door_key", owners=("Hrolf the Hermit Smith",),
                lock_difficulty=STURDY, label="The Hermit's Forge"),
    ]
    p.props += [
        bed("hunter_bed", HUNTER_BED, facing=180.0),
        PropDef("hunter_drying_rack", (-114.6, 0, 103.0), (-112.6, 1.8, 103.3), texture="wood", kind="fence"),
        PropDef("hunter_fire", (-110.6, 0, 99.4), (-109.4, 0.4, 100.6), texture="stone", kind="campfire", solid=False),
        bed("hrolf_bed", HROLF_BED, facing=270.0),
        PropDef("hrolf_forge", (-150.2, 0, 24.4), (-148.4, 1.0, 25.7), texture="stone", kind="forge", facing=180.0),
        PropDef("sign_hermit", (-153.6, 2.3, 22.9), (-152.8, 3.1, 23.7), texture="sign_smith", kind="sign", facing=270.0,
                solid=False),
        woodpile("hrolf_woodpile", -161.2, 25.0, -160.3, 31.0),
        *barrels("hrolf_barrels", [(-144.4, 29.6), (-144.4, 30.5)]),
        PropDef("hrolf_quench", (-146.4, 0, 24.6), (-145.6, 0.8, 25.4), texture="barrel", kind="barrel"),
    ]
    p.containers += [
        table("hunter_table", "Table", (-120.4, 0.0, 99.5), owner="Ysolde the Hunter", size=(0.6, 0.45),
              items=[ItemDef("venison", "Smoked Venison", value=6.0, stackable=True, quantity=11, is_food=True, category="food")]),
        chest("hunter_chest", "Hunter's Chest", (-121.0, 0.0, 97.5), owner="Ysolde the Hunter", along_x=False,
              items=[coins(25), ItemDef("shadow_boots", "Shadow Boots", value=300.0, category="clothing"),
                     ItemDef("pelt", "Fox Pelt", value=18.0, stackable=True, quantity=3)]),
        table("hrolf_table", "Table", (-154.2, 0.0, 27.8), owner="Hrolf the Hermit Smith", size=(0.6, 0.45),
              items=[STEW(9), ItemDef("venison", "Smoked Venison", value=6.0, stackable=True, quantity=3, is_food=True,
                                      category="food")]),
        ContainerDef("hrolf_rack", "Weapon Rack", (-150.6, 0, 29.9), (-147.4, 1.8, 30.6), owner="Hrolf the Hermit Smith",
                     texture="shelf", items=(ItemDef("steel_sword", "Steel Sword", value=60.0, category="weapon"),
                                             ItemDef("hunting_knife", "Hunting Knife", value=20.0, category="weapon"),
                                             ItemDef("pickaxe", "Pickaxe", value=35.0, category="tool"),
                                             ItemDef("mallet", "Mallet", value=12.0, category="tool"),
                                             ItemDef("iron_ore", "Iron Ore", value=6.0, stackable=True, quantity=6, category="ore"))),
        chest("hrolf_chest", "Hermit's Chest", (-159.0, 0.0, 26.0), owner="Hrolf the Hermit Smith", along_x=False, locked=True,
              lock_difficulty=STURDY, key_id="hrolf_door",
              items=[coins(70), ItemDef("fathers_hammer", "Father's Hammer", value=90.0, category="tool"),
                     ItemDef("letter_gareth", "Unsent Letter to Gareth", value=1.0)]),
    ]
    p.stations += [
        StationDef("hrolf_anvil", "Anvil", (-148.1, 0, 26.2), (-147.1, 0.8, 27.0), texture="stone",
                   consumes=("iron_ore", 1), consumes_label="1 Iron Ore", requires_item_id="mallet", tool_label="a mallet",
                   produces=ItemDef("iron_lockpick", "Iron Lockpick", value=5.0, is_lockpick=True, stackable=True, quantity=3,
                                    category="tool")),
        ore_seam("thornwood_ore", "Ore Seam", HROLF_ORE[0], HROLF_ORE[2]),
        ore_seam("quarry_ore", "Ore Seam", -94.5, 41.5),
    ]
    p.lights += [
        LightDef("fire", (-110.0, 0.6, 100.0), room="the hunter's clearing", range=8.0),
        LightDef("fire", (-149.3, 1.1, 25.0), room="the hermit's forge", range=8.0),
        wall_torch(-120.0 - 0.6, 96.0, 0.0, "the hunter's hut"), wall_torch(-158.0 - 0.6, hz1, 0.0, "the hermit's hut"),
    ]
    p.areas += [
        AreaDef("the hunter's hut", (-122.0, 96.0), (-115.0, 102.0), owner="Ysolde the Hunter", restricted=True),
        AreaDef("the hermit's hut", (hx1, hz1), (hx2, hz2), owner="Hrolf the Hermit Smith", restricted=True),
        AreaDef("the hermit's forge", (-151.5, 23.5), (-144.0, 31.5), owner="Hrolf the Hermit Smith"),
    ]
    p.npcs += [
        NPCDef("Ysolde the Hunter", HUNTER_BED, role="hunter", gold=40.0, inventory_reset_days=3,
               inventory=(key_item("hunter_door", "Hut Key"), ItemDef("hunting_knife", "Hunting Knife", value=20.0, category="weapon")),
               lines=("The deer come down to Thornmere at dusk. So do other things. Keep to the path.",
                      "Grimmaw's lot set snares in my wood. I cut them. They cut mine. It's a sort of conversation.",
                      "Hrolf? Gruff as a badger, honest as an axe. Don't mention his brother."),
               schedule=resident_tasks("Ysolde", "hunter_door", HUNTER_BED, HUNTER_SEAT, "hunter_table", [
                   TaskDef(6, 13, HUNTING_ROUTE[0], AIState.STROLLING, "Ysolde is stalking deer in the Thornwood.",
                           patrol_points=HUNTING_ROUTE, depart_text="The deer won't wait."),
                   TaskDef(13, 14, HUNTER_SEAT, AIState.EATING, "Ysolde is eating.", facing=0.0, food_container="hunter_table",
                           unlock_doors=("hunter_door",)),
                   TaskDef(14, 18, HUNTING_ROUTE[2], AIState.STROLLING, "Ysolde is out checking her snares.",
                           patrol_points=HUNTING_ROUTE[2:] + HUNTING_ROUTE[:2]),
                   TaskDef(18, 20, (-111.5, 0.0, 100.0), AIState.WORKING, "Ysolde is curing pelts by her fire.", facing=90.0),
               ], wake=5, bedtime=21)),
        NPCDef("Hrolf the Hermit Smith", HROLF_BED, can_trade=True, role="smith", trade_categories=("weapon", "armor", "ore", "tool"),
               shop_containers=("hrolf_rack",), restock_days=3, inventory_reset_days=3, gold=260.0,
               trade_availability="awake",
               inventory=(key_item("hrolf_door", "Hermit's Key"),
                          ItemDef("hammer", "Smith's Hammer", value=20.0, for_sale=False, category="tool"),
                          ItemDef("pickaxe", "Pickaxe", value=35.0, for_sale=False, category="tool")),
               lines=("Gareth still hammering away in town? Good. Somebody should use our father's forge. Not me. Not any more.",
                      "Bring me ore and I'll pay fair. The seam by the big rock is good iron, if you've a pick and a strong back.",
                      "I made armour for the King's men twenty years. Now I make nails and pickaxes and I sleep at night.",
                      "The old mine north-east of the town? Iron and silver both, if the roof doesn't come down on you. It did, on Harl."),
               schedule=resident_tasks("Hrolf", "hrolf_door", HROLF_BED, HROLF_SEAT, "hrolf_table", [
                   TaskDef(6, 11, HROLF_ANVIL, AIState.WORKING, "Hrolf is hammering at his anvil in the wood.", facing=0.0,
                           depart_text="Iron waits for no hermit."),
                   TaskDef(11, 12, HROLF_FORGE, AIState.WORKING, "Hrolf is working his forge.", facing=0.0),
                   TaskDef(12, 13, HROLF_SEAT, AIState.EATING, "Hrolf is eating at his table.", facing=90.0,
                           food_container="hrolf_table", unlock_doors=("hrolf_door",)),
                   TaskDef(13, 16, (HROLF_ORE[0], 0.0, HROLF_ORE[2] + 2.2), AIState.WORKING,
                           "Hrolf is breaking ore out of the seam by the big rock.", facing=0.0),
                   TaskDef(16, 19, HROLF_ANVIL, AIState.WORKING, "Hrolf is back at the anvil.", facing=0.0),
                   TaskDef(19, 20, HROLF_FORGE, AIState.WORKING, "Hrolf is banking the forge for the night.", facing=0.0),
               ], wake=5, bedtime=21, facing=90.0)),
    ]
    return p


# ------------------------------------------------------------------------------- the Howling Deep
def howling_deep() -> Part:
    """The bandit king's lair, dug under the crag at the mountain's foot: a cutting down to the
    mouth, a long tunnel, steps down into the great hall (fire, table, the straw men they shoot
    at), the bunks off it to the west, the spring to the east and, behind it, a crack nobody has
    squeezed through in years; north of the hall Grimmaw's own hall and throne, and behind him
    the vault."""
    p = Part()
    F, Fi = DEEP_FLOOR, DEEP_HALL_FLOOR
    chambers = [
        Chamber("deep_mouth", (-153.0, -80.0, -147.0, DEEP_MOUTH_Z), F, 4.2),
        Chamber("deep_hall", (-168.0, -102.0, -134.0, -80.0), Fi, 6.5),
        Chamber("deep_barracks", (-187.0, -98.0, -168.0, -84.0), Fi, 4.5),
        Chamber("deep_pool", (-134.0, -98.0, -120.0, -84.0), Fi, 4.6),
        Chamber("deep_nook", (-120.0, -97.0, -114.0, -91.0), Fi, 2.7),
        Chamber("deep_kings_hall", (-162.0, -120.0, -140.0, -102.0), Fi, 6.0),
        Chamber("deep_vault", (-157.0, -130.0, -145.0, -120.0), Fi, 4.2),
        Chamber("deep_cubby_west", (-173.0, -84.0, -168.0, -79.0), Fi, 3.4),         # little cubbies off the hall
        Chamber("deep_cubby_north", (-140.0, -106.0, -136.0, -102.0), Fi, 3.4),      # (not into the king's hall: its floor fought his)
    ]
    passages = [
        (-152.0, -81.0, -148.0, -79.0),            # the tunnel into the hall (steps down)
        (-169.0, -93.5, -167.0, -88.5),            # hall - bunks
        (-135.0, -93.5, -133.0, -88.5),            # hall - the spring
        (-121.0, -95.0, -119.0, -93.0),            # the spring - the crack
        (-155.0, -103.0, -147.0, -101.0),          # hall - the king's hall
        (VAULT_DOOR_X, -121.0, VAULT_DOOR_X + 3.0, -119.0),   # the vault door
        (-169.0, -82.6, -167.0, -80.4), (-140.0, -103.0, -137.8, -101.0),
    ]
    dug = dig("deep", chambers, passages, Mouth("deep_mouth", "south", -151.5, -148.5), seed=77)
    p.walls += dug.walls
    p.buildings += dug.buildings
    p.props += dug.props
    p.holes += dug.holes
    # steps down from the tunnel into the hall
    p.props += rock_steps("deep_steps", -152.0, -148.0, -80.0, -1.0, F - Fi, 0.0)
    # the hall
    p.props += [
        PropDef("deep_fire", (DEEP_FIRE[0] - 0.6, 0.0, DEEP_FIRE[1] - 0.6), (DEEP_FIRE[0] + 0.6, 0.4, DEEP_FIRE[1] + 0.6),
                texture="stone", kind="campfire", solid=False),
        dummy("deep_dummy_1", -163.5, -94.0, facing=90.0), dummy("deep_dummy_2", -163.5, -89.5, facing=90.0),
        PropDef("deep_weapon_rack", (-146.5, 0.0, -100.9), (-143.5, 2.0, -100.3), texture="shelf"),
        crates("deep_crates_1", -138.0, -101.0, -135.5, -98.8), crates("deep_crates_2", -166.8, -86.8, -164.6, -84.8, h=1.2),
        *(barrel(f"deep_barrel_{i}", x, z) for i, (x, z) in enumerate([(-136.0, -82.0), (-137.0, -82.0), (-165.0, -99.0)])),
        banner("deep_banner_hall", -151.0, -101.3, 180.0, "banner_deep", y=2.6, height=2.6, width=1.2),
    ]
    p.props += [
        *(bedroll(f"deep_bedroll_{i}", at(spot), facing=270.0) for i, spot in enumerate(DEEP_BUNKS[:4])),
        bedroll("deep_bedroll_4", at(DEEP_BUNKS[4]), facing=0.0),
        crates("deep_barracks_crates", -179.0, -87.2, -176.5, -85.2, h=1.0),
        pool("deep_spring", DEEP_POOL[0], DEEP_POOL[1], r=1.8),
        throne("deep_throne", BOSS_POST[0], BOSS_POST[1] - 1.2, facing=180.0),
        bed("deep_boss_bed", at(BOSS_BED), facing=270.0, double=True),
        banner("deep_banner_throne", -151.0, -119.3, 180.0, "banner_deep", y=3.0, height=2.4, width=1.0),
    ]
    shrooms, glows = mushrooms("deep_shrooms", [(-122.0, -86.0), (-129.4, -96.6), (-121.5, -96.0), (-118.6, -92.0),
                                                 (-166.5, -100.5), (-185.5, -97.0), (-160.5, -119.0), (-146.0, -129.0),
                                                 (-171.6, -80.3), (-136.9, -105.2)],
                               room="the Howling Deep")
    p.props += shrooms
    p.lights += glows
    p.lights += [
        LightDef("fire", (DEEP_FIRE[0], 0.6, DEEP_FIRE[1]), room="the Howling Deep", range=13.0),
        sconce(-156.5, -101.3, 180.0, "the Howling Deep", y=2.6), sconce(-145.5, -101.3, 180.0, "the Howling Deep", y=2.6),
        sconce(-141.0, -80.7, 0.0, "the Howling Deep", y=2.6),
        sconce(-152.3, -70.0, 90.0, "the Howling Deep", y=2.4), sconce(-147.7, -76.0, 270.0, "the Howling Deep", y=2.4),
        sconce(-134.6, -86.0, 270.0, "the Howling Deep", y=2.6), sconce(-167.1, -96.0, 90.0, "the Howling Deep", y=2.6),
        sconce(-186.1, -90.0, 90.0, "the Howling Deep", y=2.4),
        LightDef("fire", (-154.5, 1.4, -114.5), room="the Deep King's hall", range=9.0),       # braziers by the throne
        LightDef("fire", (-147.5, 1.4, -114.5), room="the Deep King's hall", range=9.0),
        LightDef("lantern", (-151.0, 2.9, -128.8), facing=0.0, room="the Deep vault", range=7.0),
    ]
    p.props += [PropDef(f"deep_brazier_{i}", (x - 0.4, 0.0, -114.9), (x + 0.4, 1.2, -114.1), texture="iron", kind="brazier")
                for i, x in enumerate((-154.5, -147.5))]
    p.doors.append(DoorDef("deep_vault_door", (VAULT_DOOR_X, 0.0, -120.0), is_locked=True, key_id="deep_vault_door_key",
                           owners=("Grimmaw the Bandit King",), lock_difficulty=LEGENDARY, height=3.2, label="The Deep King's Vault"))
    p.containers += [
        table("deep_table", "Bandits' Table", at(DEEP_TABLE), owner="the Howling Deep", size=(1.4, 0.6),
              items=[STEW(18), ALE(10), ItemDef("dice", "Loaded Dice", value=8.0)]),
        chest("deep_spoils", "Bandit Spoils", at((-158.8, -104.0)), owner="the Howling Deep", locked=True,
              lock_difficulty=STRONG, key_id="deep_vault_door",
              items=[coins(120), ItemDef("black_hood", "Black Cloth Hood", value=360.0, category="clothing")]),
        chest("ned_chest", "Ned's Loot Box", at((-181.0, -85.5)), owner="Ned Two-Fingers", locked=True,
              lock_difficulty=SIMPLE, items=[coins(48), ItemDef("gold_ring", "Gold Ring", value=150.0),
                                             ItemDef("pocket_watch", "Stolen Pocket Watch", value=85.0)]),
        pedestal("deep_pedestal", "The Deep King's Pedestal", at(VAULT_PEDESTAL),
                 owner="the Howling Deep", locked=False,
                 items=[ItemDef("amulet_open_door", "Amulet of the Open Door", value=3000.0, category="clothing")]),
        chest("deep_vault_chest", "Vault Chest", at((-147.0, -128.6)), owner="the Howling Deep", strongbox=True, locked=True,
              lock_difficulty=MASTER, key_id="deep_vault_door",
              items=[coins(500), ItemDef("black_shoes", "Black Cloth Shoes", value=620.0, category="clothing")]),
        # an earlier visitor to the vault, who found the way in and not the way out
        skeleton("deep_vault_bones", "Bones of an Unlucky Thief", at((-155.4, -125.0)),
                 items=[coins(37), ItemDef("iron_lockpick", "Iron Lockpick", value=5.0, is_lockpick=True, stackable=True, quantity=2,
                                           category="tool")]),
        # behind the spring, in the crack: a prospector who came for the silver and found the bandits first
        skeleton("deep_nook_bones", "Bones of a Prospector", at((-117.2, -94.0)), along_x=True,
                 items=[coins(64), ItemDef("silver_ore", "Silver Ore", value=22.0, stackable=True, quantity=2, category="ore")]),
        stash("deep_nook_stash", "Loose Stones", at((-115.4, -95.6)),
              items=[coins(90), ItemDef("gold_ring", "Prospector's Ring", value=150.0)]),
        sacks("deep_cubby_sacks", "Sacks of Loot", -171.9, -82.9, -170.9, -80.9, owner="the Howling Deep",
              items=(ItemDef("silk_bolt", "Bolt of Silk", value=70.0), ItemDef("silver_plate", "Silver Plate", value=60.0), coins(34))),
        stash("deep_cubby_cache", "Loose Stones", at((-138.5, -104.5)), owner="Sable the Archer",
              items=[coins(52), ItemDef("fine_arrows", "Sable's Fletched Arrows", value=12.0, stackable=True, quantity=12)]),
        stash("deep_barracks_stash", "Loose Stones", at((-185.6, -85.6)), owner="Gutter",
              items=[coins(41), ItemDef("lucky_tooth", "Gutter's Lucky Tooth", value=2.0)]),
    ]
    p.stations += [ore_seam("deep_silver_vein", "Silver Vein", -115.8, -92.3, yields=SILVER_ORE, size=0.8),
                   ore_seam("deep_iron_vein", "Ore Seam", -131.8, -96.2, size=1.0)]
    p.readables += [
        journal("deep_prospector_pages", "A Prospector's Last Pages", -116.6, 0.05, -95.9, (
            "Day 9. Silver in the crack behind the spring, a good vein, just as old Harl swore before the mine took him.",
            "Day 11. Men in the hall. Not miners. They sing at night and somebody screams. I keep very quiet.",
            "Day 14. Their king sits before an iron door and never sleeps more than three hours. The door has no lock I have seen the like of.",
            "Day 16. The water is running out and they have a man at the spring every dawn. I have hidden the ring under the stones. Whoever finds this - "),
                facing=180.0),
    ]
    p.areas += [
        AreaDef("the Deep vault", (-157.0, -130.0), (-145.0, -120.0), owner="the Howling Deep", restricted=True, underground=True),
        AreaDef("the Deep King's hall", (-162.0, -120.0), (-140.0, -102.0), owner="the Howling Deep", restricted=True,
                underground=True),
        AreaDef("the Howling Deep", (-187.0, -102.0), (-114.0, DEEP_MOUTH_Z), owner="the Howling Deep", restricted=True,
                underground=True),
    ]
    domain = ("the Howling Deep", "the Deep King's hall", "the Deep vault")
    bandit_look = lambda tunic, hood: Outfit(tunic=tunic, legs=(0.2, 0.18, 0.16), hat="hood", hat_color=hood)   # noqa: E731
    boss_look = Outfit(tunic=(0.3, 0.1, 0.1), legs=(0.15, 0.12, 0.12), armor=True, hat="hood", hat_color=(0.12, 0.1, 0.1),
                       hair=(0.12, 0.1, 0.08))
    fire_seat = lambda i: at(DEEP_FIRE_SEATS[i])                                                          # noqa: E731
    seat = lambda i: at(DEEP_SEATS[i])                                                                    # noqa: E731
    route = tuple(at(pt) for pt in DEEP_ROUNDS)
    mouth = at(DEEP_MOUTH_POST)
    outlaw = dict(faction="the Howling Deep", warden=domain, warden_exit=DEEP_EXIT, inventory_reset_days=2)
    p.npcs += [
        NPCDef("Grimmaw the Bandit King", at(BOSS_BED), role="boss", gold=180.0, outfit=boss_look,
               **{**outlaw, "inventory_reset_days": 5},
               inventory=(key_item("deep_vault_door", "The Deep King's Key"),
                          ItemDef("gold_crown", "Battered Gold Crown", value=400.0, for_sale=False)),
               lines=("I wore Greyhold's grey once. Took an oath. Then I took their silver, and they took my spurs. Fair trade.",
                      "Three hours' sleep is all a king needs. The rest of the night I listen. I hear you, by the way.",
                      "Everything in this hole is mine. Everything. Even the echoes."),
               schedule=(
                   *task(2, 5, at(BOSS_BED), AIState.SLEEPING, "Grimmaw is asleep on his furs."),
                   *task(5, 6, seat(0), AIState.EATING, "Grimmaw is breaking his fast at the long table.", facing=90.0,
                         food_container="deep_table"),
                   *rounds(6, 9, "Grimmaw", (at((-151.0, -92.0)), at((-176.0, -90.0)), at((-130.0, -90.0)),
                                             at((-150.0, -84.0))), "Grimmaw is walking his halls, looking his crew over."),
                   *task(9, 14, at(BOSS_POST), AIState.GUARDING, "Grimmaw is brooding on his throne before the vault.",
                         facing=180.0),
                   *task(14, 15, seat(1), AIState.EATING, "Grimmaw is eating with his crew.", facing=270.0, food_container="deep_table"),
                   *task(15, 22, at(BOSS_POST), AIState.GUARDING, "Grimmaw is holding court on his throne.", facing=180.0),
                   *task(22, 2, fire_seat(2), AIState.DRINKING, "Grimmaw is drinking by the great fire.", facing=180.0),
               )),
        NPCDef("Vey the Cutthroat", at(DEEP_BUNKS[0]), role="bandit", gold=30.0, outfit=bandit_look((0.25, 0.22, 0.2), (0.18, 0.16, 0.15)),
               **outlaw,
               lines=("Walk on, stranger. Quick, while you still can.",
                      "I keep the mouth. Nobody gets past Vey. Well. Nobody gets past Vey twice."),
               schedule=(
                   *task(0, 4, mouth, AIState.GUARDING, "Vey is keeping watch at the mouth of the Deep.", facing=180.0),
                   *rounds(4, 6, "Vey", route, "Vey is walking the tunnels."),
                   *task(6, 7, seat(2), AIState.EATING, "Vey is eating.", facing=90.0, food_container="deep_table"),
                   *task(7, 10, at(DEEP_RACK_SPOT), AIState.WORKING, "Vey is whetting his knives by the rack.", facing=0.0),
                   *task(10, 16, at(DEEP_BUNKS[0]), AIState.SLEEPING, "Vey is asleep."),
                   *task(16, 18, mouth, AIState.GUARDING, "Vey is watching the way in.", facing=180.0),
                   *rounds(18, 20, "Vey", route[2:] + route[:2], "Vey is making his rounds."),
                   *task(20, 24, fire_seat(0), AIState.DRINKING, "Vey is drinking by the fire.", facing=90.0),
               )),
        NPCDef("Sable the Archer", at(DEEP_BUNKS[1]), role="bandit", gold=26.0,
               outfit=bandit_look((0.2, 0.28, 0.2), (0.14, 0.2, 0.14)), **outlaw,
               lines=("I could put an arrow through your eye from the throne. Don't make me get up.",
                      "Grimmaw found me starving in a ditch. I'd follow him into the sea."),
               schedule=(
                   *task(4, 12, at(DEEP_BUNKS[1]), AIState.SLEEPING, "Sable is asleep."),
                   *task(12, 13, seat(3), AIState.EATING, "Sable is eating.", facing=270.0, food_container="deep_table"),
                   *task(13, 16, at(DEEP_DUMMY_SPOT), AIState.WORKING, "Sable is loosing arrows at the straw men.", facing=270.0),
                   *rounds(16, 18, "Sable", route[3:] + route[:3], "Sable is walking the tunnels."),
                   *task(18, 20, fire_seat(1), AIState.DRINKING, "Sable is at the fire with the others.", facing=270.0),
                   *task(20, 24, mouth, AIState.GUARDING, "Sable has the watch at the mouth.", facing=180.0),
                   *rounds(0, 4, "Sable", route, "Sable is prowling the tunnels."),
               )),
        NPCDef("Gutter", at(DEEP_BUNKS[2]), role="bandit", gold=18.0, outfit=bandit_look((0.35, 0.25, 0.18), (0.25, 0.2, 0.15)),
               **outlaw,
               lines=("Wha'?", "Gutter not stupid. Gutter just big. Different thing.", "Don't touch Gutter's stones."),
               schedule=(
                   *task(13, 21, at(DEEP_BUNKS[2]), AIState.SLEEPING, "Gutter is snoring in the bunk room."),
                   *task(21, 3, fire_seat(3), AIState.DRINKING, "Gutter is drinking by the fire.", facing=0.0),
                   *task(3, 5, at((DEEP_POOL[0] - 2.2, DEEP_POOL[1])), AIState.WORKING, "Gutter is hauling water from the spring.",
                         facing=90.0),
                   *task(5, 7, seat(1), AIState.EATING, "Gutter is eating for three.", facing=270.0, food_container="deep_table"),
                   *task(7, 13, at((-158.0, -85.0)), AIState.WORKING, "Gutter is lifting rocks for want of anything better.",
                         facing=0.0),
               )),
        NPCDef("Ned Two-Fingers", at(DEEP_BUNKS[3]), role="bandit", gold=40.0, outfit=bandit_look((0.3, 0.28, 0.32), (0.2, 0.18, 0.22)),
               **outlaw,
               lines=("Fancy a game? Dice. Honest dice. Mostly honest dice.",
                      "Lost three fingers to a lockbox in Greyhold. Pays to know when to stop. I never did."),
               schedule=(
                   *task(0, 8, at(DEEP_BUNKS[3]), AIState.SLEEPING, "Ned is asleep."),
                   *task(8, 9, seat(2), AIState.EATING, "Ned is eating.", facing=90.0, food_container="deep_table"),
                   *task(9, 13, seat(3), AIState.WORKING, "Ned is throwing dice at the long table.", facing=270.0),
                   *task(13, 16, at((-181.0, -86.6)), AIState.WORKING, "Ned is counting his loot in the bunk room.", facing=0.0),
                   *rounds(16, 19, "Ned", route[1:] + route[:1], "Ned is wandering the halls, whistling."),
                   *task(19, 24, seat(0), AIState.DRINKING, "Ned is dicing and drinking at the table.", facing=90.0),
               )),
        NPCDef("Old Crake", at(DEEP_BUNKS[4]), role="bandit", gold=12.0, outfit=bandit_look((0.42, 0.36, 0.28), (0.3, 0.26, 0.2)),
               **outlaw,
               lines=("Stew's stew. Don't ask what's in it. I don't.", "Been cooking for outlaws forty years. They die. I don't."),
               schedule=(
                   *task(21, 4, at(DEEP_BUNKS[4]), AIState.SLEEPING, "Old Crake is asleep by the wall."),
                   *task(4, 6, fire_seat(1), AIState.WORKING, "Old Crake is stirring the pot over the fire.", facing=270.0),
                   *task(6, 7, seat(1), AIState.EATING, "Old Crake is eating his own cooking.", facing=270.0, food_container="deep_table"),
                   *task(7, 11, at((DEEP_POOL[0] + 2.2, DEEP_POOL[1])), AIState.WORKING, "Old Crake is scouring pots at the spring.",
                         facing=270.0),
                   *task(11, 13, fire_seat(1), AIState.WORKING, "Old Crake is cooking.", facing=270.0),
                   *task(13, 14, seat(3), AIState.EATING, "Old Crake is eating.", facing=270.0, food_container="deep_table"),
                   *task(14, 17, at((-170.5, -88.0)), AIState.WORKING, "Old Crake is mending blankets in the bunk room.", facing=90.0),
                   *task(17, 20, fire_seat(1), AIState.WORKING, "Old Crake is cooking supper.", facing=270.0),
                   *task(20, 21, seat(2), AIState.EATING, "Old Crake is eating.", facing=90.0, food_container="deep_table"),
               )),
    ]
    return p


# ------------------------------------------------------------------------------- the Saltcove
def saltcove() -> Part:
    """The smugglers: a cleft up from the beach into the bluff, a tunnel, the den (fire, table,
    bedrolls), the store behind it (the stash, the sacks) and a crack in the den's west wall. On
    the beach their boat, drawn up, and the rock the lookout sits on."""
    p = Part()
    F = COVE_FLOOR
    chambers = [
        Chamber("cove_mouth", (114.0, 17.0, COVE_MOUTH_X, 23.0), F, 4.0),
        Chamber("cove_den", (99.0, 10.0, 114.0, 30.0), F, 5.0),
        Chamber("cove_store", (102.0, 30.0, 113.0, 38.0), F, 4.0),
        Chamber("cove_nook", (94.0, 13.0, 99.0, 18.0), F, 2.6),
        Chamber("cove_cubby", (95.0, 23.5, 99.0, 27.5), F, 3.0),                      # the rum keg's hidey-hole
    ]
    passages = [
        (113.0, 18.5, 115.0, 21.5),
        (105.0, 29.0, 109.0, 31.0),
        (98.0, 14.6, 100.0, 16.4),
        (98.0, 24.4, 100.0, 26.6),
    ]
    dug = dig("cove", chambers, passages, Mouth("cove_mouth", "east", 18.5, 21.5), seed=31)
    p.walls += dug.walls
    p.buildings += dug.buildings
    p.props += dug.props
    p.holes += dug.holes
    p.props += [
        PropDef("cove_fire", (COVE_FIRE[0] - 0.6, 0.0, COVE_FIRE[1] - 0.6), (COVE_FIRE[0] + 0.6, 0.4, COVE_FIRE[1] + 0.6),
                texture="stone", kind="campfire", solid=False),
        *(bedroll(f"cove_bedroll_{i}", at(spot), facing=270.0 if spot[0] < 105 else 0.0) for i, spot in enumerate(COVE_BEDS)),
        crates("cove_crates_1", 109.6, 34.6, 111.9, 36.9), crates("cove_crates_2", 103.1, 31.1, 105.5, 33.3),
        crates("cove_crates_den", 111.2, 26.4, 112.8, 28.8, h=1.1),
        *(barrel(f"cove_barrel_{i}", x, z) for i, (x, z) in enumerate([(111.6, 31.6), (111.6, 32.6), (100.6, 28.2), (101.6, 28.4)])),
        # on the beach
        rowboat("cove_boat", COVE_BOAT[0], COVE_BOAT[1], facing=90.0),
        crates("cove_beach_crates", 132.2, 23.4, 134.0, 25.6, h=1.1),
        boulder("cove_lookout_rock", COVE_LOOKOUT[0] + 1.9, COVE_LOOKOUT[1], r=1.1, h=1.2),
        *(boulder(f"cove_shore_rock_{i}", x, z, r=r, h=r * 1.1, texture="cliff") for i, (x, z, r) in enumerate(
            [(127.5, 16.5, 1.3), (126.5, 23.6, 1.2), (130.0, 14.0, 1.6), (131.0, 26.5, 0.9)])),
        PropDef("cove_nets", (COVE_NETS[0] - 1.0, 0.0, COVE_NETS[1] - 1.3), (COVE_NETS[0] + 1.0, 1.2, COVE_NETS[1] - 0.7),
                texture="wood", kind="fence"),
    ]
    shrooms, glows = mushrooms("cove_shrooms", [(95.0, 16.8), (98.0, 13.8), (112.4, 15.0), (102.3, 37.3), (96.0, 26.8)],
                               room="the smugglers' cave")
    p.props += shrooms
    p.lights += glows
    p.lights += [
        LightDef("fire", (COVE_FIRE[0], 0.6, COVE_FIRE[1]), room="the smugglers' cave", range=10.0),
        sconce(119.0, 17.7, 180.0, "the smugglers' cave", y=2.4), sconce(108.0, 37.3, 0.0, "the smugglers' cave", y=2.3),
        sconce(113.3, 14.0, 270.0, "the smugglers' cave", y=2.5),
    ]
    p.containers += [
        table("cove_table", "Smugglers' Table", at(COVE_TABLE), owner="the smugglers", size=(1.0, 0.55),
              items=[STEW(24), ItemDef("brandy", "Contraband Brandy", value=30.0, stackable=True, quantity=2, category="food"),
                     ItemDef("cards", "Greasy Deck of Cards", value=4.0)]),
        chest("smugglers_stash", "Smugglers' Stash", at((108.0, 36.6)), owner="the smugglers", locked=True,
              lock_difficulty=STRONG, key_id="smugglers_stash",
              items=[coins(150), ItemDef("shadowsilk_hood", "Shadowsilk Hood", value=900.0, category="clothing"),
                     ItemDef("black_tunic", "Black Linen Tunic", value=320.0, category="clothing"),
                     ItemDef("brandy", "Contraband Brandy", value=30.0, stackable=True, quantity=4, category="food")]),
        sacks("smuggler_sacks", "Sacks of Contraband", 103.2, 33.8, 105.4, 36.9, owner="the smugglers",
              items=(ItemDef("pepper", "Sack of Pepper", value=40.0, stackable=True, quantity=2),
                     ItemDef("silk_bolt", "Bolt of Silk", value=70.0))),
        # in the crack: Jory, who tried to keep a share for himself
        skeleton("cove_nook_bones", "Bones of a Drowned Man", at((96.4, 15.6)), along_x=True,
                 items=[coins(52), ItemDef("silver_locket", "Silver Locket", value=55.0)]),
        ContainerDef("cove_rum_keg", "Rum Keg", (96.0, 0.0, 24.6), (97.0, 1.1, 25.6), owner="the smugglers", texture="barrel",
                     kind="barrel_store", items=(ItemDef("rum", "Dark Rum", value=18.0, stackable=True, quantity=3, category="food"),
                                                 coins(45))),
        stash("cove_kell_stash", "Loose Stones", at((100.4, 11.3)), owner="Kell the Smuggler",
              items=[coins(110), ItemDef("sea_chart", "Smuggler's Sea Chart", value=40.0)]),
    ]
    p.readables += [
        journal("cove_jory_letter", "A Water-Stained Letter", 95.2, 0.05, 14.6, (
            "Bess - if Kell finds this I'm a dead man, so I'll keep it short.",
            "I've put by my share of the silk money, and more than my share. It's in the crack behind the den, where only a thin man fits.",
            "Kell counts the take every evening and sleeps late. Rusk hauls the crates from the boat at dusk. Nobody watches the crack.",
            "When the tide's right I'll come for you. Burn this. - Jory"), facing=90.0),
    ]
    p.areas += [AreaDef("the smugglers' cave", (94.0, 10.0), (COVE_MOUTH_X, 38.0), owner="the smugglers", restricted=True,
                        underground=True)]
    sailor = lambda tunic: Outfit(tunic=tunic, legs=(0.24, 0.24, 0.3), hat="scarf", hat_color=(0.5, 0.14, 0.12))   # noqa: E731
    domain = ("the smugglers' cave",)
    crew = dict(faction="the smugglers", warden=domain, warden_exit=SMUGGLERS_EXIT, inventory_reset_days=2)
    fire_seat = lambda i: at(COVE_FIRE_SEATS[i])                                                    # noqa: E731
    seat = lambda i: at(COVE_SEATS[i])                                                              # noqa: E731
    den_round = (at((106.0, 24.0)), at((108.0, 34.0)), at((117.0, 20.0)), at((102.8, 20.0)))
    beach = lambda spot: (spot[0], 0.0, spot[1])                                                       # noqa: E731
    p.npcs += [
        NPCDef("Kell the Smuggler", at(COVE_BEDS[0]), role="smuggler", gold=45.0, outfit=sailor((0.22, 0.3, 0.4)), **crew,
               inventory=(ItemDef("smugglers_key", "Stash Key", value=1.0, is_key=True, key_door_id="smugglers_stash",
                                  for_sale=False, category="key"),
                          ItemDef("brandy", "Contraband Brandy", value=30.0, stackable=True, quantity=1, category="food")),
               lines=("Fishing. We're fishermen. Very quiet fishermen, with very heavy fish.",
                      "The excise men at Greyhold would hang the lot of us. So we pay them not to look. Everybody pays somebody.",
                      "Grimmaw wanted a cut of our runs. I told him where to put his cut. We don't go inland any more."),
               schedule=(
                   *task(8, 14, at(COVE_BEDS[0]), AIState.SLEEPING, "Kell is asleep in the cave."),
                   *task(14, 15, seat(0), AIState.EATING, "Kell is eating at the table.", facing=90.0, food_container="cove_table"),
                   *task(15, 18, at(COVE_STORE_SPOT), AIState.WORKING, "Kell is tallying the take in the store.", facing=180.0),
                   *task(18, 20, beach((COVE_BOAT[0] - 2.6, COVE_BOAT[1] + 1.0)), AIState.WORKING,
                         "Kell is down at the boat, looking out to sea.", facing=90.0),
                   *task(20, 24, fire_seat(0), AIState.DRINKING, "Kell is drinking by the fire.", facing=90.0),
                   *rounds(0, 2, "Kell", den_round, "Kell is looking the cave over."),
                   *task(2, 8, seat(1), AIState.WORKING, "Kell is playing cards at the table.", facing=270.0),
               )),
        NPCDef("Rusk the Smuggler", at(COVE_BEDS[1]), role="smuggler", gold=35.0, outfit=sailor((0.4, 0.3, 0.22)), **crew,
               lines=("Move. You're standing where I put crates.", "I can carry two kegs up that beach. Kell can carry a tune. Barely."),
               schedule=(
                   *task(12, 20, at(COVE_BEDS[1]), AIState.SLEEPING, "Rusk is asleep in the cave."),
                   *rounds(20, 22, "Rusk", COVE_HAUL, "Rusk is hauling crates up from the boat."),
                   *task(22, 2, fire_seat(1), AIState.DRINKING, "Rusk is drinking by the fire.", facing=270.0),
                   *task(2, 6, seat(0), AIState.WORKING, "Rusk is playing cards with Kell, and losing.", facing=90.0),
                   *task(6, 7, seat(2), AIState.EATING, "Rusk is eating.", facing=0.0, food_container="cove_table"),
                   *task(7, 12, at(COVE_MOUTH_POST), AIState.GUARDING, "Rusk is minding the way in.", facing=90.0),
               )),
        NPCDef("Maggs", at(COVE_BEDS[2]), role="smuggler", gold=20.0,
               outfit=Outfit(tunic=(0.5, 0.42, 0.32), legs=(0.3, 0.26, 0.22), apron="cloth", hat="scarf", hat_color=(0.6, 0.5, 0.4)),
               **crew,
               lines=("Wipe your feet. It's a cave, not a barn.", "I cooked on a privateer for nine years. This lot eat better and thank me less."),
               schedule=(
                   *task(22, 6, at(COVE_BEDS[2]), AIState.SLEEPING, "Maggs is asleep."),
                   *task(6, 8, fire_seat(2), AIState.WORKING, "Maggs is cooking over the fire.", facing=0.0),
                   *task(8, 9, seat(2), AIState.EATING, "Maggs is eating.", facing=0.0, food_container="cove_table"),
                   *task(9, 12, beach(COVE_NETS), AIState.WORKING, "Maggs is gutting fish on the beach.", facing=90.0),
                   *task(12, 14, fire_seat(2), AIState.WORKING, "Maggs is cooking.", facing=0.0),
                   *task(14, 18, at((106.2, 33.0)), AIState.WORKING, "Maggs is mending sacks in the store.", facing=90.0),
                   *task(18, 20, fire_seat(2), AIState.WORKING, "Maggs is cooking supper.", facing=0.0),
                   *task(20, 22, seat(1), AIState.EATING, "Maggs is having her supper.", facing=270.0, food_container="cove_table"),
               )),
        NPCDef("Finch", at(COVE_BEDS[3]), role="smuggler", gold=8.0, outfit=sailor((0.55, 0.5, 0.3)), **crew,
               lines=("I'm the lookout! Well. I look out. Mostly at the sea.", "Kell says one day I'll get my own boat. I'd call her Bess."),
               schedule=(
                   *task(6, 14, at(COVE_BEDS[3]), AIState.SLEEPING, "Finch is asleep."),
                   *task(14, 15, seat(1), AIState.EATING, "Finch is eating.", facing=270.0, food_container="cove_table"),
                   *task(15, 22, beach(COVE_LOOKOUT), AIState.GUARDING, "Finch is on the lookout rock, watching the sea.", facing=90.0),
                   *rounds(22, 2, "Finch", ((133.0, 0.0, 44.0), (137.0, 0.0, 30.0), (133.0, 0.0, 18.0), (138.2, 0.0, 36.0)),
                           "Finch is walking the beach with a shuttered lantern."),
                   *task(2, 6, beach(COVE_LOOKOUT), AIState.GUARDING, "Finch is on the lookout rock.", facing=90.0),
               )),
        NPCDef("Old Brannoc", at(COVE_BEDS[4]), role="smuggler", gold=16.0, outfit=sailor((0.3, 0.3, 0.36)), **crew,
               lines=("Forty years before the mast. The sea gives, the sea takes. Mostly takes.",
                      "See that crack in the den wall? Jory went in there one night and never came out. Nobody goes in now."),
               schedule=(
                   *task(0, 7, at(COVE_BEDS[4]), AIState.SLEEPING, "Old Brannoc is asleep."),
                   *task(7, 8, seat(2), AIState.EATING, "Old Brannoc is eating.", facing=0.0, food_container="cove_table"),
                   *task(8, 12, beach((COVE_NETS[0], COVE_NETS[1] + 1.2)), AIState.WORKING, "Old Brannoc is mending nets on the beach.",
                         facing=0.0),
                   *task(12, 16, beach((COVE_BOAT[0] - 1.6, COVE_BOAT[1] - 1.6)), AIState.WORKING, "Old Brannoc is caulking the boat.",
                         facing=90.0),
                   *task(16, 18, fire_seat(1), AIState.DRINKING, "Old Brannoc is spinning yarns by the fire.", facing=270.0),
                   *task(18, 20, seat(0), AIState.EATING, "Old Brannoc is at his supper.", facing=90.0, food_container="cove_table"),
                   *task(20, 24, at((106.5, 33.5)), AIState.WORKING, "Old Brannoc is checking the ropes in the store.", facing=0.0),
               )),
    ]
    return p


# ------------------------------------------------------------------------------- the old mine
def old_mine() -> Part:
    """Emberhollow's iron mine, worked out and fallen in: an adit into the spur, a long gallery
    on pit props, the collapsed stope to the east (where Harl the pit-boss lies) and the flooded
    sump down steps to the west. Ore left in the walls, mushrooms in the damp, nobody about."""
    p = Part()
    F, Fs = MINE_FLOOR, MINE_SUMP_FLOOR
    chambers = [
        Chamber("mine_adit", (98.0, -80.0, 103.0, MINE_MOUTH_Z), F, 3.6),
        Chamber("mine_gallery", (84.0, -98.0, 114.0, -80.0), F, 5.0),
        Chamber("mine_stope", (114.0, -96.0, 126.0, -84.0), F, 4.0),
        Chamber("mine_sump", (86.0, -112.0, 98.0, -98.0), Fs, 4.0),
        Chamber("mine_cubby", (80.0, -90.5, 84.0, -86.0), F, 3.0),                    # a miner's niche
    ]
    passages = [(99.0, -81.0, 102.0, -79.0), (113.0, -92.0, 115.0, -88.0), (90.0, -99.0, 94.0, -97.0), (83.0, -89.4, 85.0, -87.2)]
    dug = dig("mine", chambers, passages, Mouth("mine_adit", "south", 99.0, 102.0), seed=53)
    p.walls += dug.walls
    p.buildings += dug.buildings
    p.props += dug.props
    p.holes += dug.holes
    p.props += rock_steps("mine_steps", 90.0, 94.0, -98.0, -1.0, F - Fs, 0.0)
    p.props += [
        *(timber(f"mine_timber_adit_{i}", 100.5, z, 4.6, 3.3) for i, z in enumerate((-69.0, -73.0, -77.0))),
        *(timber(f"mine_timber_gallery_{i}", x, -89.0, 17.0, 4.6, along_x=False) for i, x in enumerate((88.0, 98.0, 106.0))),     # (clear of the ways in and out)
        minecart("mine_cart", 96.0, -86.0, facing=90.0),
        minecart("mine_cart_2", 110.0, -95.0),
        *(boulder(f"mine_rubble_{i}", x, z, r=r, h=r * 1.2, texture="cave_rock") for i, (x, z, r) in enumerate(
            [(124.0, -93.0, 1.4), (121.0, -92.2, 1.0), (125.0, -89.2, 1.1), (120.0, -95.0, 0.9)])),
        pool("mine_sump_water", 92.0, -107.0, r=2.6),
    ]
    shrooms, glows = mushrooms("mine_shrooms", [(88.0, -104.0), (94.0, -110.8), (87.3, -108.6), (95.0, -100.5), (118.0, -85.5),
                                                 (85.5, -82.0), (81.0, -89.6), (104.5, -96.6), (89.5, -96.6),
                                                 (112.6, -81.4)], room="the old mine")
    p.props += shrooms
    p.lights += glows
    p.stations += [
        ore_seam("mine_ore_west", "Ore Seam", 86.4, -91.0, size=1.1),
        ore_seam("mine_ore_east", "Ore Seam", 111.8, -83.2, size=1.0),
        ore_seam("mine_ore_stope", "Ore Seam", 123.8, -85.6, size=0.9),
        ore_seam("mine_silver_vein", "Silver Vein", 96.2, -110.4, yields=SILVER_ORE, size=0.9),
    ]
    p.containers += [
        ContainerDef("mine_tool_crate", "Miners' Tool Crate", (86.0, 0.0, -84.0), (87.6, 0.9, -82.4), texture="crate",
                     items=(ItemDef("pickaxe", "Rusty Pickaxe", value=20.0, category="tool"),
                            ItemDef("candle_stub", "Tallow Candle", value=2.0, stackable=True, quantity=3))),
        skeleton("mine_harl_bones", "Bones of Harl the Pit-Boss", at((119.6, -90.0)), along_x=False,
                 items=[coins(46), ItemDef("iron_ore", "Iron Ore", value=6.0, stackable=True, quantity=3, category="ore")]),
        # the wages Harl never paid out: under a loose stone by the stope's timbers
        stash("mine_wage_box", "Loose Stones", at((116.4, -94.4)),
              items=[coins(140), ItemDef("pit_boss_seal", "Pit-Boss's Seal", value=30.0)]),
        skeleton("mine_sump_bones", "Bones of a Drowned Miner", at((88.0, -110.6)), along_x=True,
                 items=[coins(21)]),
        stash("mine_niche_stash", "Loose Stones", at((81.4, -87.6)),
              items=[coins(33), ItemDef("candle_stub", "Tallow Candle", value=2.0, stackable=True, quantity=4)]),
    ]
    p.readables += [
        journal("mine_tally_book", "Harl's Tally Book", 118.6, 0.05, -88.4, (
            "The east stope is rich - richest seam in the hill - but the roof's bad. I've asked the Reeve for timber thrice.",
            "Wages are late again. The owners in town want ore, not excuses. I've kept back the men's pay in the box by the timbers until they pay for props.",
            "The sump floods faster every week. Old Tam swears there's silver down there. Old Tam swears a lot.",
            "The roof is talking tonight. If it comes down, the pay is under the loose stones - see that the lads get it."),
                facing=0.0),
        plaque("mine_warning", "A Warning Board", 97.9, MINE_MOUTH_Z + 0.4, 180.0, (
            "THIS MINE IS CLOSED by order of the Reeve of Emberhollow, after the fall of the east stope.",
            "Trespassers enter at their own peril. The Reeve will not dig you out."), y=1.5),
    ]
    p.areas += [AreaDef("the old mine", (84.0, -112.0), (126.0, MINE_MOUTH_Z), underground=True)]
    return p


# ------------------------------------------------------------------------------- Greyhold
def greyhold() -> Part:
    """The old border keep on its rise by the shore: the Crown's tithe-house for the March.
    Three of the Order of the Grey and a squire hold it; the tithe of town and village lies in
    its treasury until the King's collector comes at midsummer - and with it the Nightweave,
    taken long ago from a thief-queen and never returned."""
    p = Part()
    kx1, kz1, kx2, kz2 = 72.0, 96.0, 116.0, 136.0          # the curtain wall
    t, h = 1.2, 6.0
    p.walls += [
        WallDef((kx1, 0, kz1), (kx2, h, kz1 + t), texture="wall_cracked"),
        WallDef((kx1, 0, kz2 - t), (kx2, h, kz2), texture="wall"),
        WallDef((kx1, 0, kz1 + t), (kx1 + t, h, 111.5), texture="wall"),               # the west wall, the gateway in it
        WallDef((kx1, 0, 118.5), (kx1 + t, h, kz2 - t), texture="wall_cracked"),
        WallDef((kx2 - t, 0, kz1 + t), (kx2, h, 118.0), texture="wall_cracked"),
        WallDef((kx2 - t, 0, 122.0), (kx2, h, kz2 - t), texture="wall"),                # (a breach, long fallen)
        # corner towers and the gate towers (each in one piece, its top level for the merlons)
        WallDef((kx1 - 1.0, 0, kz1 - 1.0), (kx1 + 3.0, 8.0, kz1 + 3.0), texture="wall", segment=5.0),
        WallDef((kx2 - 3.0, 0, kz1 - 1.0), (kx2 + 1.0, 8.0, kz1 + 3.0), texture="wall_cracked", segment=5.0),
        WallDef((kx1 - 1.0, 0, kz2 - 3.0), (kx1 + 3.0, 8.0, kz2 + 1.0), texture="wall_cracked", segment=5.0),
        WallDef((kx2 - 3.0, 0, kz2 - 3.0), (kx2 + 1.0, 8.0, kz2 + 1.0), texture="wall", segment=5.0),
        WallDef((kx1 - 1.5, 0, 108.5), (kx1 + 1.5, 7.5, 111.5), texture="wall", segment=5.0),
        WallDef((kx1 - 1.5, 0, 118.5), (kx1 + 1.5, 7.5, 121.5), texture="wall", segment=5.0),
        WallDef((kx1, 5.0, 111.5), (kx1 + t, 7.5, 118.5), texture="wall"),               # the arch over the gateway
    ]
    from ..tower import crenellations
    for (x1, z1, x2, z2, top) in ((kx1 - 1.0, kz1 - 1.0, kx1 + 3.0, kz1 + 3.0, 8.0), (kx2 - 3.0, kz1 - 1.0, kx2 + 1.0, kz1 + 3.0, 8.0),
                                  (kx1 - 1.0, kz2 - 3.0, kx1 + 3.0, kz2 + 1.0, 8.0), (kx2 - 3.0, kz2 - 3.0, kx2 + 1.0, kz2 + 1.0, 8.0),
                                  (kx1 - 1.5, 108.5, kx1 + 1.5, 111.5, 7.5), (kx1 - 1.5, 118.5, kx1 + 1.5, 121.5, 7.5)):
        p.walls += crenellations(x1, z1, x2, z2, top, wall=0.6, texture="wall", in_the_open=True)
    # the keep: a great hall and, behind a stout door, the treasury
    p.walls += house_walls(86, 104, 106, 126, 93.5, "north", texture="stone", height=6.0)
    p.walls += partition_z(117.7, 86 + WALL, 106 - WALL, height=6.0, door=(94.5, 97.5), texture="stone")
    p.buildings += [BuildingDef("greyhold_keep", (86.0, 104.0), (106.0, 126.0), wall_height=6.0, roof_height=2.8,
                                roof_texture="roof_slate", floor_texture="stone")]
    p.doors += [
        DoorDef("keep_door", (93.5, 0.0, 104.0), is_locked=False, key_id="keep_door_key", owners=("Greyhold",),
                lock_difficulty=STRONG, swing_out=False, label="Greyhold Keep"),
        # the gates in the west wall (doors on a north-south line): two leaves, standing open into
        # the courtyard by day and night - the Order fears nobody - and pushed to by nobody
        DoorDef("greyhold_gate_north", (kx1 + t / 2, 0.0, 111.5), axis="z", width=3.5, height=4.8, swing_out=True, is_open=True,
                is_locked=False, key_id="greyhold_gate_key", lock_difficulty=STRONG, gate="greyhold_gate", label="Greyhold Gate"),
        DoorDef("greyhold_gate_south", (kx1 + t / 2, 0.0, 118.5), axis="z", width=3.5, height=4.8, swing_out=True, is_open=True,
                hinge_right=True, is_locked=False, key_id="greyhold_gate_key", lock_difficulty=STRONG, gate="greyhold_gate",
                label="Greyhold Gate"),
        DoorDef("treasury_door", (94.5, 0.0, 117.85), is_locked=True, key_id="treasury_door_key", owners=("Greyhold",),
                lock_difficulty=MASTER, label="Greyhold Treasury"),
    ]
    p.props += [
        *(bed(f"knight_bed_{i}", spot, facing=270.0) for i, spot in enumerate(KNIGHT_BEDS)),
        bed("squire_bed", (104.45, 0.0, 112.0), facing=90.0),
        PropDef("keep_armour_rack", (105.0, 0, 106.0), (105.5, 2.2, 110.0), texture="shelf"),
        PropDef("keep_hearth", (99.5, 0, 104.6), (102.5, 1.3, 105.6), texture="stone", kind="oven", facing=180.0),
        PropDef("keep_banner_stain", (86.0, 1.0, 112.0), (86.0, 5.0, 115.0), texture="stain_decal", kind="stain", facing=270.0,
                solid=False),
        *(boulder(f"greyhold_rubble_{i}", x, z, r=0.9, h=0.9, texture="stone") for i, (x, z) in enumerate(
            [(114.0, 119.0), (117.5, 121.0), (78.0, 133.0), (108.0, 98.5)])),
        *barrels("keep_barrels", [(107.0, 127.5), (108.0, 127.5)]),
        *(oak(f"greyhold_oak_{i}", x, z, height=4.4) for i, (x, z) in enumerate([(64.0, 100.0), (60.0, 126.0), (122.0, 104.0)])),
        # the courtyard: the well, the practice yard, the banners of the Order
        PropDef("keep_well", (78.7, 0, 123.7), (81.3, 1.0, 126.3), texture="wall", kind="well"),
        dummy("keep_dummy_1", 80.0, 104.5), dummy("keep_dummy_2", 83.0, 104.5),
        banner("keep_banner_door_w", 92.0, 104.0, 180.0, "banner_greyhold", y=3.0, height=2.6),
        banner("keep_banner_door_e", 98.5, 104.0, 180.0, "banner_greyhold", y=3.0, height=2.6),
        banner("keep_banner_gate", kx1 - 1.5, 110.0, 270.0, "banner_greyhold", y=3.2, height=3.0, width=1.1),
        banner("keep_banner_gate_2", kx1 - 1.5, 120.0, 270.0, "banner_greyhold", y=3.2, height=3.0, width=1.1),
        banner("keep_banner_hall", 96.0, 117.7, 180.0, "banner_greyhold", y=2.6, height=2.6, width=1.2),
    ]
    p.containers += [
        table("keep_table", "Long Table", (97.0, 0.0, 111.3), owner="Greyhold", size=(3.0, 0.7),
              items=[ItemDef("roast_pheasant", "Roast Pheasant", value=14.0, stackable=True, quantity=12, is_food=True, category="food"),
                     ItemDef("wine", "Bottle of Wine", value=12.0, stackable=True, quantity=3, category="food"), BREAD(12)]),
        chest("greyhold_strongbox", "Greyhold Treasury", (96.0, 0.0, 124.8), owner="Greyhold", strongbox=True, locked=True,
              lock_difficulty=MASTER, key_id="treasury_door",
              items=[ItemDef("nightweave_cloak", "Nightweave Cloak", value=2800.0, category="clothing"),
                     ItemDef("duskwalker_boots", "Duskwalker Boots", value=950.0, category="clothing"), coins(400)]),
        chest("greyhold_tithe", "Tithe Chest", (101.0, 0.0, 124.8), owner="Greyhold", locked=True, lock_difficulty=STRONG,
              key_id="treasury_door", items=[coins(260), ItemDef("tithe_ledger", "The Tithe Ledger", value=5.0)]),
        chest("greyhold_armoury", "Armoury Chest", (104.8, 0.0, 114.6), owner="Greyhold", along_x=False, locked=True,
              lock_difficulty=STRONG, key_id="keep_door",
              items=[ItemDef("steel_sword", "Steel Sword", value=60.0, category="weapon"),
                     ItemDef("plate_cuirass", "Plate Cuirass", value=90.0, category="armor"), coins(60)]),
    ]
    p.lights += [
        wall_torch(kx1 - 1.5, 110.8, 270.0, "Greyhold"), wall_torch(kx1 - 1.5, 119.2, 270.0, "Greyhold"),
        wall_torch(93.5 - 0.6, 104.0, 0.0, "Greyhold"), wall_torch(96.5 + 0.6, 104.0, 0.0, "Greyhold"),
        LightDef("fire", (101.0, 0.9, 105.2), room="the keep hall", range=9.0),
    ]
    p.readables += [
        notice_board("greyhold_notice", "Notice at the Gate of Greyhold", 68.6, 112.0, (
            "GREYHOLD KEEP - TITHE-HOUSE OF THE CROWN for the March of Emberhollow.",
            "Here the tithes of Emberhollow and of Thornbury are held under the guard of the Order of the Grey until the King's Collector comes at Midsummer.",
            "No admittance to any not of the Order. Trespassers will be put out upon the road and fined, or worse.",
            "A reward of 200 crowns is offered for the outlaw GRIMMAW, once of this Order, struck from its rolls for theft. - Ser Caddock, Castellan."),
                     facing=270.0),
        plaque("greyhold_plaque", "Stone by the Keep Door", 91.2, 104.0, 0.0, (
            "RAISED IN THE YEAR OF THE LONG WINTER BY ALDOUS THE GREY, TO HOLD THE MARCH AGAINST THE HILL-CLANS.",
            "HERE WAS TAKEN FROM MORWEN THE THIEF-QUEEN THE CLOAK CALLED NIGHTWEAVE, AND HERE IT IS KEPT.",
            "WHAT GREYHOLD HOLDS, GREYHOLD KEEPS."), y=1.4, width=0.9),
    ]
    p.areas += [
        AreaDef("the Greyhold treasury", (86.0, 117.7), (106.0, 126.0), owner="Greyhold", restricted=True),
        AreaDef("the keep hall", (86.0, 104.0), (106.0, 117.7), owner="Greyhold", restricted=True),
        AreaDef("Greyhold", (kx1, kz1), (kx2, kz2), owner="Greyhold"),
    ]
    knight_look = Outfit(tunic=(0.22, 0.3, 0.55), legs=(0.2, 0.2, 0.26), armor=True, hat="helmet", hat_color=(0.62, 0.64, 0.7))
    squire_look = Outfit(tunic=(0.48, 0.5, 0.56), legs=(0.26, 0.26, 0.3), hair=(0.55, 0.38, 0.2))
    domain = ("the keep hall", "the Greyhold treasury")
    order = dict(faction="Greyhold", warden=domain, warden_exit=GREYHOLD_EXIT, inventory_reset_days=2)
    p.npcs += [
        NPCDef("Ser Caddock", KNIGHT_BEDS[0], role="knight", gold=60.0, outfit=knight_look, **order,
               inventory=(key_item("keep_door", "Keep Key"), key_item("treasury_door", "Treasury Key")),
               lines=("Greyhold keeps its own peace, traveller. Keep it with us.",
                      "The Collector comes at midsummer. Until then every crown of the tithe is my neck on the block.",
                      "Grimmaw stood where you stand once, in the grey. Took an oath. Broke it over a lockbox. Never trust a man who counts other men's coin."),
               schedule=(
                   TaskDef(0, 6, KNIGHT_BEDS[0], AIState.SLEEPING, "Ser Caddock is asleep in the keep.", lock_doors=("keep_door",)),
                   TaskDef(6, 8, KEEP_TABLE_SEATS[0], AIState.EATING, "Ser Caddock is breaking his fast.", facing=180.0,
                           food_container="keep_table", unlock_doors=("keep_door",)),
                   TaskDef(8, 12, COURTYARD_ROUTE[0], AIState.PATROL, "Ser Caddock is walking the walls of Greyhold.",
                           patrol_points=COURTYARD_ROUTE),
                   TaskDef(12, 14, (96.0, 0.0, 121.5), AIState.WORKING, "Ser Caddock is counting the tithe in the treasury.",
                           facing=180.0),
                   TaskDef(14, 20, COURTYARD_ROUTE[2], AIState.PATROL, "Ser Caddock is walking the walls.",
                           patrol_points=COURTYARD_ROUTE[2:] + COURTYARD_ROUTE[:2]),
                   TaskDef(20, 22, KEEP_TABLE_SEATS[0], AIState.EATING, "Ser Caddock is at supper in the hall.", facing=180.0,
                           food_container="keep_table"),
                   TaskDef(22, 24, KNIGHT_BEDS[0], AIState.SLEEPING, "Ser Caddock turned in.", lock_doors=("keep_door",)),
               )),
        NPCDef("Dame Ilsa", KNIGHT_BEDS[1], role="knight", gold=55.0, outfit=knight_look, **order,
               inventory=(key_item("keep_door", "Keep Key"),),
               lines=("State your business at the keep.",
                      "I keep the gate at night. The sea wind off the Saltcove carries voices. Fishermen, Caddock says. Fishermen don't whisper."),
               schedule=(
                   *span((18, 6), AIState.GUARDING, "Dame Ilsa is keeping the gateway.", GATEWAY_POST, facing=270.0),
                   TaskDef(6, 8, KEEP_TABLE_SEATS[1], AIState.EATING, "Dame Ilsa is eating in the hall.", facing=0.0,
                           food_container="keep_table"),
                   TaskDef(8, 16, KNIGHT_BEDS[1], AIState.SLEEPING, "Dame Ilsa is asleep in the keep."),
                   TaskDef(16, 18, COURTYARD_ROUTE[2], AIState.PATROL, "Dame Ilsa is walking the walls.",
                           patrol_points=COURTYARD_ROUTE[2:] + COURTYARD_ROUTE[:2]),
               )),
        NPCDef("Ser Aldwin", KNIGHT_BEDS[2], role="knight", gold=50.0, outfit=knight_look, **order,
               inventory=(key_item("keep_door", "Keep Key"),),
               lines=("Mind where you tread - the old walls are coming down.",
                      "Aldous the Grey built this keep with his own hands, they say. He must have had a great many hands."),
               schedule=(
                   TaskDef(0, 4, COURTYARD_ROUTE[1], AIState.PATROL, "Ser Aldwin is walking the walls by night.",
                           patrol_points=COURTYARD_ROUTE[1:] + COURTYARD_ROUTE[:1]),
                   TaskDef(4, 12, KNIGHT_BEDS[2], AIState.SLEEPING, "Ser Aldwin is asleep."),
                   TaskDef(12, 13, KEEP_TABLE_SEATS[2], AIState.EATING, "Ser Aldwin is eating.", facing=180.0, food_container="keep_table"),
                   TaskDef(13, 18, GATEWAY_POST, AIState.GUARDING, "Ser Aldwin is keeping the gateway.", facing=270.0),
                   TaskDef(18, 20, (81.5, 0.0, 106.5), AIState.WORKING, "Ser Aldwin is at sword practice in the yard.", facing=0.0),
                   TaskDef(20, 21, KEEP_TABLE_SEATS[2], AIState.EATING, "Ser Aldwin is at supper.", facing=180.0,
                           food_container="keep_table"),
                   TaskDef(21, 24, COURTYARD_ROUTE[0], AIState.PATROL, "Ser Aldwin is walking the walls.",
                           patrol_points=COURTYARD_ROUTE),
               )),
        NPCDef("Tobin the Squire", (104.45, 0.0, 112.0), role="squire", gold=6.0, outfit=squire_look, faction="Greyhold",
               inventory_reset_days=2, inventory=(key_item("keep_door", "Keep Key"),),
               lines=("I'm to be knighted at midsummer! If I polish enough armour. I've polished a very great deal of armour.",
                      "Ser Caddock says I mustn't talk to strangers about the tithe. So I won't. It's in the treasury, behind the big door. Oh."),
               schedule=(
                   TaskDef(0, 5, (104.45, 0.0, 112.0), AIState.SLEEPING, "Tobin is asleep by the armour rack."),
                   TaskDef(5, 7, (80.0, 0.0, 122.6), AIState.WORKING, "Tobin is drawing water at the courtyard well.", facing=180.0),
                   TaskDef(7, 8, (95.0, 0.0, 113.0), AIState.EATING, "Tobin is eating the knights' leavings.", facing=0.0,
                           food_container="keep_table"),
                   TaskDef(8, 12, (104.2, 0.0, 108.0), AIState.WORKING, "Tobin is polishing armour.", facing=90.0),
                   TaskDef(12, 15, (81.5, 0.0, 106.4), AIState.WORKING, "Tobin is thrashing the practice dummies.", facing=0.0),
                   TaskDef(15, 17, FIELD_EAST[0], AIState.STROLLING, "Tobin is running an errand to Thornbury.",
                           patrol_points=(FIELD_EAST[0], (15.0, 0.0, 72.0), (46.5, 0.0, 76.0))),
                   TaskDef(17, 20, (104.2, 0.0, 108.0), AIState.WORKING, "Tobin is polishing armour. Again.", facing=90.0),
                   TaskDef(20, 21, (95.0, 0.0, 113.0), AIState.EATING, "Tobin is having his supper.", facing=0.0,
                           food_container="keep_table"),
                   TaskDef(21, 24, (104.45, 0.0, 112.0), AIState.SLEEPING, "Tobin is asleep by the armour rack."),
               )),
    ]
    p.roads += [RoadDef((73.2, 111.5), (84.5, 118.5), texture="cobble_cracked")]                    # the courtyard's flags
    return p


# ------------------------------------------------------------------------------- the shrine
def shrine() -> Part:
    """The ruined shrine of Saint Wendel at the end of the south road, where the Deepwood begins:
    broken walls, an altar, offerings nobody has dared to take - and a pilgrim who never went home."""
    p = Part()
    sx, sz = SHRINE
    p.walls += [
        WallDef((sx - 5.0, 0, sz - 4.0), (sx - 2.0, 2.6, sz - 3.4), texture="stone"),
        WallDef((sx + 2.0, 0, sz - 4.0), (sx + 5.0, 1.6, sz - 3.4), texture="stone"),
        WallDef((sx - 5.0, 0, sz - 3.4), (sx - 4.4, 3.2, sz + 4.0), texture="stone"),
        WallDef((sx + 4.4, 0, sz - 3.4), (sx + 5.0, 1.2, sz + 1.0), texture="wall_cracked"),
        WallDef((sx - 4.4, 0, sz + 3.4), (sx + 1.0, 2.2, sz + 4.0), texture="stone"),
    ]
    p.props += [
        PropDef("shrine_altar", (sx - 1.4, 0, sz + 2.0), (sx + 1.4, 1.0, sz + 3.0), texture="marble"),
        *(boulder(f"shrine_rubble_{i}", x, z, r=r, h=r, texture="stone") for i, (x, z, r) in enumerate(
            [(sx + 3.6, sz + 3.0, 0.7), (sx + 5.8, sz + 2.0, 0.9), (sx - 1.0, sz - 5.2, 0.6)])),
    ]
    p.containers += [
        ContainerDef("shrine_offerings", "Offering Bowl", (sx - 0.4, 1.0, sz + 2.2), (sx + 0.4, 1.2, sz + 2.8), owner="Saint Wendel",
                     texture="stone", crime_name="Theft", items=(coins(23), ItemDef("prayer_beads", "Prayer Beads", value=15.0))),
        skeleton("shrine_pilgrim", "Bones of a Pilgrim", (sx + 2.6, 0.0, sz - 0.5), items=[coins(12)]),
    ]
    p.readables += [
        plaque("shrine_inscription", "Words Cut in the Altar", sx, sz + 2.0, 180.0, (
            "SAINT WENDEL WALKED THE DEEPWOOD FORTY DAYS AND CAME OUT WITH NOTHING, AND WAS CONTENT.",
            "TAKE NOTHING FROM THIS PLACE BUT THE PEACE OF IT."), y=0.3, width=1.4),
        journal("shrine_pilgrim_pages", "A Pilgrim's Prayer", sx + 3.2, 0.05, sz - 1.6, (
            "Saint Wendel, I walked from the coast with nothing but my sins and twelve crowns for the bowl.",
            "I have the cough my father had. Let me rest here a while before I start back."), facing=0.0),
    ]
    p.areas += [AreaDef("the old shrine", (sx - 6.0, sz - 6.0), (sx + 6.0, sz + 6.0))]
    return p


# =============================================================================== trees and the edges of the world
def _near_polyline(x: float, z: float, points: Sequence[Tuple[float, float]], reach: float) -> bool:
    for (x1, z1), (x2, z2) in zip(points, points[1:]):
        sx, sz = x2 - x1, z2 - z1
        t = max(0.0, min(1.0, ((x - x1) * sx + (z - z1) * sz) / max(sx * sx + sz * sz, 1e-9)))
        if math.hypot(x - (x1 + t * sx), z - (z1 + t * sz)) < reach:
            return True
    return False


def _slope(x: float, z: float) -> float:
    d = 1.0
    gx = (ground(x + d, z) - ground(x - d, z)) / (2 * d)
    gz = (ground(x, z + d) - ground(x, z - d)) / (2 * d)
    return math.hypot(gx, gz)


WEST_EDGE = BOUNDS[0] + 4.0          # the thicket the Thornwood becomes: nobody gets further
SOUTH_EDGE = BOUNDS[3] - 4.0         # ... nor through the Deepwood to the south


def forest(keep_out: Sequence[Tuple[float, float, float, float]], roads: Sequence[RoadDef]) -> List[PropDef]:
    """Trees (pines mostly, oaks at the edges), bushes and boulders over the Thornwood, the
    country and the foot of the mountain - deterministically, clear of the ways, the buildings,
    the water, the sand and the steeps. Then the edges of the world: a wall of thicket where the
    Thornwood and the Deepwood close in, and the wood carrying on, dense, as far as the eye goes."""
    rng = random.Random(1789)
    props: List[PropDef] = []
    blocked = list(keep_out) + [(r.min_xz[0] - 2.5, r.min_xz[1] - 2.5, r.max_xz[0] + 2.5, r.max_xz[1] + 2.5) for r in roads]
    lake = LAKES[0]
    stream = ((-42.0, -61.5), (-53.0, 53.5)) + MILLSTREAM_OUTSIDE
    ways = [(trail.points, trail.width / 2.0 + trail.shoulder + 1.2) for trail in TRAILS]

    def clear(x: float, z: float, margin: float = 0.0) -> bool:
        if any(a - margin <= x <= c + margin and b - margin <= z <= d + margin for a, b, c, d in blocked):
            return False
        if ((x - lake.center[0]) / (lake.radius_x * 1.45)) ** 2 + ((z - lake.center[1]) / (lake.radius_z * 1.45)) ** 2 <= 1.0:
            return False
        if _near_polyline(x, z, stream, 6.0) or any(_near_polyline(x, z, pts, reach) for pts, reach in ways):
            return False
        if LAND.sea_distance(x, z) > -SEA.beach - 3.0:
            return False                                   # (the beach and the sea)
        return _slope(x, z) < 0.85                         # (nothing roots on the bare rock)

    x1, z1, x2, z2 = BOUNDS
    regions = (
        ((x1 + 6, -70.0, -88.0, z2 - 10), 8.5, 0.72, 0.75),     # the Thornwood: dense, mostly pine
        ((-84.0, -72.0, 126.0, -64.5), 9.0, 0.5, 0.8),          # the foot of the mountain behind the town
        ((84.0, -62.0, 130.0, 90.0), 11.0, 0.45, 0.3),          # open country east of the town
        ((-84.0, 124.0, 66.0, z2 - 10), 9.0, 0.6, 0.5),         # the Deepwood, south beyond the fields
        ((50.0, 60.0, 124.0, 94.0), 12.0, 0.35, 0.3),
    )
    n = 0
    for (rx1, rz1, rx2, rz2), step, density, pine_share in regions:
        z = rz1
        while z <= rz2:
            x = rx1
            while x <= rx2:
                px = min(rx2, max(rx1, x + rng.uniform(-step * 0.35, step * 0.35)))
                pz = min(rz2, max(rz1, z + rng.uniform(-step * 0.35, step * 0.35)))
                roll = rng.random()
                if roll < density and clear(px, pz, 1.5):
                    height = rng.uniform(3.6, 5.4)
                    if rng.random() < pine_share:
                        props.append(pine(f"wood_pine_{n}", px, pz, height=height + 0.6))
                    else:
                        props.append(oak(f"wood_oak_{n}", px, pz, height=height - 0.4))
                    n += 1
                elif roll < density + 0.12 and clear(px, pz, 1.0):
                    props.append(bush(f"wood_bush_{n}", px, pz, r=rng.uniform(0.6, 1.0)))
                    n += 1
                elif roll < density + 0.16 and clear(px, pz, 1.5):
                    props.append(boulder(f"wood_boulder_{n}", px, pz, r=rng.uniform(0.7, 1.4), h=rng.uniform(0.7, 1.4)))
                    n += 1
                x += step
            z += step
    props += _edge_of_the_wood(rng)
    return props


def _edge_of_the_wood(rng: random.Random) -> List[PropDef]:
    """Where the woods close in: a wall of bramble thicket along the west and the south (each
    clump's box meets the next: nobody squeezes through), and beyond it old trees crowded
    trunk to trunk out past the edge of the map, so the wood goes on as far as anyone can see."""
    props: List[PropDef] = []
    x1, z1, x2, z2 = BOUNDS
    coast_south = float(LAND.coast_x(z2))

    def thicket(name, x, z):
        r = rng.uniform(0.95, 1.2)
        props.append(PropDef(name, (x - r, 0.0, z - r), (x + r, rng.uniform(1.6, 2.3), z + r), texture="leaves", kind="bush"))

    z = z1 + 8.0
    i = 0
    while z <= z2:
        thicket(f"thicket_w_{i}", WEST_EDGE + rng.uniform(-0.3, 0.3), z)
        z += 1.75
        i += 1
    x = x1
    i = 0
    while x <= coast_south - SEA.beach:
        thicket(f"thicket_s_{i}", x, SOUTH_EDGE + rng.uniform(-0.3, 0.3))
        x += 1.75
        i += 1
    # beyond the thicket: close-set trees for a dozen paces, then the wood thins into the haze
    n = 0
    bands = (
        # (x1, z1, x2, z2, spacing)
        (WEST_EDGE - 13.5, z1 - 10, WEST_EDGE - 1.5, z2 + 4.0, 4.0),
        (x1 - TERRAIN_SKIRT + 2, z1 - 10, WEST_EDGE - 13.5, z2 + TERRAIN_SKIRT - 2, 6.5),
        (x1 - 13.5, SOUTH_EDGE + 1.5, coast_south - SEA.beach - 4.0, SOUTH_EDGE + 13.5, 4.0),
        (WEST_EDGE - 13.5, SOUTH_EDGE + 13.5, coast_south - SEA.beach - 4.0, z2 + TERRAIN_SKIRT - 2, 6.5),
    )
    for bx1, bz1, bx2, bz2, step in bands:
        zz = bz1
        while zz <= bz2:
            xx = bx1
            while xx <= bx2:
                px, pz = xx + rng.uniform(-1.2, 1.2), zz + rng.uniform(-1.2, 1.2)
                if _slope(px, pz) < 0.9:
                    props.append(pine(f"deep_wood_{n}", px, pz, height=rng.uniform(4.6, 6.8)))
                    n += 1
                xx += step
            zz += step
    return props


def coast() -> Part:
    """The shore: driftwood, rocks, a fisherman's old upturned boat."""
    p = Part()
    p.props += [
        *(boulder(f"shore_rock_{i}", x, z, r=r, h=r, texture="cliff") for i, (x, z, r) in enumerate(
            [(140.0, 64.0, 1.4), (138.5, 96.0, 1.0), (133.0, 128.0, 1.6), (143.0, -20.0, 1.8), (145.0, -44.0, 1.2),
             (141.0, 8.0, 1.1)])),
        *(PropDef(f"driftwood_{i}", (x - 1.6, 0.0, z - 0.2), (x + 1.6, 0.4, z + 0.2), texture="bark", kind="woodpile", facing=90.0)
          for i, (x, z) in enumerate([(137.0, 58.0), (135.5, 84.0), (138.0, -5.0)])),
        rowboat("shore_wreck", 136.0, 74.0, facing=30.0),
    ]
    return p


# =============================================================================== everything, merged
_parts = [village(), thornwood(), howling_deep().clear_cave_floors(), saltcove().clear_cave_floors(),
          old_mine().clear_cave_floors(), greyhold(), shrine(), coast()]


def _all(attr):
    return tuple(item for part in _parts for item in getattr(part, attr))


WALLS: Tuple[WallDef, ...] = _all("walls")
BUILDINGS: Tuple[BuildingDef, ...] = _all("buildings")
DOORS: Tuple[DoorDef, ...] = _all("doors")
CONTAINERS: Tuple[ContainerDef, ...] = _all("containers")
LIGHTS: Tuple[LightDef, ...] = _all("lights")
STATIONS: Tuple[StationDef, ...] = _all("stations")
READABLES: Tuple[ReadableDef, ...] = _all("readables")
HOLES: Tuple[Tuple[float, float, float, float], ...] = _all("holes")
AREAS: Tuple[AreaDef, ...] = _all("areas") + (
    AreaDef("Thornbury", (-50.0, 56.0), (50.0, 124.0)),
    AreaDef("the lake shore", (-126.0, 132.0), (-58.0, 182.0)),
    AreaDef("the beach", (126.0, -95.0), (175.0, 212.0)),
    AreaDef("the Thornwood", (BOUNDS[0], -75.0), (-86.0, BOUNDS[3])),
)
# (the town's own streets come before these in the level's list - see town.py)
COUNTRYSIDE = AreaDef("the countryside", (BOUNDS[0], BOUNDS[1]), (BOUNDS[2], BOUNDS[3]))
NPCS: Tuple[NPCDef, ...] = _all("npcs")
ROADS: Tuple[RoadDef, ...] = _all("roads")
STAIRS: tuple = ()
FLATS = (
    FlatDef((-67.0, 66.5), (-53.0, 77.0), name="lane_bridge"),     # where the village lane crosses the millstream
    FlatDef((SHRINE[0] - 7.0, SHRINE[1] - 7.0), (SHRINE[0] + 7.0, SHRINE[1] + 7.0), name="shrine"),
)
_KEEP_OUT = (
    (-82.0, -64.0, 82.0, 56.0),                  # the town and its wall
    *((b.min_xz[0] - 5.0, b.min_xz[1] - 5.0, b.max_xz[0] + 5.0, b.max_xz[1] + 5.0) for b in BUILDINGS if not b.underground),
    *((b.min_xz[0] - 3.0, b.min_xz[1] - 3.0, b.max_xz[0] + 3.0, b.max_xz[1] + 3.0) for b in BUILDINGS if b.underground),
    (70.0, 94.0, 118.0, 138.0),                  # Greyhold
    (-70.0, 62.0, -50.0, 80.0),                  # the lane bridge
    (-164.0, 4.0, -140.0, 36.0),                 # the hermit's clearing
    (SHRINE[0] - 9.0, SHRINE[1] - 9.0, SHRINE[0] + 9.0, SHRINE[1] + 9.0),
    *((s.min_point[0] - 3.0, s.min_point[2] - 3.0, s.max_point[0] + 3.0, s.max_point[2] + 3.0) for s in STATIONS),
    *((r.min_point[0] - 2.0, r.min_point[2] - 2.0, r.max_point[0] + 2.0, r.max_point[2] + 2.0) for r in READABLES),
    *((x - 3.0, z - 3.0, x + 3.0, z + 3.0) for x, _, z in HUNTING_ROUTE),     # the hunter's stands
)
PROPS: Tuple[PropDef, ...] = _all("props") + tuple(forest(_KEEP_OUT, ROADS))
