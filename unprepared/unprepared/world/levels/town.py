"""Emberhollow: a walled market town on the south face of a hill, the watchtower at its peak.

    x -80 ......................................................................... 80
    z -62  [north wall]        (the millstream enters under a grate at x -42)
           orchard          Ashgrove Manor   park   Vane Manor            guard house + watchtower (the peak)
                      ====================== Upper Street (z -42 .. -36) =================
                            North Lane (x -3..3)                        East Lane (x 34..40)
      smithy shed  (stream) The Drowsy Dragon | store |      | tailor | old granary |  cooper's yard
    =========================== Market Street (z -6 .. 6), the well at 0,0 ===========================
      Barnaby | Tamsin (stream) Fenwick | hideout (Jae) | Gate Road | Lys & Dorran | Pell |     jail
                               chapel   ====== South Lane (z 27..32) ======   stable
    z  54  [south wall] ................. the town gate (x -4..4) ................................

The ground rises from the gate (about level with the fields outside) up to the watchtower,
so every building stands on a levelled pad (``terrain``): floors are flat, streets are not.
The town gate is shut and barred until the tutorial is over; beyond it lie the village, the
woods, the caves and the old keep (``wilds.py``).

Every door hinge sits on the building wall that faces its street; doors in a north-side
building swing into it, doors in a south-side building out onto the street (``Door`` leaves
span +x from the hinge - or -x for a ``hinge_right`` leaf - and fold into -z). All yards are
measured so that nothing overlaps: walls are 0.5 thick and every piece of furniture stands
clear of them. Named spots used by the schedules are module constants so tests and scenarios
can refer to them (their y is the floor they stand on: 0 downstairs, 4 on an upper storey).
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

from ...entities import AIState, DoorKind, Outfit
from ..builders import (WALL, barrels, bed, bridge, bush, chest, coins, crack, fence, grate, haystack, house_walls, stash,
                        key_item, oak, pine, resident_tasks, span, stain, stall, stroll, table, torches_beside_door, visit,
                        wall_torch, woodpile)
from ..level import (AreaDef, BuildingDef, ContainerDef, DoorDef, InnDef, InnRoomDef, ItemDef, JailDef, LevelDefinition,
                     LightDef, NPCDef, PropDef, ReadableDef, RoadDef, StationDef, StockTier, TaskDef, WallDef)
from ..terrain import FlatDef, StreamDef, TerrainDef
from ..tower import build_tower
from . import manors, quarters, stories, wilds

STREET_NORTH, STREET_SOUTH = -6.0, 6.0
TOWN_X1, TOWN_Z1, TOWN_X2, TOWN_Z2 = -80.0, -62.0, 80.0, 54.0      # outer faces of the town wall
TOWN_WALL, TOWN_WALL_HEIGHT = 1.0, 5.0

# --- named spots ------------------------------------------------------------------------------
WELL = (0.0, 0.0, 0.0)
PLAYER_SPAWN = (-4.5, 0.0, 4.0)        # on the well square, looking north up the street
# general store (north side, across the street from the hideout)
SHOP_COUNTER = (-13.0, 0.0, -12.2)
SHOP_FLOOR = (-13.0, 0.0, -8.4)        # where a customer stands to be served
SHOP_BACKROOM = (-9.4, 0.0, -12.85)    # beside the back-room table
SHOP_PORTER = (-8.0, 0.0, -9.0)        # Dorran's spot as he fetches and carries
# inn
INN_BAR = (-36.0, 0.0, -12.8)          # Hilde behind the bar
INN_KITCHEN = (-35.9, 0.0, -14.0)      # between the kitchen table and the bar
INN_TABLES = ((-29.25, 0.0, -14.0), (-29.25, 0.0, -8.5), (-24.75, 0.0, -11.0), (-24.75, 0.0, -7.8), (-36.75, 0.0, -8.2))
# seats west of each table face +x (90), seats east of it face -x (270) (the regulars of quarters.py
# take the last four, by the door and by the hearth)
INN_SEATS = ((-30.6, 0.0, -14.0), (-30.6, 0.0, -8.5), (-26.1, 0.0, -11.0), (-27.9, 0.0, -14.0), (-27.9, 0.0, -8.5),
             (-23.4, 0.0, -11.0), (-26.1, 0.0, -7.8), (-23.4, 0.0, -7.8), (-38.1, 0.0, -8.2), (-35.4, 0.0, -8.2))
INN_KEEPER_BED = (-37.75, 0.0, -20.0)
INN_ROOM_BEDS = ((-33.25, 0.0, -20.0), (-28.75, 0.0, -20.0), (-24.25, 0.0, -20.0))
# smithy
ANVIL_SPOT = (-62.0, 0.0, -7.8)
FORGE_SPOT = (-60.6, 0.0, -8.9)        # at the forge, the coals in front of him
SMITH_BED = (-72.6, 0.0, -13.3)
SMITH_TABLE_SEAT = (-68.2, 0.0, -8.3)
ORE_DEPOSIT = (-62.5, 0.0, -19.5)
# tailor
TAILOR_COUNTER = (11.5, 0.0, -11.1)
TAILOR_FLOOR = (11.5, 0.0, -8.0)
TAILOR_BED = (15.25, 0.0, -12.4)
TAILOR_TABLE_SEAT = (12.6, 0.0, -7.6)
# homes (south side): beds head against a wall (see ``builders.bed``)
BARNABY_BED = (-67.3, 0.0, 13.4)
BARNABY_TABLE_SEAT = (-71.5, 0.0, 12.0)
TAMSIN_BED = (-60.7, 0.0, 11.4)
TAMSIN_TABLE_SEAT = (-56.25, 0.0, 10.2)
JAE_DEN = (-10.0, 0.0, 12.0)
JAE_BED = (-15.8, 0.0, 15.4)
JAE_TABLE_SEAT = (-14.0, 0.0, 11.0)
FENWICK_BED = (-36.8, 0.0, 11.4)
FENWICK_TABLE_SEAT = (-31.75, 0.0, 10.1)
FISHING_SPOT = (-46.5, 0.0, 16.0)
PARK_BENCH = (2.3, 0.0, -54.2)         # before the park's north bench      # on the millstream's bank, below the bridge
LYS_BED = (15.45, 0.0, 13.25)
DORRAN_BED = (15.45, 0.0, 10.25)
COUPLE_TABLE_SEATS = ((10.0, 0.0, 13.2), (11.8, 0.0, 11.6))      # Lys south of the table (faces -z), Dorran east (faces -x)
PELL_BED = (27.4, 0.0, 11.4)
PELL_TABLE_SEAT = (23.5, 0.0, 10.2)
# the watch: a barracks of six on Upper Street, the tower behind it at the top of the hill
GUARD_HOUSE = (42.0, -53.0, 58.0, -43.0)
GUARD_BEDS = ((43.55, 0.0, -51.6), (43.55, 0.0, -49.9), (43.55, 0.0, -48.2),      # along the west wall, heads west
              (56.45, 0.0, -51.6), (56.45, 0.0, -49.9), (56.45, 0.0, -48.2))      # along the east wall, heads east
GUARD_TABLE_SEATS = ((47.7, 0.0, -49.0), (52.3, 0.0, -49.0), (50.0, 0.0, -46.9))
POST_DOOR = (53.5, 0.0, -40.5)         # outside the guard house door, watching Upper Street and the manors
POST_WEST = (-20.5, 0.0, -4.0)         # by the inn's east corner, watching the west half of Market Street
TOWER_X1, TOWER_Z1, TOWER_X2, TOWER_Z2 = 48.0, -59.0, 54.0, -53.0   # the watchtower: behind the guard house, entered from it
TOWER_PLATFORM = 13.0
TOWER_DOOR_X = 50.0                    # the doorway 50..52 in the guard house's north wall
WATCHTOWER = build_tower("watchtower", TOWER_X1, TOWER_Z1, TOWER_X2, TOWER_Z2, TOWER_PLATFORM, TOWER_DOOR_X, room="the watchtower")
TOWER_TOP = WATCHTOWER.stair.top       # the look-out spot on the platform
# the jail (Market Street, east end)
JAIL_POST = (50.8, 0.0, 10.5)          # in the jail office, facing the cell gate
JAIL_GUARD_BED = (49.55, 0.0, 8.0)
JAIL_CELL_BED = (56.45, 0.0, 14.9)
JAIL_CELL_SPOT = (55.0, 0.0, 13.0)
# the town gate (south wall): two leaves, a guard either side
GATE_X1, GATE_X2 = -4.0, 4.0
GATE_Z = 53.5                          # hinge line, the middle of the wall
GATE_POSTS = ((-6.2, 0.0, 49.8), (6.2, 0.0, 49.8))
GATE_INSIDE = (0.0, 0.0, 46.0)         # where somebody turned away is sent back to
GATE_LODGE = (8.0, 45.5, 15.0, 52.5)    # the gate watch's lodge, just inside the wall east of the gate
GATE_LODGE_BEDS = ((13.45, 0.0, 47.1), (13.45, 0.0, 50.5))
GATE_LODGE_SEAT = (9.3, 0.0, 49.6)          # (west of the table, facing it)
GATE_OUTSIDE = (0.0, 0.0, 60.0)
# the manors (Upper Street, north side): two storeys, the family upstairs, a servant below (see manors.py)
ASHGROVE_X1, VANE_X1 = -34.0, 8.0
MANOR_Z1, MANOR_Z2, MANOR_W = -59.0, -43.0, manors.MANOR_W
PATROL_ROUTE = ((37.0, 0.0, -3.0), (37.0, 0.0, -30.0), (30.0, 0.0, -39.0), (0.0, 0.0, -39.0), (0.0, 0.0, -12.0),
                (-20.0, 0.0, -3.0), (-48.0, 0.0, -2.0), (-68.0, 0.0, 2.0), (-48.0, 0.0, 3.0), (-25.0, 0.0, 3.5),
                (-5.0, 0.0, 4.0), (0.0, 0.0, 20.0), (0.0, 0.0, 44.0), (5.0, 0.0, 4.0), (20.0, 0.0, 3.5), (37.0, 0.0, 3.5))
# where people stroll: a loop along the street with a stop at the well
STROLL_WEST = ((-62.0, 0.0, 2.0), (-40.0, 0.0, -2.5), (-20.0, 0.0, 3.0), (-3.0, 0.0, 2.4), (-22.0, 0.0, -3.0), (-45.0, 0.0, 3.5))
STROLL_EAST = ((30.0, 0.0, 2.0), (12.0, 0.0, -3.0), (2.4, 0.0, -2.4), (-4.0, 0.0, 3.0), (14.0, 0.0, 3.5), (36.0, 0.0, -2.0))
# where the rich take the air: Upper Street and the little park between their houses, looking down on the town
PROMENADE = ((-8.0, 0.0, -39.0), (4.0, 0.0, -47.0), (26.0, 0.0, -39.0), (0.0, 0.0, -30.0), (-28.0, 0.0, -39.0))

SHOP_OPEN_HOUR, SHOP_LUNCH_HOUR, SHOP_CLOSE_HOUR = 8, 13, 20
TAILOR_OPEN_HOUR, TAILOR_CLOSE_HOUR = 9, 18
INN_OPEN_HOUR, INN_CLOSE_HOUR = 5, 24
GUARD_GOLD, SHOPKEEPER_GOLD, FENCE_GOLD, INNKEEPER_GOLD, SMITH_GOLD, TAILOR_GOLD = 15.0, 400.0, 600.0, 250.0, 350.0, 300.0
TOWNSFOLK_GOLD = 30.0
RICH_GOLD = 260.0
STARTING_GOLD = 60.0             # the player's purse at the start (enough for a pickaxe and a mallet at asking price)

# lock tiers (``config.LOCK_TIER_NAMES``): what a door or a chest is worth protecting
FLIMSY, SIMPLE, STURDY, STRONG, MASTER, LEGENDARY = range(6)

GUARD_KEY = ItemDef("guard_key", "Guard House Key", value=1.0, is_key=True, key_door_id="guardhouse_door", for_sale=False,
                    category="key")
TOWER_KEY = ItemDef("tower_key", "Watchtower Key", value=1.0, is_key=True, key_door_id="tower_door", for_sale=False, category="key")
RATIONS = ItemDef("ration", "Guard's Ration", value=3.0, stackable=True, quantity=2, is_food=True, category="food")
GATE_LODGE_KEY = ItemDef("gate_lodge_door_key", "Gate Lodge Key", value=1.0, is_key=True, key_door_id="gate_lodge_door", for_sale=False,
                         category="key")

LORD_OUTFIT = Outfit(tunic=(0.36, 0.1, 0.28), legs=(0.16, 0.14, 0.2), body="doublet", neck="mantle", neck_color=(0.55, 0.08, 0.12),
                     hat="feathered_hat", hat_color=(0.2, 0.08, 0.26), hair=(0.55, 0.55, 0.55))
LADY_OUTFIT = Outfit(tunic=(0.18, 0.32, 0.55), legs=(0.18, 0.3, 0.5), body="gown", neck="choker", neck_color=(0.95, 0.94, 0.9),
                     hat="circlet", hat_color=(0.95, 0.8, 0.35), hair=(0.6, 0.45, 0.2))


def guard(name: str, patrol: tuple, meal: tuple, post: tuple, post_spot, sleep: tuple, bed_spot, meal_seat,
          post_facing: float = 270.0) -> NPCDef:
    """A guard with an 8-hour shift rotation."""
    tasks = []
    tasks += span(patrol, AIState.PATROL, f"{name} is patrolling the town.", PATROL_ROUTE[0],
                  patrol_points=PATROL_ROUTE, depart_text="Time for my rounds.")
    tasks += span(meal, AIState.EATING, f"{name} is eating at the guard house table.", meal_seat, facing=90.0,
                  food_container="guard_table")
    tasks += span(post, AIState.GUARDING, f"{name} is standing guard.", post_spot, facing=post_facing)
    tasks += span(sleep, AIState.SLEEPING, f"{name} is asleep in the guard house.", bed_spot)
    return NPCDef(name, bed_spot, is_guard=True, inventory=(GUARD_KEY, TOWER_KEY, RATIONS), schedule=tuple(tasks),
                  inventory_reset_days=1, gold=GUARD_GOLD, role="guard")


def gate_guard(name: str, duty: Tuple[int, int], first_post: int, meal: int, sleep: Tuple[int, int], bed_spot, meal_seat,
               evening: Tuple[int, int], supper: Optional[Tuple[int, int]] = None) -> NPCDef:
    """A gate warden: a long watch at the town gate, changing sides with the hours (the watch
    rotates: nobody stares at the same stretch of road all day), a meal, an evening off and bed."""
    tasks: List[TaskDef] = []
    start, end = duty
    hours = [h % 24 for h in range(start, start + ((end - start) % 24 or 24))]
    side = first_post
    for i, hour in enumerate(hours):
        if i and i % 3 == 0:
            side = 1 - side
        tasks.append(TaskDef(hour, hour + 1, GATE_POSTS[side], AIState.GUARDING, f"{name} is keeping the town gate.",
                             facing=180.0, depart_text="Right. The gate." if i == 0 else None))
    tasks.append(TaskDef(meal, meal + 1, meal_seat, AIState.EATING, f"{name} is eating in the gate lodge.", facing=90.0,
                         food_container="gate_lodge_table"))
    tasks += span(evening, AIState.DRINKING, f"{name} is off duty at the Drowsy Dragon.", INN_SEATS[5], facing=270.0)
    if supper is not None:                    # (back from the inn in good time for the watch: it is a long walk)
        tasks += span(supper, AIState.EATING, f"{name} is having supper in the gate lodge before the watch.", meal_seat,
                      facing=90.0, food_container="gate_lodge_table")
    tasks += span(sleep, AIState.SLEEPING, f"{name} is asleep in the gate lodge.", bed_spot)
    return NPCDef(name, bed_spot, is_guard=True, inventory=(GUARD_KEY, GATE_LODGE_KEY, RATIONS),
                  schedule=tuple(sorted(tasks, key=lambda t: t.start_hour)), inventory_reset_days=1, gold=GUARD_GOLD, role="guard")


# --- the manors --------------------------------------------------------------------------------
def rich_resident(name: str, prefix: str, m: dict, outfit: Outfit, inventory, day: List[TaskDef], lines=(),
                  other_days: Tuple[List[TaskDef], ...] = ()) -> NPCDef:
    def plan(tasks):
        return resident_tasks(name.split()[-1], f"{prefix}_door", m["bed"], m["seat"], f"{prefix}_dining_table", tasks,
                              wake=8, bedtime=22)
    return NPCDef(name, m["bed"], role="rich", gold=RICH_GOLD, inventory_reset_days=3, outfit=outfit, inventory=tuple(inventory),
                  lines=tuple(lines), schedule=plan(day), rota=tuple(plan(d) for d in other_days))


BUTLER_OUTFIT = Outfit(tunic=(0.1, 0.1, 0.13), legs=(0.08, 0.08, 0.1), sash=(0.85, 0.85, 0.88), hair=(0.7, 0.7, 0.72))
MAID_OUTFIT = Outfit(tunic=(0.22, 0.24, 0.36), legs=(0.2, 0.2, 0.3), apron="cloth", hat="coif", hat_color=(0.96, 0.96, 0.96),
                     hair=(0.55, 0.3, 0.14))


def servant(name: str, role: str, prefix: str, m: dict, outfit: Outfit, errands: List[TaskDef], lines=(), extra=(),
            other_errands: Tuple[List[TaskDef], ...] = ()) -> NPCDef:
    """The household's servant: up first and to bed last, the range, the silver, the serving,
    the dusting upstairs - and an errand or two in the town between (a morning one from 09:00,
    the silver waiting until it is done; an evening one 17:00 - 19:00). ``other_errands``: the
    errands of the other days of the servant's round."""
    first = name.split()[0]
    sideboard = (m["seat"][0] - 1.6, 0.0, m["seat"][2] + 1.2)      # (by the table, clear of the fireplace)
    upstairs = (m["study"], m["landing"], (m["bed"][0] - 2.6, m["bed"][1], m["bed"][2] + 2.8), m["dressing"])

    def plan(jobs):
        back = jobs[0].end_hour if jobs else 9
        serve = 13 if back >= 11 else 12          # (back late from a long errand: the silver first, then the table)
        day = [
            TaskDef(7, 9, m["kitchen"], AIState.WORKING, f"{first} is at the range, cooking for the family.", facing=0.0,
                    depart_text="The family will want their breakfast."),
            *jobs[:1],
            TaskDef(back, serve, m["store"], AIState.WORKING, f"{first} is polishing the silver in the store room.", facing=270.0),
            TaskDef(serve, 14, sideboard, AIState.WORKING, f"{first} is waiting at table.", facing=90.0),
            TaskDef(14, 17, upstairs[0], AIState.CLEANING, f"{first} is dusting upstairs.", patrol_points=upstairs),
            *jobs[1:2],
            TaskDef(19, 21, m["kitchen"], AIState.WORKING, f"{first} is cooking supper.", facing=0.0),
        ]
        return resident_tasks(first, f"{prefix}_door", m["servant_bed"], m["servant_seat"], f"{prefix}_kitchen_table", day,
                              wake=6, bedtime=22, facing=180.0)
    return NPCDef(name, m["servant_bed"], role=role, gold=TOWNSFOLK_GOLD, inventory_reset_days=3, outfit=outfit,
                  inventory=(key_item(f"{prefix}_door", "Servants' Key"), key_item(f"{prefix}_servant_door", f"{first}'s Room Key"),
                             *extra), lines=tuple(lines),
                  schedule=plan(errands), rota=tuple(plan(jobs) for jobs in other_errands))


# --- the level --------------------------------------------------------------------------------
def town_walls() -> List[WallDef]:
    """The curtain wall: four sides, culverts where the millstream runs under it (an arch over
    the water, a grate below) and the gatehouse - two squat towers either side of the gate with
    an arch over it. Some stretches are cracked with age."""
    t, h = TOWN_WALL, TOWN_WALL_HEIGHT
    x1, z1, x2, z2 = TOWN_X1, TOWN_Z1, TOWN_X2, TOWN_Z2
    walls: List[WallDef] = []

    def run_x(za, zb, xa, xb, gaps):
        """A wall along x from xa to xb with ``gaps`` [(g1, g2, arch_bottom or None)]."""
        x = xa
        for i, (g1, g2, arch) in enumerate(gaps):
            if g1 > x:
                walls.append(WallDef((x, 0, za), (g1, h, zb), texture="wall_cracked" if i % 2 else "wall"))
            if arch is not None:
                walls.append(WallDef((g1, arch, za), (g2, h, zb), texture="wall"))
            x = g2
        if xb > x:
            walls.append(WallDef((x, 0, za), (xb, h, zb), texture="wall"))

    run_x(z1, z1 + t, x1, x2, [(-44.0, -40.0, 1.7), (10.0, 22.0, None), (22.0, 22.0, None)])
    walls.append(WallDef((10.0, 0, z1), (22.0, h, z1 + t), texture="wall_cracked"))          # an old breach, patched
    run_x(z2 - t, z2, x1, x2, [(-55.0, -51.0, 1.7), (-7.0, 7.0, None)])
    walls += [WallDef((x1, 0, z1 + t), (x1 + t, h, -20.0), texture="wall"),
              WallDef((x1, 0, -20.0), (x1 + t, h, 10.0), texture="wall_cracked"),
              WallDef((x1, 0, 10.0), (x1 + t, h, z2 - t), texture="wall"),
              WallDef((x2 - t, 0, z1 + t), (x2, h, 0.0), texture="wall"),
              WallDef((x2 - t, 0, 0.0), (x2, h, 25.0), texture="wall_cracked"),
              WallDef((x2 - t, 0, 25.0), (x2, h, z2 - t), texture="wall")]
    # the gatehouse: two towers and the arch between them over the leaves
    walls += [WallDef((-7.0, 0, z2 - 2.5), (GATE_X1, 7.0, z2 + 1.5), texture="wall"),
              WallDef((GATE_X2, 0, z2 - 2.5), (7.0, 7.0, z2 + 1.5), texture="wall"),
              WallDef((GATE_X1, 4.6, z2 - t), (GATE_X2, 7.0, z2), texture="wall")]
    return walls


def town_terrain() -> TerrainDef:
    """The lie of the land (``wilds.BASE``: the town's hill, the mountain behind it, the sea) with
    the town's own level places, the millstream and everything the realm outside adds to it."""
    return dataclasses.replace(
        wilds.BASE,
        flats=(
            FlatDef((-11.0, -8.5), (11.0, 8.5), name="square"),            # the well square (and the houses round it)
            FlatDef((-10.0, 45.0), (10.0, 64.0), name="gate"),             # inside and outside the gate
            FlatDef((-52.5, -7.0), (-42.5, 7.0), name="bridge"),           # the bridge over the millstream
            FlatDef((-6.5, -56.0), (4.5, -46.0), name="park"),             # the park's lawn round the fountain
            *wilds.FLATS,
        ),
        streams=(StreamDef("millstream", MILLSTREAM, width=2.4, bank=1.8, bank_drop=0.9, flow=0.7),),
        lakes=wilds.LAKES,
        trails=wilds.TRAILS,
        holes=wilds.HOLES,
    )


# the millstream: in under the north wall, down the west of the town, out under the south wall
# and on through the fields to the lake by the woods
MILLSTREAM = ((-40.0, -74.0), (-42.0, -61.5), (-44.0, -45.0), (-46.0, -25.0), (-47.5, 0.0), (-50.0, 20.0), (-52.0, 38.0),
              (-53.0, 53.5), *wilds.MILLSTREAM_OUTSIDE)


def _quarters(attr: str) -> tuple:
    """Everything the newer quarters of the town (``quarters.py``) add to one of the level's lists."""
    return tuple(item for part in quarters.PARTS for item in getattr(part, attr))


def build_town_level() -> LevelDefinition:
    walls: List[WallDef] = town_walls()
    walls += WATCHTOWER.walls
    # north side of Market Street -----------------------------------------------------------------
    walls += house_walls(-74, -15, -66, -6, -71.5, "south")                         # smithy (home of the smith)
    walls += house_walls(-40, -22, -22, -6, -34.0, "south")                         # the inn
    walls += [                                                                       # inn: guest rooms along the north
        WallDef((-35.5, 0, -21.5), (-35.2, 4, -16.8)),
        WallDef((-31.0, 0, -21.5), (-30.7, 4, -16.8)),
        WallDef((-26.5, 0, -21.5), (-26.2, 4, -16.8)),
        WallDef((-39.5, 0, -16.8), (-37.2, 4, -16.5)), WallDef((-35.6, 0, -16.8), (-35.2, 4, -16.5)),    # keeper's room door 37.2..35.6
        WallDef((-35.2, 0, -16.8), (-34.1, 4, -16.5)), WallDef((-32.5, 0, -16.8), (-30.7, 4, -16.5)),    # room 1 door 34.1..32.5
        WallDef((-30.7, 0, -16.8), (-29.6, 4, -16.5)), WallDef((-28.0, 0, -16.8), (-26.2, 4, -16.5)),    # room 2 door 29.6..28.0
        WallDef((-26.2, 0, -16.8), (-25.1, 4, -16.5)), WallDef((-23.5, 0, -16.8), (-22.5, 4, -16.5)),    # room 3 door 25.1..23.5
    ]
    walls += house_walls(-18, -16, -6, -6, -15.0, "south")                          # general store
    walls += house_walls(7, -14, 17, -6, 10.0, "south")                             # the tailor
    walls += house_walls(24, -18, 34, -6, 27.5, "south", texture="planks_rot")      # the old granary (damp and rotting)
    # the smith's shed: a roof on four posts
    walls += [WallDef((x, 0, z), (x + 0.3, 4, z + 0.3)) for x, z in ((-65.0, -14.0), (-58.3, -14.0), (-65.0, -6.3), (-58.3, -6.3))]
    # Upper Street: the guard house (its north wall stops where the tower's own, with the door in it, takes over)
    gx1, gz1, gx2, gz2 = GUARD_HOUSE
    walls += house_walls(gx1, gz1, gx2, gz2, 48.5, "south", texture="wall", door_x2=TOWER_X1, door_width2=TOWER_X2 - TOWER_X1)
    # south side of Market Street -------------------------------------------------------------------
    walls += house_walls(-74, 7, -66, 15, -71.5, "north")                           # Barnaby's house
    walls += house_walls(-62, 6, -55, 13, -59.0, "north")                           # Tamsin's cottage
    walls += house_walls(-38, 6, -30, 13, -35.5, "north")                           # Fenwick's cottage
    walls += house_walls(-17, 7.5, -7, 17, -13.5, "north")                          # the hideout (Jae)
    walls += house_walls(8, 7, 17, 16, 11.0, "north")                               # Lys & Dorran
    walls += house_walls(22, 6, 29, 13, 24.0, "north")                              # Pell's cottage
    walls += house_walls(48, 6, 58, 16, 51.5, "north", texture="wall")              # jail
    walls += house_walls(*GATE_LODGE, 9.0, "north", texture="wall")                   # the gate lodge
    walls += [                                                                       # jail: the barred cell in the east
        WallDef((53.0, 0, 11.5), (53.5, 4, 15.5), texture="wall"),                  # cell west wall
        WallDef((53.0, 0, 11.5), (54.0, 4, 12.0), texture="wall"),                  # cell north wall stubs; bars gap 54..57
        WallDef((57.0, 0, 11.5), (57.5, 4, 12.0), texture="wall"),
    ]
    # stable posts (an open-sided roof)
    walls += [WallDef((x, 0, z), (x + 0.3, 3.6, z + 0.3)) for x, z in ((12.0, 34.0), (23.7, 34.0), (12.0, 43.7), (23.7, 43.7))]

    ashgrove = manors.manor("ashgrove", "Ashgrove Manor", ASHGROVE_X1, MANOR_Z1, "Lady Seraphine Ashgrove", MASTER, "Ottilie",
                            wealth="ashgrove", upstairs=[], downstairs=[])
    slots = ashgrove["slots"]
    ashgrove["containers"] += [
        chest("ashgrove_strongbox", "Strongbox", slots["strongbox"], owner="Lady Seraphine Ashgrove", strongbox=True, locked=True,
              lock_difficulty=MASTER, key_id="ashgrove_strongbox",
              items=[ItemDef("fingers_of_the_magpie", "Fingers of the Magpie", value=2400.0, category="clothing"), coins(220)]),
        chest("ashgrove_wardrobe", "Wardrobe Chest", slots["wardrobe"], owner="Lady Seraphine Ashgrove", locked=True,
              lock_difficulty=STRONG, key_id="ashgrove_door",
              items=[ItemDef("pearl_choker", "Pearl Choker", value=420.0, category="clothing"),
                     ItemDef("velvet_cloak", "Velvet Cloak", value=300.0, category="clothing"),
                     ItemDef("gilt_circlet", "Gilt Circlet", value=160.0, category="clothing")]),
        chest("ashgrove_coffer", "Household Coffer", slots["coffer"], owner="Lady Seraphine Ashgrove", along_x=False, locked=True,
              lock_difficulty=STRONG, key_id="ashgrove_door", items=[coins(140), ItemDef("silver_plate", "Silver Plate", value=60.0)]),
        manors.servant_stash("ashgrove", "Ottilie", slots["hidden"],
                             [coins(38), ItemDef("love_letter_hobb", "A Letter from Hobb", value=1.0)]),
    ]
    vane = manors.manor("vane", "Vane Manor", VANE_X1, MANOR_Z1, "Lord Aurelian Vane", MASTER, "Greaves", wealth="vane",
                        upstairs=[], downstairs=[])
    slots = vane["slots"]
    vane["containers"] += [
        chest("vane_strongbox", "Strongbox", slots["strongbox"], owner="Lord Aurelian Vane", strongbox=True, locked=True,
              lock_difficulty=MASTER, key_id="vane_strongbox",
              items=[ItemDef("black_cloak", "Black Wool Cloak", value=520.0, category="clothing"),
                     ItemDef("ruby_necklace", "Ruby Necklace", value=380.0, category="clothing"), coins(260)]),
        chest("vane_wardrobe", "Wardrobe Chest", slots["wardrobe"], owner="Lord Aurelian Vane", locked=True, lock_difficulty=STRONG,
              key_id="vane_door",
              items=[ItemDef("ravenfeather_cloak", "Ravenfeather Cloak", value=450.0, category="clothing"),
                     ItemDef("feathered_hat", "Feathered Hat", value=45.0, category="clothing"),
                     ItemDef("signet_ring_chain", "Signet on a Chain", value=150.0, category="clothing")]),
        chest("vane_coffer", "Household Coffer", slots["coffer"], owner="Lord Aurelian Vane", along_x=False, locked=True,
              lock_difficulty=STRONG, key_id="vane_door", items=[coins(180), ItemDef("silver_candelabra", "Silver Candelabra", value=90.0)]),
        # what the butler has skimmed off the housekeeping, fifteen years of it
        manors.servant_stash("vane", "Greaves", slots["hidden"],
                             [coins(175), ItemDef("butlers_ledger", "Greaves's Second Ledger", value=5.0)]),
    ]
    for m in (ashgrove, vane):
        walls += m["walls"]

    buildings = (
        BuildingDef("smithy", (-74.0, -15.0), (-66.0, -6.0)),
        BuildingDef("smithy_shed", (-65.0, -14.0), (-58.0, -6.0), roof_texture="roof_slate", open_sides=True, pad_with="smithy"),
        # (the houses round the well square stand on its level; the inn and Tamsin's on the bridge's)
        BuildingDef("inn", (-40.0, -22.0), (-22.0, -6.0), pad_with="bridge"),
        BuildingDef("shop", (-18.0, -16.0), (-6.0, -6.0), pad_with="square"),
        BuildingDef("tailor", (7.0, -14.0), (17.0, -6.0), pad_with="square"),
        BuildingDef("granary", (24.0, -18.0), (34.0, -6.0), roof_texture="thatch", floor_texture="planks_rot"),
        BuildingDef("guard_house", (gx1, gz1), (gx2, gz2), roof_texture="roof_slate"),
        BuildingDef("watchtower", (TOWER_X1, TOWER_Z1), (TOWER_X2, TOWER_Z2), wall_height=WATCHTOWER.wall_top, floor_texture="stone",
                    open_top=True, pad_with="guard_house"),
        ashgrove["building"], vane["building"],
        BuildingDef("barnaby_house", (-74.0, 7.0), (-66.0, 15.0)),
        BuildingDef("tamsin_house", (-62.0, 6.0), (-55.0, 13.0), roof_texture="thatch", pad_with="bridge"),
        BuildingDef("fenwick_house", (-38.0, 6.0), (-30.0, 13.0), roof_texture="thatch"),
        BuildingDef("hideout", (-17.0, 7.5), (-7.0, 17.0), pad_with="square"),
        BuildingDef("couple_house", (8.0, 7.0), (17.0, 16.0), pad_with="square"),
        BuildingDef("pell_house", (22.0, 6.0), (29.0, 13.0), roof_texture="thatch"),
        BuildingDef("jail", (48.0, 6.0), (58.0, 16.0), roof_texture="roof_slate"),
        BuildingDef("gate_lodge", GATE_LODGE[:2], GATE_LODGE[2:], roof_texture="roof_slate", pad_with="gate"),
        BuildingDef("stable", (12.0, 34.0), (24.0, 44.0), wall_height=3.6, roof_texture="thatch", open_sides=True, pad_with="gate"),
    )

    roads = [
        # Market Street, laid in lengths (some of it cracked and heaved), the bridge span left to the planks
        RoadDef((-78.5, STREET_NORTH), (-24.0, STREET_SOUTH)),          # (not drawn over the water: the bridge spans it)
        RoadDef((-24.0, STREET_NORTH), (-11.0, STREET_SOUTH), texture="cobble_cracked"),
        RoadDef((-11.0, -8.5), (11.0, 8.5)),                                         # the well square
        RoadDef((11.0, STREET_NORTH), (40.0, STREET_SOUTH)),
        RoadDef((40.0, STREET_NORTH), (52.0, STREET_SOUTH), texture="cobble_cracked"),
        RoadDef((52.0, STREET_NORTH), (78.5, STREET_SOUTH)),
        # the lanes
        RoadDef((-3.0, -36.0), (3.0, -8.5)),                                           # North Lane
        RoadDef((34.0, -36.0), (40.0, STREET_NORTH)),                                  # East Lane
        RoadDef((-36.0, -42.0), (62.0, -36.0)),                                        # Upper Street
        RoadDef((-3.0, 8.5), (3.0, 24.0)),                                             # Gate Road
        RoadDef((-3.0, 24.0), (3.0, 32.0), texture="cobble_cracked"),
        RoadDef((-3.0, 32.0), (3.0, 53.0)),
        RoadDef((3.0, 43.0), (12.5, 45.5)),                                            # to the gate lodge's door
        RoadDef((-46.0, 27.0), (-3.0, 32.0), texture="dirt"),                          # South Lane, from Nessa's door ...
        RoadDef((3.0, 27.0), (37.5, 32.0), texture="dirt"),                            # ... to Old Bertram's
        RoadDef((-65.5, -14.5), (-57.5, STREET_NORTH)),                                # the smith's yard under the shed
        RoadDef((-64.5, -22.0), (-60.5, -14.5), texture="dirt"),                       # path to the ore deposit
        RoadDef((-8.0, -58.0), (6.0, -42.0), texture="dirt"),                          # the park's walk
        *_quarters("roads"),
        *wilds.ROADS,
    ]
    roads = tuple(r for r in roads if r.max_xz[0] - r.min_xz[0] > 1.0)

    props: List[PropDef] = [
        PropDef("well", (-1.3, 0, -1.3), (1.3, 1.0, 1.3), texture="wall", kind="well"),
        # smithy
        bed("smith_bed", SMITH_BED, facing=0.0),
        PropDef("forge", (-61.5, 0, -11.2), (-59.7, 1.0, -9.9), texture="stone", kind="forge", facing=180.0),   # mouth to the south
        # inn
        PropDef("inn_bar", (-40.0 + WALL, 0, -11.8), (-35.0, 1.0, -11.2), texture="wood"),      # the bar; staff pass at its east end
        PropDef("inn_oven", (-39.5, 0, -16.3), (-37.7, 1.5, -14.7), texture="stone", kind="oven", facing=180.0),   # mouth to the south
        # shop signs hung beside the doors (bracketed to the wall, the board standing out from it)
        PropDef("sign_smith", (-68.1, 2.3, -5.7), (-67.3, 3.1, -4.9), texture="sign_smith", kind="sign", facing=270.0, solid=False),
        PropDef("sign_inn", (-30.6, 2.3, -5.7), (-29.8, 3.1, -4.9), texture="sign_inn", kind="sign", facing=270.0, solid=False),
        PropDef("sign_store", (-11.6, 2.3, -5.7), (-10.8, 3.1, -4.9), texture="sign_store", kind="sign", facing=270.0, solid=False),
        PropDef("sign_tailor", (13.4, 2.3, -5.7), (14.2, 3.1, -4.9), texture="sign_tailor", kind="sign", facing=270.0, solid=False),
        bed("inn_keeper_bed", INN_KEEPER_BED),
        bed("inn_room_1_bed", INN_ROOM_BEDS[0]),
        bed("inn_room_2_bed", INN_ROOM_BEDS[1]),
        bed("inn_room_3_bed", INN_ROOM_BEDS[2]),
        # general store
        PropDef("shop_counter", (-16.0, 0, -11.2), (-10.0, 1.0, -10.6), texture="wall"),
        PropDef("shop_table", (-8.5, 0, -13.6), (-7.0, 0.8, -12.1)),
        # tailor
        PropDef("tailor_counter", (9.0, 0, -10.2), (14.0, 1.0, -9.6), texture="wall"),
        bed("tailor_bed", TAILOR_BED),
        # the watch: two rows of bunks, heads to the walls
        *(bed(f"guard_bed_{i + 1}", spot, facing=270.0 if spot[0] < 50.0 else 90.0) for i, spot in enumerate(GUARD_BEDS)),
        PropDef("guard_weapon_rack", (gx1 + WALL, 0, gz2 - 3.2), (gx1 + WALL + 0.4, 2.0, gz2 - 1.0), texture="shelf"),
        *WATCHTOWER.props,
        bed("jail_guard_bed", JAIL_GUARD_BED, facing=270.0),
        bed("jail_cell_bed", JAIL_CELL_BED, facing=90.0),
        # homes
        bed("barnaby_bed", BARNABY_BED, facing=180.0),
        bed("tamsin_bed", TAMSIN_BED, facing=180.0),
        bed("jae_bed", JAE_BED, facing=180.0),
        bed("fenwick_bed", FENWICK_BED, facing=180.0),
        bed("lys_bed", LYS_BED, facing=90.0), bed("dorran_bed", DORRAN_BED, facing=90.0),
        bed("pell_bed", PELL_BED, facing=180.0),
        *ashgrove["props"], *vane["props"],
        # the bridge over the millstream where Market Street crosses it
        bridge("market_bridge", -51.5, -5.5, -43.5, 5.5, span="x"),          # Market Street crosses it east-west
        grate("north_culvert_grate", -44.0, -61.7, -40.0, -61.3), grate("south_culvert_grate", -55.0, 53.3, -51.0, 53.7),
        # --- a lived-in town: barrels, firewood, trees, hedges, stalls, cracks and damp ---
        *barrels("inn_barrels", [(-21.3, -8.0), (-21.3, -9.0), (-21.3, -13.0)]),
        *barrels("smithy_barrels", [(-57.2, -13.0), (-57.2, -12.0)]),
        *barrels("cooper_barrels", [(52.0, -10.0), (53.0, -10.4), (52.4, -11.3), (60.0, -9.0), (61.0, -9.2), (70.0, -10.4),
                                    (71.0, -10.6), (70.5, -11.5)]),
        *barrels("granary_barrels", [(35.0, -7.5), (35.0, -8.5)]),
        *barrels("jail_barrels", [(59.0, 7.5)]),
        *barrels("gate_barrels", [(-8.5, 47.5), (-8.5, 48.5), (16.0, 46.8)]),
        # the gate lodge: two bunks and a table for the gate watch
        *(bed(f"gate_lodge_bed_{i + 1}", spot, facing=90.0) for i, spot in enumerate(GATE_LODGE_BEDS)),
        *barrels("square_barrels", [(-4.7, -7.5)]),
        woodpile("smithy_woodpile", -75.2, -14.0, -74.3, -8.0),
        woodpile("inn_woodpile", -41.3, -20.0, -40.3, -14.0),
        woodpile("barnaby_woodpile", -65.6, 8.5, -64.8, 13.5),
        woodpile("tamsin_woodpile", -63.4, 7.0, -62.5, 11.0),
        woodpile("guard_woodpile", gx1 - 1.1, gz1 + 1.0, gx1 - 0.2, gz2 - 1.5),
        woodpile("pell_woodpile", 29.3, 7.0, 30.2, 11.5),
        stall("market_stall_1", 7.5, 4.6, facing=0.0), stall("market_stall_2", -7.0, -4.3, facing=180.0),
        stall("market_stall_3", 7.0, -4.3, facing=180.0),
        # the orchard in the north-west and trees about the lanes
        *(oak(f"orchard_oak_{i}", x, z) for i, (x, z) in enumerate(
            [(-74.0, -55.0), (-67.0, -52.0), (-60.0, -56.0), (-72.0, -45.0), (-64.0, -42.0), (-56.0, -47.0), (-74.0, -33.0),
             (-66.0, -30.0), (-56.0, -36.0), (-51.0, -55.0)])),
        *(oak(f"lane_oak_{i}", x, z, height=3.8) for i, (x, z) in enumerate(
            [(-35.0, -30.0), (-20.0, -24.0), (4.5, -24.0), (31.0, -23.0), (46.0, -22.0), (66.0, -33.0), (72.0, -48.0),
             (-42.5, 23.0), (-12.0, 44.0), (17.6, 50.6), (34.0, 46.0), (43.0, 37.0), (76.0, 40.0), (71.0, 19.0),
             (-70.0, 30.0), (-64.0, 44.0), (60.0, -26.0), (-74.0, -22.0)])),
        *(pine(f"lane_pine_{i}", x, z) for i, (x, z) in enumerate(
            [(-76.0, 48.0), (-72.0, 38.0), (76.0, 48.0), (74.0, 10.0), (76.0, -20.0), (-38.0, -57.0), (40.0, -58.0),
             (-60.0, 24.0)])),
        *(oak(f"park_oak_{i}", x, z, height=4.2) for i, (x, z) in enumerate([(-7.0, -55.0), (5.0, -53.0), (-6.0, -47.0)])),
        *(bush(f"park_bush_{i}", x, z) for i, (x, z) in enumerate([(-2.0, -57.0), (2.0, -49.0), (-9.0, -51.0), (7.0, -57.0)])),
        *(bush(f"manor_hedge_{i}", x, z, r=0.7) for i, (x, z) in enumerate([(-12.5, -44.5), (6.5, -44.5)])),
        # the park between the great houses: a fountain and benches to be seen on
        PropDef("park_fountain", (-2.6, 0.0, -52.6), (0.6, 2.2, -49.4), texture="marble", kind="fountain"),
        PropDef("park_bench_1", (-3.6, 0.0, -46.4), (-1.0, 0.5, -45.8), texture="wood", kind="bench"),
        PropDef("park_bench_2", (1.0, 0.0, -55.6), (3.6, 0.5, -55.0), texture="wood", kind="bench"),
        *(bush(f"lane_bush_{i}", x, z, r=0.6, h=0.9) for i, (x, z) in enumerate(
            [(-18.5, 18.5), (18.5, 17.5), (-40.5, 14.0), (30.5, 14.0), (46.0, 18.0), (-36.5, -24.0), (20.0, -16.0)])),
        # gardens between Market Street and South Lane
        fence("garden_fence_1", -38.0, 19.0, -20.0, 19.2), fence("garden_fence_2", 20.0, 19.0, 36.0, 19.2),
        haystack("stable_hay_1", 15.0, 40.5), haystack("stable_hay_2", 20.0, 39.0, r=0.9, h=1.3),
        # cracks in the level squares
        crack("square_crack_1", -6.5, 5.0), crack("square_crack_2", 5.0, -6.5, w=1.2, d=1.6), crack("square_crack_3", 8.5, 3.0),
        crack("gate_crack_1", -5.0, 57.0, w=1.8), crack("gate_crack_2", 6.0, 46.0),
        # water damage down old walls
        stain("granary_stain_1", 25.0, -6.0, 27.0, -6.0, 180.0), stain("granary_stain_2", 34.0, -16.0, 34.0, -12.5, 90.0, y2=3.6),
        stain("inn_stain", -40.0, -13.0, -40.0, -10.5, 270.0), stain("tamsin_stain", -55.0, 8.0, -55.0, 10.0, 90.0, y2=2.6),
        stain("jail_stain", 58.0, 9.0, 58.0, 12.0, 90.0), stain("chapel_stain", -29.0, 37.0, -29.0, 39.0, 270.0, y2=4.2),
        *_quarters("props"),
        *wilds.PROPS,
    ]

    stew = lambda n: ItemDef("stew", "Pot of Stew", value=4.0, stackable=True, quantity=n, is_food=True, category="food")  # noqa: E731
    bread = lambda n: ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=n, is_food=True, category="food")  # noqa: E731
    ale = lambda n: ItemDef("ale", "Mug of Ale", value=3.0, stackable=True, quantity=n, is_food=True, category="food")  # noqa: E731

    containers = [
        # general store
        ContainerDef("shop_shelf_west", "Shop Shelf", (-17.5, 0, -15.5), (-14.5, 1.8, -14.7), owner="Shopkeeper Barnaby",
                     crime_name="Shoplifting",
                     items=(ItemDef("candlestick", "Silver Candlestick", value=45.0),
                            ItemDef("wine", "Bottle of Wine", value=12.0, stackable=True, quantity=3, category="food"),
                            ItemDef("thread", "Spool of Silk Thread", value=4.0, stackable=True, quantity=5))),
        ContainerDef("shop_shelf_east", "Shop Shelf", (-9.5, 0, -15.5), (-6.5, 1.8, -14.7), owner="Shopkeeper Barnaby",
                     crime_name="Shoplifting",
                     items=(ItemDef("lantern", "Brass Lantern", value=30.0),
                            ItemDef("honey", "Jar of Honey", value=9.0, stackable=True, quantity=2, category="food"),
                            ItemDef("plate", "Pewter Plate", value=18.0),
                            ItemDef("pickaxe", "Pickaxe", value=35.0, category="tool"),
                            ItemDef("mallet", "Mallet", value=12.0, category="tool"))),
        # smithy: the rack under the shed and the smith's own chest
        ContainerDef("smithy_rack", "Weapon Rack", (-64.4, 0, -13.9), (-61.2, 1.8, -13.2), owner="Gareth the Smith",
                     crime_name="Theft", texture="shelf",
                     items=(ItemDef("iron_dagger", "Iron Dagger", value=15.0, category="weapon"),       # steel is cheap ...
                            ItemDef("short_sword", "Short Sword", value=30.0, category="weapon"),
                            ItemDef("leather_cap", "Leather Cap", value=25.0, category="armor"),
                            ItemDef("iron_helm", "Iron Helm", value=20.0, category="armor"),             # ... bright and loud
                            ItemDef("chain_shirt", "Chain Shirt", value=40.0, category="armor"),
                            ItemDef("leather_jerkin", "Brown Leather Jerkin", value=45.0, category="armor"),
                            ItemDef("iron_ore", "Iron Ore", value=6.0, stackable=True, quantity=4, category="ore"),
                            ItemDef("mallet", "Mallet", value=12.0, category="tool"))),
        table("smith_table", "Kitchen Table", (-68.2, 0.0, -9.6), items=[stew(14)], owner="Gareth the Smith"),
        chest("smith_chest", "Smith's Chest", (-69.5, 0.0, -14.0), owner="Gareth the Smith", locked=True, lock_difficulty=STURDY,
              key_id="smithy_door", items=[coins(45), ItemDef("smith_apron", "Smith's Leather Apron", value=80.0, category="clothing")]),
        # the tailor's rack
        ContainerDef("tailor_rack", "Clothes Rack", (8.0, 0, -13.5), (12.0, 1.8, -12.7), owner="Mirelle the Tailor",
                     crime_name="Shoplifting", texture="shelf",
                     items=(ItemDef("linen_shirt", "White Linen Shirt", value=4.0, category="clothing"),    # bright and cheap
                            ItemDef("white_cloak", "White Wool Cloak", value=8.0, category="clothing"),
                            ItemDef("red_cap", "Red Wool Cap", value=3.0, category="clothing"),
                            ItemDef("silk_gloves", "Silk Gloves", value=60.0, category="clothing", noise_modifier=-0.1),
                            ItemDef("soft_slippers", "Pale Cloth Slippers", value=45.0, category="clothing"),
                            ItemDef("needles", "Steel Needles", value=6.0, stackable=True, quantity=3, category="tool"))),
        table("tailor_table_food", "Side Table", (14.5, 0.0, -7.6), items=[bread(11)], owner="Mirelle the Tailor"),
        # the inn: kitchen table and three dining tables
        table("inn_kitchen_table", "Kitchen Table", (-37.5, 0.0, -13.5), items=[stew(12), bread(10), ale(12)],
              owner="Hilde the Innkeeper"),
        table("inn_table_1", "Inn Table", INN_TABLES[0], items=[ale(3)], owner="Hilde the Innkeeper"),
        table("inn_table_2", "Inn Table", INN_TABLES[1], items=[ale(3)], owner="Hilde the Innkeeper"),
        table("inn_table_3", "Inn Table", INN_TABLES[2], items=[bread(2)], owner="Hilde the Innkeeper"),
        table("inn_table_4", "Inn Table", INN_TABLES[3], items=[ale(2)], owner="Hilde the Innkeeper"),
        table("gate_lodge_table", "Lodge Table", (10.9, 0.0, 49.6), owner="guards", size=(0.6, 0.45),
              items=[ItemDef("ration", "Guard's Ration", value=3.0, stackable=True, quantity=10, is_food=True, category="food")]),
        # what the townsfolk keep where they think nobody looks
        stash("barnaby_takings", "Loose Board under the Counter", (-15.8, 0.0, -11.9), owner="Shopkeeper Barnaby",
              items=[coins(90), ItemDef("ivory_dice", "Ivory Dice", value=20.0)]),
        stash("fenwick_hearthstone", "Loose Hearthstone", (-31.4, 0.0, 11.8), owner="Fenwick",
              items=[coins(26), ItemDef("fenwick_locket", "Enamelled Locket", value=65.0)]),
        ContainerDef("hilde_ale_cask", "Old Ale Cask", (-41.4, 0.0, -21.0), (-40.4, 1.1, -20.0), owner="Hilde the Innkeeper",
                     texture="barrel", kind="barrel_store",
                     items=(coins(130), ItemDef("hilde_recipes", "Hilde's Book of Receipts", value=15.0))),
        table("inn_table_5", "Inn Table", INN_TABLES[4], items=[ale(2)], owner="Hilde the Innkeeper"),
        # homes
        table("barnaby_table", "Kitchen Table", (-71.5, 0.0, 10.4), items=[stew(8)], owner="Shopkeeper Barnaby"),
        table("tamsin_table", "Kitchen Table", (-56.25, 0.0, 8.6), items=[stew(14)], owner="Tamsin"),
        table("jae_table", "Table", (-14.0, 0.0, 9.5), owner="Jae the Fence",
              items=[ItemDef("dried_meat", "Dried Meat", value=5.0, stackable=True, quantity=8, is_food=True, category="food")]),
        table("fenwick_table", "Kitchen Table", (-31.75, 0.0, 8.5), items=[stew(11)], owner="Fenwick"),
        table("couple_table", "Kitchen Table", (10.0, 0.0, 11.6), items=[stew(24)], owner="Lys", size=(0.75, 0.9)),
        table("pell_table", "Kitchen Table", (23.5, 0.0, 8.6), items=[bread(11)], owner="Pell"),
        chest("barnaby_chest", "Barnaby's Chest", (-73.0, 0.0, 8.5), owner="Shopkeeper Barnaby", along_x=False, locked=True,
              lock_difficulty=STURDY, key_id="home_door", items=[coins(60), ItemDef("ledger", "Shop Ledger", value=5.0)]),
        chest("tamsin_chest", "Blanket Chest", (-61.0, 0.0, 7.4), owner="Tamsin", along_x=False,
              items=[coins(12), ItemDef("wedding_ring", "Wedding Ring", value=70.0)]),
        # guards
        table("guard_table", "Guard Table", (50.0, 0.0, -49.0), owner="guards", size=(1.5, 1.0),
              items=[ItemDef("ration", "Guard's Ration", value=3.0, stackable=True, quantity=16, is_food=True, category="food")]),
        chest("guard_chest", "Watch Chest", (gx2 - WALL - 0.4, 0.0, gz2 - 2.0), owner="guards", along_x=False, locked=True,
              lock_difficulty=STRONG, key_id="guardhouse_door",
              items=[coins(75), ItemDef("steel_sword", "Steel Sword", value=60.0, category="weapon")]),
        # the confiscation lockbox: the cell key opens it too; unlocked for a released prisoner
        ContainerDef("jail_storage", "Confiscation Lockbox", (48.8, 0, 13.5), (50.3, 1.0, 15.0), owner="guards",
                     texture="lockbox", kind="lockbox", locked=True, lock_difficulty=STRONG, key_id="cell_door"),
        # the old granary: nobody's in particular
        ContainerDef("granary_sacks", "Grain Sacks", (30.5, 0, -17.3), (33.3, 1.0, -15.5), texture="hay",
                     items=(ItemDef("grain", "Sack of Grain", value=4.0, stackable=True, quantity=3, category="food", is_food=True),
                            ItemDef("rusty_lantern", "Rusty Lantern", value=6.0))),
        *ashgrove["containers"], *vane["containers"],
        *_quarters("containers"),
        *wilds.CONTAINERS,
    ]

    stations = (
        StationDef("ore_deposit", "Ore Deposit", (-64.0, 0, -21.0), (-61.0, 1.6, -18.0), texture="ore", kind="ore",
                   yields=ItemDef("iron_ore", "Iron Ore", value=6.0, stackable=True, category="ore"),
                   requires_item_id="pickaxe", tool_label="a pickaxe"),
        StationDef("anvil", "Anvil", (-62.5, 0, -9.4), (-61.5, 0.8, -8.6), texture="stone",
                   consumes=("iron_ore", 1), consumes_label="1 Iron Ore", requires_item_id="mallet", tool_label="a mallet",
                   produces=ItemDef("iron_lockpick", "Iron Lockpick", value=5.0, is_lockpick=True, stackable=True, quantity=3,
                                    category="tool")),
        *wilds.STATIONS,
    )

    areas = (
        # merchants (north)
        AreaDef("the smithy", (-74.0, -15.0), (-66.0, -6.0), owner="Gareth the Smith", restricted=True),
        AreaDef("the smith's yard", (-65.5, -14.5), (-57.5, -6.0), owner="Gareth the Smith"),
        AreaDef("the inn kitchen", (-40.0, -16.5), (-35.2, -11.0), owner="Hilde the Innkeeper", restricted=True),   # behind the bar
        AreaDef("the keeper's room", (-40.0, -22.0), (-35.5, -16.5), owner="Hilde the Innkeeper", restricted=True),
        AreaDef("room one", (-35.5, -22.0), (-31.0, -16.5), owner="Hilde the Innkeeper", restricted=True),
        AreaDef("room two", (-31.0, -22.0), (-26.5, -16.5), owner="Hilde the Innkeeper", restricted=True),
        AreaDef("room three", (-26.5, -22.0), (-22.0, -16.5), owner="Hilde the Innkeeper", restricted=True),
        AreaDef("the inn", (-40.0, -22.0), (-22.0, -6.0), owner="Hilde the Innkeeper"),    # a public house at every hour
        AreaDef("behind the counter", (-18.0, -16.0), (-6.0, -10.6), owner="Shopkeeper Barnaby", restricted=True),
        AreaDef("the shop", (-18.0, -10.6), (-6.0, -6.0), owner="Shopkeeper Barnaby", restricted=True,
                public_hours=(SHOP_OPEN_HOUR, SHOP_CLOSE_HOUR)),
        AreaDef("the tailor's back room", (7.0, -14.0), (17.0, -9.6), owner="Mirelle the Tailor", restricted=True),
        AreaDef("the tailor's", (7.0, -9.6), (17.0, -6.0), owner="Mirelle the Tailor", restricted=True,
                public_hours=(TAILOR_OPEN_HOUR, TAILOR_CLOSE_HOUR)),
        AreaDef("the old granary", (24.0, -18.0), (34.0, -6.0)),
        AreaDef("the watchtower", (TOWER_X1, TOWER_Z1), (TOWER_X2, TOWER_Z2), owner="guards", restricted=True),
        AreaDef("the guard house", (gx1, gz1), (gx2, gz2), owner="guards", restricted=True),
        ashgrove["area"], vane["area"],
        # homes (south)
        AreaDef("Barnaby's house", (-74.0, 7.0), (-66.0, 15.0), owner="Shopkeeper Barnaby", restricted=True),
        AreaDef("Tamsin's cottage", (-62.0, 6.0), (-55.0, 13.0), owner="Tamsin", restricted=True),
        AreaDef("the hideout", (-17.0, 7.5), (-7.0, 17.0), owner="Jae the Fence", restricted=True,
                trusted_by=("Jae the Fence",), curfew_hours=(22, 6)),
        AreaDef("Fenwick's cottage", (-38.0, 6.0), (-30.0, 13.0), owner="Fenwick", restricted=True),
        AreaDef("Lys and Dorran's house", (8.0, 7.0), (17.0, 16.0), owner="Lys", restricted=True),
        AreaDef("Pell's cottage", (22.0, 6.0), (29.0, 13.0), owner="Pell", restricted=True),
        AreaDef("the jail cell", (53.0, 11.5), (58.0, 16.0), owner="guards", restricted=True),
        AreaDef("the jail", (48.0, 6.0), (58.0, 16.0), owner="guards"),
        AreaDef("the gate lodge", GATE_LODGE[:2], GATE_LODGE[2:], owner="guards", restricted=True),
        *_quarters("areas"),
        AreaDef("the well", (-11.0, -8.5), (11.0, 8.5)),
        AreaDef("the town gate", (-12.0, 44.0), (12.0, 53.0)),
        AreaDef("Upper Street", (-36.0, -42.0), (62.0, -36.0)),
        AreaDef("main street", (-78.5, -6.0), (78.5, 6.0)),
        AreaDef("the streets", (TOWN_X1, TOWN_Z1), (TOWN_X2, TOWN_Z2)),
        *wilds.AREAS,
        wilds.COUNTRYSIDE,
    )

    doors = (
        # north side: hinges on the south walls, leaves swing into the buildings
        DoorDef("smithy_door", (-71.5, 0.0, -6.0), is_locked=True, key_id="smithy_door_key", owners=("Gareth the Smith",),
                lock_difficulty=STURDY, label="Gareth's Smithy"),
        DoorDef("inn_door", (-34.0, 0.0, -6.0), is_locked=False, key_id="inn_door_key", owners=("Hilde the Innkeeper",),
                lock_difficulty=STURDY, label='The Drowsy Dragon'),
        DoorDef("keeper_room_door", (-37.2, 0.0, -16.5), is_locked=True, key_id="keeper_room_door_key", width=1.6,
                owners=("Hilde the Innkeeper",), lock_difficulty=STURDY, label="Hilde's Room"),
        DoorDef("inn_room_1_door", (-34.1, 0.0, -16.5), is_locked=True, key_id="inn_room_1_door_key", width=1.6,
                owners=("Hilde the Innkeeper",), lock_difficulty=SIMPLE, label='Room One'),
        DoorDef("inn_room_2_door", (-29.6, 0.0, -16.5), is_locked=True, key_id="inn_room_2_door_key", width=1.6,
                owners=("Hilde the Innkeeper",), lock_difficulty=SIMPLE, label='Room Two'),
        DoorDef("inn_room_3_door", (-25.1, 0.0, -16.5), is_locked=True, key_id="inn_room_3_door_key", width=1.6,
                owners=("Hilde the Innkeeper",), lock_difficulty=SIMPLE, label='Room Three'),
        # (Barnaby's back door is an old, simple lock: the one every urchin learns on)
        DoorDef("shop_door", (-15.0, 0.0, -6.0), is_locked=True, key_id="shop_key", owners=("Shopkeeper Barnaby",),
                lock_difficulty=SIMPLE, label="Barnaby's General Store"),
        DoorDef("tailor_door", (10.0, 0.0, -6.0), is_locked=True, key_id="tailor_door_key", owners=("Mirelle the Tailor",),
                lock_difficulty=STURDY, label="Mirelle's Tailoring"),
        DoorDef("granary_door", (27.5, 0.0, -6.0), is_locked=True, key_id="granary_door_key", lock_difficulty=SIMPLE, label='The Old Granary'),
        DoorDef("guardhouse_door", (48.5, 0.0, GUARD_HOUSE[3]), is_locked=True, key_id="guard_key", owners=("guards",),
                lock_difficulty=STRONG, label='Guard House'),
        DoorDef("tower_door", (TOWER_DOOR_X, 0.0, TOWER_Z2), is_locked=True, key_id="tower_key", owners=("guards",),
                lock_difficulty=STURDY, width=2.0, swing_out=True, label='Watchtower'),     # opens into the guard house: the stair starts right inside
        ashgrove["door"], vane["door"], ashgrove["servant_door"], vane["servant_door"],
        # south side: hinges on the north walls
        DoorDef("home_door", (-71.5, 0.0, 7.0), is_locked=True, key_id="home_key", owners=("Shopkeeper Barnaby",),
                lock_difficulty=SIMPLE, label="Barnaby's House"),
        DoorDef("tamsin_door", (-59.0, 0.0, 6.0), is_locked=True, key_id="tamsin_door_key", owners=("Tamsin",), lock_difficulty=SIMPLE, label="Tamsin's Cottage"),
        DoorDef("hideout_door", (-13.5, 0.0, 7.5), is_locked=False, key_id="hideout_door_key", owners=("Jae the Fence",), label="Jae's House"),
        DoorDef("fenwick_door", (-35.5, 0.0, 6.0), is_locked=True, key_id="fenwick_door_key", owners=("Fenwick",), lock_difficulty=SIMPLE, label="Fenwick's Cottage"),
        DoorDef("couple_door", (11.0, 0.0, 7.0), is_locked=True, key_id="couple_door_key", owners=("Lys", "Dorran"),
                lock_difficulty=SIMPLE, label="Lys & Dorran's House"),
        DoorDef("pell_door", (24.0, 0.0, 6.0), is_locked=True, key_id="pell_door_key", owners=("Pell",), lock_difficulty=SIMPLE, label="Pell's Cottage"),
        # the jail house is open to all (nobody owns - i.e. locks - its door); the cell is a barred gate
        DoorDef("jail_door", (51.5, 0.0, 6.0), is_locked=False, key_id="guard_key", lock_difficulty=STRONG, label='Town Jail'),
        DoorDef("gate_lodge_door", (9.0, 0.0, GATE_LODGE[1]), is_locked=True, key_id="gate_lodge_door_key", owners=("guards",),
                lock_difficulty=STURDY, label="Gate Lodge"),
        DoorDef("cell_door", (54.0, 0.0, 12.0), is_locked=True, key_id="cell_key", owners=("guards",), lock_difficulty=STRONG,
                kind=DoorKind.BARS, label='Jail Cell'),
        # the town gate: two leaves hinged on the gatehouse towers, barred shut until the town lets you go
        DoorDef("town_gate_west", (GATE_X1, 0.0, GATE_Z), is_locked=True, key_id="town_gate_key", width=4.0, height=4.6,
                lock_difficulty=LEGENDARY, gate="town_gate", label='Town Gate'),
        DoorDef("town_gate_east", (GATE_X2, 0.0, GATE_Z), is_locked=True, key_id="town_gate_key", width=4.0, height=4.6,
                lock_difficulty=LEGENDARY, gate="town_gate", hinge_right=True, label='Town Gate'),
        *_quarters("doors"),
        *wilds.DOORS,
    )

    stock_tiers = (
        StockTier(2, (ItemDef("fine_lockpick", "Fine Steel Lockpick", value=15.0, is_lockpick=True, stackable=True, quantity=3,
                              category="tool"),), container="shop_shelf_east"),
        StockTier(4, (ItemDef("jeweled_dagger", "Jeweled Dagger", value=180.0, category="weapon"),), container="shop_shelf_west"),
    )
    # the tailor sells brown wool at the dear end: dark cloth is never on a rack (it is stolen, from
    # the rich, the smugglers and the keep)
    tailor_tiers = (
        StockTier(2, (ItemDef("wool_cloak", "Brown Wool Cloak", value=120.0, category="clothing"),), container="tailor_rack"),
        StockTier(4, (ItemDef("feathered_hat", "Feathered Hat", value=45.0, category="clothing"),), container="tailor_rack"),
    )
    smith_tiers = (
        StockTier(2, (ItemDef("plate_cuirass", "Plate Cuirass", value=90.0, category="armor"),), container="smithy_rack"),
        StockTier(3, (ItemDef("steel_sword", "Steel Sword", value=60.0, category="weapon"),), container="smithy_rack"),
    )

    npcs = (
        # ---- merchants -------------------------------------------------------------------------
        NPCDef(
            "Shopkeeper Barnaby", BARNABY_BED, can_trade=True, role="shopkeeper",
            shop_containers=("shop_shelf_west", "shop_shelf_east"), restock_days=3, restock_tiers=stock_tiers,
            inventory_reset_days=3, gold=SHOPKEEPER_GOLD,
            inventory=(
                ItemDef("shop_key", "Store Backroom Key", value=1.0, is_key=True, key_door_id="shop_door", for_sale=False, category="key"),
                ItemDef("home_key", "Barnaby's House Key", value=1.0, is_key=True, key_door_id="home_door", for_sale=False, category="key"),
                bread(4),                                       # the same loaf every kitchen table holds
                ItemDef("apple_1", "Crisp Apple", value=8.0, stackable=True, quantity=6, is_food=True, category="food"),
                ItemDef("torch_item", "Portable Torch", value=25.0, category="tool"),
            ),
            schedule=resident_tasks("Barnaby", "home_door", BARNABY_BED, BARNABY_TABLE_SEAT, "barnaby_table", [
                TaskDef(SHOP_OPEN_HOUR, SHOP_LUNCH_HOUR, SHOP_COUNTER, AIState.SHOPKEEPING,
                        "Barnaby opened the shop for business.", unlock_doors=("shop_door",),
                        depart_text="Time to open up the shop.", facing=180.0),
                TaskDef(SHOP_LUNCH_HOUR, SHOP_LUNCH_HOUR + 1, SHOP_BACKROOM, AIState.EATING,
                        "Barnaby is having lunch in the back room.", facing=90.0),
                TaskDef(SHOP_LUNCH_HOUR + 1, SHOP_CLOSE_HOUR, SHOP_COUNTER, AIState.SHOPKEEPING,
                        "Barnaby is back behind the counter.", facing=180.0),
            ], wake=7, supper=(SHOP_CLOSE_HOUR, SHOP_CLOSE_HOUR + 2)),       # (a long walk home from the shop)
        ),
        NPCDef(
            "Hilde the Innkeeper", INN_KEEPER_BED, can_trade=True, role="innkeeper", trade_availability="awake",
            shop_containers=("inn_kitchen_table",), restock_days=3, inventory_reset_days=1, gold=INNKEEPER_GOLD,
            inventory=(
                key_item("inn_door", "Inn Key"), key_item("keeper_room_door", "Keeper's Room Key"),
                key_item("inn_room_1_door", "Room One Key"), key_item("inn_room_2_door", "Room Two Key"),
                key_item("inn_room_3_door", "Room Three Key"),
                ItemDef("rent_night", "A Room for the Night", value=10.0, category="service"),
                ItemDef("rent_week", "A Room for the Week", value=50.0, category="service"),
                ItemDef("candle_stub", "Tallow Candle", value=2.0, stackable=True, quantity=6),
                ItemDef("soap", "Lye Soap", value=4.0, stackable=True, quantity=3),
            ),
            schedule=(
                TaskDef(0, 5, INN_KEEPER_BED, AIState.SLEEPING, "Hilde is asleep in the room behind the bar.",
                        lock_doors=("keeper_room_door",)),
                TaskDef(5, 6, INN_KITCHEN, AIState.WORKING, "Hilde is lighting the stove and setting the kitchen to rights.",
                        unlock_doors=("keeper_room_door",), facing=0.0, depart_text="Up with the lark. Bread won't bake itself."),
                TaskDef(6, 10, INN_BAR, AIState.SHOPKEEPING, "Hilde is behind the bar of the Drowsy Dragon.", facing=180.0),
                TaskDef(10, 11, INN_SEATS[0], AIState.CLEANING, "Hilde is wiping down the tables.",
                        patrol_points=(INN_SEATS[0], INN_SEATS[2], INN_SEATS[1])),
                TaskDef(11, 15, INN_BAR, AIState.SHOPKEEPING, "Hilde is back behind the bar.", facing=180.0),
                TaskDef(15, 16, INN_SEATS[1], AIState.CLEANING, "Hilde is clearing the tables.",
                        patrol_points=(INN_SEATS[1], INN_SEATS[2], INN_SEATS[0])),
                TaskDef(16, 20, INN_BAR, AIState.SHOPKEEPING, "Hilde is pouring for the evening crowd.", facing=180.0),
                TaskDef(20, 21, INN_SEATS[2], AIState.CLEANING, "Hilde is sweeping around the tables.",
                        patrol_points=(INN_SEATS[2], INN_SEATS[0], INN_SEATS[1])),
                TaskDef(21, 24, INN_BAR, AIState.SHOPKEEPING, "Hilde keeps the bar open for the late crowd.", facing=180.0),
            ),
        ),
        NPCDef(
            "Gareth the Smith", SMITH_BED, can_trade=True, role="smith", trade_categories=("weapon", "armor", "ore", "tool"),
            shop_containers=("smithy_rack",), restock_days=3, restock_tiers=smith_tiers, inventory_reset_days=3, gold=SMITH_GOLD,
            inventory=(key_item("smithy_door", "Smithy Key"),
                       ItemDef("hammer", "Smith's Hammer", value=20.0, for_sale=False, category="tool"),
                       ItemDef("mallet", "Mallet", value=12.0, for_sale=False, category="tool"),
                       ItemDef("pickaxe", "Pickaxe", value=35.0, for_sale=False, category="tool")),
            schedule=resident_tasks("Gareth", "smithy_door", SMITH_BED, SMITH_TABLE_SEAT, "smith_table", [
                TaskDef(8, 12, ANVIL_SPOT, AIState.WORKING, "Gareth is hammering at the anvil.", facing=0.0,
                        depart_text="Right. Iron won't shape itself."),
                TaskDef(12, 13, SMITH_TABLE_SEAT, AIState.EATING, "Gareth is eating at his table.", facing=0.0,
                        food_container="smith_table"),
                TaskDef(13, 16, FORGE_SPOT, AIState.WORKING, "Gareth is working the forge.", facing=0.0),
                TaskDef(16, 18, ANVIL_SPOT, AIState.WORKING, "Gareth is back at the anvil.", facing=0.0),
                TaskDef(18, 19, SMITH_TABLE_SEAT, AIState.EATING, "Gareth is having supper.", facing=0.0, food_container="smith_table"),
                visit(19, 20, INN_SEATS[3], AIState.DRINKING, "Gareth is having an ale at the Drowsy Dragon.", facing=270.0),
            ], wake=7, bedtime=21),
        ),
        NPCDef(
            "Mirelle the Tailor", TAILOR_BED, can_trade=True, role="tailor", trade_categories=("clothing", "tool"),
            shop_containers=("tailor_rack",), restock_days=3, restock_tiers=tailor_tiers, inventory_reset_days=3, gold=TAILOR_GOLD,
            inventory=(key_item("tailor_door", "Tailor's Key"),
                       ItemDef("thimble", "Silver Thimble", value=12.0, category="tool")),
            schedule=(
                TaskDef(0, 8, TAILOR_BED, AIState.SLEEPING, "Mirelle is asleep in her back room.", lock_doors=("tailor_door",)),
                TaskDef(8, TAILOR_OPEN_HOUR, TAILOR_TABLE_SEAT, AIState.EATING, "Mirelle is having breakfast.", facing=90.0,
                        food_container="tailor_table_food"),
                TaskDef(TAILOR_OPEN_HOUR, 13, TAILOR_COUNTER, AIState.SHOPKEEPING, "Mirelle opened the tailor's shop.",
                        unlock_doors=("tailor_door",), facing=180.0),
                TaskDef(13, 14, TAILOR_TABLE_SEAT, AIState.EATING, "Mirelle is having lunch.", facing=90.0, food_container="tailor_table_food"),
                TaskDef(14, TAILOR_CLOSE_HOUR, TAILOR_COUNTER, AIState.SHOPKEEPING, "Mirelle is back at her counter.", facing=180.0),
                stroll(TAILOR_CLOSE_HOUR, 19, "Mirelle", STROLL_EAST, "Mirelle is stretching her legs after a day's sewing."),
                visit(19, 21, INN_SEATS[4], AIState.DRINKING, "Mirelle is taking a cup of wine at the inn.", facing=270.0),
                TaskDef(21, 22, TAILOR_TABLE_SEAT, AIState.EATING, "Mirelle is having a late supper.", facing=90.0,
                        food_container="tailor_table_food"),
                TaskDef(22, 24, TAILOR_BED, AIState.SLEEPING, "Mirelle locked up and went to bed.", lock_doors=("tailor_door",)),
            ),
        ),
        NPCDef(
            "Jae the Fence", JAE_DEN, is_fence=True, can_trade=True, inventory_reset_days=3, gold=FENCE_GOLD, role="fence",
            trade_availability="always",                        # no rest for the wicked: woken for business, then back to bed
            inventory=(
                key_item("hideout_door", "Hideout Key"),
                ItemDef("lockpick_pack", "Master Lockpick", value=20.0, is_lockpick=True, stackable=True, quantity=3, category="tool"),
                ItemDef("gold_ring", "Gold Ring", value=150.0),
                ItemDef("black_hood", "Black Cloth Hood", value=360.0, category="clothing"),     # a thief's hood, at a thief's price
                ItemDef("dried_meat", "Dried Meat", value=5.0, stackable=True, quantity=4, is_food=True, for_sale=False, category="food"),
            ),
            schedule=(
                TaskDef(0, 6, JAE_BED, AIState.SLEEPING, "Jae is asleep in the hideout."),
                TaskDef(6, 7, JAE_TABLE_SEAT, AIState.EATING, "Jae is eating at his table.", facing=0.0, food_container="jae_table"),
                TaskDef(7, 22, JAE_DEN, AIState.SHOPKEEPING, "Jae is in his den, ready to deal in stolen goods.", facing=0.0),
                TaskDef(22, 23, JAE_TABLE_SEAT, AIState.EATING, "Jae is having a late supper.", facing=0.0,
                        food_container="jae_table"),
                TaskDef(23, 24, JAE_BED, AIState.SLEEPING, "Jae turned in for the night."),
            ),
        ),
        # ---- townsfolk -------------------------------------------------------------------------
        # (their days come round in turn - ``NPCDef.rota``: one day at the inn, the next at the chapel
        # or by the water; and wherever a day ends at the far end of town, the next hour's task at
        # home is long enough to walk back for)
        NPCDef(
            "Tamsin", TAMSIN_BED, role="widow", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
            inventory=(key_item("tamsin_door", "Cottage Key"), ItemDef("locket", "Silver Locket", value=55.0, for_sale=False)),
            schedule=resident_tasks("Tamsin", "tamsin_door", TAMSIN_BED, TAMSIN_TABLE_SEAT, "tamsin_table", [
                stroll(8, 10, "Tamsin", STROLL_WEST),
                visit(10, 11, SHOP_FLOOR, AIState.SHOPPING, "Tamsin is picking over Barnaby's shelves."),
                TaskDef(11, 13, TAMSIN_TABLE_SEAT, AIState.EATING, "Tamsin is having a bite at home.", unlock_doors=("tamsin_door",),
                        facing=0.0, food_container="tamsin_table"),
                stroll(13, 15, "Tamsin", STROLL_EAST, "Tamsin is taking the afternoon air."),
                visit(15, 16, TAILOR_FLOOR, AIState.SHOPPING, "Tamsin is admiring Mirelle's cloth."),
                visit(16, 18, INN_SEATS[4], AIState.DRINKING, "Tamsin is sharing gossip over a cup at the inn.", facing=270.0),
                TaskDef(18, 20, TAMSIN_TABLE_SEAT, AIState.EATING, "Tamsin is home for the evening.", unlock_doors=("tamsin_door",),
                        facing=0.0, food_container="tamsin_table"),
            ], wake=7, bedtime=21),
            rota=(
                # her husband's grave, and Nessa's washing line for the news
                resident_tasks("Tamsin", "tamsin_door", TAMSIN_BED, TAMSIN_TABLE_SEAT, "tamsin_table", [
                    visit(8, 10, quarters.CHAPEL_PEWS[1], AIState.WORKING, "Tamsin is praying in the chapel.", facing=180.0),
                    visit(10, 12, quarters.GRAVEYARD[1], AIState.WORKING, "Tamsin is tending her husband's grave.", facing=270.0),
                    TaskDef(12, 14, TAMSIN_TABLE_SEAT, AIState.EATING, "Tamsin is having a bite at home.",
                            unlock_doors=("tamsin_door",), facing=0.0, food_container="tamsin_table"),
                    visit(14, 17, quarters.NESSA_GOSSIP, AIState.WORKING, "Tamsin is helping Nessa with the washing, and the news.",
                          facing=90.0),
                    TaskDef(17, 20, TAMSIN_TABLE_SEAT, AIState.EATING, "Tamsin is home for the evening.",
                            unlock_doors=("tamsin_door",), facing=0.0, food_container="tamsin_table"),
                ], wake=7, bedtime=21),
            ),
        ),
        NPCDef(
            "Fenwick", FENWICK_BED, role="elder", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
            inventory=(key_item("fenwick_door", "Cottage Key"), ItemDef("pipe", "Clay Pipe", value=9.0, for_sale=False)),
            schedule=resident_tasks("Fenwick", "fenwick_door", FENWICK_BED, FENWICK_TABLE_SEAT, "fenwick_table", [
                visit(9, 12, INN_SEATS[2], AIState.DRINKING, "Fenwick has his usual seat at the Drowsy Dragon.", facing=90.0),
                stroll(12, 14, "Fenwick", STROLL_WEST, "Fenwick is walking off his ale."),
                TaskDef(14, 15, FENWICK_TABLE_SEAT, AIState.EATING, "Fenwick is having a bite at home.", unlock_doors=("fenwick_door",),
                        facing=0.0, food_container="fenwick_table"),
                stroll(15, 18, "Fenwick", STROLL_EAST, "Fenwick is ambling along the street."),
                visit(18, 20, INN_SEATS[2], AIState.DRINKING, "Fenwick is back at the inn for the evening.", facing=90.0),
            ], wake=8, bedtime=21),
            rota=(
                # a sober day: a line in the millstream by the bridge, his pipe on the square
                resident_tasks("Fenwick", "fenwick_door", FENWICK_BED, FENWICK_TABLE_SEAT, "fenwick_table", [
                    visit(9, 13, FISHING_SPOT, AIState.WORKING, "Fenwick is fishing in the millstream. Nothing is biting.",
                          facing=270.0),
                    TaskDef(13, 14, FENWICK_TABLE_SEAT, AIState.EATING, "Fenwick is having a bite at home.",
                            unlock_doors=("fenwick_door",), facing=0.0, food_container="fenwick_table"),
                    visit(14, 17, quarters.BERTRAM_SPOT, AIState.WORKING, "Fenwick is smoking his pipe on the bench in the square.",
                          facing=0.0),
                    stroll(17, 20, "Fenwick", STROLL_WEST, "Fenwick is taking a turn about the street."),
                ], wake=8, bedtime=21),
            ),
        ),
        NPCDef(
            "Lys", LYS_BED, role="cook", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
            inventory=(key_item("couple_door", "House Key"), ItemDef("ladle", "Wooden Ladle", value=3.0, for_sale=False)),
            schedule=resident_tasks("Lys", "couple_door", LYS_BED, COUPLE_TABLE_SEATS[0], "couple_table", [
                TaskDef(8, 14, INN_KITCHEN, AIState.WORKING, "Lys is helping in the inn kitchen.", facing=270.0,
                        depart_text="Off to the inn - Hilde will want the pots on."),
                visit(14, 15, SHOP_FLOOR, AIState.SHOPPING, "Lys is buying provisions at the store."),
                stroll(15, 17, "Lys", STROLL_EAST, "Lys is enjoying a walk in the afternoon sun."),
                TaskDef(17, 20, COUPLE_TABLE_SEATS[0], AIState.EATING, "Lys is cooking supper at home.", unlock_doors=("couple_door",),
                        facing=0.0, food_container="couple_table"),
            ], wake=7, bedtime=21),
            rota=(
                # the inn's late shift: the evening crowd wants feeding
                resident_tasks("Lys", "couple_door", LYS_BED, COUPLE_TABLE_SEATS[0], "couple_table", [
                    visit(8, 10, quarters.BAKERY_FLOOR, AIState.SHOPPING, "Lys is buying flour and gossip at the bakery."),
                    TaskDef(10, 12, COUPLE_TABLE_SEATS[0], AIState.WORKING, "Lys is mending at home.", unlock_doors=("couple_door",),
                            facing=0.0),
                    TaskDef(12, 20, INN_KITCHEN, AIState.WORKING, "Lys is cooking for the evening crowd at the inn.", facing=270.0,
                            depart_text="The late shift. Hilde's stew won't stir itself."),
                ], wake=7, supper=(20, 22)),
            ),
        ),
        NPCDef(
            "Dorran", DORRAN_BED, role="porter", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
            inventory=(key_item("couple_door", "House Key"), ItemDef("whetstone", "Whetstone", value=6.0, for_sale=False)),
            schedule=resident_tasks("Dorran", "couple_door", DORRAN_BED, COUPLE_TABLE_SEATS[1], "couple_table", facing=270.0, day=[
                TaskDef(8, 12, SHOP_PORTER, AIState.WORKING, "Dorran is hauling crates for Barnaby.", facing=270.0,
                        depart_text="Barnaby's crates won't carry themselves."),
                TaskDef(12, 13, COUPLE_TABLE_SEATS[1], AIState.EATING, "Dorran is home for his midday meal.", unlock_doors=("couple_door",),
                        facing=270.0, food_container="couple_table"),
                TaskDef(13, 17, SHOP_PORTER, AIState.WORKING, "Dorran is back at work in the store.", facing=270.0),
                visit(17, 19, INN_SEATS[3], AIState.DRINKING, "Dorran is having an ale after work.", facing=270.0),
                TaskDef(19, 20, COUPLE_TABLE_SEATS[1], AIState.EATING, "Dorran is home for supper.", unlock_doors=("couple_door",),
                        facing=270.0, food_container="couple_table"),
            ], wake=7, bedtime=21),
            rota=(
                # hauling for the cooper and the bakery instead, and home for supper
                resident_tasks("Dorran", "couple_door", DORRAN_BED, COUPLE_TABLE_SEATS[1], "couple_table", facing=270.0, day=[
                    TaskDef(8, 12, quarters.BARREL_ROLL[0], AIState.WORKING, "Dorran is loading barrels at the cooper's.",
                            facing=270.0, depart_text="Corwin's barrels today."),
                    TaskDef(12, 13, COUPLE_TABLE_SEATS[1], AIState.EATING, "Dorran is home for his midday meal.",
                            unlock_doors=("couple_door",), facing=270.0, food_container="couple_table"),
                    visit(13, 17, quarters.BAKERY_FLOOR, AIState.WORKING, "Dorran is carrying flour sacks for Wynn."),
                    TaskDef(17, 20, COUPLE_TABLE_SEATS[1], AIState.EATING, "Dorran is home, feet up.", unlock_doors=("couple_door",),
                            facing=270.0, food_container="couple_table"),
                ], wake=7, bedtime=21),
            ),
        ),
        NPCDef(
            "Pell", PELL_BED, role="lad", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
            inventory=(key_item("pell_door", "Cottage Key"), ItemDef("sling", "Leather Sling", value=7.0, for_sale=False, category="weapon")),
            schedule=resident_tasks("Pell", "pell_door", PELL_BED, PELL_TABLE_SEAT, "pell_table", [
                stroll(8, 11, "Pell", STROLL_EAST, "Pell is kicking about the street."),
                visit(11, 12, TAILOR_FLOOR, AIState.SHOPPING, "Pell is eyeing a new cloak he cannot afford."),
                TaskDef(12, 13, PELL_TABLE_SEAT, AIState.EATING, "Pell is home for a bite.", unlock_doors=("pell_door",),
                        facing=0.0, food_container="pell_table"),
                stroll(13, 17, "Pell", STROLL_WEST, "Pell is wandering down to the smithy and back."),
                visit(17, 20, INN_SEATS[5], AIState.DRINKING, "Pell is nursing a small ale at the inn.", facing=270.0),
            ], wake=7, supper=(20, 22)),                                  # (a long way home from the inn)
            rota=(
                # at the stable with Hobb, and watching the gate
                resident_tasks("Pell", "pell_door", PELL_BED, PELL_TABLE_SEAT, "pell_table", [
                    visit(8, 12, quarters.STALLS_HELP, AIState.WORKING, "Pell is helping Hobb muck out stalls with no horses in them.",
                          facing=0.0),
                    TaskDef(12, 13, PELL_TABLE_SEAT, AIState.EATING, "Pell is home for a bite.", unlock_doors=("pell_door",),
                            facing=0.0, food_container="pell_table"),
                    visit(13, 16, quarters.GATE_BENCH_SPOT, AIState.WORKING, "Pell is watching the road, dreaming of leaving.",
                          facing=180.0),
                    stroll(16, 20, "Pell", STROLL_EAST, "Pell is kicking about the street."),
                ], wake=7, bedtime=21),
            ),
        ),
        # ---- the rich: two old families on the hill, and everybody else beneath them -------------
        rich_resident("Lord Aurelian Vane", "vane", vane, LORD_OUTFIT,
                      [key_item("vane_door", "Vane Manor Key"), key_item("vane_strongbox", "Small Iron Key"),
                       ItemDef("gold_snuffbox", "Gold Snuffbox", value=140.0, for_sale=False)],
                      day=[
                          stroll(9, 11, "Lord Vane", PROMENADE, "Lord Vane is taking the air above the town."),
                          visit(11, 12, TAILOR_FLOOR, AIState.SHOPPING, "Lord Vane is finding fault with Mirelle's cloth."),
                          TaskDef(12, 14, vane["seat"], AIState.EATING, "Lord Vane is dining at home.", facing=0.0,
                                  food_container="vane_dining_table"),
                          stroll(14, 16, "Lord Vane", STROLL_EAST, "Lord Vane is inspecting the lower town, nose high."),
                          visit(16, 18, INN_SEATS[1], AIState.DRINKING, "Lord Vane is complaining about the wine at the inn.",
                                facing=90.0),
                          stroll(18, 20, "Lord Vane", PROMENADE, "Lord Vane is walking off his supper."),
                          TaskDef(20, 21, vane["study"], AIState.WORKING, "Lord Vane is at his papers in the study.", facing=0.0),
                      ], other_days=([
                          visit(9, 11, quarters.REEVE_DOORSTEP, AIState.WORKING, "Lord Vane is lecturing the Reeve about the tithe.",
                                facing=180.0),
                          stroll(11, 12, "Lord Vane", PROMENADE, "Lord Vane is taking the air above the town."),
                          TaskDef(12, 14, vane["seat"], AIState.EATING, "Lord Vane is dining at home.", facing=0.0,
                                  food_container="vane_dining_table"),
                          visit(14, 16, quarters.CHAPEL_PEWS[2], AIState.WORKING, "Lord Vane is in the front pew, being seen to pray.",
                                facing=180.0),
                          visit(16, 18, PARK_BENCH, AIState.WORKING, "Lord Vane is sneering at the park's pigeons.", facing=180.0),
                          stroll(18, 20, "Lord Vane", PROMENADE, "Lord Vane is walking off his supper."),
                          TaskDef(20, 21, vane["study"], AIState.WORKING, "Lord Vane is at his papers in the study.", facing=0.0),
                      ],), lines=(
                          "Do I know you? No. I thought not.",
                          "My family has held the hill since before Greyhold had a gate. The Ashgroves arrived last century. Trade.",
                          "The strongbox key never leaves my person. Greaves has a key to everything else. Greaves is above suspicion.",
                      )),
        rich_resident("Lady Seraphine Ashgrove", "ashgrove", ashgrove, LADY_OUTFIT,
                      [key_item("ashgrove_door", "Ashgrove Manor Key"), key_item("ashgrove_strongbox", "Small Silver Key"),
                       ItemDef("pearl_earrings", "Pearl Earrings", value=120.0, for_sale=False)],
                      day=[
                          visit(9, 10, SHOP_FLOOR, AIState.SHOPPING, "Lady Ashgrove is sneering at Barnaby's wares."),
                          stroll(10, 12, "Lady Ashgrove", PROMENADE, "Lady Ashgrove is promenading above the town."),
                          TaskDef(12, 14, ashgrove["seat"], AIState.EATING, "Lady Ashgrove is taking luncheon at home.", facing=0.0,
                                  food_container="ashgrove_dining_table"),
                          visit(14, 16, TAILOR_FLOOR, AIState.SHOPPING, "Lady Ashgrove is ordering Mirelle about."),
                          stroll(16, 18, "Lady Ashgrove", STROLL_WEST, "Lady Ashgrove is touring the lower town with a handkerchief."),
                          stroll(18, 20, "Lady Ashgrove", PROMENADE, "Lady Ashgrove is taking the evening air."),
                          TaskDef(20, 21, ashgrove["study"], AIState.WORKING, "Lady Ashgrove is at her letters in her study.",
                                  facing=0.0),
                      ], other_days=([
                          visit(9, 11, quarters.HERB_SHOP_FLOOR, AIState.SHOPPING, "Lady Ashgrove is buying lavender water from the herbwife."),
                          stroll(11, 12, "Lady Ashgrove", PROMENADE, "Lady Ashgrove is promenading above the town."),
                          TaskDef(12, 14, ashgrove["seat"], AIState.EATING, "Lady Ashgrove is taking luncheon at home.", facing=0.0,
                                  food_container="ashgrove_dining_table"),
                          visit(14, 16, quarters.CHAPEL_PEWS[0], AIState.WORKING, "Lady Ashgrove is at prayer, audibly.", facing=180.0),
                          visit(16, 18, quarters.BAKERY_FLOOR, AIState.SHOPPING, "Lady Ashgrove is sending back a loaf."),
                          stroll(18, 20, "Lady Ashgrove", PROMENADE, "Lady Ashgrove is taking the evening air."),
                          TaskDef(20, 21, ashgrove["study"], AIState.WORKING, "Lady Ashgrove is at her letters in her study.",
                                  facing=0.0),
                      ],), lines=(
                          "You are standing in my light. Move.",
                          "Lord Vane and I have not spoken in eleven years. He knows why. Everybody knows why.",
                          "My grandmother's gloves - the Magpie's Fingers, the vulgar call them - are not for sale. Not at any price.",
                      )),
        servant("Greaves", "butler", "vane", vane, BUTLER_OUTFIT, errands=[
            visit(9, 11, SHOP_FLOOR, AIState.SHOPPING, "Greaves is buying for the Vane household, very particularly."),
            visit(17, 19, INN_KITCHEN, AIState.WORKING, "Greaves is seeing Hilde about the wine, and staying for a glass."),
        ], other_errands=([
            visit(9, 10, quarters.REEVE_DOORSTEP, AIState.WORKING, "Greaves is settling the household's dues with the Reeve."),
            visit(17, 19, quarters.COOPER_YARD, AIState.WORKING, "Greaves is inspecting a new wine cask at the cooper's."),
        ],), lines=(
            "His lordship is not receiving. His lordship is never receiving.",
            "Fifteen years I have kept this house. Fifteen years of accounts, balanced to the penny. Mostly balanced.",
            "The strongbox? His lordship keeps its key on a chain about his neck. He sleeps in it. Naturally.",
        )),
        servant("Ottilie", "maid", "ashgrove", ashgrove, MAID_OUTFIT, errands=[
            visit(9, 11, (-47.0, 0.0, 27.5), AIState.WORKING, "Ottilie is taking her ladyship's linen down to Nessa."),
            visit(17, 19, quarters.PARK_MEETING[1], AIState.WORKING, "Ottilie is 'taking the air' in the park. Hobb is there. Fancy.",
                  facing=270.0),
        ], other_errands=([
            visit(9, 10, quarters.HERB_SHOP_FLOOR, AIState.SHOPPING, "Ottilie is fetching lavender for her ladyship's pillows."),
            visit(17, 19, quarters.CHAPEL_PEWS[1], AIState.WORKING, "Ottilie is at the chapel, praying for Hobb. Or about him.",
                  facing=180.0),
        ],), lines=(
            "Her ladyship changes her gowns four times a day. I wash them five.",
            "Don't tell anyone, but Hobb at the stable is ever so handsome. He writes me letters. I keep them under the kitchen floor.",
            "Her ladyship wears the strongbox key on a ribbon. She sleeps like the dead, mind - eleven till eight, and never stirs.",
        )),
        # ---- the watch -------------------------------------------------------------------------
        # (the watch eats at the guard house, up the hill: every meal comes before a shift there or
        # after a watch at its door - never straight after a round that may end at the far wall)
        guard("Guard Captain Vance", patrol=(6, 14), meal=(5, 6), post=(14, 22), post_spot=POST_WEST, post_facing=90.0,
              sleep=(22, 5), bed_spot=GUARD_BEDS[0], meal_seat=GUARD_TABLE_SEATS[0]),
        # the tower watch: up the winding stair at dawn, down for a meal, back up until bedtime
        NPCDef(
            "Guard Aldric", GUARD_BEDS[3], is_guard=True, inventory=(GUARD_KEY, TOWER_KEY, RATIONS), inventory_reset_days=1, gold=GUARD_GOLD,
            role="guard",
            schedule=(
                TaskDef(0, 6, GUARD_BEDS[3], AIState.SLEEPING, "Guard Aldric is asleep in the guard house."),
                TaskDef(6, 7, GUARD_TABLE_SEATS[1], AIState.EATING, "Guard Aldric is eating at the guard house table.",
                        facing=270.0, food_container="guard_table"),
                TaskDef(7, 14, TOWER_TOP, AIState.GUARDING, "Guard Aldric is on watch at the top of the tower.", facing=180.0,
                        depart_text="Up the tower. Forty steps, every morning."),
                TaskDef(14, 15, GUARD_TABLE_SEATS[1], AIState.EATING, "Guard Aldric is eating at the guard house table.",
                        facing=270.0, food_container="guard_table"),
                TaskDef(15, 22, TOWER_TOP, AIState.GUARDING, "Guard Aldric is back on watch up the tower.", facing=180.0),
                TaskDef(22, 24, GUARD_BEDS[3], AIState.SLEEPING, "Guard Aldric is asleep in the guard house."),
            ),
        ),
        guard("Guard Orrin", patrol=(14, 22), meal=(13, 14), post=(6, 13), post_spot=POST_DOOR,
              sleep=(22, 6), bed_spot=GUARD_BEDS[1], meal_seat=GUARD_TABLE_SEATS[1]),
        guard("Guard Mira", patrol=(22, 6), meal=(21, 22), post=(14, 21), post_spot=POST_DOOR,
              sleep=(6, 14), bed_spot=GUARD_BEDS[2], meal_seat=GUARD_TABLE_SEATS[2]),
        # the gate wardens: a long watch each, changing sides every few hours
        gate_guard("Guard Bram", duty=(6, 18), first_post=0, meal=18, evening=(19, 21), sleep=(21, 6), bed_spot=GATE_LODGE_BEDS[0],
                   meal_seat=GATE_LODGE_SEAT),
        gate_guard("Guard Wenna", duty=(18, 6), first_post=0, meal=6, evening=(14, 16), supper=(16, 18), sleep=(7, 14),
                   bed_spot=GATE_LODGE_BEDS[1], meal_seat=GATE_LODGE_SEAT),
        NPCDef(
            "Jailer Hesk", JAIL_GUARD_BED, is_guard=True, inventory_reset_days=1, gold=GUARD_GOLD, role="jailer",
            inventory=(GUARD_KEY, ItemDef("cell_key", "Cell Key", value=1.0, is_key=True, key_door_id="cell_door", for_sale=False,
                                          category="key"), RATIONS),
            schedule=(
                TaskDef(0, 6, JAIL_GUARD_BED, AIState.SLEEPING, "Hesk is asleep in the jail office."),
                TaskDef(6, 7, (51.5, 0.0, 8.5), AIState.EATING, "Hesk is having breakfast at the jail.", facing=0.0),
                TaskDef(7, 22, JAIL_POST, AIState.GUARDING, "Hesk is keeping watch over the cells.", facing=135.0),
                TaskDef(22, 24, JAIL_GUARD_BED, AIState.SLEEPING, "Hesk turned in for the night."),
            ),
        ),
        *_quarters("npcs"),
        *wilds.NPCS,
    )

    readables: List[ReadableDef] = []

    lights = [
        LightDef("fire", (-60.6, 1.1, -10.55), room="the smith's yard", range=9.0),      # the forge's coals
        *WATCHTOWER.lights,
        *ashgrove["lights"], *vane["lights"],
        # torches by the doors that matter after dark ...
        *torches_beside_door(-34.0, -31.0, -6.0, 180.0, "the inn"),
        *torches_beside_door(-15.0, -12.0, -6.0, 180.0, "the shop"),
        *torches_beside_door(10.0, 13.0, -6.0, 180.0, "the tailor's"),
        *torches_beside_door(48.5, 51.5, GUARD_HOUSE[3], 180.0, "the guard house"),
        *torches_beside_door(51.5, 54.5, 6.0, 0.0, "the jail"),
        wall_torch(-68.0, -6.0, 180.0, "the smithy"), wall_torch(-66.0, 11.0, 90.0, "Barnaby's house"),
        wall_torch(-13.5 - 0.6, 7.5, 0.0, "the hideout"), wall_torch(17.0, 11.5, 90.0, "Lys and Dorran's house"),
        wall_torch(-30.0, 9.5, 90.0, "Fenwick's cottage"),
        wall_torch(34.0, -12.0, 90.0, "the old granary"), wall_torch(-22.0, -10.0, 90.0, "the inn"),
        # ... and along the curtain wall and at the gate
        *(wall_torch(x, TOWN_Z1 + TOWN_WALL, 180.0, "the town wall") for x in (-60.0, -20.0, 30.0, 66.0)),
        *(wall_torch(x, TOWN_Z2 - TOWN_WALL, 0.0, "the town wall") for x in (-66.0, -30.0, 30.0, 62.0)),
        *(wall_torch(TOWN_X1 + TOWN_WALL, z, 90.0, "the town wall") for z in (-40.0, 0.0, 36.0)),
        *(wall_torch(TOWN_X2 - TOWN_WALL, z, 270.0, "the town wall") for z in (-40.0, -10.0, 36.0)),
        wall_torch(-7.0, TOWN_Z2 - 1.5, 270.0, "the town gate", y=3.2), wall_torch(7.0, TOWN_Z2 - 1.5, 90.0, "the town gate", y=3.2),
        wall_torch(12.6, GATE_LODGE[1], 0.0, "the gate lodge"),
        wall_torch(-5.5, TOWN_Z2 + 1.5, 180.0, "the town gate", y=3.2), wall_torch(5.5, TOWN_Z2 + 1.5, 180.0, "the town gate", y=3.2),
        *_quarters("lights"),
        *wilds.LIGHTS,
    ]

    return LevelDefinition(
        name="town",
        walls=tuple(walls) + _quarters("walls") + wilds.WALLS,
        doors=doors,
        npcs=stories.with_stories(npcs),
        player_spawn=PLAYER_SPAWN,
        props=tuple(props),
        containers=tuple(containers),
        stations=stations,
        areas=areas,
        buildings=buildings + _quarters("buildings") + wilds.BUILDINGS,
        roads=roads,
        lights=tuple(lights),
        stairs=(WATCHTOWER.stair, ashgrove["stair"], vane["stair"], *wilds.STAIRS),
        jail=JailDef(cell_door_id="cell_door", storage="jail_storage", cell_area="the jail cell", jail_area="the jail",
                     cell_spot=JAIL_CELL_SPOT, jailer="Jailer Hesk", post=JAIL_POST),
        inn=InnDef(keeper="Hilde the Innkeeper", rooms=(
            InnRoomDef("room one", "inn_room_1_door", INN_ROOM_BEDS[0]),
            InnRoomDef("room two", "inn_room_2_door", INN_ROOM_BEDS[1]),
            InnRoomDef("room three", "inn_room_3_door", INN_ROOM_BEDS[2]),
        )),
        bounds=wilds.BOUNDS,
        ground_half_size=60.0,
        ground_texture_repeat=12.0,
        player_gold=STARTING_GOLD,
        player_inventory=(
            ItemDef("iron_lockpick", "Iron Lockpick", value=5.0, noise_modifier=0.0, is_lockpick=True, stackable=True, category="tool"),
            ItemDef("soft_shoes", "Soft Leather Boots", value=30.0, category="clothing", equipped=True),
            ItemDef("silver_goblet", "Silver Goblet", value=120.0, is_stolen=True),
        ),
        default_area="the streets",
        intro_message="Tutorial started: Explore merchants, trade wares, and open doors!",
        terrain=town_terrain(),
        town_gate="town_gate",
        town_area=(TOWN_X1, TOWN_Z1, TOWN_X2, TOWN_Z2),
        readables=tuple(readables) + wilds.READABLES,
    )
