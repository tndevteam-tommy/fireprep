"""More of Emberhollow: the chapel of Saint Wendel with its steeple, the bakery, the cooper's
house and yard, Old Bertram's cottage, the washerwoman's by the millstream - and the people
who live in them, each with a trade, a habit, a story to tell and something put by where they
think nobody will look.

    the chapel (South Lane, west): a nave under a steep roof, the steeple over its door - a
        belfry and a spire - the vestry behind, graves in the yard. Sister Maude keeps it
    Nessa's cottage (South Lane's west end, by the millstream): she washes half the town's
        linen in the stream and knows all of its secrets
    Old Bertram's cottage (South Lane's east end): the town crier, once of the watch, until he
        took an arrow in the knee
    Wynn's bakery (Market Street, south side, east): up at four, bread by seven
    Corwin's house and cooper's yard (Market Street, north side, east): oak, iron, no nonsense
    Hobb sleeps in the stable and grooms horses that were sold long ago

Everything is plain level data merged into the town (``town.py``), in the town's own frame.
"""
from __future__ import annotations

from ...entities import AIState, Outfit
from ..builders import (WALL, barrels, bed, chest, coins, fence, house_walls, key_item, resident_tasks, stash, stroll, table,
                        visit, wall_torch, wall_windows, woodpile)
from ..level import AreaDef, BuildingDef, ContainerDef, DoorDef, ItemDef, LightDef, NPCDef, PropDef, RoadDef, TaskDef, WallDef
from .wilds import Part

FLIMSY, SIMPLE, STURDY, STRONG, MASTER, LEGENDARY = range(6)
TOWNSFOLK_GOLD = 30.0

# -- the chapel: the nave, the steeple over the door, the vestry to the east
NAVE = (-29.0, 36.5, -19.0, 50.5)
STEEPLE = (-26.5, 32.5, -21.5, 36.5)
VESTRY = (-19.0, 43.0, -14.0, 50.5)
CHAPEL_DOOR_X = -25.5
CHAPEL_HOURS = (6, 21)                      # the chapel door's open hours (unlocked all day, whoever is in)
VESTRY_WAY = (46.5, 49.0)                   # the door from the nave into the vestry (in the nave's east wall)
NAVE_HEIGHT, STEEPLE_HEIGHT = 6.0, 15.5     # (the belfry stands clear over the nave's ridge, open on all four sides)
MAUDE_BED, MAUDE_SEAT = (-15.55, 0.0, 48.4), (-16.5, 0.0, 45.7)
ALTAR_SPOT = (-24.0, 0.0, 46.8)             # before the altar, facing it
BELL_ROPE = (-24.0, 0.0, 34.6)              # in the base of the steeple, the rope in her hands
GRAVEYARD = ((-35.75, 0.0, 38.0), (-35.75, 0.0, 48.8), (-33.25, 0.0, 48.8), (-33.25, 0.0, 38.0))
CHAPEL_PEWS = ((-24.0, 0.0, 40.2), (-24.0, 0.0, 42.0), (-24.0, 0.0, 43.8))   # in the aisle between the pews, facing the altar

# -- Old Bertram's cottage at South Lane's east end
BERTRAM_HOUSE = (30.0, 33.0, 37.0, 40.0)
BERTRAM_DOORSTEP = (33.5, 0.0, 31.6)         # outside his door (a caller's spot)
BERTRAM_BED, BERTRAM_SEAT = (35.45, 0.0, 38.2), (32.0, 0.0, 37.4)
CRIER_SPOT = (4.2, 0.0, 7.2)                # by the well, where the news is cried
BERTRAM_BENCH = (-8.3, 0.0, 7.1)            # the bench on the square ...
BERTRAM_SPOT = (-8.3, 0.0, 5.9)             # ... and Bertram before it
GATE_BENCH = (-9.0, 0.0, 49.6)              # a bench by the gate ...
GATE_BENCH_SPOT = (-9.0, 0.0, 50.7)         # ... where Bertram tells the watch how it was done in his day

# -- Nessa the washerwoman by the millstream
NESSA_HOUSE = (-46.0, 33.0, -39.0, 40.0)
NESSA_BED, NESSA_SEAT = (-44.45, 0.0, 38.2), (-41.2, 0.0, 36.4)
WASHING_SPOT = (-47.6, 0.0, 28.0)           # at the stream's bank, scrubbing
WASHING_LINE = (-44.0, 0.0, 42.3)
NESSA_GOSSIP = (-41.5, 0.0, 41.3)           # a neighbour helping at the line (and hearing the news)

# -- Wynn's bakery on Market Street
BAKERY = (33.0, 6.0, 43.0, 14.0)
BAKERY_DOOR_X = 36.5
BAKERY_COUNTER = (38.4, 0.0, 10.1)          # behind the counter, facing the door
BAKERY_FLOOR = (38.0, 0.0, 7.6)             # where a customer stands
BAKERY_OVEN = (34.9, 0.0, 11.8)
WYNN_BED, WYNN_SEAT = (41.45, 0.0, 12.6), (38.0, 0.0, 10.8)
BREAD_ROUND = ((-34.0, 0.0, -4.2), (2.0, 0.0, -3.0), (48.5, 0.0, -40.5), (30.0, 0.0, -39.0), (37.0, 0.0, -10.0))

# -- Corwin the cooper: his house, and his yard and shed west of it
COOPER_HOUSE = (62.0, -22.0, 72.0, -13.0)
COOPER_DOOR_X = 64.0
COOPER_SHED = (52.0, -20.0, 60.0, -13.0)
CORWIN_BED, CORWIN_SEAT = (70.45, 0.0, -20.3), (67.0, 0.0, -17.0)
COOPER_BENCH = (56.0, 0.0, -14.9)
COOPER_YARD = (57.0, 0.0, -11.0)            # in the yard by the shed, where a buyer looks a barrel over
BARREL_ROLL = ((60.0, 0.0, -7.9), (37.0, 0.0, -3.5), (-21.0, 0.0, -4.0), (-19.9, 0.0, -8.5), (37.0, 0.0, -3.5))

# -- Hobb in the stable
HOBB_BED, HOBB_SEAT = (22.4, 0.0, 42.8), (21.0, 0.0, 36.9)
PARK_MEETING = ((-4.6, 0.0, -47.2), (-3.2, 0.0, -47.2))     # Hobb and Ottilie, by the park's bench of an evening
STALLS = (17.0, 0.0, 41.0)
CHAPEL_STEPS = (-23.0, 0.0, 30.8)            # outside the chapel's door
STALLS_HELP = (17.5, 0.0, 38.5)             # a second pair of hands at the stalls


def _maude_afternoon() -> list:
    """Sister Maude's day from her dinner in the vestry (after the morning's call) to vespers."""
    return [
        TaskDef(11, 13, MAUDE_SEAT, AIState.EATING, "Sister Maude is eating in the vestry.", facing=180.0, food_container="maude_table"),
        TaskDef(13, 17, ALTAR_SPOT, AIState.WORKING, "Sister Maude is keeping the chapel.", facing=180.0),
        stroll(17, 18, "Sister Maude", ((-24.0, 0.0, 30.0), (-3.0, 0.0, 5.0), (20.0, 0.0, 3.0), (-24.0, 0.0, 30.0)),
               "Sister Maude is going her rounds with the alms bag."),
        TaskDef(18, 19, BELL_ROPE, AIState.WORKING, "Sister Maude is ringing the evening bell.", facing=0.0),
        TaskDef(19, 20, ALTAR_SPOT, AIState.WORKING, "Sister Maude is saying vespers to an empty chapel.", facing=180.0),
    ]


def chapel() -> Part:
    """Saint Wendel's chapel: a steep-roofed nave of grey stone, lancet windows down its sides, a
    steeple over the door with a belfry and a spire; inside pews, the altar, the poor box; the
    vestry where Sister Maude lives; graves in the yard to the west."""
    p = Part()
    nx1, nz1, nx2, nz2 = NAVE
    sx1, sz1, sx2, sz2 = STEEPLE
    vx1, vz1, vx2, vz2 = VESTRY
    # the nave: its north wall has the way through from the steeple's base
    p.walls += [
        WallDef((nx1, 0, nz1), (nx1 + WALL, NAVE_HEIGHT, nz2), texture="stone"),
        WallDef((nx2 - WALL, 0, nz1), (nx2, NAVE_HEIGHT, VESTRY_WAY[0]), texture="stone"),
        WallDef((nx2 - WALL, 0, VESTRY_WAY[1]), (nx2, NAVE_HEIGHT, nz2), texture="stone"),     # (a doorway to the vestry)
        WallDef((nx2 - WALL, 3.2, VESTRY_WAY[0]), (nx2, NAVE_HEIGHT, VESTRY_WAY[1]), texture="stone"),
        WallDef((nx1 + WALL, 0, nz2 - WALL), (nx2 - WALL, NAVE_HEIGHT, nz2), texture="stone"),
        WallDef((nx1 + WALL, 0, nz1), (sx1, NAVE_HEIGHT, nz1 + WALL), texture="stone"),
        WallDef((sx2, 0, nz1), (nx2 - WALL, NAVE_HEIGHT, nz1 + WALL), texture="stone"),
    ]
    # the steeple: its walls up to the belfry, open arches there, then the spire (the building's roof)
    belfry = STEEPLE_HEIGHT - 3.5
    p.walls += [
        WallDef((sx1, 0, sz1), (sx1 + WALL, belfry, sz2), texture="stone"),
        WallDef((sx2 - WALL, 0, sz1), (sx2, belfry, sz2), texture="stone"),
        WallDef((sx1 + WALL, 0, sz1), (CHAPEL_DOOR_X, belfry, sz1 + WALL), texture="stone"),
        WallDef((CHAPEL_DOOR_X + 3.0, 0, sz1), (sx2 - WALL, belfry, sz1 + WALL), texture="stone"),
        WallDef((CHAPEL_DOOR_X, 3.6, sz1), (CHAPEL_DOOR_X + 3.0, belfry, sz1 + WALL), texture="stone"),
        # the belfry: corner piers and the arches' heads, the spire sits on them
        *(WallDef((x, belfry, z), (x + 0.8, STEEPLE_HEIGHT, z + 0.8), texture="stone")
          for x, z in ((sx1, sz1), (sx2 - 0.8, sz1), (sx1, sz2 - 0.8), (sx2 - 0.8, sz2 - 0.8))),
        WallDef((sx1, STEEPLE_HEIGHT - 0.7, sz1), (sx2, STEEPLE_HEIGHT, sz1 + 0.5), texture="stone"),
        WallDef((sx1, STEEPLE_HEIGHT - 0.7, sz2 - 0.5), (sx2, STEEPLE_HEIGHT, sz2), texture="stone"),
        WallDef((sx1, STEEPLE_HEIGHT - 0.7, sz1 + 0.5), (sx1 + 0.5, STEEPLE_HEIGHT, sz2 - 0.5), texture="stone"),
        WallDef((sx2 - 0.5, STEEPLE_HEIGHT - 0.7, sz1 + 0.5), (sx2, STEEPLE_HEIGHT, sz2 - 0.5), texture="stone"),
        WallDef((sx1, belfry - 0.3, sz1), (sx2, belfry, sz2), texture="stone"),           # the belfry's floor (the porch's ceiling)
    ]
    # buttresses down the nave's sides
    for z in (39.5, 43.5, 47.5):
        p.walls += [WallDef((nx1 - 0.6, 0, z - 0.3), (nx1, 3.8, z + 0.3), texture="stone")]
        if z < vz1:
            p.walls += [WallDef((nx2, 0, z - 0.3), (nx2 + 0.6, 3.8, z + 0.3), texture="stone")]
    # the vestry: a low stone room against the nave
    p.walls += [
        WallDef((vx2 - WALL, 0, vz1), (vx2, 3.6, vz2), texture="stone"),
        WallDef((nx2, 0, vz1), (vx2 - WALL, 3.6, vz1 + WALL), texture="stone"),
        WallDef((nx2, 0, vz2 - WALL), (vx2 - WALL, 3.6, vz2), texture="stone"),
    ]
    p.buildings += [
        BuildingDef("chapel", (nx1, nz1), (nx2, nz2), wall_height=NAVE_HEIGHT, roof_height=5.6, roof_texture="roof_slate",
                    floor_texture="stone", lit=False),                                  # (its own lights: see below)
        BuildingDef("chapel_steeple", (sx1, sz1), (sx2, sz2), wall_height=STEEPLE_HEIGHT, roof_height=8.5, roof_texture="roof_slate",
                    floor_texture="stone", roof_style="spire", pad_with="chapel", lit=False),
        # (the vestry starts at the nave's outer face: its eaves would otherwise reach into the nave)
        BuildingDef("chapel_vestry", (nx2, vz1), (vx2, vz2), wall_height=3.6, roof_height=1.6, roof_texture="roof_slate",
                    floor_texture="wood", pad_with="chapel", lit=False),
    ]
    # the floor through the ways from the steeple into the nave and from the nave into the vestry
    # (each building's floor stops short of its walls: the ground would show in between)
    inset = WALL - 0.02
    p.props += [
        PropDef("chapel_threshold_steeple", (sx1 + 0.01, 0.0, sz2 - inset), (sx2 - 0.01, 0.05, nz1 + inset),
                texture="stone", kind="floor", solid=False),
        # (along the vestry's west side, where its wall is the nave's - either side of the door, whose
        # own threshold fills the doorway: ``World._thresholds``)
        # (each stops where the door's threshold - running a few millimetres into its jambs - begins)
        PropDef("chapel_vestry_boards", (nx2, 0.0, vz1 + inset), (nx2 + inset, 0.05, VESTRY_WAY[0] - 0.005),
                texture="wood", kind="floor", solid=False),
        PropDef("chapel_vestry_boards_south", (nx2, 0.0, VESTRY_WAY[1] + 0.005), (nx2 + inset, 0.05, vz2 - inset),
                texture="wood", kind="floor", solid=False),
    ]
    # the vestry's door: in the nave's east wall (a door on a north-south line), into Maude's room
    p.doors += [DoorDef("vestry_door", (nx2, 0.0, VESTRY_WAY[0]), axis="z", width=VESTRY_WAY[1] - VESTRY_WAY[0], height=3.2,
                        is_locked=False, key_id="vestry_door_key", owners=("Sister Maude",), lock_difficulty=SIMPLE,
                        label="The Vestry")]
    # (open to all from the morning bell to nightfall, whether Sister Maude is in or out - it is
    # God's house, not hers: the door's schedule unlocks it for the day and locks it for the night)
    p.doors += [DoorDef("chapel_door", (CHAPEL_DOOR_X, 0.0, sz1), is_locked=False, key_id="chapel_door_key", swing_out=False,
                        owners=("Sister Maude",), label="Chapel of Saint Wendel", open_hour=CHAPEL_HOURS[0],
                        close_hour=CHAPEL_HOURS[1], open_message="The doors of Saint Wendel's chapel stand open for the day.",
                        close_message="Saint Wendel's chapel is locked for the night.")]
    p.props += [
        # two ranks of pews, an aisle down the middle and one down either side
        *(PropDef(f"chapel_pew_{i}", (-27.7, 0, 39.0 + i * 1.7), (-25.2, 0.5, 39.6 + i * 1.7), texture="wood") for i in range(4)),
        *(PropDef(f"chapel_pew_{i + 4}", (-22.8, 0, 39.0 + i * 1.7), (-20.3, 0.5, 39.6 + i * 1.7), texture="wood") for i in range(4)),
        PropDef("chapel_altar", (-25.6, 0, 47.8), (-22.4, 1.0, 48.7), texture="marble"),
        PropDef("chapel_rug", (-25.0, 0.0, 38.0), (-23.0, 0.05, 47.4), texture="rug", kind="rug", solid=False),
        PropDef("chapel_bell", (-24.9, STEEPLE_HEIGHT - 2.5, 33.6), (-23.1, STEEPLE_HEIGHT - 0.5, 35.4), texture="gilt", kind="bell", solid=False),
        *(PropDef(f"chapel_lancet_{i}", (x, 2.0, z - 0.55), (x + 0.02, 4.6, z + 0.55), texture="window_lancet", kind="window",
                  facing=facing, solid=False)
          for i, (x, z, facing) in enumerate(((nx1, 41.5, 270.0), (nx1, 45.5, 270.0), (nx2, 39.5, 90.0), (nx1, 37.8, 270.0)))),
        PropDef("chapel_lancet_east", (-24.6, 2.4, nz2), (-23.4, 5.4, nz2 + 0.02), texture="window_lancet", kind="window", facing=180.0,
                solid=False),
        # (and the same glass seen from within)
        *(PropDef(f"chapel_lancet_in_{i}", (x - 0.02, 2.0, z - 0.55), (x, 4.6, z + 0.55), texture="window_lancet", kind="window",
                  facing=facing, solid=False)
          for i, (x, z, facing) in enumerate(((nx1 + WALL + 0.02, 41.5, 90.0), (nx1 + WALL + 0.02, 45.5, 90.0),
                                               (nx2 - WALL, 39.5, 270.0), (nx1 + WALL + 0.02, 37.8, 90.0)))),
        PropDef("chapel_lancet_east_in", (-24.6, 2.4, nz2 - WALL - 0.02), (-23.4, 5.4, nz2 - WALL), texture="window_lancet",
                kind="window", facing=0.0, solid=False),
        PropDef("chapel_lancet_steeple", (-24.6, 4.6, sz1 - 0.02), (-23.4, 7.4, sz1), texture="window_lancet", kind="window",
                facing=0.0, solid=False),
        bed("maude_bed", MAUDE_BED, facing=90.0),
        # the graveyard to the west, a low fence round it
        *(PropDef(f"grave_{i}", (x - 0.35, 0.0, z - 0.1), (x + 0.35, 1.0, z + 1.6), texture="stone", kind="headstone", facing=0.0)
          for i, (x, z) in enumerate(((-37.0, 38.5), (-34.5, 38.5), (-32.0, 38.8), (-37.0, 42.5), (-34.5, 42.7), (-37.2, 46.5),
                                      (-34.6, 46.4), (-32.0, 46.6)))),
        fence("graveyard_fence_n", -38.6, 36.0, -35.6, 36.2), fence("graveyard_fence_ne", -33.2, 36.0, -30.4, 36.2),
        fence("graveyard_fence_w", -38.8, 36.2, -38.6, 50.5),
        fence("graveyard_fence_s", -38.6, 50.5, -30.4, 50.7),
    ]
    p.containers += [
        chest("chapel_poor_box", "Poor Box", (-20.2, 0.0, 37.4), owner="the chapel", locked=True, lock_difficulty=SIMPLE,
              key_id="chapel_door", items=[coins(18)]),
        table("maude_table", "Table", (-16.5, 0.0, 44.6), owner="Sister Maude", size=(0.6, 0.45),
              items=[ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=11, is_food=True, category="food"),
                     ItemDef("herbs", "Bundle of Herbs", value=2.0, stackable=True, quantity=3)]),
        # the relic Saint Wendel brought back from the Deepwood, kept locked in the vestry
        chest("chapel_reliquary", "Reliquary", (-14.9, 0.0, 44.4), owner="the chapel", along_x=False, locked=True,
              lock_difficulty=STRONG, key_id="chapel_door",
              items=[ItemDef("wendel_reliquary", "Silver Reliquary of Saint Wendel", value=320.0), coins(40)]),
    ]
    p.lights += [
        LightDef("candle", (-24.0, 1.05, 48.25), room="the chapel", range=6.0),
        LightDef("lantern", (nx1 + 0.8, 2.9, 44.0), facing=90.0, room="the chapel", range=8.0),
        LightDef("lantern", (nx2 - 0.8, 2.9, 41.0), facing=270.0, room="the chapel", range=8.0),
        LightDef("lantern", (sx1 + 0.75, 2.9, (sz1 + sz2) / 2.0), facing=90.0, room="the chapel porch", range=6.0),
        LightDef("lantern", (vx2 - 0.75, 2.9, 46.75), facing=270.0, room="the vestry", range=7.0),
        wall_torch(CHAPEL_DOOR_X - 0.6, sz1, 0.0, "the chapel"), wall_torch(CHAPEL_DOOR_X + 3.6, sz1, 0.0, "the chapel"),
    ]
    p.areas += [
        AreaDef("the vestry", (nx2, vz1), (vx2, vz2), owner="Sister Maude", restricted=True),
        AreaDef("the chapel", (nx1, sz1), (nx2, nz2)),
        AreaDef("the graveyard", (-38.8, 36.0), (nx1, 50.7)),
    ]
    p.npcs += [
        NPCDef("Sister Maude", MAUDE_BED, role="priest", gold=12.0, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.18, 0.18, 0.22), legs=(0.15, 0.15, 0.18), hat="scarf", hat_color=(0.92, 0.92, 0.92),
                             hair=(0.6, 0.55, 0.5)),
               inventory=(key_item("chapel_door", "Chapel Key"), key_item("vestry_door", "Vestry Key"),
                          ItemDef("rosary", "Wooden Rosary", value=4.0, for_sale=False)),
               lines=("Saint Wendel walked the Deepwood forty days and came out with nothing, and was content. The shrine's at the end of the south road.",
                      "The poor box is for the poor. You're not poor. You're just untidy.",
                      "Jae came to confession once. It took four hours. I still think about some of it.",
                      "The Saint's own reliquary is kept in the vestry. Locked. Not that anybody in Emberhollow would. Would they."),
               schedule=resident_tasks("Sister Maude", "vestry_door", MAUDE_BED, MAUDE_SEAT, "maude_table", [
                   TaskDef(6, 7, BELL_ROPE, AIState.WORKING, "Sister Maude is ringing the morning bell.", facing=0.0,
                           depart_text="The bell, the bell."),
                   TaskDef(7, 9, ALTAR_SPOT, AIState.WORKING, "Sister Maude is at her prayers before the altar.", facing=180.0),
                   TaskDef(9, 10, GRAVEYARD[0], AIState.CLEANING, "Sister Maude is tending the graves.", patrol_points=GRAVEYARD),
                   visit(10, 11, (-58.5, 0.0, 4.5), AIState.WORKING, "Sister Maude is calling on the widow Tamsin."),
                   *_maude_afternoon(),
               ], wake=5, bedtime=21, facing=180.0),
               # (the other day she calls on Old Bertram and his knee instead, by way of the square)
               rota=(resident_tasks("Sister Maude", "vestry_door", MAUDE_BED, MAUDE_SEAT, "maude_table", [
                   TaskDef(6, 7, BELL_ROPE, AIState.WORKING, "Sister Maude is ringing the morning bell.", facing=0.0,
                           depart_text="The bell, the bell."),
                   TaskDef(7, 9, ALTAR_SPOT, AIState.WORKING, "Sister Maude is at her prayers before the altar.", facing=180.0),
                   visit(9, 11, BERTRAM_DOORSTEP, AIState.WORKING, "Sister Maude is calling on Old Bertram, and his knee."),
                   *_maude_afternoon(),
               ], wake=5, bedtime=21, facing=180.0),)),
    ]
    return p


def cottages() -> Part:
    """Bertram's and Nessa's cottages at the two ends of South Lane."""
    p = Part()
    bx1, bz1, bx2, bz2 = BERTRAM_HOUSE
    nx1, nz1, nx2, nz2 = NESSA_HOUSE
    p.walls += house_walls(bx1, bz1, bx2, bz2, 32.0, "north", texture="plaster")
    p.walls += house_walls(nx1, nz1, nx2, nz2, -44.0, "north", texture="plaster_timber")
    p.buildings += [
        BuildingDef("bertram_house", (bx1, bz1), (bx2, bz2), roof_texture="thatch"),
        BuildingDef("nessa_house", (nx1, nz1), (nx2, nz2), roof_texture="thatch"),
    ]
    p.doors += [
        DoorDef("bertram_door", (32.0, 0.0, bz1), is_locked=True, key_id="bertram_door_key", owners=("Old Bertram",),
                lock_difficulty=SIMPLE, label="Old Bertram's Cottage"),
        DoorDef("nessa_door", (-44.0, 0.0, nz1), is_locked=True, key_id="nessa_door_key", owners=("Nessa",), lock_difficulty=SIMPLE,
                label="Nessa's Cottage"),
    ]
    p.props += [
        bed("bertram_bed", BERTRAM_BED, facing=90.0), bed("nessa_bed", NESSA_BED, facing=270.0),
        PropDef("square_bench", (BERTRAM_BENCH[0] - 1.3, 0.0, BERTRAM_BENCH[2] - 0.3), (BERTRAM_BENCH[0] + 1.3, 0.5, BERTRAM_BENCH[2] + 0.3),
                texture="wood", kind="bench"),
        PropDef("gate_bench", (GATE_BENCH[0] - 1.3, 0.0, GATE_BENCH[2] - 0.3), (GATE_BENCH[0] + 1.3, 0.5, GATE_BENCH[2] + 0.3),
                texture="wood", kind="bench"),
        PropDef("nessa_washing_line", (-46.5, 0.0, WASHING_LINE[2] - 0.05), (-39.5, 2.0, WASHING_LINE[2] + 0.05), texture="wood",
                kind="washing_line", solid=False),
        PropDef("nessa_tub", (-48.2, 0.0, 29.2), (-47.2, 0.5, 30.2), texture="barrel", kind="barrel"),
        *wall_windows("bertram_window", bx1, bx2, bz1, 0.0, keep_clear=((32.0, 35.0), (31.15, 31.65))),
        *wall_windows("nessa_window", nx1, nx2, nz1, 0.0, keep_clear=((-44.0, -41.0), (-44.85, -44.35))),
        woodpile("bertram_woodpile", bx2 + 0.2, 34.0, bx2 + 1.1, 39.0),
    ]
    p.containers += [
        table("bertram_table", "Table", (32.0, 0.0, 36.0), owner="Old Bertram", size=(0.7, 0.45),
              items=[ItemDef("stew", "Pot of Stew", value=4.0, stackable=True, quantity=9, is_food=True, category="food")]),
        chest("bertram_chest", "Old Footlocker", (30.9, 0.0, 38.6), owner="Old Bertram", along_x=False,
              items=[coins(6), ItemDef("old_helm", "Dented Watch Helm", value=8.0, category="armor")]),
        # his secret: thirty years of watch pay, and the medal, in the rain barrel behind the cottage
        ContainerDef("bertram_rain_barrel", "Rain Barrel", (34.6, 0.0, bz2 + 0.35), (35.6, 1.1, bz2 + 1.35), owner="Old Bertram",
                     texture="barrel", kind="barrel_store",
                     items=(coins(160), ItemDef("service_medal", "Medal of the Emberhollow Watch", value=60.0))),
        table("nessa_table", "Table", (-41.2, 0.0, 37.9), owner="Nessa", size=(0.7, 0.45),
              items=[ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=11, is_food=True, category="food")]),
        # what comes out of the gentry's pockets in the wash
        ContainerDef("nessa_basket", "Laundry Basket", (-45.4, 0.0, 34.0), (-44.2, 0.8, 35.2), owner="Nessa", texture="hay",
                     kind="sacks",
                     items=(ItemDef("gold_ring", "Gold Ring", value=150.0), ItemDef("silver_buttons", "Silver Buttons", value=24.0),
                            ItemDef("perfumed_letter", "A Perfumed Letter", value=1.0), coins(14))),
    ]
    p.lights += [wall_torch(32.0 - 0.6, bz1, 0.0, "Old Bertram's cottage"), wall_torch(-44.0 - 0.6, nz1, 0.0, "Nessa's cottage")]
    p.areas += [
        AreaDef("Old Bertram's cottage", (bx1, bz1), (bx2, bz2), owner="Old Bertram", restricted=True),
        AreaDef("Nessa's cottage", (nx1, nz1), (nx2, nz2), owner="Nessa", restricted=True),
    ]
    p.npcs += [
        NPCDef("Old Bertram", BERTRAM_BED, role="crier", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.45, 0.18, 0.16), legs=(0.2, 0.2, 0.24), sash=(0.85, 0.7, 0.25), hat="cap",
                             hat_color=(0.25, 0.2, 0.16), hair=(0.8, 0.8, 0.8)),
               inventory=(key_item("bertram_door", "Cottage Key"), ItemDef("hand_bell", "Crier's Hand Bell", value=12.0, for_sale=False)),
               lines=("I used to walk the walls like Vance's lot, you know. Then I took an arrow in the knee. Now I walk the street. Slowly.",
                      "The watch at the gate changes sides every three hours. Young Bram yawns at every change. I never yawned. Well. Once.",
                      "Thirty years' pay the Captain gave me, and a medal. I don't trust Barnaby's strongbox nor nobody's. Rain doesn't steal.",
                      "Hear ye, hear ye - oh, it's you. The tithe's due at Greyhold, the Deep's bandits are about, and Hilde's ale is off. As you were."),
               schedule=resident_tasks("Bertram", "bertram_door", BERTRAM_BED, BERTRAM_SEAT, "bertram_table", [
                   TaskDef(8, 9, CRIER_SPOT, AIState.WORKING, "Old Bertram is crying the morning news by the well.", facing=0.0,
                           depart_text="Hear ye! Hear ye! ...once I get there."),
                   TaskDef(9, 11, GATE_BENCH_SPOT, AIState.WORKING, "Old Bertram is telling the gate watch how it was done in his day.",
                           facing=90.0),
                   visit(11, 13, (-26.1, 0.0, -11.0), AIState.DRINKING, "Old Bertram is having his dinner at the Drowsy Dragon.",
                         facing=90.0),
                   TaskDef(13, 14, CRIER_SPOT, AIState.WORKING, "Old Bertram is crying the afternoon news by the well.", facing=0.0),
                   TaskDef(14, 17, BERTRAM_SPOT, AIState.WORKING, "Old Bertram is resting his knee on the bench in the square.",
                           facing=0.0),
                   TaskDef(17, 18, CRIER_SPOT, AIState.WORKING, "Old Bertram is crying the evening news.", facing=0.0),
                   visit(18, 20, (-29.25 + 1.35, 0.0, -8.5), AIState.DRINKING, "Old Bertram is telling war stories at the inn.",
                         facing=270.0),
               ], wake=7, supper=(20, 22)),                             # (a slow walk home on that knee)
               # a day with the knee bad: the gate all morning, his own dinner, no inn
               rota=(resident_tasks("Bertram", "bertram_door", BERTRAM_BED, BERTRAM_SEAT, "bertram_table", [
                   TaskDef(8, 12, GATE_BENCH_SPOT, AIState.WORKING, "Old Bertram is at the gate, grumbling at the road.", facing=90.0,
                           depart_text="The knee says rain. The knee is always right."),
                   TaskDef(12, 14, BERTRAM_SEAT, AIState.EATING, "Old Bertram is having his dinner at home, knee up.", facing=0.0,
                           food_container="bertram_table", unlock_doors=("bertram_door",)),
                   TaskDef(14, 15, CRIER_SPOT, AIState.WORKING, "Old Bertram is crying the afternoon news by the well.", facing=0.0),
                   TaskDef(15, 18, BERTRAM_SPOT, AIState.WORKING, "Old Bertram is resting his knee on the bench in the square.",
                           facing=0.0),
                   TaskDef(18, 19, CRIER_SPOT, AIState.WORKING, "Old Bertram is crying the evening news.", facing=0.0),
               ], wake=7, supper=(19, 21)),)),
        NPCDef("Nessa", NESSA_BED, role="washerwoman", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.55, 0.48, 0.42), legs=(0.3, 0.26, 0.22), apron="cloth", hat="scarf", hat_color=(0.7, 0.3, 0.28),
                             hair=(0.4, 0.22, 0.12)),
               inventory=(key_item("nessa_door", "Cottage Key"), ItemDef("washboard", "Washboard", value=3.0, for_sale=False)),
               lines=("Half the town's smalls go through my hands. You'd be amazed what people leave in their pockets. I'm not.",
                      "Lady Ashgrove and Lord Vane haven't spoken in eleven years - and his handkerchiefs smell of her perfume. Make of that what you will.",
                      "Greaves up at Vane's sends the household linen, and a second lot he pays for himself. Whose, I wonder?",
                      "The millstream's cold as charity, even in summer."),
               schedule=resident_tasks("Nessa", "nessa_door", NESSA_BED, NESSA_SEAT, "nessa_table", [
                   TaskDef(7, 12, WASHING_SPOT, AIState.WORKING, "Nessa is scrubbing linen in the millstream.", facing=270.0,
                           depart_text="Other people's dirt won't wash itself."),
                   TaskDef(12, 13, NESSA_SEAT, AIState.EATING, "Nessa is eating at home.", facing=90.0, food_container="nessa_table",
                           unlock_doors=("nessa_door",)),
                   TaskDef(13, 15, (WASHING_LINE[0], 0.0, WASHING_LINE[2] - 1.0), AIState.WORKING, "Nessa is hanging out the washing.",
                           facing=180.0),
                   stroll(15, 17, "Nessa", ((-34.0, 0.0, -4.2), (-28.0, 0.0, -39.0), (18.0, 0.0, -39.0), (-3.0, 0.0, 4.0)),
                          "Nessa is carrying clean linen up to the big houses."),
                   visit(17, 20, (-29.25 - 1.35, 0.0, -14.0), AIState.DRINKING, "Nessa is trading gossip at the inn.", facing=90.0),
               ], wake=6, bedtime=21, facing=90.0)),
    ]
    return p


def bakery() -> Part:
    """Wynn's bakery: the shop at the front, the oven and the flour at the back, his bed behind the oven."""
    p = Part()
    x1, z1, x2, z2 = BAKERY
    p.walls += house_walls(x1, z1, x2, z2, BAKERY_DOOR_X, "north", texture="plaster")
    p.buildings += [BuildingDef("bakery", (x1, z1), (x2, z2), roof_texture="roof")]
    p.doors += [DoorDef("bakery_door", (BAKERY_DOOR_X, 0.0, z1), is_locked=True, key_id="bakery_door_key", owners=("Wynn the Baker",),
                        lock_difficulty=SIMPLE, label="Wynn's Bakery")]
    p.props += [
        PropDef("bakery_counter", (36.2, 0, 8.6), (40.6, 1.0, 9.1), texture="wood"),
        PropDef("bakery_oven", (33.6, 0, 12.6), (36.2, 1.6, 13.5), texture="stone", kind="oven", facing=180.0),
        PropDef("sign_bakery", (39.8, 2.3, 4.9), (40.6, 3.1, 5.7), texture="sign_bakery", kind="sign", facing=90.0, solid=False),
        bed("wynn_bed", WYNN_BED, facing=90.0),
        *wall_windows("bakery_window", x1, x2, z1, 0.0, keep_clear=((BAKERY_DOOR_X, BAKERY_DOOR_X + 3.0),
                                                                  (BAKERY_DOOR_X - 0.85, BAKERY_DOOR_X - 0.35), (39.8, 40.6))),
    ]
    bread = lambda n: ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=n, is_food=True, category="food")  # noqa: E731
    p.containers += [
        ContainerDef("bakery_shelf", "Bread Shelves", (36.6, 0, 12.9), (39.6, 1.8, 13.5), owner="Wynn the Baker", crime_name="Shoplifting",
                     texture="shelf",
                     items=(bread(10), ItemDef("meat_pie", "Meat Pie", value=6.0, stackable=True, quantity=4, is_food=True, category="food"),
                            ItemDef("sweet_roll", "Honey Roll", value=4.0, stackable=True, quantity=6, is_food=True, category="food"))),
        table("wynn_table", "Table", (38.0, 0.0, 12.0), owner="Wynn the Baker", size=(0.6, 0.4), items=[bread(9)]),
        # the flour sacks in the back: one of them is not all flour
        ContainerDef("bakery_flour", "Flour Sacks", (41.3, 0.0, 9.5), (42.4, 1.0, 10.9), owner="Wynn the Baker", texture="hay",
                     kind="sacks", items=(ItemDef("flour", "Sack of Flour", value=5.0, stackable=True, quantity=3), coins(95),
                                          ItemDef("mothers_brooch", "Wynn's Mother's Brooch", value=70.0))),
    ]
    p.lights += [LightDef("fire", (34.9, 0.9, 13.1), room="the bakery", range=6.0),
                 wall_torch(BAKERY_DOOR_X - 0.6, z1, 0.0, "the bakery")]
    p.areas += [
        AreaDef("behind the bakery counter", (x1, 9.1), (x2, z2), owner="Wynn the Baker", restricted=True),
        AreaDef("the bakery", (x1, z1), (x2, 9.1), owner="Wynn the Baker", restricted=True, public_hours=(7, 18)),
    ]
    p.npcs += [
        NPCDef("Wynn the Baker", WYNN_BED, can_trade=True, role="baker", trade_categories=("food",), shop_containers=("bakery_shelf",),
               restock_days=1, inventory_reset_days=3, gold=180.0,
               outfit=Outfit(tunic=(0.9, 0.88, 0.82), legs=(0.4, 0.34, 0.26), apron="cloth", hat="coif", hat_color=(0.96, 0.96, 0.94),
                             hair=(0.55, 0.35, 0.15)),
               inventory=(key_item("bakery_door", "Bakery Key"), bread(4)),
               lines=("Up at four, bread by seven, and not a word of thanks till the crust's gone stale.",
                      "Lady Ashgrove sends my rolls back - 'too rustic'. Her maid eats them on the back stairs.",
                      "Corwin doesn't trust Barnaby's scales, nor Barnaby's strongbox. Keeps his money in a barrel, they say. A cooper. In a barrel.",
                      "I take bread up to the watch house every noon. Guard Orrin eats two loaves and swears it was one."),
               schedule=resident_tasks("Wynn", "bakery_door", WYNN_BED, WYNN_SEAT, "wynn_table", [
                   TaskDef(5, 7, BAKERY_OVEN, AIState.WORKING, "Wynn is firing the oven and baking the day's bread.", facing=180.0),
                   TaskDef(7, 12, BAKERY_COUNTER, AIState.SHOPKEEPING, "Wynn has opened the bakery.", unlock_doors=("bakery_door",),
                           facing=0.0),
                   TaskDef(12, 14, BREAD_ROUND[0], AIState.STROLLING, "Wynn is taking bread round the inn and the watch house.",
                           patrol_points=BREAD_ROUND),
                   TaskDef(14, 18, BAKERY_COUNTER, AIState.SHOPKEEPING, "Wynn is back behind his counter.", facing=0.0),
                   TaskDef(18, 19, BAKERY_OVEN, AIState.WORKING, "Wynn is setting the dough for the morning.", facing=180.0),
                   stroll(19, 20, "Wynn", ((38.0, 0.0, 4.6), (52.0, 0.0, 4.6), (62.0, 0.0, 20.0), (38.0, 0.0, 4.6)),
                          "Wynn is taking the evening air, floury to the elbows."),
               ], wake=4, bedtime=21, facing=180.0)),
    ]
    return p


def cooper() -> Part:
    """Corwin's house, his yard and his shed: staves and hoops, barrels stood about to season."""
    p = Part()
    x1, z1, x2, z2 = COOPER_HOUSE
    p.walls += house_walls(x1, z1, x2, z2, COOPER_DOOR_X, "south", texture="building")
    sx1, sz1, sx2, sz2 = COOPER_SHED
    p.walls += [WallDef((x, 0, z), (x + 0.3, 3.4, z + 0.3)) for x, z in ((sx1, sz1), (sx2 - 0.3, sz1), (sx1, sz2 - 0.3),
                                                                         (sx2 - 0.3, sz2 - 0.3))]
    p.buildings += [
        BuildingDef("cooper_house", (x1, z1), (x2, z2), roof_texture="roof"),
        BuildingDef("cooper_shed", (sx1, sz1), (sx2, sz2), wall_height=3.4, roof_texture="thatch", open_sides=True,
                    pad_with="cooper_house"),
    ]
    p.doors += [DoorDef("cooper_door", (COOPER_DOOR_X, 0.0, z2), is_locked=True, key_id="cooper_door_key", owners=("Corwin the Cooper",),
                        lock_difficulty=SIMPLE, label="Corwin's House")]
    p.props += [
        bed("corwin_bed", CORWIN_BED, facing=90.0),
        PropDef("cooper_bench", (54.6, 0.0, -16.8), (57.4, 0.9, -15.8), texture="wood"),
        *barrels("cooper_shed_barrels", [(58.8, -18.8), (58.8, -17.8), (53.2, -18.9)]),
        PropDef("cooper_staves", (52.6, 0.0, -14.2), (55.0, 0.7, -13.4), texture="bark", kind="woodpile", facing=90.0),
        woodpile("cooper_woodpile", x2 + 0.2, z1 + 1.0, x2 + 1.1, z2 - 1.0),
        PropDef("cooper_window", (67.8, 1.2, z2), (69.2, 2.5, z2 + 0.02), texture="window_cottage", kind="window", facing=180.0,
                solid=False),
    ]
    p.containers += [
        table("corwin_table", "Table", (67.0, 0.0, -18.4), owner="Corwin the Cooper", size=(0.7, 0.45),
              items=[ItemDef("stew", "Pot of Stew", value=4.0, stackable=True, quantity=10, is_food=True, category="food")]),
        chest("corwin_chest", "Tool Chest", (63.0, 0.0, -20.5), owner="Corwin the Cooper", along_x=False,
              items=[coins(7), ItemDef("adze", "Cooper's Adze", value=14.0, category="tool")]),
        # the rain barrel round the back: the money's in the bottom, under a false head
        ContainerDef("corwin_rain_barrel", "Old Rain Barrel", (68.4, 0.0, z1 - 1.4), (69.4, 1.1, z1 - 0.4), owner="Corwin the Cooper",
                     texture="barrel", kind="barrel_store",
                     items=(coins(210), ItemDef("silver_hoop", "Silver-Bound Tankard", value=48.0))),
    ]
    p.lights += [wall_torch(COOPER_DOOR_X - 0.6, z2, 180.0, "Corwin's house")]
    p.areas += [
        AreaDef("Corwin's house", (x1, z1), (x2, z2), owner="Corwin the Cooper", restricted=True),
        AreaDef("the cooper's yard", (sx1, sz1), (x2, -6.5), owner="Corwin the Cooper"),
    ]
    p.npcs += [
        NPCDef("Corwin the Cooper", CORWIN_BED, role="cooper", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.42, 0.34, 0.26), legs=(0.3, 0.26, 0.22), apron="leather", hair=(0.3, 0.25, 0.2)),
               inventory=(key_item("cooper_door", "House Key"),),
               lines=("Oak staves, iron hoops, and no nonsense.",
                      "Barnaby's scales are short by a thumb. I've told him. He says it's the weather.",
                      "Hrolf made my hoop-iron, back before he and Gareth fell out. Best iron in the March. Now it's Gareth's or nothing.",
                      "Banks. Strongboxes. Locks. Every one of 'em made by a man who can open it. Me, I keep my own counsel."),
               schedule=resident_tasks("Corwin", "cooper_door", CORWIN_BED, CORWIN_SEAT, "corwin_table", [
                   TaskDef(7, 12, COOPER_BENCH, AIState.WORKING, "Corwin is shaping staves at his bench.", facing=180.0,
                           depart_text="Staves don't bend themselves."),
                   TaskDef(12, 13, CORWIN_SEAT, AIState.EATING, "Corwin is eating.", facing=0.0, food_container="corwin_table",
                           unlock_doors=("cooper_door",)),
                   TaskDef(13, 16, COOPER_BENCH, AIState.WORKING, "Corwin is hooping a barrel.", facing=180.0),
                   TaskDef(16, 18, BARREL_ROLL[0], AIState.STROLLING, "Corwin is rolling a new barrel down to the inn.",
                           patrol_points=BARREL_ROLL),
                   visit(18, 20, (-29.25 + 1.35, 0.0, -14.0), AIState.DRINKING, "Corwin is drinking from his own barrels, at the inn.",
                         facing=270.0),
               ], wake=6, supper=(20, 22), facing=0.0),
               # the day after: staves to plane and no inn
               rota=(resident_tasks("Corwin", "cooper_door", CORWIN_BED, CORWIN_SEAT, "corwin_table", [
                   TaskDef(7, 12, COOPER_BENCH, AIState.WORKING, "Corwin is planing staves.", facing=180.0,
                           depart_text="Staves don't bend themselves."),
                   TaskDef(12, 13, CORWIN_SEAT, AIState.EATING, "Corwin is eating.", facing=0.0, food_container="corwin_table",
                           unlock_doors=("cooper_door",)),
                   TaskDef(13, 20, COOPER_BENCH, AIState.WORKING, "Corwin is working late on a wine cask for the Vanes.", facing=180.0),
               ], wake=6, bedtime=21, facing=0.0),)),
    ]
    return p


def stable() -> Part:
    """Hobb's corner of the stable: a cot in the hay, a crate for a table, his savings under the straw."""
    p = Part()
    p.props += [bed("hobb_bed", HOBB_BED, facing=90.0)]
    p.containers += [
        table("hobb_table", "Upturned Crate", (21.0, 0.0, 35.6), owner="Hobb", size=(0.5, 0.4),
              items=[ItemDef("apple_1", "Crisp Apple", value=8.0, stackable=True, quantity=6, is_food=True, category="food"),
                     ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=6, is_food=True, category="food")]),
        stash("hobb_stash", "Under the Straw", (21.8, 0.0, 38.4), owner="Hobb",
              items=[coins(44), ItemDef("ribbon", "Blue Silk Ribbon", value=6.0)]),
    ]
    p.areas += [AreaDef("the stable", (12.0, 34.0), (24.0, 44.0), owner="Hobb")]
    p.npcs += [
        NPCDef("Hobb", HOBB_BED, role="stablehand", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
               # (Lys feeds him at noon: what she wraps up for him in the morning, eaten at her kitchen door)
               inventory=(ItemDef("bread_parcel", "Bread and Dripping", value=2.0, stackable=True, quantity=3, is_food=True,
                                  category="food"),),
               outfit=Outfit(tunic=(0.5, 0.42, 0.3), legs=(0.34, 0.28, 0.2), hat="cap", hat_color=(0.4, 0.32, 0.2), hair=(0.85, 0.65, 0.3)),
               lines=("Horses? We've no horses. Lord Vane sold them to pay his card debts. I groom the memory of them.",
                      "Ottilie up at Ashgrove's says her ladyship sleeps like the dead - eleven till eight, and never stirs.",
                      "I'm saving for a ring. Don't tell Ottilie. Don't tell anyone. It's under the straw. Oh no."),
               schedule=resident_tasks("Hobb", None, HOBB_BED, HOBB_SEAT, "hobb_table", [
                   TaskDef(6, 9, STALLS, AIState.WORKING, "Hobb is mucking out the empty stalls.", facing=0.0),
                   TaskDef(9, 12, (14.5, 0.0, 37.0), AIState.WORKING, "Hobb is forking hay for horses nobody owns.", facing=90.0),
                   visit(12, 14, (-35.9, 0.0, -12.6), AIState.EATING, "Hobb is being fed in the inn kitchen by Lys.", facing=270.0),
                   TaskDef(14, 17, STALLS, AIState.WORKING, "Hobb is mending harness.", facing=0.0),
                   TaskDef(17, 19, PARK_MEETING[0], AIState.WORKING, "Hobb is waiting for Ottilie by the park bench, hair combed.",
                           facing=90.0),
                   TaskDef(19, 21, STALLS, AIState.WORKING, "Hobb is brushing down the empty stalls, whistling.", facing=0.0),
               ], wake=5, supper=(21, 22), facing=90.0),
               # Ottilie's chapel day: he eats his own bread, and loiters outside the chapel door
               rota=(resident_tasks("Hobb", None, HOBB_BED, HOBB_SEAT, "hobb_table", [
                   TaskDef(6, 9, STALLS, AIState.WORKING, "Hobb is mucking out the empty stalls.", facing=0.0),
                   TaskDef(9, 12, (14.5, 0.0, 37.0), AIState.WORKING, "Hobb is forking hay for horses nobody owns.", facing=90.0),
                   TaskDef(12, 13, HOBB_SEAT, AIState.EATING, "Hobb is eating on his upturned crate.", facing=90.0,
                           food_container="hobb_table"),
                   TaskDef(13, 17, STALLS, AIState.WORKING, "Hobb is mending harness.", facing=0.0),
                   TaskDef(17, 19, CHAPEL_STEPS, AIState.WORKING, "Hobb is loitering outside the chapel, hat in hand.", facing=180.0),
                   TaskDef(19, 20, STALLS, AIState.WORKING, "Hobb is brushing down the empty stalls, whistling.", facing=0.0),
               ], wake=5, bedtime=21, facing=90.0),)),
    ]
    return p


# -- along the south side of Upper Street: four houses facing the great ones across the way
HERB_SHOP_FLOOR = (-28.25, 0.0, -32.8)      # in Griselda's shop, before her counter
REEVE_DOORSTEP = (12.0, 0.0, -36.8)         # outside the Reeve's door
UPPER_HOUSES = {
    "griselda": (-32.0, -35.0, -24.0, -26.0),
    "ludo": (-18.0, -35.0, -10.0, -26.0),
    "pike": (8.0, -35.0, 16.0, -26.0),
    "dunstan": (20.0, -35.0, 28.0, -26.0),
}
WELL_SONG = (-2.8, 0.0, -3.4)              # where the minstrel sings, by the well
PARK_SONG = (-1.0, 0.0, -48.0)             # ... by the fountain in the park
PARK_BORDERS = ((-5.5, 0.0, -55.0), (4.0, 0.0, -56.5), (4.0, 0.0, -47.0), (-5.5, 0.0, -50.0))   # the park's flower borders
INN_STAGE = (-23.3, 0.0, -13.8)            # where he plays in the inn of an evening

# -- Tanner's Row, off Market Street in the south-east
TANNER_HOUSE = (66.0, 27.0, 74.0, 34.0)
TANNERY = (62.0, 38.0, 72.0, 46.0)
JEB_HOUSE = (46.0, 27.0, 53.0, 34.0)
TORCH_ROUND = ((-8.5, 0.0, 46.2), (-3.0, 0.0, 24.0), (-3.0, 0.0, 8.0), (-33.0, 0.0, -4.2), (-67.0, 0.0, -4.0),
               (-34.0, 0.0, 4.5), (10.0, 0.0, -4.5), (47.0, 0.0, -4.0), (52.0, 0.0, 4.6), (37.0, 0.0, -36.5),
               (0.0, 0.0, -36.5), (48.5, 0.0, -41.0), (62.0, 0.0, 24.0))


def _house(p: Part, key: str, rect, door_x: float, door_side: str, label: str, owner: str, lock: int, texture: str = "plaster",
           roof: str = "roof") -> None:
    x1, z1, x2, z2 = rect
    p.walls += house_walls(x1, z1, x2, z2, door_x, door_side, texture=texture)
    p.buildings += [BuildingDef(f"{key}_house", (x1, z1), (x2, z2), roof_texture=roof)]
    dz = z1 if door_side == "north" else z2
    p.doors += [DoorDef(f"{key}_door", (door_x, 0.0, dz), is_locked=True, key_id=f"{key}_door_key", owners=(owner,),
                        lock_difficulty=lock, label=label)]
    facing = 0.0 if door_side == "north" else 180.0
    # windows either side of the door, where there is room (clear of the door and its torch)
    p.props += wall_windows(f"{key}_window", x1, x2, dz, facing, keep_clear=((door_x, door_x + 3.0), (door_x - 0.85, door_x - 0.35)))
    p.lights.append(wall_torch(door_x - 0.6, dz, facing, label))


def upper_street() -> Part:
    """The herbwife, the locksmith, the Reeve and the minstrel: four houses facing the great ones."""
    p = Part()
    g, lu, pk, du = (UPPER_HOUSES[k] for k in ("griselda", "ludo", "pike", "dunstan"))
    _house(p, "griselda", g, g[0] + 2.5, "north", "Griselda's Herbs", "Griselda the Herbwife", SIMPLE, texture="plaster_timber",
           roof="thatch")
    _house(p, "ludo", lu, lu[0] + 2.5, "north", "Ludo's Locks & Keys", "Ludo the Locksmith", STRONG, texture="building")
    _house(p, "pike", pk, pk[0] + 2.5, "north", "The Reeve's House", "Reeve Osric Pike", STURDY, texture="plaster")
    _house(p, "dunstan", du, du[0] + 2.5, "north", "Dunstan's Lodging", "Dunstan the Minstrel", SIMPLE, texture="plaster_timber")
    # -- Griselda: the front room is her shop, her herb garden behind
    gx1, gz1, gx2, gz2 = g
    p.props += [
        PropDef("griselda_counter", (gx1 + 2.0, 0, gz1 + 3.4), (gx1 + 5.5, 1.0, gz1 + 3.9), texture="wood"),
        bed("griselda_bed", (gx2 - 1.55, 0.0, gz2 - 1.6), facing=90.0),
        *(PropDef(f"griselda_herb_bed_{i}", (gx1 + a, 0.0, gz2 + 0.8), (gx1 + b, 0.4, gz2 + 2.2), texture="flowers", kind="planter")
          for i, (a, b) in enumerate(((0.8, 2.8), (3.4, 4.2), (6.0, 8.0)))),              # (the rue's half dug up: see the jar)
    ]
    herb = lambda id_, name, v, n: ItemDef(id_, name, value=v, stackable=True, quantity=n, category="goods")  # noqa: E731
    p.containers += [
        ContainerDef("griselda_shelf", "Herb Shelves", (gx1 + 0.55, 0.0, gz1 + 4.4), (gx1 + 1.15, 1.8, gz1 + 7.0),
                     owner="Griselda the Herbwife", crime_name="Shoplifting", texture="shelf",
                     items=(herb("healing_salve", "Healing Salve", 12.0, 4), herb("sleeping_draught", "Sleeping Draught", 30.0, 2),
                            herb("lavender", "Bundle of Lavender", 3.0, 6), herb("valerian", "Valerian Root", 8.0, 3))),
        table("griselda_table", "Table", (gx1 + 4.0, 0.0, gz2 - 1.2), owner="Griselda the Herbwife", size=(0.6, 0.45),
              items=[ItemDef("stew", "Pot of Stew", value=4.0, stackable=True, quantity=9, is_food=True, category="food")]),
        # under the middle bed of rue: a jar she buried the day the Reeve's men came asking questions
        stash("griselda_jar", "Freshly Dug Earth", (gx1 + 4.8, 0.0, gz2 + 1.5), owner="Griselda the Herbwife",
              items=[coins(120), ItemDef("poison_book", "A Little Book of Poisons", value=90.0)]),
    ]
    # -- Ludo: a workbench of locks, and the best lock in town on his own door
    lx1, lz1, lx2, lz2 = lu
    p.props += [
        PropDef("ludo_bench", (lx1 + 2.0, 0, lz1 + 3.4), (lx1 + 5.5, 1.0, lz1 + 3.9), texture="wood"),
        bed("ludo_bed", (lx2 - 1.55, 0.0, lz2 - 1.6), facing=90.0),
    ]
    p.containers += [
        ContainerDef("ludo_shelf", "Locks and Picks", (lx1 + 0.55, 0.0, lz1 + 4.4), (lx1 + 1.15, 1.8, lz1 + 7.0), owner="Ludo the Locksmith",
                     crime_name="Shoplifting", texture="shelf",
                     items=(ItemDef("iron_lockpick", "Iron Lockpick", value=5.0, is_lockpick=True, stackable=True, quantity=8, category="tool"),
                            ItemDef("fine_lockpick", "Fine Steel Lockpick", value=15.0, is_lockpick=True, stackable=True, quantity=4,
                                    category="tool"),
                            ItemDef("padlock", "Brass Padlock", value=18.0, category="tool"))),
        table("ludo_table", "Table", (lx1 + 4.0, 0.0, lz2 - 1.2), owner="Ludo the Locksmith", size=(0.6, 0.45),
              items=[ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=11, is_food=True, category="food")]),
        chest("ludo_strongbox", "Ludo's Strongbox", (lx1 + 1.2, 0.0, lz1 + 1.2), owner="Ludo the Locksmith", strongbox=True, locked=True,
              lock_difficulty=MASTER, key_id="ludo_door",
              items=[coins(150), ItemDef("lockpick_pack", "Master Lockpick", value=20.0, is_lockpick=True, stackable=True, quantity=3,
                                         category="tool"),
                     ItemDef("wax_kit", "Wax Impression Kit", value=40.0, category="tool")]),
    ]
    # -- the Reeve: the town's tithe, counted and locked up until it goes to Greyhold
    px1, pz1, px2, pz2 = pk
    p.props += [bed("pike_bed", (px2 - 1.55, 0.0, pz2 - 1.6), facing=90.0),
                PropDef("pike_bookcase", (px1 + WALL, 0.0, pz1 + 1.2), (px1 + 0.95, 2.4, pz1 + 4.4), texture="shelf", kind="bookcase",
                        facing=90.0)]
    p.containers += [
        table("pike_desk", "Reeve's Desk", (px1 + 3.0, 0.0, pz1 + 4.4), owner="Reeve Osric Pike", size=(0.9, 0.5),
              items=[ItemDef("tithe_roll", "The Town Tithe Roll", value=2.0), ItemDef("bread", "Fresh Bread", value=3.0, stackable=True,
                                                                                   quantity=11, is_food=True, category="food")]),
        chest("pike_tithe_chest", "Town Tithe Chest", (px2 - 1.2, 0.0, pz1 + 1.0), owner="Reeve Osric Pike", locked=True,
              lock_difficulty=STRONG, key_id="pike_tithe_chest",
              items=[coins(240), ItemDef("tithe_seal", "The Reeve's Seal", value=30.0)]),
        stash("pike_floorboard", "Loose Floorboard", (px1 + 1.0, 0.0, pz2 - 1.0), owner="Reeve Osric Pike",
              items=[coins(130), ItemDef("second_ledger", "The Reeve's Other Ledger", value=5.0)]),
    ]
    # -- Dunstan: a lute, a bed he is seldom up before noon from, a chest of letters
    dx1, dz1, dx2, dz2 = du
    p.props += [bed("dunstan_bed", (dx2 - 1.55, 0.0, dz2 - 1.6), facing=90.0)]
    p.containers += [
        table("dunstan_table", "Table", (dx1 + 2.5, 0.0, dz1 + 4.6), owner="Dunstan the Minstrel", size=(0.6, 0.45),
              items=[ItemDef("wine", "Bottle of Wine", value=12.0, stackable=True, quantity=2, category="food"),
                     ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=8, is_food=True, category="food")]),
        chest("dunstan_chest", "Minstrel's Chest", (dx1 + 1.0, 0.0, dz2 - 1.2), owner="Dunstan the Minstrel", along_x=False, locked=True,
              lock_difficulty=SIMPLE, key_id="dunstan_door",
              items=[ItemDef("gold_locket", "Gold Locket with an 'S'", value=140.0), ItemDef("letters_ashgrove", "Letters Signed 'S.'",
                                                                                             value=2.0), coins(22)]),
    ]
    for key, rect in UPPER_HOUSES.items():
        x1, z1, x2, z2 = rect
        owner = {"griselda": "Griselda the Herbwife", "ludo": "Ludo the Locksmith", "pike": "Reeve Osric Pike",
                 "dunstan": "Dunstan the Minstrel"}[key]
        p.areas.append(AreaDef({"griselda": "Griselda's herb shop", "ludo": "Ludo's lock shop", "pike": "the Reeve's house",
                                "dunstan": "Dunstan's lodging"}[key], (x1, z1), (x2, z2), owner=owner, restricted=True,
                               public_hours=(9, 17) if key in ("griselda", "ludo") else None))
    g_bed, g_seat = (gx2 - 1.55, 0.0, gz2 - 1.6), (gx1 + 4.0, 0.0, gz2 - 2.6)
    l_bed, l_seat = (lx2 - 1.55, 0.0, lz2 - 1.6), (lx1 + 4.0, 0.0, lz2 - 2.6)
    p_bed, p_seat = (px2 - 1.55, 0.0, pz2 - 1.6), (px1 + 3.0, 0.0, pz1 + 5.6)
    d_bed, d_seat = (dx2 - 1.55, 0.0, dz2 - 1.6), (dx1 + 2.5, 0.0, dz1 + 5.6)
    p.npcs += [
        NPCDef("Griselda the Herbwife", g_bed, can_trade=True, role="herbwife", trade_categories=("goods", "food"),
               shop_containers=("griselda_shelf",), restock_days=3, inventory_reset_days=3, gold=140.0,
               outfit=Outfit(tunic=(0.32, 0.42, 0.28), legs=(0.26, 0.24, 0.2), hat="scarf", hat_color=(0.45, 0.3, 0.5), hair=(0.7, 0.7, 0.7)),
               inventory=(key_item("griselda_door", "Cottage Key"),),
               lines=("Salve for the knees, draught for the nerves, something stronger for the husband. I don't ask.",
                      "The Reeve sent his men round asking what I put in Lord Vane's tonic. Only what he paid for, I told them.",
                      "A sleeping draught in the right cup and a whole house sleeps sound. Not that anybody would. Would they.",
                      "I keep a herb bed for every ill. The middle one I dig myself. Nobody else touches the rue."),
               schedule=resident_tasks("Griselda", "griselda_door", g_bed, g_seat, "griselda_table", [
                   TaskDef(8, 9, (gx1 + 2.0, 0.0, gz2 + 3.0), AIState.WORKING, "Griselda is weeding her herb beds.", facing=180.0),
                   TaskDef(9, 13, (gx1 + 3.75, 0.0, gz1 + 4.9), AIState.SHOPKEEPING, "Griselda has opened her herb shop.",
                           unlock_doors=("griselda_door",), facing=0.0),
                   TaskDef(13, 14, g_seat, AIState.EATING, "Griselda is eating.", facing=180.0, food_container="griselda_table"),
                   TaskDef(14, 17, (gx1 + 3.75, 0.0, gz1 + 4.9), AIState.SHOPKEEPING, "Griselda is back at her counter.", facing=0.0),
                   TaskDef(17, 19, PARK_BORDERS[0], AIState.STROLLING, "Griselda is gathering herbs in the park's borders.",
                           patrol_points=PARK_BORDERS),
                   TaskDef(19, 20, (gx1 + 2.0, 0.0, gz2 + 3.0), AIState.WORKING, "Griselda is hanging the herbs up to dry.",
                           facing=180.0),
               ], wake=7, bedtime=21, facing=180.0)),
        NPCDef("Ludo the Locksmith", l_bed, can_trade=True, role="locksmith", trade_categories=("tool",), shop_containers=("ludo_shelf",),
               restock_days=2, inventory_reset_days=3, gold=220.0,
               outfit=Outfit(tunic=(0.3, 0.3, 0.34), legs=(0.2, 0.2, 0.22), apron="leather", hat="cap", hat_color=(0.22, 0.22, 0.26),
                             hair=(0.35, 0.3, 0.28)),
               inventory=(key_item("ludo_door", "Ludo's Key"),),
               lines=("I made half the locks in this town. In my misspent youth I opened the other half.",
                      "A lock is a promise that somebody will have to work for it. The Vanes' strongbox is a very firm promise.",
                      "Practice on the simple ones - Barnaby's back door is an education. Don't tell him I said so.",
                      "Picks snap. Buy three. Buy five. Business is business."),
               schedule=resident_tasks("Ludo", "ludo_door", l_bed, l_seat, "ludo_table", [
                   TaskDef(8, 9, (lx1 + 3.75, 0.0, lz1 + 4.9), AIState.WORKING, "Ludo is oiling his locks.", facing=180.0),
                   TaskDef(9, 13, (lx1 + 3.75, 0.0, lz1 + 4.9), AIState.SHOPKEEPING, "Ludo has opened his shop.",
                           unlock_doors=("ludo_door",), facing=0.0),
                   TaskDef(13, 14, l_seat, AIState.EATING, "Ludo is eating.", facing=180.0, food_container="ludo_table"),
                   TaskDef(14, 17, (lx1 + 3.75, 0.0, lz1 + 4.9), AIState.SHOPKEEPING, "Ludo is back at his bench.", facing=0.0),
                   visit(17, 20, (-36.75 + 1.35, 0.0, -8.2), AIState.DRINKING, "Ludo is drinking alone, watching the door.", facing=270.0),
               ], wake=7, bedtime=21, facing=180.0)),
        NPCDef("Reeve Osric Pike", p_bed, role="reeve", gold=90.0, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.2, 0.26, 0.22), legs=(0.18, 0.18, 0.2), body="doublet", neck="necklace",
                             neck_color=(0.85, 0.75, 0.3), hat="cap", hat_color=(0.12, 0.12, 0.12), hair=(0.45, 0.4, 0.35)),
               inventory=(key_item("pike_door", "Reeve's Key"), ItemDef("pike_tithe_chest_key", "Tithe Chest Key", value=1.0, is_key=True,
                                                                        key_door_id="pike_tithe_chest", for_sale=False, category="key")),
               lines=("Osric Pike, Reeve of Emberhollow. The tithe, the roads, the dead dogs in the millstream - all mine.",
                      "Every crown of the town's tithe passes through my chest on its way to Greyhold. Every crown. Give or take.",
                      "The mine is closed. I closed it. Nobody goes down there. Nobody who wants to come up again."),
               schedule=resident_tasks("Pike", "pike_door", p_bed, p_seat, "pike_desk", [
                   TaskDef(8, 11, (px1 + 3.0, 0.0, pz1 + 3.3), AIState.WORKING, "The Reeve is at his ledgers.", facing=180.0,
                           depart_text="The tithe won't count itself."),
                   stroll(11, 13, "the Reeve", ((0.0, 0.0, -12.0), (-13.0, 0.0, -8.2), (11.5, 0.0, -7.8), (38.0, 0.0, 7.4),
                                                (0.0, 0.0, -12.0)), "The Reeve is collecting the market dues."),
                   TaskDef(13, 14, p_seat, AIState.EATING, "The Reeve is eating at his desk.", facing=180.0, food_container="pike_desk"),
                   TaskDef(14, 17, (px1 + 3.0, 0.0, pz1 + 3.3), AIState.WORKING, "The Reeve is counting the tithe.", facing=180.0),
                   visit(17, 19, (51.5, 0.0, -40.6), AIState.WORKING, "The Reeve is conferring with the watch at the guard house.",
                         facing=90.0),
                   TaskDef(19, 20, (px1 + 3.0, 0.0, pz1 + 3.3), AIState.WORKING, "The Reeve is at his ledgers by candlelight.",
                           facing=180.0),
               ], wake=7, bedtime=21, facing=180.0),
               # the other day: the dues of the east end, and the chapel's books with Sister Maude
               rota=(resident_tasks("Pike", "pike_door", p_bed, p_seat, "pike_desk", [
                   TaskDef(8, 10, (px1 + 3.0, 0.0, pz1 + 3.3), AIState.WORKING, "The Reeve is at his ledgers.", facing=180.0,
                           depart_text="The tithe won't count itself."),
                   stroll(10, 13, "the Reeve", ((38.0, 0.0, 4.6), (60.0, 0.0, -7.9), (62.0, 0.0, 24.0), (38.0, 0.0, 4.6)),
                          "The Reeve is collecting the dues in the east of the town."),
                   TaskDef(13, 15, p_seat, AIState.EATING, "The Reeve is eating at his desk.", facing=180.0, food_container="pike_desk"),
                   visit(15, 17, CHAPEL_PEWS[2], AIState.WORKING, "The Reeve is going over the chapel's books with Sister Maude.",
                         facing=180.0),
                   TaskDef(17, 20, (px1 + 3.0, 0.0, pz1 + 3.3), AIState.WORKING, "The Reeve is at his ledgers by candlelight.",
                           facing=180.0),
               ], wake=7, bedtime=21, facing=180.0),)),
        NPCDef("Dunstan the Minstrel", d_bed, role="minstrel", gold=24.0, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.62, 0.2, 0.5), legs=(0.3, 0.6, 0.35), sash=(0.95, 0.8, 0.3), hat="feathered_hat",
                             hat_color=(0.2, 0.5, 0.3), hair=(0.6, 0.4, 0.2)),
               inventory=(key_item("dunstan_door", "Lodging Key"), ItemDef("lute", "Battered Lute", value=35.0, for_sale=False)),
               lines=("Dunstan, minstrel, at your service. Songs of love, of war, of the very lovely weather we've been having.",
                      "I have a new ballad about a thief in the night. Nobody is in it that you would know. Especially not a lady on the hill.",
                      "I sing at the well at two and at the Dragon of an evening. The watch sing along. Badly.",
                      "Lord Vane says my music gives him a headache. Lady Ashgrove says - well. Lady Ashgrove says very little. In public."),
               schedule=resident_tasks("Dunstan", "dunstan_door", d_bed, d_seat, "dunstan_table", [
                   TaskDef(12, 14, (dx1 + 4.0, 0.0, dz1 + 3.0), AIState.WORKING, "Dunstan is tuning his lute and humming.",
                           facing=0.0),
                   TaskDef(14, 16, WELL_SONG, AIState.WORKING, "Dunstan is singing by the well for coppers.", facing=180.0,
                           depart_text="An audience awaits. Possibly."),
                   visit(16, 17, (-4.0, 0.0, -41.0), AIState.WORKING, "Dunstan is lingering below Ashgrove Manor, composing.",
                         facing=0.0),
                   TaskDef(17, 22, INN_STAGE, AIState.WORKING, "Dunstan is playing for the crowd at the Drowsy Dragon.", facing=270.0),
               ], wake=11, supper=(22, 24), facing=180.0),
               # the other day he tries the park, for a better class of copper
               rota=(resident_tasks("Dunstan", "dunstan_door", d_bed, d_seat, "dunstan_table", [
                   TaskDef(12, 13, (dx1 + 4.0, 0.0, dz1 + 3.0), AIState.WORKING, "Dunstan is tuning his lute and humming.",
                           facing=0.0),
                   TaskDef(13, 17, PARK_SONG, AIState.WORKING, "Dunstan is singing in the park for a better class of copper.",
                           facing=0.0, depart_text="The gentry need music. They just don't know it."),
                   TaskDef(17, 22, INN_STAGE, AIState.WORKING, "Dunstan is playing for the crowd at the Drowsy Dragon.", facing=270.0),
               ], wake=11, supper=(22, 24), facing=180.0),)),
    ]
    return p


def tanners_row() -> Part:
    """Tanner's Row, off the east end of Market Street: Mott's house and his reeking tannery, and
    Jeb the lamplighter's, who works the nights and sees everything."""
    p = Part()
    tx1, tz1, tx2, tz2 = TANNER_HOUSE
    jx1, jz1, jx2, jz2 = JEB_HOUSE
    _house(p, "mott", TANNER_HOUSE, 68.0, "north", "Mott's House", "Mott the Tanner", SIMPLE, texture="building", roof="thatch")
    _house(p, "jeb", JEB_HOUSE, 48.0, "north", "Jeb's House", "Jeb the Lamplighter", SIMPLE, texture="plaster")
    kx1, kz1, kx2, kz2 = TANNERY
    p.walls += [WallDef((x, 0, z), (x + 0.3, 3.2, z + 0.3)) for x, z in ((kx1, kz1), (kx2 - 0.3, kz1), (kx1, kz2 - 0.3),
                                                                         (kx2 - 0.3, kz2 - 0.3))]
    p.buildings += [BuildingDef("tannery", (kx1, kz1), (kx2, kz2), wall_height=3.2, roof_texture="thatch", open_sides=True)]
    p.props += [
        bed("mott_bed", (tx2 - 1.55, 0.0, tz2 - 1.6), facing=90.0), bed("jeb_bed", (jx2 - 1.55, 0.0, jz2 - 1.6), facing=90.0),
        *(PropDef(f"tannery_rack_{i}", (kx1 + 1.0 + i * 2.6, 0.0, kz1 + 1.2), (kx1 + 3.0 + i * 2.6, 1.9, kz1 + 1.4), texture="leather",
                  kind="fence") for i in range(3)),
        *barrels("tannery_vats", [(kx1 + 2.0, kz2 - 2.0), (kx1 + 3.2, kz2 - 2.0), (kx1 + 4.4, kz2 - 2.0)]),
        PropDef("tannery_hides", (kx2 - 3.0, 0.0, kz2 - 3.0), (kx2 - 1.0, 0.6, kz2 - 1.0), texture="leather", kind="crates"),
    ]
    p.containers += [
        table("mott_table", "Table", (tx1 + 2.5, 0.0, tz1 + 4.4), owner="Mott the Tanner", size=(0.6, 0.45),
              items=[ItemDef("stew", "Pot of Stew", value=4.0, stackable=True, quantity=9, is_food=True, category="food")]),
        # the lime vat in the corner of the yard: nobody puts a hand in there. Mott's purse sits on a ledge inside
        ContainerDef("mott_lime_vat", "Lime Vat", (kx2 - 1.6, 0.0, kz1 + 1.6), (kx2 - 0.6, 1.0, kz1 + 2.6), owner="Mott the Tanner",
                     texture="barrel", kind="barrel_store",
                     items=(coins(85), ItemDef("fine_leather", "Roll of Fine Leather", value=40.0))),
        table("jeb_table", "Table", (jx1 + 2.4, 0.0, jz1 + 4.2), owner="Jeb the Lamplighter", size=(0.6, 0.45),
              items=[ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=8, is_food=True, category="food")]),
        stash("jeb_brick", "Loose Brick", (jx1 + 0.9, 0.0, jz2 - 1.0), owner="Jeb the Lamplighter",
              items=[coins(58), ItemDef("silver_ring", "Silver Ring Found in a Gutter", value=45.0)]),
    ]
    p.areas += [
        AreaDef("Mott's house", (tx1, tz1), (tx2, tz2), owner="Mott the Tanner", restricted=True),
        AreaDef("the tannery", (kx1, kz1), (kx2, kz2), owner="Mott the Tanner"),
        AreaDef("Jeb's house", (jx1, jz1), (jx2, jz2), owner="Jeb the Lamplighter", restricted=True),
    ]
    p.roads += [RoadDef((60.0, 6.0), (64.0, 22.0), texture="dirt"),                 # Tanner's Row, down from Market Street ...
                RoadDef((44.0, 22.0), (76.0, 26.5), texture="dirt")]                # ... and along past the houses
    m_bed, m_seat = (tx2 - 1.55, 0.0, tz2 - 1.6), (tx1 + 2.5, 0.0, tz1 + 5.6)
    j_bed, j_seat = (jx2 - 1.55, 0.0, jz2 - 1.6), (jx1 + 2.4, 0.0, jz1 + 5.5)
    p.npcs += [
        NPCDef("Mott the Tanner", m_bed, role="tanner", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.45, 0.36, 0.28), legs=(0.3, 0.26, 0.22), apron="leather", hair=(0.2, 0.16, 0.12)),
               inventory=(key_item("mott_door", "House Key"),),
               lines=("Yes, it's me you can smell. Everyone can smell me. Lime, bark and dog. It's an honest stink.",
                      "Nobody steals from a tannery. Nobody puts their hand in a lime vat. That's what I'm counting on.",
                      "Jeb next door's up all night with his torches. Sees every cat, every drunk, every shadow on the wrong roof."),
               schedule=resident_tasks("Mott", "mott_door", m_bed, m_seat, "mott_table", [
                   TaskDef(7, 12, (kx1 + 3.0, 0.0, kz1 + 2.4), AIState.WORKING, "Mott is scraping hides in the tannery.", facing=0.0,
                           depart_text="Hides to scrape. Joy."),
                   TaskDef(12, 13, m_seat, AIState.EATING, "Mott is eating.", facing=180.0, food_container="mott_table",
                           unlock_doors=("mott_door",)),
                   TaskDef(13, 17, (kx1 + 3.2, 0.0, kz2 - 3.2), AIState.WORKING, "Mott is stirring the tanning vats.", facing=180.0),
                   visit(17, 20, (-36.75 - 1.35, 0.0, -8.2), AIState.DRINKING, "Mott is drinking at the inn, a table to himself.",
                         facing=90.0),
               ], wake=6, supper=(20, 22), facing=180.0),                    # (the inn is the far end of town)
               # the other day: no inn - the hides want salting, and the cooper wants a word about hoops
               rota=(resident_tasks("Mott", "mott_door", m_bed, m_seat, "mott_table", [
                   TaskDef(7, 12, (kx1 + 3.0, 0.0, kz1 + 2.4), AIState.WORKING, "Mott is scraping hides in the tannery.", facing=0.0,
                           depart_text="Hides to scrape. Joy."),
                   TaskDef(12, 13, m_seat, AIState.EATING, "Mott is eating.", facing=180.0, food_container="mott_table",
                           unlock_doors=("mott_door",)),
                   visit(13, 15, COOPER_YARD, AIState.WORKING, "Mott is haggling with Corwin over a barrel for the lime."),
                   TaskDef(15, 20, (kx1 + 3.2, 0.0, kz2 - 3.2), AIState.WORKING, "Mott is salting hides late into the evening.",
                           facing=180.0),
               ], wake=6, bedtime=21, facing=180.0),)),
        NPCDef("Jeb the Lamplighter", j_bed, role="lamplighter", gold=TOWNSFOLK_GOLD, inventory_reset_days=3,
               outfit=Outfit(tunic=(0.28, 0.3, 0.38), legs=(0.22, 0.22, 0.28), hat="cap", hat_color=(0.2, 0.2, 0.28), hair=(0.5, 0.45, 0.4)),
               inventory=(key_item("jeb_door", "House Key"), ItemDef("taper_pole", "Lamplighter's Pole", value=5.0, for_sale=False)),
               lines=("Every torch on the walls, I light at dusk and snuff at dawn. Forty-one of them. Nobody ever thanks the dark.",
                      "You see things, at night. A shadow on the Vanes' roof. Somebody crouching by the store's back door. I say nothing. Mostly.",
                      "The watch don't like me. I'm awake when they're not, and I know where they nap."),
               schedule=(
                   TaskDef(0, 4, TORCH_ROUND[6], AIState.STROLLING, "Jeb is walking the dark streets, seeing everything.",
                           patrol_points=TORCH_ROUND[6:] + TORCH_ROUND[:6]),
                   TaskDef(4, 6, TORCH_ROUND[0], AIState.STROLLING, "Jeb is snuffing the torches as the sky pales.",
                           patrol_points=TORCH_ROUND),
                   TaskDef(6, 8, j_seat, AIState.EATING, "Jeb is having his supper at breakfast time.", facing=180.0,
                           food_container="jeb_table", unlock_doors=("jeb_door",)),   # (the gate is the far end of his round)
                   TaskDef(8, 15, j_bed, AIState.SLEEPING, "Jeb is asleep, curtains drawn.", lock_doors=("jeb_door",)),
                   TaskDef(15, 16, j_seat, AIState.EATING, "Jeb is having his breakfast at teatime.", facing=180.0,
                           food_container="jeb_table", unlock_doors=("jeb_door",)),
                   visit(16, 18, (-3.0, 0.0, 46.0), AIState.WORKING, "Jeb is trimming wicks by the gate.", facing=180.0),
                   TaskDef(18, 20, TORCH_ROUND[0], AIState.STROLLING, "Jeb is lighting the torches at dusk.", patrol_points=TORCH_ROUND),
                   visit(20, 22, (-24.75 + 1.35, 0.0, -7.8), AIState.DRINKING, "Jeb is having an ale before the long night.",
                         facing=270.0),
                   TaskDef(22, 24, TORCH_ROUND[3], AIState.STROLLING, "Jeb is walking the dark streets.",
                           patrol_points=TORCH_ROUND[3:] + TORCH_ROUND[:3]),
               )),
    ]
    return p


TAILOR_FLOOR_Q = (11.5, 0.0, -8.0)

PARTS = (chapel(), cottages(), bakery(), cooper(), stable(), upper_street(), tanners_row())
