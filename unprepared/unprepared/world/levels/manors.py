"""The two great houses on Upper Street: Ashgrove Manor and Vane Manor.

Each is two storeys on a footprint 20 by 16, dressed stone below and limewashed render above,
a cornice between, sash windows with shutters all round, a pillared porch over the door and
chimneys on the ridge - the finest houses in Emberhollow, and they know it. Inside:

    ground floor  (the door in the middle of the south front, the great stair beside it)
        kitchen (range, work table, pantry) | the hall       | dining room (table, fireplace)
        store room (wine, crates, silver)   | and its stair  | the servant's room
    upper floor
        study (books, a desk, a globe)      | the landing    | the master bedroom (bed, wardrobe, strongbox)
        dressing room (a bath, wardrobes)   |                | a second bedroom / boudoir

:func:`manor` returns the level parts as a dict (see ``town.py``) plus the named spots
(``bed``, ``seat``, ``landing``, ``stair_foot``, ``servant_bed``, ``servant_seat``, ``kitchen``...).
Everything is authored with its floor at y = 0 (the ground floor) or y = 4 (the upper one).
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

from ..builders import FLOOR_THICKNESS, WALL, bed, chest, coins, partition_x, partition_z, stash, straight_stair, table, torches_beside_door, upper_floor
from ..level import AreaDef, BuildingDef, ContainerDef, DoorDef, ItemDef, LightDef, PropDef, WallDef

MANOR_W, MANOR_D = 20.0, 16.0
STOREY = 4.0
UNDER_BOARDS = STOREY - FLOOR_THICKNESS      # the top of a wall downstairs: the underside of the boards over it
HEIGHT = 8.0
DOOR_U = 8.5                 # the front door's west jamb, from the west wall
FACADE = "ashlar"            # the ground storey's dressed stone ...
UPPER = "manor_plaster"      # ... and the render over it
INSIDE = "wallpaper"         # the partitions' papered walls


def _storey(x1, z1, x2, z2, y0, y1, texture, door: Optional[tuple] = None) -> List[WallDef]:
    """Four walls of one storey (``door``: (x, width) in the south wall)."""
    walls = [WallDef((x1, y0, z1), (x1 + WALL, y1, z2), texture=texture),
             WallDef((x2 - WALL, y0, z1), (x2, y1, z2), texture=texture),
             WallDef((x1 + WALL, y0, z1), (x2 - WALL, y1, z1 + WALL), texture=texture)]
    if door is None:
        walls.append(WallDef((x1 + WALL, y0, z2 - WALL), (x2 - WALL, y1, z2), texture=texture))
    else:
        dx, dw = door
        walls += [WallDef((x1 + WALL, y0, z2 - WALL), (dx, y1, z2), texture=texture),
                  WallDef((dx + dw, y0, z2 - WALL), (x2 - WALL, y1, z2), texture=texture)]
    return walls


def _windows(prefix, x1, z1, x2, z2) -> List[PropDef]:
    """Sash windows on every face, both storeys (decals on the stone and the render)."""
    out = []
    k = 0

    def window(x, z, facing, y):
        nonlocal k
        along_x = facing in (0.0, 180.0)
        hx, hz = (0.75, 0.02) if along_x else (0.02, 0.75)
        out.append(PropDef(f"{prefix}_window_{k}", (x - hx, y, z - hz), (x + hx, y + 2.2, z + hz), texture="window_manor",
                           kind="window", facing=facing, solid=False))
        k += 1

    for u in (2.3, 5.0, 15.0, 17.7):                                   # the front: either side of the porch
        window(x1 + u, z2, 180.0, 0.9)
    for u in (2.3, 5.0, 10.0, 15.0, 17.7):
        window(x1 + u, z2, 180.0, 4.9)
    for u in (2.3, 5.0, 10.0, 15.0, 17.7):                             # the back
        window(x1 + u, z1, 0.0, 0.9 if u != 10.0 else 1.6)
        window(x1 + u, z1, 0.0, 4.9)
    for v in (4.0, 12.0):                                              # the ends
        for x, facing in ((x1, 270.0), (x2, 90.0)):
            window(x, z1 + v, facing, 0.9)
            window(x, z1 + v, facing, 4.9)
    return out


def manor(prefix: str, title: str, x1: float, z1: float, owner: str, door_lock: int, servant: str,
          upstairs: List[ContainerDef], downstairs: List[ContainerDef], wealth: str = "vane"):
    """A great house (see the module docstring). ``upstairs`` / ``downstairs``: the owner's own
    chests and treasures, placed by the caller at the spots this returns in ``slots``."""
    x2, z2 = x1 + MANOR_W, z1 + MANOR_D
    u = lambda a: x1 + a            # noqa: E731
    v = lambda b: z1 + b            # noqa: E731
    door_x = u(DOOR_U)
    walls: List[WallDef] = []
    walls += _storey(x1, z1, x2, z2, 0.0, STOREY, FACADE, door=(door_x, 3.0))
    walls += _storey(x1, z1, x2, z2, STOREY, HEIGHT, UPPER)
    # the cornice between the storeys and the quoins up the corners of the render
    c, t = 0.14, 0.3
    walls += [WallDef((x1 - c, STOREY - t, z1 - c), (x2 + c, STOREY, z1), texture="marble"),
              WallDef((x1 - c, STOREY - t, z2), (x2 + c, STOREY, z2 + c), texture="marble"),
              WallDef((x1 - c, STOREY - t, z1), (x1, STOREY, z2), texture="marble"),
              WallDef((x2, STOREY - t, z1), (x2 + c, STOREY, z2), texture="marble")]
    for qx, qz in ((x1, z1), (x2, z1), (x1, z2), (x2, z2)):
        ax1, ax2 = (qx - 0.06, qx + 0.45) if qx == x1 else (qx - 0.45, qx + 0.06)
        az1, az2 = (qz - 0.06, qz + 0.45) if qz == z1 else (qz - 0.45, qz + 0.06)
        walls.append(WallDef((ax1, STOREY, az1), (ax2, HEIGHT, az2), texture=FACADE))
    # the rooms downstairs: the kitchen and the store to the west of the hall, the dining room and
    # the servant's room to the east, a doorway from the hall into each
    # (downstairs the walls stop under the upper floor's boards: topped flush with them, they showed
    # through the floor upstairs as flickering stripes)
    for y, h in ((0.0, UNDER_BOARDS), (STOREY, STOREY)):
        walls += partition_x(u(7.0), v(WALL), v(2.6), height=h, y=y, texture=INSIDE)
        walls += partition_x(u(13.0), v(WALL), v(6.2), height=h, y=y, texture=INSIDE)
        walls += partition_x(u(13.0), v(8.2), v(11.2), height=h, y=y, texture=INSIDE)
        walls += partition_x(u(13.0), v(13.0), v(MANOR_D - WALL), height=h, y=y, texture=INSIDE)
        walls += partition_z(v(9.0), u(13.3), u(MANOR_W - WALL), height=h, y=y, texture=INSIDE)
    # (downstairs a doorway from the hall into the store; upstairs the stairwell runs along that
    # wall, so the dressing room is reached through the study instead)
    walls += partition_x(u(7.0), v(4.6), v(11.2), height=UNDER_BOARDS, texture=INSIDE)
    walls += partition_x(u(7.0), v(13.0), v(MANOR_D - WALL), height=UNDER_BOARDS, texture=INSIDE)
    walls += partition_x(u(7.0), v(4.6), v(MANOR_D - WALL), height=STOREY, y=STOREY, texture=INSIDE)
    walls += partition_z(v(8.0), u(WALL), u(7.0), height=UNDER_BOARDS, door=(u(2.6), u(4.4)), texture=INSIDE)     # kitchen - store
    walls += partition_z(v(8.0), u(WALL), u(7.0), height=STOREY, y=STOREY, door=(u(4.4), u(6.2)),
                         texture=INSIDE)                                                                     # study - dressing room

    # the great stair: up the hall from beside the door, the landing at its head
    flight = straight_stair(f"{prefix}_stair", u(7.3), u(8.9), v(14.3), STOREY, region=(x1, z1, x2, z2))
    study_door, dressing_door = (u(7.0), STOREY, v(3.6)), (u(5.3), STOREY, v(8.0))
    flight.stair = dataclasses.replace(flight.stair, rooms=(
        ((x1, z1, u(7.0), v(8.0)), (study_door,)),                                     # the study
        ((x1, v(8.0), u(7.0), z2), (study_door, dressing_door)),                       # the dressing room, through it
        ((u(13.0), z1, x2, v(9.0)), ((u(13.0), STOREY, v(7.2)),)),                     # the bedroom
        ((u(13.0), v(9.0), x2, z2), ((u(13.0), STOREY, v(12.1)),)),                    # the second bedroom / music room
    ))
    hx1, hz1, hx2, hz2 = flight.hole
    props: List[PropDef] = list(flight.props)
    props += upper_floor(f"{prefix}_upper_floor", x1 + WALL, z1 + WALL, x2 - WALL, z2 - WALL, STOREY, [flight.hole])
    props += [PropDef(f"{prefix}_rail_east", (hx2, STOREY, hz1 + 1.2), (hx2 + 0.1, STOREY + 0.95, hz2), texture="wood"),
              PropDef(f"{prefix}_rail_south", (hx1, STOREY, hz2), (hx2 + 0.1, STOREY + 0.95, hz2 + 0.1), texture="wood")]

    bed_spot = (u(MANOR_W - WALL - 1.05), STOREY, v(3.0))
    seat = (u(16.8), 0.0, v(5.0))                       # at the dining table, facing it (north)
    servant_bed = (u(MANOR_W - WALL - 1.1), 0.0, v(13.8))
    servant_seat = (u(3.5), 0.0, v(5.7))                # at the kitchen table
    kitchen = (u(2.0), 0.0, v(2.6))                     # at the range
    store = (u(3.0), 0.0, v(12.0))
    study = (u(3.5), STOREY, v(4.2))                    # at the writing desk
    dressing = (u(3.5), STOREY, v(12.5))
    hall = (u(10.0), 0.0, v(10.0))

    # --- furnishing: downstairs
    props += [
        PropDef(f"{prefix}_range", (u(0.8), 0.0, v(WALL)), (u(3.2), 2.2, v(1.6)), texture="iron", kind="stove", facing=180.0),
        PropDef(f"{prefix}_kitchen_shelf", (u(WALL), 0.0, v(1.8)), (u(0.95), 2.0, v(5.0)), texture="shelf"),
        PropDef(f"{prefix}_kitchen_barrel", (u(5.8), 0.0, v(0.8)), (u(6.6), 1.0, v(1.6)), texture="barrel", kind="barrel"),
        PropDef(f"{prefix}_winerack", (u(WALL), 0.0, v(9.0)), (u(0.95), 2.0, v(12.0)), texture="wood", kind="winerack", facing=90.0),
        PropDef(f"{prefix}_store_crates", (u(4.5), 0.0, v(13.4)), (u(6.6), 1.4, v(15.4)), texture="crate", kind="crates"),
        PropDef(f"{prefix}_store_barrels", (u(1.0), 0.0, v(14.2)), (u(1.8), 1.0, v(15.0)), texture="barrel", kind="barrel"),
        PropDef(f"{prefix}_hall_rug", (u(9.4), 0.0, v(3.0)), (u(12.6), 0.05, v(12.5)), texture="rug_blue" if wealth == "vane" else "rug",
                kind="rug", solid=False),
        PropDef(f"{prefix}_clock", (u(12.2), 0.0, v(0.6)), (u(12.7), 2.3, v(1.2)), texture="wood", kind="clock", facing=270.0),
        PropDef(f"{prefix}_bust", (u(9.6), 0.0, v(0.8)), (u(10.4), 1.9, v(1.6)), texture="marble", kind="bust"),
        PropDef(f"{prefix}_hall_chandelier", (u(10.0) - 0.5, 2.7, v(8.0) - 0.5), (u(10.0) + 0.5, 3.9, v(8.0) + 0.5), texture="gilt",
                kind="chandelier", solid=False),
        PropDef(f"{prefix}_fireplace", (u(MANOR_W - WALL - 0.7), 0.0, v(5.6)), (u(MANOR_W - WALL), 1.8, v(8.2)), texture="marble",
                kind="fireplace", facing=270.0),
        PropDef(f"{prefix}_dining_chandelier", (u(16.8) - 0.6, 2.8, v(3.6) - 0.6), (u(16.8) + 0.6, 3.9, v(3.6) + 0.6),
                texture="gilt", kind="chandelier", solid=False),
        PropDef(f"{prefix}_dining_rug", (u(14.4), 0.0, v(1.6)), (u(19.0), 0.05, v(5.8)), texture="rug", kind="rug", solid=False),
        PropDef(f"{prefix}_painting_dining", (u(17.2) - 0.9, 1.6, v(WALL)), (u(17.2) + 0.9, 3.1, v(WALL) + 0.02),
                texture="painting_portrait", kind="painting", facing=180.0, solid=False),
        PropDef(f"{prefix}_painting_hall", (u(7.0) + 0.3, 1.4, v(6.2)), (u(7.0) + 0.32, 2.9, v(8.2)), texture="painting_landscape",
                kind="painting", facing=90.0, solid=False),
        bed(f"{prefix}_servant_bed", servant_bed, facing=90.0),
    ]
    # --- furnishing: upstairs
    s = STOREY
    props += [
        bed(f"{prefix}_bed", bed_spot, facing=90.0, double=True),
        PropDef(f"{prefix}_wardrobe_left", (u(13.6), s, v(WALL)), (u(15.0), s + 2.5, v(1.15)), texture="wood", kind="wardrobe",
                facing=180.0),
        PropDef(f"{prefix}_bedroom_rug", (u(14.0), s, v(4.2)), (u(17.6), s + 0.05, v(8.2)), texture="rug", kind="rug", solid=False),
        PropDef(f"{prefix}_painting_bedroom", (u(13.3) + 0.01, s + 1.3, v(2.6)), (u(13.3) + 0.03, s + 2.7, v(4.3)),
                texture="painting_landscape", kind="painting", facing=90.0, solid=False),
        PropDef(f"{prefix}_bookcase_w", (u(WALL), s, v(1.2)), (u(1.0), s + 3.0, v(7.0)), texture="shelf", kind="bookcase", facing=90.0),
        PropDef(f"{prefix}_bookcase_n", (u(1.3), s, v(WALL)), (u(6.6), s + 3.0, v(1.0)), texture="shelf", kind="bookcase", facing=180.0),
        PropDef(f"{prefix}_globe", (u(1.4), s, v(6.9)), (u(2.2), s + 1.3, v(7.6)), texture="wood", kind="globe"),
        PropDef(f"{prefix}_study_rug", (u(1.6), s, v(2.6)), (u(5.6), s + 0.05, v(6.6)), texture="rug_blue", kind="rug", solid=False),
        PropDef(f"{prefix}_bathtub", (u(1.0), s, v(13.0)), (u(3.2), s + 0.8, v(15.3)), texture="gilt", kind="bathtub"),
        PropDef(f"{prefix}_wardrobe_dressing", (u(WALL), s, v(9.0)), (u(1.15), s + 2.5, v(11.2)), texture="wood", kind="wardrobe",
                facing=90.0),
        PropDef(f"{prefix}_painting_dressing", (u(7.0) - 0.02, s + 1.3, v(11.6)), (u(7.0), s + 2.6, v(12.9)),
                texture="painting_portrait", kind="painting", facing=270.0, solid=False),
        PropDef(f"{prefix}_landing_rug", (u(9.4), s, v(8.6)), (u(12.6), s + 0.05, v(15.0)), texture="rug", kind="rug", solid=False),
        PropDef(f"{prefix}_landing_chandelier", (u(10.5) - 0.5, s + 2.8, v(11.0) - 0.5), (u(10.5) + 0.5, s + 3.9, v(11.0) + 0.5),
                texture="gilt", kind="chandelier", solid=False),
        PropDef(f"{prefix}_landing_bust", (u(12.2), s, v(1.0)), (u(12.9), s + 1.8, v(1.7)), texture="marble", kind="bust"),
    ]
    if wealth == "ashgrove":
        props += [PropDef(f"{prefix}_harpsichord", (u(14.0), s, v(11.5)), (u(16.4), s + 1.1, v(14.0)), texture="wood",
                          kind="harpsichord", facing=0.0),
                  PropDef(f"{prefix}_sofa", (u(17.2), s, v(10.6)), (u(19.3), s + 1.0, v(11.6)), texture="cloth", kind="sofa",
                          facing=180.0)]
    else:
        props += [bed(f"{prefix}_guest_bed", (u(MANOR_W - WALL - 1.05), s, v(13.2)), facing=90.0),
                  PropDef(f"{prefix}_sofa", (u(14.0), s, v(14.6)), (u(16.2), s + 1.0, v(15.4)), texture="rug_blue", kind="sofa",
                          facing=0.0)]

    # --- the outside: windows, the porch, chimneys, the garden
    props += _windows(prefix, x1, z1, x2, z2)
    props += [
        PropDef(f"{prefix}_portico", (door_x - 0.8, 0.0, z2), (door_x + 3.8, 4.7, z2 + 1.8), texture="marble", kind="portico",
                facing=180.0, solid=False),
        # (what the porch's columns are to walk into: the porch draws them, round - these are not drawn,
        # and were: square blocks round the columns, their faces fighting the beam's)
        PropDef(f"{prefix}_portico_col_w", (door_x - 0.8, 0.0, z2 + 1.1), (door_x - 0.16, 3.5, z2 + 1.74), texture="marble",
                kind="collider", solid=True),
        PropDef(f"{prefix}_portico_col_e", (door_x + 3.16, 0.0, z2 + 1.1), (door_x + 3.8, 3.5, z2 + 1.74), texture="marble",
                kind="collider", solid=True),
        PropDef(f"{prefix}_chimney_w", (u(2.0), HEIGHT - 0.5, v(7.3)), (u(3.2), HEIGHT + 4.6, v(8.7)), texture="brick", kind="chimney",
                solid=False),
        PropDef(f"{prefix}_chimney_e", (u(17.0), HEIGHT - 0.5, v(7.3)), (u(18.2), HEIGHT + 4.6, v(8.7)), texture="brick",
                kind="chimney", solid=False),
        PropDef(f"{prefix}_topiary_w", (door_x - 2.2, 0.0, z2 + 0.6), (door_x - 1.2, 2.2, z2 + 1.6), texture="leaves", kind="topiary"),
        PropDef(f"{prefix}_topiary_e", (door_x + 4.2, 0.0, z2 + 0.6), (door_x + 5.2, 2.2, z2 + 1.6), texture="leaves", kind="topiary"),
        *(PropDef(f"{prefix}_planter_{i}", (u(a) - 0.8, 0.0, z2 + 0.05), (u(a) + 0.8, 0.6, z2 + 0.5), texture="flowers", kind="planter")
          for i, a in enumerate((2.3, 5.0, 15.0, 17.7))),
    ]
    containers = [
        table(f"{prefix}_dining_table", "Dining Table", (u(16.8), 0.0, v(3.6)), size=(1.3, 0.6), owner=owner,
              items=[ItemDef("wine", "Bottle of Wine", value=12.0, stackable=True, quantity=2, category="food"),
                     ItemDef("roast_pheasant", "Roast Pheasant", value=14.0, stackable=True, quantity=10, is_food=True,
                             category="food")]),                     # (three meals a day, three days a larder)
        table(f"{prefix}_kitchen_table", "Kitchen Table", (u(3.5), 0.0, v(4.4)), size=(1.0, 0.55), owner=owner,
              items=[ItemDef("bread", "Fresh Bread", value=3.0, stackable=True, quantity=9, is_food=True, category="food"),
                     ItemDef("cheese", "Wheel of Cheese", value=9.0, stackable=True, quantity=2, is_food=True, category="food")]),
        ContainerDef(f"{prefix}_pantry", "Pantry Shelves", (u(5.2), 0.0, v(6.4)), (u(6.7), 2.0, v(7.7)), owner=owner, texture="shelf",
                     items=(ItemDef("honey", "Jar of Honey", value=9.0, stackable=True, quantity=3, category="food"),
                            ItemDef("ham", "Smoked Ham", value=16.0, is_food=True, category="food"),
                            ItemDef("spices", "Pot of Saffron", value=55.0))),
        chest(f"{prefix}_silver_cabinet", "Silver Cabinet", (u(1.0), 0.0, v(13.0)), owner=owner, along_x=False, locked=True,
              lock_difficulty=3, key_id=f"{prefix}_door",
              items=[ItemDef("silver_teapot", "Silver Teapot", value=85.0), ItemDef("silver_cutlery", "Silver Cutlery", value=60.0)]),
        ContainerDef(f"{prefix}_sideboard", "Sideboard", (u(13.6), 0.0, v(WALL) + 0.05), (u(15.2), 1.1, v(1.15)), owner=owner,
                     texture="shelf",
                     items=(ItemDef("crystal_decanter", "Crystal Decanter", value=70.0),
                            ItemDef("wine", "Bottle of Wine", value=12.0, stackable=True, quantity=3, category="food"))),
        chest(f"{prefix}_servant_chest", f"{servant}'s Chest", (u(MANOR_W - WALL - 0.4), 0.0, v(10.6)), owner=servant, along_x=False,
              items=[coins(9), ItemDef("apron", "Starched Apron", value=4.0, category="clothing")]),
        table(f"{prefix}_writing_desk" if wealth == "vane" else f"{prefix}_dressing_table",
              "Writing Desk" if wealth == "vane" else "Dressing Table", (u(3.5), s, v(3.4)), owner=owner, size=(0.8, 0.45),
              items=[ItemDef("gold_quill", "Gilded Quill", value=45.0), ItemDef("wax_seal", "Vane Seal", value=25.0)] if wealth == "vane"
              else [ItemDef("perfume", "Bottle of Perfume", value=40.0), ItemDef("silver_comb", "Silver Comb", value=35.0)]),
        ContainerDef(f"{prefix}_clothes_press", "Clothes Press", (u(4.4), s, v(15.0)), (u(6.6), s + 1.6, v(MANOR_D - WALL)),
                     owner=owner, texture="shelf",
                     items=(ItemDef("silk_gloves", "Silk Gloves", value=60.0, category="clothing", noise_modifier=-0.1),
                            ItemDef("fine_shirt", "Fine Lawn Shirt", value=40.0, category="clothing"))),
        *downstairs, *upstairs,
    ]
    area = AreaDef(title, (x1, z1), (x2, z2), owner=owner, restricted=True)
    door = DoorDef(f"{prefix}_door", (door_x, 0.0, z2), is_locked=True, key_id=f"{prefix}_door_key", owners=(owner, servant),
                   lock_difficulty=door_lock, label=title)
    # the servant's own room, off the hall: a door in the partition (a wall running north-south),
    # swinging into the room; the servant keeps its key
    servant_door = DoorDef(f"{prefix}_servant_door", (u(13.15), 0.0, v(11.2)), axis="z", width=1.8, height=3.0, swing_out=True,
                           is_locked=False, key_id=f"{prefix}_servant_door_key", owners=(servant,), lock_difficulty=2,
                           label=f"{servant.split()[0]}'s Room")
    lights = [
        LightDef("lantern", (u(3.5), 2.9, v(WALL) + 0.25), facing=180.0, room=f"{title} kitchen", range=7.0),
        LightDef("lantern", (u(3.5), 2.9, v(MANOR_D - WALL) - 0.25), facing=0.0, room=f"{title} store", range=6.0),
        LightDef("candle", (u(10.0), 2.85, v(8.0)), room=f"{title} hall", range=8.0),
        LightDef("candle", (u(16.8), 2.95, v(3.6)), room=f"{title} dining room", range=8.0),
        LightDef("fire", (u(MANOR_W - WALL - 0.45), 0.5, v(6.9)), room=f"{title} dining room", range=6.0),
        LightDef("lantern", (u(MANOR_W - WALL) - 0.25, 2.9, v(12.5)), facing=270.0, room=f"{title} servant's room", range=6.0),
        LightDef("lantern", (u(7.0) - 0.25, 6.9, v(6.0)), facing=270.0, room=f"{title} study", range=7.0),   # (books on the others)
        LightDef("lantern", (u(3.5), 6.9, v(MANOR_D - WALL) - 0.25), facing=0.0, room=f"{title} dressing room", range=7.0),
        LightDef("candle", (u(10.5), 6.95, v(11.0)), room=f"{title} landing", range=8.0),
        LightDef("lantern", (u(16.5), 6.9, v(WALL) + 0.25), facing=180.0, room=f"{title} bedroom", range=8.0),
        LightDef("lantern", (u(16.5), 6.9, v(MANOR_D - WALL) - 0.25), facing=0.0, room=f"{title} second bedroom", range=7.0),
        *torches_beside_door(door_x - 0.8, door_x + 3.8, z2, 180.0, title),
    ]
    building = BuildingDef(prefix, (x1, z1), (x2, z2), wall_height=HEIGHT, roof_height=3.4, roof_texture="roof_slate",
                           floor_texture="carpet", floors=2, lit=False)
    slots = dict(strongbox=(u(15.8), s, v(0.9)), wardrobe=(u(18.4), s, v(0.9)), coffer=(u(MANOR_W - WALL - 0.4), 0.0, v(1.6)),
                 hidden=(u(1.6), 0.0, v(7.1)))
    return dict(walls=walls, props=props, containers=containers, area=area, door=door, servant_door=servant_door, lights=lights,
                stair=flight.stair,
                building=building, bed=bed_spot, seat=seat, landing=flight.landing, stair_foot=flight.stair.bottom,
                servant_bed=servant_bed, servant_seat=servant_seat, kitchen=kitchen, store=store, study=study, dressing=dressing,
                hall=hall, slots=slots, door_x=door_x, x2=x2, z2=z2)


def servant_stash(prefix: str, servant: str, spot, items) -> ContainerDef:
    """What a servant has put by where the family never looks: under a loose board in the kitchen."""
    return stash(f"{prefix}_servant_stash", "Loose Floorboard", spot, owner=servant, items=items)
