"""The Halcyon staff door: the yellow hazard door the player leaves the greenhouse by and
turns round to see from the alley. Both levels build it from here, so the two sides of
the same door are the same door - leaf, frame head, the STAFF ONLY neon sign, the alarm
lamp and the concrete portal around it.

Positions are given for a wall face at ``face_z`` whose outward normal points along
``facing`` (+1: into +z, -1: into -z); everything is mounted proud of that face.
"""

from __future__ import annotations

from cyberduke.world.props import Material, Prop

STAFF_DOOR_WIDTH = 3.2
STAFF_DOOR_HEIGHT = 2.6
STAFF_DOOR = Material("hazard", (0.85, 0.85, 0.85))
STAFF_DOOR_FRAME = Material("metal", (0.2, 0.2, 0.22))
STAFF_DOOR_SIGN = Material("neon_cyan", (0.2, 0.9, 1.0), emission=(0.5, 2.2, 2.6))
STAFF_DOOR_ALARM = Material(None, (0.25, 0.03, 0.03), emission=(1.2, 0.1, 0.1))
PORTAL = Material("concrete", (0.1, 0.1, 0.115))
DECAL = frozenset({"transparent"})


def staff_door_leaf(x: float, face_z: float, facing: int, thickness: float = 0.08) -> Prop:
    """The door itself (proud of the wall by its thickness, as in the alley)."""
    return Prop.box(x, STAFF_DOOR_HEIGHT / 2.0, face_z + facing * (thickness / 2.0 - 0.02),
                    STAFF_DOOR_WIDTH, STAFF_DOOR_HEIGHT, thickness, STAFF_DOOR,
                    label="Halcyon staff door (yellow)")  # fmt: skip


def staff_door_alarm(x: float, face_z: float, facing: int) -> Prop:
    """The lamp over the door (in the greenhouse it is the door's status light)."""
    return Prop.box(x, 2.9, face_z + facing * 0.06, 0.4, 0.12, 0.1, STAFF_DOOR_ALARM,
                    label="Door alarm light", tags=DECAL)  # fmt: skip


def staff_door_trim(x: float, face_z: float, facing: int, portal: bool = False) -> tuple[Prop, ...]:
    """Everything around the opening: the frame head, the neon sign and (``portal``) a
    concrete surround for a door set in a wall that is not concrete itself."""
    props = [
        Prop.box(x, 2.75, face_z + facing * 0.04, 3.58, 0.16, 0.2, STAFF_DOOR_FRAME, label="Door frame head"),
        Prop.box(x, 3.3, face_z + facing * 0.05, 2.4, 0.3, 0.05, STAFF_DOOR_SIGN, label="HALCYON DYNAMICS - STAFF ONLY", tags=DECAL),
    ]  # fmt: skip
    if portal:
        half = STAFF_DOOR_WIDTH / 2.0
        z = face_z + facing * 0.02
        for side in (-1, 1):
            props.append(Prop.box(x + side * (half + 0.35), 2.0, z, 0.7, 4.0, 0.04, PORTAL, label="Concrete door portal"))
        # (its underside 5 mm below the wall's: level with it, the two z-fought)
        bottom = STAFF_DOOR_HEIGHT - 0.005
        props.append(Prop.box(x, (bottom + 4.0) / 2.0, z, STAFF_DOOR_WIDTH, 4.0 - bottom, 0.04, PORTAL, label="Concrete door portal"))  # fmt: skip
    return tuple(props)


# ----------------------------------------------------------------------------- easing doors
DOOR_GIVE = 0.12  # metres each side: how much wider a front door is to walk through than to see
PASSAGE_REACH = 0.2  # metres a doorway's walkable strip reaches on into the rooms either side


def ease_doorways(level, give: float = DOOR_GIVE):
    """The level with every hinged door (the homes' front doors, the bar's) a little
    easier to walk through: the wall either side of the opening stops ``give`` short of
    it for anybody walking (the drawn wall and door frame are as they were - nobody's
    body is wide enough for the difference to show), and the doorway's walkable strip
    (``DoorData.passage``) widens to match. A body only just fitting a 1 m doorway had
    to come at it dead centre - 13 cm either side and it stopped on the jamb."""
    from dataclasses import replace

    from cyberduke.core.geometry import AABB

    eased = {}
    passages = {}
    for d in level.doors:
        if d.hinge is None:
            continue
        o = d.obstacle
        along_z = d.inside in ("-x", "+x")  # (the wall runs along z: the opening is in z)
        for sector in level.sectors:
            for c in sector.colliders:
                if c.min_y > 0.5:
                    continue  # (the lintel over the door)
                if along_z:
                    band = c.min_x < o.max_x and c.max_x > o.min_x
                    lo, hi, a, b = c.min_z, c.max_z, o.min_z, o.max_z
                else:
                    band = c.min_z < o.max_z and c.max_z > o.min_z
                    lo, hi, a, b = c.min_x, c.max_x, o.min_x, o.max_x
                if not band or hi - lo < give + 0.1:
                    continue
                cut_hi = abs(hi - a) < 0.03  # (the wall ends at the opening's near side)
                cut_lo = abs(lo - b) < 0.03  # (... or starts at its far side)
                if not (cut_hi or cut_lo):
                    continue
                box = eased.get(id(c), c)
                if along_z:
                    box = replace(
                        box,
                        max_z=box.max_z - give if cut_hi else box.max_z,
                        min_z=box.min_z + give if cut_lo else box.min_z,
                    )
                else:
                    box = replace(
                        box,
                        max_x=box.max_x - give if cut_hi else box.max_x,
                        min_x=box.min_x + give if cut_lo else box.min_x,
                    )
                eased[id(c)] = box
        if d.passage is not None:
            # (and a little deeper either way, so it always overlaps the floor inside and
            # out: it stopped 2 cm short of a flat's front room - a body stepping in could
            # land in the gap and stop dead in the doorway)
            p = d.passage
            if along_z:
                passages[d.door_id] = replace(
                    p, min_z=p.min_z - give, max_z=p.max_z + give,
                    min_x=p.min_x - PASSAGE_REACH, max_x=p.max_x + PASSAGE_REACH,
                )  # fmt: skip
            else:
                passages[d.door_id] = replace(
                    p, min_x=p.min_x - give, max_x=p.max_x + give,
                    min_z=p.min_z - PASSAGE_REACH, max_z=p.max_z + PASSAGE_REACH,
                )  # fmt: skip
    sectors = tuple(
        replace(s, colliders=tuple(eased.get(id(c), c) for c in s.colliders))
        for s in level.sectors
    )
    doors = tuple(
        replace(d, passage=passages[d.door_id]) if d.door_id in passages else d
        for d in level.doors
    )
    assert all(isinstance(c, AABB) for s in sectors for c in s.colliders)
    return replace(level, sectors=sectors, doors=doors)
