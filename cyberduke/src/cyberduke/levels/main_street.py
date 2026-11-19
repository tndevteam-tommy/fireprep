"""Round the corner from the alley: the near side of the main street.

The alley opens onto a pavement running left and right along the fronts of the two
tenement blocks: brick faces with rows of windows, street lamps along the kerb, a guard
rail along it (no crossing the traffic lanes - the far side is a backdrop of shops and
towers), and at either end the edge of the part of the city that is built: a barricade
(striped boards, a red sign). People come and go by the METRO steps at the right-hand end and the
UNDERPASS at the left.

Six of the people of the street live here, each behind their own front door on the
pavement (``levels.flats``): the junk dealer, the tinkerer and the stroller to the left
of the alley mouth, the office worker, the courier and the sushi chef to the right. A
cyberpunk car is parked at the kerb.

Layout (x along the street, z across it; the alley comes in at x = -4.5 .. 4.5):

    z=52.5  ══ far pavement, shopfronts, towers (backdrop) ═══════════════════════════
    z=51    ── far kerb
    z=45..51   the road (not walkable): monorail beam over it, the car at the near kerb
    z=45    ── kerb, guard rail, street lamps
    z=42.5..45 the pavement                                  ...
    z=42.5  ══ the blocks' faces: [UNDERPASS] 35 33 31 ║alley║ 32 34 36 [METRO]
           x=-34                                       x=0                       x=32
"""

from __future__ import annotations

from dataclasses import replace

from cyberduke.core.geometry import AABB
from cyberduke.levels import bar, flats, metro
from cyberduke.levels.flats import FlatPlan
from cyberduke.world import vehicles
from cyberduke.world.level import DoorData, LaneData, OpenSky, Place, PrivateArea, WalkRegion
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop

FACE_Z = 42.5  # the street faces of the blocks (the alley's walls end here)
KERB_Z = 45.0  # the near kerb: the pavement is z 42.5 .. 45
ROAD_Z1 = 53.0  # the far kerb
ROAD_Y = -0.12  # the road is a kerb's height below the pavement
WEST_END, EAST_END = -34.0, 32.0  # the barricades
# The road: a parking strip along the near kerb, a lane each way either side of a narrow
# median the monorail's pillars stand on; traffic keeps to the right (``entities.traffic``)
PARKING_Z = 47.45  # the parking strip: the near kerb .. here (a white line)
MEDIAN = (50.0, 50.5)  # the median strip
MONORAIL_Z = (MEDIAN[0] + MEDIAN[1]) / 2.0  # the monorail beam, over the median
MONORAIL_Y = 7.2  # ... the middle of the beam (its top 0.25 above)
LANE_WEST = (PARKING_Z + MEDIAN[0]) / 2.0  # the westbound lane (heading -x): the near one
LANE_EAST = (MEDIAN[1] + ROAD_Z1) / 2.0  # the eastbound lane (heading +x)
# At either end the built city stops: the blocks run into a concrete portal wall and the
# road and the monorail dive into tunnels (the traffic comes and goes through them)
TUNNEL_X = 44.0  # the portal walls' faces (x = -+44)
TUNNEL_DEPTH = 70.0  # how far the tunnels run in (out past the far plane)
_FAR_IN = TUNNEL_X + TUNNEL_DEPTH - 4.0  # where the traffic comes and goes, in the tunnels
LANES = (
    LaneData("westbound", LANE_WEST, _FAR_IN, -_FAR_IN, ROAD_Y),
    LaneData("eastbound", LANE_EAST, -_FAR_IN, _FAR_IN, ROAD_Y),
)
ROAD_OPEN = (KERB_Z, ROAD_Z1 + 0.3, 4.8)  # the road tunnel's mouth: z from, to; height
RAIL_OPEN = (48.7, 51.8, 6.4, 10.8)  # the monorail's: z from, to; y from, to
BLOCK_THICK = 1.2
WALL_H = 12.0
STOREY = flats.STOREY

BRICK_L = Material("brick", (0.12, 0.12, 0.13))
BRICK_R = Material("brick", (0.09, 0.11, 0.14))
CONCRETE = Material("concrete", (0.1, 0.1, 0.115))
PAVING = Material("concrete", (0.22, 0.22, 0.24), specular=(0.3, 0.3, 0.35), shininess=30.0)
ASPHALT = Material("asphalt", (0.27, 0.29, 0.35), specular=(0.5, 0.55, 0.65), shininess=40.0)
STEEL = Material("metal", (0.3, 0.31, 0.34))
DARK_STEEL = Material("metal", (0.12, 0.12, 0.14))
RAIL = Material("metal", (0.45, 0.47, 0.5), specular=(0.6, 0.6, 0.65), shininess=40.0)
HAZARD = Material("hazard", (0.95, 0.8, 0.2))
WINDOW_DARK = Material(None, (0.08, 0.09, 0.12))
WINDOW_WARM = Material(None, (0.9, 0.7, 0.4), emission=(0.55, 0.4, 0.2))
WINDOW_COOL = Material(None, (0.5, 0.75, 1.0), emission=(0.25, 0.4, 0.6))
LAMP_HEAD = Material(None, (1.0, 0.9, 0.7), emission=(1.8, 1.5, 1.0))
NEON_CYAN = Material("neon_cyan", (0.2, 0.9, 1.0), emission=(0.5, 2.2, 2.6))
NEON_MAGENTA = Material("neon_magenta", (1.0, 0.3, 0.8), emission=(2.4, 0.5, 1.8))
NEON_AMBER = Material("hazard", (1.0, 0.75, 0.2), emission=(2.0, 1.3, 0.3))
DECAL = frozenset({"transparent"})

# ----------------------------------------------------------------------------- the homes
# Nix's flat: across the alley from where he smokes, behind its west wall (the door faces
# the alley, +x). The others: on the main street, doors facing the pavement (+z).
# (every flat done up its own way: ``flats.STYLES``)
FLAT17 = FlatPlan("17", "smoker", "+x", -4.5, -1.4, 4.6, 1.6, style=1)
STREET_FLATS = (
    FlatPlan("31", "junk_dealer", "+z", FACE_Z, -16.6, -11.4, -12.7, style=2),
    FlatPlan("33", "tinkerer", "+z", FACE_Z, -23.4, -18.2, -19.5, style=3),
    FlatPlan("35", "stroller", "+z", FACE_Z, -30.2, -25.0, -26.3, style=0),
    FlatPlan("32", "office_worker", "+z", FACE_Z, 7.6, 12.8, 11.5, style=4),
    FlatPlan("34", "courier", "+z", FACE_Z, 14.4, 19.6, 18.3, style=5),
    FlatPlan("36", "sushi_chef", "+z", FACE_Z, 21.2, 26.4, 25.1, style=6),
)
ALL_FLATS = (FLAT17, *STREET_FLATS)
HOMES = {plan.owner_role: plan for plan in ALL_FLATS}

METRO = metro.DOORWAY  # the doorway into the METRO station's hall (``levels.metro``)
UNDERPASS = (-33.4, -31.2)
BAR = bar.DOOR_X  # the Neon Moth's door, just round the corner from the alley (``levels.bar``)


def flat_parts():
    """(props, colliders, doors, walk regions, places, routes, private areas, lights,
    access panels) of every generated flat."""
    props: list[Prop] = []
    colliders: list[AABB] = []
    doors: list[DoorData] = []
    regions: list[WalkRegion] = []
    places: list[Place] = []
    routes: list[tuple[str, str]] = []
    areas: list[PrivateArea] = []
    lights: list[Light] = []
    panels: list = []
    for plan in ALL_FLATS:
        p, c = flats.build(plan)
        props += p
        colliders += c
        doors.append(flats.door(plan))
        regions += flats.walk_regions(plan)
        pl, ro = flats.places(plan)
        places += pl
        routes += ro
        areas.append(flats.area(plan))
        lights.append(flats.light(plan))
        panels.append(flats.panel(plan))
    return props, colliders, doors, regions, places, routes, areas, lights, panels


# ----------------------------------------------------------------------------- the blocks
def _gaps() -> dict[int, list[tuple[float, float]]]:
    """Where each block's ground storey is not solid wall: the flats (their own front
    walls are the storey there) and the two doorways at the ends."""
    out: dict[int, list[tuple[float, float]]] = {-1: [UNDERPASS, (bar.XW, bar.XE)], 1: [metro.HALL_X]}
    for plan in STREET_FLATS:
        side = -1 if plan.u0 < 0 else 1
        out[side].append((plan.u0 - flats.SIDE, plan.u1 + flats.SIDE))
    return out


def _block_faces() -> tuple[list[Prop], list[AABB]]:
    """The two blocks' faces along the street: upper storeys whole, the ground storey a
    wall between the flats, and each block's end (the alley's own walls end at the corner:
    x = +-5.7, z = 42.5)."""
    props: list[Prop] = []
    colliders: list[AABB] = []
    z0, z1 = FACE_Z - BLOCK_THICK, FACE_Z
    zc = (z0 + z1) / 2.0
    for side, (a, c), brick in ((-1, (-TUNNEL_X, -5.7), BRICK_L), (1, (5.7, TUNNEL_X), BRICK_R)):
        label = f"Tenement block face ({'west' if side < 0 else 'east'})"
        # the upper storeys: whole - but for the metro station's glass front (its hall
        # goes up most of the building: only the top storey is brick over it)
        uppers = [(a, c, STOREY)] if side < 0 else [(a, metro.HALL_X[0], STOREY), (metro.HALL_X[0], metro.HALL_X[1], metro.ROOF_Y + 0.2), (metro.HALL_X[1], c, STOREY)]
        for u0, u1, bottom in uppers:
            props.append(Prop.box((u0 + u1) / 2.0, (bottom + WALL_H) / 2.0, zc, u1 - u0, WALL_H - bottom, BLOCK_THICK, brick, solid=True, label=label))
            colliders.append(AABB(u0, u1, z0, z1, bottom, WALL_H))
        cuts = sorted(_gaps()[side])
        x = a
        for g0, g1 in (*cuts, (c, c)):
            if g0 - x > 0.02:
                props.append(Prop.box((x + g0) / 2.0, STOREY / 2.0, zc, g0 - x, STOREY, BLOCK_THICK, brick, solid=True, label=label))
                colliders.append(AABB(x, g0, z0, z1, 0.0, STOREY))
            x = max(x, g1)
        # a skirting along the foot of the ground storey (stopping at every door)
        x = a
        for g0, g1 in (*cuts, (c, c)):
            if g0 - x > 0.05:
                props.append(Prop.box((x + g0) / 2.0, 0.45, FACE_Z + 0.06, g0 - x, 0.9, 0.12, CONCRETE, solid=True, label="Block skirting"))
                colliders.append(AABB(x, g0, FACE_Z, FACE_Z + 0.12, 0.0, 0.9))
            x = max(x, g1)
    # the doorways at the ends: dark stairwells, a lit sign over each
    for (g0, g1), name, mat in (
        (UNDERPASS, "UNDERPASS", NEON_AMBER),
    ):
        gc = (g0 + g1) / 2.0
        props.append(Prop.box(gc, STOREY / 2.0, z0 + 0.3, g1 - g0, STOREY, 0.6, Material(None, (0.02, 0.02, 0.03)), solid=True, label=f"{name} stairwell (dark)"))
        props.append(Prop.box(gc, 2.6, FACE_Z + 0.065, g1 - g0 + 0.3, 0.3, 0.12, mat, label=f"{name} sign", tags=DECAL))
        for e in (g0, g1):
            props.append(Prop.box(e, 1.3, FACE_Z + 0.05, 0.1, 2.6, 0.1, DARK_STEEL, label=f"{name} doorway post"))
        colliders.append(AABB(g0, g1, z0, FACE_Z - 0.3, 0.0, STOREY))
    return props, colliders


def _windows() -> list[Prop]:
    """Windows up the blocks' faces (the flats' own, on the ground floor, are theirs)."""
    import random

    rng = random.Random(4242)
    props: list[Prop] = []
    for a, c in ((-TUNNEL_X + 0.6, -6.2), (6.2, TUNNEL_X - 0.6)):
        for floor in range(3):
            y = 4.4 + floor * 2.6
            x = a + 0.9 + (floor % 2) * 0.6
            while x < c - 0.6:
                roll = rng.random()
                if metro.HALL_X[0] - 0.6 < x < metro.HALL_X[1] + 0.6:  # (the station's glass)
                    x += 2.3 + rng.random() * 0.6
                    continue
                mat = WINDOW_DARK if roll < 0.45 else (WINDOW_WARM if roll < 0.8 else WINDOW_COOL)
                props.append(Prop.box(x, y, FACE_Z + 0.025, 0.8, 1.1, 0.05, mat, label="Block window"))
                props.append(Prop.box(x, y - 0.6, FACE_Z + 0.08, 0.95, 0.05, 0.16, CONCRETE, label="Window sill"))
                x += 2.3 + rng.random() * 0.6
    # (each flat's own window, lit, is part of the flat: ``flats.build``)
    return props


# ----------------------------------------------------------------------------- the street
def _pavement_and_road() -> tuple[list[Prop], list[AABB]]:
    props: list[Prop] = []
    colliders: list[AABB] = []
    span0, span1 = -TUNNEL_X, TUNNEL_X  # (the pavement runs up to the portal walls)
    mid, width = (span0 + span1) / 2.0, span1 - span0
    reach = TUNNEL_X + TUNNEL_DEPTH  # (the road runs on into the tunnels)
    white = Material(None, (0.75, 0.75, 0.72))
    yellow = Material(None, (0.7, 0.65, 0.4))
    props += [
        # the pavement: from the blocks' faces to the kerb, its top level with the alley
        Prop.box(mid, -0.06, (FACE_Z + KERB_Z) / 2.0, width, 0.12, KERB_Z - FACE_Z, PAVING, label="Pavement"),
        Prop.box(mid, -0.07, KERB_Z + 0.075, width, 0.14, 0.15, CONCRETE, label="Kerb"),
        Prop.box(0.0, ROAD_Y - 0.05, (KERB_Z + 0.15 + ROAD_Z1) / 2.0, 2.0 * reach, 0.1, ROAD_Z1 - KERB_Z - 0.15, ASPHALT, label="Main street asphalt"),
        Prop.box(0.0, -0.06, ROAD_Z1 + 0.15, 2.0 * reach, 0.12, 0.3, CONCRETE, label="Far kerb"),
        # the median: a kerbed strip down the middle (the monorail's pillars stand on it)
        Prop.box(0.0, ROAD_Y + 0.06, MONORAIL_Z, 2.0 * reach, 0.12, MEDIAN[1] - MEDIAN[0], CONCRETE, label="Median strip"),
        Prop.box(0.0, ROAD_Y + 0.1205, MONORAIL_Z, 2.0 * reach - 0.02, 0.001, 0.08, yellow, label="Median line", tags=DECAL),
    ]
    # the lane markings: the parking strip's white line, dashes along the lanes' edges
    x = -reach + 1.0
    while x < reach - 1.0:
        props.append(Prop.box(x, ROAD_Y + 0.001, PARKING_Z, 2.0, 0.002, 0.1, white, label="Parking strip line", tags=DECAL))
        for z in (MEDIAN[0] - 0.2, MEDIAN[1] + 0.2):
            props.append(Prop.box(x + 1.5, ROAD_Y + 0.001, z, 3.0, 0.002, 0.1, yellow, label="Lane edge line", tags=DECAL))
        x += 6.0
    # the guard rail along the kerb: posts and two rails (nobody crosses the lanes here)
    rail_z = KERB_Z - 0.12
    x = WEST_END + 0.2
    while x <= EAST_END - 0.2:
        props.append(Prop.box(x, 0.5, rail_z, 0.08, 1.0, 0.08, RAIL, label="Guard rail post"))
        x += 2.0
    for y in (0.55, 0.95):  # (ending short of the barricades' boards)
        props.append(Prop.box((WEST_END + EAST_END) / 2.0, y, rail_z, EAST_END - WEST_END - 0.1, 0.07, 0.04, RAIL, label="Guard rail"))
    colliders.append(AABB(WEST_END, EAST_END, rail_z - 0.1, KERB_Z + 0.3, 0.0, 1.0))
    # the barricades at the ends of the built city: striped boards on trestles, a lamp
    for x0, facing in ((WEST_END, 1.0), (EAST_END, -1.0)):
        for z in (43.1, 44.3):
            props.append(Prop.box(x0, 0.75, z, 0.08, 0.35, 1.2, HAZARD, label="Barricade board"))
            props.append(Prop.box(x0, 0.35, z, 0.07, 0.7, 0.08, DARK_STEEL, label="Barricade trestle"))
        props.append(Prop.box(x0, 1.25, 43.7, 0.06, 0.3, 1.6, Material(None, (1.0, 0.2, 0.2), emission=(1.6, 0.2, 0.2)), label="ROAD CLOSED sign", tags=DECAL))
        props.append(Prop.box(x0 + facing * 0.05, 1.55, 42.8, 0.12, 0.12, 0.12, Material(None, (1.0, 0.5, 0.1), emission=(1.8, 0.7, 0.1)), label="Barricade lamp"))
        colliders.append(AABB(x0 - 0.3, x0 + 0.3, FACE_Z, KERB_Z, 0.0, 1.6))
    # street lamps along the kerb
    for x in LAMP_XS:
        props.append(Prop.cylinder(x, 0.0, KERB_Z - 0.35, 0.07, 0.06, 4.8, DARK_STEEL, slices=8, label="Street lamp pole"))
        props.append(Prop.box(x, 4.75, KERB_Z + 0.1, 0.12, 0.08, 0.9, DARK_STEEL, label="Street lamp arm"))
        props.append(Prop.box(x, 4.65, KERB_Z + 0.45, 0.3, 0.1, 0.5, LAMP_HEAD, label="Street lamp head", tags=DECAL))
        colliders.append(AABB(x - 0.12, x + 0.12, KERB_Z - 0.47, KERB_Z - 0.23, 0.0, 4.8))
    return props, colliders


LAMP_XS = (-27.0, -17.0, -7.5, 7.5, 17.0, 27.0)

# The weather (``systems.weather``): the street is open to the sky between the portal
# walls - the pavement, the road either side of the median (puddles gather on them), the
# median and the far pavement (none on those)
OPEN_SKY = (
    OpenSky(-TUNNEL_X, TUNNEL_X, FACE_Z, KERB_Z, 0.0),
    OpenSky(-TUNNEL_X, TUNNEL_X, KERB_Z + 0.15, MEDIAN[0], ROAD_Y),
    OpenSky(-TUNNEL_X, TUNNEL_X, MEDIAN[0], MEDIAN[1], ROAD_Y + 0.12, puddles=False),
    OpenSky(-TUNNEL_X, TUNNEL_X, MEDIAN[1], ROAD_Z1, ROAD_Y),
    OpenSky(-TUNNEL_X, TUNNEL_X, ROAD_Z1, ROAD_Z1 + 3.0, 0.0, puddles=False),
)


def street_lights() -> tuple[Light, ...]:
    return tuple(
        Light(
            x, 4.4, KERB_Z + 0.3,
            ambient=(0.11, 0.1, 0.08), diffuse=(2.2, 1.9, 1.35), specular=(1.5, 1.3, 1.0),
            linear_attenuation=0.1, quadratic_attenuation=0.02, fixture="none", night_only=True,
        )  # fmt: skip
        for x in LAMP_XS
    )


# ----------------------------------------------------------------------------- the monorail
PILLAR_XS = (-39.0, -24.0, -9.0, 9.0, 24.0, 39.0)
PORTAL = Material("concrete", (0.13, 0.13, 0.145))
TUNNEL_WALL = Material("concrete", (0.08, 0.08, 0.09))
TUNNEL_LAMP = Material(None, (1.0, 0.7, 0.3), emission=(1.8, 1.1, 0.4))
TUNNEL_STRIP = Material(None, (0.3, 0.8, 1.0), emission=(0.3, 1.2, 1.6))


def _monorail() -> list[Prop]:
    """The monorail's beam, along the street over the median and on into the tunnels at
    either end, a strip of light along each side, and the pillars it stands on: a
    concrete column on the median, a steel cap under the beam."""
    reach = TUNNEL_X + TUNNEL_DEPTH
    y, z = MONORAIL_Y, MONORAIL_Z
    props = [
        Prop.box(0.0, y, z, 2.0 * reach, 0.5, 1.2, DARK_STEEL, label="Monorail beam over the street"),
    ]
    for side in (-1.0, 1.0):
        props.append(Prop.box(0.0, y + 0.08, z + side * 0.605, 2.0 * reach - 0.02, 0.05, 0.01, NEON_CYAN, label="Monorail guide light", tags=DECAL))
    for x in PILLAR_XS:
        props.append(Prop.box(x, 3.4, z, 0.45, 6.8, 0.45, CONCRETE, label="Monorail pillar"))
        props.append(Prop.box(x, y - 0.375, z, 0.7, 0.25, 1.3, DARK_STEEL, label="Monorail pillar cap"))
        props.append(Prop.box(x, 0.3, z, 0.5, 0.6, 0.5, HAZARD, label="Monorail pillar hazard band"))
    return props


# ----------------------------------------------------------------------------- the tunnels
def _tunnels() -> tuple[list[Prop], list[AABB]]:
    """Where the built city stops: a concrete portal wall across the end of the street
    at either end, the blocks running into it, and two mouths in it - the road's, lit
    amber inside, and the monorail's above it - tunnels running on out of sight (past
    the far plane: the traffic comes out of the distance and goes off into it)."""
    props: list[Prop] = []
    colliders: list[AABB] = []
    z_back, z_front = FACE_Z - 4.0, ROAD_Z1 + 3.0  # (behind the blocks .. behind the far shops)
    rz0, rz1, rh = ROAD_OPEN
    qz0, qz1, qy0, qy1 = RAIL_OPEN
    top = 18.0
    for s in (-1.0, 1.0):
        face = s * TUNNEL_X
        xc = face + s * 0.5  # the wall: 1 m thick

        def wall(z0, z1, y0, y1, label="Tunnel portal wall", xc=xc):
            props.append(Prop.box(xc, (y0 + y1) / 2.0, (z0 + z1) / 2.0, 1.0, y1 - y0, z1 - z0, PORTAL, solid=True, label=label))
            colliders.append(AABB(xc - 0.5, xc + 0.5, z0, z1, y0, y1))

        wall(z_back, rz0, 0.0, top)
        wall(rz1, z_front, 0.0, top)
        wall(rz0, qz0, rh, top)
        wall(qz1, rz1, rh, top)
        wall(qz0, qz1, rh, qy0)
        wall(qz0, qz1, qy1, top)
        # hazard trim round the road's mouth, a sign over it (a hair proud of the wall)
        proud = face - s * 0.01
        for z in (rz0 + 0.15, rz1 - 0.15):
            props.append(Prop.box(proud, rh / 2.0, z, 0.02, rh, 0.3, HAZARD, label="Tunnel mouth hazard trim", tags=DECAL))
        props.append(Prop.box(proud, rh + 0.15, (rz0 + rz1) / 2.0, 0.02, 0.3, rz1 - rz0 - 0.6, HAZARD, label="Tunnel mouth hazard trim", tags=DECAL))
        props.append(Prop.box(proud, rh + 0.85, (rz0 + rz1) / 2.0 + 1.5, 0.02, 0.6, 3.2, NEON_AMBER, label="EXPRESSWAY sign", tags=DECAL))
        props.append(Prop.box(proud, qy1 + 0.4, (qz0 + qz1) / 2.0, 0.02, 0.35, qz1 - qz0, NEON_CYAN, label="MONORAIL sign", tags=DECAL))
        # the tunnels themselves: walls, a roof, lamps down the middle of each lane
        x0 = face + s * 1.0
        length = TUNNEL_DEPTH
        xm = x0 + s * length / 2.0
        far_x = x0 + s * length
        for z0, z1 in ((rz0 - 0.4, rz0), (rz1, rz1 + 0.4)):
            props.append(Prop.box(xm, (ROAD_Y + rh) / 2.0, (z0 + z1) / 2.0, length, rh - ROAD_Y, 0.4, TUNNEL_WALL, solid=True, label="Road tunnel wall"))
            colliders.append(AABB(min(x0, far_x), max(x0, far_x), z0, z1, ROAD_Y, rh))
        props.append(Prop.box(xm, rh + 0.2, (rz0 + rz1) / 2.0, length, 0.4, rz1 - rz0 + 0.8, TUNNEL_WALL, label="Road tunnel roof"))
        for z0, z1 in ((qz0 - 0.4, qz0), (qz1, qz1 + 0.4)):
            props.append(Prop.box(xm, (rh + 0.4 + qy1) / 2.0, (z0 + z1) / 2.0, length, qy1 - rh - 0.4, 0.4, TUNNEL_WALL, label="Monorail tunnel wall"))
        props.append(Prop.box(xm, qy1 + 0.2, (qz0 + qz1) / 2.0, length, 0.4, qz1 - qz0 + 0.8, TUNNEL_WALL, label="Monorail tunnel roof"))
        for z in (rz0 + 0.012, rz1 - 0.012):  # a strip of light along each wall
            props.append(Prop.box(xm, 1.1, z, length, 0.06, 0.024, TUNNEL_STRIP, label="Road tunnel wall light strip", tags=DECAL))
        k = 4.0
        while k < length - 1.0:
            for z in (LANE_WEST, LANE_EAST):
                props.append(Prop.box(x0 + s * k, rh - 0.02, z, 1.2, 0.04, 0.25, TUNNEL_LAMP, label="Road tunnel lamp", tags=DECAL))
            k += 7.0
        far = x0 + s * length
        props.append(Prop.box(far + s * 0.5, 6.0, (rz0 + qz1) / 2.0 + 0.5, 1.0, 12.5, rz1 - rz0 + 2.0, Material(None, (0.0, 0.0, 0.0)), solid=True, label="Tunnel far end (dark)"))
        colliders.append(AABB(min(far, far + s * 1.0), max(far, far + s * 1.0), (rz0 + qz1) / 2.0 + 0.5 - (rz1 - rz0 + 2.0) / 2.0, (rz0 + qz1) / 2.0 + 0.5 + (rz1 - rz0 + 2.0) / 2.0, -0.25, 12.25))
    return props, colliders


# ----------------------------------------------------------------------------- the car
CAR_AT = (12.5, 46.35)  # at the near kerb, nose towards -x
CAR_PAINT = vehicles.Paint((0.42, 0.55, 0.72), (1.0, 0.3, 0.8))  # gunmetal blue, magenta neon
UNDERGLOW = Material(None, (1.0, 0.25, 0.8), alpha=0.38)


def car() -> tuple[list[Prop], list[AABB]]:
    """A low, wedge-nosed electric coupe parked at the kerb (``world.vehicles``: the same
    coupe as the ones driving past): gunmetal-blue lacquer, a magenta neon sill and a
    magenta underglow on the wet road. Set down at ``CAR_AT``, inside one collider."""
    cx, cz = CAR_AT
    props = [
        replace(p, x=p.x + cx, y=p.y + ROAD_Y, z=p.z + cz)
        for p in vehicles.MODELS["coupe"].parts(CAR_PAINT)
    ]
    props.append(Prop.disc(cx, ROAD_Y + 0.004, cz, 2.5, 1.25, 0.45, UNDERGLOW, slices=20, label="Car underglow on the road", tags=frozenset({"glow", "transparent"})))
    return props, [AABB(cx - 2.6, cx + 2.5, cz - 1.05, cz + 1.05, ROAD_Y, ROAD_Y + 1.2)]


# ----------------------------------------------------------------------------- assembly
def build() -> tuple[list[Prop], list[AABB]]:
    props: list[Prop] = []
    colliders: list[AABB] = []
    for part in (_block_faces, _pavement_and_road, car):
        p, c = part()
        props += p
        colliders += c
    props += _windows()
    props += _monorail()
    p, c = _tunnels()
    props += p
    colliders += c
    return props, colliders


PAVEMENT = WalkRegion(
    WEST_END + 0.35, EAST_END - 0.35, FACE_Z, KERB_Z - 0.3,
    inset=frozenset({"min_x", "max_x", "max_z"}),
)  # fmt: skip

PLACES = (
    # off round the corner and away: down the METRO steps (right), the UNDERPASS (left)
    Place("metro", (METRO[0] + METRO[1]) / 2.0, FACE_Z + 0.35, yaw=180.0),
    Place("underpass", (UNDERPASS[0] + UNDERPASS[1]) / 2.0, FACE_Z + 0.35, yaw=180.0),
    # a while out of sight - in the bar round the corner to the left, or off along the
    # street to the right (``ai.alley.leave_and_return``)
    Place("mouth", 7.0, FACE_Z + 1.0, yaw=90.0),
    Place("mouth.left", (BAR[0] + BAR[1]) / 2.0, FACE_Z + 0.35, yaw=180.0),
)

