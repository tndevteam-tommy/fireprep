"""The metro station at the east end of the main street: Neon District, line 3.

The trains run along the monorail over the street (``levels.main_street``: its beam on
pillars down the median, tunnels at either end) and stop at an elevated platform over the
near side of the road. The way up is the station hall behind the METRO doorway in the
east block: a glass front the height of the building (you can watch people go up and
come down), ticket gates across it just inside the doorway - no pass, no entry - a long
straight stair up the back wall to a landing at the top, and a covered footbridge from
the landing out over the pavement to the platform: a deck on columns standing in the
parking strip, a canopy over it, benches, the station's name, a departures board.

People come and go this way (``ai.alley``: the bartender commutes, a regular comes for
a drink), through the gates, up and down the stair, across the bridge, waiting on the
platform and getting on and off the trains (``entities.metro``) - nobody appears or
disappears: they walk into a train and it takes them away down the tunnel.

Layout (x along the street, z across it; y up):

    z=48.8  ─ platform edge ─ (the train: its floor 7.6 up, over the monorail beam) ─
    z=45.8  ═ platform (x 26 .. 43.6, 7.6 up) ══════╗ footbridge (x 40.2 .. 41.4)
    z=42.5  ══ glass front ══[doorway]═══════════════╝══  (the pavement below)
    z=41.35   gates ─────────────────────────────────────   (the paid side beyond)
    z=38.5..40  stair: foot at x 31, top (7.6 up) at x 39.4  ─ landing x 39.4 .. 41.8
           x=27.2                                  x=42
"""

from __future__ import annotations

from cyberduke.core.geometry import AABB
from cyberduke.levels import furnish as f
from cyberduke.levels.furnish import Builder, Wall, WorldFrame
from cyberduke.world.level import DoorData, MetroData, Place, WalkRegion
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop
from cyberduke.world.surfaces import Flight

DECAL = frozenset({"transparent"})

FACE = 42.5  # the block's street face
HALL_X = (27.2, 42.0)  # the hall, its end walls included
X0, X1 = 27.4, 41.8  # its inside
Z0 = 38.4  # the inside face of its back wall
GLASS_Z = (42.38, 42.5)  # the glass front (mullions); the panes in the middle of it
ROOF_Y = 10.4  # the hall's ceiling
DOORWAY = (28.6, 30.8)  # (``levels.main_street.METRO``)
DOOR_TOP = 2.6
GATES_Z = (41.2, 41.5)  # the gate line across the hall
LANES = (29.15, 30.25)  # the two gates' middles
GATE_UNITS = (28.55, 29.7, 30.8)  # the gate cabinets between and either side of the lanes

FLOOR_Y = 7.6  # the upper landing, the footbridge, the platform: all level with the train
STEPS = 40
RISE = FLOOR_Y / STEPS  # 0.19
RUN = 0.21
STAIR_X0 = 31.0  # the foot of the stair
STAIR_X1 = STAIR_X0 + STEPS * RUN  # the top (39.4)
STAIR_Z = (38.5, 40.0)
LANDING_X = (STAIR_X1, X1)
BAYS = (X0, DOORWAY[0], DOORWAY[1], *(DOORWAY[1] + k * (X1 - DOORWAY[1]) / 7.0 for k in range(1, 8)))
BRIDGE_X = (BAYS[-2] + 0.12, X1 - 0.12)  # through the last bay of the glass front
BRIDGE_Z = (FACE - 0.2, 45.8)
PLATFORM_X = (26.0, 43.6)
PLATFORM_Z = (45.8, 48.8)  # the back .. the edge (the train's side is 5 cm beyond it)
CANOPY_Y = 10.6
BRIDGE_HEADROOM = 2.3  # the way out of the glass front onto the footbridge: this high

# the train (``entities.metro``): along the beam, its floor level with the platform
TRACK_Z = 50.25  # (``levels.main_street.MONORAIL_Z``)
STOP_X = (PLATFORM_X[0] + PLATFORM_X[1]) / 2.0

TILE = Material("tile_subway", (0.75, 0.8, 0.85))
FLOOR = Material("tile_hex", (0.5, 0.52, 0.56), specular=(0.4, 0.4, 0.45), shininess=40.0)
CONCRETE = Material("concrete", (0.3, 0.3, 0.32))
DECK = Material("grate", (0.42, 0.43, 0.46), specular=(0.5, 0.5, 0.55), shininess=40.0)
STEEL = Material("metal", (0.22, 0.23, 0.26), specular=(0.6, 0.6, 0.65), shininess=50.0)
CHROME = f.CHROME
GLASS = Material(None, (0.45, 0.6, 0.7), specular=(1.0, 1.0, 1.0), shininess=120.0, alpha=0.18)
STEP = Material("metal", (0.34, 0.35, 0.38), specular=(0.5, 0.5, 0.55), shininess=40.0)
NOSING = Material(None, (0.95, 0.78, 0.15), emission=(0.5, 0.4, 0.05))
NEON_CYAN = Material(None, (0.2, 0.9, 1.0), emission=(0.5, 2.2, 2.6))
NEON_MAGENTA = Material(None, (1.0, 0.3, 0.8), emission=(2.4, 0.5, 1.8))
LAMP = Material(None, (0.95, 1.0, 1.0), emission=(1.9, 2.1, 2.2))
EDGE = Material("hazard", (0.95, 0.8, 0.2))
RED_GLOW = Material(None, (1.0, 0.15, 0.15), emission=(2.2, 0.2, 0.2))
GREEN_GLOW = Material(None, (0.2, 1.0, 0.35), emission=(0.3, 2.0, 0.5))


def _sign(tex: str) -> Material:
    return Material(tex, (1.0, 1.0, 1.0), emission=(1.3, 1.3, 1.3))


# ----------------------------------------------------------------------------- the hall
def _hall(b: Builder) -> None:
    """The shell: the back wall (white subway tiles), the end walls, the floor, the
    ceiling, the glass front with the doorway in it and the footbridge's way out."""
    hx0, hx1 = HALL_X
    b.add(hx0, hx1, Z0 - 0.2, Z0, 0.0, ROOF_Y + 0.2, TILE, "Metro hall back wall", solid=True)
    for a, c in ((hx0, X0), (X1, hx1)):
        b.add(a, c, Z0, FACE, 0.0, ROOF_Y + 0.2, TILE, "Metro hall end wall", solid=True)
    b.add(X0, X1, Z0, GLASS_Z[0], -0.02, 0.002, FLOOR, "Metro hall floor")
    b.add(X0, X1, Z0, FACE - 0.005, ROOF_Y, ROOF_Y + 0.2, Material("concrete", (0.16, 0.16, 0.18)), "Metro hall ceiling")
    # the glass front: a mullion between every two bays, a transom every storey, a pane in
    # every frame - the doorway at the bottom (and the bridge's way out at the top) left open
    rows = [0.0, DOOR_TOP, 5.2, FLOOR_Y, ROOF_Y]
    z0, z1 = GLASS_Z
    (d0, d1) = DOORWAY
    bridge_bay = (BAYS[-2], BAYS[-1])
    for x in BAYS[1:-1]:  # (the end walls are the outermost)
        b.add(x - 0.04, x + 0.04, z0, z1, 0.0, ROOF_Y, STEEL, "Glass front mullion")
    for y in rows[1:-1]:
        spans = [(X0, X1)] if y != FLOOR_Y else [(X0, bridge_bay[0] - 0.04), (bridge_bay[1], X1)]
        for a, c in spans:
            if c - a > 0.01:
                b.add(a, c, z0 + 0.01, z1 - 0.01, y - 0.05, y + 0.05, STEEL, "Glass front transom")
    for xa, xc in zip(BAYS, BAYS[1:], strict=False):
        for ya, yc in zip(rows, rows[1:], strict=False):
            if ya == 0.0 and (xa, xc) == DOORWAY:
                continue  # (the doorway)
            if ya == FLOOR_Y and (xa, xc) == bridge_bay:
                continue  # (the bridge's way out)
            e0 = 0.005 if xa == X0 else 0.045
            e1 = 0.005 if xc == X1 else 0.045
            y_lo = ya + (0.055 if ya > 0.0 else 0.005)
            b.add(xa + e0, xc - e1, z0 + 0.05, z0 + 0.062, y_lo, yc - 0.055, GLASS, "Glass front pane", tags=DECAL)
    # (see through it, but not walk through it - but for the doorway, and the bridge's
    # way out onto the footbridge from the landing)
    b.colliders.append(AABB(X0, d0, z0, z1, 0.0, ROOF_Y, blocks_sight=False))
    b.colliders.append(AABB(d1, bridge_bay[0], z0, z1, 0.0, ROOF_Y, blocks_sight=False))
    b.colliders.append(AABB(bridge_bay[0], X1, z0, z1, 0.0, FLOOR_Y, blocks_sight=False))
    b.colliders.append(AABB(bridge_bay[0], X1, z0, z1, FLOOR_Y + BRIDGE_HEADROOM, ROOF_Y, blocks_sight=False))
    b.colliders.append(AABB(d0, d1, z0, z1, DOOR_TOP, ROOF_Y, blocks_sight=False))
    b.add(d0 + 0.045, d1 - 0.045, z0, FACE - 0.002, -0.05, 0.006, Material("concrete", (0.22, 0.22, 0.24)), "Metro threshold")
    # a lit sign over the doorway
    b.add(d0 - 0.2, d1 + 0.2, z1, z1 + 0.1, DOOR_TOP + 0.08, DOOR_TOP + 0.5, _sign("sign_metro"), "METRO sign", tags=DECAL)
    # lights: tubes along the ceiling, a strip along the top of the back wall
    for x in (30.0, 34.0, 38.0):
        b.add(x - 0.8, x + 0.8, 40.2, 40.35, ROOF_Y - 0.06, ROOF_Y, LAMP, "Metro hall light tube", tags=DECAL)
    Wall(b, "v", Z0, 1.0).box(X0 + 0.1, X1 - 0.1, 9.8, 9.86, 0.0, 0.03, NEON_CYAN, "Metro hall neon strip")


def _gates(b: Builder) -> None:
    """Across the hall just inside the doorway: three gate cabinets (a reader on each, the
    lane's light red), glass flaps across the two lanes between them, and glass barriers
    out to the walls either side - NO PASS NO ENTRY over them. The flaps open for whoever
    has a pass (``render.metro`` draws them, ``entities.metro`` opens them)."""
    g0, g1 = GATES_Z
    for x in GATE_UNITS:
        b.add(x - 0.1, x + 0.1, g0 - 0.15, g1 + 0.15, 0.0, 1.05, STEEL, "Ticket gate cabinet")
        b.add(x - 0.1, x + 0.1, g0 - 0.15, g1 + 0.15, 1.05, 1.08, Material(None, (0.1, 0.1, 0.12)), "Ticket gate top")
        b.add(x - 0.07, x + 0.07, g1 + 0.15, g1 + 0.16, 0.85, 0.98, RED_GLOW, "Ticket gate reader (red)", tags=DECAL)
    for a, c in ((X0, GATE_UNITS[0] - 0.1), (GATE_UNITS[-1] + 0.1, X1)):
        b.add(a + 0.005, c - 0.005, g0 + 0.12, g0 + 0.14, 0.05, 1.195, GLASS, "Gate line glass barrier", tags=DECAL)
        b.add(a, c, g0 + 0.11, g0 + 0.15, 1.2, 1.24, CHROME, "Gate line barrier rail")
    # the cabinets and the barriers are solid; the lanes between them are the gates
    # (``GATE_DOORS``: shut but for a pass - or a netrunner who forces one)
    for x in GATE_UNITS:
        b.colliders.append(AABB(x - 0.1, x + 0.1, g0 - 0.15, g1 + 0.15, 0.0, 1.08, blocks_sight=False))
    for a, c in ((X0, GATE_UNITS[0] - 0.1), (GATE_UNITS[-1] + 0.1, X1)):
        b.colliders.append(AABB(a, c, g0, g1, 0.0, 1.24, blocks_sight=False))
    b.add(28.3, 31.1, g1 + 0.2, g1 + 0.22, 2.1, 2.46, _sign("sign_no_pass"), "NO PASS NO ENTRY sign", tags=DECAL)
    for x in (28.4, 31.0):
        b.add(x - 0.01, x + 0.01, g1 + 0.2, g1 + 0.22, 2.46, ROOF_Y, STEEL, "Sign hanger")


def _stair(b: Builder) -> None:
    """The stair up the back wall: forty steps (a yellow nosing on each), its open side
    closed under them by a steel stringer, a glass balustrade on that side with a rail; the
    landing at the top - and its balustrade. Anybody can climb it (``FLIGHT``)."""
    sz0, sz1 = STAIR_Z
    for k in range(STEPS):
        x0 = STAIR_X0 + k * RUN
        y1 = (k + 1) * RISE
        b.add(x0, x0 + RUN, sz0, sz1, y1 - RISE, y1, STEP, "Metro stair step")
        b.add(x0, x0 + 0.03, sz0 + 0.02, sz1 - 0.02, y1, y1 + 0.004, NOSING, "Metro stair nosing", tags=DECAL)
        if k:  # (its open side closed under the treads: a steel stringer, step by step)
            b.add(x0, x0 + RUN, sz1 - 0.02, sz1, 0.0, y1 - RISE - 0.005, STEEL, "Metro stair stringer")
    # the landing at the top (over the lobby), out to the glass front
    lx0, lx1 = LANDING_X
    b.add(lx0, lx1, Z0, GLASS_Z[0], FLOOR_Y - 0.2, FLOOR_Y, CONCRETE, "Metro hall landing")
    b.add(lx0, lx1, Z0, GLASS_Z[0] - 0.01, FLOOR_Y, FLOOR_Y + 0.002, FLOOR, "Metro hall landing floor")
    # the balustrades: glass along the stair's open side (stepping up with it) and along
    # the landing's edge, a chrome rail along the top
    for k in range(0, STEPS, 4):
        x0 = STAIR_X0 + k * RUN
        y0 = (k + 1) * RISE
        b.add(x0 + 0.01, x0 + 4 * RUN - 0.01, sz1 + 0.01, sz1 + 0.03, y0 + 0.02, y0 + 0.995, GLASS, "Metro stair balustrade", tags=DECAL)
        b.add(x0, x0 + 4 * RUN, sz1 + 0.005, sz1 + 0.035, y0 + 1.0, y0 + 1.04, CHROME, "Metro stair handrail")
    b.add(lx0 + 0.01, lx0 + 0.03, sz1 + 0.045, GLASS_Z[0] - 0.055, FLOOR_Y + 0.02, FLOOR_Y + 0.995, GLASS, "Landing balustrade", tags=DECAL)
    b.add(lx0 - 0.005, lx0 + 0.035, sz1 + 0.04, GLASS_Z[0] - 0.05, FLOOR_Y + 1.0, FLOOR_Y + 1.04, CHROME, "Landing rail")
    # Anybody can climb it (``FLIGHT``): nothing stops them on its footprint - but its open
    # side is closed by the balustrade all the way down to the floor (no stepping off it,
    # no walking in under it), its top end by a wall under the landing's edge, and the
    # landing's edge by its own balustrade
    b.add(lx0, lx0 + 0.1, Z0, sz1 + 0.04, 0.0, FLOOR_Y - 0.2, TILE, "Wall under the top of the stair", solid=True)
    for k in range(0, STEPS, 4):  # (stepping up with the balustrade, a section at a time)
        x0 = STAIR_X0 + k * RUN
        b.collide(x0, x0 + 4 * RUN, sz1, sz1 + 0.04, 0.0, (k + 1) * RISE + 1.04)
    b.collide(lx0 - 0.005, lx0 + 0.035, sz1 + 0.04, GLASS_Z[0], FLOOR_Y, FLOOR_Y + 1.04)


# ----------------------------------------------------------------------------- outside
def _bridge(b: Builder) -> None:
    """The covered footbridge from the landing out over the pavement to the platform."""
    x0, x1 = BRIDGE_X
    z0, z1 = GLASS_Z[0], PLATFORM_Z[0]
    b.add(x0, x1, z0, z1, FLOOR_Y - 0.2, FLOOR_Y, STEEL, "Footbridge deck")
    b.add(x0 + 0.025, x1 - 0.025, z0, z1, FLOOR_Y, FLOOR_Y + 0.002, DECK, "Footbridge floor")
    for x in (x0, x1):
        b.add(x - 0.02, x + 0.02, z0, z1, FLOOR_Y, FLOOR_Y + 1.1, GLASS, "Footbridge glass side", tags=DECAL)
        b.add(x - 0.03, x + 0.03, z0, z1, FLOOR_Y + 2.3, FLOOR_Y + 2.36, STEEL, "Footbridge roof edge")
        for z in (z0 + 0.4, (z0 + z1) / 2.0, z1 - 0.4):
            b.add(x - 0.03, x + 0.03, z - 0.03, z + 0.03, FLOOR_Y + 0.002, FLOOR_Y + 2.3, STEEL, "Footbridge post", solid=True)
    b.add(x0 - 0.05, x1 + 0.05, z0, z1, FLOOR_Y + 2.36, FLOOR_Y + 2.42, Material("metal", (0.2, 0.22, 0.26)), "Footbridge roof")
    b.add(x0 + 0.4, x1 - 0.4, z0 + 0.1, z1 - 0.1, FLOOR_Y + 2.33, FLOOR_Y + 2.36, NEON_MAGENTA, "Footbridge light", tags=DECAL)


def _platform(b: Builder) -> None:
    """The platform: a concrete deck on columns standing in the parking strip, the
    yellow edge, a glass balustrade along the back, the canopy on its posts, benches, the
    station's name hanging under the canopy, the departures board, lamps."""
    x0, x1 = PLATFORM_X
    z0, z1 = PLATFORM_Z
    b.add(x0, x1, z0, z1, FLOOR_Y - 0.3, FLOOR_Y, CONCRETE, "Platform deck")
    b.add(x0, x1, z1 - 0.35, z1 - 0.002, FLOOR_Y, FLOOR_Y + 0.003, EDGE, "Platform edge (yellow)", tags=DECAL)
    b.add(x0, x1, z0 + 0.002, z1 - 0.35, FLOOR_Y, FLOOR_Y + 0.002, FLOOR, "Platform floor")
    for x in (28.0, 34.0, 40.0):  # the columns it stands on (in the parking strip)
        b.add(x - 0.25, x + 0.25, 46.35, 46.85, -0.12, FLOOR_Y - 0.3, CONCRETE, "Platform column")
        b.add(x - 0.27, x + 0.27, 46.33, 46.87, -0.118, 0.6, EDGE, "Platform column hazard band")
    # the balustrade along the back (open where the bridge comes in) and at the ends
    for a, c in ((x0, BRIDGE_X[0] - 0.05), (BRIDGE_X[1] + 0.05, x1)):
        b.add(a, c, z0 + 0.02, z0 + 0.04, FLOOR_Y + 0.02, FLOOR_Y + 1.045, GLASS, "Platform balustrade", tags=DECAL)
        b.add(a, c, z0 + 0.015, z0 + 0.045, FLOOR_Y + 1.05, FLOOR_Y + 1.09, CHROME, "Platform rail")
    for x in (x0 + 0.03, x1 - 0.03):
        b.add(x - 0.01, x + 0.01, z0 + 0.05, z1 - 0.4, FLOOR_Y + 0.02, FLOOR_Y + 1.05, GLASS, "Platform end balustrade", tags=DECAL)
    # the canopy: posts along the back, a roof out to over the edge, lamps under it
    for x in (27.0, 31.0, 35.0, 39.0, 42.6):
        b.add(x - 0.06, x + 0.06, z0 + 0.12, z0 + 0.24, FLOOR_Y, CANOPY_Y, STEEL, "Canopy post", solid=True)
    b.add(x0, x1, z0, z1 - 0.3, CANOPY_Y, CANOPY_Y + 0.15, Material("metal", (0.18, 0.2, 0.24)), "Platform canopy")
    b.add(x0, x1, z1 - 0.32, z1 - 0.28, CANOPY_Y - 0.08, CANOPY_Y, NEON_CYAN, "Canopy edge neon", tags=DECAL)
    for x in (29.0, 33.0, 37.0, 41.0):
        b.add(x - 0.7, x + 0.7, 47.0, 47.12, CANOPY_Y - 0.05, CANOPY_Y, LAMP, "Platform lamp", tags=DECAL)
    # the station's name, hanging under the canopy (both sides read), the departures board
    for x in (30.0, 38.0):
        b.add(x - 1.5, x + 1.5, 46.78, 46.82, 9.3, 9.8, _sign("sign_station"), "Station sign: NEON DISTRICT", tags=DECAL)
        for hx in (x - 1.3, x + 1.3):
            b.add(hx - 0.01, hx + 0.01, 46.79, 46.81, 9.8, CANOPY_Y, STEEL, "Sign hanger")
    b.add(34.4, 35.6, 46.5, 46.6, 9.0, 9.6, Material(None, (0.02, 0.03, 0.05)), "Departures board", solid=True)
    for z0_, z1_ in ((46.496, 46.5), (46.6, 46.604)):
        b.add(34.45, 35.55, z0_, z1_, 9.05, 9.55, Material("screen_net", (1.0, 1.0, 1.0), emission=(0.9, 1.0, 1.1)), "Departures board screen", tags=DECAL)
    for hx in (34.6, 35.4):
        b.add(hx - 0.01, hx + 0.01, 46.54, 46.56, 9.6, CANOPY_Y, STEEL, "Departures board hanger")
    # benches along the back
    for x in (29.5, 36.5):
        b.add(x - 0.9, x + 0.9, 46.0, 46.4, FLOOR_Y + 0.42, FLOOR_Y + 0.46, Material("wood", (0.5, 0.35, 0.2)), "Platform bench")
        b.collide(x - 0.9, x + 0.9, 46.0, 46.4, FLOOR_Y, FLOOR_Y + 0.46)
        for lx in (x - 0.75, x + 0.75):
            b.add(lx - 0.03, lx + 0.03, 46.05, 46.35, FLOOR_Y + 0.002, FLOOR_Y + 0.42, STEEL, "Platform bench leg")
    # a bin, a vending screen of ads at the end
    b.cylinder(43.0, 46.3, FLOOR_Y + 0.002, 0.18, 0.18, 0.7, STEEL, "Platform bin", slices=10)
    b.collide(42.82, 43.18, 46.12, 46.48, FLOOR_Y, FLOOR_Y + 0.7)
    b.add(26.2, 26.3, 46.2, 47.6, FLOOR_Y + 0.3, FLOOR_Y + 2.3, Material("tv_ads_2", (1.0, 1.0, 1.0), emission=(1.0, 1.0, 1.0)), "Platform advert screen", solid=True)


def build() -> tuple[list[Prop], list[AABB]]:
    b = Builder(WorldFrame())
    _hall(b)
    _gates(b)
    _stair(b)
    _bridge(b)
    _platform(b)
    return b.props, b.colliders


LIGHTS = (
    Light(
        34.0, 9.6, 40.5,
        ambient=(0.08, 0.09, 0.1), diffuse=(1.3, 1.45, 1.6), specular=(0.6, 0.65, 0.7),
        linear_attenuation=0.12, quadratic_attenuation=0.02, fixture="none",
    ),
    Light(
        35.0, 10.2, 47.2,
        ambient=(0.06, 0.07, 0.08), diffuse=(1.2, 1.35, 1.5), specular=(0.5, 0.55, 0.6),
        linear_attenuation=0.12, quadratic_attenuation=0.02, fixture="none",
    ),
)  # fmt: skip


# ----------------------------------------------------------------------------- for people
WAIT_SPOTS = ((29.0, 46.9), (32.2, 47.1), (35.6, 46.8), (38.4, 47.2), (42.2, 46.9))


def places() -> tuple[Place, ...]:
    """The way from the pavement to the platform (``ai.alley``: walked by script - the
    stair and the bridge are nobody's to route over but theirs)."""
    mid = (DOORWAY[0] + DOORWAY[1]) / 2.0
    sz = (STAIR_Z[0] + STAIR_Z[1]) / 2.0
    bm = (BRIDGE_X[0] + BRIDGE_X[1]) / 2.0
    out = [
        Place("metro.inside", mid, 41.95, yaw=180.0),  # in the doorway, at the gates
        Place("metro.paid", mid, GATES_Z[0] - 0.6, yaw=180.0),  # through them
        Place("metro.foot", STAIR_X0 - 0.05, sz, yaw=90.0),
        Place("metro.top", STAIR_X1 + 0.05, sz, yaw=90.0, y=FLOOR_Y),
        Place("metro.landing", bm, 40.9, yaw=0.0, y=FLOOR_Y),
        Place("metro.bridge", bm, 44.2, yaw=0.0, y=FLOOR_Y),
        Place("metro.platform", bm, 46.6, yaw=0.0, y=FLOOR_Y),
    ]
    for k, (x, z) in enumerate(WAIT_SPOTS, start=1):
        out.append(Place(f"metro.wait{k}", x, z, yaw=0.0, y=FLOOR_Y))
    for k, x in enumerate(LANES, start=1):
        out.append(Place(f"metro.gate{k}", x, GATES_Z[1] + 0.45, yaw=180.0))
        out.append(Place(f"metro.gate{k}.in", x, GATES_Z[0] - 0.45, yaw=180.0))
    return tuple(out)


# ----------------------------------------------------------------------------- walking it
# The ticket gates: a lane each, shut unless somebody shows a pass (``entities.metro``
# opens it for them) - or a netrunner forces it (a door lock on the METRO TRANSIT
# network: FORCE OPEN). Forced, a gate opens for whoever walks up to it, until the
# station's system sets it again a minute later.
GATE_FACTION = "metro_transit"
GATE_HALF = 0.45  # half a lane's width, cabinet to cabinet
GATE_DOORS = tuple(
    DoorData(
        f"metro_gate_{k}",
        AABB(x - GATE_HALF, x + GATE_HALF, GATES_Z[0], GATES_Z[1], 0.0, 1.24, blocks_sight=False),
        None, travel=1.0, speed=3.5, passable_at=0.7, faction=GATE_FACTION,
        approach_range=1.4, relock_after=60.0,
    )  # fmt: skip
    for k, x in enumerate(LANES, start=1)
)
# Where the netrunner can walk (as well as the doorway up to the gates: ``levels.street``):
# through a gate that is open, the paid side of the hall (and under the landing), up the
# stair (``FLIGHT``), along the landing, over the footbridge and out onto the platform -
# the raised floors (``DECKS``) kept to by their balustrades and the edges of these
WALK_REGIONS = (
    *(
        WalkRegion(
            x - GATE_HALF, x + GATE_HALF, GATES_Z[0] - 0.4, GATES_Z[1] + 0.4,
            inset=frozenset({"min_x", "max_x"}), requires_door=d.door_id,
        )  # fmt: skip
        for x, d in zip(LANES, GATE_DOORS, strict=True)
    ),
    WalkRegion(X0, X1, Z0, GATES_Z[0], inset=frozenset({"min_x", "max_x", "min_z"})),
    WalkRegion(LANDING_X[0], X1, GATES_Z[0] - 0.3, GLASS_Z[0], inset=frozenset({"max_x", "max_z"})),
    WalkRegion(BRIDGE_X[0], BRIDGE_X[1], GLASS_Z[0] - 0.35, PLATFORM_Z[0] + 0.35, inset=frozenset({"min_x", "max_x"})),
    WalkRegion(PLATFORM_X[0], PLATFORM_X[1], PLATFORM_Z[0], PLATFORM_Z[1]),
)
FLIGHT = Flight(STAIR_X0, STAIR_X1, STAIR_Z[0], STAIR_Z[1], 0.0, FLOOR_Y, STEPS, "+x")
DECKS = (
    AABB(LANDING_X[0], X1, Z0, GLASS_Z[0], FLOOR_Y - 0.2, FLOOR_Y),
    AABB(BRIDGE_X[0], BRIDGE_X[1], GLASS_Z[0] - 0.02, PLATFORM_Z[0] + 0.02, FLOOR_Y - 0.2, FLOOR_Y),
    AABB(PLATFORM_X[0], PLATFORM_X[1], PLATFORM_Z[0], PLATFORM_Z[1], FLOOR_Y - 0.3, FLOOR_Y),
)
# ... and nobody else wanders in there: the people who go up (``ai.nightlife``) walk it by
# script, and the crowd's routes keep out of the paid side whether the gates are open or not
# (the platform's walkable floor is over the road: nothing the crowd should aim for)
NAV_BLOCKERS = (
    AABB(X0, X1, Z0, GATES_Z[1] + 0.05, 0.0, 3.0),
    AABB(PLATFORM_X[0], PLATFORM_X[1], PLATFORM_Z[0], PLATFORM_Z[1], 0.0, 3.0),
)


DATA = MetroData(
    track_z=TRACK_Z, floor_y=FLOOR_Y, stop_x=STOP_X, edge_z=PLATFORM_Z[1],
    x_west=-112.0, x_east=112.0, gates=LANES, gates_z=GATES_Z,
    stair_foot=(STAIR_X0, (STAIR_Z[0] + STAIR_Z[1]) / 2.0),
    stair_top=(STAIR_X1, (STAIR_Z[0] + STAIR_Z[1]) / 2.0), stair_steps=STEPS,
    gate_doors=tuple(d.door_id for d in GATE_DOORS),
)  # fmt: skip


__all__ = [
    "DATA", "DECKS", "FLIGHT", "GATE_DOORS", "GATE_FACTION", "LIGHTS", "NAV_BLOCKERS",
    "WALK_REGIONS", "build", "places",
]  # fmt: skip
