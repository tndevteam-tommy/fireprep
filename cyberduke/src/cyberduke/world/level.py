"""Level description: everything a ``Room``, a ``Simulation`` and the renderer need to build
and draw a playable facility. Pure data; see ``cyberduke.levels.facility`` for the shipped level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cyberduke.core.geometry import AABB
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop
from cyberduke.world.surfaces import Catwalk, Flight

Bounds = tuple[float, float, float, float]  # (min_x, max_x, min_z, max_z)
Edges = frozenset[str]
ALL_EDGES: Edges = frozenset({"min_x", "max_x", "min_z", "max_z"})
RGB = tuple[float, float, float]

DEFAULT_FACTION = "security"
"""Faction id used by entities that do not name one (every level has at least this one)."""


@dataclass(frozen=True)
class FactionData:
    """A group whose devices and people share one local network. Breaching any computer on
    the network (jacking in, or a direct quickhack) exposes every member to the netrunner."""

    faction_id: str
    name: str
    color: RGB = (1.0, 0.55, 0.1)  # HUD marker colour for devices reached via the network


@dataclass(frozen=True)
class WalkRegion:
    """A rectangle of floor an agent may stand in. ``inset`` names the edges that are shrunk
    by the agent's radius (edges shared with an adjoining region are *not* inset so the
    agent can cross between them). ``requires_door`` makes the region usable only while
    that door is passable."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float
    inset: Edges = ALL_EDGES
    requires_door: str | None = None

    def contains(self, x: float, z: float, radius: float) -> bool:
        r = self.inset
        return (self.min_x + (radius if "min_x" in r else 0.0)) <= x <= (
            self.max_x - (radius if "max_x" in r else 0.0)
        ) and (self.min_z + (radius if "min_z" in r else 0.0)) <= z <= (
            self.max_z - (radius if "max_z" in r else 0.0)
        )


@dataclass(frozen=True)
class NoGoZone:
    """A region that is deliberately unreachable for an agent class. The traversability
    tests fail on any reachable-looking area that is not covered by one of these, so every
    inaccessible pocket in a level is a documented decision rather than an accident."""

    box: AABB
    reason: str
    for_player: bool = True
    for_npc: bool = True


@dataclass(frozen=True)
class SectorData:
    name: str
    bounds: Bounds
    props: tuple[Prop, ...] = ()
    colliders: tuple[AABB, ...] = ()
    nav_blockers: tuple[AABB, ...] = ()  # additional no-go boxes for NPC navigation only
    occluders: tuple[AABB, ...] = ()  # block *sight* only (walkable solids: decks, stair blocks)
    lights: tuple[Light, ...] = ()
    walk_regions: tuple[WalkRegion, ...] = ()


@dataclass(frozen=True)
class DoorData:
    """A sliding blast door. ``leaf`` props slide by ``-open_amount`` along X; ``indicator``
    props glow ``indicator_open`` / ``indicator_closed`` and slide with the leaf; ``lamps``
    glow the same way but are fixed to the wall (a lamp over the door)."""

    door_id: str
    obstacle: AABB
    unlocked_by: str | None  # terminal whose full hack (or door override) opens it; None = none
    travel: float = 3.2
    speed: float = 2.4
    passable_at: float = 1.5
    passage: WalkRegion | None = None  # floor strip through the doorway, usable once passable
    faction: str = DEFAULT_FACTION
    high_security: bool = False  # cannot be forced open by a quickhack; needs its terminal
    # Encryption-keyed: opens for a player carrying this intel (stolen data) who walks up
    # to it. Such a door is never forced by a quickhack.
    key_intel: str | None = None
    leaf: tuple[Prop, ...] = ()
    indicators: tuple[Prop, ...] = ()  # status strips on the leaf (they slide with it)
    lamps: tuple[Prop, ...] = ()  # status lamps fixed beside the doorway (they stay put)
    # None: once unlocked the door slides open and stays open. A distance: once unlocked it
    # opens only while somebody is within that many metres of it and slides shut behind
    # them (a staff door, not a blast door) - so it is still there, closed, green-lit,
    # when the netrunner who unlocked it from across the dome walks up to it.
    approach_range: float | None = None
    # A front door with a latch: locked or not, it opens for anybody on the inside who
    # walks up to it (within ``approach_range``). The side is "-x", "+x", "-z" or "+z" of
    # the door's obstacle box. None: no latch.
    inside: str | None = None
    # A hinged door (x, z of the hinge): the leaf props swing about it by ``swing``
    # degrees (about +y) when fully open, instead of sliding. None: it slides.
    hinge: tuple[float, float] | None = None
    swing: float = -85.0
    # Whether the netrunner's scanner finds it (and FORCE OPEN has a go at it). People's
    # own front doors are plain keyed locks, nothing to hack: False.
    hackable: bool = True
    # A lock that sets itself again this many seconds after it was forced (once nobody
    # is in the way): a metro ticket gate, reset by the station's system. None: never.
    relock_after: float | None = None
    # Status strips are lights, not paint: the emission carries the colour and the diffuse
    # colour is kept dark. (A diffuse colour above 1 multiplied every room light past white,
    # so a locked door's red strip - and an open one's green - both rendered plain white.)
    indicator_open: Material = Material(emission=(0.1, 3.6, 0.35), color=(0.02, 0.25, 0.06))
    indicator_closed: Material = Material(emission=(3.6, 0.2, 0.15), color=(0.25, 0.02, 0.02))


@dataclass(frozen=True)
class TerminalData:
    terminal_id: str
    x: float
    y: float
    z: float
    yaw: float = 0.0
    objective: bool = False  # stealing its data (by any route) wins the mission
    faction: str = DEFAULT_FACTION
    intel: str = ""  # name of the private data stored on it ("" = nothing worth stealing)
    secured: bool = False  # private: visible to the scanner, but never reachable via the network
    critical: bool = False  # holds critical secrets: cannot be overloaded/destroyed
    hardened: bool = False  # air-gapped: no wireless hacks at all, jack in with [E] or nothing
    # How the desk looks: "triple" (three screens), "double", "single", "damaged" (a dead
    # side screen, the other one gone), "overgrown" (two screens, fungus on the desk),
    # "home" (somebody's cheap little desk at home: one screen, a much smaller footprint)
    style: str = "triple"
    # What the netrunner reads on its screen once jacked in (a note, a file listing...)
    notes: str = ""


@dataclass(frozen=True)
class CameraData:
    x: float
    y: float
    z: float
    mount_yaw: float = 0.0
    sweep_angle: float = 35.0
    sweep_speed: float = 1.2
    camera_id: str = ""  # "" = assigned by the simulation (camera_1, camera_2, ...)
    faction: str = DEFAULT_FACTION
    # A home camera only cares about somebody inside the home: the id of the private area
    # (``LevelData.private_areas``) it reports intruders in. None: reports anybody it sees.
    watch_area: str | None = None


@dataclass(frozen=True)
class PanelData:
    """An access panel beside a door: it works the door's lock. Scanned and hacked
    (BYPASS LOCK), it releases the lock."""

    panel_id: str
    x: float
    y: float
    z: float
    door_id: str
    yaw: float = 0.0  # which way its face points (NPC convention: 0 = +z, 90 = +x)
    faction: str = DEFAULT_FACTION
    name: str = ""  # what the HUD calls it ("" = from the id)


@dataclass(frozen=True)
class PickupData:
    """Something the player can pick up with [E]: an item (``core.inventory``) lying at
    (x, y, z) - ``y`` is the surface it lies on."""

    pickup_id: str
    item_id: str
    x: float
    y: float
    z: float
    yaw: float = 0.0


@dataclass(frozen=True)
class PrivateArea:
    """Somebody's home. The netrunner is a trespasser inside ``boxes``: the owner (the
    person with ``owner_role``) and any camera watching the area report them. Caught in
    here (``LevelData.on_caught == "evict"``) they are thrown out onto ``exit`` (x, z,
    yaw), relieved of anything in ``confiscate`` (item ids) taken from inside, and the
    door (``door_id``) is locked behind them."""

    area_id: str
    name: str
    boxes: tuple[AABB, ...]
    exit: tuple[float, float, float]
    door_id: str | None = None
    confiscate: tuple[str, ...] = ()
    owner_role: str = ""

    def contains(self, x: float, z: float) -> bool:
        return any(b.contains_xz(x, z) for b in self.boxes)


@dataclass(frozen=True)
class VendingMachineData:
    x: float
    y: float
    z: float
    yaw: float = 0.0
    machine_id: str = ""  # "" = assigned by the simulation
    faction: str = DEFAULT_FACTION


@dataclass(frozen=True)
class OpenSky:
    """Ground out of doors with nothing but sky over it (``systems.weather``): the rain
    falls on it (down to ``ground``, its surface), and - if ``puddles`` - puddles gather
    on it when it has rained."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float
    ground: float = 0.0
    puddles: bool = True


@dataclass(frozen=True)
class LaneData:
    """A traffic lane (``entities.traffic``): a straight run along x at ``z``, the road's
    surface at ``y``. Vehicles come in at ``x_from`` (out of sight, in a tunnel) and
    drive to ``x_to`` (out of sight again), keeping to it."""

    lane_id: str
    z: float
    x_from: float
    x_to: float
    y: float = 0.0

    @property
    def heading(self) -> float:
        """+1 driving towards +x, -1 towards -x."""
        return 1.0 if self.x_to > self.x_from else -1.0

    @property
    def length(self) -> float:
        return abs(self.x_to - self.x_from)


@dataclass(frozen=True)
class MetroData:
    """A metro line through the level (``entities.metro``): its trains run along x at
    ``track_z`` between ``x_west`` and ``x_east`` (out of sight, in tunnels) and stop at a
    platform with their middle at ``stop_x``, their floor ``floor_y`` up, the platform's
    edge at ``edge_z`` (the doors open on that side). ``gates``: the x of each ticket gate
    lane at the station's entrance, across ``gates_z``."""

    track_z: float
    floor_y: float
    stop_x: float
    edge_z: float
    x_west: float
    x_east: float
    gates: tuple[float, ...] = ()
    gates_z: tuple[float, float] = (0.0, 0.0)
    # the station's stair (walked a tread at a time: ``ai.alley.Stairs``): the (x, z) of
    # the front edge of its first tread and of its top tread's back edge, and how many
    # treads it has (it climbs ``floor_y`` in them)
    stair_foot: tuple[float, float] = (0.0, 0.0)
    stair_top: tuple[float, float] = (0.0, 0.0)
    stair_steps: int = 0
    # the door (``LevelData.doors``) that stands for each gate lane's flaps: shut unless
    # forced (the pass-holders' opening of it is the metro's own, ``Metro.open_gate``)
    gate_doors: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardData:
    x: float
    y: float
    z: float
    patrol_z_min: float
    patrol_z_max: float
    wall_pos: tuple[float, float] | None  # lean spot; None = never leaves the patrol
    wall_yaw: float = 90.0
    initial_direction: float = 1.0  # +1 starts pacing toward patrol_z_max, -1 toward patrol_z_min
    guard_id: str = ""  # "" = assigned by the simulation
    faction: str = DEFAULT_FACTION
    essential: bool = False  # too important or too hardened to be knocked out by a quickhack
    # A closed loop of waypoints walked with A* instead of pacing the z lane (maze guards).
    patrol_route: tuple[tuple[float, float], ...] = ()

    @property
    def patrol_points(self) -> tuple[tuple[float, float], ...]:
        """Every point the guard's routine must be able to reach."""
        if self.patrol_route:
            return self.patrol_route
        return ((self.x, self.patrol_z_min), (self.x, self.patrol_z_max))


@dataclass(frozen=True)
class CivilianData:
    """A passer-by: spawn point, the points it wanders between, a seed for its look, and
    (``role``) the story it lives out - see ``ai.alley``; none: it just wanders."""

    x: float
    y: float
    z: float
    waypoints: tuple[tuple[float, float], ...]
    seed: int = 0
    civilian_id: str = ""
    role: str = ""


@dataclass(frozen=True)
class MerchantData:
    """Someone working a stall: where they stand, which way the counter faces, what they
    do ("cook" or "hawk") and (``role``) the trade they run - see ``ai.alley``."""

    x: float
    y: float
    z: float
    face_yaw: float
    style: str = "cook"
    seed: int = 100
    merchant_id: str = ""
    role: str = ""


@dataclass(frozen=True)
class Place:
    """A named spot the people of a level use: where to stand (``x``, ``z``, facing
    ``yaw``) or where something is (``y`` its height) - "sushi.customer", "sushi.board",
    "door16.stoop"... The stories in ``ai.alley`` find their way about by these."""

    name: str
    x: float
    z: float
    yaw: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class ItemData:
    """Something small lying about when the level starts (a knife on a chopping board),
    which people can pick up and put down (see ``render.items`` for what can be drawn)."""

    name: str
    x: float
    y: float
    z: float
    yaw: float = 0.0
    item_id: str = ""


@dataclass(frozen=True)
class LevelData:
    name: str
    sectors: tuple[SectorData, ...]
    brief: str = ""  # one line under the level title when the player arrives
    doors: tuple[DoorData, ...] = ()
    catwalk: Catwalk = field(default_factory=Catwalk)
    nav_bounds: Bounds = (-11.0, 11.0, -11.0, 43.0)
    spawn: tuple[float, float, float] = (0.0, 6.0, -90.0)  # x, z, yaw
    no_go_zones: tuple[NoGoZone, ...] = ()
    terminals: tuple[TerminalData, ...] = ()
    cameras: tuple[CameraData, ...] = ()
    vending_machines: tuple[VendingMachineData, ...] = ()
    guards: tuple[GuardData, ...] = ()
    factions: tuple[FactionData, ...] = ()  # empty = one faction per id the entities mention
    # Stepping into this box loads ``exit_to`` (the name of the next level in
    # ``levels.LEVELS``). There is no winning: the game goes on.
    exit_zone: AABB | None = None
    exit_to: str | None = None
    civilians: tuple[CivilianData, ...] = ()
    merchants: tuple[MerchantData, ...] = ()
    places: tuple[Place, ...] = ()
    items: tuple[ItemData, ...] = ()
    # What being caught costs: "restart" - a new game from the very beginning (the
    # facility: get out of the biodome without being caught or start over); "evict" -
    # thrown out of whichever private area the netrunner was caught in.
    on_caught: str = "restart"
    # Out of doors: the time of day shows (``core.clock.WorldClock.daylight``: a smoggy,
    # golden-brown day, a little brighter than the night; street lamps off in the day).
    outdoor: bool = False
    panels: tuple[PanelData, ...] = ()
    pickups: tuple[PickupData, ...] = ()
    private_areas: tuple[PrivateArea, ...] = ()
    # Low solids the player can step up onto (a doorstep): inside one the floor is at
    # its ``max_y``. NPCs keep off them (``SectorData.nav_blockers``) unless scripted.
    platforms: tuple[AABB, ...] = ()
    # Pairs of place names joined by a straight, clear walk (indoors, where the grid
    # router cannot thread the doorways): people follow them from place to place.
    routes: tuple[tuple[str, str], ...] = ()
    # The road's traffic lanes (``entities.traffic``): cars, taxis and vans driving by.
    lanes: tuple[LaneData, ...] = ()
    # A metro line with a station (``entities.metro``): trains that stop and go.
    metro: MetroData | None = None
    # The weather (``systems.weather``): where the rain falls and puddles gather (out of
    # doors), and the roofs over parts of that ground it stops on (their ``max_y``).
    open_sky: tuple[OpenSky, ...] = ()
    shelters: tuple[AABB, ...] = ()
    # Flights of stairs anybody can climb (``world.surfaces.Flight``: the metro's stair)
    # and the raised floors they lead to (decks: the floor at each box's ``max_y`` - the
    # landing, the footbridge, the platform), which people walk under as well as on.
    flights: tuple[Flight, ...] = ()
    decks: tuple[AABB, ...] = ()

    # ------------------------------------------------------------------ helpers
    def sector(self, name: str) -> SectorData:
        for s in self.sectors:
            if s.name == name:
                return s
        raise KeyError(name)

    def private_area(self, area_id: str) -> PrivateArea:
        for a in self.private_areas:
            if a.area_id == area_id:
                return a
        raise KeyError(area_id)

    @property
    def faction_ids_in_use(self) -> list[str]:
        """Every faction id referenced by an entity or door, in first-seen order."""
        seen: dict[str, None] = {}
        for group in (
            self.terminals, self.cameras, self.vending_machines, self.guards, self.doors,
            self.panels,
        ):  # fmt: skip
            for item in group:
                if getattr(item, "hackable", True):  # (a plain keyed front door: on no net)
                    seen.setdefault(item.faction, None)
        return list(seen)

    def resolved_factions(self) -> tuple[FactionData, ...]:
        """The declared factions, plus an auto-generated one for every id in use that the
        level did not declare (so a minimal level needs no faction table at all)."""
        declared = {f.faction_id: f for f in self.factions}
        extra = tuple(
            FactionData(fid, fid.replace("_", " ").upper())
            for fid in self.faction_ids_in_use
            if fid not in declared
        )
        return self.factions + extra

    @property
    def props(self) -> list[Prop]:
        return [p for s in self.sectors for p in s.props]

    @property
    def lights(self) -> list[Light]:
        return [light for s in self.sectors for light in s.lights]

    @property
    def walk_regions(self) -> list[WalkRegion]:
        return [r for s in self.sectors for r in s.walk_regions]
