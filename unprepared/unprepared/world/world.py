"""Live world state built from a :class:`LevelDefinition`, plus the town-level AI:
alarms and guard responses, investigations, trespass, curfews, shopkeeping and justice.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import GameSettings, DEFAULT_SETTINGS
from ..core import dialogue
from ..core.daynight import DayNightCycle
from ..core.math3d import BoundingBox3D, Vector3D, ray_intersects_box
from ..entities import AIState, Door, Item, NPC, Player, SkillType, TheftReport
from ..entities.inventory import add_item, take_item
from ..systems import commerce, justice
from ..systems import progression
from ..systems.stealth import observer_challenge, roll, sneak_challenge, sneak_chance
from .fixtures import generate_fixtures
from .geometry import disjoint_boxes
from .spatial import BoxIndex, PointIndex, SightBlockers
from .placement import Placer
from .terrain import Terrain
from .inn import InnState
from .level import AreaDef, BuildingDef, Container, LevelDefinition, LightDef, Prop, Readable, RoadDef, Station, Wall
from .navigation import NavGrid
from .tower import STEP_UP, StairDef

Notify = Optional[Callable[[str], None]]

NPC_HALF_WIDTH, NPC_HEIGHT, NPC_LYING_THICKNESS = 0.4, 1.8, 0.5
SLEEPER_ABOVE_TOP = 0.06           # a sleeper lies this far below a bed's top (on the blanket: see rendering.structures)
FLOOR_KINDS = ("step", "floor", "bridge")      # props people walk on (above the ground beneath)


def is_table_name(name: str) -> bool:
    """A thing named as a table ("shop_table", "inn_table_1") - a word of its name, not any name
    with the letters in it (the stable's haystacks are not tables)."""
    return "table" in name.split("_")


def _xyz(v: Vector3D) -> Tuple[float, float, float]:
    return (v.x, v.y, v.z)


@dataclass
class Relocation:
    """A request to move the player somewhere else (thrown out of a house, into a cell...).
    The presentation layer may delay ``apply`` until the screen has faded to black."""
    position: Vector3D
    yaw: Optional[float]
    reason: str
    applied: bool = False
    on_applied: Optional[Callable[[], None]] = None    # run once the move has actually happened

    def apply(self, player: Player) -> None:
        player.position = Vector3D(self.position.x, player.position.y, self.position.z)
        if self.yaw is not None:
            player.rotation_y = self.yaw
            player.rotation_x = 0.0
        self.applied = True
        if self.on_applied is not None:
            callback, self.on_applied = self.on_applied, None
            callback()


class WorldClock:
    """Simulated time of day. ``time_scale`` simulated minutes pass per real second."""

    def __init__(self, start_hour: int = 7, time_scale: float = 2.5, minutes_per_hour: float = 60.0):
        self.hour = start_hour
        self.minute_accumulator = 0.0
        self.time_scale = time_scale
        self.minutes_per_hour = minutes_per_hour

    def advance(self, delta_time: float) -> bool:
        """Advance by ``delta_time`` real seconds; returns True when the hour ticked over."""
        self.minute_accumulator += delta_time * self.time_scale
        if self.minute_accumulator >= self.minutes_per_hour:
            self.minute_accumulator -= self.minutes_per_hour     # keep the remainder: no drift on long frames
            self.hour = (self.hour + 1) % 24
            return True
        return False

    @property
    def fractional_hour(self) -> float:
        """Hour of day as a float (e.g. 7.5 = 07:30) - drives the sun and moon."""
        return self.hour + min(self.minute_accumulator, self.minutes_per_hour) / self.minutes_per_hour


class World:
    def __init__(self, level: LevelDefinition, settings: GameSettings = DEFAULT_SETTINGS, notify_callback: Notify = None,
                 rng: Optional[random.Random] = None):
        self.level = level
        self.settings = settings
        self.notify = notify_callback
        # where the town's routine news goes (who went where, the hour striking): the same log, told
        # apart by the caller that wants to (``Game`` marks it as news: no pop-up for it)
        self.news_callback: Optional[Notify] = None
        self.clock = WorldClock(settings.time.start_hour, settings.time.time_scale, settings.time.minutes_per_hour)
        self.daynight = DayNightCycle(settings.lighting.ambient_day_level, settings.lighting.ambient_night_level)
        # the lie of the land: everything below is authored with its floor at y = 0 and lifted onto it
        self.terrain = Terrain.shared(level.terrain, level.bounds, level.buildings)
        self.placer = placer = Placer(self.terrain)
        rebuild_wall = lambda original, mn, mx: Wall(BoundingBox3D(mn, mx), original.texture)   # noqa: E731
        # walls authored overlapping (a side wall and a front wall both covering a corner) are made
        # disjoint: the same solid union, but no two faces drawn in one plane to z-fight in the distance
        self.walls: List[Wall] = disjoint_boxes([piece for w in level.walls
                                                 for piece in placer.wall(w.build(), rebuild_wall, getattr(w, "segment", None))],
                                                rebuild_wall)
        self.props: List[Prop] = [p.build() for p in level.props]
        self.containers: List[Container] = [c.build() for c in level.containers]
        self.fixtures: List[LightDef] = [dataclasses.replace(f, position=_xyz(placer.point(Vector3D(*f.position))))
                                         for f in generate_fixtures(level, self.tables())]
        for prop in self.props:
            placer.box(prop, prop.anchor)
        for container in self.containers:
            placer.box(container)
        # what each household keeps on its table: the food it started with, bought back in every few days
        self.pantries: Dict[str, List[Item]] = {c.name: [i.build(owner=c.owner) for i in c.items if i.is_food]
                                                for c in level.containers if c.kind == "table"}
        self.stations: List[Station] = [placer.box(s.build()) for s in level.stations]
        self.readables: List[Readable] = [placer.box(r.build()) for r in getattr(level, "readables", ())]
        self.doors: List[Door] = [d.build() for d in level.doors]
        for door in self.doors:
            door.hinge_position = placer.point(door.hinge_position)
        self.npcs: List[NPC] = [self._lift_npc(n.build(settings.perception, settings.npc)) for n in level.npcs]
        # a bridge's rails: solid to walk into (not to see through - they are posts and a bar),
        # and fenced off on the navigation grid after the deck is cleared over the water, so a
        # way over the water runs onto the deck at its ends and down its middle
        self.rails: List[Prop] = self._bridge_rails()
        self.nav = NavGrid(self.static_obstacles(), dataclasses.replace(settings.navigation, bounds=level.bounds),
                           walls=self.walls, blocked_discs=self._water_discs(), blocked_ellipses=self._lake_ellipses(),
                           clear_boxes=[p for p in self.props if p.kind == "bridge"], ground=self.terrain.height,
                           blocked_where=self._unwalkable, fences=self.rails)
        # (a lintel over a gap some wall already fills - a tower's own door head - adds nothing but
        # faces in the same plane as that wall's: only what no wall covers is kept)
        self.lintels: List[Wall] = [piece for piece in disjoint_boxes(
            [*self.walls, *self._build_lintels()], lambda original, mn, mx: Wall(BoundingBox3D(mn, mx), original.texture)
        )[len(self.walls):]]
        self.stairs: List[StairDef] = [self._lift_stair(st) for st in level.stairs]
        self.props += self._thresholds()
        self.floors: List[Prop] = [p for p in self.props if p.kind in FLOOR_KINDS]   # walkable surfaces above the ground
        # what never moves, bucketed on a coarse grid: collisions and lines of sight look only nearby
        self._static_index = BoxIndex([*self.static_obstacles(), *self.rails])
        self._sight_index = BoxIndex([*self.static_obstacles(), *self.floors, *self._cave_roofs()])
        self._floor_index = BoxIndex(self.floors, cell=4.0)
        self._wall_index = BoxIndex(self.walls)
        self._prop_index = BoxIndex([*self.props, *self.containers], cell=4.0)
        self._beds: List[Prop] = [p for p in self.props if p.is_bed]
        self._indoor_lights = PointIndex([f for f in self.fixtures if f.kind != "torch"])
        self._indoor_lights_of = self.fixtures
        self.roads: List[RoadDef] = self._build_roads()
        self.day_count = 0              # days elapsed since the game began
        self.elapsed = 0.0              # real seconds simulated
        self.player: Optional[Player] = None
        self.player_is_solid = True            # (the start screen's roving eye is nobody to walk round)
        self.quick_travel = False              # (``pass_hour``: walks are not planned, only whether there is a way)
        self.planning_budget: Optional[float] = None    # seconds a frame may spend planning the next hour's walks
        self.switch_budget: Optional[float] = None      # seconds a frame may spend setting people off on the hour
        self.rng = rng if rng is not None else random.Random(settings.random_seed)    # gameplay rolls
        # Phrasings come from a stream of their own, so how the townsfolk happen to word things
        # never changes the outcome of a roll (scripted runs stay predictable).
        self.phrase_rng = random.Random(settings.random_seed)
        self.theft_reports: List[TheftReport] = []
        self.unanswered_reports: list = []          # (report, made at world time): waiting for a guard to come free
        self.suppress_curfews = False               # the tutorial: Jae has taken the player in, nobody is shown out tonight
        # the tutorial again: the town's guards stand where they are, blind and deaf, until it is
        # over - all but the ones named here (the watchman the tutorial itself brings in)
        self.guards_frozen = False
        self.unfrozen_guards: set = set()
        self.jail = justice.JailState.from_def(level.jail)
        if level.jail is not None:
            self.jail.cell_spot = placer.point(self.jail.cell_spot)
            self.jail.post = placer.optional_point(self.jail.post)
        self.inn = InnState.from_def(level.inn)
        self.conversations: list = []
        self.recent_topics: list = []               # openers used lately (kept short) so gossip varies
        self._chat_clock = 0.0
        # presentation hooks (set by Game): deferred relocations and "you have been caught" events
        self.relocation_handler: Optional[Callable[[Relocation], None]] = None
        self.caught_handler: Optional[Callable[[NPC], None]] = None
        self.relocations_in_flight: List[Relocation] = []
        self.wait_interruption: Optional[str] = None   # set while a wait is running to cut it short
        for npc in self.npcs:
            npc.rng = self.phrase_rng
        self.resync_schedules(announce=False, teleport=True)
        for npc in self.npcs:
            if npc.is_shopkeeper:
                commerce.remember_stock(npc, self)

    # --- convenience ---------------------------------------------------------------
    def reset_transient(self) -> None:
        """Drop everything that belongs to the running session and not to the saved state:
        conversations in progress, a relocation still fading, an interrupted wait, the jail's
        transfer lock. Every NPC forgets its own (``NPC.reset_transient``). Called before a save
        is applied to this world, so a load never inherits the old session's moment.

        Anybody who is not part of the level - the tutorial's watchman, brought in for one scripted
        arrest - leaves with the session too: a save carries no such person, so one left standing
        would keep whatever crime he held against the player into a game where it never happened
        (the tutorial calls him back itself if the save is of that very arrest)."""
        residents = {n.name for n in self.level.npcs}
        self.npcs = [npc for npc in self.npcs if npc.name in residents]
        self.unfrozen_guards = {name for name in self.unfrozen_guards if name in residents}
        for conv in self.conversations:
            conv.finished = True
        self.conversations = []
        self.recent_topics = []
        self._chat_clock = 0.0
        self.relocations_in_flight = []
        self.wait_interruption = None
        self.jail.transfer_pending = False
        self.inn.reset_welcome()
        for npc in self.npcs:
            npc.reset_transient()

    @property
    def save_blocked(self) -> Optional[str]:
        """Why the game must not be saved right now (None when it may be): the player is on their
        way to the cells, or being moved somewhere else - a save taken mid-move would put them
        back where they were with the consequences already applied."""
        if self.jail.transfer_pending:
            return "You are being taken to the cells."
        if self.relocating:
            return "You are being shown somewhere else."
        return None

    @property
    def simulated_hour(self) -> int:
        return self.clock.hour

    @simulated_hour.setter
    def simulated_hour(self, value: int) -> None:
        self.clock.hour = value

    @property
    def ambient_light(self) -> float:
        return self.daynight.ambient_light(self.clock.hour)

    def spawn_player(self) -> Player:
        lvl = self.level
        self.player = Player(self.spot(*lvl.player_spawn), gold=lvl.player_gold,
                             inventory=[i.build() for i in lvl.player_inventory], settings=self.settings.player,
                             clothing=self.settings.clothing)
        for item in self.player.inventory:
            if not item.is_stolen:
                item.owner = item.PLAYER_OWNER       # (stolen goods in the starting kit keep the owner the level gave them)
        return self.player

    @property
    def player_level(self) -> int:
        return self.player.level if self.player is not None else 1

    # --- the ground -----------------------------------------------------------------------
    def ground_height(self, x: float, z: float) -> float:
        """The terrain (or a building's floor) under (x, z)."""
        return self.terrain.height(x, z)

    def spot(self, x: float, y: float = 0.0, z: float = 0.0) -> Vector3D:
        """A point given in level coordinates (``y`` above the local floor) in the world."""
        return Vector3D(x, y + self.terrain.height(x, z), z)

    def ground_point(self, x: float, z: float) -> Vector3D:
        return Vector3D(x, self.terrain.height(x, z), z)

    def building_elevation(self, building: BuildingDef) -> float:
        """The height a building's floor is laid at (its pad)."""
        heights = getattr(self, "_pad_heights", None)
        if heights is None:
            heights = self._pad_heights = {p.name: p.height for p in self.terrain.pads}
        return heights.get(building.name, 0.0)

    def _lift_npc(self, npc: NPC) -> NPC:
        p = self.placer
        npc.position = p.point(npc.position)
        npc.home_position = p.point(npc.home_position)
        seen = set()
        for plan in (npc.schedule, *getattr(npc, "day_plans", ())):     # (today's is one of the plans: lift it once)
            if id(plan) in seen:
                continue
            seen.add(id(plan))
            for task in plan:
                task.target_pos = p.point(task.target_pos)
                task.patrol_points = tuple(p.point(v) for v in task.patrol_points)
        return npc

    def _lift_stair(self, stair: StairDef) -> StairDef:
        if self.terrain.flat:
            return stair
        lift = lambda v: _xyz(self.placer.point(Vector3D(*v)))   # noqa: E731
        return dataclasses.replace(stair, bottom=lift(stair.bottom), top=lift(stair.top),
                                   waypoints=tuple(lift(w) for w in stair.waypoints),
                                   rooms=tuple((rect, tuple(lift(d) for d in doors)) for rect, doors in stair.rooms))

    def _water_discs(self) -> List[Tuple[float, float, float]]:
        """(x, z, radius) discs along every stream's bed: nobody plans a walk through water (the
        player may wade; a bridge clears its span again)."""
        discs = []
        for water in self.terrain.water:
            pts = water["points"]
            half = water["width"] / 2.0 + 0.3
            for (x1, z1, _), (x2, z2, _) in zip(pts, pts[1:]):
                n = max(1, int(math.hypot(x2 - x1, z2 - z1) / 0.5))
                discs += [(x1 + (x2 - x1) * i / n, z1 + (z2 - z1) * i / n, half) for i in range(n + 1)]
        return discs

    RAIL_HEIGHT = 1.0           # a bridge's rail stands this high over its deck (``rendering.structures.draw_bridge``)
    RAIL_THICK = 0.12

    def _bridge_rails(self) -> List[Prop]:
        """The rails along every bridge's sides over the water (the way across runs between them:
        along x for a bridge ``facing`` 90/270, along z for 0/180) - as drawn, as boxes."""
        rails = []
        for b in self.props:
            if b.kind != "bridge":
                continue
            mn, mx = b.min_point, b.max_point
            top = mx.y + self.RAIL_HEIGHT
            if int(round(b.facing / 90.0)) % 2 == 1:
                boxes = [(Vector3D(mn.x, mx.y, mn.z), Vector3D(mx.x, top, mn.z + self.RAIL_THICK)),
                         (Vector3D(mn.x, mx.y, mx.z - self.RAIL_THICK), Vector3D(mx.x, top, mx.z))]
            else:
                boxes = [(Vector3D(mn.x, mx.y, mn.z), Vector3D(mn.x + self.RAIL_THICK, top, mx.z)),
                         (Vector3D(mx.x - self.RAIL_THICK, mx.y, mn.z), Vector3D(mx.x, top, mx.z))]
            rails += [Prop(f"{b.name}_rail_{i}", BoundingBox3D(lo, hi), texture="wood", kind="rail") for i, (lo, hi) in enumerate(boxes)]
        return rails

    WALK_GRADE = 1.0            # nobody climbs ground steeper than this (rise over run: 45 degrees)

    def _unwalkable(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Cell centres nobody plans a walk over: ground too steep to climb (a mountainside, the
        walls of a cutting) and the open sea."""
        t = self.terrain
        if t.flat:
            return np.zeros(X.shape, dtype=bool)
        slope = t.walk_slope()
        c = np.clip(np.floor((X - t.min_x) / t.cell).astype(int), 0, t.cols - 2)
        r = np.clip(np.floor((Z - t.min_z) / t.cell).astype(int), 0, t.rows - 2)
        blocked = slope[c, r] > self.WALK_GRADE
        # (steps cut between two floors at different levels - down into a cave's hall - are walked, however
        # sheer the drop between the floors would be without them)
        for p in self.props:
            if p.kind in ("step", "bridge"):
                blocked &= ~((X >= p.min_point.x) & (X <= p.max_point.x) & (Z >= p.min_point.z) & (Z <= p.max_point.z))
        if t.sea is not None:
            blocked |= X > t.coast_x(Z) + 1.5
        return blocked

    def _lake_ellipses(self) -> List[Tuple[float, float, float, float]]:
        return [(lake.center[0], lake.center[1], lake.radius_x, lake.radius_z) for lake in self.terrain.lakes]

    def find_area(self, name: str) -> Optional[AreaDef]:
        return next((a for a in self.level.areas if a.name == name), None)

    def area_def_at(self, position: Vector3D) -> Optional[AreaDef]:
        """The first named area the point lies in - an underground one (a cave's) only for somebody
        down in the cave, not for somebody up on the hillside over it."""
        below = None
        for area in self._by_cell("_area_grid", self.level.areas, position.x, position.z):
            if area.contains(position.x, position.z):
                if getattr(area, "underground", False):
                    if below is None:
                        building = self.building_at(position)
                        below = building is not None and building.underground
                    if not below:
                        continue
                return area
        return None

    def area_of(self, position: Vector3D) -> str:
        area = self.area_def_at(position)
        return area.name if area is not None else self.level.default_area

    # --- indoors / outdoors -------------------------------------------------------------
    def building_at(self, position: Vector3D, include_open: bool = False) -> Optional[BuildingDef]:
        """The walled building whose footprint contains ``position`` (None outdoors). Roofs on
        posts (``open_sides``) only count when ``include_open`` is set."""
        for building in self._by_cell("_building_grid", self.level.buildings, position.x, position.z):
            if building.open_sides and not include_open:
                continue
            (x1, z1), (x2, z2) = building.min_xz, building.max_xz
            if x1 <= position.x <= x2 and z1 <= position.z <= z2:
                if building.underground and position.y > self.building_elevation(building) + building.wall_height + 0.5:
                    continue                                  # up on the hillside over the cave, not in it
                return building
        return None

    LOOKUP_CELL = 8.0

    def _by_cell(self, attr: str, items, x: float, z: float):
        """The ``items`` (buildings, areas: anything with ``min_xz`` / ``max_xz``) whose footprint
        may hold (x, z), in their order - from a grid of ``LOOKUP_CELL`` cells built on first use
        (asked thousands of times a frame - who is indoors, who is where - over a hundred-odd
        footprints, most of them nowhere near)."""
        grid = self.__dict__.get(attr)
        if grid is None or grid[0] is not items:
            c = self.LOOKUP_CELL
            cells: Dict[Tuple[int, int], list] = {}
            for item in items:
                (x1, z1), (x2, z2) = item.min_xz, item.max_xz
                for cx in range(int(math.floor(x1 / c)), int(math.floor(x2 / c)) + 1):
                    for cz in range(int(math.floor(z1 / c)), int(math.floor(z2 / c)) + 1):
                        cells.setdefault((cx, cz), []).append(item)
            grid = self.__dict__[attr] = (items, cells)
        c = self.LOOKUP_CELL
        return grid[1].get((int(math.floor(x / c)), int(math.floor(z / c))), ())

    def _cave_roofs(self) -> List[BoundingBox3D]:
        """The rock over every cave chamber, as far as eyes go: nobody down there sees the hillside
        over their heads, nor anybody up on it them."""
        roofs = []
        for b in self.level.buildings:
            if b.underground:
                y = self.building_elevation(b) + b.wall_height
                roofs.append(BoundingBox3D(Vector3D(b.min_xz[0], y, b.min_xz[1]), Vector3D(b.max_xz[0], y + 0.4, b.max_xz[1])))
        return roofs

    def underground_zones(self) -> List[Tuple[float, float, float, float, float, float]]:
        """(min_x, min_z, max_x, max_z, floor, ceiling) of every cave chamber: the dark places."""
        zones = getattr(self, "_underground_zones", None)
        if zones is None:
            zones = []
            for b in self.level.buildings:
                if b.underground:
                    y = self.building_elevation(b)
                    zones.append((b.min_xz[0], b.min_xz[1], b.max_xz[0], b.max_xz[1], y - 0.6, y + b.wall_height + 0.45))
            self._underground_zones = zones
        return zones

    def is_indoors(self, position: Vector3D) -> bool:
        return self.building_at(position) is not None

    def same_room(self, a: Vector3D, b: Vector3D) -> bool:
        """Both outdoors, or both inside the same building."""
        return self.building_at(a) is self.building_at(b)

    def tables(self) -> List[BoundingBox3D]:
        """Every table-shaped prop or container (they get candles)."""
        found: List[BoundingBox3D] = [p for p in self.props if is_table_name(p.name)]
        found += [c for c in self.containers if getattr(c, "kind", "") == "table" or is_table_name(c.name)]
        return found

    def is_dark_at(self, position: Vector3D) -> bool:
        """The torch rule, kept simple: outdoors it is dark whenever the sun is down or nearly so
        (``DayNightCycle.is_torch_time``, about 17:30-06:30) - no outdoor light changes that;
        indoors it is dark only where no lantern or candle reaches."""
        if self.is_indoors(position):
            if self._indoor_lights_of is not self.fixtures:           # (the fixtures were replaced: index them again)
                self._indoor_lights = PointIndex([f for f in self.fixtures if f.kind != "torch"])
                self._indoor_lights_of = self.fixtures
            return not self._indoor_lights.lights(position)           # (wall torches burn outside)
        return DayNightCycle.is_torch_time(self.clock.fractional_hour)

    def same_chamber(self, a: Vector3D, b: Vector3D) -> bool:
        """Both outdoors, or in the same building with no wall or shut door between them: the
        inn's hall and a guest room behind a closed door are *not* the same chamber, the shop
        floor and the space behind the counter are."""
        building = self.building_at(a)
        if building is not self.building_at(b):
            return False
        if building is None:
            return True
        floor = self.building_elevation(building)
        pa, pb = Vector3D(a.x, floor + 1.0, a.z), Vector3D(b.x, floor + 1.0, b.z)
        return not any(ray_intersects_box(pa, pb, box) for box in self.walls_of(building))

    def walls_of(self, building: BuildingDef) -> List[BoundingBox3D]:
        """Wall segments and shut door leaves inside a building's footprint (interior partitions)."""
        (x1, z1), (x2, z2) = building.min_xz, building.max_xz
        boxes = [w for w in self.walls if w.min_point.x < x2 and w.max_point.x > x1 and w.min_point.z < z2 and w.max_point.z > z1]
        boxes += [d.collider() for d in self.doors if not d.is_open and not d.is_bars
                  and x1 <= d.center.x <= x2 and z1 <= d.center.z <= z2]
        return boxes

    def noise_attenuation(self, source: Vector3D, listener: Vector3D) -> float:
        """How much of a noise made at ``source`` reaches ``listener``: everything within the
        same chamber or open air, a fraction (``wall_noise_attenuation``) through walls -
        the walls of a building and the partitions and shut doors inside it alike."""
        return 1.0 if self.same_chamber(source, listener) else self.settings.perception.wall_noise_attenuation

    def exit_of(self, area_name: str) -> Optional[Vector3D]:
        """A spot just outside the door of a named building area (None for open areas).
        An area without a door of its own (a back room, the space behind a counter) uses
        the door of the building it is part of."""
        area = self.find_area(area_name)
        if area is None:
            return None
        centre = Vector3D((area.min_xz[0] + area.max_xz[0]) / 2.0, 0.0, (area.min_xz[1] + area.max_xz[1]) / 2.0)
        building = self.building_at(centre)
        if building is None:
            return None                                   # open ground: nothing to be thrown out of
        return self._exit_of_building(building, set())

    def _exit_of_building(self, building: BuildingDef, visited: set) -> Optional[Vector3D]:
        """A spot in the open just outside one of the building's doors. A door that only leads
        into *another* building (the watchtower's opens into the guard house) counts for that
        building's own way out instead, so nobody is ever "thrown out" into a second room."""
        visited.add(building.name)
        indirect: List[BuildingDef] = []
        for door in self.doors_of(building):
            outward = self._outward(door, building)
            c = door.center
            spot = self.ground_point(*self._across(door, c, outward * 2.0))
            beyond = self.building_at(spot)
            if beyond is None:
                return spot
            if beyond is not building and beyond.name not in visited:
                indirect.append(beyond)
        for beyond in indirect:
            spot = self._exit_of_building(beyond, visited)
            if spot is not None:
                return spot
        return None

    def door_of(self, building: BuildingDef) -> Optional[Door]:
        """The building's street door: one in its walls that opens onto the open air (None for
        a building whose only door leads into another building, or with no door at all)."""
        for door in self.doors_of(building):
            if door.is_bars:
                continue
            x, z = self._across(door, door.center, self._outward(door, building) * 2.0)
            if self.building_at(Vector3D(x, 0.0, z)) is None:
                return door
        return None

    def doors_of(self, building: BuildingDef) -> List[Door]:
        """The doors in (or on the edge of) a building's footprint."""
        (x1, z1), (x2, z2) = building.min_xz, building.max_xz
        return [d for d in self.doors if x1 - 0.6 <= d.center.x <= x2 + 0.6 and z1 - 0.6 <= d.center.z <= z2 + 0.6]

    @staticmethod
    def _outward(door: Door, building: BuildingDef) -> float:
        """+1 or -1: which way across its wall a door leads out of the building (away from its middle)."""
        (x1, z1), (x2, z2) = building.min_xz, building.max_xz
        middle = door.across_of((x1 + x2) / 2.0, (z1 + z2) / 2.0)
        return 1.0 if door.line >= middle else -1.0

    @staticmethod
    def yaw_away_from(door: Door, spot: Vector3D) -> float:
        """The yaw (0 = -z, 90 = +x) of somebody at ``spot`` turned away from ``door``'s wall."""
        side = door.side_of(spot.x, spot.z)
        if door.along_x:
            return 180.0 if side > 0 else 0.0
        return 90.0 if side > 0 else 270.0

    @staticmethod
    def _across(door: Door, point: Vector3D, offset: float) -> Tuple[float, float]:
        """(x, z) of ``point`` moved ``offset`` across the door's wall."""
        return door.to_xz(door.along_of(point.x, point.z), door.across_of(point.x, point.z) + offset)

    def waiting_spot_for(self, target: Vector3D) -> Optional[Vector3D]:
        """Where to wait for a building that is shut: just outside its door."""
        return self.exit_of(self.area_of(target))

    def find_door(self, door_id: str) -> Optional[Door]:
        return next((d for d in self.doors if d.door_id == door_id), None)

    def find_npc(self, name: str) -> Optional[NPC]:
        return next((n for n in self.npcs if n.name == name), None)

    def find_container(self, name: str) -> Optional[Container]:
        return next((c for c in self.containers if c.name == name), None)

    def find_station(self, name: str) -> Optional[Station]:
        return next((s for s in self.stations if s.name == name), None)

    def guards(self) -> List[NPC]:
        return [n for n in self.npcs if n.is_guard]

    def is_frozen(self, npc: NPC) -> bool:
        """A guard held still by the tutorial: no rounds, no eyes, no ears, no part in any alarm."""
        return self.guards_frozen and npc.is_guard and npc.name not in self.unfrozen_guards

    def freeze_guards(self, except_for: Sequence[str] = ()) -> None:
        """Hold every guard where they stand (see ``is_frozen``); ``except_for`` names stay free.
        Whoever was walking stops; whoever was after somebody forgets it."""
        self.guards_frozen = True
        self.unfrozen_guards = set(except_for)
        for guard in self.guards():
            if self.is_frozen(guard):
                guard.path, guard.goal = [], None
                if guard.state in (AIState.PATROL, AIState.TRAVELING, AIState.RESPONDING, AIState.ALERT_GUARDS,
                                   AIState.INVESTIGATING, AIState.FOLLOWING):
                    guard.state = AIState.GUARDING

    def unfreeze_guards(self) -> None:
        """The guards take up the hour's duties again (without a word about it)."""
        if not self.guards_frozen:
            return
        self.guards_frozen = False
        self.unfrozen_guards = set()
        for guard in self.guards():
            guard.update_schedule(self.clock.hour, None, world=self)

    def merchant_stock(self, npc: NPC) -> commerce.MerchantStock:
        return commerce.MerchantStock(npc, self)

    def owner_inventory_for(self, item: Item) -> List[Item]:
        """Where a stolen item goes back to, by its ``owner``: the owner's shelf or table that
        stocks that kind of thing, else the owner's own pockets (an NPC) or store (a faction's
        table or lockbox). Ownerless goods end up in the confiscation lockbox."""
        owner = item.owner
        if owner and owner != Item.PLAYER_OWNER:
            for cdef in self.level.containers:
                if cdef.owner == owner and any(i.id == item.id for i in cdef.items):
                    container = self.find_container(cdef.name)
                    if container is not None:
                        return container.inventory
            npc = self.find_npc(owner)
            if npc is not None:
                return npc.inventory
            for cdef in self.level.containers:
                if cdef.owner == owner:
                    container = self.find_container(cdef.name)
                    if container is not None:
                        return container.inventory
        chest = self.find_container(self.jail.storage_name)
        return chest.inventory if chest is not None else self.npcs[0].inventory

    # --- geometry ------------------------------------------------------------------
    def static_obstacles(self) -> List[BoundingBox3D]:
        """Everything solid that never moves: walls, props (bar decoration), containers and stations."""
        return [*self.walls, *(p for p in self.props if p.solid), *self.containers, *self.stations,
                *(r for r in getattr(self, "readables", ()) if r.solid)]

    THRESHOLD_INTO_JAMB = 0.005
    FLOOR_INSET = 0.48          # a building's floor stops this far inside its outline (``rendering.buildings.draw_floor``)

    def _thresholds(self) -> List[Prop]:
        """A strip of floor under every door in a building's outer wall: the floor inside stops at
        the wall's inner face, so the ground would show between it and the door leaf."""
        out: List[Prop] = []
        for door in self.doors:
            h = door.hinge_position
            lo, hi = door.span
            for sign in (-1.0, 1.0):                         # the building on the -across or the +across side of the leaf
                x, z = self._across(door, door.center, sign * 0.3)
                b = self.building_at(Vector3D(x, h.y + 0.5, z))
                if b is None or b.underground or b.open_sides:
                    continue
                (bx1, bz1), (bx2, bz2) = b.min_xz, b.max_xz
                edge = (door.across_of(bx1, bz1) if sign > 0 else door.across_of(bx2, bz2))
                if abs(edge - door.line) > 0.05:
                    continue                                  # a door within the building: its floor runs under it
                a_lo, a_hi = sorted((edge, edge + sign * self.FLOOR_INSET))
                # (a few millimetres into the jambs either side: its ends lay in the planes of the shut
                # leaf's edges and flickered there)
                (x1, z1), (x2, z2) = door.to_xz(lo - self.THRESHOLD_INTO_JAMB, a_lo), door.to_xz(hi + self.THRESHOLD_INTO_JAMB, a_hi)
                side = ("s" if sign > 0 else "n") if door.along_x else ("e" if sign > 0 else "w")
                out.append(Prop(f"{door.door_id}_threshold_{side}",
                                BoundingBox3D(Vector3D(min(x1, x2), h.y, min(z1, z2)), Vector3D(max(x1, x2), h.y + 0.05, max(z1, z2))),
                                texture=b.floor_texture, kind="floor", solid=False))
        return out

    def _build_lintels(self) -> List[Wall]:
        """Beams filling the gap between every door top and the ceiling. Purely visual:
        above head height, so they are neither colliders nor navigation obstacles."""
        lintels = []
        for door in self.doors:
            h = door.hinge_position
            # up to the eaves - of the lower of the two buildings a door between two joins (the tower's
            # door out of the guard house: a beam under the guard house's ceiling, not a slab up the tower)
            probes = [self._across(door, door.center, d) for d in (-0.3, 0.3)]
            sides = [b for b in (self.building_at(Vector3D(x, 0.0, z)) for x, z in probes) if b is not None]
            # (in a house of two storeys, up to the first floor: the upper storey's wall stands over it)
            top = (min(b.wall_height / max(1, getattr(b, "floors", 1)) for b in sides) if sides else door.height + 0.5) + h.y
            lo, hi = door.span
            # (no higher than a wall that heads the doorway and stops short of the eaves: over the
            # steeple's door stands a wall up to the belfry, whose arch above it is open to the sky)
            for wall in self.walls:
                w_lo, w_hi = (wall.min_point.x, wall.max_point.x) if door.along_x else (wall.min_point.z, wall.max_point.z)
                a_lo, a_hi = (wall.min_point.z, wall.max_point.z) if door.along_x else (wall.min_point.x, wall.max_point.x)
                if (w_lo <= lo + 1e-6 and w_hi >= hi - 1e-6 and a_lo <= door.line + 1e-6 and a_hi >= door.line - 1e-6
                        and h.y + door.height - 0.05 <= wall.min_point.y < top and wall.max_point.y < top - 0.05):
                    top = wall.min_point.y
            # (nor into the boards of an upper floor over it: flush with their top, it showed through them)
            for floor in self.props:
                if (floor.kind == "floor" and floor.min_point.x - 0.3 <= door.center.x <= floor.max_point.x + 0.3
                        and floor.min_point.z - 0.3 <= door.center.z <= floor.max_point.z + 0.3
                        and h.y + door.height - 0.05 <= floor.min_point.y < top):
                    top = floor.min_point.y
            if top <= h.y + door.height:
                continue
            texture = "building"
            for wall in self.walls:          # match the neighbouring wall's look
                w_lo, w_hi = (wall.min_point.x, wall.max_point.x) if door.along_x else (wall.min_point.z, wall.max_point.z)
                a_lo, a_hi = (wall.min_point.z, wall.max_point.z) if door.along_x else (wall.min_point.x, wall.max_point.x)
                if a_lo - 0.3 <= door.line <= a_hi + 0.3 and (abs(w_hi - lo) < 0.6 or abs(w_lo - hi) < 0.6):
                    texture = wall.texture
                    break
            half = 0.25
            (x1, z1), (x2, z2) = door.to_xz(lo, door.line - half), door.to_xz(hi, door.line + half)
            lintels.append(Wall(BoundingBox3D(Vector3D(min(x1, x2), h.y + door.height, min(z1, z2)),
                                              Vector3D(max(x1, x2), top, max(z1, z2))), texture))
        return lintels

    def _build_roads(self) -> List[RoadDef]:
        """The level's paving plus a doorstep for every door that stands back from it."""
        roads = list(self.level.roads)
        for door in self.doors:
            c = door.along_of(door.center.x, door.center.z)
            line = door.line
            best, best_gap = None, None
            for road in self.level.roads:
                (x1, z1), (x2, z2) = road.min_xz, road.max_xz
                a1, a2 = door.along_of(x1, z1), door.along_of(x2, z2)
                v1, v2 = door.across_of(x1, z1), door.across_of(x2, z2)
                if not (a1 <= c <= a2):
                    continue
                if v1 <= line <= v2:
                    best, best_gap = None, 0.0        # the door opens straight onto the paving
                    break
                gap = v1 - line if line < v1 else line - v2
                if best_gap is None or gap < best_gap:
                    best, best_gap = road, gap
            if best is not None and best_gap > 0.05:
                (x1, z1), (x2, z2) = best.min_xz, best.max_xz
                v1, v2 = door.across_of(x1, z1), door.across_of(x2, z2)
                va, vb = (line, v1) if line < v1 else (v2, line)
                lo, hi = door.span
                (ax, az), (bx, bz) = door.to_xz(lo - 0.5, va), door.to_xz(hi + 0.5, vb)
                roads.append(RoadDef((min(ax, bx), min(az, bz)), (max(ax, bx), max(az, bz)), best.texture))
        return roads

    def body_circles(self, exclude: Optional[NPC] = None) -> List[Tuple[float, float, float, float]]:
        """(x, z, radius, y) of every awake NPC (and the player) somebody could bump into."""
        bodies = []
        for npc in self.npcs:
            if npc is exclude or npc.is_asleep:
                continue
            bodies.append((npc.position.x, npc.position.z, NPC.BODY_RADIUS, npc.position.y))
        if self.player is not None and self.player_is_solid:
            bodies.append((self.player.position.x, self.player.position.z, self.settings.player.collision_radius,
                           self.player.position.y))
        return bodies

    def bodies_near(self, npc: NPC, radius: float = 6.0) -> List[Tuple[float, float, float]]:
        """(x, z, radius) of the people an NPC could bump into: close by and on the same level
        (somebody up the tower is no obstacle to somebody at its foot)."""
        p = npc.position
        return [(x, z, r) for x, z, r, y in self.body_circles(exclude=npc)
                if (x - p.x) ** 2 + (z - p.z) ** 2 <= radius * radius and abs(y - p.y) < 1.5]

    def npc_colliders(self) -> List[BoundingBox3D]:
        """Upright NPC bodies as boxes the player cannot walk through (sleepers are on beds, which already block)."""
        return [BoundingBox3D.around(n.position, NPC.BODY_RADIUS, NPC_HEIGHT, NPC.BODY_RADIUS)
                for n in self.npcs if not n.is_asleep]

    COLLIDER_REACH = 6.0        # ``colliders(near=...)``: boxes within this of the walker (a frame's step is far less)

    def colliders(self, include_npcs: bool = True, walker_y: Optional[float] = None,
                  near: Optional[Vector3D] = None) -> List[BoundingBox3D]:
        """Everything the player collides with right now (static geometry, door leaves, people) -
        just what is within ``COLLIDER_REACH`` of ``near`` if given. Steps and platforms
        (``floors``) are solid too - except the ones a walker standing at ``walker_y`` could step
        onto, which the floor rule handles instead."""
        if near is not None:
            r = self.COLLIDER_REACH
            boxes = [*self._static_index.near(near.x, near.z, r),
                     *(door.collider() for door in self.doors if door.hinge_position.distance_to(near) <= r + door.width)]
            floors = self._floor_index.near(near.x, near.z, r)
        else:
            boxes = [*self.static_obstacles(), *self.rails, *(door.collider() for door in self.doors)]
            floors = self.floors
        y = self.player.position.y if walker_y is None and self.player is not None else (walker_y or 0.0)
        boxes.extend(f for f in floors if f.max_point.y > y + STEP_UP)
        if include_npcs:
            boxes.extend(self.npc_colliders())
        return boxes

    def floor_height_at(self, x: float, z: float, current_y: float) -> float:
        """The floor under (x, z) for somebody standing at ``current_y``: the highest step,
        platform or bridge there that is no more than a step up (``STEP_UP``), else the ground."""
        best = self.terrain.height(x, z)
        for f in self._floor_index.near(x, z, 0.0):
            if f.min_point.x <= x <= f.max_point.x and f.min_point.z <= z <= f.max_point.z:
                top = f.max_point.y
                if top <= current_y + STEP_UP and top > best:
                    best = top
        return best

    def stair_under(self, position: Vector3D) -> Optional[StairDef]:
        """The stair somebody up a tower (above the floor inside its footprint) is on."""
        if position.y <= self.terrain.height(position.x, position.z) + 0.05:
            return None
        return next((st for st in self.stairs if st.contains(position.x, position.z)), None)

    def stair_to(self, goal: Vector3D) -> Optional[StairDef]:
        """The stair that leads to ``goal`` (a spot up a tower, an upper floor)."""
        if goal.y <= self.terrain.height(goal.x, goal.z) + 0.05:
            return None
        return next((st for st in self.stairs if st.contains(goal.x, goal.z)), None)

    def stair_region_at(self, x: float, z: float) -> Optional[StairDef]:
        return next((st for st in self.stairs if st.contains(x, z)), None)

    def settle(self, npc: NPC, target: Vector3D, fraction: float) -> None:
        """Set a walker's height after a step towards ``target`` (``fraction`` of the way there):
        on a stair, rise or drop with its waypoints; anywhere else, stand on the ground (or on a
        bridge) under their feet."""
        p = npc.position
        if self.stairs and self.stair_region_at(p.x, p.z) is not None:
            if target.y != p.y:                       # inside a stair's footprint the waypoints say
                p.y = target.y if fraction >= 1.0 else p.y + (target.y - p.y) * fraction
            return
        p.y = self.floor_height_at(p.x, p.z, p.y)

    def push_out_of_doors(self, player: Player) -> bool:
        """If a door leaf was swung into the player, shove them clear of it (never trap them)."""
        moved = False
        for door in self.doors:
            leaf = door.collider()
            box = player.collision_box()
            if not box.intersects(leaf):
                continue
            r = self.settings.player.collision_radius + 0.05
            if (leaf.max_point.x - leaf.min_point.x) < (leaf.max_point.z - leaf.min_point.z):
                # the leaf lies along z (a door along x standing open, or one along z shut): push in x
                if player.position.x >= door.hinge_position.x:
                    player.position.x = leaf.max_point.x + r
                else:
                    player.position.x = leaf.min_point.x - r
            else:                 # the leaf lies along x: push in z
                if player.position.z >= door.hinge_position.z:
                    player.position.z = leaf.max_point.z + r
                else:
                    player.position.z = leaf.min_point.z - r
            moved = True
        return moved

    def sight_blockers(self) -> SightBlockers:
        """Geometry that blocks line of sight: static obstacles, steps and platforms (nobody sees
        through a floor) plus closed door leaves (bars are see-through). A sequence of them all,
        which also answers ``along(a, b)`` with just the ones near one line of sight."""
        return SightBlockers(self._sight_index, [door.collider() for door in self.doors if door.blocks_sight])

    # --- the town gate and the water -------------------------------------------------
    def gate_leaves(self, gate: str) -> List[Door]:
        return [d for d in self.doors if d.gate == gate]

    def open_gate(self, gate: str) -> None:
        """Unbar and swing open both leaves of a gateway (the town lets the player go)."""
        for door in self.gate_leaves(gate):
            door.is_locked = False
            door.is_open = True
        if self.player is not None:
            self.push_out_of_doors(self.player)

    def close_gate(self, gate: str) -> None:
        for door in self.gate_leaves(gate):
            door.is_open = False
            door.is_locked = True

    DEEP_WATER = 0.8            # water deeper than this over the bed is not waded into

    def is_deep_water(self, x: float, z: float) -> bool:
        level = self.terrain.water_level_at(x, z)
        return level is not None and level - self.terrain.height(x, z) > self.DEEP_WATER

    def can_step(self, x0: float, z0: float, y0: float, x1: float, z1: float) -> bool:
        """May somebody standing at (x0, y0, z0) move to (x1, z1)? Not up ground steeper than
        ``WALK_GRADE`` (a mountainside, the side of a cutting), not out of the map, not into the
        open sea, and not down through a hillside into the cave beneath it. Steps and floors are
        the floor rule's business (``floor_height_at``)."""
        x_lo, z_lo, x_hi, z_hi = self.level.bounds
        if not (x_lo <= x1 <= x_hi and z_lo <= z1 <= z_hi):
            return False
        t = self.terrain
        if t.flat:
            return True
        if self.is_deep_water(x1, z1):
            return False
        cave = t.underground_at(x1, z1)
        if cave is not None and y0 > cave.height + 2.5 and t.underground_at(x0, z0) is None:
            return False                                   # on the hillside over a cave, not in it
        y1 = self.floor_height_at(x1, z1, y0)
        ground = t.height(x1, z1)
        if y1 > ground + 0.01:
            return True                                    # onto a step, a bridge, a floor
        rise = ground - y0
        if y0 > t.height(x0, z0) + 0.01 and rise <= STEP_UP:
            return True                                    # off a step or a deck onto the floor beside it
        run = math.hypot(x1 - x0, z1 - z0)
        return rise <= self.WALK_GRADE * run + 0.02

    READABLE_REACH = 6.0

    def doors_in(self, x1: float, z1: float, x2: float, z2: float) -> List[Door]:
        """The doors whose doorways overlap the rectangle (their extents worked out once: doors
        swing but never move)."""
        extents = getattr(self, "_door_extents", None)
        if extents is None or len(extents) != len(self.doors):
            extents = self._door_extents = [(d, d.x1, d.z1, d.x2, d.z2) for d in self.doors]
        return [d for d, a1, b1, a2, b2 in extents if a1 <= x2 and a2 >= x1 and b1 <= z2 and b2 >= z1]

    INTERACTION_REACH = 12.0    # nothing further than this from the eye can be the crosshair's target

    def interaction_candidates(self, position: Vector3D):
        """(doors, npcs, containers, stations, beds) within ``INTERACTION_REACH`` of ``position`` -
        all the crosshair could settle on (its reach is a few paces: the rest of the town's
        hundreds of doors, chests and beds need not be ray-cast every frame)."""
        static = getattr(self, "_static_centres", None)
        if static is None:
            def centre(box):
                return ((box.min_point.x + box.max_point.x) / 2.0, (box.min_point.z + box.max_point.z) / 2.0)
            static = self._static_centres = {
                "doors": [(d, d.center.x, d.center.z) for d in self.doors],
                "containers": [(c, *centre(c)) for c in self.containers],
                "stations": [(st, *centre(st)) for st in self.stations],
                "beds": [(b, *centre(b)) for b in self._beds],
            }
        r = self.INTERACTION_REACH
        px, pz = position.x, position.z

        def near(entries):
            return [o for o, x, z in entries if -r <= x - px <= r and -r <= z - pz <= r]
        npcs = [n for n in self.npcs if -r <= n.position.x - px <= r and -r <= n.position.z - pz <= r]
        return near(static["doors"]), npcs, near(static["containers"]), near(static["stations"]), near(static["beds"])

    def readables_near(self, position: Vector3D) -> List[Readable]:
        """Notices, signposts and journals within reading reach."""
        r = self.READABLE_REACH
        return [rd for rd in self.readables if abs(rd.center.x - position.x) <= r and abs(rd.center.z - position.z) <= r]

    def walls_near(self, position: Vector3D, radius: float) -> List[BoundingBox3D]:
        """The wall pieces within ``radius`` of ``position`` (what can hide a target from the crosshair)."""
        return self._wall_index.near(position.x, position.z, radius)

    def prop_under(self, position: Vector3D) -> Optional[BoundingBox3D]:
        """The prop/container whose footprint contains ``position`` (a bed an NPC lies on), if any -
        on the floor they stand on, not one a storey above or below them."""
        for box in self._prop_index.near(position.x, position.z, 0.0):
            if getattr(box, "kind", "") in FLOOR_KINDS:
                continue
            if not (box.min_point.y - 0.5 <= position.y <= box.max_point.y + 0.1):
                continue
            if box.min_point.x <= position.x <= box.max_point.x and box.min_point.z <= position.z <= box.max_point.z:
                return box
        return None

    def npc_body_box(self, npc: NPC) -> BoundingBox3D:
        """The NPC's body volume: upright normally, lying flat (on top of its bed) while asleep."""
        p = npc.position
        if not npc.is_asleep:
            return BoundingBox3D.around(p, NPC_HALF_WIDTH, NPC_HEIGHT, NPC_HALF_WIDTH)
        bed = self.prop_under(p)
        base_y = (bed.max_point.y - SLEEPER_ABOVE_TOP) if bed is not None else p.y      # on the blanket, not above it
        if bed is not None and getattr(bed, "kind", "") == "bed":
            along_x = int(round(bed.facing / 90.0)) % 2 == 1                           # the bed's length runs head to foot
        else:
            along_x = bed is None or (bed.max_point.x - bed.min_point.x) >= (bed.max_point.z - bed.min_point.z)
        half_len, half_w = NPC_HEIGHT / 2.0, NPC_HALF_WIDTH
        hx, hz = (half_len, half_w) if along_x else (half_w, half_len)
        return BoundingBox3D(Vector3D(p.x - hx, base_y, p.z - hz), Vector3D(p.x + hx, base_y + NPC_LYING_THICKNESS, p.z + hz))

    def sleep_head(self, npc: NPC) -> str:
        """Which end of the lying body the head is at ("-x", "+x", "-z", "+z"): the bed's headboard."""
        bed = self.prop_under(npc.position)
        if bed is None or getattr(bed, "kind", "") != "bed":
            box = self.npc_body_box(npc)
            return "-x" if (box.max_point.x - box.min_point.x) >= (box.max_point.z - box.min_point.z) else "-z"
        return {0: "-z", 1: "+x", 2: "+z", 3: "-x"}[int(round(bed.facing / 90.0)) % 4]

    def find_path(self, npc: NPC, goal: Vector3D) -> List[Vector3D]:
        """A walk from the NPC to ``goal``: over the navigation grid on the ground, with the
        stair waypoints spliced in to get down from (or up) a tower. In a quick hour (``pass_hour``)
        only whether there is a way counts: the walk is the goal alone (nobody walks it)."""
        if self.quick_travel:
            return [goal.copy()] if self.nav.route_exists(npc.position, goal, self.doors, npc.can_open) else []
        here = self.stair_under(npc.position)
        there = self.stair_to(goal)
        path: List[Vector3D] = []
        start = npc.position
        if here is not None:
            if there is here:
                return self._stair_path(here, npc.position, goal)
            path += self._stair_path(here, npc.position, Vector3D(*here.bottom))
            start = Vector3D(*here.bottom)
        if there is not None:
            ground = self._on_the_ground(self.nav.find_path(start, Vector3D(*there.bottom), self.doors, npc.can_open))
            if not ground and start.distance_to(Vector3D(*there.bottom)) > self.settings.npc.arrival_tolerance:
                return []
            return path + ground + self._stair_path(there, Vector3D(*there.bottom), goal)
        ground = self._on_the_ground(self.nav.find_path(start, goal, self.doors, npc.can_open))
        if here is not None and not ground and start.distance_to(goal) > self.settings.npc.arrival_tolerance:
            return []
        return path + ground

    PLAN_FROM_MINUTE = 50.0     # ``plan_ahead`` works in the hour's last minutes (the doors as they will be)

    def plan_ahead(self, budget: float) -> int:
        """Spend up to ``budget`` seconds planning the walks people will set off on at the next stroke
        of the hour, from where they stand now (the navigation grid keeps every route it finds).
        On the hour itself a score of them set off at once, and a frame that had to plan them all
        stalled for a fifth of a second; planned ahead, the hour costs nothing. Only the timing
        changes: the walk planned on the hour is the same one. Returns the routes planned."""
        import time as _time
        if not self.npcs or self.quick_travel:
            return 0
        if self.clock.minute_accumulator < self.PLAN_FROM_MINUTE:
            return 0                      # (too early: the doors will have changed by the hour, and the routes with them)
        start = _time.perf_counter()
        next_hour = (self.clock.hour + 1) % 24
        tolerance = self.settings.npc.arrival_tolerance
        planned = 0
        n = len(self.npcs)
        doors = tuple((d.is_open, d.is_locked) for d in self.doors)
        for _ in range(n):
            self._plan_cursor = (getattr(self, "_plan_cursor", -1) + 1) % n
            npc = self.npcs[self._plan_cursor]
            key = (self.day_count, next_hour, doors)
            if getattr(npc, "_planned_for", None) == key:
                continue
            npc._planned_for = key
            task, upcoming = npc.current_task, npc.task_for(next_hour)
            if (upcoming is None or upcoming is task or task is None or npc.path or npc.is_busy
                    or npc.position.distance_to(task.target_pos) > tolerance):
                continue
            self.find_path(npc, upcoming.target_pos)              # (into the route cache)
            planned += 1
            if _time.perf_counter() - start >= budget:
                break
        return planned

    def _on_the_ground(self, waypoints: List[Vector3D]) -> List[Vector3D]:
        """Ground legs of a walk: every waypoint but the goal at the height of the ground under it."""
        if self.terrain.flat:
            return waypoints
        for wp in waypoints[:-1]:
            wp.y = self.floor_height_at(wp.x, wp.z, wp.y)
        return waypoints

    @staticmethod
    def _stair_path(stair: StairDef, start: Vector3D, goal: Vector3D) -> List[Vector3D]:
        """Waypoints along a stair between two of its points (up or down)."""
        points = [Vector3D(*stair.bottom), *(Vector3D(*p) for p in stair.waypoints)]
        top = points[-1]

        def nearest(p: Vector3D) -> int:
            if p.y >= top.y - 0.2:
                return len(points) - 1              # on the floor the stair arrives at: by way of its landing
            return min(range(len(points)), key=lambda i: points[i].distance_to(p))

        a, b = nearest(start), nearest(goal)
        leg = points[a:b + 1] if a <= b else points[b:a + 1][::-1]
        # out of one room upstairs and into another through their doorways (by way of the landing,
        # unless the two rooms share the way in: the dressing room through the study)
        last = len(points) - 1
        out = [Vector3D(*d) for d in stair.doorways_to(start.x, start.z)] if a == last else []
        into = [Vector3D(*d) for d in stair.doorways_to(goal.x, goal.z)] if b == last else []
        shared = 0
        while shared < min(len(out), len(into)) and out[shared] == into[shared]:
            shared += 1
        if shared:
            leg = out[shared:][::-1] + into[shared:]
        else:
            leg = out[::-1] + leg + into
        if not leg or goal.distance_to(leg[-1]) > 0.3:
            leg = leg + [goal.copy()]
        return [p.copy() for p in leg]

    # --- trust ----------------------------------------------------------------------
    def area_trusts_player(self, area: AreaDef) -> bool:
        """Is the player a welcome guest here (outside curfew hours)? Explicitly via
        ``trusted_by``, or - the standing rule - because a fence owns the place: a fence's
        house is always open to the player by day, and nobody, guards included, objects."""
        if area.curfew_active(self.clock.hour):
            return False
        if area.trusted_by:
            return True
        if self.inn.is_rented(self.day_count, self.clock.hour) and area.name in (self.inn.rented_area, self.inn.area):
            return True                                   # a paying guest's room - and the inn's hall after closing time
        owner = self.find_npc(area.owner) if area.owner else None
        return owner is not None and owner.is_fence

    @property
    def beds(self) -> List[Prop]:
        return self._beds

    def player_may_sleep_in(self, bed: Prop) -> bool:
        """The player's own bed (``owner == "player"``), the bed of the inn room they are renting, or
        - for as long as the sentence lasts, and no longer - the bed in the cell they are serving
        it in."""
        if bed.owner == Item.PLAYER_OWNER:
            return True
        if self.is_prisoners_bed(bed):
            return True
        inn = self.inn
        if inn.rented_room is None or not inn.is_rented(self.day_count, self.clock.hour):
            return False
        bx, _, bz = inn.rooms[inn.rented_room].bed
        return bed.min_point.x <= bx <= bed.max_point.x and bed.min_point.z <= bz <= bed.max_point.z

    def is_prisoners_bed(self, bed: Prop) -> bool:
        """A bed in the jail cell while the player is locked up in it (``JailState.sealed``): theirs
        until the sentence is served (or they break out)."""
        jail = self.jail
        if not (jail.present and jail.sealed):
            return False
        cell = self.find_area(jail.cell_area)
        if cell is None:
            return False
        cx, cz = (bed.min_point.x + bed.max_point.x) / 2.0, (bed.min_point.z + bed.max_point.z) / 2.0
        (x1, z1), (x2, z2) = cell.min_xz, cell.max_xz
        return x1 <= cx <= x2 and z1 <= cz <= z2

    def is_complimentary(self, container: Container, item: Item) -> bool:
        """Food and drink in the inn are on the house for a paying guest."""
        inn = self.inn
        return (inn.present and inn.keeper is not None and container.owner == inn.keeper and item.is_food
                and inn.is_rented(self.day_count, self.clock.hour))

    def note_complimentary(self, container: Container, item: Item) -> None:
        """A free meal taken: keep the keeper's stock count in step so it is not a theft to be
        discovered at the next stock-take."""
        keeper = self.find_npc(self.inn.keeper) if self.inn.keeper else None
        if keeper is not None and container.name in keeper.shop_containers:
            commerce.note_sale(keeper, item, -1)

    def is_trespassing(self, player: Player, npc: NPC) -> bool:
        """Would ``npc`` object to where the player stands right now?"""
        area = self.area_def_at(player.position)
        if area is None or not area.restricted or area.is_open_to_public(self.clock.hour):
            return False
        if self.area_trusts_player(area) or npc.name in area.trusted_by:
            return False
        if area.name in (self.jail.cell_area, self.jail.jail_area):
            if self.jail.sealed:
                return False                          # a prisoner belongs in the cell
            if self.jail.released or self.elapsed < self.jail.grace_until:
                return False                          # just released: collecting their things
        responsible = area.owner is not None and area.owner in (npc.name, npc.faction)   # (nobody's is nobody's)
        return responsible or npc.is_guard

    def player_unwelcome_at(self, position: Vector3D) -> Optional[AreaDef]:
        """The restricted area the player has no business in at ``position`` (None if fine)."""
        area = self.area_def_at(position)
        if area is None or not area.restricted or area.is_open_to_public(self.clock.hour):
            return None
        if area.name in (self.jail.cell_area, self.jail.jail_area) and (
                self.jail.sealed or self.jail.released or self.elapsed < self.jail.grace_until):
            return None
        if self.area_trusts_player(area):
            return None
        return area

    # --- relocation (thrown out / jailed) -----------------------------------------------
    def relocate_player(self, position: Vector3D, yaw: Optional[float] = None, reason: str = "",
                        on_applied: Optional[Callable[[], None]] = None) -> Relocation:
        """Move the player, via the presentation's handler (fade to black) when one is installed;
        ``on_applied`` runs once the move has actually taken place (at black, or at once)."""
        request = Relocation(position.copy(), yaw, reason, on_applied=on_applied)
        if self.relocation_handler is not None:
            self.relocations_in_flight.append(request)
            self.relocation_handler(request)
        elif self.player is not None:
            request.apply(self.player)
        return request

    @property
    def relocating(self) -> bool:
        """A relocation has been requested but not applied yet (the screen is fading)."""
        self.relocations_in_flight = [r for r in self.relocations_in_flight if not r.applied]
        return bool(self.relocations_in_flight)

    def player_caught(self, by: NPC) -> None:
        """An NPC has the player bang to rights: the player turns to face them and whatever
        screen they had open (a mini-game, somebody's pockets) is closed by the game."""
        if self.player is not None:
            self.player.hold_attention(by.position, self.settings.justice.caught_freeze_seconds)
        self.interrupt_wait("caught")
        if self.caught_handler is not None:
            self.caught_handler(by)

    def throw_out(self, player: Player, area_name: str, by: Optional[NPC] = None, line: Optional[str] = None) -> bool:
        """The standard eviction: out of the building's door, facing away from it."""
        exit_pos = self.exit_of(area_name)
        if exit_pos is None:
            return False
        if by is not None:
            self.player_caught(by)
        self.interrupt_wait("evicted")
        if line and self.notify:
            self.notify(line)
        yaw = None
        area = self.find_area(area_name)
        if area is not None:
            cz = (area.min_xz[1] + area.max_xz[1]) / 2.0
            yaw = 180.0 if exit_pos.z > cz else 0.0     # face away from the door
        self.relocate_player(exit_pos, yaw=yaw, reason="evicted")
        return True

    def exit_of_room(self, door_id: str, area_name: str) -> Optional[Vector3D]:
        """A spot just outside a room's own door, on the far side from the room (the inn's hall
        for a guest room) - where somebody shown out of the *room* ends up, still indoors."""
        door = self.find_door(door_id)
        area = self.find_area(area_name)
        if door is None or area is None:
            return None
        room_middle = door.across_of((area.min_xz[0] + area.max_xz[0]) / 2.0, (area.min_xz[1] + area.max_xz[1]) / 2.0)
        outward = -1.0 if room_middle > door.line else 1.0
        return self.ground_point(*self._across(door, door.center, outward * 1.5))

    def show_out_of_room(self, player: Player, area_name: str, door_id: str, by: Optional[NPC] = None,
                         line: Optional[str] = None) -> bool:
        """Shown out of one room into the rest of the building (a guest whose rent has run out is
        put out of the room, not out of the inn). Falls back to the building's door."""
        spot = self.exit_of_room(door_id, area_name)
        if spot is None:
            return self.throw_out(player, area_name, by=by, line=line)
        if by is not None:
            self.player_caught(by)
        self.interrupt_wait("evicted")
        if line and self.notify:
            self.notify(line)
        door = self.find_door(door_id)
        yaw = self.yaw_away_from(door, spot) if door is not None else 180.0             # face away from the door
        self.relocate_player(spot, yaw=yaw, reason="evicted")
        return True

    def evict_if_trespassing(self, player: Player, guard: NPC, notify: Notify = None) -> bool:
        """Caught somewhere they should not be: the guard shows them the door as well."""
        area = self.player_unwelcome_at(player.position)
        if area is None:
            return False
        return self.throw_out(player, area.name, by=guard,
                              line=f"({guard.name}): \"And you've no business in {area.name}. Out!\" You are thrown out.")

    # --- simulation ---------------------------------------------------------------
    def resync_schedules(self, announce: bool = True, teleport: bool = False, doors: bool = True) -> None:
        """Put every NPC on the task for the current hour. ``doors=False`` (loading a save) keeps
        every door exactly as saved: NPCs resume their tasks without re-applying lock/unlock effects."""
        notify = self.notify if announce else None
        for npc in self.npcs:
            npc.use_day(self.day_count)                       # (a save loaded on another day of the round)
            task = npc.task_for(self.clock.hour)
            if teleport and task is not None:
                if doors:
                    self._replay_day_until(npc, self.clock.hour)      # (a load keeps its doors as they were saved)
                npc.position = task.target_pos.copy()
            npc.update_schedule(self.clock.hour, notify, world=self, doors=doors)
        if doors:
            for door in self.doors:                               # (a chapel is open by day, whoever came and went)
                if door.schedule is not None and door.open_hours_cover(self.clock.hour):
                    door.is_locked = False

    def _replay_day_until(self, npc: NPC, hour: int) -> None:
        """Apply the door side effects of every task the NPC has already completed today,
        including locking its own doors behind it whenever it moved between areas."""
        tasks = sorted(npc.schedule, key=lambda t: t.start_hour)
        for idx, task in enumerate(tasks):
            if task.end_hour > hour:
                continue
            for door_id in task.unlock_doors:
                door = self.find_door(door_id)
                if door is not None:
                    door.is_locked = False
            for door_id in task.open_doors:
                door = self.find_door(door_id)
                if door is not None:
                    door.unlock_and_open()
            for door_id in task.close_doors:
                door = self.find_door(door_id)
                if door is not None:
                    door.is_open = False
            for door_id in task.lock_doors:
                door = self.find_door(door_id)
                if door is not None:
                    door.lock_and_close()
            following = tasks[idx + 1] if idx + 1 < len(tasks) else None
            if following is not None and self.building_at(task.target_pos, True) is not self.building_at(following.target_pos, True):
                for door in self.doors:               # leaving the building: its owner locks up behind itself
                    if door.auto_closes and npc.owns(door) and npc.can_lock(door):
                        door.lock_and_close()

    NEWS_RANGE = 40.0           # the player hears what people are up to within this (or under the same roof)

    def in_town(self, position: Vector3D) -> bool:
        """Inside the town wall (a level without one: everywhere)."""
        area = self.level.town_area
        return area is None or (area[0] <= position.x <= area[2] and area[1] <= position.z <= area[3])

    def news_of(self, npc: NPC) -> Notify:
        """Where an NPC's comings and goings are reported: the log, if the player is near enough
        to know of it (or under the same roof) - nobody in the square hears that the smugglers have
        gone to bed in their cave, nor, with sixty people about, what everybody in town is doing
        (the log holds a handful of lines: the hour's news from every street would push out the
        warnings that matter)."""
        news = self._news
        if news is None or self.player is None:
            return news
        p = self.player.position
        if npc.position.distance_to(p) <= self.NEWS_RANGE:
            return news
        here = self.building_at(p, True)
        if here is not None and here is self.building_at(npc.position, True):
            return news
        return None

    @property
    def _news(self) -> Notify:
        return self.news_callback if (self.news_callback is not None and self.notify is not None) else self.notify

    def update_time(self, delta_time: float) -> bool:
        """Advance the clock; on a new hour apply door schedules and NPC schedules."""
        self.elapsed += delta_time
        if not self.clock.advance(delta_time):
            return False
        hour = self.clock.hour
        for npc in self.npcs:
            npc.time = self.elapsed                 # so anything said on the hour is timed right
        if hour == 0:
            self.day_count += 1
            self._daily_upkeep()
            for npc in self.npcs:
                npc.use_day(self.day_count)              # a new day: some have a different one to live
        for door in self.doors:
            msg = door.apply_schedule(hour)
            if msg and self.notify:
                self._news(msg)                          # (the town's news: a chapel opening, a gate shutting)
        if self.notify:
            self._news(f"Time Update: {hour:02d}:00 hours")
        for npc in self.npcs:
            if self.is_frozen(npc):
                continue
            if self.switch_budget is not None and not self.quick_travel and npc.task_for(hour) is not npc.current_task:
                continue                             # (sets off a moment later: ``update_npcs`` staggers them)
            npc.update_schedule(hour, self.news_of(npc), world=self)
        if self.player is not None:
            justice.update_jail(self, self.player, self.notify)
            self.inn.update(self, self.player, self.notify)
        return True

    def _daily_upkeep(self) -> None:
        """Midnight: NPCs replace missing personal items (a guard requisitions a new key) and,
        every restock cycle, top their purses back up to their usual amount."""
        for npc in self.npcs:
            if self.day_count - npc.last_inventory_reset_day >= npc.inventory_reset_days:
                restored = npc.reset_inventory(self.day_count)
                if restored and self.notify and any(i.is_key for i in restored):
                    self.notify(f"({npc.name}) has been issued a new key.")
            if self.day_count - npc.last_gold_restock_day >= npc.restock_days:
                npc.gold = max(npc.gold, npc.base_gold)
                npc.last_gold_restock_day = self.day_count
        if self.day_count % max(1, self.settings.economy.pantry_restock_days) == 0:
            self._restock_pantries()

    def _restock_pantries(self) -> None:
        """Households buy provisions: every kitchen table is topped back up to the food it started with."""
        for container in self.containers:
            for stock in self.pantries.get(container.name, ()):
                have = next((i for i in container.inventory if i.id == stock.id), None)
                if have is None:
                    add_item(container.inventory, dataclasses.replace(stock))
                elif have.quantity < stock.quantity:
                    have.quantity = stock.quantity

    def wait_hours(self, hours: float, step: float = 0.5, player: Optional[Player] = None) -> float:
        """Pass time quickly (the player stands still); NPCs keep living their routines. With a
        ``player`` the town keeps its eyes open meanwhile (perception runs every step) and the
        wait stops the moment somebody lays hands on them or shows them the door
        (``interrupt_wait``). Returns the hours that actually passed."""
        waited = hours
        for waited in self.waiting(hours, step, player):
            pass
        return waited

    WAIT_SLICE = 2              # steps of a wait between the moments it hands back (``waiting``: a frame's worth)

    def waiting(self, hours: float, step: float = 0.5, player: Optional[Player] = None):
        """``wait_hours`` a slice at a time: a generator yielding the hours passed so far every
        ``WAIT_SLICE`` steps (a caller that must keep drawing - the [T] wait under its progress
        box - takes it a frame's worth at a time). The last value is the hours that passed."""
        self.wait_interruption = None
        steps = int(hours * (self.clock.minutes_per_hour / self.clock.time_scale) / step)
        per_hour = self.clock.minutes_per_hour / self.clock.time_scale
        for done in range(steps):
            self.update_time(step)
            self.update_npcs(step)
            if player is not None:
                self.update_perception(player, False, False, step)
                if self.wait_interruption is not None:
                    yield (done + 1) * step / per_hour
                    return
            if (done + 1) % self.WAIT_SLICE == 0 and done + 1 < steps:
                yield (done + 1) * step / per_hour
        yield float(hours)

    def interrupt_wait(self, reason: str) -> None:
        """Something happened to the player that a wait ([T], ``wait_hours`` with a player) must not
        skip over. Sleep (``wait_until``) is not cut short: a sleeper shown out of a room whose rent
        ran out wakes where they were put, at the hour they meant to."""
        self.wait_interruption = reason

    # --- long waits: an hour at a time, the people put where they were going ------------------
    def seconds_to_next_hour(self) -> float:
        c = self.clock
        return max(0.0, (c.minutes_per_hour - c.minute_accumulator) / c.time_scale)

    def pass_hour(self, quick: bool = True, step: float = 0.5, settle: bool = True) -> None:
        """Carry the world on to the next stroke of the hour. Slowly (``quick=False``), the way
        every frame does; or quickly - the clock jumps to the hour, the hour's tick runs (doors,
        the day's upkeep, the jail, the inn, everybody's schedule) and whoever sets off for the
        new task is put where they were going and arrives there (eating, unlocking, locking up
        behind them as they would have). A night's sleep or days in the cells is hours of this:
        a few milliseconds each, not the half second of living every moment of them.
        ``settle=False``: the hour strikes and everybody sets off, but nobody is put anywhere (the
        last stroke of a long wait - the town is seen on the move when the player comes to)."""
        if not quick:
            seconds = self.seconds_to_next_hour()
            hour = self.clock.hour
            while self.clock.hour == hour:
                dt = min(step, max(1e-3, seconds))
                self.update_time(dt)
                self.update_npcs(dt)
                seconds -= dt
            return
        self.elapsed += self.seconds_to_next_hour()
        self.clock.minute_accumulator = self.clock.minutes_per_hour
        self.quick_travel = True
        try:
            self.update_time(0.0)                          # (the accumulator is full: the hour strikes)
            if settle:
                self.settle_npcs()
        finally:
            self.quick_travel = False
        if not settle:
            for npc in self.npcs:                          # (the walks they set off on, planned for real)
                task = npc.current_task
                if npc.state is AIState.TRAVELING and task is not None and not npc.is_busy and not npc.locked_out:
                    npc.path = self.find_path(npc, task.target_pos)

    def settle_npcs(self) -> None:
        """Everybody on the way to a task (or waiting on a doorstep for one to open) is put there
        and arrives - the quick hours of ``pass_hour``. Somebody busy (a report, a search) is left
        to finish it when time runs at its usual pace again."""
        hour = self.clock.hour
        for npc in self.npcs:
            if self.is_frozen(npc) or npc.is_busy:
                continue
            news = self.news_of(npc)
            if npc.locked_out and npc.current_task is not None:
                npc.retry_timer = 0.0
                npc.retry_blocked_task(self, 0.0, news)           # (the door may have opened by now)
            task = npc.current_task
            if task is None or task is not npc.task_for(hour) or npc.state is not AIState.TRAVELING or npc.locked_out:
                continue
            left = self.building_at(npc.position, True)
            if left is not self.building_at(task.target_pos, True):
                for door in self.doors:                         # (out of the house: the owner locks up behind)
                    if (door.auto_closes and npc.owns(door) and npc.can_lock(door)
                            and not (door.schedule is not None and door.open_hours_cover(hour))):
                        door.lock_and_close()
            npc.position = task.target_pos.copy()
            npc.path = []
            npc.goal = None
            npc._process_pending_locks(self)                    # (a door it opened on the way out, shut again)
            npc.arrive(self, news)

    def wait_until(self, hour: int, step: float = 0.5) -> None:
        """Pass time until the clock next strikes ``hour`` (a full day if it is that hour already)."""
        for _ in self.sleeping_until(hour, step):
            pass

    def hours_until(self, hour: int) -> int:
        """Strokes of the clock until it next strikes ``hour`` (24 if it is that hour now)."""
        return (hour - self.clock.hour - 1) % 24 + 1

    def sleeping_until(self, hour: int, step: float = 0.5):
        """``wait_until`` as a generator of progress (0 .. 1): the first hour lived in full (what
        was under way - a report, a walk home - plays out), the rest quick hours (``pass_hour``),
        handed back one at a time. Ends exactly on the stroke of ``hour``."""
        total = self.hours_until(hour)
        for strokes in range(total):
            if strokes == 0:
                before = self.clock.hour
                seconds = self.seconds_to_next_hour()
                done = 0
                while self.clock.hour == before:
                    dt = min(step, max(1e-3, seconds))
                    self.update_time(dt)
                    self.update_npcs(dt)
                    seconds -= dt
                    done += 1
                    if done % self.WAIT_SLICE == 0:
                        yield min(0.99, (1.0 - seconds / (self.clock.minutes_per_hour / self.clock.time_scale)) / total)
            else:
                self.pass_hour(quick=True, settle=strokes < total - 1)
            yield (strokes + 1) / total
        self.clock.minute_accumulator = 0.0

    def update_npcs(self, delta_time: float) -> None:
        hour = self.clock.hour
        import time as _time
        switch_time, switches = 0.0, 0
        for npc in self.npcs:
            if self.is_frozen(npc):
                npc.has_torch = not npc.is_asleep and self.is_dark_at(npc.position)
                continue                                              # held by the tutorial: a statue with a torch
            news = self.news_of(npc)                                   # (the log hears of the player's part of the world)
            # missed the hour tick (or, in the app, staggered: a frame plans only ``switch_budget``
            # seconds of new walks, so a score of people setting off on the hour is spread over
            # a few frames instead of stalling one)? take it up now
            if self.switch_budget is None:
                npc.catch_up_schedule(hour, news, world=self)
            elif switches == 0 or switch_time < self.switch_budget:
                began = _time.perf_counter()
                if npc.catch_up_schedule(hour, news, world=self):
                    switches += 1
                    switch_time += _time.perf_counter() - began
            if not npc.is_asleep and not npc.path:
                self._get_out_of_bed(npc)
            npc.update_movement(delta_time, self, news)
            self._tick_ai(npc, delta_time)
            npc.has_torch = not npc.is_asleep and self.is_dark_at(npc.position)     # lit wherever it is dark
        self._tick_talk(delta_time)
        if self.planning_budget:
            self.plan_ahead(self.planning_budget)
        if self.unanswered_reports:
            self._dispatch_unanswered_reports()
        if self.player is not None:
            self.push_out_of_doors(self.player)
            self._tick_jail()
            self._tick_curfews(delta_time)
            self._tick_inn_welcome(delta_time)

    def _get_out_of_bed(self, npc: NPC) -> None:
        """An awake NPC standing on a bed (just woken) steps off it to the nearest free spot."""
        under = self.prop_under(npc.position)
        if under is None or "bed" not in getattr(under, "name", ""):
            return
        cell = self.nav.nearest_free(self.nav.to_cell(npc.position.x, npc.position.z), self.nav.static_blocked)
        if cell is not None:
            spot = self.nav.to_world(cell, npc.position.y)
            npc.position.x, npc.position.z = spot.x, spot.z

    # --- jail house ----------------------------------------------------------------
    def _tick_jail(self) -> None:
        jail = self.jail
        if self.relocating:
            return                                   # the player is on their way somewhere; judge them on arrival
        if jail.serving:
            justice.check_jailbreak(self, self.player, self.notify)
        elif jail.released:
            left = self.area_of(self.player.position) not in (jail.cell_area, jail.jail_area)
            if left or self.elapsed >= jail.grace_until:
                justice.end_release(self, self.notify)

    # --- town AI ----------------------------------------------------------------------
    def _tick_ai(self, npc: NPC, delta_time: float) -> None:
        """Investigations, alarm responses and tailing: what an NPC does off its routine."""
        npc.retry_blocked_task(self, delta_time, self.notify)
        if npc.state is AIState.INVESTIGATING:
            if npc.heard_noise_at is not None and npc.goal is None and npc.is_guard:
                self._investigate(npc, npc.heard_noise_at)
                npc.heard_noise_at = None
            if not npc.path:
                npc.search_timer += delta_time
                if (npc.goal is not None and npc.search_timer > 3.0
                        and npc.suspicion_meter < self.settings.perception.schedule_override_threshold):
                    # nothing found here: have a look around (a guard with a suspect in view keeps
                    # watching them instead - see _watch_and_follow)
                    npc.turn_towards(Vector3D(npc.position.x + 1, 0, npc.position.z))
                if npc.search_timer >= npc.SEARCH_SECONDS * 0.5:
                    npc.resume_routine(self, self.notify)
        elif npc.state is AIState.RESPONDING and npc.report_to is not None:
            self._go_and_take_report(npc, delta_time)
        elif npc.state is AIState.RESPONDING:
            if not npc.path:
                npc.search_timer += delta_time
                if npc.search_timer >= npc.SEARCH_SECONDS:
                    if npc.search_timer - delta_time < npc.SEARCH_SECONDS:
                        npc.say(npc.phrase("gone"), self.notify)
                    npc.resume_routine(self, self.notify)
        elif npc.state is AIState.FOLLOWING:
            npc.follow_timer += delta_time
            if npc.follow_timer >= npc.FOLLOW_SECONDS:
                npc.resume_routine(self, self.notify)
        elif npc.state is AIState.ALERT_GUARDS:
            # after a thief: ``pursue`` (called while the player is in view) keeps the chase going
            # and the search timer at zero; out of sight for long enough the guard gives up
            npc.repath_timer = max(0.0, npc.repath_timer - delta_time)
            npc.search_timer += delta_time
            if npc.search_timer >= npc.SEARCH_SECONDS:
                npc.say(npc.phrase("gone"), self.notify)
                npc.resume_routine(self, self.notify)
        elif npc.state is AIState.REPORTING:
            npc.search_timer += delta_time
            if npc.search_timer >= npc.SEARCH_SECONDS * 0.5:
                npc.resume_routine(self, self.notify)
        elif npc.state is AIState.WAITING:
            if self.player is not None and not npc.path:
                npc.turn_towards(self.player.position)
            if npc.waiting_for == "release" and not self.jail.released:
                npc.stop_waiting(self, self.notify)
        elif npc.is_guard and npc.heard_noise_at is not None and npc.state in (AIState.PATROL, AIState.GUARDING):
            where = npc.heard_noise_at
            npc.heard_noise_at = None
            if self._noise_from_cell(where):
                npc.scold("cell_noise", "cell_noise", self.notify, grace=False)
            else:
                self._investigate(npc, where)
                npc.say(npc.phrase("investigate"), self.notify)
        else:
            npc.heard_noise_at = None
        self._ambient_chatter(npc, delta_time)

    def _noise_from_cell(self, where: Vector3D) -> bool:
        """A sealed prisoner rattling about is never a reason to open the cell."""
        return self.jail.sealed and self.area_of(where) == self.jail.cell_area

    def _investigate(self, npc: NPC, where: Vector3D) -> None:
        if self._noise_from_cell(where):
            return
        npc.go_to(self, where, AIState.INVESTIGATING)

    def report_crime_to_witnesses(self, crime_name: str, player_pos: Vector3D, exclude: Sequence[NPC] = ()) -> List[NPC]:
        """Let every NPC (but ``exclude`` - the victim of a pickpocket is not a bystander) try to
        witness a crime; fresh witnesses raise the alarm, ones already alarmed see it quietly.
        Returns everyone who saw it. Sight is checked against the player's chest at their current
        stance, like all perception."""
        blockers = self.sight_blockers()
        height = NPC.chest_height(self.player) if self.player is not None else None
        witnesses: List[NPC] = []
        fresh: List[NPC] = []
        for npc in self.npcs:
            if npc in exclude or self.is_frozen(npc):
                continue
            was_alarmed = npc.is_alarmed
            if npc.witness_crime(crime_name, player_pos, blockers, self.notify, target_height=height):
                witnesses.append(npc)
                if not was_alarmed:
                    fresh.append(npc)
        if witnesses:
            self.declare_crime(crime_name, player_pos.copy(), witnessed=True,
                               reporter=witnesses[0].name, witnesses=witnesses, new_witnesses=fresh)
        return witnesses

    def declare_crime(self, crime: str, location: Vector3D, witnessed: bool, reporter: str,
                      witnesses: Optional[List[NPC]] = None, new_witnesses: Optional[List[NPC]] = None) -> None:
        """A crime became known: guards who saw it remember it for weeks, the others hear the
        cry for help (if they are close enough to), remember it for days and come running. The
        same deed declared again within moments (the bystanders' roll and then the victim's) is
        held once per guard (``NPC.remember_crime``), sends nobody running twice, and only
        ``new_witnesses`` (default: all of them) count against the player's record."""
        penalty = justice.penalty_for(crime, self.is_indoors(location), self.settings.justice)
        witnesses = witnesses or []
        reporting = self.find_npc(reporter)
        for witness in [*witnesses, *([reporting] if reporting is not None else [])]:
            witness.remember_sighting(self.area_of(location), self.day_count)
        ready_at = self.elapsed + self.settings.justice.charge_delay     # the chase comes before the charge
        for guard in self.guards():
            saw_it = guard in witnesses
            if self.is_frozen(guard) or (not saw_it and not self.can_hear_cry(guard, location)):
                continue
            already = guard.recent_memory(crime, self.day_count, self.is_indoors(location), self.elapsed) is not None
            guard.remember_crime(crime, self.day_count, saw_it, penalty.fine, self.rng, jail_days=penalty.jail_days,
                                 jail_days_unpaid=penalty.jail_days_unpaid, indoors=self.is_indoors(location),
                                 now=self.elapsed)
            guard.charge_ready_at = max(guard.charge_ready_at, ready_at)
            if not saw_it and not already:
                self._send_guard(guard, location)
        if self.player is not None:
            self.player.crimes_witnessed += len(witnesses if new_witnesses is None else new_witnesses)

    def can_hear_cry(self, npc: NPC, location: Vector3D) -> bool:
        """A shout for the guards carries ``shout_radius`` in the open, less through walls. A
        look-out up a tower is in the open air, whatever the footprint below him says."""
        s = self.settings.perception
        up_high = npc.position.y > self.terrain.height(npc.position.x, npc.position.z) + 2.0     # (a look-out up his tower)
        in_the_open = self.same_room(location, npc.position) or (up_high and not self.is_indoors(location))
        reach = s.shout_radius * (1.0 if in_the_open else s.shout_wall_attenuation)
        return npc.position.distance_to(location) <= reach

    def _send_guard(self, guard: NPC, location: Vector3D) -> None:
        if guard.state is AIState.RESPONDING or self.jail.sealed and guard.name == self.jail.jailer:
            return
        was_asleep = guard.is_asleep
        guard.end_conversation()
        guard.state = AIState.GUARDING
        guard.search_timer = 0.0
        guard.go_to(self, location, AIState.RESPONDING)
        guard.say(guard.phrase("cry_response"), self.notify, log_prefix=f"{guard.name} woke up" if was_asleep else None)

    PERCEPTION_RANGE = 60.0     # beyond this nobody sees or hears the player (the loudest step carries ~50)

    def update_perception(self, player: Player, is_moving: bool, is_sprinting: bool, delta_time: float = 0.0) -> None:
        ambient = self.ambient_light
        blockers = self.sight_blockers()
        sneaking = player.is_crouching and is_moving
        player.indoors = self.is_indoors(player.position)          # (the torch law is the street's)
        if sneaking and delta_time > 0:
            self._sneak_practice(player, delta_time, blockers)
        for npc in self.npcs:
            if self.is_frozen(npc):
                continue
            if (npc.position.distance_to(player.position) > self.PERCEPTION_RANGE and npc.suspicion_meter <= 0.0
                    and not npc.is_alarmed and npc.warden_seize_at is None):
                continue                                     # out of sight and earshot, and calm: nothing to update
            npc.sneak_hidden = npc.sneak_unnoticed_until > self.elapsed
            self._shopkeeper_watches(npc, player, blockers)
            visible = npc.can_see(player.position, blockers, NPC.chest_height(player))
            if sneaking and visible:
                self._sneak_check(npc, player, delta_time, ambient)
            npc.sneak_hidden = npc.sneak_unnoticed_until > self.elapsed
            trespassing = self.is_trespassing(player, npc) and visible and not npc.sneak_hidden
            was_alarmed = npc.is_alarmed
            npc.evaluate_perception(player, ambient, is_moving, is_sprinting, blockers, self.clock.hour,
                                    notify_callback=self.notify, delta_time=delta_time, trespassing=trespassing,
                                    noise_attenuation=self.noise_attenuation(player.position, npc.position))
            if (npc.is_guard and is_moving and not npc.is_asleep and npc.heard_noise_at is None
                    and self._noise_from_cell(player.position)):
                # a prisoner rattling the bars is heard (and shouted at) though skulking in a lit cell is no
                # crime; ``_noise_from_cell`` keeps the jailer from opening up to look
                loudness = player.get_noise_level(is_moving, is_sprinting) * self.noise_attenuation(player.position, npc.position)
                if npc.position.distance_to(player.position) <= npc.hearing_radius_for(loudness):
                    npc.heard_noise_at = player.position.copy()
            if npc.is_alarmed and not was_alarmed:
                # what the alarm is *about*: being somewhere forbidden, creeping about, or - merely
                # walking around unlit after dark - the torch law
                unlit = not DayNightCycle.is_daytime(self.clock.hour) and not player.has_lit_torch and not player.indoors
                crime = ("Trespassing" if trespassing else "Skulking about" if player.is_crouching
                         else "No torch after dark" if unlit else "Skulking about")
                self.declare_crime(crime, player.position.copy(), witnessed=npc.is_guard, reporter=npc.name,
                                   witnesses=[npc] if npc.is_guard else [])
                if npc.is_guard:
                    npc.search_timer = 0.0
            if npc.pending_crime is not None:                    # warned enough (the torch law): on the record
                self.declare_crime(npc.pending_crime, player.position.copy(), witnessed=True, reporter=npc.name, witnesses=[npc])
                npc.pending_crime = None
            self._watch_and_follow(npc, player, visible and not npc.sneak_hidden, sneaking)
            if npc.suspicion_meter >= self.settings.perception.schedule_override_threshold:
                # anyone who has taken notice remembers where (a shopkeeper tells the guards about it later)
                npc.remember_sighting(self.area_of(player.position), self.day_count)
                self._keeper_objects_to_skulking(npc, player)
            if npc.is_guard and visible and not npc.sneak_hidden and not npc.is_asleep:
                self._guard_sees_player(npc, player)
            if npc.warden_of:
                self._warden_watch(npc, player, visible and not npc.sneak_hidden)

    # --- wardens: outlaws and knights keep their own peace ---------------------------------
    WARDEN_GRAB_DELAY = 2.0      # seconds from the shout before hands are laid on the intruder ...
    WARDEN_REACH = 3.0           # ... once the warden is this close
    WARDEN_GIVE_UP = 20.0        # out of sight this long after the shout, they let it go
    WARDEN_TOLL = 0.25           # the share of the purse they help themselves to ...
    WARDEN_TOLL_MIN = 10.0       # ... and never less than this, if the purse holds it

    def _warden_watch(self, npc: NPC, player: Player, visible: bool) -> None:
        """A warden who has raised the alarm goes after the intruder while they are in view (or in
        the domain) and, once close, throws them out (``_warden_seize``); out of sight long enough,
        it settles back to its routine."""
        if npc.is_asleep or self.relocating or self.jail.sealed:
            return
        if npc.warden_seize_at is None:
            if npc.is_alarmed and (visible or self.area_of(player.position) in npc.warden_of):
                npc.warden_seize_at = self.elapsed + self.WARDEN_GRAB_DELAY
            return
        if visible:
            npc.pursue(self, player.position, self.notify)
            if self.elapsed >= npc.warden_seize_at and npc.position.distance_to(player.position) <= self.WARDEN_REACH:
                self._warden_seize(npc, player)
        elif self.elapsed >= npc.warden_seize_at + self.WARDEN_GIVE_UP:
            npc.warden_seize_at = None
            npc.suspicion_meter = 0.0
            npc.resume_routine(self, None)

    def _warden_seize(self, warden: NPC, player: Player) -> None:
        """The intruder is dragged out to the warden's ``warden_exit``: whatever they stole from the
        warden's people goes back where it came from, and a toll comes out of their purse. No
        town watch, no cells - the outlaws and the knights settle it themselves."""
        faction = warden.faction or warden.name
        band = [n for n in self.npcs if n is warden or (n.warden_of and (n.faction or n.name) == faction)]
        owners = {faction, *(n.name for n in band)}
        recovered: List[Item] = []
        for item in list(player.inventory):
            if item.is_stolen and item.owner in owners:
                unit = take_item(player.inventory, item, item.quantity)
                if unit is not None:
                    unit.is_stolen = False
                    add_item(self.owner_inventory_for(unit), unit)
                    recovered.append(unit)
        toll = min(player.gold, max(self.WARDEN_TOLL_MIN, round(player.gold * self.WARDEN_TOLL)))
        player.gold -= toll
        warden.gold += toll
        self.player_caught(warden)
        where = next((a for a in warden.warden_of), "their ground")
        took = []
        if recovered:
            took.append("take back what you stole")
        if toll > 0:
            took.append(f"help themselves to {toll:.0f} gold")
        if self.notify:
            self.notify(f"({warden.name}): \"{dialogue.warden_line(warden, 'seize', self.phrase_rng)}\" You are dragged out of "
                        f"{where}" + (f" - they {' and '.join(took)}." if took else "."))
        player.crimes_witnessed += 1
        for n in band:
            n.warden_seize_at = None
            n.suspicion_meter = 0.0
            n.forget_crimes()
            n.resume_routine(self, None)
        exit_spot = warden.warden_exit.copy() if warden.warden_exit is not None else self.exit_of(where)
        if exit_spot is not None:
            self.relocate_player(self.ground_point(exit_spot.x, exit_spot.z), reason="thrown out")

    def sneak_xp_rate(self, player: Player, blockers=None) -> float:
        """Sneak XP per second for crouch-moving right now: a trickle alone, more near every
        person who has not noticed you - sleepers and backs turned at the "behind" rate, right
        under an awake nose at the "in front" rate - each scaled by how hard that person is to
        slip past for the player's base level (``progression.challenge_factor`` of their
        vigilance: a guard teaches more than a villager, and a novice learns more from either
        than a master) - and nothing from anyone already on to you."""
        s = self.settings.stealth
        blockers = self.sight_blockers() if blockers is None else blockers
        base = player.skills.level(SkillType.SNEAK)
        rate = 0.0
        for npc in self.npcs:
            dist = npc.position.distance_to(player.position)
            if dist > s.sneak_xp_range:
                continue
            if npc.suspicion_meter >= self.settings.perception.schedule_override_threshold:
                continue
            if npc.is_asleep:
                if not self.same_room(npc.position, player.position):
                    continue
                npc_rate = s.sneak_xp_behind
            else:
                if not npc.has_line_of_sight(player.position, blockers, NPC.chest_height(player)):
                    continue
                npc_rate = s.sneak_xp_in_front if npc.faces(player.position) else s.sneak_xp_behind
            rate += npc_rate * progression.challenge_factor(base, observer_challenge(npc, s), self.settings.progression)
        if rate > 0.0:
            return rate
        return s.sneak_xp_alone * progression.challenge_factor(base, 0.0, self.settings.progression)

    def _sneak_practice(self, player: Player, delta_time: float, blockers) -> None:
        """Sneak XP while crouch-moving (``sneak_xp_rate``); rewards shrink with level like every skill."""
        progression.grant(player, SkillType.SNEAK, self.sneak_xp_rate(player, blockers) * delta_time,
                          notify=self.notify, s=self.settings.progression)

    def _keeper_objects_to_skulking(self, npc: NPC, player: Player) -> None:
        """A merchant - or a fence - whose suspicion has been roused by somebody creeping about
        their premises tells them to stand up (the eviction in three asks), whether or not a
        sneak roll has failed yet: a watched customer cannot crouch about at leisure."""
        if not (player.is_crouching and npc.can_trade and npc.is_open_for_business and not npc.is_asleep):
            return
        if npc.state is AIState.WAITING or npc.is_alarmed or self.player_on_premises_of(npc) is None:
            return
        self.start_eviction(npc, "skulking", "shop_skulk")

    def _shopkeeper_watches(self, npc: NPC, player: Player, blockers) -> None:
        """A shopkeeper at work - and a fence in his den - keeps an eye on anyone on the premises,
        turning to follow them around, unless a successful sneak roll keeps the player out of mind."""
        if not (npc.can_trade and npc.state is AIState.SHOPKEEPING) or npc.sneak_hidden:
            return
        here, there = self.area_def_at(npc.position), self.area_def_at(player.position)
        if here is None or there is None or here.owner != npc.name or there.owner != npc.name:
            return
        if (player.position.distance_to(npc.position) <= npc.view_distance
                and npc.has_line_of_sight(player.position, blockers, NPC.chest_height(player))):
            npc.turn_towards(player.position)

    def _watch_and_follow(self, npc: NPC, player: Player, visible: bool, sneaking: bool) -> None:
        """Guards turn to watch a skulking player and tail them if they keep it up."""
        if not visible:
            return
        if npc.is_guard and npc.suspicion_meter >= self.settings.perception.schedule_override_threshold:
            npc.turn_towards(player.position)
            if sneaking and npc.state in (AIState.PATROL, AIState.GUARDING, AIState.FOLLOWING):
                if npc.state is not AIState.FOLLOWING or (npc.goal is None or npc.goal.distance_to(player.position) > 1.5):
                    npc.go_to(self, player.position, AIState.FOLLOWING)
                npc.follow_timer = 0.0

    def _guard_sees_player(self, guard: NPC, player: Player) -> None:
        """A guard with something against the player goes after them on sight - "Halt!", then the
        chase - and charges them once ``charge_delay`` has passed since the crime was declared
        *and* they are within ``charge_reach``: a thief who runs may get away (this time).
        Responding guards stop running once they have the player in view."""
        if guard.state is AIState.RESPONDING and (guard.report_to is None or guard.active_memories(self.day_count)):
            # (one on his way to take a shopkeeper's report walks on past an innocent bystander)
            guard.path = []
            guard.turn_towards(player.position)
        confronted = False
        if guard.pending_confrontations:
            self._confront(guard, player)
            confronted = True                    # "Hand it over!": the guard has them already, no chase needed
        if not guard.active_memories(self.day_count) or guard.is_asleep:
            return
        if self.jail.sealed or self.relocating:
            return
        if not confronted and (self.elapsed < guard.charge_ready_at or
                               guard.position.distance_to(player.position) > self.settings.justice.charge_reach):
            guard.pursue(self, player.position, self.notify)
            return
        outcome = justice.charge_player(self, guard, player, self.notify)
        if outcome != "nothing":
            for g in self.guards():
                if g.state in (AIState.RESPONDING, AIState.ALERT_GUARDS, AIState.FOLLOWING, AIState.INVESTIGATING):
                    g.resume_routine(self, None)

    # --- stealth checks -------------------------------------------------------------
    def _sneak_check(self, npc: NPC, player: Player, delta_time: float, ambient: float) -> None:
        """Once a second while an NPC watches the crouching player, roll Sneak: success hides the
        player from that NPC for the next second, failure makes the NPC notice."""
        s = self.settings.stealth
        npc.sneak_check_timer += delta_time
        if npc.sneak_check_timer < s.sneak_check_interval:
            return
        npc.sneak_check_timer = 0.0
        rewards = self.settings.skill_rewards
        visibility = player.get_visibility(ambient, True)
        challenge = sneak_challenge(npc, visibility, s)
        if roll(self.rng, sneak_chance(player, npc, s, visibility, self.settings.progression)):
            npc.sneak_unnoticed_until = self.elapsed + s.sneak_check_interval + 1e-6
            progression.grant(player, SkillType.SNEAK, rewards.sneak_success, notify=self.notify,
                              s=self.settings.progression, challenge=challenge)
        else:
            npc.sneak_unnoticed_until = 0.0
            npc.suspicion_meter += s.sneak_fail_suspicion
            progression.grant(player, SkillType.SNEAK, rewards.sneak_fail, notify=self.notify,
                              s=self.settings.progression, challenge=challenge)
            if npc.is_open_for_business and npc.state is not AIState.WAITING and self.player_on_premises_of(npc) is not None:
                # a shopkeeper who catches a customer creeping about the shop: stand up, or out
                self.start_eviction(npc, "skulking", "shop_skulk")
            else:
                npc.scold("skulking", "skulking_citizen" if not npc.is_guard else "skulking", self.notify, player=player)

    def _confront(self, guard: NPC, player: Player) -> None:
        """A guard who saw the player lurking where goods went missing calls them out - and takes the goods back."""
        report = guard.pending_confrontations.pop(0)
        names = ", ".join(report.names.values())
        # Only the reporter's own stolen goods are taken back, and no more of each than went
        # missing: a bottle bought and paid for stays the player's, whatever it is called.
        owed = dict(report.items)
        recovered: List[Item] = []
        for item in list(player.inventory):
            if not item.is_stolen or item.owner != report.reporter or owed.get(item.id, 0) <= 0:
                continue
            count = min(item.quantity, owed[item.id])
            owed[item.id] -= count
            unit = take_item(player.inventory, item, count)
            if unit is not None:
                recovered.append(unit)
        self.player_caught(guard)
        # his own sighting, or the shopkeeper's description passed on when he took the report
        seen = ("You! You were skulking around" if self.saw_someone(guard, report)
                else f"You! {dialogue.short_name(self.find_npc(report.reporter)) if self.find_npc(report.reporter) else report.reporter}"
                     f" says you were poking about")
        if recovered:
            for item in recovered:
                item.is_stolen = False                    # back with its owner
                add_item(self.owner_inventory_for(item), item)
            player.crimes_witnessed += 1
            guard.crimes_witnessed += 1
            where = self.find_area(report.area)
            indoors = where is not None and self.is_indoors(Vector3D((where.min_xz[0] + where.max_xz[0]) / 2.0, 0.0,
                                                                     (where.min_xz[1] + where.max_xz[1]) / 2.0))
            penalty = justice.penalty_for("Theft", indoors, self.settings.justice)
            guard.remember_crime("Theft", self.day_count, True, penalty.fine, self.rng, jail_days=penalty.jail_days,
                                 jail_days_unpaid=penalty.jail_days_unpaid, indoors=indoors, now=self.elapsed)
            guard.say(f"{seen} {report.area} when {names} went missing. Hand it over!", None)
            if self.notify:
                self.notify(f"({guard.name}): \"{seen} {report.area} when {names} went missing. "
                            f"Hand it over!\" The goods are returned to {report.reporter}.")
        else:
            guard.say(f"{seen} {report.area} before {names} went missing. I've got my eye on you.", self.notify)

    # --- curfews: a welcome guest is still shown the door at night ---------------------------
    def _tick_curfews(self, delta_time: float) -> None:
        """Evictions in progress (a late guest under curfew, a customer skulking about the shop):
        repeat the request, show them the door on the third; and start the curfew ones."""
        if self.relocating:
            return
        player = self.player
        area = self.area_def_at(player.position)
        hour = self.clock.hour
        for npc in self.npcs:
            if self.is_frozen(npc):
                continue
            if self.suppress_curfews and npc.state is AIState.WAITING and npc.waiting_for == "curfew":
                npc.stop_waiting(self, None)
                continue
            if npc.state is AIState.WAITING and npc.waiting_for in ("curfew", "skulking"):
                reason = npc.waiting_for
                offending = self._still_offending(npc, reason)
                if offending is None:
                    npc.stop_waiting(self, None)           # they left, stood up, or morning came
                    continue
                npc.curfew_ask_timer += delta_time
                if npc.curfew_ask_timer >= self.settings.justice.curfew_ask_interval:
                    npc.curfew_ask_timer = 0.0
                    npc.curfew_asks += 1
                    again, final, log_line = self._eviction_lines(reason)
                    if npc.curfew_asks >= self.settings.justice.curfew_asks_before_eviction:
                        npc.say(npc.phrase(final), self.notify)
                        self.throw_out(player, offending.name, by=npc, line=log_line)
                        npc.stop_waiting(self, None)
                    else:
                        npc.say(npc.phrase(again), self.notify)
                        if reason == "skulking":
                            npc.warning_grace_until = npc.clock + self.settings.justice.curfew_ask_interval * 2
                continue
            if area is None or not area.curfew_active(hour) or area.owner is None or area.owner not in (npc.name, npc.faction):
                continue
            if self.area_def_at(npc.position) is not area or npc.is_asleep or npc.is_alarmed or npc.state in (
                    AIState.RESPONDING, AIState.ALERT_GUARDS, AIState.WAITING):
                continue
            if npc.awake_for_business or self.suppress_curfews:
                continue                                 # roused for a deal: business first, the door afterwards
            self.start_eviction(npc, "curfew", "leave_late")

    # --- the inn: a visitor without a room is welcomed -------------------------------------------
    def inn_building(self):
        area = self.find_area(self.inn.area) if self.inn.present else None
        if area is None:
            return None
        return self.building_at(Vector3D((area.min_xz[0] + area.max_xz[0]) / 2.0, 0.0, (area.min_xz[1] + area.max_xz[1]) / 2.0))

    def _tick_inn_welcome(self, delta_time: float) -> None:
        """The inn is open to all at every hour. Somebody who walks into the hall without a room
        is welcomed by the keeper: from behind the bar if she is there, otherwise she comes over
        - woken if need be (no rest for an innkeeper) - says her piece and stays a while so a
        room can be rented; then back to what she was doing (or to bed). Once per visit."""
        inn = self.inn
        player = self.player
        if not inn.present or inn.keeper is None or player is None or self.relocating:
            return
        keeper = self.find_npc(inn.keeper)
        building = self.inn_building()
        if keeper is None or building is None:
            return
        in_the_inn = self.building_at(player.position) is building
        in_the_hall = self.area_of(player.position) == inn.area
        rented = inn.is_rented(self.day_count, self.clock.hour)
        if inn.welcome is not None:
            over = (not in_the_inn or rented or keeper.is_alarmed or keeper.state is not AIState.WELCOMING
                    or inn.welcome_timer > inn.WELCOME_PATIENCE)
            if over:
                self._end_welcome(keeper)
                return
            inn.welcome_timer += delta_time
            if inn.welcome == "going":
                if not keeper.path:
                    keeper.turn_towards(player.position)
                    keeper.say(keeper.phrase("inn_welcome_woken" if inn.welcome_woken else "inn_welcome"), self.notify)
                    inn.welcome = "greeting"
            else:
                keeper.turn_towards(player.position)
                if (not inn.welcome_from_post and in_the_hall
                        and keeper.position.distance_to(player.position) > self.CHAT_RANGE + 1.5):
                    keeper.go_to(self, player.position, AIState.WELCOMING)       # the visitor wandered off: follow
                    inn.welcome = "going"
            return
        if not in_the_inn:
            inn.welcomed_this_visit = False                  # a new visit is a new welcome
            return
        if not in_the_hall or rented or inn.welcomed_this_visit:
            return
        if keeper.is_alarmed or keeper.state in (AIState.RESPONDING, AIState.ALERT_GUARDS, AIState.WAITING,
                                                  AIState.INVESTIGATING, AIState.FOLLOWING) or keeper.conversation is not None:
            return
        inn.welcomed_this_visit = True
        inn.welcome_timer = 0.0
        inn.welcome_woken = keeper.is_asleep
        if keeper.is_asleep:
            keeper.awake_for_business = True                 # roused: ``back_to_business_as_usual`` puts her back to bed
        keeper.path = []
        keeper.patrol_wait = 0.0
        keeper.end_conversation()
        at_her_post = keeper.state is AIState.SHOPKEEPING
        near = keeper.position.distance_to(player.position) <= self.CHAT_RANGE
        keeper.state = AIState.WELCOMING
        inn.welcome_from_post = at_her_post
        inn.welcome = "going"
        if not (at_her_post or near):
            keeper.go_to(self, player.position, AIState.WELCOMING)   # else: greeted from where she stands, next tick

    def _end_welcome(self, keeper: NPC) -> None:
        self.inn.welcome = None
        self.inn.welcome_timer = 0.0
        if keeper.state is AIState.WELCOMING:
            if keeper.awake_for_business:
                keeper.back_to_business_as_usual(self, self.notify)   # back to bed
            else:
                keeper.resume_routine(self, self.notify)

    # --- talk: conversations and ambient chatter ------------------------------------------
    CHAT_RANGE = 4.5              # how close two people must be to strike up a conversation (across a table)
    MAX_CONVERSATIONS = 2
    CHAT_CHECK_INTERVAL = 1.0

    def _tick_talk(self, delta_time: float) -> None:
        """Play out running conversations a line at a time and, now and then, start new ones
        between people who have stopped near each other."""
        for conv in list(self.conversations):
            self._advance_conversation(conv)
        self._chat_clock += delta_time
        if self._chat_clock < self.CHAT_CHECK_INTERVAL:
            return
        self._chat_clock = 0.0
        if len(self.conversations) >= self.MAX_CONVERSATIONS:
            return
        free = [n for n in self.npcs if n.conversation is None and n.is_sociable and n.next_chat_time <= self.elapsed
                and not self.is_frozen(n)]
        for i, a in enumerate(free):
            for b in free[i + 1:]:
                if a.conversation is not None or b.conversation is not None:
                    continue
                if a.position.distance_to(b.position) > self.CHAT_RANGE or not self.same_room(a.position, b.position):
                    continue
                chance = 0.35
                if a.is_guard or b.is_guard:
                    chance = 0.12                                  # the watch has less time for gossip
                if self.player is not None and self._chat_near_player() and \
                        min(a.position.distance_to(self.player.position), b.position.distance_to(self.player.position)) < 16.0:
                    continue                                       # one conversation within earshot at a time
                if not roll(self.rng, chance):
                    continue
                self.start_conversation(a, b)
                if len(self.conversations) >= self.MAX_CONVERSATIONS:
                    return

    def _chat_near_player(self) -> bool:
        for conv in self.conversations:
            if any(n.position.distance_to(self.player.position) < 16.0 for n in conv.participants):
                return True
        return False

    def start_conversation(self, a: NPC, b: NPC, exchanges: Optional[int] = None) -> dialogue.Conversation:
        conv = dialogue.build_conversation(a, b, self.clock.hour, self.rng, exchanges, recent=self.recent_topics)
        for speaker, text in conv.lines[2:-2]:
            self.recent_topics.append(text)
        del self.recent_topics[:-12]
        return self._begin_conversation(conv)

    def _begin_conversation(self, conv: dialogue.Conversation) -> dialogue.Conversation:
        """Both parties stop, turn to face each other and the lines start after a beat."""
        conv.next_line_at = self.elapsed + 0.8
        for npc, other in ((conv.a, conv.b), (conv.b, conv.a)):
            npc.conversation = conv
            npc.chat_pause = True
            npc.turn_towards(other.position)
        self.conversations.append(conv)
        return conv

    def _conversation_broken(self, conv: dialogue.Conversation) -> bool:
        if conv.official:
            return any(n.is_asleep or n.is_alarmed for n in conv.participants)
        return any(not n.is_sociable for n in conv.participants)       # somebody has better things to do

    def _advance_conversation(self, conv) -> None:
        if not conv.finished and self._conversation_broken(conv):
            conv.finished = True
        if not conv.finished and self.elapsed >= conv.next_line_at:
            if conv.index >= len(conv.lines):
                conv.finished = True
            else:
                speaker, text = conv.lines[conv.index]
                speaker.say(text, self.notify if conv.logged else None)   # gossip is a bubble only, never logged
                conv.index += 1
                conv.next_line_at = self.elapsed + 2.8 + 0.04 * len(text)
        if conv.finished:
            for npc in conv.participants:
                if npc.conversation is conv:
                    npc.conversation = None
                npc.chat_pause = False
                npc.next_chat_time = self.elapsed + self.rng.uniform(90.0, 200.0)
            if conv in self.conversations:
                self.conversations.remove(conv)
            if conv.on_finish is not None:
                callback, conv.on_finish = conv.on_finish, None
                callback()

    AMBIENT_LINES = {AIState.SHOPPING: "shopping", AIState.DRINKING: "drinking", AIState.CLEANING: "cleaning"}
    WORKING_LINES = {"smith": "working_smith", "cook": "working_kitchen", "innkeeper": "working_kitchen",
                     "porter": "working_porter"}

    def _ambient_chatter(self, npc: NPC, delta_time: float) -> None:
        """A word now and then from people going about their business: a customer haggling (and
        paying the merchant a few coins), a drinker calling for another, the smith's hammer."""
        if npc.conversation is not None or npc.is_asleep:
            return
        kind = self.AMBIENT_LINES.get(npc.state)
        if npc.state is AIState.WORKING:
            kind = self.WORKING_LINES.get(npc.role_tag)
        if npc.role_tag == "rich" and npc.state in (AIState.SHOPPING, AIState.STROLLING, AIState.DRINKING):
            kind = "rich_ambient"                                      # sneering at everything they pass
        if kind is None:
            return
        npc.chatter_timer -= delta_time
        if npc.chatter_timer > 0.0:
            return
        npc.chatter_timer = self.rng.uniform(35.0, 80.0)
        npc.say(npc.phrase(kind))                                      # bubble only
        if npc.state is AIState.SHOPPING:
            merchant = next((m for m in self.npcs if m.can_trade and m is not npc and m.is_open_for_business
                             and m.position.distance_to(npc.position) <= 6.0), None)
            if merchant is not None and npc.gold >= 2.0:
                spent = min(npc.gold, float(self.rng.randint(2, 6)))
                npc.gold -= spent                                  # a simulated purchase: coin changes hands
                merchant.gold += spent

    # --- shopkeeping -------------------------------------------------------------------
    def on_task_arrived(self, npc: NPC, task) -> None:
        if npc.is_shopkeeper and task.target_state is AIState.SHOPKEEPING:
            self._open_shop(npc)

    def on_task_left(self, npc: NPC, previous, next_task) -> None:
        """A shopkeeper leaving the counter for somewhere *off the premises* closes up: stock is
        counted and the shop cleared. Moving about inside the building the counter stands in -
        Barnaby's lunch in his back room, Mirelle's at the side table of her own shop, the
        innkeeper wiping the tables - is not leaving, whatever the areas are called."""
        if not (npc.is_shopkeeper and previous.target_state is AIState.SHOPKEEPING):
            return
        counter, destination = previous.target_pos, next_task.target_pos
        building = self.building_at(counter)
        if self.area_of(destination) == self.area_of(counter) or (
                building is not None and self.building_at(destination) is building):
            return
        self._take_stock(npc, "closing up")
        self._clear_the_premises(npc, counter)

    # --- the shopkeeper's premises: closing time and skulking customers ----------------------
    def player_on_premises_of(self, npc: NPC, where: Optional[Vector3D] = None) -> Optional[AreaDef]:
        """The indoor area of ``npc``'s own shop (or a fence's den) the player is standing in -
        with the keeper there too (``where``: the keeper's spot; defaults to where they stand) -
        else None."""
        if self.player is None or not npc.can_trade:
            return None
        here = self.area_def_at(where if where is not None else npc.position)
        there = self.area_def_at(self.player.position)
        if here is None or there is None or here.owner != npc.name or there.owner != npc.name:
            return None
        building = self.building_at(self.player.position)
        if building is None or building is not self.building_at(where if where is not None else npc.position):
            return None
        return there

    def _clear_the_premises(self, npc: NPC, counter: Vector3D) -> None:
        """Closing time: a keeper leaving the shop looks round for customers. A player standing
        there is shown the door; one crouching out of the way gets a Sneak roll - success and
        the keeper locks up none the wiser (the player is trespassing from then on), failure
        and they are found and shown out like anybody else. A paying inn guest is left be."""
        player = self.player
        area = self.player_on_premises_of(npc, counter)
        if area is None or self.relocating or not area.restricted:
            return                                           # a public house (the inn's hall) is never cleared
        if self.area_trusts_player(area) or (npc.name == self.inn.keeper and self.inn.is_rented(self.day_count, self.clock.hour)):
            return
        s = self.settings.stealth
        if player.is_crouching:
            visibility = player.get_visibility(self.ambient_light, False)
            hidden = roll(self.rng, sneak_chance(player, npc, s, visibility, self.settings.progression))
            progression.grant(player, SkillType.SNEAK, self.settings.skill_rewards.sneak_success if hidden else
                              self.settings.skill_rewards.sneak_fail, notify=self.notify, s=self.settings.progression,
                              challenge=sneak_challenge(npc, visibility, s))
            if hidden:
                if self.notify:
                    self.notify(f"{dialogue.short_name(npc)} locks up without noticing you.")
                return
        npc.turn_towards(player.position)
        npc.say(npc.phrase("shop_closing"), self.notify)
        self.throw_out(player, area.name, by=npc, line=f"You are shown out of {area.name}.")

    def _eviction_lines(self, reason: str):
        """(again, final, log line) for an eviction in progress."""
        if reason == "skulking":
            return "shop_skulk_again", "shop_out", "You are thrown out of the shop."
        return "leave_again", "leave_now", "You are shown the door."

    def _still_offending(self, npc: NPC, reason: str) -> Optional[AreaDef]:
        """The area the player is still in the wrong in (None: they complied or left)."""
        player = self.player
        if reason == "skulking":
            return self.player_on_premises_of(npc) if player.is_crouching else None
        area = self.area_def_at(player.position)
        if area is None or area is not self.area_def_at(npc.position) or not area.curfew_active(self.clock.hour):
            return None
        return area

    def start_eviction(self, npc: NPC, reason: str, first_line: str) -> None:
        """The resident/keeper stops what they are doing, faces the player and asks them to
        comply (leave / stand up); ``_tick_evictions`` repeats the request every
        ``curfew_ask_interval`` and shows them the door on the third."""
        npc.wait_for(reason, self.player.position)
        npc.curfew_asks = 1
        npc.curfew_ask_timer = 0.0
        npc.say(npc.phrase(first_line), self.notify)
        if reason == "skulking":     # the keeper's own response is the eviction: no alarm meanwhile
            npc.warning_grace_until = npc.clock + self.settings.justice.curfew_ask_interval * 3

    def _open_shop(self, npc: NPC) -> None:
        self._take_stock(npc, "opening up")
        if commerce.restock_due(npc, self):
            added = commerce.restock(npc, self, self.player_level)
            if added and self.notify:
                self.notify(f"({npc.name}): \"Fresh stock in! {', '.join(i.display_name for i in added[:4])}"
                            f"{'...' if len(added) > 4 else ''}\"")

    def _take_stock(self, npc: NPC, when: str) -> None:
        report = commerce.take_stock(npc, self)
        if report is None:
            return
        self.report_theft(report, when)

    def report_theft(self, report: TheftReport, when: str = "") -> None:
        """A shopkeeper finds goods missing and shouts for the guards. Every guard within
        earshot comes running and takes the report from the shopkeeper face to face
        (``_go_and_take_report``); one who saw the player lurking there earlier - or is told
        the shopkeeper did - will confront them."""
        self.theft_reports.append(report)
        missing = self.missing_goods_text(report)
        reporter = self.find_npc(report.reporter)
        if reporter is not None:
            reporter.say(f"Thieves! {missing} missing from {report.area}! Guards!", self.notify,
                         log_prefix=f"{report.reporter}, {when}")
        elif self.notify:
            self.notify(f"({report.reporter}, {when}): \"Thieves! {missing} missing from {report.area}! Guards!\"")
        for guard in self.guards():
            guard.learn_theft(report)
        if reporter is not None and not self._dispatch_report(report, reporter):
            self.unanswered_reports.append((report, self.elapsed))     # every guard in earshot is busy: wait for one

    REPORT_WAIT = 600.0               # seconds an unanswered report waits for a free guard before it is let go

    def _dispatch_report(self, report: TheftReport, reporter: NPC) -> bool:
        """Send every guard in earshot who is free to take it - never one already taking (or
        owing) another shopkeeper's report, nor one after a thief. Returns True if anyone went."""
        sent = False
        for guard in self.guards():
            if guard.on_a_report or guard.is_alarmed or self.is_frozen(guard) or not self.can_hear_cry(guard, reporter.position):
                continue
            self._send_guard(guard, reporter.position)
            if guard.state is AIState.RESPONDING:
                guard.report_to = report
                guard.report_timer = 0.0
                sent = True
        return sent

    def _dispatch_unanswered_reports(self) -> None:
        """Reports nobody was free for get the next free guard; stale ones are let go."""
        still_waiting = []
        for report, made_at in self.unanswered_reports:
            reporter = self.find_npc(report.reporter)
            if reporter is None or reporter.is_asleep or self.elapsed - made_at > self.REPORT_WAIT:
                continue
            if not self._dispatch_report(report, reporter):
                still_waiting.append((report, made_at))
        self.unanswered_reports = still_waiting

    def resume_report(self, guard: NPC, notify: Notify = None) -> bool:
        """A guard whose report-taking a crime cut short goes back to the shopkeeper to finish
        it (with an apology). False if there is nobody to go back to."""
        report = guard.report_interrupted
        guard.report_interrupted = None
        reporter = self.find_npc(report.reporter) if report is not None else None
        if report is None or reporter is None or reporter.is_asleep:
            guard.report_apologise = False
            return False
        guard.goal = None
        guard.search_timer = 0.0
        guard.follow_timer = 0.0
        guard.end_conversation()
        guard.state = AIState.GUARDING
        guard.go_to(self, reporter.position, AIState.RESPONDING)
        guard.report_to = report
        guard.report_timer = 0.0
        return True

    @staticmethod
    def missing_goods_text(report: TheftReport) -> str:
        return ", ".join(f"{name} x{report.items[iid]}" for iid, name in report.names.items())

    @staticmethod
    def saw_someone(npc: NPC, report: TheftReport) -> bool:
        """Did this person notice somebody skulking about the area the goods went from lately?"""
        return any(s.area == report.area and report.day - s.day <= 2 for s in npc.sightings)

    REPORT_PATIENCE = 90.0            # seconds a guard keeps trying to reach a shopkeeper to take the report

    def _go_and_take_report(self, guard: NPC, delta_time: float) -> None:
        """A responding guard makes for the shopkeeper who raised the alarm (following if they
        have moved on) and, once beside them, takes the report in a face-to-face exchange;
        both go back to what they were doing afterwards."""
        report = guard.report_to
        reporter = self.find_npc(report.reporter)
        guard.report_timer += delta_time
        if reporter is None or reporter.is_asleep or guard.report_timer > self.REPORT_PATIENCE:
            guard.resume_routine(self, self.notify)
            return
        if guard.position.distance_to(reporter.position) <= self.CHAT_RANGE and self.same_room(guard.position, reporter.position):
            guard.path = []
            if reporter.is_alarmed:
                return                                   # still shouting: let them calm down, then take it
            self._take_report(guard, reporter, report)
        elif not guard.path or (guard.goal is not None and guard.goal.distance_to(reporter.position) > 2.0):
            guard.go_to(self, reporter.position, AIState.RESPONDING)      # the shopkeeper has moved on: follow

    def _take_report(self, guard: NPC, reporter: NPC, report: TheftReport) -> None:
        guard.report_to = None
        guard.report_in_progress = report
        guard.state = AIState.GUARDING
        reporter.end_conversation()
        guard_saw, merchant_saw = self.saw_someone(guard, report), self.saw_someone(reporter, report)
        conv = dialogue.build_report_exchange(guard, reporter, self.missing_goods_text(report), report.area,
                                              guard_saw, merchant_saw, self.rng, apologise=guard.report_apologise)
        guard.report_apologise = False

        def done() -> None:
            if conv.index < len(conv.lines):
                # broken off: by a crime in front of the guard (he comes back to it, see
                # ``raise_alarm`` / ``resume_report``) or by the shopkeeper dropping off
                if guard.report_interrupted is None:
                    guard.report_in_progress = None
                    guard.resume_routine(self, self.notify)
                return
            guard.report_in_progress = None
            if merchant_saw and report not in guard.pending_confrontations:
                guard.pending_confrontations.append(report)        # the shopkeeper's lead is his now
            guard.resume_routine(self, self.notify)

        conv.on_finish = done
        self._begin_conversation(conv)
