"""
Firestarter - WorldManager Subsystem
Implements DESIGN Section 8: coordinates terrain, weather, ecosystem, the thermodynamic fire
grid, bushcraft crafting, the player, the camp (fire pit, drying rack, woodpile, bow-drill
kit) and every interaction (raycast targeting, pickups, fuel placement with holographic
blueprint snapping, tools, breath, purchases and time skips) on fixed 60 Hz physics and
20 Hz thermal ticks.
"""

import math
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from firestarter.config import PHYSICS_DT, THERMAL_DT
from firestarter.thermodynamics.combustion_body import CombustionBody, CombustionStage, orientation_to_axis
from firestarter.thermodynamics.wood_properties import WoodSpeciesType
from firestarter.thermodynamics.thermal_grid import ThermodynamicGrid
from firestarter.bushcraft.crafting_controller import BushcraftCraftingController
from firestarter.bushcraft.fungi import SmolderingFungus, FungusType
from firestarter.forestry.tree import Tree, TreeGrowthStage
from firestarter.forestry.felling import TreeFellingSystem
from firestarter.forestry.bucking import FallenTree, Limb, SECTION_LEN
from firestarter.forestry.seasoning import RaisedDryingRack, GroundSeasoningSimulator, step_moisture
from firestarter.forestry.ecosystem import ForestEcosystem
from firestarter.simulation.weather import WeatherSystem, Season
from firestarter.simulation.time_engine import TimeEngine
from firestarter.simulation.player import Player
from firestarter.simulation.inventory import (
    Inventory, FuelItem, FUEL_ORDER, make_tinder, make_kindling, make_firewood, make_log
)
from firestarter.simulation import terrain
from firestarter.recipes.blueprints import ALL_BLUEPRINTS, match_structure, KIND_COMPAT

REACH = 3.0
LIGHTER_REACH = 2.4
PIT_INNER_RADIUS = 0.42
TOOLS = ["pocket_lighter", "pocket_knife", "felling_axe", "bow_drill_spindle", "accelerant"]
TOOL_LABELS = {"pocket_lighter": "Lighter", "pocket_knife": "Knife", "felling_axe": "Axe",
               "bow_drill_spindle": "Bow Drill", "accelerant": "Gasoline"}
OFFHAND = ["tinder", "kindling", "firewood", "log", "fungus"]


class _PileStub:
    """Duck-typed stand-in so the seasoning model can dry a pile of twigs or needles."""

    def __init__(self, m: float, radius: float):
        self.m_surface = m
        self.m_core = m
        self.radius = radius

    def set_moisture(self, surface: float = None, core: float = None):
        if surface is not None:
            self.m_surface = surface
        if core is not None:
            self.m_core = core


class WorldItem:
    """Something in the world the player can interact with (loose fuel, resource piles...)."""
    _ids = 0

    def __init__(self, kind: str, position: np.ndarray, radius: float = 0.35, label: str = "",
                 fuel: Optional[FuelItem] = None, count: int = 1, material: str = "", yaw: float = 0.0,
                 species: Optional[WoodSpeciesType] = None):
        WorldItem._ids += 1
        self.id = WorldItem._ids
        self.kind = kind            # fuel, tinder_source, kindling_pile, woodpile, stone
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.label = label
        self.fuel = fuel
        self.count = count
        self.max_count = count
        self.material = material
        self.yaw = yaw
        self.species = species
        self.on_ground = True
        self.decay_days = 0.0
        self.regrow_days = 0.0

    # duck-typing for the seasoning simulator
    @property
    def m_surface(self):
        return self.fuel.m_surface if self.fuel else 0.0

    @property
    def m_core(self):
        return self.fuel.m_core if self.fuel else 0.0


class WorldManager:
    def __init__(self, seed: int = 7, start_hour: float = 18.5, tutorial_camp: bool = True):
        self.rng = np.random.default_rng(seed)
        self.weather = WeatherSystem(seed=seed, start_hour=start_hour)
        self.time_engine = TimeEngine()
        self.ecosystem = ForestEcosystem(seed=42)
        self.fire_pit_pos = np.array([0.0, float(terrain.height(0.0, 0.0)), 0.0])
        self.thermal_grid = ThermodynamicGrid(pit_center=self.fire_pit_pos.copy(), seed=seed)
        self.thermal_grid.structure_detector = self._detect_structure
        self.inventory = Inventory()
        self.crafting = BushcraftCraftingController(self.inventory)
        self.player = Player()
        self.player.look_at(self.fire_pit_pos + np.array([0.0, 0.1, 0.0]))
        self.reference_book = None      # linked by the game (for discoveries / pinned guides)
        self.credits = None             # linked by the game (CreditSystem)
        self.clock_scale = 1.0
        self.sim_time = 0.0

        self.drying_rack = RaisedDryingRack(position=np.array([3.4, float(terrain.height(3.4, 1.6)), 1.6]), has_tarp=False)
        self.hearthboard_pos = np.array([1.05, float(terrain.height(1.05, 0.85)) + 0.01, 0.85])
        self.woodpile_pos = np.array([-2.5, float(terrain.height(-2.5, 1.4)), 1.4])
        self.world_items: List[WorldItem] = []
        self.ground_logs: List[WorldItem] = []
        self.fallen_trees: List[FallenTree] = []

        self.physics_accumulator = 0.0
        self.thermal_accumulator = 0.0
        self._settle_timer = 0.0

        # interaction state
        self.target: Optional[Dict[str, Any]] = None
        self.ghost: Optional[Dict[str, Any]] = None
        self.offhand_index = 0
        self.offhand_empty = False
        self.primary_held = False
        self.secondary_held = False
        self.breath_start: Optional[float] = None
        self.breath_charge = 0.0
        self.last_breath_velocity = 0.0
        self.bow_input_speed = 0.0
        self.bow_active = False           # friction drilling in progress (bow or hand drill)
        self.friction_mode = "bow"        # "bow" drill or "hand" drill (press 4 again to switch)
        self.lighter_point: Optional[np.ndarray] = None

        # notifications / events for UI, audio and particles
        self.notification = "Welcome to Firestarter. Gather tinder and kindling, then build your first fire."
        self.notification_timer = 8.0
        self.toasts: List[Dict[str, Any]] = []
        self.events: List[Tuple[str, Any]] = []
        self.last_skip_report: Optional[Dict[str, Any]] = None
        self.pending_time_skip: Optional[float] = None
        self._pending_skip_frames = 0
        self.last_time_skip_result: Optional[Dict[str, Any]] = None

        # field log statistics
        self.stats = {"fires_lit": 0, "best_temp": 0.0, "energy_mj": 0.0, "fuel_kg": 0.0,
                      "trees_felled": 0, "embers_made": 0, "longest_fire_min": 0.0,
                      "pieces_burned": 0, "days_elapsed": 0.0}
        self._fire_was_lit = False
        self._fire_start = 0.0
        self.gas_flash_timer = 0.0
        self.structure_key = "freeform"
        self.structure_score = 0.0

        self._update_colliders()
        if tutorial_camp:
            self._setup_initial_camp()

    # ------------------------------------------------------------------ setup
    def _setup_initial_camp(self):
        rng = self.rng
        g = lambda x, z: float(terrain.height(x, z))
        # tinder sources near camp
        self.world_items.append(WorldItem("tinder_source", [1.7, g(1.7, -1.3) + 0.05, -1.3], 0.35,
                                          "Birch bark curls", count=3, material="birch_bark"))
        self.world_items.append(WorldItem("tinder_source", [-2.0, g(-2.0, -1.7) + 0.04, -1.7], 0.4,
                                          "Dry pine needles", count=3, material="pine_needles"))
        self.world_items.append(WorldItem("tinder_source", [2.9, g(2.9, -2.8) + 0.05, -2.8], 0.4,
                                          "Dry grass tuft", count=2, material="dry_grass"))
        self.world_items.append(WorldItem("tinder_source", [-1.6, g(-1.6, -4.6) + 0.05, -4.6], 0.4,
                                          "Cattail fluff", count=2, material="cattail_fluff"))
        self.world_items.append(WorldItem("kindling_pile", [1.9, g(1.9, 1.3) + 0.06, 1.3], 0.45,
                                          "Bundle of dead pine twigs", count=12, species=WoodSpeciesType.WHITE_PINE))
        self.world_items.append(WorldItem("kindling_pile", [-3.4, g(-3.4, -0.8) + 0.06, -0.8], 0.45,
                                          "Dead cedar twigs", count=8, species=WoodSpeciesType.RED_CEDAR))
        wp = WorldItem("woodpile", self.woodpile_pos + np.array([0, 0.25, 0]), 0.7, "Woodpile (seasoned)", count=10)
        wp.stock = ([make_firewood(WoodSpeciesType.WHITE_PINE, 0.11, 0.14, "woodpile", rng) for _ in range(4)]
                    + [make_firewood(WoodSpeciesType.PAPER_BIRCH, 0.12, 0.15, "woodpile", rng) for _ in range(3)]
                    + [make_log(WoodSpeciesType.WHITE_OAK, 0.13, 0.17, "woodpile", rng) for _ in range(3)])
        self.world_items.append(wp)
        # green logs on the drying rack and one on bare soil (seasoning demonstration)
        for _ in range(2):
            self.drying_rack.add_log(make_log(WoodSpeciesType.WHITE_OAK, 0.80, 1.10, "felled", rng))
        gl = WorldItem("fuel", [-4.0, g(-4.0, 2.2) + 0.07, 2.2], 0.5, "Green oak log",
                       fuel=make_log(WoodSpeciesType.WHITE_OAK, 0.80, 1.10, "felled", rng), yaw=25.0)
        self.world_items.append(gl)
        self.ground_logs.append(gl)
        # a loose damp branch and a punk-wood chunk for experimentation
        self.world_items.append(WorldItem("fuel", [5.2, g(5.2, -1.0) + 0.03, -1.0], 0.4, "Damp fallen branch",
                                          fuel=make_firewood(WoodSpeciesType.PAPER_BIRCH, 0.55, 0.45, "forest", rng), yaw=70.0))
        punk = FuelItem("firewood", WoodSpeciesType.PUNK_WOOD, 0.09, 0.12, length=0.3, radius=0.05, source="forest", is_punk=True)
        self.world_items.append(WorldItem("fuel", [-6.2, g(-6.2, 4.6) + 0.05, 4.6], 0.4, "Dry punk wood chunk", fuel=punk, yaw=10.0))

    LEAF_PILES = ((3.6, -3.2), (-4.4, -2.6), (0.8, 4.6))

    def _update_seasonal_items(self):
        """DESIGN 7.1: in autumn, drifts of fallen dry leaves are free (temporary) tinder."""
        autumn = self.weather.current_season.value == "Autumn"
        piles = [w for w in self.world_items if w.kind == "tinder_source" and w.material == "dry_leaves"]
        if autumn and not piles:
            for x, z in self.LEAF_PILES:
                self.world_items.append(WorldItem("tinder_source", [x, float(terrain.height(x, z)) + 0.04, z], 0.45,
                                                  "Fallen dry leaves", count=3, material="dry_leaves"))
        elif not autumn and piles:
            for w in piles:
                self.world_items.remove(w)

    def _update_colliders(self):
        cols = [(t.position[0], t.position[2], t.collision_radius) for t in self.ecosystem.trees
                if t.collision_radius > 0 and abs(t.position[0]) < 70 and abs(t.position[2]) < 70]
        cols.append((self.fire_pit_pos[0], self.fire_pit_pos[2], 0.45))
        cols.append((self.drying_rack.position[0], self.drying_rack.position[2], 0.5))
        cols.append((self.woodpile_pos[0], self.woodpile_pos[2], 0.55))
        self.player.colliders = cols

    # ------------------------------------------------------------------ messaging
    def notify(self, msg: str, duration: float = 4.0, kind: str = "info"):
        self.notification = msg
        self.notification_timer = duration
        self.toasts.append({"text": msg, "time": duration, "kind": kind})
        self.toasts = self.toasts[-4:]

    def emit(self, name: str, data: Any = None):
        self.events.append((name, data))

    def pop_events(self) -> List[Tuple[str, Any]]:
        ev = self.events
        self.events = []
        return ev

    # ------------------------------------------------------------------ structure recognition
    def _detect_structure(self, bodies) -> Tuple[str, float]:
        key, score, _ = match_structure(bodies, self.fire_pit_pos)
        self.structure_key, self.structure_score = key, score
        if key != "freeform" and self.reference_book is not None and score >= 0.75:
            if self.reference_book.discover(key):
                self.notify(f"Discovered the {ALL_BLUEPRINTS[key].name}! Added to your Field Compendium.", 6.0, "success")
        return key, score

    # ------------------------------------------------------------------ main update
    def request_time_skip(self) -> float:
        """Queue the time-menu selection; it is applied on the next frame so the player sees a
        'time passes' card while the fire, racks and forest are simulated (can take seconds)."""
        te = self.time_engine
        self.pending_time_skip = te.selected.hours
        self._pending_skip_frames = 0
        te.is_menu_open = False
        return self.pending_time_skip

    def update(self, frame_dt: float):
        """Advances physics (60 Hz) and the thermal grid (20 Hz) with fixed time steps."""
        if self.pending_time_skip is not None:
            if self._pending_skip_frames >= 1:
                self.pending_time_skip = None
                res = self.time_engine.execute_selected_skip(self.apply_time_skip)
                self.last_time_skip_result = res
            else:
                self._pending_skip_frames += 1
        self.sim_time += frame_dt
        if self.notification_timer > 0.0:
            self.notification_timer = max(0.0, self.notification_timer - frame_dt)
            if self.notification_timer <= 0.0:
                self.notification = ""
        for t in self.toasts:
            t["time"] -= frame_dt
        self.toasts = [t for t in self.toasts if t["time"] > 0]
        self.weather.update(frame_dt, self.clock_scale)
        self._update_seasonal_items()

        self.physics_accumulator += min(0.1, frame_dt)
        while self.physics_accumulator >= PHYSICS_DT:
            self._update_physics(PHYSICS_DT)
            self.physics_accumulator -= PHYSICS_DT

        self.thermal_accumulator += min(0.1, frame_dt)
        while self.thermal_accumulator >= THERMAL_DT:
            self._update_thermodynamics(THERMAL_DT)
            self.thermal_accumulator -= THERMAL_DT

        self._update_interaction(frame_dt)

    def _update_physics(self, dt: float):
        for tree in self.ecosystem.trees:
            if tree.chop_shake > 0:
                tree.chop_shake = max(0.0, tree.chop_shake - dt * 3.0)
            if tree.is_falling:
                hit, pieces = TreeFellingSystem.update_falling_tree(tree, dt, self.rng)
                if hit:
                    self.stats["trees_felled"] += 1
                    self.emit("tree_fall", tree.position + np.array([math.cos(tree.fall_direction), 0, math.sin(tree.fall_direction)]) * tree.height_m * 0.5)
                    sap = 1.12 if self.weather.current_season.value == "Spring" else 1.0   # spring sap flow
                    ft = FallenTree(tree, self.rng, sap_factor=sap)
                    ft.settle(terrain.height)
                    self.fallen_trees.append(ft)
                    tot = ft.totals()
                    self.notify(f"The {tree.species.value.lower()} crashes down! Limb it and buck it into rounds with the axe: "
                                f"~{tot['logs']} logs, {tot['fire_logs']} fire logs, {tot['kindling']} kindling sticks, "
                                f"{tot['tinder']} handfuls of tinder.", 8.0)
                    self.ecosystem.version += 1
                    self._update_colliders()
        self._settle_timer -= dt
        if self._settle_timer <= 0.0:
            self._settle_timer = 0.1
            self._settle_bodies(0.1)
        self.crafting.update(dt, self.weather.ambient_temp_c)
        self.player.stamina = min(self.player.max_stamina, self.player.stamina)
        if self.gas_flash_timer > 0:
            self.gas_flash_timer = max(0.0, self.gas_flash_timer - dt)

    def _update_thermodynamics(self, dt: float):
        self.thermal_grid.update(dt, weather=self.weather, sheltered=False)
        g = self.thermal_grid
        if g.coal_bed.ash_kg > 0.4:
            self.ecosystem.add_campfire_ash(self.fire_pit_pos, g.coal_bed.ash_kg - 0.4)
            g.coal_bed.ash_kg = 0.4
        # collapse events -> sparks / sound
        while g.collapse_positions:
            self.emit("collapse", g.collapse_positions.pop())
            self.stats["pieces_burned"] += 1
        # field-log statistics
        lit = g.flame_hrr > 400.0
        if lit and not self._fire_was_lit:
            self.stats["fires_lit"] += 1
            self._fire_start = self.sim_time
            self.emit("fire_lit", g.fire_centroid.copy())
        if g.is_burning:
            self.stats["longest_fire_min"] = max(self.stats["longest_fire_min"], (self.sim_time - self._fire_start) / 60.0)
        self._fire_was_lit = lit or (self._fire_was_lit and g.is_burning)
        self.stats["best_temp"] = max(self.stats["best_temp"], g.core_coal_temp if g.is_burning else 0.0)
        self.stats["energy_mj"] = g.total_energy_produced_joules / 1e6
        self.stats["fuel_kg"] = g.total_fuel_consumed_kg
        # flaming nest in hand -> tuck it under the kindling
        nest = self.crafting.active_tinder_nest
        if nest is not None and nest.is_flaming:
            self._place_flaming_nest()

    # ------------------------------------------------------------------ settling physics
    def _settle_bodies(self, dt: float):
        bodies = [b for b in self.thermal_grid.bodies]
        if not bodies:
            return
        floor = self.fire_pit_pos[1]
        moved = False
        for b in bodies:
            if b.geometry_type == "tinder":
                continue
            r = max(0.004, float(np.max(b.segment_radii())) if b.n_segments else b.initial_radius)
            tilt = float(b.orientation[0])
            others = [o for o in bodies if o is not b and not o.is_consumed]
            if tilt > 12.0 and tilt < 89.0:
                # leaning piece: its top end must rest on something
                a, c = b.endpoints()
                top = c if c[1] >= a[1] else a
                bot = a if c[1] >= a[1] else c
                supported = top[1] - r <= floor + 0.02
                # teepee apex: sticks pushed into the soil lean together over the tinder and
                # hold each other up as long as something sits under / around the apex
                apex = np.hypot(top[0] - self.fire_pit_pos[0], top[2] - self.fire_pit_pos[2]) < 0.09
                grounded = bot[1] - r <= float(terrain.height(bot[0], bot[2])) + 0.03
                if apex and tilt >= 40.0 and grounded and b.burned_fraction < 0.45:
                    # ends pushed into the soil hold a lean on their own until the base burns through
                    supported = True
                if apex and tilt >= 40.0 and grounded and not supported:
                    for o in others:
                        if o.geometry_type == "tinder" or float(o.orientation[0]) >= 35.0:
                            oa, oc = o.endpoints()
                            otop = oc if oc[1] >= oa[1] else oa
                            if np.hypot(otop[0] - top[0], otop[2] - top[2]) < 0.16:
                                supported = True
                                break
                for o in others:
                    P = o.segment_positions()
                    R = o.segment_radii()
                    d = np.linalg.norm(P - top[None, :], axis=1) - R - r
                    if np.any(d < 0.05):
                        supported = True
                        break
                if not supported:
                    bottom = a if a[1] < c[1] else c
                    new_tilt = max(0.0, tilt - 90.0 * dt * (1.0 + b.burned_fraction))
                    b.orientation[0] = new_tilt
                    ax = orientation_to_axis(b.orientation)
                    if np.dot(ax, top - bottom) < 0:
                        ax = -ax
                    b.position = bottom + ax * b.length * 0.5
                    b.position[1] = max(b.position[1], floor + r)
                    moved = True
                    if new_tilt < 12.0:
                        b.orientation[0] = 0.0
                        self.emit("collapse", b.position.copy())
                continue
            # flat / upright piece: drop onto whatever lies beneath
            P_b = b.segment_positions()
            support = float(np.max(terrain.height(P_b[:, 0], P_b[:, 2]))) + r
            for o in others:
                P = o.segment_positions()
                R = o.segment_radii()
                for p_b in P_b:
                    horiz = np.hypot(P[:, 0] - p_b[0], P[:, 2] - p_b[2])
                    below = P[:, 1] < p_b[1] - 0.005
                    ok = below & (horiz < R + r * 0.9)
                    if np.any(ok):
                        support = max(support, float(np.max(P[ok, 1] + R[ok] + r - 0.004)))
            bottom_y = float(np.min(P_b[:, 1]))
            if tilt >= 89.0:
                continue
            if bottom_y > support + 0.01:
                drop = min(bottom_y - support, 1.2 * dt)
                b.position = b.position - np.array([0.0, drop, 0.0])
                moved = True
                if bottom_y - drop <= support + 0.011 and b.is_flaming:
                    self.emit("collapse", b.position.copy())
        if moved:
            self.thermal_grid._update_geometry()

    # ------------------------------------------------------------------ raycast & interaction
    def _ray(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.player.eye.copy(), self.player.get_forward_ray()

    @staticmethod
    def _ray_sphere(o, d, c, r) -> Optional[float]:
        oc = o - c
        b = float(np.dot(oc, d))
        cc = float(np.dot(oc, oc)) - r * r
        disc = b * b - cc
        if disc < 0:
            return None
        t = -b - math.sqrt(disc)
        if t < 0:
            t = -b + math.sqrt(disc)
        return t if t >= 0 else None

    @staticmethod
    def _ray_capsule(o, d, a, b, r) -> Optional[float]:
        # closest approach between ray and segment, accept if within radius
        u = b - a
        w = o - a
        uu = float(np.dot(u, u))
        ud = float(np.dot(u, d))
        uw = float(np.dot(u, w))
        dw = float(np.dot(d, w))
        den = uu - ud * ud
        if den < 1e-9:
            s = 0.0
        else:
            s = max(0.0, min(1.0, (uw - ud * dw) / den))
        t = max(0.0, float(np.dot(a + u * s - o, d)))
        p_ray = o + d * t
        s = max(0.0, min(1.0, float(np.dot(p_ray - a, u)) / max(uu, 1e-9)))
        dist = float(np.linalg.norm(p_ray - (a + u * s)))
        if dist <= r:
            return t
        return None

    def _ground_hit(self, o, d, max_t=REACH + 2.0) -> Optional[np.ndarray]:
        t = 0.0
        step = 0.05
        while t < max_t:
            p = o + d * t
            if p[1] <= float(terrain.height(p[0], p[2])):
                return p
            t += step
        return None

    def raycast(self) -> Optional[Dict[str, Any]]:
        o, d = self._ray()
        hits = []
        # fuel in the fire pit
        for b in self.thermal_grid.bodies:
            a, c = b.endpoints()
            t = self._ray_capsule(o, d, a, c, max(0.014, b.initial_radius + 0.008))
            if t is not None and t < REACH:
                hits.append((t, {"kind": "pit_fuel", "obj": b, "label": b.describe()}))
        for wi in self.world_items:
            if wi.kind == "fuel" and wi.fuel is not None:
                ax = orientation_to_axis(np.array([0.0, wi.yaw, 0.0]))
                half = ax * wi.fuel.length * 0.5
                t = self._ray_capsule(o, d, wi.position - half, wi.position + half, wi.fuel.radius + 0.05)
            else:
                t = self._ray_sphere(o, d, wi.position, wi.radius)
            if t is not None and t < REACH:
                hits.append((t, {"kind": wi.kind, "obj": wi, "label": wi.label}))
        # drying rack, hearthboard kit
        t = self._ray_sphere(o, d, self.drying_rack.position + np.array([0, 0.55, 0]), 0.75)
        if t is not None and t < REACH:
            hits.append((t + 0.05, {"kind": "rack", "obj": self.drying_rack, "label": "Raised drying rack"}))
        t = self._ray_sphere(o, d, self.hearthboard_pos + np.array([0, 0.05, 0]), 0.28)
        if t is not None and t < REACH:
            hits.append((t, {"kind": "hearthboard", "obj": None, "label": "Bow-drill kit & hearthboard"}))
        # trees (vertical capsules)
        for tree in self.ecosystem.trees:
            if not tree.is_standing or tree.stage == TreeGrowthStage.SAPLING:
                continue
            dx, dz = tree.position[0] - o[0], tree.position[2] - o[2]
            if dx * dx + dz * dz > 25.0:
                continue
            a = tree.position.copy()
            c = tree.position + np.array([0.0, min(2.6, tree.height_m), 0.0])
            t = self._ray_capsule(o, d, a, c, tree.trunk_radius_m + 0.05)
            if t is not None and t < REACH:
                hits.append((t, {"kind": "tree", "obj": tree, "label": tree.describe()}))
        # felled trees lying on the ground: remaining rounds of the stem and the limbs
        for ft in self.fallen_trees:
            rel = ft.base - o
            if rel[0] * rel[0] + rel[2] * rel[2] > (ft.height + REACH) ** 2:
                continue
            for i in range(ft.n_sections):
                if ft.removed[i]:
                    continue
                s0, s1 = i * SECTION_LEN, (i + 1) * SECTION_LEN
                pa, pb = ft.point_at(s0, terrain.height), ft.point_at(s1, terrain.height)
                if min(np.linalg.norm(pa - o), np.linalg.norm(pb - o)) > REACH + 0.6:
                    continue
                t = self._ray_capsule(o, d, pa, pb, ft.radius_at(s0) + 0.03)
                if t is not None and t < REACH:
                    hp = o + d * t
                    s_hit = float(np.clip(np.dot(hp - ft.base, ft.dir), s0, s1))
                    hits.append((t, {"kind": "fallen_trunk", "obj": ft, "s": s_hit, "label": "Felled trunk"}))
            if not ft.top_taken:
                pa, pb = ft.point_at(ft.bole_length, terrain.height), ft.point_at(ft.height, terrain.height)
                t = self._ray_capsule(o, d, pa, pb, 0.05)
                if t is not None and t < REACH:
                    hits.append((t + 0.02, {"kind": "fallen_trunk", "obj": ft, "s": ft.bole_length + 0.2, "label": "Felled trunk"}))
            for limb in ft.limbs:
                if limb.removed:
                    continue
                a = ft.point_at(limb.s, terrain.height)
                tip = a + ft.limb_direction(limb) * limb.length
                t = self._ray_capsule(o, d, a, tip, limb.radius + 0.06)
                if t is not None and t < REACH:
                    hits.append((t - 0.01, {"kind": "fallen_limb", "obj": ft, "limb": limb, "label": "Limb"}))
        # the fire pit floor
        if abs(d[1]) > 1e-4:
            t = (self.fire_pit_pos[1] + 0.02 - o[1]) / d[1]
            if 0 < t < REACH + 0.6:
                p = o + d * t
                if math.hypot(p[0] - self.fire_pit_pos[0], p[2] - self.fire_pit_pos[2]) < 0.62:
                    hits.append((t + 0.02, {"kind": "fire_pit", "obj": None, "label": "Stone-ringed fire pit", "point": p}))
        if not hits:
            gp = self._ground_hit(o, d)
            if gp is not None and np.linalg.norm(gp - o) < REACH:
                return {"kind": "ground", "obj": None, "label": "", "point": gp, "distance": float(np.linalg.norm(gp - o))}
            return None
        hits.sort(key=lambda h: h[0])
        t, h = hits[0]
        h["distance"] = t
        h.setdefault("point", o + d * t)
        return h

    def _update_interaction(self, dt: float):
        self.target = self.raycast()
        self.ghost = self._compute_ghost()
        inv = self.inventory
        p = self.player
        # hold-to-burn lighter
        if self.primary_held and p.primary_hand == "pocket_lighter" and self.crafting.lighter_is_lit:
            pt = self._lighter_point()
            self.lighter_point = pt
            if pt is not None:
                self.thermal_grid.add_pilot(pt, duration=0.12)
                self._lighter_touch_extras(pt, dt)
        else:
            self.lighter_point = None
            if self.crafting.lighter_is_lit and not self.primary_held:
                self.crafting.release_lighter()
        # bow drilling
        if self.bow_active:
            if not (self.primary_held and p.primary_hand == "bow_drill_spindle"):
                self.bow_active = False
            elif self.friction_mode == "hand":
                # palms rubbing the spindle: palm speed (m/s) -> peak spindle speed v / r
                hd = self.crafting.hand_drill
                speed = self.bow_input_speed
                res = hd.drill_step(speed / hd.spindle_radius_m, p.stamina, dt, stone=self.secondary_held)
                if speed > 0.05:
                    p.stamina = max(0.0, p.stamina - res.get("stamina_spent", 0.0))
                    p.exertion = 0.8
                    self.crafting.active_task = "hand_drilling"
                    self.crafting.task_timer = 0.25
                if res.get("reason"):
                    res["message"] = res["reason"]
            else:
                force = 72.0 if self.secondary_held else 48.0
                speed = self.bow_input_speed
                res = self.crafting.stroke_bow_drill(force, speed, dt, p.stamina)
                if speed > 0.05:
                    # hard, sustained work: ~60 s of good bowing from full stamina
                    p.stamina = max(0.0, p.stamina - (0.5 + force * 0.008) * dt)
                    p.exertion = 0.6
            if self.bow_active:
                if res.get("has_ember") and not getattr(self, "_ember_notified", False):
                    self._ember_notified = True
                    self.stats["embers_made"] += 1
                    self.emit("ember", self.hearthboard_pos.copy())
                    self.notify("A glowing coal has formed in the V-notch! Tip it into tinder (RMB), then blow (F).", 7.0, "success")
                if res.get("message"):
                    if res["message"] != getattr(self, "_last_bow_msg", ""):
                        self.notify(res["message"], 3.0, "warn")
                    self._last_bow_msg = res["message"]
                    # snapped cord with no spare: the bow is useless; exhaustion just pauses strokes
                    if "no spare" in res["message"]:
                        self.bow_active = False
                else:
                    self._last_bow_msg = ""
            self.bow_input_speed *= max(0.0, 1.0 - 6.0 * dt)
        if not self.crafting.bow_drill.has_ember:
            self._ember_notified = False
        # kneel and lean over the hearthboard while spinning the hand drill
        if self.bow_active:
            to = self.hearthboard_pos - p.feet
            to[1] = 0.0
            d = float(np.linalg.norm(to))
            # kneel over the board: head ~45 cm back from the notch and low
            reach = max(0.0, min(0.5, d - 0.45))
            p.lean_target = (to / max(d, 1e-6)) * reach + np.array([0.0, -0.18, 0.0])
        else:
            p.lean_target = np.zeros(3)
        p.lean += (p.lean_target - p.lean) * min(1.0, 5.0 * dt)
        if self.bow_active:
            # eyes on the work: the gaze settles on the notch while you drill
            d = self.hearthboard_pos + np.array([0.05, 0.05, 0.03]) - p.eye
            yaw_t = math.degrees(math.atan2(d[2], d[0]))
            pitch_t = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
            k = min(1.0, 3.0 * dt)
            p.yaw += (((yaw_t - p.yaw + 180.0) % 360.0) - 180.0) * k
            p.pitch += (pitch_t - p.pitch) * k
        # breath charging
        if self.breath_start is not None:
            self.breath_charge = min(2.0, self.sim_time - self.breath_start)

    def _lighter_point(self) -> Optional[np.ndarray]:
        t = self.target
        if t is None:
            return None
        # reach through the gaps of the fire lay to the tinder near the line of sight
        o, d = self._ray()
        best, bd = None, 0.13
        for b in self.thermal_grid.bodies:
            if b.geometry_type != "tinder":
                continue
            tt = float(np.dot(b.position - o, d))
            if 0 < tt < LIGHTER_REACH + 0.3:
                dist = float(np.linalg.norm(o + d * tt - b.position))
                if dist < bd:
                    best, bd = b.position + np.array([0.0, -0.02, 0.0]) + (o + d * tt - b.position) * 0.5, dist
        if best is not None:
            return best
        if t.get("distance", 99) > LIGHTER_REACH:
            return None
        if t["kind"] in ("pit_fuel", "fire_pit", "fuel", "ground", "hearthboard"):
            return np.array(t["point"], dtype=float)
        return None

    def _lighter_touch_extras(self, pt: np.ndarray, dt: float):
        # lighting a fungus directly from the lighter flame
        if self.current_offhand() == "fungus" and self.inventory.fungi and self.crafting.smoldering_fungus is None:
            fg = next((f for f in self.inventory.fungi if not f.is_smoldering), None)
            if fg is not None and getattr(self, "_fungus_heat", 0.0) > 2.5:
                if self.crafting.light_fungus(fg, 600.0):
                    self.notify("The dry fungus catches and begins to smoulder - a slow-burning ember carrier.", 5.0, "success")
                self._fungus_heat = 0.0
            self._fungus_heat = getattr(self, "_fungus_heat", 0.0) + dt

    def toggle_friction_mode(self) -> str:
        """Slot 4 holds the friction-fire kit: a bow drill or a long hand-drill spindle."""
        self.bow_active = False
        self.friction_mode = "hand" if self.friction_mode == "bow" else "bow"
        if self.friction_mode == "hand":
            self.notify("Hand drill: a 50 cm spindle spun between the palms. Hold RMB to press down with the river stone.", 4.0)
        else:
            self.notify("Bow drill: saw the bow to spin the spindle; hold RMB to bear down.", 3.0)
        return self.friction_mode

    def current_offhand(self) -> str:
        """The off-hand (left) selection, or "none" when that hand is empty."""
        if self.offhand_empty:
            return "none"
        return OFFHAND[self.offhand_index % len(OFFHAND)]

    def set_offhand(self, kind: str):
        if kind in OFFHAND:
            self.offhand_index = OFFHAND.index(kind)
            self.offhand_empty = False
            self.player.secondary_hand = kind
        elif kind == "none":
            self.offhand_empty = True
            self.player.secondary_hand = "none"

    def cycle_offhand(self, step: int = 1):
        """Q / E: step through the off-hand selections, with an empty hand as the last slot."""
        n = len(OFFHAND)
        cur = n if self.offhand_empty else self.offhand_index % n
        nxt = (cur + step) % (n + 1)
        self.set_offhand("none" if nxt == n else OFFHAND[nxt])

    def empty_hands(self, primary: bool = True, secondary: bool = True):
        """Put everything away: nothing held in one or both hands."""
        if primary:
            self.crafting.release_lighter()
            self.bow_active = False
            self.player.primary_hand = "none"
        if secondary:
            self.set_offhand("none")

    # ------------------------------------------------------------------ placement ghost
    def _support_height(self, pos: np.ndarray, r: float, axis: np.ndarray, length: float, ignore=None) -> float:
        floor = max(float(terrain.height(pos[0] + axis[0] * length * f, pos[2] + axis[2] * length * f)) for f in (-0.45, 0.0, 0.45))
        support = floor + r
        pts = [pos + axis * length * f for f in (-0.4, -0.15, 0.15, 0.4)]
        for o in self.thermal_grid.bodies:
            if o is ignore:
                continue
            P = o.segment_positions()
            R = o.segment_radii()
            for p in pts:
                horiz = np.hypot(P[:, 0] - p[0], P[:, 2] - p[2])
                ok = horiz < (R + r) * 0.95
                if np.any(ok):
                    support = max(support, float(np.max(P[ok, 1] + R[ok] + r - 0.004)))
        return support

    def _compute_ghost(self) -> Optional[Dict[str, Any]]:
        t = self.target
        kind = self.current_offhand()
        if kind in ("fungus", "none"):
            return None
        if t is None or t["kind"] not in ("fire_pit", "pit_fuel"):
            return None
        item = self.inventory.peek(kind)
        center = self.fire_pit_pos
        # aim at where the ray meets the pit floor (where the piece's base will go)
        o, d = self._ray()
        if d[1] < -0.05:
            tf = (center[1] - o[1]) / d[1]
            aim = o + d * tf
        else:
            aim = np.array(t["point"], dtype=float)
        rel = aim - center
        horiz = math.hypot(rel[0], rel[2])
        if horiz > PIT_INNER_RADIUS + 0.1:
            rel[[0, 2]] *= (PIT_INNER_RADIUS + 0.1) / max(horiz, 1e-6)
            horiz = PIT_INNER_RADIUS + 0.1
        L = item.length if item else {"tinder": 0.14, "kindling": 0.3, "firewood": 0.38, "log": 0.46}[kind]
        r = item.radius if item else {"tinder": 0.045, "kindling": 0.008, "firewood": 0.03, "log": 0.06}[kind]
        # blueprint snapping (holographic guide)
        slot = None
        bp_key = self.reference_book.selected_blueprint_key if self.reference_book else None
        if bp_key:
            bp = ALL_BLUEPRINTS[bp_key]
            pieces = [(b.geometry_type, b.position - center, b.axis) for b in self.thermal_grid.bodies]
            bp.match_pieces(pieces)
            slot = bp.next_open_slot(kind, rel)
        if slot is not None:
            pos = center + slot.position
            orient = slot.orientation.copy()
            return {"kind": kind, "position": pos, "orientation": orient, "length": L, "radius": r,
                    "slot": slot.slot_id, "blueprint": bp_key, "valid": item is not None}
        p = self.player
        yaw_view = p.yaw + 90.0            # lie across the line of sight by default
        if kind == "tinder":
            pos = center + np.array([rel[0], 0.0, rel[2]])
            pos[1] = self._support_height(pos, r, np.array([1.0, 0, 0]), 0.05) if self.thermal_grid.bodies else center[1] + r
            orient = np.array([0.0, p.held_item_rotation, 0.0])
        else:
            auto_tilt = p.held_item_tilt
            # sticks and splits naturally lean into a teepee; logs lie flat (Shift+Wheel to change)
            lean = auto_tilt is None and kind in ("kindling", "firewood") and horiz > 0.05
            if lean or (auto_tilt is not None and auto_tilt > 5.0):
                tilt = auto_tilt if auto_tilt is not None else (62.0 if kind == "kindling" else 56.0)
                theta = math.atan2(rel[2], rel[0])
                base_r = max(horiz, 0.08)
                yaw = math.degrees(theta) + 180.0 + p.held_item_rotation * 0.0
                bx, bz = center[0] + base_r * math.cos(theta), center[2] + base_r * math.sin(theta)
                bottom = np.array([bx, float(terrain.height(bx, bz)) + r, bz])
                ax = orientation_to_axis(np.array([tilt, yaw, 0.0]))
                pos = bottom + ax * L * 0.5
                orient = np.array([tilt, yaw, 0.0])
            else:
                yaw = yaw_view + p.held_item_rotation
                orient = np.array([0.0, yaw, 0.0])
                ax = orientation_to_axis(orient)
                pos = center + np.array([rel[0], 0.0, rel[2]])
                pos[1] = self._support_height(pos, r, ax, L)
        return {"kind": kind, "position": pos, "orientation": orient, "length": L, "radius": r,
                "slot": None, "blueprint": bp_key, "valid": item is not None}

    # ------------------------------------------------------------------ primary (LMB)
    def primary_down(self) -> Dict[str, Any]:
        self.primary_held = True
        t = self.target
        p = self.player
        tool = p.primary_hand
        # context: picking things up always works with the free hand
        if (t is not None and t["kind"] == "fuel" and tool == "pocket_knife" and getattr(t["obj"], "fungi", 0) > 0):
            return self._knife_on_log(t["obj"])
        if t is not None and t["kind"] in ("fallen_trunk", "fallen_limb"):
            if tool != "felling_axe":
                self.notify("Take the axe (3) to limb and buck the felled tree.", 2.5)
                return {"success": False}
            return self.limb_tree(t["obj"], t["limb"]) if t["kind"] == "fallen_limb" else self.buck_tree(t["obj"], t["s"])
        if t is not None and t["kind"] in ("tinder_source", "kindling_pile", "woodpile", "fuel") and tool not in ("felling_axe",):
            return self.pick_up(t)
        if t is not None and t["kind"] == "fuel" and tool == "felling_axe" and t["obj"].fuel and t["obj"].fuel.kind == "log":
            return self.split_log(t["obj"])
        if t is not None and t["kind"] == "fuel":
            return self.pick_up(t)
        if t is not None and t["kind"] == "rack" and tool not in ("felling_axe",):
            if self.inventory.tarps > 0 and not self.drying_rack.has_tarp:
                self.inventory.tarps -= 1
                self.drying_rack.install_tarp()
                self.emit("tarp", self.drying_rack.position.copy())
                self.notify("Stretched the waterproof tarp over the drying rack. Rain can no longer soak the wood.", 5.0, "success")
                return {"success": True, "action": "tarp"}
            return self.take_from_rack()
        if t is not None and t["kind"] == "pit_fuel" and tool not in ("pocket_lighter", "accelerant"):
            return self.take_from_pit(t["obj"])
        return self.interact_primary()

    def primary_up(self):
        self.primary_held = False
        if self.player.primary_hand == "pocket_lighter":
            self.crafting.release_lighter()
        self.bow_active = False

    def interact_primary(self) -> Dict[str, Any]:
        """Tool action of the primary hand (LMB)."""
        item = self.player.primary_hand
        t = self.target
        if item == "none":
            # bare hands: pick things up (handled above); otherwise nothing to do
            return {"success": False, "action": "none"}
        if item == "pocket_lighter":
            res = self.crafting.strike_lighter()
            if res["success"]:
                pt = self._lighter_point()
                if pt is None:
                    self.notify("Lighter lit. Get within arm's reach of the tinder and hold LMB on it.", 3.5)
                else:
                    self.notify("Lighter flame on the fuel - hold it there...", 2.5)
                self.emit("lighter", pt)
            else:
                self.notify(res["message"], 3.5, "warn")
            return res
        if item == "pocket_knife":
            if t is not None and t["kind"] == "hearthboard":
                res = self.crafting.carve_hearthboard_notch()
            elif t is not None and t["kind"] == "tree":
                res = self._knife_on_tree(t["obj"])
            else:
                res = self.crafting.whittle_shavings()
            self.notify(res["message"], 3.5, "success" if res.get("success") else "warn")
            self.emit("knife", None)
            return res
        if item == "felling_axe":
            if t is not None and t["kind"] == "tree":
                return self.chop_tree(t["obj"])
            self.emit("swing", None)
            self.notify("Swing at a standing tree to fell it, at a felled tree to limb / buck it, or at a round to split it.", 3.0)
            return {"success": False}
        if item == "bow_drill_spindle":
            near = t is not None and t["kind"] == "hearthboard" or np.linalg.norm(self.player.feet[[0, 2]] - self.hearthboard_pos[[0, 2]]) < 1.4
            if near and self.friction_mode == "hand":
                self.bow_active = True
                if not self.crafting.bow_drill.has_v_notch:
                    self.notify("No V-notch yet: dust will scatter and no coal can form. Carve one with the knife (2).", 4.5, "warn")
                else:
                    self.notify("Rub the mouse left-right fast to spin the spindle between your palms. Hold RMB to press with the river stone.", 4.0)
                return {"success": True, "action": "hand_drill"}
            if near:
                if self.inventory.cordage <= 0 and self.crafting.bow_drill.cordage_condition <= 0:
                    self.notify("Your bow has no cordage. Strip inner bark (knife on a tree) and twist cord (C).", 4.0, "warn")
                    return {"success": False}
                self.bow_active = True
                if not self.crafting.bow_drill.has_v_notch:
                    self.notify("No V-notch yet: dust will scatter and no coal can form. Carve one with the knife (2).", 4.5, "warn")
                else:
                    self.notify("Saw the mouse left-right in long even strokes. Hold RMB to bear down.", 3.5)
                return {"success": True, "action": "bow_drill"}
            self.notify("Kneel by the hearthboard (next to the pit) to use the friction-fire kit.", 3.0)
            return {"success": False}
        if item == "accelerant":
            return self.pour_accelerant()
        return {"success": False}

    def _knife_on_tree(self, tree: Tree) -> Dict[str, Any]:
        if tree.has_fungi and tree.fungi_count > 0:
            tree.fungi_count -= 1
            if tree.fungi_count <= 0:
                tree.has_fungi = False
            ft = FungusType.BIRCH_POLYPORE if tree.species == WoodSpeciesType.PAPER_BIRCH else FungusType.TINDER_FUNGUS
            self.inventory.fungi.append(SmolderingFungus(ft, diameter_cm=float(self.rng.uniform(6.0, 10.0)), moisture=0.12))
            self.ecosystem.version += 1
            self.emit("harvest", tree.position.copy())
            return {"success": True, "message": f"Cut a {ft.value.split('(')[0].strip().lower()} bracket from the dead trunk. Select it (Q/E) to use as an ember carrier."}
        if tree.stage == TreeGrowthStage.MATURE and tree.bark_strips_left > 0:
            tree.bark_strips_left -= 1
            tree.bark_peeled = min(1.0, tree.bark_peeled + 0.34)
            res = self.crafting.strip_tree_bark()
            res["message"] += " Press C to twist cordage."
            self.ecosystem.version += 1
            return res
        return {"success": False, "message": "Nothing more to harvest from this tree with the knife."}

    def _knife_on_log(self, wi: "WorldItem") -> Dict[str, Any]:
        wi.fungi -= 1
        self.inventory.fungi.append(SmolderingFungus(FungusType.TINDER_FUNGUS, diameter_cm=float(self.rng.uniform(6.0, 9.0)),
                                                     moisture=0.14))
        self.emit("harvest", wi.position.copy())
        self.emit("knife", None)
        self.notify("Cut a tinder-fungus bracket from the rotting log. Select it (0) to use as an ember carrier.", 4.0, "success")
        return {"success": True, "action": "harvest_fungus"}

    def chop_tree(self, tree: Tree) -> Dict[str, Any]:
        axe = self.inventory.axe
        p = self.player
        if p.stamina < 4.0:
            self.notify("Too exhausted to swing - catch your breath.", 2.0, "warn")
            return {"success": False}
        p.stamina = max(0.0, p.stamina - 4.5)
        p.exertion = 1.2
        self.crafting._task("chopping", 0.7)
        heading = math.radians(p.yaw)
        if tree.notch_depth_m < tree.trunk_radius_m * 0.7 - 1e-9:
            res = TreeFellingSystem.chop_notch(tree, axe)
        else:
            res = TreeFellingSystem.chop_backcut(tree, axe, fall_heading=heading)
        self.emit("chop", tree.position + np.array([0, 0.9, 0]))
        if res.get("tree_falling"):
            self.emit("tree_creak", tree.position.copy())
            # the standing instance must disappear the moment the falling trunk takes over
            self.ecosystem.version += 1
        self.notify(res["message"], 2.5)
        return res

    # ------------------------------------------------------------------ processing a felled tree
    def _axe_swing_cost(self) -> bool:
        p = self.player
        if p.stamina < 4.0:
            self.notify("Too exhausted to swing - catch your breath.", 2.0, "warn")
            return False
        p.stamina = max(0.0, p.stamina - 4.5)
        p.exertion = 1.2
        self.crafting._task("chopping", 0.7)
        return True

    def _drop_fuel(self, item: FuelItem, pos: np.ndarray, yaw: float, label: str) -> WorldItem:
        pos = np.array(pos, dtype=float)
        pos[1] = float(terrain.height(pos[0], pos[2])) + item.radius
        wi = WorldItem("fuel", pos, 0.3 + min(0.35, item.radius * 2.0), label, fuel=item, yaw=yaw)
        self.world_items.append(wi)
        self.ground_logs.append(wi)
        return wi

    def _drop_pile(self, kind: str, pos: np.ndarray, count: int, label: str, moisture: float,
                   species: WoodSpeciesType = None, material: str = "") -> Optional[WorldItem]:
        if count <= 0:
            return None
        pos = np.array(pos, dtype=float)
        pos[1] = float(terrain.height(pos[0], pos[2])) + 0.06
        wi = WorldItem(kind, pos, 0.45, label, count=count, material=material, species=species)
        wi.moisture = float(moisture)
        wi.max_count = count
        wi.no_regrow = True
        self.world_items.append(wi)
        return wi

    def limb_tree(self, ft: FallenTree, limb: Limb) -> Dict[str, Any]:
        """One axe swing at a branch where it meets the trunk."""
        if not self._axe_swing_cost():
            return {"success": False}
        depth = FallenTree.swing_depth(self.inventory.axe, ft.species, ft.is_punk)
        a = ft.point_at(limb.s, terrain.height)
        self.emit("chop", a.copy())
        limb.cut_depth += depth
        need = 2.0 * limb.radius * 0.85
        if limb.cut_depth < need:
            self.notify(f"Limbing... {min(99, limb.cut_depth / need * 100):.0f}% through a {limb.radius * 200:.0f} cm branch", 1.5)
            return {"success": True, "action": "limb_progress", "progress": limb.cut_depth / need}
        limb.removed = True
        y = ft.limb_yield(limb)
        sp = ft.species
        ldir = ft.limb_direction(limb)
        flat = np.array([ldir[0], 0.0, ldir[2]])
        flat = flat / max(1e-6, np.linalg.norm(flat))
        yaw = math.degrees(math.atan2(flat[2], flat[0]))
        ms, mc = ft.m_surface * 0.92, ft.m_core * 0.9
        for fl in y["fire_logs"]:
            it = FuelItem("firewood", sp, ms, mc, length=fl["length"], radius=max(0.02, fl["radius"]), source="limb")
            self._drop_fuel(it, a + flat * fl["s"] + ft.side * float(self.rng.uniform(-0.15, 0.15)),
                            yaw + float(self.rng.uniform(-12, 12)), f"Green {sp.value.lower()} fire log")
        mid = a + flat * limb.length * 0.55
        self._drop_pile("kindling_pile", mid, y["kindling"], f"Pile of green {sp.value.lower()} branches", ms,
                        species=sp)
        tip = a + flat * limb.length * 0.9 + ft.side * 0.3
        moist_t = 0.25 if ft.tinder_material == "birch_bark" else min(1.4, ms * 1.1)
        self._drop_pile("tinder_source", tip, y["tinder"], ft.tinder_label, moist_t, material=ft.tinder_material)
        self.emit("tree_creak", a.copy())
        self.notify(f"Limb off: {len(y['fire_logs'])} fire logs, {y['kindling']} kindling sticks, {y['tinder']} handfuls of tinder - "
                    f"all green ({ms * 100:.0f} % moisture), season them before burning.", 5.0, "success")
        self._cleanup_fallen(ft)
        return {"success": True, "action": "limb", "yield": y}

    def buck_tree(self, ft: FallenTree, s: float) -> Dict[str, Any]:
        """One axe swing cross-cutting the stem at the nearest 45 cm mark."""
        i = ft.nearest_boundary(s) if s < ft.bole_length + 0.1 else ft.n_sections
        if i is None or ft.cut[i]:
            i = ft.nearest_boundary(min(s, ft.bole_length))
        if i is None:
            return {"success": False}
        s_cut = i * SECTION_LEN
        blocking = ft.blocking_limb(s_cut)
        if i == ft.n_sections:
            blocking = next((l for l in ft.limbs if not l.removed and l.s > s_cut - 0.35), blocking)
        if blocking is not None:
            self.notify("Branches are in the way - limb this part of the tree first.", 2.5, "warn")
            return {"success": False, "reason": "limbs"}
        if not self._axe_swing_cost():
            return {"success": False}
        r = ft.radius_at(s_cut)
        ft.cut_depth[i] += FallenTree.swing_depth(self.inventory.axe, ft.species, ft.is_punk)
        p_cut = ft.point_at(s_cut, terrain.height)
        self.emit("chop", p_cut.copy())
        need = 2.0 * r * 0.9
        if ft.cut_depth[i] < need:
            self.notify(f"Bucking... kerf {ft.cut_depth[i] * 100:.0f} / {need * 100:.0f} cm through the {2 * r * 100:.0f} cm stem", 1.5)
            return {"success": True, "action": "buck_progress", "progress": ft.cut_depth[i] / need}
        ft.cut[i] = True
        freed = []
        for k in (i - 1, i):
            if 0 <= k < ft.n_sections and not ft.removed[k] and ft.cut[k] and ft.cut[k + 1]:
                ft.removed[k] = True
                freed.append(k)
        spawned = []
        for k in freed:
            y = ft.round_yield(k)
            # a freed round drops off the kerf and rolls a little to one side
            pos = ft.point_at(y["s"], terrain.height) + ft.side * float(self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.12, 0.4))
            if y["kind"] == "kindling":
                self._drop_pile("kindling_pile", pos, 6, f"Green {ft.species.value.lower()} top wood", ft.m_surface, species=ft.species)
                continue
            it = FuelItem(y["kind"], ft.species, ft.m_surface, ft.m_core, length=y["length"], radius=y["radius"],
                          source="bucked", is_punk=ft.is_punk)
            what = "round" if y["kind"] == "log" else "fire log"
            wi = self._drop_fuel(it, pos, math.degrees(ft.heading) + float(self.rng.uniform(-25, 25)),
                                 f"Green {ft.species.value.lower()} {what} ({y['radius'] * 200:.0f} cm)")
            spawned.append(wi)
        if i == ft.n_sections and not ft.top_taken:
            ft.top_taken = True
            top = ft.point_at((ft.bole_length + ft.height) * 0.5, terrain.height)
            self._drop_pile("kindling_pile", top, ft.top_kindling(), f"Green {ft.species.value.lower()} leader & tip", ft.m_surface,
                            species=ft.species)
        if spawned:
            it = spawned[0].fuel
            hint = " Split it with the axe - it is far too heavy to carry whole." if it.mass_kg > 25 else ""
            self.notify(f"A {it.radius * 200:.0f} cm round comes free: {it.mass_kg:.0f} kg of green {ft.species.value.lower()}.{hint}", 5.0, "success")
        else:
            self.notify("Cut through! Cut the other end of the round to free it.", 3.0)
        self._cleanup_fallen(ft)
        return {"success": True, "action": "buck", "freed": len(spawned)}

    def _cleanup_fallen(self, ft: FallenTree):
        if ft.limbs_left == 0 and ft.rounds_left == 0 and not ft.top_taken:
            ft.top_taken = True
            top = ft.point_at((ft.bole_length + ft.height) * 0.5, terrain.height)
            self._drop_pile("kindling_pile", top, ft.top_kindling(), f"Green {ft.species.value.lower()} leader & tip", ft.m_surface,
                            species=ft.species)
        if ft.finished and ft in self.fallen_trees:
            self.fallen_trees.remove(ft)
            self.notify("The whole tree is worked up into fuel.", 4.0, "success")

    def split_log(self, wi: WorldItem) -> Dict[str, Any]:
        """Splitting a round with the axe: big rounds take several blows and break into many
        wedges (about one split per 11 cm of diameter-squared, like real firewood)."""
        f = wi.fuel
        if not self._axe_swing_cost():
            return {"success": False}
        self.emit("chop", wi.position.copy())
        n = int(max(2, min(12, round((f.radius / 0.055) ** 2))))
        blows = max(1, n // 3)
        wi.split_blows = getattr(wi, "split_blows", 0) + 1
        if wi.split_blows < blows:
            self.notify(f"The round cracks... ({wi.split_blows}/{blows} blows)", 1.5)
            return {"success": True, "action": "split_progress"}
        self.world_items.remove(wi)
        if wi in self.ground_logs:
            self.ground_logs.remove(wi)
        r_split = max(0.02, f.radius / math.sqrt(n) * 1.1)
        for i in range(n):
            fw = FuelItem("firewood", f.species, f.m_surface * 0.9 + f.m_core * 0.1, f.m_core, length=f.length * 0.95,
                          radius=r_split, source="split", is_punk=f.is_punk)
            # the pieces fly apart around the chopping spot
            ang = 2.0 * math.pi * (i + float(self.rng.uniform(0.1, 0.9))) / n
            dist = f.radius * 0.8 + float(self.rng.uniform(0.05, 0.3))
            off = np.array([math.cos(ang), 0, math.sin(ang)]) * dist
            self._drop_fuel(fw, wi.position + off, float(self.rng.uniform(0, 360)), f"Split {f.species.value.lower()}")
        self.notify(f"Split the {f.species.value.lower()} round into {n} pieces - split wood seasons and catches far faster.", 4.0, "success")
        return {"success": True, "action": "split", "pieces": n}

    def pour_accelerant(self) -> Dict[str, Any]:
        inv = self.inventory
        if inv.gasoline_kg <= 0.01:
            self.notify("No accelerant. Gasoline can be bought in the Store app (Tab).", 3.0, "warn")
            return {"success": False}
        t = self.target
        if t is None or t["kind"] not in ("pit_fuel", "fire_pit") or t.get("distance", 99) > REACH:
            self.notify("Aim at the fuel in the fire pit to pour.", 2.5)
            return {"success": False}
        dose = min(inv.gasoline_kg, 0.08)
        inv.gasoline_kg -= dose
        bodies = self.thermal_grid.bodies
        if bodies:
            per = dose / len(bodies)
            for b in bodies:
                b.add_accelerant(per, "gasoline")
        else:
            self.thermal_grid.add_burner(self.fire_pit_pos + np.array([0, 0.01, 0]), 0.0, 0.0)
        if self.thermal_grid.is_burning:
            # vapour flashes back over the flames: a fireball - never do this in real life
            self.thermal_grid.add_burner(self.fire_pit_pos + np.array([0, 0.15, 0]), 350.0, 1.6, diameter=0.9, tag="flash")
            self.gas_flash_timer = 1.6
            self.emit("gas_flash", self.fire_pit_pos.copy())
            self.notify("WHOOMPH! Gasoline vapour flashed over the fire - a fireball scorches your eyebrows. Never pour on a lit fire!", 6.0, "danger")
        else:
            self.emit("pour", self.fire_pit_pos.copy())
            self.notify(f"Doused the fuel with {dose * 1000:.0f} g of gasoline. Its vapour will ignite from any flame.", 4.0, "warn")
        return {"success": True, "action": "pour", "kg": dose}

    # ------------------------------------------------------------------ pickups
    def pick_up(self, t: Dict[str, Any]) -> Dict[str, Any]:
        wi: WorldItem = t["obj"]
        inv = self.inventory
        if wi.kind == "tinder_source":
            if wi.count <= 0:
                self.notify(f"The {wi.label.lower()} is picked clean - it will regrow with time.", 3.0)
                return {"success": False}
            moist = 0.07 if self.weather.rain_intensity < 0.05 and self.weather.ground_wetness < 0.3 else 0.08 + 0.5 * self.weather.ground_wetness
            if getattr(wi, "moisture", None) is not None:
                moist = wi.moisture          # green material from a felled tree (dries as it seasons)
            item = make_tinder(wi.material, moisture=moist, source="gathered")
            if not inv.add_fuel(item):
                self.notify("Your pack is too heavy.", 2.0, "warn")
                return {"success": False}
            wi.count -= 1
            if wi.count <= 0 and getattr(wi, "no_regrow", False):
                self.world_items.remove(wi)
            self.emit("pickup", wi.position.copy())
            self.set_offhand("tinder")
            self.notify(f"Gathered {item.name.lower()} (moisture {item.moisture * 100:.0f} %). Tinder: {inv.count('tinder')}", 3.0)
            return {"success": True, "action": "pickup", "kind": "tinder"}
        if wi.kind == "kindling_pile":
            if wi.count <= 0:
                self.notify("No more dry twigs here. Fell a tree for branches or find another pile.", 3.0)
                return {"success": False}
            n = min(3, wi.count)
            m = getattr(wi, "moisture", None)
            if m is None:
                m = 0.09 + 0.3 * self.weather.ground_wetness
            for _ in range(n):
                st = make_kindling(wi.species or WoodSpeciesType.WHITE_PINE, m, "gathered", self.rng)
                st.m_core = max(st.m_core, m)
                if not inv.add_fuel(st):
                    self.notify("Too heavy to carry more.", 2.0, "warn")
                    n = _ + 0
                    break
            wi.count -= n
            if wi.count <= 0 and getattr(wi, "no_regrow", False):
                self.world_items.remove(wi)
            self.emit("pickup", wi.position.copy())
            self.set_offhand("kindling")
            self.notify(f"Gathered {n} dry twigs. Kindling: {inv.count('kindling')}", 3.0)
            return {"success": True, "action": "pickup", "kind": "kindling"}
        if wi.kind == "woodpile":
            stock = getattr(wi, "stock", [])
            if not stock:
                self.notify("The woodpile is empty. Fell, split and season more wood.", 3.0)
                return {"success": False}
            kind = "log" if self.current_offhand() == "log" and any(s.kind == "log" for s in stock) else "firewood"
            cand = [s for s in stock if s.kind == kind] or stock
            item = cand[0]
            if not inv.add_fuel(item):
                self.notify("Your pack is too heavy.", 2.0, "warn")
                return {"success": False}
            stock.remove(item)
            wi.count = len(stock)
            self.emit("pickup_wood", wi.position.copy())
            self.set_offhand(item.kind)
            self.notify(f"Took {item.name.lower()} ({item.moisture * 100:.0f} % MC). {len(stock)} pieces left.", 3.0)
            return {"success": True, "action": "pickup", "kind": item.kind}
        if wi.kind == "fuel" and wi.fuel is not None:
            if not inv.add_fuel(wi.fuel):
                if wi.fuel.mass_kg > inv.MAX_CARRY_KG * 0.6:
                    self.notify(f"That {wi.fuel.mass_kg:.0f} kg {wi.fuel.name.lower()} is too heavy to carry - split it with the axe (3).", 3.5, "warn")
                else:
                    self.notify("Too heavy to carry more.", 2.0, "warn")
                return {"success": False}
            self.world_items.remove(wi)
            if wi in self.ground_logs:
                self.ground_logs.remove(wi)
            self.emit("pickup_wood", wi.position.copy())
            self.set_offhand(wi.fuel.kind)
            self.notify(f"Picked up {wi.fuel.name.lower()} - {wi.fuel.moisture * 100:.0f} % moisture ({wi.fuel.moisture_label()}).", 3.5)
            return {"success": True, "action": "pickup", "kind": wi.fuel.kind}
        return {"success": False}

    def take_from_rack(self) -> Dict[str, Any]:
        it = self.drying_rack.take_driest()
        if it is None:
            self.notify("The rack is empty. Place green logs on it (RMB) to season them.", 3.0)
            return {"success": False}
        if not self.inventory.add_fuel(it):
            self.drying_rack.add_log(it)
            self.notify("Too heavy to carry more.", 2.0, "warn")
            return {"success": False}
        self.emit("pickup_wood", self.drying_rack.position.copy())
        self.set_offhand(it.kind)
        self.notify(f"Took {it.name.lower()} from the rack: {it.moisture * 100:.0f} % moisture ({it.moisture_label()}).", 4.0)
        return {"success": True, "action": "rack_take"}

    def take_from_pit(self, body: CombustionBody) -> Dict[str, Any]:
        if body.t_surface > 70.0 or body.is_flaming:
            self.notify("Too hot to touch!", 2.0, "warn")
            return {"success": False}
        self.thermal_grid.remove_body(body)
        item = FuelItem.from_body(body, "pit")
        self.inventory.add_fuel(item)
        self.emit("pickup_wood", body.position.copy())
        self.notify(f"Took the {item.name.lower()} back out of the pit.", 2.5)
        return {"success": True, "action": "pit_take"}

    # ------------------------------------------------------------------ secondary (RMB)
    def secondary_down(self) -> Dict[str, Any]:
        self.secondary_held = True
        if self.bow_active or (self.player.primary_hand == "bow_drill_spindle" and self.crafting.bow_drill.has_ember):
            if self.crafting.bow_drill.has_ember and not self.bow_active:
                res = self.crafting.transfer_ember_to_nest()
                self.notify(res["message"], 5.0, "success" if res["success"] else "warn")
                return res
            return {"success": True, "action": "bear_down"}
        return self.interact_secondary()

    def secondary_up(self):
        self.secondary_held = False

    def interact_secondary(self) -> Dict[str, Any]:
        """RMB: place the off-hand item (fuel into the pit / logs onto the rack / fungus)."""
        kind = self.current_offhand()
        t = self.target
        if kind == "none":
            self.notify("Your off hand is empty - select fuel with 6-0 (or Q / E).", 2.5)
            return {"success": False}
        if kind == "fungus":
            return self._use_fungus()
        if t is not None and t["kind"] == "rack":
            if kind not in ("log", "firewood"):
                self.notify("Only logs and split wood go on the drying rack.", 2.5)
                return {"success": False}
            it = self.inventory.take(kind)
            if it is None:
                self.notify(f"You carry no {kind}.", 2.0)
                return {"success": False}
            if not self.drying_rack.add_log(it):
                self.inventory.add_fuel(it)
                self.notify("The rack is full.", 2.0)
                return {"success": False}
            self.emit("place_wood", self.drying_rack.position.copy())
            self.notify(f"Stacked {it.name.lower()} on the raised rack to season ({it.moisture * 100:.0f} % now). Pass time (T) to cure it.", 4.5)
            return {"success": True, "action": "rack_place"}
        if t is not None and t["kind"] == "ground" and kind in ("log", "firewood", "kindling"):
            it = self.inventory.take(kind)
            if it is None:
                return {"success": False}
            gp = np.array(t["point"])
            gp[1] = float(terrain.height(gp[0], gp[2])) + it.radius
            wi = WorldItem("fuel", gp, 0.45 if kind != "kindling" else 0.3, it.name.capitalize(), fuel=it, yaw=self.player.yaw + 90.0)
            self.world_items.append(wi)
            self.ground_logs.append(wi)
            self.emit("place_wood", gp)
            self.notify(f"Set the {it.name.lower()} down on the ground.", 2.0)
            return {"success": True, "action": "drop"}
        return self.place_in_pit()

    def place_in_pit(self) -> Dict[str, Any]:
        gh = self.ghost
        kind = self.current_offhand()
        if gh is None:
            self.notify("Aim at the fire pit to place fuel.", 2.0)
            return {"success": False}
        item = self.inventory.take(kind)
        if item is None:
            self.notify(f"You carry no {kind}. Gather some first.", 2.5, "warn")
            return {"success": False}
        body = item.to_body(gh["position"], gh["orientation"], temp_c=self.weather.ambient_temp_c)
        self.thermal_grid.add_body(body)
        self.emit("place_tinder" if kind == "tinder" else "place_wood", gh["position"].copy())
        extra = f" (guide slot {gh['slot']})" if gh.get("slot") else ""
        self.notify(f"Placed {item.name.lower()} - {item.moisture * 100:.0f} % moisture{extra}.", 2.5)
        return {"success": True, "action": "place", "kind": kind}

    def _use_fungus(self) -> Dict[str, Any]:
        inv = self.inventory
        if not inv.fungi:
            self.notify("No fungus. Cut brackets from standing dead trees with the knife.", 3.0)
            return {"success": False}
        cf = self.crafting
        near_pit = np.linalg.norm(self.player.feet[[0, 2]] - self.fire_pit_pos[[0, 2]]) < 2.5
        if cf.smoldering_fungus is None:
            if near_pit and self.thermal_grid.core_coal_temp > 300.0:
                fg = next((f for f in inv.fungi if not f.is_smoldering), None)
                if fg and cf.light_fungus(fg, self.thermal_grid.core_coal_temp):
                    self.notify("You press the fungus to the coals - it catches and smoulders. Carry it anywhere.", 5.0, "success")
                    return {"success": True, "action": "light_fungus"}
                self.notify("The fungus is too damp to take a coal.", 3.0, "warn")
                return {"success": False}
            self.notify("Light the fungus from glowing coals (RMB near a fire) or a lighter (hold LMB with the fungus selected).", 4.0)
            return {"success": False}
        res = cf.transfer_ember_to_nest()
        self.notify(res["message"], 4.0, "success" if res["success"] else "warn")
        return res

    def _place_flaming_nest(self):
        cf = self.crafting
        nest = cf.active_tinder_nest
        if nest is None:
            return
        dist = float(np.linalg.norm(self.player.feet[[0, 2]] - self.fire_pit_pos[[0, 2]]))
        cf.active_tinder_nest = None
        if dist > 4.0:
            self.notify("The nest burst into flame in your hands - but you were too far from the fire lay and had to drop it.", 5.0, "warn")
            return
        body = CombustionBody(WoodSpeciesType.RED_CEDAR if nest.material != "birch_bark" else WoodSpeciesType.PAPER_BIRCH,
                              dry_mass=max(0.01, nest.dry_mass_g / 1000.0), initial_m_surface=nest.moisture,
                              initial_m_core=nest.moisture, initial_temp=350.0, geometry_type="tinder",
                              position=self.fire_pit_pos + np.array([0.0, 0.045, 0.0]))
        body._v("phi")[:] = 0.6
        body._v("mb")[:] = 0.004
        self.thermal_grid.add_body(body)
        self.emit("nest_flame", body.position.copy())
        self.notify("FLAME! You tuck the blazing nest under the fire lay.", 5.0, "success")

    # ------------------------------------------------------------------ breath (F)
    def breath_down(self):
        self.breath_start = self.sim_time
        self.breath_charge = 0.0

    def breath_up(self) -> Dict[str, Any]:
        if self.breath_start is None:
            return {"success": False}
        held = max(0.05, self.sim_time - self.breath_start)
        self.breath_start = None
        self.breath_charge = 0.0
        return self.blow_breath(held_seconds=held)

    @staticmethod
    def breath_velocity_from_hold(held: float) -> float:
        """Tap = soft puff, 0.3-0.8 s = deep even breath, > 1.2 s = too forceful."""
        if held < 0.15:
            return 0.6
        if held < 0.25:
            return 1.2
        if held <= 0.9:
            return 1.6 + (held - 0.25) * 3.0
        return 3.6 + (held - 0.9) * 6.0

    def blow_breath(self, held_seconds: float = 0.5, velocity: Optional[float] = None) -> Dict[str, Any]:
        """Player breathes (F): into a tinder nest with an ember, or into the coals."""
        v = velocity if velocity is not None else self.breath_velocity_from_hold(held_seconds)
        self.last_breath_velocity = v
        cf = self.crafting
        self.emit("breath", {"velocity": v, "cold": self.weather.ambient_temp_c < 5.0})
        nest = cf.active_tinder_nest
        if nest is not None and nest.has_ember:
            res = cf.blow_tinder(breath_velocity=v, dt=min(1.2, held_seconds + 0.2))
            st = res.get("status")
            if res.get("is_flaming"):
                self._place_flaming_nest()
            elif st == "blown_out":
                self.notify("Too hard! The coal scattered out of the nest.", 3.5, "warn")
            elif st == "too_weak":
                self.notify(f"Too soft - the coal is cooling ({res.get('core_temp', 0):.0f} C). Longer, steadier breaths.", 2.5)
            elif st == "too_strong":
                self.notify(f"Easy! Too forceful ({v:.1f} m/s).", 2.5, "warn")
            else:
                self.notify(f"The coal brightens... core {res.get('core_temp', 0):.0f} C (flame at 550 C)", 2.0)
            return res
        d = float(np.linalg.norm(self.player.feet[[0, 2]] - self.fire_pit_pos[[0, 2]]))
        if d < 2.4 and (self.thermal_grid.bodies or self.thermal_grid.coal_bed.char_kg > 0.005):
            cf._task("blowing", 0.8)
            target = self.fire_pit_pos + np.array([0.0, 0.06, 0.0])
            power, blown = self.thermal_grid.blow_into_fire(v)
            self.thermal_grid.breath_target = target
            if blown:
                self.notify("Overblown! Ash and sparks scatter from the pit.", 2.5, "warn")
                self.emit("ash_scatter", target)
            else:
                self.notify(f"You blow into the base of the fire ({v:.1f} m/s) - the coals flare.", 2.0)
            return {"power_boost": power, "blown_away": blown}
        return {"status": "no_target"}

    # ------------------------------------------------------------------ crafting shortcuts
    def twist_cordage(self) -> Dict[str, Any]:
        res = self.crafting.twist_two_ply_cordage()
        self.notify(res["message"], 3.5, "success" if res["success"] else "warn")
        return res

    # ------------------------------------------------------------------ purchases
    def apply_purchase(self, item_id: str) -> str:
        inv = self.inventory
        if item_id == "lighter_refill":
            inv.lighter_starts += 4
            return "Lighter refilled: +4 fire starts."
        if item_id == "waterproof_tarp":
            inv.tarps += 1
            return "Tarp in your pack. Aim at the drying rack and press LMB to stretch it over."
        if item_id == "steel_knife":
            inv.knife = "steel"
            return "High-carbon steel knife equipped (slot 2)."
        if item_id == "steel_axe":
            inv.axe = "steel"
            return "Drop-forged felling axe equipped (slot 3)."
        if item_id == "gasoline_canister":
            inv.gasoline_kg += 2.8
            return "Gasoline canister (slot 5). Pour on COLD fuel only."
        if item_id == "reference_book_advanced":
            if self.reference_book is not None:
                self.reference_book.unlock_editions()
            return "Advanced blueprints unlocked in the Field Compendium (B)."
        if item_id == "paracord":
            inv.cordage += 2
            return "+2 cordage for the bow drill."
        return ""

    # ------------------------------------------------------------------ time skip
    def _fire_skip_step(self, h: float):
        """World-side physics during a time skip: burnt-through pieces slump into the coals,
        ash spills out of the pit into the soil."""
        self._skip_settle_t = getattr(self, "_skip_settle_t", 0.0) + h
        if self._skip_settle_t >= 20.0:
            self._skip_settle_t = 0.0
            self._settle_bodies(1.0)
        g = self.thermal_grid
        if g.coal_bed.ash_kg > 0.4:
            self.ecosystem.add_campfire_ash(self.fire_pit_pos, g.coal_bed.ash_kg - 0.4)
            g.coal_bed.ash_kg = 0.4
        if g.collapse_positions:
            self.stats["pieces_burned"] += len(g.collapse_positions)
            g.collapse_positions.clear()

    def apply_time_skip(self, hours: float) -> Dict[str, Any]:
        """Called by the TimeEngine ('T' key) to fast-forward hours, days or months."""
        days = hours / 24.0
        report: Dict[str, Any] = {"hours": hours}
        rack_before = self.drying_rack.mean_moisture()
        ground_before = {id(w): (w.fuel.moisture, w.fuel.species) for w in self.ground_logs if w.fuel}
        # 1. the fire burns on (detailed for the first hours, analytic afterwards)
        fire_before = self.thermal_grid.is_burning
        ff = self.thermal_grid.fast_forward(min(hours, 48.0) * 3600.0, weather=self.weather, on_step=self._fire_skip_step)
        report["fire_energy_mj"] = ff["energy_mj"]
        # 2. weather & calendar, day by day, seasoning racks and ground logs along the way
        step_h = 24.0 if hours >= 24.0 else hours
        done = 0.0
        while done < hours - 1e-9:
            h = min(step_h, hours - done)
            wx = self.weather.advance_time(h)
            d = h / 24.0
            rain_frac = wx["rain_hours"] / max(1e-6, h)
            temp = wx["mean_temp"]
            rh = self.weather.relative_humidity
            season = self.weather.current_season.value
            self.drying_rack.update_seasoning(d, season, wind_speed=self.weather.wind_speed_m_s,
                                              temp_c=temp, rh=rh, rain_fraction=rain_frac)
            for w in self.ground_logs:
                if w.fuel is None:
                    continue
                GroundSeasoningSimulator.step_ground_log(w.fuel, d, season, rain_frac,
                                                         soil_moisture=terrain.ground_moisture(w.position[0], w.position[2]),
                                                         temp_c=temp, rh=rh, wind=self.weather.wind_speed_m_s)
                if w.fuel.species == WoodSpeciesType.PUNK_WOOD and "punk" not in w.label.lower():
                    w.label = "Rotting punk-wood log"
                if w.fuel.species == WoodSpeciesType.PUNK_WOOD:
                    # late decay: bracket fungi fruit on the sodden log, then it melts into humus
                    w.decay_days += d
                    if season in ("Autumn", "Summer", "Spring") and w.decay_days > 20 and getattr(w, "fungi", 0) < 3:
                        w.fungi = getattr(w, "fungi", 0) + (1 if self.rng.random() < min(1.0, 0.15 * d) else 0)
                    if w.decay_days > 240:
                        w.humus = True
            for ft in self.fallen_trees:
                GroundSeasoningSimulator.step_ground_log(ft, d, season, rain_frac,
                                                         soil_moisture=terrain.ground_moisture(ft.base[0], ft.base[2]),
                                                         temp_c=temp, rh=rh, wind=self.weather.wind_speed_m_s)
            self.inventory.equilibrate_all(h, rh, temp)
            eco = self.ecosystem.step_ecosystem(d, self.weather.wind_vector, season)
            for k, v in eco.items():
                report[k] = report.get(k, 0) + v
            # tinder sources regrow
            for wi in self.world_items:
                if getattr(wi, "moisture", None) is not None:
                    # brush piles from a felled tree dry (or soak) where they lie
                    thin = _PileStub(wi.moisture, 0.008 if wi.kind == "kindling_pile" else 0.003)
                    step_moisture(thin, d, temp, rh, self.weather.wind_speed_m_s * 0.5, rain=rain_frac, on_ground=True,
                                  soil_moisture=terrain.ground_moisture(wi.position[0], wi.position[2]) * 0.5)
                    wi.moisture = float(0.5 * (thin.m_surface + thin.m_core))
                if getattr(wi, "no_regrow", False):
                    continue
                if wi.kind in ("tinder_source", "kindling_pile") and wi.count < wi.max_count:
                    wi.regrow_days += d
                    if wi.regrow_days > 3.0:
                        wi.count = min(wi.max_count, wi.count + int(wi.regrow_days // 3.0))
                        wi.regrow_days = wi.regrow_days % 3.0
            done += h
        rotted = [w for w in self.ground_logs if getattr(w, "humus", False)]
        for w in rotted:
            self.ground_logs.remove(w)
            if w in self.world_items:
                self.world_items.remove(w)
            self.ecosystem.add_campfire_ash(w.position, 0.5)   # nutrients return to the soil
        if rotted:
            report["humus"] = len(rotted)
        self._update_seasonal_items()
        self.stats["days_elapsed"] += days
        self.time_engine.total_days_elapsed = self.stats["days_elapsed"]
        report["fire_before"] = fire_before
        report["fire_after"] = self.thermal_grid.is_burning
        g = self.thermal_grid
        report["fire_pieces_left"] = len(g.bodies)
        report["fire_mass_left_kg"] = float(sum(b.dry_mass for b in g.bodies) + g.coal_bed.char_kg)
        report["fire_kw_after"] = (g.flame_hrr + g.char_hrr) / 1000.0
        report["fire_burn_s"] = ff["simulated_s"] if not ff.get("burning") else None
        report["rack_before"] = rack_before
        report["rack_after"] = self.drying_rack.mean_moisture()
        report["ground_logs"] = []
        for w in self.ground_logs:
            if w.fuel and id(w) in ground_before:
                report["ground_logs"].append({"name": w.label, "before": ground_before[id(w)][0], "after": w.fuel.moisture,
                                              "punk": w.fuel.species == WoodSpeciesType.PUNK_WOOD})
        report["season"] = self.weather.current_season.value
        report["day"] = int(self.weather.day_of_season)
        report["clock"] = self.weather.clock_string()
        self.last_skip_report = report
        msg = f"{hours:.0f} h passed - {report['season']} day {report['day']}, {report['clock']}."
        if rack_before is not None and report["rack_after"] is not None:
            msg += f" Rack wood: {rack_before * 100:.0f}% -> {report['rack_after'] * 100:.0f}% MC."
        self.notify(msg, 7.0)
        self.emit("time_skip", hours)
        self._update_colliders()
        return report

    # ------------------------------------------------------------------ legacy API
    def interact_secondary_legacy(self):
        return self.interact_secondary()
