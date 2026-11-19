"""The game simulation: owns the world, entities, NPCs, and game-state machine.

Completely independent of pygame and OpenGL so it can be driven by tests (or a headless
server) with synthetic ``InputState`` frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cyberduke.ai import BaseNPC, Civilian, CyberGuard, Merchant
from cyberduke.ai.alley import AlleyLife, LooseItem
from cyberduke.config import CLOCK, INTERACT_RANGE, PLAYER, SIMULATION
from cyberduke.core.clock import WorldClock, clamp_dt
from cyberduke.core.dialogue import Bark, Conversation, Line
from cyberduke.core.events import EventBus, NoiseEmitted
from cyberduke.core.geometry import AABB, Viewpoint
from cyberduke.core.interp import wrap_angle_deg
from cyberduke.core.inventory import CLOSEUP_HACK_SPEED, DETECTION_GAIN, item
from cyberduke.core.state import GameState
from cyberduke.entities import AccessPanel, Pickup, Player, SecurityCamera, Terminal, VendingMachine
from cyberduke.entities.metro import Metro
from cyberduke.entities.player import PlayerController
from cyberduke.entities.traffic import Traffic
from cyberduke.levels import LEVELS, build_facility
from cyberduke.systems import DetectionSystem, HackingSystem, Scanner
from cyberduke.systems.quests import NAMES, QuestDirector, QuestLog
from cyberduke.systems.weather import PuddleSite, Weather, puddle_sites
from cyberduke.world import LevelData, PrivateArea, Room, SlidingDoor

TALK_RANGE = 2.2  # metres within which [E] talks to somebody in front of you
TALK_CONE = 40.0  # degrees off the line of sight they may be
TOAST_SECONDS = 4.5  # how long a notice ("NEW QUEST: ...") stays up
TALK_COOLDOWN = 0.8  # after a conversation, before [E] can start another


@dataclass
class InputState:
    """One frame of player intent, independent of the input device."""

    forward: float = 0.0  # +1 = W, -1 = S
    strafe: float = 0.0  # +1 = A (left), -1 = D (right)
    interact: bool = False  # [E] held
    look_dx: float = 0.0  # mouse delta, pixels
    look_dy: float = 0.0
    toggle_crouch: bool = False  # [C] pressed this frame
    reset: bool = False  # [R] pressed this frame
    toggle_pause: bool = False  # [Esc] pressed this frame: open / close the pause menu
    menu_move: int = 0  # [Up]/[Down] pressed this frame while the pause menu is open
    menu_move_x: int = 0  # [Left]/[Right]: change the selected setting
    menu_select: bool = False  # [Enter]
    scan: bool = False  # [Tab] pressed this frame: scanner pulse
    hack_slot: int = 0  # [1]-[9] pressed this frame: upload that option to the focused target
    disconnect: bool = False  # [Q] pressed this frame: leave a hijacked camera feed
    cycle_camera: int = 0  # [Left]/[Right] pressed this frame: -1 / +1 switch linked feed
    interact_pressed: bool = False  # [E] went down this frame: talk, pick up, skip a line
    # the menus and screens (``game.Game``; the simulation itself ignores these)
    toggle_inventory: bool = False  # [Tab]
    tab: int = 0  # [ / ]: the previous / next tab of the inventory screen
    menu_back: bool = False  # [Esc] / [Backspace] inside a screen
    mouse_x: float = -1.0  # the cursor (screen pixels), while a screen shows one
    mouse_y: float = -1.0
    click: bool = False  # left button went down this frame
    wheel: int = 0  # mouse wheel: +1 up, -1 down
    scroll: int = 0  # [PgUp] / [PgDn]: -1 / +1


class Simulation:
    def __init__(self, level: LevelData | None = None) -> None:
        self.time = 0.0
        self.paused = False
        self.level_generation = 0  # bumps on every level load (renderers drop their caches)
        self.player = Player(spawn=(0.0, 0.0, 0.0))
        self.first_level = level or build_facility()  # where a fresh run begins
        self.quests = QuestLog()
        self.director = QuestDirector(self)
        # a camera that is not the netrunner's eyes (the start menu's drifting view)
        self.camera_override: Viewpoint | None = None
        # the time of day in the city (it keeps running from level to level)
        self.clock = WorldClock(CLOCK.new_game_at, CLOCK.minutes_per_second)
        # ... and its weather (``systems.weather``): it rains on whichever level is loaded
        self.weather = Weather()
        self._build(self.first_level)

    def _build(self, level: LevelData, *, keep_intel: bool = False) -> None:
        """Stands the whole world up for ``level``: room, entities, NPCs, networks."""
        self.level = level
        self.bus = EventBus()
        self.room = Room(self.level)
        self.room.bind(self.bus)
        self.player.spawn = self.level.spawn
        self.player.reset(keep_intel=keep_intel)
        self.controller = PlayerController(self.player, self.room)
        self.detection = DetectionSystem()

        bus = self.bus
        doors_of = {
            t.terminal_id: tuple(
                d.door_id for d in self.level.doors if d.unlocked_by == t.terminal_id
            )
            for t in self.level.terminals
        }
        self.terminals = [
            Terminal(
                t.x, t.y, t.z, yaw=t.yaw, terminal_id=t.terminal_id, bus=bus,
                faction=t.faction, intel=t.intel, secured=t.secured, critical=t.critical,
                door_ids=doors_of[t.terminal_id], hardened=t.hardened, style=t.style,
                notes=t.notes,
            )
            for t in self.level.terminals
        ]  # fmt: skip
        self.exit_zone = self.level.exit_zone
        # the level's key computer (the datacore - ``TerminalData.objective``), if it has one
        objectives = [
            t for t, d in zip(self.terminals, self.level.terminals, strict=True) if d.objective
        ]
        self.objective_terminal: Terminal | None = objectives[0] if objectives else None
        self.cameras = [
            SecurityCamera(
                c.x, c.y, c.z, c.mount_yaw, c.sweep_angle, c.sweep_speed,
                camera_id=c.camera_id or f"camera_{i + 1}", faction=c.faction,
                watch_area=self.level.private_area(c.watch_area) if c.watch_area else None,
            )
            for i, c in enumerate(self.level.cameras)
        ]  # fmt: skip
        self.panels = [
            AccessPanel(
                pd.x, pd.y, pd.z, pd.door_id, yaw=pd.yaw, panel_id=pd.panel_id,
                faction=pd.faction, name=pd.name, bus=bus,
            )
            for pd in self.level.panels
        ]  # fmt: skip
        for panel in self.panels:
            panel.door = self.room.doors.get(panel.door_id)
        self.pickups = [
            Pickup(pk.pickup_id, pk.item_id, pk.x, pk.y, pk.z, pk.yaw) for pk in self.level.pickups
        ]
        self.vending_machines = [
            VendingMachine(
                v.x, v.y, v.z, yaw=v.yaw, bus=bus,
                machine_id=v.machine_id or f"vending_{i + 1}", faction=v.faction,
            )
            for i, v in enumerate(self.level.vending_machines)
        ]  # fmt: skip
        self.guards = [
            CyberGuard(
                g.x, g.y, g.z, self.room.nav_bounds,
                patrol_z_min=g.patrol_z_min, patrol_z_max=g.patrol_z_max,
                wall_pos=g.wall_pos, wall_yaw=g.wall_yaw, initial_direction=g.initial_direction,
                guard_id=g.guard_id or f"guard_{i + 1}", faction=g.faction, essential=g.essential,
                patrol_route=g.patrol_route,
            )
            for i, g in enumerate(self.level.guards)
        ]  # fmt: skip
        # The street's shared life (orders at the counters, things lying about) - for the
        # people with a role to play (see ai.alley)
        self.life: AlleyLife | None = AlleyLife(self.level) if self.level.places else None
        if self.life is not None:
            self.life.clock = self.clock
            life = self.life
            # a noise in the street (the vending machine shorting out): whoever cares hears it
            self.bus.subscribe(NoiseEmitted, lambda e: life.hear(e.x, e.z, e.source))
        self.civilians = [
            Civilian(
                c.x, c.y, c.z, self.room.nav_bounds, c.waypoints, seed=c.seed or i + 1,
                civilian_id=c.civilian_id or f"civilian_{i + 1}", role=c.role, life=self.life,
            )
            for i, c in enumerate(self.level.civilians)
        ]  # fmt: skip
        self.merchants = [
            Merchant(
                m.x, m.y, m.z, self.room.nav_bounds, m.face_yaw, style=m.style, seed=m.seed,
                merchant_id=m.merchant_id or f"merchant_{i + 1}", role=m.role, life=self.life,
            )
            for i, m in enumerate(self.level.merchants)
        ]  # fmt: skip
        for npc in self.npcs:
            npc.bind(bus)
        # the traffic on the road (``entities.traffic``): cars coming and going
        self.traffic: Traffic | None = (
            Traffic(self.level.lanes, clock=self.clock) if self.level.lanes else None
        )
        # where the puddles gather when it rains (out of doors)
        self.puddles: list[PuddleSite] = puddle_sites(level) if level.outdoor else []
        # the metro (``entities.metro``): trains stopping at the station
        self.metro: Metro | None = (
            Metro(self.level.metro, clock=self.clock) if self.level.metro is not None else None
        )
        if self.metro is not None:  # (the ticket gates' locks, for the flaps to show)
            self.metro.gate_doors = {
                k: self.room.doors[d]
                for k, d in enumerate(self.metro.data.gate_doors)
                if d in self.room.doors
            }
        if self.life is not None:
            self.life.doors = self.room.doors
            self.life.machines = {v.machine_id: v for v in self.vending_machines}
            self.life.metro = self.metro
        # homes and their owners: the owner minds who is in there
        for area in self.level.private_areas:
            owner = self.npc_with_role(area.owner_role) if area.owner_role else None
            if owner is not None:
                owner.watch_area = area
        # a conversation in progress, remarks shouted in passing, notices on the screen
        self.conversation: Conversation | None = None
        self.barks: list[Bark] = []
        self.toasts: list[list] = []  # [text, seconds left]
        self.fade_in = 0.0  # seconds left of fading in from black (caught, or a new level)
        self.fade_in_total = SIMULATION.fade_in  # (what ``fade_in`` started from)
        # Walking out of the level's last door: the level it leads to, and how long the
        # picture has been fading to black (``level_fade_out``) before it is loaded
        self.leaving: str | None = None
        self.leave_timer = 0.0
        self.talk_cooldown = 0.0
        self.caught_area: PrivateArea | None = None
        self.caught_by = ""  # who caught the netrunner (for the red screen)

        # Netrunning: everything electronic is a hack target on its faction's network.
        self.hacking = HackingSystem(
            bus, self.player, self.hack_targets, self.level.resolved_factions()
        )

        self.state = GameState.PLAYING
        self.state_timer = 0.0
        # After [Q] leaves a terminal, [E] must be let go before it counts again: a breached
        # terminal reconnects instantly, so a still-held key re-jacked the player next frame.
        self.interact_needs_release = False
        self.obstacles: list[AABB] = self._collect_obstacles()
        self.player_obstacles: list[AABB] = self.obstacles
        self.nav_obstacles: list[AABB] = self._collect_nav_obstacles()
        self.sight_blockers: list[AABB] = self._collect_sight_blockers()
        self.hacking.update(self.viewpoint, 0.0, self.sight_blockers)

    def load_level(self, level: LevelData) -> None:
        """Swaps the world out for another level (the player keeps what they have learned
        and carries; everything else starts fresh). Renderers watch ``level_generation``."""
        self.paused = False
        self._build(level, keep_intel=True)
        self.level_generation += 1

    def new_game(self) -> None:
        """A new game: back to the first level, nothing carried over (no inventory beyond
        what the netrunner starts with, no quests)."""
        self.paused = False
        self.quests = QuestLog()
        self.director = QuestDirector(self)
        self.time = 0.0
        self.clock.set(CLOCK.new_game_at)
        self.weather = Weather()
        if self.level is self.first_level:
            self.reset()
        else:
            self._build(self.first_level)
            self.level_generation += 1

    restart_run = new_game  # (the old name)

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Restarts the *current* level: the spawn point, fresh guards, clean deck."""
        self.paused = False
        self.player.reset()
        for entity in (
            *self.terminals, *self.cameras, *self.vending_machines, *self.npcs, *self.panels,
            *self.pickups,
        ):  # fmt: skip
            entity.reset()
        self.room.reset()
        self.conversation = None
        self.barks = []
        self.toasts = []
        self.caught_area = None
        self.caught_by = ""
        self.leaving = None
        self.leave_timer = 0.0
        if self.life is not None:
            self.life.reset()
        if self.traffic is not None:
            self.traffic.reset()
        if self.metro is not None:
            self.metro.reset()
        self.detection.reset()
        self.hacking.reset()
        self.bus.clear()
        self.state = GameState.PLAYING
        self.state_timer = 0.0
        self.interact_needs_release = False
        self.obstacles = self._collect_obstacles()
        self.player_obstacles = self.obstacles
        self.nav_obstacles = self._collect_nav_obstacles()
        self.sight_blockers = self._collect_sight_blockers()
        self.hacking.update(self.viewpoint, 0.0, self.sight_blockers)

    @property
    def npcs(self) -> list[BaseNPC]:
        return [*self.guards, *self.civilians, *self.merchants]

    @property
    def items(self) -> list[LooseItem]:
        """Small things lying about the street (a knife on a board, a parcel on a step)."""
        return self.life.items if self.life is not None else []

    @property
    def hack_targets(self) -> list:
        """Every electronic thing the scanner can find: computers, cameras, vending machines,
        doors (a door with an access panel is hacked through the panel), access panels and
        people with cyberware (the guards)."""
        paneled = {p.door_id for p in self.panels}
        return [
            *self.terminals,
            *self.cameras,
            *self.vending_machines,
            *(
                d
                for d in self.room.doors.values()
                if d.door_id not in paneled and (d.data is None or d.data.hackable)
            ),
            *self.panels,
            *self.guards,
        ]

    def npc_with_role(self, role: str):
        return next((n for n in self.npcs if getattr(n, "role", "") == role), None)

    def toast(self, text: str, seconds: float = TOAST_SECONDS) -> None:
        """A notice across the screen for a few seconds (a new quest, an item received)."""
        self.toasts.append([text, seconds])

    def say(self, speaker: str, text: str) -> None:
        """A remark in passing, captioned (no answer expected)."""
        self.barks = [b for b in self.barks if b.line.speaker != speaker]
        self.barks.append(Bark(Line(speaker, text)))

    # ------------------------------------------------------------------ interaction
    @property
    def talk_target(self):
        """The person in front of the netrunner whom [E] would talk to, if any."""
        if self.hacking.session is not None or self.hijacked_camera is not None:
            return None
        if self.talk_cooldown > 0.0 or self.conversation is not None:
            return None
        p = self.player
        best = None
        for npc in self.npcs:
            if getattr(npc, "hidden", False) or getattr(npc, "asleep", False):
                continue
            if not self.director.can_talk(npc):
                continue
            d = math.hypot(npc.x - p.x, npc.z - p.z)
            if d > TALK_RANGE:
                continue
            off = abs(wrap_angle_deg(math.degrees(math.atan2(npc.z - p.z, npc.x - p.x)) - p.yaw))
            if off <= TALK_CONE and (best is None or d < best[0]):
                best = (d, npc)
        return best[1] if best else None

    @property
    def pickup_in_reach(self) -> Pickup | None:
        """The thing lying about that [E] would pick up, if any."""
        if self.hacking.session is not None or self.hijacked_camera is not None:
            return None
        view = self.player.viewpoint()
        found = [
            (a, pk) for pk in self.pickups if (a := pk.in_reach(view.eye, view.forward)) is not None
        ]
        return min(found, key=lambda f: f[0])[1] if found else None

    @property
    def prompt(self) -> str:
        """What [E] does right now, for the HUD ("" = nothing special)."""
        if self.conversation is not None:
            return ""
        npc = self.talk_target
        if npc is not None:
            return f"[E] TALK TO {NAMES.get(getattr(npc, 'role', ''), 'THEM')}"
        pk = self.pickup_in_reach
        if pk is not None:
            return f"[E] TAKE {pk.name}"
        return ""

    def take(self, pickup: Pickup) -> None:
        pickup.taken = True
        self.player.inventory.add(pickup.item_id)
        self.toast(f"PICKED UP: {pickup.name}")
        self.hacking.log.append(f"PICKED UP: {pickup.name}")

    def start_conversation(self, npc) -> bool:
        conv = self.director.conversation_for(npc)
        if conv is None:
            return False
        self.conversation = conv
        npc.talking_to = self.player
        return True

    def _update_talk(self, dt: float, inp: InputState) -> None:
        conv = self.conversation
        if conv is not None:
            if inp.interact_pressed:
                conv.advance()
            conv.update(dt)
            partner = conv.partner
            too_far = (
                partner is not None
                and math.hypot(partner.x - self.player.x, partner.z - self.player.z)
                > TALK_RANGE + 1.5
            )
            if conv.finished or too_far:
                if partner is not None:
                    partner.talking_to = None
                self.conversation = None
                self.talk_cooldown = TALK_COOLDOWN  # (a last [E] does not start it over)
        self.talk_cooldown = max(0.0, self.talk_cooldown - dt)
        for bark in self.barks:
            bark.update(dt)
        self.barks = [b for b in self.barks if not b.done]
        for t in self.toasts:
            t[1] -= dt
        self.toasts = [t for t in self.toasts if t[1] > 0.0]

    def trespassing_in(self) -> PrivateArea | None:
        p = self.player
        return next((a for a in self.level.private_areas if a.contains(p.x, p.z)), None)

    def evict(self) -> list[str]:
        """Thrown out of the home the netrunner was caught in: onto the street outside it,
        without anything of the owner's they had taken (it goes back where it was), the
        door locked behind them. Returns the ids of what was taken off them."""
        area = self.caught_area or (
            self.level.private_areas[0] if self.level.private_areas else None
        )
        lost: list[str] = []
        p = self.player
        if area is not None:
            for item_id in area.confiscate:
                n = p.inventory.count(item_id)
                if n:
                    p.inventory.remove(item_id, n)
                    lost.append(item_id)
                    for pk in self.pickups:
                        if pk.item_id == item_id:
                            pk.restore()
            p.x, p.z, p.yaw = area.exit
            door = self.room.doors.get(area.door_id) if area.door_id else None
            if door is not None:
                door.lock()
                door.open_amount = 0.0
        p.crouching = False
        p.pitch = 0.0
        self.hacking.end_session()
        self.hacking.disconnect()
        self.hacking.cancel_upload()
        self.detection.reset()
        self.conversation = None
        for npc in self.npcs:
            npc.talking_to = None
            if hasattr(npc, "alarm"):
                npc.alarm = 0.0
        self.director.on_evicted(lost)
        name = area.name if area is not None else "THERE"
        note = f"THROWN OUT OF {name}"
        if lost:
            note += " - " + ", ".join(item(i).name for i in lost) + " TAKEN BACK"
        self.toast(note, 6.0)
        self.hacking.log.append(note)
        return lost

    @property
    def scanner(self) -> Scanner:
        return self.hacking.scanner

    @property
    def hijacked_camera(self) -> SecurityCamera | None:
        return self.hacking.hijacked_camera

    @property
    def viewpoint(self) -> Viewpoint:
        """Where the player is looking from: their own eyes, or a hijacked camera's lens."""
        if self.camera_override is not None:
            return self.camera_override
        cam = self.hacking.hijacked_camera
        return cam.viewpoint if cam is not None else self.player.viewpoint()

    @property
    def hud_terminal(self) -> Terminal | None:
        """The open terminal interface, or the one nearest the player (drives the HUD)."""
        session = self.hacking.session
        if session is not None:
            return session
        if not self.terminals:
            return None
        p = self.player
        return min(self.terminals, key=lambda t: math.hypot(t.x - p.x, t.z - p.z))

    @property
    def at_exit(self) -> bool:
        return self.exit_zone is not None and self.exit_zone.contains_xz(
            self.player.x, self.player.z
        )

    @property
    def keyed_door_in_reach(self) -> tuple[SlidingDoor, bool] | None:
        """The nearest encryption-keyed door within interaction range, and whether the
        player carries its cipher (drives the HUD prompt)."""
        p = self.player
        best = None
        for door in self.room.doors.values():
            if door.keyed and door.distance_to(p.x, p.z) <= INTERACT_RANGE + 0.5:
                d = door.distance_to(p.x, p.z)
                if best is None or d < best[0]:
                    best = (d, door)
        if best is None:
            return None
        door = best[1]
        return door, door.key_intel in p.intel

    def _unlock_keyed_doors(self) -> None:
        """Encryption-keyed doors read the cipher off the netrunner's deck as they approach."""
        p = self.player
        for door in self.room.doors.values():
            if (
                door.keyed
                and not door.unlocked
                and door.key_intel in p.intel
                and door.distance_to(p.x, p.z) <= INTERACT_RANGE
            ):
                door.unlock()
                self.hacking.log.append(f"CIPHER ACCEPTED: {door.display_name} UNLOCKED")

    def _entity_bounds(self) -> list[AABB]:
        obs = [t.get_bounds() for t in self.terminals]
        obs.extend(v.get_bounds() for v in self.vending_machines)
        return obs

    def _collect_obstacles(self) -> list[AABB]:
        """What blocks movement: entity footprints plus the level's colliders."""
        return self._entity_bounds() + self.room.get_obstacles()

    def _collect_nav_obstacles(self) -> list[AABB]:
        """What ground-walking NPCs route around: the obstacles plus the level's NPC-only
        no-go boxes (the staircases - NPCs cannot climb, so without these a guard chasing
        a last sighting on the stairs would walk straight into the solid steps)."""
        return self._entity_bounds() + self.room.get_nav_obstacles()

    def _collect_sight_blockers(self) -> list[AABB]:
        """What blocks vision: opaque entities and colliders plus walkable occluders."""
        return self._entity_bounds() + self.room.get_sight_blockers()

    # ------------------------------------------------------------------ step
    def update(self, raw_dt: float, inp: InputState) -> None:
        if inp.toggle_pause:
            self.paused = not self.paused
        if self.paused:
            # A paused game consumes no time at all: nothing moves, no timer ticks, no
            # animation advances, and the frame the player resumes on picks up exactly
            # where it left off. Only [R] (restart) is honoured while paused.
            if inp.reset:
                self.reset()
            return

        dt = clamp_dt(raw_dt, SIMULATION.max_dt)
        self.time += dt
        self.clock.advance(dt)
        self.weather.update(dt, self.clock)

        if inp.reset:
            self.reset()
            return

        if self.fade_in > 0.0:
            self.fade_in = max(0.0, self.fade_in - dt)
        if self.state is GameState.PLAYING and self.leaving is not None:
            self._update_leaving(dt)
        elif self.state is GameState.PLAYING:
            self._update_playing(dt, inp)
        elif self.state is GameState.CAUGHT:
            self.state_timer += dt
            if self.state_timer >= self.caught_hold + SIMULATION.caught_fade:
                self._pay_for_being_caught()

    @property
    def caught_hold(self) -> float:
        """How long the red "you were caught" screen stays up before fading to black."""
        return SIMULATION.evict_hold if self.level.on_caught == "evict" else SIMULATION.caught_hold

    @property
    def caught_rule(self) -> str:
        return self.level.on_caught

    def _pay_for_being_caught(self) -> None:
        """The red screen has faded to black: a new game (the facility - escape the
        biodome uncaught, or start over), or thrown out (somebody's home in the city)."""
        if self.level.on_caught == "evict":
            self.evict()
            self.state = GameState.PLAYING
            self.state_timer = 0.0
        else:
            self.new_game()
        self.fade_in = self.fade_in_total = SIMULATION.fade_in

    def _update_leaving(self, dt: float) -> None:
        """At the level's last door: the netrunner stands at it while the world goes on
        and the picture fades to black; then the next level is loaded (the deck comes
        along) and the picture fades back in on it."""
        self.leave_timer += dt
        self._update_playing(dt, InputState())  # (hands off: no walking back, no hacks)
        if self.leave_timer < SIMULATION.level_fade_out:
            return
        target = self.leaving
        self.leaving = None
        self.leave_timer = 0.0
        if target is not None:
            self.load_level(LEVELS[target]())
        self.fade_in = self.fade_in_total = SIMULATION.level_fade_in

    @property
    def screen_black(self) -> float:
        """How far the picture has faded to black (0 = not at all, 1 = black): walking out
        of a level's last door, and fading back in on the next level (or after being
        caught - the red "caught" screen draws its own fade out)."""
        if self.leaving is not None:
            return min(1.0, self.leave_timer / max(SIMULATION.level_fade_out, 1e-6))
        if self.fade_in > 0.0:
            return min(1.0, self.fade_in / max(self.fade_in_total, 1e-6))
        return 0.0

    def _shove_people(self, dx: float, dz: float) -> bool:
        """The netrunner walking into somebody pushes them out of the way (the people of
        the street: ``StreetPerson.shove``) - somewhere they can stand: clear of the walls
        and furniture, on the floor, not into somebody else - so nobody can pin the
        netrunner in a corner or against a sofa. Returns whether anybody was moved."""
        if abs(dx) + abs(dz) < 1e-9:
            return False
        p = self.player
        nx, nz = p.x + dx, p.z + dz
        feet, top = p.floor_y, p.floor_y + p.body_height
        moved = False
        for npc in self.npcs:
            if not getattr(npc, "shovable", False) or not npc.blocks_player:
                continue
            if not npc.body_box().overlaps_cylinder(nx, nz, p.radius, feet, top):
                continue

            def clear(x: float, z: float, me=npc) -> bool:
                r = me.collision_radius
                if not self.room.is_valid_position(x, z, r * 0.6):
                    return False
                if me._body_collides(x, z, self.obstacles):
                    return False
                return all(
                    o is me
                    or getattr(o, "hidden", False)
                    or o.fall_angle
                    or math.hypot(o.x - x, o.z - z) >= (o.collision_radius + r) * 0.85
                    for o in self.npcs
                )

            moved |= npc.shove(nx, nz, p.radius, dx, dz, clear)
        return moved

    def _get_caught(self) -> None:
        self.state = GameState.CAUGHT
        self.state_timer = 0.0
        self.caught_area = self.trespassing_in()
        watchers = [c.display_name for c in self.cameras if c.is_alerted]
        watchers += [
            NAMES.get(n.role, n.role.upper())
            for n in self.civilians
            if getattr(n, "alarm", 0.0) > 0
        ]
        watchers += [g.display_name for g in self.guards if g.time_since_seen == 0.0]
        self.caught_by = ", ".join(watchers)
        self.hacking.end_session()
        self.hacking.disconnect()
        self.conversation = None

    def _update_playing(self, dt: float, inp: InputState) -> None:
        self.obstacles = self._collect_obstacles()
        self.nav_obstacles = self._collect_nav_obstacles()
        self.sight_blockers = self._collect_sight_blockers()
        p = self.player

        if not inp.interact:
            self.interact_needs_release = False
        if inp.disconnect:
            if self.hacking.session is not None:
                self.hacking.end_session()
                self.interact_needs_release = inp.interact
            else:
                self.hacking.disconnect()
        if inp.cycle_camera:
            self.hacking.cycle_camera(inp.cycle_camera)
        cam = self.hacking.hijacked_camera
        # People are solid to the player (you cannot walk through a passer-by); the NPCs
        # themselves route on the static obstacles and steer around each other.
        self.player_obstacles = self.obstacles + [
            n.body_box()
            for n in self.npcs
            if n.blocks_player and n.fall_angle == 0.0 and not getattr(n, "hidden", False)
        ]
        self.detection.gain_scale = p.stat(DETECTION_GAIN)
        if self.hacking.session is not None and cam is None:
            # Jacked into a computer at the desk: the hands are busy, the eyes are free.
            self.controller.look(inp.look_dx, inp.look_dy)
            self.controller.move(0.0, 0.0, dt, self.player_obstacles)
            interact = False
        elif cam is not None:
            # Behind a camera lens: the mouse steers the head (pan and tilt); the body stays
            # put (and stays visible to anyone who walks past it).
            cam.steer(
                -inp.look_dx * PLAYER.mouse_sensitivity, inp.look_dy * PLAYER.mouse_sensitivity
            )
            self.controller.move(0.0, 0.0, dt, self.player_obstacles)
            interact = False
        else:
            if inp.toggle_crouch:
                self.controller.toggle_crouch()
            self.controller.look(inp.look_dx, inp.look_dy)
            dx, dz = self.controller.displacement(inp.forward, inp.strafe, dt)
            if self._shove_people(dx, dz):
                self.player_obstacles = self.obstacles + [
                    n.body_box()
                    for n in self.npcs
                    if n.blocks_player and n.fall_angle == 0.0 and not getattr(n, "hidden", False)
                ]
            self.controller.move_by(dx, dz, dt, self.player_obstacles)
            interact = inp.interact and not self.interact_needs_release
            # [E] pressed: a line of dialogue skipped, somebody talked to, a thing picked up
            # (and then [E] must be let go before it works a terminal)
            if inp.interact_pressed:
                used = False
                if self.conversation is not None:
                    used = True  # (advanced in _update_talk)
                elif (npc := self.talk_target) is not None:
                    used = self.start_conversation(npc)
                elif (pk := self.pickup_in_reach) is not None:
                    self.take(pk)
                    used = True
                if used:
                    self.interact_needs_release = True
                    interact = False
            elif self.conversation is not None:
                interact = False

        closeup = p.stat(CLOSEUP_HACK_SPEED)
        for t in self.terminals:
            if t.update(p.x, p.z, interact, dt, rate_scale=closeup):
                self.hacking.connect(t)  # the link completed: the interface opens
        for v in self.vending_machines:
            v.update(p.x, p.z, interact, dt)

        # Netrunning. A number key drives the open terminal interface when there is one,
        # otherwise it runs a quickhack on last frame's focus - what the HUD showed when
        # the key was pressed. The focus is recomputed below for this frame's view.
        if inp.hack_slot:
            if self.hacking.session is not None:
                self.hacking.session_request(inp.hack_slot)
            else:
                self.hacking.request(inp.hack_slot)
        if inp.scan:
            self.hacking.scan(self.viewpoint.eye)
        self.hacking.update(self.viewpoint, dt, self.sight_blockers)

        # Deliver everything the interactables announced (noises, hacks) before the world
        # and the AI react to it, so reactions land in the same frame as the cause.
        self.bus.dispatch()
        self._unlock_keyed_doors()
        walkers = [
            (n.x, n.z)
            for n in self.npcs
            if not getattr(n, "is_down", False) and not getattr(n, "hidden", False)
        ]
        self.room.update(dt, [(p.x, p.z), *walkers])

        if self.life is not None:
            self.life.update(dt)
        if self.traffic is not None:
            self.traffic.update(dt)
        if self.puddles:  # (cars splashing through the puddles on the road)
            self.weather.spray(self.traffic.cars if self.traffic else (), self.puddles, dt)
        if self.metro is not None:
            self.metro.update(dt)
        for panel in self.panels:
            panel.update(dt)
        self._update_talk(dt, inp)
        self.detection.begin_frame()
        for camera in self.cameras:
            camera.update(p, dt, self.obstacles, self.detection, sight_blockers=self.sight_blockers)
        npcs = self.npcs
        for npc in npcs:
            npc.update(
                p,
                dt,
                obstacles=self.nav_obstacles,
                detection=self.detection,
                sight_blockers=self.sight_blockers,
                others=npcs,
            )
        self.detection.end_frame(dt)
        for npc in self.civilians:  # somebody shouted something (an owner finding a stranger)
            shout = getattr(npc, "bark", "")
            if shout:
                self.say(NAMES.get(npc.role, npc.role.upper()), shout)
                npc.bark = ""
        self.director.update()

        # Transitions (evaluated every frame, not only while moving - bug B3)
        if self.leaving is not None:
            return  # (on the threshold: fading out, nobody catches them now)
        if self.at_exit and self.level.exit_to is not None:
            # at the last door: the picture fades to black, then the next level loads
            self.leaving = self.level.exit_to
            self.leave_timer = 0.0
            self.hacking.end_session()
            self.hacking.disconnect()
            return
        if self.detection.is_compromised:
            self._get_caught()
