"""CyberGuard: a patrolling guard built from small, independent behaviour states.

    PACING ──(pacing_duration)──▶ WALKING_TO_WALL ──(arrived)──▶ LEANING
      ▲                                                              │ (lean_duration)
      └──────────(arrived)────── WALKING_TO_PATROL ◀─────────────────┘
                                        ▲
    NoiseEmitted ──▶ INVESTIGATING ──(arrived)──▶ INSPECTING ──(inspect duration)──────┘
                                                  (gesture: stare, or hit the machine)

    Guards given a ``patrol_route`` walk it waypoint by waypoint (PATROL_ROUTE) instead of
    PACING a lane; ``wall_pos=None`` skips the wall lean altogether.

    sighting ──▶ ALERTED (stop, look) ──(meter > threshold)──▶ SUSPICIOUS (creep to last seen)
                    │ lost sight                                  │ lost sight   │ arrived
                    ▼                                             ▼              ▼
               resume routine ◀────────────────────────── resume ◀── LOOKING_AROUND

    quickhack BLIND ──▶ BLINDED (stands, cannot see) ──(blind_duration)──▶ resume routine
    quickhack ELECTROCUTE ──▶ KNOCKED_OUT (down for good; only reset() revives)

Each state is ~20 lines; the guard class itself only holds configuration, the machine, and
perception. To add a behaviour, add a ``State`` subclass - nothing else needs to change.
"""

from __future__ import annotations

import math
import random
import zlib
from collections.abc import Callable, Sequence
from dataclasses import replace

from cyberduke.ai.fsm import State, StateMachine
from cyberduke.ai.npc import BaseNPC, HumanoidSpec
from cyberduke.ai.pathfinding import Bounds
from cyberduke.animation import (
    HumanoidPose,
    bang_machine_animation,
    blinded_animation,
    electrocuted_animation,
    inspect_weapon_animation,
    investigate_animation,
)
from cyberduke.animation.look import guard_look
from cyberduke.animation.ragdoll import BodyState, Collapse
from cyberduke.config import DETECTION, GUARD, HACKING, GuardConfig
from cyberduke.core.events import EventBus, NoiseEmitted
from cyberduke.core.geometry import AABB, Vec3, yaw_towards
from cyberduke.core.interp import clamp, lerp
from cyberduke.core.inventory import starting_inventory
from cyberduke.entities.player import Player
from cyberduke.systems.detection import SightingSink
from cyberduke.world.level import DEFAULT_FACTION

GUARD_SPEC = HumanoidSpec(
    has_visor=True, has_core_glow=True, has_holster=True, prop="laser_blaster"
)
ELITE_GUARD_SPEC = HumanoidSpec(
    has_visor=True,
    has_core_glow=True,
    has_holster=True,
    prop="laser_blaster",
    visor_color=(1.0, 0.75, 0.1),  # gold visor: hardened cyberware, cannot be put down
    hair_color=(0.05, 0.05, 0.06),
)


def guard_spec(guard_id: str, elite: bool = False) -> HumanoidSpec:
    """One guard's look: the Halcyon uniform - armour, visor, sidearm, the chrome right
    arm - with their own hair, rank stripes, trousers, a jaw plate on some and the visor
    a slightly different red (an elite's is gold) (``animation.look.guard_look``)."""
    seed = sum(ord(c) * (k + 1) for k, c in enumerate(guard_id))
    look, palette, visor = guard_look(seed, elite)
    base = ELITE_GUARD_SPEC if elite else GUARD_SPEC
    skin = ((1.0, 1.0, 1.0), (0.85, 0.72, 0.6), (0.62, 0.45, 0.32), (0.75, 0.6, 0.5))
    return replace(
        base, look=look, hair_color=palette.hair, hair_color2=palette.hair2,
        pants_color=palette.pants, neon_color=palette.neon, visor_color=visor,
        skin_color=skin[seed % len(skin)],
    )  # fmt: skip


Point = tuple[float, float]


# =============================================================================== states
class GuardState(State["CyberGuard"]):
    """Base for guard states. ``interruptible`` states are abandoned for ALERTED the moment
    the guard sees the player; the reaction states handle sightings themselves."""

    interruptible = True


class Pacing(GuardState):
    """Walks back and forth along the patrol lane (x locked to ``home_x``)."""

    name = "PACING"

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        # Back onto the lane at walking pace: a guard returning from the wall or a noise
        # arrives within its stop distance of the lane, and snapping it across was a
        # visible sideways jump.
        step = g.walk_speed * dt
        g.x += clamp(g.home_x - g.x, -step, step)
        new_z = g.z + g.walk_speed * g.patrol_dir * dt

        if new_z > g.patrol_z_max:
            g.patrol_dir = -1.0
            g.target_yaw = 180.0
        elif new_z < g.patrol_z_min:
            g.patrol_dir = 1.0
            g.target_yaw = 0.0
        elif g.frame_obstacles and g._body_collides(g.home_x, new_z, g.frame_obstacles):
            g.patrol_dir *= -1.0
            g.target_yaw = 0.0 if g.patrol_dir > 0 else 180.0
        else:
            g.z = new_z

        g.advance_walk_cycle(dt)
        g.rotate_towards(g.target_yaw, dt)

        if g.wall_pos is not None and g.fsm.time_in_state > g.cfg.pacing_duration:
            return WalkingToWall()
        return None


class PatrolRoute(GuardState):
    """Walks a closed loop of waypoints (A* between them), lingering briefly at each. The
    maze guards use this instead of a straight lane."""

    name = "PATROL_ROUTE"

    def __init__(self, index: int | None = None) -> None:
        self.index = index
        self.linger = 0.0

    def enter(self, g: CyberGuard) -> None:
        if self.index is None:
            self.index = g.route_index
        g.route_index = self.index
        g._clear_route()

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        if not g.patrol_route:
            return Pacing()
        if self.linger > 0.0:
            self.linger -= dt
            g.pose = HumanoidPose(head_yaw=math.sin(g.fsm.time_in_state * 2.0) * 20.0)
            if self.linger <= 0.0:
                g.route_index = (g.route_index + 1) % len(g.patrol_route)
                g._clear_route()
            return None
        wx, wz = g.patrol_route[g.route_index]
        if g.navigate_towards(wx, wz, dt, g.frame_obstacles, stop_dist=g.cfg.noise_stop_distance):
            self.linger = g.cfg.route_pause
        if g.wall_pos is not None and g.fsm.time_in_state > g.cfg.pacing_duration:
            return WalkingToWall()
        return None


class WalkingToWall(GuardState):
    """A* route to the lean spot."""

    name = "WALKING_TO_WALL"

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        if g.wall_pos is None:
            return g.patrol_state()
        wx, wz = g.wall_pos
        if g.navigate_towards(wx, wz, dt, g.frame_obstacles, stop_dist=g.cfg.wall_stop_distance):
            return Leaning()
        return None


class Leaning(GuardState):
    """Leans on the wall and runs the weapon-inspection routine for ``lean_duration``."""

    name = "LEANING"

    def enter(self, g: CyberGuard) -> None:
        g.target_yaw = g.wall_yaw

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        g.rotate_towards(g.target_yaw, dt)
        g.pose = inspect_weapon_animation(g.fsm.time_in_state, g.cfg.lean_duration)
        if g.fsm.time_in_state > g.cfg.lean_duration:
            return WalkingToPatrol()
        return None


class WalkingToPatrol(GuardState):
    """Returns to the patrol: the nearest point of the lane, or the next route waypoint."""

    name = "WALKING_TO_PATROL"

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        if g.patrol_route:
            return PatrolRoute(g.nearest_route_index())
        target_z = clamp(g.z, g.patrol_z_min, g.patrol_z_max)
        if g.navigate_towards(
            g.home_x, target_z, dt, g.frame_obstacles, stop_dist=g.cfg.wall_stop_distance
        ):
            g.target_yaw = 0.0 if g.patrol_dir > 0 else 180.0
            return Pacing()
        return None


class Investigating(GuardState):
    """Walks to a noise's investigation point."""

    name = "INVESTIGATING"

    def __init__(
        self,
        target: Point,
        source: Point | None,
        duration: float | None = None,
        gesture: str = "look",
    ) -> None:
        self.target = target
        self.source = source
        self.duration = duration
        self.gesture = gesture

    def enter(self, g: CyberGuard) -> None:
        g.investigate_target = self.target
        g.investigate_source = self.source
        g._clear_route()  # force a fresh plan toward the new goal

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        tx, tz = self.target
        if g.navigate_towards(tx, tz, dt, g.frame_obstacles, stop_dist=g.cfg.noise_stop_distance):
            return Inspecting(self.source, self.duration, self.gesture)
        return None


class Inspecting(GuardState):
    """Stands at the noise site facing the source for ``duration`` seconds (the noise
    decides how long it is worth; default ``cfg.investigate_duration``). Gesture ``"look"``
    sways and scans; ``"bang"`` hits the thing for ``cfg.bang_duration`` first, then stares."""

    name = "INSPECTING"

    def __init__(
        self, source: Point | None, duration: float | None = None, gesture: str = "look"
    ) -> None:
        self.source = source
        self.duration = duration
        self.gesture = gesture

    def enter(self, g: CyberGuard) -> None:
        if self.source is not None:
            g.target_yaw = yaw_towards(g.x, g.z, *self.source)
        if self.duration is None:
            self.duration = g.cfg.investigate_duration

    def is_banging(self, g: CyberGuard) -> bool:
        return self.gesture == "bang" and g.fsm.time_in_state < min(
            self.duration or 0.0, g.cfg.bang_duration
        )

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        t = g.fsm.time_in_state
        if self.is_banging(g):
            g.rotate_towards(g.target_yaw, dt)  # squared up to the machine
            g.pose = bang_machine_animation(t)
        else:
            sway = math.sin(t * g.cfg.scan_rate) * g.cfg.scan_amplitude_deg
            g.rotate_towards(g.target_yaw + sway, dt)
            g.pose = investigate_animation(t)
        if t >= self.duration:
            return WalkingToPatrol()
        return None

    def exit(self, g: CyberGuard) -> None:
        g.investigate_target = None
        g.investigate_source = None


class Alerted(GuardState):
    """Stops and looks at where the player was seen. Escalates to SUSPICIOUS once the
    detection meter passes the threshold; gives up after a short grace without a sighting."""

    name = "ALERTED"
    interruptible = False

    def __init__(self, resume: Callable[[], GuardState]) -> None:
        self.resume = resume

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        if g.last_seen is not None:
            g.target_yaw = yaw_towards(g.x, g.z, *g.last_seen)
        g.rotate_towards(g.target_yaw, dt)
        g.pose = HumanoidPose()  # head up, arms at rest: fully attentive
        if g.detection_level >= g.cfg.suspicion_threshold:
            return Suspicious(self.resume)
        if g.time_since_seen > g.cfg.alert_grace:
            return self.resume()
        return None


class Suspicious(GuardState):
    """Creeps toward the last place the player was seen (re-targeting while they remain in
    view). Resumes the routine if sight is lost for a while; looks around on arrival."""

    name = "SUSPICIOUS"
    interruptible = False

    def __init__(self, resume: Callable[[], GuardState]) -> None:
        self.resume = resume

    def enter(self, g: CyberGuard) -> None:
        g.speed_factor = g.cfg.suspicious_speed_factor
        g._clear_route()

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        if g.time_since_seen > g.cfg.suspicion_grace:
            return self.resume()
        if g.last_seen is None:
            return self.resume()
        tx, tz = g.last_seen
        if g.navigate_towards(tx, tz, dt, g.frame_obstacles, stop_dist=g.cfg.noise_stop_distance):
            return LookingAround(self.resume)
        return None

    def exit(self, g: CyberGuard) -> None:
        g.speed_factor = 1.0


class LookingAround(GuardState):
    """Stands at the last-seen spot scanning. Seeing the player again re-alerts."""

    name = "LOOKING_AROUND"

    def __init__(self, resume: Callable[[], GuardState]) -> None:
        self.resume = resume

    def enter(self, g: CyberGuard) -> None:
        g.resume_after_look = self.resume

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        t = g.fsm.time_in_state
        sway = math.sin(t * g.cfg.scan_rate) * g.cfg.scan_amplitude_deg
        g.rotate_towards(g.target_yaw + sway, dt)
        g.pose = HumanoidPose(head_yaw=math.sin(t * 2.0) * 25.0)
        if t >= g.cfg.look_around_duration:
            return self.resume()
        return None

    def exit(self, g: CyberGuard) -> None:
        g.resume_after_look = None


class Blinded(GuardState):
    """Optics jammed by a quickhack: stands where it is, hands over the visor, sees nothing.
    When the static clears it has lost track of whatever it was doing and goes back to
    its routine (``resume``)."""

    name = "BLINDED"
    interruptible = False

    def __init__(self, duration: float, resume: Callable[[], GuardState]) -> None:
        self.duration = duration
        self.resume = resume

    def enter(self, g: CyberGuard) -> None:
        g._clear_route()
        g.speed_factor = 1.0

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        g.pose = blinded_animation(g.fsm.time_in_state)
        if g.fsm.time_in_state >= self.duration:
            return self.resume()
        return None


class KnockedOut(GuardState):
    """Cyberware overloaded. Convulses on its feet for ``knockout_shock_duration``, then
    collapses like a rag doll (``animation.ragdoll``) - back, forward or to one side (the
    guard's ``fall_direction`` if set, else at random): the knees buckle and the body goes
    down as a dead weight. Against a wall it does not stay propped up: it topples off to
    the side it has room on and ends on the floor (or draped over something low). Nothing
    of the body ever passes through what is around it. Terminal: nothing but ``reset()``
    gets the guard back up."""

    name = "KNOCKED_OUT"
    interruptible = False

    def __init__(self, direction: str | None = None) -> None:
        self.direction = direction
        self.collapse: Collapse | None = None  # started when the shock ends

    def enter(self, g: CyberGuard) -> None:
        g._clear_route()
        g.speed_factor = 1.0
        g.shock_timer = HACKING.knockout_shock_duration
        g.fall_angle = 0.0
        g.body = None
        g.pose = electrocuted_animation(0.0)

    def update(self, g: CyberGuard, dt: float) -> GuardState | None:
        t = g.fsm.time_in_state
        shock = HACKING.knockout_shock_duration
        if t < shock:
            g.pose = electrocuted_animation(t)
            return None
        if self.collapse is None:
            self.collapse = Collapse.begin(
                g.x, g.y, g.z, g.yaw, g.frame_obstacles, direction=self.direction,
                rng=g.rng, start_pose=g.pose,
            )  # fmt: skip
            body = self.collapse.state
        elif not self.collapse.done:
            body = self.collapse.step(dt)
        else:
            body = self.collapse.state
        g.body = body
        g.fall_angle = body.fall_angle
        g.pose = body.pose
        return None


# What a state does to a guard's face (the mouth and brows below the visor); the rest
# of the time it wears its resting expression.
_STATE_FACES = {
    "ALERTED": "frown",
    "SUSPICIOUS": "grimace",
    "LOOKING_AROUND": "frown",
    "INVESTIGATING": "frown",
    "INSPECTING": "frown",
    "BLINDED": "shout",
    "KNOCKED_OUT": "grimace",
}


# =============================================================================== guard
class CyberGuard(BaseNPC):
    """Tactical guard: patrols, leans on a wall to inspect its weapon, investigates noises,
    and reports the player to the detection system when it sees them."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        nav_bounds: Bounds,
        patrol_z_min: float = -8.5,
        patrol_z_max: float = -3.0,
        wall_pos: Point | None = (-5.5, -6.0),
        wall_yaw: float = 90.0,
        initial_direction: float = 1.0,
        config: GuardConfig = GUARD,
        guard_id: str = "guard",
        faction: str = DEFAULT_FACTION,
        essential: bool = False,
        patrol_route: Sequence[Point] = (),
    ) -> None:
        super().__init__(x, y, z, spec=guard_spec(guard_id, essential), nav_bounds=nav_bounds)
        self.cfg = config
        self.inventory = starting_inventory("guard")
        self.guard_id = guard_id
        self.faction = faction
        self.essential = essential  # too important / too hardened to knock out
        self.shock_timer = 0.0  # electrocution sparks, for the renderer
        # A collapsed body (where its hips are, how its trunk lies, its limbs) - None while
        # the guard is on its feet. ``fall_angle`` is how far the trunk has tipped over.
        self.body: BodyState | None = None
        self.rng = random.Random(zlib.crc32(guard_id.encode()))  # which way it falls
        self.initial_direction = 1.0 if initial_direction >= 0 else -1.0
        self.home_x = x
        self.patrol_z_min = patrol_z_min
        self.patrol_z_max = patrol_z_max
        self.wall_pos = wall_pos
        self.wall_yaw = wall_yaw
        self.patrol_route: tuple[Point, ...] = tuple(patrol_route)
        self.route_index = 0

        self.patrol_dir = self.initial_direction
        self.yaw = self.target_yaw = 0.0 if self.patrol_dir > 0 else 180.0
        self.investigate_target: Point | None = None
        self.investigate_source: Point | None = None
        self.frame_obstacles: list[AABB] = []
        self.frame_sight_blockers: list[AABB] = []
        # Sighting memory
        self.last_seen: Point | None = None
        self.time_since_seen: float = math.inf
        self.detection_level: float = 0.0  # mirrored from the detection system each frame
        self.resume_after_look: Callable[[], GuardState] | None = None
        self.fsm: StateMachine[CyberGuard] = StateMachine(self, self.patrol_state())

    def patrol_state(self) -> GuardState:
        """The guard's idle routine: the waypoint loop if it has one, else the lane."""
        return PatrolRoute() if self.patrol_route else Pacing()

    def nearest_route_index(self) -> int:
        if not self.patrol_route:
            return 0
        return min(
            range(len(self.patrol_route)),
            key=lambda i: math.dist(self.patrol_route[i], (self.x, self.z)),
        )

    # ------------------------------------------------------------------ introspection
    @property
    def state(self) -> str:
        return self.fsm.name

    @property
    def state_timer(self) -> float:
        return self.fsm.time_in_state

    @property
    def arrived_at_distraction(self) -> bool:
        return isinstance(self.fsm.state, Inspecting)

    @property
    def is_banging(self) -> bool:
        """Currently hitting a machine it was lured to (renderer / tests)."""
        state = self.fsm.state
        return isinstance(state, Inspecting) and state.is_banging(self)

    @property
    def is_reacting(self) -> bool:
        return isinstance(self.fsm.state, (Alerted, Suspicious, LookingAround))

    @property
    def is_blinded(self) -> bool:
        return isinstance(self.fsm.state, Blinded)

    @property
    def is_down(self) -> bool:
        return isinstance(self.fsm.state, KnockedOut)

    @property
    def blind_time_left(self) -> float:
        state = self.fsm.state
        if isinstance(state, Blinded):
            return max(0.0, state.duration - self.fsm.time_in_state)
        return 0.0

    @property
    def is_incapacitated(self) -> bool:
        return self.is_blinded or self.is_down

    # ------------------------------------------------------------------ hack target
    target_kind = "guard"

    @property
    def hack_id(self) -> str:
        return self.guard_id

    @property
    def display_name(self) -> str:
        name = self.guard_id.replace("_", " ").upper()
        return f"{name} (ELITE)" if self.essential else name

    @property
    def secured(self) -> bool:
        return False

    @property
    def hack_position(self) -> Vec3:
        """The chest; as the body goes down the marker follows it to the floor."""
        if self.body is None:
            return (self.x, self.y + 1.6, self.z)
        x, y, z = self.body.point(0.0, 1.45, 0.0)
        return (x, max(y, self.y + 0.12) + 0.12, z)

    def blind(self, duration: float = HACKING.blind_duration) -> bool:
        """Jams the optics for ``duration`` seconds. A guard that is already down cannot be
        blinded (False); re-blinding a blinded guard restarts the timer."""
        if self.is_down:
            return False
        state = self.fsm.state
        resume = state.resume if isinstance(state, Blinded) else self.routine_to_resume()
        self.fsm.transition_to(Blinded(duration, resume))
        return True

    def knock_out(self, direction: str | None = None) -> bool:
        """Electrocutes the guard: permanently down. False if already down. (Whether an
        *essential* guard may be targeted is the hacking system's rule, not this one's.)
        ``direction`` ("back", "front", "left", "right") is which way it falls; by default
        the current picks one at random."""
        if self.is_down:
            return False
        self.fsm.transition_to(KnockedOut(direction))
        return True

    def routine_to_resume(self) -> Callable[[], GuardState]:
        """What to go back to after a reaction: the wall trip if that was under way, or the
        patrol lane (via WalkingToPatrol, which re-enters Pacing on arrival)."""
        state = self.fsm.state
        if isinstance(state, LookingAround) and self.resume_after_look is not None:
            return self.resume_after_look
        if isinstance(state, (WalkingToWall, Leaning)) and self.wall_pos is not None:
            return WalkingToWall
        return WalkingToPatrol

    # ------------------------------------------------------------------ events
    def bind(self, bus: EventBus) -> None:
        bus.subscribe(NoiseEmitted, self._on_noise)

    def _on_noise(self, noise: NoiseEmitted) -> None:
        if math.hypot(noise.x - self.x, noise.z - self.z) < noise.radius:
            self.distract(
                noise.investigate_x,
                noise.investigate_z,
                source=(noise.x, noise.z),
                duration=noise.inspect_duration,
                gesture=noise.gesture,
            )

    def distract(
        self,
        target_x: float,
        target_z: float,
        source: Point | None = None,
        duration: float | None = None,
        gesture: str = "look",
    ) -> None:
        """Lures the guard out of its routine to stand at the target point, facing ``source``,
        for ``duration`` seconds (default ``cfg.investigate_duration``). A blinded or
        knocked-out guard cannot respond."""
        if self.is_incapacitated:
            return
        self.fsm.transition_to(Investigating((target_x, target_z), source, duration, gesture))

    def reset(self) -> None:
        super().reset()
        self.patrol_dir = self.initial_direction
        self.yaw = self.target_yaw = 0.0 if self.patrol_dir > 0 else 180.0
        self.investigate_target = None
        self.investigate_source = None
        self.last_seen = None
        self.time_since_seen = math.inf
        self.detection_level = 0.0
        self.resume_after_look = None
        self.shock_timer = 0.0
        self.body = None
        self.rng = random.Random(zlib.crc32(self.guard_id.encode()))
        self.route_index = 0
        self.inventory = starting_inventory("guard")
        self.fsm.reset(self.patrol_state())

    # ------------------------------------------------------------------ per frame
    def update(
        self,
        player: Player,
        dt: float,
        obstacles: list[AABB] | None = None,
        detection: SightingSink | None = None,
        sight_blockers: list[AABB] | None = None,
        others: Sequence[BaseNPC] = (),
    ) -> bool:
        self.frame_obstacles = obstacles or []
        self.frame_sight_blockers = (
            self.frame_obstacles if sight_blockers is None else sight_blockers
        )
        self.detection_level = detection.level if detection is not None else 0.0
        if self.shock_timer > 0.0:
            self.shock_timer = max(0.0, self.shock_timer - dt)
        self.fsm.update(dt)
        mood = _STATE_FACES.get(self.fsm.state.name)
        if mood is not None and self.pose.expression is None:
            self.pose.expression = mood  # (what the state does to the face under the visor)

        spotted = self.can_see_player(player)
        if spotted:
            self.last_seen = (player.x, player.z)
            self.time_since_seen = 0.0
            if detection is not None:
                detection.report_sighting(self.sighting_rate(player))
            if self.fsm.state.interruptible:
                self.fsm.transition_to(Alerted(self.routine_to_resume()))
        else:
            self.time_since_seen += dt
        return spotted

    def sighting_rate(self, player: Player) -> float:
        """Detection gain per second for the current sighting: crouching is harder to make
        out, and so is distance (full rate up close, far_rate_fraction at max range)."""
        base = DETECTION.guard_rate_crouched if player.crouching else DETECTION.guard_rate_standing
        dist = math.hypot(player.x - self.x, player.z - self.z)
        near, far = self.cfg.near_full_rate_distance, self.cfg.alert_range
        if dist <= near:
            return base
        u = clamp((dist - near) / max(far - near, 1e-6), 0.0, 1.0)
        return base * lerp(1.0, self.cfg.far_rate_fraction, u)

    def can_see_player(self, player: Player) -> bool:
        """Vision narrows sharply while the head is tilted down at the weapon, and is gone
        entirely while blinded or knocked out."""
        if self.is_incapacitated:
            return False
        if self.pose.head_pitch > self.cfg.engrossed_head_pitch:
            max_range, fov = self.cfg.engrossed_range, self.cfg.engrossed_fov
        else:
            max_range, fov = self.cfg.alert_range, self.cfg.alert_fov
        return self.can_see_target(
            player.x,
            player.z,
            max_range=max_range,
            fov_angle=fov,
            obstacles=self.frame_sight_blockers,
            target_y=player.sight_target_y,
        )
