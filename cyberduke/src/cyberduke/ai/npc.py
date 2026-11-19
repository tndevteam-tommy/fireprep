"""Base NPC: transform, locomotion (A* navigation with sliding), and perception.

Rendering is handled by ``cyberduke.render.humanoid``; an NPC only carries a ``HumanoidSpec``
describing how it should look.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from cyberduke.ai.pathfinding import Bounds, Pathfinder
from cyberduke.animation import HumanoidPose, idle_animation, walk_animation
from cyberduke.animation.look import PLAIN, Look
from cyberduke.assets.faces import face_name
from cyberduke.config import NPC
from cyberduke.core.events import EventBus
from cyberduke.core.geometry import (
    AABB,
    any_cylinder_overlap,
    line_of_sight_clear,
    segment_clear_2d,
    yaw_towards,
)
from cyberduke.core.interp import wrap_angle_deg
from cyberduke.core.inventory import Inventory
from cyberduke.entities.player import Player
from cyberduke.systems.detection import SightingSink

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class HumanoidSpec:
    """Visual configuration of a humanoid character (pure data; consumed by the renderer)."""

    has_visor: bool = False
    has_core_glow: bool = False
    has_holster: bool = False
    hair_color: RGB = (0.12, 0.10, 0.08)
    visor_color: RGB = (1.0, 0.1, 0.1)
    prop: str | None = None  # name of a registered prop mesh, e.g. "laser_blaster"
    # Civilian variety: tints multiplied into the textured body parts, and an overall scale
    shirt_color: RGB = (1.0, 1.0, 1.0)
    pants_color: RGB = (1.0, 1.0, 1.0)
    skin_color: RGB = (1.0, 1.0, 1.0)
    height: float = 1.0
    has_armor: bool = True  # chest plate, shoulder and knee pads (guards); civilians go without
    # How they look (``animation.look``): the hairstyle, cybernetic eyes, chrome, the cut
    # of their clothes - and the colours that go with it
    look: Look = PLAIN
    hair_color2: RGB | None = None  # dyed tips, a streak (None: the hair colour)
    jacket_color: RGB = (0.1, 0.1, 0.12)
    neon_color: RGB = (0.1, 0.9, 1.0)  # neon trim, glowing soles, the glow of implants
    eye_color: RGB = (0.1, 0.9, 1.0)  # cybernetic eyes
    boot_color: RGB = (0.08, 0.08, 0.09)
    # The face (``assets.faces``): its resting expression - most people's is "neutral";
    # they move their mouths to talk, eat, smoke (``HumanoidPose.expression``) - and the
    # mark on it ("", "circuit", "tattoo", "paint", "scar")
    mood: str = "neutral"
    mark: str = ""

    @property
    def face(self) -> str:
        """The texture of the resting face (skin tone, mark, mood)."""
        return face_name(self.skin_color, self.mark, self.mood)

    def face_for(self, expression: str | None) -> str:
        """The texture of the face wearing ``expression`` (None: the resting one)."""
        return face_name(self.skin_color, self.mark, expression or self.mood)


class BaseNPC:
    """Spatial agent providing heading control, A* navigation with obstacle sliding, and vision."""

    STALL_GIVE_UP = 2.0  # seconds without progress before an unreachable goal counts as reached
    SIDESTEP_AFTER = 0.6  # seconds held up by somebody before walking round them
    TAKEN_AFTER = 1.0  # seconds held up by somebody standing on the goal before giving it up
    TAKEN_NEAR = 1.6  # ... when we are at least this near it
    SLIDE_FOR = 1.5  # seconds of sliding along a corner before counting as pinned
    SHORTCUT_EVERY = 0.3  # seconds between looks for a corner of the route to cut
    CORNER_REACH = 0.35  # how near a corner of the route counts as turning it
    DEADLOCK_PUSH = 0.45  # steering this weak means the pushes cancel: go round instead
    REPLAN_AFTER_FAILURE = 1.0  # seconds before a failed search is tried again

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        spec: HumanoidSpec,
        nav_bounds: Bounds,
        walk_speed: float = NPC.walk_speed,
        turn_speed: float = NPC.turn_speed_deg,
        collision_radius: float = NPC.collision_radius,
        body_height: float = NPC.body_height,
        eye_height: float = NPC.eye_height,
    ):
        self.base_x, self.base_y, self.base_z = x, y, z
        self.x, self.y, self.z = x, y, z
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.walk_speed = walk_speed
        self.speed_factor = 1.0  # temporary multiplier (e.g. creeping while suspicious)
        self.turn_speed = turn_speed
        self.collision_radius = collision_radius
        self.body_height = body_height
        self.eye_height = eye_height
        self.walk_cycle = 0.0

        self.path: list[tuple[float, float]] = []
        self.current_goal: tuple[float, float] | None = None
        self.destination: tuple[float, float] | None = None  # where the current route ends
        self.stalled_time = 0.0  # seconds spent pressing against an obstacle without progress
        self.blocked_time = 0.0  # seconds spent held up by somebody standing in the way
        self.slide_time = 0.0  # seconds spent sliding along something square in the way
        self.slide_side: tuple[float, float] | None = None  # ... and which way
        self.held_up_by: tuple[float, float, float] | None = None  # who is in the way
        self.taken_time = 0.0  # seconds the goal has had somebody standing on it
        self.since_shortcut = 0.0  # seconds since the route was last checked for a shortcut
        self.steer_side = 0.0  # going round somebody on this side (+1 right, -1 left; 0 not)
        # Progress watchdog: two people steering round each other can dance on the spot
        # for ever; whoever notices first stands still for a moment and lets the other by.
        self.progress_wp: tuple[float, float] | None = None
        self.progress_best = math.inf
        self.progress_clock = 0.0
        self.giving_way = 0.0  # seconds left standing aside, letting somebody past
        self.gave_way = 0.0  # metres stepped back so far
        self.frame_blockers: list[AABB] = []  # this frame's obstacles (for stepping back)
        # how long without progress before giving way - different person to person, so
        # that of two people in each other's way one gives way first
        self.give_way_after = 1.6 + 0.55 * (int(abs(x * 7.3 + z * 3.1) * 10.0) % 5)
        # why the last navigate_towards() call reported arrival: "there" (at the goal),
        # "taken" (somebody is standing on it and we are as near as we will get) or
        # "stalled" (pinned against an obstacle - the goal may well be out of reach)
        self.arrival = "there"
        self.route_found = True  # whether the last search reached its goal
        self.since_search = 0.0  # seconds since the last search
        self.pathfinder = Pathfinder(
            grid_size=0.4, agent_radius=collision_radius, agent_height=body_height
        )

        self.nav_bounds = nav_bounds
        self.spec = spec
        self.pose = HumanoidPose()
        self.fall_angle = 0.0  # whole-body tilt in degrees (0 upright, 90 flat on the back)
        self.blocks_player = False  # civilians are solid to the player; guards are not
        self.inventory = Inventory(capacity=12)  # what they carry (``core.inventory``)
        self.talking_to = None  # the netrunner, while they are in conversation

    # ------------------------------------------------------------------ contract
    def bind(self, bus: EventBus) -> None:  # noqa: B027 - optional hook
        """Subscribe to whatever events this NPC reacts to (default: nothing)."""

    def update(
        self,
        player: Player,
        dt: float,
        obstacles: list[AABB] | None = None,
        detection: SightingSink | None = None,
        sight_blockers: list[AABB] | None = None,
        others: Sequence[BaseNPC] = (),
    ) -> bool:
        """Advances behaviour by ``dt``. Returns True if the player was seen this frame.

        ``obstacles`` are what the NPC must walk around; ``sight_blockers`` are what it cannot
        see through (defaults to ``obstacles``). They differ for walkable solids such as
        catwalk decks, which occlude without colliding. ``others`` are the other NPCs (for
        those that keep out of each other's way).
        """
        raise NotImplementedError

    def reset(self):
        self.x, self.y, self.z = self.base_x, self.base_y, self.base_z
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.walk_cycle = 0.0
        self.speed_factor = 1.0
        self._clear_route()
        self.pose = HumanoidPose()
        self.fall_angle = 0.0

    def body_box(self) -> AABB:
        """A collision box for the standing body (what a solid NPC is to the player)."""
        r = self.collision_radius * 0.85
        return AABB(
            self.x - r, self.x + r, self.z - r, self.z + r, self.y, self.y + self.body_height
        )

    def _clear_route(self) -> None:
        self.path = []
        self.current_goal = None
        self.destination = None
        self.stalled_time = 0.0
        self.blocked_time = 0.0
        self.slide_time = 0.0
        self.slide_side = None
        self.held_up_by = None
        self.taken_time = 0.0
        self.progress_wp = None
        self.progress_clock = 0.0
        self.giving_way = 0.0
        self.route_found = True

    def rotate_towards(self, target_yaw: float, dt: float) -> bool:
        """Smoothly interpolates yaw toward target_yaw. Returns True when facing target."""
        diff = wrap_angle_deg(target_yaw - self.yaw)
        step = self.turn_speed * dt
        if abs(diff) <= step:
            self.yaw = target_yaw
            return True
        self.yaw += math.copysign(step, diff)
        return False

    def navigate_towards(
        self,
        target_x: float,
        target_z: float,
        dt: float,
        obstacles: list[AABB] | None = None,
        stop_dist: float = 0.15,
        avoid: Sequence[tuple[float, float, float]] = (),
    ) -> bool:
        """Walks toward (target_x, target_z) along an A* route with obstacle sliding.

        Returns True on arrival. If the target itself is unreachable (inside an obstacle),
        arrival means "at the nearest reachable point" rather than never.

        ``avoid`` lists other bodies as (x, z, radius): the step steers away from any of
        them that are close and slows down for one directly ahead, so crowds part instead
        of walking through each other.
        """
        self.arrival = "there"
        if math.hypot(target_x - self.x, target_z - self.z) <= stop_dist:
            self._clear_route()
            return True

        goal_key = (round(target_x, 2), round(target_z, 2))
        # Already planned for this goal and standing at its (possibly adjusted) destination?
        if (
            self.current_goal == goal_key
            and self.destination is not None
            and math.hypot(self.destination[0] - self.x, self.destination[1] - self.z) <= stop_dist
        ):
            self.path = []
            return True

        self.since_search += dt
        need_replan = (self.current_goal != goal_key) or not self.path
        # A blocked next leg means the world changed: plan again - but after a search that
        # found no route the fallback *is* a blocked straight line, and re-running an
        # exhausted search every frame cost tens of milliseconds a frame, so retry at most
        # once every REPLAN_AFTER_FAILURE seconds.
        retry_ok = self.route_found or self.since_search >= self.REPLAN_AFTER_FAILURE
        if not need_replan and obstacles and self.path and retry_ok:
            wp = self.path[0]
            if not self.pathfinder.line_clear(self.x, self.z, wp[0], wp[1], obstacles, self.y):
                need_replan = True

        if need_replan:
            if obstacles:
                route = self.pathfinder.find_path(
                    self.x, self.z, target_x, target_z, obstacles, self.nav_bounds, self.y
                )
                self.path = list(route.waypoints)
                self.destination = route.destination
                self.route_found = route.found
            else:
                self.path = [(target_x, target_z)]
                self.destination = (target_x, target_z)
                self.route_found = True
            self.current_goal = goal_key
            self.since_search = 0.0

        while self.path:
            wx, wz = self.path[0]
            # a corner on the way is turned at once; only the goal itself counts from
            # ``stop_dist`` away (cutting a corner by a generous stop_dist walked people
            # straight into whatever the corner went round)
            reach = stop_dist if len(self.path) == 1 else min(stop_dist, self.CORNER_REACH)
            if math.hypot(wx - self.x, wz - self.z) <= reach:
                self.path.pop(0)
            else:
                break

        # Pushed off the planned line (round somebody, say), the next corner may be out of
        # the way now while the one after it is in plain view: cut the corner. Without
        # this two people each steering round the other short of their next corners
        # could circle each other for good.
        self.since_shortcut += dt
        if obstacles and len(self.path) >= 2 and self.since_shortcut >= self.SHORTCUT_EVERY:
            self.since_shortcut = 0.0
            after = self.path[1]
            feet, top = self.y, self.y + self.body_height
            if segment_clear_2d(
                self.x, self.z, after[0], after[1],
                (o for o in obstacles if o.overlaps_vertical_span(feet, top)),
                radius=self.collision_radius + 0.08,  # with room to spare, not grazing
            ):  # fmt: skip
                self.path.pop(0)

        if not self.path:
            return True

        tx, tz = self.path[0]
        dx = tx - self.x
        dz = tz - self.z
        d = math.hypot(dx, dz)
        if d <= 0.001:
            return True
        self.frame_blockers = obstacles or []
        if avoid and self._goal_taken(target_x, target_z, dt, avoid):
            self._clear_route()
            self.current_goal = goal_key
            self.destination = (self.x, self.z)
            self.arrival = "taken"
            self.pose = idle_animation(0.0)
            return True
        if avoid:
            remaining = d + sum(
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(self.path, self.path[1:], strict=False)
            )
            if self._giving_way(remaining, dt, avoid):
                return False

        ux, uz = dx / d, dz / d
        slow = 1.0
        if avoid:
            ux, uz, slow = self._steer_clear(ux, uz, avoid)
        self.target_yaw = yaw_towards(self.x, self.z, self.x + ux, self.z + uz)
        self.rotate_towards(self.target_yaw, dt)

        move_step = min(d, self.walk_speed * self.speed_factor * slow * dt)
        vx = ux * move_step
        vz = uz * move_step

        new_x = self.x + vx
        new_z = self.z + vz
        blocker = self._blocking_body(new_x, new_z, avoid) if avoid else None
        self.held_up_by = blocker
        if blocker is not None:
            self.blocked_time += dt
            # Somebody is standing on the very spot we are going to, and we are nearly
            # there: this is as near as we get (the spot is taken).
            gx, gz = self.destination or (target_x, target_z)
            if (
                self.blocked_time >= self.TAKEN_AFTER
                and math.hypot(gx - blocker[0], gz - blocker[1])
                < blocker[2] + self.collision_radius + 0.3
                and math.hypot(gx - self.x, gz - self.z) < self.TAKEN_NEAR
            ):
                self._clear_route()
                self.current_goal = goal_key
                self.destination = (self.x, self.z)
                self.arrival = "taken"
                return True
            # Held up for a moment by somebody who is not moving off: walk round them.
            if self.blocked_time >= self.SIDESTEP_AFTER and self._side_step(
                blocker, ux, uz, move_step, obstacles or [], avoid
            ):
                self.advance_walk_cycle(dt)
                return False
            self.advance_walk_cycle(dt * 0.3)  # shuffling on the spot, waiting for a gap
            return False
        self.blocked_time = 0.0

        # Physical collision avoidance with per-axis obstacle sliding (height-aware)
        old_x, old_z = self.x, self.z
        if obstacles:
            if self._body_collides(self.x, self.z, obstacles):
                # Put inside something by a scripted step (off a stool, out of a doorway):
                # any step that gets us further out of it is allowed - with per-axis
                # sliding alone every step "collides" and the body is stuck for good.
                if self._penetration(new_x, new_z, obstacles) < self._penetration(
                    self.x, self.z, obstacles
                ):
                    self.x, self.z = new_x, new_z
                else:
                    self._escape(obstacles, move_step)
            else:
                if not self._body_collides(new_x, self.z, obstacles):
                    self.x = new_x
                if not self._body_collides(self.x, new_z, obstacles):
                    self.z = new_z
                if math.hypot(self.x - old_x, self.z - old_z) >= 0.2 * move_step:
                    self.slide_time = 0.0
                    self.slide_side = None
                elif self.route_found and self.slide_time < self.SLIDE_FOR:
                    # flat against a corner of something on a route that does go round
                    # it: slide along it the way the route bends for a moment, rather
                    # than walk on the spot until giving up
                    self.slide_time += dt
                    self._slide_along(ux, uz, tx, tz, move_step, obstacles, avoid)
        else:
            self.x = new_x
            self.z = new_z

        # Pinned against an obstacle with no way round (no route to a goal the grid could
        # not reach): after a moment that *is* the nearest reachable point, so report
        # arrival instead of leaving the agent walking on the spot forever.
        if math.hypot(self.x - old_x, self.z - old_z) < 0.2 * move_step:
            self.stalled_time += dt
            if self.stalled_time >= self.STALL_GIVE_UP:
                self._clear_route()
                self.current_goal = goal_key
                self.destination = (self.x, self.z)
                self.pose = HumanoidPose()
                self.arrival = "stalled"
                return True
        else:
            self.stalled_time = 0.0

        # the legs keep time with how far the body really went: slowed down behind
        # somebody or squeezing past a corner, full-speed strides would be a moonwalk
        full = self.walk_speed * self.speed_factor * dt
        moved = math.hypot(self.x - old_x, self.z - old_z)
        self.advance_walk_cycle(dt * min(1.0, max(0.2, moved / full)) if full > 0.0 else dt)
        return False

    def _steer_clear(
        self, ux: float, uz: float, avoid: Sequence[tuple[float, float, float]]
    ) -> tuple[float, float, float]:
        """Bends the unit direction away from nearby bodies and returns (ux, uz, speed
        factor). A body closer than the sum of radii plus a personal-space margin pushes
        the direction sideways; one right in front slows the walk down and is passed on
        the side it is (slightly) on - or on the right when it is dead ahead."""
        px, pz, slow = 0.0, 0.0, 1.0
        for ox, oz, radius in avoid:
            dx, dz = self.x - ox, self.z - oz
            dist = math.hypot(dx, dz)
            if dist < 1e-6:
                continue
            reach = self.collision_radius + radius + 0.9
            if dist >= reach:
                continue
            weight = (reach - dist) / reach
            # ahead of us (within ~70 degrees): weigh it more, and slow down
            ahead = (-dx * ux - dz * uz) / dist
            if ahead > 0.35:
                slow = min(slow, max(0.25, 1.0 - weight * 1.2))
                weight *= 1.8
                # sidestep: the body's side of our line of travel (right when dead ahead)
                cross = dx * uz - dz * ux
                side = 1.0 if cross >= -1e-6 else -1.0
                px += uz * side * weight * 0.8
                pz += -ux * side * weight * 0.8
            px += dx / dist * weight
            pz += dz / dist * weight
        nx, nz = ux + px * 1.4, uz + pz * 1.4
        n = math.hypot(nx, nz)
        if n < self.DEADLOCK_PUSH:
            # The push away from whoever is ahead all but cancels the way we want to go:
            # the sum points anywhere and flips about from one frame to the next (a walk
            # on the spot). Go round them instead - on one side, kept to until clear.
            if self.steer_side == 0.0:
                self.steer_side = 1.0 if (ux * pz - uz * px) <= 0.0 else -1.0
            side = self.steer_side
            rx, rz = uz * side + ux * 0.3, -ux * side + uz * 0.3
            m = math.hypot(rx, rz)
            return rx / m, rz / m, max(slow, 0.5)
        self.steer_side = 0.0
        if n < 1e-6:
            return ux, uz, slow
        return nx / n, nz / n, slow

    GIVE_WAY_NEAR = 1.6  # giving way to somebody within this many metres ...
    GIVE_WAY_FOR = 4.0  # ... for at most this long,
    GIVE_WAY_STEP = 0.6  # stepping back from them by at most this far

    def _giving_way(
        self, remaining: float, dt: float, avoid: Sequence[tuple[float, float, float]]
    ) -> bool:
        """The progress watchdog, on the length of the route still to walk (which a
        replan round somebody does not reset). Two people steering round each other can
        otherwise dance on the spot for good: after ``give_way_after`` seconds of getting
        no nearer with somebody close, this one steps back out of their way and waits
        until they have gone by. Returns True while giving way."""
        if self.giving_way > 0.0:
            self.giving_way -= dt
            near = [
                (math.hypot(ox - self.x, oz - self.z), ox, oz)
                for ox, oz, _ in avoid
                if math.hypot(ox - self.x, oz - self.z) < self.GIVE_WAY_NEAR + 0.5
            ]
            if not near and self.giving_way < self.GIVE_WAY_FOR - 0.5:
                self.giving_way = 0.0  # they have gone by
                return False
            moved = False
            if near and self.gave_way < self.GIVE_WAY_STEP:
                d, ox, oz = min(near)
                ax, az = (self.x - ox) / max(d, 1e-6), (self.z - oz) / max(d, 1e-6)
                step = 0.5 * dt
                nx, nz = self.x + ax * step, self.z + az * step
                if not (self.frame_blockers and self._body_collides(nx, nz, self.frame_blockers)):
                    self.x, self.z = nx, nz
                    self.gave_way += step
                    moved = True
            if moved:
                self.advance_walk_cycle(dt * 0.6)
            else:
                self.pose = idle_animation(self.giving_way)
            return True
        goal = self.current_goal
        d = remaining
        if self.progress_wp != goal:
            self.progress_wp, self.progress_best, self.progress_clock = goal, d, 0.0
        elif d < self.progress_best - 0.25:
            self.progress_best, self.progress_clock = d, 0.0
        else:
            self.progress_clock += dt
        if self.progress_clock < self.give_way_after:
            return False
        self.progress_best, self.progress_clock = d, 0.0
        if any(math.hypot(ox - self.x, oz - self.z) < self.GIVE_WAY_NEAR for ox, oz, _ in avoid):
            self.giving_way = self.GIVE_WAY_FOR
            self.gave_way = 0.0
            self.pose = idle_animation(0.0)
            return True
        return False

    def _goal_taken(
        self,
        target_x: float,
        target_z: float,
        dt: float,
        avoid: Sequence[tuple[float, float, float]],
    ) -> bool:
        """Somebody has been standing on the goal for a moment and we are nearly there:
        this is as near as we will get (steering round them we would circle for good)."""
        gx, gz = self.destination or (target_x, target_z)
        near = math.hypot(gx - self.x, gz - self.z) < self.TAKEN_NEAR
        on_it = any(math.hypot(gx - ox, gz - oz) < r + 0.25 for ox, oz, r in avoid)
        self.taken_time = self.taken_time + dt if near and on_it else 0.0
        return self.taken_time >= self.TAKEN_AFTER * 0.5

    def _blocking_body(
        self, new_x: float, new_z: float, avoid: Sequence[tuple[float, float, float]]
    ) -> tuple[float, float, float] | None:
        """The body the step would press into (one already touching us), if any."""
        for body in avoid:
            ox, oz, radius = body
            limit = (self.collision_radius + radius) * 0.9
            now = math.hypot(self.x - ox, self.z - oz)
            after = math.hypot(new_x - ox, new_z - oz)
            if after < limit and after < now:
                return body
        return None

    def _side_step(
        self,
        blocker: tuple[float, float, float],
        ux: float,
        uz: float,
        step: float,
        obstacles: list[AABB],
        avoid: Sequence[tuple[float, float, float]],
    ) -> bool:
        """A step round ``blocker`` - along its edge, on the side nearer the way we are
        going (or the other side, if that one is walled off) - when it is free of
        obstacles and of other people. Returns whether the step was taken."""
        dx, dz = self.x - blocker[0], self.z - blocker[1]
        d = math.hypot(dx, dz)
        if d < 1e-6:
            dx, dz, d = -ux, -uz, 1.0
        dx, dz = dx / d, dz / d
        sides = sorted(((-dz, dx), (dz, -dx)), key=lambda t: -(t[0] * ux + t[1] * uz))
        for tx, tz in sides:
            # round the edge, leaning a little away from them
            sx, sz = tx + 0.35 * dx, tz + 0.35 * dz
            n = math.hypot(sx, sz)
            nx, nz = self.x + sx / n * step * 0.8, self.z + sz / n * step * 0.8
            if obstacles and self._body_collides(nx, nz, obstacles):
                continue
            if self._blocking_body(nx, nz, avoid) is not None:
                continue
            self.x, self.z = nx, nz
            self.target_yaw = yaw_towards(0.0, 0.0, sx, sz)
            return True
        return False

    def _slide_along(
        self,
        ux: float,
        uz: float,
        tx: float,
        tz: float,
        step: float,
        obstacles: list[AABB],
        avoid: Sequence[tuple[float, float, float]],
    ) -> None:
        """Pinned square-on against an obstacle (neither axis of the step was free): try
        the two directions at right angles to the way we were going, the one towards the
        next waypoint first."""
        sides = ((-uz, ux), (uz, -ux))
        to_wp = (tx - self.x, tz - self.z)
        order = sorted(sides, key=lambda t: -(t[0] * to_wp[0] + t[1] * to_wp[1]))
        if self.slide_side is not None:  # keep sliding the way we started (no jitter)
            order.sort(key=lambda t: -(t[0] * self.slide_side[0] + t[1] * self.slide_side[1]))
        for sx, sz in order:
            nx, nz = self.x + sx * step * 0.7, self.z + sz * step * 0.7
            if self._body_collides(nx, nz, obstacles):
                continue
            if avoid and self._blocking_body(nx, nz, avoid) is not None:
                continue
            self.x, self.z = nx, nz
            self.slide_side = (sx, sz)
            return

    def _penetration(self, x: float, z: float, obstacles: list[AABB]) -> float:
        """How deep a body at (x, z) is inside the obstacles it overlaps: for each, how far
        it would have to move to be clear of it (the same square test as
        ``_body_collides``), summed."""
        total = 0.0
        r = self.collision_radius
        y0, y1 = self.y, self.y + self.body_height
        for box in obstacles:
            if not box.overlaps_vertical_span(y0, y1):
                continue
            out = min(
                x - (box.min_x - r), box.max_x + r - x, z - (box.min_z - r), box.max_z + r - z
            )
            if out >= 0.0:  # touching counts (it does for _body_collides)
                total += out + 1e-3
        return total

    def _escape(self, obstacles: list[AABB], step: float) -> None:
        """Stuck inside something and the route leads nowhere out of it: take whichever
        of eight directions gets out of it fastest."""
        best, here = None, self._penetration(self.x, self.z, obstacles)
        for k in range(8):
            a = k * math.pi / 4.0
            nx, nz = self.x + math.cos(a) * step, self.z + math.sin(a) * step
            depth = self._penetration(nx, nz, obstacles)
            if depth < here - 1e-6 and (best is None or depth < best[0]):
                best = (depth, nx, nz)
        if best is not None:
            self.x, self.z = best[1], best[2]

    def _steps_into_a_body(
        self, new_x: float, new_z: float, avoid: Sequence[tuple[float, float, float]]
    ) -> bool:
        """True when the step would press into a body that is already touching us."""
        return self._blocking_body(new_x, new_z, avoid) is not None

    def _body_collides(self, x: float, z: float, obstacles: list[AABB]) -> bool:
        return any_cylinder_overlap(
            obstacles, x, z, self.collision_radius, self.y, self.y + self.body_height
        )

    def advance_walk_cycle(self, dt: float) -> None:
        """Advances the walk cycle in proportion to distance walked and applies the walk pose."""
        self.walk_cycle += self.walk_speed * self.speed_factor * dt * NPC.walk_cycle_rate
        self.pose = walk_animation(self.walk_cycle)

    def can_see_target(
        self,
        target_x: float,
        target_z: float,
        max_range: float = 5.2,
        fov_angle: float = 55.0,
        obstacles: list[AABB] | None = None,
        target_y: float | None = None,
    ) -> bool:
        """Checks distance, FOV cone (including head yaw), and 3D obstacle occlusion.

        ``target_y`` is the world-space height of the point being looked at (e.g. the
        player's chest). When omitted the test is made at the NPC's own eye height.
        """
        if math.hypot(target_x - self.x, target_z - self.z) > max_range:
            return False

        effective_gaze = self.yaw + self.pose.head_yaw
        angle_diff = wrap_angle_deg(
            yaw_towards(self.x, self.z, target_x, target_z) - effective_gaze
        )
        if abs(angle_diff) > fov_angle:
            return False

        if obstacles:
            eye = (self.x, self.y + self.eye_height, self.z)
            if target_y is None:
                target_y = eye[1]
            if not line_of_sight_clear(eye, (target_x, target_y, target_z), obstacles):
                return False
        return True
