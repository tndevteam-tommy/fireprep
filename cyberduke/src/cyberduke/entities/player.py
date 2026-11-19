"""First-person player state and movement controller."""

from __future__ import annotations

import math
from collections.abc import Sequence

from cyberduke.config import DETECTION, PLAYER
from cyberduke.core.geometry import AABB, Viewpoint, any_cylinder_overlap, forward_vector
from cyberduke.core.interp import clamp
from cyberduke.core.inventory import Inventory, starting_inventory
from cyberduke.systems.hacking import HackKind
from cyberduke.world.room import Room
from cyberduke.world.surfaces import ElevationState


class Player:
    def __init__(self, spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.spawn = spawn  # (x, z, yaw) - provided by the level
        # Netrunner progression. Survives ``reset()`` (a respawn does not unlearn anything).
        # Until the levelling system exists every hack is known from the start.
        self.known_hacks: set[HackKind] = set(HackKind)
        self.reset()

    def reset(self, *, keep_intel: bool = False) -> None:
        """Back to the spawn point, standing, empty-handed - unless ``keep_intel`` (a level
        transition: what was stolen stays on the deck)."""
        self.x, self.z, self.yaw = self.spawn
        self.floor_y = 0.0
        self.y = self.floor_y + PLAYER.eye_height_standing
        self.pitch = 0.0
        self.crouching = False
        self.elevation = ElevationState()
        if not keep_intel or not hasattr(self, "intel"):
            self.intel: list[str] = []  # private data stolen this run
            # what the netrunner carries: the deck, cyberware (installed or not), loot
            self.inventory: Inventory = starting_inventory("player")

    def stat(self, name: str) -> float:
        """The equipped cyberware's multiplier for ``name`` (``core.inventory`` stats)."""
        return self.inventory.stat(name)

    def add_intel(self, intel: str) -> None:
        """Stolen data: onto the deck (keyed doors read it) and into the inventory as a
        data shard."""
        if intel not in self.intel:
            self.intel.append(intel)
        if not self.inventory.has("data_shard", intel):
            self.inventory.add("data_shard", meta=intel)

    # ------------------------------------------------------------------ netrunning
    def knows(self, hack_kind: HackKind) -> bool:
        return hack_kind in self.known_hacks

    def learn(self, hack_kind: HackKind) -> None:
        self.known_hacks.add(hack_kind)

    def forget_all_hacks(self) -> None:
        """Hook for the future levelling system: start with nothing and ``learn`` as you go."""
        self.known_hacks.clear()

    def look_direction(self) -> tuple[float, float, float]:
        """Unit forward vector (camera convention: yaw 0 = +X, 90 = +Z; pitch up positive)."""
        return forward_vector(self.yaw, self.pitch, yaw_zero_is_x=True)

    def viewpoint(self) -> Viewpoint:
        fx, fy, fz = self.look_direction()
        return Viewpoint(self.x, self.y, self.z, fx, fy, fz, source="player")

    # ------------------------------------------------------------------ body metrics
    @property
    def eye_height(self) -> float:
        return PLAYER.eye_height_crouched if self.crouching else PLAYER.eye_height_standing

    @property
    def body_height(self) -> float:
        return PLAYER.body_height_crouched if self.crouching else PLAYER.body_height_standing

    @property
    def move_speed(self) -> float:
        return PLAYER.crouch_speed if self.crouching else PLAYER.walk_speed

    @property
    def body_top(self) -> float:
        return self.floor_y + self.body_height

    @property
    def radius(self) -> float:
        return PLAYER.collision_radius

    @property
    def sight_target_y(self) -> float:
        """World-space height observers aim their line-of-sight ray at (the chest)."""
        h = DETECTION.target_height_crouched if self.crouching else DETECTION.target_height_standing
        return self.floor_y + h

    # ------------------------------------------------------------------ update
    def update_height(self, dt: float, floor_y: float) -> None:
        """Eases the camera toward the eye height above the current floor.

        Uses the frame-rate-independent exponential form so no ``dt`` can overshoot.
        """
        self.floor_y = floor_y
        target_y = floor_y + self.eye_height
        self.y += (target_y - self.y) * (1.0 - math.exp(-PLAYER.height_smoothing * dt))


class PlayerController:
    """Turns look/move intents into player state, with boundary + obstacle collision."""

    def __init__(self, player: Player, room: Room) -> None:
        self.player = player
        self.room = room

    def look(self, dx_pixels: float, dy_pixels: float) -> None:
        p = self.player
        p.yaw += dx_pixels * PLAYER.mouse_sensitivity
        p.pitch = clamp(
            p.pitch - dy_pixels * PLAYER.mouse_sensitivity, -PLAYER.max_pitch, PLAYER.max_pitch
        )

    def toggle_crouch(self) -> None:
        self.player.crouching = not self.player.crouching

    def displacement(self, forward: float, strafe: float, dt: float) -> tuple[float, float]:
        """The world-space step ``move`` would take this frame (before anything stops it)."""
        p = self.player
        rad = math.radians(p.yaw)  # camera convention: yaw 0 = +X, 90 = +Z
        move_x = math.cos(rad) * forward + math.sin(rad) * strafe
        move_z = math.sin(rad) * forward - math.cos(rad) * strafe
        length = math.hypot(move_x, move_z)
        if length <= 1e-3:
            return 0.0, 0.0
        step = p.move_speed * dt / length
        return move_x * step, move_z * step

    def move(self, forward: float, strafe: float, dt: float, obstacles: Sequence[AABB]) -> None:
        """``forward``/``strafe`` are intents in [-1, 1] (W/S and A/D). Diagonals are normalised."""
        dx, dz = self.displacement(forward, strafe, dt)
        self.move_by(dx, dz, dt, obstacles)

    def move_by(self, dx: float, dz: float, dt: float, obstacles: Sequence[AABB]) -> None:
        """Applies a world-space displacement with per-axis obstacle sliding, then updates
        the floor elevation and eases the eye height. The building block for ``move`` and
        for the traversability tests."""
        p = self.player
        floor_y, p.elevation = self.room.floor_height(p.x, p.z, p.elevation)
        feet, top = floor_y, floor_y + p.body_height
        if dx or dz:
            # Something already in the netrunner's space (somebody who stepped into them, a
            # door that swung in): no trap - any move that does not go further into it is
            # allowed, so they can always step out of it.
            r = p.radius
            inside = [o for o in obstacles if o.overlaps_cylinder(p.x, p.z, r, feet, top)]
            if inside:
                others = [o for o in obstacles if not any(o is i for i in inside)]

                def free(x: float, z: float) -> bool:
                    return self._free(x, z, feet, top, others) and all(
                        _depth(o, x, z, r) <= _depth(o, p.x, p.z, r) + 1e-9 for o in inside
                    )
            else:

                def free(x: float, z: float) -> bool:
                    return self._free(x, z, feet, top, obstacles)

            def clear(x: float, z: float) -> bool:
                # (stepping up a stair, the body is over the tread it steps onto: what
                # is below that does not stop it - a big step up to the next tread no
                # more than a small one. Stepping down, it is still where it was.)
                nonlocal feet, top
                ahead, _ = self.room.floor_height(x, z, p.elevation)
                if ahead <= floor_y:
                    return free(x, z)
                feet, top = ahead, ahead + p.body_height
                try:
                    return free(x, z)
                finally:
                    feet, top = floor_y, floor_y + p.body_height

            x0, z0 = p.x, p.z
            if clear(p.x + dx, p.z):
                p.x += dx
            if clear(p.x, p.z + dz):
                p.z += dz
            step = math.hypot(dx, dz)
            if math.hypot(p.x - x0, p.z - z0) < 0.5 * step:
                self._ease_round(dx / step, dz / step, step, clear)
            floor_y, p.elevation = self.room.floor_height(p.x, p.z, p.elevation)
        p.update_height(dt, floor_y)

    EASE_REACH = 0.36  # metres: how far to either side a way on is looked for

    EASE_NEAR_DOOR = 1.6  # metres: only this close to a doorway (that is open, or opening)

    def _near_open_door(self) -> bool:
        """Within reach of a front door the netrunner can go through (open, unlocked, or
        being left by its latch side)."""
        p = self.player
        for door in self.room.doors.values():
            data = door.data
            if data is None or data.hinge is None:
                continue
            # (one that will open for them: unlocked, or from the inside - the latch)
            if not (door.passable or door.unlocked or door.on_inside(p.x, p.z)):
                continue
            if door.distance_to(p.x, p.z) <= self.EASE_NEAR_DOOR:
                return True
        return False

    def _ease_round(self, ux: float, uz: float, step: float, clear) -> None:
        """Walking into the jamb of an open doorway the netrunner is not dead in line with,
        the body is eased sideways towards the opening (a step's worth a frame at most)
        instead of stopping dead. Only at a doorway (a front door that is open): anywhere
        else walking into something stops as it always did."""
        if not self._near_open_door():
            return
        p = self.player
        px, pz = -uz, ux  # (square to the way they are going)
        k = 0.04
        while k <= self.EASE_REACH + 1e-9:
            for sign in (1.0, -1.0):
                ox, oz = p.x + px * k * sign, p.z + pz * k * sign
                if clear(ox + ux * step, oz + uz * step) and clear(ox, oz):
                    move = min(k, step)
                    nx, nz = p.x + px * move * sign, p.z + pz * move * sign
                    if clear(nx, nz):
                        p.x, p.z = nx, nz
                    return
            k += 0.04

    def _free(self, x: float, z: float, feet: float, top: float, obstacles: Sequence[AABB]) -> bool:
        r = self.player.radius
        return self.room.is_valid_position(x, z, r) and not any_cylinder_overlap(
            obstacles, x, z, r, feet, top
        )


def _depth(box: AABB, x: float, z: float, radius: float) -> float:
    """How far a body of ``radius`` at (x, z) is into ``box``'s footprint (<= 0: clear) -
    the overlap on the axis it is least into."""
    into_x = min(x + radius - box.min_x, box.max_x - (x - radius))
    into_z = min(z + radius - box.min_z, box.max_z - (z - radius))
    return min(into_x, into_z)
