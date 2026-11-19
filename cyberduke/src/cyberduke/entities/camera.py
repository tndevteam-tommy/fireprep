"""Ceiling-mounted security camera with a sweeping detection cone."""

from __future__ import annotations

import math
from collections.abc import Sequence

from cyberduke.config import DETECTION, HACKING
from cyberduke.core.geometry import (
    AABB,
    Vec3,
    Viewpoint,
    forward_vector,
    line_of_sight_clear,
)
from cyberduke.core.interp import clamp, wrap_angle_deg
from cyberduke.entities.player import Player
from cyberduke.systems.detection import SightingSink
from cyberduke.world.level import DEFAULT_FACTION


class SecurityCamera:
    """Sweeps its head and reports the player to the detection system.

    Quickhack primitives: ``shut_down(seconds)`` cuts the camera out of the security system
    for the full duration - it reports nothing and its head goes limp and hangs from the
    swivel (``head_droop``). That countdown is not affected by anything the player does
    afterwards: a netrunner can ``hijack()`` a dark camera (the optics answer to whoever
    holds the feed) and look around through it, and it still comes back online only when
    the timer runs out.

    While hijacked the camera stops reporting, its head lifts and follows ``steer`` freely
    (yaw within the mount's sweep range, tilt between ``MIN_TILT`` and ``MAX_TILT``), and
    ``viewpoint`` describes the view for the renderer and for scans and quickhacks launched
    from the feed.
    """

    target_kind = "camera"
    TILT_DEG = 22.0  # the head is pitched this far down toward the floor while sweeping
    MIN_TILT, MAX_TILT = -15.0, 80.0  # free-look limits when hijacked (negative = up)
    LENS_DROP = 0.1  # lens sits this far below the mount point
    DROOP_DEG = 78.0  # how far a dead head hangs down
    DROOP_RATE = 4.0  # exponential ease rate of the droop/recovery (per second)

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        mount_yaw: float = 0.0,
        sweep_angle: float = 35.0,
        sweep_speed: float = 1.2,
        max_range: float = 8.5,
        fov_deg: float = 45.0,
        camera_id: str = "camera",
        faction: str = DEFAULT_FACTION,
        watch_area=None,
    ) -> None:
        self.x, self.y, self.z = x, y, z
        self.mount_yaw = mount_yaw
        self.sweep_angle = sweep_angle
        self.sweep_speed = sweep_speed
        self.max_range = max_range
        self.fov_deg = fov_deg
        self.camera_id = camera_id
        self.faction = faction
        # A home camera (``CameraData.watch_area``): it only reports somebody inside the
        # home - anything with ``contains(x, z)``. None: it reports anybody it sees.
        self.watch_area = watch_area
        self.reset()

    def reset(self) -> None:
        self.anim_time = 0.0
        self.current_yaw = self.mount_yaw
        self.current_tilt = self.TILT_DEG
        self.is_alerted = False
        self.alert_pulse = 0.0
        self.shutdown_timer = 0.0
        self.head_droop = 0.0  # 0 = powered and level, 1 = hanging dead from the swivel
        self.hijacked = False
        self.hijack_yaw = self.mount_yaw
        self.hijack_tilt = self.TILT_DEG

    def get_bounds(self) -> AABB | None:
        return None  # ceiling mounted; no ground footprint

    # ------------------------------------------------------------------ hack target
    @property
    def hack_id(self) -> str:
        return self.camera_id

    @property
    def display_name(self) -> str:
        return self.camera_id.replace("_", " ").upper()

    @property
    def secured(self) -> bool:
        return False

    @property
    def hack_position(self) -> Vec3:
        return (self.x, self.y - 0.05, self.z)

    @property
    def is_online(self) -> bool:
        return self.shutdown_timer <= 0.0

    @property
    def lens(self) -> Vec3:
        return (self.x, self.y - self.LENS_DROP, self.z)

    @property
    def viewpoint(self) -> Viewpoint:
        """Eye and forward vector of the lens (NPC yaw convention; tilt positive = down)."""
        fx, fy, fz = forward_vector(self.current_yaw, -self.current_tilt, yaw_zero_is_x=False)
        lx, ly, lz = self.lens
        return Viewpoint(lx, ly, lz, fx, fy, fz, source=self.camera_id)

    # ------------------------------------------------------------------ quickhack primitives
    def shut_down(self, seconds: float = HACKING.camera_shutdown_duration) -> bool:
        """Cuts the camera out of the security system for ``seconds``. Ejects a hijacker.
        False if it is already down (the running countdown is never shortened or restarted
        by anything else)."""
        if not self.is_online:
            return False
        self.shutdown_timer = seconds
        self.is_alerted = False
        self.hijacked = False
        return True

    def hijack(self) -> bool:
        """Takes over the feed. Works on a dark camera too (the shutdown keeps counting
        down underneath). False only if this netrunner is already connected to it."""
        if self.hijacked:
            return False
        self.hijacked = True
        # The feed opens centred and level-ish; ``steer`` moves it from there.
        self.hijack_yaw, self.hijack_tilt = self.mount_yaw, self.TILT_DEG
        self.current_yaw, self.current_tilt = self.mount_yaw, self.TILT_DEG
        self.is_alerted = False
        return True

    def release(self) -> None:
        self.hijacked = False
        self.current_tilt = self.TILT_DEG

    def steer(self, delta_yaw: float, delta_tilt: float = 0.0) -> None:
        """Free look for a hijacked head: yaw limited to the mount's mechanical sweep range,
        tilt between looking slightly up and straight down."""
        lo, hi = self.mount_yaw - self.sweep_angle, self.mount_yaw + self.sweep_angle
        self.hijack_yaw = clamp(self.hijack_yaw + delta_yaw, lo, hi)
        self.hijack_tilt = clamp(self.hijack_tilt + delta_tilt, self.MIN_TILT, self.MAX_TILT)

    # ------------------------------------------------------------------ perception
    def can_see(self, player: Player, obstacles: Sequence[AABB]) -> bool:
        if not self.is_online or self.hijacked:
            return False
        dx, dz = player.x - self.x, player.z - self.z
        dist = math.hypot(dx, dz)
        if not (0.2 < dist <= self.max_range):
            return False
        angle_to_player = math.degrees(math.atan2(dx, dz))
        if abs(wrap_angle_deg(angle_to_player - self.current_yaw)) > self.fov_deg * 0.5:
            return False
        # A 3D ray from the lens to the player's chest: the camera hangs high enough to see
        # over a coffee table, but not through a wall, a shelf, or the crate you crouch behind.
        return line_of_sight_clear(
            self.lens, (player.x, player.sight_target_y, player.z), obstacles
        )

    def update(
        self,
        player: Player,
        dt: float,
        obstacles: Sequence[AABB],
        detection: SightingSink | None = None,
        sight_blockers: Sequence[AABB] | None = None,
    ) -> bool:
        """Sweeps the head and tests for the player. Returns True if the player was seen.
        ``sight_blockers`` defaults to ``obstacles``."""
        self.anim_time += dt
        ease = 1.0 - math.exp(-self.DROOP_RATE * dt)
        dark = self.shutdown_timer > 0.0
        if dark:
            # The countdown runs no matter what the player does with the feed meanwhile.
            self.shutdown_timer = max(0.0, self.shutdown_timer - dt)
            self.is_alerted = False
        # A dead head hangs from its swivel; a hijacker drives the motors, so the head comes
        # back up for as long as they hold the feed.
        target_droop = 1.0 if (dark and not self.hijacked) else 0.0
        self.head_droop += (target_droop - self.head_droop) * ease
        if self.hijacked:
            self.current_yaw, self.current_tilt = self.hijack_yaw, self.hijack_tilt
            self.is_alerted = False
            return False
        if dark:
            return False  # no power: no sweep, no sightings
        self.current_tilt = self.TILT_DEG
        self.current_yaw = (
            self.mount_yaw + math.sin(self.anim_time * self.sweep_speed) * self.sweep_angle
        )
        self.is_alerted = self.can_see(
            player, obstacles if sight_blockers is None else sight_blockers
        ) and (self.watch_area is None or self.watch_area.contains(player.x, player.z))
        if self.is_alerted:
            self.alert_pulse += dt * 8.0
            if detection is not None:
                rate = (
                    DETECTION.camera_rate_crouched
                    if player.crouching
                    else DETECTION.camera_rate_standing
                )
                detection.report_sighting(rate)
        return self.is_alerted
