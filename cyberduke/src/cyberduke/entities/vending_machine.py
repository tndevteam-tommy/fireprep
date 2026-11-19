"""'NEON COLD' vending machine: an interactable acoustic distraction."""

from __future__ import annotations

import math

from cyberduke.config import HACKING, INTERACT_RANGE
from cyberduke.core.events import EventBus, NoiseEmitted
from cyberduke.core.geometry import AABB, Vec3
from cyberduke.world.level import DEFAULT_FACTION


class VendingMachine:
    """Pressing [E] at the machine - or quickhacking it from anywhere on its network -
    short-circuits it: a loud, repeatable distraction (``COOLDOWN`` seconds between shorts).

    A short leaves it **broken**: after the shower of sparks it stays out of order - the
    marquee dead but for a stutter, no cans, the odd spark - until somebody gives it a good
    kick (``repair``: Dell does, if he is out and sees it), or it comes back by itself
    after ``BROKEN_FOR`` seconds with nobody to see to it."""

    target_kind = "vending"

    SPARK_DURATION = 18.0  # long enough for a guard to walk over and start hitting it
    COOLDOWN = 22.0
    BROKEN_FOR = 300.0  # (five minutes: Dell has long since been out on an errand by then)
    NOISE_RADIUS = 22.0
    INVESTIGATE_OFFSET_Z = 1.2  # where a guard stands to inspect the machine

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        bus: EventBus | None = None,
        machine_id: str = "vending_machine",
        faction: str = DEFAULT_FACTION,
    ) -> None:
        self.x, self.y, self.z = x, y, z
        self.yaw = yaw
        self.bus = bus
        self.machine_id = machine_id
        self.faction = faction
        self.reset()

    # ------------------------------------------------------------------ hack target
    @property
    def hack_id(self) -> str:
        return self.machine_id

    @property
    def display_name(self) -> str:
        return self.machine_id.replace("_", " ").upper()

    @property
    def secured(self) -> bool:
        return False

    @property
    def hack_position(self) -> Vec3:
        return (self.x, self.y + 1.3, self.z)

    @property
    def can_trigger(self) -> bool:
        return self.cooldown <= 0.0

    def reset(self) -> None:
        self.is_sparking = False
        self.spark_timer = 0.0
        self.cooldown = 0.0
        self.anim_time = 0.0
        self.broken = False
        self.broken_timer = 0.0  # seconds until it comes back by itself

    def repair(self) -> None:
        """A good kick: the machine shudders back to life (the sparks stop, it sells cans)."""
        self.broken = False
        self.broken_timer = 0.0
        self.is_sparking = False
        self.spark_timer = 0.0

    def get_bounds(self) -> AABB:
        # (1.3 wide, 1.1 deep; turned side-on - against a wall along z - the other way)
        hx, hz = (0.55, 0.65) if abs(math.sin(math.radians(self.yaw))) > 0.7 else (0.65, 0.55)
        return AABB(self.x - hx, self.x + hx, self.z - hz, self.z + hz, self.y, self.y + 2.30)

    @property
    def front(self) -> tuple[float, float]:
        """Where somebody stands to look it over (or bang on it): in front of it."""
        r = math.radians(self.yaw)
        d = self.INVESTIGATE_OFFSET_Z
        return self.x + math.sin(r) * d, self.z + math.cos(r) * d

    def in_range(self, px: float, pz: float) -> bool:
        return math.hypot(px - self.x, pz - self.z) < INTERACT_RANGE

    def update(self, player_x: float, player_z: float, interact_pressed: bool, dt: float) -> None:
        self.anim_time += dt
        if self.cooldown > 0.0:
            self.cooldown = max(0.0, self.cooldown - dt)
        if self.is_sparking:
            self.spark_timer -= dt
            if self.spark_timer <= 0.0:
                self.is_sparking = False
        if self.broken:
            self.broken_timer -= dt
            if self.broken_timer <= 0.0:
                self.repair()
        if interact_pressed and self.can_trigger and self.in_range(player_x, player_z):
            self.trigger_distraction()

    def trigger_distraction(self) -> bool:
        """Short-circuits the machine and announces the noise; whoever is listening reacts.
        Returns False (and does nothing) while the machine is still recharging."""
        if not self.can_trigger:
            return False
        self.is_sparking = True
        self.spark_timer = self.SPARK_DURATION
        self.cooldown = self.COOLDOWN
        self.broken = True
        self.broken_timer = max(self.BROKEN_FOR, self.SPARK_DURATION)
        if self.bus is not None:
            self.bus.emit(
                NoiseEmitted(
                    self.x,
                    self.z,
                    radius=self.NOISE_RADIUS,
                    investigate_x=self.front[0],
                    investigate_z=self.front[1],
                    source="vending_machine",
                    inspect_duration=HACKING.vending_inspect_duration,
                    gesture="bang",
                )
            )
        return True
