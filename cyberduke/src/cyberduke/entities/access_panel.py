"""An access panel beside a door: the keypad and card reader that work its lock.

It is a hack target of its own (``target_kind = "panel"``): scanned, and in range and in
sight, its lock can be bypassed (``HackKind.BYPASS``), which unlocks the door it is wired
to (``DoorUnlocked``). The panel shows the lock's state: red while locked, green once
released - and red again if the owner locks the door behind them.
"""

from __future__ import annotations

import math

from cyberduke.core.events import DoorUnlocked, EventBus
from cyberduke.core.geometry import Vec3
from cyberduke.world.level import DEFAULT_FACTION


class AccessPanel:
    target_kind = "panel"

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        door_id: str,
        yaw: float = 0.0,
        panel_id: str = "panel",
        faction: str = DEFAULT_FACTION,
        name: str = "",
        bus: EventBus | None = None,
    ) -> None:
        self.x, self.y, self.z = x, y, z
        self.yaw = yaw
        self.door_id = door_id
        self.panel_id = panel_id
        self.faction = faction
        self.name = name
        self.bus = bus
        self.door = None  # the live door (set by the simulation): the lock it shows
        self.reset()

    def reset(self) -> None:
        self.anim_time = 0.0
        self.bypasses = 0  # how many times the netrunner has cracked it

    # ------------------------------------------------------------------ hack target
    @property
    def hack_id(self) -> str:
        return self.panel_id

    @property
    def display_name(self) -> str:
        return self.name or self.panel_id.replace("_", " ").upper()

    @property
    def secured(self) -> bool:
        return False

    @property
    def hack_position(self) -> Vec3:
        return (self.x, self.y, self.z)

    @property
    def lock_released(self) -> bool:
        return self.door is not None and self.door.unlocked

    def bypass(self) -> bool:
        """Releases the door's lock. False if it is already released."""
        if self.lock_released:
            return False
        self.bypasses += 1
        if self.door is not None:
            self.door.unlock()
        if self.bus is not None:
            self.bus.emit(DoorUnlocked(self.door_id, source=self.panel_id))
        return True

    def distance_to(self, x: float, z: float) -> float:
        return math.hypot(self.x - x, self.z - z)

    def update(self, dt: float) -> None:
        self.anim_time += dt
