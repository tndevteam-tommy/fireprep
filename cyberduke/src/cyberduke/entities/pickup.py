"""Things lying about that the player can take with [E] (the data stick on a nightstand).

A ``Pickup`` is an item (``core.inventory``) at a spot in the world. The nearest one in
reach and roughly in front of the player is what [E] takes (``Simulation.pickup_in_reach``);
taken, it is in the inventory and no longer drawn. ``restore`` puts it back where it was
(an item confiscated from a thrown-out trespasser goes home).
"""

from __future__ import annotations

import math

from cyberduke.core.geometry import Vec3
from cyberduke.core.inventory import item

REACH = 1.7  # metres (horizontal) within which [E] picks a thing up
CONE = 50.0  # degrees off the line of sight it may be


class Pickup:
    def __init__(
        self, pickup_id: str, item_id: str, x: float, y: float, z: float, yaw: float = 0.0
    ) -> None:
        item(item_id)  # an unknown item fails at load time, not when picked up
        self.pickup_id = pickup_id
        self.item_id = item_id
        self.x, self.y, self.z = x, y, z
        self.yaw = yaw
        self.reset()

    def reset(self) -> None:
        self.taken = False

    def restore(self) -> None:
        self.taken = False

    @property
    def name(self) -> str:
        return item(self.item_id).name

    @property
    def position(self) -> Vec3:
        return (self.x, self.y, self.z)

    def in_reach(self, eye: Vec3, forward: Vec3) -> float | None:
        """How far off the line of sight it is (degrees), if it can be taken from here."""
        if self.taken:
            return None
        dx, dy, dz = self.x - eye[0], self.y + 0.03 - eye[1], self.z - eye[2]
        if math.hypot(dx, dz) > REACH:
            return None
        d = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        cos = (dx * forward[0] + dy * forward[1] + dz * forward[2]) / d
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        return angle if angle <= CONE else None
