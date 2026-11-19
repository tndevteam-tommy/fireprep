"""Doors: animated barriers that react to events."""

from __future__ import annotations

import math
from collections.abc import Sequence

from cyberduke.core.events import DoorUnlocked, EventBus
from cyberduke.core.geometry import AABB, Vec3
from cyberduke.world.level import DEFAULT_FACTION, DoorData


class SlidingDoor:
    """A blast door that slides sideways once unlocked.

    Two things unlock it:

    * ``DoorUnlocked`` naming this ``door_id`` - sent by the terminal named in
      ``unlocked_by`` when its operator chooses to override the locks (at the desk or over
      the network), or by a direct quickhack on the door itself;
    * ``unlock()`` called directly.

    Merely jacking into the controlling terminal does *not* open it: being inside a computer
    and telling it to open a door are two different things.

    ``high_security`` doors ignore quickhacks aimed at the door itself - they can only be
    opened from their controlling terminal. ``key_intel`` doors are encryption-keyed: they
    open for a player who walks up carrying that intel (see ``Simulation``), and cannot be
    forced either. While it is not yet passable the door's ``obstacle`` box blocks movement
    and sight.
    """

    target_kind = "door"

    def __init__(
        self,
        obstacle: AABB,
        unlocked_by: str | None,
        travel: float = 3.2,
        speed: float = 2.4,
        passable_at: float = 1.5,
        door_id: str = "door",
        data: DoorData | None = None,
        faction: str = DEFAULT_FACTION,
        high_security: bool = False,
        key_intel: str | None = None,
        approach_range: float | None = None,
        inside: str | None = None,
        relock_after: float | None = None,
    ) -> None:
        self.obstacle = obstacle
        self.unlocked_by = unlocked_by
        self.travel = travel
        self.speed = speed
        self.passable_at = passable_at
        self.door_id = door_id
        self.data = data  # visual description (leaf/indicator props) for the renderer
        self.faction = faction
        self.high_security = high_security
        self.key_intel = key_intel  # encryption-keyed: opens for whoever carries this intel
        self.approach_range = approach_range  # None: stays open once unlocked
        self.inside = inside  # the side a latch opens it from, locked or not (see DoorData)
        self.relock_after = relock_after  # (a ticket gate: it sets itself again)
        self.reset()

    @classmethod
    def from_data(cls, d: DoorData) -> SlidingDoor:
        return cls(
            d.obstacle, d.unlocked_by, d.travel, d.speed, d.passable_at, d.door_id, d,
            faction=d.faction, high_security=d.high_security, key_intel=d.key_intel,
            approach_range=d.approach_range, inside=d.inside, relock_after=d.relock_after,
        )  # fmt: skip

    def reset(self) -> None:
        self.unlocked = False
        self.open_amount = 0.0  # metres the leaf has slid
        self.unlocked_for = 0.0  # seconds since it was unlocked

    # ------------------------------------------------------------------ hack target
    @property
    def hack_id(self) -> str:
        return self.door_id

    @property
    def display_name(self) -> str:
        return self.door_id.replace("_", " ").upper()

    @property
    def secured(self) -> bool:
        return False  # doors are always on the network; high_security limits *what* works

    @property
    def hack_position(self) -> Vec3:
        o = self.obstacle
        return (o.center_x, min(o.max_y, o.min_y + 1.6), o.center_z)

    # ------------------------------------------------------------------ events
    def bind(self, bus: EventBus) -> None:
        bus.subscribe(DoorUnlocked, self._on_door_unlocked)

    def _on_door_unlocked(self, event: DoorUnlocked) -> None:
        if event.door_id == self.door_id:
            self.unlock()

    def unlock(self) -> None:
        self.unlocked = True
        self.unlocked_for = 0.0

    def lock(self) -> None:
        """Locks it again (a front door its owner locks behind them); it swings shut."""
        self.unlocked = False

    def on_inside(self, x: float, z: float) -> bool:
        """Whether (x, z) is on the latch side of the door (see ``DoorData.inside``)."""
        o, side = self.obstacle, self.inside
        if side == "-x":
            return x < o.min_x
        if side == "+x":
            return x > o.max_x
        if side == "-z":
            return z < o.min_z
        if side == "+z":
            return z > o.max_z
        return False

    @property
    def keyed(self) -> bool:
        return self.key_intel is not None

    @property
    def forceable(self) -> bool:
        """Whether a FORCE OPEN quickhack has any chance on this lock."""
        return not self.high_security and not self.keyed

    def distance_to(self, x: float, z: float) -> float:
        o = self.obstacle
        return math.hypot(o.center_x - x, o.center_z - z)

    @property
    def passable(self) -> bool:
        return self.open_amount > self.passable_at

    @property
    def fully_open(self) -> bool:
        return self.open_amount >= self.travel

    def wants_open(self, bodies: Sequence[tuple[float, float]] = ()) -> bool:
        """Unlocked, and (for an approach door) somebody is close enough to go through -
        or, for a door with a latch, somebody inside walks up to it, locked or not."""
        if self.approach_range is None:
            return self.unlocked
        near = [(x, z) for x, z in bodies if self.distance_to(x, z) <= self.approach_range]
        if self.unlocked:
            return bool(near)
        if self.inside is not None:
            # the latch: from inside, or for whoever is in the doorway on their way out
            return any(self.on_inside(x, z) or self.obstacle.contains_xz(x, z) for x, z in near)
        return False

    def update(self, dt: float, bodies: Sequence[tuple[float, float]] = ()) -> None:
        """Slides the leaf. ``bodies`` are the (x, z) of everyone about (the player and
        the NPCs): an approach door opens for them and shuts again when nobody is near.
        A lock that resets itself (``relock_after``) does so once its time is up and
        nobody is in the way."""
        if self.unlocked and self.relock_after is not None:
            self.unlocked_for += dt
            if self.unlocked_for >= self.relock_after and not self.wants_open(bodies):
                self.lock()
        if self.wants_open(bodies):
            if self.open_amount < self.travel:
                self.open_amount = min(self.travel, self.open_amount + self.speed * dt)
        elif (self.unlocked or self.inside is not None or self.approach_range is not None) and (
            self.open_amount > 0.0
        ):
            # (a locked door that was open - its owner locked it behind them - swings shut)
            self.open_amount = max(0.0, self.open_amount - self.speed * dt)
