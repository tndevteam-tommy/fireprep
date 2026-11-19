from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..core.math3d import BoundingBox3D, Vector3D


class DoorKind(Enum):
    """How a door behaves beyond open/closed/locked.

    * ``LEAF`` - an ordinary solid door: NPCs shut (and relock) it behind themselves.
    * ``BARS`` - a barred gate such as a jail cell: see-through, and *nobody* closes or
      locks it as a side effect of walking through. It only changes state when something
      does so on purpose (the justice system locking a prisoner in, a release opening it,
      the player operating it). That is what keeps a released prisoner - and the jailer
      who let them out - from being shut in together.
    """
    LEAF = "leaf"
    BARS = "bars"


@dataclass
class DoorSchedule:
    """Automatic lock/unlock of a door at fixed hours (for doors nobody owns)."""
    open_hour: int
    close_hour: int
    open_message: Optional[str] = None
    close_message: Optional[str] = None


@dataclass
class Door:
    """A door leaf hinged at ``hinge_position``. A door in a wall running east-west (``axis`` "x",
    the usual) spans +x for ``width`` when closed (-x when ``hinge_right``) and swings 90 degrees
    into -z when open (+z when ``swing_out``); one in a wall running north-south (``axis`` "z")
    spans +z (-z when ``hinge_right``) and swings into -x (+x when ``swing_out``). The helpers below
    speak of *along* (the axis the doorway runs on) and *across* (the other one, through the wall)
    so that callers need not care which it is.

    ``owners`` lists who is responsible for it: NPC names and/or faction tags
    (e.g. ``"guards"``). Owners shut and lock it behind themselves - but even
    an owner needs the key item for ``door_id`` to lock or unlock it.
    """
    door_id: str
    hinge_position: Vector3D
    width: float = 3.0
    height: float = 3.5
    thickness: float = 0.2
    is_locked: bool = True
    is_open: bool = False
    key_id: str = "shop_key"
    angle: float = 0.0
    schedule: Optional[DoorSchedule] = None
    owners: Tuple[str, ...] = ()
    lock_difficulty: int = 1           # the lock's tier, 0 (flimsy) .. 5 (legendary): see ``config.LOCK_TIER_NAMES``
    kind: DoorKind = DoorKind.LEAF
    sealed: bool = False               # BARS only: a prisoner is inside - no NPC opens it, whatever keys it has
    swing_out: bool = False            # the leaf folds into +across (+z / +x) when open instead of -across
    hinge_right: bool = False          # hinged at the far end: the leaf spans -along from the hinge (a gate's second leaf)
    gate: Optional[str] = None         # the gateway this leaf belongs to (both leaves of a town gate share it)
    label: Optional[str] = None        # what the door is called on the HUD ("Guard House", "Jae's House", "Town Gate")
    axis: str = "x"                    # "x": in a wall running east-west; "z": in one running north-south

    @property
    def is_bars(self) -> bool:
        return self.kind is DoorKind.BARS

    @property
    def blocks_sight(self) -> bool:
        """Closed leaves hide what is behind them; bars never do."""
        return not self.is_open and not self.is_bars

    @property
    def auto_closes(self) -> bool:
        """Whether NPCs shut this door behind themselves (never for bars, nor a town gate: a gate
        stands as the watch left it)."""
        return not self.is_bars and self.gate is None

    def owned_by(self, name: str, faction: Optional[str] = None) -> bool:
        return name in self.owners or (faction is not None and faction in self.owners)

    # --- geometry: along / across -----------------------------------------------------
    @property
    def along_x(self) -> bool:
        return self.axis != "z"

    def along_of(self, x: float, z: float) -> float:
        """Where (x, z) lies along the doorway's axis."""
        return x if self.along_x else z

    def across_of(self, x: float, z: float) -> float:
        """Where (x, z) lies across the wall (compare with ``line``)."""
        return z if self.along_x else x

    def to_xz(self, along: float, across: float) -> Tuple[float, float]:
        """The (x, z) of a point given along and across the doorway."""
        return (along, across) if self.along_x else (across, along)

    @property
    def line(self) -> float:
        """The wall's line the leaf closes on (its across coordinate)."""
        h = self.hinge_position
        return h.z if self.along_x else h.x

    @property
    def span(self) -> Tuple[float, float]:
        """(low, high) ends of the doorway along its axis."""
        h = self.hinge_position
        u = h.x if self.along_x else h.z
        return (u - self.width, u) if self.hinge_right else (u, u + self.width)

    def side_of(self, x: float, z: float) -> int:
        """Which side of the wall (x, z) is on: +1 (+across) or -1."""
        return 1 if self.across_of(x, z) >= self.line else -1

    def point(self, along: float, across_offset: float, y: Optional[float] = None) -> Vector3D:
        """A point ``along`` the doorway, ``across_offset`` from the wall's line."""
        x, z = self.to_xz(along, self.line + across_offset)
        return Vector3D(x, self.hinge_position.y if y is None else y, z)

    def strip(self, depth: float, slack: float = 0.0) -> Tuple[float, float, float, float]:
        """(min_x, min_z, max_x, max_z) of the doorway ``depth`` either side of the wall's line,
        ``slack`` beyond its ends."""
        lo, hi = self.span
        ax1, cx1 = self.to_xz(lo - slack, self.line - depth)
        ax2, cx2 = self.to_xz(hi + slack, self.line + depth)
        return (min(ax1, ax2), min(cx1, cx2), max(ax1, ax2), max(cx1, cx2))

    # --- cached geometry: the doorway and the leaf are asked for every frame by every NPC that
    # walks near a door (thousands of times a frame): worked out once per pose, not each time
    def _geometry_key(self) -> tuple:
        h = self.hinge_position
        return (h.x, h.y, h.z, self.width, self.height, self.thickness, self.axis, self.hinge_right, self.swing_out)

    def _extent(self) -> Tuple[float, float, float, float]:
        key = self._geometry_key()
        cached = self.__dict__.get("_extent_cache")
        if cached is None or cached[0] != key:
            cached = self.__dict__["_extent_cache"] = (key, self.strip(0.0))
        return cached[1]

    # (the doorway's extents in x and z: a door along x runs x1..x2 on the line z1 = z2)
    @property
    def x1(self) -> float:
        return self._extent()[0]

    @property
    def x2(self) -> float:
        return self._extent()[2]

    @property
    def z1(self) -> float:
        return self._extent()[1]

    @property
    def z2(self) -> float:
        return self._extent()[3]

    def get_bounding_box(self) -> BoundingBox3D:
        """The doorway region (leaf when closed, plus a little slack either side)."""
        h = self.hinge_position
        x1, z1, x2, z2 = self.strip(0.5)
        return BoundingBox3D(Vector3D(x1, h.y, z1), Vector3D(x2, h.y + self.height, z2))

    @property
    def center(self) -> Vector3D:
        key = self._geometry_key()
        cached = self.__dict__.get("_center_cache")
        if cached is None or cached[0] != key:
            lo, hi = self.span
            cached = self.__dict__["_center_cache"] = (key, self.point((lo + hi) / 2.0, 0.0))
        return cached[1].copy()                     # (a copy: callers may move it about)

    def collider(self) -> BoundingBox3D:
        """Solid box of the leaf in its current pose (closed: across the doorway; open: folded
        into -across, or +across for a ``swing_out`` door, from the hinge). The same box is handed
        out until the door moves or swings (treat it as read-only)."""
        key = (self._geometry_key(), self.is_open)
        cached = self.__dict__.get("_collider_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        box = self._leaf_box()
        self.__dict__["_collider_cache"] = (key, box)
        return box

    def _leaf_box(self) -> BoundingBox3D:
        h, half = self.hinge_position, self.thickness / 2.0
        u = h.x if self.along_x else h.z
        if self.is_open:
            v1, v2 = (self.line - half, self.line + self.width) if self.swing_out else (self.line - self.width, self.line + half)
            a = self.to_xz(u - half, v1)
            b = self.to_xz(u + half, v2)
        else:
            lo, hi = self.span
            a = self.to_xz(lo, self.line - half)
            b = self.to_xz(hi, self.line + half)
        return BoundingBox3D(Vector3D(min(a[0], b[0]), h.y, min(a[1], b[1])),
                             Vector3D(max(a[0], b[0]), h.y + self.height, max(a[1], b[1])))

    # The renderer models the leaf spanning +x from its hinge and turns it by these angles about
    # the vertical (glRotatef: a turn of a sends +x to (cos a, 0, -sin a)).
    @property
    def closed_angle(self) -> float:
        """Degrees the leaf is turned when shut: along +x (0), -x (180), +z (270) or -z (90)."""
        if self.along_x:
            return 180.0 if self.hinge_right else 0.0
        return 90.0 if self.hinge_right else 270.0

    @property
    def open_angle(self) -> float:
        """Degrees the leaf is turned when open: folded into -across, or +across (``swing_out``)."""
        if self.along_x:
            return 270.0 if self.swing_out else 90.0        # +z / -z
        return 0.0 if self.swing_out else 180.0             # +x / -x

    # --- state ------------------------------------------------------------------------
    def toggle(self) -> bool:
        """Open/close an unlocked door; returns the new ``is_open`` state."""
        self.is_open = not self.is_open
        return self.is_open

    def unlock_and_open(self) -> None:
        self.is_locked = False
        self.is_open = True

    def lock_and_close(self) -> None:
        self.is_locked = True
        self.is_open = False

    def open_hours_cover(self, hour: int) -> bool:
        """Whether ``hour`` falls in the door's scheduled open hours (from ``open_hour`` up to
        ``close_hour``, round midnight if need be) - when nobody, owner or not, locks it."""
        s = self.schedule
        if s is None or s.open_hour < 0 or s.close_hour < 0:
            return False
        if s.open_hour <= s.close_hour:
            return s.open_hour <= hour < s.close_hour
        return hour >= s.open_hour or hour < s.close_hour

    def apply_schedule(self, hour: int) -> Optional[str]:
        """Apply the door's schedule for ``hour``; returns a message to show, if any. Through the
        open hours a door somebody locked is unlocked again on the hour (a chapel stays open by
        day whoever comes and goes)."""
        if self.schedule is None:
            return None
        if hour == self.schedule.open_hour:
            self.unlock_and_open()
            return self.schedule.open_message
        if hour == self.schedule.close_hour:
            self.lock_and_close()
            return self.schedule.close_message
        if self.is_locked and self.open_hours_cover(hour):
            self.is_locked = False
        return None

    # --- persistence ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"door_id": self.door_id, "is_locked": self.is_locked, "is_open": self.is_open, "sealed": self.sealed}

    def load_dict(self, data: dict) -> None:
        self.is_locked = data.get("is_locked", self.is_locked)
        self.is_open = data.get("is_open", self.is_open)
        self.sealed = data.get("sealed", self.sealed)
