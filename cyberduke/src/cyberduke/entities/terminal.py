"""Hackable netrunner terminal (a "computer" on the facility network)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from cyberduke.config import HACKING, INTERACT_RANGE
from cyberduke.core.events import DoorUnlocked, EventBus, IntelStolen, NoiseEmitted, TerminalHacked
from cyberduke.core.geometry import AABB, Vec3
from cyberduke.world.level import DEFAULT_FACTION


class Terminal:
    """Holding the interact key within range fills ``progress``; at 100 the netrunner is
    *jacked in* (``is_hacked`` / ``is_connected``) and the terminal's interface opens. Being
    connected does nothing by itself: the operator then picks what to do with it - unlock
    the doors it controls, hijack its network's cameras, short-circuit it, copy its data -
    through ``HackingSystem``'s session (see ``systems/hacking.py``).

    The same primitives back the remote quickhacks: ``unlock_doors`` (override the locks it
    controls), ``steal_intel`` (copy the private data) and ``overload`` (fry it - loud
    enough to pull guards over, at the price of losing everything it controlled).
    A ``critical`` terminal refuses to be overloaded; a ``secured`` one is walled off from
    the network and can only be hacked with a direct line of sight (or by jacking in); a
    ``hardened`` one has no wireless interface at all - jack in or forget it.
    """

    target_kind = "terminal"

    HACK_RATE = 60.0  # connection progress per second while interacting (~1.7 s to jack in)
    DECAY_RATE = 20.0  # progress lost per second when interrupted
    FOOTPRINT = [(-1.4, -0.7), (1.4, -0.7), (1.4, 0.9), (-1.4, 0.9)]  # desk + chair, local XZ
    # somebody's little desk at home: a screen, a keyboard, a stool
    HOME_FOOTPRINT = [(-0.55, -0.32), (0.55, -0.32), (0.55, 0.55), (-0.55, 0.55)]
    HEIGHT = 1.85
    INVESTIGATE_DISTANCE = 1.7  # where a guard stands to look at a fried terminal (local +Z)

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        terminal_id: str = "terminal",
        bus: EventBus | None = None,
        faction: str = DEFAULT_FACTION,
        intel: str = "",
        secured: bool = False,
        critical: bool = False,
        door_ids: Sequence[str] = (),
        hardened: bool = False,
        style: str = "triple",
        notes: str = "",
    ) -> None:
        self.x, self.y, self.z = x, y, z
        self.yaw = yaw
        self.terminal_id = terminal_id
        self.bus = bus
        self.faction = faction
        self.intel = intel
        self.secured = secured
        self.critical = critical
        self.hardened = hardened  # air-gapped: no quickhacks, only a physical jack-in
        self.style = style  # the desk's look (renderer): triple/double/single/damaged/overgrown
        self.door_ids = tuple(door_ids)  # doors this terminal can override
        self.notes = notes  # what its screen shows the netrunner once jacked in
        self.reset()

    def reset(self) -> None:
        self.progress = 0.0
        self.is_hacked = False
        self.is_connected = False
        self.intel_stolen = False
        self.doors_overridden = False
        self.destroyed = False
        self.spark_timer = 0.0
        self.anim_time = 0.0

    # ------------------------------------------------------------------ geometry
    @property
    def footprint(self) -> list[tuple[float, float]]:
        return self.HOME_FOOTPRINT if self.style == "home" else self.FOOTPRINT

    def get_bounds(self) -> AABB:
        corners = [self.local_to_world(lx, lz) for lx, lz in self.footprint]
        xs = [c[0] for c in corners]
        zs = [c[1] for c in corners]
        return AABB(min(xs), max(xs), min(zs), max(zs), self.y, self.y + self.HEIGHT)

    def local_to_world(self, lx: float, lz: float) -> tuple[float, float]:
        """Local desk coordinates (chair side is +Z) to world XZ, with the same handedness
        as the renderer's ``rotate(yaw, 0, 1, 0)``: yaw 90 turns the chair side toward +X."""
        rad = math.radians(self.yaw)
        cos_y, sin_y = math.cos(rad), math.sin(rad)
        return (self.x + lx * cos_y + lz * sin_y, self.z - lx * sin_y + lz * cos_y)

    @property
    def front(self) -> tuple[float, float]:
        """Where someone sits to use it (1.5 m out on the chair side)."""
        return self.local_to_world(0.0, 1.5)

    def in_range(self, px: float, pz: float) -> bool:
        reach = 1.6 if self.style == "home" else INTERACT_RANGE  # a small desk: step up to it
        return math.hypot(px - self.x, pz - self.z) < reach

    # ------------------------------------------------------------------ hack target
    @property
    def hack_id(self) -> str:
        return self.terminal_id

    @property
    def display_name(self) -> str:
        return self.terminal_id.replace("_", " ").upper()

    @property
    def hack_position(self) -> Vec3:
        if self.style == "home":
            return (self.x, self.y + 1.05, self.z)  # the one little screen
        return (self.x, self.y + 1.3, self.z)  # the main monitor

    @property
    def has_intel(self) -> bool:
        return bool(self.intel) and not self.intel_stolen

    @property
    def is_online(self) -> bool:
        return not self.destroyed

    # ------------------------------------------------------------------ per frame
    def update(
        self,
        player_x: float,
        player_z: float,
        interact_held: bool,
        dt: float,
        rate_scale: float = 1.0,
    ) -> bool:
        """Advances the connection attempt. Returns True on the frame the link completes
        (the caller opens the interface); an already-breached terminal reconnects at once.
        ``rate_scale``: how much faster the netrunner's hands are (the QuickJack mod)."""
        self.anim_time += dt
        if self.spark_timer > 0.0:
            self.spark_timer = max(0.0, self.spark_timer - dt)
        if self.destroyed or self.is_connected:
            return False
        if interact_held and self.in_range(player_x, player_z):
            if self.is_hacked:  # already breached once: the link is instant
                self.progress = 100.0
            else:
                self.progress = min(100.0, self.progress + self.HACK_RATE * rate_scale * dt)
            return self.progress >= 100.0
        self.progress = max(0.0, self.progress - self.DECAY_RATE * dt)
        return False

    # ------------------------------------------------------------------ session
    def connect(self) -> bool:
        """Opens the terminal's interface. The first connection is the breach itself
        (``TerminalHacked``); what the netrunner does next is up to them."""
        if self.destroyed or self.is_connected:
            return False
        self.is_connected = True
        self.progress = 100.0
        if not self.is_hacked:
            self.is_hacked = True
            self._emit(TerminalHacked(self.terminal_id))
        return True

    def disconnect(self) -> None:
        self.is_connected = False
        self.progress = 0.0  # no stale "link" bar lingering on the HUD

    # ------------------------------------------------------------------ quickhack primitives
    def unlock_doors(self) -> bool:
        """Overrides the locks of every door this terminal controls. False if it has none
        (or is dead)."""
        if self.destroyed or not self.door_ids:
            return False
        self.doors_overridden = True
        for door_id in self.door_ids:
            self._emit(DoorUnlocked(door_id, source=self.terminal_id))
        return True

    def steal_intel(self) -> str | None:
        """Copies the private data off the terminal. Returns it, or None if there is nothing
        (left) to take."""
        if self.destroyed or not self.has_intel:
            return None
        self.intel_stolen = True
        self._emit(IntelStolen(self.terminal_id, self.intel))
        return self.intel

    @property
    def controls_doors(self) -> bool:
        return bool(self.door_ids)

    def overload(self) -> bool:
        """Fries the terminal: a loud bang guards come to investigate. Destroys it for good
        (no more door control, no more data). Refused for ``critical`` terminals."""
        if self.destroyed or self.critical:
            return False
        self.destroyed = True
        self.progress = 0.0
        self.is_connected = False
        self.spark_timer = HACKING.terminal_overload_spark_duration
        ix, iz = self.local_to_world(0.0, self.INVESTIGATE_DISTANCE)
        self._emit(
            NoiseEmitted(
                self.x,
                self.z,
                radius=HACKING.terminal_overload_noise_radius,
                investigate_x=ix,
                investigate_z=iz,
                source="terminal_overload",
                inspect_duration=HACKING.terminal_overload_inspect_duration,
            )
        )
        return True

    def _emit(self, event: object) -> None:
        if self.bus is not None:
            self.bus.emit(event)
