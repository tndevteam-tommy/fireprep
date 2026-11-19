"""Quickhacking: targets, access routes, per-target options, uploads and faction networks.

The flow, from the player's point of view:

1. **Scan** (``scan``): a pulse reveals every electronic target in range - computers,
   cameras, doors, vending machines, people with cyberware - and the HUD highlights them.
2. **Focus**: whatever revealed target sits nearest the crosshair becomes the *focus*.
3. **Choose**: the focus lists its hacks. Which ones are *available* depends on the target
   (a critical computer cannot be overloaded, an essential guard cannot be knocked out, a
   high-security door cannot be forced) and on what the player has learned.
4. **Upload** (``request``): the chosen hack uploads for a few seconds, then takes effect
   through the target's own primitives (``guard.blind``, ``camera.shut_down``, ...), so the
   effect engages exactly the code the rest of the game already runs on.

Two access routes reach a target:

* ``Access.DIRECT`` - it was scanned recently, it is inside ``quickhack_range`` and the
  viewpoint (the player's eyes, or a hijacked camera's lens) has line of sight to it.
* ``Access.NETWORK`` - its faction's network has been *breached*. Hacking any computer on
  a network (a direct quickhack, or jacking in with [E]) breaches it: from then on every
  member is highlighted across the map and can be hacked from anywhere. ``secured``
  computers are the exception: they sit behind their own firewall, so they show up on the
  scanner but can only be hacked directly.

Nothing here imports entity classes; targets satisfy the ``Hackable`` protocol plus the
per-kind attributes documented in ``_availability`` / ``_apply``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from cyberduke.config import HACKING, HackingConfig
from cyberduke.core.events import (
    DoorUnlocked,
    EventBus,
    HackApplied,
    IntelStolen,
    NetworkBreached,
    TerminalHacked,
)
from cyberduke.core.geometry import AABB, Vec3, Viewpoint, segment_intersects_aabb_3d
from cyberduke.core.inventory import CLOSEUP_HACK_SPEED, PING_RANGE
from cyberduke.systems.scanner import Scanner
from cyberduke.world.level import FactionData


# =============================================================================== vocabulary
class HackKind(str, Enum):
    # computers (terminals)
    OVERRIDE_DOORS = "override_doors"
    STEAL_DATA = "steal_data"
    OVERLOAD = "overload"
    # people with cyberware
    BLIND = "blind"
    ELECTROCUTE = "electrocute"
    # doors
    FORCE_OPEN = "force_open"
    # cameras
    SHUTDOWN = "shutdown"
    HIJACK = "hijack"
    # vending machines
    DISTRACT = "distract"
    # access panels (a door's lock)
    BYPASS = "bypass"


@dataclass(frozen=True)
class HackSpec:
    kind: HackKind
    label: str  # HUD name
    upload_time: float  # seconds from request to effect
    description: str


_SPECS = (
    HackSpec(HackKind.OVERRIDE_DOORS, "OVERRIDE DOORS", 1.5, "unlock every door it controls"),
    HackSpec(HackKind.STEAL_DATA, "STEAL DATA", 2.5, "copy the private data stored on it"),
    HackSpec(HackKind.OVERLOAD, "OVERLOAD", 1.5, "fry it: a loud distraction; gone for good"),
    HackSpec(HackKind.BLIND, "BLIND", 1.0, "jam their optics for a few seconds"),
    HackSpec(HackKind.ELECTROCUTE, "ELECTROCUTE", 2.0, "overload their cyberware: down for good"),
    HackSpec(HackKind.FORCE_OPEN, "FORCE OPEN", 1.5, "override the lock"),
    HackSpec(HackKind.SHUTDOWN, "SHUT DOWN", 1.0, "cut the feed for a while"),
    HackSpec(HackKind.HIJACK, "HIJACK FEED", 0.8, "look through the camera; hack from there"),
    HackSpec(HackKind.DISTRACT, "DISTRACT", 0.6, "short it out: guards come looking"),
    HackSpec(HackKind.BYPASS, "BYPASS LOCK", 2.5, "release the lock it works"),
)
HACKS: dict[HackKind, HackSpec] = {s.kind: s for s in _SPECS}

# Sitting at the desk is a different interface: the same four things a computer can do for
# you, named the way the machine names them, and quicker because you are plugged into it.
SESSION_HACKS: tuple[HackKind, ...] = (
    HackKind.OVERRIDE_DOORS,
    HackKind.HIJACK,
    HackKind.SHUTDOWN,
    HackKind.OVERLOAD,
    HackKind.STEAL_DATA,
)
_SESSION_SPECS = (
    HackSpec(HackKind.OVERRIDE_DOORS, "UNLOCK DOORS", 1.0, "release every lock it controls"),
    HackSpec(HackKind.HIJACK, "HIJACK CAMERAS", 1.0, "take over its network's camera feeds"),
    HackSpec(HackKind.SHUTDOWN, "DISABLE CAMERAS", 1.5, "cut every camera on its network"),
    HackSpec(HackKind.OVERLOAD, "SHORT CIRCUIT (DISTRACT & DESTROY)", 1.2, "blow it out"),
    HackSpec(HackKind.STEAL_DATA, "STEAL DATA", 2.0, "copy the private data stored on it"),
)
SESSION_SPECS: dict[HackKind, HackSpec] = {s.kind: s for s in _SESSION_SPECS}

HACKS_BY_TARGET_KIND: dict[str, tuple[HackKind, ...]] = {
    "terminal": (HackKind.OVERRIDE_DOORS, HackKind.STEAL_DATA, HackKind.OVERLOAD),
    "guard": (HackKind.BLIND, HackKind.ELECTROCUTE),
    "door": (HackKind.FORCE_OPEN,),
    "camera": (HackKind.SHUTDOWN, HackKind.HIJACK),
    "vending": (HackKind.DISTRACT,),
    "panel": (HackKind.BYPASS,),
}

TARGET_KIND_LABELS = {
    "terminal": "COMPUTER",
    "guard": "CYBERWARE",
    "door": "DOOR LOCK",
    "camera": "CAMERA",
    "vending": "VENDING MACHINE",
    "panel": "ACCESS PANEL",
}

CLOSE_RANGE = 3.0  # metres: a quickhack from this close counts as close-up (the QuickJack mod)


class Access(str, Enum):
    DIRECT = "direct"  # scanned, in range, in line of sight
    NETWORK = "network"  # reached through a breached faction network


class Hackable(Protocol):
    """What every hack target exposes. ``target_kind`` picks the hack list; the kind-
    specific attributes/primitives are looked up on the object (see ``_availability``)."""

    target_kind: str
    faction: str

    @property
    def hack_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def secured(self) -> bool: ...

    @property
    def hack_position(self) -> Vec3: ...


@dataclass(frozen=True)
class HackOption:
    spec: HackSpec
    available: bool
    reason: str = ""  # why not, for the HUD ("LOCKED: CRITICAL SYSTEM", "RECHARGING", ...)

    @property
    def kind(self) -> HackKind:
        return self.spec.kind


@dataclass
class Upload:
    target: Any  # Hackable
    spec: HackSpec
    access: Access
    elapsed: float = 0.0
    session: bool = False  # run from a terminal's own interface rather than as a quickhack
    speed: float = 1.0  # how much faster than ``spec.upload_time`` it goes (cyberware)

    @property
    def duration(self) -> float:
        return self.spec.upload_time / max(self.speed, 1e-6)

    @property
    def progress(self) -> float:
        if self.duration <= 0.0:
            return 1.0
        return min(1.0, self.elapsed / self.duration)

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration


@dataclass
class Network:
    """One faction's local network: its members and whether the player is inside it."""

    faction: FactionData
    members: list[Any] = field(default_factory=list)
    breached: bool = False
    breached_via: str = ""

    @property
    def faction_id(self) -> str:
        return self.faction.faction_id

    def reset(self) -> None:
        self.breached = False
        self.breached_via = ""


@dataclass(frozen=True)
class TargetStatus:
    """One line of the HUD's target list."""

    target: Any
    access: Access | None  # how it can be hacked right now (None: visible but unreachable)
    scanned: bool  # lit up by a recent scan pulse
    on_network: bool  # lit up because its network is breached
    focused: bool
    distance: float

    @property
    def highlighted(self) -> bool:
        return self.scanned or self.on_network


class NetrunnerLike(Protocol):
    """The slice of the player the hacking system needs."""

    intel: list[str]

    def knows(self, hack_kind: HackKind) -> bool: ...

    def stat(self, name: str) -> float: ...  # equipped cyberware (optional: 1.0 without)


# =============================================================================== system
class HackingSystem:
    def __init__(
        self,
        bus: EventBus,
        player: NetrunnerLike,
        targets: Iterable[Any],
        factions: Iterable[FactionData] = (),
        config: HackingConfig = HACKING,
    ) -> None:
        self.bus = bus
        self.player = player
        self.cfg = config
        self.targets: list[Any] = list(targets)
        self.scanner = Scanner(config)
        self.networks: dict[str, Network] = {f.faction_id: Network(f) for f in factions}
        for t in self.targets:
            self.networks.setdefault(
                t.faction, Network(FactionData(t.faction, t.faction.replace("_", " ").upper()))
            ).members.append(t)
        self._by_id: dict[str, Any] = {t.hack_id: t for t in self.targets}
        if len(self._by_id) != len(self.targets):
            raise ValueError("hack target ids must be unique")
        self.log: deque[str] = deque(maxlen=6)
        bus.subscribe(TerminalHacked, self._on_terminal_hacked)
        bus.subscribe(IntelStolen, self._on_intel_stolen)
        self.reset()

    def reset(self) -> None:
        if getattr(self, "session", None) is not None:
            self.session.disconnect()
        self.scanner.reset()
        for net in self.networks.values():
            net.reset()
        self.view: Viewpoint | None = None
        self.focus: Any | None = None
        self.upload: Upload | None = None
        self.hijacked_camera: Any | None = None
        self.session: Any | None = None  # the terminal whose interface is open
        self._access: dict[str, Access | None] = {}
        self._sight_blockers: Sequence[AABB] = ()
        self.log.clear()
        self.notice: str = ""  # last rejection, for the HUD
        self.notice_timer = 0.0

    # ------------------------------------------------------------------ lookup
    def target(self, hack_id: str) -> Any:
        return self._by_id[hack_id]

    def network_of(self, target: Any) -> Network:
        return self.networks[target.faction]

    # ------------------------------------------------------------------ networks
    def is_breached(self, faction_id: str) -> bool:
        net = self.networks.get(faction_id)
        return net is not None and net.breached

    def breach(self, faction_id: str, via: str) -> bool:
        """Opens a faction's network to the player. Returns True if it was not already open."""
        net = self.networks.get(faction_id)
        if net is None or net.breached:
            return False
        net.breached = True
        net.breached_via = via
        self.bus.emit(NetworkBreached(faction_id, via))
        self._say(f"NETWORK BREACHED: {net.faction.name.upper()} via {via.upper()}")
        return True

    def on_network(self, target: Any) -> bool:
        """Reachable through a breached network (secured and air-gapped devices never are)."""
        return not target.secured and self.wireless(target) and self.is_breached(target.faction)

    @staticmethod
    def wireless(target: Any) -> bool:
        """False for air-gapped (``hardened``) computers: they can be scanned and shown, but
        no route - direct or network - ever reaches them."""
        return not getattr(target, "hardened", False)

    def _on_terminal_hacked(self, event: TerminalHacked) -> None:
        """Jacking into a computer at the desk is the other way onto its network."""
        t = self._by_id.get(event.terminal_id)
        if t is not None and not t.secured:
            self.breach(t.faction, via=t.hack_id)

    def _on_intel_stolen(self, event: IntelStolen) -> None:
        add = getattr(self.player, "add_intel", None)
        if add is not None:
            add(event.intel)
        else:
            self.player.intel.append(event.intel)
        self._say(f"DATA COPIED: {event.intel.upper()}")

    # ------------------------------------------------------------------ visibility & access
    def is_highlighted(self, target: Any) -> bool:
        return self.scanner.is_revealed(target.hack_id) or self.on_network(target)

    def access_for(
        self,
        target: Any,
        view: Viewpoint | None = None,
        sight_blockers: Sequence[AABB] | None = None,
    ) -> Access | None:
        view = view or self.view
        blockers = self._sight_blockers if sight_blockers is None else sight_blockers
        if not self.wireless(target):
            return None
        if (
            view is not None
            and self.scanner.is_revealed(target.hack_id)
            and view.distance_to(target.hack_position) <= self.cfg.quickhack_range
            and self.line_of_sight(view.eye, target.hack_position, blockers)
        ):
            return Access.DIRECT
        if self.on_network(target):
            return Access.NETWORK
        return None

    @staticmethod
    def line_of_sight(eye: Vec3, point: Vec3, blockers: Iterable[AABB]) -> bool:
        """3D sight test that ignores the box the point itself sits in (a terminal's monitor
        is inside the terminal's own collision box)."""
        for b in blockers:
            if not b.blocks_sight:
                continue
            if b.contains_xz(point[0], point[2]) and b.min_y <= point[1] <= b.max_y:
                continue
            if segment_intersects_aabb_3d(eye, point, b):
                return False
        return True

    def access(self, target: Any) -> Access | None:
        """Access as computed by the last ``update``."""
        return self._access.get(target.hack_id)

    def explain_access(self, target: Any) -> str:
        """Short HUD text: the route in use, or why a highlighted target is out of reach."""
        access = self.access(target)
        if access is Access.DIRECT:
            return "DIRECT"
        if access is Access.NETWORK:
            return "NETWORK"
        if not self.wireless(target):
            return "AIR-GAPPED"
        view = self.view
        if view is None or not self.scanner.is_revealed(target.hack_id):
            return "NO LINK"
        if view.distance_to(target.hack_position) > self.cfg.quickhack_range:
            return "OUT OF RANGE"
        if not self.line_of_sight(view.eye, target.hack_position, self._sight_blockers):
            return "NO LINE OF SIGHT"
        return "NO LINK"

    def statuses(self) -> list[TargetStatus]:
        """Every highlighted target, nearest first (what the HUD draws)."""
        out = []
        view = self.view
        for t in self.targets:
            scanned = self.scanner.is_revealed(t.hack_id)
            on_net = self.on_network(t)
            if not (scanned or on_net):
                continue
            dist = view.distance_to(t.hack_position) if view is not None else 0.0
            out.append(TargetStatus(t, self.access(t), scanned, on_net, t is self.focus, dist))
        out.sort(key=lambda s: s.distance)
        return out

    # ------------------------------------------------------------------ options
    def options(self, target: Any) -> list[HackOption]:
        kinds = HACKS_BY_TARGET_KIND.get(target.target_kind, ())
        out = []
        for kind in kinds:
            if kind is HackKind.STEAL_DATA and not target.intel:
                continue  # nothing worth stealing: not even on the menu
            spec = HACKS[kind]
            if not self.player.knows(kind):
                out.append(HackOption(spec, False, "NOT LEARNED"))
                continue
            reason = _availability(target, kind)
            out.append(HackOption(spec, reason == "", reason))
        return out

    def focus_options(self) -> list[HackOption]:
        return self.options(self.focus) if self.focus is not None else []

    # ------------------------------------------------------------------ commands
    def scan(self, origin: Vec3 | None = None) -> list[Any]:
        """Fires a scanner pulse from ``origin`` (default: the current viewpoint)."""
        if origin is None:
            if self.view is None:
                return []
            origin = self.view.eye
        hits = self.scanner.pulse_from(
            origin, self.targets, range_scale=self.player_stat(PING_RANGE)
        )
        if self.scanner.pulse is not None and self.scanner.pulse.age == 0.0:
            self._say(f"SCAN: {len(hits)} DEVICE(S) IN RANGE" if hits else "SCAN: NOTHING IN RANGE")
        return hits

    def request(self, slot: int) -> bool:
        """Starts uploading option number ``slot`` (1-based) of the focused target. One
        upload at a time; once started it completes even if the target leaves the view."""
        if self.upload is not None:
            return self._reject("UPLOAD IN PROGRESS")
        target = self.focus
        if target is None:
            return self._reject("NO TARGET")
        options = self.options(target)
        if not 1 <= slot <= len(options):
            return self._reject("NO SUCH HACK")
        option = options[slot - 1]
        if not option.available:
            return self._reject(f"{option.spec.label}: {option.reason}")
        access = self.access(target)
        if access is None:
            return self._reject("NO ACCESS")
        self.upload = Upload(target, option.spec, access, speed=self.hack_speed(target, access))
        route = access.value.upper()
        self._say(f"UPLOADING {option.spec.label} -> {target.display_name} [{route}]")
        return True

    def cancel_upload(self) -> None:
        self.upload = None

    def player_stat(self, name: str) -> float:
        """The netrunner's cyberware multiplier for ``name`` (1.0 for a player without)."""
        stat = getattr(self.player, "stat", None)
        return float(stat(name)) if stat is not None else 1.0

    def is_close_up(self, target: Any, access: Access | None = None) -> bool:
        """A direct hack from the netrunner's own eyes within arm's reach (not through a
        camera lens, not over the network)."""
        view = self.view
        return (
            (access or self.access(target)) is Access.DIRECT
            and view is not None
            and view.source == "player"
            and view.distance_to(target.hack_position) <= CLOSE_RANGE
        )

    def hack_speed(self, target: Any, access: Access | None = None) -> float:
        """How fast a quickhack on ``target`` uploads: close-up ones go at the QuickJack
        mod's speed, everything else at the deck's own."""
        if self.is_close_up(target, access):
            return self.player_stat(CLOSEUP_HACK_SPEED)
        return 1.0

    def disconnect(self) -> None:
        """Leaves a hijacked camera feed."""
        cam = self.hijacked_camera
        if cam is not None:
            cam.release()
            self._say(f"DISCONNECTED FROM {cam.display_name}")
        self.hijacked_camera = None

    def linked_cameras(self, camera: Any) -> list[Any]:
        """Every camera on the same network as ``camera`` (itself included), in level order:
        the feeds a hijacker can switch between. A camera that has been shut down is still
        on the list - being dark to the security system does not lock the netrunner out."""
        return [
            t for t in self.targets if t.target_kind == "camera" and t.faction == camera.faction
        ]

    def cycle_camera(self, direction: int) -> bool:
        """Switches a hijacked feed to the next (+1) / previous (-1) live camera on the same
        network. False when not hijacking, or when there is nothing to switch to."""
        cam = self.hijacked_camera
        if cam is None or direction == 0:
            return False
        feeds = self.linked_cameras(cam)
        if len(feeds) < 2 or cam not in feeds:
            return self._reject("NO OTHER FEED ON THIS NETWORK")
        nxt = feeds[(feeds.index(cam) + (1 if direction > 0 else -1)) % len(feeds)]
        cam.release()
        if not nxt.hijack():
            self.hijacked_camera = None
            return False
        self.hijacked_camera = nxt
        self.scanner.reveal(nxt.hack_id)
        self._say(f"FEED: {nxt.display_name}")
        return True

    # ------------------------------------------------------------------ terminal sessions
    def connect(self, terminal: Any) -> bool:
        """Opens a computer's own interface (the player is at the desk, jacked in)."""
        if self.session is terminal or not terminal.connect():
            return False
        self.session = terminal
        self.upload = None
        self._say(f"CONNECTED: {terminal.display_name}")
        return True

    def end_session(self) -> None:
        if self.session is not None:
            self.session.disconnect()
            self._say(f"DISCONNECTED FROM {self.session.display_name}")
            self.session = None
            self.upload = None

    def session_options(self) -> list[HackOption]:
        """What the open terminal offers. Sitting at the desk bypasses the wireless rules
        (``secured`` / ``hardened``); only the machine's own limits apply. STEAL DATA is
        only listed on a computer that actually holds data that matters."""
        terminal = self.session
        if terminal is None:
            return []
        out = []
        for kind in SESSION_HACKS:
            if kind is HackKind.STEAL_DATA and not terminal.intel:
                continue
            spec = SESSION_SPECS[kind]
            reason = self._session_availability(terminal, kind)
            out.append(HackOption(spec, reason == "", reason))
        return out

    def _session_availability(self, t: Any, kind: HackKind) -> str:
        if t.destroyed:
            return "OFFLINE"
        if kind is HackKind.OVERRIDE_DOORS:
            if not t.controls_doors:
                return "NO DOORS LINKED"
            if t.doors_overridden:
                return "DOORS ALREADY OPEN"
        elif kind is HackKind.HIJACK:
            if not self.network_cameras(t.faction):
                return "NO CAMERAS ON THIS NETWORK"
        elif kind is HackKind.SHUTDOWN:
            cameras = self.network_cameras(t.faction)
            if not cameras:
                return "NO CAMERAS ON THIS NETWORK"
            if all(not c.is_online for c in cameras):
                return "CAMERAS ALREADY DOWN"
        elif kind is HackKind.OVERLOAD:
            if t.critical:
                return "LOCKED: CRITICAL SYSTEM"
        elif kind is HackKind.STEAL_DATA:
            if not t.intel:
                return "NO DATA"
            if t.intel_stolen:
                return "DATA ALREADY COPIED"
        return ""

    def network_cameras(self, faction_id: str) -> list[Any]:
        return [t for t in self.targets if t.target_kind == "camera" and t.faction == faction_id]

    def session_request(self, slot: int) -> bool:
        """Runs option ``slot`` (1-based) of the open terminal interface."""
        terminal = self.session
        if terminal is None:
            return self._reject("NO TERMINAL")
        if self.upload is not None:
            return self._reject("BUSY")
        options = self.session_options()
        if not 1 <= slot <= len(options):
            return self._reject("NO SUCH OPTION")
        option = options[slot - 1]
        if not option.available:
            return self._reject(f"{option.spec.label}: {option.reason}")
        self.upload = Upload(
            terminal, option.spec, Access.DIRECT, session=True,
            speed=self.player_stat(CLOSEUP_HACK_SPEED),
        )  # fmt: skip
        self._say(f"{terminal.display_name}: {option.spec.label}...")
        return True

    def apply_session(self, terminal: Any, kind: HackKind) -> bool:
        """Carries out a choice made at the desk."""
        reason = self._session_availability(terminal, kind)
        if reason:
            self._say(f"{SESSION_SPECS[kind].label} REFUSED ({reason})")
            return False
        ok = False
        if kind is HackKind.OVERRIDE_DOORS:
            ok = bool(terminal.unlock_doors())
        elif kind is HackKind.STEAL_DATA:
            ok = terminal.steal_intel() is not None
        elif kind is HackKind.OVERLOAD:
            ok = bool(terminal.overload())
            if ok:
                self.session = None  # nothing left to talk to
        elif kind is HackKind.HIJACK:
            ok = self._hijack_network_camera(terminal.faction)
            if ok:
                self.end_session()  # you are behind a lens now, not at the desk
        elif kind is HackKind.SHUTDOWN:
            ok = self._disable_network_cameras(terminal.faction)
        if not ok:
            self._say(f"{SESSION_SPECS[kind].label} FAILED ON {terminal.display_name}")
            return False
        if not terminal.secured:
            self.breach(terminal.faction, via=terminal.hack_id)
        self.bus.emit(HackApplied(terminal.hack_id, kind.value, Access.DIRECT.value))
        self._say(f"{SESSION_SPECS[kind].label}: {terminal.display_name}")
        return True

    def _disable_network_cameras(self, faction_id: str) -> bool:
        """The computer's version of SHUT DOWN: every camera on its network at once (a
        direct quickhack only ever takes down the one camera it is aimed at)."""
        hit = 0
        for cam in self.network_cameras(faction_id):
            if cam.shut_down(self.cfg.camera_shutdown_duration):
                hit += 1
                if self.hijacked_camera is cam:
                    self.hijacked_camera = None
        return hit > 0

    def _hijack_network_camera(self, faction_id: str) -> bool:
        """Drops the netrunner into the first usable feed on a network (dark ones count)."""
        cameras = self.network_cameras(faction_id)
        if not cameras:
            return False
        target = next((c for c in cameras if c.is_online and not c.hijacked), None)
        target = target or next((c for c in cameras if not c.hijacked), None)
        if target is None:
            return False
        return self.apply(target, HackKind.HIJACK, Access.DIRECT)

    # ------------------------------------------------------------------ per frame
    def update(self, view: Viewpoint, dt: float, sight_blockers: Sequence[AABB] = ()) -> None:
        self.view = view
        self._sight_blockers = sight_blockers
        self.scanner.update(dt)
        if self.notice_timer > 0.0:
            self.notice_timer = max(0.0, self.notice_timer - dt)

        # A hijacked camera that went dark drops the connection.
        cam = self.hijacked_camera
        if cam is not None and not cam.hijacked:
            self.hijacked_camera = None

        self._access = {t.hack_id: self.access_for(t, view, sight_blockers) for t in self.targets}
        self.focus = self._pick_focus(view)

        if self.upload is not None:
            self.upload.elapsed += dt
            if self.upload.done:
                up, self.upload = self.upload, None
                if up.session:
                    self.apply_session(up.target, up.spec.kind)
                else:
                    self.apply(up.target, up.spec.kind, up.access)
        if self.session is not None and (self.session.destroyed or self.hijacked_camera):
            self.end_session()

    def _pick_focus(self, view: Viewpoint) -> Any | None:
        """The accessible target nearest the crosshair. Distance counts a little too, so a
        networked door far behind the computer you are actually looking at does not steal
        the focus just because it happens to sit dead centre."""
        best, best_score = None, math.inf
        for t in self.targets:
            if self._access.get(t.hack_id) is None:
                continue
            angle = view.angle_to(t.hack_position)
            if angle > self.cfg.focus_cone_deg:
                continue
            score = angle + self.cfg.focus_distance_weight * view.distance_to(t.hack_position)
            if score < best_score:
                best, best_score = t, score
        return best

    # ------------------------------------------------------------------ effects
    def apply(self, target: Any, kind: HackKind, access: Access = Access.DIRECT) -> bool:
        """Applies a hack's effect immediately (the end of an upload). Returns False if the
        rules forbid it (air-gapped, essential, ...) or the target refused (e.g. it changed
        state during the upload)."""
        if kind not in HACKS_BY_TARGET_KIND.get(target.target_kind, ()):
            return False
        reason = _availability(target, kind)
        if reason:
            self._say(f"{HACKS[kind].label} REFUSED: {target.display_name} ({reason})")
            return False
        ok = _apply(self, target, kind)
        if not ok:
            self._say(f"{HACKS[kind].label} FAILED ON {target.display_name}")
            return False
        if target.target_kind == "terminal" and not target.secured:
            self.breach(target.faction, via=target.hack_id)
        self.scanner.reveal(target.hack_id)
        self.bus.emit(HackApplied(target.hack_id, kind.value, access.value))
        self._say(f"{HACKS[kind].label}: {target.display_name}")
        return True

    # ------------------------------------------------------------------ messages
    def _say(self, line: str) -> None:
        self.log.append(line)

    def _reject(self, why: str) -> bool:
        self.notice = why
        self.notice_timer = 2.0
        return False


# =============================================================================== per-kind rules
def _availability(t: Any, kind: HackKind) -> str:
    """Why ``kind`` cannot run on ``t`` right now ("" = it can). Encodes the design rules:
    critical computers keep their data safe from overloads, essential people cannot be put
    down, high-security doors only answer to their terminal, dead things stay dead."""
    k = t.target_kind
    if k == "terminal":
        if t.destroyed:
            return "OFFLINE"
        if t.hardened:
            return "LOCKED: AIR-GAPPED (JACK IN)"
        if kind is HackKind.OVERRIDE_DOORS:
            if not t.door_ids:
                return "NO DOORS LINKED"
            if t.doors_overridden:
                return "DOORS ALREADY OPEN"
        elif kind is HackKind.STEAL_DATA:
            if not t.intel:
                return "NO DATA"
            if t.intel_stolen:
                return "DATA ALREADY COPIED"
        elif kind is HackKind.OVERLOAD and t.critical:
            return "LOCKED: CRITICAL SYSTEM"
    elif k == "guard":
        if t.is_down:
            return "OFFLINE"
        if kind is HackKind.BLIND and t.is_blinded:
            return "ALREADY BLINDED"
        if kind is HackKind.ELECTROCUTE and t.essential:
            return "LOCKED: HARDENED ICE"
    elif k == "door":
        if t.keyed:
            return f"LOCKED: ENCRYPTED (NEEDS {t.key_intel.upper()})"
        if t.high_security:
            return "LOCKED: HIGH SECURITY"
        if t.unlocked:
            return "ALREADY OPEN"
    elif k == "camera":
        if kind is HackKind.SHUTDOWN and not t.is_online:
            return f"ALREADY DOWN {t.shutdown_timer:.0f}s"
        if kind is HackKind.HIJACK and t.hijacked:
            return "ALREADY CONNECTED"
    elif k == "vending" and not t.can_trigger:
        return "RECHARGING"
    elif k == "panel" and t.lock_released:
        return "LOCK ALREADY RELEASED"
    return ""


def _apply(system: HackingSystem, t: Any, kind: HackKind) -> bool:
    """Runs the effect through the target's own primitive."""
    cfg = system.cfg
    if kind is HackKind.OVERRIDE_DOORS:
        return bool(t.unlock_doors())
    if kind is HackKind.STEAL_DATA:
        return t.steal_intel() is not None
    if kind is HackKind.OVERLOAD:
        return bool(t.overload())
    if kind is HackKind.BLIND:
        return bool(t.blind(cfg.blind_duration))
    if kind is HackKind.ELECTROCUTE:
        return bool(t.knock_out())
    if kind is HackKind.FORCE_OPEN:
        if not t.forceable or t.unlocked:
            return False
        t.unlock()
        system.bus.emit(DoorUnlocked(t.door_id, source="quickhack"))
        return True
    if kind is HackKind.SHUTDOWN:
        ok = bool(t.shut_down(cfg.camera_shutdown_duration))
        if ok and system.hijacked_camera is t:
            system.hijacked_camera = None  # shutting down the feed you are in ejects you
        return ok
    if kind is HackKind.HIJACK:
        if not t.hijack():
            return False
        if system.hijacked_camera is not None and system.hijacked_camera is not t:
            system.hijacked_camera.release()  # hopping feeds from inside another camera
        system.hijacked_camera = t
        return True
    if kind is HackKind.DISTRACT:
        return bool(t.trigger_distraction())
    if kind is HackKind.BYPASS:
        return bool(t.bypass())
    return False
