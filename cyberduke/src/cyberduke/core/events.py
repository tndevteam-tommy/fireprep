"""A small synchronous event bus.

Entities *emit* events while they update; the simulation *dispatches* the queue at a
well-defined point in the frame so handlers always see a consistent world. Event types are
plain frozen dataclasses and handlers are subscribed by type.

Why: it lets the vending machine say "there was a noise here" without knowing that guards
exist, and lets a door open when *a* terminal is hacked without the main loop wiring the two
together by hand. New interactables and reactors can be added without touching either side.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

Event = Any
E = TypeVar("E")
Handler = Callable[[E], None]


# ----------------------------------------------------------------------------- event types
@dataclass(frozen=True)
class NoiseEmitted:
    """Something made an audible noise that NPCs may investigate."""

    x: float
    z: float
    radius: float  # how far away the noise can be heard
    investigate_x: float  # where an investigator should walk to
    investigate_z: float
    source: str = ""  # free-form tag, e.g. "vending_machine"
    inspect_duration: float | None = None  # how long to stay and inspect (None: guard default)
    gesture: str = "look"  # what an investigator does on arrival: "look" or "bang"


@dataclass(frozen=True)
class TerminalHacked:
    """A terminal was fully compromised by jacking in (holding [E] at the desk)."""

    terminal_id: str


@dataclass(frozen=True)
class DoorUnlocked:
    """A door's lock was overridden: remotely (quickhack) or from a controlling terminal."""

    door_id: str
    source: str = ""  # the hack id of whatever unlocked it (terminal id, "quickhack", ...)


@dataclass(frozen=True)
class NetworkBreached:
    """The player gained access to a faction's local network (every device on it becomes
    visible and, unless secured, remotely hackable)."""

    faction_id: str
    via: str  # hack id of the computer that was used as the entry point


@dataclass(frozen=True)
class IntelStolen:
    """Private data was copied off a computer."""

    terminal_id: str
    intel: str


@dataclass(frozen=True)
class HackApplied:
    """A quickhack finished uploading and took effect on ``target_id``."""

    target_id: str
    kind: str  # HackKind value
    access: str  # "direct" or "network"


# ----------------------------------------------------------------------------- bus
class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)
        self._queue: deque[Event] = deque()

    def subscribe(self, event_type: type[E], handler: Handler[E]) -> Callable[[], None]:
        """Registers ``handler`` for ``event_type``. Returns a function that unsubscribes."""
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    def emit(self, event: Event) -> None:
        """Queues an event for the next ``dispatch()``."""
        self._queue.append(event)

    def publish(self, event: Event) -> None:
        """Delivers an event to its handlers immediately."""
        for handler in list(self._handlers.get(type(event), ())):
            handler(event)

    def dispatch(self) -> int:
        """Delivers every queued event in FIFO order (including events emitted by handlers
        during dispatch). Returns the number of events delivered."""
        delivered = 0
        while self._queue:
            self.publish(self._queue.popleft())
            delivered += 1
        return delivered

    def clear(self) -> None:
        self._queue.clear()

    @property
    def pending(self) -> int:
        return len(self._queue)
