"""core.events: subscription, queued dispatch, ordering, unsubscribe."""

from dataclasses import dataclass

from cyberduke.core.events import EventBus, NoiseEmitted, TerminalHacked


@dataclass(frozen=True)
class Ping:
    n: int


def test_publish_delivers_immediately_to_type_subscribers_only():
    bus = EventBus()
    got = []
    bus.subscribe(Ping, got.append)
    bus.subscribe(TerminalHacked, lambda e: got.append("wrong"))
    bus.publish(Ping(1))
    assert got == [Ping(1)]


def test_emit_is_queued_until_dispatch_and_preserves_fifo_order():
    bus = EventBus()
    got = []
    bus.subscribe(Ping, got.append)
    bus.emit(Ping(1))
    bus.emit(Ping(2))
    assert got == [] and bus.pending == 2
    assert bus.dispatch() == 2
    assert got == [Ping(1), Ping(2)] and bus.pending == 0


def test_events_emitted_by_handlers_during_dispatch_are_delivered_in_the_same_dispatch():
    bus = EventBus()
    got = []

    def chain(e: Ping):
        got.append(e.n)
        if e.n < 3:
            bus.emit(Ping(e.n + 1))

    bus.subscribe(Ping, chain)
    bus.emit(Ping(1))
    assert bus.dispatch() == 3
    assert got == [1, 2, 3]


def test_unsubscribe():
    bus = EventBus()
    got = []
    off = bus.subscribe(Ping, got.append)
    off()
    off()  # idempotent
    bus.publish(Ping(1))
    assert got == []


def test_clear_drops_queued_events():
    bus = EventBus()
    bus.emit(Ping(1))
    bus.clear()
    assert bus.dispatch() == 0


def test_event_types_are_immutable_values():
    n = NoiseEmitted(1.0, 2.0, radius=5.0, investigate_x=1.0, investigate_z=3.0)
    assert n == NoiseEmitted(1.0, 2.0, 5.0, 1.0, 3.0, "")
    assert hash(TerminalHacked("a")) == hash(TerminalHacked("a"))
