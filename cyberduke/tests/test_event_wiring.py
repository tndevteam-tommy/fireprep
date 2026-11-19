"""Emitters and reactors talk only through the bus."""

import pytest

from cyberduke.ai import CyberGuard
from cyberduke.core.events import DoorUnlocked, EventBus, NoiseEmitted, TerminalHacked
from cyberduke.core.geometry import AABB
from cyberduke.entities import Terminal, VendingMachine
from cyberduke.levels import build_facility
from cyberduke.world import Room, SlidingDoor

DT = 1 / 60
NAV = (-10.0, 10.0, -10.0, 50.0)


# ----------------------------------------------------------------------------- terminal
def test_terminal_emits_hacked_exactly_once_with_its_id():
    bus = EventBus()
    got = []
    bus.subscribe(TerminalHacked, got.append)
    t = Terminal(0, 0, 0, terminal_id="t1", bus=bus)
    for _ in range(400):  # well past the time needed
        if t.update(0.0, 1.0, True, DT):
            t.connect()
        bus.dispatch()
    assert t.is_hacked
    assert got == [TerminalHacked("t1")]


def test_terminal_without_a_bus_still_works():
    t = Terminal(0, 0, 0)
    for _ in range(200):
        if t.update(0.0, 1.0, True, DT):
            t.connect()
    assert t.is_hacked and t.is_connected


# ----------------------------------------------------------------------------- door
def test_door_opens_for_its_own_override_only_and_ignores_a_mere_breach():
    bus = EventBus()
    door = SlidingDoor(AABB(-1, 1, 9.9, 10.1), unlocked_by="front")
    door.bind(bus)
    bus.publish(TerminalHacked("front"))  # someone jacked into the controlling terminal...
    door.update(1.0)
    assert door.open_amount == 0.0 and not door.passable, "...which is not the same as opening it"
    bus.publish(DoorUnlocked("other", source="front"))
    door.update(1.0)
    assert door.open_amount == 0.0
    bus.publish(DoorUnlocked("door", source="front"))
    for _ in range(40):
        door.update(0.1)
    assert door.open_amount == door.travel and door.passable


def test_door_reset_relocks():
    door = SlidingDoor(AABB(-1, 1, 9.9, 10.1), unlocked_by="front")
    door.unlocked = True
    door.update(10.0)
    door.reset()
    assert not door.unlocked and door.open_amount == 0.0


def test_room_blast_door_reacts_to_its_override_through_the_bus():
    bus = EventBus()
    room = Room(build_facility())
    room.bind(bus)
    door = room.doors["blast_door"]
    assert door.obstacle in room.get_obstacles()
    bus.publish(DoorUnlocked(door.door_id, source=door.unlocked_by))
    for _ in range(30):
        room.update(0.1)
    assert door.passable
    assert door.obstacle not in room.get_obstacles()


# ----------------------------------------------------------------------------- vending
def test_vending_machine_emits_a_noise_with_an_investigation_point():
    bus = EventBus()
    got = []
    bus.subscribe(NoiseEmitted, got.append)
    vm = VendingMachine(-7.5, 0.0, 25.0, bus=bus)
    vm.update(-7.0, 25.5, True, DT)
    bus.dispatch()
    assert len(got) == 1
    noise = got[0]
    assert (noise.x, noise.z) == (-7.5, 25.0)
    assert noise.radius == VendingMachine.NOISE_RADIUS
    assert (noise.investigate_x, noise.investigate_z) == (
        -7.5,
        25.0 + VendingMachine.INVESTIGATE_OFFSET_Z,
    )
    assert noise.source == "vending_machine"


def test_vending_machine_respects_cooldown():
    bus = EventBus()
    got = []
    bus.subscribe(NoiseEmitted, got.append)
    vm = VendingMachine(0, 0, 0, bus=bus)
    for _ in range(120):
        vm.update(0.5, 0.5, True, DT)
        bus.dispatch()
    assert len(got) == 1
    assert vm.cooldown > 0


# ----------------------------------------------------------------------------- guard
def test_guard_reacts_only_to_noises_within_radius():
    bus = EventBus()
    near = CyberGuard(0.0, 0.0, 0.0, NAV)
    far = CyberGuard(0.0, 0.0, 40.0, NAV)
    near.bind(bus)
    far.bind(bus)
    bus.publish(NoiseEmitted(2.0, 2.0, radius=10.0, investigate_x=2.0, investigate_z=3.0))
    assert near.state == "INVESTIGATING" and near.investigate_target == (2.0, 3.0)
    assert far.state == "PACING"


def test_guard_faces_the_noise_source_on_arrival():
    bus = EventBus()
    g = CyberGuard(0.0, 0.0, 0.0, NAV)
    g.bind(bus)
    # stand 1.2 m in +Z of the source, so facing it means yaw 180 (-Z)
    bus.publish(NoiseEmitted(0.0, 0.0, radius=5.0, investigate_x=0.0, investigate_z=1.2))

    class P:  # minimal player stand-in far away
        x, z, crouching, sight_target_y = 50.0, 50.0, False, 1.5

    for _ in range(300):
        g.update(P(), DT, obstacles=[])
        if g.arrived_at_distraction:
            break
    assert g.arrived_at_distraction
    assert g.target_yaw == pytest.approx(180.0)


# ----------------------------------------------------------------------------- quickhack events
def test_door_opens_on_a_door_unlocked_event_for_its_own_id():
    bus = EventBus()
    door = SlidingDoor(AABB(-1, 1, 9.9, 10.1), unlocked_by=None, door_id="side")
    door.bind(bus)
    bus.publish(DoorUnlocked("other"))
    assert not door.unlocked
    bus.publish(DoorUnlocked("side", source="quickhack"))
    assert door.unlocked
    bus.publish(TerminalHacked("anything"))  # no controlling terminal: never matters


def test_terminal_override_emits_one_door_unlocked_per_controlled_door():
    bus = EventBus()
    got = []
    bus.subscribe(DoorUnlocked, got.append)
    t = Terminal(0, 0, 0, terminal_id="hub", bus=bus, door_ids=("a", "b"))
    assert t.unlock_doors()
    bus.dispatch()
    assert got == [DoorUnlocked("a", "hub"), DoorUnlocked("b", "hub")]
    assert t.doors_overridden and not t.is_hacked
    assert not Terminal(0, 0, 0, bus=bus).unlock_doors(), "nothing linked"


def test_terminal_steal_intel_emits_once_and_jack_in_does_not_repeat_it():
    from cyberduke.core.events import IntelStolen

    bus = EventBus()
    got = []
    bus.subscribe(IntelStolen, got.append)
    t = Terminal(0, 0, 0, terminal_id="t", bus=bus, intel="payroll")
    assert t.steal_intel() == "payroll"
    assert t.steal_intel() is None
    for _ in range(400):
        if t.update(0.0, 1.0, True, DT):
            t.connect()
    bus.dispatch()
    assert t.is_hacked and got == [IntelStolen("t", "payroll")]


def test_jacking_in_only_opens_the_interface_and_announces_the_breach():
    """Being inside a computer is not the same as having used it: the data is still on it
    and its doors are still shut until the operator picks those options."""
    from cyberduke.core.events import IntelStolen

    bus = EventBus()
    got = []
    bus.subscribe(IntelStolen, got.append)
    bus.subscribe(TerminalHacked, got.append)
    t = Terminal(0, 0, 0, terminal_id="t", bus=bus, intel="payroll", door_ids=("d",))
    opened = []
    for _ in range(400):
        if t.update(0.0, 1.0, True, DT):
            opened.append(t.connect())  # what the simulation does with the signal
    bus.dispatch()
    assert opened.count(True) == 1, "the link completes exactly once"
    assert got == [TerminalHacked("t")]
    assert t.is_hacked and t.is_connected
    assert not t.intel_stolen and not t.doors_overridden
    t.disconnect()
    assert not t.is_connected and t.is_hacked
    assert t.update(0.0, 1.0, True, DT) and t.connect(), "reconnecting is instant"
    bus.dispatch()
    assert got == [TerminalHacked("t")], "and does not announce a second breach"


def test_terminal_overload_is_a_noise_in_front_of_the_desk_and_respects_critical():
    bus = EventBus()
    got = []
    bus.subscribe(NoiseEmitted, got.append)
    t = Terminal(0.0, 0.0, 30.0, yaw=180.0, terminal_id="t", bus=bus)
    assert t.overload()
    bus.dispatch()
    [noise] = got
    assert noise.source == "terminal_overload" and (noise.x, noise.z) == (0.0, 30.0)
    assert noise.investigate_x == pytest.approx(0.0)
    assert noise.investigate_z == pytest.approx(30.0 - Terminal.INVESTIGATE_DISTANCE), (
        "rotated 180: the chair side faces -Z"
    )
    assert not t.overload(), "already dead"
    critical = Terminal(0, 0, 0, bus=bus, critical=True)
    assert not critical.overload() and not critical.destroyed


def test_destroyed_terminal_cannot_be_jacked_into():
    t = Terminal(0, 0, 0)
    t.overload()
    for _ in range(400):
        assert not t.update(0.0, 1.0, True, DT)
    assert not t.is_hacked and not t.connect() and t.progress == 0.0


def test_vending_machine_trigger_reports_whether_it_fired():
    vm = VendingMachine(0, 0, 0)
    assert vm.trigger_distraction() is True
    assert vm.trigger_distraction() is False and not vm.can_trigger
    vm.update(50.0, 50.0, False, VendingMachine.COOLDOWN + 0.1)
    assert vm.can_trigger
