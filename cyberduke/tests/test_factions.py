"""Factions and their networks: breach one computer, see (and hack) everything on the net -
except the secured computers, which stay private."""

import math
from dataclasses import replace

import pytest

from cyberduke.core.events import EventBus, NetworkBreached, TerminalHacked
from cyberduke.core.geometry import AABB
from cyberduke.entities import Player, SecurityCamera, Terminal, VendingMachine
from cyberduke.levels import FACILITY, build_facility
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems.hacking import Access, HackingSystem, HackKind
from cyberduke.world import FactionData, SlidingDoor

DT = 1 / 60


def run(sim, seconds, inp=None):
    inp = inp or InputState()
    for _ in range(int(round(seconds / DT))):
        sim.update(DT, inp)


def look_at(sim, x, z):
    p = sim.player
    p.yaw = math.degrees(math.atan2(z - p.z, x - p.x))


# ----------------------------------------------------------------------------- level data
def test_the_facility_declares_two_factions_and_every_device_belongs_to_one():
    halcyon, bio = FACILITY.factions
    assert halcyon == FactionData("halcyon", "Halcyon Dynamics Security", color=(1.0, 0.55, 0.1))
    assert bio.faction_id == "halcyon_bio" and bio.name == "Halcyon Biotech Research"
    assert FACILITY.faction_ids_in_use == ["halcyon", "halcyon_bio"]
    assert FACILITY.resolved_factions() == (halcyon, bio)
    # security runs the first three sectors, biotech the greenhouse
    for t in FACILITY.terminals:
        assert t.faction == ("halcyon_bio" if t.terminal_id.startswith("node_") else "halcyon")
    for c in FACILITY.cameras:
        assert c.faction == ("halcyon" if c.camera_id == "airlock_camera" else "halcyon_bio")


def test_undeclared_factions_are_synthesised_so_tiny_levels_need_no_table():
    level = replace(
        build_facility(),
        factions=(),
        guards=tuple(replace(g, faction="street_gang") for g in build_facility().guards),
    )
    ids = [f.faction_id for f in level.resolved_factions()]
    assert ids == ["halcyon", "halcyon_bio", "street_gang"]
    assert level.resolved_factions()[2].name == "STREET GANG"


def test_the_datacore_terminal_is_the_secured_private_computer():
    staging, datacore, maintenance, east, west = FACILITY.terminals
    assert not staging.secured and not staging.intel, "nothing worth stealing on it"
    assert staging.critical, "the only way out of the staging bay: it must not be destroyed"
    assert [t.terminal_id for t in FACILITY.terminals if t.intel] == ["datacore_terminal"], (
        "only data with a function in the mission counts as data"
    )
    assert not maintenance.critical and not maintenance.secured, "the one you may fry"
    assert datacore.secured and datacore.critical and datacore.objective and datacore.intel
    assert datacore.x == 0.0 and datacore.z == 30.0, "the one in the vending-machine/catwalk room"
    assert datacore.intel == "greenhouse door cipher", "its data is the key to the next area"
    assert east.hardened and west.hardened and not (east.secured or west.secured)


# ----------------------------------------------------------------------------- networks
def test_simulation_builds_one_network_per_faction_with_every_target_as_a_member():
    sim = Simulation()
    assert list(sim.hacking.networks) == ["halcyon", "halcyon_bio"]
    net, bio = sim.hacking.networks["halcyon"], sim.hacking.networks["halcyon_bio"]
    assert net.faction.name == "Halcyon Dynamics Security"
    members = {m.hack_id for m in net.members} | {m.hack_id for m in bio.members}
    assert members == {t.hack_id for t in sim.hack_targets}
    assert {m.hack_id for m in bio.members} == {
        "node_east_terminal", "node_west_terminal", "greenhouse_entry_camera",
        "greenhouse_spine_camera", "east_node_camera", "exit_camera", "maze_gate",
        "street_exit", "greenhouse_guard", "greenhouse_elite",
    }  # fmt: skip
    assert not net.breached and not bio.breached


def test_jacking_into_the_staging_terminal_breaches_the_network():
    sim = Simulation()
    got = []
    sim.bus.subscribe(NetworkBreached, got.append)
    sim.player.x, sim.player.z = 3.5, -2.5
    run(sim, 3.0, InputState(interact=True))
    assert sim.terminals[0].is_hacked and sim.hacking.session is sim.terminals[0]
    assert got == [NetworkBreached("halcyon", "staging_terminal")]
    net = sim.hacking.networks["halcyon"]
    assert net.breached and net.breached_via == "staging_terminal"
    assert sim.player.intel == [], "the data is a menu option, not a side effect of jacking in"


def test_a_direct_quickhack_on_a_computer_also_breaches_the_network():
    sim = Simulation()
    look_at(sim, 3.5, -4.0)
    sim.update(DT, InputState(scan=True))
    sim.update(DT, InputState(hack_slot=1))  # OVERRIDE DOORS
    run(sim, 3.0)
    assert sim.hacking.is_breached("halcyon")


def test_breach_highlights_every_member_across_the_map_except_secured_computers():
    sim = Simulation()
    h = sim.hacking
    assert h.statuses() == []
    h.breach("halcyon", via="staging_terminal")
    sim.update(DT, InputState())
    shown = {s.target.hack_id: s for s in h.statuses()}
    expected = {
        "staging_terminal", "vault_maintenance_terminal", "airlock_camera", "neon_cold_vendor",
        "blast_door", "vault_bulkhead", "greenhouse_door", "staging_guard", "vault_guard",
    }  # fmt: skip
    assert set(shown) == expected, "the security net only; the biotech net is separate"
    assert all(s.on_network and s.highlighted for s in shown.values())
    assert "datacore_terminal" not in shown, "secured: private even on a breached network"
    assert not sim.hacking.is_breached("halcyon_bio")
    # Everything on the net is reachable from anywhere, through walls, with no scan.
    for s in shown.values():
        assert s.access is Access.NETWORK
    assert h.access(sim.terminals[1]) is None
    assert not h.on_network(sim.terminals[1])


def test_secured_computer_can_still_be_scanned_and_hacked_directly():
    sim = Simulation()
    h = sim.hacking
    h.breach("halcyon", via="staging_terminal")
    datacore = sim.terminals[1]
    sim.guards[1].x, sim.guards[1].z = 9.0, 41.0
    sim.player.x, sim.player.z = 0.0, 19.0
    look_at(sim, datacore.x, datacore.z)
    sim.update(DT, InputState(scan=True))
    sim.update(DT, InputState())
    st = {s.target.hack_id: s for s in h.statuses()}["datacore_terminal"]
    assert st.scanned and not st.on_network and st.access is Access.DIRECT
    assert h.focus is datacore


def test_hacking_the_secured_computer_does_not_open_the_network():
    sim = Simulation()
    assert sim.hacking.apply(sim.terminals[1], HackKind.STEAL_DATA)
    assert not sim.hacking.is_breached("halcyon")
    # ...and neither does jacking into it
    sim = Simulation()
    sim.bus.publish(TerminalHacked("datacore_terminal"))
    assert not sim.hacking.is_breached("halcyon")


def test_network_access_lets_the_player_hack_the_vault_from_the_staging_bay():
    """Hack the staging terminal, then - without moving - blind the vault guard, open the
    bulkhead, shut the camera down and short the vending machine."""
    sim = Simulation()
    h = sim.hacking
    sim.player.x = 1.0  # a step off the door axis, so the open blast door is not what's aimed at
    assert h.apply(sim.terminals[0], HackKind.OVERRIDE_DOORS)
    sim.bus.dispatch()
    for target, kind in (
        (sim.guards[1], HackKind.BLIND),
        (sim.room.doors["vault_bulkhead"], HackKind.FORCE_OPEN),
        (sim.cameras[0], HackKind.SHUTDOWN),
        (sim.vending_machines[0], HackKind.DISTRACT),
    ):
        tx, ty, tz = target.hack_position
        look_at(sim, tx, tz)
        flat = math.hypot(tx - sim.player.x, tz - sim.player.z)
        sim.player.pitch = math.degrees(math.atan2(ty - sim.player.y, flat))
        sim.update(DT, InputState())
        assert h.focus is target, target.hack_id
        assert h.access(target) is Access.NETWORK
        slot = next(i for i, o in enumerate(h.options(target), 1) if o.kind is kind)
        assert h.request(slot)
        run(sim, h.upload.spec.upload_time + 0.1)
    assert sim.guards[1].is_blinded
    assert sim.room.doors["vault_bulkhead"].unlocked
    assert not sim.cameras[0].is_online
    assert sim.vending_machines[0].is_sparking


def test_two_factions_have_separate_networks():
    bus = EventBus()
    corp = Terminal(0, 0, 0, terminal_id="corp_pc", bus=bus, faction="corp", intel="x")
    corp_cam = SecurityCamera(0, 3, 2, camera_id="corp_cam", faction="corp")
    gang = Terminal(10, 0, 0, terminal_id="gang_pc", bus=bus, faction="gang")
    gang_vm = VendingMachine(12, 0, 0, bus=bus, machine_id="gang_vm", faction="gang")
    gang_door = SlidingDoor(AABB(10, 11, 5, 5.2), None, door_id="gang_door", faction="gang")
    player = Player()
    h = HackingSystem(
        bus, player, [corp, corp_cam, gang, gang_vm, gang_door],
        [FactionData("corp", "Corp"), FactionData("gang", "Gang", (0.2, 1.0, 0.2))],
    )  # fmt: skip
    assert {m.hack_id for m in h.networks["gang"].members} == {"gang_pc", "gang_vm", "gang_door"}
    assert h.apply(corp, HackKind.STEAL_DATA)
    assert h.is_breached("corp") and not h.is_breached("gang")
    assert h.on_network(corp_cam) and not h.on_network(gang_vm) and not h.on_network(gang_door)
    bus.publish(TerminalHacked("gang_pc"))
    assert h.is_breached("gang") and h.on_network(gang_door)


def test_duplicate_target_ids_are_rejected():
    bus = EventBus()
    a = Terminal(0, 0, 0, terminal_id="same", bus=bus)
    b = VendingMachine(1, 0, 0, bus=bus, machine_id="same")
    with pytest.raises(ValueError):
        HackingSystem(bus, Player(), [a, b])
