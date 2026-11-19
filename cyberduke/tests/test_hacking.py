"""Quickhacking: scan -> focus -> choose -> upload -> effect, driven through the real
simulation so every hack engages the code the rest of the game runs on (guard perception,
the detection meter, doors, noises, camera sightings, the win condition)."""

import math
from dataclasses import replace

import pytest

from cyberduke.ai.guard import Inspecting
from cyberduke.config import GUARD, HACKING, SIMULATION
from cyberduke.core.events import DoorUnlocked, HackApplied, NetworkBreached, NoiseEmitted
from cyberduke.core.state import GameState
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems.hacking import HACKS, HACKS_BY_TARGET_KIND, Access, HackKind

DT = 1 / 60


# ----------------------------------------------------------------------------- helpers
def run(sim, seconds, inp=None):
    inp = inp or InputState()
    for _ in range(int(round(seconds / DT))):
        sim.update(DT, inp)


def look_at(sim, x, z, pitch=0.0):
    p = sim.player
    p.yaw = math.degrees(math.atan2(z - p.z, x - p.x))
    p.pitch = pitch


def scan(sim):
    sim.update(DT, InputState(scan=True))
    sim.update(DT, InputState())


def stare_at_player(sim, guard_index=0, dist=4.0):
    """Parks a guard ``dist`` m in front of the player (on the -Z side), stationary and
    attentive, with an effectively infinite inspection time."""
    g = sim.guards[guard_index]
    p = sim.player
    g.x, g.z, g.yaw = p.x, p.z - dist, 0.0
    g.cfg = replace(g.cfg, investigate_duration=1e9)
    g.fsm.transition_to(Inspecting(source=(p.x, p.z)))
    return g


def slot_of(sim, target, kind):
    """1-based menu slot of ``kind`` on ``target`` (what the player presses)."""
    for i, opt in enumerate(sim.hacking.options(target), start=1):
        if opt.kind is kind:
            return i
    raise AssertionError(f"{kind} is not offered on {target.hack_id}")


def hack(sim, target, kind, access=Access.DIRECT):
    """Applies a hack immediately (skipping the upload) and delivers its events."""
    assert sim.hacking.apply(target, kind, access)
    sim.bus.dispatch()


# =============================================================================== vocabulary
def test_every_hack_is_unlocked_for_the_player_until_levelling_exists():
    sim = Simulation()
    assert sim.player.known_hacks == set(HackKind)
    assert all(sim.player.knows(k) for k in HackKind)
    assert set(HACKS) == set(HackKind), "every kind has a spec"
    offered = {k for kinds in HACKS_BY_TARGET_KIND.values() for k in kinds}
    assert offered == set(HackKind), "every hack is offered on some kind of target"


def test_unlearned_hacks_are_listed_but_locked():
    sim = Simulation()
    sim.player.forget_all_hacks()
    sim.player.learn(HackKind.BLIND)
    opts = sim.hacking.options(sim.guards[0])
    assert [(o.spec.label, o.available, o.reason) for o in opts] == [
        ("BLIND", True, ""),
        ("ELECTROCUTE", False, "NOT LEARNED"),
    ]


def test_the_facility_exposes_every_kind_of_electronic_target():
    sim = Simulation()
    kinds = {t.target_kind for t in sim.hack_targets}
    assert kinds == {"terminal", "guard", "door", "camera", "vending"}
    ids = [t.hack_id for t in sim.hack_targets]
    assert len(ids) == len(set(ids))
    assert {t.faction for t in sim.hack_targets} == {"halcyon", "halcyon_bio"}


# =============================================================================== access
def test_direct_access_needs_a_scan_range_and_line_of_sight():
    sim = Simulation()
    h = sim.hacking
    terminal = sim.terminals[0]
    look_at(sim, terminal.x, terminal.z)
    sim.update(DT, InputState())
    assert h.access(terminal) is None, "not scanned yet"
    scan(sim)
    assert h.access(terminal) is Access.DIRECT
    assert h.explain_access(terminal) == "DIRECT"
    # The vault bulkhead is pinged through the blast door but cannot be seen through it.
    bulkhead = sim.room.doors["vault_bulkhead"]
    assert sim.scanner.is_revealed("vault_bulkhead")
    assert h.access(bulkhead) is None and h.explain_access(bulkhead) == "NO LINE OF SIGHT"
    # Range: 14 m. Walk the player back until the terminal drops out.
    sim.player.z = terminal.z + HACKING.quickhack_range + 1.0
    look_at(sim, terminal.x, terminal.z)
    sim.update(DT, InputState())
    assert h.access(terminal) is None and h.explain_access(terminal) == "OUT OF RANGE"


def test_scanned_targets_lose_direct_access_when_the_reveal_expires():
    sim = Simulation()
    look_at(sim, 3.5, -4.0)
    scan(sim)
    assert sim.hacking.access(sim.terminals[0]) is Access.DIRECT
    run(sim, HACKING.reveal_duration + 0.5)
    assert sim.hacking.access(sim.terminals[0]) is None
    assert sim.hacking.statuses() == [], "nothing highlighted any more"


def test_focus_is_the_accessible_target_nearest_the_crosshair():
    sim = Simulation()
    h = sim.hacking
    terminal, guard = sim.terminals[0], sim.guards[0]
    guard.home_x = guard.x = -2.0  # move its patrol lane: same distance band, other side
    guard.z = -4.0
    scan(sim)
    look_at(sim, terminal.x, terminal.z)
    sim.update(DT, InputState())
    assert h.focus is terminal
    look_at(sim, guard.x, guard.z)
    sim.update(DT, InputState())
    assert h.focus is guard
    look_at(sim, -5.5, 6.0)  # the left wall, nothing there within the focus cone
    sim.update(DT, InputState())
    assert h.focus is None
    assert h.request(1) is False and h.notice == "NO TARGET"


def test_line_of_sight_ignores_the_box_the_target_itself_sits_in():
    """A terminal's monitor is inside the terminal's own collision box; a door's lock
    inside the door's obstacle. Those must not occlude themselves."""
    sim = Simulation()
    h = sim.hacking
    terminal = sim.terminals[0]
    eye = (terminal.x, 1.6, terminal.z + 3.0)
    assert h.line_of_sight(eye, terminal.hack_position, [terminal.get_bounds()])
    door = sim.room.doors["blast_door"]
    assert h.line_of_sight((0.0, 1.6, 7.0), door.hack_position, [door.obstacle])
    # ... while a wall in between still blocks
    from cyberduke.core.geometry import AABB

    wall = AABB(-5.0, 5.0, 8.0, 8.2, 0.0, 4.0)
    assert not h.line_of_sight((0.0, 1.6, 7.0), door.hack_position, [wall, door.obstacle])


# =============================================================================== options
def test_computer_options_follow_the_terminal_flags():
    sim = Simulation()
    h = sim.hacking
    staging, datacore, maintenance, east, west = sim.terminals
    assert [(o.spec.label, o.available, o.reason) for o in h.options(staging)] == [
        ("OVERRIDE DOORS", True, ""),
        # no STEAL DATA: it holds nothing that matters, so it is not even on the menu;
        # and it is the only way out of sector 1, so it cannot be destroyed
        ("OVERLOAD", False, "LOCKED: CRITICAL SYSTEM"),
    ]
    assert [(o.spec.label, o.available, o.reason) for o in h.options(maintenance)] == [
        ("OVERRIDE DOORS", False, "NO DOORS LINKED"),
        ("OVERLOAD", True, ""),
    ]
    # The datacore terminal controls no doors and is critical: it can never be destroyed.
    assert [(o.spec.label, o.available, o.reason) for o in h.options(datacore)] == [
        ("OVERRIDE DOORS", False, "NO DOORS LINKED"),
        ("STEAL DATA", True, ""),
        ("OVERLOAD", False, "LOCKED: CRITICAL SYSTEM"),
    ]
    # The greenhouse control nodes are air-gapped: nothing works wirelessly, at any range.
    for node in (east, west):
        assert all(o.reason == "LOCKED: AIR-GAPPED (JACK IN)" for o in h.options(node))
        assert h.access_for(node, sim.player.viewpoint(), []) is None
        assert h.explain_access(node) == "AIR-GAPPED"


def test_essential_guard_cannot_be_knocked_out_but_can_be_blinded():
    sim = Simulation()
    staging, vault = sim.guards[:2]
    assert not staging.essential and vault.essential
    assert [(o.kind, o.available, o.reason) for o in sim.hacking.options(vault)] == [
        (HackKind.BLIND, True, ""),
        (HackKind.ELECTROCUTE, False, "LOCKED: HARDENED ICE"),
    ]
    assert all(o.available for o in sim.hacking.options(staging))


def test_high_security_door_cannot_be_forced_but_a_standard_one_can():
    sim = Simulation()
    blast, bulkhead = sim.room.doors["blast_door"], sim.room.doors["vault_bulkhead"]
    [opt] = sim.hacking.options(blast)
    assert opt.kind is HackKind.FORCE_OPEN and not opt.available
    assert opt.reason == "LOCKED: HIGH SECURITY"
    [opt] = sim.hacking.options(bulkhead)
    assert opt.available


def test_options_reflect_live_state_of_the_target():
    sim = Simulation()
    h = sim.hacking
    cam, vm = sim.cameras[0], sim.vending_machines[0]
    guard, terminal = sim.guards[0], sim.terminals[2]  # the maintenance terminal
    cam.shut_down(5.0)
    assert [(o.available, o.reason) for o in h.options(cam)] == [
        (False, "ALREADY DOWN 5s"),
        (True, ""),  # a dark camera can still be hijacked: the optics answer to the netrunner
    ]
    vm.trigger_distraction()
    assert [(o.available, o.reason) for o in h.options(vm)] == [(False, "RECHARGING")]
    guard.blind()
    assert [(o.available, o.reason) for o in h.options(guard)] == [
        (False, "ALREADY BLINDED"),
        (True, ""),
    ]
    guard.knock_out()
    assert all(o.reason == "OFFLINE" for o in h.options(guard))
    terminal.overload()
    assert all(o.reason == "OFFLINE" for o in h.options(terminal))


# =============================================================================== uploads
def test_request_starts_an_upload_that_applies_after_the_spec_time():
    sim = Simulation()
    h = sim.hacking
    guard = sim.guards[0]
    look_at(sim, guard.x, guard.z)
    scan(sim)
    assert h.focus is guard
    assert h.request(slot_of(sim, guard, HackKind.BLIND))
    up = h.upload
    assert up is not None and up.spec.kind is HackKind.BLIND and up.access is Access.DIRECT
    assert not guard.is_blinded, "not until the upload finishes"
    run(sim, HACKS[HackKind.BLIND].upload_time - 0.1)
    assert h.upload is not None and 0.0 < h.upload.progress < 1.0
    assert not guard.is_blinded
    run(sim, 0.2)
    assert h.upload is None and guard.is_blinded


def test_one_upload_at_a_time_and_unavailable_options_are_rejected():
    sim = Simulation()
    h = sim.hacking
    vault = sim.guards[1]
    sim.player.x, sim.player.z = 0.0, 19.0
    look_at(sim, vault.x, vault.z)
    scan(sim)
    assert h.focus is vault
    assert not h.request(slot_of(sim, vault, HackKind.ELECTROCUTE))
    assert h.notice == "ELECTROCUTE: LOCKED: HARDENED ICE" and h.upload is None
    assert not h.request(9) and h.notice == "NO SUCH HACK"
    assert h.request(slot_of(sim, vault, HackKind.BLIND))
    assert not h.request(slot_of(sim, vault, HackKind.BLIND))
    assert h.notice == "UPLOAD IN PROGRESS"


def test_number_keys_drive_the_focused_targets_menu():
    """End to end through InputState: [Tab] scans, [2] uploads the second option."""
    sim = Simulation()
    guard = sim.guards[0]
    look_at(sim, guard.x, guard.z)
    sim.update(DT, InputState(scan=True))
    sim.update(DT, InputState(hack_slot=2))  # ELECTROCUTE
    assert sim.hacking.upload is not None and sim.hacking.upload.spec.kind is HackKind.ELECTROCUTE
    run(sim, HACKS[HackKind.ELECTROCUTE].upload_time + 0.1)
    assert guard.is_down


def test_hack_applied_event_is_published_with_target_kind_and_route():
    sim = Simulation()
    got = []
    sim.bus.subscribe(HackApplied, got.append)
    hack(sim, sim.guards[0], HackKind.BLIND, Access.NETWORK)
    assert got == [HackApplied("staging_guard", "blind", "network")]
    assert "BLIND: STAGING GUARD" in sim.hacking.log


def test_a_hack_the_target_refuses_reports_failure_instead_of_applying():
    sim = Simulation()
    blast = sim.room.doors["blast_door"]
    assert not sim.hacking.apply(blast, HackKind.FORCE_OPEN)
    assert not blast.unlocked
    assert sim.hacking.log[-1] == "FORCE OPEN REFUSED: BLAST DOOR (LOCKED: HIGH SECURITY)"
    # ...and a primitive that refuses at the last moment is reported as a failure
    vm = sim.vending_machines[0]
    vm.trigger_distraction()
    vm.cooldown = 0.0  # looks available to the rules...
    vm.is_sparking = True
    vm.trigger_distraction = lambda: False  # ...but the machine itself says no
    assert not sim.hacking.apply(vm, HackKind.DISTRACT)
    assert sim.hacking.log[-1] == "DISTRACT FAILED ON NEON COLD VENDOR"


# =============================================================================== BLIND
def test_blinded_guard_cannot_detect_the_player_for_the_duration_then_sees_again():
    sim = Simulation()
    g = stare_at_player(sim)
    run(sim, 0.5)
    assert sim.detection.level > 0.0, "the guard is looking straight at the player"
    hack(sim, g, HackKind.BLIND)
    assert g.is_blinded and g.state == "BLINDED"
    level_at_blind = sim.detection.level
    run(sim, HACKING.blind_duration - 0.2)
    assert g.is_blinded
    assert sim.detection.level == 0.0, "no sightings for six seconds: the meter decayed away"
    assert sim.detection.level < level_at_blind
    assert not g.can_see_player(sim.player)
    run(sim, 0.5)
    assert not g.is_blinded, "optics back"
    run(sim, 0.3)
    assert sim.detection.level > 0.0, "the player is still standing there"


def test_blinded_guard_stands_still_and_then_resumes_its_routine():
    sim = Simulation()
    g = sim.guards[0]
    run(sim, 0.5)
    x, z = g.x, g.z
    hack(sim, g, HackKind.BLIND)
    run(sim, 2.0)
    assert (g.x, g.z) == (x, z)
    assert g.pose.arm_l[0] < -90, "hands up at the visor"
    run(sim, HACKING.blind_duration)
    assert g.state in {"WALKING_TO_PATROL", "PACING", "WALKING_TO_WALL"}


def test_blinding_an_alerted_guard_makes_it_lose_track_of_the_player():
    sim = Simulation()
    g = sim.guards[0]
    g.cfg = replace(g.cfg, pacing_duration=1e9)
    g.x, g.z, g.yaw = 0.0, -2.0, 0.0
    sim.player.x, sim.player.z = 0.0, 4.0
    sim.update(DT, InputState())
    assert g.state == "ALERTED"
    hack(sim, g, HackKind.BLIND)
    # hide while it is blind
    sim.player.x, sim.player.z, sim.player.crouching = -4.5, 4.6, True
    run(sim, HACKING.blind_duration + 0.5)
    assert not g.is_reacting and g.state in {"WALKING_TO_PATROL", "PACING"}


# =============================================================================== ELECTROCUTE
def test_electrocuted_guard_is_permanently_down_and_never_sees_or_reacts():
    sim = Simulation()
    g = stare_at_player(sim)
    hack(sim, g, HackKind.ELECTROCUTE)
    assert g.is_down and g.state == "KNOCKED_OUT" and g.shock_timer > 0.0
    run(sim, 30.0)
    assert g.is_down and sim.detection.level == 0.0 and g.fall_angle >= 75.0
    assert not g.can_see_player(sim.player)
    assert g.update(sim.player, DT, obstacles=[]) is False
    # A noise next to it is ignored
    sim.bus.publish(NoiseEmitted(g.x + 1.0, g.z, 10.0, g.x + 1.0, g.z))
    assert g.state == "KNOCKED_OUT"
    assert sim.state is GameState.PLAYING


def test_reset_revives_a_knocked_out_guard():
    sim = Simulation()
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)
    run(sim, 1.0)
    sim.update(DT, InputState(reset=True))
    g = sim.guards[0]
    assert not g.is_down and g.state == "PACING" and g.fall_angle == 0.0 and g.shock_timer == 0.0


def test_essential_guard_refuses_the_electrocute_hack_through_the_menu():
    sim = Simulation()
    vault = sim.guards[1]
    sim.player.x, sim.player.z = 0.0, 19.0
    look_at(sim, vault.x, vault.z)
    scan(sim)
    assert not sim.hacking.request(slot_of(sim, vault, HackKind.ELECTROCUTE))
    run(sim, 5.0)
    assert not vault.is_down


# =============================================================================== computers
def test_override_doors_on_the_staging_terminal_opens_the_blast_door_and_breaches_the_net():
    sim = Simulation()
    terminal = sim.terminals[0]
    door = sim.room.doors["blast_door"]
    assert door.obstacle in sim.obstacles
    hack(sim, terminal, HackKind.OVERRIDE_DOORS)
    assert door.unlocked and terminal.doors_overridden
    assert not terminal.is_hacked, "a remote override is not a full jack-in"
    run(sim, 2.0)
    assert door.passable and door.obstacle not in sim.obstacles
    assert sim.hacking.is_breached("halcyon")


def test_steal_data_copies_the_intel_once_and_the_datacore_cipher_opens_the_greenhouse():
    sim = Simulation()
    staging, datacore = sim.terminals[:2]
    assert not sim.hacking.apply(staging, HackKind.STEAL_DATA), "nothing on it to steal"
    assert sim.player.intel == []
    assert sim.state is GameState.PLAYING
    hack(sim, datacore, HackKind.STEAL_DATA)
    assert [(o.available, o.reason) for o in sim.hacking.options(datacore)][1] == (
        False,
        "DATA ALREADY COPIED",
    )
    assert sim.player.intel[-1] == "greenhouse door cipher"
    sim.update(DT, InputState())
    door = sim.room.doors["greenhouse_door"]
    assert not door.unlocked, "the cipher is read at the door, not from across the vault"
    sim.player.x, sim.player.z = 0.0, 40.5
    sim.update(DT, InputState())
    assert door.unlocked and sim.hacking.log[-1] == "CIPHER ACCEPTED: GREENHOUSE DOOR UNLOCKED"


def test_datacore_can_be_stolen_directly_from_the_vault_entrance_without_touching_it():
    """The secured terminal is hackable with a line of sight: scan it from the doorway and
    upload STEAL DATA - no need to walk to the desk. (It is still not on the network.)"""
    sim = Simulation()
    datacore = sim.terminals[1]
    sim.guards[1].x, sim.guards[1].z = 9.0, 41.0  # park the vault guard out of the way
    sim.player.x, sim.player.z = 0.0, 19.0
    look_at(sim, datacore.x, datacore.z, pitch=-3.0)
    scan(sim)
    assert sim.hacking.focus is datacore and sim.hacking.access(datacore) is Access.DIRECT
    assert sim.hacking.request(slot_of(sim, datacore, HackKind.STEAL_DATA))
    run(sim, HACKS[HackKind.STEAL_DATA].upload_time + 0.1)
    assert datacore.intel_stolen and not datacore.is_hacked
    assert "greenhouse door cipher" in sim.player.intel


def test_overload_destroys_the_terminal_and_pulls_the_guard_over_to_investigate():
    sim = Simulation()
    terminal, guard = sim.terminals[2], sim.guards[1]  # vault maintenance desk, vault guard
    noises = []
    sim.bus.subscribe(NoiseEmitted, noises.append)
    hack(sim, terminal, HackKind.OVERLOAD)
    assert terminal.destroyed and terminal.spark_timer > 0.0
    [noise] = noises
    assert noise.source == "terminal_overload" and (noise.x, noise.z) == (7.5, 35.5)
    assert guard.state == "INVESTIGATING"
    front = terminal.local_to_world(0.0, terminal.INVESTIGATE_DISTANCE)
    assert guard.investigate_target == pytest.approx(front)
    for _ in range(int(14 / DT)):
        sim.update(DT, InputState())
        if guard.arrived_at_distraction:
            break
    assert guard.arrived_at_distraction, "walked to the fried desk"
    assert math.hypot(guard.x - front[0], guard.z - front[1]) < 0.7
    # Dead for good: no jack-in, no more options
    sim.player.x, sim.player.z = front[0], front[1]
    run(sim, 3.0, InputState(interact=True))
    assert not terminal.is_hacked and terminal.progress == 0.0
    assert sim.hacking.session is None
    assert all(not o.available for o in sim.hacking.options(terminal))


def test_a_fried_computer_controls_nothing_any_more():
    """The design trade-off that makes OVERLOAD a decision. (It is why the one computer
    that opens the way out of the staging bay is flagged critical.)"""
    sim = Simulation()
    terminal = sim.terminals[2]
    terminal.door_ids = ("blast_door",)  # pretend this desk controlled a door
    hack(sim, terminal, HackKind.OVERLOAD)
    assert not terminal.unlock_doors()
    assert not sim.room.doors["blast_door"].unlocked
    assert not sim.hacking.apply(sim.room.doors["blast_door"], HackKind.FORCE_OPEN)


def test_critical_terminal_survives_an_overload_attempt():
    sim = Simulation()
    datacore = sim.terminals[1]
    assert not sim.hacking.apply(datacore, HackKind.OVERLOAD)
    assert not datacore.destroyed and datacore.has_intel


# =============================================================================== doors
def test_force_open_unlocks_the_vault_bulkhead_and_lets_the_player_through():
    sim = Simulation()
    bulkhead = sim.room.doors["vault_bulkhead"]
    unlocked = []
    sim.bus.subscribe(DoorUnlocked, unlocked.append)
    sim.player.x, sim.player.z = 0.0, 14.0
    look_at(sim, 0.0, 18.0)
    scan(sim)
    assert sim.hacking.focus is bulkhead
    assert sim.hacking.request(1)
    run(sim, HACKS[HackKind.FORCE_OPEN].upload_time + 0.1)
    assert bulkhead.unlocked and unlocked == [DoorUnlocked("vault_bulkhead", "quickhack")]
    run(sim, 2.0)
    assert bulkhead.passable
    look_at(sim, 0.0, 30.0)
    run(sim, 2.5, InputState(forward=1.0))
    assert sim.player.z > 18.5, "walked through the open bulkhead into the vault"


def test_blast_door_still_opens_from_its_terminal_when_forced_open_fails():
    sim = Simulation()
    blast = sim.room.doors["blast_door"]
    assert not sim.hacking.apply(blast, HackKind.FORCE_OPEN)
    hack(sim, sim.terminals[0], HackKind.OVERRIDE_DOORS)
    assert blast.unlocked


# =============================================================================== cameras
def _player_under_the_camera(sim):
    sim.room.doors["blast_door"].open_amount = sim.room.doors["blast_door"].travel
    sim.player.x, sim.player.z = 0.0, 14.0  # standing, in the corridor the camera sweeps


def test_shut_down_camera_stops_reporting_and_freezes_until_it_reboots():
    sim = Simulation()
    cam = sim.cameras[0]
    _player_under_the_camera(sim)
    seen = 0.0
    for _ in range(int(8 / DT)):  # one full sweep period is ~5 s
        sim.update(DT, InputState())
        seen = max(seen, sim.detection.level)
        if sim.detection.level > 20:
            break
    assert seen > 0.0, "the sweeping camera catches a player standing in the corridor"
    sim.reset()
    _player_under_the_camera(sim)
    hack(sim, cam, HackKind.SHUTDOWN)
    assert not cam.is_online and cam.shutdown_timer == HACKING.camera_shutdown_duration
    yaw = cam.current_yaw
    run(sim, 8.0)
    assert sim.detection.level == 0.0 and not cam.is_alerted
    assert cam.current_yaw == yaw, "the head does not move without power"
    # sit out the rest of the three minutes (the camera alone; the world is covered above)
    for _ in range(int((HACKING.camera_shutdown_duration - 8.0 + 0.05) / DT)):
        cam.update(sim.player, DT, [])
    assert cam.is_online
    for _ in range(int(8 / DT)):
        sim.update(DT, InputState())
        if sim.detection.level > 0:
            break
    assert sim.detection.level > 0.0, "back online, back to catching the player"


def test_hijacked_camera_provides_the_viewpoint_and_does_not_report_the_player():
    sim = Simulation()
    cam = sim.cameras[0]
    _player_under_the_camera(sim)
    hack(sim, cam, HackKind.HIJACK)
    assert cam.hijacked and sim.hijacked_camera is cam
    view = sim.viewpoint
    assert view.source == "airlock_camera" and view.eye == cam.lens
    assert view.fy < 0, "the camera looks down into the corridor"
    assert view.fz > 0.9, "along +Z (mount yaw 0, centred on hijack)"
    run(sim, 6.0)
    assert sim.detection.level == 0.0 and not cam.is_alerted
    sim.update(DT, InputState(disconnect=True))
    assert not cam.hijacked and sim.hijacked_camera is None
    assert sim.viewpoint.source == "player"


def test_while_hijacking_the_body_stays_put_and_the_mouse_pans_the_camera_within_its_sweep():
    sim = Simulation()
    cam = sim.cameras[0]
    hack(sim, cam, HackKind.HIJACK)
    x, z, yaw = sim.player.x, sim.player.z, sim.player.yaw
    run(sim, 1.0, InputState(forward=1.0, strafe=1.0, look_dx=40.0, toggle_crouch=True))
    assert (sim.player.x, sim.player.z, sim.player.yaw) == (x, z, yaw)
    assert not sim.player.crouching
    assert cam.hijack_yaw == pytest.approx(-cam.sweep_angle), "panned right, stopped at the stop"
    assert cam.current_yaw == cam.hijack_yaw
    run(sim, 1.0, InputState(look_dx=-400.0))
    assert cam.hijack_yaw == pytest.approx(cam.sweep_angle)


def test_mouse_right_turns_the_view_right_for_both_the_player_and_a_hijacked_camera():
    """The two yaw conventions differ; the player must not feel the difference."""

    def right_of(view):
        fx, _, fz = view.forward
        return (-fz, fx)  # forward x up (Y), on the ground plane: the viewer's right-hand side

    sim = Simulation()
    before = sim.viewpoint
    sim.update(DT, InputState(look_dx=30.0))
    after = sim.viewpoint
    rx, rz = right_of(before)
    assert after.fx * rx + after.fz * rz > 0, "player: forward swung toward the right"
    hack(sim, sim.cameras[0], HackKind.HIJACK)
    before = sim.viewpoint
    sim.update(DT, InputState(look_dx=30.0))
    after = sim.viewpoint
    rx, rz = right_of(before)
    assert after.fx * rx + after.fz * rz > 0, "camera: forward swung toward the right"


def test_direct_hacks_can_be_launched_from_a_hijacked_cameras_point_of_view():
    """Through the airlock camera the bulkhead is 7 m away in plain sight, even though the
    player's body is back in the staging bay behind a shut blast door."""
    sim = Simulation()
    cam, bulkhead = sim.cameras[0], sim.room.doors["vault_bulkhead"]
    sim.player.x, sim.player.z = 0.0, 6.0
    hack(sim, cam, HackKind.HIJACK)
    scan(sim)
    assert sim.hacking.access(bulkhead) is Access.DIRECT
    assert sim.hacking.focus is bulkhead, "the centred feed looks straight at it"
    assert sim.hacking.request(1)
    run(sim, HACKS[HackKind.FORCE_OPEN].upload_time + 0.1)
    assert bulkhead.unlocked


def test_shutting_down_a_hijacked_camera_drops_the_feed():
    sim = Simulation()
    cam = sim.cameras[0]
    hack(sim, cam, HackKind.HIJACK)
    hack(sim, cam, HackKind.SHUTDOWN)
    assert not cam.hijacked and sim.hijacked_camera is None
    assert sim.viewpoint.source == "player"
    # ...but the netrunner can climb straight back into the dark camera to look around
    assert sim.hacking.apply(cam, HackKind.HIJACK)
    assert sim.hijacked_camera is cam and not cam.is_online


def test_a_shut_down_camera_stays_dark_for_the_full_three_minutes_even_if_hijacked():
    """Regression: looking through a camera used to let it (and every other dark camera)
    come back online early. The countdown is independent of what the netrunner does."""
    sim = Simulation()
    cams = {c.hack_id: c for c in sim.cameras}
    _open_all_doors(sim)
    for cam in sim.cameras:
        hack(sim, cam, HackKind.SHUTDOWN)
    assert all(c.shutdown_timer == pytest.approx(180.0) for c in sim.cameras)
    hack(sim, cams["airlock_camera"], HackKind.HIJACK)  # hop into a dark one and look around
    run(sim, 30.0, InputState(look_dx=20.0))
    assert all(not c.is_online for c in sim.cameras), "none of them rebooted"
    assert cams["airlock_camera"].hijacked and sim.viewpoint.source == "airlock_camera"
    assert cams["airlock_camera"].shutdown_timer == pytest.approx(150.0, abs=0.2)
    assert cams["airlock_camera"].head_droop < 0.05, "the head lifts while you drive it"
    sim.update(DT, InputState(disconnect=True))
    run(sim, 1.0)
    assert cams["airlock_camera"].head_droop > 0.9, "and hangs again when you let go"
    # the rest of the three minutes, ticking the cameras alone (the sim is covered above)
    for _ in range(int(149.0 / DT)):
        for cam in sim.cameras:
            cam.update(sim.player, DT, [])
    assert all(c.is_online for c in sim.cameras), "back up after three minutes, not before"


# =============================================================================== vending
def test_vending_machine_hack_distracts_the_vault_guard_and_can_be_repeated():
    sim = Simulation()
    vm, guard = sim.vending_machines[0], sim.guards[1]
    hack(sim, vm, HackKind.DISTRACT)
    assert vm.is_sparking and guard.state == "INVESTIGATING"
    assert guard.investigate_target == (vm.x, vm.z + vm.INVESTIGATE_OFFSET_Z)
    assert not sim.hacking.apply(vm, HackKind.DISTRACT), "recharging"
    assert sim.hacking.options(vm)[0].reason == "RECHARGING"
    run(sim, vm.COOLDOWN + 0.1)
    assert sim.hacking.options(vm)[0].available
    hack(sim, vm, HackKind.DISTRACT)
    assert vm.is_sparking


def test_vending_machine_can_be_hacked_from_the_staging_bay_over_the_network():
    sim = Simulation()
    vm, guard = sim.vending_machines[0], sim.guards[1]
    sim.player.x, sim.player.z = 0.0, 6.0
    hack(sim, sim.terminals[0], HackKind.OVERRIDE_DOORS)  # breach
    look_at(sim, vm.x, vm.z)
    sim.update(DT, InputState())
    assert sim.hacking.access(vm) is Access.NETWORK and sim.hacking.focus is vm
    assert sim.hacking.request(1)
    run(sim, HACKS[HackKind.DISTRACT].upload_time + 0.1)
    assert vm.is_sparking and guard.state == "INVESTIGATING"


# =============================================================================== events
def test_network_breached_is_published_once_per_network():
    sim = Simulation()
    got = []
    sim.bus.subscribe(NetworkBreached, got.append)
    hack(sim, sim.terminals[0], HackKind.OVERRIDE_DOORS)
    hack(sim, sim.terminals[2], HackKind.OVERLOAD)
    assert got == [NetworkBreached("halcyon", "staging_terminal")]


def test_reset_clears_uploads_focus_networks_and_the_feed():
    sim = Simulation()
    hack(sim, sim.terminals[0], HackKind.OVERRIDE_DOORS)
    hack(sim, sim.cameras[0], HackKind.HIJACK)
    look_at(sim, 3.5, -4.0)
    scan(sim)
    sim.hacking.request(2)
    sim.update(DT, InputState(reset=True))
    h = sim.hacking
    assert h.upload is None and h.hijacked_camera is None and not h.is_breached("halcyon")
    assert h.statuses() == [] and sim.player.intel == []
    assert sim.player.known_hacks == set(HackKind), "a respawn does not unlearn hacks"


def test_caught_while_hijacking_respawns_cleanly():
    sim = Simulation()
    g = stare_at_player(sim)
    hack(sim, sim.cameras[0], HackKind.HIJACK)
    run(sim, 3.0)
    assert sim.state is GameState.CAUGHT
    run(sim, SIMULATION.caught_respawn_delay + 0.1)
    assert sim.state is GameState.PLAYING and sim.hijacked_camera is None
    assert not g.is_incapacitated


def test_guard_config_for_detection_is_untouched_by_the_hacking_layer():
    """Sanity: blinding uses the FSM, not a tweak of the perception constants."""
    assert GUARD.alert_range == 11.5 and GUARD.alert_fov == 55.0


# =============================================================================== linked feeds
def _open_all_doors(sim):
    for d in sim.room.doors.values():
        d.open_amount = d.travel


def test_arrow_keys_switch_between_the_live_cameras_of_the_same_network():
    sim = Simulation()
    h = sim.hacking
    cams = {c.hack_id: c for c in sim.cameras}
    bio = ["greenhouse_entry_camera", "greenhouse_spine_camera", "east_node_camera", "exit_camera"]
    hack(sim, cams["greenhouse_entry_camera"], HackKind.HIJACK)
    assert [c.hack_id for c in h.linked_cameras(cams["greenhouse_entry_camera"])] == bio
    assert "airlock_camera" not in [c.hack_id for c in h.linked_cameras(sim.hijacked_camera)]
    seen = [sim.hijacked_camera.hack_id]
    for _ in range(4):
        sim.update(DT, InputState(cycle_camera=1))
        seen.append(sim.hijacked_camera.hack_id)
    assert seen == bio + [bio[0]], "forward through every feed and back round"
    assert sim.viewpoint.source == bio[0]
    sim.update(DT, InputState(cycle_camera=-1))
    assert sim.hijacked_camera.hack_id == bio[-1], "backwards wraps too"
    live = [c for c in sim.cameras if c.hijacked]
    assert live == [sim.hijacked_camera], "exactly one camera is hijacked at a time"


def test_feed_switching_includes_dark_cameras_and_needs_a_feed_to_switch_from():
    sim = Simulation()
    h = sim.hacking
    cams = {c.hack_id: c for c in sim.cameras}
    assert not h.cycle_camera(1), "not hijacking anything"
    cams["greenhouse_spine_camera"].shut_down(30.0)
    hack(sim, cams["greenhouse_entry_camera"], HackKind.HIJACK)
    assert h.cycle_camera(1) and sim.hijacked_camera.hack_id == "greenhouse_spine_camera"
    assert not sim.hijacked_camera.is_online, "a dark camera is still a feed you can use"
    # the airlock camera is alone on its network: nothing to switch to
    hack(sim, cams["airlock_camera"], HackKind.HIJACK)
    assert sim.hijacked_camera is cams["airlock_camera"] and not cams["east_node_camera"].hijacked
    assert not h.cycle_camera(1) and h.notice == "NO OTHER FEED ON THIS NETWORK"


def test_mouse_tilts_a_hijacked_camera_up_and_down():
    sim = Simulation()
    cam = sim.cameras[0]
    hack(sim, cam, HackKind.HIJACK)
    run(sim, 0.5, InputState(look_dy=-300.0))  # mouse up
    assert cam.hijack_tilt == cam.MIN_TILT and sim.viewpoint.fy > 0.0
    run(sim, 0.5, InputState(look_dy=900.0))  # mouse down
    assert cam.hijack_tilt == cam.MAX_TILT and sim.viewpoint.fy < -0.9


def test_hijack_chain_through_the_greenhouse_from_the_spore_lock():
    """Scan and hack from the cameras themselves: each greenhouse camera is within
    quickhack range and line of sight of the next, so from just inside the greenhouse door
    the netrunner can hop entry -> east node -> spine -> exit without entering the maze,
    seeing the gate, both control nodes and the exit from the feeds."""
    sim = Simulation()
    h = sim.hacking
    _open_all_doors(sim)
    sim.player.x, sim.player.z = 0.0, 44.0
    look_at(sim, 0.0, 47.6, pitch=20.0)
    scan(sim)
    chain = [
        "greenhouse_entry_camera",
        "east_node_camera",
        "greenhouse_spine_camera",
        "exit_camera",
    ]
    cams = {c.hack_id: c for c in sim.cameras}
    assert h.focus is cams[chain[0]] and h.access(cams[chain[0]]) is Access.DIRECT
    assert h.request(slot_of(sim, cams[chain[0]], HackKind.HIJACK))
    run(sim, HACKS[HackKind.HIJACK].upload_time + 0.1)
    for prev, nxt in zip(chain, chain[1:], strict=False):
        assert sim.viewpoint.source == prev
        run(sim, HACKING.scan_cooldown)  # let the scanner recharge
        scan(sim)  # a scan from the feed, around the camera
        target = cams[nxt]
        assert sim.scanner.is_revealed(nxt), f"{nxt} is within scan range of {prev}"
        assert h.access_for(target) is Access.DIRECT, f"{nxt}: {h.explain_access(target)}"
        assert h.apply(target, HackKind.HIJACK, Access.DIRECT)
        assert not cams[prev].hijacked and sim.hijacked_camera is target
    assert sim.viewpoint.source == "exit_camera"
    assert sim.scanner.is_revealed("street_exit") and sim.scanner.is_revealed("maze_gate")
    assert sim.scanner.is_revealed("node_east_terminal")
    assert h.explain_access(h.target("node_east_terminal")) == "AIR-GAPPED"
    assert sim.player.z == 44.0, "the body never moved"


# =============================================================================== greenhouse
def test_keyed_door_opens_only_for_a_player_carrying_its_cipher_and_only_up_close():
    sim = Simulation()
    door = sim.room.doors["greenhouse_door"]
    sim.player.x, sim.player.z = 0.0, 40.5
    assert sim.keyed_door_in_reach == (door, False)
    run(sim, 1.0)
    assert not door.unlocked
    assert not sim.hacking.apply(door, HackKind.FORCE_OPEN)
    assert sim.hacking.options(door)[0].reason == "LOCKED: ENCRYPTED (NEEDS GREENHOUSE DOOR CIPHER)"
    sim.player.intel.append("greenhouse door cipher")
    sim.player.z = 36.0  # too far
    sim.update(DT, InputState())
    assert not door.unlocked and sim.keyed_door_in_reach is None
    sim.player.z = 40.5
    sim.update(DT, InputState())
    assert door.unlocked and sim.keyed_door_in_reach == (door, True)
    run(sim, 1.5)
    assert door.passable


def test_air_gapped_terminal_is_jack_in_only_and_jacking_in_breaches_its_network():
    sim = Simulation()
    h = sim.hacking
    east = h.target("node_east_terminal")
    _open_all_doors(sim)
    hack(sim, h.target("east_node_camera"), HackKind.SHUTDOWN)  # it watches this very desk
    sim.player.x, sim.player.z = 9.6, 56.1
    look_at(sim, east.x, east.z)
    scan(sim)
    assert sim.scanner.is_revealed("node_east_terminal")
    assert h.access(east) is None and h.focus is not east
    assert not h.is_breached("halcyon_bio")
    run(sim, 3.0, InputState(interact=True))
    assert east.is_hacked and h.session is east and h.is_breached("halcyon_bio")
    assert h.on_network(h.target("greenhouse_elite")) and h.on_network(h.target("exit_camera"))
    assert not h.on_network(east), "air-gapped: still not reachable wirelessly"
    # its gate opens from its own interface
    assert not sim.room.doors["maze_gate"].unlocked
    slot = next(i for i, o in enumerate(h.session_options(), 1) if o.spec.label == "UNLOCK DOORS")
    sim.update(DT, InputState(hack_slot=slot))
    run(sim, 1.2)
    assert sim.room.doors["maze_gate"].unlocked


def test_the_facility_exit_hands_over_to_the_street_and_the_game_goes_on():
    sim = Simulation()
    sim.terminals[1].intel_stolen = True
    sim.update(DT, InputState())
    assert sim.state is GameState.PLAYING
    sim.player.x, sim.player.z = 0.0, 78.6
    run(sim, SIMULATION.level_fade_out + 2 * DT)  # (the picture fades out at the door)
    # the facility's exit zone is a hand-over: the street is the next level, and there is
    # no winning there either - the alley mouth is a wall now
    assert sim.state is GameState.PLAYING and sim.level.name == "Night Market Alley"
    assert sim.exit_zone is None


def test_overloaded_computer_holds_the_guard_for_forty_seconds():
    sim = Simulation()
    terminal, guard = sim.terminals[2], sim.guards[1]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)  # nobody else may catch us meanwhile
    hack(sim, terminal, HackKind.OVERLOAD)
    for _ in range(int(15 / DT)):
        sim.update(DT, InputState())
        if guard.arrived_at_distraction:
            break
    assert guard.arrived_at_distraction
    assert guard.fsm.state.duration == HACKING.terminal_overload_inspect_duration == 40.0
    run(sim, 35.0)
    assert guard.state == "INSPECTING", "still staring at the fried desk"
    assert terminal.spark_timer > 0.0 or terminal.destroyed
    run(sim, 6.0)
    assert guard.state != "INSPECTING"


def test_shorted_vending_machine_is_hit_for_a_while_and_watched_for_forty_five_seconds():
    sim = Simulation()
    vm, guard = sim.vending_machines[0], sim.guards[1]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)  # the staging guard must not catch us meanwhile
    hack(sim, vm, HackKind.DISTRACT)
    for _ in range(int(15 / DT)):
        sim.update(DT, InputState())
        if guard.arrived_at_distraction:
            break
    assert guard.arrived_at_distraction and guard.is_banging and vm.is_sparking
    assert guard.fsm.state.duration == HACKING.vending_inspect_duration == 45.0
    run(sim, sim.guards[1].cfg.bang_duration + 0.2)
    assert not guard.is_banging and guard.state == "INSPECTING"
    run(sim, 30.0)
    assert guard.state == "INSPECTING"
    run(sim, 4.0)
    assert guard.state != "INSPECTING"


# =============================================================================== terminal interface
def _jack_in(sim, terminal, x, z, seconds=3.0):
    sim.player.x, sim.player.z = x, z
    for _ in range(int(seconds / DT)):
        sim.update(DT, InputState(interact=True))
        if sim.hacking.session is terminal:
            return True
    return False


def test_holding_e_opens_the_terminal_interface_instead_of_doing_everything_at_once():
    sim = Simulation()
    h = sim.hacking
    terminal = sim.terminals[0]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)
    sim.player.x, sim.player.z = 3.5, -2.5
    sim.update(DT, InputState(interact=True))
    assert 0 < terminal.progress < 100 and h.session is None, "the link takes a moment"
    assert _jack_in(sim, terminal, 3.5, -2.5)
    assert terminal.is_connected and terminal.is_hacked
    assert not terminal.doors_overridden and not terminal.intel_stolen
    assert [(o.spec.label, o.available, o.reason) for o in h.session_options()] == [
        ("UNLOCK DOORS", True, ""),
        ("HIJACK CAMERAS", True, ""),
        ("DISABLE CAMERAS", True, ""),
        ("SHORT CIRCUIT (DISTRACT & DESTROY)", False, "LOCKED: CRITICAL SYSTEM"),
    ], "no STEAL DATA row: this computer holds nothing that matters"
    assert h.is_breached("halcyon"), "jacking in is the breach"


def test_terminal_interface_options_do_what_they_say():
    sim = Simulation()
    h = sim.hacking
    terminal = sim.terminals[0]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)
    assert _jack_in(sim, terminal, 3.5, -2.5)
    sim.update(DT, InputState(hack_slot=1))  # UNLOCK DOORS
    run(sim, 1.1)
    assert sim.room.doors["blast_door"].unlocked and terminal.doors_overridden
    assert h.session_options()[0].reason == "DOORS ALREADY OPEN"
    sim.update(DT, InputState(hack_slot=3))  # DISABLE CAMERAS: every camera on this network
    run(sim, 1.6)
    assert not sim.cameras[0].is_online, "the airlock camera is the security net's only one"
    assert all(c.is_online for c in sim.cameras[1:]), "the biotech cameras are untouched"
    assert h.session_options()[2].reason == "CAMERAS ALREADY DOWN"
    assert h.session is terminal
    sim.update(DT, InputState(disconnect=True))
    assert h.session is None and not terminal.is_connected
    # reconnecting to a breached terminal is instant
    sim.update(DT, InputState(interact=True))
    assert h.session is terminal


def test_hijack_cameras_from_a_computer_puts_the_player_in_a_feed_and_closes_the_desk():
    sim = Simulation()
    h = sim.hacking
    terminal = sim.terminals[0]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)
    assert _jack_in(sim, terminal, 3.5, -2.5)
    slot = next(i for i, o in enumerate(h.session_options(), 1) if o.spec.label == "HIJACK CAMERAS")
    sim.update(DT, InputState(hack_slot=slot))
    run(sim, 1.1)
    assert sim.hijacked_camera is sim.cameras[0], "the camera on this computer's network"
    assert sim.viewpoint.source == "airlock_camera"
    assert h.session is None and not terminal.is_connected, "you are behind a lens now"
    # and from there the usual feed controls work
    sim.update(DT, InputState(disconnect=True))
    assert sim.hijacked_camera is None


def test_short_circuit_is_offered_on_an_expendable_computer_and_destroys_it():
    sim = Simulation()
    h = sim.hacking
    maintenance = sim.terminals[2]
    sim.guards[1].x, sim.guards[1].z = 2.1, 24.0  # keep the vault guard at the far end
    _open_all_doors(sim)
    front = maintenance.front
    assert _jack_in(sim, maintenance, front[0], front[1])
    labels = {o.spec.label: o for o in h.session_options()}
    assert labels["SHORT CIRCUIT (DISTRACT & DESTROY)"].available
    assert labels["UNLOCK DOORS"].reason == "NO DOORS LINKED"
    slot = next(i for i, o in enumerate(h.session_options(), 1) if o.spec.label.startswith("SHORT"))
    sim.update(DT, InputState(hack_slot=slot))
    run(sim, 1.3)
    assert maintenance.destroyed and h.session is None
    assert sim.guards[1].state in {"INVESTIGATING", "ALERTED", "SUSPICIOUS"}
    assert not _jack_in(sim, maintenance, front[0], front[1], seconds=1.0), (
        "nothing left to talk to"
    )


def test_terminal_interface_rejects_nonsense_and_one_action_at_a_time():
    sim = Simulation()
    h = sim.hacking
    terminal = sim.terminals[0]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)
    assert _jack_in(sim, terminal, 3.5, -2.5)
    assert not h.session_request(9) and h.notice == "NO SUCH OPTION"
    assert not h.session_request(4), "critical system"
    assert h.session_request(1)
    assert not h.session_request(3) and h.notice == "BUSY"
    run(sim, 1.1)
    assert h.session_request(3), "free again once the first one finished"
    h.end_session()
    assert not h.session_request(1) and h.notice == "NO TERMINAL"


def test_steal_data_is_offered_at_the_desk_only_where_there_is_data_and_the_link_bar_once():
    sim = Simulation()
    h = sim.hacking
    datacore = sim.terminals[1]
    _open_all_doors(sim)
    sim.guards[1].x, sim.guards[1].z = 9.0, 41.0
    assert _jack_in(sim, datacore, 0.0, 28.5)
    labels = [o.spec.label for o in h.session_options()]
    assert labels[-1] == "STEAL DATA" and "STEAL DATA" in labels
    sim.update(DT, InputState(hack_slot=len(labels)))
    run(sim, 2.1)
    assert sim.player.intel == ["greenhouse door cipher"]
    assert h.session_options()[-1].reason == "DATA ALREADY COPIED"
    # leaving and coming back: no link bar (progress stays at zero, the link is instant)
    sim.update(DT, InputState(disconnect=True))
    assert datacore.progress == 0.0 and datacore.is_hacked
    sim.update(DT, InputState(interact=True))
    assert h.session is datacore and datacore.progress == 100.0


def test_disabling_a_camera_directly_takes_down_only_that_camera():
    sim = Simulation()
    cams = sim.cameras
    hack(sim, cams[1], HackKind.SHUTDOWN)
    assert not cams[1].is_online and all(c.is_online for c in cams if c is not cams[1])


def test_the_interface_closes_when_the_mission_resets():
    sim = Simulation()
    terminal = sim.terminals[0]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)
    assert _jack_in(sim, terminal, 3.5, -2.5)
    sim.update(DT, InputState(reset=True))
    assert sim.hacking.session is None and not terminal.is_connected and not terminal.is_hacked


def test_standing_at_a_desk_keeps_the_player_put_but_not_blind():
    sim = Simulation()
    terminal = sim.terminals[0]
    hack(sim, sim.guards[0], HackKind.ELECTROCUTE)
    assert _jack_in(sim, terminal, 3.5, -2.5)
    x, z, yaw = sim.player.x, sim.player.z, sim.player.yaw
    run(sim, 1.0, InputState(forward=1.0, strafe=-1.0, look_dx=30.0))
    assert (sim.player.x, sim.player.z) == (x, z), "the hands are on the deck"
    assert sim.player.yaw != yaw, "but you can still look around the room"
