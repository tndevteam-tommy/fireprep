"""End-to-end simulation behaviour driven with synthetic input (no window, no GL)."""

import contextlib
from dataclasses import replace

import pytest

from cyberduke.ai.guard import Inspecting
from cyberduke.config import SIMULATION
from cyberduke.core.state import GameState
from cyberduke.simulation import InputState, Simulation

DT = 1 / 60


def run(sim, seconds, inp=None):
    inp = inp or InputState()
    for _ in range(int(seconds / DT)):
        sim.update(DT, inp)


def test_idle_simulation_stays_in_playing_state():
    sim = Simulation()
    run(sim, 5.0)
    assert sim.state is GameState.PLAYING


def test_player_is_not_in_view_for_the_opening_seconds():
    """Guard 1 starts walking away from the spawn; standing still for a few seconds is safe."""
    sim = Simulation()
    run(sim, 4.0)
    assert sim.detection.level == 0.0
    assert not sim.guards[0].is_reacting


def test_regression_b3_state_transitions_do_not_require_the_player_to_move():
    """The original win check lived inside `if move_len > 0.001`. The datacore jack-in
    now yields the greenhouse cipher (the way out is the street exit); the check that
    matters is that standing still and holding [E] completes it and the state machine
    notices the consequences without any movement."""
    sim = Simulation()
    sim.guards[1].x, sim.guards[1].z = 9.0, 41.0  # park the vault guard far away, out of sight
    sim.player.x, sim.player.z = 0.0, 28.5  # standing still in front of the datacore terminal
    run(sim, 4.0, InputState(interact=True))  # hold [E], no movement keys
    assert sim.objective_terminal.is_hacked
    steal = next(
        i for i, o in enumerate(sim.hacking.session_options(), 1) if o.spec.label == "STEAL DATA"
    )
    sim.update(DT, InputState(hack_slot=steal))
    run(sim, 2.2)
    assert "greenhouse door cipher" in sim.player.intel
    assert sim.state is GameState.PLAYING
    # ...and the exit zone hands over to the street without moving either: teleport
    # onto it, stand still, and once the picture has faded out the next level is loaded
    sim.player.x, sim.player.z = 0.0, 78.6
    sim.update(DT, InputState())
    assert sim.leaving == "night_market", "at the door: fading out"
    run(sim, SIMULATION.level_fade_out + 2 * DT)
    assert sim.level.name == "Night Market Alley" and sim.state is GameState.PLAYING
    assert "greenhouse door cipher" in sim.player.intel, "the deck comes along"


def test_jacking_into_the_first_terminal_and_choosing_unlock_doors_opens_the_blast_door():
    sim = Simulation()
    sim.player.x, sim.player.z = 3.5, -2.5
    run(sim, 2.0, InputState(interact=True))  # hold [E]: the link comes up
    terminal = sim.terminals[0]
    assert terminal.is_hacked and sim.hacking.session is terminal
    door = sim.room.doors["blast_door"]
    assert not door.unlocked, "being inside the computer does not open anything by itself"
    labels = [o.spec.label for o in sim.hacking.session_options()]
    assert labels == [
        "UNLOCK DOORS",
        "HIJACK CAMERAS",
        "DISABLE CAMERAS",
        "SHORT CIRCUIT (DISTRACT & DESTROY)",
    ]
    sim.update(DT, InputState(hack_slot=1))  # UNLOCK DOORS
    run(sim, 1.2)
    assert door.unlocked
    run(sim, 2.0)
    assert door.passable
    assert door.obstacle not in sim.obstacles
    # [Q] leaves the desk again
    sim.update(DT, InputState(disconnect=True))
    assert sim.hacking.session is None and not terminal.is_connected


def _stare_at_player(sim):
    """Parks guard 0 four metres from the spawn in the INSPECTING state (stationary, head up,
    slow scan) facing the player, with an effectively infinite inspection time."""
    g = sim.guards[0]
    g.x, g.z, g.yaw = 0.0, 2.0, 0.0
    g.cfg = replace(g.cfg, investigate_duration=1e9)
    g.fsm.transition_to(Inspecting(source=(0.0, 6.0)))  # look toward the spawn point


def test_caught_then_automatic_respawn():
    sim = Simulation()
    _stare_at_player(sim)
    sim.player.x = 1.0
    run(sim, 3.0)  # 45/s -> 100 in ~2.2 s; respawn delay has not elapsed yet
    assert sim.state is GameState.CAUGHT
    # respawn happens after the delay; the guard resets too so the player is no longer seen
    run(sim, SIMULATION.caught_respawn_delay + 0.1)
    assert sim.state is GameState.PLAYING
    assert sim.player.x == 0.0 and sim.detection.level == 0.0


def test_regression_b4_large_frame_deltas_are_clamped():
    sim = Simulation()
    z0 = sim.player.z
    sim.update(5.0, InputState(forward=1.0))  # a 5-second hitch
    assert abs(sim.player.z - z0) <= sim.player.move_speed * SIMULATION.max_dt + 1e-6
    assert sim.player.y > 0.5, "eye height must not overshoot below the floor"


def test_detection_decays_at_configured_rate_with_two_guards_present():
    """End-to-end version of the B1 regression: 2 guards + camera, none seeing the player."""
    sim = Simulation()
    sim.detection.level = 60.0
    sim.update(DT, InputState())
    assert sim.detection.level == pytest.approx(60.0 - sim.detection.decay_rate * DT)


def test_guard_in_line_of_sight_raises_detection():
    sim = Simulation()
    _stare_at_player(sim)
    sim.update(DT, InputState())
    assert sim.detection.level > 0.0


def test_vending_machine_distracts_the_vault_guard():
    sim = Simulation()
    sim.player.x, sim.player.z = -6.0, 25.0
    sim.update(DT, InputState(interact=True))
    assert sim.vending_machines[0].is_sparking
    assert sim.guards[1].state == "INVESTIGATING"
    assert sim.guards[0].state != "INVESTIGATING", "Sector 1 guard is out of acoustic range"


# =============================================================================== netrunner run
def test_full_playthrough_from_the_spawn_to_the_street():
    """The whole mission, driven purely with InputState (real movement, real doors, real
    guards), mixing quickhacks and jack-ins:

    1. staging bay: scan, OVERRIDE DOORS on the staging terminal (network breached),
       ELECTROCUTE the guard, SHUT DOWN the airlock camera, FORCE OPEN the vault bulkhead
    2. vault: BLIND the elite, scan the secured datacore terminal, STEAL DATA (the cipher)
    3. the greenhouse door reads the cipher off the deck as the player walks up to it
    4. greenhouse: through the spore lock, east through the maze to the gated control
       node; jack into its air-gapped terminal (biotech network breached) and pick UNLOCK
       DOORS / STEAL DATA from its interface
    5. north past the gate to the top corridor, west to the other control node; jack in and
       unlock the street exit; back east and out through it: onto the street (the next
       level), then up the alley, out onto the main street's pavement and along it to the
       barricade at its end: there is no winning, the game goes on.
    """
    import math

    from cyberduke.levels.facility import gh_cell_center
    from cyberduke.systems.hacking import HACKS, Access, HackKind

    sim = Simulation()
    h = sim.hacking
    p = sim.player
    states = set()

    def step(inp=None):
        sim.update(DT, inp or InputState())
        states.add(sim.state)

    def wait(seconds, inp=None):
        for _ in range(int(round(seconds / DT))):
            step(inp)

    def aim(target):
        tx, ty, tz = target.hack_position
        p.yaw = math.degrees(math.atan2(tz - p.z, tx - p.x))
        p.pitch = math.degrees(math.atan2(ty - p.y, math.hypot(tx - p.x, tz - p.z)))
        step()
        assert h.focus is target, (target.hack_id, h.focus and h.focus.hack_id)

    def quickhack(target, kind, expect_access):
        aim(target)
        assert h.access(target) in expect_access
        slot = next(i for i, o in enumerate(h.options(target), 1) if o.kind is kind)
        step(InputState(hack_slot=slot))
        assert h.upload is not None
        wait(HACKS[kind].upload_time + 0.05)
        assert h.upload is None

    def walk_to(x, z, timeout=12.0, keep_blind=None):
        """Walks (W + yaw) toward a point, sliding along whatever is in the way. While
        ``keep_blind`` names a guard, it is re-blinded over the network whenever its optics
        clear and it is anywhere near. Stops when the run is won or another level has
        been loaded on the way."""
        gen = sim.level_generation
        for _ in range(int(timeout / DT)):
            if math.hypot(x - p.x, z - p.z) < 0.35:
                return
            if sim.level_generation != gen:
                return
            g = keep_blind
            if g is not None and not g.is_blinded and math.hypot(g.x - p.x, g.z - p.z) < 14.0:
                assert h.apply(g, HackKind.BLIND, Access.NETWORK)
            p.yaw = math.degrees(math.atan2(z - p.z, x - p.x))
            p.pitch = 0.0
            step(InputState(forward=1.0))
        raise AssertionError(f"did not reach ({x:.1f}, {z:.1f}); at ({p.x:.2f}, {p.z:.2f})")

    def walk_cells(cells, **kw):
        for c in cells:
            walk_to(*gh_cell_center(*c), **kw)

    def jack_in(terminal, x, z, *choices):
        """Walks to a desk, holds [E] until the interface opens, then picks options by name."""
        walk_to(x, z)
        for _ in range(int(4.0 / DT)):
            step(InputState(interact=True))
            if h.session is terminal:
                break
        assert h.session is terminal, f"no link to {terminal.hack_id}"
        for label in choices:
            options = h.session_options()
            slot = next(i for i, o in enumerate(options, 1) if o.spec.label == label)
            assert options[slot - 1].available, (label, options[slot - 1].reason)
            step(InputState(hack_slot=slot))
            wait(options[slot - 1].spec.upload_time + 0.1)
        if h.session is terminal:
            step(InputState(disconnect=True))

    staging_t, datacore, maintenance_t, east_t, west_t = sim.terminals
    staging_guard, vault_guard, gh_guard, gh_elite = sim.guards
    airlock_cam = sim.cameras[0]
    doors = sim.room.doors
    either = {Access.DIRECT, Access.NETWORK}

    # 1. Staging bay (a step off the door axis, so the blast door and the bulkhead behind
    #    it are separate things to aim at)
    p.x = -0.8
    step(InputState(scan=True))
    quickhack(staging_t, HackKind.OVERRIDE_DOORS, {Access.DIRECT})
    assert doors["blast_door"].unlocked and h.is_breached("halcyon")
    quickhack(staging_guard, HackKind.ELECTROCUTE, {Access.DIRECT})
    quickhack(airlock_cam, HackKind.SHUTDOWN, either)
    quickhack(doors["vault_bulkhead"], HackKind.FORCE_OPEN, either)
    wait(2.0)
    walk_to(0.0, 19.0, timeout=8.0)
    assert sim.detection.level == 0.0

    # 2. Vault: the secured datacore terminal is not on the network - scan it, hack it in sight
    quickhack(vault_guard, HackKind.BLIND, {Access.NETWORK})
    step(InputState(scan=True))
    assert h.access(datacore) is Access.DIRECT and not h.on_network(datacore)
    quickhack(datacore, HackKind.STEAL_DATA, {Access.DIRECT})
    assert "greenhouse door cipher" in p.intel and sim.state is GameState.PLAYING

    # 2b. The cipher is a menu choice at the desk or a remote STEAL DATA - here it was the
    #     quickhack, so nothing else on the datacore was touched.
    assert not datacore.is_connected and not datacore.doors_overridden

    # 3. The greenhouse door reads the cipher as the player approaches it (under the skybridge)
    greenhouse_door = doors["greenhouse_door"]
    assert not greenhouse_door.unlocked
    walk_to(-2.2, 36.0, keep_blind=vault_guard)  # round the monolith on the left
    walk_to(-1.0, 40.6, keep_blind=vault_guard)
    assert greenhouse_door.unlocked, "cipher accepted"
    wait(1.0, InputState())
    assert greenhouse_door.passable

    # 4. Spore lock and the maze, east to the control node. The biotech network is not
    #    breached yet, so every camera is dealt with in sight: scan, SHUT DOWN, move. (The
    #    spine guard is put down directly once a scan shows it; the elite gets blinded
    #    whenever it comes close.)
    cams = {c.hack_id: c for c in sim.cameras}
    walk_to(0.0, 44.0)
    step(InputState(scan=True))
    assert "greenhouse_entry_camera" in sim.scanner.revealed_ids
    assert not h.is_breached("halcyon_bio")
    quickhack(cams["greenhouse_entry_camera"], HackKind.SHUTDOWN, {Access.DIRECT})
    walk_to(0.0, 48.0)
    assert h.apply(gh_guard, HackKind.ELECTROCUTE, Access.DIRECT)
    walk_cells([(0, -4), (0, -3), (1, -3), (2, -3), (2, -2)])
    step(InputState(scan=True))
    walk_cells([(3, -2)])
    quickhack(cams["east_node_camera"], HackKind.SHUTDOWN, {Access.DIRECT})
    assert not doors["maze_gate"].unlocked
    assert not h.apply(east_t, HackKind.OVERRIDE_DOORS), "air-gapped: no quickhack"
    jack_in(east_t, 9.6, 56.1, "UNLOCK DOORS")  # at the desk: open the gate
    assert east_t.is_hacked and doors["maze_gate"].unlocked
    assert h.is_breached("halcyon_bio"), "jacking in breaches the biotech network"
    wait(1.5)
    assert doors["maze_gate"].passable
    # On the network now: the exit camera over the top corridor goes dark from here.
    quickhack(cams["exit_camera"], HackKind.SHUTDOWN, {Access.NETWORK})
    assert cams["greenhouse_spine_camera"].is_online, "a direct shutdown is one camera only"

    # 5. Through the gate, up the east side, west along the top corridor to the west node
    walk_to(11.2, 58.6, keep_blind=gh_elite)  # past the desk, in front of the gate
    walk_cells([(4, 0), (4, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 4), (0, 4)],
               keep_blind=gh_elite)  # fmt: skip
    walk_cells([(-1, 4), (-2, 4), (-2, 3), (-2, 2), (-3, 2), (-3, 1)], keep_blind=gh_elite)
    assert not doors["street_exit"].unlocked
    jack_in(west_t, -9.6, 62.4, "UNLOCK DOORS", "DISABLE CAMERAS")
    assert west_t.is_hacked and doors["street_exit"].unlocked
    walk_cells([(-3, 2), (-2, 2), (-2, 3), (-2, 4), (-1, 4), (0, 4)], keep_blind=gh_elite)
    assert all(not c.is_online for c in sim.cameras[1:]), "DISABLE CAMERAS took the bio net down"
    walk_to(0.0, 74.8, keep_blind=gh_elite)
    walk_to(0.0, 78.6, keep_blind=gh_elite)
    # at the outer staff door the picture fades to black, and the alley loads behind it
    assert sim.leaving == "night_market" or sim.level_generation == 1
    for _ in range(int(2 * SIMULATION.level_fade_out / DT)):
        if sim.level_generation == 1:
            break
        step()
    assert sim.screen_black > 0.9, "arriving in the black, fading in"

    # 6. Out of the door: the facility is behind us, the night market in front
    assert sim.level.name == "Night Market Alley" and sim.state is GameState.PLAYING
    assert sim.level_generation == 1 and not sim.guards and sim.civilians
    assert (p.x, p.z) == sim.level.spawn[:2], "at the alley's facility end"
    assert p.intel == ["greenhouse door cipher"], "the only data that matters in the facility"
    # up the alley, round the stalls and the locals, out of its mouth onto the pavement
    walk_to(0.0, 38.0, timeout=60.0)
    walk_to(0.0, 43.7, timeout=30.0)
    assert p.z > 42.5, "out on the main street"
    # and along it to the barricade at the west end (it is a barricade: the walk cannot
    # get all the way)
    with contextlib.suppress(AssertionError):
        walk_to(-36.0, 43.7, timeout=60.0)
    assert -34.0 < p.x < -31.0, "walked the length of the street, stopped by the barricade"
    assert sim.state is GameState.PLAYING, "no winning: the game goes on"
    assert states == {GameState.PLAYING}, "never caught"


# =============================================================================== pause
def _world_snapshot(sim):
    """Everything that moves, ticks, animates or decays in one tuple."""
    g = sim.guards[0]
    cam = sim.cameras[0]
    vm = sim.vending_machines[0]
    terminal = sim.terminals[0]
    door = sim.room.doors["blast_door"]
    return (
        sim.time,
        (sim.player.x, sim.player.z, sim.player.y, sim.player.yaw),
        (g.x, g.z, g.yaw, g.state, g.state_timer, g.walk_cycle, tuple(g.pose.arm_l)),
        (cam.current_yaw, cam.anim_time, cam.shutdown_timer, cam.head_droop),
        (vm.anim_time, vm.cooldown, vm.spark_timer),
        (terminal.anim_time, terminal.progress, terminal.spark_timer),
        (door.open_amount, sim.room.anim_time),
        (sim.detection.level, sim.state, sim.state_timer),
        (sim.scanner.time_left("staging_terminal"), sim.scanner.cooldown),
        sim.hacking.upload.elapsed if sim.hacking.upload else None,
    )


def test_pause_freezes_every_moving_part_and_resumes_exactly_where_it_stopped():
    from cyberduke.systems.hacking import HackKind

    sim = Simulation()
    # get as much as possible in motion: a guard walking, a camera sweeping, a door opening,
    # a machine on cooldown, a scan running and a quickhack uploading
    sim.player.x, sim.player.z, sim.player.yaw = 0.0, 6.0, -90.0
    run(sim, 1.0)
    sim.hacking.apply(sim.terminals[0], HackKind.OVERRIDE_DOORS)
    sim.hacking.apply(sim.vending_machines[0], HackKind.DISTRACT)
    sim.update(DT, InputState(scan=True))
    run(sim, 0.5)
    sim.hacking.request(1)  # start an upload on whatever is focused, if anything
    run(sim, 0.3)
    assert 0.0 < sim.room.doors["blast_door"].open_amount < sim.room.doors["blast_door"].travel

    sim.update(DT, InputState(toggle_pause=True))
    assert sim.paused
    before = _world_snapshot(sim)
    for _ in range(120):  # two seconds of frames, including input the player might mash
        sim.update(DT, InputState(forward=1.0, strafe=1.0, look_dx=50.0, interact=True, scan=True))
    assert _world_snapshot(sim) == before, "a paused game must not move at all"

    sim.update(DT, InputState(toggle_pause=True))
    assert not sim.paused
    run(sim, 0.5)
    after = _world_snapshot(sim)
    assert after != before, "and it carries straight on afterwards"
    assert sim.room.doors["blast_door"].open_amount > before[6][0]
    assert sim.guards[0].state_timer > before[2][4]


def test_pause_works_in_every_game_state_and_restart_clears_it():
    sim = Simulation()
    for state in (GameState.PLAYING, GameState.CAUGHT):
        sim.state = state
        sim.paused = False
        sim.update(DT, InputState(toggle_pause=True))
        assert sim.paused
        snapshot = (sim.state, sim.state_timer, sim.time)
        run(sim, 3.0)
        assert (sim.state, sim.state_timer, sim.time) == snapshot
        sim.update(DT, InputState(toggle_pause=True))
        assert not sim.paused
    sim.update(DT, InputState(toggle_pause=True))
    sim.update(DT, InputState(reset=True))  # the menu's RESTART
    assert not sim.paused and sim.state is GameState.PLAYING


def test_pause_does_not_swallow_a_queued_event_or_an_open_interface():
    from cyberduke.systems.hacking import HackKind

    sim = Simulation()
    sim.player.x, sim.player.z = 3.5, -2.5
    run(sim, 2.0, InputState(interact=True))
    assert sim.hacking.session is sim.terminals[0]
    sim.update(DT, InputState(hack_slot=1))  # start UNLOCK DOORS
    sim.update(DT, InputState(toggle_pause=True))
    progress = sim.hacking.upload.progress
    run(sim, 2.0, InputState(hack_slot=3))
    assert sim.hacking.upload is not None and sim.hacking.upload.progress == progress
    assert sim.hacking.session is sim.terminals[0], "still at the desk"
    sim.update(DT, InputState(toggle_pause=True))
    run(sim, 1.2)
    assert sim.room.doors["blast_door"].unlocked, "the upload finished after the pause"
    assert sim.hacking.apply(sim.cameras[0], HackKind.SHUTDOWN)


@pytest.mark.parametrize("spot", [(7.8, 22.6), (-7.8, 23.5)])
def test_guards_never_walk_into_the_staircases(spot):
    """Regression: the simulation handed NPCs the plain colliders, not the level's NPC no-go
    boxes, so a guard creeping after a sighting on the stairs (or sent to a noise there)
    walked straight into the solid steps at floor level. It now stops at the foot of the
    stairs (or under the deck behind them), never inside, and still looks around."""
    from cyberduke.ai.guard import Suspicious

    sim = Simulation()
    for door in sim.room.doors.values():
        door.unlock()
        door.open_amount = door.travel
    sim.player.x, sim.player.z = 0.0, -5.0  # out of sight in the staging bay
    sim.guards[0].blind(1e9)
    guard = next(g for g in sim.guards if g.guard_id == "vault_guard")
    guard.cfg = replace(guard.cfg, suspicion_grace=1e9)
    guard.last_seen, guard.time_since_seen = spot, 0.0
    guard.fsm.transition_to(Suspicious(guard.routine_to_resume()))
    stairs = sim.room.level.sector("DatacoreVault").nav_blockers
    looked_around = False
    for _ in range(int(25.0 / DT)):
        sim.update(DT, InputState())
        r = guard.collision_radius
        for b in stairs:
            assert not (
                b.min_x - r < guard.x < b.max_x + r and b.min_z - r < guard.z < b.max_z + r
            ), f"the guard walked into the staircase at ({guard.x:.2f}, {guard.z:.2f})"
        looked_around |= guard.state == "LOOKING_AROUND"
    assert looked_around, "it reached the nearest point it can stand on and searched there"


def test_disconnecting_with_the_interact_key_still_held_does_not_jack_straight_back_in():
    """Regression: a breached terminal reconnects instantly, so pressing [Q] while still
    holding [E] re-opened the interface on the very next frame - [Q] did nothing."""
    sim = Simulation()
    sim.player.x, sim.player.z = 3.5, -2.5
    for _ in range(int(3.0 / DT)):
        sim.update(DT, InputState(interact=True))
        if sim.hacking.session is not None:
            break
    terminal = sim.hacking.session
    assert terminal is sim.terminals[0]
    sim.update(DT, InputState(interact=True, disconnect=True))
    assert sim.hacking.session is None
    run(sim, 0.5, InputState(interact=True))  # the key is still down
    assert sim.hacking.session is None, "no instant reconnect while [E] is still held"
    sim.update(DT, InputState())  # released...
    sim.update(DT, InputState(interact=True))  # ...and pressed again: straight back in
    assert sim.hacking.session is terminal
