"""Scanner: a pulse reveals nearby electronics (through walls) for a while."""

from dataclasses import replace

import pytest

from cyberduke.config import HACKING
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems.scanner import Scanner

DT = 1 / 60


class Dev:
    """Minimal pingable target."""

    def __init__(self, hack_id, x, y, z):
        self.hack_id = hack_id
        self.hack_position = (x, y, z)


def test_pulse_reveals_targets_inside_the_range_and_nothing_beyond():
    s = Scanner(replace(HACKING, scan_range=10.0))
    near, edge, far = (
        Dev("near", 3.0, 1.0, 0.0),
        Dev("edge", 10.0, 0.0, 0.0),
        Dev("far", 10.1, 0, 0),
    )
    hits = s.pulse_from((0.0, 0.0, 0.0), [near, edge, far])
    assert [d.hack_id for d in hits] == ["near", "edge"]
    assert s.is_revealed("near") and s.is_revealed("edge") and not s.is_revealed("far")
    assert s.revealed_ids == ["near", "edge"]


def test_reveal_lasts_the_configured_duration_then_expires():
    s = Scanner(replace(HACKING, reveal_duration=2.0))
    s.pulse_from((0, 0, 0), [Dev("d", 1, 0, 0)])
    assert s.time_left("d") == pytest.approx(2.0)
    s.update(1.9)
    assert s.is_revealed("d")
    s.update(0.2)
    assert not s.is_revealed("d") and s.time_left("d") == 0.0


def test_scanner_recharges_between_pulses():
    s = Scanner(replace(HACKING, scan_cooldown=1.0))
    assert s.pulse_from((0, 0, 0), [Dev("d", 1, 0, 0)])
    assert not s.ready
    assert s.pulse_from((0, 0, 0), [Dev("e", 1, 0, 0)]) == [], "ignored while recharging"
    assert not s.is_revealed("e")
    s.update(1.0)
    assert s.ready and s.pulse_from((0, 0, 0), [Dev("e", 1, 0, 0)])
    assert s.pulses_fired == 2


def test_pulse_visual_expands_to_the_scan_range_then_finishes():
    s = Scanner(replace(HACKING, scan_range=16.0, pulse_visual_duration=1.0))
    s.pulse_from((1.0, 0.0, 2.0), [])
    pulse = s.pulse
    assert pulse is not None and pulse.origin == (1.0, 0.0, 2.0) and pulse.radius == 0.0
    s.update(0.5)
    assert s.pulse.radius == pytest.approx(8.0)
    s.update(0.6)
    assert s.pulse is None, "the ring is gone once it reaches the scan range"


def test_manual_reveal_and_reset():
    s = Scanner()
    s.reveal("x", 3.0)
    assert s.is_revealed("x") and s.time_left("x") == 3.0
    s.reset()
    assert not s.is_revealed("x") and s.pulse is None and s.ready


# ----------------------------------------------------------------------------- in the game
def test_tab_scans_from_the_spawn_and_reveals_the_staging_bay_electronics_through_walls():
    """From the spawn the scan reaches the staging terminal, the guard, the blast door
    and the airlock camera and bulkhead behind it (a ping does not need line of sight),
    but not the vault 20+ m away."""
    sim = Simulation()
    sim.update(DT, InputState(scan=True))
    revealed = set(sim.scanner.revealed_ids)
    assert {"staging_terminal", "staging_guard", "blast_door", "airlock_camera"} <= revealed
    assert "vault_bulkhead" in revealed, "12 m away, behind the shut blast door: still pinged"
    assert not {"datacore_terminal", "vault_guard", "neon_cold_vendor"} & revealed
    assert sim.scanner.pulse is not None and sim.scanner.pulse.origin == sim.viewpoint.eye


def test_scanning_from_a_hijacked_camera_pings_around_the_camera_not_the_player():
    from cyberduke.systems.hacking import HackKind

    sim = Simulation()
    sim.player.x, sim.player.z = 0.0, 6.0
    sim.hacking.apply(sim.cameras[0], HackKind.HIJACK)
    sim.update(DT, InputState(scan=True))
    assert sim.scanner.pulse.origin == sim.cameras[0].lens
    assert "neon_cold_vendor" in sim.scanner.revealed_ids, "14.8 m from the lens"
