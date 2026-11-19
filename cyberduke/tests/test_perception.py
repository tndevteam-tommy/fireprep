"""NPC vision: range, FOV (including head yaw), and 3D occlusion; camera perception under hacks."""

import math

import pytest

from cyberduke.ai.npc import BaseNPC, HumanoidSpec
from cyberduke.animation import HumanoidPose
from cyberduke.core.geometry import AABB


def make_npc(x=0.0, z=0.0, yaw=0.0):
    npc = BaseNPC(x, 0.0, z, spec=HumanoidSpec(), nav_bounds=(-10, 10, -10, 10))
    npc.yaw = yaw
    return npc


def test_out_of_range():
    assert not make_npc().can_see_target(0.0, 6.0, max_range=5.0, fov_angle=60)


def test_inside_cone_and_outside_cone():
    npc = make_npc()  # facing +Z
    assert npc.can_see_target(0.5, 3.0, max_range=5.0, fov_angle=30)
    assert not npc.can_see_target(3.0, 0.5, max_range=5.0, fov_angle=30)


def test_head_yaw_shifts_the_gaze():
    npc = make_npc()
    npc.pose = HumanoidPose(head_yaw=80.0)  # looking toward +X
    assert npc.can_see_target(3.0, 0.5, max_range=5.0, fov_angle=30)
    assert not npc.can_see_target(0.0, 3.0, max_range=5.0, fov_angle=30)


def test_wall_blocks_sight_but_railing_does_not():
    npc = make_npc()
    wall = AABB(-1, 1, 1.0, 1.2, 0.0, 3.0)
    rail = AABB(-1, 1, 1.0, 1.2, 0.0, 3.0, blocks_sight=False)
    assert not npc.can_see_target(0.0, 3.0, obstacles=[wall], target_y=1.5)
    assert npc.can_see_target(0.0, 3.0, obstacles=[rail], target_y=1.5)


def test_regression_b2_crouching_behind_a_crate_hides_the_player():
    """The original code aimed the LOS ray at eye height + 1.2 m (above the head), so a
    crouched player behind a 1.65 m crate was visible. The ray must target the body."""
    guard = make_npc(-1.0, 28.5, yaw=-90.0)  # facing -X, as in the vault
    crate = AABB(-5.2, -3.8, 27.5, 29.5, 0.0, 1.65)
    crouched_chest = 0.0 + 0.9
    standing_chest = 0.0 + 1.5
    assert not guard.can_see_target(
        -6.2, 28.5, max_range=6.0, obstacles=[crate], target_y=crouched_chest
    )
    assert not guard.can_see_target(
        -6.2, 28.5, max_range=6.0, obstacles=[crate], target_y=standing_chest
    )
    # and the same spot is visible once the crate is gone
    assert guard.can_see_target(-6.2, 28.5, max_range=6.0, obstacles=[], target_y=crouched_chest)


def test_player_on_catwalk_is_visible_over_a_low_box():
    guard = make_npc(0.0, 0.0)
    low_box = AABB(-1, 1, 1.0, 1.5, 0.0, 1.0)
    assert guard.can_see_target(0.0, 3.0, obstacles=[low_box], target_y=2.75 + 1.5)


# =============================================================================== camera hacks
def test_camera_shut_down_cannot_see_and_does_not_sweep():
    from cyberduke.entities import Player, SecurityCamera

    cam = SecurityCamera(0.0, 3.6, 0.0, camera_id="c")
    p = Player()
    p.x, p.z = 0.0, 3.0
    assert cam.can_see(p, [])
    assert cam.shut_down(2.0)
    assert not cam.is_online and not cam.can_see(p, [])
    assert not cam.shut_down(2.0), "already down"
    yaw = cam.current_yaw
    for _ in range(60):
        assert cam.update(p, 1 / 60, []) is False
    assert cam.current_yaw == yaw
    for _ in range(70):
        cam.update(p, 1 / 60, [])
    assert cam.is_online


def test_hijacked_camera_is_steerable_within_its_sweep_and_blind_to_the_player():
    from cyberduke.entities import Player, SecurityCamera

    cam = SecurityCamera(0.0, 3.6, 0.0, mount_yaw=10.0, sweep_angle=30.0, camera_id="c")
    p = Player()
    p.x, p.z = 0.0, 3.0
    assert cam.hijack()
    assert not cam.hijack(), "already connected"
    assert cam.hijack_yaw == 10.0 and cam.current_yaw == 10.0
    cam.steer(100.0)
    assert cam.hijack_yaw == 40.0
    cam.steer(-500.0)
    assert cam.hijack_yaw == -20.0
    assert cam.update(p, 1 / 60, []) is False and cam.current_yaw == -20.0
    assert not cam.can_see(p, [])
    cam.release()
    cam.update(p, 1 / 60, [])
    assert cam.can_see(p, []) or abs(cam.current_yaw - 10.0) > 22.5


def test_camera_viewpoint_is_the_lens_looking_along_its_yaw_tilted_down():
    from cyberduke.entities import SecurityCamera

    cam = SecurityCamera(1.0, 3.6, 2.0, mount_yaw=90.0, camera_id="c")  # NPC yaw 90 = +X
    v = cam.viewpoint
    assert v.eye == (1.0, 3.5, 2.0) and v.source == "c"
    assert v.fx == pytest.approx(math.cos(math.radians(22.0)))
    assert v.fy == pytest.approx(-math.sin(math.radians(22.0)))
    assert v.fz == pytest.approx(0.0, abs=1e-9)


def test_hijacked_camera_tilts_freely_within_limits_and_the_view_follows():
    from cyberduke.entities import Player, SecurityCamera

    cam = SecurityCamera(0.0, 3.6, 0.0, camera_id="c")
    cam.hijack()
    assert cam.hijack_tilt == cam.TILT_DEG
    cam.steer(0.0, -100.0)  # look up
    assert cam.hijack_tilt == cam.MIN_TILT
    cam.update(Player(), 1 / 60, [])
    assert cam.current_tilt == cam.MIN_TILT and cam.viewpoint.fy > 0, "looking up"
    cam.steer(0.0, 500.0)  # straight down
    cam.update(Player(), 1 / 60, [])
    assert cam.current_tilt == cam.MAX_TILT and cam.viewpoint.fy < -0.9
    cam.release()
    cam.update(Player(), 1 / 60, [])
    assert cam.current_tilt == cam.TILT_DEG, "back to its working tilt"


def test_dead_camera_droops_and_lifts_its_head_again_when_power_returns():
    from cyberduke.entities import Player, SecurityCamera

    cam = SecurityCamera(0.0, 3.6, 0.0, camera_id="c")
    assert cam.head_droop == 0.0
    cam.shut_down(2.0)
    droops = []
    for _ in range(60):
        cam.update(Player(), 1 / 60, [])
        droops.append(cam.head_droop)
    assert droops == sorted(droops) and droops[-1] > 0.9, "hangs down within a second"
    for _ in range(120):
        cam.update(Player(), 1 / 60, [])
    assert cam.is_online
    for _ in range(60):
        cam.update(Player(), 1 / 60, [])
    assert cam.head_droop < 0.1, "rebooted: head back up"
