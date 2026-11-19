"""Eye-space transforms used to feed the lighting shader."""
import pytest

from tests.helpers import spot
from unprepared.rendering.camera import look_at_matrix
from unprepared.rendering.lighting import Camera


def test_camera_basis_and_eye_space():
    eye = (0.0, 2.0, 0.0)
    cam = Camera(eye, look_at_matrix(eye, (0.0, 2.0, -1.0), (0.0, 1.0, 0.0)))    # looking down -z
    assert cam.side == pytest.approx((1.0, 0.0, 0.0))
    assert cam.up == pytest.approx((0.0, 1.0, 0.0))
    assert cam.to_eye_point((0.0, 2.0, -5.0)) == pytest.approx((0.0, 0.0, -5.0))    # straight ahead
    assert cam.to_eye_point((3.0, 2.0, 0.0)) == pytest.approx((3.0, 0.0, 0.0))      # to the right
    assert cam.to_eye_dir((0.0, 1.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))


def test_rotated_camera():
    eye = (1.0, 1.0, 1.0)
    cam = Camera(eye, look_at_matrix(eye, (2.0, 1.0, 1.0), (0.0, 1.0, 0.0)))      # looking +x
    assert cam.to_eye_point((5.0, 1.0, 1.0)) == pytest.approx((0.0, 0.0, -4.0))


def test_inverse_view_matrix_maps_eye_space_back_to_world():
    import math
    eye = (3.0, 1.5, -2.0)
    cam = Camera(eye, look_at_matrix(eye, (3.0 + math.cos(0.7), 1.5, -2.0 - math.sin(0.7)), (0.0, 1.0, 0.0)))
    m = cam.inverse_view_matrix()                     # column-major 4x4
    assert len(m) == 16 and m[12:16] == [3.0, 1.5, -2.0, 1.0]
    for p in ((0.0, 0.0, 0.0), (4.0, 2.0, -9.0), (-6.0, 0.5, 3.0)):
        e = cam.to_eye_point(p)
        back = tuple(m[0 + i] * e[0] + m[4 + i] * e[1] + m[8 + i] * e[2] + m[12 + i] for i in range(3))
        assert back == pytest.approx(p, abs=1e-9)


class TestRoomBoundedLights:
    """Point lights are confined to the building they burn in (no light through walls)."""

    def test_bounds_stop_just_inside_the_outer_wall_faces(self, game):
        from unprepared.rendering.lighting import LightingRig
        w = game.world
        inside = spot(game.world, -70.0, 11.0)                                   # Barnaby's house (-74..-66, 7..15)
        x1, z1, x2, z2 = LightingRig._room_bounds(w, inside)
        assert (x1, z1, x2, z2) == pytest.approx((-73.95, 7.05, -66.05, 14.95))
        door = w.find_door("home_door")                                        # in the north wall at z = 7
        leaf = door.collider()
        assert leaf.min_point.z < z1 < leaf.max_point.z                       # the leaf straddles the wall plane:
        assert z1 > 7.0                                                        # its inner face (z 7.1) is lit, its outer
        #                                                                        face (z 6.9) and the wall's (z 7.0) are not
        assert LightingRig._room_bounds(w, spot(game.world, 0.0, 0.0)) is None    # the street: unbounded
        assert LightingRig._room_bounds(w, spot(game.world, -62.0, -8.0)) is None  # the open shed: no walls to stop it

    def test_gathered_lights_carry_their_bounds(self, game):
        from unprepared.rendering.lighting import LightingRig
        from unprepared.config import LightingSettings
        rig = LightingRig(LightingSettings())
        rig.max_lights = 64
        game.player.position = spot(game.world, -70.0, 11.0)
        game.player.has_lit_torch = True
        lights = rig.gather(game)
        torch = lights[0]
        assert torch.bounds == pytest.approx((-73.95, 7.05, -66.05, 14.95))
        indoor = [l for l in lights[1:] if l.bounds is not None]
        outdoor = [l for l in lights[1:] if l.bounds is None]
        assert indoor and outdoor                                              # lanterns/candles vs the kiln fire
        for light in indoor:
            x1, z1, x2, z2 = light.bounds
            assert x1 <= light.position[0] <= x2 and z1 <= light.position[2] <= z2
        assert any(abs(l.position[0] + 60.6) < 0.01 and abs(l.position[2] + 10.55) < 0.01 for l in outdoor)  # the forge

    def test_the_nearest_lights_get_the_slots(self, game):
        from unprepared.rendering.lighting import LightingRig
        from unprepared.config import LightingSettings
        rig = LightingRig(LightingSettings())
        game.player.position = spot(game.world, -70.0, 11.0)
        lights = rig.gather(game)
        assert len(lights) <= rig.max_lights
        eye = game.player.position
        dists = [((l.position[0] - eye.x) ** 2 + (l.position[2] - eye.z) ** 2) ** 0.5 for l in lights]
        assert dists[:3] == sorted(dists[:3]) and dists[0] < 4.0               # nearest first: this house's lantern/candle
        assert max(dists) < 40.0                                               # nothing from the far end of town

    def test_lights_shut_away_from_the_camera_rank_last(self, game):
        """In the street at night the slots go to what can be seen from the street - the watch's
        torches and the forge - before the candles behind every shutter in town, however near."""
        from unprepared.rendering.lighting import LightingRig
        from unprepared.config import LightingSettings
        w = game.world
        w.clock.hour = 22
        w.resync_schedules(announce=False, teleport=True)
        game.update(0.016)
        rig = LightingRig(LightingSettings())
        rig.max_lights = 8
        game.player.position = spot(game.world, -55.0, 0.0)                      # on Market Street by the smith's yard
        watch = w.find_npc("Guard Mira")
        watch.path, watch.current_task, watch.schedule = [], None, []
        watch.position = spot(game.world, -45.0, 2.0)                          # the night round, coming over the bridge
        game.update(0.016)
        lights = rig.gather(game)
        eye = game.player.position
        visible = [l for l in lights if l.bounds is None or (l.bounds[0] <= eye.x <= l.bounds[2] and l.bounds[1] <= eye.z <= l.bounds[3])]
        assert visible and lights[:len(visible)] == visible                    # every reachable light before any shut-away one
        assert watch.has_torch and any(abs(l.position[0] - watch.position.x) < 0.01 for l in visible)   # the night watch's torch
        assert any(abs(l.position[0] + 60.6) < 0.01 for l in visible)          # the forge fire
        assert any(f.kind == "torch" and abs(l.position[0] - f.position[0]) < 0.01 for f in w.fixtures for l in visible)   # a wall torch
        # indoors it is the other way round: the player's own room first, the street's torches after
        game.player.position = spot(game.world, -12.0, 12.0)                     # the hideout
        lights = rig.gather(game)
        assert lights[0].bounds is not None and lights[0].bounds[0] <= -12.0 <= lights[0].bounds[2]

    def test_lights_beyond_the_draw_distance_are_dropped(self, game):
        from unprepared.rendering.lighting import LightingRig
        from unprepared.config import LightingSettings
        rig = LightingRig(LightingSettings())
        rig.max_lights = 64
        game.player.position = spot(game.world, -45.0, 28.0)                      # the far south-west corner
        everything = rig.gather(game)
        near = rig.gather(game, draw_distance=20.0)
        assert len(near) < len(everything)
        eye = game.player.position
        for l in near:
            assert ((l.position[0] - eye.x) ** 2 + (l.position[2] - eye.z) ** 2) ** 0.5 - l.range <= 20.0 + 1e-6


class TestDoorwaySpill:
    def test_a_torch_carried_through_a_door_still_lights_the_open_leaf(self, game):
        """A south-side door folds out onto the street: the moment the player stepped over the
        threshold their torch was confined to the house and the leaf beside them went dark. The
        light now keeps a patch over the nearest open leaf."""
        from unprepared.rendering.lighting import LightingRig
        from unprepared.config import LightingSettings
        w = game.world
        door = w.find_door("hideout_door")                   # hinge (-19.5, 7.5), leaf folds to z 4.5..7.5 when open
        door.is_locked, door.is_open = False, True
        rig = LightingRig(LightingSettings())
        game.player.has_lit_torch = True
        game.player.position = spot(game.world, -12.0, 8.5)      # a step inside the hideout
        torch = rig.gather(game)[0]
        assert torch.bounds is not None and torch.bounds[1] > 7.0          # confined to the house ...
        assert torch.patch is not None
        leaf = door.collider()
        x1, z1, x2, z2 = torch.patch
        assert x1 <= leaf.min_point.x and x2 >= leaf.max_point.x and z1 <= leaf.min_point.z and z2 >= leaf.max_point.z   # ... but reaching the leaf
        door.is_open = False
        assert rig.gather(game)[0].patch is None                            # a shut door spills nothing
        game.player.position = spot(game.world, -12.0, 3.0)                    # back in the street: unconfined
        torch = rig.gather(game)[0]
        assert torch.bounds is None and torch.patch is None

    def test_room_lights_spill_through_their_open_door_too(self, game):
        from unprepared.rendering.lighting import LightingRig
        from unprepared.config import LightingSettings
        w = game.world
        door = w.find_door("hideout_door")
        door.is_locked, door.is_open = False, True
        rig = LightingRig(LightingSettings())
        rig.max_lights = 64
        lights = rig.gather(game)
        hideout = [l for l in lights if l.bounds is not None and l.bounds[0] <= -12.0 <= l.bounds[2] and l.bounds[1] <= 12.0 <= l.bounds[3]]
        assert hideout and all(l.patch is not None for l in hideout)
        door.is_open = False
        lights = rig.gather(game)
        hideout = [l for l in lights if l.bounds is not None and l.bounds[0] <= -12.0 <= l.bounds[2] and l.bounds[1] <= 12.0 <= l.bounds[3]]
        assert hideout and all(l.patch is None for l in hideout)
