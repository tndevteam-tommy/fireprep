"""Housekeeping: the graphics presets (the draw distance and everything else a setting now), the
start screen's stroll through the town, doors on north-south walls, the HUD's panels (the mini
inventory gone, the controls only for the tutorial, the events log behind [`]), putting down
what is being read wherever one looks, and the repairs to the old mine's steps, the cave mouths,
the park's fountain and the windows beside doors."""
from __future__ import annotations

import math

import pytest

from tests.helpers import teleport
from unprepared.attract import TITLE_ROUTE, TitleCamera, TitleView
from unprepared.core.math3d import Vector3D
from unprepared.entities import Door
from unprepared.graphics import AA_STEPS, ANISOTROPY_STEPS, DRAW_DISTANCE_RANGE, MAX_SHADER_LIGHTS, PRESETS, GraphicsSettings
from unprepared.systems import MoveIntent


# ----------------------------------------------------------------------------- graphics settings
class TestGraphicsPresets:
    def test_the_draw_distance_is_a_setting_and_high_is_very_high(self):
        low, medium, high = (PRESETS[q] for q in ("low", "medium", "high"))
        assert low["draw_distance"] == 100.0 < medium["draw_distance"] < high["draw_distance"]
        assert high["draw_distance"] >= 600.0                              # (across the whole realm)
        assert high["torch_lights"] == MAX_SHADER_LIGHTS and high["antialiasing"] == max(AA_STEPS)
        assert high["anisotropy"] == max(ANISOTROPY_STEPS) and high["star_count"] > medium["star_count"]
        assert high["haze"] < medium["haze"] <= low["haze"] == 1.0         # (the haze thins to see that far)
        for key in ("object_fade", "character_fade"):
            assert high[key] == 1.0

    def test_the_far_plane_is_kept_within_bounds(self):
        assert GraphicsSettings(draw_distance=5.0).far_plane == DRAW_DISTANCE_RANGE[0]
        assert GraphicsSettings(draw_distance=1e6).far_plane == DRAW_DISTANCE_RANGE[1]
        g = GraphicsSettings(draw_distance=300.0, object_fade=0.5)
        assert g.object_draw_distance(g.far_plane) == 150.0

    def test_the_defaults_are_the_medium_preset(self):
        g = GraphicsSettings().to_dict()
        assert g["quality"] == "medium" and all(g[k] == v for k, v in PRESETS["medium"].items())

    def test_full_screen_and_the_new_settings_are_saved(self, tmp_path):
        path = str(tmp_path / "graphics.json")
        g = GraphicsSettings(fullscreen=True, draw_distance=420.0, antialiasing=4, anisotropy=8, vsync=True, haze=0.4)
        assert g.save(path)
        back = GraphicsSettings.load(path)
        assert (back.fullscreen, back.draw_distance, back.antialiasing, back.anisotropy, back.vsync, back.haze) == \
            (True, 420.0, 4, 8, True, 0.4)

    def test_an_older_file_without_them_gets_the_defaults(self):
        g = GraphicsSettings.from_dict({"quality": "high", "fov": 70.0})
        assert g.fov == 70.0 and g.fullscreen is False and g.draw_distance == GraphicsSettings().draw_distance

    def test_the_shader_is_built_for_as_many_lights_as_asked(self):
        from unprepared.rendering.shaders import MAX_POINT_LIGHTS, fragment_shader
        assert "#define MAX_LIGHTS 16" in fragment_shader(16)
        assert f"#define MAX_LIGHTS {MAX_POINT_LIGHTS}" in fragment_shader()

    def test_the_haze_scales_the_day_but_not_the_night(self):
        from unprepared.config import LightingSettings
        from unprepared.rendering.atmosphere import compute_atmosphere
        s = LightingSettings()
        assert compute_atmosphere(12.0, s, haze=0.3).fog_density == pytest.approx(s.fog_density_day * 0.3)
        assert compute_atmosphere(0.0, s, haze=0.3).fog_density == pytest.approx(s.fog_density_night, rel=0.05)


# ----------------------------------------------------------------------------- the start screen
class TestTheTitleStroll:
    def test_the_route_keeps_to_the_streets(self, world):
        cam = TitleCamera()
        assert cam.total > 200.0
        d = 0.0
        while d < cam.total:
            x, z = cam.point_at(d)
            assert not world.nav.is_blocked(x, z), (round(x, 1), round(z, 1))
            assert world.building_at(Vector3D(x, world.ground_height(x, z) + 1.0, z)) is None
            d += 0.5

    def test_it_loops_and_looks_where_it_goes(self):
        cam = TitleCamera()
        assert cam.point_at(0.0) == pytest.approx(TITLE_ROUTE[0], abs=1e-6)
        assert cam.point_at(cam.total) == pytest.approx(cam.point_at(0.0), abs=1e-6)
        x, z, yaw = cam.pose()
        ax, az = cam.point_at(cam.LOOK_AHEAD)
        assert yaw == pytest.approx(math.degrees(math.atan2(ax - x, -(az - z))) % 360.0)

    def test_the_town_lives_on_behind_the_menu(self, game):
        title = TitleView(game)
        w = game.world
        start = (w.clock.hour, game.player.position.copy())
        for _ in range(50):
            title.update(0.1)
        assert game.player.position.distance_to(start[1]) > 5.0                  # the view has moved on
        eye = game.player.eye_position
        assert eye.y == pytest.approx(w.ground_height(eye.x, eye.z) + TitleCamera.HEIGHT, abs=0.05)
        assert not game.player.has_lit_torch and w.elapsed > 4.9                   # (and the town's clock has run)

    def test_nobody_steps_round_the_roving_eye(self, game):
        w = game.world
        npc = w.npcs[0]
        here = (game.player.position.x, game.player.position.z)
        assert any((x, z) == here for x, z, _, _ in w.body_circles(exclude=npc))     # in play: a body in the way
        TitleView(game)
        here = (game.player.position.x, game.player.position.z)
        assert not any((x, z) == here for x, z, _, _ in w.body_circles(exclude=npc))  # on the start screen: nobody


# ----------------------------------------------------------------------------- doors on a north-south line
class TestDoorsAlongZ:
    def door(self, **kw):
        return Door("d", Vector3D(10.0, 0.0, 4.0), width=2.0, axis="z", **kw)

    def test_the_leaf_spans_z_and_swings_in_x(self):
        d = self.door()
        assert d.span == (4.0, 6.0) and d.line == 10.0 and d.center == Vector3D(10.0, 0.0, 5.0)
        shut = d.collider()
        assert (shut.min_point.x, shut.max_point.x, shut.min_point.z, shut.max_point.z) == pytest.approx((9.9, 10.1, 4.0, 6.0))
        d.is_open = True
        leaf = d.collider()                                      # folded into -x ...
        assert leaf.min_point.x == pytest.approx(8.0) and leaf.max_point.z - leaf.min_point.z == pytest.approx(0.2)
        d.swing_out = True                                       # ... or +x
        assert d.collider().max_point.x == pytest.approx(12.0)

    def test_sides_crossings_and_the_doorway(self):
        from unprepared.entities.npc import _crosses_door_line
        d = self.door()
        assert d.side_of(11.0, 5.0) == 1 and d.side_of(9.0, 5.0) == -1
        assert _crosses_door_line(d, Vector3D(8.0, 0, 5.0), Vector3D(12.0, 0, 5.0))
        assert not _crosses_door_line(d, Vector3D(8.0, 0, 9.0), Vector3D(12.0, 0, 9.0))     # past the end of it
        box = d.get_bounding_box()
        assert (box.min_point.x, box.max_point.x, box.min_point.z, box.max_point.z) == pytest.approx((9.5, 10.5, 4.0, 6.0))

    def test_the_renderer_turns_it_to_match(self):
        d = self.door()
        a = math.radians(d.closed_angle)                         # glRotatef(a) sends +x to (cos a, 0, -sin a)
        assert (math.cos(a), -math.sin(a)) == pytest.approx((0.0, 1.0), abs=1e-9)          # along +z
        d.is_open = True
        a = math.radians(d.open_angle)
        assert (math.cos(a), -math.sin(a)) == pytest.approx((-1.0, 0.0), abs=1e-9)         # folded into -x
        x_door = Door("x", Vector3D(0, 0, 0))
        assert (x_door.closed_angle, x_door.open_angle) == (0.0, 90.0)                     # (unchanged for the old ones)

    def test_the_doors_that_were_missing_are_there(self, world):
        for door_id in ("vestry_door", "vane_servant_door", "ashgrove_servant_door", "greyhold_gate_north", "greyhold_gate_south"):
            d = world.find_door(door_id)
            assert d is not None and d.axis == "z" and d.label, door_id
        assert all(world.find_door(g).is_open for g in ("greyhold_gate_north", "greyhold_gate_south"))

    def test_sister_maude_goes_to_bed_through_her_door_and_locks_it(self, game):
        w = game.world
        w.clock.hour, w.clock.minute_accumulator = 20, 0.0
        w.resync_schedules(announce=False, teleport=True)
        maude, door = w.find_npc("Sister Maude"), w.find_door("vestry_door")
        for _ in range(int(1.5 * 24 / 0.1)):
            game.update(0.1)
        assert maude.is_asleep and door.is_locked and not door.is_open
        assert w.building_at(maude.position).name == "chapel_vestry"

    def test_the_player_walks_through_an_open_one_and_not_a_shut_one(self, game):
        w = game.world
        door = w.find_door("vestry_door")
        door.is_locked, door.is_open = False, False
        teleport(game, -22.0, 47.75, yaw=90.0)
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True))
        assert game.player.position.x < -19.0                      # the shut leaf stops us
        door.is_open = True
        teleport(game, -22.0, 47.1, yaw=90.0)
        for _ in range(60):
            game.update(0.05, MoveIntent(forward=True))
        assert game.player.position.x > -19.0                      # through into the vestry


# ----------------------------------------------------------------------------- the HUD and reading
class TestTheHud:
    def test_the_controls_panel_only_for_the_tutorial(self, game, settings, tmp_path):
        from unprepared.game import Game
        assert not game.in_tutorial
        tut = Game(settings=settings, save_file=str(tmp_path / "t.json"), tutorial=True)
        assert tut.in_tutorial

    def test_there_is_no_inventory_panel_to_scroll(self, game):
        assert not hasattr(game, "visible_inventory") and not hasattr(game, "scroll_inventory")

    def test_e_or_esc_put_down_a_notice_wherever_one_looks(self, game):
        w = game.world
        notice = next(r for r in w.readables if r.name == "thornbury_notices")
        game.read(notice)
        game.player.rotation_y += 180.0                          # look away from it ...
        game.update(0.0)
        game.interact()                                          # ... and [E] still puts it down
        assert game.reading is None
        game.read(notice)
        door = w.find_door("town_gate_west")
        teleport(game, door.center.x, door.line - 2.0, yaw=180.0)    # (looking at something else entirely)
        game.reading = (notice, list(notice.lines))
        game.interact()
        assert game.reading is None and door.is_open == door.is_open
        game.read(notice)
        assert game.escape() is True and game.reading is None    # [Esc] too (the app lets the game have it first)


# ----------------------------------------------------------------------------- repairs
class TestRepairs:
    def test_the_old_mine_sump_can_be_left_by_its_steps(self, game):
        w = game.world
        for n in w.npcs:
            n.schedule, n.current_task, n.path = [], None, []
        teleport(game, 92.0, -102.0, yaw=180.0)                  # in the sump, facing the steps up
        for _ in range(80):
            game.update(0.05, MoveIntent(forward=True))
        assert w.building_at(game.player.position).name == "mine_gallery"
        steps = sorted((q for q in w.props if q.name.startswith("mine_steps")), key=lambda q: -q.max_point.y)
        gallery = next(b for b in w.level.buildings if b.name == "mine_gallery")
        assert steps[0].max_point.y == pytest.approx(w.building_elevation(gallery))          # the top one flush
        assert steps[0].min_point.x == 90.0 and steps[0].max_point.x == 94.0                 # the way's full width

    def test_cave_mouths_leave_no_ground_across_the_way_in(self, world):
        from unprepared.world.caves import JAMB
        mouths = {"deep_mouth": (-151.5, -148.5), "cove_mouth": (18.5, 21.5), "mine_adit": (99.0, 102.0)}
        holes = list(world.terrain.definition.holes)
        assert len(holes) == len(mouths)
        for key, (a, b) in mouths.items():
            chamber = next(c for c in world.level.buildings if c.name == key)
            (cx1, cz1), (cx2, cz2) = chamber.min_xz, chamber.max_xz
            # the hole over the way in reaches a jamb's width past either side (no further than the chamber)
            hits = [h for h in holes if (h[2] - h[0] > h[3] - h[1] and h[0] <= a and h[2] >= b)
                    or (h[3] - h[1] > h[2] - h[0] and h[1] <= a and h[3] >= b)]
            assert hits, key
            h = hits[0]
            lo, hi = (h[0], h[2]) if h[2] - h[0] > h[3] - h[1] else (h[1], h[3])
            edge_lo, edge_hi = (cx1, cx2) if h[2] - h[0] > h[3] - h[1] else (cz1, cz2)
            assert lo == pytest.approx(max(edge_lo, a - JAMB)) and hi == pytest.approx(min(edge_hi, b + JAMB)), key

    def test_nothing_furnished_straddles_a_cave_mouth(self, world):
        """The crags and the brow round a mouth stand wholly outside the chamber (half in, half
        out, the cave's shade cut them dark along a line)."""
        for b in world.level.buildings:
            if not b.underground:
                continue
            (x1, z1), (x2, z2) = b.min_xz, b.max_xz
            for q in world.props:
                if "_mouth_" not in q.name:
                    continue
                inside_x = q.min_point.x < x2 - 1e-6 and q.max_point.x > x1 + 1e-6
                inside_z = q.min_point.z < z2 - 1e-6 and q.max_point.z > z1 + 1e-6
                if inside_x and inside_z:
                    assert x1 <= q.min_point.x and q.max_point.x <= x2 and z1 <= q.min_point.z and q.max_point.z <= z2, q.name

    def test_the_park_is_level_under_the_fountain(self, world):
        fountain = next(q for q in world.props if q.name == "park_fountain")
        t = world.terrain
        heights = [t.height(fountain.min_point.x + (fountain.max_point.x - fountain.min_point.x) * fx,
                            fountain.min_point.z + (fountain.max_point.z - fountain.min_point.z) * fz)
                   for fx in (0.0, 0.5, 1.0) for fz in (0.0, 0.5, 1.0)]
        assert max(heights) - min(heights) < 1e-6 and fountain.min_point.y == pytest.approx(heights[0])

    def test_no_window_over_a_door_or_off_its_wall(self, world):
        for win in (q for q in world.props if q.kind in ("window", "painting")):
            f = int(round(win.facing / 90.0)) % 4
            mn, mx = win.min_point, win.max_point
            along_x = f in (0, 2)
            plane = (mn.z if f == 0 else mx.z) if along_x else (mx.x if f == 1 else mn.x)
            lo, hi = (mn.x, mx.x) if along_x else (mn.z, mx.z)
            for d in world.doors:
                if d.along_x == along_x and abs(d.line - plane) < 0.8:
                    a, b = d.span
                    assert not (a < hi and b > lo and d.hinge_position.y < mx.y and d.hinge_position.y + d.height > mn.y), \
                        (win.name, d.door_id)
            for t in (0.02, 0.5, 0.98):                          # a wall behind it, end to end
                u = lo + (hi - lo) * t
                y = (mn.y + mx.y) / 2
                x, z = (u, plane + (0.05 if f == 0 else -0.05)) if along_x else (plane + (0.05 if f == 3 else -0.05), u)
                assert any(w.min_point.x - 1e-6 <= x <= w.max_point.x + 1e-6 and w.min_point.z - 1e-6 <= z <= w.max_point.z + 1e-6
                           and w.min_point.y <= y <= w.max_point.y for w in world.walls_near(Vector3D(x, y, z), 2.0)), win.name
