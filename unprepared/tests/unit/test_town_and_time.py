"""The Greyspine made a range of mountains, the steeple's belfry open to the street, the chapel
open by day, the bridges' rails, the graves' headstones, the cell's bed the prisoner's, waits of
any length (and the box over them), and the work done to keep the frames coming: quick hours,
the walks planned ahead of the hour and begun a few at a time, the crosshair's and the doors'
shortcuts, the view frustum."""
from __future__ import annotations

import math
import random
import time

import numpy as np
import pytest

from tests.helpers import spot, teleport
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState
from unprepared.game import Game, LongTask
from unprepared.systems import MoveIntent, justice
from unprepared.world.levels import quarters, wilds


def serving(game, days: int):
    """The player locked up for ``days`` days, standing in the cell."""
    justice.imprison(game.world, game.player, days)
    for _ in range(40):
        game.update(0.1)
    assert game.world.jail.sealed
    return game.world.jail


# ----------------------------------------------------------------------------- the mountains
class TestTheGreyspine:
    def test_a_range_of_peaks_not_a_wall(self, world):
        t = world.terrain
        xs = np.arange(-190.0, 170.0, 2.0)
        front = np.array([t.natural_point(x, -118.0) for x in xs])
        back = np.array([t.natural_point(x, -148.0) for x in xs])
        assert front.max() < 110.0 and back.max() < 140.0                 # mountains, not a 270-unit cliff
        assert back.max() > 80.0                                          # ... but mountains all the same
        peaks = [i for i in range(1, len(front) - 1) if front[i] > front[i - 1] and front[i] >= front[i + 1] and front[i] > 40.0]
        assert len(peaks) >= 5                                            # summits with saddles between
        assert (front.max() - np.percentile(front, 20)) > 20.0

    def test_snow_on_the_peaks_and_rock_on_their_faces(self, world):
        t = world.terrain
        high = t.grid > t.definition.snow_line + 6.0
        assert high.any() and t.snow[high].mean() > 0.6
        assert t.rock[t.slope > 1.5].mean() > 0.8

    def test_the_town_below_is_not_lifted_by_them(self, world):
        """The mountains' flanks fall away well before the town wall."""
        t = world.terrain
        mountains = set(wilds._mountains())
        without = wilds.BASE.__class__(**{**wilds.BASE.__dict__, "hills": tuple(h for h in wilds.HILLS if h not in mountains)})
        from unprepared.world.terrain import Terrain
        bare = Terrain.landform(without, wilds.BOUNDS)
        for x in (-60.0, 0.0, 50.0):
            assert t.natural_point(x, -58.0) - bare.natural_point(x, -58.0) < 1.5

    def test_nobody_climbs_past_the_foot(self, world):
        start = world.spot(0.0, 0.0, 60.0)
        for x in (-175.0, -60.0, 30.0, 110.0):                          # (not over a cave: those are reached underground)
            assert not world.nav.find_path(start, world.spot(x, 0.0, -125.0)), x

    def test_every_cave_is_still_under_rock(self, world):
        t = world.terrain
        for b in world.level.buildings:
            if not b.underground or "mouth" in b.name or "adit" in b.name:
                continue
            top = world.building_elevation(b) + b.wall_height
            (x1, z1), (x2, z2) = b.min_xz, b.max_xz
            for x in np.linspace(x1 + 0.5, x2 - 0.5, 5):
                for z in np.linspace(z1 + 0.5, z2 - 0.5, 5):
                    assert t._sample(t.grid, x, z) > top + 1.0, (b.name, x, z)       # (the hillside drawn over it)

    def test_crags_are_the_same_at_a_point_and_on_the_grid(self, world):
        t = world.terrain
        for x, z in ((-100.0, -110.0), (30.0, -131.0), (-150.0, -84.0), (120.0, -140.0)):
            c = int(round((x - t.min_x) / t.cell))
            r = int(round((z - t.min_z) / t.cell))
            gx, gz = t.min_x + c * t.cell, t.min_z + r * t.cell
            assert t.natural[c, r] == pytest.approx(t._raw_scalar(gx, gz), abs=1e-6)


# ----------------------------------------------------------------------------- the chapel
class TestTheChapel:
    def test_the_belfry_is_open_on_all_four_sides(self, world):
        sx1, sz1, sx2, sz2 = quarters.STEEPLE
        b = next(b for b in world.level.buildings if b.name == "chapel_steeple")
        y = world.building_elevation(b) + quarters.STEEPLE_HEIGHT - 2.0          # the bell's height
        cx, cz = (sx1 + sx2) / 2.0, (sz1 + sz2) / 2.0
        for x, z in ((cx, sz1 + 0.2), (cx, sz2 - 0.2), (sx1 + 0.2, cz), (sx2 - 0.2, cz)):
            solid = [w for w in (*world.walls, *world.lintels) if w.min_point.x <= x <= w.max_point.x
                     and w.min_point.z <= z <= w.max_point.z and w.min_point.y <= y <= w.max_point.y]
            assert solid == [], (x, z)

    def test_the_belfry_stands_over_the_naves_ridge(self, world):
        nave = next(b for b in world.level.buildings if b.name == "chapel")
        steeple = next(b for b in world.level.buildings if b.name == "chapel_steeple")
        assert quarters.STEEPLE_HEIGHT - 3.5 > nave.wall_height + nave.roof_height
        bell = next(p for p in world.props if p.name == "chapel_bell")
        assert bell.min_point.y > world.building_elevation(nave) + nave.wall_height + nave.roof_height
        assert bell.max_point.y < world.building_elevation(steeple) + steeple.wall_height

    def test_its_door_is_open_by_day_whoever_is_in(self, game):
        w = game.world
        door = w.find_door("chapel_door")
        maude = w.find_npc("Sister Maude")
        for hour in (10, 17):                                             # calling on the widow; on her rounds
            w.clock.hour = hour
            w.resync_schedules(announce=False, teleport=True)
            assert not door.is_locked, hour
            assert w.building_at(maude.position) is None or w.building_at(maude.position).name != "chapel", hour
        w.clock.hour = 9
        w.resync_schedules(announce=False, teleport=True)
        for _ in range(3):
            w.wait_hours(1)
            assert not door.is_locked, w.clock.hour
        door.lock_and_close()                                             # (somebody locked it all the same)
        w.wait_hours(1)
        assert not door.is_locked                                         # open again on the hour

    def test_it_is_locked_at_night(self, game):
        w = game.world
        door = w.find_door("chapel_door")
        w.clock.hour, w.clock.minute_accumulator = 20, 59.0
        w.wait_hours(1)
        assert w.clock.hour >= 21 and door.is_locked
        w.clock.hour = 5
        w.wait_hours(1.05)
        assert not door.is_locked

    def test_sister_maude_keeps_the_vestry_not_the_chapel(self, world):
        maude = world.find_npc("Sister Maude")
        plans = [maude.schedule, *getattr(maude, "day_plans", [])]
        for plan in plans:
            locks = {d for t in plan for d in t.lock_doors}
            assert "vestry_door" in locks and "chapel_door" not in locks

    def test_headstones_stand_at_the_head_of_their_graves(self, world):
        from unprepared.rendering import audit
        from unprepared.rendering.furnishings import draw_headstone
        rec = audit.GeometryRecorder()
        grave = next(p for p in world.props if p.kind == "headstone")
        with rec.installed():
            rec.owner = "grave"
            draw_headstone(grave, rec.textures, lambda *a, **k: None, 0.0)
        stone = [t for t in rec.triangles if t.texture == "stone"]
        mound = [t for t in rec.triangles if t.texture == "leaves"]
        stone_z = min(min(t.a[2], t.b[2], t.c[2]) for t in stone)
        mound_z = max(max(t.a[2], t.b[2], t.c[2]) for t in mound)
        assert stone_z > mound_z                                          # the stone beyond the mound (facing 0: it looks -z)
        assert max(max(t.a[2], t.b[2], t.c[2]) for t in stone) <= grave.max_point.z + 0.35   # (leaning back a little)
        assert min(min(t.a[2], t.b[2], t.c[2]) for t in mound) >= grave.min_point.z - 1e-6    # all of it on the plot


# ----------------------------------------------------------------------------- the bridges
class TestTheBridges:
    def test_the_rails_are_solid_to_walk_into(self, game):
        w = game.world
        rails = [r for r in w.rails if r.name.startswith("market_bridge")]
        assert len(rails) == 2
        rail = min(rails, key=lambda r: r.min_point.z)                  # the north rail, along x
        teleport(game, -47.5, rail.max_point.z + 1.0, yaw=0.0)             # on the deck, facing north at it
        for _ in range(40):
            game.update(0.05, MoveIntent(forward=True))
        assert game.player.position.z > rail.max_point.z                  # stopped at the rail
        assert any(r in w.colliders(near=game.player.position) for r in rails)

    def test_no_way_is_planned_through_a_rail_or_round_the_bridge_through_the_water(self, world):
        nav = world.nav
        for rail in world.rails:
            c = nav.to_cell((rail.min_point.x + rail.max_point.x) / 2.0, (rail.min_point.z + rail.max_point.z) / 2.0)
            assert nav.static_blocked[c]
        bridge = next(p for p in world.props if p.name == "market_bridge")
        path = nav.find_path(world.spot(-58.0, 0.0, -4.5), world.spot(-37.0, 0.0, 4.8), world.doors)
        assert path
        pts = [world.spot(-58.0, 0.0, -4.5), *path]
        for a, b in zip(pts, pts[1:]):                                   # every leg over the water is on the deck
            for k in range(21):
                x, z = a.x + (b.x - a.x) * k / 20, a.z + (b.z - a.z) * k / 20
                if world.terrain.water_level_at(x, z) is not None:
                    assert bridge.min_point.x <= x <= bridge.max_point.x and bridge.min_point.z + 0.4 < z < bridge.max_point.z - 0.4

    def test_nobody_crosses_a_rail_or_wades_beside_the_bridge_all_day(self, world):
        bridges = [p for p in world.props if p.kind == "bridge"]
        last = {}
        for _ in range(int(24 * 24 / 0.25)):                             # a day, a quarter of a second at a time
            world.update_time(0.25)
            world.update_npcs(0.25)
            for npc in world.npcs:
                p = npc.position
                for b in bridges:
                    mn, mx = b.min_point, b.max_point
                    if not (mn.x - 6 < p.x < mx.x + 6 and mn.z - 6 < p.z < mx.z + 6):
                        continue
                    on_deck = mn.x <= p.x <= mx.x and mn.z <= p.z <= mx.z
                    water = world.terrain.water_level_at(p.x, p.z)
                    assert on_deck or water is None or water <= world.terrain.height(p.x, p.z) + 0.02, (npc.name, p)
                    q = last.get(npc.name)
                    if q is not None and mn.x + 0.2 < p.x < mx.x - 0.2:
                        for edge in (mn.z, mx.z):
                            assert (q[1] - edge) * (p.z - edge) >= 0, (npc.name, q, p)
                last[npc.name] = (p.x, p.z)


# ----------------------------------------------------------------------------- the cells
class TestThePrisonersBed:
    def test_the_cell_bed_is_the_prisoners_for_the_sentence_and_no_longer(self, game):
        w = game.world
        bed = next(p for p in w.props if p.name == "jail_cell_bed")
        hesk = next(p for p in w.props if p.name == "jail_guard_bed")
        assert not w.player_may_sleep_in(bed)
        serving(game, 2)
        assert w.player_may_sleep_in(bed) and not w.player_may_sleep_in(hesk)
        assert game.sleep(bed)                                            # and lying down in it passes the night
        for _ in range(400):
            game.update(0.1)
            if game.fade is None:
                break
        assert w.jail.sealed and w.clock.hour in (Game.WAKE_HOUR, Game.NIGHTFALL_HOUR)
        game.wait_out_sentence()
        assert not w.jail.sealed and not w.player_may_sleep_in(bed)

    def test_a_breakout_ends_it_too(self, game):
        w = game.world
        bed = next(p for p in w.props if p.name == "jail_cell_bed")
        serving(game, 3)
        game.player.position = spot(w, -45.0, -28.0)
        game.update(0.05)
        assert w.jail.escaped and not w.player_may_sleep_in(bed)


# ----------------------------------------------------------------------------- long waits
class TestLongWaits:
    def test_a_month_in_the_cells_is_served_whole_and_quickly(self, game):
        jail = serving(game, 30)
        end = jail.sentence_end_day
        started = time.perf_counter()
        waited = game.wait_out_sentence()
        assert time.perf_counter() - started < 20.0
        assert not jail.serving and jail.released
        assert game.world.day_count == end and game.world.clock.hour == game.settings.justice.release_hour
        assert waited >= 30 * 24 - 24                                     # (no fifteen-day cap any more)

    def test_the_town_lived_through_it(self, game):
        serving(game, 5)
        w = game.world
        eaten = {c.name: sum(i.quantity for i in c.inventory if i.is_food) for c in w.containers if c.name == "barnaby_table"}
        game.wait_out_sentence()
        assert not [n.name for n in w.npcs if n.hungry]                   # everybody ate (the tables restocked)
        assert w.day_count == 5 and eaten

    def test_in_the_app_it_passes_a_frame_at_a_time_under_a_box(self, game):
        game.long_task_budget = 0.01
        jail = serving(game, 4)
        game.wait_out_sentence()
        assert game.long_task is not None and game.busy_label == "Serving your sentence"
        assert game.frozen and not game.can_save
        frames, progress = 0, []
        while game.long_task is not None and frames < 5000:
            game.update(1 / 60)
            frames += 1
            if game.long_task is not None:
                progress.append(game.long_task.progress)
        assert frames > 3 and not jail.serving
        assert progress == sorted(progress) and progress[-1] <= 1.0

    def test_sleep_is_quick_hours_and_ends_on_the_stroke(self, game):
        w = game.world
        w.clock.hour, w.clock.minute_accumulator = 21, 30.0
        started = time.perf_counter()
        w.wait_until(6)
        assert time.perf_counter() - started < 5.0
        assert w.clock.hour == 6 and w.clock.minute_accumulator == 0.0 and w.day_count == 1

    def test_the_t_wait_is_one_too_and_a_load_drops_it(self, game):
        assert game.save_game()
        game.long_task_budget = 0.001
        assert game.wait()
        assert game.long_task is not None and game.busy_label == "Waiting" and game.frozen
        assert not game.save_game()                                      # (refused while the hour passes)
        assert game.load_game()
        assert game.long_task is None and not game.frozen

    def test_a_long_task_reports_how_far_it_has_got(self):
        seen = []

        def steps():
            for i in range(1, 11):
                time.sleep(0.001)
                yield i / 10

        task = LongTask("Counting", steps(), on_done=lambda: seen.append("done"))
        assert not task.advance(0.0) and 0.0 < task.progress < 1.0
        assert task.advance(None) and task.progress == 1.0
        task.finish()
        task.finish()
        assert seen == ["done"]


class TestQuickHours:
    def test_a_quick_hour_puts_everybody_where_they_were_going(self, game):
        w = game.world
        w.clock.hour = 7
        w.resync_schedules(announce=False, teleport=True)
        w.pass_hour(quick=True)
        assert w.clock.hour == 8 and w.clock.minute_accumulator == 0.0
        for npc in w.npcs:
            task = npc.current_task
            if task is None or npc.is_busy or npc.locked_out or w.is_frozen(npc):
                continue
            assert npc.state is not AIState.TRAVELING or npc.path, npc.name

    def test_owners_lock_up_behind_them_as_they_would_walking(self, game):
        w = game.world
        w.clock.hour = 21
        w.resync_schedules(announce=False, teleport=True)
        for _ in range(12):
            w.pass_hour(quick=True)                                      # 21:00 -> 09:00
        barnaby = w.find_npc("Shopkeeper Barnaby")
        assert barnaby.state is AIState.SHOPKEEPING
        assert w.find_door("home_door").is_locked                        # (he left home for the shop)
        assert not w.find_door("shop_door").is_locked                    # (and opened it for business)

    def test_a_quick_day_costs_little(self, game):
        started = time.perf_counter()
        for _ in range(24):
            game.world.pass_hour(quick=True)
        assert time.perf_counter() - started < 3.0


# ----------------------------------------------------------------------------- keeping the frames coming
class TestFrameTime:
    def test_walks_are_planned_ahead_of_the_hour(self, game):
        w = game.world
        w.planning_budget = 0.05
        while w.clock.minute_accumulator < w.PLAN_FROM_MINUTE:
            w.update_time(0.5)
            w.update_npcs(0.5)
        misses = []
        real = w.nav._astar

        def counting(*args):
            misses.append(args)
            return real(*args)
        for _ in range(40):
            w.update_npcs(0.5)
        w.nav._astar = counting
        hour = w.clock.hour
        while w.clock.hour == hour:
            w.update_time(0.25)
        w.update_npcs(0.01)
        changed = [n for n in w.npcs if n.current_task is not None and n.current_task.start_hour == w.clock.hour]
        assert changed and len(misses) < len(changed) / 2              # most walks came out of the cache

    def test_setting_off_is_staggered_in_the_app(self, game):
        w = game.world
        w.switch_budget = 0.0                                             # one a frame
        w.clock.hour, w.clock.minute_accumulator = 7, 59.99
        w.update_time(0.1)
        waiting = [n for n in w.npcs if n.task_for(8) is not n.current_task and not w.is_frozen(n)]
        assert waiting
        w.update_npcs(0.01)
        still = [n for n in w.npcs if n.task_for(8) is not n.current_task and not w.is_frozen(n)]
        assert len(still) == len(waiting) - 1
        for _ in range(len(waiting) + 5):
            w.update_npcs(0.01)
        assert not [n for n in w.npcs if n.task_for(8) is not n.current_task and not w.is_frozen(n)
                    and not n.is_busy and n.suspicion_meter < n.settings.schedule_override_threshold]

    def test_the_crosshair_looks_only_at_what_is_near(self, game):
        from unprepared.systems import find_interaction_target
        w, p = game.world, game.player
        rng = random.Random(3)
        spots = [(-13.0, -9.0), (-4.5, 4.0), (-30.0, -10.0), (52.0, 10.0), (-24.0, 40.0), (0.0, 90.0)]
        for x, z in spots:
            for _ in range(6):
                teleport(game, x + rng.uniform(-2, 2), z + rng.uniform(-2, 2), yaw=rng.uniform(0, 360),
                         pitch=rng.uniform(-30, 10))
                doors, npcs, containers, stations, beds = w.interaction_candidates(p.position)
                near = find_interaction_target(p, doors, npcs, game.settings.interaction, containers,
                                               w.walls_near(p.position, game.TARGET_OCCLUDER_REACH), stations,
                                               game.settings, beds=beds, may_sleep_in=w.player_may_sleep_in, hour=w.clock.hour)
                full = find_interaction_target(p, w.doors, w.npcs, game.settings.interaction, w.containers,
                                               w.walls_near(p.position, game.TARGET_OCCLUDER_REACH), w.stations,
                                               game.settings, beds=w.beds, may_sleep_in=w.player_may_sleep_in, hour=w.clock.hour)
                assert (near.obj if near else None) is (full.obj if full else None)

    def test_lookups_by_cell_agree_with_searching_everything(self, world):
        rng = random.Random(5)
        for _ in range(400):
            p = Vector3D(rng.uniform(-100, 100), rng.uniform(0, 20), rng.uniform(-70, 130))
            brute = next((b for b in world.level.buildings if not b.open_sides and b.min_xz[0] <= p.x <= b.max_xz[0]
                          and b.min_xz[1] <= p.z <= b.max_xz[1] and not (b.underground and p.y > world.building_elevation(b)
                                                                          + b.wall_height + 0.5)), None)
            assert world.building_at(p) is brute
            area = next((a for a in world.level.areas if a.contains(p.x, p.z)), None)
            if area is None or not getattr(area, "underground", False):
                assert world.area_def_at(p) is area
            x1, z1 = p.x - 3.0, p.z - 3.0
            found = set(d.door_id for d in world.doors_in(x1, z1, p.x + 3.0, p.z + 3.0))
            brute_doors = {d.door_id for d in world.doors if d.x1 <= p.x + 3 and d.x2 >= x1 and d.z1 <= p.z + 3 and d.z2 >= z1}
            assert found == brute_doors

    def test_a_doors_cached_shape_follows_it(self):
        from unprepared.entities import Door
        door = Door("d", Vector3D(0.0, 0.0, 0.0), width=2.0, is_locked=False)
        shut = door.collider()
        assert door.collider() is shut
        door.toggle()
        opened = door.collider()
        assert opened is not shut and opened.max_point.x - opened.min_point.x < 0.5
        door.hinge_position = Vector3D(5.0, 0.0, 0.0)
        assert door.x1 == pytest.approx(5.0) and door.collider().min_point.x > 4.0
        c = door.center
        c.x = 99.0
        assert door.center.x != 99.0                                      # (the cached centre is handed out as a copy)

    def test_the_line_of_sight_is_never_clearer_than_sampling_it(self, world):
        from unprepared.world.navigation import _Blocked
        nav = world.nav
        blocked = _Blocked(nav._static_rows, set())
        rng = random.Random(11)

        def sampled(a, b):
            rows = blocked.rows
            steps = max(1, int(4 * max(abs(b[0] - a[0]), abs(b[1] - a[1]))))
            prev = a
            for i in range(1, steps + 1):
                t = i / steps
                cx = int(math.floor(a[0] + 0.5 + (b[0] - a[0]) * t))
                cz = int(math.floor(a[1] + 0.5 + (b[1] - a[1]) * t))
                if rows[cx][cz]:
                    return False
                if cx != prev[0] and cz != prev[1] and (rows[prev[0]][cz] or rows[cx][prev[1]]):
                    return False
                prev = (cx, cz)
            return True
        agree = 0
        for _ in range(3000):
            a = (rng.randrange(300, 500), rng.randrange(100, 400))
            b = (a[0] + rng.randrange(-25, 26), a[1] + rng.randrange(-25, 26))
            fast = nav._line_clear(a, b, blocked)
            assert not (fast and not sampled(a, b)), (a, b)
            agree += fast == sampled(a, b)
        assert agree > 2950

    def test_the_view_frustum(self):
        from unprepared.rendering.camera import Frustum
        f = Frustum((0.0, 1.6, 0.0), (0.0, 1.6, -1.0), 60.0, 4.0 / 3.0, margin=0.0)
        assert f.sees_box(-1, 0, -11, 1, 3, -9)                          # straight ahead
        assert not f.sees_box(-1, 0, 9, 1, 3, 11)                        # behind
        assert not f.sees_box(30, 0, -11, 32, 3, -9)                     # far off to the right
        assert f.sees_box(-1, 0, -1, 1, 3, 1)                            # all round the eye
        assert f.sees_sphere(0, 1.6, -50, 1.0) and not f.sees_sphere(0, 1.6, 50, 1.0)

    def test_a_new_game_of_the_same_town_reuses_the_bake(self, game, settings, tmp_path):
        from unprepared.rendering.batching import StaticScene
        other = Game(settings=settings, save_file=str(tmp_path / "other.json"))
        scene = StaticScene.__new__(StaticScene)
        scene.world, scene.textures, scene.ground_cells = game.world, object(), 20
        assert scene.matches(other.world, scene.textures, 20)
        assert not scene.matches(other.world, object(), 20) and not scene.matches(other.world, scene.textures, 40)
