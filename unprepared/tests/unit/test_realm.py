"""The realm filled in: the caves dug into the mountainside, the trails between everything, the
edges of the map (mountain, thicket, sea), bridges with their rails on the water sides, door
names over the prompt, things to read, hiding places only a searcher finds, the manors' upper
floors walked through their doorways, the townsfolk's stories and trades, and pantries that
last the three days between shopping trips."""
from __future__ import annotations

import collections
import math

import pytest

from tests.helpers import spot, teleport, yaw_towards
from unprepared.core import dialogue
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState
from unprepared.systems import MoveIntent
from unprepared.systems.targeting import HIDDEN_REACH, TargetKind
from unprepared.world import get_level
from unprepared.world.caves import Chamber, Mouth, clear_the_floor, dig, is_dressing
from unprepared.world.level import PropDef
from unprepared.world.levels import wilds


def look_at(game, target: Vector3D) -> None:
    """Turn the player's eyes onto ``target``."""
    eye = game.player.eye_position
    game.player.rotation_y = yaw_towards(eye, target)
    flat = math.hypot(target.x - eye.x, target.z - eye.z)
    game.player.rotation_x = math.degrees(math.atan2(target.y - eye.y, flat))
    game.update(0.0)


def settle(game, seconds, dt=0.05, intent=MoveIntent()):
    for _ in range(int(seconds / dt)):
        game.update(dt, intent)


def calm(world):
    for n in world.npcs:
        n.schedule, n.current_task, n.path = [], None, []


# ----------------------------------------------------------------------------- caves
class TestCaves:
    CAVES = {"deep": "the Howling Deep", "cove": "the smugglers' cave", "mine": "the old mine"}

    def test_caves_are_dug_down_under_the_hillside(self, world):
        """Every chamber lies under the ground above it (its roof below the hillside's surface),
        bar the mouth, where the way in is cut into the slope; a hole in the drawn ground opens
        the mouth."""
        t = world.terrain
        caves = [b for b in world.level.buildings if b.underground]
        assert len(caves) >= 15
        mouths = {"deep_mouth", "cove_mouth", "mine_adit"}
        for b in caves:
            (x1, z1), (x2, z2) = b.min_xz, b.max_xz
            roof = b.elevation + b.wall_height
            if b.name in mouths:
                continue
            lowest = min(t.natural_height(x1 + (x2 - x1) * fx, z1 + (z2 - z1) * fz)
                         for fx in (0.1, 0.5, 0.9) for fz in (0.1, 0.5, 0.9))
            assert lowest > roof, (b.name, lowest, roof)
        holes = t.definition.holes
        assert len(holes) == 3 and all(t.in_hole((h[0] + h[2]) / 2, (h[1] + h[3]) / 2) for h in holes)

    def test_the_caves_are_spread_out(self, world):
        centres = {}
        for b in world.level.buildings:
            if b.underground:
                cave = b.name.split("_")[0]
                centres.setdefault(cave, []).append(((b.min_xz[0] + b.max_xz[0]) / 2, (b.min_xz[1] + b.max_xz[1]) / 2))
        mid = {k: (sum(p[0] for p in v) / len(v), sum(p[1] for p in v) / len(v)) for k, v in centres.items()}
        for a in mid:
            for b in mid:
                if a < b:
                    assert math.dist(mid[a], mid[b]) > 100.0, (a, b)

    def test_inside_a_cave_is_underground_dark_and_named(self, world):
        for x, z, name in ((-151.0, -91.0, "the Howling Deep"), (106.0, 24.0, "the smugglers' cave"), (100.0, -88.0, "the old mine")):
            p = world.spot(x, 0.0, z)
            b = world.building_at(p)
            assert b is not None and b.underground and not b.lit, name
            assert world.area_of(p) == name
            assert p.y < world.terrain.natural_height(x, z) - 3.0            # down under the hill
        # (and on the hillside over the Deep's hall it is the open air)
        above = Vector3D(-151.0, world.terrain.natural_height(-151.0, -91.0), -91.0)
        assert world.building_at(above) is None

    def test_the_player_walks_in_by_the_mouth_and_down_the_steps(self, game):
        w, p = game.world, game.player
        calm(w)
        teleport(game, -150.0, -45.0, yaw=0.0)                          # on the trail, below the mouth
        settle(game, 14.0, intent=MoveIntent(forward=True))
        b = w.building_at(p.position)
        assert b is not None and b.name == "deep_hall" and p.position.y == pytest.approx(wilds.DEEP_HALL_FLOOR, abs=0.05)

    def test_every_cave_has_ore_bones_and_a_hiding_place(self, world):
        for prefix, area in self.CAVES.items():
            inside = [c for c in world.containers if world.area_of(c.center) == area]
            assert any(c.kind == "skeleton" for c in inside), prefix
            assert any(c.hidden for c in inside), prefix
            seams = [s for s in world.stations if world.area_of(s.center) == area]
            assert seams or prefix == "cove", prefix
            shrooms = [q for q in world.props if q.kind == "mushroom" and q.name.startswith(prefix)]
            assert shrooms, prefix
        assert {f.kind for f in world.fixtures if f.room == "the old mine"} == {"glow"}     # (nobody tends a lamp there)

    def test_the_dug_rock_keeps_off_what_is_furnished(self, world):
        things = [c for c in world.containers] + list(world.stations) + [q for q in world.props if q.kind in ("bedroll", "table")]
        dressing = [q for q in world.props if q.kind in ("boulder", "stalagmite") and ("_rock_" in q.name or "_mite_" in q.name)]
        assert dressing
        for rock in dressing:
            for t in things:
                assert not (rock.min_point.x < t.max_point.x and t.min_point.x < rock.max_point.x
                            and rock.min_point.z < t.max_point.z and t.min_point.z < rock.max_point.z), (rock.name, t)

    def test_clear_the_floor_takes_up_rock_in_the_way(self):
        rock = PropDef("x_hall_rock_1", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), kind="boulder")
        mite = PropDef("x_hall_mite_1", (5.0, 0.0, 5.0), (5.3, 1.0, 5.3), kind="stalagmite")
        tite = PropDef("x_hall_tite_1", (8.0, 2.0, 8.0), (8.3, 3.0, 8.3), kind="stalactite", solid=False)
        chest = PropDef("chest", (0.5, 0.0, 0.5), (1.5, 1.0, 1.5))
        beam = PropDef("beam", (7.0, 2.5, 7.0), (9.0, 3.0, 9.0))
        assert is_dressing(rock) and is_dressing(mite) and not is_dressing(tite) and not is_dressing(chest)
        kept = clear_the_floor([rock, mite, tite, chest, beam], [chest, beam], spots=[(5.6, 5.1)])
        assert kept == [chest, beam]                    # the rock under the chest, the mite on a spot, the tite in the beam

    def test_dig_cuts_passages_lintels_and_a_mouth(self):
        chambers = [Chamber("a", (0.0, 0.0, 10.0, 10.0), 0.0, 4.0), Chamber("b", (10.0, 2.0, 20.0, 8.0), -1.0, 3.0)]
        dug = dig("t", chambers, [(9.0, 4.0, 11.0, 6.0)], Mouth("a", "south", 4.0, 6.0), seed=1)
        assert {b.name for b in dug.buildings} == {"a", "b"} and all(b.underground for b in dug.buildings)
        assert dug.holes == [(2.5, 9.0, 7.5, 10.0)]                  # (the way in and a jamb's length either side)
        # no rock stands in the passage below its lintel, nor in the mouth
        for w in dug.walls:
            (x1, y1, z1), (x2, _, z2) = w.min_point, w.max_point
            in_passage = x1 < 11.0 and x2 > 9.0 and z1 < 6.0 and z2 > 4.0
            in_mouth = x1 < 6.0 and x2 > 4.0 and z2 > 9.0
            assert not ((in_passage or in_mouth) and y1 < 2.0), w
        # the lower chamber is tall enough for the way into the higher one
        b = next(c for c in dug.buildings if c.name == "b")
        assert b.elevation + b.wall_height >= 0.0 + 3.2


# ----------------------------------------------------------------------------- the land
class TestTheLand:
    def test_trails_are_never_too_steep_to_walk(self, world):
        t = world.terrain
        for trail in t.definition.trails:
            pts = trail.points
            for (x1, z1), (x2, z2) in zip(pts, pts[1:]):
                n = max(2, int(math.hypot(x2 - x1, z2 - z1) / 1.0))
                for i in range(n):
                    ax, az = x1 + (x2 - x1) * i / n, z1 + (z2 - z1) * i / n
                    bx, bz = x1 + (x2 - x1) * (i + 1) / n, z1 + (z2 - z1) * (i + 1) / n
                    rise = abs(t.height(bx, bz) - t.height(ax, az))
                    assert rise / math.hypot(bx - ax, bz - az) <= world.WALK_GRADE, (trail.name, ax, az)

    def test_every_trail_starts_and_ends_somewhere(self, world):
        """No trail peters out in the woods: each end meets a road, another trail, or a place."""
        trails = world.terrain.definition.trails
        roads = world.level.roads
        places = [(b.min_xz, b.max_xz) for b in world.level.buildings] + [((x - 8, z - 8), (x + 8, z + 8)) for x, z in
                                                                             ((wilds.SHRINE[0], wilds.SHRINE[1]),)]
        places += [((lake.center[0] - lake.radius_x - 6, lake.center[1] - lake.radius_z - 6),       # (down to the lake shore)
                    (lake.center[0] + lake.radius_x + 6, lake.center[1] + lake.radius_z + 6)) for lake in wilds.LAKES]

        def near_road(x, z, reach=3.0):
            return any(r.min_xz[0] - reach <= x <= r.max_xz[0] + reach and r.min_xz[1] - reach <= z <= r.max_xz[1] + reach
                       for r in roads)

        def near_trail(x, z, me, reach=3.0):
            for other in trails:
                if other is me:
                    continue
                for (ax, az), (bx, bz) in zip(other.points, other.points[1:]):
                    sx, sz = bx - ax, bz - az
                    k = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / max(sx * sx + sz * sz, 1e-9)))
                    if math.hypot(x - ax - k * sx, z - az - k * sz) <= reach:
                        return True
            return False

        def near_place(x, z, reach=4.0):
            return any(a[0] - reach <= x <= b[0] + reach and a[1] - reach <= z <= b[1] + reach for a, b in places)
        for trail in trails:
            for x, z in (trail.points[0], trail.points[-1]):
                assert near_road(x, z) or near_trail(x, z, trail) or near_place(x, z), (trail.name, x, z)

    def test_the_sea_is_deep_and_the_beach_sandy(self, world):
        t = world.terrain
        assert t.water_level_at(170.0, 0.0) is not None and t.water_level_at(170.0, 0.0) - t.height(170.0, 0.0) > 1.0
        assert t.water_level_at(100.0, 0.0) is None                       # inland
        assert not world.can_step(160.0, 0.0, t.height(160.0, 0.0), 170.0, 0.0)
        assert world.area_of(world.spot(140.0, 0.0, 60.0)) == "the beach"

    def test_nobody_plans_a_walk_up_a_cliff_or_out_to_sea(self, world):
        start = world.spot(0.0, 0.0, 60.0)
        assert not world.nav.find_path(start, world.spot(-120.0, 0.0, -110.0))        # the top of the Greyspine
        assert not world.nav.find_path(start, world.spot(172.0, 0.0, 60.0))           # out to sea

    def test_bridges_carry_rails_along_their_water_sides(self, world):
        bridges = [q for q in world.props if q.kind == "bridge"]
        assert bridges
        for b in bridges:
            # the deck's facing says which way it spans (90: from bank to bank along x); the rails run
            # along the span, on its two sides - over the water, which runs the other way under it
            assert b.facing in (0.0, 90.0, 180.0, 270.0)
            spans_x = b.facing in (90.0, 270.0)
            nearest = None
            for water in world.terrain.water:
                for (x1, z1, _), (x2, z2, _) in zip(water["points"], water["points"][1:]):
                    d = math.hypot((x1 + x2) / 2 - b.center.x, (z1 + z2) / 2 - b.center.z)
                    if nearest is None or d < nearest[0]:
                        nearest = (d, x2 - x1, z2 - z1)
            _, dx, dz = nearest
            assert (abs(dz) > abs(dx)) == spans_x, b.name


# ----------------------------------------------------------------------------- doors and reading
class TestNamesAndNotices:
    def test_every_door_has_a_name_shown_over_its_prompt(self, game):
        w = game.world
        assert all(d.label for d in w.doors)
        door = w.find_door("shop_door")
        look_at(game, door.center)
        teleport(game, door.center.x, door.hinge_position.z + 2.0, yaw=0.0)
        look_at(game, door.center)
        assert game.target is not None and game.target.kind is TargetKind.DOOR and game.interaction_prompt
        assert game.interaction_title == "Barnaby's General Store"

    def test_a_notice_is_read_on_screen_and_logged_once(self, game):
        w = game.world
        notice = next(r for r in w.readables if r.name == "thornbury_notices")
        teleport(game, notice.center.x, notice.center.z + 2.2, yaw=0.0)
        look_at(game, notice.center)
        assert game.target is not None and game.target.kind is TargetKind.READABLE
        assert game.interaction_prompt == "[E] Read the notice" and game.interaction_title == notice.label
        logged = len(game.messages.messages)
        game.interact()
        assert game.reading is not None and game.reading[0] is notice and game.reading[1] == list(notice.lines)
        assert len(game.messages.messages) == logged + 1 and game.messages.messages[-1].startswith(notice.label)
        game.interact()                                                   # put it down
        assert game.reading is None
        game.interact()
        assert game.reading is not None and len(game.messages.messages) == logged + 1    # (not logged twice)
        teleport(game, notice.center.x, notice.center.z + 9.0)          # walk off: stop reading
        game.update(0.1)
        assert game.reading is None

    def test_what_was_read_is_remembered_by_a_save(self, game, tmp_path):
        notice = next(r for r in game.world.readables if r.name == "thornbury_notices")
        game.read(notice)
        game.read(notice)
        path = str(tmp_path / "read.json")
        assert game.save_game(path)
        game.read_already.clear()
        assert game.load_game(path)
        assert "thornbury_notices" in game.read_already and game.reading is None

    def test_a_hiding_place_is_found_only_by_somebody_crouched_and_close(self, game):
        w = game.world
        stash = w.find_container("barnaby_takings")
        assert stash.hidden
        teleport(game, stash.center.x, stash.center.z + 1.6)
        look_at(game, stash.center)
        assert game.target is None or game.target.obj is not stash               # standing: nothing there
        game.toggle_crouch()
        settle(game, 1.0)
        look_at(game, stash.center)
        assert game.target is not None and game.target.obj is stash
        teleport(game, stash.center.x, stash.center.z + HIDDEN_REACH + 1.0)      # too far, even crouched
        look_at(game, stash.center)
        assert game.target is None or game.target.obj is not stash


# ----------------------------------------------------------------------------- buildings
class TestBuildings:
    def test_the_towers_are_crenellated(self, world):
        """Merlons stand round the top of the watchtower and of Greyhold's towers, gaps between them."""
        from unprepared.world.levels import town
        from unprepared.world.tower import MERLON_H, crenellations
        for x1, z1, x2, z2 in ((town.TOWER_X1, town.TOWER_Z1, town.TOWER_X2, town.TOWER_Z2), (71.0, 95.0, 75.0, 99.0),
                               (113.0, 133.0, 117.0, 137.0)):
            merlons = [w for w in world.walls if x1 - 0.01 <= w.min_point.x and w.max_point.x <= x2 + 0.01
                       and z1 - 0.01 <= w.min_point.z and w.max_point.z <= z2 + 0.01
                       and w.max_point.y - w.min_point.y == pytest.approx(MERLON_H)]
            assert len(merlons) >= 8, (x1, z1)
            assert all(w.max_point.x - w.min_point.x < 1.0 or w.max_point.z - w.min_point.z < 1.0 for w in merlons)
        pieces = crenellations(0.0, 0.0, 4.0, 4.0, 8.0, wall=0.6)
        north = sorted((w.min_point[0], w.max_point[0]) for w in pieces if w.min_point[2] == 0.0)
        assert north[0][0] == 0.0 and north[-1][1] == 4.0 and all(b[0] > a[1] for a, b in zip(north, north[1:]))   # embrasures

    def test_the_chapel_has_a_steeple_with_a_spire_and_a_bell(self, world):
        steeple = next(b for b in world.level.buildings if b.name == "chapel_steeple")
        nave = next(b for b in world.level.buildings if b.name == "chapel")
        assert steeple.roof_style == "spire" and steeple.wall_height > nave.wall_height + 3.0
        assert any(q.kind == "bell" for q in world.props)

    def test_upstairs_in_a_manor_nobody_walks_through_a_partition(self, world):
        """The study, the dressing room (through the study), the bedroom and the second bedroom
        are walked into by their doorways: no leg of a walk upstairs crosses a wall."""
        greaves = world.find_npc("Greaves")
        stair = next(s for s in world.stairs if s.name == "vane_stair")
        top = Vector3D(*stair.top)
        walls = [w for w in world.walls_near(top, 14.0) if w.min_point.y <= top.y + 1.0 <= w.max_point.y]
        spots = [task.target_pos for n in world.npcs for task in n.schedule
                 if stair.contains(task.target_pos.x, task.target_pos.z) and task.target_pos.y > top.y - 0.2]
        spots += [p for task in greaves.schedule for p in task.patrol_points if p.y > top.y - 0.2]
        assert len(spots) >= 4

        def crosses(a, b, w):
            for i in range(1, 40):
                x, z = a.x + (b.x - a.x) * i / 40, a.z + (b.z - a.z) * i / 40
                if w.min_point.x < x < w.max_point.x and w.min_point.z < z < w.max_point.z:
                    return True
            return False
        for a in spots:
            for b in spots:
                greaves.position = a.copy()
                path = [a] + world.find_path(greaves, b)
                for p, q in zip(path, path[1:]):
                    assert not any(crosses(p, q, w) for w in walls), (a, b, p, q)

    def test_the_manors_are_furnished_like_the_rich_live(self, world):
        kinds = collections.Counter(q.kind for q in world.props if q.name.startswith(("vane_", "ashgrove_")))
        for kind in ("wardrobe", "bookcase", "chandelier", "fireplace", "stove", "painting", "rug", "window", "portico",
                     "chimney", "bathtub"):
            assert kinds[kind] >= 2, kind
        for prefix in ("vane", "ashgrove"):
            assert world.find_container(f"{prefix}_servant_stash").hidden

    def test_a_rug_or_a_painting_in_a_bedroom_is_not_a_bed(self, world):
        rug = next(q for q in world.props if q.name == "vane_bedroom_rug")
        assert not rug.is_bed and rug not in world.beds
        assert all(b.kind in ("bed", "bedroll") for b in world.beds)

    def test_every_lantern_hangs_on_a_wall(self, world):
        for f in world.fixtures:
            if f.kind not in ("lantern", "sconce"):
                continue
            p = Vector3D(*f.position)
            assert any(w.min_point.x - 0.45 <= p.x <= w.max_point.x + 0.45 and w.min_point.z - 0.45 <= p.z <= w.max_point.z + 0.45
                       and w.min_point.y <= p.y <= w.max_point.y for w in world.walls_near(p, 2.0)), f

    def test_no_ground_shows_through_a_doorway(self, world):
        """A building's floor stops at its walls' inner faces: under every door in an outer wall (and
        through the chapel's open ways) a threshold carries it to the door."""
        for door in world.doors:
            h, c = door.hinge_position, door.center
            for sign in (-1.0, 1.0):
                x, z = door.to_xz(door.along_of(c.x, c.z), door.line + sign * 0.3)
                b = world.building_at(Vector3D(x, h.y + 0.5, z))
                if b is None or b.underground or b.open_sides:
                    continue
                edge = door.across_of(*b.min_xz) if sign > 0 else door.across_of(*b.max_xz)
                if abs(edge - door.line) > 0.05:
                    continue
                px, pz = door.to_xz(door.along_of(c.x, c.z), door.line + sign * 0.25)
                assert any(q.kind == "floor" and q.min_point.x <= px <= q.max_point.x and q.min_point.z <= pz <= q.max_point.z
                           and q.max_point.y == pytest.approx(h.y + 0.05) for q in world.floors), (door.door_id, sign)
        for name in ("chapel_threshold_steeple", "vestry_door_threshold_w", "vestry_door_threshold_e", "chapel_vestry_boards"):
            assert any(q.name == name for q in world.floors), name

    def test_candles_burn_only_on_tables_under_a_roof(self, world):
        for f in world.fixtures:
            if f.kind == "candle" and f.room.endswith("table"):
                assert world.building_at(Vector3D(*f.position)) is not None, f


# ----------------------------------------------------------------------------- people
class TestTheTownsfolk:
    def test_the_town_is_full_of_people_with_trades(self, world):
        in_town = [n for n in world.npcs if world.in_town(n.home_position)]
        assert len(in_town) >= 30
        roles = {n.role_tag for n in in_town}
        assert {"priest", "crier", "baker", "cooper", "stablehand", "herbwife", "locksmith", "reeve", "minstrel",
                "tanner", "lamplighter", "butler", "maid", "washerwoman"} <= roles

    def test_everybody_in_town_has_a_story_to_tell(self, world):
        for n in world.npcs:
            if world.in_town(n.home_position) and not n.is_guard or n.role_tag in ("rich", "butler", "maid"):
                assert len(n.personal_lines) >= 2, n.name
        bertram = world.find_npc("Old Bertram")
        assert any("arrow in the knee" in line for line in bertram.personal_lines)

    def test_a_story_is_told_a_line_every_other_time_at_a_civil_hour(self, world):
        bertram = world.find_npc("Old Bertram")
        bertram.state = AIState.WORKING
        first = bertram.greeting(10)
        assert first not in bertram.personal_lines                      # a greeting first ...
        assert bertram.greeting(10) == bertram.personal_lines[0]         # ... then his story, in order
        bertram.greeting(10)
        assert bertram.greeting(10) == bertram.personal_lines[1]
        bertram.greeting(23)
        assert bertram.greeting(23) not in bertram.personal_lines         # (not at that hour)

    def test_the_story_so_far_is_saved(self, world):
        bertram = world.find_npc("Old Bertram")
        bertram.personal_index, bertram.times_greeted = 2, 5
        data = bertram.to_dict()
        bertram.personal_index = bertram.times_greeted = 0
        bertram.load_dict(data)
        assert (bertram.personal_index, bertram.times_greeted) == (2, 5)

    def test_a_topic_is_opened_by_the_one_who_asks_it(self, world):
        import random
        rng = random.Random(5)
        tobin, caddock = world.find_npc("Tobin the Squire"), world.find_npc("Ser Caddock")
        dorran, vance = world.find_npc("Dorran"), world.find_npc("Guard Captain Vance")
        for a, b in ((tobin, caddock), (caddock, tobin)):
            for _ in range(12):
                conv = dialogue.build_conversation(a, b, 14, rng, exchanges=2)
                for k, (who, text) in enumerate(conv.lines):
                    if any(text == o.format(other=dialogue.short_name(caddock)) for o, *_ in dialogue.TOPICS["squire"]):
                        assert who is tobin                                    # the squire asks
        for a, b in ((dorran, vance), (vance, dorran)):
            for _ in range(12):
                conv = dialogue.build_conversation(a, b, 14, rng, exchanges=2)
                for who, text in conv.lines:
                    if any(text == o.format(other=dialogue.short_name(vance)) for o, *_ in dialogue.TOPICS["guard"]):
                        assert who is dorran                                   # asked of the watch, not by it

    def test_outlaws_talk_shop_in_exchanges_of_several_lines(self):
        long_ones = [e for topic in ("smuggler", "bandit", "boss", "knight", "squire") for e in dialogue.TOPICS[topic] if len(e) > 2]
        assert long_ones
        for topic, exchanges in dialogue.TOPICS.items():
            assert all(len(e) >= 2 for e in exchanges), topic

    def test_every_table_holds_three_days_of_meals(self):
        """The pantries are laid again every third night: whoever eats at a table finds food
        there until then (pockets count, for those who carry their meals)."""
        level = get_level("town")
        need = collections.Counter()                       # meals over the three days, each NPC living its round
        for n in level.npcs:
            plans = (n.schedule, *n.rota)
            for day in range(3):
                for t in plans[day % len(plans)]:
                    if t.state is AIState.EATING:
                        need[(t.food_container or f"pockets:{n.name}", n.name, day)] += 1
        have = {c.name: sum(i.quantity for i in c.items if i.is_food) for c in level.containers}
        reset = {}
        for n in level.npcs:
            have[f"pockets:{n.name}"] = sum(i.quantity for i in n.inventory if i.is_food)
            reset[f"pockets:{n.name}"] = n.inventory_reset_days
        totals = collections.Counter()
        for (source, name, day), meals in need.items():
            if day < min(reset.get(source, 3), 3):        # (pockets are refilled on their own cycle)
                totals[source] += meals
        for source, meals in totals.items():
            assert have.get(source, 0) >= meals, (source, meals, have.get(source, 0))

    def test_everybody_keeps_to_their_day_hour_by_hour(self):
        for n in get_level("town").npcs:
            for plan in (n.schedule, *n.rota):                 # (every day of the round)
                hours = sorted((t.start_hour, t.end_hour) for t in plan)
                assert hours[0][0] == 0 and hours[-1][1] == 24 and all(a[1] == b[0] for a, b in zip(hours, hours[1:])), n.name
                assert all(t.start_hour < t.end_hour for t in plan), n.name

    SHOPKEEPERS = ("Shopkeeper Barnaby", "Hilde the Innkeeper", "Gareth the Smith", "Mirelle the Tailor", "Jae the Fence",
                   "Wynn the Baker", "Griselda the Herbwife", "Ludo the Locksmith", "Hrolf the Hermit Smith")

    def test_shopkeepers_keep_the_same_hours_everybody_else_varies_their_days(self, world):
        for name in self.SHOPKEEPERS:
            assert len(world.find_npc(name).day_plans) == 1, name
        varied = [n for n in world.npcs if len(n.day_plans) > 1]
        assert len(varied) >= 18
        assert {"Tamsin", "Fenwick", "Pell", "Lord Aurelian Vane", "Sister Maude", "Old Bertram", "Odo the Farmer"} <= \
            {n.name for n in varied}
        for n in varied:
            plans = n.day_plans
            assert any([t.action_text for t in p] != [t.action_text for t in plans[0]] for p in plans[1:]), n.name

    def test_nobody_is_sent_further_than_the_hour_allows(self, world):
        """Every task of every day can be walked to from the one before it in well under its time
        (three quarters of it at the most): in a town this size the long walks - home from the inn
        at the far end, down to the village - get two hours, or go on another day."""
        for door in world.doors:
            door.is_locked, door.is_open = False, True
        speed = world.settings.npc.move_speed
        per_hour = world.clock.minutes_per_hour / world.clock.time_scale
        for npc in world.npcs:
            for plan in npc.day_plans:
                tasks = sorted(plan, key=lambda t: t.start_hour)
                for prev, task in zip(tasks[-1:] + tasks[:-1], tasks):
                    npc.position = prev.target_pos.copy()
                    path = [prev.target_pos] + world.find_path(npc, task.target_pos)
                    walk = sum(a.distance_to(b) for a, b in zip(path, path[1:])) / speed
                    hours = (task.end_hour - task.start_hour) % 24 or 24
                    assert walk <= 0.75 * hours * per_hour, (npc.name, task.action_text, round(walk, 1))

    def test_the_round_turns_at_midnight_and_survives_a_save(self, game, tmp_path):
        w = game.world
        tamsin = w.find_npc("Tamsin")
        assert tamsin.schedule is tamsin.day_plans[0]
        w.clock.hour, w.clock.minute_accumulator = 23, w.clock.minutes_per_hour - 0.01
        w.update_time(1.0)                                              # midnight: day 1
        assert w.day_count == 1 and tamsin.schedule is tamsin.day_plans[1]
        assert tamsin.task_for(9).action_text == "Tamsin is praying in the chapel."
        path = str(tmp_path / "round.json")
        assert game.save_game(path)
        w.day_count = 0
        tamsin.use_day(0)
        assert game.load_game(path)
        assert game.world.find_npc("Tamsin").schedule is game.world.find_npc("Tamsin").day_plans[1]

    def test_the_news_the_player_hears_is_local(self, game):
        """The log carries the comings and goings of people near the player (or under the same
        roof), not of all sixty souls of the realm at every hour."""
        w = game.world
        teleport(game, 0.0, 0.0)
        near, far = w.find_npc("Shopkeeper Barnaby"), w.find_npc("Guard Wenna")
        near.position, far.position = spot(w, 5.0, 5.0), spot(w, 0.0, 52.0)
        assert w.news_of(near) is not None and w.news_of(far) is None
        far.position = spot(w, 60.0, 0.0)
        assert w.news_of(far) is None
