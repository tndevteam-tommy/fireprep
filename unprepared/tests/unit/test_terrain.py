"""The lie of the land: a town on a hillside where every floor is level and nothing floats,
sinks or pokes through anything else.

* every building stands on a levelled pad: the ground is exactly its floor inside its walls and
  for a skirt round them (the drawn mesh too - no grass through the floorboards), and no two
  pads at different levels overlap;
* walls in the open step with the ground (their foot below the lowest ground under them);
* props, containers, doors, lights and people are lifted onto the ground where they stand;
* the millstream runs downhill, under bridges a walker can step onto;
* the town climbs from the gate to the watchtower, with the well in the middle.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from unprepared.config import GameSettings
from unprepared.core.math3d import Vector3D
from unprepared.world import World, get_level
from unprepared.world.level import LevelDefinition
from unprepared.world.levels import wilds
from unprepared.world.placement import SINK, WALL_FOOTING
from unprepared.world.terrain import PAD_SKIRT, FlatDef, HillDef, StreamDef, Terrain, TerrainDef
from unprepared.world.tower import STEP_UP


@pytest.fixture(scope="module")
def w():
    return World(get_level("town"), GameSettings(echo_messages_to_stdout=False))


def _vertices_in(terrain, x1, z1, x2, z2):
    """Heightfield vertices inside a rectangle (what the renderer's mesh is built from)."""
    c1 = int(math.ceil((x1 - terrain.min_x) / terrain.cell))
    c2 = int(math.floor((x2 - terrain.min_x) / terrain.cell))
    r1 = int(math.ceil((z1 - terrain.min_z) / terrain.cell))
    r2 = int(math.floor((z2 - terrain.min_z) / terrain.cell))
    return terrain.grid[c1:c2 + 1, r1:r2 + 1]


class TestPads:
    def test_no_two_pads_fight(self, w):
        assert w.terrain.pad_conflicts() == []

    def test_every_building_floor_is_level(self, w):
        t = w.terrain
        for b in w.level.buildings:
            y = w.building_elevation(b)
            (x1, z1), (x2, z2) = b.min_xz, b.max_xz
            for fx in (0.05, 0.3, 0.5, 0.77, 0.95):
                for fz in (0.05, 0.4, 0.6, 0.95):
                    x, z = x1 + (x2 - x1) * fx, z1 + (z2 - z1) * fz
                    assert t.height(x, z) == pytest.approx(y, abs=1e-9), b.name
            assert w.floor_height_at((x1 + x2) / 2, (z1 + z2) / 2, y) >= y

    def test_the_drawn_ground_never_rises_through_a_floor(self, w):
        """The mesh's vertices over a building and its skirt are exactly the floor's height (the
        floor is drawn a hair above: nothing pokes through, and no vertex beyond the walls can tilt
        a triangle up into the room)."""
        t = w.terrain
        for b in w.level.buildings:
            y = w.building_elevation(b)
            (x1, z1), (x2, z2) = b.min_xz, b.max_xz
            if b.underground:
                continue                          # (a cave's floor is not the drawn ground: see TestCaves)
            verts = _vertices_in(t, x1 - PAD_SKIRT + 0.01, z1 - PAD_SKIRT + 0.01, x2 + PAD_SKIRT - 0.01, z2 + PAD_SKIRT - 0.01)
            assert verts.size > 0
            assert np.allclose(verts, y, atol=1e-6), (b.name, float(verts.min()), float(verts.max()), y)

    def test_pads_share_a_level_when_asked(self, w):
        pads = {p.name: p.height for p in w.terrain.pads}
        assert pads["watchtower"] == pads["guard_house"]
        assert pads["smithy_shed"] == pads["smithy"]
        assert pads["shop"] == pads["hideout"] == pads["square"]        # the houses round the well square
        assert pads["inn"] == pads["bridge"]

    def test_level_order_does_not_matter(self):
        """Pads are blended by how near each is, not one after the other: shuffling them changes
        nothing (it used to let a later pad's margin dip into an earlier pad's floor)."""
        from unprepared.world.level import BuildingDef
        d = TerrainDef(hills=(HillDef(0.0, -30.0, 6.0, 20.0),), noise_amplitude=0.0)
        a = BuildingDef("a", (-6.0, -8.0), (0.0, -2.0))
        b = BuildingDef("b", (3.0, -8.0), (9.0, -2.0))
        t1 = Terrain(d, (-20.0, -30.0, 20.0, 10.0), (a, b))
        t2 = Terrain(d, (-20.0, -30.0, 20.0, 10.0), (b, a))
        assert np.allclose(t1.grid, t2.grid)
        for t in (t1, t2):
            assert t.height(-3.0, -5.0) == pytest.approx(t.pads[[p.name for p in t.pads].index("a")].height)

    def test_a_flat_level_has_no_terrain(self):
        t = Terrain(None, (-10.0, -10.0, 10.0, 10.0))
        assert t.flat and t.height(3.0, 4.0) == 0.0 and t.range_over(-1, -1, 1, 1) == (0.0, 0.0)


class TestThingsStandOnTheGround:
    def test_walls_in_the_open_reach_below_the_ground(self, w):
        t = w.terrain
        for wall in w.walls:
            mn, mx = wall.min_point, wall.max_point
            if t.pad_at((mn.x + mx.x) / 2, (mn.z + mx.z) / 2) is not None:
                continue
            lo, hi = t.range_over(mn.x, mn.z, mx.x, mx.z)
            if mn.y <= lo + 1e-6 or mn.y > lo + 1.0:        # (an arch over a culvert starts well up the wall)
                continue
            if any(abs(o.max_point.y - mn.y) < 1e-6 and o.min_point.x <= mn.x + 1e-6 and o.max_point.x >= mx.x - 1e-6
                   and o.min_point.z <= mn.z + 1e-6 and o.max_point.z >= mx.z - 1e-6 for o in w.walls):
                continue                                     # (the upper piece of a wall cut where another met it)
            pytest.fail(f"wall {mn}->{mx} floats over ground at {lo:.2f}")
        # (and the whole walls, before being cut into non-overlapping pieces, have their footing)
        from unprepared.world.level import Wall
        from unprepared.core.math3d import BoundingBox3D
        rebuild = lambda original, a, b: Wall(BoundingBox3D(a, b), original.texture)   # noqa: E731
        for wdef in w.level.walls:
            if wdef.min_point[1] > 0:
                continue
            for piece in w.placer.wall(wdef.build(), rebuild, wdef.segment):
                mn, mx = piece.min_point, piece.max_point
                if t.pad_at((mn.x + mx.x) / 2, (mn.z + mx.z) / 2) is None:
                    lo, _ = t.range_over(mn.x, mn.z, mx.x, mx.z)
                    assert mn.y == pytest.approx(lo - WALL_FOOTING), (mn, mx)

    def test_walls_on_a_pad_start_on_its_floor(self, w):
        t = w.terrain
        for wall in w.walls:
            mn, mx = wall.min_point, wall.max_point
            pad = t.pad_at((mn.x + mx.x) / 2, (mn.z + mx.z) / 2)
            if pad is not None and mn.y < pad.height + 0.5:
                assert mn.y <= pad.height + 1e-6, (mn, mx, pad.name)      # on the floor, or footed below it

    def test_props_in_the_open_sit_on_their_lowest_ground(self, w):
        t = w.terrain
        for prop in w.props:
            mn, mx = prop.min_point, prop.max_point
            if prop.kind in ("step", "floor", "stain") or t.pad_at((mn.x + mx.x) / 2, (mn.z + mx.z) / 2) is not None:
                continue
            lo, hi = t.range_over(mn.x, mn.z, mx.x, mx.z)
            expected = hi if prop.anchor == "high" else lo - SINK
            local = next(p for p in w.level.props if p.name == prop.name).min_point[1]
            assert mn.y == pytest.approx(expected + local, abs=1e-6), prop.name

    def test_doors_hang_at_the_floor(self, w):
        for door in w.doors:
            h = door.hinge_position
            assert h.y == pytest.approx(w.ground_height(h.x, h.z), abs=1e-6), door.door_id

    def test_people_start_on_the_ground_or_their_floor(self, w):
        for npc in w.npcs:
            p = npc.position
            floor = w.floor_height_at(p.x, p.z, p.y)
            assert p.y == pytest.approx(floor, abs=0.05) or p.y >= w.ground_height(p.x, p.z) - 1e-6, npc.name
            assert p.y >= w.ground_height(p.x, p.z) - 1e-6, npc.name

    def test_the_player_spawns_standing_on_the_square(self, w):
        player = w.spawn_player()
        assert player.position.y == pytest.approx(w.ground_height(player.position.x, player.position.z))
        assert w.area_of(player.position) == "the well"


class TestTheHillside:
    def test_the_town_climbs_from_the_gate_to_the_watchtower(self, w):
        pads = {p.name: p.height for p in w.terrain.pads}
        gate, square, tower = pads["gate"], pads["square"], pads["watchtower"]
        assert gate < square < tower
        town_buildings = [b.name for b in w.level.buildings if w.in_town(Vector3D((b.min_xz[0] + b.max_xz[0]) / 2, 0,
                                                                                  (b.min_xz[1] + b.max_xz[1]) / 2))]
        assert max(town_buildings, key=lambda n: pads[n]) in ("watchtower", "guard_house")
        assert tower - gate > 6.0                                       # a real hill, not a bump

    def test_the_well_is_at_the_heart_of_the_town(self, w):
        well = next(p for p in w.props if p.name == "well")
        cx, cz = (well.min_point.x + well.max_point.x) / 2, (well.min_point.z + well.max_point.z) / 2
        x1, z1, x2, z2 = w.level.town_area
        assert (cx, cz) == (0.0, 0.0)
        assert abs(cx - (x1 + x2) / 2) < 1.0 and abs(cz - (z1 + z2) / 2) < 6.0

    def test_streets_are_walkable_slopes(self, w):
        """Nowhere along the main streets (over the bridge where one crosses the millstream) does
        the way rise or fall more than a step between two paces."""
        for (x1, z1), (x2, z2) in (((-78, 0), (78, 0)), ((0, 8), (0, 52)), ((0, -36), (0, -8)), ((-30, -39), (60, -39))):
            n = int(max(abs(x2 - x1), abs(z2 - z1)) / 0.5)
            y = w.ground_height(x1, z1)
            heights = []
            for i in range(n + 1):
                y = w.floor_height_at(x1 + (x2 - x1) * i / n, z1 + (z2 - z1) * i / n, y)
                heights.append(y)
            assert max(abs(b - a) for a, b in zip(heights, heights[1:])) < STEP_UP


class TestWater:
    def test_streams_run_downhill(self, w):
        for water in w.terrain.water:
            levels = [p[2] for p in water["points"]]
            assert all(b <= a for a, b in zip(levels, levels[1:])), water["name"]

    def test_the_millstream_runs_through_the_town_to_the_lake_by_the_wood(self, w):
        stream = next(s for s in w.terrain.water if s["name"] == "millstream")
        pts = stream["points"]
        x1, z1, x2, z2 = w.level.town_area
        assert any(x1 < x < x2 and z1 < z < z2 for x, z, _ in pts)              # through the town
        lake = w.terrain.lakes[0]
        end = pts[-1]
        assert ((end[0] - lake.center[0]) / lake.radius_x) ** 2 + ((end[1] - lake.center[1]) / lake.radius_z) ** 2 < 1.0
        assert lake.center[0] < wilds.BOUNDS[0] + 120                           # out west, by the Thornwood

    def test_water_lies_below_its_banks(self, w):
        t = w.terrain
        for water in t.water:
            for x, z, level in water["points"][1:-1]:
                assert t.height(x, z) < level                                  # the bed is under water
                half = water["width"] / 2 + water["bank"] + 0.5
                assert t.natural_height(x + half, z) > level - 0.05 or t.natural_height(x - half, z) > level - 0.05

    def test_the_streams_ribbon_stops_at_the_lakes_rim(self, w):
        # two blended sheets of water over the same ground showed as a darker rectangle in the lake
        from unprepared.rendering.terrain import LAKE_MOUTH_OVERLAP, _clipped_to_lakes, _in_lake
        t = w.terrain
        for stream in t.water:
            clipped = _clipped_to_lakes(stream, t.lakes)["points"]
            assert sum(_in_lake(t.lakes, x, z) for x, z, _ in clipped) <= 1
            assert _in_lake(t.lakes, *stream["points"][-1][:2]) == _in_lake(t.lakes, *clipped[-1][:2])
            if _in_lake(t.lakes, *clipped[-1][:2]):
                (x1, z1, _), (x2, z2, _) = clipped[-2], clipped[-1]
                back = LAKE_MOUTH_OVERLAP + 0.05
                d = math.hypot(x2 - x1, z2 - z1)
                assert not _in_lake(t.lakes, x2 - (x2 - x1) / d * back, z2 - (z2 - z1) / d * back)

    def test_bridges_can_be_stepped_onto_from_both_banks(self, w):
        for bridge in (p for p in w.props if p.kind == "bridge"):
            top = bridge.max_point.y
            for x in (bridge.min_point.x - 0.3, bridge.max_point.x + 0.3):
                z = (bridge.min_point.z + bridge.max_point.z) / 2
                assert top - w.ground_height(x, z) <= STEP_UP, bridge.name
            assert w.floor_height_at((bridge.min_point.x + bridge.max_point.x) / 2, (bridge.min_point.z + bridge.max_point.z) / 2,
                                     top) == pytest.approx(top)

    def test_nobody_wades_into_the_lake(self, w):
        lake = w.terrain.lakes[0]
        assert w.is_deep_water(*lake.center)
        assert not w.is_deep_water(0.0, 0.0)

    def test_a_stream_through_a_flat_keeps_its_channel(self):
        d = TerrainDef(noise_amplitude=0.0, flats=(FlatDef((-5.0, -5.0), (5.0, 5.0), name="square"),),
                       streams=(StreamDef("brook", ((0.0, -15.0), (0.0, 15.0))),))
        t = Terrain(d, (-20.0, -20.0, 20.0, 20.0))
        assert t.pad_at(0.0, 0.0) is None and t.height(0.0, 0.0) < 0.0          # the channel, not the square's level
        assert t.pad_at(4.5, 4.5) is not None and t.height(4.5, 4.5) == 0.0


class TestTerrainSharing:
    def test_worlds_of_one_level_share_their_terrain(self):
        a = World(get_level("town"), GameSettings(echo_messages_to_stdout=False))
        b = World(get_level("town"), GameSettings(echo_messages_to_stdout=False))
        assert a.terrain is b.terrain

    def test_a_level_without_terrain_is_flat(self):
        level = get_level("town")
        flat = LevelDefinition(**{**level.__dict__, "terrain": None})
        w = World(flat, GameSettings(echo_messages_to_stdout=False))
        assert w.terrain.flat and w.ground_height(10.0, 10.0) == 0.0
