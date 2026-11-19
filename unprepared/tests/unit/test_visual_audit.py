"""The visual audit (``unprepared.rendering.audit``, ``tools/visual_audit.py``): the static scene
"drawn" into a recorder and searched for what would look wrong on screen - faces fighting in one
plane, things through walls, the ground through a floor, things floating, decals off their walls.

The level is kept clean of all of it (the first test); the rest make sure each search finds what
it is meant to and nothing it is not (a decal drawn with a polygon offset, a face turned into a
wall, a thing meant to hang)."""
from __future__ import annotations

import pytest

from unprepared.config import GameSettings
from unprepared.core.math3d import BoundingBox3D, Vector3D
from unprepared.rendering import audit
from unprepared.world import World, get_level
from unprepared.world.level import Prop


@pytest.fixture(scope="module")
def world():
    return World(get_level("town"), GameSettings(echo_messages_to_stdout=False))


@pytest.fixture(scope="module")
def triangles(world):
    return audit.record_static_scene(world)


def box_triangles(owner, mn, mx, **flags):
    """The twelve triangles of an axis-aligned box, as the recorder would keep them."""
    rec = audit.GeometryRecorder()
    rec.owner = owner
    x1, y1, z1 = mn
    x2, y2, z2 = mx
    faces = [((x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2)), ((x2, y1, z1), (x1, y1, z1), (x1, y2, z1), (x2, y2, z1)),
             ((x1, y2, z1), (x1, y2, z2), (x2, y2, z2), (x2, y2, z1)), ((x1, y1, z1), (x2, y1, z1), (x2, y1, z2), (x1, y1, z2)),
             ((x2, y1, z2), (x2, y1, z1), (x2, y2, z1), (x2, y2, z2)), ((x1, y1, z1), (x1, y1, z2), (x1, y2, z2), (x1, y2, z1))]
    out = []
    for q in faces:
        out += [audit.Triangle(owner, q[0], q[1], q[2], "t", **flags), audit.Triangle(owner, q[0], q[2], q[3], "t", **flags)]
    return out


class TestTheLevelIsClean:
    def test_nothing_fights_pokes_through_floats_or_hangs_off(self, world, triangles):
        findings = audit.audit_world(world, triangles=triangles)
        assert findings == [], "\n" + "\n".join(str(f) for f in findings[:30])

    def test_the_recording_covers_the_whole_static_scene(self, world, triangles):
        owners = {t.owner.split(":")[0] for t in triangles}
        assert {"wall", "prop", "container", "station", "building", "fixture"} <= owners
        chapel = [t for t in triangles if t.owner == "building:chapel_steeple"]
        assert chapel and max(max(t.a[1], t.b[1], t.c[1]) for t in chapel) > 20.0     # (the spire, lifted onto its pad)


class TestTheSearches:
    def test_two_boxes_sharing_a_face_fight(self):
        a = box_triangles("prop:a", (0, 0, 0), (2, 1, 1))
        b = box_triangles("prop:b", (1, 0, 0), (3, 1, 1))                  # the same front, back, top and bottom planes
        found = audit.find_zfighting(a + b)
        assert found and all(f.what == ("prop:a", "prop:b") for f in found)
        assert {f.detail.split()[-1] for f in found} >= {"+z", "-z", "up", "down"}

    def test_faces_a_few_millimetres_apart_fight_too_but_not_a_decal_off_the_wall(self):
        wall = box_triangles("wall", (0, 0, 0), (4, 3, 0.5))
        close = box_triangles("prop:poster", (1, 1, 0.5), (2, 2, 0.502))     # 2 mm proud: as good as flush at a distance
        proud = box_triangles("prop:window", (1, 1, 0.5), (2, 2, 0.52))      # 2 cm: the way decals are hung
        assert any(f.detail.endswith("+z") for f in audit.find_zfighting(wall + close))
        assert audit.find_zfighting(wall + proud) == []

    def test_an_offset_or_blended_decal_is_meant_to_lie_flush(self):
        floor = box_triangles("prop:floor", (0, 0, 0), (4, 0.05, 4))
        stain = box_triangles("prop:stain", (1, 0, 1), (2, 0.05, 2), offset=True)
        sand = box_triangles("overlay", (1, 0, 1), (2, 0.05, 2), blended=True)
        assert audit.find_zfighting(floor + stain + sand) == []

    def test_one_things_own_faces_are_its_own_business_unless_asked(self):
        a = box_triangles("prop:a", (0, 0, 0), (2, 1, 1)) + box_triangles("prop:a", (1, 0, 0), (3, 1, 1))
        assert audit.find_zfighting(a) == []
        assert audit.find_zfighting(a, same_owner=True)

    def test_faces_that_look_into_something_solid_are_never_seen(self, world):
        """Two boxes stood side by side on the same floor share their bottom plane - but under them
        is the ground, and nobody sees a face that looks into it."""
        t = world.terrain
        x, z = 30.0, 100.0
        y = min(t.height(x + dx, z + dz) for dx in (0.0, 0.75, 1.5) for dz in (0.0, 0.5, 1.0)) - 0.05   # (set in the ground)
        a = box_triangles("prop:a", (x, y, z), (x + 1, y + 1, z + 1))
        b = box_triangles("prop:b", (x + 0.5, y, z), (x + 1.5, y + 1, z + 1))
        seen = audit.find_zfighting(a + b, solids=audit.Solids(world))
        assert seen and not any("down" in f.detail for f in seen)
        assert any("down" in f.detail for f in audit.find_zfighting(a + b))             # (without the solids: found)

    def test_a_shelf_through_a_wall_between_two_rooms(self, world):
        wall = next(w for w in world.walls if w.max_point.y - w.min_point.y > 2.0 and world.building_at(
            Vector3D((w.min_point.x + w.max_point.x) / 2, w.min_point.y + 1, (w.min_point.z + w.max_point.z) / 2)) is not None
            and (w.max_point.x - w.min_point.x) < 0.6 and (w.max_point.z - w.min_point.z) > 3.0)
        mn, mx = wall.min_point, wall.max_point
        shelf = Prop("test_shelf", BoundingBox3D(Vector3D(mn.x - 0.4, mn.y + 0.5, mn.z + 1.0), Vector3D(mx.x + 0.4, mn.y + 1.5, mn.z + 2.0)))
        world.props.append(shelf)
        try:
            found = audit.find_through_walls(world)
        finally:
            world.props.remove(shelf)
        assert any(f.what[0] == "prop:test_shelf" for f in found)

    def test_a_crate_in_the_air_floats_and_a_lantern_hangs(self, world):
        t = world.terrain
        x, z = 40.0, 100.0
        y = t.height(x, z)
        crate = Prop("test_crate", BoundingBox3D(Vector3D(x, y + 0.6, z), Vector3D(x + 1, y + 1.6, z + 1)))
        lantern = Prop("test_lantern", BoundingBox3D(Vector3D(x + 3, y + 2.0, z), Vector3D(x + 3.3, y + 2.4, z + 0.3)), kind="lantern")
        world.props += [crate, lantern]
        try:
            found = {f.what[0] for f in audit.find_floating(world)}
        finally:
            world.props.remove(crate)
            world.props.remove(lantern)
        assert "prop:test_crate" in found and "prop:test_lantern" not in found

    def test_a_window_hung_on_nothing(self, world):
        window = Prop("test_window", BoundingBox3D(Vector3D(45.0, 1.2, 100.0), Vector3D(46.4, 2.5, 100.02)), kind="window")
        world.props.append(window)
        try:
            found = audit.find_decals_off_walls(world)
        finally:
            world.props.remove(window)
        assert [f.detail for f in found if f.what[0] == "prop:test_window"] == ["on no wall"]

    def test_the_ground_through_a_floor(self, world):
        from dataclasses import replace
        b = next(b for b in world.level.buildings if b.name == "barnaby_house")
        sunk = replace(b, name="test_sunk", elevation=world.building_elevation(b) - 0.5)
        buildings = world.level.buildings
        object.__setattr__(world.level, "buildings", (*buildings, sunk))
        try:
            found = audit.find_ground_in_floors(world)
        finally:
            object.__setattr__(world.level, "buildings", buildings)
        assert [f.what for f in found] == [("building:test_sunk",)]


class TestTheTool:
    def test_the_command_line_reports_and_exits_clean(self, capsys):
        from tools.visual_audit import main
        assert main(["--checks", "through_wall", "floating", "decal_off_wall", "ground_in_floor"]) == 0
        assert "0 finding(s)" in capsys.readouterr().out
