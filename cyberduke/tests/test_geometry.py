"""core.geometry: the single source of truth for boxes, overlaps, and segment tests."""

import math

import pytest

from cyberduke.core.geometry import (
    AABB,
    DEFAULT_OBSTACLE_HEIGHT,
    any_cylinder_overlap,
    line_of_sight_clear,
    segment_clear_2d,
    segment_hits_aabb_2d,
    segment_hits_aabb_3d,
    segment_intersects_aabb_2d,
    segment_intersects_aabb_3d,
    yaw_towards,
)

BOX = AABB(0.0, 2.0, 0.0, 2.0, 0.0, 1.0)


class TestAABB:
    def test_from_tuple_accepts_legacy_4_and_6_layouts(self):
        assert AABB.from_tuple((0, 1, 2, 3)) == AABB(0, 1, 2, 3, 0.0, DEFAULT_OBSTACLE_HEIGHT)
        assert AABB.from_tuple((0, 1, 2, 3, 4, 5)) == AABB(0, 1, 2, 3, 4, 5)
        assert AABB.from_tuple(BOX) is BOX

    def test_from_center(self):
        assert AABB.from_center(1.0, 1.0, 2.0, 2.0, 0.0, 1.0) == BOX

    def test_inflated_grows_footprint_only(self):
        g = BOX.inflated(0.5)
        assert (g.min_x, g.max_x, g.min_z, g.max_z) == (-0.5, 2.5, -0.5, 2.5)
        assert (g.min_y, g.max_y) == (0.0, 1.0)
        assert g.blocks_sight is True

    def test_contains_xz_with_margin(self):
        assert BOX.contains_xz(1.0, 1.0)
        assert not BOX.contains_xz(2.1, 1.0)
        assert BOX.contains_xz(2.1, 1.0, margin=0.2)

    @pytest.mark.parametrize(
        ("feet", "top", "expected"),
        [
            (0.0, 1.8, True),  # standing on the floor, box is 0..1 high
            (1.0, 2.9, False),  # standing exactly on top of the box (epsilon lets it through)
            (1.02, 2.9, False),
            (0.9, 2.0, True),  # feet below top of box
            (-2.0, -0.5, False),  # entirely below
            (-2.0, 0.04, False),  # top just under the epsilon
            (-2.0, 0.06, True),
        ],
    )
    def test_overlaps_vertical_span(self, feet, top, expected):
        assert BOX.overlaps_vertical_span(feet, top) is expected

    def test_overlaps_cylinder_combines_height_and_footprint(self):
        assert BOX.overlaps_cylinder(2.2, 1.0, 0.35, 0.0, 1.8)  # within radius
        assert not BOX.overlaps_cylinder(2.4, 1.0, 0.35, 0.0, 1.8)  # outside radius
        assert not BOX.overlaps_cylinder(1.0, 1.0, 0.35, 2.75, 4.6)  # on the catwalk above

    def test_any_cylinder_overlap(self):
        boxes = [AABB(10, 11, 10, 11), BOX]
        assert any_cylinder_overlap(boxes, 1.0, 1.0, 0.3, 0.0, 1.8)
        assert not any_cylinder_overlap(boxes, 5.0, 5.0, 0.3, 0.0, 1.8)


class TestSegments2D:
    def test_crossing_segment_hits_with_entry_t(self):
        t = segment_hits_aabb_2d(-1.0, 1.0, 3.0, 1.0, BOX)
        assert t == pytest.approx(0.25)

    def test_segment_starting_inside_returns_zero(self):
        assert segment_hits_aabb_2d(1.0, 1.0, 5.0, 1.0, BOX) == 0.0

    def test_miss(self):
        assert segment_hits_aabb_2d(-1.0, 3.0, 3.0, 3.0, BOX) is None
        assert not segment_intersects_aabb_2d(-1.0, 3.0, 3.0, 3.0, BOX)

    def test_segment_ending_before_box_misses(self):
        assert segment_hits_aabb_2d(-3.0, 1.0, -1.0, 1.0, BOX) is None

    def test_axis_aligned_degenerate_directions(self):
        assert segment_intersects_aabb_2d(1.0, -1.0, 1.0, 3.0, BOX)  # dx == 0, crosses
        assert not segment_intersects_aabb_2d(3.0, -1.0, 3.0, 3.0, BOX)  # dx == 0, beside
        assert segment_intersects_aabb_2d(-1.0, 1.0, 3.0, 1.0, BOX)  # dz == 0

    def test_segment_clear_2d_respects_inflation(self):
        # passes 0.2 m beside the box: clear for a point, blocked for a 0.35 m agent
        assert segment_clear_2d(-1.0, 2.2, 3.0, 2.2, [BOX])
        assert not segment_clear_2d(-1.0, 2.2, 3.0, 2.2, [BOX], radius=0.35)


class TestSegments3D:
    def test_ray_over_the_box_misses(self):
        assert not segment_intersects_aabb_3d((-1.0, 1.5, 1.0), (3.0, 1.5, 1.0), BOX)

    def test_ray_through_the_box_hits(self):
        assert segment_intersects_aabb_3d((-1.0, 0.5, 1.0), (3.0, 0.5, 1.0), BOX)
        assert segment_hits_aabb_3d((-1.0, 0.5, 1.0), (3.0, 0.5, 1.0), BOX) == pytest.approx(0.25)

    def test_descending_ray_clips_the_top_edge(self):
        # from (−1, 2) down to (3, 0): crosses y=1 at x=1 which is inside the box footprint
        assert segment_intersects_aabb_3d((-1.0, 2.0, 1.0), (3.0, 0.0, 1.0), BOX)

    def test_line_of_sight_ignores_non_sight_blocking_boxes(self):
        rail = AABB(0.0, 2.0, 0.0, 2.0, 0.0, 3.0, blocks_sight=False)
        wall = AABB(0.0, 2.0, 0.0, 2.0, 0.0, 3.0)
        p1, p2 = (-1.0, 1.5, 1.0), (3.0, 1.5, 1.0)
        assert line_of_sight_clear(p1, p2, [rail])
        assert not line_of_sight_clear(p1, p2, [rail, wall])


def test_yaw_convention_zero_is_plus_z_ninety_is_plus_x():
    assert yaw_towards(0, 0, 0, 1) == pytest.approx(0.0)
    assert yaw_towards(0, 0, 1, 0) == pytest.approx(90.0)
    assert yaw_towards(0, 0, 0, -1) == pytest.approx(180.0)
    assert abs(yaw_towards(0, 0, -1, 0)) == pytest.approx(90.0)
    assert math.copysign(1, yaw_towards(0, 0, -1, 0)) == -1


# ----------------------------------------------------------------------------- viewpoints
def test_forward_vector_conventions():
    from cyberduke.core.geometry import forward_vector

    assert forward_vector(0.0, 0.0, yaw_zero_is_x=True) == pytest.approx((1.0, 0.0, 0.0))
    assert forward_vector(90.0, 0.0, yaw_zero_is_x=True) == pytest.approx((0.0, 0.0, 1.0))
    assert forward_vector(0.0, 0.0, yaw_zero_is_x=False) == pytest.approx((0.0, 0.0, 1.0))
    assert forward_vector(90.0, 0.0, yaw_zero_is_x=False) == pytest.approx((1.0, 0.0, 0.0))
    up = forward_vector(0.0, 90.0, yaw_zero_is_x=True)
    assert up == pytest.approx((0.0, 1.0, 0.0))


def test_viewpoint_angles_and_distances():
    from cyberduke.core.geometry import Viewpoint

    v = Viewpoint(0.0, 1.6, 0.0, 0.0, 0.0, 1.0)
    assert v.angle_to((0.0, 1.6, 5.0)) == pytest.approx(0.0)
    assert v.angle_to((5.0, 1.6, 5.0)) == pytest.approx(45.0)
    assert v.angle_to((0.0, 1.6, -5.0)) == pytest.approx(180.0)
    assert v.angle_to((0.0, 1.6, 0.0)) == 0.0, "a point at the eye is 'dead ahead'"
    assert v.distance_to((3.0, 5.6, 0.0)) == pytest.approx(5.0)
    assert v.target == (0.0, 1.6, 1.0) and v.eye == (0.0, 1.6, 0.0)
