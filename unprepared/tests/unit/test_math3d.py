import math

from unprepared.core.math3d import BoundingBox3D, Vector3D, normalized, ray_intersects_box, view_direction, yaw_basis


class TestVector3D:
    def test_distance(self):
        assert Vector3D(0, 0, 0).distance_to(Vector3D(3, 4, 0)) == 5.0
        assert Vector3D(1, 2, 3).distance_to(Vector3D(1, 2, 3)) == 0.0

    def test_dict_roundtrip(self):
        v = Vector3D(1.5, -2.0, 3.25)
        assert Vector3D.from_dict(v.to_dict()) == v
        assert v.to_dict() == {"x": 1.5, "y": -2.0, "z": 3.25}

    def test_copy_is_independent(self):
        v = Vector3D(1, 2, 3)
        c = v.copy()
        c.x = 99
        assert v.x == 1


class TestBoundingBox3D:
    def test_intersects_overlapping(self):
        a = BoundingBox3D(Vector3D(0, 0, 0), Vector3D(2, 2, 2))
        b = BoundingBox3D(Vector3D(1, 1, 1), Vector3D(3, 3, 3))
        assert a.intersects(b) and b.intersects(a)

    def test_intersects_touching_edge_counts(self):
        a = BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1))
        b = BoundingBox3D(Vector3D(1, 0, 0), Vector3D(2, 1, 1))
        assert a.intersects(b)

    def test_disjoint(self):
        a = BoundingBox3D(Vector3D(0, 0, 0), Vector3D(1, 1, 1))
        b = BoundingBox3D(Vector3D(1.1, 0, 0), Vector3D(2, 1, 1))
        assert not a.intersects(b)

    def test_around(self):
        box = BoundingBox3D.around(Vector3D(10, 0, -5), 0.4, 1.8, 0.4)
        assert box.min_point == Vector3D(9.6, 0, -5.4)
        assert box.max_point == Vector3D(10.4, 1.8, -4.6)


class TestRayIntersectsBox:
    wall = BoundingBox3D(Vector3D(5, 0, -10), Vector3D(5.5, 4, 0))

    def test_segment_through_wall(self):
        assert ray_intersects_box(Vector3D(0, 1, -5), Vector3D(10, 1, -5), self.wall)

    def test_segment_stops_short(self):
        assert not ray_intersects_box(Vector3D(0, 1, -5), Vector3D(4, 1, -5), self.wall)

    def test_segment_misses_sideways(self):
        assert not ray_intersects_box(Vector3D(0, 1, 5), Vector3D(10, 1, 5), self.wall)

    def test_segment_above_wall(self):
        assert not ray_intersects_box(Vector3D(0, 6, -5), Vector3D(10, 6, -5), self.wall)

    def test_axis_aligned_zero_direction_component(self):
        # direction has zero x-component and origin x is outside the slab -> miss
        assert not ray_intersects_box(Vector3D(0, 1, -20), Vector3D(0, 1, 20), self.wall)
        # origin inside x-slab with zero x-direction -> hit
        assert ray_intersects_box(Vector3D(5.2, 1, -20), Vector3D(5.2, 1, 20), self.wall)

    def test_origin_inside_box(self):
        assert ray_intersects_box(Vector3D(5.2, 1, -5), Vector3D(20, 1, -5), self.wall)

    def test_matches_baseline_on_random_segments(self, baseline):
        import random
        rng = random.Random(1234)
        L = baseline.models
        for _ in range(2000):
            a = Vector3D(rng.uniform(-30, 30), rng.uniform(-1, 6), rng.uniform(-30, 30))
            b = Vector3D(rng.uniform(-30, 30), rng.uniform(-1, 6), rng.uniform(-30, 30))
            bmin = Vector3D(rng.uniform(-25, 20), 0, rng.uniform(-25, 20))
            box = BoundingBox3D(bmin, Vector3D(bmin.x + rng.uniform(0.5, 10), 4, bmin.z + rng.uniform(0.5, 10)))
            la, lb = L.Vector3D(a.x, a.y, a.z), L.Vector3D(b.x, b.y, b.z)
            lbox = L.BoundingBox3D(L.Vector3D(bmin.x, 0, bmin.z), L.Vector3D(box.max_point.x, 4, box.max_point.z))
            assert ray_intersects_box(a, b, box) == L.ray_intersects_box(la, lb, lbox)


class TestAngles:
    def test_view_direction_default_looks_down_negative_z(self):
        dx, dy, dz = view_direction(0.0, 0.0)
        assert (round(dx, 9), round(dy, 9), round(dz, 9)) == (0.0, 0.0, -1.0)

    def test_view_direction_yaw_90_looks_positive_x(self):
        dx, dy, dz = view_direction(90.0, 0.0)
        assert round(dx, 9) == 1.0 and abs(dz) < 1e-9

    def test_view_direction_pitch_up(self):
        dx, dy, dz = view_direction(0.0, 90.0)
        assert round(dy, 9) == 1.0

    def test_view_direction_is_unit_length(self):
        for yaw in range(0, 360, 37):
            for pitch in (-89, -45, 0, 30, 89):
                assert math.isclose(math.sqrt(sum(c * c for c in view_direction(yaw, pitch))), 1.0)

    def test_yaw_basis_is_orthonormal(self):
        for yaw in (0, 45, 90, 180, 270, 333):
            (fx, fz), (rx, rz) = yaw_basis(yaw)
            assert math.isclose(fx * rx + fz * rz, 0.0, abs_tol=1e-12)
            assert math.isclose(fx * fx + fz * fz, 1.0)

    def test_normalized_zero_vector_unchanged(self):
        assert normalized(0, 0, 0) == (0, 0, 0, 0)

    def test_normalized(self):
        x, y, z, mag = normalized(0, 3, 4)
        assert (x, y, z, mag) == (0, 0.6, 0.8, 5)
