"""The GLU replacement must produce exactly the matrices gluPerspective/gluLookAt do."""
import math

import pytest

from unprepared.rendering.camera import look_at_matrix, perspective_matrix


def test_perspective_matrix_matches_textbook():
    m = perspective_matrix(60.0, 1024 / 768, 0.1, 100.0)
    f = 1.0 / math.tan(math.radians(60.0) / 2)
    assert m[0] == pytest.approx(f / (1024 / 768))
    assert m[5] == pytest.approx(f)
    assert m[10] == pytest.approx((100.0 + 0.1) / (0.1 - 100.0))
    assert m[11] == -1.0
    assert m[14] == pytest.approx(2 * 100.0 * 0.1 / (0.1 - 100.0))
    assert m[15] == 0.0
    assert all(m[i] == 0.0 for i in (1, 2, 3, 4, 6, 7, 8, 9, 12, 13))


def test_perspective_rejects_degenerate():
    with pytest.raises(ValueError):
        perspective_matrix(60.0, 0.0, 0.1, 100.0)


def test_look_at_identity_when_looking_down_negative_z():
    m = look_at_matrix((0, 0, 0), (0, 0, -1), (0, 1, 0))
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    assert all(a == pytest.approx(b) for a, b in zip(m, identity))


def test_look_at_rotation_is_orthonormal():
    m = look_at_matrix((3, 2, 1), (-4, 0.5, 6), (0, 1, 0))
    cols = [m[0:3], m[4:7], m[8:11]]
    for i in range(3):
        assert sum(c * c for c in cols[i]) == pytest.approx(1.0)
        for j in range(i + 1, 3):
            assert sum(a * b for a, b in zip(cols[i], cols[j])) == pytest.approx(0.0, abs=1e-12)


def test_look_at_maps_forward_to_negative_z():
    eye, center = (1.0, 2.0, 3.0), (4.0, 2.0, 3.0)    # looking +x
    m = look_at_matrix(eye, center, (0, 1, 0))
    fx, fy, fz = 1.0, 0.0, 0.0
    # column-major: transformed = M^T * v for rotation rows
    tx = m[0] * fx + m[4] * fy + m[8] * fz
    ty = m[1] * fx + m[5] * fy + m[9] * fz
    tz = m[2] * fx + m[6] * fy + m[10] * fz
    assert (round(tx, 9), round(ty, 9), round(tz, 9)) == (0.0, 0.0, -1.0)
