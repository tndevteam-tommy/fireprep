import pytest

from cyberduke.core.interp import clamp, lerp, smoothstep, wrap_angle_deg


def test_clamp():
    assert clamp(5, 0, 1) == 1
    assert clamp(-5, 0, 1) == 0
    assert clamp(0.5, 0, 1) == 0.5


def test_lerp_endpoints_and_midpoint():
    assert lerp(2.0, 4.0, 0.0) == 2.0
    assert lerp(2.0, 4.0, 1.0) == 4.0
    assert lerp(2.0, 4.0, 0.5) == 3.0


def test_smoothstep_is_clamped_and_symmetric():
    assert smoothstep(0, 1, -1) == 0.0
    assert smoothstep(0, 1, 2) == 1.0
    assert smoothstep(0, 1, 0.5) == pytest.approx(0.5)
    assert smoothstep(0, 1, 0.25) == pytest.approx(1 - smoothstep(0, 1, 0.75))


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0, 0), (180, -180), (-180, -180), (190, -170), (-190, 170), (540, -180)],
)
def test_wrap_angle_deg(angle, expected):
    assert wrap_angle_deg(angle) == pytest.approx(expected)
