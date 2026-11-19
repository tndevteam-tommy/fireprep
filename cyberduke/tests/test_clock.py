from cyberduke.core.clock import clamp_dt


def test_normal_frames_pass_through():
    assert clamp_dt(1 / 60, 0.1) == 1 / 60


def test_hitches_are_capped():
    assert clamp_dt(0.5, 0.1) == 0.1


def test_negative_deltas_are_zeroed():
    assert clamp_dt(-0.02, 0.1) == 0.0
