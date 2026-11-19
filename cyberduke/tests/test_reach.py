"""Arm IK (animation.reach): a hand goes where a gesture needs it."""

import math
import random

import numpy as np
import pytest

from cyberduke.animation import HumanoidPose
from cyberduke.animation.reach import FORE, UPPER, _rx, _ry, _rz, arm_to, hand, shoulder
from cyberduke.animation.street import MOUTH, crouch_reach, eat, smoke


def _fingertips(pose: HumanoidPose, side: str) -> np.ndarray:
    a = pose.arm_l if side == "l" else pose.arm_r
    r = _rx(a[0]) @ _ry(a[1]) @ _rz(a[2])
    return (
        shoulder(side, pose)
        + r @ np.array((0.0, -UPPER, 0.0))
        + r @ _rx(a[3]) @ np.array((0.0, -FORE, 0.0))
    )


@pytest.mark.parametrize("side", ["l", "r"])
def test_the_hand_reaches_any_target_within_arms_length(side):
    rng = random.Random(4)
    for _ in range(300):
        target = np.array((rng.uniform(-0.6, 0.6), rng.uniform(0.9, 2.1), rng.uniform(-0.2, 0.7)))
        if not 0.25 < np.linalg.norm(target - shoulder(side)) < 0.79:
            continue
        pose = HumanoidPose()
        setattr(pose, f"arm_{side}", arm_to(target, side))
        assert np.allclose(_fingertips(pose, side), target, atol=1e-6)
        assert getattr(pose, f"arm_{side}")[3] <= 1e-9, "the elbow only ever bends forwards"


@pytest.mark.parametrize("lean", [25.0, 60.0])
def test_a_bent_body_still_puts_its_hands_where_they_are_asked(lean):
    """With the upper body bent forward at the hips (``HumanoidPose.lean``) the shoulders
    go with it; the IK and ``hand`` both know."""
    from cyberduke.animation.reach import bent

    rng = random.Random(7)
    pose = HumanoidPose(lean=lean, bob=-0.2)
    for _ in range(200):
        target = np.array((rng.uniform(-0.5, 0.5), rng.uniform(0.0, 1.4), rng.uniform(0.0, 0.9)))
        if not 0.25 < np.linalg.norm(target - shoulder("r", pose)) < 0.79:
            continue
        pose.arm_r = arm_to(target, "r", bob=pose.bob, lean=lean)
        pos, fore = hand(pose, "r")
        assert np.allclose(pos - fore * 0.02, target, atol=1e-6)
    # the shoulders are bent forward round the hips
    assert np.allclose(shoulder("l", pose), bent((0.32, 1.58 + pose.bob, 0.0), lean, pose.bob))
    assert shoulder("l", pose)[2] > 0.25 and shoulder("l", pose)[1] < 1.58 + pose.bob


@pytest.mark.parametrize("sitting", [False, True])
def test_the_chopsticks_tips_go_into_the_box_and_up_to_the_mouth(sitting):
    """The tips of the chopsticks - not the hand - go down to the food in the box and up
    to the mouth (the hand a chopstick's length behind them), on a stool as well as
    standing; the sticks point down into the box, and back up at the lips."""
    from cyberduke.animation.street import (
        BITE,
        CHOPSTICK_LENGTH,
        EAT_AT,
        PICK_AT,
        SEAT_BOB,
        food_in,
        sit,
    )  # fmt: skip

    bob = SEAT_BOB if sitting else 0.0
    food = food_in("sushi")

    def at(t):
        pose = eat(t, bob=bob, food=food)
        return sit(pose) if sitting else pose

    mouth = np.array(MOUTH) + (0.0, bob, 0.0)
    for k, piece in enumerate(food):
        picking = at(k * BITE + PICK_AT)
        box, _ = hand(picking, "l")
        where = box + (piece[0], piece[1] - 0.02, piece[2])  # (level things: 2 cm down)
        tips = np.array(picking.aim_r)
        assert np.linalg.norm(tips - where) < 0.01, f"bite {k}: the tips at the piece"
        pos, _ = hand(picking, "r")
        assert abs(np.linalg.norm(tips - pos) - CHOPSTICK_LENGTH) < 0.03
        assert tips[1] < pos[1] - 0.1, "pointing down into the box"
        eating = at(k * BITE + EAT_AT)
        tips = np.array(eating.aim_r)
        pos, _ = hand(eating, "r")
        assert np.linalg.norm(tips - mouth) < 0.01, f"bite {k}: the tips at the mouth"
        assert tips[1] > pos[1] and tips[2] < pos[2], "pointing back up at the lips"


def test_a_cigarette_goes_to_the_lips_pointing_away_from_the_face():
    """On a drag the filter is at the lips and the cigarette sticks out forwards, away
    from the face - not down the forearm into it; it turns with the hand."""
    from cyberduke.render.items import held_transform

    drag = smoke(1.5)
    pos, fore = hand(drag, "r")
    assert np.linalg.norm(pos - np.array(MOUTH)) < 0.07
    (x, y, z), yaw, pitch = held_transform("cigarette", tuple(pos), fore, side="r")
    d = np.array(
        (
            math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)),
            -math.sin(math.radians(pitch)),
            math.cos(math.radians(pitch)) * math.cos(math.radians(yaw)),
        )
    )
    assert d[2] > 0.5, "the lit end out in front of the face"
    assert abs(float(d @ fore)) < 1e-6, "across the fingers, square to the forearm"
    filter_end = np.array((x, y, z)) - d * 0.035
    assert np.linalg.norm(filter_end - np.array(MOUTH)) < 0.06, "the filter at the lips"
    # down by the hip the hand has turned, and the cigarette with it
    low = smoke(4.5 if not _flicks(0) else 5.5)
    pos2, fore2 = hand(low, "r")
    (_, _, _), yaw2, pitch2 = held_transform("cigarette", tuple(pos2), fore2, side="r")
    assert abs(yaw2 - yaw) + abs(pitch2 - pitch) > 10.0


def _flicks(puff):
    from cyberduke.animation.street import flicks_after

    return flicks_after(puff)


def test_the_ash_is_flicked_off_every_second_or_third_puff():
    from cyberduke.animation.street import FLICK, SMOKE_CYCLE, flicks_after

    flicked = [k for k in range(20) if flicks_after(k)]
    gaps = {b - a for a, b in zip(flicked, flicked[1:], strict=False)}
    assert gaps <= {2, 3} and gaps == {2, 3}, flicked
    k = flicked[0]
    middle = k * SMOKE_CYCLE + (FLICK[0] + FLICK[1]) / 2.0
    tilts = [smoke(middle + dt).tilt_r for dt in np.linspace(-0.4, 0.4, 25)]
    assert max(tilts) - min(tilts) > 15.0, "the cigarette rolled and tapped in the fingers"
    assert all(smoke((k - 1) * SMOKE_CYCLE + c).tilt_r == 0.0 for c in np.linspace(0, 5.9, 30)), (
        "not after every puff"
    )


def test_crouching_keeps_the_feet_flat_on_the_floor():
    from cyberduke.animation.skeleton import body_boxes

    for u in (0.0, 0.5, 1.0):
        pose = crouch_reach(u)
        boots = [m for name, m in body_boxes(pose) if name.startswith("boot")]
        corners = np.array(
            [(x, y, z, 1.0) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
        )
        for m in boots:
            ys = (corners @ m.T)[:, 1]
            assert ys.min() == pytest.approx(0.0, abs=0.02), "on the floor"
            assert ys.max() - ys.min() < 0.3, "sole flat, not standing on its toes"
