"""Putting a hand where it has to be: two-bone arm IK, and where a posed hand is.

Everything is in the body's own frame (feet at the origin, facing +z, the body's left
along +x) - the frame ``render.humanoid`` draws in. An arm hangs from its shoulder at
(+-0.32, 1.58 + bob, 0): upper arm 0.40 to the elbow, forearm 0.40 to the hand. The pose
angles are (pitch about x, yaw about y, roll about z) at the shoulder, applied in that
order, and the elbow's bend about x (negative folds the forearm forwards).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cyberduke.animation.pose import HumanoidPose

UPPER = 0.40
FORE = 0.40
SHOULDER_X = 0.32
SHOULDER_Y = 1.58
HIP_Y = 0.90  # the upper body bends forward about the hips (``HumanoidPose.lean``)
HAND = 0.42  # a held thing sits this far down the forearm from the elbow

Vec = tuple[float, float, float]


def shoulder(side: str, pose: HumanoidPose | None = None, bob: float = 0.0) -> np.ndarray:
    """Where a shoulder joint is (the lean included)."""
    lean = 0.0
    if pose is not None:
        bob = pose.bob
        shrug = pose.shrug_l if side == "l" else pose.shrug_r
        lean = pose.lean
    else:
        shrug = 0.0
    x = SHOULDER_X - shrug if side == "l" else -SHOULDER_X + shrug
    return bent(np.array((x, SHOULDER_Y + bob, 0.0)), lean, bob)


def bent(p: Sequence[float], lean: float, bob: float = 0.0) -> np.ndarray:
    """A point of the upper body, bent forward with it by ``lean`` degrees."""
    p = np.asarray(p, dtype=float)
    if not lean:
        return p
    pivot = np.array((0.0, HIP_Y + bob, 0.0))
    return pivot + _rx(lean) @ (p - pivot)


def unbent(p: Sequence[float], lean: float, bob: float = 0.0) -> np.ndarray:
    """``bent`` undone: a point in the body frame, in the upright upper body's frame."""
    p = np.asarray(p, dtype=float)
    if not lean:
        return p
    pivot = np.array((0.0, HIP_Y + bob, 0.0))
    return pivot + _rx(-lean) @ (p - pivot)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _unit(v: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array(fallback, dtype=float)


def arm_to(
    target: Sequence[float],
    side: str,
    bob: float = 0.0,
    elbow: Sequence[float] | None = None,
    shrug: float = 0.0,
    lean: float = 0.0,
) -> list[float]:
    """Arm angles [pitch, yaw, roll, elbow] that put the hand of the ``side`` ("l"/"r")
    arm at ``target``. ``elbow`` is the direction the elbow should point (default: down
    and a little out and back, the way a relaxed arm bends); out of reach, the arm points
    straight at the target. ``lean``: the upper body is bent forward that far (the
    target is still in the body's own, upright frame)."""
    s = np.array(
        (SHOULDER_X - shrug if side == "l" else -SHOULDER_X + shrug, SHOULDER_Y + bob, 0.0)
    )
    v = unbent(target, lean, bob) - s
    d = float(np.linalg.norm(v))
    d = min(max(d, abs(UPPER - FORE) + 0.12), UPPER + FORE - 1e-4)
    # interior angle at the elbow, and the bend that gives it
    cos_int = (UPPER**2 + FORE**2 - d * d) / (2.0 * UPPER * FORE)
    interior = math.degrees(math.acos(max(-1.0, min(1.0, cos_int))))
    bend = -(180.0 - interior)
    # the hand and the elbow's offset, before the shoulder turns the arm
    h0 = np.array(
        (0.0, -UPPER - FORE * math.cos(math.radians(bend)), -FORE * math.sin(math.radians(bend)))
    )
    h0n = _unit(h0, (0.0, -1.0, 0.0))
    elbow0 = np.array((0.0, -UPPER, 0.0))
    e0 = _unit(elbow0 - h0n * float(elbow0 @ h0n), (0.0, 0.0, -1.0))
    # where they should be
    vn = _unit(v, (0.0, -1.0, 0.0))
    out = 1.0 if side == "l" else -1.0
    hint = np.array(elbow if elbow is not None else (0.45 * out, -1.0, -0.35), dtype=float)
    w = hint - vn * float(hint @ vn)
    w = _unit(w, (out, 0.0, 0.0) if abs(vn[1]) > 0.9 else (0.0, -1.0, 0.0))
    f0 = np.stack((h0n, e0, np.cross(h0n, e0)), axis=1)
    f1 = np.stack((vn, w, np.cross(vn, w)), axis=1)
    r = f1 @ f0.T
    # R = Rx(pitch) Ry(yaw) Rz(roll)
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, float(r[0, 2])))))
    pitch = math.degrees(math.atan2(-float(r[1, 2]), float(r[2, 2])))
    roll = math.degrees(math.atan2(-float(r[0, 1]), float(r[0, 0])))
    return [pitch, yaw, roll, bend]


def hand(pose: HumanoidPose, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(position of the hand, unit direction the forearm points - elbow to hand) of a
    posed arm, in the body frame."""
    angles = pose.arm_l if side == "l" else pose.arm_r
    r = _rx(pose.lean) @ _rx(angles[0]) @ _ry(angles[1]) @ _rz(angles[2])
    elbow_pos = shoulder(side, pose) + r @ np.array((0.0, -UPPER, 0.0))
    fore = r @ _rx(angles[3]) @ np.array((0.0, -1.0, 0.0))
    return elbow_pos + fore * HAND, fore
