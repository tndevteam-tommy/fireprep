"""Procedural hand and finger posing solver.

Finger poses are angle sets (MCP / PIP / DIP flexion in degrees plus a thumb
rotation) that can be blended continuously, so a hand can open while it hops
along a rail and close again as it lands, or ease from a hook into a palm press
while the body rises over an edge.

Grip vocabulary (real climbing):
  CRIMP   - thin edge / crevasse slot: knuckles up, PIP sharply bent, DIP slightly
            hyper-extended ("half crimp").  Fingertips in the slot, wrist nearly straight.
  HOOK    - deep ledge (a wall top, a sill, a balcony edge): fingers draped over the
            top surface and curled down behind it, an open-hand grip.
  PRESS   - palm flat on a surface, fingers extended and spread (mantle, lowering,
            pushing off a wall).
  OPEN    - hand open and slightly cupped, ready to close on a hold (hops, reaching).
  REST    - relaxed, lightly curled.
"""
import numpy as np
from math3d.transform import rotate_x, rotate_y, rotate_z


class HandPoseType:
    REST_SWAY = 0
    PRE_REACH = 1
    CRIMP = 2
    MANTLE_PALM = 3
    DYNO_PUSH = 4
    HOOK = 5
    OPEN = 6


# (mcp, pip, dip, thumb_x, thumb_y, thumb_z) - thumb angles are mirrored for the left hand
POSES = {
    HandPoseType.CRIMP:       (48.0, 88.0, -12.0, -15.0, 25.0, 30.0),
    HandPoseType.HOOK:        (62.0, 42.0, 18.0, -12.0, 30.0, 22.0),
    HandPoseType.MANTLE_PALM: (5.0, 5.0, 0.0, -10.0, 40.0, 10.0),
    HandPoseType.PRE_REACH:   (-12.0, 18.0, 10.0, -20.0, 35.0, -10.0),
    HandPoseType.OPEN:        (8.0, 22.0, 8.0, -16.0, 32.0, -4.0),
    HandPoseType.DYNO_PUSH:   (-10.0, 5.0, 0.0, -5.0, 45.0, 0.0),
    HandPoseType.REST_SWAY:   (16.0, 24.0, 10.0, 15.0, 25.0, 15.0),
}

SPREAD = [-6.0, -1.0, 4.0, 9.0]


def pose_angles(pose_type, curl_factor=0.0):
    a = list(POSES.get(pose_type, POSES[HandPoseType.REST_SWAY]))
    if pose_type == HandPoseType.REST_SWAY and curl_factor:
        # relaxed hanging hand -> loose running fist
        a[0] += curl_factor * 34.0
        a[1] += curl_factor * 46.0
        a[2] += curl_factor * 20.0
    return tuple(a)


def blend_angles(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class FingerSolver:
    _cache = {}

    @classmethod
    def get_finger_joint_transforms(cls, pose_type, is_left=False, curl_factor=0.0):
        return cls.transforms_for(pose_angles(pose_type, curl_factor), is_left)

    @classmethod
    def transforms_for(cls, angles, is_left=False):
        """14 local joint transforms for an angle set; cached on angles rounded to 0.5 degrees."""
        key = (tuple(round(float(v) * 2.0) / 2.0 for v in angles), bool(is_left))
        hit = cls._cache.get(key)
        if hit is None:
            hit = cls._compute(key[0], is_left)
            if len(cls._cache) > 4096:
                cls._cache.clear()
            cls._cache[key] = hit
        return hit

    @staticmethod
    def _compute(angles, is_left=False):
        """
        Returns local 4x4 transforms for:
        Thumb (prox, dist) - 2 joints
        Index (prox, inter, dist) - 3 joints
        Middle (prox, inter, dist) - 3 joints
        Ring (prox, inter, dist) - 3 joints
        Pinky (prox, inter, dist) - 3 joints
        Total = 14 joint relative transforms.
        """
        side_mult = -1.0 if is_left else 1.0
        mcp_flex, pip_flex, dip_flex, tx, ty, tz = angles
        thumb_rot = (tx, ty * side_mult, tz * side_mult)

        transforms = []

        # 1. Thumb (2 joints: prox, dist)
        t_prox = rotate_y(np.radians(thumb_rot[1])) @ rotate_x(np.radians(thumb_rot[0])) @ rotate_z(np.radians(thumb_rot[2]))
        t_dist = rotate_x(np.radians(20.0))
        transforms.extend([t_prox, t_dist])

        # 4 Fingers: Index, Middle, Ring, Pinky (3 joints each)
        spread_angles = [-a for a in SPREAD] if is_left else list(SPREAD)
        for i in range(4):
            spread = spread_angles[i]
            m_prox = rotate_y(np.radians(spread)) @ rotate_x(np.radians(mcp_flex))
            m_inter = rotate_x(np.radians(pip_flex))
            m_dist = rotate_x(np.radians(dip_flex))
            transforms.extend([m_prox, m_inter, m_dist])

        return transforms
