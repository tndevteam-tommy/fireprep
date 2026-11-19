"""Analytical Two-Bone Inverse Kinematics solver with anatomical limits.

The solver never stretches or compresses the bones: if the requested wrist
position is out of reach the wrist is pulled back along the shoulder->target
direction until the elbow sits at its maximum extension; if it is too close the
wrist is pushed out until the elbow reaches its flexion limit.  The elbow is
always placed on the side of the pole direction (expressed relative to the
shoulder), so it can never flip behind the arm plane.
"""
import numpy as np
from math3d.geometry import cross3
import config


class IKResult:
    __slots__ = ("elbow", "wrist", "elbow_angle_deg", "clamped")

    def __init__(self, elbow, wrist, elbow_angle_deg, clamped):
        self.elbow = elbow
        self.wrist = wrist
        self.elbow_angle_deg = elbow_angle_deg
        self.clamped = clamped


def reach_limits(l1, l2, min_angle_deg=config.ELBOW_MIN_ANGLE_DEG, max_angle_deg=config.ELBOW_MAX_ANGLE_DEG):
    """Shoulder->wrist distance range allowed by the elbow flexion limits."""
    d_min = float(np.sqrt(l1 * l1 + l2 * l2 - 2.0 * l1 * l2 * np.cos(np.radians(min_angle_deg))))
    d_max = float(np.sqrt(l1 * l1 + l2 * l2 - 2.0 * l1 * l2 * np.cos(np.radians(max_angle_deg))))
    return d_min, d_max


def solve_two_bone_ik(shoulder, target_wrist, l1, l2, pole_dir, forward_dir=None,
                      min_angle_deg=config.ELBOW_MIN_ANGLE_DEG, max_angle_deg=config.ELBOW_MAX_ANGLE_DEG):
    """
    Analytically solves a 2-segment limb (arm: shoulder / elbow / wrist; leg: hip / knee / ankle).

    Args:
        shoulder:     root pivot (world)
        target_wrist: desired end position (world)
        l1, l2:       upper / lower segment lengths
        pole_dir:     direction (relative to the root) the middle joint should point toward
        forward_dir:  optional torso forward; the target is kept in front of the chest plane
        min_angle_deg, max_angle_deg: interior joint angle limits (elbow by default; a knee
                      passes KNEE_MIN_ANGLE_DEG / KNEE_MAX_ANGLE_DEG)
    Returns:
        IKResult(elbow, wrist, elbow_angle_deg, clamped)
    """
    shoulder = np.asarray(shoulder, dtype=np.float32)
    target = np.asarray(target_wrist, dtype=np.float32)
    pole = np.asarray(pole_dir, dtype=np.float32)

    D = target - shoulder
    d = float(np.linalg.norm(D))
    clamped = False

    if d < 1e-5:
        D_dir = -pole / (np.linalg.norm(pole) + 1e-7)
        d = 1e-5
    else:
        D_dir = D / d

    # Keep the hand in front of the torso (shoulder can't reach far behind the back)
    if forward_dir is not None:
        fwd = np.asarray(forward_dir, dtype=np.float32)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-7)
        behind = float(np.dot(D_dir, fwd))
        if behind < -0.25:
            D_dir = D_dir - fwd * (behind + 0.25)
            D_dir /= (np.linalg.norm(D_dir) + 1e-7)
            clamped = True

    # Respect elbow flexion limits: bones are rigid, so clamp the reach distance
    d_min, d_max = reach_limits(l1, l2, min_angle_deg, max_angle_deg)
    d_solve = float(np.clip(d, d_min, d_max))
    if abs(d_solve - d) > 1e-5:
        clamped = True

    # Law of Cosines for shoulder interior angle alpha
    cos_alpha = (l1 * l1 + d_solve * d_solve - l2 * l2) / (2.0 * l1 * d_solve)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    sin_alpha = float(np.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha)))

    pole_n = pole / (np.linalg.norm(pole) + 1e-7)

    # Arm bend plane: contains D and the pole direction
    cross = cross3(D_dir, pole_n)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-4:
        fallback = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        cross = cross3(D_dir, fallback)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-4:
            cross = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            cross_norm = 1.0
    n_arm = cross / cross_norm

    # In-plane perpendicular pointing toward the pole side
    u_arm = cross3(n_arm, D_dir)
    u_arm /= (np.linalg.norm(u_arm) + 1e-7)
    if float(np.dot(u_arm, pole_n)) < 0.0:
        u_arm = -u_arm

    elbow = shoulder + l1 * (cos_alpha * D_dir + sin_alpha * u_arm)
    wrist = shoulder + D_dir * d_solve

    cos_beta = (l1 * l1 + l2 * l2 - d_solve * d_solve) / (2.0 * l1 * l2)
    elbow_angle = float(np.degrees(np.arccos(np.clip(cos_beta, -1.0, 1.0))))

    return IKResult(elbow.astype(np.float32), wrist.astype(np.float32), elbow_angle, clamped)
