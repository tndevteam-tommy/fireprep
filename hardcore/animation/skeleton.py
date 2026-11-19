"""38-Joint Skeletal Hierarchy and Uniform Matrix Palette Generator."""
import numpy as np
from math3d.geometry import cross3
import config
from math3d.transform import identity, translate
from math3d.quaternion import quat_to_mat4, quat_from_vectors


class BoneID:
    # Left Arm (0..18)
    L_CLAVICLE = 0
    L_SHOULDER = 1
    L_ELBOW = 2
    L_WRIST = 3
    L_PALM = 4
    L_THUMB_PROX = 5
    L_THUMB_DIST = 6
    L_INDEX_PROX = 7
    L_INDEX_INTER = 8
    L_INDEX_DIST = 9
    L_MIDDLE_PROX = 10
    L_MIDDLE_INTER = 11
    L_MIDDLE_DIST = 12
    L_RING_PROX = 13
    L_RING_INTER = 14
    L_RING_DIST = 15
    L_PINKY_PROX = 16
    L_PINKY_INTER = 17
    L_PINKY_DIST = 18

    # Right Arm (19..37)
    R_OFFSET = 19
    ARM_BONES = 38

    # Body & legs (38..45)
    TORSO = 38
    PELVIS = 39
    L_THIGH = 40
    L_SHIN = 41
    L_FOOT = 42
    R_THIGH = 43
    R_SHIN = 44
    R_FOOT = 45
    TOTAL_BONES = 46


class ArmSkeleton:
    def __init__(self):
        self.wrist_angles = [0.0, 0.0]     # hand-to-forearm bend per arm (telemetry)
        self.matrices = np.zeros((BoneID.TOTAL_BONES, 4, 4), dtype=np.float32)
        for i in range(BoneID.TOTAL_BONES):
            self.matrices[i] = identity()

    @staticmethod
    def _orient_segment(p_start, p_end):
        """Builds a transformation matrix starting at p_start pointing along (p_end - p_start)."""
        diff = p_end - p_start
        length = float(np.linalg.norm(diff))
        if length < 1e-5:
            return translate(p_start[0], p_start[1], p_start[2])

        dir_vec = diff / length
        # Default local segment points along +Y
        q = quat_from_vectors(np.array([0.0, 1.0, 0.0], dtype=np.float32), dir_vec)
        rot_mat = quat_to_mat4(q)
        trans_mat = translate(p_start[0], p_start[1], p_start[2])
        return trans_mat @ rot_mat

    @staticmethod
    def _frame(origin, y_axis, z_hint):
        """4x4 transform at `origin` with local +Y along y_axis and local +Z as close as possible to z_hint."""
        y = np.asarray(y_axis, dtype=np.float32)
        y = y / (np.linalg.norm(y) + 1e-7)
        z = np.asarray(z_hint, dtype=np.float32)
        z = z - y * float(np.dot(z, y))
        n = float(np.linalg.norm(z))
        if n < 1e-5:
            z = cross3(y, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            if np.linalg.norm(z) < 1e-5:
                z = cross3(y, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            n = float(np.linalg.norm(z))
        z = z / n
        x = cross3(y, z)
        m = identity()
        m[0:3, 0] = x
        m[0:3, 1] = y
        m[0:3, 2] = z
        m[0:3, 3] = np.asarray(origin, dtype=np.float32)
        return m

    @staticmethod
    def clamp_wrist(forearm_axis, wrist_rot, max_angle_deg=config.WRIST_MAX_ANGLE_DEG):
        """
        Limit how far the hand can bend away from the forearm axis.  The hand's finger
        direction (local +Y) may deviate at most `max_angle_deg` from the forearm
        (flexion / extension / deviation combined); pronation (roll about the forearm)
        is free.  Returns the clamped wrist rotation and the angle in degrees.
        """
        fa = np.asarray(forearm_axis, dtype=np.float32)
        fa = fa / (np.linalg.norm(fa) + 1e-7)
        fingers = wrist_rot[0:3, 1].copy()
        palm = wrist_rot[0:3, 2].copy()
        cos_a = float(np.clip(np.dot(fa, fingers), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_a)))
        if angle <= max_angle_deg:
            return wrist_rot, angle
        # rotate the finger direction back toward the forearm axis in their common plane
        perp = fingers - fa * cos_a
        pn = float(np.linalg.norm(perp))
        if pn < 1e-6:
            # hand pointing straight back along the forearm: bend it over toward the palm side
            perp = palm - fa * float(np.dot(palm, fa))
            pn = float(np.linalg.norm(perp))
            if pn < 1e-6:
                return wrist_rot, angle
        perp /= pn
        a = np.radians(max_angle_deg)
        new_fingers = fa * np.cos(a) + perp * np.sin(a)
        new_palm = palm - new_fingers * float(np.dot(palm, new_fingers))
        if np.linalg.norm(new_palm) < 1e-5:
            new_palm = cross3(new_fingers, wrist_rot[0:3, 0])
        new_palm /= (np.linalg.norm(new_palm) + 1e-7)
        m = identity()
        m[0:3, 1] = new_fingers
        m[0:3, 2] = new_palm
        m[0:3, 0] = cross3(new_fingers, new_palm)
        return m, max_angle_deg

    def solve_arm(self, offset, shoulder_pos, elbow_pos, wrist_pos, wrist_rot, finger_transforms, is_left=False):
        """Solves 19 bones for a single arm hierarchy with anatomically consistent bone frames."""
        forearm_axis = wrist_pos - elbow_pos
        upper_axis = elbow_pos - shoulder_pos
        wrist_rot, self.wrist_angles[1 if offset else 0] = self.clamp_wrist(forearm_axis, wrist_rot)
        palm_dir = wrist_rot[0:3, 2]

        # 0: Clavicle
        clavicle_pos = shoulder_pos + (np.array([-0.05, 0.02, -0.05], dtype=np.float32) if is_left else np.array([0.05, 0.02, -0.05], dtype=np.float32))
        self.matrices[offset + 0] = self._orient_segment(clavicle_pos, shoulder_pos)

        # 1: Upper arm - its roll follows the elbow bend plane so the sleeve reads as one limb
        bend_normal = cross3(upper_axis, forearm_axis)
        if np.linalg.norm(bend_normal) < 1e-5:
            bend_normal = cross3(upper_axis, palm_dir)
        z_upper = cross3(bend_normal, upper_axis)      # in the bend plane, toward the forearm side
        self.matrices[offset + 1] = self._frame(shoulder_pos, upper_axis, z_upper)

        # 2: Forearm - its roll follows the palm (pronation / supination)
        self.matrices[offset + 2] = self._frame(elbow_pos, forearm_axis, palm_dir)

        # 3: Wrist
        wrist_trans = translate(wrist_pos[0], wrist_pos[1], wrist_pos[2])
        wrist_world = wrist_trans @ wrist_rot
        self.matrices[offset + 3] = wrist_world

        # 4: Palm (offset slightly forward from wrist)
        palm_local = translate(0.0, 0.04, 0.0)
        palm_world = wrist_world @ palm_local
        self.matrices[offset + 4] = palm_world

        # Fingers: 14 joints
        # Finger base offsets from palm (Thumb, Index, Middle, Ring, Pinky)
        side = -1.0 if is_left else 1.0
        finger_bases = [
            # Thumb base
            translate(side * 0.035, 0.02, 0.015),
            # Index base
            translate(side * 0.025, 0.075, 0.0),
            # Middle base
            translate(side * 0.008, 0.082, 0.0),
            # Ring base
            translate(side * -0.010, 0.077, 0.0),
            # Pinky base
            translate(side * -0.026, 0.068, 0.0),
        ]

        joint_lens = [
            [0.030, 0.025],          # Thumb (prox, dist)
            [0.035, 0.024, 0.018],   # Index
            [0.038, 0.026, 0.020],   # Middle
            [0.034, 0.023, 0.018],   # Ring
            [0.028, 0.018, 0.015],   # Pinky
        ]

        tf_idx = 0
        bone_idx = 5

        # Thumb
        t_base = palm_world @ finger_bases[0]
        # Proximal
        t_prox = t_base @ finger_transforms[tf_idx]
        self.matrices[offset + bone_idx] = t_prox
        tf_idx += 1
        bone_idx += 1
        # Distal
        t_dist = t_prox @ translate(0.0, joint_lens[0][0], 0.0) @ finger_transforms[tf_idx]
        self.matrices[offset + bone_idx] = t_dist
        tf_idx += 1
        bone_idx += 1

        # 4 Fingers (Index, Middle, Ring, Pinky)
        for f in range(1, 5):
            f_base = palm_world @ finger_bases[f]
            # Proximal
            f_prox = f_base @ finger_transforms[tf_idx]
            self.matrices[offset + bone_idx] = f_prox
            tf_idx += 1
            bone_idx += 1
            # Intermediate
            f_inter = f_prox @ translate(0.0, joint_lens[f][0], 0.0) @ finger_transforms[tf_idx]
            self.matrices[offset + bone_idx] = f_inter
            tf_idx += 1
            bone_idx += 1
            # Distal
            f_dist = f_inter @ translate(0.0, joint_lens[f][1], 0.0) @ finger_transforms[tf_idx]
            self.matrices[offset + bone_idx] = f_dist
            tf_idx += 1
            bone_idx += 1

    def solve(self, left_shoulder, left_elbow, left_wrist, left_wrist_rot, left_fingers,
                    right_shoulder, right_elbow, right_wrist, right_wrist_rot, right_fingers):
        self.solve_arm(0, left_shoulder, left_elbow, left_wrist, left_wrist_rot, left_fingers, is_left=True)
        self.solve_arm(BoneID.R_OFFSET, right_shoulder, right_elbow, right_wrist, right_wrist_rot, right_fingers, is_left=False)

    def set_bone(self, bone, matrix):
        self.matrices[bone] = matrix

    def get_palette(self):
        return self.matrices
