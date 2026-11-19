"""Procedural First-Person Skeletal Arm Rig and Mesh Generator."""
import numpy as np
from math3d.geometry import cross3
import config
from math3d.transform import identity
from animation.skeleton import ArmSkeleton, BoneID
from animation.two_bone_ik import solve_two_bone_ik
from animation.finger_solver import FingerSolver, HandPoseType, pose_angles, blend_angles
from animation.hand_planner import HandPlanner, wrist_on_rail, hand_right_of, keep_clear_of_shoulder
from animation.grip import HandSpec, GripKind, grip_kind, hold_pose_for, wrist_offset, press_offset, resolve_hand, hang_extension
from animation.finger_solver import POSES
from parkour.state_machine import ParkourState
from animation.leg_rig import FirstPersonLegRig


# Vertex layout: position(3) normal(3) uv(2) bone_ids(4) weights(4).  uv.x carries the material
# (0 = by bone: sleeve / skin, 1 = skin, 2 = black cloth, 3 = brown boot leather).
MAT_BY_BONE, MAT_SKIN, MAT_CLOTH, MAT_BOOT, MAT_SHIRT = 0.0, 1.0, 2.0, 3.0, 4.0


def _vertex(p, n, bone_id, material=MAT_BY_BONE):
    return [p[0], p[1], p[2], n[0], n[1], n[2], material, 0.0, float(bone_id), 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _make_prism(bone_id, r0, r1, length, sides=8, y_offset=0.0, cap_start=True, cap_end=True,
                material=MAT_BY_BONE, centre=(0.0, 0.0), centre1=None):
    """
    Tapered prism along local +Y with an elliptical cross-section (rx, rz) that goes
    from r0 at the base to r1 at the tip.  `centre` (x, z) offsets the base ring and
    `centre1` the tip ring (default: same).  Flat-shaded (vertices duplicated per face).
    Returns (vertices (N,16), indices).
    """
    verts = []
    indices = []
    ring0 = []
    ring1 = []
    c0 = centre
    c1 = centre if centre1 is None else centre1
    for i in range(sides):
        a = 2.0 * np.pi * (i + 0.5) / sides
        c, s_ = np.cos(a), np.sin(a)
        ring0.append(np.array([c0[0] + r0[0] * c, y_offset, c0[1] + r0[1] * s_], dtype=np.float64))
        ring1.append(np.array([c1[0] + r1[0] * c, y_offset + length, c1[1] + r1[1] * s_], dtype=np.float64))
    for i in range(sides):
        j = (i + 1) % sides
        p0, p1, p2, p3 = ring0[i], ring0[j], ring1[j], ring1[i]
        # outward face normal (counter-clockwise seen from outside, so back-face culling keeps
        # the outer wall and hides the joint caps inside the limb)
        n = cross3(p3 - p0, p1 - p0)
        n = n / max(np.linalg.norm(n), 1e-12)
        base = len(verts)
        for p in (p0, p1, p2, p3):
            verts.append(_vertex(p, n, bone_id, material))
        indices.extend([base, base + 2, base + 1, base, base + 3, base + 2])
    for ring, ny, flip in ((ring0, -1.0, True), (ring1, 1.0, False)):
        if (ring is ring0 and not cap_start) or (ring is ring1 and not cap_end):
            continue
        cen = np.mean(ring, axis=0)
        base = len(verts)
        verts.append(_vertex(cen, (0.0, ny, 0.0), bone_id, material))
        for p in ring:
            verts.append(_vertex(p, (0.0, ny, 0.0), bone_id, material))
        for i in range(sides):
            j = (i + 1) % sides
            tri = [base, base + 1 + i, base + 1 + j] if flip else [base, base + 1 + j, base + 1 + i]
            indices.extend(tri)
    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _make_sphere(bone_id, radius, rings=4, segs=8, squash=(1.0, 1.0, 1.0), material=MAT_BY_BONE, centre=(0.0, 0.0, 0.0)):
    """Low-poly flat-shaded ellipsoid at `centre` in bone space (joint caps)."""
    verts = []
    indices = []
    pts = []
    c = np.asarray(centre, dtype=np.float64)
    for r in range(rings + 1):
        phi = np.pi * r / rings
        row = []
        for sgm in range(segs):
            th = 2.0 * np.pi * sgm / segs
            p = np.array([radius * squash[0] * np.sin(phi) * np.cos(th),
                          radius * squash[1] * np.cos(phi),
                          radius * squash[2] * np.sin(phi) * np.sin(th)], dtype=np.float64)
            row.append(p)
        pts.append(row)
    for r in range(rings):
        for sgm in range(segs):
            j = (sgm + 1) % segs
            p0, p1, p2, p3 = pts[r][sgm], pts[r][j], pts[r + 1][j], pts[r + 1][sgm]
            n = (p0 + p1 + p2 + p3) * 0.25
            n = n / max(np.linalg.norm(n), 1e-12)
            base = len(verts)
            for p in (p0, p1, p2, p3):
                verts.append(_vertex(p + c, n, bone_id, material))
            # (the pole rows collapse to a point: one triangle per quad there)
            if r > 0:
                indices.extend([base, base + 1, base + 2])
            if r < rings - 1:
                indices.extend([base, base + 2, base + 3])
    return np.array(verts, dtype=np.float32), np.array(indices, dtype=np.uint32)


def _make_tapered_box(bone_id, w0, d0, w1, d1, length, y_offset=0.0, material=MAT_BY_BONE, centre=(0.0, 0.0), centre1=None):
    """Box along +Y whose (width, depth) changes linearly from the base to the tip (palms, boots)."""
    return _make_prism(bone_id, (w0 * 0.5, d0 * 0.5), (w1 * 0.5, d1 * 0.5), length, sides=4, y_offset=y_offset,
                       material=material, centre=centre, centre1=centre1)


class FirstPersonArmRig:
    def __init__(self):
        self.zombie_weight = 0.0        # zombie tag: arms reaching forward (blended in / out)
        self.skeleton = ArmSkeleton()
        self.sway_timer = 0.0
        self.gait_phase = 0.0
        self.gait_weight = 0.0
        self.gait_curl = 0.0
        self.arms_up = 0.0          # 0 = hanging at the sides, 1 = up in the running pump
        self.air_weight = 0.0       # airborne balance pose blend
        self.shoulder_lag = np.zeros(3, dtype=np.float32)
        self.pose_type = HandPoseType.REST_SWAY
        self.last = {}
        self.planner = HandPlanner()
        self._was_on_rail = False
        self.torso_yaw = None
        self._smooth = None
        self._release_from = None
        self._reach_pos = [None, None]
        self._last_ext = [None, None]      # wrist extension each hand was shown with last frame
        self._last_clear = [0.0, 0.0]      # wall-clearance flex each open hand was shown with last frame
        self._last_open_ext = [None, None]  # designed extension of a freed (sliding) hand last frame
        self._prev_shoulders = None
        self._hop_src_clear = [0.0, 0.0]
        self._hop_src_ext = [None, None]

        # Procedurally build the complete 3D skinned mesh for both arms, then the body and legs
        # (one skinned draw; the legs write into the same bone palette)
        self.legs = FirstPersonLegRig(self.skeleton)
        arm_v, arm_i = self._build_arm_mesh()
        self.vertices = np.concatenate([arm_v, self.legs.vertices], axis=0)
        self.indices = np.concatenate([arm_i, self.legs.indices + len(arm_v)], axis=0).astype(np.uint32)

    def _build_arm_mesh(self):
        """
        Procedural arm: tapered 8-sided limbs with joint caps, a palm that widens toward
        the knuckles and tapered fingers - no loose rectangles at the joints.
        """
        all_verts = []
        all_indices = []

        def add(v, i):
            idx_offset = len(all_verts)
            all_verts.extend(v)
            all_indices.extend(i + idx_offset)

        for side_offset in (0, BoneID.R_OFFSET):
            # Upper arm: shoulder cap, tapering toward the elbow (sleeve)
            add(*_make_sphere(side_offset + 1, 0.054, rings=4, segs=8))
            add(*_make_prism(side_offset + 1, (0.048, 0.046), (0.040, 0.038), config.UPPER_ARM_LEN, sides=8, cap_end=False))
            # Elbow cap + forearm: round at the elbow, flattening and narrowing toward the wrist
            add(*_make_sphere(side_offset + 2, 0.047, rings=4, segs=8))
            add(*_make_prism(side_offset + 2, (0.040, 0.037), (0.030, 0.021), config.FOREARM_LEN, sides=8, cap_start=False))
            # Wrist joint (ellipsoid, wider than thick) sits on the wrist bone so it follows the hand
            add(*_make_sphere(side_offset + 3, 0.034, rings=3, segs=8, squash=(1.0, 0.8, 0.72)))
            # Palm: widens from the wrist to the knuckles, slightly thicker at the heel
            add(*_make_tapered_box(side_offset + 4, 0.060, 0.030, 0.084, 0.024, 0.082, y_offset=-0.006))

            # Thumb (2 joints)
            add(*_make_prism(side_offset + 5, (0.012, 0.011), (0.010, 0.009), 0.030, sides=6))
            add(*_make_prism(side_offset + 6, (0.010, 0.009), (0.008, 0.007), 0.025, sides=6))
            # Index / Middle / Ring / Pinky (3 joints each)
            finger_dims = [
                (0.0095, 0.0085, 0.0075, [0.035, 0.024, 0.018]),
                (0.0100, 0.0090, 0.0078, [0.038, 0.026, 0.020]),
                (0.0092, 0.0082, 0.0072, [0.034, 0.023, 0.018]),
                (0.0080, 0.0070, 0.0062, [0.028, 0.018, 0.015]),
            ]
            bone = side_offset + 7
            for r_a, r_b, r_c, lens in finger_dims:
                add(*_make_prism(bone, (r_a, r_a * 0.9), (r_b, r_b * 0.9), lens[0], sides=6))
                add(*_make_prism(bone + 1, (r_b, r_b * 0.9), (r_c, r_c * 0.9), lens[1], sides=6))
                add(*_make_prism(bone + 2, (r_c, r_c * 0.9), (r_c * 0.8, r_c * 0.7), lens[2], sides=6))
                bone += 3

        v_arr = np.array(all_verts, dtype=np.float32)
        i_arr = np.array(all_indices, dtype=np.uint32)
        return v_arr, i_arr

    # ------------------------------------------------------------------ frames
    @staticmethod
    def _hand_frame(finger_dir, palm_dir):
        """
        Build a wrist rotation (columns = world axes of the hand's local frame).
        Local +Y = finger direction, local +Z = palm normal (direction the fingers
        curl toward), local +X = Y x Z.
        """
        y = np.asarray(finger_dir, dtype=np.float32)
        y = y / (np.linalg.norm(y) + 1e-7)
        z = np.asarray(palm_dir, dtype=np.float32)
        z = z - y * float(np.dot(z, y))
        zn = float(np.linalg.norm(z))
        if zn < 1e-5:
            # Degenerate: choose any perpendicular
            z = cross3(y, np.array([1.0, 0.0, 0.0], dtype=np.float32))
            if np.linalg.norm(z) < 1e-5:
                z = cross3(y, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            zn = float(np.linalg.norm(z))
        z = z / zn
        x = cross3(y, z)
        m = identity()
        m[0:3, 0] = x
        m[0:3, 1] = y
        m[0:3, 2] = z
        return m

    @staticmethod
    def _rail_hand_right(rail):
        """Unit vector along the rail pointing to the hanging player's right."""
        r = cross3(-rail.normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        n = float(np.linalg.norm(r))
        if n < 1e-5:
            return rail.tangent.copy()
        return r / n

    def _grip_hands(self, grip, rail, lift=0.0):
        """Wrist positions for two hands crimping a rail at `grip`."""
        hr = self._rail_hand_right(rail)
        half = config.HAND_SPACING * 0.5
        # Wrist sits just outside and a little below the lip; fingers hook over it
        base = grip + rail.normal * (config.HAND_GRIP_LEN * 0.6) - np.array([0.0, 0.03 - lift, 0.0], dtype=np.float32)
        return base - hr * half, base + hr * half

    @staticmethod
    def _keep_elbow_off_wall(ik, shoulder, rail, pole_dir, radius=0.05):
        """Rotate the elbow around the shoulder->wrist axis until it is `radius` clear of the wall plane."""
        n = rail.normal
        plane = float(np.dot(rail.p1, n))
        depth = float(np.dot(ik.elbow, n)) - plane
        if depth >= radius:
            return
        axis = ik.wrist - shoulder
        ln = float(np.linalg.norm(axis))
        if ln < 1e-5:
            return
        axis /= ln
        rel = ik.elbow - shoulder
        along = axis * float(np.dot(rel, axis))
        radial = rel - along
        r = float(np.linalg.norm(radial))
        if r < 1e-5:
            return
        u = radial / r
        v = cross3(axis, u)
        pole = pole_dir / (np.linalg.norm(pole_dir) + 1e-7)
        # Smallest rotation about the shoulder-wrist axis that clears the wall (continuous in the
        # penetration depth); of the two directions prefer the one heading toward the pole.
        best = None
        best_key = None
        for sign in (1.0, -1.0):
            for deg in range(2, 181, 2):
                a = np.radians(deg * sign)
                cand = shoulder + along + (u * np.cos(a) + v * np.sin(a)) * r
                if float(np.dot(cand, n)) - plane >= radius:
                    toward_pole = float(np.dot(cand - ik.elbow, pole))
                    key = (deg, -toward_pole)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = cand
                    break
        if best is not None:
            ik.elbow = best.astype(np.float32)

    def _reach_clamped(self, shoulder, target, fraction=0.97):
        """Point along shoulder->target no farther than the arm can physically reach."""
        d = target - shoulder
        dist = float(np.linalg.norm(d))
        max_d = (config.UPPER_ARM_LEN + config.FOREARM_LEN) * fraction
        if dist > max_d:
            return shoulder + d * (max_d / dist)
        return target

    def update(self, dt, player_pos, cam_pos, cam_yaw, cam_pitch, fsm_state, active_rail, targeted_rail,
               velocity, on_ground, grip_point=None, climb_phase=0.0, climb_next=None, grab_candidate=None,
               shimmy_dir=0.0, corner_link=None, corner_entry_s=0.0, mantle_reverse=False, climb_direction=1,
               supported_hang=False, spin=None, zombie=False, move_rate=1.0, move_time_left=None):
        """
        Calculates realistic wrist targets, solves analytical 2-bone IK for elbows,
        applies contextual finger curling, poses the legs to match, and updates the 46-bone palette.
        move_rate: how many times faster than nominal the body moves (ParkourStateMachine.move_rate):
        hand hops, wrist opening and reaching keep time with it.
        move_time_left: seconds until the climb in progress ends (ParkourStateMachine.move_time_left):
        a hand-over to the next hold is timed to land no later than that.
        """
        rate = float(np.clip(move_rate, 0.5, 2.5))
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        yaw_rad = np.radians(cam_yaw)
        pitch_rad = np.radians(cam_pitch)

        # Camera orientation frame (look direction)
        fwd = np.array([
            -np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
            -np.cos(yaw_rad) * np.cos(pitch_rad)
        ], dtype=np.float32)
        fwd /= (np.linalg.norm(fwd) + 1e-7)

        # Torso frame: yaw only - shoulders do not pitch with the head, and they trail quick
        # turns slightly (the torso has inertia the head does not).  While on a rail the torso
        # squares up to the wall no matter where the head is looking.
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        on_rail = fsm_state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY, ParkourState.CLIMB_UP,
                                ParkourState.MANTLE_VAULT, ParkourState.CORNER_TURN) and active_rail is not None
        if on_rail:
            fwd_h = -active_rail.normal.astype(np.float32)
            if fsm_state == ParkourState.CORNER_TURN and corner_link is not None:
                # The torso swings from facing this wall to facing the next one over the turn
                fwd_h = fwd_h * (1.0 - climb_phase) - corner_link.rail.normal * climb_phase
                fwd_h /= (np.linalg.norm(fwd_h) + 1e-7)
            right = cross3(fwd_h, world_up)
            self.torso_yaw = float(np.degrees(np.arctan2(-fwd_h[0], -fwd_h[2])))
        else:
            if self.torso_yaw is None:
                self.torso_yaw = float(cam_yaw)
            delta = (float(cam_yaw) - self.torso_yaw + 180.0) % 360.0 - 180.0
            self.torso_yaw += delta * min(1.0, config.TORSO_YAW_FOLLOW * dt)
            ty = np.radians(self.torso_yaw)
            fwd_h = np.array([-np.sin(ty), 0.0, -np.cos(ty)], dtype=np.float32)
            right = np.array([np.cos(ty), 0.0, -np.sin(ty)], dtype=np.float32)
        right /= (np.linalg.norm(right) + 1e-7)
        cam_up = cross3(right, fwd)
        cam_up /= (np.linalg.norm(cam_up) + 1e-7)

        # The camera sits CAMERA_FORWARD ahead of the body in the *head's* facing; the shoulders
        # belong to the torso, so turning the head swings the eyes, not the shoulders.
        head_fwd_h = np.array([-np.sin(yaw_rad), 0.0, -np.cos(yaw_rad)], dtype=np.float32)
        body_eye = cam_pos - head_fwd_h * config.CAMERA_FORWARD
        l_offset = config.SHOULDER_OFFSET_L
        r_offset = config.SHOULDER_OFFSET_R
        left_shoulder = body_eye + right * l_offset[0] + world_up * l_offset[1] + fwd_h * config.SHOULDER_FORWARD
        right_shoulder = body_eye + right * r_offset[0] + world_up * r_offset[1] + fwd_h * config.SHOULDER_FORWARD

        speed = float(np.linalg.norm([velocity[0], velocity[2]]))
        pose_type = HandPoseType.REST_SWAY

        # ---- Locomotion ----------------------------------------------------------------
        # Arms are heavy: standing still they hang straight down beside the thighs with a slight
        # bend at the elbow and do not move at all.  Walking they swing from the shoulder, in
        # opposition to the legs (one cycle per stride).  Only once the player is actually
        # running do they come up into a pump, harder and higher the faster the run; in the air
        # they come up and out a little for balance.
        moving_on_ground = on_ground and speed > 0.5 and fsm_state in (ParkourState.GROUND, ParkourState.SPRINT)
        target_weight = 1.0 if moving_on_ground else 0.0
        self.gait_weight += (target_weight - self.gait_weight) * min(1.0, config.GAIT_BLEND_RATE * dt)
        if moving_on_ground:
            steps_per_s = config.GAIT_BASE_CADENCE + config.GAIT_CADENCE_PER_MS * speed
            self.gait_phase += 2.0 * np.pi * (steps_per_s * 0.5) * dt
        else:
            # settle toward a neutral phase so the arms come to rest rather than freezing mid-swing
            target = np.round(self.gait_phase / np.pi) * np.pi
            self.gait_phase += (target - self.gait_phase) * min(1.0, 6.0 * dt)
        ph = self.gait_phase
        w = self.gait_weight
        # how far the arms have come up from hanging into the running pump (by actual speed)
        up_target = float(np.clip((speed - config.ARMS_UP_SPEED_LOW) / (config.ARMS_UP_SPEED_HIGH - config.ARMS_UP_SPEED_LOW), 0.0, 1.0))
        if fsm_state == ParkourState.SPRINT:
            up_target = 1.0                  # running (even through a hard cut that costs speed)
        if not on_ground:
            up_target = self.arms_up          # a leap keeps whatever the arms were doing
        self.arms_up += (up_target - self.arms_up) * min(1.0, 6.0 * dt)
        arms_up = self.arms_up * self.arms_up * (3.0 - 2.0 * self.arms_up)
        air_target = 0.0 if on_ground else 1.0
        self.air_weight += (air_target - self.air_weight) * min(1.0, 5.0 * dt)
        air = self.air_weight
        u = float(np.clip((speed - config.JOG_SPEED) / (config.SPRINT_SPEED - config.JOG_SPEED), 0.0, 1.0))
        pump = u * w * arms_up
        self.gait_curl = arms_up * (0.7 + 0.3 * u)      # loose fist whenever the arms are up

        # The resting arm is posed forward-kinematically - a shoulder swing angle (from straight
        # down, + forward) and an elbow interior angle - and the fist that produces is handed to
        # the IK, so the elbow always ends up at a chosen, human angle:
        #   hanging (standing / walking): theta = +-22 deg pendulum, elbow 168 -> 158 deg
        #   pump (jog -> sprint):         theta = 5 +- 30..40 deg, elbow 85 (front) .. 130 (back)
        #   airborne balance:             forearms up and out, elbow ~100 deg
        U, F = config.UPPER_ARM_LEN, config.FOREARM_LEN
        walk_k = float(np.clip(speed / config.WALK_SPEED, 0.0, 1.0)) * w

        def fk(theta_deg, beta_deg, abduct_deg, side_sign):
            th = np.radians(theta_deg)
            fa = np.radians(theta_deg + 180.0 - beta_deg)        # forearm angle from straight down
            ab = np.radians(abduct_deg)                          # arm plane tilted out from the body
            down = -world_up * np.cos(ab) + right * side_sign * np.sin(ab)
            return fwd_h * (U * np.sin(th) + F * np.sin(fa)) + down * (U * np.cos(th) + F * np.cos(fa))

        def rest_pose(side_sign, phase):
            # side_sign: outward direction for this arm (-1 left, +1 right)
            sn = np.sin(phase)
            hang = (config.ARM_HANG_SWING_DEG * walk_k * sn,
                    180.0 - config.ARM_HANG_ELBOW_BEND_DEG - 10.0 * w * 0.5 * (1.0 + sn),
                    5.0)
            pump_amp = (30.0 + 12.0 * u) * w
            pump = (8.0 + pump_amp * sn, 104.0 - (18.0 + 8.0 * u) * sn * w, -4.0)
            bal = (35.0, 100.0, 12.0)
            th = hang[0] + (pump[0] - hang[0]) * arms_up
            be = hang[1] + (pump[1] - hang[1]) * arms_up
            la = hang[2] + (pump[2] - hang[2]) * arms_up
            th += (bal[0] - th) * air
            be += (bal[1] - be) * air
            la += (bal[2] - la) * air
            return fk(th, be, la, side_sign)

        rest_left = left_shoulder + rest_pose(-1.0, ph)
        rest_right = right_shoulder + rest_pose(+1.0, ph + np.pi)
        # zombie tag: a zombie's arms reach out in front (you see your green hands grasping ahead)
        self.zombie_weight += ((1.0 if zombie else 0.0) - self.zombie_weight) * min(1.0, 4.0 * dt)
        if self.zombie_weight > 1e-3:
            zw = self.zombie_weight * (1.0 - air * 0.6)
            for side, sh_pos, name in ((-1.0, left_shoulder, "l"), (1.0, right_shoulder, "r")):
                wob = 0.04 * np.sin(self.gait_phase + (0.0 if side < 0 else 1.7))
                reach = (sh_pos + fwd_h * 0.50 - world_up * (0.16 + wob) + right * side * 0.06)
                if name == "l":
                    rest_left = rest_left + (reach - rest_left) * zw
                else:
                    rest_right = rest_right + (reach - rest_right) * zw
        up_mix = max(arms_up, air)
        # hand frames (used as targets when something blends out of a press into the rest pose)
        roll_in = 0.8 - 0.45 * pump
        rest_rot_l = self._hand_frame(fwd_h * up_mix - world_up * (1.0 - up_mix * 0.6), right * (1.0 - up_mix) + (-world_up * 0.6 + right * roll_in) * up_mix)
        rest_rot_r = self._hand_frame(fwd_h * up_mix - world_up * (1.0 - up_mix * 0.6), -right * (1.0 - up_mix) + (-world_up * 0.6 - right * roll_in) * up_mix)

        left_wrist, right_wrist = rest_left, rest_right
        left_wrist_rot, right_wrist_rot = rest_rot_l, rest_rot_r
        rest_angles = pose_angles(HandPoseType.REST_SWAY, curl_factor=self.gait_curl)
        # Hand specs are resolved into wrist frames after the IK (they depend on the forearm axis).
        # Hanging: the hand continues the forearm with a relaxed flexion, palm toward the thigh.
        # Pumping: fist nearly in line with the forearm, thumbs up.  The wrist never twists just
        # because the arm swings.
        rest_bend = fwd_h * (1.0 - up_mix) - world_up * up_mix
        # (the roll's sign flips between the two poses because the bend axis does)
        rest_roll = 90.0 * (1.0 - up_mix) - (config.WRIST_REST_ROLL - 20.0 * pump) * up_mix
        spec_l = HandSpec(HandSpec.OPEN, None, config.WRIST_EXT_REST, rest_angles, bend=rest_bend, roll=rest_roll)
        spec_r = HandSpec(HandSpec.OPEN, None, config.WRIST_EXT_REST, rest_angles, bend=rest_bend, roll=-rest_roll)
        half = config.HAND_SPACING * 0.5

        reach_len = (config.UPPER_ARM_LEN + config.FOREARM_LEN) * config.HANDOVER_REACH_FRACTION
        shoulders = (left_shoulder, right_shoulder)
        # Where the shoulders will be when a hand-over lands: a climber reaches for where the body
        # is going, so a hold that comes into reach during the hop is grabbed as the body arrives.
        # (A hand leaving a deep press opens slower, so its hop - and its look-ahead - is longer.)
        open_time = [max(0.0, (e if e is not None else 0.0) - config.WRIST_EXT_OPEN) / (config.WRIST_OPEN_RATE * rate)
                     for e in self._last_ext]
        horizons = [max(config.HANDOVER_TIME / rate, ot / config.HOP_OPEN_FRACTION) for ot in open_time]
        if self._prev_shoulders is None or dt <= 0.0:
            ahead = (left_shoulder, right_shoulder)
        else:
            ahead = tuple(sh + (sh - prev) * (hz / dt) for sh, prev, hz in zip(shoulders, self._prev_shoulders, horizons))
        self._prev_shoulders = (left_shoulder.copy(), right_shoulder.copy())

        def planted_specs(default_rail, prop_angles=None):
            """Hang specs for the two planner hands: each on its own rail; a hopping hand opens on
            lift-off and closes again as it lands (continuous wrist and finger angles)."""
            out = []
            for i, hand in enumerate(self.planner.hands):
                sh_y = float(shoulders[i][1])
                if hand.moving:
                    src = hand.rail or default_rail
                    dst = hand.to_rail or default_rail
                    src_kind, dst_kind = grip_kind(src), grip_kind(dst)
                    if hand.t <= dt * 1.01 + 1e-6 or self._hop_src_ext[i] is None:
                        # hop just started: open from the extension the hand was actually shown with
                        self._hop_src_ext[i] = self._last_ext[i] if self._last_ext[i] is not None else hang_extension(src_kind, float(hand.from_pos[1]), sh_y)
                        self._hop_src_clear[i] = self._last_clear[i]
                    ext_src = self._hop_src_ext[i]
                    clear_src = self._hop_src_clear[i]
                    ext_dst = hang_extension(dst_kind, float(wrist_on_rail(dst, hand.to_d)[1]), sh_y)
                    open_angles = pose_angles(HandPoseType.OPEN)
                    t = float(np.clip(hand.t, 0.0, 1.0))
                    if t < config.HOP_OPEN_FRACTION:
                        k = t / config.HOP_OPEN_FRACTION
                        ext = ext_src + (config.WRIST_EXT_OPEN - ext_src) * k
                        angles = blend_angles(POSES[hold_pose_for(src_kind)], open_angles, k)
                        clear = clear_src + (1.0 - clear_src) * k
                    elif t > 1.0 - config.HOP_OPEN_FRACTION:
                        k = (t - (1.0 - config.HOP_OPEN_FRACTION)) / config.HOP_OPEN_FRACTION
                        ext = config.WRIST_EXT_OPEN + (ext_dst - config.WRIST_EXT_OPEN) * k
                        angles = blend_angles(open_angles, POSES[hold_pose_for(dst_kind)], k)
                        clear = 1.0 - k
                    else:
                        ext, angles, clear = config.WRIST_EXT_OPEN, open_angles, 1.0
                    out.append(HandSpec(HandSpec.OPEN, dst, ext, angles, wall_clear=clear))
                else:
                    rail_i = hand.rail or default_rail
                    wy = float(hand.planted_pos[1])
                    out.append(HandSpec.hang(rail_i, wy, sh_y, angles=prop_angles))
            return out

        def plant_and_reach(default_rail, desired, threshold=0.08, max_duration=None):
            """
            Drive the planner with `desired` per-hand (rail, d) targets and return the shown wrist
            positions and specs.  A planted hand whose edge has moved out of the arm's reach (the body
            climbed or dropped away from it) does not stay glued to it: it opens and slides toward its
            desired target at arm's length until the hop takes it there.  Hops start from the shown
            position, so nothing ever snaps back to the old edge.
            """
            shown = [None, None]
            hop_desired = [None, None]
            for i, hand in enumerate(self.planner.hands):
                if desired[i] is not None:
                    target = wrist_on_rail(desired[i][0], desired[i][1])
                    # a hop is only launched toward a target the arm can reach by the time it lands
                    if (float(np.linalg.norm(target - shoulders[i])) <= reach_len or hand.rail is None
                            or float(np.linalg.norm(target - ahead[i])) <= reach_len):
                        hop_desired[i] = desired[i]
                if hand.moving or hand.rail is None or desired[i] is None:
                    self._reach_pos[i] = None
                    continue
                if float(np.linalg.norm(hand.planted_pos - shoulders[i])) <= reach_len:
                    self._reach_pos[i] = None
                    continue
                # shoulder-relative so the body carries the hand; speed-limited so it slides, not snaps
                goal = self._reach_clamped(shoulders[i], wrist_on_rail(desired[i][0], desired[i][1])) - shoulders[i]
                prev = self._reach_pos[i]
                if prev is None:
                    prev = self._reach_clamped(shoulders[i], hand.planted_pos) - shoulders[i]
                # a hand sliding up past its shoulder swings a little wide of it (the whole
                # motion, sideways included, is speed-limited so it never snaps)
                lateral = hand_right_of(desired[i][0]) * (1.0 if i == 1 else -1.0)
                step_v = keep_clear_of_shoulder(goal, lateral) - prev
                dist = float(np.linalg.norm(step_v))
                max_step = config.REACH_SLIDE_SPEED * rate * dt
                if dist > max_step:
                    step_v *= max_step / dist
                nxt = keep_clear_of_shoulder(prev + step_v, lateral) - prev
                dist = float(np.linalg.norm(nxt))
                if dist > max_step:
                    nxt *= max_step / dist
                self._reach_pos[i] = prev + nxt
                shown[i] = shoulders[i] + self._reach_pos[i]
            # a hand leaving a deep press (wrist bent far back) needs longer to open than one
            # leaving a hook: its hop is stretched so the wrist unbends at a human rate
            self.planner.update(dt, hop_desired, threshold=threshold, positions=shown, anchors=shoulders,
                                min_open_time=open_time, rate=rate, max_duration=max_duration)
            pos = list(self.planner.positions())
            specs = planted_specs(default_rail)
            for i in range(2):
                if shown[i] is not None and not self.planner.hands[i].moving:
                    pos[i] = shown[i]
                    # the freed hand opens gradually (from whatever extension it was holding with)
                    # (eased from the designed extension, not the shown angle, so the wall-clearing
                    # flex below does not compound with the easing)
                    start = self._last_open_ext[i]
                    if start is None:
                        start = self._last_ext[i] if self._last_ext[i] is not None else config.WRIST_EXT_OPEN
                    ease = min(1.0, dt / config.CLEAR_BLEND_TIME)
                    ext = start + (config.WRIST_EXT_OPEN - start) * ease
                    clear = self._last_clear[i] + (1.0 - self._last_clear[i]) * ease
                    specs[i] = HandSpec(HandSpec.OPEN, desired[i][0], ext, pose_angles(HandPoseType.OPEN), wall_clear=clear)
                elif self.planner.hands[i].moving:
                    self._reach_pos[i] = None
            return pos[0], pos[1], specs[0], specs[1]

        on_rail_now = False
        if fsm_state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY) and active_rail is not None and grip_point is not None:
            # Hands planted on the rail; a traverse is hand-over-hand (see HandPlanner)
            rail = active_rail
            on_rail_now = True
            grip_d = rail.distance_along(grip_point)
            if not self._was_on_rail or self.planner.hands[0].rail is None:
                self.planner.snap(rail, grip_d - half, grip_d + half)
            left_wrist, right_wrist, spec_l, spec_r = plant_and_reach(rail, self.planner.resting_targets(rail, grip_d, shimmy_dir))
            pose_type = hold_pose_for(grip_kind(rail))

        elif fsm_state == ParkourState.CORNER_TURN and active_rail is not None and corner_link is not None:
            # Lead hand reaches around the corner onto the joined edge first, the other follows
            rail = active_rail
            on_rail_now = True
            q = corner_link.rail
            entry_d = q.length * float(corner_entry_s)
            lead = self.planner.RIGHT if corner_link.side == "right" else self.planner.LEFT
            trail = 1 - lead
            corner_d = 0.0 if corner_link.side == "right" else q.length

            def around_corner(i, d_full):
                # The hand reaches around the corner as far along the new edge as the arm allows
                # (the body is still on the old wall), then walks out to its resting spot as the
                # body comes round.  None while not even the corner itself is within reach.
                best = None
                for k in range(13):
                    d = corner_d + (d_full - corner_d) * (k / 12.0)
                    if float(np.linalg.norm(wrist_on_rail(q, d) - shoulders[i])) <= reach_len:
                        best = d
                return None if best is None else (q, best)

            desired = [None, None]
            desired[lead] = around_corner(lead, entry_d + (half if lead == self.planner.RIGHT else -half))
            if self.planner.hands[lead].rail is q:
                desired[trail] = around_corner(trail, entry_d + (half if trail == self.planner.RIGHT else -half))
            if desired[lead] is None:
                desired[lead] = (rail, self.planner.hands[lead].d)     # not yet: keep the old edge
            if desired[trail] is None:
                # the trailing hand keeps its spot on the old edge until the lead hand has landed
                # (and lets it slip to arm's length if the body has already moved too far round)
                desired[trail] = (rail, self.planner.hands[trail].d)
            left_wrist, right_wrist, spec_l, spec_r = plant_and_reach(rail, desired, threshold=0.06)
            pose_type = hold_pose_for(grip_kind(rail))

        elif fsm_state == ParkourState.CLIMB_UP and active_rail is not None and grip_point is not None:
            # Pull-up: hands stay on the hold while the chest rises to it (wrist extends as it does).
            # Push-up: on a deep ledge the palms press the lip; on a thin edge the hands stay hooked.
            # Then the hands hand over to the next hold, opening as they hop.
            rail = active_rail
            on_rail_now = True
            grip_d = rail.distance_along(grip_point)
            if not self._was_on_rail or self.planner.hands[0].rail is None:
                self.planner.snap(rail, grip_d - half, grip_d + half)
            pull_frac = config.CLIMB_PULL_TIME / config.CLIMB_TOTAL_TIME
            cur = self.planner.resting_targets(rail, grip_d, 0.0)
            if climb_phase < pull_frac or climb_next is None:
                desired = list(cur)
            else:
                # Hand over to the next hold only once it is actually within that arm's reach; until
                # then keep holding the current edge (or reach up the wall toward the new hold if the
                # current edge has dropped out of reach as the legs push the body up)
                nrail, ns = climb_next
                nxt = self.planner.resting_targets(nrail, nrail.length * float(ns), 0.0)
                desired = list(cur)
                for i, sh in enumerate(shoulders):
                    hand = self.planner.hands[i]
                    nxt_pos = wrist_on_rail(nxt[i][0], nxt[i][1])
                    if (hand.rail is nrail or float(np.linalg.norm(nxt_pos - sh)) <= reach_len
                            or float(np.linalg.norm(nxt_pos - ahead[i])) <= reach_len):
                        desired[i] = nxt[i]
                    elif float(np.linalg.norm(hand.planted_pos - sh)) > reach_len:
                        desired[i] = nxt[i]      # out of reach of both: reach toward the new hold
                    else:
                        desired[i] = cur[i]
            caps = None
            if climb_next is not None and move_time_left is not None:
                # a hand-over to the next hold lands no later than the body gets there
                caps = [float(move_time_left) if (d is not None and d[0] is climb_next[0]) else None for d in desired]
            left_wrist, right_wrist, spec_l, spec_r = plant_and_reach(rail, desired, max_duration=caps)
            if climb_phase >= pull_frac and grip_kind(rail) == GripKind.LEDGE and climb_direction > 0:
                # palms prop the body on the lip of a deep ledge while the legs push (eased in)
                b = float(np.clip((climb_phase - pull_frac) * config.CLIMB_TOTAL_TIME / config.PRESS_BLEND_TIME, 0.0, 1.0))
                for i, spec in enumerate((spec_l, spec_r)):
                    hand = self.planner.hands[i]
                    if not hand.moving and hand.rail is rail and spec.mode == HandSpec.HANG:
                        hook_pos = wrist_on_rail(rail, hand.d)
                        press_pos = rail.point_at_distance(hand.d) + press_offset(rail)
                        pos = hook_pos + (press_pos - hook_pos) * b
                        spec.blend = b
                        if i == 0:
                            left_wrist = pos
                        else:
                            right_wrist = pos
            pose_type = hold_pose_for(grip_kind(rail))

        elif fsm_state == ParkourState.MANTLE_VAULT and active_rail is not None and grip_point is not None:
            rail = active_rail
            hr = self._rail_hand_right(rail)
            on_rail_now = True
            hook_l = grip_point - hr * half + wrist_offset(rail)
            hook_r = grip_point + hr * half + wrist_offset(rail)
            press_l = grip_point - hr * half + press_offset(rail)
            press_r = grip_point + hr * half + press_offset(rail)
            if mantle_reverse:
                # Lowering over the edge: palms on the top first, then the hands roll over the
                # edge into a hook as the body drops below it
                b = 1.0 - float(np.clip((climb_phase - 0.30) / 0.35, 0.0, 1.0))
            else:
                # Hoist: from the hook on the edge into a palm press as the chest passes the edge
                b = float(np.clip((climb_phase - 0.2) / 0.3, 0.0, 1.0))
            left_wrist = hook_l + (press_l - hook_l) * b
            right_wrist = hook_r + (press_r - hook_r) * b
            spec_l = HandSpec.hang(rail, float(hook_l[1]), float(left_shoulder[1]))
            spec_r = HandSpec.hang(rail, float(hook_r[1]), float(right_shoulder[1]))
            spec_l.blend = b
            spec_r.blend = b
            pose_type = HandPoseType.MANTLE_PALM if b > 0.5 else hold_pose_for(grip_kind(rail))
            if not mantle_reverse and climb_phase > config.MANTLE_RELEASE_PHASE:
                # Step-through: the palms leave the surface and the arms swing forward into the rest pose
                r = float(np.clip((climb_phase - config.MANTLE_RELEASE_PHASE) / (1.0 - config.MANTLE_RELEASE_PHASE), 0.0, 1.0))
                r = r * r * (3.0 - 2.0 * r)
                if self._release_from is None:
                    # remember where the hands let go, relative to the eye, so they travel with the body
                    self._release_from = (left_wrist - cam_pos, right_wrist - cam_pos)
                left_wrist = cam_pos + self._release_from[0] + ((rest_left - cam_pos) - self._release_from[0]) * r
                right_wrist = cam_pos + self._release_from[1] + ((rest_right - cam_pos) - self._release_from[1]) * r
                angles = blend_angles(pose_angles(HandPoseType.MANTLE_PALM), rest_angles, r)
                spec_l = HandSpec.release(rail, rest_rot_l, angles, r)
                spec_r = HandSpec.release(rail, rest_rot_r, angles, r)
                on_rail_now = r < 0.999

        elif fsm_state in (ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT):
            # Dynamic push off surface: palms flat, arms driving down/forward
            left_wrist = left_shoulder + fwd_h * 0.42 - world_up * 0.05 - right * 0.08
            right_wrist = right_shoulder + fwd_h * 0.42 - world_up * 0.05 + right * 0.08
            rot = self._hand_frame(world_up, fwd_h)
            spec_l = HandSpec.free(rot, pose_angles(HandPoseType.DYNO_PUSH))
            spec_r = HandSpec.free(rot, pose_angles(HandPoseType.DYNO_PUSH))
            left_wrist_rot = right_wrist_rot = rot
            pose_type = HandPoseType.DYNO_PUSH

        elif grab_candidate is not None or targeted_rail is not None:
            # Anticipation: the hands drift toward the hold being looked at.  Airborne they
            # reach for it in earnest; on the ground it is a subtle lift, never past arm's length.
            if grab_candidate is not None:
                rail, pt_target = grab_candidate.rail, grab_candidate.grip
            else:
                rail, pt_target = targeted_rail[0], targeted_rail[1]
            dist = float(np.linalg.norm(pt_target - cam_pos))
            if on_ground:
                blend = float(np.clip((3.2 - dist) / 2.2, 0.0, 1.0)) * config.GROUND_AIM_BLEND
            else:
                blend = float(np.clip((2.2 - dist) / 1.2, 0.0, 1.0))
            if blend > 0.0:
                tl, tr = self._grip_hands(pt_target, rail)
                tl = self._reach_clamped(left_shoulder, tl)
                tr = self._reach_clamped(right_shoulder, tr)
                left_wrist = rest_left + (tl - rest_left) * blend
                right_wrist = rest_right + (tr - rest_right) * blend
                # fingers turn toward the hold, the hand opens, palms roll toward it as the reach completes
                to_hold = pt_target - cam_pos
                to_hold /= (np.linalg.norm(to_hold) + 1e-7)
                f_dir = (fwd - cam_up * 0.35) * (1.0 - blend) + to_hold * blend
                left_wrist_rot = self._hand_frame(f_dir, (-cam_up * 0.6 + right * 0.8) * (1.0 - blend) - rail.normal * blend)
                right_wrist_rot = self._hand_frame(f_dir, (-cam_up * 0.6 - right * 0.8) * (1.0 - blend) - rail.normal * blend)
                angles = blend_angles(rest_angles, pose_angles(HandPoseType.OPEN), blend)
                spec_l = HandSpec.free(left_wrist_rot, angles)
                spec_r = HandSpec.free(right_wrist_rot, angles)
                pose_type = HandPoseType.PRE_REACH if blend > 0.5 else HandPoseType.REST_SWAY

        # ---- Acrobatics: flips and rolls --------------------------------------------------
        # The body is posed upright and tucked here; the whole-body rotation is applied to the
        # finished palette (and the camera) by the renderer, so the pose stays anatomical.  A tuck
        # brings the hands to the shins just below the knees; a roll first puts them to the
        # ground ahead.
        tuck = spin.tuck() if spin is not None else 0.0
        plant = spin.reach() if spin is not None else 0.0
        if tuck > 1e-3 or plant > 1e-3:
            hip_c = body_eye - world_up * (config.PLAYER_EYE_HEIGHT - config.HIP_HEIGHT)
            targets = []
            for side, shoulder in ((-1.0, left_shoulder), (1.0, right_shoulder)):
                knee = hip_c + fwd_h * 0.36 + world_up * 0.12 + right * side * 0.14
                shin = knee + fwd_h * 0.05 - world_up * 0.09
                ground = shoulder + fwd_h * 0.42 - world_up * 0.40 + right * side * 0.04
                t = shin + (ground - shin) * plant if tuck > 1e-3 else ground
                targets.append(self._reach_clamped(shoulder, t))
            w = max(tuck, plant)
            left_wrist = left_wrist + (targets[0] - left_wrist) * w
            right_wrist = right_wrist + (targets[1] - right_wrist) * w
            fists = blend_angles(rest_angles, pose_angles(HandPoseType.CRIMP), tuck)
            angles = blend_angles(fists, pose_angles(HandPoseType.DYNO_PUSH), plant)
            down = -world_up * 0.8 + fwd_h * 0.45
            left_wrist_rot = self._hand_frame(down, -fwd_h * tuck + (-world_up) * plant + right * 0.3)
            right_wrist_rot = self._hand_frame(down, -fwd_h * tuck + (-world_up) * plant - right * 0.3)
            spec_l = HandSpec.free(left_wrist_rot, angles)
            spec_r = HandSpec.free(right_wrist_rot, angles)

        # Smoothing state is kept relative to the eye so body motion never drags the hands behind
        if on_rail_now or fsm_state in (ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT) or self._smooth is None:
            self._smooth = [left_wrist - cam_pos, right_wrist - cam_pos, left_wrist_rot.copy(), right_wrist_rot.copy()]
        else:
            # Free hands ease toward their targets instead of snapping (critically damped feel)
            k = min(1.0, config.HAND_FOLLOW_RATE * dt)
            self._smooth[0] += ((left_wrist - cam_pos) - self._smooth[0]) * k
            self._smooth[1] += ((right_wrist - cam_pos) - self._smooth[1]) * k
            for idx, rot in ((2, left_wrist_rot), (3, right_wrist_rot)):
                cur = self._smooth[idx]
                y = cur[0:3, 1] + (rot[0:3, 1] - cur[0:3, 1]) * k
                z = cur[0:3, 2] + (rot[0:3, 2] - cur[0:3, 2]) * k
                self._smooth[idx] = self._hand_frame(y, z)
            left_wrist, right_wrist = cam_pos + self._smooth[0], cam_pos + self._smooth[1]
            # world-defined frames (aiming, reaching) are smoothed as frames; forearm-relative rest
            # hands instead ease their wrist extension from whatever they were just doing
            for i, spec in enumerate((spec_l, spec_r)):
                if spec.mode == HandSpec.FREE:
                    if i == 0:
                        spec_l = HandSpec.free(self._smooth[2], spec.angles)
                    else:
                        spec_r = HandSpec.free(self._smooth[3], spec.angles)
                elif self._last_ext[i] is not None:
                    spec.ext = self._last_ext[i] + (spec.ext - self._last_ext[i]) * k

        if not on_rail_now:
            self.planner.reset()
        if not (fsm_state == ParkourState.MANTLE_VAULT and not mantle_reverse and climb_phase > config.MANTLE_RELEASE_PHASE):
            self._release_from = None
        self._was_on_rail = on_rail_now

        # Solve Two-Bone IK for elbows (pole directions relative to the shoulder, torso frame)
        l_pole = right * config.POLE_DIR_L[0] + world_up * config.POLE_DIR_L[1] + fwd_h * config.POLE_DIR_L[2]
        r_pole = right * config.POLE_DIR_R[0] + world_up * config.POLE_DIR_R[1] + fwd_h * config.POLE_DIR_R[2]

        ik_l = solve_two_bone_ik(left_shoulder, left_wrist, config.UPPER_ARM_LEN, config.FOREARM_LEN, l_pole, fwd_h)
        ik_r = solve_two_bone_ik(right_shoulder, right_wrist, config.UPPER_ARM_LEN, config.FOREARM_LEN, r_pole, fwd_h)

        # Elbows cannot go into the wall the hands are on: rotate them about the shoulder-wrist
        # axis (keeping both bone lengths) until they clear it
        if on_rail_now and active_rail is not None:
            wall_rail = active_rail
            for ik, sh, pole in ((ik_l, left_shoulder, l_pole), (ik_r, right_shoulder, r_pole)):
                self._keep_elbow_off_wall(ik, sh, wall_rail, pole)

        # Resolve the hand frames against the forearms the IK produced (designed wrist angles)
        for spec, ik in ((spec_l, ik_l), (spec_r, ik_r)):
            if spec.mode == HandSpec.OPEN and spec.rail is not None:
                spec.wall_dist = float(np.dot(ik.wrist - spec.rail.p1, spec.rail.normal))
        left_wrist_rot, l_angles = resolve_hand(spec_l, ik_l.wrist - ik_l.elbow)
        right_wrist_rot, r_angles = resolve_hand(spec_r, ik_r.wrist - ik_r.elbow)
        l_fingers = FingerSolver.transforms_for(l_angles, is_left=True)
        r_fingers = FingerSolver.transforms_for(r_angles, is_left=False)

        self.skeleton.solve(
            left_shoulder, ik_l.elbow, ik_l.wrist, left_wrist_rot, l_fingers,
            right_shoulder, ik_r.elbow, ik_r.wrist, right_wrist_rot, r_fingers
        )
        if on_rail_now or fsm_state in (ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT):
            # free-hand smoothing continues from the frames actually shown
            self._smooth[2] = left_wrist_rot.copy()
            self._smooth[3] = right_wrist_rot.copy()
        self._last_ext = [float(self.skeleton.wrist_angles[0]), float(self.skeleton.wrist_angles[1])]
        self._last_clear = [spec_l.wall_clear if spec_l.mode == HandSpec.OPEN and spec_l.rail is not None else 0.0,
                            spec_r.wall_clear if spec_r.mode == HandSpec.OPEN and spec_r.rail is not None else 0.0]
        self._last_open_ext = [spec.ext if spec.mode == HandSpec.OPEN and spec.rail is not None and not self.planner.hands[i].moving else None
                               for i, spec in enumerate((spec_l, spec_r))]

        # Legs: same torso frame and gait as the arms
        self.legs.update(dt, player_pos, fwd_h, right, fsm_state, active_rail, grip_point, climb_phase,
                         climb_next, velocity, on_ground, mantle_reverse, climb_direction, supported_hang,
                         self.gait_phase, self.gait_weight, corner_link=corner_link, body_tuck=tuck)

        # Telemetry (used by the test-suite and debugging overlays)
        self.pose_type = pose_type
        self.last = {
            "left_shoulder": left_shoulder, "right_shoulder": right_shoulder,
            "left_elbow": ik_l.elbow, "right_elbow": ik_r.elbow,
            "left_wrist": ik_l.wrist, "right_wrist": ik_r.wrist,
            "left_target": np.asarray(left_wrist, dtype=np.float32),
            "right_target": np.asarray(right_wrist, dtype=np.float32),
            "left_elbow_angle": ik_l.elbow_angle_deg, "right_elbow_angle": ik_r.elbow_angle_deg,
            "left_pole": l_pole, "right_pole": r_pole,
            "forward_h": fwd_h, "right": right,
            "moving_hand": self.planner.moving_hand(),
            "wrist_angles": tuple(self.skeleton.wrist_angles),
            "hand_modes": (spec_l.effective_mode, spec_r.effective_mode),
            "hand_frames": (left_wrist_rot, right_wrist_rot),
            "finger_angles": (tuple(l_angles), tuple(r_angles)),
            "hand_rails": (self.planner.hands[0].rail, self.planner.hands[1].rail),
            "hand_d": tuple((h.to_d if h.moving else h.d) for h in self.planner.hands),
        }

    def get_palette(self):
        return self.skeleton.get_palette()
