"""Procedural first-person legs: hips, knees, ankles and boots that work with the arms.

The legs are a two-bone chain per side (thigh / shin) solved with the same analytical IK as
the arms, with *knee* limits (40-179 deg interior angle: deep flexion to straight, never
hyper-extended) and a knee pole that always points forward-up, so a knee can only ever bend
the way a knee bends.  Each state of the parkour FSM supplies ankle targets:

  * standing      - feet under the hips, a hair of knee bend, perfectly still
  * walk / run    - a gait locked to the arm swing (the left leg drives forward with the
                    right arm): the stance foot is on the ground and travels backward under
                    the body, the swing foot lifts and comes through, higher and further the
                    faster the run; the hips sink a little so the spread legs stay in reach
  * jump / fall   - knees drive up after take-off, the legs extend for the landing
  * hang / shimmy - legs hang, knees a touch bent, toes at the wall (never inside it)
  * climb-up      - the push foot smears the wall below the hold during the pull-up, steps
                    onto the hold once the hips are high enough, stays planted there while
                    the leg extends and the body rises, then lets go and hangs
  * mantle        - hoist, a knee comes up and the foot plants on the top surface, the leg
                    extends into the stand, the trailing foot swings over the edge
  * climb down    - the reverse: one foot over the edge, then the other, into a hang
  * dyno / eject  - both legs drive off the wall then tuck

Everything is written into the shared skeleton palette (bones TORSO..R_FOOT) and the mesh
(black pants and shirt, brown boots) is appended to the arm mesh, so one skinned draw
renders the whole body.
"""
import numpy as np
from math3d.geometry import cross3
import config
from animation.skeleton import ArmSkeleton, BoneID
from animation.two_bone_ik import solve_two_bone_ik
from animation.grip import grip_kind, GripKind
from parkour.state_machine import ParkourState

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class LegState:
    HANG = "hang"          # dangling
    STAND = "stand"        # weight on the ground
    SWING = "swing"        # in the air during a gait cycle
    PLANT = "plant"        # planted on a hold / surface (world-fixed)
    SMEAR = "smear"        # sole against the wall
    TUCK = "tuck"          # airborne, knee up
    EXTEND = "extend"      # airborne, reaching for the landing


class FirstPersonLegRig:
    LEFT, RIGHT = 0, 1

    def __init__(self, skeleton: ArmSkeleton):
        self.skeleton = skeleton
        self.ankle_s = [None, None]          # smoothed ankle targets (world)
        self.ankle_rel = [None, None]        # the same, relative to the hip (free legs ease in this space)
        self.toe_s = [None, None]            # smoothed toe directions (world)
        self.planted = [None, None]          # world-fixed plant position while PLANT
        self.leg_state = [LegState.STAND, LegState.STAND]
        self.push_leg = self.RIGHT           # leg that steps onto the hold; alternates per climb
        self._last_climb_state = None
        self.foothold = None                 # rail the feet last stood on while climbing (kept while it is in reach)
        self.foot_spot = [None, None]        # where each foot stands on the foothold (world)
        self.step = [None, None]             # (from, to, t) while a foot steps along the foothold
        self.hip_drop = 0.0
        self.last = {}
        self.vertices, self.indices = self._build_mesh()

    # ------------------------------------------------------------------ mesh
    @staticmethod
    def _build_mesh():
        from animation.arm_rig import _make_prism, _make_sphere, _make_tapered_box, MAT_CLOTH, MAT_BOOT, MAT_SHIRT
        verts, idx = [], []

        def add(v, i):
            off = len(verts)
            verts.extend(v)
            idx.extend(i + off)

        k = 1.0 / np.cos(np.pi / 4.0)      # a 4-sided prism's corners sit at 45 deg: scale to true width
        torso_len = config.SHOULDER_HEIGHT - config.HIP_HEIGHT
        # Torso: from the hip centre up to the shoulders (shirt), set back a little from the centre
        add(*_make_tapered_box(BoneID.TORSO, 0.32 * k, 0.20 * k, 0.40 * k, 0.19 * k, torso_len + 0.03,
                               y_offset=0.0, material=MAT_SHIRT, centre=(0.0, -0.02), centre1=(0.0, -0.03)))
        # Pelvis: waist block around the hip joints
        add(*_make_tapered_box(BoneID.PELVIS, 0.34 * k, 0.22 * k, 0.33 * k, 0.21 * k, 0.20,
                               y_offset=-0.14, material=MAT_CLOTH, centre=(0.0, -0.01)))
        for thigh, shin, foot, side in ((BoneID.L_THIGH, BoneID.L_SHIN, BoneID.L_FOOT, -1.0),
                                        (BoneID.R_THIGH, BoneID.R_SHIN, BoneID.R_FOOT, 1.0)):
            # Thigh: hip cap and a tapered leg of the pants down to the knee
            add(*_make_sphere(thigh, 0.078, rings=4, segs=8, squash=(1.0, 0.9, 1.0), material=MAT_CLOTH))
            add(*_make_prism(thigh, (0.076, 0.080), (0.060, 0.062), config.THIGH_LEN, sides=8, cap_end=False, material=MAT_CLOTH))
            # Knee cap + shin (pants) with the boot shaft over the lower shin
            add(*_make_sphere(shin, 0.062, rings=4, segs=8, material=MAT_CLOTH))
            add(*_make_prism(shin, (0.058, 0.060), (0.046, 0.050), config.SHIN_LEN - 0.14, sides=8, cap_start=False, material=MAT_CLOTH))
            add(*_make_prism(shin, (0.052, 0.056), (0.048, 0.052), 0.16, sides=8, y_offset=config.SHIN_LEN - 0.16, material=MAT_BOOT))
            # Boot (local +Y = toe direction, local +Z = up): an ankle cap, a rounded heel, a
            # ten-sided body that tapers toward the toe and a domed toe cap - no square tips
            sole = -config.ANKLE_HEIGHT
            heel_r, toe_r = (0.052, 0.040), (0.044, 0.030)
            add(*_make_sphere(foot, 0.046, rings=3, segs=8, squash=(1.0, 0.9, 1.0), material=MAT_BOOT))
            add(*_make_sphere(foot, heel_r[0], rings=4, segs=10, squash=(1.0, 0.9, heel_r[1] / heel_r[0]),
                              material=MAT_BOOT, centre=(0.0, -0.03, sole + heel_r[1])))
            add(*_make_prism(foot, heel_r, toe_r, 0.21, sides=10, y_offset=-0.05, material=MAT_BOOT,
                             centre=(0.0, sole + heel_r[1]), centre1=(0.0, sole + toe_r[1])))
            add(*_make_sphere(foot, toe_r[0], rings=4, segs=10, squash=(1.0, 1.25, toe_r[1] / toe_r[0]),
                              material=MAT_BOOT, centre=(0.0, 0.16, sole + toe_r[1])))
            # a thin black sole under it all
            add(*_make_tapered_box(foot, 0.098 * k, 0.014 * k, 0.082 * k, 0.014 * k, 0.27, y_offset=-0.07,
                                   material=MAT_CLOTH, centre=(0.0, sole + 0.006), centre1=(0.0, sole + 0.006)))
        return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _ease(cur, target, rate, dt):
        if cur is None:
            return np.asarray(target, dtype=np.float32).copy()
        return cur + (np.asarray(target, dtype=np.float32) - cur) * min(1.0, rate * dt)

    @staticmethod
    def _clamp_ankle(shin_axis, toe_dir, fwd_ref):
        """Keep the foot within the ankle's range: 20 deg of dorsiflexion to 50 deg of plantar
        flexion from the neutral right angle to the shin."""
        a = shin_axis / (np.linalg.norm(shin_axis) + 1e-7)          # knee -> ankle (points down the shin)
        t = toe_dir - a * float(np.dot(toe_dir, a))
        n = float(np.linalg.norm(t))
        if n < 1e-5:
            t = fwd_ref - a * float(np.dot(fwd_ref, a))
            n = float(np.linalg.norm(t))
            if n < 1e-5:
                return toe_dir
        t = t / n
        # neutral foot is perpendicular to the shin; angle from neutral: + = toes down (plantar)
        ang = float(np.degrees(np.arctan2(float(np.dot(toe_dir, a)), float(np.dot(toe_dir, t)))))
        ang = float(np.clip(ang, -20.0, 50.0))
        r = np.radians(ang)
        out = t * np.cos(r) + a * np.sin(r)
        return out / (np.linalg.norm(out) + 1e-7)

    # ---------------------------------------------------------------- update
    def update(self, dt, player_pos, fwd_h, right, fsm_state, active_rail, grip_point, climb_phase,
               climb_next, velocity, on_ground, mantle_reverse, climb_direction, supported_hang,
               gait_phase, gait_weight, corner_link=None, body_tuck=0.0):
        player_pos = np.asarray(player_pos, dtype=np.float32)
        fwd_h = np.asarray(fwd_h, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        speed = float(np.hypot(velocity[0], velocity[2]))
        feet_y = float(player_pos[1]) - config.PLAYER_HALF_HEIGHT
        hip_c = player_pos + UP * (config.HIP_HEIGHT - config.PLAYER_HALF_HEIGHT)
        leg_len = config.THIGH_LEN + config.SHIN_LEN
        on_rail = active_rail is not None and fsm_state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY,
                                                              ParkourState.CORNER_TURN, ParkourState.CLIMB_UP,
                                                              ParkourState.MANTLE_VAULT)
        wall_n = active_rail.normal if on_rail else None
        # how far the hips are from the wall plane (along its normal)
        wall_dist = float(np.dot(hip_c - active_rail.p1, wall_n)) if on_rail else None

        # alternate the stepping leg from one climb / mantle to the next
        climbing_now = fsm_state in (ParkourState.CLIMB_UP, ParkourState.MANTLE_VAULT)
        if climbing_now and self._last_climb_state != fsm_state:
            self.push_leg = 1 - self.push_leg
        self._last_climb_state = fsm_state if climbing_now else None

        moving_on_ground = on_ground and speed > 0.5 and fsm_state in (ParkourState.GROUND, ParkourState.SPRINT)
        w = gait_weight

        # hips sink with the stride so the spread legs stay within reach of the ground
        u_s = float(np.clip(speed / config.SPRINT_SPEED, 0.0, 1.0))
        drop_target = (0.03 + 0.13 * u_s * u_s) * w if moving_on_ground else 0.0
        self.hip_drop += (drop_target - self.hip_drop) * min(1.0, 8.0 * dt)
        hip_c = hip_c - UP * self.hip_drop

        hips = [hip_c - right * config.HIP_HALF_WIDTH, hip_c + right * config.HIP_HALF_WIDTH]
        targets = [None, None]
        toes = [None, None]
        states = [None, None]
        rates = [14.0, 14.0]
        ground_ankle_y = feet_y + config.ANKLE_HEIGHT
        support_y = ground_ankle_y if (on_ground or supported_hang) else None

        def stand_target(i):
            side = -1.0 if i == 0 else 1.0
            return hips[i] + right * side * 0.03 + fwd_h * 0.02 - UP * (hips[i][1] - ground_ankle_y)

        def hang_target(i, toe_in=0.10, bend=0.05):
            # legs dangle; knees a touch bent; toes toward the wall but never inside it
            t = hips[i] + fwd_h * toe_in - UP * (leg_len - bend)
            if wall_n is not None:
                min_out = config.FOOT_LEN * 0.75 + 0.02          # ankle this far out keeps the toe clear
                d = float(np.dot(t - active_rail.p1, wall_n))
                if d < min_out:
                    t = t + wall_n * (min_out - d)
            return t

        def smear_target(i):
            # sole against the wall below the hips, knee bent ~90 deg
            if wall_n is None:
                return hang_target(i)
            out = config.FOOT_LEN * 0.75 + 0.02
            t = hips[i] - UP * 0.42
            d = float(np.dot(t - active_rail.p1, wall_n))
            return t + wall_n * (out - d)

        def plant_on_hold(rail, point, i):
            side = -1.0 if i == 0 else 1.0
            lateral = cross3(-rail.normal, UP)
            lateral = lateral / (np.linalg.norm(lateral) + 1e-7)
            if grip_kind(rail) == GripKind.LEDGE:
                # foot flat on the top surface behind the edge
                return point - rail.normal * 0.10 + UP * config.ANKLE_HEIGHT + lateral * side * 0.12, fwd_h
            # toe on the lip of a thin edge, heel out
            return point + rail.normal * (config.FOOT_LEN * 0.7) - UP * 0.02 + lateral * side * 0.12, (fwd_h * 0.9 - UP * 0.44)

        def foothold_spot(rail, i):
            """Where foot i would stand on `rail`: on the rail under its own hip, a little ahead
            of a traversing body (feet step ahead like the hands do)."""
            ahead = np.array([velocity[0], 0.0, velocity[2]], dtype=np.float32) * 0.25
            d = rail.distance_along(hips[i] + ahead)
            m = config.HAND_RAIL_END_MARGIN
            d = float(np.clip(d, m, max(m, rail.length - m)))
            return plant_on_hold(rail, rail.point_at_distance(d), i)

        def foothold_usable(rail, i):
            """A remembered foothold is used while it is below the hips and within the leg's reach."""
            spot, _ = foothold_spot(rail, i)
            below = float(hips[i][1] - spot[1])
            return 0.25 <= below and float(np.linalg.norm(spot - hips[i])) <= leg_len * 0.97

        def stepping_target(i, spot):
            """A planted foot keeps its spot; when the body has moved on it takes a short step
            (one foot at a time, unless its spot has gone out of reach)."""
            if self.foot_spot[i] is None:
                self.foot_spot[i] = np.asarray(spot, dtype=np.float32).copy()
            st = self.step[i]
            if st is not None:
                frm, to, t = st
                t = min(1.0, t + dt / 0.16)
                k = t * t * (3.0 - 2.0 * t)
                pos = frm + (to - frm) * k + UP * (0.05 * np.sin(np.pi * t))
                if t >= 1.0:
                    self.foot_spot[i] = to.copy()
                    self.step[i] = None
                else:
                    self.step[i] = (frm, to, t)
                return pos
            drift = float(np.linalg.norm(np.asarray(spot) - self.foot_spot[i]))
            out_of_reach = float(np.linalg.norm(self.foot_spot[i] - hips[i])) > leg_len * 0.95
            if drift > 0.12 and (self.step[1 - i] is None or out_of_reach):
                self.step[i] = (self.foot_spot[i].copy(), np.asarray(spot, dtype=np.float32).copy(), 0.0)
            return self.foot_spot[i]

        if not on_rail:
            self.foothold = None
            self.foot_spot = [None, None]
            self.step = [None, None]

        # ------------------------------------------------------------ ground
        if fsm_state in (ParkourState.GROUND, ParkourState.SPRINT) and on_ground and not on_rail:
            if moving_on_ground and w > 0.02:
                steps_per_s = config.GAIT_BASE_CADENCE + config.GAIT_CADENCE_PER_MS * speed
                cycle = 1.0 / max(steps_per_s * 0.5, 0.1)
                half_stride = min(0.50, 0.08 + 0.05 * speed) * w
                lift = (0.06 + 0.035 * speed) * w
                for i in range(2):
                    # the left leg drives forward with the right arm: legs are a quarter cycle
                    # behind the arm phase, and a half cycle apart
                    p = gait_phase - np.pi * 0.5 + (np.pi if i == 0 else 0.0)
                    sn = float(np.sin(p))
                    fore = half_stride * float(np.cos(p))
                    side = -1.0 if i == 0 else 1.0
                    base = hips[i] + right * side * 0.02 + fwd_h * fore
                    if sn >= 0.0:
                        # stance: on the ground, travelling back under the body
                        targets[i] = base - UP * (hips[i][1] - ground_ankle_y)
                        toes[i] = fwd_h
                        states[i] = LegState.STAND
                    else:
                        # swing: lift (heel toward the buttock early, knee through, reach for the ground)
                        h = lift * (-sn)
                        targets[i] = base - UP * (hips[i][1] - ground_ankle_y - h)
                        toes[i] = fwd_h * np.cos(0.35 * -sn) - UP * np.sin(0.35 * -sn)
                        states[i] = LegState.SWING
                    rates[i] = 40.0
                _ = cycle
            else:
                for i in range(2):
                    targets[i] = stand_target(i)
                    toes[i] = fwd_h
                    states[i] = LegState.STAND
                    rates[i] = 12.0

        # ----------------------------------------------------------- airborne
        elif fsm_state in (ParkourState.AIRBORNE, ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT) or (not on_ground and not on_rail):
            vy = float(velocity[1])
            k = _smoothstep((vy + 2.0) / 5.0)          # 1 = rising hard (tuck), 0 = falling (extend)
            for i in range(2):
                side = -1.0 if i == 0 else 1.0
                lead = 0.06 if i == self.push_leg else -0.04
                tuck = hips[i] + fwd_h * (0.16 + lead) + right * side * 0.03 - UP * 0.52
                extend = hips[i] + fwd_h * (0.10 + lead * 0.5) + right * side * 0.04 - UP * 0.76
                targets[i] = extend + (tuck - extend) * k
                toes[i] = fwd_h * np.cos(0.5 * k + 0.15) - UP * np.sin(0.5 * k + 0.15)
                states[i] = LegState.TUCK if k > 0.5 else LegState.EXTEND
                rates[i] = 14.0

        # -------------------------------------------------- hanging / shimmy
        elif fsm_state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY, ParkourState.CORNER_TURN):
            if self.foothold is not None and (self.foothold is active_rail or not any(foothold_usable(self.foothold, i) for i in range(2))):
                self.foothold = None          # climbed away from it (or it became the hand hold)
            for i in range(2):
                if support_y is not None:
                    t = stand_target(i)
                    t = t + UP * (support_y - t[1])
                    targets[i] = t
                    toes[i] = fwd_h
                    states[i] = LegState.STAND
                    rates[i] = 8.0
                elif self.foothold is not None and foothold_usable(self.foothold, i):
                    # feet on the hold below (a rest), stepping along it as the body traverses
                    spot, toe = foothold_spot(self.foothold, i)
                    targets[i] = stepping_target(i, spot)
                    toes[i] = toe
                    states[i] = LegState.PLANT
                    rates[i] = 30.0
                else:
                    # the two legs hang a little differently (one knee slightly more bent)
                    self.foot_spot[i] = None
                    self.step[i] = None
                    targets[i] = hang_target(i, toe_in=0.10 if i == self.push_leg else 0.07, bend=0.05 if i == self.push_leg else 0.02)
                    toes[i] = fwd_h * 0.5 - UP * 0.85
                    states[i] = LegState.HANG
                    rates[i] = 8.0

        # ----------------------------------------------------------- climb-up
        elif fsm_state == ParkourState.CLIMB_UP and active_rail is not None and grip_point is not None:
            pull_frac = config.CLIMB_PULL_TIME / config.CLIMB_TOTAL_TIME
            # the hold the foot works from: the one the hands started on going up, the lower one going down
            if climb_direction > 0 or climb_next is None:
                hold_rail, hold_pt = active_rail, grip_point
            else:
                hold_rail, hold_pt = climb_next[0], climb_next[0].get_point(climb_next[1])
            for i in range(2):
                if i != self.push_leg:
                    # the other leg stays on the hold it was resting on while that is in reach,
                    # otherwise hangs with the toes on the wall and a bit of knee
                    if self.foothold is not None and self.foothold is not hold_rail and foothold_usable(self.foothold, i):
                        spot, toe = foothold_spot(self.foothold, i)
                        targets[i] = stepping_target(i, spot)
                        toes[i] = toe
                        states[i] = LegState.PLANT
                        rates[i] = 30.0
                    else:
                        self.foot_spot[i] = None
                        self.step[i] = None
                        targets[i] = hang_target(i, toe_in=0.12, bend=0.06)
                        toes[i] = fwd_h * 0.6 - UP * 0.8
                        states[i] = LegState.HANG
                        rates[i] = 8.0
                    continue
                plant, toe = plant_on_hold(hold_rail, hold_pt, i)
                hip_above = float(hips[i][1] - plant[1])
                reach_ok = float(np.linalg.norm(plant - hips[i])) <= leg_len * 0.985
                # with the hips this close to the wall the knee can only close so far: the foot
                # goes onto the lip once the hips have risen past it (pushing off the wall face
                # until then), and comes off it on the way down at the same point
                can_step = hip_above >= config.FOOT_ON_HOLD_MIN_HIP_ABOVE
                if can_step and reach_ok:
                    # planted: world-fixed, the leg extends as the body rises (or bends as it lowers)
                    targets[i] = plant
                    toes[i] = toe
                    states[i] = LegState.PLANT
                    rates[i] = 16.0
                    self.foothold = hold_rail
                    self.foot_spot[i] = np.asarray(plant, dtype=np.float32).copy()
                    self.step[i] = None
                elif hip_above < config.FOOT_ON_HOLD_MIN_HIP_ABOVE:
                    # too low to get the foot on: smear the wall below the hold and push on that
                    targets[i] = smear_target(i)
                    toes[i] = fwd_h * 0.8 - UP * 0.6
                    states[i] = LegState.SMEAR
                    rates[i] = 10.0
                else:
                    # the body has risen past the foot: let go and hang (climbing up), or hang
                    # until the lower hold comes within reach (climbing down)
                    targets[i] = hang_target(i, toe_in=0.10, bend=0.06)
                    toes[i] = fwd_h * 0.5 - UP * 0.85
                    states[i] = LegState.HANG
                    rates[i] = 8.0
            _ = pull_frac

        # ------------------------------------------------------ mantle / vault
        elif fsm_state == ParkourState.MANTLE_VAULT and active_rail is not None and grip_point is not None:
            top_pt = grip_point - active_rail.normal * 0.30 + UP * config.ANKLE_HEIGHT
            lateral = cross3(-active_rail.normal, UP)
            lateral = lateral / (np.linalg.norm(lateral) + 1e-7)

            def over_the_edge(i, on_top):
                """A foot going from outside the wall onto the top goes *over* the edge: up first,
                then in (never through the corner)."""
                cur = self.ankle_s[i]
                if cur is not None and float(np.dot(cur - active_rail.p1, wall_n)) > 0.04 and float(cur[1]) < float(on_top[1]) + 0.06:
                    return on_top + wall_n * 0.45 + UP * 0.14
                return on_top

            def off_the_edge(i, hang):
                """The reverse: a foot leaving the top lifts and comes out past the edge before it drops."""
                cur = self.ankle_s[i]
                if cur is not None and float(np.dot(cur - active_rail.p1, wall_n)) < 0.10 and float(cur[1]) > float(active_rail.p1[1]) - 0.05:
                    return cur + wall_n * 0.3 + UP * 0.10
                return hang

            for i in range(2):
                side = -1.0 if i == 0 else 1.0
                on_top = top_pt + lateral * side * 0.12
                if not mantle_reverse:
                    # hoist: the lead knee comes up and the foot plants on the top as soon as the
                    # hips are high enough; the trailing foot swings over the edge in the step phase
                    hip_above = float(hips[i][1] - on_top[1])
                    lead = i == self.push_leg
                    if lead and hip_above >= -0.15:
                        targets[i] = over_the_edge(i, on_top)
                        toes[i] = fwd_h
                        states[i] = LegState.PLANT
                        rates[i] = 18.0
                    elif (not lead) and hip_above >= 0.35 and wall_dist < 0.15:
                        # the body is over the edge: the trailing foot comes up onto the top too
                        trail = on_top + active_rail.normal * 0.12 + lateral * side * 0.02
                        targets[i] = over_the_edge(i, trail)
                        toes[i] = fwd_h
                        states[i] = LegState.PLANT
                        rates[i] = 16.0
                    elif hip_above >= -0.15 and not lead:
                        # trailing leg: knee up, sole against the wall face, waiting to come over
                        t = hips[i] - UP * 0.45
                        out = config.FOOT_LEN * 0.75 + 0.02
                        d = float(np.dot(t - active_rail.p1, wall_n))
                        targets[i] = t + wall_n * max(0.0, out - d)
                        toes[i] = fwd_h * 0.7 - UP * 0.7
                        states[i] = LegState.SMEAR
                        rates[i] = 10.0
                    else:
                        targets[i] = smear_target(i) if wall_dist > 0.25 else hang_target(i)
                        toes[i] = fwd_h * 0.8 - UP * 0.6
                        states[i] = LegState.SMEAR
                        rates[i] = 10.0
                else:
                    # lowering over the edge: the lead foot goes over first and hangs; the other
                    # stays on the top while the body lowers, then follows into the hang
                    lead = i == self.push_leg
                    hip_above = float(hips[i][1] - on_top[1])
                    if lead or climb_phase >= 0.62 or hip_above < -0.30:
                        targets[i] = off_the_edge(i, hang_target(i, toe_in=0.10, bend=0.05))
                        toes[i] = fwd_h * 0.5 - UP * 0.85
                        states[i] = LegState.HANG
                        rates[i] = 8.0
                    else:
                        targets[i] = on_top
                        toes[i] = fwd_h
                        states[i] = LegState.PLANT
                        rates[i] = 14.0

        else:
            for i in range(2):
                targets[i] = stand_target(i) if on_ground else hang_target(i)
                toes[i] = fwd_h
                states[i] = LegState.STAND if on_ground else LegState.HANG
                rates[i] = 10.0

        # ------------------------------------------------ tuck (flips, rolls)
        # knees up to the chest, heels under the buttocks; the whole-body rotation is applied later
        if body_tuck > 1e-3:
            for i in range(2):
                side = -1.0 if i == 0 else 1.0
                t = hips[i] + fwd_h * 0.24 + right * side * 0.05 - UP * 0.20
                targets[i] = np.asarray(targets[i], dtype=np.float32) + (t - np.asarray(targets[i], dtype=np.float32)) * body_tuck
                toes[i] = fwd_h * 0.5 - UP * 0.85 if body_tuck > 0.5 else toes[i]
                states[i] = LegState.TUCK if body_tuck > 0.5 else states[i]
                rates[i] = 30.0

        # ------------------------------------------- smoothing, IK, palette
        knees, ankles, knee_angles, toe_dirs = [], [], [], []
        for i in range(2):
            side = -1.0 if i == 0 else 1.0
            tgt = np.asarray(targets[i], dtype=np.float32)
            if support_y is not None and states[i] in (LegState.STAND, LegState.SWING, LegState.PLANT, LegState.HANG):
                tgt = tgt.copy()
                tgt[1] = max(float(tgt[1]), support_y)        # never through the ground
            if states[i] == LegState.PLANT or (states[i] == LegState.STAND and not moving_on_ground):
                # world-fixed contact: ease in world space
                self.ankle_s[i] = self._ease(self.ankle_s[i], tgt, rates[i], dt)
                self.ankle_rel[i] = self.ankle_s[i] - hips[i]
            elif states[i] == LegState.STAND:
                self.ankle_s[i] = tgt                          # stance foot is ground-locked, no lag
                self.ankle_rel[i] = self.ankle_s[i] - hips[i]
            else:
                # a free leg moves with the body: ease relative to the hip so body motion never
                # drags the foot behind (which would pull the knee straight)
                self.ankle_rel[i] = self._ease(self.ankle_rel[i], tgt - hips[i], rates[i], dt)
                self.ankle_s[i] = hips[i] + self.ankle_rel[i]
            # anatomical hard limit: a foot is never carried higher than just above its own hip
            if float(self.ankle_s[i][1]) > float(hips[i][1]) + 0.12:
                self.ankle_s[i] = self.ankle_s[i].copy()
                self.ankle_s[i][1] = float(hips[i][1]) + 0.12
                self.ankle_rel[i] = self.ankle_s[i] - hips[i]
            self.toe_s[i] = self._ease(self.toe_s[i], toes[i], 12.0, dt)
            self.toe_s[i] /= (np.linalg.norm(self.toe_s[i]) + 1e-7)
            pole = fwd_h + UP * 0.35 + right * side * 0.12
            ik = solve_two_bone_ik(hips[i], self.ankle_s[i], config.THIGH_LEN, config.SHIN_LEN, pole, None,
                                   min_angle_deg=config.KNEE_MIN_ANGLE_DEG, max_angle_deg=config.KNEE_MAX_ANGLE_DEG)
            shin_axis = ik.wrist - ik.elbow
            toe = self._clamp_ankle(shin_axis, self.toe_s[i], fwd_h)
            knees.append(ik.elbow)
            ankles.append(ik.wrist)
            knee_angles.append(ik.elbow_angle_deg)
            toe_dirs.append(toe)
            thigh_bone, shin_bone, foot_bone = ((BoneID.L_THIGH, BoneID.L_SHIN, BoneID.L_FOOT) if i == 0
                                                else (BoneID.R_THIGH, BoneID.R_SHIN, BoneID.R_FOOT))
            # thigh and shin roll so their local +Z (front of the leg) faces the knee's direction
            bend_n = cross3(ik.elbow - hips[i], ik.wrist - ik.elbow)
            front = cross3(bend_n, ik.elbow - hips[i]) if np.linalg.norm(bend_n) > 1e-6 else fwd_h
            if float(np.dot(front, fwd_h)) < 0.0:
                front = -front
            self.skeleton.set_bone(thigh_bone, ArmSkeleton._frame(hips[i], ik.elbow - hips[i], front))
            self.skeleton.set_bone(shin_bone, ArmSkeleton._frame(ik.elbow, shin_axis, front))
            # foot: local +Y along the toes, local +Z up out of the boot
            foot_up = cross3(cross3(toe, UP), toe)
            if np.linalg.norm(foot_up) < 1e-6 or float(np.dot(foot_up, UP)) < 0.0:
                foot_up = UP
            self.skeleton.set_bone(foot_bone, ArmSkeleton._frame(ik.wrist, toe, foot_up))
            self.leg_state[i] = states[i]

        torso_origin = player_pos + UP * (config.HIP_HEIGHT - config.PLAYER_HALF_HEIGHT)
        self.skeleton.set_bone(BoneID.TORSO, ArmSkeleton._frame(torso_origin, UP, fwd_h))
        self.skeleton.set_bone(BoneID.PELVIS, ArmSkeleton._frame(torso_origin, UP, fwd_h))

        self.last = {
            "hips": (hips[0], hips[1]),
            "knees": (knees[0], knees[1]),
            "ankles": (ankles[0], ankles[1]),
            "knee_angles": (knee_angles[0], knee_angles[1]),
            "toe_dirs": (toe_dirs[0], toe_dirs[1]),
            "states": (states[0], states[1]),
            "targets": (np.asarray(targets[0], dtype=np.float32), np.asarray(targets[1], dtype=np.float32)),
            "ground_ankle_y": ground_ankle_y,
            "hip_drop": self.hip_drop,
            "push_leg": self.push_leg,
            "forward_h": fwd_h,
        }
