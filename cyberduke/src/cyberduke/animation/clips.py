"""Reusable procedural animation clips. Each clip maps a time/phase value to a HumanoidPose."""

from __future__ import annotations

import math

from cyberduke.animation.pose import HumanoidPose
from cyberduke.core.interp import lerp, smoothstep


def walk_animation(
    cycle: float, arm_swing_scale: float = 1.0, prop_holstered: bool = True
) -> HumanoidPose:
    """Universal humanoid walk cycle shared across guards, civilians, and thugs."""
    sin_l = math.sin(cycle)
    sin_r = math.sin(cycle + math.pi)

    hip_l = sin_l * 18.0
    hip_r = sin_r * 18.0
    knee_l = (sin_l * 38.0) if sin_l > 0 else (-sin_l * 15.0)
    knee_r = (sin_r * 38.0) if sin_r > 0 else (-sin_r * 15.0)
    bob = abs(math.sin(cycle * 2.0)) * 0.025
    arm_swing = sin_l * 16.0 * arm_swing_scale

    elbow_l = -(15.0 + abs(math.sin(cycle)) * 20.0)
    elbow_r = -(15.0 + abs(math.sin(cycle + math.pi)) * 20.0)

    prop_pos = [-0.28, 0.98, 0.0] if prop_holstered else [0.0, 0.0, 0.0]
    prop_rot = [80.0, 0.0, 0.0] if prop_holstered else [0.0, 0.0, 0.0]

    return HumanoidPose(
        bob=bob,
        arm_l=[arm_swing, 0.0, 0.0, elbow_l],
        arm_r=[-arm_swing, 0.0, 0.0, elbow_r],
        leg_l=[hip_l, 0.0, max(0.0, knee_l)],
        leg_r=[hip_r, 0.0, max(0.0, knee_r)],
        has_prop=prop_holstered,
        prop_pos=prop_pos,
        prop_rot=prop_rot,
    )


INSPECT_WEAPON_DURATION = 15.0
"""Authored length of ``inspect_weapon_animation`` in seconds (draw, inspect, scan, holster)."""


def inspect_weapon_animation(t: float, duration: float = INSPECT_WEAPON_DURATION) -> HumanoidPose:
    """Wall lean & 5-phase weapon inspection routine: draw, inspect, room scan, re-inspect,
    holster. Authored over 15 s; ``duration`` time-scales the whole routine so the weapon is
    back in the holster exactly when the caller's lean ends (fixes the visible snap when the
    guard used to walk off mid-holster)."""
    t = t * (INSPECT_WEAPON_DURATION / duration)
    leg_l = [0.0, 0.0, 0.0]
    leg_r = [14.0, 0.0, 12.0]

    arm_rest_l = [0.0, 0.0, 0.0, -15.0]
    arm_rest_r = [0.0, 0.0, 0.0, -15.0]
    arm_holster_r = [0.0, 3.0, 5.0, -20.0]

    arm_hold_l = [-8.0, -32.0, 5.0, -98.0]
    arm_hold_r = [0.0, 15.0, 25.0, -95.0]

    pos_holster = [-0.28, 0.98, 0.0]
    pos_chest = [-0.02, 1.28, 0.35]

    rot_holster = [80.0, 0.0, 0.0]
    rot_chest = [10.0, 75.0, 25.0]  # Observed long-ways on its side

    head_pitch, head_yaw = 0.0, 0.0

    if t < 0.5:
        u = smoothstep(0.0, 0.5, t)
        ar = [lerp(arm_rest_r[i], arm_holster_r[i], u) for i in range(4)]
        al = list(arm_rest_l)
        gun_pos = list(pos_holster)
        gun_rot = list(rot_holster)
        head_pitch = 25.0 * u
        head_yaw = -20.0 * u

    elif t < 2.0:
        u = smoothstep(0.5, 2.0, t)
        y_arc = math.sin(u * math.pi) * 0.08
        gun_pos = [lerp(pos_holster[i], pos_chest[i], u) for i in range(3)]
        gun_pos[1] += y_arc
        gun_rot = [lerp(rot_holster[i], rot_chest[i], u) for i in range(3)]

        ar = [lerp(arm_holster_r[i], arm_hold_r[i], u) for i in range(4)]
        u_l = smoothstep(0.8, 2.0, t)
        al = [lerp(arm_rest_l[i], arm_hold_l[i], u_l) for i in range(4)]

        head_pitch = 25.0 + (32.0 - 25.0) * u
        head_yaw = -20.0 * (1.0 - u)

    elif t < 6.5:
        cycle = (t - 2.0) * 1.6
        tilt = math.sin(cycle) * 8.0
        gun_pos = list(pos_chest)
        gun_rot = [
            rot_chest[0] + math.cos(cycle) * 3.0,
            rot_chest[1] + math.sin(cycle) * 5.0,
            rot_chest[2] + tilt,
        ]

        ar = [arm_hold_r[0], arm_hold_r[1], arm_hold_r[2], arm_hold_r[3] + tilt * 0.3]
        al = [arm_hold_l[0], arm_hold_l[1], arm_hold_l[2], arm_hold_l[3] - tilt * 0.3]
        head_pitch = 30.0 + math.sin(cycle) * 3.0
        head_yaw = math.sin(cycle) * 7.0

    elif t < 10.5:
        scan_t = t - 6.5
        if scan_t < 0.8:
            u = smoothstep(0.0, 0.8, scan_t)
            head_pitch = 30.0 * (1.0 - u)
            head_yaw = -40.0 * u
        elif scan_t < 2.4:
            u = smoothstep(0.8, 2.4, scan_t)
            head_pitch = 0.0
            head_yaw = -40.0 + 80.0 * u
        elif scan_t < 3.5:
            u = smoothstep(2.4, 3.5, scan_t)
            head_pitch = 0.0
            head_yaw = 40.0 - 40.0 * u
        else:
            head_pitch, head_yaw = 0.0, 0.0

        gun_pos = list(pos_chest)
        gun_rot = list(rot_chest)
        ar = list(arm_hold_r)
        al = list(arm_hold_l)

    elif t < 12.8:
        u = smoothstep(10.5, 11.2, t)
        cycle = (t - 10.5) * 2.0
        tilt = math.sin(cycle) * 6.0
        gun_pos = list(pos_chest)
        gun_rot = [rot_chest[0], rot_chest[1] + math.sin(cycle) * 4.0, rot_chest[2] + tilt]
        ar = [arm_hold_r[0], arm_hold_r[1], arm_hold_r[2], arm_hold_r[3] + tilt * 0.3]
        al = [arm_hold_l[0], arm_hold_l[1], arm_hold_l[2], arm_hold_l[3] - tilt * 0.3]
        head_pitch = 32.0 * u
        head_yaw = math.sin(cycle) * 5.0

    else:
        u = smoothstep(12.8, 14.6, t)
        gun_pos = [lerp(pos_chest[i], pos_holster[i], u) for i in range(3)]
        gun_rot = [lerp(rot_chest[i], rot_holster[i], u) for i in range(3)]

        u_l = smoothstep(12.8, 13.8, t)
        al = [lerp(arm_hold_l[i], arm_rest_l[i], u_l) for i in range(4)]
        ar = [lerp(arm_hold_r[i], arm_holster_r[i], u) for i in range(4)]

        head_pitch = 32.0 + (20.0 - 32.0) * u
        head_yaw = -20.0 * u

        if t > 14.6:
            u_fin = smoothstep(14.6, 15.0, t)
            ar = [lerp(arm_holster_r[i], arm_rest_r[i], u_fin) for i in range(4)]
            head_pitch = 20.0 * (1.0 - u_fin)
            head_yaw = -20.0 * (1.0 - u_fin)

    drawn = 0.5 <= t < 14.6  # the weapon out of the holster, in the right hand
    return HumanoidPose(
        bob=0.0,
        head_pitch=head_pitch,
        head_yaw=head_yaw,
        arm_l=al,
        arm_r=ar,
        leg_l=leg_l,
        leg_r=leg_r,
        has_prop=True,
        prop_pos=gun_pos,
        prop_rot=gun_rot,
        grip_r="fist" if drawn else None,
        grip_l="open" if 0.8 <= t < 13.8 else None,
    )


def investigate_animation(t: float) -> HumanoidPose:
    """Standing inspection of something at ground/waist level: head tilted down, slowly
    looking side to side (used by guards at a distraction site)."""
    return HumanoidPose(
        head_pitch=12.0 + math.sin(t * 2.0) * 6.0,
        head_yaw=math.sin(t * 1.5) * 20.0,
    )


def blinded_animation(t: float) -> HumanoidPose:
    """Optics jammed: both hands clamped over the visor - palms on it, the fingers spread
    up over the brow, the elbows out to the sides - the head jerking about trying to shake
    the static out (the hands going with it), the stance staggering. Loops for as long as
    the blindness lasts."""
    from cyberduke.animation import anatomy
    from cyberduke.animation.reach import _rx, _ry, _rz, arm_to

    jerk = math.sin(t * 9.0) * 10.0
    stagger = math.sin(t * 3.0) * 0.01
    pose = HumanoidPose(
        bob=stagger,
        head_pitch=14.0 + math.sin(t * 5.0) * 6.0,
        head_yaw=jerk,
        head_roll=math.sin(t * 4.0) * 6.0,
        grip_l="face",  # palms over the visor
        grip_r="face",
        leg_l=[6.0, 4.0, 6.0],
        leg_r=[-8.0, -4.0, 10.0],
        has_prop=True,
        prop_pos=[-0.28, 0.98, 0.0],
        prop_rot=[80.0, 0.0, 0.0],
    )
    # where the hands go: just in front of the visor either side of the nose, in the
    # head's frame - turned with the head wherever it jerks
    turn = _ry(pose.head_yaw) @ _rx(pose.head_pitch) @ _rz(pose.head_roll)
    head = (0.0, anatomy.HEAD_Y + pose.bob, 0.0)
    for side, x in (("l", 0.07), ("r", -0.07)):
        y = anatomy.EYE_LINE - 0.02
        on_face = (x, y, anatomy.surface_z(x, y) + BLIND_HAND_OUT)
        target = [head[k] + float((turn @ on_face)[k]) for k in range(3)]
        out = 1.0 if side == "l" else -1.0
        angles = arm_to(target, side, bob=pose.bob, elbow=(out, -0.5, 0.35))
        if side == "l":
            pose.arm_l = angles
        else:
            pose.arm_r = angles
    return pose


BLIND_HAND_OUT = 0.06  # the hand point this far in front of the face (the palm on the visor)


def electrocuted_animation(t: float) -> HumanoidPose:
    """Current through the cyberware: rigid, on its feet, every joint twitching. Arms
    locked half-out and shaking, head snapping about, knees juddering."""
    fast = math.sin(t * 31.0)
    faster = math.sin(t * 47.0 + 1.3)
    return HumanoidPose(
        bob=0.01 * fast,
        head_pitch=-10.0 + 9.0 * faster,
        head_yaw=14.0 * fast,
        head_roll=10.0 * faster,
        arm_l=[-40.0 + 12.0 * fast, -20.0, 35.0 + 8.0 * faster, -30.0 + 15.0 * fast],
        arm_r=[-40.0 - 12.0 * fast, 20.0, -35.0 - 8.0 * faster, -30.0 - 15.0 * fast],
        # the fingers clawed and twitching with the current
        fingers_l=tuple(0.55 + 0.3 * math.sin(t * 37.0 + k * 1.9) for k in range(5)),  # type: ignore[arg-type]
        fingers_r=tuple(0.55 + 0.3 * math.sin(t * 41.0 + k * 2.3) for k in range(5)),  # type: ignore[arg-type]
        leg_l=[-4.0, 3.0, 14.0 + 8.0 * fast],
        leg_r=[4.0, -3.0, 14.0 - 8.0 * fast],
        has_prop=True,
        prop_pos=[-0.28, 0.98, 0.0],
        prop_rot=[80.0, 0.0, 0.0],
    )


def bang_machine_animation(t: float) -> HumanoidPose:
    """Hitting a misbehaving machine: the right fist comes up and slams down about once a
    second, the left hand braces on the cabinet, the head is down on the problem and the
    body leans into each hit."""
    cycle = (t * 1.15) % 1.0
    # quick wind-up (0..0.3), sharp slam (0.3..0.45), hold (0.45..1)
    if cycle < 0.3:
        swing = smoothstep(0.0, 0.3, cycle)  # 0 -> 1 raised
    elif cycle < 0.45:
        swing = 1.0 - smoothstep(0.3, 0.45, cycle)  # slam down
    else:
        swing = 0.0
    raised_pitch, down_pitch = -120.0, -55.0
    lean = 0.012 * (1.0 - swing)
    return HumanoidPose(
        bob=-lean,
        head_pitch=22.0 + 6.0 * (1.0 - swing),
        head_yaw=math.sin(t * 0.9) * 6.0,
        arm_l=[-70.0, -15.0, 8.0, -35.0],  # braced on the cabinet
        arm_r=[lerp(down_pitch, raised_pitch, swing), 10.0, -8.0, -70.0 - 30.0 * swing],
        grip_l="flat",
        grip_r="fist",
        leg_l=[-6.0, 0.0, 8.0],
        leg_r=[10.0, 0.0, 4.0],
        has_prop=True,
        prop_pos=[-0.28, 0.98, 0.0],
        prop_rot=[80.0, 0.0, 0.0],
        expression="grimace",  # (teeth gritted at it)
    )


def idle_animation(t: float, seed: float = 0.0) -> HumanoidPose:
    """Standing about: weight shifting from foot to foot, arms loose, looking around."""
    s = t + seed
    shift = math.sin(s * 0.8)
    return HumanoidPose(
        bob=0.004 * math.sin(s * 1.6),
        head_pitch=4.0 + 3.0 * math.sin(s * 0.7),
        head_yaw=22.0 * math.sin(s * 0.45) + 8.0 * math.sin(s * 1.9),
        head_roll=2.0 * math.sin(s * 0.6),
        arm_l=[4.0 * shift, -4.0, 6.0 + 3.0 * shift, -18.0 - 6.0 * math.sin(s * 1.1)],
        arm_r=[-4.0 * shift, 4.0, -6.0 + 3.0 * shift, -18.0 - 6.0 * math.sin(s * 1.3)],
        leg_l=[2.0 * shift, 2.0, max(0.0, 6.0 * shift)],
        leg_r=[-2.0 * shift, -2.0, max(0.0, -6.0 * shift)],
    )


def cook_animation(t: float) -> HumanoidPose:
    """Working a wok: the right hand tosses (a quick lift and drop about once a second),
    the left holds the ladle, the head stays down on the food with an occasional glance up."""
    cycle = (t * 1.1) % 1.0
    toss = smoothstep(0.0, 0.25, cycle) * (1.0 - smoothstep(0.3, 0.55, cycle))
    return HumanoidPose(
        bob=-0.01 * toss,
        head_pitch=26.0 - 20.0 * (math.sin(t * 0.35) > 0.92),
        head_yaw=6.0 * math.sin(t * 0.9),
        arm_l=[-55.0, -20.0, 10.0, -80.0 - 10.0 * math.sin(t * 2.2)],
        arm_r=[-45.0 - 35.0 * toss, 15.0, -10.0, -70.0 - 20.0 * toss],
        grip_l="fist",  # the ladle
        grip_r="fist",  # the wok's handle
        leg_l=[0.0, 2.0, 4.0],
        leg_r=[0.0, -2.0, 6.0],
    )


def hawker_animation(t: float) -> HumanoidPose:
    """Selling: one hand on the table, leaning forward, the head swinging along the alley
    looking for customers - and now and then the other hand up beside the head, the palm
    to the front, waving them over."""
    from cyberduke.animation.reach import arm_to

    beckon = 0.5 + 0.5 * math.sin(t * 0.9)
    up = smoothstep(0.55, 0.85, beckon)
    wave = math.sin(t * 6.0) * up
    bob = 0.004 * math.sin(t * 1.3)
    rest = (-0.3, 0.88 + bob, 0.12)
    raised = (-0.36 + 0.07 * wave, 1.93 + bob, 0.26)
    right = tuple(rest[k] + (raised[k] - rest[k]) * up for k in range(3))
    return HumanoidPose(
        bob=bob,
        head_pitch=-4.0,
        head_yaw=35.0 * math.sin(t * 0.5),
        arm_l=[-40.0, -10.0, 15.0, -60.0],
        arm_r=arm_to(right, "r", bob=bob, elbow=(-1.0, -0.6 + 0.15 * up, 0.1)),
        grip_l="flat",  # on the table
        grip_r="wave" if up > 0.3 else None,
        leg_l=[-4.0, 3.0, 6.0],
        leg_r=[5.0, -3.0, 8.0],
    )
