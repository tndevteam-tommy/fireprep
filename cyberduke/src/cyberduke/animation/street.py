"""Everyday gestures for the people of the street: ordering, paying, eating, smoking,
chopping, wrapping, carrying, knocking, browsing...

Every clip puts the hands where the task needs them (``animation.reach.arm_to``) rather
than swinging joints about: a hand goes to the mouth, onto the counter, to the chin. All
positions are in the body's own frame (feet at the origin, facing +z, the body's left
along +x); a few landmarks below. Clips take the time into the gesture (and whatever
they need) and return a ``HumanoidPose``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cyberduke.animation import anatomy
from cyberduke.animation.clips import walk_animation
from cyberduke.animation.pose import HumanoidPose
from cyberduke.animation.reach import arm_to
from cyberduke.core.interp import lerp, smoothstep

Vec = tuple[float, float, float]

MOUTH: Vec = (0.0, 1.87, 0.2)
CHIN: Vec = (0.0, 1.8, 0.22)
CHEST: Vec = (0.0, 1.3, 0.32)
LOOK: Vec = (0.0, 1.78, 0.42)  # something held up to look at
COUNTER_Y = 1.03  # the stall counters' tops (the junk table is a little lower)
REST_L: Vec = (0.3, 0.86, 0.08)  # a hand hanging loose at the side (elbow a little bent)
REST_R: Vec = (-0.3, 0.86, 0.08)
POCKET_L: Vec = (0.27, 0.93, 0.1)  # a hand in the front pocket
SEAT_BOB = -0.27  # sitting on a stool: how far the upper body drops
PARCEL_GRIP = 0.18  # hands either side of a parcel (0.34 wide), this far from its middle
CARRY_TWO: Vec = (0.0, 1.12, 0.33)  # the middle of something carried in both hands
# Things carried in both hands are held by their sides, this far up from the underside
# (``render.items.held_transform``; one hand holds a small thing from below: 2 cm).
TWO_HANDED_GRIP = {"parcel": 0.1, "phone": -0.056}  # (negative: up off the hands)


def _mix(a: Sequence[float], b: Sequence[float], u: float) -> tuple[float, ...]:
    return tuple(lerp(x, y, u) for x, y in zip(a, b, strict=True))


def _at(p: Vec, bob: float) -> Vec:
    return (p[0], p[1] + bob, p[2])


def _stance(t: float, seed: float = 0.0) -> HumanoidPose:
    """Standing with the weight shifting now and then: legs, hips, a little bob."""
    shift = math.sin((t + seed) * 0.7)
    return HumanoidPose(
        bob=0.004 * math.sin((t + seed) * 1.5),
        leg_l=[2.0 * shift, 2.0, max(0.0, 6.0 * shift)],
        leg_r=[-2.0 * shift, -2.0, max(0.0, -6.0 * shift)],
    )


def _arms(pose: HumanoidPose, left: Vec | None, right: Vec | None) -> HumanoidPose:
    if left is not None:
        pose.arm_l = arm_to(_at(left, pose.bob), "l", bob=pose.bob)
    if right is not None:
        pose.arm_r = arm_to(_at(right, pose.bob), "r", bob=pose.bob)
    return pose


# ----------------------------------------------------------------------------- the mouth
SYLLABLES = ("talk", "talk_o", "mumble", "talk", "neutral", "talk_o", "talk", "mumble", "talk")


def speech(t: float, seed: float = 0.0, rate: float = 5.5) -> str:
    """The mouth of somebody talking, ``t`` seconds in: a syllable at a time - open on a
    vowel, round on an "o", nearly shut on a consonant - and a breath now and then."""
    s = t + seed * 0.37
    if (s % 3.3) > 2.75:  # (a pause for breath)
        return "neutral"
    k = int(s * rate + seed * 3.0)
    return SYLLABLES[(k * 5 + k // 3) % len(SYLLABLES)]


def mutter(t: float, seed: float = 0.0) -> str:
    """Mumbling to oneself (weighing up the menu, thinking it over): lips barely parted,
    on and off."""
    s = t + seed * 0.29
    if (s % 2.6) > 1.7:
        return "neutral"
    return ("mumble", "mumble", "talk_o", "mumble", "neutral")[int(s * 4.0) % 5]


def idle(t: float, seed: float = 0.0) -> HumanoidPose:
    """Waiting about: weight shifting, hands loose, looking around."""
    pose = _stance(t, seed)
    s = t + seed
    pose.head_pitch = 3.0 + 3.0 * math.sin(s * 0.7)
    pose.head_yaw = 20.0 * math.sin(s * 0.45) + 6.0 * math.sin(s * 1.7)
    sway = 0.02 * math.sin(s * 0.9)
    return _arms(pose, (0.3, 0.86 + sway, 0.09), (-0.3, 0.86 - sway, 0.09))


def talk(t: float, seed: float = 0.0, listening: bool = False) -> HumanoidPose:
    """In conversation: the head nods along, the hands talk (less so when listening)."""
    pose = _stance(t, seed)
    s = t + seed
    pose.head_pitch = 2.0 + 4.0 * math.sin(s * 2.3) * (0.4 if listening else 1.0)
    pose.head_yaw = 6.0 * math.sin(s * 0.8)
    pose.head_roll = 3.0 * math.sin(s * 0.5)
    if listening:
        return _arms(pose, (0.14, 1.05, 0.22), (-0.14, 1.05 + 0.02 * math.sin(s), 0.22))
    pose.expression = speech(t, seed)
    beat = math.sin(s * 3.1) * 0.5 + 0.5
    # the hands talk: open, the fingers loosening and closing with the beat
    open_r = 0.08 + 0.22 * (1.0 - beat)
    pose.grip_l = pose.grip_r = "open"
    pose.fingers_r = (0.1, open_r, open_r + 0.04, open_r + 0.08, open_r + 0.12)
    right = (-0.16 + 0.08 * math.sin(s * 1.7), 1.28 + 0.1 * beat, 0.36 + 0.05 * math.sin(s * 2.3))
    left_up = smoothstep(0.55, 0.9, 0.5 + 0.5 * math.sin(s * 0.6))
    left = _mix((0.22, 1.0, 0.12), (0.16, 1.22 + 0.06 * beat, 0.34), left_up)
    return _arms(pose, left, right)  # type: ignore[arg-type]


def point_at(t: float, direction: Vec, seed: float = 0.0) -> HumanoidPose:
    """Pointing (at a menu board, up the alley): the right arm out along ``direction``
    (body frame), the head looking the same way."""
    pose = _stance(t, seed)
    d = math.sqrt(sum(c * c for c in direction)) or 1.0
    u = smoothstep(0.0, 0.4, t)
    sx, sy = -0.32, 1.58 + pose.bob
    far = (sx + direction[0] / d * 0.74, sy + direction[1] / d * 0.74, direction[2] / d * 0.74)
    pose.arm_r = arm_to(_mix(REST_R, far, u), "r", bob=pose.bob)  # type: ignore[arg-type]
    pose.arm_l = arm_to(REST_L, "l", bob=pose.bob)
    pose.grip_r = "point"
    pose.head_yaw = math.degrees(math.atan2(direction[0], direction[2])) * u
    pose.head_pitch = (
        -math.degrees(math.atan2(direction[1], math.hypot(direction[0], direction[2]))) * u
    )
    pose.expression = mutter(t, seed)  # (reading it off under their breath)
    return pose


def reach(
    u: float, target: Vec, side: str = "r", t: float = 0.0, seed: float = 0.0
) -> HumanoidPose:
    """A hand going out to ``target`` (``u`` 0..1 there; 1..0 coming back): handing a box
    over a counter, taking one, paying, putting something down. The head watches it."""
    pose = _stance(t, seed)
    rest = REST_R if side == "r" else REST_L
    hand = _mix(rest, target, smoothstep(0.0, 1.0, u))
    other = REST_L if side == "r" else REST_R
    _arms(pose, *((other, hand) if side == "r" else (hand, other)))  # type: ignore[arg-type]
    # an empty hand opens as it goes out for something (and closes on it: what it takes
    # has its own grip)
    k = smoothstep(0.2, 0.8, u)
    curl = 0.3 - 0.22 * k
    fingers = (0.2 - 0.12 * k, curl, curl + 0.03, curl + 0.06, curl + 0.09)
    if side == "r":
        pose.fingers_r = fingers
    else:
        pose.fingers_l = fingers
    pose.head_pitch = 18.0 * u + 4.0
    pose.head_yaw = 8.0 * (1 if side == "l" else -1) * u
    return pose


def offer(u: float, target: Vec, t: float = 0.0) -> HumanoidPose:
    """Both hands out together with something (a box, a parcel) to ``target``."""
    pose = _stance(t)
    k = smoothstep(0.0, 1.0, u)
    mid = _mix(CARRY_TWO, target, k)
    pose.arm_l = arm_to(_at((mid[0] + PARCEL_GRIP, mid[1], mid[2]), pose.bob), "l", bob=pose.bob)
    pose.arm_r = arm_to(_at((mid[0] - PARCEL_GRIP, mid[1], mid[2]), pose.bob), "r", bob=pose.bob)
    pose.head_pitch = 10.0 + 14.0 * k
    return pose


def hold_up(t: float, seed: float = 0.0) -> HumanoidPose:
    """Holding something up in front of the face to look at it - turning it in the light."""
    pose = _stance(t, seed)
    lift = smoothstep(0.0, 0.6, t)
    turn = 0.04 * math.sin(t * 1.6)
    right = _mix(CHEST, (LOOK[0] - 0.05 + turn, LOOK[1], LOOK[2]), lift)
    left = _mix(REST_L, (LOOK[0] + 0.07 + turn, LOOK[1] - 0.06, LOOK[2] - 0.02), lift)
    pose = _arms(pose, left, right)  # type: ignore[arg-type]
    pose.head_pitch = -6.0 * lift + 2.0 * math.sin(t * 1.1)
    pose.head_yaw = 8.0 * math.sin(t * 0.8)
    pose.head_roll = 6.0 * math.sin(t * 0.6) * lift
    pose.expression = "smile" if lift > 0.5 else None  # (pleased with it)
    return pose


SUSHI_PIECES = 4  # in a takeaway box of nigiri
NOODLE_BITES = 4  # chopstickfuls in a box of chow mein


def sushi_piece_at(j: int) -> Vec:
    """Where piece ``j`` of the sushi sits in its open box (the box's frame: the middle
    of its underside at the origin), at the top of the piece."""
    return (-0.06 + 0.04 * j, 0.077, 0.02)


def noodles_top(bites: int) -> float:
    """How high the chow mein comes up a noodle box with ``bites`` left in it."""
    return 0.02 + 0.11 * bites / NOODLE_BITES


def food_in(dish: str) -> list[Vec]:
    """Where each bite is taken from, in order, in the box of ``dish`` ("sushi" eaten
    from the right-hand end; "noodles" from the top of what is left)."""
    if dish == "sushi":
        return [
            (sushi_piece_at(j)[0], sushi_piece_at(j)[1] - 0.012, sushi_piece_at(j)[2])
            for j in reversed(range(SUSHI_PIECES))
        ]
    return [
        (0.012 * (-1) ** k, noodles_top(NOODLE_BITES - k) - 0.01, 0.0) for k in range(NOODLE_BITES)
    ]


BITE = 3.2  # one bite: chopsticks into the box, up to the mouth, chew (seconds)
PICK_AT = 0.6  # ... the food is taken up in the chopsticks this far into it
EAT_AT = 1.45  # ... and eaten this far into it
BOX_HOLD: Vec = (0.06, 1.24, 0.34)  # the box held in the left hand, under the chin
CHOPSTICK_LENGTH = 0.2  # from the fingers to the tips (render.items.CHOPSTICKS)
_DOWN: Vec = (0.3, -0.85, 0.35)  # chopsticks dipping into the box, tips down and in
_UP: Vec = (0.42, 0.28, -0.55)  # ... up to the mouth, tips back towards it


def _unit(v: Sequence[float]) -> Vec:
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def eat(
    t: float,
    bob: float = 0.0,
    seed: float = 0.0,
    food: Sequence[Vec] | None = None,
) -> HumanoidPose:
    """Eating out of a box held in the left hand with chopsticks: each ``BITE`` seconds
    the tips go down into the box to the next bite (``food[k]``: where bite ``k`` is, in
    the box's own frame - the middle of the box's underside at the origin; by default
    the middle of the box), pick it up, carry it up to the mouth (the sticks turning from
    pointing down into the box to pointing back at the lips), and come down again to
    chew. The chopsticks' tips are what goes to the food and the mouth - the hand is a
    chopstick's length behind them. ``pose.aim_r`` is where the tips are."""
    from cyberduke.animation.reach import hand as hand_of

    pose = _stance(t, seed) if bob == 0.0 else HumanoidPose(bob=bob)
    k = int(t // BITE)
    cycle = t - k * BITE
    bite = food[min(k, len(food) - 1)] if food else (0.0, 0.07, 0.0)
    # the box in the left hand (a level thing sits 2 cm under the hand)
    pose.arm_l = arm_to(_at(BOX_HOLD, bob), "l", bob=bob)
    hold, _ = hand_of(pose, "l")
    food_at = (hold[0] + bite[0], hold[1] - 0.02 + bite[1], hold[2] + bite[2])
    above = (food_at[0] - 0.02, food_at[1] + 0.06, food_at[2] - 0.01)
    mouth = _at(MOUTH, bob)
    if cycle < 0.45:  # down into the box
        tip, d = _mix(above, food_at, smoothstep(0.0, 0.45, cycle)), _DOWN
    elif cycle < PICK_AT + 0.1:  # taking hold of it
        tip, d = food_at, _DOWN
    elif cycle < 1.3:  # up to the mouth, the sticks turning
        u = smoothstep(PICK_AT + 0.1, 1.3, cycle)
        lift = 0.05 * math.sin(u * math.pi)
        tip = _mix(food_at, mouth, u)
        tip = (tip[0], tip[1] + lift, tip[2] + lift)
        d = _mix(_unit(_DOWN), _unit(_UP), u)
    elif cycle < 1.6:  # in
        tip, d = mouth, _UP
    elif cycle < 2.3:  # back down over the box
        u = smoothstep(1.6, 2.3, cycle)
        tip, d = _mix(mouth, above, u), _mix(_unit(_UP), _unit(_DOWN), u)
    else:  # chewing, the sticks ready over the box
        tip, d = above, _DOWN
    d = _unit(d)
    hand = tuple(tip[i] - d[i] * CHOPSTICK_LENGTH for i in range(3))
    pose.arm_r = arm_to(hand, "r", bob=bob)
    pose.aim_r = tuple(tip)  # type: ignore[assignment]
    chewing = cycle >= 1.6
    if 0.95 < cycle < 1.6:  # open wide for it
        pose.expression = "eat"
    elif chewing and cycle < BITE - 0.2:  # and chew
        pose.expression = "mumble" if int(t * 6.0) % 2 else "neutral"
    looking_down = cycle < 0.8 or cycle > 2.2
    pose.head_pitch = (24.0 if looking_down else 3.0) + (
        2.5 * math.sin(t * 14.0) if chewing else 0.0
    )
    pose.head_yaw = 5.0 * math.sin(t * 0.4 + seed)
    return pose


def flicks_after(puff: int) -> bool:
    """Whether the ash is flicked off after puff number ``puff``: after every second or
    third one (2, 3, 2, 3...)."""
    return puff % 5 in (1, 4)


SMOKE_CYCLE = 6.0  # one puff: up to the lips, a drag, down, and a while by the hip
FLICK = (3.0, 4.6)  # ... when in it the ash is flicked off (after some puffs)


DRAG = (0.8, 2.2)  # in a puff, the filter at the lips and a long pull on it
EXHALE = (2.35, 3.9)  # ... then, the cigarette on its way down, the smoke blown out


def smoke_state(cycle: float) -> tuple[float, bool, float]:
    """(ember, inhaling, exhale) ``cycle`` seconds into a puff: the cherry flares as the
    drag starts and cools off after it; nothing curls off the tip while it is drawn on;
    the lungful comes out in a puff once the cigarette is away from the lips."""
    d0, d1 = DRAG
    ember = smoothstep(d0, d0 + 0.35, cycle) * (1.0 - smoothstep(d1, d1 + 0.9, cycle))
    inhaling = d0 + 0.1 < cycle < d1
    e0, e1 = EXHALE
    exhale = 0.0
    if e0 < cycle < e1:
        u = (cycle - e0) / (e1 - e0)
        exhale = smoothstep(0.0, 0.15, u) * (1.0 - smoothstep(0.45, 1.0, u))
    return ember, inhaling, exhale


def smoke(t: float, seed: float = 0.0) -> HumanoidPose:
    """A cigarette: up to the lips every few seconds, a long drag, back down by the hip;
    the other hand in a pocket. The cigarette sticks out from between the fingers away
    from the face (``render.items``: "outward"), turning with the hand; after every
    second or third puff the hand goes out to the side and the thumb flicks the ash off -
    three taps, the cigarette rolling in the fingers with them (``pose.tilt_r``)."""
    pose = _stance(t, seed)
    puff = int((t + seed) // SMOKE_CYCLE)
    cycle = (t + seed) - puff * SMOKE_CYCLE
    low = (-0.2, 1.12, 0.2)
    lips = (MOUTH[0] - 0.035, MOUTH[1] - 0.01, MOUTH[2] + 0.035)  # the filter at the lips
    aside = (-0.28, 1.08, 0.3)  # held out to one side, clear of the clothes
    tilt = 0.0
    if cycle < 0.8:
        right = _mix(low, lips, smoothstep(0.0, 0.8, cycle))
    elif cycle < 2.2:
        right = lips
    elif cycle < 3.0:
        right = _mix(lips, low, smoothstep(2.2, 3.0, cycle))
    elif flicks_after(puff) and cycle < FLICK[1]:
        out = smoothstep(FLICK[0], FLICK[0] + 0.3, cycle) * (
            1.0 - smoothstep(FLICK[1] - 0.3, FLICK[1], cycle)
        )
        taps = FLICK[0] + 0.3 < cycle < FLICK[1] - 0.3
        tap = max(0.0, math.sin((cycle - FLICK[0] - 0.3) * 2.0 * math.pi * 3.0)) if taps else 0.0
        right = _mix(low, aside, out)
        right = (right[0], right[1] - 0.025 * tap, right[2])
        tilt = out * (-35.0 + 30.0 * tap)
    else:
        right = low
    pose = _arms(pose, POCKET_L, right)  # type: ignore[arg-type]
    pose.tilt_r = tilt
    if tilt:  # the thumb taps the filter to knock the ash off
        tap = 0.5 + 0.5 * math.sin((cycle - FLICK[0]) * 2.0 * math.pi * 3.0)
        pose.fingers_r = (0.25 + 0.45 * tap, 0.16, 0.2, 0.62, 0.72)
    drag = 0.8 < cycle < 2.2
    pose.ember, pose.inhaling, pose.exhale = smoke_state(cycle)
    if DRAG[0] - 0.1 < cycle < DRAG[1]:  # the lips pursed round the filter
        pose.expression = "pursed"
    elif pose.exhale > 0.05:  # the smoke blown out through rounded lips
        pose.expression = "talk_o"
    flicking = tilt != 0.0
    pose.head_pitch = -8.0 if drag else (14.0 if flicking else 2.0 + 3.0 * math.sin(t * 0.5))
    # the head comes round to the cigarette as it comes up to the lips, faces it through
    # the drag, and is free again once it goes back down (glancing at passers-by: the
    # pose's ``head_held`` tells whoever turns the head to look not to, meanwhile)
    held = smoothstep(0.0, 0.55, cycle) * (1.0 - smoothstep(DRAG[1] + 0.1, 2.9, cycle))
    roam = -18.0 if flicking else 25.0 * math.sin(t * 0.23 + seed)
    pose.head_yaw = roam * (1.0 - held)
    pose.head_held = held
    return pose


PHONE_AT: Vec = (0.0, 1.38, 0.3)  # the middle of a phone held up in both hands
PHONE_TILT = 38.0  # degrees its far end is tipped up towards the face
PHONE_HALF = 0.047  # half its width: a hand at either side of it
PHONE_STAGGER = 0.012  # the right hand this much further up the phone, the left down it
PHONE_THUMB = 0.25  # how curled the thumbs are (tapping: more)
# (index, middle, ring, little): curled in under the phone, which rests up on them - an
# inch higher than the hands' middle (``TWO_HANDED_GRIP``): the index and middle fingers
# used to curl straight through it (``tests/test_hands.py`` holds it out of every finger)
PHONE_FINGERS = (0.5, 0.4, 0.45, 0.3)


def phone(t: float, seed: float = 0.0) -> HumanoidPose:
    """Head down over a phone held up in both hands (``pose.hand_l`` and ``hand_r`` both on
    it: ``hands.TWO_HANDED``) - a hand at either side of it, the fingers curled in under
    it, the phone resting up on them, tipped up towards the face - the thumbs going by its
    edges, now one, now the other, the phone moving a little with them."""
    pose = _stance(t, seed)
    x, y, z = PHONE_AT
    sway = 0.006 * math.sin(t * 1.3 + seed)
    tap = 0.004 * math.sin(t * 11.0)
    # (one hand a little further up the phone than the other: their fingers, round its
    # back, lie side by side instead of meeting)
    a = math.radians(PHONE_TILT)
    up, on = PHONE_STAGGER * math.sin(a), PHONE_STAGGER * math.cos(a)
    pose = _arms(
        pose,
        (x + PHONE_HALF + sway, y + tap - up, z - on),
        (x - PHONE_HALF + sway, y - tap + up, z + on),
    )
    pose.hand_l = pose.hand_r = "phone"  # (a phone of their own, unless they hold something)
    pose.tilt_r = PHONE_TILT
    right = PHONE_THUMB + 0.3 * max(0.0, math.sin(t * 11.0))
    left = PHONE_THUMB + 0.3 * max(0.0, math.sin(t * 9.0 + 1.7))
    pose.fingers_r = (right, *PHONE_FINGERS)
    pose.fingers_l = (left, *PHONE_FINGERS)
    pose.head_pitch = 32.0
    pose.head_yaw = 3.0 * math.sin(t * 0.7)
    return pose


SIP = 3.4  # one sip: the can up to the lips, a swallow or two, down again, a breather
CAN_LOW: Vec = (-0.19, 1.2, 0.3)  # the can held at the chest between sips
CAN_HALF = 0.06  # the can's middle (where it is held) to its rim
ELBOW_DOWN: Vec = (-0.45, -1.0, -0.35)  # the right elbow relaxed, down by the side ...
ELBOW_OUT: Vec = (-1.0, -0.55, 0.15)  # ... and out to the side, not tucked in
SIP_TILT = 55.0  # degrees the can is tipped up towards the face, at the lips


def _mouth(head_pitch: float, bob: float = 0.0) -> Vec:
    """Where the lips are with the head tipped back (``head_pitch`` < 0) or forward."""
    a = math.radians(head_pitch)
    pivot = anatomy.HEAD_Y  # (the head turns about its middle)
    y, z = MOUTH[1] - pivot, MOUTH[2] - 0.04  # (the lips' surface, about the head's middle)
    return (
        MOUTH[0],
        pivot + bob + y * math.cos(a) - z * math.sin(a),
        y * math.sin(a) + z * math.cos(a),
    )


def drink(t: float, seed: float = 0.0) -> HumanoidPose:
    """Drinking from a can in the right hand: up to the lips with the elbow out to the
    side, the can tipped up in the fingers (``pose.tilt_r``: the hand turns it about the
    forearm - ``animation.hands``) a little more over the swallow, the head going back
    with it; down to the chest again."""
    from cyberduke.animation.hands import held_axis

    pose = _stance(t, seed)
    cycle = t % SIP
    up = smoothstep(0.0, 0.75, cycle) * (1.0 - smoothstep(1.8, 2.4, cycle))
    swallow = smoothstep(0.75, 1.5, cycle) * (1.0 - smoothstep(1.6, 2.0, cycle))
    pose.head_pitch = 4.0 - 14.0 * up - 10.0 * swallow
    pose.head_yaw = 3.0 * math.sin(t * 0.4 + seed) * (1.0 - up)
    pose.tilt_r = (SIP_TILT * 0.75 + SIP_TILT * 0.25 * swallow) * up
    pose.hand_r = "soda"  # (the grip: the hand fits round the can)
    elbow = _mix(ELBOW_DOWN, ELBOW_OUT, up)
    lips = _mouth(pose.head_pitch, pose.bob)
    lips = (lips[0] - 0.01, lips[1] - 0.005, lips[2] + 0.02)  # the rim on the bottom lip
    hand = _mix(CAN_LOW, lips, up)
    for _ in range(3):  # the can's rim - not the hand - to the lips, however it is tipped
        pose.arm_r = arm_to(_at(hand, 0.0), "r", bob=pose.bob, elbow=elbow)  # type: ignore[arg-type]
        axis = held_axis(pose, "r")
        if axis is None or up <= 0.0:
            break
        rim = tuple(_at(hand, 0.0)[i] + CAN_HALF * float(axis[i]) for i in range(3))
        target = _mix(CAN_LOW, lips, up)
        want = _mix(rim, target, 1.0)
        hand = tuple(hand[i] + (want[i] - rim[i]) * up for i in range(3))  # type: ignore[assignment]
    pose.arm_l = arm_to(_at(REST_L, pose.bob), "l", bob=pose.bob)
    pose.hand_r = None
    if up > 0.6:  # the lips on the can
        pose.expression = "pursed"
    return pose


RAP = 0.07  # how far the hand draws back between raps


def front_of(pose: HumanoidPose, part: str) -> float:
    """How far ahead of the body (its +z) the front of one of its boxes reaches."""
    from cyberduke.animation.skeleton import body_boxes

    group = "arm_" + part[-1] if "arm" in part else "trunk"
    corners = [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    m = next(m for name, m in body_boxes(pose, parts=(group,)) if name == part)
    return max(float(m[2, 0] * x + m[2, 1] * y + m[2, 2] * z + m[2, 3]) for x, y, z in corners)


# ----------------------------------------------------------------------------- stairs
THIGH, SHIN = 0.44, 0.36  # (``animation.rig``: hip to knee, knee to ankle)
ANKLE_UP = 0.1  # the ankle over the sole of the foot
STAIR_CROUCH = -0.09  # the hips a little lower on a stair: the knees never straight
FOOT_REACH = 0.215  # the ankle this far back from the riser ahead (the toe against it)
FOOT_BACK = 0.135  # ... this far on from the riser behind (going down: the heel against it)


def stair_height(p: float, rise: float, steps: int) -> float:
    """How far up (``rise`` > 0) or down (< 0) a stair somebody ``p`` treads along it is
    (the level their feet are about): level with the floor at the foot of it at ``p`` =
    -0.5, level with the floor at the top once ``p`` is ``steps`` - 0.5, and the body
    going up (down) at a steady rate in between."""
    return rise * min(max(p + 0.5, 0.0), float(steps))


def _leg_to(ankle_z: float, ankle_y: float, hip_y: float) -> tuple[float, float, float]:
    """(hip pitch, knee bend, foot pitch) putting the ankle at (``ankle_z`` ahead,
    ``ankle_y`` up) from under the hip - the foot kept level."""
    dz, dy = ankle_z, ankle_y - hip_y
    d = max(0.25, min(THIGH + SHIN - 1e-3, math.hypot(dz, dy)))
    knee = 180.0 - math.degrees(
        math.acos(max(-1.0, min(1.0, (THIGH**2 + SHIN**2 - d * d) / (2 * THIGH * SHIN))))
    )
    ahead = math.degrees(math.atan2(dz, -dy))
    bend = math.degrees(
        math.acos(max(-1.0, min(1.0, (THIGH**2 + d * d - SHIN**2) / (2 * THIGH * d))))
    )
    hip = -(ahead + bend)
    return hip, knee, -(hip + knee)


def stairs(p: float, rise: float, run: float, steps: int, seed: float = 0.0) -> HumanoidPose:
    """Going up (``rise`` > 0) or down (< 0) a stair of ``steps`` treads ``run`` deep,
    ``p`` treads along it (-0.5: at the foot - or the top - of it, ``steps``: off it):
    a foot on every tread in turn - one planted while the body goes
    over it, the other lifted over the edge of the next and brought down two treads on
    (step over step) - the knees bending to take the height, the body leaning into the
    climb (going up) or upright and looking down at the steps (going down), the arms
    swinging with the stride. The pose's feet are on the treads when the body is
    ``stair_height(p)`` up from where the stair starts and ``p * run`` along it."""
    up = rise > 0.0
    body_y = stair_height(p, rise, steps)
    pose = HumanoidPose(bob=STAIR_CROUCH + 0.01 * math.sin(p * math.pi))
    hip_y = 0.9 + pose.bob
    legs = {}
    for tag, offset in (("l", 0.0), ("r", 1.0)):
        m = (p + offset) / 2.0
        k = math.floor(m)
        f = m - k
        tread = 2 * k + 1 - offset  # (the one it stands on, or swings from)

        def spot(t: float) -> tuple[float, float]:
            """Where the ankle goes for a foot on tread ``t`` (its front edge ``t`` runs
            along): going up, the toe to the riser (the heel over the edge below); going
            down, the heel to the riser (the toes over the edge)."""
            t = max(0.0, t)
            z = t * run - FOOT_REACH if up else (t - 1.0) * run + FOOT_BACK
            return z, rise * min(t, float(steps))

        if f < 0.5:  # planted
            z, y = spot(tread)
            lift = 0.0
        else:  # swinging on to two treads further: up and over the edge, then down on it
            u = (f - 0.5) / 0.5
            (z0, y0), (z1, y1) = spot(tread), spot(tread + 2)
            if up:
                uy, uz = smoothstep(0.0, 0.65, u), smoothstep(0.2, 1.0, u)
            else:
                uy, uz = smoothstep(0.35, 1.0, u), smoothstep(0.0, 0.75, u)
            z, y = lerp(z0, z1, uz), lerp(y0, y1, uy)
            lift = math.sin(math.pi * u) * (0.06 if up else 0.05)
            if up and u > 0.12:  # (the toe over the edge of the tread between)
                y = max(y, rise * min(tread + 1.0, float(steps)) + 0.03 - lift)
        legs[tag] = _leg_to(z - p * run, y + lift + ANKLE_UP - body_y, hip_y)
    for tag in ("l", "r"):
        hip, knee, foot = legs[tag]
        if tag == "l":
            pose.leg_l, pose.foot_l = [hip, 0.0, knee], foot
        else:
            pose.leg_r, pose.foot_r = [hip, 0.0, knee], foot
    swing = math.sin(p * math.pi)  # (the arms with the stride, against the legs)
    pose.lean = 9.0 if up else 2.0
    pose.head_pitch = 8.0 if up else 18.0
    pose.head_yaw = 4.0 * math.sin(p * 0.7 + seed)
    bob = pose.bob
    pose.arm_l = arm_to((0.27, 0.9 + bob + 0.05 * swing, 0.1 - 0.1 * swing), "l", bob=bob)
    pose.arm_r = arm_to((-0.27, 0.9 + bob - 0.05 * swing, 0.1 + 0.1 * swing), "r", bob=bob)
    return pose


def knock(t: float, door: float = 0.55) -> HumanoidPose:
    """Three raps on a door ``door`` metres ahead, a pause to listen, again. On each rap
    the front of the forearm comes forward to the door's face and no further."""
    pose = _stance(t)
    cycle = t % 2.2
    hit = max(0.0, math.sin(cycle / 0.3 * math.pi)) if cycle < 0.9 else 0.0
    face = door - 0.005 - RAP * (1.0 - hit)  # where the knuckles are to be
    reach = face - 0.095  # the forearm box stands proud of the IK target by about this
    for _ in range(2):  # ... exactly by this, for this arm angle
        pose = _arms(pose, REST_L, (-0.08, 1.6, reach))
        reach += face - front_of(pose, "forearm_r")
    pose = _arms(pose, REST_L, (-0.08, 1.6, reach))
    pose.grip_r = "fist"
    pose.head_pitch = -2.0
    pose.head_roll = 6.0 if cycle > 1.2 else 0.0  # listening
    return pose


def _reachable(lean: float, drop: float, targets: Sequence[Vec]) -> bool:
    from cyberduke.animation.reach import SHOULDER_X, SHOULDER_Y, bent

    for target in targets:
        side = 1.0 if target[0] >= 0.0 else -1.0
        s = bent((side * SHOULDER_X, SHOULDER_Y - drop, 0.0), lean, -drop)
        if math.dist(s, target) > 0.76:
            return False
    return True


def stoop_for(targets: Sequence[Vec]) -> tuple[float, float]:
    """The least bend (hips ``drop`` metres lower and the upper body leant forward by
    ``lean`` degrees) that brings every target within arm's reach: (lean, drop)."""
    best = (70.0, 0.5)
    for drop in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5):
        for lean in range(0, 75, 5):
            if _reachable(float(lean), drop, targets):
                # bending the knees rather than the back: 10 cm of squat is worth 3 degrees
                if lean + drop * 30.0 < best[0] + best[1] * 30.0:
                    best = (float(lean), drop)
                break
    return best


def stoop_reach(
    u: float,
    target: Vec,
    both: bool = False,
    width: float = PARCEL_GRIP,
    side: str = "r",
    carry: Vec | None = None,
) -> HumanoidPose:
    """Bending down to something low in front (a parcel on a step): the knees give, the
    upper body leans forward over it and the hand - or, ``both``, the two hands either
    side of it, ``width`` from its middle - goes down to ``target``. ``u`` 0..1 goes
    down (1..0 comes back up). The hands start (and end) hanging at the sides, or
    holding something in front at ``carry`` (its middle)."""
    k = smoothstep(0.0, 1.0, u)
    if both:
        hands = (
            (target[0] + width, target[1], target[2]),
            (target[0] - width, target[1], target[2]),
        )
    else:
        hands = (target,)
    lean_max, drop_max = _stoop_cached(tuple(tuple(round(c, 3) for c in h) for h in hands))
    lean, drop = lean_max * k, drop_max * k
    leg_l, leg_r, bob = _squat_legs(drop)
    pose = HumanoidPose(bob=bob, leg_l=leg_l, leg_r=leg_r, lean=lean)
    pose.foot_l = pose.foot_r = -leg_l[0] - leg_l[2]  # soles flat
    if carry is not None:
        rest_l: Vec = (carry[0] + width, carry[1], carry[2])
        rest_r: Vec = (carry[0] - width, carry[1], carry[2])
    else:
        rest_l, rest_r = REST_L, REST_R
    if both:
        left, right = _mix(rest_l, hands[0], k), _mix(rest_r, hands[1], k)
    elif side == "r":
        left, right = rest_l, _mix(rest_r, target, k)
    else:
        left, right = _mix(rest_l, target, k), rest_r
    pose.arm_l = arm_to(left, "l", bob=bob, lean=lean)
    pose.arm_r = arm_to(right, "r", bob=bob, lean=lean)
    pose.head_pitch = 12.0 * k
    return pose


_STOOPS: dict = {}


def _stoop_cached(hands: tuple) -> tuple[float, float]:
    if hands not in _STOOPS:
        _STOOPS[hands] = stoop_for(hands)
    return _STOOPS[hands]


def browse(
    t: float, seed: float = 0.0, table_y: float = 0.9, reach_out: float = 0.62
) -> HumanoidPose:
    """Looking over what is on a table: head down, eyes going along it, a hand reaching
    in now and then to move something."""
    pose = _stance(t, seed)
    s = t + seed
    poke = smoothstep(0.6, 0.95, 0.5 + 0.5 * math.sin(s * 0.9))
    right = _mix(
        (-0.2, 1.12, 0.25), (-0.1 + 0.25 * math.sin(s * 0.3), table_y + 0.06, reach_out), poke
    )
    pose = _arms(pose, (0.18, 1.05, 0.22), right)  # type: ignore[arg-type]
    pose.head_pitch = 32.0
    pose.head_yaw = 24.0 * math.sin(s * 0.55)
    return pose


def think(t: float, seed: float = 0.0) -> HumanoidPose:
    """Weighing it up: a hand at the chin, the other arm across under it, head tilted."""
    pose = _stance(t, seed)
    pose = _arms(pose, (-0.1, 1.33, 0.24), (CHIN[0] - 0.02, CHIN[1] - 0.02, CHIN[2] + 0.03))
    pose.fingers_r = (0.4, 0.25, 0.6, 0.7, 0.75)  # the index along the jaw
    pose.head_pitch = 6.0 + 3.0 * math.sin(t * 0.6)
    pose.head_roll = 8.0
    pose.head_yaw = 10.0 * math.sin(t * 0.35 + seed)
    pose.expression = mutter(t, seed)  # (talking it over with themselves)
    return pose


KICK_PERIOD = 1.45  # seconds from one kick to the next
KICK_STRIKE = 0.55  # ...and how far into each the foot lands (a fraction of the period)
KICK_REACH = 0.44  # how far ahead of the body the toe reaches on the strike (body frame)


def kick(t: float, seed: float = 0.0) -> HumanoidPose:
    """Kicking a machine that has swallowed the cred: the right foot drawn back and driven
    into the front of the cabinet at shin height every ``KICK_PERIOD`` seconds, the body
    rocking back on the strike, the arms out a little for balance, the head down on the
    problem. (The feet do the work: whatever is in the hands stays in them.)"""
    cycle = (t / KICK_PERIOD) % 1.0
    if cycle < 0.4:  # drawing the foot back
        windup, strike = smoothstep(0.0, 0.4, cycle), 0.0
    elif cycle < 0.55:  # driving it in
        u = smoothstep(0.4, 0.55, cycle)
        windup, strike = 1.0 - u, u
    else:  # and back down onto the ground
        windup, strike = 0.0, 1.0 - smoothstep(0.62, 1.0, cycle)
    pose = _stance(t * 0.3, seed)
    drop = 0.025 * windup + 0.02 * strike  # (down a little on the standing knee)
    stand, _, pose.bob = _squat_legs(drop)
    pose.leg_l = [stand[0], 3.0, stand[2]]
    pose.foot_l = -stand[0] - stand[2]  # (the standing foot flat on the ground)
    through = 4.0 * windup * strike  # (swinging through under the hip: the knee up a bit)
    pose.leg_r = [
        28.0 * windup - 30.0 * strike,
        -4.0,
        78.0 * windup + 30.0 * strike + 40.0 * through,
    ]
    # the sole kept level while the foot is near the ground (tipping with the shin it
    # scraped the toe through the floor), pointed once it is up
    flat = -pose.leg_r[0] - pose.leg_r[2]
    up = smoothstep(0.45, 1.0, max(windup, strike))
    pose.foot_r = lerp(flat, 18.0 * windup - 8.0 * strike, up)
    pose.lean = 4.0 * windup - 7.0 * strike
    sway = 0.06 * strike
    pose = _arms(
        pose, (0.36 + sway, 1.0 + 0.08 * strike, 0.2), (-0.38 - sway, 0.98, -0.02 + 0.1 * windup)
    )
    pose.head_pitch = 24.0 - 6.0 * strike
    pose.head_yaw = 5.0 * math.sin(t * 0.8 + seed)
    pose.expression = "grimace"
    return pose


def wipe(t: float, counter_y: float = COUNTER_Y) -> HumanoidPose:
    """Wiping the counter (its top ``counter_y`` up, in the body's frame) down with a rag
    in slow circles."""
    pose = _stance(t)
    a = t * 3.2
    right = (-0.1 + 0.13 * math.cos(a), counter_y + 0.03, 0.58 + 0.1 * math.sin(a))
    pose = _arms(pose, (0.2, counter_y + 0.05, 0.42), right)
    pose.grip_l = "flat"  # steadying on the counter
    pose.grip_r = "flat"
    pose.head_pitch = 32.0
    pose.head_yaw = -6.0 + 4.0 * math.sin(a)
    return pose


def chop(t: float, board: Vec = (0.0, COUNTER_Y, 0.62)) -> HumanoidPose:
    """A knife going up and down fast over the board, the other hand holding the fish down
    and working back along it."""
    pose = _stance(t * 0.3)
    strike = abs(math.sin(t * 7.5))
    along = 0.05 * math.sin(t * 0.8)
    right = (board[0] - 0.08 + along, board[1] + 0.05 + 0.14 * strike, board[2] - 0.02)
    left = (board[0] + 0.13 + along, board[1] + 0.05, board[2] - 0.02)
    pose = _arms(pose, left, right)
    pose.grip_l = "flat"  # holding the fish down
    pose.head_pitch = 40.0
    pose.head_yaw = 4.0 * math.sin(t * 0.9)
    return pose


def wrap(t: float, at: Vec = (0.0, COUNTER_Y, 0.58)) -> HumanoidPose:
    """Both hands round something on the counter, folding the wrap over it and tying it
    off - circling in opposite directions."""
    pose = _stance(t * 0.3)
    a = t * 4.2
    left = (at[0] + 0.09 + 0.04 * math.cos(a), at[1] + 0.08 + 0.04 * math.sin(a), at[2])
    right = (at[0] - 0.09 - 0.04 * math.cos(a), at[1] + 0.08 - 0.04 * math.sin(a), at[2])
    pose = _arms(pose, left, right)
    fold = 0.5 + 0.5 * math.sin(a)
    pose.grip_l = pose.grip_r = "open"
    pose.fingers_l = (0.3, 0.2 + 0.3 * fold, 0.25 + 0.3 * fold, 0.3 + 0.3 * fold, 0.3 + 0.3 * fold)
    pose.fingers_r = (0.3, 0.5 - 0.3 * fold, 0.55 - 0.3 * fold, 0.6 - 0.3 * fold, 0.6 - 0.3 * fold)
    pose.head_pitch = 36.0
    return pose


def _squat_legs(drop: float) -> tuple[list[float], list[float], float]:
    """Leg angles (hip pitch, twist, knee) and bob for a squat that lowers the hips by
    ``drop`` with the feet flat where they are."""
    hip_y = 0.9 - drop
    ankle_y = 0.9 - 0.36 - 0.44  # where the ankle sits standing (0.1 above the floor)
    d = max(0.2, min(0.799, hip_y - ankle_y))
    thigh, shin = 0.44, 0.36
    cos_knee = (thigh**2 + shin**2 - d * d) / (2 * thigh * shin)
    knee = 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_knee))))
    cos_hip = (thigh**2 + d * d - shin**2) / (2 * thigh * d)
    hip = -math.degrees(math.acos(max(-1.0, min(1.0, cos_hip))))
    return [hip, 0.0, knee], [hip, 0.0, knee], -drop


def crouch_reach(u: float, target: Vec = (0.0, 0.45, 0.5), side: str = "r") -> HumanoidPose:
    """Squatting down to reach into something low (a cooler, a crate): the knees bend,
    the hips drop, a hand goes in."""
    k = smoothstep(0.0, 1.0, u)
    leg_l, leg_r, bob = _squat_legs(0.42 * k)
    pose = HumanoidPose(bob=bob, leg_l=leg_l, leg_r=leg_r)
    pose.foot_l = pose.foot_r = -leg_l[0] - leg_l[2]  # soles flat
    rest = REST_R if side == "r" else REST_L
    hand = _mix(rest, target, k)
    other = (0.18, 0.95 - 0.2 * k, 0.3) if side == "r" else (-0.18, 0.95 - 0.2 * k, 0.3)
    if side == "r":
        pose.arm_r = arm_to(hand, "r", bob=bob)  # type: ignore[arg-type]
        pose.arm_l = arm_to(other, "l", bob=bob)
    else:
        pose.arm_l = arm_to(hand, "l", bob=bob)  # type: ignore[arg-type]
        pose.arm_r = arm_to(other, "r", bob=bob)
    pose.head_pitch = 10.0 + 30.0 * k
    return pose


STOOL_SEAT_Y = 0.61
SOFA_SEAT_Y = 0.5  # a sagging sofa cushion


def sit(
    upper: HumanoidPose | None = None, seat_y: float = STOOL_SEAT_Y, stool: bool | None = None
) -> HumanoidPose:
    """Sitting on a seat ``seat_y`` up (a stool by default: feet dangling; a sofa: feet
    on the floor): thighs forward, shins hanging, the upper body whatever ``upper`` is
    doing (for a stool built with ``bob=SEAT_BOB``)."""
    pose = upper or idle(0.0)
    pose.bob = SEAT_BOB + (seat_y - STOOL_SEAT_Y)
    if stool if stool is not None else seat_y >= STOOL_SEAT_Y - 1e-6:
        pose.leg_l = [-86.0, 4.0, 82.0]
        pose.leg_r = [-84.0, -4.0, 86.0]
    else:  # low: thighs level, shins straight down to the floor
        pose.leg_l = [-88.0, 6.0, 88.0]
        pose.leg_r = [-88.0, -6.0, 88.0]
        pose.arm_l = arm_to((0.2, 0.9 + pose.bob + 0.12, 0.32), "l", bob=pose.bob)
        pose.arm_r = arm_to((-0.2, 0.9 + pose.bob + 0.12, 0.32), "r", bob=pose.bob)
    pose.foot_l = pose.foot_r = 0.0
    return pose


def watch_tv(
    t: float, seat_y: float = SOFA_SEAT_Y, click: float = 0.0, seed: float = 0.0
) -> HumanoidPose:
    """Slumped on a sofa ``seat_y`` up in front of the television, the remote in the right
    hand resting on the thigh; ``click`` (0..1) brings it up and points it at the set (a
    thumb on the button) - flicking through the channels."""
    pose = sit(idle(t * 0.3, seed), seat_y=seat_y, stool=False)
    pose.head_pitch = 2.0 + 2.0 * math.sin(t * 0.3 + seed)
    pose.head_yaw = 4.0 * math.sin(t * 0.2 + seed)
    rest = (-0.2, 0.9 + pose.bob + 0.12, 0.34)
    aim = (-0.16, 1.02 + pose.bob + 0.2, 0.5)
    k = smoothstep(0.0, 1.0, click)
    pose.arm_r = arm_to(_mix(rest, aim, k), "r", bob=pose.bob)  # type: ignore[arg-type]
    press = 0.3 + 0.5 * k * (0.5 + 0.5 * math.sin(t * 18.0))
    pose.fingers_r = (press, 0.8, 0.85, 0.88, 0.9)  # the thumb on the button
    return pose


def typing_seated(t: float, desk_y: float = 0.76, seat_y: float = 0.48) -> HumanoidPose:
    """At a desk on a chair ``seat_y`` up: feet on the floor, both hands on the keyboard
    on the desk top ``desk_y`` up (``typing``, solved for the lowered shoulders)."""
    bob = SEAT_BOB + (seat_y - STOOL_SEAT_Y)
    pose = typing(t, desk_y - bob)
    pose.bob = bob
    pose.leg_l = [-88.0, 6.0, 88.0]
    pose.leg_r = [-88.0, -6.0, 88.0]
    pose.foot_l = pose.foot_r = 0.0
    return pose


def _tapping(t: float, phase: float) -> tuple[float, float, float, float, float]:
    """Fingers on keys: each one coming down now and then, out of step with the others."""
    taps = [max(0.0, math.sin(t * 13.0 + phase + k * 2.1)) ** 6 for k in range(4)]
    return (0.3, *(0.3 + 0.25 * a for a in taps))  # type: ignore[return-value]


def typing(t: float, desk_y: float = 0.78) -> HumanoidPose:
    """At a keyboard on a desk in front (its top ``desk_y`` up): both hands on it, fingers
    going, eyes on the screen."""
    pose = _stance(t * 0.3)
    tap = 0.008 * math.sin(t * 17.0)
    y = desk_y + 0.02
    pose = _arms(pose, (0.12, y + tap, 0.42), (-0.12, y - tap, 0.42))
    pose.grip_l = pose.grip_r = "type"
    pose.fingers_l = _tapping(t, 0.0)
    pose.fingers_r = _tapping(t, 1.7)
    pose.head_pitch = 14.0 + 2.0 * math.sin(t * 0.7)
    pose.head_yaw = 4.0 * math.sin(t * 0.4)
    return pose


def toss(u: float) -> HumanoidPose:
    """Throwing something away underarm into a bin: back, then out and up, let go."""
    pose = _stance(0.0)
    back = (-0.26, 1.05, -0.1)
    out = (-0.1, 1.5, 0.65)
    hand = (
        _mix(back, out, smoothstep(0.3, 0.8, u))
        if u > 0.3
        else _mix(CHEST, back, smoothstep(0.0, 0.3, u))
    )
    pose = _arms(pose, REST_L, hand)  # type: ignore[arg-type]
    if u > 0.75:
        pose.grip_r = "open"  # let go
    pose.head_pitch = 12.0
    return pose


def carrying(cycle: float, how: str, moving: bool = True) -> HumanoidPose:
    """Walking (or standing) with something: "two" hands round a box in front, "one" a
    bag hanging from the right hand, "held" a small thing carried in the right hand,
    "tucked" a parcel under the left arm."""
    pose = walk_animation(cycle) if moving else _stance(cycle)
    if how == "two":
        c = CARRY_TWO
        pose = _arms(pose, (c[0] + PARCEL_GRIP, c[1], c[2]), (c[0] - PARCEL_GRIP, c[1], c[2]))
    elif how == "one":
        pose.arm_r = arm_to(_at((-0.28, 0.9, 0.08), pose.bob), "r", bob=pose.bob)
    elif how == "held":
        pose.arm_r = arm_to(_at((-0.18, 1.12, 0.3), pose.bob), "r", bob=pose.bob)
    elif how == "tucked":  # under the left arm, against the hip
        pose.arm_l = arm_to(_at((0.31, 0.98, 0.12), pose.bob), "l", bob=pose.bob)
    return pose


def beckon(t: float) -> HumanoidPose:
    """Calling out to passers-by: the right hand up beside the head, the palm to the
    front, waving from side to side - and every few waves the fingers curling in, "come
    on over" - the left hand easy at the side."""
    pose = _stance(t)
    wave = math.sin(t * 6.5)
    bob = pose.bob
    right = (-0.36 + 0.075 * wave, 1.93 + bob, 0.26 + 0.02 * math.cos(t * 6.5))
    pose.arm_r = arm_to(right, "r", bob=bob, elbow=(-1.0, -0.45, 0.1))
    pose.arm_l = arm_to(_at(REST_L, bob), "l", bob=bob)
    pose.grip_r = "wave"
    come = max(0.0, math.sin(t * 1.4)) ** 3  # (now and then: come here!)
    c = 0.05 + 0.7 * come * (0.5 + 0.5 * math.sin(t * 9.0))
    pose.fingers_r = (0.12, c, c + 0.03, c + 0.05, c + 0.08)
    pose.head_pitch = -4.0
    pose.head_yaw = 25.0 * math.sin(t * 0.5)
    calling = speech(t, 1.0, rate=4.0)  # (calling out to them)
    pose.expression = "shout" if calling == "talk" else calling
    return pose


def open_door(t: float, side: str = "l", door: float = 0.57) -> HumanoidPose:
    """Fumbling a key into a lock (with the ``side`` hand) and pushing the door, whose
    face is ``door`` metres ahead (the key hand stops short of it)."""
    pose = _stance(t)
    jiggle = 0.015 * math.sin(t * 12.0)
    z = door - 0.075  # the forearm's front a couple of centimetres short of the door
    if side == "l":
        pose = _arms(pose, (0.05, 1.25 + jiggle, z), REST_R)
        pose.grip_l = "pinch"  # (the key)
    else:
        pose = _arms(pose, REST_L, (-0.05, 1.25 + jiggle, z))
        pose.grip_r = "pinch"
    pose.head_pitch = 20.0
    return pose
