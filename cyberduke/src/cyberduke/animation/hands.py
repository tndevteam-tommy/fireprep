"""Hands: a palm, four fingers of two bones each and a thumb - and how they hold things.

A hand hangs from the wrist at the end of the forearm (``animation.reach``: the elbow's
frame, the forearm running down its -y, x across it - the body's left at rest - and z to
the front). In the hand's own frame the fingers point down (-y), the palm faces the body
(+x for the right hand, -x for the left: ``PALM_SIGN``) and the thumb is at the front (+z).

What the hand does is a ``Grip``: how far each finger and the thumb curl (0 straight, 1 a
fist), how far the fingers fan out, which way the palm turns (``palm``: to the side, up
under a box, down on a counter, towards the other hand round a parcel, to the face round
a cigarette) and - when it holds something - where the thing sits in the hand
(``contact``: how far in front of the palm, how far down it from the wrist, how far
across). The thing held is drawn at the hand point (``reach.hand``: 0.42 down the forearm
from the elbow), so the hand is fitted round it: the wrist bends (``hand_frame``) to bring
the place the thing sits in the hand onto the forearm's line at the hand point, and sits
as far up the forearm as that needs - a can sits against the palm in curled fingers, a
phone lies in an upturned palm under the thumb, a cigarette between two fingers.

Which grip goes with what: ``GRIP_FOR_ITEM`` by the thing held (``HumanoidPose.hand_l`` /
``hand_r``; both hands on one thing: flat against its sides), or the pose's own
``grip_l`` / ``grip_r`` (a fist to knock, a pointing finger); the pose can override the
curls too (``fingers_l`` / ``fingers_r``: thumb, index, middle, ring, little finger - for
fingers that move: typing, beckoning, scrolling a phone with the thumb).

Pure NumPy: the renderer (``render.humanoid``) and the collision body
(``animation.skeleton``) build the hand from the same numbers (``animation.rig``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cyberduke.animation.pose import HumanoidPose

HAND_AT = 0.42  # the hand point (what is held sits there), down the forearm from the elbow
WRIST = 0.33  # an empty hand's wrist, down the forearm
PALM_SIGN = {"r": 1.0, "l": -1.0}  # which way along x the palm faces (the body's middle)

PALM_LENGTH = 0.095  # wrist to knuckles
PALM_WIDTH = 0.088
PALM_THICK = 0.034
KNUCKLES = (0.031, 0.0105, -0.0105, -0.031)  # across the palm (z): index .. little finger
FINGER_BONES = ((0.046, 0.04), (0.05, 0.044), (0.047, 0.041), (0.038, 0.033))
FINGER_THICK = (0.0205, 0.021, 0.02, 0.018)
FINGER_FAN = (-6.0, -1.5, 4.0, 10.0)  # degrees each finger fans out (index towards the thumb)
THUMB_ROOT = (0.012, -0.024, 0.036)  # (x along the palm normal, scaled by the side's sign)
THUMB_BONES = (0.042, 0.034)
THUMB_THICK = (0.025, 0.022)
CURL_KNUCKLE, CURL_MIDDLE = 92.0, 100.0  # a finger's two joints, fully curled (a fist)
THUMB_CURL = (55.0, 70.0)

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Grip:
    """A way of holding the hand (see the module docstring)."""

    curls: tuple[float, float, float, float, float]  # thumb, index, middle, ring, little
    spread: float = 1.0
    thumb_out: float = 30.0  # degrees the thumb stands off the side of the palm
    thumb_turn: float = 20.0  # degrees it turns in to face the palm (opposition)
    palm: str = "free"  # "free", "side", "up", "down", "toward", "out", "face", "forward"
    contact: Vec3 | None = None  # (in front of the palm, down from the wrist, across)
    flex: float = 8.0  # degrees the wrist is bent (palm-ward) when nothing is held
    # how a curl shares out between a finger's two joints (the knuckle, the middle): round
    # something thick the knuckles bend less and the fingers wrap more
    bend: tuple[float, float] = (1.0, 1.0)
    # which way the thumb's side of the hand (its +z: along the line across the knuckles)
    # is turned instead of the palm: "up" wraps the fingers round something upright (a can
    # - which goes with the hand, tipping as it is lifted to the lips: ``held_axis``)
    axis: str | None = None


GRIPS: dict[str, Grip] = {
    "relaxed": Grip((0.18, 0.2, 0.25, 0.3, 0.36)),
    "open": Grip((0.04, 0.05, 0.05, 0.07, 0.1), spread=1.7, thumb_out=45.0, flex=-6.0),
    "flat": Grip(
        (0.08, 0.04, 0.04, 0.06, 0.08), spread=0.8, thumb_out=25.0, palm="down", flex=-10.0
    ),
    "fist": Grip(
        (0.7, 1.0, 1.0, 1.0, 1.0), spread=0.3, thumb_out=8.0, thumb_turn=60.0, palm="side", flex=0.0
    ),
    "point": Grip(
        (0.65, 0.0, 0.95, 1.0, 1.0), spread=0.3, thumb_out=8.0, thumb_turn=55.0, flex=0.0
    ),
    "type": Grip((0.3, 0.38, 0.36, 0.36, 0.4), spread=0.9, palm="down", flex=-4.0),
    "beckon": Grip((0.2, 0.3, 0.35, 0.4, 0.45), spread=1.0, palm="up", flex=0.0),
    # waving somebody over: the palm to the front, fingers up (curling "come here": the
    # pose's own curls)
    "wave": Grip(
        (0.1, 0.06, 0.06, 0.08, 0.1), spread=1.3, thumb_out=40.0, palm="forward", flex=-8.0
    ),
    # a hand over the face (blinded): the palm on it, the fingers flat over the brow
    "face": Grip(
        (0.15, 0.02, 0.03, 0.04, 0.06), spread=1.2, thumb_out=35.0, palm="face", flex=-4.0
    ),
    # holding things
    "can": Grip(
        (0.42, 0.55, 0.55, 0.53, 0.46),
        spread=0.4,
        thumb_out=58.0,
        thumb_turn=72.0,
        palm="side",
        contact=(0.038, 0.098, 0.0),
        axis="up",
        bend=(0.45, 1.0),
    ),
    "tray": Grip(
        (0.3, 0.14, 0.14, 0.16, 0.2),
        spread=0.7,
        thumb_out=50.0,
        thumb_turn=10.0,
        palm="up",
        contact=(0.02, 0.09, 0.0),
    ),
    "phone": Grip(
        (0.5, 0.34, 0.36, 0.38, 0.4),
        spread=0.3,
        thumb_out=48.0,
        thumb_turn=55.0,
        palm="up",
        contact=(0.026, 0.082, 0.0),
    ),
    "pinch": Grip(
        (0.45, 0.45, 0.55, 0.8, 0.88),
        spread=0.4,
        thumb_out=40.0,
        thumb_turn=70.0,
        palm="side",
        contact=(0.012, 0.125, 0.02),
    ),
    "chopsticks": Grip(
        (0.4, 0.34, 0.48, 0.74, 0.82),
        spread=0.4,
        thumb_out=35.0,
        thumb_turn=62.0,
        palm="side",
        contact=(0.008, 0.12, 0.016),
    ),
    "handle": Grip(
        (0.62, 0.82, 0.86, 0.88, 0.9),
        spread=0.2,
        thumb_out=14.0,
        thumb_turn=62.0,
        palm="side",
        contact=(0.014, 0.1, 0.0),
    ),
    "cigarette": Grip(
        (0.25, 0.16, 0.2, 0.62, 0.72),
        spread=0.7,
        thumb_out=34.0,
        thumb_turn=30.0,
        palm="out",
        contact=(0.0, 0.122, 0.021),
    ),
    "hook": Grip(
        (0.55, 0.86, 0.86, 0.86, 0.86),
        spread=0.2,
        thumb_out=12.0,
        thumb_turn=40.0,
        palm="side",
        contact=(0.006, 0.11, 0.0),
    ),
    "press": Grip(
        (0.1, 0.05, 0.05, 0.07, 0.09),
        spread=0.8,
        thumb_out=25.0,
        palm="down",
        contact=(-0.008, 0.09, 0.0),
    ),
    # a phone in both hands: a hand either side of it, the palms turned in to each other,
    # the fingers curled in under it (it rests up on them), the thumbs standing up by its
    # edges (the pose's own curls: ``street.phone``)
    "text": Grip(
        (0.25, 0.5, 0.4, 0.45, 0.3),
        spread=0.3,
        thumb_out=85.0,
        thumb_turn=40.0,
        palm="toward",
        contact=(0.008, 0.085, 0.0),
    ),
    "both": Grip(
        (0.12, 0.08, 0.08, 0.1, 0.12),
        spread=0.6,
        thumb_out=30.0,
        thumb_turn=5.0,
        palm="toward",
        contact=(-0.01, 0.1, 0.0),
    ),
}

GRIP_FOR_ITEM: dict[str, str] = {
    "soda": "can",
    "beer": "can",
    "glass": "can",
    "fish": "tray",
    "sushi": "tray",
    "takeaway": "tray",
    "takeaway_open": "tray",
    "noodles": "tray",
    "parcel": "tray",
    "gift": "tray",
    "cyberdeck": "tray",
    "phone": "phone",
    "remote": "handle",
    "chip": "pinch",
    "cred": "pinch",
    "datastick": "pinch",
    "chopsticks": "chopsticks",
    "chopsticks_sushi": "chopsticks",
    "chopsticks_noodles": "chopsticks",
    "knife": "handle",
    "cigarette": "cigarette",
    "bag": "hook",
    "rag": "press",
}


# Things held in both hands that are held a way of their own (else: "both", by the sides)
TWO_HANDED: dict[str, str] = {"phone": "text"}


def grip_name(pose: HumanoidPose, side: str) -> str:
    """The grip a hand is in: what it holds says how it is held; an empty hand is in the
    pose's own grip (a fist, a pointing finger) - or relaxed."""
    own = pose.grip_l if side == "l" else pose.grip_r
    held = pose.hand_l if side == "l" else pose.hand_r
    if not held:
        return own or "relaxed"
    if held:
        if pose.hand_l is not None and pose.hand_l == pose.hand_r:
            return TWO_HANDED.get(held, "both")
        name = GRIP_FOR_ITEM.get(held)
        if name is None:
            for prefix, grip in (("sushi_open", "tray"), ("noodles", "tray")):
                if held.startswith(prefix):
                    return grip
            return "tray"
        return name
    return own or "relaxed"


def grip_of(pose: HumanoidPose, side: str) -> Grip:
    return GRIPS.get(grip_name(pose, side), GRIPS["relaxed"])


def curls_of(pose: HumanoidPose, side: str, grip: Grip | None = None) -> tuple[float, ...]:
    own = pose.fingers_l if side == "l" else pose.fingers_r
    if own is not None:
        return tuple(own)
    return (grip or grip_of(pose, side)).curls


# ----------------------------------------------------------------------------- rotations
def _rx(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def axis_angle(r: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    """(degrees, unit axis) of a rotation matrix - a ``rotate`` call."""
    cos_a = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if angle < 1e-7:
        return 0.0, (0.0, 1.0, 0.0)
    if math.pi - angle < 1e-4:
        w, v = np.linalg.eigh((r + r.T) / 2.0)
        axis = v[:, int(np.argmax(w))]
    else:
        axis = np.array((r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]))
        axis = axis / (2.0 * math.sin(angle))
    return math.degrees(angle), (float(axis[0]), float(axis[1]), float(axis[2]))


def _align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The shortest rotation taking unit vector ``a`` to unit vector ``b``."""
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.identity(3) if c > 0 else _rz(180.0)
    k = v / s
    kx = np.array(((0.0, -k[2], k[1]), (k[2], 0.0, -k[0]), (-k[1], k[0], 0.0)))
    ang = math.atan2(s, c)
    return np.identity(3) + math.sin(ang) * kx + (1.0 - math.cos(ang)) * (kx @ kx)


# ----------------------------------------------------------------------------- the frame
def forearm_rotation(pose: HumanoidPose, side: str) -> np.ndarray:
    """Body frame <- forearm frame (the lean, the shoulder's three angles, the elbow)."""
    a = pose.arm_l if side == "l" else pose.arm_r
    return _rx(pose.lean) @ _rx(a[0]) @ _ry(a[1]) @ _rz(a[2]) @ _rx(a[3])


OUTWARD = {"r": (-0.85, -0.2, 0.75), "l": (0.85, -0.2, 0.75)}  # forwards and out to the side


def outward(side: str, forearm, tilt: float = 0.0) -> tuple[float, float, float]:
    """Which way something held across the fingers points (a cigarette): the part of
    "forwards and a little out to the side" square to the forearm, turned ``tilt`` degrees
    about it (rolled in the fingers)."""
    fx, fy, fz = (float(c) for c in forearm)
    n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
    fx, fy, fz = fx / n, fy / n, fz / n
    ox, oy, oz = OUTWARD[side]
    k = ox * fx + oy * fy + oz * fz
    dx, dy, dz = ox - k * fx, oy - k * fy, oz - k * fz
    m = math.sqrt(dx * dx + dy * dy + dz * dz)
    if m < 1e-6:
        dx, dy, dz, m = 0.0, 1.0, 0.0, 1.0
    dx, dy, dz = dx / m, dy / m, dz / m
    if tilt:  # Rodrigues about the forearm (d is square to it)
        c, s = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
        cx, cy, cz = fy * dz - fz * dy, fz * dx - fx * dz, fx * dy - fy * dx
        dx, dy, dz = dx * c + cx * s, dy * c + cy * s, dz * c + cz * s
    return dx, dy, dz


def _palm_target(
    pose: HumanoidPose, side: str, grip: Grip, fore_rot: np.ndarray
) -> np.ndarray | None:
    """Which way the palm should face, in the forearm's frame (None: leave it be)."""
    n = PALM_SIGN[side]
    if grip.palm == "free":
        return None
    if grip.palm == "side":
        # to the body's middle: for the right hand +x, the left -x (in the body frame)
        d = np.array((n, 0.0, 0.0))
    elif grip.palm == "up":
        d = np.array((0.0, 1.0, 0.0))
    elif grip.palm == "down":
        d = np.array((0.0, -1.0, 0.0))
    elif grip.palm == "toward":
        from cyberduke.animation.reach import hand as hand_at

        other = "l" if side == "r" else "r"
        d = hand_at(pose, other)[0] - hand_at(pose, side)[0]
    elif grip.palm == "forward":  # (to the front: to whoever is waved at)
        from cyberduke.animation.reach import bent

        d = bent((0.0, 0.0, 1.0), pose.lean) - bent((0.0, 0.0, 0.0), pose.lean)
    elif grip.palm == "face":  # (towards the middle of the head)
        from cyberduke.animation import anatomy
        from cyberduke.animation.reach import bent
        from cyberduke.animation.reach import hand as hand_at

        head = bent((0.0, anatomy.HEAD_Y + pose.bob, 0.02), pose.lean, pose.bob)
        d = head - hand_at(pose, side)[0]
    elif grip.palm == "out":
        fore = fore_rot @ np.array((0.0, -1.0, 0.0))
        tilt = pose.tilt_l if side == "l" else pose.tilt_r
        d = -np.array(outward(side, fore, tilt))
    else:
        return None
    return fore_rot.T @ d


@dataclass(frozen=True)
class HandFrame:
    """Where a hand is on its forearm: the wrist down the forearm (``wrist``, metres from
    the elbow), and the hand's rotation in the forearm's frame."""

    wrist: float
    rotation: np.ndarray
    grip: Grip
    curls: tuple[float, ...]


def _frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Columns a, b', a x b' - b made square to a first (both unit)."""
    b = b - a * float(a @ b)
    b = b / (float(np.linalg.norm(b)) or 1.0)
    return np.column_stack((a, b, np.cross(a, b)))


def hand_frame(pose: HumanoidPose, side: str) -> HandFrame:
    """The hand's place on its forearm for the pose: bent at the wrist round what it holds
    (the contact point brought onto the forearm's line at the hand point) and turned about
    the forearm so the palm faces the grip's way - or, for a grip with an ``axis``, so the
    line across the knuckles lies as near that way as the forearm allows."""
    grip = grip_of(pose, side)
    n = PALM_SIGN[side]
    curls = curls_of(pose, side, grip)
    fore_rot = forearm_rotation(pose, side)
    if grip.contact is not None:
        r, down, across = grip.contact
        c = np.array((n * (PALM_THICK / 2.0 + r), -down, across))
        length = float(np.linalg.norm(c))
        wrist = HAND_AT - length
        if grip.axis is not None:
            # (tipped towards the face by the pose's tilt: a can lifted to the lips)
            tilt = math.radians(pose.tilt_l if side == "l" else pose.tilt_r)
            up = AXES[grip.axis]
            want = np.array((up[0], up[1] * math.cos(tilt), -up[1] * math.sin(tilt)))
            t = fore_rot.T @ want  # (in the forearm's frame)
            t = t - np.array((0.0, t[1], 0.0))
            if float(np.linalg.norm(t)) > 1e-6:
                # the contact direction onto the forearm (-y), the thumb's side (+z) onto t
                hand = _frame(c / length, np.array((0.0, 0.0, 1.0)))
                forearm = _frame(np.array((0.0, -1.0, 0.0)), t / float(np.linalg.norm(t)))
                return HandFrame(wrist, forearm @ hand.T, grip, curls)
        bend = _align(c / length, np.array((0.0, -1.0, 0.0)))
    else:
        bend = _rz(n * grip.flex)
        wrist = WRIST
    target = _palm_target(pose, side, grip, fore_rot)
    roll = np.identity(3)
    if target is not None:
        palm = bend @ np.array((n, 0.0, 0.0))
        t = target - np.array((0.0, target[1], 0.0))  # square to the forearm
        if float(np.linalg.norm(t)) > 1e-6 and abs(palm[0]) + abs(palm[2]) > 1e-6:
            # the turn about the forearm (y) taking the palm's heading to the target's
            psi = math.degrees(math.atan2(palm[2], palm[0]) - math.atan2(t[2], t[0]))
            roll = _ry(psi)
    return HandFrame(wrist, roll @ bend, grip, curls)


AXES = {"up": np.array((0.0, 1.0, 0.0))}


def held_axis(pose: HumanoidPose, side: str) -> np.ndarray | None:
    """For a grip with an ``axis`` (a can): the way the held thing's own up points, in the
    body frame - out of the top of the fist (the thumb's side), so it turns with the
    hand."""
    frame = hand_frame(pose, side)
    if frame.grip.axis is None:
        return None
    return forearm_rotation(pose, side) @ frame.rotation @ np.array((0.0, 0.0, 1.0))


__all__ = [
    "GRIPS",
    "GRIP_FOR_ITEM",
    "HAND_AT",
    "Grip",
    "HandFrame",
    "axis_angle",
    "grip_name",
    "hand_frame",
    "held_axis",
    "outward",
]
