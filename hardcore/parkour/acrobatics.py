"""Acrobatics: mid-air flips and the landing roll - the whole-body rotations.

A flip or a roll turns the whole body (and with it the eyes) through 360 degrees
about the body's centre of mass.  The simulation keeps the collision box upright -
a tucked body fits in it - so a spin is purely the body's *orientation*: a `Spin`
is advanced by the state machine every physics tick and `body_transform()` turns
it into the rigid transform that is applied to the camera and to the first-person
body mesh (and, over the network, to other players' avatars).

Flip  (`Space` again in the air after a jump, while no hold is within reach):
    front flip (default), back flip (`S` held, or a standing jump), side flip
    (`A` / `D` held without `W` / `S`).  The rotation is timed to the airtime
    left, so it always finishes before the feet touch down; landing early
    finishes it at once.
Roll  (landing after a fall of more than ROLL_MIN_DROP, 7 ft):
    the body absorbs the landing into a crouch, rolls over along the direction
    of travel (a backward roll when falling backwards) and comes up standing,
    keeping the momentum to run on.
"""
import math

import numpy as np

import config

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)

FRONT, BACK, LEFT, RIGHT = "front", "back", "left", "right"
ROLL = "roll"                      # a landing roll (forward or backward: the body rolls along its travel)
KINDS = (FRONT, BACK, LEFT, RIGHT, ROLL)


def _smoothstep(t):
    t = min(1.0, max(0.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


class Spin:
    """One whole-body rotation in progress (a flip or a roll)."""

    __slots__ = ("kind", "duration", "t", "heading", "backward", "rush_rate")

    def __init__(self, kind, duration, heading, backward=False):
        self.kind = kind
        self.duration = float(max(1e-3, duration))
        self.t = 0.0
        h = np.array([heading[0], 0.0, heading[2]], dtype=np.float32)
        n = float(np.linalg.norm(h))
        self.heading = h / n if n > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.backward = bool(backward)     # a roll taken facing away from the travel
        self.rush_rate = None              # set when the spin must end now (landed mid-flip)

    # ------------------------------------------------------------------ time
    @property
    def u(self):
        return min(1.0, self.t / self.duration)

    @property
    def done(self):
        return self.t >= self.duration

    @property
    def is_roll(self):
        return self.kind == ROLL

    def advance(self, dt):
        if self.rush_rate is not None:
            self.t += dt * self.rush_rate
        else:
            self.t += dt

    def rush(self, within=config.FLIP_RUSH_TIME):
        """Finish what is left of the rotation within `within` seconds (the feet touched down)."""
        left = self.duration - self.t
        if left > 0.0:
            self.rush_rate = max(1.0, left / max(1e-3, within))

    def remaining_time(self):
        left = max(0.0, self.duration - self.t)
        return left / (self.rush_rate or 1.0)

    # ---------------------------------------------------------------- shape
    def angle(self):
        """Degrees turned so far (0 -> 360).  A flip turns at a steady rate with soft ends; a roll
        first absorbs the landing, then rolls over and is upright again before it stands up."""
        u = self.u
        if self.is_roll:
            a, b = config.ROLL_SPIN_START, config.ROLL_SPIN_END
            v = min(1.0, max(0.0, (u - a) / (b - a)))
        else:
            v = u
        eased = 0.5 * v + 0.5 * (v - math.sin(2.0 * math.pi * v) / (2.0 * math.pi))
        return 360.0 * eased

    def tuck(self):
        """0 = open body, 1 = knees to the chest (arms round the shins)."""
        u = self.u
        if self.is_roll:
            return _smoothstep(u / 0.18) * _smoothstep((1.0 - u) / 0.25)
        return _smoothstep(u / 0.22) * _smoothstep((1.0 - u) / 0.22)

    def crouch(self):
        """How far the body has sunk toward the ground (a roll only)."""
        if not self.is_roll:
            return 0.0
        u = self.u
        return config.ROLL_CROUCH * _smoothstep(u / 0.16) * _smoothstep((1.0 - u) / 0.28)

    def reach(self):
        """A roll starts with the hands going to the ground ahead (0..1)."""
        if not self.is_roll:
            return 0.0
        u = self.u
        return _smoothstep(u / 0.08) * _smoothstep((0.30 - u) / 0.12) if u < 0.30 else 0.0

    def head_pitch(self):
        """Degrees the head is tucked relative to the body (negative = chin to the chest): into a
        front flip or a roll you look at your knees, into a back flip the head goes back."""
        if self.kind in (FRONT, ROLL):
            return -config.FLIP_HEAD_TUCK_DEG * max(self.tuck(), self.reach())
        if self.kind == BACK:
            return config.FLIP_HEAD_BACK_DEG * self.tuck()
        return 0.0

    # ---------------------------------------------------------------- geometry
    def axis_and_sign(self):
        """World rotation axis and the sign of the rotation for this spin."""
        h = self.heading
        right = np.cross(h, UP).astype(np.float32)
        if self.kind in (FRONT, ROLL):
            return right, -1.0             # head goes forward and down (a roll: the body rolls along its travel)
        if self.kind == BACK:
            return right, 1.0
        if self.kind == RIGHT:
            return h, 1.0
        return h, -1.0

    def pivot(self, body_center):
        c = np.asarray(body_center, dtype=np.float32)
        if self.is_roll:
            feet = float(c[1]) - config.PLAYER_HALF_HEIGHT
            return np.array([c[0], feet + config.ROLL_PIVOT_HEIGHT, c[2]], dtype=np.float32)
        return c + UP * config.FLIP_PIVOT_UP

    def to_wire(self):
        """Compact form for the network: [kind, degrees, crouch, heading x, heading z]."""
        return [self.kind, round(self.angle(), 1), round(self.crouch(), 2),
                round(float(self.heading[0]), 3), round(float(self.heading[2]), 3)]


def rotation(axis, angle_rad):
    """3x3 rotation about a unit axis (Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    x, y, z = a
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]], dtype=np.float32)


def transform_for(kind, angle_deg, crouch, heading, body_center):
    """4x4 rigid transform taking the upright body (and the eyes) to where the spin puts them:
    sink by `crouch`, then rotate by `angle_deg` about the spin's pivot."""
    spin = Spin(kind, 1.0, heading)
    axis, sign = spin.axis_and_sign()
    R = rotation(axis, math.radians(angle_deg) * sign)
    pivot = spin.pivot(body_center)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    # p' = pivot + R (p - crouch*UP - pivot)
    T[:3, 3] = pivot - R @ (pivot + UP * float(crouch))
    return T


def body_transform(spin, body_center):
    """The transform for a Spin in progress (None when there is none)."""
    if spin is None:
        return None
    return transform_for(spin.kind, spin.angle(), spin.crouch(), spin.heading, body_center)


def view_for(spin, body_center, eye, forward, up, right):
    """Where the eyes are and which way they look during a spin: the head tuck (a pitch about the
    body's right) and then the whole-body rotation.  Returns (eye, forward, up, T) - T is None
    (and the inputs come back unchanged) when there is no spin."""
    if spin is None:
        return eye, forward, up, None
    T = body_transform(spin, body_center)
    hp = spin.head_pitch()
    if abs(hp) > 1e-3:
        Rh = rotation(right, math.radians(hp))
        forward = Rh @ np.asarray(forward, dtype=np.float32)
        up = Rh @ np.asarray(up, dtype=np.float32)
    R = T[:3, :3]
    return apply(T, eye), (R @ forward).astype(np.float32), (R @ up).astype(np.float32), T


def apply(T, p):
    p = np.asarray(p, dtype=np.float32)
    return (T[:3, :3] @ p + T[:3, 3]).astype(np.float32)


def flip_kind(forward, back, left, right, speed_h):
    """Which flip the held keys ask for: back with S (or from a standing jump), side with A / D alone,
    otherwise a front flip."""
    if back and not forward:
        return BACK
    if (left != right) and not forward and not back:
        return LEFT if left else RIGHT
    if not forward and speed_h < 1.0:
        return BACK                        # standing: the back tuck
    return FRONT


def time_to_land(feet_y, vy, ground_y):
    """Seconds until feet at feet_y moving up at vy fall to ground_y (None if never)."""
    if ground_y is None:
        return None
    g = -config.GRAVITY
    dh = float(feet_y) - float(ground_y)
    # feet(t) = feet_y + vy t - g t^2 / 2 = ground_y
    disc = vy * vy + 2.0 * g * dh
    if disc < 0.0:
        return 0.0
    return max(0.0, (vy + math.sqrt(disc)) / g)
