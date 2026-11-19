"""The head, as numbers: its size, its shape, and where the features are on it.

One description shared by everything that needs it - ``render.shapes`` builds the skull's
mesh from it, ``animation.rig`` sets eyes, a fringe or a mohawk on its surface, and
``assets.faces`` paints the face where the modelled features are - so the painted eyes
stay on the eyes however the head changes. Pure math (no NumPy needed here).

The head fills a box ``HEAD_SIZE`` round its middle (``HEAD_Y`` above the feet, standing):
``t`` runs 0 (the chin) .. 1 (the crown); a cross-section at height ``t`` is a
superellipse, rounder over the face (``P_FRONT``) than at the back (``P_BACK``: a flatter
back to the skull, as real heads have), each half its own depth. Compared with the first
heads (0.38 m tall) this one is 3 cm taller, all of it forehead: the chin, the eyes, the
nose and the mouth are where they always were.
"""

from __future__ import annotations

import math

HEAD_SIZE = (0.32, 0.41, 0.32)
HEAD_Y = 2.015  # the middle of the head above the feet (standing, bob 0)
CHIN_Y = HEAD_Y - HEAD_SIZE[1] / 2.0  # 1.81: where the head starts
EYE_LINE = 0.06  # the eyes above the middle of the head (2.075 above the feet)
EYE_X = 0.066  # ... either side of the nose
BROW_LINE = EYE_LINE + 0.042
MOUTH_LINE = -0.106  # the line of the lips
NOSE_Y = -0.015  # the middle of the nose
EAR_Y = 0.015
HAIRLINE_FRONT = 0.147  # metres above the middle: a good high forehead

P_FRONT = 2.3  # the cross-section's roundness over the face ...
P_BACK = 3.3  # ... and behind the ears (squarer: the back of the skull is flatter)
CAP = (0.17, 0.35)  # the share of the height each rounded end takes (chin, crown)
Q = (2.3, 2.45)  # how round each end is (bigger: flatter)
_H = HEAD_SIZE[1]


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def width(t: float) -> float:
    """The head's width (of the box) from the chin (0) to the crown (1): a jaw narrowing to
    the chin, cheekbones, the broad skull."""
    jaw = 0.171 / _H  # (where the jaw ends: the same 17 cm above the chin as ever)
    if t < jaw:
        u = t / jaw
        return 0.8 + 0.18 * (1.0 - (1.0 - u) ** 2)
    return 0.98 + 0.02 * math.sin(math.pi * min(1.0, (t - jaw) / (1.0 - jaw)))


def depth_front(t: float) -> float:
    """How far the face and forehead come forward of the middle (a share of the box)."""
    return 0.5 * (0.9 + 0.1 * math.sin(math.pi * min(1.0, t * 1.05)))


def depth_back(t: float) -> float:
    """How far back the skull goes: nearly as far all the way up (a flat-ish back), the
    nape a little in under it."""
    return 0.5 * (0.86 + 0.12 * _smooth(t / 0.45))


def forward(t: float) -> float:
    """The chin sits a little forward of the neck."""
    return 0.035 * (1.0 - t) ** 2


def rounding(t: float) -> float:
    """The rounded ends: 1 along the middle, down to 0 at the very chin and crown."""
    c0, c1 = CAP
    if t < c0:
        d = (c0 - t) / c0
        return max(0.0, 1.0 - d ** Q[0]) ** (1.0 / Q[0])
    if t > 1.0 - c1:
        d = (t - (1.0 - c1)) / c1
        return max(0.0, 1.0 - d ** Q[1]) ** (1.0 / Q[1])
    return 1.0


def _se(v: float, p: float) -> float:
    return math.copysign(abs(v) ** (2.0 / p), v)


def point(t: float, theta: float) -> tuple[float, float, float]:
    """A point of the head's surface in its unit box (-0.5 .. 0.5): height ``t``, and
    ``theta`` round it (0 its left side, 90 degrees straight ahead, 180 the right)."""
    s = rounding(t)
    c, n = math.cos(theta), math.sin(theta)
    p = P_FRONT if n >= 0.0 else P_BACK
    half = depth_front(t) if n >= 0.0 else depth_back(t)
    x = _se(c, p) * 0.5 * width(t) * s
    z = _se(n, p) * half * s + forward(t) * s
    return x, t - 0.5, z


def surface_z(x: float, y: float) -> float:
    """How far forward (metres) the face's surface is at (``x``, ``y``) metres from the
    middle of the head - where to set something on it (an eye's lens, a fringe)."""
    t = min(1.0, max(0.0, y / _H + 0.5))
    s = rounding(t)
    half_w = 0.5 * width(t) * s * HEAD_SIZE[0]
    if half_w <= 1e-9:
        return forward(t) * s * HEAD_SIZE[2]
    u = min(1.0, abs(x) / half_w)
    z = (1.0 - u**P_FRONT) ** (1.0 / P_FRONT) * depth_front(t) * s
    return (z + forward(t) * s) * HEAD_SIZE[2]


def hairline(ahead: float) -> float:
    """Where the hair stops (unit height of the head, from its middle) by which way round
    the head it is (``ahead``: 1 the front, 0 the sides, -1 the back): high on the
    forehead, just above the ears at the sides, down to the nape at the back."""
    front, side, back = HAIRLINE_FRONT / _H, 0.045 / _H, -0.09 / _H
    b = (front - back) / 2.0
    a = (front + back) / 2.0 - side
    return side + b * ahead + a * ahead * ahead


def midline(k: float) -> tuple[float, float]:
    """(y, z) in metres from the middle of the head along the top of the skull: ``k`` = 0
    at the front hairline, 0.5 over the crown, 1 at the nape."""
    front = hairline(1.0) + 0.5
    back = hairline(-1.0) + 0.5
    if k <= 0.5:  # up the front of the skull
        t = front + (1.0 - front) * (k / 0.5)
        _, y, z = point(t, math.pi / 2.0)
    else:  # down the back
        t = 1.0 - (1.0 - back) * ((k - 0.5) / 0.5)
        _, y, z = point(t, -math.pi / 2.0)
    return y * _H, z * HEAD_SIZE[2]


__all__ = [
    "BROW_LINE",
    "CHIN_Y",
    "EAR_Y",
    "EYE_LINE",
    "EYE_X",
    "HEAD_SIZE",
    "HEAD_Y",
    "MOUTH_LINE",
    "NOSE_Y",
    "hairline",
    "midline",
    "point",
    "surface_z",
]
