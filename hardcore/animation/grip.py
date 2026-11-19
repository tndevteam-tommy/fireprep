"""Grip model: how the wrist and fingers hold a given kind of edge in a given phase.

Two kinds of edge exist in the world:

* CRIMP - a thin edge: crevasse slots and any lip with less than ~25 cm of
  surface behind it.  The fingertips go into the slot / onto the lip in a half
  crimp; the hand hangs nearly in line with the forearm (wrist extension ~15°).
* LEDGE - a deep edge: wall tops, roof edges, sills, balcony and platform edges.
  The fingers drape over the top surface in an open-hand hook (extension ~26°).

The wrist bends a lot only when the body is *above* the hands: pulling the chest
up to the hold, propping the body on the palms while mantling or lowering
over an edge, pushing off a wall.  `hang_extension()` makes that a continuous
function of where the wrist is relative to the shoulder, so pull-ups and
lowers flow through the right angles without state-specific scripting.

Hand frames are built relative to the forearm axis the IK produced, so the
designed wrist angle is what the skeleton actually shows.
"""
import numpy as np
from math3d.geometry import cross3
import config
from animation.finger_solver import HandPoseType, POSES, blend_angles

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


class GripKind:
    CRIMP = "crimp"
    LEDGE = "ledge"


def grip_kind(rail):
    """Thin edge (crimp) or deep edge (hook over a ledge)."""
    if rail is None:
        return GripKind.LEDGE
    if rail.is_crevasse:
        return GripKind.CRIMP
    if rail.owner is not None:
        size = rail.owner.bounds.size
        depth = float(np.dot(size, np.abs(rail.normal)))
        if depth < config.GRIP_LEDGE_MIN_DEPTH:
            return GripKind.CRIMP
    return GripKind.LEDGE


def hang_ext_for(kind):
    return config.WRIST_EXT_HANG_CRIMP if kind == GripKind.CRIMP else config.WRIST_EXT_HANG_LEDGE


def hold_pose_for(kind):
    return HandPoseType.CRIMP if kind == GripKind.CRIMP else HandPoseType.HOOK


def wrist_offset(rail, kind=None):
    """Wrist position relative to the grip point on the edge for a hanging hold."""
    kind = kind or grip_kind(rail)
    if kind == GripKind.CRIMP:
        return rail.normal * config.WRIST_OUT_CRIMP - UP * config.WRIST_DROP_CRIMP
    return rail.normal * config.WRIST_OUT_LEDGE - UP * config.WRIST_DROP_LEDGE


def press_offset(rail):
    """Wrist position relative to the grip point when the palm presses the top surface behind the edge."""
    return -rail.normal * config.WRIST_PRESS_IN + UP * config.WRIST_PRESS_UP


def hang_extension(kind, wrist_y, shoulder_y):
    """
    Wrist extension (degrees) for a hand holding an edge, from where the hand is
    relative to the shoulder: a full hang (hand well above the shoulder) is nearly
    straight; as the body rises and the hand comes down to the chest the wrist
    extends toward the prop angle.
    """
    rise = float(wrist_y - shoulder_y)
    t = float(np.clip((rise - config.WRIST_EXT_LOW_RISE) / (config.WRIST_EXT_HIGH_RISE - config.WRIST_EXT_LOW_RISE), 0.0, 1.0))
    return config.WRIST_EXT_PROP + (hang_ext_for(kind) - config.WRIST_EXT_PROP) * t


class HandSpec:
    """What one hand wants to do this frame; resolved into a frame after the IK."""
    HANG = "hang"      # hooked on an edge, wrist bent `ext` degrees toward the wall
    PRESS = "press"    # palm flat on the surface behind the edge
    OPEN = "open"      # hand open along the forearm (hops, reaching)
    FREE = "free"      # explicit world frame (rest pose, pre-reach, pushing off)
    RELEASE = "release"  # easing from a press on an edge into a target frame (letting go of the surface)

    def __init__(self, mode, rail=None, ext=None, angles=None, frame=None, blend=0.0, bend=None, roll=0.0,
                 wall_clear=0.0):
        self.mode = mode
        self.wall_clear = wall_clear  # 0..1: how much an open hand flexes back to keep its fingers off the wall
        self.wall_dist = None         # wrist distance from the wall plane (set by the rig for open hands)
        self.rail = rail
        self.ext = ext
        self.angles = angles
        self.frame = frame
        self.blend = blend      # HANG->PRESS blend for mantle transitions
        self.bend = bend        # bend direction for rail-less forearm-relative hands (rest / locomotion)
        self.roll = roll        # pronation about the finger axis (degrees), + = palm toward local -X

    @property
    def effective_mode(self):
        """Mode as shown: a hook fully blended into a palm press reads as PRESS."""
        if self.mode == HandSpec.HANG and self.blend >= 0.99:
            return HandSpec.PRESS
        return self.mode

    @staticmethod
    def hang(rail, wrist_y, shoulder_y, kind=None, angles=None):
        kind = kind or grip_kind(rail)
        ext = hang_extension(kind, wrist_y, shoulder_y)
        return HandSpec(HandSpec.HANG, rail, ext, angles or POSES[hold_pose_for(kind)])

    @staticmethod
    def press(rail, angles=None):
        return HandSpec(HandSpec.PRESS, rail, config.WRIST_EXT_PRESS, angles or POSES[HandPoseType.MANTLE_PALM])

    @staticmethod
    def open(rail=None, angles=None):
        return HandSpec(HandSpec.OPEN, rail, config.WRIST_EXT_OPEN, angles or POSES[HandPoseType.OPEN])

    @staticmethod
    def free(frame, angles):
        return HandSpec(HandSpec.FREE, None, None, angles, frame)

    @staticmethod
    def release(rail, target_frame, angles, blend):
        """Hand leaving a palm press: wrist extension and bend direction ease toward `target_frame`."""
        return HandSpec(HandSpec.RELEASE, rail, config.WRIST_EXT_PRESS, angles, target_frame, blend)


def _frame(fingers, palm):
    y = np.asarray(fingers, dtype=np.float32)
    y = y / (np.linalg.norm(y) + 1e-7)
    z = np.asarray(palm, dtype=np.float32)
    z = z - y * float(np.dot(z, y))
    n = float(np.linalg.norm(z))
    if n < 1e-5:
        z = cross3(y, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if np.linalg.norm(z) < 1e-5:
            z = cross3(y, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        n = float(np.linalg.norm(z))
    z = z / n
    x = cross3(y, z)
    m = np.identity(4, dtype=np.float32)
    m[0:3, 0] = x
    m[0:3, 1] = y
    m[0:3, 2] = z
    return m


def resolve_hand(spec, forearm_axis):
    """
    Turn a HandSpec into (wrist rotation 4x4, finger angles) given the forearm axis
    (elbow -> wrist) that the IK produced.
    """
    if spec.mode == HandSpec.FREE:
        return spec.frame, spec.angles
    f = np.asarray(forearm_axis, dtype=np.float32)
    f = f / (np.linalg.norm(f) + 1e-7)
    if spec.rail is not None:
        w = -spec.rail.normal
    elif spec.bend is not None:
        w = np.asarray(spec.bend, dtype=np.float32)
    else:
        w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    # component of "into the wall" perpendicular to the forearm: the plane the wrist bends in
    w_perp = w - f * float(np.dot(w, f))
    if np.linalg.norm(w_perp) < 1e-4:
        w_perp = -UP - f * float(np.dot(-UP, f))
    w_perp = w_perp / (np.linalg.norm(w_perp) + 1e-7)

    if spec.mode == HandSpec.PRESS:
        return _frame(w, -UP), spec.angles

    if spec.mode == HandSpec.RELEASE:
        # The wrist angle is a function of the blend only (never of the swinging forearm): it eases
        # from the press extension to a relaxed angle while the bend direction turns from "into the
        # wall" toward the target frame's direction.  The free-hand smoothing takes it from there.
        tf = spec.frame[0:3, 1]
        t_perp = tf - f * float(np.dot(tf, f))
        tn = float(np.linalg.norm(t_perp))
        t_perp = t_perp / tn if tn > 1e-5 else w_perp
        b = float(np.clip(spec.blend, 0.0, 1.0))
        bend_dir = w_perp * (1.0 - b) + t_perp * b
        bend_dir = bend_dir - f * float(np.dot(bend_dir, f))
        bend_dir = bend_dir / (np.linalg.norm(bend_dir) + 1e-7)
        ext = np.radians(config.WRIST_MAX_ANGLE_DEG * (1.0 - b) + config.WRIST_EXT_RELAXED * b)
        fingers = f * np.cos(ext) + bend_dir * np.sin(ext)
        palm = (-UP) * (1.0 - b) + spec.frame[0:3, 2] * b
        return _frame(fingers, palm), spec.angles

    th = np.radians(spec.ext)
    if spec.mode == HandSpec.OPEN and spec.rail is not None and spec.wall_clear > 0.0:
        # An open hand travelling along a wall keeps its fingertips off the surface: the wrist
        # flexes back (within limits) until the fingers only point into the wall as far as the
        # wrist's distance from it allows
        a, b = float(np.dot(f, w)), float(np.dot(w_perp, w))
        allowed = 1.0 if spec.wall_dist is None else float(np.clip((spec.wall_dist - 0.03) / config.HAND_OPEN_LEN, 0.0, 1.0))
        if b > 1e-5 and allowed < 1.0:
            th_par = float(np.arctan2(b, a)) - float(np.arccos(allowed))
            th_par = max(th_par, -np.radians(config.WRIST_FLEX_CLEAR_DEG))
            if th_par < th:
                th = th + (th_par - th) * float(np.clip(spec.wall_clear, 0.0, 1.0))
    fingers = f * np.cos(th) + w_perp * np.sin(th)
    palm = w_perp * np.cos(th) - f * np.sin(th)
    if spec.roll:
        # pronation / supination: rotate the palm normal about the finger axis
        r = np.radians(spec.roll)
        side = cross3(fingers, palm)
        palm = palm * np.cos(r) + side * np.sin(r)
    if spec.mode == HandSpec.HANG and spec.blend > 0.0:
        # easing from the hook into a palm press (mantling over the edge)
        b = float(np.clip(spec.blend, 0.0, 1.0))
        fingers = fingers * (1.0 - b) + w * b
        palm = palm * (1.0 - b) - UP * b
        angles = blend_angles(spec.angles, POSES[HandPoseType.MANTLE_PALM], b)
        return _frame(fingers, palm), angles
    return _frame(fingers, palm), spec.angles
