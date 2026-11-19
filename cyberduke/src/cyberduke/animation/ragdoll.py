"""A knocked-out body collapsing - a dead weight, against whatever is around it.

The body is the renderer's own boxes (``animation.skeleton``), tested in 3D against the
obstacles around it at every step, so no part of it - head, boots, an arm, the holstered
blaster - ever passes through a wall, a crate, a planter or the floor.

How it goes down
================
* **A direction.** The current drops the guard one way - back, forward, or to its left or
  right (``direction``; random by default). The knees buckle and the hips sink as the
  trunk tips that way under gravity, the limbs going slack.
* **Nothing in the way:** it ends flat - on its back, face down (head turned to one side),
  or on its side (knees drawn up, the lower arm out in front, the top leg across onto the
  floor).
* **Something in the way** (the fall stalls against it):

  - something low it can go over - a planter, a low crate - with the head already above
    its top: the body stays draped over it;
  - anything else (a wall, a rack, a tall crate): a heavy body does not stay propped up
    against it. It topples off to the left or right - whichever side there is room to
    lie on - sliding down what it leaned on, and ends on the floor on its side (or on its
    back / front when it first fell sideways into the thing);
  - no room either side (a tight corner): the hips skid away from it until the body can
    lie back as far as it goes.

* The body always rests on its lowest point: every frame it is lowered (or raised) until
  that point touches the floor, so nothing hovers and nothing sinks.

``Collapse.step(dt)`` advances it; ``BodyState`` is what the renderer draws (the hip
joint's world position, the trunk's rotation, the joint angles).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from cyberduke.animation.pose import HumanoidPose
from cyberduke.animation.skeleton import GUARD_KIT, HIP_HEIGHT, BodyKit, body_boxes, box_points
from cyberduke.config import HACKING
from cyberduke.core.geometry import AABB
from cyberduke.core.interp import lerp, smoothstep

DIRECTIONS = ("back", "front", "left", "right")
_RELATIVE_HEADING = {"back": 180.0, "front": 0.0, "left": 90.0, "right": -90.0}

FALL_DURATION = HACKING.knockout_fall_duration  # shock over -> down (or against something)
TOPPLE_DURATION = 0.85  # propped against something: tipping off it to the side
SLIDE_DURATION = 0.9  # no room to either side: the hips skid away from it
STALL_TIME = 0.1  # the fall has been held up this long: it has met something
CONTACT_MARGIN = 0.012  # how close a body point may come to an obstacle
FLOOR_SINK = 0.008  # the lowest point rests this far into the floor: in contact
REACH = 3.2  # obstacles further than this from where the guard stood cannot be touched
# Lying on its side: (tilt, twist about its spine - positive rolls it back towards lying on
# its back). Tuned, with the side poses in ``rest_pose``, so the shoulder, ribs, hip and
# head all rest on the floor (the right side has the holster under it).
SIDE_REST = {"side_left": (88.0, 0.0), "side_right": (92.0, 40.0)}
# How far each way of lying tips the trunk from upright (not quite 90: the head and the
# chest, the shoulders and the hips, do not all stick out equally far)
REST_TILT = {"supine": 86.0, "prone": 88.0, "side_left": 88.0, "side_right": 92.0}
DRAPE_TOP = 0.8  # something no taller than this can be fallen over (and hung over)
MAX_SLIDE = 0.8
SLUMP_UNTIL = 40.0  # held up before tipping this far: it slides down what holds it first
SLUMP_MAX = 78.0  # ... until it is sitting (the knees' buckle, degrees)
SLUMP_RATE = 110.0  # degrees of buckle a second
SQUAT = 55.0  # buckled this far the knees are as bent as they go: further, it sits down

# What each way of falling ends as: "supine" (on the back), "prone" (face down) or on the
# left / right side
_FAMILY = {"back": "supine", "front": "prone", "left": "side_left", "right": "side_right"}
_HIP_SHIFT = {"back": 0.24, "front": 0.34, "left": 0.16, "right": 0.16}  # hips move that way


# ----------------------------------------------------------------------------- geometry
def heading(yaw_deg: float, relative: float) -> np.ndarray:
    """Unit world (x, 0, z) vector ``relative`` degrees from the facing of yaw ``yaw_deg``
    (NPC yaw 0 faces +z; +90 relative is the body's left)."""
    r = math.radians(yaw_deg + relative)
    return np.array((math.sin(r), 0.0, math.cos(r)))


def yaw_rotation(yaw_deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _spin(angle_deg: float) -> np.ndarray:
    """Rotation about the body's own spine (its +y)."""
    return yaw_rotation(angle_deg)


def tilt(direction: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotation tipping the vertical ``angle_deg`` towards the horizontal ``direction``."""
    dx, dz = float(direction[0]), float(direction[2])
    n = math.hypot(dx, dz) or 1.0
    ax, az = dz / n, -dx / n  # up x direction
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return np.array(
        (
            (c + ax * ax * (1 - c), -az * s, ax * az * (1 - c)),
            (az * s, c, -ax * s),
            (az * ax * (1 - c), ax * s, c + az * az * (1 - c)),
        )
    )


def axis_angle(r: np.ndarray) -> tuple[float, tuple[float, float, float]]:
    """(degrees, unit axis) of a rotation matrix - what ``DrawContext.rotate`` takes."""
    cos_a = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if angle < 1e-6:
        return 0.0, (0.0, 1.0, 0.0)
    if math.pi - angle < 1e-4:  # 180 degrees: the axis is the +1 eigenvector
        w, v = np.linalg.eigh((r + r.T) / 2.0)
        axis = v[:, int(np.argmax(w))]
    else:
        axis = np.array((r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]))
        axis = axis / (2.0 * math.sin(angle))
    return math.degrees(angle), (float(axis[0]), float(axis[1]), float(axis[2]))


# ----------------------------------------------------------------------------- the body
@dataclass(frozen=True, eq=False)
class BodyState:
    """A collapsing body at one moment: the hip joint's world position, the trunk's
    rotation (body frame -> world) and the joint angles."""

    hip: tuple[float, float, float]
    rotation: np.ndarray
    pose: HumanoidPose
    scale: float = 1.0  # the person's size (``HumanoidSpec.height``)

    @property
    def fall_angle(self) -> float:
        """How far the trunk has tipped from upright (degrees; 90 = lying along the floor)."""
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.rotation[1, 1])))))

    def matrix(self) -> np.ndarray:
        """Body frame (feet at the origin, as the renderer draws it) -> world."""
        m = np.identity(4)
        m[:3, :3] = self.rotation * self.scale
        m[:3, 3] = np.array(self.hip) - self.rotation @ np.array(
            (0.0, HIP_HEIGHT * self.scale, 0.0)
        )
        return m

    def point(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """World position of a point of the body frame."""
        p = self.matrix() @ np.array((x, y, z, 1.0))
        return (float(p[0]), float(p[1]), float(p[2]))


class Surroundings:
    """The obstacles a body might touch (as arrays, margin included) and the floor."""

    def __init__(self, obstacles: Iterable[AABB], x: float, z: float, floor_y: float) -> None:
        near = [
            b for b in obstacles
            if b.max_x > x - REACH and b.min_x < x + REACH and b.max_z > z - REACH
            and b.min_z < z + REACH and b.max_y > floor_y + 0.005 and b.min_y < floor_y + 2.6
        ]  # fmt: skip
        self.boxes = near
        self.floor_y = floor_y
        m = CONTACT_MARGIN
        self.lo = np.array([(b.min_x - m, b.min_y - m, b.min_z - m) for b in near]).reshape(-1, 3)
        self.hi = np.array([(b.max_x + m, b.max_y + m, b.max_z + m) for b in near]).reshape(-1, 3)

    def touching(self, points: np.ndarray) -> list[int]:
        """Indices of the obstacles any of the points are in (within the margin)."""
        if not len(self.boxes):
            return []
        inside = (points[:, None, :] > self.lo[None]) & (points[:, None, :] < self.hi[None])
        return [int(i) for i in np.flatnonzero(inside.all(axis=2).any(axis=0))]

    def depth(self, points: np.ndarray) -> float:
        """How far the deepest point is inside an obstacle (its margin included); 0 when
        none is."""
        if not len(self.boxes):
            return 0.0
        a = points[:, None, :] - self.lo[None]
        b = self.hi[None] - points[:, None, :]
        d = np.minimum(a, b).min(axis=2)
        return float(max(0.0, d.max()))

    def clear(self, points: np.ndarray) -> bool:
        if not len(self.boxes):
            return True
        inside = (points[:, None, :] > self.lo[None]) & (points[:, None, :] < self.hi[None])
        return not bool(inside.all(axis=2).any())


# ----------------------------------------------------------------------------- poses
def _leg_aim(rotation: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(hip pitch, hip roll) that point a leg along the world direction ``target``."""
    d = rotation.T @ (target / (np.linalg.norm(target) or 1.0))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(d[2])))))
    roll = math.degrees(math.atan2(float(d[0]), -float(d[1])))
    return pitch, roll


def _blend(a: HumanoidPose, b: HumanoidPose, u: float) -> HumanoidPose:
    """Joint angles interpolated from ``a`` to ``b``."""
    u = max(0.0, min(1.0, u))

    def mix(p, q):
        return [lerp(x, y, u) for x, y in zip(p, q, strict=False)]

    return HumanoidPose(
        bob=lerp(a.bob, b.bob, u),
        head_pitch=lerp(a.head_pitch, b.head_pitch, u),
        head_yaw=lerp(a.head_yaw, b.head_yaw, u),
        head_roll=lerp(a.head_roll, b.head_roll, u),
        arm_l=mix(a.arm_l, b.arm_l),
        arm_r=mix(a.arm_r, b.arm_r),
        leg_l=mix(a.leg_l, b.leg_l),
        leg_r=mix(a.leg_r, b.leg_r),
        foot_l=lerp(a.foot_l, b.foot_l, u),
        foot_r=lerp(a.foot_r, b.foot_r, u),
        hip_roll_l=lerp(a.hip_roll_l, b.hip_roll_l, u),
        hip_roll_r=lerp(a.hip_roll_r, b.hip_roll_r, u),
        shrug_l=lerp(a.shrug_l, b.shrug_l, u),
        shrug_r=lerp(a.shrug_r, b.shrug_r, u),
        has_prop=True,
        prop_pos=[-0.28, 0.98, 0.0],
        prop_rot=mix(a.prop_rot, b.prop_rot),
    )


def _holstered(pose: HumanoidPose) -> HumanoidPose:
    """The blaster rides along the right thigh (hanging straight down it went through the
    floor under a sitting body)."""
    pose.prop_rot = [pose.leg_r[0] + 80.0, 0.0, 0.0]
    return pose


def planted_pose(rotation: np.ndarray, yaw_deg: float, buckle: float, front: bool) -> HumanoidPose:
    """Legs still under the body (feet on the floor) while the trunk tips: the thighs stay
    near vertical, swinging forward by ``buckle`` degrees as the knees give (``front``: a
    kneel, the shins going back along the floor)."""
    forward = heading(yaw_deg, 0.0)
    down = np.array((0.0, -1.0, 0.0))
    angle = math.radians(0.0 if front else buckle)
    pitch, roll = _leg_aim(rotation, down * math.cos(angle) + forward * math.sin(angle))
    if front:
        knee = 90.0 * min(1.0, buckle / 40.0)
    elif buckle <= SQUAT:
        knee = 2.0 * buckle
    else:  # past a squat the seat goes down onto the floor and the legs slide out in front
        knee = max(12.0, 2.0 * SQUAT - (buckle - SQUAT) * 4.0)
    pose = HumanoidPose(
        head_pitch=12.0,
        head_yaw=6.0,
        head_roll=4.0,
        arm_l=[-20.0, -8.0, 22.0, -30.0],
        arm_r=[-18.0, 8.0, -22.0, -30.0],
        leg_l=[pitch, 0.0, knee],
        leg_r=[pitch, 0.0, knee],
        hip_roll_l=roll + 2.0,
        hip_roll_r=roll - 2.0,
        foot_l=0.8 * knee if front else -buckle,
        foot_r=0.8 * knee if front else -buckle,
        has_prop=True,
        prop_pos=[-0.28, 0.98, 0.0],
    )
    return _holstered(pose)


def rest_pose(family: str, rotation: np.ndarray, variant: int = 0) -> HumanoidPose:
    """A limp body lying as ``family`` says, trunk turned by ``rotation``: on its back the
    legs lie out along the floor (aimed away from the head, whatever the trunk's tilt - so
    a body draped over a planter still has its legs on the floor), face down the head is
    turned to one side, on its side the knees come up, the lower arm lies out in front and
    the top leg crosses onto the floor in front of the lower one."""
    flip = -1.0 if variant % 2 else 1.0
    if family in ("supine", "prone"):
        spine = rotation @ np.array((0.0, 1.0, 0.0))
        along = np.array((-spine[0], 0.0, -spine[2]))
        if np.linalg.norm(along) < 0.2:  # still nearly upright: legs straight down
            along = np.array((0.0, -1.0, 0.0))
        aim = along / np.linalg.norm(along)  # along the floor, level with the hips
        pitch, roll = _leg_aim(rotation, aim)
        if family == "supine":
            pose = HumanoidPose(
                head_pitch=-6.0, head_yaw=25.0 * flip, head_roll=8.0 * flip,
                arm_l=[12.0, 0.0, 22.0, -18.0], arm_r=[12.0, 0.0, -20.0, -12.0],
                leg_l=[pitch - 7.0, 0.0, 14.0], leg_r=[pitch - 5.0, 0.0, 8.0],
                hip_roll_l=roll + 7.0, hip_roll_r=roll - 6.0, foot_l=-20.0, foot_r=-20.0,
            )  # fmt: skip
        else:
            pose = HumanoidPose(
                head_pitch=0.0, head_yaw=70.0 * flip, head_roll=0.0,
                arm_l=[6.0, 0.0, 14.0, -6.0], arm_r=[6.0, 0.0, -14.0, -6.0],
                leg_l=[pitch - 4.0, 0.0, 20.0], leg_r=[pitch - 2.0, 0.0, 20.0],
                hip_roll_l=roll + 6.0, hip_roll_r=roll - 6.0, foot_l=40.0, foot_r=40.0,
            )  # fmt: skip
    elif family == "side_left":
        # lying on its left side: knees drawn up, the top (right) leg across in front onto
        # the floor, the top arm draped over the chest
        pose = HumanoidPose(
            head_pitch=10.0, head_yaw=8.0, head_roll=-20.0,
            arm_l=[-30.0, 0.0, 0.0, -20.0], arm_r=[-38.0, 0.0, 18.0, -62.0],
            leg_l=[-50.0, 0.0, 40.0], leg_r=[-90.0, 0.0, 70.0],
            hip_roll_l=-6.0, hip_roll_r=24.0, shrug_l=0.14, foot_l=20.0, foot_r=20.0,
        )  # fmt: skip
    else:
        # on its right side - that is the holster side, so it has rolled a little on over
        # towards its back, the top (left) leg lying along the lower one
        pose = HumanoidPose(
            head_pitch=10.0, head_yaw=-8.0, head_roll=20.0,
            arm_l=[20.0, 0.0, -18.0, -40.0], arm_r=[0.0, 0.0, 0.0, -10.0],
            leg_l=[-10.0, 0.0, 20.0], leg_r=[-10.0, 0.0, 40.0],
            hip_roll_l=-24.0, hip_roll_r=0.0, shrug_r=0.10, foot_l=20.0, foot_r=20.0,
        )  # fmt: skip
    pose.has_prop = True
    pose.prop_pos = [-0.28, 0.98, 0.0]
    return _holstered(pose)


# A limb that would go into something folds out of the way: these, in order.
_FOLDS: dict[str, tuple[tuple[float, ...], ...]] = {
    "arm_l": (
        (-40.0, 0.0, -10.0, -95.0),
        (-30.0, 0.0, -30.0, -130.0),
    ),  # bent, then across the chest
    "arm_r": ((-40.0, 0.0, 10.0, -95.0), (-30.0, 0.0, 30.0, -130.0)),
    "leg_l": ((-55.0, 0.0, 95.0), (-95.0, 0.0, 130.0)),  # knee up, then drawn in
    "leg_r": ((-55.0, 0.0, 95.0), (-95.0, 0.0, 130.0)),
}


# Lying down, a limb may not hold the body up off the floor: folded forwards (or tucked
# under the chest) an arm of a body lying face down points into the floor, a knee drawn up
# digs into it - and the body rested on that limb, floating. When a limb would do that it
# is laid flat instead: the first of these that keeps clear of everything else. (Arm
# angles: shoulder pitch - negative swings it forwards, to the body's front -, yaw, roll -
# positive out to the left -, elbow; a leg's: hip pitch, yaw, knee - positive bends the
# shin back. ``None`` for a leg: as the way of lying has it.)
_FLAT_ARMS = (
    (-165.0, 0.0, 12.0, -12.0),  # overhead along the floor (face down)
    (165.0, 0.0, 12.0, 0.0),  # overhead along the floor (on the back)
    (6.0, 0.0, 3.0, -4.0),  # down along the side
    (-120.0, 0.0, 10.0, -60.0),  # up by the head (on a side)
    (-165.0, 0.0, 35.0, -40.0),  # overhead, bent out
    (120.0, 0.0, 45.0, -10.0),
)
_FLAT_KNEES = {"prone": (70.0, 115.0), "supine": (8.0, 25.0), "side": (60.0, 100.0)}
PROP_TOLERANCE = 0.04  # a limb may reach this far below the trunk's lowest point
LYING_MARGIN = 4.0  # within this many degrees of its way of lying's tilt, the body is down


def _with_limb(
    pose: HumanoidPose,
    limb: str,
    angles: tuple[float, ...],
    hip_roll: float = 0.0,
    foot: float = 20.0,
) -> HumanoidPose:
    out = _blend(pose, pose, 0.0)
    if limb == "arm_l":
        out.arm_l = list(angles)
    elif limb == "arm_r":
        out.arm_r = list(angles)
    elif limb == "leg_l":
        out.leg_l, out.hip_roll_l, out.foot_l = list(angles), hip_roll, foot
    else:
        out.leg_r, out.hip_roll_r, out.foot_r = list(angles), hip_roll, foot
    return out


def _flat_kind(family: str) -> str:
    return family if family in ("prone", "supine") else "side"


def flat_limbs(family: str, rotation: np.ndarray, limb: str, variant: int = 0) -> list:
    """The ways ``limb`` can lie flat on the floor for a body lying as ``family`` says:
    (angles, hip roll, foot) to try, in order."""
    kind = _flat_kind(family)
    if limb.startswith("arm"):  # (the same list however it lies: a fold's index keeps its
        mirror = 1.0 if limb == "arm_l" else -1.0  # meaning when it rolls over)
        return [((a, b, c * mirror, d), 0.0, 0.0) for a, b, c, d in _FLAT_ARMS]
    rest = rest_pose(family, rotation, variant)
    angles, roll = (
        (rest.leg_l, rest.hip_roll_l) if limb == "leg_l" else (rest.leg_r, rest.hip_roll_r)
    )
    out = []
    spread = 1.0 if limb == "leg_l" else -1.0  # (positive hip roll swings the left leg out)
    for splay in (0.0, 25.0, -20.0, 45.0):
        for knee in (*_FLAT_KNEES[kind], angles[2]):
            out.append(((angles[0], 0.0, knee), roll + spread * splay, 20.0))
    if kind == "side":  # (no room along the floor - a tight corner: the knee drawn right up)
        for splay in (0.0, 20.0, -20.0, 40.0):
            for hip, knee in _TUCKED:
                out.append(((hip, 0.0, knee), roll + spread * splay, 20.0))
    return out


_TUCKED = ((-95.0, 140.0), (-120.0, 80.0), (-40.0, 140.0), (-120.0, 0.0))


# ----------------------------------------------------------------------------- the collapse
@dataclass(frozen=True, eq=False)
class _Placed:
    state: BodyState
    points: np.ndarray
    folds: dict[str, int]


@dataclass
class _Phase:
    name: str  # "tip", "topple", "slide", "rest"
    time: float = 0.0
    stalled: float = 0.0


@dataclass(eq=False)
class Collapse:
    """One body going down. Build it when the shock ends, then ``step(dt)`` every frame."""

    x: float
    y: float
    z: float
    yaw: float
    world: Surroundings
    direction: str
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    start_pose: HumanoidPose = field(default_factory=HumanoidPose)
    kit: BodyKit = GUARD_KIT

    def __post_init__(self) -> None:
        self.v1 = heading(self.yaw, _RELATIVE_HEADING[self.direction])
        self.v2: np.ndarray | None = None  # where it topples off what stopped it
        self.family = _FAMILY[self.direction]
        self.variant = self.rng.randrange(2)
        self.theta = 0.0  # tipped this far towards v1
        self.phi = 0.0  # ... then this far towards v2
        self.phi_rest = 90.0
        self.theta_rest = REST_TILT[self.family]
        self.offset = np.zeros(3)  # hips' horizontal shift from the spot it stood on
        self.folded: dict[str, int] = {}  # limbs folded out of the way (and how far)
        self._memo: dict[tuple, _Placed] = {}
        self.slump = 0.0  # held up early: the knees keep giving, sliding it down what held it
        self.shift = self._plan_shift()
        self.slide_from = np.zeros(3)
        self.slide_to = np.zeros(3)
        self.lean_from = self.lean_to = 0.0
        self.limbs_from: HumanoidPose | None = None  # the pose when it started to topple
        self.draped = False
        self.slid = False
        self.phase = _Phase("tip")
        self._adopt(self._make(0.0, 0.0, self.offset, 0.0))

    # ------------------------------------------------------------------ construction
    @classmethod
    def begin(
        cls,
        x: float,
        y: float,
        z: float,
        yaw: float,
        obstacles: Iterable[AABB],
        direction: str | None = None,
        rng: random.Random | None = None,
        start_pose: HumanoidPose | None = None,
        kit: BodyKit = GUARD_KIT,
    ) -> Collapse:
        rng = rng or random.Random(0)
        direction = direction or rng.choice(DIRECTIONS)
        if direction not in _RELATIVE_HEADING:
            raise ValueError(f"fall direction must be one of {DIRECTIONS}, got {direction!r}")
        start = start_pose or HumanoidPose(
            has_prop=True, prop_pos=[-0.28, 0.98, 0.0], prop_rot=[80.0, 0.0, 0.0]
        )
        return cls(x, y, z, yaw, Surroundings(obstacles, x, z, y), direction, rng, start, kit)

    # ------------------------------------------------------------------ the body at a moment
    def rotation(self, theta: float, phi: float, phi_rest: float | None = None) -> np.ndarray:
        """The trunk tipped ``theta`` towards where it fell, then ``phi`` towards where it
        toppled off to - and, going down onto its side, rolled a little about its spine."""
        r = yaw_rotation(self.yaw)
        if self.family in SIDE_REST:
            if self.v2 is not None:
                progress = phi / (phi_rest or self.phi_rest)
            else:
                progress = theta / self.theta_rest
            twist = SIDE_REST[self.family][1] * smoothstep(0.3, 1.0, progress)
            if twist:
                r = r @ _spin(twist)
        r = tilt(self.v1, theta) @ r
        if self.v2 is not None and phi > 0.0:
            r = tilt(self.v2, phi) @ r
        return r

    def _pose(self, rotation: np.ndarray, theta: float, phi: float, time: float) -> HumanoidPose:
        if self.phase.name in ("topple",) or (self.phase.name == "rest" and self.v2 is not None):
            u = smoothstep(0.0, 1.0, phi / max(self.phi_rest, 1e-6))
            target = rest_pose(self.family, rotation, self.variant)
            return _holstered(_blend(self.limbs_from or target, target, u))
        front = self.direction == "front"
        buckle = max(self.slump, 40.0 * smoothstep(0.0, 0.45 * FALL_DURATION, time))
        planted = planted_pose(rotation, self.yaw, buckle, front)
        start = _blend(self.start_pose, planted, smoothstep(0.0, 0.25, time))
        progress = smoothstep(0.25, 1.0, theta / self.theta_rest)
        if self.draped:
            progress = 1.0
        rest = rest_pose(self.family, rotation, self.variant)
        if front:  # stays kneeling until it is well over
            progress = smoothstep(0.45, 1.0, theta / self.theta_rest)
        pose = _blend(planted if time > 0.25 else start, rest, progress)
        return _holstered(pose)

    def _make_slump(self, slump: float, time: float) -> _Placed:
        saved, self.slump = self.slump, slump
        try:
            return self._make(self.theta, 0.0, self.offset, time)
        finally:
            self.slump = saved

    def _make(self, theta: float, phi: float, offset: np.ndarray, time: float) -> _Placed:
        key = (theta, phi, float(offset[0]), float(offset[2]), time, self.slump, self.phase.name,
               self.family, self.draped)  # fmt: skip
        found = self._memo.get(key)
        if found is None:
            rotation = self.rotation(theta, phi)
            pose = self._pose(rotation, theta, phi, time)
            found = self._memo[key] = self._placed(rotation, pose, offset)
        return found

    def _placed(self, rotation: np.ndarray, pose: HumanoidPose, offset: np.ndarray) -> _Placed:
        """The body with its hips over ``offset``, lowered onto its lowest point. A limb that
        would go into something is folded out of the way instead (an arm across the chest,
        a knee drawn up) - and stays folded for the rest of the fall. Lying down, no limb
        holds the body up off the floor: one that would is laid flat (``flat_limbs``), so
        the trunk rests on the floor."""
        probe = BodyState((self.x + offset[0], 0.0, self.z + offset[2]), rotation, pose)
        base = probe.matrix()
        groups = {
            g: box_points(body_boxes(pose, self.kit, base, (g,)))
            for g in ("trunk", "arm_l", "arm_r", "leg_l", "leg_r")
        }
        floor = self.world.floor_y - FLOOR_SINK
        trunk_lift = floor - float(groups["trunk"][:, 1].min())
        tipped = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[1, 1])))))
        lying = tipped >= REST_TILT[self.family] - LYING_MARGIN and not self.draped
        if lying:  # ... with nothing under the trunk but the floor
            trunk = groups["trunk"].copy()
            trunk[:, 1] += trunk_lift
            lying = self.world.clear(trunk)
        start_pose, start_groups = pose, dict(groups)
        lift = trunk_lift if lying else floor - min(float(p[:, 1].min()) for p in groups.values())
        for _ in range(3):  # (a limb moved can move the lowest point: settle the lift)
            pose, groups = start_pose, dict(start_groups)
            folds: dict[str, int] = {}
            for limb in _FOLDS:
                folds[limb], pose, groups[limb] = self._choose_limb(
                    limb, pose, groups[limb], base, rotation, lift, lying
                )
            if not lying:
                pose = self._sit_down(pose, groups, folds, base, lift)
            settled = floor - min(float(p[:, 1].min()) for p in groups.values())
            if abs(settled - lift) < 1e-3:
                break
            lift = settled
        pose = _holstered(pose)
        groups["blaster"] = box_points(body_boxes(pose, self.kit, base, ("blaster",)))
        pts = np.concatenate(list(groups.values()))
        lift = floor - float(pts[:, 1].min())
        pts[:, 1] += lift
        return _Placed(BodyState((probe.hip[0], lift, probe.hip[2]), rotation, pose), pts, folds)

    def _choose_limb(self, limb, pose, points, base, rotation, lift, lying):
        """(fold index, pose, the limb's points) for ``limb``: as it is if it is clear of
        everything (and, lying down, not under the trunk), else the first fold, else the
        first way of lying flat, that is. Failing all, the least bad that is clear."""
        tried = [None, *[(o, 0.0, 20.0) for o in _FOLDS[limb]]]
        tried += flat_limbs(self.family, rotation, limb, self.variant)
        floor = self.world.floor_y - FLOOR_SINK - PROP_TOLERANCE
        best = None
        for k in range(self.folded.get(limb, 0), len(tried)):
            trial, pts_k = pose, points
            if k > 0:
                angles, roll, foot = tried[k]
                trial = _with_limb(pose, limb, angles, roll, foot)
                pts_k = box_points(body_boxes(trial, self.kit, base, (limb,)))
            pts = pts_k.copy()
            pts[:, 1] += lift
            if not self.world.clear(pts):
                continue
            low = float(pts[:, 1].min())
            if not lying or low >= floor:
                return k, trial, pts_k
            if best is None or low > best[0]:  # (the least propping, failing all)
                best = (low, k, trial, pts_k)
        if best is not None:
            return best[1:]
        k = len(tried) - 1  # nothing is clear: the last resort, as it always was
        angles, roll, foot = tried[k]
        trial = _with_limb(pose, limb, angles, roll, foot)
        return k, trial, box_points(body_boxes(trial, self.kit, base, (limb,)))

    def _sit_down(self, pose, groups, folds, base, lift) -> HumanoidPose:
        """One leg folded out of the way and the other still planted, a half-fallen body
        stood on one leg: it sits down instead - the other leg comes up the same way."""
        for limb, other in (("leg_l", "leg_r"), ("leg_r", "leg_l")):
            k = folds.get(limb, 0)
            if 0 < k <= len(_FOLDS[other]) and folds.get(other, 0) < k:
                trial = _with_limb(pose, other, _FOLDS[other][k - 1])
                pts_k = box_points(body_boxes(trial, self.kit, base, (other,)))
                pts = pts_k.copy()
                pts[:, 1] += lift
                if self.world.clear(pts):
                    folds[other], pose, groups[other] = k, trial, pts_k
        return pose

    def _adopt(self, placed: _Placed) -> None:
        """This is where the body is now: a limb folded out of the way stays folded."""
        self.state = placed.state
        for limb, k in placed.folds.items():
            if k:
                self.folded[limb] = max(self.folded.get(limb, 0), k)

    def points(self, state: BodyState) -> np.ndarray:
        return box_points(body_boxes(state.pose, self.kit, state.matrix()))

    def clear(self, placed: _Placed) -> bool:
        return self.world.clear(placed.points)

    def fits(self, placed: _Placed, allowed: float) -> bool:
        """Clear - or, if the body already touches something, no deeper into it."""
        return self.world.depth(placed.points) <= allowed + 1e-4

    # ------------------------------------------------------------------ planning helpers
    def _plan_shift(self) -> np.ndarray:
        """How far the hips move the way the body falls as the knees give - no further
        than the upright body can go without touching anything."""
        best = np.zeros(3)
        steps = int(_HIP_SHIFT[self.direction] / 0.04)
        upright = yaw_rotation(self.yaw)
        pose = planted_pose(upright, self.yaw, 0.0, False)
        for k in range(1, steps + 1):
            off = self.v1 * (0.04 * k)
            if not self.clear(self._placed(upright, pose, off)):
                break
            best = off
        return best

    def _drape_ok(self, state: BodyState) -> bool:
        """Stopped by something low enough to go over, with the head already above it."""
        probe = self._make(min(90.0, self.theta + 3.0), self.phi, self.offset, 9.0)
        hit = self.world.touching(probe.points)
        if not hit:
            return False
        top = max(self.world.boxes[i].max_y for i in hit) - self.world.floor_y
        head = state.point(0.0, 2.0, 0.0)[1] - self.world.floor_y
        return top <= DRAPE_TOP and head >= top - 0.02 and self.theta >= 35.0

    def _topple_target(self, side: np.ndarray) -> tuple | None:
        """Toppling off towards ``side``: (final lean towards where it first fell, final
        tilt towards ``side``, final family, hip offset) of the first way of lying there
        that fits, or None. Leaning on something tall, the head slides off it sideways:
        the less lean towards it the body keeps, the further from it the head ends up."""
        saved = (self.v2, self.family, self.phase)
        self.v2 = side
        relative = math.degrees(math.atan2(side[0], side[2])) - self.yaw
        rel = (relative + 180.0) % 360.0 - 180.0
        if self.direction in ("back", "front"):
            family = "side_left" if rel > 0 else "side_right"
        else:
            family = "supine" if abs(rel) > 90.0 else "prone"
        want = math.cos(math.radians(REST_TILT[family]))
        self.family = family
        self.phase = _Phase("rest")
        found = None
        for lean in (self.theta, self.theta * 0.6, self.theta * 0.3, 0.0):
            ct = math.cos(math.radians(lean))
            phi = math.degrees(math.acos(max(-1.0, min(1.0, want / ct))))
            for slide in (0.0, 0.12, 0.25):
                off = self.offset - self.v1 * slide
                rot = self.rotation(lean, phi, phi_rest=phi)
                placed = self._placed(rot, _holstered(rest_pose(family, rot, self.variant)), off)
                if self.clear(placed):
                    found = (lean, phi, family, off)
                    break
            if found is not None:
                break
        self.v2, self.family, self.phase = saved
        return found

    def _resolve_stall(self) -> None:
        """The fall has been stopped by something."""
        if self.theta >= self.theta_rest - 10.0:
            self.phase = _Phase("rest")
            return
        if self._drape_ok(self.state):
            self.draped = True
            self.phase = _Phase("rest")
            placed = self._make(self.theta, 0.0, self.offset, 9.0)
            if self.clear(placed):
                self._adopt(placed)
            else:  # the slack limbs would not fit: keep them as they are
                self.draped = False
            return
        sides = [heading(self.yaw, _RELATIVE_HEADING[self.direction] + 90.0),
                 heading(self.yaw, _RELATIVE_HEADING[self.direction] - 90.0)]  # fmt: skip
        self.rng.shuffle(sides)
        for side in sides:
            found = self._topple_target(side)
            if found is not None:
                lean, phi, family, off = found
                self.v2, self.family, self.phi_rest = side, family, phi
                self.lean_from, self.lean_to = self.theta, lean
                self.limbs_from = self.state.pose
                self.slide_from, self.slide_to = self.offset.copy(), np.array(off)
                self.phase = _Phase("topple")
                return
        self._start_slide()

    def _start_slide(self) -> None:
        """No room to topple either way: the hips skid away from what it fell against,
        letting the trunk lie back further as they go."""
        self.slid = True
        self.slide_from = self.offset.copy()
        self.slide_to = self.offset - self.v1 * MAX_SLIDE
        self.phase = _Phase("slide")

    def _finish(self) -> None:
        """Done moving - unless that left it propped up against something, in which case
        it has one more go: the hips skid out from under it."""
        if self.state.fall_angle < 70.0 and not self.draped and not self.slid:
            self._start_slide()
        else:
            self.phase = _Phase("rest")

    # ------------------------------------------------------------------ stepping
    def _furthest(self, lo: float, hi: float, make, allowed: float = 0.0) -> float:
        """The largest angle in [lo, hi] at which the body fits (see ``fits``)."""
        if hi <= lo:
            return lo
        if self.fits(make(hi), allowed):
            return hi
        for _ in range(5):
            mid = (lo + hi) / 2.0
            if self.fits(make(mid), allowed):
                lo = mid
            else:
                hi = mid
        return lo

    def step(self, dt: float) -> BodyState:
        self._memo.clear()  # (what the body looks like at a given angle, within one step)
        ph = self.phase
        ph.time += dt
        if ph.name == "tip":
            t = ph.time
            free = self.theta_rest * min(1.0, t / FALL_DURATION) ** 2  # gravity
            offset = self.shift * smoothstep(0.0, 0.8 * FALL_DURATION, t)
            ahead = self._make(free, 0.0, offset, t)
            allowed = 0.0
            if self.clear(ahead):  # nothing in the way (the usual case): one test
                self.offset, theta = offset, free
            else:
                # (the limbs move on with time too: where the body already touches
                # something, nothing may take it any deeper in)
                allowed = self.world.depth(self._make(self.theta, 0.0, self.offset, t).points)
                # the hips move on as planned only while the body still fits there
                if self.fits(self._make(self.theta, 0.0, offset, t), allowed):
                    self.offset = offset
                theta = self._furthest(
                    self.theta, free, lambda a: self._make(a, 0.0, self.offset, t), allowed
                )
            moved = theta > self.theta + 1e-6
            self.theta = theta
            self._adopt(self._make(theta, 0.0, self.offset, t))
            if self.theta >= self.theta_rest - 0.5:
                self.phase = _Phase("rest")
            elif free - self.theta > 2.0 and not moved:
                # held up: the knees keep giving and the body slides down what holds it
                # until it is sitting - only then does it go any further
                ph.stalled += dt
                if self.theta < SLUMP_UNTIL and self.direction != "front":
                    slump = min(SLUMP_MAX, max(self.slump, 40.0) + SLUMP_RATE * dt)
                    if slump > self.slump and self.fits(self._make_slump(slump, t), allowed):
                        self.slump = slump
                        self._adopt(self._make(theta, 0.0, self.offset, t))
                        ph.stalled = 0.0 if slump < SLUMP_MAX else ph.stalled
                if ph.stalled >= STALL_TIME:
                    self._resolve_stall()
            else:
                ph.stalled = 0.0
        elif ph.name == "topple":
            u = min(1.0, ph.time / TOPPLE_DURATION)
            free = self.phi_rest * u**2
            # the head slides off what it leaned on as the body goes over
            self.theta = lerp(self.lean_from, self.lean_to, smoothstep(0.0, 1.0, u))
            offset = self.slide_from + (self.slide_to - self.slide_from) * smoothstep(0.0, 1.0, u)
            if self.clear(self._make(self.theta, free, offset, 9.0)):
                self.offset, phi = offset, free
            else:
                allowed = self.world.depth(
                    self._make(self.theta, self.phi, self.offset, 9.0).points
                )
                if self.fits(self._make(self.theta, self.phi, offset, 9.0), allowed):
                    self.offset = offset
                phi = self._furthest(
                    self.phi, free, lambda a: self._make(self.theta, a, self.offset, 9.0), allowed
                )
            self.phi = phi
            self._adopt(self._make(self.theta, phi, self.offset, 9.0))
            if self.phi >= self.phi_rest - 0.5 or (u >= 1.0 and phi < free - 2.0):
                self._finish()
        elif ph.name == "slide":
            u = min(1.0, ph.time / SLIDE_DURATION)
            target = self.slide_from + (self.slide_to - self.slide_from) * smoothstep(0.0, 1.0, u)
            # skid only as far as the body still fits, reclining further as it goes
            allowed = self.world.depth(self._make(self.theta, self.phi, self.offset, 9.0).points)
            if self.fits(self._make(self.theta, self.phi, target, 9.0), allowed):
                self.offset = target
            theta = self._furthest(
                self.theta,
                self.theta_rest,
                lambda a: self._make(a, self.phi, self.offset, 9.0),
                allowed,
            )
            self.theta = theta
            self._adopt(self._make(theta, self.phi, self.offset, 9.0))
            if u >= 1.0 or self.state.fall_angle >= 70.0:
                self.phase = _Phase("rest")
        return self.state

    @property
    def done(self) -> bool:
        return self.phase.name == "rest"


CIVILIAN_KIT = BodyKit(armor=False, visor=False, core_glow=False, holster=False, blaster=False)


def lying_on(
    x: float,
    top: float,
    z: float,
    head_yaw: float,
    kit: BodyKit = CIVILIAN_KIT,
    side: int = 0,
    scale: float = 1.0,
    crown: bool = False,
) -> BodyState:
    """A body lying on its back on something (a bed: its top at ``top``), its head towards
    ``head_yaw`` (NPC convention: 0 is +z) - asleep. (x, z) is where the middle of the body
    goes - or, with ``crown``, the top of its head (on the pillow, however tall it is).
    ``side`` (-1 / +1) turns it a little onto that side; ``scale`` is the person's size."""
    d = heading(head_yaw, 0.0)
    up = np.array((0.0, 1.0, 0.0))
    rotation = np.column_stack((np.cross(d, up), d, up))  # body x, y (to the head), z (front)
    if side:
        c, s = math.cos(math.radians(25.0 * side)), math.sin(math.radians(25.0 * side))
        roll = np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))  # about the spine
        rotation = rotation @ roll
    pose = rest_pose("supine", rotation)
    pose.has_prop = False
    pose.head_yaw, pose.head_roll = 15.0 * (side or 1), 6.0 * (side or 1)
    pose.arm_l = [8.0, 0.0, 14.0, -35.0]
    pose.arm_r = [8.0, 0.0, -12.0, -20.0]
    # the legs out straight along the bed (a fallen body's have its knees a little up)
    pitch, roll = _leg_aim(rotation, -d)
    pose.leg_l, pose.leg_r = [pitch + 2.0, 0.0, 4.0], [pitch + 1.0, 0.0, 6.0]
    pose.hip_roll_l, pose.hip_roll_r = roll + 4.0, roll - 4.0
    pose.foot_l = pose.foot_r = 0.0
    probe = BodyState((0.0, 0.0, 0.0), rotation, pose, scale)
    if crown:
        # the furthest point of it towards the head end goes to (x, z)
        pts = box_points(body_boxes(pose, kit, probe.matrix()))
        reach = float((pts[:, [0, 2]] @ d[[0, 2]]).max())
        hip = np.array((x, 0.0, z)) - d * reach
    else:
        # the middle of the body (~1.05 m above the soles standing) over (x, z)
        mid = np.array(probe.point(0.0, 1.05, 0.0))
        hip = np.array((x, 0.0, z)) - np.array((mid[0], 0.0, mid[2]))
    state = BodyState(tuple(hip), rotation, pose, scale)
    pts = box_points(body_boxes(pose, kit, state.matrix()))
    lift = top - float(pts[:, 1].min())  # the lowest point of it on the top of the bed
    return BodyState((float(hip[0]), float(lift), float(hip[2])), rotation, pose, scale)


def rest_of(collapse: Collapse, dt: float = 1 / 60, limit: float = 6.0) -> BodyState:
    """Runs a collapse to the end (for tests and tools)."""
    t = 0.0
    while not collapse.done and t < limit:
        collapse.step(dt)
        t += dt
    return collapse.state


__all__ = [
    "DIRECTIONS",
    "BodyState",
    "Collapse",
    "Surroundings",
    "axis_angle",
    "heading",
    "lying_on",
    "planted_pose",
    "rest_of",
    "rest_pose",
    "tilt",
]
