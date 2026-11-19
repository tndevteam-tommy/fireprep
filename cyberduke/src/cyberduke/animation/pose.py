"""Skeletal pose data shared by every humanoid character."""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class HumanoidPose:
    """Holds skeletal rotation angles and translations for any humanoid character."""

    bob: float = 0.0

    # Head rotations (degrees)
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0

    # Left Arm: [pitch, yaw, roll, elbow]
    arm_l: list = field(default_factory=lambda: [0.0, 0.0, 0.0, -15.0])
    # Right Arm: [pitch, yaw, roll, elbow]
    arm_r: list = field(default_factory=lambda: [0.0, 0.0, 0.0, -15.0])

    # Left Leg: [hip_pitch, hip_yaw, knee_bend]
    leg_l: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Right Leg: [hip_pitch, hip_yaw, knee_bend]
    leg_r: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Ankles: foot pitch relative to the shin (degrees; positive tips the toes down)
    foot_l: float = 0.0
    foot_r: float = 0.0
    # Hip roll (degrees about the body's forward axis, applied before the hip pitch):
    # positive swings the leg out to the body's left (+x). A body lying on its side
    # brings its top leg down onto the floor in front of the other one with it.
    hip_roll_l: float = 0.0
    hip_roll_r: float = 0.0
    # Shoulders drawn in towards the neck (metres): a body lying on its side has its lower
    # shoulder pushed in under it, not sticking out like a shelf under the chest.
    shrug_l: float = 0.0
    shrug_r: float = 0.0

    # The upper body bent forward at the hips (degrees; the head, arms and whatever they
    # hold go with it): stooping to put a parcel down on a step or to pick one up.
    lean: float = 0.0

    # Things held in the hands (names of registered item meshes, drawn in the hand) -
    # a takeaway box, a knife, a datachip... None for an empty hand.
    hand_l: str | None = None
    hand_r: str | None = None
    # Where a held stick-like thing points (a body-frame point; None: the way the item
    # usually sits - see ``render.items``): the tips of the chopsticks go into the box and
    # up to the mouth.
    aim_l: tuple[float, float, float] | None = None
    aim_r: tuple[float, float, float] | None = None
    # A held thing turned in the fingers about the forearm (degrees): a cigarette rolled
    # and tapped while the ash is flicked off.
    tilt_l: float = 0.0
    tilt_r: float = 0.0
    # A cigarette: how hot its tip is (0: smouldering, 1: drawn on - the cherry flares on
    # a drag), whether it is being drawn on (no smoke curls off the tip while it is), and
    # smoke being blown out of the mouth (0..1: how hard, this moment). ``render.smoke``
    # makes the smoke of it; ``render.items`` lights the cherry.
    ember: float = 0.0
    inhaling: bool = False
    exhale: float = 0.0
    # How firmly the head is held where the clip puts it (0: free to turn and look at
    # whoever is about; 1: held - a cigarette at the lips: the head comes back round to it
    # for the drag, however interesting the passer-by)
    head_held: float = 0.0

    # The hands (``animation.hands``): a grip of the pose's own instead of the one that
    # goes with what is held (or with an empty hand) - "fist", "point", "open", "flat",
    # "type", "beckon"... - and the fingers' curl, moment to moment, for fingers that move
    # (thumb, index, middle, ring, little finger: 0 straight .. 1 curled into a fist).
    grip_l: str | None = None
    grip_r: str | None = None
    fingers_l: tuple[float, float, float, float, float] | None = None
    fingers_r: tuple[float, float, float, float, float] | None = None

    # The face of the moment (``assets.faces.EXPRESSIONS``: "talk", "eat", "pursed",
    # "mumble", "smile", "frown", "grimace"...); None: the person's resting face
    expression: str | None = None

    # Attached Prop (weapon, datapad, tool)
    has_prop: bool = False
    prop_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    prop_rot: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


def mix_poses(a: HumanoidPose, b: HumanoidPose, w: float) -> HumanoidPose:
    """``a`` blended towards ``b`` by ``w`` (0: ``a``, 1: ``b``): every angle and offset in
    between; what is held, and whether the prop is out, from whichever is nearer."""
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b
    out = HumanoidPose()
    for f in fields(HumanoidPose):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, float | int) and not isinstance(va, bool):
            setattr(out, f.name, va + (vb - va) * w)
        elif isinstance(va, list):
            setattr(out, f.name, [x + (y - x) * w for x, y in zip(va, vb, strict=True)])
        elif f.name.startswith("fingers") and isinstance(va, tuple) and isinstance(vb, tuple):
            setattr(out, f.name, tuple(x + (y - x) * w for x, y in zip(va, vb, strict=True)))
        else:
            setattr(out, f.name, vb if w >= 0.5 else va)
    return out
