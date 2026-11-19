"""The humanoid's body, part by part - written once for the renderer and the physics.

``build`` walks the joint hierarchy (a matrix stack of glRotatef-style rotations) and
emits every part as ``part(name, size, shape, material)``: a ``shape`` (``render.shapes``:
a rounded skull, a tapering limb, a finger bone, a boot) filling a box of ``size`` at the
current origin. The renderer (``render.humanoid``) turns that into a scaled shape with
the material's colour and texture; the collision body (``animation.skeleton``) into the
box itself. So what is drawn and what collides cannot drift apart - and
``tests/test_skeleton.py`` still checks them box for box.

Frames: the body's own - feet at the origin, facing +z, its left along +x. Joint places
are the ones the animation clips and the arm IK (``animation.reach``) are written for:
hips at 0.90, shoulders at (+-0.32, 1.58), an elbow 0.40 down the upper arm, the hand
point 0.42 down the forearm (``animation.hands`` fits the hand round it).

Groups (posed from the trunk on their own - the rag doll tests them one at a time):
"trunk" (head, neck, chest, pelvis, belt and the kit), "arm_l" / "arm_r" (shoulder, upper
arm, elbow, forearm, hand), "blaster" (the holstered sidearm, a guard's) and "leg_l" /
"leg_r" (thigh, knee, shin, boot).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cyberduke.animation import anatomy, hairdo, hands
from cyberduke.animation.look import PLAIN, Look
from cyberduke.animation.pose import HumanoidPose

HIP_HEIGHT = 0.90
HEAD_Y = anatomy.HEAD_Y
EYE_LINE = anatomy.EYE_LINE
ANKLE_DROP = 0.36
BOOT_DROP = 0.34
BOOT_HEIGHT = 0.24
HEAD_SIZE = anatomy.HEAD_SIZE
FACE_PROUD = 1.012  # the face stands this much proud of the head (about 2 mm)
# the visor band's box, in the head's unit cube: (middle, size) - ``render.shapes.visor_bounds``
VISOR_BOX = ((0.0, 0.1463, 0.1567), (1.0585, 0.241, 0.7619))
DECOR = "~"  # parts named so are drawn, but are not part of the collision body

Size = tuple[float, float, float]


class Builder(Protocol):
    def push(self) -> None: ...

    def pop(self) -> None: ...

    def translate(self, x: float, y: float, z: float) -> None: ...

    def rotate(self, angle: float, x: float, y: float, z: float) -> None: ...

    def part(self, name: str, size: Size, shape: str, material: str) -> None: ...


@dataclass(frozen=True)
class BodyKit:
    """Which optional parts the body has (``HumanoidSpec`` flags), and whether its hands
    have fingers (``False``: mittens - the renderer's far-off people)."""

    armor: bool = True
    visor: bool = True
    core_glow: bool = True
    holster: bool = True
    blaster: bool = True
    fingers: bool = True
    look: Look = PLAIN  # hair, eyes, chrome, trim (``animation.look``): the decoration


GUARD_KIT = BodyKit()

GROUPS = ("trunk", "arm_l", "arm_r", "blaster", "leg_l", "leg_r")


# ----------------------------------------------------------------------------- trunk
def trunk(b: Builder, pose: HumanoidPose, kit: BodyKit, shoulders: bool = False) -> None:
    bob = pose.bob
    look = kit.look
    b.push()
    b.translate(0.0, HEAD_Y + bob, 0.0)
    b.rotate(pose.head_yaw, 0, 1, 0)
    b.rotate(pose.head_pitch, 1, 0, 0)
    b.rotate(pose.head_roll, 0, 0, 1)
    b.part("head", HEAD_SIZE, "head", "skin")
    # the face: the front of the skull, a hair's breadth proud of it (its texture is the
    # person's expression); the nose; the ears; the hair, a cap over the skull (or stubble,
    # or a bare scalp)
    b.part("face", tuple(s * FACE_PROUD for s in HEAD_SIZE), "face", "face")  # type: ignore[arg-type]
    b.push()
    b.translate(0.0, anatomy.NOSE_Y, 0.175)
    b.part("nose", (0.055, 0.085, 0.065), "nose", "skin")
    b.pop()
    for side in (0.168, -0.168):
        b.push()
        b.translate(side, anatomy.EAR_Y, -0.005)
        b.part("ear", (0.045, 0.105, 0.085), "ear", "skin")
        b.pop()
    # the cap of the hairstyle (``animation.hairdo``): a crop, stubble, a bare scalp -
    # what the collision body knows as the hair (the rest of the style is decoration)
    (cap, cap_material), _ = hairdo.pieces(look.hair, look.seed)
    middle, size = hairdo.box(hairdo.CAP)
    b.push()
    b.translate(*middle)
    b.part("hair", size, cap, cap_material)
    b.pop()
    if kit.visor:  # (a band round the front of the head, across the eyes)
        _band(b, "visor", "visor")
    head_decor(b, look, kit)
    b.pop()
    b.push()
    b.translate(0.0, 1.76 + bob, 0.0)
    b.part("neck", (0.165, 0.17, 0.155), "neck", "skin")
    b.pop()
    b.push()
    b.translate(0.0, 1.35 + bob, 0.0)
    b.part("torso", (0.50, 0.68, 0.30), "torso", "torso")
    b.pop()
    b.push()  # the slope of the shoulders up to the neck (the shirt's collar)
    b.translate(0.0, 1.685 + bob, -0.005)
    b.part("collar", (0.36, 0.12, 0.24), "sphere", "torso")
    b.pop()
    if kit.armor:
        b.push()
        b.translate(0.0, 1.42 + bob, 0.13)
        b.part("armor", (0.38, 0.42, 0.08), "plate", "plate")
        b.pop()
    if kit.core_glow:
        b.push()
        b.translate(0.0, 1.45 + bob, 0.17)
        b.part("core", (0.12, 0.12, 0.02), "plate", "core")
        b.pop()
    b.push()
    b.translate(0.0, 0.96 + bob, 0.0)
    b.part("belt", (0.48, 0.12, 0.28), "belt", "belt")
    b.pop()
    b.push()  # the seat of the trousers: fills in round the tops of the thighs
    b.translate(0.0, 0.86 + bob, 0.0)
    b.part("pelvis", (0.44, 0.2, 0.26), "pelvis", "pants")
    b.pop()
    if kit.holster:
        b.push()
        b.translate(-0.28, 0.96 + bob, 0.0)
        b.part("holster", (0.14, 0.22, 0.18), "rbox", "holster")
        b.pop()
    torso_decor(b, look, bob)
    if shoulders:
        for tag, sx in (("l", 0.32 - pose.shrug_l), ("r", -0.32 + pose.shrug_r)):
            b.push()
            b.translate(sx - (SHOULDER_IN if tag == "l" else -SHOULDER_IN), 1.58 + bob, 0.0)
            b.part(f"shoulder_{tag}", SHOULDER, "sphere", f"shoulder_{tag}")
            b.pop()


# ----------------------------------------------------------------------------- decoration
# (hair, cybernetic eyes, chrome, neon: drawn, not in the collision body - a mohawk does
# not prop up a falling guard)


def _band(b: Builder, name: str, material: str) -> None:
    """The visor's wraparound band across the eyes (a guard's visor, cyber shades)."""
    (mx, my, mz), (sx, sy, sz) = VISOR_BOX
    hx, hy, hz = HEAD_SIZE
    b.push()
    b.translate(mx * hx, my * hy, mz * hz)
    b.part(name, (sx * hx, sy * hy, sz * hz), "visor", material)
    b.pop()


def _on_face(x: float, y: float, out: float = 0.0) -> tuple[float, float, float]:
    """A point on the face's skin (``animation.anatomy``), ``out`` metres in front of it."""
    return x, y, anatomy.surface_z(x, y) * FACE_PROUD + out


def _at(b: Builder, where, name: str, size, shape: str, material: str) -> None:
    b.push()
    b.translate(*where)
    b.part(name, size, shape, material)
    b.pop()


def head_decor(b: Builder, look: Look, kit: BodyKit) -> None:
    """The hairstyle's crest, spikes, tails or fringe; cybernetic eyes; chrome; extras."""
    # the hairstyle's pieces over its cap (``animation.hairdo``: a crest, spikes, a
    # swept top, a bob, long hair, a top knot, twin tails), each in its own box
    _, pieces = hairdo.pieces(look.hair, look.seed)
    for name, material in pieces:
        middle, size = hairdo.box(name)
        _at(b, middle, f"{DECOR}{name}", size, name, material)
    # the eyes (behind a visor, nobody sees them)
    if not kit.visor and "shades" not in look.extras:
        for side in (1.0, -1.0):
            x = side * anatomy.EYE_X
            if look.eyes == "optic" and side > 0:  # a big optic in place of the left eye
                _at(b, _on_face(x, EYE_LINE, 0.012), f"{DECOR}optic", (0.046, 0.046, 0.03),
                    "sphere", "chrome")  # fmt: skip
                _at(b, _on_face(x, EYE_LINE, 0.028), f"{DECOR}eye", (0.03, 0.03, 0.006),
                    "sphere", "eye")  # fmt: skip
            elif look.eyes == "chrome":
                _at(b, _on_face(x, EYE_LINE, 0.002), f"{DECOR}eye_chrome", (0.026, 0.022, 0.005),
                    "sphere", "chrome")  # fmt: skip
                _at(b, _on_face(x, EYE_LINE, 0.004), f"{DECOR}eye", (0.01, 0.01, 0.004),
                    "sphere", "eye")  # fmt: skip
            elif look.eyes in ("glow", "optic"):
                _at(b, _on_face(x, EYE_LINE, 0.0015), f"{DECOR}eye", (0.02, 0.02, 0.004),
                    "sphere", "eye")  # fmt: skip
    if "shades" in look.extras:
        _band(b, f"{DECOR}shades", "shades")
    # chrome
    if "jaw" in look.implants:
        _at(b, (-0.15, -0.14, 0.035), f"{DECOR}jaw", (0.02, 0.075, 0.13), "rbox", "chrome")
        _at(b, (-0.161, -0.14, 0.035), f"{DECOR}jaw_seam", (0.004, 0.05, 0.09), "rbox", "implant")
    if "temples" in look.implants:
        for side in (-1.0, 1.0):
            _at(b, (side * 0.152, 0.085, 0.07), f"{DECOR}temple", (0.014, 0.03, 0.05), "rbox",
                "implant")  # fmt: skip
    if "ear" in look.implants:
        _at(b, (-0.171, anatomy.EAR_Y, -0.005), f"{DECOR}ear_tech", (0.056, 0.118, 0.096), "rbox",
            "chrome")  # fmt: skip
        _at(b, (-0.2, anatomy.EAR_Y + 0.025, 0.0), f"{DECOR}ear_led", (0.01, 0.01, 0.01), "sphere",
            "implant")  # fmt: skip
    # extras
    if "respirator" in look.extras:
        _at(b, _on_face(0.0, anatomy.MOUTH_LINE + 0.025, 0.012), f"{DECOR}mask",
            (0.15, 0.09, 0.06), "rbox", "mask")  # fmt: skip
        for side in (-1.0, 1.0):
            _at(b, _on_face(side * 0.06, anatomy.MOUTH_LINE + 0.01, 0.018), f"{DECOR}filter",
                (0.05, 0.05, 0.045), "sphere", "chrome")  # fmt: skip
        _at(b, _on_face(0.0, anatomy.MOUTH_LINE + 0.004, 0.045), f"{DECOR}mask_glow",
            (0.05, 0.012, 0.008), "rbox", "neon")  # fmt: skip
    if "goggles" in look.extras:
        for side in (-1.0, 1.0):
            x = side * 0.047
            y = anatomy.HAIRLINE_FRONT + 0.005
            _at(b, _on_face(x, y, 0.01), f"{DECOR}goggle_rim", (0.058, 0.05, 0.022), "sphere",
                "chrome")  # fmt: skip
            _at(b, _on_face(x, y, 0.022), f"{DECOR}goggle", (0.046, 0.04, 0.012), "sphere",
                "lens")  # fmt: skip
    if "earrings" in look.extras:
        for side in (-1.0, 1.0):
            for k in range(2):
                _at(b, (side * 0.176, anatomy.EAR_Y - 0.045 - 0.022 * k, 0.005), f"{DECOR}earring",
                    (0.016, 0.016, 0.016), "sphere", "gold" if k == 0 else "neon")  # fmt: skip


def torso_decor(b: Builder, look: Look, bob: float) -> None:
    """Neon trim round the chest or the hem, stripes, an open jacket's shirt, a high
    collar, rank stripes on a guard's chest plate, a port in the neck."""
    if look.top == "jacket":
        _at(b, (0.0, 1.36 + bob, 0.141), f"{DECOR}shirt_front", (0.15, 0.5, 0.02), "rbox", "shirt")
    if look.trim == "chest":
        _at(b, (0.0, 1.49 + bob, -0.002), f"{DECOR}trim", (0.52, 0.035, 0.315), "torso", "neon")
    elif look.trim == "hem":
        _at(b, (0.0, 1.04 + bob, 0.0), f"{DECOR}trim", (0.45, 0.03, 0.28), "torso", "neon")
    elif look.trim == "stripes":
        for side in (-1.0, 1.0):
            _at(b, (side * 0.105, 1.36 + bob, 0.142), f"{DECOR}stripe", (0.03, 0.5, 0.02), "rbox",
                "neon")  # fmt: skip
    elif look.trim == "rank":  # chevrons on a guard's chest plate
        for k in range(3 if look.implants else 2):
            _at(b, (0.1, 1.57 - k * 0.03 + bob, 0.172), f"{DECOR}rank", (0.08, 0.013, 0.01),
                "rbox", "neon")  # fmt: skip
    if "collar" in look.extras:
        b.push()
        b.translate(0.0, 1.76 + bob, -0.07)
        b.rotate(-18.0, 1, 0, 0)
        b.part(f"{DECOR}collar", (0.3, 0.11, 0.045), "rbox", "jacket")
        b.pop()
    if "neck" in look.implants:
        _at(b, (0.07, 1.755 + bob, 0.035), f"{DECOR}neck_port", (0.016, 0.06, 0.035), "rbox",
            "implant")  # fmt: skip


# ----------------------------------------------------------------------------- arms
UPPER_ARM = {"l": (0.14, 0.4, 0.16), "r": (0.16, 0.4, 0.18)}
SHOULDER = (0.19, 0.15, 0.17)  # the round of the shoulder over the top of the arm ...
SHOULDER_IN = 0.025  # ... its middle this much in from the joint: it runs into the chest
FOREARM = {"l": (0.12, 0.14), "r": (0.14, 0.16)}  # (width, depth); the length is the wrist's


def arm(b: Builder, pose: HumanoidPose, tag: str, kit: BodyKit, with_shoulder: bool = True) -> None:
    sx = 0.32 - pose.shrug_l if tag == "l" else -0.32 + pose.shrug_r
    angles = pose.arm_l if tag == "l" else pose.arm_r
    b.push()
    b.translate(sx, 1.58 + pose.bob, 0.0)
    if with_shoulder:
        b.push()
        b.translate(-SHOULDER_IN if tag == "l" else SHOULDER_IN, 0.0, 0.0)
        b.part(f"shoulder_{tag}", SHOULDER, "sphere", f"shoulder_{tag}")
        b.pop()
    look = kit.look
    if "pads" in look.extras:  # a jacket's shoulder pads, rimmed in neon
        out = 0.03 if tag == "l" else -0.03
        _at(b, (out, 0.07, 0.0), f"{DECOR}pad_{tag}", (0.2, 0.055, 0.2), "rbox", "jacket")
        _at(b, (out, 0.04, 0.0), f"{DECOR}pad_rim_{tag}", (0.21, 0.012, 0.21), "rbox", "neon")
    b.rotate(angles[0], 1, 0, 0)
    b.rotate(angles[1], 0, 1, 0)
    b.rotate(angles[2], 0, 0, 1)
    b.push()
    b.translate(0, -0.2, 0)
    b.part(f"upper_arm_{tag}", UPPER_ARM[tag], "upper_arm", f"upper_{tag}")
    b.pop()
    b.translate(0, -0.4, 0)
    w, d = FOREARM[tag]
    b.part(f"elbow_{tag}", (w * 0.98, w * 0.98, d * 0.92), "sphere", f"fore_{tag}")
    b.rotate(angles[3], 1, 0, 0)
    frame = hands.hand_frame(pose, tag)
    length = frame.wrist + 0.012  # (into the wrist a little: no gap at a bent wrist)
    b.push()
    b.translate(0, -length / 2.0, 0)
    b.part(f"forearm_{tag}", (w, length, d), "forearm", f"fore_{tag}")
    if f"arm_{tag}" in look.implants:  # a cyberarm: a glowing seam down the forearm
        _at(b, (0.0, 0.0, d / 2.0 - 0.004), f"{DECOR}arm_seam_{tag}", (0.012, length * 0.7, 0.012),
            "rbox", "implant")  # fmt: skip
    b.pop()
    b.translate(0, -frame.wrist, 0)
    angle, axis = hands.axis_angle(frame.rotation)
    if angle:
        b.rotate(angle, *axis)
    hand(b, tag, frame, kit.fingers)
    b.pop()


def hand(b: Builder, tag: str, frame: hands.HandFrame, fingers: bool = True) -> None:
    """A hand in its own frame at the wrist (see ``animation.hands``): the palm, then each
    finger from its knuckle - fanned out, curled at two joints - and the thumb."""
    n = hands.PALM_SIGN[tag]
    grip = frame.grip
    curls = frame.curls
    mat = f"hand_{tag}"
    pl = hands.PALM_LENGTH
    b.push()
    b.translate(0.0, -pl / 2.0, 0.0)
    b.part(f"palm_{tag}", (hands.PALM_THICK, pl, hands.PALM_WIDTH), "palm", mat)
    b.pop()
    if not fingers:
        # a mitten: the four fingers as one, curled as the middle finger is
        c = curls[2]
        b.push()
        b.translate(n * 0.004, -pl + 0.004, 0.0)
        b.rotate(n * hands.CURL_KNUCKLE * c, 0, 0, 1)
        b.push()
        b.translate(0.0, -0.025, 0.0)
        b.part(f"fingers_{tag}", (0.022, 0.06, hands.PALM_WIDTH * 0.95), "rbox", mat)
        b.pop()
        b.translate(0.0, -0.05, 0.0)
        b.rotate(n * hands.CURL_MIDDLE * c, 0, 0, 1)
        b.push()
        b.translate(0.0, -0.02, 0.0)
        b.part(f"fingertips_{tag}", (0.02, 0.045, hands.PALM_WIDTH * 0.9), "rbox", mat)
        b.pop()
        b.pop()
    else:
        for k, across in enumerate(hands.KNUCKLES):
            c = curls[k + 1]
            l1, l2 = hands.FINGER_BONES[k]
            d = hands.FINGER_THICK[k]
            b.push()
            b.translate(n * 0.004, -pl + 0.004, across)
            b.rotate(hands.FINGER_FAN[k] * grip.spread, 1, 0, 0)
            b.rotate(n * hands.CURL_KNUCKLE * c * grip.bend[0], 0, 0, 1)
            top, bottom = 0.5 * d, -(l1 + 0.35 * d)
            b.push()
            b.translate(0.0, (top + bottom) / 2.0, 0.0)
            b.part(f"finger{k}a_{tag}", (d, top - bottom, d), "finger", mat)
            b.pop()
            b.translate(0.0, -l1, 0.0)
            b.rotate(n * hands.CURL_MIDDLE * c * grip.bend[1], 0, 0, 1)
            top, bottom = 0.45 * d, -l2
            b.push()
            b.translate(0.0, (top + bottom) / 2.0, 0.0)
            b.part(f"finger{k}b_{tag}", (d * 0.92, top - bottom, d * 0.92), "finger", mat)
            b.pop()
            b.pop()
    # the thumb: off the side of the palm near the wrist, standing out from it, turned in
    # to face the fingers, curled at its two joints
    c = curls[0]
    tx, ty, tz = hands.THUMB_ROOT
    b.push()
    b.translate(n * tx, ty, tz)
    b.rotate(-grip.thumb_out, 1, 0, 0)
    b.rotate(n * grip.thumb_turn, 0, 1, 0)
    b.rotate(n * hands.THUMB_CURL[0] * c, 0, 0, 1)
    l1, l2 = hands.THUMB_BONES
    d1, d2 = hands.THUMB_THICK
    top, bottom = 0.45 * d1, -(l1 + 0.3 * d1)
    b.push()
    b.translate(0.0, (top + bottom) / 2.0, 0.0)
    b.part(f"thumb_a_{tag}", (d1, top - bottom, d1), "finger", mat)
    b.pop()
    if fingers:
        b.translate(0.0, -l1, 0.0)
        b.rotate(n * hands.THUMB_CURL[1] * c, 0, 0, 1)
        top, bottom = 0.45 * d2, -l2
        b.push()
        b.translate(0.0, (top + bottom) / 2.0, 0.0)
        b.part(f"thumb_b_{tag}", (d2, top - bottom, d2), "finger", mat)
        b.pop()
    b.pop()


# ----------------------------------------------------------------------------- legs
def leg(b: Builder, pose: HumanoidPose, tag: str, kit: BodyKit) -> None:
    side, foot, angles, roll = (
        (0.14, pose.foot_l, pose.leg_l, pose.hip_roll_l)
        if tag == "l"
        else (-0.14, pose.foot_r, pose.leg_r, pose.hip_roll_r)
    )
    b.push()
    b.translate(side, 0.90 + pose.bob, 0.0)
    if roll:
        b.rotate(roll, 0, 0, 1)
    b.rotate(angles[0], 1, 0, 0)
    b.rotate(angles[1], 0, 1, 0)
    b.push()
    b.translate(0, -0.22, 0)
    b.part(f"thigh_{tag}", (0.18, 0.44, 0.20), "thigh", "pants")
    if kit.look.stripe:
        out = 0.086 if tag == "l" else -0.086
        _at(b, (out, 0.0, 0.0), f"{DECOR}stripe_{tag}", (0.014, 0.38, 0.03), "rbox", "neon")
    b.pop()
    b.push()
    b.translate(0, -0.44, 0)
    if kit.armor:
        b.part(f"knee_{tag}", (0.16, 0.12, 0.18), "rbox", "kneepad")
    else:
        b.part(f"knee_{tag}", (0.16, 0.16, 0.17), "sphere", "pants")
    b.rotate(max(0.0, angles[2]), 1, 0, 0)
    b.push()
    b.translate(0, -0.20, 0)
    b.part(f"shin_{tag}", (0.16, 0.40, 0.18), "shin", "pants")
    b.pop()
    b.push()
    b.translate(0, -ANKLE_DROP, 0.0)
    b.rotate(foot, 1, 0, 0)
    b.translate(0, ANKLE_DROP - BOOT_DROP, 0.04)
    b.part(f"boot_{tag}", (0.22, BOOT_HEIGHT, 0.32), "boot", "boot")
    if kit.look.soles:  # glowing soles
        _at(b, (0.0, -BOOT_HEIGHT / 2.0 + 0.013, 0.0), f"{DECOR}sole_{tag}", (0.226, 0.022, 0.326),
            "rbox", "neon")  # fmt: skip
    b.pop()
    b.pop()
    b.pop()


# ----------------------------------------------------------------------------- blaster
def blaster(b: Builder, pose: HumanoidPose) -> None:
    b.push()
    b.translate(pose.prop_pos[0], pose.prop_pos[1] + pose.bob, pose.prop_pos[2])
    b.rotate(pose.prop_rot[0], 1, 0, 0)
    b.rotate(pose.prop_rot[1], 0, 1, 0)
    b.rotate(pose.prop_rot[2], 0, 0, 1)
    for name, (tx, ty, tz), tilt, size in (
        ("grip", (0.0, -0.06, 0.0), 12.0, (0.05, 0.14, 0.07)),
        ("receiver", (0.0, 0.02, 0.02), 0.0, (0.06, 0.08, 0.20)),
        ("barrel", (0.0, 0.03, 0.16), 0.0, (0.04, 0.04, 0.14)),
        ("emitter", (0.0, 0.03, 0.235), 0.0, (0.03, 0.03, 0.02)),
    ):
        b.push()
        b.translate(tx, ty, tz)
        b.rotate(tilt, 1, 0, 0)
        b.part(f"blaster_{name}", size, "cube", "blaster")
        b.pop()
    b.pop()


def lean(b: Builder, pose: HumanoidPose) -> None:
    """The upper body bends forward at the hips (everything but the legs)."""
    if pose.lean:
        b.translate(0.0, HIP_HEIGHT + pose.bob, 0.0)
        b.rotate(pose.lean, 1, 0, 0)
        b.translate(0.0, -(HIP_HEIGHT + pose.bob), 0.0)


__all__ = [
    "GROUPS",
    "GUARD_KIT",
    "BodyKit",
    "Builder",
    "arm",
    "blaster",
    "build_group",
    "hand",
    "leg",
    "lean",
    "trunk",
]


def build_group(b: Builder, pose: HumanoidPose, group: str, kit: BodyKit, whole: bool) -> None:
    """One group of the body, from the body's frame (the lean included)."""
    b.push()
    if not group.startswith("leg"):
        lean(b, pose)
    if group == "trunk":
        trunk(b, pose, kit, shoulders=not whole)
    elif group.startswith("arm"):
        arm(b, pose, group[-1], kit, with_shoulder=whole)
    elif group == "blaster":
        if pose.has_prop and kit.blaster:
            blaster(b, pose)
    else:
        leg(b, pose, group[-1], kit)
    b.pop()
