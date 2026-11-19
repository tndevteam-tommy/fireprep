"""Hands with fingers that hold what they hold (``animation.hands``, ``animation.rig``),
rounded bodies drawn in one batch each (``render.shapes``, ``render.batch``), and a can
drunk from with the elbow out (``animation.street.drink``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.test_visual_regressions import TEXTURES, MatrixRecorder

from cyberduke.ai.civilian import civilian_spec
from cyberduke.animation import HumanoidPose
from cyberduke.animation import street as clips
from cyberduke.animation.hands import (
    GRIP_FOR_ITEM,
    GRIPS,
    HAND_AT,
    grip_name,
    hand_frame,
    held_axis,
)
from cyberduke.animation.reach import hand
from cyberduke.animation.rig import BodyKit
from cyberduke.animation.skeleton import body_boxes
from cyberduke.render.batch import BatchContext
from cyberduke.render.humanoid import draw_humanoid
from cyberduke.render.items import ITEMS
from cyberduke.render.shapes import BUILDERS, mesh

KIT = BodyKit(armor=False, visor=False, core_glow=False, holster=False, blaster=False)


def _parts(pose: HumanoidPose, side: str = "r") -> dict[str, np.ndarray]:
    return {name: m for name, m in body_boxes(pose, KIT, parts=(f"arm_{side}",))}


def _centre(m: np.ndarray) -> np.ndarray:
    return m[:3, 3]


def test_every_thing_that_can_be_held_has_a_way_of_being_held():
    held = [name for name in ITEMS if name not in ("closed_sign", "stall_cover", "home_door")]
    for name in held:
        pose = clips.idle(0.0)
        pose.hand_r = name
        assert grip_name(pose, "r") in GRIPS, name
    assert set(GRIP_FOR_ITEM) <= set(ITEMS)


@pytest.mark.parametrize("item", ["soda", "phone", "cigarette", "knife", "takeaway", "bag"])
def test_the_fingers_close_round_what_is_held(item):
    """Holding something, the hand is in that thing's grip - not the empty hand's - and the
    thing is in the hand: between the palm and the fingertips, against the palm."""
    pose = clips.carrying(0.0, "held", moving=False)
    empty = _parts(pose)
    pose.hand_r = item
    full = _parts(pose)
    moved = [
        k for k in full if k.startswith(("finger", "thumb")) and not np.allclose(full[k], empty[k])
    ]
    assert len(moved) >= 6, "the fingers move to hold it"
    at, _ = hand(pose, "r")
    palm = _centre(full["palm_r"])
    tips = [_centre(full[f"finger{k}b_r"]) for k in range(4)]
    assert np.linalg.norm(palm - at) < 0.12, "in the hand, by the palm"
    assert min(np.linalg.norm(t - at) for t in tips) < 0.12, "the fingers round it"


def test_a_can_sits_in_curled_fingers_against_the_palm():
    pose = clips.carrying(0.0, "held", moving=False)
    pose.hand_r = "soda"
    parts = _parts(pose)
    at, _ = hand(pose, "r")
    axis = held_axis(pose, "r")
    assert axis is not None and axis[1] > 0.5, "upright, more or less"

    def off_axis(p):
        d = p - at
        return float(np.linalg.norm(d - axis * float(d @ axis)))

    # (the can is 0.033 round): the palm's inner face is against it, the fingers outside it
    frame = hand_frame(pose, "r")
    assert frame.grip is GRIPS["can"] and frame.wrist < HAND_AT - 0.05
    palm = parts["palm_r"]
    normal = palm[:3, 0] / np.linalg.norm(palm[:3, 0])  # (the right palm faces its +x)
    # the palm's inner face at the knuckles (its -y end): against the can
    surface = _centre(palm) - palm[:3, 1] / 2.0 + normal * np.linalg.norm(palm[:3, 0]) / 2.0
    assert 0.03 < off_axis(surface) < 0.045, "the palm against the can"
    for k in range(4):
        for seg in "ab":
            assert off_axis(_centre(parts[f"finger{k}{seg}_r"])) > 0.036, "round the can, not in it"


def test_gestures_have_their_own_hands():
    knock = _parts(clips.knock(0.3))
    idle = _parts(clips.idle(0.3))
    fist_curl = knock["finger1b_r"][:3, 1] @ knock["palm_r"][:3, 1]
    open_curl = idle["finger1b_r"][:3, 1] @ idle["palm_r"][:3, 1]
    assert fist_curl < open_curl, "a fist to knock with"
    point = _parts(clips.point_at(1.0, (0.0, 0.0, 1.0)))

    def along_palm(m):  # (each part is built down its -y: the way it points)
        palm = point["palm_r"][:3, 1]
        return float(m[:3, 1] @ palm / np.linalg.norm(m[:3, 1]) / np.linalg.norm(palm))

    assert along_palm(point["finger0b_r"]) > 0.95, "the index points on from the palm"
    assert along_palm(point["finger1b_r"]) < 0.0, "the others are curled right over"
    taps = {round(clips.typing(t).fingers_r[2], 3) for t in np.linspace(0.0, 2.0, 40)}
    assert len(taps) > 10, "typing fingers move"
    scroll = {round(clips.phone(t).fingers_r[0], 3) for t in np.linspace(0.0, 2.0, 40)}
    assert len(scroll) > 10, "the thumb scrolls the phone"


def test_drinking_the_elbow_goes_out_and_the_can_tips_up_at_the_lips():
    """The drinker's elbow was tucked in against the chest, the can upright at the mouth.
    Now the elbow comes up and out to the side, and the can turns with the hand: upright
    at the chest, tipped well back at the lips - rim to the lips."""
    rest = clips.drink(3.0)
    rest.hand_r = "soda"
    sip = clips.drink(1.3)
    sip.hand_r = "soda"
    up = held_axis(rest, "r")
    tipped = held_axis(sip, "r")
    assert math.degrees(math.acos(up[1])) < 20.0, "upright between sips"
    assert math.degrees(math.acos(tipped[1])) > 30.0, "tipped up to drink"
    assert tipped[2] < -0.3, "its top towards the face"
    elbow = _centre({n: m for n, m in body_boxes(sip, KIT, parts=("arm_r",))}["elbow_r"])
    # (it used to be at (-0.06, 1.40, 0.24): in front of the middle of the chest)
    assert elbow[0] < -0.25, "the elbow out to the side, not tucked in"
    assert elbow[1] > 1.55 and elbow[2] > 0.25, "up, and clear of the chest"
    at, _ = hand(sip, "r")
    rim = at + tipped * clips.CAN_HALF
    lips = np.array(clips._mouth(sip.head_pitch, sip.bob))
    assert np.linalg.norm(rim - lips) < 0.04, "the rim at the lips"


def test_the_rounded_shapes_are_closed_and_face_outwards():
    for name in BUILDERS:
        if name.endswith("_under"):  # (a hairstyle's underside: the inside of its shell)
            continue
        v, idx = mesh(name)
        tri = idx.reshape(-1, 3)
        p = v[:, :3].astype(float)
        a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
        assert volume > 0.0, f"{name}: inside out"
        assert np.allclose(np.linalg.norm(v[:, 3:6], axis=1), 1.0, atol=1e-3), f"{name}: normals"
        lo_v, _ = mesh(name, lo=True)
        assert len(lo_v) <= len(v), f"{name}: the far-off version is not finer"


def test_every_hairstyle_is_a_shell_with_its_outside_out():
    """A hairstyle's pieces - the outside of the hair, its (dyed) underside, the crest,
    the spikes, the tails and their ties - together enclose the hair: outsides facing
    out, undersides facing in (towards the head)."""
    from cyberduke.animation import hairdo

    def volume(name: str) -> float:
        p, _, t = hairdo.raw(name)
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    for style in hairdo.STYLES:
        for seed in range(4):
            _, pieces = hairdo.pieces(style, seed)
            for name, _ in pieces:
                assert (volume(name) < 0.0) == name.endswith("_under"), (style, name)
            if pieces:
                assert sum(volume(n) for n, _ in pieces) > 0.0, style


def test_a_person_is_one_batch_with_every_part_in_it():
    """Drawn through ``BatchContext`` a person is one ``draw_batch`` holding exactly the
    triangles of every part, each where the part-by-part drawing puts it."""
    spec = civilian_spec(7)
    pose = clips.idle(0.5)
    pose.hand_r = "soda"

    class Target:
        texture_tiles: dict = {}
        batches: list = []

        def draw_batch(self, vertices, indices, ranges):
            self.batches.append((vertices, indices, ranges))

    target = Target()
    batch = BatchContext(target)
    draw_humanoid(batch, spec, pose, TEXTURES)
    batch.flush()
    assert len(target.batches) == 1
    vertices, indices, ranges = target.batches[0]
    parts = MatrixRecorder()
    draw_humanoid(parts, spec, pose, TEXTURES)
    expected = 0
    for _, _, kind, _, _ in parts.boxes:
        expected += len(mesh(kind)[1]) if kind in BUILDERS else (36 if kind == "cube" else 0)
    cyl = sum(1 for b in parts.boxes if b[2] == "cylinder")
    assert cyl == 1, "the can"
    assert len(indices) >= expected and sum(r[2] for r in ranges) == len(indices)
    # the head's vertices sit in the head's box
    head = next(b for b in parts.boxes if b[2] == "head")
    inv = np.linalg.inv(head[4])
    local = (np.c_[vertices[:, :3], np.ones(len(vertices))] @ inv.T)[:, :3]
    inside = (np.abs(local) <= 0.5 + 1e-4).all(axis=1)
    assert inside.sum() >= len(mesh("head")[0]), "the head where it is drawn"


def test_a_blinded_guard_claps_the_hands_over_the_visor_not_through_the_head():
    """Blinded, both palms go over the visor - in front of the face wherever the head
    jerks, never inside it (the hands used to sink through the head)."""
    from cyberduke.animation import anatomy
    from cyberduke.animation import clips as guard_clips
    from cyberduke.animation.reach import _rx, _ry, _rz
    from cyberduke.animation.skeleton import GUARD_KIT

    for t in np.linspace(0.0, 3.0, 31):
        pose = guard_clips.blinded_animation(float(t))
        boxes = dict(body_boxes(pose, GUARD_KIT))
        turn = _ry(pose.head_yaw) @ _rx(pose.head_pitch) @ _rz(pose.head_roll)
        middle = np.array((0.0, anatomy.HEAD_Y + pose.bob, 0.0))
        for name, m in boxes.items():
            if not name.startswith(("palm", "finger", "thumb", "forearm")):
                continue
            x, y, z = turn.T @ (_centre(m) - middle)  # (in the head's own frame)
            assert z > anatomy.surface_z(x, y) - 0.002, f"{name} inside the head at t={t:.1f}"
        for side in "lr":
            x, y, z = turn.T @ (_centre(boxes[f"palm_{side}"]) - middle)
            assert z - anatomy.surface_z(x, y) < 0.1, "the palm over the face"
            assert abs(x) < 0.12 and abs(y - anatomy.EYE_LINE) < 0.1, "over the eyes"


@pytest.mark.parametrize(
    "clip",
    [
        lambda t: clips.beckon(t),
        lambda t: __import__("cyberduke.animation.clips", fromlist=["x"]).hawker_animation(t),
    ],
)
def test_a_wave_is_the_palm_to_the_front_beside_the_head(clip):
    """Calling people over, the hand goes up beside the head with its palm to the front
    (it used to be turned away, the back of the hand to them) and waves side to side."""
    seen = 0
    xs = []
    for t in np.linspace(0.0, 14.0, 141):
        pose = clip(float(t))
        if pose.grip_r != "wave":
            continue
        seen += 1
        palm = _parts(pose)["palm_r"]
        normal = palm[:3, 0] / np.linalg.norm(palm[:3, 0])  # (the right palm faces its +x)
        assert normal[2] > 0.6, f"the palm to the front at t={t:.1f}"
        at, _ = hand(pose, "r")
        if at[1] > 1.75:  # (up beside the head, not on its way up or down)
            assert -0.5 < at[0] < -0.2, "beside the head, to its side"
            xs.append(at[0])
    assert seen > 10 and len(xs) > 10, "up beside the head"
    assert max(xs) - min(xs) > 0.08, "waving side to side"


def _phone_frame(pose):
    """(middle, rotation: columns its width, its face's normal, its length) of the phone
    held up in both hands, as drawn (``render.items.held_transform``)."""
    from cyberduke.render.items import held_transform

    left, fore = hand(pose, "l")
    right, _ = hand(pose, "r")
    (x, y, z), yaw, pitch = held_transform(
        "phone", tuple((left + right) / 2.0), fore, both=True, tilt=pose.tilt_r
    )
    a, b = math.radians(yaw), math.radians(pitch)
    ry = np.array(((math.cos(a), 0, math.sin(a)), (0, 1, 0), (-math.sin(a), 0, math.cos(a))))
    rx = np.array(((1, 0, 0), (0, math.cos(b), -math.sin(b)), (0, math.sin(b), math.cos(b))))
    return np.array((x, y, z)), ry @ rx


PHONE_HALF_SIZE = np.array((0.035, 0.008, 0.07))  # (the phone's box and its screen on top)


def test_a_phone_is_held_up_in_both_hands_and_nothing_goes_into_it():
    """On the phone, both hands hold it - the phone resting up on the curled fingers, the
    hands either side of it, the thumbs up by its edges (the ``text`` grip), tipped up to
    the face. No part of either hand - palm, finger or thumb - goes into the phone at any
    moment of it (the fingers used to curl straight through it), and the two hands do not
    go into each other."""
    from cyberduke.animation.hands import TWO_HANDED

    radius = {"palm": 0.017, "thumb": 0.01, "finger": 0.009}
    for t in np.linspace(0.0, 4.0, 41):
        pose = clips.phone(float(t))
        assert pose.hand_l == pose.hand_r == "phone"
        assert grip_name(pose, "l") == grip_name(pose, "r") == TWO_HANDED["phone"] == "text"
        middle, turn = _phone_frame(pose)
        parts = {
            name: m
            for name, m in body_boxes(pose, KIT, parts=("arm_l", "arm_r"))
            if name.startswith(("palm", "finger", "thumb"))
        }
        for name, m in parts.items():
            shape = "palm" if name.startswith("palm") else "finger"
            v, _ = mesh(shape)
            pts = v[:, :3] @ m[:3, :3].T + m[:3, 3]
            local = (pts - middle) @ turn
            into = np.all(np.abs(local) < PHONE_HALF_SIZE - 0.0015, axis=1)
            assert not into.any(), f"{name} in the phone at t={t:.1f}"
        for a in parts:
            for b in parts:
                if a.endswith("_l") and b.endswith("_r"):
                    ka = next(k for k in radius if a.startswith(k))
                    kb = next(k for k in radius if b.startswith(k))
                    gap = float(np.linalg.norm(_centre(parts[a]) - _centre(parts[b])))
                    assert gap > radius[ka] + radius[kb] - 0.003, (a, b, t)
        # the phone above the fingers (on them), the palms either side of it
        for k in range(4):
            for side in "lr":
                tip = (_centre(parts[f"finger{k}b_{side}"]) - middle) @ turn
                assert tip[1] < 0.0, f"finger {k} under the phone"
        for side in "lr":
            palm = (_centre(parts[f"palm_{side}"]) - middle) @ turn
            assert 0.035 < abs(palm[0]) < 0.09, "a hand at either side of it"
    thumbs = {round(clips.phone(t).fingers_r[0], 3) for t in np.linspace(0.0, 2.0, 40)}
    assert len(thumbs) > 10, "the thumbs going"
    middle, turn = _phone_frame(clips.phone(0.5))
    assert turn[1, 1] < 0.9 and turn[2, 1] < -0.3, "tipped up, its face towards the face"
