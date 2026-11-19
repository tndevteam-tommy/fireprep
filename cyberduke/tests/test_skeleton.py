"""The collision body (animation.skeleton) is the body the renderer draws, box for box."""

import random

import numpy as np
import pytest
from tests.test_visual_regressions import TEXTURES, MatrixRecorder

from cyberduke.ai.guard import GUARD_SPEC
from cyberduke.animation import HumanoidPose
from cyberduke.animation.skeleton import body_boxes, box_points
from cyberduke.render.humanoid import draw_humanoid


def _random_pose(seed: int) -> HumanoidPose:
    rng = random.Random(seed)

    def a(span=90.0):
        return rng.uniform(-span, span)

    return HumanoidPose(
        bob=rng.uniform(-0.05, 0.05),
        head_pitch=a(40), head_yaw=a(60), head_roll=a(20),
        arm_l=[a(), a(), a(), a()], arm_r=[a(), a(), a(), a()],
        leg_l=[a(), a(20), abs(a())], leg_r=[a(), a(20), abs(a())],
        foot_l=a(40), foot_r=a(40), hip_roll_l=a(30), hip_roll_r=a(30),
        has_prop=True, prop_pos=[-0.28, 0.98, 0.0], prop_rot=[a(), a(), a()],
        lean=0.0 if seed % 2 else a(60),  # half of them bent forward at the hips
    )  # fmt: skip


@pytest.mark.parametrize("seed", range(6))
def test_the_collision_body_is_the_drawn_body(seed):
    pose = _random_pose(seed)
    ctx = MatrixRecorder()
    draw_humanoid(ctx, GUARD_SPEC, pose, TEXTURES)
    drawn = [m for *_, m in ctx.boxes]
    model = [m for _, m in body_boxes(pose)]
    assert len(drawn) == len(model)
    for a, b in zip(drawn, model, strict=True):
        assert np.allclose(a, b, atol=1e-9)


def test_box_points_cover_every_face():
    boxes = body_boxes(HumanoidPose())
    pts = box_points(boxes, spacing=0.07)
    _, torso = next(b for b in boxes if b[0] == "torso")
    inv = np.linalg.inv(torso)
    local = (pts - torso[:3, 3]) @ inv[:3, :3].T
    on_torso = local[(np.abs(local) <= 0.5 + 1e-9).all(axis=1)]
    for axis in range(3):
        for sign in (-1, 1):
            face = on_torso[np.isclose(on_torso[:, axis], 0.5 * sign)]
            assert len(face) >= 9, "a grid over each face, not only its corners"


def test_the_rounded_parts_fill_their_boxes():
    """Each rounded shape fills the unit cube it is drawn in (the collision box bounds what
    is drawn, and not much more) - the visor band its own box, which the rig places."""
    from cyberduke.animation.rig import VISOR_BOX
    from cyberduke.render.shapes import BUILDERS, mesh, visor_bounds

    for name in BUILDERS:
        v, tris = mesh(name)
        lo, hi = v[:, :3].min(axis=0), v[:, :3].max(axis=0)
        assert (lo >= -0.5 - 1e-6).all() and (hi <= 0.5 + 1e-6).all(), name
        # (the face is a skin over the front of the head's box; stubble a skin over the
        # skull in the crop's box - under the crop's volume; a hairstyle's piece fills its
        # own box, ``animation.hairdo.box``)
        if name == "stubble_cap":
            assert (hi - lo).min() > 0.75 and hi[1] > 0.4, "stubble over the crown"
        elif name != "face":
            assert (hi - lo).min() > 0.75, f"{name} fills its box"
        assert len(tris) % 3 == 0 and tris.max() < len(v)
    middle, size = visor_bounds()
    assert np.allclose(middle, VISOR_BOX[0], atol=1e-3) and np.allclose(
        size, VISOR_BOX[1], atol=1e-3
    )


class _Names:
    """A builder that only notes the names of the parts."""

    def __init__(self):
        self.names = []

    def push(self):
        pass

    def pop(self):
        pass

    def translate(self, x, y, z):
        pass

    def rotate(self, a, x, y, z):
        pass

    def part(self, name, size, shape, material):
        self.names.append(name)


@pytest.mark.parametrize("seed", range(8))
def test_hair_chrome_and_neon_are_drawn_but_not_collided_with(seed):
    """The look (mohawks, spikes, twin tails, optics, jaw plates, neon trim) is drawn with
    the body, but the collision body is the body's own boxes: a spike of hair does not
    stop a door or prop a fallen body up."""
    from cyberduke.ai.civilian import civilian_spec
    from cyberduke.animation import rig
    from cyberduke.animation.rig import DECOR, GROUPS, BodyKit

    spec = civilian_spec(seed)
    kit = BodyKit(
        armor=False, visor=False, core_glow=False, holster=False, blaster=False, look=spec.look
    )
    drawn = _Names()
    for group in GROUPS:
        rig.build_group(drawn, HumanoidPose(), group, kit, True)
    decor = [n for n in drawn.names if n.startswith(DECOR)]
    assert decor, "every street look has something to it"
    model = [n for n, _ in body_boxes(HumanoidPose(), kit)]
    assert model == [n for n in drawn.names if not n.startswith(DECOR)]
