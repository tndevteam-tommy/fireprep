"""Hierarchical humanoid renderer driven by (HumanoidSpec, HumanoidPose)."""

from __future__ import annotations

import math
from collections.abc import Callable

from cyberduke.ai.npc import BaseNPC, HumanoidSpec
from cyberduke.animation import HumanoidPose, hands, rig
from cyberduke.animation.ragdoll import axis_angle
from cyberduke.animation.reach import hand as hand_position
from cyberduke.animation.rig import BodyKit
from cyberduke.assets.faces import flesh_name
from cyberduke.render.batch import BatchContext
from cyberduke.render.context import DrawContext
from cyberduke.render.effects import draw_sparks
from cyberduke.render.items import draw_item, held_transform
from cyberduke.render.textures import TextureMap

HIP_HEIGHT = rig.HIP_HEIGHT  # hip joints above the feet
BOOT_HEIGHT = rig.BOOT_HEIGHT
BOOT_DROP = rig.BOOT_DROP  # knee to boot centre: a straight leg puts the sole exactly on the floor
ANKLE_DROP = rig.ANKLE_DROP  # knee to ankle joint (the boot pivots there)
EYE_LINE = rig.EYE_LINE  # eyes above the middle of the head (the face texture paints them there)


def draw_laser_blaster(ctx: DrawContext) -> None:
    """Default laser blaster prop."""
    ctx.emission([0.25, 0.25, 0.25, 1.0])
    ctx.color(0.82, 0.84, 0.90)

    # Grip
    ctx.push()
    ctx.translate(0.0, -0.06, 0.0)
    ctx.rotate(12, 1, 0, 0)
    ctx.scale(0.05, 0.14, 0.07)
    ctx.cube(None)
    ctx.pop()

    # Receiver
    ctx.push()
    ctx.translate(0.0, 0.02, 0.02)
    ctx.scale(0.06, 0.08, 0.20)
    ctx.cube(None)
    ctx.pop()

    # Barrel
    ctx.push()
    ctx.translate(0.0, 0.03, 0.16)
    ctx.scale(0.04, 0.04, 0.14)
    ctx.cube(None)
    ctx.pop()

    # Cyan Emitter Tip
    ctx.emission([0.0, 0.85, 1.0, 1.0])
    ctx.color(0.0, 0.9, 1.0)
    ctx.push()
    ctx.translate(0.0, 0.03, 0.235)
    ctx.scale(0.03, 0.03, 0.02)
    ctx.cube(None)
    ctx.pop()

    ctx.emission([0.0, 0.0, 0.0, 1.0])
    ctx.color(1.0, 1.0, 1.0)


WHITE = (1.0, 1.0, 1.0)
GLOVE = (0.32, 0.32, 0.36)  # a guard's left hand: a dark tactical glove
CHROME = (0.85, 0.87, 0.92)  # ... the right: the cyberarm's chrome fingers
CORE = (0.0, 0.9, 1.0)
POLISHED = (0.78, 0.8, 0.86)  # a civilian's chrome: untextured and shiny
FINGERS_WITHIN = 9.0  # metres: nearer than this, a person's hands have fingers (mittens beyond)
COARSE_BEYOND = 14.0  # ... and further than this, their rounded parts have fewer facets

Material = tuple  # (colour, texture id, emission or None, shininess)
SKIN_RGB = (0.97, 0.8, 0.68)  # (the stubble texture's light over this matches the flesh's)


def _mix(a, b, k: float) -> tuple[float, float, float]:
    return tuple(x + (y - x) * k for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def _times(a, b) -> tuple[float, float, float]:
    return tuple(x * y for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def materials(
    spec: HumanoidSpec, textures: TextureMap, expression: str | None = None
) -> dict[str, Material]:
    """What each kind of part (``animation.rig``'s material names) is made of, for one
    person: skin, the face they are making, hair (and stubble, and a shaved scalp), their
    cybernetic eyes and chrome, the neon they wear, their shirt, jacket, trousers, boots -
    or a guard's armour, glove and chrome cyberarm."""
    tex = textures.get
    skin, shirt, pants = spec.skin_color, spec.shirt_color, spec.pants_color
    look = spec.look
    neon, eye = spec.neon_color, spec.eye_color
    hair2 = spec.hair_color2 or spec.hair_color
    civilian = not spec.has_armor
    # a civilian's shirt is light cloth that takes its colour; a guard's is dark fabric
    cloth = tex("shirt") if spec.has_armor else tex("cloth")
    flesh = tex("flesh")
    strands = tex("hair_strands")
    metal = tex("metal")
    jacket = (spec.jacket_color, None, None, 60.0)  # (shiny vinyl)
    # The skin is drawn the way the face is: the tone in the picture (``flesh_name``), the
    # colour white - so the head, the ears, the nose, the neck and the hands are lit just
    # like the face (a tone multiplied in before the lights are clamped came out brighter
    # than the face under a strong light) and the face does not show as a different colour
    toned = tex(flesh_name(skin))
    bare_skin: Material = (WHITE, toned, None, 0.0) if toned else (skin, flesh, None, 0.0)
    m: dict[str, Material] = {
        "skin": bare_skin,
        "face": (WHITE, tex(spec.face_for(expression)) or tex("skin"), None, 0.0),
        "hair": (spec.hair_color, strands, None, 24.0),
        "hair2": (hair2, strands, None, 24.0),
        "stubble": (_mix(_times(SKIN_RGB, skin), spec.hair_color, 0.2), tex("stubble"), None, 0.0),
        "scalp": (*bare_skin[:3], 30.0),
        "visor": (spec.visor_color, None, spec.visor_color, 0.0),
        "shades": ((0.05, 0.05, 0.07), None, _times(neon, (0.35, 0.35, 0.35)), 90.0),
        "eye": (eye, None, eye, 0.0),
        "chrome": (POLISHED, None, None, 110.0),  # (street chrome: bright, polished)
        "implant": (neon, None, neon, 0.0),
        "neon": (neon, None, neon, 0.0),
        "mask": ((0.12, 0.12, 0.14), None, None, 20.0),
        "lens": ((0.1, 0.5, 0.6), None, (0.05, 0.25, 0.3), 100.0),
        "gold": ((0.95, 0.75, 0.3), metal, None, 90.0),
        "shirt": (shirt, cloth, None, 0.0),
        "jacket": jacket,
        "torso": (shirt, cloth, None, 0.0),
        "plate": (WHITE, tex("armor"), None, 0.0),
        "core": (CORE, None, CORE, 0.0),
        "belt": (WHITE, tex("leather"), None, 0.0),
        "holster": (WHITE, tex("leather"), None, 0.0),
        "pants": (pants, tex("cloth") if civilian else tex("pants"), None, 0.0),
        "kneepad": (WHITE, tex("armor"), None, 0.0),
        "boot": (spec.boot_color, None, None, 50.0)
        if civilian
        else (WHITE, tex("leather"), None, 0.0),
    }
    if spec.has_armor:
        # armoured shoulders; a bare left upper arm, a dark sleeve and glove; the right arm
        # chromed cyberware to the fingertips
        m.update(
            shoulder_l=(WHITE, tex("armor"), None, 0.0),
            shoulder_r=(WHITE, tex("armor"), None, 0.0),
            upper_l=bare_skin,
            fore_l=(WHITE, tex("shirt"), None, 0.0),
            hand_l=(GLOVE, tex("leather"), None, 0.0),
            upper_r=(WHITE, metal, None, 0.0),
            fore_r=(WHITE, metal, None, 0.0),
            hand_r=(CHROME, metal, None, 0.0),
        )
        return m
    bare = m["skin"]
    top = look.top
    if top in ("jacket", "vest"):
        m["torso"] = jacket
    for side in "lr":
        if top == "jacket":  # long sleeves, bare hands
            m[f"shoulder_{side}"] = m[f"upper_{side}"] = m[f"fore_{side}"] = jacket
            m[f"hand_{side}"] = bare
        elif top == "vest":  # bare arms
            m[f"shoulder_{side}"] = jacket
            m[f"upper_{side}"] = m[f"fore_{side}"] = m[f"hand_{side}"] = bare
        else:  # a tee: short sleeves, bare forearms and hands
            m[f"shoulder_{side}"] = m[f"upper_{side}"] = m["torso"]
            m[f"fore_{side}"] = m[f"hand_{side}"] = bare
        if f"arm_{side}" in look.implants:  # a cyberarm: chrome from the shoulder down
            m[f"upper_{side}"] = m[f"fore_{side}"] = m[f"hand_{side}"] = m["chrome"]
    return m


class _Painter:
    """``animation.rig``'s builder for the screen: each part a rounded shape, scaled into
    its box, in its material."""

    def __init__(self, ctx: DrawContext, mats: dict[str, Material]) -> None:
        self.ctx = ctx
        self.mats = mats
        self.push, self.pop = ctx.push, ctx.pop
        self.translate, self.rotate = ctx.translate, ctx.rotate
        self.fast = getattr(ctx, "add_part", None)  # (a batch takes a part in one call)

    def part(self, name: str, size, shape: str, material: str) -> None:
        ctx = self.ctx
        color, tex, emission, shine = self.mats.get(material, (WHITE, None, None, 0.0))
        if self.fast is not None:
            self.fast(shape, size, color, tex, emission, shine)
            return
        ctx.push()
        ctx.scale(*size)
        ctx.color(*color)
        if emission is not None:
            ctx.emission([*emission, 1.0])
        if shine:
            ctx.specular((0.5, 0.5, 0.5), shine)
        ctx.shape(shape, tex)
        if shine:
            ctx.specular(None)
        if emission is not None:
            ctx.emission([0.0, 0.0, 0.0, 1.0])
        ctx.pop()


def draw_humanoid(
    ctx: DrawContext,
    spec: HumanoidSpec,
    pose: HumanoidPose,
    textures: TextureMap,
    fingers: bool = True,
) -> None:
    """Draws a posed humanoid at the current model-view origin (feet at y=0, facing +Z):
    the body part by part (``animation.rig``), what it holds, the sidearm."""
    kit = BodyKit(
        armor=spec.has_armor, visor=spec.has_visor, core_glow=spec.has_core_glow,
        holster=spec.has_holster, blaster=False, fingers=fingers, look=spec.look,
    )  # fmt: skip
    painter = _Painter(ctx, materials(spec, textures, pose.expression))
    rig.build_group(painter, pose, "trunk", kit, whole=True)
    rig.build_group(painter, pose, "arm_l", kit, whole=True)
    rig.build_group(painter, pose, "arm_r", kit, whole=True)
    ctx.color(1, 1, 1)

    # 4. ATTACHED PROP (Rendered at pose-driven coords)
    prop_drawer = PROP_DRAWERS.get(spec.prop) if spec.prop else None
    if pose.has_prop and prop_drawer:
        ctx.push()
        rig.lean(ctx, pose)
        ctx.translate(pose.prop_pos[0], pose.prop_pos[1] + pose.bob, pose.prop_pos[2])
        ctx.rotate(pose.prop_rot[0], 1, 0, 0)
        ctx.rotate(pose.prop_rot[1], 0, 1, 0)
        ctx.rotate(pose.prop_rot[2], 0, 0, 1)
        prop_drawer(ctx)
        ctx.pop()

    # Whatever the hands are holding (a takeaway box, a knife, a parcel...). Both hands on
    # the same thing: it sits between them.
    held = {"l": pose.hand_l, "r": pose.hand_r}
    if held["l"] or held["r"]:
        both = held["l"] is not None and held["l"] == held["r"]
        for side in ("r",) if both else ("l", "r"):
            name = held[side]
            if not name:
                continue
            pos, fore = hand_position(pose, side)
            if both:
                other, _ = hand_position(pose, "l")
                pos = (pos + other) / 2.0
            (x, y, z), yaw, pitch = held_transform(
                name,
                tuple(pos),
                fore,
                both=both,
                aim=pose.aim_l if side == "l" else pose.aim_r,
                tilt=pose.tilt_l if side == "l" else pose.tilt_r,
                side=side,
                axis=None if both else hands.held_axis(pose, side),
            )
            ctx.push()
            ctx.translate(x, y, z)
            if yaw:
                ctx.rotate(yaw, 0, 1, 0)
            if pitch:
                ctx.rotate(pitch, 1, 0, 0)
            draw_item(ctx, name, textures, ember=pose.ember)
            ctx.pop()

    # 5. LEGS
    rig.build_group(painter, pose, "leg_l", kit, whole=True)
    rig.build_group(painter, pose, "leg_r", kit, whole=True)
    ctx.color(1, 1, 1)


PROP_DRAWERS: dict[str, Callable[[DrawContext], None]] = {
    "laser_blaster": draw_laser_blaster,
}
"""Registry of hand-held prop meshes, looked up by ``HumanoidSpec.prop``."""


def register_prop(name: str, drawer: Callable[[DrawContext], None]) -> None:
    PROP_DRAWERS[name] = drawer


def draw_npc(
    ctx: DrawContext,
    npc: BaseNPC,
    textures: TextureMap,
    sparks: bool = True,
    eye: tuple[float, float, float] | None = None,
) -> None:
    """Positions an NPC in world space and draws its humanoid. A collapsed body (``body``,
    see ``animation.ragdoll``) is placed by its hip joint and turned by its trunk's rotation
    - the limbs are already posed to lie where they fall. While ``shock_timer`` runs,
    electricity crackles over it.

    On a real backend (``ctx.batches``) the whole person goes to the GPU as one batch
    (``render.batch``); seen from ``eye`` further off than ``FINGERS_WITHIN`` the hands
    are mittens, further than ``COARSE_BEYOND`` the rounded parts coarser."""
    ctx.push()
    body = getattr(npc, "body", None)
    height = npc.spec.height
    if body is not None:
        angle, axis = axis_angle(body.rotation)
        ctx.translate(*body.hip)
        if angle:
            ctx.rotate(angle, *axis)
        scale = getattr(body, "scale", 1.0)
        if scale != 1.0:
            ctx.scale(scale, scale, scale)
        ctx.translate(0.0, -HIP_HEIGHT, 0.0)
        height = scale
    else:
        ctx.translate(npc.x, npc.y, npc.z)
        ctx.rotate(npc.yaw, 0, 1, 0)
        if height != 1.0:
            ctx.scale(height, height, height)
    far = 0.0 if eye is None else math.dist(eye, (npc.x, npc.y + 1.0, npc.z))
    fingers = far < FINGERS_WITHIN
    if getattr(ctx, "batches", False):
        batch = BatchContext(ctx, world_scale=height, lo=far > COARSE_BEYOND)
        draw_humanoid(batch, npc.spec, npc.pose, textures, fingers=fingers)
        batch.flush()
    else:
        draw_humanoid(ctx, npc.spec, npc.pose, textures, fingers=fingers)
    ctx.pop()
    shock = getattr(npc, "shock_timer", 0.0) if sparks else 0.0
    if shock > 0.0:
        y = npc.y + 0.1
        height = 1.6 * (1.0 - min(1.0, npc.fall_angle / 90.0)) + 0.3
        draw_sparks(ctx, npc.x, y, npc.z, t=shock * 7.0, height=height, seed=npc.x + npc.z)
