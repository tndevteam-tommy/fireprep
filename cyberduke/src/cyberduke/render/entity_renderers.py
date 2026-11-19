"""Renderers for the interactive entities (terminal, security camera, vending machine).

Meshes are unchanged from the original ``entities.py``; the entity classes themselves now
hold only state and behaviour.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from cyberduke.core.geometry import AABB
from cyberduke.entities.camera import SecurityCamera
from cyberduke.entities.terminal import Terminal
from cyberduke.entities.vending_machine import VendingMachine
from cyberduke.render.beams import BeamOccluders, BoxSet, clip_rays
from cyberduke.render.context import DrawContext
from cyberduke.render.effects import draw_sparks
from cyberduke.render.textures import TextureMap


def _led_tint(emission) -> tuple[float, float, float]:
    """A dark diffuse colour in the hue of an over-bright emission, for lights (LEDs,
    lenses). Using the emission itself as the diffuse colour let the room lights push every
    channel past 1, so a red "alert" lens or a red LED came out white."""
    r, g, b = (float(c) for c in emission[:3])
    peak = max(r, g, b, 1e-6)
    return (0.25 * r / peak, 0.25 * g / peak, 0.25 * b / peak)


def _entity_transform(ctx: DrawContext, self) -> None:
    ctx.translate(self.x, self.y, self.z)
    if self.yaw != 0.0:
        ctx.rotate(self.yaw, 0, 1, 0)


def draw_terminal(
    ctx: DrawContext, self: Terminal, textures: TextureMap, sparks: bool = True
) -> None:
    """Netrunner workstation: desk, triple monitors, tower PC, cabling and swivel chair.

    The furniture is static and is compiled into a display list per terminal when a
    ``cache`` is given; only the central screen (state-driven glow/texture) is immediate.
    """
    ctx.cached(("terminal", id(self)), lambda: draw_terminal_static(ctx, self, textures))
    draw_terminal_screen(ctx, self, textures, sparks=sparks)


def _box(ctx: DrawContext, x, y, z, sx, sy, sz, rgb, tex=None) -> None:
    ctx.color(*rgb)
    ctx.box(x, y, z, sx, sy, sz, tex)


def draw_home_terminal_static(ctx: DrawContext, self: Terminal, textures: TextureMap) -> None:
    """Somebody's cheap desk at home (``style="home"``): a chipboard desk against the
    wall, a boxy old monitor, a keyboard with missing keys, a tower on the floor, a mug,
    a mess of cable, a stool. Local frame: the desk's back at -z, the stool side +z."""
    metal, wood = textures.get("metal"), textures.get("wood")
    ctx.push()
    _entity_transform(ctx, self)
    _box(ctx, 0.0, 0.72, -0.02, 1.06, 0.04, 0.56, (0.5, 0.42, 0.32), wood)  # top
    for lx in (-0.5, 0.5):
        for lz in (-0.27, 0.23):
            _box(ctx, lx, 0.35, lz, 0.04, 0.7, 0.04, (0.2, 0.2, 0.22), metal)
    _box(ctx, 0.0, 0.96, -0.14, 0.44, 0.4, 0.3, (0.55, 0.53, 0.48), metal)  # monitor case
    _box(ctx, 0.0, 0.76, -0.14, 0.16, 0.06, 0.14, (0.3, 0.3, 0.3), metal)  # its foot
    _box(ctx, 0.02, 0.745, 0.12, 0.4, 0.02, 0.14, (0.16, 0.16, 0.18), metal)  # keyboard
    for k in range(10):  # keys (a couple missing)
        if k in (3, 7):
            continue
        _box(ctx, -0.16 + k * 0.036, 0.758, 0.1, 0.028, 0.008, 0.026, (0.4, 0.42, 0.45))
    _box(ctx, 0.36, 0.0 + 0.24, -0.1, 0.18, 0.48, 0.42, (0.12, 0.12, 0.14), metal)  # tower
    ctx.emission((0.1, 0.9, 0.3))
    _box(ctx, 0.36, 0.4, 0.112, 0.02, 0.02, 0.005, (0.1, 0.9, 0.3))  # its power LED
    ctx.emission(None)
    ctx.color(0.85, 0.85, 0.8)
    ctx.push()
    ctx.translate(-0.36, 0.74, 0.08)
    ctx.cylinder(0.04, 0.04, 0.1, 8, None)  # a mug
    ctx.pop()
    _box(ctx, -0.2, 0.745, -0.22, 0.5, 0.01, 0.02, (0.05, 0.05, 0.06))  # cables
    _box(ctx, 0.05, 0.4, -0.27, 0.02, 0.7, 0.02, (0.05, 0.05, 0.06))
    ctx.color(0.25, 0.22, 0.2)  # the stool
    ctx.push()
    ctx.translate(0.0, 0.0, 0.45)
    ctx.cylinder(0.03, 0.03, 0.45, 8, metal)
    ctx.translate(0.0, 0.45, 0.0)
    ctx.cylinder(0.15, 0.15, 0.04, 10, textures.get("leather"))
    ctx.pop()
    ctx.color(1.0, 1.0, 1.0)
    ctx.pop()


def draw_terminal_static(ctx: DrawContext, self: Terminal, textures: TextureMap) -> None:
    if getattr(self, "style", "") == "home":
        draw_home_terminal_static(ctx, self, textures)
        return
    ctx.push()
    _entity_transform(ctx, self)

    no_emission = [0.0, 0.0, 0.0, 1.0]
    tex_metal = textures.get("metal")
    tex_screen_net = textures.get("screen_net", tex_metal)
    tex_screen_radar = textures.get("screen_radar", tex_metal)
    tex_soda = textures.get("soda_can", tex_metal)
    tex_leather = textures.get("leather", tex_metal)

    # -----------------------------------------------------------------------
    # 1. GROUNDED STEEL DESK FRAME & LEGS (Touching y=0.0)
    # -----------------------------------------------------------------------
    ctx.specular((0.5, 0.5, 0.6), 32.0)
    ctx.color(0.22, 0.24, 0.28)

    # Left & Right Foot Plates directly on floor
    for leg_x in [-1.15, 1.15]:
        # Floor foot skid
        ctx.push()
        ctx.translate(leg_x, 0.02, 0.0)
        ctx.scale(0.16, 0.04, 0.90)
        ctx.cube(tex_metal)
        ctx.pop()

        # Main vertical support column
        ctx.push()
        ctx.translate(leg_x, 0.43, -0.12)
        ctx.scale(0.08, 0.78, 0.08)
        ctx.cube(tex_metal)
        ctx.pop()

        # Diagonal rear brace
        ctx.push()
        ctx.translate(leg_x, 0.40, -0.28)
        ctx.rotate(-22, 1, 0, 0)
        ctx.scale(0.06, 0.75, 0.06)
        ctx.cube(tex_metal)
        ctx.pop()

    # Rear lower crossbar stretcher
    ctx.push()
    ctx.translate(0.0, 0.22, -0.32)
    ctx.scale(2.22, 0.06, 0.06)
    ctx.cube(tex_metal)
    ctx.pop()

    # -----------------------------------------------------------------------
    # 2. DESKTOP SURFACE (Height y=0.85)
    # -----------------------------------------------------------------------
    # Heavy beveled dark composite tabletop
    ctx.color(0.18, 0.20, 0.24)
    ctx.push()
    ctx.translate(0.0, 0.84, 0.0)
    ctx.scale(2.55, 0.05, 1.05)
    ctx.cube(tex_metal)
    ctx.pop()

    # Ergonomic front wrist rest edge
    ctx.color(0.12, 0.14, 0.18)
    ctx.push()
    ctx.translate(0.0, 0.84, 0.51)
    ctx.scale(2.45, 0.035, 0.06)
    ctx.cube(tex_metal)
    ctx.pop()

    # -----------------------------------------------------------------------
    # 3. UNDER-DESK NETRUNNER RIG & CABLE HARNESS
    # -----------------------------------------------------------------------
    # Tower chassis suspended on right floor bracket
    ctx.color(0.15, 0.16, 0.19)
    ctx.push()
    ctx.translate(0.85, 0.36, -0.10)
    ctx.scale(0.28, 0.62, 0.60)
    ctx.cube(tex_metal)
    ctx.pop()

    # Glowing front intake fan & LEDs
    ctx.emission([0.0, 0.8, 1.0, 1.0])
    ctx.color(0.0, 0.8, 1.0)
    ctx.push()
    ctx.translate(0.85, 0.36, 0.21)
    ctx.scale(0.20, 0.45, 0.02)
    ctx.cube(None)
    ctx.pop()
    ctx.emission(no_emission)

    # Thick braided black & cyan power conduits pooling onto the floor
    ctx.color(0.06, 0.06, 0.08)
    # Cable from desk to floor
    ctx.push()
    ctx.translate(0.0, 0.42, -0.42)
    ctx.rotate(12, 0, 0, 1)
    ctx.scale(0.08, 0.84, 0.08)
    ctx.cube(None)
    ctx.pop()
    # Cable pool along floor
    ctx.push()
    ctx.translate(0.20, 0.02, -0.445)  # (its front 5 mm behind the cable coming down)
    ctx.scale(0.60, 0.03, 0.12)
    ctx.cube(None)
    ctx.pop()

    # -----------------------------------------------------------------------
    # 4. ARTICULATED MONITOR MOUNTING MAST
    # -----------------------------------------------------------------------
    ctx.color(0.25, 0.26, 0.30)
    # Base desk clamp
    ctx.push()
    ctx.translate(0.0, 0.865, -0.38)
    ctx.scale(0.14, 0.04, 0.14)
    ctx.cube(tex_metal)
    ctx.pop()

    # Vertical mounting mast rising from desktop
    ctx.push()
    ctx.translate(0.0, 1.30, -0.38)
    ctx.scale(0.06, 0.82, 0.06)
    ctx.cube(tex_metal)
    ctx.pop()

    # Horizontal mounting crossbar arm
    ctx.push()
    ctx.translate(0.0, 1.36, -0.35)
    ctx.scale(1.90, 0.04, 0.04)
    ctx.cube(tex_metal)
    ctx.pop()

    # -----------------------------------------------------------------------
    # 5. MONITOR ARRAY - what is on this particular desk depends on its style
    # -----------------------------------------------------------------------
    style = getattr(self, "style", "triple")
    has_left = style in ("triple", "double", "damaged", "overgrown")
    has_right = style == "triple"
    left_dead = style == "damaged"

    # A. Central Interactive Terminal Monitor (the face is drawn live: see draw_terminal_screen)
    ctx.push()
    ctx.translate(0.0, 1.36, -0.22)
    ctx.rotate(10, 1, 0, 0)  # Tilted back for viewing
    ctx.color(0.14, 0.15, 0.18)
    ctx.push()
    ctx.scale(1.24, 0.76, 0.06)
    ctx.cube(tex_metal)
    ctx.pop()
    ctx.pop()

    if has_left:
        # B. Left Auxiliary Monitor (Network Topology / Code) - dead and knocked askew on a
        #    damaged desk, facing the chair otherwise
        ctx.push()
        ctx.translate(-0.95, 1.36 if not left_dead else 1.30, -0.16)
        ctx.rotate(24 if not left_dead else 44, 0, 1, 0)
        ctx.rotate(6, 1, 0, 0)
        if left_dead:
            ctx.rotate(-9, 0, 0, 1)
        ctx.color(0.14, 0.15, 0.18)
        ctx.push()
        ctx.scale(0.82, 0.62, 0.05)
        ctx.cube(tex_metal)
        ctx.pop()
        if left_dead:
            ctx.specular((0.6, 0.6, 0.7), 60.0)
            ctx.color(0.05, 0.05, 0.06)
            ctx.push()
            ctx.translate(0.0, 0.0, 0.026)
            ctx.scale(0.76, 0.56, 0.01)
            ctx.cube(None)
            ctx.pop()
            ctx.specular(None)
            ctx.emission([0.2, 0.25, 0.35, 1.0])  # the crack catches what light there is
            ctx.color(0.8, 0.85, 1.0)
            ctx.push()
            ctx.translate(0.05, 0.04, 0.033)
            ctx.rotate(28, 0, 0, 1)
            ctx.scale(0.5, 0.006, 0.004)
            ctx.cube(None)
            ctx.pop()
            ctx.emission(no_emission)
        else:
            ctx.emission([0.0, 0.45, 0.70, 1.0])
            ctx.color(1.0, 1.0, 1.0)
            ctx.push()
            ctx.translate(0.0, 0.0, 0.026)
            ctx.scale(0.76, 0.56, 0.01)
            ctx.cube(tex_screen_net)
            ctx.pop()
            ctx.emission(no_emission)
        ctx.pop()

    if has_right:
        # C. Right Auxiliary Monitor (Radar / Telemetry)
        ctx.push()
        ctx.translate(0.95, 1.36, -0.16)
        ctx.rotate(-24, 0, 1, 0)
        ctx.rotate(6, 1, 0, 0)
        ctx.color(0.14, 0.15, 0.18)
        ctx.push()
        ctx.scale(0.82, 0.62, 0.05)
        ctx.cube(tex_metal)
        ctx.pop()
        ctx.emission([0.65, 0.45, 0.05, 1.0])
        ctx.color(1.0, 1.0, 1.0)
        ctx.push()
        ctx.translate(0.0, 0.0, 0.026)
        ctx.scale(0.76, 0.56, 0.01)
        ctx.cube(tex_screen_radar)
        ctx.pop()
        ctx.emission(no_emission)
        ctx.pop()
    elif style == "damaged":
        # the right monitor is gone: a bare arm stub and the cable it left dangling
        ctx.color(0.2, 0.21, 0.24)
        ctx.push()
        ctx.translate(0.7, 1.2, -0.2)
        ctx.rotate(-30, 0, 0, 1)
        ctx.scale(0.05, 0.45, 0.05)
        ctx.cube(tex_metal)
        ctx.pop()
        ctx.color(0.08, 0.08, 0.1)
        ctx.push()
        ctx.translate(0.86, 0.98, -0.12)
        ctx.rotate(12, 0, 0, 1)
        ctx.scale(0.02, 0.3, 0.02)
        ctx.cube(None)
        ctx.pop()

    if style in ("triple", "double"):
        # D. Overhead Narrow Diagnostic Ticker
        ctx.push()
        ctx.translate(0.0, 1.82, -0.28)
        ctx.rotate(-16, 1, 0, 0)
        ctx.color(0.14, 0.15, 0.18)
        ctx.push()
        ctx.scale(1.05, 0.18, 0.04)
        ctx.cube(tex_metal)
        ctx.pop()
        ctx.emission([0.0, 0.7, 0.8, 1.0])
        ctx.color(0.0, 0.8, 1.0)
        ctx.push()
        ctx.translate(0.0, 0.0, 0.022)
        ctx.scale(0.98, 0.12, 0.01)
        ctx.cube(None)
        ctx.pop()
        ctx.emission(no_emission)
        ctx.pop()

    # -----------------------------------------------------------------------
    # 6. DESKTOP PERIPHERALS & NETRUNNER GEAR (per style)
    # -----------------------------------------------------------------------
    # Mechanical Backlit Cyberdeck / Keyboard - skewed on the untidier desks
    kb_yaw = {"triple": 0.0, "double": -7.0, "single": 12.0, "damaged": -18.0, "overgrown": 5.0}[
        style
    ]
    ctx.push()
    ctx.translate(0.0 if style != "single" else 0.12, 0.88, 0.18)
    ctx.rotate(kb_yaw, 0, 1, 0)
    ctx.color(0.12, 0.13, 0.15)
    ctx.push()
    ctx.scale(0.62, 0.03, 0.24)
    ctx.cube(tex_metal)
    ctx.pop()
    if style != "damaged":  # a dead keyboard on the damaged desk
        ctx.emission([0.0, 0.6, 0.8, 1.0])
        ctx.color(0.0, 0.7, 0.9)
        ctx.push()
        ctx.translate(0.0, 0.016, 0.0)
        ctx.scale(0.56, 0.01, 0.18)
        ctx.cube(None)
        ctx.pop()
        ctx.emission(no_emission)
    ctx.pop()

    if style in ("triple", "double"):
        # Left Datapad Deck
        ctx.color(0.18, 0.19, 0.22)
        ctx.push()
        ctx.translate(-0.65, 0.875, 0.12)
        ctx.rotate(-20 if style == "double" else 0, 0, 1, 0)
        ctx.scale(0.24, 0.02, 0.30)
        ctx.cube(tex_metal)
        ctx.pop()

    if style in ("triple", "damaged"):
        # Discarded Cyber-soda Can on right corner
        ctx.color(1.0, 1.0, 1.0)
        ctx.push()
        ctx.translate(0.82, 0.865, 0.25)
        ctx.cylinder(0.035, 0.035, 0.12, slices=8, tex_id=tex_soda)
        ctx.pop()
    if style in ("double", "single", "damaged"):
        # more cans, on their sides: somebody has been here a long time
        ctx.color(1.0, 1.0, 1.0)
        for k, (cx_, cz_, yaw_) in enumerate(
            ((0.55, 0.45, 40.0), (-0.9, 0.3, 110.0), (0.95, -0.05, 75.0))
        ):
            if style == "single" and k == 0:
                continue
            ctx.push()
            ctx.translate(cx_, 0.9, cz_)
            ctx.rotate(yaw_, 0, 1, 0)
            ctx.rotate(90, 1, 0, 0)
            ctx.cylinder(0.035, 0.035, 0.12, slices=8, tex_id=tex_soda)
            ctx.pop()
    if style in ("single", "damaged", "overgrown"):
        # a stack of printouts and loose sheets
        ctx.color(0.7, 0.68, 0.6)
        for k in range(4 if style != "overgrown" else 2):
            ctx.push()
            ctx.translate(-0.75 + 0.03 * k, 0.87 + 0.006 * k, 0.3 - 0.02 * k)
            ctx.rotate(8.0 * k - 10.0, 0, 1, 0)
            ctx.scale(0.22, 0.004, 0.3)
            ctx.cube(None)
            ctx.pop()
    if style == "single":
        # a mug
        ctx.color(0.85, 0.82, 0.78)
        ctx.push()
        ctx.translate(0.62, 0.865, 0.38)
        ctx.cylinder(0.045, 0.045, 0.1, slices=8, tex_id=None)
        ctx.pop()
    if style == "overgrown":
        # fungus has claimed the desk: vines over the edge and three caps on the top
        ctx.color(0.18, 0.3, 0.17)
        ctx.push()
        ctx.translate(0.9, 0.7, 0.0)
        ctx.rotate(12, 0, 0, 1)
        ctx.scale(0.08, 0.5, 0.6)
        ctx.cube(textures.get("moss"))
        ctx.pop()
        for cx_, cz_, sc in ((0.75, 0.35, 0.5), (0.95, 0.1, 0.35), (-0.85, -0.1, 0.42)):
            ctx.color(0.45, 0.43, 0.38)
            ctx.push()
            ctx.translate(cx_, 0.86, cz_)
            ctx.cylinder(0.09 * sc, 0.07 * sc, 0.32 * sc, slices=8, tex_id=textures.get("fungus"))
            ctx.pop()
            ctx.emission([0.25, 1.1, 3.4, 1.0])
            ctx.color(0.35, 0.6, 1.0)
            ctx.push()
            ctx.translate(cx_, 0.86 + 0.3 * sc, cz_)
            ctx.cylinder(0.26 * sc, 0.04 * sc, 0.16 * sc, slices=10, tex_id=textures.get("spore"))
            ctx.pop()
            ctx.emission(no_emission)
    ctx.color(1.0, 1.0, 1.0)

    # -----------------------------------------------------------------------
    # 7. ERGONOMIC NETRUNNER SWIVEL CHAIR (Grounded at y=0.0)
    # -----------------------------------------------------------------------
    ctx.push()
    # Positioned in front of the desk; how it was left says something about who sat there
    chair_yaw = {"triple": 16, "double": -38, "single": 70, "damaged": 150, "overgrown": -95}[style]
    chair_x = {"triple": 0.10, "double": -0.25, "single": 0.35, "damaged": -0.55, "overgrown": 0.4}[
        style
    ]
    # A chair left turned away has its backrest toward the desk: it sits further out so the
    # backrest clears the desk's front edge instead of passing through it.
    chair_z = {"damaged": 0.92, "overgrown": 0.88}.get(style, 0.72)
    ctx.translate(chair_x, 0.0, chair_z)
    ctx.rotate(chair_yaw, 0, 1, 0)

    # 5 Caster wheels touching the floor (y=0.0)
    ctx.color(0.12, 0.12, 0.14)
    chair_rad = 0.32
    for i in range(5):
        ang = 2.0 * math.pi * i / 5.0
        cx = math.cos(ang) * chair_rad
        cz = math.sin(ang) * chair_rad

        # Caster wheel block
        ctx.push()
        ctx.translate(cx, 0.02, cz)
        ctx.scale(0.05, 0.04, 0.05)
        ctx.cube(None)
        ctx.pop()

        # Star spoke arm from the hub out to the caster (from the hub's edge: meeting in the
        # middle, the five spokes' tops shared a plane and z-fought)
        inner = 0.03
        ctx.push()
        ctx.translate(
            math.cos(ang) * (inner + chair_rad) / 2.0,
            0.03,
            math.sin(ang) * (inner + chair_rad) / 2.0,
        )
        ctx.rotate(90.0 - math.degrees(ang), 0, 1, 0)  # (its length, local z, out to the caster)
        ctx.scale(0.04, 0.025, chair_rad - inner)
        ctx.cube(tex_metal)
        ctx.pop()

    # Center Hub & Hydraulic Lift Piston (y=0.03 to y=0.44)
    ctx.color(0.28, 0.30, 0.35)
    ctx.push()
    ctx.translate(0.0, 0.03, 0.0)
    ctx.cylinder(0.035, 0.035, 0.41, slices=8, tex_id=tex_metal)
    ctx.pop()

    # Seat Cushion (Height y=0.46)
    ctx.color(0.16, 0.18, 0.22)
    ctx.push()
    ctx.translate(0.0, 0.46, 0.0)
    ctx.scale(0.54, 0.07, 0.50)
    ctx.cube(tex_leather)
    ctx.pop()

    # Neon Cyan Accent Piping on seat edge
    ctx.emission([0.0, 0.7, 0.9, 1.0])
    ctx.color(0.0, 0.8, 1.0)
    ctx.push()
    ctx.translate(0.0, 0.46, 0.252)
    ctx.scale(0.52, 0.015, 0.01)
    ctx.cube(None)
    ctx.pop()
    ctx.emission(no_emission)

    # High Bucket Seat Backrest (Rising to y=1.12)
    ctx.color(0.15, 0.16, 0.20)
    ctx.push()
    ctx.translate(0.0, 0.80, 0.24)
    ctx.rotate(-8, 1, 0, 0)  # Ergonomic recline
    ctx.scale(0.48, 0.62, 0.06)
    ctx.cube(tex_leather)
    ctx.pop()

    # Headrest cushion
    ctx.color(0.12, 0.13, 0.16)
    ctx.push()
    ctx.translate(0.0, 1.15, 0.28)
    ctx.scale(0.32, 0.14, 0.08)
    ctx.cube(tex_leather)
    ctx.pop()

    # Left & Right Armrests
    ctx.color(0.18, 0.20, 0.24)
    for side in [-0.28, 0.28]:
        # Upright stem
        ctx.push()
        ctx.translate(side, 0.56, 0.05)
        ctx.scale(0.04, 0.18, 0.04)
        ctx.cube(tex_metal)
        ctx.pop()
        # Arm pad
        ctx.push()
        ctx.translate(side, 0.66, 0.05)
        ctx.scale(0.07, 0.03, 0.28)
        ctx.cube(tex_leather)
        ctx.pop()

    ctx.pop()  # End Chair

    ctx.pop()  # End Workstation

    # Reset default specular material
    ctx.specular(None)
    ctx.color(1.0, 1.0, 1.0)


def draw_terminal_screen(
    ctx: DrawContext, self: Terminal, textures: TextureMap, sparks: bool = True
) -> None:
    """The central monitor face: red/amber-pulsing while locked or hacking, green when hacked
    (or remotely emptied of its data), dead black with a dying flicker once overloaded."""
    no_emission = [0.0, 0.0, 0.0, 1.0]
    tex_locked = textures.get("terminal_locked")
    tex_unlocked = textures.get("terminal_unlocked")
    home = getattr(self, "style", "") == "home"
    ctx.push()
    _entity_transform(ctx, self)
    if home:  # the old monitor's face
        ctx.translate(0.0, 0.97, 0.0)
    else:
        ctx.translate(0.0, 1.36, -0.22)
        ctx.rotate(10, 1, 0, 0)  # Tilted back for viewing
    # Same specular as the rest of the workstation (the screen is glossy too)
    ctx.specular((0.5, 0.5, 0.6), 32.0)
    # Screen display face
    if self.destroyed:
        dying = max(0.0, self.spark_timer / 4.0) * (0.5 + 0.5 * math.sin(self.anim_time * 40.0))
        screen_emission = [0.25 * dying, 0.05 * dying, 0.0, 1.0]
        screen_tex = None
    elif self.is_hacked or self.intel_stolen or self.doors_overridden:
        screen_emission = [0.0, 0.85, 0.35, 1.0]
        screen_tex = tex_unlocked
    elif self.progress > 0.0:
        pulse = 0.5 + 0.5 * math.sin(self.anim_time * 12.0)
        screen_emission = [0.8 * pulse + 0.2, 0.4 * pulse, 0.0, 1.0]
        screen_tex = tex_locked
    else:
        screen_emission = [0.65, 0.08, 0.08, 1.0]
        screen_tex = tex_locked

    ctx.emission(screen_emission)
    if self.destroyed:
        ctx.color(0.03, 0.03, 0.04)  # dead glass
    else:
        ctx.color(1.0, 1.0, 1.0)
    ctx.push()
    if home:
        ctx.translate(0.0, 0.0, 0.012)
        ctx.scale(0.36, 0.28, 0.01)
    else:
        ctx.translate(0.0, 0.0, 0.032)
        ctx.scale(1.18, 0.70, 0.01)
    ctx.cube(screen_tex)
    ctx.pop()
    ctx.emission(no_emission)
    ctx.specular(None)
    ctx.pop()
    ctx.color(1.0, 1.0, 1.0)
    if sparks and self.spark_timer > 0.0:
        draw_sparks(ctx, self.x, self.y + 0.9, self.z, t=self.anim_time, radius=0.5, height=0.9)


def draw_security_camera(
    ctx: DrawContext,
    self: SecurityCamera,
    obstacles: Sequence[AABB],
    slices: int = 12,
    occluders: BeamOccluders | None = None,
    mount_top: float | None = None,
) -> None:
    """Ceiling camera with sweeping head and a laser frustum fan clipped where it meets
    something. While the player is hijacking it the head is not drawn (the view is from
    inside the lens) and the beam shows what the feed covers.

    ``mount_top`` is the height of whatever the camera hangs from (the ceiling, or the dome
    above the greenhouse maze): a drop rod runs up to it, so the camera is not left
    floating in the air.

    ``occluders`` is the level's visible geometry (``render.beams``); the beam then stops
    on the surfaces the player sees, with ``obstacles`` adding the moving or entity boxes
    (door leaves, desks). Without it the beam is clipped against ``obstacles`` alone."""
    obs = obstacles
    plate_top = self.y + 0.21
    if mount_top is not None and mount_top > plate_top + 0.02:
        # Drop rod from the mounting plate up to the ceiling / dome, with a fixing plate.
        length = mount_top - plate_top
        ctx.color(0.14, 0.15, 0.18)
        ctx.box(self.x, plate_top + length / 2.0, self.z, 0.05, length, 0.05)
        ctx.box(self.x, mount_top - 0.01, self.z, 0.18, 0.02, 0.18)
        ctx.color(1.0, 1.0, 1.0)
    if self.hijacked:
        _draw_camera_beam(ctx, self, obs, slices, occluders)
        return
    ctx.push()
    ctx.translate(self.x, self.y, self.z)

    no_emission = [0.0, 0.0, 0.0, 1.0]

    # 1. Base Mounting Plate attached to ceiling
    ctx.color(0.20, 0.22, 0.25)
    ctx.push()
    ctx.translate(0.0, 0.18, 0.0)
    ctx.scale(0.40, 0.06, 0.40)
    ctx.cube(None)
    ctx.pop()

    # 2. Swivel Neck & Pivot Ball
    ctx.color(0.12, 0.14, 0.16)
    ctx.push()
    ctx.translate(0.0, 0.08, 0.0)
    ctx.cylinder(0.08, 0.08, 0.12, slices=8, tex_id=None)
    ctx.pop()

    # 3. Pivoting Camera Body & Lens. A dead camera hangs limp from the swivel (the head
    # droops forward and sags a little sideways); a live one sits at its working tilt.
    droop = self.head_droop
    ctx.rotate(self.current_yaw, 0, 1, 0)
    ctx.rotate(self.current_tilt + droop * (self.DROOP_DEG - self.current_tilt), 1, 0, 0)
    if droop > 0.0:
        ctx.rotate(droop * 14.0, 0, 0, 1)

    # Camera Chassis
    ctx.color(0.16, 0.18, 0.22)
    ctx.push()
    ctx.translate(0.0, -0.05, 0.0)
    ctx.scale(0.24, 0.18, 0.42)
    ctx.cube(None)
    ctx.pop()

    # Top Sun-shield Visor
    ctx.color(0.10, 0.11, 0.13)
    ctx.push()
    ctx.translate(0.0, 0.06, 0.06)
    ctx.scale(0.28, 0.03, 0.48)
    ctx.cube(None)
    ctx.pop()

    # Camera Optical Lens Ring (seated on the chassis face at z=0.21, out to z=0.26)
    ctx.color(0.25, 0.26, 0.30)
    ctx.push()
    ctx.translate(0.0, -0.05, 0.20)
    ctx.rotate(90, 1, 0, 0)
    ctx.cylinder(0.07, 0.07, 0.06, slices=10, tex_id=None)
    ctx.pop()

    # Glowing Optical Eye Sensor: red when it sees you, cyan sweeping, green under your
    # control, dark (with a faint red standby dot) while shut down.
    if not self.is_online:
        lens_color = [0.25, 0.02, 0.02, 1.0]
    elif self.hijacked:
        lens_color = [0.3, 4.0, 0.6, 1.0]
    elif self.is_alerted:
        pulse = 0.5 + 0.5 * math.sin(self.alert_pulse)
        lens_color = [4.0, 0.2 * pulse, 0.2 * pulse, 1.0]
    else:
        lens_color = [0.0, 3.4, 4.0, 1.0]

    ctx.emission(lens_color)
    ctx.color(*_led_tint(lens_color))
    ctx.push()
    ctx.translate(0.0, -0.05, 0.265)  # in front of the ring: it used to sit inside it, hidden
    ctx.scale(0.10, 0.10, 0.01)
    ctx.cube(None)
    ctx.pop()
    ctx.emission(no_emission)

    ctx.pop()  # End camera transforms

    if not self.is_online:
        ctx.color(1.0, 1.0, 1.0)
        return  # no power, no beam
    _draw_camera_beam(ctx, self, obs, slices, occluders)


def _draw_camera_beam(
    ctx: DrawContext,
    self: SecurityCamera,
    obs: Sequence[AABB],
    slices: int = 12,
    occluders: BeamOccluders | None = None,
) -> None:
    # Only what the beam can reach matters (the level has a hundred-plus colliders and
    # over a thousand props; the beam is 8.5 m long).
    reach = self.max_range
    near = [
        b
        for b in obs or ()
        if b.min_x - reach <= self.x <= b.max_x + reach
        and b.min_z - reach <= self.z <= b.max_z + reach
    ]
    boxes = BoxSet.from_aabbs(near)
    if occluders is not None:
        boxes = occluders.around(self.x, self.z, reach).concat(boxes)
    # 4. Projected Ground Laser Cone (Frustum Visualizer, blocked by walls & obstacles)
    if self.hijacked:
        beam_rgba = (0.3, 1.0, 0.5, 0.18)
    elif self.is_alerted:
        beam_rgba = (1.0, 0.15, 0.15, 0.45)
    else:
        beam_rgba = (0.0, 0.85, 1.0, 0.22)

    rad_pitch = math.radians(max(8.0, self.current_tilt))  # a level head still lights the floor
    t = self.y / math.sin(rad_pitch)  # Distance along ray to floor y=0

    half_fov = self.fov_deg * 0.5
    num_slices = max(2, slices)
    y_lens = self.y - 0.1
    lens = (self.x, y_lens, self.z)

    ground = []
    for i in range(num_slices + 1):
        ang = -half_fov + (self.fov_deg * i / num_slices)
        tot_rad = math.radians(self.current_yaw + ang)
        gx = self.x + math.sin(tot_rad) * math.cos(rad_pitch) * t
        gz = self.z + math.cos(tot_rad) * math.cos(rad_pitch) * t

        # Clamp to max_range
        dist = math.hypot(gx - self.x, gz - self.z)
        if dist > self.max_range:
            s = self.max_range / dist
            gx = self.x + (gx - self.x) * s
            gz = self.z + (gz - self.z) * s
        ground.append((gx, 0.02, gz))

    # Each ray stops at the first surface it meets in 3D: it passes over a low crate and
    # lands behind it, ends on the face of a wall, or on the top of a rack.
    hits = clip_rays(lens, ground, boxes)
    fan_pts = [p for p, _ in hits]

    segments = []
    # Direct laser beam rays from lens to perimeter endpoints (a quarter of them)
    step = max(1, num_slices // 4)
    segments.extend((lens, fan_pts[idx]) for idx in range(0, num_slices + 1, step))
    # Perimeter boundary connecting adjacent endpoints
    segments.extend((fan_pts[i], fan_pts[i + 1]) for i in range(len(fan_pts) - 1))
    # Vertical drop seam line where a ray strikes an upright face above the ground
    segments.extend((p, (p[0], 0.02, p[2])) for p, upright in hits if upright)
    ctx.lines(segments, beam_rgba)
    ctx.color(1.0, 1.0, 1.0)


def draw_vending_machine(
    ctx: DrawContext, self: VendingMachine, textures: TextureMap, sparks: bool = True
) -> None:
    """'NEON COLD' soda dispenser with marquee, product display, cred slot and hopper.

    Cabinet geometry is compiled into a display list when a ``cache`` is given; the
    marquee, product cans, cred-slot LED and dispensed can depend on state and stay immediate.
    """
    ctx.cached(("vending", id(self)), lambda: draw_vending_machine_static(ctx, self, textures))
    draw_vending_machine_dynamic(ctx, self, textures, sparks=sparks)


def draw_vending_machine_static(
    ctx: DrawContext, self: VendingMachine, textures: TextureMap
) -> None:
    ctx.push()
    _entity_transform(ctx, self)

    tex_metal = textures.get("metal", None)
    tex_hazard = textures.get("hazard", None)

    # 1. Main Outer Chassis (W=1.2, H=2.3, D=0.9)
    ctx.color(0.18, 0.20, 0.24)
    ctx.push()
    ctx.translate(0.0, 1.15, 0.0)
    ctx.scale(1.20, 2.30, 0.90)
    ctx.cube(tex_metal)
    ctx.pop()

    # 2. Base Kickplate with hazard caution stripes
    ctx.color(0.85, 0.85, 0.85)
    ctx.push()
    ctx.translate(0.0, 0.10, 0.02)
    ctx.scale(1.22, 0.20, 0.92)
    ctx.cube(tex_hazard)
    ctx.pop()

    # 4. Beverage Display Window & Shelves
    ctx.color(0.08, 0.10, 0.12)
    ctx.push()
    ctx.translate(0.0, 1.30, 0.46)
    ctx.scale(1.02, 1.05, 0.03)
    ctx.cube(None)
    ctx.pop()

    # 5. Keypad, Coin / Cred Acceptor Slot
    ctx.color(0.12, 0.14, 0.16)
    ctx.push()
    ctx.translate(0.38, 0.65, 0.465)  # (5 mm proud of the marquee it overlaps)
    ctx.scale(0.24, 0.42, 0.03)
    ctx.cube(tex_metal)
    ctx.pop()

    # 6. Dispense Hopper Tray at Bottom
    ctx.color(0.10, 0.11, 0.14)
    ctx.push()
    ctx.translate(0.0, 0.35, 0.46)
    ctx.scale(0.72, 0.22, 0.08)
    ctx.cube(tex_metal)
    ctx.pop()

    ctx.pop()
    ctx.color(1.0, 1.0, 1.0)


def draw_vending_machine_dynamic(
    ctx: DrawContext, self: VendingMachine, textures: TextureMap, sparks: bool = True
) -> None:
    no_emission = [0.0, 0.0, 0.0, 1.0]
    tex_soda = textures.get("soda_can", None)
    tex_sign = textures.get("neon_sign", None)
    ctx.push()
    _entity_transform(ctx, self)

    # Out of order after a short (``VendingMachine.broken``): the marquee dead but for a
    # stutter now and then, the display dim, the slot's lamp blinking red
    broken = getattr(self, "broken", False) and not self.is_sparking
    t = self.anim_time
    stutter = math.sin(t * 7.3) + math.sin(t * 2.9 + 1.0) > 1.55

    # 3. Illuminated Top Header Marquee ("NEON COLD")
    if self.is_sparking:
        flicker = 0.5 + 0.5 * math.sin(self.anim_time * 35.0)
        marquee_emission = [4.0 * flicker, 0.5 * flicker, 3.5 * flicker, 1.0]
    elif broken:
        k = 0.9 if stutter else 0.06
        marquee_emission = [4.0 * k, 0.0, 3.2 * k, 1.0]
    else:
        marquee_emission = [4.0, 0.0, 3.2, 1.0]

    ctx.emission(marquee_emission)
    dead = 0.12 if broken and not stutter else 1.0  # (its tubes out: dark in any light)
    ctx.color(dead, dead, dead)
    ctx.push()
    ctx.translate(0.0, 2.05, 0.46)
    ctx.scale(1.05, 0.35, 0.04)
    ctx.cube(tex_sign)
    ctx.pop()
    ctx.emission(no_emission)

    # Cans on Display Shelves
    if self.is_sparking:
        ctx.emission([3.0, 2.0, 0.0])
        ctx.color(1.0, 1.0, 1.0)
    elif broken:
        ctx.emission([0.0, 0.12, 0.15])
        ctx.color(0.3, 0.3, 0.32)
    else:
        ctx.emission([0.0, 2.0, 2.5])
        ctx.color(1.0, 1.0, 1.0)
    for row_y in [1.05, 1.35, 1.62]:
        for col_x in [-0.35, -0.12, 0.12, 0.35]:
            ctx.push()
            ctx.translate(col_x, row_y, 0.45)
            ctx.cylinder(0.035, 0.035, 0.11, slices=8, tex_id=tex_soda)
            ctx.pop()
    ctx.emission(no_emission)

    # Cred Slot Status LED
    if self.is_sparking:
        slot_emission = [4.0, 0.2, 0.2, 1.0]
    elif broken:
        on = int(t * 2.5) % 2 == 0
        slot_emission = [4.0, 0.2, 0.2, 1.0] if on else [0.4, 0.02, 0.02, 1.0]
    elif self.cooldown > 0.0:
        slot_emission = [2.5, 2.0, 0.0, 1.0]
    else:
        slot_emission = [0.0, 3.6, 0.8, 1.0]

    ctx.emission(slot_emission)
    ctx.color(*_led_tint(slot_emission))
    ctx.push()
    ctx.translate(0.38, 0.76, 0.48)
    ctx.scale(0.14, 0.03, 0.02)
    ctx.cube(None)
    ctx.pop()
    ctx.emission(no_emission)

    # Dispensed Soda Can sitting in tray
    if self.is_sparking or self.cooldown > 4.0:
        ctx.color(1.0, 1.0, 1.0)
        ctx.push()
        ctx.translate(0.0, 0.30, 0.48)
        ctx.rotate(90, 1, 0, 0)
        ctx.cylinder(0.035, 0.035, 0.11, slices=8, tex_id=tex_soda)
        ctx.pop()

    ctx.pop()  # End Vending Machine
    ctx.color(1.0, 1.0, 1.0)

    # Shorted: electricity pops off the cabinet for as long as it sparks, in bursts
    if sparks and self.is_sparking:
        burst = 0.5 + 0.5 * math.sin(self.anim_time * 9.0) > 0.35
        if burst:
            draw_sparks(
                ctx, self.x, self.y + 0.5, self.z, t=self.anim_time, radius=0.75, height=1.7,
                count=10, seed=self.x,
            )  # fmt: skip
    elif sparks and broken and stutter:  # still broken: a spark or two with each stutter
        draw_sparks(
            ctx, self.x, self.y + 1.2, self.z, t=self.anim_time, radius=0.5, height=0.9,
            count=3, seed=self.x + 1.0,
        )  # fmt: skip


def draw_access_panel(ctx: DrawContext, panel, textures: TextureMap) -> None:
    """A door's access panel on the wall beside it: a grimy keypad and card reader, a
    little screen, and the lock's lamp - red locked, green released."""
    metal = textures.get("metal")
    ctx.push()
    ctx.translate(panel.x, panel.y, panel.z)
    ctx.rotate(panel.yaw, 0, 1, 0)  # its face points along local +z
    _box(ctx, 0.0, 0.0, -0.02, 0.2, 0.3, 0.05, (0.22, 0.23, 0.26), metal)
    _box(ctx, 0.0, 0.09, 0.006, 0.14, 0.05, 0.01, (0.05, 0.08, 0.06))
    released = panel.lock_released
    ctx.emission((0.2, 1.4, 0.5) if released else (1.2, 0.2, 0.15))
    _box(
        ctx, 0.0, 0.09, 0.012, 0.12, 0.035, 0.004, (0.2, 0.9, 0.4) if released else (0.9, 0.2, 0.15)
    )
    _box(
        ctx,
        0.07,
        0.125,
        0.012,
        0.02,
        0.02,
        0.006,
        (0.2, 0.9, 0.4) if released else (0.9, 0.2, 0.15),
    )
    ctx.emission(None)
    for row in range(4):
        for col in range(3):
            _box(
                ctx,
                -0.04 + col * 0.04,
                0.035 - row * 0.032,
                0.006,
                0.028,
                0.022,
                0.012,
                (0.5, 0.52, 0.55),
            )
    _box(ctx, 0.0, -0.11, 0.006, 0.12, 0.012, 0.01, (0.02, 0.02, 0.02))  # the card slot
    ctx.color(1.0, 1.0, 1.0)
    ctx.pop()


def draw_pickup(ctx: DrawContext, pickup, textures: TextureMap) -> None:
    """Something lying about that can be picked up (drawn by its item's mesh)."""
    if pickup.taken:
        return
    from cyberduke.render.items import draw_item

    ctx.push()
    ctx.translate(pickup.x, pickup.y, pickup.z)
    if pickup.yaw:
        ctx.rotate(pickup.yaw, 0, 1, 0)
    draw_item(ctx, pickup.item_id, textures)
    ctx.pop()
