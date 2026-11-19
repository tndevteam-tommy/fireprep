"""Cigarette smoke: a thin trail curling up off the lit end, and the lungful blown out
after a drag.

The smoking clip (``animation.street.smoke``) says, moment to moment, how hot the tip is
(``HumanoidPose.ember``: the cherry flares on a drag - ``render.items`` lights it),
whether the cigarette is being drawn on (``inhaling``: nothing curls off the tip while
it is - the smoke goes down the smoker's throat instead) and how hard smoke is being blown
out of the mouth (``exhale``). ``SmokeRenderer`` turns that into particles: a few a second
off the tip, rising slowly, swaying and spreading as they go; a burst from the mouth,
pushed forwards, slowing, rising and billowing out. Each is drawn as a soft disc turned
to face the viewer, fading as it thins.

It is purely visual - nothing in the game reacts to smoke - so it lives with the renderer
and advances with the frame's ``dt`` (a paused game passes 0: the smoke hangs still).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from cyberduke.animation.reach import hand as hand_position
from cyberduke.animation.street import MOUTH
from cyberduke.render.context import DrawContext
from cyberduke.render.items import CHERRY_AT, held_transform

Vec3 = tuple[float, float, float]

TRAIL_RATE = 14.0  # particles a second off a smouldering tip
PUFF_RATE = 55.0  # ... out of the mouth at the height of an exhale
TRAIL_LIFE, PUFF_LIFE = 2.6, 2.3  # seconds
MAX_PARTICLES = 600
SMOKE_RGB = (0.7, 0.72, 0.76)  # blue-grey
TRAIL_ALPHA, PUFF_ALPHA = 0.3, 0.42
GLOW_RGB = (1.0, 0.42, 0.1)  # the halo round a cigarette's lit end


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    life: float
    size0: float
    size1: float
    alpha: float
    phase: float
    age: float = 0.0
    kind: str = "trail"  # or "puff"

    @property
    def size(self) -> float:
        u = min(1.0, self.age / self.life)
        return self.size0 + (self.size1 - self.size0) * math.sqrt(u)

    @property
    def opacity(self) -> float:
        u = min(1.0, self.age / self.life)
        fade_in = min(1.0, self.age / 0.15)
        return self.alpha * fade_in * (1.0 - u) ** 1.5


# ----------------------------------------------------------------------------- where
def _root(npc) -> tuple[float, float, float, float, float]:
    scale = getattr(getattr(npc, "spec", None), "height", 1.0) or 1.0
    return npc.x, npc.y, npc.z, npc.yaw, scale


def _to_world(npc, bx: float, by: float, bz: float) -> Vec3:
    """A point of the npc's drawn (body) frame in the world - as ``render.humanoid``
    places it: translated, turned by its yaw, scaled by its height."""
    x, y, z, yaw, k = _root(npc)
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return (x + (bx * c + bz * s) * k, y + by * k, z + (-bx * s + bz * c) * k)


def _dir_to_world(npc, dx: float, dy: float, dz: float) -> Vec3:
    c, s = math.cos(math.radians(npc.yaw)), math.sin(math.radians(npc.yaw))
    return (dx * c + dz * s, dy, -dx * s + dz * c)


def cigarette_side(npc) -> str | None:
    """Which hand holds a cigarette (None: neither)."""
    pose = getattr(npc, "pose", None)
    if pose is None:
        return None
    if pose.hand_r == "cigarette":
        return "r"
    if pose.hand_l == "cigarette":
        return "l"
    return None


def cigarette_tip(npc) -> Vec3 | None:
    """World position of the lit end of the cigarette ``npc`` holds (None: none held, or
    not standing - lying on the floor, say)."""
    side = cigarette_side(npc)
    if side is None or getattr(npc, "body", None) is not None:
        return None
    pose = npc.pose
    pos, fore = hand_position(pose, side)
    (x, y, z), yaw, pitch = held_transform(
        "cigarette",
        tuple(pos),
        fore,
        aim=pose.aim_l if side == "l" else pose.aim_r,
        tilt=pose.tilt_l if side == "l" else pose.tilt_r,
        side=side,
    )
    # the item's frame: turned by the yaw about +y, then the pitch about +x
    px, py, pz = CHERRY_AT
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    py, pz = py * cp - pz * sp, py * sp + pz * cp
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    px, pz = px * cy + pz * sy, -px * sy + pz * cy
    return _to_world(npc, x + px, y + py, z + pz)


def mouth(npc) -> tuple[Vec3, Vec3]:
    """(world position of the mouth, the way the face points - a little down)."""
    pose = npc.pose
    mx, my, mz = MOUTH
    yaw = math.radians(pose.head_yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    bx, bz = mx * c + mz * s, -mx * s + mz * c
    at = _to_world(npc, bx, my + pose.bob, bz)
    pitch = math.radians(pose.head_pitch)  # positive: head tipped down
    fwd = (s * math.cos(pitch), -math.sin(pitch) - 0.15, c * math.cos(pitch))
    return at, _dir_to_world(npc, *fwd)


# ----------------------------------------------------------------------------- the smoke
class SmokeRenderer:
    """Emits, moves and draws the smoke of everybody smoking."""

    def __init__(self, seed: int = 5) -> None:
        self.particles: list[Particle] = []
        self.embers: list[tuple[Vec3, float]] = []  # (tip, ember) of every lit cigarette
        self.rng = random.Random(seed)
        self._debt: dict[tuple[int, str], float] = {}  # fractional particles owed

    def clear(self) -> None:
        self.particles.clear()
        self._debt.clear()

    def _owed(self, key: tuple[int, str], rate: float, dt: float) -> int:
        debt = self._debt.get(key, 0.0) + rate * dt
        n = int(debt)
        self._debt[key] = debt - n
        return n

    def update(self, npcs, dt: float) -> None:
        self.embers = []
        for npc in npcs:
            if getattr(npc, "hidden", False) or cigarette_side(npc) is None:
                continue
            tip = cigarette_tip(npc)
            if tip is not None:
                self.embers.append((tip, npc.pose.ember))
        if dt <= 0.0:
            return  # (paused: the smoke hangs where it is)
        for npc in npcs:
            if getattr(npc, "hidden", False) or cigarette_side(npc) is None:
                continue
            pose = npc.pose
            tip = cigarette_tip(npc)
            if tip is not None and not pose.inhaling:
                for _ in range(self._owed((id(npc), "trail"), TRAIL_RATE, dt)):
                    self._trail(tip)
            if pose.exhale > 0.02:
                at, fwd = mouth(npc)
                for _ in range(self._owed((id(npc), "puff"), PUFF_RATE * pose.exhale, dt)):
                    self._puff(at, fwd, pose.exhale)
        alive = []
        for p in self.particles:
            p.age += dt
            if p.age >= p.life:
                continue
            if p.kind == "puff":
                drag = math.exp(-1.8 * dt)
                p.vx, p.vz = p.vx * drag, p.vz * drag
                p.vy = p.vy * drag + 0.16 * dt  # warm: it rises as it slows
            else:
                p.vy += 0.02 * dt
            sway = 0.05 * math.sin(p.age * 2.6 + p.phase)
            p.x += (p.vx + sway) * dt
            p.y += p.vy * dt
            p.z += (p.vz + 0.6 * sway) * dt
            alive.append(p)
        self.particles = alive[-MAX_PARTICLES:]

    def _trail(self, tip: Vec3) -> None:
        r = self.rng
        self.particles.append(
            Particle(
                tip[0] + r.uniform(-0.004, 0.004),
                tip[1] + 0.005,
                tip[2] + r.uniform(-0.004, 0.004),
                r.uniform(-0.015, 0.015), r.uniform(0.09, 0.15), r.uniform(-0.015, 0.015),
                TRAIL_LIFE * r.uniform(0.8, 1.1), 0.012, r.uniform(0.08, 0.13),
                TRAIL_ALPHA, r.uniform(0.0, 6.3), kind="trail",
            )
        )  # fmt: skip

    def _puff(self, at: Vec3, fwd: Vec3, strength: float) -> None:
        r = self.rng
        speed = (0.45 + 0.35 * strength) * r.uniform(0.8, 1.2)
        spread = 0.18
        self.particles.append(
            Particle(
                at[0] + fwd[0] * 0.03, at[1], at[2] + fwd[2] * 0.03,
                fwd[0] * speed + r.uniform(-spread, spread) * speed,
                fwd[1] * speed + r.uniform(-spread, spread) * speed * 0.6,
                fwd[2] * speed + r.uniform(-spread, spread) * speed,
                PUFF_LIFE * r.uniform(0.8, 1.15), 0.04, r.uniform(0.28, 0.42),
                PUFF_ALPHA, r.uniform(0.0, 6.3), kind="puff",
            )
        )  # fmt: skip

    @staticmethod
    def _facing(ctx: DrawContext, at: Vec3, eye: Vec3, size: float, rgba, inner: float) -> None:
        """A soft disc of radius ``size`` at ``at``, turned to face ``eye``."""
        dx, dy, dz = eye[0] - at[0], eye[1] - at[1], eye[2] - at[2]
        n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        dx, dy, dz = dx / n, dy / n, dz / n
        turn = ((math.degrees(math.atan2(dx, dz)), 0.0, 1.0, 0.0),
                (math.degrees(math.acos(max(-1.0, min(1.0, dy)))), 1.0, 0.0, 0.0))  # fmt: skip
        with ctx.transformed(at, rotate=turn):
            ctx.scale(size, 1.0, size)
            ctx.soft_disc(rgba, slices=12, inner=inner)

    def draw(self, ctx: DrawContext, eye: Vec3) -> None:
        """Each particle a soft disc turned to face ``eye``, furthest first, without
        writing depth (see-through, like the fog) - and a faint halo round every lit
        cigarette's end: dull red, flaring orange on a drag."""
        if not self.particles and not self.embers:
            return
        ex, ey, ez = eye
        order = sorted(
            self.particles, key=lambda p: -((p.x - ex) ** 2 + (p.y - ey) ** 2 + (p.z - ez) ** 2)
        )
        ctx.depth_mask(False)
        for tip, ember in self.embers:
            size = 0.03 + 0.07 * ember
            self._facing(ctx, tip, eye, size, (*GLOW_RGB, 0.35 + 0.6 * ember), 0.2)
        for p in order:
            a = p.opacity
            if a >= 0.004:
                self._facing(ctx, (p.x, p.y, p.z), eye, p.size, (*SMOKE_RGB, a), 0.1)
        ctx.depth_mask(True)
        ctx.alpha(1.0)
        ctx.color(1.0, 1.0, 1.0)


__all__ = ["SmokeRenderer", "cigarette_tip", "mouth"]
