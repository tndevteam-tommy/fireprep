"""Draws the traffic (``entities.traffic``): each vehicle where it is on the road, its
wheels turning, its brake lights on when it brakes - and, in the late translucent pass,
the light it throws on the wet road: a pool ahead of the headlights, a red glow behind,
the underglow under a coupe.

A vehicle's body and each of its wheels are baked once per model and paint job into a
batch of triangles in the vehicle's own frame (``render.batch``: the parts of
``world.vehicles``); a frame then draws each as one ``draw_batch`` under the vehicle's
transform (and each wheel's spin). A context that cannot draw batches (a test's recorder)
is given the parts one by one instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from cyberduke.entities.traffic import Car
from cyberduke.render.batch import BatchContext
from cyberduke.render.context import DrawContext, Vec3
from cyberduke.render.props import draw_prop
from cyberduke.render.sectors import is_translucent
from cyberduke.render.textures import TextureMap
from cyberduke.world.props import Prop
from cyberduke.world.vehicles import Paint, VehicleModel

GLOW_LIFT = 0.012  # the light pools lie this far over the road (over its markings)
HEADLIGHT_POOL = (0.85, 0.92, 1.0, 0.16)
TAIL_GLOW = (1.0, 0.1, 0.12, 0.22)
BRAKE = (3.6, 0.25, 0.3)  # the tail lights' glow, braking


class _Capture:
    """What a ``BatchContext`` flushes, kept instead of drawn."""

    def __init__(self, texture_tiles) -> None:
        self.texture_tiles = texture_tiles
        self.batch: tuple[Any, Any, list] | None = None

    def draw_batch(self, vertices, indices, ranges) -> None:
        self.batch = (vertices.copy(), indices.copy(), list(ranges))


def _bake(ctx: DrawContext, props: Iterable[Prop], textures: TextureMap):
    capture = _Capture(getattr(ctx, "texture_tiles", {}))
    batch = BatchContext(capture)  # type: ignore[arg-type]
    for p in props:
        if not is_translucent(p):
            draw_prop(batch, p, textures)
    batch.flush()
    return capture.batch


class TrafficRenderer:
    def __init__(self) -> None:
        self._baked: dict[tuple, Any] = {}

    def clear(self) -> None:
        self._baked.clear()

    def _body(self, ctx, model: VehicleModel, paint: Paint, textures):
        key = ("body", model.name, paint)
        if key not in self._baked:
            self._baked[key] = _bake(ctx, model.body(paint), textures)
        return self._baked[key]

    def _wheel(self, ctx, model: VehicleModel, paint: Paint, side: float, textures):
        key = ("wheel", model.name, paint, side)
        if key not in self._baked:
            self._baked[key] = _bake(ctx, model.wheel(paint, side), textures)
        return self._baked[key]

    # ------------------------------------------------------------------ the vehicles
    def draw(
        self,
        ctx: DrawContext,
        cars: Iterable[Car],
        textures: TextureMap,
        eye: Vec3 | None = None,
        view_distance: float = math.inf,
    ) -> None:
        batches = getattr(ctx, "batches", False)
        for car in cars:
            if not _in_view(car, eye, view_distance):
                continue
            model, paint = car.model, car.paint
            ctx.push()
            ctx.translate(car.x, car.y, car.z)
            ctx.rotate(car.yaw, 0, 1, 0)
            if batches:
                body = self._body(ctx, model, paint, textures)
                if body is not None:
                    ctx.draw_batch(*body)
            else:
                for p in model.body(paint):
                    if not is_translucent(p):
                        draw_prop(ctx, p, textures)
            for x, z, side in model.wheels():
                ctx.push()
                ctx.translate(x, model.wheel_radius, z)
                ctx.rotate(car.spin, 0, 0, 1)  # (rolling towards the nose: -x)
                if batches:
                    wheel = self._wheel(ctx, model, paint, side, textures)
                    if wheel is not None:
                        ctx.draw_batch(*wheel)
                else:
                    for p in model.wheel(paint, side):
                        draw_prop(ctx, p, textures)
                ctx.pop()
            if car.braking:  # the brake lights: the tail lights flaring up
                tx, ty = model.tail
                ctx.color(1.0, 0.15, 0.2)
                ctx.emission(BRAKE)
                ctx.box(tx + 0.018, ty, 0.0, 0.012, 0.09, model.width * 0.82)
                ctx.emission(None)
                ctx.color(1.0, 1.0, 1.0)
            ctx.pop()

    # ------------------------------------------------------------------ the light they throw
    def draw_glow(
        self,
        ctx: DrawContext,
        cars: Iterable[Car],
        eye: Vec3 | None = None,
        view_distance: float = math.inf,
    ) -> None:
        """The pools of light on the road: translucent, drawn without writing depth."""
        ctx.depth_mask(False)
        for car in cars:
            if not _in_view(car, eye, view_distance):
                continue
            model = car.model
            ctx.push()
            ctx.translate(car.x, car.y + GLOW_LIFT, car.z)
            ctx.rotate(car.yaw, 0, 1, 0)
            hx = model.headlights[0]
            with ctx.transformed((hx - 4.0, 0.0, 0.0), scale=(4.2, 1.0, 1.5)):
                ctx.soft_disc(HEADLIGHT_POOL, slices=20, inner=0.15)
            tail = (TAIL_GLOW[0], TAIL_GLOW[1], TAIL_GLOW[2], 0.38 if car.braking else 0.2)
            with ctx.transformed((model.tail[0] + 0.9, 0.0, 0.0), scale=(1.2, 1.0, 1.1)):
                ctx.soft_disc(tail, slices=16, inner=0.1)
            if model.underglow:
                with ctx.transformed(scale=(model.length / 2.0, 1.0, model.width / 1.55)):
                    ctx.soft_disc((*car.paint.glow, 0.36), slices=20, inner=0.45)
            ctx.pop()
        ctx.depth_mask(True)
        ctx.alpha(1.0)
        ctx.color(1.0, 1.0, 1.0)


def _in_view(car: Car, eye: Vec3 | None, view_distance: float) -> bool:
    if eye is None:
        return True
    reach = view_distance + car.model.length
    return (car.x - eye[0]) ** 2 + (car.z - eye[2]) ** 2 <= reach * reach


__all__ = ["TrafficRenderer"]
