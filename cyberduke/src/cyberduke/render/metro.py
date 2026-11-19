"""Draws the metro (``entities.metro``): the train where it is on the line - each car
baked once (``world.vehicles.train_car``) and drawn as one batch - its door leaves
sliding apart as the doors open, and the station's ticket gates, their glass flaps
folding back for whoever has a pass (the lane's light green while it is open)."""

from __future__ import annotations

import math
from typing import Any

from cyberduke.entities.metro import Metro
from cyberduke.render.context import DrawContext, Vec3
from cyberduke.render.props import draw_prop
from cyberduke.render.sectors import is_translucent
from cyberduke.render.textures import TextureMap
from cyberduke.render.traffic import _bake
from cyberduke.world.vehicles import (
    CAR_HALF_WIDTH,
    CAR_LENGTH,
    DOOR_HEIGHT,
    DOOR_WIDTH,
    TRAIN_CARS,
    TRAIN_WHITE,
    door_xs,
    train_car,
)

LEAF = TRAIN_WHITE
LEAF_WINDOW = (0.55, 0.62, 0.66)
GATE_GLASS = (0.55, 0.7, 0.8)


class MetroRenderer:
    def __init__(self) -> None:
        self._cars: dict[float, Any] = {}

    def clear(self) -> None:
        self._cars.clear()

    def _car(self, ctx, cab: float, textures):
        if cab not in self._cars:
            self._cars[cab] = _bake(ctx, train_car(cab), textures)
        return self._cars[cab]

    def draw(
        self,
        ctx: DrawContext,
        metro: Metro,
        textures: TextureMap,
        eye: Vec3 | None = None,
        view_distance: float = math.inf,
    ) -> None:
        d = metro.data
        self._gates(ctx, metro)
        if not metro.on_the_line:
            return
        batches = getattr(ctx, "batches", False)
        y, z = d.floor_y, d.track_z
        for cab, middle in zip(TRAIN_CARS, metro.car_xs(), strict=True):
            if eye is not None:
                reach = view_distance + CAR_LENGTH
                if (middle - eye[0]) ** 2 + (z - eye[2]) ** 2 > reach * reach:
                    continue
            ctx.push()
            ctx.translate(middle, y, z)
            if batches:
                car = self._car(ctx, cab, textures)
                if car is not None:
                    ctx.draw_batch(*car)
            else:
                for p in train_car(cab):
                    if not is_translucent(p):
                        draw_prop(ctx, p, textures)
            self._doors(ctx, metro.doors)
            ctx.pop()
        ctx.emission(None)
        ctx.color(1.0, 1.0, 1.0)

    def _doors(self, ctx: DrawContext, amount: float) -> None:
        """A car's door leaves on its platform side (-z): two to a door, sliding apart
        along the outside of the car as the doors open, a window in each."""
        z = -CAR_HALF_WIDTH - 0.025
        half = DOOR_WIDTH / 2.0
        slide = amount * (half - 0.04)
        for m in door_xs(0.0):
            for side in (-1.0, 1.0):
                cx = m + side * (half / 2.0 + slide)
                ctx.color(*LEAF.color)
                ctx.specular(LEAF.specular, LEAF.shininess)
                ctx.box(cx, DOOR_HEIGHT / 2.0, z, half - 0.01, DOOR_HEIGHT - 0.01, 0.03)
                ctx.specular(None)
                ctx.color(*LEAF_WINDOW)
                ctx.emission((0.42, 0.5, 0.55))
                ctx.box(cx, 1.45, z - 0.017, half - 0.2, 0.8, 0.004)
                ctx.emission(None)

    def _gates(self, ctx: DrawContext, metro: Metro) -> None:
        """The ticket gates' flaps: two glass panels across each lane, folding back into
        the cabinets either side as the gate opens; the lane's light green while open."""
        d = metro.data
        if not d.gates:
            return
        z = (d.gates_z[0] + d.gates_z[1]) / 2.0
        for lane, x in enumerate(d.gates):
            k = metro.gate(lane)
            for side in (-1.0, 1.0):
                width = 0.38 * (1.0 - k) + 0.02
                cx = x + side * (0.47 - width / 2.0)
                ctx.color(*GATE_GLASS)
                ctx.alpha(0.45)
                ctx.box(cx, 0.7, z, width, 0.6, 0.02)
                ctx.alpha(1.0)
            if k > 0.0:  # (a green lamp on the cabinet at the lane's right)
                ctx.color(0.2, 1.0, 0.35)
                ctx.emission((0.3, 2.0, 0.5))
                ctx.box(x + 0.55, 1.095, z, 0.1, 0.03, 0.1)
                ctx.emission(None)
        ctx.color(1.0, 1.0, 1.0)


__all__ = ["MetroRenderer"]
