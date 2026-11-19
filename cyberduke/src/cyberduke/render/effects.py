"""World-space visual effects drawn with unlit lines (no new backend API needed)."""

from __future__ import annotations

import math

from cyberduke.render.context import DrawContext
from cyberduke.systems.scanner import ScanPulse


def draw_scan_pulse(ctx: DrawContext, pulse: ScanPulse | None, segments: int = 48) -> None:
    """The scanner ping: a ring expanding along the floor from the pulse origin, fading as
    it reaches the scan range, plus a faint vertical "wall" so it reads in corridors."""
    if pulse is None or pulse.done:
        return
    ox, _oy, oz = pulse.origin
    r = pulse.radius
    if r < 0.05:
        return
    fade = 1.0 - pulse.age / pulse.duration
    ring = []
    wall = []
    prev = None
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        p = (ox + math.cos(a) * r, 0.03, oz + math.sin(a) * r)
        if prev is not None:
            ring.append((prev, p))
            if i % 4 == 0:
                wall.append((p, (p[0], 0.6 * fade + 0.1, p[2])))
        prev = p
    ctx.lines(ring, (0.0, 1.0, 1.0, 0.75 * fade), width=2.0)
    ctx.lines(wall, (0.0, 0.9, 1.0, 0.35 * fade))


def draw_sparks(
    ctx: DrawContext,
    x: float,
    y: float,
    z: float,
    t: float,
    radius: float = 0.45,
    height: float = 1.4,
    count: int = 8,
    seed: float = 0.0,
) -> None:
    """Crackling electricity around a point: short jagged lines that jump every frame."""
    segs = []
    for i in range(count):
        phase = t * 23.0 + i * 1.7 + seed
        a = phase * 1.3
        y0 = y + (math.sin(phase * 0.7 + i) * 0.5 + 0.5) * height
        x0 = x + math.cos(a) * radius
        z0 = z + math.sin(a) * radius
        x1 = x0 + math.cos(a + 2.0) * 0.25
        y1 = y0 + math.sin(phase * 2.1) * 0.2
        z1 = z0 + math.sin(a + 2.0) * 0.25
        segs.append(((x0, y0, z0), (x1, y1, z1)))
    flicker = 0.6 + 0.4 * math.sin(t * 60.0)
    ctx.lines(segs, (0.6, 0.9, 1.0, flicker), width=2.0)
