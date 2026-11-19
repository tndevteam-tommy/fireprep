"""Scalar interpolation and angle helpers shared by animation, AI, and entities."""

from __future__ import annotations


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def wrap_angle_deg(angle: float) -> float:
    """Wraps an angle in degrees into the half-open range [-180, 180)."""
    return ((angle + 180.0) % 360.0) - 180.0
