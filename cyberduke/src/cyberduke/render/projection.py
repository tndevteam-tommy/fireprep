"""World-to-screen projection for HUD overlays (scanner markers, labels).

Pure NumPy: builds the same perspective/view matrices the modern backend uses and maps
world points to overlay pixels (origin top-left, y down), so markers line up on both
backends without reading anything back from OpenGL.
"""

from __future__ import annotations

import numpy as np

from cyberduke.render.backends.modern import look_at, perspective

Vec3 = tuple[float, float, float]


class Projector:
    def __init__(
        self,
        width: int,
        height: int,
        fov_y_deg: float,
        near: float,
        far: float,
        eye: Vec3,
        target: Vec3,
    ) -> None:
        self.width, self.height = width, height
        self.view_proj = perspective(fov_y_deg, width / height, near, far) @ look_at(eye, target)

    def project(self, x: float, y: float, z: float) -> tuple[float, float, float] | None:
        """Screen position (px, py) and normalised depth for a world point, or ``None`` when
        the point is behind the viewer. Points outside the viewport are still returned (the
        caller decides whether to clamp them to the edge)."""
        clip = self.view_proj @ np.array((x, y, z, 1.0), dtype=np.float32)
        w = float(clip[3])
        if w <= 1e-5:
            return None
        ndc_x, ndc_y, ndc_z = (float(c) / w for c in clip[:3])
        return (
            (ndc_x * 0.5 + 0.5) * self.width,
            (1.0 - (ndc_y * 0.5 + 0.5)) * self.height,
            ndc_z,
        )

    def on_screen(self, px: float, py: float, margin: float = 0.0) -> bool:
        return -margin <= px <= self.width + margin and -margin <= py <= self.height + margin
