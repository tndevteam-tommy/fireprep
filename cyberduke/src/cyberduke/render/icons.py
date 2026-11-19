"""The inventory icons on the GPU: generated once (``assets.icons``), uploaded on first
use, kept by name."""

from __future__ import annotations

import numpy as np

from cyberduke.assets.icons import ICON, generate_icon
from cyberduke.render.context import DrawContext


class IconCache:
    def __init__(self, ctx: DrawContext) -> None:
        self.ctx = ctx
        self._tex: dict[str, int] = {}

    def get(self, name: str) -> int:
        tex = self._tex.get(name)
        if tex is None:
            img = np.flipud(generate_icon(name))  # overlay images have row 0 at the bottom
            tex = self._tex[name] = self.ctx.upload_texture_rgba(
                np.ascontiguousarray(img).tobytes(), ICON, ICON
            )
        return tex
