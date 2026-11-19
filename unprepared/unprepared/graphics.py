"""User-adjustable graphics settings with High/Medium/Low presets.

Unlike :mod:`unprepared.config` (frozen tuning constants) these are *mutable*
and persisted to ``graphics.json`` next to the save file (in the per-user data
directory, see :mod:`unprepared.paths`). The renderer reads them every frame,
so the options menu can change them live.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Dict, Optional

from .paths import GRAPHICS_FILE_NAME, write_json

# "high" is meant for a proper graphics card: the whole realm in view (the draw distance reaches
# across the map and the day's haze thins to match), every torch lit, 8x multisampling and 16x
# anisotropic filtering. "medium" is the old "high"; "low" the cheapest that still plays.
PRESETS: Dict[str, dict] = {
    "high": dict(shaders=True, mipmaps=True, stars=True, sky_bodies=True, star_count=1500, ground_cells=40,
                 object_fade=1.0, character_fade=1.0, torch_lights=16, roofs=True, draw_distance=700.0, haze=0.3,
                 antialiasing=8, anisotropy=16),
    "medium": dict(shaders=True, mipmaps=True, stars=True, sky_bodies=True, star_count=400, ground_cells=20,
                   object_fade=0.8, character_fade=0.8, torch_lights=8, roofs=True, draw_distance=160.0, haze=0.85,
                   antialiasing=2, anisotropy=4),
    "low": dict(shaders=False, mipmaps=False, stars=False, sky_bodies=True, star_count=0, ground_cells=4,
                object_fade=0.45, character_fade=0.6, torch_lights=2, roofs=True, draw_distance=100.0, haze=1.0,
                antialiasing=0, anisotropy=1),
}
DRAW_DISTANCE_RANGE = (60.0, 1000.0)       # world units (the map is some 375 x 350)
AA_STEPS = (0, 2, 4, 8)                    # multisampling: samples per pixel (0: off)
ANISOTROPY_STEPS = (1, 2, 4, 8, 16)        # anisotropic texture filtering (1: off)
MAX_SHADER_LIGHTS = 16                     # the most torches and lanterns the lighting shader is built for
QUALITY_ORDER = ("low", "medium", "high")


@dataclass
class GraphicsSettings:
    """The fields' defaults are the "medium" preset (what a machine that has not been measured
    yet gets); the start-up benchmark picks the preset that suits it."""
    quality: str = "medium"
    auto_quality: bool = True           # pick the preset from a start-up benchmark
    fov: float = 60.0                   # vertical field of view, degrees
    object_fade: float = 0.8            # fraction of the far plane beyond which world objects are not drawn
    character_fade: float = 0.8         # same for NPCs
    contrast: float = 1.0
    shaders: bool = True
    mipmaps: bool = True
    stars: bool = True
    sky_bodies: bool = True
    star_count: int = 400
    ground_cells: int = 20
    torch_lights: int = 8
    roofs: bool = True
    draw_distance: float = 160.0        # the far plane, world units (``object_fade`` / ``character_fade`` are fractions of it)
    haze: float = 0.85                  # the day's haze, as a share of the weather's (less haze: further to see)
    antialiasing: int = 2               # multisampling samples (0 off; applied when the window is opened)
    anisotropy: int = 4                 # anisotropic texture filtering (1 off)
    fullscreen: bool = False            # the whole screen at the desktop's resolution (applied at once)
    vsync: bool = False                 # wait for the screen's refresh (applied with the window)
    show_perf_overlay: bool = False
    detected_renderer: str = ""
    benchmark_ms: float = 0.0
    _version: int = field(default=1, repr=False)

    # --- presets ---------------------------------------------------------------
    def apply_preset(self, name: str) -> None:
        for key, value in PRESETS[name].items():
            setattr(self, key, value)
        self.quality = name

    def next_quality(self, step: int = 1) -> str:
        current = QUALITY_ORDER.index(self.quality) if self.quality in QUALITY_ORDER else QUALITY_ORDER.index("high")
        idx = (current + step) % len(QUALITY_ORDER)
        self.apply_preset(QUALITY_ORDER[idx])
        self.auto_quality = False
        return self.quality

    # --- derived -----------------------------------------------------------------
    @property
    def far_plane(self) -> float:
        """The draw distance, kept within ``DRAW_DISTANCE_RANGE``."""
        lo, hi = DRAW_DISTANCE_RANGE
        return max(lo, min(hi, float(self.draw_distance)))

    def object_draw_distance(self, far_plane: float) -> float:
        return far_plane * max(0.1, min(1.0, self.object_fade))

    def character_draw_distance(self, far_plane: float) -> float:
        return far_plane * max(0.1, min(1.0, self.character_fade))

    # --- persistence -------------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, data: dict) -> "GraphicsSettings":
        """Settings from a saved dict. Unknown keys are ignored, and so is a value of the wrong
        type for its field (``"fov": "wide"``, a fractional star count) - the default stands in
        for it; a preset name this version does not know (a hand edit, a newer version's file)
        falls back to the default preset's name."""
        defaults = cls()
        clean = {}
        for f in fields(cls):
            if f.name.startswith("_") or f.name not in data:
                continue
            value, default = data[f.name], getattr(defaults, f.name)
            if isinstance(default, bool):
                ok = isinstance(value, bool)
            elif isinstance(default, int):
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, float):
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
                value = float(value) if ok else value
            else:
                ok = isinstance(value, type(default))
            if ok:
                clean[f.name] = value
        settings = cls(**clean)
        if settings.quality not in PRESETS:
            settings.quality = defaults.quality
        return settings

    def update_from_dict(self, data: dict) -> "GraphicsSettings":
        """Overwrite this object's fields in place (unknown keys ignored). Used to restore a
        snapshot after a benchmark so every reference to the settings - the renderer, the
        open graphics menu - keeps seeing the same object."""
        known = {f.name for f in fields(self)}
        for key, value in data.items():
            if key in known and not key.startswith("_"):
                setattr(self, key, value)
        return self

    def save(self, path: str) -> bool:
        """Persist the settings (atomically: a crash never leaves a half-written file). Returns
        False instead of raising when the file cannot be written (a read-only or full disk): the
        game goes on with the settings it has, it just cannot remember them."""
        try:
            write_json(path, self.to_dict(), indent=2)
        except OSError:
            return False
        return True

    @classmethod
    def load(cls, path: str) -> Optional["GraphicsSettings"]:
        """The settings saved at ``path``, or None when there are none or the file is not usable
        (not JSON, or JSON that is not an object of settings) - the caller then starts afresh."""
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return cls.from_dict(data)
        except (OSError, ValueError, TypeError):
            return None


def graphics_settings_path(save_file: str) -> str:
    """``graphics.json`` beside the given save file."""
    return os.path.join(os.path.dirname(os.path.abspath(save_file)), GRAPHICS_FILE_NAME)
