"""Graphics quality presets and GPU detection.

Pure data and string matching - no OpenGL here, so the presets and the detection rules can
be unit-tested headlessly. ``scene.SceneRenderer`` reads the GL strings once and asks
``detect_quality`` what the machine deserves; the pause menu can override it at any time
and the renderer re-applies the preset live (view distance, light budget, ambient light and
the ambient detail effects).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

RGB = tuple[float, float, float]


class Quality(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @classmethod
    def parse(cls, name: str) -> Quality:
        return cls(name.strip().upper())


@dataclass(frozen=True)
class GraphicsSettings:
    """Everything a quality preset decides. One record, consumed by the renderer."""

    quality: Quality
    view_distance: float  # far plane, metres: how much of the facility is drawn at all
    max_lights: int  # hardware lights; the nearest ones to the eye win
    global_ambient: RGB  # base light level of the whole scene
    effect_distance: float  # how far bioluminescent pulsing props are animated
    sparks: bool  # electrical arcs on fried computers, shorted machines, electrocutions
    scan_ring: bool  # the expanding scanner pulse on the floor
    beam_slices: int  # rays per security-camera laser fan
    hud_scanlines: bool  # camera-feed scanline overlay
    prefer_modern_backend: bool  # per-pixel lighting when the driver can take it
    rain_drops: int = 900  # streaks of rain round the eye in the heaviest rain

    @property
    def name(self) -> str:
        return self.quality.value


PRESETS: dict[Quality, GraphicsSettings] = {
    Quality.LOW: GraphicsSettings(
        quality=Quality.LOW,
        view_distance=30.0,
        max_lights=2,
        global_ambient=(0.27, 0.32, 0.42),  # flatter, but nothing is left pitch black
        effect_distance=12.0,
        sparks=False,
        scan_ring=False,
        beam_slices=4,
        hud_scanlines=False,
        prefer_modern_backend=False,
        rain_drops=350,
    ),
    Quality.MEDIUM: GraphicsSettings(
        quality=Quality.MEDIUM,
        view_distance=45.0,
        max_lights=5,
        global_ambient=(0.25, 0.31, 0.43),
        effect_distance=26.0,
        sparks=True,
        scan_ring=True,
        beam_slices=8,
        hud_scanlines=True,
        prefer_modern_backend=True,
        rain_drops=900,
    ),
    Quality.HIGH: GraphicsSettings(
        quality=Quality.HIGH,
        view_distance=75.0,
        max_lights=8,
        global_ambient=(0.25, 0.31, 0.44),
        effect_distance=48.0,
        sparks=True,
        scan_ring=True,
        beam_slices=12,
        hud_scanlines=True,
        prefer_modern_backend=True,
        rain_drops=1600,
    ),
}

QUALITY_ORDER: tuple[Quality, ...] = (Quality.LOW, Quality.MEDIUM, Quality.HIGH)

# Substrings of GL_RENDERER / GL_VENDOR, most specific first.
SOFTWARE_MARKERS = ("llvmpipe", "softpipe", "swrast", "software", "lavapipe", "gallium,")
DISCRETE_MARKERS = (
    "geforce", "rtx", "gtx", "quadro", "titan", "radeon rx", "radeon pro", "firepro",
    "navi", "arc a", "apple m1", "apple m2", "apple m3", "apple m4",
)  # fmt: skip
INTEGRATED_MARKERS = ("intel", "uhd graphics", "hd graphics", "iris", "vega", "adreno", "mali")


def settings_for(quality: Quality | str) -> GraphicsSettings:
    if isinstance(quality, str):
        quality = Quality.parse(quality)
    return PRESETS[quality]


def detect_quality(renderer: str = "", vendor: str = "") -> Quality:
    """Picks a preset from the GL renderer/vendor strings: software rasterisers get LOW,
    recognised discrete GPUs get HIGH, everything else (integrated, unknown) MEDIUM."""
    text = f"{renderer} {vendor}".lower()
    if any(m in text for m in SOFTWARE_MARKERS):
        return Quality.LOW
    if any(m in text for m in DISCRETE_MARKERS):
        return Quality.HIGH
    if any(m in text for m in INTEGRATED_MARKERS):
        return Quality.MEDIUM
    return Quality.MEDIUM


def cycle(quality: Quality, delta: int = 1) -> Quality:
    """Next/previous preset, wrapping (drives the pause menu's graphics row)."""
    index = (QUALITY_ORDER.index(quality) + delta) % len(QUALITY_ORDER)
    return QUALITY_ORDER[index]


def lights_for(settings: GraphicsSettings, lights: Sequence, eye: tuple[float, float, float]):
    """The ``max_lights`` lights nearest the eye, in the level's own order (so the set is
    stable and the scene does not flicker when two lights swap distance rank)."""
    if len(lights) <= settings.max_lights:
        return list(lights)
    ranked = sorted(
        range(len(lights)),
        key=lambda i: math.dist((lights[i].x, lights[i].y, lights[i].z), eye),
    )
    keep = sorted(ranked[: settings.max_lights])
    return [lights[i] for i in keep]
