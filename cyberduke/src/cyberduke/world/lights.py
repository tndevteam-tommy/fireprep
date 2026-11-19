"""Light-source records. One record drives BOTH the hardware GL light and the visible
fixture mesh, so the two can no longer drift apart."""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Light:
    x: float
    y: float
    z: float
    ambient: RGB
    diffuse: RGB
    specular: RGB
    linear_attenuation: float = 0.02
    quadratic_attenuation: float = 0.005
    # Visible fixture
    fixture: str = "none"  # "hanging_cage" | "caution_lamp" | "halo" | "flood_bar" | "none"
    fixture_emission: RGB = (0.0, 0.0, 0.0)  # GL_EMISSION of the tube/panel
    fixture_color: RGB = (1.0, 1.0, 1.0)  # glColor of the emissive tube/panel
    fixture_texture: str = "neon_cyan"  # texture style name for the tube/panel
    fixture_offset_y: float = 0.0  # fixture mesh offset from the light position
    ceiling_y: float = 4.82  # where a hanging fixture's cables attach
    # A lamp that is switched off in the day (a street lamp): outdoors, the renderer dims
    # it with the daylight (``render.scene``).
    night_only: bool = False

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)
