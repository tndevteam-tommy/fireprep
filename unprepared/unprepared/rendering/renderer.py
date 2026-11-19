"""Top-level renderer: GL context setup plus the per-frame 3D + HUD passes."""
from __future__ import annotations

from typing import Optional

import pygame
from OpenGL.GL import (GL_AMBIENT_AND_DIFFUSE, GL_COLOR_MATERIAL, GL_DEPTH_TEST, GL_FRONT_AND_BACK,
                       GL_LIGHTING, GL_MODELVIEW, GL_MODULATE, GL_PROJECTION, GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,
                       glColorMaterial, glEnable, glLoadIdentity, glMatrixMode, glTexEnvi, glViewport)

from ..config import GameSettings
from ..graphics import GraphicsSettings
from .camera import apply_perspective
from .hud import HudRenderer
from .lighting import LightingRig
from .scene import SceneRenderer
from .shaders import try_create_lighting_shader
from .text import TextRenderer
from .textures import TextureRegistry


class Renderer:
    """Owns all GL resources. Create *after* the OpenGL window exists."""

    def __init__(self, settings: GameSettings, controls_help: list, use_shaders: bool = True,
                 graphics: Optional[GraphicsSettings] = None):
        self.settings = settings
        self.graphics = graphics or GraphicsSettings(shaders=use_shaders)
        use_shaders = use_shaders and self.graphics.shaders
        w = settings.window
        self.font = pygame.font.SysFont(w.font_name, w.font_size, bold=True)
        self.font_small = pygame.font.SysFont(w.font_name, w.font_small_size)
        self.font_tag = pygame.font.SysFont(w.tag_font_name, w.font_small_size, bold=True)

        self.init_gl()
        self.shader = try_create_lighting_shader(self.graphics.torch_lights) if use_shaders else None

        self.textures = TextureRegistry(settings.asset_dir, mipmaps=self.graphics.mipmaps,
                                        anisotropy=getattr(self.graphics, "anisotropy", 1))
        self.textures.preload()
        self.text = TextRenderer(self.font, self.font_small, font_tag=self.font_tag)
        self.lighting = LightingRig(settings.lighting, self.shader)
        self.scene = SceneRenderer(self.textures, self.lighting, self.shader, self.graphics, self.far_plane)
        self.hud = HudRenderer(self.text, settings.hud_layout(), settings.hud_colors, controls_help)
        self.renderer_name = self._query_renderer_name()
        self._applied = (self.graphics.fov, self.far_plane)

    @property
    def far_plane(self) -> float:
        """The draw distance: the graphics settings' (the window settings' for older callers)."""
        far = getattr(self.graphics, "far_plane", None)
        return far if far is not None else self.settings.window.far_plane

    @staticmethod
    def _query_renderer_name() -> str:
        try:
            from OpenGL.GL import GL_RENDERER, glGetString
            value = glGetString(GL_RENDERER)
            return value.decode() if isinstance(value, bytes) else str(value)
        except Exception:
            return "unknown"

    @property
    def uses_shaders(self) -> bool:
        return self.shader is not None

    def init_gl(self) -> None:
        w = self.settings.window
        glViewport(0, 0, w.width, w.height)
        self.apply_projection()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)
        if getattr(self.graphics, "antialiasing", 0) > 0:
            try:                         # (the window was opened with multisampling: switch it on)
                from OpenGL.GL import GL_MULTISAMPLE
                glEnable(GL_MULTISAMPLE)
            except Exception:
                pass
        glEnable(GL_LIGHTING)            # only matters for the fixed-function fallback
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

    def resize(self, settings: GameSettings) -> None:
        """The window has another size than it was opened with (the window manager had the last
        word): the viewport, the projection's aspect and the HUD's layout follow it - the GL
        context, and everything baked in it, stays as it is."""
        self.settings = settings
        w = settings.window
        glViewport(0, 0, w.width, w.height)
        self.apply_projection()
        self.hud.layout = settings.hud_layout()

    def apply_projection(self) -> None:
        w = self.settings.window
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        apply_perspective(self.graphics.fov, w.width / w.height, w.near_plane, self.far_plane)
        glMatrixMode(GL_MODELVIEW)
        self._applied = (self.graphics.fov, self.far_plane)

    def render_frame(self, game) -> None:
        self.render_scene_only(game)
        self.render_hud(game)

    def render_hud(self, game) -> None:
        """The 2D overlay - and, with the equipment view open, the player's figure drawn in 3D into
        the preview frame the HUD left for it."""
        self.hud.render(game)
        if getattr(game, "in_equipment", False):
            w = self.settings.window
            self.scene.draw_player_preview(game, self.hud.equipment_preview_rect(), (w.width, w.height), game.preview_yaw)

    def prebake(self, game) -> None:
        """Bake the whole static scene for ``game`` now, instead of a little at a time as it comes
        into view (for measuring frames)."""
        self.scene.static_scene(game.world).bake_all()

    def needs_baking(self, game) -> bool:
        """Whether drawing ``game`` now would first bake the static scene (seconds of work: the app
        puts its busy box up before it)."""
        return self.scene.needs_baking(game.world)

    def render_scene_only(self, game) -> None:
        if (self.graphics.fov, self.far_plane) != self._applied:
            self.apply_projection()          # (the field of view or the draw distance changed in the menu)
        self.scene.far_plane = self.far_plane
        w = self.settings.window
        self.scene.aspect = w.width / max(1, w.height)
        self.hud.fov = self.graphics.fov
        self.scene.render(game)

    def release(self) -> None:
        self.scene.release()
        self.text.clear_cache()
        self.textures.release()
