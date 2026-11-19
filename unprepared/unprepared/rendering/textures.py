"""Texture loading for the renderer.

Textures are 128x128 tileable PNGs in ``assets/textures`` produced by the
tooling in ``tools/textures`` (``python tools/textures/gen_textures.py``).
A missing file is replaced by a loud magenta placeholder. Uploads use mipmaps
so distant ground/walls do not shimmer.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pygame
from OpenGL.GL import (GL_GENERATE_MIPMAP, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_REPEAT, GL_RGBA, GL_TEXTURE_2D,
                       GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TRUE,
                       GL_UNSIGNED_BYTE, glBindTexture, glDeleteTextures, glGenerateMipmap, glGenTextures, glTexImage2D,
                       glTexParameteri)

SIZE = 128
DEFAULT_TEXTURE_KEYS = ("ground", "building", "wall", "guard", "npc", "wood", "bed", "shelf", "door", "ore", "skin",
                        "flame", "sun", "sun_glow", "moon", "stone", "torch", "roof", "roof_slate", "iron", "lockbox",
                        "lockbox_side", "wax", "cobble", "cobble_edge", "cloth", "leather", "water", "coals",
                        "sign_smith", "sign_store", "sign_tailor", "sign_inn",
                        "bark", "leaves", "pine", "barrel", "log_end", "crate", "hay", "thatch", "cobble_cracked",
                        "cobble_cracked_edge", "wall_cracked", "planks_rot", "crack_decal", "stain_decal", "dirt", "dirt_edge",
                        "rock", "grate", "marble", "carpet", "plaster", "plaster_timber", "field", "field_edge",
                        "stream", "lake", "sand", "snow", "cliff", "cave_rock", "cave_floor",
                        "mushroom", "bone", "notice", "sign_arm", "plaque", "parchment", "banner_greyhold", "banner_deep",
                        "ashlar", "manor_plaster", "window_manor", "window_cottage", "window_lancet", "painting_portrait",
                        "painting_landscape", "rug", "rug_blue", "tile_floor", "brick", "books", "clock_face", "wallpaper",
                        "flowers", "gilt", "sign_bakery")


def placeholder_surface(key: str = "") -> pygame.Surface:
    """Loud magenta tile that makes a missing texture obvious."""
    surf = pygame.Surface((SIZE, SIZE))
    surf.fill((255, 0, 255))
    return surf



# ----------------------------------------------------------------------------- GL upload
def max_anisotropy() -> float:
    """The driver's largest anisotropic filtering degree (1.0: the extension is not there)."""
    try:
        from OpenGL.GL import glGetFloatv
        from OpenGL.GL.EXT.texture_filter_anisotropic import GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT
        return float(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT))
    except Exception:
        return 1.0


def _set_anisotropy(degree: float) -> None:
    """Anisotropic filtering on the bound texture (ground and walls seen at a glancing angle stay
    sharp); quietly nothing on a driver without the extension."""
    if degree <= 1.0:
        return
    try:
        from OpenGL.GL import glTexParameterf
        from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, float(degree))
    except Exception:
        pass


def upload_surface(surface: pygame.Surface, mipmap: bool = False, repeat: bool = True, anisotropy: float = 1.0) -> int:
    """Upload a pygame surface as a GL_TEXTURE_2D; returns the texture id. ``anisotropy`` > 1
    asks for that degree of anisotropic filtering (capped by the driver)."""
    image_data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    # ``glGenerateMipmap`` (GL 3.0 / ARB_framebuffer_object) is the supported way; the old
    # GL_GENERATE_MIPMAP hint is only a fallback for drivers without it.
    modern_mipmaps = mipmap and bool(glGenerateMipmap)
    if mipmap and not modern_mipmaps:
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR if mipmap else GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    wrap = GL_REPEAT
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    if modern_mipmaps:
        glGenerateMipmap(GL_TEXTURE_2D)
    if mipmap:
        _set_anisotropy(anisotropy)
    return texture_id


class TextureRegistry:
    """Maps texture keys ("wall", "npc", ...) to GL texture ids, loading lazily from ``asset_dir``."""

    def __init__(self, asset_dir: str, mipmaps: bool = True, anisotropy: float = 1.0):
        self.asset_dir = asset_dir
        self.mipmaps = mipmaps
        self.anisotropy = min(float(anisotropy), max_anisotropy()) if anisotropy > 1 else 1.0
        self._ids: Dict[str, int] = {}
        self.missing: List[str] = []        # keys that fell back to the placeholder

    def path_for(self, key: str) -> str:
        return os.path.join(self.asset_dir, f"{key}.png")

    def load(self, key: str) -> int:
        """Upload ``key``'s PNG; a missing file - or one that cannot be decoded (truncated, not an
        image) - becomes the magenta placeholder and is listed in ``missing`` rather than stopping
        the game from starting."""
        path = self.path_for(key)
        surface = None
        if os.path.exists(path):
            try:
                surface = pygame.image.load(path)
            except (pygame.error, OSError, ValueError):
                surface = None
        if surface is None:
            surface = placeholder_surface(key)
            self.missing.append(key)
        # ``upload_surface`` hands GL the rows bottom-up, so the image's top ends up at v = 1:
        # the top of the picture is the top of the wall / box it is painted on.
        self._ids[key] = upload_surface(surface, mipmap=self.mipmaps, anisotropy=self.anisotropy)
        return self._ids[key]

    def preload(self, keys=DEFAULT_TEXTURE_KEYS) -> None:
        for key in keys:
            self.load(key)

    def get(self, key: str) -> int:
        tex = self._ids.get(key)
        if tex is None:
            tex = self.load(key)
        return tex

    __getitem__ = get

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` is a texture there is art for (loaded, or a PNG in the asset folder). Without
        this ``in`` would fall back on indexing 0, 1, 2 ... - loading a placeholder for each, for ever."""
        return isinstance(key, str) and (key in self._ids or os.path.exists(self.path_for(key)))

    def release(self) -> None:
        if self._ids:
            glDeleteTextures(list(self._ids.values()))
            self._ids.clear()
