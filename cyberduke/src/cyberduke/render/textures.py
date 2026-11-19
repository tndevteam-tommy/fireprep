"""Uploads procedural images to the GPU and hands out texture ids by name.

Every texture gets a full mipmap chain (built here with NumPy, so it works on any GL) and
trilinear filtering - without it, fine detail such as screen scanlines, grating bars or
rack vents sampled from far away aliased into flickering black bands that crawled as the
camera moved. Anisotropic filtering is switched on where the driver offers it (floors and
walls seen at a grazing angle stay sharp instead of smearing).
"""

from __future__ import annotations

import contextlib

import numpy as np
from OpenGL.GL import (
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_REPEAT,
    GL_RGB,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MAX_LEVEL,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glGenTextures,
    glGetFloatv,
    glPixelStorei,
    glTexImage2D,
    glTexParameterf,
    glTexParameteri,
)

from cyberduke.assets.faces import face_from_name, flesh_from_name
from cyberduke.assets.textures import available_textures, generate_texture, tile_size


class TextureMap(dict):
    """Texture ids by name. Faces (``assets.faces.face_name``: "face|tone|mark|expression")
    are painted and uploaded the first time they are asked for: there are hundreds of
    them (every skin tone, mark and expression), and a street shows a few dozen."""

    anisotropy = 0.0

    def get(self, name, default=None):  # type: ignore[override]
        found = super().get(name)
        if found is None and isinstance(name, str) and name.startswith("face|"):
            found = self[name] = upload_texture(face_from_name(name), self.anisotropy)
        elif found is None and isinstance(name, str) and name.startswith("flesh|"):
            # (skin in a person's own tone: drawn on rounded parts, mapped by their own
            # coordinates - never tiled)
            found = self[name] = upload_texture(flesh_from_name(name), self.anisotropy)
        return found if found is not None else default


GL_TEXTURE_MAX_ANISOTROPY = 0x84FE  # EXT_texture_filter_anisotropic / GL 4.6 core
GL_MAX_TEXTURE_MAX_ANISOTROPY = 0x84FF
ANISOTROPY = 8.0

# Character material slots used by the humanoid renderer -> procedural style names
CHARACTER_SLOTS = {
    "skin": "skin",  # the face
    "flesh": "flesh",  # arms, hands, neck
    "shirt": "fabric",  # a guard's dark under-armour fabric
    "cloth": "cloth",  # a civilian's shirt (light, takes the tint)
    "pants": "pants",
    "leather": "leather",
    "armor": "armor",
    "metal": "metal",
}


def mip_chain(img: np.ndarray) -> list[np.ndarray]:
    """The image and every 2x2-averaged reduction of it down to 1 x 1 (uint8 RGB)."""
    levels = [img]
    cur = img.astype(np.float32)
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        h, w = max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2)
        cur = cur[: h * 2, : w * 2].reshape(h, cur.shape[0] // h, w, cur.shape[1] // w, 3)
        cur = cur.mean(axis=(1, 3))
        levels.append(np.clip(np.rint(cur), 0, 255).astype(np.uint8))
    return levels


def _max_anisotropy() -> float:
    try:
        return float(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY))
    except Exception:  # noqa: BLE001 - the extension is optional
        return 0.0


def upload_texture(img: np.ndarray, anisotropy: float = 0.0) -> int:
    """Uploads an image (row 0 = top, as the generators draw them) with mipmaps."""
    img = np.ascontiguousarray(img[::-1])  # GL's first row is the bottom of the picture
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # the small mip levels have odd row sizes
    levels = mip_chain(img)
    for level, data in enumerate(levels):
        h, w = data.shape[:2]
        glTexImage2D(GL_TEXTURE_2D, level, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, len(levels) - 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    if anisotropy > 1.0:
        with contextlib.suppress(Exception):  # optional extension
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY, anisotropy)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
    return int(tex_id)


_IMAGES: dict[str, np.ndarray] = {}


def _image(name: str) -> np.ndarray:
    """Generated once per run: a new window (going fullscreen) only uploads them again."""
    if name not in _IMAGES:
        _IMAGES[name] = generate_texture(name)
    return _IMAGES[name]


def load_texture_suite() -> TextureMap:
    """Generates and uploads every registered style once. Keys are the style names plus
    the character material slot aliases (``shirt`` -> fabric, ...)."""
    anisotropy = min(ANISOTROPY, _max_anisotropy())
    textures = TextureMap(
        {name: upload_texture(_image(name), anisotropy) for name in available_textures()}
    )
    textures.anisotropy = anisotropy
    for slot, style in CHARACTER_SLOTS.items():
        textures[slot] = textures[style]
    return textures


def texture_tiles(textures: TextureMap) -> dict[int, float]:
    """Texture id -> world tile size for the seamless surface materials (for the backend's
    ``set_texture_tiles``)."""
    tiles: dict[int, float] = {}
    for name in available_textures():
        size = tile_size(name)
        if size is not None and name in textures:
            tiles[textures[name]] = size
    return tiles
