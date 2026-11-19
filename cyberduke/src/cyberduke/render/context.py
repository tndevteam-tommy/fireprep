"""The drawing interface every renderer in ``cyberduke.render`` targets.

Mesh code (props, humanoids, entities, HUD) is written once against ``DrawContext`` and runs
on either backend:

* ``backends.legacy``  - fixed-function OpenGL 1.x with display lists (the original look).
* ``backends.modern``  - VBOs + GLSL shaders; static geometry is batched per texture.

The API mirrors the handful of fixed-function concepts the game used (matrix stack, current
colour, emission/specular, unit cube/cylinder, unlit lines, a 2D overlay) so the port was
mechanical and the two backends can be diffed pixel for pixel.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

RGB = tuple[float, float, float]
RGBA = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]

NO_EMISSION: RGB = (0.0, 0.0, 0.0)
WHITE: RGB = (1.0, 1.0, 1.0)


class DrawContext(ABC):
    """Stateful immediate-style drawing. Methods are grouped as the fixed-function pipeline
    grouped them; see each backend for what they map to."""

    # ------------------------------------------------------------------ frame
    @abstractmethod
    def begin_frame(self, eye: Vec3, target: Vec3) -> None:
        """Clears and sets the camera for a 3D frame."""

    @abstractmethod
    def end_frame(self) -> None: ...

    # ------------------------------------------------------------------ matrix stack
    @abstractmethod
    def push(self) -> None: ...

    @abstractmethod
    def pop(self) -> None: ...

    @abstractmethod
    def translate(self, x: float, y: float, z: float) -> None: ...

    @abstractmethod
    def rotate(self, angle_deg: float, x: float, y: float, z: float) -> None: ...

    @abstractmethod
    def scale(self, x: float, y: float, z: float) -> None: ...

    @contextmanager
    def pushed(self) -> Iterator[None]:
        self.push()
        try:
            yield
        finally:
            self.pop()

    @contextmanager
    def transformed(
        self,
        translate: Sequence[float] | None = None,
        rotate: Sequence[tuple[float, float, float, float]] | None = None,
        scale: Sequence[float] | None = None,
    ) -> Iterator[None]:
        self.push()
        try:
            if translate is not None:
                self.translate(*translate)
            for angle, ax, ay, az in rotate or ():
                self.rotate(angle, ax, ay, az)
            if scale is not None:
                self.scale(*scale)
            yield
        finally:
            self.pop()

    # ------------------------------------------------------------------ material
    @abstractmethod
    def color(self, r: float, g: float, b: float) -> None: ...

    @abstractmethod
    def emission(self, rgb: Sequence[float] | None) -> None:
        """Self-illumination; ``None`` (or zeros) switches it off."""

    @abstractmethod
    def specular(self, rgb: Sequence[float] | None, shininess: float = 0.0) -> None: ...

    def alpha(self, a: float) -> None:  # noqa: B027 - optional hook
        """Opacity of what is drawn next (1 = opaque). Backends that cannot blend ignore it."""

    def depth_mask(self, enabled: bool) -> None:  # noqa: B027 - optional hook
        """Whether drawing writes depth (off for translucent layers such as fog)."""

    @contextmanager
    def emissive(
        self, rgb: Sequence[float], color: Sequence[float] | None = None
    ) -> Iterator[None]:
        self.emission(rgb)
        if color is not None:
            self.color(*color[:3])
        try:
            yield
        finally:
            self.emission(None)
            if color is not None:
                self.color(*WHITE)

    # ------------------------------------------------------------------ primitives
    @abstractmethod
    def cube(self, tex_id: int | None = None) -> None:
        """Unit cube centred on the origin (extent -0.5..0.5) under the current transform."""

    @abstractmethod
    def cylinder(
        self, r_bot: float, r_top: float, h: float, slices: int = 10, tex_id: int | None = None
    ) -> None:
        """Vertical cylinder/cone from y=0 to y=h under the current transform."""

    def shape(self, name: str, tex_id: int | None = None) -> None:
        """A rounded unit shape (``render.shapes``: a head, a limb, a finger, a boot...)
        filling the unit cube under the current transform, like ``cube``. A backend that
        cannot draw meshes draws the cube it fills."""
        self.cube(tex_id)

    def draw_batch(self, vertices: Any, indices: Any, ranges: Sequence[Any]) -> None:  # noqa: B027
        """Draws a batch of triangles under the current transform (``render.batch``):
        ``vertices`` rows x, y, z, nx, ny, nz, u, v, r, g, b, emission r, g, b, specular
        strength, shininess; ``ranges`` of (texture id, first index, count). Backends
        override this; the default draws nothing."""

    def box(
        self,
        cx: float,
        cy: float,
        cz: float,
        sx: float,
        sy: float,
        sz: float,
        tex_id: int | None = None,
    ) -> None:
        with self.transformed((cx, cy, cz), scale=(sx, sy, sz)):
            self.cube(tex_id)

    @abstractmethod
    def lines(self, segments: Sequence[tuple[Vec3, Vec3]], rgba: RGBA, width: float = 1.0) -> None:
        """Unlit, alpha-blended world-space line segments (current transform applies)."""

    @abstractmethod
    def soft_disc(self, rgba: RGBA, slices: int = 24, inner: float = 0.0) -> None:
        """Unlit, alpha-blended unit disc in the XZ plane under the current transform whose
        opacity is ``rgba[3]`` out to ``inner`` of the radius and fades to nothing at the
        edge - the soft pool of light under a mushroom, a puff of fog, a puddle."""

    def glow_mesh(self, key: Any, rgba: RGBA, build: Callable[[], Any]) -> None:  # noqa: B027
        """Unlit, alpha-blended world-space triangles (``build()`` returns an (N, 4) array:
        x, y, z and an opacity factor, three rows per triangle) in the colour ``rgba``: the
        baked light pools (``render.glow``). The mesh is built on first use and kept under
        ``key`` until ``invalidate``. Backends override this; the default draws nothing."""

    # ------------------------------------------------------------------ static caching
    @abstractmethod
    def cached(self, key: Any, build: Callable[[], None]) -> None:
        """Draws static geometry produced by ``build`` (a function issuing cube/cylinder calls
        on this context). The first call records it - a display list or a batched VBO -
        and later calls replay it."""

    @abstractmethod
    def invalidate(self, key: Any = None) -> None: ...

    # ------------------------------------------------------------------ 2D overlay
    @abstractmethod
    def begin_overlay(self, width: int, height: int) -> None:
        """Switches to a pixel-space orthographic 2D mode (y down), lighting/depth off."""

    @abstractmethod
    def end_overlay(self) -> None: ...

    @abstractmethod
    def rect(self, x: float, y: float, w: float, h: float, rgba: RGBA) -> None: ...

    @abstractmethod
    def rect_outline(
        self, x: float, y: float, w: float, h: float, rgba: RGBA, width: float = 2.0
    ) -> None: ...

    @abstractmethod
    def image(self, tex_id: int, x: float, y: float, w: float, h: float) -> None:
        """Textured quad in overlay space (texture row 0 at the bottom, as uploaded by pygame)."""

    def clip(self, x: float, y: float, w: float, h: float) -> None:  # noqa: B027 - optional
        """Restricts overlay drawing to a rectangle (a scrolling list's window) until
        ``unclip``. A backend without it draws everything (the lists still only draw the
        rows that are in view)."""

    def unclip(self) -> None:  # noqa: B027 - optional
        """Ends ``clip``."""

    # ------------------------------------------------------------------ tiled textures
    texture_tiles: dict[int, float] = {}  # noqa: RUF012 - replaced per instance

    def set_texture_tiles(self, tiles: dict[int, float]) -> None:
        """Texture id -> the world size (metres) of one tile. Cubes and cylinders drawn with
        such a texture repeat it across their real size (from the current transform)
        instead of stretching one copy over the whole face."""
        self.texture_tiles = dict(tiles)

    @staticmethod
    def cube_repeat(axis_lengths, tile: float) -> tuple[float, float, float]:
        """Repeats along a cube's local x, y, z for a world tile size."""
        return tuple(max(a, 1e-4) / tile for a in axis_lengths)  # type: ignore[return-value]

    @staticmethod
    def cylinder_repeat(
        r_bot: float, r_top: float, h: float, axis_lengths, tile: float
    ) -> tuple[float, float, float]:
        """(around, along, across the caps) repeats of a cylinder for a world tile size."""
        sx, sy, sz = axis_lengths
        r = max(r_bot, r_top) * (sx + sz) / 2.0
        around, along, across = 2.0 * math.pi * r, h * sy, 2.0 * r
        return (max(around, 1e-4) / tile, max(along, 1e-4) / tile, max(across, 1e-4) / tile)

    # ------------------------------------------------------------------ quality
    # Optional capabilities: a backend that cannot change these at runtime simply keeps the
    # values it was built with (the default no-ops), and the rest of the renderer does not
    # have to care which backend it is talking to.
    def set_view_distance(self, far: float) -> None:  # noqa: B027 - optional hook
        """Moves the far plane (the graphics quality preset's view distance)."""

    def set_lights(self, lights: Sequence[Any]) -> None:  # noqa: B027 - optional hook
        """Replaces the active hardware lights (the preset's light budget)."""

    def set_global_ambient(self, rgb: RGB) -> None:  # noqa: B027 - optional hook
        """Base light level of the whole scene."""

    def set_sky(self, rgb: RGB) -> None:  # noqa: B027 - optional hook
        """The colour the frame is cleared to (what shows where nothing is drawn)."""

    # Whether ``set_fog`` fogs the picture (the fixed-function backend's fog is worked
    # out per vertex: a long wall or the pavement, its corners far off, went grey even
    # right in front of the eye - it has none, and so draws as far as ever in a fog)
    fog_capable = False

    def set_fog(self, rgb: RGB, density: float) -> None:  # noqa: B027 - optional hook
        """Distance fog (``systems.weather``): a thing ``d`` metres from the eye keeps
        exp(-(density * d)^2) of its own colour and takes the rest from ``rgb`` (what is
        lit from within - neon, screens - shows through it further). 0: clear air."""

    # ------------------------------------------------------------------ resources
    @abstractmethod
    def upload_texture_rgba(self, data: bytes, w: int, h: int) -> int: ...

    @abstractmethod
    def delete_texture(self, tex_id: int) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
