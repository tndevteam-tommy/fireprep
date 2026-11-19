"""Frame orchestration: camera, world, entities, NPCs, HUD - on a chosen backend."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pygame

from cyberduke.core.menu import PauseMenu
from cyberduke.render.beams import BeamOccluders
from cyberduke.render.context import DrawContext
from cyberduke.render.effects import draw_scan_pulse
from cyberduke.render.entity_renderers import (
    draw_access_panel,
    draw_pickup,
    draw_security_camera,
    draw_terminal,
    draw_vending_machine,
)
from cyberduke.render.hud import Hud
from cyberduke.render.humanoid import draw_npc
from cyberduke.render.items import draw_loose_items
from cyberduke.render.metro import MetroRenderer
from cyberduke.render.projection import Projector
from cyberduke.render.quality import (
    GraphicsSettings,
    Quality,
    detect_quality,
    lights_for,
    settings_for,
)
from cyberduke.render.sectors import (
    door_leaf_boxes,
    draw_room,
    draw_translucent_props,
    visible_sectors,
)
from cyberduke.render.smoke import SmokeRenderer
from cyberduke.render.television import draw_screens
from cyberduke.render.textures import TextureMap, load_texture_suite, texture_tiles
from cyberduke.render.traffic import TrafficRenderer
from cyberduke.render.weather import WeatherRenderer, fog_colour
from cyberduke.simulation import InputState, Simulation
from cyberduke.systems.weather import open_sky_at, visibility
from cyberduke.world import Light, Room

BACKENDS = ("auto", "modern", "legacy")
# The time of day out of doors (see ``SceneRenderer._apply_time_of_day``): the smog keeps
# the day dim - a golden-brown lift of the light, about a fifth brighter than the night.
DAY_AMBIENT = (0.3, 0.215, 0.09)  # added to the quality preset's ambient at noon
NIGHT_SKY = (0.012, 0.014, 0.03)
DAY_SKY = (0.15, 0.115, 0.07)  # smog, lit from above
INDOOR_SKY = (0.0, 0.0, 0.0)
LAMPS_OFF_BY_DAY = 0.4  # how far the street lamps are dimmed at noon
# The weather (``systems.weather``, ``render.weather``)
FLASH_AMBIENT = (0.55, 0.6, 0.8)  # added to the ambient light by a full lightning stroke
FLASH_SKY = (0.5, 0.55, 0.75)  # ... and to the sky
RAIN_GLOOM = 0.3  # how much of the day's light heavy rain clouds take away
FOG_INDOORS = 0.3  # how much of the fog there is in a room (under a roof, off the street)
MIN_SIGHT = 12.0  # the far plane never comes in closer than this, however thick the fog

LIGHT_EXPOSURE = 0.75
"""Overall strength of the level's lights (emissive neon, screens and signs are not
affected): a little under full keeps the rooms moody without hiding anything."""


def time_of_day_light(
    base_ambient: tuple[float, float, float], daylight: float, outdoor: bool
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """(global ambient, sky colour, street-lamp strength) for ``daylight`` (0 night .. 1
    noon - ``WorldClock.daylight``). Out of doors the smog keeps the day dim: the ambient
    light lifts by ``DAY_AMBIENT`` (golden-brown), the sky goes from a near-black blue to
    a murky brown, the street lamps dim to ``1 - LAMPS_OFF_BY_DAY`` - in all, the street
    about a fifth brighter at noon than at night. Indoors: the night's light at any hour."""
    if not outdoor:
        return tuple(base_ambient), INDOOR_SKY, 1.0  # type: ignore[return-value]
    d = max(0.0, min(1.0, daylight))
    ambient = tuple(b + d * a for b, a in zip(base_ambient, DAY_AMBIENT, strict=True))
    sky = tuple(n + (s - n) * d for n, s in zip(NIGHT_SKY, DAY_SKY, strict=True))
    return ambient, sky, 1.0 - LAMPS_OFF_BY_DAY * d  # type: ignore[return-value]


@dataclass(frozen=True)
class WeatherLight:
    """How the weather lights the scene (``weather_light``)."""

    ambient: tuple[float, float, float]
    sky: tuple[float, float, float]
    fog: tuple[float, float, float]  # the fog's colour
    density: float  # ... and its thickness (per metre; 0 clear air)
    far: float  # the far plane: as far as can be seen


def weather_light(
    base_ambient: tuple[float, float, float],
    daylight: float,
    outdoor: bool,
    weather=None,
    fog_share: float = 1.0,
    view_distance: float = 75.0,
) -> WeatherLight:
    """The time of day's light (``time_of_day_light``) with the weather's over it: heavy
    rain clouds dim the day; a lightning flash lights up everything and the sky; the fog
    (``fog_share`` of it: less in a room off the street) takes on the colour of the night
    or the smog of the day and fills the sky, and the far plane comes in to as far as can
    be seen through it (never nearer than ``MIN_SIGHT``). Indoor levels: no weather."""
    ambient, sky, _ = time_of_day_light(base_ambient, daylight, outdoor)
    if weather is None or not outdoor:
        return WeatherLight(ambient, sky, (0.0, 0.0, 0.0), 0.0, view_distance)
    d = max(0.0, min(1.0, daylight))
    rain, flash = weather.rain, weather.flash
    density = weather.density * fog_share
    gloom = 1.0 - RAIN_GLOOM * rain * d  # (heavy cloud: a darker day)
    ambient = tuple(b + (a - b) * gloom for a, b in zip(ambient, base_ambient, strict=True))
    sky = tuple(c * (1.0 - 0.5 * RAIN_GLOOM * rain) for c in sky)
    fog = fog_colour(d, flash, sky)
    thick = min(1.0, density / 0.05)
    sky = tuple(c + (f - c) * thick for c, f in zip(sky, fog, strict=True))
    ambient = tuple(a + flash * f for a, f in zip(ambient, FLASH_AMBIENT, strict=True))
    sky = tuple(min(1.0, c + flash * f) for c, f in zip(sky, FLASH_SKY, strict=True))
    far = max(MIN_SIGHT, min(view_distance, round(visibility(density), 1)))
    return WeatherLight(ambient, sky, fog, density, far)  # type: ignore[arg-type]


def exposed(light: Light, factor: float) -> Light:
    """A light with its ambient, diffuse and specular colours scaled by ``factor``."""

    def scale(rgb):
        return tuple(c * factor for c in rgb)

    return replace(
        light,
        ambient=scale(light.ambient),
        diffuse=scale(light.diffuse),
        specular=scale(light.specular),
    )


SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "software")


@dataclass(frozen=True)
class ViewportConfig:
    width: int = 1024
    height: int = 768
    fov_y_deg: float = 60.0
    # Depth precision is set by near/far: the player never gets closer than 0.32 m to a
    # wall (their collision radius), so a 0.2 m near plane costs nothing and doubles it.
    near: float = 0.2
    far: float = 55.0


DEPTH_BITS = 24


def open_window(viewport: ViewportConfig, fullscreen: bool = False) -> int:
    """Opens the OpenGL window (``fullscreen``: the whole screen, at the viewport's size)
    with a 24-bit depth buffer and returns the depth bits the driver actually gave. SDL
    only asks for 16 bits unless told otherwise, and with 16 bits everything layered a
    centimetre or two apart (signs on their panels, screens in their bezels, rack faces)
    z-fights at a distance: flickering black bands through the glow."""
    from cyberduke.render.primitives import reset_primitives

    # A new window is a new GL context: display lists compiled in an earlier one (a
    # window closed by pygame.quit(), or by close_window) are not valid in it.
    reset_primitives()
    flags = pygame.OPENGL | pygame.DOUBLEBUF | (pygame.FULLSCREEN if fullscreen else 0)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, DEPTH_BITS)
    try:
        pygame.display.set_mode((viewport.width, viewport.height), flags)
    except pygame.error:  # no 24-bit depth on this driver: take what there is
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 16)
        pygame.display.set_mode((viewport.width, viewport.height), flags)
    try:
        from OpenGL.GL import GL_DEPTH_BITS, glGetIntegerv

        bits = int(glGetIntegerv(GL_DEPTH_BITS))
    except Exception:  # noqa: BLE001 - core profiles do not report it; not fatal
        bits = DEPTH_BITS
    if bits < DEPTH_BITS:
        print(f"[render] only a {bits}-bit depth buffer is available")
    return bits


def close_window() -> None:
    """Closes the OpenGL window (to open it again in another mode - SDL cannot reliably
    switch an OpenGL window between windowed and fullscreen in place). Its context goes
    with it, and so does every GL object made in it: the fixed-function primitives'
    process-wide display lists are forgotten too, or the next context would replay ids
    that now name its own lists (a list calling itself: the first frame never ends)."""
    from cyberduke.render.primitives import reset_primitives

    pygame.display.quit()
    pygame.display.init()
    reset_primitives()


def gpu_strings() -> tuple[str, str]:
    """(GL_RENDERER, GL_VENDOR) as lower-case text; empty strings if the driver is mute."""
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString

    def read(name) -> str:
        return (glGetString(name) or b"").decode(errors="ignore").lower()

    return read(GL_RENDERER), read(GL_VENDOR)


def is_software_renderer() -> bool:
    """True when the GL implementation is a CPU rasterizer (no GPU). Per-pixel lighting is
    several times more expensive there than the fixed-function path."""
    name, _ = gpu_strings()
    return any(tag in name for tag in SOFTWARE_RENDERERS)


def detect_graphics_settings() -> GraphicsSettings:
    """Reads the GL strings and picks the quality preset the machine can carry."""
    renderer, vendor = gpu_strings()
    quality = detect_quality(renderer, vendor)
    print(f"[render] GPU '{renderer or 'unknown'}' -> {quality.value} graphics")
    return settings_for(quality)


def create_context(
    backend: str,
    viewport: ViewportConfig,
    room: Room,
    settings: GraphicsSettings | None = None,
) -> DrawContext:
    """Creates the drawing backend. ``"auto"`` follows the quality preset (per-pixel
    lighting on anything but the lowest); ``"modern"`` falls back to ``"legacy"`` if the
    driver cannot compile the shaders, so the game always starts."""
    # (the backends hold at most eight; the renderer picks the nearest eight every frame)
    lights = list(room.get_lights())[:8]
    far = settings.view_distance if settings is not None else viewport.far
    if backend == "auto":
        prefer_modern = settings.prefer_modern_backend if settings else not is_software_renderer()
        backend = "modern" if prefer_modern else "legacy"
    if backend == "modern":
        try:
            from cyberduke.render.backends.modern import ModernContext

            return ModernContext(
                viewport.width, viewport.height, viewport.fov_y_deg, viewport.near, far, lights
            )
        except Exception as exc:  # noqa: BLE001 - any GL/shader failure means: use the safe path
            print(f"[render] modern backend unavailable ({exc}); falling back to legacy")
            backend = "legacy"
    if backend == "legacy":
        from cyberduke.render.backends.legacy import LegacyContext

        return LegacyContext(
            viewport.width, viewport.height, viewport.fov_y_deg, viewport.near, far, lights
        )
    raise ValueError(f"unknown render backend {backend!r}; choose one of {BACKENDS}")


class SceneRenderer:
    """Owns the backend and GPU resources (textures, cached geometry) and draws one frame."""

    def __init__(
        self,
        viewport: ViewportConfig,
        room: Room,
        backend: str = "auto",
        graphics: GraphicsSettings | Quality | str | None = None,
    ) -> None:
        self.viewport = viewport
        requested = graphics if graphics is not None else detect_graphics_settings()
        initial = requested if isinstance(requested, GraphicsSettings) else settings_for(requested)
        self.ctx = create_context(backend, viewport, room, initial)
        self.textures: TextureMap = load_texture_suite()
        self.ctx.set_texture_tiles(texture_tiles(self.textures))
        self.hud = Hud(self.ctx, viewport.width, viewport.height)
        self.room_lights = list(room.get_lights())
        self._active_lights: list | None = None
        self._level_generation: int | None = None  # which sim level the caches belong to
        self.beam_occluders = BeamOccluders(room.level.props)  # what camera beams stop on
        self._occluder_level = room.level
        self._camera_mounts: dict[str, float | None] = {}  # camera id -> what it hangs from
        self.settings = initial
        self.daylight = 0.0  # (out of doors: 0 night .. 1 day)
        self._sky_set = False
        self._pictures: dict[str, int | None] = {}  # quest picture id -> texture
        self.smoke = SmokeRenderer()  # cigarette smoke (visual only: see render.smoke)
        self.traffic = TrafficRenderer()  # the cars on the road (``render.traffic``)
        self.metro = MetroRenderer()  # the trains on the monorail (``render.metro``)
        self.weather = WeatherRenderer()  # the rain, the puddles, the spray
        self.view_distance = initial.view_distance  # (the preset's - or less, in a fog)
        self._weather_key: tuple | None = None
        self._fog_inside = 1.0  # (1 out in the street .. FOG_INDOORS in a room)
        self.hud.pictures = self.quest_picture
        self.apply_graphics(initial)

    # ------------------------------------------------------------------ quality
    def apply_graphics(self, graphics: GraphicsSettings | Quality | str) -> GraphicsSettings:
        """Switches quality preset live: far plane, ambient level and light budget now, the
        detail effects on the next frame."""
        settings = graphics if isinstance(graphics, GraphicsSettings) else settings_for(graphics)
        self.settings = settings
        self.view_distance = settings.view_distance
        self._weather_key = None
        self.ctx.set_view_distance(settings.view_distance)
        self.ctx.set_global_ambient(settings.global_ambient)
        self._active_lights = None  # force a refresh on the next frame
        self._sky_set = False  # (and the time of day's light on top of the preset's)
        self.hud.settings = settings
        return settings

    def _update_lights(self, eye) -> None:
        lights = lights_for(self.settings, self.room_lights, eye)
        lamps = round(time_of_day_light((0.0, 0.0, 0.0), self.daylight, True)[2], 2)
        key = (lights, lamps)
        if key != self._active_lights:
            self.ctx.set_lights(
                [
                    exposed(light, LIGHT_EXPOSURE * (lamps if light.night_only else 1.0))
                    for light in lights
                ]
            )
            self._active_lights = key

    def _apply_time_of_day(self, sim: Simulation, eye=None) -> None:
        """Outdoors, the city's day shows (``time_of_day_light``) and its weather
        (``weather_light``). Indoors (the facility) the light is the same at any hour,
        whatever the weather."""
        clock = getattr(sim, "clock", None)
        outdoor = sim.level.outdoor
        day = clock.daylight if (clock is not None and outdoor) else 0.0
        weather = getattr(sim, "weather", None) if outdoor else None
        if weather is not None and eye is not None:
            inside = open_sky_at(eye[0], eye[2], sim.level.open_sky) is None
            target = FOG_INDOORS if inside else 1.0
            self._fog_inside += (target - self._fog_inside) * 0.12  # (eased, doorway to street)
        key = (
            round(day, 2),
            (round(weather.flash, 2), round(weather.density, 4), round(weather.rain, 2))
            if weather is not None
            else None,
            round(self._fog_inside, 2),
        )
        if key == self._weather_key and self._sky_set:
            return
        self._weather_key = key
        self.daylight = day
        self._sky_set = True
        # (a backend with no fog of its own draws as far as ever: the far plane pulled in
        # with nothing fading into it would just cut the street off)
        share = self._fog_inside if self.ctx.fog_capable else 0.0
        light = weather_light(
            self.settings.global_ambient, day, outdoor, weather, share,
            self.settings.view_distance,
        )  # fmt: skip
        self.ctx.set_global_ambient(light.ambient)
        self.ctx.set_sky(light.sky)
        self.ctx.set_fog(light.fog, light.density)
        self.view_distance = light.far
        self.ctx.set_view_distance(light.far)

    @property
    def backend(self) -> str:
        return self.ctx.name

    def invalidate_static(self) -> None:
        """Rebuilds every cached piece of static geometry on the next frame."""
        self.ctx.invalidate()

    def follow_level(self, sim: Simulation) -> bool:
        """Notices a level swap (``sim.level_generation`` moved on): drops the cached
        geometry of the old level and takes the new room's lights. Returns True when it
        did so."""
        if sim.level_generation == self._level_generation:
            return False
        if self._level_generation is not None:
            self.ctx.invalidate()
        self._level_generation = sim.level_generation
        self.smoke.clear()
        self.traffic.clear()
        self.metro.clear()
        self.weather.clear(self.ctx)
        self.room_lights = list(sim.room.get_lights())
        self._active_lights = None
        if self.beam_occluders is None or sim.room.level is not self._occluder_level:
            self.beam_occluders = BeamOccluders(sim.room.level.props)
            self._camera_mounts.clear()
        self._occluder_level = sim.room.level
        return True

    def camera_mount(self, cam) -> float | None:
        """Height of the ceiling (or dome) above a camera, found once per camera and level."""
        if cam.camera_id not in self._camera_mounts:
            self._camera_mounts[cam.camera_id] = self.beam_occluders.height_above(
                cam.x, cam.y + 0.22, cam.z
            )
        return self._camera_mounts[cam.camera_id]

    def draw_frame(
        self,
        sim: Simulation,
        dt: float = 0.0,
        menu: PauseMenu | None = None,
        game=None,
    ) -> None:
        """Draws the world from ``sim.viewpoint`` (the player's eyes, a hijacked camera's
        lens, or the start menu's drifting camera) and the overlay on top. ``dt`` only
        drives HUD animation - a paused game passes 0, so the overlay freezes with
        everything else. ``game`` (``cyberduke.game.Game``) decides which overlay: the
        start menu, the pause menu, the [Tab] screen or the HUD; ``menu`` draws a pause
        menu over the HUD without one."""
        ctx = self.ctx
        self.follow_level(sim)
        if game is not None and game.mode.value == "SCREEN":
            self._capture_quest_pictures(sim)
        view = sim.viewpoint
        self.smoke.update(sim.npcs, dt)
        self._apply_time_of_day(sim, view.eye)
        self._update_lights(view.eye)
        ctx.begin_frame(view.eye, view.target)
        self._draw_world(sim, view)
        self.smoke.draw(ctx, view.eye)
        ctx.end_frame()

        v = self.viewport
        projector = Projector(
            v.width, v.height, v.fov_y_deg, v.near, self.view_distance, view.eye, view.target
        )
        self.hud.draw(sim, projector, dt, menu=menu, game=game)

    def _draw_world(self, sim: Simulation, view) -> None:
        ctx = self.ctx
        q = self.settings
        far = self.view_distance  # (the preset's, or as far as can be seen in the fog)
        draw_room(ctx, sim.room, self.textures, view.eye, far, q.effect_distance)
        for terminal in sim.terminals:
            draw_terminal(ctx, terminal, self.textures, sparks=q.sparks)
        for vm in sim.vending_machines:
            draw_vending_machine(ctx, vm, self.textures, sparks=q.sparks)
        for panel in getattr(sim, "panels", ()):
            draw_access_panel(ctx, panel, self.textures)
        for pickup in getattr(sim, "pickups", ()):
            draw_pickup(ctx, pickup, self.textures)
        if sim.cameras:
            # Beams stop on the level's visible geometry plus what moves or is an entity.
            beam_boxes = [t.get_bounds() for t in sim.terminals]
            beam_boxes += [v.get_bounds() for v in sim.vending_machines]
            for door in sim.room.doors.values():
                beam_boxes += door_leaf_boxes(door)
            reach = far + 10.0
            for cam in sim.cameras:
                if (cam.x - view.x) ** 2 + (cam.z - view.z) ** 2 > reach * reach:
                    continue  # beyond the far plane: nothing of it would be drawn anyway
                draw_security_camera(
                    ctx,
                    cam,
                    beam_boxes,
                    slices=q.beam_slices,
                    occluders=self.beam_occluders,
                    mount_top=self.camera_mount(cam),
                )
        traffic = getattr(sim, "traffic", None)
        if traffic is not None:
            self.traffic.draw(ctx, traffic.cars, self.textures, view.eye, far)
        metro = getattr(sim, "metro", None)
        if metro is not None:
            self.metro.draw(ctx, metro, self.textures, view.eye, far)
        for npc in sim.npcs:
            if not getattr(npc, "hidden", False):  # indoors, or off up the main road
                draw_npc(ctx, npc, self.textures, sparks=q.sparks, eye=view.eye)
        draw_loose_items(ctx, sim.items, self.textures)
        if q.scan_ring:
            draw_scan_pulse(ctx, sim.scanner.pulse)
        shown = visible_sectors(sim.room, view.eye, far)
        # the televisions (on, off, which channel) and the view out of the windows
        life = getattr(sim, "life", None)
        draw_screens(
            ctx, shown, self.textures, life.tvs if life is not None else {},
            life.time if life is not None else sim.time, self.daylight, view.eye,
            far, opens=life.open if life is not None else None,
        )  # fmt: skip
        weather = getattr(sim, "weather", None) if sim.level.outdoor else None
        if weather is not None:  # the puddles on the ground (under any fog or steam)
            self.weather.draw_puddles(ctx, getattr(sim, "puddles", ()), weather, view.eye, sim.time)
        draw_translucent_props(ctx, shown, self.textures, sim.room.anim_time, view.eye)
        if traffic is not None:
            self.traffic.draw_glow(ctx, traffic.cars, view.eye, far)
        if weather is not None:  # the spray off the cars' wheels, the rain
            self.weather.draw_spray(ctx, weather, view.eye)
            light = max(self.daylight * 0.6, weather.flash)
            self.weather.draw_rain(ctx, weather, sim.level, view.eye, sim.time, q.rain_drops, light)

    # ------------------------------------------------------------------ quest pictures
    PICTURE_SIZE = (512, 288)

    def _capture_quest_pictures(self, sim: Simulation) -> None:
        """Renders (once each) the pictures of the quests in the log. A picture of the
        level that is loaded now is taken in it; one of another level (a save loaded in
        the facility with the alley's quest running) from a stand-in simulation of that
        level, built once, drawn, and thrown away again."""
        others: dict[str, Simulation] = {}
        for state in sim.quests.quests:
            for pic in state.quest.pictures:
                if pic.picture_id in self._pictures:
                    continue
                if _level_name(pic.level) == sim.level.name:
                    self._pictures[pic.picture_id] = self.capture(sim, pic.eye, pic.target)
                    continue
                other = others.get(pic.level)
                if other is None:
                    other = others[pic.level] = _stand_in(pic.level)
                if other is None:
                    self._pictures[pic.picture_id] = None
                    continue
                self._pictures[pic.picture_id] = self._capture_elsewhere(other, pic)

    def _capture_elsewhere(self, other: Simulation, pic) -> int | None:
        """``capture`` in another level: its geometry, lights and camera mounts stand in
        for the loaded level's for one frame, and the loaded level's caches are built
        again on the next one."""
        saved = (self.room_lights, self.beam_occluders, self._occluder_level, self._camera_mounts)
        self.ctx.invalidate()
        self.room_lights = list(other.room.get_lights())
        self.beam_occluders = BeamOccluders(other.room.level.props)
        self._occluder_level, self._camera_mounts = other.room.level, {}
        self._active_lights = None
        try:
            return self.capture(other, pic.eye, pic.target)
        finally:
            self.ctx.invalidate()
            self.room_lights, self.beam_occluders, self._occluder_level, self._camera_mounts = saved
            self._active_lights = None

    def quest_picture(self, picture) -> int | None:
        return self._pictures.get(picture.picture_id)

    def capture(self, sim: Simulation, eye, target) -> int | None:
        """Draws the world from ``eye`` looking at ``target`` and keeps it as a 16:9
        texture (``PICTURE_SIZE``). The frame drawn over it afterwards replaces it on
        screen; None if the driver will not give the pixels back."""
        import numpy as np
        from OpenGL.GL import GL_RGBA, GL_UNSIGNED_BYTE, glReadPixels

        from cyberduke.core.geometry import Viewpoint

        dx, dy, dz = (target[i] - eye[i] for i in range(3))
        n = (dx * dx + dy * dy + dz * dz) ** 0.5 or 1.0
        view = Viewpoint(*eye, dx / n, dy / n, dz / n, source="picture")
        self._update_lights(view.eye)
        self.ctx.begin_frame(view.eye, view.target)
        self._draw_world(sim, view)
        self.ctx.end_frame()
        w, h = self.viewport.width, self.viewport.height
        try:
            raw = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)
        except Exception:  # noqa: BLE001 - no read-back: no picture
            return None
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        crop_h = min(h, int(w * 9 / 16))
        top = (h - crop_h) // 2
        img = img[top : top + crop_h]
        pw, ph = self.PICTURE_SIZE
        ys = (np.arange(ph) * crop_h / ph).astype(int)
        xs = (np.arange(pw) * w / pw).astype(int)
        small = img[ys][:, xs].astype(np.float32)
        # exposed for a photo: the night is dark, and a picture of a door in the dark is
        # a picture of nothing (a little lift of the shadows, a stronger one of the mids)
        small = 255.0 * np.clip(small / 255.0 * 1.6, 0.0, 1.0) ** 0.8
        small = small.astype(np.uint8)
        small[..., 3] = 255
        self._active_lights = None  # the next frame puts its own lights back
        return self.ctx.upload_texture_rgba(np.ascontiguousarray(small).tobytes(), pw, ph)


def _stand_in(key: str) -> Simulation | None:
    """A fresh simulation of the level ``key`` (``levels.LEVELS``), settled for a moment."""
    from cyberduke.levels import LEVELS

    build = LEVELS.get(key)
    if build is None:
        return None
    other = Simulation(build())
    other.update(1 / 30, InputState())
    return other


def _level_name(key: str) -> str:
    from cyberduke.levels import LEVELS

    build = LEVELS.get(key)
    return build().name if build is not None else ""
