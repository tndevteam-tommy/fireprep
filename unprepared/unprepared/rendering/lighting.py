"""Per-frame lighting state: the atmosphere (sun/moon/sky/fog) plus every torch.

With GLSL available the values go to the lighting shader; otherwise the
fixed-function lights and ``glFog`` are configured to approximate the same look.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from OpenGL.GL import (GL_AMBIENT, GL_CONSTANT_ATTENUATION, GL_DIFFUSE, GL_EXP2, GL_FOG, GL_FOG_COLOR, GL_FOG_DENSITY,
                       GL_FOG_END, GL_FOG_MODE, GL_FOG_START, GL_LINEAR, GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3, GL_LIGHT4, GL_LIGHT5, GL_LIGHT6,
                       GL_LIGHT7, GL_LIGHT_MODEL_AMBIENT, GL_LINEAR_ATTENUATION, GL_POSITION, GL_QUADRATIC_ATTENUATION,
                       glDisable, glEnable, glFogf, glFogfv, glFogi, glLightModelfv, glLightf, glLightfv)

from ..config import LightingSettings
from ..core.daynight import DayNightCycle
from ..core.math3d import Vector3D
from .atmosphere import Atmosphere, compute_atmosphere
from .shaders import LightingShader

Vec3 = Tuple[float, float, float]
FIXED_LIGHTS = (GL_LIGHT2, GL_LIGHT3, GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7)


@dataclass(frozen=True)
class PointLight:
    position: Vec3        # world space
    color: Vec3
    range: float
    bounds: Optional[Tuple[float, float, float, float]] = None   # (min_x, min_z, max_x, max_z) it may not leave
    patch: Optional[Tuple[float, float, float, float]] = None    # a second rectangle it may light: the doorway it spills through


class Camera:
    """View basis for the frame (used to move lights into eye space and to face billboards)."""

    def __init__(self, eye: Vec3, view_matrix: Sequence[float]):
        self.eye = eye
        m = view_matrix
        self.side = (m[0], m[4], m[8])
        self.up = (m[1], m[5], m[9])
        self.back = (m[2], m[6], m[10])          # -forward

    def to_eye_point(self, p: Vec3) -> Vec3:
        d = (p[0] - self.eye[0], p[1] - self.eye[1], p[2] - self.eye[2])
        return self.to_eye_dir(d)

    def to_eye_dir(self, d: Vec3) -> Vec3:
        return (self.side[0] * d[0] + self.side[1] * d[1] + self.side[2] * d[2],
                self.up[0] * d[0] + self.up[1] * d[1] + self.up[2] * d[2],
                self.back[0] * d[0] + self.back[1] * d[1] + self.back[2] * d[2])

    def inverse_view_matrix(self) -> List[float]:
        """Column-major eye -> world: world = R^T * eye + eye_position."""
        s, u, b, e = self.side, self.up, self.back, self.eye
        return [s[0], s[1], s[2], 0.0,
                u[0], u[1], u[2], 0.0,
                b[0], b[1], b[2], 0.0,
                e[0], e[1], e[2], 1.0]


class LightingRig:
    def __init__(self, settings: LightingSettings, shader: Optional[LightingShader] = None):
        self.settings = settings
        self.shader = shader
        self.point_lights: List[PointLight] = []
        self.max_lights = 8
        self.haze = 1.0                  # the day's haze, as a share of the weather's (``GraphicsSettings.haze``)

    # --- gathering ------------------------------------------------------------------
    def flicker(self, time: float, phase: float = 0.0) -> float:
        s = self.settings.torch_flicker
        return 1.0 - s + s * (0.5 + 0.5 * math.sin(time * 11.0 + phase) * math.cos(time * 5.3 + phase * 0.7))

    WALL_FACE_INSET = 0.05      # see _room_bounds
    FAR_LIGHT = 40.0            # a light this far beyond its own reach from the eye ranks with the hidden ones
    DOORWAY_SPILL = 0.35        # how far beyond an open leaf's box the light of a doorway patch reaches

    @classmethod
    def _room_bounds(cls, world, position) -> Optional[Tuple[float, float, float, float]]:
        """A light inside a building is confined to its footprint (walls do not let light through).

        The bounds stop a hair *inside* the outer wall faces, which lie exactly on the footprint
        edge: those faces are excluded, so no light leaks out, while everything set into the
        wall - a door leaf and its lintel sit in the wall's plane, not inside the room - is
        still lit. (An inset of the whole wall thickness used to leave every closed door dark
        from the inside, torch or no torch.)"""
        building = world.building_at(position) if hasattr(world, "building_at") else None
        if building is None:
            return None
        (x1, z1), (x2, z2) = building.min_xz, building.max_xz
        inset = cls.WALL_FACE_INSET
        return (x1 + inset, z1 + inset, x2 - inset, z2 - inset)

    @staticmethod
    def _inside(bounds: Optional[Tuple[float, float, float, float]], x: float, z: float) -> bool:
        return bounds is None or (bounds[0] <= x <= bounds[2] and bounds[1] <= z <= bounds[3])

    @classmethod
    def _doorway_patch(cls, world, building, position) -> Optional[Tuple[float, float, float, float]]:
        """The rectangle a bounded light may *also* reach: the open leaf (and a little around it)
        of the building's door nearest to ``position``. A south-side door folds out onto the
        street when open, so without this the leaf went dark the moment somebody carrying a torch
        stepped over the threshold - light spills through an open doorway, as it should."""
        if building is None or not hasattr(world, "doors"):
            return None
        (x1, z1), (x2, z2) = building.min_xz, building.max_xz
        best, best_dist = None, float("inf")
        doors_in = getattr(world, "doors_in", None)
        for door in (doors_in(x1 - 2.0, z1 - 2.0, x2 + 2.0, z2 + 2.0) if doors_in is not None else world.doors):
            c = door.center
            if not (x1 - 0.6 <= c.x <= x2 + 0.6 and z1 - 0.6 <= c.z <= z2 + 0.6):
                continue
            dist = (c.x - position.x) ** 2 + (c.z - position.z) ** 2
            if dist < best_dist:
                best, best_dist = door, dist
        if best is None or not best.is_open:
            return None
        leaf = best.collider()
        s = cls.DOORWAY_SPILL
        return (leaf.min_point.x - s, leaf.min_point.z - s, leaf.max_point.x + s, leaf.max_point.z + s)

    def _confined(self, world, position) -> Tuple[Optional[Tuple[float, float, float, float]],
                                                  Optional[Tuple[float, float, float, float]]]:
        """(bounds, doorway patch) for a light at ``position``: unbounded in the open."""
        bounds = self._room_bounds(world, position)
        if bounds is None:
            return None, None
        building = world.building_at(position)
        return bounds, self._doorway_patch(world, building, position)

    def gather(self, game, draw_distance: float = 0.0) -> List[PointLight]:
        """Every lit torch and indoor fixture in the world as a point light, the ones that matter
        first - the slots are few (``max_lights``), so they go to the lights that can actually be
        seen from where the player stands:

        * a light that cannot reach anything inside the draw distance is dropped outright;
        * a light confined to a room (``bounds``) the camera is not in lights nothing the camera
          can see except through a doorway, so it ranks behind every light in the open or in the
          player's own room - in the street at night the watch's torches and the forge come
          before the candles behind the shutters;
        * among those, the nearest first (candles a little behind lanterns and torches).
        """
        s, time, world = self.settings, game.world.elapsed, game.world
        lights: List[PointLight] = []
        player = game.player
        if player.has_lit_torch:
            e = player.eye_position
            k = self.flicker(time)
            bounds, patch = self._confined(world, player.position)
            lights.append(PointLight((e.x, e.y - 0.3, e.z), tuple(c * k for c in s.torch_color), s.player_torch_range,
                                     bounds, patch))
        eye = player.position
        ranked = []          # (shut away from the camera, distance bias, light)

        def consider(light: PointLight, dist: float, bias: float = 0.0) -> None:
            if draw_distance > 0.0 and dist - light.range > draw_distance:
                return                                        # cannot touch anything that is drawn
            hidden = not (self._inside(light.bounds, eye.x, eye.z) or
                          (light.patch is not None and self._inside(light.patch, eye.x, eye.z)))
            # a fire in the open far off lights only what lies far off (and fogged): it ranks with the
            # shut-away lights, after the candles near at hand
            far = dist - light.range > self.FAR_LIGHT
            ranked.append((hidden or far, dist + bias, light))

        for idx, npc in enumerate(world.npcs):
            if npc.has_torch and not npc.is_asleep:
                p, k = npc.position, self.flicker(time, idx * 1.7)
                bounds, patch = self._confined(world, p)
                consider(PointLight((p.x, p.y + s.npc_torch_height + 0.3, p.z), tuple(c * k for c in s.torch_color),
                                    s.torch_range, bounds, patch), p.distance_to(eye))
        dark = DayNightCycle.is_torch_time(world.clock.fractional_hour) if hasattr(world, "clock") else True
        for idx, fixture in enumerate(getattr(world, "fixtures", ())):
            if fixture.kind == "torch" and not dark:
                continue                                      # the town's wall torches burn from dusk to dawn
            fx, fy, fz = fixture.position
            k = self.flicker(time, idx * 2.3 + 0.9) if fixture.kind == "candle" else 0.85 + 0.15 * self.flicker(time, idx)
            color = (s.fire_color if fixture.kind == "fire" else s.candle_color if fixture.kind == "candle"
                     else s.torch_color if fixture.kind in ("torch", "sconce")
                     else s.glow_color if fixture.kind == "glow" else s.lantern_color)
            if fixture.kind == "glow":
                k = 1.0                                       # (mushrooms do not flicker)
            dist = ((fx - eye.x) ** 2 + (fy - eye.y) ** 2 + (fz - eye.z) ** 2) ** 0.5
            bounds, patch = self._confined(world, Vector3D(fx, fy, fz))
            consider(PointLight((fx, fy + 0.15, fz), tuple(c * k for c in color), fixture.range, bounds, patch),
                     dist, 2.0 if fixture.kind == "candle" else 0.0)
        ranked.sort(key=lambda entry: (entry[0], entry[1]))
        lights += [light for _, _, light in ranked]
        self.point_lights = lights[:max(0, self.max_lights)]
        return self.point_lights

    # --- applying ---------------------------------------------------------------------
    def begin_frame(self, game, camera: Camera, fog_far: float = 0.0, draw_distance: float = 0.0) -> Atmosphere:
        """Light the frame. ``fog_far`` is a draw distance short of the far plane: beyond it
        nothing is drawn, so the fog is made to close in just before it (on low settings this
        hides the clipping). ``draw_distance`` (the object limit, whatever it is) lets ``gather``
        drop lights that cannot reach anything drawn."""
        atm = compute_atmosphere(game.world.clock.fractional_hour, self.settings, getattr(self, "haze", 1.0))
        lights = self.gather(game, draw_distance=draw_distance or fog_far)
        if self.shader is not None:
            self.shader.use()
            self.shader.set_fog_far(fog_far)
            self.shader.set_atmosphere(camera.to_eye_dir(atm.sun_dir), atm.sun_color, camera.to_eye_dir(atm.moon_dir),
                                       atm.moon_color, atm.ambient, atm.fog_color, atm.fog_density)
            self.shader.set_inverse_view(camera.inverse_view_matrix())
            self.shader.set_point_lights([(camera.to_eye_point(l.position), l.color, l.range, l.bounds, l.patch)
                                          for l in lights])
            self.shader.set_shade_zones(self.shade_zones(game.world), self.settings.underground_light)
            self.shader.set_material()
        else:
            self._apply_fixed_function(atm, lights, fog_far, self._eye_shaded(game.world, camera.eye))
        return atm

    @staticmethod
    def shade_zones(world) -> List[Tuple[float, float, float, float, float, float]]:
        """The caves: every underground building as a box the sky does not light."""
        zones = getattr(world, "underground_zones", None)
        return zones() if callable(zones) else []

    def _eye_shaded(self, world, eye: Vec3) -> bool:
        return any(x1 <= eye[0] <= x2 and z1 <= eye[2] <= z2 and y0 <= eye[1] <= y1
                   for x1, z1, x2, z2, y0, y1 in self.shade_zones(world))

    STUDIO_KEY = (0.45, 0.7, 0.55)          # world-space direction of the key light for the [Tab] preview
    STUDIO_FILL = (-0.6, 0.3, 0.75)

    def begin_studio(self, camera: Camera) -> None:
        """Neutral lighting for the equipment preview: a key light, a softer fill, a flat ambient,
        no fog and no torches - the same figure always looks the same on its stand."""
        key, fill = self.STUDIO_KEY, self.STUDIO_FILL
        self.point_lights = []
        if self.shader is not None:
            self.shader.use()
            self.shader.set_fog_far(0.0)
            self.shader.set_atmosphere(camera.to_eye_dir(key), (0.95, 0.92, 0.86), camera.to_eye_dir(fill),
                                       (0.28, 0.3, 0.36), (0.4, 0.4, 0.44), (0.0, 0.0, 0.0), 0.0)
            self.shader.set_inverse_view(camera.inverse_view_matrix())
            self.shader.set_point_lights([])
            self.shader.set_shade_zones([])
            self.shader.set_material()
            return
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.4, 0.4, 0.44, 1.0])
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [*key, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.95, 0.92, 0.86, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [*fill, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.28, 0.3, 0.36, 1.0])
        for slot in FIXED_LIGHTS:
            glDisable(slot)
        glDisable(GL_FOG)

    def end_frame(self) -> None:
        if self.shader is not None:
            self.shader.stop()
        else:
            glDisable(GL_FOG)

    @staticmethod
    def distance_fog_wins(fog_density: float, fog_far: float) -> bool:
        """Whether the draw-distance fog closes in before the weather's own haze would."""
        if fog_far <= 0.0:
            return False
        haze_reach = 1.2 / max(fog_density, 1e-6)       # where exp2 fog has swallowed ~75% of a thing
        return fog_far < haze_reach

    def _apply_fixed_function(self, atm: Atmosphere, lights: Sequence[PointLight], fog_far: float = 0.0,
                              underground: bool = False) -> None:
        # (one set of lights for the whole frame: in a cave the sky is dimmed everywhere)
        sky = self.settings.underground_light if underground else 1.0
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [*(c * sky for c in atm.ambient), 1.0])
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [*atm.sun_dir, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [*(c * sky for c in atm.sun_color), 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [*atm.moon_dir, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [*(c * sky for c in atm.moon_color), 1.0])
        for slot, light in zip(FIXED_LIGHTS, list(lights) + [None] * len(FIXED_LIGHTS)):
            if light is None:
                glDisable(slot)
                continue
            glEnable(slot)
            glLightfv(slot, GL_POSITION, [*light.position, 1.0])
            glLightfv(slot, GL_DIFFUSE, [*light.color, 1.0])
            glLightf(slot, GL_CONSTANT_ATTENUATION, 1.0)
            glLightf(slot, GL_LINEAR_ATTENUATION, 2.0 / light.range)
            glLightf(slot, GL_QUADRATIC_ATTENUATION, 4.0 / (light.range * light.range))
        glEnable(GL_FOG)
        glFogfv(GL_FOG_COLOR, [*atm.fog_color, 1.0])
        if self.distance_fog_wins(atm.fog_density, fog_far):
            glFogi(GL_FOG_MODE, GL_LINEAR)           # the fixed pipeline has one fog: use whichever is thicker
            glFogf(GL_FOG_START, fog_far * 0.55)
            glFogf(GL_FOG_END, fog_far)
        else:
            glFogi(GL_FOG_MODE, GL_EXP2)
            glFogf(GL_FOG_DENSITY, atm.fog_density)
