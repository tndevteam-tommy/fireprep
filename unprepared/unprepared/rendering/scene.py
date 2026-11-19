"""Draws the 3D world for the current frame, honouring the graphics settings.

Everything static is baked into display lists once (:mod:`batching`) and culled by chunk;
per frame only what moves or flickers is issued live: door poses (the leaves themselves
are pre-recorded), NPCs (their outfits are pre-recorded in local space), torches, lantern
and candle flames, the forge and oven, the sky, the tutorial's waypoint arrow and the
player's held torch.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from OpenGL.GL import (GL_AMBIENT_AND_DIFFUSE, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_EMISSION, GL_FRONT_AND_BACK,
                       GL_MODELVIEW, GL_PROJECTION, GL_SCISSOR_TEST, glClear, glClearColor, glColor3f, glDisable, glEnable,
                       glLoadIdentity, glMaterialfv, glMatrixMode, glPopMatrix, glPushMatrix, glRotatef, glScissor,
                       glViewport)

from ..core.daynight import DayNightCycle
from ..core.math3d import view_direction
from ..graphics import GraphicsSettings
from .batching import Compiled, StaticScene
from .camera import Frustum, apply_look_at, apply_perspective, look_at_matrix
from .gear import draw_figure
from .lighting import Camera, LightingRig
from .models import draw_face, draw_fixture_flame, draw_held_torch, draw_npc
from .primitives import draw_door, draw_door_geometry
from .shaders import LightingShader
from .sky import draw_sky_bodies, make_star_field
from .furnishings import DRAWERS as FURNISHINGS
from .structures import DRAWERS as _STRUCTURES

STRUCTURES = {**_STRUCTURES, **FURNISHINGS}
from .terrain import draw_water
from .textures import TextureRegistry
from .waypoint import draw_waypoint_arrow, relative_yaw


class SceneRenderer:
    def __init__(self, textures: TextureRegistry, lighting: LightingRig, shader: Optional[LightingShader],
                 graphics: Optional[GraphicsSettings] = None, far_plane: float = 100.0):
        self.textures = textures
        self.lighting = lighting
        self.shader = shader
        self.graphics = graphics or GraphicsSettings()
        self.far_plane = far_plane
        self._stars = None
        self._star_count = -1
        self._static: Optional[StaticScene] = None
        self._outfits: Dict[str, Tuple[object, Compiled]] = {}      # NPC name -> (the Outfit recorded, its lists)
        self._doors: Dict[str, Compiled] = {}                          # door id -> its leaf / bars in local space
        self.drawn_objects = 0           # statistics for the performance overlay
        self.culled_objects = 0
        self.aspect = 4.0 / 3.0          # the window's (the renderer keeps it up to date): for the view frustum
        self.cull_to_view = True
        self.bake_budget: Optional[float] = 0.008      # seconds a frame may spend baking chunks (None: all at once)
        self._fresh_bake = False

    # --- material abstraction (shader or fixed function) -----------------------------
    def material(self, color=(1.0, 1.0, 1.0), emission=(0.0, 0.0, 0.0), textured: bool = True, fog: float = 1.0,
                 vertex_alpha: bool = False) -> None:
        if self.shader is not None:
            self.shader.set_material(color, emission, textured, fog, vertex_alpha)
        else:
            glColor3f(*color)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [*emission, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [*color, 1.0])

    def stars(self):
        if self._star_count != self.graphics.star_count:
            self._stars = make_star_field(self.graphics.star_count)
            self._star_count = self.graphics.star_count
        return self._stars

    # --- baked geometry ---------------------------------------------------------------
    def static_scene(self, world) -> StaticScene:
        """The world's static geometry, baked on first use (and again for another world or
        a change of ground detail)."""
        cells = self.graphics.ground_cells
        if self._static is None or not self._static.matches(world, self.textures, cells):
            if self._static is not None:
                self._static.release()
            self._static = StaticScene(world, self.textures, STRUCTURES, cells)
            self._fresh_bake = True                  # (its first picture is drawn whole: see ``render``)
        self._static.world = world                   # (another world of the same level: the bake serves it too)
        return self._static

    def needs_baking(self, world) -> bool:
        return self._static is None or not self._static.matches(world, self.textures, self.graphics.ground_cells)

    def outfit_for(self, npc) -> Compiled:
        """An NPC's clothes recorded in local space (re-recorded if the outfit or what is worn changes)."""
        from .models import draw_outfit
        clothes = (npc.clothes, tuple((slot.value, item.gear, item.gear_color) for slot, item in sorted(
            npc.equipment.items(), key=lambda kv: kv[0].value)))
        entry = self._outfits.get(npc.name)
        if entry is None or entry[0] != clothes:
            if entry is not None:
                entry[1].release()
            compiled = Compiled.record(lambda m: draw_outfit(npc, self.textures, m))
            self._outfits[npc.name] = entry = (clothes, compiled)
        return entry[1]

    def door_geometry(self, door) -> Compiled:
        compiled = self._doors.get(door.door_id)
        if compiled is None:
            tex = self.textures
            compiled = self._doors[door.door_id] = Compiled.record(
                lambda m: draw_door_geometry(door, tex["door"], bars_texture_id=tex["iron"]))
        return compiled

    def release(self) -> None:
        """Free every display list (the renderer is being rebuilt or shut down)."""
        if self._static is not None:
            self._static.release()
            self._static = None
        for _, compiled in self._outfits.values():
            compiled.release()
        self._outfits.clear()
        for compiled in self._doors.values():
            compiled.release()
        self._doors.clear()

    # --- frame --------------------------------------------------------------------------
    def camera_for(self, player):
        dir_x, dir_y, dir_z = view_direction(player.rotation_y, player.rotation_x)
        e = player.eye_position
        eye = (e.x, e.y, e.z)
        center = (eye[0] + dir_x, eye[1] + dir_y, eye[2] + dir_z)
        return Camera(eye, look_at_matrix(eye, center, (0.0, 1.0, 0.0))), center

    def _within(self, box, eye, limit: float) -> bool:
        cx = (box.min_point.x + box.max_point.x) / 2 - eye[0]
        cz = (box.min_point.z + box.max_point.z) / 2 - eye[2]
        radius = max(box.max_point.x - box.min_point.x, box.max_point.z - box.min_point.z) / 2
        return (cx * cx + cz * cz) ** 0.5 - radius <= limit

    def render(self, game) -> None:
        world, tex, time, gfx = game.world, self.textures, game.world.elapsed, self.graphics
        camera, center = self.camera_for(game.player)
        object_limit = gfx.object_draw_distance(self.far_plane)
        npc_limit = gfx.character_draw_distance(self.far_plane)

        self.lighting.max_lights = gfx.torch_lights
        self.lighting.haze = getattr(gfx, "haze", 1.0)
        # The view matrix goes on *before* the frame is lit: fixed-function lights take their
        # positions through whatever modelview is current when they are set, so lighting first
        # would place the sun, the moon and every torch with the previous frame's camera - or
        # with none at all after a frame that ended drawing in view space (the held torch, the
        # tutorial's arrow), leaving them to swing round with the player's head.
        glLoadIdentity()
        apply_look_at(camera.eye, center)
        atm = self.lighting.begin_frame(game, camera, fog_far=object_limit if object_limit < self.far_plane else 0.0,
                                        draw_distance=object_limit)
        if self.shader is not None:
            self.shader.set_contrast(gfx.contrast)
        glClearColor(*atm.sky_color, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        draw_sky_bodies(atm, camera, tex, self.material, self.lighting.settings, show_bodies=gfx.sky_bodies,
                        stars=self.stars() if gfx.stars else (), star_alpha=atm.star_alpha, shader=self.shader)

        # what is in view: everything further out to the sides or behind than the field of view
        # (and a margin) is skipped, on every draw distance - the card is handed only what it shows
        frustum = Frustum(camera.eye, center, gfx.fov, self.aspect) if self.cull_to_view else None
        # everything that never moves: ground, roads, walls, furniture, lantern bodies, floors and roofs
        static = self.static_scene(world)
        # (a new bake's first picture is drawn whole, behind the app's busy box; afterwards what comes
        # into view for the first time is baked a little each frame - nearest first - and the rest ahead)
        budget = None if self._fresh_bake else self.bake_budget
        self._fresh_bake = False
        self.drawn_objects, self.culled_objects = static.draw(self.material, camera.eye, object_limit, roofs=gfx.roofs,
                                                              frustum=frustum, bake_budget=budget)

        def seen(box) -> bool:
            return frustum is None or frustum.sees_box(box.min_point.x - 1.0, box.min_point.y - 1.0, box.min_point.z - 1.0,
                                                       box.max_point.x + 1.0, box.max_point.y + 2.0, box.max_point.z + 1.0)
        # the live part of the world
        for prop in world.props:
            if getattr(prop, "kind", "box") in StaticScene.ANIMATED_STRUCTURES:
                if self._within(prop, camera.eye, object_limit) and seen(prop):
                    STRUCTURES[prop.kind](prop, tex, self.material, time)
                    self.drawn_objects += 1
                else:
                    self.culled_objects += 1
        from .batching import CONTAINER_DRAWERS
        for container in world.containers:
            if getattr(container, "kind", "box") in StaticScene.ANIMATED_CONTAINERS and self._within(container, camera.eye, object_limit):
                CONTAINER_DRAWERS[container.kind](container, tex, self.material, time)
        for door in world.doors:
            leaf = door.collider()
            if self._within(leaf, camera.eye, object_limit) and seen(leaf):
                draw_door(door, tex["door"], geometry=lambda d=door: self.door_geometry(d).draw(self.material))
                self.drawn_objects += 1
        dark = DayNightCycle.is_torch_time(world.clock.fractional_hour)
        for fixture in world.fixtures:
            fx, fy, fz = fixture.position
            if fixture.kind == "torch" and not dark:
                continue                                   # wall torches are lit at dusk
            if (((fx - camera.eye[0]) ** 2 + (fz - camera.eye[2]) ** 2) ** 0.5 <= object_limit
                    and (frustum is None or frustum.sees_sphere(fx, fy, fz, 1.0))):
                draw_fixture_flame(fixture, tex, self.material, time)
        for npc in world.npcs:
            p = npc.position
            if npc.position.distance_to(game.player.position) <= npc_limit and (
                    frustum is None or frustum.sees_sphere(p.x, p.y + 1.0, p.z, 1.6)):
                outfit = None if npc.is_asleep else self.outfit_for(npc).draw
                draw_npc(npc, tex, self.material, world.npc_body_box(npc), time, outfit=outfit,
                         head=world.sleep_head(npc) if npc.is_asleep and hasattr(world, "sleep_head") else "")
                self.drawn_objects += 1
            else:
                self.culled_objects += 1

        terrain = getattr(world, "terrain", None)
        if terrain is not None and not terrain.flat:
            draw_water(terrain, tex, self.material, time, self.shader)      # (after the opaque world: it is see-through)

        waypoint = getattr(game, "waypoint", None) if hasattr(game, "waypoint") else getattr(game, "tutorial_waypoint", None)
        if waypoint is not None and not getattr(game, "in_equipment", False):
            glLoadIdentity()
            draw_waypoint_arrow(relative_yaw(game.player, waypoint), self.material, time)

        if game.player.has_lit_torch:
            glLoadIdentity()
            draw_held_torch(tex, self.material, time)

        self.lighting.end_frame()

    # --- the player's figure on its stand (the [Tab] equipment view) ------------------------
    PREVIEW_FOV = 30.0
    PREVIEW_EYE = (0.0, 1.0, 4.7)
    PREVIEW_LOOK = (0.0, 0.92, 0.0)

    def draw_player_preview(self, game, rect, window_size, yaw: float) -> None:
        """Draw the player's dressed figure (``gear.draw_figure`` + a face) turned by ``yaw`` into
        the pixel ``rect`` (x1, y1, x2, y2; y down) of the window, under studio lighting. The
        3D state is restored afterwards; call after the HUD has painted the backdrop."""
        x1, y1, x2, y2 = rect
        width, height = window_size
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        glViewport(x1, height - y2, w, h)
        glEnable(GL_SCISSOR_TEST)
        glScissor(x1, height - y2, w, h)
        glClear(GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        apply_perspective(self.PREVIEW_FOV, w / h, 0.1, 20.0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        eye, look = self.PREVIEW_EYE, self.PREVIEW_LOOK
        camera = Camera(eye, look_at_matrix(eye, look, (0.0, 1.0, 0.0)))
        apply_look_at(eye, look)
        self.lighting.begin_studio(camera)
        if self.shader is not None:
            self.shader.set_contrast(1.0)
        glRotatef(180.0 + yaw, 0.0, 1.0, 0.0)             # yaw 0: the figure faces the camera (its front is local -z)
        player = game.player
        draw_figure(player.figure, player.equipment, self.textures, self.material)
        draw_face(self.material)
        self.material()
        self.lighting.end_frame()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, width, height)
