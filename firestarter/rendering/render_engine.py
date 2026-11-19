"""
Firestarter - 3D RenderEngine (ModernGL, DESIGN Section 8)

Frame pipeline
  1. Shadow map from the sun (or moon) around the player.
  2. HDR scene: procedural sky -> terrain -> instanced vegetation & camp props -> fuel pieces
     (glowing char fissures, ash) -> water -> ember bed -> lit smoke / steam -> procedural
     flames -> sparks & embers -> holographic placement / blueprint guides.
  3. First-person viewmodel (hands, tools, the fuel in your off hand) with its own projection.
  4. Post: heat shimmer, bloom, ACES tonemap, colour grade, vignette, dither, FXAA.
  5. 2D UI overlay (PIL-rendered, re-uploaded only when it changes).
"""

import math
import time as _time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from firestarter.config import WINDOW_WIDTH, WINDOW_HEIGHT, FOV_DEGREES, GRAPHICS_PRESETS
from firestarter.thermodynamics.combustion_body import CombustionStage, orientation_to_axis
from firestarter.thermodynamics.wood_properties import WoodSpeciesType, get_wood_properties
from firestarter.forestry.tree import TreeGrowthStage
from firestarter.forestry.bucking import SECTION_LEN
from firestarter.simulation import terrain
from firestarter.rendering import shaders as S
from firestarter.rendering import mesh_generator as mg
from firestarter.rendering import scene_assets as sa
from firestarter.rendering import gl_utils as G
from firestarter.rendering.particle_system import ParticleSystem, K_SMOKE, K_STEAM
from firestarter.rendering.atmosphere import Atmosphere
from firestarter.rendering.ui_renderer import UIRenderer

CAMERA_NEAR = 0.05
CAMERA_FAR = 900.0
VM_NEAR = 0.01
VM_FOV = 58.0

# re-exported legacy helpers
matrix_perspective = G.perspective
matrix_look_at = G.look_at
matrix_translate = G.translate
matrix_scale = G.scale
matrix_rotate_x = G.rot_x
matrix_rotate_y = G.rot_y
matrix_rotate_z = G.rot_z
matrix_ortho = G.orthographic

FLAME_STRIDE = 12
MAX_FLAMES = 512
MAX_PARTICLES = 8000


class RenderEngine:
    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT, ctx=None, standalone: bool = False,
                 preset: str = "High"):
        self.width = width
        self.height = height
        self.ctx = ctx
        self.standalone = standalone
        self.is_gl_available = False
        self.init_error: Optional[str] = None
        self.preset_name = preset
        self.preset = dict(GRAPHICS_PRESETS.get(preset, GRAPHICS_PRESETS["High"]))
        self.particle_system = ParticleSystem(MAX_PARTICLES)
        self.atmosphere = Atmosphere()
        self.ui_renderer = UIRenderer(width, height)
        self.time = 0.0
        self.frame = 0
        self.last_frame_ms = 0.0
        self.stats: Dict[str, Any] = {}
        self.target_fbo = None
        self._veg_key = None
        self._ui_sig = None
        self._last_lighting = None
        self.pending_events = []
        self._flame_instances = np.zeros((0, FLAME_STRIDE), np.float32)
        self._weather_cloud = 0.2
        try:
            self._init_moderngl()
        except Exception as e:   # pragma: no cover - reported to the caller / tests
            import traceback
            self.init_error = f"{e}\n{traceback.format_exc()}"
            self.is_gl_available = False

    # ================================================================== initialisation
    def _init_moderngl(self):
        import moderngl
        self.mgl = moderngl
        if self.ctx is None:
            self.ctx = moderngl.create_standalone_context(require=330) if self.standalone else moderngl.create_context(require=330)
        ctx = self.ctx
        self.gl_renderer = ctx.info.get("GL_RENDERER", "?")
        P = lambda vs, fs: ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.p_sky = P(S.SKY_VS, S.SKY_FS)
        self.p_terrain = P(S.TERRAIN_VS, S.TERRAIN_FS)
        self.p_obj = P(S.OBJECT_VS, S.OBJECT_FS)
        self.p_obj_inst = P(S.with_defines(S.OBJECT_VS, "INSTANCED"), S.OBJECT_FS)
        self.p_fuel = P(S.OBJECT_VS, S.FUEL_FS)
        self.p_water = P(S.OBJECT_VS, S.WATER_FS)
        self.p_ember = P(S.OBJECT_VS, S.EMBER_FS)
        self.p_ghost = P(S.OBJECT_VS, S.GHOST_FS)
        self.p_flame = P(S.FLAME_VS, S.FLAME_FS)
        self.p_part = P(S.PARTICLE_VS, S.PARTICLE_FS)
        self.p_shadow = P(S.SHADOW_VS, S.SHADOW_FS)
        self.p_shadow_inst = P(S.with_defines(S.SHADOW_VS, "INSTANCED"), S.SHADOW_FS)
        self.p_down = P(S.FULLSCREEN_VS, S.BLOOM_DOWN_FS)
        self.p_up = P(S.FULLSCREEN_VS, S.BLOOM_UP_FS)
        self.p_comp = P(S.FULLSCREEN_VS, S.COMPOSITE_FS)
        self.p_fxaa = P(S.FULLSCREEN_VS, S.FXAA_FS)
        self.p_ui = P(S.UI_VS, S.UI_FS)
        self.lit_programs = [self.p_terrain, self.p_obj, self.p_obj_inst, self.p_fuel, self.p_water, self.p_ember,
                             self.p_part, self.p_sky]
        self.empty_vao = {}
        for name, prog in (("sky", self.p_sky), ("down", self.p_down), ("up", self.p_up), ("comp", self.p_comp),
                           ("fxaa", self.p_fxaa), ("ui", self.p_ui)):
            self.empty_vao[name] = ctx.vertex_array(prog, [])
        self._build_static_geometry()
        self._build_dynamic_buffers()
        self._create_targets()
        self.is_gl_available = True

    def _vao(self, prog, mesh: mg.MeshData, inst_buf=None):
        vbo = self.ctx.buffer(mesh.vertices.tobytes())
        ibo = self.ctx.buffer(mesh.indices.tobytes())
        content = [(vbo, "3f 3f 2f 4f", "in_position", "in_normal", "in_uv", "in_color")]
        if inst_buf is not None:
            content.append((inst_buf, "4f 4f 4f/i", "in_i0", "in_i1", "in_i2"))
        return self.ctx.vertex_array(prog, content, ibo, skip_errors=True)

    class _Mesh:
        def __init__(self, eng, mesh: mg.MeshData, instanced: bool = False):
            self.mesh = mesh
            self.instanced = instanced
            self.vbo = eng.ctx.buffer(mesh.vertices.tobytes())
            self.ibo = eng.ctx.buffer(mesh.indices.tobytes())
            self.vaos = {}
            self.eng = eng

        def vao(self, prog, inst_buf=None):
            key = (id(prog), id(inst_buf))
            if key not in self.vaos:
                content = [(self.vbo, "3f 3f 2f 4f", "in_position", "in_normal", "in_uv", "in_color")]
                if inst_buf is not None:
                    content.append((inst_buf, "4f 4f 4f/i", "in_i0", "in_i1", "in_i2"))
                self.vaos[key] = self.eng.ctx.vertex_array(prog, content, self.ibo, skip_errors=True)
            return self.vaos[key]

        def release(self):
            for v in self.vaos.values():
                v.release()
            self.vaos.clear()
            self.vbo.release()
            self.ibo.release()

    def _build_static_geometry(self):
        M = lambda mesh: RenderEngine._Mesh(self, mesh)
        self.m_terrain = [M(m) for m in sa.build_terrain()]
        self.m_creek = M(sa.build_creek())
        pit_y = float(terrain.height(0.0, 0.0))
        self.m_ring = M(mg.create_hearth_ring_mesh(ground_fn=lambda x, z: float(terrain.height(x, z)) - pit_y))
        self.m_log = M(mg.create_log_mesh())
        self.m_split = M(mg.create_split_fuel_mesh())          # riven firewood wedge
        self._split_off = G.translate(-mg.split_fuel_centroid(), 0.0, 0.0)
        # a felled tree's limbs (tapered, base at 0) and their foliage sprays
        self.m_branch = M(mg.create_frustum(1.0, 0.62, 1.0, 8, 2, (1, 1, 1, 1), caps=True, cap_alpha=0.0))
        self.m_branch_tip = M(mg.create_frustum(1.0, 0.12, 1.0, 8, 3, (1, 1, 1, 1), caps=False))
        self.m_limb_foliage = {k: M(mg.create_limb_foliage(k, i)) for i, k in enumerate(("pine", "cedar", "oak", "birch"))}
        self._fallen_cache: Dict[Any, Any] = {}
        self.m_nest = M(mg.create_tinder_nest_mesh())
        self.m_woodpile = M(mg.create_woodpile_mesh())
        self.m_rack = M(mg.create_drying_rack_mesh())
        self.m_tarp = M(mg.create_tarp_mesh())
        self.m_needles = M(mg.create_needle_pile())
        self.m_curls = M(mg.create_birch_curls())
        self.m_cattails = M(mg.create_cattails())
        self.m_grass_clump = M(mg.create_grass_tuft(seed=77, color=(0.62, 0.52, 0.28, 1.0), blades=18, height=0.45))
        self.m_kindling_bundle = M(mg.create_kindling_bundle())
        self.m_cedar_bundle = M(mg.create_kindling_bundle(color=(0.46, 0.28, 0.20, 1.0), n=8))
        self.m_hearthboard = M(mg.create_hearthboard_kit(False))
        self.m_hearthboard_notched = M(mg.create_hearthboard_kit(True, 1.0))
        self.m_bow = M(mg.create_bow_mesh())
        self.m_unit_cyl = M(mg.create_frustum(1.0, 1.0, 1.0, 10, 1, (1, 1, 1, 1), centered=True))
        self.m_sphere = M(mg.create_icosphere(2, 1.0, (1, 1, 1, 1)))
        self.m_stone = M(mg.create_rock(seed=4, color=(0.55, 0.55, 0.53, 1.0), flatten=0.55))
        # felled stump for the fuel shader: root flare, bark sides, sawn / chopped top (end grain)
        self.m_stump_fuel = M(mg.create_frustum(1.22, 1.0, 1.0, 14, 3, (1, 1, 1, 1), caps=True, cap_alpha=0.0))
        self.m_fungus = M(mg.create_fungus_bracket())
        self.m_palm = M(mg.create_palm())
        self.m_glove = {k: M(v) for k, v in mg.create_glove_parts().items()}
        self.m_finger = M(mg.create_finger_segment())
        self.m_finger_tip = M(mg.create_finger_segment(tip=True))
        self.m_forearm = M(mg.create_frustum(1.0, 0.85, 1.0, 10, 1, (1, 1, 1, 1), caps=False))
        self.m_lighter = M(mg.create_lighter_mesh())
        self.m_knife = [M(mg.create_knife_mesh(False)), M(mg.create_knife_mesh(True))]
        self.m_axe = [M(mg.create_axe_mesh(False)), M(mg.create_axe_mesh(True))]
        self.m_gas = M(mg.create_gas_can_mesh())
        self.m_backpack = M(mg.create_backpack_mesh())
        self.m_seat = M(mg.create_frustum(0.19, 0.17, 1.6, 12, 2, (0.24, 0.19, 0.14, 1.0), centered=True, cap_alpha=1.0))
        # procedural leaf / needle spray atlas for the foliage cards
        from firestarter.rendering import textures as TX
        atlas = TX.foliage_atlas()
        self.leaf_tex = self.ctx.texture((atlas.shape[1], atlas.shape[0]), 4, atlas.tobytes())
        self.leaf_tex.build_mipmaps()
        self.leaf_tex.filter = (self.mgl.LINEAR_MIPMAP_LINEAR, self.mgl.LINEAR)
        self.leaf_tex.anisotropy = 4.0
        # vegetation meshes
        tm = sa.build_tree_meshes()
        self.tree_meshes = {}
        for (sp, v), (trunk, foliage) in tm.items():
            self.tree_meshes[(sp, "trunk" if sp != "dead" else "dead", v)] = M(trunk)
            if foliage is not None:
                self.tree_meshes[(sp, "foliage", v)] = M(foliage)
        for (sp, v), far in sa.build_far_foliage_meshes().items():
            self.tree_meshes[(sp, "foliage_far", v)] = M(far)
        self.veg_meshes = {
            "bush": M(mg.create_bush(seed=1)),
            "fern": M(mg.create_fern(seed=2)),
            "grass": M(mg.create_grass_tuft(seed=3, color=(1.0, 1.0, 1.0, 1.0))),
            "rock": M(mg.create_rock(seed=5)),
            "dead_log": M(mg.create_frustum(1.0, 0.9, 1.0, 9, 2, (0.30, 0.27, 0.22, 1.0), centered=True)),
        }
        self.veg_scatter = sa.scatter_vegetation(grass_density=self.preset["grass"])
        self.veg_inst_bufs = {}
        for k, arr in self.veg_scatter.items():
            if len(arr):
                self.veg_inst_bufs[k] = (self.ctx.buffer(arr.tobytes()), len(arr))
        self.tree_inst_bufs: Dict[Tuple, Tuple[Any, int]] = {}
        self.fungus_buf = None
        self.far_raw = sa.far_forest_instances()
        self.far_key = None
        self.far_bufs: Dict[Tuple, Tuple[Any, int]] = {}

    def _build_dynamic_buffers(self):
        ctx = self.ctx
        corners = np.array([[-1, 0], [1, 0], [1, 1], [-1, 0], [1, 1], [-1, 1]], dtype=np.float32)
        self.flame_quad = ctx.buffer(corners.tobytes())
        self.flame_buf = ctx.buffer(reserve=MAX_FLAMES * FLAME_STRIDE * 4)
        self.flame_vao = ctx.vertex_array(self.p_flame, [(self.flame_quad, "2f", "in_corner"),
                                                         (self.flame_buf, "4f 4f 4f/i", "in_f0", "in_f1", "in_f2")], skip_errors=True)
        pc = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], dtype=np.float32)
        self.part_quad = ctx.buffer(pc.tobytes())
        self.part_buf = ctx.buffer(reserve=MAX_PARTICLES * 12 * 4)
        self.part_vao = ctx.vertex_array(self.p_part, [(self.part_quad, "2f", "in_corner"),
                                                       (self.part_buf, "4f 4f 4f/i", "in_p0", "in_p1", "in_p2")], skip_errors=True)

    def _create_targets(self):
        ctx = self.ctx
        rs = self.preset["render_scale"]
        self.rw = max(64, int(self.width * rs))
        self.rh = max(64, int(self.height * rs))
        self.scene_tex = ctx.texture((self.rw, self.rh), 4, dtype="f2")
        self.scene_tex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        self.scene_tex.repeat_x = self.scene_tex.repeat_y = False
        self.scene_depth = ctx.depth_texture((self.rw, self.rh))
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene_tex], depth_attachment=self.scene_depth)
        self.depth_only_fbo = ctx.framebuffer(depth_attachment=self.scene_depth)
        self.depth_copy = ctx.depth_texture((self.rw, self.rh))
        self.depth_copy.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
        self.depth_copy.compare_func = ""     # sample raw depth (moderngl enables comparison by default)
        self._depth_copy_color = ctx.texture((self.rw, self.rh), 4, dtype="f2")
        self.depth_copy_fbo = ctx.framebuffer(color_attachments=[self._depth_copy_color], depth_attachment=self.depth_copy)
        self.bloom_levels = []
        w, h = self.rw, self.rh
        for i in range(6):
            w, h = max(4, w // 2), max(4, h // 2)
            t = ctx.texture((w, h), 4, dtype="f2")
            t.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
            t.repeat_x = t.repeat_y = False
            self.bloom_levels.append((t, ctx.framebuffer(color_attachments=[t])))
        self.ldr_tex = ctx.texture((self.width, self.height), 4)
        self.ldr_tex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        self.ldr_fbo = ctx.framebuffer(color_attachments=[self.ldr_tex])
        ss = self.preset["shadow_size"]
        self.shadow_tex = ctx.depth_texture((ss, ss))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.ui_texture = ctx.texture((self.width, self.height), 4)
        self.ui_texture.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
        if self.standalone or self.target_fbo is not None:
            self.out_tex = ctx.texture((self.width, self.height), 4)
            self.out_fbo = ctx.framebuffer(color_attachments=[self.out_tex])
        else:
            self.out_fbo = None

    def set_preset(self, name: str):
        if name == self.preset_name or name not in GRAPHICS_PRESETS:
            return
        self.preset_name = name
        self.preset = dict(GRAPHICS_PRESETS[name])
        self.particle_system.quality = self.preset["particles"]
        if self.is_gl_available:
            self._release_targets()
            self._create_targets()
            self.veg_scatter = sa.scatter_vegetation(grass_density=self.preset["grass"])
            for k, (buf, n) in list(self.veg_inst_bufs.items()):
                buf.release()
            self.veg_inst_bufs = {}
            for k, arr in self.veg_scatter.items():
                if len(arr):
                    self.veg_inst_bufs[k] = (self.ctx.buffer(arr.tobytes()), len(arr))
            for m in self.veg_meshes.values():
                m.vaos.clear()

    def _release_targets(self):
        for obj in [self.scene_tex, self.scene_depth, self.scene_fbo, self.depth_only_fbo, self.depth_copy, self.depth_copy_fbo, self._depth_copy_color,
                    self.ldr_tex, self.ldr_fbo, self.shadow_tex, self.shadow_fbo, self.ui_texture]:
            try:
                obj.release()
            except Exception:
                pass
        for t, f in self.bloom_levels:
            t.release()
            f.release()

    def handle_resize(self, width: int, height: int):
        self.width = max(160, int(width))
        self.height = max(120, int(height))
        self.ui_renderer.resize(self.width, self.height)
        self._ui_sig = None
        if self.is_gl_available:
            self._release_targets()
            self._create_targets()

    # ================================================================== uniforms
    @staticmethod
    def _set(prog, name: str, val):
        if name not in prog:
            return
        u = prog[name]
        if isinstance(val, (bytes, bytearray)):
            u.write(val)
        elif isinstance(val, np.ndarray):
            if val.ndim == 2:
                u.write(G.to_gl(val))
            else:
                v = tuple(float(x) for x in val)
                u.value = v
        elif isinstance(val, (list, tuple)) and len(val) and isinstance(val[0], (list, tuple, np.ndarray)):
            u.value = [tuple(float(x) for x in v) for v in val]
        else:
            u.value = val

    def _common_uniforms(self, prog, frame):
        L = frame["lighting"]
        st = self._set
        st(prog, "u_view", frame["view"])
        st(prog, "u_proj", frame["proj"])
        st(prog, "u_cam_pos", frame["cam_pos"])
        st(prog, "u_time", float(self.time))
        st(prog, "u_sun_dir", L.light_dir)
        st(prog, "u_sun_color", L.light_color)
        st(prog, "u_sky_amb", L.sky_amb)
        st(prog, "u_ground_amb", L.ground_amb)
        st(prog, "u_fire_pos", frame["fire_pos"])
        st(prog, "u_fire_color", frame["fire_col"])
        st(prog, "u_fire_intensity", float(frame["fire_int"]))
        st(prog, "u_fire_range", float(frame["fire_range"]))
        st(prog, "u_light2_pos", frame["l2_pos"])
        st(prog, "u_light2_color", frame["l2_col"])
        st(prog, "u_light2_intensity", float(frame["l2_int"]))
        st(prog, "u_fog_color", L.fog_color)
        st(prog, "u_fog_sun_color", L.fog_sun)
        st(prog, "u_fog_density", float(L.fog_density))
        st(prog, "u_fog_height", 0.045)
        st(prog, "u_snow", float(frame["snow"]))
        st(prog, "u_wetness", float(frame["wet"]))
        st(prog, "u_wind", frame["wind"])
        st(prog, "u_light_vp", frame["light_vp"])
        st(prog, "u_shadows", 1 if frame["shadows"] else 0)
        if "u_shadow_map" in prog:
            prog["u_shadow_map"].value = 4
        if "u_leaf_tex" in prog:
            prog["u_leaf_tex"].value = 6

    # ================================================================== main render
    def render(self, world: Any, player: Any, hands: Any, smartphone: Any, ref_book: Any, input_mgr: Any,
               dt: float = 1.0 / 60.0, extra_ui: Optional[Dict[str, Any]] = None):
        if not self.is_gl_available or not self.ctx:
            return
        t0 = _time.perf_counter()
        self.time += dt
        self.frame += 1
        ctx = self.ctx
        frame = self._frame_state(world, player, hands, dt)
        self._update_fx(world, player, hands, frame, dt)

        # 1. shadows
        if frame["shadows"]:
            self._shadow_pass(world, frame)

        # 2. HDR scene
        self.scene_fbo.use()
        ctx.viewport = (0, 0, self.rw, self.rh)
        ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        self.shadow_tex.use(location=4)
        self.leaf_tex.use(location=6)
        self._sky_pass(frame)
        ctx.enable(self.mgl.DEPTH_TEST)
        ctx.disable(self.mgl.BLEND)
        ctx.enable(self.mgl.CULL_FACE)
        self._terrain_pass(frame, world)
        self._vegetation_pass(frame, world)
        self._camp_pass(frame, world, player)
        self._fuel_pass(frame, world)
        self._fallen_tree_pass(frame, world)
        # depth copy for soft particles, flames and heat shimmer
        ctx.copy_framebuffer(self.depth_copy_fbo, self.scene_fbo)
        self.scene_fbo.use()
        self._transparent_pass(frame, world)
        # 3. viewmodel
        self.scene_fbo.depth_mask = True
        ctx.enable(self.mgl.DEPTH_TEST)
        self._clear_depth()
        self._viewmodel_pass(frame, world, player, hands)
        # 4. post
        self._post_pass(frame, world)
        # 5. UI
        self._ui_pass(world, player, hands, smartphone, ref_book, input_mgr, extra_ui)
        self.last_frame_ms = (_time.perf_counter() - t0) * 1000.0

    def _clear_depth(self):
        # clear only the depth buffer via a depth-only framebuffer sharing the same attachment
        self.depth_only_fbo.use()
        self.ctx.clear(depth=1.0)
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, self.rw, self.rh)

    # ================================================================== frame state
    def _frame_state(self, world, player, hands, dt) -> Dict[str, Any]:
        aspect = self.width / float(self.height)
        cam = player.eye if hasattr(player, "eye") else player.position
        fwd = player.get_forward_ray()
        view = G.look_at(cam, cam + fwd)
        proj = G.perspective(FOV_DEGREES, aspect, CAMERA_NEAR, CAMERA_FAR)
        vm_proj = G.perspective(VM_FOV, aspect, VM_NEAR, 10.0)
        grid = world.thermal_grid
        fire_dist = float(np.linalg.norm(np.asarray(cam) - grid.pit_center))
        L = self.atmosphere.compute(world.weather, dt, grid.current_heat_flux / 1000.0, fire_dist)
        fpos, fcol, fint, frng = self.atmosphere.fire_light(grid, self.time, dt)
        if getattr(world, "gas_flash_timer", 0.0) > 0:
            fint += 6.0 * world.gas_flash_timer
            frng = 25.0
        # secondary light: lighter flame / nest glow / smouldering fungus
        l2_pos, l2_col, l2_int = np.zeros(3), np.zeros(3), 0.0
        right, up, f = G.view_basis(view)
        hm = hands.get_hand_render_matrices() if hands is not None else None
        cf = world.crafting
        if hm is not None and cf.lighter_is_lit and player.primary_hand == "pocket_lighter":
            p = cam + right * hm["right_pos"][0] + up * (hm["right_pos"][1] + 0.07) + f * hm["right_pos"][2]
            l2_pos, l2_col, l2_int = p, np.array([1.0, 0.62, 0.25]), 0.55 * (1.0 + 0.1 * math.sin(self.time * 30.0))
        nest = cf.active_tinder_nest
        if nest is not None and nest.has_ember and hm is not None:
            p = cam + up * -0.14 + f * 0.3
            g = 0.05 + 0.5 * nest.glow_intensity
            l2_pos, l2_col, l2_int = p, np.array([1.0, 0.35, 0.08]), g
        if cf.bow_drill.has_ember:
            l2_pos, l2_col, l2_int = world.hearthboard_pos + np.array([0.05, 0.03, 0.03]), np.array([1.0, 0.3, 0.06]), 0.12
        # shadow camera
        light_vp = np.identity(4, dtype=np.float32)
        shadows = bool(self.preset["shadows"]) and float(np.max(L.light_color)) > 0.03
        if shadows:
            ld = np.asarray(L.light_dir, dtype=float)
            ld = ld / np.linalg.norm(ld)
            if ld[1] < 0.08:
                ld = np.array([ld[0], 0.08, ld[2]])
                ld /= np.linalg.norm(ld)
            center = np.array([cam[0], terrain.height(cam[0], cam[2]), cam[2]]) + fwd * np.array([8.0, 0.0, 8.0])
            lv = G.look_at(center + ld * 80.0, center)
            ext = 30.0
            light_vp = G.orthographic(-ext, ext, -ext, ext, 1.0, 180.0) @ lv
        w = world.weather
        return {
            "view": view, "proj": proj, "vm_proj": vm_proj, "cam_pos": np.asarray(cam, dtype=float), "fwd": fwd,
            "right": right, "up": up, "lighting": L, "fire_pos": fpos, "fire_col": fcol, "fire_int": fint,
            "fire_range": frng, "l2_pos": l2_pos, "l2_col": l2_col, "l2_int": l2_int, "light_vp": light_vp,
            "shadows": shadows, "snow": min(1.0, w.snow_accumulation_m / 0.2), "wet": w.ground_wetness * 0.8,
            "wind": w.wind_vector, "season": w.current_season.value, "hm": hm, "aspect": aspect,
        }

    # ================================================================== passes
    def _shadow_pass(self, world, frame):
        ctx = self.ctx
        self.shadow_fbo.use()
        ss = self.preset["shadow_size"]
        ctx.viewport = (0, 0, ss, ss)
        ctx.clear(depth=1.0)
        ctx.enable(self.mgl.DEPTH_TEST)
        ctx.enable(self.mgl.CULL_FACE)
        ctx.front_face = "ccw"
        vp = frame["light_vp"]
        p = self.p_shadow_inst
        self.leaf_tex.use(location=6)
        self._set(p, "u_light_vp", vp)
        if "u_leaf_tex" in p:
            p["u_leaf_tex"].value = 6
        ctx.disable(self.mgl.CULL_FACE)
        self._ensure_tree_instances(world, frame["season"])
        for key, (buf, n) in self.tree_inst_bufs.items():
            mesh = self.tree_meshes.get(key)
            if mesh is not None and n:
                self._set(p, "u_alpha_test", 1 if (key[1] == "foliage" and mg.CARD_FOLIAGE) else 0)
                mesh.vao(p, buf).render(instances=n)
        for k in ("rock", "bush", "dead_log"):
            if k in self.veg_inst_bufs:
                buf, n = self.veg_inst_bufs[k]
                self._set(p, "u_alpha_test", 1 if (k == "bush" and mg.CARD_FOLIAGE) else 0)
                self.veg_meshes[k].vao(p, buf).render(instances=n)
        ps = self.p_shadow
        self._set(ps, "u_light_vp", vp)
        self._set(ps, "u_alpha_test", 0)
        for model, mesh in self._camp_static_models(world):
            self._set(ps, "u_model", model)
            mesh.vao(ps).render()
        for b in world.thermal_grid.bodies:
            if b.geometry_type == "tinder":
                self._set(ps, "u_model", self._body_model(b))
                self.m_nest.vao(ps).render()
                continue
            mesh, m = self._piece_mesh(getattr(b, "source_item", None), self._body_model(b))
            self._set(ps, "u_model", m)
            mesh.vao(ps).render()
        for ft in world.fallen_trees:
            parts = self._fallen_parts(ft, frame["season"])
            if parts["bole"] is not None:
                self._set(ps, "u_model", parts["frame"])
                parts["bole"].vao(ps).render()
            for model, _, _, tip in parts["branch"]:
                self._set(ps, "u_model", model)
                (self.m_branch_tip if tip else self.m_branch).vao(ps).render()
        ctx.enable(self.mgl.CULL_FACE)

    def _sky_pass(self, frame):
        ctx = self.ctx
        L = frame["lighting"]
        p = self.p_sky
        ctx.disable(self.mgl.DEPTH_TEST)
        ctx.disable(self.mgl.BLEND)
        inv_vp = np.linalg.inv(frame["proj"] @ frame["view"])
        self._set(p, "u_inv_vp", inv_vp)
        self._set(p, "u_cam_pos", frame["cam_pos"])
        self._set(p, "u_zenith", L.zenith)
        self._set(p, "u_horizon", L.horizon)
        self._set(p, "u_ground_col", L.ground_col)
        self._set(p, "u_sun_dir2", L.sun_dir)
        self._set(p, "u_moon_dir", L.moon_dir)
        self._set(p, "u_sun_disc_col", L.sun_disc)
        self._set(p, "u_cloud_cover", float(self._weather_cloud))
        self._set(p, "u_night", float(L.night))
        self._set(p, "u_time2", float(self.time))
        self._set(p, "u_cloud_offset", (self.time * 0.004 + 3.0, self.time * 0.0025))
        self._set(p, "u_cloud_lit", L.cloud_lit)
        self._set(p, "u_cloud_shadow", L.cloud_shadow)
        self._set(p, "u_moon_intensity", float(L.moon_intensity))
        self._set(p, "u_sun_dir", L.light_dir)
        self.empty_vao["sky"].render(self.mgl.TRIANGLES, vertices=3)
        ctx.enable(self.mgl.DEPTH_TEST)

    def _terrain_pass(self, frame, world):
        p = self.p_terrain
        self._common_uniforms(p, frame)
        gc, gd = sa.grass_season_colors(frame["season"], frame["snow"])
        self._set(p, "u_grass_col", gc)
        self._set(p, "u_grass_dry", gd)
        self._set(p, "u_pit_center", world.thermal_grid.pit_center)
        g = world.thermal_grid
        self._set(p, "u_ash_amount", float(min(1.0, 0.25 + g.coal_bed.ash_kg * 4.0 + (0.5 if g.total_energy_produced_joules > 1e5 else 0.0))))
        for m in self.m_terrain:
            m.vao(p).render()

    def _ensure_tree_instances(self, world, season):
        eco = world.ecosystem
        key = (eco.version, season)
        if key == self._veg_key:
            return
        self._veg_key = key
        for buf, n in self.tree_inst_bufs.values():
            buf.release()
        self.tree_inst_bufs = {}
        groups = sa.tree_instances(eco.trees, season)
        for (sp, part, v), arr in groups.items():
            if len(arr) == 0:
                continue
            mesh_key = (sp, part, v)
            self.tree_inst_bufs[mesh_key] = (self.ctx.buffer(arr.tobytes()), len(arr))
        for m in self.tree_meshes.values():
            m.vaos.clear()
        fi = sa.fungus_instances(eco.trees)
        if self.fungus_buf is not None:
            self.fungus_buf[0].release()
        self.fungus_buf = (self.ctx.buffer(fi.tobytes()), len(fi)) if len(fi) else None
        self.m_fungus.vaos.clear()

    def _ensure_far_forest(self, season):
        if self.far_key == season:
            return
        self.far_key = season
        for buf, n in self.far_bufs.values():
            buf.release()
        self.far_bufs = {}
        for (sp, part, v), arr in self.far_raw.items():
            a = arr.copy()
            if part == "foliage":
                tint = sa.SEASON_FOLIAGE[sp].get(season, (1.0, 1.0, 1.0))
                if tint is None:
                    continue
                a[:, 8:11] *= np.array(tint, dtype=np.float32)
            self.far_bufs[(sp, part, v)] = (self.ctx.buffer(a.tobytes()), len(a))
        for m in self.tree_meshes.values():
            m.vaos.clear()

    def _vegetation_pass(self, frame, world):
        ctx = self.ctx
        self._ensure_tree_instances(world, frame["season"])
        p = self.p_obj_inst
        self._common_uniforms(p, frame)
        self._set(p, "u_model", np.identity(4, dtype=np.float32))
        self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
        self._set(p, "u_emissive", 0.0)
        self._set(p, "u_emit_color", (0.0, 0.0, 0.0))
        self._set(p, "u_highlight", 0.0)
        ctx.disable(self.mgl.CULL_FACE)
        for (sp, part, v), (buf, n) in self.tree_inst_bufs.items():
            mesh = self.tree_meshes.get((sp, part, v))
            if mesh is None:
                continue
            if part == "foliage":
                self._set(p, "u_material", 13 if mg.CARD_FOLIAGE else 2)
                self._set(p, "u_sway", 1.0)
                self._set(p, "u_snow_cap", 1.0 if sp in ("pine", "cedar") else 0.5)
            else:
                self._set(p, "u_material", 8 if sp == "birch" else 1)
                self._set(p, "u_sway", 0.35)
                self._set(p, "u_snow_cap", 0.3)
            mesh.vao(p, buf).render(instances=n)
        self._ensure_far_forest(frame["season"])
        for (sp, part, v), (buf, n) in self.far_bufs.items():
            mesh = self.tree_meshes.get((sp, "foliage_far" if part == "foliage" else part, v)) or self.tree_meshes.get((sp, part, v))
            if mesh is None:
                continue
            self._set(p, "u_material", (13 if mg.CARD_FOLIAGE else 2) if part == "foliage" else (8 if sp == "birch" else 1))
            self._set(p, "u_sway", 0.6 if part == "foliage" else 0.2)
            self._set(p, "u_snow_cap", 1.0)
            mesh.vao(p, buf).render(instances=n)
        for k, mat, sway, snowcap in (("bush", 13 if mg.CARD_FOLIAGE else 2, 0.8, 0.8), ("fern", 2, 1.0, 0.3), ("grass", 11, 1.0, 0.2),
                                      ("rock", 3, 0.0, 1.0), ("dead_log", 1, 0.0, 1.0)):
            if k not in self.veg_inst_bufs:
                continue
            if k == "fern" and frame["season"] == "Winter":
                continue
            buf, n = self.veg_inst_bufs[k]
            self._set(p, "u_material", mat)
            self._set(p, "u_sway", sway)
            self._set(p, "u_snow_cap", snowcap)
            tint = (1.0, 1.0, 1.0, 1.0)
            if k in ("fern", "bush"):
                tint = {"Summer": (1.0, 1.0, 1.0, 1), "Autumn": (1.45, 1.05, 0.7, 1), "Winter": (1.2, 1.05, 0.85, 1),
                        "Spring": (0.95, 1.15, 0.95, 1)}[frame["season"]]
            elif k == "grass":
                gc, gd = sa.grass_season_colors(frame["season"], frame["snow"])
                tint = (gc[0] * 1.05, gc[1] * 1.05, gc[2] * 1.05, 1.0)
            self._set(p, "u_color", tint)
            self.veg_meshes[k].vao(p, buf).render(instances=n)
        self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
        if self.fungus_buf is not None:
            self._set(p, "u_material", 9)
            self._set(p, "u_sway", 0.0)
            self._set(p, "u_snow_cap", 1.0)
            self.m_fungus.vao(p, self.fungus_buf[0]).render(instances=self.fungus_buf[1])
        # trees caught mid-fall (individual; stumps and felled stems are drawn by _fallen_tree_pass)
        po = self.p_obj
        self._common_uniforms(po, frame)
        self._set(po, "u_emissive", 0.0)
        self._set(po, "u_emit_color", (0.0, 0.0, 0.0))
        self._set(po, "u_sway", 0.0)
        self._set(po, "u_highlight", 0.0)
        self._set(po, "u_snow_cap", 1.0)
        for t in world.ecosystem.trees:
            if t.stage == TreeGrowthStage.STUMP or t.is_falling:
                if abs(t.position[0] - frame["cam_pos"][0]) > 90 or abs(t.position[2] - frame["cam_pos"][2]) > 90:
                    continue
                key = sa.SPECIES_KEY.get(t.species, "pine")
                if t.is_falling:          # once down it becomes a FallenTree (drawn by _fallen_tree_pass)
                    nominal = sa.NOMINAL_CONIFER_H if key in ("pine", "cedar") else sa.NOMINAL_DECID_H
                    s = t.height_m / nominal
                    axis = np.array([-math.sin(t.fall_direction), 0.0, math.cos(t.fall_direction)])
                    ang = t.fall_angle
                    # rotate about the horizontal axis perpendicular to the fall heading
                    R = self._axis_angle(axis, -ang)
                    m = G.translate(*(t.position + np.array([0, 0.4, 0]))) @ R @ G.rot_y(-t.yaw) @ G.scale(s, s, s)
                    self._set(po, "u_model", m)
                    self._set(po, "u_material", 8 if key == "birch" else 1)
                    self.tree_meshes[(key, "trunk", t.variant)].vao(po).render()
                    fol = self.tree_meshes.get((key, "foliage", t.variant))
                    if fol is not None and frame["season"] != "Winter" or (fol is not None and key in ("pine", "cedar")):
                        self._set(po, "u_material", 13 if mg.CARD_FOLIAGE else 2)
                        fol.vao(po).render()
        ctx.enable(self.mgl.CULL_FACE)

    def _bole_frame(self, ft) -> np.ndarray:
        """Model matrix of a felled stem: origin at the butt centre, +Y along the stem, +Z up."""
        Y = ft.axis
        Z = np.array([0.0, 1.0, 0.0]) - Y * Y[1]
        Z /= np.linalg.norm(Z)
        X = np.cross(Y, Z)
        m = np.identity(4, dtype=np.float32)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = X, Y, Z, ft.point_at(0.0, terrain.height)
        return m

    def _bole_runs(self, ft):
        """Contiguous uncut pieces of the stem as rings (y, r, notch) for create_bole_mesh."""
        k = math.sqrt(1.0 + (ft.rest[1] if ft.rest else 0.0) ** 2)
        kerf = 0.007
        runs, cur = [], None

        def ring(sv, notch=0.0):
            cur["rings"].append((sv * k, max(0.004, ft.radius_at(sv)), notch))

        def close(cap1=True):
            nonlocal cur
            if cur and len(cur["rings"]) >= 2:
                cur["cap1"] = cap1
                runs.append(cur)
            cur = None

        n = ft.n_sections
        for i in range(n):
            if ft.removed[i]:
                close()
                continue
            s0, s1 = i * SECTION_LEN, (i + 1) * SECTION_LEN
            if cur is None or ft.cut[i]:
                close()
                cur = {"rings": [], "cap0": True}
                ring(s0 + (kerf if i > 0 else 0.0))
            # a kerf being opened at the far end of this round shows as a V-notch
            d = float(ft.cut_depth[i + 1]) if not ft.cut[i + 1] else 0.0
            if d > 0.0:
                w = min(0.14, d * 0.8) + 0.006
                ring(s1 - w)
                ring(s1, min(d, 1.9 * ft.radius_at(s1)))
                if i + 1 < n or not ft.top_taken:
                    ring(s1 + w)
            else:
                ring(s1 - (kerf if ft.cut[i + 1] else 0.0))
            if ft.cut[i + 1]:
                close()
        # the whippy leader beyond the last round
        if not ft.top_taken:
            if cur is None:
                cur = {"rings": [], "cap0": True}
                ring(ft.bole_length + kerf)
            for sv in np.linspace(ft.bole_length + 0.4, ft.height * 0.995, max(2, int((ft.height - ft.bole_length) / 0.5))):
                ring(float(sv))
            close(cap1=False)
        close()
        return runs, ft.height * k

    def _fallen_parts(self, ft, season):
        """Everything drawn for one felled tree: the stem mesh (rebuilt only when it is cut),
        the limbs still attached (with a few side shoots) and their foliage sprays."""
        key = (ft.cut.tobytes(), np.round(ft.cut_depth, 3).tobytes(), ft.removed.tobytes(),
               tuple(l.removed for l in ft.limbs), ft.top_taken, season)
        hit = self._fallen_cache.get(id(ft))
        if hit is not None and hit[0] == key:
            return hit[1]
        if hit is not None and hit[1]["bole"] is not None:
            hit[1]["bole"].release()
        runs, total = self._bole_runs(ft)
        md = mg.create_bole_mesh(runs, total)
        frame_m = self._bole_frame(ft)
        parts = {"frame": frame_m, "bole": RenderEngine._Mesh(self, md) if md.index_count else None,
                 "branch": [], "foliage": []}
        H = terrain.height
        key_sp = sa.SPECIES_KEY.get(ft.bark_species, "pine")
        tint = sa.SEASON_FOLIAGE[key_sp].get(season, (1.0, 1.0, 1.0))
        if not ft.top_taken and tint is not None and key_sp in ("pine", "cedar"):
            pa = ft.point_at(ft.bole_length, H)
            L = ft.height - ft.bole_length
            parts["foliage"].append((G.translate(*pa) @ G.align_y_to(ft.axis) @ G.scale(L * 1.3, L, L * 1.3), tint, "top"))
        for limb in ft.limbs:
            if limb.removed:
                continue
            rng = np.random.default_rng(limb.id * 7919 + ft.tree_id)     # stable as other limbs come off
            d = ft.limb_direction(limb)
            a = ft.point_at(limb.s, H) + d * ft.radius_at(limb.s) * 0.5
            tip = a + d * limb.length
            # a limb can't pass through the soil: it is bent up to lie along the ground
            g = float(H(tip[0], tip[2])) + 0.06
            if tip[1] < g:
                tip[1] = g
                d = (tip - a) / max(1e-6, np.linalg.norm(tip - a))
            rr = max(0.008, limb.radius)
            roll = limb.id * 57.0 % 360
            # a limb is drawn in two pieces with a kink: real branches curve up towards the light
            side = np.cross(d, np.array([0.0, 1.0, 0.0]))
            if np.linalg.norm(side) < 1e-3:
                side = ft.side.copy()
            side /= np.linalg.norm(side)
            L1 = limb.length * 0.5
            mid = a + d * L1
            d2 = d + np.array([0.0, float(rng.uniform(0.15, 0.45)), 0.0]) + side * float(rng.uniform(-0.3, 0.3))
            d2 /= np.linalg.norm(d2)
            L2 = limb.length * 0.5
            end = mid + d2 * L2
            g = float(H(end[0], end[2])) + 0.05
            if end[1] < g:
                end[1] = g
                d2 = (end - mid) / max(1e-6, np.linalg.norm(end - mid))
            parts["branch"].append((G.orientation_model(a, d, roll, rr, L1 + rr * 0.3, rr), rr, limb, 0))
            parts["branch"].append((G.orientation_model(mid, d2, roll, rr * 0.62, L2, rr * 0.62), rr, limb, 1))
            # side shoots
            for j in range(4):
                f = 0.25 + 0.18 * j + float(rng.uniform(-0.05, 0.05))
                sgn = 1.0 if (j + limb.id) % 2 else -1.0
                base_d = d if f < 0.5 else d2
                sp = a + d * limb.length * f if f < 0.5 else mid + d2 * limb.length * (f - 0.5)
                sd = base_d * 0.7 + side * sgn * 0.65 + np.array([0.0, float(rng.uniform(-0.05, 0.4)), 0.0])
                sd /= np.linalg.norm(sd)
                sl = limb.length * 0.36 * (1.05 - f)
                sr = max(0.005, rr * (1.0 - f) * 0.5)
                parts["branch"].append((G.orientation_model(sp, sd, 0.0, sr, sl, sr), sr, limb, 1))
            if tint is not None:
                chord = end - a
                cl = float(np.linalg.norm(chord))
                parts["foliage"].append((G.translate(*a) @ G.align_y_to(chord / max(cl, 1e-6)) @ G.rot_y(roll) @
                                         G.scale(cl, cl, cl), tint, limb))
        self._fallen_cache[id(ft)] = (key, parts)
        return parts

    def _stump_pass(self, frame, world):
        p = self.p_fuel
        self._common_uniforms(p, frame)
        self._set(p, "u_sway", 0.0)
        cam = frame["cam_pos"]
        amb = world.weather.ambient_temp_c
        z3 = [0.0] * 3
        for t in world.ecosystem.trees:
            if not (t.stage == TreeGrowthStage.STUMP or t.is_falling):
                continue
            if abs(t.position[0] - cam[0]) > 90 or abs(t.position[2] - cam[2]) > 90:
                continue
            r = t.trunk_radius_m
            # old stumps (not felled this game) are weathered grey and damp
            wet = 0.15 if t.is_fallen or t.is_falling else 0.45
            self._fuel_uniforms(p, t.species, [amb] * 3, z3, z3, z3, [wet] * 3, False, 0.0)
            self._set(p, "u_model", G.translate(t.position[0], t.position[1] - 0.08, t.position[2]) @ G.rot_y(t.id * 33.0 % 360)
                      @ G.scale(r, 0.5, r))
            self.m_stump_fuel.vao(p).render()

    def _fallen_tree_pass(self, frame, world):
        self._stump_pass(frame, world)
        live = {id(ft) for ft in world.fallen_trees}
        for k in [k for k in self._fallen_cache if k not in live]:
            old = self._fallen_cache.pop(k)[1]
            if old["bole"] is not None:
                old["bole"].release()
        if not world.fallen_trees:
            return
        tgt = world.target
        p = self.p_fuel
        self._common_uniforms(p, frame)
        self._set(p, "u_sway", 0.0)
        amb = world.weather.ambient_temp_c
        markers = []
        for ft in world.fallen_trees:
            parts = self._fallen_parts(ft, frame["season"])
            wet = [min(1.0, ft.m_surface)] * 3
            on_trunk = bool(tgt and tgt["obj"] is ft and tgt["kind"] == "fallen_trunk")
            hl_l = tgt["limb"] if (tgt and tgt["obj"] is ft and tgt["kind"] == "fallen_limb") else None
            self._fuel_uniforms(p, ft.bark_species, [amb] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3, wet, False,
                                0.22 if on_trunk else 0.0)
            if parts["bole"] is not None:
                self._set(p, "u_model", parts["frame"])
                parts["bole"].vao(p).render()
            if on_trunk:
                i_cut = ft.nearest_boundary(min(tgt["s"], ft.bole_length))
                if i_cut is not None:
                    s_cut = i_cut * SECTION_LEN
                    markers.append((ft, s_cut, ft.blocking_limb(s_cut) is not None))
            # young branches are darker than the trunk (birch twigs are red-brown, not white)
            if ft.bark_species == WoodSpeciesType.PAPER_BIRCH:
                self._set(p, "u_bark_col", (0.30, 0.19, 0.15))
                self._set(p, "u_bark_style", 0.0)
            for model, r, ref, tip in parts["branch"]:
                self._set(p, "u_highlight", 0.5 if (hl_l is not None and ref is hl_l) else 0.0)
                self._set(p, "u_model", model)
                (self.m_branch_tip if tip else self.m_branch).vao(p).render()
            self._set(p, "u_highlight", 0.0)
        po = self.p_obj
        self._obj_defaults(po, frame)
        # where the next bucking cut will fall (amber), or red if limbs are in the way
        for ft, s_cut, blocked in markers:
            r = ft.radius_at(s_cut) * 1.05 + 0.004
            k = math.sqrt(1.0 + (ft.rest[1] if ft.rest else 0.0) ** 2)
            self._set(po, "u_model", self._fallen_cache[id(ft)][1]["frame"] @ G.translate(0.0, s_cut * k, 0.0) @ G.scale(r, 0.018, r))
            self._set(po, "u_color", (0.9, 0.25, 0.15, 1.0) if blocked else (1.0, 0.78, 0.35, 1.0))
            self._set(po, "u_emissive", 0.6)
            self._set(po, "u_emit_color", (0.5, 0.12, 0.05) if blocked else (0.55, 0.38, 0.12))
            self.m_unit_cyl.vao(po).render()
        self._obj_defaults(po, frame)
        # foliage sprays (alpha-cut cards, both faces)
        self._set(po, "u_material", 13 if mg.CARD_FOLIAGE else 2)
        self.ctx.disable(self.mgl.CULL_FACE)
        for ft in world.fallen_trees:
            key_sp = sa.SPECIES_KEY.get(ft.bark_species, "pine")
            mesh = self.m_limb_foliage.get(key_sp)
            # cut foliage slowly wilts and browns over the weeks after felling
            wilt = min(1.0, ft.decay_days / 40.0)
            for model, tint, _ in self._fallen_cache[id(ft)][1]["foliage"]:
                col = (tint[0] * (1 + 0.9 * wilt), tint[1] * (1 - 0.2 * wilt), tint[2] * (1 - 0.4 * wilt), 1.0)
                self._set(po, "u_color", col)
                self._set(po, "u_model", model)
                mesh.vao(po).render()
        self._set(po, "u_color", (1.0, 1.0, 1.0, 1.0))
        self.ctx.enable(self.mgl.CULL_FACE)

    @staticmethod
    def _axis_angle(axis, deg):
        a = math.radians(deg)
        axis = np.asarray(axis, dtype=float) / max(1e-9, np.linalg.norm(axis))
        x, y, z = axis
        c, s = math.cos(a), math.sin(a)
        C = 1 - c
        m = np.identity(4, dtype=np.float32)
        m[:3, :3] = [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]
        return m

    def _camp_static_models(self, world) -> List[Tuple[np.ndarray, Any]]:
        pit = world.thermal_grid.pit_center
        out = [(G.translate(pit[0], pit[1], pit[2]), self.m_ring)]
        wp = world.woodpile_pos
        out.append((G.translate(*wp) @ G.rot_y(20.0), self.m_woodpile))
        rp = world.drying_rack.position
        out.append((G.translate(rp[0], rp[1] + 0.1, rp[2]), self.m_rack))
        seat = np.array([-1.9, terrain.height(-1.9, -1.2) + 0.15, -1.2])
        out.append((G.translate(*seat) @ G.rot_y(-60.0) @ G.rot_z(90.0), self.m_seat))
        bp = np.array([-2.9, terrain.height(-2.9, 0.4), 0.4])
        out.append((G.translate(*bp) @ G.rot_y(40.0) @ G.rot_x(-8.0), self.m_backpack))
        return out

    def _obj_defaults(self, p, frame):
        self._common_uniforms(p, frame)
        self._set(p, "u_emissive", 0.0)
        self._set(p, "u_emit_color", (0.0, 0.0, 0.0))
        self._set(p, "u_sway", 0.0)
        self._set(p, "u_snow_cap", 1.0)
        self._set(p, "u_highlight", 0.0)
        self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
        self._set(p, "u_material", 0)

    def _camp_pass(self, frame, world, player):
        ctx = self.ctx
        p = self.p_obj
        self._obj_defaults(p, frame)
        tgt = world.target
        tgt_obj = tgt["obj"] if tgt else None
        tgt_kind = tgt["kind"] if tgt else None
        for model, mesh in self._camp_static_models(world):
            mat = 3 if mesh is self.m_ring else (7 if mesh is self.m_backpack else (1 if mesh is self.m_seat else 0))
            self._set(p, "u_model", model)
            self._set(p, "u_material", mat)
            hl = 0.0
            if mesh is self.m_woodpile and tgt_kind == "woodpile":
                hl = 0.6
            if mesh is self.m_rack and tgt_kind == "rack":
                hl = 0.45
            self._set(p, "u_highlight", hl)
            mesh.vao(p).render()
        self._set(p, "u_highlight", 0.0)
        # tarp
        if world.drying_rack.has_tarp:
            ctx.disable(self.mgl.CULL_FACE)
            rp = world.drying_rack.position
            self._set(p, "u_model", G.translate(rp[0], rp[1] + 0.1, rp[2]))
            self._set(p, "u_material", 7)
            self.m_tarp.vao(p).render()
            ctx.enable(self.mgl.CULL_FACE)
        # hearthboard kit (bow & spindle lie beside it when not in use)
        hb = world.hearthboard_pos
        bd = world.crafting.bow_drill
        self._set(p, "u_material", 0)
        self._set(p, "u_highlight", 0.6 if tgt_kind == "hearthboard" else 0.0)
        self._set(p, "u_model", G.translate(*hb) @ G.rot_y(30.0))
        (self.m_hearthboard_notched if bd.has_v_notch else self.m_hearthboard).vao(p).render()
        self._set(p, "u_highlight", 0.0)
        if not world.bow_active:
            self._set(p, "u_model", G.translate(hb[0] + 0.05, hb[1] + 0.02, hb[2] + 0.16) @ G.rot_y(20.0))
            self.m_bow.vao(p).render()
            self._set(p, "u_model", G.translate(hb[0] - 0.12, hb[1] + 0.012, hb[2] + 0.1) @ G.rot_y(70.0) @ G.rot_z(90.0) @ G.scale(0.011, 0.22, 0.011))
            self._set(p, "u_color", (0.78, 0.64, 0.44, 1.0))
            self.m_unit_cyl.vao(p).render()
            self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
            self._set(p, "u_model", G.translate(hb[0] - 0.2, hb[1] + 0.02, hb[2] - 0.08) @ G.scale(0.045, 0.045, 0.04))
            self._set(p, "u_material", 3)
            self.m_stone.vao(p).render()
        # ember glowing in the notch
        if bd.has_ember or bd.socket_temp > 300:
            T = bd.ember_temp if bd.has_ember else bd.socket_temp
            self._set(p, "u_material", 10)
            self._set(p, "u_model", G.translate(hb[0] + 0.05, hb[1] + 0.022, hb[2] + 0.03) @ G.scale(0.008, 0.005, 0.008))
            self._set(p, "u_color", (0.05, 0.03, 0.02, 1.0))
            self._set(p, "u_emissive", 1.0)
            self._set(p, "u_emit_color", tuple(G.blackbody_rgb(max(600.0, T)) * 6.0 * (1.0 + 0.2 * math.sin(self.time * 5.0))))
            self.m_sphere.vao(p).render()
            self._set(p, "u_emissive", 0.0)
            self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
        # resource piles & loose fuel
        for wi in world.world_items:
            hl = 0.6 if (tgt_obj is wi) else 0.0
            self._set(p, "u_highlight", hl)
            if wi.kind == "tinder_source":
                if wi.count <= 0:
                    continue
                s = 0.6 + 0.4 * wi.count / max(1, wi.max_count)
                self._set(p, "u_material", 0)
                if wi.material == "birch_bark":
                    self._set(p, "u_model", G.translate(*(wi.position - np.array([0, 0.05, 0]))) @ G.scale(s, s, s))
                    self.m_curls.vao(p).render()
                elif wi.material == "pine_needles":
                    self._set(p, "u_model", G.translate(*(wi.position - np.array([0, 0.04, 0]))) @ G.scale(0.28 * s, 0.3 * s, 0.24 * s))
                    self.m_needles.vao(p).render()
                elif wi.material == "cattail_fluff":
                    self._set(p, "u_model", G.translate(*(wi.position - np.array([0, 0.05, 0]))))
                    self.m_cattails.vao(p).render()
                elif wi.material == "dry_leaves":
                    self._set(p, "u_model", G.translate(*(wi.position - np.array([0, 0.04, 0]))) @ G.rot_y(wi.position[0] * 40)
                              @ G.scale(0.4 * s, 0.16 * s, 0.33 * s))
                    self._set(p, "u_color", (1.45, 0.95, 0.62, 1.0))
                    self.m_needles.vao(p).render()
                    self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
                else:
                    ctx.disable(self.mgl.CULL_FACE)
                    self._set(p, "u_model", G.translate(*(wi.position - np.array([0, 0.05, 0]))) @ G.scale(s, s, s))
                    self._set(p, "u_material", 2)
                    self.m_grass_clump.vao(p).render()
                    ctx.enable(self.mgl.CULL_FACE)
            elif wi.kind == "kindling_pile":
                if wi.count <= 0:
                    continue
                self._set(p, "u_material", 0)
                s = 0.5 + 0.5 * wi.count / max(1, wi.max_count)
                mesh = self.m_cedar_bundle if wi.species == WoodSpeciesType.RED_CEDAR else self.m_kindling_bundle
                self._set(p, "u_model", G.translate(*(wi.position - np.array([0, 0.06, 0]))) @ G.rot_y(wi.position[0] * 50) @ G.scale(1.0, s, 1.0))
                mesh.vao(p).render()
        self._set(p, "u_highlight", 0.0)

    @staticmethod
    def _is_split(item) -> bool:
        """Split (riven) firewood is drawn as a wedge; branch wood and bucked tops stay round."""
        return (item is not None and getattr(item, "kind", "") == "firewood" and not getattr(item, "is_punk", False)
                and getattr(item, "source", "") not in ("limb", "bucked"))

    def _piece_mesh(self, item, model):
        """(mesh, model) for a cylindrical fuel piece, swapping in the split wedge when it is one."""
        if self._is_split(item):
            return self.m_split, model @ self._split_off
        return self.m_log, model

    def _body_model(self, b) -> np.ndarray:
        if b.geometry_type == "tinder":
            L = b.length
            return G.translate(*b.position) @ G.rot_y(float(b.orientation[1])) @ G.scale(L * 0.5, b.initial_radius * 1.1, L * 0.5)
        return G.orientation_model(b.position, b.axis, float(b.id * 37 % 360), b.initial_radius, b.length, b.initial_radius)

    def _fuel_uniforms(self, p, species, seg_T, seg_char, seg_phi, seg_ash, seg_wet, tinder=False, highlight=0.0,
                       material=""):
        props = get_wood_properties(species)
        bark = props.bark_color
        wood = props.wood_color
        if tinder:
            tcol = {"birch_bark": (0.92, 0.88, 0.80), "pine_needles": (0.55, 0.36, 0.18), "dry_grass": (0.72, 0.62, 0.36),
                    "dry_leaves": (0.62, 0.36, 0.14),
                    "cattail_fluff": (0.86, 0.80, 0.70), "cedar_shavings": (0.82, 0.58, 0.40), "feather_stick": (0.88, 0.76, 0.55)}
            bark = tcol.get(material, (0.82, 0.62, 0.42))
        self._set(p, "u_bark_col", bark)
        self._set(p, "u_wood_col", wood)
        self._set(p, "u_seg_T", [float(x) for x in seg_T])
        self._set(p, "u_seg_char", [float(x) for x in seg_char])
        self._set(p, "u_seg_phi", [float(x) for x in seg_phi])
        self._set(p, "u_seg_ash", [float(x) for x in seg_ash])
        self._set(p, "u_seg_wet", [float(x) for x in seg_wet])
        self._set(p, "u_tinder", 1.0 if tinder else 0.0)
        self._set(p, "u_highlight", highlight)
        self._set(p, "u_bark_style", float({WoodSpeciesType.PAPER_BIRCH: 1, WoodSpeciesType.WHITE_PINE: 2,
                                            WoodSpeciesType.WHITE_OAK: 3, WoodSpeciesType.RED_CEDAR: 4}.get(species, 0)))

    @staticmethod
    def _pad3(a) -> List[float]:
        a = list(np.asarray(a, dtype=float).reshape(-1))
        if not a:
            return [0.0, 0.0, 0.0]
        while len(a) < 3:
            a.append(a[-1])
        return a[:3]

    def _fuel_pass(self, frame, world):
        p = self.p_fuel
        self._common_uniforms(p, frame)
        self._set(p, "u_sway", 0.0)
        self._set(p, "u_model", np.identity(4, dtype=np.float32))
        tgt = world.target
        tgt_obj = tgt["obj"] if tgt else None
        for b in world.thermal_grid.bodies:
            st = b.segment_states()
            seg_len = b.length / max(1, b.n_segments)
            area = 2 * math.pi * max(b.initial_radius, 0.004) * seg_len
            ash = np.clip(st["ash_kg"] / (280.0 * area) * 1000.0 / 3.0, 0.0, 1.0)
            charv = np.maximum(st["char_frac"], st["charred"])
            mat = getattr(getattr(b, "source_item", None), "material", "")
            self._fuel_uniforms(p, b.species_type, self._pad3(st["T_surf"]), self._pad3(charv), self._pad3(st["phi"]),
                                self._pad3(ash), self._pad3(st["moisture"]), b.geometry_type == "tinder",
                                0.5 if tgt_obj is b else 0.0, mat)
            if b.geometry_type == "tinder":
                self._set(p, "u_model", self._body_model(b))
                self.m_nest.vao(p).render()
            else:
                r_now = float(np.max(st["radius"])) if len(st["radius"]) else b.initial_radius
                r = max(0.35 * b.initial_radius, r_now)
                m = G.orientation_model(b.position, b.axis, float(b.id * 37 % 360), r, b.length, r)
                mesh, m = self._piece_mesh(getattr(b, "source_item", None), m)
                self._set(p, "u_model", m)
                mesh.vao(p).render()
        # loose fuel on the ground and on the rack
        amb = world.weather.ambient_temp_c
        items = [(wi.fuel, wi.position, wi.yaw, wi) for wi in world.world_items if wi.kind == "fuel" and wi.fuel]
        rp = world.drying_rack.position
        for i, it in enumerate(world.drying_rack.logs):
            row, col = divmod(i, 5)
            pos = np.array([rp[0] - 0.5 + col * 0.25, rp[1] + 0.1 + 0.65 + it.radius + row * 0.13, rp[2]])
            items.append((it, pos, 90.0, None))
        for it, pos, yaw, wi in items:
            wet = 0.3 * it.m_surface + 0.7 * it.m_core
            T = [amb] * 3
            sp = it.species
            self._fuel_uniforms(p, sp, T, [0.0] * 3, [0.0] * 3, [0.0] * 3, [min(1.0, it.m_surface)] * 3, False,
                                0.55 if (wi is not None and tgt_obj is wi) else 0.0)
            ax = orientation_to_axis(np.array([0.0, yaw, 0.0]))
            m = G.orientation_model(pos, ax, yaw * 3.0, it.radius, it.length, it.radius)
            mesh, m = self._piece_mesh(it, m)
            self._set(p, "u_model", m)
            mesh.vao(p).render()
        # bracket fungi fruiting on rotting punk-wood logs (late decay, DESIGN 7.2)
        po = self.p_obj
        fung = [(wi, n) for wi in world.world_items if wi.kind == "fuel" and wi.fuel and (n := getattr(wi, "fungi", 0)) > 0]
        if fung:
            self._obj_defaults(po, frame)
            self._set(po, "u_material", 9)
            for wi, n in fung:
                ax = orientation_to_axis(np.array([0.0, wi.yaw, 0.0]))
                side = np.cross(ax, np.array([0.0, 1.0, 0.0]))
                for k in range(int(n)):
                    p0 = wi.position + ax * wi.fuel.length * (k - 1) * 0.25 + side * wi.fuel.radius * 0.95
                    yaw = math.degrees(math.atan2(side[2], side[0]))
                    self._set(po, "u_model", G.translate(*p0) @ G.rot_y(-yaw) @ G.scale(0.07, 0.07, 0.07))
                    self.m_fungus.vao(po).render()

    def _transparent_pass(self, frame, world):
        ctx = self.ctx
        mgl = self.mgl
        self.depth_copy.use(location=5)
        ctx.enable(mgl.BLEND)
        ctx.enable(mgl.DEPTH_TEST)
        self.scene_fbo.depth_mask = False
        # water
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        p = self.p_water
        self._common_uniforms(p, frame)
        self._set(p, "u_model", np.identity(4, dtype=np.float32))
        self._set(p, "u_sway", 0.0)
        L = frame["lighting"]
        self._set(p, "u_zenith", L.zenith)
        self._set(p, "u_horizon", L.horizon)
        self._set(p, "u_rain", float(world.weather.rain_intensity))
        ctx.disable(mgl.CULL_FACE)
        # spring snow-melt and heavy rain swell the creek (DESIGN 7.1)
        wx = world.weather
        flood = (0.22 * (1.0 - wx.season_progress()) if wx.current_season.value == "Spring" else 0.0) + 0.12 * wx.rain_intensity
        self._set(p, "u_model", G.translate(0.0, flood, 0.0))
        self.m_creek.vao(p).render()
        self._set(p, "u_model", np.identity(4, dtype=np.float32))
        # ember bed
        bed = world.thermal_grid.coal_bed
        ash_total = bed.ash_kg
        if bed.char_kg > 0.002 or ash_total > 0.01:
            pe = self.p_ember
            self._common_uniforms(pe, frame)
            self._set(pe, "u_sway", 0.0)
            pc = world.thermal_grid.pit_center
            rad = max(0.1, min(0.38, bed.radius * 1.4 + 0.05 + min(0.15, ash_total * 0.3)))
            self._set(pe, "u_model", G.translate(pc[0], pc[1] + 0.012, pc[2]) @ G.scale(1.0, 0.08, 1.0))
            self._set(pe, "u_bed_T", float(bed.T if bed.char_kg > 0.002 else 20.0))
            self._set(pe, "u_bed_amount", float(min(1.0, bed.char_kg * 6.0 + ash_total * 2.0)))
            self._set(pe, "u_ash", float(min(1.0, ash_total * 3.0)))
            self._set(pe, "u_bed_radius", float(rad))
            self._set(pe, "u_breath", float(min(1.0, world.thermal_grid.breath_timer * 1.2)))
            ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
            self.m_disc_vao().render()
        # smoke & steam (premultiplied alpha)
        alpha_inst, add_inst = self.particle_system.get_instances(frame["cam_pos"])
        pp = self.p_part
        self._common_uniforms(pp, frame)
        if "u_depth" in pp:
            pp["u_depth"].value = 5
        self._set(pp, "u_screen", (float(self.rw), float(self.rh)))
        self._set(pp, "u_near", CAMERA_NEAR)
        self._set(pp, "u_far", CAMERA_FAR)
        if len(alpha_inst):
            self._set(pp, "u_additive", 0)
            ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
            n = min(len(alpha_inst), MAX_PARTICLES)
            self.part_buf.write(alpha_inst[:n].tobytes())
            self.part_vao.render(mgl.TRIANGLES, vertices=6, instances=n)
        # flames (additive)
        fl = self._flame_instances
        if len(fl):
            pf = self.p_flame
            self._set(pf, "u_view", frame["view"])
            self._set(pf, "u_proj", frame["proj"])
            self._set(pf, "u_cam_pos", frame["cam_pos"])
            self._set(pf, "u_time", float(self.time))
            self._set(pf, "u_gain", float(np.clip(1.3 / max(0.1, frame["lighting"].exposure), 0.25, 1.6)))
            if "u_depth" in pf:
                pf["u_depth"].value = 5
            self._set(pf, "u_screen", (float(self.rw), float(self.rh)))
            self._set(pf, "u_near", CAMERA_NEAR)
            self._set(pf, "u_far", CAMERA_FAR)
            # premultiplied 'over': overlapping tongues converge on flame colour instead of white
            ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
            n = min(len(fl), MAX_FLAMES)
            self.flame_buf.write(fl[:n].tobytes())
            self.flame_vao.render(mgl.TRIANGLES, vertices=6, instances=n)
        # sparks / embers / fireflies (additive)
        if len(add_inst):
            self._set(pp, "u_additive", 1)
            self._set(pp, "u_emit_gain", float(np.clip(1.6 / max(0.1, frame["lighting"].exposure), 0.3, 1.6)))
            ctx.blend_func = (mgl.ONE, mgl.ONE)
            n = min(len(add_inst), MAX_PARTICLES)
            self.part_buf.write(add_inst[:n].tobytes())
            self.part_vao.render(mgl.TRIANGLES, vertices=6, instances=n)
        # placement ghost & pinned blueprint hologram
        self._ghost_pass(frame, world)
        ctx.enable(mgl.CULL_FACE)
        self.scene_fbo.depth_mask = True
        ctx.disable(mgl.BLEND)

    def m_disc_vao(self):
        if not hasattr(self, "_disc"):
            self._disc = RenderEngine._Mesh(self, mg.create_frustum(1.0, 1.0, 0.1, 32, 1, (1, 1, 1, 1), caps=True))
        return self._disc.vao(self.p_ember)

    def _ghost_pass(self, frame, world):
        ctx = self.ctx
        mgl = self.mgl
        p = self.p_ghost
        self._set(p, "u_view", frame["view"])
        self._set(p, "u_proj", frame["proj"])
        self._set(p, "u_cam_pos", frame["cam_pos"])
        self._set(p, "u_time", float(self.time))
        self._set(p, "u_sway", 0.0)
        ctx.blend_func = (mgl.ONE, mgl.ONE)
        pit = world.thermal_grid.pit_center
        book = world.reference_book
        if book is not None and book.selected_blueprint_key:
            from firestarter.recipes.blueprints import ALL_BLUEPRINTS
            bp = ALL_BLUEPRINTS[book.selected_blueprint_key]
            pieces = [(b.geometry_type, b.position - pit, b.axis) for b in world.thermal_grid.bodies]
            bp.match_pieces(pieces)
            for slot in bp.slots:
                if slot.is_filled:
                    continue
                L, r = float(slot.dimensions[0]), float(slot.dimensions[1])
                if slot.expected_type == "tinder":
                    m = G.translate(*(pit + slot.position)) @ G.scale(L * 0.5, r * 1.1, L * 0.5)
                    mesh = self.m_nest
                else:
                    m = G.orientation_model(pit + slot.position, slot.axis, 0.0, r, L, r)
                    mesh = self.m_log
                self._set(p, "u_model", m)
                self._set(p, "u_color", (0.35, 0.75, 1.0, 0.55))
                mesh.vao(p).render()
        gh = world.ghost
        if gh is not None and gh["valid"]:
            col = (1.0, 0.78, 0.35, 0.8) if gh["valid"] else (1.0, 0.25, 0.2, 0.6)
            if gh.get("slot"):
                col = (0.45, 1.0, 0.7, 0.9)
            if gh["kind"] == "tinder":
                m = G.translate(*gh["position"]) @ G.scale(gh["length"] * 0.5, gh["radius"] * 1.1, gh["length"] * 0.5)
                mesh = self.m_nest
            else:
                ax = orientation_to_axis(gh["orientation"])
                m = G.orientation_model(gh["position"], ax, 0.0, gh["radius"], gh["length"], gh["radius"])
                mesh = self.m_log
            self._set(p, "u_model", m)
            self._set(p, "u_color", col)
            mesh.vao(p).render()

    # ================================================================== effects (flames, particles)
    def _update_fx(self, world, player, hands, frame, dt):
        ps = self.particle_system
        grid = world.thermal_grid
        wind = world.weather.wind_vector
        flames: List[List[float]] = []
        tilt = np.asarray(getattr(grid, "flame_tilt", np.zeros(3)), dtype=float)
        tx, tz = float(tilt[0]) * 0.6, float(tilt[2]) * 0.6
        a = grid.store.a
        rng = self.particle_system.rng
        if grid.store.n:
            P = grid.seg_pos
            R = grid.seg_rad
            hrr = a["hrr_flame"]
            steam = a["steam_rate"]
            smoke = a["smoke_rate"]
            phi = a["phi"]
            T = a["T_surf"]
            for i in np.nonzero(hrr > 40.0)[0]:
                q = hrr[i] / 1000.0
                # Heskestad-like length of the flame rising off this stretch of stick
                h = max(0.05, 0.26 * q ** 0.4)
                w = 0.035 + 2.4 * R[i] + 0.22 * h
                inten = min(1.0, 0.4 + 0.35 * q ** 0.35)
                base = P[i] + np.array([0.0, R[i] * 0.7, 0.0])
                n_sheets = 2 if q > 0.8 else 1
                for k in range(n_sheets):
                    seed = float((i * 7 + k * 13) % 97) / 97.0
                    off = np.array([math.sin(seed * 40) * R[i], 0.0, math.cos(seed * 40) * R[i]])
                    flames.append([*(base + off), w * (0.85 + 0.3 * k), h * (0.8 + 0.4 * seed), seed, inten * (1.0 - 0.2 * k), 0.0,
                                   tx, tz, float(T[i]), 0.0])
                if rng.random() < 0.05 * q * dt:
                    ps.spawn_sparks(P[i] + np.array([0, h * 0.3, 0]), 1, 0.7, 0.03)
            # steam only shows as white vapour when a lot of water boils off (wet / green wood);
            # a little moisture from seasoned wood stays invisible in the hot plume
            wet_seg = a["water"].sum(axis=1) / np.maximum(a["dry0"].sum(axis=1), 1e-9)
            for i in np.nonzero((steam > 1.5e-4) & (wet_seg > 0.22))[0]:
                ps.spawn_steam(P[i] + np.array([0, R[i], 0]), float(steam[i]), dt)
            vol = a["vol_rate"]
            for i in np.nonzero((smoke > 2e-5) | ((phi <= 0) & (smoke > 4e-6)))[0]:
                g_s = float(smoke[i]) * 1000.0
                sooty = smoke[i] / max(1e-9, vol[i]) > 0.3     # oxygen-starved flame: unburnt tar & soot
                # thick yellow-grey smoke from smouldering / starved fires, thin grey from clean flames
                if phi[i] > 0 and not sooty:
                    # a clean, well-aerated flame: only wisps of smoke escape past the tips
                    ps.spawn_smoke(P[i] + np.array([0, 0.3, 0]), min(3.0, 0.15 + g_s * 1.2), dt, color=(0.45, 0.45, 0.46),
                                   alpha=0.07, key="flamesmoke", size=(0.14, 0.24))
                else:
                    ps.spawn_pyrolysis_smoke(P[i] + np.array([0, R[i], 0]), g_s, dt)
        # merged fire body: a few large licking sheets around the fire's centroid
        q_tot = grid.flame_hrr / 1000.0
        if q_tot > 2.5 and grid.flame_height > 0.12:
            c = grid.fire_centroid
            base = np.array([c[0], grid.pit_center[1] + 0.03, c[2]])
            D = max(0.12, grid.fire_diameter)
            n_body = int(min(6, 3 + q_tot / 15.0))
            for k in range(n_body):
                seed = (0.13 + 0.618 * k) % 1.0
                ang = k * 2.39996
                off = np.array([math.sin(ang) * D * 0.22, 0.0, math.cos(ang) * D * 0.22])
                # tall fires get proportionally wide bodies (a flash-over is a rolling ball, not a pillar)
                width = max(D * (0.55 + 0.12 * (k % 3)) + 0.06, 0.42 * grid.flame_height * (0.8 + 0.2 * (k % 2)))
                flames.append([*(base + off), width, grid.flame_height * (0.78 + 0.3 * seed), seed,
                               min(1.0, 0.55 + 0.05 * q_tot ** 0.5), 0.0, tx, tz, 900.0, 0.0])
            # thin smoke column & rising embers above a healthy fire
            ps.spawn_smoke(base + np.array([0.0, grid.flame_height * 1.05, 0.0]), min(5.0, 0.3 + q_tot * 0.05), dt,
                           color=(0.42, 0.42, 0.44), alpha=0.06 + 0.05 * min(1.0, grid.smoke_rate * 300), key="column",
                           size=(0.25, 0.4), life=(3.0, 6.0), spread=D * 0.2)
            ps.spawn_embers(base + np.array([0, grid.flame_height * 0.4, 0]), min(3.0, 0.15 + q_tot * 0.04), dt, radius=D * 0.25)
        # coal bed: CO flames and embers
        bed = grid.coal_bed
        if bed.char_kg > 0.01 and bed.T > 650.0:
            pc = grid.pit_center
            n_blue = int(min(10, 2 + bed.char_kg * 20))
            for k in range(n_blue):
                ang = k * 2.399 + self.time * 0.3
                rr = bed.radius * 0.7 * math.sqrt((k + 0.5) / n_blue)
                flames.append([pc[0] + math.cos(ang) * rr, pc[1] + 0.02, pc[2] + math.sin(ang) * rr, 0.05, 0.05 + 0.04 * (bed.T - 650) / 400,
                               (k * 0.37) % 1.0, 0.5 * min(1.0, (bed.T - 650) / 300), 1.0, tx, tz, bed.T, 0.0])
            if grid.flame_hrr < 2000:
                ps.spawn_smoke(pc + np.array([0, 0.1, 0]), 1.2, dt, color=(0.6, 0.62, 0.66), alpha=0.05, key="bedhaze",
                               size=(0.15, 0.3), life=(2.5, 4.0), spread=bed.radius * 0.4)
        for bu in grid.active_burners():
            D = bu["diameter"]
            Lf = max(0.2, 0.235 * bu["q_kw"] ** 0.4 - 1.02 * D)
            for k in range(4):
                off = np.array([math.sin(k * 1.7) * D * 0.2, 0.0, math.cos(k * 1.7) * D * 0.2])
                flames.append([*(bu["pos"] + off), D * 1.1, Lf * (1.0 - 0.08 * k), 0.21 * k, 1.3, 0.0, tx, tz, 1000.0, 0.0])
        # lighter flame in the hand
        hm = frame["hm"]
        if hm is not None and world.crafting.lighter_is_lit and player.primary_hand == "pocket_lighter":
            hand = self._right_hand_matrix(frame, hm)
            grip = G.translate(0.0, -0.024, 0.047) @ G.rot_z(90.0) @ G.translate(0.0, 0.021, 0.0) @ G.rot_y(90.0)
            tip_vm = (hand @ grip @ np.array([0.0, 0.047, 0.0, 1.0]))[:3]
            tip, k = self._vm_to_world(frame, tip_vm)
            self._lighter_tip = tip
            flames.append([*tip, 0.02 * k, 0.055 * k, 0.5, 1.1, 2.0, 0.0, 0.0, 1000.0, 0.0])
        else:
            self._lighter_tip = None
        if world.lighter_point is not None:
            lp = world.lighter_point
            flames.append([*lp, 0.03, 0.06, 0.77, 1.0, 2.0, 0.0, 0.0, 1000.0, 0.0])
        self._flame_instances = np.array(flames, dtype=np.float32).reshape(-1, FLAME_STRIDE)

        # event-driven effects
        for pos in grid.pop_events:
            ps.spawn_sparks(pos, 4, 1.0)
        for name, data in self._drain_world_events(world):
            if name == "collapse" and data is not None:
                ps.spawn_sparks(np.asarray(data) + np.array([0, 0.05, 0]), 22, 1.3, 0.1)
                ps.spawn_embers(np.asarray(data), 40.0, 0.5, 0.12)
            elif name == "gas_flash":
                ps.spawn_sparks(np.asarray(data) + np.array([0, 0.4, 0]), 60, 2.2, 0.3)
            elif name == "breath":
                if data and data.get("cold"):
                    ps.spawn_breath(frame["cam_pos"] + frame["fwd"] * 0.25 - frame["up"] * 0.08, frame["fwd"] * 1.2, data.get("velocity", 2.0) / 2.5)
                if np.linalg.norm(frame["cam_pos"][[0, 2]] - grid.pit_center[[0, 2]]) < 2.6 and bed.char_kg > 0.005:
                    ps.spawn_sparks(grid.pit_center + np.array([0, 0.05, 0]), int(4 + data.get("velocity", 2) * 3) if data else 6, 0.7)
            elif name == "ash_scatter":
                ps.spawn_ash(np.asarray(data), 30)
            elif name == "fire_lit" and data is not None:
                ps.spawn_sparks(np.asarray(data), 10, 0.8)
            elif name == "chop" and data is not None:
                ps.spawn_chips(np.asarray(data), 14)
            elif name == "nest_flame" and data is not None:
                ps.spawn_sparks(np.asarray(data), 14, 0.9)
        # bow drill smoke from the socket
        bd = world.crafting.bow_drill
        if bd.is_smoking or bd.has_ember:
            # a thin curling wisp from the socket (it thickens as the dust chars)
            hb = world.hearthboard_pos
            ps.spawn_smoke(hb + np.array([0.05, 0.03, 0.03]), 3.0 + 9.0 * bd.smoke_intensity, dt, color=(0.80, 0.78, 0.74),
                           alpha=0.13 + 0.1 * bd.smoke_intensity, spread=0.008, size=(0.015, 0.03), life=(1.2, 2.4),
                           key="bowdrill", rise=0.22, grow=0.09)
        nest = world.crafting.active_tinder_nest
        if nest is not None and nest.has_ember:
            p0 = frame["cam_pos"] + frame["up"] * -0.12 + frame["fwd"] * 0.34
            ps.spawn_smoke(p0, 2.0 + 6.0 * nest.smoke_density, dt, color=(0.82, 0.8, 0.76), alpha=0.12, spread=0.015,
                           size=(0.02, 0.04), life=(1.0, 2.0), key="nest", rise=0.25, grow=0.1)
        fg = world.crafting.smoldering_fungus
        if fg is not None and fg.is_smoldering:
            p0 = frame["cam_pos"] + frame["right"] * -0.22 + frame["up"] * -0.2 + frame["fwd"] * 0.45
            ps.spawn_smoke(p0, 2.0, dt, color=(0.75, 0.74, 0.72), alpha=0.1, spread=0.01, size=(0.02, 0.035), life=(1.0, 2.0),
                           key="fungus", rise=0.2, grow=0.07)
        # weather & ambience particles
        w = world.weather
        night = frame["lighting"].night
        fireflies = 1.0 if (w.current_season.value == "Summer" and night > 0.6 and w.rain_intensity < 0.05) else 0.0
        ps.spawn_weather(frame["cam_pos"], rain_intensity=w.rain_intensity, snow_intensity=w.snow_intensity,
                         wind_vec=wind, dt=dt, leaves=w.falling_leaves * (1.0 - night * 0.5), fireflies=fireflies)
        ps.set_plume(np.array([grid.fire_centroid[0], grid.pit_center[1], grid.fire_centroid[2]]), q_tot + grid.char_hrr / 2000.0,
                     max(0.12, grid.fire_diameter * 0.5))
        ps.update(dt, wind, ground_fn=lambda x, z: np.asarray(terrain.height(x, z)))
        self._weather_cloud = w.cloud_cover
        self.stats = {"particles": ps.count(), "flames": len(self._flame_instances)}

    def push_events(self, events):
        """World events (collapse, breath, gas flash...) forwarded by the game loop."""
        self.pending_events.extend(events)

    def _drain_world_events(self, world):
        out = self.pending_events
        self.pending_events = []
        return out

    # ================================================================== viewmodel
    def _hand_basis(self, frame, pos_cam, rot_deg):
        r, u, f = frame["right"], frame["up"], frame["fwd"]
        C = np.identity(4, dtype=np.float32)
        C[:3, 0] = r
        C[:3, 1] = u
        C[:3, 2] = f
        wp = frame["cam_pos"] + r * pos_cam[0] + u * pos_cam[1] + f * pos_cam[2]
        R = G.rot_y(rot_deg[1]) @ G.rot_x(rot_deg[0]) @ G.rot_z(rot_deg[2])
        return G.translate(*wp) @ C @ R, wp

    HAND_SCALE = 0.9

    def _right_hand_matrix(self, frame, hm) -> np.ndarray:
        base, _ = self._hand_basis(frame, hm["right_pos"], hm["right_rot"])
        return base @ G.scale(self.HAND_SCALE, self.HAND_SCALE, self.HAND_SCALE)

    def _vm_to_world(self, frame, p_vm: np.ndarray):
        """A point placed with the viewmodel projection (narrower FOV) -> the world point that
        projects to the same pixel with the world projection.  Returns (point, size factor)."""
        k = math.tan(math.radians(FOV_DEGREES) * 0.5) / math.tan(math.radians(VM_FOV) * 0.5)
        rel = np.asarray(p_vm, dtype=float) - frame["cam_pos"]
        x, y, z = float(rel @ frame["right"]), float(rel @ frame["up"]), float(rel @ frame["fwd"])
        return frame["cam_pos"] + frame["right"] * x * k + frame["up"] * y * k + frame["fwd"] * z, k

    def _world_to_vm(self, frame, p_world: np.ndarray) -> np.ndarray:
        """Inverse of _vm_to_world: where to draw a view-model part so it lines up with a world point."""
        k = math.tan(math.radians(FOV_DEGREES) * 0.5) / math.tan(math.radians(VM_FOV) * 0.5)
        rel = np.asarray(p_world, dtype=float) - frame["cam_pos"]
        return frame["cam_pos"] + frame["right"] * float(rel @ frame["right"]) / k + frame["up"] * float(rel @ frame["up"]) / k \
            + frame["fwd"] * float(rel @ frame["fwd"])

    def _viewmodel_pass(self, frame, world, player, hands):
        hm = frame["hm"]
        if hm is None:
            return
        ctx = self.ctx
        p = self.p_obj
        vm_frame = dict(frame)
        vm_frame["proj"] = frame["vm_proj"]
        self._obj_defaults(p, vm_frame)
        self._set(p, "u_snow", 0.0)
        self._set(p, "u_fog_density", 0.0)
        self._set(p, "u_shadows", 0)
        ctx.enable(self.mgl.DEPTH_TEST)
        ctx.enable(self.mgl.CULL_FACE)
        glove = (0.34, 0.25, 0.17, 1.0)          # worn tan leather work gloves
        glove_dark = (0.22, 0.16, 0.11, 1.0)
        cuff = (0.14, 0.15, 0.11, 1.0)           # dark waxed-canvas jacket
        gp = self.m_glove
        self._hand_mats = {}
        for side in ("right", "left"):
            is_right = side == "right"
            base, wp = self._hand_basis(frame, hm[f"{side}_pos"], hm[f"{side}_rot"])
            sx = 1.0 if is_right else -1.0
            flex = float(hm[f"{side}_flex"])
            mirror = G.scale(sx, 1.0, 1.0)
            ctx.front_face = "ccw" if is_right else "cw"
            self._set(p, "u_material", 12)
            self._set(p, "u_color", glove)
            hs = self.HAND_SCALE                   # overall hand scale (gloved adult hand)
            base = base @ G.scale(hs, hs, hs)
            hand = base @ mirror
            self._hand_mats[side] = hand
            self._set(p, "u_model", hand)
            gp["palm"].vao(p).render()
            for fi, (fx, fz, lengths, rad) in enumerate(mg.FINGERS):
                m = hand @ G.translate(fx, -0.002, fz)
                curl = flex * (1.0 + 0.06 * fi)
                for j, L in enumerate(lengths):
                    m = m @ G.rot_x(curl * (74.0, 96.0, 52.0)[j])
                    self._set(p, "u_model", m)
                    gp[f"f{fi}_{j}"].vao(p).render()
                    m = m @ G.translate(0.0, 0.0, L)
            self._set(p, "u_model", hand @ G.translate(-0.030, -0.006, -0.012) @ G.rot_y(-25.0))
            gp["thumb_base"].vao(p).render()
            # thumb: splayed when the hand is open, folded along the index finger in a grip
            m = hand @ G.translate(-0.036, -0.012, 0.002) @ G.rot_y(-48.0 + 62.0 * flex) @ G.rot_z(-40.0)
            for j, L in enumerate(mg.THUMB[0]):
                m = m @ G.rot_x(flex * (34.0, 36.0)[j])
                self._set(p, "u_model", m)
                gp[f"t_{j}"].vao(p).render()
                m = m @ G.translate(0.0, 0.0, L)
            ctx.front_face = "ccw"
            # wrist and sleeve toward the elbow
            elbow = hm[f"{side}_elbow"]
            ew = frame["cam_pos"] + frame["right"] * elbow[0] + frame["up"] * elbow[1] + frame["fwd"] * elbow[2]
            wrist = wp - base[:3, 2] * 0.035
            d = ew - wrist
            L = float(np.linalg.norm(d))
            if L > 1e-4:
                self._set(p, "u_material", 12)
                self._set(p, "u_color", glove_dark)
                self._set(p, "u_model", G.orientation_model(wrist, d / L, 0.0, 0.034, 0.09, 0.031))
                self.m_forearm.vao(p).render()
                self._set(p, "u_material", 5)
                self._set(p, "u_color", cuff)
                self._set(p, "u_model", G.orientation_model(wrist + d / L * 0.07, d / L, 0.0, 0.036, min(L * 1.4, 0.34), 0.04))
                self.m_forearm.vao(p).render()
            self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
            if is_right:
                self._draw_right_item(p, hm, world, frame)
            else:
                self._draw_left_item(p, hm, world, frame)
        # hand-drill spindle between the palms
        if world.bow_active and world.friction_mode == "hand":
            _, lwp = self._hand_basis(frame, hm["left_pos"], hm["left_rot"])
            _, rwp = self._hand_basis(frame, hm["right_pos"], hm["right_rot"])
            mid = (lwp + rwp) * 0.5
            # the spindle runs from the palms down into the hearthboard's notch (the board is a
            # world object: map it into the narrower view-model projection so they line up)
            bottom = self._world_to_vm(frame, world.hearthboard_pos + np.array([0.05, 0.025, 0.03]))
            # the spindle runs between the palms and ends just above them
            axis_v = mid - bottom
            axis_v = axis_v / max(1e-6, float(np.linalg.norm(axis_v)))
            top = mid + axis_v * 0.07
            self._set(p, "u_material", 0)
            self._set(p, "u_color", (0.74, 0.62, 0.44, 1.0))
            self._set(p, "u_model", G.orientation_model((top + bottom) * 0.5, top - bottom, self.time * 3000.0 % 360, 0.009,
                                                        float(np.linalg.norm(top - bottom)), 0.009))
            self.m_unit_cyl.vao(p).render()
            self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
            if world.secondary_held:
                # smooth river stone capping the spindle, pressed down by the palms
                self._set(p, "u_material", 3)
                self._set(p, "u_model", G.translate(*(top + frame["up"] * 0.008)) @ G.scale(0.03, 0.014, 0.026))
                self.m_stone.vao(p).render()
        # bow-drill apparatus while drilling (in the hands)
        if world.bow_active and world.friction_mode != "hand":
            _, lwp = self._hand_basis(frame, hm["left_pos"], hm["left_rot"])
            top = lwp - frame["up"] * 0.035
            bottom = self._world_to_vm(frame, world.hearthboard_pos + np.array([0.05, 0.025, 0.03]))
            self._set(p, "u_material", 0)
            self._set(p, "u_color", (0.78, 0.64, 0.44, 1.0))
            self._set(p, "u_model", G.orientation_model((top + bottom) * 0.5, top - bottom, self.time * 2000.0 % 360, 0.011,
                                                        float(np.linalg.norm(top - bottom)), 0.011))
            self.m_unit_cyl.vao(p).render()
            self._set(p, "u_color", (1.0, 1.0, 1.0, 1.0))
            self._set(p, "u_material", 3)
            self._set(p, "u_model", G.translate(*(top + frame["up"] * 0.012)) @ G.scale(0.04, 0.02, 0.035))
            self.m_stone.vao(p).render()
            _, rwp = self._hand_basis(frame, hm["right_pos"], hm["right_rot"])
            mid = (top + bottom) * 0.5
            self._set(p, "u_material", 0)
            self._set(p, "u_model", self._cam_item(frame, (rwp + mid) * 0.5, G.rot_y(90.0) @ G.scale(0.9, 0.9, 0.9)))
            self.m_bow.vao(p).render()

    def _cam_item(self, frame, pos_world, local: np.ndarray) -> np.ndarray:
        C = np.identity(4, dtype=np.float32)
        C[:3, 0] = frame["right"]
        C[:3, 1] = frame["up"]
        C[:3, 2] = frame["fwd"]
        return G.translate(*pos_world) @ C @ local

    def _draw_right_item(self, p, hm, world, frame):
        """Tools are drawn in the right hand's own frame so they sit inside the closed fist.
        Hand frame: x across the knuckles (index at -x), y = back of the hand, z = fingers.
        Grip axis of the fist runs along x at (y=-0.024, z=0.047)."""
        item = hm.get("right_item", "pocket_lighter")
        inv = world.inventory
        hand = self._hand_mats.get("right")
        _, wp = self._hand_basis(frame, hm["right_pos"], hm["right_rot"])
        r, u, f = frame["right"], frame["up"], frame["fwd"]
        # item +y (its 'up') points out of the thumb side of the fist (local -x)
        grip = G.translate(0.0, -0.024, 0.047) @ G.rot_z(90.0)
        if item == "pocket_lighter" and hand is not None:
            self._set(p, "u_material", 0)
            self._set(p, "u_model", hand @ grip @ G.translate(0.0, 0.021, 0.0) @ G.rot_y(90.0))
            self.m_lighter.vao(p).render()
        elif item == "pocket_knife" and hand is not None:
            self._set(p, "u_material", 6 if inv.knife == "steel" else 0)
            self._set(p, "u_model", hand @ grip @ G.rot_x(-20.0) @ G.translate(0.0, 0.012, 0.0) @ G.rot_y(90.0))
            self.m_knife[1 if inv.knife == "steel" else 0].vao(p).render()
        elif item == "felling_axe" and hand is not None:
            self._set(p, "u_material", 0)
            # held near the end of the haft, leaning forward over the wrist, head up and the broad
            # cutting edge facing away from you
            axe_grip = G.translate(0.0, -0.024, 0.047) @ G.rot_y(28.0) @ G.rot_z(90.0) @ G.translate(0.0, 0.12, 0.0)
            self._set(p, "u_model", hand @ axe_grip)
            self.m_axe[1 if inv.axe == "steel" else 0].vao(p).render()
        elif item == "bow_drill_spindle" and not world.bow_active:
            self._set(p, "u_material", 0)
            self._set(p, "u_model", self._cam_item(frame, wp + u * 0.02 + f * 0.05 - r * 0.06, G.rot_y(75.0) @ G.scale(0.85, 0.85, 0.85)))
            self.m_bow.vao(p).render()
        elif item == "accelerant" and inv.gasoline_kg > 0:
            self._set(p, "u_material", 0)
            self._set(p, "u_model", self._cam_item(frame, wp - u * 0.075 + f * 0.01, G.rot_y(-70.0)))
            self.m_gas.vao(p).render()

    def _draw_left_item(self, p, hm, world, frame):
        _, wp = self._hand_basis(frame, hm["left_pos"], hm["left_rot"])
        r, u, f = frame["right"], frame["up"], frame["fwd"]
        nest = world.crafting.active_tinder_nest
        if nest is not None and nest.has_ember:
            pf = self.p_fuel
            vm = dict(frame)
            vm["proj"] = frame["vm_proj"]
            self._common_uniforms(pf, vm)
            self._set(pf, "u_fog_density", 0.0)
            T = nest.core_temp
            glow = [T, T * 0.85, T * 0.85]
            self._fuel_uniforms(pf, WoodSpeciesType.RED_CEDAR, glow, [0.6, 0.3, 0.3], [0.0] * 3, [0.0] * 3, [0.0] * 3, True,
                                0.0, nest.material)
            c = frame["cam_pos"] + u * -0.11 + f * 0.31
            self._set(pf, "u_model", G.translate(*c) @ G.scale(0.065, 0.04, 0.065))
            self.m_nest.vao(pf).render()
            # the coal glowing in the hollow of the nest: grows and brightens with every good breath
            po = self.p_obj
            self._obj_defaults(po, vm)
            self._set(po, "u_fog_density", 0.0)
            self._set(po, "u_material", 10)
            self._set(po, "u_color", (0.05, 0.03, 0.02, 1.0))
            self._set(po, "u_emissive", 1.0)
            glow_r = 0.006 + 0.0035 * min(8.0, nest.glow_area_cm2) ** 0.5
            pulse = 1.0 + 0.15 * math.sin(self.time * 6.0) + 0.5 * max(0.0, 1.0 - nest.time_since_breath)
            self._set(po, "u_emit_color", tuple(G.blackbody_rgb(max(550.0, T)) * 5.0 * pulse))
            self._set(po, "u_model", G.translate(*(c + u * 0.018)) @ G.scale(glow_r, glow_r * 0.6, glow_r))
            self.m_sphere.vao(po).render()
            self._set(po, "u_emissive", 0.0)
            self._set(po, "u_color", (1.0, 1.0, 1.0, 1.0))
            return
        kind = world.current_offhand()
        inv = world.inventory
        self._set(p, "u_material", 0)
        if kind == "fungus":
            if inv.fungi:
                self._set(p, "u_material", 9)
                sm = world.crafting.smoldering_fungus is not None
                self._set(p, "u_emissive", 0.8 if sm else 0.0)
                self._set(p, "u_emit_color", (1.0, 0.25, 0.05))
                self._set(p, "u_model", self._cam_item(frame, wp + u * 0.03 + f * 0.01, G.rot_y(90.0) @ G.scale(0.05, 0.05, 0.05)))
                self.m_fungus.vao(p).render()
                self._set(p, "u_emissive", 0.0)
            return
        item = inv.peek(kind)
        if item is None:
            return
        pf = self.p_fuel
        vm = dict(frame)
        vm["proj"] = frame["vm_proj"]
        self._common_uniforms(pf, vm)
        self._set(pf, "u_fog_density", 0.0)
        amb = world.weather.ambient_temp_c
        self._fuel_uniforms(pf, item.species, [amb] * 3, [0.0] * 3, [0.0] * 3, [0.0] * 3, [min(1.0, item.m_surface)] * 3,
                            kind == "tinder", 0.0, item.material)
        if kind == "tinder":
            self._set(pf, "u_model", self._cam_item(frame, wp + u * 0.034 + f * 0.012, G.scale(0.058, 0.032, 0.058)))
            self.m_nest.vao(pf).render()
        else:
            # (big rounds / wedges are drawn a little slimmer so the held piece never fills the view)
            L = min(item.length, 0.36)
            rr = min(item.radius, 0.03 if self._is_split(item) else 0.045)
            axis = r * 0.7 + f * 0.7 + u * 0.1
            axis = axis / np.linalg.norm(axis)
            c = wp + u * (0.018 + rr) + f * 0.01 + axis * L * 0.36     # mostly ahead of the grip
            roll = 30.0
            if self._is_split(item):
                # hold a split piece bark-side out, riven faces away from the eye
                M0 = G.align_y_to(axis)
                want = u * 0.5 - f * 0.85
                roll = math.degrees(math.atan2(-float(np.dot(want, M0[:3, 2])), float(np.dot(want, M0[:3, 0]))))
            mesh, m = self._piece_mesh(item, G.orientation_model(c, axis, roll, rr, L, rr))
            self._set(pf, "u_model", m)
            mesh.vao(pf).render()

    # ================================================================== post & UI
    def _post_pass(self, frame, world):
        ctx = self.ctx
        mgl = self.mgl
        ctx.disable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND)
        ctx.disable(mgl.CULL_FACE)
        # bloom: progressive downsample then tent upsample
        src = self.scene_tex
        pd = self.p_down
        for i, (tex, fbo) in enumerate(self.bloom_levels):
            fbo.use()
            ctx.viewport = (0, 0, tex.width, tex.height)
            src.use(location=0)
            pd["u_src"].value = 0
            self._set(pd, "u_texel", (1.0 / src.width, 1.0 / src.height))
            self._set(pd, "u_first", 1 if i == 0 else 0)
            # bright-pass in exposed space: only truly bright things (flames, embers, the sun) bloom
            self._set(pd, "u_threshold", float(1.15 / max(0.2, frame["lighting"].exposure)))
            self.empty_vao["down"].render(mgl.TRIANGLES, vertices=3)
            src = tex
        pu = self.p_up
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE)
        for i in range(len(self.bloom_levels) - 1, 0, -1):
            src_t = self.bloom_levels[i][0]
            dst_t, dst_f = self.bloom_levels[i - 1]
            dst_f.use()
            ctx.viewport = (0, 0, dst_t.width, dst_t.height)
            src_t.use(location=0)
            pu["u_src"].value = 0
            self._set(pu, "u_texel", (1.0 / src_t.width, 1.0 / src_t.height))
            self._set(pu, "u_radius", 1.0)
            self.empty_vao["up"].render(mgl.TRIANGLES, vertices=3)
        ctx.disable(mgl.BLEND)
        # composite -> LDR
        L = frame["lighting"]
        pc = self.p_comp
        self.ldr_fbo.use()
        ctx.viewport = (0, 0, self.width, self.height)
        self.scene_tex.use(location=0)
        self.bloom_levels[0][0].use(location=1)
        self.depth_copy.use(location=2)
        pc["u_scene"].value = 0
        pc["u_bloom"].value = 1
        if "u_depth" in pc:
            pc["u_depth"].value = 2
        self._set(pc, "u_exposure", float(L.exposure))
        self._set(pc, "u_bloom_strength", 0.07 if self.preset["bloom"] else 0.0)
        self._set(pc, "u_time", float(self.time))
        grid = world.thermal_grid
        heat = (0.0, 0.0, 0.0, 0.0)
        top = 0.0
        fdepth = 1.0
        if grid.flame_hrr > 300.0 or grid.coal_bed.T > 500:
            base = np.array([grid.fire_centroid[0], grid.pit_center[1] + 0.1, grid.fire_centroid[2]])
            top_w = base + np.array([0.0, max(0.4, grid.flame_height * 1.8), 0.0])
            vp = frame["proj"] @ frame["view"]
            b4 = vp @ np.array([*base, 1.0])
            t4 = vp @ np.array([*top_w, 1.0])
            if b4[3] > 0.1 and t4[3] > 0.1:
                bs = b4[:2] / b4[3] * 0.5 + 0.5
                ts = t4[:2] / t4[3] * 0.5 + 0.5
                radius = max(0.03, min(0.35, 0.3 / b4[3] * (0.5 + grid.fire_diameter)))
                strength = min(1.0, (grid.flame_hrr + grid.char_hrr) / 15000.0 + 0.25)
                heat = (float(bs[0]), float(bs[1]), float(radius), float(strength))
                top = float(ts[1])
                fdepth = float(b4[3])
        self._set(pc, "u_heat", heat)
        self._set(pc, "u_heat_top", top)
        self._set(pc, "u_fire_depth", fdepth)
        self._set(pc, "u_near", CAMERA_NEAR)
        self._set(pc, "u_far", CAMERA_FAR)
        self._set(pc, "u_grade_shadow", L.grade_shadow)
        self._set(pc, "u_grade_high", L.grade_high)
        self._set(pc, "u_saturation", float(L.saturation))
        self._set(pc, "u_vignette", 0.32 + 0.12 * float(L.night))
        self._set(pc, "u_night_contrast", float(L.night) * 0.85)
        self._set(pc, "u_flash", float(min(0.35, getattr(world, "gas_flash_timer", 0.0) * 0.3)))
        self._set(pc, "u_aspect", (frame["aspect"], 1.0))
        self.empty_vao["comp"].render(mgl.TRIANGLES, vertices=3)
        # FXAA -> output
        out = self.out_fbo if self.out_fbo is not None else self.ctx.screen
        out.use()
        ctx.viewport = (0, 0, self.width, self.height)
        pf = self.p_fxaa
        self.ldr_tex.use(location=0)
        pf["u_src"].value = 0
        self._set(pf, "u_texel", (1.0 / self.width, 1.0 / self.height))
        self._set(pf, "u_enabled", 1 if self.preset["fxaa"] else 0)
        self.empty_vao["fxaa"].render(mgl.TRIANGLES, vertices=3)

    def _ui_pass(self, world, player, hands, smartphone, ref_book, input_mgr, extra_ui):
        ctx = self.ctx
        mgl = self.mgl
        img_bytes, changed = self.ui_renderer.render_if_changed(world, player, hands, smartphone, ref_book, input_mgr,
                                                                self, extra_ui)
        if changed and img_bytes is not None:
            self.ui_texture.write(img_bytes)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE_MINUS_SRC_ALPHA)
        self.ui_texture.use(location=0)
        self.p_ui["u_texture"].value = 0
        self.empty_vao["ui"].render(mgl.TRIANGLES, vertices=3)
        ctx.disable(mgl.BLEND)

    # ================================================================== capture
    def read_pixels(self) -> np.ndarray:
        """Returns the last rendered frame as an (h, w, 3) uint8 array (top row first)."""
        fbo = self.out_fbo if self.out_fbo is not None else self.ctx.screen
        data = fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        return img[::-1].copy()
