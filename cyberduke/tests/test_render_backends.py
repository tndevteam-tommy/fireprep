"""Pure-maths parts of the modern backend (no GL context), plus DrawContext helpers."""

import math

import numpy as np
import pytest

from cyberduke.render.backends import modern as m
from cyberduke.render.context import DrawContext


# ----------------------------------------------------------------------------- matrices
def test_perspective_matches_the_fixed_function_frustum():
    p = m.perspective(60.0, 4 / 3, 0.1, 55.0)
    f = 1 / math.tan(math.radians(30.0))
    assert p[0, 0] == pytest.approx(f / (4 / 3)) and p[1, 1] == pytest.approx(f)
    assert p[3, 2] == -1.0 and p[3, 3] == 0.0
    # near plane maps to -1, far plane to +1 in NDC
    for z, expect in ((-0.1, -1.0), (-55.0, 1.0)):
        clip = p @ np.array([0, 0, z, 1], dtype=np.float32)
        assert clip[2] / clip[3] == pytest.approx(expect, abs=1e-4)


def test_look_at_puts_the_target_on_the_negative_z_axis():
    v = m.look_at((1.0, 2.0, 3.0), (1.0, 2.0, -7.0))
    eye = v @ np.array([1, 2, 3, 1], dtype=np.float32)
    target = v @ np.array([1, 2, -7, 1], dtype=np.float32)
    assert np.allclose(eye[:3], 0, atol=1e-6)
    assert np.allclose(target[:3], (0, 0, -10), atol=1e-5)


def test_rotation_translation_scaling_compose_like_gl():
    mm = m.translation(1, 0, 0) @ m.rotation(90, 0, 1, 0) @ m.scaling(2, 2, 2)
    p = mm @ np.array(
        [1, 0, 0, 1], dtype=np.float32
    )  # scale->(2,0,0), rotate 90 about Y->(0,0,-2), translate
    assert np.allclose(p[:3], (1, 0, -2), atol=1e-5)


def test_normal_matrix_corrects_non_uniform_scale():
    model = m.scaling(1, 4, 1)
    n = m.normal_matrix(model) @ np.array([0, 1, 0], dtype=np.float32)
    assert np.allclose(n / np.linalg.norm(n), (0, 1, 0))
    side = m.normal_matrix(m.scaling(1, 4, 1) @ m.rotation(45, 0, 0, 1)) @ np.array(
        [1, 0, 0], dtype=np.float32
    )
    assert side[0] > 0


def test_ortho_maps_pixels_y_down():
    o = m.ortho(1024, 768)
    tl = o @ np.array([0, 0, 0, 1], dtype=np.float32)
    br = o @ np.array([1024, 768, 0, 1], dtype=np.float32)
    assert np.allclose(tl[:2], (-1, 1)) and np.allclose(br[:2], (1, -1))


# ----------------------------------------------------------------------------- geometry
def test_unit_cube_has_24_vertices_with_outward_normals_and_matches_legacy_extent():
    cube = m.unit_cube_vertices()
    assert cube.shape == (24, 8)
    assert cube[:, :3].min() == -0.5 and cube[:, :3].max() == 0.5
    for row in cube:
        pos, normal = row[:3], row[3:6]
        assert np.dot(pos, normal) == pytest.approx(0.5), "normal points out of its face"
    assert len(m.quad_indices(6)) == 36


def test_unit_cylinder_geometry():
    verts, idx = m.unit_cylinder_vertices(8)
    assert verts[:, 1].min() == 0.0 and verts[:, 1].max() == 1.0
    assert len(idx) == 8 * 6 + 2 * 8 * 3
    radii = np.hypot(verts[:, 0], verts[:, 2])
    assert set(np.round(radii, 6)) <= {0.0, 1.0}


def test_with_material_expands_to_the_shared_layout():
    base = m.unit_cube_vertices()
    full = m.with_material(base, (0.1, 0.2, 0.3), (1, 2, 3), (0.5, 32))
    assert full.shape == (24, m.STRIDE_FLOATS)
    assert np.allclose(full[:, 8:11], (0.1, 0.2, 0.3)) and np.allclose(full[:, 14:16], (0.5, 32))
    assert m.STRIDE_BYTES == m.STRIDE_FLOATS * 4 == sum(a[1] for a in m.ATTRIBS) * 4


# ----------------------------------------------------------------------------- context helpers
class RecordingContext(DrawContext):
    """Minimal DrawContext that records every call (exercises the shared helper methods)."""

    name = "recording"

    def __init__(self):
        self.calls = []

    def _rec(self, *args):
        self.calls.append(args)

    def begin_frame(self, eye, target):
        self._rec("begin_frame", eye, target)

    def end_frame(self):
        self._rec("end_frame")

    def push(self):
        self._rec("push")

    def pop(self):
        self._rec("pop")

    def translate(self, x, y, z):
        self._rec("translate", x, y, z)

    def rotate(self, a, x, y, z):
        self._rec("rotate", a, x, y, z)

    def scale(self, x, y, z):
        self._rec("scale", x, y, z)

    def color(self, r, g, b):
        self._rec("color", r, g, b)

    def emission(self, rgb):
        self._rec("emission", rgb)

    def specular(self, rgb, shininess=0.0):
        self._rec("specular", rgb, shininess)

    def cube(self, tex_id=None):
        self._rec("cube", tex_id)

    def cylinder(self, r_bot, r_top, h, slices=10, tex_id=None):
        self._rec("cylinder", r_bot, r_top, h, slices, tex_id)

    def lines(self, segments, rgba, width=1.0):
        self._rec("lines", len(segments), rgba)

    def soft_disc(self, rgba, slices=24, inner=0.0):
        self._rec("soft_disc", rgba, slices, inner)

    def cached(self, key, build):
        self._rec("cached", key)
        build()

    def invalidate(self, key=None):
        self._rec("invalidate", key)

    def begin_overlay(self, width, height):
        self._rec("begin_overlay", width, height)

    def end_overlay(self):
        self._rec("end_overlay")

    def rect(self, x, y, w, h, rgba):
        self._rec("rect", x, y, w, h)

    def rect_outline(self, x, y, w, h, rgba, width=2.0):
        self._rec("rect_outline", x, y, w, h)

    def image(self, tex_id, x, y, w, h):
        self._rec("image", tex_id, x, y, w, h)

    def upload_texture_rgba(self, data, w, h):
        self._rec("upload", w, h)
        return 1

    def delete_texture(self, tex_id):
        self._rec("delete", tex_id)


def _make_recorder():
    return RecordingContext()


def test_transformed_and_emissive_helpers_bracket_state_correctly():
    ctx = _make_recorder()
    with ctx.transformed((1, 2, 3), rotate=((90, 0, 1, 0),), scale=(2, 2, 2)):
        ctx.cube(None)
    names = [c[0] for c in ctx.calls]
    assert names == ["push", "translate", "rotate", "scale", "cube", "pop"]
    ctx.calls.clear()
    with ctx.emissive((1, 0, 0), (0.5, 0.5, 0.5)):
        pass
    assert [c[0] for c in ctx.calls] == ["emission", "color", "emission", "color"]
    assert ctx.calls[-2] == ("emission", None) and ctx.calls[-1] == ("color", 1.0, 1.0, 1.0)


def test_box_helper_scales_a_unit_cube():
    ctx = _make_recorder()
    ctx.box(1, 2, 3, 4, 5, 6, 7)
    assert ("translate", 1, 2, 3) in ctx.calls and ("scale", 4, 5, 6) in ctx.calls
    assert ("cube", 7) in ctx.calls


def test_every_renderer_draws_through_the_context_only():
    """The whole scene can be drawn into a recording context: no renderer touches GL directly."""
    from cyberduke.render.entity_renderers import (
        draw_security_camera,
        draw_terminal,
        draw_vending_machine,
    )
    from cyberduke.render.humanoid import draw_npc
    from cyberduke.render.sectors import draw_room
    from cyberduke.simulation import Simulation

    sim = Simulation()
    ctx = _make_recorder()
    textures = {}
    draw_room(ctx, sim.room, textures)
    for t in sim.terminals:
        draw_terminal(ctx, t, textures)
    for v in sim.vending_machines:
        draw_vending_machine(ctx, v, textures)
    for c in sim.cameras:
        draw_security_camera(ctx, c, sim.obstacles)
    for n in sim.npcs:
        draw_npc(ctx, n, textures)
    names = [c[0] for c in ctx.calls]
    assert names.count("push") == names.count("pop"), "unbalanced matrix stack"
    assert names.count("cube") > 300 and names.count("cylinder") > 10 and "lines" in names
    sectors = len(sim.level.sectors)
    assert names.count("cached") == sectors + len(sim.terminals) + len(sim.vending_machines)


def test_pulsing_props_are_drawn_dynamically_with_breathing_emission():
    from cyberduke.levels import build_facility
    from cyberduke.render.sectors import draw_pulsing_props, is_pulsing, pulse_material

    level = build_facility()
    greenhouse = level.sector("FungalGreenhouse")
    pulsing = [p for p in greenhouse.props if is_pulsing(p)]
    assert len(pulsing) > 40 and all(p.material.emission for p in pulsing)
    assert not any(is_pulsing(p) for p in level.sector("StagingBay").props), "biodome only"
    cap = pulsing[0]
    e0 = pulse_material(cap, 0.0).emission
    e1 = pulse_material(cap, 1.0).emission
    assert e0 != e1 and all(0.0 <= c <= m for c, m in zip(e1, cap.material.emission, strict=True))
    ctx = _make_recorder()
    draw_pulsing_props(ctx, greenhouse, {}, time=2.0)
    names = [c[0] for c in ctx.calls]
    assert names.count("cylinder") == len(pulsing) and "cached" not in names


def test_sector_culling_only_ever_drops_what_is_beyond_the_view_distance():
    """Regression: culling used to hide whole sectors behind shut doors - but a sector owns
    the walls it shares with its neighbours (the vault owns the partition wall around the
    airlock door), so the player could see straight through them into the next room. Only
    distance may cull, and only beyond the far plane."""
    from cyberduke.render.sectors import visible_sectors
    from cyberduke.simulation import Simulation

    sim = Simulation()
    room = sim.room

    def names(z, view_distance):
        return sorted(s.name for s in visible_sectors(room, (0.0, 1.6, z), view_distance))

    all_names = sorted(s.name for s in room.sectors)
    # Standing in the airlock with every door shut: the vault's partition wall is still
    # drawn, because the vault sector is still drawn.
    assert "DatacoreVault" in names(14.0, 55.0)
    assert names(14.0, 200.0) == all_names
    # Far-away sectors drop out only when they are beyond the far plane.
    assert names(-8.0, 15.0) == ["StagingBay"], "the airlock starts 18 m away"
    assert names(-8.0, 20.0) == ["AirlockCorridor", "StagingBay"]
    assert "FungalGreenhouse" not in names(6.0, 30.0)
    assert "FungalGreenhouse" in names(6.0, 40.0), "36 m away: inside a 40 m view distance"
    # Doors never matter to culling any more.
    before = names(14.0, 55.0)
    for door in room.doors.values():
        door.open_amount = door.travel
    assert names(14.0, 55.0) == before


def test_quality_settings_drive_the_scene_detail():
    """Low quality draws fewer beam rays, no sparks and no scan ring; high draws them all."""
    from cyberduke.entities import SecurityCamera
    from cyberduke.render.effects import draw_scan_pulse
    from cyberduke.render.entity_renderers import draw_security_camera, draw_vending_machine
    from cyberduke.render.quality import Quality, settings_for
    from cyberduke.simulation import Simulation
    from cyberduke.systems.scanner import ScanPulse

    low, high = settings_for(Quality.LOW), settings_for(Quality.HIGH)
    cam = SecurityCamera(0.0, 3.6, 0.0, camera_id="c")

    def beam_segments(slices):
        ctx = _make_recorder()
        draw_security_camera(ctx, cam, [], slices=slices)
        return sum(c[1] for c in ctx.calls if c[0] == "lines")

    assert beam_segments(low.beam_slices) < beam_segments(high.beam_slices)

    vm = Simulation().vending_machines[0]
    vm.trigger_distraction()

    def vending_lines(sparks):
        ctx = _make_recorder()
        draw_vending_machine(ctx, vm, {}, sparks=sparks)
        return len([c for c in ctx.calls if c[0] == "lines"])

    assert vending_lines(False) == 0 and vending_lines(True) >= 0
    vm.anim_time = 0.0  # a burst frame
    assert vending_lines(True) > vending_lines(False)

    ctx = _make_recorder()
    draw_scan_pulse(ctx, ScanPulse((0.0, 0.0, 0.0), 0.3, 16.0, 1.0))
    assert [c for c in ctx.calls if c[0] == "lines"], "the ring is a line effect"


def test_pulsing_props_are_skipped_beyond_the_effect_distance():
    from cyberduke.levels import build_facility
    from cyberduke.render.sectors import draw_pulsing_props, is_pulsing

    greenhouse = build_facility().sector("FungalGreenhouse")
    total = sum(1 for p in greenhouse.props if is_pulsing(p))
    near = _make_recorder()
    draw_pulsing_props(near, greenhouse, {}, 0.0, (0.0, 1.6, 61.0), 10.0)
    far = _make_recorder()
    draw_pulsing_props(far, greenhouse, {}, 0.0, (0.0, 1.6, 61.0), 100.0)
    drawn_near = near.calls.count(("cylinder", 0.26, 0.04, 0.16, 10, None))
    assert 0 < len([c for c in near.calls if c[0] == "cylinder"]) < total
    assert len([c for c in far.calls if c[0] == "cylinder"]) == total
    del drawn_near


def test_contexts_accept_the_quality_calls_even_when_they_cannot_honour_them():
    """The quality hooks are optional capabilities with no-op defaults, so a minimal
    context (and the test recorder) keeps working."""
    ctx = _make_recorder()
    ctx.set_view_distance(40.0)
    ctx.set_lights([])
    ctx.set_global_ambient((0.1, 0.1, 0.1))


def test_the_street_draws_through_the_context_only_with_its_crowd():
    from cyberduke.levels import build_night_market
    from cyberduke.render.humanoid import draw_npc
    from cyberduke.render.sectors import draw_room, draw_translucent_props
    from cyberduke.simulation import Simulation

    sim = Simulation(build_night_market())
    ctx = _make_recorder()
    draw_room(ctx, sim.room, {})
    for n in sim.npcs:
        draw_npc(ctx, n, {})
    draw_translucent_props(ctx, list(sim.room.sectors), {}, 0.0, (0.0, 1.6, 0.0))
    names = [c[0] for c in ctx.calls]
    assert names.count("push") == names.count("pop"), "unbalanced matrix stack"
    assert names.count("cached") == 1, "one sector, cached once"
    assert names.count("soft_disc") >= 5, "lantern pools, neon on the pavement, puddles"
    assert names.count("cube") > len(sim.npcs) * 15, "eight people and a lot of scenery"


def test_the_scene_renderer_follows_a_level_swap(monkeypatch):
    """After the player steps out of the facility the renderer must drop the facility's
    cached geometry and pick up the street's lights, without being told."""
    import pygame

    from cyberduke.render import scene
    from cyberduke.simulation import InputState, Simulation

    pygame.font.init()
    recorder = _make_recorder()
    monkeypatch.setattr(scene, "create_context", lambda *a, **k: recorder)
    monkeypatch.setattr(scene, "load_texture_suite", lambda: {})
    sim = Simulation()
    renderer = scene.SceneRenderer(scene.ViewportConfig(), sim.room, backend="legacy")
    facility_lights = list(renderer.room_lights)
    renderer.draw_frame(sim, 1 / 60)
    keys = [c[1] for c in recorder.calls if c[0] == "cached"]
    assert ("sector", "StagingBay") in keys and ("invalidate", None) not in recorder.calls
    from cyberduke.config import SIMULATION

    sim.player.x, sim.player.z = 0.0, 78.6
    for _ in range(int(SIMULATION.level_fade_out * 60) + 2):  # (the fade out at the door)
        sim.update(1 / 60, InputState())
    assert sim.level.name == "Night Market Alley"
    recorder.calls.clear()
    renderer.draw_frame(sim, 1 / 60)
    assert ("invalidate", None) in recorder.calls, "the facility's display lists are gone"
    keys = [c[1] for c in recorder.calls if c[0] == "cached"]
    assert ("sector", "NightMarketAlley") in keys and ("sector", "StagingBay") not in keys
    assert renderer.room_lights != facility_lights
    assert renderer.room_lights == list(sim.room.get_lights())
    recorder.calls.clear()
    renderer.draw_frame(sim, 1 / 60)
    assert ("invalidate", None) not in recorder.calls, "only once per swap"


# ----------------------------------------------------------------------------- legacy lights
def _record_gl(monkeypatch, module, names, log):
    for name in names:
        monkeypatch.setattr(module, name, lambda *a, _n=name: log.append((_n, a)))


def test_legacy_backend_places_the_lights_in_the_world_every_frame(monkeypatch):
    """Regression: fixed-function GL stores GL_POSITION in eye space using the modelview
    matrix current at the call. The positions were only ever set from ``setup_lighting``
    (identity or a stale view), so every light hung off the camera and moved with it. They
    must be re-specified after each frame's view transform."""
    from cyberduke.render import lighting
    from cyberduke.render.backends import legacy
    from cyberduke.world.lights import Light

    log = []
    _record_gl(monkeypatch, legacy, ["glClear", "glLoadIdentity"], log)
    monkeypatch.setattr(legacy, "apply_look_at", lambda eye, target: log.append(("look_at",)))
    _record_gl(monkeypatch, lighting, ["glLightfv"], log)
    lights = [
        Light(1.0, 2.0, 3.0, (0, 0, 0), (1, 1, 1), (1, 1, 1)),
        Light(4.0, 5.0, 6.0, (0, 0, 0), (1, 1, 1), (1, 1, 1)),
    ]
    ctx = object.__new__(legacy.LegacyContext)
    ctx._lights = lights
    ctx.begin_frame((0.0, 1.6, 0.0), (0.0, 1.6, 1.0))
    names = [entry[0] for entry in log]
    assert "look_at" in names
    after_view = log[names.index("look_at") + 1 :]
    positions = [a for n, a in after_view if n == "glLightfv" and a[1] == lighting.GL_POSITION]
    assert [tuple(p[2][:3]) for p in positions] == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_legacy_backend_keeps_the_presets_ambient_when_the_light_set_changes(monkeypatch):
    """Regression: ``setup_lighting`` reset the global ambient to its built-in base, so the
    quality preset's ambient (LOW's flatter level) was lost on the first light refresh."""
    from cyberduke.render import lighting
    from cyberduke.render.backends import legacy

    log = []
    _record_gl(
        monkeypatch,
        lighting,
        ["glEnable", "glDisable", "glColorMaterial", "glLightfv", "glLightf"],
        log,
    )
    _record_gl(monkeypatch, lighting, ["glLightModelfv"], log)
    _record_gl(monkeypatch, legacy, ["glLightModelfv"], log)
    ctx = object.__new__(legacy.LegacyContext)
    ctx._lights, ctx._ambient = [], tuple(lighting.AMBIENT_BASE[:3])
    ctx.set_global_ambient((0.34, 0.40, 0.52))
    ctx.set_lights([])
    ambients = [a[1] for n, a in log if n == "glLightModelfv"]
    assert ambients[-1] == (0.34, 0.40, 0.52, 1.0)


# ----------------------------------------------------------------------------- tiled textures
def test_both_backends_tile_a_cube_face_along_the_same_axes():
    """A tiled texture repeats by the world size of each face: the modern shader / static
    batch path (``tiled_uvs``) and the legacy display lists must agree per face."""
    from cyberduke.render.backends.modern import tiled_uvs, unit_cube_vertices
    from cyberduke.render.primitives import _CUBE_FACES

    rep = (2.0, 3.0, 4.0)
    base = unit_cube_vertices()
    uv = tiled_uvs(base, 1, rep)
    for f, (normal, _corners, (ua, va)) in enumerate(_CUBE_FACES):
        rows = slice(4 * f, 4 * f + 4)
        assert tuple(base[rows][0, 3:6]) == normal
        assert uv[rows, 0].max() == pytest.approx(rep[ua]) and uv[rows, 1].max() == pytest.approx(
            rep[va]
        )
    assert np.array_equal(tiled_uvs(base, 0, rep), base[:, 6:8]), "untiled: unchanged"


def test_tile_repeats_follow_the_real_size_of_the_shape():
    from cyberduke.render.context import DrawContext

    assert DrawContext.cube_repeat((4.0, 2.0, 1.0), 2.0) == pytest.approx((2.0, 1.0, 0.5))
    around, along, across = DrawContext.cylinder_repeat(0.5, 0.5, 3.0, (1.0, 1.0, 1.0), 1.0)
    assert around == pytest.approx(math.pi) and along == pytest.approx(3.0)
    assert across == pytest.approx(1.0)


def test_legacy_backend_measures_the_transformed_cube_for_tiling(monkeypatch):
    from cyberduke.render.backends import legacy

    for name in ("glPushMatrix", "glPopMatrix", "glRotatef", "glScalef", "glTranslatef"):
        monkeypatch.setattr(legacy, name, lambda *a: None)
    calls = []
    monkeypatch.setattr(legacy, "draw_cube", lambda *a: calls.append(a))
    ctx = object.__new__(legacy.LegacyContext)
    ctx._lin, ctx._lin_stack = np.identity(3), []
    ctx.set_texture_tiles({7: 2.0})
    ctx.push()
    ctx.translate(5.0, 0.0, 0.0)
    ctx.rotate(90.0, 0.0, 1.0, 0.0)  # turning it does not change its size
    ctx.scale(4.0, 2.0, 1.0)
    ctx.cube(7)
    ctx.cube(8)  # not a tiled texture: stretched once over each face
    ctx.pop()
    assert calls[0][0] == 7 and calls[0][1] == pytest.approx((2.0, 1.0, 0.5))
    assert calls[1] == (8,)
    assert np.array_equal(ctx._lin, np.identity(3)), "pop restores the size tracking"


# ----------------------------------------------------------------------------- depth
def test_logarithmic_depth_spans_the_view_distance():
    from cyberduke.render.backends.modern import log_depth_coefficient

    k = log_depth_coefficient(55.0)
    depth = [math.log2(1.0 + d) * k for d in (0.2, 1.0, 10.0, 30.0, 55.0)]
    assert depth == sorted(depth) and depth[-1] == pytest.approx(1.0) and depth[0] > 0.0


def test_the_window_asks_for_a_24_bit_depth_buffer(monkeypatch):
    """With SDL's default 16-bit depth buffer, layers a centimetre apart z-fought at a
    distance (black lines flickering through signs and screens)."""
    import OpenGL.GL
    import pygame

    from cyberduke.render import scene

    log = []
    monkeypatch.setattr(
        pygame.display, "gl_set_attribute", lambda attr, value: log.append((attr, value))
    )

    def set_mode(size, flags):
        log.append(("set_mode", size))
        if any(v == 24 for a, v in log if a == pygame.GL_DEPTH_SIZE) and len(log) < 3:
            raise pygame.error("no 24-bit depth here")

    monkeypatch.setattr(pygame.display, "set_mode", set_mode)
    monkeypatch.setattr(OpenGL.GL, "glGetIntegerv", lambda name: 16)
    bits = scene.open_window(scene.ViewportConfig())
    assert log[0] == (pygame.GL_DEPTH_SIZE, 24), "asks for 24 bits first"
    assert (pygame.GL_DEPTH_SIZE, 16) in log and log[-1][0] == "set_mode", "then falls back"
    assert bits == 16


# ----------------------------------------------------------------------------- light pools
def test_a_clipped_light_pool_stays_inside_its_rectangle_and_fades_out():
    from cyberduke.render.glow import disc_triangles, pool_triangles
    from cyberduke.world.props import Material, Prop

    whole = Prop.disc(1.0, 0.5, 2.0, 0.4, 0.4, 0.2, Material(None, (0.3, 0.6, 1.0), alpha=0.5))
    tris = disc_triangles(whole)
    assert len(tris) % 3 == 0 and len(tris) > 0
    r = np.hypot(tris[:, 0] - 1.0, tris[:, 2] - 2.0)
    assert np.allclose(tris[:, 1], 0.5)
    assert tris[r < 0.05, 3].min() == pytest.approx(1.0), "full in the middle"
    assert tris[r > 0.399, 3].max() <= 1e-6, "nothing at the rim"
    cut = replace_clip(whole, (-1.0, 0.25, -0.5, 1.0))  # stops 0.1 m right of the centre
    tris = disc_triangles(cut)
    assert tris[:, 0].max() <= 1.0 + 0.1 + 1e-6 and tris[:, 2].min() >= 2.0 - 0.2 - 1e-6
    baked = pool_triangles([cut])
    assert baked[:, 3].max() == pytest.approx(0.5), "each pool's own opacity is baked in"


def replace_clip(prop, clip):
    from dataclasses import replace

    return replace(prop, clip=clip)
