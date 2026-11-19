"""
CPU side of the renderer: procedural meshes, particles, sky/lighting model, hand rig and the
2D UI overlay.  (The GPU passes are covered by tests/integration + tests/visual.)
"""

import inspect
import math

import numpy as np
import pytest

from firestarter.rendering import mesh_generator as mg
from firestarter.rendering import particle_system as ps
from firestarter.rendering.atmosphere import Atmosphere
from firestarter.simulation.weather import WeatherSystem, Season
from firestarter.input.hand_physics_rig import HandController


# ------------------------------------------------------------------------ meshes
def _all_default_mesh_builders():
    out = []
    for name, fn in inspect.getmembers(mg, inspect.isfunction):
        if not name.startswith("create_") or fn.__module__ != mg.__name__:
            continue
        sig = inspect.signature(fn)
        if all(p.default is not inspect.Parameter.empty for p in sig.parameters.values()):
            out.append((name, fn))
    return out


@pytest.mark.parametrize("name,fn", _all_default_mesh_builders())
def test_mesh_is_valid(name, fn):
    res = fn()
    meshes = res if isinstance(res, tuple) else (res,)
    for m in meshes:
        if not isinstance(m, mg.MeshData):
            continue
        v = m.as_array()
        assert m.vertex_count > 0 and m.index_count % 3 == 0, name
        assert np.isfinite(v).all(), name
        assert m.indices.max() < m.vertex_count, name
        n = np.linalg.norm(v[:, 3:6], axis=1)
        assert np.all(np.abs(n - 1.0) < 0.02), name                       # unit normals
        assert np.all((v[:, 8:12] >= 0.0) & (v[:, 8:12] <= 1.5)), name    # sane vertex colours
        tri = m.indices.reshape(-1, 3)
        p = v[:, 0:3]
        area = np.linalg.norm(np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]]), axis=1)
        assert (area > 1e-12).mean() > 0.97, name                         # (almost) no degenerate triangles


def test_trees_have_natural_proportions():
    trunk, crown = mg.create_conifer("pine", 0, 15.0)
    lo, hi = crown.as_array()[:, 1].min(), crown.as_array()[:, 1].max()
    assert 12.0 < hi < 17.0 and lo > 0.5
    trunk, crown = mg.create_deciduous("oak", 0, 14.0)
    c = crown.as_array()
    width = c[:, 0].max() - c[:, 0].min()
    assert 0.3 * 14.0 < width < 1.1 * 14.0                                  # broad, rounded crown


def test_terrain_mesh_matches_height_field():
    from firestarter.simulation import terrain
    m = mg.create_terrain_mesh(size=40.0, resolution=21, elevation_fn=terrain.height)
    v = m.as_array()
    h = terrain.height(v[:, 0].astype(float), v[:, 2].astype(float))
    assert np.allclose(v[:, 1], h, atol=0.02)


# ------------------------------------------------------------------------ particles
def test_smoke_rises_and_fades():
    s = ps.ParticleSystem(2000)
    for _ in range(30):
        s.spawn_pyrolysis_smoke(np.zeros(3), 2.0, 1 / 30)
    y0 = s.pos[s.alive][:, 1].mean()
    for _ in range(60):
        s.update(1 / 30)
    alive = s.alive & (s.kind == ps.K_SMOKE)
    assert s.pos[alive][:, 1].mean() > y0 + 0.3
    assert s.size[alive].mean() > 0.2                                        # plumes billow out


def test_steam_is_white_and_smoke_is_yellow_grey():
    s = ps.ParticleSystem(2000)
    s.spawn_steam(np.zeros(3), 0.01, 0.5)
    s.spawn_pyrolysis_smoke(np.array([1.0, 0, 0]), 2.0, 0.5)
    steam = s.col[s.alive & (s.kind == ps.K_STEAM)][:, :3].mean(axis=0)
    smoke = s.col[s.alive & (s.kind == ps.K_SMOKE)][:, :3].mean(axis=0)
    assert steam.min() > 0.85
    assert smoke[0] > smoke[2] + 0.05 and smoke.max() < 0.75


def test_sparks_cool_from_yellow_to_red_and_die():
    s = ps.ParticleSystem(500)
    s.spawn_sparks(np.zeros(3), 60)
    s.update(1 / 60)
    g0 = s.col[s.alive & (s.kind == ps.K_SPARK), 1].mean()
    for _ in range(40):
        s.update(1 / 60)
    alive = s.alive & (s.kind == ps.K_SPARK)
    if alive.any():
        assert s.col[alive, 1].mean() < g0
    for _ in range(400):
        s.update(1 / 60)
    assert s.count(ps.K_SPARK) == 0


def test_rain_falls_and_dies_on_the_ground():
    s = ps.ParticleSystem(6000)
    cam = np.array([0.0, 1.7, 0.0])
    s.spawn_weather(cam, rain_intensity=1.0, dt=0.1)
    n = s.count(ps.K_RAIN)
    assert n > 200
    for _ in range(60):
        s.update(1 / 30, ground_fn=lambda x, z: np.zeros_like(x))
    assert s.count(ps.K_RAIN) < 0.05 * n
    a, add = s.get_instances(cam)
    assert a.shape[1] == 12


def test_instances_are_split_and_sorted_for_blending():
    s = ps.ParticleSystem(3000)
    s.spawn_pyrolysis_smoke(np.zeros(3), 10.0, 1.0)
    s.spawn_embers(np.zeros(3), 50.0, 1.0)
    cam = np.array([3.0, 1.0, 0.0])
    alpha, add = s.get_instances(cam)
    assert len(alpha) and len(add)
    d = np.sum((alpha[:, :3] - cam) ** 2, axis=1)
    assert np.all(np.diff(d) <= 1e-5)                                         # back to front
    assert np.all(add[:, 11] == 6.0)


def test_particle_pool_never_overflows():
    s = ps.ParticleSystem(300)
    for _ in range(50):
        s.spawn_sparks(np.zeros(3), 100)
    assert s.count() <= 300


# ------------------------------------------------------------------------ atmosphere
def _light(hour, season=Season.SUMMER, rain=0.0, fire_kw=0.0, settle=True):
    w = WeatherSystem(season=season, start_hour=hour)
    w.forced_cloud, w.cloud_cover = 0.1, 0.1
    if rain:
        w.force_precipitation(rain)
    a = Atmosphere()
    s = a.compute(w, 0.0, fire_kw, 2.0)
    if settle:
        for _ in range(40):
            s = a.compute(w, 0.5, fire_kw, 2.0)
    return s


def test_noon_sky_is_blue_and_bright_night_is_dark():
    noon, night = _light(12.5), _light(1.0)
    assert noon.zenith[2] > noon.zenith[0] and noon.zenith[2] > 0.5
    assert np.mean(night.zenith) < 0.03
    assert np.linalg.norm(noon.light_color) > 10 * np.linalg.norm(night.light_color)


def test_golden_hour_is_warm():
    g = _light(19.3)          # sun ~4 deg above the horizon on the summer solstice
    assert g.horizon[0] > g.horizon[2] + 0.2
    assert g.light_color[0] > g.light_color[2]


def test_eyes_adapt_to_darkness_and_fire_pulls_exposure_back():
    day, night = _light(12.0), _light(1.0)
    night_fire = _light(1.0, fire_kw=40.0)
    assert night.exposure > 2 * day.exposure
    assert night_fire.exposure < night.exposure


def test_overcast_rain_desaturates_and_fogs():
    clear, rain = _light(13.0), _light(13.0, rain=1.0)
    assert rain.fog_density > 2 * clear.fog_density
    assert rain.saturation < clear.saturation


def test_fire_light_grows_with_sqrt_of_heat_release(state):
    from firestarter.automation.staging import build_log_fire
    a = Atmosphere()
    g = state.world.thermal_grid
    pos, col, i0, r0 = a.fire_light(g, 0.0, 1.0)
    assert i0 == 0.0
    build_log_fire(state.world)
    for _ in range(900):
        state.step(1 / 30, move=False)
    for k in range(30):
        pos, col, i1, r1 = a.fire_light(g, k * 0.05, 0.05)
    assert i1 > 0.5 and r1 > 5.0
    assert col[0] >= col[1] >= col[2]                                           # orange, not white/blue
    assert pos[1] > g.pit_center[1]


# ------------------------------------------------------------------------ hands
def test_hand_springs_settle_into_view():
    h = HandController()
    for _ in range(120):
        h.update(1 / 60, "idle", {"right_item": "pocket_lighter", "left_item": "tinder"})
    m = h.get_hand_render_matrices()
    for side in ("left", "right"):
        p = m[f"{side}_pos"]
        assert 0.25 < p[2] < 0.7                                                # in front of the camera
        assert abs(p[0]) < 0.4 and -0.35 < p[1] < 0.0                         # lower corners of the view
    assert m["right_pos"][0] > 0 > m["left_pos"][0]


def test_lit_lighter_reaches_toward_the_target():
    h = HandController()
    for _ in range(90):
        h.update(1 / 60, "idle", {"right_item": "pocket_lighter", "lighter_lit": True,
                                  "reach_target": np.array([0.0, -0.8, 1.0])})
    assert h.pos_right[1] < -0.3 and h.pos_right[2] > 0.3


def test_empty_off_hand_drops_out_of_view():
    h = HandController()
    for _ in range(120):
        h.update(1 / 60, "idle", {"left_item": "none"})
    assert h.pos_left[1] < -0.3


# ------------------------------------------------------------------------ UI overlay
@pytest.fixture
def ui():
    from firestarter.rendering.ui_renderer import UIRenderer
    return UIRenderer(1280, 720)


def _render(ui, st):
    return np.asarray(ui.render_image(st.world, st.world.player, st.hands, st.phone, st.book, st.input, None, None))


def _alpha_box(img, thr=200):
    ys, xs = np.nonzero(img[:, :, 3] >= thr)
    return xs.min(), ys.min(), xs.max(), ys.max()


def test_hud_keeps_the_view_clear(ui, state):
    state.step(0.1)
    img = _render(ui, state)
    assert img.shape == (720, 1280, 4)
    centre = img[200:520, 300:980, 3]
    assert (centre > 30).mean() < 0.04                                          # only crosshair / prompts
    assert (img[:, :, 3] > 0).mean() > 0.01                                     # but a HUD is drawn


@pytest.mark.parametrize("screen", ["phone_home", "phone_store", "phone_map", "book", "time", "pause", "help"])
def test_every_menu_screen_renders_inside_the_window(ui, state, screen):
    st = state
    inp = st.input
    if screen.startswith("phone"):
        inp.handle_key_down("tab")
        if screen == "phone_store":
            inp.handle_key_down("return")
        elif screen == "phone_map":
            inp.handle_key_down("d")
            inp.handle_key_down("return")
    elif screen == "book":
        inp.handle_key_down("b")
    elif screen == "time":
        inp.handle_key_down("t")
    elif screen == "pause":
        inp.handle_key_down("escape")
    elif screen == "help":
        inp.handle_key_down("h")
    img = _render(ui, st)
    x0, y0, x1, y1 = _alpha_box(img)
    assert x0 >= 2 and y0 >= 2 and x1 <= 1277 and y1 <= 717, (x0, y0, x1, y1)
    solid = (img[:, :, 3] >= 200).mean()
    assert solid > 0.05, solid                                                  # a real panel is on screen
    rgb = img[:, :, :3][img[:, :, 3] >= 200].astype(float)
    assert rgb.std() > 20                                                       # not a flat slab: text / art


def test_ui_scales_with_resolution(state):
    from firestarter.rendering.ui_renderer import UIRenderer
    state.input.handle_key_down("tab")
    small = np.asarray(UIRenderer(960, 540).render_image(state.world, state.world.player, state.hands, state.phone,
                                                        state.book, state.input, None, None))
    big = np.asarray(UIRenderer(1920, 1080).render_image(state.world, state.world.player, state.hands, state.phone,
                                                        state.book, state.input, None, None))
    fs = (small[:, :, 3] >= 200).mean()
    fb = (big[:, :, 3] >= 200).mean()
    assert fs == pytest.approx(fb, rel=0.25)


def test_ui_rerenders_only_when_something_changes(ui, state):
    a, changed = ui.render_if_changed(state.world, state.world.player, state.hands, state.phone, state.book, state.input)
    assert changed
    b, changed2 = ui.render_if_changed(state.world, state.world.player, state.hands, state.phone, state.book, state.input)
    assert not changed2
    state.input.handle_key_down("tab")
    c, changed3 = ui.render_if_changed(state.world, state.world.player, state.hands, state.phone, state.book, state.input)
    assert changed3
