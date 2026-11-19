"""
The OpenGL renderer initialises, renders every pass at every graphics preset, survives
window resizes and menus, and produces a sensible image.  Needs a display (Xvfb is fine).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gl


@pytest.fixture(scope="module")
def game(request):
    import os
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        pytest.skip("no display: run under xvfb-run")
    from firestarter.game import Game
    g = Game(width=640, height=360, preset="Low", audio=False, tutorial=False, fixed_dt=1 / 30)
    yield g
    g.pg.quit()


def frame(game, n=1):
    for _ in range(n):
        game.tick(1 / 30, render=True)
    return game.renderer.read_pixels()


def test_gl_context_is_core_33_or_better(game):
    r = game.renderer
    assert r.is_gl_available, r.init_error
    assert r.ctx.version_code >= 330


def test_first_frame_is_a_real_scene(game):
    img = frame(game, 3)
    assert img.shape == (360, 640, 3)
    assert img.std() > 12 and 25 < img.mean() < 230
    top, bottom = img[:90].astype(float), img[-90:].astype(float)
    assert abs(top.mean() - bottom.mean()) > 5          # sky and ground differ


@pytest.mark.parametrize("preset", ["Low", "Medium", "High", "Ultra"])
def test_every_preset_renders(game, preset):
    game.state.settings.graphics_preset = preset
    img = frame(game, 2)
    assert game.renderer.preset_name == preset
    assert img.std() > 12


def test_resize_rebuilds_targets(game):
    """A window resize re-creates the HDR / bloom / LDR targets at the new size (read back from
    the renderer's own LDR target: the test window itself keeps its size)."""
    eng = game.renderer
    eng.handle_resize(800, 450)
    assert (eng.width, eng.height) == (800, 450)
    assert eng.ldr_tex.size == (800, 450) and eng.scene_tex.size == (eng.rw, eng.rh)
    game.tick(1 / 30, render=True)
    img = np.frombuffer(eng.ldr_fbo.read(components=3, alignment=1), np.uint8).reshape(450, 800, 3)
    assert img.std() > 12
    eng.handle_resize(640, 360)
    assert frame(game).shape == (360, 640, 3)


def test_lit_fire_brightens_the_night(game):
    from firestarter.automation.staging import build_log_fire
    st = game.state
    st.settings.graphics_preset = "Low"
    st.world.weather.set_time(23.0)
    dark = frame(game, 40).astype(float)
    build_log_fire(st.world)
    for _ in range(300):
        st.step(1 / 30, move=False)
    lit = frame(game, 20).astype(float)
    assert lit.mean() > dark.mean() + 3
    # the brightest region is warm (fire), never blue
    lum = lit.mean(axis=2)
    hot = lit[lum >= np.percentile(lum, 99.5)]
    assert hot[:, 0].mean() >= hot[:, 2].mean()
    # flames animate: consecutive frames differ around the fire
    a, b = frame(game).astype(float), frame(game, 3).astype(float)
    assert np.abs(a - b).mean() > 0.05


def test_menus_draw_over_the_scene(game):
    st = game.state
    base = frame(game).astype(float)
    st.input.handle_key_down("tab")
    phone = frame(game).astype(float)
    st.input.handle_key_down("tab")
    st.input.handle_key_down("b")
    book = frame(game).astype(float)
    st.input.handle_key_down("b")
    assert np.abs(phone - base).mean() > 5 and np.abs(book - base).mean() > 10


def test_frame_time_statistics_are_recorded(game):
    frame(game, 5)
    assert len(game.perf) >= 5 and all(ms > 0 for ms in game.perf[-5:])
