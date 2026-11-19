"""
Visual quality gates on the screenshots captured by the playback scripts.

Every screenshot must be a properly exposed, detailed, colourful frame (no black / blown /
flat images).  On top of that each named shot has *scene-specific* expectations - a night
campfire is dark with a warm pool of light, a winter day is bright and cool, autumn shows
orange-red canopies, the reference book is warm paper, and so on.

Run the playback suite first (``run_tests.sh`` does), or point FIRESTARTER_ARTIFACTS at a
folder that holds ``playback/*.png``.
"""

import glob
import json
import os

import numpy as np
import pytest
from PIL import Image

from tests.visual import image_metrics as M

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.environ.get("FIRESTARTER_ARTIFACTS", os.path.join(ROOT, "test_artifacts"))
SHOT_DIR = os.path.join(ART, "playback")
SHOTS = sorted(glob.glob(os.path.join(SHOT_DIR, "*.png")))

pytestmark = pytest.mark.skipif(not SHOTS, reason="no playback screenshots - run the playback suite first")


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def shot(name):
    cands = [s for s in SHOTS if os.path.basename(s).endswith(f"__{name}.png")]
    if not cands:
        pytest.skip(f"screenshot {name} not captured")
    return load(cands[0])


def scene(img):
    """The 3D view without the HUD strips (top bar, hotbar) for scene statistics."""
    return M.region(img, 0.13, 0.84, 0.0, 1.0)


_metrics = {}


@pytest.fixture(scope="module", autouse=True)
def write_metrics():
    yield
    if _metrics:
        os.makedirs(ART, exist_ok=True)
        with open(os.path.join(ART, "visual_metrics.json"), "w") as f:
            json.dump(_metrics, f, indent=1, sort_keys=True)


# --------------------------------------------------------------------------- every frame
@pytest.mark.parametrize("path", SHOTS, ids=lambda p: os.path.splitext(os.path.basename(p))[0])
def test_frame_is_well_formed(path):
    img = load(path)
    name = os.path.basename(path)
    s = M.summary(img)
    _metrics[name] = s
    H, W = img.shape[:2]
    assert (W, H) == (1280, 720)
    assert s["std_luma"] > 0.06, s                # not a flat / blank frame
    assert s["clipped_fraction"] < 0.06, s        # no blown-out white areas
    assert s["dark_fraction"] < 0.97, s           # never a black screen
    assert s["sharpness"] > 40.0, s               # real detail, not a blur
    assert s["colorfulness"] > 8.0, s             # not grey mush
    # no NaN / uninitialised-buffer garbage: no pure magenta or pure green pixels
    f = img.astype(int)
    garbage = ((f[..., 0] > 245) & (f[..., 1] < 10) & (f[..., 2] > 245)) | ((f[..., 0] < 10) & (f[..., 1] > 245) & (f[..., 2] < 10))
    assert garbage.mean() < 0.001


# --------------------------------------------------------------------------- daylight & seasons
def test_noon_is_bright_with_a_blue_sky():
    img = shot("02_noon_clearing")
    assert 0.28 < M.luminance(scene(img)).mean() < 0.7
    # blue sky visible between the crowns
    assert M.hue_fraction(M.region(img, 0.0, 0.35), 190, 235, min_sat=0.2, min_value=0.45) > 0.03
    assert M.hue_fraction(scene(img), 70, 160) > 0.25    # green vegetation


def test_golden_hour_light_is_warm():
    img = shot("01_golden_hour")
    r, g, b = M.mean_rgb(scene(img))
    assert r > b + 0.04
    assert M.warm_fraction(img) > 0.002          # the fire is visible


def test_morning_mist_softens_the_distance():
    mist = shot("03_morning_mist")
    noon = shot("02_noon_clearing")
    far_mist = M.region(mist, 0.25, 0.45)
    far_noon = M.region(noon, 0.25, 0.45)
    assert M.luminance(far_mist).std() < M.luminance(far_noon).std()
    assert M.colorfulness(far_mist) < M.colorfulness(far_noon)


def test_dusk_sky_has_sunset_colours():
    img = shot("04_dusk_sunset_sky")
    sky = M.region(img, 0.12, 0.45)
    assert M.hue_fraction(sky, 0, 45, min_sat=0.15, min_value=0.2) + M.hue_fraction(sky, 250, 340, min_sat=0.1, min_value=0.15) > 0.02


def test_autumn_canopy_turns_orange_and_red():
    img = shot("08_autumn_colours")
    assert M.hue_fraction(M.region(img, 0.05, 0.6), 10, 48, min_sat=0.4, min_value=0.3) > 0.04


def test_winter_day_is_bright_snowy_and_cool():
    img = shot("09_winter_snow_day")
    sc = scene(img)
    assert M.luminance(sc).mean() > 0.5
    snow = (M.to_float(sc).min(axis=2) > 0.6).mean()
    assert snow > 0.25                             # lots of near-white snow
    r, g, b = M.mean_rgb(M.region(img, 0.55, 0.84))
    assert b >= r - 0.07                          # white / blue-white snow, not yellow


def test_spring_is_fresh_green():
    img = shot("11_spring_morning")
    assert M.hue_fraction(scene(img), 65, 150, min_sat=0.3) > 0.25


def test_creek_water_is_visible():
    img = shot("12_creek")
    lower = M.region(img, 0.45, 0.84)
    assert M.hue_fraction(lower, 180, 240, min_sat=0.15, min_value=0.2) > 0.02


# --------------------------------------------------------------------------- night & fire
def test_night_campfire_is_a_warm_pool_of_light_in_the_dark():
    img = shot("05_night_campfire")
    L = M.luminance(scene(img))
    assert L.mean() < 0.22                        # it is night
    assert M.warm_fraction(img) > 0.004           # bright orange-yellow flames
    H, W = L.shape
    centre = L[int(H * 0.3):int(H * 0.8), int(W * 0.35):int(W * 0.65)].mean()
    corners = np.mean([L[:H // 5, :W // 5].mean(), L[:H // 5, -W // 5:].mean()])
    assert centre > 3.0 * corners + 0.02          # light falls off away from the fire
    hot = M.to_float(img)[M.luminance(img) > np.percentile(M.luminance(img), 99.7)]
    assert hot[:, 0].mean() >= hot[:, 2].mean()   # the brightest pixels are warm, never blue


def test_clear_night_sky_shows_stars():
    img = shot("06_night_stars")
    sky = M.region(img, 0.12, 0.55)
    assert M.luminance(sky).mean() < 0.2
    assert M.bright_points(sky, 0.3) > 25


def test_rain_at_night_keeps_the_fire_and_shows_streaks():
    img = shot("07_rain_at_night")
    assert M.warm_fraction(img) > 0.002
    assert M.luminance(scene(img)).mean() < 0.3


def test_winter_night_fire_lights_the_snow():
    img = shot("10_winter_night_fire_snowfall")
    L = M.luminance(scene(img))
    assert M.warm_fraction(img) > 0.003
    H, W = L.shape
    assert L[int(H * 0.5):, int(W * 0.3):int(W * 0.7)].mean() > L[:H // 4].mean()


def test_flames_animate_between_frames():
    a = shot("05b_flicker_a")
    b = shot("05b_flicker_b")
    diff = np.abs(M.to_float(a) - M.to_float(b)).max(axis=2)
    H, W = diff.shape
    fire = diff[int(H * 0.25):int(H * 0.75), int(W * 0.3):int(W * 0.7)]
    edge = diff[:int(H * 0.12), :int(W * 0.25)]
    assert fire.mean() > 0.004                     # flickering flames / sparks / light
    assert fire.mean() > 2.0 * edge.mean()         # ...while the rest of the scene holds still


# --------------------------------------------------------------------------- gameplay moments
@pytest.mark.parametrize("name", ["01_tutorial_walkthrough__05_tinder_flames", "01_tutorial_walkthrough__06_kindling_burning",
                                  "02_bow_drill_primitive_fire__04_friction_fire_flames",
                                  "06_fire_science__04_log_cabin_burning", "06_fire_science__05_gasoline_fireball"])
def test_fire_shots_show_flames(name):
    path = os.path.join(SHOT_DIR, f"{name}.png")
    if not os.path.exists(path):
        pytest.skip("not captured")
    img = load(path)
    assert M.warm_fraction(img, min_value=0.5) > 0.003


def test_bow_drill_smoke_is_visible():
    img = shot("01_bow_drilling_smoke")
    h, s, v = M.hsv(M.region(img, 0.2, 0.85, 0.2, 0.8))
    smoke = (s < 0.18) & (v > 0.45)
    assert smoke.mean() > 0.003


def test_green_log_steams_white():
    img = shot("02_green_log_steaming")
    h, s, v = M.hsv(M.region(img, 0.1, 0.7, 0.25, 0.75))
    assert ((s < 0.15) & (v > 0.55)).mean() > 0.004


def test_choked_fire_smoulders_with_grey_smoke():
    img = shot("01_choked_smouldering")
    h, s, v = M.hsv(M.region(img, 0.1, 0.45, 0.35, 0.65))     # the air above the pit
    assert ((s < 0.3) & (v > 0.2)).mean() > 0.08             # a pall of grey smoke
    assert M.warm_fraction(img) < 0.02             # no roaring flames


# --------------------------------------------------------------------------- UI screens
def _panel_fraction(img, lo=0.55, hi=0.98, warm=True):
    f = M.to_float(img)
    v = f.max(axis=2)
    paper = (v > lo) & (v < hi) & (f[..., 0] >= f[..., 2]) if warm else (v > lo) & (v < hi)
    return paper.mean()


@pytest.mark.parametrize("name", ["02_book_combustion", "03_book_species", "04_book_teepee_pinned", "06_book_seasoning_chart"])
def test_reference_book_is_warm_paper(name):
    img = shot(name)
    assert _panel_fraction(img) > 0.35
    assert M.sharpness(img) > 150                   # crisp text & diagrams


PHONE = (0.08, 0.96, 0.665, 0.955)        # the handset is held up on the right of the view


@pytest.mark.parametrize("name", ["08_phone_home", "09_phone_topo_map", "11_phone_store"])
def test_phone_screens_are_crisp(name):
    img = shot(name)
    phone = M.region(img, *PHONE)
    side = M.region(img, 0.1, 0.95, 0.0, 0.3)
    assert M.sharpness(phone) > 2.0 * M.sharpness(side)     # the phone holds the detail
    assert M.luminance(phone).std() > 0.1
    # dark bezel framing the screen
    assert (M.luminance(M.region(img, 0.3, 0.7, 0.665, 0.675)) < 0.12).mean() > 0.5


def test_topo_map_has_contours_and_markers():
    img = shot("09_phone_topo_map")
    screen = M.region(img, 0.2, 0.85, 0.69, 0.93)
    assert M.sharpness(screen) > 300
    assert M.hue_fraction(screen, 180, 230, min_sat=0.2) > 0.002        # the creek in blue


def test_pause_and_help_overlays_dim_the_scene():
    for name in ("13_pause_settings", "01_help_overlay"):
        img = shot(name)
        assert M.luminance(M.region(img, 0.2, 0.8, 0.3, 0.7)).std() > 0.1
