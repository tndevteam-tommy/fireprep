import numpy as np
import pytest

from cyberduke.assets.textures import (
    SIZE,
    available_textures,
    generate_texture,
    tile_size,
    tiled_textures,
)


@pytest.mark.parametrize("name", available_textures())
def test_generator_returns_square_rgb_uint8(name):
    img = generate_texture(name)
    assert img.shape == (SIZE, SIZE, 3)
    assert img.dtype == np.uint8


def test_generators_are_deterministic():
    for name in available_textures():
        assert np.array_equal(generate_texture(name), generate_texture(name)), name


def test_unknown_style_raises_with_hint():
    with pytest.raises(KeyError, match="known:"):
        generate_texture("plasma")


def test_required_styles_exist():
    required = {
        "wall", "floor", "metal", "hazard", "grate", "terminal_locked", "terminal_unlocked",
        "screen_net", "screen_radar", "server_rack", "crate", "cardboard", "soda_can",
        "neon_sign", "skin", "fabric", "pants", "leather", "armor", "neon_cyan", "neon_magenta",
    }  # fmt: skip
    assert required <= set(available_textures())


@pytest.mark.parametrize("name", tiled_textures())
def test_tiled_textures_are_seamless(name):
    """A tiled surface texture repeats across walls and floors: the step from its last
    column to its first (and last row to first) must look like any other step inside the
    image - no visible seam every tile."""
    img = generate_texture(name).astype(float)
    for axis in (0, 1):
        rows = np.moveaxis(img, axis, 0)
        inside = np.abs(np.diff(rows, axis=0)).mean(axis=(1, 2)).max()
        wrap = np.abs(rows[0] - rows[-1]).mean()
        assert wrap <= 1.2 * inside + 1.0, f"{name}: seam {wrap:.1f} vs {inside:.1f} inside"


def test_surface_materials_tile_at_a_real_world_size():
    for name in ("wall", "floor", "metal", "brick", "concrete", "asphalt", "wood", "grate"):
        size = tile_size(name)
        assert size is not None and 0.2 <= size <= 3.0, name
    assert tile_size("skin") is None, "the face is painted once across the face plate"
    assert tile_size("neon_sign") is None


def test_the_painted_eyes_and_mouth_line_up_with_the_modelled_face():
    """The face plate is the height of the head (``animation.anatomy``): the pupils must
    sit on the eye line (where the visor is) and the mouth on the mouth line, not a few
    centimetres below them."""
    from cyberduke.animation import anatomy
    from cyberduke.assets.faces import FH
    from cyberduke.render.humanoid import EYE_LINE

    img = generate_texture("skin").astype(float)
    row_height = FH / SIZE

    def height(rows):  # image row (0 = top) -> metres above the middle of the head
        return (SIZE / 2 - (rows.mean() + 0.5)) * row_height

    pupils = np.argwhere(img.sum(axis=2) < 60)[:, 0]
    assert height(pupils) == pytest.approx(EYE_LINE, abs=0.01)
    mouth = np.argwhere(np.all(np.abs(img - (120, 56, 56)) < 2, axis=2))[:, 0]
    assert height(mouth) == pytest.approx(anatomy.MOUTH_LINE, abs=0.006)


def _mouth_window():
    from cyberduke.animation import anatomy
    from cyberduke.assets.faces import col, row

    r = int(round(row(anatomy.MOUTH_LINE)))
    c0, c1 = int(col(-0.05)), int(col(0.05)) + 1
    return slice(r - 12, r + 12), slice(c0, c1)


def test_three_faces_with_their_mouths_in_the_same_place():
    """Neutral, smiling and grimacing faces: the same eyes, and each mouth (outlined in
    the lip colour) where the mouth is - but a different mouth."""
    from cyberduke.animation import anatomy
    from cyberduke.assets.faces import FH
    from cyberduke.assets.textures import FACES, MOUTH_RGB

    imgs = {name: generate_texture(name).astype(float) for name in FACES}
    row_height = FH / SIZE
    for name, img in imgs.items():
        rows = np.argwhere(np.all(np.abs(img - MOUTH_RGB) < 2, axis=2))[:, 0]
        assert len(rows) > 12, name
        middle = (SIZE / 2 - (rows.mean() + 0.5)) * row_height
        assert middle == pytest.approx(anatomy.MOUTH_LINE, abs=0.012), name
    mouth = _mouth_window()
    neutral, smile, grimace = (imgs[n][mouth] for n in FACES)
    assert np.abs(smile - neutral).mean() > 4 and np.abs(grimace - neutral).mean() > 4
    assert np.abs(smile - grimace).mean() > 4
    teeth = np.all(np.abs(imgs["skin_grimace"][mouth] - (238, 234, 224)) < 3, axis=2).sum()
    assert teeth > 40, "gritted teeth"
    assert not np.all(np.abs(neutral - (238, 234, 224)) < 3, axis=2).any(), "a closed mouth"


def test_every_expression_moves_the_mouth_and_keeps_the_eyes():
    """Talking, eating, smoking, shouting: each expression paints its own mouth in the same
    place, the eyes staying where they are - and the whites of the eyes and the teeth stay
    white whatever the skin tone."""
    from cyberduke.assets.faces import EXPRESSIONS, face_image

    mouth = _mouth_window()
    dark = (0.45, 0.32, 0.25)
    faces = {e: face_image(dark, "", e).astype(float) for e in EXPRESSIONS}
    neutral = faces["neutral"]
    for name, img in faces.items():
        if name != "neutral":
            changed = np.abs(img[mouth] - neutral[mouth]).max(axis=2) > 30
            assert changed.sum() > 15, name
    for a in ("talk", "talk_o", "shout", "eat", "pursed", "mumble"):
        for b in ("talk", "talk_o", "shout", "eat", "pursed", "mumble"):
            if a < b:
                assert np.abs(faces[a][mouth] - faces[b][mouth]).mean() > 0.5, (a, b)
    open_wide = np.all(np.abs(faces["eat"][mouth] - (70, 24, 30)) < 3, axis=2).sum()
    talking = np.all(np.abs(faces["talk"][mouth] - (70, 24, 30)) < 3, axis=2).sum()
    assert open_wide > talking > 0, "a bite opens the mouth wider than a word"
    teeth = np.all(np.abs(faces["grimace"][mouth] - (238, 234, 224)) < 3, axis=2).sum()
    assert teeth > 40, "the teeth are not the colour of the skin"
    whites = np.all(np.abs(neutral - (236, 232, 226)) < 20, axis=2).sum()
    assert whites > 20, "the whites of the eyes are not the colour of the skin"


def test_faces_are_painted_per_tone_mark_and_expression_on_demand():
    from cyberduke.assets.faces import face_from_name, face_name

    name = face_name((0.5, 0.4, 0.3), "circuit", "talk")
    assert name == "face|0.50,0.40,0.30|circuit|talk"
    img = face_from_name(name)
    assert img.shape == (SIZE, SIZE, 3) and img.dtype == np.uint8
    plain = face_from_name(face_name((0.5, 0.4, 0.3), "", "talk"))
    assert np.abs(img.astype(float) - plain).mean() > 0.5, "the circuit lines are on it"


def test_mip_chain_halves_down_to_one_pixel():
    from cyberduke.render.textures import mip_chain

    img = generate_texture("floor")
    levels = mip_chain(img)
    assert [lv.shape[0] for lv in levels] == [SIZE >> k for k in range(len(levels))]
    assert levels[-1].shape == (1, 1, 3) and levels[-1].dtype == np.uint8
    assert np.allclose(levels[-1][0, 0], img.reshape(-1, 3).mean(axis=0), atol=1.0)


@pytest.mark.parametrize(
    "tone", [(1.0, 1.0, 1.0), (0.85, 0.72, 0.6), (0.62, 0.45, 0.32), (1.0, 0.92, 0.85)]
)
def test_the_face_is_the_colour_of_the_rest_of_the_head_where_they_meet(tone):
    """The face's picture fades back to the plain skin towards its edges - the very flesh
    the head, the ears, the nose and the neck are drawn with, in the same tone (it used to
    darken towards the sides, and the face showed as a different colour with a seam) -
    and the body's skin is drawn the way the face is: the tone in the picture, lit white
    (a tone multiplied into the lighting came out brighter than the face under a lamp)."""
    from cyberduke.ai.npc import HumanoidSpec
    from cyberduke.assets import faces
    from cyberduke.render.humanoid import WHITE, materials

    face = faces.face_image(tone).astype(float)
    flesh = faces.flesh_from_name(faces.flesh_name(tone)).astype(float)
    w = faces._feature_weight(faces.XM, faces.YM)
    edge = (w < 0.02) & (faces.YM > -0.12) & (faces.YM < 0.1)
    half_w = np.array([0.5 * anatomy_width(y) for y in faces.YM[:, 0]])[:, None]
    on_head = np.abs(faces.XM) < half_w * 0.97
    near = edge & on_head
    assert near.sum() > 100, "the face's edge, on the head"
    diff = np.abs(face[near].mean(axis=0) - flesh.mean(axis=(0, 1)))
    assert (diff < 0.03 * flesh.mean(axis=(0, 1)) + 1.0).all(), diff
    # ... pixel for pixel, not just on average
    assert np.abs(face[near] - flesh[near]).max() < 6.0
    # the skin parts drawn with the flesh in the person's own tone, lit white
    names = {}

    class Textures(dict):
        def get(self, name, default=None):
            names.setdefault(name, len(names) + 1)
            return names[name]

    m = materials(HumanoidSpec(skin_color=tone), Textures())
    colour, tex = m["skin"][0], m["skin"][1]
    assert colour == WHITE and tex == names[faces.flesh_name(tone)]
    assert m["face"][0] == WHITE


def anatomy_width(y: float) -> float:
    from cyberduke.animation import anatomy

    t = min(1.0, max(0.0, y / anatomy.HEAD_SIZE[1] + 0.5))
    return anatomy.width(t) * anatomy.rounding(t) * anatomy.HEAD_SIZE[0]
