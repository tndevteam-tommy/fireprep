"""How the people look: their own hair, eyes, chrome and clothes; heads shaped like
heads; faces that move with what they are doing (``animation.look``, ``anatomy``,
``assets.faces``, the street clips)."""

import math

import pytest

from cyberduke.animation import anatomy
from cyberduke.animation import street as clips
from cyberduke.animation.look import HAIRSTYLES, street_look
from cyberduke.levels import build_night_market


@pytest.fixture(scope="module")
def street():
    from cyberduke.simulation import Simulation

    return Simulation(build_night_market())


def test_no_two_people_of_the_street_look_alike(street):
    people = street.civilians + street.merchants
    looks = [
        (p.spec.look, p.spec.hair_color, p.spec.shirt_color, p.spec.skin_color) for p in people
    ]
    assert len(set(looks)) == len(people), "everybody their own look"
    assert len({p.spec.look.hair for p in people}) >= 4, "all sorts of hair"
    assert len({p.spec.look.eyes for p in people}) >= 2, "all sorts of eyes"
    chromed = [p for p in people if p.spec.look.implants]
    assert len(chromed) == len(people), "a century of cybernetics: everybody has some chrome"


def test_street_looks_are_loud_and_seeded():
    import random

    a = street_look(random.Random(5))
    assert a == street_look(random.Random(5)), "the same seed, the same look"
    looks = [street_look(random.Random(k))[0] for k in range(60)]
    assert {look.hair for look in looks} == set(HAIRSTYLES[1:]), "every style turns up"
    assert all(look.hair != "cap" for look in looks), "hardly anybody has plain hair"
    assert sum(look.eyes != "organic" for look in looks) > 40, "most eyes are replaced"


def test_the_head_has_a_forehead_and_a_flatter_back():
    """The head is taller than the first ones (all of it forehead: the features are
    where they were) and the back of the skull is flatter than the face is round."""
    assert anatomy.HEAD_SIZE[1] > 0.4
    assert anatomy.HAIRLINE_FRONT - anatomy.BROW_LINE > 0.04, "a forehead above the brows"
    assert anatomy.MOUTH_LINE < anatomy.NOSE_Y < anatomy.EYE_LINE < anatomy.BROW_LINE
    assert anatomy.P_BACK > anatomy.P_FRONT
    t = 0.6
    # at 45 degrees behind the ear the skull is fuller (flatter back) than at 45 in front
    front = anatomy.point(t, math.radians(45.0))
    back = anatomy.point(t, math.radians(-45.0))
    assert math.hypot(back[0], back[2]) > math.hypot(front[0], front[2]) * 0.98


def test_talking_moves_the_mouth_syllable_by_syllable():
    mouths = [clips.talk(t * 0.05, 1.0).expression for t in range(120)]
    assert {"talk", "talk_o"} <= set(mouths), "open on the vowels"
    changes = sum(a != b for a, b in zip(mouths, mouths[1:], strict=False))
    assert changes > 12, "the mouth moves as they talk"
    assert clips.talk(0.5, 1.0, listening=True).expression is None, "listening: mouth shut"


def test_eating_opens_wide_for_each_bite_and_chews():
    faces = [clips.eat(t * 0.05).expression for t in range(int(clips.BITE / 0.05))]
    assert "eat" in faces and "mumble" in faces


def test_a_drag_on_a_cigarette_purses_the_lips_and_the_smoke_comes_out_round():
    faces = {clips.smoke(t * 0.05).expression for t in range(int(clips.SMOKE_CYCLE / 0.05))}
    assert {"pursed", "talk_o"} <= faces


def test_reading_the_menu_under_their_breath():
    faces = {clips.point_at(t * 0.1, (0.0, 0.2, 1.0)).expression for t in range(40)}
    assert "mumble" in faces


def test_an_owner_who_finds_the_netrunner_at_home_shouts():
    from cyberduke.entities.player import Player
    from cyberduke.simulation import Simulation

    sim = Simulation(build_night_market())
    owner = sim.npc_with_role("office_worker")
    player = Player((owner.x, owner.z + 1.5, 0.0))
    faces = set()
    for _ in range(30):
        owner._face(player, 1 / 30, angry=True)
        faces.add(owner.pose.expression)
    assert faces <= {"shout", "grimace"} and "shout" in faces


# ----------------------------------------------------------------------------- hair
def test_every_hairstyle_is_textured_with_strands_that_run_the_way_it_grows():
    """All the hair - caps, crests, spikes, bobs, tails, knots - is drawn in the hair
    texture, whose strands run along ``v`` (down the texture): the pieces' ``v`` runs
    the way the hair grows (along a spike, down a curtain of hair, along a tail)."""
    import numpy as np

    from cyberduke.ai.civilian import civilian_spec
    from cyberduke.animation import hairdo
    from cyberduke.assets.textures import generate_texture
    from cyberduke.render.humanoid import materials

    img = generate_texture("hair_strands").astype(float).mean(axis=2)
    across = np.abs(np.diff(img, axis=1)).mean()  # (strand to strand)
    along = np.abs(np.diff(img, axis=0)).mean()  # (down a strand)
    assert across > 3.0 * along, "strands, not noise"
    textures = {"hair_strands": 5, "stubble": 6, "flesh": 7}
    mats = materials(civilian_spec(3), textures)
    assert mats["hair"][1] == 5 and mats["hair2"][1] == 5, "both colours of hair have strands"
    assert mats["stubble"][1] == 6
    for style in hairdo.STYLES:
        (cap, cap_mat), pieces = hairdo.pieces(style)
        assert cap_mat in ("hair", "stubble", "scalp")
        for name, mat in pieces:
            assert mat in ("hair", "hair2", "neon"), (style, name)
            pos, uv, _ = hairdo.raw(name)
            if mat == "neon":
                continue
            assert np.ptp(uv[:, 1]) > 0.25, f"{name}: the strands run along it"


def test_no_rolls_or_slabs_of_hair_only_real_haircuts():
    """The old hair was a cap with rolls and slabs stuck on it (a 'swoop', a 'bob' of two
    ear muffs, a long slab down the back): every style is grown on the skull now."""
    from cyberduke.render.shapes import BUILDERS

    for old in ("lock", "tail", "hair_half", "spike"):
        assert old not in BUILDERS, old


def test_long_hair_and_bobs_frame_the_face_and_clear_the_shoulders():
    import numpy as np

    from cyberduke.animation import anatomy, hairdo

    brow = anatomy.BROW_LINE
    for name in ("bob", "long", "undercut_l", "undercut_r"):
        pos, _, _ = hairdo.raw(name)
        in_front = pos[(pos[:, 2] > 0.12) & (np.abs(pos[:, 0]) < 0.09)]
        assert (in_front[:, 1] > brow).all(), f"{name}: nothing over the eyes"
        assert pos[:, 1].min() > anatomy.CHIN_Y - anatomy.HEAD_Y - 0.07, (
            f"{name}: clear of the shoulders"
        )
    # the bob to the jaw, the long hair further; a fringe over the forehead
    assert hairdo.raw("long")[0][:, 1].min() < hairdo.raw("bob")[0][:, 1].min() - 0.1


def test_hairstyles_differ_by_the_seed_where_they_have_a_side():
    from cyberduke.animation import hairdo

    assert hairdo.pieces("undercut", 0) != hairdo.pieces("undercut", 1)
    assert hairdo.pieces("spikes", 0) != hairdo.pieces("spikes", 2)


def test_every_spike_of_a_mohawk_comes_to_a_point():
    """The crest's spikes are all alike: round, tapering to a sharp point - none stubby or
    flat (the short one over the forehead came out a blunt stub as fat as the tall ones)."""
    import numpy as np

    from cyberduke.animation import hairdo

    pos, _, _ = hairdo.raw("crest_tips")
    base, _, _ = hairdo.raw("crest")
    per = len(pos) // len(hairdo.CREST)
    per_base = len(base) // len(hairdo.CREST)
    for k, (_, length, _) in enumerate(hairdo.CREST):
        tip = pos[k * per : (k + 1) * per]
        root = base[k * per_base : (k + 1) * per_base]
        ring = per // 8  # (8 rings of 11)
        last = tip[-ring:]
        assert np.ptp(last, axis=0).max() < 0.006, f"spike {k} ends in a point"
        first = root[:ring]
        width = float(np.ptp(first, axis=0).max())
        assert length / width > 2.4, f"spike {k} is slender, not a stub ({length / width:.2f})"
    lengths = [n for _, n, _ in hairdo.CREST]
    assert min(lengths) >= 0.7 * max(lengths), "none a stub beside the others"


def test_a_mohawks_spikes_run_the_length_of_the_crest_with_no_gap():
    """Seen from the side, the crest's spikes stand evenly from the forehead to the nape -
    not four bunched at the front and three at the back with a bare strip over the crown
    between them (they were spaced by the skull's height, not by the length over it)."""
    import numpy as np

    from cyberduke.animation import hairdo

    feet = [np.array(hairdo.midline_along(u)) for u, _, _ in hairdo.CREST]
    gaps = [float(np.linalg.norm(b - a)) for a, b in zip(feet, feet[1:], strict=False)]
    assert len(hairdo.CREST) >= 9
    assert max(gaps) < 0.075 and max(gaps) < 1.25 * min(gaps), [round(g, 3) for g in gaps]
    # over the crown (the top of the skull, where there used to be none) there are some
    crown = [f for f in feet if f[0] > 0.18]
    assert len(crown) >= 3
    # every one stands out of the skull (none turned in)
    pos, _, _ = hairdo.raw("crest_tips")
    per = len(pos) // len(hairdo.CREST)
    for k, foot in enumerate(feet):
        tip = pos[(k + 1) * per - 1]
        out = np.array((foot[0] + 0.03, foot[1]))  # (from a little below the middle)
        assert float(np.array((tip[1] - foot[0], tip[2] - foot[1])) @ out) > 0.0, k
