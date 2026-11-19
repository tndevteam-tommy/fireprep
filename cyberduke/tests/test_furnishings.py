"""The homes are furnished like homes (``levels.furnish``, ``levels.flats``,
``levels.apartment``): cupboards with doors, a sink let into the worktop with a tap, a
toilet with a bowl, a seat and the water in it, a window with a view (and the same window
lit from the street), shelves, pictures, food - and no two of them alike: each done up in
its own style, with the things of whoever lives there."""

from __future__ import annotations

from collections import defaultdict

import pytest

from cyberduke.levels import build_night_market
from cyberduke.levels.flats import STYLES, style_of
from cyberduke.levels.main_street import ALL_FLATS
from cyberduke.render.television import is_screen, screen_of

LEVEL = build_night_market()
AREAS = {a.area_id: a for a in LEVEL.private_areas}
PICTURES = {
    "poster_band", "poster_film", "poster_idol", "poster_fish", "poster_sports", "poster_corp",
    "city_map", "art_synthwave", "art_abstract", "photo_family", "photo_city", "certificate",
    "calendar", "pegboard",
}  # fmt: skip
FOOD = ("Cereal", "bread", "Kettle", "oil", "pickles", "noodle", "Fruit", "Takeaway", "Tin",
        "rice", "soda", "Biscuit", "crisps", "hot sauce")  # fmt: skip


def _homes() -> dict[str, list]:
    """Every home's props (inside it, or on its front wall outside)."""
    out: dict[str, list] = defaultdict(list)
    for p in LEVEL.props:
        for area_id, area in AREAS.items():
            if area.contains(p.x, p.z):
                out[area_id].append(p)
    return out


HOMES = _homes()


def _labels(area_id: str) -> list[str]:
    return [p.label for p in HOMES[area_id]]


def _has(area_id: str, *words: str) -> bool:
    return any(any(w in label for w in words) for label in _labels(area_id))


@pytest.mark.parametrize("area_id", sorted(AREAS))
def test_every_home_has_a_real_kitchen_and_bathroom(area_id):
    assert _has(area_id, "cupboard door"), "cupboards with doors"
    assert _has(area_id, "cupboard door handle"), "...and handles on them"
    assert _has(area_id, "Sink basin (bottom)") and _has(area_id, "Sink drain"), (
        "a sink you can see into"
    )
    assert _has(area_id, "Kitchen tap"), "a tap over it"
    for part in ("Toilet bowl", "Toilet seat", "Toilet water", "Toilet cistern", "Toilet flush"):
        assert _has(area_id, part), f"a toilet: {part}"
    assert _has(area_id, "Wash basin pedestal") and _has(area_id, "Wash basin tap")
    assert _has(area_id, "Toilet roll") and _has(area_id, "Towel")


@pytest.mark.parametrize("area_id", sorted(AREAS))
def test_every_home_has_a_window_shelves_pictures_and_food(area_id):
    props = HOMES[area_id]
    assert any(is_screen(p) and screen_of(p)[0] == "view" for p in props), "a window with a view"
    assert _has(area_id, "Window frame") and _has(area_id, "Window sill")
    assert _has(area_id, "shelf", "Shelf", "Bookcase"), "shelves"
    assert sum(1 for p in props if p.material.texture in PICTURES) >= 2, "pictures on the walls"
    food = [label for label in _labels(area_id) if any(f in label for f in FOOD)]
    assert len(food) >= 3, f"food about: {food}"


def test_every_window_is_the_same_window_from_outside_lit():
    """The view inside is matched by a lit window in the building's face outside."""
    for area_id, props in HOMES.items():
        inside = next(p for p in props if is_screen(p) and screen_of(p)[0] == "view")
        outside = [p for p in LEVEL.props if p.label.startswith("Window of number")]
        number = AREAS[area_id].name.split()[-1]
        lit = next(p for p in outside if p.label == f"Window of number {number}")
        assert lit.material.emission is not None, "lit"
        a, b = inside.footprint(), lit.footprint()
        along = (a.min_x, a.max_x) if a.max_x - a.min_x > 0.3 else (a.min_z, a.max_z)
        out = (b.min_x, b.max_x) if b.max_x - b.min_x > 0.3 else (b.min_z, b.max_z)
        mid_in, mid_out = sum(along) / 2.0, sum(out) / 2.0
        assert abs(mid_in - mid_out) < 0.1, f"{area_id}: the window outside is where it is inside"


def test_no_two_flats_are_done_up_alike():
    assert len({style_of(plan) for plan in ALL_FLATS}) == len(ALL_FLATS), "a style each"
    looks = {(s.walls[0], s.floor[0], s.sofa, s.window, s.tv) for s in STYLES}
    assert len(looks) == len(STYLES)
    walls = {style_of(plan).walls for plan in ALL_FLATS}
    floors = {style_of(plan).floor for plan in ALL_FLATS}
    assert len(walls) == len(ALL_FLATS) and len(floors) == len(ALL_FLATS)


def test_everybody_lives_among_their_own_things():
    """What is in each flat that is in no other: the things of whoever lives there."""
    own = {
        "smoker": ("Dartboard", "Carton of cigarettes", "Record"),
        "junk_dealer": ("Dead cyberdeck", "Coil of cable", "Old monitor"),
        "tinkerer": ("Tool board", "Soldering station", "Little robot"),
        "stroller": ("Fairy light", "Big plant", "Bookcase"),
        "office_worker": ("Binder", "Photo frame standing on the desk", "Certificate"),
        "courier": ("Map of the district", "Parcel", "Cycling helmet"),
        "sushi_chef": ("Lucky cat", "Knife strip", "Chart of the fish of the bay"),
    }
    by_owner = {a.owner_role: a.area_id for a in AREAS.values()}
    for role, things in own.items():
        mine = by_owner[role]
        for thing in things:
            assert _has(mine, thing), f"{role} has {thing}"
            others = [a for a in AREAS if a != mine and _has(a, thing)]
            assert not others or thing in ("Parcel", "Bookcase"), f"{thing} is {role}'s alone"
    assert {p.material.texture for p in HOMES["flat16"]} >= {"poster_idol", "poster_corp",
                                                             "mirror_cracked"}  # fmt: skip


def test_dells_kitchen_has_a_real_fridge_and_microwave():
    """Dell's fridge was a bare box and the microwave a slab: the fridge has a freezer
    door over the fridge door, chrome bar handles, a plinth, his sticker and rust at its
    foot; the microwave a door with a window, a handle, a control panel and a clock."""
    from cyberduke.levels.apartment import FLAT16

    labels = [p.label for p in LEVEL.props if FLAT16.contains(p.x, p.z)]
    for part in ("Dented fridge door", "Dented fridge freezer door", "Dented fridge handle",
                 "Dented fridge plinth", "Rust on the dented fridge", "Fridge magnet"):  # fmt: skip
        assert part in labels, part
    for part in ("Microwave door", "Microwave window", "Microwave door handle",
                 "Microwave control panel"):  # fmt: skip
        assert part in labels, part
    assert any(label.startswith("Microwave clock") for label in labels)
