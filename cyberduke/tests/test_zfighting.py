"""Z-fighting: two surfaces in one plane, facing the same way, over the same patch - the
depth buffer cannot tell which is in front, and the picture flickers between them (the
flat's ceiling showing through the outside wall of number 16, the alley's asphalt
through its lino, the bathroom mirror "glitching" into the tiles it was set into).

``render.audit`` finds them by drawing a level into a recorder and comparing every face
with every other; these tests show it finds the kinds of thing that went wrong (and
nothing that did not), and hold every shipped level to zero.
"""

from __future__ import annotations

import pytest

from cyberduke.render.audit import GeometryRecorder, audit_level, find_zfights
from cyberduke.render.props import draw_prop
from cyberduke.world.props import Material, Prop

STONE = Material("concrete", (0.5, 0.5, 0.5))
GLOW = Material(None, (1.0, 0.5, 0.2), alpha=0.3)


def fights(*props: Prop):
    ctx = GeometryRecorder()
    for p in props:
        ctx.label = p.label
        draw_prop(ctx, p, {})
        ctx.alpha(1.0)
    return find_zfights(ctx.faces, ctx.solids)


def box(x0, x1, y0, y1, z0, z1, label, **kw) -> Prop:
    return Prop.box(
        (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0, x1 - x0, y1 - y0, z1 - z0, STONE,
        label=label, **kw,
    )  # fmt: skip


# ----------------------------------------------------------------------------- the search
def test_two_slabs_level_with_each_other_fight_where_they_overlap():
    hits = fights(box(0, 4, -0.1, 0.0, 0, 4, "street"), box(2, 6, -0.02, 0.0, 1, 3, "floor"))
    assert len(hits) == 1
    hit = hits[0]
    assert {hit.a, hit.b} == {"street", "floor"}
    assert hit.normal[1] == pytest.approx(1.0) and hit.area == pytest.approx(2.0 * 2.0)
    assert hit.point[0] == pytest.approx(3.0) and hit.point[1] == pytest.approx(0.0)


def test_a_few_millimetres_apart_or_side_by_side_they_do_not():
    assert not fights(box(0, 4, -0.1, 0.0, 0, 4, "street"), box(2, 6, -0.02, 0.004, 1, 3, "floor"))
    assert not fights(box(0, 2, -0.1, 0.0, 0, 4, "street"), box(2, 6, -0.1, 0.0, 0, 4, "floor"))


def test_faces_that_meet_front_to_front_are_not_a_fight():
    """A box standing on another: the top of one and the underside of the other share a
    plane, but face each other - neither is ever seen."""
    assert not fights(box(0, 2, 0, 1, 0, 2, "crate"), box(0.5, 1.5, 1, 2, 0.5, 1.5, "crate on it"))


def test_a_fight_buried_inside_something_solid_is_not_reported():
    a = box(0, 1, 0, 0.5, 0, 1, "a")
    b = box(0.5, 1.5, 0, 0.5, 0.2, 0.8, "b")  # their tops (y 0.5) overlap at x 0.5 .. 1
    assert fights(a, b)
    lid = box(-0.5, 2.0, 0.5, 1.0, -0.5, 1.5, "lid")  # ... covered by a lid lying on both
    assert not fights(a, b, lid)


def test_turned_boxes_are_found_too():
    a = Prop.box(0.0, 0.5, 0.0, 2.0, 1.0, 0.3, STONE, rotate=((30.0, 0.0, 1.0, 0.0),), label="wall")
    # (along the wall's own length: their fronts and backs in the wall's planes)
    b = Prop.box(
        0.52, 0.5, -0.3, 1.0, 0.5, 0.3, STONE, rotate=((30.0, 0.0, 1.0, 0.0),), label="panel"
    )
    hits = fights(a, b)
    assert {frozenset((h.a, h.b)) for h in hits} == {frozenset(("wall", "panel"))}
    assert len(hits) == 2, "its front and its back"


def test_a_see_through_pool_of_light_flush_with_the_floor_fights_it():
    floor = box(-3, 3, -0.1, 0.0, -3, 3, "floor")
    assert fights(floor, Prop.disc(0.0, 0.0, 0.0, 1.0, 1.0, 0.3, GLOW, label="pool"))
    assert not fights(floor, Prop.disc(0.0, 0.004, 0.0, 1.0, 1.0, 0.3, GLOW, label="pool"))
    # (two see-through layers never fight: neither writes depth)
    assert not fights(
        Prop.disc(0.0, 0.0, 0.0, 1.0, 1.0, 0.3, GLOW, label="a"),
        Prop.disc(0.2, 0.0, 0.0, 1.0, 1.0, 0.3, GLOW, label="b"),
    )


def test_a_cylinder_flush_with_a_shelf_fights_it():
    shelf = box(-1, 1, 0.9, 1.0, -1, 1, "shelf")
    jar = Prop.cylinder(0.0, 0.9, 0.0, 0.1, 0.1, 0.1, STONE, label="jar")  # top at 1.0
    assert any({h.a, h.b} == {"shelf", "jar"} for h in fights(shelf, jar))


# ----------------------------------------------------------------------------- what went wrong
def test_the_three_glitches_reported_in_number_16_are_found_and_their_fixes_are_clean():
    """As they were built: the flat's ceiling ran out to the outer face of its front wall;
    the alley's asphalt ran under the flat at floor level; the bathroom mirror was set in
    the tiles with its face flush with theirs. As they are built now, none fights."""
    wall = box(-4.79, -4.5, 0.0, 3.0, 27, 34, "front wall (brick)")
    ceiling_then = box(-11.0, -4.5, 2.7, 3.0, 26.6, 35.8, "ceiling")
    ceiling_now = box(-11.0, -4.7, 2.7, 2.99, 26.6, 35.8, "ceiling")
    assert any("ceiling" in (h.a, h.b) for h in fights(wall, ceiling_then))
    assert not fights(wall, ceiling_now)

    floor = box(-8.2, -4.8, -0.02, 0.0, 26.8, 32.6, "lino")
    asphalt_then = box(-5.5, 5.5, -0.1, 0.0, -2.0, 44.0, "asphalt")
    asphalt_now = box(-4.8, 4.8, -0.1, 0.0, -2.5, 42.5, "asphalt")
    assert fights(floor, asphalt_then)
    assert not fights(floor, asphalt_now)

    tiles = box(-10.8, -8.32, 0.0, 1.5, 26.805, 26.82, "tiles")
    mirror_then = box(-9.5, -9.1, 1.3, 1.8, 26.81, 26.82, "mirror")
    mirror_now = box(-9.5, -9.1, 1.3, 1.8, 26.815, 26.825, "mirror")
    assert fights(tiles, mirror_then)
    assert not fights(tiles, mirror_now)


# ----------------------------------------------------------------------------- the levels
def test_a_generated_flat_has_no_two_surfaces_in_one_plane():
    from cyberduke.levels import flats
    from cyberduke.levels.flats import FlatPlan

    ctx = GeometryRecorder()
    plans = (
        FlatPlan("99", "x", "+z", 10.0, 0.0, 5.2, 3.9),
        FlatPlan("98", "x", "+x", -20.0, 0.0, 6.0, 3.0),
    )
    for plan in plans:
        props, _ = flats.build(plan)
        props += list(flats.door(plan).leaf)
        for p in props:
            ctx.label = p.label
            draw_prop(ctx, p, {})
            ctx.alpha(1.0)
    hits = find_zfights(ctx.faces, ctx.solids)
    assert not hits, "\n".join(str(h) for h in hits)


@pytest.mark.parametrize("key", ["facility", "night_market"])
def test_no_shipped_level_has_z_fighting(key):
    """Every prop, light fixture, door, computer, camera, vending machine, access panel
    and thing lying about, drawn as the game draws it: no two surfaces fight."""
    from cyberduke.levels import LEVELS

    hits = audit_level(LEVELS[key]())
    assert not hits, f"{len(hits)} z-fighting patch(es):\n" + "\n".join(str(h) for h in hits[:40])
