"""Pause menu model and the building generator."""
import numpy as np
from harness import *  # noqa
from core.menu_model import MenuModel
from world.building import BuildingGenerator, BuildingSpec, SOUTH, EAST, NORTH


# ------------------------------------------------------------------ pause menu
def _menu():
    m = MenuModel()
    m.rects = {0: (540, 300, 200, 50), 1: (600, 380, 80, 50)}
    return m


def test_menu_ignores_input_while_closed():
    m = _menu()
    assert m.handle("confirm") is None and m.handle("down") is None


def test_menu_keyboard_navigation_wraps_and_confirms():
    m = _menu()
    m.open()
    assert m.selected == 0
    assert m.handle("down") is None and m.selected == 1
    assert m.handle("down") is None and m.selected == 0, "wraps around"
    assert m.handle("up") is None and m.selected == 1
    assert m.handle("confirm") == MenuModel.QUIT
    m.handle("up")
    assert m.handle("confirm") == MenuModel.RESUME


def test_menu_escape_resumes_and_mouse_hover_click():
    m = _menu()
    m.open()
    assert m.handle("back") == MenuModel.RESUME
    assert m.handle("hover", (640, 400)) is None and m.selected == 1
    assert m.handle("hover", (10, 10)) is None and m.selected == 1, "hovering nothing keeps the selection"
    assert m.handle("click", (10, 10)) is None
    assert m.handle("click", (640, 320)) == MenuModel.RESUME
    m.close()
    assert m.handle("click", (640, 320)) is None


# -------------------------------------------------------------------- building
def _building():
    spec = BuildingSpec(0.0, 0.0, 10.0, 8.0, floors=3, facade_faces=(SOUTH, EAST), door_face=SOUTH,
                        balconies=((NORTH, 1, 1.2),), rooftop=True, name="b")
    parts = BuildingGenerator.build(spec)
    ls = LedgeSystem()
    floor = Obstacle(-60.0, -0.5, -60.0, 120.0, 0.5, 120.0, add_crevasses=False)
    floor.rails.clear()
    obs = [floor] + parts
    ls.build(obs)
    return spec, parts, ls, obs


def test_building_has_door_windows_and_roof_at_expected_heights():
    spec, parts, ls, obs = _building()
    names = [p.name for p in parts]
    assert any("cornice" in n for n in names) and any("rooftop" in n for n in names) and any("balcony" in n for n in names)
    south = [r for r in ls.rails if abs(r.p1[2] - 8.0) < 1e-3 and r.normal[2] > 0.5]
    heights = sorted(set(round(r.height, 2) for r in south))
    # sills (band tops) and window heads (pier tops) on every floor ...
    assert heights == [1.0, 2.2, 4.0, 5.2, 7.0, 8.2], heights
    # ... and the roof edge on the cornice, 0.3 m proud of the facade
    roof_south = [r for r in ls.rails if abs(r.height - 9.0) < 1e-3 and r.normal[2] > 0.5]
    assert len(roof_south) == 1 and abs(roof_south[0].p1[2] - 8.3) < 1e-3
    heads = [r for r in south if abs(r.height - 2.2) < 1e-3]
    assert len(heads) >= 4, "a pier (window head hold) between every window"
    # the door: the ground band is split, leaving a gap wider than a door where no 1.0 m sill exists
    sills = sorted([r for r in south if abs(r.height - 1.0) < 1e-3], key=lambda r: min(r.p1[0], r.p2[0]))
    assert len(sills) == 2
    gap = min(sills[1].p1[0], sills[1].p2[0]) - max(sills[0].p1[0], sills[0].p2[0])
    assert abs(gap - BuildingGenerator.WINDOW_WIDTH) < 1e-3


def test_building_hidden_rails_are_clipped_and_short_edges_dropped():
    spec, parts, ls, obs = _building()
    core = [p for p in parts if p.name.endswith("core")][0]
    # The core's faces behind the facades carry no rails any more
    for r in ls.rails:
        if r.owner is core:
            assert r.normal[2] < 0.5 and r.normal[0] < 0.5, "core rail survived behind a facade"
    assert all(r.length >= config.MIN_RAIL_LENGTH - 1e-6 for r in ls.rails)
    # every rail's probe point is outside every other box
    for r in ls.rails:
        mid = r.midpoint + r.normal * config.RAIL_COVER_PROBE
        for o in obs:
            if o is r.owner:
                continue
            b = o.bounds
            inside = (b.min[0] < mid[0] < b.max[0] and b.min[1] < mid[1] < b.max[1] and b.min[2] < mid[2] < b.max[2])
            assert not inside, f"{r} is buried in {o}"


def test_window_sill_is_a_hold_but_not_a_surface():
    spec, parts, ls, obs = _building()
    sill = [r for r in ls.rails if abs(r.height - 4.0) < 1e-3 and r.normal[2] > 0.5][0]
    s = sill.clamp_s(sill.project_s(np.array([2.5, 4.0, 8.0])))
    assert not BodyModel.mantle_is_valid(sill, s, obs), "no room to stand in a 1.2 m high window recess"
    assert BodyModel.pose_is_free(BodyModel.hang_center(sill, s), obs)
    head = [r for r in ls.rails if abs(r.height - 2.2) < 1e-3 and r.normal[2] > 0.5][0]
    nxt = BodyModel.find_hold_above(ls.rails, head, 0.5, obs)
    assert nxt is not None and abs(nxt[0].height - 4.0) < 1e-3, "from a window head the next hold is the sill above"


def test_roof_edges_link_all_the_way_round():
    spec, parts, ls, obs = _building()
    roof = [r for r in ls.rails if abs(r.height - spec.height) < 1e-3]
    assert len(roof) == 4
    for r in roof:
        assert r.links["right"].kind == "convex" and r.links["left"].kind == "convex"
