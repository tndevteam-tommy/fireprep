"""DESIGN Section 5 (credits, store, smartphone, topo map) and Section 6 (blueprints & book)."""

import numpy as np
import pytest

from firestarter.economy.credits import CreditSystem
from firestarter.economy.store import StoreApp, DEFAULT_CATALOG
from firestarter.economy.smartphone import SmartphoneOS, PhoneAppType
from firestarter.economy.topomap import TopographicalMap
from firestarter.recipes.blueprints import ALL_BLUEPRINTS, match_structure, KIND_COMPAT
from firestarter.recipes.reference_book import ReferenceBook, PAGES
from firestarter.thermodynamics.combustion_body import CombustionBody
from firestarter.thermodynamics.wood_properties import WoodSpeciesType as W


# ------------------------------------------------------------------------ credits (5.1)
def test_credits_integrate_heat_and_efficiency():
    c = CreditSystem(starting_credits=0.0)
    earned_hot = sum(c.update(1.0, 800.0, 20e3, efficiency_ratio=1.0) for _ in range(60))
    c2 = CreditSystem(starting_credits=0.0)
    earned_smoky = sum(c2.update(1.0, 400.0, 20e3, efficiency_ratio=0.3) for _ in range(60))
    assert earned_hot > earned_smoky > 0
    assert c.credits == 0.0 and c.pending == pytest.approx(earned_hot)


def test_cold_or_tiny_fires_earn_nothing():
    c = CreditSystem(0.0)
    assert c.update(10.0, 90.0, 50e3) == 0.0
    assert c.update(10.0, 600.0, 100.0) == 0.0


def test_pending_credits_are_banked_when_the_phone_opens():
    phone = SmartphoneOS(CreditSystem(20.0))
    phone.credits.update(30.0, 700.0, 15e3, efficiency_ratio=0.9)
    pending = phone.credits.pending
    assert pending > 0
    phone.toggle()
    assert phone.credits.credits == pytest.approx(20.0 + pending) and phone.credits.pending == 0.0
    assert "claimed" in phone.status_message


# ------------------------------------------------------------------------ store
def test_catalog_contains_the_design_items():
    ids = {i.item_id for i in DEFAULT_CATALOG}
    assert {"lighter_refill", "waterproof_tarp", "steel_knife", "steel_axe", "gasoline_canister",
            "reference_book_advanced"} <= ids
    assert all(i.cost > 0 and i.description for i in DEFAULT_CATALOG)


def test_purchases_debit_credits_and_unique_items_sell_out():
    c = CreditSystem(1000.0)
    s = StoreApp()
    axe = next(i for i in s.catalog if i.item_id == "steel_axe")
    r = s.purchase(axe, c)
    assert r["success"] and c.credits == pytest.approx(1000.0 - axe.cost)
    again = s.purchase(axe, c)
    assert not again["success"] and "already own" in again["message"]
    poor = CreditSystem(5.0)
    assert not StoreApp().purchase(axe, poor)["success"] and poor.credits == 5.0


def test_phone_navigation():
    p = SmartphoneOS(CreditSystem(500.0))
    assert p.handle_input("select")["handled"] is False      # closed phone ignores input
    p.toggle()
    assert p.active_app == PhoneAppType.HOME
    p.handle_input("select")
    assert p.active_app == PhoneAppType.STORE
    p.handle_input("down")
    assert p.store.selected_index == 1
    r = p.handle_input("select")
    assert r["purchase"]["success"] and p.purchase_log
    p.handle_input("switch")
    assert p.active_app == PhoneAppType.TOPO_MAP
    z0 = p.map_zoom
    p.handle_input("up")
    assert p.map_zoom > z0
    p.handle_input("back")
    assert p.active_app == PhoneAppType.HOME
    st = p.get_display_state(np.array([0.0, 0.0, -40.0]), 18.0)
    assert st["gps_coords"] == (0.0, -40.0)


# ------------------------------------------------------------------------ topo map (5.2)
def test_topo_map_microclimates():
    m = TopographicalMap()
    assert m.get_microclimate_at(0.0, -40.0)[0] == "Marshy Creek Bed"
    assert m.get_microclimate_at(20.0, 50.0)[0] == "Sun-Baked Ridge"
    assert m.get_microclimate_at(-40.0, 10.0)[0] == "Hardwood Basin"
    assert m.get_microclimate_at(0.0, 0.0)[0] == "Forest Floor"
    assert m.get_microclimate_at(0.0, -40.0)[1] > m.get_microclimate_at(20.0, 50.0)[1]


def test_contour_lines_follow_the_terrain():
    m = TopographicalMap()
    segs = m.contour_segments(0.0, 0.0, 80.0, resolution=48)
    assert len(segs) > 50
    from firestarter.simulation import terrain
    for x1, z1, x2, z2, level in segs[::25]:
        assert terrain.height(x1, z1) == pytest.approx(level, abs=1.5)


# ------------------------------------------------------------------------ blueprints (6)
def test_all_five_architectures_exist_with_consistent_slots():
    assert set(ALL_BLUEPRINTS) == {"teepee", "log_cabin", "lean_to", "swedish_torch", "star"}
    for bp in ALL_BLUEPRINTS.values():
        assert bp.slots and bp.description and bp.draft_profile
        for s in bp.slots:
            assert s.expected_type in KIND_COMPAT
            assert abs(np.linalg.norm(s.axis) - 1.0) < 1e-9
            assert s.position[1] >= 0.0          # nothing below the pit floor


@pytest.mark.parametrize("key", ["teepee", "log_cabin", "lean_to", "swedish_torch", "star"])
def test_building_exactly_the_blueprint_is_recognised(key):
    bp = ALL_BLUEPRINTS[key]
    bodies = []
    for s in bp.slots:
        kind = s.expected_type
        b = CombustionBody(W.WHITE_PINE, geometry_type=kind, position=s.position.copy(), orientation=s.orientation.copy(),
                           dimensions=s.dimensions.copy())
        bodies.append(b)
    best, score, scores = match_structure(bodies, np.zeros(3))
    assert best == key and score == pytest.approx(1.0), scores


def test_random_pile_is_freeform():
    rng = np.random.default_rng(0)
    bodies = [CombustionBody(W.WHITE_PINE, geometry_type="firewood",
                             position=np.array([rng.uniform(-0.3, 0.3), 0.03, rng.uniform(-0.3, 0.3)]),
                             orientation=np.array([0.0, rng.uniform(0, 360), 0.0])) for _ in range(6)]
    assert match_structure(bodies, np.zeros(3))[0] == "freeform"


def test_next_open_slot_builds_bottom_up():
    bp = ALL_BLUEPRINTS["log_cabin"]
    for s in bp.slots:
        s.is_filled = False
    assert bp.next_open_slot("tinder").expected_type == "tinder"
    first_wood = bp.next_open_slot("firewood")
    assert first_wood.position[1] == pytest.approx(min(s.position[1] for s in bp.slots if s.expected_type == "firewood"))


# ------------------------------------------------------------------------ reference book
def test_reference_book_locks_and_unlocks_advanced_blueprints():
    book = ReferenceBook()
    assert book.is_blueprint_unlocked("teepee") and book.is_blueprint_unlocked("log_cabin")
    assert not book.is_blueprint_unlocked("swedish_torch")
    book.goto("swedish_torch")
    assert "Locked" in book.activate_current() and book.selected_blueprint_key is None
    assert book.get_current_page_data()["locked"]
    book.unlock_editions()
    book.activate_current()
    assert book.selected_blueprint_key == "swedish_torch"
    book.activate_current()
    assert book.selected_blueprint_key is None


def test_discovery_by_building_unlocks_a_page():
    book = ReferenceBook()
    assert book.discover("star") and book.is_blueprint_unlocked("star")
    assert not book.discover("teepee")        # built-ins are always known


def test_book_has_rich_pages_and_wraps():
    book = ReferenceBook()
    assert book.total_pages == len(PAGES) >= 12
    for _ in range(book.total_pages):
        d = book.get_current_page_data()
        assert d["title"] and d["subtitle"]
        book.next_page()
    assert book.current_page == 0
    book.prev_page()
    assert book.current_page == book.total_pages - 1
