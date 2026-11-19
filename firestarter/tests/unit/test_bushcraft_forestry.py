"""DESIGN Sections 3 (primitive fire) and 4 (forestry & seasoning)."""

import math

import numpy as np
import pytest

from firestarter.bushcraft.bow_drill import BowDrillSystem
from firestarter.bushcraft.hand_drill import HandDrillSystem
from firestarter.bushcraft.tinder_nest import TinderNest
from firestarter.bushcraft.fungi import SmolderingFungus, FungusType
from firestarter.thermodynamics.airflow_draft import AirflowDraftSolver
from firestarter.forestry.tree import Tree, TreeGrowthStage
from firestarter.forestry.felling import TreeFellingSystem
from firestarter.forestry.seasoning import RaisedDryingRack, GroundSeasoningSimulator, drying_time_constants
from firestarter.forestry.ecosystem import ForestEcosystem
from firestarter.simulation.inventory import make_log, make_firewood
from firestarter.thermodynamics.wood_properties import WoodSpeciesType as W


def bow_until_ember(drill, seconds=240.0, dt=0.1, speed=1.2, force=55.0):
    t = 0.0
    while t < seconds:
        drill.stroke(0.5, speed, force, dt)
        t += dt
        if drill.has_ember:
            return t
    return None


# ------------------------------------------------------------------------ bow drill (3.1)
def test_friction_power_formula():
    d = BowDrillSystem()
    # P = mu F omega r  (dry wood: no moisture penalty)
    assert d.calculate_friction_power(50.0, 15.0, 0.55) == pytest.approx(0.55 * 50 * 15 * 0.01)


def test_steady_bowing_makes_an_ember_in_about_a_minute():
    d = BowDrillSystem(has_v_notch=True)
    t = bow_until_ember(d)
    assert t is not None and 35.0 <= t <= 110.0, t
    assert d.socket_temp >= 380.0 and d.wood_dust_mass >= 0.08


def test_no_v_notch_no_ember():
    d = BowDrillSystem(has_v_notch=False)
    assert bow_until_ember(d, 200) is None
    assert d.socket_temp > 300          # it smokes, but the dust scatters


def test_damp_hearthboard_fails_and_lazy_bowing_fails():
    assert bow_until_ember(BowDrillSystem(has_v_notch=True, hearthboard_moisture=0.30), 200) is None
    assert bow_until_ember(BowDrillSystem(has_v_notch=True), 200, speed=0.05, force=20.0) is None


def test_socket_cools_when_you_stop():
    d = BowDrillSystem(has_v_notch=True)
    for _ in range(450):
        d.stroke(0.5, 1.2, 55, 0.1)
    hot = d.socket_temp
    assert hot > 200
    for _ in range(200):
        d.idle(0.1)
    assert d.socket_temp < hot - 60          # ~40 s thermal time constant


def test_cordage_wears_and_snaps():
    d = BowDrillSystem(has_v_notch=True)
    d.cordage_condition = 0.002
    for _ in range(100):
        r = d.stroke(0.5, 2.0, 80, 0.1)
    assert d.cordage_condition == 0.0 and r["success"] is False
    d.replace_cordage()
    assert d.stroke(0.5, 1.2, 55, 0.1)["success"]


def test_hand_drill_is_harder_and_stone_helps():
    plain = HandDrillSystem()
    stone = HandDrillSystem(has_stone_handle=True)
    for d in (plain, stone):
        for _ in range(900):
            d.drill_step(22.0, 100.0, 0.1)
    assert stone.socket_temp > plain.socket_temp
    assert HandDrillSystem().drill_step(20.0, 0.0, 0.1)["success"] is False


# ------------------------------------------------------------------------ breath (3.3)
def test_breath_regimes():
    weak, _, _ = AirflowDraftSolver.calculate_breath_effect(0.5, 500.0, reactivity=8.0)
    good, o2, _ = AirflowDraftSolver.calculate_breath_effect(2.5, 500.0, reactivity=8.0)
    _, _, blown = AirflowDraftSolver.calculate_breath_effect(7.0, 500.0)
    assert weak < good and good > 0 and blown
    assert o2 == pytest.approx(0.165)            # exhaled air ~16.5 % O2


def test_tinder_nest_bursts_into_flame_with_good_breaths():
    n = TinderNest("cedar_shavings", moisture=0.08)
    n.insert_ember(420.0)
    breaths = 0
    while not n.is_flaming and breaths < 40:
        n.blow(2.5, 0.6)
        n.update(1.2)
        breaths += 1
    assert n.is_flaming and 4 <= breaths <= 20


def test_blowing_too_hard_scatters_the_ember_and_too_soft_fades():
    n = TinderNest()
    n.insert_ember(420.0)
    assert n.blow(7.5)["status"] == "blown_out" and n.is_destroyed
    n2 = TinderNest()
    n2.insert_ember(420.0)
    for _ in range(10):
        n2.blow(0.4)
    assert not n2.is_flaming and n2.core_temp < 420.0


def test_unattended_ember_dies_in_a_few_minutes():
    n = TinderNest()
    n.insert_ember(420.0)
    t = 0.0
    while n.has_ember and t < 900:
        n.update(1.0)
        t += 1.0
    assert 60 < t < 400


def test_damp_tinder_will_not_flame():
    n = TinderNest("dry_grass", moisture=0.32)
    n.insert_ember(450.0)
    for _ in range(30):
        n.blow(2.5, 0.6)
        n.update(1.0)
    assert not n.is_flaming


def test_fungus_smoulders_at_design_rate():
    f = SmolderingFungus(FungusType.TINDER_FUNGUS, diameter_cm=8.0)
    assert f.catch_spark_or_ember(420.0)
    for _ in range(3600):
        f.update(1.0)
    assert 7.0 < f.diameter_cm < 7.6            # 0.5-1.5 cm per hour
    assert f.transfer_ember_to_tinder() is not None
    wet = SmolderingFungus(moisture=0.3)
    assert not wet.catch_spark_or_ember(500.0)


# ------------------------------------------------------------------------ felling (4.1)
def test_notch_then_backcut_fells_the_tree_and_yields_green_wood():
    t = Tree(W.WHITE_PINE, np.zeros(3), height_m=14.0, trunk_radius_m=0.2)
    assert not TreeFellingSystem.chop_backcut(t)["success"]        # notch first
    swings = 0
    while not TreeFellingSystem.chop_notch(t)["is_complete"]:
        swings += 1
    falling = False
    while not falling and swings < 200:
        falling = TreeFellingSystem.chop_backcut(t, fall_heading=0.3).get("tree_falling", False)
        swings += 1
    assert falling and 8 < swings < 60
    pieces, t_fall, hit = [], 0.0, False
    while not hit:
        hit, pieces = TreeFellingSystem.update_falling_tree(t, 0.02)
        t_fall += 0.02
    # rigid rod falling from ~1.5 deg: a few seconds (physically ~ 4-6 s for 14 m)
    assert 2.0 < t_fall < 8.0
    logs = [p for p in pieces if p["kind"] == "log"]
    assert len(logs) >= 2 and all(p["m_core"] == pytest.approx(1.10) for p in logs)
    assert t.stage == TreeGrowthStage.STUMP


def test_steel_axe_beats_stone_axe_and_oak_is_tougher():
    def swings(sp, axe):
        t = Tree(sp, np.zeros(3), trunk_radius_m=0.2)
        n = 0
        while not TreeFellingSystem.chop_notch(t, axe)["is_complete"]:
            n += 1
        return n
    assert swings(W.WHITE_PINE, "steel") * 2 <= swings(W.WHITE_PINE, "stone")
    assert swings(W.WHITE_OAK, "steel") > swings(W.WHITE_PINE, "steel")


# ------------------------------------------------------------------------ seasoning (4.2)
def test_tarped_rack_seasons_green_log_to_about_12_percent_in_60_summer_days():
    log = make_log(W.WHITE_OAK, m_surface=0.80, m_core=1.10)
    rack = RaisedDryingRack(np.zeros(3), has_tarp=True)
    rack.add_log(log)
    for _ in range(60):
        rack.update_seasoning(1.0, "Summer", rain_fraction=0.1)
    assert 0.08 < log.m_core < 0.2, log.m_core
    assert log.m_surface < log.m_core + 0.02


def test_tarp_matters_in_the_rain():
    a, b = make_log(W.WHITE_PINE, 0.8, 1.1), make_log(W.WHITE_PINE, 0.8, 1.1)
    open_rack, tarp_rack = RaisedDryingRack(np.zeros(3)), RaisedDryingRack(np.zeros(3), has_tarp=True)
    open_rack.add_log(a)
    tarp_rack.add_log(b)
    for _ in range(30):
        open_rack.update_seasoning(1.0, "Autumn", rain_fraction=0.45)
        tarp_rack.update_seasoning(1.0, "Autumn", rain_fraction=0.45)
    assert b.m_core < a.m_core - 0.05


def test_split_firewood_seasons_about_4x_faster_than_round_logs():
    _, tau_log = drying_time_constants(0.06, 25.0, 2.0)
    _, tau_split = drying_time_constants(0.03, 25.0, 2.0)
    assert tau_log / tau_split == pytest.approx(4.0, rel=0.05)
    _, tau_winter = drying_time_constants(0.06, -7.0, 2.0)
    assert tau_winter > 3.0 * tau_log


def test_log_left_on_wet_autumn_soil_rots_into_punk():
    log = make_log(W.PAPER_BIRCH, 0.5, 0.6)
    log.is_on_ground = True
    for day in range(90):
        GroundSeasoningSimulator.step_ground_log(log, 1.0, "Autumn", rain_intensity=0.45, soil_moisture=0.8)
    assert log.m_core > 0.8
    assert getattr(log, "is_decayed", False) and log.species == W.PUNK_WOOD


def test_ecosystem_zones_and_growth():
    eco = ForestEcosystem(seed=3, n_trees=120)
    assert len(eco.trees) >= 120
    assert all(math.hypot(t.position[0], t.position[2]) > 6.0 for t in eco.trees)
    def frac(sp, inside):
        sel = [t for t in eco.trees if inside(t.position[0], t.position[2])]
        return sum(t.species == sp for t in sel) / max(1, len(sel))
    basin = lambda x, z: math.hypot(x + 40.0, z - 10.0) < 30.0
    creek = lambda x, z: abs(z - (-40.0 + 6.0 * math.sin(x * 0.045) + 2.5 * math.sin(x * 0.11 + 1.3))) < 22.0
    # DESIGN 4.3 micro-climates: oak/birch hardwood basin, cedar along the creek
    assert frac(W.WHITE_OAK, basin) > 1.5 * frac(W.WHITE_OAK, lambda x, z: not basin(x, z))
    assert frac(W.RED_CEDAR, creek) > 1.5 * frac(W.RED_CEDAR, lambda x, z: not creek(x, z))
    n0 = len(eco.trees)
    eco.add_campfire_ash(np.zeros(3), 2.0)
    assert eco.fertility_at(np.zeros(3)) > 1.5
    stats = eco.step_ecosystem(90.0, np.array([2.0, 0, 0]), "Summer")
    assert len(eco.trees) == n0 + stats["new_saplings"]
    assert eco.nearest_tree(np.array([7.4, 0, -8.0]), 2.0) is not None


def test_hand_drill_with_stone_bearing_makes_an_ember():
    """Vigorous palm rubbing (~1 m/s) with a river-stone bearing: an ember in well under a minute."""
    bow = BowDrillSystem(has_v_notch=True)
    hd = HandDrillSystem(socket=bow)
    t = 0.0
    while not hd.has_ember and t < 120:
        hd.drill_step(1.0 / hd.spindle_radius_m, 100.0, 0.1, stone=True)
        t += 0.1
    assert hd.has_ember and 15 < t < 70, t
    # without the stone the hands slide down 3x faster and press less: slower or no ember
    bare = HandDrillSystem(socket=BowDrillSystem(has_v_notch=True))
    t2 = 0.0
    while not bare.has_ember and t2 < 120:
        bare.drill_step(1.0 / bare.spindle_radius_m, 100.0, 0.1, stone=False)
        t2 += 0.1
    assert (not bare.has_ember) or t2 > t
    assert bare.resets >= 2                     # bare hands slide down fast and keep re-gripping


# ------------------------------------------------------------------------ limbing & bucking (4.1)
def _felled(species=W.PAPER_BIRCH, h=12.5, r=0.15, direction=0.0):
    from firestarter.forestry.bucking import FallenTree
    t = Tree(species, np.array([0.0, 0.0, 0.0]), height_m=h, trunk_radius_m=r)
    t.fall_direction = direction
    ft = FallenTree(t, np.random.default_rng(3))
    ft.settle(lambda x, z: 0.0)
    return ft


def test_felled_tree_yield_scales_with_its_size():
    small, big = _felled(h=8.0, r=0.09), _felled(h=16.0, r=0.24)
    ts, tb = small.totals(), big.totals()
    assert tb["logs"] > ts["logs"] >= 1
    assert tb["kindling"] > ts["kindling"] and tb["tinder"] > ts["tinder"]
    # the bole tapers: butt rounds are thick logs, the thin top becomes fire logs / kindling
    assert big.round_yield(0)["kind"] == "log" and big.round_yield(0)["radius"] > big.round_yield(big.n_sections - 1)["radius"]
    assert big.round_yield(big.n_sections - 1)["kind"] in ("firewood", "kindling")
    # branch wood is a species-typical fraction of the stem (broadleaves carry more than conifers)
    oak, pine = _felled(W.WHITE_OAK, 14.0, 0.2), _felled(W.WHITE_PINE, 14.0, 0.2)
    frac = lambda ft: sum(l.volume for l in ft.limbs) / ft.stem_volume()
    assert frac(oak) == pytest.approx(0.32, rel=0.25) and frac(pine) == pytest.approx(0.15, rel=0.25)


def test_limb_yields_fire_logs_kindling_and_tinder():
    ft = _felled(W.WHITE_OAK, 14.0, 0.25)
    big = max(ft.limbs, key=lambda l: l.radius)
    y = ft.limb_yield(big)
    assert len(y["fire_logs"]) >= 1 and y["kindling"] > 10 and y["tinder"] >= 1
    assert all(0.01 < f["radius"] <= big.radius for f in y["fire_logs"])
    small = min(ft.limbs, key=lambda l: l.radius)
    assert len(ft.limb_yield(small)["fire_logs"]) <= len(y["fire_logs"])


def test_felled_stem_rests_straight_and_rounds_follow_the_taper():
    ft = _felled(direction=math.pi / 2)
    assert np.allclose(ft.dir, [0.0, 0.0, 1.0], atol=1e-9)
    a, b = ft.point_at(0.0), ft.point_at(5.0)
    assert a[1] == pytest.approx(ft.r0 * 0.9, rel=0.2)             # lying on flat ground
    assert b[2] - a[2] == pytest.approx(5.0)
    assert ft.radius_at(0.0) == pytest.approx(0.15) and ft.radius_at(ft.height) == pytest.approx(0.0)


def test_bucking_frees_rounds_only_when_both_ends_are_cut(state):
    w = state.world
    ft = _felled()
    w.fallen_trees.append(ft)
    w.inventory.axe = "steel"
    n0 = len(w.world_items)

    def swing(fn, *a):
        w.player.stamina = 100.0
        return fn(*a)

    # the crown is in the way near the top until it is limbed
    top_limb = ft.limbs[-1]
    assert swing(w.buck_tree, ft, top_limb.s)["success"] is False
    # cutting boundary 2 first frees nothing; boundary 1 then frees rounds 0 and 1
    while not ft.cut[2]:
        swing(w.buck_tree, ft, 0.9)
    assert not ft.removed.any() and len(w.world_items) == n0
    swings = 0
    while not ft.cut[1]:
        swing(w.buck_tree, ft, 0.5)
        swings += 1
    assert ft.removed[0] and ft.removed[1] and ft.rounds_left == ft.n_sections - 2
    rounds = [wi for wi in w.world_items[n0:] if wi.fuel and wi.fuel.kind == "log"]
    assert len(rounds) == 2 and all(r.fuel.m_core > 0.9 for r in rounds)     # green wood
    # a 30 cm steel-axe cut takes several swings; a stone axe takes far more
    need = 2 * ft.radius_at(0.45) * 0.9
    assert swings == math.ceil(need / FallenTreeDepth("steel"))
    assert FallenTreeDepth("stone") < FallenTreeDepth("steel") / 2


def FallenTreeDepth(axe):
    from firestarter.forestry.bucking import FallenTree
    return FallenTree.swing_depth(axe, W.PAPER_BIRCH)


def test_limbing_drops_fuel_piles_and_whole_tree_can_be_worked_up(state):
    w = state.world
    ft = _felled(h=9.0, r=0.11)
    w.fallen_trees.append(ft)
    w.inventory.axe = "steel"
    tot = ft.totals()
    n0 = len(w.world_items)
    for l in ft.limbs:
        while not l.removed:
            w.player.stamina = 100.0
            w.limb_tree(ft, l)
    piles = [wi for wi in w.world_items[n0:] if wi.kind in ("kindling_pile", "tinder_source")]
    assert piles and all(getattr(p, "no_regrow", False) for p in piles)
    guard = 0
    while ft in w.fallen_trees and guard < 2000:
        i = ft.nearest_boundary(ft.bole_length)
        w.player.stamina = 100.0
        w.buck_tree(ft, i * 0.45 if i is not None else ft.bole_length + 0.2)
        guard += 1
    assert ft not in w.fallen_trees and ft.finished
    new = w.world_items[n0:]
    logs = sum(1 for wi in new if wi.fuel and wi.fuel.kind == "log")
    kindling = sum(wi.count for wi in new if wi.kind == "kindling_pile")
    tinder = sum(wi.count for wi in new if wi.kind == "tinder_source")
    assert logs == tot["logs"] and tinder == tot["tinder"]
    assert kindling >= tot["kindling"] * 0.8


def test_big_rounds_split_into_more_pieces_over_several_blows(state):
    from firestarter.simulation.inventory import FuelItem
    w = state.world
    counts = {}
    for r in (0.06, 0.14):
        it = FuelItem("log", W.WHITE_OAK, 0.6, 0.8, length=0.45, radius=r)
        wi = w._drop_fuel(it, np.array([3.0, 0.0, 3.0]), 0.0, "Green oak round")
        blows = 0
        while wi in w.world_items:
            w.player.stamina = 100.0
            res = w.split_log(wi)
            blows += 1
        counts[r] = (res["pieces"], blows)
    assert counts[0.06][0] == 2 and counts[0.06][1] == 1
    assert counts[0.14][0] >= 6 and counts[0.14][1] >= 2
