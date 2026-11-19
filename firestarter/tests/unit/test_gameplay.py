"""
Gameplay through the real input layer (README controls, tutorial, DESIGN Sections 3-7).

Every action here goes through ``InputManager`` (keys / mouse buttons / mouse motion) exactly as
``Game.handle_event`` feeds it from pygame - only the window and renderer are absent.
"""

import math

import numpy as np
import pytest

from tests.driver import Driver


def ring(radius, angles):
    return [[radius * math.cos(math.radians(a)), 0.0, radius * math.sin(math.radians(a))] for a in angles]


@pytest.fixture
def drv(state):
    return Driver(state)


@pytest.fixture
def tdrv(tutorial_state):
    return Driver(tutorial_state)


def gather(d, label, clicks=1, stop=1.3):
    assert d.walk_to({"item": label}, stop=stop)
    d.aim_at({"item": label})
    for _ in range(clicks):
        d.click("left")


def build_teepee_by_hand(d, n=5):
    d.walk_to({"pit": [0, 0, 0]}, stop=1.25)
    d.key("6")
    d.aim_at({"pit": [0, 0, 0]})
    assert d.world.ghost is not None and d.world.ghost["kind"] == "tinder"
    d.click("right")
    for tgt in ring(0.13, range(10, 370, 360 // n))[:n]:
        d.key("7")
        d.aim_at({"pit": tgt}, 0.2)
        d.click("right")
    d.step(0.5)


# ------------------------------------------------------------------------ movement & look
def test_wasd_moves_and_mouse_looks(drv):
    p = drv.player
    x0 = p.position.copy()
    yaw0 = p.yaw
    drv.motion(200, 0)
    assert p.yaw == pytest.approx(yaw0 + 200 * drv.st.settings.mouse_sensitivity)
    drv.key_down("w")
    drv.step(1.0)
    drv.key_up("w")
    moved = np.linalg.norm((p.position - x0)[[0, 2]])
    assert 1.5 < moved < 3.0               # ~2.6 m/s walking pace
    assert p.position[1] == pytest.approx(p.ground_height() + p.eye_height, abs=1e-6)


def test_sprint_drains_stamina_and_crouch_lowers_eye(drv):
    p = drv.player
    drv.motion(900, 0)                      # turn away from the pit
    drv.key_down("left shift")
    drv.key_down("w")
    drv.step(2.0)
    drv.key_up("w")
    drv.key_up("left shift")
    assert p.stamina < 85
    drv.key("left ctrl")
    drv.step(1.0)
    assert p.eye_height < 1.1
    drv.key("left ctrl")
    drv.step(1.0)
    assert p.eye_height > 1.5


def test_crouch_is_a_toggle_and_sprint_stands_up(drv):
    p = drv.player
    drv.key("left ctrl")                    # press and release: stays crouched
    drv.step(1.0)
    assert p.is_crouching and p.eye_height < 1.1
    drv.key_down("left shift")              # breaking into a run stands you up
    drv.step(0.5)
    drv.key_up("left shift")
    drv.step(1.0)
    assert not p.is_crouching and p.eye_height > 1.5


def test_hands_can_be_emptied(drv):
    w, p = drv.world, drv.player
    drv.key("3")
    drv.key("3")                            # tool keys always equip (no accidental holstering)
    assert p.primary_hand == "felling_axe"
    drv.key("1")
    drv.key("7")
    drv.key("7")                            # off-hand keys always select (8, RMB, 8, RMB ... feeding)
    assert p.primary_hand == "pocket_lighter" and w.current_offhand() == "kindling"
    drv.key("x")                            # X empties both hands
    assert p.primary_hand == "none" and w.current_offhand() == "none"
    drv.key("e")                            # Q / E cycle: the empty hand is the last slot
    assert w.current_offhand() == "tinder"
    drv.key("q")
    assert w.current_offhand() == "none"
    # an empty hand can still pick things up
    drv.key("x")
    assert p.primary_hand == "none"
    n0 = w.inventory.count("tinder")
    src = min((wi for wi in w.world_items if wi.kind == "tinder_source" and wi.count > 0),
              key=lambda wi: np.linalg.norm(wi.position - p.position))
    assert drv.walk_to(src.position, stop=1.2)
    drv.aim_at(src.position)
    drv.click("left")
    assert w.inventory.count("tinder") == n0 + 1


def test_time_skip_lets_a_log_fire_burn_down_realistically(state):
    from firestarter.automation import staging
    w = state.world
    g = w.thermal_grid
    staging.build_log_fire(w)
    for _ in range(30 * 60):                # a minute of real burning first
        state.step(1 / 30, move=False)
    m0 = sum(b.dry_mass for b in g.bodies)
    rep = w.apply_time_skip(1)
    # an hour later three oak logs are still burning, with part of the wood left
    assert rep["fire_after"] and rep["fire_kw_after"] > 3.0
    assert 0.1 * m0 < rep["fire_mass_left_kg"] < 0.7 * m0
    assert g.bodies and g.is_burning
    rep = w.apply_time_skip(8)
    # by the next morning it has burned out: only charcoal remnants and ash
    assert not rep["fire_after"] and not g.is_burning
    assert g.coal_bed.T < 80.0 and g.coal_bed.ash_kg > 0.02


def test_cannot_walk_into_the_fire_pit(drv):
    drv.aim_at({"pit": [0, 0, 0]})
    drv.key_down("w")
    drv.step(3.0)
    drv.key_up("w")
    d = math.hypot(drv.player.position[0], drv.player.position[2])
    assert d > 0.6


def test_invert_y_and_sensitivity_from_pause_menu(drv):
    inp = drv.st.input
    drv.key("escape")
    assert inp.is_paused and inp.in_menu
    t0 = drv.st.world.sim_time
    drv.step(1.0)
    assert drv.st.world.sim_time == t0          # the world is frozen while paused
    for _ in range(5):
        drv.key("s")                             # -> Mouse Sensitivity
    drv.key("d")
    assert drv.st.settings.mouse_sensitivity == pytest.approx(0.17)
    drv.key("s")                                 # -> Invert Mouse Y
    drv.key("return")
    assert drv.st.settings.invert_y
    drv.key("escape")
    assert not inp.is_paused
    pitch0 = drv.player.pitch
    drv.motion(0, 50)
    assert drv.player.pitch > pitch0             # inverted: mouse down looks up


# ------------------------------------------------------------------------ the tutorial, by hand
def test_tutorial_walkthrough_end_to_end(tdrv):
    d = tdrv
    tut = d.st.tutorial
    gather(d, "Birch bark curls", 2)
    gather(d, "Dry pine needles", 1)
    assert d.world.inventory.count("tinder") >= 3 and tut.index >= 1
    gather(d, "Bundle of dead pine twigs", 2)
    assert d.world.inventory.count("kindling") >= 6 and tut.index >= 2
    build_teepee_by_hand(d, 5)
    kind = [b for b in d.grid.bodies if b.geometry_type == "kindling"]
    assert len(kind) == 5 and all(b.orientation[0] > 40 for b in kind)
    assert d.world.structure_key == "teepee"
    assert tut.index >= 3
    # strike the lighter while crouched, hold it on the tinder
    d.key("left ctrl")
    d.step(0.4)
    d.key("1")
    d.aim_at({"pit": [0, 0.03, 0]})
    d.mouse_down("left")
    d.step(2.5)
    d.mouse_up("left")
    d.key("left ctrl")
    assert d.wait_until(lambda: d.grid.flame_hrr > 800, 8.0) is not None
    assert d.world.inventory.lighter_starts == 3
    assert d.wait_until(lambda: sum(1 for b in d.grid.bodies if b.geometry_type == "kindling" and b.is_flaming) >= 3, 40) is not None
    # feed split firewood from the woodpile
    d.walk_to({"woodpile": True}, stop=1.4)
    d.key("8")
    d.aim_at({"woodpile": True})
    for _ in range(3):
        d.click("left")
    assert d.world.inventory.count("firewood") >= 3
    d.walk_to({"pit": [0, 0, 0]}, stop=1.35)
    for tgt in ring(0.21, (50, 170, 290)):
        d.key("8")
        d.aim_at({"pit": tgt}, 0.2)
        d.click("right")
    assert sum(1 for b in d.grid.bodies if b.geometry_type == "firewood") == 3
    assert d.wait_until(lambda: tut.index >= 5, 90) is not None
    d.step(60.0)
    assert d.st.phone.credits.pending > 2
    d.key("tab")
    assert d.st.phone.is_open and d.st.phone.last_claimed > 2
    d.step(0.2)
    assert tut.current is None                   # all 6 steps done


# ------------------------------------------------------------------------ bow drill -> fire
def test_bow_drill_to_flame_without_the_lighter(drv):
    d = drv
    gather(d, "Birch bark curls", 1)
    gather(d, "Bundle of dead pine twigs", 1)
    build_kindling = ring(0.13, range(10, 370, 72))
    d.walk_to({"pit": [0, 0, 0]}, stop=1.25)
    for tgt in build_kindling:
        d.key("7")
        d.aim_at({"pit": tgt}, 0.2)
        d.click("right")
    d.walk_to({"hearthboard": True}, stop=0.85)
    d.key("2")
    d.key("left ctrl")
    d.step(0.4)
    d.aim_at({"hearthboard": True})
    assert d.world.target["kind"] == "hearthboard"
    d.click("left")
    assert d.world.crafting.bow_drill.has_v_notch
    d.key("4")
    d.aim_at({"hearthboard": True})
    d.mouse_down("left")
    d.mouse_down("right")                        # bear down
    assert d.world.bow_active
    t = 0.0
    while not d.world.crafting.bow_drill.has_ember and t < 120:
        d.saw(1.0)
        t += 1.0
    assert d.world.crafting.bow_drill.has_ember, d.world.crafting.bow_drill.socket_temp
    assert 30 < t < 110
    assert d.player.stamina < 70
    d.mouse_up("left")
    d.mouse_up("right")
    d.step(0.2)
    d.click("right")                             # tip the coal into the nest
    nest = d.world.crafting.active_tinder_nest
    assert nest is not None and nest.has_ember
    d.key("left ctrl")
    breaths = 0
    while d.world.crafting.active_tinder_nest is not None and breaths < 25:
        d.key("f", hold=0.5)
        d.step(0.9)
        breaths += 1
    assert d.world.crafting.active_tinder_nest is None and breaths <= 20
    assert d.wait_until(lambda: d.grid.flame_hrr > 400, 8) is not None
    assert d.world.inventory.lighter_starts == 4 and d.world.stats["embers_made"] == 1


def test_breath_hold_duration_maps_to_velocity(drv):
    v = drv.world.breath_velocity_from_hold
    assert v(0.1) < 0.8                          # a puff is too weak
    assert 1.5 < v(0.5) < 4.0                    # deep, even breath
    assert v(1.5) > 6.0                          # forceful: scatters embers


# ------------------------------------------------------------------------ forestry & time
def test_fell_split_and_season_with_tarp_and_time_skip(drv):
    d = drv
    st = d.st
    st.phone.credits.credits = 600.0             # staged: credits from earlier fires
    d.key("tab")
    d.key("return")                              # Store
    d.key("s"); d.key("s")
    d.key("return")                              # tarp
    d.key("s"); d.key("s")
    d.key("return")                              # steel axe
    assert d.world.inventory.tarps == 1 and d.world.inventory.axe == "steel"
    assert st.phone.credits.credits == pytest.approx(600 - 110 - 240)
    d.key("tab")
    # fell the landmark birch north-east of camp
    p = d.player
    p.position[0], p.position[2] = 6.1, -7.3
    p.snap_to_ground()
    tree = min((t for t in d.world.ecosystem.trees if t.is_standing and "Birch" in t.species.value),
               key=lambda t: np.linalg.norm(t.position[[0, 2]] - p.position[[0, 2]]))
    d.key("3")
    d.aim_at(tree.position + np.array([0.0, 1.0, 0.0]))
    assert d.world.target["kind"] == "tree"
    n_items = len(d.world.world_items)
    swings = 0
    while not tree.is_falling and not tree.is_fallen and swings < 40:
        d.click("left")
        d.step(0.4)
        swings += 1
    assert tree.is_falling or tree.is_fallen
    assert d.wait_until(lambda: tree.is_fallen, 10) is not None
    assert d.world.stats["trees_felled"] == 1
    # the whole tree lies on the ground, to be limbed and bucked (nothing is spawned yet)
    ft = next(f for f in d.world.fallen_trees if f.tree_id == tree.id)
    assert len(d.world.world_items) == n_items
    tot = ft.totals()
    assert tot["logs"] >= 5 and tot["kindling"] > 50 and tot["tinder"] >= 5
    # buck the first 45 cm round off the butt
    cut_pt = ft.point_at(0.3)
    d.walk_to(cut_pt + ft.side * 1.0, stop=0.35)
    d.aim_at(cut_pt)
    assert d.world.target["kind"] == "fallen_trunk"
    swings = 0
    while not ft.removed[0] and swings < 60:
        d.click("left")
        d.step(0.5)
        swings += 1
    assert ft.removed[0] and ft.cut[1]
    rnd = next(w for w in reversed(d.world.world_items) if w.kind == "fuel" and w.fuel and w.fuel.kind == "log")
    assert "round" in rnd.label and rnd.fuel.m_core > 0.9 and abs(rnd.fuel.length - 0.45) < 1e-6
    # split the round (big rounds take more than one blow)
    d.aim_at(rnd.position)
    n_before = len(d.world.world_items)
    blows = 0
    while rnd in d.world.world_items and blows < 6:
        d.click("left")
        d.step(0.5)
        blows += 1
    pieces = [w for w in d.world.world_items[n_before - 1:] if w.label.startswith("Split")]
    assert rnd not in d.world.world_items and len(pieces) >= 2
    # carry a green split piece to the rack (one that is in plain view, not behind the trunk)
    d.key("1")
    piece = None
    for cand in sorted(pieces, key=lambda w: np.linalg.norm(w.position - p.position)):
        d.aim_at(cand.position, 0.15)
        if d.world.target is not None and d.world.target["obj"] is cand:
            piece = cand
            break
    assert piece is not None and piece.fuel.m_core > 0.9
    d.click("left")
    assert d.world.inventory.count("firewood") >= 1
    d.walk_to({"rack": True}, stop=1.4)
    assert d.world.current_offhand() == "firewood"   # picking it up put it in the off hand
    d.aim_at({"rack": True})
    n_rack = len(d.world.drying_rack.logs)
    d.click("right")
    assert len(d.world.drying_rack.logs) == n_rack + 1
    d.click("left")                              # stretch the tarp over the rack
    assert d.world.drying_rack.has_tarp and d.world.inventory.tarps == 0
    before = d.world.drying_rack.mean_moisture()
    d.key("t")
    assert d.world.time_engine.is_menu_open
    while d.world.time_engine.selected.hours != 1440.0:
        d.key("s")
    d.key("return")
    rep = d.world.last_skip_report
    assert rep is not None and rep["hours"] == 1440.0
    after = d.world.drying_rack.mean_moisture()
    assert after < 0.25 < before
    assert rep["season"] in ("Summer", "Autumn")


def test_axe_splits_a_log_on_the_ground(drv):
    d = drv
    green = next(w for w in d.world.world_items if w.label == "Green oak log")
    d.walk_to(green.position, stop=1.3)
    d.key("3")
    d.aim_at(green.position)
    d.click("left")
    splits = [w for w in d.world.world_items if w.label.startswith("Split")]
    assert len(splits) >= 2 and green not in d.world.world_items


def test_knife_harvests_fungus_from_standing_dead_and_bark_for_cordage(drv):
    d = drv
    w = d.world
    dead = min((t for t in w.ecosystem.trees if t.is_standing and t.has_fungi),
               key=lambda t: np.linalg.norm(t.position[[0, 2]]))
    d.player.position[0], d.player.position[2] = dead.position[0] + 1.2, dead.position[2]
    d.player.snap_to_ground()
    d.key("2")
    d.aim_at(dead.position + np.array([0, 1.0, 0]))
    d.click("left")
    assert len(w.inventory.fungi) == 1
    live = min((t for t in w.ecosystem.trees if t.is_standing and t.stage.name == "MATURE"),
               key=lambda t: np.linalg.norm(t.position[[0, 2]] - d.player.position[[0, 2]]))
    d.player.position[0], d.player.position[2] = live.position[0] + 1.2, live.position[2]
    d.player.snap_to_ground()
    d.aim_at(live.position + np.array([0, 1.0, 0]))
    cord0 = w.inventory.cordage
    d.click("left")
    d.click("left")
    d.key("c")
    assert w.inventory.cordage == cord0 + 1


# ------------------------------------------------------------------------ accelerant & placement
def test_gasoline_on_cold_fuel_flashes(drv):
    d = drv
    st = d.st
    st.phone.credits.credits = 400.0
    d.key("tab"); d.key("return")
    for _ in range(5):
        d.key("s")
    d.key("return")
    d.key("tab")
    assert d.world.inventory.gasoline_kg == pytest.approx(2.8)
    from firestarter.automation.staging import build_teepee
    build_teepee(d.world, 5, 0)
    d.walk_to({"pit": [0, 0, 0]}, stop=1.3)
    d.aim_at({"pit": [0, 0.05, 0]})
    d.key("5")
    d.click("left")
    assert d.world.inventory.gasoline_kg < 2.8
    assert any(b.accelerant_mass > 0 for b in d.grid.bodies)
    d.key("1")
    d.mouse_down("left")
    d.step(1.0)
    d.mouse_up("left")
    peak = 0.0
    for _ in range(60):
        d.step(frames=1)
        peak = max(peak, d.grid.flame_hrr)
    assert peak > 40e3                           # a gasoline flash is violent


def test_scroll_rotates_and_shift_scroll_tilts_placement(drv):
    d = drv
    gather(d, "Bundle of dead pine twigs", 1)
    d.walk_to({"pit": [0, 0, 0]}, stop=1.25)
    d.key("7")
    d.aim_at({"pit": [0.1, 0, 0]})
    g0 = d.world.ghost
    assert g0 is not None
    d.click("wheel_up")
    assert d.player.held_item_rotation == pytest.approx(15.0)
    d.key_down("left shift")
    d.click("wheel_down")
    d.key_up("left shift")
    assert d.player.held_item_tilt == pytest.approx(0.0)
    d.step(frames=2)
    assert d.world.ghost["orientation"][0] == pytest.approx(0.0, abs=1.0)   # lying flat now
    d.click("middle")
    assert d.player.held_item_tilt is None and d.player.held_item_rotation == 0.0


def test_take_cold_fuel_back_out_of_the_pit(drv):
    d = drv
    build_teepee_by_hand_ok = True
    gather(d, "Birch bark curls", 1)
    d.walk_to({"pit": [0, 0, 0]}, stop=1.25)
    d.key("6")
    d.aim_at({"pit": [0, 0, 0]})
    d.click("right")
    assert len(d.grid.bodies) == 1
    d.key("2")
    d.aim_at(d.grid.bodies[0].position)
    d.click("left")
    assert len(d.grid.bodies) == 0 and d.world.inventory.count("tinder") == 1 and build_teepee_by_hand_ok


# ------------------------------------------------------------------------ menus
def test_book_pin_blueprint_and_ghost_snaps_to_slot(drv):
    d = drv
    d.key("b")
    assert d.st.book.is_open
    d.key("d"); d.key("d"); d.key("d"); d.key("d")      # -> V. Teepee
    assert d.st.book.get_current_page_data()["key"] == "teepee"
    d.key("return")
    assert d.st.book.selected_blueprint_key == "teepee"
    d.key("b")
    gather(d, "Birch bark curls", 1)
    d.walk_to({"pit": [0, 0, 0]}, stop=1.25)
    d.key("6")
    d.aim_at({"pit": [0.03, 0, 0.02]})
    assert d.world.ghost.get("slot") == "center_tinder"


def test_time_menu_navigation_and_one_hour_skip(drv):
    d = drv
    h0 = d.world.weather.total_hours
    d.key("t")
    d.key("w"); d.key("w")                       # 1 Hour
    assert d.world.time_engine.selected.hours == 1.0
    d.key("return")
    assert d.world.weather.total_hours == pytest.approx(h0 + 1.0, abs=0.05)
    assert not d.world.time_engine.is_menu_open


def test_help_overlay_and_escape_chain(drv):
    d = drv
    d.key("h")
    assert d.st.input.show_help
    d.key("escape")
    assert not d.st.input.show_help and not d.st.input.is_paused
    d.key("tab")
    d.key("escape")
    assert not d.st.phone.is_open and not d.st.input.is_paused


def test_hand_drill_with_river_stone(drv):
    """DESIGN 3.2: press 4 twice for the hand drill; palms spin the spindle, RMB presses the stone."""
    d = drv
    d.walk_to({"hearthboard": True}, stop=0.85)
    d.key("2")
    d.key("left ctrl")
    d.step(0.4)
    d.aim_at({"hearthboard": True})
    d.click("left")                              # carve the V-notch
    d.key("4")
    assert d.world.friction_mode == "bow"
    d.key("4")
    assert d.world.friction_mode == "hand"
    d.aim_at({"hearthboard": True})
    d.mouse_down("left")
    d.mouse_down("right")
    assert d.world.bow_active
    t = 0.0
    while not d.world.crafting.bow_drill.has_ember and t < 90:
        d.saw(1.0, amplitude=170, rate=1.8)
        t += 1.0
    d.mouse_up("left")
    d.mouse_up("right")
    assert d.world.crafting.bow_drill.has_ember and t < 80, (t, d.world.crafting.bow_drill.socket_temp)
    assert d.player.stamina < 60                 # hand drilling is exhausting
    assert d.world.crafting.bow_drill.cordage_condition == 1.0   # no cord involved


@pytest.mark.slow
def test_log_cabin_built_from_the_pinned_blueprint_catches(drv):
    """DESIGN 6: pin the Log Cabin, place pieces (they snap to the guide), light it - the roofed
    walls catch and the cabin draws a strong draft."""
    d = drv
    from firestarter.automation.staging import stock_for_log_cabin
    d.key("b")
    for _ in range(5):
        d.key("d")
    d.key("return")
    d.key("b")
    assert d.st.book.selected_blueprint_key == "log_cabin"
    stock_for_log_cabin(d.world)
    d.walk_to({"pit": [0, 0, 0]}, stop=1.15)
    d.key("6"); d.aim_at({"pit": [0, 0, 0]}); d.click("right")
    for _ in range(4):
        d.key("7"); d.aim_at({"pit": [0.05, 0, 0]}, 0.15); d.click("right")
    for _ in range(8):
        d.key("8"); d.aim_at({"pit": [0.0, 0, 0.15]}, 0.15); d.click("right")
    for _ in range(3):
        d.key("7"); d.aim_at({"pit": [0.0, 0.2, 0.0]}, 0.15); d.click("right")
    d.step(0.6)
    assert d.world.structure_key == "log_cabin", (d.world.structure_key, d.world.structure_score)
    d.key("left ctrl"); d.step(0.4); d.key("1")
    d.aim_at({"pit": [0, 0.03, 0]})
    d.mouse_down("left"); d.step(2.5); d.mouse_up("left"); d.key("left ctrl")
    assert d.wait_until(lambda: d.grid.flame_hrr > 800, 10) is not None
    d.step(90.0)
    assert sum(1 for b in d.grid.bodies if b.geometry_type == "firewood" and b.is_flaming) >= 3
    assert d.grid.c_d > 0.6 and d.grid.draft_velocity > 1.0


def test_autumn_leaf_litter_is_seasonal_tinder(drv):
    """DESIGN 7.1: autumn's fallen leaves are temporary tinder."""
    from firestarter.simulation.weather import Season
    d = drv
    w = d.world
    assert not any(wi.material == "dry_leaves" for wi in w.world_items)
    w.weather.set_time(13.0, Season.AUTUMN, 10)
    w.weather.force_precipitation(0.0)
    w.weather.ground_wetness = 0.1
    d.step(0.2)
    piles = [wi for wi in w.world_items if wi.material == "dry_leaves"]
    assert len(piles) == 3
    d.walk_to({"item": "Fallen dry leaves"}, stop=1.3)
    d.aim_at({"item": "Fallen dry leaves"})
    d.click("left")
    leaves = w.inventory.peek("tinder")
    assert leaves is not None and leaves.material == "dry_leaves" and leaves.moisture < 0.2
    w.weather.set_time(13.0, Season.WINTER, 1)
    d.step(0.2)
    assert not any(wi.material == "dry_leaves" for wi in w.world_items)


def test_punk_log_fruits_fungi_then_returns_to_the_soil(drv):
    """DESIGN 7.2: sound wood -> wet punk -> bracket fungi -> humus (fertile soil)."""
    from firestarter.simulation.weather import Season
    d = drv
    w = d.world
    log = next(wi for wi in w.world_items if wi.label == "Green oak log")
    w.weather.set_time(12.0, Season.AUTUMN, 1)
    w.apply_time_skip(24 * 80)
    assert log.fuel.species.name == "PUNK_WOOD"
    assert getattr(log, "fungi", 0) >= 1
    # harvest a bracket with the knife
    d.player.position[0], d.player.position[2] = log.position[0] + 1.0, log.position[2]
    d.player.snap_to_ground()
    d.key("2")
    d.aim_at(log.position)
    n0 = len(w.inventory.fungi)
    d.click("left")
    assert len(w.inventory.fungi) == n0 + 1
    fert0 = w.ecosystem.fertility_at(log.position)
    w.apply_time_skip(24 * 270)
    assert log not in w.world_items and log not in w.ground_logs
    assert w.ecosystem.fertility_at(log.position) > fert0


def test_spring_sap_makes_felled_wood_wetter(drv):
    from firestarter.simulation.weather import Season
    from firestarter.forestry.felling import TreeFellingSystem
    d = drv
    w = d.world
    out = {}
    for season in (Season.SUMMER, Season.SPRING):
        w.weather.set_time(12.0, season, 5)
        tree = next(t for t in w.ecosystem.trees if t.is_standing and t.stage.name == "MATURE" and t.species.name == "WHITE_PINE"
                    and t.id not in out.values())
        n0 = len(w.world_items)
        tree.notch_depth_m = tree.trunk_radius_m * 0.7
        while not tree.is_falling:
            TreeFellingSystem.chop_backcut(tree, "steel")
        while not tree.is_fallen:
            w.update(1 / 30)
        ft = next(f for f in w.fallen_trees if f.tree_id == tree.id)
        assert len(w.world_items) == n0
        out[season] = ft.m_core
    assert out[Season.SPRING] > out[Season.SUMMER] * 1.05
