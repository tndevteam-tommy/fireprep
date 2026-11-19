"""
Generates the JSON playback scripts in this folder.

    python automation/scripts/generate_scripts.py

Each script drives the real game through keyboard / mouse events (see automation/playback.py)
and records assertions plus screenshots that the visual test-suite analyses.  Steps that
*stage* a scene directly (``exec`` / ``teleport`` / ``set_time``) are only used to compose
shots or to skip hours of grinding, and are commented as such.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def ring(radius, angles):
    return [[round(radius * math.cos(math.radians(a)), 3), 0.0, round(radius * math.sin(math.radians(a)), 3)] for a in angles]


def write(name, description, settings, steps):
    data = {"name": name, "description": description, "settings": settings, "steps": steps}
    with open(os.path.join(HERE, f"{name}.json"), "w") as f:
        json.dump(data, f, indent=1)
    print(f"{name}: {len(steps)} steps")


BASE = {"width": 1280, "height": 720, "seed": 7, "start_hour": 18.5, "fixed_dt": 0.0333}


# ------------------------------------------------------------------------------------------
def tutorial():
    S = []
    A = S.append
    A({"log": "Tutorial walkthrough: gather, build a teepee, light it, feed it, claim credits"})
    A({"wait": 0.6}); A({"screenshot": "00_arrival"})
    A({"assert": "tutorial.index == 0", "message": "tutorial starts at step 1"})
    A({"assert": "world.inventory.count('tinder') == 0 and world.inventory.count('kindling') == 0", "message": "starts with no fuel"})
    A({"walk_to": {"item": "Birch bark curls"}, "stop": 1.35, "timeout": 12})
    A({"aim_at": {"item": "Birch bark curls"}, "duration": 0.35})
    A({"assert": "world.target is not None and world.target['kind'] == 'tinder_source'", "message": "crosshair targets the birch curls"})
    A({"screenshot": "01_aim_birch_prompt"})
    A({"click": "left"}); A({"wait": 0.2}); A({"click": "left"}); A({"wait": 0.2})
    A({"walk_to": {"item": "Dry pine needles"}, "stop": 1.35, "timeout": 12})
    A({"aim_at": {"item": "Dry pine needles"}}); A({"click": "left"}); A({"wait": 0.2})
    A({"assert": "world.inventory.count('tinder') >= 3", "message": "gathered 3 tinder bundles", "value": "world.inventory.count('tinder')"})
    A({"assert": "tutorial.index >= 1", "message": "tutorial advanced past 'Gather tinder'"})
    A({"expect_sound": "rustle"})
    A({"walk_to": {"item": "Bundle of dead pine twigs"}, "stop": 1.3, "timeout": 12})
    A({"aim_at": {"item": "Bundle of dead pine twigs"}})
    A({"click": "left"}); A({"wait": 0.2}); A({"click": "left"}); A({"wait": 0.2})
    A({"assert": "world.inventory.count('kindling') >= 6", "message": "gathered 6 kindling twigs", "value": "world.inventory.count('kindling')"})
    A({"walk_to": {"pit": [0.0, 0.0, 0.0]}, "stop": 1.25, "timeout": 12})
    A({"key": "6"}); A({"aim_at": {"pit": [0.0, 0.0, 0.0]}}); A({"wait": 0.1})
    A({"assert": "world.ghost is not None and world.ghost['kind'] == 'tinder'", "message": "placement ghost shown for tinder"})
    A({"click": "right"}); A({"wait": 0.2})
    A({"assert": "any(b.geometry_type == 'tinder' for b in grid.bodies)", "message": "tinder nest placed in the pit"})
    for i, tgt in enumerate(ring(0.13, range(10, 370, 72))):
        A({"key": "7"}); A({"aim_at": {"pit": tgt}, "duration": 0.25}); A({"wait": 0.05})
        if i == 2:
            A({"screenshot": "02_placing_kindling_ghost"})
        A({"click": "right"}); A({"wait": 0.15})
    A({"wait": 0.5})
    A({"assert": "sum(1 for b in grid.bodies if b.geometry_type == 'kindling') >= 5", "message": "5 kindling sticks in the pit"})
    A({"assert": "all(b.orientation[0] > 40 for b in grid.bodies if b.geometry_type == 'kindling')", "message": "kindling leans into a teepee (tilt > 40 deg)"})
    A({"assert": "world.structure_key == 'teepee'", "message": "the lay is recognised as a Teepee fire", "value": "(world.structure_key, round(world.structure_score, 2))"})
    A({"assert": "tutorial.index >= 3", "message": "tutorial reached 'Strike the lighter'"})
    A({"screenshot": "03_fire_lay_teepee"})
    A({"key": "left ctrl"}); A({"wait": 0.4}); A({"key": "1"})
    A({"aim_at": {"pit": [0.0, 0.03, 0.0]}})
    A({"mouse_down": "left"}); A({"wait": 1.0}); A({"screenshot": "04_lighter_in_hand"}); A({"wait": 1.6}); A({"mouse_up": "left"})
    A({"key": "left ctrl"})
    A({"wait_until": "grid.flame_hrr > 800", "timeout": 8, "message": "tinder bursts into flame"})
    A({"assert": "world.inventory.lighter_starts == 3", "message": "one lighter start consumed", "value": "world.inventory.lighter_starts"})
    A({"expect_sound": "lighter_strike"})
    A({"wait": 3.0}); A({"screenshot": "05_tinder_flames"})
    A({"wait_until": "sum(1 for b in grid.bodies if b.geometry_type == 'kindling' and b.is_flaming) >= 3", "timeout": 40, "message": "kindling catches from the tinder"})
    A({"wait": 4.0}); A({"screenshot": "06_kindling_burning"})
    A({"walk_to": {"woodpile": True}, "stop": 1.4, "timeout": 12})
    A({"key": "8"}); A({"aim_at": {"woodpile": True}})
    for _ in range(3):
        A({"click": "left"}); A({"wait": 0.15})
    A({"assert": "world.inventory.count('firewood') >= 3", "message": "took 3 split firewood"})
    A({"walk_to": {"pit": [0.0, 0.0, 0.0]}, "stop": 1.35, "timeout": 12})
    for tgt in ring(0.21, (50, 170, 290)):
        A({"key": "8"}); A({"aim_at": {"pit": tgt}, "duration": 0.25}); A({"click": "right"}); A({"wait": 0.15})
    A({"assert": "sum(1 for b in grid.bodies if b.geometry_type == 'firewood') >= 3", "message": "firewood added to the fire"})
    A({"wait": 6.0}); A({"screenshot": "07_feeding_firewood"})
    A({"aim_at": {"pit": [0.0, 0.05, 0.0]}})
    A({"key": "f", "hold": 0.55}); A({"wait": 0.3})
    A({"assert": "world.last_breath_velocity > 1.5 and world.last_breath_velocity < 4.5", "message": "a 0.55 s breath is a deep, even breath", "value": "world.last_breath_velocity"})
    A({"wait_until": "sum(1 for b in grid.bodies if b.geometry_type == 'firewood' and (b.is_flaming or b.t_surface > 350)) >= 1", "timeout": 90, "message": "split firewood catches"})
    A({"screenshot": "08_firewood_caught"})
    A({"fast_forward": 120}); A({"wait": 1.0})
    A({"assert": "grid.is_burning and grid.core_coal_temp > 400", "message": "a hot fire with glowing coals after 2 more minutes", "value": "(grid.fire_stage(), round(grid.core_coal_temp))"})
    A({"assert": "phone.credits.pending > 5", "message": "fire-mastery credits accrued", "value": "round(phone.credits.pending, 1)"})
    A({"screenshot": "09_coals_forming"})
    A({"key": "tab"}); A({"wait": 0.6})
    A({"assert": "phone.is_open and phone.last_claimed > 5 and phone.credits.credits > 25", "message": "opening the phone banks the credits", "value": "(round(phone.last_claimed, 1), round(phone.credits.credits, 1))"})
    A({"assert": "tutorial.current is None", "message": "tutorial completed"})
    A({"screenshot": "10_phone_claimed"})
    A({"key": "tab"}); A({"wait": 0.3})
    A({"assert": "not phone.is_open", "message": "Tab closes the phone"})
    write("01_tutorial_walkthrough", "README Early-Game Tutorial, end to end, driven only by keyboard/mouse events.",
          dict(BASE, preset="High", render_stride=12), S)


# ------------------------------------------------------------------------------------------
def bow_drill():
    S = []
    A = S.append
    A({"log": "Primitive fire: carve a V-notch, bow-drill an ember, nest it in tinder and breathe it into flame"})
    A({"wait": 0.4})
    A({"walk_to": {"item": "Birch bark curls"}, "stop": 1.35, "timeout": 12}); A({"aim_at": {"item": "Birch bark curls"}}); A({"click": "left"}); A({"wait": 0.2})
    A({"walk_to": {"item": "Bundle of dead pine twigs"}, "stop": 1.3, "timeout": 12}); A({"aim_at": {"item": "Bundle of dead pine twigs"}})
    A({"click": "left"}); A({"wait": 0.2}); A({"click": "left"}); A({"wait": 0.2})
    A({"walk_to": {"pit": [0, 0, 0]}, "stop": 1.25, "timeout": 12})
    for tgt in ring(0.13, range(10, 370, 72)):
        A({"key": "7"}); A({"aim_at": {"pit": tgt}, "duration": 0.25}); A({"click": "right"}); A({"wait": 0.15})
    A({"wait": 0.6})
    A({"assert": "sum(1 for b in grid.bodies if b.geometry_type == 'kindling' and b.orientation[0] > 40) >= 5", "message": "kindling teepee stands without tinder (ends pushed into the soil)"})
    A({"walk_to": {"hearthboard": True}, "stop": 0.85, "timeout": 12})
    A({"key": "2"}); A({"key": "left ctrl"}); A({"wait": 0.4}); A({"aim_at": {"hearthboard": True}})
    A({"assert": "world.target is not None and world.target['kind'] == 'hearthboard'", "message": "aiming at the hearthboard"})
    A({"assert": "not world.crafting.bow_drill.has_v_notch", "message": "board starts without a notch"})
    A({"click": "left"}); A({"wait": 0.3})
    A({"assert": "world.crafting.bow_drill.has_v_notch", "message": "knife carved a 45-degree V-notch"})
    A({"expect_sound": "whittle"})
    A({"key": "4"}); A({"wait": 0.2}); A({"aim_at": {"hearthboard": True}})
    A({"mouse_down": "left"}); A({"mouse_down": "right"}); A({"wait": 0.2})
    A({"assert": "world.bow_active", "message": "bow drill engaged"})
    A({"saw": {"duration": 14, "amplitude": 150, "rate": 1.4}})
    A({"assert": "world.crafting.bow_drill.socket_temp >= 99", "message": "friction heats the socket to the drying plateau", "value": "round(world.crafting.bow_drill.socket_temp)"})
    A({"saw": {"duration": 22, "amplitude": 150, "rate": 1.4}})
    A({"assert": "world.crafting.bow_drill.is_smoking", "message": "the socket smokes (> 200 C)", "value": "round(world.crafting.bow_drill.socket_temp)"})
    A({"screenshot": "01_bow_drilling_smoke"})
    A({"saw": {"duration": 55, "amplitude": 150, "rate": 1.4}})
    A({"wait_until": "world.crafting.bow_drill.has_ember", "timeout": 5, "message": "an ember forms in the notch"})
    A({"assert": "player.stamina < 60", "message": "bow drilling is hard work (stamina spent)", "value": "round(player.stamina)"})
    A({"expect_sound": "friction_squeak"})
    A({"screenshot": "02_ember_in_notch"})
    A({"mouse_up": "left"}); A({"mouse_up": "right"}); A({"wait": 0.2})
    A({"click": "right"}); A({"wait": 0.3})
    A({"assert": "world.crafting.active_tinder_nest is not None and world.crafting.active_tinder_nest.has_ember", "message": "ember tipped into the tinder nest"})
    A({"key": "left ctrl"}); A({"wait": 0.3})
    A({"screenshot": "03_nest_with_ember"})
    for _ in range(12):
        A({"key": "f", "hold": 0.5}); A({"wait": 0.9})
    A({"wait_until": "grid.flame_hrr > 400", "timeout": 8, "message": "breathing the nest into flame lights the teepee"})
    A({"assert": "world.crafting.active_tinder_nest is None", "message": "the flaming nest was tucked into the fire lay"})
    A({"assert": "any(b.geometry_type == 'tinder' for b in grid.bodies)", "message": "burning nest now sits in the pit"})
    A({"expect_sound": "breath"})
    A({"aim_at": {"pit": [0.0, 0.2, 0.0]}}); A({"wait": 2.0})
    A({"screenshot": "04_friction_fire_flames"})
    A({"wait_until": "sum(1 for b in grid.bodies if b.geometry_type == 'kindling' and b.is_flaming) >= 2", "timeout": 45, "message": "the kindling catches"})
    A({"assert": "world.stats['embers_made'] >= 1 and world.inventory.lighter_starts == 4", "message": "fire made by friction alone - the lighter was never used"})
    write("02_bow_drill_primitive_fire", "DESIGN 3.1 / 3.3: carve notch, bow drill (mouse sawing), ember, tinder nest, breathing.",
          dict(BASE, preset="Medium", render_stride=30, no_tutorial=True), S)


# ------------------------------------------------------------------------------------------
def forestry():
    S = []
    A = S.append
    A({"log": "Forestry: buy gear, fell a birch, limb & buck it, split a round, rack the green wood, tarp it, season 60 days, watch a ground log rot"})
    A({"exec": "phone.credits.credits += 600  # staged: credits earned from earlier fires"})
    A({"wait": 0.3})
    A({"key": "tab"}); A({"wait": 0.3})
    A({"assert": "phone.is_open and phone.active_app.value == 'Home'", "message": "phone opens on the home screen"})
    A({"key": "return"}); A({"wait": 0.2})
    A({"assert": "phone.active_app.value == 'Store ($)'", "message": "Enter opens the Store"})
    for _ in range(2):
        A({"key": "s"})
    A({"key": "return"}); A({"wait": 0.2})
    A({"assert": "world.inventory.tarps == 1", "message": "bought a waterproof tarp"})
    for _ in range(2):
        A({"key": "s"})
    A({"key": "return"}); A({"wait": 0.2})
    A({"assert": "world.inventory.axe == 'steel'", "message": "bought the drop-forged felling axe"})
    A({"expect_sound": "purchase"})
    A({"screenshot": "01_store_purchases"})
    A({"key": "tab"}); A({"wait": 0.2})
    A({"teleport": [6.1, -7.3], "look_at": [7.4, 1.0, -8.6]})
    A({"key": "3"}); A({"aim_at": {"tree": "nearest", "species": "Birch", "stage": "Mature", "height": 1.0}}); A({"wait": 0.2})
    A({"assert": "world.target is not None and world.target['kind'] == 'tree'", "message": "aiming at a standing birch"})
    A({"exec": "scratch['tree'] = world.target['obj']; scratch['n_items'] = len(world.world_items)"})
    for i in range(16):
        A({"click": "left"}); A({"wait": 0.45})
    A({"assert": "scratch['tree'].notch_depth_m > 0 and scratch['tree'].backcut_depth_m > 0", "message": "face notch and back-cut chopped"})
    A({"expect_sound": "chop"})
    A({"wait_until": "scratch['tree'].is_fallen", "timeout": 8, "message": "the birch falls (rigid-body pivot about the hinge)"})
    A({"exec": "scratch['ft'] = next(f for f in world.fallen_trees if f.tree_id == scratch['tree'].id)"})
    A({"assert": "len(world.world_items) == scratch['n_items'] and scratch['ft'].limbs_left > 3 and scratch['ft'].rounds_left > 5",
       "message": "the whole birch lies on the ground, crown and all, waiting to be limbed and bucked",
       "value": "(scratch['ft'].rounds_left, scratch['ft'].limbs_left, scratch['ft'].totals())"})
    A({"expect_sound": "tree_fall"})
    A({"wait": 0.5}); A({"screenshot": "02_felled_birch"})
    # limbing: cut the lowest branch off where it meets the trunk
    A({"exec": "scratch['limb'] = min((l for l in scratch['ft'].limbs if not l.removed), key=lambda l: l.s); "
               "scratch['la'] = scratch['ft'].point_at(scratch['limb'].s, terrain.height); "
               "player.position[0], player.position[2] = (scratch['la'] - scratch['ft'].dir * 1.1 + scratch['ft'].side * 0.5)[[0, 2]]; "
               "player.snap_to_ground(); player.look_at(scratch['la'] + scratch['ft'].limb_direction(scratch['limb']) * 0.25)  # staged: walk up the trunk"})
    A({"wait": 2.5})
    A({"assert": "world.target is not None and world.target['kind'] == 'fallen_limb'", "message": "crosshair on a limb of the felled birch",
       "value": "world.target['kind'] if world.target else None"})
    A({"screenshot": "02b_limbing_prompt"})
    A({"exec": "scratch['n_items'] = len(world.world_items)"})
    for i in range(6):
        A({"click": "left"}); A({"wait": 0.7})
    A({"assert": "scratch['limb'].removed and any(w.kind in ('kindling_pile', 'tinder_source') for w in world.world_items[scratch['n_items']:])",
       "message": "the limb comes off as green kindling and a tinder bundle of twigs & bark",
       "value": "[w.label for w in world.world_items[scratch['n_items']:]]"})
    # bucking: cross-cut a 45 cm round off the butt
    A({"exec": "scratch['cp'] = scratch['ft'].point_at(0.3, terrain.height); "
               "player.position[0], player.position[2] = (scratch['cp'] + scratch['ft'].side * 1.05)[[0, 2]]; "
               "player.snap_to_ground(); player.look_at(scratch['cp'])  # staged: walk back to the butt"})
    A({"wait": 4.0})
    A({"assert": "world.target is not None and world.target['kind'] == 'fallen_trunk'", "message": "crosshair on the felled trunk"})
    for i in range(5):
        A({"click": "left"}); A({"wait": 0.7})
    A({"screenshot": "02c_bucking_kerf"})
    A({"wait": 3.0})
    for i in range(9):
        A({"click": "left"}); A({"wait": 0.7})
    A({"wait_until": "scratch['ft'].removed[0]", "timeout": 12, "message": "the first round comes free once its kerf is cut through"})
    A({"exec": "scratch['log'] = [w for w in world.world_items if w.kind == 'fuel' and w.fuel and w.fuel.kind == 'log' and 'round' in w.label][-1]"})
    A({"assert": "scratch['log'].fuel.m_core > 0.9", "message": "freshly felled wood is green (core > 90 % moisture)", "value": "round(scratch['log'].fuel.m_core, 2)"})
    A({"exec": "player.look_at(scratch['log'].position)"}); A({"wait": 2.5})
    A({"assert": "world.target is not None and world.target['obj'] is scratch['log']", "message": "aiming at the freed round"})
    for i in range(3):
        A({"click": "left"}); A({"wait": 0.7})
    A({"assert": "scratch['log'] not in world.world_items and sum(1 for w in world.world_items if w.label.startswith('Split')) >= 2",
       "message": "the round splits into firewood with the axe",
       "value": "sum(1 for w in world.world_items if w.label.startswith('Split'))"})
    A({"screenshot": "02d_split_round"})
    A({"key": "1"})
    A({"exec": "for _p in sorted((w for w in world.world_items if w.label.startswith('Split')), key=lambda w: np.linalg.norm(w.position - player.position)):\n"
               "    player.look_at(_p.position)\n"
               "    _t = world.raycast()\n"
               "    if _t is not None and _t['obj'] is _p:\n"
               "        break  # a split piece in plain view (not hidden behind the trunk)"})
    A({"wait": 0.3})
    A({"click": "left"}); A({"wait": 0.2})
    A({"assert": "world.inventory.count('firewood') >= 1 and world.current_offhand() == 'firewood'", "message": "picked up a green split piece"})
    A({"teleport": [3.3, 3.3], "look_at": {"rack": True}}); A({"wait": 0.2})
    A({"aim_at": {"rack": True}})
    A({"key": "8"}); A({"click": "right"}); A({"wait": 0.2})
    A({"assert": "len(world.drying_rack.logs) == 3", "message": "green firewood stacked on the raised rack"})
    A({"click": "left"}); A({"wait": 0.3})
    A({"assert": "world.drying_rack.has_tarp and world.inventory.tarps == 0", "message": "tarp stretched over the rack"})
    A({"expect_sound": "tarp"})
    A({"screenshot": "03_tarped_rack"})
    A({"exec": "scratch['rack_m0'] = world.drying_rack.mean_moisture(); scratch['ground0'] = world.ground_logs[0].fuel.moisture"})
    A({"key": "t"}); A({"wait": 0.2})
    A({"assert": "world.time_engine.is_menu_open", "message": "T opens the Pass Time menu"})
    for _ in range(3):
        A({"key": "s"})
    A({"screenshot": "04_time_menu"})
    A({"assert": "world.time_engine.selected.hours == 1440", "message": "selected the 60-day cure period"})
    A({"key": "return"}); A({"wait": 0.5})
    A({"assert": "world.weather.day_of_season >= 60", "message": "60 days passed", "value": "(world.weather.current_season.value, world.weather.day_of_season)"})
    A({"assert": "world.drying_rack.mean_moisture() < 0.22 and world.drying_rack.mean_moisture() < scratch['rack_m0'] * 0.3", "message": "tarped rack seasoned green wood toward ~12-20 % EMC", "value": "(round(scratch['rack_m0'], 2), round(world.drying_rack.mean_moisture(), 3))"})
    A({"assert": "min(l.m_core for l in world.drying_rack.logs) < 0.2", "message": "the best log cured below 20 % core moisture", "value": "[round(l.m_core, 3) for l in world.drying_rack.logs]"})
    A({"screenshot": "05_after_60_days"})
    A({"key": "t"}); A({"key": "s"}); A({"key": "return"}); A({"wait": 0.5})
    A({"assert": "world.weather.current_season.value == 'Autumn'", "message": "a further 90 days turned the season to autumn"})
    A({"assert": "any(w.fuel.species.value == 'Punk Wood' for w in world.ground_logs if w.fuel)", "message": "the green oak left on the soil rotted into punk wood", "value": "[(w.label, round(w.fuel.moisture, 2)) for w in world.ground_logs if w.fuel][:3]"})
    A({"set_time": {"hour": 13.5}})   # staged: view the autumn camp by daylight (the skip ends after dusk)
    A({"teleport": [-1.0, 6.0], "look_at": [-4.0, 0.3, 2.2]}); A({"wait": 0.3})
    A({"screenshot": "06_autumn_camp"})
    A({"aim_at": {"rack": True}}); A({"teleport": [3.3, 3.3], "look_at": {"rack": True}}); A({"wait": 0.2}); A({"aim_at": {"rack": True}})
    A({"click": "left"}); A({"wait": 0.2})
    A({"assert": "any(world.inventory.peek(k) is not None and world.inventory.peek(k).moisture < 0.25 for k in ('log', 'firewood'))", "message": "took a seasoned piece off the rack", "value": "[round(world.inventory.peek(k).moisture, 3) for k in ('log', 'firewood') if world.inventory.peek(k)]"})
    write("03_forestry_seasoning", "DESIGN 4.x / 5.1: store, felling, limbing & bucking, splitting, drying rack, tarp, 60-day cure, ground rot to punk wood.",
          dict(BASE, preset="Medium", render_stride=40, no_tutorial=True), S)


# ------------------------------------------------------------------------------------------
def ui_menus():
    S = []
    A = S.append
    A({"log": "UI: help overlay, Field Compendium (pin a blueprint -> hologram snapping), smartphone apps, time menu, pause settings"})
    A({"wait": 0.3})
    A({"key": "h"}); A({"wait": 0.2}); A({"screenshot": "01_help_overlay"})
    A({"assert": "input.show_help", "message": "H shows the controls overlay"}); A({"key": "h"})
    A({"key": "b"}); A({"wait": 0.3})
    A({"assert": "book.is_open and book.current_page == 0", "message": "B opens the Field Compendium"})
    A({"screenshot": "02_book_combustion"})
    A({"key": "d"}); A({"wait": 0.1}); A({"screenshot": "03_book_species"})
    A({"key": "d"}); A({"key": "d"}); A({"key": "d"}); A({"wait": 0.1})
    A({"assert": "book.get_current_page_data()['key'] == 'teepee'", "message": "paged forward to the Teepee blueprint"})
    A({"key": "return"}); A({"wait": 0.1})
    A({"assert": "book.selected_blueprint_key == 'teepee'", "message": "Enter pins the Teepee as a holographic guide"})
    A({"screenshot": "04_book_teepee_pinned"})
    A({"key": "d"}); A({"key": "d"}); A({"wait": 0.1})
    A({"assert": "book.get_current_page_data()['locked']", "message": "advanced blueprints start locked"})
    A({"key": "return"}); A({"wait": 0.1})
    A({"assert": "book.selected_blueprint_key == 'teepee'", "message": "a locked blueprint cannot be pinned"})
    A({"screenshot": "05_book_locked_lean_to"})
    for _ in range(5):
        A({"key": "d"})
    A({"wait": 0.1}); A({"screenshot": "06_book_seasoning_chart"})
    A({"key": "b"}); A({"wait": 0.2})
    A({"assert": "not book.is_open", "message": "B closes the book"})
    A({"walk_to": {"item": "Bundle of dead pine twigs"}, "stop": 1.3, "timeout": 12}); A({"aim_at": {"item": "Bundle of dead pine twigs"}}); A({"click": "left"}); A({"wait": 0.2})
    A({"walk_to": {"pit": [0, 0, 0]}, "stop": 1.3, "timeout": 12})
    A({"key": "7"}); A({"aim_at": {"pit": [0.12, 0.0, 0.02]}}); A({"wait": 0.2})
    A({"assert": "world.ghost is not None and world.ghost['slot'] is not None", "message": "the kindling ghost snaps to a blueprint slot", "value": "world.ghost and world.ghost['slot']"})
    A({"screenshot": "07_blueprint_hologram"})
    A({"click": "right"}); A({"wait": 0.2})
    A({"key": "tab"}); A({"wait": 0.3}); A({"screenshot": "08_phone_home"})
    A({"key": "right"}); A({"key": "return"}); A({"wait": 0.3})
    A({"assert": "phone.active_app.value == 'Topographical Map'", "message": "opened the Topo Map app"})
    A({"screenshot": "09_phone_topo_map"})
    A({"key": "w"}); A({"key": "w"}); A({"wait": 0.2})
    A({"assert": "phone.map_zoom > 1.4", "message": "W zooms the map in", "value": "phone.map_zoom"})
    A({"screenshot": "10_phone_topo_zoomed"})
    A({"key": "backspace"}); A({"key": "left"}); A({"key": "return"}); A({"wait": 0.2})
    A({"assert": "phone.active_app.value == 'Store ($)'", "message": "back home, then into the Store"})
    A({"exec": "scratch['cr'] = phone.credits.credits"})
    A({"key": "return"}); A({"wait": 0.2})
    A({"assert": "phone.credits.credits == scratch['cr'] and 'Insufficient' in phone.status_message", "message": "cannot buy a lighter refill with 20 credits", "value": "phone.status_message"})
    A({"expect_sound": "ui_error"})
    A({"screenshot": "11_phone_store"})
    A({"key": "tab"}); A({"wait": 0.2})
    A({"exec": "scratch['clock'] = weather.total_hours"})
    A({"key": "t"}); A({"key": "s"}); A({"key": "s"}); A({"wait": 0.2}); A({"screenshot": "12_time_menu"})
    A({"key": "escape"}); A({"wait": 0.2})
    A({"assert": "not world.time_engine.is_menu_open and weather.total_hours - scratch['clock'] < 0.01", "message": "Esc cancels the time menu without skipping"})
    A({"key": "escape"}); A({"wait": 0.2})
    A({"assert": "input.is_paused", "message": "Esc pauses the game"})
    A({"exec": "scratch['t_sim'] = world.sim_time"}); A({"wait": 0.5})
    A({"assert": "abs(world.sim_time - scratch['t_sim']) < 1e-6", "message": "the simulation is frozen while paused"})
    A({"key": "s"}); A({"key": "d"}); A({"key": "d"})
    A({"assert": "abs(input.settings.master_volume - 1.0) < 1e-6", "message": "D raises the master volume", "value": "input.settings.master_volume"})
    A({"key": "s"}); A({"key": "s"}); A({"key": "s"}); A({"key": "a"}); A({"wait": 0.2})
    A({"assert": "input.settings.graphics_preset == 'Medium'", "message": "A steps the graphics preset down"})
    A({"screenshot": "13_pause_settings"})
    A({"assert": "renderer.preset_name == 'Medium'", "message": "the renderer applied the new preset"})
    A({"key": "escape"}); A({"wait": 0.2})
    A({"assert": "not input.is_paused", "message": "Esc resumes"})
    write("04_ui_menus", "Help, Field Compendium, blueprint holograms, smartphone Home/Store/Topo Map, time menu, pause & settings.",
          dict(BASE, preset="High", render_stride=40, no_tutorial=True), S)


# ------------------------------------------------------------------------------------------
def atmosphere():
    S = []
    A = S.append
    A({"log": "Atmosphere showcase (staged): the same camp across the day, the weather and the seasons"})
    A({"exec": "from firestarter.automation.staging import build_log_fire; build_log_fire(world)  # staged fire lay"})
    A({"fast_forward": 90})
    A({"weather": {"clear": True}})
    shots = [
        ("01_golden_hour", {"hour": 18.8}, [2.6, 2.9], [0.0, 0.45, 0.0], {}),
        ("02_noon_clearing", {"hour": 12.5}, [4.8, 5.6], [-1.0, 1.2, -4.0], {}),
        ("03_morning_mist", {"hour": 6.1}, [3.0, 4.5], [-2.0, 1.2, -6.0], {"fog": 0.75}),
        ("04_dusk_sunset_sky", {"hour": 19.85}, [1.5, 1.5], [-30.0, 7.0, -14.0], {}),
        ("05_night_campfire", {"hour": 22.4}, [2.0, 2.3], [0.0, 0.4, 0.0], {}),
        ("06_night_stars", {"hour": 23.5}, [1.0, 2.0], [-5.0, 16.0, -20.0], {}),
    ]
    for name, t, pos, look, wx in shots:
        A({"set_time": t})
        if wx:
            A({"weather": wx})
        A({"teleport": pos, "look_at": look}); A({"wait": 0.5}); A({"screenshot": name})
        if "fog" in wx:
            A({"weather": {"fog": 0.0}})     # the mist burns off: do not carry it into later shots
        if name == "05_night_campfire":
            # two frames 0.2 s apart: the flames must visibly move while the camp holds still
            A({"screenshot": "05b_flicker_a"}); A({"wait": 0.2}); A({"screenshot": "05b_flicker_b"})
    A({"weather": {"fog": 0.0}})
    A({"set_time": {"hour": 21.6}}); A({"weather": {"rain": 0.85}}); A({"wait": 1.5})
    A({"teleport": [2.3, 2.6], "look_at": [0.0, 0.5, 0.0]}); A({"wait": 1.0}); A({"screenshot": "07_rain_at_night"})
    A({"assert": "weather.rain_intensity > 0.5", "message": "heavy rain falling"})
    A({"assert": "grid.is_burning", "message": "the established fire survives the rain"})
    A({"expect_sound": "step"}) if False else None
    A({"weather": {"rain": 0.0, "cloud": 0.3}})
    A({"set_time": {"hour": 15.0, "season": "Autumn", "day": 20}}); A({"wait": 0.5})
    A({"teleport": [5.0, 6.0], "look_at": [-3.0, 3.0, -8.0]}); A({"wait": 0.5}); A({"screenshot": "08_autumn_colours"})
    A({"set_time": {"hour": 11.5, "season": "Winter", "day": 30}}); A({"exec": "weather.snow_accumulation_m = 0.35"}); A({"weather": {"cloud": 0.6}}); A({"wait": 0.5})
    A({"teleport": [4.0, 5.0], "look_at": [-2.0, 1.0, -5.0]}); A({"wait": 0.5}); A({"screenshot": "09_winter_snow_day"})
    A({"set_time": {"hour": 19.0, "season": "Winter", "day": 30}}); A({"weather": {"rain": 0.7}}); A({"wait": 1.0})
    A({"assert": "weather.snow_intensity > 0.3 and weather.rain_intensity == 0", "message": "precipitation falls as snow below freezing"})
    A({"teleport": [2.2, 2.5], "look_at": [0.0, 0.5, 0.0]}); A({"wait": 1.0}); A({"screenshot": "10_winter_night_fire_snowfall"})
    A({"weather": {"rain": 0.0, "cloud": 0.2}})
    A({"set_time": {"hour": 10.0, "season": "Spring", "day": 15}}); A({"exec": "weather.snow_accumulation_m = 0.0"}); A({"wait": 0.5})
    A({"teleport": [-3.0, 8.0], "look_at": [2.0, 2.0, -6.0]}); A({"wait": 0.5}); A({"screenshot": "11_spring_morning"})
    A({"teleport": [0.0, -31.0], "look_at": [10.0, -2.5, -40.0]}); A({"wait": 0.5}); A({"screenshot": "12_creek"})
    S = [s for s in S if s]
    write("05_atmosphere_showcase", "Visual showcase: golden hour, noon, mist, dusk, night & stars, rain, autumn, winter snow, spring, creek.",
          dict(BASE, preset="High", render_stride=60, no_tutorial=True, no_audio=True), S)


# ------------------------------------------------------------------------------------------
def fire_science():
    S = []
    A = S.append
    A({"log": "Fire science: damp tinder, choked layout, green log, log cabin via blueprint snapping, gasoline flash-over"})
    A({"exec": "from firestarter.simulation.inventory import make_tinder; inv.add_fuel(make_tinder('dry_grass', 0.45)) # staged: a soaked bundle"})
    A({"teleport": [0.0, 1.1], "look_at": {"pit": [0, 0, 0]}}); A({"key": "6"}); A({"aim_at": {"pit": [0, 0, 0]}}); A({"click": "right"}); A({"wait": 0.2})
    A({"key": "left ctrl"}); A({"key": "1"}); A({"aim_at": {"pit": [0, 0.03, 0]}})
    A({"mouse_down": "left"}); A({"wait": 6.0}); A({"mouse_up": "left"}); A({"key": "left ctrl"})
    A({"assert": "grid.flame_hrr < 50", "message": "damp tinder (45 % MC) only steams under a lighter - no flame", "value": "round(grid.flame_hrr)"})
    A({"assert": "grid.total_water_evaporated_kg > 0 or any(b.m_surface < 0.45 for b in grid.bodies)", "message": "the lighter boiled some water out of the damp bundle"})
    A({"exec": "grid.clear()"})
    # open vs choked packing (same fuel, different geometry)
    A({"exec": "from firestarter.automation.staging import build_teepee, light; build_teepee(world, 6, 4); light(world)"})
    A({"fast_forward": 40}); A({"exec": "scratch['open_hrr'] = grid.current_heat_flux; scratch['open_o2'] = grid.oxygen_fraction"})
    A({"exec": "grid.clear()"})
    A({"exec": "from firestarter.automation.staging import build_choked; build_choked(world); light(world)"})
    A({"teleport": [0.3, 1.6], "look_at": [0.0, 0.1, 0.0]}); A({"wait": 4.0})
    A({"assert": "grid.oxygen_fraction < 0.16", "message": "the sealed pack starves the tinder of oxygen", "value": "round(grid.oxygen_fraction, 3)"})
    A({"screenshot": "01_choked_smouldering"})
    A({"fast_forward": 36}); A({"wait": 0.5})
    A({"assert": "grid.analysis.choke_fraction > 0.6", "message": "edge-to-edge logs (< 2 cm gaps) detected as a choked layout", "value": "round(grid.analysis.choke_fraction, 2)"})
    A({"assert": "grid.c_d < 0.2", "message": "discharge coefficient collapses (C_d -> 0)", "value": "round(grid.c_d, 3)"})
    A({"assert": "grid.current_heat_flux < 0.5 * scratch['open_hrr']", "message": "the choked pack releases far less heat than an open teepee of the same fuel", "value": "(round(scratch['open_hrr']), round(grid.current_heat_flux))"})
    A({"assert": "world.structure_key == 'freeform'", "message": "a sealed pack is not mistaken for a Lean-To or Swedish Torch", "value": "(world.structure_key, round(world.structure_score, 2))"})
    A({"exec": "grid.clear()"})
    # green log on a good fire
    A({"exec": "from firestarter.automation.staging import build_log_fire; build_log_fire(world, oak_logs=0)"})
    A({"fast_forward": 60})
    A({"exec": "from firestarter.automation.staging import add_green_log; scratch['green'] = add_green_log(world)"})
    A({"teleport": [1.6, 1.9], "look_at": [0.0, 0.3, 0.0]})
    A({"fast_forward": 45}); A({"wait": 1.0})
    A({"assert": "scratch['green'].steam_rate > 2e-4 and not scratch['green'].is_flaming", "message": "a green log on the fire hisses steam but will not flame", "value": "(round(scratch['green'].steam_rate * 1000, 3), scratch['green'].stage.value)"})
    A({"screenshot": "02_green_log_steaming"})
    A({"exec": "grid.clear()"})
    # log cabin with the pinned blueprint
    A({"key": "b"}); A({"key": "d"}); A({"key": "d"}); A({"key": "d"}); A({"key": "d"}); A({"key": "d"}); A({"key": "return"}); A({"key": "b"})
    A({"assert": "book.selected_blueprint_key == 'log_cabin'", "message": "pinned the Log Cabin blueprint"})
    A({"exec": "from firestarter.automation.staging import stock_for_log_cabin; stock_for_log_cabin(world)  # staged: an armful of fuel"})
    A({"teleport": [0.0, 1.15], "look_at": {"pit": [0, 0, 0]}})
    A({"key": "6"}); A({"aim_at": {"pit": [0, 0, 0]}}); A({"click": "right"}); A({"wait": 0.1})
    for _ in range(4):
        A({"key": "7"}); A({"aim_at": {"pit": [0.05, 0, 0.0]}}); A({"click": "right"}); A({"wait": 0.1})
    for _ in range(8):
        A({"key": "8"}); A({"aim_at": {"pit": [0.0, 0, 0.15]}}); A({"click": "right"}); A({"wait": 0.1})
    for _ in range(3):   # kindling 'roof' laid across the top tier
        A({"key": "7"}); A({"aim_at": {"pit": [0.0, 0.2, 0.0]}}); A({"click": "right"}); A({"wait": 0.1})
    A({"wait": 0.6})
    A({"assert": "world.structure_key == 'log_cabin'", "message": "the pieces snapped into a recognised Log Cabin", "value": "(world.structure_key, round(world.structure_score, 2))"})
    A({"teleport": [1.3, 1.5], "look_at": [0.0, 0.15, 0.0]}); A({"wait": 0.3}); A({"screenshot": "03_log_cabin_built"})
    A({"key": "left ctrl"}); A({"key": "1"}); A({"teleport": [0.0, 0.95], "look_at": {"pit": [0, 0, 0]}}); A({"aim_at": {"pit": [0, 0.03, 0]}})
    A({"mouse_down": "left"}); A({"wait": 2.5}); A({"mouse_up": "left"}); A({"key": "left ctrl"})
    A({"wait_until": "grid.flame_hrr > 800", "timeout": 10, "message": "the log cabin lights"})
    A({"fast_forward": 90})
    A({"assert": "grid.c_d > 0.6 and grid.draft_velocity > 1.0", "message": "an open cabin draws a strong draft", "value": "(round(grid.c_d, 2), round(grid.draft_velocity, 2))"})
    A({"assert": "sum(1 for b in grid.bodies if b.geometry_type == 'firewood' and b.is_flaming) >= 3", "message": "the roofed cabin walls catch from the kindling", "value": "sum(1 for b in grid.bodies if b.geometry_type == 'firewood' and b.is_flaming)"})
    A({"teleport": [1.6, 1.8], "look_at": [0.0, 0.35, 0.0]}); A({"wait": 0.5}); A({"screenshot": "04_log_cabin_burning"})
    # gasoline flash-over
    A({"exec": "inv.gasoline_kg = 2.8  # staged: a can from the store"})
    A({"key": "5"}); A({"aim_at": {"pit": [0, 0.1, 0]}}); A({"click": "left"}); A({"wait": 0.35})
    A({"assert": "world.gas_flash_timer > 0 and 'WHOOMPH' in world.notification", "message": "gasoline poured on a lit fire flashes into a fireball", "value": "world.notification[:40]"})
    A({"screenshot": "05_gasoline_fireball"})
    A({"expect_sound": "gas_whoosh"})
    write("06_fire_science", "DESIGN 2.x physics: moisture, choked layouts, green wood, structure draft, accelerant danger.",
          dict(BASE, preset="Medium", render_stride=40, no_tutorial=True, start_hour=20.6), S)


# ------------------------------------------------------------------------------------------
def controls():
    S = []
    A = S.append
    A({"log": "Movement & controls: WASD, sprint, crouch, collisions, mouse look, placement rotation/tilt, off-hand cycling"})
    A({"wait": 0.3})
    A({"mouse_move": [600, 0], "duration": 0.3}); A({"wait": 0.1})
    A({"exec": "scratch['p0'] = player.position.copy(); scratch['yaw0'] = player.yaw"})
    A({"key": "w", "hold": 1.5}); A({"wait": 0.2})
    A({"assert": "np.linalg.norm(player.position[[0,2]] - scratch['p0'][[0,2]]) > 2.5", "message": "W walks forward", "value": "round(float(np.linalg.norm(player.position[[0,2]] - scratch['p0'][[0,2]])), 2)"})
    A({"assert": "np.linalg.norm(player.position[[0,2]] - world.fire_pit_pos[[0,2]]) > 0.7", "message": "the stone ring blocks walking into the fire"})
    A({"key": "s", "hold": 1.0}); A({"key": "a", "hold": 0.8}); A({"key": "d", "hold": 0.8})
    A({"exec": "scratch['st'] = player.stamina"})
    A({"key_down": "left shift"}); A({"key_down": "w"}); A({"wait": 2.0}); A({"key_up": "w"}); A({"key_up": "left shift"})
    A({"assert": "player.stamina < scratch['st'] - 15", "message": "sprinting drains stamina", "value": "round(player.stamina)"})
    A({"key": "left ctrl"}); A({"wait": 0.6})
    A({"assert": "player.eye_height < 1.2 and player.is_crouching", "message": "Ctrl toggles crouching on (no need to hold it)", "value": "round(player.eye_height, 2)"})
    A({"key": "left ctrl"}); A({"wait": 0.6})
    A({"assert": "player.eye_height > 1.5 and not player.is_crouching", "message": "pressing Ctrl again stands up"})
    A({"mouse_move": [300, 0], "duration": 0.3}); A({"wait": 0.1})
    A({"assert": "abs(((player.yaw - scratch['yaw0'] + 180) % 360) - 180) > 20", "message": "mouse look turns the view", "value": "round(player.yaw, 1)"})
    A({"teleport": [9.5, -9.0], "look_at": [7.4, 1.0, -8.6]})
    A({"key": "w", "hold": 2.0}); A({"wait": 0.2})
    A({"assert": "np.linalg.norm(player.position[[0,2]] - np.array([7.4, -8.6])) > 0.3", "message": "tree trunks block the player"})
    A({"walk_to": {"item": "Bundle of dead pine twigs"}, "stop": 1.3, "timeout": 14}); A({"aim_at": {"item": "Bundle of dead pine twigs"}}); A({"click": "left"})
    A({"walk_to": {"pit": [0, 0, 0]}, "stop": 1.3, "timeout": 12})
    A({"key": "7"}); A({"aim_at": {"pit": [0.15, 0, 0.0]}}); A({"wait": 0.1})
    A({"exec": "scratch['g0'] = world.ghost['orientation'].copy()"})
    A({"assert": "scratch['g0'][0] > 40", "message": "kindling defaults to a teepee lean"})
    A({"key_down": "left shift"}); A({"click": "wheel_down"}); A({"click": "wheel_down"}); A({"click": "wheel_down"}); A({"click": "wheel_down"}); A({"key_up": "left shift"}); A({"wait": 0.1})
    A({"assert": "world.ghost['orientation'][0] < 5", "message": "Shift+Wheel tilts the piece flat", "value": "world.ghost['orientation'][0]"})
    A({"exec": "scratch['y0'] = world.ghost['orientation'][1]"})
    A({"click": "wheel_up"}); A({"click": "wheel_up"}); A({"wait": 0.1})
    A({"assert": "abs(((world.ghost['orientation'][1] - scratch['y0'] + 180) % 360) - 180 - 30) < 1", "message": "Wheel rotates the piece 15 degrees per notch"})
    A({"click": "middle"}); A({"wait": 0.1})
    A({"assert": "player.held_item_tilt is None and player.held_item_rotation == 0", "message": "MMB resets placement"})
    A({"exec": "scratch['off'] = world.current_offhand()"}); A({"key": "e"})
    A({"assert": "world.current_offhand() != scratch['off']", "message": "E cycles the off-hand item"}); A({"key": "q"})
    A({"assert": "world.current_offhand() == scratch['off']", "message": "Q cycles back"})
    A({"key": "0"}); A({"assert": "world.current_offhand() == 'fungus'", "message": "0 selects the fungus slot"})
    A({"key": "3"}); A({"assert": "player.primary_hand == 'felling_axe'", "message": "3 equips the axe"})
    A({"screenshot": "01_axe_in_hand"})
    A({"key": "2"}); A({"screenshot": "02_knife_in_hand"})
    A({"key": "x"}); A({"wait": 0.4})
    A({"assert": "player.primary_hand == 'none' and world.current_offhand() == 'none'", "message": "X empties both hands"})
    A({"screenshot": "02b_empty_hands"})
    A({"key": "e"}); A({"assert": "world.current_offhand() == 'tinder'", "message": "E from an empty hand picks up the first off-hand slot"})
    A({"key": "2"})
    # the hand drill: carve a notch, press 4 twice, spin the spindle between the palms
    A({"walk_to": {"hearthboard": True}, "stop": 0.85, "timeout": 12})
    A({"key": "left ctrl"}); A({"wait": 0.4}); A({"aim_at": {"hearthboard": True}}); A({"click": "left"}); A({"wait": 0.2})
    A({"key": "4"}); A({"key": "4"}); A({"wait": 0.1})
    A({"assert": "world.friction_mode == 'hand'", "message": "pressing 4 again swaps the bow for the hand drill"})
    A({"aim_at": {"hearthboard": True}}); A({"mouse_down": "left"}); A({"mouse_down": "right"})
    A({"saw": {"duration": 9, "amplitude": 170, "rate": 1.8}})
    A({"assert": "world.crafting.bow_drill.socket_temp > 70", "message": "palm-spun spindle heats the socket", "value": "round(world.crafting.bow_drill.socket_temp)"})
    A({"screenshot": "03_hand_drilling"})
    A({"mouse_up": "left"}); A({"mouse_up": "right"}); A({"key": "left ctrl"})
    write("07_movement_controls", "README controls table: movement, sprint, crouch, collision, look, placement rotation / tilt, tool and off-hand slots.",
          dict(BASE, preset="Medium", render_stride=60, no_tutorial=True, no_audio=True), S)


if __name__ == "__main__":
    tutorial()
    bow_drill()
    forestry()
    ui_menus()
    atmosphere()
    fire_science()
    controls()
