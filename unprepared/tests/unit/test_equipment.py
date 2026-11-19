"""Equipment: slots, the catalogue, the townsfolk's outfits as worn items, the player's [Tab] view."""
import pytest

from tests.helpers import spot
from unprepared.core.math3d import Vector3D
from unprepared.entities import AIState, Item, Player
from unprepared.entities.equipment import (EquipSlot, GEAR_CATALOGUE, GEAR_SLOTS, SLOT_ORDER, equipment_from_outfit,
                                           gear_for, slot_of)
from unprepared.entities.inventory import take_item
from unprepared.entities.outfit import GUARD_UNIFORM, outfit_for
from unprepared.game import GameMode
from unprepared.systems import MoveIntent
from unprepared.world.levels.town import build_town_level


def gear(item_id, name="Thing", **kw):
    spec = gear_for(item_id)
    return Item(item_id, name, gear=spec.gear, gear_color=spec.color, material=spec.material, **kw)


# --- the catalogue and the models --------------------------------------------------------------------
class TestCatalogue:
    def test_every_catalogue_entry_has_a_slot_and_a_model(self):
        from unprepared.rendering.gear import GEAR_DRAWERS
        for item_id, spec in GEAR_CATALOGUE.items():
            assert spec.gear in GEAR_SLOTS, item_id
            assert spec.gear in GEAR_DRAWERS, item_id
            assert spec.slot in SLOT_ORDER
        assert set(GEAR_DRAWERS) == set(GEAR_SLOTS)                      # every model mounts somewhere, and vice versa

    def test_slots_by_kind(self):
        assert slot_of(gear("iron_helm")) is EquipSlot.HEAD
        assert slot_of(gear("ruby_necklace")) is EquipSlot.NECK and slot_of(gear("wool_cloak")) is EquipSlot.NECK
        assert slot_of(gear("chain_shirt")) is EquipSlot.BODY
        assert slot_of(gear("shadow_boots")) is EquipSlot.FEET
        assert slot_of(gear("short_sword")) is EquipSlot.WEAPON and slot_of(gear("sling")) is EquipSlot.WEAPON
        assert slot_of(Item("candlestick", "Silver Candlestick")) is None              # not equipment
        assert slot_of(Item("needles", "Needles", stackable=True, gear="dagger")) is None   # a stack is never worn

    def test_the_levels_items_pick_up_their_gear_from_the_catalogue(self):
        level = build_town_level()
        built = {}
        for c in level.containers:
            for d in c.items:
                built[d.id] = d.build(owner=c.owner)
        for npc in level.npcs:
            for tier in npc.restock_tiers:
                for d in tier.items:
                    built[d.id] = d.build()
        for d in level.player_inventory:
            built[d.id] = d.build()
        for item_id in ("iron_dagger", "short_sword", "leather_cap", "iron_helm", "chain_shirt", "wool_cloak",
                        "soft_slippers", "shadow_boots", "velvet_cloak", "steel_sword", "plate_cuirass", "jeweled_dagger",
                        "ruby_necklace", "soft_shoes"):
            assert built[item_id].is_equipment, item_id
            assert built[item_id].gear == GEAR_CATALOGUE[item_id].gear
        assert built["soft_shoes"].equipped                                  # the player starts in their boots
        assert not built["iron_helm"].equipped
        assert not built["candlestick"].is_equipment and not built["silk_gloves"].is_equipment   # no hands on the figure


# --- the townsfolk wear real equipment ---------------------------------------------------------------
class TestOutfitsAsEquipment:
    def test_a_guards_uniform(self):
        worn = equipment_from_outfit(GUARD_UNIFORM, owner="Guard Orrin")
        assert set(worn) == {EquipSlot.BODY, EquipSlot.HEAD}
        assert worn[EquipSlot.BODY].gear == "breastplate" and worn[EquipSlot.HEAD].gear == "helmet"
        assert all(item.equipped and not item.for_sale and item.owner == "Guard Orrin" for item in worn.values())

    @pytest.mark.parametrize("name,role,expected", [
        ("Hilde the Innkeeper", "innkeeper", {"body": "tunic", "neck": "apron_cloth", "head": "coif"}),
        ("Gareth the Smith", "smith", {"body": "tunic", "neck": "apron_leather"}),
        ("Shopkeeper Barnaby", "shopkeeper", {"body": "tunic", "neck": "sash", "head": "cap"}),
        ("Jae the Fence", "fence", {"body": "tunic", "head": "hood"}),
        ("Tamsin", "widow", {"body": "tunic", "head": "scarf"}),
        ("Pell", "lad", {"body": "tunic"}),
    ])
    def test_each_role_is_dressed_as_drawn(self, name, role, expected):
        outfit = outfit_for(name, role)
        worn = equipment_from_outfit(outfit)
        assert {slot.value: item.gear for slot, item in worn.items()} == expected
        if EquipSlot.BODY in worn and not outfit.armor:
            assert worn[EquipSlot.BODY].gear_color == outfit.tunic

    def test_npcs_carry_their_outfit_as_equipment_but_not_in_their_pockets(self, game):
        vance = game.world.find_npc("Guard Captain Vance")
        assert vance.equipment[EquipSlot.HEAD].gear == "helmet"
        assert not any(i.gear for i in vance.inventory)                     # a helmet on a head is not loot
        assert vance.figure.legs == vance.clothes.legs and vance.figure.hair == vance.clothes.hair


# --- the player ----------------------------------------------------------------------------------------
class TestPlayerEquipment:
    def test_one_item_per_slot(self):
        p = Player(Vector3D(0, 0, 0), inventory=[gear("iron_helm", "Iron Helm"), gear("leather_cap", "Leather Cap")])
        helm, cap = p.inventory
        assert p.equip(helm) and p.equipment == {EquipSlot.HEAD: helm}
        assert p.equip(cap) and p.equipment == {EquipSlot.HEAD: cap} and not helm.equipped
        assert p.toggle_equipped(cap) is False and p.equipment == {}
        assert p.toggle_equipped(helm) is True and p.equipped_in(EquipSlot.HEAD) is helm
        assert p.toggle_equipped(Item("plate", "Pewter Plate")) is None
        assert not p.equip(gear("short_sword"))                               # not in the inventory

    def test_a_second_worn_item_for_a_slot_is_unworn_on_sight(self):
        helm, cap = gear("iron_helm", equipped=True), gear("leather_cap", equipped=True)
        p = Player(Vector3D(0, 0, 0), inventory=[helm, cap])
        assert p.equipment == {EquipSlot.HEAD: helm} and not cap.equipped

    def test_what_leaves_the_inventory_is_no_longer_worn(self):
        helm = gear("iron_helm", equipped=True)
        p = Player(Vector3D(0, 0, 0), inventory=[helm])
        assert p.equipment
        take_item(p.inventory, helm)
        assert not helm.equipped and p.equipment == {}

    def test_footsteps_count_only_what_is_worn(self):
        shoes = gear("black_shoes", "Black Cloth Shoes")                       # cloth: silent
        mail = gear("chain_shirt", "Chain Shirt")                              # steel: clanks
        gloves = Item("silk_gloves", "Silk Gloves", noise_modifier=-0.1)       # no slot: worn whenever carried
        p = Player(Vector3D(0, 0, 0), inventory=[shoes, mail, gloves])
        assert p.clothing_noise_factor() == pytest.approx(0.9)               # only the gloves count
        p.equip(shoes)
        quiet = p.clothing_noise_factor()
        assert quiet < 0.9
        p.equip(mail)
        assert p.clothing_noise_factor() > quiet
        p.unequip(shoes)
        assert p.clothing_noise_factor() > 1.0

    def test_worn_state_survives_a_save(self, game):
        helm = game.player.add_item(gear("iron_helm", "Iron Helm", owner="player"))
        game.player.equip(helm)
        game.update(0.016)
        assert game.save_game()
        helm.equipped = False
        assert game.load_game()
        worn = game.player.equipment
        assert worn[EquipSlot.FEET].id == "soft_shoes" and worn[EquipSlot.HEAD].id == "iron_helm"

    def test_a_save_from_before_equipment_puts_the_kit_on(self, game, tmp_path):
        import json
        game.player.add_item(gear("wool_cloak", "Wool Cloak", owner="player"))
        data = game.saves.serialize(game.world, game.player)
        data["version"] = 7
        for item in data["player"]["inventory"]:
            item.pop("gear"); item.pop("gear_color"); item.pop("equipped")  # as a v7 save would have it
        path = tmp_path / "old.json"
        path.write_text(json.dumps(data))
        assert game.load_game(str(path))
        worn = {slot.value: item.id for slot, item in game.player.equipment.items()}
        assert worn == {"feet": "soft_shoes", "neck": "wool_cloak"}          # everything carried counted as worn before


# --- the [Tab] view ------------------------------------------------------------------------------------
class TestEquipmentView:
    def test_tab_opens_and_closes_the_view(self, game):
        game.update(0.016)
        assert game.mode is GameMode.EXPLORE and game.mouse_captured
        game.tab()
        assert game.mode is GameMode.EQUIPMENT and game.in_equipment and not game.mouse_captured
        assert not game.accepts_free_look
        before = game.player.position.copy()
        game.update(0.1, MoveIntent(forward=True))
        assert game.player.position == before                                 # no walking about while dressing
        game.tab()
        assert game.mode is GameMode.EXPLORE and game.mouse_captured
        game.tab()
        assert game.escape() and game.mode is GameMode.EXPLORE               # ESC closes it too

    def test_the_world_stands_still_while_the_view_is_open(self, game):
        """[Tab] pauses the game: no time passes, nobody moves, nobody perceives anything."""
        w = game.world
        game.update(0.016)
        elapsed, minutes = w.elapsed, w.clock.minute_accumulator
        vance = w.find_npc("Guard Captain Vance")
        where = vance.position.copy()
        game.player.is_crouching = True
        game.player.position = spot(game.world, vance.position.x, vance.position.z + 1.5)     # skulking in his face
        vance.turn_towards(game.player.position)
        game.tab()
        for _ in range(120):
            game.update(0.1)
        assert w.elapsed == elapsed and w.clock.minute_accumulator == minutes
        assert vance.position == where and vance.suspicion_meter == 0.0
        assert game.preview_yaw > 0.0                                         # only the stand turns
        game.tab()
        game.update(0.1)
        assert w.elapsed > elapsed                                            # and the world goes on

    def test_not_over_another_screen(self, game):
        jae = game.world.find_npc("Jae the Fence")
        jae.state = AIState.SHOPKEEPING
        game.open_trade(jae)
        side = game.transfer.side
        game.tab()                                                            # in a transfer: switches the column
        assert game.transfer is not None and game.transfer.side is not side and not game.in_equipment
        game.close_transfer()
        game.player.freeze_timer = 1.0
        assert not game.open_equipment()                                      # not while the controls are locked
        game.player.freeze_timer = 0.0
        game.ask("Sure?", on_yes=lambda: None)
        assert not game.open_equipment()

    def test_cursor_wraps_and_enter_toggles(self, game):
        helm = game.player.add_item(gear("iron_helm", "Iron Helm", owner="player"))
        game.update(0.016)
        game.tab()
        n = len(game.player.inventory)
        assert game.equipment_cursor == 0
        game.cursor_up()
        assert game.equipment_cursor == n - 1 and game.equipment_selection is helm
        game.transfer_confirm()                                               # Enter: put it on
        assert helm.equipped and "You put on Iron Helm." in game.messages.messages
        game.transfer_confirm()
        assert not helm.equipped and "You take off Iron Helm." in game.messages.messages
        game.cursor_down()
        assert game.equipment_cursor == 0                                     # wrapped round
        game.transfer_confirm()                                               # the lockpick: not equipment
        assert any("cannot wear or wield" in m for m in game.messages.messages)

    def test_left_and_right_turn_the_figure(self, game):
        game.tab()
        assert game.preview_spin
        game.update(0.5)
        spun = game.preview_yaw
        assert spun > 0.0
        game.lockpick_rotate(+1)
        assert game.preview_yaw == pytest.approx((spun + 15.0) % 360.0) and not game.preview_spin
        game.update(0.5)
        assert game.preview_yaw == pytest.approx((spun + 15.0) % 360.0)       # no longer turning by itself

    def test_scrolling_keeps_the_cursor_in_view(self, game):
        for i in range(30):
            game.player.add_item(Item(f"plate{i}", f"Plate {i}"))
        game.tab()
        first, rows = game.equipment_visible(10)
        assert first == 0 and len(rows) == 10
        for _ in range(25):
            game.cursor_down()
        first, rows = game.equipment_visible(10)
        assert first <= game.equipment_cursor < first + 10 and len(rows) == 10
        game.player.inventory = game.player.inventory[:3]                     # the bag shrank under the cursor
        assert game.equipment_selection is game.player.inventory[2]

    def test_being_caught_closes_the_view(self, game):
        game.tab()
        game.world.player_caught(game.world.find_npc("Guard Orrin"))
        assert not game.in_equipment

    def test_snapshot_reports_what_is_worn(self, game):
        snap = game.snapshot()
        assert snap["equipment"] == {"open": False, "cursor": 0, "worn": {"feet": "soft_shoes"}}


# --- the clothing model: what you wear decides how you are seen and heard -----------------------------
from unprepared.config import ClothingSettings                                                    # noqa: E402
from unprepared.entities.equipment import clothing_effect, describe_clothing, luminance, material_of   # noqa: E402
from unprepared.systems.stealth import sneak_chance                                                # noqa: E402
from tests.conftest import said                                                                     # noqa: E402

BLACK_CLOTH = ("black_hood", "black_cloak", "black_tunic", "black_shoes")
BRIGHT_STEEL = ("iron_helm", "white_cloak", "plate_cuirass", "steel_sword")


def dressed(ids, **kw):
    p = Player(Vector3D(0, 0, 0), inventory=[gear(i, i) for i in ids], **kw)
    for item in list(p.inventory):
        p.equip(item)
    return p


class TestClothingModel:
    def test_materials_and_shades(self):
        assert material_of(gear("iron_helm")) == "metal" and material_of(gear("leather_cap")) == "leather"
        assert material_of(gear("black_shoes")) == "cloth"                  # cloth shoes on the ``shoes`` model
        assert material_of(gear("soft_shoes")) == "leather"
        assert luminance((1.0, 1.0, 1.0)) == pytest.approx(1.0) and luminance((0.0, 0.0, 0.0)) == 0.0
        assert luminance(gear("white_cloak").gear_color) > luminance(gear("wool_cloak").gear_color) > luminance(gear("black_cloak").gear_color)

    def test_each_piece_pulls_the_way_it_should(self):
        c = ClothingSettings()
        steel, black, brown, white = (clothing_effect(gear(i), c) for i in ("plate_cuirass", "black_tunic", "leather_jerkin", "linen_shirt"))
        assert steel.visibility > white.visibility > 0 > brown.visibility > black.visibility     # glint, bright, dull, dark
        assert steel.noise > 0 > brown.noise > black.noise                                       # clank, soft, silent
        assert white.noise == black.noise                                                        # colour is seen, not heard
        shoes, helm = clothing_effect(gear("black_shoes"), c), clothing_effect(gear("black_hood"), c)
        assert shoes.noise < helm.noise < 0                                                      # shoes are what is heard
        assert clothing_effect(Item("plate", "Pewter Plate"), c) is None

    def test_head_to_toe_black_cloth_versus_steel_and_a_white_cloak(self):
        shadow, tin = dressed(BLACK_CLOTH), dressed(BRIGHT_STEEL)
        plain = Player(Vector3D(0, 0, 0))
        assert shadow.visibility_factor() < 0.45 < 1.0 == plain.visibility_factor() < 1.8 < tin.visibility_factor()
        assert shadow.clothing_noise_factor() < 0.75 < 1.0 == plain.clothing_noise_factor() < 1.3 < tin.clothing_noise_factor()
        c = ClothingSettings()
        assert c.visibility_floor <= shadow.visibility_factor() and tin.visibility_factor() <= c.visibility_ceiling

    def test_steel_glints_in_the_dark_black_cloth_vanishes(self):
        """At night (ambient 0.1) plain clothes are below the notice threshold; polished steel and a
        white cloak are not - and black cloth is below it even under an NPC's torch, crouched."""
        from unprepared.config import PerceptionSettings
        notice = PerceptionSettings().min_visibility_to_notice
        shadow, tin, plain = dressed(BLACK_CLOTH), dressed(BRIGHT_STEEL), Player(Vector3D(0, 0, 0))
        assert plain.get_visibility(0.1) < notice < tin.get_visibility(0.1)
        assert shadow.get_visibility(0.1) < notice
        for p in (shadow, tin, plain):
            p.is_crouching = True
        assert shadow.get_visibility(0.5, is_moving=False) < notice          # torch-lit, crouched, holding still: nothing
        assert shadow.get_visibility(1.0, is_moving=False) < notice          # even at noon, crouched and still
        assert shadow.get_visibility(1.0, is_moving=True) > 0                # moving by day: a shadow, but a visible one
        assert plain.get_visibility(1.0, is_moving=False) == pytest.approx(1.0 * 0.4 * 0.5)

    def test_crouching_still_next_to_a_guard_at_noon_in_black_cloth_goes_unnoticed(self, game):
        w = game.world
        orrin = w.find_npc("Guard Orrin")
        orrin.path, orrin.state, orrin.schedule = [], AIState.GUARDING, []
        game.player.inventory = [gear(i, i, owner="player") for i in BLACK_CLOTH]
        for item in list(game.player.inventory):
            game.player.equip(item)
        game.player.is_crouching = True
        game.player.position = spot(game.world, orrin.position.x, orrin.position.z + 1.5)
        orrin.turn_towards(game.player.position)
        w.clock.hour = 12
        for _ in range(100):
            w.update_perception(game.player, False, False, 0.1)              # ten seconds, holding still
        assert orrin.suspicion_meter == 0.0 and not orrin.is_alarmed
        # the same in a white linen shirt: a crouching figure a pace away is noticed within seconds
        game.player.inventory = [gear("linen_shirt", "White Linen Shirt", owner="player")]
        game.player.equip(game.player.inventory[0])
        for _ in range(100):
            w.update_perception(game.player, False, False, 0.1)
            if orrin.suspicion_meter > 0:
                break
        assert orrin.suspicion_meter > 0.0

    def test_black_cloth_improves_every_sneak_roll(self):
        from unprepared.config import StealthSettings
        from unprepared.entities import NPC
        guard = NPC("G", True, False, False, Vector3D(0, 0, 0))
        s = StealthSettings()
        shadow, tin = dressed(BLACK_CLOTH), dressed(BRIGHT_STEEL)
        for p in (shadow, tin):
            p.is_crouching = True
        assert sneak_chance(shadow, guard, s, shadow.get_visibility(1.0)) > sneak_chance(tin, guard, s, tin.get_visibility(1.0))

    def test_steel_and_a_white_cloak_at_a_run_wake_the_street_and_draw_remarks(self, game):
        """Sprinting past the guard house in plate at night is no crime with a torch lit - but the
        off-duty watch wake up through the wall and the guard on the post tells you what he
        thinks of it."""
        w = game.world
        w.clock.hour = 23
        w.resync_schedules(announce=False, teleport=True)
        game.player.inventory = [gear(i, i, owner="player") for i in BRIGHT_STEEL]
        for item in list(game.player.inventory):
            game.player.equip(item)
        game.player.has_lit_torch = True
        from tests.helpers import spot
        outside = spot(w, 44.5, -42.4)                                        # right outside the guard house's south wall
        sleepers = [n for n in w.npcs if n.is_asleep and w.area_of(n.position) == "the guard house"
                    and n.position.distance_to(outside) < 8.0]                # (the bunks along the near wall)
        assert sleepers
        mira = w.find_npc("Guard Mira")                                       # on the night round, awake
        mira.path, mira.state, mira.schedule = [], AIState.GUARDING, []
        mira.position = spot(w, 44.5, -40.0)
        game.player.position = outside
        assert game.player.get_noise_level(True, True) >= game.settings.perception.racket_noise
        for _ in range(30):
            w.update_perception(game.player, True, True, 0.1)                 # three seconds of sprinting on the spot
        assert all(not n.is_asleep for n in sleepers)                         # woken through the wall
        assert said(game.messages.messages, "woke_up")
        assert said(game.messages.messages, "racket_guard")
        assert not any(m.crime == "Skulking about" for m in mira.crime_memories)   # loud, not a crime

    def test_plain_clothes_at_a_run_wake_nobody_through_a_wall(self, game):
        w = game.world
        w.clock.hour = 23
        w.resync_schedules(announce=False, teleport=True)
        game.player.has_lit_torch = True
        sleepers = [n for n in w.npcs if n.is_asleep and w.area_of(n.position) == "the guard house"]
        from tests.helpers import spot
        game.player.position = spot(w, 44.5, -42.4)
        for _ in range(30):
            w.update_perception(game.player, True, True, 0.1)
        assert all(n.is_asleep for n in sleepers)
        assert not said(game.messages.messages, "racket_guard") and not said(game.messages.messages, "racket_citizen")

    def test_descriptions_for_the_tab_view(self):
        c = ClothingSettings()
        assert describe_clothing(gear("black_shoes"), c) == "black cloth - hard to see, silent"
        assert describe_clothing(gear("plate_cuirass"), c) == "bright metal - glints, clanks"
        assert describe_clothing(gear("leather_jerkin"), c).startswith("dark leather") or describe_clothing(gear("leather_jerkin"), c).startswith("drab leather")
        assert describe_clothing(Item("plate", "Pewter Plate"), c) == ""

    def test_the_economy_prices_stealth_not_steel(self):
        """Bright cloth for pennies, steel cheap, dark cloth dear - and the rarer, the dearer: the
        legendary pieces are worth more than anything else in the realm. None of the dark stuff is
        ever stock on a rack: it is stolen (see ``test_world.test_the_tailor_sells_no_dark_cloth``)."""
        from unprepared.entities.equipment import RARITY_ORDER
        level = build_town_level()
        prices, rarity = {}, {}
        defs = [d for c in level.containers for d in c.items] + [d for npc in level.npcs for d in npc.inventory]
        defs += [d for npc in level.npcs for tier in npc.restock_tiers for d in tier.items]
        for d in defs:
            prices[d.id] = max(prices.get(d.id, 0.0), d.value)
            rarity[d.id] = d.build().rarity
        steel = max(prices[i] for i in ("iron_helm", "chain_shirt", "plate_cuirass", "short_sword", "steel_sword"))
        bright = max(prices[i] for i in ("linen_shirt", "white_cloak", "red_cap"))
        dark = min(prices[i] for i in ("black_hood", "black_tunic", "black_cloak", "black_shoes", "velvet_cloak"))
        assert bright < 10 < steel < 100 < dark                              # bright for pennies, steel cheap, dark dear
        legendary = [i for i, r in rarity.items() if r == "legendary"]
        assert sorted(legendary) == ["amulet_open_door", "fingers_of_the_magpie", "nightweave_cloak"]
        assert min(prices[i] for i in legendary) == sorted(prices.values())[-len(legendary)]     # the three dearest things
        by_rank = {}
        for i, r in rarity.items():
            if r in ("rare", "epic", "legendary"):
                by_rank.setdefault(RARITY_ORDER.index(r), []).append(prices[i])
        assert max(by_rank[2]) < min(by_rank[4]) and max(by_rank[3]) < min(by_rank[4])
        tiers = {d.id: tier.min_player_level for npc in level.npcs for tier in npc.restock_tiers for d in tier.items}
        assert "linen_shirt" not in tiers and "iron_helm" not in tiers        # the cheap stuff is always on the racks

