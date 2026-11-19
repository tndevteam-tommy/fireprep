"""Inventories (core.inventory) and the cyber mods: everybody carries things; the netrunner
starts with the deck, two mods fresh out of the box and a little money; installed mods
change what the netrunner can do."""

from __future__ import annotations

import math

import pytest

from cyberduke.assets.icons import available_icons
from cyberduke.core.inventory import (
    CLOSEUP_HACK_SPEED,
    ITEMS,
    PING_RANGE,
    Inventory,
    InventoryFullError,
    ItemStack,
    starting_inventory,
)
from cyberduke.levels import build_night_market
from cyberduke.simulation import InputState, Simulation

DT = 1 / 30


def test_every_item_has_an_icon_and_a_description():
    for item in ITEMS.values():
        assert item.icon in available_icons(), item.item_id
        assert item.name and item.description and item.kind_label


def test_stacks_stack_and_other_things_do_not():
    inv = Inventory(capacity=4)
    inv.add("eddies", 100)
    inv.add("eddies", 40)
    assert inv.count("eddies") == 140 and len(inv) == 1
    inv.add("datastick")
    inv.add("datastick")  # not stackable: two of them, two slots
    assert inv.count("datastick") == 2 and len(inv) == 3
    assert inv.remove("eddies", 40) and inv.count("eddies") == 100
    assert not inv.remove("eddies", 1000), "not that many"
    assert inv.count("eddies") == 100, "and nothing taken"
    inv.add("phone")
    with pytest.raises(InventoryFullError):
        inv.add("keys_16")
    with pytest.raises(KeyError, match="unknown item"):
        inv.add("plasma_rifle")


def test_data_shards_are_one_item_told_apart_by_what_is_on_them():
    inv = Inventory()
    a = inv.add("data_shard", meta="greenhouse door cipher")
    inv.add("data_shard", meta="payroll")
    assert len(inv) == 2 and a.name == "DATA SHARD: GREENHOUSE DOOR CIPHER"
    assert inv.has("data_shard", "payroll") and not inv.has("data_shard", "nothing")


def test_one_mod_per_slot_and_the_stats_multiply():
    inv = starting_inventory("player")
    ping, jack = inv.find("mod_ping_amp"), inv.find("mod_quickjack")
    assert not ping.equipped and not jack.equipped, "fresh out of the box"
    assert inv.stat(PING_RANGE) == 1.0
    assert inv.equip(ping) and inv.equip(jack)
    assert inv.stat(PING_RANGE) == pytest.approx(1.6)
    assert inv.stat(CLOSEUP_HACK_SPEED) == pytest.approx(1.65)
    assert not inv.equip(inv.find("eddies")), "only cyber mods go in"
    # a second mod for the same slot pushes the first one out
    other = ItemStack("mod_ping_amp")
    inv.stacks.append(other)
    inv.equip(other)
    assert other.equipped and not ping.equipped
    assert inv.toggle_equipped(other) is False and inv.stat(PING_RANGE) == 1.0
    assert ITEMS["mod_wraith"].effect_lines() == ["DETECTION GAIN -35%"]


def test_inventories_round_trip():
    inv = starting_inventory("player")
    inv.equip(inv.find("mod_quickjack"))
    inv.add("data_shard", meta="payroll")
    again = Inventory.from_dict(inv.to_dict())
    assert again == inv
    assert again.stat(CLOSEUP_HACK_SPEED) == inv.stat(CLOSEUP_HACK_SPEED)


def test_everybody_carries_something_and_nix_has_the_reward():
    sim = Simulation(build_night_market())
    for npc in sim.npcs:
        assert len(npc.inventory) > 0, getattr(npc, "role", npc)
    assert sim.npc_with_role("smoker").inventory.has("mod_wraith")
    assert sim.npc_with_role("resident").inventory.has("keys_16")
    facility = Simulation()
    assert all(g.inventory.has("halcyon_badge") for g in facility.guards)
    p = facility.player
    assert [s.item_id for s in p.inventory] == [
        "cyberdeck",
        "mod_ping_amp",
        "mod_quickjack",
        "eddies",
    ]


def test_stolen_data_goes_on_the_deck_and_into_the_inventory():
    sim = Simulation()
    datacore = sim.terminals[1]
    datacore.steal_intel()
    sim.update(DT, InputState())
    assert "greenhouse door cipher" in sim.player.intel
    assert sim.player.inventory.has("data_shard", "greenhouse door cipher")


# ----------------------------------------------------------------------------- the mods
def _equip(sim, item_id):
    inv = sim.player.inventory
    inv.equip(inv.find(item_id))


def test_the_ping_amplifier_pings_further():
    sim = Simulation()
    sim.player.x, sim.player.z = 0.0, 0.0
    sim.update(DT, InputState())
    far = [
        t for t in sim.hack_targets if 16.0 < math.dist(sim.viewpoint.eye, t.hack_position) <= 25.0
    ]
    assert far, "something between 16 m and 25 m away"
    sim.update(DT, InputState(scan=True))
    assert not any(sim.scanner.is_revealed(t.hack_id) for t in far), "out of the stock ping"
    _equip(sim, "mod_ping_amp")
    for _ in range(40):  # the scanner's cooldown
        sim.update(DT, InputState())
    sim.update(DT, InputState(scan=True))
    assert all(sim.scanner.is_revealed(t.hack_id) for t in far), "the amplified ping reaches"
    assert sim.scanner.pulse.max_radius == pytest.approx(16.0 * 1.6)


def test_the_quickjack_speeds_up_close_up_hacks_and_only_those():
    def link_time(equipped: bool) -> float:
        sim = Simulation()
        if equipped:
            _equip(sim, "mod_quickjack")
        sim.player.x, sim.player.z = 3.5, -2.5
        t = 0.0
        while sim.hacking.session is None and t < 5.0:
            sim.update(DT, InputState(interact=True))
            t += DT
        return t

    stock, fast = link_time(False), link_time(True)
    assert fast < stock / 1.5, (stock, fast)

    from cyberduke.systems.hacking import Access

    sim = Simulation()
    _equip(sim, "mod_quickjack")
    sim.update(DT, InputState())
    door = sim.room.doors["blast_door"]
    near = sim.terminals[0]
    sim.player.x, sim.player.z = near.x, near.z + 1.8  # arm's length from the desk
    sim.update(DT, InputState())
    assert sim.hacking.hack_speed(near, Access.DIRECT) == pytest.approx(1.65)
    assert sim.hacking.hack_speed(door, Access.NETWORK) == 1.0, "over the network: not close-up"
    sim.player.x, sim.player.z = near.x, near.z + 8.0
    sim.update(DT, InputState())
    assert sim.hacking.hack_speed(near, Access.DIRECT) == 1.0, "eight metres is not close-up"


def test_the_wraith_damper_slows_the_detection_meter():
    from cyberduke.ai.guard import Inspecting

    def seen_for(seconds: float, equipped: bool) -> float:
        sim = Simulation()
        if equipped:
            sim.player.inventory.add("mod_wraith")
            _equip(sim, "mod_wraith")
        g = sim.guards[0]
        g.x, g.z, g.yaw = 0.0, 2.0, 0.0
        g.fsm.transition_to(Inspecting(source=(0.0, 6.0), duration=1e9))
        sim.player.x, sim.player.z = 1.0, 6.0
        for _ in range(int(seconds / DT)):
            sim.update(DT, InputState())
        return sim.detection.level

    plain, damped = seen_for(1.0, False), seen_for(1.0, True)
    assert 0.0 < damped < plain * 0.7
