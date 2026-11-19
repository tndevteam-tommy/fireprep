from unprepared.core.math3d import Vector3D
from unprepared.entities import Door, DoorSchedule, Item


class TestItem:
    def test_defaults(self):
        item = Item("x", "Thing")
        assert item.value == 50.0 and not item.is_stolen and not item.is_key and not item.is_lockpick

    def test_dict_roundtrip(self):
        item = Item("k", "Key", value=1.0, is_key=True, key_door_id="shop_door", noise_modifier=-0.05, is_stolen=True)
        assert Item.from_dict(item.to_dict()) == item

    def test_from_dict_defaults_missing_fields(self):
        item = Item.from_dict({"id": "a", "name": "A"})
        assert item == Item("a", "A")

    def test_opens(self):
        key = Item("k", "Key", is_key=True, key_door_id="shop_door")
        assert key.opens("shop_door") and not key.opens("hideout_door")
        assert not Item("n", "Not a key", key_door_id="shop_door").opens("shop_door")

    def test_for_sale_flag_roundtrips(self):
        key = Item("k", "Key", is_key=True, key_door_id="d", for_sale=False)
        assert not key.for_sale and Item.from_dict(key.to_dict()) == key
        assert Item.from_dict({"id": "a", "name": "A"}).for_sale is True

    def test_sell_value(self):
        assert Item("a", "A", value=120.0, is_stolen=True).sell_value(0.5) == 60.0
        assert Item("a", "A", value=120.0).sell_value(0.5) == 120.0


class TestDoor:
    def test_bounding_box(self):
        door = Door("d", Vector3D(8.5, 0.0, 0.0))
        box = door.get_bounding_box()
        assert box.min_point == Vector3D(8.5, 0.0, -0.5)
        assert box.max_point == Vector3D(11.5, 3.5, 0.5)

    def test_toggle_unlock_lock(self):
        door = Door("d", Vector3D(0, 0, 0), is_locked=False)
        assert door.toggle() is True and door.is_open
        assert door.toggle() is False and not door.is_open
        door.lock_and_close()
        assert door.is_locked and not door.is_open
        door.unlock_and_open()
        assert not door.is_locked and door.is_open

    def test_schedule_opens_and_closes(self):
        door = Door("shop", Vector3D(0, 0, 0), schedule=DoorSchedule(8, 22, "open!", "closed!"))
        assert door.apply_schedule(7) is None and door.is_locked
        assert door.apply_schedule(8) == "open!" and not door.is_locked and door.is_open
        assert door.apply_schedule(12) is None and door.is_open
        assert door.apply_schedule(22) == "closed!" and door.is_locked and not door.is_open

    def test_no_schedule_is_noop(self):
        door = Door("d", Vector3D(0, 0, 0))
        assert door.apply_schedule(8) is None and door.is_locked

    def test_collider_follows_the_leaf(self):
        door = Door("d", Vector3D(8.5, 0.0, 0.0), is_locked=False)
        closed = door.collider()
        assert closed.min_point == Vector3D(8.5, 0.0, -0.1) and closed.max_point == Vector3D(11.5, 3.5, 0.1)
        door.is_open = True
        opened = door.collider()
        assert opened.min_point == Vector3D(8.4, 0.0, -3.0) and opened.max_point == Vector3D(8.6, 3.5, 0.1)

    def test_center_and_owners(self):
        door = Door("d", Vector3D(8.5, 0.0, 0.0), owners=("Bob", "guards"))
        assert door.center == Vector3D(10.0, 0.0, 0.0) and door.owners == ("Bob", "guards")
        assert door.owned_by("Bob") and door.owned_by("Mira", "guards") and not door.owned_by("Mira", "thieves")
        assert Door("x", Vector3D(0, 0, 0)).owners == ()

    def test_persistence(self):
        door = Door("d", Vector3D(0, 0, 0), is_locked=True)
        door.load_dict({"door_id": "d", "is_locked": False, "is_open": True})
        assert not door.is_locked and door.is_open
        assert door.to_dict() == {"door_id": "d", "is_locked": False, "is_open": True, "sealed": False}


class TestStacks:
    def test_display_name(self):
        assert Item("b", "Bread", stackable=True, quantity=15).display_name == "Bread (15)"
        assert Item("b", "Bread", stackable=True, quantity=1).display_name == "Bread"
        assert Item("k", "Key", quantity=3).display_name == "Key"          # not stackable: never shown

    def test_split(self):
        stack = Item("b", "Bread", value=2.0, stackable=True, quantity=5)
        unit = stack.split(2)
        assert (stack.quantity, unit.quantity) == (3, 2) and unit.same_stack(stack)
        assert stack.split(99).quantity == 3 and stack.quantity == 0

    def test_same_stack_rules(self):
        a = Item("b", "Bread", stackable=True)
        assert a.same_stack(Item("b", "Bread", stackable=True, quantity=4))
        assert not a.same_stack(Item("b", "Bread", stackable=True, is_stolen=True))
        assert not a.same_stack(Item("b", "Bread"))

    def test_dict_roundtrip_keeps_quantity(self):
        item = Item("b", "Bread", stackable=True, quantity=7)
        assert Item.from_dict(item.to_dict()) == item
        assert Item.from_dict({"id": "b", "name": "Bread"}).quantity == 1


class TestInventoryHelpers:
    def test_add_merges_and_appends(self):
        from unprepared.entities.inventory import add_item
        inv = [Item("b", "Bread", stackable=True, quantity=2)]
        add_item(inv, Item("b", "Bread", stackable=True, quantity=3))
        add_item(inv, Item("k", "Key"))
        add_item(inv, Item("k", "Key"))
        assert [(i.id, i.quantity) for i in inv] == [("b", 5), ("k", 1), ("k", 1)]
        assert sum(i.quantity for i in inv) == 7

    def test_take_splits_or_removes(self):
        from unprepared.entities.inventory import find_item, take_item
        inv = [Item("b", "Bread", stackable=True, quantity=2), Item("k", "Key")]
        unit = take_item(inv, inv[0])
        assert unit.quantity == 1 and inv[0].quantity == 1
        take_item(inv, inv[0])
        assert [i.id for i in inv] == ["k"]
        assert take_item(inv, Item("x", "X")) is None
        assert find_item(inv, "k").name == "Key" and find_item(inv, "zz") is None
