import pytest

from unprepared.config import PlayerSettings
from unprepared.core.math3d import Vector3D
from unprepared.entities import Item, Player


@pytest.fixture
def player():
    return Player(Vector3D(-12.0, 0.0, -6.0), gold=50.0, inventory=[
        Item("lockpick_1", "Iron Lockpick", value=5.0, is_lockpick=True),
        Item("soft_shoes", "Soft Leather Boots", value=30.0, noise_modifier=-0.05),
        Item("stolen_goblet", "Stolen Silver Goblet", value=120.0, is_stolen=True),
    ])


def test_defaults(player):
    assert player.rotation_x == 0.0 and player.rotation_y == 0.0
    assert not player.is_crouching and not player.has_lit_torch
    assert player.gold == 50.0 and len(player.inventory) == 3


def test_empty_player_has_no_inventory():
    assert Player(Vector3D(0, 0, 0)).inventory == []


def test_eye_position(player):
    assert player.eye_position == Vector3D(-12.0, 1.55, -6.0)                  # a townsman's eyes are at 1.6


def test_take_lockpick(player):
    assert player.take_lockpick().id == "lockpick_1"
    assert not player.has_lockpick() and player.take_lockpick() is None
    assert [i.id for i in player.inventory] == ["soft_shoes", "stolen_goblet"]


def test_collision_box(player):
    box = player.collision_box()
    assert box.min_point == Vector3D(-12.3, 0.25, -6.3) and box.max_point == Vector3D(-11.7, 1.7, -5.7)
    assert player.collision_box(Vector3D(0, 1, 0)).min_point == Vector3D(-0.3, 1.25, -0.3)


def test_key_and_lockpick_queries(player):
    assert player.has_lockpick()
    assert not player.has_key_for("shop_door")
    player.inventory.append(Item("shop_key", "Key", is_key=True, key_door_id="shop_door"))
    assert player.has_key_for("shop_door")


class TestNoise:
    def test_stationary_is_silent(self, player):
        assert player.get_noise_level(False, True) == 0.0

    def test_walking(self, player):
        # 1.5 * 1.0 * (1 - 0.05)
        assert player.get_noise_level(True, False) == pytest.approx(1.425)

    def test_sprinting(self, player):
        assert player.get_noise_level(True, True) == pytest.approx(2.85)

    def test_crouching(self, player):
        player.is_crouching = True
        assert player.get_noise_level(True, False) == pytest.approx(1.5 * 0.3 * 0.95)

    def test_clothing_cannot_make_the_player_silent(self):
        # a wardrobe of "silent" clothes bottoms out at the clothing floor (x0.25), never at zero or below
        p = Player(Vector3D(0, 0, 0), inventory=[Item("q", "Quiet", noise_modifier=-1.0), Item("r", "Quieter", noise_modifier=-0.5)])
        assert p.clothing_noise_factor() == 0.25
        assert p.get_noise_level(True, True) == pytest.approx(3.0 * 0.25)
        assert p.get_noise_level(True, False) == pytest.approx(1.5 * 0.25)

    def test_noise_floor(self):
        p = Player(Vector3D(0, 0, 0), inventory=[Item("q", "Quiet", noise_modifier=-1.0)])
        p.noise_skill_multiplier = 0.01
        assert p.get_noise_level(True, True) == 0.1                      # the absolute floor still applies


class TestVisibility:
    def test_daylight_standing(self, player):
        assert player.get_visibility(1.0) == 1.0

    def test_torch_adds_light(self, player):
        player.has_lit_torch = True
        assert player.get_visibility(0.1) == pytest.approx(1.0)

    def test_crouch_reduces(self, player):
        player.is_crouching = True
        assert player.get_visibility(1.0) == pytest.approx(0.4)


class TestLook:
    def test_yaw_and_pitch(self, player):
        player.look(100, -40)
        assert player.rotation_y == pytest.approx(15.0)
        assert player.rotation_x == pytest.approx(6.0)

    def test_pitch_clamped(self, player):
        player.look(0, -100000)
        assert player.rotation_x == 89.0
        player.look(0, 100000)
        assert player.rotation_x == -89.0

    def test_custom_sensitivity(self):
        p = Player(Vector3D(0, 0, 0), settings=PlayerSettings(mouse_sensitivity=1.0))
        p.look(10, 0)
        assert p.rotation_y == 10.0


class TestPersistence:
    def test_roundtrip(self, player):
        player.gold = 12.5
        player.has_lit_torch = True
        player.is_crouching = True
        player.position = Vector3D(1, 2, 3)
        data = player.to_dict()
        other = Player(Vector3D(0, 0, 0))
        other.load_dict(data)
        assert other.to_dict() == data
        assert other.position == Vector3D(1, 2, 3)
        assert [i.id for i in other.inventory] == ["lockpick_1", "soft_shoes", "stolen_goblet"]

    def test_load_keeps_current_values_for_missing_fields(self):
        other = Player(Vector3D(0, 0, 0), gold=50.0)
        other.rotation_y = 33.0
        other.load_dict({"position": {"x": 1, "y": 2, "z": 3}})
        assert other.gold == 50.0 and other.inventory == [] and not other.has_lit_torch
        assert other.rotation_y == 33.0

    def test_rotation_is_saved(self, player):
        player.rotation_y, player.rotation_x = 90.0, -10.0
        data = player.to_dict()
        assert data["rotation_y"] == 90.0 and data["rotation_x"] == -10.0
        other = Player(Vector3D(0, 0, 0))
        other.load_dict(data)
        assert (other.rotation_y, other.rotation_x) == (90.0, -10.0)
