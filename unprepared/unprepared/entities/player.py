from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from ..config import ClothingSettings, PlayerSettings
from ..core.math3d import BoundingBox3D, Vector3D
from .equipment import EquipSlot, Figure, SLOT_ORDER, clothing_effect, slot_of
from .inventory import add_item, take_item
from .items import Item
from .skills import SkillTree


class Player:
    def __init__(self, start_pos: Vector3D, gold: float = 50.0, inventory: Optional[Iterable[Item]] = None,
                 settings: Optional[PlayerSettings] = None, clothing: Optional[ClothingSettings] = None):
        self.settings = settings or PlayerSettings()
        self.clothing = clothing or ClothingSettings()
        self.position = start_pos
        self.rotation_y = 0.0   # yaw, degrees
        self.rotation_x = 0.0   # pitch, degrees
        self.is_crouching = False
        self.has_lit_torch = False
        self.indoors = False             # under a roof right now (set by the world each frame: the torch law is the street's)
        self.gold = gold
        self.inventory: List[Item] = list(inventory) if inventory is not None else []
        self.skills = SkillTree()
        self.current_eye_height = self.settings.eye_height   # animates between standing/crouching
        self.noise_skill_multiplier = 1.0                    # set by the game from the Sneak skill (quieter with practice)
        self.crimes_witnessed = 0                            # times an NPC caught the player in the act
        self.freeze_timer = 0.0                              # controls locked (just caught / being told off)
        self.turn_target: Optional[Vector3D] = None          # where the view is swinging towards
        self.figure = Figure(legs=(0.3, 0.27, 0.3), hair=(0.26, 0.17, 0.1), undershirt=(0.82, 0.78, 0.68))   # under the gear

    # --- queries -----------------------------------------------------------
    @property
    def level(self) -> int:
        """Overall character level: every two skill levels gained is one character level."""
        gained = sum(self.skills.level(skill) - 1 for skill in self.skills.skills)
        return 1 + gained // 2

    @property
    def target_eye_height(self) -> float:
        return self.settings.crouch_eye_height if self.is_crouching else self.settings.eye_height

    @property
    def eye_position(self) -> Vector3D:
        return Vector3D(self.position.x, self.position.y + self.current_eye_height, self.position.z)

    def update_stance(self, delta_time: float) -> float:
        """Move the camera height towards the stance's eye height; returns the new height."""
        target = self.target_eye_height
        step = self.settings.stance_transition_speed * delta_time
        if self.current_eye_height < target:
            self.current_eye_height = min(target, self.current_eye_height + step)
        elif self.current_eye_height > target:
            self.current_eye_height = max(target, self.current_eye_height - step)
        return self.current_eye_height

    def has_key_for(self, door_id: str) -> bool:
        return any(item.opens(door_id) for item in self.inventory)

    def has_lockpick(self) -> bool:
        return any(item.is_lockpick for item in self.inventory)

    def take_lockpick(self) -> Optional[Item]:
        """Remove and return one lockpick from the inventory (None if there is none)."""
        for item in self.inventory:
            if item.is_lockpick:
                return take_item(self.inventory, item, 1)
        return None

    def add_item(self, item: Item) -> Item:
        return add_item(self.inventory, item)

    # --- equipment ----------------------------------------------------------
    @property
    def equipment(self) -> Dict[EquipSlot, Item]:
        """What is worn in each slot (at most one item per slot; a second worn item found for a
        slot - say, a helmet taken back from the lockbox while another is on - is unworn)."""
        worn: Dict[EquipSlot, Item] = {}
        for item in self.inventory:
            if not item.equipped:
                continue
            slot = slot_of(item)
            if slot is None or slot in worn:
                item.equipped = False
                continue
            worn[slot] = item
        return worn

    def equipped_in(self, slot: EquipSlot) -> Optional[Item]:
        return self.equipment.get(slot)

    def equip(self, item: Item) -> bool:
        """Wear / wield ``item`` (which must be in the inventory and be equipment); whatever was in
        that slot comes off. False if it cannot be worn."""
        slot = slot_of(item)
        if slot is None or item not in self.inventory:
            return False
        current = self.equipped_in(slot)
        if current is not None and current is not item:
            current.equipped = False
        item.equipped = True
        return True

    def unequip(self, item: Item) -> bool:
        if not item.equipped:
            return False
        item.equipped = False
        return True

    def toggle_equipped(self, item: Item) -> Optional[bool]:
        """Equip if not worn, take off if worn; None if it is not equipment."""
        if slot_of(item) is None:
            return None
        if item.equipped:
            self.unequip(item)
            return False
        return self.equip(item)

    def worn_items(self) -> List[Item]:
        worn = self.equipment
        return [worn[slot] for slot in SLOT_ORDER if slot in worn]

    def collision_box(self, position: Optional[Vector3D] = None) -> BoundingBox3D:
        p = position or self.position
        s = self.settings
        return BoundingBox3D(
            Vector3D(p.x - s.collision_radius, p.y + s.collision_foot_clearance, p.z - s.collision_radius),
            Vector3D(p.x + s.collision_radius, p.y + s.collision_head_height, p.z + s.collision_radius),
        )

    def skill_bonus(self, skill) -> int:
        """Levels everything worn adds to ``skill`` (a ``SkillType`` or its name)."""
        name = getattr(skill, "value", skill)
        return sum(item.bonus_for(name) for item in self.equipment.values())

    def outfit_effects(self) -> Tuple[float, float]:
        """(visibility delta, noise delta) of everything worn, by material, shade and slot
        (``equipment.clothing_effect``)."""
        vis = noise = 0.0
        for item in self.equipment.values():
            effect = clothing_effect(item, self.clothing)
            if effect is not None:
                vis += effect.visibility
                noise += effect.noise
        return vis, noise

    def visibility_factor(self) -> float:
        """How much the outfit makes the player stand out (1 = plain clothes): steel and pale
        cloth more, dark cloth less, within ``ClothingSettings`` floor and ceiling."""
        vis, _ = self.outfit_effects()
        c = self.clothing
        return max(c.visibility_floor, min(c.visibility_ceiling, 1.0 + vis))

    def clothing_noise_factor(self) -> float:
        """How the outfit changes footstep noise: what is *worn* counts by its material and slot
        (mail rattles, cloth shoes are silent - equipment only changes your step when it is on
        you), plus any explicit ``noise_modifier`` of worn items and of things with no slot (a
        pair of gloves counts whenever it is carried). Clamped so a wardrobe of silent clothes can
        never make the player noiseless (or, worse, negative)."""
        _, noise = self.outfit_effects()
        extra = sum(item.noise_modifier for item in self.inventory if item.equipped or slot_of(item) is None)
        return max(self.settings.clothing_noise_floor, min(self.clothing.noise_ceiling, 1.0 + noise + extra))

    def get_noise_level(self, is_moving: bool, is_sprinting: bool) -> float:
        if not is_moving:
            return 0.0
        s = self.settings
        base_speed = s.noise_sprint if is_sprinting else s.noise_walk
        crouch_mod = s.noise_crouch_multiplier if self.is_crouching else 1.0
        return max(s.noise_floor, base_speed * crouch_mod * self.clothing_noise_factor() * self.noise_skill_multiplier)

    def get_visibility(self, ambient_light: float, is_moving: bool = True) -> float:
        """How easy the player is to see in ``ambient_light``: the light on them (plus their own
        torch), their stance, what they wear (``visibility_factor``) - and, crouched and holding
        still, less again (``still_crouch_multiplier``)."""
        s = self.settings
        torch_light = s.visibility_torch_bonus if self.has_lit_torch else 0.0
        stance_mod = s.visibility_crouch_multiplier if self.is_crouching else 1.0
        still = self.clothing.still_crouch_multiplier if (self.is_crouching and not is_moving) else 1.0
        return (ambient_light + torch_light) * stance_mod * self.visibility_factor() * still

    # --- mutations ----------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self.freeze_timer > 0.0

    def yaw_towards(self, target: Vector3D) -> float:
        dx, dz = target.x - self.position.x, target.z - self.position.z
        return math.degrees(math.atan2(dx, -dz)) % 360.0 if (dx or dz) else self.rotation_y

    def hold_attention(self, target: Vector3D, seconds: float) -> None:
        """An NPC has the player's attention: the controls lock for ``seconds`` and the view
        swings smoothly round to face them (``update_turning``) instead of snapping."""
        self.freeze_timer = max(self.freeze_timer, seconds)
        self.turn_target = target.copy()

    def pitch_towards(self, target: Vector3D) -> float:
        """The pitch that looks somebody standing at ``target`` in the face (their eyes at a
        standing person's eye height): level when standing, upwards from a crouch."""
        dist = math.hypot(target.x - self.position.x, target.z - self.position.z)
        face_y = target.y + self.settings.eye_height
        return max(-45.0, min(45.0, math.degrees(math.atan2(face_y - self.eye_position.y, max(dist, 0.3)))))

    def update_turning(self, delta_time: float) -> None:
        """Advance the freeze timer and the smooth turn (yaw and pitch) towards ``turn_target``."""
        if self.freeze_timer > 0.0:
            self.freeze_timer = max(0.0, self.freeze_timer - delta_time)
        if self.turn_target is None:
            return
        speed = self.settings.attention_turn_speed * delta_time
        target_yaw = self.yaw_towards(self.turn_target)
        diff = (target_yaw - self.rotation_y + 180.0) % 360.0 - 180.0
        if abs(diff) <= speed:
            self.rotation_y = target_yaw
        else:
            self.rotation_y = (self.rotation_y + math.copysign(speed, diff)) % 360.0
        target_pitch = self.pitch_towards(self.turn_target)
        if abs(self.rotation_x - target_pitch) <= speed:
            self.rotation_x = target_pitch
        else:
            self.rotation_x += math.copysign(speed, target_pitch - self.rotation_x)
        if self.rotation_y == target_yaw and self.rotation_x == target_pitch:
            self.turn_target = None

    def look(self, dx_pixels: float, dy_pixels: float) -> None:
        s = self.settings
        self.rotation_y += dx_pixels * s.mouse_sensitivity
        self.rotation_x -= dy_pixels * s.mouse_sensitivity
        self.rotation_x = max(-s.max_pitch, min(s.max_pitch, self.rotation_x))

    # --- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "position": self.position.to_dict(),
            "rotation_y": self.rotation_y,
            "rotation_x": self.rotation_x,
            "gold": self.gold,
            "has_lit_torch": self.has_lit_torch,
            "is_crouching": self.is_crouching,
            "inventory": [item.to_dict() for item in self.inventory],
            "skills": self.skills.to_dict(),
            "crimes_witnessed": self.crimes_witnessed,
        }

    def load_dict(self, data: dict) -> None:
        self.position = Vector3D.from_dict(data["position"])
        self.rotation_y = data.get("rotation_y", self.rotation_y)
        self.rotation_x = data.get("rotation_x", self.rotation_x)
        self.gold = data.get("gold", self.gold)
        self.has_lit_torch = data.get("has_lit_torch", False)
        self.is_crouching = data.get("is_crouching", False)
        self.inventory = [Item.from_dict(i) for i in data.get("inventory", [])]
        if "skills" in data:
            self.skills.load_dict(data["skills"])
        self.crimes_witnessed = data.get("crimes_witnessed", 0)
        self.current_eye_height = self.target_eye_height
