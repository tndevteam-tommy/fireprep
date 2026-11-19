from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Item:
    id: str
    name: str
    value: float = 50.0
    is_stolen: bool = False
    noise_modifier: float = 0.0
    is_key: bool = False
    key_door_id: Optional[str] = None
    is_lockpick: bool = False
    for_sale: bool = True          # False: owned but never offered in a merchant's stock (e.g. shop keys)
    stackable: bool = False        # identical small items merge into one stack
    quantity: int = 1
    is_food: bool = False          # NPCs eat these at meal times
    # Whom the item rightfully belongs to: an NPC's name, a faction tag ("guards"), ``PLAYER_OWNER``
    # or None (nobody / unknown). Theft flags the item stolen but leaves the owner as it was -
    # that is how the guards know where to return it. Ownership only changes hands legitimately:
    # a sale, a fence laundering it, a gift, a meal on the house.
    owner: Optional[str] = None
    category: str = "goods"        # what kind of thing it is (specialist merchants trade only some categories)
    # Equipment (see ``entities/equipment.py``): ``gear`` names the physical model the renderer mounts
    # on whoever wears it (which fixes the slot: head, neck, body, feet, weapon), ``gear_color`` tints
    # it, ``equipped`` says it is being worn by the inventory's owner right now.
    gear: Optional[str] = None
    gear_color: Optional[Tuple[float, float, float]] = None
    equipped: bool = False
    material: Optional[str] = None     # "metal" | "leather" | "cloth"; None = the gear kind's usual material
    # How rare the thing is ("common" .. "legendary", see ``config.RARITIES``): rarer things are
    # harder to lift from a pocket or a watched shelf, and epic and legendary ones are mounted,
    # chained or warded as well (``systems/stealth.secure_verdict``).
    rarity: str = "common"
    # What wearing it adds to the wearer's skills: ((skill name, levels), ...) e.g. (("Sneak", 12),)
    skill_bonuses: Tuple[Tuple[str, int], ...] = ()

    PLAYER_OWNER = "player"

    @property
    def is_players(self) -> bool:
        return self.owner == Item.PLAYER_OWNER

    @property
    def is_equipment(self) -> bool:
        """Can be worn or wielded (has a gear model and is a single thing, not a stack)."""
        return self.gear is not None and not self.stackable

    @property
    def display_name(self) -> str:
        """'Fresh Bread (3)' for stacks of more than one, else just the name."""
        return f"{self.name} ({self.quantity})" if self.stackable and self.quantity > 1 else self.name

    def same_stack(self, other: "Item") -> bool:
        return (self.stackable and other.stackable and self.id == other.id and self.name == other.name
                and self.value == other.value and self.is_stolen == other.is_stolen and self.is_food == other.is_food
                and self.owner == other.owner and self.rarity == other.rarity)

    def bonus_for(self, skill_name: str) -> int:
        """Levels this item adds to ``skill_name`` when worn."""
        return sum(levels for name, levels in self.skill_bonuses if name == skill_name)

    @property
    def bonus_text(self) -> str:
        """'Sneak +12, Pickpocket +4' (empty when it adds nothing)."""
        return ", ".join(f"{name} +{levels}" for name, levels in self.skill_bonuses if levels)

    def split(self, count: int = 1) -> "Item":
        """Take ``count`` units off this stack as a new Item (quantity of this one shrinks)."""
        count = max(1, min(count, self.quantity))
        self.quantity -= count
        return dataclasses.replace(self, quantity=count)      # every field by name: safe against new or reordered fields

    def opens(self, door_id: str) -> bool:
        return self.is_key and self.key_door_id == door_id

    def steal(self) -> "Item":
        """Mark the item as stolen; it still belongs to its owner."""
        self.is_stolen = True
        return self

    def transfer_to(self, owner: Optional[str], launder: bool = False) -> "Item":
        """A legitimate change of hands: the item is ``owner``'s now. A fence *launders* what it
        buys - the stolen flag comes off - while an honest buyer never takes stolen goods."""
        self.owner = owner
        if launder:
            self.is_stolen = False
        return self

    def sell_value(self, stolen_multiplier: float = 0.5) -> float:
        return self.value * stolen_multiplier if self.is_stolen else self.value

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "value": self.value,
            "is_stolen": self.is_stolen, "noise_modifier": self.noise_modifier,
            "is_key": self.is_key, "key_door_id": self.key_door_id, "is_lockpick": self.is_lockpick,
            "for_sale": self.for_sale, "stackable": self.stackable, "quantity": self.quantity, "is_food": self.is_food,
            "owner": self.owner, "category": self.category,
            "gear": self.gear, "gear_color": list(self.gear_color) if self.gear_color is not None else None,
            "equipped": self.equipped, "material": self.material,
            "rarity": self.rarity, "skill_bonuses": [[name, levels] for name, levels in self.skill_bonuses],
        }

    @staticmethod
    def from_dict(d: dict) -> "Item":
        item = Item(
            id=d["id"], name=d["name"], value=d.get("value", 50.0),
            is_stolen=d.get("is_stolen", False), noise_modifier=d.get("noise_modifier", 0.0),
            is_key=d.get("is_key", False), key_door_id=d.get("key_door_id"),
            is_lockpick=d.get("is_lockpick", False), for_sale=d.get("for_sale", True),
            stackable=d.get("stackable", False), quantity=d.get("quantity", 1), is_food=d.get("is_food", False),
            owner=d.get("owner"), category=d.get("category", "goods"),
            gear=d.get("gear"), gear_color=tuple(d["gear_color"]) if d.get("gear_color") else None,
            equipped=d.get("equipped", False), material=d.get("material"),
            rarity=d.get("rarity", "common"),
            skill_bonuses=tuple((str(name), int(levels)) for name, levels in d.get("skill_bonuses", ())),
        )
        from .equipment import gear_for
        spec = gear_for(item.id)
        if "gear" not in d and spec is not None:     # a save from before equipment existed: the catalogue knows the item
            item.gear, item.gear_color, item.material = spec.gear, spec.color, spec.material
        if "rarity" not in d and spec is not None:   # ... or from before rarities and skill bonuses
            item.rarity, item.skill_bonuses = spec.rarity, spec.bonuses
        return item
