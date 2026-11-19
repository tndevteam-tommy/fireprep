"""Items and inventories: what the netrunner (and everybody else) carries.

An ``ItemDef`` describes a kind of thing - its name, what sort of thing it is, its icon
(``assets.icons``), whether it stacks, and for a cyber mod the slot it goes in and what it
does to its wearer (``effects``: stat name -> multiplier). An ``Inventory`` is a list of
``ItemStack``s (an item, how many, and for a mod whether it is equipped) with a capacity;
``stat(name)`` multiplies together the effects of everything equipped.

Every character has one: the player's is shown on the inventory screen ([TAB]); the
people of the street and the guards carry what their jobs need (``starting_inventory``).
Pure data - no pygame, no OpenGL - and serialisable (``to_dict`` / ``from_dict``) for the
save files.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Stats the cyber mods change (multipliers; 1.0 = no change)
PING_RANGE = "ping_range"  # the scanner pulse's radius
CLOSEUP_HACK_SPEED = "closeup_hack_speed"  # jacking in, desk work, hacks from arm's length
DETECTION_GAIN = "detection_gain"  # how fast being seen fills the detection meter

STAT_LABELS = {
    PING_RANGE: "PING RANGE",
    CLOSEUP_HACK_SPEED: "CLOSE-UP HACK SPEED",
    DETECTION_GAIN: "DETECTION GAIN",
}

KINDS = {
    "cybermod": "CYBERWARE MOD",
    "quest": "QUEST ITEM",
    "data": "DATA SHARD",
    "money": "CURRENCY",
    "gear": "GEAR",
    "junk": "PERSONAL EFFECTS",
    "key": "KEY",
}


@dataclass(frozen=True)
class ItemDef:
    item_id: str
    name: str
    kind: str  # a key of ``KINDS``
    description: str
    icon: str  # icon name (``assets.icons``)
    stackable: bool = False
    slot: str | None = None  # the cyberware slot a mod is installed in (one mod per slot)
    effects: tuple[tuple[str, float], ...] = ()  # (stat, multiplier) while equipped
    value: int = 0  # what it is worth, in eddies

    @property
    def equippable(self) -> bool:
        return self.kind == "cybermod" and self.slot is not None

    @property
    def kind_label(self) -> str:
        return KINDS.get(self.kind, self.kind.upper())

    def effect_lines(self) -> list[str]:
        """Human-readable effects: "PING RANGE +60%"."""
        out = []
        for stat, mult in self.effects:
            pct = round((mult - 1.0) * 100.0)
            out.append(f"{STAT_LABELS.get(stat, stat.upper())} {'+' if pct >= 0 else ''}{pct}%")
        return out


ITEMS: dict[str, ItemDef] = {}


def register(item: ItemDef) -> ItemDef:
    ITEMS[item.item_id] = item
    return item


# ------------------------------------------------------------------------- the catalogue
register(ItemDef(
    "mod_ping_amp", "SONARIS PING AMPLIFIER MK.II", "cybermod",
    "A black-market booster clipped onto the deck's antenna stack. It shouts the scanner "
    "ping a lot further into the walls - more doors, cameras and chrome light up on every "
    "pulse. Runs hot. Smells of solder.",
    "mod_ping", slot="CORTEX", effects=((PING_RANGE, 1.6),), value=900,
))  # fmt: skip
register(ItemDef(
    "mod_quickjack", "ZETATECH QUICKJACK LINK", "cybermod",
    "Fingertip interface jacks wired straight into the deck. Whatever you hack from arm's "
    "length - jacking into a computer, working its interface, a quickhack on something "
    "right in front of you - goes through far faster.",
    "mod_quickjack", slot="HANDS", effects=((CLOSEUP_HACK_SPEED, 1.65),), value=1100,
))  # fmt: skip
register(ItemDef(
    "mod_wraith", "MILITECH WRAITH SIGNAL DAMPER", "cybermod",
    "Mil-spec. Not sold anywhere you would want to walk into. Smears your thermal and EM "
    "signature so cameras and eyes take longer to be sure it was you they saw. Something "
    "special, as promised.",
    "mod_wraith", slot="NERVOUS SYSTEM", effects=((DETECTION_GAIN, 0.65),), value=4500,
))  # fmt: skip
register(ItemDef(
    "datastick", "DATA STICK (NO.16)", "quest",
    "A scuffed data stick with a hand-written label: 'DELL - BACKUP - DO NOT LOSE'. "
    "Whatever is on it, Nix wants it badly.",
    "datastick",
))  # fmt: skip
register(ItemDef(
    "data_shard", "DATA SHARD", "data",
    "Private data copied off a corporate computer onto the deck. Some locks read their "
    "cipher straight off it.",
    "data_shard",
))  # fmt: skip
register(ItemDef(
    "eddies", "EURODOLLARS", "money",
    "Eddies on a cred chip. Not much. Never enough.",
    "eddies", stackable=True, value=1,
))  # fmt: skip
register(ItemDef(
    "cyberdeck", "TSUNAMI DEFENDER CYBERDECK", "gear",
    "Your deck. Third-hand, held together with tape and spite - but it pings, it jacks in "
    "and it uploads. Everything you hack goes through it.",
    "cyberdeck",
))  # fmt: skip
register(ItemDef(
    "halcyon_badge", "HALCYON ID BADGE", "key",
    "A Halcyon Dynamics security badge.", "badge",
))  # fmt: skip
register(ItemDef(
    "cigarettes", "CIGARETTES", "junk", "A crushed pack of synth-tobacco.", "cigarettes",
    stackable=True,
))  # fmt: skip
register(ItemDef("phone", "PHONE", "junk", "A cracked phone.", "phone"))
register(ItemDef(
    "keys_16", "KEYS TO NO.16", "key", "A key ring: the front door of number 16.", "keys",
))  # fmt: skip
register(ItemDef("parcel", "PARCEL", "junk", "A parcel for number 16.", "parcel"))
register(ItemDef("knife", "BUTCHER KNIFE", "junk", "A cleaver-ish kitchen knife.", "knife"))
register(ItemDef("stim", "BLACK-LACE STIM", "junk", "A cheap combat stim.", "stim", stackable=True))


def item(item_id: str) -> ItemDef:
    try:
        return ITEMS[item_id]
    except KeyError:
        raise KeyError(f"unknown item {item_id!r}; known: {sorted(ITEMS)}") from None


@dataclass
class ItemStack:
    item_id: str
    count: int = 1
    meta: str = ""  # what a generic item is *of* (a data shard: the data's name)
    equipped: bool = False

    @property
    def item(self) -> ItemDef:
        return item(self.item_id)

    @property
    def name(self) -> str:
        return f"{self.item.name}: {self.meta.upper()}" if self.meta else self.item.name

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"item": self.item_id, "count": self.count}
        if self.meta:
            d["meta"] = self.meta
        if self.equipped:
            d["equipped"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ItemStack:
        item(str(d["item"]))  # unknown items fail loudly
        return cls(
            str(d["item"]),
            int(d.get("count", 1)),
            str(d.get("meta", "")),
            bool(d.get("equipped", False)),
        )


class InventoryFullError(RuntimeError):
    pass


@dataclass
class Inventory:
    """Stacks of items, in the order they were picked up."""

    capacity: int = 48
    stacks: list[ItemStack] = field(default_factory=list)

    # ------------------------------------------------------------------ contents
    def add(self, item_id: str, count: int = 1, meta: str = "") -> ItemStack:
        definition = item(item_id)
        if definition.stackable:
            for s in self.stacks:
                if s.item_id == item_id and s.meta == meta:
                    s.count += count
                    return s
        if len(self.stacks) >= self.capacity:
            raise InventoryFullError(f"no room for {definition.name}")
        stack = ItemStack(item_id, count if definition.stackable else 1, meta)
        self.stacks.append(stack)
        return stack

    def remove(self, item_id: str, count: int = 1, meta: str | None = None) -> bool:
        """Takes ``count`` of an item out (a mod being removed is unequipped). False, and
        nothing taken, if there are not that many."""
        if self.count(item_id, meta) < count:
            return False
        left = count
        for s in list(self.stacks):
            if s.item_id != item_id or (meta is not None and s.meta != meta):
                continue
            take = min(left, s.count)
            s.count -= take
            left -= take
            if s.count <= 0:
                self.stacks.remove(s)
            if left == 0:
                break
        return True

    def count(self, item_id: str, meta: str | None = None) -> int:
        return sum(
            s.count
            for s in self.stacks
            if s.item_id == item_id and (meta is None or s.meta == meta)
        )

    def has(self, item_id: str, meta: str | None = None) -> bool:
        return self.count(item_id, meta) > 0

    def find(self, item_id: str, meta: str | None = None) -> ItemStack | None:
        return next(
            (s for s in self.stacks if s.item_id == item_id and (meta is None or s.meta == meta)),
            None,
        )

    def clear(self) -> None:
        self.stacks.clear()

    def __len__(self) -> int:
        return len(self.stacks)

    def __iter__(self):
        return iter(self.stacks)

    # ------------------------------------------------------------------ cyberware
    def equip(self, stack: ItemStack) -> bool:
        """Installs a mod. Whatever was in its slot comes out. False for things that are
        not mods (or not in this inventory)."""
        if stack not in self.stacks or not stack.item.equippable:
            return False
        for other in self.stacks:
            if other is not stack and other.equipped and other.item.slot == stack.item.slot:
                other.equipped = False
        stack.equipped = True
        return True

    def unequip(self, stack: ItemStack) -> bool:
        if stack not in self.stacks or not stack.equipped:
            return False
        stack.equipped = False
        return True

    def toggle_equipped(self, stack: ItemStack) -> bool:
        """Equip/unequip button. Returns whether it is equipped now."""
        if stack.equipped:
            self.unequip(stack)
        else:
            self.equip(stack)
        return stack.equipped

    @property
    def equipped(self) -> list[ItemStack]:
        return [s for s in self.stacks if s.equipped]

    def stat(self, name: str) -> float:
        """The product of every equipped mod's multiplier for ``name`` (1.0: untouched)."""
        value = 1.0
        for s in self.equipped:
            for stat, mult in s.item.effects:
                if stat == name:
                    value *= mult
        return value

    # ------------------------------------------------------------------ saving
    def to_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "stacks": [s.to_dict() for s in self.stacks]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Inventory:
        return cls(
            int(d.get("capacity", 48)), [ItemStack.from_dict(s) for s in d.get("stacks", ())]
        )

    @classmethod
    def of(cls, contents: Iterable[tuple[str, int]], capacity: int = 48) -> Inventory:
        inv = cls(capacity)
        for item_id, count in contents:
            inv.add(item_id, count)
        return inv


# What everybody starts with. The netrunner: the deck, two mods fresh out of the box (not
# yet installed) and a little money. The people of the street and the guards carry what
# their jobs need.
PLAYER_START: tuple[tuple[str, int], ...] = (
    ("cyberdeck", 1),
    ("mod_ping_amp", 1),
    ("mod_quickjack", 1),
    ("eddies", 140),
)

NPC_START: dict[str, tuple[tuple[str, int], ...]] = {
    "guard": (("halcyon_badge", 1), ("eddies", 60), ("stim", 1)),
    "smoker": (("cigarettes", 11), ("phone", 1), ("eddies", 35), ("mod_wraith", 1)),
    "resident": (("keys_16", 1), ("phone", 1), ("eddies", 80)),
    "courier": (("eddies", 25), ("phone", 1)),
    "office_worker": (("eddies", 220), ("phone", 1)),
    "tinkerer": (("eddies", 310),),
    "stroller": (("eddies", 15), ("cigarettes", 3)),
    # (the chef's butcher knife is not in a pocket: it is a real thing on the chopping
    # board or in his hand - ``LevelData.items``)
    "sushi_chef": (("eddies", 480), ("cigarettes", 6)),
    "junk_dealer": (("eddies", 1250), ("stim", 4)),
}


def starting_inventory(who: str) -> Inventory:
    """A fresh inventory for the player ("player") or a role / "guard"."""
    if who == "player":
        return Inventory.of(PLAYER_START)
    return Inventory.of(NPC_START.get(who, (("eddies", 10),)), capacity=12)
