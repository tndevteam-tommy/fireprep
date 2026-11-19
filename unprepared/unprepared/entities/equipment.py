"""Equipment: what can be worn or wielded, where on the body, and how it is drawn.

Every piece of equipment in the game is an :class:`~.items.Item` with a ``slot`` (where it goes)
and a ``gear`` kind (which physical model the renderer mounts on the figure, tinted by
``gear_color``). The five slots follow the body top to bottom:

* ``head``   - helmets, caps, scarves, hoods, coifs
* ``neck``   - aprons, sashes, necklaces and cloaks (clasped at the neck, worn over the body)
* ``body``   - tunics, shirts, mail, breastplates
* ``feet``   - boots, shoes, slippers
* ``weapon`` - what hangs from the belt or sits in the hand: daggers, swords, a sling

The townsfolk are dressed the same way the player is: an :class:`~.outfit.Outfit` (the
description in the level data) is turned into real equipment items by
:func:`equipment_from_outfit`, mounted on the same figure by the same drawing code
(``rendering/gear.py``). The player sees themselves in the [Tab] view wearing whatever they
have equipped. ``GEAR_CATALOGUE`` is the single source of truth for the equipment the level
sells, loots and starts the player with; saves made before equipment existed get their items'
slots from it by id.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .items import Item

Color = Tuple[float, float, float]


class EquipSlot(str, Enum):
    HEAD = "head"
    NECK = "neck"
    BODY = "body"
    FEET = "feet"
    WEAPON = "weapon"

    @property
    def label(self) -> str:
        return {"head": "Head", "neck": "Neck", "body": "Body", "feet": "Feet", "weapon": "Weapon"}[self.value]


SLOT_ORDER: Tuple[EquipSlot, ...] = (EquipSlot.HEAD, EquipSlot.NECK, EquipSlot.BODY, EquipSlot.FEET, EquipSlot.WEAPON)

# gear kind -> slot: the physical models the renderer knows how to mount (rendering/gear.py)
GEAR_SLOTS: Dict[str, EquipSlot] = {
    # head
    "helmet": EquipSlot.HEAD, "cap": EquipSlot.HEAD, "leather_cap": EquipSlot.HEAD, "scarf": EquipSlot.HEAD,
    "hood": EquipSlot.HEAD, "coif": EquipSlot.HEAD,
    # neck
    "apron_cloth": EquipSlot.NECK, "apron_leather": EquipSlot.NECK, "sash": EquipSlot.NECK, "necklace": EquipSlot.NECK,
    "cloak": EquipSlot.NECK,
    # body
    "tunic": EquipSlot.BODY, "breastplate": EquipSlot.BODY, "chain_shirt": EquipSlot.BODY, "cuirass": EquipSlot.BODY,
    # feet
    "boots": EquipSlot.FEET, "shoes": EquipSlot.FEET, "slippers": EquipSlot.FEET,
    # weapons
    "dagger": EquipSlot.WEAPON, "jeweled_dagger": EquipSlot.WEAPON, "sword": EquipSlot.WEAPON, "sling": EquipSlot.WEAPON,
    # finery and the rare pieces (see rendering/gear.py)
    "feathered_hat": EquipSlot.HEAD, "circlet": EquipSlot.HEAD, "shadow_hood": EquipSlot.HEAD,
    "mantle": EquipSlot.NECK, "feather_cloak": EquipSlot.NECK, "legend_cloak": EquipSlot.NECK, "amulet": EquipSlot.NECK,
    "choker": EquipSlot.NECK,
    "doublet": EquipSlot.BODY, "gown": EquipSlot.BODY,
    "winged_boots": EquipSlot.FEET,
}


# gear kind -> (what it is made of, how much of the body it covers): the stealth model's inputs.
# A kind's material may be overridden per item (``Item.material``: black *cloth* shoes use the
# ``shoes`` model in cloth).
MATERIALS = ("metal", "leather", "cloth")
GEAR_MATERIALS: Dict[str, Tuple[str, float]] = {
    "helmet": ("metal", 1.0), "cap": ("cloth", 1.0), "leather_cap": ("leather", 1.0), "scarf": ("cloth", 1.0),
    "hood": ("cloth", 1.0), "coif": ("cloth", 1.0),
    "apron_cloth": ("cloth", 1.0), "apron_leather": ("leather", 1.0), "sash": ("cloth", 0.5),
    "necklace": ("metal", 0.15), "cloak": ("cloth", 1.0),
    "tunic": ("cloth", 1.0), "breastplate": ("metal", 1.0), "chain_shirt": ("metal", 1.0), "cuirass": ("metal", 1.0),
    "boots": ("leather", 1.0), "shoes": ("leather", 1.0), "slippers": ("cloth", 1.0),
    "dagger": ("metal", 0.5), "jeweled_dagger": ("metal", 0.5), "sword": ("metal", 1.0), "sling": ("leather", 0.5),
    "feathered_hat": ("cloth", 1.0), "circlet": ("metal", 0.2), "shadow_hood": ("cloth", 1.0),
    "mantle": ("cloth", 1.0), "feather_cloak": ("cloth", 1.0), "legend_cloak": ("cloth", 1.0), "amulet": ("metal", 0.15),
    "choker": ("metal", 0.1),
    "doublet": ("cloth", 1.0), "gown": ("cloth", 1.0),
    "winged_boots": ("leather", 1.0),
}
MATERIAL_LUMINANCE = {"metal": 0.8, "leather": 0.45, "cloth": 0.55}     # an untinted piece's shade


@dataclass(frozen=True)
class GearSpec:
    """How an item id is worn: its gear kind (-> slot), tint and (if not the kind's usual)
    material - and how rare it is and what it does for the wearer's skills."""
    gear: str
    color: Optional[Color] = None
    material: Optional[str] = None
    rarity: str = "common"
    bonuses: Tuple[Tuple[str, int], ...] = ()

    @property
    def slot(self) -> EquipSlot:
        return GEAR_SLOTS[self.gear]


# Every piece of equipment that exists as an item in the game, by item id. Bright, shiny, noisy
# things are common and cheap and do nothing for a thief; brown leather is the middle ground;
# dark cloth is rare and never on a shop's rack - it is stolen from the rich, the smugglers and
# the keep - and the few legendary pieces, each in the deepest, best-kept place of the realm,
# make a thief of anybody who can get them free (``systems/stealth.secure_verdict``).
S, PP, LP, ST, BA, SM = "Sneak", "Pickpocket", "Lockpicking", "Stealing", "Barter", "Smithing"
GEAR_CATALOGUE: Dict[str, GearSpec] = {
    # the smith's rack: steel, cheap and plentiful
    "iron_dagger": GearSpec("dagger", (0.7, 0.7, 0.74)),
    "short_sword": GearSpec("sword", (0.72, 0.72, 0.76)),
    "steel_sword": GearSpec("sword", (0.85, 0.86, 0.9)),
    "jeweled_dagger": GearSpec("jeweled_dagger", (0.8, 0.8, 0.84), rarity="uncommon", bonuses=((PP, 2),)),
    "leather_cap": GearSpec("leather_cap", (0.5, 0.36, 0.24)),
    "iron_helm": GearSpec("helmet", (0.55, 0.56, 0.6)),
    "chain_shirt": GearSpec("chain_shirt", (0.78, 0.79, 0.83)),
    "plate_cuirass": GearSpec("cuirass", (0.8, 0.81, 0.85)),
    "leather_jerkin": GearSpec("tunic", (0.42, 0.3, 0.2), material="leather", rarity="uncommon", bonuses=((S, 1),)),
    "smith_apron": GearSpec("apron_leather", (0.45, 0.32, 0.2), rarity="uncommon", bonuses=((SM, 6),)),
    # the tailor's rack: bright linen for pennies, brown wool at the dear end
    "linen_shirt": GearSpec("tunic", (0.92, 0.9, 0.85)),
    "white_cloak": GearSpec("cloak", (0.9, 0.88, 0.84)),
    "red_cap": GearSpec("cap", (0.75, 0.2, 0.18)),
    "wool_cloak": GearSpec("cloak", (0.45, 0.38, 0.3), rarity="uncommon", bonuses=((S, 2),)),
    "soft_slippers": GearSpec("slippers", (0.62, 0.5, 0.45), rarity="uncommon", bonuses=((S, 2),)),
    # the rich: finery - and what they keep locked away
    "velvet_cloak": GearSpec("cloak", (0.36, 0.12, 0.26), rarity="rare", bonuses=((S, 5), (BA, 3))),
    "ravenfeather_cloak": GearSpec("feather_cloak", (0.07, 0.07, 0.1), rarity="rare", bonuses=((S, 10), (PP, 4))),
    "pearl_choker": GearSpec("choker", (0.95, 0.94, 0.9), rarity="rare", bonuses=((PP, 8), (BA, 6))),
    "ruby_necklace": GearSpec("necklace", (0.85, 0.1, 0.15), rarity="rare", bonuses=((BA, 6),)),
    "signet_ring_chain": GearSpec("necklace", (0.95, 0.8, 0.3), rarity="uncommon", bonuses=((BA, 3),)),
    # dark cloth: smugglers' stashes, the rich's wardrobes, the keep
    "black_hood": GearSpec("hood", (0.08, 0.08, 0.1), rarity="rare", bonuses=((S, 6),)),
    "black_tunic": GearSpec("tunic", (0.1, 0.1, 0.12), rarity="rare", bonuses=((S, 6),)),
    "shadow_boots": GearSpec("boots", (0.12, 0.1, 0.13), rarity="rare", bonuses=((S, 8),)),
    "black_cloak": GearSpec("cloak", (0.06, 0.06, 0.08), rarity="epic", bonuses=((S, 12), (ST, 4))),
    "black_shoes": GearSpec("shoes", (0.06, 0.06, 0.07), material="cloth", rarity="epic", bonuses=((S, 10), (PP, 6))),
    "shadowsilk_hood": GearSpec("shadow_hood", (0.1, 0.06, 0.14), rarity="epic", bonuses=((S, 12), (ST, 8))),
    "duskwalker_boots": GearSpec("winged_boots", (0.1, 0.11, 0.16), material="cloth", rarity="epic",
                                 bonuses=((S, 15), (PP, 5))),
    # legendary: one of each, where the realm keeps its worst dangers
    "amulet_open_door": GearSpec("amulet", (0.2, 0.95, 0.9), rarity="legendary", bonuses=((LP, 40), (ST, 10))),
    "nightweave_cloak": GearSpec("legend_cloak", (0.05, 0.04, 0.09), rarity="legendary", bonuses=((S, 35), (ST, 15))),
    "fingers_of_the_magpie": GearSpec("circlet", (0.85, 0.9, 1.0), rarity="legendary", bonuses=((PP, 35), (ST, 20))),
    # the player's own
    "soft_shoes": GearSpec("boots", (0.42, 0.28, 0.18)),
    # Pell's
    "sling": GearSpec("sling", (0.45, 0.33, 0.22)),
    # what the rich wear (their finery is their own; these are the pieces a thief might lift)
    "feathered_hat": GearSpec("feathered_hat", (0.3, 0.12, 0.35)),
    "gilt_circlet": GearSpec("circlet", (0.95, 0.8, 0.35), rarity="uncommon", bonuses=((BA, 2),)),
}
RARITY_ORDER = ("common", "uncommon", "rare", "epic", "legendary")


def gear_for(item_id: str) -> Optional[GearSpec]:
    return GEAR_CATALOGUE.get(item_id)


# --- the stealth model: what a piece of clothing does to how you are seen and heard -------------
def material_of(item: Item) -> Optional[str]:
    """What ``item`` is made of: its own ``material`` if set, else its gear kind's."""
    if item.material in MATERIALS:
        return item.material
    if item.gear in GEAR_MATERIALS:
        return GEAR_MATERIALS[item.gear][0]
    return None


def coverage_of(item: Item) -> float:
    return GEAR_MATERIALS.get(item.gear, ("cloth", 1.0))[1] if item.gear else 0.0


def luminance(color: Optional[Color], material: Optional[str] = None) -> float:
    """How light a colour is (0 black .. 1 white); an untinted piece takes its material's shade."""
    if color is None:
        return MATERIAL_LUMINANCE.get(material or "cloth", 0.55)
    r, g, b = color
    return max(0.0, min(1.0, 0.299 * r + 0.587 * g + 0.114 * b))


def shade_word(lum: float) -> str:
    if lum < 0.18:
        return "black"
    if lum < 0.34:
        return "dark"
    if lum < 0.58:
        return "drab"
    if lum < 0.78:
        return "light"
    return "bright"


@dataclass(frozen=True)
class ClothingEffect:
    """What one worn item adds to the player's visibility and footstep noise (multiplier deltas)."""
    visibility: float          # weighted by slot and coverage: what the item adds to the player's factor
    noise: float
    material: str
    shade: str
    raw_visibility: float = 0.0    # the material + shade alone (how this *kind* of thing behaves)
    raw_noise: float = 0.0


def clothing_effect(item: Item, settings) -> Optional[ClothingEffect]:
    """The stealth effect of wearing ``item`` (None if it is not equipment): material (metal
    glints and clanks, leather is dull and fairly quiet, cloth is dim and silent) plus shade
    (lighter than the pivot stands out, darker hides), scaled by how much of the body the piece
    covers and by its slot's weight (``ClothingSettings``)."""
    slot = slot_of(item)
    material = material_of(item)
    if slot is None or material is None:
        return None
    vis_w = dict(settings.visibility_weights).get(slot.value, 0.0)
    noise_w = dict(settings.noise_weights).get(slot.value, 0.0)
    cover = coverage_of(item)
    lum = luminance(item.gear_color, material)
    vis = getattr(settings, f"visibility_{material}") + (lum - settings.shade_pivot) * settings.shade_slope
    noise = getattr(settings, f"noise_{material}")
    return ClothingEffect(vis * vis_w * cover, noise * noise_w * cover, material, shade_word(lum), vis, noise)


def describe_clothing(item: Item, settings) -> str:
    """'black cloth - silent, hard to see' for the [Tab] view."""
    effect = clothing_effect(item, settings)
    if effect is None:
        return ""
    seen = ("glints" if effect.raw_visibility > 0.2 else "stands out" if effect.raw_visibility > 0.06
            else "hard to see" if effect.raw_visibility < -0.18 else "dim" if effect.raw_visibility < -0.04 else "plain")
    heard = ("clanks" if effect.raw_noise > 0.12 else "noisy" if effect.raw_noise > 0.03
             else "silent" if effect.raw_noise < -0.05 else "quiet" if effect.raw_noise < -0.01 else "ordinary")
    return f"{effect.shade} {effect.material} - {seen}, {heard}"


def slot_of(item: Item) -> Optional[EquipSlot]:
    """Where ``item`` is worn, or None for something that is not equipment."""
    if item.gear is None or item.stackable:
        return None
    return GEAR_SLOTS.get(item.gear)


def is_equipment(item: Item) -> bool:
    return slot_of(item) is not None


@dataclass(frozen=True)
class Figure:
    """What a person looks like underneath their equipment: the colour of their hose, their
    hair, and the plain shirt that shows when nothing is worn over it."""
    legs: Color = (0.35, 0.30, 0.28)
    hair: Color = (0.31, 0.2, 0.1)
    undershirt: Color = (0.82, 0.78, 0.68)


def worn_item(item_id: str, name: str, gear: str, color: Optional[Color] = None, owner: Optional[str] = None) -> Item:
    """An equipment item that exists only on somebody's back (never traded: not for sale)."""
    return Item(item_id, name, value=0.0, for_sale=False, owner=owner, category="clothing", gear=gear, gear_color=color,
                equipped=True)


def equipment_from_outfit(outfit, owner: Optional[str] = None) -> Dict[EquipSlot, Item]:
    """Turn an :class:`~.outfit.Outfit` (the level's description of what somebody wears) into
    the real equipment items mounted on their figure - the same items, drawn by the same code,
    the player can wear."""
    worn: Dict[EquipSlot, Item] = {}
    body = getattr(outfit, "body", "tunic")
    if outfit.armor:
        worn[EquipSlot.BODY] = worn_item("breastplate", "Steel Breastplate", "breastplate", None, owner)
    else:
        names = {"tunic": "Wool Tunic", "doublet": "Velvet Doublet", "gown": "Silk Gown"}
        worn[EquipSlot.BODY] = worn_item(body, names.get(body, "Wool Tunic"), body, outfit.tunic, owner)
    neck = getattr(outfit, "neck", None)
    if outfit.apron is not None:
        kind = "apron_leather" if outfit.apron == "leather" else "apron_cloth"
        worn[EquipSlot.NECK] = worn_item(f"{outfit.apron}_apron", f"{outfit.apron.capitalize()} Apron", kind, None, owner)
    elif outfit.sash is not None:
        worn[EquipSlot.NECK] = worn_item("sash", "Merchant's Sash", "sash", outfit.sash, owner)
    elif neck is not None:
        names = {"mantle": "Fur-trimmed Mantle", "necklace": "Gold Necklace", "choker": "Pearl Choker", "cloak": "Wool Cloak"}
        worn[EquipSlot.NECK] = worn_item(neck, names[neck], neck, outfit.neck_color, owner)
    if outfit.hat is not None:
        names = {"helmet": "Kettle Helm", "cap": "Cloth Cap", "scarf": "Head Scarf", "hood": "Deep Hood", "coif": "Linen Coif",
                 "feathered_hat": "Feathered Hat", "circlet": "Gilt Circlet"}
        worn[EquipSlot.HEAD] = worn_item(outfit.hat, names[outfit.hat], outfit.hat, outfit.hat_color, owner)
    return worn


def figure_from_outfit(outfit) -> Figure:
    return Figure(legs=outfit.legs, hair=outfit.hair)
