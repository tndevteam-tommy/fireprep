"""
Firestarter - Smartphone Store App
Implements DESIGN Section 5.1 catalog and purchase transactions.  A purchase returns an
``item_id`` which the WorldManager applies to the player's inventory / camp.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from firestarter.economy.credits import CreditSystem


@dataclass
class StoreItem:
    item_id: str
    name: str
    cost: float
    description: str
    category: str
    icon: str = "box"
    unique: bool = False


DEFAULT_CATALOG: List[StoreItem] = [
    StoreItem("lighter_refill", "Pocket Lighter Refill Fluid", 45.0,
              "Refills your pocket lighter with 4 fresh fire starts.", "Accelerants & Fire", "lighter"),
    StoreItem("paracord", "Paracord Cordage (2 pcs)", 30.0,
              "Nylon cord for the bow drill. Outlasts bark twine.", "Bushcraft", "cord"),
    StoreItem("waterproof_tarp", "Waterproof Polyethylene Tarp", 110.0,
              "Shields the raised drying rack from rain so timber seasons to ~12 % EMC.", "Shelter & Seasoning", "tarp"),
    StoreItem("steel_knife", "High-Carbon Steel Pocket Knife", 140.0,
              "Razor edge: twice the shavings per stick and clean bark strips.", "Tools", "knife", True),
    StoreItem("steel_axe", "Drop-Forged Felling Axe", 240.0,
              "Heavy steel bit: fells mature hardwoods ~2.5x faster than a stone axe.", "Tools", "axe", True),
    StoreItem("gasoline_canister", "Gasoline Canister (1 Gallon)", 280.0,
              "High-octane accelerant (2.8 kg). Explosive volatility overpowers wet timber. Use with care!", "Accelerants & Fire", "gas"),
    StoreItem("reference_book_advanced", "Field Reference Book Editions", 75.0,
              "Unlocks the Swedish Torch, Lean-To and Star Fire architectural blueprints.", "Knowledge", "book", True),
]


class StoreApp:
    def __init__(self, catalog: Optional[List[StoreItem]] = None):
        self.catalog = catalog if catalog is not None else list(DEFAULT_CATALOG)
        self.selected_index = 0
        self.owned: Dict[str, int] = {}

    def select_next(self):
        self.selected_index = (self.selected_index + 1) % len(self.catalog)

    def select_prev(self):
        self.selected_index = (self.selected_index - 1) % len(self.catalog)

    def is_sold_out(self, item: StoreItem) -> bool:
        return item.unique and self.owned.get(item.item_id, 0) > 0

    def purchase_selected(self, credit_system: CreditSystem) -> Dict[str, Any]:
        item = self.catalog[self.selected_index]
        return self.purchase(item, credit_system)

    def purchase(self, item: StoreItem, credit_system: CreditSystem) -> Dict[str, Any]:
        if self.is_sold_out(item):
            return {"success": False, "item_id": item.item_id, "name": item.name,
                    "message": f"You already own the {item.name}."}
        if credit_system.spend(item.cost):
            self.owned[item.item_id] = self.owned.get(item.item_id, 0) + 1
            return {"success": True, "item_id": item.item_id, "name": item.name,
                    "message": f"Purchased {item.name} for {item.cost:.0f} credits."}
        return {"success": False, "item_id": item.item_id, "name": item.name,
                "message": f"Insufficient credits: need {item.cost:.0f}, have {credit_system.credits:.0f}."}
