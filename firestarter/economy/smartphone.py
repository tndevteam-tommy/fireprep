"""
Firestarter - Smartphone OS & Application State Machine
Implements DESIGN Section 5.1 & 8.1 (SmartphoneApp): a rugged wilderness phone with a home
screen (credits, weather forecast widget) and two apps - Store ($) and Topographical Map.
"""

from enum import Enum
from typing import Dict, Any, Optional, List

import numpy as np

from firestarter.economy.credits import CreditSystem
from firestarter.economy.store import StoreApp
from firestarter.economy.topomap import TopographicalMap


class PhoneAppType(Enum):
    HOME = "Home"
    STORE = "Store ($)"
    TOPO_MAP = "Topographical Map"


APP_ORDER = [PhoneAppType.STORE, PhoneAppType.TOPO_MAP]


class SmartphoneOS:
    def __init__(self, credit_system: Optional[CreditSystem] = None):
        self.credits = credit_system if credit_system is not None else CreditSystem()
        self.store = StoreApp()
        self.topo_map = TopographicalMap()
        self.is_open = False
        self.active_app = PhoneAppType.HOME
        self.home_selection = 0
        self.battery_percent = 94.0
        self.screen_width = 360
        self.screen_height = 700
        self.status_message = "Ready"
        self.last_claimed = 0.0
        self.open_time = 0.0
        self.purchase_log: List[str] = []
        self.map_zoom = 1.0

    def toggle(self) -> bool:
        self.is_open = not self.is_open
        if self.is_open:
            self.open_time = 0.0
            claimed = self.credits.claim_pending()
            self.last_claimed = claimed
            if claimed >= 0.5:
                self.status_message = f"+{claimed:.0f} credits claimed for fire mastery"
        return self.is_open

    def switch_app(self, app_type: PhoneAppType):
        self.active_app = app_type
        self.status_message = f"Opened {app_type.value}"

    def next_app(self):
        if self.active_app == PhoneAppType.HOME:
            self.switch_app(APP_ORDER[self.home_selection])
            return
        idx = (APP_ORDER.index(self.active_app) + 1) % len(APP_ORDER)
        self.switch_app(APP_ORDER[idx])

    def prev_app(self):
        if self.active_app == PhoneAppType.HOME:
            return
        idx = (APP_ORDER.index(self.active_app) - 1) % len(APP_ORDER)
        self.switch_app(APP_ORDER[idx])

    def go_home(self):
        self.active_app = PhoneAppType.HOME

    def update(self, dt: float):
        if self.is_open:
            self.open_time += dt
            self.battery_percent = max(5.0, self.battery_percent - dt * 0.004)
        else:
            self.battery_percent = min(100.0, self.battery_percent + dt * 0.0005)

    def handle_input(self, action: str) -> Dict[str, Any]:
        """Phone interaction: 'up', 'down', 'left', 'right', 'select', 'switch', 'back'."""
        if not self.is_open:
            return {"handled": False}
        if action == "back":
            if self.active_app != PhoneAppType.HOME:
                self.go_home()
                return {"handled": True, "app": "home"}
            return {"handled": False}
        if action == "switch":
            self.next_app()
            return {"handled": True, "app": self.active_app.value}

        if self.active_app == PhoneAppType.HOME:
            if action in ("left", "up"):
                self.home_selection = (self.home_selection - 1) % len(APP_ORDER)
            elif action in ("right", "down"):
                self.home_selection = (self.home_selection + 1) % len(APP_ORDER)
            elif action == "select":
                self.switch_app(APP_ORDER[self.home_selection])
            return {"handled": True, "app": "home", "selection": self.home_selection}

        if self.active_app == PhoneAppType.STORE:
            if action == "down":
                self.store.select_next()
                return {"handled": True, "selected": self.store.selected_index}
            if action == "up":
                self.store.select_prev()
                return {"handled": True, "selected": self.store.selected_index}
            if action == "select":
                res = self.store.purchase_selected(self.credits)
                self.status_message = res["message"]
                if res["success"]:
                    self.purchase_log.append(res["name"])
                return {"handled": True, "purchase": res}
            if action in ("left", "right"):
                self.next_app() if action == "right" else self.prev_app()
                return {"handled": True, "app": self.active_app.value}

        if self.active_app == PhoneAppType.TOPO_MAP:
            if action == "up":
                self.map_zoom = min(3.0, self.map_zoom * 1.25)
            elif action == "down":
                self.map_zoom = max(0.5, self.map_zoom / 1.25)
            elif action in ("left", "right"):
                self.next_app() if action == "right" else self.prev_app()
            return {"handled": True, "app": "map", "zoom": self.map_zoom}
        return {"handled": False}

    def get_display_state(self, player_pos: np.ndarray, ambient_temp: float) -> Dict[str, Any]:
        """Complete state for the UI overlay renderer."""
        zone_name, humidity = self.topo_map.get_microclimate_at(player_pos[0], player_pos[2])
        return {
            "is_open": self.is_open,
            "active_app": self.active_app.value,
            "credits": self.credits.credits,
            "pending": self.credits.pending,
            "battery": self.battery_percent,
            "status": self.status_message,
            "gps_coords": (round(float(player_pos[0]), 1), round(float(player_pos[2]), 1)),
            "ambient_temp": round(float(ambient_temp), 1),
            "zone_name": zone_name,
            "humidity": round(humidity * 100.0, 0),
            "store_catalog": [
                {"id": it.item_id, "name": it.name, "cost": it.cost, "desc": it.description, "cat": it.category,
                 "icon": it.icon, "owned": self.store.owned.get(it.item_id, 0),
                 "sold_out": self.store.is_sold_out(it), "affordable": self.credits.credits >= it.cost,
                 "is_selected": (i == self.store.selected_index)}
                for i, it in enumerate(self.store.catalog)
            ],
            "map_waypoints": [{"name": wp.name, "pos": wp.position, "symbol": wp.symbol, "category": wp.category}
                              for wp in self.topo_map.waypoints],
        }
