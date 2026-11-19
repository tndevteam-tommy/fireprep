"""
Firestarter - InputManager
Maps the README control scheme onto game actions.  Key / button state is tracked from the
event stream itself (not from SDL's polled state) so scripted playback drives exactly the
same code paths as a human with a keyboard and mouse.

    W A S D        move            Shift      sprint           Ctrl     crouch (toggle)
    Mouse          look            LMB        primary hand     RMB      secondary hand
    Scroll         rotate piece    Shift+Scroll  tilt piece    MMB      reset rotation / auto-lean
    1-5            tools (Lighter, Knife, Axe, Bow / Hand Drill, Gasoline)
    6-0 / Q E      off-hand fuel (Tinder, Kindling, Firewood, Log, Fungus); Q/E also reach "empty hand"
    X              empty both hands
    F (hold)       blow - hold ~0.3-0.8 s for a deep, even breath
    C              twist bark into cordage
    Tab            smartphone      B          reference book   T        pass time
    H              controls help   Esc        pause / settings
"""

from typing import Dict, Any, Optional, Set, List

from firestarter.input.hand_physics_rig import HandController
from firestarter.simulation.world_manager import TOOLS, TOOL_LABELS, OFFHAND
from firestarter.config import GRAPHICS_PRESETS

PAUSE_ITEMS = ["Resume", "Master Volume", "Ambience Volume", "Effects Volume", "Graphics Preset",
               "Mouse Sensitivity", "Invert Mouse Y", "Sky Clock Speed", "Show FPS", "Quit to Desktop"]
CLOCK_SPEEDS = [1.0, 10.0, 30.0, 60.0]


class Settings:
    def __init__(self):
        self.master_volume = 0.8
        self.ambience_volume = 0.8
        self.effects_volume = 0.9
        self.graphics_preset = "High"
        self.mouse_sensitivity = 0.15
        self.invert_y = False
        self.clock_speed = 1.0
        self.show_fps = False

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class InputManager:
    def __init__(self, world, phone, ref_book, hands: HandController, settings: Optional[Settings] = None):
        self.world = world
        self.phone = phone
        self.ref_book = ref_book
        self.hands = hands
        self.settings = settings or Settings()
        self.tutorial = None                 # set by the Game
        self.is_paused = False
        self.pause_index = 0
        self.show_help = False
        self.quit_requested = False
        self.held: Set[str] = set()
        self.mouse_buttons: Set[int] = set()
        self.shift = False
        self.last_action: Dict[str, Any] = {}
        self.action_log: List[str] = []

    # ------------------------------------------------------------------ helpers
    @property
    def in_menu(self) -> bool:
        return self.phone.is_open or self.ref_book.is_open or self.world.time_engine.is_menu_open or self.is_paused

    def tutorial_step(self) -> int:
        return self.tutorial.step_number if self.tutorial else 0

    def get_tutorial_hint(self) -> str:
        return self.tutorial.hint_text() if self.tutorial else ""

    def movement_axes(self):
        f = (1.0 if "w" in self.held else 0.0) - (1.0 if "s" in self.held else 0.0)
        s = (1.0 if "d" in self.held else 0.0) - (1.0 if "a" in self.held else 0.0)
        if self.in_menu:
            return 0.0, 0.0
        return f, s

    def _log(self, name: str, res: Any = None) -> Dict[str, Any]:
        out = {"action": name}
        if isinstance(res, dict):
            out.update({k: v for k, v in res.items() if k != "action"})
        self.last_action = out
        self.action_log.append(name)
        self.action_log = self.action_log[-50:]
        return out

    # ------------------------------------------------------------------ keyboard
    def handle_key_down(self, key_name: str) -> Dict[str, Any]:
        key = key_name.lower()
        self.held.add(key)
        if key in ("left shift", "right shift"):
            self.shift = True
        w = self.world

        # ---- menu toggles
        if key == "tab":
            if self.is_paused:
                self.is_paused = False
            self.ref_book.is_open = False
            w.time_engine.is_menu_open = False
            is_open = self.phone.toggle()
            if is_open and self.phone.last_claimed > 0.5:
                w.notify(f"+{self.phone.last_claimed:.0f} credits claimed for fire mastery!", 4.0, "success")
                w.emit("credits", self.phone.last_claimed)
            w.emit("phone", is_open)
            return self._log("toggle_phone", {"is_open": is_open})
        if key == "escape":
            if self.phone.is_open:
                self.phone.is_open = False
                return self._log("close_phone")
            if self.ref_book.is_open:
                self.ref_book.is_open = False
                return self._log("close_book")
            if w.time_engine.is_menu_open:
                w.time_engine.is_menu_open = False
                return self._log("close_time_menu")
            if self.show_help:
                self.show_help = False
                return self._log("close_help")
            self.is_paused = not self.is_paused
            self.pause_index = 0
            return self._log("toggle_pause", {"paused": self.is_paused})

        if self.is_paused:
            return self._pause_key(key)

        if key == "b" and not self.phone.is_open:
            w.time_engine.is_menu_open = False
            is_open = self.ref_book.toggle()
            w.emit("book", is_open)
            return self._log("toggle_book", {"is_open": is_open})
        if key == "t" and not self.phone.is_open and not self.ref_book.is_open:
            is_open = w.time_engine.toggle_menu()
            return self._log("toggle_time_menu", {"is_open": is_open})

        # ---- menu navigation
        if self.phone.is_open:
            mapping = {"up": "up", "w": "up", "down": "down", "s": "down", "left": "left", "a": "left",
                       "right": "right", "d": "right", "return": "select", "space": "select", "e": "select",
                       "q": "switch", "backspace": "back"}
            act = mapping.get(key)
            if act:
                res = self.phone.handle_input(act)
                pur = res.get("purchase")
                if pur and pur.get("success"):
                    msg = w.apply_purchase(pur["item_id"])
                    w.notify(f"{pur['message']} {msg}", 5.0, "success")
                    w.emit("purchase", pur["item_id"])
                elif pur:
                    w.emit("ui_error", None)
                else:
                    w.emit("ui_click", None)
                return self._log("phone_input", res)
            return self._log("phone_input")
        if self.ref_book.is_open:
            if key in ("right", "d", "space", "page down"):
                self.ref_book.next_page()
                w.emit("page_turn", None)
            elif key in ("left", "a", "page up"):
                self.ref_book.prev_page()
                w.emit("page_turn", None)
            elif key in ("return", "e"):
                msg = self.ref_book.activate_current()
                w.notify(msg, 3.5)
                w.emit("ui_click", None)
            return self._log("book_page", {"page": self.ref_book.current_page})
        if w.time_engine.is_menu_open:
            if key in ("down", "s"):
                w.time_engine.next_option()
                w.emit("ui_click", None)
            elif key in ("up", "w"):
                w.time_engine.prev_option()
                w.emit("ui_click", None)
            elif key in ("return", "space", "e"):
                # the skip runs on the next frame, after a 'time passes...' card has been drawn
                hours = w.request_time_skip()
                return self._log("time_skip", {"hours": hours})
            return self._log("time_menu_nav")

        # ---- game actions
        if key == "h" or key == "f1":
            self.show_help = not self.show_help
            return self._log("toggle_help", {"show": self.show_help})
        if key == "f":
            w.breath_down()
            self.hands.blow_phase = 1.0
            return self._log("breath_start")
        if key == "c":
            return self._log("twist_cordage", w.twist_cordage())
        if key in ("left ctrl", "right ctrl"):
            # Ctrl toggles crouching (press again to stand)
            w.player.is_crouching = not w.player.is_crouching
            return self._log("crouch", {"crouching": w.player.is_crouching})
        if key in ("left shift", "right shift"):
            w.player.is_sprinting = True
            w.player.is_crouching = False      # breaking into a run stands you up
            return self._log("sprint")
        if key == "x":
            # put everything away: nothing in either hand
            w.empty_hands()
            w.emit("equip", "none")
            w.notify("Hands empty.", 1.5)
            return self._log("empty_hands")
        tool_keys = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
        if key in tool_keys:
            tool = TOOLS[tool_keys[key]]
            if tool == "bow_drill_spindle" and w.player.primary_hand == tool:
                mode = w.toggle_friction_mode()
                w.emit("equip", tool)
                return self._log("friction_mode", {"mode": mode})
            w.player.primary_hand = tool
            w.emit("equip", tool)
            extra = ""
            if tool == "pocket_lighter":
                extra = f" ({w.inventory.lighter_starts} fire starts left)"
            elif tool == "accelerant":
                extra = f" ({w.inventory.gasoline_kg:.1f} kg)"
            w.notify(f"Equipped: {TOOL_LABELS[tool]}{extra}", 1.8)
            return self._log("equip_tool", {"tool": tool})
        off_keys = {"6": "tinder", "7": "kindling", "8": "firewood", "9": "log", "0": "fungus"}
        if key in off_keys:
            # always selects (feeding the fire means pressing 8, RMB, 8, RMB ...); X or Q/E
            # empties the off hand
            w.set_offhand(off_keys[key])
            w.emit("equip", off_keys[key])
            return self._log("select_offhand", {"kind": off_keys[key]})
        if key == "q":
            w.cycle_offhand(-1)
            return self._log("select_offhand", {"kind": w.current_offhand()})
        if key == "e":
            w.cycle_offhand(1)
            return self._log("select_offhand", {"kind": w.current_offhand()})
        return self._log("none")

    def handle_key_up(self, key_name: str) -> Dict[str, Any]:
        key = key_name.lower()
        self.held.discard(key)
        w = self.world
        if key in ("left shift", "right shift"):
            self.shift = False
            w.player.is_sprinting = False
        if key == "f":
            if self.in_menu:
                w.breath_start = None
                return self._log("none")
            res = w.breath_up()
            return self._log("blow_breath", res)
        return self._log("key_up")

    def _pause_key(self, key: str) -> Dict[str, Any]:
        s = self.settings
        item = PAUSE_ITEMS[self.pause_index]
        if key in ("down", "s"):
            self.pause_index = (self.pause_index + 1) % len(PAUSE_ITEMS)
            self.world.emit("ui_click", None)
        elif key in ("up", "w"):
            self.pause_index = (self.pause_index - 1) % len(PAUSE_ITEMS)
            self.world.emit("ui_click", None)
        elif key in ("left", "right", "a", "d", "return", "space", "e"):
            step = -1 if key in ("left", "a") else 1
            presets = list(GRAPHICS_PRESETS.keys())
            if item == "Resume" and key in ("return", "space", "e"):
                self.is_paused = False
            elif item == "Quit to Desktop" and key in ("return", "space", "e"):
                self.quit_requested = True
            elif item == "Master Volume":
                s.master_volume = round(min(1.0, max(0.0, s.master_volume + 0.1 * step)), 2)
            elif item == "Ambience Volume":
                s.ambience_volume = round(min(1.0, max(0.0, s.ambience_volume + 0.1 * step)), 2)
            elif item == "Effects Volume":
                s.effects_volume = round(min(1.0, max(0.0, s.effects_volume + 0.1 * step)), 2)
            elif item == "Graphics Preset":
                i = presets.index(s.graphics_preset) if s.graphics_preset in presets else 2
                s.graphics_preset = presets[(i + step) % len(presets)]
            elif item == "Mouse Sensitivity":
                s.mouse_sensitivity = round(min(0.6, max(0.03, s.mouse_sensitivity + 0.02 * step)), 3)
            elif item == "Invert Mouse Y":
                s.invert_y = not s.invert_y
            elif item == "Sky Clock Speed":
                i = CLOCK_SPEEDS.index(s.clock_speed) if s.clock_speed in CLOCK_SPEEDS else 0
                s.clock_speed = CLOCK_SPEEDS[(i + step) % len(CLOCK_SPEEDS)]
                self.world.clock_scale = s.clock_speed
            elif item == "Show FPS":
                s.show_fps = not s.show_fps
            self.world.emit("ui_click", None)
        return self._log("pause_nav", {"item": PAUSE_ITEMS[self.pause_index]})

    # ------------------------------------------------------------------ mouse
    def handle_mouse_button_down(self, button: int) -> Dict[str, Any]:
        """1 LMB primary, 2 MMB, 3 RMB secondary, 4/5 scroll."""
        self.mouse_buttons.add(button)
        w = self.world
        if self.in_menu:
            if self.phone.is_open and button in (4, 5) and self.phone.active_app.value == "Topographical Map":
                self.phone.handle_input("up" if button == 4 else "down")
            return self._log("menu_click")
        if button == 1:
            res = w.primary_down()
            self.hands.trigger("primary", w.player.primary_hand)
            return self._log("primary_interact", {"result": res})
        if button == 3:
            res = w.secondary_down()
            self.hands.trigger("secondary", w.current_offhand())
            return self._log("secondary_interact", {"result": res})
        if button == 2:
            w.player.held_item_rotation = 0.0
            w.player.held_item_tilt = None
            w.notify("Placement reset: automatic lean / lie across your view.", 1.8)
            return self._log("reset_rotation")
        if button in (4, 5):
            step = 15.0 if button == 4 else -15.0
            if self.shift:
                cur = w.player.held_item_tilt if w.player.held_item_tilt is not None else 0.0
                w.player.held_item_tilt = max(0.0, min(90.0, cur + step))
                return self._log("tilt_item", {"tilt": w.player.held_item_tilt})
            w.player.rotate_held_item(step)
            return self._log("rotate_item", {"rot": w.player.held_item_rotation})
        return self._log("none")

    def handle_mouse_button_up(self, button: int) -> Dict[str, Any]:
        self.mouse_buttons.discard(button)
        w = self.world
        if button == 1:
            w.primary_up()
        elif button == 3:
            w.secondary_up()
        return self._log("mouse_up")

    def handle_mouse_motion(self, dx: float, dy: float) -> Dict[str, Any]:
        w = self.world
        if self.in_menu:
            return {"action": "none"}
        if w.bow_active:
            # sawing the bow: horizontal mouse speed drives the bow stroke (m/s)
            w.bow_input_speed = min(2.0, w.bow_input_speed + abs(dx) * 0.006)
            self.hands.bow_drive(dx)
            dy_eff = dy * 0.2
            dx_eff = 0.0
        else:
            dx_eff, dy_eff = dx, dy
        sens = self.settings.mouse_sensitivity
        inv = -1.0 if not self.settings.invert_y else 1.0
        w.player.rotate_look(dx_eff * sens, inv * dy_eff * sens)
        return {"action": "look"}
