"""Controller screen: layout geometry and the interaction model (no rendering, no pygame).

The screen shows a picture of the connected controller.  Every control lights up
while it is pressed; selecting one (click it or its label, or move to it and
press Enter) waits for a keyboard key, and that key becomes what the control
does.  Pressing a button on the controller moves the selection to it.  A panel
on the right holds the stick / vibration settings.

`ControllerLayout` computes every rectangle in screen pixels so the renderer
(`render/controller_view.py`) and the mouse hit-testing use the same geometry.
"""
import math

from core.gamepad import (Family, PAUSE_CONTROL, controls_for_family, control_label, BINDABLE_CONTROLS)
from core.keymap import key_display_name, key_action_label, PAUSE_KEY

DESIGN_W, DESIGN_H = 1280.0, 720.0

OPTIONS = ("sensitivity", "invert_y", "swap_sticks", "deadzone", "vibration", "reset", "back")
OPTION_LABELS = {
    "sensitivity": "Look sensitivity",
    "invert_y": "Invert look (Y)",
    "swap_sticks": "Swap sticks",
    "deadzone": "Stick deadzone",
    "vibration": "Vibration",
    "reset": "Reset to defaults",
    "back": "Back",
}
SLIDERS = ("sensitivity", "deadzone")
TOGGLES = ("invert_y", "swap_sticks", "vibration")


class ControllerLayout:
    """Screen geometry of the controller picture, the label boxes and the settings panel."""

    def __init__(self, family=Family.XBOX, screen_w=DESIGN_W, screen_h=DESIGN_H):
        self.family = family
        self.screen_w, self.screen_h = float(screen_w), float(screen_h)
        self.s = min(self.screen_w / DESIGN_W, self.screen_h / DESIGN_H)
        self.ox = (self.screen_w - DESIGN_W * self.s) * 0.5
        self.oy = (self.screen_h - DESIGN_H * self.s) * 0.5
        self._build()

    # design space -> screen
    def p(self, x, y):
        return (self.ox + x * self.s, self.oy + y * self.s)

    def r(self, x, y, w, h):
        return (self.ox + x * self.s, self.oy + y * self.s, w * self.s, h * self.s)

    def _build(self):
        cx, cy = 475.0, 370.0
        self.center = self.p(cx, cy)
        ps = self.family == Family.PLAYSTATION
        # --- body (design-space primitives; the view scales them)
        self.body_rect = self.r(cx - 140, cy - 95, 280, 140)
        self.body_radius = 55 * self.s
        self.grips = [(self.p(cx - 112, cy + 55), (62 * self.s, 100 * self.s), 0.42),
                      (self.p(cx + 112, cy + 55), (62 * self.s, 100 * self.s), -0.42)]

        c = {}

        def circle(name, x, y, radius):
            c[name] = {"shape": "circle", "center": self.p(x, y), "r": radius * self.s}

        def rect(name, x, y, w, h, radius=4):
            c[name] = {"shape": "rect", "rect": self.r(x - w / 2, y - h / 2, w, h), "radius": radius * self.s,
                       "center": self.p(x, y)}

        rect("leftshoulder", cx - 97, cy - 107, 105, 22, 10)
        rect("rightshoulder", cx + 97, cy - 107, 105, 22, 10)
        rect("lefttrigger", cx - 97, cy - 147, 62, 50, 14)
        rect("righttrigger", cx + 97, cy - 147, 62, 50, 14)
        if ps:
            dpad, face = (cx - 88, cy - 45), (cx + 88, cy - 45)
            lstick, rstick = (cx - 45, cy + 25), (cx + 45, cy + 25)
            rect("touchpad", cx, cy - 63, 96, 58, 8)
            rect("back", cx - 68, cy - 80, 11, 20, 4)
            rect("start", cx + 68, cy - 80, 11, 20, 4)
            circle("guide", cx, cy + 8, 11)
        else:
            lstick, dpad = (cx - 88, cy - 45), (cx - 45, cy + 22)
            rstick, face = (cx + 45, cy + 22), (cx + 88, cy - 45)
            rect("back", cx - 22, cy - 52, 20, 13, 6)
            rect("start", cx + 22, cy - 52, 20, 13, 6)
            circle("guide", cx, cy - 80, 13)
        circle("leftstick", lstick[0], lstick[1], 30)
        circle("rightstick", rstick[0], rstick[1], 30)
        off = 27
        circle("y", face[0], face[1] - off, 13)
        circle("b", face[0] + off, face[1], 13)
        circle("a", face[0], face[1] + off, 13)
        circle("x", face[0] - off, face[1], 13)
        arm, half = 23, 10.5
        dx, dy = dpad
        rect("dpup", dx, dy - arm * 0.5 - half * 0.5 + 2, half * 2, arm, 3)
        rect("dpdown", dx, dy + arm * 0.5 + half * 0.5 - 2, half * 2, arm, 3)
        rect("dpleft", dx - arm * 0.5 - half * 0.5 + 2, dy, arm, half * 2, 3)
        rect("dpright", dx + arm * 0.5 + half * 0.5 - 2, dy, arm, half * 2, 3)
        self.dpad_center = self.p(dx, dy)
        self.dpad_half = half * self.s
        self.dpad_arm = arm * self.s
        self.controls = c

        # --- label boxes ("callouts") down both sides, ordered top to bottom to keep leaders tidy
        shown = [k for k in controls_for_family(self.family) if k in c]
        centre_x = self.center[0]
        left = sorted([k for k in shown if c[k]["center"][0] < centre_x], key=lambda k: (c[k]["center"][1], c[k]["center"][0]))
        right = sorted([k for k in shown if c[k]["center"][0] >= centre_x] + [PAUSE_CONTROL],
                       key=lambda k: (c[k]["center"][1], -c[k]["center"][0]))
        self.columns = (left, right)
        self.callouts = {}
        box_w, box_h, step, y0 = 215, 42, 50, 150
        for col, names, x in ((0, left, 40), (1, right, 695)):
            n = len(names)
            start = y0 + max(0, (8 - n)) * step * 0.5
            for i, name in enumerate(names):
                self.callouts[name] = {"rect": self.r(x, start + i * step, box_w, box_h), "side": col}

        # --- settings panel
        self.panel_rect = self.r(935, 120, 310, 450)
        self.options = {}
        for i, name in enumerate(OPTIONS):
            row = self.r(950, 172 + i * 54, 280, 44)
            entry = {"rect": row}
            if name in SLIDERS:
                entry["minus"] = self.r(950 + 280 - 120, 172 + i * 54 + 6, 32, 32)
                entry["plus"] = self.r(950 + 280 - 36, 172 + i * 54 + 6, 32, 32)
            self.options[name] = entry

        self.title_pos = self.p(60, 34)
        self.device_pos = self.p(62, 96)
        self.status_rect = self.r(40, 598, 870, 40)
        self.hint_pos = self.p(640, 686)

    # ---------------------------------------------------------------- queries
    def anchor(self, control):
        """Point on the control a label's leader line attaches to."""
        return self.controls[control]["center"]

    def hit(self, pos):
        """Item under a screen point: ('control', name) / ('option', name[, '-'|'+']) / None."""
        x, y = pos

        def in_rect(rc, pad=0.0):
            return rc[0] - pad <= x <= rc[0] + rc[2] + pad and rc[1] - pad <= y <= rc[1] + rc[3] + pad

        for name, entry in self.options.items():
            for sign in ("minus", "plus"):
                if sign in entry and in_rect(entry[sign], 2):
                    return ("option", name, "-" if sign == "minus" else "+")
            if in_rect(entry["rect"]):
                return ("option", name)
        for name, entry in self.callouts.items():
            if in_rect(entry["rect"]):
                return ("control", name)
        for name, geo in self.controls.items():
            if geo["shape"] == "circle":
                cxp, cyp = geo["center"]
                if math.hypot(x - cxp, y - cyp) <= geo["r"] + 3 * self.s:
                    return ("control", name)
            elif in_rect(geo["rect"], 3 * self.s):
                return ("control", name)
        return None


class ControllerMenuModel:
    """Focus / waiting-for-a-key state machine of the controller screen."""
    BACK = "back"

    def __init__(self, bindings, screen_w=DESIGN_W, screen_h=DESIGN_H, on_change=None):
        self.bindings = bindings
        self.screen_w, self.screen_h = screen_w, screen_h
        self.on_change = on_change or (lambda b: None)
        self.visible = False
        self.family = None
        self.focus = None              # ('control', name) or ('option', name)
        self.waiting = None            # control waiting for a keyboard key
        self.status = ""
        self.status_kind = "info"      # info / ok / warn / prompt
        self.device_name = None
        self.device_recognised = True
        self.set_family(Family.XBOX)

    # ------------------------------------------------------------------ setup
    def set_family(self, family):
        if family == self.family:
            return
        self.family = family
        self.layout = ControllerLayout(family, self.screen_w, self.screen_h)
        left, right = self.layout.columns
        self.items = ([("control", c) for c in left] + [("control", c) for c in right] +
                      [("option", o) for o in OPTIONS])
        if self.focus is not None and self.focus not in self.items:
            self.focus = self.items[0]
        if self.waiting is not None and ("control", self.waiting) not in self.items:
            self.waiting = None

    def resize(self, screen_w, screen_h):
        """New screen size: the same screen, laid out again (focus and the key wait are kept)."""
        self.screen_w, self.screen_h = screen_w, screen_h
        self.layout = ControllerLayout(self.family, screen_w, screen_h)

    def set_device(self, name, family, recognised=True):
        self.device_name = name
        self.device_recognised = bool(recognised)
        self.set_family(family or Family.XBOX)

    def open(self):
        self.visible = True
        self.waiting = None
        self.focus = ("control", "a") if ("control", "a") in self.items else self.items[0]
        self._say("Press any button on your controller to find it here. Select a button, then press a keyboard key.")

    def close(self):
        self.visible = False
        self.waiting = None

    # ---------------------------------------------------------------- helpers
    def _say(self, text, kind="info"):
        self.status = text
        self.status_kind = kind

    def describe(self, control):
        """'A: Space - Jump / reach & catch' for the status line."""
        label = control_label(control, self.family)
        if control == PAUSE_CONTROL:
            return f"{label}: always opens the pause menu"
        key = self.bindings.key_for(control)
        if not key:
            return f"{label}: unbound"
        return f"{label}: {key_display_name(key)} - {key_action_label(key)}"

    def is_bindable(self, control):
        return control in BINDABLE_CONTROLS and ("control", control) in self.items

    def _changed(self):
        self.on_change(self.bindings)

    # -------------------------------------------------------------- selection
    def select(self, item):
        """Activate an item (Enter / click): a control starts waiting for a key."""
        kind, name = item[0], item[1]
        self.focus = (kind, name)
        if kind == "control":
            if not self.is_bindable(name):
                self.waiting = None
                self._say(self.describe(name) + " (it cannot be rebound)", "warn")
                return None
            self.waiting = name
            self._say(f"Press a keyboard key for {control_label(name, self.family)}   -   "
                      f"Esc cancels, Backspace unbinds", "prompt")
            return None
        return self._activate_option(name, item[2] if len(item) > 2 else None)

    def _activate_option(self, name, sign=None):
        b = self.bindings
        if name == "back":
            return self.BACK
        if name == "reset":
            b.reset()
            self._changed()
            self._say("Controller bindings and settings reset to the defaults.", "ok")
        elif name in TOGGLES:
            setattr(b, name, not getattr(b, name))
            self._changed()
            self._say(self.option_value(name, long=True), "ok")
        elif name in SLIDERS:
            self.adjust(name, -1 if sign == "-" else 1)
        return None

    def adjust(self, name, steps):
        b = self.bindings
        if name == "sensitivity":
            b.adjust_sensitivity(steps)
        elif name == "deadzone":
            b.adjust_deadzone(steps)
        elif name in TOGGLES:
            setattr(b, name, not getattr(b, name))
        else:
            return
        self._changed()
        self._say(self.option_value(name, long=True), "ok")

    def option_value(self, name, long=False):
        b = self.bindings
        if name == "sensitivity":
            v = f"{b.look_sensitivity:.1f}x"
            return f"Look sensitivity {v}" if long else v
        if name == "deadzone":
            v = f"{int(round(b.deadzone * 100))}%"
            return f"Stick deadzone {v}" if long else v
        if name == "invert_y":
            return ("Look up/down inverted" if b.invert_y else "Look up/down normal") if long else (
                "On" if b.invert_y else "Off")
        if name == "swap_sticks":
            if long:
                return f"{b.move_stick.capitalize()} stick moves, {b.look_stick} stick looks"
            return "On" if b.swap_sticks else "Off"
        if name == "vibration":
            return ("Vibration on" if b.vibration else "Vibration off") if long else ("On" if b.vibration else "Off")
        return ""

    # ------------------------------------------------------------------ input
    def handle_key(self, key_name):
        """A keyboard key (pygame key name). Returns BACK to leave the screen, else None."""
        if not self.visible or not key_name:
            return None
        k = key_name.lower()
        if self.waiting is not None:
            control = self.waiting
            label = control_label(control, self.family)
            self.waiting = None
            if k == PAUSE_KEY:
                self._say(f"Cancelled - {self.describe(control)}")
                return None
            if k in ("backspace", "delete"):
                self.bindings.clear(control)
                self._changed()
                self._say(f"{label} unbound - it does nothing now.", "ok")
                return None
            self.bindings.bind(control, k)
            self._changed()
            others = [control_label(c, self.family) for c in self.bindings.controls_for_key(k)
                      if c != control and ("control", c) in self.items]
            also = f"   (also on {', '.join(others)})" if others else ""
            self._say(f"{label} -> {key_display_name(k)}: {key_action_label(k)}{also}", "ok")
            return None
        if k == PAUSE_KEY:
            return self.BACK
        if k in ("up", "w"):
            self._move(-1)
        elif k in ("down", "s", "tab"):
            self._move(1)
        elif k in ("left", "a"):
            self.horizontal(-1)
        elif k in ("right", "d"):
            self.horizontal(1)
        elif k in ("return", "enter", "space"):
            if self.focus is not None:
                return self.select(self.focus)
        elif k in ("backspace", "delete") and self.focus and self.focus[0] == "control":
            if self.is_bindable(self.focus[1]):
                self.bindings.clear(self.focus[1])
                self._changed()
                self._say(f"{control_label(self.focus[1], self.family)} unbound - it does nothing now.", "ok")
        return None

    def _move(self, step):
        if self.focus not in self.items:
            self.focus = self.items[0]
            return
        i = self.items.index(self.focus)
        self.focus = self.items[(i + step) % len(self.items)]
        self._describe_focus()

    def horizontal(self, step):
        """Left / right: adjust a setting, or hop between the label columns and the panel."""
        if self.focus is None:
            self.focus = self.items[0]
            return
        kind, name = self.focus[0], self.focus[1]
        if kind == "option":
            if name in SLIDERS or name in TOGGLES:
                self.adjust(name, step)
            elif step < 0:
                self.focus = ("control", self.layout.columns[1][0])
            return
        left, right = self.layout.columns
        col = 0 if name in left else 1
        row = (left if col == 0 else right).index(name)
        if step > 0 and col == 0:
            self.focus = ("control", right[min(row, len(right) - 1)])
        elif step > 0 and col == 1:
            self.focus = ("option", OPTIONS[min(row, len(OPTIONS) - 1)])
        elif step < 0 and col == 1:
            self.focus = ("control", left[min(row, len(left) - 1)])
        self._describe_focus()

    def _describe_focus(self, click=False):
        if self.focus and self.focus[0] == "control":
            if self.focus[1] == PAUSE_CONTROL:
                self._say(self.describe(PAUSE_CONTROL))
            else:
                self._say(self.describe(self.focus[1]) + ("   -   click to change" if click else "   -   Enter to change"))
        elif self.focus and self.focus[1] in SLIDERS + TOGGLES:
            self._say(self.option_value(self.focus[1], long=True) +
                      ("   -   click to change" if click else "   -   Left / Right to change"))

    def handle_mouse(self, kind, pos):
        """kind: 'move' / 'click'. Returns BACK if 'Back' was clicked."""
        if not self.visible:
            return None
        hit = self.layout.hit(pos)
        if kind == "move":
            if hit is not None and self.waiting is None:
                item = (hit[0], hit[1])
                if item in self.items and item != self.focus:
                    self.focus = item
                    self._describe_focus(click=True)
            return None
        if kind == "click":
            if hit is None:
                if self.waiting is not None:
                    self.waiting = None
                    self._say("Cancelled.")
                return None
            if hit[0] == "control" and hit[1] == PAUSE_CONTROL:
                self.focus = ("control", PAUSE_CONTROL) if ("control", PAUSE_CONTROL) in self.items else self.focus
                self._say(self.describe(PAUSE_CONTROL) + " (it cannot be rebound)", "warn")
                return None
            if hit[0] == "control" and ("control", hit[1]) not in self.items:
                return None      # e.g. the guide button: owned by the OS
            return self.select(hit)
        return None

    def handle_wheel(self, steps):
        if self.focus and self.focus[0] == "option" and self.focus[1] in SLIDERS:
            self.adjust(self.focus[1], steps)

    def handle_pad(self, control, down):
        """A controller button: focus it (the picture lights it up by itself). Start leaves."""
        if not self.visible or not down:
            return None
        if control == PAUSE_CONTROL:
            if self.waiting is None:
                return self.BACK
            return None
        if ("control", control) in self.items:
            if self.waiting is None:
                self.focus = ("control", control)
                self._say(self.describe(control) + "   -   Enter or click to change")
        return None
