"""Gamepad model - controls, controller families, bindings and signal processing.

Everything here is pure Python (no pygame), so it is shared by the device layer
(`core/gamepad_device.py`), the game glue (`core/gamepad_input.py`), the
controller screen and the headless tests.

Controls are named after SDL's GameController *positions* ("a" is the bottom face
button on every pad: A on Xbox, Cross on PlayStation, B on Nintendo).  A control
is bound to a keyboard key name; pressing the control behaves exactly like
pressing that key.  The sticks are analog: one moves (with real analog speed and
direction), the other looks; `start` is always the pause button.
"""
import json
import math
import os

import config

# --------------------------------------------------------------------------- controls
# SDL_GameControllerButton order (index -> name)
SDL_BUTTONS = ("a", "b", "x", "y", "back", "guide", "start", "leftstick", "rightstick",
               "leftshoulder", "rightshoulder", "dpup", "dpdown", "dpleft", "dpright",
               "misc1", "paddle1", "paddle2", "paddle3", "paddle4", "touchpad")
# SDL_GameControllerAxis order (index -> name)
SDL_AXES = ("leftx", "lefty", "rightx", "righty", "lefttrigger", "righttrigger")
TRIGGERS = ("lefttrigger", "righttrigger")
STICK_AXES = {"left": ("leftx", "lefty"), "right": ("rightx", "righty")}

PAUSE_CONTROL = "start"              # always opens the pause menu (never rebindable)
SYSTEM_CONTROLS = ("guide",)         # owned by the OS / Steam / console overlay

# Bindable controls in the order the controller screen walks through them
BINDABLE_CONTROLS = ("lefttrigger", "leftshoulder", "back", "leftstick",
                     "dpup", "dpleft", "dpright", "dpdown",
                     "righttrigger", "rightshoulder", "y", "b", "x", "a", "rightstick",
                     "misc1", "touchpad", "paddle1", "paddle2", "paddle3", "paddle4")

DEFAULT_BINDINGS = {
    "a": "space",              # jump / hold to reach & catch
    "x": "e",                  # grab
    "b": "left ctrl",          # let go
    "lefttrigger": "left shift",   # sprint while held
    "leftstick": "caps lock",      # sprint lock (click the left stick)
    "righttrigger": "e",       # reach with the trigger so the right thumb can keep aiming
    "dpup": "w",
    "dpdown": "s",
    "dpleft": "a",
    "dpright": "d",
}


class Family:
    XBOX = "xbox"
    PLAYSTATION = "playstation"
    NINTENDO = "nintendo"
    GENERIC = "generic"
    ALL = (XBOX, PLAYSTATION, NINTENDO, GENERIC)


FAMILY_NAMES = {
    Family.XBOX: "Xbox",
    Family.PLAYSTATION: "PlayStation",
    Family.NINTENDO: "Nintendo",
    Family.GENERIC: "Generic",
}

# (short label drawn on the pad, long name for text)
_COMMON = {
    "leftstick": ("L3", "Left stick click"),
    "rightstick": ("R3", "Right stick click"),
    "dpup": ("Up", "D-pad up"),
    "dpdown": ("Down", "D-pad down"),
    "dpleft": ("Left", "D-pad left"),
    "dpright": ("Right", "D-pad right"),
    "paddle1": ("P1", "Paddle 1"),
    "paddle2": ("P2", "Paddle 2"),
    "paddle3": ("P3", "Paddle 3"),
    "paddle4": ("P4", "Paddle 4"),
    "touchpad": ("Pad", "Touchpad click"),
}
_LABELS = {
    Family.XBOX: {
        "a": ("A", "A"), "b": ("B", "B"), "x": ("X", "X"), "y": ("Y", "Y"),
        "back": ("View", "View"), "start": ("Menu", "Menu"), "guide": ("Xbox", "Xbox button"),
        "leftshoulder": ("LB", "LB"), "rightshoulder": ("RB", "RB"),
        "lefttrigger": ("LT", "LT"), "righttrigger": ("RT", "RT"),
        "leftstick": ("LS", "Left stick click"), "rightstick": ("RS", "Right stick click"),
        "misc1": ("Share", "Share"),
    },
    Family.PLAYSTATION: {
        "a": ("Cross", "Cross"), "b": ("Circle", "Circle"), "x": ("Square", "Square"),
        "y": ("Triangle", "Triangle"),
        "back": ("Create", "Create / Share"), "start": ("Options", "Options"), "guide": ("PS", "PS button"),
        "leftshoulder": ("L1", "L1"), "rightshoulder": ("R1", "R1"),
        "lefttrigger": ("L2", "L2"), "righttrigger": ("R2", "R2"),
        "misc1": ("Mic", "Mute"), "touchpad": ("Touchpad", "Touchpad click"),
    },
    Family.NINTENDO: {   # positional: the bottom button of a Nintendo pad is labelled B
        "a": ("B", "B"), "b": ("A", "A"), "x": ("Y", "Y"), "y": ("X", "X"),
        "back": ("-", "Minus"), "start": ("+", "Plus"), "guide": ("Home", "Home"),
        "leftshoulder": ("L", "L"), "rightshoulder": ("R", "R"),
        "lefttrigger": ("ZL", "ZL"), "righttrigger": ("ZR", "ZR"),
        "misc1": ("Capture", "Capture"),
    },
    Family.GENERIC: {
        "a": ("A", "A (bottom)"), "b": ("B", "B (right)"), "x": ("X", "X (left)"), "y": ("Y", "Y (top)"),
        "back": ("Select", "Select / Back"), "start": ("Start", "Start"), "guide": ("Home", "Home"),
        "leftshoulder": ("LB", "LB / L1"), "rightshoulder": ("RB", "RB / R1"),
        "lefttrigger": ("LT", "LT / L2"), "righttrigger": ("RT", "RT / R2"),
        "misc1": ("Misc", "Misc"),
    },
}

# Face-button colours (RGB) per family, drawn on the controller screen
FACE_COLORS = {
    Family.XBOX: {"a": (96, 184, 72), "b": (214, 62, 58), "x": (56, 124, 220), "y": (230, 190, 50)},
    Family.PLAYSTATION: {"a": (120, 160, 230), "b": (230, 90, 100), "x": (220, 130, 200), "y": (80, 200, 170)},
    Family.NINTENDO: {"a": (200, 200, 205), "b": (200, 200, 205), "x": (200, 200, 205), "y": (200, 200, 205)},
    Family.GENERIC: {"a": (190, 190, 196), "b": (190, 190, 196), "x": (190, 190, 196), "y": (190, 190, 196)},
}


def control_label(control, family=Family.XBOX, long=False):
    labels = _LABELS.get(family, _LABELS[Family.GENERIC])
    pair = labels.get(control) or _COMMON.get(control) or (control, control)
    return pair[1] if long else pair[0]


def pad_menu_hint(family):
    """The pause menu's controller line, in the pad's own button names."""
    a, b, start = (control_label(c, family) for c in ("a", "b", "start"))
    return (f"Controller: D-pad or left stick to choose   -   {a} to confirm   -   "
            f"{b} or {start} to resume")


def controls_for_family(family):
    """The bindable controls shown for a controller family (touchpad / mic only where they exist)."""
    out = []
    for c in BINDABLE_CONTROLS:
        if c == "touchpad" and family != Family.PLAYSTATION:
            continue
        if c.startswith("paddle") or c == "misc1":
            continue           # rare extras: still bindable from the file, not drawn on the diagram
        out.append(c)
    return out


# --------------------------------------------------------------------- family detection
_VENDOR_FAMILY = {
    0x045E: Family.XBOX,          # Microsoft
    0x0E6F: Family.XBOX,          # PDP (licensed Xbox pads)
    0x24C6: Family.XBOX,          # PowerA
    0x054C: Family.PLAYSTATION,   # Sony
    0x057E: Family.NINTENDO,      # Nintendo
}
_NAME_HINTS = (
    (Family.PLAYSTATION, ("playstation", "dualshock", "dualsense", "ps3", "ps4", "ps5", "sony")),
    (Family.NINTENDO, ("nintendo", "switch", "joy-con", "joycon", "wii u", "gamecube")),
    (Family.XBOX, ("xbox", "x-box", "xinput", "microsoft")),
)


def parse_guid(guid):
    """(vendor, product) from an SDL joystick GUID string, or (None, None)."""
    if not guid or len(guid) < 20:
        return None, None
    try:
        raw = bytes.fromhex(guid[:32])
    except ValueError:
        return None, None
    # SDL 2.0.x GUIDs: bus(2) crc(2) vendor(2) 0(2) product(2) 0(2) version(2) driver(2)
    if raw[6:8] != b"\x00\x00" or raw[10:12] != b"\x00\x00":
        return None, None
    vendor = raw[4] | (raw[5] << 8)
    product = raw[8] | (raw[9] << 8)
    if vendor == 0:
        return None, None
    return vendor, product


def detect_family(name, guid=None):
    """Xbox / PlayStation / Nintendo / generic from the device name, then the USB vendor id."""
    low = (name or "").lower()
    for family, hints in _NAME_HINTS:
        if any(h in low for h in hints):
            return family
    vendor, _ = parse_guid(guid)
    if vendor in _VENDOR_FAMILY:
        return _VENDOR_FAMILY[vendor]
    return Family.GENERIC


# -------------------------------------------------------------------- raw joystick layout
def raw_layout(num_buttons, num_axes, num_hats):
    """Best-guess layout for a joystick SDL has no gamepad mapping for (cheap USB pads).

    Most such pads follow one of two orders; the controller screen shows what lights up,
    and every button can be rebound, so a wrong guess is easy to see and fix.
    """
    buttons = {0: "a", 1: "b", 2: "x", 3: "y", 4: "leftshoulder", 5: "rightshoulder"}
    if num_buttons >= 12:
        # "PS-style" DirectInput pads: separate L2/R2 buttons
        buttons.update({6: "lefttrigger", 7: "righttrigger", 8: "back", 9: "start",
                        10: "leftstick", 11: "rightstick", 12: "guide"})
    else:
        buttons.update({6: "back", 7: "start", 8: "leftstick", 9: "rightstick", 10: "guide"})
    if num_axes >= 6:
        axes = {0: "leftx", 1: "lefty", 2: "lefttrigger", 3: "rightx", 4: "righty", 5: "righttrigger"}
    elif num_axes >= 4:
        axes = {0: "leftx", 1: "lefty", 2: "rightx", 3: "righty"}
    else:
        axes = {i: n for i, n in enumerate(("leftx", "lefty")[:num_axes])}
    buttons = {i: c for i, c in buttons.items() if i < num_buttons}
    return {"buttons": buttons, "axes": axes, "hat": 0 if num_hats > 0 else None}


# ----------------------------------------------------------------------- pad state
class PadState:
    """Live state of one controller: pressed controls (triggers included) and axis values.

    Sticks are in [-1, 1] with SDL's convention (y grows *down*); triggers in [0, 1].
    Triggers turn into digital presses with hysteresis so a resting finger never flickers.
    """

    def __init__(self):
        self.down = set()
        self.axes = {a: 0.0 for a in SDL_AXES}

    def set_button(self, control, down):
        """Returns [(control, down)] when the state changed, else []."""
        if down and control not in self.down:
            self.down.add(control)
            return [(control, True)]
        if not down and control in self.down:
            self.down.discard(control)
            return [(control, False)]
        return []

    def set_axis(self, axis, value):
        """Store an axis value; returns trigger press/release transitions."""
        if axis in TRIGGERS:
            value = min(1.0, max(0.0, float(value)))
        else:
            value = min(1.0, max(-1.0, float(value)))
        self.axes[axis] = value
        if axis in TRIGGERS:
            held = axis in self.down
            if not held and value >= config.PAD_TRIGGER_PRESS:
                return self.set_button(axis, True)
            if held and value < config.PAD_TRIGGER_RELEASE:
                return self.set_button(axis, False)
        return []

    def is_down(self, control):
        return control in self.down

    def stick(self, side):
        ax, ay = STICK_AXES[side]
        return self.axes[ax], self.axes[ay]

    def clear(self):
        """Release everything (device lost); returns the release transitions."""
        out = [(c, False) for c in sorted(self.down)]
        self.down.clear()
        for a in self.axes:
            self.axes[a] = 0.0
        return out


# ------------------------------------------------------------------ signal processing
def radial_deadzone(x, y, deadzone, outer=None):
    """Radial deadzone with rescaling: returns (x, y, magnitude), magnitude in [0, 1].

    Inside the deadzone the stick is at rest; beyond it the output ramps from 0 so there
    is no jump, and anything past `outer` is full deflection.  Square-gated pads that
    report |v| > 1 on the diagonals are clamped to the unit circle.
    """
    outer = config.PAD_STICK_OUTER if outer is None else outer
    mag = math.hypot(x, y)
    if mag <= deadzone or mag < 1e-9:
        return 0.0, 0.0, 0.0
    m = min(1.0, (mag - deadzone) / max(1e-6, outer - deadzone))
    return x / mag * m, y / mag * m, m


class StickDirections:
    """Digital directions from an analog stick, for the actions that need a 'key'.

    A direction is held while the stick is pushed past PAD_DIR_PRESS within
    PAD_DIR_HALF_ANGLE of its axis, so the diagonals hold both neighbours (like W+D)
    and the cardinal bands are wide enough that pushing "up" to climb never shimmies.
    Magnitude and angle both have hysteresis.
    """
    AXES = {"forward": (0.0, 1.0), "back": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}

    def __init__(self):
        self.held = set()

    def update(self, x, y_up, magnitude):
        held = set()
        for name, (ax, ay) in self.AXES.items():
            was = name in self.held
            threshold = config.PAD_DIR_RELEASE if was else config.PAD_DIR_PRESS
            if magnitude < threshold:
                continue
            n = math.hypot(x, y_up)
            if n < 1e-9:
                continue
            cos_a = (x * ax + y_up * ay) / n
            limit = config.PAD_DIR_HALF_ANGLE_HOLD if was else config.PAD_DIR_HALF_ANGLE
            if cos_a >= math.cos(math.radians(limit)):
                held.add(name)
        self.held = held
        return set(held)

    def reset(self):
        self.held = set()


class LookProcessor:
    """Right stick -> camera turn rates: deadzone, response curve and a rim turn boost."""

    def __init__(self):
        self.rim_time = 0.0

    def reset(self):
        self.rim_time = 0.0

    def update(self, x, y, dt, sensitivity=1.0, invert_y=False, deadzone=None):
        """Returns (turn_right_deg, look_up_deg) for this frame (y is SDL's: down positive)."""
        deadzone = config.PAD_STICK_DEADZONE if deadzone is None else deadzone
        rx, ry, m = radial_deadzone(x, y, deadzone)
        if m <= 0.0:
            self.rim_time = 0.0
            return 0.0, 0.0
        curved = m ** config.PAD_LOOK_EXPONENT
        scale = curved / m
        if m >= 0.97:
            self.rim_time += dt
        else:
            self.rim_time = 0.0
        t = (self.rim_time - config.PAD_LOOK_BOOST_DELAY) / max(1e-6, config.PAD_LOOK_BOOST_RAMP)
        t = min(1.0, max(0.0, t))
        boost = 1.0 + config.PAD_LOOK_BOOST * (t * t * (3.0 - 2.0 * t))
        turn = rx * scale * config.PAD_LOOK_YAW_RATE * sensitivity * boost * dt
        up = -ry * scale * config.PAD_LOOK_PITCH_RATE * sensitivity * boost * dt
        if invert_y:
            up = -up
        return turn, up


class MenuRepeater:
    """Held menu direction -> one step now, then auto-repeat after a delay."""

    def __init__(self, delay=None, rate=None):
        self.delay = config.PAD_MENU_REPEAT_DELAY if delay is None else delay
        self.rate = config.PAD_MENU_REPEAT_RATE if rate is None else rate
        self.direction = None
        self.timer = 0.0

    def update(self, direction, dt):
        """direction: 'up' / 'down' / 'left' / 'right' / None.  Returns the steps to apply."""
        if direction is None:
            self.direction = None
            return []
        if direction != self.direction:
            self.direction = direction
            self.timer = self.delay
            return [direction]
        self.timer -= dt
        steps = []
        while self.timer <= 0.0:
            steps.append(direction)
            self.timer += self.rate
        return steps


# ---------------------------------------------------------------------- bindings
def default_settings_path():
    """Per-user controller.json (HARDCORE_CONFIG_DIR overrides the directory)."""
    base = os.environ.get("HARDCORE_CONFIG_DIR")
    if not base:
        if os.name == "nt":
            base = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Hardcore")
        else:
            xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
            base = os.path.join(xdg, "hardcore")
    return os.path.join(base, "controller.json")


def _clamp(v, lo, hi):
    return min(hi, max(lo, v))


class ControllerBindings:
    """Control -> keyboard key bindings plus the player's controller settings."""

    VERSION = 1

    def __init__(self):
        self.keys = {}
        self.reset()

    # ------------------------------------------------------------------ defaults
    def reset(self):
        self.keys = {c: DEFAULT_BINDINGS.get(c) for c in BINDABLE_CONTROLS}
        self.look_sensitivity = 1.0
        self.invert_y = False
        self.swap_sticks = False          # southpaw: right stick moves, left stick looks
        self.deadzone = config.PAD_STICK_DEADZONE
        self.vibration = True

    # ------------------------------------------------------------------- binding
    def key_for(self, control):
        return self.keys.get(control)

    def bind(self, control, key_name):
        if control not in BINDABLE_CONTROLS:
            raise ValueError(f"'{control}' cannot be rebound")
        self.keys[control] = key_name.lower() if key_name else None

    def clear(self, control):
        self.bind(control, None)

    def controls_for_key(self, key_name):
        key_name = (key_name or "").lower()
        return [c for c in BINDABLE_CONTROLS if self.keys.get(c) == key_name]

    @property
    def move_stick(self):
        return "right" if self.swap_sticks else "left"

    @property
    def look_stick(self):
        return "left" if self.swap_sticks else "right"

    # ------------------------------------------------------------------ settings
    def adjust_sensitivity(self, steps):
        lo, hi = config.PAD_LOOK_SENSITIVITY_RANGE
        self.look_sensitivity = round(_clamp(self.look_sensitivity + 0.1 * steps, lo, hi), 2)

    def adjust_deadzone(self, steps):
        lo, hi = config.PAD_DEADZONE_RANGE
        self.deadzone = round(_clamp(self.deadzone + 0.01 * steps, lo, hi), 2)

    # --------------------------------------------------------------- persistence
    def to_dict(self):
        return {
            "version": self.VERSION,
            "bindings": {c: self.keys.get(c) for c in BINDABLE_CONTROLS},
            "look_sensitivity": self.look_sensitivity,
            "invert_y": self.invert_y,
            "swap_sticks": self.swap_sticks,
            "deadzone": self.deadzone,
            "vibration": self.vibration,
        }

    @classmethod
    def from_dict(cls, data):
        """Lenient: unknown controls / bad values are ignored and fall back to the defaults."""
        b = cls()
        if not isinstance(data, dict):
            return b
        binds = data.get("bindings")
        if isinstance(binds, dict):
            for control, key in binds.items():
                if control in BINDABLE_CONTROLS and (key is None or (isinstance(key, str) and key.strip())):
                    b.keys[control] = key.lower() if key else None

        def num(name, lo, hi, default):
            v = data.get(name, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return default
            return float(_clamp(v, lo, hi))

        b.look_sensitivity = num("look_sensitivity", *config.PAD_LOOK_SENSITIVITY_RANGE, b.look_sensitivity)
        b.deadzone = num("deadzone", *config.PAD_DEADZONE_RANGE, b.deadzone)
        for flag in ("invert_y", "swap_sticks", "vibration"):
            if isinstance(data.get(flag), bool):
                setattr(b, flag, data[flag])
        return b

    def save(self, path=None):
        """Write the JSON settings; returns False (and keeps playing) if the disk says no."""
        path = path or default_settings_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            os.replace(tmp, path)
            return True
        except OSError as exc:
            print(f"[controller] could not save settings to {path}: {exc}")
            return False

    @classmethod
    def load(cls, path=None):
        path = path or default_settings_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError) as exc:
            print(f"[controller] ignoring unreadable settings {path}: {exc}")
            return cls()
