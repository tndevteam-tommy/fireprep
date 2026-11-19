"""The keyboard map: key names -> actions, and the human-readable names of both.

Keys are identified by their pygame key *name* ("space", "left shift", "e") so the
table needs no pygame import (the headless tests use it) and controller bindings,
which point at keyboard keys, can be stored as readable JSON.  `keycode_map()`
turns the table into pygame key codes once pygame is initialised.
"""
from core.input_buffer import Action

PAUSE_KEY = "escape"

# key name -> action (the game's keyboard layout)
KEYBOARD_ACTIONS = {
    "w": Action.FORWARD,
    "s": Action.BACK,
    "a": Action.LEFT,
    "d": Action.RIGHT,
    "space": Action.JUMP,
    "e": Action.GRAB,
    "left shift": Action.SPRINT,
    "right shift": Action.SPRINT,
    "caps lock": Action.RUN_TOGGLE,
    "left ctrl": Action.LET_GO,
    "c": Action.LET_GO,
}

ACTION_LABELS = {
    Action.FORWARD: "Forward / climb up",
    Action.BACK: "Back / climb down",
    Action.LEFT: "Left / shimmy left",
    Action.RIGHT: "Right / shimmy right",
    Action.JUMP: "Jump / reach / flip",
    Action.GRAB: "Grab / reach (hold)",
    Action.SPRINT: "Sprint (hold)",
    Action.RUN_TOGGLE: "Sprint lock (toggle)",
    Action.LET_GO: "Let go",
}

_SPECIAL_KEY_NAMES = {
    "space": "Space", "left shift": "Left Shift", "right shift": "Right Shift",
    "caps lock": "Caps Lock", "left ctrl": "Left Ctrl", "right ctrl": "Right Ctrl",
    "left alt": "Left Alt", "right alt": "Right Alt", "return": "Enter", "enter": "Keypad Enter",
    "escape": "Esc", "backspace": "Backspace", "tab": "Tab", "up": "Up", "down": "Down",
    "left": "Left", "right": "Right", "left meta": "Left Meta", "right meta": "Right Meta",
    "left gui": "Left Win", "right gui": "Right Win",
}


def action_for_key(key_name):
    """The action a key drives (None if the key does nothing in the game)."""
    if key_name is None:
        return None
    return KEYBOARD_ACTIONS.get(key_name.lower())


def key_display_name(key_name):
    """'left shift' -> 'Left Shift', 'e' -> 'E', None -> 'Unbound'."""
    if not key_name:
        return "Unbound"
    k = key_name.lower()
    if k in _SPECIAL_KEY_NAMES:
        return _SPECIAL_KEY_NAMES[k]
    if len(k) == 1:
        return k.upper()
    return " ".join(part.capitalize() if not part[:1].isdigit() else part for part in k.split(" "))


def key_action_label(key_name):
    """What pressing the key does, for the controller screen ('Pause' / 'No action' included)."""
    if not key_name:
        return "Does nothing"
    if key_name.lower() == PAUSE_KEY:
        return "Pause menu"
    action = action_for_key(key_name)
    return ACTION_LABELS.get(action, "No action in Hardcore")


def keycode_map(pygame_module):
    """{pygame key code: action} for the event loop (call after pygame.init())."""
    out = {}
    for name, action in KEYBOARD_ACTIONS.items():
        out[pygame_module.key.key_code(name)] = action
    return out
