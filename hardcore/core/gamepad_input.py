"""Controller -> game input.

Buttons and triggers act as the keyboard keys they are bound to (so a controller
does exactly what the keys do, through the same 150 ms input buffer).  The move
stick adds real analog movement (any direction, speed follows the tilt) on top of
the digital directions the climbing moves need, and the look stick turns the
camera.  `start` always pauses.

No pygame in here: the device layer feeds it PadState objects, and the headless
tests drive it directly.
"""
from core.gamepad import (PAUSE_CONTROL, StickDirections, LookProcessor, MenuRepeater,
                          radial_deadzone)
from core.input_buffer import Action
from core.keymap import PAUSE_KEY, action_for_key

STICK_DIRECTION_ACTIONS = {
    "forward": Action.FORWARD,
    "back": Action.BACK,
    "left": Action.LEFT,
    "right": Action.RIGHT,
}

PAUSE = "pause"

_MENU_BUTTONS = {"a": "confirm", "b": "back", PAUSE_CONTROL: "back",
                 "dpup": "up", "dpdown": "down", "dpleft": "left", "dpright": "right"}
_MENU_DPAD = ("dpup", "dpdown", "dpleft", "dpright")
MENU_STICK_THRESHOLD = 0.6


class GamepadInput:
    def __init__(self, bindings, router):
        self.bindings = bindings
        self.router = router
        self.directions = StickDirections()
        self.look = LookProcessor()
        self.menu_repeat = MenuRepeater()
        self.suppressed = set()     # controls held through a menu: ignored until released
        self.move = (0.0, 0.0)      # last analog move (right, forward), for the HUD / tests

    # ---------------------------------------------------------------- gameplay
    def on_button(self, control, down):
        """A control changed state during play.

        Returns PAUSE when the pause menu should open, the Action that was pressed (so the
        caller can give feedback, e.g. on the sprint-lock toggle), or None.
        """
        source = ("pad", control)
        if control == PAUSE_CONTROL:
            return PAUSE if down else None
        if control in self.suppressed:
            if not down:
                self.suppressed.discard(control)
            return None
        if not down:
            # release whatever this control holds (robust even if its binding changed meanwhile)
            self.router.release_source(lambda s: s == source)
            return None
        key = self.bindings.key_for(control)
        if not key:
            return None
        if key == PAUSE_KEY:
            return PAUSE
        action = action_for_key(key)
        if action is None:
            return None
        self.router.press(source, action)
        return action

    def update(self, state, dt):
        """Per-frame analog processing. Returns (turn_right_deg, look_up_deg) for the camera."""
        if state is None:
            self._release_stick()
            return 0.0, 0.0
        b = self.bindings
        sx, sy = state.stick(b.move_stick)
        x, y, m = radial_deadzone(sx, sy, b.deadzone)
        y_up = -y
        held = self.directions.update(x, y_up, m)
        for name, action in STICK_DIRECTION_ACTIONS.items():
            self.router.set_held(("stick", name), action, name in held)
        self.router.input_buf.set_move_axis(x, y_up)
        self.move = (x, y_up)
        lx, ly = state.stick(b.look_stick)
        return self.look.update(lx, ly, dt, b.look_sensitivity, b.invert_y, b.deadzone)

    def _release_stick(self):
        self.directions.reset()
        self.look.reset()
        for name, action in STICK_DIRECTION_ACTIONS.items():
            self.router.release(("stick", name), action)
        self.router.input_buf.set_move_axis(0.0, 0.0)
        self.move = (0.0, 0.0)

    def release_all(self):
        """Drop everything the controller holds (pause, device switch, disconnect)."""
        self.router.release_source(lambda s: isinstance(s, tuple) and s[0] in ("pad", "stick"))
        self._release_stick()

    def suppress_held(self, state):
        """Ignore the buttons held right now until they are let go (e.g. A that confirmed
        'Resume' must not jump, and a trigger held through the menu must not sprint)."""
        self.suppressed = set(state.down) if state is not None else set()
        self.suppressed.discard(PAUSE_CONTROL)

    # -------------------------------------------------------------------- menus
    def menu_event(self, control, down):
        """Pad button in a menu -> 'up' / 'down' / 'left' / 'right' / 'confirm' / 'back' / None."""
        if not down:
            return None
        cmd = _MENU_BUTTONS.get(control)
        if control in _MENU_DPAD:
            # the first step happens now; holding it repeats from menu_update
            self.menu_repeat.direction = cmd
            self.menu_repeat.timer = self.menu_repeat.delay
        return cmd

    def menu_update(self, state, dt):
        """Held D-pad / stick direction in a menu -> auto-repeated steps."""
        direction = None
        if state is not None:
            for control in _MENU_DPAD:
                if state.is_down(control):
                    direction = _MENU_BUTTONS[control]
                    break
            if direction is None:
                x, y = state.stick("left")
                if max(abs(x), abs(y)) >= MENU_STICK_THRESHOLD:
                    if abs(y) >= abs(x):
                        direction = "down" if y > 0 else "up"
                    else:
                        direction = "right" if x > 0 else "left"
        return self.menu_repeat.update(direction, dt)


class Haptics:
    """Which physics moments rumble the pad: a short, light buzz - feedback, not noise.

    `on_tick` is called after every fixed physics step and returns (low, high, ms) or None.
    """
    EVENTS = {
        "GRAB": (0.25, 0.55, 90),          # hands close on a hold
        "WALL_KICK": (0.50, 0.30, 110),    # foot plants on the wall
        "DYNO_UP": (0.35, 0.20, 90),
        "DYNO_LATERAL": (0.35, 0.20, 90),
        "WALL_EJECT": (0.40, 0.20, 100),
        "VAULT": (0.15, 0.30, 70),
        "FLIP": (0.10, 0.25, 60),          # tuck
        "ROLL": (0.35, 0.25, 150),         # a big landing, absorbed by the roll
    }
    LANDING_MIN_VY = 6.5                   # m/s downward before a landing is felt

    def on_tick(self, event, was_on_ground, on_ground, prev_vy):
        if event in self.EVENTS:
            return self.EVENTS[event]
        if on_ground and not was_on_ground and prev_vy < -self.LANDING_MIN_VY:
            k = min(1.0, (-prev_vy - self.LANDING_MIN_VY) / 8.0)
            return (0.30 + 0.55 * k, 0.15 + 0.25 * k, int(90 + 90 * k))
        return None
