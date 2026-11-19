"""Control schemes: map raw keys to parkour *intents*.

The state machine never reads keys directly.  Every tick the active scheme
resolves the InputBuffer into an `Intents` object; the FSM asks for intents by
name and *consumes* the underlying buffered presses only when it actually acts
on them, so the 150 ms input buffer keeps working for every scheme.

``FlowScheme`` is the game's control scheme.  One "parkour button" (Space):
jump on the ground, *hold to reach* in the air (hands close on any hold within
reach only while it is held, so letting go mid-air is a deliberate - or
accidental - miss), pressed again in the air after a jump a flip (when no hold
is within reach - parkour/acrobatics.py), dyno in the held direction while
hanging.  Climbing is just
pushing W hold-to-hold, a top edge included: keep W held and you go straight
over it after the same short pause every hold gets; let go of W and you hang
there until you press it again.  Climbing down is S,
shimmying is A/D, Ctrl lets go, E is an explicit grab / climb-down-from-an-edge.
Caps Lock toggles run/walk (Shift is a momentary run override).
"""
import config
from core.input_buffer import Action


class ControlSchemeName:
    FLOW = "flow"


class Intents:
    """Resolved player intent for one tick."""

    def __init__(self):
        self.forward = False
        self.back = False
        self.left = False
        self.right = False
        self.run = False               # sprint / run mode active
        self.jump_held = False         # sustains the variable-jump lift
        self.wall_kick = False         # plant a foot when hitting a wall mid-air
        self.auto_step_max = 0.0       # tallest obstacle stepped onto automatically
        self.shimmy = 0.0              # -1 left, +1 right
        self.climb_up = False          # continuous: climb hold-to-hold while hanging
        # Analog movement (controller stick): (right, forward) direction + tilt in [0, 1].
        # None = digital keys; move_amount is then 1 (a key is always "fully pushed").
        self.move_analog = None
        self.move_amount = 1.0
        self._flags = {}
        self._consumers = {}

    def set(self, name, value, consumer=None):
        self._flags[name] = bool(value)
        if consumer is not None:
            self._consumers[name] = consumer

    def has(self, name):
        return self._flags.get(name, False)

    def consume(self, name):
        fn = self._consumers.get(name)
        if fn is not None:
            fn()
        self._flags[name] = False


class ControlScheme:
    name = "base"
    description = ""

    def __init__(self):
        self.run_mode = False

    def on_key_down(self, action):
        """Edge-triggered hooks (toggles). Called by the window layer on KEYDOWN."""

    def resolve(self, input_buf):
        raise NotImplementedError

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _movement(intents, buf):
        intents.forward = buf.is_down(Action.FORWARD)
        intents.back = buf.is_down(Action.BACK)
        intents.left = buf.is_down(Action.LEFT)
        intents.right = buf.is_down(Action.RIGHT)
        analog = getattr(buf, "move_axis", None)
        if analog is not None:
            intents.move_analog = analog
            intents.move_amount = min(1.0, (analog[0] * analog[0] + analog[1] * analog[1]) ** 0.5)

    @staticmethod
    def _fresh(buf, action):
        return buf.has_action(action, consume=False)


class FlowScheme(ControlScheme):
    """One parkour button, push-to-climb, Caps Lock run toggle."""
    name = ControlSchemeName.FLOW
    description = ("Space jump / hold to reach & catch / again in the air to flip / dyno while hanging | "
                   "W climb | S climb down | "
                   "A/D shimmy | Ctrl let go | E grab / climb down an edge | Caps Lock run toggle (Shift momentary)")

    def __init__(self, run_mode=False):
        super().__init__()
        self.run_mode = bool(run_mode)

    def on_key_down(self, action):
        if action == Action.RUN_TOGGLE:
            self.run_mode = not self.run_mode

    def set_run_mode(self, enabled):
        self.run_mode = bool(enabled)

    def resolve(self, buf):
        it = Intents()
        self._movement(it, buf)
        it.run = self.run_mode or buf.is_down(Action.SPRINT)
        it.jump_held = buf.is_down(Action.JUMP)
        jump_fresh = self._fresh(buf, Action.JUMP)
        reach = buf.held_or_buffered(Action.JUMP) or buf.held_or_buffered(Action.GRAB)

        # Reaching: hands only close on a hold while the parkour button (or E) is held.
        it.set("grab", reach, lambda: (buf.consume(Action.JUMP), buf.consume(Action.GRAB)))
        it.set("jump", jump_fresh, lambda: buf.consume(Action.JUMP))
        it.wall_kick = reach
        # Running flows over hip-high obstacles; walking only steps up knee-high ones
        # (a stick eased forward below the direction threshold still steps up)
        forwardish = it.forward or (it.move_analog is not None and it.move_analog[1] > 0.25)
        if forwardish:
            it.auto_step_max = config.AUTO_STEP_MAX_HEIGHT if it.run else config.AUTO_STEP_WALK_MAX
        else:
            it.auto_step_max = 0.0

        # Hanging: push to climb hold-to-hold (a top edge is climbed the same way - W held carries on
        # over it; `mantle` is the fresh press that restarts after a pause)
        it.climb_up = it.forward
        it.set("mantle", self._fresh(buf, Action.FORWARD), lambda: buf.consume(Action.FORWARD))
        it.shimmy = float(it.right) - float(it.left)
        lateral = it.left != it.right
        it.set("dyno_left", jump_fresh and it.left and not it.right, lambda: buf.consume(Action.JUMP))
        it.set("dyno_right", jump_fresh and it.right and not it.left, lambda: buf.consume(Action.JUMP))
        it.set("eject", jump_fresh and it.back and not lateral, lambda: buf.consume(Action.JUMP))
        it.set("dyno_up", jump_fresh and not it.back and not lateral, lambda: buf.consume(Action.JUMP))
        it.set("climb_down", self._fresh(buf, Action.BACK), lambda: buf.consume(Action.BACK))
        it.set("let_go", self._fresh(buf, Action.LET_GO), lambda: buf.consume(Action.LET_GO))
        # Standing at a drop: a fresh E lowers you over the edge into a hang
        it.set("climb_down_edge", self._fresh(buf, Action.GRAB), lambda: buf.consume(Action.GRAB))
        return it


def make_scheme(name=None, run_mode=False):
    name = (name or config.CONTROL_SCHEME).lower()
    if name == ControlSchemeName.FLOW:
        return FlowScheme(run_mode=run_mode)
    raise ValueError(f"unknown control scheme '{name}'")
