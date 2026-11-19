"""Timestamped input buffering queue for combos and responsive parkour.

The buffer distinguishes between *held* keys and *fresh presses*.  One-shot
actions (a dyno, a drop) must be driven by a fresh press of their trigger key so
that simply holding a chord while hanging does not re-fire the action every
tick.  Continuous actions (climbing, shimmying) accept held keys.

The clock is injectable so the buffer can be driven deterministically by a
headless simulation.
"""
import time
import config


class Action:
    JUMP = "JUMP"
    GRAB = "GRAB"
    FORWARD = "FORWARD"
    BACK = "BACK"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    SPRINT = "SPRINT"
    RUN_TOGGLE = "RUN_TOGGLE"   # Caps Lock (flow scheme)
    LET_GO = "LET_GO"           # Ctrl / C (flow scheme)


class InputBuffer:
    def __init__(self, buffer_time=config.INPUT_BUFFER_TIME, time_fn=None):
        self.buffer_time = buffer_time
        self.time_fn = time_fn if time_fn is not None else time.time
        self.queue = []  # list of [action, timestamp, consumed]
        self.held_keys = set()
        # Analog movement from a controller stick: (right, forward) with magnitude in (0, 1],
        # or None when no stick is pushed (the digital keys then drive movement)
        self.move_axis = None

    # ------------------------------------------------------------------ input
    def set_key_down(self, action):
        now = self.time_fn()
        self.held_keys.add(action)
        self.queue.append([action, now, False])

    def set_key_up(self, action):
        self.held_keys.discard(action)

    def is_down(self, action):
        return action in self.held_keys

    def set_move_axis(self, x, y):
        """Analog stick movement: x = right, y = forward, |(x, y)| <= 1.  (0, 0) clears it."""
        mag = (x * x + y * y) ** 0.5
        if mag < 1e-6:
            self.move_axis = None
        else:
            if mag > 1.0:
                x, y = x / mag, y / mag
            self.move_axis = (float(x), float(y))

    def update(self):
        now = self.time_fn()
        self.queue = [item for item in self.queue if (now - item[1]) <= self.buffer_time and not item[2]]

    # ---------------------------------------------------------------- queries
    def _find_fresh(self, action):
        now = self.time_fn()
        for item in reversed(self.queue):
            if item[0] == action and not item[2] and (now - item[1]) <= self.buffer_time:
                return item
        return None

    def has_action(self, action, consume=True):
        """True if `action` was pressed within the buffer window (fresh press)."""
        item = self._find_fresh(action)
        if item is None:
            return False
        if consume:
            item[2] = True
        return True

    def has_fresh(self, action, consume=True):
        return self.has_action(action, consume)

    def held_or_buffered(self, action):
        """Modifier semantics: key is currently held or was tapped recently."""
        return action in self.held_keys or self._find_fresh(action) is not None

    def consume(self, action):
        for item in self.queue:
            if item[0] == action:
                item[2] = True

    def clear(self):
        self.queue.clear()
        self.move_axis = None
