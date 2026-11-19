"""Routes presses from several input sources into the one InputBuffer.

The keyboard, a controller's buttons and its stick can all hold the same action
(W and the left stick both hold FORWARD).  Every press registers a fresh,
bufferable press, but the action is only released when the *last* source
holding it lets go - releasing W while the stick is still pushed keeps you
walking.

Sources are any hashable ids, e.g. ("key", 32), ("pad", "a"), ("stick", "forward").
"""


class InputRouter:
    def __init__(self, input_buf, scheme_fn=None):
        self.input_buf = input_buf
        self.scheme_fn = scheme_fn or (lambda: None)
        self.holders = {}         # action -> set(source)

    def press(self, source, action):
        """A source pressed an action. Returns True if the source was not already holding it."""
        holders = self.holders.setdefault(action, set())
        if source in holders:
            return False
        holders.add(source)
        self.input_buf.set_key_down(action)
        scheme = self.scheme_fn()
        if scheme is not None:
            scheme.on_key_down(action)
        return True

    def release(self, source, action):
        holders = self.holders.get(action)
        if not holders or source not in holders:
            return False
        holders.discard(source)
        if not holders:
            self.input_buf.set_key_up(action)
        return True

    def set_held(self, source, action, held):
        """Level-triggered helper (stick directions): press / release on change only."""
        if held:
            return self.press(source, action)
        return self.release(source, action)

    def is_held_by(self, source, action):
        return source in self.holders.get(action, ())

    def release_source(self, predicate):
        """Release every action held by sources matching predicate(source)."""
        for action, holders in list(self.holders.items()):
            for source in [s for s in holders if predicate(s)]:
                self.release(source, action)

    def release_all(self):
        for action in list(self.holders):
            for source in list(self.holders[action]):
                self.release(source, action)
        for action in list(self.input_buf.held_keys):
            self.input_buf.set_key_up(action)
