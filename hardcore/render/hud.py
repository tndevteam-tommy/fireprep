"""Short on-screen notices ("Sprint lock ON", "Controller connected: ...").

The timing logic (`ToastQueue`) is pure so it can be tested headlessly; `ToastHud`
draws the messages through the text shader as fading pills at the top of the view.
"""
import config

try:
    import pygame
    from OpenGL.GL import (glActiveTexture, glBindTexture, GL_TEXTURE0, GL_TEXTURE_2D, glDisable, glEnable,
                           GL_DEPTH_TEST, GL_CULL_FACE)
except ImportError:          # pragma: no cover
    pygame = None


def _ui_scale(w, h):
    dw, dh = config.UI_DESIGN_SIZE
    return max(0.5, min(w / float(dw), h / float(dh)))


class ToastQueue:
    FADE_IN = 0.12
    FADE_OUT = 0.45

    def __init__(self, duration=2.2, max_items=3):
        self.duration = duration
        self.max_items = max_items
        self.items = []           # [text, age, key]

    def push(self, text, key=None):
        """Show `text`; a message with the same key replaces the previous one (e.g. a toggle)."""
        key = key or text
        self.items = [it for it in self.items if it[2] != key]
        self.items.append([text, 0.0, key])
        self.items = self.items[-self.max_items:]

    def update(self, dt):
        for it in self.items:
            it[1] += dt
        self.items = [it for it in self.items if it[1] < self.duration]

    def alpha(self, age):
        if age < self.FADE_IN:
            return age / self.FADE_IN
        remaining = self.duration - age
        if remaining < self.FADE_OUT:
            return max(0.0, remaining / self.FADE_OUT)
        return 1.0

    def visible(self):
        """[(text, alpha)] newest last."""
        return [(it[0], self.alpha(it[1])) for it in self.items]


class ToastHud:
    def __init__(self, menu, screen_w=config.WINDOW_WIDTH, screen_h=config.WINDOW_HEIGHT):
        # borrows the pause menu's quad / label helpers (same text shader)
        self.menu = menu
        self.screen_w, self.screen_h = screen_w, screen_h
        self.scale = _ui_scale(screen_w, screen_h)
        self.queue = ToastQueue()
        self.labels = {}

    def resize(self, screen_w, screen_h):
        self.screen_w, self.screen_h = screen_w, screen_h
        self.scale = _ui_scale(screen_w, screen_h)
        for old in self.labels.values():
            old.release()
        self.labels.clear()

    def push(self, text, key=None):
        self.queue.push(text, key)

    def _label(self, text):
        from render.menu import TextLabel
        label = self.labels.get(text)
        if label is None:
            if len(self.labels) > 32:
                for old in self.labels.values():
                    old.release()
                self.labels.clear()
            label = self.labels[text] = TextLabel(text, max(10, int(round(30 * self.scale))), (245, 246, 248))
        return label

    def draw(self, text_shader, dt):
        self.queue.update(dt)
        items = self.queue.visible()
        if not items:
            return
        from math3d.transform import ortho
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        text_shader.use()
        text_shader.set_mat4("u_Proj", ortho(0.0, self.screen_w, self.screen_h, 0.0, -1.0, 1.0))
        text_shader.set_int("u_Tex", 0)
        y = self.screen_h * 0.06
        k = self.scale
        for text, a in reversed(items):
            label = self._label(text)
            x = self.screen_w * 0.5 - label.width * 0.5
            self.menu._draw_label(text_shader, self.menu._blank, x - 16 * k, y - 8 * k, (0.05, 0.06, 0.08, 0.72 * a),
                                  label.width + 32 * k, label.height + 16 * k)
            self.menu._draw_label(text_shader, label, x, y, (1.0, 1.0, 1.0, a))
            y += label.height + 24 * k
        glBindTexture(GL_TEXTURE_2D, 0)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
