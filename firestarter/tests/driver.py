"""
Headless input driver for gameplay tests.

Feeds key / mouse-button / mouse-motion input into ``InputManager`` exactly the way
``Game.handle_event`` does for real pygame events, and advances the ``GameState`` with a fixed
step - so the gameplay tests exercise the same code paths as a player, without a window.
"""

import math

import numpy as np

BUTTONS = {"left": 1, "middle": 2, "right": 3, "wheel_up": 4, "wheel_down": 5}


class Driver:
    def __init__(self, state, dt=1.0 / 30.0):
        self.st = state
        self.dt = dt

    # ------------------------------------------------------------------ shortcuts
    @property
    def world(self):
        return self.st.world

    @property
    def grid(self):
        return self.st.world.thermal_grid

    @property
    def player(self):
        return self.st.world.player

    def step(self, seconds=None, frames=1):
        n = frames if seconds is None else max(1, int(round(seconds / self.dt)))
        for _ in range(n):
            if not self.st.input.is_paused:
                self.st.step(self.dt)
            self.st.world.pop_events()

    def wait_until(self, pred, timeout):
        t = 0.0
        while t < timeout:
            if pred():
                return t
            self.step(frames=1)
            t += self.dt
        return None

    # ------------------------------------------------------------------ raw input
    def key_down(self, name):
        return self.st.input.handle_key_down(name)

    def key_up(self, name):
        return self.st.input.handle_key_up(name)

    def key(self, name, hold=0.05):
        r = self.key_down(name)
        self.step(hold)
        r2 = self.key_up(name)
        self.step(frames=1)
        return r2 if name == "f" else r

    def mouse_down(self, button):
        return self.st.input.handle_mouse_button_down(BUTTONS.get(button, button))

    def mouse_up(self, button):
        return self.st.input.handle_mouse_button_up(BUTTONS.get(button, button))

    def click(self, button="left", hold=0.05):
        r = self.mouse_down(button)
        self.step(hold)
        self.mouse_up(button)
        self.step(frames=1)
        return r

    def motion(self, dx, dy):
        return self.st.input.handle_mouse_motion(dx, dy)

    # ------------------------------------------------------------------ composite input
    def target_point(self, tgt):
        w = self.world
        if isinstance(tgt, (list, tuple, np.ndarray)):
            return np.array(tgt, dtype=float)
        if "item" in tgt:
            for wi in w.world_items:
                if tgt["item"].lower() in wi.label.lower():
                    return wi.position.copy()
            raise LookupError(tgt)
        if "pit" in tgt:
            return w.fire_pit_pos + np.array(tgt["pit"], dtype=float)
        if "hearthboard" in tgt:
            return w.hearthboard_pos + np.array([0.0, 0.03, 0.0])
        if "rack" in tgt:
            return w.drying_rack.position + np.array([0.0, 0.7, 0.0])
        if "woodpile" in tgt:
            return w.woodpile_pos + np.array([0.0, 0.25, 0.0])
        raise LookupError(tgt)

    def _aim_deltas(self, point):
        p = self.player
        d = point - p.eye
        yaw = math.degrees(math.atan2(d[2], d[0]))
        pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        dyaw = (yaw - p.yaw + 180.0) % 360.0 - 180.0
        dpitch = pitch - p.pitch
        sens = self.st.settings.mouse_sensitivity
        inv = 1.0 if self.st.settings.invert_y else -1.0
        return dyaw / sens, (dpitch / sens) * inv

    def aim_at(self, tgt, duration=0.3):
        pt = self.target_point(tgt)
        n = max(1, int(duration / self.dt))
        for i in range(n):
            dx, dy = self._aim_deltas(pt)
            self.motion(dx / (n - i), dy / (n - i))
            self.step(frames=1)
        dx, dy = self._aim_deltas(pt)
        self.motion(dx, dy)
        self.step(frames=1)

    def walk_to(self, tgt, stop=1.2, timeout=15.0):
        pt = self.target_point(tgt)
        self.key_down("w")
        t = 0.0
        try:
            while t < timeout:
                pl = self.player
                if math.hypot(pt[0] - pl.position[0], pt[2] - pl.position[2]) <= stop:
                    return True
                dx, _ = self._aim_deltas(np.array([pt[0], pl.eye[1], pt[2]]))
                self.motion(dx * 0.5, 0.0)
                self.step(frames=1)
                t += self.dt
            return False
        finally:
            self.key_up("w")
            self.step(frames=1)

    def saw(self, seconds, amplitude=150.0, rate=1.4):
        phase = 0.0
        t = 0.0
        while t < seconds:
            phase += self.dt * rate * 2 * math.pi
            self.motion(math.cos(phase) * amplitude * rate * 2 * math.pi * self.dt, 0.0)
            self.step(frames=1)
            t += self.dt
