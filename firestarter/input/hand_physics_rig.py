"""
Firestarter - Procedural Hand Physics Rig
Implements DESIGN Section 8.1 HandController: two procedurally animated first-person hands.

Each frame a *target pose* (position in camera space, Euler rotation, finger curl) is chosen
for each hand from the current activity (idle, walking, lighter, whittling, chopping, reaching
for a pickup / placing fuel, bow drilling, blowing into a nest, twisting cordage, pouring) and
the actual pose follows it through critically-damped springs, which gives the organic weight
and follow-through of real hands.  A two-link arm solution provides the elbow position so the
renderer can draw a forearm from the wrist back toward the shoulder.

Camera space: x = right, y = up, z = forward (metres).
"""

import math
from typing import Dict, Any, Optional

import numpy as np


class HandJointState:
    def __init__(self, position: np.ndarray, rotation: np.ndarray, flex: float = 0.0):
        self.position = position
        self.rotation = rotation
        self.flex = flex


class _Spring:
    def __init__(self, value, stiffness=90.0, damping=None):
        self.x = np.array(value, dtype=float)
        self.v = np.zeros_like(self.x)
        self.k = stiffness
        self.c = damping if damping is not None else 2.0 * math.sqrt(stiffness)

    def step(self, target, dt):
        target = np.asarray(target, dtype=float)
        steps = max(1, int(dt / 0.008) + 1)
        h = dt / steps
        for _ in range(steps):
            a = self.k * (target - self.x) - self.c * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x


REST_RIGHT = np.array([0.22, -0.165, 0.46])
REST_LEFT = np.array([-0.23, -0.18, 0.44])


class HandController:
    """Procedural animation solver for the player's two hands."""

    def __init__(self):
        self.rest_left = REST_LEFT.copy()
        self.rest_right = REST_RIGHT.copy()
        self.pos_left = self.rest_left.copy()
        self.pos_right = self.rest_right.copy()
        self.rot_left = np.zeros(3)
        self.rot_right = np.zeros(3)
        self.flex_left = 0.25
        self.flex_right = 0.25
        self._sp_pl = _Spring(self.rest_left, 110.0)
        self._sp_pr = _Spring(self.rest_right, 110.0)
        self._sp_rl = _Spring(np.zeros(3), 80.0)
        self._sp_rr = _Spring(np.zeros(3), 80.0)
        self._sp_fl = _Spring([0.25], 60.0)
        self._sp_fr = _Spring([0.25], 60.0)

        self.time = 0.0
        self.bow_stroke_phase = 0.0
        self.bow_offset = 0.0
        self.bow_velocity = 0.0
        self.whittle_phase = 0.0
        self.strike_phase = 0.0
        self.blow_phase = 0.0
        self.cordage_twist_phase = 0.0
        self.action: Optional[str] = None
        self.action_time = 0.0
        self.action_duration = 0.0
        self.right_item = "pocket_lighter"
        self.left_item = "tinder"
        self.lighter_lit = False
        self.reach_target: Optional[np.ndarray] = None   # camera-space point for reaching
        self.walk_phase = 0.0
        self.walk_amount = 0.0
        self.crouch = 0.0

    # ------------------------------------------------------------------ triggers
    def trigger(self, hand: str, item: str):
        if hand == "primary":
            if item == "felling_axe":
                self.start_action("chop", 0.75)
            elif item == "pocket_knife":
                self.start_action("whittle", 1.1)
            elif item == "pocket_lighter":
                self.start_action("strike", 0.35)
            elif item == "accelerant":
                self.start_action("pour", 1.0)
            else:
                self.start_action("reach", 0.5)
        elif hand == "secondary":
            self.start_action("place", 0.55)

    def start_action(self, name: str, duration: float):
        self.action = name
        self.action_time = 0.0
        self.action_duration = duration

    def bow_drive(self, dx: float):
        self.bow_velocity += dx * 0.0009

    def solve_ik_reach(self, target_pos: np.ndarray, is_right_hand: bool = True):
        """Two-link reach: moves the wrist to target (clamped to arm length)."""
        shoulder = np.array([0.2 if is_right_hand else -0.2, -0.18, 0.0])
        d = np.asarray(target_pos, dtype=float) - shoulder
        L = np.linalg.norm(d)
        max_len = 0.66
        if L > max_len:
            d = d / L * max_len
        tgt = shoulder + d
        if is_right_hand:
            self.pos_right = tgt
        else:
            self.pos_left = tgt
        return tgt

    def elbow_position(self, wrist: np.ndarray, is_right: bool) -> np.ndarray:
        shoulder = np.array([0.24 if is_right else -0.24, -0.30, -0.05])
        upper, fore = 0.32, 0.30
        d = wrist - shoulder
        L = float(np.clip(np.linalg.norm(d), 0.05, upper + fore - 0.005))
        dn = d / max(1e-6, np.linalg.norm(d))
        a = (upper * upper - fore * fore + L * L) / (2 * L)
        h = math.sqrt(max(0.0, upper * upper - a * a))
        pole = np.array([0.55 if is_right else -0.55, -1.0, -0.2])
        pole = pole - dn * float(np.dot(pole, dn))
        pole = pole / max(1e-6, np.linalg.norm(pole))
        return shoulder + dn * a + pole * h

    # ------------------------------------------------------------------ update
    def update(self, dt: float, current_action: str = "idle", context: Optional[Dict[str, Any]] = None):
        ctx = context or {}
        self.time += dt
        self.right_item = ctx.get("right_item", self.right_item)
        self.left_item = ctx.get("left_item", self.left_item)
        self.lighter_lit = ctx.get("lighter_lit", False)
        self.reach_target = ctx.get("reach_target")
        self.walk_phase = ctx.get("walk_phase", self.walk_phase)
        self.walk_amount = ctx.get("walk_amount", 0.0)
        self.crouch = ctx.get("crouch", 0.0)
        if self.action is not None:
            self.action_time += dt
            if self.action_time >= self.action_duration:
                self.action = None

        t = self.time
        breathe = math.sin(t * 1.6) * 0.004
        bob_x = math.sin(self.walk_phase) * 0.012 * self.walk_amount
        bob_y = -abs(math.cos(self.walk_phase)) * 0.016 * self.walk_amount
        pl = self.rest_left + np.array([-bob_x, bob_y + breathe, 0.0])
        pr = self.rest_right + np.array([bob_x, bob_y + breathe * 1.2, 0.0])
        # Euler (pitch, yaw, roll) in the hand frame: right fist thumb-up around its tool,
        # left palm turned up, cupped under whatever fuel it carries.
        rl = np.array([-8.0, 18.0, 165.0])
        rr = np.array([8.0, 0.0, -80.0])
        fl, fr = 0.38, 0.86

        if self.right_item == "none":
            # empty right hand: relaxed and lowered out of view
            fr = 0.35
            rr = np.array([-5.0, -12.0, -100.0])
            pr = pr + np.array([0.06, -0.17, -0.05])
        elif self.right_item == "felling_axe":
            pr = pr + np.array([0.03, -0.06, -0.04])
            rr = np.array([8.0, 0.0, -82.0])
            fr = 0.95
        elif self.right_item == "pocket_knife":
            rr = np.array([18.0, -12.0, -70.0])
            fr = 0.9
        elif self.right_item == "bow_drill_spindle":
            rr = np.array([8.0, -20.0, -60.0])
            fr = 0.85
        elif self.right_item == "accelerant":
            pr = pr + np.array([0.02, -0.03, 0.0])
            fr = 0.92
        if self.left_item in ("tinder", "fungus", "nest"):
            fl = 0.42
        elif self.left_item in ("kindling", "firewood"):
            fl = 0.8
        elif self.left_item == "log":
            pl = pl + np.array([0.03, -0.02, 0.02])
            fl = 0.72
        elif self.left_item == "none":
            fl = 0.3
            rl = np.array([-5.0, 12.0, 100.0])
            pl = pl + np.array([-0.06, -0.16, -0.05])     # lowered out of view

        activity = current_action
        if activity == "bow_drilling":
            # kneeling: left hand presses the bearing stone, right hand saws the bow
            self.bow_velocity *= math.exp(-4.0 * dt)
            self.bow_offset += self.bow_velocity * dt * 60.0
            self.bow_offset -= self.bow_offset * min(1.0, dt * 1.2)
            self.bow_offset = float(np.clip(self.bow_offset, -0.16, 0.16))
            self.bow_stroke_phase += abs(self.bow_velocity) * dt * 40.0
            grip = ctx.get("drill_grip")
            if grip is not None:
                g = np.asarray(grip, dtype=float)
                pl = g + np.array([0.0, 0.035, 0.0])          # palm on the bearing block
                pr = g + np.array([0.13 + self.bow_offset, -0.10, -0.02])
            else:
                pl = np.array([-0.03, -0.20, 0.50])
                pr = np.array([0.13 + self.bow_offset, -0.28, 0.48])
            rl = np.array([60.0, 0.0, 10.0])
            fl = 0.95
            rr = np.array([10.0, 0.0, -85.0])
            fr = 0.9
        elif activity == "hand_drilling":
            # palms flat against a vertical spindle, rubbing back and forth while sliding down it
            hh = float(np.clip(ctx.get("hand_height", 1.0), 0.0, 1.0))
            spd = float(ctx.get("drill_speed", 0.0))
            self.bow_stroke_phase += dt * (6.0 + 10.0 * min(1.5, spd))
            rub = math.sin(self.bow_stroke_phase) * 0.03 * min(1.0, spd * 1.2)
            grip = ctx.get("drill_grip")
            if grip is not None:
                g = np.asarray(grip, dtype=float)
                pl = g + np.array([-0.036, 0.0, rub])
                pr = g + np.array([0.036, 0.0, -rub])
            else:
                y = -0.30 + 0.22 * hh
                pl = np.array([-0.038, y, 0.43 + rub])
                pr = np.array([0.038, y, 0.43 - rub])
            rl = np.array([10.0, 5.0, 88.0])
            rr = np.array([10.0, -5.0, -88.0])
            fl = fr = 0.25
        elif activity == "blowing" or self.left_item == "nest":
            pl = np.array([-0.06, -0.12, 0.30])
            pr = np.array([0.06, -0.12, 0.30])
            rl = np.array([-20.0, 30.0, 150.0])
            rr = np.array([-20.0, -30.0, -150.0])
            fl = fr = 0.55
        elif activity == "twisting_cordage":
            self.cordage_twist_phase += 11.0 * dt
            roll = math.sin(self.cordage_twist_phase) * 40.0
            pl = np.array([-0.07, -0.20, 0.38])
            pr = np.array([0.07, -0.20, 0.38])
            rl = np.array([20.0, 20.0, 90.0 - roll])
            rr = np.array([20.0, -20.0, -90.0 + roll])
            fl = fr = 0.7
        elif activity == "whittling":
            self.whittle_phase = (self.whittle_phase + 2.6 * dt) % 1.0
            w = self.whittle_phase
            pl = np.array([-0.10, -0.18, 0.42])
            rl = np.array([10.0, 30.0, 110.0])
            fl = 0.9
            pr = np.array([-0.03 + 0.05 * w, -0.12 - 0.10 * w, 0.44 - 0.02 * w])
            rr = np.array([40.0, -40.0, -60.0])
            fr = 0.95

        # lighter held out toward the fuel
        if self.right_item == "pocket_lighter" and self.lighter_lit:
            target = np.array([0.14, -0.18, 0.55])
            if self.reach_target is not None:
                rt = np.asarray(self.reach_target, dtype=float)
                dir_ = rt / max(1e-6, np.linalg.norm(rt))
                target = np.array([0.10, -0.12, 0.0]) + dir_ * 0.55
                target[0] = max(-0.05, target[0])
            pr = target
            rr = np.array([5.0, -10.0, -80.0])
            fr = 0.88

        # one-shot actions layered on top
        if self.action is not None:
            k = self.action_time / max(1e-6, self.action_duration)
            env = math.sin(math.pi * min(1.0, k))
            if self.action == "chop":
                if k < 0.35:       # wind-up over the shoulder
                    s = k / 0.35
                    pr = pr + np.array([0.08 * s, 0.28 * s, -0.12 * s])
                    rr = rr + np.array([-80.0 * s, 0.0, 10.0 * s])
                elif k < 0.55:     # strike
                    s = (k - 0.35) / 0.2
                    pr = pr + np.array([0.08 - 0.2 * s, 0.28 - 0.45 * s, -0.12 + 0.3 * s])
                    rr = rr + np.array([-80.0 + 140.0 * s, 0.0, 10.0])
                else:              # recover
                    s = (k - 0.55) / 0.45
                    pr = pr + np.array([-0.12 * (1 - s), -0.17 * (1 - s), 0.18 * (1 - s)])
                    rr = rr + np.array([60.0 * (1 - s), 0.0, 10.0 * (1 - s)])
                pl = pl + np.array([0.10, 0.05, 0.0]) * env
            elif self.action in ("reach", "place"):
                reach = np.array([0.10 if self.action == "reach" else -0.05, -0.30, 0.74])
                if self.action == "reach":
                    pr = pr + (reach - pr) * env
                    fr = fr + (0.9 - fr) * (1.0 if k > 0.5 else 0.0)
                else:
                    pl = pl + (reach - pl) * env
                    fl = fl * (1.0 - 0.6 * env)
            elif self.action == "strike":
                pr = pr + np.array([0.0, 0.03, 0.02]) * env
                rr = rr + np.array([0.0, 0.0, -25.0]) * env
            elif self.action == "whittle":
                pl = pl + (np.array([-0.10, -0.22, 0.42]) - pl) * env
                pr = pr + (np.array([0.0, -0.2, 0.44]) - pr) * env
            elif self.action == "pour":
                pr = pr + np.array([-0.08, 0.06, 0.12]) * env
                rr = rr + np.array([0.0, 0.0, -75.0]) * env

        # crouching brings the hands a touch lower and further out
        pr = pr + np.array([0.0, -0.02 * self.crouch, 0.03 * self.crouch])
        pl = pl + np.array([0.0, -0.02 * self.crouch, 0.03 * self.crouch])

        self.pos_left = self._sp_pl.step(pl, dt).copy()
        self.pos_right = self._sp_pr.step(pr, dt).copy()
        self.rot_left = self._sp_rl.step(rl, dt).copy()
        self.rot_right = self._sp_rr.step(rr, dt).copy()
        self.flex_left = float(self._sp_fl.step([fl], dt)[0])
        self.flex_right = float(self._sp_fr.step([fr], dt)[0])
        self.blow_phase = max(0.0, self.blow_phase - dt)

    def get_hand_render_matrices(self) -> Dict[str, Any]:
        return {
            "left_pos": self.pos_left.copy(),
            "left_rot": self.rot_left.copy(),
            "left_flex": self.flex_left,
            "left_elbow": self.elbow_position(self.pos_left, False),
            "right_pos": self.pos_right.copy(),
            "right_rot": self.rot_right.copy(),
            "right_flex": self.flex_right,
            "right_elbow": self.elbow_position(self.pos_right, True),
            "right_item": self.right_item,
            "left_item": self.left_item,
            "lighter_lit": self.lighter_lit,
            "action": self.action,
            "bow_offset": self.bow_offset,
        }
