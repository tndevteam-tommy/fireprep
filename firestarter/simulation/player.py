"""
Firestarter - Player Entity (first-person body)
Movement over the procedural terrain, look orientation, crouching for close bushcraft work,
stamina, head-bob and footstep events.
"""

import math
from typing import Optional, List, Tuple, Callable

import numpy as np

from firestarter.simulation import terrain

EYE_HEIGHT = 1.66
CROUCH_EYE_HEIGHT = 0.95


class Player:
    def __init__(self, position: Optional[np.ndarray] = None):
        if position is None:
            position = np.array([0.0, 0.0, 2.6])
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.yaw = -90.0     # facing -Z (north) toward the fire pit
        self.pitch = -24.0

        self.walk_speed = 2.6
        self.sprint_speed = 5.0
        self.crouch_speed = 1.2
        self.is_sprinting = False
        self.is_crouching = False
        self.eye_height = EYE_HEIGHT
        self.stamina = 100.0
        self.max_stamina = 100.0
        self.exertion = 0.0           # seconds of recent hard work (no stamina regeneration)

        # legacy equipment names (kept for compatibility with older scripts / tests)
        self.primary_hand = "pocket_lighter"
        self.secondary_hand = "tinder"
        self.held_item_rotation = 0.0     # yaw of the fuel piece to be placed (deg)
        self.held_item_tilt = None        # None = automatic (lean into a teepee / lie flat)

        self.bob_phase = 0.0
        self.bob_amount = 0.0
        self.step_events: List[str] = []
        self._step_accum = 0.0
        self.in_water = False
        self.colliders: List[Tuple[float, float, float]] = []   # (x, z, radius)
        self.lean = np.zeros(3)             # head offset when leaning over work (hand drilling)
        self.lean_target = np.zeros(3)
        self.snap_to_ground()

    # ------------------------------------------------------------------ helpers
    def ground_height(self) -> float:
        return float(terrain.height(self.position[0], self.position[2]))

    def snap_to_ground(self):
        self.position[1] = self.ground_height() + self.eye_height

    @property
    def feet(self) -> np.ndarray:
        return np.array([self.position[0], self.ground_height(), self.position[2]])

    @property
    def eye(self) -> np.ndarray:
        bob = math.sin(self.bob_phase * 2.0) * 0.022 * self.bob_amount
        return self.position + np.array([0.0, bob, 0.0]) + self.lean

    # ------------------------------------------------------------------ movement
    def move(self, forward: float, strafe: float, dt: float):
        yaw_rad = math.radians(self.yaw)
        fwd = np.array([math.cos(yaw_rad), 0.0, math.sin(yaw_rad)])
        right = np.array([-math.sin(yaw_rad), 0.0, math.cos(yaw_rad)])
        moving = abs(forward) > 1e-3 or abs(strafe) > 1e-3
        if self.is_crouching:
            speed = self.crouch_speed
        elif self.is_sprinting and self.stamina > 5.0 and forward > 0:
            speed = self.sprint_speed
        else:
            speed = self.walk_speed
        self.exertion = max(0.0, self.exertion - dt)
        if self.is_sprinting and moving and not self.is_crouching:
            self.stamina = max(0.0, self.stamina - 14.0 * dt)
        elif self.exertion <= 0.0:
            self.stamina = min(self.max_stamina, self.stamina + 9.0 * dt)

        move_vec = fwd * forward + right * strafe
        norm = float(np.linalg.norm(move_vec))
        if norm > 1e-4:
            move_vec = move_vec / norm * speed
        # slope & water resistance
        slope = terrain.slope_degrees(self.position[0], self.position[2])
        self.in_water = terrain.is_in_water(self.position[0], self.position[2])
        if self.in_water:
            move_vec *= 0.5
        if slope > 25.0:
            move_vec *= max(0.35, 1.0 - (slope - 25.0) / 30.0)
        # smooth acceleration
        target = move_vec
        self.velocity[[0, 2]] += (target[[0, 2]] - self.velocity[[0, 2]]) * min(1.0, 10.0 * dt)
        new_pos = self.position + np.array([self.velocity[0], 0.0, self.velocity[2]]) * dt
        new_pos[0] = max(-terrain.WALKABLE_HALF_EXTENT, min(terrain.WALKABLE_HALF_EXTENT, new_pos[0]))
        new_pos[2] = max(-terrain.WALKABLE_HALF_EXTENT, min(terrain.WALKABLE_HALF_EXTENT, new_pos[2]))
        new_pos = self._resolve_collisions(new_pos)
        self.position[0], self.position[2] = new_pos[0], new_pos[2]

        target_eye = CROUCH_EYE_HEIGHT if self.is_crouching else EYE_HEIGHT
        self.eye_height += (target_eye - self.eye_height) * min(1.0, 8.0 * dt)
        self.position[1] = self.ground_height() + self.eye_height

        spd = float(np.hypot(self.velocity[0], self.velocity[2]))
        self.bob_amount += ((1.0 if spd > 0.3 else 0.0) - self.bob_amount) * min(1.0, 6.0 * dt)
        self.bob_phase += dt * (4.2 if spd < 3.5 else 6.0) * min(1.5, spd / 2.6 + 0.2)
        self._step_accum += spd * dt
        stride = 0.75 if not self.is_crouching else 0.45
        if self._step_accum > stride:
            self._step_accum -= stride
            self.step_events.append("water" if self.in_water else "step")

    def _resolve_collisions(self, p: np.ndarray) -> np.ndarray:
        for (cx, cz, r) in self.colliders:
            dx, dz = p[0] - cx, p[2] - cz
            d = math.hypot(dx, dz)
            rr = r + 0.28
            if d < rr and d > 1e-6:
                p[0] = cx + dx / d * rr
                p[2] = cz + dz / d * rr
        return p

    def rotate_look(self, delta_yaw: float, delta_pitch: float):
        self.yaw = (self.yaw + delta_yaw + 180.0) % 360.0 - 180.0
        self.pitch = max(-87.0, min(87.0, self.pitch + delta_pitch))

    def rotate_held_item(self, delta_degrees: float):
        self.held_item_rotation = (self.held_item_rotation + delta_degrees) % 360.0

    def get_forward_ray(self) -> np.ndarray:
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        return np.array([math.cos(pitch_rad) * math.cos(yaw_rad), math.sin(pitch_rad),
                         math.cos(pitch_rad) * math.sin(yaw_rad)])

    def look_at(self, target: np.ndarray):
        d = np.array(target, dtype=float) - self.eye
        self.yaw = math.degrees(math.atan2(d[2], d[0]))
        self.pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))

    def pop_step_events(self) -> List[str]:
        ev = self.step_events
        self.step_events = []
        return ev
