"""Headless player controller: camera orientation, FSM tick and collision step.

This is the single simulation entry point shared by the game loop and the
test-suite, so anything the player can experience can be reproduced without a
window or an OpenGL context.
"""
import numpy as np
import config
from physics.collision import CollisionSystem
from parkour.state_machine import ParkourStateMachine, ParkourState
from parkour.body import BodyModel
from core.controls import make_scheme
from world.spatial import ObstacleIndex


class PlayerController:
    def __init__(self, ledge_system, obstacles, spawn_pos=(-6.0, 0.9, -24.0), yaw=0.0, pitch=0.0, scheme=None):
        self.scheme = scheme if not isinstance(scheme, (str, type(None))) else make_scheme(scheme)
        self.intents = None
        self.ledge_system = ledge_system
        self.obstacles = obstacles if hasattr(obstacles, "query") else ObstacleIndex(obstacles)
        self.spawn_pos = np.array(spawn_pos, dtype=np.float32)
        self.player_size = BodyModel.SIZE.copy()

        self.pos = self.spawn_pos.copy()
        self.prev_pos = self.pos.copy()     # where the body was before the last tick (render interpolation)
        self.vel = np.zeros(3, dtype=np.float32)
        self.on_ground = True
        self.contact = None
        self.yaw = float(yaw)       # degrees (0 = looking down -Z)
        self.pitch = float(pitch)   # degrees
        self.fsm = ParkourStateMachine(ledge_system, obstacles)

        self.turn_remaining = 0.0
        self.turn_rate = 0.0
        self.respawn_count = 0
        self.sim_time = 0.0

    def set_speed_scales(self, run=1.0, climb=1.0):
        """Scale how fast this body runs and climbs (zombie tag: zombies are a bit slower)."""
        self.fsm.run_scale = float(run)
        self.fsm.climb_scale = float(climb)

    # -------------------------------------------------------------- orientation
    def apply_mouse(self, dx, dy):
        self.yaw -= dx * config.MOUSE_SENSITIVITY
        self.pitch -= dy * config.MOUSE_SENSITIVITY
        self.pitch = float(np.clip(self.pitch, config.PITCH_MIN, config.PITCH_MAX))

    def apply_look(self, turn_right_deg, look_up_deg):
        """Controller look: turn right / look up by the given angles (already scaled by frame time)."""
        self.yaw -= float(turn_right_deg)
        self.pitch = float(np.clip(self.pitch + float(look_up_deg), config.PITCH_MIN, config.PITCH_MAX))

    def look_at_point(self, target):
        """Orient the camera toward a world point (used by scripted tests)."""
        eye = self.eye_pos
        d = np.asarray(target, dtype=np.float32) - eye
        self.yaw = float(np.degrees(np.arctan2(-d[0], -d[2])))
        horiz = float(np.hypot(d[0], d[2]))
        self.pitch = float(np.clip(np.degrees(np.arctan2(d[1], horiz)), config.PITCH_MIN, config.PITCH_MAX))

    def face_direction(self, direction):
        d = np.asarray(direction, dtype=np.float32)
        self.yaw = float(np.degrees(np.arctan2(-d[0], -d[2])))

    @property
    def forward_3d(self):
        yaw_rad = np.radians(self.yaw)
        pitch_rad = np.radians(self.pitch)
        f = np.array([
            -np.sin(yaw_rad) * np.cos(pitch_rad),
            np.sin(pitch_rad),
            -np.cos(yaw_rad) * np.cos(pitch_rad)
        ], dtype=np.float32)
        return f / (np.linalg.norm(f) + 1e-7)

    @property
    def forward_h(self):
        yaw_rad = np.radians(self.yaw)
        f = np.array([-np.sin(yaw_rad), 0.0, -np.cos(yaw_rad)], dtype=np.float32)
        return f / (np.linalg.norm(f) + 1e-7)

    @property
    def right_h(self):
        yaw_rad = np.radians(self.yaw)
        r = np.array([np.cos(yaw_rad), 0.0, -np.sin(yaw_rad)], dtype=np.float32)
        return r / (np.linalg.norm(r) + 1e-7)

    @property
    def eye_pos(self):
        return BodyModel.eye_pos(self.pos)

    @property
    def camera_pos(self):
        """Where the view is rendered from: the eyes, ahead of the body centre."""
        return BodyModel.camera_pos(self.pos, self.forward_h)

    @property
    def feet_y(self):
        return BodyModel.feet_y(self.pos)

    RENDER_SNAP_DIST = 2.0          # a move this long in one tick is a teleport (respawn): never blended

    def render_pos(self, alpha):
        """Body centre to draw a frame with, `alpha` (0..1) of the way from the previous physics tick
        to the latest.  The simulation runs at a fixed 60 Hz while frames come when they come; drawing
        between the two ticks keeps the motion even when a frame gets two ticks or none."""
        a = float(np.clip(alpha, 0.0, 1.0))
        d = self.pos - self.prev_pos
        if float(np.dot(d, d)) > self.RENDER_SNAP_DIST ** 2:
            return self.pos.copy()
        return (self.prev_pos + d * np.float32(a)).astype(np.float32)

    @property
    def state(self):
        return self.fsm.state

    # ----------------------------------------------------------------- stepping
    def step(self, dt, input_buf):
        """One fixed physics tick."""
        self.sim_time += dt
        self.prev_pos = self.pos.copy()
        input_buf.update()

        # Automatic yaw blend requested by the FSM (wall eject 180 turn)
        if self.fsm.turn_request_deg != 0.0:
            self.turn_remaining = self.fsm.turn_request_deg
            self.turn_rate = self.fsm.turn_request_deg / max(1e-3, self.fsm.turn_time_request)
            self.fsm.turn_request_deg = 0.0
            self.fsm.turn_time_request = config.WALL_EJECT_TURN_TIME
        if abs(self.turn_remaining) > 1e-3:
            step = self.turn_rate * dt
            if abs(step) > abs(self.turn_remaining):
                step = self.turn_remaining
            self.yaw += step
            self.turn_remaining -= step

        self.intents = self.scheme.resolve(input_buf)
        self.pos, self.vel = self.fsm.update(
            dt, self.pos, self.vel, self.on_ground, self.contact,
            self.forward_h, self.right_h, self.forward_3d, self.intents
        )

        if self.fsm.state in ParkourState.ROOT_MOTION:
            self.on_ground = False
            self.contact = None
        else:
            self.pos, self.vel, self.contact = CollisionSystem.resolve_movement(
                self.pos, self.player_size, self.vel, dt, self.obstacles
            )
            self.on_ground = self.contact.on_ground

        if self.pos[1] < config.KILL_PLANE_Y:
            self.respawn()

    def respawn(self):
        self.pos = self.spawn_pos.copy()
        self.prev_pos = self.pos.copy()
        self.vel[:] = 0.0
        self.on_ground = True
        self.contact = None
        self.fsm.state = ParkourState.GROUND
        self.fsm._release_rail()
        self.fsm.spin = None
        self.fsm.pending_spin = None
        self.fsm.air_peak_y = None
        self.fsm.took_off = False
        self.respawn_count += 1

    # ------------------------------------------------------------------ queries
    def grab_candidate(self):
        """Best hold currently within human reach (None if nothing is grabbable). Memoised per tick/view."""
        if self.fsm.state in ParkourState.ROOT_MOTION:
            return None
        key = (self.sim_time, round(self.yaw, 2), round(self.pitch, 2))
        cached = getattr(self, "_grab_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        result = BodyModel.find_best_grab(
            self.ledge_system.rails, self.pos, self.forward_h, self.forward_3d,
            self.on_ground, self.obstacles, self.vel
        )
        self._grab_cache = (key, result)
        return result

    def climb_down_candidate(self):
        """Top edge in front of the feet that E would lower the player over (None if none)."""
        if self.fsm.state not in (ParkourState.GROUND, ParkourState.SPRINT) or not self.on_ground:
            return None
        return BodyModel.find_climb_down_edge(self.ledge_system.rails, self.pos, self.forward_h, self.obstacles)
