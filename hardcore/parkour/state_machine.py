"""Kinematic parkour state machine with combo evaluation and root motion.

States
------
GROUND / SPRINT   Locomotion with human acceleration limits, jumps, vaults.
AIRBORNE          Ballistic flight, variable jump lift, air steering, wall kicks.
LEDGE_HANG        Hands on a rail, body hanging (or feet resting on the ground).
SHIMMY            Hand-over-hand traverse along the rail.
CLIMB_UP          Crevasse -> next hold: pull-up, foot on the lip, stand, reach (or the reverse).
MANTLE_VAULT      Hands on a top edge -> standing on the top surface (or the reverse: climb down an edge).
CORNER_TURN       Shimmy around an outside corner or into an inside corner onto the joined edge.
DYNO_LAUNCH       Explosive leg push off the wall (up or sideways) - brief pose state.
WALL_EJECT        Backward cat-leap off the wall - brief pose state.
ROLL              Landing roll after a fall of more than 7 ft: momentum carried through a roll.

Flips (a second Space press in the air after a jump) are not a state: a Spin
(parkour/acrobatics.py) turns the body through 360 degrees while the air physics
carry on as usual.  Rolls use a Spin too, for the visible rotation.

Momentum (the flow of a run) is carried through the moves that used to stop it dead:
* a wall kick converts more of the run-up into height the faster you hit the wall (a wall run),
* `climb_flow` (0..1) is charged by the speed you meet a wall with and by chaining holds with W,
  and makes climbs, top-outs and corner turns quicker; topping out of a flowing climb sends you off
  at a run with the sprint build-up restored,
* a vault / step-up taken at a run (a speed vault) keeps the run going.

All positions are the player's AABB centre.
"""
import numpy as np
import config
from parkour.ledge_rail import RailType
from parkour.body import BodyModel, GrabResult, UP
from parkour import acrobatics
from parkour.acrobatics import Spin
from physics.collision import CollisionSystem


class ParkourState:
    GROUND = "GROUND"
    SPRINT = "SPRINT"
    AIRBORNE = "AIRBORNE"
    LEDGE_HANG = "LEDGE_HANG"
    SHIMMY = "SHIMMY"
    CLIMB_UP = "CLIMB_UP"
    MANTLE_VAULT = "MANTLE_VAULT"
    DYNO_LAUNCH = "DYNO_LAUNCH"
    WALL_EJECT = "WALL_EJECT"
    CORNER_TURN = "CORNER_TURN"
    ROLL = "ROLL"

    ROOT_MOTION = (LEDGE_HANG, SHIMMY, CLIMB_UP, MANTLE_VAULT, CORNER_TURN)
    AIR_PHYSICS = (AIRBORNE, DYNO_LAUNCH, WALL_EJECT)


def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class ParkourStateMachine:
    def __init__(self, ledge_system, obstacles):
        self.ledge_system = ledge_system
        self.obstacles = obstacles

        self.state = ParkourState.GROUND
        self.active_rail = None
        self.rail_s = 0.5
        self.grip_point = None          # world position of the hands while on a rail
        self.lockout_timer = 0.0
        self.jump_timer = 0.0
        self.coyote_timer = 0.0
        self.wall_kicked = False
        self.launch_pose_timer = 0.0
        self.supported_hang = False

        # Root-motion bookkeeping (mantle / climb)
        self.motion_time = 0.0
        self.motion_keys = []           # list of (time, position) keyframes
        self.motion_total = 0.0
        self.climb_next = None          # (rail, s) reached at the end of CLIMB_UP
        self.climb_phase = 0.0          # 0..1 progress for the arm rig
        self.climb_fast = False
        self.climb_direction = 1        # +1 climbing up, -1 climbing down
        self.mantle_fast = False
        self.mantle_reverse = False     # climbing down over an edge into a hang
        self.corner_link = None         # RailLink being traversed in CORNER_TURN
        self.corner_entry_s = 0.0
        self.shimmy_dir = 0.0           # last traverse direction (for the hand animation)
        self.turn_time_request = config.WALL_EJECT_TURN_TIME
        self._hold_cache = None
        self.climb_shift = None         # (rail, s): shifting along the hold to where a blocked climb is clear
        self._shift_cache = None
        self._switch_cache = None
        self.edge_switch = False        # the CORNER_TURN in progress is a switch onto an obstruction's edge
        self.motion_start = np.zeros(3, dtype=np.float32)
        self.settle_from = np.zeros(3, dtype=np.float32)
        self.settle_via = None
        self.settle_time = 1.0
        self.supported_y = 0.0

        # Visual camera effects
        self.dutch_tilt = 0.0           # Roll angle in degrees
        self.target_fov = config.BASE_FOV
        self.run_time = 0.0             # seconds of continuous running (sprint build-up)
        self.last_move_dir = None       # held movement direction last tick (for the take-off push)
        self.turn_request_deg = 0.0     # consumed by the controller (wall eject 180 turn)

        # Momentum carried through climbs and vaults
        self.climb_flow = 0.0           # 0..1: how much run-up momentum the climb still carries
        self.flight_speed = 0.0         # horizontal speed the body left the ground with (this flight)
        self.vault_speed = None         # approach speed of the ground vault in progress (None: not a vault)
        self.vault_dir = None           # ... and its direction (the run goes on that way)
        self.motion_scale = 1.0         # duration scale of the root motion in progress (< 1: quicker)
        self.shimmy_rate = 1.0          # traverse speed factor this tick (run on: quicker)

        # Movement scales (1 = a normal athlete): a zombie in zombie tag runs and climbs a little slower
        self.run_scale = 1.0            # ground speeds (walk, jog, sprint)
        self.climb_scale = 1.0          # climbs, mantles, corner turns and shimmies

        # Acrobatics: flips in the air, the landing roll (parkour/acrobatics.py)
        self.spin = None                # Spin in progress (the body's 360 deg rotation), or None
        self.pending_spin = None        # a roll waiting for a flip to finish
        self.took_off = False           # this flight began with a jump (or a wall kick): a flip is allowed
        self.flipped = False            # one flip per flight
        self.air_peak_y = None          # highest the feet have been during this flight
        self.last_drop = 0.0            # height of the last landing (feet, from the top of the flight)
        self.roll_time = 0.0
        self.roll_duration = config.ROLL_TIME
        self.roll_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.roll_speed = 0.0
        self.flip_count = 0
        self.roll_count = 0

        # Telemetry for tests / HUD
        self.last_event = None

    # ------------------------------------------------------------------ utils
    def _set_event(self, name):
        self.last_event = name

    @staticmethod
    def flow_from_speed(speed):
        """Climb flow carried by meeting a wall at this horizontal speed (0 at a jog's leap, 1 at a sprint's)."""
        lo, hi = config.CLIMB_FLOW_SPEED_RANGE
        return float(np.clip((float(speed) - lo) / (hi - lo), 0.0, 1.0))

    def _charge_flow(self, speed):
        self.climb_flow = max(self.climb_flow, self.flow_from_speed(speed))

    def _flow_scale(self):
        """Duration scale of a climbing move at the current flow (1 = no momentum)."""
        return 1.0 - config.CLIMB_FLOW_SPEEDUP * float(np.clip(self.climb_flow, 0.0, 1.0))

    @property
    def move_rate(self):
        """How many times faster than its nominal pace the current move runs (the arms and hands
        keep time with the body: a quick climb is a quick reach)."""
        if self.state in (ParkourState.CLIMB_UP, ParkourState.MANTLE_VAULT, ParkourState.CORNER_TURN):
            return float(self.climb_scale / max(1e-3, self.motion_scale))
        if self.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY):
            return float(self.climb_scale * self.shimmy_rate)
        return 1.0

    @property
    def move_time_left(self):
        """Seconds until the root motion in progress ends (None when there is none)."""
        if self.state not in (ParkourState.CLIMB_UP, ParkourState.MANTLE_VAULT, ParkourState.CORNER_TURN):
            return None
        return max(0.0, float(self.motion_total - self.motion_time)) / max(1e-3, float(self.climb_scale))

    def _chain_flow(self, run):
        """A climb was chained onto the last one: the run-up momentum fades a little, the rhythm of
        climbing at a run (W held, run on) builds some of its own."""
        kept = self.climb_flow * config.CLIMB_FLOW_KEEP
        if run:
            kept = max(kept, min(config.CLIMB_FLOW_CHAIN_MAX, self.climb_flow + config.CLIMB_FLOW_CHAIN_GAIN))
        self.climb_flow = float(kept)

    def _enter_hang(self, grab, pos, vel):
        self.state = ParkourState.LEDGE_HANG
        self.active_rail = grab.rail
        self.rail_s = grab.s
        self.grip_point = grab.grip.copy()
        self.supported_hang = grab.supported
        self.supported_y = float(grab.hang_center[1])
        self.wall_kicked = False
        self.jump_timer = 0.0
        # Blend from the catch position into the hang pose instead of teleporting - around, not
        # through, whatever the player caught it from (a roof they stood on, an awning beside it)
        self.settle_from = pos.copy()
        self.settle_via = self._settle_via(pos, grab.hang_center)
        self.settle_time = 0.0
        # a running leap (or a run off an edge) onto a hold: the climb carries on at speed.  (A launch
        # off a hold - a dyno, an eject, a drop - carries no run-up: flight_speed is 0 then.)
        self._charge_flow(self.flight_speed)
        vel[:] = 0.0
        self._set_event("GRAB")
        return pos, vel

    def _start_root_motion(self, keys, scale=1.0):
        """keys: list of (duration_from_previous, position); scale: how the move's durations were
        scaled from their nominal values (for move_rate)."""
        self.motion_scale = float(scale)
        self.motion_time = 0.0
        self.climb_phase = 0.0
        self.motion_keys = []
        t = 0.0
        for dur, p in keys:
            t += dur
            self.motion_keys.append((t, np.asarray(p, dtype=np.float32).copy()))
        self.motion_total = t

    def _eval_root_motion(self, start_pos):
        """Body centre along the keyframed move at motion_time.

        A monotone cubic Hermite through the keys: the body starts and ends at rest and passes
        every key at its time, but flows through the interior keys instead of stopping dead at
        each (the pull-up runs into the stand-up, the swing round a corner does not pause at the
        corner).  Tangents are per-axis weighted harmonic means (Fritsch-Butland), zero where the
        path turns back on an axis, so it never overshoots a key.  With a single segment (or zero
        tangents) this is exactly the old per-segment smoothstep."""
        times = [0.0] + [t for t, _ in self.motion_keys]
        pts = [np.asarray(start_pos, dtype=np.float32)] + [p for _, p in self.motion_keys]
        t_now = self.motion_time
        if t_now >= times[-1]:
            return self.motion_keys[-1][1].copy()
        k = 1
        while k < len(times) - 1 and t_now > times[k]:
            k += 1
        t0, t1 = times[k - 1], times[k]
        h = max(1e-6, t1 - t0)
        u = float(np.clip((t_now - t0) / h, 0.0, 1.0))
        m0 = self._key_tangent(times, pts, k - 1)
        m1 = self._key_tangent(times, pts, k)
        u2, u3 = u * u, u * u * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        return (h00 * pts[k - 1] + h10 * h * m0 + h01 * pts[k] + h11 * h * m1).astype(np.float32)

    @staticmethod
    def _key_tangent(times, pts, i):
        """Velocity of the root motion at key i (zero at the two ends)."""
        if i <= 0 or i >= len(pts) - 1:
            return np.zeros(3, dtype=np.float32)
        h0 = max(1e-6, times[i] - times[i - 1])
        h1 = max(1e-6, times[i + 1] - times[i])
        d0 = (pts[i] - pts[i - 1]) / h0
        d1 = (pts[i + 1] - pts[i]) / h1
        m = np.zeros(3, dtype=np.float32)
        for a in range(3):
            x0, x1 = float(d0[a]), float(d1[a])
            if x0 * x1 <= 0.0:
                continue                      # the path turns back (or stops) on this axis: no overshoot
            m[a] = 3.0 * (h0 + h1) / ((2.0 * h1 + h0) / x0 + (h1 + 2.0 * h0) / x1)
        return m

    def _begin_mantle(self, rail, s, pos, fast):
        self.state = ParkourState.MANTLE_VAULT
        self.active_rail = rail
        self.rail_s = s
        self.grip_point = rail.get_point(s).copy()
        self.mantle_fast = fast
        self.mantle_reverse = False
        self.vault_speed = None
        self.vault_dir = None
        # (flow counts half on a top-out: the hoist is the longest single lift of the body)
        flow_scale = 1.0 - 0.5 * config.CLIMB_FLOW_SPEEDUP * float(np.clip(self.climb_flow, 0.0, 1.0))
        scale = (config.MANTLE_FAST_SCALE if fast else 1.0) * flow_scale
        hoist = config.MANTLE_HOIST_TIME * scale
        step = config.MANTLE_STEP_TIME * scale
        mid = BodyModel.mantle_mid_center(rail, s)
        target = BodyModel.mantle_target_center(rail, s, self.obstacles)
        self.motion_start = pos.copy()
        pull = BodyModel.pull_up_center(rail, s)
        if float(pos[1]) < float(pull[1]) - 0.05:
            # from a hang: the pull-up flows on into the press over the edge (no stop at the chest)
            self._start_root_motion([(hoist * 0.2, pull), (hoist * 0.8, mid), (step, target)], scale)
        else:
            self._start_root_motion([(hoist, mid), (step, target)], scale)
        self.wall_kicked = False
        self.jump_timer = 0.0
        self._set_event("MANTLE_FAST" if fast else "MANTLE")

    def _begin_climb_down_edge(self, rail, s, pos):
        """Reverse mantle: crouch at the edge, turn to face the wall and lower into a hang."""
        self.state = ParkourState.MANTLE_VAULT
        self.active_rail = rail
        self.rail_s = s
        self.grip_point = rail.get_point(s).copy()
        self.mantle_fast = False
        self.mantle_reverse = True
        self.vault_speed = None
        self.vault_dir = None
        self.climb_flow = 0.0
        total = config.CLIMB_DOWN_EDGE_TIME
        mid = BodyModel.mantle_mid_center(rail, s)
        hang = BodyModel.hang_center(rail, s)
        self.motion_start = pos.copy()
        self._start_root_motion([(total * 0.35, mid), (total * 0.65, hang)])
        self.turn_request_deg = 180.0
        self.turn_time_request = total * 0.8
        self.jump_timer = 0.0
        self._set_event("CLIMB_DOWN_EDGE")

    def _begin_corner_turn(self, rail, link, entry_s, pos):
        self.state = ParkourState.CORNER_TURN
        self.edge_switch = False
        self.corner_link = link
        self.corner_entry_s = entry_s
        total = config.CORNER_TURN_TIME_CONVEX if link.kind == "convex" else config.CORNER_TURN_TIME_CONCAVE
        flow_scale = self._flow_scale()
        total *= flow_scale
        path = BodyModel.corner_path(rail, self.rail_s, link, entry_s)
        keys = [(total / len(path), p) for p in path]
        self.motion_start = pos.copy()
        self._start_root_motion(keys, flow_scale)
        # Turn the camera from facing this wall to facing the next one
        self._turn_with_body(rail, link.rail, total)
        self._set_event("CORNER_" + link.kind.upper())

    def _turn_with_body(self, from_rail, to_rail, duration):
        """Turn the camera with the body when a move takes the hands onto a hold that faces another
        way (round a corner, onto an awning's or balcony's side edge): the view keeps the same
        offset from the way the body faces, instead of staring at the side of the wall it left."""
        f0 = -from_rail.normal
        f1 = -to_rail.normal
        ang = float(np.degrees(np.arctan2(f0[0] * f1[2] - f0[2] * f1[0], f0[0] * f1[0] + f0[2] * f1[2])))
        if abs(ang) > 1.0:
            self.turn_request_deg = -ang
            self.turn_time_request = max(1e-3, duration / max(0.1, self.climb_scale))

    def _begin_vault(self, rail, s, pos, vel=None, approach=0.0):
        """Ground vault / step-up: duration scales with the height to climb.  Taken at a run it is a
        speed vault: quicker, and the run carries on out of it (see MANTLE_DONE)."""
        self.state = ParkourState.MANTLE_VAULT
        self.active_rail = rail
        self.rail_s = s
        self.grip_point = rail.get_point(s).copy()
        self.mantle_fast = True
        self.mantle_reverse = False
        speed, heading = 0.0, None
        if vel is not None:
            h = np.array([vel[0], 0.0, vel[2]], dtype=np.float32)
            along = float(np.linalg.norm(h))
            speed = float(np.hypot(along, approach))
            inward = -rail.normal
            # the run's direction over the obstacle: what was left of the velocity plus what the wall took
            d = h + np.array([inward[0], 0.0, inward[2]], dtype=np.float32) * float(approach)
            dn = float(np.linalg.norm(d))
            if dn > 1e-3:
                heading = d / dn
        self.vault_speed = speed
        self.vault_dir = heading
        rise = max(0.0, rail.height - BodyModel.feet_y(pos))
        total = float(np.clip(rise * config.VAULT_TIME_PER_METER, config.VAULT_TIME_MIN, 0.9))
        u = float(np.clip((speed - config.WALK_SPEED) / (config.SPRINT_SPEED - config.WALK_SPEED), 0.0, 1.0))
        vault_scale = 1.0 + (config.VAULT_FAST_SCALE - 1.0) * u
        total *= vault_scale
        mid = BodyModel.mantle_mid_center(rail, s)
        target = BodyModel.mantle_target_center(rail, s, self.obstacles)
        self.motion_start = pos.copy()
        self._start_root_motion([(total * 0.6, mid), (total * 0.4, target)], vault_scale)
        self.jump_timer = 0.0
        self._set_event("VAULT")

    def _begin_climb_up(self, rail, s, nxt, pos, fast):
        """Pull up on the hold, get a foot onto its lip, push up as far as needed and reach the next hold."""
        self.state = ParkourState.CLIMB_UP
        next_rail, next_s, fraction = nxt
        self.climb_next = (next_rail, next_s)
        self.climb_fast = fast
        self.climb_direction = 1
        scale = (config.CLIMB_FAST_SCALE if fast else 1.0) * self._flow_scale()
        pull = BodyModel.pull_up_center(rail, s)
        stand = BodyModel.stand_center(rail, s, max(fraction, 0.15))
        keys = [(config.CLIMB_PULL_TIME * scale, pull),
                (config.CLIMB_STAND_TIME * scale * max(0.35, fraction), stand),
                # Commit weight to the new hold and settle into the hang
                (config.CLIMB_SETTLE_TIME * scale, BodyModel.hang_center(next_rail, next_s))]
        self.motion_start = pos.copy()
        self._start_root_motion(keys, scale)
        # a hold off to the side on another face (an awning's / balcony's side edge): turn with the body
        self._turn_with_body(rail, next_rail, self.motion_total)
        self._set_event("CLIMB_UP_FAST" if fast else "CLIMB_UP")

    def _begin_climb_down(self, rail, s, lower, pos, fast):
        """Lower the feet onto the lip below, then drop the hands to that hold."""
        self.state = ParkourState.CLIMB_UP
        lower_rail, lower_s, fraction = lower
        self.climb_next = (lower_rail, lower_s)
        self.climb_fast = fast
        self.climb_direction = -1
        scale = (config.CLIMB_FAST_SCALE if fast else 1.0) * self._flow_scale()
        keys = [
            (config.CLIMB_PULL_TIME * scale, BodyModel.stand_center(lower_rail, lower_s, max(fraction, 0.15))),
            (config.CLIMB_STAND_TIME * scale * max(0.35, fraction), BodyModel.pull_up_center(lower_rail, lower_s)),
            (config.CLIMB_SETTLE_TIME * scale, BodyModel.hang_center(lower_rail, lower_s)),
        ]
        self.motion_start = pos.copy()
        self._start_root_motion(keys, scale)
        self._turn_with_body(rail, lower_rail, self.motion_total)
        self._set_event("CLIMB_DOWN")

    # ------------------------------------------------------------------ update
    def update(self, dt, player_pos, velocity, on_ground, contact, forward_dir, right_dir, look_dir, intents):
        """
        Updates the player state machine, velocity, and position.
        contact: ContactInfo from the previous collision step (or None).
        intents: core.controls.Intents resolved by the active control scheme.
        Returns:
            new_pos: np.ndarray
            new_vel: np.ndarray
        """
        pos = np.asarray(player_pos, dtype=np.float32).copy()
        vel = np.asarray(velocity, dtype=np.float32).copy()
        self.last_event = None

        # Acrobatics: advance the body's rotation; track how high the feet got in this flight
        if self.spin is not None:
            self.spin.advance(dt)
            if self.spin.done:
                self.spin = self.pending_spin
                self.pending_spin = None
        feet_now = float(pos[1]) - config.PLAYER_HALF_HEIGHT
        if self.state not in ParkourState.AIR_PHYSICS:
            # not in flight (standing, hanging, climbing, rolling): whatever jump began the last flight
            # is over - running off an edge later is a fall (a coyote jump), never a flip
            self.took_off = False
            self.flipped = False
        if self.state not in ParkourState.AIR_PHYSICS or self.air_peak_y is None:
            self.air_peak_y = feet_now
        else:
            self.air_peak_y = max(self.air_peak_y, feet_now)

        if self.lockout_timer > 0.0:
            self.lockout_timer = max(0.0, self.lockout_timer - dt)
        if self.launch_pose_timer > 0.0:
            self.launch_pose_timer = max(0.0, self.launch_pose_timer - dt)

        self.dutch_tilt += (0.0 - self.dutch_tilt) * min(1.0, config.DUTCH_TILT_SPEED * dt)
        # the view widens with actual speed (momentum carries it through a jump)
        speed_now = float(np.hypot(velocity[0], velocity[2]))
        u = float(np.clip((speed_now - config.WALK_SPEED) / (config.SPRINT_SPEED - config.WALK_SPEED), 0.0, 1.0))
        self.target_fov = config.BASE_FOV + (config.SPRINT_FOV - config.BASE_FOV) * u
        vaulting = self.state == ParkourState.MANTLE_VAULT and self.vault_speed is not None
        if vaulting:
            pass                                     # a vault is part of the run: the build-up is kept
        elif self.state in ParkourState.ROOT_MOTION or self.active_rail is not None:
            self.run_time = 0.0                      # climbing is not running
        elif not on_ground:
            self.run_time = max(0.0, self.run_time - dt)   # a leap keeps most of the build-up

        # =====================================================================
        # MANTLE VAULT (root motion)
        # =====================================================================
        if self.state == ParkourState.MANTLE_VAULT:
            self.motion_time += dt * self.climb_scale
            self.climb_phase = min(1.0, self.motion_time / max(1e-6, self.motion_total))
            if self.motion_time < self.motion_total:
                pos = self._eval_root_motion(self.motion_start)
                vel[:] = 0.0
                return pos, vel
            pos = self.motion_keys[-1][1].copy()
            if self.mantle_reverse:
                # Lowered over the edge: now hanging from it, facing the wall
                self.mantle_reverse = False
                self.state = ParkourState.LEDGE_HANG
                pos = self._ground_adjusted_hang(pos)
                self.settle_time = config.GRAB_SETTLE_TIME
                vel[:] = 0.0
                self._set_event("CLIMB_DOWN_EDGE_DONE")
                return pos, vel
            self.state = ParkourState.GROUND
            inward = -self.active_rail.normal
            direction = inward
            if self.vault_speed is not None:
                # speed vault: off the top with the run still going, the way it was going
                exit_speed = max(config.MANTLE_EXIT_SPEED,
                                 min(config.SPRINT_SPEED, self.vault_speed * config.VAULT_SPEED_KEEP))
                d = self.vault_dir
                if d is not None and float(np.dot(d, inward)) > 0.5:
                    direction = d
            elif self.mantle_fast:
                # topping out of a flowing climb: leave the edge at a run, the sprint build-up restored
                flow = float(np.clip(self.climb_flow, 0.0, 1.0))
                exit_speed = config.MANTLE_EXIT_SPEED + (config.MANTLE_FLOW_EXIT_SPEED - config.MANTLE_EXIT_SPEED) * flow
                self.run_time = max(self.run_time, flow * config.RUN_BUILD_TIME)
            else:
                exit_speed = config.MANTLE_EXIT_SPEED * 0.5
            vel = (direction * exit_speed).astype(np.float32)
            vel[1] = 0.0
            self.active_rail = None
            self.grip_point = None
            self.vault_speed = None
            self.vault_dir = None
            self.climb_flow = 0.0
            self._set_event("MANTLE_DONE")
            return pos, vel

        # =====================================================================
        # CLIMB UP (crevasse -> next hold using the feet)
        # =====================================================================
        if self.state == ParkourState.CLIMB_UP:
            self.motion_time += dt * self.climb_scale
            self.climb_phase = min(1.0, self.motion_time / max(1e-6, self.motion_total))
            if self.motion_time < self.motion_total:
                pos = self._eval_root_motion(self.motion_start)
                vel[:] = 0.0
                return pos, vel
            next_rail, next_s = self.climb_next
            pos = self.motion_keys[-1][1].copy()
            vel[:] = 0.0
            self.climb_next = None
            # Hands on the next hold - including a top edge, from which W held carries on over the top
            self.active_rail = next_rail
            self.rail_s = next_s
            self.grip_point = next_rail.get_point(next_s).copy()
            pos = self._ground_adjusted_hang(pos)
            self.state = ParkourState.LEDGE_HANG
            self._set_event("CLIMB_UP_DONE")
            return pos, vel

        # =====================================================================
        # CORNER TURN (root motion + hand-over onto the joined edge)
        # =====================================================================
        if self.state == ParkourState.CORNER_TURN:
            self.motion_time += dt * self.climb_scale
            self.climb_phase = min(1.0, self.motion_time / max(1e-6, self.motion_total))
            if self.motion_time < self.motion_total:
                pos = self._eval_root_motion(self.motion_start)
                vel[:] = 0.0
                return pos, vel
            link = self.corner_link
            self.active_rail = link.rail
            self.rail_s = self.corner_entry_s
            self.grip_point = link.rail.get_point(self.rail_s).copy()
            self.corner_link = None
            self.settle_time = config.GRAB_SETTLE_TIME
            self.state = ParkourState.SHIMMY if abs(intents.shimmy) > 0 else ParkourState.LEDGE_HANG
            pos = self._ground_adjusted_hang(self.motion_keys[-1][1])
            vel[:] = 0.0
            self._set_event("EDGE_SWITCH_DONE" if self.edge_switch else "CORNER_DONE")
            self.edge_switch = False
            return pos, vel

        # =====================================================================
        # LEDGE HANG & SHIMMY
        # =====================================================================
        if self.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY):
            vel[:] = 0.0
            rail = self.active_rail
            if rail is None:
                self.state = ParkourState.AIRBORNE
                return pos, vel

            settled = self.settle_time >= config.GRAB_SETTLE_TIME
            if self.climb_shift is not None and (self.climb_shift[0] is not rail or not intents.climb_up
                                                 or abs(intents.shimmy) > 0.0):
                self.climb_shift = None          # let go of W, steered sideways, or a different hold

            # Priority 1: Dyno Up
            if settled and intents.has("dyno_up"):
                intents.consume("dyno_up")
                self.state = ParkourState.DYNO_LAUNCH
                self.launch_pose_timer = 0.2
                vel = UP * config.DYNO_UP_VY + rail.normal * config.DYNO_UP_PUSH
                self.lockout_timer = config.GRAB_LOCKOUT_TIME
                self._release_rail()
                self._set_event("DYNO_UP")
                return pos, vel

            # Priority 2: Wall Eject (backward cat leap)
            if settled and intents.has("eject"):
                intents.consume("eject")
                self.state = ParkourState.WALL_EJECT
                self.launch_pose_timer = 0.2
                vel = rail.normal * config.WALL_EJECT_VHORIZ + UP * config.WALL_EJECT_VY
                self.lockout_timer = config.GRAB_LOCKOUT_TIME
                self.turn_request_deg = 180.0
                self._release_rail()
                self._set_event("WALL_EJECT")
                return pos, vel

            # Priority 3/4: Lateral Dyno
            for name, sign, tilt in (("dyno_left", -1.0, -config.MAX_TILT_DEG), ("dyno_right", 1.0, config.MAX_TILT_DEG)):
                if settled and intents.has(name):
                    intents.consume(name)
                    self.state = ParkourState.DYNO_LAUNCH
                    self.launch_pose_timer = 0.2
                    vel = (sign * rail.tangent * config.LATERAL_DYNO_SPEED +
                           UP * config.LATERAL_DYNO_VY +
                           rail.normal * config.LATERAL_DYNO_PUSH)
                    self.dutch_tilt = tilt
                    self.lockout_timer = config.GRAB_LOCKOUT_TIME
                    self._release_rail()
                    self._set_event("DYNO_LATERAL")
                    return pos, vel

            # Priority 5: Let go (explicit drop)
            if settled and intents.has("let_go"):
                intents.consume("let_go")
                return self._drop(rail, pos)

            # Priority 6: Top out over a top edge.  Like every other hold it is climbed by pushing
            # W: once the body has settled into the hang (the same short pause as between any two
            # holds), W still held - or pressed afresh - pulls the climber over the edge.  Let go
            # of W and you simply hang there.
            if settled and rail.rail_type == RailType.MANTLE_TOP and (intents.climb_up or intents.has("mantle")):
                if BodyModel.mantle_is_valid(rail, self.rail_s, self.obstacles):
                    if intents.has("mantle"):
                        intents.consume("mantle")
                    self._chain_flow(intents.run)
                    self._begin_mantle(rail, self.rail_s, pos, fast=intents.run)
                    return pos, vel
                self._set_event("MANTLE_BLOCKED")

            # Priority 7: Climb up hold-to-hold (W held) - only once the body has settled into the hang
            if settled and intents.climb_up:
                nxt = self._hold_above_cached(rail, look_dir)
                if nxt is not None:
                    self.climb_shift = None
                    self._chain_flow(intents.run)
                    self._begin_climb_up(rail, self.rail_s, nxt, pos, intents.run)
                    return pos, vel
                if rail.rail_type != RailType.MANTLE_TOP or not BodyModel.mantle_is_valid(rail, self.rail_s, self.obstacles):
                    # Collision buffer: something over the hands (an awning above a window head) blocks
                    # the climb here - shift along the hold to the nearest spot where it is clear
                    if self.climb_shift is None and abs(intents.shimmy) == 0.0:
                        target = self._climb_shift_target(rail, look_dir)
                        if target is not None:
                            self.climb_shift = (rail, target)
                    if self.climb_shift is None:
                        self._set_event("CLIMB_BLOCKED")

            # Priority 8/9: Climb down to a lower hold, or drop - fresh press so a held
            # chord does not auto-drop the moment a ledge is caught
            if settled and intents.has("climb_down"):
                intents.consume("climb_down")
                lower = BodyModel.find_hold_below(self.ledge_system.rails, rail, self.rail_s, self.obstacles)
                if lower is not None:
                    self._begin_climb_down(rail, self.rail_s, lower, pos, intents.run)
                    return pos, vel
                return self._drop(rail, pos)

            # Priority 10: Shimmy Left/Right (or the climb shift toward a clear spot)
            shimmy_dir = intents.shimmy if settled else 0.0
            shift_target = None
            if settled and shimmy_dir == 0.0 and self.climb_shift is not None:
                shift_target = self.climb_shift[1]
                if abs(shift_target - self.rail_s) * rail.length < 1e-3:
                    # arrived but still blocked (the world changed?): give up cleanly
                    self.climb_shift = None
                    shift_target = None
                    self._set_event("CLIMB_BLOCKED")
                else:
                    shimmy_dir = 1.0 if shift_target > self.rail_s else -1.0

            self.shimmy_rate = config.SHIMMY_FAST_SCALE if intents.run else 1.0
            if settled:
                # hanging about (or traversing) is not climbing: the momentum of the run fades
                self.climb_flow = max(0.0, self.climb_flow - config.CLIMB_FLOW_IDLE_DRAIN * dt)
            self.shimmy_dir = shimmy_dir
            if abs(shimmy_dir) > 0.0:
                self.state = ParkourState.SHIMMY
                speed = config.SHIMMY_SPEED * self.climb_scale * self.shimmy_rate
                ds = (shimmy_dir * speed * dt) / (rail.length + 1e-5)
                new_s = rail.clamp_s(self.rail_s + ds)
                if shift_target is not None:
                    new_s = min(new_s, shift_target) if shimmy_dir > 0 else max(new_s, shift_target)
                candidate = BodyModel.hang_center(rail, new_s)
                if self.supported_hang:
                    candidate[1] = self.supported_y
                moved = False
                if abs(new_s - self.rail_s) > 1e-9 and BodyModel.pose_is_free(candidate, self.obstacles):
                    self.rail_s = new_s
                    moved = True
                if not moved and shift_target is not None:
                    self.climb_shift = None          # the way is blocked after all: stop, never turn a corner
                    self._set_event("CLIMB_BLOCKED")
                elif not moved and self._near_rail_end(rail, shimmy_dir):
                    # End of this edge: continue around the corner if the edge goes on
                    link = rail.links.get("right" if shimmy_dir > 0 else "left")
                    if link is not None:
                        entry = BodyModel.corner_entry(link, self.obstacles)
                        if (entry is not None and self._corner_sweep_ok(rail, link, entry) and
                                BodyModel.path_is_free([pos] + BodyModel.corner_path(rail, self.rail_s, link, entry),
                                                       self.obstacles)):
                            self._begin_corner_turn(rail, link, entry, pos)
                            return pos, vel
                    self._set_event("EDGE_END")
                elif not moved:
                    # Blocked in the middle of the edge by something in the way (an awning, a balcony):
                    # never a corner turn from here - that would sweep the body along the wall to the
                    # far end of the edge.  If the obstruction has an edge of its own facing us, switch
                    # onto it (lower down if need be); otherwise just stop.
                    switch = self._switch_target(rail, shimmy_dir, candidate)
                    if switch is not None:
                        self._begin_edge_switch(rail, switch[0], switch[1], pos)
                        return pos, vel
                    self._set_event("SHIMMY_BLOCKED")
            else:
                self.state = ParkourState.LEDGE_HANG

            # Anchor position strictly along rail
            self.grip_point = rail.get_point(self.rail_s).copy()
            anchor = BodyModel.hang_center(rail, self.rail_s)
            if self.supported_hang:
                anchor[1] = self.supported_y
            if self.settle_time < config.GRAB_SETTLE_TIME:
                self.settle_time += dt
                t = _smoothstep(self.settle_time / config.GRAB_SETTLE_TIME)
                via = getattr(self, "settle_via", None)
                if via is None:
                    pos = self.settle_from + t * (anchor - self.settle_from)
                elif t < 0.5:
                    pos = self.settle_from + (2.0 * t) * (via - self.settle_from)
                else:
                    pos = via + (2.0 * t - 1.0) * (anchor - via)
            else:
                pos = anchor
            return pos, vel

        # =====================================================================
        # GROUND & SPRINT
        # =====================================================================
        if on_ground and self.state not in (ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT):
            self.wall_kicked = False

            # Touch-down: a fall from higher than 7 ft is turned into a roll
            if self.state in ParkourState.AIR_PHYSICS:
                self.last_drop = float(self.air_peak_y) - feet_now
                self.took_off = False
                self.flipped = False
                if self.spin is not None and not self.spin.is_roll:
                    self.spin.rush()             # landed mid-flip: come round at once
                if self.last_drop > config.ROLL_MIN_DROP:
                    self._begin_roll(pos, vel, forward_dir)
                    return pos, vel
            self.air_peak_y = feet_now
            self.climb_flow = 0.0
            self.flight_speed = float(np.hypot(vel[0], vel[2]))     # what a step off an edge carries
            if self.state == ParkourState.ROLL:
                if self._roll_step(dt, pos, vel, intents, forward_dir, right_dir):
                    return pos, vel

            # Grab / vault a rail within human reach (a buffered press stays valid until it succeeds)
            if intents.has("grab") and self.lockout_timer <= 0.0:
                grab = BodyModel.find_best_grab(self.ledge_system.rails, pos, forward_dir, look_dir, True, self.obstacles)
                if grab is not None:
                    intents.consume("grab")
                    if grab.mode == GrabResult.MANTLE:
                        self._begin_vault(grab.rail, grab.s, pos, vel, self._approach(contact))
                        return pos, vel
                    return self._enter_hang(grab, pos, vel)

            # Lower yourself over the edge you are standing at (reverse mantle)
            if intents.has("climb_down_edge") and self.lockout_timer <= 0.0:
                edge = BodyModel.find_climb_down_edge(self.ledge_system.rails, pos, forward_dir, self.obstacles)
                if edge is not None:
                    intents.consume("climb_down_edge")
                    self._begin_climb_down_edge(edge[0], edge[1], pos)
                    return pos, vel

            # Automatic step-up over low obstacles while moving into them
            if contact is not None and contact.wall_normal is not None and intents.auto_step_max > 0.0:
                auto = self._auto_step(contact, pos, forward_dir, intents.auto_step_max)
                if auto is not None:
                    self._begin_vault(auto[0], auto[1], pos, vel, self._approach(contact))
                    return pos, vel

            move_dir = self._move_dir(intents, forward_dir, right_dir)
            move_len = float(np.linalg.norm(move_dir))
            amount = self._move_amount(intents)

            # Jump initiation (legs) - an analog stick eased over pushes a shorter leap
            if intents.has("jump"):
                intents.consume("jump")
                return self._jump(pos, vel, move_dir * amount if move_len > 1e-4 else None, forward_dir)

            self.coyote_timer = config.COYOTE_TIME
            self.last_move_dir = move_dir.copy() if move_len > 1e-4 else None

            # Sprinting needs the stick (nearly) fully forward; a key always is
            is_sprinting = intents.run and intents.forward and amount >= config.PAD_SPRINT_MIN_TILT
            self.state = ParkourState.SPRINT if (is_sprinting and move_len > 0.1) else ParkourState.GROUND
            self._update_run_build(dt, vel, move_dir if move_len > 1e-4 else None, is_sprinting)
            # Walking speed follows the stick's tilt (a key is full tilt)
            target_max_speed = (self.run_target_speed() if is_sprinting else config.WALK_SPEED * amount) * self.run_scale

            if move_len > 1e-4:
                target_vel = move_dir * target_max_speed
                self._accelerate(vel, target_vel, dt)
            else:
                vel[0] -= vel[0] * min(1.0, config.GROUND_FRICTION * dt)
                vel[2] -= vel[2] * min(1.0, config.GROUND_FRICTION * dt)

            return pos, vel

        # =====================================================================
        # AIRBORNE (Jump / Fall / Dyno flight)
        # =====================================================================
        if self.launch_pose_timer <= 0.0 or self.state not in (ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT):
            self.state = ParkourState.AIRBORNE
        if on_ground and self.state == ParkourState.AIRBORNE:
            # Landed this frame (e.g. after a launch pose) - fall through to ground next tick
            self.state = ParkourState.GROUND

        if self.state == ParkourState.AIRBORNE:
            self.roll_time = 0.0
        flipping = self.spin is not None and not self.spin.is_roll

        # Flip: the parkour button pressed again in the air after a jump, with nothing to reach for
        if (not flipping and self.took_off and not self.flipped and self.state == ParkourState.AIRBORNE
                and intents.has("jump") and self._try_flip(pos, vel, forward_dir, look_dir, intents)):
            intents.consume("jump")
            return self._air_physics(dt, pos, vel, intents, forward_dir, right_dir)

        # In-flight grab / vault (the hands are round the shins during a flip)
        if intents.has("grab") and self.lockout_timer <= 0.0 and not flipping:
            grab = BodyModel.find_best_grab(self.ledge_system.rails, pos, forward_dir, look_dir, False, self.obstacles, vel)
            if grab is not None:
                intents.consume("grab")
                if grab.mode == GrabResult.MANTLE:
                    # Too low to hang from: vault it - at the speed the leap carried
                    self._begin_mantle(grab.rail, grab.s, pos, fast=True)
                    self.vault_speed = self.flight_speed
                    h = np.array([vel[0], 0.0, vel[2]], dtype=np.float32)
                    hn = float(np.linalg.norm(h))
                    self.vault_dir = h / hn if hn > 0.5 else None
                    return pos, vel
                # Catching an edge always ends in a hang; topping out is a separate W press
                return self._enter_hang(grab, pos, vel)

        # Coyote time: a jump pressed just after running off an edge still counts
        if self.coyote_timer > 0.0:
            self.coyote_timer = max(0.0, self.coyote_timer - dt)
            if intents.has("jump") and vel[1] <= 0.0:
                intents.consume("jump")
                push_dir = self._move_dir(intents, forward_dir, right_dir)
                return self._jump(pos, vel, push_dir * self._move_amount(intents)
                                  if float(np.linalg.norm(push_dir)) > 1e-4 else None, forward_dir)

        # Wall kick (tic-tac): plant a foot on the wall we just ran into
        if (contact is not None and contact.wall_normal is not None and not self.wall_kicked
                and intents.wall_kick and not flipping
                and contact.approach_speed >= config.WALL_KICK_MIN_APPROACH_SPEED
                and vel[1] >= config.WALL_KICK_MIN_VY and not on_ground):
            # wall run: the faster the body meets the wall, the more of the run-up goes into height
            lo, hi = config.WALL_RUN_SPEED_RANGE
            u = float(np.clip((contact.approach_speed - lo) / (hi - lo), 0.0, 1.0))
            extra = config.WALL_RUN_EXTRA_VY * u
            new_vy = min(max(float(vel[1]), 0.0) * config.WALL_KICK_KEEP_VY + config.WALL_KICK_BOOST_VY + extra,
                         config.WALL_KICK_MAX_VY + extra)
            vel[1] = new_vy
            self._charge_flow(contact.approach_speed)
            vel += contact.wall_normal * config.WALL_KICK_PUSH
            self.wall_kicked = True
            self.jump_timer = 0.0
            self.took_off = True                 # a kick is a take-off too: a flip may follow
            self._set_event("WALL_KICK")

        return self._air_physics(dt, pos, vel, intents, forward_dir, right_dir)

    def _air_physics(self, dt, pos, vel, intents, forward_dir, right_dir):
        """Variable-jump lift, gravity, air steering and drag for one tick in the air."""

        # Variable jump upward lift reservoir
        if self.jump_timer > 0.0:
            if intents.jump_held:
                vel[1] += config.JUMP_LIFT_ACCEL * dt
                self.jump_timer = max(0.0, self.jump_timer - dt)
            else:
                self.jump_timer = 0.0

        vel[1] += config.GRAVITY * dt

        # Air steering can redirect momentum but never adds speed beyond what the legs produced
        air_move = self._move_dir(intents, forward_dir, right_dir) * self._move_amount(intents)
        if float(np.linalg.norm(air_move)) > 1e-4:
            speed_before = float(np.hypot(vel[0], vel[2]))
            vel[0] += air_move[0] * config.AIR_ACCEL * dt
            vel[2] += air_move[2] * config.AIR_ACCEL * dt
            cap = max(speed_before, config.AIR_CONTROL_MAX_SPEED)
            speed_after = float(np.hypot(vel[0], vel[2]))
            if speed_after > cap:
                scale = cap / speed_after
                vel[0] *= scale
                vel[2] *= scale

        vel[0] -= vel[0] * (config.AIR_DRAG * dt)
        vel[2] -= vel[2] * (config.AIR_DRAG * dt)

        return pos, vel

    # ---------------------------------------------------------------- helpers
    def _hold_above_cached(self, rail, look_dir):
        """find_hold_above is the costliest query; while hanging still it is re-used for a few ticks."""
        key = (id(rail), round(float(self.rail_s), 3), tuple(np.round(look_dir, 1)))
        if self._hold_cache is not None and self._hold_cache[0] == key and self._hold_cache[2] < 0.25:
            self._hold_cache[2] += config.FIXED_DT
            return self._hold_cache[1]
        nxt = BodyModel.find_hold_above(self.ledge_system.rails, rail, self.rail_s, self.obstacles, look_dir)
        self._hold_cache = [key, nxt, 0.0]
        return nxt

    def _settle_via(self, start, end):
        """A corner point for the catch blend when the straight line would clip something: across
        then down, or down then across.  None when the straight line is clear (or nothing is)."""
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if BodyModel.path_is_free([start, end], self.obstacles, shrink=0.03):
            return None
        for via in (np.array([end[0], start[1], end[2]], dtype=np.float32),
                    np.array([start[0], end[1], start[2]], dtype=np.float32)):
            if BodyModel.path_is_free([start, via, end], self.obstacles, shrink=0.03):
                return via
        return None

    def _near_rail_end(self, rail, direction):
        """True when the hands are at (or within CORNER_TURN_END_TOLERANCE of) the end of `rail` they
        are travelling toward - the only place a traverse may continue around a corner."""
        m = rail.s_margin()
        left = ((1.0 - m) - self.rail_s) if direction > 0 else (self.rail_s - m)
        return left * rail.length <= config.CORNER_TURN_END_TOLERANCE + 1e-6

    def _corner_sweep_ok(self, rail, link, entry_s):
        """Safety net: a corner turn moves the body round a corner (under a metre), never along a wall."""
        start = BodyModel.hang_center(rail, self.rail_s)
        end = BodyModel.hang_center(link.rail, entry_s)
        return float(np.linalg.norm(end - start)) <= config.CORNER_TURN_MAX_SWEEP

    def _switch_target(self, rail, direction, blocked_pose):
        """An edge to switch onto when a traverse is blocked mid-edge: an edge of the obstruction itself
        (the awning we ran into), facing us, from just above down to a climbable drop below the
        current hold, whose hang pose is free and reachable in a straight, collision-free move.
        Returns (RailLink, entry_s) or None.  Cached per spot (holding the direction costs nothing)."""
        key = (id(rail), round(float(self.rail_s), 4), direction)
        if self._switch_cache is not None and self._switch_cache[0] == key:
            return self._switch_cache[1]
        from parkour.ledge_rail import RailLink
        from physics.collision import CollisionSystem
        result = None
        blockers = CollisionSystem.overlapping(BodyModel.box_at(blocked_pose), self.obstacles)
        travel = rail.tangent * float(direction)
        start = BodyModel.hang_center(rail, self.rail_s)
        if self.supported_hang:
            start[1] = self.supported_y
        grip = rail.get_point(self.rail_s)
        best_d = None
        for q in self.ledge_system.rails:
            if q is rail or not any(q.owner is b for b in blockers):
                continue
            if not (rail.height - config.EDGE_SWITCH_DOWN_MAX <= q.height <= rail.height + config.EDGE_SWITCH_UP_MAX):
                continue
            if float(np.dot(q.normal, travel)) > -0.5:
                continue                       # it must face back at us, like the wall of an inside corner
            s0 = q.clamp_s(q.project_s(start))
            if float(np.dot(start - q.get_point(s0), q.normal)) <= 0.0:
                continue                       # we are behind it
            step = 0.05 / max(q.length, 1e-6)
            for k in range(0, 21):
                found = None
                for sgn in ((1.0,) if k == 0 else (1.0, -1.0)):
                    s2 = s0 + sgn * k * step
                    if s2 < q.s_margin() - 1e-9 or s2 > 1.0 - q.s_margin() + 1e-9:
                        continue
                    end = BodyModel.hang_center(q, s2)
                    horiz = end - start
                    horiz[1] = 0.0
                    if float(np.linalg.norm(horiz)) > config.EDGE_SWITCH_REACH:
                        continue
                    if not BodyModel.pose_is_free(end, self.obstacles):
                        continue
                    path_ok = all(BodyModel.pose_is_free(start + (end - start) * t, self.obstacles, shrink=0.02)
                                  for t in np.linspace(0.1, 1.0, 10))
                    if path_ok:
                        found = s2
                        break
                if found is not None:
                    d = float(np.linalg.norm(BodyModel.hang_center(q, found) - start))
                    if best_d is None or d < best_d:
                        near_p1 = float(np.linalg.norm(q.p1 - grip)) <= float(np.linalg.norm(q.p2 - grip))
                        result = (RailLink(q, "right" if near_p1 else "left", RailLink.CONCAVE), float(found))
                        best_d = d
                    break
        self._switch_cache = (key, result)
        return result

    def _begin_edge_switch(self, rail, link, entry_s, pos):
        """Switch onto the edge of an obstruction met mid-traverse: an inside-corner move (turn to face
        it, hands onto it) that may also lower the body onto a lower edge."""
        self._begin_corner_turn(rail, link, entry_s, pos)
        self.edge_switch = True
        drop = max(0.0, rail.height - link.rail.height)
        total = (config.CORNER_TURN_TIME_CONCAVE + config.EDGE_SWITCH_TIME_PER_M * drop) * self.motion_scale
        end = BodyModel.hang_center(link.rail, entry_s)
        self._start_root_motion([(total, end)], self.motion_scale)
        self.turn_time_request = total
        self._set_event("EDGE_SWITCH")

    def _climb_shift_target(self, rail, look_dir):
        """Nearest spot within CLIMB_SHIFT_MAX along `rail` from which the climb (or the top-out)
        is clear, with a hangable pose all the way there; None if there is none.  Cached per spot
        so holding W against a real dead end costs nothing."""
        key = (id(rail), round(float(self.rail_s), 4))
        if self._shift_cache is not None and self._shift_cache[0] == key:
            return self._shift_cache[1]
        target = None
        length = max(rail.length, 1e-6)
        m = rail.s_margin()
        step = 0.1
        n = int(round(config.CLIMB_SHIFT_MAX / step))
        for k in range(1, n + 1):
            for sign in (1.0, -1.0):
                s2 = self.rail_s + sign * k * step / length
                if s2 < m - 1e-9 or s2 > 1.0 - m + 1e-9:
                    continue
                path_ok = True
                for j in range(1, 2 * k + 1):
                    sp = self.rail_s + sign * j * (step * 0.5) / length
                    c = BodyModel.hang_center(rail, sp)
                    if self.supported_hang:
                        c[1] = self.supported_y
                    if not BodyModel.pose_is_free(c, self.obstacles):
                        path_ok = False
                        break
                if not path_ok:
                    continue
                if ((rail.rail_type == RailType.MANTLE_TOP and BodyModel.mantle_is_valid(rail, s2, self.obstacles)) or
                        BodyModel.find_hold_above(self.ledge_system.rails, rail, s2, self.obstacles, look_dir) is not None):
                    target = float(s2)
                    break
            if target is not None:
                break
        self._shift_cache = (key, target)
        return target

    def _ground_adjusted_hang(self, center):
        """A hang pose whose feet would dip slightly into the ground rests them on it instead."""
        from physics.collision import CollisionSystem
        center = np.asarray(center, dtype=np.float32).copy()
        ground = CollisionSystem.ground_height_below(center + UP * 0.1, BodyModel.SIZE, self.obstacles, max_drop=0.3)
        if ground is not None:
            feet = BodyModel.feet_y(center)
            if ground - 0.06 <= feet < ground + 0.001:
                center[1] += (ground + 0.001) - feet
                self.supported_hang = True
                self.supported_y = float(center[1])
                return center
        self.supported_hang = False
        return center

    def _release_rail(self):
        self.active_rail = None
        self.grip_point = None
        self.flight_speed = 0.0             # a launch off a hold carries no run-up
        self.supported_hang = False
        self.corner_link = None
        self.climb_shift = None

    def _jump(self, pos, vel, push_dir=None, forward_dir=None):
        """Take off: a vertical impulse plus a push along the held direction (see jump_push).
        A forward leap (within 60 deg of where the player faces) leaves the ground with
        JUMP_FORWARD_BOOST times the horizontal momentum along its direction; backward and
        sideways leaps and straight-up jumps are unchanged."""
        self.state = ParkourState.AIRBORNE
        if push_dir is not None:
            push_dir = np.asarray(push_dir, dtype=np.float32)
            speed = float(np.hypot(vel[0], vel[2]))
            push = self.jump_push(speed) * push_dir
            vel[0] += push[0]
            vel[2] += push[2]
            amount = float(np.linalg.norm(push_dir))
            if forward_dir is not None and amount > 1e-4:
                d = push_dir / amount
                f = np.asarray(forward_dir, dtype=np.float32)
                if float(d[0] * f[0] + d[2] * f[2]) >= 0.5:
                    along = float(vel[0] * d[0] + vel[2] * d[2])
                    if along > 0.0:
                        extra = along * (config.JUMP_FORWARD_BOOST - 1.0) * min(1.0, amount)
                        vel[0] += d[0] * extra
                        vel[2] += d[2] * extra
        vel[1] = config.JUMP_IMPULSE
        self.flight_speed = float(np.hypot(vel[0], vel[2]))
        self.jump_timer = config.JUMP_HOLD_MAX
        self.coyote_timer = 0.0
        self.wall_kicked = False
        self.took_off = True
        self.flipped = False
        self.air_peak_y = float(pos[1]) - config.PLAYER_HALF_HEIGHT
        self._set_event("JUMP")
        return pos, vel

    # ------------------------------------------------------------- acrobatics
    def _try_flip(self, pos, vel, forward_dir, look_dir, intents):
        """Start a flip if there is airtime for one and no hold is about to come into reach (then the
        press is a reach, not a flip).  Returns True if the flip started."""
        feet = float(pos[1]) - config.PLAYER_HALF_HEIGHT
        ground = CollisionSystem.ground_height_below(pos, BodyModel.SIZE, self.obstacles, max_drop=60.0)
        t_land = acrobatics.time_to_land(feet, float(vel[1]), ground)
        if t_land is None:
            t_land = config.FLIP_MAX_TIME / config.FLIP_AIRTIME_USE + 1.0     # nothing below: plenty of time
        if t_land * config.FLIP_AIRTIME_USE < config.FLIP_MIN_TIME:
            return False
        rails = self.ledge_system.rails
        for t in config.FLIP_HOLD_LOOKAHEAD:
            ahead = pos + vel * t + UP * (0.5 * config.GRAVITY * t * t)
            if t > 0.0 and not BodyModel.pose_is_free(ahead, self.obstacles):
                break                            # the flight ends against something before then
            vel_t = vel + UP * (config.GRAVITY * t)
            if BodyModel.find_best_grab(rails, ahead, forward_dir, look_dir, False, self.obstacles, vel_t) is not None:
                return False
        # looking at a hold a leap away (the reticle is on it): the press means "reach for that"
        eye = BodyModel.eye_pos(pos)
        aimed = self.ledge_system.find_best_rail(eye, look_dir, max_dist=config.FLIP_AIM_CLEAR_DIST)
        if aimed[0] is not None and float(np.dot(aimed[1] - pos, vel)) > 0.0:
            return False
        speed_h = float(np.hypot(vel[0], vel[2]))
        kind = acrobatics.flip_kind(intents.forward, intents.back, intents.left, intents.right, speed_h)
        duration = float(np.clip(t_land * config.FLIP_AIRTIME_USE, config.FLIP_MIN_TIME, config.FLIP_MAX_TIME))
        self.spin = Spin(kind, duration, forward_dir)
        self.pending_spin = None
        self.flipped = True
        self.flip_count += 1
        self.jump_timer = 0.0
        self._set_event("FLIP")
        return True

    def _begin_roll(self, pos, vel, forward_dir):
        """Touch-down from high up: roll along the direction of travel (backward if falling backward)
        and keep the momentum."""
        fwd = np.array([forward_dir[0], 0.0, forward_dir[2]], dtype=np.float32)
        fwd /= (np.linalg.norm(fwd) + 1e-7)
        h = np.array([vel[0], 0.0, vel[2]], dtype=np.float32)
        speed = float(np.linalg.norm(h))
        if speed >= 1.0:
            d = h / speed
            backward = float(np.dot(d, fwd)) < -0.3
        else:
            d, backward = fwd, False
            if not BodyModel.pose_is_free(pos + d * 1.0, self.obstacles):
                d, backward = -fwd, True          # a wall in front: roll backward, away from it
        min_speed = config.ROLL_MIN_SPEED * (0.8 if backward else 1.0)
        self.roll_speed = float(min(config.SPRINT_SPEED, max(min_speed, speed * config.ROLL_SPEED_KEEP)))
        self.roll_dir = d.astype(np.float32)
        self.roll_time = 0.0
        spin = Spin(acrobatics.ROLL, config.ROLL_TIME, d, backward=backward)
        if self.spin is not None and not self.spin.done:
            # still coming round from a flip: roll as soon as it is finished
            self.pending_spin = spin
            self.roll_duration = config.ROLL_TIME + self.spin.remaining_time()
        else:
            self.spin = spin
            self.roll_duration = config.ROLL_TIME
        if not backward:
            # roll the way you are going: the view comes round to the direction of travel
            turn = float(np.degrees(np.arctan2(fwd[0] * d[2] - fwd[2] * d[0], fwd[0] * d[0] + fwd[2] * d[2])))
            if abs(turn) > 3.0:
                self.turn_request_deg = -turn
                self.turn_time_request = config.ROLL_TIME * 0.45
        self.state = ParkourState.ROLL
        self.roll_count += 1
        vel[0] = self.roll_dir[0] * self.roll_speed
        vel[2] = self.roll_dir[2] * self.roll_speed
        vel[1] = min(0.0, float(vel[1]))
        self._set_event("ROLL")

    def _roll_step(self, dt, pos, vel, intents, forward_dir, right_dir):
        """One tick of a roll.  Returns False once it is over (normal ground movement takes over this
        tick, with the momentum the roll kept)."""
        self.roll_time += dt
        frac = self.roll_time / max(1e-3, self.roll_duration)
        if frac >= 1.0:
            self.state = ParkourState.GROUND
            self._set_event("ROLL_DONE")
            return False
        if frac >= config.ROLL_JUMP_FROM and intents.has("jump"):
            # spring straight out of the roll
            intents.consume("jump")
            if self.spin is not None:
                self.spin.rush()
            self.pending_spin = None
            move_dir = self._move_dir(intents, forward_dir, right_dir)
            amount = self._move_amount(intents)
            self._jump(pos, vel, move_dir * amount if float(np.linalg.norm(move_dir)) > 1e-4 else None, forward_dir)
            return True
        self.roll_speed = max(0.0, self.roll_speed - config.ROLL_DECEL * dt)
        # a roll never carries you off the edge of what you landed on
        step = 0.32 + self.roll_speed * dt
        probe = pos + self.roll_dir * step
        feet = float(pos[1]) - config.PLAYER_HALF_HEIGHT
        ground = CollisionSystem.ground_height_below(probe + UP * 0.05, BodyModel.SIZE, self.obstacles, max_drop=2.0)
        if ground is None or ground < feet - 0.6:
            self.roll_speed = 0.0
        vel[0] = self.roll_dir[0] * self.roll_speed
        vel[2] = self.roll_dir[2] * self.roll_speed
        return True

    @staticmethod
    def jump_push(speed):
        """Horizontal speed the legs add at take-off: a full standing leap from rest, next to
        nothing at a sprint (momentum does the work)."""
        u = float(np.sqrt(np.clip(speed / config.SPRINT_SPEED, 0.0, 1.0)))
        push = config.JUMP_PUSH_STANDING + (config.JUMP_PUSH_SPRINT - config.JUMP_PUSH_STANDING) * u
        # (a first step never makes the leap shorter than from a standstill)
        push = max(push, config.JUMP_PUSH_STANDING - 0.75 * speed)
        return max(config.JUMP_PUSH_SPRINT, push)

    def run_target_speed(self):
        """Top speed allowed by the current sprint build-up: a jog at once, a sprint after RUN_BUILD_TIME."""
        k = float(np.clip(self.run_time / config.RUN_BUILD_TIME, 0.0, 1.0))
        k = k * k * (3.0 - 2.0 * k)
        return config.JOG_SPEED + (config.SPRINT_SPEED - config.JOG_SPEED) * k

    def _update_run_build(self, dt, vel, move_dir, is_sprinting):
        """Sprint build-up: fills while running hard in a steady direction, drains when easing off,
        and is lost outright on a sharp change of direction."""
        speed = float(np.hypot(vel[0], vel[2]))
        if is_sprinting and move_dir is not None and speed > 0.5:
            heading = np.array([vel[0], 0.0, vel[2]], dtype=np.float32) / speed
            if float(np.dot(heading, move_dir)) < config.RUN_BUILD_TURN_DOT:
                self.run_time = 0.0
            elif speed >= config.WALK_SPEED:
                self.run_time = min(config.RUN_BUILD_TIME, self.run_time + dt)
        else:
            self.run_time = max(0.0, self.run_time - dt * config.RUN_BUILD_DECAY)

    def _drop(self, rail, pos):
        self.state = ParkourState.AIRBORNE
        self.lockout_timer = config.DROP_LOCKOUT_TIME
        vel = rail.normal * 1.2 - UP * 0.5
        self._release_rail()
        self._set_event("DROP")
        return pos, vel

    @staticmethod
    def _move_amount(intents):
        """How far the movement input is pushed: 1 for keys, the stick's tilt for a controller."""
        if getattr(intents, "move_analog", None) is None:
            return 1.0
        return float(np.clip(getattr(intents, "move_amount", 1.0), 0.0, 1.0))

    @staticmethod
    def _move_dir(intents, forward_dir, right_dir):
        """Unit horizontal direction of the movement input (zero vector when there is none)."""
        analog = getattr(intents, "move_analog", None)
        if analog is not None:
            # A controller stick: any direction, not just the 8 of the keys
            move_dir = forward_dir * np.float32(analog[1]) + right_dir * np.float32(analog[0])
            move_dir = move_dir.astype(np.float32)
            move_dir[1] = 0.0
            n = float(np.linalg.norm(move_dir))
            if n > 1e-4:
                return move_dir / n
            return np.zeros(3, dtype=np.float32)
        move_dir = np.zeros(3, dtype=np.float32)
        if intents.forward:
            move_dir += forward_dir
        if intents.back:
            move_dir -= forward_dir
        if intents.right:
            move_dir += right_dir
        if intents.left:
            move_dir -= right_dir
        move_dir[1] = 0.0
        n = float(np.linalg.norm(move_dir))
        if n > 1e-4:
            move_dir /= n
        return move_dir

    @staticmethod
    def _approach(contact):
        """Horizontal speed the body had into the wall it just ran into (0 if none)."""
        if contact is None or contact.wall_normal is None:
            return 0.0
        return float(contact.approach_speed)

    @staticmethod
    def _accelerate(vel, target_vel, dt):
        """Human acceleration limits: brisk up to walking pace, then a sprint build-up."""
        cur = np.array([vel[0], 0.0, vel[2]], dtype=np.float32)
        speed = float(np.linalg.norm(cur))
        target_speed = float(np.linalg.norm(target_vel))
        if speed >= config.WALK_SPEED - 1e-3 and target_speed > speed:
            accel = config.SPRINT_ACCEL
        else:
            accel = config.WALK_ACCEL
        dv = target_vel - cur
        dv_len = float(np.linalg.norm(dv))
        max_dv = accel * dt
        if dv_len > max_dv:
            dv *= max_dv / dv_len
        vel[0] += dv[0]
        vel[2] += dv[2]

    def _auto_step(self, contact, pos, forward_dir, max_height):
        """Find a MANTLE_TOP rail on the wall we walked into that is low enough to step onto."""
        obs = contact.wall_obstacle
        if obs is None:
            return None
        top = float(obs.bounds.max[1])
        rise = top - BodyModel.feet_y(pos)
        if rise < config.GROUND_VAULT_MIN_HEIGHT or rise > max_height + 0.02:
            return None
        if float(np.dot(forward_dir, -contact.wall_normal)) < 0.5:
            return None
        for rail in obs.rails:
            if rail.rail_type != RailType.MANTLE_TOP:
                continue
            if float(np.dot(rail.normal, contact.wall_normal)) < 0.99:
                continue
            s = rail.clamp_s(rail.project_s(pos))
            if BodyModel.mantle_is_valid(rail, s, self.obstacles):
                return rail, s
        return None
