"""Deterministic headless simulation harness for Hardcore.

Drives the real PlayerController / ParkourStateMachine / CollisionSystem with
scripted keyboard input on a fixed 60 Hz clock.  No window or OpenGL needed.
"""
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config                                   # noqa: E402
from core.input_buffer import InputBuffer, Action   # noqa: E402
from parkour.ledge_system import LedgeSystem    # noqa: E402
from parkour.controller import PlayerController  # noqa: E402
from parkour.state_machine import ParkourState  # noqa: E402
from parkour.body import BodyModel              # noqa: E402
from parkour.ledge_rail import RailType         # noqa: E402
from world.generator import LevelGenerator      # noqa: E402
from world.obstacle import Obstacle             # noqa: E402
from animation.arm_rig import FirstPersonArmRig  # noqa: E402

DT = config.FIXED_DT

KEYS = {
    "W": Action.FORWARD, "S": Action.BACK, "A": Action.LEFT, "D": Action.RIGHT,
    "SPACE": Action.JUMP, "E": Action.GRAB, "SHIFT": Action.SPRINT,
    "CAPS": Action.RUN_TOGGLE, "CTRL": Action.LET_GO,
}


def build_playground():
    ls = LedgeSystem()
    obstacles, _, _ = LevelGenerator.build_playground(ls)
    return ls, obstacles


def build_custom(blocks, floor=True):
    """blocks: list of (x, y, z, w, h, d[, add_crevasses]) -> (ledge_system, obstacles)."""
    ls = LedgeSystem()
    obstacles = []
    if floor:
        fl = Obstacle(-60.0, -0.5, -60.0, 120.0, 0.5, 120.0, add_crevasses=False)
        fl.rails.clear()
        obstacles.append(fl)
    for b in blocks:
        add_crev = b[6] if len(b) > 6 else True
        obstacles.append(Obstacle(b[0], b[1], b[2], b[3], b[4], b[5], add_crevasses=add_crev))
    ls.build(obstacles)
    return ls, obstacles


class Sim:
    def __init__(self, ledge_system, obstacles, spawn=(0.0, 0.9, 0.0), yaw=0.0, pitch=0.0, with_arms=False,
                 scheme="flow"):
        self.time = 0.0
        self.input = InputBuffer(time_fn=lambda: self.time)
        self.player = PlayerController(ledge_system, obstacles, spawn_pos=spawn, yaw=yaw, pitch=pitch, scheme=scheme)
        self.obstacles = obstacles
        self.ledge_system = ledge_system
        self.arm_rig = FirstPersonArmRig() if with_arms else None
        self.zombie = False              # zombie tag: the first-person arms reach forward
        self.trace = []      # list of dicts per tick
        self.events = []     # (time, event)
        self.states = set()
        self.max_step = 0.0
        self.max_root_step = 0.0   # largest per-tick move while on a rail / in scripted root motion

    # --------------------------------------------------------------- input
    def hold(self, *keys):
        for k in keys:
            a = KEYS[k]
            if not self.input.is_down(a):
                self.input.set_key_down(a)
                self.player.scheme.on_key_down(a)

    def release(self, *keys):
        for k in keys:
            self.input.set_key_up(KEYS[k])

    def repress(self, key):
        """Generate a fresh press of a key that may already be held (key stays held)."""
        a = KEYS[key]
        self.input.set_key_up(a)
        self.input.set_key_down(a)
        self.player.scheme.on_key_down(a)

    def top_out_if_needed(self):
        """Climber bot helper: a fresh W press while hanging from a top edge (W held would also do)."""
        fsm = self.fsm
        if (self.state == ParkourState.LEDGE_HANG and fsm.active_rail is not None
                and fsm.active_rail.rail_type == RailType.MANTLE_TOP
                and fsm.settle_time >= config.GRAB_SETTLE_TIME):
            self.repress("W")

    def climb_to(self, target, timeout=14.0, keys=("W", "E"), hop=True):
        """
        Scripted climber: hold `keys`, look at the nearest bit of the target's top while on a
        rail, hop (fresh Space) when blocked by a wall on the ground, press W to top out, let go
        of dead-end holds and avoid them afterwards.  Returns True when standing on `target`.
        """
        b = target.bounds
        grab_keys = [k for k in keys if k in ("E", "SPACE")]
        self.hold(*keys)
        last_hop = -1.0
        blocked_since = None
        dead = []
        n = int(round(timeout / DT))

        def best_grab(exclude):
            return BodyModel.find_best_grab(self.ledge_system.rails, self.pos, self.player.forward_h, self.player.forward_3d,
                                            self.player.on_ground, self.player.obstacles, self.vel, exclude=exclude)

        for _ in range(n):
            # dead end (e.g. a crevasse right under a platform): let go and remember it
            if self.state == ParkourState.LEDGE_HANG and self.fsm.last_event == "CLIMB_BLOCKED":
                blocked_since = self.time if blocked_since is None else blocked_since
                if self.time - blocked_since > 0.5:
                    dead.append(self.fsm.active_rail)
                    self.tap("CTRL")
                    blocked_since = None
                    continue
            else:
                blocked_since = None
            if self.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY):
                aim = np.array([np.clip(self.pos[0], b.min[0] + 0.3, b.max[0] - 0.3), b.max[1],
                                np.clip(self.pos[2], b.min[2] + 0.3, b.max[2] - 0.3)], dtype=np.float32)
                self.look_at(aim)
            self.top_out_if_needed()
            c = self.player.contact
            grounded = self.player.on_ground and self.state in (ParkourState.GROUND, ParkourState.SPRINT)
            if grounded:
                # face whichever way offers the highest usable hold (e.g. the next scaffold platform behind you)
                yaw0 = self.player.yaw
                best_yaw, best_h = yaw0, -1e9
                for turn in (0.0, 90.0, -90.0, 180.0):
                    self.player.yaw = yaw0 + turn
                    cand = best_grab(dead)
                    if cand is not None and cand.rail.height <= b.max[1] + 0.05 and cand.rail.height > best_h + (0.01 if turn else 0.0):
                        best_yaw, best_h = yaw0 + turn, cand.rail.height
                self.player.yaw = best_yaw
                if best_h < -1e8:
                    # no usable hold within arm's reach: walk toward the nearest reachable-height hold nearby
                    feet = self.feet_y
                    near = BodyModel._holds_near(self.ledge_system.rails, self.pos, feet + 0.45, feet + config.STAND_REACH, radius=2.5)
                    best_d, best_dir = 1e9, None
                    for i in near:
                        r = self.ledge_system.rails[i]
                        if any(r is d for d in dead):
                            continue
                        s_r = r.clamp_s(r.project_s(self.pos))
                        g = r.get_point(s_r)
                        if g[1] > b.max[1] + 0.05 or not BodyModel.pose_is_free(BodyModel.hang_center(r, s_r), self.obstacles):
                            continue
                        d = g - self.pos
                        d[1] = 0.0
                        dn = float(np.linalg.norm(d))
                        if 0.2 < dn < best_d:
                            best_d, best_dir = dn, d
                    if best_dir is not None:
                        self.face(best_dir)
                # only reach when the hold we would close on is not a known dead end
                cand_now = best_grab(None)
                if cand_now is not None and any(cand_now.rail is d for d in dead):
                    self.release(*grab_keys)
                else:
                    self.hold(*grab_keys)
            if hop and grounded and self.time - last_hop > 0.6:
                blocked = c is not None and c.wall_normal is not None
                feet = self.feet_y
                overhead = BodyModel._holds_near(self.ledge_system.rails, self.pos, feet + config.STAND_REACH - 0.1,
                                                 feet + 3.3, radius=1.0)
                if (blocked or overhead) and best_grab(dead) is None:
                    self.repress("SPACE")
                    last_hop = self.time
            self.step()
            if self.standing_on(target):
                return True
        return False

    def release_all(self):
        for a in list(self.input.held_keys):
            self.input.set_key_up(a)

    def tap(self, *keys):
        """Press keys this tick and release them next tick."""
        for k in keys:
            self.input.set_key_down(KEYS[k])
            self.player.scheme.on_key_down(KEYS[k])
        self.step()
        for k in keys:
            self.input.set_key_up(KEYS[k])

    # ---------------------------------------------------------------- pose
    @property
    def pos(self):
        return self.player.pos

    @property
    def vel(self):
        return self.player.vel

    @property
    def state(self):
        return self.player.state

    @property
    def fsm(self):
        return self.player.fsm

    @property
    def feet_y(self):
        return self.player.feet_y

    def teleport(self, pos, yaw=None):
        self.player.pos = np.array(pos, dtype=np.float32)
        self.player.vel[:] = 0.0
        self.player.on_ground = True
        self.player.contact = None
        self.player.fsm.air_peak_y = None        # a teleport is not a fall
        if yaw is not None:
            self.player.yaw = yaw

    def face(self, direction):
        self.player.face_direction(direction)

    def look_at(self, point):
        self.player.look_at_point(point)

    def set_pitch(self, pitch):
        self.player.pitch = pitch

    # ---------------------------------------------------------------- time
    def step(self, n=1):
        for _ in range(n):
            before = self.pos.copy()
            self.time += DT
            self.player.step(DT, self.input)
            d = float(np.linalg.norm(self.pos - before))
            self.max_step = max(self.max_step, d)
            if self.state in ParkourState.ROOT_MOTION:
                self.max_root_step = max(self.max_root_step, d)
            ev = self.fsm.last_event
            if ev:
                self.events.append((round(self.time, 3), ev))
            self.states.add(self.state)
            self.trace.append({
                "t": self.time, "pos": self.pos.copy(), "vel": self.vel.copy(),
                "state": self.state, "on_ground": self.player.on_ground, "step": d,
            })
            if self.arm_rig is not None:
                tgt = self.ledge_system.find_best_rail(self.player.eye_pos, self.player.forward_3d)
                tgt = None if tgt[0] is None else tgt
                self.arm_rig.update(
                    DT, self.pos, self.player.camera_pos, self.player.yaw, self.player.pitch,
                    self.state, self.fsm.active_rail, tgt, self.vel, self.player.on_ground,
                    grip_point=self.fsm.grip_point, climb_phase=self.fsm.climb_phase,
                    climb_next=self.fsm.climb_next,
                    grab_candidate=self.player.grab_candidate() if self.fsm.spin is None else None,
                    shimmy_dir=self.fsm.shimmy_dir, corner_link=self.fsm.corner_link,
                    corner_entry_s=self.fsm.corner_entry_s, mantle_reverse=self.fsm.mantle_reverse,
                    climb_direction=self.fsm.climb_direction, supported_hang=self.fsm.supported_hang,
                    spin=self.fsm.spin, zombie=self.zombie, move_rate=self.fsm.move_rate,
                    move_time_left=self.fsm.move_time_left
                )

    def run(self, seconds, until=None, top_out=False):
        """Advance time; stop early when `until(sim)` becomes true. Returns True if it did.
        top_out=True makes the scripted climber press W to top out whenever it hangs from a top edge."""
        n = int(round(seconds / DT))
        for _ in range(n):
            if top_out:
                self.top_out_if_needed()
            self.step()
            if until is not None and until(self):
                return True
        return until is None

    def run_until_state(self, state, timeout):
        return self.run(timeout, until=lambda s: s.state == state)

    def run_until_event(self, event, timeout):
        start = len(self.events)
        return self.run(timeout, until=lambda s: any(e == event for _, e in s.events[start:]))

    def had_event(self, event):
        return any(e == event for _, e in self.events)

    def settle(self, seconds=0.5):
        self.release_all()
        self.run(seconds)

    # ------------------------------------------------------------- helpers
    def walk_to(self, x, z, speed_key=None, timeout=20.0, tol=0.25):
        """Walk on the ground to (x, z) facing the target."""
        def arrived(s):
            return float(np.hypot(s.pos[0] - x, s.pos[2] - z)) < tol
        t = 0.0
        while t < timeout:
            if arrived(self):
                break
            self.face(np.array([x - self.pos[0], 0.0, z - self.pos[2]]))
            self.hold("W")
            if speed_key:
                self.hold(speed_key)
            self.step()
            t += DT
        self.release("W", "SHIFT")
        self.run(0.4)
        return arrived(self)

    def max_height_reached(self, key="pos"):
        return max(tr["pos"][1] for tr in self.trace) if self.trace else self.pos[1]

    def standing_on(self, obstacle, tol=0.03):
        b = obstacle.bounds
        p = self.pos
        fy = self.feet_y
        return (self.player.on_ground and abs(fy - float(b.max[1])) < tol and
                b.min[0] - 0.3 <= p[0] <= b.max[0] + 0.3 and b.min[2] - 0.3 <= p[2] <= b.max[2] + 0.3)


def obstacle_by_size(obstacles, w, h, d, tol=1e-3):
    for o in obstacles:
        sz = o.bounds.size
        if abs(sz[0] - w) < tol and abs(sz[1] - h) < tol and abs(sz[2] - d) < tol:
            return o
    raise KeyError((w, h, d))
