"""Bot AI: a player that runs, climbs and leaps through the level on its own.

The bot drives a real PlayerController through the same inputs a person has (keys
into an InputBuffer, a look direction), so everything it does is something a
player can do and it needs nothing from the renderer.  It is the seed of the
server-side bot AI used to exercise multiplayer.

Behaviour
---------
* Goals: the cheapest target the game offers (game/match.py `bot_targets`: the
  emblems still out there, the flag to take, the base to bring it to, the thief to
  tag - distance plus a climbing cost), or - when there are none - a random
  building roof, preferring neighbours so it leaps from roof to roof.  A goal that
  makes no progress for a while is given up.  Moving targets (a player carrying a
  flag) are followed: the goal moves with them and the route is planned again.
* Ground: sprint (sprint lock on) toward the goal; walls in the way are grabbed and
  climbed (everything is climbable), low things are hopped or vaulted.  A tall wall it
  means to climb is met with a wall run: it jumps a stride out, plants a foot and
  catches the first-storey hold with the momentum of the run (then climbs in flow).
* Hanging: look at the goal and climb; if the climb is blocked shimmy toward the
  goal, and as a last resort let go and remember the hold as a dead end.
* Roofs: walk to the goal; at an edge leap if there is something to land on within
  reach, otherwise drop off (there is no fall damage).
* Stuck for long: ask to be respawned (the server puts it back on a spawn point).
"""
import math

import numpy as np

import config
from core.input_buffer import Action
from parkour.body import BodyModel
from parkour.state_machine import ParkourState
from physics.collision import CollisionSystem

KEY = {"W": Action.FORWARD, "S": Action.BACK, "A": Action.LEFT, "D": Action.RIGHT, "SPACE": Action.JUMP,
       "E": Action.GRAB, "CTRL": Action.LET_GO, "CAPS": Action.RUN_TOGGLE}

HANG = (ParkourState.LEDGE_HANG, ParkourState.SHIMMY)
ROOT = (ParkourState.CLIMB_UP, ParkourState.MANTLE_VAULT, ParkourState.CORNER_TURN)


def _flat(v):
    return np.array([v[0], 0.0, v[2]], dtype=np.float32)


class BotBrain:
    GOAL_TIMEOUT = 22.0          # give a goal up if it has not got closer for this long
    STUCK_RESPAWN = 30.0         # ask for a respawn if nothing has got closer for this long
    LEAP_MAX = 4.2               # horizontal gap it will try to leap across
    WANDER_RADIUS = 60.0
    WALL_RUN_TAKEOFF = (1.2, 2.1)  # take off for a wall run with the wall face this far ahead of the body centre
    WALL_RUN_MIN_SPEED = config.JOG_SPEED - 0.3
    REACH_UP_MAX = config.STAND_REACH + 1.0   # a surface this high over the feet is reached with a hop
    CORNER_SPEED = 5.0           # a sharp bend in the route (an alley mouth) is taken no faster than this ...
    CORNER_ANGLE_COS = 0.64      # ... when it turns more than ~50 degrees

    def __init__(self, world, player, input_buf, seed=0, name="Bot"):
        self.world = world
        self.player = player
        self.input = input_buf
        self.rng = np.random.RandomState(seed)
        self.name = name
        self.held = set()
        self.goal = None             # dict(kind, point, building, emblem_id)
        self.best_dist = None
        self.since_progress = 0.0
        self.since_any_progress = 0.0
        self.blacklist = {}          # goal key -> seconds left
        self.dead_holds = {}         # id(rail) -> seconds left
        self.blocked_for = 0.0
        self.climb_blocked_for = 0.0
        self.shimmy_dir = 0
        self.dropping = False
        self.wants_respawn = False
        self.time = 0.0
        self.last_hop = -10.0
        self._started = False
        self.leaps = 0
        self.climbs = 0
        self.route = []              # [(x, z)] waypoints through the streets to the foot of the goal
        self.route_goal = None
        self.route_age = 0.0
        self.waypoint_best = None
        self.was_elevated = False
        self.detour = None           # (point, seconds left): walk here first (round an unclimbable spot)
        self.detour_sign = 1.0
        self.detours = 0
        self.follow_age = 0.0
        self.wall_runs = 0

    # ------------------------------------------------------------------ keys
    def _set_keys(self, keys):
        keys = set(keys)
        for k in keys - self.held:
            self.input.set_key_down(KEY[k])
            self.player.scheme.on_key_down(KEY[k])
        for k in self.held - keys:
            self.input.set_key_up(KEY[k])
        self.held = keys

    def _tap(self, k):
        """A fresh press (released on the next decision that does not hold it)."""
        self.input.set_key_up(KEY[k])
        self.input.set_key_down(KEY[k])
        self.player.scheme.on_key_down(KEY[k])
        self.held.add(k)

    def _face(self, direction, pitch=None):
        d = _flat(direction)
        if float(np.linalg.norm(d)) > 1e-4:
            self.player.face_direction(d)
        if pitch is not None:
            self.player.pitch = float(pitch)

    # ----------------------------------------------------------------- goals
    def _goal_key(self, goal):
        if goal["kind"] == "target":
            return ("target", goal["target_id"])
        return ("point", tuple(np.round(goal["point"], 0)))

    def _pick_goal(self, targets):
        pos = self.player.pos
        feet = self.player.feet_y
        options = []
        for e in targets:
            key = ("target", e.id)
            if self.blacklist.get(key, 0.0) > 0.0 and len(targets) > 1:
                continue
            d = float(np.linalg.norm(_flat(e.pos - pos)))
            climb = max(0.0, float(e.pos[1]) - 1.0 - feet)
            radius = float(getattr(e, "radius", config.MP_PICKUP_RADIUS))
            options.append((d + 2.5 * climb + self.rng.rand() * 4.0,
                            {"kind": "target", "point": np.asarray(e.pos, dtype=np.float32).copy(),
                             "target_id": e.id, "radius": radius, "moving": getattr(e, "kind", "") == "player"}))
        if options:
            options.sort(key=lambda o: o[0])
            return options[0][1]
        return self._wander_goal()

    def _wander_goal(self):
        pos = self.player.pos
        near = []
        for spec, parts in self.world.buildings:
            c = np.array([spec.x + spec.width * 0.5, spec.height, spec.z + spec.depth * 0.5], dtype=np.float32)
            d = float(np.linalg.norm(_flat(c - pos)))
            if d < self.WANDER_RADIUS and self.blacklist.get(("point", tuple(np.round(c, 0))), 0.0) <= 0.0:
                near.append((d, spec, c))
        if near:
            near.sort(key=lambda t: t[0])
            # mostly a neighbour (roof hopping), sometimes further away
            pick = near[int(self.rng.randint(min(4, len(near))))] if self.rng.rand() < 0.7 else \
                near[int(self.rng.randint(len(near)))]
            spec, c = pick[1], pick[2]
            x = spec.x + 1.5 + self.rng.rand() * max(0.1, spec.width - 3.0)
            z = spec.z + 1.5 + self.rng.rand() * max(0.1, spec.depth - 3.0)
            top = self.world.ground_below(x, z, spec.height + 3.0)
            y = (top if top is not None else spec.height) + 1.0
            return {"kind": "roof", "point": np.array([x, y, z], dtype=np.float32), "building": spec}
        sp = self.world.spawn_points[int(self.rng.randint(len(self.world.spawn_points)))]
        return {"kind": "street", "point": sp + np.array([self.rng.uniform(-20, 20), 0.1, self.rng.uniform(-20, 20)],
                                                          dtype=np.float32)}

    def _goal_reached(self):
        g = self.goal
        d = self.player.pos - g["point"]
        if g["kind"] == "target":
            return float(np.linalg.norm(d)) <= g.get("radius", config.MP_PICKUP_RADIUS)
        return float(np.linalg.norm(_flat(d))) < 1.5 and abs(float(self.player.feet_y) - (float(g["point"][1]) - 1.0)) < 0.6

    def set_emblems_taken(self, taken_ids):
        """The server tells the bot an emblem (any target) is gone: drop it as a goal."""
        if self.goal is not None and self.goal["kind"] == "target" and self.goal["target_id"] in taken_ids:
            self.goal = None

    def _follow(self, targets, dt):
        """A goal that moves (a player carrying a flag): keep the goal on it, plan the route again
        when it has gone somewhere else."""
        g = self.goal
        for e in targets:
            if e.id != g["target_id"]:
                continue
            new = np.asarray(e.pos, dtype=np.float32)
            moved = float(np.linalg.norm(new - g["point"]))
            if moved > 0.05:
                g["point"] = new.copy()
                self.follow_age += dt
                if moved > 0.5:
                    g["box"] = self.world.surface_box(float(new[0]), float(new[2]), float(new[1]) - 1.0)
                anchor = g.get("route_from", new)
                if float(np.linalg.norm(new - anchor)) > 4.0 or self.follow_age > 2.0:
                    g["route_from"] = new.copy()
                    self.follow_age = 0.0
                    self.route_goal = None                 # plan again toward where it is now
                    self.best_dist = None
            return

    # ---------------------------------------------------------------- update
    def update(self, dt, emblems=()):
        """Decide this tick's inputs (call before PlayerController.step).  `emblems`: what to go for
        (game.match.Target / Emblem: .id, .pos and optionally .radius, .kind)."""
        self.time += dt
        p = self.player
        if not self._started:
            self._started = True
            if not getattr(p.scheme, "run_mode", False):
                self._tap("CAPS")
        for table in (self.blacklist, self.dead_holds):
            for k in list(table):
                table[k] -= dt
                if table[k] <= 0.0:
                    del table[k]

        if self.goal is not None and self.goal["kind"] == "target":
            if not any(e.id == self.goal["target_id"] for e in emblems):
                self.goal = None
            else:
                self._follow(emblems, dt)
        elif self.goal is not None and any(self.blacklist.get(("target", e.id), 0.0) <= 0.0 for e in emblems):
            self.goal = None                       # something to do turned up: stop wandering
        reached = self.goal is not None and self._goal_reached()
        if self.goal is None or reached:
            new = self._pick_goal(emblems)
            if (reached and new["kind"] == "target" and self.goal["kind"] == "target"
                    and new["target_id"] == self.goal["target_id"]):
                # still the thing to do and we are there (guarding a base): stay with it
                self.since_progress = 0.0
                self.since_any_progress = 0.0
            else:
                self.goal = new
                self.best_dist = None
                self.since_progress = 0.0
                g = self.goal["point"]
                self.goal["box"] = self.world.surface_box(float(g[0]), float(g[2]), float(g[1]) - 1.0)
                self.route = []
                self.route_goal = None

        # progress bookkeeping (distance in 3D, so climbing up counts)
        dist = float(np.linalg.norm(p.pos - self.goal["point"]))
        if self.best_dist is None:
            # a fresh goal (or the same one picked again after giving up on it) is not progress in
            # itself: only getting closer is - or a bot stuck on its last target would never respawn
            self.best_dist = dist
            self.since_progress = 0.0
        elif dist < self.best_dist - 0.5:
            self.best_dist = dist
            self.since_progress = 0.0
            self.since_any_progress = 0.0
        else:
            self.since_progress += dt
            self.since_any_progress += dt
        if self.since_progress > self.GOAL_TIMEOUT:
            self.blacklist[self._goal_key(self.goal)] = 40.0
            self.goal = None
            self.since_progress = 0.0
            self._set_keys({"CAPS"} & self.held)
            return
        if self.since_any_progress > self.STUCK_RESPAWN:
            self.wants_respawn = True
            self.since_any_progress = 0.0

        state = p.state
        if state in HANG:
            self._hanging(dt)
        elif state in ROOT:
            self._set_keys({"W"})
        elif state in (ParkourState.AIRBORNE, ParkourState.DYNO_LAUNCH, ParkourState.WALL_EJECT):
            self._airborne()
        else:
            self._grounded(dt)

    # ----------------------------------------------------------------- states
    def _aim_point(self):
        """Where to look while climbing: the nearest bit of the goal's roof / box top (the proven
        scripted-climber aim), or straight up the wall when climbing something else."""
        g = self.goal["point"]
        p = self.player.pos
        box = self.goal.get("box")
        r = self.player.fsm.active_rail
        if box is not None:
            return np.array([np.clip(p[0], box.min[0] + 0.3, box.max[0] - 0.3), float(box.max[1]),
                             np.clip(p[2], box.min[2] + 0.3, box.max[2] - 0.3)], dtype=np.float32)
        if r is not None:
            up = r.get_point(self.player.fsm.rail_s) - r.normal * 0.6
            up[1] = r.height + 2.0
            return up
        return g

    def _on_goal_obstacle(self, rail):
        box = self.goal.get("box")
        if box is None or rail is None:
            return False
        gp = rail.get_point(self.player.fsm.rail_s)
        return (box.min[0] - 1.0 <= gp[0] <= box.max[0] + 1.0 and box.min[2] - 1.0 <= gp[2] <= box.max[2] + 1.0)

    # --------------------------------------------------------------- routing
    def _plan_route(self):
        """Streets to the foot of the goal (the goal itself when it is on the ground)."""
        nav = self.world.nav
        g = self.goal["point"]
        box = self.goal.get("box")
        if box is not None:
            goals = nav.cells_around_box(box.min, box.max)
        else:
            c = nav.nearest_free(float(g[0]), float(g[2]))
            goals = {c} if c is not None else set()
        path = nav.path((float(self.player.pos[0]), float(self.player.pos[2])), goals) if goals else None
        self.route = list(path) if path else []
        self.route_goal = self._goal_key(self.goal)
        self.route_age = 0.0
        self.waypoint_best = None

    def _steer_target(self, dt):
        """The point to run at on the ground: the next waypoint, or the goal once the route is done."""
        p = self.player.pos
        if self.route_goal != self._goal_key(self.goal):
            self._plan_route()
        self.route_age += dt
        while self.route:
            wx, wz = self.route[0]
            d = math.hypot(wx - float(p[0]), wz - float(p[2]))
            if d < 0.5 or (d < 1.2 and self._next_in_sight(float(p[0]), float(p[2]))):
                self.route.pop(0)
                self.waypoint_best = None
                continue
            if self.waypoint_best is None or d < self.waypoint_best - 0.3:
                self.waypoint_best = d
                self.route_age = 0.0
            elif self.route_age > 3.0:
                self._plan_route()                 # not getting closer: plan again from here
                if not self.route:
                    break
                wx, wz = self.route[0]
            run_in = self._run_in_point(wx, wz) if len(self.route) == 1 else None
            if run_in is not None:
                return run_in
            return np.array([wx, float(p[1]), wz], dtype=np.float32)
        return self.goal["point"]

    def _reach_up_onto(self, box):
        """The goal is on top of something a hop can reach (a bus-shelter roof, an awning, a container,
        a van) and no wall is in the way to climb from: get out from under it, go to its nearest edge
        and hop for it with the hands reaching.  Returns True when it set this tick's keys."""
        p = self.player
        x, z = float(p.pos[0]), float(p.pos[2])
        cx = float(np.clip(x, box.min[0], box.max[0]))
        cz = float(np.clip(z, box.min[2], box.max[2]))
        out = np.array([x - cx, 0.0, z - cz], dtype=np.float32)
        dist = float(np.linalg.norm(out))
        if dist > 4.0:
            return False                               # still on the way: the normal run gets us there
        if dist < 0.45:
            # under it (or right at its edge): out from under by the nearest side first
            sides = [(x - float(box.min[0]), (-1.0, 0.0)), (float(box.max[0]) - x, (1.0, 0.0)),
                     (z - float(box.min[2]), (0.0, -1.0)), (float(box.max[2]) - z, (0.0, 1.0))]
            _, (ox, oz) = min(sides, key=lambda sd: sd[0])
            self._face(np.array([ox, 0.0, oz], dtype=np.float32), pitch=0.0)
            self._set_keys({"W"})
            return True
        toward = -out / dist
        self._face(toward, pitch=20.0)
        if dist > 1.3:
            self._set_keys({"W"})                      # run in at its nearest edge (not its middle)
            return True
        cand = p.grab_candidate()
        if cand is not None and id(cand.rail) not in self.dead_holds:
            self._set_keys({"W", "E"})
            return True
        if self.time - self.last_hop > 0.7:
            self.last_hop = self.time
            self._set_keys({"W", "E"})
            self._tap("SPACE")                         # hop and reach for the edge
            return True
        self._set_keys({"W", "E"})
        return True

    def _brake_for_turn(self, speed):
        """True when the next waypoint is close and the route bends sharply there, and we are going
        faster than the bend can be taken."""
        if speed <= self.CORNER_SPEED or len(self.route) < 2:
            return False
        p = self.player.pos
        (wx, wz), (nx, nz) = self.route[0], self.route[1]
        a = np.array([wx - float(p[0]), wz - float(p[2])])
        b = np.array([nx - wx, nz - wz])
        da, db = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if da < 1e-3 or db < 1e-3 or da > 1.2 + 0.15 * speed:
            return False
        return float(np.dot(a, b)) / (da * db) < self.CORNER_ANGLE_COS

    def _next_in_sight(self, x, z):
        """Near a waypoint: is the one after it in a straight line from here?  (A waypoint at the mouth
        of a narrow alley only counts as reached once the alley is in line - cutting the corner
        from a metre off would run into the building beside it.)  True for the last waypoint."""
        if len(self.route) < 2:
            return True
        nav = self.world.nav
        nx, nz = self.route[1]
        return nav.line_clear(nav.cell_of(x, z), nav.cell_of(nx, nz))

    def _run_in_point(self, wx, wz):
        """Last leg to the foot of the goal's building: rather than run to the waypoint beside it and
        turn there (meeting the wall at a walk), run straight at the bit of wall next to it - at pace,
        so the climb starts with a wall run.  Only when the way there is clear and the wall is met
        squarely enough; None otherwise."""
        box = self.goal.get("box")
        if box is None:
            return None
        p = self.player.pos
        fx = float(np.clip(wx, box.min[0], box.max[0]))
        fz = float(np.clip(wz, box.min[2], box.max[2]))
        n = np.array([wx - fx, 0.0, wz - fz], dtype=np.float32)        # out of the face, toward the waypoint
        nn = float(np.linalg.norm(n))
        to = np.array([fx - float(p[0]), 0.0, fz - float(p[2])], dtype=np.float32)
        dist = float(np.linalg.norm(to))
        if nn < 0.3 or dist < 2.5:
            return None
        if float(np.dot(-to / dist, n / nn)) < 0.7:
            return None                                  # would meet the wall at a glancing angle
        nav = self.world.nav
        if not nav.line_clear(nav.cell_of(float(p[0]), float(p[2])), nav.cell_of(wx, wz)):
            return None
        return np.array([fx, float(p[1]), fz], dtype=np.float32)

    def _hanging(self, dt):
        p = self.player
        fsm = p.fsm
        rail = fsm.active_rail
        goal = self.goal["point"]
        # caught something while the goal is below (a drop that snagged, a wall that is not ours): let go
        if (float(goal[1]) < rail.height - 2.5 and not self._on_goal_obstacle(rail)
                and id(rail) not in self.dead_holds):
            self.route_goal = None                 # and find a way round
            self.dead_holds[id(rail)] = 4.0
            self._let_go()
            return
        if id(rail) in self.dead_holds:
            self._let_go()
            return
        aim = self._aim_point()
        eye = p.eye_pos
        to = aim - eye
        p.look_at_point(eye + to)
        if fsm.last_event in ("CLIMB_BLOCKED", "MANTLE_BLOCKED", "SHIMMY_BLOCKED", "EDGE_END") and fsm.climb_shift is None:
            self.climb_blocked_for += dt
        else:
            self.climb_blocked_for = max(0.0, self.climb_blocked_for - dt * 0.5)
        keys = {"W"}
        if self.climb_blocked_for > 0.6:
            # blocked straight up: traverse toward the goal side, then give up on this hold
            if self.shimmy_dir == 0:
                side = float(np.dot(_flat(goal - rail.get_point(fsm.rail_s)), rail.tangent))
                self.shimmy_dir = 1 if side >= 0.0 else -1
                if abs(side) < 0.5:
                    self.shimmy_dir = 1 if self.rng.rand() < 0.5 else -1
            keys = {"D" if self.shimmy_dir > 0 else "A"}
            if self.climb_blocked_for > 3.5:
                self.dead_holds[id(rail)] = 8.0
                self.climb_blocked_for = 0.0
                self.shimmy_dir = 0
                self._let_go()
                return
        else:
            self.shimmy_dir = 0
        self._set_keys(keys)

    def _let_go(self):
        """Drop off the hold on purpose - and do not reach for the wall again on the way down."""
        self._set_keys(set())
        self._tap("CTRL")
        self.dropping = True

    def _airborne(self):
        p = self.player
        d = _flat(self.goal["point"] - p.pos)
        self._face(d)
        if self.dropping:
            self._set_keys({"W"})                     # falling on purpose: do not reach for holds
        else:
            # reach with E (catch the edge we are leaping for); Space stays held only if it still is
            # from the take-off - pressing it afresh in the air would be a flip
            self._set_keys({"W", "E"} | ({"SPACE"} & self.held))

    def _ahead_ground(self, direction, dist):
        p = self.player
        q = p.pos + _flat(direction) * dist
        return self.world.ground_below(float(q[0]), float(q[2]), float(p.feet_y) + 1.0)

    def _landing_ahead(self, direction):
        """Something to land on (or catch the edge of, reaching in the air) across a gap: a surface at
        most 3 m below / 1.2 m above the feet."""
        feet = float(self.player.feet_y)
        for dist in np.arange(1.5, self.LEAP_MAX + 0.01, 0.5):
            q = self.player.pos + _flat(direction) * float(dist)
            top = self.world.ground_below(float(q[0]), float(q[2]), feet + 2.0)
            if top is not None and feet - 3.0 <= top <= feet + 1.2:
                return float(dist)
        return None

    def _wall_run_facing(self, d):
        """Facing for a wall run if now is the moment to take off for one (None otherwise): running at
        pace toward a tall wall that is to be climbed, its face squarely ahead at take-off distance,
        with a hold on it that the run-up can reach."""
        p = self.player
        if float(np.hypot(p.vel[0], p.vel[2])) < self.WALL_RUN_MIN_SPEED or self.time - self.last_hop < 0.7:
            return None
        if p.contact is not None and p.contact.wall_normal is not None:
            return None                               # already against it: the standing climb takes over
        feet = float(p.feet_y)
        lift = np.array([0.0, 1.3, 0.0], dtype=np.float32)
        near, far = self.WALL_RUN_TAKEOFF
        if not BodyModel.pose_is_free(p.pos + d * near + lift, self.world.obstacles):
            return None                               # too close to take off cleanly
        obs = CollisionSystem.overlaps_any(BodyModel.box_at(p.pos + d * far + lift), self.world.obstacles)
        if obs is None or float(obs.bounds.max[1]) - feet < 2.6 or not self._should_climb_obstacle(obs):
            return None                               # nothing tall ahead, or not ours to climb
        rails = self.world.ledge_system.rails
        best = None
        for i in BodyModel._holds_near(rails, p.pos + d * far, feet + 1.8, feet + 4.9, radius=1.2):
            r = rails[i]
            if r.owner is not obs or id(r) in self.dead_holds:
                continue
            facing = -r.normal
            if float(np.dot(facing, d)) < 0.7:
                continue                              # not square enough to run up
            if best is None or r.height > best.height:
                best = r
        if best is None:
            return None
        f = -best.normal
        return np.array([f[0], 0.0, f[2]], dtype=np.float32)

    def _should_climb(self, contact):
        """Climb what blocks us only if it is the goal (its building / box), something low enough to
        vault or hop, or the end of the route (we are at the foot of the goal)."""
        return self._should_climb_obstacle(getattr(contact, "wall_obstacle", None))

    def _should_climb_obstacle(self, obs):
        if obs is None:
            return True
        b = obs.bounds
        feet = float(self.player.feet_y)
        if (self.goal is not None and float(self.goal["point"][1]) - 1.0 < feet - 1.5
                and float(b.max[1]) - feet > config.AUTO_STEP_MAX_HEIGHT + 0.1):
            # on the way down (off a roof to the street): go round what is in the way, never up it -
            # once up, the hanging rule lets go again (the goal is below), over and over
            return False
        if float(b.max[1]) - float(self.player.feet_y) <= 2.4:
            return True
        box = self.goal.get("box")
        if box is not None and (b.min[0] <= box.max[0] + 1.0 and b.max[0] >= box.min[0] - 1.0 and
                                b.min[2] <= box.max[2] + 1.0 and b.max[2] >= box.min[2] - 1.0):
            return True
        return not self.route

    def _start_detour(self, wall_n, goal_dir):
        """Blocked by a wall with nothing to grab here (a door recess, a blank spot): back off and go a
        few metres along the wall - toward the goal's side first, the other side if that failed."""
        t = np.cross(_flat(wall_n), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        t = t / (float(np.linalg.norm(t)) + 1e-7)
        toward = float(np.dot(t, goal_dir))
        sign = (1.0 if toward >= 0.0 else -1.0) if abs(toward) > 0.2 else self.detour_sign
        if self.detours % 2 == 1:
            sign = -sign                             # the last detour did not get us past: try the other way
        self.detour_sign = sign
        point = self.player.pos + t * sign * 3.5 + _flat(wall_n) * 0.8
        self.detour = (point.astype(np.float32), 3.0)
        self.detours += 1
        self.blocked_for = 0.0

    def _grounded(self, dt):
        p = self.player
        self.dropping = False
        if self.detour is not None:
            point, left = self.detour
            to_d = _flat(point - p.pos)
            if float(np.linalg.norm(to_d)) < 0.6 or left <= 0.0:
                self.detour = None
            else:
                self.detour = (point, left - dt)
                self._face(to_d, pitch=0.0)
                self._set_keys({"W"})
                return
        goal = self.goal["point"]
        feet_now = float(p.feet_y)
        ground_here = self.world.ground_below(float(p.pos[0]), float(p.pos[2]), feet_now + 0.5)
        elevated = feet_now > 1.0 and ground_here is not None and ground_here > 1.0
        if self.was_elevated and not elevated:
            self.route_goal = None                 # back on the street: plan from here
        self.was_elevated = elevated
        steer = goal if elevated else self._steer_target(dt)
        to = _flat(steer - p.pos)
        dist_h = float(np.linalg.norm(to))
        d = to / dist_h if dist_h > 1e-4 else p.forward_h
        feet = float(p.feet_y)
        goal_surface = float(goal[1]) - 1.0
        contact = p.contact
        speed = float(np.hypot(p.vel[0], p.vel[2]))
        blocked = contact is not None and contact.wall_normal is not None and speed < 1.5
        self.blocked_for = self.blocked_for + dt if blocked else 0.0

        box = self.goal.get("box")
        rise = goal_surface - feet
        if dist_h < 0.6 and rise <= config.AUTO_STEP_MAX_HEIGHT:
            # right under / next to it: step onto it (the pickup is the server's business)
            self._face(d, pitch=0.0)
            self._set_keys({"W"})
            return
        if (box is not None and config.AUTO_STEP_MAX_HEIGHT < rise <= self.REACH_UP_MAX
                and not (contact is not None and contact.wall_normal is not None)
                and self._reach_up_onto(box)):
            return

        if not elevated and self._brake_for_turn(speed):
            # a sharp bend just ahead (into a narrow alley): ease off so the turn makes it - at a sprint
            # the body would carry on past the opening into the wall beside it
            self._face(d, pitch=0.0)
            self._set_keys(set())
            return

        if self.blocked_for > 0.15 and not self._should_climb(contact):
            # a wall that is neither what we are after nor low enough to vault: go round it
            self._start_detour(contact.wall_normal, d)
            self.route_goal = None
            self._set_keys(set())
            return
        if self.blocked_for > 0.15:
            # a wall in the way: face it and go up (grab, or hop to reach higher / vault)
            wall_n = contact.wall_normal
            self._face(-wall_n, pitch=25.0)
            cand = p.grab_candidate()
            if cand is not None and id(cand.rail) not in self.dead_holds:
                self._set_keys({"W", "E"})
                self.climbs += 1
                return
            if self.time - self.last_hop > 0.7:
                self.last_hop = self.time
                self._set_keys({"W", "E"})
                self._tap("SPACE")
                return
            if self.blocked_for > 1.6:
                # nothing to climb right here: go round along the wall and try again from there
                self._start_detour(wall_n, d)
                self._set_keys(set())
                return
            self._set_keys({"W", "E"})
            return

        run = self._wall_run_facing(d)
        if run is not None:
            # a tall wall we mean to climb, met at a run: jump a stride out, plant a foot and let the
            # momentum carry us up to the first-storey hold (reaching with E; W climbs on in flow)
            self.last_hop = self.time
            self.wall_runs += 1
            self._face(run, pitch=10.0)
            self._set_keys({"W", "E"})
            self._tap("SPACE")
            return

        self._face(d, pitch=0.0)
        keys = {"W"}
        # at the edge of a roof / ledge: leap across if there is something to land on, else drop off
        ahead = self._ahead_ground(d, 0.7)
        if ahead is None or ahead < feet - 0.6:
            wants_down = goal_surface < feet - 1.0
            landing = self._landing_ahead(d)
            if landing is not None and (not wants_down or landing < 3.0):
                if self.time - self.last_hop > 0.5:
                    self.last_hop = self.time
                    self.leaps += 1
                    # take off with the jump alone: a fresh E at an edge facing out would lower the
                    # player over it (climb down) - the reach is held once airborne (_airborne)
                    self._set_keys({"W"})
                    self._tap("SPACE")
                    return
            elif wants_down or goal_surface <= feet + 0.5:
                self.dropping = True                  # walk off: no reaching on the way down
        self._set_keys(keys)
