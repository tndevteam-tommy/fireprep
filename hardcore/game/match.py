"""Multiplayer game rules: emblem hunts and capture the flag, free for all or red vs blue.

Emblem hunt      A limited number of emblems is hidden around the level (on building
                 roofs, on top of other climbable things and down in the streets);
                 touch one to pick it up.  The round ends when the last one is taken
                 (or on time): most emblems wins - the player, or in a teams game the
                 team whose players picked up the most.
Capture the flag Teams: each team has a base with its flag.  Take the other team's flag
                 and carry it into your own base while your flag is at home to score a
                 capture.  Touch an enemy carrying your flag to send it home.
                 Free for all: one flag and one drop zone; carry the flag into the zone
                 to score (then both move somewhere new).  Touch the carrier to steal it.
                 Most captures when the time is up wins (or the first to the capture
                 limit).

Everything here is deterministic for a seed and needs no networking, so the server
runs it and the tests drive it directly.  Moves come in as segments (prev -> pos),
so a fast runner cannot skip through an emblem, a flag or a player between updates.
"""
import math

import numpy as np

import config
from parkour.body import BodyModel
from parkour.ledge_rail import RailType
from physics.collision import CollisionSystem
from game import teams as T

EMBLEM_HOVER = 1.0          # an emblem floats this high above the surface it sits on


class World:
    """What the server (and the bot) need of the level: obstacles, holds and the buildings."""

    def __init__(self, obstacles, ledge_system, buildings=(), spawn_points=None):
        from world.spatial import ObstacleIndex
        self.obstacle_list = list(obstacles)
        self.obstacles = obstacles if hasattr(obstacles, "query") else ObstacleIndex(self.obstacle_list)
        self.ledge_system = ledge_system
        self.buildings = list(buildings)          # [(BuildingSpec, parts)]
        self.spawn_points = [np.asarray(p, dtype=np.float32) for p in (spawn_points or config.MP_SPAWN_POINTS)]

    @property
    def nav(self):
        """The bot's navigation grid (built on first use: a second or two for the whole city)."""
        if getattr(self, "_nav", None) is None:
            from game.nav import NavGrid
            self._nav = NavGrid(self.obstacle_list)
        return self._nav

    @property
    def sight(self):
        """Line of sight over the level for the bots (built on first use)."""
        if getattr(self, "_sight", None) is None:
            from game.sight import HeightField
            self._sight = HeightField(self.obstacle_list)
        return self._sight

    def roof_links(self, max_gap=4.5, max_rise=1.0):
        """Buildings you can leap between roof to roof: {i: [(j, gap)]} over `buildings`, for gaps up to
        max_gap whose far roof is at most max_rise higher (a leap up a storey is no leap).  Cached."""
        key = (max_gap, max_rise)
        cache = getattr(self, "_roof_links", None)
        if cache is None:
            cache = self._roof_links = {}
        if key in cache:
            return cache[key]
        specs = [spec for spec, _ in self.buildings]
        links = {i: [] for i in range(len(specs))}
        for i, a in enumerate(specs):
            for j, b in enumerate(specs):
                if i == j or b.height > a.height + max_rise:
                    continue
                gx = max(b.x - (a.x + a.width), a.x - (b.x + b.width), 0.0)
                gz = max(b.z - (a.z + a.depth), a.z - (b.z + b.depth), 0.0)
                ox = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)      # overlap along x
                oz = min(a.z + a.depth, b.z + b.depth) - max(a.z, b.z)
                if gx > 0.0 and gz == 0.0 and oz >= 2.0 and gx <= max_gap:
                    links[i].append((j, gx))
                elif gz > 0.0 and gx == 0.0 and ox >= 2.0 and gz <= max_gap:
                    links[i].append((j, gz))
        cache[key] = links
        return links

    def surface_box(self, x, z, top_y):
        """Bounds of the obstacle whose top at height top_y is under (x, z) - the roof or the box
        an emblem sits on (None for the open ground)."""
        if top_y < 0.5:
            return None
        lo = np.array([x - 0.05, top_y - 0.2, z - 0.05], dtype=np.float32)
        hi = np.array([x + 0.05, top_y + 0.05, z + 0.05], dtype=np.float32)
        best = None
        for o in self.obstacles.query(lo, hi):
            b = o.bounds
            if (abs(float(b.max[1]) - top_y) < 1e-2 and b.min[0] <= x <= b.max[0] and b.min[2] <= z <= b.max[2]):
                if best is None or (b.size[0] * b.size[2]) > (best.size[0] * best.size[2]):
                    best = b
        return best

    def ground_below(self, x, z, from_y=60.0):
        """Highest surface a body standing at (x, z) could stand on at or below height `from_y`
        (None if nothing)."""
        probe = np.array([x, from_y + config.PLAYER_HALF_HEIGHT, z], dtype=np.float32)
        return CollisionSystem.ground_height_below(probe, BodyModel.SIZE, self.obstacles, max_drop=from_y + 5.0)

    def standing_spot_free(self, x, y_surface, z):
        center = np.array([x, y_surface + config.PLAYER_HALF_HEIGHT + 0.02, z], dtype=np.float32)
        return BodyModel.pose_is_free(center, self.obstacles, shrink=0.02)

    def extent(self):
        """(lo, hi) corners of the playable ground."""
        big = [o.bounds for o in self.obstacle_list if o.bounds.size[0] > 100]
        if big:
            b = max(big, key=lambda bb: bb.size[0] * bb.size[2])
            return b.min + 8.0, b.max - 8.0
        pts = np.array([p for p in self.spawn_points])
        return pts.min(axis=0) - 30.0, pts.max(axis=0) + 30.0

    def open_area(self, x, z, radius, y_surface=None):
        """A flat, free patch of ground of this radius around (x, z) (a base, a drop zone)."""
        top = self.ground_below(x, z, 3.0) if y_surface is None else y_surface
        if top is None or not self.standing_spot_free(x, top, z):
            return False
        for k in range(8):
            a = k * math.pi / 4.0
            px, pz = x + math.cos(a) * radius, z + math.sin(a) * radius
            t = self.ground_below(px, pz, top + 0.3)
            if t is None or abs(t - top) > 0.2 or not self.standing_spot_free(px, t, pz):
                return False
        return True

    def spots_around(self, center, count, r_min=2.0, r_max=9.0, seed=0):
        """Free standing spots on the ground near `center` (team spawns next to a base)."""
        rng = np.random.RandomState(seed)
        cx, cz = float(center[0]), float(center[2])
        out = []
        for _ in range(count * 20):
            a = rng.rand() * 2.0 * math.pi
            r = r_min + rng.rand() * (r_max - r_min)
            x, z = cx + math.cos(a) * r, cz + math.sin(a) * r
            top = self.ground_below(x, z, float(center[1]) + 1.0)
            if top is None or abs(top - float(center[1])) > 0.5 or not self.standing_spot_free(x, top, z):
                continue
            if all(math.hypot(x - p[0], z - p[2]) > 1.2 for p in out):
                out.append([x, top + config.PLAYER_HALF_HEIGHT + 0.02, z])
            if len(out) >= count:
                break
        return out


class Emblem:
    __slots__ = ("id", "pos", "kind", "taken_by")

    def __init__(self, eid, pos, kind):
        self.id = int(eid)
        self.pos = np.asarray(pos, dtype=np.float32)
        self.kind = kind                 # "roof" / "top" / "street"
        self.taken_by = None

    def to_dict(self):
        return {"id": self.id, "pos": [round(float(c), 3) for c in self.pos], "kind": self.kind,
                "taken_by": self.taken_by}

    @classmethod
    def from_dict(cls, d):
        e = cls(d["id"], d["pos"], d.get("kind", "street"))
        e.taken_by = d.get("taken_by")
        return e


class EmblemPlacer:
    """Reachable, spread-out spots for emblems: building roofs, tops of climbable things, streets."""

    MIN_SPACING = 14.0

    def __init__(self, world):
        self.world = world

    def _roof_spots(self, rng):
        out = []
        for spec, parts in self.world.buildings:
            for _ in range(6):
                x = spec.x + 1.2 + rng.rand() * max(0.1, spec.width - 2.4)
                z = spec.z + 1.2 + rng.rand() * max(0.1, spec.depth - 2.4)
                top = self.world.ground_below(x, z, spec.height + 4.0)
                if top is not None and abs(top - spec.height) < 1e-3 and self.world.standing_spot_free(x, top, z):
                    out.append(((x, top, z), "roof"))
                    break
        return out

    def _top_spots(self, rng):
        """Tops of other climbable things (containers, decks, park blocks) that have a mantle edge."""
        tops = {}
        for r in self.world.ledge_system.rails:
            o = r.owner
            if (o is None or r.rail_type != RailType.MANTLE_TOP or not getattr(o, "climb_target", True)
                    or o.name.startswith("bld-") or o.name in ("city-wall", "sidewalk")):
                continue
            b = o.bounds
            size = b.max - b.min
            if size[0] >= 1.4 and size[2] >= 1.4 and 1.2 <= float(b.max[1]) <= 12.5:
                tops[id(o)] = o
        out = []
        for o in tops.values():
            b = o.bounds
            x = float(b.min[0] + b.max[0]) * 0.5
            z = float(b.min[2] + b.max[2]) * 0.5
            top = self.world.ground_below(x, z, float(b.max[1]) + 4.0)
            if top is not None and abs(top - float(b.max[1])) < 1e-3 and self.world.standing_spot_free(x, top, z):
                out.append(((x, top, z), "top"))
        return out

    def _street_spots(self, rng, n):
        """Open ground: walk a random scatter of points and keep those where a player fits."""
        out = []
        xs = [o.bounds for o in self.world.obstacle_list if o.bounds.size[0] > 200]       # the ground slab
        if xs:
            lo, hi = xs[0].min + 8.0, xs[0].max - 8.0
        else:
            pts = np.array([p for p in self.world.spawn_points])
            lo, hi = pts.min(axis=0) - 30.0, pts.max(axis=0) + 30.0
        for _ in range(n * 12):
            x = float(lo[0] + rng.rand() * (hi[0] - lo[0]))
            z = float(lo[2] + rng.rand() * (hi[2] - lo[2]))
            top = self.world.ground_below(x, z, 3.0)
            if top is not None and top <= 0.3 and self.world.standing_spot_free(x, top, z):
                out.append(((x, top, z), "street"))
            if len(out) >= n:
                break
        return out

    def place(self, count, seed):
        rng = np.random.RandomState(seed)
        pools = {"roof": self._roof_spots(rng), "top": self._top_spots(rng), "street": self._street_spots(rng, count * 3)}
        for pool in pools.values():
            rng.shuffle(pool)
        # mostly roofs (you climb for them), some tops, some hidden down in the streets
        want = {"roof": int(round(count * 0.5)), "top": int(round(count * 0.25))}
        want["street"] = count - want["roof"] - want["top"]
        chosen = []

        def far_enough(p):
            return all(math.dist(p, q) >= self.MIN_SPACING for q, _ in chosen)

        for kind in ("roof", "top", "street"):
            for p, k in pools[kind]:
                if sum(1 for _, kk in chosen if kk == kind) >= want[kind]:
                    break
                if far_enough(p):
                    chosen.append((p, k))
        # fill up from anything left if a pool ran short
        for kind in ("roof", "street", "top"):
            for p, k in pools[kind]:
                if len(chosen) >= count:
                    break
                if (p, k) not in chosen and far_enough(p):
                    chosen.append((p, k))
        return [Emblem(i, (p[0], p[1] + EMBLEM_HOVER, p[2]), k) for i, (p, k) in enumerate(chosen[:count])]


def _segment_point_dist(a, b, p):
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + ab * t - p))


TELEPORT_DIST = 6.0          # a move longer than this in one update is a respawn / teleport, not a run
FLAG_HEIGHT = 1.0            # a flag's "touch point" sits this high above its base (chest height)


class GameRules:
    """What the host picked: the game, free for all or teams, the time limit and the mode's options.
    A time limit of 0 means no time limit (the round ends on the mode's own goal)."""

    EMBLEMS = "emblems"
    CTF = "ctf"
    ZOMBIE = "zombie"
    MODES = (EMBLEMS, CTF, ZOMBIE)
    LABELS = {EMBLEMS: "Emblem hunt", CTF: "Capture the flag", ZOMBIE: "Zombie tag"}

    def __init__(self, mode=EMBLEMS, teams=False, time_limit=config.MP_MATCH_TIME_LIMIT,
                 emblems=config.MP_DEFAULT_EMBLEMS, capture_limit=config.MP_DEFAULT_CAPTURES,
                 goal=config.ZOMBIE_DEFAULT_GOAL):
        self.mode = mode if mode in self.MODES else self.EMBLEMS
        self.teams = bool(teams) and self.mode != self.ZOMBIE          # zombie tag is every player for themself
        t = float(time_limit)
        self.time_limit = 0.0 if not (t > 0.0) else float(min(3600.0, max(1.0, t)))
        lo, hi = config.MP_EMBLEM_RANGE
        self.emblems = int(min(hi, max(lo, int(emblems))))
        self.capture_limit = int(min(config.MP_CAPTURE_LIMIT_MAX, max(0, int(capture_limit))))
        glo, ghi = config.ZOMBIE_GOAL_RANGE
        self.goal = int(min(ghi, max(glo, int(goal))))

    @property
    def unlimited(self):
        return self.time_limit <= 0.0

    def to_dict(self):
        return {"mode": self.mode, "teams": self.teams, "time_limit": self.time_limit,
                "emblems": self.emblems, "capture_limit": self.capture_limit, "goal": self.goal}

    @classmethod
    def from_dict(cls, d):
        d = d if isinstance(d, dict) else {}
        try:
            return cls(d.get("mode", cls.EMBLEMS), bool(d.get("teams", False)),
                       float(d.get("time_limit", config.MP_MATCH_TIME_LIMIT)),
                       int(d.get("emblems", config.MP_DEFAULT_EMBLEMS)),
                       int(d.get("capture_limit", config.MP_DEFAULT_CAPTURES)),
                       int(d.get("goal", config.ZOMBIE_DEFAULT_GOAL)))
        except (TypeError, ValueError):
            return cls()

    @property
    def label(self):
        return self.LABELS[self.mode]

    @property
    def point_word(self):
        """What a point is called: emblem(s) / capture(s) / point(s)."""
        return {self.EMBLEMS: "emblem", self.CTF: "capture"}.get(self.mode, "point")

    def never_ends(self):
        """No time limit and nothing else that ends a round (capture the flag without a capture limit)."""
        return self.unlimited and self.mode == self.CTF and not self.capture_limit

    def describe(self):
        duration = "no time limit" if self.unlimited else f"{int(round(self.time_limit / 60.0))} min"
        extra = ""
        if self.mode == self.CTF and self.capture_limit:
            extra = f", first to {self.capture_limit}"
        elif self.mode == self.ZOMBIE:
            extra = f", first to {self.goal} points"
        sides = "red vs blue" if self.teams else ("every player for themself" if self.mode == self.ZOMBIE
                                                   else "free for all")
        return f"{self.label} - {sides} - {duration}{extra}"


class Target:
    """Something a bot can go for: an emblem, a flag, a base, a player to tag."""

    __slots__ = ("id", "pos", "radius", "kind")

    def __init__(self, tid, pos, radius, kind="item"):
        self.id = tid
        self.pos = np.asarray(pos, dtype=np.float32)
        self.radius = float(radius)
        self.kind = kind                  # "item" (touch it), "zone" (get inside), "player" (tag them)


class BaseMatch:
    """Players (and their teams), scores, the clock and the round lifecycle.
    States: 'playing' -> 'over' -> (next round) 'playing'."""

    PLAYING = "playing"
    OVER = "over"

    def __init__(self, world, rules, seed=1):
        self.world = world
        self.rules = rules
        self.seed = int(seed)
        self.time_limit = float(rules.time_limit)
        self.round = 0
        self.state = self.OVER
        self.players = {}             # pid -> team (None in a free for all)
        self.scores = {}              # pid -> emblems / captures this round
        self.team_scores = {T.RED: 0, T.BLUE: 0}
        self.positions = {}           # pid -> last reported body centre
        self.elapsed = 0.0
        self.winners = []
        self.winning_team = None
        self.draw = False
        self._spawn_turn = {}

    # ------------------------------------------------------------- players
    @property
    def teams(self):
        return self.rules.teams

    @property
    def mode(self):
        return self.rules.mode

    def round_seed(self):
        return (self.seed * 1000 + self.round) % (2 ** 32)

    def start(self, players=()):
        """A new round: new spots for everything (the seed advances), everyone back to zero.
        `players`: {pid: team} or just pids."""
        self.round += 1
        if not isinstance(players, dict):
            players = {pid: self.players.get(pid) for pid in players}
        self.players = {pid: (team if self.teams and T.valid(team) else (T.RED if self.teams else None))
                        for pid, team in players.items()}
        self.scores = {pid: 0 for pid in self.players}
        self.team_scores = {T.RED: 0, T.BLUE: 0}
        self.positions = {}
        self.elapsed = 0.0
        self.winners = []
        self.winning_team = None
        self.draw = False
        self._spawn_turn = {}
        self.state = self.PLAYING
        self._setup_round(np.random.RandomState(self.round_seed()))
        return self

    def _setup_round(self, rng):
        pass

    def add_player(self, pid, team=None):
        self.players[pid] = team if self.teams and T.valid(team) else (T.RED if self.teams else None)
        self.scores.setdefault(pid, 0)

    def remove_player(self, pid):
        """A player left.  Returns the events that causes (a flag going home)."""
        events = self._player_gone(pid, "left")
        self.players.pop(pid, None)
        self.scores.pop(pid, None)
        self.positions.pop(pid, None)
        return events

    def set_team(self, pid, team):
        """Switch sides (teams games).  Returns the events that causes."""
        if not self.teams or not T.valid(team) or self.players.get(pid) == team:
            return []
        events = self._player_gone(pid, "switched")
        self.players[pid] = team
        return events

    def team_of(self, pid):
        return self.players.get(pid)

    def team_members(self, team):
        return [pid for pid, t in self.players.items() if t == team]

    def _player_gone(self, pid, reason):
        return []

    def drain_events(self):
        """Events the match raised on its own (not from a move) since the last call."""
        return []

    # ---------------------------------------------------------------- spawns
    def _team_spawns(self, team):
        pts = self.world.spawn_points
        order = sorted(range(len(pts)), key=lambda i: (float(pts[i][0]), float(pts[i][2])))
        half = max(1, len(order) // 2)
        side = order[:half] if team == T.RED else order[half:] or order
        return [pts[i] for i in side]

    def spawn_point(self, pid):
        """Where this player (re)appears: the start line in a free for all, the team's side (or base)."""
        team = self.players.get(pid)
        if team is None:
            pts = self.world.spawn_points
            return [float(c) for c in pts[(int(pid) - 1) % len(pts)]]
        pts = self._team_spawns(team)
        k = self._spawn_turn.get(team, 0)
        self._spawn_turn[team] = k + 1
        return [float(c) for c in pts[k % len(pts)]]

    # ----------------------------------------------------------------- clock
    def tick(self, dt):
        """Advance the clock; returns True when the match just ended (on time, or the mode's goal)."""
        if self.state != self.PLAYING:
            return False
        self.elapsed += dt
        if self.time_limit > 0.0 and self.elapsed >= self.time_limit:
            self._finish()
            return True
        return False

    @property
    def time_left(self):
        """Seconds left, or None when there is no time limit."""
        if self.time_limit <= 0.0:
            return None
        return max(0.0, self.time_limit - self.elapsed)

    def _finish(self):
        self.state = self.OVER
        self.winners, self.winning_team, self.draw = [], None, False
        if self.teams:
            r, b = self.team_scores[T.RED], self.team_scores[T.BLUE]
            if r == b:
                self.draw = r > 0
            else:
                self.winning_team = T.RED if r > b else T.BLUE
                self.winners = sorted(self.team_members(self.winning_team))
        elif self.scores:
            best = max(self.scores.values())
            if best > 0:
                self.winners = sorted(pid for pid, s in self.scores.items() if s == best)
                self.draw = len(self.winners) > 1

    def results(self):
        return {"winners": list(self.winners), "team": self.winning_team, "draw": self.draw,
                "scores": {str(k): v for k, v in self.scores.items()}, "team_scores": dict(self.team_scores),
                "teams": {str(k): v for k, v in self.players.items()}, "mode": self.mode, "teams_game": self.teams}

    # -------------------------------------------------------------- movement
    def _segment(self, pid, prev_pos, pos):
        a = np.asarray(prev_pos if prev_pos is not None else pos, dtype=np.float32)
        b = np.asarray(pos, dtype=np.float32)
        teleported = float(np.linalg.norm(b - a)) > TELEPORT_DIST
        if teleported:
            a = b                                  # a teleport / respawn is not a sweep across the map
        return a, b, teleported

    def apply_move(self, pid, prev_pos, pos):
        """A player moved from prev_pos to pos.  Returns the events (dicts) it caused."""
        return []

    def _score(self, pid, n=1):
        self.scores[pid] = self.scores.get(pid, 0) + n
        team = self.players.get(pid)
        if self.teams and team in self.team_scores:
            self.team_scores[team] += n

    # ----------------------------------------------------------------- state
    def to_dict(self):
        """Everything a joining client needs."""
        return {"round": self.round, "state": self.state, "mode": self.mode, "teams_game": self.teams,
                "rules": self.rules.to_dict(), "scores": {str(k): v for k, v in self.scores.items()},
                "team_scores": dict(self.team_scores), "teams": {str(k): v for k, v in self.players.items()},
                "winners": list(self.winners), "time_left": self.time_left}

    def snapshot(self):
        """The part of the state that changes all the time (sent ~20x a second)."""
        return {"team_scores": dict(self.team_scores)}

    def bot_targets(self, pid, role="attack", claimed=()):
        return []


class EmblemMatch(BaseMatch):
    """Emblem hunt: most emblems picked up wins (per player, or summed per team)."""

    def __init__(self, world, emblem_count=config.MP_DEFAULT_EMBLEMS, seed=1,
                 time_limit=config.MP_MATCH_TIME_LIMIT, pickup_radius=config.MP_PICKUP_RADIUS, rules=None):
        rules = rules or GameRules(GameRules.EMBLEMS, False, time_limit, emblem_count)
        super().__init__(world, rules, seed)
        self.pickup_radius = float(pickup_radius)
        self.emblems = []
        self.placer = EmblemPlacer(world)

    @property
    def emblem_count(self):
        return self.rules.emblems

    def start(self, players=()):
        super().start(players)
        return self.emblems

    def _setup_round(self, rng):
        self.emblems = self.placer.place(self.emblem_count, self.round_seed())

    @property
    def remaining(self):
        return sum(1 for e in self.emblems if e.taken_by is None)

    def move_player(self, pid, prev_pos, pos):
        """Pick up every emblem the body passed through; returns them (the match may be over after)."""
        if self.state != self.PLAYING:
            return []
        a, b, _ = self._segment(pid, prev_pos, pos)
        self.positions[pid] = b
        got = []
        live = [e for e in self.emblems if e.taken_by is None]
        if not live:
            return got
        # (all the emblems at once - every player's every move checks them: it was a third of the server)
        pts = np.array([e.pos for e in live], dtype=np.float64)
        a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        ab = b64 - a64
        den = float(np.dot(ab, ab))
        if den < 1e-12:
            d = np.linalg.norm(pts - a64, axis=1)
        else:
            t = np.clip((pts - a64) @ ab / den, 0.0, 1.0)
            d = np.linalg.norm(a64 + t[:, None] * ab - pts, axis=1)
        for e, dist in zip(live, d):
            if dist <= self.pickup_radius:
                e.taken_by = pid
                self._score(pid)
                got.append(e)
        if got and self.remaining == 0:
            self._finish()
        return got

    def apply_move(self, pid, prev_pos, pos):
        team = self.players.get(pid)
        return [{"t": "pickup", "emblem": e.id, "player": pid, "team": team,
                 "score": self.scores.get(pid, 0), "remaining": self.remaining,
                 "team_scores": dict(self.team_scores)} for e in self.move_player(pid, prev_pos, pos)]

    def to_dict(self):
        d = super().to_dict()
        d["emblems"] = [e.to_dict() for e in self.emblems]
        return d

    def snapshot(self):
        d = super().snapshot()
        d["remaining"] = self.remaining
        return d

    def bot_targets(self, pid, role="attack", claimed=()):
        out = [Target(e.id, e.pos, self.pickup_radius * 0.9) for e in self.emblems if e.taken_by is None]
        free = [t for t in out if t.id not in claimed]
        return free or out


Match = EmblemMatch                    # (the original name: an emblem hunt)


class Flag:
    """A flag: at home (on its stand) or carried by a player."""

    __slots__ = ("key", "team", "home", "carrier", "since")

    def __init__(self, key, team, home):
        self.key = key                       # "red" / "blue" / "neutral"
        self.team = team                     # owner (None: the free-for-all flag)
        self.home = np.asarray(home, dtype=np.float32)     # ground point of its stand
        self.carrier = None
        self.since = 0.0                     # match time it was last picked up / stolen

    @property
    def at_home(self):
        return self.carrier is None

    def touch_point(self):
        return self.home + np.array([0.0, FLAG_HEIGHT, 0.0], dtype=np.float32)

    def to_dict(self, positions):
        pos = self.home
        if self.carrier is not None and self.carrier in positions:
            pos = positions[self.carrier]
        return {"key": self.key, "team": self.team, "home": [round(float(c), 3) for c in self.home],
                "pos": [round(float(c), 3) for c in pos], "carrier": self.carrier}


class Zone:
    """A capture zone: a team's base, or the free-for-all drop zone."""

    __slots__ = ("key", "team", "center", "radius")

    def __init__(self, key, team, center, radius=config.MP_BASE_RADIUS):
        self.key = key
        self.team = team
        self.center = np.asarray(center, dtype=np.float32)     # ground point
        self.radius = float(radius)

    def contains(self, body_center):
        c = np.asarray(body_center, dtype=np.float32)
        feet = float(c[1]) - config.PLAYER_HALF_HEIGHT
        return (math.hypot(float(c[0] - self.center[0]), float(c[2] - self.center[2])) <= self.radius
                and -0.6 <= feet - float(self.center[1]) <= 2.5)

    def to_dict(self):
        return {"key": self.key, "team": self.team, "pos": [round(float(c), 3) for c in self.center], "r": self.radius}


class CtfPlacer:
    """Bases far apart on open ground (teams), or a flag spot and a drop zone (free for all)."""

    def __init__(self, world):
        self.world = world
        self.emblems = EmblemPlacer(world)

    def open_spots(self, rng, count, radius=config.MP_BASE_RADIUS, tries=600):
        lo, hi = self.world.extent()
        out = []
        for _ in range(tries):
            x = float(lo[0] + rng.rand() * (hi[0] - lo[0]))
            z = float(lo[2] + rng.rand() * (hi[2] - lo[2]))
            top = self.world.ground_below(x, z, 3.0)
            if top is None or top > 0.3 or not self.world.open_area(x, z, radius, top):
                continue
            out.append(np.array([x, top, z], dtype=np.float32))
            if len(out) >= count:
                break
        return out

    def bases(self, rng):
        spots = self.open_spots(rng, 36)
        lo, hi = self.world.extent()
        size = float(min(hi[0] - lo[0], hi[2] - lo[2]))
        want = float(np.clip(0.55 * size, *config.MP_CTF_BASE_DISTANCE))
        best, best_key = None, None
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                d = float(np.linalg.norm(spots[i] - spots[j]))
                key = abs(d - want) + rng.rand() * 4.0
                if best_key is None or key < best_key:
                    best, best_key = (spots[i], spots[j]), key
        if best is None:
            pts = sorted(self.world.spawn_points, key=lambda p: float(p[0]))
            feet = lambda p: np.array([p[0], float(p[1]) - config.PLAYER_HALF_HEIGHT, p[2]], dtype=np.float32)
            best = (feet(pts[0]), feet(pts[-1]))
        a, b = best
        if rng.rand() < 0.5:
            a, b = b, a
        return {T.RED: a, T.BLUE: b}

    def flag_spot(self, rng):
        """Anywhere an emblem could be: a roof, the top of something, the street."""
        pools = [("roof", self.emblems._roof_spots(rng)), ("top", self.emblems._top_spots(rng)),
                 ("street", self.emblems._street_spots(rng, 12))]
        weights = {"roof": 0.45, "top": 0.25, "street": 0.30}
        pools = [(k, p) for k, p in pools if p]
        if not pools:
            p = self.world.spawn_points[0]
            return np.array([p[0], float(p[1]) - config.PLAYER_HALF_HEIGHT, p[2]], dtype=np.float32)
        w = np.array([weights[k] for k, _ in pools])
        kind, pool = pools[int(rng.choice(len(pools), p=w / w.sum()))]
        (x, y, z), _ = pool[int(rng.randint(len(pool)))]
        return np.array([x, y, z], dtype=np.float32)

    def zone_spot(self, rng, away_from, near=30.0, far=80.0):
        spots = self.open_spots(rng, 24)
        if not spots:
            p = self.world.spawn_points[-1]
            return np.array([p[0], float(p[1]) - config.PLAYER_HALF_HEIGHT, p[2]], dtype=np.float32)
        a = np.asarray(away_from, dtype=np.float32)

        def key(s):
            d = math.hypot(float(s[0] - a[0]), float(s[2] - a[2]))
            return 0.0 if near <= d <= far else min(abs(d - near), abs(d - far))
        spots.sort(key=key)
        good = [s for s in spots if key(s) == 0.0]
        return good[int(rng.randint(len(good)))] if good else spots[0]


class CtfMatch(BaseMatch):
    """Capture the flag (see the module docstring for the rules)."""

    def __init__(self, world, rules, seed=1):
        super().__init__(world, rules, seed)
        self.placer = CtfPlacer(world)
        self.flags = {}               # key -> Flag
        self.zones = {}               # key -> Zone
        self.team_spawns = {}         # team -> [spawn points by its base]
        self.rng = np.random.RandomState(self.seed)

    # ------------------------------------------------------------------ setup
    def _setup_round(self, rng):
        self.rng = rng
        self.flags, self.zones, self.team_spawns = {}, {}, {}
        if self.teams:
            bases = self.placer.bases(rng)
            for team, p in bases.items():
                self.flags[team] = Flag(team, team, p)
                self.zones[team] = Zone(team, team, p)
                spawn_c = p + np.array([0.0, 0.0, 0.0], dtype=np.float32)
                self.team_spawns[team] = self.world.spots_around(spawn_c, 8, 3.5, 9.0, seed=int(rng.randint(1 << 30)))
        else:
            self._place_neutral()

    def _place_neutral(self):
        spot = self.placer.flag_spot(self.rng)
        self.flags = {"neutral": Flag("neutral", None, spot)}
        self.zones = {"drop": Zone("drop", None, self.placer.zone_spot(self.rng, spot))}

    def _team_spawns(self, team):
        pts = self.team_spawns.get(team)
        if pts:
            return pts
        return super()._team_spawns(team)

    @property
    def capture_limit(self):
        return self.rules.capture_limit

    # --------------------------------------------------------------- players
    def carried_by(self, pid):
        for f in self.flags.values():
            if f.carrier == pid:
                return f
        return None

    def _return(self, flag, reason, by=None):
        carrier = flag.carrier
        flag.carrier = None
        ev = {"t": "flag", "action": "returned", "flag": flag.key, "player": carrier, "reason": reason}
        if by is not None:
            ev["by"] = by
        return ev

    def _player_gone(self, pid, reason):
        f = self.carried_by(pid)
        return [self._return(f, reason)] if f is not None else []

    # ------------------------------------------------------------ movement
    def apply_move(self, pid, prev_pos, pos):
        if self.state != self.PLAYING or pid not in self.players:
            return []
        a, b, teleported = self._segment(pid, prev_pos, pos)
        events = []
        if teleported:
            f = self.carried_by(pid)
            if f is not None:
                events.append(self._return(f, "respawn"))
        self.positions[pid] = b
        if self.teams:
            events += self._teams_move(pid, a, b)
        else:
            events += self._ffa_move(pid, a, b)
        return events

    def _touching(self, a, b, other_pid):
        p = self.positions.get(other_pid)
        return p is not None and _segment_point_dist(a, b, p) <= config.MP_TAG_RADIUS

    def _teams_move(self, pid, a, b):
        events = []
        team = self.players[pid]
        enemy = T.other(team)
        own, theirs = self.flags[team], self.flags[enemy]
        # tags: touching the enemy who carries our flag sends it home (either of you moving)
        if own.carrier is not None and own.carrier != pid and self._touching(a, b, own.carrier):
            events.append(self._return(own, "tagged", by=pid))
        if theirs.carrier == pid:
            for other, t in self.players.items():
                if t == enemy and self._touching(a, b, other):
                    events.append(self._return(theirs, "tagged", by=other))
                    break
        # take the enemy flag from its stand
        if theirs.at_home and self.carried_by(pid) is None and \
                _segment_point_dist(a, b, theirs.touch_point()) <= config.MP_FLAG_RADIUS:
            theirs.carrier = pid
            theirs.since = self.elapsed
            events.append({"t": "flag", "action": "taken", "flag": theirs.key, "player": pid, "team": team})
        # bring it home - your own flag has to be there too
        if theirs.carrier == pid and own.at_home and self.zones[team].contains(b):
            theirs.carrier = None
            self._score(pid)
            events.append({"t": "flag", "action": "captured", "flag": theirs.key, "player": pid, "team": team,
                           "score": self.scores.get(pid, 0), "team_scores": dict(self.team_scores)})
            if self.capture_limit and self.team_scores[team] >= self.capture_limit:
                self._finish()
        return events

    def _ffa_move(self, pid, a, b):
        events = []
        flag = self.flags["neutral"]
        zone = self.zones["drop"]
        carrier = flag.carrier
        if carrier is not None and carrier != pid and self.elapsed - flag.since >= config.MP_STEAL_COOLDOWN \
                and self._touching(a, b, carrier):
            flag.carrier, flag.since = pid, self.elapsed          # stolen
            events.append({"t": "flag", "action": "stolen", "flag": flag.key, "player": pid, "from": carrier})
        elif carrier == pid and self.elapsed - flag.since >= config.MP_STEAL_COOLDOWN:
            for other in self.players:
                if other != pid and self._touching(a, b, other):
                    flag.carrier, flag.since = other, self.elapsed
                    events.append({"t": "flag", "action": "stolen", "flag": flag.key, "player": other, "from": pid})
                    break
        elif carrier is None and _segment_point_dist(a, b, flag.touch_point()) <= config.MP_FLAG_RADIUS:
            flag.carrier, flag.since = pid, self.elapsed
            events.append({"t": "flag", "action": "taken", "flag": flag.key, "player": pid})
        if flag.carrier == pid and zone.contains(b):
            self._score(pid)
            flag.carrier = None
            self._place_neutral()                                  # a new flag somewhere else, a new zone
            events.append({"t": "flag", "action": "captured", "flag": "neutral", "player": pid,
                           "score": self.scores.get(pid, 0)})
            if self.capture_limit and self.scores[pid] >= self.capture_limit:
                self._finish()
        return events

    # ------------------------------------------------------------------ state
    def objectives(self):
        return {"flags": [f.to_dict(self.positions) for f in self.flags.values()],
                "zones": [z.to_dict() for z in self.zones.values()]}

    def to_dict(self):
        d = super().to_dict()
        d.update(self.objectives())
        return d

    def snapshot(self):
        d = super().snapshot()
        d.update(self.objectives())
        return d

    # ------------------------------------------------------------------- bots
    def _body_at(self, ground):
        return np.asarray(ground, dtype=np.float32) + np.array([0.0, config.PLAYER_HALF_HEIGHT, 0.0], dtype=np.float32)

    GUARD_DISTANCE = 7.0

    def _guard_spot(self, team):
        """Where a defender waits: GUARD_DISTANCE out from its flag toward the enemy base - close enough
        to run a thief down, far enough that a thief can take the flag and has to get past it.  (A
        defender standing on the stand tagged every thief the instant it touched the flag: bot teams of
        three or more never captured at all.)"""
        own = self.flags[team]
        enemy = self.flags.get(T.other(team))
        home = np.asarray(own.home, dtype=np.float32)
        if enemy is None:
            return home
        d = np.asarray(enemy.home, dtype=np.float32) - home
        d[1] = 0.0
        n = float(np.linalg.norm(d))
        if n < 1e-3:
            return home
        for dist in (self.GUARD_DISTANCE, 5.0, 3.5):
            p = home + d / n * dist
            top = self.world.ground_below(float(p[0]), float(p[2]), float(home[1]) + 0.5)
            if top is not None and abs(top - float(home[1])) < 0.3 and self.world.standing_spot_free(float(p[0]), top, float(p[2])):
                return np.array([p[0], top, p[2]], dtype=np.float32)
        return home

    def bot_targets(self, pid, role="attack", claimed=()):
        if self.state != self.PLAYING or pid not in self.players:
            return []
        me = self.positions.get(pid)
        if not self.teams:
            flag, zone = self.flags["neutral"], self.zones["drop"]
            if flag.carrier == pid:
                return [Target("zone:drop", self._body_at(zone.center), zone.radius * 0.6, "zone")]
            if flag.carrier is not None and flag.carrier in self.positions:
                return [Target(f"player:{flag.carrier}", self.positions[flag.carrier], config.MP_TAG_RADIUS * 0.7, "player")]
            return [Target("flag:neutral", flag.touch_point(), config.MP_FLAG_RADIUS * 0.8)]
        team = self.players[pid]
        enemy = T.other(team)
        own, theirs = self.flags[team], self.flags[enemy]
        home = Target(f"zone:{team}", self._body_at(self.zones[team].center), self.zones[team].radius * 0.6, "zone")
        if theirs.carrier == pid:
            return [home]
        chase = None
        if own.carrier is not None and own.carrier in self.positions:
            chase = Target(f"player:{own.carrier}", self.positions[own.carrier], config.MP_TAG_RADIUS * 0.7, "player")
        if role == "defend":
            return [chase] if chase is not None else [Target(f"guard:{team}", self._body_at(self._guard_spot(team)),
                                                             2.5, "zone")]
        if chase is not None and me is not None:
            # chase the thief if they are nearer than the flag we are after
            d_chase = float(np.linalg.norm(chase.pos - me))
            d_flag = float(np.linalg.norm(theirs.touch_point() - me))
            if d_chase < d_flag or theirs.carrier is not None:
                return [chase]
        if theirs.carrier is not None:
            return [home]                       # a teammate has it: meet them at home
        return [Target(f"flag:{enemy}", theirs.touch_point(), config.MP_FLAG_RADIUS * 0.8)]


class ZombieMatch(BaseMatch):
    """Zombie tag: one player is alive, everyone else is a zombie.

    The living player scores ZOMBIE_POINTS_PER_SECOND for every second they stay alive.  A zombie that
    touches them (MP_TAG_RADIUS, either of them moving) catches them: the living player turns into a
    zombie and the zombie that caught them comes alive - and starts scoring.  Nobody can be caught for
    ZOMBIE_START_GRACE at the start of a round or ZOMBIE_CATCH_GRACE after a catch, and there are no
    tag-backs: the player just caught cannot catch their catcher for ZOMBIE_TAGBACK_GRACE.  The first to the
    host's goal wins; on time, the most points.  Zombies are slower (the clients and the bots apply
    ZOMBIE_RUN_SCALE / ZOMBIE_CLIMB_SCALE to their own movement).
    """

    def __init__(self, world, rules, seed=1):
        super().__init__(world, rules, seed)
        self.living = None            # pid of the living player
        self.points = {}              # pid -> points (fractional; scores holds the whole points)
        self.grace = 0.0              # seconds during which nobody can be caught
        self.no_tagback = None        # (pid, until match time): the player just caught cannot catch back yet
        self.catches = {}             # pid -> how many times they caught the living player
        self.rng = np.random.RandomState(self.seed)
        self.living_spawn = None
        self.flee_spots = []
        self._events = []
        self._flee = {}               # pid -> (target id, position, age) for a living bot

    # ------------------------------------------------------------------ setup
    def _setup_round(self, rng):
        self.rng = rng
        self.points = {pid: 0.0 for pid in self.players}
        self.catches = {}
        self.grace = config.ZOMBIE_START_GRACE
        self.no_tagback = None
        self._flee = {}
        self.living = None
        placer = EmblemPlacer(self.world)
        streets = placer._street_spots(rng, 60)
        roofs = placer._roof_spots(rng)
        tops = placer._top_spots(rng)
        self.flee_spots = [np.array([x, y + config.PLAYER_HALF_HEIGHT, z], dtype=np.float32)
                           for (x, y, z), _ in streets + roofs + tops]
        # the living player starts away from the zombies (who start on the start line)
        start = np.mean(np.array(self.world.spawn_points, dtype=np.float32), axis=0)
        lo, hi = config.ZOMBIE_LIVING_SPAWN_DISTANCE
        ground = [p for p in self.flee_spots[:len(streets)]
                  if lo <= math.hypot(float(p[0] - start[0]), float(p[2] - start[2])) <= hi]
        if ground:
            self.living_spawn = ground[int(rng.randint(len(ground)))]
        elif self.flee_spots:
            self.living_spawn = max(self.flee_spots[:max(1, len(streets))],
                                    key=lambda p: math.hypot(float(p[0] - start[0]), float(p[2] - start[2])))
        else:
            self.living_spawn = None
        self._choose_living("start")

    def _choose_living(self, reason, exclude=None):
        candidates = sorted(pid for pid in self.players if pid != exclude)
        if not candidates:
            self.living = None
            return
        self.living = int(candidates[int(self.rng.randint(len(candidates)))])
        self._events.append({"t": "zombie", "action": "living", "player": self.living, "reason": reason})

    def drain_events(self):
        out, self._events = self._events, []
        return out

    # --------------------------------------------------------------- players
    def add_player(self, pid, team=None):
        super().add_player(pid, None)
        self.points.setdefault(pid, 0.0)
        if self.living is None and self.state == self.PLAYING:
            self._choose_living("joined")

    def remove_player(self, pid):
        events = super().remove_player(pid)
        self.points.pop(pid, None)
        if pid == self.living and self.state == self.PLAYING:
            self.living = None
            self._choose_living("left")
            self.grace = max(self.grace, config.ZOMBIE_CATCH_GRACE)
        return events + self.drain_events()

    def role(self, pid):
        return "living" if pid == self.living else "zombie"

    def spawn_point(self, pid):
        if pid == self.living and self.living_spawn is not None:
            return [float(c) for c in self.living_spawn]
        return super().spawn_point(pid)

    # ------------------------------------------------------------------ clock
    def tick(self, dt):
        if self.state != self.PLAYING:
            return False
        self.grace = max(0.0, self.grace - dt)
        if self.living is None and self.players:
            self._choose_living("joined")
        if self.living is not None:
            p = self.points.get(self.living, 0.0) + dt * config.ZOMBIE_POINTS_PER_SECOND
            self.points[self.living] = p
            self.scores[self.living] = int(p)
            if self.scores[self.living] >= self.rules.goal:
                self.elapsed += dt
                self._finish()
                return True
        return super().tick(dt)

    # --------------------------------------------------------------- catches
    def apply_move(self, pid, prev_pos, pos):
        if self.state != self.PLAYING or pid not in self.players:
            return []
        a, b, _ = self._segment(pid, prev_pos, pos)
        self.positions[pid] = b
        if self.living is None or self.grace > 0.0:
            return []
        catcher = None
        blocked = self.no_tagback[0] if self.no_tagback and self.elapsed < self.no_tagback[1] else None
        if pid != self.living:
            lp = self.positions.get(self.living)
            if pid != blocked and lp is not None and _segment_point_dist(a, b, lp) <= config.MP_TAG_RADIUS:
                catcher = pid
        else:
            best = None
            for other, op in self.positions.items():
                if other == pid or other not in self.players or other == blocked:
                    continue
                d = _segment_point_dist(a, b, op)
                if d <= config.MP_TAG_RADIUS and (best is None or d < best):
                    best, catcher = d, other
        if catcher is None:
            return []
        caught = self.living
        self.living = catcher
        self.catches[catcher] = self.catches.get(catcher, 0) + 1
        self.grace = config.ZOMBIE_CATCH_GRACE
        self.no_tagback = (caught, self.elapsed + config.ZOMBIE_TAGBACK_GRACE)
        self._flee.pop(caught, None)
        return [{"t": "zombie", "action": "caught", "player": catcher, "from": caught,
                 "score": self.scores.get(caught, 0)}]

    # ------------------------------------------------------------------ state
    def _zombie_state(self):
        tb = None
        if self.no_tagback and self.elapsed < self.no_tagback[1]:
            tb = [self.no_tagback[0], round(self.no_tagback[1] - self.elapsed, 1)]
        return {"living": self.living, "grace": round(self.grace, 2), "no_tagback": tb,
                "points": {str(k): round(v, 1) for k, v in self.points.items()}, "goal": self.rules.goal}

    def to_dict(self):
        d = super().to_dict()
        d["zombie"] = self._zombie_state()
        return d

    def snapshot(self):
        d = super().snapshot()
        d["zombie"] = self._zombie_state()
        return d

    def results(self):
        r = super().results()
        r["catches"] = {str(k): v for k, v in self.catches.items()}
        return r

    # ------------------------------------------------------------------- bots
    def bot_targets(self, pid, role="attack", claimed=()):
        if self.state != self.PLAYING or pid not in self.players or self.living is None:
            return []
        if pid != self.living:
            lp = self.positions.get(self.living)
            if lp is None:
                return []
            return [Target(f"player:{self.living}", lp, config.MP_TAG_RADIUS * 0.7, "player")]
        return [self._flee_target(pid)] if self.flee_spots else []

    def _flee_target(self, pid):
        """A living bot runs for somewhere far from the zombies, away from the nearest one - and keeps
        that goal a while (a new one when it got there, after 8 s, or when a zombie is closing in on it)."""
        me = self.positions.get(pid)
        zombies = [p for other, p in self.positions.items() if other != pid and other in self.players]
        cur = self._flee.get(pid)
        if cur is not None and me is not None:
            tid, pos, since = cur
            near_goal = zombies and min(float(np.linalg.norm(z - pos)) for z in zombies) < 8.0
            if self.elapsed - since < 8.0 and float(np.linalg.norm(pos - me)) > 3.0 and not near_goal:
                return Target(tid, pos, 2.0, "zone")
        if me is None:
            k = int(self.rng.randint(len(self.flee_spots)))
            pos = self.flee_spots[k]
        else:
            nearest = min(zombies, key=lambda z: float(np.linalg.norm(z - me))) if zombies else None
            away = (me - nearest) if nearest is not None else np.zeros(3, dtype=np.float32)
            best, best_score = None, None
            for k, spot in enumerate(self.flee_spots):
                d_me = float(np.hypot(spot[0] - me[0], spot[2] - me[2]))
                if not 12.0 <= d_me <= 70.0:
                    continue
                if nearest is not None and float(np.dot((spot - me)[[0, 2]], away[[0, 2]])) <= 0.0:
                    continue                           # not past the zombie on our heels
                d_z = min(float(np.hypot(spot[0] - z[0], spot[2] - z[2])) for z in zombies) if zombies else 50.0
                score = d_z - 0.4 * d_me + float(self.rng.rand()) * 3.0
                if best_score is None or score > best_score:
                    best, best_score = k, score
            if best is None:
                best = int(self.rng.randint(len(self.flee_spots)))
            k, pos = best, self.flee_spots[best]
        tid = f"flee:{k}:{int(self.elapsed * 10)}"
        self._flee[pid] = (tid, pos, self.elapsed)
        return Target(tid, pos, 2.0, "zone")


def create_match(world, rules, seed=1):
    """The match for these rules."""
    if rules.mode == GameRules.CTF:
        return CtfMatch(world, rules, seed)
    if rules.mode == GameRules.ZOMBIE:
        return ZombieMatch(world, rules, seed)
    return EmblemMatch(world, rules=rules, seed=seed)


def city_world(seed=None):
    """The game's level (the city) as a World - what a hosted server plays on."""
    from parkour.ledge_system import LedgeSystem
    from world.city import CityGenerator
    from world.generator import LevelGenerator
    ledge_system = LedgeSystem()
    gen = CityGenerator(config.CITY_SEED if seed is None else seed)
    obstacles = gen.build(LevelGenerator.build_zones())
    ledge_system.build(obstacles)
    return World(obstacles, ledge_system, gen.buildings)
