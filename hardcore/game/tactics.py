"""Bot tactics: what a bot *decides* to do, at its intelligence level (game/bot_skill.py).

The bot's body and motor skills (running, climbing, leaping, routing through the
streets) are game/bot.py's `BotBrain`; it goes wherever it is told.  A `Tactician`
sits above it and tells it where, from what the bot can know:

Emblem hunt   The bot does not know where the emblems are.  It sees them (game/sight.py:
              in range, in its field of view, nothing in the way) and remembers them - one,
              three or all of them depending on its level - and goes for the nearest one it
              knows of; a sighting from afar is only a rough idea of where it is, sharpened as
              the bot gets closer.  With nothing in mind it searches: it remembers which areas
              of the map it has looked over (a few, many or all - Hard bots pool theirs with the
              other Hard bots on their side and split the map between them) and heads for ones
              it has not, climbing a tall roof for a view over the streets and roofs around.
              A remembered emblem that someone else picked up is only forgotten when the bot
              sees the spot empty.
Chasing       (zombie tag, a flag thief) Simple bots run straight at the quarry.  Complex and
              Hard bots hunt as a pack (`PackPlanner`): the nearest runs it down with a lead,
              the others head where it is going - spread out to its sides (Complex), or round
              it on the open side, herding it toward the edge of the map (Hard).  Where it is
              going is a guess made from how it has been moving (and, for Hard, how a runner
              flees a pack and where a flag thief's base is); each guess is checked against
              where the quarry turned out to be and a wrong one gives way to a better one.
              Hard bots also post one hunter in a flag thief's base to meet it there.
Fleeing       (the living player in zombie tag) Simple bots run away from the nearest zombie.
              Complex and Hard bots weigh every spot by how much sooner they get there than any
              zombie, keep out of corners and away from the zombies' lines, take to the roofs
              when they have a lead at full sprint (zombies climb slower) and leap roof to roof,
              and change direction suddenly now and then so they are hard to read.

Everything here reads the match (positions, flags, emblems) and answers with
game.match.Target objects for the BotBrain - the same thing the game rules hand out.
"""
import math
from collections import OrderedDict, deque

import numpy as np

import config
from game import bot_skill
from game.match import Target, GameRules, EMBLEM_HOVER

SEARCH_CELL = 16.0                  # the search map's areas (m)
NEAR_SEEN = 7.0                     # a spot this close is looked over whatever the line of sight
SAFE_LEAD = 25.0                    # a fleeing bot this far from every zombie can afford to be clever
HUNTER_SPEED = config.SPRINT_SPEED * config.ZOMBIE_RUN_SCALE   # what a chasing zombie can do (m/s)
RUNNER_SPEED = config.SPRINT_SPEED


def _flat(v):
    return np.array([v[0], 0.0, v[2]], dtype=np.float32)


def _hdist(a, b):
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2])))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros_like(v)


# =====================================================================================
# Shared: the world's search grid, everyone's movement
# =====================================================================================
class SearchGrid:
    """The map cut into SEARCH_CELL areas, each with up to three spots an emblem could float over
    (the street, the main roof level, the highest top) - what a searching bot has to look at.
    Built once per world."""

    def __init__(self, world):
        hf = world.sight
        lo, hi = world.extent()
        self.lo = np.array([float(lo[0]), float(lo[2])], dtype=np.float64)
        self.hi = np.array([float(hi[0]), float(hi[2])], dtype=np.float64)
        self.nx = max(1, int(math.ceil((self.hi[0] - self.lo[0]) / SEARCH_CELL)))
        self.nz = max(1, int(math.ceil((self.hi[1] - self.lo[1]) / SEARCH_CELL)))
        self.cells = []                 # cell index -> (i, k)
        self.centers = []               # cell index -> (x, z)
        self.samples = []               # cell index -> [np.array(x, y, z)]  (y: 1 m over the surface)
        self.kinds = []                 # cell index -> ["street" / "roof" / "top"]
        for i in range(self.nx):
            for k in range(self.nz):
                x0 = self.lo[0] + i * SEARCH_CELL
                z0 = self.lo[1] + k * SEARCH_CELL
                x1, z1 = min(x0 + SEARCH_CELL, self.hi[0]), min(z0 + SEARCH_CELL, self.hi[1])
                i0, k0 = hf.cell_of(x0, z0)
                i1, k1 = hf.cell_of(x1 - 1e-3, z1 - 1e-3)
                i0, k0 = max(0, i0), max(0, k0)
                i1, k1 = min(hf.nx - 1, i1), min(hf.nz - 1, k1)
                if i1 < i0 or k1 < k0:
                    continue
                block = hf.h[i0:i1 + 1, k0:k1 + 1]
                cx, cz = (x0 + x1) * 0.5, (z0 + z1) * 0.5
                pts, kinds = [], []

                def at(idx):
                    a, b = np.unravel_index(idx, block.shape)
                    return (hf.x0 + (i0 + a + 0.5) * hf.CELL, float(block[a, b]), hf.z0 + (k0 + b + 0.5) * hf.CELL)
                # the street: the open ground nearest the middle of the area
                ground = np.nonzero(block.ravel() < 0.3)[0]
                if len(ground):
                    xs = hf.x0 + (i0 + ground // block.shape[1] + 0.5) * hf.CELL
                    zs = hf.z0 + (k0 + ground % block.shape[1] + 0.5) * hf.CELL
                    j = int(np.argmin((xs - cx) ** 2 + (zs - cz) ** 2))
                    x, h, z = at(int(ground[j]))
                    pts.append(np.array([x, h + EMBLEM_HOVER, z], dtype=np.float32))
                    kinds.append("street")
                # the main roof level (the most common height above head height) and the highest top
                tall = block[block >= 1.2]
                if len(tall):
                    vals, counts = np.unique(np.round(tall, 1), return_counts=True)
                    level = float(vals[int(np.argmax(counts))])
                    cand = np.nonzero(np.abs(block.ravel() - level) < 0.06)[0]
                    if len(cand):
                        xs = hf.x0 + (i0 + cand // block.shape[1] + 0.5) * hf.CELL
                        zs = hf.z0 + (k0 + cand % block.shape[1] + 0.5) * hf.CELL
                        j = int(np.argmin((xs - cx) ** 2 + (zs - cz) ** 2))
                        x, h, z = at(int(cand[j]))
                        pts.append(np.array([x, h + EMBLEM_HOVER, z], dtype=np.float32))
                        kinds.append("roof")
                        # and the far side of that roof (one view of a roof is not the whole roof)
                        far = int(np.argmax((xs - x) ** 2 + (zs - z) ** 2))
                        if (xs[far] - x) ** 2 + (zs[far] - z) ** 2 > 36.0:
                            x2, h2, z2 = at(int(cand[far]))
                            pts.append(np.array([x2, h2 + EMBLEM_HOVER, z2], dtype=np.float32))
                            kinds.append("roof")
                    top = float(block.max())
                    if top > level + 1.5:
                        x, h, z = at(int(np.argmax(block)))
                        pts.append(np.array([x, h + EMBLEM_HOVER, z], dtype=np.float32))
                        kinds.append("top")
                if not pts:
                    continue
                self.cells.append((i, k))
                self.centers.append((cx, cz))
                self.samples.append(pts)
                self.kinds.append(kinds)
        self.centers = np.array(self.centers, dtype=np.float64).reshape(-1, 2)
        self.index = {c: n for n, c in enumerate(self.cells)}

    def __len__(self):
        return len(self.cells)

    def cell_at(self, x, z):
        """Index of the area (x, z) is in (None off the map)."""
        i = int(math.floor((x - self.lo[0]) / SEARCH_CELL))
        k = int(math.floor((z - self.lo[1]) / SEARCH_CELL))
        return self.index.get((i, k))

    def neighbours(self, n, ring=1):
        i, k = self.cells[n]
        out = []
        for di in range(-ring, ring + 1):
            for dk in range(-ring, ring + 1):
                if di or dk:
                    m = self.index.get((i + di, k + dk))
                    if m is not None:
                        out.append(m)
        return out


def search_grid(world):
    g = getattr(world, "_search_grid", None)
    if g is None:
        g = world._search_grid = SearchGrid(world)
    return g


class Track:
    """One player's recent movement: where it is and how fast it is going."""

    WINDOW = 0.7                     # seconds of positions the velocity is taken over

    def __init__(self):
        self.points = deque(maxlen=64)          # (t, pos)

    def add(self, t, pos):
        if self.points and t <= self.points[-1][0] + 1e-6:
            return
        pos = np.asarray(pos, dtype=np.float32).copy()
        if self.points and _hdist(pos, self.points[-1][1]) > 10.0:
            self.points.clear()                  # a respawn, not a run
        self.points.append((t, pos))

    @property
    def pos(self):
        return self.points[-1][1] if self.points else None

    def velocity(self):
        """Horizontal velocity over the last WINDOW seconds (zero when unknown)."""
        if len(self.points) < 2:
            return np.zeros(3, dtype=np.float32)
        t1, p1 = self.points[-1]
        t0, p0 = self.points[0]
        for t, p in self.points:
            if t1 - t <= self.WINDOW:
                t0, p0 = t, p
                break
        dt = t1 - t0
        if dt < 1e-3:
            return np.zeros(3, dtype=np.float32)
        v = (p1 - p0) / dt
        v[1] = 0.0
        return v.astype(np.float32)


class Tracker:
    """Everyone's tracks, fed from the match every tick (what anybody could see on the HUD / light
    columns: the living player's beam, a flag carrier's flag)."""

    def __init__(self):
        self.tracks = {}
        self.time = 0.0

    def update(self, t, positions):
        self.time = t
        for pid, p in positions.items():
            if p is None:
                continue
            self.tracks.setdefault(pid, Track()).add(t, p)
        for pid in list(self.tracks):
            if pid not in positions:
                del self.tracks[pid]

    def get(self, pid):
        return self.tracks.get(pid)


# =====================================================================================
# Memory: emblems seen and areas searched
# =====================================================================================
class Sighting:
    __slots__ = ("id", "true_pos", "est", "seen_at", "dist", "gone")

    def __init__(self, eid, true_pos, est, seen_at, dist):
        self.id = eid
        self.true_pos = np.asarray(true_pos, dtype=np.float32)
        self.est = np.asarray(est, dtype=np.float32)
        self.seen_at = float(seen_at)
        self.dist = float(dist)
        self.gone = False               # seen to be gone (then forgotten)


class EmblemMemory:
    """Emblems a bot has in mind (at most `capacity`; None: every one it has seen)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.items = OrderedDict()      # id -> Sighting

    def __contains__(self, eid):
        return eid in self.items

    def __len__(self):
        return len(self.items)

    def get(self, eid):
        return self.items.get(eid)

    def remember(self, sighting, here):
        """Keep a sighting; over capacity the one furthest from `here` is forgotten (the bot keeps the
        nearest in mind)."""
        self.items[sighting.id] = sighting
        if self.capacity is not None:
            while len(self.items) > self.capacity:
                far = max(self.items.values(), key=lambda s: _hdist(s.est, here))
                del self.items[far.id]

    def forget(self, eid):
        self.items.pop(eid, None)

    def values(self):
        return list(self.items.values())


class SearchMemory:
    """Which areas a bot has looked over: at most `capacity` fully searched areas are kept in mind
    (the longest-ago searched are forgotten; None: all).  The spots of an area seen so far are kept
    while the area is being looked over."""

    def __init__(self, grid, capacity):
        self.grid = grid
        self.capacity = capacity
        self.partial = {}               # cell -> bitmask of its samples seen (not all yet)
        self.done = OrderedDict()       # fully searched cells, oldest first

    def mask(self, n):
        if n in self.done:
            return (1 << len(self.grid.samples[n])) - 1
        return self.partial.get(n, 0)

    def searched(self, n):
        return n in self.done

    def unseen_samples(self, n):
        if n in self.done:
            return []
        m = self.partial.get(n, 0)
        return [j for j in range(len(self.grid.samples[n])) if not (m >> j) & 1]

    def mark(self, n, j=None):
        if n in self.done:
            self.done.move_to_end(n)
            return
        full = (1 << len(self.grid.samples[n])) - 1
        m = full if j is None else (self.partial.get(n, 0) | (1 << j))
        if m != full:
            self.partial[n] = m
            return
        self.partial.pop(n, None)
        self.done[n] = True
        if self.capacity is not None:
            while len(self.done) > self.capacity:
                self.done.popitem(last=False)

    def count(self):
        return len(self.done)

    def clear(self):
        self.partial.clear()
        self.done.clear()


class Hive:
    """What the bots of one side and one level share.  Hard bots pool their emblem sightings and
    searched areas here and split the emblems and the map between them; every Complex / Hard pack
    plans its chases here."""

    def __init__(self, grid, skill):
        self.skill = skill
        self.search = SearchMemory(grid, None) if skill.shared else None
        self.emblems = EmblemMemory(None) if skill.shared else None
        self.emblem_claims = {}         # emblem id -> pid going for it
        self.explore_claims = {}        # pid -> the exploration it is on
        self.pack = PackPlanner(skill)
        self.seen_alive = {}            # pid -> last time that member decided something
        self.hunts = {}                 # pid -> (quarry, time)

    def touch(self, pid, now):
        self.seen_alive[pid] = now

    def active(self, now, within=1.0):
        return {pid for pid, t in self.seen_alive.items() if now - t <= within}

    def hunting(self, pid, quarry, now):
        self.hunts[pid] = (quarry, now)
        self.touch(pid, now)

    def hunters_of(self, quarry, now, within=0.5):
        return [pid for pid, (q, t) in self.hunts.items() if q == quarry and now - t <= within]

    def reset(self):
        if self.search is not None:
            self.search = SearchMemory(self.search.grid, None)
            self.emblems = EmblemMemory(None)
        self.emblem_claims.clear()
        self.explore_claims.clear()
        self.pack = PackPlanner(self.skill)
        self.hunts.clear()


# =====================================================================================
# Chasing: the pack
# =====================================================================================
class PackPlanner:
    """Splits a quarry between the hunters of a hive.

    A runner faster than its hunters (the living player in zombie tag is; a flag thief is as fast as
    anyone) is caught when it slips - a turn, a climb, something in the way - and only hunters close
    behind it can take the chance.  Measured with tools/bot_levels.py (DESIGN.md 11.6): every way of
    spreading a pack out that was tried - wide and tight flanks, cut-offs, a net round it, a ring ahead
    of it, waiting where it was making for, meeting it on its way - caught a faster runner less often
    than every hunter simply running it down, against three kinds of runner (including one that runs in
    long straight lines, the way people do).  So every hunter runs it down, straight at where it is,
    read live from its track every tick.  One job is worth taking a hunter off the chase: a flag thief
    has to get into its own base to score, so a Hard side posts the hunter that can beat the thief there
    in that base - the thief runs into it.
    """

    def __init__(self, skill):
        self.skill = skill
        self.plans = {}                 # quarry pid -> (time, {hunter pid: role})

    def plan(self, now, quarry, track, hunters, bot_pids, dest=None, dest_exact=False):
        """{pid: role} for the hunting bots of this hive: ("chase",) - run the quarry down - or
        ("fixed", point) - wait there.  `hunters`: {pid: pos} of every hunter; `bot_pids`: the ones this
        hive steers; `dest`: where the quarry has to get to (`dest_exact`: a flag thief's base)."""
        cached = self.plans.get(quarry)
        if cached is not None and now - cached[0] < self.skill.replan and set(cached[1]) >= set(bot_pids):
            return cached[1]
        qpos = track.pos
        mine = sorted(bot_pids, key=lambda pid: _hdist(hunters[pid], qpos))
        out = {pid: ("chase",) for pid in mine}
        if dest is not None and dest_exact and self.skill.pursuit == "pack" and len(mine) >= 2:
            # (the nearest keeps running it down; of the others, the one that can be there first waits)
            camper = min(mine[1:], key=lambda pid: _hdist(hunters[pid], dest))
            if _hdist(hunters[camper], dest) / HUNTER_SPEED < _hdist(qpos, dest) / RUNNER_SPEED + 4.0:
                out[camper] = ("fixed", np.asarray(dest, dtype=np.float32))
        self.plans[quarry] = (now, out)
        return out

    @staticmethod
    def resolve(role, track):
        """The point a role means right now (a chase: where the quarry is this tick)."""
        if role[0] == "fixed":
            return role[1], "point"
        return track.pos, "player"


# =====================================================================================
# The tactician: one per bot
# =====================================================================================
def building_roofs(world):
    """Per building: a standing spot on its roof (None if it has none) - for climbing / roof hopping."""
    cached = getattr(world, "_roof_points", None)
    if cached is not None:
        return cached
    out = []
    for spec, _ in world.buildings:
        best = None
        for fx, fz in ((0.5, 0.5), (0.3, 0.3), (0.7, 0.7), (0.3, 0.7), (0.7, 0.3)):
            x, z = spec.x + spec.width * fx, spec.z + spec.depth * fz
            top = world.ground_below(x, z, spec.height + 4.0)
            if top is not None and abs(top - spec.height) < 0.05 and world.standing_spot_free(x, top, z):
                best = np.array([x, top + config.PLAYER_HALF_HEIGHT, z], dtype=np.float32)
                break
        out.append(best)
    world._roof_points = out
    return out


def building_under(world, pos):
    """Index of the building whose roof the body stands on (None on the ground / anything else)."""
    feet = float(pos[1]) - config.PLAYER_HALF_HEIGHT
    for i, (spec, _) in enumerate(world.buildings):
        if (spec.x - 0.3 <= float(pos[0]) <= spec.x + spec.width + 0.3 and spec.z - 0.3 <= float(pos[2]) <= spec.z + spec.depth + 0.3
                and spec.height - 0.6 <= feet <= spec.height + 4.0):
            return i
    return None


class Tactician:
    SIGHT_EVERY = 0.2                   # how often the bot looks for emblems (s)
    SCAN_EVERY = 0.4                    # how often it looks an area over
    EXPLORE_TIMEOUT = 25.0              # an exploration goal not reached in this long is dropped

    def __init__(self, pid, level, world, hive, tracker, seed=0):
        self.pid = pid
        self.skill = bot_skill.skill(level)
        self.world = world
        self.hive = hive
        self.tracker = tracker
        self.rng = np.random.RandomState(seed & 0x7FFFFFFF)
        self.grid = search_grid(world)
        self.memory = hive.emblems if self.skill.shared else EmblemMemory(self.skill.emblem_memory)
        self.search = hive.search if self.skill.shared else SearchMemory(self.grid, self.skill.search_memory)
        lo, hi = world.extent()
        self.extent = (np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32))
        self.next_sight = 0.0
        self.next_scan = 0.0
        self.current = None             # emblem id going for
        self.last_refined = {}          # emblem id -> time last seen
        self.explore = None             # {"point", "cell", "since", "id", "vantage"}
        self.flee = None                # {"point", "since", "id", "rooftop"}: a run across the roofs
        self.counter = 0
        self.failed = []                # [(point, until)]: places it could not get to lately
        self.sweeps = 0                 # times the whole map was looked over without finding everything
        self.spotted = 0                # emblems spotted (telemetry)
        self.decisions = []             # recent decisions (telemetry / tests)

    # ------------------------------------------------------------------ entry
    def reset(self):
        """A new round: nothing seen, nothing searched."""
        if self.skill.shared:
            self.memory, self.search = self.hive.emblems, self.hive.search
        else:
            self.memory = EmblemMemory(self.skill.emblem_memory)
            self.search = SearchMemory(self.grid, self.skill.search_memory)
        self.current = None
        self.explore = None
        self.flee = None
        self.last_refined = {}
        self.sweeps = 0

    def targets(self, match, now, body, brain=None, role="attack", claimed=()):
        """What this bot goes for now: [Target] for its BotBrain."""
        if match.state != match.PLAYING or self.pid not in match.players:
            return []
        mode = match.mode
        if mode == GameRules.EMBLEMS:
            return self._emblem_hunt(match, now, body, brain, claimed)
        if mode == GameRules.ZOMBIE:
            living = match.living
            if living is None:
                return []
            if living == self.pid:
                if self.skill.flee == "basic":
                    return match.bot_targets(self.pid, role, claimed)
                return self._flee(match, now, body)
            hunters = {pid: match.positions[pid] for pid in match.players
                       if pid != living and match.positions.get(pid) is not None}
            return self._hunt(match, now, body, living, hunters, role=role, claimed=claimed)
        if mode == GameRules.CTF:
            objective = match.bot_targets(self.pid, role, claimed)
            if objective and objective[0].kind == "player" and str(objective[0].id).startswith("player:"):
                quarry = int(str(objective[0].id).split(":")[1])
                if match.teams:
                    side = match.players.get(self.pid)
                    hunters = {pid: match.positions[pid] for pid, t in match.players.items()
                               if t == side and match.positions.get(pid) is not None}
                    theirs = match.players.get(quarry)
                    dest = match.zones[theirs].center if theirs in match.zones else None
                else:
                    hunters = {pid: match.positions[pid] for pid in match.players
                               if pid != quarry and match.positions.get(pid) is not None}
                    dest = match.zones["drop"].center if "drop" in match.zones else None
                if dest is not None:
                    dest = np.asarray(dest, dtype=np.float32) + np.array([0.0, config.PLAYER_HALF_HEIGHT, 0.0], np.float32)
                return self._hunt(match, now, body, quarry, hunters, dest=dest, role=role, claimed=claimed,
                                  fallback=objective, dest_exact=True)
            return objective
        return match.bot_targets(self.pid, role, claimed)

    def _id(self, prefix):
        self.counter += 1
        return f"{prefix}:{self.pid}:{self.counter}"

    def _log(self, what):
        self.decisions.append(what)
        if len(self.decisions) > 50:
            del self.decisions[:10]

    # ============================================================= emblem hunt
    def _cone(self, body):
        cone = self.skill.sight_cone_deg
        if self.skill.name != bot_skill.SIMPLE and body.on_ground and body.feet_y > 3.0:
            cone = 360.0                          # up on a roof it looks all round
        return cone

    def _in_view(self, eye, fwd, point, cone, rng):
        d = float(np.linalg.norm(point - eye))
        if d > rng:
            return False
        if cone >= 359.0 or d < NEAR_SEEN:
            return True
        v = _unit(_flat(point - eye))
        return float(np.dot(v, fwd)) >= math.cos(math.radians(cone * 0.5))

    def look(self, match, now, body):
        """Spot emblems in view; notice remembered ones that are gone."""
        eye = body.eye_pos
        fwd = _unit(_flat(body.forward_h))
        cone = self._cone(body)
        rng = self.skill.sight_range
        hf = self.world.sight
        live = {e.id: e for e in match.emblems if e.taken_by is None}
        cands = [e for e in live.values() if self._in_view(eye, fwd, e.pos, cone, rng)]
        if cands:
            pts = np.array([e.pos for e in cands] + [e.pos + np.array([0.0, 0.5, 0.0], np.float32) for e in cands],
                           dtype=np.float32)
            vis = hf.visible_many(eye, pts)
            n = len(cands)
            for j, e in enumerate(cands):
                if vis[j] or vis[n + j]:
                    self._saw(e, float(np.linalg.norm(e.pos - eye)), now, body)
        # remembered ones: gone (someone got there first) is only noticed by looking at the spot
        for s in self.memory.values():
            if s.id in live:
                continue
            near = _hdist(s.est, body.pos) < 4.0 and abs(float(s.est[1]) - float(body.pos[1])) < 3.0
            if near or (self._in_view(eye, fwd, s.true_pos, cone, rng) and hf.visible(eye, s.true_pos)):
                self.memory.forget(s.id)
                self._log(("gone", s.id))
        # arrived where we thought it was and it is not there (a rough sighting): look again
        for s in self.memory.values():
            if (s.id in live and _hdist(s.est, body.pos) < 2.0 and abs(float(s.est[1]) - float(body.pos[1])) < 2.0
                    and now - self.last_refined.get(s.id, -1e9) > 1.5):
                self.memory.forget(s.id)
                self._log(("lost", s.id))

    def _saw(self, e, dist, now, body):
        err = min(self.skill.position_error * dist, self.skill.position_error * 100.0)
        old = self.memory.get(e.id)
        if old is not None and err >= float(np.linalg.norm(old.est - old.true_pos)) - 1e-6:
            old.seen_at = now
            self.last_refined[e.id] = now
            return
        est = e.pos.copy()
        if err > 0.0:
            a = ((self.pid * 7919 + int(e.id) * 104729) % 360) * math.pi / 180.0
            d = np.array([math.cos(a), 0.0, math.sin(a)], dtype=np.float32)
            surface = float(e.pos[1]) - EMBLEM_HOVER
            for _ in range(4):
                cand = e.pos + d * err
                top = self.world.ground_below(float(cand[0]), float(cand[2]), surface + 0.5)
                if top is not None and abs(top - surface) < 0.3:
                    est = cand.astype(np.float32)
                    break
                err *= 0.5
        if old is None:
            self.spotted += 1
            self._log(("spotted", e.id, round(dist, 1)))
        self.memory.remember(Sighting(e.id, e.pos, est, now, dist), body.pos)
        self.last_refined[e.id] = now

    def scan(self, now, body):
        """Look the areas in view over (and the ground right around us)."""
        grid = self.grid
        if not len(grid):
            return
        eye = body.eye_pos
        here = np.array([float(body.pos[0]), float(body.pos[2])])
        d = np.hypot(grid.centers[:, 0] - here[0], grid.centers[:, 1] - here[1])
        # what is right around us is seen whatever the line of sight
        for n in np.nonzero(d < SEARCH_CELL * 1.2)[0]:
            for j in self.search.unseen_samples(int(n)):
                if float(np.linalg.norm(grid.samples[n][j] - body.pos)) < NEAR_SEEN:
                    self.search.mark(int(n), j)
        rng = self.skill.sight_range
        cells, idx, pts = [], [], []
        for n in np.argsort(d):
            n = int(n)
            if d[n] > rng + SEARCH_CELL:
                break
            for j in self.search.unseen_samples(n):
                cells.append(n)
                idx.append(j)
                pts.append(grid.samples[n][j])
            if len(pts) > 6 * self.skill.scan_cells:
                break
        if not pts:
            return
        pts = np.array(pts, dtype=np.float32)
        rel = pts - eye[None, :]
        dist = np.linalg.norm(rel, axis=1)
        ok = dist <= rng
        cone = self._cone(body)
        if cone < 359.0:
            fwd = _unit(_flat(body.forward_h))
            flat = rel.copy()
            flat[:, 1] = 0.0
            fn = np.linalg.norm(flat, axis=1) + 1e-6
            cos = (flat @ fwd) / fn
            ok &= (cos >= math.cos(math.radians(cone * 0.5))) | (dist < NEAR_SEEN)
        sel = np.nonzero(ok)[0]
        if not len(sel):
            return
        keep, seen_cells = [], set()
        for k in sel:
            if cells[k] not in seen_cells:
                if len(seen_cells) >= self.skill.scan_cells:
                    continue
                seen_cells.add(cells[k])
            keep.append(int(k))
        vis = self.world.sight.visible_many(eye, pts[keep])
        for k, v in zip(keep, vis):
            if v:
                self.search.mark(cells[k], idx[k])

    def _emblem_hunt(self, match, now, body, brain, claimed):
        if now >= self.next_sight:
            self.next_sight = now + self.SIGHT_EVERY
            self.look(match, now, body)
        if now >= self.next_scan:
            self.next_scan = now + self.SCAN_EVERY
            self.scan(now, body)
        blocked = set()
        if brain is not None:
            blocked = {k[1] for k, v in brain.blacklist.items() if v > 0.0 and k[0] == "target"}
        known = [s for s in self.memory.values() if s.id not in blocked]
        if self.skill.shared:
            # split the emblems: leave one a hive mate is nearer to for them
            mine = []
            for s in known:
                owner = self.hive.emblem_claims.get(s.id)
                if owner is not None and owner != self.pid and owner in self.hive.active(now):
                    continue
                mine.append(s)
            known = mine or known
        elif claimed:
            free = [s for s in known if s.id not in claimed]
            known = free or known
        if known:
            feet = float(body.feet_y)
            rivals = []
            if self.skill.name == bot_skill.HARD:
                mates = set(self.hive.seen_alive) if self.skill.shared else set()
                rivals = [np.asarray(p, dtype=np.float32) for pid, p in match.positions.items()
                          if pid != self.pid and pid not in mates and p is not None
                          and (not match.teams or match.players.get(pid) != match.players.get(self.pid))]

            def cost(s):
                c = _hdist(s.est, body.pos) + 2.5 * max(0.0, float(s.est[1]) - EMBLEM_HOVER - feet)
                if rivals:
                    # a mastermind does not race for one somebody else will get to first
                    d_rival = min(_hdist(s.est, r) + 2.5 * max(0.0, float(s.est[1]) - float(r[1])) for r in rivals)
                    if d_rival < 0.75 * c:
                        c += 60.0
                return c
            best = min(known, key=cost)
            cur = next((s for s in known if s.id == self.current), None)
            if cur is not None and cost(cur) <= cost(best) * 1.25 + 2.0:
                best = cur
            if best.id != self.current:
                self._log(("go", best.id))
            self.current = best.id
            self.explore = None
            if self.skill.shared:
                self.hive.emblem_claims[best.id] = self.pid
                self.hive.touch(self.pid, now)
            return [Target(best.id, best.est, config.MP_PICKUP_RADIUS * 0.9)]
        self.current = None
        return [self._explore_target(now, body, brain)]

    # ------------------------------------------------------------------ explore
    def _failed_near(self, point, now, radius=12.0):
        self.failed = [(p, until) for p, until in self.failed if until > now]
        return any(_hdist(p, point) < radius and abs(float(p[1]) - float(point[1])) < 4.0 for p, _ in self.failed)

    def _explore_done(self, now, body, brain):
        ex = self.explore
        if ex is None:
            return True
        if now - ex["since"] > self.EXPLORE_TIMEOUT or (
                brain is not None and brain.blacklist.get(("target", ex["id"]), 0.0) > 0.0):
            self.failed.append((ex["point"], now + 90.0))     # could not get there: try elsewhere
            return True
        p = ex["point"]
        if _hdist(p, body.pos) < 3.0 and abs(float(p[1]) - float(body.pos[1])) < 2.5:
            if ex.get("vantage"):
                ex.setdefault("arrived", now)
                return now - ex["arrived"] > 1.2       # look round from up there first
            return True
        if not ex.get("vantage") and self.skill.explore != "wander" and ex["cell"] is not None \
                and self.search.searched(ex["cell"]):
            return True                                 # seen it from here: on to the next
        return False

    def _explore_target(self, now, body, brain):
        if self.explore is not None and (not self._explore_done(now, body, brain)
                                         or now - self.explore["since"] < 0.5):
            ex = self.explore
            return Target(ex["id"], ex["point"], 2.0, "zone")
        grid = self.grid
        pos = body.pos
        here = np.array([float(pos[0]), float(pos[2])])
        d = np.hypot(grid.centers[:, 0] - here[0], grid.centers[:, 1] - here[1])
        unsearched = np.array([not self.search.searched(n) for n in range(len(grid))])
        if not unsearched.any():
            # everything looked over and there are still emblems out there (the HUD says how many):
            # something was missed - look the map over again
            self.search.clear()
            self.sweeps += 1
            self._log(("search", "again"))
            unsearched = np.ones(len(grid), dtype=bool)
        cand = np.nonzero(unsearched & (d >= 20.0))[0]
        if len(cand) == 0:
            cand = np.nonzero(unsearched)[0]
        point, cell, vantage = None, None, False
        if self.skill.explore == "wander":
            near = [int(n) for n in cand if 15.0 <= d[n] <= 70.0]
            pool = near or [int(n) for n in cand]
            cell = pool[int(self.rng.randint(len(pool)))]
        else:
            fwd = _unit(_flat(body.forward_h))
            claims = []
            if self.skill.shared:
                claims = [c["point"] for pid, c in self.hive.explore_claims.items()
                          if pid != self.pid and pid in self.hive.active(now)]
            best, best_score = None, None
            for n in cand:
                n = int(n)
                if self.failed and self._failed_near(np.array([grid.centers[n][0], body.pos[1], grid.centers[n][1]]), now):
                    continue
                dirv = np.array([grid.centers[n][0] - here[0], 0.0, grid.centers[n][1] - here[1]])
                turn = 1.0 - float(np.dot(_unit(dirv), fwd)) if d[n] > 1.0 else 0.0
                around = sum(1 for m in grid.neighbours(n) if unsearched[m])
                score = d[n] - 5.0 * around + 8.0 * turn + 6.0 * self.rng.rand()
                for c in claims:                       # someone of ours is searching there
                    dc = math.hypot(grid.centers[n][0] - float(c[0]), grid.centers[n][1] - float(c[2]))
                    if dc < 45.0:
                        score += (45.0 - dc) * 1.5
                if best_score is None or score < best_score:
                    best, best_score = n, score
            cell = best
            if self.skill.vantage and self.sweeps == 0:
                v = self._vantage(now, body, unsearched, claims)
                if v is not None and (best_score is None or v[2] < best_score):
                    point, vantage, cell = v[0], True, None
        if point is None:
            unseen = self.search.unseen_samples(cell) or list(range(len(grid.samples[cell])))
            kinds = grid.kinds[cell]
            # street first (it is on the way); a roof we cannot see from below means climbing it.  Once
            # a look over everything has missed something, it is up on a roof: walk the roofs
            first = "street" if self.sweeps == 0 else "roof"
            order = sorted(unseen, key=lambda j: 0 if kinds[j] == first else 1)
            if self.skill.explore == "wander":
                order = [unseen[int(self.rng.randint(len(unseen)))]]

            def body_at(j):
                q = grid.samples[cell][j]
                return np.array([q[0], float(q[1]) - EMBLEM_HOVER + config.PLAYER_HALF_HEIGHT, q[2]], dtype=np.float32)
            reachable = [j for j in order if not (self.failed and self._failed_near(body_at(j), now))]
            if not reachable:
                self.search.mark(cell)                  # nothing there it can get to: count it as looked at
                reachable = order
            point = body_at(reachable[0])
        self.explore = {"point": point, "cell": cell, "since": now, "id": self._id("explore"), "vantage": vantage}
        if self.skill.shared:
            self.hive.explore_claims[self.pid] = self.explore
            self.hive.touch(self.pid, now)
        self._log(("explore", "vantage" if vantage else cell))
        return Target(self.explore["id"], point, 2.0, "zone")

    def _vantage(self, now, body, unsearched, claims):
        """A roof to climb for a look over unsearched ground: (point, spots it sees, score) or None.
        The point is near the roof's edge on the side of the unsearched ground (from the middle of a
        roof its own edge hides everything lower), and a roof only counts for the unseen spots it
        really has a line of sight to."""
        if body.feet_y > 3.0:
            return None                                  # already up high
        grid = self.grid
        hf = self.world.sight
        best = None
        for i, (spec, _) in enumerate(self.world.buildings):
            if spec.height < 5.0:
                continue
            cx, cz = spec.x + spec.width * 0.5, spec.z + spec.depth * 0.5
            dist = math.hypot(cx - float(body.pos[0]), cz - float(body.pos[2]))
            if dist > 45.0 or any(math.hypot(cx - float(c[0]), cz - float(c[2])) < 25.0 for c in claims):
                continue
            reach = min(self.skill.sight_range, 25.0 + 6.0 * spec.height)
            dc = np.hypot(grid.centers[:, 0] - cx, grid.centers[:, 1] - cz)
            near = np.nonzero(unsearched & (dc < reach) & (dc > 6.0))[0]
            if len(near) < 4:
                continue
            # stand near the edge facing the unsearched ground
            aim = np.mean(grid.centers[near], axis=0)
            dx, dz = aim[0] - cx, aim[1] - cz
            n = math.hypot(dx, dz)
            if n > 1e-3:
                dx, dz = dx / n, dz / n
            spot = None
            for inset in (1.5, 3.0, None):
                if inset is None:
                    x, z = cx, cz
                else:
                    x = float(np.clip(cx + dx * spec.width, spec.x + inset, spec.x + spec.width - inset))
                    z = float(np.clip(cz + dz * spec.depth, spec.z + inset, spec.z + spec.depth - inset))
                top = self.world.ground_below(x, z, spec.height + 4.0)
                if top is not None and abs(top - spec.height) < 0.05 and self.world.standing_spot_free(x, top, z):
                    spot = np.array([x, top + config.PLAYER_HALF_HEIGHT, z], dtype=np.float32)
                    break
            if spot is None or (self.failed and self._failed_near(spot, now)):
                continue
            pts = [grid.samples[int(m)][j] for m in near for j in self.search.unseen_samples(int(m))]
            if not pts:
                continue
            eye = spot + np.array([0.0, config.PLAYER_EYE_HEIGHT - config.PLAYER_HALF_HEIGHT, 0.0], np.float32)
            seen = int(np.sum(hf.visible_many(eye, np.array(pts, dtype=np.float32))))
            if seen < 4:
                continue
            score = dist + 2.0 * spec.height - 3.0 * seen
            if best is None or score < best[2]:
                best = (spot, seen, score)
        return best

    # ================================================================ chasing
    def _hunt(self, match, now, body, quarry, hunters, dest=None, role="attack", claimed=(), fallback=None,
              dest_exact=False):
        track = self.tracker.get(quarry)
        if track is None or track.pos is None:
            return fallback if fallback is not None else match.bot_targets(self.pid, role, claimed)
        if self.skill.pursuit == "direct" or self.pid not in hunters:
            return [Target(f"player:{quarry}", track.pos, config.MP_TAG_RADIUS * 0.7, "player")]
        self.hive.hunting(self.pid, quarry, now)
        mates = [pid for pid in self.hive.hunters_of(quarry, now) if pid in hunters]
        plan = self.hive.pack.plan(now, quarry, track, hunters, mates, dest=dest, dest_exact=dest_exact)
        role = plan.get(self.pid, ("chase",))
        point, kind = PackPlanner.resolve(role, track)
        radius = config.MP_TAG_RADIUS * 0.7 if kind == "player" else 2.0
        return [Target(f"player:{quarry}", point, radius, "player")]

    # ================================================================= fleeing
    def _flee(self, match, now, body):
        """Run from the zombies (Complex and Hard; Simple bots use the game's plain rule).

        Measured (tools/bot_levels.py flee, DESIGN.md 11.6): against zombies that are only a little
        slower, nothing beat the plain rule - ZombieMatch._flee_target: the spot furthest from them, past
        the one on its heels - and spending a lead on climbing onto roofs or on sudden changes of
        direction let the zombies catch up.  So the clever bots run by the plain rule, with one addition
        that costs no lead: up on a roof with a lead, with another roof a leap away, they run on across
        the rooftops (Hard bots much more readily) - they never climb down into the streets to do it."""
        me = body.pos
        zombies = [np.asarray(match.positions[pid], dtype=np.float32) for pid in match.players
                   if pid != self.pid and match.positions.get(pid) is not None]
        hard = self.skill.flee == "mastermind"
        threat = min((_hdist(z, me) for z in zombies), default=99.0)
        climbing = body.state in ("CLIMB_UP", "LEDGE_HANG", "SHIMMY", "MANTLE_VAULT", "CORNER_TURN")
        cur = self.flee
        on = building_under(self.world, me)
        roofs = building_roofs(self.world)
        linked = {j for j, _ in self.world.roof_links(max_gap=4.0).get(on, []) if roofs[j] is not None} \
            if on is not None else set()
        if cur is not None and cur.get("rooftop"):
            reached = _hdist(cur["point"], me) < 3.0 and abs(float(cur["point"][1]) - float(me[1])) < 2.5
            if climbing or body.state in ("AIRBORNE", "ROLL") or not (reached or now - cur["since"] > 8.0
                                                                      or threat < 8.0):
                return [Target(cur["id"], cur["point"], 2.0, "zone")]     # on its way across the roofs
            cur = self.flee = None
        if threat >= SAFE_LEAD and linked and not climbing:
            nearest = min(zombies, key=lambda z: _hdist(z, me)) if zombies else None
            best, best_score = None, None
            for j in linked:
                sp = roofs[j]
                if nearest is not None and float(np.dot(_flat(sp - me), _flat(me - nearest))) <= 0.0:
                    continue                             # not back toward them
                d_z = min((_hdist(sp, z) for z in zombies), default=50.0)
                score = d_z - 0.4 * _hdist(sp, me) + (6.0 if hard else 3.0) + self.rng.rand() * 3.0
                if best_score is None or score > best_score:
                    best, best_score = sp, score
            # it has to beat what the plain rule would pick (taken with the same measure)
            plain = match._flee_target(self.pid) if hasattr(match, "_flee_target") else None
            if best is not None and plain is not None:
                d_plain = min((_hdist(plain.pos, z) for z in zombies), default=50.0) - 0.4 * _hdist(plain.pos, me)
                if best_score > d_plain:
                    self.flee = {"point": np.asarray(best, dtype=np.float32).copy(), "since": now,
                                 "id": self._id("flee"), "rooftop": True}
                    self._log(("flee", "rooftop", round(best_score, 1)))
                    return [Target(self.flee["id"], self.flee["point"], 2.0, "zone")]
        self.flee = None
        return match.bot_targets(self.pid)
