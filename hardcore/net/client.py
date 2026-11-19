"""Game client: joins a server, sends the local player's pose, keeps the world state it hears.

`connect()` does the hello / welcome handshake synchronously (so joining can fail
with a clear reason), then a background thread receives.  Remote players are
drawn MP_INTERP_DELAY in the past, interpolated between the two poses around that
moment, so they move smoothly even though snapshots arrive 20x a second.

Every pose carries the server-clock time it is from (`pt`), and the client maps the
server clock onto its own (the smallest arrival delay seen, creeping up slowly), so
remote players move on the timeline they actually moved on: not on when a snapshot
happened to arrive (network jitter), and a pose that two snapshots both carry (no
new update from that player in between) is not shown as a stop followed by a jump.
A pose that is late is briefly extrapolated along the last motion instead of freezing.
"""
import collections
import math
import socket
import threading
import time

import numpy as np

import config
from net.protocol import PROTOCOL_VERSION, LineReader, ProtocolError, encode


def _lerp_acro(a, b, k):
    """Interpolate a flip / roll between two snapshots (same spin: the angle; else the newer one)."""
    if a is None or b is None or a[0] != b[0] or b[1] < a[1]:
        return b
    return [b[0], a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k, b[3], b[4]]


class RemotePlayer:
    def __init__(self, pid, name, bot=False, team=None):
        self.pid = pid
        self.name = name
        self.bot = bot
        self.team = team
        self.score = 0
        self.history = collections.deque(maxlen=16)     # (pose time on the server clock, snapshot entry)

    def add(self, t, entry):
        """Record a pose from time t (server clock).  The same pose carried by another snapshot (same
        stamp) only refreshes the entry; a stamp from before the newest one is dropped."""
        h = self.history
        if h:
            last = h[-1][0]
            if abs(t - last) <= 1e-6:
                h[-1] = (last, entry)
                return
            if t < last:
                return
        h.append((t, entry))

    def sample(self, t):
        """Pose at time t (server clock): interpolated between the poses around it, briefly
        extrapolated past the newest one (None before the first)."""
        h = self.history
        if not h:
            return None
        if t <= h[0][0]:
            return h[0][1]
        items = list(h)
        for (t0, a), (t1, b) in zip(items[:-1], items[1:]):
            if t0 <= t <= t1:
                k = (t - t0) / max(1e-6, t1 - t0)
                out = dict(b)
                pa, pb = np.asarray(a["p"], np.float32), np.asarray(b["p"], np.float32)
                if float(np.linalg.norm(pb - pa)) < 5.0:            # not across a respawn
                    out["p"] = list(pa + (pb - pa) * k)
                    dy = (b["yaw"] - a["yaw"] + 180.0) % 360.0 - 180.0
                    out["yaw"] = a["yaw"] + dy * k
                    out["ac"] = _lerp_acro(a.get("ac"), b.get("ac"), k)
                return out
        # past the newest pose (its update is late): carry on along the last motion for a moment
        t1, b = items[-1]
        if len(items) >= 2:
            t0, a = items[-2]
            span = t1 - t0
            ahead = min(t - t1, config.MP_EXTRAPOLATE_MAX)
            pa, pb = np.asarray(a["p"], np.float32), np.asarray(b["p"], np.float32)
            if 1e-3 < span <= 0.25 and ahead > 0.0 and float(np.linalg.norm(pb - pa)) < 5.0:
                out = dict(b)
                out["p"] = list(pb + (pb - pa) * (ahead / span))
                return out
        return b


class GameClient:
    def __init__(self, host, port, name, timeout=5.0, clock=time.monotonic, team=None):
        self.host = host
        self.port = int(port)
        self.name = name
        self.team = team             # the side asked for; the server's answer replaces it
        self.timeout = timeout
        self.clock = clock
        self.sock = None
        self.id = None
        self.server_info = {}
        self.match = {}
        self.emblems = {}            # id -> {"id", "pos", "kind", "taken_by"}
        self.players = {}            # pid -> RemotePlayer (everyone, us included)
        self.remaining = 0
        self.time_left = 0.0
        self.rules = {}              # GameRules dict (mode, teams, time_limit, ...)
        self.team_scores = {}
        self.objectives = {"flags": [], "zones": []}     # capture the flag
        self.zombie = {}             # zombie tag: {"living", "grace", "points", "goal"}
        self.spawn = None
        self.events = collections.deque()
        self.lock = threading.Lock()
        self.connected = False
        self._send_lock = threading.Lock()
        self._thread = None
        self._reader = LineReader()
        self.clock_offset = None     # local clock minus server clock (None until the first snapshot)
        self._render_offset = None   # ... and the one remote players are drawn with (slews toward it)
        self._render_last = None

    # --------------------------------------------------------------- connection
    def connect(self):
        """Connect and complete the handshake.  Raises ConnectionError with a readable reason."""
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise ConnectionError(f"could not reach {self.host}:{self.port} ({exc.strerror or exc})") from exc
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock
        self._send({"t": "hello", "v": PROTOCOL_VERSION, "name": self.name, "team": self.team})
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._abort()
                raise ConnectionError("the server did not answer")
            sock.settimeout(remaining)
            try:
                data = sock.recv(65536)
            except OSError as exc:
                self._abort()
                raise ConnectionError(f"connection failed ({exc})") from exc
            if not data:
                self._abort()
                raise ConnectionError("the server closed the connection")
            for msg in self._reader.feed(data):
                if msg["t"] == "reject":
                    self._abort()
                    raise ConnectionError(str(msg.get("reason", "rejected")))
                if msg["t"] == "welcome":
                    self._welcome(msg)
                else:
                    self._handle(msg)
            if self.connected:
                break
        sock.settimeout(None)
        self._thread = threading.Thread(target=self._recv_loop, name="hardcore-client", daemon=True)
        self._thread.start()
        return self

    def _abort(self):
        try:
            self.sock.close()
        except OSError:
            pass
        self.sock = None

    def close(self):
        if self.sock is not None and self.connected:
            try:
                self._send({"t": "bye"})
            except OSError:
                pass
        self.connected = False
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _send(self, msg):
        data = encode(msg)
        with self._send_lock:
            if self.sock is None:
                return
            self.sock.sendall(data)

    def _recv_loop(self):
        try:
            while self.connected and self.sock is not None:
                data = self.sock.recv(65536)
                if not data:
                    break
                for msg in self._reader.feed(data):
                    self._handle(msg)
        except (OSError, ProtocolError):
            pass
        if self.connected:
            self.connected = False
            self.events.append({"t": "disconnected", "reason": "connection to the server lost"})

    # ---------------------------------------------------------------- messages
    def _welcome(self, msg):
        with self.lock:
            self.id = int(msg["id"])
            self.server_info = msg.get("server", {})
            self.rules = self.server_info.get("rules", {}) or {}
            self.team = msg.get("team")
            self._set_match(msg.get("match", {}))
            self.spawn = msg.get("spawn")
            now = self.clock()
            for p in msg.get("players", []):
                rp = self.players.setdefault(p["id"], RemotePlayer(p["id"], p.get("name", "?"), p.get("bot", False),
                                                                   p.get("team")))
                rp.score = p.get("score", 0)
                if p.get("p") is not None:
                    rp.add(self._stamp(p, None, now), p)
            me = self.players.setdefault(self.id, RemotePlayer(self.id, self.name, team=self.team))
            me.team = self.team
            self.connected = True
        self.events.append({"t": "welcome", "spawn": self.spawn, "match": self.match})

    def _set_match(self, match):
        self.match = match
        if match.get("rules"):
            self.rules = match["rules"]
        self.emblems = {e["id"]: dict(e) for e in match.get("emblems", [])}
        self.remaining = sum(1 for e in self.emblems.values() if e.get("taken_by") is None)
        self.time_left = match.get("time_left", 0.0)
        self.team_scores = dict(match.get("team_scores", {}))
        self.objectives = {"flags": list(match.get("flags", [])), "zones": list(match.get("zones", []))}
        self.zombie = dict(match.get("zombie") or {})
        for pid, s in match.get("scores", {}).items():
            if int(pid) in self.players:
                self.players[int(pid)].score = s
        for pid, team in match.get("teams", {}).items():
            if int(pid) in self.players:
                self.players[int(pid)].team = team

    def _handle(self, msg):
        t = msg["t"]
        now = self.clock()
        with self.lock:
            if t == "snap":
                seen = set()
                srv = msg.get("now")
                srv = float(srv) if isinstance(srv, (int, float)) and not isinstance(srv, bool) else None
                if srv is not None:
                    self._observe_clock(now - srv)
                for p in msg.get("players", []):
                    pid = p.get("id")
                    if not isinstance(pid, int) or p.get("p") is None:
                        continue
                    rp = self.players.get(pid)
                    if rp is None:
                        rp = self.players[pid] = RemotePlayer(pid, p.get("name", "?"), p.get("bot", False), p.get("team"))
                    rp.name = p.get("name", rp.name)
                    rp.score = p.get("score", rp.score)
                    rp.team = p.get("team", rp.team)
                    rp.add(self._stamp(p, srv, now), p)
                    seen.add(pid)
                self.remaining = msg.get("remaining", self.remaining)
                self.time_left = msg.get("time_left", self.time_left)
                m = msg.get("match") or {}
                if "team_scores" in m:
                    self.team_scores = dict(m["team_scores"])
                if "flags" in m:
                    self.objectives = {"flags": list(m.get("flags", [])), "zones": list(m.get("zones", []))}
                if "zombie" in m:
                    self.zombie = dict(m["zombie"] or {})
                return
            if t == "pickup":
                e = self.emblems.get(msg.get("emblem"))
                if e is not None:
                    e["taken_by"] = msg.get("player")
                rp = self.players.get(msg.get("player"))
                if rp is not None:
                    rp.score = msg.get("score", rp.score)
                self.remaining = msg.get("remaining", self.remaining)
                if "team_scores" in msg:
                    self.team_scores = dict(msg["team_scores"])
            elif t == "flag":
                rp = self.players.get(msg.get("player"))
                if rp is not None and "score" in msg:
                    rp.score = msg["score"]
                if "team_scores" in msg:
                    self.team_scores = dict(msg["team_scores"])
            elif t == "zombie":
                if msg.get("player") is not None:
                    self.zombie["living"] = msg["player"]
                if msg.get("action") == "caught":
                    self.zombie["grace"] = config.ZOMBIE_CATCH_GRACE
            elif t == "join":
                self.players.setdefault(msg["id"], RemotePlayer(msg["id"], msg.get("name", "?"), msg.get("bot", False),
                                                                msg.get("team")))
            elif t == "team":
                rp = self.players.get(msg.get("id"))
                if rp is not None:
                    rp.team = msg.get("team")
                if msg.get("id") == self.id:
                    self.team = msg.get("team")
            elif t == "respawn":
                self.spawn = msg.get("spawn")
                if msg.get("team"):
                    self.team = msg["team"]
            elif t == "leave":
                if msg.get("id") == -1:
                    self.connected = False
                    self.events.append({"t": "disconnected", "reason": "the host closed the game"})
                    return
                self.players.pop(msg.get("id"), None)
            elif t == "start":
                self._set_match(msg.get("match", {}))
                self.spawn = msg.get("spawn")
                if msg.get("team"):
                    self.team = msg["team"]
                for rp in self.players.values():
                    rp.score = 0
                    rp.history.clear()
            elif t == "over":
                for pid, s in msg.get("scores", {}).items():
                    rp = self.players.get(int(pid))
                    if rp is not None:
                        rp.score = s
        self.events.append(msg)

    # ------------------------------------------------------------------ client API
    def send_state(self, pos, yaw, pitch, state, hang_normal=None, speed=0.0, acro=None):
        msg = {"t": "state", "p": [round(float(c), 3) for c in pos], "yaw": round(float(yaw), 2),
               "pitch": round(float(pitch), 2), "st": str(state),
               "n": None if hang_normal is None else [round(float(hang_normal[0]), 3), round(float(hang_normal[2]), 3)],
               "spd": round(float(speed), 2), "ac": acro}
        try:
            self._send(msg)
        except OSError:
            if self.connected:
                self.connected = False
                self.events.append({"t": "disconnected", "reason": "connection to the server lost"})

    def send_team(self, team):
        try:
            self._send({"t": "team", "team": team})
        except OSError:
            pass

    def send_ready(self):
        try:
            self._send({"t": "ready"})
        except OSError:
            pass

    def poll_events(self):
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out

    # ------------------------------------------------------------------ clocks
    def _observe_clock(self, sample):
        """A snapshot arrived `sample` = local time - server time after it was sent.  The smallest
        such delay is the best estimate of the offset (every other arrival was only delayed more);
        it creeps up slowly so a one-off early arrival or clock drift cannot pin it for good."""
        if self.clock_offset is None or sample < self.clock_offset:
            self.clock_offset = sample
        else:
            self.clock_offset += (sample - self.clock_offset) * config.MP_CLOCK_CREEP

    def _stamp(self, entry, srv_now, local_now):
        """Server-clock time of a pose entry (its `pt`; a server without stamps: when it arrived)."""
        pt = entry.get("pt")
        if isinstance(pt, (int, float)) and not isinstance(pt, bool) and math.isfinite(pt):
            return float(pt)
        if srv_now is not None:
            return srv_now
        return local_now - (self.clock_offset or 0.0)

    def server_time(self, now=None):
        """The server clock now (None before the first snapshot)."""
        if self.clock_offset is None:
            return None
        return (self.clock() if now is None else now) - self.clock_offset

    def _draw_offset(self, now):
        """The clock offset remote players are drawn with: it follows the estimate by slewing (a few
        per cent faster or slower for a moment) instead of stepping, so a better estimate never
        makes everyone lurch; only a gross error (a stall, the first snapshot) is corrected at once."""
        target = self.clock_offset
        if target is None:
            return None
        last = self._render_last
        if self._render_offset is None or abs(target - self._render_offset) > config.MP_CLOCK_SNAP:
            self._render_offset = target
        elif last is not None:
            step = config.MP_CLOCK_SLEW * max(0.0, now - last)
            self._render_offset += float(np.clip(target - self._render_offset, -step, step))
        self._render_last = now if last is None else max(last, now)
        return self._render_offset

    def remote_poses(self, now=None):
        """[(RemotePlayer, pose dict)] for everyone but us, drawn MP_INTERP_DELAY in the past.
        `now` is a time on the local clock (default: the clock now)."""
        now = self.clock() if now is None else now
        with self.lock:
            offset = self._draw_offset(now)
        t = float("inf") if offset is None else now - offset - config.MP_INTERP_DELAY
        with self.lock:
            out = []
            for pid, rp in self.players.items():
                if pid == self.id:
                    continue
                pose = rp.sample(t)
                if pose is not None:
                    out.append((rp, pose))
            return out

    def scoreboard(self):
        """[(score, name, is_me, is_bot, team, pid)] best first."""
        with self.lock:
            rows = [(rp.score, rp.name, rp.pid == self.id, rp.bot, rp.team, rp.pid) for rp in self.players.values()]
        rows.sort(key=lambda r: (-r[0], r[1]))
        return rows

    @property
    def teams_game(self):
        return bool(self.rules.get("teams"))

    @property
    def mode(self):
        return self.rules.get("mode", "emblems")

    @property
    def living(self):
        """Zombie tag: the id of the living player (None in other games)."""
        return self.zombie.get("living") if self.mode == "zombie" else None

    def untaken_emblems(self):
        with self.lock:
            return [e for e in self.emblems.values() if e.get("taken_by") is None]
