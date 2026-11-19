"""The listen server: hosted from inside a game client (or headless), up to 15 players.

One background thread runs everything: non-blocking sockets on a selector, the
match (emblem hunt or capture the flag, free for all or red vs blue: game/match.py),
the bots (real PlayerControllers driven by game.bot.BotBrain, stepped at the physics
rate) and the snapshots sent to every client.  The server is authoritative for the
game: a pickup, a flag taken or captured, a tag, happens when a player's reported
movement passes through it.  Players never collide with each other (there is no
player-vs-player collision in the engine at all).

Slots: `max_players` counts everyone, bots included.  The host asks for a number of
bots (at most max_players - 1: the host has a slot); a bot gives its slot up when a
person joins a full game and comes back when someone leaves.  In a teams game
people pick their side and the bots move between the sides to keep them even.
"""
import selectors
import socket
import threading
import time

import numpy as np

import config
from game import teams as T
from game import bot_skill
from game.match import GameRules, create_match
from net.protocol import (PROTOCOL_VERSION, LineReader, ProtocolError, encode, finite_vec, finite, clean_name,
                          clean_acro)

MAX_OUTBUF = 4 * 1024 * 1024       # a client that stops reading is dropped before it eats the host's memory
BOT_NAMES = ("Ace", "Blitz", "Comet", "Dash", "Echo", "Flux", "Gale", "Hex", "Ion", "Jolt", "Kite", "Lux",
             "Moss", "Nova")


class PlayerInfo:
    def __init__(self, pid, name, bot=False, team=None):
        self.pid = pid
        self.name = name
        self.bot = bot
        self.team = team
        self.pos = None
        self.prev_pos = None
        self.yaw = 0.0
        self.pitch = 0.0
        self.st = "GROUND"
        self.n = None              # hang facing (rail normal x, z) while on a wall
        self.spd = 0.0
        self.ac = None             # flip / roll in progress
        self.pt = 0.0              # server clock (GameServer.now) at which pos was the player's pose
        self.ready = False

    def to_dict(self, score):
        return {"id": self.pid, "name": self.name, "bot": self.bot, "team": self.team,
                "p": None if self.pos is None else [round(float(c), 3) for c in self.pos],
                "yaw": round(self.yaw, 2), "pitch": round(self.pitch, 2), "st": self.st, "n": self.n,
                "spd": round(self.spd, 2), "ac": self.ac, "pt": round(float(self.pt), 4), "score": score}


class ClientConn:
    def __init__(self, sock, addr):
        self.sock = sock
        self.addr = addr
        self.reader = LineReader()
        self.out = bytearray()
        self.player = None         # PlayerInfo once the hello was accepted
        self.closing = False


class BotPlayer:
    """A server-side bot: a headless PlayerController driven by BotBrain (its body and motor
    skills), told what to go for by a Tactician at the bot's intelligence level (game/tactics.py)."""

    def __init__(self, world, pid, spawn, seed, name="Bot", team=None, level=bot_skill.DEFAULT):
        from core.input_buffer import InputBuffer
        from parkour.controller import PlayerController
        from game.bot import BotBrain
        self.clock = 0.0
        self.input = InputBuffer(time_fn=lambda: self.clock)
        self.controller = PlayerController(world.ledge_system, world.obstacles, spawn_pos=tuple(spawn), yaw=180.0)
        self.brain = BotBrain(world, self.controller, self.input, seed=seed, name=name)
        self.info = PlayerInfo(pid, name, bot=True, team=team)
        self.level = bot_skill.normalize(level)
        self.seed = seed
        self.tactician = None          # set by the server (it knows the bot's side and hive)
        self.accum = 0.0
        self.lives = 0                 # respawn count: moves computed before a respawn are stale

    @property
    def pid(self):
        return self.info.pid

    def respawn(self, spawn):
        c = self.controller
        c.spawn_pos = np.asarray(spawn, dtype=np.float32)
        c.respawn()
        self.brain.goal = None
        self.brain.route = []
        self.info.prev_pos = None
        self.lives += 1

    def step(self, dt, targets, now=None):
        """Advance at the fixed physics rate; returns the list of (prev, pos) moves made.  `now`: the
        server clock the resulting pose is stamped with (see PlayerInfo.pt)."""
        moves = []
        self.accum = min(self.accum + dt, 0.25)
        while self.accum >= config.FIXED_DT:
            self.accum -= config.FIXED_DT
            self.clock += config.FIXED_DT
            prev = self.controller.pos.copy()
            self.brain.update(config.FIXED_DT, targets)
            self.controller.step(config.FIXED_DT, self.input)
            moves.append((prev, self.controller.pos.copy()))
        c = self.controller
        info = self.info
        info.pos = c.pos.copy()
        info.yaw = float(c.yaw)
        info.pitch = float(c.pitch)
        info.st = c.fsm.state
        r = c.fsm.active_rail
        info.n = None if r is None else [round(float(r.normal[0]), 3), round(float(r.normal[2]), 3)]
        info.spd = float(np.hypot(c.vel[0], c.vel[2]))
        spin = c.fsm.spin
        info.ac = spin.to_wire() if spin is not None else None
        if now is not None:
            info.pt = float(now)
        return moves


class GameServer:
    def __init__(self, world, host="0.0.0.0", port=config.MP_DEFAULT_PORT, max_players=config.MP_MAX_PLAYERS,
                 bots=0, rules=None, seed=None, tick_rate=60.0, snapshot_rate=config.MP_SNAPSHOT_RATE,
                 next_match_timeout=config.MP_NEXT_MATCH_TIMEOUT, log=print,
                 bot=None, emblems=None, time_limit=None, bot_level=bot_skill.DEFAULT):
        self.world = world
        self.host = host
        self.port = int(port)
        self.max_players = int(max(1, min(config.MP_MAX_PLAYERS, max_players)))
        if bot is not None and not bots:
            bots = 1 if bot else 0                 # (the old on/off switch)
        self.bot_target = int(max(0, min(self.max_players - 1, bots)))
        if rules is None:
            rules = GameRules(GameRules.EMBLEMS, False,
                              time_limit if time_limit is not None else config.MP_MATCH_TIME_LIMIT,
                              emblems if emblems is not None else config.MP_DEFAULT_EMBLEMS)
        self.rules = rules
        self.seed = int(seed if seed is not None else time.time()) & 0x7FFFFFFF
        self.match = create_match(world, rules, seed=self.seed)
        self.tick_dt = 1.0 / tick_rate
        self.snapshot_dt = 1.0 / snapshot_rate
        self.next_match_timeout = float(next_match_timeout)
        self.log = log or (lambda *a: None)
        self.clients = {}             # socket -> ClientConn
        self.players = {}             # pid -> PlayerInfo (humans and bots)
        self.bots = {}                # pid -> BotPlayer
        self._next_id = 1
        self._sel = None
        self._listener = None
        self._thread = None
        self._stop = threading.Event()
        self.bot_level = bot_skill.normalize(bot_level)
        self.hives = {}               # (level, side) -> tactics.Hive shared by those bots
        self.tracker = None           # tactics.Tracker: everyone's recent movement (built with the first bot)
        self.time = 0.0
        self._t0 = time.monotonic()   # the server clock (now()) that poses and snapshots are stamped with
        self.over_since = None
        self.error = None
        self.lock = threading.RLock()  # guards the state shared with other threads (status)

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        """Bind and start serving in a background thread.  Raises OSError if the port is unavailable."""
        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind((self.host, self.port))
        ls.listen(16)
        ls.setblocking(False)
        self._listener = ls
        self.port = ls.getsockname()[1]
        self._sel = selectors.DefaultSelector()
        self._sel.register(ls, selectors.EVENT_READ, None)
        self.match.start({})
        self._fill_bots()
        self._thread = threading.Thread(target=self._run, name="hardcore-server", daemon=True)
        self._thread.start()
        self.log(f"[server] hosting on port {self.port} - {self.rules.describe()} - up to {self.max_players} players"
                 f"{f' ({self.bot_target} bots)' if self.bot_target else ''}")
        return self.port

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        for conn in list(self.clients.values()):
            try:
                conn.sock.sendall(encode({"t": "leave", "id": -1, "name": "host", "reason": "server closed"}))
            except OSError:
                pass
            self._close(conn, announce=False)
        if self._listener is not None:
            try:
                self._sel.unregister(self._listener)
            except Exception:      # noqa: BLE001
                pass
            self._listener.close()
            self._listener = None
        if self._sel is not None:
            self._sel.close()

    def now(self):
        """Server clock: seconds since the server was created (real time, unlike `time`, which is
        the match time and skips a stall).  Clients interpolate the other players on it."""
        return time.monotonic() - self._t0

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def human_count(self):
        return sum(1 for c in self.clients.values() if c.player is not None)

    @property
    def bot(self):
        """(The first bot - kept for callers that knew a single bot.)"""
        return next(iter(self.bots.values()), None)

    def status(self):
        with self.lock:
            return {"port": self.port, "humans": self.human_count, "bots": len(self.bots),
                    "bot": bool(self.bots), "bot_level": self.bot_level,
                    "max_players": self.max_players, "round": self.match.round,
                    "state": self.match.state, "rules": self.rules.to_dict(),
                    "teams": {T.RED: sum(1 for p in self.players.values() if p.team == T.RED),
                              T.BLUE: sum(1 for p in self.players.values() if p.team == T.BLUE)}}

    # ----------------------------------------------------------------------- loop
    def _run(self):
        last = time.monotonic()
        next_snap = 0.0
        try:
            while not self._stop.is_set():
                timeout = max(0.0, self.tick_dt - (time.monotonic() - last))
                for key, mask in self._sel.select(timeout):
                    if key.data is None:
                        self._accept()
                    else:
                        conn = key.data
                        if mask & selectors.EVENT_READ:
                            self._read(conn)
                        if mask & selectors.EVENT_WRITE and not conn.closing:
                            self._flush(conn)
                now = time.monotonic()
                dt = now - last
                if dt >= self.tick_dt:
                    last = now
                    self._tick(min(dt, 0.25))
                    if self.time >= next_snap:
                        next_snap = self.time + self.snapshot_dt
                        self._broadcast_snapshot()
        except Exception as exc:       # noqa: BLE001 - never take the host's game down with it
            self.error = exc
            self.log(f"[server] stopped by an error: {exc!r}")

    def _bot_roles(self):
        """Capture the flag in teams: on a side of three or more, its first bot stays back to defend."""
        roles = {}
        if self.rules.mode == GameRules.CTF and self.rules.teams:
            for team in T.TEAMS:
                members = [p for p in self.players.values() if p.team == team]
                bots = sorted(p.pid for p in members if p.bot)
                if len(members) >= 3 and bots:
                    roles[bots[0]] = "defend"
        return roles

    def _tick(self, dt):
        with self.lock:
            self.time += dt
            if self.bots:
                self.tracker.update(self.time, self.match.positions)
                roles = self._bot_roles()
                for pid, bot in list(self.bots.items()):
                    if pid not in self.bots:
                        continue
                    claimed = set()
                    if self.rules.teams:
                        for other in self.bots.values():
                            if other is not bot and other.info.team == bot.info.team and other.brain.goal is not None:
                                claimed.add(other.brain.goal.get("target_id"))
                    targets = bot.tactician.targets(self.match, self.time, bot.controller, bot.brain,
                                                    roles.get(pid, "attack"), claimed)
                    lives = bot.lives
                    for prev, pos in bot.step(dt, targets, now=self.now()):
                        self._moved(bot.info, prev, pos)
                        if self.match.state != self.match.PLAYING or bot.lives != lives:
                            break                  # round over, or sent home (tagged): the rest is stale
                    if bot.brain.wants_respawn:
                        bot.brain.wants_respawn = False
                        self._respawn_bot(bot)
            ended = self.match.tick(dt)
            self._broadcast_events(self.match.drain_events())
            self._apply_roles()
            if ended and self.over_since is None:
                self._match_over()
            if self.match.state == self.match.OVER and self.over_since is not None:
                humans = [c.player for c in self.clients.values() if c.player is not None]
                all_ready = all(p.ready for p in humans)
                if all_ready or self.time - self.over_since >= self.next_match_timeout:
                    self._start_next_match()

    # ------------------------------------------------------------------- sockets
    def _accept(self):
        try:
            sock, addr = self._listener.accept()
        except OSError:
            return
        sock.setblocking(False)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn = ClientConn(sock, addr)
        self.clients[sock] = conn
        self._sel.register(sock, selectors.EVENT_READ, conn)

    def _read(self, conn):
        try:
            data = conn.sock.recv(65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            self._close(conn)
            return
        if not data:
            self._close(conn)
            return
        try:
            msgs = conn.reader.feed(data)
        except ProtocolError:
            self._close(conn)
            return
        for msg in msgs:
            if conn.closing:
                break
            with self.lock:
                self._handle(conn, msg)

    def _send(self, conn, msg, data=None):
        if conn.closing:
            return
        conn.out.extend(data if data is not None else encode(msg))
        if len(conn.out) > MAX_OUTBUF:
            self._close(conn)
            return
        self._flush(conn)

    def _flush(self, conn):
        try:
            while conn.out:
                n = conn.sock.send(conn.out)
                if n <= 0:
                    break
                del conn.out[:n]
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:
            self._close(conn)
            return
        events = selectors.EVENT_READ | (selectors.EVENT_WRITE if conn.out else 0)
        try:
            self._sel.modify(conn.sock, events, conn)
        except (KeyError, ValueError, OSError):
            pass

    def _broadcast(self, msg, skip=None):
        data = encode(msg)                    # once, however many players there are
        for conn in list(self.clients.values()):
            if conn.player is not None and conn is not skip:
                self._send(conn, msg, data)

    def _conn_of(self, pid):
        for conn in self.clients.values():
            if conn.player is not None and conn.player.pid == pid:
                return conn
        return None

    def _close(self, conn, announce=True):
        if conn.closing:
            return
        conn.closing = True
        try:
            self._sel.unregister(conn.sock)
        except Exception:              # noqa: BLE001
            pass
        try:
            conn.sock.close()
        except OSError:
            pass
        self.clients.pop(conn.sock, None)
        if conn.player is not None:
            pid = conn.player.pid
            self.players.pop(pid, None)
            events = self.match.remove_player(pid)
            self.log(f"[server] {conn.player.name} left")
            if announce:
                self._broadcast({"t": "leave", "id": pid, "name": conn.player.name, "team": conn.player.team})
                self._broadcast_events(events)
            if not self._stop.is_set():
                self._fill_bots()
                self._balance_bots()

    # --------------------------------------------------------------------- bots
    def _bot_name(self):
        taken = {p.name for p in self.players.values()}
        for n in BOT_NAMES:
            if f"{n} (bot)" not in taken:
                return f"{n} (bot)"
        return self._unique_name("Bot")

    def _add_bot(self, team=None):
        pid = self._next_id
        self._next_id += 1
        if self.rules.teams:
            team = team if T.valid(team) else T.smaller_team(self._team_counts())
        else:
            team = None
        name = self._bot_name()
        self.match.add_player(pid, team)
        self._broadcast_events(self.match.drain_events())
        spawn = self.match.spawn_point(pid)
        bot = BotPlayer(self.world, pid, spawn, seed=self.seed + pid, name=name, team=team, level=self.bot_level)
        self._think(bot)
        self.bots[pid] = bot
        self.players[pid] = bot.info
        self._broadcast({"t": "join", "id": pid, "name": name, "bot": True, "team": team})
        return bot

    # ------------------------------------------------------------------ bot minds
    def _hive(self, level, team):
        from game.tactics import Hive, search_grid
        key = (level, team if self.rules.teams else None)
        hive = self.hives.get(key)
        if hive is None:
            hive = self.hives[key] = Hive(search_grid(self.world), bot_skill.skill(level))
        return hive

    def _think(self, bot):
        """(Re)build a bot's tactician for its level and side."""
        from game.tactics import Tracker, Tactician
        if self.tracker is None:
            self.tracker = Tracker()
        bot.tactician = Tactician(bot.pid, bot.level, self.world, self._hive(bot.level, bot.info.team),
                                  self.tracker, seed=bot.seed)

    def set_bot_level(self, level, pid=None):
        """The intelligence of every bot (or of one: mixed levels, for tests and experiments)."""
        level = bot_skill.normalize(level)
        with self.lock:
            if pid is None:
                self.bot_level = level
            for p, bot in self.bots.items():
                if pid is None or p == pid:
                    bot.level = level
                    self._think(bot)

    def _remove_bot(self, pid):
        bot = self.bots.pop(pid, None)
        if bot is None:
            return
        self.players.pop(pid, None)
        events = self.match.remove_player(pid)
        self._broadcast({"t": "leave", "id": pid, "name": bot.info.name, "bot": True, "team": bot.info.team})
        self._broadcast_events(events)

    def _fill_bots(self):
        """Bring the bots up to the host's number, as far as the free slots allow."""
        want = min(self.bot_target, self.max_players - self.human_count)
        while len(self.bots) > max(0, want):
            self._remove_bot(max(self.bots))
        while len(self.bots) < want:
            self._add_bot()

    def _make_room(self, team=None):
        """A person wants in and every slot is taken: a bot gives its place up (one from the side the
        person is joining, if any, so the teams stay even)."""
        if self.human_count + len(self.bots) < self.max_players or not self.bots:
            return
        same = [pid for pid, b in self.bots.items() if b.info.team == team]
        self._remove_bot(max(same) if same else max(self.bots))

    def _team_counts(self):
        counts = {T.RED: 0, T.BLUE: 0}
        for p in self.players.values():
            if p.team in counts:
                counts[p.team] += 1
        return counts

    def _balance_bots(self):
        """Teams games: bots change sides so the teams differ by one player at most."""
        if not self.rules.teams:
            return
        for _ in range(len(self.bots)):
            counts = self._team_counts()
            big = T.RED if counts[T.RED] > counts[T.BLUE] else T.BLUE
            if counts[big] - counts[T.other(big)] <= 1:
                return
            movers = sorted(pid for pid, b in self.bots.items() if b.info.team == big)
            if not movers:
                return
            self._switch_team(self.players[movers[-1]], T.other(big))

    def _send_home(self, pid):
        """A flag carrier who was tagged starts again from their own side."""
        player = self.players.get(pid)
        if player is None:
            return
        spawn = self.match.spawn_point(pid)
        if player.bot:
            bot = self.bots.get(pid)
            if bot is not None:
                bot.respawn(spawn)
        else:
            conn = self._conn_of(pid)
            if conn is not None:
                self._send(conn, {"t": "respawn", "spawn": spawn, "team": player.team, "reason": "tagged"})
        player.prev_pos = None

    def _apply_roles(self):
        """Zombie tag: the bots that are zombies run and climb slower (people's games do the same)."""
        zombie_game = self.rules.mode == GameRules.ZOMBIE
        living = getattr(self.match, "living", None)
        for pid, bot in self.bots.items():
            if zombie_game and pid != living:
                bot.controller.set_speed_scales(config.ZOMBIE_RUN_SCALE, config.ZOMBIE_CLIMB_SCALE)
            else:
                bot.controller.set_speed_scales(1.0, 1.0)

    def _respawn_bot(self, bot):
        self._broadcast_events(self.match._player_gone(bot.pid, "respawn"))
        bot.respawn(self.match.spawn_point(bot.pid))

    # ------------------------------------------------------------------ messages
    def _handle(self, conn, msg):
        t = msg["t"]
        if conn.player is None:
            if t != "hello":
                self._close(conn)
                return
            if msg.get("v") != PROTOCOL_VERSION:
                self._send(conn, {"t": "reject", "reason": f"version mismatch (server {PROTOCOL_VERSION})"})
                self._flush(conn)
                self._close(conn, announce=False)
                return
            if self.human_count >= self.max_players:
                self._send(conn, {"t": "reject", "reason": f"server full ({self.max_players} players)"})
                self._flush(conn)
                self._close(conn, announce=False)
                return
            want = msg.get("team")
            self._make_room(want if self.rules.teams else None)
            team = None
            if self.rules.teams:
                team = want if T.valid(want) else T.smaller_team(self._team_counts())
            name = self._unique_name(clean_name(msg.get("name")))
            pid = self._next_id
            self._next_id += 1
            player = PlayerInfo(pid, name, team=team)
            conn.player = player
            self.players[pid] = player
            self.match.add_player(pid, team)
            spawn = self.match.spawn_point(pid)
            self._send(conn, {"t": "welcome", "id": pid, "v": PROTOCOL_VERSION, "team": team,
                              "server": {"max_players": self.max_players, "bots": len(self.bots),
                                         "bot": bool(self.bots), "rules": self.rules.to_dict(),
                                         "emblems": self.rules.emblems},
                              "match": self.match.to_dict(),
                              "players": [p.to_dict(self.match.scores.get(p.pid, 0)) for p in self.players.values()],
                              "spawn": spawn})
            self._broadcast({"t": "join", "id": pid, "name": name, "bot": False, "team": team}, skip=conn)
            self._broadcast_events(self.match.drain_events())
            self.log(f"[server] {name} joined{f' the {T.name(team)} team' if team else ''} "
                     f"({self.human_count}/{self.max_players}, {len(self.bots)} bots)")
            self._balance_bots()
            return
        player = conn.player
        if t == "state":
            p = finite_vec(msg.get("p"))
            if p is None:
                return
            pos = np.asarray(p, dtype=np.float32)
            player.yaw = finite(msg.get("yaw"))
            player.pitch = finite(msg.get("pitch"))
            st = msg.get("st")
            player.st = st if isinstance(st, str) and len(st) < 24 else "GROUND"
            player.n = finite_vec(msg.get("n"), 2, limit=2.0)
            player.spd = finite(msg.get("spd"), limit=100.0)
            player.ac = clean_acro(msg.get("ac"))
            player.pt = self.now()             # the pose as of its arrival: one clock for everyone's poses
            self._moved(player, player.pos, pos)
        elif t == "team":
            if self.rules.teams and T.valid(msg.get("team")) and msg["team"] != player.team:
                self._switch_team(player, msg["team"])
                self._balance_bots()
        elif t == "ready":
            player.ready = True
            if self.match.state == self.match.OVER:
                self._broadcast(self._ready_status())
        elif t == "bye":
            self._close(conn)

    def _switch_team(self, player, team):
        events = self.match.set_team(player.pid, team)
        player.team = team
        self._broadcast({"t": "team", "id": player.pid, "name": player.name, "team": team, "bot": player.bot})
        self._broadcast_events(events)
        spawn = self.match.spawn_point(player.pid)
        if player.bot:
            bot = self.bots.get(player.pid)
            if bot is not None:
                bot.respawn(spawn)
                self._think(bot)                    # its side's hive now
        else:
            conn = self._conn_of(player.pid)
            if conn is not None:
                self._send(conn, {"t": "respawn", "spawn": spawn, "team": team})
        player.prev_pos = None
        player.pos = None

    def _unique_name(self, name):
        taken = {p.name for p in self.players.values()}
        if name not in taken:
            return name
        for i in range(2, 100):
            if f"{name} {i}" not in taken:
                return f"{name} {i}"
        return name

    def _broadcast_events(self, events):
        for ev in events:
            for key, out in (("player", "name"), ("by", "by_name"), ("from", "from_name")):
                pid = ev.get(key)
                if pid is not None and pid in self.players:
                    ev[out] = self.players[pid].name
            self._broadcast(ev)
            if ev.get("t") == "flag" and ev.get("action") == "returned" and ev.get("reason") == "tagged":
                self._send_home(ev.get("player"))          # tagged with the flag: back to your base
            taken = ev.get("emblem") if ev.get("t") == "pickup" else None
            for bot in self.bots.values():
                if taken is not None:
                    # the bot that took it knows; the others only notice when they see the spot empty
                    if ev.get("player") == bot.pid:
                        bot.brain.set_emblems_taken({taken})
                elif ev.get("t") in ("flag", "zombie"):
                    bot.brain.goal = None            # the objectives changed: think again

    def _moved(self, player, prev, pos):
        player.prev_pos = player.pos
        player.pos = np.asarray(pos, dtype=np.float32)
        if player.pid not in self.match.players:
            return
        self._broadcast_events(self.match.apply_move(player.pid, prev, player.pos))
        if self.match.state == self.match.OVER and self.over_since is None:
            self._match_over()

    # -------------------------------------------------------------------- rounds
    def _match_over(self):
        self.over_since = self.time
        for p in self.players.values():
            p.ready = p.bot
        names = {str(pid): p.name for pid, p in self.players.items()}
        msg = {"t": "over", "names": names, "timeout": self.next_match_timeout}
        msg.update(self.match.results())
        self._broadcast(msg)
        if self.match.winning_team:
            self.log(f"[server] match over - the {T.name(self.match.winning_team)} team won")
        elif self.match.winners:
            self.log("[server] match over - " + ", ".join(names.get(str(w), "?") for w in self.match.winners) + " won")

    def _ready_status(self):
        humans = [c.player for c in self.clients.values() if c.player is not None]
        return {"t": "ready_status", "ready": [p.pid for p in humans if p.ready], "needed": len(humans),
                "timeout": max(0.0, self.next_match_timeout - (self.time - (self.over_since or self.time)))}

    def _start_next_match(self):
        self.over_since = None
        self.match.start({pid: p.team for pid, p in self.players.items()})
        self.match.drain_events()                    # (the start message carries the new round's state)
        self._apply_roles()
        for p in self.players.values():
            p.ready = False
            p.prev_pos = None
            p.pos = None
        for hive in self.hives.values():
            hive.reset()
        for bot in self.bots.values():
            bot.respawn(self.match.spawn_point(bot.pid))
            if bot.tactician is not None:
                bot.tactician.reset()
        for conn in list(self.clients.values()):
            if conn.player is not None:
                self._send(conn, {"t": "start", "match": self.match.to_dict(), "team": conn.player.team,
                                  "spawn": self.match.spawn_point(conn.player.pid)})
        self.log(f"[server] round {self.match.round} started")

    # ------------------------------------------------------------------ snapshots
    def _broadcast_snapshot(self):
        with self.lock:
            players = [p.to_dict(self.match.scores.get(p.pid, 0)) for p in self.players.values() if p.pos is not None]
            msg = {"t": "snap", "time": round(self.time, 3), "now": round(self.now(), 4), "players": players,
                   "match": self.match.snapshot(),
                   "time_left": None if self.match.time_left is None else round(self.match.time_left, 1),
                   "state": self.match.state}
            if hasattr(self.match, "remaining"):
                msg["remaining"] = self.match.remaining
        self._broadcast(msg)
