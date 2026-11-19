"""The game's side of a multiplayer session (no rendering): host or join, sync, rounds.

Hosting starts a GameServer inside this process and joins it over the loopback,
so the host plays exactly like everyone else.  Every frame `update()` sends the
local player's pose (MP_STATE_RATE times a second), applies what the server said
(spawn points at the start of a round, pickups, flags, team changes, results) and
returns short notices for the HUD.
"""
import time

import numpy as np

import config
from game import teams as T
from game import bot_skill
from game.match import GameRules
from net.client import GameClient
from net.server import GameServer
from parkour.state_machine import ParkourState


def minutes_choices():
    return tuple(config.MP_MATCH_MINUTES)


class HostSettings:
    def __init__(self, port=config.MP_DEFAULT_PORT, max_players=config.MP_MAX_PLAYERS, bots=config.MP_DEFAULT_BOTS,
                 emblems=config.MP_DEFAULT_EMBLEMS, mode=GameRules.EMBLEMS, teams=False,
                 minutes=config.MP_MATCH_TIME_LIMIT / 60.0, capture_limit=config.MP_DEFAULT_CAPTURES, bot=None,
                 goal=config.ZOMBIE_DEFAULT_GOAL, bot_level=bot_skill.DEFAULT):
        self.port = int(port)
        self.max_players = int(max(2, min(config.MP_MAX_PLAYERS, max_players)))
        if bot is not None:                       # (the old on/off switch)
            bots = 1 if bot else 0
        self.bots = int(max(0, min(self.max_players - 1, int(bots))))
        lo, hi = config.MP_EMBLEM_RANGE
        self.emblems = int(max(lo, min(hi, emblems)))
        self.mode = mode if mode in GameRules.MODES else GameRules.EMBLEMS
        self.teams = bool(teams)
        m = float(minutes)
        self.minutes = 0.0 if not (m > 0.0) else float(max(1.0, min(60.0, m)))     # 0 = no time limit
        self.capture_limit = int(max(0, min(config.MP_CAPTURE_LIMIT_MAX, int(capture_limit))))
        glo, ghi = config.ZOMBIE_GOAL_RANGE
        self.goal = int(max(glo, min(ghi, int(goal))))
        self.bot_level = bot_skill.normalize(bot_level)      # the bots' intelligence (game/bot_skill.py)

    @property
    def bot(self):
        return self.bots > 0

    def rules(self):
        return GameRules(self.mode, self.teams, self.minutes * 60.0, self.emblems, self.capture_limit, self.goal)


def parse_address(text, default_port=config.MP_DEFAULT_PORT):
    """'host', 'host:port' or '[v6]:port' -> (host, port).  Raises ValueError."""
    text = (text or "").strip()
    if not text:
        raise ValueError("no address")
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif text.count(":") == 1:
        host, port = text.split(":")
    else:
        host, port = text, ""
    port = int(port) if port else default_port
    if not host or not (1 <= port <= 65535):
        raise ValueError(f"bad address '{text}'")
    return host, port


class MultiplayerSession:
    def __init__(self, client, server=None):
        self.client = client
        self.server = server
        self.send_timer = 0.0
        self.results = None          # {"winners", "scores", "names", "ready", "needed", "timeout", "joined", ...}
        self.ended = None            # reason, once the session is over
        self.round = client.match.get("round", 1)

    # ------------------------------------------------------------ construction
    @classmethod
    def host(cls, world, settings, name, log=print, team=None):
        server = GameServer(world, port=settings.port, max_players=settings.max_players, bots=settings.bots,
                            rules=settings.rules(), log=log, bot_level=settings.bot_level)
        server.start()
        try:
            client = GameClient("127.0.0.1", server.port, name, team=team).connect()
        except ConnectionError:
            server.stop()
            raise
        return cls(client, server)

    @classmethod
    def start_host(cls, world_spec, settings, name, team=None, log=print, extra_paths=()):
        """Host with the server in a process of its own (what the game does): returns a HostLauncher
        at once - poll() it each frame until the session is up (building the level takes a while)."""
        from net.hosted import ServerProcess
        proc = ServerProcess(world_spec, settings.port, settings.max_players, settings.bots, settings.rules(),
                             extra_paths=extra_paths, log=log, bot_level=settings.bot_level).start()
        return HostLauncher(proc, name, team)

    @classmethod
    def join(cls, address, name, team=None):
        host, port = parse_address(address)
        return cls(GameClient(host, port, name, team=team).connect())

    @property
    def hosting(self):
        return self.server is not None

    @property
    def rules(self):
        return GameRules.from_dict(self.client.rules)

    @property
    def team(self):
        return self.client.team

    def describe(self):
        if self.server is not None:
            st = self.server.status()
            level = bot_skill.LABELS.get(st.get("bot_level"), "")
            bots = (f" + {st['bots']} {level.lower() + ' ' if level else ''}bot{'s' if st['bots'] != 1 else ''}"
                    if st["bots"] else "")
            people = f"{st['humans']} player{'s' if st['humans'] != 1 else ''}"
            return f"Hosting on port {st['port']} - {people}{bots} ({st['max_players']} max)"
        return f"Connected to {self.client.host}:{self.client.port}"

    def describe_game(self):
        r = self.rules
        text = r.describe()
        if r.teams and T.valid(self.team):
            text += f"  -  you are on the {T.name(self.team)} team"
        return text

    # ------------------------------------------------------------------- frame
    def _spawn(self, player, spawn, turn=True):
        if spawn is None:
            return
        player.spawn_pos = np.asarray(spawn, dtype=np.float32)
        player.respawn()
        player.respawn_count -= 1                 # a new round is not a fall off the map
        if turn:
            player.yaw = 180.0
        player.fsm.state = ParkourState.GROUND

    def _name(self, ev, key="name", pid_key="player"):
        if ev.get(pid_key) == self.client.id:
            return "You"
        return ev.get(key, "?")

    def _flag_notice(self, ev):
        c = self.client
        action, flag = ev.get("action"), ev.get("flag")
        who = self._name(ev)
        mine = ev.get("player") == c.id
        if flag in T.TEAMS:
            fname = f"the {T.name(flag)} flag"
            if c.teams_game and flag == c.team:
                fname = "your flag"
        else:
            fname = "the flag"
        if action == "taken":
            return f"{who} {'have' if mine else 'has'} {fname}!" + ("  Bring it home!" if mine and flag in T.TEAMS else
                                                                   "  Run it to the drop zone!" if mine else "")
        if action == "stolen":
            thief = who
            return f"{thief} stole the flag from {self._name(ev, 'from_name', 'from')}!"
        if action == "captured":
            if c.teams_game:
                ts = ev.get("team_scores", c.team_scores)
                score = f"  ({T.name(T.RED)} {ts.get(T.RED, 0)} - {ts.get(T.BLUE, 0)} {T.name(T.BLUE)})"
                team = ev.get("team")
                return f"{who} captured {fname} for the {T.name(team)} team!{score}"
            return f"{who} captured the flag! ({ev.get('score', 0)})  -  a new flag is out"
        if action == "returned":
            reason = ev.get("reason")
            if reason == "tagged":
                by = self._name(ev, "by_name", "by")
                return f"{by} tagged {who.lower() if who == 'You' else who} - {fname} goes home"
            if flag in T.TEAMS:
                return f"{fname[0].upper() + fname[1:]} is back at its base"
            return "The flag is back on its stand"
        return None

    def update(self, dt, player):
        """Send our pose, apply the server's news.  Returns [(kind, text)] notices for the HUD."""
        notices = []
        c = self.client
        if c.connected:
            self.send_timer -= dt
            if self.send_timer <= 0.0:
                # (keep the cadence: the overshoot counts toward the next one, so 30 Hz at 60 fps is
                # every other frame, not every third)
                self.send_timer = max(0.0, self.send_timer + 1.0 / config.MP_STATE_RATE)
                rail = player.fsm.active_rail
                spin = player.fsm.spin
                c.send_state(player.pos, player.yaw, player.pitch, player.fsm.state,
                             None if rail is None else rail.normal,
                             float(np.hypot(player.vel[0], player.vel[2])),
                             spin.to_wire() if spin is not None else None)
        # zombie tag: zombies run and climb a little slower than the living player
        if c.living is not None and c.living != c.id:
            player.set_speed_scales(config.ZOMBIE_RUN_SCALE, config.ZOMBIE_CLIMB_SCALE)
        else:
            player.set_speed_scales(1.0, 1.0)
        for ev in c.poll_events():
            t = ev.get("t")
            if t == "welcome":
                self._spawn(player, ev.get("spawn"))
                rules = self.rules
                goal = {GameRules.EMBLEMS: "find the glowing emblems!",
                        GameRules.ZOMBIE: "zombie tag! " + self._role_line()}.get(
                    rules.mode, "capture the enemy flag!" if rules.teams else "grab the flag and run it to the drop zone!")
                team = f"  You are on the {T.name(c.team)} team." if rules.teams and T.valid(c.team) else ""
                notices.append(("info", f"Joined - {goal}{team}"))
                if c.match.get("state") == "over":
                    self.results = {"winners": [], "scores": {}, "names": {}, "ready": [], "needed": 0,
                                    "timeout": 0.0, "joined": False, "between": True}
            elif t == "start":
                self.results = None
                self.round = c.match.get("round", self.round + 1)
                self._spawn(player, ev.get("spawn"))
                r = self.rules
                what = {GameRules.EMBLEMS: f"{len(c.emblems)} emblems hidden around the city",
                        GameRules.ZOMBIE: "zombie tag! " + self._role_line()}.get(r.mode, "capture the flag!")
                notices.append(("info", f"Round {self.round} - {what}"))
            elif t == "respawn":
                self._spawn(player, ev.get("spawn"), turn=False)
                if ev.get("reason") == "tagged":
                    notices.append(("flag", "Tagged! The flag went home - and so did you"))
            elif t == "pickup":
                who = self._name(ev)
                if c.teams_game:
                    ts = ev.get("team_scores", {})
                    notices.append(("pickup", f"{who} picked up an emblem for {T.name(ev.get('team'))}  "
                                              f"({T.name(T.RED)} {ts.get(T.RED, 0)} - {ts.get(T.BLUE, 0)} {T.name(T.BLUE)})"
                                              f"  -  {ev.get('remaining', 0)} left"))
                else:
                    notices.append(("pickup", f"{who} picked up an emblem ({ev.get('score', 0)})  -  {ev.get('remaining', 0)} left"))
            elif t == "flag":
                text = self._flag_notice(ev)
                if text:
                    notices.append(("flag", text))
            elif t == "zombie":
                text = self._zombie_notice(ev)
                if text:
                    notices.append(("zombie", text))
            elif t == "team":
                if ev.get("id") == c.id:
                    notices.append(("info", f"You are on the {T.name(ev.get('team'))} team now"))
                elif not ev.get("bot"):
                    notices.append(("info", f"{ev.get('name', '?')} joined the {T.name(ev.get('team'))} team"))
            elif t == "join":
                team = f" ({T.name(ev.get('team'))})" if T.valid(ev.get("team")) else ""
                notices.append(("info", f"{ev.get('name', '?')} joined{team}"))
            elif t == "leave":
                notices.append(("info", f"{ev.get('name', '?')} left"))
            elif t == "over":
                self.results = {"winners": ev.get("winners", []), "scores": ev.get("scores", {}),
                                "names": ev.get("names", {}), "team": ev.get("team"), "draw": ev.get("draw", False),
                                "team_scores": ev.get("team_scores", {}), "teams": ev.get("teams", {}),
                                "teams_game": ev.get("teams_game", False), "mode": ev.get("mode", "emblems"),
                                "ready": [], "needed": 0,
                                "timeout": ev.get("timeout", config.MP_NEXT_MATCH_TIMEOUT), "joined": False}
            elif t == "ready_status" and self.results is not None:
                self.results.update(ready=ev.get("ready", []), needed=ev.get("needed", 0), timeout=ev.get("timeout", 0.0))
            elif t == "disconnected":
                self.ended = ev.get("reason", "disconnected")
                notices.append(("info", f"Multiplayer ended: {self.ended}"))
        if self.results is not None:
            self.results["timeout"] = max(0.0, self.results.get("timeout", 0.0) - dt)
        return notices

    def _player_name(self, pid):
        rp = self.client.players.get(pid)
        return rp.name if rp is not None else "?"

    def _role_line(self):
        c = self.client
        if c.living is None:
            return ""
        if c.living == c.id:
            return "You are ALIVE - run, and stay away from the zombies!"
        return f"{self._player_name(c.living)} is alive - catch them!"

    def _zombie_notice(self, ev):
        c = self.client
        action = ev.get("action")
        if action == "caught":
            catcher, caught = ev.get("player"), ev.get("from")
            if catcher == c.id:
                return f"You caught {ev.get('from_name', '?')} - you're ALIVE! Run!"
            if caught == c.id:
                return f"{ev.get('name', '?')} caught you - you're a zombie now! (no tag-backs for {int(config.ZOMBIE_TAGBACK_GRACE)} s)"
            return f"{ev.get('name', '?')} caught {ev.get('from_name', '?')} - {ev.get('name', '?')} is alive now"
        if action == "living":
            if ev.get("reason") == "start":
                return None                            # (the round-start notice says it)
            if ev.get("player") == c.id:
                return "You are ALIVE - run, and stay away from the zombies!"
            return f"{ev.get('name', '?')} is alive now - catch them!"
        return None

    def switch_team(self, team):
        """Ask to play for the other side (teams games)."""
        if self.client.teams_game and T.valid(team) and team != self.client.team:
            self.client.send_team(team)
            return True
        return False

    # ----------------------------------------------------------------- results
    def join_next(self):
        """'Join new game' on the results screen."""
        if self.results is not None and not self.results.get("joined"):
            self.results["joined"] = True
            self.client.send_ready()

    def winner_text(self):
        r = self.results or {}
        names = r.get("names", {})
        winners = r.get("winners", [])
        scores = r.get("scores", {})
        me = self.client.id
        word = {GameRules.CTF: "capture", GameRules.ZOMBIE: "point"}.get(r.get("mode"), "emblem")

        def n(k):
            return f"{k} {word}{'s' if k != 1 else ''}"
        if r.get("teams_game"):
            ts = r.get("team_scores", {})
            red, blue = ts.get(T.RED, 0), ts.get(T.BLUE, 0)
            if r.get("team") in T.TEAMS:
                won = r["team"]
                mine = " - your team wins!" if self.client.team == won else ""
                return f"The {T.name(won)} team wins {max(red, blue)} - {min(red, blue)}{mine}"
            if r.get("draw"):
                return f"Draw! {red} - {red}"
            return "Nobody scored"
        if not winners:
            return "Nobody scored" if word == "capture" else "Nobody picked up an emblem"
        best = scores.get(str(winners[0]), 0)
        if len(winners) > 1:
            who = " and ".join("You" if w == me else names.get(str(w), "?") for w in winners)
            return f"Draw! {who} - {n(best)} each"
        w = winners[0]
        if w == me:
            return f"You win with {n(best)}!"
        return f"{names.get(str(w), '?')} wins with {n(best)}"

    def results_view(self):
        """The results screen: title, winner line, one line per player (text, is_me, team), waiting line."""
        r = self.results or {}
        c = self.client
        names = r.get("names", {})
        teams_of = r.get("teams", {})
        rows = []
        for k, v in r.get("scores", {}).items():
            rp = c.players.get(int(k))
            name = rp.name if rp is not None else names.get(k, "?")
            team = teams_of.get(k) if r.get("teams_game") else None
            rows.append((int(v), name, int(k) == c.id, team))
        if r.get("teams_game"):
            order = {r.get("team") or T.RED: 0}
            rows.sort(key=lambda t: (order.get(t[3], 1), -t[0], t[1]))
        else:
            rows.sort(key=lambda t: (-t[0], t[1]))
        lines = []
        for score, name, me, team in rows:
            tag = f"[{T.name(team)}]  " if team else ""
            lines.append((f"{tag}{name}{'  (you)' if me else ''}   {score}", me, team))
        needed = r.get("needed", 0)
        waiting = (f"Joined - waiting for players ({len(r.get('ready', []))}/{needed})  -  next game in {int(r.get('timeout', 0))} s"
                   if r.get("joined") else f"Next game starts in {int(r.get('timeout', 0))} s")
        return {"title": "ROUND OVER", "winner": self.winner_text(), "lines": lines, "waiting": waiting,
                "team": r.get("team")}

    # ------------------------------------------------------------------- views
    def remote_poses(self):
        return self.client.remote_poses()

    def emblems(self):
        return self.client.untaken_emblems()

    def objectives(self):
        return self.client.objectives

    def scoreboard(self):
        return self.client.scoreboard()

    def hud_state(self, player_pos=None):
        """What the HUD shows (mode, scores, time, the flags) - see render/mp_render.py."""
        c = self.client
        carrying = None
        for f in c.objectives.get("flags", []):
            if f.get("carrier") == c.id:
                carrying = f.get("key")
        hud = {"mode": c.mode, "teams": c.teams_game, "team": c.team, "remaining": c.remaining,
               "scores": self.scoreboard(), "team_scores": dict(c.team_scores), "time_left": c.time_left,
               "objectives": c.objectives, "carrying": carrying, "me": c.id,
               "capture_limit": c.rules.get("capture_limit", 0), "chase": []}
        if c.mode == GameRules.ZOMBIE:
            z = c.zombie
            living = z.get("living")
            me_alive = living == c.id
            hud["zombie"] = {"living": living, "living_name": self._player_name(living) if living is not None else "",
                             "alive": me_alive, "grace": float(z.get("grace", 0.0) or 0.0),
                             "points": float((z.get("points") or {}).get(str(c.id), 0.0)),
                             "goal": int(z.get("goal", c.rules.get("goal", config.ZOMBIE_DEFAULT_GOAL))),
                             "no_tagback": (float(z["no_tagback"][1]) if z.get("no_tagback") and
                                            z["no_tagback"][0] == c.id else 0.0)}
            poses = self.remote_poses()
            if me_alive and player_pos is not None:
                # the nearest zombies closing in (within 40 m)
                near = sorted(((float(np.linalg.norm(np.asarray(q["p"]) - player_pos)), rp, q) for rp, q in poses),
                              key=lambda t: t[0])
                hud["chase"] = [{"pos": q["p"], "label": "Zombie", "kind": "zombie"} for d, rp, q in near[:3] if d < 40.0]
            elif living is not None:
                for rp, q in poses:
                    if rp.pid == living:
                        hud["chase"] = [{"pos": q["p"], "label": f"ALIVE: {rp.name}", "kind": "living"}]
        return hud

    def leave(self):
        self.client.close()
        if self.server is not None:
            self.server.stop()
            self.server = None
        if self.ended is None:
            self.ended = "left the game"


class HostLauncher:
    """A server process on its way up.  poll(): None while it starts, then the MultiplayerSession
    (joined as the host); raises ConnectionError if the server could not start or be joined."""

    def __init__(self, proc, name, team=None):
        self.proc = proc
        self.name = name
        self.team = team
        self.session = None

    def poll(self):
        if self.session is not None:
            return self.session
        try:
            if self.proc.poll() is None:
                return None
        except OSError as exc:
            self.proc.stop()
            raise ConnectionError(str(exc)) from exc
        try:
            client = GameClient("127.0.0.1", self.proc.port, self.name, team=self.team).connect()
        except ConnectionError:
            self.proc.stop()
            raise
        self.proc.wait_status(lambda st: st.get("humans", 0) >= 1)      # (so the first notice counts us)
        self.session = MultiplayerSession(client, self.proc)
        return self.session

    def wait(self, timeout=90.0):
        """Block until the session is up (tests, the command line)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            s = self.poll()
            if s is not None:
                return s
            time.sleep(0.05)
        self.cancel()
        raise ConnectionError("the server did not start in time")

    @property
    def elapsed(self):
        return time.monotonic() - (self.proc.started_at or time.monotonic())

    def cancel(self):
        self.proc.stop()


def parse_mp_args(argv):
    """Multiplayer command-line options:
        --host [--port N] [--max-players N] [--bots N | --bot | --no-bot] [--bot-level simple|complex|hard]
               [--mode emblems|ctf|zombie] [--teams | --ffa] [--time MINUTES (0: no limit)] [--emblems N]
               [--captures N] [--goal POINTS]
        --join ADDRESS        (host or host:port)
        --name NAME  --team red|blue
    Returns a dict of the options given (missing ones are absent)."""
    out = {}
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        nxt = args[i + 1] if i + 1 < len(args) else None
        if a == "--host":
            out["host"] = True
        elif a == "--join" and nxt is not None:
            out["join"] = nxt
            i += 1
        elif a == "--bot":
            out["bots"] = 1
        elif a == "--no-bot":
            out["bots"] = 0
        elif a == "--teams":
            out["teams"] = True
        elif a == "--ffa":
            out["teams"] = False
        elif a in ("--port", "--max-players", "--emblems", "--bots", "--captures", "--time", "--goal") \
                and nxt is not None and nxt.isdigit():
            key = {"--captures": "capture_limit", "--time": "minutes"}.get(a, a[2:].replace("-", "_"))
            out[key] = int(nxt)
            i += 1
        elif a == "--bot-level" and nxt is not None and nxt.lower() in bot_skill.LEVELS:
            out["bot_level"] = nxt.lower()
            i += 1
        elif a == "--mode" and nxt is not None and nxt.lower() in GameRules.MODES:
            out["mode"] = nxt.lower()
            i += 1
        elif a == "--team" and nxt is not None and nxt.lower() in T.TEAMS:
            out["team"] = nxt.lower()
            i += 1
        elif a == "--name" and nxt is not None:
            out["name"] = nxt
            i += 1
        i += 1
    return out
