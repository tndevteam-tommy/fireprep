"""The hosted game server in a process of its own.

Bots are real players run at the physics rate - a dozen of them keep a CPU core
busy - so when you host from the game the server runs in a child process: the
bots never take time from your frames, and a crash in the server cannot take the
game down.  The child builds its own copy of the level (the city is generated
from its seed, so it is the same city) and serves on the port; the game joins it
over the loopback like everyone else.

Parent <-> child: the child's stdout carries JSON lines ({"t": "ready", "port"},
{"t": "error", "reason"}, {"t": "status", ...} four times a second); its stdin takes
"stop".  Server logs go to stderr.  When the game goes away the child's stdin
closes and it shuts down on its own.

Run it by hand for a dedicated (headless) server:
    python -m net.hosted --port 24950 --bots 6 --mode ctf --teams --time 10
"""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ServerProcess:
    """A GameServer in a child process.  start() returns at once; `ready` / `error` tell how it went."""

    def __init__(self, world_spec, port, max_players, bots, rules, seed=None, next_match_timeout=None,
                 extra_paths=(), log=print, bot_level=None):
        self.spec = {"world": list(world_spec), "port": int(port), "max_players": int(max_players),
                     "bots": int(bots), "rules": rules.to_dict(), "seed": seed, "paths": list(extra_paths)}
        if bot_level is not None:
            self.spec["bot_level"] = str(bot_level)
        if next_match_timeout is not None:
            self.spec["next_match_timeout"] = float(next_match_timeout)
        self.log = log or (lambda *a: None)
        self.proc = None
        self.port = int(port)
        self.ready = threading.Event()
        self.error = None
        self._status = {}
        self._reader = None
        self.started_at = None

    # ---------------------------------------------------------------- control
    def start(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "net.hosted", json.dumps(self.spec)], cwd=ROOT,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env,
                                     text=True, bufsize=1)
        self.started_at = time.monotonic()
        self._reader = threading.Thread(target=self._read, name="hardcore-server-pipe", daemon=True)
        self._reader.start()
        return self

    def _read(self):
        try:
            for line in self.proc.stdout:
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                t = msg.get("t")
                if t == "ready":
                    self.port = int(msg.get("port", self.port))
                    self.ready.set()
                elif t == "error":
                    self.error = str(msg.get("reason", "the server failed"))
                    self.ready.set()
                elif t == "status":
                    self._status = msg.get("status", {})
        except (OSError, ValueError):
            pass
        if not self.ready.is_set():
            self.error = self.error or "the server process ended"
            self.ready.set()

    def wait(self, timeout=60.0):
        """Block until the server listens (returns the port) or failed (raises OSError)."""
        if not self.ready.wait(timeout):
            self.stop()
            raise OSError("the server did not start in time")
        if self.error:
            raise OSError(self.error)
        return self.port

    def poll(self):
        """None while starting, True once listening; raises OSError if it failed."""
        if not self.ready.is_set():
            if self.proc is not None and self.proc.poll() is not None:
                time.sleep(0.05)                  # let the reader catch the child's last words
                if not self.ready.is_set():
                    self.error = self.error or "the server process ended"
                    self.ready.set()
            else:
                return None
        if self.error:
            raise OSError(self.error)
        return True

    def wait_status(self, pred, timeout=1.5):
        """Wait (briefly) for a status report that satisfies pred (e.g. our own join has counted)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._status and pred(self._status):
                return True
            time.sleep(0.02)
        return False

    def status(self):
        st = dict(self._status) if self._status else {}
        st.setdefault("port", self.port)
        st.setdefault("humans", 0)
        st.setdefault("bots", 0)
        st.setdefault("bot", False)
        st.setdefault("max_players", self.spec["max_players"])
        return st

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("stop\n")
                self.proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=4.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2.0)
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f.close()
            except (OSError, ValueError):
                pass


# --------------------------------------------------------------------- the child
def _load_world(spec):
    import importlib
    module, func, args = spec
    return getattr(importlib.import_module(module), func)(*args)


def _say(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def serve(spec):
    for p in spec.get("paths", []):
        if p not in sys.path:
            sys.path.insert(0, p)
    from game.match import GameRules
    from net.server import GameServer

    def log(*a):
        print(*a, file=sys.stderr, flush=True)
    try:
        world = _load_world(spec["world"])
        kw = {}
        if "next_match_timeout" in spec:
            kw["next_match_timeout"] = spec["next_match_timeout"]
        if "bot_level" in spec:
            kw["bot_level"] = spec["bot_level"]
        server = GameServer(world, port=spec["port"], max_players=spec["max_players"], bots=spec["bots"],
                            rules=GameRules.from_dict(spec["rules"]), seed=spec.get("seed"), log=log, **kw)
        server.start()
    except Exception as exc:        # noqa: BLE001 - reported to the game
        reason = getattr(exc, "strerror", None) or str(exc) or type(exc).__name__
        _say({"t": "error", "reason": reason})
        return 1
    _say({"t": "ready", "port": server.port})
    stop = threading.Event()

    def watch_stdin():
        try:
            for line in sys.stdin:
                if line.strip() == "stop":
                    break
        except (OSError, ValueError):
            pass
        stop.set()                               # "stop", or the game is gone (stdin closed)
    threading.Thread(target=watch_stdin, daemon=True).start()
    try:
        while not stop.wait(0.25):
            _say({"t": "status", "status": server.status()})
            if not server.running:
                break
    except (BrokenPipeError, OSError):
        pass
    finally:
        server.stop()
    return 0


def _dedicated(argv):
    """python -m net.hosted [--port N] [--max-players N] [--bots N] [--bot-level simple|complex|hard]
    [--mode emblems|ctf|zombie] [--teams|--ffa] [--time MIN] [--emblems N] [--captures N] [--goal N]
    - a headless server on the city."""
    import config
    from game.session import parse_mp_args, HostSettings
    from game.match import GameRules
    from net.server import GameServer
    a = parse_mp_args(argv)
    hs = HostSettings(a.get("port", config.MP_DEFAULT_PORT), a.get("max_players", config.MP_MAX_PLAYERS),
                      a.get("bots", 0), a.get("emblems", config.MP_DEFAULT_EMBLEMS), a.get("mode", GameRules.EMBLEMS),
                      a.get("teams", False), a.get("minutes", config.MP_MATCH_TIME_LIMIT / 60.0),
                      a.get("capture_limit", config.MP_DEFAULT_CAPTURES), goal=a.get("goal", config.ZOMBIE_DEFAULT_GOAL),
                      bot_level=a.get("bot_level"))
    # a dedicated server has no host player: every slot can be a bot
    print("[server] building the city ...", flush=True)
    from game.match import city_world
    server = GameServer(city_world(), port=hs.port, max_players=hs.max_players, rules=hs.rules(), log=print,
                        bot_level=hs.bot_level)
    server.bot_target = min(hs.max_players, a.get("bots", 0))
    server.start()
    try:
        while server.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("{"):
        sys.exit(serve(json.loads(sys.argv[1])))
    sys.exit(_dedicated(sys.argv[1:]))
