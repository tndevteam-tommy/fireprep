"""Multiplayer screen model (no rendering): you (name, team), joining, hosting options.

Two columns: on the left who you are and where to join; on the right the game you
host - player limit, bots, the game (emblem hunt / capture the flag / zombie tag),
free for all or teams, the match time (or unlimited), and the game's own option
(emblems / capture limit / goal points).
Rows are walked with Up / Down (keyboard, D-pad or stick); Left / Right change
numbers and toggles; text rows take typing when focused; Enter / A activates.
Settings persist in multiplayer.json next to the other settings.
"""
import json
import os

import config
from game import teams as T
from game import bot_skill

MODES = ("emblems", "ctf", "zombie")
MODE_LABELS = {"emblems": "Emblem hunt", "ctf": "Capture the flag", "zombie": "Zombie tag"}
MODE_OPTION = {"emblems": "emblems", "ctf": "captures", "zombie": "goal"}     # each game's own option row

YOU_ROWS = ("name", "team", "address", "join")
HOST_ROWS = ("port", "max_players", "bots", "bot_level", "game", "teams", "minutes", "emblems", "captures", "goal",
             "host")
END_ROWS = ("leave", "back")
ROWS = YOU_ROWS + HOST_ROWS + END_ROWS
TEXT_ROWS = ("name", "port", "address")
NUMBER_ROWS = ("max_players", "bots", "minutes", "emblems", "captures", "goal")
TOGGLE_ROWS = ("team", "game", "teams", "bot_level")
BUTTON_ROWS = ("join", "host", "leave", "back")
LABELS = {"name": "Your name", "team": "Your team", "address": "Join address", "join": "Join game",
          "port": "Port", "max_players": "Max players", "bots": "Bots", "bot_level": "Bot skill",
          "game": "Game", "teams": "Teams",
          "minutes": "Match time", "emblems": "Emblems", "captures": "Capture limit", "goal": "Goal points",
          "host": "Host a game",
          "leave": "Leave game", "back": "Back"}


def settings_path():
    from core.gamepad import default_settings_path
    return os.path.join(os.path.dirname(default_settings_path()), "multiplayer.json")


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


class MultiplayerMenuModel:
    HOST, JOIN, LEAVE, BACK, TEAM = "host", "join", "leave", "back", "team"

    def __init__(self, values=None):
        self.values = {"name": "Player", "team": T.RED, "address": f"127.0.0.1:{config.MP_DEFAULT_PORT}",
                       "port": str(config.MP_DEFAULT_PORT), "max_players": config.MP_MAX_PLAYERS,
                       "bots": config.MP_DEFAULT_BOTS, "game": "emblems", "teams": False,
                       "minutes": int(config.MP_MATCH_TIME_LIMIT // 60), "emblems": config.MP_DEFAULT_EMBLEMS,
                       "captures": config.MP_DEFAULT_CAPTURES, "goal": config.ZOMBIE_DEFAULT_GOAL,
                       "bot_level": bot_skill.DEFAULT}
        if values:
            self.load_values(values)
        self.focus = 0
        self.in_session = False
        self.status = ""
        self.status_kind = "info"
        self.rects = {}            # row -> (x, y, w, h), filled by the renderer

    # ---------------------------------------------------------------- values
    def load_values(self, data):
        """Take what is valid from saved settings / the command line; ignore the rest."""
        if not isinstance(data, dict):
            return
        v = self.values
        if isinstance(data.get("name"), str) and data["name"].strip():
            v["name"] = data["name"].strip()[:20]
        if isinstance(data.get("address"), str) and data["address"].strip():
            v["address"] = data["address"].strip()[:64]
        if isinstance(data.get("port"), (str, int)) and not isinstance(data.get("port"), bool) \
                and str(data["port"]).isdigit():
            v["port"] = str(int(data["port"]))
        if T.valid(data.get("team")):
            v["team"] = data["team"]
        if data.get("game") in MODES:
            v["game"] = data["game"]
        if data.get("mode") in MODES:                      # (command line: --mode)
            v["game"] = data["mode"]
        if isinstance(data.get("teams"), bool):
            v["teams"] = data["teams"]
        if isinstance(data.get("bot_level"), str) and data["bot_level"].strip().lower() in bot_skill.LEVELS:
            v["bot_level"] = data["bot_level"].strip().lower()
        if isinstance(data.get("bot"), bool):              # (the old on/off switch)
            v["bots"] = 1 if data["bot"] else 0
        for k, lo, hi in (("max_players", 2, config.MP_MAX_PLAYERS), ("emblems", *config.MP_EMBLEM_RANGE),
                          ("bots", 0, config.MP_MAX_PLAYERS - 1), ("captures", 0, config.MP_CAPTURE_LIMIT_MAX),
                          ("minutes", 0, 60), ("goal", *config.ZOMBIE_GOAL_RANGE)):
            src = k if k != "captures" else ("captures" if "captures" in data else "capture_limit")
            if _int(data.get(src)):
                v[k] = max(lo, min(hi, data[src]))
        if v["minutes"] not in config.MP_MATCH_MINUTES:
            v["minutes"] = min(config.MP_MATCH_MINUTES, key=lambda m: abs(m - v["minutes"]))
        step = config.ZOMBIE_GOAL_STEP
        v["goal"] = int(round(v["goal"] / step) * step) if v["goal"] % step else v["goal"]
        v["bots"] = min(v["bots"], v["max_players"] - 1)

    def save(self, path=None):
        path = path or settings_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.values, f, indent=2)
            return True
        except OSError:
            return False

    @classmethod
    def load(cls, path=None):
        path = path or settings_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls(json.load(f))
        except (OSError, ValueError):
            return cls()

    # ------------------------------------------------------------------ rows
    def rows(self):
        game = self.values["game"]
        hidden = {MODE_OPTION[m] for m in MODES if m != game}
        if game == "zombie":
            hidden.add("teams")                        # zombie tag: every player for themself
        if not self.values["bots"]:
            hidden.add("bot_level")                    # no bots: nothing to be clever
        if not self.in_session:
            hidden.add("leave")
        return [r for r in ROWS if r not in hidden]

    def column(self, row):
        return "you" if row in YOU_ROWS else ("host" if row in HOST_ROWS else "end")

    @property
    def focused(self):
        rows = self.rows()
        return rows[min(self.focus, len(rows) - 1)]

    def value_text(self, row):
        v = self.values
        if row in ("name", "address", "port"):
            return v[row]
        if row == "team":
            return T.name(v["team"])
        if row == "game":
            return MODE_LABELS[v["game"]]
        if row == "teams":
            return "Red vs Blue" if v["teams"] else "Free for all"
        if row == "bot_level":
            return bot_skill.LABELS[bot_skill.normalize(v["bot_level"])]
        if row == "minutes":
            return f"{v['minutes']} min" if v["minutes"] else "Unlimited"
        if row == "captures":
            return str(v["captures"]) if v["captures"] else "Off"
        if row in ("max_players", "emblems", "bots"):
            return str(v[row])
        if row == "goal":
            return f"{v['goal']}"
        return ""

    def say(self, text, kind="info"):
        self.status, self.status_kind = text, kind

    def _adjust(self, row, step):
        """Left / Right on a row.  Returns an action for changes that matter right now (your team)."""
        v = self.values
        if row == "max_players":
            v[row] = max(2, min(config.MP_MAX_PLAYERS, v[row] + step))
            v["bots"] = min(v["bots"], v[row] - 1)
        elif row == "bots":
            v[row] = max(0, min(v["max_players"] - 1, v[row] + step))
            if step > 0 and v[row] == v["max_players"] - 1 and v["max_players"] < config.MP_MAX_PLAYERS:
                self.say(f"At most {v['max_players'] - 1} bots with {v['max_players']} players "
                         "(raise Max players for more)", "info")
        elif row == "emblems":
            lo, hi = config.MP_EMBLEM_RANGE
            v[row] = max(lo, min(hi, v[row] + step))
        elif row == "captures":
            v[row] = max(0, min(config.MP_CAPTURE_LIMIT_MAX, v[row] + step))
        elif row == "goal":
            lo, hi = config.ZOMBIE_GOAL_RANGE
            v[row] = max(lo, min(hi, v[row] + step * config.ZOMBIE_GOAL_STEP))
        elif row == "minutes":
            opts = config.MP_MATCH_MINUTES
            i = opts.index(v[row]) if v[row] in opts else 0
            v[row] = opts[max(0, min(len(opts) - 1, i + step))]
        elif row == "game":
            v[row] = MODES[(MODES.index(v[row]) + (1 if step > 0 else -1)) % len(MODES)]
            rows = self.rows()
            self.focus = rows.index("game")
        elif row == "teams":
            v[row] = not v[row]
        elif row == "bot_level":
            v[row] = bot_skill.cycle(v[row], step)
            self.say({bot_skill.SIMPLE: "Simple bots: chase straight at you, wander to find emblems",
                      bot_skill.COMPLEX: "Complex bots: hunt as a pack, search and remember, flee over roofs",
                      bot_skill.HARD: "Hard bots: masterminds - they share what they see and trap you"}[v[row]], "info")
        elif row == "team":
            v[row] = T.other(v[row])
            if self.in_session:
                return (self.TEAM, v[row])
        return None

    def host_request(self):
        port = self.values["port"]
        if not port.isdigit() or not (1024 <= int(port) <= 65535):
            self.say("The port must be a number from 1024 to 65535", "warn")
            return None
        from game.session import HostSettings
        v = self.values
        hs = HostSettings(int(port), v["max_players"], v["bots"], v["emblems"], v["game"], v["teams"],
                          v["minutes"], v["captures"], goal=v["goal"], bot_level=v["bot_level"])
        if hs.rules().never_ends():
            self.say("No time limit and no capture limit: the round only ends when you leave", "info")
        return (self.HOST, hs)

    def _activate(self, row):
        if row == "host":
            return self.host_request()
        if row == "join":
            if not self.values["address"].strip():
                self.say("Type the host's address first (e.g. 192.168.1.20:24950)", "warn")
                return None
            return (self.JOIN, self.values["address"].strip())
        if row == "leave":
            return (self.LEAVE,)
        if row == "back":
            return (self.BACK,)
        if row in TOGGLE_ROWS:
            return self._adjust(row, 1)
        return None

    # ------------------------------------------------------------------ input
    def handle_key(self, key, char=""):
        """A keyboard key (pygame key name + the typed character).  Returns an action tuple or None."""
        row = self.focused
        if key == "escape":
            return (self.BACK,)
        if key in ("up",):
            self.focus = (self.focus - 1) % len(self.rows())
        elif key in ("down", "tab"):
            self.focus = (self.focus + 1) % len(self.rows())
        elif key in ("left", "right") and row not in TEXT_ROWS:
            return self._adjust(row, -1 if key == "left" else 1)
        elif key in ("return", "enter"):
            return self._activate(row)
        elif row in TEXT_ROWS:
            text = self.values[row]
            if key == "backspace":
                self.values[row] = text[:-1]
            elif char and char.isprintable() and len(text) < (20 if row == "name" else 64):
                if row != "port" or char.isdigit():
                    self.values[row] = text + char
        return None

    def handle_nav(self, cmd):
        """Controller: 'up' / 'down' / 'left' / 'right' / 'confirm' / 'back'."""
        if cmd == "back":
            return (self.BACK,)
        if cmd == "confirm":
            return self._activate(self.focused)
        return self.handle_key({"up": "up", "down": "down", "left": "left", "right": "right"}.get(cmd, ""))

    def handle_mouse(self, kind, pos):
        for row, (x, y, w, h) in self.rects.items():
            if x <= pos[0] <= x + w and y <= pos[1] <= y + h and row in self.rows():
                self.focus = self.rows().index(row)
                if kind == "click":
                    if row in NUMBER_ROWS:
                        return self._adjust(row, 1 if pos[0] > x + w * 0.5 else -1)
                    return self._activate(row)
                return None
        return None
