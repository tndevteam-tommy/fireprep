"""Wire protocol: one JSON object per line over TCP.

Client -> server
    hello  {v, name, team}                 first message; answered by welcome or reject
    state  {p, yaw, pitch, st, n, spd, ac} the player's pose ~30x a second (ac: a flip / roll in progress)
    team   {team}                          switch sides (teams games)
    ready  {}                              join the next match (after a match is over)
    bye    {}
Server -> client
    welcome {id, v, team, server{...}, match{...}, players[...], spawn}
    reject  {reason}
    snap    {time, now, players[{id, name, bot, team, p, yaw, pitch, st, n, spd, ac, pt, score}], match{...}, time_left}
            (time_left is null when there is no time limit; `now` is the server clock and each player's
            `pt` the server-clock time its pose is from - clients interpolate on that timeline, so a pose
            is never shown twice or skipped just because updates and snapshots are not in step)
    pickup  {emblem, player, name, team, score, remaining, team_scores}
    flag    {action: taken / captured / returned / stolen, flag, player, name, team, by, ...}
    zombie  {action: living / caught, player (alive now), name, from (caught), from_name, reason}
    join / leave {id, name, bot, team}
    team    {id, name, team}               someone switched sides
    respawn {spawn}                        you were moved (switched sides)
    over    {winners[ids], team, draw, scores{id: n}, team_scores, teams{id: team}, names{id: name}, timeout}
    ready_status {ready[ids], needed, timeout}
    start   {match{...}, spawn, team}

Lines are capped (MAX_LINE) and anything that does not parse as a JSON object with a
string "t" is dropped, so a bad peer cannot wedge the other end.
"""
import json
import math

PROTOCOL_VERSION = 4              # 4: pose timestamps (pt / now) and the momentum movement tuning
MAX_LINE = 256 * 1024


class ProtocolError(Exception):
    pass


def encode(msg):
    return (json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


class LineReader:
    """Accumulates bytes and yields complete, valid messages."""

    def __init__(self, max_line=MAX_LINE):
        self.buf = bytearray()
        self.max_line = max_line
        self.dropped = 0

    def feed(self, data):
        self.buf.extend(data)
        out = []
        while True:
            i = self.buf.find(b"\n")
            if i < 0:
                if len(self.buf) > self.max_line:
                    raise ProtocolError("line too long")
                return out
            line = bytes(self.buf[:i])
            del self.buf[:i + 1]
            if not line.strip():
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                self.dropped += 1
                continue
            if isinstance(msg, dict) and isinstance(msg.get("t"), str):
                out.append(msg)
            else:
                self.dropped += 1


def finite_vec(v, n=3, limit=1e5):
    """A list of n finite numbers within +-limit, or None."""
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    out = []
    for c in v:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or abs(c) > limit:
            return None
        out.append(float(c))
    return out


def finite(x, default=0.0, limit=1e6):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > limit:
        return default
    return float(x)


def clean_name(name, fallback="Player"):
    if not isinstance(name, str):
        return fallback
    name = "".join(ch for ch in name if ch.isprintable()).strip()[:20]
    return name or fallback


ACRO_KINDS = ("front", "back", "left", "right", "roll")


def clean_acro(ac):
    """A flip / roll pose [kind, degrees, crouch, heading x, heading z] - or None if it is not one."""
    if not isinstance(ac, (list, tuple)) or len(ac) != 5 or ac[0] not in ACRO_KINDS:
        return None
    nums = finite_vec(list(ac[1:]), 4, limit=720.0)
    if nums is None:
        return None
    ang, crouch, hx, hz = nums
    if not (0.0 <= ang <= 360.0 and 0.0 <= crouch <= 1.5 and abs(hx) <= 1.01 and abs(hz) <= 1.01):
        return None
    return [ac[0], ang, crouch, hx, hz]
