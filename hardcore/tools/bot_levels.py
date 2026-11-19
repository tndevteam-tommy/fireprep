"""Bot intelligence benchmark: how the Simple / Complex / Hard levels compare in whole games.

Runs real (headless) server matches on the city with bots of chosen levels and prints a table.
Whole games are chaotic - one seed proves nothing - so every experiment runs many seeds (in
parallel worker processes) and reports the mean and the median.

    python tools/bot_levels.py hunt    [--seeds 6]  [--emblems 12] [--time 300]   one bot alone, emblems found
    python tools/bot_levels.py race    [--seeds 8]  [--emblems 12] [--time 300]   one bot of each level, same match
    python tools/bot_levels.py chase   [--seeds 16] [--runner simple]            3 zombies of each level: time to catch
    python tools/bot_levels.py flee    [--seeds 16] [--zombies simple]           the living bot of each level: time alive
    python tools/bot_levels.py ctf     [--seeds 4]  [--time 300]                 3 v 3, each level against Simple

(--workers N sets the number of processes; the default is the number of CPUs.)
"""
import argparse
import os
import statistics
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LEVELS = ("simple", "complex", "hard")
_WORLD = {}


def _world():
    if "city" not in _WORLD:
        from game.match import city_world
        _WORLD["city"] = city_world()
    return _WORLD["city"]


def _server(rules, n_bots, seed):
    from net.server import GameServer
    srv = GameServer(_world(), port=0, max_players=max(2, n_bots + 1), bots=n_bots, rules=rules, seed=seed, log=None)
    events = []
    srv._broadcast = lambda msg, skip=None: events.append(msg)
    srv.match.start({})
    srv._fill_bots()
    return srv, events


def _ticks(seconds):
    return int(seconds * 60)


# ----------------------------------------------------------------------------- experiments
def hunt(level, seed, emblems, seconds):
    """One bot alone: emblems found in `seconds` (and when the last was found, if it found them all)."""
    from game.match import GameRules
    srv, events = _server(GameRules(GameRules.EMBLEMS, False, 0, emblems=emblems), 1, seed)
    srv.set_bot_level(level)
    for i in range(_ticks(seconds)):
        srv._tick(1.0 / 60.0)
        if any(e.get("t") == "over" for e in events):
            return emblems, i / 60.0
    return sum(1 for e in events if e.get("t") == "pickup"), None


def race(seed, emblems, seconds):
    """One bot of each level in the same emblem hunt: emblems each got."""
    from game.match import GameRules
    srv, events = _server(GameRules(GameRules.EMBLEMS, False, 0, emblems=emblems), 3, seed)
    pids = sorted(srv.bots)
    for pid, level in zip(pids, LEVELS):
        srv.set_bot_level(level, pid)
    for _ in range(_ticks(seconds)):
        srv._tick(1.0 / 60.0)
        over = [e for e in events if e.get("t") == "over"]
        if over:
            scores = {int(k): v for k, v in over[0]["scores"].items()}
            return {level: scores.get(pid, 0) for pid, level in zip(pids, LEVELS)}
    return {level: srv.match.scores.get(pid, 0) for pid, level in zip(pids, LEVELS)}


def zombie(zombie_level, living_level, seed, seconds):
    """Zombie tag, 3 zombies and the living bot: seconds until the first catch (None: not in time)."""
    from game.match import GameRules
    srv, events = _server(GameRules(GameRules.ZOMBIE, False, 0, goal=600), 4, seed)
    living = srv.match.living
    for pid in srv.bots:
        srv.set_bot_level(living_level if pid == living else zombie_level, pid)
    for i in range(_ticks(seconds)):
        srv._tick(1.0 / 60.0)
        if any(e.get("t") == "zombie" and e.get("action") == "caught" for e in events):
            return i / 60.0
    return None


def ctf(red_level, blue_level, seed, seconds, per_team=3):
    """Capture the flag, red vs blue bots: captures and tags (thieves sent home) per side."""
    from game.match import GameRules
    from game import teams as T
    srv, events = _server(GameRules(GameRules.CTF, True, 0, capture_limit=0), per_team * 2, seed)
    for pid, b in srv.bots.items():
        srv.set_bot_level(red_level if b.info.team == T.RED else blue_level, pid)
    for _ in range(_ticks(seconds)):
        srv._tick(1.0 / 60.0)
    caps = {T.RED: 0, T.BLUE: 0}
    tags = {T.RED: 0, T.BLUE: 0}
    for e in events:
        if e.get("t") == "flag" and e.get("action") == "captured":
            caps[e["team"]] += 1
        elif e.get("t") == "flag" and e.get("action") == "returned" and e.get("reason") == "tagged":
            tags[e["flag"]] += 1               # the flag's own side tagged the thief
    return caps, tags


def _call(job):
    fn, args = job
    return globals()[fn](*args)


# ----------------------------------------------------------------------------------- report
def _summary(values, cap=None):
    vals = [cap if v is None else v for v in values]
    missing = sum(1 for v in values if v is None)
    return statistics.mean(vals), statistics.median(vals), missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("hunt", "race", "chase", "flee", "ctf"))
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--emblems", type=int, default=12)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--runner", default="simple", choices=LEVELS)
    ap.add_argument("--zombies", default="simple", choices=LEVELS)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    seeds = list(range(1, 1 + (a.seeds or {"hunt": 6, "race": 8, "chase": 16, "flee": 16, "ctf": 4}[a.what])))
    t0 = time.time()
    with Pool(a.workers) as pool:
        if a.what == "hunt":
            secs = a.time or 300.0
            print(f"Emblem hunt, one bot alone: emblems found of {a.emblems} in {secs:.0f} s ({len(seeds)} seeds)")
            for level in LEVELS:
                res = pool.map(_call, [("hunt", (level, s, a.emblems, secs)) for s in seeds])
                found = [n for n, _ in res]
                done = [t for _, t in res if t is not None]
                print(f"  {level:8s} found mean {statistics.mean(found):5.1f}  per seed {found}"
                      f"  cleared {len(done)}/{len(seeds)}" + (f" (mean {statistics.mean(done):.0f} s)" if done else ""))
        elif a.what == "race":
            secs = a.time or 300.0
            res = pool.map(_call, [("race", (s, a.emblems, secs)) for s in seeds])
            print(f"Emblem race, one bot of each level in the same match ({len(seeds)} seeds, {secs:.0f} s)")
            for level in LEVELS:
                got = [r[level] for r in res]
                wins = sum(1 for r in res if r[level] == max(r.values()) and list(r.values()).count(r[level]) == 1)
                print(f"  {level:8s} emblems {sum(got):3d}  per seed {got}  outright wins {wins}")
        elif a.what in ("chase", "flee"):
            secs = a.time or 150.0
            if a.what == "chase":
                print(f"Zombie tag, 3 zombies of each level vs a {a.runner} living bot: seconds to the first catch "
                      f"({len(seeds)} seeds, capped at {secs:.0f} s; lower is better)")
                jobs = {lv: [("zombie", (lv, a.runner, s, secs)) for s in seeds] for lv in LEVELS}
            else:
                print(f"Zombie tag, the living bot of each level vs 3 {a.zombies} zombies: seconds alive "
                      f"({len(seeds)} seeds, capped at {secs:.0f} s; higher is better)")
                jobs = {lv: [("zombie", (a.zombies, lv, s, secs)) for s in seeds] for lv in LEVELS}
            for level in LEVELS:
                mean, median, missing = _summary(pool.map(_call, jobs[level]), cap=secs)
                print(f"  {level:8s} mean {mean:6.1f}  median {median:6.1f}  not caught in time {missing}")
        else:
            secs = a.time or 300.0
            print(f"Capture the flag, 3 v 3 ({len(seeds)} seeds, {secs:.0f} s): each level as red against Simple blue")
            for level in LEVELS:
                res = pool.map(_call, [("ctf", (level, "simple", s, secs)) for s in seeds])
                caps_r = sum(c["red"] for c, _ in res)
                caps_b = sum(c["blue"] for c, _ in res)
                tags_r = sum(t["red"] for _, t in res)
                tags_b = sum(t["blue"] for _, t in res)
                print(f"  {level:8s} red captures {caps_r:2d} vs {caps_b:2d}   thieves tagged by red {tags_r:2d} vs {tags_b:2d}")
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
