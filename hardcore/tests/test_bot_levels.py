"""How the bot intelligence levels compare in whole games: headless servers on the city, minutes of
simulated play (these are the slow tests).  Whole games are chaotic, so only differences large enough
to show on a couple of seeds are pinned here; the many-seed comparisons are tools/bot_levels.py (its
results are in DESIGN.md section 11.6)."""
from harness import *  # noqa
from game import bot_skill
from game.match import GameRules, city_world
from net.server import GameServer

LEVELS = bot_skill.LEVELS
_CITY = {}


def city():
    if "w" not in _CITY:
        _CITY["w"] = city_world()
    return _CITY["w"]


def _server(rules, n_bots, seed):
    srv = GameServer(city(), port=0, max_players=n_bots + 1, bots=n_bots, rules=rules, seed=seed, log=None)
    events = []
    srv._broadcast = lambda msg, skip=None: events.append(msg)
    srv.match.start({})
    srv._fill_bots()
    return srv, events


def _hunt(level, seed, seconds):
    srv, events = _server(GameRules(GameRules.EMBLEMS, False, 0, emblems=12), 1, seed)
    srv.set_bot_level(level)
    bot = next(iter(srv.bots.values()))
    gone_for = []
    for _ in range(int(seconds * 60)):
        srv._tick(1.0 / 60.0)
        g = bot.brain.goal
        if g is not None and g["kind"] == "target" and isinstance(g.get("target_id"), int):
            if not gone_for or gone_for[-1] != g["target_id"]:
                gone_for.append(g["target_id"])
        if any(e.get("t") == "over" for e in events):
            break
    found = sum(1 for e in events if e.get("t") == "pickup")
    return found, bot.tactician, gone_for


def test_smarter_bots_find_more_emblems():
    found = {lv: 0 for lv in LEVELS}
    for seed in (1, 3):
        for lv in LEVELS:
            found[lv] += _hunt(lv, seed, 240.0)[0]
    assert found["hard"] > found["complex"] > found["simple"], found


def test_a_bot_only_goes_for_emblems_it_has_spotted():
    found, tac, gone_for = _hunt("complex", 2, 60.0)
    spotted = [d[1] for d in tac.decisions if d[0] == "spotted"]
    assert gone_for, "it went for something"
    assert set(gone_for) <= set(spotted) | {d[1] for d in tac.decisions if d[0] == "go"}
    # and at the start it did not simply run for the nearest emblem: what it knew is what it saw
    assert tac.spotted <= 12 and tac.search.count() > 0
