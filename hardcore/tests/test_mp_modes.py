"""Game modes and teams: emblem hunts and capture the flag, free for all and red vs blue, bot counts,
team choice and balance, match time - the rules headless, then over real sockets, then bots playing."""
import math
import time

import numpy as np

from mp_helpers import *  # noqa
from game import teams as T
from game.match import (GameRules, EmblemMatch, CtfMatch, create_match, Target, TELEPORT_DIST, FLAG_HEIGHT)
from game.session import MultiplayerSession, HostSettings
from net.server import GameServer
from net.client import GameClient
from net.protocol import clean_acro

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _body(ground):
    return np.asarray(ground, dtype=np.float32) + UP * config.PLAYER_HALF_HEIGHT


def _ctf(teams=True, seed=3, **kw):
    m = create_match(small_world(), GameRules(GameRules.CTF, teams, **kw), seed=seed)
    return m


# ------------------------------------------------------------------------- rules
def test_game_rules_are_clamped_described_and_survive_the_wire():
    r = GameRules("ctf", True, 5 * 60, emblems=999, capture_limit=99)
    assert r.emblems == config.MP_EMBLEM_RANGE[1] and r.capture_limit == config.MP_CAPTURE_LIMIT_MAX
    assert r.describe().startswith("Capture the flag - red vs blue - 5 min")
    back = GameRules.from_dict(r.to_dict())
    assert back.to_dict() == r.to_dict()
    assert GameRules.from_dict({"mode": "golf", "time_limit": "x"}).mode == GameRules.EMBLEMS
    assert GameRules("emblems", False).describe() == "Emblem hunt - free for all - 10 min"
    assert isinstance(create_match(small_world(), GameRules("ctf")), CtfMatch)
    assert isinstance(create_match(small_world(), GameRules("emblems")), EmblemMatch)


def test_a_teams_emblem_hunt_adds_up_each_side():
    w = small_world()
    m = create_match(w, GameRules(GameRules.EMBLEMS, True, emblems=5), seed=4)
    m.start({1: T.RED, 2: T.BLUE, 3: T.RED})
    em = m.emblems
    ev = m.apply_move(1, em[0].pos, em[0].pos)
    assert ev and ev[0]["t"] == "pickup" and ev[0]["team"] == T.RED and ev[0]["team_scores"][T.RED] == 1
    m.apply_move(3, em[1].pos, em[1].pos)
    m.apply_move(2, em[2].pos, em[2].pos)
    m.apply_move(2, em[3].pos, em[3].pos)
    assert m.team_scores == {T.RED: 2, T.BLUE: 2} and m.scores[2] == 2
    m.apply_move(2, em[4].pos, em[4].pos)
    assert m.state == m.OVER and m.winning_team == T.BLUE and sorted(m.winners) == [2]
    res = m.results()
    assert res["team"] == T.BLUE and res["team_scores"] == {T.RED: 2, T.BLUE: 3} and res["teams"]["1"] == T.RED
    # the sides start apart on the start line
    reds = [m.spawn_point(1) for _ in range(3)]
    blues = [m.spawn_point(2) for _ in range(3)]
    assert max(p[0] for p in reds) < min(p[0] for p in blues)
    # a draw
    m.start({1: T.RED, 2: T.BLUE})
    m.apply_move(1, m.emblems[0].pos, m.emblems[0].pos)
    m.apply_move(2, m.emblems[1].pos, m.emblems[1].pos)
    m.tick(m.time_limit + 1.0)
    assert m.draw and m.winning_team is None and m.winners == []


def _run(m, pid, a, b, step=4.0):
    """Move a player from a to b in short hops (a run, not a teleport); returns all the events."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
    events, prev = [], a
    for k in range(1, n + 1):
        p = a + (b - a) * (k / n)
        events += m.apply_move(pid, prev, p)
        prev = p
    return events


def test_capture_the_flag_in_teams():
    m = _ctf(True)
    m.start({1: T.RED, 2: T.BLUE, 3: T.BLUE})
    red, blue = m.flags[T.RED], m.flags[T.BLUE]
    R, B = _body(red.home), _body(blue.home)
    side = np.array([0.0, 0.0, 25.0], np.float32)
    assert float(np.linalg.norm(red.home - blue.home)) >= config.MP_CTF_BASE_DISTANCE[0], "bases far apart"
    assert red.at_home and blue.at_home
    for pid, team in ((1, T.RED), (2, T.BLUE)):
        sp = np.asarray(m.spawn_point(pid))
        assert float(np.linalg.norm((sp - m.flags[team].home)[[0, 2]])) < 10.0, "you start at your own base"
    # your own flag is not yours to take
    assert m.apply_move(1, R, R) == []
    # run through the blue stand: take it
    ev = m.apply_move(1, B + [3.0, 0, 0], B - [3.0, 0, 0])
    assert [e["action"] for e in ev] == ["taken"] and blue.carrier == 1
    # blue's Ben takes red's flag and runs off with it: red cannot score while its own flag is away
    m.apply_move(2, R, R)
    assert red.carrier == 2
    _run(m, 2, R, R + side)
    ev = _run(m, 1, B, R)
    assert not [e for e in ev if e["action"] == "captured"], "your flag must be at home"
    assert blue.carrier == 1
    # blue's Bea (3) tags red's carrier (1) by the red base: blue's flag goes home
    _run(m, 1, R, R + [5.0, 0, 0])
    m.apply_move(3, R + [17.0, 0, 0], R + [17.0, 0, 0])
    ev = _run(m, 3, R + [17.0, 0, 0], R + [5.5, 0, 0])
    assert any(e["action"] == "returned" and e["reason"] == "tagged" and e["by"] == 3 and e["flag"] == T.BLUE
               for e in ev) and blue.at_home
    _run(m, 3, R + [5.5, 0, 0], R + [17.0, 0, 0] - side)
    # red's Ann runs down Ben, who has red's flag: it goes home
    ev = _run(m, 1, R + [5.0, 0, 0], R + side)
    assert any(e["action"] == "returned" and e["flag"] == T.RED and e["by"] == 1 for e in ev) and red.at_home
    # a clean capture: take blue's flag, bring it home (both flags at home now)
    _run(m, 2, R + side, R + side * 2.0)
    _run(m, 1, R + side, B)
    assert blue.carrier == 1
    ev = _run(m, 1, B, R + [1.0, 0, 1.0])
    cap = [e for e in ev if e["action"] == "captured"]
    assert cap and cap[0]["team_scores"] == {T.RED: 1, T.BLUE: 0} and blue.at_home and m.scores[1] == 1
    # a respawn / teleport while carrying sends the flag home
    _run(m, 1, R, B)
    assert blue.carrier == 1
    ev = m.apply_move(1, B, B + [TELEPORT_DIST + 20.0, 0, 0])
    assert any(e["action"] == "returned" and e["reason"] == "respawn" for e in ev) and blue.at_home
    # leaving or switching sides with the flag sends it home too
    _run(m, 1, B + [TELEPORT_DIST + 20.0, 0, 0], B)
    assert blue.carrier == 1
    assert [e["reason"] for e in m.set_team(1, T.BLUE)] == ["switched"] and blue.at_home and m.team_of(1) == T.BLUE
    _run(m, 2, R + side * 2.0, R)
    assert red.carrier == 2
    assert [e["reason"] for e in m.remove_player(2)] == ["left"] and red.at_home


def test_capture_limit_and_time_limit_decide_the_winner():
    m = _ctf(True, capture_limit=2)
    m.start({1: T.RED, 2: T.BLUE})
    red, blue = m.flags[T.RED], m.flags[T.BLUE]
    for i in range(2):
        assert m.state == m.PLAYING
        m.apply_move(1, _body(blue.home), _body(blue.home))
        m.apply_move(1, _body(red.home), _body(red.home))
    assert m.state == m.OVER and m.winning_team == T.RED and m.team_scores[T.RED] == 2, "first to the limit"
    # no limit: the time runs out, most captures wins; level = a draw
    m = _ctf(True, capture_limit=0, time_limit=60.0)
    m.start({1: T.RED, 2: T.BLUE})
    m.apply_move(2, _body(m.flags[T.RED].home), _body(m.flags[T.RED].home))
    m.apply_move(2, _body(m.flags[T.BLUE].home), _body(m.flags[T.BLUE].home))
    assert not m.tick(59.0) and m.tick(2.0) and m.winning_team == T.BLUE
    m.start({1: T.RED, 2: T.BLUE})
    m.tick(61.0)
    assert m.state == m.OVER and not m.draw and m.winning_team is None, "nobody scored"


def test_capture_the_flag_free_for_all():
    m = _ctf(False, capture_limit=2)
    m.start([1, 2, 3])
    flag, zone = m.flags["neutral"], m.zones["drop"]
    assert flag.team is None and np.linalg.norm((flag.home - zone.center)[[0, 2]]) > 10.0
    ev = m.apply_move(1, flag.touch_point(), flag.touch_point())
    assert ev[0]["action"] == "taken" and flag.carrier == 1
    # 2 touches the carrier at once: too soon after the grab to steal
    far = m.positions[1] + np.array([20.0, 0, 0], np.float32)
    m.positions[1] = far
    assert m.apply_move(2, far + [1.0, 0, 0], far + [0.4, 0, 0]) == []
    m.elapsed += config.MP_STEAL_COOLDOWN + 0.1
    ev = m.apply_move(2, far + [1.0, 0, 0], far + [0.4, 0, 0])
    assert ev[0]["action"] == "stolen" and ev[0]["from"] == 1 and flag.carrier == 2
    # 2 runs it into the zone: a capture, and a new flag + zone somewhere else
    old = (tuple(flag.home), tuple(zone.center))
    ev = m.apply_move(2, _body(zone.center) + [5.0, 0, 0], _body(zone.center))
    assert ev[-1]["action"] == "captured" and m.scores[2] == 1
    assert (tuple(m.flags["neutral"].home), tuple(m.zones["drop"].center)) != old and m.flags["neutral"].at_home
    f = m.flags["neutral"]
    m.apply_move(3, f.touch_point(), f.touch_point())
    m.apply_move(3, _body(m.zones["drop"].center), _body(m.zones["drop"].center))
    f = m.flags["neutral"]
    m.apply_move(2, f.touch_point(), f.touch_point())
    m.apply_move(2, _body(m.zones["drop"].center), _body(m.zones["drop"].center))
    assert m.state == m.OVER and m.winners == [2] and m.scores[2] == 2, "first to two captures"


def test_bots_are_told_what_to_go_for():
    m = _ctf(True)
    m.start({1: T.RED, 2: T.BLUE, 3: T.RED})
    red, blue = m.flags[T.RED], m.flags[T.BLUE]
    m.positions.update({1: _body(red.home), 2: _body(blue.home), 3: _body(red.home)})
    t = m.bot_targets(1)
    assert len(t) == 1 and t[0].id == "flag:blue", "attack: the enemy flag"
    assert m.bot_targets(3, role="defend")[0].id == "guard:red", "defend: stay by your flag"
    m.apply_move(1, _body(blue.home), _body(blue.home))
    assert m.bot_targets(1)[0].id == "zone:red" and m.bot_targets(1)[0].kind == "zone", "carrying: go home"
    assert m.bot_targets(3)[0].id == "zone:red", "a teammate has it: meet them at home"
    m.apply_move(2, _body(red.home), _body(red.home))
    assert m.bot_targets(3, role="defend")[0].id == "player:2", "our flag taken: chase the thief"
    # free for all: flag -> carrier -> zone
    f = _ctf(False)
    f.start([1, 2])
    assert f.bot_targets(1)[0].id == "flag:neutral"
    f.apply_move(2, f.flags["neutral"].touch_point(), f.flags["neutral"].touch_point())
    assert f.bot_targets(1)[0].id == "player:2" and f.bot_targets(2)[0].id == "zone:drop"
    # emblems: team bots split the emblems between them
    e = create_match(small_world(), GameRules(GameRules.EMBLEMS, True, emblems=4), seed=2)
    e.start({1: T.RED, 2: T.RED})
    all_ids = {t.id for t in e.bot_targets(1)}
    assert len(all_ids) == 4 and {t.id for t in e.bot_targets(1, claimed={0, 1})} == all_ids - {0, 1}


def test_acrobatics_travel_safely_over_the_wire():
    assert clean_acro(["front", 120.0, 0.0, 0.6, -0.8]) == ["front", 120.0, 0.0, 0.6, -0.8]
    for bad in (None, "x", ["spin", 1, 0, 0, 0], ["roll", float("nan"), 0, 0, 0], ["roll", 999, 0, 0, 0],
                ["front", 10, 0, 5.0, 0], ["back", 10, 0, 0]):
        assert clean_acro(bad) is None


# ------------------------------------------------------------------- server
def _server(**kw):
    kw.setdefault("port", 0)
    kw.setdefault("seed", 5)
    kw.setdefault("log", None)
    srv = GameServer(small_world(), host="127.0.0.1", **kw)
    srv.start()
    return srv


def _events(client, kind, timeout=3.0, where=lambda e: True):
    got = []

    def check():
        got.extend(e for e in client.poll_events() if e.get("t") == kind and where(e))
        return bool(got)
    wait_for(check, timeout)
    return got


def test_bot_count_is_limited_by_the_player_limit():
    for max_players, bots, want in ((4, 10, 3), (4, 0, 0), (2, 1, 1), (15, 14, 14)):
        srv = GameServer(small_world(), port=0, max_players=max_players, bots=bots, log=None)
        assert srv.bot_target == want, (max_players, bots)
    srv = _server(max_players=4, bots=3)
    try:
        assert srv.status()["bots"] == 3
        a = _client_ok(srv, "Ann")
        assert srv.status()["bots"] == 3 and srv.status()["humans"] == 1, "the host's slot was kept free"
        b = _client_ok(srv, "Ben")
        assert srv.status()["bots"] == 2, "a bot gave its slot to Ben"
        b.close()
        assert wait_for(lambda: srv.status()["bots"] == 3), "and took it back when Ben left"
        a.close()
    finally:
        srv.stop()


def _client_ok(srv, name, team=None):
    return GameClient("127.0.0.1", srv.port, name, team=team).connect()


def test_people_pick_their_team_and_bots_even_the_sides():
    srv = _server(max_players=8, bots=4, rules=GameRules(GameRules.EMBLEMS, True, emblems=4))
    try:
        assert srv.status()["teams"] == {T.RED: 2, T.BLUE: 2}
        a = _client_ok(srv, "Ann", T.BLUE)
        b = _client_ok(srv, "Ben", T.BLUE)
        assert a.team == T.BLUE and b.team == T.BLUE, "you get the side you asked for"
        st = srv.status()["teams"]
        assert abs(st[T.RED] - st[T.BLUE]) <= 1 and st[T.RED] + st[T.BLUE] == 6, "bots moved over to even it up"
        assert a.teams_game and a.rules["mode"] == "emblems"
        # switch sides in the game: told, moved to the new side's spawn, bots re-balance
        a.send_team(T.RED)
        ev = _events(a, "respawn")
        assert ev and ev[0]["team"] == T.RED and wait_for(lambda: a.team == T.RED)
        assert _events(b, "team", where=lambda e: e["id"] == a.id)
        assert wait_for(lambda: abs(srv.status()["teams"][T.RED] - srv.status()["teams"][T.BLUE]) <= 1)
        c = _client_ok(srv, "Cat", "purple")
        assert c.team in T.TEAMS, "no valid choice: the smaller side"
        # everyone sees everyone's team
        assert wait_for(lambda: a.players.get(b.id) is not None and a.players[b.id].team == T.BLUE)
    finally:
        srv.stop()


def test_free_for_all_games_have_no_teams():
    srv = _server(bots=1, rules=GameRules(GameRules.EMBLEMS, False))
    try:
        a = _client_ok(srv, "Ann", T.BLUE)
        assert a.team is None and not a.teams_game
        a.send_team(T.RED)
        assert not _events(a, "respawn", timeout=0.3)
        assert all(p.team is None for p in srv.players.values())
    finally:
        srv.stop()


def test_capture_the_flag_over_the_network():
    rules = GameRules(GameRules.CTF, True, time_limit=120.0, capture_limit=1)
    srv = _server(bots=0, rules=rules)
    try:
        a = _client_ok(srv, "Ann", T.RED)
        b = _client_ok(srv, "Ben", T.BLUE)
        flags = {f["key"]: f for f in a.objectives["flags"]}
        assert set(flags) == {T.RED, T.BLUE} and len(a.objectives["zones"]) == 2
        blue_home, red_home = np.array(flags[T.BLUE]["home"]), np.array(flags[T.RED]["home"])
        sa = np.array(a.spawn)
        assert float(np.linalg.norm((sa - red_home)[[0, 2]])) < 10.0, "Ann starts by the red base"
        a.send_state(_body(blue_home) + [4.0, 0, 0], 0, 0, "SPRINT")
        time.sleep(0.05)
        a.send_state(_body(blue_home), 0, 0, "SPRINT")
        taken = _events(b, "flag", where=lambda e: e["action"] == "taken")
        assert taken and taken[0]["name"] == "Ann" and taken[0]["flag"] == T.BLUE
        assert wait_for(lambda: any(f["carrier"] == a.id for f in b.objectives["flags"])), "snapshots carry the flags"
        # run home (two updates: a long run is still a run, not a teleport)
        for p in np.linspace(_body(blue_home), _body(red_home), int(np.linalg.norm(red_home - blue_home) / 4.0) + 2):
            a.send_state(p, 0, 0, "SPRINT")
            time.sleep(0.004)
        seen = []
        assert wait_for(lambda: seen.extend(b.poll_events()) or any(e["t"] == "over" for e in seen), 3.0)
        cap = [e for e in seen if e["t"] == "flag" and e["action"] == "captured"]
        assert cap and cap[0]["team_scores"][T.RED] == 1 and cap[0]["name"] == "Ann"
        over = [e for e in seen if e["t"] == "over"]
        assert over[0]["team"] == T.RED and over[0]["teams_game"] and over[0]["teams"][str(a.id)] == T.RED
    finally:
        srv.stop()


def test_match_time_is_the_hosts_choice():
    rules = GameRules(GameRules.EMBLEMS, False, time_limit=1.0, emblems=4)
    srv = _server(bots=0, rules=rules, next_match_timeout=60.0)
    try:
        a = _client_ok(srv, "Ann")
        assert 0.0 < a.match["time_left"] <= 1.0
        over = _events(a, "over", timeout=4.0)
        assert over, "the round ended on the host's time limit"
    finally:
        srv.stop()


def test_hosted_session_settings_reach_the_server():
    w = small_world()
    hs = HostSettings(port=0, max_players=6, bots=2, mode="ctf", teams=True, minutes=3, capture_limit=4,
                      bot_level="hard")
    s = MultiplayerSession.host(w, hs, "Host", log=None, team=T.BLUE)
    try:
        st = s.server.status()
        assert st["max_players"] == 6 and st["bots"] == 2 and st["bot_level"] == "hard"
        assert all(b.level == "hard" and b.tactician.skill.name == "hard" for b in s.server.bots.values())
        assert st["rules"] == {"mode": "ctf", "teams": True, "time_limit": 180.0, "emblems": config.MP_DEFAULT_EMBLEMS,
                               "capture_limit": 4, "goal": config.ZOMBIE_DEFAULT_GOAL}
        assert s.team == T.BLUE and "you are on the Blue team" in s.describe_game()
        assert "Capture the flag - red vs blue - 3 min, first to 4" in s.describe_game()
        assert "1 player + 2 hard bots (6 max)" in s.describe()
    finally:
        s.leave()


# --------------------------------------------------------------- the session side
class _Player:
    def __init__(self):
        from parkour.controller import PlayerController
        w = small_world()
        self.pc = PlayerController(w.ledge_system, w.obstacles, spawn_pos=(0.0, 0.9, -15.0))


def test_results_and_notices_for_teams():
    class FakeClient:
        id = 1
        team = T.RED
        teams_game = True
        team_scores = {T.RED: 2, T.BLUE: 1}
        rules = {"mode": "ctf", "teams": True}
        mode = "ctf"
        players = {}
        match = {"round": 1}
        objectives = {"flags": [{"key": T.BLUE, "carrier": 1}], "zones": []}
        remaining = 0
        time_left = 42.0

        def scoreboard(self):
            return []
    s = MultiplayerSession(FakeClient())
    s.results = {"winners": [1, 3], "team": T.RED, "draw": False, "team_scores": {T.RED: 2, T.BLUE: 1},
                 "scores": {"1": 2, "2": 1, "3": 0}, "teams": {"1": T.RED, "2": T.BLUE, "3": T.RED},
                 "names": {"1": "Ann", "2": "Ben", "3": "Cy"}, "teams_game": True, "mode": "ctf", "timeout": 9.0}
    assert s.winner_text() == "The Red team wins 2 - 1 - your team wins!"
    view = s.results_view()
    assert [ln[2] for ln in view["lines"]] == [T.RED, T.RED, T.BLUE], "winners first, by team"
    assert view["lines"][0][1] and "Ann" in view["lines"][0][0]
    s.results.update(team=None, draw=True, team_scores={T.RED: 1, T.BLUE: 1})
    assert s.winner_text() == "Draw! 1 - 1"
    hud = s.hud_state()
    assert hud["carrying"] == T.BLUE and hud["teams"] and hud["team"] == T.RED and hud["time_left"] == 42.0
    n = s._flag_notice({"action": "captured", "flag": T.BLUE, "player": 1, "team": T.RED,
                        "team_scores": {T.RED: 3, T.BLUE: 1}})
    assert n == "You captured the Blue flag for the Red team!  (Red 3 - 1 Blue)"
    n = s._flag_notice({"action": "returned", "flag": T.RED, "player": 2, "name": "Ben", "reason": "tagged",
                        "by": 1, "by_name": "Ann"})
    assert n == "You tagged Ben - your flag goes home"
    n = s._flag_notice({"action": "taken", "flag": T.RED, "player": 2, "name": "Ben"})
    assert n == "Ben has your flag!"


# ---------------------------------------------------------------- bots playing
def _simulate(srv, seconds, until=None):
    srv.match.start({})
    srv._fill_bots()
    steps = int(seconds * 60)
    for i in range(steps):
        srv._tick(1.0 / 60.0)
        if until is not None and until(srv):
            return i / 60.0
    return None


def test_bots_play_capture_the_flag_in_teams():
    rules = GameRules(GameRules.CTF, True, time_limit=600.0, capture_limit=0)
    srv = GameServer(small_world(), port=0, max_players=5, bots=4, rules=rules, seed=11, log=None)
    events = []
    srv._broadcast = lambda msg, skip=None: events.append(msg)
    took = _simulate(srv, 240.0, until=lambda s: any(e.get("action") == "captured" for e in events))
    flags = [e for e in events if e.get("t") == "flag"]
    assert took is not None, f"no capture in 240 s: {flags[:6]}"
    acts = {e["action"] for e in flags}
    assert {"taken", "captured"} <= acts
    teams = [p.team for p in srv.players.values()]
    assert teams.count(T.RED) == 2 and teams.count(T.BLUE) == 2 and all(p.bot for p in srv.players.values())


def test_bots_play_capture_the_flag_free_for_all():
    rules = GameRules(GameRules.CTF, False, time_limit=600.0, capture_limit=0)
    srv = GameServer(small_world(), port=0, max_players=4, bots=3, rules=rules, seed=7, log=None)
    events = []
    srv._broadcast = lambda msg, skip=None: events.append(msg)
    took = _simulate(srv, 240.0, until=lambda s: any(e.get("action") == "captured" for e in events))
    assert took is not None, f"no capture in 240 s: {[e for e in events if e.get('t') == 'flag'][:6]}"
    assert any(e.get("t") == "flag" and e["action"] == "taken" for e in events)
    assert srv.match.scores and max(srv.match.scores.values()) >= 1


# ------------------------------------------------------------ the server process
def test_hosting_runs_the_server_in_a_process_of_its_own():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    hs = HostSettings(port=0, max_players=6, bots=2, mode="ctf", teams=True, minutes=5)
    launcher = MultiplayerSession.start_host(("mp_helpers", "small_world", []), hs, "Host", team=T.BLUE,
                                             log=None, extra_paths=[here])
    assert launcher.poll() is None or launcher.session is not None, "starts in the background"
    s = launcher.wait(60.0)
    try:
        proc = s.server
        assert proc.running and proc.proc.pid != os.getpid()
        assert s.client.team == T.BLUE and s.rules.mode == "ctf" and s.rules.teams
        guest = MultiplayerSession.join(f"127.0.0.1:{proc.port}", "Guest", team=T.RED)
        assert wait_for(lambda: proc.status()["humans"] == 2 and proc.status()["bots"] == 2, 5.0), proc.status()
        assert "Hosting on port" in s.describe() and "2 players + 2 complex bots (6 max)" in s.describe()
        assert wait_for(lambda: sum(1 for r in guest.scoreboard() if r[3]) == 2)
        guest.leave()
    finally:
        s.leave()
    assert not proc.running, "leaving stops the server process"
    # a port that is taken is reported, not a crash
    import socket
    blocker = socket.socket()
    blocker.bind(("0.0.0.0", 0))
    blocker.listen(1)
    try:
        bad = MultiplayerSession.start_host(("mp_helpers", "small_world", []),
                                            HostSettings(port=blocker.getsockname()[1]), "Host", log=None,
                                            extra_paths=[here])
        try:
            bad.wait(60.0)
            raise AssertionError("hosting on a taken port must fail")
        except ConnectionError as exc:
            assert "in use" in str(exc).lower() or "address" in str(exc).lower(), str(exc)
    finally:
        blocker.close()


# ------------------------------------------------------------------- visuals
def test_avatars_wear_team_colours_flip_roll_and_carry_flags():
    from render.mp_render import avatar_parts, flag_stand_parts, zone_ring_parts, shirt_color, flag_color
    base = {"p": [0.0, 0.9, 0.0], "yaw": 0.0, "st": "GROUND", "spd": 0.0, "n": None}
    red = shirt_color(3, team=T.RED)
    assert red[0] > 0.6 and red[2] < 0.3, "red team shirt"
    assert shirt_color(4, team=T.BLUE)[2] > 0.7, "blue team shirt"
    parts = avatar_parts(base, 0.0, pid=3, team=T.RED)
    assert any(np.allclose(col, red) for _, _, _, col, _ in parts)

    def head_y(parts):
        return float(parts[2][0][1])                          # the head box

    def hip_y(parts):
        return float(parts[0][0][1])
    upright = avatar_parts(base, 0.0)
    assert head_y(upright) > hip_y(upright) + 0.5
    flip = avatar_parts(dict(base, st="AIRBORNE", ac=["front", 180.0, 0.0, 0.0, -1.0]), 0.0)
    assert head_y(flip) < hip_y(flip), "upside down half way round a flip"
    full = avatar_parts(dict(base, st="AIRBORNE", ac=["front", 360.0, 0.0, 0.0, -1.0]), 0.0)
    assert head_y(full) > hip_y(full), "and upright at the end"
    roll = avatar_parts(dict(base, st="ROLL", ac=["roll", 90.0, 0.8, 0.0, -1.0]), 0.0)
    lowest = min(c[1] - 0.5 * float(np.max(np.abs(r) @ s)) for c, s, r, _, _ in roll)
    assert max(c[1] for c, *_ in roll) < 1.3 and lowest > -0.35, "a roll stays low, on the ground"
    carrying = avatar_parts(base, 0.0, carrying=T.BLUE)
    assert len(carrying) == len(upright) + 2 and any(np.allclose(c[3], flag_color(T.BLUE)) for c in carrying)
    stand = flag_stand_parts([0.0, 0.0, 0.0], "neutral", 0.0)
    assert max(c[1] + s[1] * 0.5 for c, s, *_ in stand) > 2.4, "a flag you can see over a car"
    ring = zone_ring_parts([0.0, 0.0, 0.0], 3.0, T.RED)
    assert all(abs(math.hypot(c[0], c[2]) - 3.0) < 1e-3 for c, *_ in ring)


def test_objective_markers_point_the_way():
    import types
    from math3d.transform import perspective, look_at
    from render.mp_render import MultiplayerRenderer
    eye = np.array([0.0, 1.6, 0.0], np.float32)
    view = look_at(eye, eye + np.array([0.0, 0.0, -1.0], np.float32), np.array([0.0, 1.0, 0.0], np.float32))
    proj = perspective(80.0, 16 / 9, 0.05, 500.0)
    flags = [{"key": T.RED, "team": T.RED, "home": [0.0, 0.0, -30.0], "pos": [0.0, 0.0, -30.0], "carrier": None},
             {"key": T.BLUE, "team": T.BLUE, "home": [0.0, 0.0, 40.0], "pos": [20.0, 0.9, 0.0], "carrier": 7}]
    zones = [{"key": T.RED, "team": T.RED, "pos": [0.0, 0.0, -30.0], "r": 3.0}]
    fake = types.SimpleNamespace(hud={"me": 1, "team": T.RED, "teams": True, "carrying": None}, cam_pos=eye,
                                 _objectives=lambda: (flags, zones))
    marks = MultiplayerRenderer.markers(fake, view, proj, 1280, 720)
    by = {m[2].split("  ")[0]: m for m in marks}
    assert set(by) == {"Your flag", "Blue flag"}
    x, y, text, col, on = by["Your flag"]
    assert on and abs(x - 640) < 5 and "30 m" in text, "straight ahead, on screen"
    x, y, text, col, on = by["Blue flag"]
    assert not on and x > 1100, "off to the right: pinned to the right edge"
    flags[0]["home"] = [-10.0, 0.0, 30.0]                       # behind you, to the left
    x, y, text, col, on = {m[2].split("  ")[0]: m for m in MultiplayerRenderer.markers(fake, view, proj, 1280, 720)}["Your flag"]
    assert not on and x < 640 and y > 600, "behind: along the bottom, on its side"
    # carrying: your base is marked too
    fake.hud["carrying"] = T.BLUE
    assert any(m[2].startswith("Your base") for m in MultiplayerRenderer.markers(fake, view, proj, 1280, 720))
