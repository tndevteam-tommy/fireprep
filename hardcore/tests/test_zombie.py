"""Zombie tag (one living player scoring by the second, everyone else a slower zombie; a catch swaps
them over) and games with no time limit - the rules headless, the movement, then over real sockets,
the screens, and bots playing it."""
import json
import time
import types

import numpy as np

from mp_helpers import *  # noqa
from game.match import GameRules, ZombieMatch, create_match
from game.session import MultiplayerSession, HostSettings, parse_mp_args
from core.mp_menu import MultiplayerMenuModel
from net.server import GameServer
from net.client import GameClient
from net.protocol import encode

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _zm(goal=120, time_limit=0.0, seed=3, players=(1, 2, 3)):
    m = create_match(small_world(), GameRules(GameRules.ZOMBIE, False, time_limit, goal=goal), seed=seed)
    m.start(list(players))
    m.drain_events()
    return m


def _place(m, pid, pos):
    m.apply_move(pid, pos, pos)


def _past_grace(m):
    m.tick(m.grace + 0.01)


# ------------------------------------------------------------------------ rules
def test_zombie_rules_are_every_player_for_themself():
    r = GameRules("zombie", True, 0, goal=9999)
    assert r.mode == "zombie" and not r.teams, "no teams in zombie tag"
    assert r.goal == config.ZOMBIE_GOAL_RANGE[1] and GameRules("zombie", goal=1).goal == config.ZOMBIE_GOAL_RANGE[0]
    assert r.unlimited and r.describe() == f"Zombie tag - every player for themself - no time limit, first to {r.goal} points"
    assert GameRules.from_dict(r.to_dict()).to_dict() == r.to_dict()
    assert r.point_word == "point"
    assert isinstance(create_match(small_world(), r), ZombieMatch)


def test_one_random_player_starts_alive_away_from_the_zombies():
    livings = set()
    for seed in range(8):
        m = _zm(seed=seed, players=(1, 2, 3, 4))
        assert m.living in (1, 2, 3, 4)
        livings.add(m.living)
        start = np.mean(np.array(small_world().spawn_points), axis=0)
        lp = np.asarray(m.spawn_point(m.living))
        d = float(np.hypot(*(lp - start)[[0, 2]]))
        lo, hi = config.ZOMBIE_LIVING_SPAWN_DISTANCE
        assert lo - 0.01 <= d <= hi + 0.01, "the living player starts well away from the zombies' start line"
        assert all(float(np.hypot(*(np.asarray(m.spawn_point(z)) - start)[[0, 2]])) < 15.0
                   for z in (1, 2, 3, 4) if z != m.living)
    assert len(livings) >= 3, "chosen at random"


def test_the_living_player_scores_by_the_second():
    m = _zm(goal=60)
    alive = m.living
    for _ in range(30):
        m.tick(1.0)
    assert m.scores[alive] == 30 and abs(m.points[alive] - 30.0) < 1e-6
    assert all(m.scores[p] == 0 for p in m.players if p != alive), "zombies score nothing"
    snap = json.loads(encode(m.snapshot()))
    assert snap["zombie"]["living"] == alive and snap["zombie"]["goal"] == 60


def test_a_catch_swaps_the_living_player_and_the_zombie():
    m = _zm()
    alive = m.living
    zombies = [p for p in m.players if p != alive]
    here = np.array([0.0, 0.9, -10.0], np.float32)
    _place(m, alive, here)
    # during the start grace nobody can be caught
    assert m.grace > 0 and m.apply_move(zombies[0], here + [3.0, 0, 0], here) == []
    _past_grace(m)
    m.tick(10.0)
    ev = m.apply_move(zombies[0], here + [3.0, 0, 0], here + [0.5, 0, 0])
    assert ev and ev[0]["action"] == "caught" and ev[0]["player"] == zombies[0] and ev[0]["from"] == alive
    assert m.living == zombies[0] and m.role(alive) == "zombie" and m.role(zombies[0]) == "living"
    before = dict(m.scores)
    m.tick(5.0)
    assert m.scores[zombies[0]] == before[zombies[0]] + 5 and m.scores[alive] == before[alive], \
        "the catcher scores now, the caught one keeps what they had"
    # right after a catch: a head start (no catching), and no tag-back for longer
    new = m.living
    np_ = here + [0.5, 0, 0]
    m.grace = 0.0                                        # (head start over)
    assert m.apply_move(alive, np_ + [1.0, 0, 0], np_) == [], "no tag-backs"
    other = zombies[1]
    ev = m.apply_move(other, np_ + [2.0, 0, 0], np_ + [0.2, 0, 0])
    assert ev and ev[0]["player"] == other and m.living == other, "another zombie can catch at once"
    # the living player running into a zombie is caught too; tag-backs are fine after the window
    m.elapsed += config.ZOMBIE_TAGBACK_GRACE + 0.1
    m.grace = 0.0
    _place(m, alive, np_ + [20.0, 0, 0])
    ev = m.apply_move(other, np_ + [10.0, 0, 0], np_ + [19.5, 0, 0])
    assert ev and ev[0]["player"] == alive and ev[0]["from"] == other and m.living == alive


def test_no_tag_back_for_a_while_even_with_two_players():
    m = _zm(players=(1, 2))
    a = m.living
    b = 2 if a == 1 else 1
    p = np.array([0.0, 0.9, -10.0], np.float32)
    _place(m, a, p)
    _past_grace(m)
    m.apply_move(b, p + [2.0, 0, 0], p)
    assert m.living == b
    m.grace = 0.0
    assert m.apply_move(a, p + [1.0, 0, 0], p) == [], "no tag-backs"
    m.tick(config.ZOMBIE_TAGBACK_GRACE + 0.1)
    assert m.apply_move(a, p + [1.0, 0, 0], p) and m.living == a, "after the window it is fair game again"


def test_the_goal_or_the_clock_ends_the_round():
    m = _zm(goal=30)
    alive = m.living
    for _ in range(40):
        if m.tick(1.0):
            break
    assert m.state == m.OVER and m.winners == [alive] and m.scores[alive] == 30, "first to the goal wins"
    m = _zm(goal=600, time_limit=20.0)
    first = m.living
    _place(m, first, np.array([0.0, 0.9, -10.0], np.float32))
    m.tick(12.0)
    z = [p for p in m.players if p != first][0]
    m.apply_move(z, np.array([2.0, 0.9, -10.0], np.float32), np.array([0.0, 0.9, -10.0], np.float32))
    m.tick(5.0)
    ended = m.tick(5.0)
    assert ended and m.state == m.OVER and m.winners == [first], "time up: the most points wins"
    assert m.results()["catches"] == {str(z): 1}


def test_the_living_player_leaving_hands_life_to_someone_else():
    m = _zm(players=(1, 2, 3))
    alive = m.living
    ev = m.remove_player(alive)
    living_ev = [e for e in ev if e.get("action") == "living"]
    assert living_ev and living_ev[0]["reason"] == "left" and m.living in m.players and m.living != alive
    assert m.grace >= config.ZOMBIE_CATCH_GRACE
    for p in list(m.players):
        m.remove_player(p)
    assert m.living is None
    m.add_player(9)
    assert m.living == 9 and m.drain_events()[-1]["player"] == 9, "someone joining an empty game is alive"


def test_bots_chase_the_living_player_or_run_from_the_zombies():
    m = _zm(players=(1, 2, 3))
    alive = m.living
    zs = [p for p in m.players if p != alive]
    here = np.array([0.0, 0.9, -10.0], np.float32)
    _place(m, alive, here)
    _place(m, zs[0], here + [5.0, 0, 0])
    _place(m, zs[1], here + [8.0, 0, 3.0])
    t = m.bot_targets(zs[0])
    assert t and t[0].id == f"player:{alive}" and t[0].kind == "player"
    flee = m.bot_targets(alive)
    assert flee and flee[0].id.startswith("flee:")
    goal = flee[0].pos
    assert float(np.dot((goal - here)[[0, 2]], (here - (here + [5.0, 0, 0]))[[0, 2]])) > 0, "away from the nearest zombie"
    d_goal = min(float(np.linalg.norm((goal - z)[[0, 2]])) for z in (here + [5.0, 0, 0], here + [8.0, 0, 3.0]))
    assert d_goal > 10.0
    assert m.bot_targets(alive)[0].id == flee[0].id, "keeps its goal a while (no dithering)"


# ------------------------------------------------------------------- movement
def test_zombies_run_and_climb_a_little_slower():
    assert 0.8 <= config.ZOMBIE_RUN_SCALE < 1.0 and 0.8 <= config.ZOMBIE_CLIMB_SCALE < 1.0

    def sprint(scale):
        ls, obs = build_custom([])
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 50.0), yaw=0.0)
        sim.player.set_speed_scales(*scale)
        sim.hold("CAPS", "W")
        sim.run(5.0)
        return float(np.hypot(sim.vel[0], sim.vel[2]))
    living, zombie = sprint((1.0, 1.0)), sprint((config.ZOMBIE_RUN_SCALE, config.ZOMBIE_CLIMB_SCALE))
    assert abs(zombie / living - config.ZOMBIE_RUN_SCALE) < 0.02 and living > zombie + 0.6

    def climb(scale):
        ls, obs = build_custom([(-10.0, 0.0, -5.0, 20.0, 5.4, 4.0)])
        sim = Sim(ls, obs, spawn=(0.0, 0.9, 1.2), yaw=0.0)
        sim.player.set_speed_scales(*scale)
        sim.hold("W", "E")
        assert sim.run(6.0, until=lambda s: s.state == ParkourState.CLIMB_UP)
        t0 = sim.time
        sim.run(4.0, until=lambda s: s.state != ParkourState.CLIMB_UP)
        return sim.time - t0
    t_living, t_zombie = climb((1.0, 1.0)), climb((config.ZOMBIE_RUN_SCALE, config.ZOMBIE_CLIMB_SCALE))
    assert abs(t_living / t_zombie - config.ZOMBIE_CLIMB_SCALE) < 0.05, (t_living, t_zombie)


# ---------------------------------------------------------------- no time limit
def test_every_game_can_run_without_a_time_limit():
    w = small_world()
    for mode in GameRules.MODES:
        r = GameRules(mode, False, 0, goal=600, capture_limit=3)
        assert r.unlimited and "no time limit" in r.describe()
        m = create_match(w, r, seed=2)
        m.start([1, 2])
        if mode == GameRules.ZOMBIE:
            assert not m.tick(500.0), "no clock: only the goal ends zombie tag"
        else:
            for _ in range(50):
                assert not m.tick(100.0)
        assert m.state == m.PLAYING and m.time_left is None
        json.loads(encode({"t": "snap", "time_left": m.time_left, "match": m.snapshot()}))
    assert GameRules("ctf", False, 0, capture_limit=0).never_ends()
    assert not GameRules("ctf", False, 0, capture_limit=2).never_ends() and not GameRules("emblems", False, 0).never_ends()
    assert HostSettings(minutes=0).rules().unlimited and HostSettings(minutes=5).rules().time_limit == 300.0


# --------------------------------------------------------------------- network
def _server(**kw):
    kw.setdefault("port", 0)
    kw.setdefault("seed", 5)
    kw.setdefault("log", None)
    srv = GameServer(small_world(), host="127.0.0.1", **kw)
    srv.start()
    return srv


class _Player:
    def __init__(self):
        from parkour.controller import PlayerController
        w = small_world()
        self.pc = PlayerController(w.ledge_system, w.obstacles, spawn_pos=(30.0, 0.9, 30.0))


def test_zombie_tag_over_the_network():
    rules = GameRules(GameRules.ZOMBIE, False, 0, goal=300)
    srv = _server(bots=0, rules=rules)
    try:
        a = GameClient("127.0.0.1", srv.port, "Ann").connect()
        assert a.mode == "zombie" and a.living == a.id, "alone: you are the living player"
        sa = np.array(a.spawn)
        a.send_state(sa, 0, 0, "GROUND")
        b = GameClient("127.0.0.1", srv.port, "Ben").connect()
        assert b.living == a.id and b.time_left is None, "Ben joins as a zombie; no time limit"
        assert wait_for(lambda: a.scoreboard()[0][0] >= 1 and a.scoreboard()[0][5] == a.id, 4.0), "Ann scores"
        # Ben walks up to Ann (after the start grace) and catches her
        srv.match.grace = 0.0
        here = sa + [0.0, 0.0, 0.0]
        a.send_state(here, 0, 0, "GROUND")
        time.sleep(0.1)
        for k in range(6, -1, -1):
            b.send_state(here + [k * 0.8, 0.0, 0.0], 0, 0, "SPRINT")
            time.sleep(0.01)
        seen = []
        assert wait_for(lambda: seen.extend(a.poll_events()) or any(e.get("action") == "caught" for e in seen), 3.0)
        ev = [e for e in seen if e.get("t") == "zombie" and e.get("action") == "caught"][0]
        assert ev["action"] == "caught" and ev["name"] == "Ben" and ev["from_name"] == "Ann"
        assert wait_for(lambda: a.living == b.id and b.living == b.id)
    finally:
        srv.stop()


def test_the_session_slows_a_zombie_and_says_who_is_who():
    w = small_world()
    s = MultiplayerSession.host(w, HostSettings(port=0, bots=0, mode="zombie", minutes=0), "Host", log=None)
    try:
        me = _Player().pc
        notes = []
        assert wait_for(lambda: notes.extend(s.update(0.05, me)) or s.client.living is not None)
        assert s.client.living == s.client.id and me.fsm.run_scale == 1.0, "alive: full speed"
        guest = MultiplayerSession.join(f"127.0.0.1:{s.server.port}", "Guest")
        gp = _Player().pc
        assert wait_for(lambda: (guest.update(0.05, gp) or True) and gp.fsm.run_scale == config.ZOMBIE_RUN_SCALE)
        assert gp.fsm.climb_scale == config.ZOMBIE_CLIMB_SCALE, "a zombie is slower"
        hud = guest.hud_state(gp.pos)
        assert hud["zombie"]["living"] == s.client.id and not hud["zombie"]["alive"]
        assert wait_for(lambda: (guest.update(0.05, gp) or True) and guest.hud_state(gp.pos)["chase"], 3.0)
        chase = guest.hud_state(gp.pos)["chase"]
        assert chase[0]["kind"] == "living" and "Host" in chase[0]["label"], "the zombie is shown where the living player is"
        me_hud = s.hud_state(me.pos)
        assert me_hud["zombie"]["alive"] and me_hud["time_left"] is None
        n = s._zombie_notice({"action": "caught", "player": s.client.id, "from": guest.client.id,
                              "name": "Host", "from_name": "Guest"})
        assert n == "You caught Guest - you're ALIVE! Run!"
        n = guest._zombie_notice({"action": "caught", "player": s.client.id, "from": guest.client.id,
                                  "name": "Host", "from_name": "Guest"})
        assert n.startswith("Host caught you - you're a zombie now!")
        s.results = {"winners": [s.client.id], "scores": {str(s.client.id): 120}, "names": {str(s.client.id): "Host"},
                     "mode": "zombie", "teams_game": False}
        assert s.winner_text() == "You win with 120 points!"
        guest.leave()
    finally:
        s.leave()


def test_server_bots_that_are_zombies_are_slower():
    rules = GameRules(GameRules.ZOMBIE, False, 0, goal=600)
    srv = GameServer(small_world(), port=0, max_players=4, bots=3, rules=rules, seed=4, log=None)
    srv._broadcast = lambda msg, skip=None: None
    srv.match.start({})
    srv._fill_bots()
    srv._tick(1.0 / 60.0)
    living = srv.match.living
    for pid, bot in srv.bots.items():
        want = 1.0 if pid == living else config.ZOMBIE_RUN_SCALE
        assert bot.controller.fsm.run_scale == want


def test_bots_play_zombie_tag():
    rules = GameRules(GameRules.ZOMBIE, False, 0, goal=600)
    # (simple bots: the flow of the game - life passing on - not how hard a clever runner is to catch)
    srv = GameServer(small_world(), port=0, max_players=5, bots=4, rules=rules, seed=4, log=None,
                     bot_level="simple")
    events = []
    srv._broadcast = lambda msg, skip=None: events.append(msg)
    srv.match.start({})
    srv._fill_bots()
    first = srv.match.living
    for _ in range(int(150 * 60)):
        srv._tick(1.0 / 60.0)
        if sum(1 for e in events if e.get("t") == "zombie" and e["action"] == "caught") >= 2:
            break
    caught = [e for e in events if e.get("t") == "zombie" and e["action"] == "caught"]
    assert len(caught) >= 2, f"the zombies catch the living player ({len(caught)} catches in 150 s)"
    assert caught[0]["from"] == first and caught[1]["from"] == caught[0]["player"], "life passes on"
    assert sum(1 for p, v in srv.match.points.items() if v > 1.0) >= 2, "more than one player has scored"


# ----------------------------------------------------------------- screens
def test_the_menu_offers_zombie_tag_a_goal_and_no_time_limit():
    m = MultiplayerMenuModel()
    m.focus = m.rows().index("game")
    m.handle_key("right")
    m.handle_key("right")
    assert m.values["game"] == "zombie" and m.value_text("game") == "Zombie tag"
    rows = m.rows()
    assert "goal" in rows and "teams" not in rows and "emblems" not in rows and "captures" not in rows
    m.focus = rows.index("goal")
    m.handle_key("right")
    assert m.values["goal"] == config.ZOMBIE_DEFAULT_GOAL + config.ZOMBIE_GOAL_STEP and m.value_text("goal") == "150"
    for _ in range(50):
        m.handle_key("left")
    assert m.values["goal"] == config.ZOMBIE_GOAL_RANGE[0]
    m.values["minutes"] = 0
    assert m.value_text("minutes") == "Unlimited"
    m.focus = m.rows().index("host")
    hs = m.handle_key("return")[1]
    r = hs.rules()
    assert r.mode == "zombie" and r.goal == config.ZOMBIE_GOAL_RANGE[0] and r.unlimited and not r.teams
    m.load_values({"game": "zombie", "goal": 95, "minutes": 0})
    assert m.values["goal"] == 90 and m.values["minutes"] == 0
    a = parse_mp_args(["--host", "--mode", "zombie", "--goal", "240", "--time", "0"])
    assert a == {"host": True, "mode": "zombie", "goal": 240, "minutes": 0}


def test_zombies_look_like_zombies():
    from render.mp_render import avatar_parts, ZOMBIE_SKIN, ZOMBIE_EYES, SKIN
    base = {"p": [0.0, 0.9, 0.0], "yaw": 0.0, "st": "GROUND", "spd": 3.0, "n": None}
    person = avatar_parts(base, 0.0, pid=2)
    zombie = avatar_parts(base, 0.0, pid=2, zombie=True)
    assert any(np.allclose(p[3], SKIN) for p in person) and not any(np.allclose(p[3], ZOMBIE_SKIN) for p in person)
    assert sum(1 for p in zombie if np.allclose(p[3], ZOMBIE_SKIN)) >= 3, "green head and hands"
    assert sum(1 for p in zombie if np.allclose(p[3], ZOMBIE_EYES) and p[4] > 0.5) == 2, "glowing eyes"
    # arms out in front (facing -z): the hands well ahead of the chest, at about shoulder height
    hands = [p for p in zombie if np.allclose(p[3], ZOMBIE_SKIN)][-2:]
    assert all(h[0][2] < -0.35 and h[0][1] > 1.1 for h in hands), [h[0] for h in hands]
    # shambling: the body lurches from side to side over the stride
    torsos = [avatar_parts(base, t, pid=2, zombie=True)[1][0] for t in np.linspace(0.0, 2.0, 9)]
    assert max(t[0] for t in torsos) - min(t[0] for t in torsos) > 0.05


def test_hud_marks_the_living_player_or_the_nearest_zombies():
    from math3d.transform import perspective, look_at
    from render.mp_render import MultiplayerRenderer
    eye = np.array([0.0, 1.6, 0.0], np.float32)
    view = look_at(eye, eye + np.array([0.0, 0.0, -1.0], np.float32), np.array([0.0, 1.0, 0.0], np.float32))
    proj = perspective(80.0, 16 / 9, 0.05, 500.0)
    hud = {"me": 1, "teams": False, "carrying": None, "mode": "zombie",
           "chase": [{"pos": [0.0, 0.9, -25.0], "label": "ALIVE: Ann", "kind": "living"}]}
    fake = types.SimpleNamespace(hud=hud, cam_pos=eye, _objectives=lambda: ([], []))
    marks = MultiplayerRenderer.markers(fake, view, proj, 1280, 720)
    assert len(marks) == 1 and marks[0][2].startswith("ALIVE: Ann") and marks[0][4]


def test_a_zombie_sees_its_own_arms_reaching_ahead():
    ls, obs = build_custom([])
    sim = Sim(ls, obs, spawn=(0.0, 0.9, 20.0), yaw=0.0, with_arms=True)
    sim.hold("W")
    sim.run(1.0)
    L = sim.arm_rig.last
    person = [float(np.dot(L[s + "_wrist"] - L[s + "_shoulder"], L["forward_h"])) for s in ("left", "right")]
    sim.zombie = True
    for _ in range(int(1.5 / DT)):
        sim.step()
        L = sim.arm_rig.last
        for s in ("left", "right"):
            assert abs(np.linalg.norm(L[s + "_elbow"] - L[s + "_shoulder"]) - config.UPPER_ARM_LEN) < 1e-3
            assert abs(np.linalg.norm(L[s + "_wrist"] - L[s + "_elbow"]) - config.FOREARM_LEN) < 1e-3
            assert config.ELBOW_MIN_ANGLE_DEG - 0.01 <= L[s + "_elbow_angle"] <= config.ELBOW_MAX_ANGLE_DEG + 0.01
    zombie = [float(np.dot(L[s + "_wrist"] - L[s + "_shoulder"], L["forward_h"])) for s in ("left", "right")]
    assert min(zombie) > 0.4 and min(zombie) > max(person) + 0.15, (person, zombie)
    assert all(L[s + "_wrist"][1] > L[s + "_shoulder"][1] - 0.3 for s in ("left", "right")), "held up, not hanging"
