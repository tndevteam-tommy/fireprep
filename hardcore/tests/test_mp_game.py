"""Multiplayer game rules and the bot, headless: emblems, matches, navigation, the bot, the menu, avatars."""
import math

import numpy as np

from mp_helpers import *  # noqa
from game.match import Match, EmblemPlacer, EMBLEM_HOVER
from game.nav import NavGrid
from game.bot import BotBrain
from game.session import parse_mp_args, HostSettings
from core.mp_menu import MultiplayerMenuModel
from core.input_buffer import InputBuffer
from parkour.controller import PlayerController


# ----------------------------------------------------------------------- emblems
def test_emblems_are_spread_out_reachable_and_deterministic():
    w = small_world()
    a = EmblemPlacer(w).place(6, seed=11)
    b = EmblemPlacer(w).place(6, seed=11)
    c = EmblemPlacer(w).place(6, seed=12)
    assert [tuple(e.pos) for e in a] == [tuple(e.pos) for e in b], "same seed, same spots"
    assert [tuple(e.pos) for e in a] != [tuple(e.pos) for e in c]
    assert len(a) == 6 and {e.kind for e in a} >= {"roof", "street"}
    for i, e in enumerate(a):
        for f in a[i + 1:]:
            assert math.dist(e.pos, f.pos) >= EmblemPlacer.MIN_SPACING - 1e-3
        surface = float(e.pos[1]) - EMBLEM_HOVER
        assert w.standing_spot_free(float(e.pos[0]), surface, float(e.pos[2])), "a player fits where it floats"
        assert w.ground_below(float(e.pos[0]), float(e.pos[2]), surface + 0.01) is not None
        if e.kind == "roof":
            assert any(abs(surface - spec.height) < 1e-3 for spec, _ in w.buildings)


def test_pickups_are_swept_and_the_most_emblems_wins():
    m = Match(small_world(), emblem_count=4, seed=3)
    em = m.start([1, 2])
    e0, e1 = em[0], em[1]
    assert m.move_player(1, e0.pos + [0, 0, 3.0], e0.pos + [0, 0, 2.0]) == [], "not there yet"
    assert m.move_player(1, e0.pos + [0, 0, 2.0], e0.pos - [0, 0, 2.0]) == [e0], "ran straight through it"
    assert m.move_player(2, e0.pos, e0.pos) == [], "already taken"
    assert m.move_player(2, e1.pos + [40.0, 0, 0], e1.pos - [40.0, 0, 0]) == [], "a teleport is not a sweep"
    assert m.move_player(2, e1.pos, e1.pos) == [e1]
    assert m.remaining == 2 and m.scores == {1: 1, 2: 1}
    m.move_player(2, em[2].pos, em[2].pos)
    m.move_player(1, em[3].pos, em[3].pos)
    assert m.state == Match.OVER and m.winners == [1, 2], "a draw: both win"
    m.start([1, 2])
    m.move_player(2, m.emblems[0].pos, m.emblems[0].pos)
    for e in m.emblems[1:]:
        m.move_player(2, e.pos, e.pos)
    assert m.winners == [2] and m.round == 2


def test_matches_end_on_the_time_limit_and_nobody_wins_an_empty_round():
    m = Match(small_world(), emblem_count=4, seed=3, time_limit=10.0)
    m.start([1, 2])
    assert not m.tick(9.0) and m.state == Match.PLAYING
    m.move_player(2, m.emblems[0].pos, m.emblems[0].pos)
    assert m.tick(1.5) and m.state == Match.OVER and m.winners == [2]
    m.start([1])
    m.tick(11.0)
    assert m.winners == [], "no pickups: no winner"
    assert Match(small_world(), emblem_count=999).emblem_count == config.MP_EMBLEM_RANGE[1]


# ------------------------------------------------------------------------ nav
def test_nav_grid_blocks_buildings_and_paths_go_round_them():
    w = small_world()
    nav = w.nav
    spec = [s for s, _ in w.buildings if s.name == "bB"][0]
    ci, ck = nav.cell_of(spec.x + spec.width / 2, spec.z + spec.depth / 2)
    assert nav.blocked[ci, ck], "inside a building is blocked"
    i, k = nav.cell_of(0.0, 0.0)
    assert not nav.blocked[i, k], "the open ground is free"
    start = (spec.x + spec.width / 2, spec.z + spec.depth + 6.0)     # south of bB
    goal = nav.cell_of(spec.x + spec.width / 2, spec.z - 12.0)        # north, behind it and bD
    path = nav.path(start, {nav.nearest_free(*nav.center(*goal))})
    assert path, "there is a way round"
    pts = [start] + path
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 20):
            x, z = x0 + (x1 - x0) * t, z0 + (z1 - z0) * t
            inside = spec.x < x < spec.x + spec.width and spec.z < z < spec.z + spec.depth
            assert not inside, "the path never crosses the building"
    around = nav.cells_around_box(np.array([spec.x, 0, spec.z]), np.array([spec.x + spec.width, spec.height, spec.z + spec.depth]))
    assert around and all(not nav.blocked[c] for c in around)


# ------------------------------------------------------------------------ bot
def _run_bot(world, match, seconds, seed=0):
    t = [0.0]
    buf = InputBuffer(time_fn=lambda: t[0])
    pl = PlayerController(world.ledge_system, world.obstacles, spawn_pos=world.spawn_points[0], yaw=0.0)
    bot = BotBrain(world, pl, buf, seed=seed)
    prev = pl.pos.copy()
    states = set()
    for _ in range(int(seconds / DT)):
        t[0] += DT
        bot.update(DT, [e for e in match.emblems if e.taken_by is None])
        pl.step(DT, buf)
        states.add(pl.state)
        match.move_player(99, prev, pl.pos)
        prev = pl.pos.copy()
        if match.state == Match.OVER:
            break
    return bot, pl, states, t[0]


def test_the_bot_climbs_buildings_and_collects_every_emblem():
    for seed in (1, 4):
        m = Match(small_world(), emblem_count=6, seed=seed)
        m.start([99])
        bot, pl, states, took = _run_bot(small_world(), m, 120.0, seed)
        assert m.scores[99] == 6, f"seed {seed}: {m.scores[99]}/6 in {took:.0f}s"
        assert {ParkourState.LEDGE_HANG, ParkourState.CLIMB_UP, ParkourState.MANTLE_VAULT} <= states, "it climbed"
        assert pl.respawn_count == 0


def test_the_bot_wanders_building_to_building_without_emblems():
    w = small_world()
    m = Match(w, emblem_count=4, seed=2)
    m.start([99])
    for e in m.emblems:
        e.taken_by = 1                              # nothing to collect: free roaming
    bot, pl, states, took = _run_bot(w, m, 90.0, seed=3)
    assert ParkourState.CLIMB_UP in states and bot.climbs > 0, "climbs buildings"
    assert bot.wants_respawn is False and pl.respawn_count == 0, "never stuck for good"


def test_the_bot_only_climbs_what_it_is_after():
    w = small_world()
    pl = PlayerController(w.ledge_system, w.obstacles, spawn_pos=(0.0, 0.9, -15.0))
    bot = BotBrain(w, pl, InputBuffer())
    spec = [s for s, _ in w.buildings if s.name == "bD"][0]
    bot.goal = {"kind": "roof", "point": np.array([spec.x + 2, spec.height + 1, spec.z + 2], np.float32)}
    bot.goal["box"] = w.surface_box(spec.x + 2, spec.z + 2, spec.height)
    assert bot.goal["box"] is not None
    bb_core = [p for s, ps in w.buildings if s.name == "bB" for p in ps if p.name.endswith("core")][0]
    fake = type("C", (), {})()
    fake.wall_obstacle = bb_core
    bot.route = [(0.0, 0.0)]
    assert not bot._should_climb(fake), "a tall building in the way is gone round, not climbed"
    bd_core = [p for s, ps in w.buildings if s.name == "bD" for p in ps if p.name.endswith("core")][0]
    fake.wall_obstacle = bd_core
    assert bot._should_climb(fake), "the goal's own building is climbed"


# ------------------------------------------------------------------- menu & args
def test_multiplayer_menu_edits_values_and_asks_to_host_or_join():
    m = MultiplayerMenuModel()
    assert m.focused == "name"
    for _ in range(6):
        m.handle_key("backspace")
    for ch in "Ann!":
        m.handle_key(ch.lower(), ch)
    assert m.values["name"] == "Ann!"
    m.handle_key("down")                                   # your team: red <-> blue
    assert m.focused == "team"
    m.handle_key("right")
    assert m.values["team"] == "blue" and m.value_text("team") == "Blue"
    m.focus = m.rows().index("port")                       # port: digits only
    m.handle_key("a", "a")
    assert m.values["port"].isdigit()
    m.handle_key("down")
    m.handle_key("right")
    assert m.values["max_players"] == config.MP_MAX_PLAYERS, "capped at 15"
    m.handle_key("down")
    assert m.focused == "bots"
    for _ in range(30):
        m.handle_key("right")
    assert m.values["bots"] == config.MP_MAX_PLAYERS - 1, "bots: up to the player limit less the host's own slot"
    m.focus = m.rows().index("max_players")
    for _ in range(10):
        m.handle_key("left")
    assert m.values["max_players"] == 5 and m.values["bots"] == 4, "fewer players: fewer bots fit"
    m.focus = m.rows().index("bots")
    for _ in range(10):
        m.handle_key("left")
    assert m.values["bots"] == 0 and m.value_text("bots") == "0", "zero bots is no bots"
    m.handle_key("right")
    # the game: emblem hunt <-> capture the flag (the option row follows the game)
    m.focus = m.rows().index("game")
    assert "emblems" in m.rows() and "captures" not in m.rows()
    m.handle_key("right")
    assert m.values["game"] == "ctf" and m.value_text("game") == "Capture the flag"
    assert "captures" in m.rows() and "emblems" not in m.rows() and m.focused == "game"
    m.handle_key("down")
    assert m.focused == "teams"
    m.handle_key("return")
    assert m.values["teams"] is True and m.value_text("teams") == "Red vs Blue"
    m.handle_key("down")
    assert m.focused == "minutes"
    m.handle_key("left")
    assert m.values["minutes"] == 8 and m.value_text("minutes") == "8 min"
    for _ in range(6):
        m.handle_key("right")
    assert m.values["minutes"] == 60
    m.handle_key("right")
    assert m.values["minutes"] == 0 and m.value_text("minutes") == "Unlimited", "no time limit: the last choice"
    m.handle_key("right")
    assert m.values["minutes"] == 0
    m.handle_key("down")
    assert m.focused == "captures"
    for _ in range(10):
        m.handle_key("left")
    assert m.values["captures"] == 0 and m.value_text("captures") == "Off"
    m.focus = m.rows().index("host")
    action = m.handle_key("return")
    hs = action[1]
    assert action[0] == "host" and hs.max_players == 5 and hs.bots == 1
    assert hs.mode == "ctf" and hs.teams is True and hs.minutes == 0 and hs.capture_limit == 0
    rules = hs.rules()
    assert rules.mode == "ctf" and rules.teams and rules.unlimited and rules.never_ends()
    assert "only ends when you leave" in m.status, "the host is told such a round never ends"
    m.values["minutes"] = 60
    rules = m.handle_key("return")[1].rules()
    assert rules.time_limit == 3600.0 and not rules.unlimited
    m.values["port"] = "80"
    assert m.handle_key("return") is None and m.status_kind == "warn", "privileged port refused"
    m.focus = m.rows().index("join")
    assert m.handle_key("return") == ("join", m.values["address"])
    assert "leave" not in m.rows()
    m.in_session = True
    assert "leave" in m.rows()
    # in a game, changing your team asks to switch sides now
    m.focus = m.rows().index("team")
    assert m.handle_key("right") == ("team", "red")
    assert m.handle_nav("back") == ("back",)


def test_multiplayer_menu_settings_persist():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "multiplayer.json")
        m = MultiplayerMenuModel()
        m.values.update(name="Zed", emblems=20, bots=3, address="10.0.0.5:3000", game="ctf", teams=True,
                        minutes=15, captures=7, team="blue")
        assert m.save(path)
        n = MultiplayerMenuModel.load(path)
        for k in ("name", "emblems", "bots", "address", "game", "teams", "minutes", "captures", "team"):
            assert n.values[k] == m.values[k], k
        with open(path, "w") as f:
            f.write('{"emblems": 9999, "bot": false, "name": "", "game": "golf", "minutes": 7, "team": "green",'
                    ' "max_players": 4, "bots": 9}')
        n = MultiplayerMenuModel.load(path)
        assert n.values["emblems"] == config.MP_EMBLEM_RANGE[1] and n.values["name"] == "Player"
        assert n.values["game"] == "emblems" and n.values["minutes"] == 8 and n.values["team"] == "red"
        assert n.values["bots"] == 3, "never more bots than the player limit allows"
        with open(path, "w") as f:
            f.write('{"bot": true}')                                    # an old settings file
        assert MultiplayerMenuModel.load(path).values["bots"] == 1
        with open(path, "w") as f:
            f.write('{"bot": false}')
        assert MultiplayerMenuModel.load(path).values["bots"] == 0


def test_command_line_options():
    a = parse_mp_args(["--host", "--port", "3000", "--bots", "4", "--max-players", "8", "--emblems", "6", "--name", "Ann",
                       "--mode", "ctf", "--teams", "--time", "5", "--captures", "3", "--team", "blue"])
    assert a == {"host": True, "port": 3000, "bots": 4, "max_players": 8, "emblems": 6, "name": "Ann", "mode": "ctf",
                 "teams": True, "minutes": 5, "capture_limit": 3, "team": "blue"}
    assert parse_mp_args(["--join", "1.2.3.4:5", "--no-bot"]) == {"join": "1.2.3.4:5", "bots": 0}
    assert parse_mp_args(["--bot", "--ffa"]) == {"bots": 1, "teams": False}
    assert parse_mp_args(["--windowed", "--mode", "golf", "--team", "green"]) == {}
    m = MultiplayerMenuModel()
    m.load_values({"mode": "ctf", "teams": True, "minutes": 5, "capture_limit": 3, "bots": 4, "team": "blue"})
    assert (m.values["game"], m.values["teams"], m.values["minutes"], m.values["captures"], m.values["bots"],
            m.values["team"]) == ("ctf", True, 5, 3, 4, "blue")
    s = HostSettings(max_players=50, emblems=1, bots=99)
    assert s.max_players == config.MP_MAX_PLAYERS and s.emblems == config.MP_EMBLEM_RANGE[0]
    assert s.bots == config.MP_MAX_PLAYERS - 1
    assert HostSettings(max_players=4, bots=10).bots == 3 and HostSettings(bot=False).bots == 0


# ----------------------------------------------------------------------- avatars
def test_avatar_poses():
    from render.mp_render import avatar_parts, facing_yaw, emblem_transform
    base = {"p": [0.0, 0.9, 0.0], "yaw": 0.0, "st": "GROUND", "spd": 0.0, "n": None}
    parts = avatar_parts(base, 0.0)
    top = max(c[1] + s[1] * 0.5 for c, s, r, col, e in parts)
    bottom = min(c[1] - s[1] * 0.5 for c, s, r, col, e in parts)
    assert 1.65 < top - bottom < 2.0, "about a person tall"
    assert abs(bottom - 0.0) < 0.12, "standing on the ground"
    hang = dict(base, st="LEDGE_HANG", n=[0.0, 1.0])
    assert abs(facing_yaw(hang)) < 1e-6, "hanging: facing the wall (normal +z -> looking -z)"
    hp = avatar_parts(hang, 0.0)
    hands = max(c[1] + s[1] * 0.5 for c, s, r, col, e in hp)
    assert hands > top + 0.2, "arms up over the head"
    run0 = avatar_parts(dict(base, spd=8.0), 0.0)
    run1 = avatar_parts(dict(base, spd=8.0), 0.15)
    assert any(not np.allclose(a[0], b[0]) for a, b in zip(run0, run1)), "the legs move when running"
    bot = avatar_parts(base, 0.0, pid=3, bot=True)
    assert any(e == 1.0 for *_, e in bot), "the bot has its glowing visor"
    m0, m1 = emblem_transform([0, 1, 0], 0.0), emblem_transform([0, 1, 0], 0.6)
    assert not np.allclose(m0, m1), "emblems spin and bob"
    assert abs(m0[1, 3] - 1.0) <= 0.121 and abs(m1[1, 3] - 1.0) <= 0.121, "bobbing slightly"


# ------------------------------------------------------------------ bot robustness
def test_a_bot_whose_only_target_is_out_of_reach_asks_to_respawn():
    """Regression: giving up on a goal and picking the same one again (it was the only one left)
    counted as progress, so a bot stuck on its last target never asked to be respawned."""
    from game.match import Target
    w = small_world()
    t = [0.0]
    buf = InputBuffer(time_fn=lambda: t[0])
    pl = PlayerController(w.ledge_system, w.obstacles, spawn_pos=w.spawn_points[0], yaw=0.0)
    bot = BotBrain(w, pl, buf, seed=1)
    target = Target(1, np.array([0.0, 40.0, -15.0], dtype=np.float32), 1.0)   # floating high over open ground
    asked = None
    for _ in range(int((BotBrain.STUCK_RESPAWN + BotBrain.GOAL_TIMEOUT + 20.0) / DT)):
        t[0] += DT
        bot.update(DT, [target])
        pl.step(DT, buf)
        if bot.wants_respawn:
            asked = t[0]
            break
    assert asked is not None, "stuck on an unreachable target for good"


def test_a_bot_turns_into_a_narrow_alley_at_a_sprint_without_overshooting_it():
    """Regression (city seed 7, emblem seed 1): the route to an emblem turned 90 degrees into a 2.8 m
    alley between two buildings.  Arriving at a sprint, the bot carried on past the alley's mouth into
    the wall of the building beside it, then went back and forth along that wall for the rest of the
    round.  It now eases off before a sharp bend (and only counts the mouth as reached once the alley
    is in line)."""
    from game.match import World, Target
    ls = LedgeSystem()
    floor = Obstacle(-200.0, -0.5, -200.0, 400.0, 0.5, 400.0, add_crevasses=False)
    floor.rails.clear()
    south = Obstacle(47.5, 0.0, -141.0, 13.08, 5.7, 10.86)          # (the two city buildings, as built)
    north = Obstacle(47.5, 0.0, -127.34, 13.08, 11.7, 25.84)
    box = Obstacle(44.7, 0.0, -129.4, 1.6, 2.4, 4.0)               # the emblem sits on top of this
    obs = [floor, south, north, box]
    ls.build(obs)
    w = World(obs, ls, [], spawn_points=[(62.0, 0.9, -140.0)])
    goal = np.array([45.5, 3.4, -127.4], dtype=np.float32)
    for start in ((61.5, 0.9, -145.0), (62.0, 0.9, -140.0), (61.3, 0.9, -136.0), (61.5, 0.9, -112.0)):
        t = [0.0]
        buf = InputBuffer(time_fn=lambda: t[0])
        pl = PlayerController(w.ledge_system, w.obstacles, spawn_pos=start, yaw=0.0)
        bot = BotBrain(w, pl, buf, seed=1)
        target = Target(7, goal, config.MP_PICKUP_RADIUS)
        reached = False
        for _ in range(int(15.0 / DT)):
            t[0] += DT
            bot.update(DT, [target])
            pl.step(DT, buf)
            if float(np.linalg.norm(pl.pos - goal)) <= config.MP_PICKUP_RADIUS:
                reached = True
                break
        assert reached, f"from {start}: stuck at {np.round(pl.pos, 1)} ({bot.detours} detours)"
        assert bot.detours == 0, f"from {start}: ran into the building beside the alley ({bot.detours} detours)"


def test_a_bot_gets_onto_a_bus_shelter_roof_it_can_only_hop_up_to():
    """Regression (city seed 7, emblem seed 2): the last emblem floated over a bus-shelter roof (a thin
    slab at 2.42 m on two posts, a bench under it).  The bot ran in under the roof, and within 0.6 m
    of the emblem's spot just held W ('step onto it') for the rest of the round: nothing under a slab
    blocks you, so it never climbed.  Now it gets out from under it and hops for the edge."""
    from game.match import World, Target
    from parkour.ledge_rail import LedgeRail, RailType

    def box(x, y, z, w, h, d, holds=False):
        o = Obstacle(x, y, z, w, h, d, add_crevasses=False)
        if not holds:
            o.rails.clear()
        return o
    floor = box(-200.0, -0.5, -200.0, 400.0, 0.5, 400.0)
    walk = box(141.0, 0.0, 67.0, 3.0, 0.15, 40.0)
    posts = [box(142.42, 0.0, 100.14, 0.16, 2.3, 0.16), box(142.42, 0.0, 103.74, 0.16, 2.3, 0.16)]
    bench = box(142.25, 0.25, 101.12, 0.5, 0.25, 1.8)
    roof = box(141.7, 2.3, 100.02, 1.6, 0.12, 4.0)
    y = 2.42
    for p1, p2, n in (((141.7, y, 104.02), (143.3, y, 104.02), (0, 0, 1)), ((143.3, y, 100.02), (141.7, y, 100.02), (0, 0, -1)),
                      ((143.3, y, 104.02), (143.3, y, 100.02), (1, 0, 0)), ((141.7, y, 100.02), (141.7, y, 104.02), (-1, 0, 0))):
        roof.rails.append(LedgeRail(list(p1), list(p2), list(n), RailType.MANTLE_TOP, owner=roof))
    obs = [floor, walk] + posts + [bench, roof]
    ls = LedgeSystem()
    ls.build(obs)
    w = World(obs, ls, [], spawn_points=[(130.0, 0.9, 95.0)])
    goal = np.array([142.5, 3.42, 102.0], dtype=np.float32)
    for start in ((130.0, 0.9, 95.0), (141.9, 1.1, 102.0), (146.0, 0.9, 110.0)):
        t = [0.0]
        buf = InputBuffer(time_fn=lambda: t[0])
        pl = PlayerController(w.ledge_system, w.obstacles, spawn_pos=start, yaw=0.0)
        bot = BotBrain(w, pl, buf, seed=1)
        target = Target(6, goal, config.MP_PICKUP_RADIUS)
        reached = False
        for _ in range(int(12.0 / DT)):
            t[0] += DT
            bot.update(DT, [target])
            pl.step(DT, buf)
            if float(np.linalg.norm(pl.pos - goal)) <= config.MP_PICKUP_RADIUS:
                reached = True
                break
        assert reached, f"from {start}: stuck at {np.round(pl.pos, 1)}"
