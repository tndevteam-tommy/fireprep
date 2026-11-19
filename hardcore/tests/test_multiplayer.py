"""Multiplayer over real sockets: hosting, joining, snapshots, emblems, rounds, limits, robustness."""
import json
import socket
import time

import numpy as np

from mp_helpers import *  # noqa
from net.server import GameServer
from net.client import GameClient
from net.protocol import encode, LineReader, ProtocolError, PROTOCOL_VERSION, clean_name, finite_vec
from game.session import MultiplayerSession, HostSettings, parse_address


def _server(**kw):
    kw.setdefault("port", 0)
    kw.setdefault("emblems", 4)
    kw.setdefault("seed", 5)
    kw.setdefault("log", None)
    srv = GameServer(small_world(), host="127.0.0.1", **kw)
    srv.start()
    return srv


def _client(srv, name):
    return GameClient("127.0.0.1", srv.port, name).connect()


def _events(client, kind, timeout=3.0):
    got = []

    def check():
        got.extend(e for e in client.poll_events() if e.get("t") == kind)
        return bool(got)
    wait_for(check, timeout)
    return got


# --------------------------------------------------------------------- protocol
def test_protocol_lines_are_parsed_defensively():
    r = LineReader(max_line=200)
    msgs = r.feed(encode({"t": "a", "x": 1}) + b"not json\n[1,2]\n" + b'{"no_t": 1}\n' + encode({"t": "b"})[:5])
    assert [m["t"] for m in msgs] == ["a"] and r.dropped == 3
    assert [m["t"] for m in r.feed(encode({"t": "b"})[5:])] == ["b"], "a message split across reads"
    try:
        r.feed(b"x" * 500)
        raise AssertionError("an endless line must be refused")
    except ProtocolError:
        pass
    assert finite_vec([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert finite_vec([1, float("nan"), 3]) is None and finite_vec([1, 2]) is None and finite_vec("abc") is None
    assert clean_name("  Ann\x00ie  ") == "Annie" and clean_name("") == "Player" and len(clean_name("x" * 99)) == 20


def test_addresses_parse():
    assert parse_address("example.org") == ("example.org", config.MP_DEFAULT_PORT)
    assert parse_address("10.0.0.2:4000") == ("10.0.0.2", 4000)
    assert parse_address("[::1]:4001") == ("::1", 4001)
    for bad in ("", "host:0", "host:99999", ":123"):
        try:
            parse_address(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


# --------------------------------------------------------------------- sessions
def test_host_and_join_see_each_other_and_the_bot():
    srv = _server(bot=True)
    try:
        a, b = _client(srv, "Ann"), _client(srv, "Ben")
        assert a.id != b.id and a.emblems and set(a.emblems) == set(b.emblems), "same emblems for everyone"
        a.send_state([1.0, 0.9, -15.0], 90.0, 0.0, "GROUND")
        b.send_state([4.0, 0.9, -15.0], 0.0, 0.0, "SPRINT", speed=6.0)
        bot_name = srv.bot.info.name
        assert bot_name.endswith("(bot)")
        assert wait_for(lambda: {p.name for p, _ in a.remote_poses(time.monotonic() + 1.0)} >= {"Ben", bot_name})
        pose = dict((p.name, q) for p, q in a.remote_poses(time.monotonic() + 1.0))
        assert abs(pose["Ben"]["p"][0] - 4.0) < 1e-3 and pose["Ben"]["st"] == "SPRINT"
        assert "Ann" not in pose, "we are not drawn as a remote player"
        bot_pose = pose[bot_name]
        assert wait_for(lambda: dict((p.name, q) for p, q in a.remote_poses(time.monotonic() + 1.0))[bot_name]["p"]
                        != bot_pose["p"], 3.0), "the bot moves about"
        assert srv.status()["humans"] == 2 and srv.status()["bot"]
        b.close()
        assert _events(a, "leave")
    finally:
        srv.stop()


def test_emblems_are_picked_up_by_touching_them_and_the_most_wins():
    srv = _server(bot=False, emblems=4)
    try:
        a, b = _client(srv, "Ann"), _client(srv, "Ben")
        em = sorted(a.emblems.values(), key=lambda e: e["id"])
        assert len(em) == 4
        a.send_state(em[0]["pos"], 0, 0, "GROUND")
        # Ben sweeps through an emblem between two updates (a fast run): still picked up
        p1 = np.array(em[1]["pos"]) + [-2.5, 0, 0]
        b.send_state(p1, 0, 0, "SPRINT")
        b.send_state(p1 + [5.0, 0, 0], 0, 0, "SPRINT")
        picks = _events(a, "pickup")
        assert wait_for(lambda: len(picks) >= 2 or picks.extend(e for e in a.poll_events() if e["t"] == "pickup"), 3.0)
        assert {(e["emblem"], e["name"]) for e in picks} == {(em[0]["id"], "Ann"), (em[1]["id"], "Ben")}
        a.send_state(em[2]["pos"], 0, 0, "GROUND")
        time.sleep(0.05)
        a.send_state(em[3]["pos"], 0, 0, "GROUND")
        over = _events(a, "over")
        assert over and over[0]["winners"] == [a.id] and over[0]["scores"][str(a.id)] == 3
        assert b.remaining == 0 or wait_for(lambda: b.remaining == 0)
    finally:
        srv.stop()


def test_everyone_joining_the_next_game_starts_a_new_round_with_new_emblems():
    srv = _server(bot=False, emblems=4)
    try:
        a, b = _client(srv, "Ann"), _client(srv, "Ben")
        first = {e["id"]: tuple(e["pos"]) for e in a.emblems.values()}
        for i, e in enumerate(sorted(a.emblems.values(), key=lambda e: e["id"])):
            (a if i % 2 == 0 else b).send_state(e["pos"], 0, 0, "GROUND")
            time.sleep(0.03)
        assert _events(a, "over")
        a.send_ready()
        status = _events(b, "ready_status")
        assert status and status[-1]["ready"] == [a.id] and status[-1]["needed"] == 2
        assert not _events(a, "start", timeout=0.3), "waits for everyone"
        b.send_ready()
        starts = _events(b, "start")
        assert starts and starts[0]["spawn"] is not None
        assert {e["id"]: tuple(e["pos"]) for e in b.emblems.values()} != first, "new spots each round"
        assert all(e["taken_by"] is None for e in b.emblems.values()) and b.match["round"] == 2
    finally:
        srv.stop()


def test_next_game_starts_on_its_own_if_someone_never_answers():
    srv = _server(bot=False, emblems=4, next_match_timeout=0.6)
    try:
        a = _client(srv, "Ann")
        for e in sorted(a.emblems.values(), key=lambda e: e["id"]):
            a.send_state(e["pos"], 0, 0, "GROUND")
            time.sleep(0.03)
        assert _events(a, "over")
        assert _events(a, "start", timeout=3.0)
    finally:
        srv.stop()


def test_the_bot_collects_emblems_in_a_hosted_game():
    srv = _server(bot=True, emblems=4)
    try:
        a = _client(srv, "Ann")
        got = []
        wait_for(lambda: got.extend(e for e in a.poll_events() if e["t"] == "pickup" and e["name"].endswith("(bot)"))
                 or len(got) >= 1,
                 timeout=60.0, step=0.05)
        assert got, "the bot picks up emblems by itself"
    finally:
        srv.stop()


# ------------------------------------------------------------------ limits / abuse
def test_player_limit_and_version_are_enforced():
    srv = _server(bot=True, max_players=2)
    try:
        _client(srv, "Ann")
        _client(srv, "Ann")                        # same name: made unique
        assert {p.name for p in srv.players.values()} >= {"Ann", "Ann 2"}
        try:
            _client(srv, "Cat")
            raise AssertionError("a third human must be refused")
        except ConnectionError as exc:
            assert "full" in str(exc)
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=2)
        s.sendall(encode({"t": "hello", "v": PROTOCOL_VERSION + 1, "name": "Old"}))
        reply = LineReader().feed(s.recv(4096))
        assert reply and reply[0]["t"] == "reject" and "version" in reply[0]["reason"]
        s.close()
        assert GameServer(small_world(), port=0, max_players=99, log=None).max_players == config.MP_MAX_PLAYERS
    finally:
        srv.stop()


def test_a_full_game_of_15_where_bots_give_way_to_people():
    srv = _server(bots=config.MP_MAX_PLAYERS - 1, max_players=config.MP_MAX_PLAYERS)
    clients = []
    try:
        assert srv.status()["bots"] == config.MP_MAX_PLAYERS - 1, "the host asked for 14 bots (its own slot left free)"
        for i in range(config.MP_MAX_PLAYERS):
            clients.append(_client(srv, f"P{i}"))
            st = srv.status()
            assert st["humans"] == i + 1 and st["humans"] + st["bots"] <= config.MP_MAX_PLAYERS
        assert srv.status()["bots"] == 0, "every bot gave its slot to a person"
        try:
            _client(srv, "Late")
            raise AssertionError("a 16th person must be refused")
        except ConnectionError as exc:
            assert "full" in str(exc)
        # everyone moves; every client sees the other 14 players, moving
        for step in range(6):
            for i, c in enumerate(clients):
                c.send_state([i * 2.0, 0.9, float(step)], 0.0, 0.0, "GROUND")
            time.sleep(0.06)

        def all_see_everyone():
            return all(len(c.remote_poses()) == config.MP_MAX_PLAYERS - 1 for c in clients)
        assert wait_for(all_see_everyone, 5.0), [len(c.remote_poses()) for c in clients]
        poses = {rp.name: pose for rp, pose in clients[0].remote_poses()}
        assert abs(poses["P3"]["p"][0] - 6.0) < 1e-3
        # two people leave: two bots take their places again
        clients.pop().close()
        clients.pop().close()
        assert wait_for(lambda: srv.status()["bots"] == 2 and srv.status()["humans"] == config.MP_MAX_PLAYERS - 2)
        assert wait_for(lambda: sum(1 for r in clients[0].scoreboard() if r[3]) == 2)
    finally:
        for c in clients:
            c.close()
        srv.stop()


def test_junk_and_abuse_never_take_the_server_down():
    srv = _server(bot=False)
    try:
        a = _client(srv, "Ann")
        # garbage before hello: dropped connection, nothing else
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=2)
        s.sendall(b"GET / HTTP/1.0\r\n\r\n")
        time.sleep(0.2)
        s.close()
        # a joined client sending junk / hostile values
        raw = a.sock
        raw.sendall(b"{not json}\n" + encode({"t": "state", "p": [1e30, 0, 0]}) + encode({"t": "state", "p": "x"}) +
                    encode({"t": "state", "p": [0, 0, 0], "yaw": "NaN", "st": "X" * 500}) + encode({"t": "mystery"}))
        # an endless line
        s2 = socket.create_connection(("127.0.0.1", srv.port), timeout=2)
        s2.sendall(encode({"t": "hello", "v": PROTOCOL_VERSION, "name": "Eve"}))
        try:
            s2.sendall(b"x" * (300 * 1024))
        except OSError:
            pass
        time.sleep(0.3)
        s2.close()
        b = _client(srv, "Ben")                    # the server still serves
        b.send_state([0.0, 0.9, -15.0], 0, 0, "GROUND")
        assert wait_for(lambda: any(p.name == "Ben" for p, _ in a.remote_poses(time.monotonic() + 1.0)))
        assert srv.running and srv.error is None
    finally:
        srv.stop()


def test_clients_hear_when_the_host_closes_the_game():
    srv = _server(bot=False)
    a = _client(srv, "Ann")
    srv.stop()
    assert _events(a, "disconnected") and not a.connected
    try:
        GameClient("127.0.0.1", srv.port, "Late", timeout=1.0).connect()
        raise AssertionError("nobody is listening any more")
    except ConnectionError:
        pass


# ------------------------------------------------------------------- the session
class _FakePlayer:
    """What MultiplayerSession needs of a PlayerController."""

    def __init__(self):
        from parkour.controller import PlayerController
        w = small_world()
        self.pc = PlayerController(w.ledge_system, w.obstacles, spawn_pos=(30.0, 0.9, 30.0))

    def __getattr__(self, k):
        return getattr(self.pc, k)


def test_a_hosted_session_spawns_syncs_and_reports():
    w = small_world()
    s = MultiplayerSession.host(w, HostSettings(port=0, bot=False, emblems=4), "Host", log=None)
    try:
        assert s.hosting and "Hosting on port" in s.describe()
        me = _FakePlayer().pc
        notes = []
        wait_for(lambda: notes.extend(s.update(0.05, me)) or any("Joined" in t for _, t in notes))
        assert np.allclose(me.pos, w.spawn_points[0]), "spawned on the server's spawn point"
        guest = MultiplayerSession.join(f"127.0.0.1:{s.server.port}", "Guest")
        gp = _FakePlayer().pc
        wait_for(lambda: notes.extend(s.update(0.05, me)) or any("Guest joined" in t for _, t in notes))
        guest.update(0.05, gp)
        assert wait_for(lambda: (guest.update(0.05, gp) or True) and any(p.name == "Host" for p, _ in guest.remote_poses()))
        # the host walks into every emblem: pickups reported, then the results
        for e in sorted(s.emblems(), key=lambda e: e["id"]):
            me.pos = np.asarray(e["pos"], dtype=np.float32)
            s.send_timer = 0.0
            notes.extend(s.update(0.05, me))
            time.sleep(0.05)
        assert wait_for(lambda: notes.extend(s.update(0.05, me)) or s.results is not None)
        assert any(k == "pickup" and t.startswith("You picked up") for k, t in notes)
        assert s.winner_text().startswith("You win with 4 emblems")
        assert wait_for(lambda: (guest.update(0.05, gp) or True) and guest.results is not None)
        assert guest.winner_text() == "Host wins with 4 emblems"
        s.join_next()
        guest.join_next()
        assert wait_for(lambda: (s.update(0.05, me) or True) and s.results is None and s.round == 2)
        guest.leave()
    finally:
        s.leave()
    assert s.ended == "left the game" and not s.client.connected


def test_a_server_started_without_a_seed_works_round_after_round():
    """Regression: the clock-derived seed (up to 2**31) times 1000 overflowed numpy's seed range."""
    srv = GameServer(small_world(), host="127.0.0.1", port=0, emblems=4, seed=None, log=None)
    srv.start()
    try:
        assert srv.seed > 0 and len(srv.match.emblems) == 4
        for _ in range(3):
            srv.match.start([1])
        assert srv.match.round == 4
    finally:
        srv.stop()
