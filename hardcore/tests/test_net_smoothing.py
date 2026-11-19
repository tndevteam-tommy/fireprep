"""Remote players move on the timeline they actually moved on (server-clock pose stamps), not on
when snapshots happened to arrive: no stutter from network jitter or from a pose being carried by
two snapshots, and a late update is bridged instead of freezing."""
import numpy as np

from harness import *  # noqa
from net.client import GameClient, RemotePlayer


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _client():
    clock = FakeClock()
    c = GameClient("127.0.0.1", 1, "Me", clock=clock)
    c.id = 1
    return c, clock


def _entry(x, pt, pid=2):
    return {"id": pid, "name": "Runner", "bot": False, "team": None, "p": [x, 0.9, 0.0], "yaw": 0.0,
            "pitch": 0.0, "st": "SPRINT", "n": None, "spd": 8.0, "ac": None, "pt": pt, "score": 0}


def _feed(c, clock, arrive_local, server_now, entries):
    clock.t = arrive_local
    c._handle({"t": "snap", "time": server_now, "now": server_now, "players": entries})


def test_a_steady_runner_is_drawn_at_a_steady_speed_despite_jitter_and_mismatched_rates():
    """The runner sends 30 poses a second, the server snapshots 20 times a second (so a snapshot
    sometimes carries the same pose as the last one and sometimes skips one), and snapshots arrive
    with up to 30 ms of jitter.  Drawn on the pose stamps, the runner still moves at 8 m/s."""
    c, clock = _client()
    rng = np.random.default_rng(3)
    speed = 8.0
    latency = 0.040
    snaps = []
    for k in range(120):
        server_now = k / 20.0
        pt = np.floor(server_now * 30.0) / 30.0          # newest 30 Hz update the server has
        snaps.append((100.0 + server_now + latency + float(rng.uniform(0.0, 0.030)), server_now, pt))
    snaps.sort()
    xs = []
    frame = 100.0
    i = 0
    while frame < 105.5:                                  # the game's frames, 60 a second
        while i < len(snaps) and snaps[i][0] <= frame:
            arrive, server_now, pt = snaps[i]
            _feed(c, clock, arrive, server_now, [_entry(speed * pt, pt)])
            i += 1
        clock.t = frame
        if frame > 101.0:
            xs.append(c.remote_poses()[0][1]["p"][0])
        frame += 1.0 / 60.0
    sp = np.diff(np.array(xs)) * 60.0
    assert np.all(np.abs(sp - speed) < 0.5), f"speed ranges {sp.min():.2f} .. {sp.max():.2f} m/s"


def test_the_old_arrival_time_interpolation_would_have_stuttered():
    """(The same feed drawn on arrival times: the speed swings wildly - what players used to see.)"""
    rng = np.random.default_rng(3)
    rp = RemotePlayer(2, "Runner")
    for k in range(80):
        server_now = k / 20.0
        pt = np.floor(server_now * 30.0) / 30.0
        arrive = server_now + 0.040 + float(rng.uniform(0.0, 0.030))
        rp.history.append((arrive, _entry(8.0 * pt, pt)))
    xs = [rp.sample(t)["p"][0] for t in np.arange(1.0, 3.5, 1.0 / 60.0)]
    sp = np.diff(xs) * 60.0
    assert sp.max() - sp.min() > 4.0


def test_a_pose_carried_by_two_snapshots_is_recorded_once():
    rp = RemotePlayer(2, "Runner")
    rp.add(1.0, _entry(0.0, 1.0))
    rp.add(1.0, dict(_entry(0.0, 1.0), score=3))
    rp.add(1.05, _entry(0.4, 1.05))
    rp.add(0.9, _entry(-1.0, 0.9))                      # older than what we have: ignored
    assert [t for t, _ in rp.history] == [1.0, 1.05]
    assert rp.history[0][1]["score"] == 3, "the entry is refreshed (score, team)"


def test_a_late_update_is_bridged_along_the_last_motion_then_held():
    rp = RemotePlayer(2, "Runner")
    rp.add(1.00, _entry(0.0, 1.00))
    rp.add(1.05, _entry(0.4, 1.05))                     # 8 m/s
    assert abs(rp.sample(1.075)["p"][0] - 0.6) < 1e-4, "25 ms late: carried on"
    far = rp.sample(1.05 + 5.0)["p"][0]
    assert abs(far - (0.4 + 8.0 * config.MP_EXTRAPOLATE_MAX)) < 1e-4, "never further than MP_EXTRAPOLATE_MAX"


def test_no_extrapolation_across_a_respawn_or_a_single_pose():
    rp = RemotePlayer(2, "Runner")
    rp.add(1.0, _entry(4.0, 1.0))
    assert rp.sample(3.0)["p"][0] == 4.0
    rp.add(1.05, _entry(60.0, 1.05))                    # teleported (respawn)
    assert rp.sample(1.10)["p"][0] == 60.0


def test_the_clock_offset_is_the_quickest_arrival_and_creeps_up_slowly():
    c, clock = _client()
    _feed(c, clock, 100.10, 0.0, [])
    assert abs(c.clock_offset - 100.10) < 1e-9
    _feed(c, clock, 100.15, 0.1, [])                   # quicker arrival: 100.05
    assert abs(c.clock_offset - 100.05) < 1e-9
    _feed(c, clock, 100.60, 0.2, [])                   # a 350 ms hiccup barely moves it
    assert c.clock_offset < 100.05 + 0.02
    assert abs(c.server_time(200.0) - (200.0 - c.clock_offset)) < 1e-9


def test_before_any_snapshot_the_newest_pose_is_shown():
    c, clock = _client()
    c._welcome({"id": 1, "server": {}, "team": None, "match": {},
                "players": [_entry(3.0, 12.5)], "spawn": [0, 0.9, 0]})
    poses = c.remote_poses()
    assert poses and poses[0][1]["p"][0] == 3.0
