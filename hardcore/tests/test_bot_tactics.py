"""Bot intelligence (game/bot_skill.py, game/sight.py, game/tactics.py): what the bots can see and
remember, how they search for emblems, hunt as a pack and flee - the pieces, one at a time.
(How the levels compare in whole games is tests/test_bot_levels.py.)"""
import numpy as np

from mp_helpers import *  # noqa
from game import bot_skill
from game.match import GameRules, create_match, Emblem
from game.tactics import (EmblemMemory, SearchMemory, Sighting, Tactician, Tracker, Track, Hive, PackPlanner,
                          search_grid)
from net.server import GameServer


class Body:
    """A stand-in for a bot's PlayerController (what a Tactician reads)."""

    def __init__(self, pos, facing=(0.0, 0.0, -1.0), vel=(0.0, 0.0, 0.0), state="GROUND"):
        self.pos = np.asarray(pos, dtype=np.float32)
        self.forward_h = np.asarray(facing, dtype=np.float32)
        self.vel = np.asarray(vel, dtype=np.float32)
        self.state = state
        self.on_ground = True

    @property
    def feet_y(self):
        return float(self.pos[1]) - config.PLAYER_HALF_HEIGHT

    @property
    def eye_pos(self):
        return self.pos + np.array([0.0, config.PLAYER_EYE_HEIGHT - config.PLAYER_HALF_HEIGHT, 0.0], np.float32)


def _emblem_match(world, emblems):
    m = create_match(world, GameRules(GameRules.EMBLEMS, False, 0, emblems=4), seed=1)
    m.start([1, 2])
    m.emblems = [Emblem(i, p, "street") for i, p in enumerate(emblems)]
    return m


def _tactician(level, world=None, pid=1, hive=None):
    world = world or small_world()
    hive = hive or Hive(search_grid(world), bot_skill.skill(level))
    return Tactician(pid, level, world, hive, Tracker(), seed=pid)


# ---------------------------------------------------------------------------- levels
def test_levels_are_named_ordered_and_validated():
    assert bot_skill.LEVELS == ("simple", "complex", "hard")
    assert bot_skill.normalize("HARD") == "hard" and bot_skill.normalize("genius") == bot_skill.DEFAULT
    assert bot_skill.cycle("hard", 1) == "simple" and bot_skill.cycle("simple", -1) == "hard"
    s, c, h = (bot_skill.skill(x) for x in bot_skill.LEVELS)
    assert s.sight_range < c.sight_range < h.sight_range
    assert s.sight_cone_deg < c.sight_cone_deg < h.sight_cone_deg == 360.0
    assert (s.emblem_memory, c.emblem_memory, h.emblem_memory) == (1, 3, None), "one, three, all"
    assert s.search_memory < c.search_memory and h.search_memory is None
    assert s.position_error > c.position_error > h.position_error == 0.0
    assert not s.shared and not c.shared and h.shared, "only Hard bots pool what they know"
    assert (s.pursuit, c.pursuit, h.pursuit) == ("direct", "direct", "pack")


# ---------------------------------------------------------------------------- sight
def test_walls_and_roofs_hide_what_is_behind_and_on_them():
    w = small_world()
    hf = w.sight
    spec = [s for s, _ in w.buildings if s.name == "bB"][0]
    roof = np.array([spec.x + spec.width * 0.5, spec.height + 1.0, spec.z + spec.depth * 0.5], np.float32)
    street_eye = np.array([spec.x + spec.width * 0.5, 1.65, spec.z + spec.depth + 3.0], np.float32)
    assert not hf.visible(street_eye, roof), "an emblem on a roof is hidden from the street below"
    behind = np.array([spec.x + spec.width * 0.5, 1.0, spec.z - 3.0], np.float32)
    assert not hf.visible(street_eye, behind), "the building hides the street behind it"
    roof_eye = roof + np.array([4.0, 0.65, 0.0], np.float32)
    assert hf.visible(roof_eye, roof), "up on the roof it is in plain sight"
    open_a = np.array([0.0, 1.65, -10.0], np.float32)
    assert hf.visible(open_a, np.array([20.0, 1.0, 5.0], np.float32)), "open ground: nothing in the way"
    many = hf.visible_many(street_eye, np.array([roof, behind, street_eye + [5.0, -0.6, 5.0]], np.float32))
    assert list(many) == [False, False, True]


# ---------------------------------------------------------------------------- memory
def test_emblem_memory_keeps_the_nearest_one_three_or_all():
    here = np.array([0.0, 0.9, 0.0], np.float32)
    for cap, kept in ((1, {0}), (3, {0, 1, 2}), (None, {0, 1, 2, 3, 4})):
        mem = EmblemMemory(cap)
        for i, d in enumerate((40.0, 10.0, 20.0, 70.0, 5.0)[::-1]):
            p = np.array([d, 1.0, 0.0], np.float32)
            mem.remember(Sighting(i, p, p, 0.0, d), here)
        dist = {s.id: float(s.est[0]) for s in mem.values()}
        nearest = sorted(dist.values())
        assert len(dist) == (5 if cap is None else cap)
        assert sorted(dist.values()) == sorted((5.0, 10.0, 20.0, 40.0, 70.0))[:len(dist)], (cap, dist)
        assert nearest[0] == 5.0, "the nearest is always kept in mind"


def test_search_memory_forgets_the_longest_ago_searched_areas():
    grid = search_grid(small_world())
    mem = SearchMemory(grid, 3)
    for n in range(6):
        mem.mark(n)
    assert [mem.searched(n) for n in range(6)] == [False, False, False, True, True, True]
    mem.mark(3)                       # looked at again: remembered longest
    mem.mark(6)
    assert mem.searched(3) and not mem.searched(4)
    full = SearchMemory(grid, None)
    for n in range(len(grid)):
        full.mark(n)
    assert full.count() == len(grid)
    part = SearchMemory(grid, 2)
    multi = next(n for n in range(len(grid)) if len(grid.samples[n]) > 1)
    part.mark(multi, 0)
    assert not part.searched(multi) and part.unseen_samples(multi) == list(range(1, len(grid.samples[multi])))


# ------------------------------------------------------------------------- emblem hunt
def test_a_bot_goes_for_emblems_it_has_seen_not_ones_it_has_not():
    w = small_world()
    spec = [s for s, _ in w.buildings if s.name == "bB"][0]
    on_roof = (spec.x + spec.width * 0.5, spec.height + 1.0, spec.z + spec.depth * 0.5)
    in_view = (4.0, 1.0, -26.0)                             # in the street in front of the building
    m = _emblem_match(w, [on_roof, in_view])
    for level in bot_skill.LEVELS:
        t = _tactician(level, w)
        body = Body((0.0, 0.9, -20.0), facing=(0.0, 0.0, -1.0))
        goal = t.targets(m, 0.0, body)
        assert goal and goal[0].id == 1, f"{level}: the one in plain sight, never the hidden one on the roof"
        assert 0 not in t.memory
        assert float(np.linalg.norm(goal[0].pos - np.asarray(in_view))) <= 0.08 * 8.0, "a good idea where it is"


def test_a_sighting_from_afar_is_rough_and_sharpens_closer_in():
    w = small_world()
    m = _emblem_match(w, [(30.0, 1.0, -55.0)])             # (down the open lane east of the buildings)
    t = _tactician("simple", w)
    t.look(m, 0.0, Body((30.0, 0.9, 8.0)))
    far = t.memory.get(0)
    err_far = float(np.linalg.norm(far.est - far.true_pos))
    assert 1.0 < err_far <= 0.08 * 70.0, "a rough idea where it is"
    t.look(m, 1.0, Body((30.0, 0.9, -48.0)))
    near = t.memory.get(0)
    assert float(np.linalg.norm(near.est - near.true_pos)) < err_far * 0.3, "sharper as it gets closer"
    hard = _tactician("hard", w)
    hard.look(m, 0.0, Body((30.0, 0.9, 8.0)))
    assert float(np.linalg.norm(hard.memory.get(0).est - hard.memory.get(0).true_pos)) < 1e-4


def test_seeing_several_the_nearest_comes_first_and_simple_bots_keep_only_it_in_mind():
    w = small_world()
    m = _emblem_match(w, [(25.0, 1.0, -18.0), (3.0, 1.0, -6.0), (-25.0, 1.0, -14.0)])
    for level, kept in (("simple", 1), ("complex", 3), ("hard", 3)):
        t = _tactician(level, w)
        goal = t.targets(m, 0.0, Body((0.0, 0.9, 8.0)))
        assert goal[0].id == 1, f"{level}: the closest first"
        assert len(t.memory) == kept


def test_an_emblem_someone_else_took_is_forgotten_only_when_the_spot_is_seen_empty():
    w = small_world()
    m = _emblem_match(w, [(0.0, 1.0, -30.0)])
    t = _tactician("complex", w)
    t.targets(m, 0.0, Body((0.0, 0.9, 8.0)))
    assert 0 in t.memory
    m.emblems[0].taken_by = 2
    t.targets(m, 1.0, Body((0.0, 0.9, 8.0), facing=(0.0, 0.0, 1.0)))     # looking the other way
    assert 0 in t.memory, "it does not know yet"
    t.targets(m, 2.0, Body((0.0, 0.9, 8.0), facing=(0.0, 0.0, -1.0)))    # looks at the spot: gone
    assert 0 not in t.memory


def test_with_nothing_in_mind_a_bot_searches_somewhere_it_has_not_looked():
    w = small_world()
    m = _emblem_match(w, [(60.0, 1.0, 60.0)])
    m.emblems[0].taken_by = 2                               # (nothing to see anywhere)
    for level in bot_skill.LEVELS:
        t = _tactician(level, w)
        body = Body((0.0, 0.9, -15.0))
        goal = t.targets(m, 0.0, body)
        assert goal and goal[0].kind == "zone" and str(goal[0].id).startswith("explore:")
        n = t.grid.cell_at(float(goal[0].pos[0]), float(goal[0].pos[2]))
        assert n is None or not t.search.searched(n) or t.explore["vantage"], "somewhere not looked over yet"
        assert t.targets(m, 0.1, body)[0].id == goal[0].id, "it keeps to its plan"


def test_hard_bots_share_what_they_see_and_split_the_emblems():
    w = small_world()
    m = _emblem_match(w, [(25.0, 1.0, -18.0), (3.0, 1.0, -6.0)])
    hive = Hive(search_grid(w), bot_skill.skill("hard"))
    a = _tactician("hard", w, pid=1, hive=hive)
    b = _tactician("hard", w, pid=2, hive=hive)
    ga = a.targets(m, 0.0, Body((0.0, 0.9, 8.0)))
    assert b.memory is a.memory and 0 in b.memory and 1 in b.memory, "one sees, both know"
    gb = b.targets(m, 0.05, Body((30.0, 0.9, -25.0), facing=(0.0, 0.0, 1.0)))
    assert ga[0].id == 1 and gb[0].id == 0, "each takes a different one"
    # complex bots keep their own counsel
    c1, c2 = _tactician("complex", w, pid=1), _tactician("complex", w, pid=2)
    c1.targets(m, 0.0, Body((0.0, 0.9, 8.0)))
    assert len(c1.memory) and not len(c2.memory)


# ------------------------------------------------------------------------------ chasing
def _track(points, dt=0.1):
    tr = Track()
    for k, p in enumerate(points):
        tr.add(k * dt, np.asarray(p, np.float32))
    return tr


def test_a_track_measures_velocity_and_forgets_a_respawn():
    tr = _track([(0.0, 0.9, -0.8 * k) for k in range(10)])           # 8 m/s along -z
    v = tr.velocity()
    assert abs(float(v[2]) + 8.0) < 0.2 and abs(float(v[0])) < 1e-3
    tr.add(5.0, np.array([80.0, 0.9, 80.0], np.float32))            # teleported
    assert len(tr.points) == 1 and float(np.linalg.norm(tr.velocity())) == 0.0


def test_every_hunter_runs_the_quarry_down_on_its_live_track():
    """Measured (tools/bot_levels.py chase): a straight run at a faster runner by every hunter beats every
    way of spreading the pack out - so every level chases, reading the quarry's position every tick."""
    for level in bot_skill.LEVELS:
        pack = PackPlanner(bot_skill.skill(level))
        tr = _track([(0.0, 0.9, 40.0 - 0.85 * k) for k in range(10)])
        q = tr.pos
        hunters = {1: q + np.array([0.0, 0.0, 10.0], np.float32), 2: q + np.array([6.0, 0.0, -24.0], np.float32),
                   3: q + np.array([-20.0, 0.0, 30.0], np.float32), 4: q + np.array([30.0, 0.0, 5.0], np.float32)}
        plan = pack.plan(1.0, 99, tr, hunters, [1, 2, 3, 4])
        assert all(role == ("chase",) for role in plan.values()), level
        p1, kind = PackPlanner.resolve(plan[3], tr)
        assert kind == "player" and np.allclose(p1, q)
        tr.add(1.0, np.asarray(q, np.float32) + np.array([0.0, 0.0, -0.85], np.float32))
        assert np.allclose(PackPlanner.resolve(plan[3], tr)[0], tr.pos), "the live position, not the planned one"


def test_hard_posts_a_hunter_in_the_flag_thiefs_base():
    tr = _track([(0.0, 0.9, -0.8 * k) for k in range(10)])
    base = np.array([0.0, 0.9, -60.0], np.float32)
    hunters = {1: tr.pos + np.array([0.0, 0.0, 10.0], np.float32), 2: base + np.array([5.0, 0.0, -5.0], np.float32)}
    plan = PackPlanner(bot_skill.skill("hard")).plan(1.0, 99, tr, hunters, [1, 2], dest=base, dest_exact=True)
    assert plan[1] == ("chase",), "the nearest keeps running it down"
    assert plan[2][0] == "fixed" and float(np.linalg.norm(plan[2][1] - base)) < 1e-3, "the other waits in its base"
    far = {1: hunters[1], 2: base + np.array([0.0, 0.0, -120.0], np.float32)}
    plan = PackPlanner(bot_skill.skill("hard")).plan(1.0, 99, tr, far, [1, 2], dest=base, dest_exact=True)
    assert plan[2] == ("chase",), "(only one that can be there first)"
    plan = PackPlanner(bot_skill.skill("complex")).plan(1.0, 99, tr, hunters, [1, 2], dest=base, dest_exact=True)
    assert all(r == ("chase",) for r in plan.values()), "a Hard side's trick"


def test_simple_bots_chase_straight_at_the_quarry():
    w = small_world()
    m = create_match(w, GameRules(GameRules.ZOMBIE, False, 0), seed=2)
    m.start([1, 2])
    living = m.living
    zombie = 2 if living == 1 else 1
    m.positions[living] = np.array([10.0, 0.9, -10.0], np.float32)
    m.positions[zombie] = np.array([0.0, 0.9, -10.0], np.float32)
    t = _tactician("simple", w, pid=zombie)
    t.tracker.update(0.0, m.positions)
    goal = t.targets(m, 0.0, Body(m.positions[zombie]))
    assert goal[0].id == f"player:{living}" and np.allclose(goal[0].pos, m.positions[living])


# ------------------------------------------------------------------------------ fleeing
def test_a_fleeing_bot_runs_away_commits_and_does_not_dither():
    w = small_world()
    m = create_match(w, GameRules(GameRules.ZOMBIE, False, 0), seed=2)
    m.start([1, 2, 3])
    living = m.living
    zs = [p for p in m.players if p != living]
    here = np.array([0.0, 0.9, -10.0], np.float32)
    m.positions.update({living: here, zs[0]: here + [6.0, 0.0, 0.0], zs[1]: here + [9.0, 0.0, 4.0]})
    for level in ("complex", "hard"):
        t = _tactician(level, w, pid=living)
        t.tracker.update(0.0, m.positions)
        body = Body(here, facing=(-1.0, 0.0, 0.0), vel=(-6.0, 0.0, 0.0))
        goal = t.targets(m, 0.0, body)[0]
        assert float(np.dot((goal.pos - here)[[0, 2]], (here - m.positions[zs[0]])[[0, 2]])) > 0, "away from them"
        ids = {t.targets(m, 0.05 * k, body)[0].id for k in range(1, 8)}
        assert ids == {goal.id}, f"{level}: commits to its way out (no dithering)"
        climbing = Body(here, state="CLIMB_UP")
        assert t.targets(m, 20.0, climbing)[0].id == goal.id, "half way up a wall it sees the climb through"


# ------------------------------------------------------------------------ the server
def test_the_host_picks_the_bots_level_and_each_bot_thinks_at_it():
    rules = GameRules(GameRules.EMBLEMS, True, 0, emblems=4)
    srv = GameServer(small_world(), port=0, max_players=5, bots=4, rules=rules, seed=3, log=None, bot_level="hard")
    srv._broadcast = lambda msg, skip=None: None
    srv.match.start({})
    srv._fill_bots()
    assert srv.status()["bot_level"] == "hard"
    assert all(b.tactician.skill.name == "hard" for b in srv.bots.values())
    red = [b for b in srv.bots.values() if b.info.team == "red"]
    blue = [b for b in srv.bots.values() if b.info.team == "blue"]
    assert red[0].tactician.hive is red[1].tactician.hive, "a side shares one mind"
    assert red[0].tactician.hive is not blue[0].tactician.hive, "the other side does not read it"
    srv.set_bot_level("simple", pid=red[0].pid)
    assert red[0].tactician.skill.name == "simple" and red[1].tactician.skill.name == "hard"
    for _ in range(60):
        srv._tick(1.0 / 60.0)
    assert all(b.brain.goal is not None for b in srv.bots.values()), "everyone has something to do"
    assert GameServer(small_world(), port=0, log=None, bot_level="bogus").bot_level == bot_skill.DEFAULT


def test_a_new_round_forgets_what_was_seen_and_searched():
    rules = GameRules(GameRules.EMBLEMS, False, 0, emblems=4)
    srv = GameServer(small_world(), port=0, max_players=3, bots=2, rules=rules, seed=3, log=None, bot_level="complex")
    srv._broadcast = lambda msg, skip=None: None
    srv.match.start({})
    srv._fill_bots()
    for _ in range(120):
        srv._tick(1.0 / 60.0)
    bot = next(iter(srv.bots.values()))
    assert bot.tactician.search.count() > 0
    srv._start_next_match()
    assert bot.tactician.search.count() == 0 and len(bot.tactician.memory) == 0


def test_the_multiplayer_screen_offers_bot_skill_when_there_are_bots(tmp_path):
    from core.mp_menu import MultiplayerMenuModel
    from game.session import parse_mp_args
    m = MultiplayerMenuModel()
    assert m.values["bot_level"] == bot_skill.DEFAULT and "bot_level" in m.rows()
    m.values["bots"] = 0
    assert "bot_level" not in m.rows(), "no bots: nothing to be clever"
    m.values["bots"] = 3
    m.focus = m.rows().index("bot_level")
    m.handle_key("right")
    assert m.values["bot_level"] == bot_skill.cycle(bot_skill.DEFAULT, 1) and m.status
    assert m.value_text("bot_level") == bot_skill.LABELS[m.values["bot_level"]]
    assert m.host_request()[1].bot_level == m.values["bot_level"]
    path = str(tmp_path / "multiplayer.json")
    m.save(path)
    assert MultiplayerMenuModel.load(path).values["bot_level"] == m.values["bot_level"]
    assert MultiplayerMenuModel({"bot_level": "genius"}).values["bot_level"] == bot_skill.DEFAULT
    assert parse_mp_args(["--host", "--bot-level", "hard"])["bot_level"] == "hard"
    assert "bot_level" not in parse_mp_args(["--host", "--bot-level", "genius"])
