"""The Neon Moth and the metro (``levels.bar``, ``levels.metro``, ``entities.metro``,
``ai.nightlife``): a bar with a bartender who comes in on the metro, opens at five, serves
whoever comes in for a drink and goes home on the metro after three; a station with ticket
gates, a stair, a footbridge and a platform the trains stop at - and nobody appearing or
disappearing on the way: riders walk into a train and the doors close on them."""

from __future__ import annotations

import math

import pytest

from cyberduke.ai.nightlife import bar_hours, in_the_bar
from cyberduke.entities.metro import Metro
from cyberduke.levels import bar, build_night_market, metro
from cyberduke.simulation import InputState, Simulation
from cyberduke.world.vehicles import CAR_HALF_WIDTH, VESTIBULE_BACK, train_doors

DT = 1 / 20
LEVEL = build_night_market()


def street(hour: float, rate: float = 0.5) -> Simulation:
    sim = Simulation(LEVEL)
    sim.clock.set(hour * 60.0)
    sim.clock.rate = rate
    sim.player.x, sim.player.z = 0.0, -1.2  # (out of everybody's way)
    return sim


def run(sim: Simulation, seconds: float, watch=None) -> None:
    for _ in range(int(seconds / DT)):
        sim.update(DT, InputState())
        if watch is not None:
            watch(sim)


def walk_to(sim, x, z, seconds=6.0) -> bool:
    p = sim.player
    for _ in range(int(seconds / DT)):
        if math.hypot(x - p.x, z - p.z) < 0.3:
            return True
        p.yaw = math.degrees(math.atan2(z - p.z, x - p.x))
        p.pitch = 0.0
        sim.update(DT, InputState(forward=1.0))
    return False


def labels(*words: str) -> list[str]:
    return [p.label for p in LEVEL.props if any(w in p.label for w in words)]


# ============================================================================ the bar
def test_the_bar_is_furnished_like_a_bar():
    assert len(labels("Bar stool seat")) == len(bar.STOOL_ZS) == 4
    assert len(labels("Bar table top")) == 2 and len(labels("Bar chair seat")) == 4
    assert labels("Bar counter top") and labels("Beer tap handle") and labels("Till")
    assert len(labels("Bottle on the back bar")) > 40, "shelves of bottles"
    assert labels("Jukebox") and labels("Neon moth") and labels("Dartboard")
    assert labels("Sink basin (bottom)"), "a glass-washer sink in the back bar"
    screens = [p for p in LEVEL.props if "tv:bar" in p.tags]
    assert len(screens) == 1, "a television for the customers"
    glass = [p for p in LEVEL.props if p.label == "Bar window glass"]
    assert len(glass) == 2 and all(p.material.alpha < 1.0 for p in glass), "you can see in"


def test_the_bar_door_opens_for_anybody_while_it_is_open_and_not_when_it_is_shut():
    sim = street(19.5, rate=0.0)
    run(sim, 0.5)
    door = sim.room.doors[bar.DOOR_ID]
    assert sim.life.open["bar"] and door.unlocked, "open at half past seven"
    p = sim.player
    p.x, p.z = -8.55, 44.0
    assert walk_to(sim, -8.55, 40.5), "in through the door"
    assert in_the_bar(sim.life, p.x, p.z)
    sim.life.open["bar"] = False
    door.lock()
    p.x, p.z = -8.55, 44.0
    run(sim, 2.0)
    assert not walk_to(sim, -8.55, 40.5, seconds=3.0)
    assert not in_the_bar(sim.life, p.x, p.z) and p.z > bar.FACE, "a locked door stops you"


def test_the_open_sign_is_lit_only_while_the_bar_is_open():
    from tests.test_visual_regressions import TEXTURES, MatrixRecorder

    from cyberduke.render.television import draw_screens, screen_of

    sign = next(p for p in LEVEL.props if "open:bar" in p.tags)
    assert screen_of(sign) == ("open", "bar")
    glow = tuple(sign.material.emission)
    sectors = [s for s in LEVEL.sectors if sign in s.props]

    class Recorder(MatrixRecorder):
        def __init__(self):
            super().__init__()
            self.glows = []

        def emission(self, rgb):
            self.glows.append(None if rgb is None else tuple(rgb[:3]))

    for is_open in (True, False):
        ctx = Recorder()
        draw_screens(ctx, sectors, TEXTURES, {}, 0.0, opens={"bar": is_open})
        assert (glow in ctx.glows) is is_open, is_open


# ============================================================================ the station
def test_the_station_has_gates_a_stair_a_bridge_and_a_platform_level_with_the_train():
    steps = [p for p in LEVEL.props if p.label == "Metro stair step"]
    assert len(steps) == metro.STEPS
    tops = sorted(p.y + p.size[1] / 2.0 for p in steps)
    assert tops[-1] == pytest.approx(metro.FLOOR_Y) and tops[0] == pytest.approx(metro.RISE)
    deck = next(p for p in LEVEL.props if p.label == "Platform deck")
    assert deck.y + deck.size[1] / 2.0 == pytest.approx(metro.FLOOR_Y)
    edge = deck.z + deck.size[2] / 2.0
    assert metro.TRACK_Z - CAR_HALF_WIDTH - edge == pytest.approx(0.05, abs=0.02), "a 5 cm gap"
    doors = train_doors(metro.STOP_X)
    assert all(metro.PLATFORM_X[0] + 0.7 < x < metro.PLATFORM_X[1] - 0.7 for x in doors)
    panes = [p for p in LEVEL.props if p.label == "Glass front pane"]
    assert len(panes) > 30 and all(p.material.alpha < 1.0 for p in panes), "a glass front"
    assert labels("Ticket gate cabinet") and labels("NO PASS NO ENTRY sign")


def test_no_pass_no_entry_the_gates_stop_the_netrunner():
    sim = street(19.5, rate=0.0)
    p = sim.player
    p.x, p.z = 29.15, 43.4
    assert not walk_to(sim, 29.15, 39.5, seconds=5.0)
    assert p.z > metro.GATES_Z[1], "no further than the gates"
    assert p.z < 42.5, "into the doorway, though"


def _force_gate(sim, lane: int = 1):
    """Stands at the gates, scans, and quickhacks gate ``lane`` open (FORCE OPEN)."""
    from cyberduke.systems.hacking import HACKS, HackKind

    p = sim.player
    x = metro.LANES[lane - 1]
    p.x, p.z, p.pitch = x, 42.9, -12.0
    p.yaw = -90.0  # (looking down the lane at the flaps: -z)
    sim.update(DT, InputState(scan=True))
    sim.update(DT, InputState())
    gate = sim.room.doors[f"metro_gate_{lane}"]
    assert sim.hacking.focus is gate, "the gate's lock is on the net: the scanner finds it"
    assert sim.hacking.request(1)
    run(sim, HACKS[HackKind.FORCE_OPEN].upload_time + 0.2)
    return gate


def _walk_path(sim, path, seconds=20.0) -> None:
    for x, z in path:
        assert walk_to(sim, x, z, seconds), (x, z, sim.player.x, sim.player.z)


UP = (
    (metro.LANES[0], 40.6),
    (30.6, 39.25),
    (39.7, 39.25),
    (41.0, 41.6),
    (41.0, 44.5),
    (41.0, 46.9),
)


def test_a_forced_gate_lets_the_netrunner_through_and_up_to_the_platform():
    """The gates are a door lock on the METRO TRANSIT network: FORCE OPEN and the flaps
    swing back as the netrunner walks up - through, up the stair a tread at a time, along
    the landing, over the footbridge and out onto the platform, level with the train (and
    back down the same way)."""
    sim = street(19.5, rate=0.0)
    gate = _force_gate(sim)
    assert gate.unlocked and gate.data.faction == metro.GATE_FACTION
    p = sim.player
    climbed = []

    def watch(s):
        if metro.STAIR_X0 < p.x < metro.STAIR_X1 and p.z < metro.STAIR_Z[1]:
            climbed.append((p.x, p.floor_y))

    for x, z in UP:
        for _ in range(int(20.0 / DT)):
            if math.hypot(x - p.x, z - p.z) < 0.3:
                break
            p.yaw = math.degrees(math.atan2(z - p.z, x - p.x))
            sim.update(DT, InputState(forward=1.0))
            watch(sim)
        assert math.hypot(x - p.x, z - p.z) < 0.3, (x, z, p.x, p.z)
        if (x, z) == UP[0]:
            assert sim.metro.gate(0) > 0.5, "the flaps open for the netrunner"
    assert p.floor_y == pytest.approx(metro.FLOOR_Y) and p.z > metro.PLATFORM_Z[0]
    assert p.elevation.deck_y == pytest.approx(metro.FLOOR_Y), "on the platform"
    # the stair climbed a tread at a time: the feet on the tread underfoot all the way up
    assert len(climbed) > 20
    for x, y in climbed:
        assert y == pytest.approx(LEVEL.flights[0].height(x, 39.25)), x
    assert max(y for _, y in climbed) > 7.0 and min(y for _, y in climbed) < 0.6
    # ... and back down to the street
    _walk_path(sim, [pt for pt in reversed(UP)] + [(metro.LANES[0], 42.3)])
    assert p.floor_y == 0.0 and p.elevation.deck_y is None


def test_up_there_nobody_steps_off_the_edge():
    """The landing, the footbridge and the platform are kept to by their balustrades and
    edges: walking at them stops the netrunner up there - nobody drops through them to
    the hall floor or the road. Underneath the landing the floor is the hall's."""
    sim = street(19.5, rate=0.0)
    _force_gate(sim)
    p = sim.player
    _walk_path(sim, UP[:4])
    for target in ((36.0, 41.6), (41.0, 38.0), (43.5, 41.6)):  # off the landing's edge
        walk_to(sim, *target, seconds=3.0)
        assert p.floor_y == pytest.approx(metro.FLOOR_Y) and p.x > metro.LANDING_X[0]
        _walk_path(sim, [UP[3]])
    _walk_path(sim, [UP[4]])
    for target in ((38.0, 44.5), (44.0, 44.5)):  # off the footbridge's sides
        walk_to(sim, *target, seconds=3.0)
        assert p.floor_y == pytest.approx(metro.FLOOR_Y)
        assert metro.BRIDGE_X[0] < p.x < metro.BRIDGE_X[1]
        _walk_path(sim, [UP[4]])
    _walk_path(sim, [UP[5]])
    for target in ((41.0, 52.0), (20.0, 47.2), (46.0, 47.2)):  # off the platform's edge, ends
        walk_to(sim, *target, seconds=8.0)
        assert p.floor_y == pytest.approx(metro.FLOOR_Y)
        assert metro.PLATFORM_X[0] < p.x < metro.PLATFORM_X[1]
        assert metro.PLATFORM_Z[0] < p.z < metro.PLATFORM_Z[1]
        _walk_path(sim, [UP[5]])
    # under the landing, on the hall floor
    sim2 = street(19.5, rate=0.0)
    _force_gate(sim2)
    _walk_path(sim2, [UP[0], (40.6, 40.6)])
    assert sim2.player.floor_y == 0.0 and sim2.player.elevation.deck_y is None
    # (and nobody walks into the stair from the side, or in under it)
    walk_to(sim2, 35.0, 39.2, seconds=4.0)
    assert sim2.player.floor_y == 0.0 and sim2.player.z > metro.STAIR_Z[1]


def test_a_forced_gate_sets_itself_again():
    """A minute after it was forced - once nobody is in the way - the station's system
    resets the gate: shut again, no pass no entry."""
    sim = street(19.5, rate=0.0)
    gate = _force_gate(sim)
    sim.player.x, sim.player.z = 0.0, -1.2  # (walked off)
    run(sim, 65.0)
    assert not gate.unlocked and not gate.passable and sim.metro.gate(0) == 0.0
    p = sim.player
    p.x, p.z = metro.LANES[0], 43.4
    assert not walk_to(sim, metro.LANES[0], 39.5, seconds=4.0)
    assert p.z > metro.GATES_Z[1]


def test_the_crowd_keeps_out_of_the_paid_side_even_with_a_gate_open():
    sim = street(19.5, rate=0.0)
    gate = _force_gate(sim)
    blockers = [b for s in LEVEL.sectors for b in s.nav_blockers]
    paid = (metro.X0 + 1.0, 40.0)
    assert any(b.contains_xz(*paid) for b in blockers), "no routes through the paid side"
    assert gate.unlocked


# ============================================================================ the trains
def test_trains_come_stop_at_the_platform_open_and_shut_their_doors_and_go():
    m = Metro(LEVEL.metro)
    seen = []
    for _ in range(int(400 / 0.05)):
        m.update(0.05)
        seen.append((m.phase, m.x, m.doors, m.heading))
    stops = [s for s in seen if s[0] == "stopped"]
    assert stops and all(s[1] == pytest.approx(metro.STOP_X) for s in stops)
    assert max(s[2] for s in stops) == 1.0, "the doors open"
    assert m.trains >= 3, "every couple of minutes"
    headings = {h for ph, _, _, h in seen if ph == "arriving"}
    assert headings == {1.0, -1.0}, "one way, then the other"
    moving = [s for s in seen if s[0] != "stopped"]
    assert all(s[2] == 0.0 for s in moving), "doors shut on the move"
    away = [s for s in seen if s[0] == "away"]
    assert all(abs(s[1]) > 100.0 for s in away), "out of sight down a tunnel when not about"


def test_a_train_waits_for_somebody_on_their_way_to_its_doors():
    m = Metro(LEVEL.metro)
    while m.phase != "stopped":
        m.update(0.05)
    late = object()
    m.start_boarding(late)
    for _ in range(int((Metro.DWELL + 5.0) / 0.05)):
        m.update(0.05)
    assert m.phase == "stopped" and m.doors == 1.0, "held"
    m.board(late)
    for _ in range(int(6.0 / 0.05)):
        m.update(0.05)
    assert m.phase in ("stopped", "departing") and m.doors < 1.0


def test_trains_are_fewer_in_the_small_hours():
    from cyberduke.core.clock import WorldClock

    night = Metro(LEVEL.metro, WorldClock(3 * 60.0, 0.0)).interval()
    evening = Metro(LEVEL.metro, WorldClock(20 * 60.0, 0.0)).interval()
    assert night > 1.5 * evening


# ============================================================================ the people
@pytest.fixture(scope="module")
def evening():
    """Half past seven to a quarter to ten: the bar open, people coming and going."""
    sim = street(19.5)
    record: dict[str, list[tuple[float, str]]] = {}
    vanished, appeared, gates = [], [], set()
    hidden = {}

    def watch(s):
        for n in s.civilians:
            steps = record.setdefault(n.role, [])
            if not steps or steps[-1][1] != n.activity:
                steps.append((s.clock.hour, n.activity))
            was = hidden.get(n.role, n.hidden)
            if n.hidden != was:
                (vanished if n.hidden else appeared).append(
                    (n.role, n.x, n.y, n.z, s.metro.doors, s.metro.phase)
                )
            hidden[n.role] = n.hidden
        for lane in range(2):
            if s.metro.gate(lane) > 0.0:
                gates.add(lane)

    run(sim, 280.0, watch)
    return sim, record, vanished, appeared, gates


def _acts(record, role):
    return [a for _, a in record[role]]


def test_the_bartender_serves_whoever_comes_in_for_a_drink(evening):
    sim, record, *_ = evening
    bartender = _acts(record, "bartender")
    for act in ("pulling a pint", "serving a drink", "clearing a glass", "washing a glass"):
        assert act in bartender, act
    drinkers = [
        role
        for role, steps in record.items()
        if role != "bartender" and "having a drink" in [a for _, a in steps]
    ]
    assert drinkers, "somebody came in for a drink"
    for role in drinkers:
        acts = _acts(record, role)
        i = acts.index("going into the bar")
        order = ["going into the bar", "sitting down at the bar", "ordering a drink",
                 "waiting for a drink", "taking the drink", "having a drink"]  # fmt: skip
        k = i
        for step in order:
            k = acts.index(step, k)
        assert "finishing a drink" in acts[k:] or sim.life.bar.customers


def test_the_regular_comes_in_on_the_metro_for_a_drink(evening):
    sim, record, vanished, appeared, gates = evening
    acts = _acts(record, "regular")
    order = ["waiting for the train", "getting on the train", "on the metro",
             "getting off the train", "going down the stairs", "showing a pass",
             "through the ticket gates", "going into the bar"]  # fmt: skip
    k = 0
    for step in order:
        k = acts.index(step, k)
    assert gates, "the gates opened for him"


def test_riders_walk_into_the_dark_of_the_train_and_out_of_it(evening):
    """Nobody appears or disappears: somebody who gets on a train walks in through a door,
    through its vestibule and on into the dark of the car - out of sight there, not stood
    in the doorway facing out until the doors shut; somebody who gets off comes out of
    that dark."""
    sim, record, vanished, appeared, gates = evening
    assert vanished and appeared
    dark = metro.TRACK_Z + VESTIBULE_BACK  # (the back of a vestibule: the dark beyond it)
    for role, x, y, z, doors, phase in vanished + appeared:
        assert role in ("regular", "bartender"), role
        assert y == pytest.approx(metro.FLOOR_Y) and dark + 0.2 < z < metro.TRACK_Z, (role, z)
        assert min(abs(x - d) for d in train_doors(metro.STOP_X)) < 0.7, "through a doorway"
        assert phase == "stopped", (role, doors, phase)
    others = {role for role, *_ in vanished + appeared}
    assert others <= {"regular", "bartender"}
    # nobody waits in a doorway facing out: on board, straight on into the car and gone
    for role in ("regular", "bartender"):
        acts = _acts(record, role)
        if "getting on the train" in acts:
            assert acts[acts.index("getting on the train") + 1] != "on the train", acts


def test_the_bar_opens_and_closes_with_the_bartender_and_he_goes_home_by_metro():
    sim = street(2.85)
    bt = sim.npc_with_role("bartender")
    assert bar_hours(sim.clock.hour)
    run(sim, 2.0)
    assert sim.life.open["bar"] and sim.life.tv("bar").on
    steps = []

    def watch(s):
        if not steps or steps[-1] != bt.activity:
            steps.append(bt.activity)

    run(sim, 240.0, watch)
    door = sim.room.doors[bar.DOOR_ID]
    assert not sim.life.open["bar"] and not door.unlocked, "shut and locked after three"
    assert not sim.life.tv("bar").on
    order = ["leaving the bar", "locking up the bar", "walking to the metro", "showing a pass",
             "climbing the stairs to the platform", "waiting for the train",
             "getting on the train"]  # fmt: skip
    k = 0
    for step in order:
        k = steps.index(step, k)
    assert bt.riding and bt.hidden, "home on the metro"


def test_the_bartender_comes_back_on_the_metro_and_opens_at_five():
    sim = street(15.9)
    bt = sim.npc_with_role("bartender")
    bt.riding = bt.hidden = True  # (home across town)
    opened = []

    def watch(s):
        if s.life.open["bar"] and not opened:
            opened.append(s.clock.hour)

    run(sim, 200.0, watch)
    assert not bt.riding and in_the_bar(sim.life, bt.x, bt.z)
    assert opened and 17.0 <= opened[0] < 17.5, opened


def test_a_rider_is_still_aboard_after_a_save_and_a_load(tmp_path):
    from cyberduke.systems.savegame import load_game, save_game

    sim = street(2.85)
    bt = sim.npc_with_role("bartender")
    bt.riding = bt.hidden = True
    bt.x, bt.y, bt.z = 30.0, metro.FLOOR_Y, 49.3
    path = save_game(sim, tmp_path)
    other = street(12.0)
    load_game(other, path)
    again = other.npc_with_role("bartender")
    assert again.riding and again.hidden


def test_the_gate_flaps_fold_back_for_a_pass():
    m = Metro(LEVEL.metro)
    assert m.gate(0) == 0.0
    m.open_gate(0)
    m.update(0.4)
    assert m.gate(0) == 1.0 and m.gate(1) == 0.0
    m.update(Metro.GATE_OPEN)
    assert m.gate(0) == 0.0


def test_the_train_is_drawn_on_the_beam_with_its_doors_opening():
    from tests.test_traffic import _BatchRecorder

    from cyberduke.render.metro import MetroRenderer

    m = Metro(LEVEL.metro)
    m.phase, m.x, m.dwell = "stopped", metro.STOP_X, 0.0
    ctx = _BatchRecorder()
    MetroRenderer().draw(ctx, m, {})
    cars = ctx.drawn
    assert len(cars) == 2
    for pts in cars:
        assert pts[:, 1].min() > 6.6 and pts[:, 1].max() < 11.0, "up on the beam"
        assert abs(pts[:, 2].mean() - metro.TRACK_Z) < 0.3
    shut = [b for b in ctx.inner.boxes if b[2] == "cube"]
    m.dwell = Metro.DOORS + 1.0
    ctx2 = _BatchRecorder()
    MetroRenderer().draw(ctx2, m, {})
    opened = [b for b in ctx2.inner.boxes if b[2] == "cube"]
    moved = max(
        math.dist(tuple(a[0].mean(axis=0)), tuple(b[0].mean(axis=0)))
        for a, b in zip(shut, opened, strict=True)
    )
    assert moved > 0.4, "the leaves slide apart"


# ============================================================================ the stairs
def _boot_soles(pose):
    import numpy as np

    from cyberduke.animation.rig import BodyKit
    from cyberduke.animation.skeleton import body_boxes

    kit = BodyKit(armor=False, visor=False, core_glow=False, holster=False, blaster=False)
    out = {}
    for name, m in body_boxes(pose, kit):
        if name.startswith("boot_") or name.startswith("shin_"):
            ends = (-0.5, 0.5)
            corners = np.array([(a, b, c, 1.0) for a in ends for b in ends for c in ends])
            out[name] = (m @ corners.T).T[:, :3]
    return out


@pytest.mark.parametrize("rise", [metro.RISE, -metro.RISE])
def test_on_the_stairs_a_foot_on_every_tread(rise):
    """Up (or down) the station's stair a tread at a time (``animation.street.stairs``):
    no boot or shin ever goes into a step, there is always a foot on a tread, and the
    feet take turns - one planted, the other lifted over the edge and put down two treads
    on (going up, the knee comes up with it)."""
    from cyberduke.animation import street as clips

    run, steps = metro.RUN, metro.STEPS
    knees = []
    for i in range(int((steps + 0.5) / 0.05) + 1):
        p = -0.5 + i * 0.05
        pose = clips.stairs(p, rise, run, steps)
        body_y, body_z = clips.stair_height(p, rise, steps), p * run
        lowest = []
        for name, pts in _boot_soles(pose).items():
            gaps = []
            for _, y, z in pts:
                z, y = z + body_z, y + body_y
                tread = max(0, min(steps, math.floor(z / run) + 1)) if z >= 0.0 else 0
                gaps.append(y - rise * tread)
            assert min(gaps) > -0.02, f"p={p:.2f}: {name} into a step ({min(gaps):.3f})"
            if name.startswith("boot_"):
                lowest.append(min(gaps))
        assert min(lowest) < 0.03, f"p={p:.2f}: both feet in the air"
        knees.append((pose.leg_l[2], pose.leg_r[2]))
    if rise > 0:
        assert max(k for pair in knees for k in pair) > 60.0, "the knee lifts for the step"
    lefts = [k[0] for k in knees]
    assert max(lefts) - min(lefts) > 25.0, "the legs take turns"


def test_people_climb_the_stairs_a_step_at_a_time():
    """Off home on the metro the bartender climbs the station's stair - not glides up it:
    the stair clip's legs, his height rising tread by tread with where he is along it."""
    from cyberduke.ai.alley import Stairs

    sim = street(2.95)
    bt = sim.npc_with_role("bartender")
    heights, on_stairs = [], []

    def watch(s):
        act = bt.storyteller.act if bt.storyteller is not None else None
        if isinstance(act, Stairs):
            on_stairs.append((bt.x, bt.y, bt.pose.leg_l[2], bt.pose.leg_r[2]))
            heights.append(bt.y)

    run(sim, 110.0, watch)
    assert on_stairs, "he went up the stairs"
    assert heights[-1] == pytest.approx(metro.FLOOR_Y, abs=0.01) and heights[0] < 0.2
    assert all(b >= a - 1e-9 for a, b in zip(heights, heights[1:], strict=False)), "up only"
    for x, y, *_ in on_stairs:  # (where he is along it says how high)
        p = (x - metro.STAIR_X0) / metro.RUN
        assert y == pytest.approx(max(0.0, min(metro.FLOOR_Y, (p + 0.5) * metro.RISE)), abs=0.02)
    assert max(max(left, right) for _, _, left, right in on_stairs) > 60.0, "knees up"
