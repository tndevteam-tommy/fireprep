"""Street life (ai.alley): the people of the night market going about their business -
the chef making sushi from the fish in the cooler to the box on the counter, the dealer
wrapping a sale and fetching a parcel, customers ordering, paying, waiting, taking,
eating and binning the box, the courier's parcel ending up in the resident's hands, two
people stopping for a chat - each in a sensible order, with things always somewhere
(in a hand or lying about, never both, never lost)."""

from __future__ import annotations

import math
from collections import Counter as Tally
from dataclasses import replace

import pytest

from cyberduke.ai import Civilian
from cyberduke.ai.alley import NEEDS, STORIES, AlleyLife, Away
from cyberduke.levels import build_night_market
from cyberduke.levels.loader import level_from_dict, level_to_dict
from cyberduke.render.items import ITEMS
from cyberduke.simulation import InputState, Simulation

DT = 1 / 30
LEVEL = build_night_market()
RUN = 300.0  # five minutes of street life


def _aboard(sim, p) -> bool:
    """Inside a metro train standing at the station, in through a door and past the back
    of its vestibule - in the dark of the car (``ai.nightlife``): the only place anybody
    ever goes out of sight (or comes back into it)."""
    m = sim.metro
    if m is None or not m.stopped:
        return False
    from cyberduke.world.vehicles import CAR_HALF_WIDTH, VESTIBULE_BACK

    d = m.data
    in_a_door = any(abs(p.x - x) < 0.7 for x in m.door_xs())
    inside = d.track_z + VESTIBULE_BACK + 0.2 < p.z < d.track_z + CAR_HALF_WIDTH
    return in_a_door and inside and abs(p.y - d.floor_y) < 0.05


class Recording:
    """What everybody was doing, frame by frame, over one long run."""

    def __init__(self, seconds: float) -> None:
        self.sim = Simulation(build_night_market())
        # (the evening stands still: these are the stories, not the hours people keep -
        # see tests/test_schedule.py)
        self.sim.clock.rate = 0.0
        self.steps: dict[str, list[tuple[float, str]]] = {}
        self.hands: dict[str, set[str]] = {}
        self.items_seen: Tally = Tally()
        self.min_gap = 99.0
        self.knife_states: set[str] = set()
        self.door_swung = False
        self.frames = 0
        self.treading: list[tuple[float, str, str]] = []  # walking on the spot for 2 s
        self.vanished: list[tuple[str, float, float]] = []  # where people went out of sight
        self.boarded: set[str] = set()  # who got on a metro train (and out of sight in it)
        self.jumps: list[tuple[float, str, float]] = []  # further in a frame than anybody walks
        self.went_home: set[str] = set()  # who went in through their own front door
        self.tossed: list[tuple[str, float, float]] = []  # where boxes went in the bin
        self.door_clips: list[tuple] = []  # somebody's arm (or parcel) in door 16
        self.held: dict[str, list[tuple]] = {}  # what each had in (left, right) hand, in turn
        self.posed: set[tuple] = set()  # what anybody was drawn holding (left, right)
        trail: dict[str, list[tuple[float, float, float]]] = {}
        people = self.sim.civilians + self.sim.merchants
        for k in range(int(seconds / DT)):
            self.sim.player.x, self.sim.player.z = 0.0, -1.2  # the player keeps out of it
            hidden = {id(p): p.hidden for p in people}
            where = {id(p): (p.x, p.z) for p in people}
            self.sim.update(DT, InputState())
            t = k * DT
            self.frames += 1
            for p in people:
                who = p.civilian_id if isinstance(p, Civilian) else p.merchant_id
                steps = self.steps.setdefault(who, [])
                if not steps or steps[-1][1] != p.activity:
                    steps.append((t, p.activity))
                    if p.activity == "throwing the box away":
                        self.tossed.append((who, p.x, p.z))
                if p.hidden and not hidden[id(p)]:
                    if _aboard(self.sim, p):
                        self.boarded.add(who)
                    else:
                        self.vanished.append((who, p.x, p.z))
                moved = math.hypot(p.x - where[id(p)][0], p.z - where[id(p)][1])
                if hidden[id(p)] and _aboard(self.sim, p):
                    moved = 0.0  # (getting off a train: out of a door they were behind)
                if moved > 0.15:  # (a brisk walk is 4 cm a frame; a scripted step, 9)
                    self.jumps.append((round(t, 2), who, round(moved, 2)))
                home = self.sim.life.homes.get(p.role)
                if home is not None and self.sim.life.inside(home.area_id, p.x, p.z):
                    self.went_home.add(p.role)
                # frames with the legs going and the body (all but) standing still
                track = trail.setdefault(who, [])
                before = track[-1] if track else (p.x, p.z, p.walk_cycle)
                still = math.hypot(p.x - before[0], p.z - before[1]) < 0.002
                track.append((p.x, p.z, p.walk_cycle, p.walk_cycle > before[2] and still))
                if len(track) > 60:
                    track.pop(0)
                    if not p.hidden and sum(1 for f in track if f[3]) >= 45:
                        self.treading.append((t, who, p.activity))
                        track.clear()
                for item in p.hands.values():
                    if item:
                        self.hands.setdefault(p.role, set()).add(item)
                held = self.held.setdefault(who, [])
                now = (p.hands["l"], p.hands["r"])
                if not held or held[-1] != now:
                    held.append(now)
                self.posed.add((p.pose.hand_l, p.pose.hand_r))
            out = [p for p in people if not p.hidden and p.fall_angle == 0.0]
            for i, a in enumerate(out):
                for b in out[i + 1 :]:
                    self.min_gap = min(self.min_gap, math.hypot(a.x - b.x, a.z - b.z))
            # the knife is always exactly one place: on the board, or in the chef's hand
            chef = next(m for m in self.sim.merchants if m.role == "sushi_chef")
            on_board = sum(1 for i in self.sim.items if i.name == "knife")
            in_hand = sum(1 for v in chef.hands.values() if v == "knife")
            assert on_board + in_hand == 1, (
                f"t={t:.1f}: knife on board {on_board}, in hand {in_hand}"
            )
            self.knife_states.add("hand" if in_hand else "board")
            door = self.sim.room.doors["door16"]
            self.door_swung |= door.open_amount > 0.9
            # nobody reaches (or walks, or holds a parcel) into the door while it is shut
            if door.open_amount < 0.02:
                face = door.obstacle.max_x
                z0, z1 = door.obstacle.min_z, door.obstacle.max_z
                for p in people:
                    outside = face < p.x < face + 1.2  # (not in the flat behind it)
                    if p.hidden or not outside or not z0 - 0.3 < p.z < z1 + 0.3:
                        continue
                    deepest, part = _min_x(_world_boxes(p))
                    if deepest < face - 0.004:
                        self.door_clips.append((round(t, 2), p.role, p.activity, part))
            for item in self.sim.items:
                self.items_seen[item.name] += 1

    def activities(self, who: str) -> list[str]:
        return [a for _, a in self.steps[who]]


@pytest.fixture(scope="module")
def run() -> Recording:
    return Recording(RUN)


def _in_order(seq: list[str], wanted: list[str]) -> bool:
    """``wanted`` appears in ``seq`` as a subsequence (in that order)."""
    it = iter(seq)
    return all(any(w == s for s in it) for w in wanted)


def test_every_role_in_the_alley_has_its_places_and_a_story():
    life = AlleyLife(LEVEL)
    for person in (*LEVEL.civilians, *LEVEL.merchants):
        assert person.role in STORIES, person.role
        assert life.has(*NEEDS[person.role]), f"{person.role} is missing a place"
    sim = Simulation(LEVEL)
    assert all(p.has_story for p in sim.civilians + sim.merchants)


def test_the_chef_makes_sushi_from_the_fish_to_the_box_in_order(run):
    chef = run.activities("sushi_and_wok")
    assert _in_order(
        chef,
        [
            "taking the order",
            "going to the fish cooler",
            "taking a fish out of the cooler",
            "laying the fish on the board",
            "picking up the knife",
            "cutting the fish",
            "putting the knife down",
            "taking a box",
            "packing the sushi",
            "closing the box",
            "handing the order over",
        ],
    ), chef
    assert "tossing chow mein in the wok" in chef, "and noodles, for whoever orders them"
    assert {"wiping the counter down", "calling out to passers-by"} & set(chef), "chores in between"
    assert run.knife_states == {"board", "hand"}, "the knife is picked up and put back"
    assert {"fish", "knife", "takeaway_open", "takeaway", "noodles", "rag"} <= run.hands[
        "sushi_chef"
    ]
    for item in ("fish", "sushi", "sushi_open", "takeaway", "knife"):
        assert run.items_seen[item], f"{item} never lay on the counter"


def test_a_customer_orders_pays_waits_takes_eats_and_bins_the_box(run):
    worker = run.activities("civilian_office_worker")
    assert _in_order(
        worker,
        [
            "walking to the sushi bar",
            "pointing at the menu",
            "ordering nigiri",
            "paying",
            "waiting for the order",
            "taking the order",
            "sitting down",
            "eating at the counter",
            "getting off the stool",
            "throwing the box away",
            "going home",
            "unlocking the door",
            "going in",
            "going out",
            "locking up",
        ],
    ), worker
    # (home for a while in between: about the flat, not straight back out of the door)
    inside = worker[worker.index("going in") + 1 : worker.index("going out")]
    assert inside, "pottering about the flat between going in and going out"
    assert {"cred", "takeaway", "sushi_open", "chopsticks"} <= run.hands["office_worker"]
    # (checking the phone while they queue or wait: a phone of their own, in both hands)
    assert ("phone", "phone") in run.posed


def test_the_sushi_is_eaten_a_piece_at_a_time_and_the_box_empties(run):
    """Each piece is picked out of the box with the chopsticks (the box holding one piece
    fewer), carried up to the mouth in them and eaten; the empty box goes in the bin."""
    held = run.held["civilian_office_worker"]
    start = held.index(("sushi_open", "chopsticks"))
    wanted = [
        ("sushi_open", "chopsticks"),
        ("sushi_open3", "chopsticks_sushi"),
        ("sushi_open3", "chopsticks"),
        ("sushi_open2", "chopsticks_sushi"),
        ("sushi_open2", "chopsticks"),
        ("sushi_open1", "chopsticks_sushi"),
        ("sushi_open1", "chopsticks"),
        ("takeaway_open", "chopsticks_sushi"),
        ("takeaway_open", "chopsticks"),
        (None, "takeaway_open"),
        (None, None),  # in the bin
    ]
    assert held[start : start + len(wanted)] == wanted, held[start : start + len(wanted)]
    noodles = [h for who in run.held for h in run.held[who] if h[1] == "chopsticks_noodles"]
    assert noodles, "and the noodles the same way, a chopstickful at a time"
    assert {l for who in run.held for l, _ in run.held[who]} >= {"noodles", "noodles0"}


def test_the_tinkerer_browses_inspects_thinks_haggles_buys_and_goes(run):
    steps = run.activities("civilian_tinkerer")
    assert _in_order(
        steps,
        [
            "browsing the table",
            "picking something up",
            "turning it over in the light",
            "putting it back",
            "thinking it over",
            "haggling",
            "paying",
            "taking it",
            "strolling off with the purchase",
        ],
    ), steps
    dealer = run.activities("cyberware_cheap")
    assert _in_order(dealer, ["haggling", "picking it up", "wrapping it up", "holding it out"])
    assert "gift" in run.hands["tinkerer"] and "gift" in run.hands["junk_dealer"]


def test_the_parcel_goes_from_the_dealer_to_the_courier_to_the_step_to_the_resident(run):
    courier = run.activities("civilian_courier")
    assert _in_order(
        courier,
        ["asking for the pickup", "taking it", "delivering a parcel to number 16", "up the step",
         "knocking"],
    ), courier  # fmt: skip
    dealer = run.activities("cyberware_cheap")
    assert _in_order(dealer, ["checking the pickup", "lifting out a parcel", "holding it out"])
    resident = run.activities("civilian_resident")
    assert _in_order(
        resident,
        [
            "coming home with the shopping",
            "unlocking the door",
            "pushing the door open",
            "going in",
            "putting the shopping away",
        ],
    ), resident
    # at home he is about the flat (he used to vanish indoors)
    assert {"cooking on the hotplate", "washing up", "at his computer"} & set(resident), resident
    # nobody in: the parcel is left on the step and found later; somebody in: they answer
    # the door and it changes hands on the doorstep (which, depends on where he is when
    # the courier knocks: ``test_the_courier_knocks_on_door_16_without_reaching_through_it``
    # plays the nobody-in one through on its own)
    left = _in_order(
        courier, ["waiting for an answer", "leaving the parcel on the step", "down the step"]
    )
    handed = _in_order(
        courier, ["knocking", "stepping back down", "handing over the parcel", "saying goodnight"]
    )
    assert left or handed, courier
    if left:
        assert _in_order(resident, ["finding a parcel on the step", "reading the label"])
        assert run.items_seen["parcel"], "it lay on the step for a while"
    if handed:
        assert _in_order(
            resident,
            [
                "coming to the door",
                "answering the door",
                "taking the parcel",
                "thanking the courier",
            ],
        ), resident
    assert "parcel" in run.hands["resident"] and "bag" in run.hands["resident"]
    assert run.door_swung, "the door of number 16 opened and shut"


def test_the_stroller_and_the_smoker_stop_for_a_chat(run):
    assert "chatting" in run.activities("civilian_stroller")
    smoker = run.activities("civilian_smoker")
    assert "chatting" in smoker and "having a smoke" in smoker and "scrolling the phone" in smoker
    assert {"cigarette", "phone"} <= run.hands["smoker"]


def test_nobody_walks_through_anybody_and_everybody_keeps_busy(run):
    assert run.min_gap >= 0.4, f"two people {run.min_gap:.2f} m apart"
    for who, steps in run.steps.items():
        if who in run.boarded:
            continue  # (off home on the metro: tests/test_nightlife.py)
        assert len({a for _, a in steps}) >= 6, f"{who} hardly did anything: {steps}"


def test_nobody_walks_on_the_spot(run):
    """Held up behind somebody at the counter, people used to walk on the spot until the
    walk timed out; now they wait their place in the queue or walk round whoever is in
    the way."""
    assert not run.treading, run.treading[:5]


def test_nobody_appears_or_disappears_they_come_and_go_by_their_own_front_doors(run):
    """People used to vanish into thin air at the mouth of the alley (and pop back out of
    it a minute later). Now nobody ever goes out of sight or jumps from one place to
    another: when they leave the alley they walk home, in through their own front door,
    and come back out of it - or (the bartender's regular) walk into a metro train and
    the doors close on them."""
    assert not run.vanished, run.vanished
    assert run.boarded <= {"civilian_regular", "civilian_bartender"}
    assert not run.jumps, run.jumps[:5]
    walkers = {"office_worker", "tinkerer", "courier", "stroller"}
    assert walkers <= run.went_home, walkers - run.went_home
    for who, steps in run.steps.items():
        assert not any(a in ("away", "at work") for _, a in steps), (who, steps)


def test_nothing_goes_through_the_shut_door_of_number_16(run):
    assert not run.door_clips, run.door_clips[:5]


def test_boxes_go_in_the_bin_from_right_next_to_it(run):
    """Off the stool the office worker used to be stuck inside its collider: the walk to
    the bin 'arrived' at once and the box was thrown at a bin ten metres away."""
    bin_spot = run.sim.life.place("bin")
    assert any(who == "civilian_office_worker" for who, _, _ in run.tossed)
    for who, x, z in run.tossed:
        assert math.hypot(x - bin_spot.x, z - bin_spot.z) < 0.8, (who, x, z)


def test_everything_held_can_be_drawn(run):
    held = set().union(*run.hands.values())
    assert held <= set(ITEMS), held - set(ITEMS)
    assert set(run.items_seen) <= set(ITEMS)


def test_the_street_comes_back_as_it_was_on_reset():
    sim = Simulation(build_night_market())
    for _ in range(int(60 / DT)):
        sim.update(DT, InputState())
    sim.reset()
    assert [(i.name, i.x, i.z) for i in sim.items] == [(i.name, i.x, i.z) for i in LEVEL.items]
    assert all(
        not p.hidden and p.hands == {"l": None, "r": None} for p in sim.civilians + sim.merchants
    )
    assert not any(c.orders or c.queue or c.serving for c in sim.life.counters.values())


def test_places_items_and_roles_round_trip_through_the_loader():
    again = level_from_dict(level_to_dict(LEVEL))
    assert again.places == LEVEL.places and again.items == LEVEL.items
    assert [c.role for c in again.civilians] == [c.role for c in LEVEL.civilians]
    assert [m.role for m in again.merchants] == [m.role for m in LEVEL.merchants]


def test_a_role_whose_places_are_missing_just_wanders():
    level = replace(LEVEL, places=tuple(p for p in LEVEL.places if not p.name.startswith("sushi.")))
    sim = Simulation(level)
    worker = next(c for c in sim.civilians if c.role == "office_worker")
    assert not worker.has_story
    x, z = worker.x, worker.z
    for _ in range(int(10 / DT)):
        sim.update(DT, InputState())
    assert math.hypot(worker.x - x, worker.z - z) > 1.0, "wanders like any passer-by"


def _directed(sim, role: str, story):
    """Takes ``role`` off its own story and gives it ``story`` (a generator) instead."""
    person = next(p for p in sim.civilians + sim.merchants if p.role == role)
    person.storyteller.story = story(person)
    person.storyteller.act = None
    person.last_result = None
    return person


def _run_until(sim, done, seconds: float) -> float:
    for k in range(int(seconds / DT)):
        sim.player.x, sim.player.z = 0.0, -1.2
        sim.update(DT, InputState())
        if done():
            return k * DT
    return math.inf


def test_getting_off_the_stool_still_gets_the_box_to_the_bin_and_the_worker_home():
    """Regression: getting off the stool put the office worker 0.5 m from it - still
    inside its collider - so every step 'collided', the walk to the bin gave up at once
    and counted as arrived, the box was thrown from the counter and the walk home ended
    three steps later with him vanishing in the middle of the alley. (Nobody vanishes
    now: he walks all the way home, to number 32, and lets himself in.)"""
    from cyberduke.ai.alley import bin_it, leave_and_return

    sim = Simulation(build_night_market())
    stool, bin_spot = sim.life.place("sushi.stool"), sim.life.place("bin")
    for other in sim.civilians:  # an empty street: nobody else in the way
        if other.role != "office_worker":
            other.x, other.z = 30.0, 80.0

    def story(me):
        me.hold("r", "takeaway")
        yield from bin_it(me, sim.life)
        yield from leave_and_return(me, sim.life, 999.0)

    worker = _directed(sim, "office_worker", story)
    worker.x, worker.z = stool.x + 0.5, stool.z  # where the old glide off the stool ended
    tossed = []

    home = sim.life.homes["office_worker"]

    def done() -> bool:
        if worker.activity == "throwing the box away" and not tossed:
            tossed.append((worker.x, worker.z))
        assert not worker.hidden, "nobody vanishes"
        return sim.life.inside(home.area_id, worker.x, worker.z)

    assert _run_until(sim, done, 150.0) < 150.0, "home: in through his own front door"
    assert tossed and math.hypot(tossed[0][0] - bin_spot.x, tossed[0][1] - bin_spot.z) < 0.8


def test_the_queue_forms_behind_whoever_is_already_in_it():
    """Regression: everybody queueing walked to the one queue spot, so whoever came
    second walked on the spot behind the first until their walk timed out."""
    from cyberduke.ai.alley import QUEUE_STEP, queue_spot

    sim = Simulation(build_night_market())
    counter = sim.life.counters["sushi"]
    queue = sim.life.place("sushi.queue")
    a, b, c = (object() for _ in range(3))
    counter.serving = a
    counter.queue[:] = [b, c]
    assert queue_spot(counter, b, queue) == (queue.x, queue.z)
    assert queue_spot(counter, c, queue) == (queue.x + QUEUE_STEP[0], queue.z + QUEUE_STEP[1])
    counter.queue.remove(b)  # b's turn: c moves up
    assert queue_spot(counter, c, queue) == (queue.x, queue.z)
    assert math.hypot(*QUEUE_STEP) > 0.75, "room for two people, one behind the other"


def _world_boxes(person):
    """Every box of ``person``'s posed body in the world, and the box of a parcel held in
    both hands (as ``render.humanoid`` draws them)."""
    import numpy as np

    from cyberduke.animation.reach import hand
    from cyberduke.animation.skeleton import BodyKit, body_boxes, rotation

    kit = BodyKit(armor=False, visor=False, core_glow=False, holster=False, blaster=False)
    h = person.spec.height
    base = np.identity(4)
    base[:3, 3] = (person.x, person.y, person.z)
    base = base @ rotation(person.yaw, 0, 1, 0) @ np.diag((h, h, h, 1.0))
    boxes = body_boxes(person.pose, kit, base)
    if person.pose.hand_l == "parcel" and person.pose.hand_r == "parcel":
        mid = (hand(person.pose, "l")[0] + hand(person.pose, "r")[0]) / 2.0
        m = np.identity(4)
        m[:3, 3] = mid  # held by its sides half way up: the parcel's middle
        boxes.append(("parcel", base @ m @ np.diag((0.34, 0.2, 0.26, 1.0))))
    return boxes


def _min_x(boxes) -> tuple[float, str]:
    """How far to -x the boxes reach, and which one does."""
    import numpy as np

    corners = np.array(
        [[x, y, z, 1.0] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]
    ).T
    return min((float((m @ corners)[0].min()), name) for name, m in boxes)


def test_the_courier_knocks_on_door_16_without_reaching_through_it():
    """Regression: the courier stood on the step and rapped at a point 0.5 m ahead - a
    hand's length beyond the door, his forearm sunk in it - with the parcel held out in
    both hands through it as well. He knocks on its face now, the parcel under his arm,
    and sets it down in the corner of the step clear of the door and its frame, letting
    go of it exactly where it then lies."""
    import numpy as np

    from cyberduke.animation.reach import hand
    from cyberduke.animation.street import TWO_HANDED_GRIP

    sim = Simulation(build_night_market())
    life = sim.life
    courier = next(c for c in sim.civilians if c.role == "courier")
    door = sim.room.doors["door16"]
    face = door.obstacle.max_x  # the shut leaf's face
    drop = life.place("door16.parcel")
    seen, let_go = set(), []

    def stays_in(me):  # nobody in: the parcel is left on the step
        while True:
            yield Away(1e9)

    _directed(sim, "resident", stays_in)
    for _ in range(int(120 / DT)):
        sim.player.x, sim.player.z = 0.0, -1.2
        was = courier.hands["r"]
        sim.update(DT, InputState())
        act = courier.activity
        if act in (
            "up the step",
            "knocking",
            "waiting for an answer",
            "leaving the parcel on the step",
        ):
            seen.add(act)
            deepest, part = _min_x(_world_boxes(courier))
            assert deepest >= face - 0.004, f"{act}: {part} {face - deepest:.3f} m into the door"
        if was == "parcel" and courier.hands["r"] is None and act.startswith("leaving"):
            pose = courier.pose
            mid = (hand(pose, "l")[0] + hand(pose, "r")[0]) / 2.0 * courier.spec.height
            r = math.radians(courier.yaw)
            wx = courier.x + mid[0] * math.cos(r) + mid[2] * math.sin(r)
            wz = courier.z - mid[0] * math.sin(r) + mid[2] * math.cos(r)
            let_go.append((wx, courier.y + mid[1], wz))
        if act == "down the step":
            break
    assert {"knocking", "waiting for an answer", "leaving the parcel on the step"} <= seen, seen
    parcel = next(i for i in life.items if i.name == "parcel")
    assert (parcel.x, parcel.y, parcel.z) == (drop.x, drop.y, drop.z)
    grip = TWO_HANDED_GRIP["parcel"]  # held by the sides, half way up
    assert let_go and np.allclose(let_go[0], (drop.x, drop.y + grip, drop.z), atol=0.03), let_go
    # lying there it is clear of the door, its frame posts and the step's edge
    r = math.radians(parcel.yaw)
    half_x = 0.17 * abs(math.cos(r)) + 0.13 * abs(math.sin(r))
    half_z = 0.17 * abs(math.sin(r)) + 0.13 * abs(math.cos(r))
    assert parcel.x - half_x > face + 0.05, "clear of the door leaf"
    post = (door.obstacle.max_z, door.obstacle.max_z + 0.12)  # the frame's latch post
    assert parcel.z + half_z < post[0] or parcel.x - half_x > face + 0.1
    assert parcel.x + half_x <= -3.9 + 1e-6, "on the step, not over its edge"
    stoop = life.place("door16.stoop")
    assert parcel.z - half_z > stoop.z + 0.25, "out of the way of whoever stands on the step"


def test_somebody_standing_about_makes_way_for_somebody_held_up_by_them():
    """A person queueing (or eating, or smoking) who has somebody pressed up against them,
    trying to get by, shuffles a step aside - out of that person's line of travel - and
    no further than a step."""
    from types import SimpleNamespace

    from cyberduke.ai.alley import MAKE_WAY_MAX, Wait, make_way
    from cyberduke.ai.civilian import Civilian

    sim = Simulation(build_night_market())
    me = next(c for c in sim.civilians if c.role == "smoker")
    me.x, me.z, me.frame_obstacles = 0.0, 20.0, []
    # walking up the alley (+z, yaw 0) straight into me, held up for a second now
    walker = SimpleNamespace(x=0.1, z=19.4, yaw=0.0, held_up_by=(0.0, 20.0, 0.35), blocked_time=1.0)
    me.neighbours = [walker]
    act = Wait(lambda: False, lambda t: None, label="queueing for the sushi bar")
    for _ in range(60):
        make_way(me, act, 1 / 30)
        walker.held_up_by = (me.x, me.z, 0.35)
    assert me.x < -0.3, "aside - away from the walker's side of the line"
    assert abs(me.z - 20.0) < 1e-6, "across the walker's path, not along it"
    assert abs(me.x) <= MAKE_WAY_MAX + 1e-6
    busy = Wait(lambda: False, lambda t: None, label="paying")  # at the counter: stays put
    x = me.x
    make_way(me, busy, 1 / 30)
    assert me.x == x
    assert isinstance(me, Civilian)


def test_the_smoker_turns_back_to_his_cigarette_for_a_drag():
    """Nix looks at whoever is about between puffs - but when the cigarette comes up to
    his lips the head comes back round to it for the drag (he used to keep staring at the
    netrunner, the cigarette at a mouth that was looking away), and goes back to looking
    after."""
    sim = Simulation(build_night_market())
    sim.clock.rate = 0.0
    nix = next(c for c in sim.civilians if c.role == "smoker")
    p = sim.player
    dragging, between = [], []
    for _ in range(int(40.0 / DT)):
        r = math.radians(nix.yaw + 70.0)  # (the netrunner off to his side, two metres away)
        p.x, p.z = nix.x + 2.0 * math.sin(r), nix.z + 2.0 * math.cos(r)
        sim.update(DT, InputState())
        if nix.activity != "having a smoke":
            continue
        if nix.pose.inhaling:
            dragging.append(nix.pose.head_yaw)
        elif nix.pose.head_held == 0.0 and nix.hands["r"] == "cigarette":
            between.append(nix.pose.head_yaw)
    assert dragging and between
    assert max(abs(y) for y in dragging) < 3.0, "facing the cigarette through the drag"
    assert max(between) > 45.0, "and looking at the netrunner between puffs"
