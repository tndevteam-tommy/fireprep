"""The Neon Moth and the metro: the bartender's night, a drink at the bar, and getting
on and off the trains (``levels.bar``, ``levels.metro``, ``entities.metro``).

* **The bartender** lives across town: he comes in on the metro in the late afternoon
  (off the train, down the station stair, out through the ticket gates and along the
  street), lets himself in, sets up, and opens at five - unlocks the door, the television
  on. Until three in the morning he works the bar: pulls a pint for whoever orders one and
  sets it down in front of them (walking it out to the tables), collects the empties and
  washes them, and in between wipes down the bar, polishes glasses, has a word with
  whoever is sat at it, looks up at the television (and changes the channel). At three it
  is last orders: once the last customer has gone he clears up, switches off, locks up and
  walks back to the metro, and a train takes him home.
* **A regular** comes in on the metro at half past eight for a few drinks and takes a
  train home again after midnight.
* **The people of the street** drop in now and then for a drink (``ai.alley``: the office
  worker, the tinkerer, the stroller): in through the door, a stool at the bar (or a
  chair at a table if the bar is full), a drink ordered and drunk, the glass left for the
  bartender, and out again.

Nobody appears or disappears: a rider walks into a train's door, the doors close on them
and the train takes them off down the tunnel (``StreetPerson.riding``: aboard, out of
sight); coming back they step out of its doors.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from cyberduke.ai import alley
from cyberduke.ai.alley import (
    NEEDS,
    STORIES,
    AlleyLife,
    BarOrder,
    Glide,
    Play,
    Stairs,
    Story,
    Turn,
    Wait,
    Walk,
    _body,
    _dipped_reach,
    _occupied,
    _there_and_back,
    _up,
    _wait_for_door,
    _walk_indoors,
    face_towards,
)
from cyberduke.animation import street as clips
from cyberduke.animation.pose import HumanoidPose
from cyberduke.core.clock import within
from cyberduke.world.level import Place
from cyberduke.world.vehicles import VESTIBULE_BACK

Clip = Callable[[float], HumanoidPose]

# The bar's hours, and the heights of the things in it (world heights: ``levels.bar``)
OPENS, CLOSES = 17.0, 3.0
COMES_IN = 16.0  # the bartender takes a train in from four (in by a quarter to five)
BAR_DOOR = "bar_door"
BAR_TOP = 1.06
BAR_STOOL_TOP = 0.78
BAR_CHAIR = 0.47
TAP = (-7.24, BAR_TOP + 0.28, 39.6)  # the middle tap's spout (world)
SIPS = 4  # sips to a pint
# the regular's evening at the bar
REGULAR_HOURS = (20.5, 0.5)


def bar_hours(hour: float) -> bool:
    return within(hour, OPENS, CLOSES)


def bar_open(life: AlleyLife) -> bool:
    """Open, and not at last orders: somebody can come in for a drink."""
    return bool(life.open.get("bar")) and not life.bar.last_orders


def in_the_bar(life: AlleyLife, x: float, z: float) -> bool:
    """Inside the Neon Moth (its floor: the bar's places are all in it)."""
    return -11.2 < x < -5.7 and 36.0 < z < 42.2


# ============================================================================ sitting
def _seat_height(seat: str) -> tuple[float, bool]:
    """(the seat's height, whether it is a stool - legs dangling - or a chair)."""
    return (BAR_STOOL_TOP, True) if ".stool" in seat else (BAR_CHAIR, False)


def _sat(me, seat: str, clip: Clip) -> Clip:
    """``clip`` sitting on ``seat`` (on a stool, feet on the ring; on a chair, the arms
    as the clip has them)."""
    height, stool = _seat_height(seat)
    y = _up(me, height)

    def play(t: float) -> HumanoidPose:
        pose = clip(t)
        arms = (pose.arm_l, pose.arm_r, pose.grip_l, pose.grip_r)
        pose = clips.sit(pose, y, stool=stool)
        if not stool:  # (a low seat puts the hands in the lap: not while drinking)
            pose.arm_l, pose.arm_r, pose.grip_l, pose.grip_r = arms
        return pose

    return play


def _beside(seat: str) -> str:
    """The place beside a seat (where one stands before sitting on it)."""
    _, name = seat.split(".")
    return f"bar.by_{name}"


def _served_from(seat: str) -> str:
    """Where the bartender stands to put a drink down for whoever sits on ``seat``."""
    _, name = seat.split(".")
    if name.startswith("stool"):
        return f"bar.serve{name[5:]}"
    return f"bar.by_table{name[5]}"


def _look_at_tv(me, life: AlleyLife) -> Clip:
    """Idling with the head turned up to the bar's television."""
    tv_x, tv_y, tv_z = -8.9, 2.1, 36.1

    def play(t: float) -> HumanoidPose:
        pose = clips.idle(t, me.seed)
        dx, dz = tv_x - me.x, tv_z - me.z
        yaw = math.degrees(math.atan2(dx, dz)) - me.yaw
        yaw = (yaw + 180.0) % 360.0 - 180.0
        pose.head_yaw = max(-70.0, min(70.0, yaw))
        dist = max(0.5, math.hypot(dx, dz))
        pose.head_pitch = -math.degrees(
            math.atan2(tv_y - (me.y + 1.9 * (me.spec.height or 1.0)), dist)
        )
        return pose

    return play


# ============================================================================ the door
def go_into_the_bar(me, life: AlleyLife) -> Story:
    """Along the street to the bar's door, and in (it opens for whoever walks up while
    the bar is open)."""
    front = life.place("bar.front")
    for _ in range(3):
        there = yield Walk(front.x, front.z, label="going to the bar", stop=0.3)
        if there:
            break
    if not bar_open(life):
        return
    yield Turn(front.yaw, label="going into the bar")
    yield _wait_for_door(life, "going into the bar", lambda t: clips.idle(t, me.seed), BAR_DOOR)
    entry = life.place("bar.entry")
    yield Glide(
        entry.x, entry.z, yaw=entry.yaw, y=0.0, duration=1.7, label="going into the bar",
        walk_first=False,
    )  # fmt: skip


def leave_the_bar(me, life: AlleyLife, lock: bool = False) -> Story:
    """To the door, out (the latch opens it from inside), onto the pavement - and, for
    the bartender closing up, the door locked behind him."""
    yield from _walk_indoors(me, life, "bar.entry", label=_walking(me))
    yield Turn(0.0, label="leaving the bar")
    yield _wait_for_door(life, "leaving the bar", lambda t: clips.idle(t, me.seed), BAR_DOOR)
    door_spot, front = life.place("bar.door"), life.place("bar.front")
    yield Glide(
        door_spot.x, door_spot.z, yaw=0.0, duration=1.3, label="leaving the bar",
        walk_first=False,
    )  # fmt: skip
    yield Glide(front.x, front.z, yaw=0.0, duration=0.6, label="leaving the bar", walk_first=False)
    if lock:
        yield Turn(front.yaw, label="locking up the bar")
        yield Play(lambda t: clips.open_door(t, "l", 0.75), 1.0, label="locking up the bar")
        door = life.door(BAR_DOOR)
        if door is not None:
            door.lock()
        yield Turn(0.0, label="locking up the bar")


# ============================================================================ a drink
def drink_at_the_bar(
    me,
    life: AlleyLife,
    rounds: int = 1,
    until: Callable[[], bool] | None = None,
    pace: float = 1.0,
) -> Story:
    """In for a drink (or ``rounds`` of them, or until ``until()``): a seat, a pint
    ordered, waited for, drunk a sip at a time, the glass left for the bartender - and
    out again. Nothing if the bar is shut (or full: they look round and go). ``pace``:
    how long they take over a pint (a regular nurses his)."""
    if not bar_open(life) or life.door(BAR_DOOR) is None:
        return
    if _where_on_the_way(me, life) > 0:  # (up at the station: down to the street first)
        yield from _down_to_the_street(me, life)
    yield from go_into_the_bar(me, life)
    if not in_the_bar(life, me.x, me.z):
        return  # (shut in their face)
    seat = life.bar.take_seat(me, me.rng)
    if seat is None:
        yield Play(lambda t: clips.idle(t, me.seed), 3.0, label="looking for a seat")
        yield from leave_the_bar(me, life)
        return
    yield from _sit_down(me, life, seat)
    n = 0
    while n < rounds:
        if not bar_open(life):
            break
        order = yield from _order(me, life, seat)
        if order is None:
            break
        yield from _drink_up(me, life, seat, order, pace)
        n += 1
        if until is not None and until():
            break
    yield from _get_up(me, life, seat)
    yield from leave_the_bar(me, life)


def _sit_down(me, life: AlleyLife, seat: str) -> Story:
    yield from _walk_indoors(me, life, _beside(seat), label=_walking(me))
    s = life.place(seat)
    yield Turn(face_towards(me, s.x, s.z), label="sitting down at the bar")
    yield Glide(
        s.x, s.z, yaw=s.yaw, duration=0.8, label="sitting down at the bar", walk_first=False
    )


def _get_up(me, life: AlleyLife, seat: str) -> Story:
    side = life.place(_beside(seat))
    yield Glide(side.x, side.z, duration=0.8, label="getting up", walk_first=False)
    life.bar.leave_seat(me)


def _order(me, life: AlleyLife, seat: str) -> Story:
    """A pint ordered from the seat, waited for, and taken up: the order (or None: the
    bartender never came - or it is closing time)."""
    order = BarOrder(me, seat)
    life.bar.orders.append(order)
    yield Play(_sat(me, seat, lambda t: clips.talk(t, me.seed)), 1.6, label="ordering a drink")
    waiting = _sat(me, seat, _look_at_tv(me, life))
    got = yield Wait(
        lambda: order.state == "served", waiting, label="waiting for a drink", timeout=120.0
    )
    if not got or order.glass is None:
        if order.state != "served":
            order.state = "cancelled"
        return None
    glass = order.glass
    at = _body(me, glass.x, glass.y + 0.07, glass.z)
    take = _there_and_back(1.1)

    def lift() -> None:
        life.remove(glass)
        me.hold("r", "beer")

    yield Play(
        _sat(me, seat, lambda t: clips.reach(take(t), at)), 1.1, label="taking the drink",
        events={0.55: lift},
    )  # fmt: skip
    yield Play(_sat(me, seat, lambda t: clips.talk(t, me.seed)), 1.0, label="saying cheers")
    return order


def _drink_up(me, life: AlleyLife, seat: str, order: BarOrder, pace: float = 1.0) -> Story:
    """A pint a sip at a time - a while between sips, talking or looking at the television -
    and the empty glass put down where the drink was, for the bartender."""
    spot = life.place(f"{seat}.glass")
    for k in range(SIPS):
        yield Play(
            _sat(me, seat, lambda t: clips.drink(t, me.seed)), clips.SIP, label="having a drink"
        )
        pause = (4.0 + 5.0 * me.rng.random()) * pace
        listening = k % 2 == 0
        between = (
            (lambda t, listening=listening: clips.talk(t, me.seed, listening=listening))
            if me.rng.random() < 0.5
            else _look_at_tv(me, life)
        )
        yield Play(_sat(me, seat, _holding_low(between)), pause, label="having a drink")
    me.hold("r", "glass")
    at = _body(me, spot.x, spot.y + 0.07, spot.z)
    put = _there_and_back(1.1)

    def down() -> None:
        me.hold("r", None)
        item = life.put("glass", spot.x, spot.y, spot.z)
        life.bar.empties.append(item)

    yield Play(
        _sat(me, seat, lambda t: clips.reach(put(t), at)), 1.1, label="finishing a drink",
        events={0.55: down},
    )  # fmt: skip
    order.state = "drunk"


def _holding_low(clip: Clip) -> Clip:
    """``clip`` with the right hand holding a glass down by the chest (between sips)."""

    def play(t: float) -> HumanoidPose:
        pose = clip(t)
        pose.arm_r = clips.arm_to((-0.2, 1.12, 0.26), "r", bob=pose.bob)
        return pose

    return play


# ============================================================================ the bartender
def bartender(me, life: AlleyLife) -> Story:
    """Comes in on the metro in the late afternoon and opens the bar at five; works it
    until three in the morning; closes up and goes home on the metro."""
    while True:
        hour = life.hour
        on_duty = within(hour, COMES_IN, CLOSES)
        inside = in_the_bar(life, me.x, me.z)
        if me.riding:  # (on the metro: home, or on the way in)
            if on_duty and not within(hour, CLOSES - 0.5, CLOSES):
                yield from ride_in(me, life)
            else:
                yield Wait(
                    lambda: within(life.hour, COMES_IN, CLOSES - 0.5),
                    lambda t: clips.idle(t, me.seed), label="at home across town", timeout=60.0,
                )  # fmt: skip
            continue
        if not on_duty:  # closing time (or about the place out of hours)
            if inside:
                yield from close_the_bar(me, life)
            yield from take_the_metro(me, life)
            continue
        if not inside:
            yield from _let_self_in(me, life)
            continue
        yield from _walk_indoors(me, life, "bar.behind", label=_walking(me))
        if not bar_hours(life.hour):  # in early: setting up until five
            yield from _bar_chores(me, life, until=lambda: bar_hours(life.hour))
            continue
        _open_up(me, life)
        yield from work_the_bar(me, life)


def _let_self_in(me, life: AlleyLife) -> Story:
    """Along the street to the bar's door, key in the lock, in (and locked behind him
    again if it is not five yet)."""
    front = life.place("bar.front")
    for _ in range(3):
        there = yield Walk(front.x, front.z, label="going to work", stop=0.25)
        if there:
            break
    yield Turn(front.yaw, label="unlocking the bar")
    yield Play(lambda t: clips.open_door(t, "l", 0.8), 1.2, label="unlocking the bar")
    door = life.door(BAR_DOOR)
    if door is not None:
        door.unlock()
    yield _wait_for_door(life, "going into the bar", lambda t: clips.idle(t, me.seed), BAR_DOOR)
    entry = life.place("bar.entry")
    yield Glide(
        entry.x, entry.z, yaw=entry.yaw, y=0.0, duration=1.7, label="going into the bar",
        walk_first=False,
    )  # fmt: skip
    if door is not None and not bar_open(life):
        door.lock()  # (not open yet: the latch keeps it while he is in the doorway)


def _open_up(me, life: AlleyLife) -> None:
    """Five o'clock (or later): the door unlocked, the television on, open."""
    if life.open.get("bar"):
        return
    life.open["bar"] = True
    life.bar.last_orders = False
    door = life.door(BAR_DOOR)
    if door is not None:
        door.unlock()
    tv = life.tv("bar")
    tv.switch(True, life.time)
    while tv.showing not in alley.WATCHABLE:
        tv.next_channel(life.time)


def _walking(me) -> str:
    """What walking about the bar looks like from outside."""
    return (
        "walking behind the bar"
        if getattr(me, "role", "") == "bartender"
        else "walking through the bar"
    )


def _has_work(life: AlleyLife) -> bool:
    return life.bar.next_order() is not None or bool(life.bar.empties)


def work_the_bar(me, life: AlleyLife) -> Story:
    """Open until three: orders first, then the empties, chores in between."""
    while bar_hours(life.hour):
        order = life.bar.next_order()
        if order is not None:
            yield from _serve(me, life, order)
            continue
        if life.bar.empties:
            yield from _clear_glass(me, life, life.bar.empties[0])
            continue
        yield from _bar_chores(me, life, until=lambda: _has_work(life) or not bar_hours(life.hour))


def _serve(me, life: AlleyLife, order: BarOrder) -> Story:
    """Pulls a pint at the taps and puts it down in front of whoever ordered it."""
    if order.customer not in life.bar.customers or order.customer.hidden:
        order.state = "cancelled"
        return
    order.state = "pouring"
    yield from _walk_indoors(me, life, "bar.tap", label=_walking(me))
    tap = life.place("bar.tap")
    yield Turn(tap.yaw, label="pulling a pint")
    spout = _body(me, *TAP)
    pour = _there_and_back(2.6, out=0.25)
    me.hold("r", "glass")
    yield Play(
        lambda t: clips.reach(pour(t), (spout[0], spout[1] - 0.12, spout[2])), 2.6,
        label="pulling a pint", events={1.9: lambda: me.hold("r", "beer")},
    )  # fmt: skip
    me.hold("r", "beer")
    yield from _walk_indoors(me, life, _served_from(order.seat), carry="held", label=_walking(me))
    spot = life.place(f"{order.seat}.glass")
    yield Turn(face_towards(me, spot.x, spot.z), label="serving a drink")
    at = _body(me, spot.x, spot.y + 0.07, spot.z)
    put = _there_and_back(1.2)

    def down() -> None:
        me.hold("r", None)
        order.glass = life.put("beer", spot.x, spot.y, spot.z)
        order.state = "served"

    yield Play(
        lambda t: _dipped_reach(put(t), at), 1.2, label="serving a drink", events={0.6: down}
    )
    if order.state != "served":  # (the event fires by the end anyway)
        down()
    yield Play(_smiling(lambda t: clips.talk(t, me.seed)), 1.2, label="there you go")


def _smiling(clip: Clip) -> Clip:
    def play(t: float) -> HumanoidPose:
        pose = clip(t)
        if pose.expression in (None, "neutral"):
            pose.expression = "smile"
        return pose

    return play


def _clear_glass(me, life: AlleyLife, glass) -> Story:
    """An empty glass: over to it, picked up, taken to the back bar's sink and washed."""

    def away(s: str) -> float:
        spot = life.place(f"{s}.glass")
        return math.hypot(spot.x - glass.x, spot.z - glass.z)

    seat = min(life.bar.seats, key=away)
    yield from _walk_indoors(me, life, _served_from(seat), label=_walking(me))
    yield Turn(face_towards(me, glass.x, glass.z), label="clearing a glass")
    at = _body(me, glass.x, glass.y + 0.07, glass.z)
    take = _there_and_back(1.1)

    def lift() -> None:
        life.remove(glass)
        if glass in life.bar.empties:
            life.bar.empties.remove(glass)
        me.hold("r", "glass")

    yield Play(
        lambda t: _dipped_reach(take(t), at), 1.1, label="clearing a glass", events={0.55: lift}
    )
    if glass in life.bar.empties:
        lift()
    yield from _walk_indoors(me, life, "bar.sink", carry="held", label=_walking(me))
    sink = life.place("bar.sink")
    yield Turn(sink.yaw, label="washing a glass")
    yield Play(lambda t: clips.wipe(t, _up(me, 0.94)), 3.0, label="washing a glass")
    me.hold("r", None)


CHORES = ("wipe", "polish", "tv", "chat", "till")


def _bar_chores(me, life: AlleyLife, until: Callable[[], bool]) -> Story:
    """Something to do behind the bar until ``until()``: wipe it down, polish a glass, a
    word with whoever is sat at it, the till - or a look up at the television (changing the
    channel if nothing is on)."""
    choice = CHORES[int(me.rng.random() * len(CHORES))]
    seconds = 5.0 + 6.0 * me.rng.random()
    sitting = [s for s, who in life.bar.seats.items() if who is not None and ".stool" in s]
    if choice == "chat" and sitting:
        seat = sitting[int(me.rng.random() * len(sitting))]
        spot = life.place(_served_from(seat))
        yield from _walk_indoors(me, life, spot.name, label=_walking(me))
        yield Turn(spot.yaw, label="having a word")
        yield Wait(
            until, lambda t: clips.talk(t, me.seed + 2.0), label="having a word", timeout=seconds
        )
    elif choice in ("wipe", "chat"):
        k = 1 + int(me.rng.random() * 4)
        spot = life.place(f"bar.serve{k}")
        yield from _walk_indoors(me, life, spot.name, label=_walking(me))
        yield Turn(spot.yaw, label="wiping down the bar")
        me.hold("r", "rag")
        yield Wait(
            until, lambda t: clips.wipe(t, _up(me, BAR_TOP)), label="wiping down the bar",
            timeout=seconds,
        )  # fmt: skip
        me.hold("r", None)
    elif choice == "polish":
        yield from _walk_indoors(me, life, "bar.shelves", label=_walking(me))
        yield Turn(life.place("bar.shelves").yaw, label="polishing a glass")
        me.hold("r", "glass")
        yield Wait(
            until, lambda t: clips.hold_up(t, me.seed), label="polishing a glass", timeout=seconds
        )
        me.hold("r", None)
    elif choice == "till":
        yield from _walk_indoors(me, life, "bar.till", label=_walking(me))
        till = life.place("bar.till")
        yield Turn(till.yaw, label="at the till")
        press = _there_and_back(1.4)
        yield Play(
            lambda t: clips.reach(press(t), (0.05, _up(me, BAR_TOP) + 0.12, 0.45)),
            1.4,
            label="at the till",
        )
        yield Wait(
            until, lambda t: clips.idle(t, me.seed), label="at the till", timeout=seconds / 2.0
        )
    else:  # the television
        yield from _walk_indoors(me, life, "bar.behind", label=_walking(me))
        tv = life.tv("bar")
        if tv.on and tv.showing not in alley.WATCHABLE or me.rng.random() < 0.25:
            tv.next_channel(life.time)
        yield Wait(until, _look_at_tv(me, life), label="watching the bar's TV", timeout=seconds)


def close_the_bar(me, life: AlleyLife) -> Story:
    """Three o'clock: last orders - nobody new comes in; whatever was ordered is served -
    and once the last of them has gone (or after a while), the glasses cleared, the
    television off, out and locked up."""
    life.bar.last_orders = True
    for _ in range(20):  # (what was ordered before three is still poured)
        order = life.bar.next_order()
        if order is None:
            break
        yield from _serve(me, life, order)
    yield from _walk_indoors(me, life, "bar.behind", label=_walking(me))
    yield Wait(
        lambda: not life.bar.customers, _smiling(lambda t: clips.talk(t, me.seed)),
        label="calling last orders", timeout=90.0,
    )  # fmt: skip
    for glass in list(life.bar.empties):
        yield from _clear_glass(me, life, glass)
    for order in life.bar.orders:
        if order.state == "served" and order.glass is not None and order.glass in life.items:
            life.remove(order.glass)  # (left on the counter)
    life.bar.orders.clear()
    life.tv("bar").switch(False, life.time)
    life.open["bar"] = False
    yield from leave_the_bar(me, life, lock=True)
    life.bar.last_orders = False


# ============================================================================ the metro
UP = ("metro", "metro.inside", "gate", "gate.in", "metro.foot", "metro.top", "metro.landing",
      "metro.bridge", "metro.platform")  # fmt: skip


def _lane(me) -> int:
    return 1 + int(getattr(me, "seed", 0)) % 2


def _route_place(life: AlleyLife, name: str, lane: int) -> Place:
    if name.startswith("gate"):
        return life.place(f"metro.gate{lane}{name[4:]}")
    return life.place(name)


def _where_on_the_way(me, life: AlleyLife) -> int:
    """How far up the way to the platform somebody already is (an index into ``UP``)."""
    metro = life.metro
    if me.y > 3.0:  # up top: the landing, the bridge or the platform
        if me.z > 45.5:
            return UP.index("metro.platform")
        if me.z > 42.5:
            return UP.index("metro.bridge")
        return UP.index("metro.landing")
    gates_z = metro.data.gates_z if metro is not None else (41.2, 41.5)
    if 27.0 < me.x < 42.0 and me.z < gates_z[0]:  # (on the paid side of the gates)
        return UP.index("metro.foot") - 1
    if 27.0 < me.x < 42.0 and me.z < 42.6:
        return UP.index("metro.inside")
    return 0


def _glide_to(me, p: Place, label: str, speed: float = 1.1, yaw: float | None = None) -> Glide:
    dist = math.hypot(p.x - me.x, p.z - me.z)
    heading = math.degrees(math.atan2(p.x - me.x, p.z - me.z)) if dist > 0.05 else me.yaw
    return Glide(
        p.x, p.z, yaw=heading if yaw is None else yaw, y=p.y, duration=max(0.5, dist / speed),
        label=label, walk_first=False,
    )  # fmt: skip


def _through_the_gate(me, life: AlleyLife, lane: int, going_in: bool) -> Story:
    """At a ticket gate: the pass to the reader, the flaps open, through."""
    metro = life.metro
    gate = life.place(f"metro.gate{lane}" if going_in else f"metro.gate{lane}.in")
    yield Turn(gate.yaw if going_in else (gate.yaw + 180.0) % 360.0, label="showing a pass")
    tap = _there_and_back(0.9)
    reader = (-0.45, _up(me, 0.95), 0.35)
    yield Play(lambda t: clips.reach(tap(t), reader), 0.9, label="showing a pass")
    if metro is not None:
        metro.open_gate(lane - 1)
    other = life.place(f"metro.gate{lane}.in" if going_in else f"metro.gate{lane}")
    yield _glide_to(me, other, "through the ticket gates")


def take_the_metro(me, life: AlleyLife) -> Story:
    """Off home on the metro: along the street to the station, through the ticket gates,
    up the stair, over the bridge, a wait on the platform - and onto the next train that
    stops, through a door and into the dark of the car; it takes them away."""
    metro = life.metro
    if metro is None:
        yield Play(lambda t: clips.idle(t, me.seed), 5.0, label="standing about")
        return
    lane = _lane(me)
    start = _where_on_the_way(me, life)
    if start == 0:
        street = life.place("metro")
        for _ in range(3):
            there = yield Walk(street.x, street.z, label="walking to the metro", stop=0.3)
            if there:
                break
    for name in UP[max(1, start + 1) :]:
        if name == "gate":
            gate = _route_place(life, "gate", lane)
            yield _glide_to(me, gate, "going into the station")
            yield from _through_the_gate(me, life, lane, going_in=True)
            continue
        if name == "gate.in":
            continue  # (through it already)
        p = _route_place(life, name, lane)
        if name == "metro.top":
            yield from _the_stairs(me, life, up=True)
        yield _glide_to(me, p, "going up to the platform")
    # somewhere to wait along the platform
    spots = [life.place(n) for n in sorted(life.places) if n.startswith("metro.wait")]
    free = [s for s in spots if not _occupied(me, s.x, s.z, margin=0.4)] or spots
    spot = free[int(me.rng.random() * len(free))] if free else life.place("metro.platform")
    yield _glide_to(me, spot, "walking along the platform", yaw=0.0)

    def waiting(t: float) -> HumanoidPose:
        return clips.phone(t, me.seed) if (t % 20) < 12 else clips.idle(t, me.seed)

    yield Wait(lambda: metro.boarding_time, waiting, label="waiting for the train", timeout=600.0)
    if not metro.boarding_time:
        return  # (no train came: the story starts over from here)
    yield from _board(me, life)


INTO_THE_DARK = 0.5  # metres past the back of a door's vestibule: inside the dark car


def _inside_z(life: AlleyLife) -> float:
    """Where a rider is once on the train: past the back of the vestibule, in the dark
    of the car (nothing of them shows through the door)."""
    return life.metro.data.track_z + VESTIBULE_BACK + INTO_THE_DARK


def _board(me, life: AlleyLife) -> Story:
    """To the nearest door, through its vestibule and on into the dark of the car: gone
    from sight (not stood in the doorway facing out, waiting for the doors to shut)."""
    metro = life.metro
    d = metro.data
    door_x = min(metro.door_xs(), key=lambda x: abs(x - me.x))
    metro.start_boarding(me)
    walk = max(0.6, abs(door_x - me.x) / 1.1 + 0.4)
    yield Glide(
        door_x, d.edge_z - 0.3, yaw=0.0, y=d.floor_y, duration=walk, label="getting on the train",
        walk_first=False,
    )  # fmt: skip
    yield Glide(
        door_x, _inside_z(life), yaw=0.0, y=d.floor_y, duration=1.6,
        label="getting on the train", walk_first=False,
    )  # fmt: skip
    metro.board(me)
    me.riding = True
    me.hidden = True
    me.hold_both(None)


def ride_in(me, life: AlleyLife) -> Story:
    """Aboard, until a train stops here - then off it: out of the dark of the car through
    a door's vestibule as the doors open, along the platform, over the bridge, down the
    stair, out through the ticket gates and onto the street."""
    metro = life.metro
    if metro is None:
        me.riding = me.hidden = False
        return
    d = metro.data
    yield Wait(
        lambda: metro.stopped and metro.doors < 0.5 and metro.dwell < metro.DOORS,
        lambda t: clips.idle(t, me.seed), label="on the metro", timeout=600.0,
    )  # fmt: skip
    if not metro.stopped:
        return
    doors = metro.door_xs()
    door_x = doors[int(me.rng.random() * len(doors))]
    me.x, me.z, me.y = door_x, _inside_z(life), d.floor_y
    me.yaw = me.target_yaw = 180.0
    me.hidden = False  # (in the dark of the car: nothing shows yet)
    me.riding = False
    metro.alight(me)
    yield Wait(
        lambda: metro.doors_open,
        lambda t: clips.idle(t, me.seed),
        label="getting off the train",
        timeout=5.0,
    )
    out = Place("off", door_x, d.edge_z - 0.6, yaw=180.0, y=d.floor_y)
    yield _glide_to(me, out, "getting off the train", yaw=180.0)
    yield from _down_to_the_street(me, life)


def _the_stairs(me, life: AlleyLife, up: bool) -> Story:
    """The station's stair, a tread at a time (``alley.Stairs``): to the foot of it (or the
    top) and up (or down) to the other end."""
    d = life.metro.data if life.metro is not None else None
    if d is None or d.stair_steps <= 0:
        return
    (fx, fz), (tx, tz) = d.stair_foot, d.stair_top
    length = math.hypot(tx - fx, tz - fz)
    run = length / d.stair_steps
    ux, uz = (tx - fx) / length, (tz - fz) / length
    if up:
        start, direction, y0 = (fx, fz), (ux, uz), 0.0
        label, pace = "climbing the stairs to the platform", 0.42
    else:
        start, direction, y0 = (tx - ux * run, tz - uz * run), (-ux, -uz), d.floor_y
        label, pace = "going down the stairs", 0.38
    yaw = math.degrees(math.atan2(direction[0], direction[1]))
    yield _glide_to(me, Place("stair", start[0], start[1], yaw=yaw, y=y0), label, yaw=yaw)
    rise = d.floor_y / d.stair_steps * (1.0 if up else -1.0)
    yield Stairs(start, direction, y0, rise, run, d.stair_steps, label=label, pace=pace)


def _down_to_the_street(me, life: AlleyLife) -> Story:
    """From wherever they are in the station down to the street (the way ``UP`` walked
    back down)."""
    lane = _lane(me)
    here = _where_on_the_way(me, life)
    if here <= 0:
        return
    for name in reversed(UP[1 : here + 1]):  # from here .. the doorway
        if name == "gate.in":
            p = _route_place(life, "gate.in", lane)
            yield _glide_to(me, p, "going down to the street")
            yield from _through_the_gate(me, life, lane, going_in=False)
            continue
        if name == "gate":
            continue
        p = _route_place(life, name, lane)
        if name == "metro.foot":
            yield from _the_stairs(me, life, up=False)
        yield _glide_to(me, p, "going down to the street")
    street = life.place("metro")
    yield _glide_to(me, Place("out", street.x, street.z + 0.2, yaw=0.0), "out of the station")


# ============================================================================ a regular
def regular(me, life: AlleyLife) -> Story:
    """Comes in on the metro after half past eight for a few drinks at the Neon Moth, and
    home again on the metro after midnight."""
    while True:
        evening = within(life.hour, *REGULAR_HOURS)
        if me.riding:
            if evening:
                yield from ride_in(me, life)
            else:
                yield Wait(
                    lambda: within(life.hour, *REGULAR_HOURS), lambda t: clips.idle(t, me.seed),
                    label="at home across town", timeout=60.0,
                )  # fmt: skip
            continue
        if evening and bar_open(life):
            yield from drink_at_the_bar(
                me, life, rounds=2, until=lambda: not within(life.hour, *REGULAR_HOURS),
                pace=2.5,
            )  # fmt: skip
            if within(life.hour, *REGULAR_HOURS) and bar_open(life):
                me.hold("r", "cigarette")
                yield Play(
                    lambda t: clips.smoke(t, me.seed), 25.0, label="having a smoke outside the bar"
                )
                me.hold("r", None)
                continue
        yield from take_the_metro(me, life)


# ============================================================================ registering
def a_drink(me, life: AlleyLife) -> Story:
    """For the people of the street (``ai.alley``): a drink at the bar if it is open."""
    yield from drink_at_the_bar(me, life, rounds=1)


STORIES["bartender"] = bartender
STORIES["regular"] = regular
_BAR = ("bar.front", "bar.door", "bar.entry", "bar.floor", "bar.pass", "bar.behind", "bar.tap",
        "bar.sink", "bar.shelves", "bar.till", "bar.serve1", "bar.stool1", "bar.stool1.glass",
        "bar.by_stool1")  # fmt: skip
_METRO = ("metro", "metro.inside", "metro.gate1", "metro.gate1.in", "metro.gate2",
          "metro.gate2.in", "metro.foot", "metro.top", "metro.landing", "metro.bridge",
          "metro.platform", "metro.wait1")  # fmt: skip
NEEDS["bartender"] = _BAR + _METRO
NEEDS["regular"] = _BAR + _METRO

__all__ = [
    "bar_hours",
    "bar_open",
    "bartender",
    "drink_at_the_bar",
    "regular",
    "ride_in",
    "take_the_metro",
]
