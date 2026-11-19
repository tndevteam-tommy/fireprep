"""Street life: the people of the alley going about their business, together.

Everyone on the street with a ``role`` lives out a *story* - a loop of jobs that follow
on from one another, written as a Python generator that yields *acts* (walk there, step
up onto the stoop, chop for four seconds, wait until the order is ready...). The acts do
the moving and the posing; the stories only say what happens next. People meet through
``AlleyLife``, the street's shared state: the orders waiting at each counter, the things
lying about (a knife on a board, a box on the counter, a parcel on a doorstep) and who
has asked whom for a chat.

The cast of the night market (see ``levels.street``):

* **the sushi chef** takes each order, fetches a fish from the cooler, lays it on the
  board, picks up the knife and cuts it, packs the slices in a box, closes it and hands
  it over the counter - or tosses noodles in the wok for a chow mein - and between
  customers wipes the counter down, stirs the wok and calls out to passers-by;
* **the junk dealer** haggles, wraps what was bought and hands it over, fetches parcels
  for the courier from the stock at the back, and tidies the table in between;
* **the office worker** comes down the alley for sushi, eats it sitting at the counter,
  bins the box and goes home; **the tinkerer** browses the junk table, picks things up
  and turns them in the light, thinks, haggles, buys and wanders
  off with it (and grabs some noodles); **the courier** collects a parcel from the dealer
  and carries it to door 16, knocks, and leaves it on the step when nobody answers;
  **the resident** of number 16 comes home with the shopping, later finds the parcel on
  the step and takes it in, and goes out for sushi to eat at home; **the smoker** hangs
  about by the wall with a cigarette and a phone, chats with whoever stops by and gets a
  noodle box now and then; **the stroller** wanders the alley, looks up at the neon,
  browses a little and stops for a chat.

Nobody appears or disappears: whoever leaves the street walks home, lets themselves in at
their own front door and lives there a while (the sofa and the TV, the hob, the desk...)
before coming back out of it (``leave_and_return``, ``keep_hours``, ``stay_home``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field

from cyberduke.ai.schedule import schedule, stall_hours_open
from cyberduke.animation import street as clips
from cyberduke.animation.clips import bang_machine_animation, cook_animation, walk_animation
from cyberduke.animation.pose import HumanoidPose, mix_poses
from cyberduke.animation.ragdoll import lying_on
from cyberduke.animation.reach import arm_to
from cyberduke.config import CLOCK
from cyberduke.core.clock import WorldClock
from cyberduke.core.geometry import yaw_towards
from cyberduke.core.interp import lerp, smoothstep, wrap_angle_deg
from cyberduke.world.level import LevelData, Place

Clip = Callable[[float], HumanoidPose]


# ============================================================================ shared state
@dataclass
class LooseItem:
    """Something lying about: its mesh name (``render.items``), where (the middle of its
    underside) and which way it is turned."""

    name: str
    x: float
    y: float
    z: float
    yaw: float = 0.0
    item_id: str = ""


@dataclass
class Order:
    """One customer's order at a counter: ``dish`` ("nigiri", "noodles", "gift",
    "parcel"), moving ``placed`` -> ``making`` -> ``ready`` -> ``taken``. When it is
    ready it is either lying on the counter (``item``) or held out by the seller."""

    customer: object
    dish: str
    state: str = "placed"
    item: LooseItem | None = None


@dataclass
class Counter:
    """A place to buy something: who is being served, who is on the way up to it, who is
    waiting their turn, and what has been ordered."""

    name: str
    serving: object | None = None  # at the counter
    coming: object | None = None  # their turn: on the way up to it
    queue: list = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)

    def free_for(self, who) -> bool:
        if who in (self.serving, self.coming):
            return True
        waiting = [q for q in self.queue if not getattr(q, "hidden", False)]
        return self.serving is None and self.coming is None and (not waiting or waiting[0] is who)

    def step_up(self, who) -> None:
        """``who``'s turn: on their way up to the counter."""
        if who in self.queue:
            self.queue.remove(who)
        self.coming = who

    def arrive(self, who) -> None:
        """``who`` is at the counter now."""
        if self.coming is who:
            self.coming = None
        self.serving = who

    def leave(self, who) -> None:
        if self.serving is who:
            self.serving = None
        if self.coming is who:
            self.coming = None
        if who in self.queue:
            self.queue.remove(who)

    def next_order(self) -> Order | None:
        return next((o for o in self.orders if o.state == "placed"), None)


@dataclass
class BarOrder:
    """A drink ordered in the bar (``ai.nightlife``) by whoever sits in ``seat`` (a place:
    "bar.stool2", "bar.chair1a"): ``placed`` -> ``pouring`` -> ``served`` (the glass is on
    the counter or the table in front of them: ``glass``) - or ``cancelled``."""

    customer: object
    seat: str
    state: str = "placed"
    glass: LooseItem | None = None


@dataclass
class BarState:
    """The bar's shared state: who sits where, the drinks ordered, the empty glasses left
    about, and whether it is last orders (nobody new comes in, nobody orders again)."""

    seats: dict[str, object | None] = field(default_factory=dict)
    orders: list[BarOrder] = field(default_factory=list)
    empties: list[LooseItem] = field(default_factory=list)
    last_orders: bool = False

    def take_seat(self, who, rng=None) -> str | None:
        """A free seat for ``who`` (at the bar if there is one, else at a table)."""
        for seat, sat in self.seats.items():
            if sat is who:
                return seat
        free = [s for s, sat in self.seats.items() if sat is None]
        if not free:
            return None
        stools = [s for s in free if ".stool" in s]
        pool = stools or free
        seat = pool[int(rng.random() * len(pool))] if rng is not None else pool[0]
        self.seats[seat] = who
        return seat

    def leave_seat(self, who) -> None:
        for seat, sat in self.seats.items():
            if sat is who:
                self.seats[seat] = None

    def next_order(self) -> BarOrder | None:
        return next((o for o in self.orders if o.state == "placed"), None)

    @property
    def customers(self) -> list:
        return [sat for sat in self.seats.values() if sat is not None]


def _is_bar_seat(name: str) -> bool:
    parts = name.split(".")
    return len(parts) == 2 and parts[0] == "bar" and parts[1].startswith(("stool", "chair"))


# The channels a television gets, in the order the remote steps through them: a few
# stations with the snow of dead channels between them (``assets.homes.TV_CHANNELS``)
CHANNEL_LINEUP = ("news", "static", "cartoon", "ads", "static", "music", "static")
WATCHABLE = frozenset({"news", "cartoon", "music"})  # what holds somebody's attention


@dataclass
class TvSet:
    """A flat's television (one in every home, ``AlleyLife.tvs``, by the home's area id):
    switched on or off, on which channel (an index into ``CHANNEL_LINEUP``), and when
    the channel last changed (``AlleyLife.time``: a moment of snow while it retunes)."""

    on: bool = False
    channel: int = 0
    changed: float = -1e9

    @property
    def showing(self) -> str:
        """What is on: "off", or the channel ("news", "static", "cartoon"...)."""
        return CHANNEL_LINEUP[self.channel % len(CHANNEL_LINEUP)] if self.on else "off"

    def switch(self, on: bool, now: float) -> None:
        if on != self.on:
            self.on = on
            self.changed = now

    def next_channel(self, now: float) -> None:
        self.channel = (self.channel + 1) % len(CHANNEL_LINEUP)
        self.changed = now


@dataclass(frozen=True)
class Home:
    """Somebody's flat, as their story needs it: the front door, the private area, the
    prefix of its places ("apt" for number 16: "apt.entry", "apt.kitchen"...; "h31" for
    number 31), where to stand outside (``front``) and at the door to unlock it
    (``stoop`` - up the step, at number 16)."""

    door_id: str
    area_id: str
    prefix: str
    front: str
    stoop: str

    def place(self, name: str) -> str:
        return f"{self.prefix}.{name}"


class AlleyLife:
    """The street's shared state (one per level with role-playing people)."""

    def __init__(self, level: LevelData) -> None:
        self.places: dict[str, Place] = {p.name: p for p in level.places}
        self._initial = level.items
        self.areas = {a.area_id: a for a in level.private_areas}  # people's homes
        self.routes: dict[str, list[str]] = {}  # the place graph (``LevelData.routes``)
        for a, b in level.routes:
            self.routes.setdefault(a, []).append(b)
            self.routes.setdefault(b, []).append(a)
        self.doors: dict = {}  # the live doors, by id (the simulation fills this in)
        self.machines: dict = {}  # the live vending machines, by id (likewise)
        self.metro = None  # the live metro (``entities.metro``), if the level has one
        # the time of day (the simulation's clock; a lone one for a life on its own)
        self.clock = WorldClock(CLOCK.new_game_at, CLOCK.minutes_per_second)
        # everybody's home, by the role of whoever lives there
        self.homes: dict[str, Home] = {}
        for area in level.private_areas:
            if not area.owner_role or area.door_id is None:
                continue
            number = area.area_id.removeprefix("flat")
            if f"h{number}.entry" in self.places:
                home = Home(
                    area.door_id, area.area_id, f"h{number}", f"h{number}.front", f"h{number}.door"
                )
            elif "apt.entry" in self.places and "door16.stoop" in self.places:
                home = Home(area.door_id, area.area_id, "apt", "door16.front", "door16.stoop")
            else:
                continue
            self.homes[area.owner_role] = home
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        self.items: list[LooseItem] = [
            LooseItem(i.name, i.x, i.y, i.z, i.yaw, i.item_id) for i in self._initial
        ]
        self.counters: dict[str, Counter] = {"sushi": Counter("sushi"), "junk": Counter("junk")}
        self.chats: dict[object, object] = {}  # who has been asked -> who asked
        self.flags: dict[str, object] = {}
        self.swings: dict[str, tuple[float, float, float, float]] = {}  # id -> from, to, t0, dur
        # the stalls: open (their merchant behind them) or shut up for the night
        self.open: dict[str, bool] = {"sushi": True, "junk": True}
        # the television in every home (off until somebody sits down in front of it)
        self.tvs: dict[str, TvSet] = {area_id: TvSet() for area_id in self.areas}
        # the Neon Moth (``ai.nightlife``): shut until the bartender opens up
        self.bar = BarState(seats={name: None for name in self.places if _is_bar_seat(name)})
        self.open["bar"] = False
        self.noise: tuple[float, float, float, str] | None = None  # (x, z, when, source)

    def update(self, dt: float) -> None:
        self.time += dt
        for item_id, (a, b, t0, dur) in list(self.swings.items()):
            item = self.item(item_id)
            u = (self.time - t0) / dur
            if item is not None and u >= 0.0:
                item.yaw = lerp(a, b, smoothstep(0.0, 1.0, min(1.0, u)))
            if u >= 1.0:
                del self.swings[item_id]

    def swing(self, item_id: str, yaw: float, duration: float = 0.8, delay: float = 0.0) -> None:
        """Turns a thing (a door on its hinge) to ``yaw`` over ``duration`` seconds."""
        item = self.item(item_id)
        if item is not None:
            start = self.swings[item_id][1] if item_id in self.swings else item.yaw
            self.swings[item_id] = (start, yaw, self.time + delay, duration)

    @property
    def hour(self) -> float:
        """The time of day (hours since midnight)."""
        return self.clock.hour

    def place(self, name: str) -> Place:
        return self.places[name]

    def has(self, *names: str) -> bool:
        return all(n in self.places for n in names)

    def item(self, item_id: str) -> LooseItem | None:
        return next((i for i in self.items if i.item_id == item_id), None)

    def put(
        self, name: str, x: float, y: float, z: float, yaw: float = 0.0, item_id: str = ""
    ) -> LooseItem:
        item = LooseItem(name, x, y, z, yaw, item_id)
        self.items.append(item)
        return item

    def remove(self, item: LooseItem | None) -> None:
        if item is not None and item in self.items:
            self.items.remove(item)

    def door(self, door_id: str):
        return self.doors.get(door_id)

    def tv(self, area_id: str) -> TvSet:
        """The television in the home ``area_id`` (a set nobody has is made on asking)."""
        return self.tvs.setdefault(area_id, TvSet())

    def broken_machine(self):
        """The vending machine in the street, if it is out of order (shorted, and nobody
        has kicked it back to life yet)."""
        return next((m for m in self.machines.values() if getattr(m, "broken", False)), None)

    def inside(self, area_id: str, x: float, z: float) -> bool:
        area = self.areas.get(area_id)
        return area is not None and area.contains(x, z)

    def route(self, start: str, goal: str) -> list[str]:
        """The places from ``start`` to ``goal`` along the place graph (breadth first),
        without ``start``; [] if there is no way."""
        if start == goal:
            return []
        came: dict[str, str] = {start: start}
        frontier = [start]
        while frontier:
            nxt = []
            for name in frontier:
                for other in self.routes.get(name, ()):
                    if other not in came:
                        came[other] = name
                        nxt.append(other)
            if goal in came:
                break
            frontier = nxt
        if goal not in came:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        return path[-2::-1]

    def home_of(self, npc) -> Home | None:
        return self.homes.get(getattr(npc, "role", ""))

    def hear(self, x: float, z: float, source: str) -> None:
        """Something made a noise (the vending machine shorting out): whoever cares can
        look into it (``noise``)."""
        self.noise = (x, z, self.time, source)

    def heard(self, source: str, within: float = 10.0) -> bool:
        """Whether ``source`` made a noise in the last ``within`` seconds."""
        return (
            self.noise is not None
            and self.noise[3] == source
            and self.time - self.noise[2] <= within
        )

    def nearest_on_routes(self, x: float, z: float, prefix: str = "") -> str | None:
        names = [n for n in self.routes if n in self.places and n.startswith(prefix)]
        if not names:
            return None
        return min(names, key=lambda n: math.hypot(self.places[n].x - x, self.places[n].z - z))


# ============================================================================ acts
class Act:
    """One thing somebody is doing. ``step`` returns True when it is over; ``result`` is
    what the story gets back."""

    label = ""
    result: object = None

    def begin(self, npc) -> None:
        """Called once, the frame the act starts."""

    def restart(self, npc) -> None:
        """Taken up again after an aside (``Storyteller``: something caught their eye on
        the way): by default, begun afresh from wherever they are now."""
        self.begin(npc)

    def step(self, npc, dt: float) -> bool:
        raise NotImplementedError


class Walk(Act):
    """Walk to (x, z) round everything in the way, steering round people. ``carry`` is
    how something is being carried ("two", "one", "held").

    Arrival is checked against where the walker really is: the router gives up on a goal
    it has been pinned short of for a couple of seconds and calls that arrival, which used
    to end the walk on the spot - the box tossed at a bin ten metres off, a walk home up
    the main road over after three steps. A walk pinned short of its goal plans again
    from where it stands (``RETRIES`` times); one held up by somebody standing on the goal
    is over (it is as near as it gets). ``patience`` (by default generous for the
    distance) bounds the whole walk. Result: whether it got there."""

    RETRIES = 4
    TAKEN_NEAR = 1.2  # this near a spot somebody is standing on, the walk is over

    def __init__(
        self,
        x: float,
        z: float,
        label: str = "walking",
        stop: float = 0.3,
        carry: str | None = None,
        patience: float | None = None,
    ) -> None:
        self.x, self.z, self.label, self.stop, self.carry = x, z, label, stop, carry
        self.patience = patience
        self.time = 0.0
        self.retries = 0

    def begin(self, npc) -> None:
        if self.patience is None:
            distance = math.hypot(self.x - npc.x, self.z - npc.z)
            self.patience = 20.0 + 2.5 * distance / max(0.3, npc.walk_speed)

    def restart(self, npc) -> None:
        """On with the walk from wherever they got to (a fresh route, fresh patience)."""
        self.time = 0.0
        self.retries = 0
        self.patience = None
        npc._clear_route()
        self.begin(npc)

    def step(self, npc, dt: float) -> bool:
        self.time += dt
        arrived = npc.navigate_towards(
            self.x, self.z, dt, npc.frame_obstacles, stop_dist=self.stop, avoid=npc.frame_bodies
        )
        left = math.hypot(self.x - npc.x, self.z - npc.z)
        short = left > self.stop + 0.35
        if not arrived and left < self.TAKEN_NEAR and _occupied(npc, self.x, self.z, margin=0.1):
            # somebody is standing on the very spot: this is as near as it gets
            npc._clear_route()
            npc.arrival = "taken"
            arrived = True
        if arrived and short and npc.arrival == "stalled" and self.retries < self.RETRIES:
            self.retries += 1
            npc._clear_route()  # plan afresh from wherever it got to
            arrived = False
        if self.carry is not None:
            npc.pose = clips.carrying(npc.walk_cycle, self.carry, moving=not arrived)
        timed_out = self.patience is not None and self.time > self.patience
        if arrived or timed_out:
            self.result = not short or (arrived and npc.arrival == "taken")
            return True
        return False


class Glide(Act):
    """A short scripted step (onto a stool or a doorstep, along behind a counter) -
    through the furniture the router would not let it into. Waits if somebody is
    standing right where it is going."""

    def __init__(
        self,
        x: float,
        z: float,
        yaw: float | None = None,
        y: float | None = None,
        duration: float = 0.6,
        label: str = "stepping",
        carry: str | None = None,
        walk_first: bool = True,
    ) -> None:
        self.x, self.z, self.yaw, self.y = x, z, yaw, y
        self.duration, self.label, self.carry = max(duration, 1e-3), label, carry
        self.walk_first = walk_first  # a long way off: route there first (not indoors)
        self.time = 0.0
        self.waited = 0.0
        self.start: tuple[float, float, float, float] | None = None

    FAR = 1.6  # further than this, the way there is walked first (round things, not through)
    STRIDE = 0.3  # shorter than this, it is a shuffle (no walking stride)

    def begin(self, npc) -> None:
        self.start = (npc.x, npc.z, npc.y, npc.yaw)
        self.walked = 0.0

    def step(self, npc, dt: float) -> bool:
        assert self.start is not None
        x0, z0, y0, yaw0 = self.start
        far = math.hypot(self.x - x0, self.z - z0) > self.FAR and self.walk_first
        if self.time == 0.0 and far and self.walked < 25.0:
            # a long way from the spot (held up on the way, say): walk most of it first
            self.walked += dt
            near = npc.navigate_towards(
                self.x, self.z, dt, npc.frame_obstacles, stop_dist=self.FAR * 0.6,
                avoid=npc.frame_bodies,
            )  # fmt: skip
            if near or math.hypot(self.x - npc.x, self.z - npc.z) <= self.FAR * 0.6:
                self.start = (npc.x, npc.z, npc.y, npc.yaw)
            return False
        u = smoothstep(0.0, 1.0, min(1.0, (self.time + dt) / self.duration))
        nx, nz = lerp(x0, self.x, u), lerp(z0, self.z, u)
        # somebody in the way (or standing right where it is going): wait for them
        if self.waited < 12.0 and (
            (self.time == 0.0 and _occupied(npc, self.x, self.z)) or _crowding(npc, nx, nz)
        ):
            self.waited += dt
            npc.pose = clips.idle(self.waited, npc.seed)
            return False
        self.time += dt
        npc.x, npc.z = nx, nz
        if self.y is not None:
            npc.y = lerp(y0, self.y, u)
        if self.yaw is not None:
            npc.yaw = npc.target_yaw = yaw0 + wrap_angle_deg(self.yaw - yaw0) * u
        # a real step or two - not a shuffle round on the spot, where a swinging arm
        # only flails about (into a door beside it, say)
        moving = math.hypot(self.x - x0, self.z - z0) > self.STRIDE
        npc.walk_cycle += dt * 5.0 if moving else 0.0
        if self.carry:
            walking = clips.carrying(npc.walk_cycle, self.carry, moving=True)
            standing = clips.carrying(npc.walk_cycle, self.carry, moving=False)
        else:
            walking, standing = walk_animation(npc.walk_cycle), clips.idle(self.time, npc.seed)
        # into the stride and out of it again: arriving mid-stride at a door, the swing of
        # an arm went into it
        stride = smoothstep(0.0, 0.2, u) * (1.0 - smoothstep(0.7, 1.0, u)) if moving else 0.0
        npc.pose = mix_poses(standing, walking, stride)
        return self.time >= self.duration


class Stairs(Act):
    """Up (or down) a straight stair, a foot on every tread (``animation.street.stairs``):
    from its ``start`` (x, z) - the edge of its first tread, at height ``y0`` - ``steps``
    treads ``run`` deep along ``direction``, each ``rise`` higher (lower, if negative),
    ``pace`` seconds a tread. The body goes where the pose's feet say it is."""

    def __init__(
        self,
        start: tuple[float, float],
        direction: tuple[float, float],
        y0: float,
        rise: float,
        run: float,
        steps: int,
        label: str = "on the stairs",
        pace: float = 0.45,
        carry: str | None = None,
    ) -> None:
        length = math.hypot(*direction) or 1.0
        self.start, self.y0 = start, y0
        self.direction = (direction[0] / length, direction[1] / length)
        self.rise, self.run, self.steps = rise, run, steps
        self.label, self.pace, self.carry = label, pace, carry
        self.p = -0.5  # (the foot of it: a foot on the floor, the other on its way up)

    def place(self, npc) -> None:
        p = min(self.p, float(self.steps))
        npc.x = self.start[0] + self.direction[0] * p * self.run
        npc.z = self.start[1] + self.direction[1] * p * self.run
        npc.y = self.y0 + clips.stair_height(p, self.rise, self.steps)
        npc.yaw = npc.target_yaw = math.degrees(math.atan2(self.direction[0], self.direction[1]))

    def begin(self, npc) -> None:
        self.place(npc)

    def restart(self, npc) -> None:
        self.place(npc)  # (on from the tread it was on)

    def step(self, npc, dt: float) -> bool:
        self.p = min(float(self.steps), self.p + dt / self.pace)
        self.place(npc)
        npc.pose = clips.stairs(self.p, self.rise, self.run, self.steps, npc.seed)
        return self.p >= self.steps


def _occupied(npc, x: float, z: float, margin: float = 0.25) -> bool:
    return any(math.hypot(bx - x, bz - z) < r + margin for bx, bz, r in npc.frame_bodies)


def _crowding(npc, x: float, z: float) -> bool:
    """Would stepping to (x, z) press into somebody - closer than 0.45 m and closing?"""
    for bx, bz, _ in npc.frame_bodies:
        after = math.hypot(bx - x, bz - z)
        if after < 0.45 and after < math.hypot(bx - npc.x, bz - npc.z) - 1e-6:
            return True
    return False


class Turn(Act):
    """Turn on the spot to face ``yaw`` (or a point)."""

    def __init__(self, yaw: float, label: str = "turning") -> None:
        self.yaw, self.label = yaw, label
        self.time = 0.0

    def step(self, npc, dt: float) -> bool:
        self.time += dt
        done = npc.rotate_towards(self.yaw, dt)
        npc.pose = clips.idle(self.time, npc.seed)
        return done or self.time > 3.0


class Play(Act):
    """A gesture ``clip(t)`` for ``duration`` seconds; ``events`` are things that happen
    at given times into it (picking something up, letting go of it)."""

    def __init__(
        self,
        clip: Clip,
        duration: float,
        label: str = "",
        events: dict[float, Callable[[], None]] | None = None,
    ) -> None:
        self.clip, self.duration, self.label = clip, duration, label
        self.events = sorted((events or {}).items())
        self.time = 0.0

    def step(self, npc, dt: float) -> bool:
        self.time += dt
        while self.events and self.events[0][0] <= self.time:
            self.events.pop(0)[1]()
        npc.pose = self.clip(min(self.time, self.duration))
        if self.time >= self.duration:
            for _, event in self.events:  # anything not yet fired still happens
                event()
            self.events = []
            return True
        return False


class Wait(Act):
    """Keep doing ``clip`` until ``until()`` is true (result True) or ``timeout`` runs out
    (result False)."""

    def __init__(
        self,
        until: Callable[[], bool],
        clip: Clip,
        label: str = "waiting",
        timeout: float | None = None,
    ) -> None:
        self.until, self.clip, self.label, self.timeout = until, clip, label, timeout
        self.time = 0.0

    def step(self, npc, dt: float) -> bool:
        self.time += dt
        if self.until():
            self.result = True
            return True
        if self.timeout is not None and self.time >= self.timeout:
            self.result = False
            return True
        npc.pose = self.clip(self.time)
        return False


class Away(Act):
    """Out of sight for a while. No story uses it - nobody on the street goes out of sight:
    people go home through their own front doors (``leave_and_return``, ``stay_home``) -
    but a test can park somebody off the stage with it."""

    label = "away"

    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.time = 0.0

    def begin(self, npc) -> None:
        npc.hidden = True

    def step(self, npc, dt: float) -> bool:
        self.time += dt
        # back round the corner - but not into somebody standing right there (two people
        # who went off the same way came back as one)
        if self.time >= self.duration and not _occupied(npc, npc.x, npc.z, margin=0.45):
            npc.hidden = False
            return True
        return False


Story = Generator[Act, object, None]


# ============================================================================ helpers
def _scale(npc) -> float:
    spec = getattr(npc, "spec", None)
    return getattr(spec, "height", 1.0) or 1.0


def _up(npc, height: float) -> float:
    """A height in the world (a counter's top, a seat) in the npc's own drawn frame - where
    its pose has to put a hand (or its seat) to be there, however tall it is."""
    return height / _scale(npc)


# The tops of things people put their hands on or sit on (world heights)
COUNTER_TOP = 1.01  # the sushi counter
TABLE_TOP = 0.9  # the junk dealer's trestle table
KITCHEN_TOP = 0.93  # a flat's kitchen counter
DESK_TOP = 0.76  # Dell's computer desk
STOOL_TOP = 0.61  # a stool at the counter
SOFA_SEAT = 0.5  # a sagging sofa cushion


def _body(npc, wx: float, wy: float, wz: float) -> tuple[float, float, float]:
    """A world point in the npc's own frame (x its left, z ahead, y up from its feet) -
    the frame its pose is drawn in, so scaled by its height (a tall person's hands are
    further from the floor in the world than the same pose's of a short one)."""
    dx, dz = wx - npc.x, wz - npc.z
    r = math.radians(npc.yaw)
    c, s = math.cos(r), math.sin(r)
    k = 1.0 / _scale(npc)
    # body +z is (sin yaw, cos yaw) in the world, body +x (its left) (cos yaw, -sin yaw)
    return ((dx * c - dz * s) * k, (wy - npc.y) * k, (dx * s + dz * c) * k)


def _world(npc, bx: float, by: float, bz: float) -> tuple[float, float, float]:
    """A point of the npc's own (drawn) frame in the world."""
    r = math.radians(npc.yaw)
    c, s = math.cos(r), math.sin(r)
    k = _scale(npc)
    return (npc.x + (bx * c + bz * s) * k, npc.y + by * k, npc.z + (-bx * s + bz * c) * k)


def _dipped_reach(u: float, target, side: str = "r", dip: float = 0.1) -> HumanoidPose:
    """Reaching down to a counter: the knees give a little so the hand gets there."""
    pose = clips.reach(u, (target[0], target[1] + dip * smoothstep(0.0, 1.0, u), target[2]), side)
    k = smoothstep(0.0, 1.0, u) * dip
    leg, _, bob = clips._squat_legs(k)
    pose.bob = bob
    pose.leg_l, pose.leg_r = list(leg), list(leg)
    pose.foot_l = pose.foot_r = -leg[0] - leg[2]
    # the arm was solved for the standing shoulder: solve again for the dipped one
    hand = clips._mix(
        clips.REST_R if side == "r" else clips.REST_L, target, smoothstep(0.0, 1.0, u)
    )
    if side == "r":
        pose.arm_r = arm_to(hand, "r", bob=bob)  # type: ignore[arg-type]
    else:
        pose.arm_l = arm_to(hand, "l", bob=bob)  # type: ignore[arg-type]
    return pose


def _there_and_back(duration: float, out: float = 0.45) -> Callable[[float], float]:
    """u going 0 -> 1 over the first ``out`` of the time, holding, then back to 0."""

    def u(t: float) -> float:
        f = t / duration
        if f < out:
            return smoothstep(0.0, out, f)
        return 1.0 - smoothstep(0.75, 1.0, f)

    return u


def face_towards(npc, x: float, z: float) -> float:
    return yaw_towards(npc.x, npc.z, x, z)


def approach(spot: Place, back: float = 0.5) -> tuple[float, float]:
    """A point ``back`` metres behind ``spot`` (away from the way it faces): somewhere the
    router can always get to, for the last step up to a counter to be a scripted one."""
    r = math.radians(spot.yaw)
    return spot.x - math.sin(r) * back, spot.z - math.cos(r) * back


# ============================================================================ the stories
# ---------------------------------------------------------------- opening and closing
STALLS = {
    # stall: (where the merchant stands, the back way out, what goes out when it shuts,
    # where that goes)
    "sushi": ("sushi.chef", "sushi.exit", "closed_sign", "sushi.pass"),
    "junk": ("junk.dealer", "junk.exit", "stall_cover", "junk.table"),
}


def _behind_counter(me, spot: Place) -> bool:
    return math.hypot(me.x - spot.x, me.z - spot.z) < 1.2


def _counter_idle(counter: Counter) -> bool:
    return (
        counter.serving is None and counter.coming is None and counter.next_order() is None
        and not any(o.state == "making" for o in counter.orders)
    )  # fmt: skip


def _lane(life: AlleyLife, stall: str) -> list[tuple[float, float]]:
    """The way from behind the counter out of the back of the stall (along the lane)."""
    spot, out = life.place(STALLS[stall][0]), life.place(STALLS[stall][1])
    # along the lane (the merchant's x) to level with the way out, then out
    return [(spot.x, out.z - math.copysign(0.6, out.z - spot.z)), (out.x, out.z)]


STALL_SIGN_YAW = {"sushi": 90.0}  # the counter runs along the alley: the sign faces across it


def _stall_sign(me, life: AlleyLife, stall: str, closed: bool, reach: bool = True) -> Story:
    """Puts out (``closed``) - or takes in - what says a stall is shut: the CLOSED sign on
    the sushi counter, the cover over the junk table."""
    _, _, thing, where = STALLS[stall]
    item = life.item(f"{stall}_closed")
    if closed and item is None:
        if reach:
            at = life.place(where)
            put = _there_and_back(1.2)
            yield Play(
                lambda t: clips.reach(put(t), _body(me, at.x, at.y + 0.1, at.z)),
                1.2,
                label="shutting up the stall",
            )
        at = life.place(where)
        # (the sign faces out over the counter, to the alley - and back at the chef; the
        # cover lies along the table)
        life.put(
            thing, at.x, at.y, at.z, yaw=STALL_SIGN_YAW.get(stall, 0.0), item_id=f"{stall}_closed"
        )
    elif not closed and item is not None:
        if reach:
            at = life.place(where)
            take = _there_and_back(1.2)
            yield Play(
                lambda t: clips.reach(take(t), _body(me, at.x, at.y + 0.1, at.z)),
                1.2,
                label="opening up the stall",
                events={0.6: lambda: life.remove(item)},
            )
        else:
            life.remove(item)


def _close_stall(me, life: AlleyLife, stall: str) -> Story:
    """Closing time: nobody more is served, the sign (or the cover) goes out, the merchant
    comes out from behind the counter by the back way."""
    life.open[stall] = False
    counter = life.counters[stall]
    counter.queue.clear()
    me.hold_both(None)
    spot = life.place(STALLS[stall][0])
    yield Turn(spot.yaw, label="shutting up the stall")
    yield from _stall_sign(me, life, stall, closed=True)
    for x, z in _lane(life, stall):
        yield Glide(
            x,
            z,
            yaw=yaw_towards(me.x, me.z, x, z),
            duration=0.9,
            label="leaving the stall",
            walk_first=False,
        )


def _open_stall(me, life: AlleyLife, stall: str) -> Story:
    """Back to work: round to the back of the stall, in behind the counter, the sign in."""
    spot, out = life.place(STALLS[stall][0]), life.place(STALLS[stall][1])
    for _ in range(3):
        there = yield Walk(out.x, out.z, label="going to open the stall", stop=0.25)
        if there:
            break
    lane = _lane(life, stall)
    for x, z in (lane[0], (spot.x, spot.z)):
        yield Glide(
            x,
            z,
            yaw=yaw_towards(me.x, me.z, x, z),
            duration=0.9,
            label="opening up the stall",
            walk_first=False,
        )
    yield Turn(spot.yaw, label="opening up the stall")
    yield from _stall_sign(me, life, stall, closed=False)
    life.open[stall] = True


# ---------------------------------------------------------------- the sushi chef
def sushi_chef(me, life: AlleyLife) -> Story:
    """Works the sushi bar from 14:00 until 02:00 (``ai.schedule``): takes each order and
    makes it, chores in between. At closing time - once the last customer is served - he
    puts the CLOSED sign out, comes out from behind the counter and goes home round the
    corner; back in the afternoon to open up again."""
    home = life.place("sushi.chef")
    counter = life.counters["sushi"]
    if not _behind_counter(me, home):
        # (a saved game - or the middle of the night: wherever he is, he sorts himself out)
        if stall_hours_open("sushi", life.hour):
            yield from _open_stall(me, life, "sushi")
        else:
            life.open["sushi"] = False
            yield from _stall_sign(me, life, "sushi", closed=True, reach=False)
            yield from stay_home(me, life, lambda: stall_hours_open("sushi", life.hour))
            yield from _open_stall(me, life, "sushi")
    yield Glide(home.x, home.z, yaw=home.yaw, duration=0.1, label="at the counter")
    chores = ("wipe", "wok", "call", "wipe", "tidy")
    k = 0
    while True:
        if not stall_hours_open("sushi", life.hour) and _counter_idle(counter):
            yield from _close_stall(me, life, "sushi")
            yield from stay_home(me, life, lambda: stall_hours_open("sushi", life.hour))
            yield from _open_stall(me, life, "sushi")
            continue
        order = counter.next_order()
        if order is None and counter.serving is not None:
            # somebody at the counter making up their mind: attend to them
            yield Wait(
                lambda: counter.next_order() is not None or counter.serving is None,
                lambda t: clips.talk(t, me.seed, listening=True),
                label="at the counter, listening",
                timeout=20.0,
            )
            continue
        if order is None:
            chore = chores[k % len(chores)]
            k += 1
            yield from _chef_chore(me, life, chore)
            continue
        order.state = "making"
        me.serving = order.customer
        yield Turn(home.yaw, label="taking the order")
        yield Play(lambda t: clips.talk(t, me.seed), 1.6, label="taking the order")
        if order.dish == "noodles":
            yield from _make_noodles(me, life, order)
        else:
            yield from _make_sushi(me, life, order)
        yield from _hand_over(me, life, order)
        me.serving = None


def _chef_chore(me, life: AlleyLife, chore: str) -> Story:
    counter = life.counters["sushi"]
    home = life.place("sushi.chef")

    def busy() -> bool:
        return counter.next_order() is not None or counter.serving is not None

    if chore == "wok":
        wok = life.place("sushi.wok")
        yield Glide(wok.x, wok.z, yaw=wok.yaw, duration=0.8, label="going to the wok")
        yield Wait(busy, cook_animation, label="stirring the wok", timeout=6.0)
        yield Glide(home.x, home.z, yaw=home.yaw, duration=0.8, label="back to the counter")
    elif chore == "call":
        yield Wait(busy, clips.beckon, label="calling out to passers-by", timeout=4.0)
    elif chore == "tidy":
        yield Wait(
            busy,
            lambda t: clips.browse(t, me.seed, table_y=_up(me, COUNTER_TOP), reach_out=0.45),
            label="tidying the counter",
            timeout=5.0,
        )
    else:
        me.hold("r", "rag")
        top = _up(me, COUNTER_TOP)
        yield Wait(busy, lambda t: clips.wipe(t, top), label="wiping the counter down", timeout=7.0)
        me.hold("r", None)
    yield Wait(busy, lambda t: clips.idle(t, me.seed), label="waiting for customers", timeout=2.0)


def _make_sushi(me, life: AlleyLife, order: Order) -> Story:
    home = life.place("sushi.chef")
    board = life.place("sushi.board")
    boxes = life.place("sushi.boxes")
    cooler = life.place("sushi.cooler")
    # a fish out of the cooler
    yield Glide(cooler.x, cooler.z, yaw=cooler.yaw, duration=0.9, label="going to the fish cooler")
    grab = _there_and_back(2.2)
    yield Play(
        lambda t: clips.crouch_reach(grab(t), target=(0.0, 0.7, 0.5)),
        2.2,
        label="taking a fish out of the cooler",
        events={1.1: lambda: me.hold("r", "fish")},
    )
    yield Glide(home.x, home.z, yaw=home.yaw, duration=0.9, label="back to the board", carry="held")
    # onto the board
    spot = _body(me, board.x, board.y, board.z)
    lay = _there_and_back(1.2)
    fish: list[LooseItem] = []

    def lay_fish() -> None:
        me.hold("r", None)
        fish.append(
            life.put("fish", board.x, board.y, board.z, yaw=home.yaw + 90.0, item_id="board_fish")
        )

    yield Play(
        lambda t: _dipped_reach(lay(t), spot),
        1.2,
        label="laying the fish on the board",
        events={0.6: lay_fish},
    )
    # the knife
    knife = life.item("knife")
    knife_spot = _body(me, knife.x, knife.y, knife.z) if knife is not None else spot
    pick = _there_and_back(1.0)

    def take_knife() -> None:
        life.remove(life.item("knife"))
        me.hold("r", "knife")

    yield Play(
        lambda t: _dipped_reach(pick(t), knife_spot),
        1.0,
        label="picking up the knife",
        events={0.5: take_knife},
    )

    def cut() -> None:
        if fish:
            fish[0].name = "sushi"

    board_b = _body(me, board.x, board.y, board.z)
    yield Play(lambda t: clips.chop(t, board_b), 4.0, label="cutting the fish", events={2.4: cut})

    def knife_down() -> None:
        me.hold("r", None)
        if knife is not None:
            life.items.append(knife)

    yield Play(
        lambda t: _dipped_reach(pick(t), knife_spot),
        1.0,
        label="putting the knife down",
        events={0.5: knife_down},
    )
    # a box off the stack, the sushi into it, the lid closed
    stack = _body(me, boxes.x, boxes.y, boxes.z)
    box: list[LooseItem] = []
    pack = (board.x + (boxes.x - board.x) * 0.0, board.y, board.z + 0.28 * _side_sign(home, boxes))

    def take_box() -> None:
        me.hold("r", "takeaway_open")

    def put_box() -> None:
        me.hold("r", None)
        box.append(life.put("takeaway_open", pack[0], pack[1], pack[2], yaw=home.yaw + 90.0))

    yield Play(
        lambda t: _dipped_reach(pick(t), stack), 1.0, label="taking a box", events={0.5: take_box}
    )
    pack_b = _body(me, *pack)
    yield Play(
        lambda t: _dipped_reach(pick(t), pack_b),
        1.0,
        label="opening the box",
        events={0.5: put_box},
    )

    def into_box() -> None:
        if fish:
            life.remove(fish[0])
        if box:
            box[0].name = "sushi_open"

    yield Play(
        lambda t: _dipped_reach(pick(t), board_b),
        1.2,
        label="packing the sushi",
        events={0.7: into_box},
    )

    def close() -> None:
        if box:
            box[0].name = "takeaway"

    yield Play(lambda t: clips.wrap(t, pack_b), 2.0, label="closing the box", events={1.6: close})
    order.item = box[0] if box else None


def _side_sign(home: Place, spot: Place) -> float:
    return 1.0 if spot.z > home.z else -1.0


def _make_noodles(me, life: AlleyLife, order: Order) -> Story:
    home = life.place("sushi.chef")
    wok = life.place("sushi.wok")
    board = life.place("sushi.board")
    yield Glide(wok.x, wok.z, yaw=wok.yaw, duration=0.9, label="going to the wok")
    yield Play(cook_animation, 4.5, label="tossing chow mein in the wok")
    scoop = _there_and_back(1.4)
    yield Play(
        lambda t: _dipped_reach(scoop(t), (0.0, 1.0, 0.5)),
        1.4,
        label="filling a noodle box",
        events={0.8: lambda: me.hold("r", "noodles")},
    )
    yield Glide(
        home.x, home.z, yaw=home.yaw, duration=0.9, label="back to the counter", carry="held"
    )
    spot = _body(me, board.x, board.y, board.z + 0.28 * _side_sign(home, life.place("sushi.boxes")))
    box: list[LooseItem] = []

    def put() -> None:
        me.hold("r", None)
        w = _world(me, *spot)
        box.append(life.put("noodles", w[0], board.y, w[2]))

    put_down = _there_and_back(1.0)
    yield Play(
        lambda t: _dipped_reach(put_down(t), spot),
        1.0,
        label="putting the noodles down",
        events={0.5: put},
    )
    order.item = box[0] if box else None


def _hand_over(me, life: AlleyLife, order: Order) -> Story:
    """Across to the pass with the box and set it down on the counter in front of the
    customer; then wait for them to pick it up."""
    serve = life.place("sushi.serve")
    pass_spot = life.place("sushi.pass")
    item = order.item
    name = item.name if item is not None else "takeaway"
    if item is not None:
        here = _body(me, item.x, item.y, item.z)
        pick = _there_and_back(0.9)

        def lift() -> None:
            life.remove(item)
            me.hold("r", name)

        yield Play(
            lambda t: _dipped_reach(pick(t), here),
            0.9,
            label="picking up the order",
            events={0.45: lift},
        )
    yield Glide(
        serve.x, serve.z, yaw=serve.yaw, duration=0.8, label="bringing the order over", carry="held"
    )
    target = _body(me, pass_spot.x, pass_spot.y, pass_spot.z)
    give = _there_and_back(1.2)

    def set_down() -> None:
        me.hold("r", None)
        order.item = life.put(name, pass_spot.x, pass_spot.y, pass_spot.z, yaw=pass_spot.yaw)
        order.state = "ready"

    yield Play(
        lambda t: _dipped_reach(give(t), target),
        1.2,
        label="handing the order over",
        events={0.6: set_down},
    )
    yield Wait(
        lambda: order.state == "taken",
        lambda t: clips.talk(t, me.seed),
        label="enjoy!",
        timeout=25.0,
    )
    if order.state != "taken":  # nobody came for it: it goes back under the counter
        life.remove(order.item)
        order.state = "taken"
    home = life.place("sushi.chef")
    yield Glide(home.x, home.z, yaw=home.yaw, duration=0.8, label="back to the board")


# ---------------------------------------------------------------- a customer at the sushi bar
def buy_food(me, life: AlleyLife, dish: str) -> Story:
    """Queue at the sushi bar, order (pointing at the menu), pay, wait, take the box off
    the counter and look it over. Leaves the box in the right hand."""
    counter = life.counters["sushi"]
    spot = life.place("sushi.customer")
    queue = life.place("sushi.queue")
    menu = life.place("sushi.menu")
    if not life.open.get("sushi", True):
        return  # (shut: no sushi tonight)
    yield from _queue_up(me, life, counter, queue, "the sushi bar", stall="sushi")
    if not life.open.get("sushi", True):
        counter.leave(me)
        return
    me.hold("l", None)
    counter.step_up(me)
    near = approach(spot)
    yield Walk(near[0], near[1], label="stepping up to the counter", stop=0.3)
    yield Glide(spot.x, spot.z, yaw=spot.yaw, duration=0.6, label="stepping up to the counter")
    counter.arrive(me)
    direction = _body(me, menu.x, menu.y, menu.z)
    yield Play(
        lambda t: clips.point_at(t, (direction[0], direction[1] - 1.58, direction[2])),
        1.8,
        label="pointing at the menu",
    )
    order = Order(me, dish)
    counter.orders.append(order)
    yield Play(lambda t: clips.talk(t, me.seed), 1.5, label=f"ordering {dish}")
    pay = _there_and_back(1.4)
    pass_b = _body(
        me, life.place("sushi.pass").x, clips.COUNTER_Y + 0.1, life.place("sushi.pass").z
    )
    yield Play(
        lambda t: clips.reach(pay(t), pass_b),
        1.4,
        label="paying",
        events={0.0: lambda: me.hold("r", "cred"), 0.9: lambda: me.hold("r", None)},
    )
    yield Wait(
        lambda: order.state in ("ready", "taken"),
        lambda t: clips.phone(t, me.seed) if (t % 12) > 6 else clips.idle(t, me.seed),
        label="waiting for the order",
        timeout=70.0,
    )
    if order.state != "ready" or order.item is None:
        counter.leave(me)
        order.state = "taken"
        return
    here = _body(me, order.item.x, order.item.y, order.item.z)
    take = _there_and_back(1.0)
    item = order.item

    def lift() -> None:
        life.remove(item)
        me.hold("r", item.name)
        order.state = "taken"

    yield Play(
        lambda t: _dipped_reach(take(t), here), 1.0, label="taking the order", events={0.5: lift}
    )
    yield Play(lambda t: clips.talk(t, me.seed), 1.0, label="saying thanks")
    yield from step_aside(me, spot, carry="held", keep_clear=[(queue.x, queue.z)])
    counter.leave(me)


QUEUE_STEP = (0.3, 0.75)  # from one place in a queue to the next one back (x, z)


def queue_spot(counter: Counter, me, queue: Place) -> tuple[float, float]:
    """Where ``me`` stands in ``counter``'s queue: the queue's place for whoever is next,
    a step further back along the counter for each person ahead of them."""
    waiting = [q for q in counter.queue if not getattr(q, "hidden", False)]
    k = waiting.index(me) if me in waiting else len(waiting)
    return queue.x + QUEUE_STEP[0] * k, queue.z + QUEUE_STEP[1] * k


def _queue_up(
    me, life: AlleyLife, counter: Counter, queue: Place, what: str, stall: str = ""
) -> Story:
    """Walk to one's place in the queue (behind whoever is already in it - not into
    them), wait there, and move up a place each time the queue does, until it is one's
    turn (or a minute and a half has gone by)."""
    if me not in counter.queue and me not in (counter.serving, counter.coming):
        # one's place in the queue is taken on getting there, not on setting out (or a
        # late starter from nearby finds a place kept for somebody still up the alley)
        yield Walk(queue.x, queue.z, label=f"walking to {what}", stop=2.5)
        counter.queue.append(me)
    start = life.time
    walked = False

    def shut() -> bool:
        return bool(stall) and not life.open.get(stall, True)

    while not counter.free_for(me) or not walked:
        if shut():
            counter.leave(me)
            return
        spot = queue_spot(counter, me, queue)
        if math.hypot(spot[0] - me.x, spot[1] - me.z) > 0.45:
            yield Walk(spot[0], spot[1], label=f"walking to {what}", stop=0.3)
        walked = True
        if counter.free_for(me) or life.time - start > 90.0:
            return
        yield Wait(
            lambda spot=spot: (
                counter.free_for(me) or queue_spot(counter, me, queue) != spot or shut()
            ),
            lambda t: clips.phone(t, me.seed) if (t % 14) < 7 else clips.idle(t, me.seed),
            label=f"queueing for {what}",
            timeout=max(1.0, 90.0 - (life.time - start)),
        )
        if life.time - start > 90.0:
            return


def step_aside(
    me, spot: Place, carry: str | None = None, keep_clear: Sequence[tuple[float, float]] = ()
) -> Story:
    """Out of the way of whoever is next at a counter: a step back and to the side - the
    side with more room, and never onto ``keep_clear`` (the queue's place)."""
    r = math.radians(spot.yaw)
    back = (-math.sin(r), -math.cos(r))
    side = (math.cos(r), -math.sin(r))

    def room(k: float) -> float:  # how far the nearest other person would be
        x, z = (
            spot.x + back[0] * 0.9 + side[0] * 0.7 * k,
            spot.z + back[1] * 0.9 + side[1] * 0.7 * k,
        )
        others = [(bx, bz) for bx, bz, _ in me.frame_bodies] + list(keep_clear)
        return min((math.hypot(bx - x, bz - z) for bx, bz in others), default=9.0)

    k = max((1.0, -1.0), key=room)  # the side with more room
    x, z = spot.x + back[0] * 0.9 + side[0] * 0.7 * k, spot.z + back[1] * 0.9 + side[1] * 0.7 * k
    yield Glide(
        x,
        z,
        yaw=spot.yaw + 180.0 - 30.0 * k,
        duration=0.9,
        label="stepping away from the counter",
        carry=carry,
    )


def eat_at_counter(me, life: AlleyLife) -> Story:
    """Sit on a stool at the counter and eat, then get down again."""
    stool = life.place("sushi.stool")
    yield Walk(stool.x + 0.45, stool.z, label="going to a stool", stop=0.3, carry="held")
    yield Glide(stool.x, stool.z, yaw=stool.yaw, duration=0.7, label="sitting down", carry="held")
    yield from _eat_box(me, "eating at the counter", sitting=True)
    # well clear of the stool: standing any nearer, the body is still inside its collider
    yield Glide(stool.x + 0.8, stool.z, duration=0.8, label="getting off the stool", carry="held")


def find_a_spot_to_eat(me, life: AlleyLife) -> Story:
    """Away from the counter (standing about in front of it, a customer with a box is in
    the way of the queue) to the spot by the awning pole - or next to it, if somebody is
    already eating there."""
    if not life.has("sushi.lean"):
        return
    lean = life.place("sushi.lean")
    spots = [(lean.x + 0.85 * k, lean.z - 0.3 * k) for k in range(3)]
    x, z = next((s for s in spots if not _occupied(me, *s, margin=0.4)), spots[-1])
    yield Walk(x, z, label="finding somewhere to eat", stop=0.3, carry="held")
    yield Turn(lean.yaw, label="finding somewhere to eat")


def eat_standing(me, life: AlleyLife, seconds: float = 12.0) -> Story:
    """Eat out of the box standing (``seconds`` is only a guide: every bite is eaten)."""
    yield from _eat_box(me, "eating", sitting=False)


def box_with(dish: str, left: int) -> str:
    """The item name of an open box of ``dish`` with ``left`` bites still in it."""
    if dish == "sushi":
        if left <= 0:
            return "takeaway_open"
        return "sushi_open" if left >= clips.SUSHI_PIECES else f"sushi_open{left}"
    return "noodles" if left >= clips.NOODLE_BITES else f"noodles{max(0, left)}"


def _eat_box(me, label: str, sitting: bool) -> Story:
    """Open the box (sushi, or noodles) in the left hand, chopsticks in the right, and eat
    it a bite at a time - each piece of sushi picked out of the box (which empties as it
    goes), carried up to the mouth in the chopsticks and eaten. Leaves the empty box in
    the right hand, for the bin."""
    dish = "sushi" if (me.hands["r"] or "takeaway") == "takeaway" else "noodles"
    food = clips.food_in(dish)
    n = len(food)
    me.hold("l", box_with(dish, n))
    me.hold("r", "chopsticks")
    events: dict[float, Callable[[], None]] = {}
    for k in range(n):

        def pick(left: int = n - k - 1) -> None:
            me.hold("l", box_with(dish, left))
            me.hold("r", f"chopsticks_{dish}")

        def swallow() -> None:
            me.hold("r", "chopsticks")

        events[k * clips.BITE + clips.PICK_AT] = pick
        events[k * clips.BITE + clips.EAT_AT] = swallow

    def clip(t: float) -> HumanoidPose:
        if sitting:
            seat = _up(me, STOOL_TOP)
            bob = clips.SEAT_BOB + seat - clips.STOOL_SEAT_Y
            return clips.sit(clips.eat(t, bob=bob, seed=me.seed, food=food), seat, stool=True)
        return clips.eat(t, seed=me.seed, food=food)

    yield Play(clip, n * clips.BITE, label=label, events=events)
    me.hold("l", None)
    me.hold("r", box_with(dish, 0))  # the empty box


def bin_it(me, life: AlleyLife) -> Story:
    """Over to the dumpster and throw the empty box in."""
    spot = life.place("bin")
    for _ in range(3):
        there = yield Walk(spot.x, spot.z, label="taking the empty box to the bin", carry="held")
        if there:
            break
    if math.hypot(spot.x - me.x, spot.z - me.z) > 1.2:
        return  # never got to it: the box goes home with them
    yield Turn(spot.yaw, label="at the bin")
    yield Play(
        clips.toss, 1.0, label="throwing the box away", events={0.75: lambda: me.hold("r", None)}
    )


def leave_and_return(me, life: AlleyLife, away: float) -> Story:
    """Home for a while: along the street to their own front door, in (locking it behind
    them), about the flat for ``away`` seconds or so, and out again - nobody just goes out
    of sight. (Whoever has no home to go to strolls about instead.)"""
    home = life.home_of(me)
    if home is None:
        yield Play(lambda t: clips.idle(t, me.seed), 3.0, label="standing about")
        return
    stay = max(away, MIN_TIME_AT_HOME)
    since: list[float] = []

    def over() -> bool:  # (first asked once they are in: the time is counted from then)
        if not since:
            since.append(life.time)
        return life.time - since[0] >= stay

    yield from stay_home(me, life, over)


MIN_TIME_AT_HOME = 30.0  # seconds: a trip home is worth the walk


def _bar_open(life: AlleyLife) -> bool:
    """The Neon Moth is open for a drink (``ai.nightlife``)."""
    from cyberduke.ai.nightlife import bar_open

    return life.has("bar.front", "bar.entry") and bar_open(life)


def _a_drink(me, life: AlleyLife) -> Story:
    """A drink at the Neon Moth, round the corner (``ai.nightlife``)."""
    from cyberduke.ai.nightlife import drink_at_the_bar

    yield from drink_at_the_bar(me, life)


# ---------------------------------------------------------------- the junk dealer
def junk_dealer(me, life: AlleyLife) -> Story:
    """Sells from 10:00 until midnight (``ai.schedule``); at closing time he throws a
    cover over the table, goes home round the corner, and comes back in the morning."""
    home = life.place("junk.dealer")
    counter = life.counters["junk"]
    if not _behind_counter(me, home):
        if stall_hours_open("junk", life.hour):
            yield from _open_stall(me, life, "junk")
        else:
            life.open["junk"] = False
            yield from _stall_sign(me, life, "junk", closed=True, reach=False)
            yield from stay_home(me, life, lambda: stall_hours_open("junk", life.hour))
            yield from _open_stall(me, life, "junk")
    yield Glide(home.x, home.z, yaw=home.yaw, duration=0.1, label="behind the table")
    chores = ("tidy", "beckon", "restock", "tidy", "beckon")
    k = 0
    while True:
        if not stall_hours_open("junk", life.hour) and _counter_idle(counter):
            yield from _close_stall(me, life, "junk")
            yield from stay_home(me, life, lambda: stall_hours_open("junk", life.hour))
            yield from _open_stall(me, life, "junk")
            continue
        order = counter.next_order()
        if order is None and counter.serving is not None:
            yield Wait(
                lambda: counter.next_order() is not None or counter.serving is None,
                lambda t: clips.talk(t, me.seed, listening=True),
                label="behind the table, listening",
                timeout=20.0,
            )
            continue
        if order is None:
            chore = chores[k % len(chores)]
            k += 1
            yield from _dealer_chore(me, life, chore)
            continue
        order.state = "making"
        me.serving = order.customer
        if order.dish == "parcel":
            yield from _fetch_parcel(me, life, order)
        else:
            yield from _sell(me, life, order)
        me.serving = None


def _dealer_chore(me, life: AlleyLife, chore: str) -> Story:
    counter = life.counters["junk"]
    home = life.place("junk.dealer")

    def busy() -> bool:
        return counter.next_order() is not None or counter.serving is not None

    if chore == "restock" and not busy():
        stock = life.place("junk.stock")
        yield Glide(stock.x, stock.z, yaw=stock.yaw, duration=0.7, label="going to the stock")
        grab = _there_and_back(2.0)
        yield Play(
            lambda t: clips.crouch_reach(grab(t), target=(0.0, 0.62, 0.5)),
            2.0,
            label="digging in a crate",
            events={1.0: lambda: me.hold("r", "cyberdeck")},
        )
        yield Glide(
            home.x, home.z, yaw=home.yaw, duration=0.7, label="back to the table", carry="held"
        )
        table = _body(me, home.x + 0.65 * math.sin(math.radians(home.yaw)), 0.92, home.z + 0.4)
        put = _there_and_back(1.0)
        yield Play(
            lambda t: _dipped_reach(put(t), table, dip=0.14),
            1.0,
            label="putting stock out",
            events={0.5: lambda: me.hold("r", None)},
        )
    elif chore == "beckon":
        yield Wait(busy, clips.beckon, label="calling out to passers-by", timeout=5.0)
    else:
        yield Wait(
            busy,
            lambda t: clips.browse(t, me.seed, table_y=_up(me, TABLE_TOP), reach_out=0.5),
            label="arranging the merchandise",
            timeout=6.0,
        )
    yield Wait(busy, lambda t: clips.idle(t, me.seed), label="waiting for customers", timeout=2.5)


def _sell(me, life: AlleyLife, order: Order) -> Story:
    """Haggle, take the cred, wrap the thing up and hand it over."""
    home = life.place("junk.dealer")
    yield Turn(home.yaw, label="serving")
    yield Play(lambda t: clips.talk(t, me.seed), 3.5, label="haggling")
    pick = _there_and_back(1.0)
    table = _body(me, home.x + 0.6 * math.sin(math.radians(home.yaw)), 0.92, home.z)
    yield Play(
        lambda t: _dipped_reach(pick(t), table, dip=0.14),
        1.0,
        label="picking it up",
        events={0.5: lambda: me.hold("r", "cyberdeck")},
    )
    # down on the table between the hands, wrapped, the tape round it, picked up again
    spot = _world(me, 0.0, 0.0, 0.45)
    w = (spot[0], 0.9, spot[2])  # on the table top, whatever the dealer's height
    at = _body(me, *w)
    lying: list[LooseItem] = []

    def put_down() -> None:
        me.hold("r", None)
        lying.append(life.put("cyberdeck", w[0], w[1], w[2], yaw=me.yaw))

    def wrapped() -> None:
        if lying:
            lying[0].name = "gift"

    def pick_up() -> None:
        if lying:
            life.remove(lying[0])
        me.hold("r", "gift")

    yield Play(
        lambda t: clips.wrap(t, at),
        3.2,
        label="wrapping it up",
        events={0.1: put_down, 2.2: wrapped, 3.1: pick_up},
    )
    yield from _give(me, life, order, "gift")


def _fetch_parcel(me, life: AlleyLife, order: Order) -> Story:
    home = life.place("junk.dealer")
    stock = life.place("junk.stock")
    yield Play(lambda t: clips.talk(t, me.seed), 1.5, label="checking the pickup")
    yield Glide(stock.x, stock.z, yaw=stock.yaw, duration=0.7, label="going to the stock")
    grab = _there_and_back(2.2)
    yield Play(
        lambda t: _lift_pose(grab(t)),
        2.2,
        label="lifting out a parcel",
        events={1.1: lambda: me.hold_both("parcel")},
    )
    yield Glide(
        home.x, home.z, yaw=home.yaw, duration=0.8, label="carrying the parcel over", carry="two"
    )
    yield from _give(me, life, order, "parcel")


def _lift_pose(u: float) -> HumanoidPose:
    """Squatting to a crate and lifting something out of it with both hands."""
    pose = clips.crouch_reach(u, target=(0.0, 0.62, 0.5))
    if u > 0.05:
        y = 0.62 + 0.35 * (1.0 - u)
        pose.arm_l = arm_to((clips.PARCEL_GRIP, y, 0.5), "l", bob=pose.bob)
        pose.arm_r = arm_to((-clips.PARCEL_GRIP, y, 0.5), "r", bob=pose.bob)
    return pose


def _give(me, life: AlleyLife, order: Order, item: str) -> Story:
    """Hold it out over the table until the customer takes it."""
    two = item == "parcel"
    if two:
        me.hold_both(item)
    else:
        me.hold("r", item)
    out = (0.0, 1.25, 0.62)
    order.state = "ready"
    order.item = None
    me.offering = item
    yield Wait(
        lambda: order.state == "taken",
        lambda t: (
            clips.offer(min(1.0, t / 0.8), out) if two else clips.reach(min(1.0, t / 0.8), out)
        ),
        label="holding it out",
        timeout=25.0,
    )
    me.offering = None
    me.hold("r", None)
    me.hold("l", None)
    order.state = "taken"
    yield Play(lambda t: clips.talk(t, me.seed), 1.2, label="seeing them off")


def buy_at_junk(me, life: AlleyLife, dish: str, browse: bool = True) -> Story:
    """Browse the table (pick things up, turn them over, put them down, think), haggle,
    pay and take what was bought from the dealer's hands."""
    counter = life.counters["junk"]
    spot = life.place("junk.customer")
    if not life.open.get("junk", True):
        return  # (covered up for the night)
    if me not in counter.queue and counter.serving is not me:
        counter.queue.append(me)
    if browse:
        look = life.place("junk.browse")
        yield Walk(look.x, look.z, label="wandering over to the junk table", stop=0.35)
        yield Turn(look.yaw, label="looking over the junk")
        yield Play(lambda t: _browse_low(t, me.seed, _scale(me)), 5.0, label="browsing the table")
        grab = _there_and_back(1.2)
        yield Play(
            lambda t: _dipped_reach(grab(t), (-0.05, 0.92, 0.55), dip=0.14),
            1.2,
            label="picking something up",
            events={0.6: lambda: me.hold("r", "cyberdeck")},
        )
        yield Play(lambda t: clips.hold_up(t, me.seed), 3.0, label="turning it over in the light")
        yield Play(
            lambda t: _dipped_reach(grab(t), (-0.05, 0.92, 0.55), dip=0.14),
            1.2,
            label="putting it back",
            events={0.6: lambda: me.hold("r", None)},
        )
        yield Play(lambda t: clips.think(t, me.seed), 3.0, label="thinking it over")
    yield Wait(
        lambda: counter.free_for(me) or not life.open.get("junk", True),
        lambda t: _browse_low(t, me.seed, _scale(me)),
        label="waiting their turn",
        timeout=60.0,
    )
    if not life.open.get("junk", True):
        counter.leave(me)
        return
    counter.step_up(me)
    near = approach(spot)
    yield Walk(near[0], near[1], label="going up to the dealer", stop=0.3)
    yield Glide(spot.x, spot.z, yaw=spot.yaw, duration=0.6, label="going up to the dealer")
    counter.arrive(me)
    order = Order(me, dish)
    counter.orders.append(order)
    if dish == "gift":
        yield Play(lambda t: clips.talk(t, me.seed + 3.0), 3.5, label="haggling")
        pay = _there_and_back(1.2)
        yield Play(
            lambda t: clips.reach(pay(t), (0.0, 1.15, 0.5)),
            1.2,
            label="paying",
            events={0.0: lambda: me.hold("r", "cred"), 0.8: lambda: me.hold("r", None)},
        )
    else:
        yield Play(lambda t: clips.talk(t, me.seed), 1.5, label="asking for the pickup")
    yield Wait(
        lambda: order.state == "ready",
        lambda t: clips.talk(t, me.seed, listening=True),
        label="waiting",
        timeout=40.0,
    )
    if order.state != "ready":
        counter.leave(me)
        return
    two = dish == "parcel"
    take = _there_and_back(1.2)
    out = (0.0, 1.25, 0.55)

    def got_it() -> None:
        order.state = "taken"
        if two:
            me.hold_both(dish)
        else:
            me.hold("r", dish)

    yield Play(
        lambda t: clips.offer(take(t), out) if two else clips.reach(take(t), out),
        1.2,
        label="taking it",
        events={0.6: got_it},
    )
    yield from step_aside(me, spot, carry="two" if two else "held")
    counter.leave(me)


def _browse_low(t: float, seed: float, scale: float = 1.0) -> HumanoidPose:
    pose = clips.browse(t, seed, table_y=TABLE_TOP / scale, reach_out=0.55)
    leg, _, bob = clips._squat_legs(0.1)
    pose.bob = bob
    pose.leg_l, pose.leg_r = list(leg), list(leg)
    pose.foot_l = pose.foot_r = -leg[0] - leg[2]
    pose.arm_l = arm_to((0.18, 1.0, 0.22), "l", bob=bob)
    return pose


# ---------------------------------------------------------------- the civilians
# ---------------------------------------------------------------- keeping hours
def keep_hours(me, life: AlleyLife) -> Story:
    """Between one outing and the next: if it is their time to be at home - or, for the
    office worker, at work, which he does from home (``ai.schedule``) - they go home, in
    through their own front door, and come back out of it when it is over."""
    plan = schedule(me.role)
    home = life.home_of(me)
    inside = home is not None and life.inside(home.area_id, me.x, me.z)
    if plan.out(life.hour):
        if inside:  # (a saved game, or the end of a nap)
            yield from _let_self_out(me, life, home)
        return
    yield from stay_home(me, life, lambda: plan.out(life.hour))


def office_worker(me, life: AlleyLife) -> Story:
    """Works from home 07:30 - 17:00 (at the computer in his front room); after work down
    the alley for sushi (the evening rush), eaten at the counter; the box in the bin; home
    for a while and back later, sometimes for noodles. Home for the night at 23:00 -
    number 32, round the corner."""
    visit = 0
    while True:
        yield from keep_hours(me, life)
        dish = "nigiri" if visit % 3 != 2 else "noodles"
        yield from buy_food(me, life, dish)
        if me.hands["r"]:
            if visit % 2 == 0:
                yield from eat_at_counter(me, life)
            else:
                yield from find_a_spot_to_eat(me, life)
                yield from eat_standing(me, life, 10.0)
            yield from bin_it(me, life)
        if not me.hands["r"] and not life.open.get("sushi", True):
            yield from vending_run(me, life)  # (the stall is shut: a can, then)
        if visit % 3 == 1 and _bar_open(life):
            yield from _a_drink(me, life)  # (a pint at the Neon Moth, round the corner)
        else:
            yield from leave_and_return(me, life, 20.0 + me.rng.uniform(0.0, 20.0))
        visit += 1


def tinkerer(me, life: AlleyLife) -> Story:
    """Comes for the junk: browses, picks things up and turns them in the light, thinks,
    haggles, buys it, then strolls off with it - via a box of noodles - and home
    with it for a while. A late riser (out from 11:00), home at half past eleven at night
    (number 33)."""
    while True:
        yield from keep_hours(me, life)
        yield from buy_at_junk(me, life, "gift")
        if me.hands["r"] == "gift":
            away = life.place("junk.browse")
            yield Walk(
                away.x - 1.2, away.z - 3.0, label="strolling off with the purchase", carry="held"
            )
        me.hold("r", None)  # into a pocket
        yield from buy_food(me, life, "noodles")
        if me.hands["r"]:
            yield from find_a_spot_to_eat(me, life)
            yield from eat_standing(me, life, 12.0)
            yield from bin_it(me, life)
        if not me.hands["r"] and _bar_open(life) and me.rng.random() < 0.35:
            yield from _a_drink(me, life)
        yield from leave_and_return(me, life, 25.0 + me.rng.uniform(0.0, 15.0))


DOOR16 = "door16"
FLAT16 = "flat16"  # number 16's flat (``LevelData.private_areas``)


def courier(me, life: AlleyLife) -> Story:
    """Collects a parcel from the dealer for number 16, carries it there and knocks. If the
    resident is in they open up and he hands it over at the door; if not, he leaves it on
    the step. Then home for a while (number 34) before the next one."""
    stoop = life.place("door16.stoop")
    front = life.place("door16.front")
    drop = life.place("door16.parcel")
    while True:
        yield from keep_hours(me, life)
        if not life.open.get("junk", True):  # (nothing to deliver till the dealer opens)
            yield from leave_and_return(me, life, 25.0)
            continue
        yield from buy_at_junk(me, life, "parcel", browse=False)
        if me.hands["r"] != "parcel":
            yield from leave_and_return(me, life, 15.0)
            continue
        yield Walk(front.x, front.z, label="delivering a parcel to number 16", carry="two")
        # up to the door with the parcel under one arm, a hand free to knock: held out in
        # front in both hands it went straight through the door
        me.hold_both(None)
        me.hold("l", "parcel")
        yield Glide(
            stoop.x,
            stoop.z,
            yaw=stoop.yaw,
            y=stoop.y,
            duration=0.6,
            label="up the step",
            carry="tucked",
        )
        life.flags["knock16"] = me
        door = door16_distance(me, life)
        yield Play(lambda t, door=door: _knock_tucked(t, door), 4.4, label="knocking")
        answered = yield Wait(
            lambda: life.flags.get("answer16") is not None,
            lambda t: _listening_at_door(t, me.seed),
            label="waiting for an answer",
            timeout=2.5,
        )
        if answered:
            yield from _hand_over_at_door(me, life, front)
        else:
            life.flags.pop("knock16", None)
            yield from _leave_parcel(me, life, drop)
            # off the step to one side - whoever is waiting to go up is right in front of it
            yield Glide(front.x + 0.05, front.z + 0.95, y=0.0, duration=0.8, label="down the step")
        yield from leave_and_return(me, life, 35.0 + me.rng.uniform(0.0, 20.0))


def door16_distance(me, life: AlleyLife) -> float:
    """How far ahead of ``me`` the (shut) face of the door of number 16 is."""
    door = life.door(DOOR16)
    face = door.obstacle.max_x if door is not None else -4.4
    return abs(_body(me, face, 0.0, me.z)[2])


def _knock_tucked(t: float, door: float) -> HumanoidPose:
    pose = clips.knock(t, door)
    pose.arm_l = clips.carrying(0.0, "tucked", moving=False).arm_l
    return pose


def _listening_at_door(t: float, seed: float) -> HumanoidPose:
    pose = clips.carrying(t, "tucked", moving=False)
    pose.head_roll = 7.0  # an ear to the door
    pose.head_pitch = 4.0 + 2.0 * math.sin(t * 0.8 + seed)
    return pose


def _leave_parcel(me, life: AlleyLife, drop: Place) -> Story:
    """Nobody in: a step back from the door to turn along the step (turning right in
    front of it, a shoulder went through it), the parcel into both hands, bend down and
    set it in the corner of the step - let go of exactly where it then lies."""
    beside = life.place("door16.beside")
    yield Glide(
        beside.x, beside.z, yaw=beside.yaw, y=beside.y, duration=0.8,
        label="leaving the parcel on the step", carry="tucked",
    )  # fmt: skip
    me.hold_both("parcel")  # from under the arm into both hands
    yield Play(
        lambda t: clips.offer(min(1.0, t / 0.4), clips.CARRY_TWO),
        0.4,
        label="leaving the parcel on the step",
    )
    # hands either side of it, half way up (render.items.TWO_HANDED_GRIP)
    at = _body(me, drop.x, drop.y + clips.TWO_HANDED_GRIP["parcel"], drop.z)
    put = _there_and_back(2.0)

    def let_go() -> None:
        me.hold_both(None)
        life.flags["parcel"] = life.put("parcel", drop.x, drop.y, drop.z, yaw=me.yaw)

    yield Play(
        lambda t: clips.stoop_reach(
            put(t), at, both=True, carry=clips.CARRY_TWO if t < 0.9 else None
        ),
        2.0,
        label="leaving the parcel on the step",
        events={0.9: let_go},
    )


def _hand_over_at_door(me, life: AlleyLife, front: Place) -> Story:
    """Somebody is coming to the door: step back down off the step to let it open, hold
    the parcel out, let them take it, say goodnight."""
    yield Glide(
        front.x,
        front.z,
        yaw=front.yaw,
        y=0.0,
        duration=0.6,
        label="stepping back down",
        carry="tucked",
    )
    me.hold_both("parcel")
    life.flags["waiting16"] = me
    yield Wait(
        lambda: life.flags.get("at_door16") is not None,
        lambda t: clips.carrying(t, "two", moving=False),
        label="waiting at the door",
        timeout=8.0,
    )
    me.offering = "parcel"
    out = (0.0, 1.22, 0.42)
    yield Wait(
        lambda: bool(life.flags.get("taken16")),
        lambda t: clips.offer(min(1.0, t / 0.8), out),
        label="handing over the parcel",
        timeout=8.0,
    )
    me.offering = None
    if me.hands["r"] == "parcel":  # they never took it: on the step it goes
        drop = life.place("door16.parcel")
        me.hold_both(None)
        life.flags["parcel"] = life.put("parcel", drop.x, drop.y, drop.z, yaw=drop.yaw)
    yield Play(lambda t: clips.talk(t, me.seed), 1.6, label="saying goodnight")
    for flag in ("knock16", "waiting16"):
        life.flags.pop(flag, None)


def resident(me, life: AlleyLife) -> Story:
    """Dell lives at number 16. Home, he is about the flat - cooking on the hotplate,
    slumped on the sofa flicking through the channels, at his computer, washing up, at the
    mirror; every minute or so he goes out - for sushi (brought home to eat) while the
    stall is open, to the vending machine in the alley when it is not - and locks up
    behind him. He answers the door to the courier and takes the parcel in - or finds it
    on the step when he next comes out. In the small hours he sleeps (``ai.schedule``).

    The vending machine on the blink (shorted by the netrunner) is his business too - but
    only if he is out and sees it: walking anywhere out of doors he spots it and goes over
    to kick it back to life (``ASIDES``) before he gets on with his errand - the
    netrunner's chance to keep him out of his flat longer. From inside the flat he knows
    nothing about it. Somebody in his flat is his business (the simulation has him report
    them: ``Civilian.watch_area``)."""
    home = _resident_home(life)
    stoop = life.place("door16.stoop")
    front = life.place("door16.front")
    plan = schedule("resident")
    if not life.inside(FLAT16, me.x, me.z):  # (a fresh start - or a saved game - outside)
        me.hold("r", "bag")
        yield Walk(front.x, front.z, label="coming home with the shopping", carry="one")
        yield from _let_self_in(me, life, carry="one")
        yield from _put_down_at_home(me, life, "apt.kitchen", "putting the shopping away")

    while True:
        if plan.asleep(life.hour):
            yield from sleep_in_bed(me, life, home, lambda: not plan.asleep(life.hour))
            continue
        # (a parcel left on the step while he was out: he is back out for it in a moment)
        waiting = isinstance(life.flags.get("parcel"), LooseItem)
        stay = 6.0 if waiting else 40.0 + me.rng.uniform(0.0, 35.0)
        yield from _at_home(
            me, life, life.time + stay, home,
            interrupt=lambda: "knock16" in life.flags or plan.asleep(life.hour),
        )  # fmt: skip
        if "knock16" in life.flags:
            yield from _answer_door(me, life, stoop)
            continue
        if plan.asleep(life.hour):
            continue
        yield from _let_self_out(me, life)
        parcel = life.flags.get("parcel")
        if isinstance(parcel, LooseItem) and parcel in life.items:
            yield from _take_in_parcel(me, life, parcel, stoop)
            continue
        yield Glide(front.x, front.z, y=0.0, duration=0.6, label="down the step")
        if life.open.get("sushi", True):
            yield from buy_food(me, life, "nigiri")
        if me.hands["r"]:
            yield Walk(front.x, front.z, label="taking dinner home", carry="held")
            yield from _let_self_in(me, life, carry="held")
            yield from _walk_indoors(me, life, "apt.kitchen", carry="held")
            yield Turn(life.place("apt.kitchen").yaw, label="eating in the kitchen")
            yield from _eat_box(me, "eating in the kitchen", sitting=False)
            me.hold("r", None)  # (the empty box in the bag by the door)
            continue
        # the stall is shut: a can from the machine, then
        yield from vending_run(me, life, take_home=True)
        yield Walk(
            front.x, front.z, label="taking a can home", carry="held" if me.hands["r"] else None
        )
        yield from _let_self_in(me, life, carry="held" if me.hands["r"] else None)
        if me.hands["r"]:
            yield from _walk_indoors(me, life, "apt.living")
            yield Play(lambda t: _drink(t, me.seed), 6.0, label="drinking a can")
        me.hold("r", None)


SEES_THE_MACHINE_FROM = 30.0  # metres: anywhere in the alley, or at its mouth


def dell_sees_the_machine(me, life: AlleyLife) -> Story | None:
    """Dell's aside (``ASIDES``): out of doors and on his way somewhere, he sees the
    vending machine is on the blink - and goes to see to it. Never from inside his flat
    (he would not know), never while he is queueing at a counter (he keeps his place)."""
    if life.broken_machine() is None or not life.has("vending"):
        return None
    if me.hidden or me.asleep or life.inside(_resident_home(life).area_id, me.x, me.z):
        return None
    if any(me in c.queue or me in (c.serving, c.coming) for c in life.counters.values()):
        return None
    spot = life.place("vending")
    if math.hypot(spot.x - me.x, spot.z - me.z) > SEES_THE_MACHINE_FROM:
        return None
    return _see_to_the_machine(me, life)


def _carrying(me) -> str | None:
    """How whatever is in their hands is carried (walking with it)."""
    held = me.hands["r"]
    if held is None:
        return None
    if held == "bag":
        return "one"
    return "two" if me.hands["l"] == held else "held"


def _face_ahead(me, box) -> float:
    """How far ahead of ``me`` (along the way they face) the box's near face is (the
    vending machine's front); 0.5 if the way ahead misses it."""
    r = math.radians(me.yaw)
    dx, dz = math.sin(r), math.cos(r)
    near, far = 0.0, 1e9
    for o, d, lo, hi in ((me.x, dx, box.min_x, box.max_x), (me.z, dz, box.min_z, box.max_z)):
        if abs(d) < 1e-9:
            if not lo <= o <= hi:
                return 0.5
            continue
        a, b = (lo - o) / d, (hi - o) / d
        near, far = max(near, min(a, b)), min(far, max(a, b))
    return near if near <= far else 0.5


def _see_to_the_machine(me, life: AlleyLife, walk: bool = True) -> Story:
    """The vending machine down the alley is on the blink - the one that has sold him a
    can every night for years: over he goes, looks it over, tries the buttons, steps in
    and kicks it until it shudders back to life (``VendingMachine.repair``), and stares it
    down a while. Then back to whatever he was doing (carrying whatever he was carrying)."""
    spot = life.place("vending")
    carry = _carrying(me)
    if walk:
        there = yield Walk(
            spot.x, spot.z, label="going to see what's up with the vending machine", stop=0.3,
            carry=carry,
        )  # fmt: skip
        if not there:
            return
    yield Turn(spot.yaw, label="looking at the vending machine")
    yield Play(
        _wearing("frown", lambda t: clips.think(t, me.seed)), 3.5,
        label="looking at the vending machine",
    )  # fmt: skip
    if me.hands["r"] is None:
        press = _there_and_back(1.2)
        yield Play(
            lambda t: clips.reach(press(t), (0.12, 1.3, 0.5)), 1.2, label="trying the buttons"
        )
        yield Play(lambda t: clips.idle(t, me.seed), 1.0, label="trying the buttons")
    machine = life.broken_machine()
    if machine is not None:
        # a step in (the toe just reaching its front on the strike) and the boot in
        ahead = _face_ahead(me, machine.get_bounds())
        step = max(0.0, ahead - clips.KICK_REACH * _scale(me) + 0.02)
        kx, kz = _in_front(Place("kick", me.x, me.z, yaw=spot.yaw), step)
        yield Glide(
            kx, kz, yaw=spot.yaw, duration=0.5, label="kicking the vending machine",
            carry=carry, walk_first=False,
        )  # fmt: skip
        kicks = 3
        yield Play(
            lambda t: clips.kick(t, me.seed),
            kicks * clips.KICK_PERIOD,
            label="kicking the vending machine",
            events={(kicks - 1 + clips.KICK_STRIKE) * clips.KICK_PERIOD: machine.repair},
        )
        yield Glide(
            spot.x, spot.z, yaw=spot.yaw, duration=0.5, label="staring at the vending machine",
            carry=carry, walk_first=False,
        )  # fmt: skip
    yield Play(
        _wearing("frown", lambda t: clips.think(t, me.seed + 1.0)), 3.0,
        label="staring at the vending machine",
    )  # fmt: skip


def _wearing(expression: str, clip: Clip) -> Clip:
    """``clip`` with the face set to ``expression`` (a scowl at a dead machine)."""

    def play(t: float) -> HumanoidPose:
        pose = clip(t)
        pose.expression = expression
        return pose

    return play


def _drink(t: float, seed: float) -> HumanoidPose:
    """Sips from a can (``animation.street.drink``)."""
    return clips.drink(t, seed)


def vending_run(me, life: AlleyLife, take_home: bool = False, walk: bool = True) -> Story:
    """Over to the vending machine in the alley: a button pressed, the can out of the
    slot at the bottom - drunk there and the can binned, or (``take_home``) carried off."""
    if not life.has("vending", "vending.slot"):
        return
    spot = life.place("vending")
    if walk:
        if _occupied(me, spot.x, spot.z, margin=0.3):
            yield Wait(
                lambda: not _occupied(me, spot.x, spot.z, margin=0.3),
                lambda t: clips.idle(t, me.seed),
                label="waiting for the vending machine",
                timeout=15.0,
            )
        there = yield Walk(spot.x, spot.z, label="going to the vending machine", stop=0.25)
        if not there:
            return
    yield Turn(spot.yaw, label="at the vending machine")
    if life.broken_machine() is not None:
        if me.role == "resident":  # his machine: he sees to it, then gets his can
            yield from _see_to_the_machine(me, life, walk=False)
        else:  # out of order: a thump for luck, and off without a can
            press = _there_and_back(1.2)
            yield Play(
                lambda t: clips.reach(press(t), (0.12, 1.3, 0.5)), 1.2, label="choosing a drink"
            )
            yield Play(
                _wearing("frown", lambda t: clips.think(t, me.seed)), 1.6,
                label="the vending machine is out of order",
            )  # fmt: skip
            yield Play(bang_machine_animation, 1.8, label="thumping the dead vending machine")
            return
    press = _there_and_back(1.2)
    yield Play(lambda t: clips.reach(press(t), (0.12, 1.3, 0.5)), 1.2, label="choosing a drink")
    yield Play(lambda t: clips.idle(t, me.seed), 1.0, label="waiting for the can")
    grab = _there_and_back(1.6)
    yield Play(
        lambda t: clips.crouch_reach(grab(t), target=(0.05, 0.42, 0.45)),
        1.6,
        label="taking the can out of the slot",
        events={0.8: lambda: me.hold("r", "soda")},
    )
    if take_home:
        return
    yield Play(lambda t: _drink(t, me.seed), 6.0, label="drinking a can")
    yield from bin_it(me, life)


HOME_CHORES: tuple[tuple[str, str, float], ...] = (
    ("apt.kitchen", "cooking on the hotplate", 12.0),
    ("apt.sofa", "watching TV", 18.0),
    ("apt.desk", "at his computer", 12.0),
    ("apt.sink", "washing up", 9.0),
    ("apt.rail", "looking for a clean shirt", 7.0),
    ("apt.bath", "at the bathroom mirror", 7.0),
)
# In any other flat (``levels.flats``): its own places, by name
FLAT_CHORES: tuple[tuple[str, str, float], ...] = (
    ("kitchen", "cooking at the hob", 12.0),
    ("sofa", "watching TV", 22.0),
    ("sink", "washing up", 8.0),
    ("bath", "at the bathroom mirror", 6.0),
    ("wardrobe", "looking for something to wear", 6.0),
    ("desk", "at the computer", 14.0),
)
WORK = "working at the computer"  # (the chore of the hours somebody works from home)
FLAT_DESK_TOP = 0.76  # a flat's computer desk (``levels.flats``)
CHAIR_SEAT = 0.48  # its chair


def _chore_clip(me, chore: str) -> Clip:
    if chore.startswith("cooking"):
        return cook_animation
    if chore.startswith(("watching", "flicking")):
        seat = _up(me, SOFA_SEAT)
        # (every few seconds the remote comes up: see ``WatchTv`` for the real clicks)
        return lambda t: clips.watch_tv(
            t, seat, click=max(0.0, math.sin(t * 1.3)) ** 6, seed=me.seed
        )
    if chore.startswith(("at the computer", WORK)):
        desk, seat = _up(me, FLAT_DESK_TOP), _up(me, CHAIR_SEAT)
        return lambda t: clips.typing_seated(t, desk, seat)
    if chore.startswith("at his computer"):
        desk = _up(me, DESK_TOP)
        return lambda t: clips.typing(t, desk)
    if chore.startswith("washing"):
        top = _up(me, KITCHEN_TOP)
        return lambda t: clips.wipe(t, top)
    if chore.startswith("looking"):
        return lambda t: clips.browse(t, me.seed, table_y=1.3, reach_out=0.45)
    return lambda t: clips.think(t, me.seed)


def _resident_home(life: AlleyLife) -> Home:
    return life.homes.get("resident") or Home(DOOR16, FLAT16, "apt", "door16.front", "door16.stoop")


def _chores(home: Home) -> tuple[tuple[str, str, float], ...]:
    if home.prefix == "apt":
        return HOME_CHORES
    return tuple((home.place(p), label, t) for p, label, t in FLAT_CHORES)


def _at_home(
    me,
    life: AlleyLife,
    until: float | Callable[[], bool],
    home: Home | None = None,
    interrupt: Callable[[], bool] | None = None,
) -> Story:
    """Chores about the flat until ``until`` (life's clock - or a condition) or until
    ``interrupt()`` (for Dell: a knock at the door, by default): the sofa in front of the
    television (``WatchTv``: on it goes, and they flick through the channels), the hob,
    the sink, the mirror, the wardrobe, the computer (sitting down at the desk, where a
    flat has one) - and in the hours somebody works from home, mostly the computer."""
    home = home or _resident_home(life)
    if interrupt is None:
        interrupt = lambda: "knock16" in life.flags  # noqa: E731
    done = until if callable(until) else (lambda: life.time >= until)
    last = ""
    sofa, desk = home.place("sofa"), home.place("desk")
    by_desk = home.place("by_desk")
    seated_desk = desk in life.places and by_desk in life.places
    plan = schedule(me.role)
    while not done() and not interrupt():
        choices = [c for c in _chores(home) if c[0] != last and c[0] in life.places]
        if seated_desk and plan.working(life.hour) and me.rng.random() < 0.75:
            place, chore, seconds = desk, WORK, 40.0
        else:
            place, chore, seconds = choices[int(me.rng.random() * len(choices))]
        last = place
        timeout = seconds * (0.8 + 0.4 * me.rng.random())
        if place == sofa:  # onto the sofa from in front of it
            yield from _walk_indoors(me, life, home.place("by_sofa"))
            seat = life.place(sofa)
            fx, fz = _in_front(seat, 0.7)
            yield Glide(
                fx, fz, yaw=seat.yaw, duration=0.7, label="going to the sofa", walk_first=False
            )
            yield Glide(
                seat.x,
                seat.z,
                yaw=seat.yaw,
                duration=0.8,
                label="slumping on the sofa",
                walk_first=False,
            )
            remote = me.hands["r"] is None
            if remote:
                me.hold("r", "remote")
            yield WatchTv(
                life.tv(home.area_id), lambda: life.time, _up(me, SOFA_SEAT),
                lambda: interrupt() or done(), timeout, remote=remote,
            )  # fmt: skip
            if remote:
                me.hold("r", None)  # (down on the cushion)
            fx, fz = _in_front(seat, 0.75)
            yield Glide(fx, fz, duration=0.7, label="getting up off the sofa", walk_first=False)
            continue
        if place == desk and seated_desk:  # the chair pulled out, and sat on
            yield from _walk_indoors(me, life, by_desk)
            chair = life.place(desk)
            yield Turn(chair.yaw, label="sitting down at the desk")
            yield Glide(
                chair.x, chair.z, yaw=chair.yaw, duration=0.7, label="sitting down at the desk",
                walk_first=False,
            )  # fmt: skip
            yield Wait(
                lambda: interrupt() or done(), _chore_clip(me, chore), label=chore,
                timeout=timeout,
            )  # fmt: skip
            back = life.place(by_desk)
            yield Glide(
                back.x, back.z, duration=0.7, label="getting up from the desk", walk_first=False
            )
            continue
        yield from _walk_indoors(me, life, place)
        yield Turn(life.place(place).yaw, label=chore)
        yield Wait(
            lambda: interrupt() or done(),
            _chore_clip(me, chore),
            label=chore,
            timeout=timeout,
        )


class WatchTv(Act):
    """On the sofa in front of the television (``TvSet``): it goes on with a click of the
    remote, and they flick through the channels - the snow of the dead ones between the
    stations - until something holds them (the news, a cartoon, the music channel); a
    while later they may flick on again. Until ``until()`` (result True) or ``timeout``
    (result False). Without the ``remote`` (their hands full) it is on, and stays put."""

    SURF = (0.8, 1.5)  # seconds from one click to the next, flicking through
    HOLD = (7.0, 16.0)  # seconds a good channel holds them before they may flick on
    SETTLE = 0.7  # the chance that a watchable channel holds them at all
    RAISE, LOWER = 0.35, 0.45  # seconds the remote takes to come up, and to go down again
    WATCHING = {
        "news": "watching the news",
        "cartoon": "watching cartoons",
        "music": "watching the music channel",
    }

    def __init__(
        self,
        tv: TvSet,
        now: Callable[[], float],
        seat_y: float,
        until: Callable[[], bool],
        timeout: float,
        remote: bool = True,
    ) -> None:
        self.tv, self.now, self.seat_y = tv, now, seat_y
        self.until, self.timeout, self.remote = until, timeout, remote
        self.time = 0.0
        self.clicked = -9.0
        self.next_click = 0.5
        self.label = "switching the TV on"

    def begin(self, npc) -> None:
        tv = self.tv
        if tv.on and tv.showing in WATCHABLE:  # (left on something good)
            self.next_click = npc.rng.uniform(*self.HOLD)
            self.label = self.WATCHING[tv.showing]
        elif tv.on:
            self.label = "flicking through the channels"
        if not self.remote:
            tv.switch(True, self.now())
            self.next_click = math.inf
            self.label = self.WATCHING.get(tv.showing, "watching TV")

    def _click(self, npc) -> None:
        tv = self.tv
        if tv.on:
            tv.next_channel(self.now())
        else:
            tv.switch(True, self.now())
        self.clicked = self.time
        if tv.showing in WATCHABLE and npc.rng.random() < self.SETTLE:
            self.next_click = self.time + npc.rng.uniform(*self.HOLD)
            self.label = self.WATCHING[tv.showing]
        else:
            self.next_click = self.time + npc.rng.uniform(*self.SURF)
            self.label = "flicking through the channels"

    def arm(self) -> float:
        """How far the remote is up (0 resting on the thigh .. 1 pointed at the set)."""
        before = self.next_click - self.time
        after = self.time - self.clicked
        up = 1.0 - before / self.RAISE if 0.0 <= before < self.RAISE else 0.0
        if 0.0 <= after < self.LOWER:
            up = max(up, 1.0 - after / self.LOWER)
        return up

    def step(self, npc, dt: float) -> bool:
        self.time += dt
        if self.until():
            self.result = True
            return True
        if self.time >= self.timeout:
            self.result = False
            return True
        if self.time >= self.next_click:
            self._click(npc)
        npc.pose = clips.watch_tv(self.time, self.seat_y, click=self.arm(), seed=npc.seed)
        return False


def _in_front(place: Place, by: float) -> tuple[float, float]:
    """A point ``by`` metres in front of ``place`` (the way it faces)."""
    r = math.radians(place.yaw)
    return place.x + math.sin(r) * by, place.z + math.cos(r) * by


def _walk_indoors(
    me,
    life: AlleyLife,
    goal: str,
    carry: str | None = None,
    label: str = "pottering about the flat",
) -> Story:
    """From wherever they are to the place ``goal`` along their flat's routes
    (``life.route``): straight walks, round the furniture, through the doorways."""
    prefix = goal.split(".")[0] + "."
    start = life.nearest_on_routes(me.x, me.z, prefix)
    if start is None:
        return
    path = (
        [start] if math.hypot(life.place(start).x - me.x, life.place(start).z - me.z) > 0.2 else []
    )
    path += life.route(start, goal)
    for name in path:
        p = life.place(name)
        dist = math.hypot(p.x - me.x, p.z - me.z)
        if dist < 0.05:
            continue
        heading = yaw_towards(me.x, me.z, p.x, p.z)
        yield Glide(
            p.x, p.z, yaw=heading, y=0.0, duration=max(0.4, dist / 1.05),
            label=label, carry=carry, walk_first=False,
        )  # fmt: skip


def _step_clear(me, stoop: Place) -> bool:
    return not any(math.hypot(bx - stoop.x, bz - stoop.z) < 0.9 for bx, bz, _ in me.frame_bodies)


def _wait_for_door(life: AlleyLife, label: str, clip: Clip, door_id: str = "door16") -> Act:
    door = life.door(door_id)
    return Wait(lambda: door is None or door.passable, clip, label=label, timeout=3.0)


def door_distance(me, life: AlleyLife, home: Home) -> float:
    """How far ahead of ``me`` the (shut) outer face of ``home``'s front door is."""
    door = life.door(home.door_id)
    if door is None:
        return 0.55
    o, side = door.obstacle, door.inside
    if side in ("-x", "+x"):
        face = (o.max_x if side == "-x" else o.min_x, me.z)
    else:
        face = (me.x, o.max_z if side == "-z" else o.min_z)
    return abs(_body(me, face[0], 0.0, face[1])[2])


def _let_self_in(me, life: AlleyLife, carry: str | None, home: Home | None = None) -> Story:
    """Up to the door (up the step, at number 16), key in the lock, the door swings in,
    in, and locked behind them. A parcel carried in both hands goes under the left arm
    first (held out in front it would go through the shut door, and the right hand is
    wanted for the key)."""
    home = home or _resident_home(life)
    stoop = life.place(home.stoop)
    if carry == "two":
        item = me.hands["r"]
        me.hold_both(None)
        me.hold("l", item)
        carry = "tucked"
    if math.hypot(me.x - stoop.x, me.z - stoop.z) > 0.1:
        yield Wait(
            lambda: _step_clear(me, stoop),
            lambda t: clips.carrying(t, carry, moving=False) if carry else clips.idle(t, me.seed),
            label="waiting for the step to clear",
            timeout=20.0,
        )
        up = "up the step" if stoop.y > 0.0 else "to the door"
        yield Glide(
            stoop.x, stoop.z, yaw=stoop.yaw, y=stoop.y, duration=0.6, label=up, carry=carry,
        )  # fmt: skip
    yield Turn(stoop.yaw, label="at the door")
    side = "r" if carry == "tucked" else "l"
    distance = door_distance(me, life, home)

    def unlocking(t: float, side: str = side) -> HumanoidPose:
        pose = clips.open_door(t, side, distance)
        if carry == "tucked":
            pose.arm_l = clips.carrying(0.0, "tucked", moving=False).arm_l
        return pose

    yield Play(unlocking, 1.2, label="unlocking the door")
    door = life.door(home.door_id)
    if door is not None:
        door.unlock()
    yield _wait_for_door(life, "pushing the door open", unlocking, home.door_id)
    entry = life.place(home.place("entry"))
    yield Glide(
        entry.x,
        entry.z,
        yaw=entry.yaw,
        y=0.0,
        duration=1.1,
        label="going in",
        carry=carry,
        walk_first=False,
    )
    if door is not None:
        door.lock()  # (the latch keeps it open while they are in the doorway)


def _let_self_out(me, life: AlleyLife, home: Home | None = None) -> Story:
    """To the front door, out (the latch opens it from the inside), the door shut and
    locked behind them."""
    home = home or _resident_home(life)
    stoop = life.place(home.stoop)
    out = (stoop.yaw + 180.0) % 360.0  # (the stoop faces the door; out is the other way)
    life.tv(home.area_id).switch(False, life.time)  # (off on the way out)
    yield from _walk_indoors(me, life, home.place("entry"))
    yield Turn(out, label="going out")
    yield _wait_for_door(life, "opening the door", lambda t: clips.idle(t, me.seed), home.door_id)
    # out onto the step - far enough out to turn round without an elbow in the door
    ox, oz = _in_front(Place("out", stoop.x, stoop.z, yaw=out), 0.2)
    yield Glide(ox, oz, yaw=out, y=stoop.y, duration=1.1, label="going out", walk_first=False)
    yield Turn(stoop.yaw, label="locking up")
    yield Play(
        lambda t: clips.open_door(t, "l", door_distance(me, life, home)), 0.9, label="locking up"
    )
    door = life.door(home.door_id)
    if door is not None:
        door.lock()
    yield Turn(out, label="locking up")


def _put_down_at_home(me, life: AlleyLife, place: str, label: str) -> Story:
    """Whatever is in their hands goes down at ``place`` (the shopping on the counter)."""
    yield from _walk_indoors(me, life, place, carry="one" if me.hands["r"] == "bag" else None)
    yield Turn(life.place(place).yaw, label=label)
    put = _there_and_back(1.0)
    yield Play(
        lambda t: clips.reach(put(t), (-0.1, 0.95, 0.5)),
        1.0,
        label=label,
        events={0.5: lambda: me.hold_both(None)},
    )


class Sleep(Act):
    """In bed, lying on their back (``BodyState``: the renderer lays the body on the
    bed), asleep until ``until()``. Asleep, they notice nothing."""

    label = "asleep"

    def __init__(self, bed: Place, until: Callable[[], bool]) -> None:
        self.bed, self.until = bed, until
        self.side = 0

    def begin(self, npc) -> None:
        self.side = (-1, 0, 1)[int(npc.rng.random() * 3)]
        # (the bed place is where the top of the head goes, on the pillow: a short sleeper's
        # feet end further up the bed, nobody's head goes through the headboard)
        npc.body = lying_on(
            self.bed.x, self.bed.y, self.bed.z, self.bed.yaw, side=self.side,
            scale=_scale(npc), crown=True,
        )  # fmt: skip
        npc.asleep = True
        npc.fall_angle = 90.0  # (lying down: not in anybody's way)
        npc.hold_both(None)

    def step(self, npc, dt: float) -> bool:
        npc.pose = npc.body.pose if npc.body is not None else clips.idle(0.0, npc.seed)
        if self.until():
            npc.body = None
            npc.asleep = False
            npc.fall_angle = 0.0
            return True
        return False


def sleep_in_bed(me, life: AlleyLife, home: Home, until: Callable[[], bool]) -> Story:
    """To bed: beside it, into it, asleep until ``until()``; up again beside it."""
    bedside = home.place("bedside")
    bed = home.place("bed")
    if bed not in life.places or bedside not in life.places:
        yield from _at_home(me, life, until, home, interrupt=lambda: False)
        return
    life.tv(home.area_id).switch(False, life.time)  # (off for the night)
    yield from _walk_indoors(me, life, bedside)
    yield Turn(face_towards(me, life.place(bed).x, life.place(bed).z), label="going to bed")
    yield Sleep(life.place(bed), until)
    side = life.place(bedside)
    me.x, me.z, me.y = side.x, side.z, 0.0
    yield Play(lambda t: clips.think(t, me.seed), 1.5, label="waking up")


def go_home(me, life: AlleyLife, home: Home, carry: str | None = None) -> Story:
    """Along the street to their own front door, and in."""
    front = life.place(home.front)
    for _ in range(3):
        there = yield Walk(front.x, front.z, label="going home", carry=carry)
        if there:
            break
    yield from _let_self_in(me, life, carry, home)


def stay_home(me, life: AlleyLife, until: Callable[[], bool], interrupt=None) -> Story:
    """Home (going there first if they are out) until ``until()``: pottering about the
    flat, and in bed through the hours they sleep (``ai.schedule``). Out of the door
    again at the end (unless ``interrupt()`` cut it short: they stay in)."""
    home = life.home_of(me)
    if home is None:
        yield Wait(until, lambda t: clips.idle(t, me.seed), label="standing about", timeout=60.0)
        return
    if not life.inside(home.area_id, me.x, me.z):
        yield from go_home(me, life, home, carry=_carrying(me))
        if me.hands["r"] is not None or me.hands["l"] is not None:
            # (whatever they came home with goes down on the kitchen counter)
            yield from _put_down_at_home(me, life, home.place("kitchen"), "putting things down")
    plan = schedule(me.role)
    stop = interrupt or (lambda: False)
    while not until() and not stop():
        if plan.asleep(life.hour):
            yield from sleep_in_bed(
                me, life, home, lambda: not plan.asleep(life.hour) or until() or stop()
            )
        else:
            yield from _at_home(
                me, life, lambda: until() or plan.asleep(life.hour), home, interrupt=stop
            )
    if not stop():
        yield from _let_self_out(me, life, home)


def _take_in_parcel(me, life: AlleyLife, parcel: LooseItem, stoop: Place) -> Story:
    """A parcel on the step: along the step to it, bend down, both hands either side of
    it, up with it, read the label, and take it in."""
    beside = life.place("door16.beside")
    yield Glide(
        beside.x, beside.z, yaw=beside.yaw, y=beside.y, duration=0.8,
        label="finding a parcel on the step",
    )  # fmt: skip
    at = _body(me, parcel.x, parcel.y + clips.TWO_HANDED_GRIP["parcel"], parcel.z)
    pick = _there_and_back(2.0)

    def lift(parcel: LooseItem = parcel) -> None:
        life.remove(parcel)
        life.flags.pop("parcel", None)
        me.hold_both("parcel")

    yield Play(
        lambda t, pick=pick, at=at: clips.stoop_reach(
            pick(t), at, both=True, carry=clips.CARRY_TWO if t > 0.9 else None
        ),
        2.0,
        label="finding a parcel on the step",
        events={0.9: lift},
    )
    yield Play(lambda t: clips.hold_up(t, me.seed), 2.4, label="reading the label")
    me.hold_both(None)
    me.hold("l", "parcel")  # under the arm, to get the key out
    yield Glide(
        stoop.x, stoop.z, yaw=stoop.yaw, y=stoop.y, duration=0.8,
        label="turning back to the door", carry="tucked",
    )  # fmt: skip
    yield from _let_self_in(me, life, carry="tucked")
    yield from _put_down_at_home(me, life, "apt.desk", "putting the parcel down")


def _answer_door(me, life: AlleyLife, stoop: Place) -> Story:
    """Somebody knocked: to the door, open up, out onto the step, take the parcel, thank
    them, back in, locked up again."""
    life.flags["answer16"] = me
    courier = life.flags.get("knock16")
    yield from _walk_indoors(me, life, "apt.entry")
    yield Turn(90.0, label="coming to the door")
    yield Wait(
        lambda: "waiting16" in life.flags,
        lambda t: clips.idle(t, me.seed),
        label="coming to the door",
        timeout=8.0,
    )
    door = life.door(DOOR16)
    if door is not None:
        door.unlock()  # (held open while he is out on the step)
    yield _wait_for_door(life, "opening the door", lambda t: clips.idle(t, me.seed))
    yield Glide(
        stoop.x, stoop.z, yaw=90.0, y=stoop.y, duration=1.0, label="answering the door",
        walk_first=False,
    )  # fmt: skip
    life.flags["at_door16"] = me
    yield Wait(
        lambda: getattr(courier, "offering", None) == "parcel",
        lambda t: clips.talk(t, me.seed, listening=True),
        label="answering the door",
        timeout=6.0,
    )
    if getattr(courier, "offering", None) == "parcel":
        take = _there_and_back(1.2)

        def got_it() -> None:
            courier.hold_both(None)  # type: ignore[union-attr]
            me.hold_both("parcel")
            life.flags["taken16"] = True

        yield Play(
            lambda t: clips.offer(take(t), (0.0, 1.22, 0.42)),
            1.2,
            label="taking the parcel",
            events={0.6: got_it},
        )
        yield Play(lambda t: clips.talk(t, me.seed), 1.4, label="thanking the courier")
        yield Play(lambda t: clips.hold_up(t, me.seed), 2.0, label="reading the label")
    yield Turn(stoop.yaw, label="turning back to the door")
    entry = life.place("apt.entry")
    yield Glide(
        entry.x, entry.z, yaw=entry.yaw, y=0.0, duration=1.0, label="going in",
        carry="two" if me.hands["r"] else None, walk_first=False,
    )  # fmt: skip
    if door is not None:
        door.lock()
    for flag in ("answer16", "at_door16", "taken16"):
        life.flags.pop(flag, None)
    if me.hands["r"] == "parcel":
        yield from _put_down_at_home(me, life, "apt.desk", "putting the parcel down")


def smoker(me, life: AlleyLife) -> Story:
    """Hangs about by the wall near the Halcyon door: a cigarette, the phone, another
    cigarette; stops for a chat with whoever comes by; noodles now and then."""
    spot = life.place("smoke")
    rounds = 0
    yield from keep_hours(me, life)
    yield Walk(spot.x, spot.z, label="going back to the wall")
    while True:
        if not schedule(me.role).out(life.hour):  # (a nap, or a late lunch: number 17)
            me.hold("r", None)
            yield from keep_hours(me, life)
            yield Walk(spot.x, spot.z, label="going back to the wall")
        yield Turn(spot.yaw, label="leaning on the wall")
        me.hold("r", "cigarette")
        yield Wait(
            lambda: me in life.chats,
            lambda t: clips.smoke(t, me.seed),
            label="having a smoke",
            timeout=24.0,
        )
        me.hold("r", None)
        if me in life.chats:
            yield from chat(me, life, life.chats[me], starts=False)
            continue
        me.hold("r", "phone")
        yield Wait(
            lambda: me in life.chats,
            lambda t: clips.phone(t, me.seed),
            label="scrolling the phone",
            timeout=10.0,
        )
        me.hold("r", None)
        if me in life.chats:
            yield from chat(me, life, life.chats[me], starts=False)
            continue
        rounds += 1
        if rounds % 3 == 0 and not life.open.get("sushi", True):
            yield from vending_run(me, life)
            yield Walk(spot.x, spot.z, label="going back to the wall")
        elif rounds % 3 == 0:
            yield from buy_food(me, life, "noodles")
            if me.hands["r"]:
                yield Walk(spot.x, spot.z, label="back to the wall with the noodles", carry="held")
                yield Turn(spot.yaw)
                yield from eat_standing(me, life, 12.0)
                yield from bin_it(me, life)
            yield Walk(spot.x, spot.z, label="going back to the wall")


def chat(me, life: AlleyLife, other, starts: bool) -> Story:
    """Stand facing each other and talk, taking turns, for a while."""
    yield Turn(face_towards(me, other.x, other.z), label="chatting")
    total = 12.0

    def turn_taking(t: float) -> HumanoidPose:
        mine = int((t + (0.0 if starts else 2.0)) / 2.0) % 2 == 0
        pose = clips.talk(t, me.seed, listening=not mine)
        pose.head_yaw = (
            wrap_angle_deg(face_towards(me, other.x, other.z) - me.yaw) * 0.8 + pose.head_yaw * 0.2
        )
        return pose

    yield Play(turn_taking, total, label="chatting")
    life.chats.pop(me, None)
    life.chats.pop(other, None)


def stroller(me, life: AlleyLife) -> Story:
    """Wanders up and down the alley: stops to look up at the neon, browses the junk
    table for a bit without buying, stops for a chat with the smoker, goes home (number
    35) now and then."""
    points = list(me.waypoints)
    n = 0
    while True:
        yield from keep_hours(me, life)
        n += 1
        target = me.rng.choice(points)
        yield Walk(target[0], target[1], label="strolling")
        roll = me.rng.random()
        if roll < 0.3:
            yield Play(lambda t: _look_up(t, me.seed), 4.0, label="looking up at the neon")
        elif roll < 0.5 and life.has("junk.browse"):
            look = life.place("junk.browse")
            # either end of the table - whichever nobody is standing at
            ends = [(look.x - 0.05, look.z + 0.65), (look.x - 0.05, look.z - 2.05)]
            free = [e for e in ends if not _occupied(me, *e, margin=0.45)]
            if free:
                yield Walk(*free[0], label="having a look at the junk")
                yield Turn(look.yaw)
                yield Play(
                    lambda t: _browse_low(t, me.seed, _scale(me)),
                    6.0,
                    label="browsing without buying",
                )
        else:
            yield Play(lambda t: clips.idle(t, me.seed), 2.5, label="standing about")
        if n % 3 == 0:
            smokers = [
                o
                for o in me.neighbours
                if o.role == "smoker" and not o.hidden and o not in life.chats
            ]
            if smokers:
                other = smokers[0]
                spot = life.place("chat")
                yield Walk(spot.x, spot.z, label="going over for a chat")
                if math.hypot(other.x - me.x, other.z - me.z) < 2.5 and other not in life.chats:
                    life.chats[other] = me
                    life.chats[me] = other
                    yield from chat(me, life, other, starts=True)
        if n % 5 == 0 and _bar_open(life) and me.rng.random() < 0.5:
            yield from _a_drink(me, life)  # (a drink at the Neon Moth)
        elif n % 7 == 0:
            yield from leave_and_return(me, life, 20.0 + me.rng.uniform(0.0, 10.0))


def _look_up(t: float, seed: float) -> HumanoidPose:
    pose = clips.idle(t, seed)
    pose.head_pitch = -28.0 + 4.0 * math.sin(t * 0.5)
    pose.head_yaw = 30.0 * math.sin(t * 0.35 + seed)
    return pose


STORIES: dict[str, Callable[..., Story]] = {
    "sushi_chef": sushi_chef,
    "junk_dealer": junk_dealer,
    "office_worker": office_worker,
    "tinkerer": tinkerer,
    "courier": courier,
    "resident": resident,
    "smoker": smoker,
    "stroller": stroller,
}
"""Role name -> story. Every place a story uses must be in the level (``needs``)."""

ASIDES: dict[str, Callable[..., Story | None]] = {
    "resident": dell_sees_the_machine,
}
"""Role name -> what can catch their eye while they are walking somewhere out of doors:
``notice(me, life)`` returns an aside story to play out first (or None), after which the
walk goes on (``Storyteller``)."""

NEEDS: dict[str, tuple[str, ...]] = {
    "sushi_chef": (
        "sushi.chef",
        "sushi.board",
        "sushi.boxes",
        "sushi.cooler",
        "sushi.wok",
        "sushi.serve",
        "sushi.pass",
    ),
    "junk_dealer": ("junk.dealer", "junk.stock"),
    "office_worker": (
        "sushi.customer",
        "sushi.queue",
        "sushi.menu",
        "sushi.pass",
        "sushi.stool",
        "bin",
        "mouth",
    ),
    "tinkerer": ("junk.customer", "junk.browse", "sushi.customer", "sushi.queue", "bin", "mouth"),
    "courier": (
        "junk.customer",
        "door16.stoop",
        "door16.front",
        "door16.parcel",
        "door16.beside",
        "mouth",
    ),
    "resident": (
        "door16.stoop",
        "door16.front",
        "door16.beside",
        "apt.entry",
        "apt.kitchen",
        "apt.sofa",
        "apt.by_sofa",
        "apt.desk",
        "apt.sink",
        "apt.rail",
        "apt.bath",
        "sushi.customer",
        "sushi.queue",
        "sushi.menu",
        "sushi.pass",
    ),
    "smoker": ("smoke", "chat", "sushi.customer", "bin"),
    "stroller": ("chat", "mouth"),
}


class Storyteller:
    """Runs one person's story: feeds it each act's result, steps the current act, and
    starts the story over if it ever ends."""

    def __init__(self, npc, life: AlleyLife, role: str) -> None:
        self.npc, self.life, self.role = npc, life, role
        self.story: Story | None = None
        self.act: Act | None = None
        self.returning = 0.0
        # the story (and the walk it was on) put on hold while an aside plays out
        self.parked: tuple[Story, Act] | None = None

    def reset(self) -> None:
        self.story = None
        self.act = None
        self.returning = 0.0
        self.parked = None

    @property
    def in_aside(self) -> bool:
        return self.parked is not None

    def _aside(self) -> None:
        """Something catches their eye on the way somewhere (``ASIDES``: Dell, out on an
        errand, seeing the vending machine down the alley on the blink): the walk is put on
        hold and the aside is played out; then the walk goes on from wherever they are."""
        if self.parked is not None or not isinstance(self.act, Walk) or self.story is None:
            return
        notice = ASIDES.get(self.role)
        aside = notice(self.npc, self.life) if notice is not None else None
        if aside is None:
            return
        try:
            first = next(aside)  # (a fresh generator takes no ``send`` of a result)
        except StopIteration:
            return
        self.parked = (self.story, self.act)
        self.story, self.act = aside, first
        first.begin(self.npc)

    RETURN_NEAR = 0.12  # this near the spot they were pushed from, they carry on
    RETURN_PATIENCE = 8.0  # seconds to get back to it before carrying on from where they are

    def _get_back(self, dt: float) -> bool:
        """After being pushed aside (``StreetPerson.shove``): back to where they were
        (round whatever is in the way) and facing the way they were, then the act goes
        on - a walk goes on from wherever they are, a scripted step from where they got
        back to. Returns True once done."""
        npc = self.npc
        back = getattr(npc, "shoved_from", None)
        if back is None:
            return True
        act = self.act
        x, z, yaw = back
        self.returning = getattr(self, "returning", 0.0) + dt
        done = isinstance(act, Walk) or act is None
        if not done and math.hypot(x - npc.x, z - npc.z) > self.RETURN_NEAR:
            arrived = npc.navigate_towards(
                x, z, dt, npc.frame_obstacles, stop_dist=self.RETURN_NEAR * 0.5,
                avoid=npc.frame_bodies,
            )  # fmt: skip
            done = (arrived and npc.arrival != "there") or self.returning > self.RETURN_PATIENCE
            if not done:
                return False
        if not done and not npc.rotate_towards(yaw, dt) and self.returning < self.RETURN_PATIENCE:
            npc.pose = clips.idle(self.returning, npc.seed)
            return False
        npc.shoved_from = None
        self.returning = 0.0
        if isinstance(act, Glide) and act.start is not None:
            # the step goes on from here, in the time it had left
            act.duration = max(0.3, act.duration - act.time)
            act.time = 0.0
            act.start = (npc.x, npc.z, npc.y, npc.yaw)
        return True

    def step(self, dt: float) -> None:
        if not self._get_back(dt):
            return
        self._aside()
        if self.act is not None and dt > 0.0:
            make_way(self.npc, self.act, dt)
        for _ in range(6):  # an act can finish at once (an event, a turn already done)
            if self.act is None:
                try:
                    if self.story is None:
                        self.story = _from_where_they_are(self.npc, self.life, self.role)
                        self.act = next(self.story)
                    else:
                        self.act = self.story.send(self.npc.last_result)
                except StopIteration:
                    if self.parked is not None:  # the aside is over: back to the walk
                        self.story, self.act = self.parked
                        self.parked = None
                        self.act.restart(self.npc)
                    else:
                        self.story = None
                        continue
                else:
                    self.act.begin(self.npc)
            done = self.act.step(self.npc, dt)
            self.npc.activity = self.act.label or self.npc.activity
            if not done:
                return
            self.npc.last_result = self.act.result
            self.act = None
            dt = 0.0


def _from_where_they_are(me, life: AlleyLife, role: str) -> Story:
    """``role``'s story - begun from wherever they are: somebody who finds themselves in
    the bar when their story starts (over - a game loaded while they sat over a drink)
    gets up and goes out first; the bartender's own story sees to himself."""
    from cyberduke.ai.nightlife import in_the_bar, leave_the_bar

    if role != "bartender" and life.has("bar.entry") and in_the_bar(life, me.x, me.z):
        life.bar.leave_seat(me)
        me.hold_both(None)
        yield from leave_the_bar(me, life)
    yield from STORIES[role](me, life)


MAKE_WAY_WHILE = (
    "queueing", "eating", "standing about", "scrolling the phone", "having a smoke",
    "waiting their turn", "finding somewhere",
)  # fmt: skip
MAKE_WAY_SPEED = 0.6  # m/s: a shuffle to one side
MAKE_WAY_MAX = 0.9  # metres one act will give way by, all told


def make_way(npc, act: Act, dt: float) -> None:
    """Somebody has been held up walking into ``npc`` (see ``BaseNPC.held_up_by``) while
    it stands about doing something that can be done a step to one side: it shuffles
    out of their way - sideways to their line of travel, away from it - as long as that
    is clear of obstacles."""
    if not isinstance(act, (Wait, Play)) or not act.label.startswith(MAKE_WAY_WHILE):
        return
    moved = getattr(act, "made_way", 0.0)
    if moved >= MAKE_WAY_MAX:
        return
    for other in getattr(npc, "neighbours", ()):
        held = getattr(other, "held_up_by", None)
        if held is None or getattr(other, "blocked_time", 0.0) < 0.5:
            continue
        if math.hypot(held[0] - npc.x, held[1] - npc.z) > 0.05:
            continue
        r = math.radians(other.yaw)
        lateral = (math.cos(r), -math.sin(r))  # the walker's left
        away = (npc.x - other.x) * lateral[0] + (npc.z - other.z) * lateral[1]
        sign = 1.0 if away >= 0.0 else -1.0
        step = MAKE_WAY_SPEED * dt
        for k in (sign, -sign):
            nx, nz = npc.x + lateral[0] * k * step, npc.z + lateral[1] * k * step
            obstacles = getattr(npc, "frame_obstacles", None) or []
            if obstacles and npc._body_collides(nx, nz, obstacles):
                continue
            npc.x, npc.z = nx, nz
            act.made_way = moved + step  # type: ignore[attr-defined]
            return


def people_about(npcs: Sequence, player, me) -> list[tuple[float, float, float]]:
    """Everyone ``me`` should steer round: the others who are out (not indoors, not lying
    down) and the player."""
    bodies = [
        (o.x, o.z, o.collision_radius)
        for o in npcs
        if o is not me and not getattr(o, "hidden", False) and not o.fall_angle
    ]
    if player is not None:
        bodies.append((player.x, player.z, player.radius))
    return bodies


__all__ = [
    "NEEDS",
    "STORIES",
    "AlleyLife",
    "Counter",
    "LooseItem",
    "Order",
    "Storyteller",
    "people_about",
]


# The bar and the metro: the bartender's and the regular's stories, a drink at the bar,
# getting on and off the trains - registered with the stories above (``ai.nightlife``)
from cyberduke.ai import nightlife as _nightlife  # noqa: E402, F401
