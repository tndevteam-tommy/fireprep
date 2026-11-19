"""The people of the street: civilians who wander and keep out of each other's way, and
merchants who work a stall and turn to whoever comes close.

Neither is a threat - they never report the player - but they are solid to the player
(``blocks_player``), they see each other and the player (a head turn toward the nearest
body in view), and they steer around everyone (``BaseNPC.navigate_towards(avoid=...)``),
so a crowd parts instead of walking through itself.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import replace

from cyberduke.ai.alley import NEEDS, STORIES, AlleyLife, Storyteller, people_about
from cyberduke.ai.npc import BaseNPC, HumanoidSpec
from cyberduke.ai.pathfinding import Bounds
from cyberduke.animation import cook_animation, hawker_animation, idle_animation
from cyberduke.animation import street as street_clips
from cyberduke.animation.look import street_look
from cyberduke.config import DETECTION
from cyberduke.core.geometry import AABB, yaw_towards
from cyberduke.core.interp import wrap_angle_deg
from cyberduke.core.inventory import starting_inventory
from cyberduke.entities.player import Player
from cyberduke.systems.detection import SightingSink

Point = tuple[float, float]
Body = tuple[float, float, float]  # x, z, radius

# Palettes for seeded variety: no two passers-by dressed alike
SHIRTS = (
    (0.9, 0.3, 0.3), (0.3, 0.5, 0.95), (0.85, 0.75, 0.2), (0.35, 0.8, 0.6), (0.7, 0.4, 0.9),
    (0.95, 0.55, 0.2), (0.4, 0.4, 0.45), (0.2, 0.8, 0.9), (0.9, 0.9, 0.9), (0.5, 0.25, 0.2),
)  # fmt: skip
PANTS = ((0.25, 0.25, 0.3), (0.15, 0.2, 0.45), (0.5, 0.45, 0.4), (0.1, 0.1, 0.12), (0.4, 0.3, 0.5))
HAIR = (
    (0.1, 0.08, 0.06),
    (0.8, 0.5, 0.1),
    (0.9, 0.2, 0.6),
    (0.2, 0.8, 0.9),
    (0.85, 0.85, 0.8),
    (0.5, 0.1, 0.1),
)
SKIN = ((1.0, 1.0, 1.0), (0.85, 0.72, 0.6), (0.62, 0.45, 0.32), (1.0, 0.92, 0.85), (0.75, 0.6, 0.5))


# Most people on the street wear a plain face (and move their mouths to talk, eat,
# smoke...); a cheerful few rest on a smile (``HumanoidSpec.mood``).
SMILING_ROLES = frozenset({"sushi_chef", "stroller"})
SMOKING_ROLES = frozenset({"smoker"})


def civilian_spec(seed: int, role: str = "") -> HumanoidSpec:
    """A deterministic look for somebody of the street: a skin tone and a height, and -
    this being a city a century into organic cybernetics - a look of their own
    (``animation.look.street_look``): a mohawk or anime spikes, cybernetic eyes, chrome,
    neon trim and loud colours."""
    rng = random.Random(seed)
    rng.choice(HAIR), rng.choice(SHIRTS), rng.choice(PANTS)  # (kept: the heights stay put)
    skin = rng.choice(SKIN)
    # (1.75 - 2.0 m: the world's doors, counters, stools and beds are built for it)
    height = rng.uniform(0.79, 0.9)
    look, palette, mark = street_look(random.Random(seed * 104729 + 17))
    if role in SMOKING_ROLES and "respirator" in look.extras:
        # (a cigarette goes to the lips: nobody smokes through a respirator)
        look = replace(look, extras=look.extras - {"respirator"})
    return HumanoidSpec(
        hair_color=palette.hair,
        hair_color2=palette.hair2,
        shirt_color=palette.shirt,
        jacket_color=palette.jacket,
        pants_color=palette.pants,
        neon_color=palette.neon,
        eye_color=palette.eye,
        boot_color=palette.boots,
        skin_color=skin,
        height=height,
        has_armor=False,
        look=look,
        mood="smile" if role in SMILING_ROLES else "neutral",
        mark=mark,
    )


def _bodies(others: Sequence[BaseNPC], player: Player | None, me: BaseNPC) -> list[Body]:
    return people_about(others, player, me)


class StreetPerson(BaseNPC):
    """Someone with a life on the street: a ``role`` (a story from ``ai.alley``) run
    against the street's shared ``life``, things in their hands, and the option of being
    out of sight (``hidden``: indoors, or gone up the main road)."""

    role = ""
    life: AlleyLife | None = None

    def _setup_story(self, role: str, life: AlleyLife | None) -> None:
        self.role = role
        self.life = life
        self.hands: dict[str, str | None] = {"l": None, "r": None}
        self.hidden = False
        self.riding = False  # aboard a metro train (out of sight: ``ai.nightlife``)
        self.activity = ""
        self.last_result: object = None
        self.serving: object | None = None
        self.offering: str | None = None
        self.neighbours: Sequence[BaseNPC] = ()
        self.frame_obstacles: list[AABB] = []
        self.frame_bodies: list[Body] = []
        self.watch_area = None  # their home (a ``PrivateArea``): they mind who is in it
        self.asleep = False  # in bed (``ai.alley.Sleep``): notices nothing
        self.body = None  # lying down (asleep): how the renderer lays the body out
        self.alarm = 0.0  # seconds left glaring at an intruder
        self.bark = ""  # something they shouted this frame (the simulation captions it)
        self.shouts = 0
        self.face_time = 0.0
        self.shove_timer = 0.0  # seconds left of staggering after being pushed aside
        self.shoved_from: tuple[float, float, float] | None = None  # where they stood
        ok = bool(role) and life is not None and role in STORIES and life.has(*NEEDS[role])
        self.storyteller = Storyteller(self, life, role) if ok and life is not None else None
        self.inventory = starting_inventory(role or "civilian")

    @property
    def has_story(self) -> bool:
        return self.storyteller is not None

    def hold(self, side: str, item: str | None) -> None:
        self.hands[side] = item

    def hold_both(self, item: str | None) -> None:
        self.hands["l"] = self.hands["r"] = item

    def _reset_story(self) -> None:
        self.hands = {"l": None, "r": None}
        self.hidden = False
        self.riding = False
        self.activity = ""
        self.last_result = None
        self.serving = None
        self.offering = None
        self.y = self.base_y
        self.alarm = 0.0
        self.bark = ""
        self.shouts = 0
        self.face_time = 0.0
        self.talking_to = None
        self.asleep = False
        self.body = None
        self.shove_timer = 0.0
        self.shoved_from = None
        self.fall_angle = 0.0
        self.inventory = starting_inventory(self.role or "civilian")
        if self.storyteller is not None:
            self.storyteller.reset()

    # ------------------------------------------------------------------ being pushed
    SHOVE_PAUSE = 0.5  # seconds staggering after the last push, before carrying on
    SHOVE_MAX = 0.25  # metres one frame's push can move somebody
    # which ways to try pushing them, degrees off the push, towards the side of it they are
    # already on first: aside rather than on ahead (nobody is shunted up the street)
    SHOVE_TURNS = (55.0, 30.0, 80.0, -55.0, -30.0, -80.0, 0.0, 110.0, -110.0)

    @property
    def shovable(self) -> bool:
        """Whether the netrunner can push this person out of the way: standing or walking
        (not out of sight, not lying in bed, not sat on a stool or a sofa)."""
        if self.hidden or self.asleep or self.body is not None or self.fall_angle:
            return False
        legs = self.pose.leg_l, self.pose.leg_r
        return not any(leg[0] < -50.0 for leg in legs)

    def shove(
        self,
        px: float,
        pz: float,
        radius: float,
        dx: float,
        dz: float,
        clear,
        half: float | None = None,
    ) -> bool:
        """Pushed by somebody stepping to (px, pz) (a body ``radius`` round, moving by
        (dx, dz)): moves just far enough out of their way - along the push, or as near to
        it as leaves them somewhere they can stand (``clear(x, z)``) - and staggers for a
        moment before carrying on. Returns whether they could be moved."""
        if not self.shovable:
            return False
        length = math.hypot(dx, dz)
        if length < 1e-9:
            return False
        half = self.collision_radius * 0.85 if half is None else half
        gap = half + radius + 0.01  # (the body's square box against the pusher's)
        ux, uz = dx / length, dz / length
        cx, cz = self.x - px, self.z - pz
        # (the side of the push they are on: turning the push that way moves them aside)
        side = 1.0 if (ux * cz - uz * cx) >= 0.0 else -1.0
        for turn in self.SHOVE_TURNS:
            turn *= side
            c, s_ = math.cos(math.radians(turn)), math.sin(math.radians(turn))
            vx, vz = ux * c - uz * s_, ux * s_ + uz * c
            step = self._clearing_step(cx, cz, vx, vz, gap)
            if step is None or step > self.SHOVE_MAX:
                continue
            tx, tz = self.x + vx * step, self.z + vz * step
            if clear(tx, tz):
                if self.shoved_from is None:
                    self.shoved_from = (self.x, self.z, self.yaw)
                self.x, self.z = tx, tz
                self.shove_timer = self.SHOVE_PAUSE
                self._clear_route()
                return True
        return False

    @staticmethod
    def _clearing_step(cx: float, cz: float, vx: float, vz: float, gap: float) -> float | None:
        """The shortest move along (vx, vz) taking a box centred (cx, cz) from the pusher
        to ``gap`` from it on x or on z (the boxes stop overlapping)."""
        best = None
        for c, v in ((cx, vx), (cz, vz)):
            if abs(v) < 1e-6:
                continue
            target = gap if v > 0 else -gap
            step = (target - c) / v
            if step < 0.0:
                step = 0.0 if abs(c) >= gap else None  # (already clear on this axis)
            if step is not None and (best is None or step < best):
                best = step
        return best

    def _stagger(self, player: Player, dt: float) -> None:
        """Pushed aside: a stumbling step back, a look at whoever did it."""
        self.shove_timer = max(0.0, self.shove_timer - dt)
        k = self.shove_timer / self.SHOVE_PAUSE
        pose = street_clips.idle(self.face_time, getattr(self, "seed", 0.0))
        pose.lean = -6.0 * k
        pose.arm_l[1] -= 12.0 * k
        pose.arm_r[1] += 12.0 * k
        pose.arm_l[2] += 18.0 * k
        pose.arm_r[2] -= 18.0 * k
        pose.grip_l = pose.grip_r = "open"
        look = wrap_angle_deg(yaw_towards(self.x, self.z, player.x, player.z) - self.yaw)
        pose.head_yaw = max(-70.0, min(70.0, look))
        pose.hand_l, pose.hand_r = self.hands["l"], self.hands["r"]
        self.pose = pose

    # ------------------------------------------------------------------ home, talk
    ALARM_HOLD = 2.5  # seconds an owner keeps glaring after losing sight of an intruder
    INTRUDER_RATE = (70.0, 25.0)  # detection gain per second: standing, crouched
    SHOUTS = (
        "Hey! What the hell are you doing in my place?!",
        "Get out! GET OUT of my flat!",
        "Who let you in here?! I'm calling somebody!",
    )

    def _mind_home(self, player: Player, dt: float, blockers, detection) -> bool:
        """An owner (``watch_area``: their home) who can see the netrunner inside it
        reports them - the detection meter fills - but only *notices* them once the meter
        has gone red (``DETECTION.red_level``): then they drop what they are doing, turn
        on the stranger and shout. Until then they carry on, only half aware of something.
        Nobody minds the netrunner anywhere else. Returns whether they have noticed."""
        area = self.watch_area
        if area is None or self.hidden or self.asleep:
            self.alarm = 0.0
            return False
        if self.alarm > 0.0:
            self.alarm = max(0.0, self.alarm - dt)
        if not area.contains(player.x, player.z):
            return False
        home = area.contains(self.x, self.z) or self._at_area_door(area)
        if not home:
            return False
        seen = self.can_see_target(
            player.x, player.z, max_range=9.0, fov_angle=75.0, obstacles=blockers,
            target_y=player.sight_target_y,
        )  # fmt: skip
        if not seen:
            return False
        if detection is not None:
            standing, crouched = self.INTRUDER_RATE
            detection.report_sighting(crouched if player.crouching else standing)
        level = DETECTION.max_level if detection is None else detection.level
        if self.alarm <= 0.0 and level <= DETECTION.red_level:
            return False  # (not sure of it yet: they carry on with what they were doing)
        if self.alarm <= 0.0:
            self.bark = self.SHOUTS[self.shouts % len(self.SHOUTS)]
            self.shouts += 1
        self.alarm = self.ALARM_HOLD
        return True

    def _at_area_door(self, area) -> bool:
        life = self.life
        door = life.door(area.door_id) if life is not None and area.door_id else None
        return door is not None and door.distance_to(self.x, self.z) < 2.0

    def _face(self, player: Player, dt: float, angry: bool) -> None:
        """Turns to the netrunner: pointing and shouting (an intruder), or talking."""
        self.target_yaw = yaw_towards(self.x, self.z, player.x, player.z)
        self.rotate_towards(self.target_yaw, dt)
        self.face_time += dt
        if angry:
            pose = street_clips.point_at(min(self.face_time, 0.4) + 0.4, (0.0, 0.1, 1.0))
            pose.head_roll = 4.0 * math.sin(self.face_time * 9.0)
            pose.expression = "shout" if int(self.face_time * 5.0) % 3 else "grimace"
        else:
            pose = street_clips.talk(self.face_time, getattr(self, "seed", 0.0))
        pose.hand_l, pose.hand_r = self.hands["l"], self.hands["r"]
        if self.hands["r"] == "cigarette":  # keeps the cigarette low while talking
            pose.arm_r = street_clips.arm_to((-0.2, 1.12, 0.2), "r", bob=pose.bob)
        pose.head_yaw = 0.0
        self.pose = pose

    def _live(self, player: Player, dt: float, obstacles, others) -> None:
        """One frame of the story."""
        self.frame_obstacles = obstacles or []
        self.neighbours = others
        self.frame_bodies = people_about(others, player, self)
        assert self.storyteller is not None
        self.storyteller.step(dt)
        clip = (self.pose.hand_l, self.pose.hand_r)
        self.pose.hand_l, self.pose.hand_r = self.hands["l"], self.hands["r"]
        if (
            clip == ("phone", "phone")
            and self.hands["l"] is None
            and self.hands["r"]
            in (
                None,
                "phone",
            )
        ):  # (on the phone: their own, out of a pocket, in both hands)
            self.pose.hand_l = self.pose.hand_r = "phone"


def _nearest_in_view(me: BaseNPC, bodies: Sequence[Body], fov: float = 70.0, reach: float = 4.0):
    """The closest body inside the view cone, as (x, z, distance), or None."""
    best = None
    for x, z, _ in bodies:
        d = math.hypot(x - me.x, z - me.z)
        if d > reach or d < 1e-6:
            continue
        if abs(wrap_angle_deg(yaw_towards(me.x, me.z, x, z) - me.yaw)) > fov:
            continue
        if best is None or d < best[2]:
            best = (x, z, d)
    return best


class Civilian(StreetPerson):
    """A passer-by. With a ``role`` it lives out that story (see ``ai.alley``); without
    one it wanders between waypoints (A* between them, steering around other bodies),
    pausing now and then. Either way it looks at whoever comes close and never raises an
    alarm."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        nav_bounds: Bounds,
        waypoints: Sequence[Point],
        seed: int = 0,
        civilian_id: str = "civilian",
        role: str = "",
        life: AlleyLife | None = None,
    ) -> None:
        super().__init__(
            x, y, z, spec=civilian_spec(seed, role), nav_bounds=nav_bounds,
            walk_speed=0.75 + 0.35 * random.Random(seed + 1).random(), turn_speed=160.0,
        )  # fmt: skip
        self.civilian_id = civilian_id
        self.waypoints = [tuple(w) for w in waypoints]
        self.seed = seed
        self.rng = random.Random(seed)
        self.blocks_player = True
        self.state = "WALKING"
        self.pause_timer = 0.0
        self.target: Point = self._pick_target()
        self.idle_time = 0.0
        self.watching: Point | None = None
        self._setup_story(role, life)

    def _pick_target(self) -> Point:
        far = [w for w in self.waypoints if math.hypot(w[0] - self.x, w[1] - self.z) > 2.0]
        return self.rng.choice(far or self.waypoints)

    def reset(self) -> None:
        super().reset()
        self.rng = random.Random(self.seed)
        self.state = "WALKING"
        self.pause_timer = 0.0
        self.target = self._pick_target()
        self.idle_time = 0.0
        self.watching = None
        self._reset_story()

    def update(
        self,
        player: Player,
        dt: float,
        obstacles: list[AABB] | None = None,
        detection: SightingSink | None = None,
        sight_blockers: list[AABB] | None = None,
        others: Sequence[BaseNPC] = (),
    ) -> bool:
        bodies = _bodies(others, player, self)
        if self.shove_timer > 0.0:  # just pushed aside: a moment to recover
            self.frame_obstacles = obstacles or []
            self.frame_bodies = bodies
            self._stagger(player, dt)
            return False
        if self.storyteller is not None:
            spotted = self._mind_home(player, dt, sight_blockers or obstacles or [], detection)
            if spotted or self.alarm > 0.0 or self.talking_to is not None:
                # an intruder in the flat, or the netrunner talking to them: the story
                # waits while they face the netrunner
                self._face(player, dt, angry=spotted or self.alarm > 0.0)
                self.frame_bodies = bodies
                return spotted
            self._live(player, dt, obstacles, others)
            if self.hidden:
                self.watching = None
                return False
            seen = _nearest_in_view(self, bodies)
            self.watching = (seen[0], seen[1]) if seen else None
            # eye contact while walking about or waiting - not in the middle of a job
            if self.watching is not None and _glancing(self.activity):
                look = wrap_angle_deg(yaw_towards(self.x, self.z, *self.watching) - self.yaw)
                _look(self.pose, max(-60.0, min(60.0, look)))
            return False
        seen = _nearest_in_view(self, bodies)
        self.watching = (seen[0], seen[1]) if seen else None

        if self.state == "PAUSING":
            self.pause_timer -= dt
            self.idle_time += dt
            self.pose = idle_animation(self.idle_time, seed=self.seed)
            if self.pause_timer <= 0.0:
                self.state = "WALKING"
                self.target = self._pick_target()
                self._clear_route()
        else:
            arrived = self.navigate_towards(
                self.target[0], self.target[1], dt, obstacles or [], stop_dist=0.35, avoid=bodies
            )
            if arrived:
                self.state = "PAUSING"
                self.pause_timer = self.rng.uniform(1.0, 4.0)
                self.idle_time = 0.0

        # Eye contact: the head follows whoever is nearest in view (body keeps walking)
        if self.watching is not None:
            look = wrap_angle_deg(yaw_towards(self.x, self.z, *self.watching) - self.yaw)
            _look(self.pose, max(-60.0, min(60.0, look)))
        return False  # civilians never report anyone


class Merchant(StreetPerson):
    """Works a stall. With a ``role`` it runs the stall's trade (see ``ai.alley``: taking
    orders, making and wrapping, handing over, chores in between) and glances at whoever
    steps up; without one it faces the counter, turns to whoever comes within
    ``greet_range`` in front of the stall and keeps working - cooking (``style="cook"``)
    or hawking."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        nav_bounds: Bounds,
        face_yaw: float,
        style: str = "cook",
        seed: int = 100,
        merchant_id: str = "merchant",
        greet_range: float = 3.5,
        role: str = "",
        life: AlleyLife | None = None,
    ) -> None:
        super().__init__(
            x, y, z, spec=civilian_spec(seed, role), nav_bounds=nav_bounds, turn_speed=140.0
        )
        self.seed = seed
        self.rng = random.Random(seed)
        self.waypoints: list[Point] = []
        self.merchant_id = merchant_id
        self.face_yaw = face_yaw
        self.style = style
        self.greet_range = greet_range
        self.blocks_player = True
        self.yaw = self.target_yaw = face_yaw
        self.work_time = 0.0
        self.greeting: Point | None = None
        self._setup_story(role, life)

    def reset(self) -> None:
        super().reset()
        self.yaw = self.target_yaw = self.face_yaw
        self.work_time = 0.0
        self.greeting = None
        self.rng = random.Random(self.seed)
        self._reset_story()

    def update(
        self,
        player: Player,
        dt: float,
        obstacles: list[AABB] | None = None,
        detection: SightingSink | None = None,
        sight_blockers: list[AABB] | None = None,
        others: Sequence[BaseNPC] = (),
    ) -> bool:
        self.work_time += dt
        bodies = _bodies(others, player, self)
        if self.shove_timer > 0.0:
            self.frame_obstacles = obstacles or []
            self.frame_bodies = bodies
            self._stagger(player, dt)
            return False
        if self.storyteller is not None:
            self._live(player, dt, obstacles, others)
            # whoever steps up to the counter gets a look
            self.greeting = self._customer(bodies)
            if self.greeting is not None and _glancing(self.activity):
                look = wrap_angle_deg(yaw_towards(self.x, self.z, *self.greeting) - self.yaw)
                _look(self.pose, max(-70.0, min(70.0, look)))
            return False
        self.greeting = self._customer(bodies)
        self.target_yaw = (
            yaw_towards(self.x, self.z, *self.greeting) if self.greeting else self.face_yaw
        )
        self.rotate_towards(self.target_yaw, dt)
        self.pose = (
            cook_animation(self.work_time)
            if self.style == "cook"
            else hawker_animation(self.work_time)
        )
        if self.greeting is not None and self.style == "cook":
            self.pose.head_pitch = 4.0  # looks up from the wok at the customer
        return False

    def _customer(self, bodies) -> Point | None:
        """The nearest body in front of the stall within greet range."""
        customer = None
        for bx, bz, _ in bodies:
            d = math.hypot(bx - self.x, bz - self.z)
            facing = abs(wrap_angle_deg(yaw_towards(self.x, self.z, bx, bz) - self.face_yaw))
            if d <= self.greet_range and facing <= 80.0 and (customer is None or d < customer[2]):
                customer = (bx, bz, d)
        return (customer[0], customer[1]) if customer else None


_GLANCE_WHILE = (
    "walking", "strolling", "waiting", "queueing", "standing", "calling", "having a smoke",
    "heading", "going", "at the counter", "behind the table", "coming", "enjoy",
)  # fmt: skip


def _look(pose, yaw: float) -> None:
    """Turns the head to look ``yaw`` degrees off straight ahead - as far as the pose lets
    it (``HumanoidPose.head_held``: a smoker brings the head back round to the cigarette
    for a drag, and looks back at the passer-by after)."""
    pose.head_yaw = pose.head_yaw + (yaw - pose.head_yaw) * (1.0 - pose.head_held)


def _glancing(activity: str) -> bool:
    """Whether somebody doing ``activity`` would look up at a passer-by."""
    return activity.startswith(_GLANCE_WHILE)
