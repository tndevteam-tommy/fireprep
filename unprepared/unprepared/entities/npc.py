from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import NPCSettings, PerceptionSettings
from ..core.daynight import DayNightCycle
from ..core import dialogue
from ..core.math3d import BoundingBox3D, Vector3D, ray_intersects_box
from .door import Door
from .equipment import EquipSlot, Figure, equipment_from_outfit, figure_from_outfit
from .inventory import add_item, take_item
from .items import Item
from .outfit import Outfit, outfit_for
from .player import Player

if TYPE_CHECKING:  # pragma: no cover
    from ..world.world import World

Notify = Optional[Callable[[str], None]]


class AIState(Enum):
    SLEEPING = 1
    PATROL = 2
    SHOPKEEPING = 3
    INVESTIGATING = 4   # walking to where a noise came from / looking around
    REPORTING = 5       # a citizen crying for the guards
    ALERT_GUARDS = 6    # a guard who has seen a crime and is after the player
    TRAVELING = 7       # walking to the location of the current schedule task
    EATING = 8
    GUARDING = 9        # standing post (guard house / tower / jail), awake and watchful
    RESPONDING = 10     # a guard running to a cry for help
    FOLLOWING = 11      # a guard tailing a skulking player
    IDLE = 12           # at a task location but unable to do it (no food...)
    WAITING = 13        # standing awake, watching the player, until they leave (a jailer after a
    #                     release, a resident who has asked a late guest to go)
    STROLLING = 14      # a townsperson wandering a route along the street (pausing at each stop)
    SHOPPING = 15       # standing at a merchant's counter, browsing (a simulated customer)
    WORKING = 16        # at a work station: the anvil, the forge, the kitchen, hauling crates
    DRINKING = 17       # seated at an inn table
    CLEANING = 18       # the innkeeper going table to table
    WELCOMING = 19      # the innkeeper going to (and standing with) somebody who has just come in without a room


ALARMED_STATES = (AIState.REPORTING, AIState.ALERT_GUARDS)
BUSY_STATES = (AIState.INVESTIGATING, AIState.REPORTING, AIState.ALERT_GUARDS, AIState.RESPONDING, AIState.FOLLOWING,
               AIState.WAITING, AIState.WELCOMING)
ROUTE_STATES = (AIState.PATROL, AIState.STROLLING, AIState.CLEANING)     # tasks with ``patrol_points`` walk a round
# states in which an NPC is at leisure to stop for a chat
SOCIABLE_STATES = (AIState.PATROL, AIState.GUARDING, AIState.SHOPKEEPING, AIState.STROLLING, AIState.SHOPPING,
                   AIState.WORKING, AIState.DRINKING, AIState.CLEANING, AIState.EATING, AIState.IDLE)


@dataclass
class Speech:
    """Something an NPC just said (at ``started``): shown as a bubble above their head until ``until``."""
    text: str
    started: float
    until: float


@dataclass
class TheftReport:
    """A shopkeeper's report that goods went missing from an area."""
    reporter: str
    area: str
    day: int
    items: Dict[str, int]        # item id -> missing quantity
    names: Dict[str, str]        # item id -> display name


@dataclass
class Sighting:
    """A guard's memory of the player acting suspiciously somewhere."""
    area: str
    day: int


@dataclass
class CrimeMemory:
    """What a guard holds against the player. Reported (unseen) crimes fade in days,
    crimes the guard saw with their own eyes last for weeks. The penalty is fixed when the
    crime is declared (a theft indoors costs more than one in the street)."""
    crime: str
    day: int
    witnessed: bool
    expires_day: int
    fine: float
    jail_days: int = 0             # served on top of paying the fine
    jail_days_unpaid: int = 1      # served instead when the fine cannot be paid
    indoors: bool = False
    declared_at: float = 0.0       # world time it was declared (the same deed re-declared moments later is one crime)

    def active(self, day: int) -> bool:
        return day < self.expires_day

    @property
    def key(self):
        """The same crime remembered by several guards is punished once."""
        return (self.crime, self.day, self.indoors)

    @property
    def label(self) -> str:
        return f"{self.crime} ({'indoors' if self.indoors else 'in the street'})" if self.jail_days else self.crime


@dataclass
class ScheduleTask:
    """One block of an NPC's daily routine.

    The NPC walks to ``target_pos`` (path-finding through doors it can open),
    then adopts ``target_state``. Door bookkeeping lets residents/shopkeepers
    open up on arrival and lock up when they leave:

    * ``unlock_doors`` - unlocked (left as they are otherwise) when the NPC arrives
    * ``open_doors``   - unlocked and opened when the NPC arrives
    * ``close_doors``  - closed (left unlocked) when the NPC arrives
    * ``lock_doors``   - locked and closed when the NPC arrives (locking up for the night)

    (Owners always shut and lock their own doors behind them anyway - see ``_touch_doors``.)
    ``patrol_points`` turns the task into a round: after arriving the NPC keeps
    walking from point to point (pausing briefly at each) until the task ends.
    ``food_container`` names a container (a table) to eat from during EATING tasks.
    """
    start_hour: int
    end_hour: int
    target_pos: Vector3D
    target_state: AIState
    action_text: str
    unlock_doors: Tuple[str, ...] = ()
    close_doors: Tuple[str, ...] = ()
    lock_doors: Tuple[str, ...] = ()
    depart_text: Optional[str] = None
    patrol_points: Tuple[Vector3D, ...] = ()
    open_doors: Tuple[str, ...] = ()
    facing: Optional[float] = None   # yaw to turn to on arrival (0 = -z, 90 = +x); None keeps the walking direction
    food_container: Optional[str] = None

    def covers(self, hour: int) -> bool:
        return self.start_hour <= hour < self.end_hour


@dataclass
class DoorPassage:
    """A door this NPC opened (or owns) and has to shut again once it is through."""
    door_id: str
    lock: bool               # lock it again after closing
    entry_side: int = 0      # sign of (z - hinge z) when the NPC reached the door; 0 = unknown
    passed: bool = False


class NPC:
    EYE_HEIGHT = 1.6
    TARGET_HEIGHT = 1.35     # the player's chest when standing: low furniture (counters, tables) does not hide them
    SEARCH_SECONDS = 20.0    # how long a responding guard looks around before giving up
    FOLLOW_SECONDS = 12.0    # how long a guard keeps tailing a player who stopped skulking
    BODY_RADIUS = 0.4        # personal space: NPCs never walk into each other or the player
    CROWD_TOLERANCE = 2.0    # within this of the goal a blocked NPC is happy to stop where it stands
    BLOCKED_PATIENCE = 3.0   # seconds of being blocked before settling for "near enough"
    BLOCKED_GIVE_UP = 8.0    # seconds of being blocked anywhere before abandoning the walk
    RUN_MULTIPLIER = 1.6     # a guard answering a cry or chasing a thief runs this much faster than a walk
    REPATH_INTERVAL = 0.5    # seconds between a pursuing guard's re-plans towards a moving player

    def __init__(self, name: str, is_guard: bool, is_fence: bool, can_trade: bool, home_pos: Vector3D,
                 settings: Optional[PerceptionSettings] = None, npc_settings: Optional[NPCSettings] = None,
                 faction: Optional[str] = None):
        self.settings = settings or PerceptionSettings()
        self.npc_settings = npc_settings or NPCSettings()
        self.name = name
        self.faction = faction if faction is not None else ("guards" if is_guard else None)
        self.is_guard = is_guard
        self.is_fence = is_fence
        self.can_trade = can_trade
        self.position = home_pos
        self.home_position = home_pos.copy()
        self.state = AIState.SLEEPING
        self.suspicion_meter = 0.0
        self.has_torch = False
        self.schedule: List[ScheduleTask] = []
        # the days an NPC lives in turn (``use_day``): ``schedule`` is today's; somebody who does
        # the same every day (a shopkeeper) has just the one
        self.day_plans: List[List[ScheduleTask]] = []
        self.inventory: List[Item] = []
        self.view_distance = self.settings.guard_view_distance if is_guard else self.settings.civilian_view_distance
        # routine / movement
        self.current_task: Optional[ScheduleTask] = None
        self.path: List[Vector3D] = []
        self.pending_locks: List[DoorPassage] = []
        self.patrol_index = 0
        self.patrol_wait = 0.0
        self.facing_yaw = 180.0          # direction the NPC looks (degrees; 0 = -z, 90 = +x)
        self.blocked_timer = 0.0         # seconds spent unable to step forward (someone in the way)
        self.gold = 0.0                  # coins in the purse (stolen by pickpockets, spent on purchases)
        self.base_gold = 0.0             # topped back up to this every restock cycle
        self.last_gold_restock_day = 0
        self.trade_categories: Tuple[str, ...] = ()   # a specialist merchant's categories (empty = trades anything)
        self.trade_availability = "at_work"   # when the player may trade: "at_work" (at the counter / anvil), "awake"
        #                                       (an innkeeper: rooms are let whatever she is doing), "always" (a fence:
        #                                       woken up for business and back to bed afterwards)
        self.awake_for_business = False       # a fence roused from sleep by a customer (``resume_routine`` afterwards)
        self.role_tag = ""               # flavour for dialogue ("smith", "innkeeper", ...)
        self.outfit: Optional[Outfit] = None   # what they wear (None: ``outfit_for`` the role when drawn)
        self.warden_of: Tuple[str, ...] = ()   # areas an outlaw or a knight keeps (see ``World._warden_watch``)
        self.warden_exit: Optional[Vector3D] = None
        self.warden_seize_at: Optional[float] = None   # world time the warden lays hands on the intruder
        self.vigilance: Optional[float] = None  # how alert (skill checks); None: by role
        self.personal_lines: Tuple[str, ...] = ()   # their own story, told a line at a time (``greeting``)
        self.personal_index = 0               # how much of it the player has heard
        self.times_greeted = 0                # how often the player has talked to them (``greeting``)
        self._equipment: Optional[Dict[EquipSlot, Item]] = None   # the outfit as real worn items (built on first use)
        self.rng = random.Random(zlib.crc32(name.encode("utf-8")) & 0xFFFF)   # for picking phrasings (the world installs its shared one)
        # talk
        self.time = 0.0                  # world time as of the last movement update
        self.speech: Optional[Speech] = None
        self.conversation = None         # the Conversation this NPC is part of, if any
        self.next_chat_time = 0.0        # world time before which this NPC will not start another chat
        self.chat_pause = False          # standing still mid-conversation
        self.chatter_timer = 20.0        # seconds until the next bit of ambient chatter
        # crime bookkeeping
        self.crimes_witnessed = 0
        self.sightings: List[Sighting] = []
        self.known_thefts: List[TheftReport] = []
        self.pending_confrontations: List[TheftReport] = []
        self.crime_memories: List[CrimeMemory] = []
        # alarm / investigation
        self.goal: Optional[Vector3D] = None       # where an investigating/responding guard is heading
        self.report_to: Optional[TheftReport] = None   # a guard on his way to take a shopkeeper's theft report
        self.report_timer = 0.0                   # how long he has been trying to reach the reporter
        self.report_in_progress: Optional[TheftReport] = None   # the report being taken face to face right now
        self.report_interrupted: Optional[TheftReport] = None   # a report a crime in front of him cut short: he goes back
        self.report_apologise = False             # ... and says sorry for the interruption when he does
        self.search_timer = 0.0
        self.follow_timer = 0.0
        self.charge_ready_at = 0.0       # world time before which a guard holding a crime against the player may not charge
        self.repath_timer = 0.0          # seconds until a pursuing guard re-plans the way to a moving player
        self.heard_noise_at: Optional[Vector3D] = None
        self.clock = 0.0                 # seconds of perception time, for cooldowns
        self.last_scold: Dict[str, float] = {}     # warning kind -> when it was last said
        self.torch_warnings = 0          # "light a torch" warnings given since the player last had one lit
        self.pending_crime: Optional[str] = None   # a crime this NPC has just declared for the world to record
        self.warning_grace_until = 0.0   # suspicion cannot reach the alarm before this (time to comply)
        # asking a late guest to leave
        self.curfew_asks = 0
        self.curfew_ask_timer = 0.0
        self.waiting_for: Optional[str] = None     # why the NPC is WAITING ("release", "curfew")
        self.locked_out = False          # walking to / standing outside a building it cannot get into yet
        self.retry_timer = 0.0           # seconds until a blocked task is tried again
        # shopkeeping / household
        self.shop_containers: List[str] = []
        self.restock_days = 3
        self.restock_tiers: list = []
        self.base_stock: list = []
        self.base_inventory: list = []
        self.inventory_reset_days = 1
        self.last_inventory_reset_day = 0
        self.expected_stock: Dict[str, int] = {}
        self.expected_names: Dict[str, str] = {}
        self.last_restock_day = 0
        self.hungry = False
        # stealth bookkeeping
        self.sneak_check_timer = 0.0
        self.sneak_unnoticed_until = 0.0
        self.sneak_hidden = False        # set by the world while a successful sneak roll is in effect

    # ------------------------------------------------------------------ identity
    @property
    def title(self) -> str:
        return self.name

    @property
    def is_alarmed(self) -> bool:
        return self.state in ALARMED_STATES

    @property
    def is_busy(self) -> bool:
        return self.state in BUSY_STATES or self.awake_for_business

    @property
    def is_asleep(self) -> bool:
        return self.state is AIState.SLEEPING

    @property
    def is_patrolling(self) -> bool:
        """Walking a round (guards on patrol, strollers, the innkeeper cleaning)."""
        return (self.state in ROUTE_STATES and self.current_task is not None
                and bool(self.current_task.patrol_points))

    @property
    def is_open_for_business(self) -> bool:
        """Whether the player may trade right now (``trade_availability``): merchants from their
        counter (the smith also at his anvil and forge); an innkeeper whenever she is awake and
        not dealing with an alarm; a fence always - even asleep (``wake_for_business``)."""
        if not self.can_trade or self.is_alarmed:
            return False
        if self.trade_availability == "always":
            return True
        if self.trade_availability == "awake":
            return not self.is_asleep and (self.state not in BUSY_STATES or self.state is AIState.WELCOMING)
        return self.state in (AIState.SHOPKEEPING, AIState.WORKING)

    def wake_for_business(self) -> bool:
        """A customer at a sleeping fence's bedside: up, and open for business until the deal is
        done (``is_busy`` meanwhile, so the routine does not reclaim them). Returns True if woken."""
        if not self.is_asleep:
            return False
        self.state = AIState.SHOPKEEPING
        self.awake_for_business = True
        return True

    def back_to_business_as_usual(self, world: "World", notify_callback: Notify = None) -> None:
        """The customer has gone: a fence roused for a deal goes back to what the hour calls for (bed)."""
        if self.awake_for_business:
            self.awake_for_business = False
            self.resume_routine(world, notify_callback)

    @property
    def is_sociable(self) -> bool:
        return not self.is_asleep and self.state in SOCIABLE_STATES and self.suspicion_meter < self.settings.schedule_override_threshold

    @property
    def clothes(self) -> Outfit:
        """The outfit to draw: the one the level gave them, or the one their role calls for."""
        if self.outfit is None:
            self.outfit = outfit_for(self.name, self.role_tag, self.is_guard)
        return self.outfit

    @property
    def equipment(self) -> Dict[EquipSlot, Item]:
        """What they wear, as equipment items mounted on their figure - the outfit made real
        (``equipment_from_outfit``). Kept apart from the pockets: a helmet on a head is not
        something a pickpocket lifts."""
        if self._equipment is None:
            self._equipment = equipment_from_outfit(self.clothes, owner=self.name)
        return self._equipment

    @property
    def figure(self) -> Figure:
        return figure_from_outfit(self.clothes)

    def buys(self, item: Item) -> bool:
        """A specialist (the smith, the tailor) only deals in certain kinds of goods."""
        return not self.trade_categories or item.category in self.trade_categories

    # ------------------------------------------------------------------ talk
    SPEECH_BASE_SECONDS = 7.5            # a bubble stays up this long ...
    SPEECH_SECONDS_PER_CHAR = 0.16       # ... plus reading time for a longer line

    def say(self, text: str, notify_callback: Notify = None, duration: Optional[float] = None,
            log_prefix: Optional[str] = None) -> Speech:
        """Speak a line: it goes to the log as ``(Name): "text"`` (``log_prefix`` replaces the name,
        e.g. "Orrin witnessed Theft!") and into a speech bubble above the NPC's head."""
        if duration is None:
            duration = self.SPEECH_BASE_SECONDS + self.SPEECH_SECONDS_PER_CHAR * len(text)
        self.speech = Speech(text, self.time, self.time + duration)
        if notify_callback:
            notify_callback(f"({log_prefix or self.name}): \"{text}\"")
        return self.speech

    def phrase(self, kind: str, **fmt) -> str:
        return dialogue.line(kind, self.rng, **fmt)

    def current_speech(self, now: float) -> Optional[Speech]:
        if self.speech is not None and self.speech.until <= now:
            self.speech = None
        return self.speech

    @property
    def is_shopkeeper(self) -> bool:
        return self.can_trade and not self.is_fence

    @property
    def stock(self) -> List[Item]:
        """Items this NPC offers for sale from its own pockets (never keys or other ``for_sale=False`` items)."""
        return [item for item in self.inventory if item.for_sale]

    @property
    def forward(self) -> Tuple[float, float]:
        """Unit (x, z) vector the NPC faces."""
        rad = math.radians(self.facing_yaw)
        return math.sin(rad), -math.cos(rad)

    def faces(self, target: Vector3D) -> bool:
        """True if ``target`` lies inside the NPC's field of view (in front of it)."""
        dx, dz = target.x - self.position.x, target.z - self.position.z
        dist = math.hypot(dx, dz)
        if dist < 1e-6:
            return True
        fx, fz = self.forward
        return (fx * dx + fz * dz) / dist >= math.cos(math.radians(self.settings.view_half_angle_degrees))

    def turn_towards(self, target: Vector3D) -> None:
        dx, dz = target.x - self.position.x, target.z - self.position.z
        if dx or dz:
            self.facing_yaw = math.degrees(math.atan2(dx, -dz)) % 360.0

    def has_line_of_sight(self, target: Vector3D, walls: Sequence[BoundingBox3D], target_height: Optional[float] = None) -> bool:
        """Eye to the target's chest (``target_height`` above its feet; a crouching player's is
        lower, so a counter or table can hide them)."""
        eye = Vector3D(self.position.x, self.position.y + self.EYE_HEIGHT, self.position.z)
        aim = Vector3D(target.x, target.y + (self.TARGET_HEIGHT if target_height is None else target_height), target.z)
        candidates = walls.along(eye, aim) if hasattr(walls, "along") else walls      # (an indexed world: just the nearby ones)
        return not any(ray_intersects_box(eye, aim, wall) for wall in candidates)

    @staticmethod
    def chest_height(player: Player) -> float:
        """Where to look for the player: a little below their eyes, whatever their stance."""
        return max(0.3, player.current_eye_height - 0.2)

    def can_see(self, target: Vector3D, blockers: Sequence[BoundingBox3D], target_height: Optional[float] = None) -> bool:
        """Awake, within view distance, inside the field of view and not behind a wall."""
        return (not self.is_asleep and self.position.distance_to(target) <= self.view_distance
                and self.faces(target) and self.has_line_of_sight(target, blockers, target_height))

    # ------------------------------------------------------------------ doors & keys
    def has_key_for(self, door: Door) -> bool:
        return any(item.opens(door.door_id) for item in self.inventory)

    def owns(self, door: Door) -> bool:
        return door.owned_by(self.name, self.faction)

    def can_open(self, door: Door) -> bool:
        """Anyone can open an unlocked door; a locked one needs the key - even its owner.
        A *sealed* gate (a prisoner behind the bars) is never opened by an NPC."""
        if door.sealed:
            return False
        return (not door.is_locked) or self.has_key_for(door)

    def can_lock(self, door: Door) -> bool:
        return self.has_key_for(door)

    # ------------------------------------------------------------------ memory
    def remember_sighting(self, area: str, day: int) -> None:
        if not any(s.area == area and s.day == day for s in self.sightings):
            self.sightings.append(Sighting(area, day))
            del self.sightings[:-12]

    def learn_theft(self, report: TheftReport) -> None:
        """Guards hear a shopkeeper's report; if they saw the player lurking there, they will confront them."""
        self.known_thefts.append(report)
        if any(s.area == report.area and report.day - s.day <= 2 for s in self.sightings):
            self.pending_confrontations.append(report)

    CRIME_REPEAT_WINDOW = 5.0    # seconds within which the same crime declared again is the same deed

    def recent_memory(self, crime: str, day: int, indoors: bool, now: float) -> Optional[CrimeMemory]:
        """The memory of this very deed, if it was declared within ``CRIME_REPEAT_WINDOW`` of ``now``."""
        for memory in self.crime_memories:
            if (memory.key == (crime, day, indoors) and memory.active(day)
                    and 0.0 <= now - memory.declared_at <= self.CRIME_REPEAT_WINDOW):
                return memory
        return None

    def remember_crime(self, crime: str, day: int, witnessed: bool, fine: float, rng=None, jail_days: int = 0,
                       jail_days_unpaid: int = 1, indoors: bool = False, now: float = 0.0) -> CrimeMemory:
        """Hold a crime against the player; unseen (reported) crimes fade in 3-5 days, witnessed in 3-5 weeks.
        One deed declared several times within moments - a fumbled pickpocket is reported by the
        bystanders' roll, the victim and the cry for help alike - is held once: the existing memory
        is returned (upgraded to *witnessed* if the guard has now seen it for themselves). The same
        kind of crime later in the day is a new one."""
        span = (rng.randint(21, 35) if rng else 28) if witnessed else (rng.randint(3, 5) if rng else 4)
        memory = self.recent_memory(crime, day, indoors, now)
        if memory is not None:
            if witnessed and not memory.witnessed:
                memory.witnessed = True
                memory.expires_day = max(memory.expires_day, day + span)
            return memory
        memory = CrimeMemory(crime, day, witnessed, day + span, fine, jail_days, jail_days_unpaid, indoors, declared_at=now)
        self.crime_memories.append(memory)
        return memory

    def active_memories(self, day: int) -> List[CrimeMemory]:
        self.crime_memories = [m for m in self.crime_memories if m.active(day)]
        return list(self.crime_memories)

    def forget_crimes(self) -> None:
        self.crime_memories.clear()
        self.pending_confrontations.clear()

    @property
    def on_a_report(self) -> bool:
        """Busy with a shopkeeper's theft report: on the way to take it, taking it, or owing one a
        return visit. Another report waits for another guard."""
        return self.report_to is not None or self.report_in_progress is not None or self.report_interrupted is not None

    def raise_alarm(self, notify_callback: Notify, guard_line: Optional[str] = None, citizen_line: Optional[str] = None,
                    log_prefix: Optional[str] = None) -> None:
        """Drop everything and shout: guards go after the player, citizens cry for the guards.
        Lines default to a phrasing of "halt" / "cry_for_help". A guard taking a theft report
        remembers it (``report_interrupted``) and goes back to finish it afterwards."""
        self.path = []
        self.patrol_wait = 0.0
        if self.is_guard and (self.report_to is not None or self.report_in_progress is not None):
            self.report_interrupted = self.report_to or self.report_in_progress
            self.report_to = None
            self.report_in_progress = None
            self.report_apologise = True
        self.end_conversation()
        if self.warden_of:
            # an outlaw or a knight calls no watch: they come for the intruder themselves
            self.state = AIState.ALERT_GUARDS
            self.say(dialogue.warden_line(self, "shout", self.rng), notify_callback, log_prefix=log_prefix)
        elif self.is_guard:
            self.state = AIState.ALERT_GUARDS
            self.say(guard_line or self.phrase("halt"), notify_callback, log_prefix=log_prefix)
        else:
            self.state = AIState.REPORTING
            self.say(citizen_line or self.phrase("cry_for_help"), notify_callback, log_prefix=log_prefix)

    def end_conversation(self) -> None:
        """Break off a chat (the world finishes the bookkeeping for the other party)."""
        conv = self.conversation
        if conv is not None:
            conv.finished = True
        self.chat_pause = False

    # ------------------------------------------------------------------ daily routine
    def use_day(self, day: int) -> bool:
        """Take up the plan for day ``day`` of the game (the plans come round in turn). Returns
        whether the schedule changed."""
        if len(self.day_plans) < 2:
            return False
        plan = self.day_plans[day % len(self.day_plans)]
        if plan is self.schedule:
            return False
        self.schedule = plan
        return True

    def task_for(self, hour: int) -> Optional[ScheduleTask]:
        for task in self.schedule:
            if task.covers(hour):
                return task
        return None

    def update_schedule(self, current_hour: int, notify_callback: Notify = None, world: Optional["World"] = None,
                        doors: bool = True) -> None:
        """Switch to the task covering ``current_hour`` (unless too suspicious or busy to care).
        ``doors=False`` adopts the task without its door side effects - used when a save is
        loaded, so a door the player unlocked is not relocked by an NPC merely *resuming* its
        task. (Torches are lit by the world wherever it is dark, not by the hour.)"""
        task = self.task_for(current_hour)
        if task is None or self.suspicion_meter >= self.settings.schedule_override_threshold or self.is_busy:
            return
        if task is not self.current_task:
            self.begin_task(task, world, notify_callback, doors=doors)
        elif self.state not in (AIState.TRAVELING, task.target_state):
            # Calmed down after an alarm/investigation: resume the routine.
            self.begin_task(task, world, notify_callback, doors=doors)

    def catch_up_schedule(self, current_hour: int, notify_callback: Notify = None, world: Optional["World"] = None) -> bool:
        """Called every frame: an NPC that could not switch tasks on the hour (too suspicious, or
        busy with an alarm, an investigation, a wait) takes up the hour's task as soon as it is
        free to - not a whole hour later. Returns True if a task was adopted."""
        task = self.task_for(current_hour)
        if task is None or task is self.current_task:
            return False
        if self.suspicion_meter >= self.settings.schedule_override_threshold or self.is_busy:
            return False
        self.begin_task(task, world, notify_callback)
        return True

    def retry_blocked_task(self, world: "World", delta_time: float, notify_callback: Notify = None) -> None:
        """Locked out: every ``locked_out_retry`` seconds see whether the door has opened."""
        if not (self.locked_out and self.state is AIState.IDLE and self.current_task is not None):
            return
        self.retry_timer -= delta_time
        if self.retry_timer > 0.0:
            return
        task = self.current_task
        self.current_task = None                     # so begin_task treats it as fresh (no departure effects)
        self.begin_task(task, world, notify_callback)

    def resume_routine(self, world: "World", notify_callback: Notify = None, skip_reports: bool = False) -> None:
        """Drop whatever interrupted the NPC and go back to the current task - unless a theft
        report was cut short by a crime in front of them: that is finished first
        (``World.resume_report``)."""
        if self.report_interrupted is not None and not skip_reports and world is not None and hasattr(world, "resume_report"):
            if world.resume_report(self, notify_callback):
                return
        self.goal = None
        self.report_to = None
        self.report_in_progress = None
        self.search_timer = 0.0
        self.follow_timer = 0.0
        task = self.task_for(world.clock.hour)
        self.current_task = None
        if task is not None:
            self.begin_task(task, world, notify_callback)
        else:
            self.state = AIState.IDLE

    def begin_task(self, task: ScheduleTask, world: Optional["World"] = None, notify_callback: Notify = None,
                   doors: bool = True) -> None:
        previous = self.current_task
        self.current_task = task
        if previous is not task:
            self.patrol_index = 0
        self.path = world.find_path(self, task.target_pos) if world is not None else []
        if previous is not None and previous is not task and world is not None:
            world.on_task_left(self, previous, task)
        self.locked_out = False
        if not self.path or self.position.distance_to(task.target_pos) <= self.npc_settings.arrival_tolerance:
            self.path = []
            if world is not None and self.position.distance_to(task.target_pos) > self.npc_settings.arrival_tolerance:
                # No route yet (a shop not open, a door somebody else has the key to): go and wait
                # outside its door and try again in a little while.
                spot = world.waiting_spot_for(task.target_pos)
                outside = world.find_path(self, spot) if spot is not None else []
                self.locked_out = True
                if outside and self.position.distance_to(spot) > self.npc_settings.arrival_tolerance * 3:
                    self.path = outside
                    self.state = AIState.TRAVELING
                else:
                    self.state = AIState.IDLE
                    self.retry_timer = self.npc_settings.locked_out_retry
                    if previous is not task:
                        self.say(self.phrase("locked_out"))
                return
            self.arrive(world, notify_callback, doors=doors)
        else:
            self.state = AIState.TRAVELING
            if task.depart_text:
                self.say(task.depart_text, notify_callback)

    def arrive(self, world: Optional["World"] = None, notify_callback: Notify = None, doors: bool = True) -> None:
        task = self.current_task
        if task is None:
            return
        if self.locked_out:                          # reached the doorstep of a place still shut
            self.state = AIState.IDLE
            self.retry_timer = self.npc_settings.locked_out_retry
            self.say(self.phrase("locked_out"))
            return
        self.state = task.target_state
        self.patrol_wait = 0.0
        if task.facing is not None:
            self.facing_yaw = task.facing
        if world is not None and doors:
            for door_id in task.unlock_doors:
                door = world.find_door(door_id)
                if door is not None and self.can_lock(door):
                    door.is_locked = False
            for door_id in task.open_doors:
                door = world.find_door(door_id)
                if door is not None and self.can_open(door):
                    door.unlock_and_open()
            for door_id in task.close_doors:
                door = world.find_door(door_id)
                if door is not None:
                    door.is_open = False
            for door_id in task.lock_doors:
                door = world.find_door(door_id)
                if door is not None and self.can_lock(door):
                    door.lock_and_close()
        if task.target_state is AIState.EATING:
            self._eat(world, notify_callback, task)
        elif notify_callback:
            notify_callback(task.action_text)      # narration ("Barnaby opened the shop."): logged plainly, never as speech
        if world is not None:
            world.on_task_arrived(self, task)

    def _eat(self, world: Optional["World"], notify_callback: Notify, task: ScheduleTask) -> None:
        """Meals consume one food unit from the NPC's own pockets or the table it sits at."""
        sources = []
        if world is not None and task.food_container:
            container = world.find_container(task.food_container)
            if container is not None:
                sources.append(container.inventory)     # what is on the table first, then the pockets
        sources.append(self.inventory)
        for source in sources:
            for item in source:
                if item.is_food:
                    take_item(source, item, 1)
                    if source is self.inventory and item.id in self.expected_stock:
                        self.expected_stock[item.id] = max(0, self.expected_stock[item.id] - 1)   # eating stock is not theft
                    self.hungry = False
                    if notify_callback:
                        notify_callback(task.action_text)          # narration, not speech
                    return
        self.hungry = True
        self.state = AIState.IDLE
        self.say(self.phrase("no_food"), notify_callback)

    # ------------------------------------------------------------------ movement
    def go_to(self, world: "World", target: Vector3D, state: AIState) -> bool:
        """Leave the routine to walk somewhere (investigate / respond). False if unreachable."""
        path = world.find_path(self, target)
        self.goal = target.copy()
        self.path = path
        self.state = state
        self.patrol_wait = 0.0
        return bool(path)

    def pursue(self, world: "World", target: Vector3D, notify_callback: Notify = None) -> None:
        """A guard with a crime against the player and the player in view: shout "Halt!" (once)
        and run after them, re-planning the way every ``REPATH_INTERVAL`` as they move. The
        charge itself waits until the guard has caught up (``World._guard_sees_player``)."""
        if self.state is not AIState.ALERT_GUARDS:
            self.raise_alarm(notify_callback)
        self.search_timer = 0.0                      # the quarry is in sight
        if self.repath_timer > 0.0:
            return
        moved = self.goal is None or self.goal.distance_to(target) > 1.0
        stalled = not self.path and self.position.distance_to(target) > self.CROWD_TOLERANCE
        if moved or stalled:                         # (a stale goal with no path left: start again)
            self.go_to(world, target, AIState.ALERT_GUARDS)
            self.repath_timer = self.REPATH_INTERVAL

    def update_movement(self, delta_time: float, world: "World", notify_callback: Notify = None) -> None:
        """Walk along the current path, operating doors on the way; call every frame.

        Other NPCs and the player are solid: a step into somebody is refused (a sidestep
        is tried first). Being blocked near the goal counts as arriving - a crowd of
        guards converging on one spot stands *around* it instead of inside each other -
        and being blocked anywhere for long enough abandons the walk.
        """
        start = self.position.copy()
        self.time = getattr(world, "elapsed", self.time + delta_time)
        if self.chat_pause:
            self._process_pending_locks(world)
            return                                   # stopped for a chat: the round resumes afterwards
        if self.is_patrolling and not self.path:
            self._advance_patrol(world, delta_time)
        moving_states = (AIState.TRAVELING, AIState.RESPONDING, AIState.INVESTIGATING, AIState.FOLLOWING, AIState.WAITING,
                         AIState.ALERT_GUARDS, AIState.WELCOMING)
        if (self.state in moving_states or self.is_patrolling) and self.path:
            running = self.state in (AIState.RESPONDING, AIState.ALERT_GUARDS)    # answering a cry, or after a thief
            remaining = self.npc_settings.move_speed * delta_time * (self.RUN_MULTIPLIER if running else 1.0)
            bodies = world.bodies_near(self) if hasattr(world, "bodies_near") else []
            # somebody already overlapping us (spawned on the same spot) must not pin us down
            bodies = [b for b in bodies if not self._crowded(self.position.x, self.position.z, [b])]
            blocked = False
            while remaining > 0 and self.path:
                target = self.path[0]
                dx, dz = target.x - self.position.x, target.z - self.position.z
                dist = (dx * dx + dz * dz) ** 0.5
                if dist > 1e-6:
                    self.facing_yaw = math.degrees(math.atan2(dx, -dz)) % 360.0
                if dist <= remaining or dist <= self.npc_settings.arrival_tolerance:
                    step_x, step_z = target.x, target.z
                    cost = dist
                else:
                    step_x = self.position.x + dx / dist * remaining
                    step_z = self.position.z + dz / dist * remaining
                    cost = remaining
                if bodies and self._crowded(step_x, step_z, bodies):
                    goal = self.path[-1]
                    if self.position.distance_to(goal) <= self.CROWD_TOLERANCE:
                        self.path = []                 # somebody is standing on our spot: near enough
                        break
                    side = self._sidestep(dx, dz, dist, remaining, bodies)
                    if side is None:
                        blocked = True
                        break
                    self.position.x, self.position.z = side
                    self._settle(world, target, 0.0)
                    remaining = 0
                    continue
                self.position.x, self.position.z = step_x, step_z
                self._settle(world, target, 1.0 if cost == dist else cost / dist)
                remaining -= cost
                if cost == dist:
                    self.path.pop(0)
            if blocked:
                self.blocked_timer += delta_time
                goal = self.path[-1] if self.path else None
                near_goal = goal is not None and self.position.distance_to(goal) <= self.CROWD_TOLERANCE
                if near_goal and self.blocked_timer >= self.BLOCKED_PATIENCE:
                    self.path = []                     # near enough: stand here
                    self.blocked_timer = 0.0
                elif self.blocked_timer >= self.BLOCKED_GIVE_UP:
                    self.path = []
                    self.blocked_timer = 0.0
                    if self.state is AIState.TRAVELING:
                        self.state = AIState.IDLE      # try again on the next hour tick
                        self._touch_doors(world, start, self.position)
                        self._process_pending_locks(world)
                        return
            else:
                self.blocked_timer = 0.0
            self._touch_doors(world, start, self.position)
            if not self.path and self.state is AIState.TRAVELING:
                self.arrive(world, notify_callback)
        elif self.state is AIState.TRAVELING:
            self.arrive(world, notify_callback)
        self._process_pending_locks(world)

    def _settle(self, world, target: Vector3D, fraction: float) -> None:
        """Height after a step: the world says (stairs, the ground, a bridge); with no world that
        knows, the old rule - rise or drop with the waypoints."""
        settle = getattr(world, "settle", None)
        if settle is not None:
            settle(self, target, fraction)
        elif target.y != self.position.y:
            self.position.y = target.y if fraction >= 1.0 else self.position.y + (target.y - self.position.y) * fraction

    def _crowded(self, x: float, z: float, bodies) -> bool:
        """Would standing at (x, z) overlap another body? ``bodies`` are (x, z, radius)."""
        for bx, bz, radius in bodies:
            if math.hypot(x - bx, z - bz) < radius + self.BODY_RADIUS:
                return True
        return False

    def _sidestep(self, dx: float, dz: float, dist: float, step: float, bodies):
        """Try stepping to either side of the blocked direction, then straight away from the
        people in the way (hemmed in between two of them, sideways is no use); the free spot
        or None."""
        if dist <= 1e-6:
            return None
        fx, fz = dx / dist, dz / dist                # forward
        px, pz = -fz, fx                             # perpendicular
        candidates = [(fx * 0.3 + px * sign, fz * 0.3 + pz * sign) for sign in (1.0, -1.0)]
        near = sorted(bodies, key=lambda b: math.hypot(self.position.x - b[0], self.position.z - b[1]))[:2]
        if near:
            ax = sum(self.position.x - b[0] for b in near)
            az = sum(self.position.z - b[1] for b in near)
            if abs(ax) + abs(az) > 1e-6:
                candidates.append((ax, az))          # away from the (one or two) nearest bodies
        for cx, cz in candidates:
            mag = math.hypot(cx, cz) or 1.0
            x = self.position.x + cx / mag * step
            z = self.position.z + cz / mag * step
            if not self._crowded(x, z, bodies):
                return x, z
        return None

    def _advance_patrol(self, world: "World", delta_time: float) -> None:
        """Pause at the current waypoint, then plan the walk to the next one."""
        points = self.current_task.patrol_points
        self.patrol_wait += delta_time
        if self.patrol_wait < self.npc_settings.patrol_pause:
            return
        self.patrol_wait = 0.0
        self.patrol_index = (self.patrol_index + 1) % len(points)
        self.path = world.find_path(self, points[self.patrol_index])

    def _door_region(self, door: Door) -> BoundingBox3D:
        return _inflate(door.get_bounding_box(), self.npc_settings.door_operate_reach)

    def _add_passage(self, door_id: str, lock: bool) -> DoorPassage:
        for passage in self.pending_locks:
            if passage.door_id == door_id:
                passage.lock = passage.lock or lock
                return passage
        passage = DoorPassage(door_id, lock)
        self.pending_locks.append(passage)
        return passage

    def _will_pass_through(self, door: Door, start: Optional[Vector3D] = None) -> bool:
        """True if the path crosses the door's plane inside the doorway - the step just taken
        (from ``start``; a long frame can carry an NPC clean through a doorway) or the next
        few waypoints."""
        points = [self.position, *self.path[:3]]
        if start is not None:
            points.insert(0, start)
        return any(_crosses_door_line(door, a, b) for a, b in zip(points, points[1:]))

    def _touch_doors(self, world: "World", start: Vector3D, end: Vector3D) -> None:
        """Open a closed door the NPC is about to walk through (if it has the right to) and
        remember to shut it again. A door the NPC *owns* is always shut and locked behind
        it - even if somebody else left it open; a door it merely opened is closed again and
        relocked if it was locked. Walking past an open door changes nothing."""
        reach = self.npc_settings.door_operate_reach + 1.0
        lo_x, hi_x = min(start.x, end.x), max(start.x, end.x)
        lo_z, hi_z = min(start.z, end.z), max(start.z, end.z)
        doors_in = getattr(world, "doors_in", None)
        nearby = doors_in(lo_x - reach, lo_z - reach, hi_x + reach, hi_z + reach) if doors_in is not None else world.doors
        for door in nearby:
            if hi_x < door.x1 - reach or lo_x > door.x2 + reach or hi_z < door.z1 - reach or lo_z > door.z2 + reach:
                continue                                 # (the step is nowhere near this doorway)
            region = self._door_region(door)
            if not (_contains_xz(region, end) or _segment_crosses_xz(region, start, end)):
                continue
            if not self._will_pass_through(door, start):
                continue
            kept_open = door.schedule is not None and door.open_hours_cover(world.clock.hour)   # (its open hours: nobody locks it)
            if not door.is_open:
                if not self.can_open(door):
                    continue
                was_locked = door.is_locked
                door.unlock_and_open()
                if not door.auto_closes:
                    continue                     # bars stay as they were left: open and unlocked
                passage = self._add_passage(door.door_id, lock=(was_locked or self.owns(door)) and self.can_lock(door)
                                            and not kept_open)
            elif door.auto_closes and self.owns(door) and self.can_lock(door) and not kept_open:
                passage = self._add_passage(door.door_id, lock=True)
            else:
                continue
            passage.passed = True
            if passage.entry_side == 0:
                passage.entry_side = _side_of(door, start)

    def _process_pending_locks(self, world: "World") -> None:
        """Shut (and lock) doors the NPC walked through once it is clear of them."""
        for passage in list(self.pending_locks):
            door = world.find_door(passage.door_id)
            if door is None:
                self.pending_locks.remove(passage)
                continue
            if _contains_xz(self._door_region(door), self.position):
                passage.passed = True
                continue
            if self.position.distance_to(door.center) > self.npc_settings.door_lock_clearance:
                went_through = passage.entry_side == 0 or _side_of(door, self.position) != passage.entry_side
                if went_through or not passage.passed:
                    if passage.lock and self.can_lock(door) and not (door.schedule is not None and door.open_hours_cover(world.clock.hour)):
                        door.lock_and_close()
                    else:
                        door.is_open = False
                self.pending_locks.remove(passage)

    # ------------------------------------------------------------------ perception
    def hearing_radius_for(self, noise: float) -> float:
        return noise * self.settings.hearing_radius_per_noise

    def evaluate_perception(self, player: Player, ambient_light: float, is_moving: bool, is_sprinting: bool,
                            walls: Sequence[BoundingBox3D], simulated_hour: int, notify_callback: Notify = None,
                            delta_time: float = 1.0 / 60.0, trespassing: bool = False,
                            noise_attenuation: float = 1.0) -> None:
        """Update suspicion for one frame of ``delta_time`` seconds.

        Hearing only wakes sleepers and makes NPCs look/investigate (they cannot tell
        who is out there); only *seeing* the player misbehaving raises the suspicion
        that ends in an alarm. ``noise_attenuation`` is what the walls let through
        (1.0 when player and NPC share a room or are both outside). A fence watches as
        warily as any merchant but never calls the watch over skulking - only a theft from
        them brings the cry. Warnings are rate-limited (``scold``) and open a grace window
        before the alarm can go off.
        """
        s = self.settings
        self.clock += delta_time
        if self.is_asleep and not is_moving:
            self._decay(delta_time)
            return
        is_daytime = DayNightCycle.is_daytime(simulated_hour)
        # the torch law is for the streets after dark: indoors (a lit room, a house) nobody expects one
        dark_and_unlit = not is_daytime and not player.has_lit_torch and not getattr(player, "indoors", False)
        if player.has_lit_torch:
            self.torch_warnings = 0                  # they listened
        is_acting_suspicious = player.is_crouching or (dark_and_unlit and is_moving) or trespassing

        p_noise = player.get_noise_level(is_moving, is_sprinting) * max(0.0, min(1.0, noise_attenuation))
        dist = self.position.distance_to(player.position)
        if is_moving and dist <= self.hearing_radius_for(p_noise) and not self.sneak_hidden:
            if self.is_asleep:
                if p_noise > s.wake_noise_threshold:
                    self.state = AIState.INVESTIGATING
                    self.heard_noise_at = player.position.copy()
                    self.say(self.phrase("woke_up"), notify_callback, log_prefix=f"{self.name} woke up!")
            else:
                if is_acting_suspicious:
                    self.heard_noise_at = player.position.copy()
                if p_noise >= s.racket_noise and not self.is_alarmed:
                    # a racket - steel clanking down the street at a run: not a crime, but remarked on
                    self.scold("racket", "racket_guard" if self.is_guard else "racket_citizen", notify_callback, grace=False)

        if not is_acting_suspicious:
            self._decay(delta_time)
            return
        if self.is_asleep:
            return

        effective_view_dist = self.view_distance * (
            s.torch_view_multiplier if self.has_torch else max(s.min_light_view_multiplier, ambient_light))
        p_vis = player.get_visibility(ambient_light, is_moving)
        if self.has_torch and dist <= s.torch_light_reach:
            # the NPC's own torch lights up whoever is close - as seen through their clothes and stance
            p_vis = max(p_vis, player.get_visibility(s.torch_lit_visibility, is_moving))
        seen = (dist <= effective_view_dist and self.faces(player.position) and not self.sneak_hidden
                and self.has_line_of_sight(player.position, walls, self.chest_height(player))
                and (p_vis > s.min_visibility_to_notice or trespassing))
        if seen:
            gain = (p_vis / max(dist, 1.0)) * s.seen_suspicion_gain
            if trespassing:
                gain += s.trespass_suspicion_gain
                self.scold("trespass", "trespass", notify_callback, player=player)
            if dark_and_unlit and self.is_guard and self.scold("torch", "torch_law", notify_callback, player=player):
                self.torch_warnings += 1
                if self.torch_warnings >= s.torch_warnings_before_crime:
                    self.pending_crime = "No torch after dark"      # warned enough: the world records the crime
                    self.torch_warnings = 0
            self.suspicion_meter += gain * delta_time
            if self.suspicion_meter >= s.schedule_override_threshold:
                self.turn_towards(player.position)
                if self.is_guard:
                    self.scold("skulking", "skulking", notify_callback, player=player)
            if self.in_warning_grace:
                self.suspicion_meter = min(self.suspicion_meter, s.alarm_threshold - 1.0)
        else:
            self._decay(delta_time * 0.5)

        if self.suspicion_meter >= s.alarm_threshold and not self.is_alarmed and seen and not self.is_fence:
            self.raise_alarm(notify_callback)

    def _decay(self, delta_time: float) -> None:
        if self.suspicion_meter > 0:
            self.suspicion_meter = max(0.0, self.suspicion_meter - self.settings.suspicion_decay * delta_time)

    def scold(self, kind: str, line: str, notify_callback: Notify, grace: bool = True,
              player: Optional[Player] = None) -> bool:
        """Say ``line`` (a phrasing is picked when ``line`` names a dialogue kind) unless the same
        kind of warning was given within ``scold_cooldown``. A warning opens a grace window:
        suspicion may climb but cannot trigger the alarm until it has passed, giving the player
        time to do as they are told. A guard's dressing-down holds the player's attention (the
        controls lock and the view swings round to face them)."""
        last = self.last_scold.get(kind)
        if last is not None and self.clock - last < self.settings.scold_cooldown:
            return False
        self.last_scold[kind] = self.clock
        if grace:
            self.warning_grace_until = max(self.warning_grace_until, self.clock + self.settings.warning_grace)
        if self.warden_of and kind == "trespass":
            text = dialogue.warden_line(self, "warn", self.rng)      # an outlaw threatens no watch; a knight has none to call
        else:
            text = dialogue.line(line, self.rng) if line in dialogue.LINES else line
        self.say(text, notify_callback)
        if player is not None and self.is_guard:
            player.hold_attention(self.position, self.settings.scold_freeze_seconds)
        return True

    @property
    def in_warning_grace(self) -> bool:
        return self.clock < self.warning_grace_until

    # ------------------------------------------------------------------ waiting on the player
    def wait_for(self, reason: str, target: Optional[Vector3D] = None) -> None:
        """Stand still, awake, watching ``target`` until the world says the wait is over."""
        self.path = []
        self.goal = None
        self.patrol_wait = 0.0
        self.state = AIState.WAITING
        self.waiting_for = reason
        if target is not None:
            self.turn_towards(target)

    def stop_waiting(self, world: "World", notify_callback: Notify = None) -> None:
        self.waiting_for = None
        self.curfew_asks = 0
        self.curfew_ask_timer = 0.0
        self.resume_routine(world, notify_callback)

    def sees_crime(self, player_pos: Vector3D, walls: Sequence[BoundingBox3D], target_height: Optional[float] = None) -> bool:
        """Could this NPC see a crime committed at ``player_pos`` right now: awake, in range, in the
        field of view and with a clear line to the player's chest (``target_height``:
        ``chest_height(player)`` - a crouching thief behind a counter is hidden from this exactly
        as from ordinary perception)."""
        if self.is_asleep:
            return False
        dist = self.position.distance_to(player_pos)
        return dist <= self.view_distance and self.faces(player_pos) and self.has_line_of_sight(player_pos, walls, target_height)

    def witness_crime(self, crime_name: str, player_pos: Vector3D, walls: Sequence[BoundingBox3D],
                      notify_callback: Notify = None, target_height: Optional[float] = None) -> bool:
        """See the player commit a crime (``sees_crime``) and raise the alarm. An NPC already
        raising the alarm sees it too but does not shout again or count it twice - one fumbled
        dip into somebody's pockets is one crime, however many rolls noticed it. Returns True
        whenever the crime was seen."""
        if not self.sees_crime(player_pos, walls, target_height):
            return False
        if not self.is_alarmed:
            self.crimes_witnessed += 1
            self.raise_alarm(notify_callback, self.guard_callout(), self.phrase("thief_guards"),
                             log_prefix=f"{self.title} witnessed {crime_name}!")
        return True

    GUARD_CALLOUTS = dialogue.LINES["caught_red_handed"]

    def guard_callout(self) -> str:
        """Escalating line for the n-th crime this guard has seen (the last line repeats)."""
        return self.GUARD_CALLOUTS[min(self.crimes_witnessed, len(self.GUARD_CALLOUTS)) - 1]

    # ------------------------------------------------------------------ dialogue
    STORY_HOURS = (6, 22)        # people only open up about themselves at a civil hour

    def greeting(self, hour: int = 12) -> str:
        """What they say when the player talks to them: one of several phrasings suited to who
        they are and whether they are open / awake - and every other time, when they are up and
        about at a civil hour, the next line of their own story instead (``personal_lines``, in
        order; once it is all told, now and again one of them again)."""
        lines = self.personal_lines
        self.times_greeted += 1
        chatty = (lines and self.times_greeted % 2 == 0 and self.STORY_HOURS[0] <= hour < self.STORY_HOURS[1]
                  and not self.is_asleep and not self.is_alarmed and not (self.can_trade and not self.is_open_for_business))
        if chatty:
            if self.personal_index < len(lines):
                self.personal_index += 1
                return lines[self.personal_index - 1]
            if self.rng.random() < 0.5:
                return self.rng.choice(list(lines))
        return dialogue.player_greeting(self, hour, self.rng)

    def interact(self, notify_callback: Notify = None, hour: int = 12) -> None:
        self.say(self.greeting(hour), notify_callback)

    # ------------------------------------------------------------------ inventory upkeep
    def reset_inventory(self, day: int) -> List[Item]:
        """Replace missing personal items (keys, rations) from the base inventory. Returns what was restored."""
        restored: List[Item] = []
        have = {item.id: item for item in self.inventory}
        for item_def in self.base_inventory:
            current = have.get(item_def.id)
            if current is None:
                unit = item_def.build(owner=self.name)
                add_item(self.inventory, unit)
                restored.append(unit)
            elif item_def.stackable and current.quantity < item_def.quantity:
                unit = item_def.build(owner=self.name)
                unit.quantity = item_def.quantity - current.quantity
                add_item(self.inventory, unit)
                restored.append(unit)
        if self.is_fence:
            for item in self.inventory:
                item.is_stolen = False
        self.last_inventory_reset_day = day
        return restored

    # ------------------------------------------------------------------ persistence
    def reset_transient(self) -> None:
        """Forget everything that only makes sense mid-session: a conversation in progress, a wait
        on the player, a warning just given, an investigation under way. Called when a save is
        loaded into this NPC, so nothing from the old session leaks into the new one."""
        self.path = []
        self.pending_locks = []
        self.patrol_wait = 0.0
        self.blocked_timer = 0.0
        self.goal = None
        self.report_to = None
        self.report_timer = 0.0
        self.report_in_progress = None
        self.report_interrupted = None
        self.report_apologise = False
        self.search_timer = 0.0
        self.follow_timer = 0.0
        self.repath_timer = 0.0
        self.heard_noise_at = None
        self.pending_crime = None
        self.last_scold = {}
        self.torch_warnings = 0
        self.warning_grace_until = 0.0
        self.curfew_asks = 0
        self.curfew_ask_timer = 0.0
        self.waiting_for = None
        self.locked_out = False
        self.warden_seize_at = None
        self.retry_timer = 0.0
        self.awake_for_business = False
        self.speech = None
        self.conversation = None
        self.chat_pause = False
        self.next_chat_time = 0.0
        self.hungry = False
        self.sneak_check_timer = 0.0
        self.sneak_unnoticed_until = 0.0
        self.sneak_hidden = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "position": self.position.to_dict(),
            "suspicion_meter": self.suspicion_meter,
            "state": self.state.value,
            "has_torch": self.has_torch,
            "inventory": [item.to_dict() for item in self.inventory],
            "crimes_witnessed": self.crimes_witnessed,
            "facing_yaw": self.facing_yaw,
            "gold": self.gold,
            "last_gold_restock_day": self.last_gold_restock_day,
            "expected_stock": dict(self.expected_stock),
            "expected_names": dict(self.expected_names),
            "last_restock_day": self.last_restock_day,
            "last_inventory_reset_day": self.last_inventory_reset_day,
            "sightings": [{"area": s.area, "day": s.day} for s in self.sightings],
            "known_thefts": [_report_to_dict(r) for r in self.known_thefts],
            "pending_confrontations": [_report_to_dict(r) for r in self.pending_confrontations],
            "crime_memories": [{"crime": m.crime, "day": m.day, "witnessed": m.witnessed, "expires_day": m.expires_day,
                                "fine": m.fine, "jail_days": m.jail_days, "jail_days_unpaid": m.jail_days_unpaid,
                                "indoors": m.indoors, "declared_at": m.declared_at} for m in self.crime_memories],
            "charge_ready_at": self.charge_ready_at,
            "personal_index": self.personal_index,
            "times_greeted": self.times_greeted,
        }

    def load_dict(self, data: dict) -> None:
        # Forget the running session's moment first: a flag of the old session (a fence roused for a
        # deal, an innkeeper woken to welcome somebody) must not colour the state being loaded.
        self.reset_transient()
        self.position = Vector3D.from_dict(data["position"])
        self.suspicion_meter = data.get("suspicion_meter", 0.0)
        state = data.get("state", 1)
        self.state = AIState(state) if state in {s.value for s in AIState} else AIState.SLEEPING
        if self.is_busy:
            self.state = AIState.IDLE            # interrupted activities are not resumed from a save
        self.has_torch = data.get("has_torch", self.has_torch)
        if "inventory" in data:
            self.inventory = [Item.from_dict(i) for i in data["inventory"]]
        self.crimes_witnessed = data.get("crimes_witnessed", 0)
        self.facing_yaw = data.get("facing_yaw", self.facing_yaw)
        self.gold = float(data.get("gold", self.gold))
        self.last_gold_restock_day = data.get("last_gold_restock_day", self.last_gold_restock_day)
        if "expected_stock" in data:
            self.expected_stock = dict(data["expected_stock"])
        self.expected_names = dict(data.get("expected_names", self.expected_names))
        self.last_restock_day = data.get("last_restock_day", self.last_restock_day)
        self.last_inventory_reset_day = data.get("last_inventory_reset_day", self.last_inventory_reset_day)
        self.sightings = [Sighting(s["area"], s["day"]) for s in data.get("sightings", [])]
        self.known_thefts = [_report_from_dict(r) for r in data.get("known_thefts", [])]
        self.pending_confrontations = [_report_from_dict(r) for r in data.get("pending_confrontations", [])]
        self.crime_memories = [CrimeMemory(m["crime"], m["day"], m["witnessed"], m["expires_day"], m.get("fine", 0.0),
                                           m.get("jail_days", 0), m.get("jail_days_unpaid", 1), m.get("indoors", False),
                                           m.get("declared_at", 0.0))
                               for m in data.get("crime_memories", [])]
        self.charge_ready_at = float(data.get("charge_ready_at", 0.0))
        self.personal_index = min(int(data.get("personal_index", 0)), len(self.personal_lines))
        self.times_greeted = int(data.get("times_greeted", 0))
        # Routine is re-planned from the loaded position by ``World.resync_schedules``; nothing
        # of the previous session's moment-to-moment state survives (``reset_transient`` above).
        self.current_task = None


# ---------------------------------------------------------------------- helpers
def _report_to_dict(r: TheftReport) -> dict:
    return {"reporter": r.reporter, "area": r.area, "day": r.day, "items": dict(r.items), "names": dict(r.names)}


def _report_from_dict(d: dict) -> TheftReport:
    return TheftReport(d["reporter"], d["area"], d["day"], dict(d.get("items", {})), dict(d.get("names", {})))


def _inflate(box: BoundingBox3D, amount: float) -> BoundingBox3D:
    return BoundingBox3D(Vector3D(box.min_point.x - amount, box.min_point.y, box.min_point.z - amount),
                         Vector3D(box.max_point.x + amount, box.max_point.y, box.max_point.z + amount))


def _contains_xz(box: BoundingBox3D, p: Vector3D) -> bool:
    return box.min_point.x <= p.x <= box.max_point.x and box.min_point.z <= p.z <= box.max_point.z


def _side_of(door: Door, p: Vector3D) -> int:
    return door.side_of(p.x, p.z)


def _crosses_door_line(door: Door, a: Vector3D, b: Vector3D) -> bool:
    """Does segment a->b cross the doorway's line (the wall's line, within the door's width)?"""
    line = door.line
    va, vb = door.across_of(a.x, a.z), door.across_of(b.x, b.z)
    if (va - line) * (vb - line) > 0 or va == vb:
        return False
    t = (line - va) / (vb - va)
    ua, ub = door.along_of(a.x, a.z), door.along_of(b.x, b.z)
    u = ua + (ub - ua) * t
    lo, hi = door.span
    return lo - 0.3 <= u <= hi + 0.3


def _segment_crosses_xz(box: BoundingBox3D, a: Vector3D, b: Vector3D) -> bool:
    flat = BoundingBox3D(Vector3D(box.min_point.x, -1e9, box.min_point.z), Vector3D(box.max_point.x, 1e9, box.max_point.z))
    return ray_intersects_box(Vector3D(a.x, 0.0, a.z), Vector3D(b.x, 0.0, b.z), flat)
