"""The tutorial: a street urchin in Jae's debt learns the trade the hard way.

A new game starts here, at night. Jae has taken the player in and wants paying back: the
silver candlestick from Barnaby's store. The tutorial teaches the torch law first (it is
dark), then walks the player through picking the store's lock and the theft - at which a
watchman steps in, so they are charged and jailed and learn how the town's justice works -
then, free again, through getting their things back from the confiscation lockbox, dressing
in the [Tab] view, waiting for night with [T], the theft done properly and the sale to Jae
for next to nothing. Jae's sixty gold pays for a pickaxe and a mallet; ore from the deposit
and the anvil make lockpicks; the inn has beds. Then the town is theirs.

Until the candlestick is sold every other door, shelf, pocket and deal is shut
(:meth:`Tutorial.refuses`): the player can only do the job they were given. The stage is
saved with the game. Nothing here draws; the HUD shows :attr:`Tutorial.hint`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

from .core.daynight import DayNightCycle
from .core.math3d import BoundingBox3D, Vector3D, ray_intersects_box
from .entities import AIState, Item

if TYPE_CHECKING:  # pragma: no cover
    from .game import Game


class Stage(str, Enum):
    INTRO = "intro"                # Jae explains the debt; leave the hideout
    TORCH = "torch"                # the torch law, first of all: it is night
    TO_THE_SHOP = "to_the_shop"    # cross the street, pick the store's lock
    IN_THE_SHOP = "in_the_shop"    # the west shelf: the candlestick
    CAUGHT = "caught"              # a guard steps in: "Halt!", the bill, the cells
    JAIL = "jail"                  # serve the night, or break out
    LOCKBOX = "lockbox"            # your things are in the confiscation lockbox
    DRESS = "dress"                # [Tab]: boots on; what clothes do in the shadows
    WAIT_FOR_NIGHT = "wait_for_night"   # [T] until the streets empty
    STEAL_AGAIN = "steal_again"    # the candlestick, unseen this time
    SELL = "sell"                  # to Jae, for barely anything
    TOOLS = "tools"                # a pickaxe and a mallet, bought or stolen
    MINE = "mine"                  # iron ore from the deposit
    FORGE = "forge"                # lockpicks at the anvil
    INN = "inn"                    # somewhere to sleep
    DONE = "done"


LOCKED_STAGES = (Stage.INTRO, Stage.TORCH, Stage.TO_THE_SHOP, Stage.IN_THE_SHOP, Stage.CAUGHT, Stage.JAIL, Stage.LOCKBOX,
                 Stage.DRESS, Stage.WAIT_FOR_NIGHT, Stage.STEAL_AGAIN, Stage.SELL)

TARGET_ITEM = "candlestick"
TARGET_SHELF = "shop_shelf_west"
TARGET_DOOR = "shop_door"
MERCHANT = "Shopkeeper Barnaby"
FENCE = "Jae the Fence"
START_HOUR = 22
START_SPOT = (-12.0, 0.0, 11.0)       # in the hideout, by Jae at his supper (on its floor: ``World.spot`` lifts it)
KINDNESS_GOLD = 60.0
PITY_SENTENCE_DAYS = 1

WATCHMAN = "Guard Tobin"                  # the watchman who steps in at the theft (he exists for that moment only)
WATCH_DELAY = 3.0                         # seconds he stands there shouting before he lays hands on the thief
WATCH_DISTANCE = 2.2                      # paces away he appears (within a guard's reach)
DONE_SECONDS = 30.0                       # the closing message stays in the panel this long

HINTS = {
    Stage.INTRO: "Jae took you in off the street, and now he wants paying back. Listen to what he has to say, then leave "
                 "the hideout: look at the door and press [E] to open it. Barnaby's store is straight across the street to "
                 "the north.",
    Stage.TORCH: "It is night. The law says that anyone out after dark must carry a lit torch, and any guard who sees you "
                 "without one will stop you. Press [1] to light your torch. A lit torch also makes you easy to see, so put it "
                 "out again whenever you need to stay hidden. Light it now.",
    Stage.TO_THE_SHOP: "Cross the street to Barnaby's store, the door under the sign with the scales. The door is locked. "
                       "Crouch with [Ctrl], look at the door and press [E] to pick the lock. Turn the pick with [LEFT] and "
                       "[RIGHT] until the ring glows, then press [SPACE]. If you miss, the pick snaps, but you will be given "
                       "another. Stay out of sight of the guards while you work.",
    Stage.IN_THE_SHOP: "You are in. The shelf on the west wall holds the candlestick. Look at the shelf and press [E] to "
                       "search it. Everything on it belongs to Barnaby, so every row is marked THEFT. Select the Silver "
                       "Candlestick and press [ENTER] to take it.",
    Stage.CAUGHT: "You have been seen. The guard adds up your crimes and the stolen goods you are carrying, and if you "
                  "cannot pay the fine you go to the cells. The candlestick goes back to Barnaby, and everything you own "
                  "except one lockpick is locked in the jail's confiscation lockbox.",
    Stage.JAIL: "This is the cell. Press [T] and choose to wait out your sentence. A first offence costs you one night. You "
                "could also crouch and pick the bars with your last lockpick, but breaking out is a crime of its own and "
                "your belongings would stay locked up. Waiting is wiser.",
    Stage.LOCKBOX: "You are free. Your belongings are in the confiscation lockbox in the jail office, against the west wall. "
                   "Look at it and press [E] to open it, then press [X] to take everything back. Leave the jail when you are "
                   "done; the jailer is waiting for you to go.",
    Stage.DRESS: "Press [Tab] to open your equipment. Your pack is on the left and your figure stands in the middle. Select "
                 "the Soft Leather Boots with the arrow keys and press [ENTER] to put them on. The SEEN and HEARD numbers on "
                 "the right show how easily people notice you: metal shines and rattles, while dark cloth keeps you hidden "
                 "and quiet. Press [Tab] again to close the view.",
    Stage.WAIT_FOR_NIGHT: "Thieves work at night, when the streets are empty. Press [T] to wait an hour, and keep pressing "
                          "it until the clock passes 20:00. Light your torch again once it is dark.",
    Stage.STEAL_AGAIN: "Go back to Barnaby's store and take the candlestick properly this time. Pick the lock if the door is "
                       "locked, crouch to move quietly, and keep out of sight. Put your torch out once you are inside. Take "
                       "only the candlestick, then bring it to Jae.",
    Stage.SELL: "You have the candlestick. Go back to the hideout, look at Jae and press [E] to trade. Press [TAB] to move "
                "to your own column, select the candlestick and press [ENTER] to sell it. A fence pays very little for "
                "stolen goods.",
    Stage.TOOLS: "Jae has paid you sixty gold. Lockpicks are forged from iron ore, so you will need a pickaxe and a mallet. "
                 "Barnaby sells both from the east shelf in his store, and the smith sells mallets too. Buy them, or steal "
                 "them.",
    Stage.MINE: "Take the pickaxe to the ore deposit: the rocks behind the smithy's open shed, over the bridge at the west "
                "end of the street. Look at the rocks and press [E] to mine iron ore.",
    Stage.FORGE: "The anvil stands under the smith's open shed. With ore and the mallet in your pack, look at the anvil and "
                 "press [E] to forge lockpicks. A beginner ruins some ore and gets a single pick when it works, and the "
                 "Smithing skill improves with every attempt.",
    Stage.INN: "You need somewhere to sleep. The Drowsy Dragon, the big inn on the north side of the street just west of "
               "Barnaby's store, lets rooms for ten gold a night. Go inside and the innkeeper will offer you one. In a room of "
               "your own you can sleep until morning or until nightfall.",
    Stage.DONE: "",
}
DONE_MESSAGE = ("That is everything you need to know. The whole town is open to you now: every door, every pocket and every "
                "deal, and every consequence - and the town gate stands open, to the village, the Thornwood and the old keep "
                "beyond. Talk to Jae: he may have more work for you. Keep your torch lit after dark, keep your steps quiet, and "
                "do not get caught.")
GATE_CLOSED_TEXT = "The town gate is barred. Nobody leaves the town until Jae's business is done."

JAE_INTRO = (
    (1.0, "Awake, are you? I found you in the gutter, half-starved. I fed you and gave you a roof."),
    (6.0, "Nothing is free. You will pay me back. Barnaby's store is across the street, and there is a silver candlestick on his shelf."),
    (11.0, "Take these picks. The door is locked, so crouch and work the lock. Bring me that candlestick. Go on."),
)
JAE_PAID = (
    (0.5, "Ha! There's the lad. It is worth next to nothing, but here: sixty gold, and we will call it even for now."),
    (6.0, "A thief needs lockpicks, and lockpicks are forged from ore. Barnaby sells a pickaxe and a mallet. The ore is behind the smithy, and the smith's anvil turns it into picks."),
)
JAE_NOT_YET = "Bring me the candlestick first. Then we will talk."
JAE_ANOTHER_PICK = "Jae slipped you another lockpick. Try not to snap this one."
WATCHMAN_LINE = "I thought I saw you sneaking about!"
PITY_LINE = "A first offence, and a hungry one by the look of you. One night in the cells, and we will say no more."
REFUSALS = {
    "door": "You cannot go through that door yet. Jae wants the candlestick from Barnaby's store first.",
    "container": "Leave that alone. Jae wants the candlestick from the shelf in Barnaby's store.",
    "item": "Take only the candlestick. Nothing else.",
    "loot_all": "Take only the candlestick. Nothing else leaves that shelf.",
    "trade": "Nobody will deal with you until you have settled with Jae.",
    "pickpocket": "Not now. Bring Jae the candlestick first.",
    "station": "Not yet. Bring Jae the candlestick first.",
}


@dataclass
class Tutorial:
    """The tutorial's state: the stage, the lines still to be spoken, the odd flag."""
    game: "Game"
    stage: Stage = Stage.INTRO
    hint: str = ""
    lines: List[Tuple[float, str, str]] = field(default_factory=list)   # (at time, speaker, text)
    clock: float = 0.0
    tab_opened: bool = False
    waited: bool = False
    sentence_trimmed: bool = False
    picks_given: int = 0
    caught_guard: Optional[str] = None
    stage_clock: float = 0.0           # seconds in the current stage
    TORCH_LESSON_SECONDS = 8.0         # by day the torch law is explained, not enforced, for this long
    LOG_LEAD_CHARS = 40                # the log gets the hint's opening sentences, at least this much of them

    # ------------------------------------------------------------------ setting up
    def begin(self) -> None:
        """A new game: night in the hideout, an empty purse, three picks, boots in the pack."""
        g = self.game
        w, p = g.world, g.player
        w.clock.hour = START_HOUR
        w.clock.minute_accumulator = 0.0
        w.resync_schedules(announce=False, teleport=True)
        w.suppress_curfews = True                                  # Jae has taken you in: no "time to go" tonight
        p.position = w.spot(*START_SPOT)
        p.rotation_y, p.rotation_x = 270.0, 0.0
        p.gold = 0.0
        p.inventory = [i for i in p.inventory if i.id != "silver_goblet"]     # an urchin owns nothing hot yet
        for item in p.inventory:
            item.equipped = False                                  # barefoot: the boots are a lesson
            if item.is_lockpick:
                item.quantity = 3
        jae = w.find_npc(FENCE)
        if jae is not None:
            jae.turn_towards(p.position)
        # The town's guards stand where they are, seeing and hearing nothing, until the tutorial
        # is over: the lessons are in locks and shadows, not in being chased. The catch is scripted.
        w.freeze_guards()
        self._say_later(JAE_INTRO)
        self._enter(Stage.INTRO)

    def _enter(self, stage: Stage) -> None:
        self.stage = stage
        self.stage_clock = 0.0
        self.hint = HINTS[stage]
        if self.hint:
            lead = ""
            for sentence in self.hint.split(". "):                # the opening of the hint goes in the log too
                lead = f"{lead} {sentence.rstrip('.')}.".strip()
                if len(lead) >= self.LOG_LEAD_CHARS:
                    break
            self.game.messages.add(f"TUTORIAL: {lead}")

    def _say_later(self, script) -> None:
        for delay, text in script:
            self.lines.append((self.clock + delay, FENCE, text))

    # ------------------------------------------------------------------ queries
    @property
    def active(self) -> bool:
        return self.stage is not Stage.DONE

    @property
    def locked(self) -> bool:
        """Everything but the job is shut until the candlestick is sold."""
        return self.stage in LOCKED_STAGES

    def refuses(self, kind: str, obj=None) -> Optional[str]:
        """Why the player may not do ``kind`` to ``obj`` right now (None: go ahead). The town gate
        stays barred until the very end: a guard at it turns the player away."""
        if kind == "door" and getattr(obj, "gate", None) and self.active:
            return self._turned_away_at_the_gate()
        if not self.locked:
            return None
        if kind == "door":
            if obj.door_id in (TARGET_DOOR, "hideout_door", "jail_door", "cell_door"):
                return None
            return REFUSALS["door"]
        if kind == "container":
            if obj.name in (TARGET_SHELF, "jail_storage"):
                return None
            return REFUSALS["container"]
        if kind == "item":                       # taking something from a container
            container, item = obj
            if container.name == "jail_storage" or item.id == TARGET_ITEM or item.is_players:
                return None
            return REFUSALS["item"]
        if kind == "loot_all":
            return None if obj.name == "jail_storage" else REFUSALS["loot_all"]
        if kind == "trade":
            if obj.name == FENCE:
                return None if self.stage is Stage.SELL else JAE_NOT_YET
            return REFUSALS["trade"]
        if kind == "trade_item":                 # a unit moving in Jae's deal
            return None if obj.id == TARGET_ITEM else JAE_NOT_YET
        if kind == "pickpocket":
            return REFUSALS["pickpocket"]
        if kind == "station":
            return REFUSALS["station"]
        return None

    # ------------------------------------------------------------------ per frame
    def update(self, delta_time: float) -> None:
        g = self.game
        w, p = g.world, g.player
        self.clock += delta_time
        self.stage_clock += delta_time
        self._speak()
        if self.stage is Stage.DONE:
            if self.hint and self.stage_clock >= DONE_SECONDS:
                self.hint = ""                                    # the closing words have been up long enough
            return
        jail = w.jail
        if jail.serving and not self.sentence_trimmed:
            self._trim_sentence()
        if self.stage not in (Stage.CAUGHT, Stage.JAIL):
            self._keep_a_pick()                                   # never stuck for want of a pick
        if self.stage in (Stage.INTRO, Stage.TORCH, Stage.TO_THE_SHOP, Stage.IN_THE_SHOP) and jail.serving:
            self._enter(Stage.JAIL)                               # caught at the lock, or on the way: the lesson comes early
            return
        dark = DayNightCycle.is_torch_time(w.clock.fractional_hour)
        if self.stage is Stage.INTRO:
            if not self.lines or w.area_of(p.position) != "the hideout":
                self._enter(Stage.TORCH)
        elif self.stage in (Stage.TORCH, Stage.TO_THE_SHOP, Stage.IN_THE_SHOP) and self._has(TARGET_ITEM):
            self._the_watch_arrives()                             # (however they got there)
            self._enter(Stage.CAUGHT)
        elif self.stage in (Stage.TORCH, Stage.TO_THE_SHOP):
            door = w.find_door(TARGET_DOOR)
            building = w.building_at(p.position)
            if (door is not None and not door.is_locked) or (building is not None and building.name == "shop"):
                self._enter(Stage.IN_THE_SHOP)                    # the lesson was skipped: in the store is in the store
            elif self.stage is Stage.TORCH and (p.has_lit_torch or (not dark and self.stage_clock >= self.TORCH_LESSON_SECONDS)):
                self._enter(Stage.TO_THE_SHOP)
        elif self.stage is Stage.CAUGHT:
            if jail.serving:
                self._dismiss_the_watch()
                self._enter(Stage.JAIL)
            else:
                self._keep_the_watch_on_them()
        elif self.stage is Stage.JAIL:
            if jail.released or jail.escaped or not jail.serving:
                self._enter(Stage.LOCKBOX)
        elif self.stage is Stage.LOCKBOX:
            if self._has("soft_shoes") or w.area_of(p.position) not in (jail.cell_area, jail.jail_area):
                self._enter(Stage.DRESS)
        elif self.stage is Stage.DRESS:
            boots = next((i for i in p.inventory if i.id == "soft_shoes"), None)
            if (boots is not None and boots.equipped) or (boots is None and self.tab_opened):
                self.waited = False
                self._enter(Stage.WAIT_FOR_NIGHT)
        elif self.stage is Stage.WAIT_FOR_NIGHT:
            if DayNightCycle.is_night(w.clock.hour) and (self.waited or self.stage_clock >= self.TORCH_LESSON_SECONDS):
                self._enter(Stage.STEAL_AGAIN)
        elif self.stage is Stage.STEAL_AGAIN:
            if self._has(TARGET_ITEM):
                self._enter(Stage.SELL)
        elif self.stage is Stage.SELL:
            jae = w.find_npc(FENCE)
            if jae is not None and any(i.id == TARGET_ITEM for i in jae.inventory) and not self._has(TARGET_ITEM):
                p.gold += KINDNESS_GOLD
                self._say_later(JAE_PAID)
                self._enter(Stage.TOOLS)
            elif not self._has(TARGET_ITEM):
                self._enter(Stage.STEAL_AGAIN)                  # lost it again (confiscated): once more
        elif self.stage is Stage.TOOLS:
            if self._has("pickaxe") and self._has("mallet"):
                self._enter(Stage.MINE)
        elif self.stage is Stage.MINE:
            if self._has("iron_ore"):
                self._enter(Stage.FORGE)
        elif self.stage is Stage.FORGE:
            anvil = w.find_station("anvil")
            if anvil is not None and anvil.uses > 0 and p.has_lockpick():
                self._enter(Stage.INN)
        elif self.stage is Stage.INN:
            inn = w.inn_building()
            if (inn is not None and w.building_at(p.position) is inn) or w.inn.is_rented(w.day_count, w.clock.hour):
                self._finish()

    def _speak(self) -> None:
        due = [line for line in self.lines if line[0] <= self.clock]
        for line in due:
            self.lines.remove(line)
            npc = self.game.world.find_npc(line[1])
            if npc is not None:
                npc.turn_towards(self.game.player.position)
                npc.say(line[2], self.game.messages.add)

    def _has(self, item_id: str) -> bool:
        return any(i.id == item_id for i in self.game.player.inventory)

    def _keep_a_pick(self) -> None:
        """Out of lockpicks, outside the cells: Jae slips you another. The tutorial is never lost
        for want of a pick - keep trying until it works."""
        g = self.game
        if g.player.has_lockpick() or g.in_lockpicking or g.world.jail.serving or g.in_transition:
            return
        g.player.add_item(Item("iron_lockpick", "Iron Lockpick", value=5.0, is_lockpick=True, stackable=True,
                               category="tool", owner=Item.PLAYER_OWNER))
        self.picks_given += 1
        g.messages.add(JAE_ANOTHER_PICK)

    # --- the scripted catch: a watchman steps in, and steps out of the world again afterwards
    def _the_watch_arrives(self) -> None:
        """A watchman of the town's guard steps in a couple of paces from the thief - "I thought
        I saw you sneaking about!" - the shelf screen closes and the thief turns to face him; he
        stands there shouting for ``WATCH_DELAY`` seconds, in plain view, then lays hands on them
        and the town's own justice does the rest (the bill, the cells). He exists for this moment
        only (``_dismiss_the_watch``) and is the one guard the tutorial's freeze does not hold."""
        from .entities import NPC
        g = self.game
        w, p = g.world, g.player
        g.close_screens()
        guard = NPC(WATCHMAN, True, False, False, self._spot_for_the_watch(),
                    settings=w.settings.perception, npc_settings=w.settings.npc)
        guard.rng = w.phrase_rng
        guard.role_tag = "guard"
        guard.state = AIState.ALERT_GUARDS
        guard.time = w.elapsed
        guard.has_torch = True
        guard.last_scold = {"trespass": 0.0, "torch": 0.0, "skulking": 0.0}     # he has one thing to say: the theft
        guard.turn_towards(p.position)
        w.npcs.append(guard)
        w.unfrozen_guards.add(guard.name)
        self.caught_guard = guard.name
        guard.crimes_witnessed += 1
        guard.raise_alarm(g.messages.add, guard_line=WATCHMAN_LINE, log_prefix=f"{guard.name} witnessed Shoplifting!")
        w.declare_crime("Shoplifting", p.position.copy(), witnessed=True, reporter=guard.name, witnesses=[guard])
        guard.charge_ready_at = w.elapsed + WATCH_DELAY             # a moment in view first ...
        guard.repath_timer = WATCH_DELAY                            # ... standing still, not closing in
        p.hold_attention(guard.position, WATCH_DELAY + w.settings.justice.caught_freeze_seconds)

    def _spot_for_the_watch(self) -> Vector3D:
        """Where the watchman appears: ``WATCH_DISTANCE`` from the thief, on the open floor of the
        same room, with nothing between them - towards the store's door for choice (behind the
        thief at the shelf: they turn round to find him there), otherwise wherever is clear."""
        from .entities import NPC
        from .world.world import NPC_HEIGHT
        w, p = self.game.world, self.game.player
        door = w.find_door(TARGET_DOOR)
        towards = Vector3D(door.center.x, 0.0, door.center.z) if door is not None \
            else Vector3D(p.position.x, 0.0, p.position.z + 1.0)
        base = math.degrees(math.atan2(towards.x - p.position.x, towards.z - p.position.z))
        room = w.building_at(p.position)
        blockers = w.sight_blockers()
        colliders = w.colliders(include_npcs=True)
        eye = Vector3D(p.position.x, p.position.y + 1.0, p.position.z)
        for offset in (0.0, 40.0, -40.0, 80.0, -80.0, 120.0, -120.0, 180.0):
            angle = math.radians(base + offset)
            spot = Vector3D(p.position.x + math.sin(angle) * WATCH_DISTANCE, p.position.y,
                            p.position.z + math.cos(angle) * WATCH_DISTANCE)
            if w.building_at(spot) is not room:
                continue
            body = BoundingBox3D.around(spot, NPC.BODY_RADIUS, NPC_HEIGHT, NPC.BODY_RADIUS)
            if any(body.intersects(c) for c in colliders):
                continue
            if any(ray_intersects_box(Vector3D(spot.x, spot.y + 1.0, spot.z), eye, b) for b in blockers):
                continue
            return spot
        return Vector3D(p.position.x + 1.6, p.position.y, p.position.z)      # at their elbow, whatever is there

    def _keep_the_watch_on_them(self) -> None:
        """Should the thief somehow slip the watchman's grasp, he is at their elbow again. A save
        loaded mid-arrest in a fresh game has no Guard Tobin in it (he is not part of the town):
        he is summoned afresh."""
        from .systems import justice
        w, p = self.game.world, self.game.player
        if self.game.in_transition:
            return
        guard = w.find_npc(self.caught_guard) if self.caught_guard else None
        if guard is None:
            self._the_watch_arrives()
            return
        w.unfrozen_guards.add(guard.name)
        guard.turn_towards(p.position)
        if guard.position.distance_to(p.position) > w.settings.justice.charge_reach:
            guard.path, guard.goal = [], None
            guard.position = self._spot_for_the_watch()
            guard.turn_towards(p.position)
            guard.state = AIState.ALERT_GUARDS
            guard.charge_ready_at = w.elapsed
            p.hold_attention(guard.position, w.settings.justice.caught_freeze_seconds)
        elif w.elapsed >= guard.charge_ready_at and not w.jail.serving and not w.relocating:
            justice.charge_player(w, guard, p, w.notify)          # whatever he could or could not see: he has them

    def _dismiss_the_watch(self) -> None:
        """The watchman who made the arrest goes back to wherever he came from: out of the world,
        leaving nothing behind - no memories to charge, no conversation, no post to return to."""
        w = self.game.world
        guard = w.find_npc(self.caught_guard) if self.caught_guard else None
        if guard is None:
            return
        guard.end_conversation()
        guard.forget_crimes()
        w.npcs = [n for n in w.npcs if n is not guard]
        w.unfrozen_guards.discard(guard.name)
        self.caught_guard = None

    def _trim_sentence(self) -> None:
        """A first offender: the captain takes pity - one night, not a week."""
        w = self.game.world
        self.sentence_trimmed = True
        if w.jail.sentence_end_day is not None and w.jail.sentence_end_day - w.day_count > PITY_SENTENCE_DAYS:
            w.jail.sentence_end_day = w.day_count + PITY_SENTENCE_DAYS
            self.game.messages.add(f"(Guard Captain Vance): \"{PITY_LINE}\"")

    def note_wait(self) -> None:
        self.waited = True

    GATE_WARDEN_REACH = 16.0        # a gate guard this close to the player speaks for the gate

    def _turned_away_at_the_gate(self) -> str:
        """The gate is barred until the tutorial is done; a guard at it says so (face to face)."""
        from .core import dialogue
        w, p = self.game.world, self.game.player
        guards = [g for g in w.guards() if not g.is_asleep and g.position.distance_to(p.position) <= self.GATE_WARDEN_REACH]
        if not guards:
            return GATE_CLOSED_TEXT
        guard = min(guards, key=lambda g: g.position.distance_to(p.position))
        guard.turn_towards(p.position)
        text = dialogue.line("gate_refusal", w.phrase_rng)
        guard.say(text)
        return f"({guard.name}): \"{text}\" {GATE_CLOSED_TEXT}"

    def _finish(self) -> None:
        """The town opens up: curfews apply, the guards walk their rounds again, and the closing
        words stay in the panel for ``DONE_SECONDS`` (``update`` takes them down)."""
        w = self.game.world
        w.suppress_curfews = False
        w.unfreeze_guards()
        if w.level.town_gate:
            w.open_gate(w.level.town_gate)                       # the town lets the player go
        self._enter(Stage.DONE)
        self.hint = DONE_MESSAGE
        self.game.messages.add(f"TUTORIAL COMPLETE. {DONE_MESSAGE}")

    # ------------------------------------------------------------------ the waypoint
    @property
    def waypoint(self) -> Optional[Vector3D]:
        """Where the tutorial wants the player to go next (None when the next step is a key, not
        a place): the hideout's door, the store's door and then its west shelf, the lockbox, Jae,
        the east shelf, the ore, the anvil, the inn. Routed through doors: from inside one
        building towards somewhere else the waypoint is that building's door; from the street
        towards a place indoors it is the door of the building it is in."""
        target = self._destination()
        return self._route(target) if target is not None else None

    def _destination(self) -> Optional[Vector3D]:
        w, p = self.game.world, self.game.player
        stage = self.stage

        def door(door_id):
            d = w.find_door(door_id)
            return d.center.copy() if d is not None else None

        def box(name, finder):
            b = finder(name)
            return Vector3D(b.center.x, 0.0, b.center.z) if b is not None else None

        def npc(name):
            n = w.find_npc(name)
            return n.position.copy() if n is not None else None

        shop_door = door(TARGET_DOOR)
        here = w.building_at(p.position)
        in_the_shop = here is not None and here.name == "shop"
        if stage in (Stage.INTRO, Stage.TORCH, Stage.TO_THE_SHOP, Stage.WAIT_FOR_NIGHT):
            return shop_door
        if stage in (Stage.IN_THE_SHOP, Stage.STEAL_AGAIN):
            return box(TARGET_SHELF, w.find_container) if in_the_shop else shop_door
        if stage is Stage.LOCKBOX:
            return box("jail_storage", w.find_container)
        if stage is Stage.SELL:
            return npc(FENCE) if w.area_of(p.position) == "the hideout" else door("hideout_door")
        if stage is Stage.TOOLS:
            return box("shop_shelf_east", w.find_container) if in_the_shop else shop_door
        if stage is Stage.MINE:
            return box("ore_deposit", w.find_station)
        if stage is Stage.FORGE:
            return box("anvil", w.find_station)
        if stage is Stage.INN:
            inn = w.inn_building()
            d = w.door_of(inn) if inn is not None else None
            return d.center.copy() if d is not None else None
        return None                                               # CAUGHT, JAIL, DRESS, DONE: nowhere to go yet

    def _route(self, target: Vector3D) -> Vector3D:
        w, p = self.game.world, self.game.player
        here, there = w.building_at(p.position), w.building_at(target)
        if here is there:
            return target
        building = here if here is not None else there
        d = w.door_of(building)
        if d is None:
            return target
        return d.center.copy()

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"stage": self.stage.value, "tab_opened": self.tab_opened, "waited": self.waited,
                "sentence_trimmed": self.sentence_trimmed, "picks_given": self.picks_given, "caught_guard": self.caught_guard}

    @classmethod
    def from_dict(cls, game: "Game", data: dict) -> "Tutorial":
        t = cls(game)
        value = data.get("stage", Stage.INTRO.value)
        value = {"wait": Stage.INN.value}.get(value, value)      # an earlier draft's last stage
        t.stage = Stage(value) if value in Stage._value2member_map_ else Stage.INTRO
        t.hint = HINTS[t.stage]
        t.tab_opened = data.get("tab_opened", False)
        t.waited = data.get("waited", False)
        t.sentence_trimmed = data.get("sentence_trimmed", False)
        t.picks_given = data.get("picks_given", 0)
        t.caught_guard = data.get("caught_guard")
        if t.stage is Stage.DONE:
            t.hint = ""                                            # the closing words were for the moment
        game.world.suppress_curfews = t.locked
        if t.active:
            game.world.freeze_guards(except_for=[t.caught_guard] if t.caught_guard else ())
        else:
            game.world.unfreeze_guards()
        return t
