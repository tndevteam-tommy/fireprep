"""Quests: work people put the player's way once the tutorial is over.

A :class:`QuestDef` is plain data - who gives it, what is to be stolen from whom and where it
is kept, what it pays. :class:`QuestLog` follows each one through its stages:

* ``available`` - the giver has something to say (the tutorial must be over);
* ``active``    - taken on: the HUD shows the objective and the waypoint points the way;
* ``done``      - paid.

The player *talks* to the giver ([E] on them) to be offered the job and to hand it in; a giver
who is also a merchant still trades the rest of the time. Handing the piece in pays the reward
and - for Jae's job - the piece is the player's to keep, honestly their own from then on. The
log is saved with the game; nothing here draws.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .core.math3d import Vector3D
from .entities import Item

if TYPE_CHECKING:  # pragma: no cover
    from .game import Game

AVAILABLE, ACTIVE, DONE = "available", "active", "done"


@dataclass(frozen=True)
class QuestDef:
    id: str
    title: str
    giver: str
    target_item: str                    # item id to bring back
    target_name: str
    target_owner: str                   # whom it is stolen from
    target_container: str               # where it is kept (for the waypoint)
    target_door: str                    # the way into the place it is kept
    target_building: str
    giver_door: str                     # the way in to the giver
    giver_area: str
    reward_gold: float
    offer: Tuple[Tuple[float, str], ...]
    reminder: str
    thanks: Tuple[Tuple[float, str], ...]
    objective: str
    return_objective: str
    player_keeps: bool = True           # the piece is the player's own once handed in


STICKING_IT_TO_THEM = QuestDef(
    id="sticking_it_to_them",
    title="Sticking It to the Jerks",
    giver="Jae the Fence",
    target_item="ravenfeather_cloak",
    target_name="Ravenfeather Cloak",
    target_owner="Lord Aurelian Vane",
    target_container="vane_wardrobe",
    target_door="vane_door",
    target_building="vane",
    giver_door="hideout_door",
    giver_area="the hideout",
    reward_gold=150.0,
    offer=(
        (0.0, "You've got hands on you, kid. I've a job - one I'd do for nothing, if my knees still bent the right way."),
        (5.5, "Lord Vane. Up the hill, the big house on Upper Street. Looks at me like I'm something on his boot."),
        (11.0, "He's got a cloak of raven feathers he shows off at every chance. Keeps it in a wardrobe chest upstairs. "
               "Bring it to me."),
        (17.0, "Mind yourself - a rich man's door has a rich man's lock, and the chest is locked too. Go at night, when "
               "he's snoring. He keeps his keys on him."),
    ),
    reminder="The ravenfeather cloak, kid. Lord Vane's. Wardrobe chest, upstairs. Go on.",
    thanks=(
        (0.0, "Ha! Look at that. Feathers and all. I'd give a week's takings to see his face in the morning."),
        (5.0, "Here - a hundred and fifty for your trouble. And keep the cloak. Wear it under his nose; just for sticking "
              "it to those jerks."),
    ),
    objective="Steal Lord Vane's Ravenfeather Cloak from the wardrobe chest upstairs in Vane Manor, on Upper Street up the "
              "hill, and bring it to Jae in the hideout. The manor door has a Master lock and the chest a Strong one: "
              "go at night, while Lord Vane sleeps upstairs (he carries the keys), or pick them.",
    return_objective="You have the Ravenfeather Cloak. Take it back to Jae in the hideout and talk to him ([E]).",
)

QUESTS: Tuple[QuestDef, ...] = (STICKING_IT_TO_THEM,)


@dataclass
class QuestLog:
    game: "Game"
    stages: Dict[str, str] = field(default_factory=dict)
    lines: List[Tuple[float, str, str]] = field(default_factory=list)     # (at time, speaker, text) still to be said
    clock: float = 0.0

    def __post_init__(self):
        for q in QUESTS:
            self.stages.setdefault(q.id, AVAILABLE)

    # ------------------------------------------------------------------ queries
    def stage(self, quest: QuestDef) -> str:
        return self.stages.get(quest.id, AVAILABLE)

    def _open(self) -> bool:
        """Quests wait until the tutorial is over."""
        return not self.game.in_tutorial

    @property
    def active(self) -> Optional[QuestDef]:
        if not self._open():
            return None
        return next((q for q in QUESTS if self.stage(q) == ACTIVE), None)

    def _carried(self, quest: QuestDef) -> Optional[Item]:
        return next((i for i in self.game.player.inventory if i.id == quest.target_item), None)

    @property
    def hint(self) -> str:
        quest = self.active
        if quest is None:
            return ""
        return quest.return_objective if self._carried(quest) is not None else quest.objective

    @property
    def title(self) -> str:
        quest = self.active
        return quest.title if quest is not None else ""

    @property
    def waypoint(self) -> Optional[Vector3D]:
        """The wardrobe chest (from inside the manor) or the manor's door; once the piece is in
        hand, Jae (from inside the hideout) or the hideout's door."""
        quest = self.active
        if quest is None:
            return None
        w, p = self.game.world, self.game.player
        here = w.building_at(p.position)
        if self._carried(quest) is None:
            if here is not None and here.name == quest.target_building:
                chest = w.find_container(quest.target_container)
                return chest.center.copy() if chest is not None else None
            door = w.find_door(quest.target_door)
            return door.center.copy() if door is not None else None
        if w.area_of(p.position) == quest.giver_area:
            giver = w.find_npc(quest.giver)
            return giver.position.copy() if giver is not None else None
        door = w.find_door(quest.giver_door)
        return door.center.copy() if door is not None else None

    # ------------------------------------------------------------------ talking
    def talk(self, npc) -> bool:
        """The player speaks to ``npc``: True if a quest took the conversation (handing the piece
        in) - otherwise the usual [E] goes ahead (an offer is made in passing: Jae talks business
        while the trade is open)."""
        if not self._open():
            return False
        for quest in QUESTS:
            if npc.name != quest.giver:
                continue
            stage = self.stage(quest)
            if stage == AVAILABLE:
                self.stages[quest.id] = ACTIVE
                self._say(npc, quest.offer)
                self.game.messages.add(f"QUEST: {quest.title}. {quest.objective}")
                return False
            if stage == ACTIVE:
                item = self._carried(quest)
                if item is not None:
                    self._complete(quest, npc, item)
                    return True
                npc.say(quest.reminder, self.game.messages.add)
        return False

    def _complete(self, quest: QuestDef, npc, item: Optional[Item]) -> None:
        p = self.game.player
        self.stages[quest.id] = DONE
        if item is not None and quest.player_keeps:
            item.transfer_to(Item.PLAYER_OWNER, launder=True)          # honestly the player's now
        p.gold += quest.reward_gold
        self._say(npc, quest.thanks)
        kept = f" The {quest.target_name} is yours to keep." if item is not None and quest.player_keeps else ""
        self.game.messages.add(f"QUEST COMPLETE: {quest.title}. Jae paid you {quest.reward_gold:.0f} gold.{kept}")

    def _say(self, npc, script) -> None:
        first, *rest = script
        npc.turn_towards(self.game.player.position)
        npc.say(first[1], self.game.messages.add)
        for delay, text in rest:
            self.lines.append((self.clock + delay, npc.name, text))

    # ------------------------------------------------------------------ per frame
    def update(self, delta_time: float) -> None:
        self.clock += delta_time
        due = [line for line in self.lines if line[0] <= self.clock]
        for line in due:
            self.lines.remove(line)
            npc = self.game.world.find_npc(line[1])
            if npc is not None:
                npc.say(line[2], self.game.messages.add)
        # the piece sold to the giver over the counter instead of handed in: the job is done all the same
        quest = self.active
        if quest is not None:
            giver = self.game.world.find_npc(quest.giver)
            if giver is not None and any(i.id == quest.target_item for i in giver.inventory) and self._carried(quest) is None:
                self._complete(quest, giver, None)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"stages": dict(self.stages)}

    def load_dict(self, data: Optional[dict]) -> None:
        self.lines = []
        self.stages = {q.id: AVAILABLE for q in QUESTS}
        for qid, stage in ((data or {}).get("stages") or {}).items():
            if qid in self.stages and stage in (AVAILABLE, ACTIVE, DONE):
                self.stages[qid] = stage
