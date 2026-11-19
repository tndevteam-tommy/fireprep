"""Quests: what somebody asked the netrunner to do, how far they have got, and who to
talk to about it.

* ``QuestDef`` - the quest as written: title, who gave it, where, a summary and the full
  text, pictures of what it is about (``Picture``: a named view of the world - the quest
  screen renders it from the level itself) and the reward. ``QUESTS`` is the catalogue.
* ``QuestLog`` - the player's quests: status, the flags that record what has happened,
  and a journal. Serialisable for the save files.
* ``QuestDirector`` - ties the quests to the game: the conversations people have with the
  netrunner ([E] in front of them: ``conversation_for``), and each frame what has changed
  (``update``: the lock released, inside the flat, the note read, the stick in hand, thrown
  out...). The current objective is worked out from the flags and the inventory
  (``objective``), so it moves back as well as forward - thrown out of the flat without
  the stick, the objective is to find it again.

The standard for a quest is the ``number16`` one: written details (``description`` and
the journal), the current objective, and pictures of the objective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cyberduke.core.dialogue import Conversation, Line
from cyberduke.core.geometry import Vec3
from cyberduke.core.inventory import item

ACTIVE = "active"
COMPLETE = "complete"


@dataclass(frozen=True)
class Picture:
    """A view of the world that illustrates a quest: rendered from ``eye`` looking at
    ``target`` in the level named ``level`` (``levels.LEVELS``)."""

    picture_id: str
    caption: str
    level: str
    eye: Vec3
    target: Vec3


@dataclass(frozen=True)
class QuestDef:
    quest_id: str
    title: str
    giver: str
    location: str
    summary: str
    description: tuple[str, ...]  # paragraphs
    pictures: tuple[Picture, ...] = ()
    reward: str = ""  # an item id


QUESTS: dict[str, QuestDef] = {}


def _register(q: QuestDef) -> QuestDef:
    QUESTS[q.quest_id] = q
    return q


NUMBER16 = _register(
    QuestDef(
        quest_id="number16",
        title="SOMETHING SPECIAL",
        giver="NIX",
        location="NIGHT MARKET ALLEY - NUMBER 16",
        summary="Steal the data stick from Dell's computer in number 16 for Nix.",
        description=(
            "Nix - the smoker who hangs about by the wall near the Halcyon door - has a job "
            "for somebody with a deck. The man in number 16, Dell, has been skimming off "
            "the wrong people, and keeps what he skimmed on a data stick plugged into his "
            "computer.",
            "Get into number 16, take the stick and bring it back to Nix. He says he will "
            "give you something special for it - something you will not find on any junk "
            "table.",
            "The front door has a cheap access panel beside it: ping it [F], and with it in "
            "your sights bypass its lock [1]. There is a camera inside - a cheap home job "
            "on its own little network - and Dell does not take kindly to strangers in his "
            "flat. He goes out every so often. Get caught in there and you will be thrown "
            "out, and anything of his you took comes out of your pockets.",
        ),
        pictures=(
            Picture(
                "door16",
                "NUMBER 16: THE DOOR AND ITS ACCESS PANEL",
                "night_market",
                eye=(-1.55, 1.75, 28.2),
                target=(-4.45, 1.25, 30.9),
            ),
            Picture(
                "datastick",
                "THE DATA STICK: BEDROOM NIGHTSTAND",
                "night_market",
                eye=(-9.45, 1.05, 31.6),
                target=(-10.48, 0.5, 30.72),
            ),
        ),
        reward="mod_wraith",
    )
)


# ============================================================================== the log
@dataclass
class QuestState:
    quest_id: str
    status: str = ACTIVE
    flags: set[str] = field(default_factory=set)
    journal: list[tuple[float, str]] = field(default_factory=list)
    started_at: float = 0.0

    @property
    def quest(self) -> QuestDef:
        return QUESTS[self.quest_id]

    @property
    def done(self) -> bool:
        return self.status == COMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "quest": self.quest_id,
            "status": self.status,
            "flags": sorted(self.flags),
            "journal": [[t, text] for t, text in self.journal],
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuestState:
        qid = str(d["quest"])
        if qid not in QUESTS:
            raise KeyError(f"unknown quest {qid!r}")
        return cls(
            qid,
            str(d.get("status", ACTIVE)),
            set(d.get("flags", ())),
            [(float(t), str(text)) for t, text in d.get("journal", ())],
            float(d.get("started_at", 0.0)),
        )


class QuestLog:
    """The player's quests, newest first."""

    def __init__(self) -> None:
        self.states: dict[str, QuestState] = {}
        self.order: list[str] = []

    def __contains__(self, quest_id: str) -> bool:
        return quest_id in self.states

    def get(self, quest_id: str) -> QuestState | None:
        return self.states.get(quest_id)

    @property
    def quests(self) -> list[QuestState]:
        return [self.states[q] for q in self.order]

    def start(self, quest_id: str, time: float = 0.0) -> QuestState:
        if quest_id in self.states:
            return self.states[quest_id]
        state = QuestState(quest_id, started_at=time)
        state.journal.append((time, QUESTS[quest_id].summary))
        self.states[quest_id] = state
        self.order.insert(0, quest_id)
        return state

    def note(self, quest_id: str, text: str, time: float) -> None:
        state = self.states.get(quest_id)
        if state is not None:
            state.journal.append((time, text))

    def flag(self, quest_id: str, name: str, time: float = 0.0, note: str = "") -> bool:
        """Sets a flag (True if it was not set); ``note`` goes in the journal the first
        time."""
        state = self.states.get(quest_id)
        if state is None or name in state.flags:
            return False
        state.flags.add(name)
        if note:
            state.journal.append((time, note))
        return True

    def unflag(self, quest_id: str, name: str) -> None:
        state = self.states.get(quest_id)
        if state is not None:
            state.flags.discard(name)

    def complete(self, quest_id: str, time: float, note: str = "") -> None:
        state = self.states.get(quest_id)
        if state is not None and not state.done:
            state.status = COMPLETE
            if note:
                state.journal.append((time, note))

    def to_dict(self) -> dict[str, Any]:
        return {"quests": [self.states[q].to_dict() for q in self.order]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QuestLog:
        log = cls()
        for qd in d.get("quests", ()):
            state = QuestState.from_dict(qd)
            log.states[state.quest_id] = state
            log.order.append(state.quest_id)
        return log


# ============================================================================== the director
NIX = "NIX"
DELL = "DELL"
# who people are in conversation (role -> name)
NAMES = {"smoker": NIX, "resident": DELL}


class QuestDirector:
    """Runs the quests against a live simulation (duck-typed: ``player``, ``quests``,
    ``level``, ``hacking``, ``pickups``, ``panels``, ``time``, ``say``...)."""

    def __init__(self, sim: Any) -> None:
        self.sim = sim

    # ------------------------------------------------------------------ conversations
    def can_talk(self, npc: Any) -> bool:
        return getattr(npc, "role", "") == "smoker"

    def conversation_for(self, npc: Any) -> Conversation | None:
        """What ``npc`` says when the netrunner talks to them (None: nothing to say)."""
        if getattr(npc, "role", "") != "smoker":
            return None
        sim = self.sim
        state = sim.quests.get(NUMBER16.quest_id)
        if state is None:
            lines = [
                Line(NIX, "Hey. You - the one who came out of the Halcyon door like a ghost."),
                Line(NIX, "Relax. I don't care who you are. You've got a deck on you. I can tell."),
                Line(NIX, "Number 16, across the alley. Dell. "
                          "He's been skimming off the wrong people."),
                Line(NIX, "There's a data stick plugged into his computer. I want it."),
                Line(NIX, "Bring me that stick and I'll give you something special. "
                          "Something you won't find on any junk table."),
                Line(NIX, "Door's got a cheap access panel. Ping it, crack it. "
                          "He goes out now and then - don't let him catch you inside."),
            ]  # fmt: skip
            return Conversation(lines, on_done=self._start_number16, partner=npc)
        if state.done:
            return Conversation(
                [Line(NIX, "We're square. Now get lost before somebody sees us together.")],
                partner=npc,
            )
        if sim.player.inventory.has("datastick"):
            lines = [
                Line(NIX, "Well, look at that. You actually got it."),
                Line(NIX, "Dell's going to have a very bad week. Not my problem."),
                Line(NIX, "Here. Militech. Mil-spec. Don't ask me where it came from."),
                Line(NIX, "It'll keep the cameras guessing a little longer. "
                          "Something special - like I said."),
            ]  # fmt: skip
            return Conversation(lines, on_done=self._finish_number16, partner=npc)
        if "read_note" in state.flags:
            line = "Not on the computer? Then he's stashed it. Bedroom, I'd bet. Go look."
        elif "inside" in state.flags:
            line = "You were in there and you came out empty-handed? Go back. The stick, choom."
        else:
            line = "Number 16. The stick. The panel by the door. Go."
        return Conversation([Line(NIX, line)], partner=npc)

    def _start_number16(self) -> None:
        sim = self.sim
        sim.quests.start(NUMBER16.quest_id, sim.time)
        sim.toast(f"NEW QUEST: {NUMBER16.title}   [TAB] THEN [ ] FOR QUESTS")
        sim.hacking.log.append(f"QUEST: {NUMBER16.title}")

    def _finish_number16(self) -> None:
        sim = self.sim
        inv = sim.player.inventory
        inv.remove("datastick")
        giver = sim.npc_with_role("smoker")
        reward = NUMBER16.reward
        if giver is not None and giver.inventory.has(reward):
            giver.inventory.remove(reward)
        inv.add(reward)
        sim.quests.complete(
            NUMBER16.quest_id, sim.time, f"Gave Nix the stick. He paid: {item(reward).name}."
        )
        sim.toast(f"QUEST COMPLETE: {NUMBER16.title}  -  RECEIVED {item(reward).name}")
        sim.hacking.log.append(f"RECEIVED: {item(reward).name}")

    # ------------------------------------------------------------------ per frame
    def update(self) -> None:
        sim = self.sim
        log = sim.quests
        state = log.get(NUMBER16.quest_id)
        if state is None or state.done:
            return
        t = sim.time
        door = sim.room.doors.get("door16")
        if (
            door is not None
            and door.unlocked
            and any(p.bypasses for p in sim.panels if p.door_id == "door16")
        ):
            log.flag(
                NUMBER16.quest_id, "lock_released", t, "Bypassed the lock on number 16's door."
            )
        area = sim.level.private_areas[0] if sim.level.private_areas else None
        if area is not None and area.contains(sim.player.x, sim.player.z):
            log.flag(NUMBER16.quest_id, "inside", t, "Got inside number 16.")
        session = sim.hacking.session
        if session is not None and getattr(session, "notes", ""):
            log.flag(
                NUMBER16.quest_id, "read_note", t,
                "Dell's computer: no stick in it. A note says he moved it to the bedroom - "
                "the nightstand.",
            )  # fmt: skip
        if sim.player.inventory.has("datastick"):
            log.flag(
                NUMBER16.quest_id, "found", t, "Found the data stick on the bedroom nightstand."
            )

    def on_evicted(self, lost: list[str]) -> None:
        """Thrown out of the flat: anything taken from it is gone again."""
        state = self.sim.quests.get(NUMBER16.quest_id)
        if state is None or state.done:
            return
        note = "Caught in number 16 and thrown out."
        if "datastick" in lost:
            note += " They took the stick back."
        self.sim.quests.note(NUMBER16.quest_id, note, self.sim.time)

    # ------------------------------------------------------------------ what to do next
    def objective(self, state: QuestState) -> str:
        if state.quest_id != NUMBER16.quest_id:
            return ""
        if state.done:
            return "Done. Nix paid up - the Wraith damper is in your inventory."
        if self.sim.player.inventory.has("datastick"):
            return "Take the data stick to Nix, by the wall near the Halcyon door."
        if "read_note" in state.flags:
            return "Find the data stick in Dell's bedroom - on the nightstand."
        if "inside" in state.flags:
            return "Find the data stick. Start with Dell's computer by the front door."
        if "lock_released" in state.flags:
            return "The lock is released: get inside number 16 (mind the camera - and Dell)."
        return "Ping the access panel beside number 16's door [F] and bypass its lock."
