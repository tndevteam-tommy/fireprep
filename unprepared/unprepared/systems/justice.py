"""Crime and punishment: what guards hold against the player, fines, the jail.

A guard that catches up with a *wanted* player charges them (``charge_player``):
every remembered crime carries a :class:`~unprepared.config.Penalty` fixed when it
was declared - a fine, days in the cells on top of paying it, and more days instead
if it cannot be paid; theft crimes are punished by *where* they happened (indoors
costs far more than the street). Every stolen item still in the player's pockets
adds its value to the bill and a day to the sentence (two if it cannot be paid for).
``judge`` works the whole bill out; the fines are paid as far as the purse goes,
stolen goods go back to their owners, and any days owed are served. The prisoner
keeps one lockpick; everything else - tagged as the player's own property - waits in
the jail's confiscation lockbox until the sentence is served (or they break out,
which is a crime of its own).

The jail itself is level data (:class:`~unprepared.world.level.JailDef`): the
barred cell door, the lockbox, the two areas, the cell spot and the jailer.
While a prisoner is inside the cell door is *sealed* - no NPC opens it, whatever
keys it carries - and the jailer never goes in to "check". On release (at
``release_hour`` of the last day) the bars and the lockbox are unlocked, the
jailer stands at the post until the ex-prisoner has left the jail house, and
the lockbox is locked again behind them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from ..config import JusticeSettings, Penalty
from ..core.math3d import Vector3D
from ..entities import AIState, Item, Player
from ..entities.inventory import add_item

if TYPE_CHECKING:  # pragma: no cover
    from ..entities import NPC
    from ..world.world import World

Notify = Optional[Callable[[str], None]]


def penalty_for(crime: str, indoors: bool, settings: JusticeSettings) -> Penalty:
    """The penalty a crime carries: theft crimes by where they happened, others by name."""
    if crime in settings.theft_crimes:
        return settings.indoors if indoors else settings.outdoors
    return dict(settings.penalties).get(crime, settings.default_penalty)


@dataclass
class Verdict:
    """The bill a guard presents: what for, what it costs, what can be paid, what is served."""
    charges: List[str] = field(default_factory=list)      # one label per distinct crime
    fine: float = 0.0                                     # the whole bill, stolen goods included
    paid: float = 0.0                                     # what the purse covers
    days: int = 0                                         # days in the cells
    stolen_items: int = 0                                 # stolen units carried when charged
    unpaid: bool = False                                  # something could not be paid for

    @property
    def nothing(self) -> bool:
        return not self.charges

    def describe(self) -> str:
        parts = list(self.charges)
        if self.stolen_items:
            parts.append(f"{self.stolen_items} stolen item{'s' if self.stolen_items != 1 else ''} on you")
        return ", ".join(parts)


def judge(world: "World", player: Player) -> Verdict:
    """Total up every active crime (the same crime remembered by several guards counts once)
    and the stolen goods the player carries, paying from the purse in order of severity:
    a crime that can be paid for costs its fine and ``jail_days``, one that cannot costs
    ``jail_days_unpaid``; every stolen item costs its value and a day, two when its value
    cannot be covered either. Nothing is deducted here - ``charge_player`` applies it."""
    s = world.settings.justice
    memories: Dict[tuple, object] = {}
    for guard in world.guards():
        for memory in guard.active_memories(world.day_count):
            memories.setdefault(memory.key, memory)
    verdict = Verdict()
    if not memories:
        return verdict
    purse = player.gold
    for memory in sorted(memories.values(), key=lambda m: (-m.fine, m.crime)):
        verdict.charges.append(memory.label)
        verdict.fine += memory.fine
        if purse >= memory.fine:
            purse -= memory.fine
            verdict.paid += memory.fine
            verdict.days += memory.jail_days
        else:
            verdict.unpaid = True
            verdict.days += memory.jail_days_unpaid
    for item in player.inventory:
        if not item.is_stolen:
            continue
        for _ in range(item.quantity):
            verdict.stolen_items += 1
            verdict.fine += item.value
            verdict.days += s.stolen_item_jail_days
            if purse >= item.value:
                purse -= item.value
                verdict.paid += item.value
            else:
                verdict.unpaid = True
                verdict.days += s.stolen_item_unpaid_jail_days
    return verdict


@dataclass
class JailState:
    """Live jail bookkeeping; the static part comes from the level's ``JailDef``."""
    cell_door_id: str = "cell_door"
    storage_name: str = "jail_storage"
    cell_area: str = "the jail cell"
    jail_area: str = "the jail"
    cell_spot: Optional[Vector3D] = None
    jailer: Optional[str] = None
    post: Optional[Vector3D] = None
    present: bool = True                        # False: this level has no jail at all
    sentence_end_day: Optional[int] = None      # None: not serving
    escaped: bool = False
    grace_until: float = 0.0                    # world.elapsed until which a released prisoner may linger
    released: bool = False                      # between release and the ex-prisoner leaving the jail house
    transfer_pending: bool = False              # the player is being marched to the cell (no saving until they are in it;
    #                                             never persisted: a loaded game is never mid-march)

    @classmethod
    def from_def(cls, jail_def) -> "JailState":
        if jail_def is None:
            return cls(cell_spot=Vector3D(0.0, 0.0, 0.0), present=False)
        return cls(cell_door_id=jail_def.cell_door_id, storage_name=jail_def.storage, cell_area=jail_def.cell_area,
                   jail_area=jail_def.jail_area, cell_spot=Vector3D(*jail_def.cell_spot), jailer=jail_def.jailer,
                   post=Vector3D(*jail_def.post) if jail_def.post else None)

    @property
    def serving(self) -> bool:
        return self.sentence_end_day is not None

    @property
    def sealed(self) -> bool:
        """A prisoner is inside: the cell stays shut to everyone."""
        return self.serving and not self.escaped

    def days_left(self, day_count: int) -> int:
        return max(0, (self.sentence_end_day or day_count) - day_count)

    def to_dict(self) -> dict:
        return {"sentence_end_day": self.sentence_end_day, "escaped": self.escaped, "grace_until": self.grace_until,
                "released": self.released}

    def load_dict(self, data: dict) -> None:
        self.sentence_end_day = data.get("sentence_end_day")
        self.escaped = data.get("escaped", False)
        self.grace_until = data.get("grace_until", 0.0)
        self.released = data.get("released", False)


def outstanding_fine(world: "World") -> float:
    """Total fine for every active crime (the same crime counted once), stolen goods aside."""
    memories: Dict[tuple, float] = {}
    for guard in world.guards():
        for memory in guard.active_memories(world.day_count):
            memories.setdefault(memory.key, memory.fine)
    return sum(memories.values())


def return_stolen_goods(world: "World", player: Player, notify: Notify = None) -> List[Item]:
    """Stolen items in the player's pockets go back to their owners (unknown origin: the confiscation lockbox)."""
    returned = []
    for item in [i for i in player.inventory if i.is_stolen]:
        player.inventory.remove(item)
        item.equipped = False                     # taken off the thief's back, if it was being worn
        item.is_stolen = False                    # back with its owner (who never stopped owning it)
        add_item(world.owner_inventory_for(item), item)
        returned.append(item)
    if returned and notify:
        notify(f"Returned to their owners: {', '.join(i.display_name for i in returned)}.")
    return returned


def charge_player(world: "World", guard: "NPC", player: Player, notify: Notify = None) -> str:
    """The guard settles the player's account (``judge``). Returns 'paid', 'jailed', 'warned'
    or 'nothing'.

    The player is made to face the guard and any screen they had open is closed
    (``World.player_caught``); the fines are paid as far as the purse goes and the stolen
    goods go back to their owners. A sentence means the cells; a paid bill without one
    means, if they were caught somewhere they had no business being, being shown the door."""
    verdict = judge(world, player)
    if verdict.nothing:
        return "nothing"
    world.player_caught(guard)
    player.gold -= verdict.paid
    return_stolen_goods(world, player, notify)
    what = verdict.describe()
    days_text = f"{verdict.days} day{'s' if verdict.days != 1 else ''} in the cells"
    if not verdict.unpaid and verdict.days == 0:
        if notify:
            notify(f"({guard.name}): \"{what} - that's {verdict.fine:.0f} gold in fines. Pay up.\" You pay {verdict.fine:.0f}g.")
        _clear_memories(world)
        world.evict_if_trespassing(player, guard, notify)
        return "paid"
    if not world.jail.present:
        if notify:
            notify(f"({guard.name}): \"{what} - {verdict.fine:.0f} gold. There's no cell to put you in, so I'll take what "
                   f"you've got. Consider this your only warning.\" You pay {verdict.paid:.0f}g.")
        _clear_memories(world)
        world.evict_if_trespassing(player, guard, notify)
        return "warned"
    if notify:
        if verdict.unpaid:
            notify(f"({guard.name}): \"{what} - {verdict.fine:.0f} gold, and you can't pay it all. "
                   f"It's {days_text} for you.\" You pay {verdict.paid:.0f}g.")
        else:
            notify(f"({guard.name}): \"{what} - {verdict.fine:.0f} gold, and {days_text}.\" You pay {verdict.fine:.0f}g.")
    imprison(world, player, verdict.days, notify)
    _clear_memories(world)
    return "jailed"


def _clear_memories(world: "World") -> None:
    for guard in world.npcs:
        if guard.is_guard:
            guard.forget_crimes()
            guard.suspicion_meter = 0.0


def confiscate(world: "World", player: Player) -> Optional[Item]:
    """Move the player's belongings (minus one lockpick) into the lockbox, tagged as theirs.
    Returns the lockpick they keep, if any."""
    jail = world.jail
    storage = world.find_container(jail.storage_name)
    kept_pick = None
    for item in list(player.inventory):
        if item.is_lockpick and kept_pick is None:
            kept_pick = item.split(1) if item.stackable and item.quantity > 1 else item
            if kept_pick is item:
                continue
        player.inventory.remove(item)
        item.equipped = False                     # stripped of what they wore: it waits in the lockbox
        item.owner = Item.PLAYER_OWNER
        if storage is not None:
            add_item(storage.inventory, item)
    player.inventory = [kept_pick] if kept_pick is not None else []
    return kept_pick


def imprison(world: "World", player: Player, days: int, notify: Notify = None) -> None:
    """Lock the player in the cell for ``days`` days: belongings confiscated (what is left of
    the purse after the fines stays theirs), cell sealed, lockbox locked. The move itself goes
    through ``World.relocate_player`` so the presentation can fade to black around it - and
    saving is refused (``JailState.transfer_pending``) from the first line here until the
    player actually stands in the cell, so no save can catch them sentenced but still free."""
    jail = world.jail
    jail.transfer_pending = True          # no saving from here until the player stands in the cell
    confiscate(world, player)
    storage = world.find_container(jail.storage_name)
    if storage is not None:
        storage.lock_and_close()
    door = world.find_door(jail.cell_door_id)
    if door is not None:
        door.lock_and_close()
        door.sealed = True
    jail.sentence_end_day = world.day_count + days
    jail.escaped = False
    jail.released = False
    player.is_crouching = False
    player.has_lit_torch = False

    def in_the_cell() -> None:
        jail.transfer_pending = False     # the move has happened: saving is allowed again

    world.relocate_player(Vector3D(jail.cell_spot.x, player.position.y, jail.cell_spot.z), yaw=180.0, reason="jailed",
                          on_applied=in_the_cell)
    if notify:
        notify(f"You are thrown in the cell. Your belongings are locked in the confiscation lockbox; "
               f"release in {days} day{'s' if days != 1 else ''}. (You still have one lockpick...)")


def release(world: "World", player: Player, notify: Notify = None) -> None:
    """Time served: open the bars and the lockbox; the jailer waits until the player has gone."""
    jail = world.jail
    jail.sentence_end_day = None
    jail.escaped = False
    jail.released = True
    jail.grace_until = world.elapsed + world.settings.justice.release_grace
    door = world.find_door(jail.cell_door_id)
    if door is not None:
        door.sealed = False
        door.is_locked = False
        door.is_open = True
    storage = world.find_container(jail.storage_name)
    if storage is not None:
        storage.unlock_and_open()
    jailer = world.find_npc(jail.jailer) if jail.jailer else None
    if jailer is not None:
        jailer.wait_for("release", player.position)
        if jail.post is not None and jailer.position.distance_to(jail.post) > 1.0:
            jailer.go_to(world, jail.post, AIState.WAITING)      # walk to the post and watch from there
    who = jailer.name if jailer is not None else "Jailer"
    if notify:
        notify(f"({who}): \"Your time is served. Collect your things from the lockbox and get out.\"")


def end_release(world: "World", notify: Notify = None) -> None:
    """The ex-prisoner has left (or overstayed): lock the lockbox again and let the jailer go back to work."""
    jail = world.jail
    jail.released = False
    jail.grace_until = 0.0
    storage = world.find_container(jail.storage_name)
    if storage is not None:
        storage.lock_and_close()
    jailer = world.find_npc(jail.jailer) if jail.jailer else None
    if jailer is not None and jailer.state is AIState.WAITING and jailer.waiting_for == "release":
        jailer.stop_waiting(world, None)


def update_jail(world: "World", player: Player, notify: Notify = None) -> None:
    """Called every hour tick: release when the day and hour come, or raise the alarm on a jailbreak."""
    jail = world.jail
    if not jail.serving:
        return
    if world.day_count >= jail.sentence_end_day and world.clock.hour >= world.settings.justice.release_hour:
        release(world, player, notify)
        return
    check_jailbreak(world, player, notify)


def check_jailbreak(world: "World", player: Player, notify: Notify = None) -> bool:
    """A serving prisoner outside the jail has broken out: unseal the cell and declare the crime."""
    jail = world.jail
    if not jail.serving or jail.escaped:
        return False
    if world.area_of(player.position) in (jail.cell_area, jail.jail_area):
        return False
    jail.escaped = True
    door = world.find_door(jail.cell_door_id)
    if door is not None:
        door.sealed = False
    if notify:
        notify("You have broken out of jail! The guards will not forget this.")
    # the empty cell is where the alarm is raised: the guards in earshot of the jail come running there
    world.declare_crime("Jailbreak", jail.cell_spot.copy(), witnessed=False, reporter=jail.jailer or "the jailer")
    jail.sentence_end_day = None
    return True
