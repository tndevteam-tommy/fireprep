"""Save-game persistence.

Version 2 added the clock minutes, player orientation, door states and NPC
inventories/torches; version 3 containers, item stacks and crime counters;
version 4 the jail; version 5 NPC purses, container locks, item ownership and a
``meta`` block so the save menus can list saves without loading them; version 6
puts the player's position at their feet (older saves are lowered on loading);
version 7 gives every item an owner (older saves: whoever held it); version 8 adds
equipment (what an item is worn as, and whether the player has it on - older saves:
the catalogue supplies the gear by item id and whatever the player carried that can be
worn is put on, since it counted as worn before); version 9 is the rebuilt town on its
hill with the country beyond the gate, items' rarity and skill bonuses and the quest log
(older saves: the player is put back at the town's spawn point and everybody at their
post for the hour - the old coordinates point at nothing now).
Older files still load (missing fields keep their current values). Files are written
atomically (temp file + rename) and a corrupt one is reported, never loaded.

Saves live in two places: the *quick save* file (``filename``, F5/F9) and the
``saves/`` directory next to it, one file per named save. ``list_saves`` returns
both, newest first, for the Save / Load / Continue menus. By default ``filename``
is ``savegame.json`` in the per-user data directory (see :mod:`unprepared.paths`),
so saves never depend on the working directory the game was launched from.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..entities import Player
from ..entities.equipment import slot_of
from ..paths import SAVES_SUBDIR, default_save_file, write_json
from .world import World

Notify = Optional[Callable[[str], None]]

SAVE_VERSION = 9


@dataclass(frozen=True)
class SaveInfo:
    path: str
    label: str            # "Day 3 09:00 - 120g - 2024-05-01 12:33"
    modified: float       # mtime (seconds since the epoch)
    quick: bool = False   # the F5 quick-save file

    @property
    def name(self) -> str:
        return "Quick Save" if self.quick else os.path.splitext(os.path.basename(self.path))[0]


class SaveManager:
    def __init__(self, filename: Optional[str] = None, notify_callback: Notify = None):
        self.filename = os.path.abspath(filename or default_save_file())
        self.notify = notify_callback
        self.last_loaded: Optional[dict] = None      # the whole of the last save applied (the game reads its own blocks)

    @property
    def save_dir(self) -> str:
        return os.path.join(os.path.dirname(self.filename), SAVES_SUBDIR)

    # ------------------------------------------------------------------ (de)serialisation
    @staticmethod
    def serialize(world: World, player: Player, extra: Optional[dict] = None) -> dict:
        """The save as a dict; ``extra`` (the game's own blocks, e.g. the tutorial) is merged in."""
        data = {
            "version": SAVE_VERSION,
            "meta": {
                "day": world.day_count, "hour": world.simulated_hour, "gold": round(player.gold, 1),
                "level": player.level, "saved_at": time.time(),
            },
            "simulated_hour": world.simulated_hour,
            "minute_accumulator": world.clock.minute_accumulator,
            "day_count": world.day_count,
            "elapsed": world.elapsed,
            "player": player.to_dict(),
            "npcs": [npc.to_dict() for npc in world.npcs],
            "doors": [door.to_dict() for door in world.doors],
            "containers": [c.to_dict() for c in world.containers],
            "jail": world.jail.to_dict(),
            "inn": world.inn.to_dict(),
        }
        if extra:
            data.update(extra)
        return data

    @staticmethod
    def apply(save_data: dict, world: World, player: Player) -> None:
        world.reset_transient()                # nothing of the running session survives a load
        world.simulated_hour = save_data.get("simulated_hour", 7)
        world.clock.minute_accumulator = save_data.get("minute_accumulator", 0.0)
        world.day_count = save_data.get("day_count", 0)
        world.elapsed = save_data.get("elapsed", 0.0)
        player.load_dict(save_data["player"])
        npc_data_map = {n["name"]: n for n in save_data.get("npcs", [])}
        for npc in world.npcs:
            if npc.name in npc_data_map:
                npc.load_dict(npc_data_map[npc.name])
        door_data_map = {d["door_id"]: d for d in save_data.get("doors", [])}
        for door in world.doors:
            if door.door_id in door_data_map:
                door.load_dict(door_data_map[door.door_id])
        container_map = {c["name"]: c for c in save_data.get("containers", [])}
        for container in world.containers:
            if container.name in container_map:
                container.load_dict(container_map[container.name])
        world.jail.load_dict(save_data.get("jail", {}))
        world.inn.load_dict(save_data.get("inn", {}))
        if save_data.get("version", 1) < 5:
            for item in player.inventory:          # pre-ownership saves: what you carry (unstolen) is yours
                if not item.is_stolen and item.owner is None:
                    item.owner = item.PLAYER_OWNER
        if save_data.get("version", 1) < 6:
            player.position.y = 0.0                # older saves kept the player a unit up; the feet are on the floor now
        if save_data.get("version", 1) < 7:
            # pre-ownership saves: whatever an NPC or a container holds (unstolen) is theirs
            for npc in world.npcs:
                for item in npc.inventory:
                    if item.owner is None and not item.is_stolen:
                        item.owner = npc.name
            for container in world.containers:
                for item in container.inventory:
                    if item.owner is None and not item.is_stolen:
                        item.owner = container.owner
        if save_data.get("version", 1) < 8:
            # before equipment existed everything carried counted as worn: put it on, one per slot
            for item in player.inventory:
                if item.is_equipment and player.equipped_in(slot_of(item)) is None:
                    player.equip(item)
        if save_data.get("version", 1) < 9:
            # the town has been rebuilt since: the saved coordinates point at nothing now
            player.position = world.spot(*world.level.player_spawn)
            world.resync_schedules(announce=False, teleport=True, doors=False)
            return
        world.resync_schedules(announce=False, doors=False)      # doors stay exactly as saved

    # ------------------------------------------------------------------ files
    UNREADABLE = "(unreadable)"

    @staticmethod
    def read(path: str) -> Optional[dict]:
        """The parsed save, or None if the file cannot be read or is not a save (corrupt, truncated
        by a crash, not JSON, not a JSON object)."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    @classmethod
    def describe(cls, path: str) -> str:
        """Human-readable label for a save file (reads only the small ``meta`` block when present);
        a corrupt file is labelled as such so the menus can show it without choking on it. This
        feeds every menu (the start screen's Continue included), so it never raises: a file that
        is valid JSON but not shaped like a save - a ``meta`` that is not an object, an hour that
        is not a number - is flagged unreadable like any other broken file."""
        unreadable = f"{os.path.basename(path)} {cls.UNREADABLE}"
        data = cls.read(path)
        if data is None:
            return unreadable
        try:
            meta = data.get("meta")
            if not meta:
                player = data.get("player")
                meta = {"day": data.get("day_count", 0), "hour": data.get("simulated_hour", 0),
                        "gold": player.get("gold", 0.0) if isinstance(player, dict) else 0.0,
                        "saved_at": os.path.getmtime(path)}
            if not isinstance(meta, dict):
                return unreadable
            saved_at = meta.get("saved_at")
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(
                float(saved_at) if saved_at is not None else os.path.getmtime(path)))
            return (f"Day {int(meta.get('day', 0)) + 1} {int(meta.get('hour', 0)):02d}:00 - "
                    f"{float(meta.get('gold', 0.0)):.0f}g - {stamp}")
        except (TypeError, ValueError, OverflowError, OSError):
            return unreadable

    def list_saves(self) -> List[SaveInfo]:
        """Every save (quick save + ``saves/*.json``), newest first."""
        found: List[SaveInfo] = []
        candidates = [(os.path.abspath(self.filename), True)]
        if os.path.isdir(self.save_dir):
            try:
                names = os.listdir(self.save_dir)
            except OSError:
                names = []
            candidates += [(os.path.join(self.save_dir, name), False) for name in names if name.lower().endswith(".json")]
        for path, quick in candidates:
            try:
                if not os.path.isfile(path):
                    continue
                found.append(SaveInfo(path, self.describe(path), os.path.getmtime(path), quick=quick))
            except OSError:
                continue                         # deleted (or made unreadable) while the list was being made
        found.sort(key=lambda info: info.modified, reverse=True)
        return found

    def latest(self) -> Optional[SaveInfo]:
        saves = self.list_saves()
        return saves[0] if saves else None

    def has_saves(self) -> bool:
        return bool(self.list_saves())

    def new_save_path(self) -> str:
        """A fresh, unused file name in the saves directory."""
        os.makedirs(self.save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.save_dir, f"save-{stamp}.json")
        n = 1
        while os.path.exists(path):
            n += 1
            path = os.path.join(self.save_dir, f"save-{stamp}-{n}.json")
        return path

    # ------------------------------------------------------------------ save / load
    def save(self, world: World, player: Player, path: Optional[str] = None, extra: Optional[dict] = None) -> Optional[str]:
        """Write the game to ``path`` (default: the quick-save file). Returns the path written, or
        None if it could not be (a full disk, a read-only directory: reported, never raised). The
        write is atomic (``write_json``): a failure mid-save leaves the previous file intact."""
        path = path or self.filename
        try:
            write_json(path, self.serialize(world, player, extra))
        except OSError as exc:
            if self.notify:
                self.notify(f"Could not write the save file: {exc}")
            return None
        if self.notify:
            self.notify("Game state saved successfully!" if path == self.filename
                        else f"Game saved to {os.path.basename(path)}.")
        return path

    def save_new(self, world: World, player: Player) -> Optional[str]:
        return self.save(world, player, self.new_save_path())

    def load(self, world: World, player: Player, path: Optional[str] = None) -> bool:
        """Load ``path`` (default: the quick-save file) into ``world``/``player``. A missing,
        corrupt or malformed file is reported and leaves the game as it was (False) - one that
        fails part-way through being applied is rolled back to the state before the load."""
        path = path or self.filename
        if not os.path.exists(path):
            if self.notify:
                self.notify("No save file found!")
            return False
        save_data = self.read(path)
        if save_data is None or "player" not in save_data:
            if self.notify:
                self.notify(f"The save file {os.path.basename(path)} is corrupt or not a save; nothing was loaded.")
            return False
        # The world is overwritten field by field, so a save that turns out to be malformed half-way
        # through (a block of the wrong shape, an item without an id) would leave a half-loaded game
        # behind. The game as it stands is kept first and put back if the load fails.
        before = self.serialize(world, player)
        try:
            self.apply(save_data, world, player)
            self.last_loaded = save_data
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
            self.apply(before, world, player)
            if self.notify:
                self.notify(f"The save file {os.path.basename(path)} is damaged ({type(exc).__name__}); "
                            "nothing was loaded.")
            return False
        if self.notify:
            self.notify("Game state loaded successfully!")
        return True
