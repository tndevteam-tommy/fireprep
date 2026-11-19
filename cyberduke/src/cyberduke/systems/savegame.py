"""Save games: the whole state of a run, as JSON, in ``<home>/Unprepared/Saves``.

``snapshot(sim)`` gathers everything that makes this run this run - which level, where the
netrunner stands (and whether they are crouching, what they are carrying, what cyberware
is installed, what they have stolen), every door, terminal, camera (timers included),
vending machine, access panel and thing to pick up, every guard (where, what it is doing,
down or blinded), the people of the street (where they are, what they carry), the things
lying about, the networks breached and what the scanner has revealed, the quests and the
detection meter. ``restore(sim, data)`` rebuilds the level and puts all of it back.

``save_game`` writes a new file named by the date and time (``save_2031-04-02_23-15-07``;
a second save in the same second gets ``_2``) - a save never overwrites another.
``list_saves`` finds them, newest first, with what the load menu shows; ``load_game``
reads one into a simulation.

**Keeping it current**: anything new that a run can change belongs here - a new entity
kind gets a ``_save_x`` / ``_load_x`` pair below (and a round-trip test in
``tests/test_savegame.py``). Bump ``SAVE_VERSION`` when the format changes in a way old
files cannot be read with; ``restore`` refuses files from the future.

Some things are deliberately not saved: a conversation in progress, a terminal session,
an upload half done, a guard mid-investigation (it goes back to its routine from where it
stood) and the people of the street's current errands (each starts their story again
from where they stand).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cyberduke.core.inventory import Inventory
from cyberduke.core.userdata import saves_dir
from cyberduke.levels import LEVELS
from cyberduke.systems.quests import QuestLog

SAVE_FORMAT = "cyberduke-save"
SAVE_VERSION = 1
TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


class SaveError(ValueError):
    pass


# ============================================================================== snapshot
def level_key(level) -> str:
    """The ``LEVELS`` name of a shipped level (a level loaded from a file cannot be saved:
    there is no way to build it again)."""
    for key, build in LEVELS.items():
        if build().name == level.name:
            return key
    raise SaveError(f"level {level.name!r} is not a shipped level and cannot be saved")


def _r(v: float) -> float:
    return round(float(v), 5)


def snapshot(sim) -> dict[str, Any]:
    p = sim.player
    data: dict[str, Any] = {
        "format": SAVE_FORMAT,
        "version": SAVE_VERSION,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "level": level_key(sim.level),
        "level_name": sim.level.name,
        "time": _r(sim.time),
        "clock": {"minutes": _r(sim.clock.minutes)},
        "weather": sim.weather.to_dict(),
        "player": {
            "x": _r(p.x),
            "y": _r(p.y),
            "z": _r(p.z),
            "yaw": _r(p.yaw),
            "pitch": _r(p.pitch),
            "crouching": p.crouching,
            "floor_y": _r(p.floor_y),
            "on_catwalk": p.elevation.on_catwalk,
            "intel": list(p.intel),
            "inventory": p.inventory.to_dict(),
            "known_hacks": sorted(k.value for k in p.known_hacks),
        },  # fmt: skip
        "detection": _r(sim.detection.level if not sim.detection.is_compromised else 0.0),
        "doors": {
            d.door_id: {"unlocked": d.unlocked, "open_amount": _r(d.open_amount)}
            for d in sim.room.doors.values()
        },
        "terminals": {
            t.terminal_id: {
                "progress": _r(0.0 if t.is_connected else t.progress),
                "is_hacked": t.is_hacked,
                "intel_stolen": t.intel_stolen,
                "doors_overridden": t.doors_overridden,
                "destroyed": t.destroyed,
                "spark_timer": _r(t.spark_timer),
            }  # fmt: skip
            for t in sim.terminals
        },
        "cameras": {
            c.camera_id: {
                "anim_time": _r(c.anim_time),
                "shutdown_timer": _r(c.shutdown_timer),
                "head_droop": _r(c.head_droop),
            }  # fmt: skip
            for c in sim.cameras
        },
        "vending_machines": {
            v.machine_id: {
                "is_sparking": v.is_sparking,
                "spark_timer": _r(v.spark_timer),
                "cooldown": _r(v.cooldown),
                "broken": v.broken,
                "broken_timer": _r(v.broken_timer),
            }  # fmt: skip
            for v in sim.vending_machines
        },
        "panels": {pn.panel_id: {"bypasses": pn.bypasses} for pn in sim.panels},
        "pickups": {pk.pickup_id: {"taken": pk.taken} for pk in sim.pickups},
        "guards": {g.guard_id: _save_guard(g) for g in sim.guards},
        "civilians": {c.civilian_id: _save_person(c) for c in sim.civilians},
        "merchants": {m.merchant_id: _save_person(m) for m in sim.merchants},
        "life": _save_life(sim.life) if sim.life is not None else None,
        "hacking": {
            "breached": {
                fid: net.breached_via for fid, net in sim.hacking.networks.items() if net.breached
            },
            "revealed": {k: _r(v) for k, v in sim.scanner._revealed.items()},
            "scan_cooldown": _r(sim.scanner.cooldown),
            "log": list(sim.hacking.log),
            "hijacked": sim.hijacked_camera.camera_id if sim.hijacked_camera is not None else None,
        },
        "quests": sim.quests.to_dict(),
    }
    return data


def _save_guard(g) -> dict[str, Any]:
    d: dict[str, Any] = {
        "x": _r(g.x), "y": _r(g.y), "z": _r(g.z), "yaw": _r(g.yaw), "state": g.state,
        "patrol_dir": g.patrol_dir, "route_index": g.route_index,
        "inventory": g.inventory.to_dict(),
    }  # fmt: skip
    if g.is_blinded:
        d["blind_left"] = _r(g.blind_time_left)
    if g.is_down:
        state = g.fsm.state
        collapse = getattr(state, "collapse", None)
        d["down"] = True
        d["fall_direction"] = collapse.direction if collapse is not None else state.direction
    return d


def _save_person(n) -> dict[str, Any]:
    return {
        "x": _r(n.x), "y": _r(n.y), "z": _r(n.z), "yaw": _r(n.yaw),
        "hidden": bool(getattr(n, "hidden", False)),
        "asleep": bool(getattr(n, "asleep", False)),
        "riding": bool(getattr(n, "riding", False)),  # (aboard a metro train)
        "inventory": n.inventory.to_dict(),
        "activity": getattr(n, "activity", ""),
    }  # fmt: skip


def _save_life(life) -> dict[str, Any]:
    parcel = life.flags.get("parcel")
    return {
        "time": _r(life.time),
        "items": [
            {
                "name": i.name,
                "x": _r(i.x),
                "y": _r(i.y),
                "z": _r(i.z),
                "yaw": _r(i.yaw),
                "item_id": i.item_id,
            }
            for i in life.items
        ],
        "parcel_on_step": life.items.index(parcel) if parcel in life.items else None,
        "open": dict(life.open),
        # the televisions in the homes: on or off, which channel
        "tvs": {k: {"on": tv.on, "channel": tv.channel} for k, tv in life.tvs.items()},
    }


# ============================================================================== restore
def restore(sim, data: dict[str, Any]) -> None:
    """Rebuilds ``sim`` as ``data`` (a ``snapshot``) describes it."""
    if data.get("format") != SAVE_FORMAT:
        raise SaveError("not a Cyberduke save file")
    version = int(data.get("version", 0))
    if version > SAVE_VERSION:
        raise SaveError(f"save file version {version} is newer than this game ({SAVE_VERSION})")
    key = str(data["level"])
    if key not in LEVELS:
        raise SaveError(f"save names unknown level {key!r}")
    sim.load_level(LEVELS[key]())  # a clean slate (and renderers drop their caches)
    sim.quests = QuestLog.from_dict(data.get("quests", {}))
    sim.time = float(data.get("time", 0.0))
    clock = data.get("clock")
    if clock:  # (older saves also counted days: only the time of day is read)
        sim.clock.set(float(clock.get("minutes", 0.0)))
    weather = data.get("weather")
    if weather:  # (older saves: the weather as it is)
        sim.weather.load(weather)

    from cyberduke.systems.hacking import HackKind

    pd = data["player"]
    p = sim.player
    p.x, p.y, p.z = float(pd["x"]), float(pd["y"]), float(pd["z"])
    p.yaw, p.pitch = float(pd["yaw"]), float(pd["pitch"])
    p.crouching = bool(pd.get("crouching", False))
    p.floor_y = float(pd.get("floor_y", 0.0))
    p.elevation = type(p.elevation)(on_catwalk=bool(pd.get("on_catwalk", False)))
    p.intel = [str(i) for i in pd.get("intel", ())]
    p.inventory = Inventory.from_dict(pd["inventory"])
    if "known_hacks" in pd:
        p.known_hacks = {HackKind(k) for k in pd["known_hacks"] if k in HackKind._value2member_map_}

    for door_id, dd in data.get("doors", {}).items():
        door = sim.room.doors.get(door_id)
        if door is not None:
            door.unlocked = bool(dd["unlocked"])
            door.open_amount = float(dd["open_amount"])
    for t in sim.terminals:
        td = data.get("terminals", {}).get(t.terminal_id)
        if td:
            t.progress = float(td["progress"])
            t.is_hacked, t.intel_stolen = bool(td["is_hacked"]), bool(td["intel_stolen"])
            t.doors_overridden, t.destroyed = bool(td["doors_overridden"]), bool(td["destroyed"])
            t.spark_timer = float(td.get("spark_timer", 0.0))
    for c in sim.cameras:
        cd = data.get("cameras", {}).get(c.camera_id)
        if cd:
            c.anim_time = float(cd["anim_time"])
            c.shutdown_timer = float(cd["shutdown_timer"])
            c.head_droop = float(cd.get("head_droop", 0.0))
    for v in sim.vending_machines:
        vd = data.get("vending_machines", {}).get(v.machine_id)
        if vd:
            v.is_sparking, v.spark_timer = bool(vd["is_sparking"]), float(vd["spark_timer"])
            v.cooldown = float(vd["cooldown"])
            v.broken = bool(vd.get("broken", v.is_sparking))
            v.broken_timer = float(vd.get("broken_timer", v.spark_timer))
    for pn in sim.panels:
        pn.bypasses = int(data.get("panels", {}).get(pn.panel_id, {}).get("bypasses", 0))
    for pk in sim.pickups:
        pk.taken = bool(data.get("pickups", {}).get(pk.pickup_id, {}).get("taken", False))

    if sim.life is not None and data.get("life"):
        _load_life(sim.life, data["life"])
    obstacles = sim._collect_nav_obstacles()
    for g in sim.guards:
        gd = data.get("guards", {}).get(g.guard_id)
        if gd:
            _load_guard(g, gd, obstacles)
    for group, key_name in ((sim.civilians, "civilians"), (sim.merchants, "merchants")):
        for n in group:
            nid = n.civilian_id if key_name == "civilians" else n.merchant_id
            nd = data.get(key_name, {}).get(nid)
            if nd:
                _load_person(n, nd)

    hd = data.get("hacking", {})
    for fid, via in hd.get("breached", {}).items():
        net = sim.hacking.networks.get(fid)
        if net is not None:
            net.breached, net.breached_via = True, str(via)
    sim.scanner._revealed = {str(k): float(v) for k, v in hd.get("revealed", {}).items()}
    sim.scanner.cooldown = float(hd.get("scan_cooldown", 0.0))
    sim.hacking.log.clear()
    sim.hacking.log.extend(hd.get("log", ()))
    hijacked = hd.get("hijacked")
    if hijacked:
        cam = next((c for c in sim.cameras if c.camera_id == hijacked), None)
        if cam is not None and cam.hijack():
            sim.hacking.hijacked_camera = cam

    sim.detection.level = float(data.get("detection", 0.0))
    sim.obstacles = sim._collect_obstacles()
    sim.player_obstacles = sim.obstacles
    sim.nav_obstacles = sim._collect_nav_obstacles()
    sim.sight_blockers = sim._collect_sight_blockers()
    sim.hacking.update(sim.viewpoint, 0.0, sim.sight_blockers)


def _load_guard(g, gd: dict[str, Any], obstacles) -> None:
    from cyberduke.ai.guard import WalkingToPatrol, WalkingToWall

    g.x, g.y, g.z = float(gd["x"]), float(gd["y"]), float(gd["z"])
    g.yaw = g.target_yaw = float(gd["yaw"])
    g.patrol_dir = float(gd.get("patrol_dir", g.patrol_dir))
    g.route_index = int(gd.get("route_index", 0))
    if "inventory" in gd:
        g.inventory = Inventory.from_dict(gd["inventory"])
    g.frame_obstacles = list(obstacles)
    g.frame_sight_blockers = list(obstacles)
    if gd.get("down"):
        # down again, and the fall played out to where the body came to rest
        g.knock_out(gd.get("fall_direction"))
        for _ in range(400):
            g.fsm.update(1.0 / 30.0)
            state = g.fsm.state
            if getattr(state, "collapse", None) is not None and state.collapse.done:
                break
        g.shock_timer = 0.0
    elif "blind_left" in gd:
        g.blind(max(0.05, float(gd["blind_left"])))
    elif gd.get("state") in ("WALKING_TO_WALL", "LEANING") and g.wall_pos is not None:
        g.fsm.transition_to(WalkingToWall())  # (a lean starts over once it gets there)
    elif gd.get("state") not in ("PACING", "PATROL_ROUTE"):
        g.fsm.transition_to(WalkingToPatrol())  # back to its routine from where it stood


def _load_person(n, nd: dict[str, Any]) -> None:
    n.x, n.y, n.z = float(nd["x"]), float(nd["y"]), float(nd["z"])
    n.yaw = n.target_yaw = float(nd["yaw"])
    if "inventory" in nd:
        n.inventory = Inventory.from_dict(nd["inventory"])
    # each starts their story over from where they stand (what they were in the middle
    # of - an order half made, a parcel in hand - is not carried over)
    # (aboard a metro train: still aboard - out of sight until a train brings them back)
    n.hidden = bool(nd.get("riding", False))
    if hasattr(n, "riding"):
        n.riding = n.hidden
    # (asleep in bed: their story puts them back to bed, from beside it)
    n.asleep, n.body, n.fall_angle = False, None, 0.0
    if getattr(n, "storyteller", None) is not None:
        n.storyteller.reset()


def _load_life(life, ld: dict[str, Any]) -> None:
    from cyberduke.ai.alley import LooseItem

    life.time = float(ld.get("time", 0.0))
    life.items = [
        LooseItem(
            str(i["name"]),
            float(i["x"]),
            float(i["y"]),
            float(i["z"]),
            float(i.get("yaw", 0.0)),
            str(i.get("item_id", "")),
        )
        for i in ld.get("items", ())
    ]
    # things with an identity that somebody's story needs are always somewhere: if it was
    # in a hand when the game was saved, it is back where it started
    for start in life._initial:
        if start.item_id and life.item(start.item_id) is None:
            life.items.append(
                LooseItem(start.name, start.x, start.y, start.z, start.yaw, start.item_id)
            )
    for stall, is_open in ld.get("open", {}).items():
        life.open[str(stall)] = bool(is_open)
    for area_id, td in ld.get("tvs", {}).items():
        tv = life.tv(str(area_id))
        tv.on, tv.channel = bool(td.get("on", False)), int(td.get("channel", 0))
    idx = ld.get("parcel_on_step")
    if idx is not None and 0 <= int(idx) < len(life.items):
        life.flags["parcel"] = life.items[int(idx)]


# ============================================================================== files
@dataclass(frozen=True)
class SaveInfo:
    """One save file, as the load menu lists it."""

    path: Path
    saved_at: datetime
    level_name: str
    play_time: float
    quests: int
    where: str
    clock: str = ""  # the time of day in the city when it was saved ("21:37")

    @property
    def title(self) -> str:
        return self.saved_at.strftime("%Y-%m-%d  %H:%M:%S")

    @property
    def detail(self) -> str:
        minutes = int(self.play_time // 60)
        played = f"{minutes:02d}:{int(self.play_time % 60):02d} PLAYED"
        when = f"  |  {self.clock}" if self.clock else ""
        return f"{self.level_name.upper()}{when}  |  {played}  |  {self.where}"


def _new_path(directory: Path, when: datetime) -> Path:
    stem = f"save_{when.strftime(TIME_FORMAT)}"
    path = directory / f"{stem}.json"
    k = 2
    while path.exists():
        path = directory / f"{stem}_{k}.json"
        k += 1
    return path


def save_game(sim, directory: Path | None = None, when: datetime | None = None) -> Path:
    """Writes a new save file (never over an old one). Returns its path."""
    directory = Path(directory) if directory is not None else saves_dir()
    directory.mkdir(parents=True, exist_ok=True)
    when = when or datetime.now()
    data = snapshot(sim)
    data["saved_at"] = when.isoformat(timespec="seconds")
    data["where"] = describe_where(sim)
    path = _new_path(directory, when)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def describe_where(sim) -> str:
    area = sim.trespassing_in() if hasattr(sim, "trespassing_in") else None
    if area is not None:
        return f"INSIDE {area.name}"
    return "IN THE FACILITY" if sim.level.on_caught == "restart" else "ON THE STREET"


def list_saves(directory: Path | None = None) -> list[SaveInfo]:
    directory = Path(directory) if directory is not None else saves_dir()
    if not directory.is_dir():
        return []
    out = []
    for path in directory.glob("save_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("format") != SAVE_FORMAT:
                continue
            when = datetime.fromisoformat(str(data["saved_at"]))
            out.append(
                SaveInfo(
                    path,
                    when,
                    str(data.get("level_name", data.get("level", "?"))),
                    float(data.get("time", 0.0)),
                    len(data.get("quests", {}).get("quests", ())),
                    str(data.get("where", "")),
                    _clock_text(data.get("clock")),
                )  # fmt: skip
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue  # a broken or foreign file is not offered
    out.sort(key=lambda s: (s.saved_at, s.path.name), reverse=True)
    return out


def _clock_text(clock) -> str:
    if not clock:
        return ""
    m = int(float(clock.get("minutes", 0.0)))
    return f"{m // 60:02d}:{m % 60:02d}"


def load_game(sim, path: Path) -> None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SaveError(f"cannot read {path}: {e}") from e
    restore(sim, data)


def distance_moved(a: dict[str, Any], b: dict[str, Any]) -> float:
    """(For tests) how far the player is from one snapshot to another."""
    pa, pb = a["player"], b["player"]
    return math.hypot(pa["x"] - pb["x"], pa["z"] - pb["z"])
