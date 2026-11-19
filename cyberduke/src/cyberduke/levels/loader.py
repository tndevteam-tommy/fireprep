"""Serialise ``LevelData`` to/from plain dictionaries, JSON and TOML.

The typed Python level (``levels/facility.py``) stays the source of truth for the shipped
level; this module lets a level be exported for external editing and loaded back:

    python -m cyberduke.levels export my_level.json      # or .toml
    python -m cyberduke.levels check my_level.json
    cyberduke --level my_level.json

JSON is read and written with the standard library. TOML is read with ``tomllib``
(Python 3.11+; ``tomli`` on 3.10) and written with ``tomli_w`` when that optional package
is installed.

The format is versioned (``SCHEMA_VERSION``); decoding validates references (doors named
by walk regions, terminals named by doors) so a bad file fails loudly at load time rather
than as a KeyError mid-game.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from cyberduke.core.geometry import AABB
from cyberduke.world.level import (
    CameraData,
    CivilianData,
    DoorData,
    FactionData,
    GuardData,
    ItemData,
    LaneData,
    LevelData,
    MerchantData,
    MetroData,
    NoGoZone,
    OpenSky,
    PanelData,
    PickupData,
    Place,
    PrivateArea,
    SectorData,
    TerminalData,
    VendingMachineData,
    WalkRegion,
)
from cyberduke.world.lights import Light
from cyberduke.world.props import Material, Prop
from cyberduke.world.surfaces import Catwalk, Flight

SCHEMA_VERSION = 1


class LevelFormatError(ValueError):
    pass


# =============================================================================== encoding
def _aabb(b: AABB) -> dict[str, Any]:
    d = {"min_x": b.min_x, "max_x": b.max_x, "min_z": b.min_z, "max_z": b.max_z, "min_y": b.min_y, "max_y": b.max_y}
    if not b.blocks_sight:
        d["blocks_sight"] = False
    return d


def _material(m: Material) -> dict[str, Any]:
    d: dict[str, Any] = {}
    if m.texture is not None:
        d["texture"] = m.texture
    if m.color != (1.0, 1.0, 1.0):
        d["color"] = list(m.color)
    if m.emission is not None:
        d["emission"] = list(m.emission)
    if m.specular is not None:
        d["specular"] = list(m.specular)
        d["shininess"] = m.shininess
    if m.alpha != 1.0:
        d["alpha"] = m.alpha
    return d


def _prop(p: Prop) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": p.kind, "x": p.x, "y": p.y, "z": p.z, "size": list(p.size)}
    mat = _material(p.material)
    if mat:
        d["material"] = mat
    if p.rotate:
        d["rotate"] = [list(r) for r in p.rotate]
    if p.kind in ("cylinder", "disc"):
        d["slices"] = p.slices
    if p.solid:
        d["solid"] = True
    if p.label:
        d["label"] = p.label
    if p.tags:
        d["tags"] = sorted(p.tags)
    if p.clip is not None:
        d["clip"] = list(p.clip)
    return d


def _light(light: Light) -> dict[str, Any]:
    return {
        "x": light.x, "y": light.y, "z": light.z,
        "ambient": list(light.ambient), "diffuse": list(light.diffuse), "specular": list(light.specular),
        "linear_attenuation": light.linear_attenuation, "quadratic_attenuation": light.quadratic_attenuation,
        "fixture": light.fixture, "fixture_emission": list(light.fixture_emission),
        "fixture_color": list(light.fixture_color), "fixture_texture": light.fixture_texture,
        "fixture_offset_y": light.fixture_offset_y, "ceiling_y": light.ceiling_y,
        "night_only": light.night_only,
    }  # fmt: skip


def _walk_region(r: WalkRegion) -> dict[str, Any]:
    d: dict[str, Any] = {"min_x": r.min_x, "max_x": r.max_x, "min_z": r.min_z, "max_z": r.max_z, "inset": sorted(r.inset)}
    if r.requires_door:
        d["requires_door"] = r.requires_door
    return d


def level_to_dict(level: LevelData) -> dict[str, Any]:
    cw = level.catwalk
    return {
        "schema": SCHEMA_VERSION,
        "name": level.name,
        "nav_bounds": list(level.nav_bounds),
        "spawn": list(level.spawn),
        "catwalk": {
            "inner_x": cw.inner_x, "outer_x": cw.outer_x,
            "stairs_z_start": cw.stairs_z_start, "stairs_z_end": cw.stairs_z_end,
            "deck_z_end": cw.deck_z_end, "bridge_z_start": cw.bridge_z_start,
            "height": cw.height, "steps": cw.steps, "commit_fraction": cw.commit_fraction,
        },  # fmt: skip
        "sectors": [
            {
                "name": s.name,
                "bounds": list(s.bounds),
                "props": [_prop(p) for p in s.props],
                "colliders": [_aabb(c) for c in s.colliders],
                "nav_blockers": [_aabb(c) for c in s.nav_blockers],
                "occluders": [_aabb(c) for c in s.occluders],
                "lights": [_light(light) for light in s.lights],
                "walk_regions": [_walk_region(r) for r in s.walk_regions],
            }
            for s in level.sectors
        ],
        "doors": [
            {
                "door_id": d.door_id,
                "obstacle": _aabb(d.obstacle),
                "unlocked_by": d.unlocked_by,
                "travel": d.travel, "speed": d.speed, "passable_at": d.passable_at,
                "passage": _walk_region(d.passage) if d.passage else None,
                "leaf": [_prop(p) for p in d.leaf],
                "indicators": [_prop(p) for p in d.indicators],
                "lamps": [_prop(p) for p in d.lamps],
                "indicator_open": _material(d.indicator_open),
                "indicator_closed": _material(d.indicator_closed),
                "faction": d.faction, "high_security": d.high_security, "key_intel": d.key_intel,
                "approach_range": d.approach_range, "inside": d.inside,
                "hinge": list(d.hinge) if d.hinge is not None else None, "swing": d.swing,
                "hackable": d.hackable, "relock_after": d.relock_after,
            }  # fmt: skip
            for d in level.doors
        ],
        "factions": [
            {"faction_id": f.faction_id, "name": f.name, "color": list(f.color)} for f in level.factions
        ],
        "terminals": [
            {
                "terminal_id": t.terminal_id, "x": t.x, "y": t.y, "z": t.z, "yaw": t.yaw, "objective": t.objective,
                "faction": t.faction, "intel": t.intel, "secured": t.secured, "critical": t.critical,
                "hardened": t.hardened, "style": t.style, "notes": t.notes,
            }  # fmt: skip
            for t in level.terminals
        ],
        "cameras": [
            {
                "x": c.x, "y": c.y, "z": c.z, "mount_yaw": c.mount_yaw, "sweep_angle": c.sweep_angle, "sweep_speed": c.sweep_speed,
                "camera_id": c.camera_id, "faction": c.faction, "watch_area": c.watch_area,
            }  # fmt: skip
            for c in level.cameras
        ],
        "vending_machines": [
            {"x": v.x, "y": v.y, "z": v.z, "yaw": v.yaw, "machine_id": v.machine_id, "faction": v.faction}
            for v in level.vending_machines
        ],
        "guards": [
            {
                "x": g.x, "y": g.y, "z": g.z,
                "patrol_z_min": g.patrol_z_min, "patrol_z_max": g.patrol_z_max,
                "wall_pos": list(g.wall_pos) if g.wall_pos is not None else None,
                "wall_yaw": g.wall_yaw, "initial_direction": g.initial_direction,
                "guard_id": g.guard_id, "faction": g.faction, "essential": g.essential,
                "patrol_route": [list(p) for p in g.patrol_route],
            }  # fmt: skip
            for g in level.guards
        ],
        "exit_zone": _aabb(level.exit_zone) if level.exit_zone is not None else None,
        "exit_to": level.exit_to,
        "brief": level.brief,
        "civilians": [
            {"x": c.x, "y": c.y, "z": c.z, "waypoints": [list(w) for w in c.waypoints], "seed": c.seed, "civilian_id": c.civilian_id, "role": c.role}
            for c in level.civilians
        ],
        "merchants": [
            {"x": m.x, "y": m.y, "z": m.z, "face_yaw": m.face_yaw, "style": m.style, "seed": m.seed, "merchant_id": m.merchant_id, "role": m.role}
            for m in level.merchants
        ],
        "places": [{"name": p.name, "x": p.x, "z": p.z, "yaw": p.yaw, "y": p.y} for p in level.places],
        "items": [{"name": i.name, "x": i.x, "y": i.y, "z": i.z, "yaw": i.yaw, "item_id": i.item_id} for i in level.items],
        "no_go_zones": [
            {"box": _aabb(z.box), "reason": z.reason, "for_player": z.for_player, "for_npc": z.for_npc}
            for z in level.no_go_zones
        ],
        "on_caught": level.on_caught,
        "outdoor": level.outdoor,
        "panels": [
            {"panel_id": p.panel_id, "x": p.x, "y": p.y, "z": p.z, "door_id": p.door_id, "yaw": p.yaw, "faction": p.faction, "name": p.name}
            for p in level.panels
        ],
        "pickups": [
            {"pickup_id": p.pickup_id, "item_id": p.item_id, "x": p.x, "y": p.y, "z": p.z, "yaw": p.yaw}
            for p in level.pickups
        ],
        "private_areas": [
            {
                "area_id": a.area_id, "name": a.name, "boxes": [_aabb(b) for b in a.boxes],
                "exit": list(a.exit), "door_id": a.door_id, "confiscate": list(a.confiscate),
                "owner_role": a.owner_role,
            }  # fmt: skip
            for a in level.private_areas
        ],
        "platforms": [_aabb(b) for b in level.platforms],
        "routes": [list(r) for r in level.routes],
        "lanes": [
            {"lane_id": ln.lane_id, "z": ln.z, "x_from": ln.x_from, "x_to": ln.x_to, "y": ln.y}
            for ln in level.lanes
        ],
        "metro": (
            {
                "track_z": m.track_z, "floor_y": m.floor_y, "stop_x": m.stop_x, "edge_z": m.edge_z,
                "x_west": m.x_west, "x_east": m.x_east, "gates": list(m.gates),
                "gates_z": list(m.gates_z), "stair_foot": list(m.stair_foot),
                "stair_top": list(m.stair_top), "stair_steps": m.stair_steps,
                "gate_doors": list(m.gate_doors),
            }  # fmt: skip
            if (m := level.metro) is not None
            else None
        ),
        "open_sky": [
            {
                "min_x": r.min_x, "max_x": r.max_x, "min_z": r.min_z, "max_z": r.max_z,
                "ground": r.ground, "puddles": r.puddles,
            }  # fmt: skip
            for r in level.open_sky
        ],
        "shelters": [_aabb(b) for b in level.shelters],
        "flights": [
            {
                "min_x": f.min_x, "max_x": f.max_x, "min_z": f.min_z, "max_z": f.max_z,
                "bottom": f.bottom, "top": f.top, "steps": f.steps, "axis": f.axis,
                "commit_fraction": f.commit_fraction,
            }  # fmt: skip
            for f in level.flights
        ],
        "decks": [_aabb(b) for b in level.decks],
    }


# =============================================================================== decoding
def _rgb(v: Any, what: str) -> tuple[float, float, float]:
    try:
        r, g, b = (float(c) for c in v)
    except (TypeError, ValueError):
        raise LevelFormatError(f"{what}: expected three numbers, got {v!r}") from None
    return (r, g, b)


def _d_aabb(d: Mapping[str, Any], what: str) -> AABB:
    try:
        return AABB(
            float(d["min_x"]), float(d["max_x"]), float(d["min_z"]), float(d["max_z"]),
            float(d.get("min_y", 0.0)), float(d.get("max_y", 2.5)), bool(d.get("blocks_sight", True)),
        )  # fmt: skip
    except KeyError as e:
        raise LevelFormatError(f"{what}: missing {e}") from None


def _d_material(d: Mapping[str, Any] | None) -> Material:
    if not d:
        return Material()
    return Material(
        texture=d.get("texture"),
        color=_rgb(d["color"], "material.color") if "color" in d else (1.0, 1.0, 1.0),
        emission=_rgb(d["emission"], "material.emission") if d.get("emission") is not None else None,
        specular=_rgb(d["specular"], "material.specular") if d.get("specular") is not None else None,
        shininess=float(d.get("shininess", 0.0)),
        alpha=float(d.get("alpha", 1.0)),
    )


def _d_prop(d: Mapping[str, Any]) -> Prop:
    kind = d.get("kind", "box")
    if kind not in ("box", "cylinder", "disc"):
        raise LevelFormatError(f"prop kind must be 'box', 'cylinder' or 'disc', got {kind!r}")
    size = tuple(float(v) for v in d["size"])
    if len(size) != 3:
        raise LevelFormatError(f"prop size must have three numbers: {d.get('label', '')!r}")
    rotate = tuple(tuple(float(c) for c in r) for r in d.get("rotate", ()))
    for r in rotate:
        if len(r) != 4:
            raise LevelFormatError("prop rotate entries are [angle, axis_x, axis_y, axis_z]")
    clip = tuple(float(v) for v in d["clip"]) if d.get("clip") is not None else None
    if clip is not None and (len(clip) != 4 or kind != "disc"):
        raise LevelFormatError("clip is [u0, u1, v0, v1], on discs only")
    return Prop(
        kind, float(d["x"]), float(d["y"]), float(d["z"]), size, _d_material(d.get("material")),
        rotate, int(d.get("slices", {"cylinder": 10, "disc": 24}.get(kind, 0))), bool(d.get("solid", False)),
        str(d.get("label", "")), frozenset(d.get("tags", ())), clip,  # type: ignore[arg-type]
    )  # fmt: skip


def _d_light(d: Mapping[str, Any]) -> Light:
    return Light(
        float(d["x"]), float(d["y"]), float(d["z"]),
        ambient=_rgb(d["ambient"], "light.ambient"), diffuse=_rgb(d["diffuse"], "light.diffuse"),
        specular=_rgb(d["specular"], "light.specular"),
        linear_attenuation=float(d.get("linear_attenuation", 0.02)),
        quadratic_attenuation=float(d.get("quadratic_attenuation", 0.005)),
        fixture=str(d.get("fixture", "none")),
        fixture_emission=_rgb(d.get("fixture_emission", (0, 0, 0)), "light.fixture_emission"),
        fixture_color=_rgb(d.get("fixture_color", (1, 1, 1)), "light.fixture_color"),
        fixture_texture=str(d.get("fixture_texture", "neon_cyan")),
        fixture_offset_y=float(d.get("fixture_offset_y", 0.0)),
        ceiling_y=float(d.get("ceiling_y", 4.82)),
        night_only=bool(d.get("night_only", False)),
    )  # fmt: skip


def _d_walk_region(d: Mapping[str, Any]) -> WalkRegion:
    inset = frozenset(d.get("inset", ("min_x", "max_x", "min_z", "max_z")))
    bad = inset - {"min_x", "max_x", "min_z", "max_z"}
    if bad:
        raise LevelFormatError(f"walk region inset names unknown edges: {sorted(bad)}")
    return WalkRegion(float(d["min_x"]), float(d["max_x"]), float(d["min_z"]), float(d["max_z"]), inset, d.get("requires_door"))


def level_from_dict(data: Mapping[str, Any]) -> LevelData:
    schema = data.get("schema")
    if schema != SCHEMA_VERSION:
        raise LevelFormatError(f"unsupported level schema {schema!r} (expected {SCHEMA_VERSION})")

    sectors = tuple(
        SectorData(
            name=str(s["name"]),
            bounds=tuple(float(v) for v in s["bounds"]),
            props=tuple(_d_prop(p) for p in s.get("props", ())),
            colliders=tuple(_d_aabb(c, f"sector {s['name']} collider") for c in s.get("colliders", ())),
            nav_blockers=tuple(_d_aabb(c, f"sector {s['name']} nav_blocker") for c in s.get("nav_blockers", ())),
            occluders=tuple(_d_aabb(c, f"sector {s['name']} occluder") for c in s.get("occluders", ())),
            lights=tuple(_d_light(light) for light in s.get("lights", ())),
            walk_regions=tuple(_d_walk_region(r) for r in s.get("walk_regions", ())),
        )
        for s in data.get("sectors", ())
    )
    doors = tuple(
        DoorData(
            door_id=str(d["door_id"]),
            obstacle=_d_aabb(d["obstacle"], f"door {d['door_id']} obstacle"),
            unlocked_by=str(d["unlocked_by"]) if d.get("unlocked_by") is not None else None,
            travel=float(d.get("travel", 3.2)), speed=float(d.get("speed", 2.4)),
            passable_at=float(d.get("passable_at", 1.5)),
            passage=_d_walk_region(d["passage"]) if d.get("passage") else None,
            leaf=tuple(_d_prop(p) for p in d.get("leaf", ())),
            indicators=tuple(_d_prop(p) for p in d.get("indicators", ())),
            lamps=tuple(_d_prop(p) for p in d.get("lamps", ())),
            indicator_open=_d_material(d.get("indicator_open")) if d.get("indicator_open") else DoorData.indicator_open,
            indicator_closed=_d_material(d.get("indicator_closed")) if d.get("indicator_closed") else DoorData.indicator_closed,
            faction=str(d.get("faction", DoorData.faction)),
            high_security=bool(d.get("high_security", False)),
            key_intel=str(d["key_intel"]) if d.get("key_intel") else None,
            approach_range=float(d["approach_range"]) if d.get("approach_range") is not None else None,
            inside=str(d["inside"]) if d.get("inside") else None,
            hinge=(float(d["hinge"][0]), float(d["hinge"][1])) if d.get("hinge") is not None else None,
            swing=float(d.get("swing", DoorData.swing)),
            hackable=bool(d.get("hackable", True)),
            relock_after=float(d["relock_after"]) if d.get("relock_after") is not None else None,
        )  # fmt: skip
        for d in data.get("doors", ())
    )
    cw = data.get("catwalk", {})
    level = LevelData(
        name=str(data.get("name", "level")),
        sectors=sectors,
        doors=doors,
        catwalk=Catwalk(**{k: (int(v) if k == "steps" else float(v)) for k, v in cw.items()}),
        nav_bounds=tuple(float(v) for v in data["nav_bounds"]),
        spawn=tuple(float(v) for v in data["spawn"]),
        no_go_zones=tuple(
            NoGoZone(_d_aabb(z["box"], "no_go_zone"), str(z.get("reason", "")), bool(z.get("for_player", True)), bool(z.get("for_npc", True)))
            for z in data.get("no_go_zones", ())
        ),
        factions=tuple(
            FactionData(str(f["faction_id"]), str(f.get("name", f["faction_id"])), _rgb(f["color"], "faction.color") if "color" in f else FactionData.color)
            for f in data.get("factions", ())
        ),
        terminals=tuple(
            TerminalData(
                str(t["terminal_id"]), float(t["x"]), float(t["y"]), float(t["z"]), float(t.get("yaw", 0.0)), bool(t.get("objective", False)),
                faction=str(t.get("faction", TerminalData.faction)), intel=str(t.get("intel", "")),
                secured=bool(t.get("secured", False)), critical=bool(t.get("critical", False)),
                hardened=bool(t.get("hardened", False)), style=str(t.get("style", "triple")),
                notes=str(t.get("notes", "")),
            )  # fmt: skip
            for t in data.get("terminals", ())
        ),
        cameras=tuple(
            CameraData(
                float(c["x"]), float(c["y"]), float(c["z"]), float(c.get("mount_yaw", 0.0)), float(c.get("sweep_angle", 35.0)), float(c.get("sweep_speed", 1.2)),
                camera_id=str(c.get("camera_id", "")), faction=str(c.get("faction", CameraData.faction)),
                watch_area=str(c["watch_area"]) if c.get("watch_area") else None,
            )  # fmt: skip
            for c in data.get("cameras", ())
        ),
        vending_machines=tuple(
            VendingMachineData(
                float(v["x"]), float(v["y"]), float(v["z"]), float(v.get("yaw", 0.0)),
                machine_id=str(v.get("machine_id", "")), faction=str(v.get("faction", VendingMachineData.faction)),
            )  # fmt: skip
            for v in data.get("vending_machines", ())
        ),
        guards=tuple(
            GuardData(
                float(g["x"]), float(g["y"]), float(g["z"]), float(g["patrol_z_min"]), float(g["patrol_z_max"]),
                tuple(float(v) for v in g["wall_pos"]) if g.get("wall_pos") is not None else None,
                float(g.get("wall_yaw", 90.0)), float(g.get("initial_direction", 1.0)),
                guard_id=str(g.get("guard_id", "")), faction=str(g.get("faction", GuardData.faction)),
                essential=bool(g.get("essential", False)),
                patrol_route=tuple((float(p[0]), float(p[1])) for p in g.get("patrol_route", ())),
            )  # fmt: skip
            for g in data.get("guards", ())
        ),
        exit_zone=_d_aabb(data["exit_zone"], "exit_zone") if data.get("exit_zone") else None,
        exit_to=str(data["exit_to"]) if data.get("exit_to") else None,
        brief=str(data.get("brief", "")),
        civilians=tuple(
            CivilianData(
                float(c["x"]), float(c["y"]), float(c["z"]),
                tuple((float(w[0]), float(w[1])) for w in c["waypoints"]),
                int(c.get("seed", 0)), str(c.get("civilian_id", "")), str(c.get("role", "")),
            )  # fmt: skip
            for c in data.get("civilians", ())
        ),
        merchants=tuple(
            MerchantData(
                float(m["x"]), float(m["y"]), float(m["z"]), float(m["face_yaw"]),
                str(m.get("style", "cook")), int(m.get("seed", 100)), str(m.get("merchant_id", "")),
                str(m.get("role", "")),
            )  # fmt: skip
            for m in data.get("merchants", ())
        ),
        places=tuple(
            Place(str(p["name"]), float(p["x"]), float(p["z"]), float(p.get("yaw", 0.0)), float(p.get("y", 0.0)))
            for p in data.get("places", ())
        ),
        items=tuple(
            ItemData(str(i["name"]), float(i["x"]), float(i["y"]), float(i["z"]), float(i.get("yaw", 0.0)), str(i.get("item_id", "")))
            for i in data.get("items", ())
        ),
        on_caught=str(data.get("on_caught", "restart")),
        outdoor=bool(data.get("outdoor", False)),
        panels=tuple(
            PanelData(
                str(p["panel_id"]), float(p["x"]), float(p["y"]), float(p["z"]), str(p["door_id"]),
                yaw=float(p.get("yaw", 0.0)), faction=str(p.get("faction", PanelData.faction)), name=str(p.get("name", "")),
            )  # fmt: skip
            for p in data.get("panels", ())
        ),
        pickups=tuple(
            PickupData(str(p["pickup_id"]), str(p["item_id"]), float(p["x"]), float(p["y"]), float(p["z"]), float(p.get("yaw", 0.0)))
            for p in data.get("pickups", ())
        ),
        private_areas=tuple(
            PrivateArea(
                str(a["area_id"]), str(a.get("name", a["area_id"])),
                tuple(_d_aabb(b, f"private area {a['area_id']}") for b in a.get("boxes", ())),
                tuple(float(v) for v in a["exit"]),  # type: ignore[arg-type]
                door_id=str(a["door_id"]) if a.get("door_id") else None,
                confiscate=tuple(str(i) for i in a.get("confiscate", ())),
                owner_role=str(a.get("owner_role", "")),
            )  # fmt: skip
            for a in data.get("private_areas", ())
        ),
        platforms=tuple(_d_aabb(b, "platform") for b in data.get("platforms", ())),
        routes=tuple((str(r[0]), str(r[1])) for r in data.get("routes", ())),
        lanes=tuple(
            LaneData(str(ln["lane_id"]), float(ln["z"]), float(ln["x_from"]), float(ln["x_to"]), float(ln.get("y", 0.0)))
            for ln in data.get("lanes", ())
        ),
        metro=_d_metro(data.get("metro")),
        open_sky=tuple(
            OpenSky(
                float(r["min_x"]), float(r["max_x"]), float(r["min_z"]), float(r["max_z"]),
                float(r.get("ground", 0.0)), bool(r.get("puddles", True)),
            )  # fmt: skip
            for r in data.get("open_sky", ())
        ),
        shelters=tuple(_d_aabb(b, "shelter") for b in data.get("shelters", ())),
        flights=tuple(
            Flight(
                float(f["min_x"]), float(f["max_x"]), float(f["min_z"]), float(f["max_z"]),
                float(f["bottom"]), float(f["top"]), int(f["steps"]), str(f.get("axis", "+x")),
                float(f.get("commit_fraction", 0.95)),
            )  # fmt: skip
            for f in data.get("flights", ())
        ),
        decks=tuple(_d_aabb(b, "deck") for b in data.get("decks", ())),
    )
    _validate(level)
    return level


def _d_metro(m: Any) -> MetroData | None:
    if not m:
        return None
    return MetroData(
        track_z=float(m["track_z"]), floor_y=float(m["floor_y"]), stop_x=float(m["stop_x"]),
        edge_z=float(m["edge_z"]), x_west=float(m["x_west"]), x_east=float(m["x_east"]),
        gates=tuple(float(g) for g in m.get("gates", ())),
        gates_z=tuple(float(g) for g in m.get("gates_z", (0.0, 0.0))),  # type: ignore[arg-type]
        stair_foot=tuple(float(v) for v in m.get("stair_foot", (0.0, 0.0))),  # type: ignore[arg-type]
        stair_top=tuple(float(v) for v in m.get("stair_top", (0.0, 0.0))),  # type: ignore[arg-type]
        stair_steps=int(m.get("stair_steps", 0)),
        gate_doors=tuple(str(d) for d in m.get("gate_doors", ())),
    )  # fmt: skip


def _validate(level: LevelData) -> None:
    door_ids = {d.door_id for d in level.doors}
    if len(door_ids) != len(level.doors):
        raise LevelFormatError("door ids must be unique")
    for region in level.walk_regions:
        if region.requires_door is not None and region.requires_door not in door_ids:
            raise LevelFormatError(f"walk region requires unknown door {region.requires_door!r}")
    terminal_ids = {t.terminal_id for t in level.terminals}
    if len(terminal_ids) != len(level.terminals):
        raise LevelFormatError("terminal ids must be unique")
    for door in level.doors:
        if door.unlocked_by is not None and door.unlocked_by not in terminal_ids:
            raise LevelFormatError(f"door {door.door_id!r} is unlocked by unknown terminal {door.unlocked_by!r}")
    objectives = sum(t.objective for t in level.terminals)
    if level.exit_to is not None and level.exit_zone is None:
        raise LevelFormatError("exit_to needs an exit zone to step into")
    if level.exit_to is not None:
        from cyberduke.levels import LEVELS  # the registry imports this module: import late

        if level.exit_to not in LEVELS:
            known = ", ".join(sorted(LEVELS))
            raise LevelFormatError(
                f"exit_to names unknown level {level.exit_to!r} (registered: {known})"
            )
    if objectives > 1:
        raise LevelFormatError("at most one objective terminal")
    intel_names = {t.intel for t in level.terminals if t.intel}
    for door in level.doors:
        if door.key_intel is not None and door.key_intel not in intel_names:
            raise LevelFormatError(f"door {door.door_id!r} is keyed to intel {door.key_intel!r} that no terminal holds")
    if level.on_caught not in ("restart", "evict"):
        raise LevelFormatError(f"on_caught must be 'restart' or 'evict', got {level.on_caught!r}")
    for panel in level.panels:
        if panel.door_id not in door_ids:
            raise LevelFormatError(f"panel {panel.panel_id!r} works unknown door {panel.door_id!r}")
    area_ids = {a.area_id for a in level.private_areas}
    for cam in level.cameras:
        if cam.watch_area is not None and cam.watch_area not in area_ids:
            raise LevelFormatError(f"camera {cam.camera_id!r} watches unknown area {cam.watch_area!r}")
    place_names = {p.name for p in level.places}
    for a, b in level.routes:
        if a not in place_names or b not in place_names:
            raise LevelFormatError(f"route {a!r} - {b!r} names a place the level does not have")
    from cyberduke.core.inventory import ITEMS

    for pk in level.pickups:
        if pk.item_id not in ITEMS:
            raise LevelFormatError(f"pickup {pk.pickup_id!r} is an unknown item {pk.item_id!r}")
    names = [s.name for s in level.sectors]
    if len(set(names)) != len(names):
        raise LevelFormatError("sector names must be unique")
    faction_ids = [f.faction_id for f in level.factions]
    if len(set(faction_ids)) != len(faction_ids):
        raise LevelFormatError("faction ids must be unique")
    if level.factions:  # a declared faction table must cover every reference
        unknown = sorted(set(level.faction_ids_in_use) - set(faction_ids))
        if unknown:
            raise LevelFormatError(f"entities reference undeclared factions: {unknown}")
    explicit_ids = [
        *(c.camera_id for c in level.cameras if c.camera_id),
        *(v.machine_id for v in level.vending_machines if v.machine_id),
        *(g.guard_id for g in level.guards if g.guard_id),
        *terminal_ids,
        *door_ids,
        *(p.panel_id for p in level.panels),
    ]
    if len(set(explicit_ids)) != len(explicit_ids):
        raise LevelFormatError("hack target ids (terminals, doors, cameras, vending machines, guards) must be unique")


# =============================================================================== files
def _toml_reader():
    try:
        import tomllib  # Python 3.11+

        return tomllib.loads
    except ModuleNotFoundError:  # pragma: no cover - 3.10 only
        try:
            import tomli

            return tomli.loads
        except ModuleNotFoundError:
            raise LevelFormatError("reading TOML needs Python 3.11+ or the 'tomli' package") from None


def _strip_none(obj: Any) -> Any:
    """TOML has no null; drop None-valued keys (decoding treats absence as None)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(v) for v in obj]
    return obj


def dumps(level: LevelData, fmt: str = "json") -> str:
    data = level_to_dict(level)
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if fmt == "toml":
        try:
            import tomli_w
        except ModuleNotFoundError:
            raise LevelFormatError("writing TOML needs the optional 'tomli_w' package") from None
        return tomli_w.dumps(_strip_none(data))
    raise LevelFormatError(f"unknown level format {fmt!r} (json or toml)")


def loads(text: str, fmt: str = "json") -> LevelData:
    if fmt == "json":
        return level_from_dict(json.loads(text))
    if fmt == "toml":
        return level_from_dict(_toml_reader()(text))
    raise LevelFormatError(f"unknown level format {fmt!r} (json or toml)")


def _format_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".toml":
        return "toml"
    raise LevelFormatError(f"cannot tell the level format from {path.name!r}; use .json or .toml")


def save_level(level: LevelData, path: str | Path) -> None:
    path = Path(path)
    path.write_text(dumps(level, _format_of(path)), encoding="utf-8")


def load_level(path: str | Path) -> LevelData:
    path = Path(path)
    return loads(path.read_text(encoding="utf-8"), _format_of(path))


# =============================================================================== cli
def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    from cyberduke.levels import build_facility

    parser = argparse.ArgumentParser(prog="python -m cyberduke.levels")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="write the shipped facility level to a .json/.toml file")
    exp.add_argument("path")
    chk = sub.add_parser("check", help="load a level file and report what it contains")
    chk.add_argument("path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "export":
        save_level(build_facility(), args.path)
        print(f"wrote {args.path}")
        return 0
    level = load_level(args.path)
    print(
        f"{level.name}: {len(level.sectors)} sectors, {len(level.props)} props, "
        f"{sum(len(s.colliders) for s in level.sectors)} colliders, {len(level.lights)} lights, "
        f"{len(level.doors)} doors, {len(level.guards)} guards"
    )
    return 0

