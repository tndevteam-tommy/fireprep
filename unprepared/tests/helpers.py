"""Placing things in the town for tests, whatever the lie of the land.

The town stands on a hillside: every spot has its own ground height, so tests never write a y
themselves - they say where on the map (x, z) and these helpers stand the player (or an NPC)
on the floor there. Doors are approached from their street side or their room side by name,
so a test reads "stand outside the shop door" rather than a pair of numbers.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from unprepared.core.math3d import Vector3D


def ground(world, x: float, z: float, y_hint: Optional[float] = None) -> float:
    """The floor under (x, z): the ground, or a step / upper storey within a step of ``y_hint``."""
    base = world.ground_height(x, z)
    return world.floor_height_at(x, z, base if y_hint is None else y_hint)


def spot(world, x: float, z: float, y_hint: Optional[float] = None) -> Vector3D:
    return Vector3D(x, ground(world, x, z, y_hint), z)


def yaw_towards(src: Vector3D, dst: Vector3D) -> float:
    """The yaw (0 = -z, 90 = +x) from ``src`` looking at ``dst``."""
    return math.degrees(math.atan2(dst.x - src.x, -(dst.z - src.z))) % 360.0


def face(game, yaw: float, pitch: float = 0.0) -> None:
    game.player.rotation_y = yaw
    game.player.rotation_x = pitch


def teleport(game, x: float, z: float, yaw: float = 0.0, pitch: float = 0.0, y_hint: Optional[float] = None,
             update: bool = True) -> Vector3D:
    """Stand the player at (x, z) on the floor there, facing ``yaw``."""
    game.player.position = spot(game.world, x, z, y_hint)
    face(game, yaw, pitch)
    if update:
        game.update(0.0)
    return game.player.position


def place(npc, world, x: float, z: float, y_hint: Optional[float] = None) -> Vector3D:
    """Stand an NPC at (x, z) on the floor there."""
    npc.position = spot(world, x, z, y_hint)
    return npc.position


def door_sides(world, door) -> Tuple[float, float]:
    """(outside, inside) as signs across the door's wall from its line (along z for a door in a
    wall running east-west, along x for one running north-south): which way the street is and
    which way the room. A door between two rooms counts the smaller room as its inside."""
    h = door.hinge_position
    c = door.center

    def probe(sign):
        x, z = door.to_xz(door.along_of(c.x, c.z), door.line + sign * 0.8)
        return Vector3D(x, h.y, z)
    probe_n, probe_s = world.building_at(probe(-1.0)), world.building_at(probe(1.0))
    if probe_n is not None and probe_s is None:
        return 1.0, -1.0
    if probe_s is not None and probe_n is None:
        return -1.0, 1.0
    area_n, area_s = world.area_def_at(probe(-1.0)), world.area_def_at(probe(1.0))
    size = lambda a: (a.max_xz[0] - a.min_xz[0]) * (a.max_xz[1] - a.min_xz[1]) if a is not None else 1e9   # noqa: E731
    return (1.0, -1.0) if size(area_n) < size(area_s) else (-1.0, 1.0)


def by_door(world, door_id: str, side: str = "outside", distance: float = 2.0, along: float = 0.5):
    """(position, yaw facing the door) ``distance`` from door ``door_id`` on its ``side``
    ("outside" / "inside"), ``along`` the doorway (0 = its low end, 1 = its high end)."""
    door = world.find_door(door_id)
    outside, inside = door_sides(world, door)
    sign = outside if side == "outside" else inside
    h = door.hinge_position
    lo, hi = door.span
    x, z = door.to_xz(lo + (hi - lo) * along, door.line + sign * distance)
    pos = spot(world, x, z, h.y)
    if door.along_x:
        return pos, (0.0 if sign > 0 else 180.0)
    return pos, (270.0 if sign > 0 else 90.0)


def stand_by_door(game, door_id: str, side: str = "outside", distance: float = 2.0, along: float = 0.5,
                  pitch: float = 0.0, update: bool = True) -> Vector3D:
    """The player at ``by_door``, facing the door."""
    pos, yaw = by_door(game.world, door_id, side, distance, along)
    game.player.position = pos
    face(game, yaw, pitch)
    if update:
        game.update(0.0)
    return pos


def stand_facing(game, target: Vector3D, distance: float = 2.0, yaw_from: float = 180.0, pitch: float = 0.0,
                 update: bool = True) -> Vector3D:
    """The player ``distance`` from ``target`` in the direction ``yaw_from`` (as seen from the target:
    180 = on its +z side), facing it."""
    a = math.radians(yaw_from)
    x, z = target.x + math.sin(a) * distance, target.z - math.cos(a) * distance
    pos = spot(game.world, x, z, target.y)
    game.player.position = pos
    face(game, yaw_towards(pos, target), pitch)
    if update:
        game.update(0.0)
    return pos


def lifted(world, xyz) -> Vector3D:
    """A level's local spot (x, y above its floor, z) where it is in the world."""
    x, y, z = xyz
    return world.spot(x, y, z)
