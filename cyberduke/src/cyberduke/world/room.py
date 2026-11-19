"""``Room``: the live state of a level - doors, collision, walkable area, elevation, lights.

All geometry comes from a ``LevelData``; nothing here is specific to one level.
"""

from __future__ import annotations

from collections.abc import Sequence

from cyberduke.core.events import EventBus
from cyberduke.core.geometry import AABB
from cyberduke.world.doors import SlidingDoor
from cyberduke.world.level import Bounds, LevelData, SectorData, WalkRegion
from cyberduke.world.lights import Light
from cyberduke.world.surfaces import ElevationState, floor_height, on_flights_and_decks


class Room:
    def __init__(self, level: LevelData) -> None:
        self.level = level
        self.sectors: tuple[SectorData, ...] = level.sectors
        self.catwalk = level.catwalk
        self.nav_bounds: Bounds = level.nav_bounds
        self.doors: dict[str, SlidingDoor] = {
            d.door_id: SlidingDoor.from_data(d) for d in level.doors
        }
        self.anim_time = 0.0
        self._static_obstacles: list[AABB] = [c for s in self.sectors for c in s.colliders]
        self._nav_blockers: list[AABB] = [b for s in self.sectors for b in s.nav_blockers]
        self._occluders: list[AABB] = [o for s in self.sectors for o in s.occluders]
        self._regions: list[tuple[WalkRegion, SlidingDoor | None]] = [
            (r, self._door_for(r.requires_door)) for r in level.walk_regions
        ] + [(d.passage, self.doors[d.door_id]) for d in level.doors if d.passage is not None]

    def _door_for(self, door_id: str | None) -> SlidingDoor | None:
        if door_id is None:
            return None
        try:
            return self.doors[door_id]
        except KeyError:
            raise KeyError(f"walk region requires unknown door {door_id!r}") from None

    # ------------------------------------------------------------------ lifecycle
    def bind(self, bus: EventBus) -> None:
        """Subscribes the level's reactive parts (doors) to the event bus."""
        for door in self.doors.values():
            door.bind(bus)

    def reset(self) -> None:
        for door in self.doors.values():
            door.reset()
        self.anim_time = 0.0

    def update(self, dt: float, bodies: Sequence[tuple[float, float]] = ()) -> None:
        """Advances the doors; ``bodies`` (x, z) are who an approach door opens for."""
        self.anim_time += dt
        for door in self.doors.values():
            door.update(dt, bodies)

    # ------------------------------------------------------------------ queries
    def is_valid_position(self, x: float, z: float, radius: float = 0.32) -> bool:
        """True when an agent's bounding circle is inside some walkable region whose door
        (if any) is currently passable."""
        return any(
            (door is None or door.passable) and region.contains(x, z, radius)
            for region, door in self._regions
        )

    def get_obstacles(self) -> list[AABB]:
        """Collision/LOS boxes from all sectors plus every door that is still shut."""
        obs = list(self._static_obstacles)
        obs.extend(d.obstacle for d in self.doors.values() if not d.passable)
        return obs

    def get_sight_blockers(self) -> list[AABB]:
        """Everything opaque: colliders that block sight (walls, cover, shut doors, rail
        panels) plus sight-only occluders (catwalk decks, stair blocks). Boxes flagged
        ``blocks_sight=False`` (e.g. balustrades with gaps) are transparent by design."""
        return [o for o in self.get_obstacles() if o.blocks_sight] + self._occluders

    def get_nav_obstacles(self) -> list[AABB]:
        """What ground-walking NPCs must route around: every collider plus the level's
        NPC-only no-go boxes (e.g. staircases, which NPCs cannot climb yet)."""
        return self.get_obstacles() + self._nav_blockers

    def get_lights(self) -> list[Light]:
        return self.level.lights

    def floor_height(
        self, x: float, z: float, state: ElevationState
    ) -> tuple[float, ElevationState]:
        """Floor elevation under (x, z) plus the agent's updated elevation-tracking state
        (the catwalk and its stairs, and any low platform - a doorstep - stepped up on)."""
        y, state = floor_height(x, z, state, self.catwalk)
        if self.level.flights or state.deck_y is not None:
            y, state = on_flights_and_decks(x, z, y, state, self.level.flights, self.level.decks)
        for top in self.level.platforms:
            if top.contains_xz(x, z):
                y = max(y, top.max_y)
        return y, state
