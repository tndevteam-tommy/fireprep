"""Baking geometry into display lists so the CPU stops re-issuing it every frame.

The world's walls, furniture, floors and roofs never move, an NPC's clothes never change
and a door leaf is the same leaf however it is swung - yet immediate mode drew every
one of their vertices through PyOpenGL each frame (some 35 000 GL calls a frame). Here
a drawing is *recorded once*:

* :class:`Compiled` runs a draw routine with a recording ``material`` callback and
  captures the geometry into display lists - one list per run of identical material,
  so the per-object colour / emission / fog settings (shader uniforms, or fixed-function
  colour and material state) are replayed *outside* the lists, where both pipelines can
  apply them. Drawing a ``Compiled`` is a handful of calls however much geometry it holds.
* :class:`StaticScene` bakes everything static in a :class:`~unprepared.world.world.World`
  into spatial *chunks*, each with a solid part and a roofs part (so the "Roofs & floors"
  option still works without rebuilding), and culls whole chunks by draw distance.

Display lists are compatibility-profile OpenGL, like all the immediate-mode drawing they
replace; the lists are keyed to the world and the texture registry that built them and
released with the renderer.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

from OpenGL.GL import GL_COMPILE, glCallList, glDeleteLists, glEndList, glGenLists, glNewList, glPopMatrix, glPushMatrix, glTranslatef

from .buildings import draw_ceiling, draw_floor, draw_roof
from .models import draw_fixture_body
from .primitives import draw_box, draw_fronted_box, draw_ground, draw_table, ensure_sphere_list
from .roads import draw_roads
from .terrain import draw_draped_roads, draw_terrain_patch, draw_trails
from .furnishings import CONTAINER_DRAWERS as _FURNISHED, READABLE_DRAWERS
from .structures import CONTAINER_DRAWERS as _CONTAINERS

CONTAINER_DRAWERS = {**_CONTAINERS, **_FURNISHED}

Material = Callable[..., None]
DEFAULT_MATERIAL = dict(color=(1.0, 1.0, 1.0), emission=(0.0, 0.0, 0.0), textured=True, fog=1.0, vertex_alpha=False)


def _material_key(color=(1.0, 1.0, 1.0), emission=(0.0, 0.0, 0.0), textured: bool = True, fog: float = 1.0,
                  vertex_alpha: bool = False) -> dict:
    return dict(color=tuple(float(c) for c in color), emission=tuple(float(c) for c in emission),
                textured=bool(textured), fog=float(fog), vertex_alpha=bool(vertex_alpha))


class Compiled:
    """A drawing recorded into display lists: ``segments`` of (material settings, list id)."""

    def __init__(self):
        self.segments: List[Tuple[dict, int]] = []

    @classmethod
    def record(cls, draw: Callable[[Material], None]) -> "Compiled":
        """Run ``draw(material)`` once, capturing its geometry. Every call to ``material`` that
        changes the settings closes the current list and opens a new one."""
        ensure_sphere_list()                      # a nested glNewList would be an error: build helpers first
        out = cls()
        current: Dict[str, object] = {"kwargs": dict(DEFAULT_MATERIAL), "list": None}

        def begin(kwargs: dict) -> None:
            lst = glGenLists(1)
            glNewList(lst, GL_COMPILE)
            current["kwargs"], current["list"] = kwargs, lst

        def end() -> None:
            glEndList()
            out.segments.append((current["kwargs"], current["list"]))

        def material(color=(1.0, 1.0, 1.0), emission=(0.0, 0.0, 0.0), textured: bool = True, fog: float = 1.0,
                     vertex_alpha: bool = False) -> None:
            kwargs = _material_key(color, emission, textured, fog, vertex_alpha)
            if kwargs == current["kwargs"]:
                return
            end()
            begin(kwargs)

        begin(dict(DEFAULT_MATERIAL))
        draw(material)
        end()
        return out

    def draw(self, material: Material) -> None:
        """Replay: apply each segment's material settings, then call its list. Leaves the
        default material set so the caller need not reset anything."""
        last = None
        for kwargs, lst in self.segments:
            material(**kwargs)
            glCallList(lst)
            last = kwargs
        if last is not None and last != DEFAULT_MATERIAL:
            material(**DEFAULT_MATERIAL)

    def release(self) -> None:
        for _, lst in self.segments:
            glDeleteLists(lst, 1)
        self.segments = []


# ----------------------------------------------------------------------------- the static world
class Chunk:
    """One cell of the static scene: its members' union footprint (for culling) and the baked lists."""

    def __init__(self):
        self.min_x = self.min_z = self.min_y = float("inf")
        self.max_x = self.max_z = self.max_y = float("-inf")
        self.ground_draws: List[Callable[[Material], None]] = []    # the terrain: drawn before the paving
        self.solid_draws: List[Callable[[Material], None]] = []
        self.roof_draws: List[Callable[[Material], None]] = []
        self.objects = 0
        self.ground: Optional[Compiled] = None
        self.solid: Optional[Compiled] = None
        self.roofs: Optional[Compiled] = None
        self.baked = False

    def cover(self, min_x: float, min_z: float, max_x: float, max_z: float, min_y: float = None, max_y: float = None) -> None:
        self.min_x, self.min_z = min(self.min_x, min_x), min(self.min_z, min_z)
        self.max_x, self.max_z = max(self.max_x, max_x), max(self.max_z, max_z)
        if min_y is not None:
            self.min_y = min(self.min_y, min_y)
        if max_y is not None:
            self.max_y = max(self.max_y, max_y)

    PAD = 5.0          # what is drawn may reach past the boxes it is filed under (a crown of leaves, eaves)
    PAD_UP = 10.0      # (a tree's crown over its trunk's box, a spire over its walls)

    def in_view(self, frustum) -> bool:
        """Whether any part of the chunk (padded) is inside the view ``frustum`` (None: always)."""
        if frustum is None:
            return True
        lo_y = self.min_y if self.min_y != float("inf") else -1e4
        hi_y = self.max_y if self.max_y != float("-inf") else 1e4
        p = self.PAD
        return frustum.sees_box(self.min_x - p, lo_y - p, self.min_z - p, self.max_x + p, hi_y + self.PAD_UP, self.max_z + p)

    def within(self, eye, limit: float) -> bool:
        """Is any part of the chunk's footprint within ``limit`` of the eye (on the ground plane)?"""
        dx = max(self.min_x - eye[0], 0.0, eye[0] - self.max_x)
        dz = max(self.min_z - eye[2], 0.0, eye[2] - self.max_z)
        return math.hypot(dx, dz) <= limit

    def distance(self, eye) -> float:
        dx = max(self.min_x - eye[0], 0.0, eye[0] - self.max_x)
        dz = max(self.min_z - eye[2], 0.0, eye[2] - self.max_z)
        return math.hypot(dx, dz)

    def bake(self) -> None:
        self.baked = True
        if self.ground_draws:
            self.ground = Compiled.record(lambda m: [draw(m) for draw in self.ground_draws])
        self.solid = Compiled.record(lambda m: [draw(m) for draw in self.solid_draws])
        if self.roof_draws:
            self.roofs = Compiled.record(lambda m: [draw(m) for draw in self.roof_draws])
        # (recorded: the drawing routines - and the world objects they hold on to - are done with)
        self.ground_draws, self.solid_draws, self.roof_draws = [], [], []

    def release(self) -> None:
        for compiled in (self.ground, self.solid, self.roofs):
            if compiled is not None:
                compiled.release()
        self.ground = self.solid = self.roofs = None
        self.baked = False


class StaticScene:
    """Everything in a world that never moves or animates, baked once: the ground, the roads,
    walls and lintels, props, containers, stations, the modelled well and signs, lantern
    and candle bodies, floors, ceilings and roofs. Built for one world and one texture
    registry (``matches`` tells when it must be rebuilt)."""
    CHUNK_SIZE = 10.0            # small enough that culling a chunk is close to culling its objects one by one
    # flames, cloth, water: drawn live
    ANIMATED_STRUCTURES = ("forge", "oven", "campfire", "banner", "pool", "brazier", "fireplace", "fountain", "washing_line")
    ANIMATED_CONTAINERS = ("pedestal",)
    UNDRAWN = ("collider",)       # props that only block the way (a porch's columns: the porch draws them)
    TERRAIN_PATCH_CELLS = 10                       # heightfield cells per baked terrain patch (each way)
    PROP_TEXTURES = {"bed": "bed", "table": "wood", "counter": "wood"}
    FRONTED_TEXTURES = {"shelf": "wood", "lockbox": "lockbox_side"}     # picture on the front, plain elsewhere

    def __init__(self, world, textures, structures: dict, ground_cells: int):
        self.world = world
        self.textures = textures
        self.ground_cells = ground_cells
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.ground: Optional[Compiled] = None
        self.roads: Optional[Compiled] = None
        self._build(structures)

    def matches(self, world, textures, ground_cells: int) -> bool:
        """Whether this bake serves ``world``: the same world, or another built from the same level
        on the same ground (a new game, a save loaded from the start screen, the title's town -
        everything baked is the level's and never changes in play, so baking it again would be
        seconds spent drawing the same town)."""
        if self.textures is not textures or self.ground_cells != ground_cells:
            return False
        if self.world is world:
            return True
        same_ground = getattr(self.world, "terrain", None) is getattr(world, "terrain", None)
        return same_ground and getattr(self.world, "level", None) == getattr(world, "level", None)

    # --- building -------------------------------------------------------------------------
    @classmethod
    def prop_texture_key(cls, prop) -> str:
        for word, key in cls.PROP_TEXTURES.items():
            if word in prop.name:
                return key
        return prop.texture

    @staticmethod
    def room_direction(world, box) -> Tuple[float, float]:
        """(dx, dz) from a box towards the middle of the room it stands in (defaults to +z)."""
        cx, cz = (box.min_point.x + box.max_point.x) / 2.0, (box.min_point.z + box.max_point.z) / 2.0
        building = world.building_at(box.center)
        if building is None:
            return (0.0, 1.0)
        bx = (building.min_xz[0] + building.max_xz[0]) / 2.0
        bz = (building.min_xz[1] + building.max_xz[1]) / 2.0
        return (bx - cx, bz - cz)

    def _chunk_for(self, min_x: float, min_z: float, max_x: float, max_z: float, min_y: float = None,
                   max_y: float = None) -> Chunk:
        cx, cz = (min_x + max_x) / 2.0, (min_z + max_z) / 2.0
        key = (int(math.floor(cx / self.CHUNK_SIZE)), int(math.floor(cz / self.CHUNK_SIZE)))
        chunk = self.chunks.get(key)
        if chunk is None:
            chunk = self.chunks[key] = Chunk()
        chunk.cover(min_x, min_z, max_x, max_z, min_y, max_y)
        chunk.objects += 1
        return chunk

    def _add_box(self, box, draw: Callable[[Material], None], roof: bool = False) -> None:
        chunk = self._chunk_for(box.min_point.x, box.min_point.z, box.max_point.x, box.max_point.z,
                                box.min_point.y, box.max_point.y)
        (chunk.roof_draws if roof else chunk.solid_draws).append(draw)

    def _build(self, structures: dict) -> None:
        world, tex, level = self.world, self.textures, self.world.level
        terrain = getattr(world, "terrain", None)
        if terrain is None or terrain.flat:
            self.ground = Compiled.record(lambda m: draw_ground(tex[level.ground_texture], level.ground_half_size,
                                                                level.ground_texture_repeat, cells=self.ground_cells))
            self.roads = Compiled.record(lambda m: draw_roads(world.roads, tex, m))
        else:
            # the heightfield in patches, baked and culled with the chunks; paving draped over it
            self.ground = Compiled.record(lambda m: None)
            step = self.TERRAIN_PATCH_CELLS
            overlays = [(tex[key], weights, tile, projection) for key, weights, tile, projection in
                        (("cliff", getattr(terrain, "rock", None), 9.0, "vertical"),
                         ("sand", getattr(terrain, "sand", None), 6.0, "above"),
                         ("snow", getattr(terrain, "snow", None), 10.0, "above"))
                        if weights is not None and weights.any()]
            for c in range(0, terrain.cols - 1, step):
                for r in range(0, terrain.rows - 1, step):
                    c2, r2 = min(c + step, terrain.cols - 1), min(r + step, terrain.rows - 1)
                    x1, z1 = terrain.min_x + c * terrain.cell, terrain.min_z + r * terrain.cell
                    x2, z2 = terrain.min_x + c2 * terrain.cell, terrain.min_z + r2 * terrain.cell
                    patch = terrain.grid[c:c2 + 1, r:r2 + 1]
                    chunk = self._chunk_for(x1, z1, x2, z2, float(patch.min()), float(patch.max()))
                    chunk.ground_draws.append(lambda m, a=(c, r, c2, r2): draw_terrain_patch(
                        terrain, *a, tex[level.ground_texture], repeat=level.ground_texture_repeat,
                        overlays=overlays, material=m))
            bridges = [(p.min_point.x, p.min_point.z, p.max_point.x, p.max_point.z) for p in world.props
                       if getattr(p, "kind", "") == "bridge"]

            def over_water(x, z):
                """No paving over the water or under a bridge's deck (the deck carries the road)."""
                if any(x1 <= x <= x2 and z1 <= z <= z2 for x1, z1, x2, z2 in bridges):
                    return True
                return terrain.water_level_at(x, z) is not None
            lattice = (terrain.min_x, terrain.min_z, terrain.cell)

            heights = getattr(terrain, "heights_at", None)

            def paving(m):
                draw_trails(getattr(terrain, "trails", ()), tex, m, terrain.height, lattice, heights=heights)
                draw_draped_roads(world.roads, tex, m, terrain.height, skip=over_water, lattice=lattice, heights=heights)
            self.roads = Compiled.record(paving)

        for wall in world.walls:
            self._add_box(wall, lambda m, w=wall: draw_box(w, tex[w.texture]))
        for lintel in world.lintels:
            self._add_box(lintel, lambda m, l=lintel: draw_box(l, tex[l.texture]))
        for prop in world.props:
            kind = getattr(prop, "kind", "box")
            if kind in self.UNDRAWN:
                continue                                              # (solid, but drawn by what it stands for)
            drawer = structures.get(kind)
            if drawer is not None:
                if kind in self.ANIMATED_STRUCTURES:
                    continue                                          # drawn live (flames)
                self._add_box(prop, lambda m, p=prop, d=drawer: d(p, tex, m, 0.0))
            elif "table" in prop.name.split("_"):
                self._add_box(prop, lambda m, p=prop: draw_table(p, tex[self.prop_texture_key(p)]))
            else:
                self._add_box(prop, lambda m, p=prop: draw_box(p, tex[self.prop_texture_key(p)],
                                                               tile=None if "bed" in p.name else 1.0))
        for container in world.containers:
            kind = getattr(container, "kind", "box")
            if kind in self.ANIMATED_CONTAINERS:
                continue                                              # drawn live (a treasure that shines)
            if kind in CONTAINER_DRAWERS:
                self._add_box(container, lambda m, c=container, d=CONTAINER_DRAWERS[kind]: d(c, tex, m, 0.0))
                continue
            if kind == "table" or "table" in container.name.split("_"):
                self._add_box(container, lambda m, c=container: draw_table(c, tex[c.texture]))
                continue
            key = "shelf" if "shelf" in container.name else container.texture
            if key in self.FRONTED_TEXTURES:
                self._add_box(container, lambda m, c=container, k=key: draw_fronted_box(
                    c, tex[k], tex[self.FRONTED_TEXTURES[k]], self.room_direction(world, c)))
            else:
                self._add_box(container, lambda m, c=container, k=key: draw_box(c, tex[k], tile=None))
        for station in world.stations:
            drawer = structures.get(getattr(station, "kind", "box"))
            if drawer is not None:
                self._add_box(station, lambda m, s=station, d=drawer: d(s, tex, m, 0.0))
            else:
                self._add_box(station, lambda m, s=station: draw_box(s, tex[s.texture], tile=1.5 if s.is_mine else None))
        for readable in getattr(world, "readables", ()):
            drawer = READABLE_DRAWERS.get(readable.kind)
            if drawer is not None:
                self._add_box(readable, lambda m, r=readable, d=drawer: d(r, tex, m, 0.0))
        for fixture in world.fixtures:
            fx, fy, fz = fixture.position
            chunk = self._chunk_for(fx, fz, fx, fz, fy - 1.0, fy + 1.0)
            chunk.solid_draws.append(lambda m, f=fixture: draw_fixture_body(f, tex, m))
        for building in level.buildings:
            (x1, z1), (x2, z2) = building.min_xz, building.max_xz
            base = world.building_elevation(building) if hasattr(world, "building_elevation") else 0.0
            chunk = self._chunk_for(x1, z1, x2, z2, base, base + building.wall_height + getattr(building, "roof_height", 0.0) + 1.0)

            def draw_building(m, b=building, y=world.building_elevation(building) if hasattr(world, "building_elevation") else 0.0):
                glPushMatrix()
                glTranslatef(0.0, y, 0.0)                              # on its pad
                if not b.open_sides:
                    draw_floor(b, tex[b.floor_texture])
                if not getattr(b, "open_top", False):                  # (a tower: its platform is a prop, the sky its roof)
                    draw_ceiling(b, tex[b.floor_texture])
                    draw_roof(b, tex[b.roof_texture], tex["wood"])
                glPopMatrix()
            chunk.roof_draws.append(draw_building)
        # (the chunks are baked as they are first wanted - see ``draw`` - not all at once here: the
        # town came up seconds sooner for it)

    # --- drawing ------------------------------------------------------------------------------
    NEAR_BAKE = 45.0             # chunks this close are baked the moment they are wanted, budget or no budget

    def draw(self, material: Material, eye, limit: float, roofs: bool = True, frustum=None,
             bake_budget: Optional[float] = None) -> Tuple[int, int]:
        """Draw the ground, the roads and every chunk with any part within ``limit`` of the eye
        and - given a ``frustum`` - in view (on a wide draw distance, most of the town is behind
        the camera or off to the side: no need to hand the card what it will only clip away).

        Chunks are baked the first time they are wanted. Without a ``bake_budget`` every wanted
        chunk is baked there and then (the first frame: a whole picture, however long it takes);
        with one, the nearest are baked at once, the rest nearest first for as long as the budget
        (seconds) lasts - a far one not yet baked is left out for a frame or two - and whatever
        budget is left bakes the chunks out of view, ready for when the player turns round.
        Returns (objects drawn, objects culled) for the performance overlay."""
        import time as _time
        started = _time.perf_counter()
        material(**DEFAULT_MATERIAL)
        self.ground.draw(material)
        wanted = [chunk for chunk in self.chunks.values() if chunk.within(eye, limit) and chunk.in_view(frustum)]
        visible = []
        if any(not chunk.baked for chunk in wanted):
            wanted.sort(key=lambda c: c.distance(eye))
        for chunk in wanted:
            if not chunk.baked:
                if (bake_budget is not None and chunk.distance(eye) > self.NEAR_BAKE
                        and _time.perf_counter() - started >= bake_budget):
                    continue                                 # (next frame)
                chunk.bake()
            visible.append(chunk)
        if bake_budget is not None and _time.perf_counter() - started < bake_budget:
            self._bake_ahead(eye, started + bake_budget)
        for chunk in visible:
            if chunk.ground is not None:                 # the lie of the land first: the paving is laid over it
                chunk.ground.draw(material)
        self.roads.draw(material)
        drawn = 0
        for chunk in visible:
            chunk.solid.draw(material)
            if roofs and chunk.roofs is not None:
                chunk.roofs.draw(material)
            drawn += chunk.objects
        return drawn, sum(chunk.objects for chunk in self.chunks.values()) - drawn

    def _bake_ahead(self, eye, deadline: float) -> None:
        """Bake chunks not yet wanted, nearest first, until ``deadline`` (a ``perf_counter`` time)."""
        import time as _time
        pending = getattr(self, "_unbaked", None)
        if pending is None:
            pending = self._unbaked = [c for c in self.chunks.values() if not c.baked]
            pending.sort(key=lambda c: c.distance(eye))       # (from where the player first stood)
        while pending and _time.perf_counter() < deadline:
            chunk = pending.pop(0)
            if not chunk.baked:
                chunk.bake()

    def bake_all(self) -> None:
        """Bake every chunk now (a benchmark measures drawing, not baking)."""
        for chunk in self.chunks.values():
            if not chunk.baked:
                chunk.bake()

    @property
    def fully_baked(self) -> bool:
        return all(chunk.baked for chunk in self.chunks.values())

    def release(self) -> None:
        for compiled in (self.ground, self.roads):
            if compiled is not None:
                compiled.release()
        for chunk in self.chunks.values():
            chunk.release()
        self.chunks.clear()
        self.ground = self.roads = None
