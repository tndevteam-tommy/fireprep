"""The lie of the land: a heightfield under the whole level.

A level's ``TerrainDef`` describes rolling ground - gaussian hills and hollows over a base
height, a little seeded noise, streams that cut their beds downhill and lakes that fill a
basin. On top of that, every building's footprint (and every ``FlatDef`` - a gate, a square)
is a **pad**: perfectly level inside, the ground easing back to its natural lie over
``pad_margin`` around it, so a wall never floats over a dip nor sinks into a rise and a floor
is never crossed by the ground.

Everything a level places is authored in *local* coordinates - the floor of a building, the
foot of a tree, at y = 0 - and ``World`` lifts it by ``height(x, z)`` when the level is built.

The field is computed once on a grid (``cell`` units) with numpy; ``height`` answers exactly
on a pad and interpolates bilinearly elsewhere. A level with no ``TerrainDef`` is flat at 0,
as every level was before there was terrain.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

XZ = Tuple[float, float]


@dataclass(frozen=True)
class HillDef:
    """A gaussian rise (``height`` > 0) or hollow (< 0) centred on (x, z), ``radius`` its spread.

    A mountain is shaped by two more knobs: ``sharpness`` - the profile is
    ``height * exp(-0.5 * (d / radius) ** sharpness)``, 2 being the gaussian and less a peak
    that draws up to a point over long flanks - and ``crags``, the share of its height that
    ridged noise (``TerrainDef.crag_scale``) adds or cuts: ribs and gullies down its sides and
    a broken summit instead of a smooth dome."""
    x: float
    z: float
    height: float
    radius: float
    sharpness: float = 2.0
    crags: float = 0.0

    def profile(self, d2):
        """The share of ``height`` standing at squared distance ``d2`` from the centre (scalar or array)."""
        if self.sharpness == 2.0:
            q = d2 / (2.0 * self.radius * self.radius)
        else:
            q = 0.5 * (d2 / (self.radius * self.radius)) ** (self.sharpness / 2.0)
        return np.exp(-np.minimum(q, 60.0)) if isinstance(q, np.ndarray) else (math.exp(-q) if q < 60.0 else 0.0)


@dataclass(frozen=True)
class FlatDef:
    """An area levelled like a building's floor (a gate, a market square), at ``height`` or at
    the natural ground at its centre. A building may stand on a named flat's level
    (``BuildingDef.pad_with``): the houses round a square, the inn by a bridge."""
    min_xz: XZ
    max_xz: XZ
    height: Optional[float] = None
    margin: Optional[float] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class StreamDef:
    """A stream along ``points`` (x, z), flowing from the first to the last: its water stands
    ``bank_drop`` below the natural ground at each point but never rises downstream, in a bed
    ``width`` wide and ``water_depth`` deep with banks ``bank`` wide. ``flow`` is the drift of the
    water's surface in units per second (the renderer scrolls it)."""
    name: str
    points: Tuple[XZ, ...]
    width: float = 2.4
    bank: float = 1.8
    bank_drop: float = 0.85
    water_depth: float = 0.45
    flow: float = 0.6


@dataclass(frozen=True)
class LakeDef:
    """A lake: an ellipse of water at ``level`` over a basin ``depth`` deep, with a sloping shore."""
    name: str
    center: XZ
    radius_x: float
    radius_z: float
    level: float
    depth: float = 2.5
    shore: float = 0.35            # fraction of the radius over which the shore rises out of the water


@dataclass(frozen=True)
class TrailDef:
    """A path beaten (or cut) across the land along ``points`` (x, z): the ground under it is
    graded - cut through rises, filled over dips - so that it never climbs or falls faster than
    ``max_grade`` (rise over run), levelled across its ``width`` and easing back to the natural
    lie over ``shoulder`` either side. A trail may be pinned to a height at either end
    (``start_height`` / ``end_height``: a cave floor it leads down to). It is drawn as a ribbon of
    ``texture`` with soft edges."""
    name: str
    points: Tuple[XZ, ...]
    width: float = 2.6
    texture: str = "dirt"
    max_grade: float = 0.16
    shoulder: float = 3.0
    start_height: Optional[float] = None
    end_height: Optional[float] = None


@dataclass(frozen=True)
class SeaDef:
    """The sea along one side of the land: ``coast`` is the waterline, points (x, z) running
    north to south (increasing z) with the sea to the east (+x) of it. The ground shelves gently
    into the water, sand ``beach`` deep above the waterline; the water stands at ``level`` and
    stretches away to the horizon."""
    coast: Tuple[XZ, ...]
    level: float = -0.8
    beach: float = 14.0
    shelf: float = 0.05            # the slope of the sand and of the sea bed beyond the waterline
    depth: float = 3.0             # how deep the sea bed falls away offshore


@dataclass(frozen=True)
class TerrainDef:
    base: float = 0.0
    hills: Tuple[HillDef, ...] = ()
    noise_amplitude: float = 0.3
    noise_scale: float = 14.0          # world units per noise lattice cell
    seed: int = 5
    pad_margin: float = 4.0            # over how far the ground eases back from a levelled pad
    flats: Tuple[FlatDef, ...] = ()
    streams: Tuple[StreamDef, ...] = ()
    lakes: Tuple[LakeDef, ...] = ()
    cell: float = 1.0                  # grid resolution of the computed field
    skirt: float = 0.0                 # ground computed (and drawn) this far beyond the level's bounds all round
    trails: Tuple[TrailDef, ...] = ()
    sea: Optional[SeaDef] = None
    # rectangles (x1, z1, x2, z2) where the ground is not drawn: the mouth of a cave, where the
    # hillside rises straight up over the way in
    holes: Tuple[Tuple[float, float, float, float], ...] = ()
    rock_slope: float = 0.9            # the ground shows bare rock where it is steeper than about this ...
    snow_line: Optional[float] = None  # ... and snow above this height
    crag_scale: float = 9.0            # world units per cell of the ridged noise on craggy hills (``HillDef.crags``)


@dataclass
class Pad:
    name: str
    min_xz: XZ
    max_xz: XZ
    height: float
    margin: float
    underground: bool = False      # a cave's floor, deep under the hillside: the ground above is left alone
    cover: float = 0.0             # (underground) how high over the floor the hillside must stand at the least

    def contains(self, x: float, z: float) -> bool:
        return self.min_xz[0] <= x <= self.max_xz[0] and self.min_xz[1] <= z <= self.max_xz[1]


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


PAD_SKIRT = 1.0        # a pad is level this far beyond its outline (the mesh's vertices round a wall are on the floor)
UNDERGROUND_COVER = 1.6   # rock over a cave's ceiling (its roof slab and a good deal more)


class Terrain:
    """The computed heightfield of one level (see the module docstring). Immutable once built:
    ``shared`` hands every world built from the same level the same one."""

    _cache: dict = {}
    CACHE_SIZE = 4

    @classmethod
    def shared(cls, definition: Optional[TerrainDef], bounds: Tuple[float, float, float, float],
               buildings: Sequence = ()) -> "Terrain":
        """The terrain for this definition, bounds and building pads - computed once (a new game, a
        loaded save and every test world of the same level share it; nothing changes it)."""
        try:
            key = (definition, tuple(bounds), tuple(buildings))
            hash(key)
        except TypeError:
            return cls(definition, bounds, buildings)
        terrain = cls._cache.get(key)
        if terrain is None:
            terrain = cls(definition, bounds, buildings)
            if len(cls._cache) >= cls.CACHE_SIZE:
                cls._cache.pop(next(iter(cls._cache)))
            cls._cache[key] = terrain
        return terrain

    @classmethod
    def landform(cls, definition: TerrainDef, bounds: Tuple[float, float, float, float]) -> "Terrain":
        """Just the lie of the land - hills, noise and the sea, no grid - to ask ``natural_point``
        of while a level is still being laid out (how high a cave's mouth is before any pad is cut)."""
        t = cls.__new__(cls)
        t.definition = definition
        t.flat = False
        t.sea = definition.sea
        x1, z1, x2, z2 = bounds
        t.min_x, t.min_z, t.max_x, t.max_z = x1 - definition.skirt, z1 - definition.skirt, x2 + definition.skirt, z2 + definition.skirt
        t._noise_lattice = t._make_noise(definition)
        return t

    def natural_point(self, x: float, z: float) -> float:
        """The ground at (x, z) before any pad, trail or water touched it (the sea's shore included)."""
        return self._natural_point(x, z)

    def __init__(self, definition: Optional[TerrainDef], bounds: Tuple[float, float, float, float],
                 buildings: Sequence = ()):
        self.definition = definition
        self.min_x, self.min_z, self.max_x, self.max_z = bounds
        self.flat = definition is None
        self.pads: List[Pad] = []
        self.water: List[dict] = []          # streams: {"name", "points": [(x, z, water_y)], "width", "flow"}
        self.lakes: List[LakeDef] = []
        if self.flat:
            return
        d = definition
        self.min_x, self.min_z = self.min_x - d.skirt, self.min_z - d.skirt
        self.max_x, self.max_z = self.max_x + d.skirt, self.max_z + d.skirt
        self.cell = d.cell
        self.cols = int(math.ceil((self.max_x - self.min_x) / d.cell)) + 1
        self.rows = int(math.ceil((self.max_z - self.min_z) / d.cell)) + 1
        xs = self.min_x + np.arange(self.cols) * d.cell
        zs = self.min_z + np.arange(self.rows) * d.cell
        self._X, self._Z = np.meshgrid(xs, zs, indexing="ij")           # [col, row]
        self._noise_lattice = self._make_noise(d)
        natural = self._raw(self._X, self._Z)
        self.sea = d.sea
        if d.sea is not None:
            natural = self._shape_sea(natural, d.sea)
        self.natural = natural
        # the pads: the flats first (a building may stand on a named flat's level), then every
        # building - on its own level (the natural ground at its centre, or its ``elevation``) or
        # sharing another building's or a flat's (``pad_with``). An underground building (a cave)
        # has a pad of its own too - its floor - but leaves the hillside over it alone
        by_name = {}
        for i, f in enumerate(d.flats):
            cx, cz = (f.min_xz[0] + f.max_xz[0]) / 2.0, (f.min_xz[1] + f.max_xz[1]) / 2.0
            height = f.height if f.height is not None else self._natural_point(cx, cz)
            pad = Pad(f.name or f"flat_{i}", tuple(f.min_xz), tuple(f.max_xz), height,
                      f.margin if f.margin is not None else d.pad_margin)
            by_name[pad.name] = pad
            self.pads.append(pad)
        for b in buildings:
            share = getattr(b, "pad_with", None)
            if share and share in by_name:
                height = by_name[share].height
            elif getattr(b, "elevation", None) is not None:
                height = float(b.elevation)
            else:
                cx, cz = (b.min_xz[0] + b.max_xz[0]) / 2.0, (b.min_xz[1] + b.max_xz[1]) / 2.0
                height = self._natural_point(cx, cz)
            underground = bool(getattr(b, "underground", False))
            cover = (getattr(b, "wall_height", 4.0) + UNDERGROUND_COVER) if underground else 0.0
            pad = Pad(b.name, tuple(b.min_xz), tuple(b.max_xz), height, d.pad_margin, underground, cover)
            by_name[b.name] = pad
            self.pads.append(pad)
        h = self._level_pads(natural)
        # trails: graded into the land, one after another (a trail that starts from another starts
        # at the height the first left there); the floors they pass keep their level
        self.trails: List[dict] = []
        for trail in d.trails:
            h = self._grade_trail(h, trail)
        if d.trails:
            h = self._restore_pads(h)
        # water: streams carve their beds downhill; lakes fill basins (the cells they cut are no
        # longer any pad's floor - a stream through a square runs in its own channel)
        before = h
        for stream in d.streams:
            h = self._carve_stream(h, stream)
        for lake in d.lakes:
            h = self._carve_lake(h, lake)
            self.lakes.append(lake)
        self._carved = h < before - 1e-6
        # the hillside over a cave always clears its ceiling
        h = self._cover_underground(h)
        self.grid = h
        self.holes = tuple(tuple(float(v) for v in hole) for hole in d.holes)
        # which pad (if any) covers each cell, for exact answers on the level floors: its index, -1
        # for none (or a stream's channel through it), -2 where outlines meet (look them all up)
        pad_at = np.full(h.shape, -1, dtype=np.int32)
        count = np.zeros(h.shape, dtype=np.int32)
        for i, pad in enumerate(self.pads):
            c1, r1 = self._cell_index(pad.min_xz[0], pad.min_xz[1])
            c2, r2 = self._cell_index(pad.max_xz[0], pad.max_xz[1])
            pad_at[max(0, c1 - 1):c2 + 2, max(0, r1 - 1):r2 + 2] = i
            count[max(0, c1 - 1):c2 + 2, max(0, r1 - 1):r2 + 2] += 1
        pad_at[count > 1] = -2
        near_water = self._carved.copy()                 # a cell any of whose corners the water cut
        near_water[:-1, :] |= self._carved[1:, :]
        near_water[:, :-1] |= self._carved[:, 1:]
        near_water[:-1, :-1] |= self._carved[1:, 1:]
        pad_at[near_water] = -1
        self._pad_at = pad_at
        self._pad_rows = pad_at.tolist()                 # (scalar lookups read plain lists: far quicker)
        self._grid_rows = h.tolist()
        self._materials(d)

    def _level_pads(self, natural):
        """The ground with every pad levelled: exactly the pad's height over it and for
        ``PAD_SKIRT`` round it, easing back to the natural lie over its margin. Where two pads'
        margins meet, each pulls by how near it is (inverse-distance weights), so the ground
        between two buildings at different levels runs smoothly from one to the other whatever
        order they are listed in."""
        levelled = [pad for pad in self.pads if not pad.underground]
        if not levelled:
            return natural.copy()
        num = np.zeros(natural.shape)
        den = np.zeros(natural.shape)
        pull = np.zeros(natural.shape)
        inside = []
        for pad in levelled:
            # (only the cells within its margin: beyond, its pull is exactly nothing - and working it
            # out over the whole map for every building took most of a second)
            window = self._window(pad, pad.margin + PAD_SKIRT + self.cell)
            dist = np.maximum(self._rect_distance(self._X[window], self._Z[window], pad) - PAD_SKIRT, 0.0)
            influence = 1.0 - _smoothstep(dist / max(pad.margin, 1e-6))
            weight = influence / (dist * dist + 1e-4)
            num[window] += weight * pad.height
            den[window] += weight
            pull[window] = np.maximum(pull[window], influence)
            inside.append((window, dist <= 0.0))
        target = np.where(den > 0.0, num / np.maximum(den, 1e-12), natural)
        h = natural + (target - natural) * pull
        for pad, (window, mask) in zip(levelled, inside):
            h[window] = np.where(mask, pad.height, h[window])     # exact on the floor and its skirt
        self._pad_masks = list(zip(levelled, inside))
        return h

    def _window(self, pad: Pad, reach: float) -> Tuple[slice, slice]:
        """The grid's [col, row] slices covering ``pad`` and ``reach`` round it."""
        c1 = max(0, int(math.floor((pad.min_xz[0] - reach - self.min_x) / self.cell)))
        c2 = min(self.cols, int(math.ceil((pad.max_xz[0] + reach - self.min_x) / self.cell)) + 1)
        r1 = max(0, int(math.floor((pad.min_xz[1] - reach - self.min_z) / self.cell)))
        r2 = min(self.rows, int(math.ceil((pad.max_xz[1] + reach - self.min_z) / self.cell)) + 1)
        return (slice(c1, c2), slice(r1, r2))

    def _restore_pads(self, h):
        """Every levelled floor back at its exact height (a trail graded past one may have nudged
        its skirt)."""
        h = h.copy()
        for pad, (window, mask) in getattr(self, "_pad_masks", ()):
            h[window] = np.where(mask, pad.height, h[window])
        return h

    # ------------------------------------------------------------------ the sea
    @staticmethod
    def _coast_x(sea: SeaDef, z):
        pts = sorted(sea.coast, key=lambda p: p[1])
        return np.interp(z, [p[1] for p in pts], [p[0] for p in pts])

    def coast_x(self, z):
        """The waterline's x at ``z`` (the sea lies east of it); the ends carry straight on."""
        return self._coast_x(self.sea, z)

    def _shape_sea(self, h, sea: SeaDef):
        """Shelve the ground into the sea: sand sloping ``shelf`` down to the waterline, the sea
        bed on down to ``depth`` offshore, the land behind the beach easing back to its own lie."""
        return self._sea_profile(h, self._X, self._Z, sea)

    def _sea_profile(self, h, x, z, sea: SeaDef):
        d = np.asarray(x, dtype=float) - self._coast_x(sea, np.asarray(z, dtype=float))   # > 0: out to sea
        target = np.where(d < 0.0, sea.level - d * sea.shelf, np.maximum(sea.level - d * sea.shelf, sea.level - sea.depth))
        # the beach and the sea bed follow the profile exactly; behind the beach the land eases back over 12 units
        pull = 1.0 - _smoothstep((-d - sea.beach) / 12.0)
        return h + (target - h) * pull

    def sea_distance(self, x: float, z: float) -> Optional[float]:
        """How far (east-west) out to sea (x, z) lies: positive offshore, negative inland; None without a sea."""
        if self.sea is None:
            return None
        return float(x - self.coast_x(z))

    def _raw_scalar(self, x: float, z: float) -> float:
        """``_raw`` for one point, without numpy (a level asks it thousands of times while it is laid out)."""
        d = self.definition
        h = d.base
        ridge = None
        for hill in d.hills:
            dx, dz = x - hill.x, z - hill.z
            if hill.sharpness == 2.0 and not hill.crags:
                q = (dx * dx + dz * dz) / (2.0 * hill.radius * hill.radius)
                if q < 30.0:
                    h += hill.height * math.exp(-q)
                continue
            share = hill.profile(dx * dx + dz * dz)
            if share < 1e-9:
                continue
            if hill.crags:
                if ridge is None:
                    ridge = float(self._ridges(np.array(x), np.array(z)))
                share *= 1.0 + hill.crags * ridge
            h += hill.height * share
        if d.noise_amplitude != 0.0:
            fx = (x - self.min_x) / d.noise_scale
            fz = (z - self.min_z) / d.noise_scale
            ix, iz = int(math.floor(fx)), int(math.floor(fz))
            tx, tz = fx - ix, fz - iz
            sx, sz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)
            lat = self._noise_lattice
            ix = min(max(ix, 0), lat.shape[0] - 2)
            iz = min(max(iz, 0), lat.shape[1] - 2)
            a, b = lat[ix, iz], lat[ix + 1, iz]
            c, e = lat[ix, iz + 1], lat[ix + 1, iz + 1]
            h += d.noise_amplitude * ((a + (b - a) * sx) * (1 - sz) + (c + (e - c) * sx) * sz)
        return float(h)

    def _natural_point(self, x: float, z: float) -> float:
        h = self._raw_scalar(float(x), float(z))
        if self.definition is not None and self.definition.sea is not None:
            sea = self.definition.sea
            d = float(x) - float(self._coast_x(sea, float(z)))
            target = sea.level - d * sea.shelf if d < 0.0 else max(sea.level - d * sea.shelf, sea.level - sea.depth)
            t = min(max((-d - sea.beach) / 12.0, 0.0), 1.0)
            pull = 1.0 - t * t * (3.0 - 2.0 * t)
            h = h + (target - h) * pull
        return float(h)

    # ------------------------------------------------------------------ trails
    def _sample(self, h, x: float, z: float) -> float:
        """``h`` (a grid) at (x, z), bilinearly."""
        fx = (x - self.min_x) / self.cell
        fz = (z - self.min_z) / self.cell
        c = min(max(int(math.floor(fx)), 0), self.cols - 2)
        r = min(max(int(math.floor(fz)), 0), self.rows - 2)
        tx, tz = min(max(fx - c, 0.0), 1.0), min(max(fz - r, 0.0), 1.0)
        a, b, e, f = h[c, r], h[c + 1, r], h[c, r + 1], h[c + 1, r + 1]
        return float((a + (b - a) * tx) * (1 - tz) + (e + (f - e) * tx) * tz)

    @staticmethod
    def resample(points: Sequence[XZ], step: float = 1.0) -> List[XZ]:
        """A polyline's points every ``step`` along it (its corners kept)."""
        out = [tuple(points[0])]
        for (x1, z1), (x2, z2) in zip(points, points[1:]):
            n = max(1, int(math.ceil(math.hypot(x2 - x1, z2 - z1) / step)))
            for i in range(1, n + 1):
                out.append((x1 + (x2 - x1) * i / n, z1 + (z2 - z1) * i / n))
        return out

    @staticmethod
    def grade_profile(heights: Sequence[float], spacing: Sequence[float], max_grade: float,
                      start: Optional[float] = None, end: Optional[float] = None, smooth: int = 4) -> List[float]:
        """The trail's levels: the ground along it, smoothed, then held to ``max_grade`` between
        every two points (``spacing[i]`` from point i to i + 1) - from the pinned end back and from
        the start forward, so a pinned end is reached exactly and the rest follows the land as
        closely as the grade allows."""
        n = len(heights)
        smoothed = []
        for i in range(n):
            lo, hi = max(0, i - smooth), min(n, i + smooth + 1)
            smoothed.append(sum(heights[lo:hi]) / (hi - lo))
        p = list(smoothed)
        if start is not None:
            p[0] = start
        if end is not None:
            p[-1] = end
        # forward from the start, then back from the end: after the backward pass every step is within the grade
        for i in range(1, n):
            step = max_grade * spacing[i - 1]
            if i == n - 1 and end is not None:
                break
            p[i] = min(max(p[i], p[i - 1] - step), p[i - 1] + step)
        for i in range(n - 2, -1, -1):
            step = max_grade * spacing[i]
            if i == 0 and start is not None:
                break
            p[i] = min(max(p[i], p[i + 1] - step), p[i + 1] + step)
        return p

    def _grade_trail(self, h, trail: TrailDef):
        pts = self.resample(trail.points, 1.0)
        spacing = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
        ground = [self._sample(h, x, z) for x, z in pts]
        levels = self.grade_profile(ground, spacing, trail.max_grade, trail.start_height, trail.end_height)
        self.trails.append({"name": trail.name, "points": [(x, z, y) for (x, z), y in zip(pts, levels)],
                            "width": trail.width, "texture": trail.texture, "shoulder": trail.shoulder})
        half = trail.width / 2.0 + 0.5                    # level a little beyond the trodden edge
        reach = half + trail.shoulder
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        c1, r1 = self._cell_index(min(xs) - reach, min(zs) - reach)
        c2, r2 = self._cell_index(max(xs) + reach, max(zs) + reach)
        c1, r1 = max(c1, 0), max(r1, 0)
        c2, r2 = min(c2 + 2, self.cols), min(r2 + 2, self.rows)
        X, Z = self._X[c1:c2, r1:r2], self._Z[c1:c2, r1:r2]
        best_d = np.full(X.shape, np.inf)
        best_y = np.zeros(X.shape)
        for i in range(len(pts) - 1):
            (x1, z1), (x2, z2) = pts[i], pts[i + 1]
            sx, sz = x2 - x1, z2 - z1
            length2 = max(sx * sx + sz * sz, 1e-9)
            t = np.clip(((X - x1) * sx + (Z - z1) * sz) / length2, 0.0, 1.0)
            dist = np.sqrt((X - (x1 + t * sx)) ** 2 + (Z - (z1 + t * sz)) ** 2)
            closer = dist < best_d
            best_d = np.where(closer, dist, best_d)
            best_y = np.where(closer, levels[i] + (levels[i + 1] - levels[i]) * t, best_y)
        pull = 1.0 - _smoothstep((best_d - half) / max(trail.shoulder, 1e-6))
        block = h[c1:c2, r1:r2]
        h = h.copy()
        h[c1:c2, r1:r2] = block + (best_y - block) * pull
        return h

    # ------------------------------------------------------------------ caves
    def _cover_underground(self, h):
        """Raise the hillside wherever it would not clear an underground floor's ceiling (with
        room to spare): a cave's roof never pokes out of the grass. Only ever raises."""
        pads = [p for p in self.pads if p.underground]
        if not pads:
            return h
        h = h.copy()
        for pad in pads:
            need = pad.height + pad.cover
            inside = ((self._X > pad.min_xz[0]) & (self._X < pad.max_xz[0]) &
                      (self._Z > pad.min_xz[1]) & (self._Z < pad.max_xz[1]))
            h = np.where(inside & (h < need), need, h)
        return h

    def underground_at(self, x: float, z: float) -> Optional[Pad]:
        """The cave floor (x, z) lies over, if any."""
        for pad in self.pads:
            if pad.underground and pad.contains(x, z):
                return pad
        return None

    def in_hole(self, x: float, z: float) -> bool:
        return any(x1 <= x <= x2 and z1 <= z <= z2 for x1, z1, x2, z2 in self.holes)

    # ------------------------------------------------------------------ what the ground is made of
    def _materials(self, d: TerrainDef) -> None:
        """Per vertex, how much bare rock (steep ground), sand (the beach) and snow (the peaks)
        shows through the grass: 0 .. 1 weights the renderer blends the textures by."""
        h = self.grid
        gx = np.zeros(h.shape)
        gz = np.zeros(h.shape)
        gx[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2.0 * self.cell)
        gz[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2.0 * self.cell)
        slope = np.sqrt(gx * gx + gz * gz)
        self.slope = slope
        self.rock = np.clip((slope - d.rock_slope * 0.7) / (d.rock_slope * 0.6), 0.0, 1.0)
        if d.sea is not None:
            dist = self._X - self.coast_x(self._Z)
            self.sand = np.clip(1.0 - (-dist - d.sea.beach) / 5.0, 0.0, 1.0) * (h < d.sea.level + d.sea.shelf * d.sea.beach + 1.2)
            self.sand = self.sand.astype(float)
        else:
            self.sand = np.zeros(h.shape)
        if d.snow_line is not None:
            self.snow = np.clip((h - d.snow_line) / 6.0, 0.0, 1.0) * (1.0 - 0.3 * self.rock)      # (it clings to the crags too)
        else:
            self.snow = np.zeros(h.shape)

    def walk_grid(self):
        """The ground people walk on, per vertex: the drawn ground, except over a cave, where it is
        the cave's floor (what ``height`` answers there)."""
        walk = getattr(self, "_walk", None)
        if walk is None:
            walk = self.grid.copy()
            taken = np.zeros(walk.shape, dtype=bool)
            for pad in self.pads:
                if pad.underground:
                    inside = ((self._X >= pad.min_xz[0]) & (self._X <= pad.max_xz[0]) &
                              (self._Z >= pad.min_xz[1]) & (self._Z <= pad.max_xz[1]))
                    # (on the line where two chambers meet, the higher floor: the steps go down from there)
                    walk = np.where(inside & taken, np.maximum(walk, pad.height), np.where(inside, pad.height, walk))
                    taken |= inside
            self._walk = walk
        return walk

    def walk_slope(self):
        """How steep the walking ground is over each *cell* [c, c+1] x [r, r+1] (the steepest of its
        four edges, rise over run; the last row and column, past the grid, are 0)."""
        slope = getattr(self, "_walk_slope", None)
        if slope is None:
            w = self.walk_grid()
            slope = np.zeros(w.shape)
            dx = np.abs(np.diff(w, axis=0)) / self.cell           # [c, r]: the edge (c, r) - (c + 1, r)
            dz = np.abs(np.diff(w, axis=1)) / self.cell           # [c, r]: the edge (c, r) - (c, r + 1)
            slope[:-1, :-1] = np.maximum(np.maximum(dx[:, :-1], dx[:, 1:]), np.maximum(dz[:-1, :], dz[1:, :]))
            self._walk_slope = slope
        return slope

    def slope_at(self, x: float, z: float) -> float:
        """How steep the ground is at (x, z) (rise over run)."""
        if self.flat:
            return 0.0
        c, r = self._cell_index(x, z)
        c, r = min(max(c, 0), self.cols - 1), min(max(r, 0), self.rows - 1)
        return float(self.slope[c, r])

    def pad_conflicts(self, tolerance: float = 0.02) -> List[Tuple[str, str]]:
        """Pairs of pads whose level skirts overlap at different heights (one would cut into the
        other's floor) - a level error; the tests keep this empty."""
        out = []
        for i, a in enumerate(self.pads):
            for b in self.pads[i + 1:]:
                if abs(a.height - b.height) <= tolerance or a.underground or b.underground:
                    continue                                  # (caves' floors meet at steps, and level nothing)
                if (a.min_xz[0] - PAD_SKIRT < b.max_xz[0] + PAD_SKIRT and b.min_xz[0] - PAD_SKIRT < a.max_xz[0] + PAD_SKIRT
                        and a.min_xz[1] - PAD_SKIRT < b.max_xz[1] + PAD_SKIRT and b.min_xz[1] - PAD_SKIRT < a.max_xz[1] + PAD_SKIRT):
                    out.append((a.name, b.name))
        return out

    # ------------------------------------------------------------------ the natural ground
    def _make_noise(self, d: TerrainDef):
        rng = random.Random(d.seed)
        nx = int(math.ceil((self.max_x - self.min_x) / d.noise_scale)) + 3
        nz = int(math.ceil((self.max_z - self.min_z) / d.noise_scale)) + 3
        return np.array([[rng.uniform(-1.0, 1.0) for _ in range(nz)] for _ in range(nx)])

    def _noise(self, x, z):
        d = self.definition
        if d.noise_amplitude == 0.0:
            return np.zeros_like(np.asarray(x, dtype=float))
        fx = (np.asarray(x, dtype=float) - self.min_x) / d.noise_scale
        fz = (np.asarray(z, dtype=float) - self.min_z) / d.noise_scale
        ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
        tx, tz = fx - ix, fz - iz
        sx, sz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)
        lat = self._noise_lattice
        ix, iz = np.clip(ix, 0, lat.shape[0] - 2), np.clip(iz, 0, lat.shape[1] - 2)
        a, b = lat[ix, iz], lat[ix + 1, iz]
        c, e = lat[ix, iz + 1], lat[ix + 1, iz + 1]
        return d.noise_amplitude * ((a + (b - a) * sx) * (1 - sz) + (c + (e - c) * sx) * sz)

    def _raw(self, x, z):
        d = self.definition
        x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        if getattr(self, "_X", None) is not None and x is self._X and z is self._Z:
            return self._raw_grid()
        h = np.full(np.broadcast(x, z).shape, d.base, dtype=float)
        ridge = None
        for hill in d.hills:
            if hill.sharpness == 2.0 and not hill.crags:
                h = h + hill.height * np.exp(-((x - hill.x) ** 2 + (z - hill.z) ** 2) / (2.0 * hill.radius ** 2))
                continue
            share = hill.profile(np.asarray((x - hill.x) ** 2 + (z - hill.z) ** 2, dtype=float))
            if hill.crags:
                if ridge is None:
                    ridge = self._ridges(x, z)
                share = share * (1.0 + hill.crags * ridge)
            h = h + hill.height * share
        return h + self._noise(x, z)

    def _raw_grid(self):
        """``_raw`` over the whole grid, each hill worked out only where it stands (within
        ``sqrt(60)`` radii: past that it adds less than e^-30 of its height) - the mountains are
        dozens of hills, and the grid a couple of hundred thousand points."""
        d = self.definition
        X, Z = self._X, self._Z
        h = np.full(X.shape, d.base, dtype=float)
        crag_windows = [self._disc_window(hill.x, hill.z, hill.radius * math.sqrt(60.0)) for hill in d.hills if hill.crags]
        ridge, origin = None, (0, 0)
        if crag_windows:                                   # (the ridged noise, over the craggy hills' ground only)
            c1 = min(w[0].start for w in crag_windows)
            c2 = max(w[0].stop for w in crag_windows)
            r1 = min(w[1].start for w in crag_windows)
            r2 = max(w[1].stop for w in crag_windows)
            ridge, origin = self._ridges(X[c1:c2, r1:r2], Z[c1:c2, r1:r2]), (c1, r1)
        for hill in d.hills:
            win = self._disc_window(hill.x, hill.z, hill.radius * math.sqrt(60.0))
            x, z = X[win], Z[win]
            if x.size == 0:
                continue
            if hill.sharpness == 2.0 and not hill.crags:
                h[win] += hill.height * np.exp(-((x - hill.x) ** 2 + (z - hill.z) ** 2) / (2.0 * hill.radius ** 2))
                continue
            share = hill.profile(np.asarray((x - hill.x) ** 2 + (z - hill.z) ** 2, dtype=float))
            if hill.crags:
                oc, orr = origin
                share = share * (1.0 + hill.crags * ridge[win[0].start - oc:win[0].stop - oc, win[1].start - orr:win[1].stop - orr])
            h[win] += hill.height * share
        return h + self._noise(X, Z)

    def _disc_window(self, x: float, z: float, reach: float) -> Tuple[slice, slice]:
        c1 = max(0, int(math.floor((x - reach - self.min_x) / self.cell)))
        c2 = max(c1, min(self.cols, int(math.ceil((x + reach - self.min_x) / self.cell)) + 1))
        r1 = max(0, int(math.floor((z - reach - self.min_z) / self.cell)))
        r2 = max(r1, min(self.rows, int(math.ceil((z + reach - self.min_z) / self.cell)) + 1))
        return (slice(c1, c2), slice(r1, r2))

    RIDGE_OCTAVES = ((1.0, 0.0, 0.0, 0.6), (2.13, 17.3, 5.9, 0.3), (4.41, 3.7, 29.1, 0.1))   # (frequency, offset x, z, weight)

    def _ridges(self, x, z):
        """Ridged noise in [-1, 1] (``1 - 2|n|`` over a few octaves): sharp crests, rounded
        hollows - the ribs and gullies of a mountainside."""
        lat = getattr(self, "_ridge_lattice", None)
        if lat is None:
            lat = self._ridge_lattice = self._make_ridge_lattice(self.definition)
        fx0 = (np.asarray(x, dtype=float) - self.min_x) / self.definition.crag_scale
        fz0 = (np.asarray(z, dtype=float) - self.min_z) / self.definition.crag_scale
        out = 0.0
        for freq, ox, oz, weight in self.RIDGE_OCTAVES:
            fx, fz = fx0 * freq + ox, fz0 * freq + oz
            ix, iz = np.floor(fx).astype(int), np.floor(fz).astype(int)
            tx, tz = fx - ix, fz - iz
            sx, sz = tx * tx * (3 - 2 * tx), tz * tz * (3 - 2 * tz)
            ix, iz = np.clip(ix, 0, lat.shape[0] - 2), np.clip(iz, 0, lat.shape[1] - 2)
            a, b = lat[ix, iz], lat[ix + 1, iz]
            c, e = lat[ix, iz + 1], lat[ix + 1, iz + 1]
            n = (a + (b - a) * sx) * (1 - sz) + (c + (e - c) * sx) * sz
            out = out + weight * (1.0 - 2.0 * np.abs(n))
        return out

    def _make_ridge_lattice(self, d: TerrainDef):
        rng = random.Random(d.seed * 7919 + 1)
        top = max(freq for freq, _, _, _ in self.RIDGE_OCTAVES)
        span = max(o for _, ox, oz, _ in self.RIDGE_OCTAVES for o in (ox, oz))
        nx = int(math.ceil((self.max_x - self.min_x) / d.crag_scale * top + span)) + 3
        nz = int(math.ceil((self.max_z - self.min_z) / d.crag_scale * top + span)) + 3
        return np.array([[rng.uniform(-1.0, 1.0) for _ in range(nz)] for _ in range(nx)])

    def natural_height(self, x: float, z: float) -> float:
        """The ground before any pad, stream or lake touched it."""
        if self.flat:
            return 0.0
        return float(self._raw(np.array(x), np.array(z)))

    @staticmethod
    def _rect_distance(x, z, pad: Pad):
        dx = np.maximum(np.maximum(pad.min_xz[0] - x, 0.0), x - pad.max_xz[0])
        dz = np.maximum(np.maximum(pad.min_xz[1] - z, 0.0), z - pad.max_xz[1])
        return np.sqrt(dx * dx + dz * dz)

    # ------------------------------------------------------------------ water
    def _carve_stream(self, h, stream: StreamDef):
        pts = list(stream.points)
        levels = []
        for i, (x, z) in enumerate(pts):
            w = float(self._raw(np.array(x), np.array(z))) - stream.bank_drop
            if levels:
                w = min(w, levels[-1] - 0.02)             # water never runs uphill
            levels.append(w)
        self.water.append({"name": stream.name, "points": [(x, z, w) for (x, z), w in zip(pts, levels)],
                           "width": stream.width, "bank": stream.bank, "flow": stream.flow})
        best_d = np.full(h.shape, np.inf)
        best_w = np.zeros(h.shape)
        for (x1, z1), (x2, z2), w1, w2 in zip(pts, pts[1:], levels, levels[1:]):
            sx, sz = x2 - x1, z2 - z1
            length2 = max(sx * sx + sz * sz, 1e-9)
            t = np.clip(((self._X - x1) * sx + (self._Z - z1) * sz) / length2, 0.0, 1.0)
            px, pz = x1 + t * sx, z1 + t * sz
            dist = np.sqrt((self._X - px) ** 2 + (self._Z - pz) ** 2)
            closer = dist < best_d
            best_d = np.where(closer, dist, best_d)
            best_w = np.where(closer, w1 + (w2 - w1) * t, best_w)
        half = stream.width / 2.0
        bed = best_w - stream.water_depth
        pull = 1.0 - _smoothstep((best_d - half) / stream.bank)          # 1 in the bed, 0 beyond the banks
        carved = h - (h - bed) * pull
        return np.where((pull > 0.0) & (h > bed), carved, h)

    def _carve_lake(self, h, lake: LakeDef):
        cx, cz = lake.center
        r = np.sqrt(((self._X - cx) / lake.radius_x) ** 2 + ((self._Z - cz) / lake.radius_z) ** 2)
        inside = lake.level - lake.depth * np.clip(1.0 - r * r, 0.0, 1.0) - 0.15
        t = _smoothstep((r - 1.0) / lake.shore)                          # 0 at the waterline, 1 up the shore
        shore = (lake.level - 0.15) + (h - (lake.level - 0.15)) * t
        target = np.where(r < 1.0, inside, shore)
        return np.where(h > target, target, h)

    def water_level_at(self, x: float, z: float) -> Optional[float]:
        """The water surface over (x, z) if it lies in a stream's bed or a lake (else None)."""
        if self.flat:
            return None
        for lake in self.lakes:
            if ((x - lake.center[0]) / lake.radius_x) ** 2 + ((z - lake.center[1]) / lake.radius_z) ** 2 <= 1.0:
                return lake.level
        if self.sea is not None and x > float(self.coast_x(z)):
            return self.sea.level
        bounds = getattr(self, "_water_bounds", None)
        if bounds is None or len(bounds) != len(self.water):     # (each stream's box: most of the map is nowhere near one)
            bounds = self._water_bounds = [(min(p[0] for p in w["points"]) - w["width"], min(p[1] for p in w["points"]) - w["width"],
                                            max(p[0] for p in w["points"]) + w["width"], max(p[1] for p in w["points"]) + w["width"])
                                           for w in self.water]
        for water, (bx1, bz1, bx2, bz2) in zip(self.water, bounds):
            if not (bx1 <= x <= bx2 and bz1 <= z <= bz2):
                continue
            pts = water["points"]
            for (x1, z1, w1), (x2, z2, w2) in zip(pts, pts[1:]):
                sx, sz = x2 - x1, z2 - z1
                t = max(0.0, min(1.0, ((x - x1) * sx + (z - z1) * sz) / max(sx * sx + sz * sz, 1e-9)))
                if math.hypot(x - (x1 + t * sx), z - (z1 + t * sz)) <= water["width"] / 2.0:
                    return w1 + (w2 - w1) * t
        return None

    # ------------------------------------------------------------------ queries
    def _cell_index(self, x: float, z: float):
        return (int(math.floor((x - self.min_x) / self.cell)), int(math.floor((z - self.min_z) / self.cell)))

    def pad_at(self, x: float, z: float) -> Optional[Pad]:
        if self.flat:
            return None
        c, r = int(math.floor((x - self.min_x) / self.cell)), int(math.floor((z - self.min_z) / self.cell))
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            return None
        index = self._pad_rows[c][r]
        if index == -1:
            return None                                  # no pad here (or a stream's channel through one)
        if index >= 0:
            pad = self.pads[index]
            return pad if pad.contains(x, z) else None
        for pad in self.pads:                        # (a cell under more than one pad's outline)
            if pad.contains(x, z):
                return pad
        return None

    def height(self, x: float, z: float) -> float:
        """Ground height at (x, z): exactly a pad's on a pad, elsewhere on the triangles of the
        drawn mesh (not a bilinear patch: paving laid on that dipped under the grass here and there)."""
        if self.flat:
            return 0.0
        pad = self.pad_at(x, z)
        if pad is not None:
            return pad.height
        fx = (x - self.min_x) / self.cell
        fz = (z - self.min_z) / self.cell
        c = min(max(int(math.floor(fx)), 0), self.cols - 2)
        r = min(max(int(math.floor(fz)), 0), self.rows - 2)
        tx, tz = min(max(fx - c, 0.0), 1.0), min(max(fz - r, 0.0), 1.0)
        col, nxt = self._grid_rows[c], self._grid_rows[c + 1]
        # the two triangles each cell is drawn as (``rendering.terrain``), split along the diagonal
        # from (c, r) to (c + 1, r + 1): what stands at this height stands on the drawn ground
        h00, h10, h01, h11 = col[r], nxt[r], col[r + 1], nxt[r + 1]
        if tz >= tx:
            return h00 + (h11 - h01) * tx + (h01 - h00) * tz
        return h00 + (h10 - h00) * tx + (h11 - h10) * tz

    def heights_at(self, xs, zs) -> np.ndarray:
        """``height`` for many points at once, the same answer as point by point (the mesh's
        triangles, a pad's exact level on a pad) - without a Python call per point."""
        xs, zs = np.asarray(xs, dtype=float), np.asarray(zs, dtype=float)
        if self.flat:
            return np.zeros(xs.shape)
        fx = (xs - self.min_x) / self.cell
        fz = (zs - self.min_z) / self.cell
        c = np.clip(np.floor(fx).astype(int), 0, self.cols - 2)
        r = np.clip(np.floor(fz).astype(int), 0, self.rows - 2)
        tx, tz = np.clip(fx - c, 0.0, 1.0), np.clip(fz - r, 0.0, 1.0)
        g = self.grid
        h00, h10, h01, h11 = g[c, r], g[c + 1, r], g[c, r + 1], g[c + 1, r + 1]
        out = np.where(tz >= tx, h00 + (h11 - h01) * tx + (h01 - h00) * tz, h00 + (h10 - h00) * tx + (h11 - h10) * tz)
        pc, pr = np.floor(fx).astype(int), np.floor(fz).astype(int)
        inside = (pc >= 0) & (pc < self.cols) & (pr >= 0) & (pr < self.rows)
        idx = np.full(xs.shape, -1, dtype=np.int32)
        idx[inside] = self._pad_at[pc[inside], pr[inside]]
        for i in np.unique(idx[idx >= 0]):
            pad = self.pads[i]
            mask = (idx == i) & (xs >= pad.min_xz[0]) & (xs <= pad.max_xz[0]) & (zs >= pad.min_xz[1]) & (zs <= pad.max_xz[1])
            out = np.where(mask, pad.height, out)
        for k in np.flatnonzero(idx == -2):                  # (where pads meet: ask them one by one)
            out.flat[k] = self.height(float(xs.flat[k]), float(zs.flat[k]))
        return out

    def heights(self, xs: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """``height`` for many points at once (the renderer's mesh)."""
        return np.vectorize(self.height, otypes=[float])(xs, zs)

    def surface(self, x: float, z: float) -> float:
        """The *drawn* ground at (x, z) - the mesh's triangles, paying no heed to pads (over a cave
        it is the hillside, not the floor beneath)."""
        if self.flat:
            return 0.0
        fx = (x - self.min_x) / self.cell
        fz = (z - self.min_z) / self.cell
        c = min(max(int(math.floor(fx)), 0), self.cols - 2)
        r = min(max(int(math.floor(fz)), 0), self.rows - 2)
        tx, tz = min(max(fx - c, 0.0), 1.0), min(max(fz - r, 0.0), 1.0)
        col, nxt = self._grid_rows[c], self._grid_rows[c + 1]
        h00, h10, h01, h11 = col[r], nxt[r], col[r + 1], nxt[r + 1]
        if tz >= tx:
            return h00 + (h11 - h01) * tx + (h01 - h00) * tz
        return h00 + (h10 - h00) * tx + (h11 - h10) * tz

    def grid_max(self, x1: float, z1: float, x2: float, z2: float) -> float:
        """The highest *drawn* ground over a rectangle (the hillside, not a cave floor beneath it)."""
        if self.flat:
            return 0.0
        c1 = min(max(int(math.floor((x1 - self.min_x) / self.cell)), 0), self.cols - 1)
        c2 = min(max(int(math.ceil((x2 - self.min_x) / self.cell)), 0), self.cols - 1)
        r1 = min(max(int(math.floor((z1 - self.min_z) / self.cell)), 0), self.rows - 1)
        r2 = min(max(int(math.ceil((z2 - self.min_z) / self.cell)), 0), self.rows - 1)
        return float(self.grid[c1:c2 + 1, r1:r2 + 1].max())

    def range_over(self, x1: float, z1: float, x2: float, z2: float, step: float = 1.0) -> Tuple[float, float]:
        """(lowest, highest) ground over a rectangle: the extremes of the heightfield's vertices
        round it, which bound every height inside it (the field is interpolated between them).
        (``step`` is kept for callers; the grid's own resolution is what counts.)"""
        if self.flat:
            return 0.0, 0.0
        c1 = min(max(int(math.floor((x1 - self.min_x) / self.cell)), 0), self.cols - 1)
        c2 = min(max(int(math.ceil((x2 - self.min_x) / self.cell)), 0), self.cols - 1)
        r1 = min(max(int(math.floor((z1 - self.min_z) / self.cell)), 0), self.rows - 1)
        r2 = min(max(int(math.ceil((z2 - self.min_z) / self.cell)), 0), self.rows - 1)
        block = self.grid[c1:c2 + 1, r1:r2 + 1]
        lo, hi = float(block.min()), float(block.max())
        for x, z in ((x1, z1), (x2, z1), (x1, z2), (x2, z2)):          # (a pad's exact level at the corners)
            y = self.height(x, z)
            lo, hi = min(lo, y), max(hi, y)
        return lo, hi
