"""Looking for what would look wrong on screen - without a screen.

The static scene is drawn through a stand-in for OpenGL (:class:`GeometryRecorder`): every
``glBegin`` / ``glVertex`` / ``draw_arrays`` the drawers issue, through every ``glTranslatef`` and
``glRotatef``, comes out as world-space triangles, each tagged with what drew it (a wall, a prop
by name, a building's roof, a fixture) and with the state it was drawn in (a texture, blending,
polygon offset). :func:`audit_world` then looks for the glitches that have bitten this game:

* **z-fighting** (``zfight``): two triangles of different things in the same plane, facing the
  same way, overlapping - the depth buffer cannot order them and they flicker into stripes
  (``NEAR`` apart is as bad: at a distance the depth buffer cannot tell them apart either).
  Geometry drawn with a polygon offset or blended over what is under it is meant to lie flush
  and is left out.
* **poking through walls** (``through_wall``): a solid prop, container or station that goes
  right through a wall and out of its far face (a shelf seen from the street).
* **the ground through a floor** (``ground_in_floor``): the terrain standing above a building's
  floor inside its walls.
* **floating** (``floating``): something standing on the ground outdoors with daylight under it
  all round.
* **decals off their wall** (``decal_off_wall``): a window, a painting or a sign not lying on a
  wall face, or hanging past its end.

``tools/visual_audit.py`` prints the report; ``tests/unit/test_visual_audit.py`` keeps it empty.
No GL context is needed: nothing is drawn, the GL functions the rendering modules imported are
swapped for the recorder's while it runs.
"""
from __future__ import annotations

import importlib
import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from OpenGL import GL as _GL

Vec3 = Tuple[float, float, float]

RENDER_MODULES = ("primitives", "structures", "furnishings", "buildings", "models", "gear", "roads", "terrain", "batching")

NEAR = 0.003            # faces closer than this (metres) fight in the depth buffer at a distance, as good as coplanar
MIN_OVERLAP = 2e-3      # square metres of overlap worth reporting (a sliver of a corner is not a glitch anyone sees)


# ----------------------------------------------------------------------------- recording
@dataclass
class Triangle:
    owner: str                      # what drew it: "wall", "prop:shop_shelf_west", "roof:chapel", "fixture:lantern"
    a: Vec3
    b: Vec3
    c: Vec3
    texture: Optional[str]
    offset: bool = False            # drawn with a polygon offset (a decal meant to lie flush)
    blended: bool = False           # blended over what is under it (an overlay)

    @property
    def normal(self) -> Optional[np.ndarray]:
        a, b, c = np.array(self.a), np.array(self.b), np.array(self.c)
        n = np.cross(b - a, c - a)
        length = float(np.linalg.norm(n))
        return None if length < 1e-12 else n / length

    @property
    def area(self) -> float:
        a, b, c = np.array(self.a), np.array(self.b), np.array(self.c)
        return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


class _Textures(dict):
    """Texture ids for the drawers: every key gets a number, and the number its name back."""

    def __init__(self):
        super().__init__()
        self.names: Dict[int, str] = {}

    def __missing__(self, key):
        tid = len(self.names) + 1
        self.names[tid] = key
        self[key] = tid
        return tid

    def __contains__(self, key):
        return True


class GeometryRecorder:
    """OpenGL's immediate mode as far as the drawers use it, keeping triangles instead of drawing."""

    def __init__(self):
        self.textures = _Textures()
        self.triangles: List[Triangle] = []
        self.owner = "?"
        self._stack = [np.identity(4)]
        self._mode = None
        self._verts: List[Vec3] = []
        self._texture: Optional[int] = None
        self._offset = False
        self._blend = False

    # --- the matrix -------------------------------------------------------------------
    @property
    def _m(self) -> np.ndarray:
        return self._stack[-1]

    def glPushMatrix(self):
        self._stack.append(self._m.copy())

    def glPopMatrix(self):
        if len(self._stack) > 1:
            self._stack.pop()

    def glLoadIdentity(self):
        self._stack[-1] = np.identity(4)

    def _mult(self, m: np.ndarray) -> None:
        self._stack[-1] = self._m @ m

    def glTranslatef(self, x, y, z):
        m = np.identity(4)
        m[:3, 3] = (x, y, z)
        self._mult(m)

    glTranslated = glTranslatef

    def glScalef(self, x, y, z):
        self._mult(np.diag((x, y, z, 1.0)))

    def glRotatef(self, angle, x, y, z):
        axis = np.array((x, y, z), dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return
        ux, uy, uz = axis / n
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        r = np.array([[c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s, 0],
                      [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s, 0],
                      [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c), 0],
                      [0, 0, 0, 1]])
        self._mult(r)

    def glMultMatrixf(self, m):
        self._mult(np.array(m, dtype=float).reshape(4, 4).T)

    glMultMatrixd = glMultMatrixf

    # --- state ------------------------------------------------------------------------
    def glBindTexture(self, target, tid):
        self._texture = tid

    def glEnable(self, cap):
        if cap == _GL.GL_POLYGON_OFFSET_FILL:
            self._offset = True
        elif cap == _GL.GL_BLEND:
            self._blend = True

    def glDisable(self, cap):
        if cap == _GL.GL_POLYGON_OFFSET_FILL:
            self._offset = False
        elif cap == _GL.GL_BLEND:
            self._blend = False

    # --- geometry ---------------------------------------------------------------------
    def glBegin(self, mode):
        self._mode = mode
        self._verts = []

    def glVertex3f(self, x, y, z):
        p = self._m @ np.array((x, y, z, 1.0))
        self._verts.append((float(p[0]), float(p[1]), float(p[2])))

    glVertex3d = glVertex3f

    def glVertex3fv(self, v):
        self.glVertex3f(*v)

    def glEnd(self):
        self._emit(self._mode, self._verts)
        self._mode, self._verts = None, []

    def _emit(self, mode, verts: List[Vec3]) -> None:
        tris = []
        if mode == _GL.GL_TRIANGLES:
            tris = [verts[i:i + 3] for i in range(0, len(verts) - 2, 3)]
        elif mode == _GL.GL_QUADS:
            for i in range(0, len(verts) - 3, 4):
                q = verts[i:i + 4]
                tris += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
        elif mode == _GL.GL_TRIANGLE_STRIP or mode == _GL.GL_QUAD_STRIP:
            tris = [(verts[i], verts[i + 1], verts[i + 2]) for i in range(len(verts) - 2)]
        elif mode in (_GL.GL_TRIANGLE_FAN, _GL.GL_POLYGON):
            tris = [(verts[0], verts[i], verts[i + 1]) for i in range(1, len(verts) - 1)]
        name = self.textures.names.get(self._texture) if self._texture else None
        for a, b, c in tris:
            self.triangles.append(Triangle(self.owner, a, b, c, name, self._offset, self._blend))

    def draw_arrays(self, mode, vertices, normals=None, texcoords=None, colors=None) -> None:
        verts = np.asarray(vertices, dtype=float)
        if len(verts) == 0:
            return
        world = (np.c_[verts, np.ones(len(verts))] @ self._m.T)[:, :3]
        self._emit(mode, [tuple(map(float, v)) for v in world])

    def _noop(self, *args, **kwargs):
        return 0

    # --- swapping it in ---------------------------------------------------------------
    @contextmanager
    def installed(self):
        """Every ``gl*`` function the rendering modules imported answers to the recorder (or does
        nothing) while the block runs; ``draw_arrays`` too."""
        saved = []
        for short in RENDER_MODULES:
            module = importlib.import_module(f"unprepared.rendering.{short}")
            for name in dir(module):
                if not name.startswith("gl") or not callable(getattr(module, name)):
                    continue
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, getattr(self, name, self._noop))
            if hasattr(module, "draw_arrays"):
                saved.append((module, "draw_arrays", module.draw_arrays))
                module.draw_arrays = self.draw_arrays
            if hasattr(module, "ensure_sphere_list"):
                saved.append((module, "ensure_sphere_list", module.ensure_sphere_list))
                module.ensure_sphere_list = self._noop
        try:
            yield self
        finally:
            for module, name, value in reversed(saved):
                setattr(module, name, value)


def _owner_of(box) -> str:
    from ..world.level import Prop
    kind = type(box).__name__
    name = getattr(box, "name", None)
    if isinstance(box, Prop) or name is not None:
        return f"{kind.lower()}:{name}"
    return kind.lower()


def record_static_scene(world, doors: bool = True) -> List[Triangle]:
    """The triangles of everything ``StaticScene`` bakes for ``world`` - walls, lintels, props,
    containers, stations, readables, fixtures' bodies, floors, ceilings and roofs - and, with
    ``doors``, every door's leaf shut and open; but not the terrain and the paving draped over it
    (the ground is checked as a heightfield)."""
    from . import batching
    from .scene import STRUCTURES
    rec = GeometryRecorder()
    owners: Dict[int, str] = {}
    scene = batching.StaticScene.__new__(batching.StaticScene)
    scene.world, scene.textures, scene.ground_cells, scene.chunks = world, rec.textures, 20, {}
    scene.ground = scene.roads = None
    original_add = scene._add_box

    def add_box(box, draw, roof=False):
        owners[id(draw)] = _owner_of(box)
        original_add(box, draw, roof)
    scene._add_box = add_box
    real_record, real_bake = batching.Compiled.record, batching.Chunk.bake
    batching.Compiled.record = classmethod(lambda cls, draw: None)          # (the ground and the paving: not recorded)
    batching.Chunk.bake = lambda self: None
    try:
        with rec.installed():
            scene._build(STRUCTURES)
            for chunk in scene.chunks.values():
                for draws, kind in ((chunk.solid_draws, "solid"), (chunk.roof_draws, "roof")):
                    for draw in draws:
                        rec.owner = owners.get(id(draw)) or _default_owner(draw, kind)
                        rec._stack = [np.identity(4)]
                        draw(lambda *a, **k: None)
        with rec.installed():                               # (flames, water, cloth: drawn live, here as at a moment)
            for prop in world.props:
                kind = getattr(prop, "kind", "box")
                if kind in batching.StaticScene.ANIMATED_STRUCTURES and kind in STRUCTURES:
                    rec.owner = _owner_of(prop)
                    rec._stack = [np.identity(4)]
                    STRUCTURES[kind](prop, rec.textures, lambda *a, **k: None, 0.0)
        if doors:
            from .primitives import draw_door
            with rec.installed():
                for door in world.doors:                    # (each leaf shut and open: two poses, one owner)
                    rec.owner = f"door:{door.door_id}"
                    was = door.is_open
                    for pose in (False, True):
                        door.is_open = pose
                        rec._stack = [np.identity(4)]
                        draw_door(door, rec.textures["door"], bars_texture_id=rec.textures["iron"])
                    door.is_open = was
    finally:
        batching.Compiled.record, batching.Chunk.bake = real_record, real_bake
    return rec.triangles


def _default_owner(draw, kind: str) -> str:
    """A draw routine added straight to a chunk: a fixture's body or a building's floor and roof
    (their closures carry the object as a default argument)."""
    for value in getattr(draw, "__defaults__", None) or ():
        name = getattr(value, "name", None)
        if name is not None:
            return f"{'building' if kind == 'roof' else type(value).__name__.lower()}:{name}"
        if hasattr(value, "kind") and hasattr(value, "position"):
            return f"fixture:{value.kind}@{tuple(round(v, 1) for v in value.position)}"
    return kind


# ----------------------------------------------------------------------------- the checks
@dataclass
class Finding:
    kind: str                       # "zfight", "through_wall", "ground_in_floor", "floating", "decal_off_wall"
    what: Tuple[str, ...]           # the things involved
    where: Vec3
    detail: str = ""

    def __str__(self) -> str:
        x, y, z = self.where
        return f"{self.kind:15s} ({x:7.2f}, {y:6.2f}, {z:7.2f})  {' / '.join(self.what)}  {self.detail}"


def _clip(poly: List[np.ndarray], a: np.ndarray, b: np.ndarray) -> List[np.ndarray]:
    """``poly`` (2-D, counter-clockwise) cut by the half-plane left of a->b (Sutherland-Hodgman)."""
    out = []
    if not poly:
        return out
    d = b - a

    def inside(p):
        return d[0] * (p[1] - a[1]) - d[1] * (p[0] - a[0]) >= -1e-12

    prev = poly[-1]
    for cur in poly:
        if inside(cur):
            if not inside(prev):
                out.append(_cross_point(prev, cur, a, b))
            out.append(cur)
        elif inside(prev):
            out.append(_cross_point(prev, cur, a, b))
        prev = cur
    return out


def _cross_point(p, q, a, b):
    r, s = q - p, b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-15:
        return q
    t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / denom
    return p + t * r


def _area2(poly) -> float:
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(poly, poly[1:] + poly[:1]))


def _ccw(poly):
    return poly if _area2(poly) >= 0 else poly[::-1]


def overlap(t1: Triangle, t2: Triangle, normal: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """(area, centre) of the overlap of two (near-)coplanar triangles, measured in their plane
    (the centre in world space, on the first triangle's plane); (0, None) if they do not overlap."""
    u = np.cross(normal, (1.0, 0.0, 0.0) if abs(normal[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    p1 = _ccw([np.array((np.dot(p, u), np.dot(p, v))) for p in (t1.a, t1.b, t1.c)])
    p2 = _ccw([np.array((np.dot(p, u), np.dot(p, v))) for p in (t2.a, t2.b, t2.c)])
    poly = p1
    for i in range(3):
        poly = _clip(poly, p2[i], p2[(i + 1) % 3])
        if len(poly) < 3:
            return 0.0, None
    area = abs(_area2(poly))
    cu, cv = (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    return area, cu * u + cv * v + float(np.dot(normal, t1.a)) * normal


def overlap_area(t1: Triangle, t2: Triangle, normal: np.ndarray) -> float:
    """Area of the overlap of two (near-)coplanar triangles, measured in their plane."""
    return overlap(t1, t2, normal)[0]


class Solids:
    """What is solid in a world, for telling whether a face can be seen at all: a face whose front
    (a couple of millimetres out along its normal) is inside a wall, a prop, a floor - or under
    the ground (the rock under a cave's floor, the earth under the grass) - is never seen, and
    two such faces in one plane fight over nothing."""

    def __init__(self, world):
        from ..world.spatial import BoxIndex
        boxes = [*world.walls, *getattr(world, "lintels", ()), *(p for p in world.props), *world.containers,
                 *world.stations]
        self.index = BoxIndex(boxes, cell=4.0)
        self.world = world

    def hidden(self, p, eps: float = 1e-3, ignore=None) -> bool:
        x, y, z = p
        for b in self.index.near(x, z, 0.01):
            if b is ignore:
                continue
            mn, mx = b.min_point, b.max_point
            if mn.x + eps < x < mx.x - eps and mn.y + eps < y < mx.y - eps and mn.z + eps < z < mx.z - eps:
                return True
        terrain = self.world.terrain
        if terrain.flat:
            return y < -eps
        cave = terrain.underground_at(x, z)
        if cave is not None and y < cave.height + cave.cover + 1.0:
            return y < cave.height - eps                      # (in the cave: only what is under its floor is rock)
        return y < terrain.height(x, z) - eps


def find_zfighting(triangles: Sequence[Triangle], near: float = NEAR, min_overlap: float = MIN_OVERLAP,
                   same_owner: bool = False, solids: Optional[Solids] = None) -> List[Finding]:
    """Pairs of triangles of different things (or of one thing, with ``same_owner``) lying in the
    same plane - within ``near`` - facing the same way and overlapping by more than ``min_overlap``
    where it can be seen (with ``solids``: not where the faces look into something solid or the
    ground). One finding per pair of owners and plane, with the area they share."""
    buckets: Dict[tuple, list] = defaultdict(list)
    kept = [t for t in triangles if not (t.offset or t.blended)]
    if not kept:
        return []
    A = np.array([t.a for t in kept], dtype=float)
    B = np.array([t.b for t in kept], dtype=float)
    C = np.array([t.c for t in kept], dtype=float)
    N = np.cross(B - A, C - A)
    length = np.linalg.norm(N, axis=1)
    ok = length > 2e-6                                       # (twice the area: slivers under a square millimetre are nothing)
    N[ok] /= length[ok, None]
    D = np.einsum("ij,ij->i", N, A)
    LO, HI = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
    keys = np.round(N, 2)
    dkeys = np.floor(D / near).astype(np.int64)
    for i in np.flatnonzero(ok):
        buckets[(tuple(keys[i]), int(dkeys[i]))].append((kept[i], N[i], float(D[i]), tuple(LO[i]), tuple(HI[i])))
    found: Dict[tuple, list] = {}
    cell = 2.0
    for key, items in buckets.items():
        normal_key, dk = key
        pool = items + buckets.get((normal_key, dk + 1), [])
        if len(pool) < 2:
            continue
        # (a coarse grid over the bucket: only triangles sharing a cell are compared)
        grid: Dict[tuple, List[int]] = defaultdict(list)
        for idx, (_, _, _, lo, hi) in enumerate(pool):
            for gx in range(int(math.floor(lo[0] / cell)), int(math.floor(hi[0] / cell)) + 1):
                for gy in range(int(math.floor(lo[1] / cell)), int(math.floor(hi[1] / cell)) + 1):
                    for gz in range(int(math.floor(lo[2] / cell)), int(math.floor(hi[2] / cell)) + 1):
                        grid[(gx, gy, gz)].append(idx)
        pairs = set()
        for members in grid.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    if a < len(items) or b < len(items):
                        pairs.add((a, b) if a < b else (b, a))
        for i, j in pairs:
            t1, n1, d1, lo1, hi1 = pool[i]
            t2, n2, d2, lo2, hi2 = pool[j]
            if (t1.owner == t2.owner and not same_owner) or abs(d1 - d2) > near or float(np.dot(n1, n2)) < 0.999:
                continue
            if any(hi1[k] < lo2[k] - near or hi2[k] < lo1[k] - near for k in range(3)):
                continue
            area, centre = overlap(t1, t2, n1)
            if area <= 1e-9:
                continue
            if solids is not None and solids.hidden(tuple(centre + n1 * 0.002)):
                continue
            pair = tuple(sorted((t1.owner, t2.owner))) + (normal_key, round(d1, 2))
            entry = found.setdefault(pair, [0.0, tuple(float(v) for v in centre), normal_key])
            entry[0] += area
    out = []
    for pair, (area, centre, normal_key) in found.items():
        if area > min_overlap:
            facing = {(1.0, 0.0, 0.0): "+x", (-1.0, 0.0, 0.0): "-x", (0.0, 1.0, 0.0): "up", (0.0, -1.0, 0.0): "down",
                      (0.0, 0.0, 1.0): "+z", (0.0, 0.0, -1.0): "-z"}.get(tuple(float(v) + 0.0 for v in normal_key), str(normal_key))
            out.append(Finding("zfight", pair[:2], centre, f"{area:.3f} m2 facing {facing}"))
    return out


# kinds of prop that are *meant* to be set into or across a wall (a threshold is the floor through a
# doorway, a beam or a shelf bracket is bedded in the masonry, a bridge crosses the stream's banks)
INTO_WALLS = ("floor", "step", "bridge", "window", "sign", "crack", "stain", "banner", "painting", "grate", "rug",
              "stair", "platform", "beam", "timber")


def find_through_walls(world, solids: Optional["Solids"] = None) -> List[Finding]:
    """Solid things that pass right through a wall and out of its far face - where both faces
    can be seen (a boulder through a cave's wall into the rock behind it shows on one side only)."""
    solids = solids or Solids(world)
    out = []
    things = [*(p for p in world.props if p.solid and getattr(p, "kind", "box") not in INTO_WALLS),
              *world.containers, *world.stations]
    walls = world.walls
    for thing in things:
        mn, mx = thing.min_point, thing.max_point
        for wall in world.walls_near(thing.center, max(mx.x - mn.x, mx.z - mn.z) + 2.0) if hasattr(world, "walls_near") else walls:
            wn, wx = wall.min_point, wall.max_point
            if not (mn.x < wx.x and wn.x < mx.x and mn.y < wx.y and wn.y < mx.y and mn.z < wx.z and wn.z < mx.z):
                continue
            thin_x = (wx.x - wn.x) <= (wx.z - wn.z)
            lo, hi, a, b = (wn.x, wx.x, mn.x, mx.x) if thin_x else (wn.z, wx.z, mn.z, mx.z)
            if a < lo - 1e-3 and b > hi + 1e-3:                 # out of both faces: seen on either side
                y = (max(mn.y, wn.y) + min(mx.y, wx.y)) / 2.0
                u = (max(mn.z, wn.z) + min(mx.z, wx.z)) / 2.0 if thin_x else (max(mn.x, wn.x) + min(mx.x, wx.x)) / 2.0
                sides = [((lo - 0.02, y, u) if thin_x else (u, y, lo - 0.02)), ((hi + 0.02, y, u) if thin_x else (u, y, hi + 0.02))]
                if any(solids.hidden(p, ignore=thing) for p in sides):
                    continue
                out.append(Finding("through_wall", (_owner_of(thing), "wall"), (thing.center.x, thing.center.y, thing.center.z),
                                   f"{(min(b, hi) - max(a, lo)):.2f} m of wall crossed"))
    return out


def find_ground_in_floors(world, margin: float = 0.6, step: float = 0.5, tolerance: float = 0.04) -> List[Finding]:
    """The terrain standing above a (walled, above-ground) building's floor inside its walls."""
    out = []
    terrain = world.terrain
    if terrain.flat:
        return out
    for b in world.level.buildings:
        if b.underground or b.open_sides:
            continue
        floor = world.building_elevation(b)
        (x1, z1), (x2, z2) = b.min_xz, b.max_xz
        xs = np.arange(x1 + margin, x2 - margin + 1e-9, step)
        zs = np.arange(z1 + margin, z2 - margin + 1e-9, step)
        if len(xs) == 0 or len(zs) == 0:
            continue
        X, Z = np.meshgrid(xs, zs)
        heights = terrain.heights_at(X.ravel(), Z.ravel()) if hasattr(terrain, "heights_at") else \
            np.array([terrain.height(x, z) for x, z in zip(X.ravel(), Z.ravel())])
        worst = float(np.max(heights - floor))
        if worst > tolerance:
            i = int(np.argmax(heights - floor))
            out.append(Finding("ground_in_floor", (f"building:{b.name}",), (float(X.ravel()[i]), float(heights[i]), float(Z.ravel()[i])),
                               f"the ground {worst:.2f} m over the floor"))
    return out


# things that hang (on a wall, from a beam) or sit on something else: never "floating"
HANGING = ("window", "sign", "banner", "painting", "lantern", "bell", "crack", "stain", "grate", "washing_line", "beam",
           "timber", "stalactite", "chandelier", "shelf", "platform", "floor", "step", "bridge", "roof", "rug", "brow",
           "collider")


def find_floating(world, tolerance: float = 0.08) -> List[Finding]:
    """Solid things standing outdoors with the ground below all of their footprint (daylight
    under them all round): their base higher than the ground at every corner and the middle."""
    from ..world.spatial import BoxIndex
    out = []
    terrain = world.terrain
    if terrain.flat:
        return out
    supports = BoxIndex([*world.floors, *world.props, *world.walls, *world.containers], cell=4.0)
    for thing in [*world.props, *world.containers, *world.stations]:
        kind = getattr(thing, "kind", "box")
        if kind in HANGING or any(word in thing.name for word in HANGING):
            continue
        mn, mx = thing.min_point, thing.max_point
        if world.building_at(thing.center, True) is not None or terrain.underground_at(thing.center.x, thing.center.z):
            continue
        if any(f is not thing and f.min_point.x <= thing.center.x <= f.max_point.x and f.min_point.z <= thing.center.z
               <= f.max_point.z and abs(f.max_point.y - mn.y) < 0.1
               for f in supports.near(thing.center.x, thing.center.z, 0.01)):
            continue                                            # (on a deck, a step, a plinth, an altar)
        pts = [(mn.x, mn.z), (mn.x, mx.z), (mx.x, mn.z), (mx.x, mx.z), ((mn.x + mx.x) / 2.0, (mn.z + mx.z) / 2.0)]
        ground = max(terrain.height(x, z) for x, z in pts)
        if mn.y > ground + tolerance:
            out.append(Finding("floating", (_owner_of(thing),), (thing.center.x, mn.y, thing.center.z),
                               f"{mn.y - ground:.2f} m over the ground"))
    return out


DECALS = ("window", "painting", "sign")


def find_decals_off_walls(world, depth: float = 0.05) -> List[Finding]:
    """A window, a painting or a sign flat on a wall: its back within ``depth`` of a wall face and
    all of it in front of that face. Reports the ones on no wall at all and the ones hanging past
    a wall's end or over its top."""
    out = []
    for p in world.props:
        kind = getattr(p, "kind", "")
        if kind not in ("window", "painting"):
            continue
        mn, mx = p.min_point, p.max_point
        thin_x = (mx.x - mn.x) < (mx.z - mn.z)
        a_lo, a_hi = (mn.z, mx.z) if thin_x else (mn.x, mx.x)
        best = None
        for wall in world.walls_near(p.center, 4.0):
            wn, wx = wall.min_point, wall.max_point
            faces = (wn.x, wx.x) if thin_x else (wn.z, wx.z)
            back = (mn.x, mx.x) if thin_x else (mn.z, mx.z)
            if not any(abs(f - b) <= depth for f in faces for b in back):
                continue
            w_lo, w_hi = (wn.z, wx.z) if thin_x else (wn.x, wx.x)
            if w_hi <= a_lo or w_lo >= a_hi or wx.y <= mn.y or wn.y >= mx.y:
                continue
            covered = (min(a_hi, w_hi) - max(a_lo, w_lo)) * (min(mx.y, wx.y) - max(mn.y, wn.y))
            best = covered if best is None else best + covered
        need = (a_hi - a_lo) * (mx.y - mn.y)
        if best is None:
            out.append(Finding("decal_off_wall", (_owner_of(p),), (p.center.x, p.center.y, p.center.z), "on no wall"))
        elif best < need * 0.98:
            out.append(Finding("decal_off_wall", (_owner_of(p),), (p.center.x, p.center.y, p.center.z),
                               f"{100.0 * (1.0 - best / need):.0f}% of it off the wall"))
    return out


def audit_world(world, checks: Iterable[str] = ("zfight", "through_wall", "ground_in_floor", "floating", "decal_off_wall"),
                triangles: Optional[Sequence[Triangle]] = None, near: float = NEAR) -> List[Finding]:
    """Every finding of the chosen ``checks`` for ``world`` (see the module docstring)."""
    checks = set(checks)
    out: List[Finding] = []
    solids = Solids(world)
    if "zfight" in checks:
        out += find_zfighting(triangles if triangles is not None else record_static_scene(world), near=near, solids=solids)
    if "through_wall" in checks:
        out += find_through_walls(world, solids)
    if "ground_in_floor" in checks:
        out += find_ground_in_floors(world)
    if "floating" in checks:
        out += find_floating(world)
    if "decal_off_wall" in checks:
        out += find_decals_off_walls(world)
    return out
