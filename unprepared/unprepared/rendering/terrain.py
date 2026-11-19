"""Drawing the lie of the land: the heightfield as a textured mesh, paving draped over it, and the
water - streams that flow and lakes that ripple.

The mesh is built straight from ``Terrain.grid`` (one vertex per cell corner, normals from the
neighbouring heights) in patches, so ``batching.StaticScene`` can bake and cull it chunk by chunk.
Roads follow the same heights a hair above the grass, with a polygon offset so the two never
fight for the same depth. Water is drawn live: its texture drifts downstream with time.
"""
from __future__ import annotations

import math

from OpenGL.GL import (GL_BLEND, GL_LEQUAL, GL_LESS, GL_ONE_MINUS_SRC_ALPHA, GL_POLYGON_OFFSET_FILL, GL_QUADS, GL_SRC_ALPHA,
                       GL_TEXTURE_2D, GL_TRIANGLE_FAN, GL_TRIANGLES, glBegin, glBindTexture, glBlendFunc, glColor4f,
                       glDepthFunc, glDepthMask, glDisable, glEnable, glEnd, glNormal3f, glPolygonOffset, glTexCoord2f,
                       glVertex3f)

ROAD_LIFT = 0.03            # paving sits this far above the ground ...
OFFSET_FACTOR, OFFSET_UNITS = -1.0, -2.0    # ... and wins any remaining depth tie with it
EDGE = 0.9
TILE = 2.0


def _normal(grid, c, r, cell):
    cols, rows = grid.shape
    hl = grid[max(c - 1, 0), r]
    hr = grid[min(c + 1, cols - 1), r]
    hd = grid[c, max(r - 1, 0)]
    hu = grid[c, min(r + 1, rows - 1)]
    nx, ny, nz = (hl - hr), 2.0 * cell, (hd - hu)
    m = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / m, ny / m, nz / m


CELL_TRIANGLES = ((0, 0), (0, 1), (1, 1), (0, 0), (1, 1), (1, 0))


def terrain_normals(terrain):
    """Per-vertex normals of the whole heightfield (cached on the terrain): from the neighbouring
    heights, as ``_normal`` works them out one vertex at a time."""
    normals = getattr(terrain, "_normals", None)
    if normals is None:
        import numpy as np
        g, cell = terrain.grid, terrain.cell
        left = np.concatenate([g[:1, :], g[:-1, :]], axis=0)
        right = np.concatenate([g[1:, :], g[-1:, :]], axis=0)
        down = np.concatenate([g[:, :1], g[:, :-1]], axis=1)
        up = np.concatenate([g[:, 1:], g[:, -1:]], axis=1)
        n = np.stack([left - right, np.full(g.shape, 2.0 * cell), down - up], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        terrain._normals = normals = n
    return normals


STEEP_PROJECTION = 0.8     # a cell steeper than this takes a 'vertical' overlay's texture side-on


def _patch_arrays(terrain, cells_c, cells_r, repeat: float, weights=None, vertical: bool = False):
    """Vertex, normal, texcoord (and colour) arrays for the given cells, two triangles each
    (``CELL_TRIANGLES``). ``vertical``: a steep cell's texture is projected from the side it
    faces (x or z against height) rather than from above, so rock on a crag shows its strata
    instead of streaks smeared down the slope (every cell has its own six vertices, so cells
    projected differently don't smear into each other)."""
    import numpy as np
    g, cell = terrain.grid, terrain.cell
    normals = terrain_normals(terrain)
    offsets = np.array(CELL_TRIANGLES)
    cc = (cells_c[:, None] + offsets[None, :, 0]).ravel()
    rr = (cells_r[:, None] + offsets[None, :, 1]).ravel()
    x = terrain.min_x + cc * cell
    z = terrain.min_z + rr * cell
    y = g[cc, rr]
    verts = np.stack([x, y, z], axis=-1)
    u, v = x, z
    if vertical:
        c, r = cells_c, cells_r
        dx = np.abs(g[c + 1, r] + g[c + 1, r + 1] - g[c, r] - g[c, r + 1]) / (2.0 * cell)
        dz = np.abs(g[c, r + 1] + g[c + 1, r + 1] - g[c, r] - g[c + 1, r]) / (2.0 * cell)
        steep = np.repeat(np.maximum(dx, dz) > STEEP_PROJECTION, len(CELL_TRIANGLES))
        faces_x = np.repeat(dx > dz, len(CELL_TRIANGLES))
        u = np.where(steep, np.where(faces_x, z, x), x)
        v = np.where(steep, -y, z)
    texcoords = np.stack([u / repeat, v / repeat], axis=-1)
    colors = None
    if weights is not None:
        w = weights[cc, rr]
        colors = np.stack([np.ones_like(w), np.ones_like(w), np.ones_like(w), w], axis=-1)
    return verts, normals[cc, rr], texcoords, colors


def draw_terrain_patch(terrain, c1: int, r1: int, c2: int, r2: int, texture_id: int, repeat: float = 8.0,
                       overlays=(), material=None) -> None:
    """Cells [c1, c2) x [r1, r2) of the heightfield as lit, textured triangles (``repeat`` world
    units per texture tile), leaving out the cells in the terrain's ``holes`` (the mouth of a cave).
    ``overlays`` - (texture id, per-vertex weights, repeat[, "vertical"]) - are blended over the grass where
    their weight is above nothing: bare rock on the steeps, sand on the beach, snow on the peaks
    (``material(vertex_alpha=True)`` lets the vertex alpha fade them in)."""
    import numpy as np
    from .primitives import draw_arrays
    cell = terrain.cell
    cs, rs = np.meshgrid(np.arange(c1, c2), np.arange(r1, r2), indexing="ij")
    cs, rs = cs.ravel(), rs.ravel()
    holes = getattr(terrain, "holes", ())
    if holes:
        mx = terrain.min_x + (cs + 0.5) * cell
        mz = terrain.min_z + (rs + 0.5) * cell
        keep = np.ones(cs.shape, dtype=bool)
        for hx1, hz1, hx2, hz2 in holes:
            keep &= ~((mx >= hx1) & (mx <= hx2) & (mz >= hz1) & (mz <= hz2))
        cs, rs = cs[keep], rs[keep]
    if len(cs) == 0:
        return
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    verts, normals, texcoords, _ = _patch_arrays(terrain, cs, rs, repeat)
    draw_arrays(GL_TRIANGLES, verts, normals, texcoords)
    if overlays and material is not None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LEQUAL)                       # (the same triangles again, exactly over the grass)
        glDepthMask(False)
        material(vertex_alpha=True)
        for tex_id, weights, tile, *projection in overlays:
            touched = np.maximum(np.maximum(weights[cs, rs], weights[cs + 1, rs]),
                                 np.maximum(weights[cs, rs + 1], weights[cs + 1, rs + 1])) > 0.01
            if not touched.any():
                continue
            glBindTexture(GL_TEXTURE_2D, tex_id)
            verts, normals, texcoords, colors = _patch_arrays(terrain, cs[touched], rs[touched], tile, weights,
                                                              vertical=bool(projection and projection[0] == "vertical"))
            draw_arrays(GL_TRIANGLES, verts, normals, texcoords, colors)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        material()
        glDepthMask(True)
        glDepthFunc(GL_LESS)
        glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)


class _Batch:
    """Draped vertices gathered up and handed to GL in one go, their heights looked up all at
    once (``heights(xs, zs)``: ``Terrain.heights_at``) - paving over a map this size is tens of
    thousands of little polygons."""

    def __init__(self, heights, lift: float):
        self.heights, self.lift = heights, lift
        self.x, self.z, self.u, self.v = [], [], [], []

    def add(self, x: float, z: float, u: float, v: float) -> None:
        self.x.append(x)
        self.z.append(z)
        self.u.append(u)
        self.v.append(v)

    def flush(self) -> None:
        if not self.x:
            return
        import numpy as np
        from .primitives import draw_arrays
        xs, zs = np.array(self.x), np.array(self.z)
        ys = self.heights(xs, zs) + self.lift
        normals = np.zeros((len(xs), 3))
        normals[:, 1] = 1.0
        draw_arrays(GL_TRIANGLES, np.stack([xs, ys, zs], axis=-1), normals, np.stack([self.u, self.v], axis=-1))
        self.x, self.z, self.u, self.v = [], [], [], []


_BATCH = None


def _emit(height, x: float, z: float, u: float, v: float, lift: float) -> None:
    if _BATCH is not None:
        _BATCH.add(x, z, u, v)
        return
    glNormal3f(0.0, 1.0, 0.0)
    glTexCoord2f(u, v)
    glVertex3f(x, height(x, z) + lift, z)


def _begin(heights, lift: float) -> None:
    """Start a run of draped triangles: batched when a vectorised ``heights`` is to hand."""
    global _BATCH
    if heights is not None:
        _BATCH = _Batch(heights, lift)
    else:
        glBegin(GL_TRIANGLES)


def _end() -> None:
    global _BATCH
    if _BATCH is not None:
        _BATCH.flush()
        _BATCH = None
    else:
        glEnd()


def _draped_quad(height, x1, z1, x2, z2, u_of, v_of, lift=ROAD_LIFT):
    """One rectangle of paving as two triangles (the mesh's diagonal): for a rectangle inside one
    cell of the ground that is exactly the ground's shape, a hair above it."""
    for x, z in ((x1, z1), (x1, z2), (x2, z2), (x1, z1), (x2, z2), (x2, z1)):
        _emit(height, x, z, u_of(x, z), v_of(x, z), lift)


def _clip_half(poly, keep):
    """Sutherland-Hodgman against one half-plane ``keep(x, z) >= 0`` (a linear function)."""
    out = []
    for i, (x1, z1) in enumerate(poly):
        x2, z2 = poly[(i + 1) % len(poly)]
        d1, d2 = keep(x1, z1), keep(x2, z2)
        if d1 >= 0.0:
            out.append((x1, z1))
        if (d1 >= 0.0) != (d2 >= 0.0):
            t = d1 / (d1 - d2)
            out.append((x1 + (x2 - x1) * t, z1 + (z2 - z1) * t))
    return out


def draped_pieces(x1, z1, x2, z2, lattice):
    """The rectangle cut along the ground's cell edges *and* the diagonal each cell is split by:
    polygons each lying in one triangle of the drawn mesh, so paving draped vertex by vertex
    follows the ground exactly and never dips under the grass in the middle of a quad."""
    ox, oz, cell = lattice
    c1, c2 = int(math.floor((x1 - ox) / cell)), int(math.ceil((x2 - ox) / cell))
    r1, r2 = int(math.floor((z1 - oz) / cell)), int(math.ceil((z2 - oz) / cell))
    for c in range(c1, c2):
        cx1, cx2 = max(x1, ox + c * cell), min(x2, ox + (c + 1) * cell)
        if cx2 - cx1 <= 1e-9:
            continue
        for r in range(r1, r2):
            cz1, cz2 = max(z1, oz + r * cell), min(z2, oz + (r + 1) * cell)
            if cz2 - cz1 <= 1e-9:
                continue
            bx, bz = ox + c * cell, oz + r * cell
            rect = [(cx1, cz1), (cx1, cz2), (cx2, cz2), (cx2, cz1)]
            for sign in (1.0, -1.0):
                piece = _clip_half(rect, lambda x, z, s=sign: s * ((z - bz) - (x - bx)))
                if len(piece) >= 3:
                    yield piece


def drape_polygon(poly, lattice):
    """A convex polygon (x, z) cut into pieces each lying in one triangle of the ground's mesh
    (like ``draped_pieces``, for shapes that are not axis-aligned: the bends of a trail)."""
    ox, oz, cell = lattice
    xs, zs = [p[0] for p in poly], [p[1] for p in poly]
    c1, c2 = int(math.floor((min(xs) - ox) / cell)), int(math.ceil((max(xs) - ox) / cell))
    r1, r2 = int(math.floor((min(zs) - oz) / cell)), int(math.ceil((max(zs) - oz) / cell))
    for c in range(c1, c2):
        bx = ox + c * cell
        strip = _clip_half(_clip_half(poly, lambda x, z: x - bx), lambda x, z: bx + cell - x)
        if len(strip) < 3:
            continue
        for r in range(r1, r2):
            bz = oz + r * cell
            piece = _clip_half(_clip_half(strip, lambda x, z: z - bz), lambda x, z: bz + cell - z)
            if len(piece) < 3:
                continue
            for sign in (1.0, -1.0):
                half = _clip_half(piece, lambda x, z, s=sign: s * ((z - bz) - (x - bx)))
                if len(half) >= 3:
                    yield half


def _draped_polygon(height, piece, u_of, v_of, lift=ROAD_LIFT):
    x0, z0 = piece[0]
    for (xa, za), (xb, zb) in zip(piece[1:], piece[2:]):
        for x, z in ((x0, z0), (xa, za), (xb, zb)):
            _emit(height, x, z, u_of(x, z), v_of(x, z), lift)


EDGE_SAMPLE = 0.25          # how finely a road's sides are checked for the paving that joins them


def _covered(road, roads, x: float, z: float) -> bool:
    return any(r is not road and r.min_xz[0] <= x <= r.max_xz[0] and r.min_xz[1] <= z <= r.max_xz[1] for r in roads)


def side_intervals(road, roads, side: str, reach: float = 0.3):
    """Where along one side of a road other paving carries on beyond it: ``(a, b, joined)`` runs
    covering the whole side (along x for "north"/"south", along z for "west"/"east"). A joined run
    is laid to the edge at full strength; the rest fades into the grass. Two fading edges side by
    side left a line of grass across the street wherever one length of paving met the next - and
    a side only partly met (a lane opening off a street, a square wider than the street through
    it) used to fade all the way along, mouth of the lane included."""
    (x1, z1), (x2, z2) = road.min_xz, road.max_xz
    if side in ("north", "south"):
        a0, b0 = x1, x2
        probe = (lambda t: (t, z1 - reach)) if side == "north" else (lambda t: (t, z2 + reach))
    else:
        a0, b0 = z1, z2
        probe = (lambda t: (x1 - reach, t)) if side == "west" else (lambda t: (x2 + reach, t))
    n = max(1, int(math.ceil((b0 - a0) / EDGE_SAMPLE)))
    marks = []
    for i in range(n):
        t = a0 + (b0 - a0) * (i + 0.5) / n
        marks.append(_covered(road, roads, *probe(t)))
    runs = []
    start = 0
    for i in range(1, n + 1):
        if i == n or marks[i] != marks[start]:
            runs.append((a0 + (b0 - a0) * start / n, a0 + (b0 - a0) * i / n, marks[start]))
            start = i
    return runs


def _covered_sides(road, roads) -> dict:
    """Which of a road's sides are joined by other paving all the way along (kept for callers)."""
    return {side: all(joined for _, _, joined in side_intervals(road, roads, side))
            for side in ("north", "south", "west", "east")}


def draw_draped_roads(roads, textures, material, height, skip=None, step: float = 1.0, lattice=None, heights=None) -> None:
    """Paving rectangles laid over the ground in ``step``-sized quads, edges fading into the grass
    except along the runs of a side where other paving carries on (``side_intervals``).
    ``skip(x, z)`` leaves out cells over water (a bridge spans those)."""
    if not roads:
        return
    material(textured=True)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(OFFSET_FACTOR, OFFSET_UNITS)
    glDepthMask(False)
    u_x, v_z = (lambda x, z: x / TILE), (lambda x, z: z / TILE)
    for road in roads:
        (x1, z1), (x2, z2) = road.min_xz, road.max_xz
        e = min(EDGE, (x2 - x1) / 3.0, (z2 - z1) / 3.0)
        runs = {side: side_intervals(road, roads, side) for side in ("north", "south", "west", "east")}
        solid, fading = [], []          # (x1, z1, x2, z2[, fade direction]) pieces
        solid.append((x1 + e, z1 + e, x2 - e, z2 - e))
        for a, b, joined in runs["north"]:
            (solid if joined else fading).append((a, z1, b, z1 + e, "north"))
        for a, b, joined in runs["south"]:
            (solid if joined else fading).append((a, z2 - e, b, z2, "south"))
        for a, b, joined in runs["west"]:
            a, b = max(a, z1 + e), min(b, z2 - e)
            if b > a:
                (solid if joined else fading).append((x1, a, x1 + e, b, "west"))
        for a, b, joined in runs["east"]:
            a, b = max(a, z1 + e), min(b, z2 - e)
            if b > a:
                (solid if joined else fading).append((x2 - e, a, x2, b, "east"))
        glBindTexture(GL_TEXTURE_2D, textures[road.texture])
        _begin(heights, ROAD_LIFT)
        for piece in solid:
            _grid(height, *piece[:4], step, u_x, v_z, skip, lattice)
        _end()
        glBindTexture(GL_TEXTURE_2D, textures[road.texture + "_edge"])
        _begin(heights, ROAD_LIFT)
        # the fading strips: v = 0 (opaque) at the paving, 1 (clear) at the grass
        for px1, pz1, px2, pz2, side in fading:
            if side == "north":
                _grid(height, px1, pz1, px2, pz2, step, u_x, lambda x, z: (z1 + e - z) / e, skip, lattice)
            elif side == "south":
                _grid(height, px1, pz1, px2, pz2, step, u_x, lambda x, z: (z - (z2 - e)) / e, skip, lattice)
            elif side == "west":
                _grid(height, px1, pz1, px2, pz2, step, v_z, lambda x, z: (x1 + e - x) / e, skip, lattice)
            else:
                _grid(height, px1, pz1, px2, pz2, step, v_z, lambda x, z: (x - (x2 - e)) / e, skip, lattice)
        _end()
    glDepthMask(True)
    glDisable(GL_POLYGON_OFFSET_FILL)
    glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)


def _grid(height, x1, z1, x2, z2, step, u_of, v_of, skip, lattice=None):
    """Paving over a rectangle as triangles: cut to the ground's mesh (``lattice``: origin x, z and
    cell size) when there is one, else in ``step``-sized squares. ``skip(x, z)`` leaves out pieces
    whose middle it picks (over the water a bridge spans)."""
    if x2 <= x1 or z2 <= z1:
        return
    if lattice is not None:
        for piece in draped_pieces(x1, z1, x2, z2, lattice):
            if skip is not None:
                mx = sum(p[0] for p in piece) / len(piece)
                mz = sum(p[1] for p in piece) / len(piece)
                if skip(mx, mz):
                    continue
            _draped_polygon(height, piece, u_of, v_of)
        return
    nx = max(1, int(math.ceil((x2 - x1) / step)))
    nz = max(1, int(math.ceil((z2 - z1) / step)))
    for i in range(nx):
        for j in range(nz):
            a1, a2 = x1 + (x2 - x1) * i / nx, x1 + (x2 - x1) * (i + 1) / nx
            b1, b2 = z1 + (z2 - z1) * j / nz, z1 + (z2 - z1) * (j + 1) / nz
            if skip is not None and skip((a1 + a2) / 2.0, (b1 + b2) / 2.0):
                continue
            _draped_quad(height, a1, b1, a2, b2, u_of, v_of)


# ----------------------------------------------------------------------------- trails
TRAIL_EDGE = 0.8            # the soft margin each side of a trail where the dirt gives way to grass
TRAIL_LIFT = 0.025


def trail_edges(points, half: float):
    """Left and right offsets at every point of a polyline (x, z, ...), on the bisector of the two
    segments meeting there so the ribbon keeps its width round a bend."""
    out = []
    n = len(points)
    for i in range(n):
        x, z = points[i][0], points[i][1]
        ax, az = (x - points[i - 1][0], z - points[i - 1][1]) if i > 0 else (0.0, 0.0)
        bx, bz = (points[i + 1][0] - x, points[i + 1][1] - z) if i + 1 < n else (0.0, 0.0)
        la, lb = math.hypot(ax, az) or 1.0, math.hypot(bx, bz) or 1.0
        tx, tz = ax / la + bx / lb, az / la + bz / lb
        lt = math.hypot(tx, tz) or 1.0
        tx, tz = tx / lt, tz / lt
        cos_half = max(0.5, (tx * (bx / lb) + tz * (bz / lb)) if i + 1 < n else 1.0)
        px, pz = -tz * half / cos_half, tx * half / cos_half
        out.append(((x - px, z - pz), (x + px, z + pz)))
    return out


def draw_trails(trails, textures, material, height, lattice, heights=None) -> None:
    """Every graded trail as a ribbon of its texture draped over the ground, its edges fading into
    the grass, its ends too (so where it meets a road or another trail the two run together)."""
    if not trails:
        return
    material(textured=True)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(OFFSET_FACTOR, OFFSET_UNITS)
    glDepthMask(False)
    for trail in trails:
        pts = trail["points"]
        if len(pts) < 2:
            continue
        half = trail["width"] / 2.0
        inner = trail_edges(pts, half - TRAIL_EDGE)
        outer = trail_edges(pts, half)
        along = [0.0]
        for a, b in zip(pts, pts[1:]):
            along.append(along[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        world_uv = (lambda x, z: x / TILE), (lambda x, z: z / TILE)
        glBindTexture(GL_TEXTURE_2D, textures[trail["texture"]])
        _begin(heights, TRAIL_LIFT)
        for i in range(len(pts) - 1):
            (l1, r1), (l2, r2) = inner[i], inner[i + 1]
            for piece in drape_polygon([l1, l2, r2, r1] if _ccw([l1, l2, r2, r1]) else [r1, r2, l2, l1], lattice):
                _draped_polygon(height, piece, *world_uv, lift=TRAIL_LIFT)
        _end()
        glBindTexture(GL_TEXTURE_2D, textures[trail["texture"] + "_edge"])
        _begin(heights, TRAIL_LIFT)
        for i in range(len(pts) - 1):
            x0, z0 = pts[i][0], pts[i][1]
            dx, dz = pts[i + 1][0] - x0, pts[i + 1][1] - z0
            seg = math.hypot(dx, dz) or 1.0
            ux, uz = dx / seg, dz / seg

            def u_of(x, z, i=i, x0=x0, z0=z0, ux=ux, uz=uz):
                return (along[i] + (x - x0) * ux + (z - z0) * uz) / TILE

            def v_of(x, z, x0=x0, z0=z0, ux=ux, uz=uz):
                across = abs((x - x0) * -uz + (z - z0) * ux)
                return min(1.0, max(0.0, (across - (half - TRAIL_EDGE)) / TRAIL_EDGE))

            for side in (0, 1):
                a1, a2 = inner[i][side], inner[i + 1][side]
                b1, b2 = outer[i][side], outer[i + 1][side]
                quad = [a1, a2, b2, b1]
                if not _ccw(quad):
                    quad = quad[::-1]
                for piece in drape_polygon(quad, lattice):
                    _draped_polygon(height, piece, u_of, v_of, lift=TRAIL_LIFT)
        _end()
    glDepthMask(True)
    glDisable(GL_POLYGON_OFFSET_FILL)
    glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)


def _ccw(poly) -> bool:
    area = 0.0
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        area += x1 * z2 - x2 * z1
    return area > 0.0


# ----------------------------------------------------------------------------- water
WATER_COLOR = (0.55, 0.72, 0.85)
WATER_GLOW = (0.02, 0.05, 0.07)


# each sheet of water is drawn twice, the second time fainter, finer and faster: two sets of
# ripples running through each other read as moving water, not a picture sliding along
WATER_LAYERS = ((0.8, 1.0, 1.0, 0.0), (0.32, 1.63, 1.7, 0.37))    # (alpha, uv scale, speed, u offset)


def draw_water(terrain, textures, material, time: float, shader=None) -> None:
    """Every stream as a ribbon of running water at its level, the ripples drifting downstream,
    and every lake as a broad sheet of wavelets stirring in the wind. Drawn with blending (the bed
    shows a little) and without writing depth, the second layer over the first."""
    sea = getattr(terrain, "sea", None)
    if terrain.flat or not (terrain.water or terrain.lakes or sea is not None):
        return
    material(color=WATER_COLOR, emission=WATER_GLOW, textured=True)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDepthMask(False)
    glDepthFunc(GL_LEQUAL)                    # (the second layer lies exactly on the first)
    for alpha, scale, speed, offset in WATER_LAYERS:
        if shader is not None:
            shader.set_alpha(alpha)
        else:
            glColor4f(*WATER_COLOR, alpha)
        glBindTexture(GL_TEXTURE_2D, textures["stream"])
        for water in terrain.water:
            _draw_stream(_clipped_to_lakes(water, terrain.lakes), time, scale, speed, offset)
        glBindTexture(GL_TEXTURE_2D, textures["lake"])
        for lake in terrain.lakes:
            _draw_lake(lake, time, scale, speed, offset)
        if sea is not None:
            _draw_sea(sea, time, scale, speed, offset)
    if shader is not None:
        shader.set_alpha(1.0)
    glDepthFunc(GL_LESS)
    glDepthMask(True)
    glDisable(GL_BLEND)
    glDisable(GL_TEXTURE_2D)
    material()


def stream_strip(water):
    """The stream's surface as one continuous strip: for each point (x, z, level) the left and right
    edges, turned on the bisector of the two segments meeting there (no gaps or overlaps at bends)."""
    pts = water["points"]
    half = water["width"] / 2.0 + water["bank"] * 0.55      # out to where the banks rise above the surface
    edges = []
    for i, (x, z, w) in enumerate(pts):
        ax, az = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) if i > 0 else (0.0, 0.0)
        bx, bz = (pts[i + 1][0] - x, pts[i + 1][1] - z) if i + 1 < len(pts) else (0.0, 0.0)
        la, lb = math.hypot(ax, az) or 1.0, math.hypot(bx, bz) or 1.0
        tx, tz = ax / la + bx / lb, az / la + bz / lb
        lt = math.hypot(tx, tz) or 1.0
        tx, tz = tx / lt, tz / lt
        # keep the strip's width across a bend: lengthen the offset by 1 / cos(half the turn)
        cos_half = max(0.35, (tx * (bx / lb) + tz * (bz / lb)) if i + 1 < len(pts) else 1.0)
        px, pz = -tz * half / cos_half, tx * half / cos_half
        edges.append(((x - px, z - pz), (x + px, z + pz), w))
    return edges, half


LAKE_SHEET = 1.08     # a lake's sheet of water reaches a little past its ellipse, up the shore


def _in_lake(lakes, x: float, z: float) -> bool:
    """Inside a lake's sheet of water as drawn."""
    return any(((x - lk.center[0]) / (lk.radius_x * LAKE_SHEET)) ** 2 + ((z - lk.center[1]) / (lk.radius_z * LAKE_SHEET)) ** 2
               <= 1.0 for lk in lakes)


LAKE_MOUTH_OVERLAP = 0.1     # how far the stream's ribbon runs on under the lake's sheet (no gap at its corners)


def _clipped_to_lakes(water, lakes):
    """The stream with its course cut where it first runs under a lake's sheet: the two blended
    sheets must not overlap (the overlap showed as a darker rectangle of water in the lake)."""
    pts = water["points"]
    if not lakes or len(pts) < 2:
        return water
    for i in range(1, len(pts)):
        (x1, z1, w1), (x2, z2, w2) = pts[i - 1], pts[i]
        if not _in_lake(lakes, x2, z2):
            continue
        if _in_lake(lakes, x1, z1):
            return water if i == 1 else {**water, "points": pts[:i]}
        lo, hi = 0.0, 1.0                       # the rim crossing on this segment, by bisection
        for _ in range(24):
            mid = (lo + hi) / 2.0
            if _in_lake(lakes, x1 + (x2 - x1) * mid, z1 + (z2 - z1) * mid):
                hi = mid
            else:
                lo = mid
        length = math.hypot(x2 - x1, z2 - z1) or 1.0
        t = min(1.0, hi + LAKE_MOUTH_OVERLAP / length)
        end = (x1 + (x2 - x1) * t, z1 + (z2 - z1) * t, w1 + (w2 - w1) * t)
        return {**water, "points": list(pts[:i]) + [end]}
    return water


def _draw_stream(water, time: float, scale: float = 1.0, speed: float = 1.0, offset: float = 0.0) -> None:
    edges, half = stream_strip(water)
    pts = water["points"]
    drift = time * water["flow"] * speed / TILE
    travelled = [0.0]
    for (x1, z1, _), (x2, z2, _) in zip(pts, pts[1:]):
        travelled.append(travelled[-1] + math.hypot(x2 - x1, z2 - z1))
    glBegin(GL_QUADS)
    glNormal3f(0.0, 1.0, 0.0)
    u_right = 2.0 * half / TILE * scale
    for i in range(len(edges) - 1):
        (l1, r1, w1), (l2, r2, w2) = edges[i], edges[i + 1]
        v1, v2 = travelled[i] / TILE * scale - drift, travelled[i + 1] / TILE * scale - drift
        for (x, z), w, u, v in ((l1, w1, offset, v1), (r1, w1, offset + u_right, v1), (r2, w2, offset + u_right, v2),
                                (l2, w2, offset, v2)):
            glTexCoord2f(u, v)
            glVertex3f(x, w, z)
    glEnd()


LAKE_TILE = 7.0          # world units per repeat of the lake's wavelets
SEA_TILE = 10.0          # ... and of the sea's swell
SEA_REACH = 900.0        # how far out (and along the coast, beyond its ends) the sea is drawn: to the horizon


def _draw_sea(sea, time: float, scale: float = 1.0, speed: float = 1.0, offset: float = 0.0) -> None:
    """The sea: a sheet at its level from a little inside the waterline out to the horizon, the
    swell rolling in towards the beach."""
    pts = sorted(sea.coast, key=lambda p: p[1])
    pts = [(pts[0][0], pts[0][1] - SEA_REACH)] + pts + [(pts[-1][0], pts[-1][1] + SEA_REACH)]
    tile = SEA_TILE / scale
    du = -time * 0.09 * speed + offset                   # (rolling in: westwards)
    dv = time * 0.02 * speed * (1.0 if offset == 0.0 else -1.0)
    far = max(p[0] for p in pts) + SEA_REACH
    glBegin(GL_QUADS)
    glNormal3f(0.0, 1.0, 0.0)
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        xa, xb = xa - 2.0, xb - 2.0                      # (a little inland: the sand runs under the water's edge)
        for x, z in ((xa, za), (far, za), (far, zb), (xb, zb)):
            glTexCoord2f(x / tile + du, z / tile + dv)
            glVertex3f(x, sea.level, z)
    glEnd()


def _draw_lake(lake, time: float, scale: float = 1.0, speed: float = 1.0, offset: float = 0.0, segments: int = 48) -> None:
    """The lake's sheet: its wavelets drift with the wind, the finer layer across the coarser."""
    cx, cz = lake.center
    tile = LAKE_TILE / scale
    du = time * 0.05 * speed + offset
    dv = time * 0.03 * speed * (1.0 if offset == 0.0 else -0.6)
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(0.0, 1.0, 0.0)
    glTexCoord2f(cx / tile + du, cz / tile + dv)
    glVertex3f(cx, lake.level, cz)
    for i in range(segments + 1):
        a = 2.0 * math.pi * i / segments
        x, z = cx + math.cos(a) * lake.radius_x * LAKE_SHEET, cz + math.sin(a) * lake.radius_z * LAKE_SHEET
        glTexCoord2f(x / tile + du, z / tile + dv)
        glVertex3f(x, lake.level, z)
    glEnd()
