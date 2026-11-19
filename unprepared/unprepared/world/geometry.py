"""Box geometry helpers used when a level is built.

Walls authored as overlapping boxes (a house's side wall running the full length of the house
and its front wall running the full width, both covering the corner) are perfectly fine for
collision, navigation and line of sight - but the renderer then draws two faces in the same
plane over the overlap, and the depth buffer cannot tell them apart: the corners "z-fight",
showing as flickering vertical streaks on every distant building. :func:`disjoint_boxes` keeps
the union of the boxes exactly as it was while removing every overlap, so no two drawn faces
ever share a plane over a visible area.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from ..core.math3d import BoundingBox3D, Vector3D

EPS = 1e-9


def _overlaps(a: BoundingBox3D, b: BoundingBox3D) -> bool:
    return (a.min_point.x < b.max_point.x - EPS and b.min_point.x < a.max_point.x - EPS and
            a.min_point.y < b.max_point.y - EPS and b.min_point.y < a.max_point.y - EPS and
            a.min_point.z < b.max_point.z - EPS and b.min_point.z < a.max_point.z - EPS)


def subtract_box(box: BoundingBox3D, hole: BoundingBox3D) -> List[Tuple[Vector3D, Vector3D]]:
    """The parts of ``box`` outside ``hole``, as at most six disjoint (min, max) corner pairs:
    slabs below/above, then in front/behind, then left/right of the hole (clipped to the box)."""
    bmin, bmax = box.min_point, box.max_point
    if not _overlaps(box, hole):
        return [(bmin.copy(), bmax.copy())]
    hx1, hx2 = max(bmin.x, hole.min_point.x), min(bmax.x, hole.max_point.x)
    hy1, hy2 = max(bmin.y, hole.min_point.y), min(bmax.y, hole.max_point.y)
    hz1, hz2 = max(bmin.z, hole.min_point.z), min(bmax.z, hole.max_point.z)
    pieces = []
    if hy1 > bmin.y + EPS:
        pieces.append((Vector3D(bmin.x, bmin.y, bmin.z), Vector3D(bmax.x, hy1, bmax.z)))
    if hy2 < bmax.y - EPS:
        pieces.append((Vector3D(bmin.x, hy2, bmin.z), Vector3D(bmax.x, bmax.y, bmax.z)))
    if hz1 > bmin.z + EPS:
        pieces.append((Vector3D(bmin.x, hy1, bmin.z), Vector3D(bmax.x, hy2, hz1)))
    if hz2 < bmax.z - EPS:
        pieces.append((Vector3D(bmin.x, hy1, hz2), Vector3D(bmax.x, hy2, bmax.z)))
    if hx1 > bmin.x + EPS:
        pieces.append((Vector3D(bmin.x, hy1, hz1), Vector3D(hx1, hy2, hz2)))
    if hx2 < bmax.x - EPS:
        pieces.append((Vector3D(hx2, hy1, hz1), Vector3D(bmax.x, hy2, hz2)))
    return pieces


_HASH_CELL = 8.0


def disjoint_boxes(boxes: Sequence[BoundingBox3D], rebuild: Callable[[BoundingBox3D, Vector3D, Vector3D], BoundingBox3D]
                   ) -> List[BoundingBox3D]:
    """Boxes covering exactly the union of ``boxes`` with no two overlapping: each box keeps
    what earlier boxes do not already cover. ``rebuild(original, min, max)`` makes a piece of
    the same kind as its original (a wall keeps its texture)."""
    kept: List[BoundingBox3D] = []
    kept_order = {}
    grid = {}                                       # a coarse spatial hash of what is kept, by cell

    def cells(b: BoundingBox3D):
        mn, mx = b.min_point, b.max_point
        for cx in range(int(mn.x // _HASH_CELL), int(mx.x // _HASH_CELL) + 1):
            for cz in range(int(mn.z // _HASH_CELL), int(mx.z // _HASH_CELL) + 1):
                yield cx, cz

    for box in boxes:
        pieces = [box]
        candidates = {id(k): k for c in cells(box) for k in grid.get(c, ())}
        for other in sorted(candidates.values(), key=lambda k: kept_order[id(k)]):
            if not any(_overlaps(p, other) for p in pieces):
                continue
            next_pieces = []
            for piece in pieces:
                if _overlaps(piece, other):
                    next_pieces += [rebuild(box, mn, mx) for mn, mx in subtract_box(piece, other)]
                else:
                    next_pieces.append(piece)
            pieces = next_pieces
        for piece in pieces:
            kept_order[id(piece)] = len(kept)
            kept.append(piece)
            for c in cells(piece):
                grid.setdefault(c, []).append(piece)
    return kept


def _face_list(b: BoundingBox3D):
    mn, mx = b.min_point, b.max_point
    return (("x", -1, mn.x, (mn.y, mx.y), (mn.z, mx.z)), ("x", 1, mx.x, (mn.y, mx.y), (mn.z, mx.z)),
            ("y", -1, mn.y, (mn.x, mx.x), (mn.z, mx.z)), ("y", 1, mx.y, (mn.x, mx.x), (mn.z, mx.z)),
            ("z", -1, mn.z, (mn.x, mx.x), (mn.y, mx.y)), ("z", 1, mx.z, (mn.x, mx.x), (mn.y, mx.y)))


def coplanar_overlaps(boxes: Sequence[BoundingBox3D], min_area: float = 1e-4, tolerance: float = 1e-4) -> list:
    """Pairs of boxes with a face each in the same plane, facing the same way, overlapping over
    more than ``min_area`` - the geometry that z-fights. Returns (a, b, axis, direction)."""
    out = []
    for i, a in enumerate(boxes):
        fa = _face_list(a)
        for b in boxes[i + 1:]:
            if (a.max_point.x < b.min_point.x - tolerance or b.max_point.x < a.min_point.x - tolerance or
                    a.max_point.y < b.min_point.y - tolerance or b.max_point.y < a.min_point.y - tolerance or
                    a.max_point.z < b.min_point.z - tolerance or b.max_point.z < a.min_point.z - tolerance):
                continue
            for (axis, d, pa, ua, va), (_, _, pb, ub, vb) in zip(fa, _face_list(b)):
                if abs(pa - pb) < tolerance:
                    area = max(0.0, min(ua[1], ub[1]) - max(ua[0], ub[0])) * max(0.0, min(va[1], vb[1]) - max(va[0], vb[0]))
                    if area > min_area:
                        out.append((a, b, axis, d))
    return out
