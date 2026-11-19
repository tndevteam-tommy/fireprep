"""Ledge graph: collects rails from obstacles, clips them against neighbouring
geometry, joins them at corners and serves spatial queries for the reticle."""
import numpy as np
import config
from parkour.ledge_rail import LedgeRail, RailLink
from world.spatial import ObstacleIndex, candidates


class LedgeSystem:
    def __init__(self):
        self.rails = []
        self._arrays = None

    def add_rail(self, rail):
        self.rails.append(rail)
        self._arrays = None

    def clear(self):
        self.rails.clear()
        self._arrays = None

    def arrays(self):
        """Stacked rail geometry (p1, p2, tangent, length, normal, height) for vectorised queries."""
        if self._arrays is None or self._arrays[0] != len(self.rails):
            r = self.rails
            if not r:
                self._arrays = (0, None)
            else:
                self._arrays = (len(r), (
                    np.stack([x.p1 for x in r]).astype(np.float32),
                    np.stack([x.p2 for x in r]).astype(np.float32),
                    np.stack([x.tangent for x in r]).astype(np.float32),
                    np.array([max(x.length, 1e-6) for x in r], dtype=np.float32),
                    np.stack([x.normal for x in r]).astype(np.float32),
                    np.array([x.height for x in r], dtype=np.float32),
                ))
        return self._arrays[1]

    # ------------------------------------------------------------------ build
    def build(self, obstacles):
        """Collect, clip and link every rail of the given obstacles."""
        self.clear()
        index = obstacles if hasattr(obstacles, "query") else ObstacleIndex(obstacles)
        raw = [r for o in obstacles for r in o.rails]
        for rail in raw:
            for piece in self.clip_rail(rail, index):
                self.rails.append(piece)
        self.link_corners()
        self._arrays = None
        return self.rails

    @staticmethod
    def _slab(a, d, lo, hi):
        """Parametric t-interval where a + t*d lies inside [lo, hi] (1D)."""
        if abs(d) < 1e-9:
            return (0.0, 1.0) if lo <= a <= hi else None
        t0 = (lo - a) / d
        t1 = (hi - a) / d
        if t0 > t1:
            t0, t1 = t1, t0
        return (max(0.0, t0), min(1.0, t1))

    @classmethod
    def clip_rail(cls, rail, obstacles):
        """
        Remove the parts of `rail` that are covered by other obstacles: a part is
        covered where a probe point RAIL_COVER_PROBE outside the edge sits inside
        another box whose vertical span contains the edge height.  Returns the
        surviving pieces (new LedgeRail objects), dropping anything shorter than
        MIN_RAIL_LENGTH.
        """
        probe = rail.normal * config.RAIL_COVER_PROBE
        a = rail.p1 + probe
        d = rail.p2 - rail.p1
        y = rail.height
        intervals = [(0.0, 1.0)]
        eps = 1e-3
        lo = np.minimum(rail.p1, rail.p2) - config.RAIL_COVER_PROBE - 0.01
        hi = np.maximum(rail.p1, rail.p2) + config.RAIL_COVER_PROBE + 0.01
        for obs in candidates(obstacles, lo, hi):
            if obs is rail.owner:
                continue
            b = obs.bounds
            if not (b.min[1] - eps <= y < b.max[1] - eps):
                continue
            tx = cls._slab(float(a[0]), float(d[0]), float(b.min[0]) + eps, float(b.max[0]) - eps)
            tz = cls._slab(float(a[2]), float(d[2]), float(b.min[2]) + eps, float(b.max[2]) - eps)
            if tx is None or tz is None:
                continue
            c0, c1 = max(tx[0], tz[0]), min(tx[1], tz[1])
            if c1 <= c0:
                continue
            new = []
            for s0, s1 in intervals:
                if c1 <= s0 or c0 >= s1:
                    new.append((s0, s1))
                    continue
                if c0 > s0:
                    new.append((s0, c0))
                if c1 < s1:
                    new.append((c1, s1))
            intervals = new
            if not intervals:
                break
        pieces = []
        for s0, s1 in intervals:
            if (s1 - s0) * rail.length + 1e-6 < config.MIN_RAIL_LENGTH:
                continue
            if s0 <= 1e-6 and s1 >= 1.0 - 1e-6:
                pieces.append(rail)
            else:
                pieces.append(LedgeRail(rail.get_point(s0), rail.get_point(s1), rail.normal,
                                        rail.rail_type, rail.thickness, owner=rail.owner))
        return pieces

    def link_corners(self):
        """
        Join rails whose endpoints meet at the same height.  Moving "right" along
        rail R (toward p2) continues onto Q when Q.p1 == R.p2; the corner is
        convex when Q's wall faces along R's tangent (the edge wraps away around
        the block) and concave when it faces back (the edge continues on a wall
        beside / behind the player).  Collinear edges simply continue straight.
        """
        tol = config.CORNER_LINK_TOL
        cell = 0.1

        def key(p):
            return (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)), int(np.floor(p[2] / cell)))

        by_p1 = {}
        by_p2 = {}
        for r in self.rails:
            r.links = {}
            by_p1.setdefault(key(r.p1), []).append(r)
            by_p2.setdefault(key(r.p2), []).append(r)

        def near(table, p):
            kx, ky, kz = key(p)
            out = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        out.extend(table.get((kx + dx, ky + dy, kz + dz), ()))
            return out

        for r in self.rails:
            for q in near(by_p1, r.p2):
                if q is r or "right" in r.links:
                    continue
                if float(np.linalg.norm(q.p1 - r.p2)) <= tol:
                    kind = RailLink.classify(r, q)
                    if kind is not None:
                        r.links["right"] = RailLink(q, "right", kind)
            for q in near(by_p2, r.p1):
                if q is r or "left" in r.links:
                    continue
                if float(np.linalg.norm(q.p2 - r.p1)) <= tol:
                    kind = RailLink.classify(q, r)
                    if kind is not None:
                        r.links["left"] = RailLink(q, "left", kind)

    # ---------------------------------------------------------------- queries
    def find_best_rail(self, eye_pos, forward_dir, max_dist=config.RETICLE_MAX_DIST, cone_deg=config.DETECTION_CONE_DEG):
        """
        Searches all rails for the best candidate within the view cone (vectorised).
        Returns:
            best_rail, pt_on_rail, dist, s_param
            (or None, None, float('inf'), 0.0 if none found)
        """
        arrays = self.arrays()
        if arrays is None:
            return None, None, float('inf'), 0.0
        eye = np.asarray(eye_pos, dtype=np.float32)
        fwd = np.asarray(forward_dir, dtype=np.float32)
        fwd = fwd / (np.linalg.norm(fwd) + 1e-7)
        # only the edges near the eye (a uniform grid over the rails - parkour/body.py)
        from parkour.body import _rails_near
        near = _rails_near(self.rails, eye, max_dist + 0.01)
        if len(near) == 0:
            return None, None, float('inf'), 0.0
        P1, P2, T, L, N, H = (a[near] for a in arrays)

        # Closest point on each segment to the look ray (segment/line parametric solution)
        ab = P2 - P1
        w0 = P1 - eye[None, :]
        b = ab @ fwd
        c = np.einsum("ij,ij->i", ab, ab)
        d = w0 @ fwd
        e = np.einsum("ij,ij->i", ab, w0)
        denom = c - b * b
        s = np.where(np.abs(denom) < 1e-6, 0.5, (b * d - e) / np.where(np.abs(denom) < 1e-6, 1.0, denom))
        s = np.clip(s, 0.0, 1.0)
        pt_seg = P1 + ab * s[:, None]
        to_pt = pt_seg - eye[None, :]
        dist_eye = np.linalg.norm(to_pt, axis=1)
        ok = (dist_eye <= max_dist) & (dist_eye >= 0.1)
        dir_to = to_pt / np.maximum(dist_eye, 1e-6)[:, None]
        ok &= np.einsum("ij,ij->i", dir_to, -N) >= 0.1
        cos_angle = np.clip(dir_to @ fwd, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        max_angle = np.radians(cone_deg)
        ok &= angle <= max_angle
        if not np.any(ok):
            return None, None, float('inf'), 0.0
        t = np.maximum(0.0, np.einsum("ij,j->i", to_pt, fwd))
        pt_ray = eye[None, :] + fwd[None, :] * t[:, None]
        line_dist = np.linalg.norm(pt_seg - pt_ray, axis=1)
        score = line_dist + 1.8 * (angle / max_angle)
        score = np.where(ok, score, np.inf)
        i = int(np.argmin(score))
        return self.rails[int(near[i])], pt_seg[i].copy(), float(dist_eye[i]), float(s[i])
