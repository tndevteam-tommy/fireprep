"""Parametric 3D Ledge Rail implementation."""
import numpy as np
import config
from math3d.geometry import point_segment_project


class RailType:
    HEAD_CREVASSE = 0
    JUMP_CREVASSE = 1
    MANTLE_TOP = 2
    LADDER_CREVASSE = 3   # extra intermediate crevasse generated so tall walls stay climbable

    @staticmethod
    def is_crevasse(rail_type):
        return rail_type in (RailType.HEAD_CREVASSE, RailType.JUMP_CREVASSE, RailType.LADDER_CREVASSE)


class RailLink:
    """Continuation of a rail around a corner (or straight on)."""
    CONVEX = "convex"      # edge wraps away around the block (outside corner)
    CONCAVE = "concave"    # edge continues on the wall beside / behind the player (inside corner)
    STRAIGHT = "straight"  # collinear edge of a neighbouring block

    def __init__(self, rail, side, kind):
        self.rail = rail
        self.side = side        # "left" or "right": which end of the source rail it hangs off
        self.kind = kind

    def __repr__(self):
        return f"RailLink({self.side}, {self.kind}, -> {self.rail})"

    @staticmethod
    def classify(r, q):
        """Kind of corner when travelling from r (arriving at r.p2) onto q (starting at q.p1)."""
        d = float(np.dot(q.normal, r.tangent))
        if d > 0.5:
            return RailLink.CONVEX
        if d < -0.5:
            return RailLink.CONCAVE
        if float(np.dot(q.normal, r.normal)) > 0.99:
            return RailLink.STRAIGHT
        return None

    @property
    def entry_s(self):
        """Parameter on the linked rail at the corner end (before clamping to margins)."""
        return 0.0 if self.side == "right" else 1.0

    @property
    def direction(self):
        """+1 if continuing the traverse moves toward the linked rail's p2, -1 toward p1."""
        return 1.0 if self.side == "right" else -1.0


class LedgeRail:
    def __init__(self, p1, p2, normal, rail_type=RailType.MANTLE_TOP, thickness=0.08, owner=None):
        self.p1 = np.asarray(p1, dtype=np.float32).copy()
        self.p2 = np.asarray(p2, dtype=np.float32).copy()
        self.normal = np.asarray(normal, dtype=np.float32).copy()
        self.normal /= (np.linalg.norm(self.normal) + 1e-7)
        self.rail_type = rail_type
        self.thickness = float(thickness)
        self.owner = owner           # Obstacle that generated this rail (may be None)
        self.links = {}              # "left"/"right" -> RailLink (filled by LedgeSystem.link_corners)

        diff = self.p2 - self.p1
        self.length = float(np.linalg.norm(diff))
        if self.length > 1e-6:
            self.tangent = diff / self.length
        else:
            self.tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def __repr__(self):
        return f"LedgeRail(type={self.rail_type}, y={self.p1[1]:.2f}, n={tuple(self.normal.round(1))}, len={self.length:.1f})"

    @property
    def midpoint(self):
        return (self.p1 + self.p2) * 0.5

    @property
    def height(self):
        return float(self.p1[1])

    @property
    def is_crevasse(self):
        return RailType.is_crevasse(self.rail_type)

    def get_point(self, s):
        s_clamped = float(np.clip(s, 0.0, 1.0))
        return self.p1 + s_clamped * (self.p2 - self.p1)

    def s_margin(self, margin=config.RAIL_END_MARGIN):
        """Normalised end margin keeping both hands on the rail."""
        if self.length <= 2.0 * margin + 1e-6:
            return 0.5
        return margin / self.length

    def clamp_s(self, s, margin=config.RAIL_END_MARGIN):
        m = self.s_margin(margin)
        return float(np.clip(s, m, 1.0 - m))

    def project_s(self, point):
        """Unclamped-then-clamped scalar parameter of the closest point on the segment."""
        _, s = point_segment_project(np.asarray(point, dtype=np.float32), self.p1, self.p2)
        return s

    def project(self, point, standoff=config.STANDOFF_DIST, hang_eye_offset=config.HANG_EYE_OFFSET,
                player_eye_height=config.PLAYER_EYE_HEIGHT):
        """
        Projects point onto rail and returns:
          hands_pos: 3D point on edge where hands grasp
          eye_pos: 3D camera position while hanging
          player_pos: 3D player AABB center position while hanging
          s_clamped: clamped scalar along rail [0, 1]
        """
        s = self.clamp_s(self.project_s(point))
        hands_pos = self.get_point(s)

        # Outward from wall along normal
        eye_pos = hands_pos + self.normal * standoff
        eye_pos[1] -= hang_eye_offset

        # Player AABB center is below eye
        player_pos = eye_pos.copy()
        player_pos[1] -= (player_eye_height - config.PLAYER_HALF_HEIGHT)

        return hands_pos, eye_pos, player_pos, s

    def distance_along(self, point):
        """Metres along the rail (from p1) of the projection of `point`."""
        return float(np.dot(np.asarray(point, dtype=np.float32) - self.p1, self.tangent))

    def point_at_distance(self, d):
        d = float(np.clip(d, 0.0, self.length))
        return self.p1 + self.tangent * d

    def same_face(self, other, tol=0.05):
        """True if `other` lies on the same vertical wall plane with the same normal."""
        if other is None or other is self:
            return False
        if float(np.dot(self.normal, other.normal)) < 0.99:
            return False
        # Same plane: offset along the normal must match
        if abs(float(np.dot(self.p1 - other.p1, self.normal))) > tol:
            return False
        return True
