"""Human body model: reach envelope, hang / lip-stand / mantle pose solving.

Everything the state machine needs to decide whether a hold is *humanly*
reachable and where the body ends up when it is held lives here, so the
numbers can be unit-tested against real-world parkour performance.
"""
import numpy as np
import config
from physics.collision import CollisionSystem
from parkour.ledge_rail import RailType

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _horizontal(v):
    out = np.array([v[0], 0.0, v[2]], dtype=np.float32)
    n = float(np.linalg.norm(out))
    return out, n


_RAIL_ARRAY_CACHE = {}


def _rail_arrays(rails):
    """Stacked (p1, tangent, length, normal, s_margin, height) arrays for a rail list.

    Cached per list identity / length / end elements (a LedgeSystem rebuild changes them).
    """
    if not rails:
        return None
    key = (id(rails), len(rails), id(rails[0]), id(rails[-1]))
    cached = _RAIL_ARRAY_CACHE.get(key)
    if cached is not None:
        return cached
    P1 = np.stack([r.p1 for r in rails]).astype(np.float32)
    T = np.stack([r.tangent for r in rails]).astype(np.float32)
    L = np.array([max(r.length, 1e-6) for r in rails], dtype=np.float32)
    N = np.stack([r.normal for r in rails]).astype(np.float32)
    M = np.array([r.s_margin() for r in rails], dtype=np.float32)
    H = np.array([r.height for r in rails], dtype=np.float32)
    arrays = (P1, T, L, N, M, H)
    _remember(_RAIL_ARRAY_CACHE, key, arrays)
    return arrays


def _remember(cache, key, value, keep=4):
    """A tiny cache: the last few rail lists (a process may hold a couple of levels)."""
    while len(cache) >= keep:
        cache.pop(next(iter(cache)))
    cache[key] = value


_RAIL_GRID_CACHE = {}
RAIL_GRID_CELL = 6.0


def _rail_grid(rails):
    """Uniform XZ grid over the rails (cell -> rail indices), so a reach query only looks at the
    edges near the body instead of every edge in the city.  Cached like _rail_arrays."""
    arrays = _rail_arrays(rails)
    if arrays is None:
        return None
    key = (id(rails), len(rails), id(rails[0]), id(rails[-1]))
    grid = _RAIL_GRID_CACHE.get(key)
    if grid is not None:
        return grid
    P1, T, L = arrays[0], arrays[1], arrays[2]
    P2 = P1 + T * L[:, None]
    lo = np.floor(np.minimum(P1, P2)[:, [0, 2]] / RAIL_GRID_CELL).astype(np.int64)
    hi = np.floor(np.maximum(P1, P2)[:, [0, 2]] / RAIL_GRID_CELL).astype(np.int64)
    cells = {}
    for i in range(len(rails)):
        for ci in range(int(lo[i, 0]), int(hi[i, 0]) + 1):
            for ck in range(int(lo[i, 1]), int(hi[i, 1]) + 1):
                cells.setdefault((ci, ck), []).append(i)
    grid = {c: np.array(v, dtype=np.int64) for c, v in cells.items()}
    _remember(_RAIL_GRID_CACHE, key, grid)
    return grid


def _rails_near(rails, point, radius):
    """Sorted indices of the rails whose XZ bounding box may come within `radius` of `point`."""
    grid = _rail_grid(rails)
    if grid is None:
        return np.zeros(0, dtype=np.int64)
    x0 = int(np.floor((float(point[0]) - radius) / RAIL_GRID_CELL))
    x1 = int(np.floor((float(point[0]) + radius) / RAIL_GRID_CELL))
    z0 = int(np.floor((float(point[2]) - radius) / RAIL_GRID_CELL))
    z1 = int(np.floor((float(point[2]) + radius) / RAIL_GRID_CELL))
    parts = [grid[c] for c in ((i, k) for i in range(x0, x1 + 1) for k in range(z0, z1 + 1)) if c in grid]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))


class GrabResult:
    """Outcome of evaluating a rail for grabbing."""
    HANG = "HANG"
    MANTLE = "MANTLE"

    def __init__(self, rail, s, grip, mode, hang_center, score, supported=False):
        self.rail = rail
        self.s = s
        self.grip = grip                 # hands position on the rail
        self.mode = mode                 # GrabResult.HANG or GrabResult.MANTLE
        self.hang_center = hang_center   # player AABB center while hanging (None for MANTLE)
        self.score = score
        self.supported = supported       # feet resting on the ground while gripping

    def __repr__(self):
        return f"GrabResult({self.mode}, {self.rail}, s={self.s:.2f}, score={self.score:.2f})"


class BodyModel:
    SIZE = np.array([config.PLAYER_WIDTH, config.PLAYER_HEIGHT, config.PLAYER_DEPTH], dtype=np.float32)

    # ----------------------------------------------------------------- anatomy
    @staticmethod
    def feet_y(center):
        return float(center[1]) - config.PLAYER_HALF_HEIGHT

    @staticmethod
    def eye_pos(center):
        return np.asarray(center, dtype=np.float32) + UP * (config.PLAYER_EYE_HEIGHT - config.PLAYER_HALF_HEIGHT)

    @staticmethod
    def camera_pos(center, facing_h):
        """The eyes sit CAMERA_FORWARD ahead of the body centre (in front of the shoulder line)."""
        f, n = _horizontal(facing_h)
        f = f / n if n > 1e-6 else np.zeros(3, dtype=np.float32)
        return BodyModel.eye_pos(center) + f * config.CAMERA_FORWARD

    @staticmethod
    def shoulder_center(center, facing_h):
        """Mid-point between the two shoulder pivots."""
        c = np.asarray(center, dtype=np.float32)
        f, n = _horizontal(facing_h)
        f = f / n if n > 1e-6 else np.zeros(3, dtype=np.float32)
        return c + UP * (config.SHOULDER_HEIGHT - config.PLAYER_HALF_HEIGHT) + f * config.SHOULDER_FORWARD

    @staticmethod
    def center_from_eye(eye):
        return np.asarray(eye, dtype=np.float32) - UP * (config.PLAYER_EYE_HEIGHT - config.PLAYER_HALF_HEIGHT)

    # ------------------------------------------------------------------- poses
    @staticmethod
    def hang_center(rail, s):
        """Player AABB center while hanging from rail at parameter s (free hang)."""
        grip = rail.get_point(s)
        eye = grip + rail.normal * config.STANDOFF_DIST - UP * config.HANG_EYE_OFFSET
        return BodyModel.center_from_eye(eye)

    @staticmethod
    def pull_up_center(rail, s):
        """Top of the pull-up: hands at the chest, eyes just above the hold."""
        grip = rail.get_point(s)
        eye = grip + rail.normal * config.STANDOFF_DIST + UP * 0.05
        return BodyModel.center_from_eye(eye)

    @staticmethod
    def lip_stand_center(rail, s):
        """Standing on the crevasse lip with the chest pressed to the wall."""
        grip = rail.get_point(s)
        standoff = config.PLAYER_WIDTH * 0.5 + 0.01
        return grip + rail.normal * standoff + UP * config.PLAYER_HALF_HEIGHT

    @staticmethod
    def mantle_mid_center(rail, s):
        grip = rail.get_point(s)
        return grip + UP * (config.PLAYER_HALF_HEIGHT + 0.05) - rail.normal * 0.10

    @staticmethod
    def owner_depth_along_normal(rail):
        if rail.owner is None:
            return 2.0
        size = rail.owner.bounds.size
        n = np.abs(rail.normal)
        return float(np.dot(size, n))

    @staticmethod
    def mantle_target_center(rail, s, obstacles=None):
        """
        Standing position after mantling over `rail`.  Prefers a comfortable step
        onto the surface; with `obstacles` given it backs the step off (a narrow
        ledge, something standing close behind the edge) and returns None when
        there is no room to stand at all.
        """
        grip = rail.get_point(s)
        depth = BodyModel.owner_depth_along_normal(rail)
        preferred = float(np.clip(depth * 0.5, config.MANTLE_STEP_IN_MIN, config.MANTLE_STEP_IN_MAX))
        steps = [preferred] + [v for v in (0.55, config.MANTLE_STEP_IN_MIN) if v < preferred - 1e-6]
        for step_in in steps:
            center = grip + UP * (config.PLAYER_HALF_HEIGHT + 0.02) - rail.normal * step_in
            if obstacles is None or BodyModel.pose_is_free(center, obstacles, shrink=0.02):
                return center
        return None

    @staticmethod
    def box_at(center, shrink=0.01):
        return CollisionSystem.player_box(center, BodyModel.SIZE, shrink=shrink)

    @staticmethod
    def pose_is_free(center, obstacles, shrink=0.01):
        return CollisionSystem.overlaps_any(BodyModel.box_at(center, shrink), obstacles) is None

    @staticmethod
    def path_is_free(points, obstacles, shrink=0.02, step=0.1):
        """True if the body can move along the polyline `points` without entering geometry
        (checked every `step` metres, the end points included)."""
        pts = [np.asarray(p, dtype=np.float32) for p in points]
        if not pts or not BodyModel.pose_is_free(pts[0], obstacles, shrink):
            return False
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / step)))
            for k in range(1, n + 1):
                if not BodyModel.pose_is_free(a + (b - a) * (k / n), obstacles, shrink):
                    return False
        return True

    # ------------------------------------------------------------------- reach
    @staticmethod
    def within_reach(shoulder, point, max_reach=config.MAX_GRAB_REACH):
        return float(np.linalg.norm(np.asarray(point, dtype=np.float32) - shoulder)) <= max_reach

    @staticmethod
    def hold_above_reachable_from_lip(lip_y, hold_y):
        """Can a hold at hold_y be grabbed while standing on a lip at lip_y?"""
        gap = hold_y - lip_y
        return 0.3 <= gap <= config.CLIMB_HOLD_GAP_MAX

    # ------------------------------------------------------------- grab solver
    @staticmethod
    def evaluate_grab(rail, center, facing_h, look_dir, on_ground, obstacles, velocity=None):
        """
        Decide if `rail` can be grabbed from the current body pose.

        Returns a GrabResult or None.  The rules encode a 1.80 m athlete:
          * the hands must be within MAX_GRAB_REACH of the shoulder pivot,
          * the hold must be in front of the chest and the chest must face the wall,
          * a hold can be caught no lower than the chest while airborne,
          * the resulting hang body must not intersect the world.  If only the
            ground is in the way the feet rest on it (supported grip); if the hold
            is too low to hang from at all, a MANTLE_TOP becomes a direct vault.
        """
        center = np.asarray(center, dtype=np.float32)
        facing, fn = _horizontal(facing_h)
        if fn < 1e-6:
            return None
        facing /= fn

        # The body must be on the outside of the wall face and face it
        if float(np.dot(facing, -rail.normal)) < 0.35:
            return None

        shoulder = BodyModel.shoulder_center(center, facing)
        reach_center = shoulder + facing * 0.25
        s = rail.clamp_s(rail.project_s(reach_center))
        grip = rail.get_point(s)

        if float(np.dot(shoulder - grip, rail.normal)) < 0.05:
            return None

        # Hold must be roughly in front of the chest (not behind / far to the side)
        to_grip_h, n = _horizontal(grip - shoulder)
        if n > 0.15 and float(np.dot(to_grip_h / n, facing)) < 0.35:
            return None

        # Arm reach from the shoulder pivot
        dist = float(np.linalg.norm(grip - shoulder))
        if dist > config.MAX_GRAB_REACH:
            return None

        feet_y = BodyModel.feet_y(center)
        hold_rise = float(grip[1]) - feet_y

        if on_ground:
            if hold_rise < config.GROUND_VAULT_MIN_HEIGHT or hold_rise > config.STAND_REACH:
                return None
        else:
            if float(grip[1]) < shoulder[1] - config.GRAB_BELOW_SHOULDER_MAX:
                return None
            vy = float(velocity[1]) if velocity is not None else 0.0
            if vy > config.RISING_GRAB_VY and float(grip[1]) < BodyModel.eye_pos(center)[1] + 0.1:
                # Still flying upward fast: holds at face height are passed, catch happens above
                return None

        # Angular preference toward the look direction for scoring
        look = np.asarray(look_dir, dtype=np.float32)
        look = look / (np.linalg.norm(look) + 1e-7)
        to_grip = grip - BodyModel.eye_pos(center)
        to_grip /= (np.linalg.norm(to_grip) + 1e-7)
        angle_pen = 1.0 - float(np.clip(np.dot(look, to_grip), -1.0, 1.0))
        score = dist + config.GRAB_LOOK_WEIGHT * angle_pen

        # ---- resolve body pose ----
        hang_center = BodyModel.hang_center(rail, s)
        supported = False
        if not BodyModel.pose_is_free(hang_center, obstacles, shrink=0.001):
            # Collision buffer: clipping something by a few centimetres (the wall beside the end of an
            # awning or balcony edge, a post) is solved by taking the hold a hand-width along the edge
            slid = BodyModel.slide_to_free_grip(rail, s, shoulder, facing, obstacles)
            if slid is not None:
                s = slid
                grip = rail.get_point(s)
                hang_center = BodyModel.hang_center(rail, s)
                dist = float(np.linalg.norm(grip - shoulder))
                to_grip = grip - BodyModel.eye_pos(center)
                to_grip /= (np.linalg.norm(to_grip) + 1e-7)
                score = dist + config.GRAB_LOOK_WEIGHT * (1.0 - float(np.clip(np.dot(look, to_grip), -1.0, 1.0)))
        if not BodyModel.pose_is_free(hang_center, obstacles, shrink=0.001):
            ground = CollisionSystem.ground_height_below(center + UP * 0.05, BodyModel.SIZE, obstacles, max_drop=3.0)
            if ground is not None:
                raised = hang_center.copy()
                raised[1] = ground + config.PLAYER_HALF_HEIGHT + 0.001
                # Feet on the ground while gripping: hands must still be at or above the shoulders
                raised_shoulder = BodyModel.shoulder_center(raised, -rail.normal)
                if (raised[1] >= hang_center[1] and
                        float(grip[1]) >= raised_shoulder[1] + 0.15 and
                        BodyModel.within_reach(raised_shoulder, grip) and
                        BodyModel.pose_is_free(raised, obstacles)):
                    hang_center = raised
                    supported = True
                elif rail.rail_type == RailType.MANTLE_TOP and BodyModel.mantle_is_valid(rail, s, obstacles):
                    return GrabResult(rail, s, grip, GrabResult.MANTLE, None, score)
                else:
                    return None
            else:
                return None

        if rail.rail_type == RailType.MANTLE_TOP and on_ground and hold_rise <= config.STAND_REACH:
            # Standing under a low top edge: this is a vault, not a hang
            if supported and BodyModel.mantle_is_valid(rail, s, obstacles):
                return GrabResult(rail, s, grip, GrabResult.MANTLE, None, score)

        return GrabResult(rail, s, grip, GrabResult.HANG, hang_center, score, supported)

    @staticmethod
    def slide_to_free_grip(rail, s, shoulder, facing, obstacles, buffer=None, step=0.02):
        """Nearest grip within `buffer` metres along the rail (either way) whose free hang pose is
        collision-free and still a valid reach (within arm's length, in front of the chest, body on
        the outside of the wall).  None if there is none - then the hold is genuinely blocked."""
        buffer = config.EDGE_COLLISION_BUFFER if buffer is None else buffer
        length = max(rail.length, 1e-6)
        m = rail.s_margin()
        n = int(round(buffer / step))
        for k in range(1, n + 1):
            for sign in (1.0, -1.0):
                s2 = s + sign * k * step / length
                if s2 < m - 1e-9 or s2 > 1.0 - m + 1e-9:
                    continue
                grip = rail.get_point(s2)
                if float(np.linalg.norm(grip - shoulder)) > config.MAX_GRAB_REACH:
                    continue
                if float(np.dot(shoulder - grip, rail.normal)) < 0.05:
                    continue
                to_grip_h, hn = _horizontal(grip - shoulder)
                if hn > 0.15 and float(np.dot(to_grip_h / hn, facing)) < 0.35:
                    continue
                if BodyModel.pose_is_free(BodyModel.hang_center(rail, s2), obstacles, shrink=0.001):
                    return float(s2)
        return None

    @staticmethod
    def mantle_is_valid(rail, s, obstacles):
        """The top surface must have room for the body to stand."""
        if rail.rail_type != RailType.MANTLE_TOP:
            return False
        if BodyModel.mantle_target_center(rail, s, obstacles) is None:
            return False
        mid = BodyModel.mantle_mid_center(rail, s)
        # Mid pose intentionally overlaps the owner's edge by 10 cm; ignore the owner
        box = BodyModel.box_at(mid, shrink=0.02)
        return CollisionSystem.overlaps_any(box, obstacles, ignore=rail.owner) is None

    @staticmethod
    def find_best_grab(rails, center, facing_h, look_dir, on_ground, obstacles, velocity=None, exclude=None):
        """Best grabbable rail for the current pose (vectorised pre-filter, exact check on survivors)."""
        arrays = _rail_arrays(rails)
        if arrays is None:
            return None
        center = np.asarray(center, dtype=np.float32)
        facing, fn = _horizontal(facing_h)
        if fn < 1e-6:
            return None
        facing /= fn
        shoulder = BodyModel.shoulder_center(center, facing)
        reach_center = shoulder + facing * 0.25
        near = _rails_near(rails, reach_center, config.MAX_GRAB_REACH + 0.5)
        if len(near) == 0:
            return None
        P1, T, L, N, M, H = (a[near] for a in arrays)
        s = np.einsum("ij,ij->i", reach_center[None, :] - P1, T) / L
        s = np.clip(s, M, 1.0 - M)
        grip = P1 + T * (s * L)[:, None]
        ok = (N @ (-facing)) >= 0.35
        ok &= np.einsum("ij,ij->i", shoulder[None, :] - grip, N) >= 0.05
        ok &= np.linalg.norm(grip - shoulder[None, :], axis=1) <= config.MAX_GRAB_REACH
        if on_ground:
            rise = grip[:, 1] - BodyModel.feet_y(center)
            ok &= (rise >= config.GROUND_VAULT_MIN_HEIGHT) & (rise <= config.STAND_REACH)
        else:
            ok &= grip[:, 1] >= shoulder[1] - config.GRAB_BELOW_SHOULDER_MAX
        best = None
        excluded = exclude if isinstance(exclude, (set, frozenset, list, tuple)) else ((exclude,) if exclude is not None else ())
        for i in near[np.nonzero(ok)[0]]:
            rail = rails[int(i)]
            if any(rail is e for e in excluded):
                continue
            res = BodyModel.evaluate_grab(rail, center, facing_h, look_dir, on_ground, obstacles, velocity)
            if res is not None and (best is None or res.score < best.score):
                best = res
        return best

    @staticmethod
    def stand_center(rail, s, fraction):
        """Body centre part-way through the stand-up on the lip of `rail` (0 = top of pull-up, 1 = standing)."""
        pull = BodyModel.pull_up_center(rail, s)
        stand = BodyModel.lip_stand_center(rail, s)
        return pull + (stand - pull) * float(np.clip(fraction, 0.0, 1.0))

    @staticmethod
    def find_hold_above(rails, rail, s, obstacles, look_dir=None):
        """
        Best hold reachable from the lip of `rail`: the climber pulls up, gets a
        foot on the lip and pushes up only as far as needed to reach the hold
        with one arm (a partial or full stand).  Any rail whose grip point is
        within MAX_GRAB_REACH of the shoulders at some point of that rise and at
        least 0.3 m above the lip qualifies - straight up the same wall (crevasse
        ladder) or off to the side (a balcony edge, a window head on the next
        bay).  The body must be collision-free at the required height (an
        overhang above the lip limits how far you can stand up).  A hold the
        player is looking at (within AIM_CONE_DEG) is preferred; otherwise the
        highest reachable hold wins, ties go to the nearest.
        Returns (rail, s, stand_fraction) or None.
        """
        lip_y = rail.height
        grip = rail.get_point(s)
        if not BodyModel.pose_is_free(BodyModel.pull_up_center(rail, s), obstacles):
            return None
        stand = BodyModel.lip_stand_center(rail, s)
        shoulder_hi = BodyModel.shoulder_center(stand, -rail.normal)
        shoulder_lo = shoulder_hi.copy()
        shoulder_lo[1] = lip_y - 0.15
        rise = float(shoulder_hi[1] - shoulder_lo[1])
        eye = BodyModel.eye_pos(BodyModel.hang_center(rail, s))
        look = None
        if look_dir is not None:
            look = np.asarray(look_dir, dtype=np.float32)
            look = look / (np.linalg.norm(look) + 1e-7)
        cos_aim = float(np.cos(np.radians(config.AIM_CONE_DEG)))
        best = None
        best_key = None
        reach2 = config.MAX_GRAB_REACH ** 2
        for idx in BodyModel._holds_near(rails, shoulder_hi, lip_y + 0.3, lip_y + config.CLIMB_HOLD_GAP_MAX):
            cand = rails[idx]
            if cand is rail:
                continue
            # Quick reject: closest point of the whole rail still out of reach
            s0 = cand.clamp_s(cand.project_s(shoulder_hi))
            g0 = cand.get_point(s0)
            horiz = g0 - shoulder_lo
            horiz[1] = 0.0
            if float(np.dot(horiz, horiz)) > reach2:
                continue
            # Try the projected spot first, then slide along the rail to find a usable grip
            step = 0.1 / max(cand.length, 1e-6)
            for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
                s2 = s0 + k * step
                if s2 < cand.s_margin() - 1e-9 or s2 > 1.0 - cand.s_margin() + 1e-9:
                    continue
                g2 = cand.get_point(s2)
                horiz = g2 - shoulder_lo
                horiz[1] = 0.0
                h2 = float(np.dot(horiz, horiz))
                if h2 > reach2:
                    continue
                # Lowest shoulder height from which the hold is within reach
                need_y = float(g2[1]) - float(np.sqrt(reach2 - h2))
                fraction = float(np.clip((need_y - shoulder_lo[1]) / max(rise, 1e-6), 0.0, 1.0))
                body = BodyModel.stand_center(rail, s, fraction)
                if float(np.dot(body - g2, cand.normal)) < 0.0:
                    continue                      # hold faces away from us
                if cand.same_face(rail):
                    lateral = g2 - grip
                    lateral[1] = 0.0
                    if float(np.linalg.norm(lateral)) > 0.6:
                        continue
                if not BodyModel.pose_is_free(body, obstacles):
                    continue
                # A top edge is a hold even when there is no room to mantle onto it (window sills)
                if not (BodyModel.mantle_is_valid(cand, s2, obstacles) or
                        BodyModel.pose_is_free(BodyModel.hang_center(cand, s2), obstacles)):
                    continue
                shoulder = shoulder_lo + (shoulder_hi - shoulder_lo) * fraction
                aim = 2.0
                if look is not None:
                    to_g = g2 - eye
                    to_g /= (np.linalg.norm(to_g) + 1e-7)
                    c = float(np.dot(to_g, look))
                    if c >= cos_aim:
                        aim = 1.0 - c         # smaller is better
                key = (aim < 2.0, -aim, round(cand.height, 2), -float(np.linalg.norm(g2 - shoulder)))
                if best_key is None or key > best_key:
                    best_key = key
                    best = (cand, s2, fraction)
                break
        return best

    @staticmethod
    def _holds_near(rails, point, y_min, y_max, radius=config.MAX_GRAB_REACH + 0.6):
        """Indices of rails within a height band whose segment passes within `radius` (XZ) of `point`."""
        arrays = _rail_arrays(rails)
        if arrays is None:
            return []
        p = np.asarray(point, dtype=np.float32)
        near = _rails_near(rails, p, radius + 0.01)
        if len(near) == 0:
            return []
        P1, T, L, N, M, H = (a[near] for a in arrays)
        ok = (H >= y_min - 1e-6) & (H <= y_max + 1e-6)
        if not np.any(ok):
            return []
        d = np.einsum("ij,ij->i", p[None, :] - P1, T)
        d = np.clip(d, 0.0, L)
        closest = P1 + T * d[:, None]
        dx = closest[:, 0] - p[0]
        dz = closest[:, 2] - p[2]
        ok &= (dx * dx + dz * dz) <= radius * radius
        return near[np.nonzero(ok)[0]].tolist()

    # ----------------------------------------------------------- corner turns
    @staticmethod
    def corner_entry(link, obstacles, max_search=1.0):
        """
        Parameter on the linked rail where the body can hang right after turning
        the corner: the closest collision-free hang pose to the corner end.
        """
        q = link.rail
        m = q.s_margin()
        s = m if link.side == "right" else 1.0 - m
        step = 0.05 / max(q.length, 1e-6)
        travelled = 0.0
        while travelled <= max_search and 0.0 <= s <= 1.0:
            if BodyModel.pose_is_free(BodyModel.hang_center(q, s), obstacles):
                return float(s)
            s += step * link.direction
            travelled += 0.05
        return None

    @staticmethod
    def corner_path(rail, s_from, link, s_to):
        """Body-centre keyframes for swinging around a corner (list of positions)."""
        q = link.rail
        start = BodyModel.hang_center(rail, s_from)
        end = BodyModel.hang_center(q, s_to)
        if link.kind == "convex":
            corner = rail.p2 if link.side == "right" else rail.p1
            diag = rail.normal + q.normal
            diag = diag / (np.linalg.norm(diag) + 1e-7)
            mid = corner + diag * (config.PLAYER_WIDTH * 0.5 * 1.414 + 0.08)
            mid[1] = start[1]
            return [mid, end]
        return [end]

    # --------------------------------------------------------- climb-down edge
    @staticmethod
    def find_climb_down_edge(rails, center, facing_h, obstacles):
        """
        A top edge of the surface the player stands on, just in front of them,
        that they could lower themselves over into a hang.  Returns (rail, s) or None.
        """
        center = np.asarray(center, dtype=np.float32)
        facing, fn = _horizontal(facing_h)
        if fn < 1e-6:
            return None
        facing /= fn
        feet = BodyModel.feet_y(center)
        best = None
        best_d = 1e9
        for idx in BodyModel._holds_near(rails, center, feet - 0.08, feet + 0.08, radius=config.CLIMB_DOWN_EDGE_MAX_DIST + 1.0):
            rail = rails[idx]
            if rail.rail_type != RailType.MANTLE_TOP:
                continue
            if float(np.dot(facing, rail.normal)) < 0.5:
                continue                      # must face out over the edge
            s = rail.clamp_s(rail.project_s(center))
            grip = rail.get_point(s)
            rel = center - grip
            lateral = rel - rail.normal * float(np.dot(rel, rail.normal))
            lateral[1] = 0.0
            if float(np.linalg.norm(lateral)) > 0.05 + 1e-3 and (rail.project_s(center) < 0.0 or rail.project_s(center) > 1.0):
                continue
            inward = -float(np.dot(rel, rail.normal))   # distance back from the edge onto the surface
            if inward < -0.05 or inward > config.CLIMB_DOWN_EDGE_MAX_DIST:
                continue
            if not BodyModel.pose_is_free(BodyModel.hang_center(rail, s), obstacles):
                continue
            box = BodyModel.box_at(BodyModel.mantle_mid_center(rail, s), shrink=0.02)
            if CollisionSystem.overlaps_any(box, obstacles, ignore=rail.owner) is not None:
                continue
            if inward < best_d:
                best_d = inward
                best = (rail, s)
        return best

    @staticmethod
    def find_hold_below(rails, rail, s, obstacles):
        """
        Hold the climber can lower themselves onto while still holding `rail`:
        the reverse of find_hold_above - standing (partially) on the candidate's
        lip, the current grip must be within reach.  Same wall or a ledge beside
        it.  Prefers the highest such hold (smallest drop).  Returns (rail, s) or None.
        """
        grip = rail.get_point(s)
        reach2 = config.MAX_GRAB_REACH ** 2
        best = None
        best_key = None
        for idx in BodyModel._holds_near(rails, grip, rail.height - config.CLIMB_HOLD_GAP_MAX, rail.height - 0.3):
            cand = rails[idx]
            if cand is rail:
                continue
            s0 = cand.clamp_s(cand.project_s(grip))
            step = 0.1 / max(cand.length, 1e-6)
            for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
                s2 = s0 + k * step
                if s2 < cand.s_margin() - 1e-9 or s2 > 1.0 - cand.s_margin() + 1e-9:
                    continue
                stand = BodyModel.lip_stand_center(cand, s2)
                sh_hi = BodyModel.shoulder_center(stand, -cand.normal)
                sh_lo = sh_hi.copy()
                sh_lo[1] = cand.height - 0.15
                horiz = grip - sh_lo
                horiz[1] = 0.0
                h2 = float(np.dot(horiz, horiz))
                if h2 > reach2:
                    continue
                need_y = float(grip[1]) - float(np.sqrt(reach2 - h2))
                fraction = float(np.clip((need_y - sh_lo[1]) / max(1e-6, sh_hi[1] - sh_lo[1]), 0.0, 1.0))
                body = BodyModel.stand_center(cand, s2, fraction)
                if float(np.dot(body - grip, rail.normal)) < 0.0:
                    continue
                if not (BodyModel.pose_is_free(body, obstacles) and
                        BodyModel.pose_is_free(BodyModel.pull_up_center(cand, s2), obstacles) and
                        BodyModel.pose_is_free(BodyModel.hang_center(cand, s2), obstacles)):
                    continue
                key = (round(cand.height, 2), -abs(k))
                if best_key is None or key > best_key:
                    best_key = key
                    best = (cand, s2, fraction)
                break
        return best
