"""Hand placement planner for rail traversal.

While hanging, each hand is *planted* at a specific spot on a specific rail.
The body slides continuously; the hands do not - a hand stays where it is until
its resting spot has drifted too far, then hops (lifting off the rail in a small
arc) to a new spot ahead.  Only one hand is in the air at a time, so a shimmy
looks like a genuine hand-over-hand traverse, the leading hand reaching ahead and
the trailing hand catching up.  A hand can never plant closer than
HAND_RAIL_END_MARGIN to the end of its rail: the leading hand stops at the end
of the edge while the body is clamped behind it.

Corner turns and climbs feed the planner explicit per-hand targets (rail,
distance) and it produces the same staggered hops, so the lead hand goes
around the corner first and the trailing hand follows.
"""
import numpy as np
from math3d.geometry import cross3
import config

UP = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def hand_right_of(rail):
    """Unit vector along the rail pointing to the hanging player's right."""
    r = cross3(-rail.normal, UP)
    n = float(np.linalg.norm(r))
    return rail.tangent.copy() if n < 1e-5 else r / n


def wrist_on_rail(rail, d, lift=0.0):
    """Wrist position for a hand holding rail at distance d along it (edge-kind dependent)."""
    from animation.grip import wrist_offset
    p = rail.point_at_distance(d)
    return p + wrist_offset(rail) + UP * lift


def keep_clear_of_shoulder(rel, lateral, min_dist=None):
    """
    Shoulder-relative hand position pushed sideways (along `lateral`, the outward direction for
    that hand) until it is at least `min_dist` from the shoulder.  A hand passing its own
    shoulder - a hand-over from a low hold to a high one, or a freed hand reaching up - swings a
    little wide rather than folding the arm onto itself or going through the chest.
    """
    m = config.HOP_MIN_SHOULDER_DIST if min_dist is None else min_dist
    rel = np.asarray(rel, dtype=np.float32)
    if float(np.dot(rel, rel)) >= m * m:
        return rel
    lat = np.asarray(lateral, dtype=np.float32)
    lat = lat / (np.linalg.norm(lat) + 1e-7)
    a = float(np.dot(rel, lat))
    perp = rel - lat * a
    p2 = float(np.dot(perp, perp))
    if p2 >= m * m:
        return rel
    need = float(np.sqrt(m * m - p2))
    if a >= need:
        return rel
    return (perp + lat * need).astype(np.float32)


class _Hand:
    def __init__(self, side_sign=1.0):
        self.side_sign = side_sign   # -1 left hand, +1 right hand
        self.rail = None
        self.d = 0.0
        self.moving = False
        self.t = 0.0
        self.from_pos = None
        self.to_rail = None
        self.to_d = 0.0
        self.duration = config.HAND_REPLANT_TIME
        self.anchor = None       # shoulder position when the hop launched ...
        self.anchor_now = None   # ... and now: hops are blended in body space

    @property
    def planted_pos(self):
        return wrist_on_rail(self.rail, self.d)

    def position(self):
        if not self.moving:
            return self.planted_pos
        k = _smoothstep(self.t)
        target = wrist_on_rail(self.to_rail, self.to_d)
        # Arc: lift off the rail and come out from the wall mid-hop (the open hand clears the edge)
        tt = float(np.clip(self.t, 0.0, 1.0))
        arc = float(np.sin(np.pi * tt))
        lift = UP * (config.HOP_LIFT * arc) + self.to_rail.normal * (config.HOP_OUT * arc)
        if self.anchor is None or self.anchor_now is None:
            return self.from_pos + (target - self.from_pos) * k + lift
        # Body-relative: the arm swings from where the hand was (relative to the shoulder) to
        # where the hold is, so a hand-over made while the body is still rising or moving
        # sideways travels with the body instead of being dragged through it.
        from_rel = self.from_pos - self.anchor
        to_rel = target - self.anchor_now
        rel = from_rel + (to_rel - from_rel) * k + lift
        # A hand swinging past its own shoulder swings slightly wide (eased in and out so the
        # hop still starts and ends exactly on its holds)
        lateral = hand_right_of(self.to_rail) * self.side_sign
        clear = keep_clear_of_shoulder(rel, lateral)
        w = min(1.0, tt / 0.2, (1.0 - tt) / 0.2)
        return self.anchor_now + rel + (clear - rel) * w


class HandPlanner:
    LEFT, RIGHT = 0, 1

    def __init__(self):
        self.hands = [_Hand(-1.0), _Hand(+1.0)]
        self.last_moved = None
        self.last_lead = None        # which hand led the last hand-over to a new rail
        self.hops = []            # telemetry: (sim_time, hand_index, rail, from_d, to_d)
        self.time = 0.0

    # ------------------------------------------------------------------ state
    @property
    def any_moving(self):
        return any(h.moving for h in self.hands)

    def reset(self):
        for h in self.hands:
            h.rail = None
            h.moving = False
        self.last_moved = None
        # last_lead is kept on purpose so alternation carries across climbs

    def snap(self, rail, d_left, d_right):
        """Both hands close on the rail at once (a catch)."""
        for hand, d in zip(self.hands, (d_left, d_right)):
            hand.rail = rail
            hand.d = self._clamp(rail, d)
            hand.moving = False

    @staticmethod
    def _clamp(rail, d):
        m = config.HAND_RAIL_END_MARGIN
        if rail.length <= 2.0 * m:
            return rail.length * 0.5
        return float(np.clip(d, m, rail.length - m))

    # ----------------------------------------------------------------- update
    def update(self, dt, desired, threshold=0.08, positions=None, anchors=None, min_open_time=None, rate=1.0,
               max_duration=None):
        """
        desired: per hand (rail, d) the hand *wants* to be at (or None to leave it).
        positions: optional per-hand world positions the hands are currently *shown* at (a hop
        then starts from there instead of from the planted spot, e.g. after reaching).
        anchors: optional per-hand shoulder positions; given, hops are blended relative to them.
        min_open_time: optional per-hand seconds the hand needs to open (a hand leaving a deep
        palm press takes longer than one leaving a hook); a hop is stretched to allow it.
        rate: how many times faster than nominal the body is moving (a quick climb is a quick
        reach): new hops take HAND_REPLANT_TIME / HANDOVER_TIME divided by it.
        max_duration: optional per-hand cap on a new hop's duration (a hand-over lands no later than
        the body arrives at the new hold).  None entries (or None) leave the hop as it is.
        Advances hops in flight and starts at most one new hop per call.
        """
        self.time += dt
        for i, hand in enumerate(self.hands):
            hand.anchor_now = None if anchors is None else np.asarray(anchors[i], dtype=np.float32).copy()
            if hand.moving and desired[i] is not None and desired[i][0] is hand.to_rail:
                # a hand in flight keeps tracking the spot it is reaching for (the body moves on)
                hand.to_d = self._clamp(hand.to_rail, desired[i][1])
            if hand.moving:
                hand.t += dt / hand.duration
                if hand.t >= 1.0:
                    hand.moving = False
                    hand.rail = hand.to_rail
                    hand.d = hand.to_d
        if self.any_moving:
            return
        # Pick the hand that is farthest from where it wants to be.  When both hands want to
        # move about equally (a hand-over to the next hold) the hand that did NOT go first last
        # time leads: real climbers alternate.
        cands = []
        for i, hand in enumerate(self.hands):
            want = desired[i]
            if want is None:
                continue
            rail, d = want
            d = self._clamp(rail, d)
            if hand.rail is None:
                hand.rail, hand.d = rail, d
                continue
            if hand.rail is rail:
                dev = abs(d - hand.d)
            else:
                dev = float(np.linalg.norm(wrist_on_rail(rail, d) - hand.planted_pos)) + 1.0
            if dev > threshold:
                cands.append((dev, i, rail, d))
        if not cands:
            return
        cands.sort(reverse=True)
        dev, i, rail, d = cands[0]
        if len(cands) > 1 and abs(cands[0][0] - cands[1][0]) < 0.05 and self.last_lead is not None:
            other = [c for c in cands if c[1] != self.last_lead]
            if other:
                dev, i, rail, d = other[0]
        if rail is not self.hands[i].rail and self.hands[1 - i].rail is not rail:
            self.last_lead = i            # first hand onto a new edge: remember who led
        hand = self.hands[i]
        self.hops.append((self.time, i, rail, hand.d, d))
        hand.moving = True
        hand.t = 0.0
        hand.from_pos = hand.planted_pos if positions is None or positions[i] is None else np.asarray(positions[i], dtype=np.float32).copy()
        hand.anchor = None if anchors is None else np.asarray(anchors[i], dtype=np.float32).copy()
        hand.to_rail = rail
        hand.to_d = d
        hand.duration = (config.HAND_REPLANT_TIME if rail is hand.rail else config.HANDOVER_TIME) / max(1e-3, float(rate))
        if min_open_time is not None and min_open_time[i]:
            hand.duration = max(hand.duration, float(min_open_time[i]) / config.HOP_OPEN_FRACTION)
        if max_duration is not None and max_duration[i] is not None:
            hand.duration = min(hand.duration, max(config.HAND_MIN_HOP_TIME, float(max_duration[i])))
        self.last_moved = i

    # ---------------------------------------------------------------- queries
    def positions(self):
        return self.hands[0].position(), self.hands[1].position()

    def moving_hand(self):
        for i, hand in enumerate(self.hands):
            if hand.moving:
                return i
        return None

    @staticmethod
    def resting_targets(rail, grip_d, shimmy_dir=0.0):
        """Where the two hands want to be for a body gripping at `grip_d` along `rail`.

        While traversing, a hop is aimed where the body will be when the hand
        lands (one replant time ahead) and the leading hand reaches a further
        stride ahead, so the trailing hand is never left hanging behind the body.
        """
        half = config.HAND_SPACING * 0.5
        ahead = shimmy_dir * config.SHIMMY_SPEED * config.HAND_REPLANT_TIME
        lead = config.SHIMMY_HAND_STRIDE * shimmy_dir
        d_left = grip_d - half + ahead + (lead if shimmy_dir < 0 else 0.0)
        d_right = grip_d + half + ahead + (lead if shimmy_dir > 0 else 0.0)
        return (rail, d_left), (rail, d_right)
