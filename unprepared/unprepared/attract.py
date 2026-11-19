"""The start screen's live background: the town going about its day while the view strolls its
streets - down Gate Road to the well, along Market Street, up to the manors and the park, round
by the guard house and back (``TITLE_ROUTE``).

Pure logic (no pygame / OpenGL): ``TitleView.update`` moves the world's clock and its people on
and puts the (otherwise idle) player - whose eyes the renderer looks through - on the route,
facing the way it goes. Nobody notices the player there: perception is not run on the start
screen, so no guard asks for a torch and no householder objects.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

# (x, z) along the town's streets, in the order the view takes them (it loops back to the start)
TITLE_ROUTE: Tuple[Tuple[float, float], ...] = (
    (0.0, 46.0), (0.0, 30.0), (0.5, 14.0), (-5.0, 3.5), (-20.0, 3.0), (-38.0, 2.0), (-52.0, 1.0), (-40.0, -2.0),
    (-18.0, -1.5), (-6.0, -2.5), (-1.0, -9.5), (0.0, -24.0), (-2.0, -37.0), (-18.0, -38.0), (-4.0, -39.0), (18.0, -38.0),
    (36.0, -37.0), (37.0, -20.0), (37.5, -3.0), (55.0, 1.0), (68.0, 2.0), (50.0, 2.5), (22.0, 2.0), (4.5, 2.0), (3.0, 8.0),
    (1.0, 22.0), (0.5, 38.0),
)


def _catmull_rom(p0, p1, p2, p3, t: float) -> Tuple[float, float]:
    t2, t3 = t * t, t * t * t
    return tuple(0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3)
                 for a, b, c, d in zip(p0, p1, p2, p3))


class TitleCamera:
    """A smooth closed path through ``route`` (a Catmull-Rom spline), walked at ``SPEED``."""

    SPEED = 2.2              # world units a second: a stroll
    HEIGHT = 2.4             # the eye over the street (a little above the townsfolk's heads)
    LOOK_AHEAD = 7.0         # the view turns towards a point this far along the way
    PITCH = -6.0             # degrees: a look a little down the street
    STEP = 0.5               # spacing of the samples the path is measured by

    def __init__(self, route: Sequence[Tuple[float, float]] = TITLE_ROUTE):
        pts = list(route)
        n = len(pts)
        samples: List[Tuple[float, float]] = []
        for i in range(n):
            p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
            seg = math.dist(p1, p2)
            steps = max(2, int(seg / self.STEP))
            samples += [_catmull_rom(p0, p1, p2, p3, k / steps) for k in range(steps)]
        self.samples = samples
        self.lengths = [0.0]
        for a, b in zip(samples, samples[1:] + samples[:1]):
            self.lengths.append(self.lengths[-1] + math.dist(a, b))
        self.total = self.lengths[-1]
        self.distance = 0.0

    def point_at(self, distance: float) -> Tuple[float, float]:
        d = distance % self.total
        lo, hi = 0, len(self.samples)
        while hi - lo > 1:                                   # (binary search the cumulative lengths)
            mid = (lo + hi) // 2
            if self.lengths[mid] <= d:
                lo = mid
            else:
                hi = mid
        a, b = self.samples[lo], self.samples[(lo + 1) % len(self.samples)]
        span = self.lengths[lo + 1] - self.lengths[lo]
        t = 0.0 if span <= 0 else (d - self.lengths[lo]) / span
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def advance(self, delta_time: float) -> None:
        self.distance = (self.distance + self.SPEED * delta_time) % self.total

    def pose(self) -> Tuple[float, float, float]:
        """(x, z, yaw) of the view now: on the path, turned towards a point a little ahead."""
        x, z = self.point_at(self.distance)
        ax, az = self.point_at(self.distance + self.LOOK_AHEAD)
        yaw = math.degrees(math.atan2(ax - x, -(az - z))) % 360.0
        return x, z, yaw


class TitleView:
    """The start screen's moving picture of the town (see the module docstring)."""

    START_HOUR = 14          # the afternoon: the town busy; by and by the light turns gold and the torches are lit
    YAW_EASE = 2.5           # how briskly the view turns to follow the street (per second)

    def __init__(self, game):
        self.game = game
        self.camera = TitleCamera()
        clock = game.world.clock
        clock.hour, clock.minute_accumulator = self.START_HOUR, 0.0
        game.world.resync_schedules(announce=False, teleport=True)
        game.world.player_is_solid = False       # (the view passes through the crowd; nobody steps round it)
        self._yaw = None
        self._place(0.0)

    def _place(self, delta_time: float) -> None:
        world, player = self.game.world, self.game.player
        x, z, yaw = self.camera.pose()
        if self._yaw is None:
            self._yaw = yaw
        else:
            turn = (yaw - self._yaw + 180.0) % 360.0 - 180.0
            self._yaw = (self._yaw + turn * min(1.0, self.YAW_EASE * delta_time)) % 360.0
        ground = world.ground_height(x, z)
        player.position.x, player.position.z = x, z
        player.position.y = world.floor_height_at(x, z, ground) + self.camera.HEIGHT - player.current_eye_height
        player.rotation_y, player.rotation_x = self._yaw, self.camera.PITCH
        player.has_lit_torch = False

    def update(self, delta_time: float) -> None:
        """One frame: the town lives on (its clock and its people - nobody's eyes are on the
        player), and the view strolls on along the route."""
        world = self.game.world
        world.update_time(delta_time)
        world.update_npcs(delta_time)
        self.camera.advance(delta_time)
        self._place(delta_time)
