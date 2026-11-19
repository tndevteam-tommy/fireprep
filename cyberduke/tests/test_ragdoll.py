"""The knocked-out rag doll: a dead weight that goes down whichever way the current sends
it, never through anything - the obstacles, or the floor - and never stays propped up
against a wall: it ends on the floor, or draped over something low."""

import math
import random
from collections import Counter

import numpy as np
import pytest
from tests.test_visual_regressions import TEXTURES, MatrixRecorder

from cyberduke.ai.guard import CyberGuard
from cyberduke.animation.ragdoll import (
    DIRECTIONS,
    DRAPE_TOP,
    Collapse,
    axis_angle,
    heading,
    rest_of,
    tilt,
)
from cyberduke.config import HACKING
from cyberduke.core.geometry import AABB
from cyberduke.entities.player import Player
from cyberduke.render.humanoid import draw_npc

DT = 1 / 30
NAV = (-10.0, 10.0, -10.0, 10.0)
RUN = HACKING.knockout_shock_duration + 4.5  # long enough for any collapse to finish

# A guard at the origin facing +z: behind is -z, its left is +x. (name, obstacles)
SCENARIOS = {
    "open floor": [],
    "wall 0.45 m behind": [AABB(-1.5, 1.5, -0.75, -0.45, 0.0, 3.0)],
    "wall 0.9 m behind": [AABB(-1.5, 1.5, -1.2, -0.9, 0.0, 3.0)],
    "wall 1.4 m behind": [AABB(-1.5, 1.5, -1.7, -1.4, 0.0, 3.0)],
    "crate 1 m tall, 0.8 m behind": [AABB(-0.6, 0.6, -1.7, -0.8, 0.0, 1.0)],
    "planter 0.55 m right behind": [AABB(-1.2, 1.2, -1.45, -0.45, 0.0, 0.55)],
    "column 0.7 m behind": [AABB(-0.2, 0.2, -1.1, -0.7, 0.0, 3.0)],
    "corner": [AABB(-1.5, 1.5, -1.0, -0.7, 0.0, 3.0), AABB(0.45, 0.75, -1.0, 1.5, 0.0, 3.0)],
    "wall 0.6 m ahead": [AABB(-1.5, 1.5, 0.6, 0.9, 0.0, 3.0)],
    "wall 0.5 m to the left": [AABB(0.5, 0.8, -1.5, 1.5, 0.0, 3.0)],
    "pipe at waist height behind": [AABB(-1.5, 1.5, -0.8, -0.6, 0.9, 1.1)],
    "corridor 1.4 m wide": [
        AABB(-1.2, -0.7, -3.0, 3.0, 0.0, 2.5),
        AABB(0.7, 1.2, -3.0, 3.0, 0.0, 2.5),
    ],
}
CASES = [(name, d) for name in SCENARIOS for d in DIRECTIONS]


def knocked_out(obstacles, direction, yaw=0.0):
    g = CyberGuard(0.0, 0.0, 0.0, NAV, wall_pos=None, patrol_z_min=0.0, patrol_z_max=0.0)
    g.yaw = g.target_yaw = yaw
    g.knock_out(direction)
    return g


def far():
    p = Player()
    p.x, p.z = 50.0, 50.0
    return p


def body_points(g) -> np.ndarray:
    """World-space corner, edge and face-centre points of every drawn part of the body."""
    ctx = MatrixRecorder()
    draw_npc(ctx, g, TEXTURES, sparks=False)
    grid = [(x, y, z) for x in (-0.5, 0.0, 0.5) for y in (-0.5, 0.0, 0.5) for z in (-0.5, 0.0, 0.5)]
    pts = []
    for *_, m in ctx.boxes:
        pts.extend((m @ np.array((*p, 1.0)))[:3] for p in grid)
    return np.array(pts)


def penetration(points: np.ndarray, obstacles) -> float:
    worst = 0.0
    for b in obstacles:
        inside = (
            (points[:, 0] > b.min_x) & (points[:, 0] < b.max_x)
            & (points[:, 1] > b.min_y) & (points[:, 1] < b.max_y)
            & (points[:, 2] > b.min_z) & (points[:, 2] < b.max_z)
        )  # fmt: skip
        if inside.any():
            p = points[inside]
            depth = np.minimum.reduce([
                p[:, 0] - b.min_x, b.max_x - p[:, 0], p[:, 1] - b.min_y,
                b.max_y - p[:, 1], p[:, 2] - b.min_z, b.max_z - p[:, 2],
            ])  # fmt: skip
            worst = max(worst, float(depth.max()))
    return worst


def run_collapse(obstacles, direction, yaw=0.0):
    g = knocked_out(obstacles, direction, yaw)
    worst, floor, when = 0.0, 0.0, 0.0
    for k in range(int(RUN / DT)):
        g.update(far(), DT, obstacles=obstacles)
        if g.body is None:
            continue
        points = body_points(g)
        floor = max(floor, -points[:, 1].min())
        depth = penetration(points, obstacles)
        if depth > worst:
            worst, when = depth, k * DT
        if g.fsm.state.collapse.done:  # lying still from here on: checked once is enough
            break
    return g, worst, floor, when


@pytest.mark.parametrize(("name", "direction"), CASES)
def test_the_body_never_passes_through_anything_and_ends_down(name, direction):
    obstacles = SCENARIOS[name]
    g, worst, floor, when = run_collapse(obstacles, direction)
    assert worst <= 0.03, f"{name}/{direction}: {worst:.3f} m into something at t={when:.2f}s"
    assert floor <= 0.02, f"{name}/{direction}: {floor:.3f} m under the floor"
    points = body_points(g)
    assert points[:, 1].min() <= 0.02, "resting on the floor, not hovering"
    # On the floor - or draped over something low, never propped against a wall
    if g.fall_angle < 70.0:
        low = [b for b in obstacles if b.max_y <= DRAPE_TOP]
        head_y = g.body.point(0.0, 2.0, 0.0)[1]
        assert low and head_y >= min(b.max_y for b in low) - 0.05, (
            f"{name}/{direction}: propped up at {g.fall_angle:.0f} degrees"
        )
    assert g.hack_position[1] < 1.3, "the chest marker came down with the body"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_in_the_open_it_lies_the_way_it_fell(direction):
    """Back: on its back. Forward: face down. Left / right: on that side."""
    g, *_ = run_collapse([], direction)
    r = g.body.rotation
    back, front, left = r @ (0, 0, -1), r @ (0, 0, 1), r @ (1, 0, 0)
    down = {"back": back, "front": front, "left": left, "right": -left}[direction]
    # (on its right side it lies rolled a little back: the holster is under that hip)
    assert down[1] < -0.75, f"{direction}: lies on the wrong side ({down})"
    # and it went down that way: the hips moved towards where it fell
    v = heading(0.0, {"back": 180.0, "front": 0.0, "left": 90.0, "right": -90.0}[direction])
    hip = np.array(g.body.hip)
    assert hip[0] * v[0] + hip[2] * v[2] > 0.0


@pytest.mark.parametrize("direction", ("back", "front"))
def test_against_a_wall_it_topples_off_sideways_onto_the_floor(direction):
    """A body that goes back (or forward) into a wall hits it, then - a dead weight does
    not stay propped up - tips off it to one side and lies on the floor on that side."""
    wall = SCENARIOS["wall 0.45 m behind" if direction == "back" else "wall 0.6 m ahead"]
    g = knocked_out(wall, direction)
    tilts = []
    for _ in range(int(RUN / DT)):
        g.update(far(), DT, obstacles=wall)
        if g.body is not None:
            tilts.append(g.fall_angle)
    assert min(tilts[5:]) < 70.0 or max(tilts) >= 70.0
    stalled = [a for a in tilts if 10.0 < a < 70.0]
    assert stalled, "it met the wall on the way down"
    assert tilts[-1] >= 70.0, "then ended on the floor"
    side = g.body.rotation @ (1, 0, 0)
    assert abs(side[1]) > 0.75, "lying on its side"


def test_it_goes_over_a_low_planter_rather_than_stopping_against_it():
    planter = SCENARIOS["planter 0.55 m right behind"]
    g, worst, *_ = run_collapse(planter, "back")
    head = g.body.point(0.0, 2.0, 0.0)
    assert g.fall_angle >= 70.0 or head[1] >= 0.5, "draped over it, or off it onto the floor"
    assert worst <= 0.03


def test_the_direction_is_random_unless_given():
    seen = Counter()
    for seed in range(40):
        g = CyberGuard(0.0, 0.0, 0.0, NAV, wall_pos=None, guard_id=f"g{seed}")
        g.knock_out()
        for _ in range(int((HACKING.knockout_shock_duration + 0.1) / DT)):
            g.update(far(), DT, obstacles=[])
        seen[g.fsm.state.collapse.direction] += 1
    assert set(seen) == set(DIRECTIONS), seen
    assert min(seen.values()) >= 4


def test_the_fall_is_reproducible():
    a = rest_of(Collapse.begin(0.0, 0.0, 0.0, 30.0, SCENARIOS["corner"], rng=random.Random(7)))
    b = rest_of(Collapse.begin(0.0, 0.0, 0.0, 30.0, SCENARIOS["corner"], rng=random.Random(7)))
    assert np.allclose(a.rotation, b.rotation) and a.hip == b.hip


def test_a_turned_guard_falls_relative_to_its_own_facing():
    """Behind is wherever the guard's back was: a wall behind a guard facing +x counts."""
    wall = [AABB(-0.75, -0.45, -1.5, 1.5, 0.0, 3.0)]  # 0.45 m along -x
    g, worst, *_ = run_collapse(wall, "back", yaw=90.0)
    assert worst <= 0.03
    assert g.fall_angle >= 70.0


def test_rotation_helpers():
    r = tilt(np.array((1.0, 0.0, 0.0)), 90.0)
    assert np.allclose(r @ (0, 1, 0), (1, 0, 0)), "tips the vertical towards the direction"
    for angle, axis in ((30.0, (0, 1, 0)), (179.9, (1, 0, 0)), (120.0, (0.3, 0.5, -0.8))):
        a = np.array(axis) / np.linalg.norm(axis)
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        k = np.array(((0, -a[2], a[1]), (a[2], 0, -a[0]), (-a[1], a[0], 0)))
        m = np.identity(3) * c + s * k + (1 - c) * np.outer(a, a)
        got_angle, got_axis = axis_angle(m)
        assert got_angle == pytest.approx(angle, abs=1e-3)
        assert np.allclose(got_axis, a, atol=1e-3)


# ----------------------------------------------------------------------------- lying on the trunk
def trunk_clearance(g, obstacles) -> float:
    """How far the trunk (head, torso, belt...) is above whatever is under it - the floor,
    or the top of something it lies on."""
    from cyberduke.animation.skeleton import body_boxes, box_points

    pts = box_points(body_boxes(g.body.pose, g.fsm.state.collapse.kit, g.body.matrix(), ("trunk",)))
    support = np.zeros(len(pts))
    for b in obstacles:
        under = (
            (pts[:, 0] > b.min_x) & (pts[:, 0] < b.max_x)
            & (pts[:, 2] > b.min_z) & (pts[:, 2] < b.max_z)
            & (b.max_y <= pts[:, 1] + 0.03)
        )  # fmt: skip
        support = np.where(under, np.maximum(support, b.max_y), support)
    return float((pts[:, 1] - support).min())


# (these all ended face down on a folded arm - or a knee drawn up - floating 30-40 cm up)
PROPPED_BEFORE = [
    ("wall 0.5 m to the left", "front", 0.0),
    ("corridor 1.4 m wide", "front", 0.0),
    ("wall 0.45 m behind", "front", 0.0),
    ("pipe at waist height behind", "front", 90.0),
    ("wall 0.6 m ahead", "front", 90.0),
    ("column 0.7 m behind", "front", 0.0),
    ("corner", "left", 0.0),
]


@pytest.mark.parametrize(("name", "direction", "yaw"), PROPPED_BEFORE)
def test_face_down_it_lies_on_the_floor_not_on_a_folded_arm(name, direction, yaw):
    """Regression: an arm folded out of the way of something as the body went down (bent
    forwards) was still folded when it landed face down - pointing into the floor - and
    the body rested on it, floating. Lying down, a limb now lies flat instead."""
    obstacles = SCENARIOS[name]
    g, worst, *_ = run_collapse(obstacles, direction, yaw)
    assert g.fall_angle >= 80.0
    assert trunk_clearance(g, obstacles) <= 0.03, "the chest on the floor"
    assert worst <= 0.03


@pytest.mark.parametrize(("name", "direction"), CASES)
def test_turned_any_way_it_never_passes_through_anything_and_lies_on_its_trunk(name, direction):
    """The scenarios again with the guard turned 37 degrees: nothing goes into anything,
    and a body that ends lying down rests on its trunk, not on a limb."""
    obstacles = SCENARIOS[name]
    g, worst, floor, when = run_collapse(obstacles, direction, yaw=37.0)
    assert worst <= 0.03, f"{name}/{direction}: {worst:.3f} m into something at t={when:.2f}s"
    assert floor <= 0.02
    if g.fall_angle >= 80.0:
        assert trunk_clearance(g, obstacles) <= 0.05, f"{name}/{direction}: floating"
