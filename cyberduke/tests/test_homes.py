"""People at home keep inside their homes: no arm through a wall while they potter about
the flat, no head through the wall behind the bed - everybody fits their bed, head on the
pillow, feet on the mattress, whatever their height (``ai.alley`` chores and ``Sleep``,
``levels.flats``, ``levels.apartment``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cyberduke.ai import alley
from cyberduke.animation.ragdoll import lying_on
from cyberduke.animation.rig import BodyKit
from cyberduke.animation.skeleton import body_boxes, box_points
from cyberduke.levels import build_night_market
from cyberduke.simulation import Simulation

KIT = BodyKit(armor=False, visor=False, core_glow=False, holster=False, blaster=False)
SIM = Simulation(build_night_market())
LIFE = SIM.life
ROLES = sorted(LIFE.homes)
WALL_WORDS = ("wall", "Wall", "Partition", "Lintel")
# what each chore is done at (a hand on it, sitting on it): its own furniture is no obstacle
AT = {
    "kitchen": ("ounter", "Hob", "Pot", "otplate"),
    "sink": ("ounter", "Sink", "Dishes"),
    "sofa": ("Sofa", "Cushion", "Blanket"),
    "wardrobe": ("Wardrobe",),
    "bath": ("basin", "Mirror"),
    "desk": ("desk", "Desk"),
    "rail": ("rail", "Coats"),
}


def _props(solid_only: bool):
    for sector in SIM.level.sectors:
        for p in sector.props:
            if p.kind == "box" and (p.solid or not solid_only):
                yield p


WALLS = [
    (p.label, p.footprint())
    for p in _props(False)
    if any(w in p.label for w in WALL_WORDS) and p.size[1] > 1.8
]
FURNITURE = [
    (p.label, p.footprint()) for p in _props(True) if not any(w in p.label for w in WALL_WORDS)
]


def _placed(x: float, y: float, z: float, yaw: float, scale: float) -> np.ndarray:
    r = math.radians(yaw)
    c, s = math.cos(r), math.sin(r)
    m = np.identity(4)
    m[:3, :3] = np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))) * scale
    m[:3, 3] = (x, y, z)
    return m


def _into(points: np.ndarray, boxes) -> list[tuple[str, float]]:
    """(label, depth) of every box any of the points are inside."""
    out = []
    for label, b in boxes:
        inside = (
            (points[:, 0] > b.min_x) & (points[:, 0] < b.max_x)
            & (points[:, 2] > b.min_z) & (points[:, 2] < b.max_z)
            & (points[:, 1] > b.min_y) & (points[:, 1] < b.max_y)
        )  # fmt: skip
        if inside.any():
            p = points[inside]
            depth = np.minimum.reduce(
                [p[:, 0] - b.min_x, b.max_x - p[:, 0], p[:, 2] - b.min_z, b.max_z - p[:, 2]]
            ).max()
            out.append((label, round(float(depth), 3)))
    return out


def _person(role: str):
    return next(n for n in SIM.civilians + SIM.merchants if n.role == role)


@pytest.mark.parametrize("role", ROLES)
def test_pottering_about_the_flat_no_arm_goes_through_a_wall_or_the_furniture(role):
    """Every chore, through its whole loop, at its place: the body (the renderer's parts,
    at the person's own height) stays out of the walls - the sushi chef at his wardrobe
    used to reach through the bathroom wall - and out of the furniture it is not using."""
    home = LIFE.homes[role]
    me = _person(role)
    area = LIFE.areas[home.area_id]
    mine = [
        (label, b)
        for label, b in FURNITURE
        if area.contains((b.min_x + b.max_x) / 2, (b.min_z + b.max_z) / 2)
    ]
    for place, chore, _ in alley._chores(home):
        spot = LIFE.places[place]
        using = AT.get(place.split(".")[1], ())
        others = [(label, b) for label, b in mine if not any(u in label for u in using)]
        clip = alley._chore_clip(me, chore)
        for t in np.linspace(0.0, 14.0, 43):
            m = _placed(spot.x, spot.y, spot.z, spot.yaw, me.spec.height)
            pts = box_points(body_boxes(clip(t), KIT, m), spacing=0.05)
            assert not _into(pts, WALLS), f"{place} ({chore}) t={t:.1f}: {_into(pts, WALLS)}"
            assert not _into(pts, others), f"{place} ({chore}) t={t:.1f}: {_into(pts, others)}"


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("side", [-1, 0, 1])
def test_everybody_fits_their_bed(role, side):
    """In bed the top of the head is on the pillow - never through the wall behind it -
    and the body lies on the mattress, however tall they are."""
    home = LIFE.homes[role]
    me = _person(role)
    bed = LIFE.places[home.place("bed")]
    body = lying_on(bed.x, bed.y, bed.z, bed.yaw, KIT, side=side, scale=me.spec.height, crown=True)
    pts = box_points(body_boxes(body.pose, KIT, body.matrix()), spacing=0.05)
    assert not _into(pts, WALLS), _into(pts, WALLS)
    assert abs(float(pts[:, 1].min()) - bed.y) < 1e-6, "lying on the top of the bed"
    pillow = next(
        b for label, b in ((p.label, p.footprint()) for p in _props(False))
        if label == "Pillow" and b.contains_xz(bed.x, bed.z, 0.35)
    )  # fmt: skip
    mattress = next(
        b for label, b in ((p.label, p.footprint()) for p in _props(True))
        if label.startswith("Mattress") and b.contains_xz(bed.x, bed.z, 0.4)
    )  # fmt: skip
    head = np.array(body.point(0.0, 2.0, 0.0))
    assert pillow.contains_xz(head[0], head[2], 0.12), "the head on the pillow"
    feet = np.array(body.point(0.0, 0.1, 0.05))
    assert mattress.contains_xz(feet[0], feet[2], 0.05), "the feet on the mattress"
    # the body over the bed, bar an arm or a hand a little over its edge
    over = mattress.contains_xz
    on = np.array([over(x, z, 0.12) for x, _, z in pts])
    assert on.mean() > 0.97, f"{1.0 - on.mean():.0%} of the body off the bed"
