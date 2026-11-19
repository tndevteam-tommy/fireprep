"""The city's vehicles, as data: each model a list of ``Prop`` parts in its own frame.

A vehicle is built standing on the road at the origin of its own frame, its nose towards
-x (x along it, z across it, y up). Its body is one list of parts (``VehicleModel.body``,
given the paint); its wheels are drawn apart from the body (``VehicleModel.wheel``: one
wheel about its hub, the axle along z) so that they can turn as it drives. The parked
coupe on the main street (``levels.main_street.car``) is the same coupe set down at the
kerb; the traffic (``entities.traffic``) drives the rest up and down the road, and
``render.traffic`` draws them where they are.

Like every prop in the levels, the parts overlap - that is how they are modelled - but
no two share a face (``render.audit``): each stands a few millimetres proud of the next.

Models:

* ``coupe`` - a low, wedge-nosed electric coupe: a lacquered body with a chrome waist
  line and a neon sill, a tinted glass canopy, a rear wing, a light bar across the nose,
  a red strip across the tail, chrome five-spoke rims with a ring glowing in each hub,
  and an underglow on the wet road;
* ``taxi`` - a boxy yellow cab: a black-and-yellow chequer band down each side, a lit
  TAXI box on the roof, bumpers, round the wheels plain steel rims;
* ``van`` - a tall delivery van: a cab with a raked windscreen, a box body with an
  advertisement down each side under a neon stripe, an amber light bar on the cab roof,
  double doors at the back.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from cyberduke.world.props import Material, Prop

RGB = tuple[float, float, float]
DECAL = frozenset({"transparent"})

CAR_DARK = Material("metal", (0.06, 0.07, 0.09), specular=(0.5, 0.5, 0.55), shininess=40.0)
CAR_TRIM = Material("metal", (0.62, 0.64, 0.7), specular=(1.0, 1.0, 1.0), shininess=90.0)
CAR_GLASS = Material(None, (0.1, 0.16, 0.24), specular=(1.0, 1.0, 1.0), shininess=110.0)
TYRE = Material(None, (0.05, 0.05, 0.06))
RIM = Material("metal", (0.5, 0.52, 0.58), specular=(1.0, 1.0, 1.0), shininess=80.0)
HEADLIGHT = Material(None, (0.9, 0.95, 1.0), emission=(2.0, 2.2, 2.4))
TAILLIGHT = Material(None, (1.0, 0.1, 0.15), emission=(2.2, 0.15, 0.25))
PLATE = Material(None, (0.2, 0.8, 0.9), emission=(0.2, 0.9, 1.0))
AMBER = Material(None, (1.0, 0.6, 0.1), emission=(2.0, 1.1, 0.2))


@dataclass(frozen=True)
class Paint:
    """A vehicle's colours: the body's lacquer and the glow of its neon (sills, hubs,
    underglow)."""

    body: RGB
    glow: RGB = (1.0, 0.3, 0.8)


def lacquer(rgb: RGB) -> Material:
    return Material("metal", rgb, specular=(1.0, 1.0, 1.1), shininess=80.0)


def neon(rgb: RGB, k: float = 2.2) -> Material:
    return Material(None, rgb, emission=tuple(c * k for c in rgb))  # type: ignore[arg-type]


class _Parts:
    """Collects a model's parts (in its frame)."""

    def __init__(self) -> None:
        self.props: list[Prop] = []

    def box(self, x, y, z, sx, sy, sz, mat, label, tags=frozenset(), tilt=0.0) -> None:
        rotate = ((tilt, 0.0, 0.0, 1.0),) if tilt else ()
        solid = "transparent" not in tags
        self.props.append(
            Prop.box(x, y, z, sx, sy, sz, mat, label=label, tags=tags, rotate=rotate, solid=solid)
        )

    def disc_across(self, x, y, z, r, thickness, mat, label, slices=16) -> None:
        """A disc/cylinder across the vehicle (its axis along z), centred at (x, y, z)."""
        self.props.append(
            Prop.cylinder(
                x, y, z - thickness / 2.0, r, r, thickness, mat, slices=slices,
                rotate=((90.0, 1.0, 0.0, 0.0),), label=label,
            )
        )  # fmt: skip


# ----------------------------------------------------------------------------- wheels
def _wheel(
    r: float, w: float, side: float, rim: Material, hub: Material | None, spokes: int,
    rim_r: float,
) -> list[Prop]:  # fmt: skip
    """One wheel about its hub (the axle along z; ``side`` the way its outer wall faces):
    the tyre, the rim, spokes each 2 mm further out than the last, a glowing ring in the
    hub (``hub``) or a plain cap."""
    b = _Parts()
    b.disc_across(0.0, 0.0, 0.0, r, w, TYRE, "Car tyre")
    face = side * w / 2.0  # the tyre's outer wall
    b.disc_across(0.0, 0.0, face + side * 0.006, rim_r, 0.012, rim, "Car wheel rim", slices=14)
    for k in range(spokes):
        b.box(
            0.0, 0.0, face + side * (0.014 + 0.002 * k), rim_r * 1.84, r * 0.125, 0.0015,
            CAR_TRIM, "Car wheel spoke", tilt=180.0 / spokes * k,
        )  # fmt: skip
    cap = hub or CAR_TRIM
    hub_at = face + side * (0.02 + 0.002 * spokes)  # (clear of the last spoke)
    b.disc_across(0.0, 0.0, hub_at, r * 0.3, 0.006, cap, "Car hub", slices=14)
    return b.props


# ----------------------------------------------------------------------------- the coupe
def _coupe(paint: Paint) -> list[Prop]:
    body = lacquer(paint.body)
    sill = neon(paint.glow)
    b = _Parts()
    box = b.box
    # the body: the lower hull, a nose dipping to the road, a tail kicking up, the waist
    box(0.0, 0.47, 0.0, 4.4, 0.38, 1.9, body, "Car body")
    box(-2.05, 0.5, 0.0, 0.9, 0.3, 1.86, body, "Car nose", tilt=12.0)
    box(2.0, 0.6, 0.0, 0.7, 0.36, 1.86, body, "Car tail", tilt=-6.0)
    box(0.1, 0.71, 0.0, 3.6, 0.1, 1.84, body, "Car waist")
    # the canopy: tinted glass, the windscreen sloping down to the bonnet, the roof
    box(0.25, 0.94, 0.0, 2.1, 0.36, 1.56, CAR_GLASS, "Car canopy (tinted glass)")
    box(-1.03, 0.9, 0.0, 0.62, 0.02, 1.5, CAR_GLASS, "Car windscreen", tilt=32.0)
    box(0.35, 1.1375, 0.0, 1.7, 0.035, 1.48, body, "Car roof")
    box(0.1, 0.765, 0.0, 3.62, 0.018, 1.86, CAR_TRIM, "Car chrome waist line")  # (a hair wider)
    for s in (-1.0, 1.0):
        box(0.1, 0.33, s * 0.957, 2.6, 0.03, 0.01, sill, "Car neon sill", tags=DECAL)
        box(-0.77, 0.86, s * 0.84, 0.1, 0.07, 0.16, CAR_DARK, "Car mirror")
    # lights, the plate, the wing
    box(-2.47, 0.49, 0.0, 0.03, 0.05, 1.5, HEADLIGHT, "Car headlight bar", tags=DECAL)
    box(2.36, 0.67, 0.0, 0.03, 0.06, 1.7, TAILLIGHT, "Car tail light strip", tags=DECAL)
    box(2.35, 0.47, 0.0, 0.01, 0.1, 0.32, PLATE, "Car number plate", tags=DECAL)
    box(2.12, 1.0, 0.0, 0.32, 0.03, 1.8, CAR_DARK, "Car rear wing")
    for s in (-1.0, 1.0):
        box(2.12, 0.885, s * 0.6, 0.05, 0.2, 0.05, CAR_DARK, "Car wing post")
    for wx in COUPE_AXLES:  # the arches over the wheels
        for s in (-1.0, 1.0):
            box(wx, 2 * 0.36 + 0.05, s * 0.985, 1.02, 0.08, 0.11, CAR_DARK, "Car wheel arch")
    return b.props


COUPE_AXLES = (-1.45, 1.4)


RIM_GLOW = Material(None, (0.2, 0.9, 1.0), emission=(0.5, 2.2, 2.6))


def _coupe_wheel(paint: Paint, side: float) -> list[Prop]:
    return _wheel(0.36, 0.28, side, RIM, RIM_GLOW, 5, 0.25)


# ----------------------------------------------------------------------------- the taxi
TAXI_AXLES = (-1.5, 1.45)


def _taxi(paint: Paint) -> list[Prop]:
    body = lacquer(paint.body)
    b = _Parts()
    box = b.box
    half = 0.92
    # the hull, the bonnet and the boot lids, the cabin (glass all round), the roof
    box(0.0, 0.56, 0.0, 4.6, 0.56, 2 * half, body, "Taxi body")
    box(-1.75, 0.865, 0.0, 1.05, 0.06, 1.8, body, "Taxi bonnet", tilt=4.0)
    box(1.85, 0.87, 0.0, 0.85, 0.06, 1.8, body, "Taxi boot lid")
    box(0.1, 1.12, 0.0, 2.2, 0.56, 1.66, CAR_GLASS, "Taxi cabin (glass)")
    box(-1.13, 1.1, 0.0, 0.5, 0.02, 1.6, CAR_GLASS, "Taxi windscreen", tilt=40.0)
    box(1.33, 1.1, 0.0, 0.42, 0.02, 1.6, CAR_GLASS, "Taxi rear window", tilt=-40.0)
    box(0.1, 1.42, 0.0, 2.0, 0.05, 1.7, body, "Taxi roof")
    for s in (-1.0, 1.0):  # the pillars between the windows
        for px in (-0.7, 0.25, 1.1):
            box(px, 1.12, s * 0.836, 0.08, 0.56, 0.01, body, "Taxi door pillar")
        box(-0.95, 1.0, s * 0.95, 0.1, 0.07, 0.14, CAR_DARK, "Taxi mirror")
        # the chequer band: black squares along the yellow (a hair proud of the side)
        for k in range(12):
            box(-1.65 + k * 0.3, 0.66 if k % 2 == 0 else 0.55, s * (half + 0.003), 0.3, 0.11,
                0.006, CAR_DARK, "Taxi chequer", tags=DECAL)  # fmt: skip
        box(0.2, 0.31, s * (half + 0.004), 2.2, 0.03, 0.008, neon(paint.glow), "Taxi neon sill",
            tags=DECAL)  # fmt: skip
    # bumpers, lamps, the plate
    box(-2.34, 0.42, 0.0, 0.1, 0.16, 1.86, CAR_DARK, "Taxi front bumper")
    box(2.34, 0.42, 0.0, 0.1, 0.16, 1.86, CAR_DARK, "Taxi rear bumper")
    for s in (-1.0, 1.0):
        box(-2.305, 0.7, s * 0.64, 0.02, 0.11, 0.36, HEADLIGHT, "Taxi headlight", tags=DECAL)
        box(2.305, 0.72, s * 0.66, 0.02, 0.13, 0.3, TAILLIGHT, "Taxi tail light", tags=DECAL)
    box(2.395, 0.47, 0.0, 0.01, 0.1, 0.34, PLATE, "Taxi number plate", tags=DECAL)
    # the TAXI box on the roof, lit
    box(0.05, 1.5, 0.0, 0.16, 0.12, 0.5, CAR_DARK, "Taxi sign base")
    lit = Material(None, (1.0, 0.85, 0.2), emission=(2.2, 1.7, 0.3))
    box(0.05, 1.64, 0.0, 0.24, 0.16, 0.86, lit, "Taxi roof sign (lit)")
    for wx in TAXI_AXLES:
        for s in (-1.0, 1.0):
            box(
                wx,
                2 * 0.34 + 0.04,
                s * (half + 0.02),
                0.95,
                0.07,
                0.07,
                CAR_DARK,
                "Taxi wheel arch",
            )
    return b.props


def _taxi_wheel(paint: Paint, side: float) -> list[Prop]:
    return _wheel(0.34, 0.25, side, Material("metal", (0.35, 0.36, 0.4)), None, 6, 0.22)


# ----------------------------------------------------------------------------- the van
VAN_AXLES = (-1.75, 1.75)
VAN_ADS = ("poster_fish", "poster_corp", "tv_ads_1", "poster_idol")


def _van(paint: Paint, ad: str = "poster_fish") -> list[Prop]:
    body = lacquer(paint.body)
    b = _Parts()
    box = b.box
    half = 0.99
    # the box body behind the cab (its floor over the wheels)
    box(0.5, 1.42, 0.0, 4.2, 1.9, 2 * half, body, "Van box body")
    # the cab: the lower cab, the glass, the roof over it, a raked windscreen
    box(-2.13, 0.9, 0.0, 0.96, 0.86, 1.96, body, "Van cab")
    box(-1.95, 1.77, 0.0, 0.6, 0.88, 1.9, CAR_GLASS, "Van cab glass")
    box(-1.97, 2.275, 0.0, 0.66, 0.13, 1.94, body, "Van cab roof")
    box(-2.43, 1.76, 0.0, 0.03, 0.92, 1.84, CAR_GLASS, "Van windscreen", tilt=-17.0)
    box(-2.62, 1.335, 0.0, 0.06, 0.02, 1.9, CAR_DARK, "Van windscreen wiper")
    for s in (-1.0, 1.0):
        box(-2.55, 1.55, s * 1.03, 0.12, 0.18, 0.06, CAR_DARK, "Van mirror")
        # down the box: an advertisement under a neon stripe, a sliding door's seam
        box(0.65, 1.55, s * (half + 0.003), 3.0, 1.05, 0.006, Material(ad, (1.0, 1.0, 1.0)),
            "Van advertisement", tags=DECAL)  # fmt: skip
        box(0.5, 2.22, s * (half + 0.004), 4.0, 0.06, 0.008, neon(paint.glow), "Van neon stripe",
            tags=DECAL)  # fmt: skip
        box(-1.25, 1.4, s * (half + 0.003), 0.02, 1.6, 0.006, CAR_DARK, "Van door seam", tags=DECAL)
        box(2.612, 1.4, s * 0.55, 0.02, 1.5, 0.08, CAR_DARK, "Van rear door hinge")
    # the back: two doors, tail lights up the corners, a step
    box(2.605, 1.5, 0.0, 0.01, 1.5, 0.02, CAR_DARK, "Van rear door seam", tags=DECAL)
    for s in (-1.0, 1.0):
        box(2.605, 1.0, s * 0.88, 0.01, 0.5, 0.14, TAILLIGHT, "Van tail light", tags=DECAL)
        box(-2.615, 0.92, s * 0.7, 0.01, 0.14, 0.32, HEADLIGHT, "Van headlight", tags=DECAL)
    box(2.68, 0.4, 0.0, 0.16, 0.08, 1.4, CAR_DARK, "Van rear step")
    box(-2.66, 0.55, 0.0, 0.08, 0.2, 1.96, CAR_DARK, "Van front bumper")
    box(2.605, 0.62, 0.0, 0.01, 0.1, 0.34, PLATE, "Van number plate", tags=DECAL)
    box(-1.95, 2.39, 0.0, 0.16, 0.1, 1.3, AMBER, "Van roof light bar")
    for wx in VAN_AXLES:
        for s in (-1.0, 1.0):
            box(wx, 2 * 0.38 + 0.05, s * (half + 0.02), 1.0, 0.08, 0.08, CAR_DARK, "Van wheel arch")
    return b.props


def _van_wheel(paint: Paint, side: float) -> list[Prop]:
    return _wheel(0.38, 0.28, side, Material("metal", (0.3, 0.31, 0.34)), None, 8, 0.24)


# ----------------------------------------------------------------------------- the models
@dataclass(frozen=True)
class VehicleModel:
    name: str
    length: float  # bumper to bumper (along x)
    width: float  # mirror to mirror
    height: float
    wheel_radius: float
    axles: tuple[float, ...]  # the axles' x (the nose at -x)
    track: float  # the wheels' middles are at z = -+track
    cruise: tuple[float, float]  # the speeds it likes to drive at (m/s)
    body: Callable[[Paint], list[Prop]]
    wheel: Callable[[Paint, float], list[Prop]]
    headlights: tuple[float, float]  # (x, y) of the headlights (for the pool they throw)
    tail: tuple[float, float]  # (x, y) of the tail lights (where the brake lights come on)
    underglow: bool = False

    def wheels(self) -> list[tuple[float, float, float]]:
        """(x, z, side) of each wheel's hub middle, ``side`` the way its outer wall faces."""
        return [(x, s * self.track, s) for x in self.axles for s in (-1.0, 1.0)]

    def parts(self, paint: Paint) -> list[Prop]:
        """The whole vehicle standing still (the body and its wheels), in its frame."""
        out = list(self.body(paint))
        for x, z, side in self.wheels():
            for p in self.wheel(paint, side):
                out.append(_moved(p, x, self.wheel_radius, z))
        return out


def _moved(p: Prop, dx: float, dy: float, dz: float) -> Prop:
    from dataclasses import replace

    return replace(p, x=p.x + dx, y=p.y + dy, z=p.z + dz)


MODELS: dict[str, VehicleModel] = {
    "coupe": VehicleModel(
        "coupe", 4.9, 2.08, 1.16, 0.36, COUPE_AXLES, 0.97 - 0.28 / 2.0 + 0.02, (12.0, 16.0),
        _coupe, _coupe_wheel, (-2.47, 0.49), (2.36, 0.67), underglow=True,
    ),
    "taxi": VehicleModel(
        "taxi", 4.8, 2.04, 1.72, 0.34, TAXI_AXLES, 0.92 - 0.25 / 2.0 + 0.02, (10.0, 13.5),
        _taxi, _taxi_wheel, (-2.31, 0.7), (2.305, 0.72),
    ),
    "van": VehicleModel(
        "van", 5.46, 2.12, 2.44, 0.38, VAN_AXLES, 0.99 - 0.28 / 2.0 + 0.02, (8.0, 11.0),
        lambda paint: _van(paint, _van_ad(paint)), _van_wheel, (-2.62, 0.92), (2.605, 1.0),
    ),
}  # fmt: skip
MODEL_WEIGHTS = {"coupe": 5, "taxi": 3, "van": 2}

PAINTS: dict[str, tuple[RGB, ...]] = {
    "coupe": (
        (0.42, 0.55, 0.72), (0.7, 0.08, 0.14), (0.9, 0.9, 0.92), (0.07, 0.07, 0.09),
        (0.5, 0.18, 0.72), (0.95, 0.5, 0.08), (0.12, 0.55, 0.45), (0.72, 0.73, 0.76),
    ),
    "taxi": ((0.95, 0.75, 0.08),),
    "van": ((0.85, 0.86, 0.88), (0.2, 0.22, 0.26), (0.55, 0.1, 0.12), (0.15, 0.3, 0.55)),
}  # fmt: skip
GLOWS: tuple[RGB, ...] = (
    (1.0, 0.3, 0.8), (0.2, 0.9, 1.0), (0.6, 1.0, 0.2), (1.0, 0.55, 0.1), (0.65, 0.3, 1.0),
)  # fmt: skip


def _van_ad(paint: Paint) -> str:
    return VAN_ADS[int(sum(paint.body) * 97.0 + sum(paint.glow) * 13.0) % len(VAN_ADS)]


def random_vehicle(rng: random.Random) -> tuple[VehicleModel, Paint]:
    """A model (coupes the commonest) and a paint job for it."""
    names = list(MODEL_WEIGHTS)
    name = rng.choices(names, weights=[MODEL_WEIGHTS[n] for n in names])[0]
    paint = Paint(rng.choice(PAINTS[name]), rng.choice(GLOWS))
    return MODELS[name], paint


def wheel_turn(distance: float, radius: float) -> float:
    """Degrees a wheel of ``radius`` turns rolling ``distance`` metres."""
    return math.degrees(distance / max(radius, 1e-6))


__all__ = [
    "MODELS",
    "Paint",
    "VehicleModel",
    "random_vehicle",
    "wheel_turn",
]


# ============================================================================ the metro
# A monorail car: its floor at the origin of its frame (level with the platform), x along
# it, z across it (the platform on its -z side), the beam it rides running through its
# skirts. Its doors are openings in its sides - a vestibule behind each, lit inside - and
# the leaves that close them are drawn apart (``render.metro``), sliding as they open.
CAR_LENGTH = 9.0
CAR_GAP = 0.6  # between the two cars (the gangway)
CAR_HALF_WIDTH = 1.4
CAR_HEIGHT = 2.75
DOOR_OFFSETS = (-2.3, 2.3)  # each car's doors, from its middle
DOOR_WIDTH = 1.3
DOOR_HEIGHT = 2.1
VESTIBULE = 0.6  # how deep a door's vestibule is (inside the side wall)
WALL = 0.1  # the side walls' thickness

TRAIN_WHITE = Material("metal", (0.86, 0.88, 0.9), specular=(1.0, 1.0, 1.0), shininess=80.0)
TRAIN_DARK = Material("metal", (0.1, 0.11, 0.13), specular=(0.6, 0.6, 0.65), shininess=50.0)
TRAIN_STRIPE = Material(None, (0.1, 0.85, 1.0), emission=(0.3, 1.9, 2.3))
TRAIN_STRIPE2 = Material(None, (1.0, 0.25, 0.7), emission=(2.0, 0.4, 1.3))
TRAIN_WINDOW = Material(None, (0.22, 0.26, 0.3), emission=(0.45, 0.52, 0.58))  # lit inside
TRAIN_INSIDE = Material(None, (0.45, 0.5, 0.52), emission=(0.16, 0.18, 0.19))
TRAIN_DARKNESS = Material(None, (0.004, 0.004, 0.006))  # the car beyond a door's vestibule
VESTIBULE_BACK = -CAR_HALF_WIDTH + WALL + VESTIBULE  # (z of its back, in the car's frame)
CAB_GLASS = Material(None, (0.05, 0.08, 0.12), specular=(1.0, 1.0, 1.0), shininess=110.0)


def door_xs(car_middle: float) -> tuple[float, ...]:
    """The middles of a car's doors (x along the train)."""
    return tuple(car_middle + d for d in DOOR_OFFSETS)


def train_car(cab: float) -> list[Prop]:
    """One car, a driver's cab at its ``cab`` end (-1: -x, +1: +x; 0: none)."""
    b = _Parts()
    box = b.box
    half, hw = CAR_LENGTH / 2.0, CAR_HALF_WIDTH
    top = CAR_HEIGHT
    # floor and roof, the rounded edge of the roof
    box(0.0, -0.1, 0.0, CAR_LENGTH, 0.2, 2 * hw, TRAIN_WHITE, "Metro car floor")
    box(0.0, top - 0.15, 0.0, CAR_LENGTH, 0.3, 2 * hw - 0.12, TRAIN_WHITE, "Metro car roof")
    for s in (-1.0, 1.0):
        box(
            0.0,
            top - 0.26,
            s * (hw - 0.065),
            CAR_LENGTH - 0.01,
            0.12,
            0.12,
            TRAIN_WHITE,
            "Metro car roof edge",
        )
    box(0.0, top + 0.06, 0.0, 3.0, 0.12, 1.2, TRAIN_DARK, "Metro car roof unit")
    # the sides: the platform side (-z) with its door openings, the far side whole (its
    # doors only outlined: they open on the platform side)
    d0 = [(m - DOOR_WIDTH / 2.0, m + DOOR_WIDTH / 2.0) for m in door_xs(0.0)]
    edges = [-half, d0[0][0], d0[0][1], d0[1][0], d0[1][1], half]
    z_wall = -hw + WALL / 2.0
    for a, c in zip(edges[0::2], edges[1::2], strict=True):
        box(
            (a + c) / 2.0,
            (top - 0.32) / 2.0,
            z_wall,
            c - a,
            top - 0.32,
            WALL,
            TRAIN_WHITE,
            "Metro car side",
        )
    above = top - 0.32 - DOOR_HEIGHT  # (the side over a door)
    vz = -hw + WALL + VESTIBULE / 2.0  # (the middle of a vestibule)
    vestibule_floor = Material(None, (0.3, 0.3, 0.33))
    for a, c in d0:
        m = (a + c) / 2.0
        box(m, DOOR_HEIGHT + above / 2.0, z_wall, c - a, above, WALL, TRAIN_WHITE,
            "Metro car side over a door")  # fmt: skip
        # the vestibule: lit walls either side of it, its back the inside of the car
        for e in (a, c):
            box(e + (0.02 if e == a else -0.02), DOOR_HEIGHT / 2.0, vz, 0.04, DOOR_HEIGHT,
                VESTIBULE, TRAIN_INSIDE, "Metro car vestibule wall")  # fmt: skip
        box(m, DOOR_HEIGHT + 0.02, vz, c - a, 0.04, VESTIBULE, TRAIN_INSIDE,
            "Metro car vestibule ceiling")  # fmt: skip
        box(m, 0.002, vz, c - a - 0.08, 0.004, VESTIBULE - 0.01, vestibule_floor,
            "Metro car vestibule floor", tags=DECAL)  # fmt: skip
        # its back: the way into the car, dark - whoever gets on walks on into it and is
        # gone into the dark (and whoever gets off comes out of it)
        box(m, DOOR_HEIGHT / 2.0, -hw + WALL + VESTIBULE - 0.004, c - a - 0.09, DOOR_HEIGHT - 0.01,
            0.004, TRAIN_DARKNESS, "Metro car doorway into the dark inside")  # fmt: skip
    box(
        0.0,
        top / 2.0 - 0.16,
        hw - WALL / 2.0,
        CAR_LENGTH,
        top - 0.32,
        WALL,
        TRAIN_WHITE,
        "Metro car side",
    )
    # the inside (lit): behind the vestibules, and the rest of the width
    inner = -hw + WALL + VESTIBULE
    box(0.0, (top - 0.32) / 2.0, (inner + hw - WALL) / 2.0, CAR_LENGTH - 0.02, top - 0.32 - 0.002,
        hw - WALL - inner, TRAIN_INSIDE, "Metro car inside")  # fmt: skip
    tall = top - 0.32 - 0.002
    for a, c in zip(edges[0::2], edges[1::2], strict=True):
        box((a + c) / 2.0, (top - 0.32) / 2.0, (-hw + WALL + inner) / 2.0, c - a - 0.01, tall,
            VESTIBULE, TRAIN_INSIDE, "Metro car inside")  # fmt: skip
    # windows (lit from inside) and the stripes along both sides
    for s in (-1.0, 1.0):
        face = s * (hw + 0.003)
        for a, c in zip(edges[0::2], edges[1::2], strict=True):
            if c - a > 0.8:
                box(
                    (a + c) / 2.0,
                    1.45,
                    face,
                    c - a - 0.3,
                    0.85,
                    0.006,
                    TRAIN_WINDOW,
                    "Metro car window",
                    tags=DECAL,
                )
        if s > 0:  # the far side's doors, outlined (closed)
            for a, c in d0:
                box(
                    (a + c) / 2.0,
                    1.45,
                    face,
                    c - a - 0.1,
                    0.85,
                    0.006,
                    TRAIN_WINDOW,
                    "Metro car window",
                    tags=DECAL,
                )
                for e in (a, c):
                    box(e, DOOR_HEIGHT / 2.0, s * (hw + 0.004), 0.02, DOOR_HEIGHT, 0.008,
                        TRAIN_DARK, "Metro car door seam", tags=DECAL)  # fmt: skip
        for a, c in zip(edges[0::2], edges[1::2], strict=True) if s < 0 else ((-half, half),):
            box(
                (a + c) / 2.0,
                0.55,
                face,
                c - a - 0.02,
                0.08,
                0.006,
                TRAIN_STRIPE,
                "Metro car stripe",
                tags=DECAL,
            )
            box(
                (a + c) / 2.0,
                0.42,
                face,
                c - a - 0.02,
                0.05,
                0.006,
                TRAIN_STRIPE2,
                "Metro car stripe",
                tags=DECAL,
            )
    # under the floor: the skirts either side of the beam, the bogies
    for s in (-1.0, 1.0):
        box(0.0, -0.5, s * 0.9, CAR_LENGTH - 0.4, 0.6, 0.1, TRAIN_DARK, "Metro car skirt")
        box(0.0, -0.2, s * 1.15, CAR_LENGTH - 0.2, 0.2, 0.4, TRAIN_WHITE, "Metro car underframe")
    for x in (-half + 1.6, half - 1.6):
        box(x, -0.58, 0.0, 1.6, 0.16, 1.85, TRAIN_DARK, "Metro car bogie")
    # the ends: a cab (a raked glass front, lamps) or a gangway to the next car
    for end in (-1.0, 1.0):
        x = end * half
        if end == cab:
            w = 2 * hw - 0.1
            # the nose: a low front, a windscreen raked back to the roof line, the wedge
            # under it filled in (in steps)
            box(x + end * 0.45, 0.65, 0.0, 0.9, 1.3, w, TRAIN_WHITE, "Metro car cab")
            rake = math.degrees(math.atan2(1.25, 0.85))
            box(x + end * 0.425, 1.925, 0.0, math.hypot(0.85, 1.25), 0.04, w - 0.1, CAB_GLASS,
                "Metro cab windscreen", tilt=-end * rake)  # fmt: skip
            box(
                x + end * 0.2,
                1.625,
                0.0,
                0.4,
                0.65,
                w - 0.12,
                TRAIN_DARK,
                "Metro cab (under the windscreen)",
            )
            box(
                x + end * 0.55,
                1.4,
                0.0,
                0.3,
                0.2,
                w - 0.12,
                TRAIN_DARK,
                "Metro cab (under the windscreen)",
            )
            box(x + end * 0.92, 0.7, 0.0, 0.04, 0.5, w - 0.3, TRAIN_DARK, "Metro cab nose")
            for s in (-1.0, 1.0):
                box(
                    x + end * 0.945,
                    0.95,
                    s * 0.9,
                    0.02,
                    0.12,
                    0.3,
                    HEADLIGHT,
                    "Metro headlight",
                    tags=DECAL,
                )
                box(
                    x + end * 0.945,
                    0.32,
                    s * 0.9,
                    0.02,
                    0.08,
                    0.3,
                    TAILLIGHT,
                    "Metro tail light",
                    tags=DECAL,
                )
            box(
                x + end * 0.906,
                1.15,
                0.0,
                0.012,
                0.08,
                w - 0.2,
                TRAIN_STRIPE,
                "Metro cab stripe",
                tags=DECAL,
            )
        else:
            box(x + end * 0.15, 1.15, 0.0, 0.3, 2.3, 1.3, TRAIN_DARK, "Metro gangway bellows")
    return b.props


TRAIN_CARS = (-1.0, 1.0)  # the two cars: the first with its cab at -x, the second at +x


def car_middles(train_x: float) -> tuple[float, ...]:
    """The middles of the train's cars (x), the train's middle at ``train_x``."""
    step = (CAR_LENGTH + CAR_GAP) / 2.0
    return (train_x - step, train_x + step)


def train_doors(train_x: float) -> tuple[float, ...]:
    """The middles of every door of the train (x)."""
    return tuple(x for m in car_middles(train_x) for x in door_xs(m))
