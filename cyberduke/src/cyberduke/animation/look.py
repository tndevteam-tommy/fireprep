"""How somebody looks, as data: hair, eyes, chrome, clothes.

A city where organic cybernetics have been a way of life for a century: most people have
had their eyes replaced the day they came of age (a glowing iris, a chrome eye, a big
optic lens), many carry more chrome - a jaw plate, ports at the temples, an ear, a whole
arm - and nobody dresses quietly: neon trim, vinyl jackets, mohawks, half-shaved heads,
anime spikes, twin tails, dip-dyed tips.

``Look`` says which (``animation.rig`` builds the parts, ``render.humanoid`` colours
them with the ``HumanoidSpec``'s colours); ``street_look`` and ``guard_look`` draw one
from a seed - no two people alike, and the guards in one uniform but each their own.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

RGB = tuple[float, float, float]

HAIRSTYLES = (
    "cap",  # an ordinary head of hair
    "buzz",  # cropped to stubble
    "shaved",  # a bare scalp
    "mohawk",  # a crest of spikes front to back, the sides shaved
    "spikes",  # anime spikes every which way
    "undercut",  # hair on one side, the other shaved to stubble
    "bob",  # a bob with a heavy fringe
    "long",  # long, down the back
    "bun",  # a top knot
    "twintails",  # two tails tied high
)
EYES = ("organic", "glow", "chrome", "optic")
TOPS = ("tee", "jacket", "vest", "armor")
TRIMS = ("", "chest", "stripes", "hem", "rank")
IMPLANTS = ("jaw", "temples", "ear", "neck", "arm_l", "arm_r")
EXTRAS = ("respirator", "goggles", "shades", "earrings", "collar", "pads")
MARKS = ("", "circuit", "tattoo", "paint", "scar")  # on the face (``assets.faces``)


@dataclass(frozen=True)
class Look:
    hair: str = "cap"
    eyes: str = "organic"
    top: str = "tee"
    trim: str = ""
    implants: frozenset[str] = field(default_factory=frozenset)
    extras: frozenset[str] = field(default_factory=frozenset)
    soles: bool = False  # glowing soles
    stripe: bool = False  # a neon stripe down the outside of each leg
    seed: int = 0  # (which way the spikes go, which side is shaved...)


PLAIN = Look()

# Neon: what the city glows in
NEON: tuple[RGB, ...] = (
    (1.0, 0.15, 0.65), (0.1, 0.95, 1.0), (0.7, 1.0, 0.15), (1.0, 0.5, 0.05),
    (0.65, 0.25, 1.0), (1.0, 0.95, 0.2), (0.2, 1.0, 0.55), (1.0, 0.25, 0.25),
)  # fmt: skip
HAIR_COLOURS: tuple[RGB, ...] = (
    (1.0, 0.2, 0.7), (0.15, 0.9, 1.0), (0.6, 1.0, 0.2), (0.95, 0.95, 0.98),
    (0.55, 0.25, 1.0), (1.0, 0.45, 0.1), (0.1, 0.1, 0.12), (0.9, 0.1, 0.15),
    (1.0, 0.85, 0.2), (0.2, 0.35, 1.0), (0.45, 0.28, 0.15), (0.2, 1.0, 0.6),
)  # fmt: skip
CLOTHES: tuple[RGB, ...] = (
    (1.0, 0.2, 0.55), (0.1, 0.85, 1.0), (0.75, 1.0, 0.2), (1.0, 0.55, 0.1),
    (0.6, 0.25, 1.0), (0.1, 0.1, 0.12), (0.95, 0.95, 0.95), (1.0, 0.9, 0.15),
    (0.15, 0.2, 0.5), (0.9, 0.1, 0.15), (0.05, 0.4, 0.35), (0.35, 0.1, 0.4),
)  # fmt: skip
EYE_COLOURS: tuple[RGB, ...] = (
    (0.1, 0.9, 1.0), (1.0, 0.2, 0.6), (1.0, 0.7, 0.1), (0.4, 1.0, 0.3),
    (0.8, 0.3, 1.0), (1.0, 0.15, 0.15), (0.9, 0.95, 1.0),
)  # fmt: skip


@dataclass(frozen=True)
class Palette:
    """A person's colours, drawn with their look (``HumanoidSpec`` takes them)."""

    hair: RGB
    hair2: RGB  # dyed tips, a streak, the other colour of a two-tone head
    shirt: RGB
    jacket: RGB
    pants: RGB
    neon: RGB  # trim, soles, stripes, the glow of implants
    eye: RGB
    boots: RGB


def _pick(rng: random.Random, seq, not_=None):
    choice = rng.choice(seq)
    while not_ is not None and choice == not_ and len(seq) > 1:
        choice = rng.choice(seq)
    return choice


def street_look(rng: random.Random) -> tuple[Look, Palette, str]:
    """A look for somebody of the street - loud, chromed, their own - their colours, and
    the mark on their face (``MARKS``)."""
    hair = _pick(rng, HAIRSTYLES[1:])  # (hardly anybody has "ordinary" hair)
    eyes = rng.choices(EYES, weights=(1, 5, 3, 2))[0]  # (most have had them replaced)
    top = rng.choices(TOPS[:3], weights=(3, 5, 2))[0]
    trim = rng.choices(TRIMS[:4], weights=(1, 3, 2, 2))[0]
    implants = frozenset(rng.sample(IMPLANTS, rng.randint(1, 3)))
    if "arm_l" in implants and "arm_r" in implants:
        implants = implants - {"arm_l"}
    extras = frozenset(rng.sample(EXTRAS[:5], rng.randint(0, 2)))
    if eyes == "optic" and "shades" in extras:
        extras = extras - {"shades"}
    if top == "jacket" and rng.random() < 0.5:
        extras = extras | {"pads"}
    look = Look(
        hair=hair, eyes=eyes, top=top, trim=trim, implants=implants, extras=extras,
        soles=rng.random() < 0.6, stripe=rng.random() < 0.45, seed=rng.randrange(1 << 16),
    )  # fmt: skip
    neon = _pick(rng, NEON)
    hair_c = _pick(rng, HAIR_COLOURS)
    palette = Palette(
        hair=hair_c,
        hair2=_pick(rng, HAIR_COLOURS, not_=hair_c),
        shirt=_pick(rng, CLOTHES),
        jacket=_pick(rng, CLOTHES),
        pants=_pick(rng, CLOTHES[5:] + ((0.2, 0.2, 0.25), (0.12, 0.12, 0.3))),
        neon=neon,
        eye=_pick(rng, EYE_COLOURS),
        boots=_pick(rng, ((0.08, 0.08, 0.09), (0.9, 0.9, 0.92), (0.5, 0.1, 0.15), neon)),
    )
    mark = rng.choices(MARKS, weights=(3, 2, 2, 2, 1))[0]
    return look, palette, mark


# (regulation cuts, close to the skull: a guard goes down hard when his cyberware is
# fried, and a crest of spikes would go through the floor he lands on)
GUARD_HAIR = ("buzz", "cap", "shaved")
GUARD_VISORS: tuple[RGB, ...] = ((1.0, 0.1, 0.1), (1.0, 0.35, 0.05), (1.0, 0.1, 0.45))
GUARD_PANTS: tuple[RGB, ...] = ((0.9, 0.9, 0.95), (0.55, 0.58, 0.7), (0.6, 0.6, 0.55))


def guard_look(seed: int, elite: bool = False) -> tuple[Look, Palette, RGB]:
    """A Halcyon guard: the same armour, visor and sidearm as every other - but their own
    head of hair, their own rank stripes on the chest plate, a jaw plate on some, the
    visor a slightly different red. Returns the look, the colours and the visor's colour."""
    rng = random.Random(seed * 7919 + 13)
    implants = frozenset({"jaw"}) if rng.random() < 0.4 else frozenset()
    look = Look(
        hair=rng.choice(GUARD_HAIR), eyes="visor", top="armor", trim="rank",
        implants=implants, soles=False, stripe=rng.random() < 0.5, seed=rng.randrange(1 << 16),
    )  # fmt: skip
    hair = rng.choice(((0.08, 0.07, 0.06), (0.9, 0.9, 0.92), (0.15, 0.8, 1.0), (0.9, 0.15, 0.2),
                       (0.35, 0.22, 0.12), (0.55, 0.2, 0.9)))  # fmt: skip
    rank = (
        (1.0, 0.75, 0.15)
        if elite
        else rng.choice(((1.0, 0.55, 0.1), (0.1, 0.85, 1.0), (1.0, 0.2, 0.2)))
    )
    palette = Palette(
        hair=hair, hair2=hair, shirt=(1.0, 1.0, 1.0), jacket=(1.0, 1.0, 1.0),
        pants=rng.choice(GUARD_PANTS), neon=rank, eye=(1.0, 0.1, 0.1), boots=(1.0, 1.0, 1.0),
    )  # fmt: skip
    visor = (1.0, 0.75, 0.1) if elite else rng.choice(GUARD_VISORS)
    return look, palette, visor


__all__ = [
    "EYES",
    "HAIRSTYLES",
    "MARKS",
    "NEON",
    "PLAIN",
    "Look",
    "Palette",
    "guard_look",
    "street_look",
]
