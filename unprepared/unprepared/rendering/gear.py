"""The physical equipment: every gear kind as a textured model mounted on a figure.

A figure stands with its feet at the local origin and its front along -z (``models.py``
puts the world transform around it). :func:`draw_figure` draws the person underneath -
hose, head, hair - and then each worn item in body order: feet, body, neck, head, weapon.
Each gear kind is one function in ``GEAR_DRAWERS`` taking the tint it was given; the
townsfolk's pieces (the watch's breastplate and kettle helm, the smith's leather apron,
Jae's hood, Barnaby's cap and sash ...) are drawn here box for box as they always were,
and the same functions dress the player in the [Tab] view.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from ..core.math3d import Vector3D
from ..entities.equipment import EquipSlot, Figure, SLOT_ORDER
from .primitives import draw_textured_box

Material = Callable[..., None]
Color = Tuple[float, float, float]

# the figure's proportions (shared with models.py)
BODY_HALF_X, BODY_HALF_Z, BODY_HEIGHT = 0.34, 0.2, 1.3
HEAD_HALF, HEAD_BASE = 0.2, 1.34
EYE_Y = HEAD_BASE + 0.26
WAIST = 0.62                    # where the tunic ends and the legs begin
HAT_TOP = HEAD_BASE + 2 * HEAD_HALF
WHITE: Color = (1.0, 1.0, 1.0)
BELT: Color = (0.5, 0.4, 0.3)
GOLD: Color = (1.0, 0.85, 0.3)
LAYER = 0.012                   # the least gap between two layers of a figure that face the same way
STEEL: Color = (0.8, 0.8, 0.85)


def _box(material, textures, key, color, min_pt, max_pt) -> None:
    material(color=color, textured=True)
    draw_textured_box(min_pt, max_pt, textures[key], tile=None)


def _belt(material, textures) -> None:
    _box(material, textures, "leather", BELT, Vector3D(-BODY_HALF_X - 0.01, WAIST - 0.05, -BODY_HALF_Z - 0.01),
         Vector3D(BODY_HALF_X + 0.01, WAIST + 0.05, BODY_HALF_Z + 0.01))


# ------------------------------------------------------------------------------------ body
def draw_tunic(material, textures, color: Optional[Color]) -> None:
    """A cloth tunic over the hose, belted at the waist."""
    _box(material, textures, "cloth", color or (0.55, 0.45, 0.7), Vector3D(-BODY_HALF_X, WAIST - 0.04, -BODY_HALF_Z),
         Vector3D(BODY_HALF_X, BODY_HEIGHT, BODY_HALF_Z))
    _belt(material, textures)


def draw_breastplate(material, textures, color: Optional[Color]) -> None:
    """The watch's steel: the plate over the whole torso."""
    _box(material, textures, "guard", color or WHITE, Vector3D(-BODY_HALF_X, WAIST - 0.04, -BODY_HALF_Z),
         Vector3D(BODY_HALF_X, BODY_HEIGHT, BODY_HALF_Z))


def draw_chain_shirt(material, textures, color: Optional[Color]) -> None:
    """A shirt of mail: iron rings to the hips, belted."""
    _box(material, textures, "iron", color or (0.62, 0.63, 0.66), Vector3D(-BODY_HALF_X - 0.005, WAIST - 0.1, -BODY_HALF_Z - 0.005),
         Vector3D(BODY_HALF_X + 0.005, BODY_HEIGHT, BODY_HALF_Z + 0.005))
    _belt(material, textures)


def draw_cuirass(material, textures, color: Optional[Color]) -> None:
    """Plate: a breastplate with pauldrons on the shoulders and a plated skirt."""
    tint = color or STEEL
    _box(material, textures, "guard", tint, Vector3D(-BODY_HALF_X, WAIST - 0.04, -BODY_HALF_Z),
         Vector3D(BODY_HALF_X, BODY_HEIGHT, BODY_HALF_Z))
    for sx in (-1.0, 1.0):
        x = sx * BODY_HALF_X
        _box(material, textures, "guard", tint, Vector3D(x - 0.07, BODY_HEIGHT - 0.2, -BODY_HALF_Z - 0.03),
             Vector3D(x + 0.07, BODY_HEIGHT + 0.05, BODY_HALF_Z + 0.03))                    # pauldrons
    _box(material, textures, "guard", tint, Vector3D(-BODY_HALF_X - 0.02, WAIST - 0.22, -BODY_HALF_Z - 0.02),
         Vector3D(BODY_HALF_X + 0.02, WAIST - 0.04, BODY_HALF_Z + 0.02))                    # faulds


# ------------------------------------------------------------------------------------ neck
def draw_sash(material, textures, color: Optional[Color]) -> None:
    """A band across the chest (a merchant's finery)."""
    _box(material, textures, "cloth", color or (0.85, 0.7, 0.25), Vector3D(-0.1, WAIST + 0.08, -BODY_HALF_Z - 0.015),
         Vector3D(0.12, BODY_HEIGHT - 0.02, -BODY_HALF_Z + 0.02))


def _apron(material, textures, key: str, color: Color) -> None:
    _box(material, textures, key, color, Vector3D(-0.26, 0.22, -BODY_HALF_Z - 0.035),
         Vector3D(0.26, BODY_HEIGHT - 0.12, -BODY_HALF_Z + 0.01))                          # hangs down the front
    _box(material, textures, key, color, Vector3D(-0.09, BODY_HEIGHT - 0.12, -BODY_HALF_Z - 0.03),
         Vector3D(0.09, BODY_HEIGHT - 0.02, -BODY_HALF_Z + 0.01))                          # the bib's strap


def draw_apron_cloth(material, textures, color: Optional[Color]) -> None:
    _apron(material, textures, "cloth", color or (0.93, 0.9, 0.84))


def draw_apron_leather(material, textures, color: Optional[Color]) -> None:
    _apron(material, textures, "leather", color or WHITE)


def draw_necklace(material, textures, color: Optional[Color]) -> None:
    """A fine chain round the neck with a jewel on the chest."""
    _box(material, textures, "iron", GOLD, Vector3D(-0.13, BODY_HEIGHT - 0.05, -BODY_HALF_Z - 0.02),
         Vector3D(0.13, BODY_HEIGHT - 0.03, -BODY_HALF_Z + 0.005))                         # the chain at the front
    for sx in (-1.0, 1.0):
        _box(material, textures, "iron", GOLD, Vector3D(sx * 0.13 - 0.01, BODY_HEIGHT - 0.05 + LAYER, -BODY_HALF_Z - 0.02 + LAYER),
             Vector3D(sx * 0.13 + 0.01, BODY_HEIGHT - 0.01, -BODY_HALF_Z - 0.005))          # up over the collarbones
    _box(material, textures, "iron", GOLD, Vector3D(-0.012, BODY_HEIGHT - 0.12, -BODY_HALF_Z - 0.02),
         Vector3D(0.012, BODY_HEIGHT - 0.05, -BODY_HALF_Z - 0.01))                         # the drop
    _box(material, textures, "cloth", color or (0.85, 0.1, 0.15), Vector3D(-0.035, BODY_HEIGHT - 0.19, -BODY_HALF_Z - 0.045),
         Vector3D(0.035, BODY_HEIGHT - 0.12, -BODY_HALF_Z - 0.01))                         # the jewel


def draw_cloak(material, textures, color: Optional[Color]) -> None:
    """A cloak clasped at the throat: over the shoulders and down the back to the calves."""
    tint = color or (0.45, 0.38, 0.3)
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X - 0.04, 0.28, BODY_HALF_Z + 0.005),
         Vector3D(BODY_HALF_X + 0.04, BODY_HEIGHT + 0.01, BODY_HALF_Z + 0.07 - LAYER))     # down the back
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X - 0.05, BODY_HEIGHT - 0.14, -BODY_HALF_Z - 0.02),
         Vector3D(BODY_HALF_X + 0.05, BODY_HEIGHT + 0.03, BODY_HALF_Z + 0.07))             # over the shoulders
    for sx in (-1.0, 1.0):
        x = sx * (BODY_HALF_X + 0.02)
        _box(material, textures, "cloth", tint, Vector3D(x - 0.03, 0.55, -BODY_HALF_Z - 0.01 - LAYER),
             Vector3D(x + 0.03, BODY_HEIGHT - 0.14, BODY_HALF_Z + 0.07 - LAYER / 2))       # the sides hang down
    _box(material, textures, "iron", GOLD, Vector3D(-0.03, BODY_HEIGHT - 0.11, -BODY_HALF_Z - 0.045),
         Vector3D(0.03, BODY_HEIGHT - 0.05, -BODY_HALF_Z - 0.015))                         # the clasp


# ------------------------------------------------------------------------------------ head
def draw_helmet(material, textures, color: Optional[Color]) -> None:
    """A steel kettle helm with a brim and a nasal bar."""
    tint = color or (0.6, 0.62, 0.66)
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.03, HAT_TOP - 0.11, -HEAD_HALF - 0.03),
         Vector3D(HEAD_HALF + 0.03, HAT_TOP + 0.1, HEAD_HALF + 0.03))
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.07, HAT_TOP - 0.11 - LAYER, -HEAD_HALF - 0.12),
         Vector3D(HEAD_HALF + 0.07, HAT_TOP - 0.07, HEAD_HALF + 0.07))                      # the brim
    _box(material, textures, "cloth", tint, Vector3D(-0.03, EYE_Y - 0.06, -HEAD_HALF - 0.03 - LAYER),
         Vector3D(0.03, HAT_TOP - 0.1, -HEAD_HALF + 0.0))                                  # nasal bar (proud of the dome)


def draw_cap(material, textures, color: Optional[Color]) -> None:
    """A cloth cap with a peak."""
    tint = color or (0.35, 0.3, 0.25)
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.03, HAT_TOP - 0.1, -HEAD_HALF - 0.03),
         Vector3D(HEAD_HALF + 0.03, HAT_TOP + 0.07, HEAD_HALF + 0.03))
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF, HAT_TOP - 0.1 - LAYER, -HEAD_HALF - 0.14),
         Vector3D(HEAD_HALF, HAT_TOP - 0.07, -HEAD_HALF))                                  # peak


def draw_leather_cap(material, textures, color: Optional[Color]) -> None:
    """A snug leather skullcap, laced under the chin."""
    tint = color or (0.5, 0.36, 0.24)
    _box(material, textures, "leather", tint, Vector3D(-HEAD_HALF - 0.025, HAT_TOP - 0.16, -HEAD_HALF - 0.025),
         Vector3D(HEAD_HALF + 0.025, HAT_TOP + 0.035, HEAD_HALF + 0.025))
    for sx in (-1.0, 1.0):
        x = sx * (HEAD_HALF + 0.02)
        _box(material, textures, "leather", tint, Vector3D(x - 0.012, HEAD_BASE + 0.08, -0.02),
             Vector3D(x + 0.012, HAT_TOP - 0.16, 0.04))                                    # ear flaps


def draw_scarf(material, textures, color: Optional[Color]) -> None:
    """A scarf wrapped round the sides of the head; the face shows."""
    tint = color or (0.2, 0.2, 0.24)
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.03, HEAD_BASE + 0.02, -HEAD_HALF - 0.03),
         Vector3D(HEAD_HALF + 0.03, HAT_TOP + 0.04, HEAD_HALF + 0.06))
    _box(material, textures, "skin", WHITE, Vector3D(-HEAD_HALF + 0.02, HEAD_BASE + 0.03, -HEAD_HALF - 0.035),
         Vector3D(HEAD_HALF - 0.02, HAT_TOP - 0.1, -HEAD_HALF - 0.034))


def draw_coif(material, textures, color: Optional[Color]) -> None:
    tint = color or (0.92, 0.9, 0.85)
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.02, HEAD_BASE + 0.05, -HEAD_HALF - 0.02),
         Vector3D(HEAD_HALF + 0.02, HAT_TOP + 0.03, HEAD_HALF + 0.04))
    _box(material, textures, "skin", WHITE, Vector3D(-HEAD_HALF + 0.025, HEAD_BASE + 0.05, -HEAD_HALF - 0.025),
         Vector3D(HEAD_HALF - 0.025, HAT_TOP - 0.06, -HEAD_HALF - 0.024))


def draw_hood(material, textures, color: Optional[Color]) -> None:
    """A deep hood with a cowl over the shoulders; the face in shadow. Hides the hair."""
    tint = color or (0.2, 0.18, 0.22)
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.05, HEAD_BASE - 0.05, -HEAD_HALF - 0.02 - LAYER / 2),
         Vector3D(HEAD_HALF + 0.05, HAT_TOP + 0.09, HEAD_HALF + 0.1))                  # (its front clear of a cloak's)
    _box(material, textures, "skin", (0.8, 0.8, 0.8), Vector3D(-HEAD_HALF + 0.03, HEAD_BASE + 0.02, -HEAD_HALF - 0.031),
         Vector3D(HEAD_HALF - 0.03, HAT_TOP - 0.08, -HEAD_HALF - 0.03))
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X - 0.02, BODY_HEIGHT - 0.3, -BODY_HALF_Z - 0.02 - LAYER),
         Vector3D(BODY_HALF_X + 0.02, BODY_HEIGHT + 0.02, BODY_HALF_Z + 0.02))              # cowl over the shoulders


# ------------------------------------------------------------------------------------ feet
def _footwear(material, textures, key: str, color: Color, height: float, toe: float) -> None:
    for sx in (-1.0, 1.0):
        x1, x2 = (0.03, BODY_HALF_X + 0.03) if sx > 0 else (-BODY_HALF_X - 0.03, -0.03)
        _box(material, textures, key, color, Vector3D(x1, 0.0, -BODY_HALF_Z - toe), Vector3D(x2, height, BODY_HALF_Z + 0.02))


def draw_boots(material, textures, color: Optional[Color]) -> None:
    """Leather boots to the calf."""
    _footwear(material, textures, "leather", color or (0.42, 0.28, 0.18), 0.3, 0.07)


def draw_shoes(material, textures, color: Optional[Color]) -> None:
    _footwear(material, textures, "leather", color or (0.35, 0.25, 0.18), 0.12, 0.06)


def draw_slippers(material, textures, color: Optional[Color]) -> None:
    """Soft cloth slippers, barely above the sole."""
    _footwear(material, textures, "cloth", color or (0.62, 0.5, 0.45), 0.08, 0.05)


# ------------------------------------------------------------------------------------ weapons
def _hilt(material, textures, x1: float, x2: float, y: float, z1: float, z2: float, grip: Color, pommel: Color) -> None:
    cx = (x1 + x2) / 2.0
    _box(material, textures, "iron", STEEL, Vector3D(x1 - 0.05, y, z1 - 0.03), Vector3D(x2 + 0.05, y + 0.025, z2 + 0.03))   # crossguard
    _box(material, textures, "leather", grip, Vector3D(cx - 0.018, y + 0.025, -0.018), Vector3D(cx + 0.018, y + 0.14, 0.018))
    _box(material, textures, "iron", pommel, Vector3D(cx - 0.03, y + 0.14, -0.03), Vector3D(cx + 0.03, y + 0.19, 0.03))


def draw_dagger(material, textures, color: Optional[Color], pommel: Color = STEEL, jewel: Optional[Color] = None) -> None:
    """A dagger sheathed at the right hip (the figure's right is local -x)."""
    x1, x2 = -BODY_HALF_X - 0.045, -BODY_HALF_X - 0.005
    _box(material, textures, "leather", (0.35, 0.25, 0.18), Vector3D(x1, 0.34, -0.05), Vector3D(x2, WAIST - 0.02, 0.0))   # sheath
    _box(material, textures, "iron", color or STEEL, Vector3D(x1 + 0.012, WAIST - 0.02, -0.035), Vector3D(x2 - 0.012, WAIST + 0.0, -0.015))
    _hilt(material, textures, x1, x2, WAIST, -0.05, 0.0, (0.3, 0.2, 0.15), pommel)
    if jewel is not None:
        cx = (x1 + x2) / 2.0
        _box(material, textures, "cloth", jewel, Vector3D(cx - 0.018, WAIST + 0.19, -0.018), Vector3D(cx + 0.018, WAIST + 0.225, 0.018))


def draw_jeweled_dagger(material, textures, color: Optional[Color]) -> None:
    draw_dagger(material, textures, color, pommel=GOLD, jewel=(0.85, 0.1, 0.15))


def draw_sword(material, textures, color: Optional[Color]) -> None:
    """A sword hanging at the left hip (local +x), point towards the ground."""
    x1, x2 = BODY_HALF_X + 0.005, BODY_HALF_X + 0.045       # (clear of a cloak's side at +0.05)
    _box(material, textures, "leather", (0.3, 0.22, 0.16), Vector3D(x1, 0.1, -0.055), Vector3D(x2, WAIST - 0.02, 0.005))  # scabbard
    _box(material, textures, "iron", color or STEEL, Vector3D(x1 + 0.01, 0.04, -0.04), Vector3D(x2 - 0.01, 0.1, -0.01))    # the chape
    _hilt(material, textures, x1, x2, WAIST - 0.02, -0.055, 0.005, (0.28, 0.18, 0.12), STEEL)


def draw_sling(material, textures, color: Optional[Color]) -> None:
    """A leather sling: two cords and a pouch looped over the belt at the right hip."""
    tint = color or (0.45, 0.33, 0.22)
    x1, x2 = -BODY_HALF_X - 0.03 - LAYER, -BODY_HALF_X - 0.005     # (clear of a boot's outer face)
    for z in (-0.06, 0.02):
        _box(material, textures, "leather", tint, Vector3D(x1, 0.3, z), Vector3D(x2, WAIST + 0.03, z + 0.015))           # cords
    _box(material, textures, "leather", tint, Vector3D(x1 - 0.01, 0.25, -0.07), Vector3D(x2 + 0.01, 0.335, 0.045))     # pouch


# ------------------------------------------------------------------------------------ finery and treasures
def _glow(material, textures, key, color, glow, min_pt, max_pt) -> None:
    """A box that shines by itself (a legendary piece's runes or gem): lit or not, it shows."""
    material(color=color, emission=glow, textured=True, fog=0.4)
    draw_textured_box(min_pt, max_pt, textures[key], tile=None)


def draw_doublet(material, textures, color: Optional[Color]) -> None:
    """A rich man's doublet: dyed cloth with gilt trim down the front, a gold chain of office."""
    tint = color or (0.35, 0.12, 0.3)
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X, WAIST - 0.04, -BODY_HALF_Z),
         Vector3D(BODY_HALF_X, BODY_HEIGHT, BODY_HALF_Z))
    _box(material, textures, "iron", GOLD, Vector3D(-0.032, WAIST - 0.02, -BODY_HALF_Z - LAYER),
         Vector3D(0.032, BODY_HEIGHT - 0.035, -BODY_HALF_Z + 0.004))                         # the gilt placket (clear of a pendant)
    for sx in (-1.0, 1.0):
        _box(material, textures, "iron", GOLD, Vector3D(sx * 0.2 - 0.025, BODY_HEIGHT - 0.32, -BODY_HALF_Z - 2 * LAYER),
             Vector3D(sx * 0.2 + 0.025, BODY_HEIGHT - 0.02, -BODY_HALF_Z - LAYER + 0.004))   # the chain's two runs
    _box(material, textures, "iron", GOLD, Vector3D(-0.2, BODY_HEIGHT - 0.36, -BODY_HALF_Z - 2 * LAYER),
         Vector3D(0.2, BODY_HEIGHT - 0.32, -BODY_HALF_Z - LAYER + 0.004))
    _box(material, textures, "iron", (0.85, 0.1, 0.2), Vector3D(-0.045, BODY_HEIGHT - 0.45, -BODY_HALF_Z - 3 * LAYER),
         Vector3D(0.045, BODY_HEIGHT - 0.36, -BODY_HALF_Z - LAYER))                         # a ruby medallion
    _belt(material, textures)


def draw_gown(material, textures, color: Optional[Color]) -> None:
    """A lady's gown: a fitted bodice and a skirt to the ankles, over the hose; pearls at the neck."""
    tint = color or (0.25, 0.35, 0.6)
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X, WAIST - 0.04, -BODY_HALF_Z),
         Vector3D(BODY_HALF_X, BODY_HEIGHT, BODY_HALF_Z))
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X - 0.06, 0.06, -BODY_HALF_Z - 0.078),
         Vector3D(BODY_HALF_X + 0.06, WAIST + 0.02, BODY_HALF_Z + 0.078))                    # the skirt (over a boot's toe)
    _box(material, textures, "iron", GOLD, Vector3D(-BODY_HALF_X - 0.012, WAIST - 0.03, -BODY_HALF_Z - 0.012),
         Vector3D(BODY_HALF_X + 0.012, WAIST + 0.035, BODY_HALF_Z + 0.012))                  # a gilt girdle
    for i in range(5):
        x = -0.12 + i * 0.06
        _box(material, textures, "cloth", (0.96, 0.95, 0.9), Vector3D(x - 0.018, BODY_HEIGHT - 0.09, -BODY_HALF_Z - 0.028),
             Vector3D(x + 0.018, BODY_HEIGHT - 0.055, -BODY_HALF_Z - LAYER))                # pearls (behind a hood's cowl)


def draw_mantle(material, textures, color: Optional[Color]) -> None:
    """A fur-trimmed mantle over the shoulders, falling to the knees behind."""
    tint = color or (0.5, 0.1, 0.15)
    # (its edges clear of a breastplate's pauldrons and faulds and of a hood's rim)
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X - 0.04, 0.42, BODY_HALF_Z + 0.005),
         Vector3D(BODY_HALF_X + 0.04, BODY_HEIGHT + 0.01, BODY_HALF_Z + 0.06))                # the back
    _box(material, textures, "cloth", (0.92, 0.9, 0.85), Vector3D(-BODY_HALF_X - 0.075, BODY_HEIGHT - 0.1, -BODY_HALF_Z - 0.05),
         Vector3D(BODY_HALF_X + 0.075, BODY_HEIGHT + 0.055, BODY_HALF_Z + 0.08))              # ermine collar


def draw_feather_cloak(material, textures, color: Optional[Color]) -> None:
    """The Ravenfeather Cloak: black, layered in rows of glossy feathers, a silver clasp."""
    tint = color or (0.07, 0.07, 0.1)
    draw_cloak(material, textures, tint)
    for row in range(5):
        y = 0.36 + row * 0.19
        shade = 0.1 + 0.03 * (row % 2)
        _box(material, textures, "leather", (shade, shade, shade + 0.05),
             Vector3D(-BODY_HALF_X - 0.05, y, BODY_HALF_Z + 0.07), Vector3D(BODY_HALF_X + 0.05, y + 0.08, BODY_HALF_Z + 0.07 + LAYER))
    _box(material, textures, "iron", STEEL, Vector3D(-0.04, BODY_HEIGHT - 0.12, -BODY_HALF_Z - 0.06),
         Vector3D(0.04, BODY_HEIGHT - 0.04, -BODY_HALF_Z - 0.045))


def draw_legend_cloak(material, textures, color: Optional[Color]) -> None:
    """The Nightweave Cloak: woven of the dark itself - a deep cowled cloak edged in faintly shining
    violet runes that pulse like a held breath. Anyone who has seen it remembers it."""
    tint = color or (0.05, 0.04, 0.09)
    draw_cloak(material, textures, tint)
    _box(material, textures, "cloth", tint, Vector3D(-BODY_HALF_X - 0.06, BODY_HEIGHT - 0.05, -BODY_HALF_Z - 0.05),
         Vector3D(BODY_HALF_X + 0.06, BODY_HEIGHT + 0.16, BODY_HALF_Z + 0.09))                # a high, stiff collar
    rune = (0.55, 0.3, 1.0)
    glow = (0.45, 0.2, 0.9)
    for i in range(6):                                                                      # runes down the back
        y = 0.4 + i * 0.15
        _glow(material, textures, "cloth", rune, glow, Vector3D(-0.05 + 0.03 * (i % 2), y, BODY_HALF_Z + 0.07 + LAYER),
              Vector3D(0.05 + 0.03 * (i % 2), y + 0.07, BODY_HALF_Z + 0.07 + 2 * LAYER))
    for sx in (-1.0, 1.0):                                                                  # a shining hem down each edge
        x = sx * (BODY_HALF_X + 0.045)
        _glow(material, textures, "cloth", rune, glow, Vector3D(x - 0.012, 0.3, BODY_HALF_Z + 0.01),
              Vector3D(x + 0.012, BODY_HEIGHT - 0.05, BODY_HALF_Z + 0.075))
    _glow(material, textures, "iron", (0.8, 0.6, 1.0), (0.6, 0.4, 1.0), Vector3D(-0.045, BODY_HEIGHT - 0.13, -BODY_HALF_Z - 0.07),
          Vector3D(0.045, BODY_HEIGHT - 0.04, -BODY_HALF_Z - 0.05))                        # the moonstone clasp


def draw_amulet(material, textures, color: Optional[Color]) -> None:
    """The Amulet of the Open Door: a heavy gold chain and a great faceted gem that burns a cold,
    living cyan - shaped like a keyhole, light spilling out of it."""
    gem = color or (0.2, 0.95, 0.9)
    _box(material, textures, "iron", GOLD, Vector3D(-0.15, BODY_HEIGHT - 0.05, -BODY_HALF_Z - 0.025),
         Vector3D(0.15, BODY_HEIGHT - 0.02, -BODY_HALF_Z + 0.005))
    for sx in (-1.0, 1.0):
        _box(material, textures, "iron", GOLD, Vector3D(sx * 0.12 - 0.012, BODY_HEIGHT - 0.26, -BODY_HALF_Z - 0.025 + LAYER / 2),
             Vector3D(sx * 0.12 + 0.012, BODY_HEIGHT - 0.05, -BODY_HALF_Z - 0.008))
    _box(material, textures, "iron", GOLD, Vector3D(-0.085, BODY_HEIGHT - 0.42, -BODY_HALF_Z - 0.035),
         Vector3D(0.085, BODY_HEIGHT - 0.24, -BODY_HALF_Z - 0.01))                          # the gold setting
    _glow(material, textures, "iron", gem, (gem[0] * 0.8, gem[1] * 0.8, gem[2] * 0.8),
          Vector3D(-0.055, BODY_HEIGHT - 0.36, -BODY_HALF_Z - 0.05), Vector3D(0.055, BODY_HEIGHT - 0.27, -BODY_HALF_Z - 0.035))
    _glow(material, textures, "iron", gem, gem, Vector3D(-0.02, BODY_HEIGHT - 0.41, -BODY_HALF_Z - 0.05),
          Vector3D(0.02, BODY_HEIGHT - 0.36, -BODY_HALF_Z - 0.035))                         # the keyhole's foot


def draw_choker(material, textures, color: Optional[Color]) -> None:
    """A pearl choker close about the throat, a drop pearl at the front."""
    pearl = color or (0.95, 0.94, 0.9)
    _box(material, textures, "cloth", pearl, Vector3D(-0.13, BODY_HEIGHT - 0.045, -BODY_HALF_Z - 0.03),
         Vector3D(0.13, BODY_HEIGHT - 0.005, -BODY_HALF_Z + 0.0))
    _box(material, textures, "cloth", pearl, Vector3D(-0.025, BODY_HEIGHT - 0.1, -BODY_HALF_Z - 0.045),
         Vector3D(0.025, BODY_HEIGHT - 0.045, -BODY_HALF_Z - 0.03))


def draw_feathered_hat(material, textures, color: Optional[Color]) -> None:
    """A broad velvet hat with a sweeping plume - nobody in town could miss who is wearing it."""
    tint = color or (0.3, 0.12, 0.35)
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.12, HAT_TOP - 0.08, -HEAD_HALF - 0.12),
         Vector3D(HEAD_HALF + 0.12, HAT_TOP - 0.04, HEAD_HALF + 0.12))                      # the brim
    _box(material, textures, "cloth", tint, Vector3D(-HEAD_HALF - 0.02, HAT_TOP - 0.04, -HEAD_HALF - 0.02),
         Vector3D(HEAD_HALF + 0.02, HAT_TOP + 0.12, HEAD_HALF + 0.02))                      # the crown
    _box(material, textures, "iron", GOLD, Vector3D(-HEAD_HALF - 0.02 - LAYER, HAT_TOP - 0.03, -HEAD_HALF - 0.02 - LAYER),
         Vector3D(HEAD_HALF + 0.02 + LAYER, HAT_TOP - 0.005, HEAD_HALF + 0.02 + LAYER))       # a gilt band
    _box(material, textures, "cloth", (0.95, 0.92, 0.8), Vector3D(HEAD_HALF - 0.02, HAT_TOP + 0.05, -0.04),
         Vector3D(HEAD_HALF + 0.04, HAT_TOP + 0.42, 0.06))                                  # the plume, upright ...
    _box(material, textures, "cloth", (0.95, 0.92, 0.8), Vector3D(HEAD_HALF + 0.04, HAT_TOP + 0.3, -0.03),
         Vector3D(HEAD_HALF + 0.16, HAT_TOP + 0.38, 0.05))                                  # ... and nodding


def draw_circlet(material, textures, color: Optional[Color]) -> None:
    """A fine circlet about the brow. (The Fingers of the Magpie: pale silver, its stones shining.)"""
    tint = color or (0.95, 0.8, 0.35)
    band = (Vector3D(-HEAD_HALF - 0.012, EYE_Y + 0.08, -HEAD_HALF - 0.012), Vector3D(HEAD_HALF + 0.012, EYE_Y + 0.13, HEAD_HALF + 0.012))
    if color is not None and color[2] > 0.95:                   # the legendary one shines
        _glow(material, textures, "iron", tint, (0.55, 0.6, 0.75), *band)
        for sx in (-0.12, 0.0, 0.12):
            _glow(material, textures, "iron", (0.4, 0.9, 1.0), (0.35, 0.85, 1.0),
                  Vector3D(sx - 0.025, EYE_Y + 0.085, -HEAD_HALF - 0.03), Vector3D(sx + 0.025, EYE_Y + 0.14, -HEAD_HALF - 0.012 - LAYER / 2))
    else:
        _box(material, textures, "iron", tint, *band)


def draw_shadow_hood(material, textures, color: Optional[Color]) -> None:
    """The Shadowsilk Hood: a deep hood whose inside is darker than night, its edge faintly violet."""
    tint = color or (0.1, 0.06, 0.14)
    draw_hood(material, textures, tint)
    _glow(material, textures, "cloth", (0.3, 0.15, 0.45), (0.16, 0.06, 0.3),
          Vector3D(-HEAD_HALF - 0.05 - LAYER, HEAD_BASE - 0.04, -HEAD_HALF - 0.02 - 2 * LAYER),
          Vector3D(HEAD_HALF + 0.05 + LAYER, HEAD_BASE + 0.01, -HEAD_HALF + 0.02))         # the violet edge


def draw_winged_boots(material, textures, color: Optional[Color]) -> None:
    """The Duskwalker Boots: soft dark boots with small silver wings at the heel."""
    tint = color or (0.1, 0.11, 0.16)
    _footwear(material, textures, "cloth", tint, 0.32, 0.06)
    for sx in (-1.0, 1.0):
        x = sx * (BODY_HALF_X + 0.03 + LAYER)
        _glow(material, textures, "iron", (0.8, 0.85, 0.95), (0.2, 0.25, 0.35), Vector3D(x - 0.008, 0.16, BODY_HALF_Z - 0.02),
              Vector3D(x + 0.008, 0.3, BODY_HALF_Z + 0.12))


GEAR_DRAWERS: Dict[str, Callable[[Material, object, Optional[Color]], None]] = {
    "doublet": draw_doublet, "gown": draw_gown, "mantle": draw_mantle, "feather_cloak": draw_feather_cloak,
    "legend_cloak": draw_legend_cloak, "amulet": draw_amulet, "choker": draw_choker, "feathered_hat": draw_feathered_hat,
    "circlet": draw_circlet, "shadow_hood": draw_shadow_hood, "winged_boots": draw_winged_boots,
    "tunic": draw_tunic, "breastplate": draw_breastplate, "chain_shirt": draw_chain_shirt, "cuirass": draw_cuirass,
    "sash": draw_sash, "apron_cloth": draw_apron_cloth, "apron_leather": draw_apron_leather, "necklace": draw_necklace,
    "cloak": draw_cloak,
    "helmet": draw_helmet, "cap": draw_cap, "leather_cap": draw_leather_cap, "scarf": draw_scarf, "coif": draw_coif,
    "hood": draw_hood,
    "boots": draw_boots, "shoes": draw_shoes, "slippers": draw_slippers,
    "dagger": draw_dagger, "jeweled_dagger": draw_jeweled_dagger, "sword": draw_sword, "sling": draw_sling,
}
HIDES_HAIR = ("hood",)

# drawing order: what is nearest the skin first, so later pieces sit over earlier ones
DRAW_ORDER: Tuple[EquipSlot, ...] = (EquipSlot.FEET, EquipSlot.BODY, EquipSlot.NECK, EquipSlot.HEAD, EquipSlot.WEAPON)
assert set(DRAW_ORDER) == set(SLOT_ORDER)


def draw_gear(material, textures, item) -> None:
    drawer = GEAR_DRAWERS.get(item.gear)
    if drawer is not None:
        drawer(material, textures, item.gear_color)


def draw_figure(figure: Figure, equipment: Dict[EquipSlot, object], textures, material: Material) -> None:
    """A dressed figure in local space: the hose, then everything worn over it in ``DRAW_ORDER``,
    the head with its hair (unless a hood hides it) between the neck and head pieces."""
    # the hose sit a little inside the garments' outline: a shirt whose sides were flush with the
    # legs' drew two faces in one plane round the waist, which z-fights into streaks at a distance
    _box(material, textures, "cloth", figure.legs, Vector3D(-BODY_HALF_X + LAYER, 0.0, -BODY_HALF_Z + LAYER),
         Vector3D(BODY_HALF_X - LAYER, WAIST, BODY_HALF_Z - LAYER))
    if EquipSlot.FEET in equipment:
        draw_gear(material, textures, equipment[EquipSlot.FEET])
    if EquipSlot.BODY in equipment:
        draw_gear(material, textures, equipment[EquipSlot.BODY])
    else:
        _box(material, textures, "cloth", figure.undershirt, Vector3D(-BODY_HALF_X, WAIST - 0.04, -BODY_HALF_Z),
             Vector3D(BODY_HALF_X, BODY_HEIGHT, BODY_HALF_Z))                                   # a plain shirt
        _belt(material, textures)
    if EquipSlot.NECK in equipment:
        draw_gear(material, textures, equipment[EquipSlot.NECK])
    hat = equipment.get(EquipSlot.HEAD)
    _box(material, textures, "skin", WHITE, Vector3D(-HEAD_HALF, HEAD_BASE, -HEAD_HALF), Vector3D(HEAD_HALF, HAT_TOP, HEAD_HALF))
    if hat is None or hat.gear not in HIDES_HAIR:
        _box(material, textures, "cloth", figure.hair, Vector3D(-HEAD_HALF - 0.01, HAT_TOP - 0.12, -HEAD_HALF + 0.06),
             Vector3D(HEAD_HALF + 0.01, HAT_TOP + 0.01, HEAD_HALF + 0.01))                        # hair: top and back
    if hat is not None:
        draw_gear(material, textures, hat)
    if EquipSlot.WEAPON in equipment:
        draw_gear(material, textures, equipment[EquipSlot.WEAPON])
    material(color=WHITE, textured=True)
