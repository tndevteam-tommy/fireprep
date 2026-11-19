"""
Firestarter - Processing a felled tree: limbing and bucking (DESIGN 4.1)

A felled tree lies on the ground as one piece.  The player works it up with the axe, exactly
like a woodcutter:

1. **Limbing** - every branch is cut off where it meets the trunk.  A limb breaks down by
   diameter: the thick part (> 5 cm) into *fire logs* (unsplit branch wood ~40 cm long),
   the thin part and its side shoots into *kindling* sticks, and the twigs / needles / leaves
   (or birch bark) into *tinder* bundles.
2. **Bucking** - the bole is cross-cut into 45 cm rounds.  A round comes free once both of its
   ends are cut; thick rounds are *logs* (split them with the axe), rounds from the thin top
   of the stem (< 9 cm) are already fire-log sized, and the whippy tip is kindling.

Everything is sized from the tree itself:

* Stem volume with a taper   r(s) = r0 (1 - s / H)^0.9   (butt radius r0, height H).
* Branch wood is a species-typical fraction of the stem volume (conifers ~15 %, broadleaves
  ~30 %), distributed over the limbs in proportion to their size; foliage / fine twigs add
  roughly a quarter of the branch mass again.
* Swings per cut follow the axe's chip depth per swing and the wood's hardness (the same
  model as the felling notch), so bucking a 60 cm oak with a stone axe is hard labour.
* Freshly cut wood carries the tree's living moisture (wetter still in spring sap flow).
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np

from firestarter.thermodynamics.wood_properties import WoodSpeciesType, get_wood_properties

SECTION_LEN = 0.45          # m: a bucked round (firewood length)
FIRELOG_MAX_R = 0.045       # rounds thinner than ~9 cm are fire logs already
KINDLING_MAX_R = 0.018      # stem / branch wood thinner than ~3.5 cm becomes kindling
FIRELOG_LEN = 0.40
KINDLING_STICK_M3 = math.pi * 0.008 ** 2 * 0.30     # one pencil-to-thumb stick
TINDER_BUNDLE_KG = 0.05

# species: (crown base as a fraction of height, branch wood / stem volume, limbs per metre of crown,
#           tinder material, tinder label)
SPECIES_CROWN = {
    WoodSpeciesType.WHITE_PINE: (0.38, 0.15, 1.4, "pine_needles", "Green pine needles & twigs"),
    WoodSpeciesType.RED_CEDAR: (0.12, 0.14, 1.6, "cedar_shavings", "Cedar bark & foliage"),
    WoodSpeciesType.PAPER_BIRCH: (0.40, 0.28, 2.0, "birch_bark", "Birch bark & twigs"),
    WoodSpeciesType.WHITE_OAK: (0.40, 0.32, 2.0, "dry_leaves", "Oak leaves & twigs"),
}


class Limb:
    _ids = 0

    def __init__(self, s: float, angle: float, length: float, radius: float):
        Limb._ids += 1
        self.id = Limb._ids
        self.s = s                  # where it leaves the trunk (m from the butt)
        self.angle = angle          # direction around the trunk
        self.length = length
        self.radius = radius        # radius where it joins the trunk
        self.cut_depth = 0.0
        self.removed = False

    @property
    def volume(self) -> float:
        return math.pi * self.radius ** 2 * self.length / 3.0


class FallenTree:
    """A felled stem lying along the fall line, waiting to be limbed and bucked."""

    def __init__(self, tree, rng: Optional[np.random.Generator] = None, sap_factor: float = 1.0):
        rng = rng or np.random.default_rng(tree.id)
        self.tree_id = tree.id
        self.species = tree.species
        self.bark_species = tree.species
        self.height = float(tree.height_m)
        self.r0 = float(tree.trunk_radius_m)
        self.heading = float(tree.fall_direction)
        self.dir = np.array([math.cos(self.heading), 0.0, math.sin(self.heading)])
        self.side = np.array([-self.dir[2], 0.0, self.dir[0]])
        # the butt kicks back off the stump a little as the hinge breaks
        self.base = np.array(tree.position, dtype=float) + self.dir * (self.r0 + 0.25)
        self.rest: Optional[tuple] = None      # (y at the butt, dy/ds) once settled on the ground
        self.m_surface = float(tree.living_moisture_surface) * sap_factor
        self.m_core = float(tree.living_moisture_core) * sap_factor
        self.decay_days = 0.0
        self.is_punk = False
        crown_frac, branch_frac, limbs_per_m, self.tinder_material, self.tinder_label = SPECIES_CROWN.get(
            tree.species, SPECIES_CROWN[WoodSpeciesType.WHITE_PINE])
        # the bole: where the stem is still thicker than kindling
        s_kindling = self.height * (1.0 - (KINDLING_MAX_R / max(self.r0, 1e-3)) ** (1 / 0.9))
        self.n_sections = max(1, int(s_kindling // SECTION_LEN))
        self.bole_length = self.n_sections * SECTION_LEN
        self.cut = np.zeros(self.n_sections + 1, dtype=bool)
        self.cut[0] = True                     # the felling cut at the stump
        self.cut_depth = np.zeros(self.n_sections + 1)
        self.removed = np.zeros(self.n_sections, dtype=bool)
        self.top_taken = False                 # the whippy leader beyond the last round
        # limbs over the crown, sized so their total volume matches the branch fraction
        crown_base = self.height * crown_frac
        n_limbs = int(max(4, min(22, round((self.height - crown_base) * limbs_per_m))))
        raw = []
        for i in range(n_limbs):
            t = (i + rng.uniform(0.1, 0.9)) / n_limbs
            s = crown_base + (self.height * 0.97 - crown_base) * t
            conifer = tree.species in (WoodSpeciesType.WHITE_PINE, WoodSpeciesType.RED_CEDAR)
            # conifers: long lower branches, short near the top; broadleaves: big central limbs
            shape = (1.0 - t) ** 0.8 if conifer else math.sin(math.pi * (0.2 + 0.8 * t)) ** 0.7
            L = max(0.8, self.height * (0.28 if conifer else 0.32) * shape * rng.uniform(0.8, 1.2))
            raw.append([s, rng.uniform(0, 2 * math.pi), L, L])
        stem_v = self.stem_volume()
        target = branch_frac * stem_v
        unit = sum(math.pi * (w * 0.02) ** 2 * L / 3.0 for _, _, L, w in raw)
        k = math.sqrt(target / max(unit, 1e-9)) * 0.02
        # a limb can't be much thicker than the stem where it forks off: where that caps the
        # big limbs, the others thicken until the branch-wood total is met
        caps = [self.radius_at(s) * 0.85 for s, _, _, _ in raw]
        for _ in range(12):
            v = sum(math.pi * min(c, k * w) ** 2 * L / 3.0 for c, (_, _, L, w) in zip(caps, raw))
            if v >= target * 0.99:
                break
            k *= min(1.5, math.sqrt(target / max(v, 1e-12)))
        self.limbs: List[Limb] = []
        for c, (s, a, L, w) in zip(caps, raw):
            self.limbs.append(Limb(s, a, L, min(c, k * w)))

    # ------------------------------------------------------------------ seasoning hooks
    # (duck-typed for forestry.seasoning: a big stem on the soil dries slowly and can rot)
    is_on_ground = True
    decay_rate = 0.15          # a whole trunk takes years, not weeks, to turn to punk

    @property
    def radius(self) -> float:
        return self.r0 * 0.6

    @property
    def initial_radius(self) -> float:
        return self.r0 * 0.6

    def set_moisture(self, surface: float = None, core: float = None):
        if surface is not None:
            self.m_surface = surface
        if core is not None:
            self.m_core = core

    def convert_to_punk(self):
        self.is_punk = True
        self.species = WoodSpeciesType.PUNK_WOOD

    # ------------------------------------------------------------------ geometry
    def radius_at(self, s: float) -> float:
        return self.r0 * max(0.0, 1.0 - s / self.height) ** 0.9

    def stem_volume(self) -> float:
        ss = np.linspace(0.0, self.height, 200)
        r = self.r0 * np.maximum(0.0, 1.0 - ss / self.height) ** 0.9
        return float(np.trapezoid(math.pi * r * r, ss)) if hasattr(np, "trapezoid") else float(np.trapz(math.pi * r * r, ss))

    def settle(self, ground_fn) -> None:
        """A stem is rigid: it comes to rest as a straight line propped on the highest ground
        under it (sunk a little into the litter), not draped over every dip."""
        ss = np.linspace(0.0, self.height, max(8, int(self.height / 0.25)))
        need = np.array([float(ground_fn(*(self.base + self.dir * s)[[0, 2]])) + max(self.radius_at(s), 0.01) * 0.9
                         for s in ss])
        b = float(np.clip(np.polyfit(ss, need, 1)[0], -0.6, 0.6))
        a = float(np.max(need - b * ss))
        self.rest = (a, b)

    @property
    def axis(self) -> np.ndarray:
        b = self.rest[1] if self.rest else 0.0
        v = self.dir + np.array([0.0, b, 0.0])
        return v / np.linalg.norm(v)

    def point_at(self, s: float, ground_fn=None) -> np.ndarray:
        """Centre of the stem s metres (horizontally) from the butt."""
        p = self.base + self.dir * s
        if self.rest is not None:
            p[1] = self.rest[0] + self.rest[1] * s
            return p
        y = float(ground_fn(p[0], p[2])) if ground_fn is not None else self.base[1]
        p[1] = y + max(self.radius_at(s), 0.01)
        return p

    def limb_direction(self, limb: Limb) -> np.ndarray:
        """Branches splay sideways / upward from the lying stem (those pointing down are
        crushed flat against the ground)."""
        up = np.array([0.0, 1.0, 0.0])
        # the crown collapses as it hits: upper branches are snapped back and lean over
        d = math.cos(limb.angle) * self.side + max(-0.1, math.sin(limb.angle)) * 0.55 * up + self.dir * 0.7
        return d / np.linalg.norm(d)

    # ------------------------------------------------------------------ state
    @property
    def limbs_left(self) -> int:
        return sum(1 for l in self.limbs if not l.removed)

    @property
    def rounds_left(self) -> int:
        return int((~self.removed).sum())

    @property
    def finished(self) -> bool:
        return self.limbs_left == 0 and self.rounds_left == 0 and self.top_taken

    def moisture(self) -> float:
        return 0.3 * self.m_surface + 0.7 * self.m_core

    def blocking_limb(self, s: float) -> Optional[Limb]:
        for l in self.limbs:
            if not l.removed and abs(l.s - s) < 0.35:
                return l
        return None

    def nearest_boundary(self, s: float) -> Optional[int]:
        """Index of the uncut section boundary nearest to s (never the felling cut)."""
        best, bd = None, 1e9
        for i in range(1, self.n_sections + 1):
            if self.cut[i]:
                continue
            left_free = i - 1 >= 0 and not self.removed[i - 1]
            right_free = i < self.n_sections and not self.removed[i]
            if not (left_free or right_free or i == self.n_sections):
                continue
            d = abs(i * SECTION_LEN - s)
            if d < bd:
                best, bd = i, d
        return best

    # ------------------------------------------------------------------ yields
    def limb_yield(self, limb: Limb) -> Dict[str, Any]:
        """Break a cut-off limb into fire logs, kindling sticks and tinder bundles."""
        rho = get_wood_properties(self.species).bulk_density
        r, L = limb.radius, limb.length
        fire_logs: List[Dict[str, float]] = []
        if r > FIRELOG_MAX_R * 0.55:
            # the limb tapers linearly: thick butt end cut into 40 cm fire logs
            L_thick = L * (1.0 - (FIRELOG_MAX_R * 0.55) / r)
            n = int(L_thick // FIRELOG_LEN)
            for k in range(n):
                s_mid = (k + 0.5) * FIRELOG_LEN
                fire_logs.append({"radius": r * (1.0 - s_mid / L), "length": FIRELOG_LEN, "s": s_mid})
        v_logs = sum(math.pi * f["radius"] ** 2 * f["length"] for f in fire_logs)
        v_rest = max(0.0, limb.volume - v_logs)
        # thin wood plus the side shoots of the branch (about as much again)
        kindling = int(round(v_rest * 1.6 / KINDLING_STICK_M3))
        # needles / leaves / fine twigs (or birch bark): ~8 % of the branch mass, in big handfuls
        foliage_kg = 0.08 * rho * limb.volume
        tinder = int(max(1, round(foliage_kg / TINDER_BUNDLE_KG)))
        return {"fire_logs": fire_logs, "kindling": kindling, "tinder": tinder}

    def round_yield(self, i: int) -> Dict[str, Any]:
        s0, s1 = i * SECTION_LEN, (i + 1) * SECTION_LEN
        r = self.radius_at(0.5 * (s0 + s1))
        kind = "log" if r >= FIRELOG_MAX_R else ("firewood" if r >= KINDLING_MAX_R else "kindling")
        return {"kind": kind, "radius": r, "length": SECTION_LEN, "s": 0.5 * (s0 + s1)}

    def totals(self) -> Dict[str, int]:
        """Everything this tree will give when fully processed."""
        out = {"logs": 0, "fire_logs": 0, "kindling": 0, "tinder": 0}
        for i in range(self.n_sections):
            k = self.round_yield(i)["kind"]
            out["logs" if k == "log" else ("fire_logs" if k == "firewood" else "kindling")] += 1
        for l in self.limbs:
            y = self.limb_yield(l)
            out["fire_logs"] += len(y["fire_logs"])
            out["kindling"] += y["kindling"]
            out["tinder"] += y["tinder"]
        out["kindling"] += self.top_kindling()
        return out

    def top_kindling(self) -> int:
        """The whippy leader beyond the last round."""
        v = 0.0
        ss = np.linspace(self.bole_length, self.height, 30)
        r = self.r0 * np.maximum(0.0, 1.0 - ss / self.height) ** 0.9
        v = float(np.sum(math.pi * r[:-1] ** 2 * np.diff(ss)))
        return int(max(1, round(v / KINDLING_STICK_M3)))

    @staticmethod
    def swing_depth(axe_type: str, species: WoodSpeciesType, punk: bool = False) -> float:
        """Depth of the kerf opened by one swing when cross-cutting (m)."""
        base = 0.028 if axe_type == "steel" else 0.011
        hard = {WoodSpeciesType.WHITE_OAK: 0.7, WoodSpeciesType.PAPER_BIRCH: 0.9,
                WoodSpeciesType.WHITE_PINE: 1.15, WoodSpeciesType.RED_CEDAR: 1.2}.get(species, 1.0)
        return base * hard * 1.25 * (2.5 if punk else 1.0)
