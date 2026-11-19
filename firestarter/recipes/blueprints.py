"""
Firestarter - Architectural Fire Blueprints
Implements DESIGN Section 6:
1. Teepee Fire   2. Log Cabin Fire   3. Lean-To Fire   4. Swedish Torch   5. Star Fire

Slots are expressed relative to the centre of the fire-pit floor (y = 0 on the floor).
Orientation = [tilt above horizontal (deg), yaw (deg, from +X toward +Z), roll].
Blueprints serve three purposes: the holographic placement guide, the reference-book
schematics, and structure recognition (which drives the draft discharge coefficient C_d).
"""

import math
from typing import List, Dict, Tuple, Any, Optional

import numpy as np

KIND_COMPAT = {
    "tinder": ("tinder",),
    "kindling": ("kindling", "firewood"),
    "firewood": ("firewood", "kindling", "log"),
    "log": ("log", "firewood"),
}


def _axis(tilt: float, yaw: float) -> np.ndarray:
    t, y = math.radians(tilt), math.radians(yaw)
    return np.array([math.cos(t) * math.cos(y), math.sin(t), math.cos(t) * math.sin(y)])


class BlueprintSlot:
    def __init__(self, slot_id: str, expected_type: str, position: np.ndarray, orientation: np.ndarray,
                 dimensions: np.ndarray, tolerance_m: float = 0.10, axis_cos: float = 0.82):
        self.slot_id = slot_id
        self.expected_type = expected_type
        self.position = np.array(position, dtype=float)
        self.orientation = np.array(orientation, dtype=float)
        self.dimensions = np.array(dimensions, dtype=float)
        self.tolerance_m = tolerance_m
        self.axis_cos = axis_cos          # minimum |cos| between piece axis and slot axis
        self.is_filled = False

    @property
    def axis(self) -> np.ndarray:
        return _axis(self.orientation[0], self.orientation[1])

    def matches(self, kind: str, rel_pos: np.ndarray, axis: np.ndarray) -> bool:
        if kind not in KIND_COMPAT.get(self.expected_type, (self.expected_type,)):
            return False
        if np.linalg.norm(self.position - rel_pos) > self.tolerance_m:
            return False
        if self.expected_type != "tinder" and abs(float(np.dot(self.axis, axis))) < self.axis_cos:
            return False
        return True


class FireBlueprint:
    def __init__(self, name: str, key: str, description: str, ideal_use: str, draft_profile: str,
                 slots: List[BlueprintSlot], builtin: bool = True):
        self.name = name
        self.key = key
        self.description = description
        self.ideal_use = ideal_use
        self.draft_profile = draft_profile
        self.slots = slots
        self.builtin = builtin

    def check_placement(self, log_positions: List[np.ndarray]) -> float:
        """Fraction of slots matched by positions only (legacy helper)."""
        if not self.slots:
            return 1.0
        used = [False] * len(log_positions)
        filled = 0
        for slot in self.slots:
            slot.is_filled = False
            for idx, p in enumerate(log_positions):
                if not used[idx] and np.linalg.norm(slot.position - p) <= slot.tolerance_m:
                    slot.is_filled = True
                    used[idx] = True
                    filled += 1
                    break
        return filled / len(self.slots)

    def match_pieces(self, pieces: List[Tuple[str, np.ndarray, np.ndarray]]) -> float:
        """pieces: (kind, relative position, axis).  Returns matched fraction, marks slots."""
        used = [False] * len(pieces)
        filled = 0
        for slot in self.slots:
            slot.is_filled = False
            best, bd = -1, 1e9
            for i, (kind, rp, ax) in enumerate(pieces):
                if used[i] or not slot.matches(kind, rp, ax):
                    continue
                d = float(np.linalg.norm(slot.position - rp))
                if d < bd:
                    best, bd = i, d
            if best >= 0:
                used[best] = True
                slot.is_filled = True
                filled += 1
        extras = sum(1 for u in used if not u)
        score = filled / max(1, len(self.slots))
        # a structure buried in random extra pieces is no longer that structure
        return score * (1.0 - min(0.5, 0.06 * max(0, extras - 2)))

    def next_open_slot(self, kind: str, near: Optional[np.ndarray] = None) -> Optional[BlueprintSlot]:
        cands = [s for s in self.slots if not s.is_filled and kind in KIND_COMPAT.get(s.expected_type, ())]
        if not cands:
            return None
        # slots made for exactly this kind of piece first, then build bottom-up (tinder first,
        # then the lowest open slots), then the one nearest the crosshair
        cands.sort(key=lambda s: (s.expected_type != kind, s.expected_type != "tinder", round(s.position[1], 2),
                                  0.0 if near is None else float(np.linalg.norm(s.position[[0, 2]] - near[[0, 2]]))))
        return cands[0]


def _lean(theta_deg: float, kind: str, base_r: float, tilt: float, L: float, r: float, sid: str) -> BlueprintSlot:
    th = math.radians(theta_deg)
    bottom = np.array([base_r * math.cos(th), r, base_r * math.sin(th)])
    yaw = theta_deg + 180.0
    ax = _axis(tilt, yaw)
    c = bottom + ax * L / 2.0
    return BlueprintSlot(sid, kind, c, np.array([tilt, yaw, 0.0]), np.array([L, r]))


def create_teepee_blueprint() -> FireBlueprint:
    slots = [BlueprintSlot("center_tinder", "tinder", np.array([0.0, 0.045, 0.0]), np.zeros(3), np.array([0.14, 0.045]))]
    for i, a in enumerate(range(0, 360, 72)):
        slots.append(_lean(a + 10, "kindling", 0.12, 62.0, 0.30, 0.008, f"cone_{i + 1}"))
    for i, a in enumerate((50, 170, 290)):
        slots.append(_lean(a, "firewood", 0.20, 56.0, 0.38, 0.03, f"fuel_{i + 1}"))
    return FireBlueprint(
        name="Teepee Fire", key="teepee",
        description="Conical structure concentrating vertical thermal updraft for quick heat and rapid kindle ignition.",
        ideal_use="Fast initial warmth, lighting damp kindling, emergency flare.",
        draft_profile="Concentrated vertical chimney updraft with strong base vacuum draw.",
        slots=slots)


def create_log_cabin_blueprint() -> FireBlueprint:
    """Square cabin of split wood around a small kindling teepee, 'roofed' with kindling laid
    across the top tier: the roof catches from the teepee's flames and radiates down onto the
    cabin walls (the classic bushcraft lay)."""
    r = 0.03
    slots = [
        BlueprintSlot("center_tinder", "tinder", np.array([0.0, 0.045, 0.0]), np.zeros(3), np.array([0.14, 0.045])),
        _lean(0, "kindling", 0.08, 60.0, 0.28, 0.008, "inner_kindling_1"),
        _lean(90, "kindling", 0.08, 60.0, 0.28, 0.008, "inner_kindling_2"),
        _lean(180, "kindling", 0.08, 60.0, 0.28, 0.008, "inner_kindling_3"),
        _lean(270, "kindling", 0.08, 60.0, 0.28, 0.008, "inner_kindling_4"),
    ]
    tiers = [(0, 0.15), (90, 0.15), (0, 0.12), (90, 0.12)]
    for t, (yaw, off) in enumerate(tiers):
        y = r + t * 2.0 * r
        for s_i, sgn in enumerate((-1.0, 1.0)):
            if yaw == 0:
                pos = np.array([0.0, y, sgn * off])
            else:
                pos = np.array([sgn * off, y, 0.0])
            slots.append(BlueprintSlot(f"tier{t + 1}_{s_i + 1}", "firewood", pos, np.array([0.0, yaw, 0.0]),
                                       np.array([0.40, r])))
    roof_y = r + 3 * 2.0 * r + r + 0.009
    for i, z in enumerate((-0.06, 0.0, 0.06)):
        slots.append(BlueprintSlot(f"roof_{i + 1}", "kindling", np.array([0.0, roof_y, z]), np.array([0.0, 0.0, 0.0]),
                                   np.array([0.30, 0.008]), tolerance_m=0.06, axis_cos=0.9))
    return FireBlueprint(
        name="Log Cabin Fire", key="log_cabin",
        description="Square cross-hatch chimney providing balanced multi-directional air intake and a sustained coal bed.",
        ideal_use="Cooking, long-duration warmth, enduring glowing coal foundation.",
        draft_profile="High-volume, multi-directional lateral intake with a stable internal heat trap.",
        slots=slots)


def create_lean_to_blueprint() -> FireBlueprint:
    slots = [
        BlueprintSlot("reflector", "log", np.array([0.0, 0.06, -0.20]), np.array([0.0, 0.0, 0.0]), np.array([0.46, 0.06])),
        BlueprintSlot("center_tinder", "tinder", np.array([0.0, 0.045, -0.06]), np.zeros(3), np.array([0.14, 0.045])),
    ]
    for i, x in enumerate((-0.10, 0.0, 0.10)):
        tilt = 28.0
        L = 0.30
        bottom = np.array([x, 0.008, 0.10])
        ax = _axis(tilt, -90.0)
        slots.append(BlueprintSlot(f"lean_{i + 1}", "kindling", bottom + ax * L / 2, np.array([tilt, -90.0, 0.0]), np.array([L, 0.008]),
                                   axis_cos=0.97))
    for i, x in enumerate((-0.07, 0.07)):
        tilt = 22.0
        L = 0.38
        bottom = np.array([x, 0.03, 0.16])
        ax = _axis(tilt, -90.0)
        slots.append(BlueprintSlot(f"fuel_{i + 1}", "firewood", bottom + ax * L / 2, np.array([tilt, -90.0, 0.0]), np.array([L, 0.03]),
                                   axis_cos=0.97))
    return FireBlueprint(
        name="Lean-To Fire", key="lean_to",
        description="Wind-deflecting layout: a thick windward back-log acts as reflector and shield for the kindling.",
        ideal_use="High wind, directional heat reflected toward a shelter.",
        draft_profile="Windward barrier deflects gusts; draft is drawn in from the lee side.",
        slots=slots, builtin=False)


def create_swedish_torch_blueprint() -> FireBlueprint:
    L = 0.40
    slots = []
    for i, (x, z) in enumerate(((0.034, 0.034), (-0.034, 0.034), (0.034, -0.034), (-0.034, -0.034))):
        # quarters stand upright (within ~15 degrees) and touch: a leaning teepee is not a torch
        slots.append(BlueprintSlot(f"quarter_{i + 1}", "firewood", np.array([x, L / 2.0, z]), np.array([90.0, 0.0, 0.0]),
                                   np.array([L, 0.03]), tolerance_m=0.05, axis_cos=0.965))
    slots.append(BlueprintSlot("bore_tinder", "tinder", np.array([0.0, L - 0.02, 0.0]), np.zeros(3), np.array([0.10, 0.03]), tolerance_m=0.12))
    return FireBlueprint(
        name="Swedish Torch (Canadian Candle)", key="swedish_torch",
        description="A single upright log split into quarters with shavings packed into the central bore. Self-feeding torch.",
        ideal_use="Cooking platform, snow or mud surfaces, efficient fuel use.",
        draft_profile="Narrow central bore: a concentrated vertical torch flame with very high chimney velocity.",
        slots=slots, builtin=False)


def create_star_fire_blueprint() -> FireBlueprint:
    slots = [BlueprintSlot("center_coals", "tinder", np.array([0.0, 0.045, 0.0]), np.zeros(3), np.array([0.14, 0.045]))]
    L = 0.45
    for i in range(5):
        a = i * 72.0 + 18.0
        th = math.radians(a)
        d = 0.07 + L / 2.0
        pos = np.array([d * math.cos(th), 0.06, d * math.sin(th)])
        slots.append(BlueprintSlot(f"spoke_{i + 1}", "log", pos, np.array([0.0, a, 0.0]), np.array([L, 0.06]), tolerance_m=0.12))
    return FireBlueprint(
        name="Star Fire", key="star",
        description="Logs radiate like wheel spokes with only their tips in the central coals; push them in as they burn.",
        ideal_use="Overnight burning, very low wood consumption, controlled smoulder.",
        draft_profile="Lateral spokes channel a steady low-velocity draft into the centre coal bed.",
        slots=slots, builtin=False)


ALL_BLUEPRINTS: Dict[str, FireBlueprint] = {
    "teepee": create_teepee_blueprint(),
    "log_cabin": create_log_cabin_blueprint(),
    "lean_to": create_lean_to_blueprint(),
    "swedish_torch": create_swedish_torch_blueprint(),
    "star": create_star_fire_blueprint(),
}


def match_structure(bodies, pit_center: np.ndarray, threshold: float = 0.65) -> Tuple[str, float, Dict[str, float]]:
    """Recognises the architecture of the fuel in the pit.  Returns (key, score, all_scores)."""
    pieces = []
    for b in bodies:
        rel = np.array(b.position, dtype=float) - pit_center
        pieces.append((b.geometry_type, rel, b.axis))
    scores = {k: bp.match_pieces(pieces) for k, bp in ALL_BLUEPRINTS.items()}
    best = max(scores, key=scores.get) if scores else "freeform"
    if scores.get(best, 0.0) >= threshold:
        return best, scores[best], scores
    return "freeform", scores.get(best, 0.0), scores
