"""
Firestarter - Airflow, Convective Chimney & Draft Vacuum
Implements Section 2.4 of the Technical Design Specification:

    U_draft = C_d * sqrt(2 * g * H * (T_int - T_amb) / T_int)
    P_base  = P_amb - 0.5 * rho_air * U_draft^2

Fire-structure geometry is analysed from the actual fuel pieces:

* H      - vertical extent of the fuel structure (chimney height)
* D      - footprint diameter
* choke  - fraction of horizontal fuel length laid edge-to-edge with < 2 cm gaps
           (DESIGN: "Choked Layout ... C_d -> 0, O2 < 10 %")
* C_d    - discharge coefficient from the recognised architecture (teepee, log cabin, ...)
           or, for freeform lays, from the bed porosity - reduced by choking.

The oxygen balance compares O2 demand (Huggett's constant: 13.1 MJ released per kg O2
consumed, plus char oxidation stoichiometry) with the air drawn in by the stack-effect draft.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Sequence

import numpy as np

from firestarter.config import (
    GRAVITY, AIR_DENSITY_STP, STANDARD_PRESSURE,
    O2_AMBIENT_FRACTION, O2_FLAMING_MIN, O2_CHOKED_MAX,
    CHOKE_GAP_METERS, BREATH_O2_FRACTION,
    BREATH_OPTIMAL_VELOCITY, BREATH_MAX_VELOCITY, BREATH_MIN_VELOCITY
)

HUGGETT_J_PER_KG_O2 = 13.1e6
STRUCTURE_CD = {
    "teepee": 0.85,
    "log_cabin": 0.75,
    "lean_to": 0.65,
    "swedish_torch": 0.70,
    "star": 0.60,
}


@dataclass
class StructureAnalysis:
    height: float = 0.05
    diameter: float = 0.1
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    choke_fraction: float = 0.0
    min_gap: float = 1.0
    packing_ratio: float = 0.0
    c_d: float = 0.6
    intake_area: float = 0.01
    n_pieces: int = 0
    structure_type: str = "freeform"


def _closest_line_distance(p1, d1, p2, d2) -> float:
    w = p1 - p2
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n)
    if nn < 1e-6:   # parallel
        return float(np.linalg.norm(np.cross(w, d1)))
    return float(abs(np.dot(w, n)) / nn)


class AirflowDraftSolver:
    @staticmethod
    def calculate_draft_velocity(
        c_d: float,
        height_meters: float,
        internal_temp_c: float,
        ambient_temp_c: float
    ) -> float:
        """
        Stack Effect:  U_draft = C_d * sqrt(2 g H (T_int - T_amb) / T_int)   (Kelvin).
        """
        t_int_k = internal_temp_c + 273.15
        t_amb_k = ambient_temp_c + 273.15
        if t_int_k <= t_amb_k or height_meters <= 0.0 or c_d <= 0.0:
            return 0.0
        val = 2.0 * GRAVITY * height_meters * ((t_int_k - t_amb_k) / t_int_k)
        return c_d * math.sqrt(max(0.0, val))

    @staticmethod
    def calculate_base_vacuum(
        ambient_pressure_pa: float,
        draft_velocity: float,
        air_density: float = AIR_DENSITY_STP
    ) -> Tuple[float, float]:
        """P_base = P_amb - 0.5 rho U^2.  Returns (P_base, vacuum_draw_pa)."""
        vacuum_draw = 0.5 * air_density * (draft_velocity ** 2)
        return max(0.0, ambient_pressure_pa - vacuum_draw), vacuum_draw

    # ------------------------------------------------------------------ geometry analysis
    @staticmethod
    def analyze_pieces(
        centers: Sequence[np.ndarray],
        axes: Sequence[np.ndarray],
        lengths: Sequence[float],
        radii: Sequence[float],
        kinds: Sequence[str],
        floor_y: float = 0.0,
        structure_type: str = "freeform",
    ) -> StructureAnalysis:
        res = StructureAnalysis(structure_type=structure_type)
        n = len(centers)
        res.n_pieces = n
        if n == 0:
            return res
        tops, bottoms, xs, zs = [], [], [], []
        vol = 0.0
        for c, a, L, r in zip(centers, axes, lengths, radii):
            e1 = c - a * L * 0.5
            e2 = c + a * L * 0.5
            tops.append(max(e1[1], e2[1]) + r)
            bottoms.append(min(e1[1], e2[1]) - r)
            xs += [e1[0], e2[0]]
            zs += [e1[2], e2[2]]
            vol += math.pi * r * r * L
        res.height = float(max(0.04, max(tops) - max(floor_y, min(bottoms))))
        cx, cz = float(np.mean(xs)), float(np.mean(zs))
        rad = max(math.hypot(x - cx, z - cz) for x, z in zip(xs, zs))
        res.diameter = float(max(0.08, 2.0 * rad))
        res.centroid = np.array([cx, floor_y, cz])
        bulk = math.pi * (res.diameter * 0.5) ** 2 * res.height
        res.packing_ratio = float(min(1.0, vol / max(1e-6, bulk)))

        # Choke detection between near-parallel, near-horizontal solid pieces
        horiz_len = 0.0
        choked_len = 0.0
        min_gap = 1.0
        solid = [i for i in range(n) if kinds[i] in ("log", "firewood", "punk")]
        for i in solid:
            if abs(axes[i][1]) < 0.5:
                horiz_len += lengths[i]
        for ii, i in enumerate(solid):
            for j in solid[ii + 1:]:
                ai, aj = axes[i], axes[j]
                if abs(ai[1]) > 0.5 or abs(aj[1]) > 0.5:
                    continue
                if abs(float(np.dot(ai, aj))) < 0.9:
                    continue
                # must be side by side at a similar height
                if abs(centers[i][1] - centers[j][1]) > (radii[i] + radii[j]) * 0.8:
                    continue
                d = _closest_line_distance(centers[i], ai, centers[j], aj)
                gap = d - radii[i] - radii[j]
                # overlap of their projections along the shared axis
                t = float(np.dot(centers[j] - centers[i], ai))
                overlap = max(0.0, min(lengths[i] * 0.5, t + lengths[j] * 0.5) - max(-lengths[i] * 0.5, t - lengths[j] * 0.5))
                if overlap < 0.25 * min(lengths[i], lengths[j]):
                    continue
                min_gap = min(min_gap, gap)
                if gap < CHOKE_GAP_METERS:
                    choked_len += overlap * 2.0 * (1.0 - max(0.0, gap) / CHOKE_GAP_METERS * 0.5)
        res.min_gap = float(min_gap)
        res.choke_fraction = float(min(1.0, choked_len / max(1e-6, horiz_len))) if horiz_len > 0 else 0.0

        if structure_type in STRUCTURE_CD:
            cd = STRUCTURE_CD[structure_type]
        else:
            # freeform: porous beds breathe, solid packs do not
            openness = 1.0 - min(1.0, res.packing_ratio / 0.35)
            cd = 0.35 + 0.40 * openness
        cd *= (1.0 - 0.96 * res.choke_fraction ** 0.8)
        res.c_d = float(max(0.01, cd))
        # Intake: open perimeter near the base
        res.intake_area = float(max(0.002, math.pi * res.diameter * min(0.12, res.height) * 0.5 * (1.0 - 0.95 * res.choke_fraction)))
        return res

    @staticmethod
    def oxygen_balance(
        analysis: StructureAnalysis,
        draft_velocity: float,
        hrr_flame_w: float,
        char_burn_kg_s: float,
        wind_speed: float,
    ) -> float:
        """Target O2 volume fraction inside the fuel structure (steady state)."""
        qc_kw = max(0.0, 0.7 * hrr_flame_w / 1000.0)
        openness = (1.0 - analysis.choke_fraction) ** 2 * (1.0 - min(0.9, analysis.packing_ratio))
        # Air drawn through the base intake by the stack effect
        m_draft = AIR_DENSITY_STP * draft_velocity * analysis.intake_area
        # Open-perimeter entrainment into the flames (Heskestad plume entrainment)
        z = max(0.05, analysis.height)
        m_open = openness * (0.071 * qc_kw ** (1.0 / 3.0) * z ** (5.0 / 3.0) * 3.0 + 0.0018 * qc_kw + 0.002)
        m_wind = AIR_DENSITY_STP * 0.05 * wind_speed * analysis.intake_area
        m_air = m_draft + m_open + m_wind
        o2_supply = 0.232 * m_air
        o2_demand = hrr_flame_w / HUGGETT_J_PER_KG_O2 + 1.9 * char_burn_kg_s
        if o2_supply <= 1e-9:
            return 0.04
        consumed = min(0.95, o2_demand / o2_supply)
        return float(max(0.02, O2_AMBIENT_FRACTION * (1.0 - consumed)))

    # ------------------------------------------------------------------ legacy helper
    @staticmethod
    def evaluate_structure_airflow(
        log_positions: List[Tuple[float, float, float]],
        log_radii: List[float],
        internal_temp_c: float,
        ambient_temp_c: float,
        structure_type: str = "custom"
    ) -> Tuple[float, float, float, float]:
        """
        Simplified evaluation from centres & radii only (pieces assumed horizontal, parallel
        to X).  Returns (c_d, updraft_velocity, oxygen_fraction, base_vacuum_pa).
        """
        if not log_positions:
            return 0.6, 0.0, O2_AMBIENT_FRACTION, 0.0
        centers = [np.array(p, dtype=float) for p in log_positions]
        axes = [np.array([1.0, 0.0, 0.0]) for _ in centers]
        lengths = [0.45 for _ in centers]
        kinds = ["log" for _ in centers]
        floor = min(c[1] for c in centers) - max(log_radii)
        st = structure_type if structure_type in STRUCTURE_CD else "freeform"
        an = AirflowDraftSolver.analyze_pieces(centers, axes, lengths, list(log_radii), kinds, floor, st)
        if structure_type == "teepee":
            an.height = max(an.height, 0.45)
        u = AirflowDraftSolver.calculate_draft_velocity(an.c_d, an.height, internal_temp_c, ambient_temp_c)
        _, vac = AirflowDraftSolver.calculate_base_vacuum(STANDARD_PRESSURE, u)
        if an.choke_fraction > 0.5:
            o2 = 0.06 + 0.03 * (1.0 - an.choke_fraction)
        else:
            o2 = O2_AMBIENT_FRACTION
        return an.c_d, u, o2, vac

    @staticmethod
    def calculate_breath_effect(
        breath_velocity: float,
        ember_temp_c: float,
        ambient_temp_c: float = 20.0,
        reactivity: float = 1.0,
    ) -> Tuple[float, float, bool]:
        """
        Blowing dynamics ('F' key) on a small ember (~3 cm^2 of glowing char):
        Returns (net_thermal_power_watts, oxygen_fraction_of_breath, is_blown_away).
        * Weak   (< 0.8 m/s): cooling dominates (dQ/dt < 0)
        * Optimal (~1.5-4 m/s): oxidation outruns convective cooling
        * > 6 m/s: hot char dust is physically blown out of the nest
        """
        if breath_velocity <= 0.05:
            return 0.0, O2_AMBIENT_FRACTION, False
        if breath_velocity > BREATH_MAX_VELOCITY:
            return -250.0, BREATH_O2_FRACTION, True
        area = 3.0e-4
        D = 0.01
        h = float(np.cbrt((1.32 * (max(1.0, ember_temp_c - ambient_temp_c) / D) ** 0.25) ** 3
                          + (0.683 * (breath_velocity * D / 1.6e-5) ** 0.466 * 0.89 * 0.026 / D) ** 3))
        q_cool = h * area * (ember_temp_c - 34.0)
        h_m = h / (1.204 * 1005.0)
        o2 = BREATH_O2_FRACTION
        m_diff = h_m * 0.232 * 1.204 * (o2 / 0.2095) / 1.9
        Tk = ember_temp_c + 273.15
        m_kin = 6.1e6 * reactivity * math.exp(-130.0e3 / (8.314 * Tk))
        m_ox = 1.0 / (1.0 / max(m_kin, 1e-15) + 1.0 / max(m_diff, 1e-15))
        if breath_velocity < BREATH_MIN_VELOCITY:
            m_ox *= 0.5
        q_ox = m_ox * area * 30.0e6 * 0.55
        return q_ox - q_cool, o2, False
