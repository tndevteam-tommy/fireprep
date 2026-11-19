"""
Firestarter - Vectorised finite-volume fuel kernel.

Every fuel piece (tinder nest, kindling stick, split firewood, log) is discretised into
``S`` lengthwise *segments* x ``R`` radial *shells* (shell 0 = outermost).  All cells of all
fuel bodies in the fire live in one contiguous set of numpy arrays (``FuelStore``), so the
whole fire advances with a fixed number of numpy operations per thermal tick, regardless of
how many sticks the player stacks.  ``CombustionBody`` objects are thin views into a store.

Physics per cell (DESIGN Sections 2.1-2.3, plus standard fire-science correlations):

* Sensible heat with temperature dependent specific heats (Simpson & TenWolde wood cp,
  charcoal cp), liquid water / ice.  Latent heat of fusion 334 kJ/kg when frozen wood passes
  0 C, latent heat of vaporisation 2260 kJ/kg with a dehydration plateau at 100-120 C
  (bound water below fibre saturation needs extra heat of sorption).
* Radial cylindrical conduction between shells and axial conduction along the grain.
  Conductivity follows the Wood-Handbook moisture dependence and blends to char.
* Bound-water diffusion between shells (Arrhenius temperature dependence).
* Three-component Arrhenius pyrolysis (hemicellulose 200-260 C, cellulose 240-350 C,
  lignin 280-500 C) plus a fast terpene / resin "extractives" pool for softwoods.
* Flaming combustion of the surface via an effective heat-of-gasification model
  (m'' = q_net / L_g), with flame heat feedback that scales with the burning rate, critical
  mass-flux ignition / extinction limits (~1-2 g/m^2 s), steam-dilution quenching (core
  moisture boiling out through the surface), and oxygen-vitiation limits (O2 > 12 %).
* Piloted ignition time from the classic thin / thick formulas
  (t = rho c delta dT / q  and  t = pi/4 k rho c dT^2 / q^2) and flame spread
  (thin-fuel and Quintiere opposed / concurrent-flow spread) along and between segments.
* Char oxidation C + O2 with kinetic / boundary-layer diffusion control in series, ash
  build-up that insulates and blocks oxygen, and breath / draft enhanced mass transfer.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from firestarter.config import (
    STEFAN_BOLTZMANN, H_VAP_WATER, H_FUS_ICE, CP_WATER, O2_FLAMING_MIN,
)

R_GAS = 8.314
R_MAX = 5            # maximum radial shells per segment
N_COMP = 4           # extractives, hemicellulose, cellulose, lignin
COMP_NAMES = ("extractives", "hemicellulose", "cellulose", "lignin")

# First-order Arrhenius parameters (A [1/s], E [J/mol]) - calibrated so the characteristic
# decomposition windows match DESIGN 2.1 (hemicellulose 200-260, cellulose 240-350,
# lignin 280-500 C; terpenes volatilise from ~150 C).
COMP_A = np.array([8.9e6, 3.8e13, 1.7e16, 2.5e4])
COMP_E = np.array([75.0e3, 150.0e3, 200.0e3, 80.0e3])
# Effective lower heat of combustion of the released volatiles [J/kg]
COMP_HOC = np.array([36.0e6, 14.5e6, 16.0e6, 17.5e6])
# Nominal char yields of each component (scaled per species to match the DESIGN table)
BASE_CHAR_YIELD = np.array([0.02, 0.20, 0.08, 0.45])
DH_PYROLYSIS = 2.5e5           # J/kg converted (net endothermic)

# Char oxidation kinetics  (m''_kin = A exp(-E/RT)  kg/m^2 s)
CHAR_OX_A = 6.1e6
CHAR_OX_E = 130.0e3
O2_PER_KG_CHAR = 1.9           # kg O2 / kg char (CO/CO2 mixture at the surface)
CHAR_HEAT_RETAINED = 0.55      # fraction of char oxidation heat deposited in the solid
BOUDOUARD_DH = 14.3e6          # J per kg of carbon gasified by C + CO2 -> 2 CO (endothermic)

RHO_AIR = 1.204
CP_AIR = 1005.0
RHO_ASH = 280.0                # kg/m^3 fluffy wood ash
EMISSIVITY_WOOD = 0.90
EMISSIVITY_CHAR = 0.95

KIND_TINDER, KIND_KINDLING, KIND_FIREWOOD, KIND_LOG, KIND_FUNGUS = 0, 1, 2, 3, 4
KIND_NAMES = {"tinder": KIND_TINDER, "kindling": KIND_KINDLING, "firewood": KIND_FIREWOOD,
              "log": KIND_LOG, "fungus": KIND_FUNGUS, "punk": KIND_FIREWOOD}

# Per-kind model constants
#  max flame feedback flux (W/m^2), reference burning flux for feedback saturation (kg/m^2 s),
#  critical ignition & extinction mass flux (kg/m^2 s), area multiplier (fine fuel),
#  kernel (initial flame) area fraction, thin-fuel half-thickness (m; 0 => thermally thick)
KIND_PARAMS = {
    KIND_TINDER:   dict(qfb=32e3, mref=0.003, mign=0.8e-3, mext=0.6e-3, area=1.6, kernel=0.30, thin=0.00025, mext_moist=0.32),
    KIND_KINDLING: dict(qfb=36e3, mref=0.003, mign=1.6e-3, mext=1.2e-3, area=1.0, kernel=0.10, thin=0.0, mext_moist=0.45),
    KIND_FIREWOOD: dict(qfb=30e3, mref=0.004, mign=2.0e-3, mext=1.5e-3, area=1.0, kernel=0.06, thin=0.0, mext_moist=0.45),
    KIND_LOG:      dict(qfb=28e3, mref=0.004, mign=2.0e-3, mext=1.5e-3, area=1.0, kernel=0.04, thin=0.0, mext_moist=0.45),
    KIND_FUNGUS:   dict(qfb=10e3, mref=0.004, mign=9.0e-3, mext=9.0e-3, area=1.0, kernel=0.0, thin=0.0, mext_moist=0.2),
}

F_COND = 0.30                  # fraction of net flame-zone flux conducted into the solid


Q_FLAME_SAT = 55.0e3           # W/m^2: incident flux on fuel inside a small wood fire saturates


def _merged_flux(q_fb: np.ndarray, q_ext: np.ndarray, is_tinder: np.ndarray) -> np.ndarray:
    """Incident flux on a burning surface from its own flame plus the surrounding fire.
    Flames are optically thin-ish and merge, so the total saturates around 40-55 kW/m^2 in
    campfire-sized fires (cone-calorimeter peak HRRs of 150-250 kW/m^2 for softwoods)."""
    q = np.maximum(q_fb, q_ext) + 0.3 * np.minimum(q_fb, q_ext)
    sat = Q_FLAME_SAT * np.tanh(q / Q_FLAME_SAT)
    return np.where(is_tinder, q, sat)


def cp_wood(T: np.ndarray) -> np.ndarray:
    """Oven-dry wood specific heat, Simpson & TenWolde (J/kg K)."""
    return np.clip(103.1 + 3.867 * (T + 273.15), 900.0, 2400.0)


def cp_char(T: np.ndarray) -> np.ndarray:
    Tk = np.clip(T + 273.15, 250.0, 1600.0)
    return 420.0 + 2.09 * Tk - 6.85e-4 * Tk * Tk


def rho_vapor_sat(T: np.ndarray) -> np.ndarray:
    """Saturation water-vapour density (kg/m^3), Magnus formula."""
    Tc = np.clip(T, -40.0, 100.0)
    e = 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))
    return e / (461.5 * (Tc + 273.15))


def h_convective(diameter: np.ndarray, velocity: np.ndarray, dT: np.ndarray) -> np.ndarray:
    """Heat-transfer coefficient for a horizontal cylinder (W/m^2 K).
    Natural convection (Churchill-Chu, simplified laminar) blended with Hilpert forced
    convection through a cube-sum."""
    D = np.maximum(diameter, 0.002)
    h_nat = 1.32 * np.power(np.abs(dT) / D + 1e-9, 0.25)
    v = np.maximum(velocity, 0.0)
    re = v * D / 1.6e-5
    nu = np.where(re < 4000.0, 0.683 * np.power(re + 1e-9, 0.466), 0.193 * np.power(re + 1e-9, 0.618)) * 0.89
    h_f = nu * 0.026 / D
    return np.cbrt(h_nat ** 3 + h_f ** 3)


class FuelStore:
    """Contiguous storage for all fuel segments.  Arrays are indexed [segment, shell(, comp)]."""

    SEG_FLOAT = [
        # static geometry / material
        "r0", "L", "area_mult", "thin", "rho", "k0", "D0", "Tig", "Tauto", "Lg", "qfb", "mref",
        "mign", "mext", "mext_moist", "kernel", "hchar", "ash_ratio", "resin", "pop", "react",
        "seg_frac",  # position of segment center along the axis (-0.5..0.5)
        # dynamic
        "phi", "ign", "ext_timer", "ash_kg", "accel", "accel_rate", "mb", "tflame",
        # per-tick outputs
        "hrr_flame", "hrr_char", "vol_rate", "smoke_rate", "steam_rate", "char_rate",
        "T_surf", "steam_frac", "flux_in", "pop_event", "hiss_event", "evap_total", "r_eff",
    ]
    SEG_INT = ["body", "seg", "nseg", "nsh", "kind", "next_seg", "prev_seg"]
    CELL_FLOAT = ["T", "water", "ice", "char", "ash", "dry0", "rin", "rout", "present"]
    CELL_COMP = ["pool", "yield_"]

    def __init__(self, n: int = 0):
        self.n = n
        self.a: Dict[str, np.ndarray] = {}
        for f in self.SEG_FLOAT:
            self.a[f] = np.zeros(n, dtype=np.float64)
        for f in self.SEG_INT:
            self.a[f] = np.zeros(n, dtype=np.int64)
        for f in self.CELL_FLOAT:
            self.a[f] = np.zeros((n, R_MAX), dtype=np.float64)
        for f in self.CELL_COMP:
            self.a[f] = np.zeros((n, R_MAX, N_COMP), dtype=np.float64)
        self.a["accel_kind"] = np.zeros(n, dtype=np.int64)   # 0 none, 1 lighter fluid, 2 gasoline

    @staticmethod
    def concat(stores: List["FuelStore"]) -> "FuelStore":
        total = sum(s.n for s in stores)
        out = FuelStore(0)
        out.n = total
        if not stores:
            return FuelStore(0)
        for k in stores[0].a:
            out.a[k] = np.concatenate([s.a[k] for s in stores], axis=0) if total else stores[0].a[k][:0].copy()
        return out

    def extract(self, i0: int, i1: int) -> "FuelStore":
        out = FuelStore(0)
        out.n = i1 - i0
        for k, v in self.a.items():
            out.a[k] = v[i0:i1].copy()
        return out


# ----------------------------------------------------------------------------------------
# Building segment arrays for a new fuel piece
# ----------------------------------------------------------------------------------------

def default_discretisation(kind: int, radius: float) -> Tuple[int, int]:
    if kind == KIND_TINDER:
        return 3, 1
    if kind == KIND_FUNGUS:
        return 1, 2
    if kind == KIND_KINDLING:
        return 3, 3
    if radius < 0.045:
        return 3, 4
    return 3, 5


def build_fuel_store(props, kind: int, length: float, radius: float, dry_mass: float,
                     m_surface: float, m_core: float, temp_c: float,
                     n_seg: Optional[int] = None, n_shell: Optional[int] = None) -> FuelStore:
    S, R = default_discretisation(kind, radius)
    if n_seg:
        S = n_seg
    if n_shell:
        R = min(R_MAX, n_shell)
    st = FuelStore(S)
    a = st.a
    kp = KIND_PARAMS[kind]
    L_seg = length / S
    volume = np.pi * radius * radius * length
    rho_eff = dry_mass / max(1e-9, volume)

    a["r0"][:] = radius
    a["L"][:] = L_seg
    a["area_mult"][:] = kp["area"]
    a["thin"][:] = kp["thin"]
    a["rho"][:] = rho_eff
    a["k0"][:] = props.thermal_conductivity
    a["D0"][:] = props.moisture_diffusion_coeff
    a["Tig"][:] = props.volatile_ignition_temp
    a["Tauto"][:] = props.autoignition_temp
    a["Lg"][:] = props.heat_of_gasification
    # Thin cylinders get stronger flame heat feedback (thin boundary layer wraps the stick)
    a["qfb"][:] = kp["qfb"] * (1.0 + 0.35 * np.sqrt(0.01 / max(radius, 0.004))) if kind != KIND_TINDER else kp["qfb"]
    a["mref"][:] = kp["mref"]
    a["mign"][:] = kp["mign"]
    a["mext"][:] = kp["mext"]
    a["mext_moist"][:] = kp["mext_moist"]
    a["kernel"][:] = kp["kernel"]
    a["hchar"][:] = props.charcoal_caloric_value
    a["ash_ratio"][:] = props.ash_fraction / max(0.05, props.char_yield_fraction)
    a["resin"][:] = props.resin_content
    a["pop"][:] = props.spark_pop_rate
    a["react"][:] = props.char_reactivity
    a["seg_frac"][:] = (np.arange(S) + 0.5) / S - 0.5
    a["tflame"][:] = 0.0
    a["seg"][:] = np.arange(S)
    a["nseg"][:] = S
    a["nsh"][:] = R
    a["kind"][:] = kind
    a["next_seg"][:] = np.where(np.arange(S) < S - 1, np.arange(S) + 1, -1)
    a["prev_seg"][:] = np.where(np.arange(S) > 0, np.arange(S) - 1, -1)

    # Composition: extractives + (hemi, cell, lignin) split of the rest, minus mineral ash
    ext = props.resin_content
    ash = props.ash_fraction
    comp = np.array(props.composition, dtype=float)
    comp = comp / comp.sum() * max(0.0, 1.0 - ext - ash)
    frac = np.array([ext, comp[0], comp[1], comp[2]])
    y_nom = float(np.dot(frac, BASE_CHAR_YIELD)) / max(1e-9, frac.sum())
    scale = props.char_yield_fraction / max(1e-6, y_nom)
    yields = np.clip(BASE_CHAR_YIELD * scale, 0.0, 0.92)
    yields[0] = BASE_CHAR_YIELD[0]

    # Geometrically graded shells: thin outer shell (steep thermal gradient near the surface)
    widths = np.array([1.5 ** k for k in range(R)], dtype=float)
    widths = widths / widths.sum() * radius
    bounds = [radius]
    for w in widths:
        bounds.append(max(0.0, bounds[-1] - w))
    bounds[-1] = 0.0
    for k in range(R_MAX):
        if k < R:
            r_out = bounds[k]
            r_in = bounds[k + 1]
            vol_frac = (r_out * r_out - r_in * r_in) / (radius * radius)
            a["rout"][:, k] = r_out
            a["rin"][:, k] = r_in
            cell_dry = dry_mass / S * vol_frac
            a["dry0"][:, k] = cell_dry
            a["present"][:, k] = 1.0
            a["pool"][:, k, :] = cell_dry * frac[None, :]
            a["ash"][:, k] = cell_dry * ash
            a["yield_"][:, k, :] = yields[None, :]
            # Moisture profile (DESIGN 2.2): M_surface in the outer shell, M_core in the interior
            m = (m_surface if k == 0 else m_core) if R > 1 else 0.5 * (m_surface + m_core)
            a["water"][:, k] = m * cell_dry
            a["T"][:, k] = temp_c
            a["ice"][:, k] = 1.0 if temp_c < 0.0 else 0.0
        else:
            a["present"][:, k] = 0.0
    a["T_surf"][:] = temp_c
    a["r_eff"][:] = radius
    return st


# ----------------------------------------------------------------------------------------
# The kernel
# ----------------------------------------------------------------------------------------

class Environment:
    """Per-segment environmental inputs, prepared by the ThermodynamicGrid."""

    def __init__(self, n: int, ambient_c: float = 20.0):
        self.n = n
        self.T_amb = ambient_c
        self.T_sky = ambient_c - 10.0
        self.rh = 0.5
        self.q_rad_in = np.zeros(n)       # absorbed radiation from flames / hot solids (W/m^2)
        self.T_gas = np.full(n, ambient_c)
        self.gas_velocity = np.full(n, 0.3)
        self.F_block = np.zeros(n)        # fraction of view blocked by other fuel
        self.T_block = np.full(n, ambient_c)
        self.o2 = np.full(n, 0.2095)
        self.pilot = np.zeros(n, dtype=bool)
        self.q_pilot = np.zeros(n)        # local pilot flame flux for kernel ignition (W/m^2)
        self.pilot_area = np.zeros(n)     # m^2 heated by a hand-held pilot (lighter)
        self.rain_flux = np.zeros(n)      # kg/m^2 s of water deposited on the segment
        self.ground_G = np.zeros(n)       # W/K contact conductance to the ground / pit floor
        self.T_ground = np.full(n, ambient_c)
        self.breath_v = np.zeros(n)       # extra forced air velocity from blowing (m/s)
        self.ext_power = np.zeros(n)      # direct external heating (W), e.g. debug / accelerant


def _outer_index(present: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    alive = present.max(axis=1) > 0.5
    idx = np.argmax(present > 0.5, axis=1)
    return idx, alive


def step_fuel(st: FuelStore, env: Environment, dt: float, rng: np.random.Generator):
    """Advance every fuel cell by dt seconds (dt <= ~1 s is stable)."""
    n = st.n
    if n == 0:
        return
    a = st.a
    rows = np.arange(n)
    T = a["T"]
    present = a["present"] > 0.5
    pool = a["pool"]
    virgin = pool.sum(axis=2)
    char = a["char"]
    water = a["water"]
    dry0 = a["dry0"]
    solids = virgin + char + 1e-12

    o_idx, alive = _outer_index(a["present"])
    has_v = present & (virgin > 1e-9)
    f_idx = np.argmax(has_v, axis=1)
    alive_v = has_v.any(axis=1)

    kind = a["kind"]
    is_tinder = kind == KIND_TINDER
    is_fungus = kind == KIND_FUNGUS

    # ---- geometry ------------------------------------------------------------------
    rin = a["rin"]
    rout = a["rout"]
    cell_vol0 = np.pi * (rout ** 2 - rin ** 2) * a["L"][:, None]
    # solid volume fraction of each cell: virgin wood + char (char keeps ~70% of the volume)
    y_mean = np.maximum(0.05, (a["yield_"] * (pool + 1e-12)).sum(2) / (virgin + 1e-12))
    vfrac = np.clip(virgin / np.maximum(dry0, 1e-12) + 0.7 * char / np.maximum(dry0 * 0.3, 1e-12), 0.0, 1.0)
    vfrac = np.where(present, vfrac, 0.0)
    r_in_o = rin[rows, o_idx]
    r_out_o = rout[rows, o_idx]
    r_eff = np.sqrt(r_in_o ** 2 + vfrac[rows, o_idx] * (r_out_o ** 2 - r_in_o ** 2))
    r_eff = np.where(alive, np.maximum(r_eff, 0.0015), 0.0)
    L = a["L"]
    end_caps = np.where(a["next_seg"] < 0, 1.0, 0.0) + np.where(a["prev_seg"] < 0, 1.0, 0.0)
    A_geo = 2.0 * np.pi * r_eff * L + end_caps * np.pi * r_eff ** 2
    A_geo = np.maximum(A_geo, 1e-6)
    A_heat = A_geo * a["area_mult"]

    # ---- moisture state --------------------------------------------------------------
    M_cell = water / np.maximum(solids, 1e-9)
    M_o = M_cell[rows, o_idx]
    T_o = T[rows, o_idx]
    char_frac_o = char[rows, o_idx] / np.maximum(solids[rows, o_idx], 1e-12)

    # ---- surface exchange coefficients --------------------------------------------------
    vel = np.sqrt(env.gas_velocity ** 2 + env.breath_v ** 2)
    h_c = h_convective(2.0 * np.maximum(r_eff, 0.002), vel, T_o - env.T_gas)
    h_c = np.where(is_tinder, h_c * 1.5, h_c)
    ash_mm = a["ash_kg"] / (RHO_ASH * A_geo) * 1000.0
    ash_ins = 1.0 / (1.0 + ash_mm / 3.0)           # ash blanket insulates (conv + rad)
    eps = EMISSIVITY_WOOD + (EMISSIVITY_CHAR - EMISSIVITY_WOOD) * np.clip(char_frac_o, 0, 1)
    Tk_o = T_o + 273.15
    T_env_rad = (1.0 - env.F_block) * (env.T_sky + 273.15) ** 4 + env.F_block * (env.T_block + 273.15) ** 4
    q_rad_loss = eps * STEFAN_BOLTZMANN * (Tk_o ** 4 - T_env_rad) * ash_ins
    q_conv = h_c * (env.T_gas - T_o) * ash_ins

    # ---- accelerant film (lighter fluid / gasoline) -------------------------------------
    accel = a["accel"]
    accel_burn = np.zeros(n)
    accel_lit = (accel > 1e-7) & ((a["phi"] > 0.0) | env.pilot | (T_o > 250.0))
    # Burning rate of a thin fuel film ~ 0.03-0.06 kg/m^2 s (pool-fire regime)
    accel_rate = np.where(a["accel_kind"] == 2, 0.055, 0.035)
    accel_burn = np.where(accel_lit, np.minimum(accel, accel_rate * A_geo * dt), 0.0)
    a["accel"] = accel - accel_burn
    hrr_accel = accel_burn * 44.0e6 / dt
    # Unlit gasoline also evaporates (flash point -43 C)
    evap_acc = np.where((~accel_lit) & (a["accel_kind"] == 2), np.minimum(a["accel"], 2e-4 * A_geo * dt), 0.0)
    a["accel"] -= evap_acc

    # ---- flame-zone (heat of gasification) model -----------------------------------------
    phi = a["phi"]
    steam_frac = a["steam_frac"]
    # char layer thickness outside the pyrolysis front
    thick = rout - rin
    charred_cell = np.where(present, np.clip(1.0 - virgin / np.maximum(dry0, 1e-12), 0.0, 1.0), 0.0)
    shell_idx = np.arange(R_MAX)[None, :]
    outside_front = shell_idx <= f_idx[:, None]
    delta_char = (thick * charred_cell * outside_front * present).sum(axis=1)
    delta_char = np.where(a["thin"] > 0, 0.0, delta_char)
    M_front = M_cell[rows, f_idx]
    L_eff = a["Lg"] * (1.0 + delta_char / 0.0035) + np.clip(M_front, 0.0, 2.0) * (H_VAP_WATER + CP_WATER * 80.0)
    T_bs = np.where(a["thin"] > 0, 420.0, 400.0 + 260.0 * np.clip(delta_char / 0.006, 0.0, 1.0))
    q_rr = EMISSIVITY_CHAR * STEFAN_BOLTZMANN * ((T_bs + 273.15) ** 4 - T_env_rad)
    # in a porous fibre bed (tinder) burning fibres mostly radiate onto neighbouring fibres
    q_rr = np.where(is_tinder, 0.4 * q_rr, q_rr)
    q_ext_in = env.q_rad_in + np.maximum(0.0, h_c * (env.T_gas - T_bs)) + hrr_accel * 0.35 / A_geo
    pilot_flux_zone = np.where(env.pilot_area > 0, env.q_pilot * np.minimum(1.0, env.pilot_area / np.maximum(A_geo * np.maximum(phi, 1e-3), 1e-9)), 0.0)
    q_ext_in = q_ext_in + pilot_flux_zone
    mb = np.maximum(a["mb"], 0.0)
    for _ in range(3):
        q_fb = a["qfb"] * (1.0 - np.exp(-mb / a["mref"])) * (1.0 - np.clip(steam_frac, 0, 1)) * (0.55 + 0.45 * phi)
        # own flame and surrounding flames merge: incident flux saturates (~30-50 kW/m^2)
        q_net = _merged_flux(q_fb, q_ext_in, is_tinder) - q_rr
        mb = np.maximum(0.0, (1.0 - F_COND) * q_net) / L_eff
    mb = np.where(alive_v & (phi > 0), mb, 0.0)

    # ---- bulk Arrhenius pyrolysis (all cells) ---------------------------------------------
    Tk = T + 273.15
    k_rate = COMP_A[None, None, :] * np.exp(-COMP_E[None, None, :] / (R_GAS * Tk[:, :, None]))
    frac_rel = 1.0 - np.exp(-k_rate * dt)
    # the involved (flaming) part of the outer/front cell is handled by the flame-zone model
    scale_outer = np.ones((n, R_MAX))
    scale_outer[rows, f_idx] = 1.0 - phi
    released = pool * frac_rel * (present * scale_outer)[:, :, None]
    # Heat-transfer limit (cone-calorimeter behaviour): a surface can only gasify as fast as
    # heat reaches the pyrolysis front, m'' <~ q_in / L_g  (plus what is stored as sensible heat)
    bulk_tot = released.sum(axis=(1, 2))
    q_cap = _merged_flux(np.zeros_like(q_ext_in), np.maximum(q_ext_in, 0.0), is_tinder)
    cap = (1.15 * q_cap * (1.0 - phi) * A_heat / L_eff + 2.0e-3 * A_heat) * dt
    lim = np.where(bulk_tot > cap, cap / np.maximum(bulk_tot, 1e-15), 1.0)
    released = released * lim[:, None, None]
    # flame-zone consumption from the pyrolysis front cell
    fz_mass = np.minimum(mb * phi * A_heat * dt, virgin[rows, f_idx])
    fz_pool = pool[rows, f_idx, :]
    fz_split = fz_pool / np.maximum(fz_pool.sum(axis=1, keepdims=True), 1e-15)
    fz_rel = fz_split * fz_mass[:, None]
    released[rows, f_idx, :] += fz_rel
    released = np.minimum(released, pool)
    # water held in the wood consumed by the flame zone boils straight into the flame
    w_front = water[rows, f_idx]
    v_front = np.maximum(virgin[rows, f_idx], 1e-12)
    water_fz = np.minimum(w_front, w_front * np.clip(fz_mass / v_front, 0.0, 1.0))
    water[rows, f_idx] -= water_fz
    char_made = (released * a["yield_"]).sum(axis=2)
    vols = released * (1.0 - a["yield_"])            # (n, R, comp)
    vol_seg = vols.sum(axis=(1, 2))
    vol_energy = (vols * COMP_HOC[None, None, :]).sum(axis=(1, 2))
    pool -= released
    char += char_made
    Q_pyro = -released.sum(axis=2) * DH_PYROLYSIS      # endotherm (J) per cell

    # ---- flaming criteria -----------------------------------------------------------
    o2 = env.o2
    chi = 0.95 * np.sqrt(np.clip((o2 - 0.10) / (0.2095 - 0.10), 0.0, 1.0))
    vol_rate = vol_seg / dt
    m_flux = vol_rate / A_geo
    prev_flaming = phi > 0
    o2_ok = o2 >= O2_FLAMING_MIN
    steam_ok = steam_frac < 0.55
    fuel_ok = alive_v & (virgin[rows, f_idx] > 1e-6 * np.maximum(dry0[rows, f_idx], 1e-9))

    # ignition progress (kernel ignition under a pilot flame) - thin / thick formulas
    cp_o = cp_wood(T_o)
    c_eff = cp_o + np.clip(M_o, 0, 3) * (CP_WATER + H_VAP_WATER / np.maximum(50.0, a["Tig"] - T_o))
    dTig = np.maximum(10.0, a["Tig"] - T_o)
    q_avail = np.maximum(1.0, env.q_pilot - 11e3)
    k_eff = a["k0"] * (1.0 + 0.021 * 100.0 * np.clip(M_o, 0, 1)) * 2.5
    t_thin = a["rho"] * c_eff * (2.0 * np.maximum(a["thin"], 1e-6)) * dTig / q_avail
    t_thick = (np.pi / 4.0) * k_eff * a["rho"] * c_eff * dTig ** 2 / q_avail ** 2
    t_ig = np.where(a["thin"] > 0, t_thin, t_thick)
    piloted = env.pilot & (env.q_pilot > 11e3) & fuel_ok & o2_ok & (~prev_flaming)
    a["ign"] = np.where(piloted, a["ign"] + dt / np.maximum(t_ig, 0.05), np.maximum(0.0, a["ign"] - dt * 0.2))
    moist_ok_kernel = M_o < a["mext_moist"]
    kernel_ign = piloted & (a["ign"] >= 1.0) & moist_ok_kernel & ~is_fungus
    # too damp: the pilot flame only boils the water out of the touched patch
    steaming = env.pilot & (env.pilot_area > 0) & ~moist_ok_kernel
    local_dry = np.where(steaming, np.minimum(water[rows, o_idx],
                         env.q_pilot * env.pilot_area * 0.8 / (H_VAP_WATER + CP_WATER * 80.0) * dt), 0.0)
    water[rows, o_idx] -= local_dry
    # bulk ignition: surface hot enough with enough volatile flux
    # piloted ignition: the volatile mass flux leaving the surface reaches the lean limit
    # (critical mass flux); the species ignition temperature is only a soft gate
    bulk_pilot = env.pilot & (T_o >= 0.72 * a["Tig"])
    bulk_auto = T_o >= a["Tauto"]
    bulk_ign = (~prev_flaming) & (bulk_pilot | bulk_auto) & (m_flux >= a["mign"]) & o2_ok & steam_ok & fuel_ok & ~is_fungus
    # accelerant film lights the segment
    accel_ign = (~prev_flaming) & (accel_burn > 0) & fuel_ok & o2_ok

    new_phi = phi.copy()
    new_phi = np.where(kernel_ign, np.maximum(new_phi, a["kernel"]), new_phi)
    new_phi = np.where(bulk_ign, np.maximum(new_phi, 0.5), new_phi)
    new_phi = np.where(accel_ign, np.maximum(new_phi, 0.3), new_phi)
    a["ign"] = np.where(kernel_ign, 0.0, a["ign"])
    # kernel ignition of thin fuels: the first fibres flash -> seed a burning rate
    mb = np.where(kernel_ign & (mb <= 0), a["mref"] * 1.2, mb)

    # sustain / spread / retreat
    sustain = prev_flaming & ((mb + m_flux) >= a["mext"]) & o2_ok & steam_ok & fuel_ok
    sustain |= prev_flaming & (accel_burn > 0)
    # flame spread velocity
    c_eff_u = cp_wood(T_o) + np.clip(M_o, 0, 3) * (CP_WATER + H_VAP_WATER / np.maximum(50.0, a["Tig"] - T_o))
    v_thin = 0.9 * a["qfb"] * 0.01 / (a["rho"] * c_eff_u * 2.0 * np.maximum(a["thin"], 1e-6) * dTig)
    phi_spread = 9.0e3   # Quintiere flame-spread parameter (W^2/m^3) for wood
    v_thick = phi_spread / (k_eff * a["rho"] * c_eff_u * dTig ** 2)
    v_spread = np.where(a["thin"] > 0, v_thin, v_thick)
    # flames that are engulfed in neighbouring flames spread much faster
    v_spread = v_spread * (1.0 + 6.0 * env.pilot)
    grow = (1.0 - new_phi) * v_spread / np.maximum(L, 0.01) * dt
    grow = grow + np.where(T_o >= a["Tig"], (1.0 - new_phi) * dt / 0.6, 0.0)
    new_phi = np.where(sustain, np.minimum(1.0, new_phi + grow), new_phi)
    retreat = prev_flaming & ~sustain
    a["ext_timer"] = np.where(retreat, a["ext_timer"] + dt, 0.0)
    new_phi = np.where(retreat, new_phi * np.exp(-dt / 0.35), new_phi)
    new_phi = np.where(new_phi < 0.015, 0.0, new_phi)
    new_phi = np.where(alive_v, new_phi, 0.0)
    a["phi"] = new_phi
    a["mb"] = np.where(new_phi > 0, mb, 0.0)
    flaming = new_phi > 0

    # ---- heat of flaming: burned volatiles -------------------------------------------
    hrr_flame = np.where(flaming, vol_energy / dt * chi, 0.0) + hrr_accel
    # the flame zone's own heating (feedback) - conducted fraction into the front cell
    q_fb_final = a["qfb"] * (1.0 - np.exp(-mb / a["mref"])) * (1.0 - np.clip(steam_frac, 0, 1)) * (0.55 + 0.45 * new_phi)
    q_net_zone = _merged_flux(q_fb_final, q_ext_in, is_tinder) - q_rr
    Q_zone_front = np.where(flaming, F_COND * np.maximum(0.0, q_net_zone) * new_phi * A_heat * dt, 0.0)
    smoke = np.where(flaming, vol_seg * (1.0 - chi), vol_seg)

    # ---- char oxidation on the exposed outer surface -------------------------------------
    char_o = char[rows, o_idx]
    exposure = np.clip((char_o / np.maximum(solids[rows, o_idx], 1e-12)) / 0.5, 0.0, 1.0)
    h_m = h_c / (RHO_AIR * CP_AIR)
    rho_o2 = 0.232 * RHO_AIR * (o2 / 0.2095)
    # Arthur's correlation: CO/CO2 product ratio at a burning carbon surface
    co_ratio = 2500.0 * np.exp(-6240.0 / Tk_o)
    f_co2 = 1.0 / (1.0 + co_ratio)
    o2_per_kg = f_co2 * 2.667 + (1.0 - f_co2) * 1.333
    m_diff = h_m * rho_o2 / o2_per_kg * np.exp(-ash_mm / 3.0)
    m_kin = CHAR_OX_A * a["react"] * np.exp(-CHAR_OX_E / (R_GAS * Tk_o))
    m_ox = 1.0 / (1.0 / np.maximum(m_kin, 1e-15) + 1.0 / np.maximum(m_diff, 1e-15))
    m_ox = m_ox * (1.0 - 0.8 * new_phi) * (o2 > 0.04)
    char_burn = np.minimum(char_o, m_ox * A_geo * a["area_mult"] * exposure * dt)
    char_burn = np.where(alive, char_burn, 0.0)
    # Above ~1000 C the char surface is also gasified by the CO2 it just made (Boudouard
    # reaction C + CO2 -> 2 CO, +172 kJ/mol = 14.3 MJ/kg C, endothermic).  In the classic
    # double-film picture the O2 then burns that CO in a sheet above the surface.  The
    # endotherm is what keeps real glowing coals at ~900-1150 C however hard they are fanned.
    f_boud = 1.0 / (1.0 + np.exp(-(T_o - 1060.0) / 55.0))
    boud = np.minimum(np.maximum(char_o - char_burn, 0.0), char_burn * 1.5 * f_boud)
    boud = np.where(alive, boud, 0.0)
    char[rows, o_idx] -= char_burn + boud
    hrr_char = (char_burn + boud) * a["hchar"] / dt
    # Only C -> CO2 releases its full heat at the surface; C -> CO releases ~28% there and the
    # CO burns as faint blue flames above the coals (counted in the total heat release).
    surface_share = f_co2 + (1.0 - f_co2) * 0.28
    Q_char_surface = char_burn * a["hchar"] * surface_share * 0.85 - boud * BOUDOUARD_DH
    char_burn = char_burn + boud
    ash_rel = np.minimum(a["ash"][rows, o_idx], char_burn * a["ash_ratio"])
    a["ash"][rows, o_idx] -= ash_rel
    a["ash_kg"] += ash_rel

    # ---- surface energy into the outer cell -------------------------------------------------
    # The flaming fraction phi of the surface is handled by the flame-zone model above; the
    # bulk surface balance only sees the un-involved fraction (no double counting).
    Q_surface = (env.q_rad_in + q_conv - q_rad_loss) * A_heat * (1.0 - new_phi) * dt
    Q_surface += Q_char_surface + hrr_accel * 0.15 * dt
    Q_surface += env.ext_power * dt
    # hand-held pilot (lighter) heats the touched patch directly
    Q_surface += env.q_pilot * np.minimum(env.pilot_area, A_heat) * dt * (~flaming)
    Q_surface -= env.ground_G * (T_o - env.T_ground) * dt

    Q_cell = Q_pyro.copy()
    Q_cell[rows, o_idx] += Q_surface
    Q_cell[rows, f_idx] += Q_zone_front

    # ---- rain & surface drying ---------------------------------------------------------
    water_add = env.rain_flux * 2.0 * r_eff * L * dt * 0.6
    water[rows, o_idx] += np.where(alive, water_add, 0.0)
    # sub-boiling evaporation from the surface (mass transfer to ambient air)
    a_w = np.clip(M_o / 0.28, 0.0, 1.0) ** 0.7
    rho_v_s = rho_vapor_sat(np.minimum(T_o, 99.0)) * a_w
    rho_v_a = rho_vapor_sat(np.array(env.T_amb)) * env.rh
    evap_surf = np.clip(h_m * (rho_v_s - rho_v_a) * A_heat * dt, 0.0, None)
    evap_surf = np.minimum(evap_surf, water[rows, o_idx] * np.where(T_o < 0.0, 0.05, 1.0))
    water[rows, o_idx] -= evap_surf
    Q_cell[rows, o_idx] -= evap_surf * H_VAP_WATER

    # ---- conduction (radial + axial) -------------------------------------------------------
    k_cell = a["k0"][:, None] * (1.0 + 2.1 * np.clip(M_cell, 0.0, 1.5))
    char_vf = np.clip((0.7 * char / np.maximum(dry0 * 0.3, 1e-12)) / np.maximum(vfrac, 1e-6), 0.0, 1.0)
    k_char = 0.08 + 2.5e-10 * Tk ** 3
    k_cell = k_cell * (1.0 - char_vf) + k_char * char_vf
    rm = np.where(rin > 0, 0.5 * (rin + rout), 0.5 * rout)
    for k in range(R_MAX - 1):
        both = present[:, k] & present[:, k + 1]
        kk = 0.5 * (k_cell[:, k] + k_cell[:, k + 1])
        G = 2.0 * np.pi * L * kk / np.maximum(np.log(np.maximum(rm[:, k], 1e-5) / np.maximum(rm[:, k + 1], 1e-6)), 0.05)
        G = np.where(both & ~is_tinder, G, 0.0)
        Qk = G * (T[:, k] - T[:, k + 1]) * dt
        Q_cell[:, k] -= Qk
        Q_cell[:, k + 1] += Qk
        # bound-water diffusion (Arrhenius in temperature)
        Tm = 0.5 * (Tk[:, k] + Tk[:, k + 1])
        Dm = a["D0"] * np.exp(-38.0e3 / R_GAS * (1.0 / Tm - 1.0 / 293.15))
        dr = np.maximum(rm[:, k] - rm[:, k + 1], 1e-4)
        area_if = 2.0 * np.pi * rin[:, k] * L
        Mk = np.clip(M_cell[:, k], 0, 3)
        Mk1 = np.clip(M_cell[:, k + 1], 0, 3)
        flux_w = Dm * area_if * a["rho"] * (Mk1 - Mk) / dr * dt
        lim = 0.25 * np.minimum(water[:, k] + 1e-12, water[:, k + 1] + 1e-12)
        flux_w = np.clip(flux_w, -lim, lim)
        flux_w = np.where(both & ~is_tinder, flux_w, 0.0)
        water[:, k] += flux_w
        water[:, k + 1] -= flux_w
    # axial conduction between neighbouring segments of the same body (along the grain)
    nxt = a["next_seg"]
    has_next = nxt >= 0
    if np.any(has_next):
        idx = np.nonzero(has_next)[0]
        j = idx + (nxt[idx] - a["seg"][idx])   # global index of the next segment
        area_x = np.pi * (rout[idx] ** 2 - rin[idx] ** 2)
        kax = 2.2 * 0.5 * (k_cell[idx] + k_cell[j])
        Gax = kax * area_x / np.maximum(L[idx, None], 0.01)
        both = present[idx] & present[j]
        Gax = np.where(both, Gax, 0.0)
        Qax = Gax * (T[idx] - T[j]) * dt
        np.subtract.at(Q_cell, idx, Qax)
        np.add.at(Q_cell, j, Qax)

    # ---- temperature update with latent heats --------------------------------------------
    ash_c = a["ash"]
    cp_s = cp_wood(T) * virgin + cp_char(T) * char + 800.0 * ash_c
    ice_frac = np.clip(a["ice"], 0.0, 1.0)
    C = cp_s + water * (CP_WATER * (1.0 - ice_frac) + 2100.0 * ice_frac)
    C_floor = 0.05 * (cp_wood(np.zeros_like(T)) * dry0)
    C = np.maximum(C, np.maximum(C_floor, 1e-4))
    E = Q_cell
    T_new = T + E / C
    # melting ice at 0 C
    ice_mass = water * ice_frac
    crossing_up = (ice_mass > 0) & (T_new > 0.0)
    excess = np.where(crossing_up, C * T_new, 0.0)   # energy above 0 C available for melting
    melt = np.minimum(ice_mass, np.maximum(0.0, excess) / H_FUS_ICE)
    T_new = np.where(crossing_up, (excess - melt * H_FUS_ICE) / C, T_new)
    ice_mass -= melt
    # freezing of liquid water below 0 C
    liquid = water - ice_mass
    crossing_dn = (liquid > 1e-12) & (T_new < 0.0)
    deficit = np.where(crossing_dn, -C * T_new, 0.0)
    freeze = np.minimum(liquid, deficit / H_FUS_ICE)
    T_new = np.where(crossing_dn, -(deficit - freeze * H_FUS_ICE) / C, T_new)
    ice_mass += freeze
    # evaporation: dehydration plateau 100-120 C
    M_now = water / np.maximum(solids, 1e-9)
    T_evap = 100.0 + 20.0 * np.clip(1.0 - M_now / 0.30, 0.0, 1.0)
    boil = (T_new > T_evap) & (water - ice_mass > 1e-12)
    H_eff = H_VAP_WATER * (1.0 + 0.15 * np.clip(1.0 - M_now / 0.30, 0.0, 1.0))
    excess = np.where(boil, C * (T_new - T_evap), 0.0)
    evap = np.minimum(water - ice_mass, excess / H_eff)
    evap = np.where(boil, np.maximum(evap, 0.0), 0.0)
    T_new = np.where(boil, T_evap + (excess - evap * H_eff) / C, T_new)
    water -= evap
    ice_mass = np.minimum(ice_mass, water)
    a["ice"] = np.where(water > 1e-12, ice_mass / np.maximum(water, 1e-12), np.where(T_new < 0, 1.0, 0.0))
    # numerical safeguard: solid fuel surfaces in open wood fires do not exceed ~1250 C
    T_new = np.clip(T_new, -60.0, 1250.0)
    T_new = np.where(present, T_new, T)
    a["T"] = T_new

    # ---- merge nearly consumed outer cells into the next shell -----------------------------
    solids_new = pool.sum(axis=2) + char
    for k in range(R_MAX - 1):
        tiny = present[:, k] & present[:, k + 1] & (solids_new[:, k] < 0.12 * np.maximum(dry0[:, k], 1e-12)) & (k == o_idx)
        if np.any(tiny):
            ii = np.nonzero(tiny)[0]
            Ck = C[ii, k]
            Ck1 = C[ii, k + 1]
            a["T"][ii, k + 1] = (a["T"][ii, k] * Ck + a["T"][ii, k + 1] * Ck1) / (Ck + Ck1)
            pool[ii, k + 1, :] += pool[ii, k, :]
            char[ii, k + 1] += char[ii, k]
            water[ii, k + 1] += water[ii, k]
            a["ash_kg"][ii] += a["ash"][ii, k]
            pool[ii, k, :] = 0.0
            char[ii, k] = 0.0
            water[ii, k] = 0.0
            a["ash"][ii, k] = 0.0
            a["present"][ii, k] = 0.0
    # innermost / last cell fully consumed
    solids_new = pool.sum(axis=2) + char
    gone = (a["present"] > 0.5) & (solids_new < 0.004 * np.maximum(dry0, 1e-12)) & (np.arange(R_MAX)[None, :] == (a["nsh"][:, None] - 1))
    if np.any(gone):
        a["ash_kg"] += np.where(gone, a["ash"], 0.0).sum(axis=1)
        a["ash"][gone] = 0.0
        pool[gone] = 0.0
        char[gone] = 0.0
        water[gone] = 0.0
        a["present"][gone] = 0.0

    # ---- outputs ---------------------------------------------------------------------------
    steam = (evap.sum(axis=1) + evap_surf + water_fz + local_dry) / dt
    a["steam_rate"] = steam
    a["vol_rate"] = vol_rate
    # Steam that actually passes through the flame: water of the consumed wood plus core
    # moisture boiling out through the burning surface (DESIGN 2.2 "quenching surface flames")
    evap_outer = evap[rows, o_idx]
    evap_inner = evap.sum(axis=1) - evap_outer
    steam_flame = (water_fz + new_phi * evap_inner + 0.3 * new_phi * (evap_outer + evap_surf)) / dt
    a["steam_frac"] = np.where(vol_rate + steam_flame > 1e-9, steam_flame / (vol_rate + steam_flame + 1e-12), 0.0)
    a["smoke_rate"] = smoke / dt
    a["hrr_flame"] = hrr_flame
    a["hrr_char"] = hrr_char
    a["char_rate"] = char_burn / dt
    a["evap_total"] += evap.sum(axis=1) + evap_surf
    a["r_eff"] = np.where(alive, r_eff, 0.0)
    o_idx2, alive2 = _outer_index(a["present"])
    T_out = a["T"][rows, o_idx2]
    a["T_surf"] = np.where(flaming, np.maximum(T_out, T_bs), T_out)
    a["flux_in"] = q_ext_in
    # ASMR events: resin pops (terpene pockets flashing to steam/gas) and steam hiss
    pop_rate = a["pop"] * (0.12 * flaming * (0.3 + a["resin"] * 4.0) + 0.06 * (T_o > 200.0) * (M_o > 0.12))
    a["pop_event"] = (rng.random(n) < pop_rate * dt).astype(np.float64)
    a["hiss_event"] = (steam > 2.5e-5).astype(np.float64)
    # ash sheds off when the blanket gets thick (falls into the ember bed)
    shed = ash_mm > 5.0
    shed_kg = np.where(shed, a["ash_kg"] * 0.5, 0.0)
    a["ash_kg"] = a["ash_kg"] - shed_kg
    return shed_kg
