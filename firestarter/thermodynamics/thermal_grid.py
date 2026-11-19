"""
Firestarter - ThermodynamicGrid & ThermalConvectionSolver
Implements DESIGN Section 2 and Section 8.1.

The grid owns every CombustionBody in the fire pit and couples them:

* Flame radiation   - point-source model  q = chi_r Q cos(theta) / (4 pi d^2),  chi_r = 0.3
* Fire plume        - McCaffrey centre-line gas temperature (continuous / intermittent /
                      plume regions) with a Gaussian radial profile, bent by the wind
* Flame height      - Heskestad  L_f = 0.235 Q^(2/5) - 1.02 D   (Q in kW)
* Solid radiation   - view-factor exchange between glowing pieces (cross-radiation is what
                      keeps a log fire alive), and blocking of the cold night sky
* Ember (coal) bed  - charcoal that falls out of burnt pieces collects on the pit floor;
                      it oxidises, radiates up into the fuel above and can rekindle a fire
* Ground contact    - cold / wet soil drains heat from pieces lying on it
* Draft & oxygen    - stack effect and O2 balance from ``AirflowDraftSolver``
* Pilots            - the player's lighter flame and breath (F key)
"""

import math
from typing import Callable, List, Dict, Tuple, Optional, Any

import numpy as np

from firestarter.config import THERMAL_DT, STEFAN_BOLTZMANN, STANDARD_PRESSURE
from firestarter.thermodynamics.combustion_body import CombustionBody, CombustionStage
from firestarter.thermodynamics.airflow_draft import AirflowDraftSolver, StructureAnalysis
from firestarter.thermodynamics import fuel_kernel as fk

CHI_RADIATIVE = 0.30
SOIL_K = 0.9            # W/m K, moist forest soil


def mccaffrey_excess_temperature(z: np.ndarray, q_kw: np.ndarray) -> np.ndarray:
    """Centre-line temperature rise (K) of a fire plume at height z (m) above a Q kW fire."""
    q = np.maximum(q_kw, 1e-4)
    zs = np.maximum(z, 1e-4) / q ** 0.4
    dT = np.where(zs < 0.08, 880.0,
                  np.where(zs < 0.2, 880.0 * (0.08 / zs), 352.0 * (zs / 0.2) ** (-5.0 / 3.0)))
    return dT


def heskestad_flame_height(q_kw: float, diameter: float) -> float:
    if q_kw <= 0.01:
        return 0.0
    return max(0.0, 0.235 * q_kw ** 0.4 - 1.02 * diameter)


class CoalBed:
    """Charcoal embers and ash accumulated on the fire-pit floor."""

    def __init__(self, ambient: float = 20.0):
        self.char_kg = 0.0
        self.ash_kg = 0.0
        self.T = ambient
        self.hrr = 0.0
        self.burn_rate = 0.0
        self.hchar = 25.0e6
        self.flare_j = 0.0          # volatiles of collapsed pieces still to flame off (J)
        self.flare_hrr = 0.0
        self.collapse_events = 0

    @property
    def area(self) -> float:
        return float(np.clip(self.char_kg / (200.0 * 0.03), 0.0, 0.45))

    @property
    def radius(self) -> float:
        return math.sqrt(max(self.area, 1e-6) / math.pi)

    @property
    def ash_depth_mm(self) -> float:
        a = max(self.area, 0.05)
        return self.ash_kg / (fk.RHO_ASH * a) * 1000.0

    def add(self, char_kg: float, temp_c: float, hchar: float):
        if char_kg <= 0:
            return
        tot = self.char_kg + char_kg
        self.T = (self.T * self.char_kg + temp_c * char_kg) / max(1e-9, tot)
        self.hchar = (self.hchar * self.char_kg + hchar * char_kg) / max(1e-9, tot)
        self.char_kg = tot

    def step(self, dt: float, ambient: float, T_sky: float, o2: float, air_v: float, breath_v: float,
             rain_flux: float, F_cover: float, T_cover: float, T_ground: float) -> float:
        """Returns heat delivered to the soil (J)."""
        self.hrr = 0.0
        self.burn_rate = 0.0
        # flare-up of volatiles released by collapsing pieces (~8 s time constant, never
        # lingering as a faint ghost flame: at least ~0.8 kW until the pocket is spent)
        if self.flare_j > 1.0:
            rel = min(self.flare_j, max(self.flare_j * min(1.0, dt / 8.0), 800.0 * dt), 12000.0 * dt) if o2 > 0.12 else 0.0
            self.flare_j -= rel
            self.flare_hrr = rel / dt
        else:
            self.flare_j = 0.0
            self.flare_hrr = 0.0
        if self.char_kg <= 1e-5:
            self.char_kg = 0.0
            self.T += (ambient - self.T) * min(1.0, dt / 60.0)
            return 0.0
        A = max(self.area, 0.004)
        Tk = self.T + 273.15
        v = math.sqrt(air_v ** 2 + breath_v ** 2)
        D = max(0.05, 2.0 * math.sqrt(A / math.pi))
        h = float(fk.h_convective(np.array([0.02]), np.array([v]), np.array([self.T - ambient]))[0])
        ins = 1.0 / (1.0 + self.ash_depth_mm / 4.0)
        h_m = h / (fk.RHO_AIR * fk.CP_AIR)
        # forced air penetrates the porous bed: it feeds oxygen to every coal but only
        # convectively cools the outer surface, and CO burns within the bed itself
        v_loss = math.sqrt(air_v ** 2 + (0.3 * breath_v) ** 2)
        h_loss = float(fk.h_convective(np.array([0.02]), np.array([v_loss]), np.array([self.T - ambient]))[0])
        co_in_bed = 0.28 + 0.45 * min(1.0, breath_v / 3.0)
        f_co2 = 1.0 / (1.0 + 2500.0 * math.exp(-6240.0 / Tk))
        o2_per = f_co2 * 2.667 + (1.0 - f_co2) * 1.333
        m_diff = h_m * 0.232 * fk.RHO_AIR * (o2 / 0.2095) / o2_per * math.exp(-self.ash_depth_mm / 6.0)
        m_kin = fk.CHAR_OX_A * math.exp(-fk.CHAR_OX_E / (fk.R_GAS * Tk))
        m_ox = 1.0 / (1.0 / max(m_kin, 1e-15) + 1.0 / max(m_diff, 1e-15))
        # porous bed: ~2.5x the projected area participates
        burn = min(self.char_kg, m_ox * A * 2.5 * dt)
        # Boudouard gasification (C + CO2 -> 2 CO, endothermic) caps coal temperatures ~1100 C
        f_boud = 1.0 / (1.0 + math.exp(-(self.T - 1060.0) / 55.0))
        boud = min(max(0.0, self.char_kg - burn), burn * 1.5 * f_boud)
        self.char_kg -= burn + boud
        self.ash_kg += (burn + boud) * 0.08
        self.burn_rate = (burn + boud) / dt
        q_ox = burn * self.hchar
        self.hrr = (burn + boud) * self.hchar / dt
        T_env4 = (1.0 - F_cover) * (T_sky + 273.15) ** 4 + F_cover * (T_cover + 273.15) ** 4
        q_rad = fk.EMISSIVITY_CHAR * STEFAN_BOLTZMANN * (Tk ** 4 - T_env4) * A * ins
        q_conv = h_loss * (self.T - ambient) * A * ins
        q_ground = 3.0 * A * (self.T - T_ground) * 4.0
        rain_w = rain_flux * A * dt
        q_rain = rain_w * (2.26e6 + 4184 * max(0.0, min(100.0, self.T) - ambient))
        C = max(0.05, self.char_kg * float(fk.cp_char(np.array([self.T]))[0]) + self.ash_kg * 800.0)
        dE = (q_ox * (f_co2 + (1.0 - f_co2) * co_in_bed) * 0.85) - boud * fk.BOUDOUARD_DH - (q_rad + q_conv + q_ground) * dt - q_rain
        self.T = max(ambient - 5.0, min(1250.0, self.T + dE / C))
        return q_ground * dt


class ThermalConvectionSolver:
    """
    Computes the per-segment environment (incident flame & solid radiation, plume gas
    temperature & velocity, sky blocking, pilots, ground contact) from the current fire state.
    """

    @staticmethod
    def build_environment(grid: "ThermodynamicGrid", env_state: Dict[str, Any]) -> fk.Environment:
        st = grid.store
        n = st.n
        amb = env_state["ambient"]
        env = fk.Environment(n, amb)
        env.T_sky = env_state.get("sky_temp", amb - 10.0)
        env.rh = env_state.get("rh", 0.5)
        if n == 0:
            return env
        a = st.a
        P = grid.seg_pos
        R = grid.seg_rad
        Lseg = a["L"]
        T_s = a["T_surf"]
        alive = (a["present"] > 0.5).any(axis=1)
        wind = np.array(env_state.get("wind_vec", np.zeros(3)), dtype=float)
        wind_local = wind * 0.45      # sheltered inside the stone ring / near the ground
        wind_speed_local = float(np.linalg.norm(wind_local))

        # ------------------------------------------------------------- flame sources
        hrr = a["hrr_flame"]
        src = np.nonzero((hrr > 4.0) & alive)[0]
        q_tot_kw = float(hrr[src].sum()) / 1000.0 if len(src) else 0.0
        bed = grid.coal_bed
        bed_q_kw = (bed.flare_hrr + 0.45 * bed.hrr) / 1000.0
        burners = grid.active_burners()
        T_gas_excess = np.zeros(n)
        q_rad = np.zeros(n)
        gas_v = np.full(n, 0.25 + wind_speed_local)
        pilot = np.zeros(n, dtype=bool)
        q_pilot = np.zeros(n)
        if len(src):
            Qi = hrr[src] / 1000.0
            Pi = P[src]
            Di = np.maximum(2.0 * R[src], 0.02)
            u_b = 1.9 * np.maximum(q_tot_kw, 0.05) ** 0.2
            tilt = wind_local / max(0.5, u_b)
            tilt_h = np.clip(np.linalg.norm(tilt), 0.0, 1.5)
            if tilt_h > 1e-6:
                tilt = tilt / np.linalg.norm(tilt) * tilt_h
            grid.flame_tilt = tilt.copy()
            Lf_i = np.maximum(0.03, 0.235 * Qi ** 0.4 - 1.02 * Di * 0.5)
            # ---- per-source plume temperatures on every receiver
            dz = P[None, :, 1] - Pi[:, None, 1]                      # (S, N)
            ax = Pi[:, None, 0] + tilt[0] * np.maximum(dz, 0.0)
            az = Pi[:, None, 2] + tilt[2] * np.maximum(dz, 0.0)
            rr = np.sqrt((P[None, :, 0] - ax) ** 2 + (P[None, :, 2] - az) ** 2)
            b = 0.5 * Di[:, None] + 0.16 * np.maximum(dz, 0.0) + 0.035
            dT0 = mccaffrey_excess_temperature(np.maximum(dz, 0.005), Qi[:, None])
            dT = np.where(dz > -0.005, dT0 * np.exp(-(rr / b) ** 2), 0.0)
            # lateral flame contact with pieces touching a burning piece at the same level
            contact = (dz <= 0.0) & (dz > -(R[src][:, None] + R[None, :] + 0.03)) & (rr < (R[src][:, None] + R[None, :] + 0.05))
            dT = np.where(contact, np.maximum(dT, 420.0 * a["phi"][src][:, None]), dT)
            same = src[:, None] == np.arange(n)[None, :]
            dT = np.where(same, 0.0, dT)
            T_gas_excess = dT.max(axis=0)
            # merged-fire plume (many small flames coalesce into one fire)
            if len(src) > 1:
                w = Qi / Qi.sum()
                c = (Pi * w[:, None]).sum(axis=0)
                D_f = max(0.1, 2.0 * float(np.sqrt(((Pi[:, [0, 2]] - c[[0, 2]]) ** 2).sum(axis=1)).max()) + 0.05)
                dzm = P[:, 1] - (c[1] - 0.03)
                rm = np.sqrt((P[:, 0] - (c[0] + tilt[0] * np.maximum(dzm, 0))) ** 2 + (P[:, 2] - (c[2] + tilt[2] * np.maximum(dzm, 0))) ** 2)
                bm = 0.5 * D_f + 0.16 * np.maximum(dzm, 0.0) + 0.03
                dTm = np.where(dzm > -0.01, mccaffrey_excess_temperature(np.maximum(dzm, 0.01), q_tot_kw) * np.exp(-(rm / bm) ** 2), 0.0)
                T_gas_excess = np.maximum(T_gas_excess, dTm)
        if bed_q_kw > 0.02:
            # the glowing ember bed (CO flames + flare-ups) is a plume source at the pit floor
            rb = max(0.05, bed.radius)
            u_bb = 1.9 * max(bed_q_kw + q_tot_kw, 0.05) ** 0.2
            tb = wind_local / max(0.5, u_bb)
            dzb = P[:, 1] - grid.floor_y
            rrb = np.sqrt((P[:, 0] - (grid.pit_center[0] + tb[0] * dzb)) ** 2 + (P[:, 2] - (grid.pit_center[2] + tb[2] * dzb)) ** 2)
            bb = rb + 0.16 * np.maximum(dzb, 0.0) + 0.03
            dTb = mccaffrey_excess_temperature(np.maximum(dzb, 0.01), bed_q_kw) * np.exp(-(rrb / bb) ** 2)
            T_gas_excess = np.maximum(T_gas_excess, dTb)
            imm = dTb > 350.0
            pilot |= imm
            q_pilot = np.maximum(q_pilot, np.where(imm, 14.0 * dTb, 0.0))
        for bu in burners:
            qb = bu["q_kw"]
            rb = bu["diameter"] * 0.5
            u_bu = 1.9 * max(qb, 0.05) ** 0.2
            tb = wind_local / max(0.5, u_bu)
            dzb = P[:, 1] - bu["pos"][1]
            rrb = np.sqrt((P[:, 0] - (bu["pos"][0] + tb[0] * np.maximum(dzb, 0))) ** 2 + (P[:, 2] - (bu["pos"][2] + tb[2] * np.maximum(dzb, 0))) ** 2)
            bb = rb + 0.16 * np.maximum(dzb, 0.0) + 0.03
            dTb = np.where(dzb > -0.02, mccaffrey_excess_temperature(np.maximum(dzb, 0.01), qb) * np.exp(-(rrb / bb) ** 2), 0.0)
            T_gas_excess = np.maximum(T_gas_excess, dTb)
            imm = dTb > 350.0
            pilot |= imm
            Lfb = heskestad_flame_height(qb, bu["diameter"])
            dfc = np.linalg.norm(P - (bu["pos"] + np.array([0.0, Lfb / 3.0, 0.0]))[None, :], axis=1)
            q_rad = q_rad + np.minimum(CHI_RADIATIVE * qb * 1000.0 / (4 * np.pi * np.maximum(dfc, 0.05) ** 2), 25e3)
            q_pilot = np.maximum(q_pilot, np.where(imm, 14.0 * dTb, 0.0))
        if len(src) or bed_q_kw > 0.02 or burners:
            T_gas_excess = np.minimum(T_gas_excess, 950.0)
            gas_v = gas_v + 1.9 * max(q_tot_kw + bed_q_kw, 0.05) ** 0.2 * np.clip(T_gas_excess / 600.0, 0.0, 1.0)
        if len(src):
            # ---- flame radiation (point source at 1/3 flame height)
            fc = Pi + np.stack([tilt[0] * Lf_i / 3.0, Lf_i / 3.0, tilt[2] * Lf_i / 3.0], axis=1)
            d = np.linalg.norm(P[None, :, :] - fc[:, None, :], axis=2)
            d = np.maximum(d, 0.04)
            qr = CHI_RADIATIVE * (Qi[:, None] * 1000.0) / (4.0 * np.pi * d * d)
            qr = np.where(same, 0.0, np.minimum(qr, 15e3))
            # total incident flame radiation limited by flame emissivity eps_f = 1 - exp(-k D)
            D_all = max(0.1, float(np.ptp(Pi[:, 0]) + np.ptp(Pi[:, 2])) * 0.5 + 0.1)
            eps_f = 1.0 - math.exp(-1.3 * D_all)
            q_rad = np.minimum(qr.sum(axis=0), eps_f * STEFAN_BOLTZMANN * 1150.0 ** 4)
            # flame immersion pilots the volatiles of neighbouring pieces
            immersed = T_gas_excess > 350.0
            pilot |= immersed
            q_pilot = np.maximum(q_pilot, np.where(immersed, 14.0 * T_gas_excess + q_rad, 0.0))

        # flames creeping along a stick pilot its next / previous segment
        phi = a["phi"]
        seg = a["seg"]
        for off_name in ("next_seg", "prev_seg"):
            off = a[off_name]
            has = off >= 0
            idx = np.nonzero(has)[0]
            if len(idx):
                j = idx + (off[idx] - seg[idx])
                creeping = phi[j] > 0.25
                pilot[idx[creeping]] = True
                q_pilot[idx[creeping]] = np.maximum(q_pilot[idx[creeping]], a["qfb"][j[creeping]] * 0.9)

        # ------------------------------------------------------------- solid-solid radiation
        Ls = Lseg
        Aproj = 2.0 * R * Ls
        dpos = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(dpos, axis=2)
        np.fill_diagonal(dist, 1e9)
        # view factor from receiver i to segment j: projected area over a hemisphere, limited
        # by the angular width of the neighbouring stick (avoids near-field over-counting)
        F = 0.5 * Aproj[None, :] / (np.pi * dist ** 2 + Aproj[None, :])
        F = np.minimum(F, (R[None, :] / (np.pi * dist)) * np.minimum(1.0, Ls[None, :] / (2.0 * dist)))
        F = np.where(alive[None, :], F, 0.0)
        F = np.minimum(F, 0.2)
        F[dist > 0.9] = 0.0
        same_body = a["body"][:, None] == a["body"][None, :]
        F = np.where(same_body, F * 0.25, F)
        Fsum = F.sum(axis=1)
        scale = np.where(Fsum > 0.6, 0.6 / np.maximum(Fsum, 1e-9), 1.0)
        F = F * scale[:, None]
        Fsum = F.sum(axis=1)
        T4 = (T_s + 273.15) ** 4
        T_block4 = np.where(Fsum > 1e-9, (F * T4[None, :]).sum(axis=1) / np.maximum(Fsum, 1e-9), 0.0)

        # coal bed below radiates up into the fuel above it
        floor_y = grid.floor_y
        if bed.char_kg > 1e-4:
            rb = bed.radius
            hz = np.maximum(0.01, P[:, 1] - floor_y)
            horiz = np.sqrt((P[:, 0] - grid.pit_center[0]) ** 2 + (P[:, 2] - grid.pit_center[2]) ** 2)
            Fb = 0.5 * (1.0 - hz / np.sqrt(hz * hz + rb * rb)) * np.exp(-np.maximum(0.0, horiz - rb) / 0.08)
            Fb = np.clip(Fb, 0.0, 0.45)
            Tb4 = (bed.T + 273.15) ** 4
            T_block4 = np.where(Fsum + Fb > 1e-9, (T_block4 * Fsum + Tb4 * Fb) / np.maximum(Fsum + Fb, 1e-9), T_block4)
            Fsum = np.minimum(0.9, Fsum + Fb)
            hot_bed = bed.T > 550.0
            touching = (P[:, 1] - R - floor_y < 0.03) & (horiz < rb + 0.03)
            pilot |= hot_bed & touching
            q_pilot = np.where(hot_bed & touching, np.maximum(q_pilot, 20e3 * min(1.0, (bed.T - 450.0) / 400.0)), q_pilot)
        env.F_block = Fsum
        env.T_block = np.where(Fsum > 1e-9, T_block4 ** 0.25 - 273.15, amb)

        # ------------------------------------------------------------- hand-held pilot (lighter)
        for pl in grid.active_pilots():
            ppos = pl["pos"]
            dd = np.linalg.norm(P - ppos[None, :], axis=1) - R - Ls * 0.35
            near = dd < pl["radius"]
            if np.any(near):
                # the small flame envelope licks every segment it touches (flux falls off)
                for k in np.nonzero(near)[0]:
                    f = max(0.25, 1.0 - max(0.0, dd[k]) / pl["radius"])
                    pilot[k] = True
                    q_pilot[k] = max(q_pilot[k], pl["flux"] * f)
                    env.pilot_area[k] = max(env.pilot_area[k], pl["area"] * f)

        env.T_gas = amb + T_gas_excess
        env.q_rad_in = q_rad * 0.9
        env.gas_velocity = gas_v
        env.pilot = pilot
        env.q_pilot = q_pilot
        env.o2[:] = grid.oxygen_fraction
        # pieces at the edge of the structure still breathe ambient air
        if grid.analysis.n_pieces > 0:
            c = grid.analysis.centroid
            horiz = np.sqrt((P[:, 0] - c[0]) ** 2 + (P[:, 2] - c[2]) ** 2)
            edge = np.clip((horiz - 0.35 * grid.analysis.diameter) / max(0.05, 0.25 * grid.analysis.diameter), 0.0, 1.0)
            top = np.clip((P[:, 1] - (floor_y + grid.analysis.height * 0.85)) / 0.05, 0.0, 1.0)
            openness = np.maximum(edge * (1.0 - grid.analysis.choke_fraction * 0.8), top)
            env.o2 = grid.oxygen_fraction + (0.2095 - grid.oxygen_fraction) * openness
        # ground / pit floor contact
        bottoms = P[:, 1] - R
        on_floor = bottoms - floor_y < 0.02
        env.ground_G = np.where(on_floor, SOIL_K * 2.0 * R * Ls * 6.0, 0.0)
        horiz0 = np.sqrt((P[:, 0] - grid.pit_center[0]) ** 2 + (P[:, 2] - grid.pit_center[2]) ** 2)
        in_bed = on_floor & (horiz0 < bed.radius + 0.02) & (bed.char_kg > 1e-3)
        env.T_ground = np.where(in_bed, bed.T, grid.pit_soil_T)
        # rain
        rain = env_state.get("rain", 0.0)
        if rain > 0 and not env_state.get("sheltered", False):
            env.rain_flux[:] = rain * 2.8e-3
        # breath (F key)
        if grid.breath_timer > 0 and grid.breath_velocity > 0:
            bt = grid.breath_target
            db = np.linalg.norm(P - bt[None, :], axis=1)
            env.breath_v = grid.breath_velocity * np.exp(-(db / 0.28) ** 2)
        return env


class ThermodynamicGrid:
    """
    Manages the 20 Hz thermal simulation of the fire pit: fuel bodies, ember bed, airflow,
    pilots and heat metrics.
    """

    def __init__(self, bounds_size: float = 2.0, grid_resolution: int = 16,
                 pit_center: Optional[np.ndarray] = None, seed: int = 11):
        self.bounds_size = bounds_size
        self.grid_resolution = grid_resolution
        self.voxel_size = bounds_size / grid_resolution
        self.pit_center = np.array(pit_center if pit_center is not None else [0.0, 0.0, 0.0], dtype=float)
        self.floor_y = float(self.pit_center[1])
        self.pit_radius = 0.45
        self.rng = np.random.default_rng(seed)

        self.bodies: List[CombustionBody] = []
        self.store = fk.FuelStore(0)
        self.seg_pos = np.zeros((0, 3))
        self.seg_rad = np.zeros(0)
        self.coal_bed = CoalBed()
        self.pit_soil_T = 20.0

        self.ambient_temp = 20.0
        self.wind_speed = 0.5
        self.internal_temp = 20.0
        self.core_coal_temp = 20.0

        # Structure airflow state
        self.structure_type = "freeform"
        self.analysis = StructureAnalysis()
        self.c_d = 0.6
        self.draft_velocity = 0.0
        self.oxygen_fraction = 0.2095
        self.base_vacuum = 0.0

        # Fire-level outputs
        self.flame_hrr = 0.0
        self.char_hrr = 0.0
        self.flame_height = 0.0
        self.fire_diameter = 0.0
        self.fire_centroid = self.pit_center.copy()
        self.flame_tilt = np.zeros(3)
        self.steam_rate = 0.0
        self.smoke_rate = 0.0          # kg/s unburnt pyrolysis products
        self.pop_events: List[np.ndarray] = []
        self.hiss = False
        self.spark_rate = 0.0

        # Energy accounting for economic credits
        self.total_energy_produced_joules = 0.0
        self.total_fuel_consumed_kg = 0.0
        self.total_water_evaporated_kg = 0.0
        self.current_heat_flux = 0.0
        self.efficiency_ratio = 0.0
        self.sim_time = 0.0

        # Pilots and breath
        self._pilots: List[Dict[str, Any]] = []
        self.collapse_positions: List[np.ndarray] = []
        self._burners: List[Dict[str, Any]] = []
        self._collapse_timer = 0.0
        self.breath_velocity = 0.0
        self.breath_timer = 0.0
        self.breath_target = self.pit_center + np.array([0.0, 0.05, 0.0])
        self.temp_history: List[float] = []
        self.hrr_history: List[float] = []
        self._history_timer = 0.0
        self._analysis_timer = 0.0
        self.structure_detector = None   # callable(bodies) -> (type, match)
        self.structure_match = 0.0

    # ------------------------------------------------------------------ bodies
    def add_body(self, body: CombustionBody):
        if body in self.bodies:
            return
        body.placed_time = self.sim_time
        self.bodies.append(body)
        self._rebuild()
        self._analysis_timer = 0.0

    def remove_body(self, body: CombustionBody):
        if body in self.bodies:
            body.detach()
            self.bodies.remove(body)
            self._rebuild()

    def clear(self):
        for b in list(self.bodies):
            self.remove_body(b)
        self.coal_bed = CoalBed(self.ambient_temp)

    def _rebuild(self):
        stores = []
        for bi, b in enumerate(self.bodies):
            part = b.store.extract(b.i0, b.i1)
            part.a["body"][:] = bi
            stores.append(part)
        self.store = fk.FuelStore.concat(stores) if stores else fk.FuelStore(0)
        i = 0
        for b in self.bodies:
            n = b.n_segments
            b.bind(self.store, i, i + n)
            i += n
        self._update_geometry()

    def _update_geometry(self):
        if not self.bodies:
            self.seg_pos = np.zeros((0, 3))
            self.seg_rad = np.zeros(0)
            return
        self.seg_pos = np.concatenate([b.segment_positions() for b in self.bodies], axis=0)
        self.seg_rad = np.concatenate([np.maximum(b.segment_radii(), 0.002) for b in self.bodies], axis=0)

    # ------------------------------------------------------------------ pilots & breath
    def add_pilot(self, pos: np.ndarray, duration: float = 0.1, flux: float = 70e3, radius: float = 0.06,
                  area: float = 4e-4, tag: str = "lighter"):
        """A hand-held flame (lighter) touching the fuel at pos for `duration` seconds."""
        self._pilots = [p for p in self._pilots if p["tag"] != tag]
        self._pilots.append({"pos": np.array(pos, dtype=float), "time": duration, "flux": flux,
                             "radius": radius, "area": area, "tag": tag})

    def add_burner(self, pos: np.ndarray, q_kw: float, duration: float, diameter: float = 0.15, tag: str = "burner"):
        """A fixed heat-release flame (pool fire of spilled accelerant, or a test burner)."""
        self._burners = [b for b in self._burners if b["tag"] != tag]
        self._burners.append({"pos": np.array(pos, dtype=float), "q_kw": q_kw, "time": duration,
                              "diameter": diameter, "tag": tag})

    def active_burners(self):
        return [b for b in self._burners if b["time"] > 0]

    def active_pilots(self):
        return [p for p in self._pilots if p["time"] > 0]

    def apply_breath(self, velocity: float, duration: float = 0.8, target: Optional[np.ndarray] = None):
        self.breath_velocity = velocity
        self.breath_timer = duration
        if target is not None:
            self.breath_target = np.array(target, dtype=float)
        for b in self.bodies:
            if np.linalg.norm(b.position - self.breath_target) < 0.35:
                b.blow(velocity)
        if velocity > 1.0:
            self.coal_bed.ash_kg *= 0.85

    # ------------------------------------------------------------------ main update
    def update(self, dt: float, ambient_temp: float = 20.0, wind_speed: float = 0.5,
               weather: Any = None, sheltered: bool = False):
        self.ambient_temp = ambient_temp
        self.wind_speed = wind_speed
        self.sim_time += dt
        env_state = {"ambient": ambient_temp, "sky_temp": ambient_temp - 12.0, "rh": 0.5,
                     "wind_vec": np.array([wind_speed, 0.0, 0.0]), "rain": 0.0, "sheltered": sheltered}
        if weather is not None:
            env_state.update({
                "ambient": weather.ambient_temp_c,
                "sky_temp": weather.sky_temperature_c(),
                "rh": weather.relative_humidity,
                "wind_vec": weather.wind_vector,
                "rain": weather.rain_intensity,
            })
            ambient_temp = weather.ambient_temp_c
            self.ambient_temp = ambient_temp
            self.wind_speed = weather.wind_speed_m_s

        # pit soil relaxes toward ambient, heated by the bed
        self.pit_soil_T += (ambient_temp - self.pit_soil_T) * min(1.0, dt / 1800.0)

        # Prune finished bodies: mostly-consumed pieces collapse into the ember bed
        self._collapse_timer -= dt
        if self._collapse_timer <= 0.0:
            self._collapse_timer = 0.25
            collapsed = False
            for b in list(self.bodies):
                if self._should_collapse(b):
                    self._collapse_into_bed(b)
                    collapsed = True
            if collapsed:
                self._rebuild()

        self._update_geometry()
        self._analysis_timer -= dt
        if self._analysis_timer <= 0.0:
            self._analyze_structure()
            self._analysis_timer = 0.5

        # Draft & oxygen from the previous tick's heat release
        self.draft_velocity = AirflowDraftSolver.calculate_draft_velocity(
            self.analysis.c_d, self.analysis.height, self.internal_temp, ambient_temp)
        _, self.base_vacuum = AirflowDraftSolver.calculate_base_vacuum(STANDARD_PRESSURE, self.draft_velocity)
        char_rate = float(self.store.a["char_rate"].sum()) if self.store.n else 0.0
        o2_target = AirflowDraftSolver.oxygen_balance(
            self.analysis, self.draft_velocity, self.flame_hrr, char_rate, self.wind_speed)
        # the air inside the structure has inertia (volume / flow ~ 0.6 s)
        self.oxygen_fraction += (o2_target - self.oxygen_fraction) * min(1.0, dt / 0.6)
        if self.breath_timer > 0:
            self.oxygen_fraction = max(self.oxygen_fraction, 0.17)

        env = ThermalConvectionSolver.build_environment(self, env_state)
        if self.store.n:
            shed = fk.step_fuel(self.store, env, dt, self.rng)
            if shed is not None:
                self.coal_bed.ash_kg += float(np.sum(shed))

        # Ember bed
        covered_F = 0.0
        cover_T = ambient_temp
        if self.store.n:
            P = self.seg_pos
            horiz = np.sqrt((P[:, 0] - self.pit_center[0]) ** 2 + (P[:, 2] - self.pit_center[2]) ** 2)
            above = horiz < max(0.1, self.coal_bed.radius + 0.05)
            if np.any(above):
                covered_F = min(0.7, 0.12 * float(above.sum()))
                cover_T = float(np.mean(self.store.a["T_surf"][above]))
        bv = self.breath_velocity if self.breath_timer > 0 else 0.0
        q_soil = self.coal_bed.step(dt, ambient_temp, env_state["sky_temp"], max(self.oxygen_fraction, 0.16),
                                    0.3 + self.draft_velocity * 0.5, bv * 0.8,
                                    env_state["rain"] * 2.8e-3 if not sheltered else 0.0,
                                    covered_F, cover_T, self.pit_soil_T)
        self.pit_soil_T += q_soil / 60000.0

        # pilots & breath decay
        for p in self._pilots:
            p["time"] -= dt
        burner_hrr = 0.0
        for bu in self._burners:
            bu["time"] -= dt
            burner_hrr += bu["q_kw"] * 1000.0 if bu["time"] > 0 else 0.0
        self._burners = [bu for bu in self._burners if bu["time"] > 0]
        self.burner_hrr = burner_hrr
        self._pilots = [p for p in self._pilots if p["time"] > 0]
        if self.breath_timer > 0:
            self.breath_timer = max(0.0, self.breath_timer - dt)

        self._aggregate(dt, env)
        self._post_step_bodies()

    def _post_step_bodies(self):
        if not self.bodies:
            return
        a = self.store.a
        starts = np.array([b.i0 for b in self.bodies])
        present = (a["present"] > 0.5).any(axis=1).astype(float)
        solids = a["pool"].sum(axis=(1, 2)) + a["char"].sum(axis=1)
        T_s = a["T_surf"]
        glow = ((a["char_rate"] > 1e-7) & (T_s >= 450.0)).astype(float)
        smolder = ((a["char_rate"] > 1e-7) & (T_s >= 300.0)).astype(float)
        r = lambda v: np.add.reduceat(v, starts)
        sums = {
            "alive": r(present), "solids": r(solids), "phi": r(a["phi"]), "glow": r(glow),
            "vol": r(a["vol_rate"]), "steam": r(a["steam_rate"]), "smolder": r(smolder),
            "pop": r(a["pop_event"]), "hiss": r(a["hiss_event"]), "hrr_flame": r(a["hrr_flame"]),
        }
        tmax = np.maximum.reduceat(T_s, starts)
        for i, b in enumerate(self.bodies):
            agg = {k: float(v[i]) for k, v in sums.items()}
            agg["tmax"] = float(tmax[i])
            b.post_step(agg)

    def _should_collapse(self, b: CombustionBody) -> bool:
        if b.stage == CombustionStage.INERT_ASH or b.is_consumed or b.burned_fraction > 0.93:
            return True
        hot = b.t_surface > 250.0
        if b.volatile_mass < 0.03 * b.initial_dry_mass and not b.is_flaming:
            return True          # a fully charred skeleton crumbles
        if not hot:
            return False
        if b.geometry_type in ("tinder", "kindling") and b.burned_fraction > 0.62:
            return True          # thin sticks burn through and fall into the coals
        radii = b.segment_radii()
        if b.geometry_type in ("firewood", "log") and (b.burned_fraction > 0.86 or
                                                        np.min(radii) < 0.25 * b.initial_radius):
            return True          # burned through in the middle -> breaks into coals
        return False

    def _collapse_into_bed(self, b: CombustionBody):
        a = b.store.a
        sl = slice(b.i0, b.i1)
        pool = a["pool"][sl]            # (seg, shell, comp)
        y = a["yield_"][sl]
        char_from_virgin = float((pool * y).sum())
        vol_energy = float((pool * (1.0 - y) * fk.COMP_HOC[None, None, :]).sum())
        self.coal_bed.add(b.char_mass + char_from_virgin, max(b.t_surface, 350.0) if b.t_surface > 250 else b.t_surface,
                          b.props.charcoal_caloric_value)
        self.coal_bed.ash_kg += b.ash_mass
        self.coal_bed.flare_j += vol_energy * 0.85
        self.total_water_evaporated_kg += b.water_mass
        self.coal_bed.collapse_events += 1
        self.collapse_positions.append(b.position.copy())
        self.bodies.remove(b)

    def _analyze_structure(self):
        if not self.bodies:
            self.analysis = StructureAnalysis()
            self.structure_type = "freeform"
            return
        stype = "freeform"
        match = 0.0
        if self.structure_detector is not None:
            stype, match = self.structure_detector(self.bodies)
        self.structure_type = stype
        self.structure_match = match
        pieces = [b for b in self.bodies if b.geometry_type != "tinder"] or self.bodies
        self.analysis = AirflowDraftSolver.analyze_pieces(
            [b.position for b in pieces], [b.axis for b in pieces], [b.length for b in pieces],
            [b.initial_radius for b in pieces], [b.geometry_type for b in pieces],
            self.floor_y, stype)
        self.c_d = self.analysis.c_d

    def _aggregate(self, dt: float, env: fk.Environment):
        a = self.store.a
        n = self.store.n
        self.flame_hrr = (float(a["hrr_flame"].sum()) if n else 0.0) + self.coal_bed.flare_hrr + getattr(self, "burner_hrr", 0.0)
        self.char_hrr = (float(a["hrr_char"].sum()) if n else 0.0) + self.coal_bed.hrr
        total = self.flame_hrr + self.char_hrr
        self.current_heat_flux = total
        self.total_energy_produced_joules += total * dt
        consumed = (float(a["vol_rate"].sum() + a["char_rate"].sum()) if n else 0.0) + self.coal_bed.burn_rate
        self.total_fuel_consumed_kg += consumed * dt
        self.steam_rate = float(a["steam_rate"].sum()) if n else 0.0
        self.total_water_evaporated_kg += self.steam_rate * dt
        self.smoke_rate = float(a["smoke_rate"].sum()) if n else 0.0
        # efficiency: useful heat vs. the heating value of what was consumed (dry wood ~18 MJ/kg)
        if consumed > 1e-7:
            inst = total / (consumed * 18.0e6)
            self.efficiency_ratio += (min(1.2, inst) - self.efficiency_ratio) * min(1.0, dt / 5.0)
        else:
            self.efficiency_ratio *= max(0.0, 1.0 - dt / 20.0)

        # Fire geometry (segment flames + ember-bed flare-ups + burners)
        pts, wts = [], []
        if n:
            fl = a["hrr_flame"] > 5.0
            if np.any(fl):
                pts.append(self.seg_pos[fl])
                wts.append(a["hrr_flame"][fl])
        if self.coal_bed.flare_hrr > 5.0:
            pts.append(self.pit_center[None, :] + np.array([[0.0, 0.03, 0.0]]))
            wts.append(np.array([self.coal_bed.flare_hrr]))
        for bu in self.active_burners():
            pts.append(bu["pos"][None, :])
            wts.append(np.array([bu["q_kw"] * 1000.0]))
        if pts and self.flame_hrr > 5.0:
            P = np.concatenate(pts, axis=0)
            W = np.concatenate(wts)
            c = (P * (W / W.sum())[:, None]).sum(axis=0)
            self.fire_centroid = c
            spread = np.sqrt(((P[:, [0, 2]] - c[[0, 2]]) ** 2).sum(axis=1)).max()
            bed_d = 2.0 * self.coal_bed.radius if self.coal_bed.flare_hrr > 5 else 0.0
            self.fire_diameter = float(max(0.06, 2.0 * spread + 0.04, bed_d))
            q_kw = self.flame_hrr / 1000.0
            hk = heskestad_flame_height(q_kw, self.fire_diameter)
            per_piece = 0.0
            if n:
                starts = np.array([b.i0 for b in self.bodies])
                q_b = np.add.reduceat(a["hrr_flame"], starts) / 1000.0
                if q_b.max() > 0.005:
                    per_piece = float(np.max(0.235 * q_b ** 0.4 - 0.02))
            self.flame_height = max(hk, per_piece)
        else:
            self.flame_height = 0.0
            self.fire_diameter = 0.0
            if self.coal_bed.char_kg > 0.01:
                self.fire_centroid = self.pit_center.copy()

        # Temperatures
        hot = []
        if n:
            alive = (a["present"] > 0.5).any(axis=1)
            burning = alive & ((a["phi"] > 0) | (a["char_rate"] > 1e-7) | (a["vol_rate"] > 1e-6))
            if np.any(burning):
                hot = list(a["T_surf"][burning])
        if self.coal_bed.char_kg > 0.005:
            hot.append(self.coal_bed.T)
        if hot:
            self.core_coal_temp = float(max(hot))
        elif n:
            self.core_coal_temp = float(np.max(a["T_surf"]))
        else:
            self.core_coal_temp = max(self.ambient_temp, self.coal_bed.T if self.coal_bed.char_kg > 0 else self.ambient_temp)
        # gas temperature inside the structure (drives the stack effect)
        if n and self.flame_hrr + self.char_hrr > 50.0:
            q_kw = (self.flame_hrr + 0.5 * self.char_hrr) / 1000.0
            mid = max(0.02, self.analysis.height * 0.5)
            t_gas = self.ambient_temp + float(mccaffrey_excess_temperature(np.array([mid]), np.array([q_kw]))[0])
            self.internal_temp += (t_gas - self.internal_temp) * min(1.0, dt / 1.5)
        else:
            self.internal_temp += (max(self.ambient_temp, self.coal_bed.T * 0.5) - self.internal_temp) * min(1.0, dt / 4.0)

        # events
        self.pop_events = []
        if n:
            pops = np.nonzero(a["pop_event"] > 0)[0]
            for k in pops:
                self.pop_events.append(self.seg_pos[k].copy())
        self.hiss = bool(n and np.any(a["hiss_event"] > 0))
        self.spark_rate = 0.002 * self.flame_hrr + 0.0008 * self.char_hrr

        self._history_timer -= dt
        if self._history_timer <= 0:
            self._history_timer = 1.0
            self.temp_history.append(self.core_coal_temp)
            self.hrr_history.append(total)
            self.temp_history = self.temp_history[-180:]
            self.hrr_history = self.hrr_history[-180:]

    # ------------------------------------------------------------------ queries
    @property
    def is_burning(self) -> bool:
        return self.flame_hrr > 30.0 or (self.coal_bed.T > 450 and self.coal_bed.char_kg > 0.01) or \
            any(b.stage in (CombustionStage.FLAMING, CombustionStage.CHAR_OXIDATION) for b in self.bodies)

    @property
    def has_flame(self) -> bool:
        return self.flame_hrr > 20.0

    def fire_stage(self) -> str:
        if not self.bodies and self.coal_bed.char_kg < 0.005:
            return "Cold Hearth" if self.coal_bed.ash_kg < 0.01 else "Cold Ashes"
        if self.has_flame:
            return CombustionStage.FLAMING.value
        if self.coal_bed.T > 450 and self.coal_bed.char_kg > 0.01:
            return CombustionStage.CHAR_OXIDATION.value
        stages = [b.stage for b in self.bodies]
        for s in (CombustionStage.CHAR_OXIDATION, CombustionStage.PYROLYSIS, CombustionStage.DEHYDRATION):
            if s in stages:
                return s.value
        return CombustionStage.AMBIENT.value

    def blow_into_fire(self, breath_velocity: float) -> Tuple[float, bool]:
        """Player breathes into the fire structure ('F' key)."""
        power_boost, o2, blown_away = AirflowDraftSolver.calculate_breath_effect(
            breath_velocity, self.core_coal_temp, self.ambient_temp)
        self.apply_breath(breath_velocity, 0.8)
        if blown_away:
            # scatter ash and cool the smallest pieces
            self.coal_bed.ash_kg *= 0.5
            for b in self.bodies:
                if b.geometry_type == "tinder":
                    b._v("phi")[:] *= 0.3
        return power_boost, blown_away

    def is_extinguished(self) -> bool:
        """No flame, no glowing char anywhere, and everything has cooled below ~80 C."""
        if self.flame_hrr > 5.0 or self.char_hrr > 5.0:
            return False
        if self.coal_bed.char_kg > 1e-4 and self.coal_bed.T > 80.0:
            return False
        if self.store.n and float(np.max(self.store.a["T_surf"])) > 80.0:
            return False
        return True

    def fast_forward(self, seconds: float, weather: Any = None, max_steps: int = 40000,
                     on_step: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Advance the fire through a time skip with the full physics, using large adaptive
        steps (the implicit-enough kernel stays accurate to ~1 % of burnt mass at 10 s steps):
        ~8 s while flames dance, 10 s over glowing coals, 30 s while the remains cool.
        ``on_step`` lets the world run its settling physics (logs slumping into the coals).
        Untended fires genuinely go out - often leaving charred log ends - they are never
        deleted by a shortcut."""
        t = 0.0
        steps = 0
        e0 = self.total_energy_produced_joules
        out_at = None
        while t < seconds - 1e-9 and steps < max_steps:
            if not self.bodies and self.coal_bed.char_kg < 1e-4 and self.coal_bed.T < self.ambient_temp + 5.0:
                break
            if self.flame_hrr > 2000.0:
                h = 8.0
            elif self.flame_hrr > 10.0 or self.char_hrr > 50.0 or self.coal_bed.T > 300.0:
                h = 10.0
            else:
                h = 30.0
            h = min(h, seconds - t)
            self.update(h, weather=weather) if weather is not None else self.update(h, self.ambient_temp, self.wind_speed)
            if on_step is not None:
                on_step(h)
            t += h
            steps += 1
            if self.is_extinguished():
                out_at = t if out_at is None else out_at
                if t - out_at > 1800.0:
                    break          # cold for half an hour: nothing more will happen
            else:
                out_at = None
        remaining = seconds - t
        if remaining > 0 and self.is_extinguished():
            # long after the fire died everything sits at the ambient temperature
            self.store.a["T"][:] = np.minimum(self.store.a["T"], self.ambient_temp)
            self.store.a["T_surf"][:] = np.minimum(self.store.a["T_surf"], self.ambient_temp)
            self.coal_bed.T = self.ambient_temp
            self.core_coal_temp = self.ambient_temp
            self.internal_temp = self.ambient_temp
            self.flame_hrr = self.char_hrr = 0.0
            self.flame_height = 0.0
        return {"simulated_s": t, "steps": steps, "burning": not self.is_extinguished(),
                "energy_mj": (self.total_energy_produced_joules - e0) / 1e6}
