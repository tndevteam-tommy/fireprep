"""
Firestarter - CombustionBody Base Class & Four-Stage State Machine
Implements Sections 2.1, 2.2, and 2.3 of the Technical Design Specification.

A CombustionBody is one physical fuel piece (tinder nest, kindling stick, split firewood,
log, punk wood chunk).  Its thermal state lives in a finite-volume grid of lengthwise
segments x radial shells (see ``fuel_kernel``), stored in a ``FuelStore``.  When a body is
added to the ``ThermodynamicGrid`` its arrays are merged into the fire's shared store so the
whole fire is simulated in vectorised form; the body keeps working views into that store.

Stage machine (derived every tick from the cell state, DESIGN 2.1):

    AMBIENT -> DEHYDRATION (steam, 100-120 C plateau) -> PYROLYSIS (smoke, 200-450 C)
            -> FLAMING (gas-phase flame, O2 > 12 %) / CHAR_OXIDATION (glowing coals 500-900 C)
            -> INERT_ASH
"""

from enum import Enum
import math
from typing import Optional, Dict, Any, List

import numpy as np

from firestarter.thermodynamics.wood_properties import (
    WoodSpeciesType, WoodProperties, get_wood_properties
)
from firestarter.thermodynamics import fuel_kernel as fk


class CombustionStage(Enum):
    AMBIENT = "Ambient"
    DEHYDRATION = "Dehydration"
    PYROLYSIS = "Pyrolysis"
    FLAMING = "Flaming Combustion"
    CHAR_OXIDATION = "Char Oxidation (Glowing Coals)"
    INERT_ASH = "Inert Ash"


# Typical real-world dimensions [length m, radius m] for each fuel class
DEFAULT_DIMENSIONS = {
    "tinder": (0.14, 0.045),     # a fist-sized nest of shavings / bark / needles
    "kindling": (0.30, 0.008),   # pencil-to-thumb thick dry twigs
    "firewood": (0.38, 0.030),   # wrist-thick split fuel wood
    "log": (0.45, 0.060),        # forearm-thick log
    "fungus": (0.08, 0.040),
}
TINDER_DEFAULT_MASS = 0.035      # kg


def orientation_to_axis(orientation: np.ndarray) -> np.ndarray:
    """orientation = [tilt above horizontal (deg), yaw (deg, from +X toward +Z), roll]."""
    tilt = math.radians(float(orientation[0]) if len(orientation) > 0 else 0.0)
    yaw = math.radians(float(orientation[1]) if len(orientation) > 1 else 0.0)
    v = np.array([math.cos(tilt) * math.cos(yaw), math.sin(tilt), math.cos(tilt) * math.sin(yaw)])
    return v / max(1e-9, np.linalg.norm(v))


class CombustionBody:
    """
    Every combustive object in the game world inherits from CombustionBody.
    Tracks M_surface, M_core, T_internal, mass, ash_mass and a spatial bounding capsule.
    """
    _next_id = 1

    def __init__(
        self,
        species: WoodSpeciesType,
        dry_mass: Optional[float] = None,
        initial_m_surface: float = 0.15,
        initial_m_core: float = 0.15,
        initial_temp: float = 20.0,
        geometry_type: str = "log",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        dimensions: Optional[np.ndarray] = None,
        n_segments: Optional[int] = None,
        n_shells: Optional[int] = None,
    ):
        self.id = CombustionBody._next_id
        CombustionBody._next_id += 1
        self.species_type = species
        self.props: WoodProperties = get_wood_properties(species)
        self.geometry_type = geometry_type
        self.kind = fk.KIND_NAMES.get(geometry_type, fk.KIND_LOG)

        dims = DEFAULT_DIMENSIONS.get(geometry_type, (0.4, 0.05))
        if dimensions is None:
            dimensions = np.array(dims, dtype=float)
        self.dimensions = np.array(dimensions, dtype=float)
        length = float(self.dimensions[0])
        radius = float(self.dimensions[1]) if len(self.dimensions) > 1 else 0.05
        if dry_mass is None:
            if geometry_type == "tinder":
                dry_mass = TINDER_DEFAULT_MASS
            else:
                dry_mass = self.props.bulk_density * math.pi * radius * radius * length
        self.initial_dry_mass = max(1e-4, float(dry_mass))

        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=float)
        self.orientation = np.array(orientation if orientation is not None else [0.0, 0.0, 0.0], dtype=float)

        self.store: fk.FuelStore = fk.build_fuel_store(
            self.props, self.kind, length, radius, self.initial_dry_mass,
            max(0.0, initial_m_surface), max(0.0, initial_m_core), initial_temp,
            n_segments, n_shells)
        self.i0 = 0
        self.i1 = self.store.n
        self.n_segments = self.store.n
        self.n_shells = int(self.store.a["nsh"][0])

        # Environmental contact flags
        self.is_on_ground = True
        self.is_tarp_covered = False
        self.is_decayed = (species == WoodSpeciesType.PUNK_WOOD)
        self.accelerant_type = "none"
        self.stage = CombustionStage.AMBIENT
        self.resin_pop_event = False
        self.hiss_event = False
        self.crackle_factor = 0.0
        self._standalone_rng = np.random.default_rng(self.id)
        self.placed_time = 0.0
        self._update_stage()

    # ------------------------------------------------------------------ store views
    def _v(self, name: str) -> np.ndarray:
        return self.store.a[name][self.i0:self.i1]

    def bind(self, store: fk.FuelStore, i0: int, i1: int):
        self.store = store
        self.i0 = i0
        self.i1 = i1

    def detach(self):
        """Copy this body's slice into a private store (when removed from a grid)."""
        self.store = self.store.extract(self.i0, self.i1)
        self.store.a["body"][:] = 0
        self.i0, self.i1 = 0, self.store.n

    # ------------------------------------------------------------------ geometry
    @property
    def axis(self) -> np.ndarray:
        return orientation_to_axis(self.orientation)

    @property
    def length(self) -> float:
        return float(self.dimensions[0])

    @property
    def radius(self) -> float:
        """Current effective outer radius (shrinks as the piece burns away)."""
        return float(np.max(self.segment_radii())) if self.n_segments else 0.0

    @property
    def initial_radius(self) -> float:
        return float(self.dimensions[1]) if len(self.dimensions) > 1 else 0.05

    def segment_positions(self) -> np.ndarray:
        key = (tuple(np.round(self.position, 6)), tuple(np.round(self.orientation, 4)))
        if getattr(self, "_pos_key", None) != key:
            f = self._v("seg_frac")
            self._pos_cache = self.position[None, :] + self.axis[None, :] * (f[:, None] * self.length)
            self._pos_key = key
        return self._pos_cache

    def segment_radii(self) -> np.ndarray:
        """Effective outer radius of each segment (shrinks as char oxidises away)."""
        return self._v("r_eff").copy()

    def _compute_segment_radii(self) -> np.ndarray:
        present = self._v("present") > 0.5
        if not np.any(present):
            return np.zeros(self.n_segments)
        pool = self._v("pool").sum(axis=2)
        char = self._v("char")
        dry0 = self._v("dry0")
        rin = self._v("rin")
        rout = self._v("rout")
        vfrac = np.clip(pool / np.maximum(dry0, 1e-12) + 0.7 * char / np.maximum(dry0 * 0.3, 1e-12), 0.0, 1.0)
        vfrac = np.where(present, vfrac, 0.0)
        o = np.argmax(present, axis=1)
        rows = np.arange(self.n_segments)
        alive = present.any(axis=1)
        r = np.sqrt(rin[rows, o] ** 2 + vfrac[rows, o] * (rout[rows, o] ** 2 - rin[rows, o] ** 2))
        return np.where(alive, r, 0.0)

    def endpoints(self):
        half = self.axis * self.length * 0.5
        return self.position - half, self.position + half

    def bottom_y(self) -> float:
        a, b = self.endpoints()
        return float(min(a[1], b[1]) - self.initial_radius)

    def top_y(self) -> float:
        a, b = self.endpoints()
        return float(max(a[1], b[1]) + self.initial_radius)

    @property
    def surface_area(self) -> float:
        r = self.radius
        return 2.0 * math.pi * r * self.length + 2.0 * math.pi * r * r

    # ------------------------------------------------------------------ mass pools
    @property
    def volatile_mass(self) -> float:
        return float(self._v("pool").sum())

    @property
    def char_mass(self) -> float:
        return float(self._v("char").sum())

    @property
    def char_yield_max(self) -> float:
        return self.initial_dry_mass * self.props.char_yield_fraction

    @property
    def dry_mass(self) -> float:
        return self.volatile_mass + self.char_mass

    @property
    def ash_mass(self) -> float:
        return float(self._v("ash").sum() + self._v("ash_kg").sum())

    @property
    def water_mass(self) -> float:
        return float(self._v("water").sum())

    @property
    def total_mass(self) -> float:
        return max(1e-4, self.dry_mass + self.water_mass + self.accelerant_mass + self.ash_mass)

    @property
    def accelerant_mass(self) -> float:
        return float(self._v("accel").sum())

    @property
    def burned_fraction(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.dry_mass / self.initial_dry_mass))

    # ------------------------------------------------------------------ moisture
    def _moisture_layers(self):
        water = self._v("water")
        solids = self._v("pool").sum(axis=2) + self._v("char") + self._v("ash")
        present = self._v("present") > 0.5
        return water, solids, present

    @property
    def m_surface(self) -> float:
        water, solids, present = self._moisture_layers()
        if not np.any(present):
            return 0.0
        o = np.argmax(present, axis=1)
        rows = np.arange(self.n_segments)
        alive = present.any(axis=1)
        w = water[rows, o][alive].sum()
        s = solids[rows, o][alive].sum()
        return float(w / max(1e-9, s))

    @m_surface.setter
    def m_surface(self, value: float):
        self.set_moisture(surface=value)

    @property
    def m_core(self) -> float:
        water, solids, present = self._moisture_layers()
        if self.n_shells <= 1:
            return self.m_surface
        inner = present.copy()
        o = np.argmax(present, axis=1)
        inner[np.arange(self.n_segments), o] = False
        if not np.any(inner):
            return self.m_surface
        return float(water[inner].sum() / max(1e-9, solids[inner].sum()))

    @m_core.setter
    def m_core(self, value: float):
        self.set_moisture(core=value)

    @property
    def moisture_content(self) -> float:
        water, solids, present = self._moisture_layers()
        return float(water.sum() / max(1e-9, solids.sum()))

    def set_moisture(self, surface: Optional[float] = None, core: Optional[float] = None):
        """Set dry-basis moisture; surface applies to the outer shell, core to inner shells."""
        water, solids, present = self._moisture_layers()
        R = self.n_shells
        for k in range(R):
            if R == 1:
                m = surface if surface is not None else core
            elif k == 0:
                m = surface
            else:
                m = core
            if m is None:
                continue
            water[:, k] = np.where(present[:, k], max(0.0, m) * solids[:, k], 0.0)

    # ------------------------------------------------------------------ temperatures
    @property
    def t_surface(self) -> float:
        alive = (self._v("present") > 0.5).any(axis=1)
        if not np.any(alive):
            return float(np.mean(self._v("T")[:, 0]))
        return float(np.max(self._v("T_surf")[alive]))

    @t_surface.setter
    def t_surface(self, value: float):
        T = self._v("T")
        present = self._v("present") > 0.5
        o = np.argmax(present, axis=1)
        T[np.arange(self.n_segments), o] = value
        self._v("T_surf")[:] = value

    @property
    def t_internal(self) -> float:
        T = self._v("T")
        present = self._v("present") > 0.5
        if not np.any(present):
            return float(np.mean(T))
        inner = np.where(present, np.arange(fk.R_MAX)[None, :], -1).max(axis=1)
        rows = np.arange(self.n_segments)
        alive = present.any(axis=1)
        return float(np.mean(T[rows, inner][alive]))

    @t_internal.setter
    def t_internal(self, value: float):
        T = self._v("T")
        present = self._v("present") > 0.5
        T[present] = value

    def set_temperature(self, value: float):
        self._v("T")[:] = value
        self._v("T_surf")[:] = value
        self._v("ice")[:] = 1.0 if value < 0 else 0.0

    @property
    def mean_temperature(self) -> float:
        return float(np.mean(self._v("T")[self._v("present") > 0.5])) if np.any(self._v("present") > 0.5) else 20.0

    # ------------------------------------------------------------------ outputs
    @property
    def heat_flux_output(self) -> float:
        return float(self._v("hrr_flame").sum() + self._v("hrr_char").sum())

    @property
    def flame_hrr(self) -> float:
        return float(self._v("hrr_flame").sum())

    @property
    def steam_rate(self) -> float:
        return float(self._v("steam_rate").sum())

    @property
    def smoke_rate(self) -> float:
        """Unburnt pyrolysis smoke (g/s)."""
        return float(self._v("smoke_rate").sum() * 1000.0)

    @property
    def flame_fraction(self) -> float:
        return float(np.mean(self._v("phi")))

    @property
    def is_flaming(self) -> bool:
        return bool(np.any(self._v("phi") > 0))

    @property
    def flame_height(self) -> float:
        """Heskestad flame height of this piece's own flames (m)."""
        q_kw = self.flame_hrr / 1000.0
        if q_kw <= 0.005:
            return 0.0
        D = max(0.02, 2.0 * self.radius)
        return max(0.03, 0.235 * q_kw ** 0.4 - 1.02 * D * 0.5)

    @property
    def ash_insulation_depth(self) -> float:
        a = self._v("ash_kg")
        A = 2.0 * math.pi * max(self.radius, 0.002) * (self.length / max(1, self.n_segments))
        return float(np.max(a / (fk.RHO_ASH * A) * 1000.0)) if len(a) else 0.0

    @property
    def is_consumed(self) -> bool:
        return not bool(np.any(self._v("present") > 0.5))

    # ------------------------------------------------------------------ actions
    def add_accelerant(self, mass_kg: float, accelerant_type: str = "lighter_fluid"):
        acc = self._v("accel")
        acc += max(0.0, mass_kg) / max(1, self.n_segments)
        self._v("accel_kind")[:] = 2 if accelerant_type == "gasoline" else 1
        self.accelerant_type = accelerant_type

    def clear_ash(self):
        """Removes insulating ash blanket when cleared or blown."""
        self._v("ash_kg")[:] *= 0.35

    def blow(self, intensity: float):
        """Disturbs ash and injects oxygen."""
        self.clear_ash()

    def apply_external_heat(self, watts: float, dt: float):
        """Applies heat directly to the outer surface of every segment."""
        if watts <= 0.0:
            return
        T = self._v("T")
        present = self._v("present") > 0.5
        if not np.any(present):
            return
        o = np.argmax(present, axis=1)
        rows = np.arange(self.n_segments)
        pool = self._v("pool").sum(axis=2)[rows, o]
        char = self._v("char")[rows, o]
        water = self._v("water")[rows, o]
        C = fk.cp_wood(T[rows, o]) * pool + fk.cp_char(T[rows, o]) * char + water * 4184.0
        C = np.maximum(C, 1e-4)
        T[rows, o] += (watts * dt / self.n_segments) / C

    def convert_to_punk(self):
        """Fungal decay: the wood becomes porous, lignin-rich punk (sponge)."""
        if self.species_type == WoodSpeciesType.PUNK_WOOD:
            return
        m_s, m_c = self.m_surface, self.m_core
        self.species_type = WoodSpeciesType.PUNK_WOOD
        self.props = get_wood_properties(WoodSpeciesType.PUNK_WOOD)
        self.is_decayed = True
        a = self.store.a
        sl = slice(self.i0, self.i1)
        # Brown/white rot consumes ~60% of the cell-wall mass and most of the cellulose
        a["pool"][sl] *= 0.4
        a["pool"][sl, :, 2] *= 0.4
        a["dry0"][sl] *= 0.4
        self.initial_dry_mass *= 0.4
        p = self.props
        a["k0"][sl] = p.thermal_conductivity
        a["D0"][sl] = p.moisture_diffusion_coeff
        a["Tig"][sl] = p.volatile_ignition_temp
        a["Tauto"][sl] = p.autoignition_temp
        a["Lg"][sl] = p.heat_of_gasification
        a["hchar"][sl] = p.charcoal_caloric_value
        a["react"][sl] = p.char_reactivity
        a["rho"][sl] *= 0.4
        a["resin"][sl] = p.resin_content
        a["pop"][sl] = p.spark_pop_rate
        self.set_moisture(surface=m_s, core=m_c)

    # ------------------------------------------------------------------ standalone update
    def update(
        self,
        dt: float,
        ambient_temp: float = 20.0,
        oxygen_fraction: float = 0.2095,
        draft_velocity: float = 0.5,
        wind_speed: float = 0.5,
    ):
        """Advance this body on its own (not part of a fire grid)."""
        env = fk.Environment(self.n_segments, ambient_temp)
        env.o2[:] = oxygen_fraction
        env.gas_velocity[:] = max(0.1, 0.5 * wind_speed + draft_velocity)
        env.T_sky = ambient_temp - 8.0
        self.step_with_env(env, dt, self._standalone_rng)

    def step_with_env(self, env: "fk.Environment", dt: float, rng: np.random.Generator):
        if self.i0 != 0 or self.i1 != self.store.n:
            raise RuntimeError("Body is bound to a shared grid store; step the grid instead.")
        remaining = dt
        while remaining > 1e-9:
            h = min(0.25, remaining)
            fk.step_fuel(self.store, env, h, rng)
            remaining -= h
        self.post_step()

    def post_step(self, agg: Optional[Dict[str, float]] = None):
        if agg is None:
            self.resin_pop_event = bool(np.any(self._v("pop_event") > 0))
            self.hiss_event = bool(np.any(self._v("hiss_event") > 0))
            self.crackle_factor = min(1.0, self.flame_hrr / 3000.0)
            self._update_stage()
            return
        self.resin_pop_event = agg["pop"] > 0
        self.hiss_event = agg["hiss"] > 0
        self.crackle_factor = min(1.0, agg["hrr_flame"] / 3000.0)
        self._stage_from_agg(agg)

    def _stage_from_agg(self, g: Dict[str, float]):
        if g["alive"] <= 0 or g["solids"] < 0.01 * self.initial_dry_mass:
            self.stage = CombustionStage.INERT_ASH
        elif g["phi"] > 0:
            self.stage = CombustionStage.FLAMING
        elif g["glow"] > 0:
            self.stage = CombustionStage.CHAR_OXIDATION
        elif g["vol"] > 2e-6 or (g["tmax"] >= 200.0 and g["vol"] > 1e-7):
            self.stage = CombustionStage.PYROLYSIS
        elif g["steam"] > 1e-6 and g["tmax"] >= 60.0:
            self.stage = CombustionStage.DEHYDRATION
        elif g["smolder"] > 0:
            self.stage = CombustionStage.CHAR_OXIDATION
        else:
            self.stage = CombustionStage.AMBIENT

    def _update_stage(self):
        a = self.store.a
        sl = slice(self.i0, self.i1)
        present = a["present"][sl] > 0.5
        if not np.any(present) or self.dry_mass < 0.01 * self.initial_dry_mass:
            self.stage = CombustionStage.INERT_ASH
            return
        phi = a["phi"][sl]
        if np.any(phi > 0):
            self.stage = CombustionStage.FLAMING
            return
        T_s = a["T_surf"][sl]
        if np.any((a["char_rate"][sl] > 1e-7) & (T_s >= 450.0)):
            self.stage = CombustionStage.CHAR_OXIDATION
            return
        if np.any(a["vol_rate"][sl] > 2e-6) or (np.max(T_s) >= 200.0 and np.any(a["vol_rate"][sl] > 1e-7)):
            self.stage = CombustionStage.PYROLYSIS
            return
        if np.any(a["steam_rate"][sl] > 1e-6) and np.max(T_s) >= 60.0:
            self.stage = CombustionStage.DEHYDRATION
            return
        if np.any((a["char_rate"][sl] > 1e-7) & (T_s >= 300.0)):
            self.stage = CombustionStage.CHAR_OXIDATION
            return
        self.stage = CombustionStage.AMBIENT

    # ------------------------------------------------------------------ serialisation helpers
    def segment_states(self) -> Dict[str, np.ndarray]:
        a = self.store.a
        sl = slice(self.i0, self.i1)
        present = a["present"][sl] > 0.5
        pool = a["pool"][sl].sum(axis=2)
        char = a["char"][sl]
        solids = pool + char
        o = np.argmax(present, axis=1)
        rows = np.arange(self.n_segments)
        char_frac = np.where(present.any(axis=1),
                             char[rows, o] / np.maximum(solids[rows, o], 1e-12), 1.0)
        charred_depth = 1.0 - (pool.sum(axis=1) / np.maximum(a["dry0"][sl].sum(axis=1), 1e-12))
        return {
            "T_surf": a["T_surf"][sl].copy(),
            "phi": a["phi"][sl].copy(),
            "char_frac": np.clip(char_frac, 0.0, 1.0),
            "charred": np.clip(charred_depth, 0.0, 1.0),
            "hrr": (a["hrr_flame"][sl] + a["hrr_char"][sl]).copy(),
            "hrr_flame": a["hrr_flame"][sl].copy(),
            "steam": a["steam_rate"][sl].copy(),
            "smoke": a["smoke_rate"][sl].copy(),
            "radius": self.segment_radii(),
            "ash_kg": a["ash_kg"][sl].copy(),
            "alive": present.any(axis=1),
            "moisture": np.where(present.any(axis=1),
                                 a["water"][sl][rows, o] / np.maximum(solids[rows, o], 1e-9), 0.0),
        }

    def get_render_state(self) -> Dict[str, Any]:
        """Provides visual rendering state."""
        return {
            "position": self.position.copy(),
            "orientation": self.orientation.copy(),
            "dimensions": self.dimensions.copy(),
            "species": self.species_type.value,
            "stage": self.stage.value,
            "t_surface": self.t_surface,
            "t_internal": self.t_internal,
            "m_surface": self.m_surface,
            "m_core": self.m_core,
            "flame_height": self.flame_height,
            "steam_rate": self.steam_rate,
            "smoke_rate": self.smoke_rate,
            "char_fraction": max(0.0, min(1.0, self.burned_fraction)),
            "ash_depth": self.ash_insulation_depth,
            "heat_output": self.heat_flux_output,
            "segments": self.segment_states(),
        }

    def describe(self) -> str:
        return (f"{self.props.species.value} {self.geometry_type} | MC {self.m_surface*100:.0f}% surf / "
                f"{self.m_core*100:.0f}% core | {self.stage.value}")
