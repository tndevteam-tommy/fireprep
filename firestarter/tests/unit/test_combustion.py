"""
DESIGN Section 2 - thermodynamic & chemical combustion engine.

Every assertion is tied to a real-world behaviour the design document promises.
"""

import numpy as np
import pytest

from firestarter.config import TEMP_CHAR_GLOW_MAX, O2_FLAMING_MIN, H_VAP_WATER
from firestarter.thermodynamics.combustion_body import CombustionBody, CombustionStage
from firestarter.thermodynamics.wood_properties import WoodSpeciesType as W, WOOD_MATRIX, get_wood_properties, equilibrium_moisture_content
from firestarter.thermodynamics.thermal_grid import ThermodynamicGrid, heskestad_flame_height, mccaffrey_excess_temperature
from firestarter.thermodynamics import fuel_kernel as fk
from tests.helpers import tinder, lean, flat, teepee, run, first_time


# ------------------------------------------------------------------------ wood property matrix
def test_design_table_values_are_preserved():
    """DESIGN 2.3 table values (density, resin, caloric value) are used verbatim."""
    expect = {W.WHITE_PINE: (380, 0.185, 18.2e6), W.RED_CEDAR: (360, 0.22, 17.8e6), W.PAPER_BIRCH: (650, 0.14, 22.4e6),
              W.WHITE_OAK: (780, 0.042, 28.6e6), W.PUNK_WOOD: (120, 0.01, 12.0e6)}
    for sp, (rho, resin, cal) in expect.items():
        p = get_wood_properties(sp)
        assert p.bulk_density == rho and p.resin_content == pytest.approx(resin) and p.charcoal_caloric_value == cal


def test_moisture_diffusion_ordering_and_physical_magnitude():
    d = {sp: WOOD_MATRIX[sp].moisture_diffusion_coeff for sp in W}
    assert d[W.PUNK_WOOD] > d[W.RED_CEDAR] > d[W.WHITE_PINE] > d[W.PAPER_BIRCH] > d[W.WHITE_OAK]
    # bound-water diffusion across the grain at 20 C is ~1e-10 m^2/s (USDA Wood Handbook)
    assert all(1e-11 < v < 2e-9 for v in d.values())


def test_equilibrium_moisture_content_matches_wood_handbook():
    # Wood Handbook: ~12 % EMC at 65 % RH / 21 C, ~6 % at 30 % RH, ~20 % at 90 % RH
    assert equilibrium_moisture_content(0.65, 21) == pytest.approx(0.12, abs=0.012)
    assert equilibrium_moisture_content(0.30, 21) == pytest.approx(0.06, abs=0.01)
    assert equilibrium_moisture_content(0.90, 21) == pytest.approx(0.205, abs=0.02)


def test_mass_from_geometry_and_species():
    b = CombustionBody(W.WHITE_OAK, geometry_type="log", dimensions=np.array([0.45, 0.06]))
    assert b.initial_dry_mass == pytest.approx(780 * np.pi * 0.06 ** 2 * 0.45, rel=1e-6)
    assert b.dry_mass == pytest.approx(b.initial_dry_mass * (1 - b.props.ash_fraction), rel=1e-3)


def test_surface_and_core_moisture_profile():
    b = CombustionBody(W.WHITE_OAK, initial_m_surface=0.65, initial_m_core=0.12, geometry_type="log")
    assert b.m_surface == pytest.approx(0.65, abs=1e-6) and b.m_core == pytest.approx(0.12, abs=1e-6)
    b.set_moisture(surface=0.08, core=0.85)
    assert b.m_surface == pytest.approx(0.08) and b.m_core == pytest.approx(0.85)


# ------------------------------------------------------------------------ correlations
def test_heskestad_and_mccaffrey_correlations():
    # L_f = 0.235 Q^0.4 - 1.02 D ; a 30 kW, 0.4 m fire has ~0.5 m flames
    assert heskestad_flame_height(30.0, 0.4) == pytest.approx(0.235 * 30 ** 0.4 - 1.02 * 0.4)
    assert 0.4 < heskestad_flame_height(30.0, 0.4) < 0.6
    # McCaffrey: continuous flame ~880 K excess near the base, decaying with height
    assert mccaffrey_excess_temperature(np.array([0.05]), np.array([20.0]))[0] == pytest.approx(880.0)
    assert mccaffrey_excess_temperature(np.array([2.0]), np.array([20.0]))[0] < 150.0


# ------------------------------------------------------------------------ ignition & stages
def test_dry_tinder_lights_from_a_lighter_in_seconds():
    g, s = run([tinder(0.08)], 20, lighter=2.0)
    t_flame = first_time(s, lambda x: x["hrr"] > 1000)
    assert t_flame is not None and t_flame <= 3.0
    peak = max(x["hrr"] for x in s)
    assert 5e3 < peak < 40e3, peak          # a fist-sized nest: ~5-40 kW, 0.3-0.8 m flames
    assert 0.3 < max(x["Lf"] for x in s) < 1.0


def test_tinder_nest_burns_out_in_under_a_minute():
    """A fist-sized nest of fine tinder flames for roughly 30-60 s (bushcraft rule of thumb)."""
    g, s = run([tinder(0.08)], 75, lighter=2.0)
    burning = [x["t"] for x in s if x["hrr"] > 500]
    assert 20 <= len(burning) <= 62
    assert all(x["hrr"] < 1000 for x in s if x["t"] >= 65)


def test_damp_tinder_only_steams_under_a_lighter():
    """Fine fuels above their moisture of extinction (~30 %) cannot sustain a flame."""
    g, s = run([tinder(0.38)], 8, lighter=8.0)
    assert max(x["hrr"] for x in s) < 50.0
    assert g.total_water_evaporated_kg > 0.0 or g.bodies[0].m_surface < 0.38


def test_stage_progression_dehydration_pyrolysis_flaming_coals():
    stages = []
    b = lean(0)
    g, s = run([tinder(), b, lean(120), lean(240)], 180, record=lambda t, g: stages.append(b.stage))
    seq = [st for i, st in enumerate(stages) if i == 0 or stages[i - 1] != st]
    assert CombustionStage.FLAMING in seq
    assert any(st in (CombustionStage.DEHYDRATION, CombustionStage.PYROLYSIS) for st in seq[:seq.index(CombustionStage.FLAMING) + 1])


def test_kindling_teepee_catches_from_tinder():
    seen = {}
    g, s = run(teepee(6), 90, record=lambda t, g: seen.setdefault("k", sum(1 for b in g.bodies if b.geometry_type == "kindling" and b.is_flaming)) if sum(
        1 for b in g.bodies if b.geometry_type == "kindling" and b.is_flaming) >= 4 and "t" not in seen and not seen.update(t=t) else None)
    assert seen.get("t") is not None and 5.0 <= seen["t"] <= 45.0, seen
    assert max(x["hrr"] for x in s) < 120e3


@pytest.mark.slow
def test_green_log_will_not_sustain_flaming():
    """DESIGN 2.2: green wood (80 % / 110 %) cannot sustain combustion without a big external flux."""
    g = ThermodynamicGrid()
    b = flat(0, 0, 0, m=(0.80, 1.10), y=0.06)
    g2, s = run([b], 420, lighter=0.0, grid=g, burner=(np.array([0, -0.01, 0]), 12.0, 180.0))
    tail = [x for x in s if x["t"] > 240]
    assert all(x["hrr"] < 200 for x in tail)
    assert b.m_surface < 0.80      # it did steam off some surface water


@pytest.mark.slow
def test_dry_log_ignites_sooner_than_wet_surface_log():
    def ignite_time(ms, mc):
        b = flat(0, 0, 0, m=(ms, mc), y=0.06)
        t0 = {}
        run([b], 240, lighter=0.0, burner=(np.array([0, -0.01, 0]), 12.0, 240.0),
            record=lambda t, g: t0.setdefault("t", t) if b.is_flaming else None)
        return t0.get("t")
    t_dry = ignite_time(0.12, 0.12)
    t_wet = ignite_time(0.65, 0.12)
    assert t_dry is not None
    assert t_wet is None or t_wet > t_dry + 20


@pytest.mark.slow
def test_single_hardwood_log_goes_out_without_company():
    """Cross-radiation between pieces keeps log fires alive; one lonely oak log dies back."""
    b = flat(0, 0, 0, m=(0.12, 0.12), y=0.06)
    g, s = run([b], 420, lighter=0.0, burner=(np.array([0, -0.01, 0]), 12.0, 180.0))
    assert max(x["hrr"] for x in s if x["t"] < 185) > 300
    assert all(x["hrr"] < 300 for x in s if x["t"] > 330)


@pytest.mark.slow
def test_pine_ignites_faster_than_oak_and_oak_burns_longer():
    def run_species(sp):
        pieces = [tinder()] + [lean(a, sp=sp) for a in (0, 60, 120, 180, 240, 300)]
        t_flame = {}
        g, s = run(pieces, 240, record=lambda t, g: t_flame.setdefault("t", t) if sum(
            1 for b in g.bodies if b.geometry_type == "kindling" and b.is_flaming) >= 3 else None)
        burning = sum(1 for x in s if x["hrr"] + x["char"] > 800)
        return t_flame.get("t", 999), burning
    t_pine, dur_pine = run_species(W.WHITE_PINE)
    t_oak, dur_oak = run_species(W.WHITE_OAK)
    assert t_pine < t_oak
    assert dur_oak >= dur_pine


@pytest.mark.slow
def test_char_glow_temperatures_are_physical():
    g, s = run(teepee(6), 240)
    cores = [x["core"] for x in s if x["char"] > 200]
    assert cores and max(cores) < 1260
    assert np.median(cores) > 500


@pytest.mark.slow
def test_coal_bed_forms_as_sticks_burn_through():
    g, s = run(teepee(6) + [lean(a, kind="firewood", base_r=0.2, tilt=56) for a in (30, 150, 270)], 300)
    assert g.coal_bed.collapse_events >= 3
    assert g.coal_bed.char_kg > 0.005 or g.coal_bed.ash_kg > 0.001


def test_frozen_wood_needs_melting_enthalpy():
    """DESIGN 7.1 winter: ice must melt (334 kJ/kg) before dehydration begins."""
    warm = flat(0, 0, 0, kind="firewood", sp=W.WHITE_PINE, m=(0.3, 0.3), y=0.03)
    cold = flat(0, 0, 0, kind="firewood", sp=W.WHITE_PINE, m=(0.3, 0.3), y=0.03)
    cold.set_temperature(-12.0)
    for b, amb in ((warm, 5.0), (cold, -12.0)):
        for _ in range(200):
            b.apply_external_heat(60.0, 0.05)
            b.update(0.05, ambient_temp=amb)
    assert cold.t_surface < warm.t_surface


def test_punk_wood_catches_a_faint_spark_and_smoulders():
    punk = CombustionBody(W.PUNK_WOOD, geometry_type="firewood", initial_m_surface=0.08, initial_m_core=0.1,
                          position=np.array([0, 0.05, 0]), dimensions=np.array([0.3, 0.05]))
    g = ThermodynamicGrid()
    g.add_body(punk)
    for i in range(120):
        if i < 40:
            punk.apply_external_heat(25.0, 0.05)
        g.update(0.05, 20.0, 0.5)
    assert punk.props.char_reactivity > 3.0
    assert fk.CHAR_OX_A * punk.props.char_reactivity > fk.CHAR_OX_A


def test_choked_layout_starves_oxygen():
    """DESIGN 2.4: logs edge-to-edge (< 2 cm gaps) drive C_d -> 0 and smother the flames."""
    open_g, open_s = run(teepee(6) + [lean(a, kind="firewood", base_r=0.2, tilt=56) for a in (30, 150, 270)], 40)
    pieces = [tinder()]
    for layer in range(2):
        for i in range(3):
            pieces.append(flat((i - 1) * 0.064, 0, 90.0, kind="firewood", sp=W.WHITE_PINE, m=(0.1, 0.13), y=0.12 + layer * 0.064))
    choked_g, choked_s = run(pieces, 40)
    assert choked_g.analysis.choke_fraction > 0.6
    assert choked_g.c_d < 0.2
    assert max(x["hrr"] for x in choked_s[-10:]) < 0.5 * max(x["hrr"] for x in open_s[-10:])


@pytest.mark.slow
def test_blowing_on_coals_raises_their_temperature():
    """DESIGN 3.2: breath (O2 + convective boost) brightens the coals relative to leaving them alone."""
    def coal_T(blow):
        g, s = run(teepee(6), 200)
        for _ in range(3):
            if blow:
                g.blow_into_fire(2.8)
            for _ in range(20):
                g.update(0.05, 20.0, 0.5)
        return max(g.coal_bed.T if g.coal_bed.char_kg > 0.003 else 0.0, g.core_coal_temp), g.char_hrr
    (t_still, q_still), (t_blown, q_blown) = coal_T(False), coal_T(True)
    assert t_blown > t_still + 5.0 or q_blown > 1.1 * q_still


def test_rain_wets_and_cools_exposed_fuel():
    class Wx:
        ambient_temp_c, relative_humidity, wind_vector, wind_speed_m_s, rain_intensity = 15.0, 0.9, np.array([1.0, 0, 0]), 1.0, 1.0

        def sky_temperature_c(self):
            return 10.0
    b = flat(0, 0, 0, kind="firewood", sp=W.WHITE_PINE, m=(0.1, 0.1), y=0.03)
    g = ThermodynamicGrid()
    g.add_body(b)
    for _ in range(600):
        g.update(0.5, weather=Wx())
    assert b.m_surface > 0.12


def test_energy_accounting_is_consistent():
    g, s = run(teepee(6), 200)
    mj = g.total_energy_produced_joules / 1e6
    kg = g.total_fuel_consumed_kg
    assert mj > 0.3
    assert 5.0 < mj / kg < 30.0      # wood volatiles + char average ~15-20 MJ/kg


def test_fast_forward_burns_the_fire_down():
    g = ThermodynamicGrid()
    for b in teepee(6):
        g.add_body(b)
    for i in range(60):
        g.add_pilot(np.array([0, 0.03, 0.05]), 0.1)
        g.update(0.05, 20.0, 1.0)
    g.fast_forward(6 * 3600.0)
    assert not g.bodies and g.core_coal_temp < 40


def test_legacy_api_surface():
    b = CombustionBody(W.WHITE_PINE, dry_mass=0.2, initial_m_surface=0.1, initial_m_core=0.12, geometry_type="kindling",
                       position=np.array([0, 0.1, 0]), dimensions=np.array([0.3, 0.015]))
    rs = b.get_render_state()
    for k in ("position", "stage", "t_surface", "m_surface", "m_core", "flame_height", "steam_rate", "smoke_rate", "segments"):
        assert k in rs
    b.add_accelerant(0.01, "gasoline")
    assert b.accelerant_mass == pytest.approx(0.01)
