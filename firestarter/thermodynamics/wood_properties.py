"""
Firestarter - Wood Species & Cellular Property Matrix
Implements Table 2.3 from the Technical Design Specification.

Values from the DESIGN table (density, resin/volatile content, caloric value) are kept
verbatim.  Two physically-motivated corrections / additions were made for real-world
accuracy:

* ``moisture_diffusion_coeff`` - the DESIGN table lists 0.6-8.5 x 10^-5 m^2/s, which is the
  diffusivity of water *vapour in open air*.  Bound-water diffusion across the grain of
  wood at 20 C is ~10^-10 m^2/s (USDA Wood Handbook, Ch. 4).  We keep the DESIGN's
  relative ordering (punk >> cedar > pine > birch > oak) but scale to physical units
  (x 10^-5).  With the DESIGN value a 12 cm log would equalise surface/core moisture in a
  few minutes, which would make the "surface-wet / core-dry" behaviour in DESIGN 2.2
  impossible.  The coefficient rises with temperature (Arrhenius, E = 38 kJ/mol).
* Chemical composition (hemicellulose / cellulose / lignin split), heat of gasification and
  flame-spread parameters, used by the finite-volume fuel kernel.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class WoodSpeciesType(Enum):
    WHITE_PINE = "White Pine"
    RED_CEDAR = "Red Cedar"
    PAPER_BIRCH = "Paper Birch"
    WHITE_OAK = "White Oak"
    PUNK_WOOD = "Punk Wood"


@dataclass(frozen=True)
class WoodProperties:
    species: WoodSpeciesType
    classification: str              # Softwood, Hardwood, Decayed Fibers
    bulk_density: float              # kg/m^3 (oven-dry)
    resin_content: float             # Fraction (0.0 to 1.0) - terpenes / oils / betulin
    charcoal_caloric_value: float    # J/kg
    moisture_diffusion_coeff: float  # m^2/s at 20 C (bound water, across grain)
    thermal_conductivity: float      # W/(m K) across grain, oven dry
    ash_fraction: float              # Fraction of dry mass left as mineral ash
    volatile_ignition_temp: float    # C  (piloted ignition surface temperature)
    char_yield_fraction: float       # Fraction of dry mass converted to solid char
    spark_pop_rate: float            # Pop rate modifier for ASMR resin bursts
    is_sponge: bool = False          # High moisture absorption for punk wood
    betulin_content: float = 0.0     # Birch bark betulin resin
    # --- added for the finite-volume fuel model ---
    composition: Tuple[float, float, float] = (0.30, 0.45, 0.25)  # hemicellulose, cellulose, lignin
    heat_of_gasification: float = 1.9e6   # J/kg (Tewarson effective L_g, dry)
    autoignition_temp: float = 470.0      # C  (unpiloted; DESIGN: > 450 C)
    char_reactivity: float = 1.0          # multiplier on char oxidation kinetics (porosity)
    bark_color: Tuple[float, float, float] = (0.36, 0.27, 0.19)
    wood_color: Tuple[float, float, float] = (0.78, 0.64, 0.44)

    @property
    def specific_gravity(self) -> float:
        return self.bulk_density / 1000.0


WOOD_MATRIX: Dict[WoodSpeciesType, WoodProperties] = {
    WoodSpeciesType.WHITE_PINE: WoodProperties(
        species=WoodSpeciesType.WHITE_PINE,
        classification="Softwood",
        bulk_density=380.0,
        resin_content=0.185,             # 18.5% (Terpenes)
        charcoal_caloric_value=18.2e6,   # 18.2 MJ/kg (Fast burn)
        moisture_diffusion_coeff=1.8e-10,
        thermal_conductivity=0.12,
        ash_fraction=0.015,
        volatile_ignition_temp=360.0,
        char_yield_fraction=0.22,
        spark_pop_rate=1.8,
        composition=(0.25, 0.44, 0.31),
        heat_of_gasification=1.8e6,
        autoignition_temp=460.0,
        bark_color=(0.34, 0.24, 0.17),
        wood_color=(0.86, 0.74, 0.52),
    ),
    WoodSpeciesType.RED_CEDAR: WoodProperties(
        species=WoodSpeciesType.RED_CEDAR,
        classification="Softwood",
        bulk_density=360.0,
        resin_content=0.220,             # 22.0% (Resinous oils)
        charcoal_caloric_value=17.8e6,   # 17.8 MJ/kg (Flaring)
        moisture_diffusion_coeff=2.1e-10,
        thermal_conductivity=0.11,
        ash_fraction=0.012,
        volatile_ignition_temp=340.0,
        char_yield_fraction=0.20,
        spark_pop_rate=2.2,
        composition=(0.24, 0.45, 0.31),
        heat_of_gasification=1.7e6,
        autoignition_temp=450.0,
        bark_color=(0.45, 0.25, 0.18),
        wood_color=(0.78, 0.50, 0.36),
    ),
    WoodSpeciesType.PAPER_BIRCH: WoodProperties(
        species=WoodSpeciesType.PAPER_BIRCH,
        classification="Hardwood",
        bulk_density=650.0,
        resin_content=0.140,             # 14.0% (Betulin in bark)
        charcoal_caloric_value=22.4e6,   # 22.4 MJ/kg (Balanced)
        moisture_diffusion_coeff=1.2e-10,
        thermal_conductivity=0.16,
        ash_fraction=0.020,
        volatile_ignition_temp=380.0,
        char_yield_fraction=0.28,
        spark_pop_rate=0.8,
        betulin_content=0.25,
        composition=(0.33, 0.45, 0.22),
        heat_of_gasification=2.0e6,
        autoignition_temp=470.0,
        bark_color=(0.90, 0.88, 0.82),
        wood_color=(0.88, 0.80, 0.64),
    ),
    WoodSpeciesType.WHITE_OAK: WoodProperties(
        species=WoodSpeciesType.WHITE_OAK,
        classification="Hardwood",
        bulk_density=780.0,
        resin_content=0.042,             # 4.2% (Low volatiles)
        charcoal_caloric_value=28.6e6,   # 28.6 MJ/kg (Enduring)
        moisture_diffusion_coeff=0.6e-10,
        thermal_conductivity=0.19,
        ash_fraction=0.035,
        volatile_ignition_temp=430.0,
        char_yield_fraction=0.35,
        spark_pop_rate=0.3,
        composition=(0.32, 0.45, 0.23),
        heat_of_gasification=2.3e6,
        autoignition_temp=490.0,
        bark_color=(0.30, 0.26, 0.21),
        wood_color=(0.74, 0.60, 0.42),
    ),
    WoodSpeciesType.PUNK_WOOD: WoodProperties(
        species=WoodSpeciesType.PUNK_WOOD,
        classification="Decayed Fibers",
        bulk_density=120.0,
        resin_content=0.010,             # 1.0% (Lignin-depleted)
        charcoal_caloric_value=12.0e6,   # 12.0 MJ/kg (Smolder only)
        moisture_diffusion_coeff=8.5e-10,
        thermal_conductivity=0.05,
        ash_fraction=0.050,
        volatile_ignition_temp=290.0,
        char_yield_fraction=0.15,
        spark_pop_rate=0.05,
        is_sponge=True,
        composition=(0.10, 0.20, 0.70),
        heat_of_gasification=2.6e6,      # porous, lignin-rich: poor flaming, excellent smoulder
        autoignition_temp=420.0,
        char_reactivity=6.0,             # huge internal surface area -> catches a faint spark
        bark_color=(0.42, 0.33, 0.24),
        wood_color=(0.62, 0.48, 0.33),
    ),
}


def get_wood_properties(species: WoodSpeciesType) -> WoodProperties:
    return WOOD_MATRIX[species]


def equilibrium_moisture_content(relative_humidity: float, temp_c: float = 20.0) -> float:
    """
    Hailwood-Horrobin single-hydrate EMC model (USDA Wood Handbook eq. 4-5).
    Returns dry-basis moisture fraction.  RH 0..1.
    """
    h = max(0.0, min(0.98, relative_humidity))
    T = temp_c * 9.0 / 5.0 + 32.0  # the handbook fit is in Fahrenheit
    W = 330.0 + 0.452 * T + 0.00415 * T * T
    k = 0.791 + 4.63e-4 * T - 8.44e-7 * T * T
    k1 = 6.34 + 7.75e-4 * T - 9.35e-5 * T * T
    k2 = 1.09 + 2.84e-2 * T - 9.04e-5 * T * T
    kh = k * h
    emc_pct = (1800.0 / W) * (kh / (1.0 - kh) + (k1 * kh + 2.0 * k1 * k2 * kh * kh) /
                               (1.0 + k1 * kh + k1 * k2 * kh * kh))
    return emc_pct / 100.0
