"""
Firestarter - Smoldering Fungi (Amadou & Chaga)
Implements DESIGN Section 3.4:
Birch Polypore (Piptoporus betulinus) and Tinder Fungus (Fomes fomentarius) harvested from
standing dead timber.  When dry, the trama layer smoulders at 0.5-1.5 cm/hour without flame,
carrying an ember across miles and hours.
"""

from enum import Enum
from typing import Dict, Any, Optional


class FungusType(Enum):
    BIRCH_POLYPORE = "Birch Polypore (Piptoporus betulinus)"
    TINDER_FUNGUS = "Tinder Fungus (Fomes fomentarius)"
    CHAGA = "Chaga (Inonotus obliquus)"


# smoulder front speed in cm/hour for dry fungus (DESIGN: 0.5 - 1.5 cm/h)
SMOLDER_RATE_CM_H = {
    FungusType.BIRCH_POLYPORE: 1.3,
    FungusType.TINDER_FUNGUS: 0.8,
    FungusType.CHAGA: 0.6,
}


class SmolderingFungus:
    def __init__(
        self,
        fungus_type: FungusType = FungusType.TINDER_FUNGUS,
        diameter_cm: float = 8.0,
        moisture: float = 0.08,
        is_smoldering: bool = False,
        smolder_temp_c: float = 350.0
    ):
        self.fungus_type = fungus_type
        self.diameter_cm = diameter_cm
        self.initial_diameter_cm = diameter_cm
        self.moisture = moisture
        self.is_smoldering = is_smoldering
        self.core_temp = smolder_temp_c if is_smoldering else 20.0
        self.smoke_rate = 0.0

    @property
    def burn_rate_cm_per_sec(self) -> float:
        base = SMOLDER_RATE_CM_H[self.fungus_type]
        # damp trama smoulders slower (and goes out when too wet)
        return base * max(0.4, 1.0 - 2.0 * max(0.0, self.moisture - 0.08)) / 3600.0

    def catch_spark_or_ember(self, contact_temp_c: float) -> bool:
        """Dry amadou / chaga catches even faint sparks."""
        if self.moisture > 0.20:
            return False
        if contact_temp_c >= 250.0:
            self.is_smoldering = True
            self.core_temp = max(320.0, contact_temp_c * 0.8)
            return True
        return False

    def update(self, dt: float, ambient_temp: float = 20.0, rain: float = 0.0):
        if not self.is_smoldering:
            self.smoke_rate = 0.0
            return
        if rain > 0.5 and self.moisture < 1.0:
            self.moisture += 0.0005 * rain * dt
        if self.moisture > 0.45:
            self.is_smoldering = False
            self.core_temp = ambient_temp
            return
        self.diameter_cm -= self.burn_rate_cm_per_sec * dt
        self.smoke_rate = 0.05
        self.core_temp = max(300.0, min(450.0, self.core_temp))
        if self.diameter_cm <= 0.2:
            self.is_smoldering = False
            self.diameter_cm = 0.0
            self.core_temp = ambient_temp

    def transfer_ember_to_tinder(self) -> Optional[float]:
        """Touches the glowing trama to a tinder nest."""
        if self.is_smoldering and self.diameter_cm > 0.5:
            return self.core_temp
        return None

    @property
    def remaining_hours(self) -> float:
        rate = self.burn_rate_cm_per_sec
        if rate <= 0:
            return 0.0
        return (self.diameter_cm / rate) / 3600.0

    def get_state(self) -> Dict[str, Any]:
        return {
            "type": self.fungus_type.value,
            "diameter_cm": self.diameter_cm,
            "is_smoldering": self.is_smoldering,
            "core_temp": self.core_temp,
            "moisture": self.moisture,
            "remaining_hours": self.remaining_hours,
        }
