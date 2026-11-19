"""
Firestarter - Economy & Credit System
Implements DESIGN Section 5.1:

    Credits = Integral( k_heat * T_core + k_eff * (Energy Produced / Fuel Consumed) ) dt

Efficiency is measured against the heating value of dry wood (~18 MJ/kg): wet, smoky,
smouldering fires waste energy as steam and unburnt smoke and earn less.  Credits earned by
a burning fire accrue as *pending fire-mastery rewards* which are banked when the player
opens the smartphone (README tutorial step 5).
"""

from firestarter.config import CREDIT_K_HEAT, CREDIT_K_EFF


class CreditSystem:
    def __init__(self, starting_credits: float = 20.0):
        self.credits = starting_credits
        self.total_earned = starting_credits
        self.pending = 0.0
        self.current_burn_rate_score = 0.0
        self.session_best_temp = 0.0
        self.last_claim = 0.0

    def update(self, dt: float, core_temp_c: float, energy_rate_watts: float,
               fuel_consumption_rate_kg_s: float = 0.0, efficiency_ratio: float = None,
               bank_immediately: bool = False) -> float:
        """Integrates the credit rate over dt.  Returns credits earned this step."""
        if core_temp_c < 120.0 or energy_rate_watts < 300.0:
            self.current_burn_rate_score = 0.0
            return 0.0
        if efficiency_ratio is None:
            # legacy signature: energy per unit fuel in MJ/kg relative to 18 MJ/kg
            if fuel_consumption_rate_kg_s > 1e-9:
                efficiency_ratio = min(1.2, energy_rate_watts / (fuel_consumption_rate_kg_s * 18.0e6))
            else:
                efficiency_ratio = 0.5
        eff = max(0.0, min(1.2, efficiency_ratio))
        rate = CREDIT_K_HEAT * (core_temp_c / 100.0) + CREDIT_K_EFF * 5.0 * eff
        earned = rate * dt
        self.session_best_temp = max(self.session_best_temp, core_temp_c)
        if bank_immediately:
            self.credits += earned
        else:
            self.pending += earned
        self.total_earned += earned
        self.current_burn_rate_score = rate
        return earned

    def claim_pending(self) -> float:
        amt = self.pending
        self.credits += amt
        self.pending = 0.0
        self.last_claim = amt
        return amt

    def can_afford(self, amount: float) -> bool:
        return self.credits >= amount

    def spend(self, amount: float) -> bool:
        if self.can_afford(amount):
            self.credits -= amount
            return True
        return False
