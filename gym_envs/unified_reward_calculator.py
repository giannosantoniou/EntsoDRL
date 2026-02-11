"""
Unified Multi-Market Reward Calculator

I calculate rewards for the unified multi-market environment that handles:
- DAM commitment compliance
- IntraDay energy trading
- aFRR capacity payments and energy activation
- mFRR energy trading

The reward formula ensures proper incentives for all market activities while
maintaining strict compliance with HEnEx regulatory requirements.

Total Reward = DAM_Revenue + IntraDay_Revenue + aFRR_Capacity + aFRR_Energy + mFRR_Revenue
               - Degradation - DAM_Violation_Penalty - aFRR_NonResponse_Penalty
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class UnifiedMarketState:
    """I hold all market prices and states for reward calculation."""
    # DAM
    dam_price: float
    dam_commitment: float  # MW (positive = sell, negative = buy)

    # IntraDay
    intraday_bid: float  # Best bid price
    intraday_ask: float  # Best ask price
    intraday_spread: float

    # aFRR
    afrr_cap_up_price: float  # EUR/MW/h for upward capacity
    afrr_cap_down_price: float  # EUR/MW/h for downward capacity
    afrr_energy_up_price: float  # EUR/MWh when activated up
    afrr_energy_down_price: float  # EUR/MWh when activated down
    afrr_activated: bool
    afrr_activation_direction: Optional[str]  # 'up' or 'down'

    # mFRR
    mfrr_price_up: float  # EUR/MWh for upward energy
    mfrr_price_down: float  # EUR/MWh for downward energy


class UnifiedRewardCalculator:
    """
    I calculate rewards for the unified multi-market battery environment.

    Key Design Decisions:
    1. DAM violations have SEVERE penalties (1500 EUR/MWh) - regulatory requirement
    2. aFRR non-response has EVEN HIGHER penalties (2000 EUR/MWh) - TSO requirement
    3. Degradation is proportional to energy cycled (15 EUR/MWh default)
    4. aFRR capacity payments are unconditional when selected
    5. All revenues scale with actual delivered energy
    """

    def __init__(
        self,
        # Degradation
        degradation_cost_per_mwh: float = 15.0,

        # Penalty coefficients
        dam_violation_penalty: float = 1500.0,  # EUR/MWh - HEnEx imbalance penalty
        afrr_nonresponse_penalty: float = 2000.0,  # EUR/MWh - TSO penalty (CRITICAL)
        physical_violation_penalty: float = 100.0,  # EUR for physically impossible actions

        # Bonuses
        dam_compliance_bonus: float = 5.0,  # EUR/MWh for perfect DAM compliance
        afrr_response_bonus: float = 10.0,  # EUR/MWh for fast aFRR response
        optimal_timing_bonus: float = 2.0,  # EUR/MWh for good trade timing

        # Risk penalties
        shortfall_risk_penalty: float = 50.0,  # Proactive penalty for risky SoC
        soc_buffer_bonus: float = 2.0,  # Bonus for maintaining healthy SoC

        # Scaling
        reward_scale: float = 0.001  # Scale rewards for NN training
    ):
        self.degradation_cost = degradation_cost_per_mwh
        self.dam_violation_penalty = dam_violation_penalty
        self.afrr_nonresponse_penalty = afrr_nonresponse_penalty
        self.physical_violation_penalty = physical_violation_penalty
        self.dam_compliance_bonus = dam_compliance_bonus
        self.afrr_response_bonus = afrr_response_bonus
        self.optimal_timing_bonus = optimal_timing_bonus
        self.shortfall_risk_penalty = shortfall_risk_penalty
        self.soc_buffer_bonus = soc_buffer_bonus
        self.reward_scale = reward_scale

    def calculate(
        self,
        market: UnifiedMarketState,
        # Actions taken
        actual_energy_mw: float,  # Actual energy traded (positive = discharge)
        afrr_capacity_committed_mw: float,  # aFRR capacity commitment
        afrr_energy_delivered_mw: float,  # aFRR energy when activated
        mfrr_energy_mw: float,  # mFRR energy traded

        # Battery state
        current_soc: float,
        capacity_mwh: float,
        time_step_hours: float = 1.0,

        # Context
        is_physical_violation: bool = False,
        shortfall_risk: float = 0.0,
        price_momentum: float = 0.0,  # Backward-looking price trend
        is_selected_for_afrr: bool = True
    ) -> Dict:
        """
        I calculate the total reward for a unified multi-market step.

        Returns a dict with:
        - reward: scaled total reward
        - components: detailed breakdown of each component
        """
        components = {}

        # =====================================================================
        # 1. DAM SETTLEMENT
        # =====================================================================
        # I calculate DAM revenue/cost based on commitment
        dam_commitment = market.dam_commitment
        dam_revenue = dam_commitment * market.dam_price * time_step_hours
        components['dam_revenue'] = dam_revenue

        # I calculate DAM imbalance settlement
        # Imbalance = actual - commitment
        dam_imbalance_mw = actual_energy_mw - dam_commitment
        if dam_imbalance_mw > 0:
            # Delivered MORE than committed - get bid price
            dam_balancing = dam_imbalance_mw * market.intraday_bid * time_step_hours
        else:
            # Delivered LESS than committed - pay ask price
            dam_balancing = dam_imbalance_mw * market.intraday_ask * time_step_hours
        components['dam_balancing'] = dam_balancing

        # I apply DAM violation penalty for non-compliance
        dam_shortfall_mw = 0.0
        dam_violation_cost = 0.0

        if dam_commitment > 0 and actual_energy_mw < dam_commitment:
            # Failed to deliver on SELL commitment
            dam_shortfall_mw = dam_commitment - actual_energy_mw
            dam_violation_cost = dam_shortfall_mw * self.dam_violation_penalty * time_step_hours
        elif dam_commitment < 0 and actual_energy_mw > dam_commitment:
            # Failed to honor BUY commitment
            dam_shortfall_mw = actual_energy_mw - dam_commitment
            dam_violation_cost = dam_shortfall_mw * self.dam_violation_penalty * time_step_hours

        components['dam_shortfall_mw'] = dam_shortfall_mw
        components['dam_violation_cost'] = dam_violation_cost

        # I add DAM compliance bonus for perfect execution
        dam_bonus = 0.0
        if abs(dam_commitment) > 0.1 and dam_shortfall_mw < 0.1:
            dam_bonus = abs(dam_commitment) * self.dam_compliance_bonus * time_step_hours
        components['dam_bonus'] = dam_bonus

        # =====================================================================
        # 2. aFRR CAPACITY REVENUE
        # =====================================================================
        # I calculate aFRR capacity payment (paid for availability)
        afrr_capacity_revenue = 0.0
        if is_selected_for_afrr and afrr_capacity_committed_mw > 0.1:
            # I get paid for committed capacity
            afrr_capacity_revenue = afrr_capacity_committed_mw * market.afrr_cap_up_price * time_step_hours
        components['afrr_capacity_revenue'] = afrr_capacity_revenue

        # =====================================================================
        # 3. aFRR ENERGY REVENUE (when activated)
        # =====================================================================
        afrr_energy_revenue = 0.0
        afrr_nonresponse_cost = 0.0

        if market.afrr_activated and is_selected_for_afrr and afrr_capacity_committed_mw > 0.1:
            # TSO activated aFRR - I MUST respond
            required_power = afrr_capacity_committed_mw

            if market.afrr_activation_direction == 'up':
                # Must discharge
                if afrr_energy_delivered_mw > 0.1:
                    # Good - I responded
                    afrr_energy_revenue = afrr_energy_delivered_mw * market.afrr_energy_up_price * time_step_hours
                    # I add response bonus for timely compliance
                    if afrr_energy_delivered_mw >= required_power * 0.9:
                        afrr_energy_revenue += required_power * self.afrr_response_bonus * time_step_hours
                else:
                    # BAD - I failed to respond (SEVERE penalty)
                    afrr_nonresponse_cost = required_power * self.afrr_nonresponse_penalty * time_step_hours

            elif market.afrr_activation_direction == 'down':
                # Must charge
                if afrr_energy_delivered_mw < -0.1:
                    # Good - I responded (delivered is negative for charging)
                    afrr_energy_revenue = abs(afrr_energy_delivered_mw) * market.afrr_energy_down_price * time_step_hours
                    if abs(afrr_energy_delivered_mw) >= required_power * 0.9:
                        afrr_energy_revenue += required_power * self.afrr_response_bonus * time_step_hours
                else:
                    # BAD - I failed to respond
                    afrr_nonresponse_cost = required_power * self.afrr_nonresponse_penalty * time_step_hours

        components['afrr_energy_revenue'] = afrr_energy_revenue
        components['afrr_nonresponse_cost'] = afrr_nonresponse_cost

        # =====================================================================
        # 4. mFRR ENERGY REVENUE
        # =====================================================================
        mfrr_revenue = 0.0
        if abs(mfrr_energy_mw) > 0.1:
            if mfrr_energy_mw > 0:
                # Discharging to mFRR (selling upward regulation)
                mfrr_revenue = mfrr_energy_mw * market.mfrr_price_up * time_step_hours
            else:
                # Charging from mFRR (buying downward regulation)
                mfrr_revenue = mfrr_energy_mw * market.mfrr_price_down * time_step_hours  # Negative * positive = negative cost

        components['mfrr_revenue'] = mfrr_revenue

        # =====================================================================
        # 5. DEGRADATION COST
        # =====================================================================
        # I calculate degradation based on total energy cycled
        total_energy_cycled = abs(actual_energy_mw) * time_step_hours
        cycle_fraction = total_energy_cycled / capacity_mwh
        degradation_cost = cycle_fraction * capacity_mwh * self.degradation_cost

        components['degradation_cost'] = degradation_cost
        components['cycle_fraction'] = cycle_fraction

        # =====================================================================
        # 6. PHYSICAL VIOLATION PENALTY
        # =====================================================================
        physical_penalty = self.physical_violation_penalty if is_physical_violation else 0.0
        components['physical_penalty'] = physical_penalty

        # =====================================================================
        # 7. SHORTFALL RISK PENALTY (proactive)
        # =====================================================================
        # I penalize risky SoC states before they become violations
        risk_penalty = 0.0
        if shortfall_risk > 0.2:
            risk_factor = (shortfall_risk - 0.2) / 0.8  # Normalize to [0, 1]
            risk_penalty = self.shortfall_risk_penalty * (risk_factor ** 1.5)  # Quadratic scaling
        components['risk_penalty'] = risk_penalty

        # =====================================================================
        # 8. SoC BUFFER BONUS
        # =====================================================================
        # I reward maintaining healthy SoC levels (flexible for commitments)
        soc_bonus = 0.0
        if 0.25 <= current_soc <= 0.75:
            # Ideal zone - full bonus
            soc_bonus = self.soc_buffer_bonus
        elif 0.15 <= current_soc < 0.25 or 0.75 < current_soc <= 0.85:
            # Acceptable zone - partial bonus
            soc_bonus = self.soc_buffer_bonus * 0.5
        else:
            # Extreme zone - penalty
            soc_bonus = -self.soc_buffer_bonus
        components['soc_bonus'] = soc_bonus

        # =====================================================================
        # 9. PRICE TIMING BONUS
        # =====================================================================
        # I reward trading with momentum (backward-looking)
        timing_bonus = 0.0
        if abs(price_momentum) > 0.05 and abs(actual_energy_mw) > 0.5:
            power_factor = abs(actual_energy_mw) / 30.0  # Normalize by max power

            if actual_energy_mw > 0 and price_momentum > 0:
                # Selling when prices rose - good timing
                timing_bonus = self.optimal_timing_bonus * power_factor * min(price_momentum, 0.3) * 100
            elif actual_energy_mw < 0 and price_momentum < 0:
                # Buying when prices fell - good timing
                timing_bonus = self.optimal_timing_bonus * power_factor * min(abs(price_momentum), 0.3) * 100

        components['timing_bonus'] = timing_bonus

        # =====================================================================
        # 10. TOTAL REWARD
        # =====================================================================
        total_revenue = (
            dam_revenue + dam_balancing + dam_bonus +
            afrr_capacity_revenue + afrr_energy_revenue +
            mfrr_revenue +
            soc_bonus + timing_bonus
        )

        total_costs = (
            degradation_cost +
            dam_violation_cost +
            afrr_nonresponse_cost +
            physical_penalty +
            risk_penalty
        )

        net_profit = total_revenue - total_costs
        components['net_profit'] = net_profit

        # I scale the reward for NN training
        scaled_reward = net_profit * self.reward_scale

        # I validate the reward
        if not np.isfinite(scaled_reward):
            print(f"WARNING: Non-finite reward detected: {scaled_reward}")
            print(f"Components: {components}")
            scaled_reward = 0.0

        return {
            'reward': scaled_reward,
            'components': components
        }

    def calculate_simple(
        self,
        dam_profit: float,
        afrr_capacity_profit: float,
        afrr_energy_profit: float,
        mfrr_profit: float,
        degradation_cost: float,
        penalties: float
    ) -> float:
        """
        I provide a simplified reward calculation for quick evaluation.

        This is useful for backtesting and analysis where full market state
        is not available.
        """
        total = (
            dam_profit +
            afrr_capacity_profit +
            afrr_energy_profit +
            mfrr_profit -
            degradation_cost -
            penalties
        )
        return total * self.reward_scale
