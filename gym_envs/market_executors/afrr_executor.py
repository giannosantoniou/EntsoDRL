"""
aFRR (automatic Frequency Restoration Reserve) Executor

I handle all aFRR-related logic:
- Capacity commitment and 4-hour block locking
- Selection probability simulation
- Activation detection and direction determination
- Energy delivery execution (mandatory when activated)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, TYPE_CHECKING

from gym_envs.market_executors import MarketResult

if TYPE_CHECKING:
    from gym_envs.battery_env_unified import BatteryEnvUnified


class AfrrExecutor:
    """I execute aFRR commitment, selection, and activation logic.

    aFRR is the second priority market after DAM. When selected and activated,
    response is MANDATORY — failure results in severe penalties.
    """

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def process_commitment(self, afrr_action: int, price_tier_action: int,
                           remaining_capacity: float):
        """I handle aFRR commitment and selection at block boundaries.

        Updates env state: afrr_commitment_mw, afrr_commitment_level,
        is_selected_for_afrr, _afrr_steps_remaining
        """
        env = self._env

        if env._afrr_steps_remaining <= 0:
            new_afrr_level = env.AFRR_LEVELS[afrr_action]
            env.afrr_commitment_level = afrr_action
            env.afrr_commitment_mw = new_afrr_level * remaining_capacity
            env._afrr_steps_remaining = env._afrr_block_steps

            price_tier = env.AFRR_PRICE_TIERS[price_tier_action]
            adjusted_selection_prob = env.selection_probability * (1.5 - price_tier * 0.5)
            adjusted_selection_prob = np.clip(adjusted_selection_prob, 0.1, 0.95)

            env.is_selected_for_afrr = (
                env.afrr_commitment_mw > 0.1 and
                np.random.random() < adjusted_selection_prob
            )

        env._afrr_steps_remaining -= 1

    def collect_capacity_revenue(self, row: pd.Series) -> float:
        """I collect aFRR capacity revenue if selected.

        Returns revenue in EUR.
        """
        env = self._env
        if env.is_selected_for_afrr:
            cap_price = row.get('afrr_cap_up_price', 20.0)
            revenue = env.afrr_commitment_mw * cap_price * env.time_step_hours
            env.afrr_capacity_profit += revenue
            return revenue
        return 0.0

    def check_activation(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """I simulate aFRR activation based on system conditions.

        Returns:
            (activated, direction): Whether activated and 'up' or 'down'
        """
        env = self._env

        if not (env.is_selected_for_afrr and env.afrr_commitment_mw > 0.1):
            return False, None

        activation_prob = env.afrr_activation_rate

        imbalance = abs(row.get('net_imbalance_mw', 0))
        if imbalance > 500:
            activation_prob *= env.high_imbalance_boost

        hour = env.df.index[env.current_step].hour
        if 17 <= hour <= 21:
            activation_prob *= env.peak_activation_boost

        if np.random.random() < activation_prob:
            afrr_up = row.get('afrr_up', 80)
            afrr_down = row.get('afrr_down', 80)

            imbalance_mw = row.get('net_imbalance_mw', 0)
            if imbalance_mw > 100:
                return True, 'up'
            elif imbalance_mw < -100:
                return True, 'down'
            else:
                if afrr_up > afrr_down:
                    return True, 'up'
                else:
                    return True, 'down'

        return False, None

    def execute_activation(self, direction: str, remaining_capacity: float,
                          row: pd.Series) -> MarketResult:
        """I execute aFRR energy delivery when activated.

        Args:
            direction: 'up' (discharge) or 'down' (charge)
            remaining_capacity: MW available after DAM
            row: Current market data

        Returns:
            MarketResult with aFRR execution details
        """
        env = self._env
        result = MarketResult()

        available_discharge_mwh = (env.soc - env.min_soc) * env.capacity_mwh
        available_charge_mwh = (env.max_soc - env.soc) * env.capacity_mwh

        max_discharge = min(
            remaining_capacity,
            available_discharge_mwh * env.eff_sqrt / env.time_step_hours
        )
        max_charge = min(
            remaining_capacity,
            available_charge_mwh / env.eff_sqrt / env.time_step_hours
        )

        # I store max deliverable for reward calculator's SoC starvation check
        if direction == 'up':
            result.metadata['max_deliverable_mw'] = max_discharge
        else:
            result.metadata['max_deliverable_mw'] = max_charge

        env.steps_since_afrr_activation = 0
        result.is_activated = True
        result.direction = direction

        if direction == 'up':
            afrr_energy_delivered = min(env.afrr_commitment_mw, max_discharge)
            if afrr_energy_delivered > 0.1:
                energy_mwh = afrr_energy_delivered * env.time_step_hours
                soc_delta = energy_mwh / env.eff_sqrt / env.capacity_mwh
                old_soc = env.soc
                env.soc = max(env.min_soc, env.soc - soc_delta)
                actual_energy = (old_soc - env.soc) * env.capacity_mwh * env.eff_sqrt

                # I apply ADMIE "highest value rule": max(aFRR, mFRR)
                afrr_price = max(row.get('afrr_up', 80.0), row.get('mfrr_price_up', 0.0))
                result.revenue = actual_energy * afrr_price

                cycle_fraction = actual_energy / env.capacity_mwh
                env.total_cycles += cycle_fraction
                env.daily_cycles += cycle_fraction

                afrr_energy_delivered = actual_energy / env.time_step_hours
                result.energy_mw = afrr_energy_delivered
                result.cycle_fraction = cycle_fraction
                result.mwh_sold = abs(afrr_energy_delivered) * env.time_step_hours

        else:  # 'down'
            afrr_energy_delivered = min(env.afrr_commitment_mw, max_charge)
            if afrr_energy_delivered > 0.1:
                energy_mwh = afrr_energy_delivered * env.time_step_hours
                soc_delta = energy_mwh * env.eff_sqrt / env.capacity_mwh
                old_soc = env.soc
                env.soc = min(env.max_soc, env.soc + soc_delta)
                actual_energy = (env.soc - old_soc) * env.capacity_mwh / env.eff_sqrt

                afrr_price = max(row.get('afrr_down', 80.0), row.get('mfrr_price_down', 0.0))
                result.revenue = actual_energy * afrr_price

                cycle_fraction = actual_energy / env.capacity_mwh
                env.total_cycles += cycle_fraction
                env.daily_cycles += cycle_fraction

                afrr_energy_delivered = -actual_energy / env.time_step_hours
                result.energy_mw = afrr_energy_delivered
                result.cycle_fraction = cycle_fraction
                result.mwh_bought = abs(afrr_energy_delivered) * env.time_step_hours

        env.afrr_energy_profit += result.revenue
        result.capacity_used_mw = abs(result.energy_mw)

        # I track aFRR energy volumes
        if result.energy_mw > 0:
            env.afrr_mwh_sold += result.mwh_sold
        elif result.energy_mw < 0:
            env.afrr_mwh_bought += result.mwh_bought

        return result
