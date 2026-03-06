"""
Trading Executor (mFRR + XBID + FreeBid)

I handle the voluntary/semi-voluntary trading markets:
- mFRR: TSO-activated, pay-as-cleared, mandatory when accepted
- XBID: Continuous intraday trading, pay-as-bid
- FreeBid: Agent-submitted balancing bids, pay-as-bid, merit-order activation

These are the last markets in the cascade: DAM → aFRR → IDA → mFRR → XBID → FreeBid
"""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from gym_envs.market_executors import MarketResult

if TYPE_CHECKING:
    from gym_envs.battery_env_unified import BatteryEnvUnified


class TradingExecutor:
    """I execute mFRR, XBID, and FreeBid trading operations.

    Each market uses remaining capacity after all higher-priority markets.
    """

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute_mfrr(self, mfrr_qty_action: int, remaining_after_ida: float,
                     row: pd.Series) -> MarketResult:
        """I execute mFRR trading (TSO-activated, pay-as-cleared).

        Args:
            mfrr_qty_action: Index into MFRR_LEVELS
            remaining_after_ida: MW available after DAM/aFRR/IDA
            row: Current market data

        Returns:
            MarketResult with mFRR execution details
        """
        env = self._env
        result = MarketResult()

        mfrr_level = env.MFRR_LEVELS[mfrr_qty_action]
        requested_mfrr = mfrr_level * remaining_after_ida

        # I determine mFRR activation from real market data
        has_activation_data = 'mfrr_activated_up_mwh' in row.index if hasattr(row, 'index') else 'mfrr_activated_up_mwh' in row
        if has_activation_data:
            total_up = row.get('mfrr_activated_up_mwh', 0.0)
            total_down = row.get('mfrr_activated_down_mwh', 0.0)

            if requested_mfrr > 0:
                if total_up <= 0:
                    requested_mfrr = 0.0
                elif np.random.random() > env.mfrr_activation_rate:
                    requested_mfrr = 0.0
            elif requested_mfrr < 0:
                if total_down <= 0:
                    requested_mfrr = 0.0
                elif np.random.random() > env.mfrr_activation_rate:
                    requested_mfrr = 0.0
        else:
            imbalance = row.get('net_imbalance_mw_lag_1h',
                                row.get('system_deviation_mwh_lag_1h',
                                        row.get('net_imbalance_mw', 0.0)))
            if abs(imbalance) <= env.mfrr_imbalance_threshold:
                requested_mfrr = 0.0
            elif imbalance < -env.mfrr_imbalance_threshold and requested_mfrr < 0:
                requested_mfrr = 0.0
            elif imbalance > env.mfrr_imbalance_threshold and requested_mfrr > 0:
                requested_mfrr = 0.0

            if abs(requested_mfrr) > 0.1:
                activation_prob = env.mfrr_activation_rate
                abs_imbalance = abs(imbalance)
                if abs_imbalance > 500:
                    activation_prob = min(0.9, activation_prob * 2.0)
                elif abs_imbalance > 200:
                    activation_prob = min(0.8, activation_prob * 1.5)
                if np.random.random() > activation_prob:
                    requested_mfrr = 0.0

        if abs(requested_mfrr) <= 0.1:
            return result

        # I recalculate limits after IDA
        available_discharge_mwh = (env.soc - env.min_soc) * env.capacity_mwh
        available_charge_mwh = (env.max_soc - env.soc) * env.capacity_mwh

        max_discharge = min(
            remaining_after_ida,
            available_discharge_mwh * env.eff_sqrt / env.time_step_hours
        )
        max_charge = min(
            remaining_after_ida,
            available_charge_mwh / env.eff_sqrt / env.time_step_hours
        )

        if requested_mfrr > 0:
            actual_mfrr = min(requested_mfrr, max_discharge)
        else:
            actual_mfrr = max(requested_mfrr, -max_charge)

        if abs(actual_mfrr) <= 0.1:
            return result

        if actual_mfrr > 0:  # Sell (discharge) — pay-as-cleared
            energy_mwh = actual_mfrr * env.time_step_hours
            soc_delta = energy_mwh / env.eff_sqrt / env.capacity_mwh
            old_soc = env.soc
            env.soc = max(env.min_soc, env.soc - soc_delta)
            actual_energy = (old_soc - env.soc) * env.capacity_mwh * env.eff_sqrt

            mfrr_price = min(row.get('mfrr_price_up', 120.0), env.mfrr_price_cap)
            result.revenue = actual_energy * mfrr_price
            result.mwh_sold = abs(actual_mfrr) * env.time_step_hours
            result.direction = 'up'

        else:  # DOWN regulation (charge)
            energy_mwh = abs(actual_mfrr) * env.time_step_hours
            soc_delta = energy_mwh * env.eff_sqrt / env.capacity_mwh
            old_soc = env.soc
            env.soc = min(env.max_soc, env.soc + soc_delta)
            actual_energy = (env.soc - old_soc) * env.capacity_mwh / env.eff_sqrt

            mfrr_price = min(row.get('mfrr_price_down', 60.0), env.mfrr_price_cap)
            result.revenue = actual_energy * mfrr_price
            result.mwh_bought = abs(actual_mfrr) * env.time_step_hours
            result.direction = 'down'

        cycle_fraction = actual_energy / env.capacity_mwh
        env.total_cycles += cycle_fraction
        # I do NOT increment daily_cycles for mFRR — it's TSO-mandated

        result.energy_mw = actual_mfrr
        result.cycle_fraction = cycle_fraction
        result.capacity_used_mw = abs(actual_mfrr)
        result.is_activated = True

        env.mfrr_profit += result.revenue

        # I track mFRR energy volumes
        if actual_mfrr > 0:
            env.mfrr_mwh_sold += result.mwh_sold
        else:
            env.mfrr_mwh_bought += result.mwh_bought

        return result

    def execute_intraday(self, xbid_action: int, remaining_after_mfrr: float,
                         row: pd.Series) -> MarketResult:
        """I execute XBID/IntraDay continuous trading.

        Args:
            xbid_action: Index into INTRADAY_LEVELS
            remaining_after_mfrr: MW available after DAM/aFRR/IDA/mFRR
            row: Current market data

        Returns:
            MarketResult with IntraDay/XBID execution details
        """
        env = self._env
        result = MarketResult()

        active_market = env._get_intraday_market(row)
        intraday_open = env._is_intraday_open(row)

        if not (intraday_open and active_market != 'closed'):
            return result

        intraday_level = env.INTRADAY_LEVELS[xbid_action]
        requested_intraday = intraday_level * remaining_after_mfrr

        if abs(requested_intraday) <= 0.1:
            return result

        available_discharge_mwh = (env.soc - env.min_soc) * env.capacity_mwh
        available_charge_mwh = (env.max_soc - env.soc) * env.capacity_mwh

        max_discharge = min(
            remaining_after_mfrr,
            available_discharge_mwh * env.eff_sqrt / env.time_step_hours
        )
        max_charge = min(
            remaining_after_mfrr,
            available_charge_mwh / env.eff_sqrt / env.time_step_hours
        )

        if requested_intraday > 0:
            actual_intraday = min(requested_intraday, max_discharge)
        else:
            actual_intraday = max(requested_intraday, -max_charge)

        if abs(actual_intraday) <= 0.1:
            return result

        if actual_intraday > 0:  # Sell (discharge)
            energy_mwh = actual_intraday * env.time_step_hours
            soc_delta = energy_mwh / env.eff_sqrt / env.capacity_mwh
            old_soc = env.soc
            env.soc = max(env.min_soc, env.soc - soc_delta)
            actual_energy = (old_soc - env.soc) * env.capacity_mwh * env.eff_sqrt

            id_price = env._get_intraday_sell_price(active_market, row)
            result.revenue = actual_energy * id_price
            result.mwh_sold = abs(actual_intraday) * env.time_step_hours
            result.direction = 'up'

        else:  # Buy (charge)
            energy_mwh = abs(actual_intraday) * env.time_step_hours
            soc_delta = energy_mwh * env.eff_sqrt / env.capacity_mwh
            old_soc = env.soc
            env.soc = min(env.max_soc, env.soc + soc_delta)
            actual_energy = (env.soc - old_soc) * env.capacity_mwh / env.eff_sqrt

            id_price = env._get_intraday_buy_price(active_market, row)
            result.revenue = -actual_energy * id_price
            result.mwh_bought = abs(actual_intraday) * env.time_step_hours
            result.direction = 'down'

        cycle_fraction = actual_energy / env.capacity_mwh
        env.total_cycles += cycle_fraction
        env.daily_cycles += cycle_fraction

        result.energy_mw = actual_intraday
        result.cycle_fraction = cycle_fraction
        result.capacity_used_mw = abs(actual_intraday)
        result.is_activated = True

        # I track XBID vs IDA profit separately
        if active_market == 'isp1':
            result.metadata['xbid_energy_mw'] = actual_intraday
            env.xbid_profit += result.revenue
        else:
            result.metadata['xbid_energy_mw'] = 0.0
        env.intraday_profit += result.revenue

        # I track IntraDay and XBID energy volumes
        if actual_intraday > 0:
            env.intraday_mwh_sold += result.mwh_sold
            if active_market == 'isp1':
                env.xbid_mwh_sold += result.mwh_sold
        else:
            env.intraday_mwh_bought += result.mwh_bought
            if active_market == 'isp1':
                env.xbid_mwh_bought += result.mwh_bought

        result.metadata['active_market'] = active_market

        return result

    def execute_free_bid(self, freebid_qty_action: int, freebid_price_action: int,
                         remaining_after_intraday: float, row: pd.Series) -> MarketResult:
        """I execute Free Bid trading (agent-submitted, pay-as-bid, merit-order).

        Args:
            freebid_qty_action: Index into FREEBID_QTY_LEVELS
            freebid_price_action: Index into FREEBID_PRICE_TIERS
            remaining_after_intraday: MW available after all prior markets
            row: Current market data

        Returns:
            MarketResult with FreeBid execution details
        """
        env = self._env
        result = MarketResult()

        if not env.enable_full_market:
            return result

        freebid_level = env.FREEBID_QTY_LEVELS[freebid_qty_action]
        requested_freebid = freebid_level * remaining_after_intraday

        if abs(requested_freebid) <= 0.1:
            return result

        available_discharge_mwh = (env.soc - env.min_soc) * env.capacity_mwh
        available_charge_mwh = (env.max_soc - env.soc) * env.capacity_mwh

        max_discharge = min(
            remaining_after_intraday,
            available_discharge_mwh * env.eff_sqrt / env.time_step_hours
        )
        max_charge = min(
            remaining_after_intraday,
            available_charge_mwh / env.eff_sqrt / env.time_step_hours
        )

        if requested_freebid > 0:
            actual_freebid = min(requested_freebid, max_discharge)
        else:
            actual_freebid = max(requested_freebid, -max_charge)

        if abs(actual_freebid) <= 0.1:
            return result

        # I set the agent's bid price
        imbalance_fb = row.get('net_imbalance_mw_lag_1h',
                               row.get('net_imbalance_mw', 0.0))
        price_tier = env.FREEBID_PRICE_TIERS[freebid_price_action]
        ref_price = row.get('free_bid_reference_price',
                            row.get('mfrr_price_up', 100.0))
        agent_bid_price = ref_price * price_tier
        env.free_bid_submissions += 1

        result.metadata['agent_bid_price'] = agent_bid_price

        # I simulate merit-order activation
        free_bid_activated = env._simulate_free_bid_activation(
            agent_bid_price, ref_price, imbalance_fb, row
        )

        if not free_bid_activated:
            return result

        if actual_freebid > 0:  # Sell (discharge) — pay-as-bid
            energy_mwh = actual_freebid * env.time_step_hours
            soc_delta = energy_mwh / env.eff_sqrt / env.capacity_mwh
            old_soc = env.soc
            env.soc = max(env.min_soc, env.soc - soc_delta)
            actual_energy = (old_soc - env.soc) * env.capacity_mwh * env.eff_sqrt

            result.revenue = actual_energy * agent_bid_price
            result.mwh_sold = abs(actual_freebid) * env.time_step_hours
            result.direction = 'up'

        else:  # Buy (charge)
            energy_mwh = abs(actual_freebid) * env.time_step_hours
            soc_delta = energy_mwh * env.eff_sqrt / env.capacity_mwh
            old_soc = env.soc
            env.soc = min(env.max_soc, env.soc + soc_delta)
            actual_energy = (env.soc - old_soc) * env.capacity_mwh / env.eff_sqrt

            result.revenue = actual_energy * agent_bid_price
            result.mwh_bought = abs(actual_freebid) * env.time_step_hours
            result.direction = 'down'

        cycle_fraction = actual_energy / env.capacity_mwh
        env.total_cycles += cycle_fraction
        env.daily_cycles += cycle_fraction

        result.energy_mw = actual_freebid
        result.cycle_fraction = cycle_fraction
        result.capacity_used_mw = abs(actual_freebid)
        result.is_activated = True

        env.free_bid_profit += result.revenue
        env.free_bid_activations += 1

        # I track Free Bid energy volumes
        if actual_freebid > 0:
            env.free_bid_mwh_sold += result.mwh_sold
        else:
            env.free_bid_mwh_bought += result.mwh_bought

        return result
