"""
DAM (Day-Ahead Market) Executor

I handle all DAM-related logic:
- DAM commitment lookup (endogenous or CSV-based)
- DAM schedule generation (endogenous, SoC-aware)
- DAM execution (mandatory, priority 1 in cascade)
- DAM SoC reserve calculation
- DAM shortfall risk assessment
- DAM SoC prediction (end-of-day)
"""

import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

from gym_envs.market_executors import MarketResult

if TYPE_CHECKING:
    from gym_envs.battery_env_unified import BatteryEnvUnified


class DamExecutor:
    """I execute DAM commitments and manage DAM-related state.

    DAM is the highest priority market in the cascade. All other markets
    operate within the capacity left over after DAM execution.
    """

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def execute(self, dam_commitment: float) -> MarketResult:
        """I execute the mandatory DAM commitment for the current step.

        Args:
            dam_commitment: DAM commitment MW (positive=sell, negative=buy)

        Returns:
            MarketResult with DAM execution details
        """
        env = self._env
        result = MarketResult()

        if abs(dam_commitment) <= 0.1:
            return result

        available_discharge_mwh = (env.soc - env.min_soc) * env.capacity_mwh
        available_charge_mwh = (env.max_soc - env.soc) * env.capacity_mwh

        if dam_commitment > 0:  # Must discharge
            max_dam_discharge = min(
                abs(dam_commitment),
                available_discharge_mwh * env.eff_sqrt / env.time_step_hours
            )
            dam_executed_mw = max_dam_discharge

            energy_mwh = dam_executed_mw * env.time_step_hours
            soc_delta = energy_mwh / env.eff_sqrt / env.capacity_mwh

            result.energy_mw = dam_executed_mw
            result.soc_delta = -soc_delta
            result.cycle_fraction = energy_mwh / env.capacity_mwh
            result.mwh_sold = energy_mwh
            result.is_activated = True
            result.direction = 'up'

        else:  # Must charge (dam_commitment < 0)
            max_dam_charge = min(
                abs(dam_commitment),
                available_charge_mwh / env.eff_sqrt / env.time_step_hours
            )
            dam_executed_mw = -max_dam_charge

            energy_mwh = max_dam_charge * env.time_step_hours
            soc_delta = energy_mwh * env.eff_sqrt / env.capacity_mwh

            result.energy_mw = dam_executed_mw
            result.soc_delta = soc_delta
            result.cycle_fraction = energy_mwh / env.capacity_mwh
            result.mwh_bought = energy_mwh
            result.is_activated = True
            result.direction = 'down'

        row = env.df.iloc[env.current_step]
        result.revenue = result.energy_mw * row.get('price', 100.0) * env.time_step_hours

        return result

    def get_commitment(self, step_idx: int) -> float:
        """I return the DAM commitment for a given step.

        I use endogenous schedule if available, otherwise CSV data.
        Uses precomputed O(1) lookup arrays.
        """
        env = self._env
        if env.enable_endogenous_dam and env.dam_schedule is not None:
            if (step_idx < len(env._step_day_id)
                    and env._step_day_id[step_idx] == env._dam_schedule_day_id):
                day_offset = env._step_day_offset[step_idx]
                if day_offset < len(env.dam_schedule):
                    return float(env.dam_schedule[day_offset])

        return float(env._dam_commitment_values[step_idx])

    def calculate_soc_reserve(self) -> Tuple[float, float]:
        """I calculate dynamic SoC reserves for future DAM commitment execution.

        Returns:
            (min_soc_reserve, max_soc_reserve): SoC bounds adjusted for future DAM
        """
        env = self._env
        min_soc_reserve = env.min_soc
        max_soc_reserve = env.max_soc

        lookahead = min(4 * env._sph, env.max_steps - env.current_step)
        if lookahead <= 0:
            return min_soc_reserve, max_soc_reserve

        commitments = []
        for i in range(1, lookahead + 1):
            future_idx = env.current_step + i
            if future_idx >= len(env.df):
                break
            dam = np.clip(env._get_dam_commitment(future_idx),
                          -env.max_power_mw, env.max_power_mw)
            commitments.append((i, dam))

        last_sell_pos = 0
        last_buy_pos = 0
        for pos, dam in commitments:
            if dam > 0.1:
                last_sell_pos = pos
            elif dam < -0.1:
                last_buy_pos = pos

        sell_energy_needed = 0.0
        buy_headroom_needed = 0.0
        intermediate_charge_potential = 0.0
        intermediate_discharge_potential = 0.0

        for pos, dam in commitments:
            if dam > 0.1:
                energy_needed = dam * env.time_step_hours / env.eff_sqrt
                sell_energy_needed += energy_needed
            elif dam < -0.1:
                energy_needed = abs(dam) * env.time_step_hours * env.eff_sqrt
                buy_headroom_needed += energy_needed
            else:
                potential = env.max_power_mw * env.time_step_hours * 0.25
                if pos < last_sell_pos:
                    intermediate_charge_potential += potential * env.eff_sqrt
                if pos < last_buy_pos:
                    intermediate_discharge_potential += potential / env.eff_sqrt

        net_sell_energy = max(0.0, sell_energy_needed - intermediate_charge_potential)
        net_buy_headroom = max(0.0, buy_headroom_needed - intermediate_discharge_potential)

        sell_soc_reserve = net_sell_energy / env.capacity_mwh
        buy_soc_reserve = net_buy_headroom / env.capacity_mwh

        buffer = 0.0
        if sell_soc_reserve > 0 or buy_soc_reserve > 0:
            buffer = env.max_power_mw * env.time_step_hours / env.capacity_mwh

        min_soc_reserve = min(env.max_soc, env.min_soc + sell_soc_reserve + buffer)
        max_soc_reserve = max(env.min_soc, env.max_soc - buy_soc_reserve - buffer)

        if min_soc_reserve > max_soc_reserve:
            midpoint = (min_soc_reserve + max_soc_reserve) / 2.0
            min_soc_reserve = midpoint
            max_soc_reserve = midpoint

        return min_soc_reserve, max_soc_reserve

    def calculate_shortfall_risk(self) -> float:
        """I calculate risk of not meeting future DAM commitments."""
        env = self._env
        sell_risk = 0.0
        buy_risk = 0.0
        total_sell_commitment = 0.0
        total_buy_commitment = 0.0

        lookahead = min(4 * env._sph, env.max_steps - env.current_step)

        for i in range(1, lookahead + 1):
            future_idx = env.current_step + i
            if future_idx >= len(env.df):
                break
            dam = np.clip(env._get_dam_commitment(future_idx),
                          -env.max_power_mw, env.max_power_mw)
            if dam > 0.1:
                total_sell_commitment += dam
            elif dam < -0.1:
                total_buy_commitment += abs(dam)

        if total_sell_commitment > 0:
            available_energy = (env.soc - env.min_soc) * env.capacity_mwh
            needed_energy = total_sell_commitment * env.time_step_hours / env.eff_sqrt
            if available_energy < needed_energy:
                sell_risk = (needed_energy - available_energy) / needed_energy
                sell_risk = np.clip(sell_risk, 0, 1)

        if total_buy_commitment > 0:
            available_headroom = (env.max_soc - env.soc) * env.capacity_mwh
            needed_headroom = total_buy_commitment * env.time_step_hours * env.eff_sqrt
            if available_headroom < needed_headroom:
                buy_risk = (needed_headroom - available_headroom) / needed_headroom
                buy_risk = np.clip(buy_risk, 0, 1)

        return max(sell_risk, buy_risk)

    def predict_soc(self) -> Tuple[float, float]:
        """I predict end-of-day SoC by simulating remaining DAM commitments.

        Returns:
            (predicted_soc_eod, net_energy_mwh)
        """
        env = self._env
        simulated_soc = env.soc
        net_energy_mwh = 0.0

        current_day_id = env._step_day_id[env.current_step]
        day_start = env._day_start[current_day_id]
        day_end = day_start + env._day_length[current_day_id]

        for step in range(env.current_step, min(day_end, len(env.df))):
            commitment = env._get_dam_commitment(step)
            if commitment < -0.1:
                energy = abs(commitment) * env.time_step_hours * env.eff_sqrt
                simulated_soc = min(env.max_soc, simulated_soc + energy / env.capacity_mwh)
                net_energy_mwh -= abs(commitment) * env.time_step_hours
            elif commitment > 0.1:
                energy = commitment * env.time_step_hours / env.eff_sqrt
                simulated_soc = max(env.min_soc, simulated_soc - energy / env.capacity_mwh)
                net_energy_mwh += commitment * env.time_step_hours

        return simulated_soc, net_energy_mwh

    def generate_endogenous_schedule(self) -> Optional[np.ndarray]:
        """I generate a DAM commitment schedule based on price forecasts.

        Returns:
            Per-step MW schedule or None if insufficient data
        """
        env = self._env
        ts = env.df.index[env.current_step]

        day_indices = env._get_day_indices_for_step(env.current_step)

        if len(day_indices) < 12 * env._sph:
            return None

        # I get price forecasts
        if env.market_forecaster is not None:
            forecast = env.market_forecaster.dam_forecaster.predict(
                env.df, env.current_step, 24
            )
        else:
            hourly_indices = day_indices[::env._sph][:24]
            forecast = np.array([env.df.iloc[i].get('price', 80.0) for i in hourly_indices])
            noise = np.random.normal(env._dam_forecast_bias, 22.0, len(forecast))
            forecast = forecast + noise

        if len(forecast) < 12:
            return None

        p25 = np.percentile(forecast, 25)
        p75 = np.percentile(forecast, 75)
        daily_spread = p75 - p25

        # I use a lower threshold than the configured min_spread
        # The configured min_spread (30 EUR) was too aggressive — forecast compression
        # makes P75-P25 spread appear small even when actual spreads are tradeable
        effective_min = min(env.dam_bidder_min_spread, 10.0)
        if daily_spread < effective_min:
            return np.zeros(len(day_indices))

        spread_factor = np.clip(daily_spread / 80.0, 0.4, 1.0)
        base_power = env.max_power_mw * spread_factor

        hourly_schedule = np.zeros(len(forecast))
        for h in range(len(forecast)):
            if forecast[h] <= p25:
                hourly_schedule[h] = -base_power * np.random.uniform(0.6, 1.0)
            elif forecast[h] >= p75:
                hourly_schedule[h] = base_power * np.random.uniform(0.6, 1.0)

        # I ensure energy balance
        total_charge = abs(hourly_schedule[hourly_schedule < 0].sum())
        total_discharge = hourly_schedule[hourly_schedule > 0].sum()
        if total_charge < total_discharge * 1.1 and total_discharge > 0:
            scale = total_charge / (total_discharge * 1.1 + 1e-6)
            hourly_schedule[hourly_schedule > 0] *= scale

        # I simulate SoC trajectory and cap commitments
        simulated_soc = env.soc
        for h in range(len(hourly_schedule)):
            commitment = hourly_schedule[h]
            if commitment < -0.1:
                soc_delta = abs(commitment) * 1.0 * env.eff_sqrt / env.capacity_mwh
                new_soc = simulated_soc + soc_delta
                if new_soc > env.max_soc:
                    headroom = (env.max_soc - simulated_soc) * env.capacity_mwh
                    max_power = headroom / (1.0 * env.eff_sqrt)
                    hourly_schedule[h] = -max(max_power, 0.0)
                    simulated_soc = env.max_soc
                else:
                    simulated_soc = new_soc
            elif commitment > 0.1:
                soc_delta = commitment * 1.0 / env.eff_sqrt / env.capacity_mwh
                new_soc = simulated_soc - soc_delta
                if new_soc < env.min_soc:
                    available = (simulated_soc - env.min_soc) * env.capacity_mwh
                    max_power = available * env.eff_sqrt / 1.0
                    hourly_schedule[h] = max(max_power, 0.0)
                    simulated_soc = env.min_soc
                else:
                    simulated_soc = new_soc

        schedule = np.repeat(hourly_schedule, env._sph)[:len(day_indices)]
        schedule = np.clip(schedule, -env.max_power_mw, env.max_power_mw)

        return schedule


class DamBidSimulator:
    """I simulate DAM bidding and EUPHEMIA market clearing.

    I generate bids based on price forecasts and agent aggressiveness,
    then simulate acceptance/rejection using actual clearing prices.

    Real EUPHEMIA is a welfare-maximizing LP. I use a simplified model:
    - SELL bid accepted if limit_price <= clearing_price
    - BUY bid accepted if limit_price >= clearing_price
    - Near the margin (±2 EUR): partial acceptance
    - Settlement always at clearing_price (marginal pricing)
    """

    def __init__(self, max_power_mw: float = 30.0, capacity_mwh: float = 146.0,
                 efficiency: float = 0.94, min_soc: float = 0.05, max_soc: float = 0.95,
                 margin_eur: float = 2.0):
        self._max_power = max_power_mw
        self._capacity = capacity_mwh
        self._eff_sqrt = np.sqrt(efficiency)
        self._min_soc = min_soc
        self._max_soc = max_soc
        self._margin = margin_eur

    def generate_bids(self, forecast: np.ndarray, aggressiveness: float,
                      current_soc: float, sph: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """I generate DAM bids based on price forecast and agent aggressiveness.

        Args:
            forecast: Price forecast array (hourly or quarter-hourly)
            aggressiveness: -1 (conservative) to +1 (aggressive)
            current_soc: Current battery SoC
            sph: Steps per hour (4 for 15-min, 1 for hourly)

        Returns:
            (bid_mw, bid_prices): arrays of MW commitments and limit prices
        """
        n_slots = len(forecast)

        # I compute spread threshold based on aggressiveness
        # -1 → 60 EUR min spread (very selective)
        #  0 → 30 EUR min spread (balanced)
        # +1 → 10 EUR min spread (aggressive)
        min_spread = 20.0 - 12.0 * aggressiveness  # 32 to 8 EUR

        p25 = np.percentile(forecast, 25)
        p75 = np.percentile(forecast, 75)
        daily_spread = p75 - p25

        bid_mw = np.zeros(n_slots)
        bid_prices = np.zeros(n_slots)

        if daily_spread < min_spread:
            return bid_mw, bid_prices

        # I compute commitment intensity based on aggressiveness
        # -1 → 30% of max power
        #  0 → 50% of max power
        # +1 → 80% of max power
        commit_ratio = 0.55 + 0.25 * aggressiveness  # 0.30 to 0.80

        base_power = self._max_power * commit_ratio

        for q in range(n_slots):
            price = forecast[q]

            if price <= p25:
                # I bid to BUY (charge) at cheap hours
                bid_mw[q] = -base_power * np.random.uniform(0.5, 1.0)
                # I set limit price near forecast with small margin
                # Aggressive bids (high aggressiveness) set limit closer to forecast
                # Conservative bids set limit further above (willing to pay more)
                margin = np.random.uniform(2.0, 8.0) * (1.0 - 0.3 * aggressiveness)
                bid_prices[q] = price + margin
            elif price >= p75:
                # I bid to SELL (discharge) at expensive hours
                bid_mw[q] = base_power * np.random.uniform(0.5, 1.0)
                # I set limit price near forecast — some will be rejected
                margin = np.random.uniform(2.0, 8.0) * (1.0 - 0.3 * aggressiveness)
                bid_prices[q] = price - margin

        # I ensure energy balance (can't sell more than we charge)
        total_charge = abs(bid_mw[bid_mw < 0].sum()) if (bid_mw < 0).any() else 0
        total_discharge = bid_mw[bid_mw > 0].sum() if (bid_mw > 0).any() else 0
        if total_charge < total_discharge * 1.1 and total_discharge > 0:
            scale = total_charge / (total_discharge * 1.1 + 1e-6)
            bid_mw[bid_mw > 0] *= scale

        # I simulate SoC trajectory to cap physically infeasible bids
        simulated_soc = current_soc
        time_step = 1.0 / sph
        for q in range(n_slots):
            mw = bid_mw[q]
            if mw < -0.1:  # Charge
                soc_delta = abs(mw) * time_step * self._eff_sqrt / self._capacity
                if simulated_soc + soc_delta > self._max_soc:
                    headroom = (self._max_soc - simulated_soc) * self._capacity
                    max_mw = headroom / (time_step * self._eff_sqrt)
                    bid_mw[q] = -max(max_mw, 0.0)
                    simulated_soc = self._max_soc
                else:
                    simulated_soc += soc_delta
            elif mw > 0.1:  # Discharge
                soc_delta = mw * time_step / self._eff_sqrt / self._capacity
                if simulated_soc - soc_delta < self._min_soc:
                    available = (simulated_soc - self._min_soc) * self._capacity
                    max_mw = available * self._eff_sqrt / time_step
                    bid_mw[q] = max(max_mw, 0.0)
                    simulated_soc = self._min_soc
                else:
                    simulated_soc -= soc_delta

        bid_mw = np.clip(bid_mw, -self._max_power, self._max_power)
        return bid_mw, bid_prices

    def simulate_euphemia(self, bid_mw: np.ndarray, bid_prices: np.ndarray,
                          actual_prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """I simulate EUPHEMIA market clearing.

        Args:
            bid_mw: Bid MW per slot (positive=sell, negative=buy)
            bid_prices: Limit prices per slot
            actual_prices: Actual DAM clearing prices per slot

        Returns:
            (accepted_mw, acceptance_rate): Accepted MW per slot and overall rate
        """
        n_slots = len(bid_mw)
        accepted_mw = np.zeros(n_slots)
        n_bids = 0
        n_accepted = 0

        for q in range(n_slots):
            mw = bid_mw[q]
            if abs(mw) < 0.1:
                continue

            n_bids += 1
            limit = bid_prices[q]
            clearing = actual_prices[q]

            if mw > 0:  # SELL bid
                # I accept if seller's limit <= clearing price
                # (seller willing to accept at or below clearing)
                if limit <= clearing:
                    distance = clearing - limit
                    if distance < self._margin:
                        # I apply partial acceptance near the margin
                        ratio = 0.5 + 0.5 * (distance / self._margin)
                        accepted_mw[q] = mw * ratio
                    else:
                        accepted_mw[q] = mw
                    n_accepted += 1
                # else: rejected (seller wants more than clearing)

            elif mw < 0:  # BUY bid
                # I accept if buyer's limit >= clearing price
                # (buyer willing to pay at or above clearing)
                if limit >= clearing:
                    distance = limit - clearing
                    if distance < self._margin:
                        ratio = 0.5 + 0.5 * (distance / self._margin)
                        accepted_mw[q] = mw * ratio
                    else:
                        accepted_mw[q] = mw
                    n_accepted += 1
                # else: rejected (buyer won't pay clearing price)

        acceptance_rate = n_accepted / max(n_bids, 1)
        return accepted_mw, acceptance_rate


class DamOptimizer:
    """I optimize DAM bidding using linear programming.

    I solve a deterministic SoC-trajectory optimization to maximize
    arbitrage revenue: buy cheap hours, sell expensive hours, respecting
    all physical battery constraints.

    I run ONCE per day at the day boundary. The RL agent does NOT control
    DAM — this is a mathematical optimization, not a learning problem.
    """

    def __init__(self, max_power_mw: float = 30.0, capacity_mwh: float = 146.0,
                 efficiency: float = 0.94, min_soc: float = 0.05, max_soc: float = 0.95,
                 time_step_hours: float = 0.25, max_daily_cycles: float = 2.5,
                 end_of_day_soc_target: float = 0.40):
        self._max_power = max_power_mw
        self._capacity = capacity_mwh
        self._eff_sqrt = np.sqrt(efficiency)
        self._min_soc = min_soc
        self._max_soc = max_soc
        self._dt = time_step_hours
        self._max_cycles = max_daily_cycles
        self._eod_soc = end_of_day_soc_target

    def optimize(self, forecast_prices: np.ndarray, current_soc: float,
                 n_slots: int = 96) -> np.ndarray:
        """I compute optimal DAM schedule using greedy spread-maximization.

        I use a greedy approach (not LP) because it's simpler, faster,
        and handles the non-linear efficiency correctly. The LP approximation
        would linearize efficiency which introduces errors.

        Algorithm:
        1. Rank hours by price: cheapest → buy, most expensive → sell
        2. Simulate SoC trajectory, capping when limits reached
        3. Ensure end-of-day SoC ≥ target

        Args:
            forecast_prices: Array of n_slots forecast prices (EUR/MWh)
            current_soc: Starting SoC (0-1)
            n_slots: Number of time slots (96 for 15-min, 24 for hourly)

        Returns:
            schedule: Array of n_slots MW values (positive=sell, negative=buy)
        """
        n = min(n_slots, len(forecast_prices))
        prices = forecast_prices[:n]
        schedule = np.zeros(n)

        if n == 0:
            return schedule

        # I compute price percentiles for buy/sell thresholds
        p25 = np.percentile(prices, 25)
        p75 = np.percentile(prices, 75)
        p_mean = np.mean(prices)
        spread = p75 - p25

        # I skip if spread too small (not worth trading after efficiency + network costs)
        # 5 EUR spread minimum: after 6% efficiency loss + 8 EUR network charges,
        # a 5 EUR spread still loses money, but 10+ EUR makes profit
        if spread < 5.0:
            return schedule

        # I rank slots by profitability
        # Buy slots: price < p25 (cheapest quartile)
        # Sell slots: price > p75 (most expensive quartile)
        buy_slots = np.where(prices <= p25)[0]
        sell_slots = np.where(prices >= p75)[0]

        # I determine power level based on spread
        # Higher spread → more aggressive trading
        intensity = np.clip(spread / 100.0, 0.3, 0.9)
        base_power = self._max_power * intensity

        # I assign buy/sell with randomized power (realistic bid diversity)
        for s in buy_slots:
            schedule[s] = -base_power * np.random.uniform(0.5, 1.0)
        for s in sell_slots:
            schedule[s] = base_power * np.random.uniform(0.5, 1.0)

        # I simulate SoC trajectory and cap infeasible commitments
        schedule = self._cap_by_soc_trajectory(schedule, current_soc, n)

        # I ensure energy balance: total charge >= total discharge (sustainability)
        total_charge = abs(schedule[schedule < 0].sum()) * self._dt
        total_discharge = schedule[schedule > 0].sum() * self._dt
        if total_discharge > total_charge * 1.2:
            # I scale down sell to maintain balance
            sell_mask = schedule > 0
            if sell_mask.any():
                scale = total_charge * 1.1 / max(total_discharge, 0.1)
                schedule[sell_mask] *= scale

        # I enforce cycle budget
        total_energy = np.sum(np.abs(schedule)) * self._dt
        max_energy = self._max_cycles * self._capacity * 2  # Full cycles = 2 × capacity
        if total_energy > max_energy:
            scale = max_energy / total_energy
            schedule *= scale

        return schedule

    def _cap_by_soc_trajectory(self, schedule: np.ndarray, start_soc: float,
                                n: int) -> np.ndarray:
        """I simulate SoC forward and cap commitments that would violate limits."""
        soc = start_soc
        capped = schedule.copy()

        for t in range(n):
            power = capped[t]
            if abs(power) < 0.1:
                continue

            if power > 0:  # Discharge
                energy = power * self._dt
                soc_needed = energy / self._eff_sqrt / self._capacity
                available = soc - self._min_soc
                if soc_needed > available:
                    # I cap to available
                    max_power = available * self._eff_sqrt * self._capacity / self._dt
                    capped[t] = max(0, max_power)
                soc -= capped[t] * self._dt / self._eff_sqrt / self._capacity
            else:  # Charge
                energy = abs(power) * self._dt
                soc_gained = energy * self._eff_sqrt / self._capacity
                available = self._max_soc - soc
                if soc_gained > available:
                    max_power = available / self._eff_sqrt * self._capacity / self._dt
                    capped[t] = -max(0, max_power)
                soc += abs(capped[t]) * self._dt * self._eff_sqrt / self._capacity

            soc = np.clip(soc, self._min_soc, self._max_soc)

        # I check end-of-day SoC target
        if soc < self._eod_soc:
            # I need more charging — scale down sells
            deficit = self._eod_soc - soc
            sell_mask = capped > 0
            if sell_mask.any():
                total_sell = capped[sell_mask].sum() * self._dt / self._eff_sqrt / self._capacity
                reduction = min(1.0, deficit / max(total_sell, 0.001))
                capped[sell_mask] *= (1 - reduction)

        return capped
