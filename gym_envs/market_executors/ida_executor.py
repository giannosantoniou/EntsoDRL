"""
IDA (Intra-Day Auction) Executor

I handle all IDA-related logic:
- IDA gate closure triggering (IDA1 15:00, IDA2 22:00, IDA3 10:00)
- IDA schedule generation (rule-based, NN-forecast driven, SoC-aware)
- IDA commitment execution (mandatory, like DAM)
- IDA revenue tracking per sub-market (IDA1/2/3)
- IDA violation detection and cost tracking
- IDA observation helpers (phase, gate timing, correction signal)

The IDA schedules are locked at gate closure and become mandatory commitments.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING

from gym_envs.market_executors import MarketResult

if TYPE_CHECKING:
    from gym_envs.battery_env_unified import BatteryEnvUnified


class IdaExecutor:
    """I execute IDA auction gates and locked schedule positions.

    IDA is mandatory like DAM once locked. It sits between aFRR and mFRR
    in the capacity cascade: DAM → aFRR → IDA → mFRR → XBID → FreeBid.
    """

    def __init__(self, env: 'BatteryEnvUnified'):
        self._env = env

    def check_gate_triggers(self, action):
        """I check if we're at an IDA gate closure and trigger if needed.

        Called at the start of each step before other market execution.
        """
        env = self._env
        if not env.enable_full_market:
            return

        ts_gate = env.df.index[env.current_step]
        gate_hour = ts_gate.hour
        gate_minute = ts_gate.minute

        ida_action_raw = action[2]
        ida_level = env.INTRADAY_LEVELS[ida_action_raw]

        if gate_hour == 15 and gate_minute == 0 and not env.ida1_locked_today:
            self._trigger_gate(ida_number=1, participation_level=ida_level)
        elif gate_hour == 22 and gate_minute == 0 and not env.ida2_locked_today:
            self._trigger_gate(ida_number=2, participation_level=ida_level)
        elif gate_hour == 10 and gate_minute == 0 and not env.ida3_locked_today:
            self._trigger_gate(ida_number=3, participation_level=ida_level)

    def execute(self, remaining_after_afrr: float) -> MarketResult:
        """I execute locked IDA positions for the current step.

        Args:
            remaining_after_afrr: MW available after DAM and aFRR

        Returns:
            MarketResult with IDA execution details
        """
        env = self._env
        result = MarketResult()

        if not env.enable_full_market:
            return result

        net_ida = self.get_commitment(env.current_step)
        net_ida = np.clip(net_ida, -remaining_after_afrr, remaining_after_afrr)

        if abs(net_ida) <= 0.1:
            # I still compute revenue from per-schedule positions
            self._compute_ida_revenue(result, 0.0)
            return result

        available_discharge_mwh = (env.soc - env.min_soc) * env.capacity_mwh
        available_charge_mwh = (env.max_soc - env.soc) * env.capacity_mwh
        ida_shortfall_mw = 0.0

        if net_ida > 0:  # Discharge (sell)
            max_ida = min(net_ida,
                          available_discharge_mwh * env.eff_sqrt / env.time_step_hours)
            if max_ida > 0.1:
                energy_mwh = max_ida * env.time_step_hours
                soc_delta = energy_mwh / env.eff_sqrt / env.capacity_mwh
                env.soc = max(env.min_soc, env.soc - soc_delta)
                result.energy_mw = max_ida
                result.cycle_fraction = energy_mwh / env.capacity_mwh
                env.total_cycles += result.cycle_fraction
                env.daily_cycles += result.cycle_fraction
                result.mwh_sold = energy_mwh
            ida_shortfall_mw = max(0, net_ida - max_ida) if max_ida > 0.1 else net_ida

        else:  # Charge (buy)
            max_ida = min(abs(net_ida),
                          available_charge_mwh / env.eff_sqrt / env.time_step_hours)
            if max_ida > 0.1:
                energy_mwh = max_ida * env.time_step_hours
                soc_delta = energy_mwh * env.eff_sqrt / env.capacity_mwh
                env.soc = min(env.max_soc, env.soc + soc_delta)
                result.energy_mw = -max_ida
                result.cycle_fraction = energy_mwh / env.capacity_mwh
                env.total_cycles += result.cycle_fraction
                env.daily_cycles += result.cycle_fraction
                result.mwh_bought = energy_mwh
            ida_shortfall_mw = max(0, abs(net_ida) - max_ida) if max_ida > 0.1 else abs(net_ida)

        result.capacity_used_mw = abs(result.energy_mw)
        result.is_activated = abs(result.energy_mw) > 0.1

        # I compute revenue from individual schedule positions
        self._compute_ida_revenue(result, result.energy_mw)

        # I track IDA energy volumes
        if result.energy_mw > 0:
            env.ida_mwh_sold += result.mwh_sold
        elif result.energy_mw < 0:
            env.ida_mwh_bought += result.mwh_bought

        # I track IDA violations
        if ida_shortfall_mw > 0.1:
            env.episode_ida_violation_steps += 1
            row = env.df.iloc[env.current_step]
            avg_ida_price = row.get('ida1_clearing_price', row.get('price', 100.0))
            ida_violation_cost = ida_shortfall_mw * avg_ida_price * 1.5 * env.time_step_hours
            env.episode_ida_violation_cost += ida_violation_cost

        result.metadata['ida_shortfall_mw'] = ida_shortfall_mw

        return result

    def get_commitment(self, step_idx: int) -> float:
        """I return the total locked IDA commitment for a given step."""
        env = self._env
        total = 0.0
        ts = env.df.index[step_idx]
        hour = ts.hour

        day_offset = env._step_day_offset[step_idx]

        if env.ida1_schedule is not None and 0 <= day_offset < len(env.ida1_schedule):
            total += env.ida1_schedule[day_offset]

        if env.ida2_schedule is not None and 0 <= day_offset < len(env.ida2_schedule):
            total += env.ida2_schedule[day_offset]

        if env.ida3_schedule is not None and hour >= 12 and 0 <= day_offset < len(env.ida3_schedule):
            total += env.ida3_schedule[day_offset]

        return total

    # ─── Schedule Generation ─────────────────────────────────────────

    def generate_schedule(self, ida_number: int,
                          participation_level: float = 1.0) -> np.ndarray:
        """I generate an IDA correction schedule based on NN price forecasts.

        At gate closure, I compare forecasted IDA clearing prices vs existing
        DAM+IDA commitments. I bid where the spread is favorable or where
        correction is needed for SoC feasibility.

        Args:
            ida_number: 1, 2, or 3
            participation_level: -1.0 to 1.0 scaling factor (from agent action)

        Returns:
            Per-QH MW schedule for the delivery window (24*sph or 12*sph values).
            Positive = sell/discharge, Negative = buy/charge.
        """
        env = self._env

        # I determine delivery window for this IDA
        if ida_number == 3:
            delivery_hours = list(range(12, 24))
        else:
            delivery_hours = list(range(0, 24))
        n_hours = len(delivery_hours)

        # I get day indices using O(1) precomputed lookup
        day_indices = env._get_day_indices_for_step(env.current_step)

        if len(day_indices) < n_hours * env._sph:
            return np.zeros(n_hours * env._sph)

        # I get NN forecast for IDA clearing prices
        if env.ida_forecaster is not None and hasattr(env.ida_forecaster, 'predict'):
            forecast_prices = env.ida_forecaster.predict(
                env.df, env.current_step, ida_number
            )
        elif env.market_forecaster is not None and hasattr(env.market_forecaster, 'ida_forecaster'):
            forecast_prices = env.market_forecaster.ida_forecaster.predict(
                env.df, env.current_step, ida_number
            )
        else:
            # I use DAM prices with noise as fallback
            forecast_prices = np.zeros(n_hours)
            for h_idx, hour in enumerate(delivery_hours):
                qh_idx = day_indices[0] + hour * env._sph
                if qh_idx < len(env.df):
                    forecast_prices[h_idx] = env.df.iloc[qh_idx].get('price', 80.0)
                else:
                    forecast_prices[h_idx] = 80.0
            # I add noise to avoid oracle effect
            noise_std = {1: 4.0, 2: 8.0, 3: 12.0}.get(ida_number, 8.0)
            forecast_prices += np.random.normal(0, noise_std, n_hours)
            forecast_prices = np.maximum(forecast_prices, 0.0)

        # I get existing committed schedule (DAM + prior IDAs)
        existing_schedule = self._get_total_committed_schedule(
            delivery_hours, day_indices
        )

        # I simulate SoC trajectory under existing commitments
        soc_trajectory = self._simulate_soc_trajectory(
            existing_schedule, delivery_hours, day_indices
        )

        # I compute remaining capacity for each QH
        ida_schedule = np.zeros(n_hours * env._sph)

        for h_idx, hour in enumerate(delivery_hours):
            # I get DAM price for this hour (known at gate closure)
            qh_idx = day_indices[0] + hour * env._sph
            if qh_idx < len(env.df):
                dam_price = env.df.iloc[qh_idx].get('price', 80.0)
            else:
                dam_price = 80.0

            ida_price = forecast_prices[h_idx]
            committed_mw = existing_schedule[h_idx * env._sph] if h_idx * env._sph < len(existing_schedule) else 0.0
            predicted_soc = soc_trajectory[h_idx] if h_idx < len(soc_trajectory) else 0.5

            # I compute remaining capacity after existing commitments
            remaining = env.max_power_mw - abs(committed_mw) - env.afrr_commitment_mw
            remaining = max(0, remaining)

            correction_mw = 0.0

            # CASE 1: SoC correction needed (safety override)
            if predicted_soc > 0.88 and committed_mw < -0.1:
                # I am overcharging — reduce charging or add sell position
                correction_mw = min(abs(committed_mw) * 0.5, remaining)

            elif predicted_soc < 0.12 and committed_mw > 0.1:
                # I am over-discharging — reduce discharge or add buy position
                correction_mw = -min(committed_mw * 0.5, remaining)

            # CASE 2: Arbitrage opportunity (spread exceeds threshold)
            elif ida_price > dam_price + env.ida_min_spread:
                # I sell on IDA (higher price than DAM commitment)
                power = remaining * min(1.0, (ida_price - dam_price) / 50.0)
                correction_mw = power * np.random.uniform(0.5, 1.0)

            elif ida_price < dam_price - env.ida_min_spread:
                # I buy on IDA (lower price than DAM commitment)
                power = remaining * min(1.0, (dam_price - ida_price) / 50.0)
                correction_mw = -power * np.random.uniform(0.5, 1.0)

            # I apply to all QHs within this hour
            for qh in range(env._sph):
                slot = h_idx * env._sph + qh
                if slot < len(ida_schedule):
                    ida_schedule[slot] = correction_mw

        # I scale by agent's participation level BEFORE SoC cap
        # Negative participation_level reverses direction (agent directional control)
        ida_schedule *= participation_level

        # I apply SoC trajectory simulation to cap infeasible positions
        ida_schedule = self._soc_cap_schedule(ida_schedule, existing_schedule,
                                              delivery_hours, day_indices)

        # I clip to inverter limits
        ida_schedule = np.clip(ida_schedule, -env.max_power_mw, env.max_power_mw)

        return ida_schedule

    def _get_total_committed_schedule(self, delivery_hours: list,
                                       day_indices: np.ndarray) -> np.ndarray:
        """I return the total committed MW schedule (DAM + prior IDAs) for the delivery window.

        Returns per-QH MW array (positive = sell, negative = buy).
        """
        env = self._env
        n_qh = len(delivery_hours) * env._sph
        schedule = np.zeros(n_qh)

        for h_idx, hour in enumerate(delivery_hours):
            for qh in range(env._sph):
                slot = h_idx * env._sph + qh
                qh_idx = day_indices[0] + hour * env._sph + qh
                if qh_idx < len(env.df):
                    # I get DAM commitment
                    dam = env._get_dam_commitment(qh_idx)
                    schedule[slot] = dam

                    # I add prior IDA commitments
                    if env.ida1_schedule is not None and qh_idx - day_indices[0] < len(env.ida1_schedule):
                        schedule[slot] += env.ida1_schedule[qh_idx - day_indices[0]]
                    if env.ida2_schedule is not None and qh_idx - day_indices[0] < len(env.ida2_schedule):
                        schedule[slot] += env.ida2_schedule[qh_idx - day_indices[0]]
                    # I don't add IDA3 here because IDA3 is the current one being generated
                    # (or not yet generated)

        return schedule

    def _simulate_soc_trajectory(self, committed_schedule: np.ndarray,
                                  delivery_hours: list,
                                  day_indices: np.ndarray) -> np.ndarray:
        """I simulate SoC trajectory under existing commitments for the delivery window.

        Returns per-hour SoC array (one value per delivery hour).
        """
        env = self._env
        soc = env.soc
        trajectory = np.zeros(len(delivery_hours))

        for h_idx, hour in enumerate(delivery_hours):
            # I sum MW across all QHs in this hour
            total_mw = 0.0
            for qh in range(env._sph):
                slot = h_idx * env._sph + qh
                if slot < len(committed_schedule):
                    total_mw += committed_schedule[slot]
            avg_mw = total_mw / env._sph if env._sph > 0 else 0.0

            # I update SoC
            if avg_mw > 0.1:  # Sell/discharge
                energy = avg_mw * 1.0 / env.eff_sqrt
                soc = max(env.min_soc, soc - energy / env.capacity_mwh)
            elif avg_mw < -0.1:  # Buy/charge
                energy = abs(avg_mw) * 1.0 * env.eff_sqrt
                soc = min(env.max_soc, soc + energy / env.capacity_mwh)

            trajectory[h_idx] = soc

        return trajectory

    def _soc_cap_schedule(self, ida_schedule: np.ndarray,
                           existing_schedule: np.ndarray,
                           delivery_hours: list,
                           day_indices: np.ndarray) -> np.ndarray:
        """I cap IDA schedule positions that would violate SoC limits.

        I simulate the combined trajectory (existing + IDA) and reduce
        IDA positions where SoC would breach min/max limits.
        """
        env = self._env
        soc = env.soc
        capped = ida_schedule.copy()

        for h_idx, hour in enumerate(delivery_hours):
            for qh in range(env._sph):
                slot = h_idx * env._sph + qh
                if slot >= len(capped):
                    break

                # I combine existing + proposed IDA
                existing_mw = existing_schedule[slot] if slot < len(existing_schedule) else 0.0
                total_mw = existing_mw + capped[slot]

                if total_mw > 0.1:  # Net sell/discharge
                    energy = total_mw * env.time_step_hours / env.eff_sqrt
                    new_soc = soc - energy / env.capacity_mwh
                    if new_soc < env.min_soc:
                        # I reduce the IDA sell position
                        available = max(0, (soc - env.min_soc) * env.capacity_mwh * env.eff_sqrt / env.time_step_hours)
                        max_ida_sell = max(0, available - existing_mw) if existing_mw > 0 else available
                        capped[slot] = min(capped[slot], max_ida_sell)
                        total_mw = existing_mw + capped[slot]
                    soc = max(env.min_soc, soc - total_mw * env.time_step_hours / env.eff_sqrt / env.capacity_mwh)

                elif total_mw < -0.1:  # Net buy/charge
                    energy = abs(total_mw) * env.time_step_hours * env.eff_sqrt
                    new_soc = soc + energy / env.capacity_mwh
                    if new_soc > env.max_soc:
                        # I reduce the IDA buy position
                        headroom = max(0, (env.max_soc - soc) * env.capacity_mwh / env.eff_sqrt / env.time_step_hours)
                        max_ida_buy = max(0, headroom - abs(existing_mw)) if existing_mw < 0 else headroom
                        capped[slot] = max(capped[slot], -max_ida_buy)
                        total_mw = existing_mw + capped[slot]
                    soc = min(env.max_soc, soc + abs(total_mw) * env.time_step_hours * env.eff_sqrt / env.capacity_mwh)

        return capped

    def _trigger_gate(self, ida_number: int, participation_level: float):
        """I trigger IDA gate closure: generate and lock the IDA schedule."""
        env = self._env
        schedule = self.generate_schedule(ida_number, participation_level)

        day_indices = env._get_day_indices_for_step(env.current_step)
        n_qh_day = len(day_indices)

        if ida_number == 1:
            env.ida1_schedule = np.zeros(n_qh_day)
            for h_idx, hour in enumerate(range(0, 24)):
                for qh in range(env._sph):
                    slot = h_idx * env._sph + qh
                    day_slot = hour * env._sph + qh
                    if slot < len(schedule) and day_slot < n_qh_day:
                        env.ida1_schedule[day_slot] = schedule[slot]
            env.ida1_locked_today = True
        elif ida_number == 2:
            env.ida2_schedule = np.zeros(n_qh_day)
            for h_idx, hour in enumerate(range(0, 24)):
                for qh in range(env._sph):
                    slot = h_idx * env._sph + qh
                    day_slot = hour * env._sph + qh
                    if slot < len(schedule) and day_slot < n_qh_day:
                        env.ida2_schedule[day_slot] = schedule[slot]
            env.ida2_locked_today = True
        elif ida_number == 3:
            env.ida3_schedule = np.zeros(n_qh_day)
            # IDA3 delivers only 12:00-24:00
            for h_idx, hour in enumerate(range(12, 24)):
                for qh in range(env._sph):
                    slot = h_idx * env._sph + qh
                    day_slot = hour * env._sph + qh
                    if slot < len(schedule) and day_slot < n_qh_day:
                        env.ida3_schedule[day_slot] = schedule[slot]
            env.ida3_locked_today = True

    def _compute_ida_revenue(self, result: MarketResult, ida_energy_mw: float):
        """I compute IDA revenue from individual schedule positions."""
        env = self._env
        row = env.df.iloc[env.current_step]
        day_offset = env._step_day_offset[env.current_step]

        ida1_pos = 0.0
        if env.ida1_schedule is not None and 0 <= day_offset < len(env.ida1_schedule):
            ida1_pos = env.ida1_schedule[day_offset]
        ida2_pos = 0.0
        if env.ida2_schedule is not None and 0 <= day_offset < len(env.ida2_schedule):
            ida2_pos = env.ida2_schedule[day_offset]
        ida3_pos = 0.0
        if (env.ida3_schedule is not None and env.df.index[env.current_step].hour >= 12
                and 0 <= day_offset < len(env.ida3_schedule)):
            ida3_pos = env.ida3_schedule[day_offset]

        ida_revenue = 0.0
        if abs(ida1_pos) > 0.01:
            p1 = row.get('ida1_clearing_price', row.get('price', 100.0))
            r1 = ida1_pos * p1 * env.time_step_hours
            ida_revenue += r1
            env.ida1_profit += r1
        if abs(ida2_pos) > 0.01:
            p2 = row.get('ida2_clearing_price', row.get('price', 100.0))
            r2 = ida2_pos * p2 * env.time_step_hours
            ida_revenue += r2
            env.ida2_profit += r2
        if abs(ida3_pos) > 0.01:
            p3 = row.get('ida3_clearing_price', row.get('price', 100.0))
            r3 = ida3_pos * p3 * env.time_step_hours
            ida_revenue += r3
            env.ida3_profit += r3

        # I scale proportionally if execution was clipped
        net_ida_committed = abs(ida1_pos) + abs(ida2_pos) + abs(ida3_pos)
        if net_ida_committed > 0.1 and abs(ida_energy_mw) > 0.01:
            execution_ratio = min(1.0, abs(ida_energy_mw) / net_ida_committed)
            ida_revenue *= execution_ratio

        result.revenue = ida_revenue
        env.ida_profit += ida_revenue

    # ─── Observation Helpers ─────────────────────────────────────────

    def get_hours_to_next_gate(self) -> float:
        """I return hours until the next IDA gate closure."""
        env = self._env
        ts = env.df.index[env.current_step]
        hour = ts.hour + ts.minute / 60.0

        # IDA gate closures: D-1 15:00 (IDA1), D-1 22:00 (IDA2), D+0 10:00 (IDA3)
        if hour < 10:
            return 10.0 - hour  # Next: IDA3 at 10:00
        elif hour < 15:
            return 15.0 - hour  # Next: IDA1 at 15:00 (for next day)
        elif hour < 22:
            return 22.0 - hour  # Next: IDA2 at 22:00
        else:
            return 24.0 - hour + 10.0  # Next: IDA3 at 10:00 tomorrow

    def get_forecast_vs_dam_spread(self, row) -> float:
        """I return the mean forecast IDA-DAM spread for the next upcoming IDA."""
        env = self._env
        ts = env.df.index[env.current_step]
        hour = ts.hour

        # I determine which IDA gate is next
        if hour < 10:
            next_ida = 3
        elif hour < 15:
            next_ida = 1
        elif hour < 22:
            next_ida = 2
        else:
            next_ida = 3  # Tomorrow's IDA3

        # I try to get forecast spread
        if env.ida_forecaster is not None and hasattr(env.ida_forecaster, 'predict_mean_spread'):
            try:
                return env.ida_forecaster.predict_mean_spread(
                    env.df, env.current_step, next_ida
                )
            except Exception:
                pass

        if (env.market_forecaster is not None
                and hasattr(env.market_forecaster, 'ida_forecaster')):
            try:
                return env.market_forecaster.ida_forecaster.predict_mean_spread(
                    env.df, env.current_step, next_ida
                )
            except Exception:
                pass

        # I fall back to pre-computed column or zero
        return row.get(f'ida{next_ida}_forecast_vs_dam', 0.0)

    def check_soc_trajectory_feasible(self) -> bool:
        """I check if the current DAM+IDA schedule can execute without SoC violation.

        I simulate the combined trajectory through end of day and check
        if SoC stays within [min_soc, max_soc] bounds.

        Uses O(1) precomputed day lookup (no self.df.index.date == X scan).
        """
        env = self._env
        soc = env.soc

        # I use precomputed day boundaries instead of O(n) date scan
        current_day_id = env._step_day_id[env.current_step]
        day_start = env._day_start[current_day_id]
        day_end = day_start + env._day_length[current_day_id]

        for step in range(env.current_step, min(day_end, len(env.df))):
            dam = env._get_dam_commitment(step)
            ida = env._get_ida_commitment(step)
            total = dam + ida

            if total > 0.1:  # Sell/discharge
                energy = total * env.time_step_hours / env.eff_sqrt
                soc -= energy / env.capacity_mwh
                if soc < env.min_soc - 0.01:
                    return False
            elif total < -0.1:  # Buy/charge
                energy = abs(total) * env.time_step_hours * env.eff_sqrt
                soc += energy / env.capacity_mwh
                if soc > env.max_soc + 0.01:
                    return False

        return True

    def get_correction_signal(self) -> float:
        """I compute the signed magnitude of SoC correction needed.

        Positive = I need to reduce discharge (buy more / sell less)
        Negative = I need to reduce charge (sell more / buy less)
        """
        env = self._env
        predicted_soc, _ = env._predict_dam_soc()

        # I also account for IDA commitments
        ida_total = env._get_ida_commitment(env.current_step)

        # I compute how far the predicted SoC is from the safe zone
        if predicted_soc < 0.15:
            # I am going too low — need positive correction (buy more)
            return (0.15 - predicted_soc) / 0.15  # 0 to 1.0
        elif predicted_soc > 0.85:
            # I am going too high — need negative correction (sell more)
            return -(predicted_soc - 0.85) / 0.15  # -1.0 to 0
        else:
            return 0.0  # No correction needed

    def get_phase(self) -> float:
        """I return IDA phase indicator based on real HEnEx gate closure schedule.

        0.0  = Before any IDA results (hour < 16, IDA1 results ~15:30)
        0.25 = After IDA1 results (16 <= hour < 23)
        0.50 = After IDA2 results (hour >= 23 on D-1 or hour < 10 on D+0)
        0.75 = After IDA3 results (hour >= 10, delivery 14:00-24:00)
        1.0  = In IDA3 delivery window (hour >= 14)
        """
        env = self._env
        hour = env.df.index[env.current_step].hour
        if hour >= 14:
            return 1.0    # In IDA3 delivery window
        elif hour >= 10:
            return 0.75   # After IDA3 results, before delivery
        elif hour >= 0 and hour < 10:
            return 0.50   # After IDA2 (D-1 22:00), before IDA3
        elif hour >= 23:
            return 0.50   # Just after IDA2 results
        elif hour >= 16:
            return 0.25   # After IDA1 results (~15:30)
        else:
            return 0.0    # Before any IDA results
