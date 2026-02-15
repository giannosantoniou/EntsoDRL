"""
Production Metrics Callback for TensorBoard
Logs production-relevant metrics during training.
"""
import os
import logging
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

# I configure a module-level logger for proper error tracking
logger = logging.getLogger(__name__)


class ProductionMetricsCallback(BaseCallback):
    """
    I log production-relevant metrics to TensorBoard every N steps.
    This helps monitor if the agent is learning useful behavior for deployment.
    """

    def __init__(
        self,
        eval_freq: int = 10000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq

        # I track metrics over the evaluation window
        self.reset_trackers()

    def reset_trackers(self):
        """Reset all tracking variables."""
        self.dam_met = 0
        self.dam_total = 0
        self.peak_discharge = 0
        self.peak_total = 0
        self.hourly_profits = defaultdict(list)
        self.trade_count = 0
        self.idle_count = 0
        self.daily_profits = []
        self.current_day_profit = 0
        self.steps_in_window = 0
        self.total_energy = 0
        self.price_action_pairs = []

    def _on_step(self) -> bool:
        """Called at every step."""
        # I get info from the environment
        infos = self.locals.get('infos', [{}])
        if not infos:
            return True

        info = infos[0]

        # Extract metrics from info dict
        power = info.get('actual_mw', 0)
        dam_rev = info.get('dam_rev', 0)
        balancing_rev = info.get('balancing_rev', 0)
        degradation = info.get('degradation', 0)
        shortfall_penalty = info.get('shortfall_penalty', 0)
        soc = info.get('soc', 0.5)
        dam_price = info.get('dam_price', 100)

        # I extract hour from timestamp (resolution-independent — works for 15-min and 1h)
        try:
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                if hasattr(env, 'current_step'):
                    break
                env = env.env

            if hasattr(env, 'df') and hasattr(env, 'current_step') and env.current_step > 0:
                ts = env.df.index[env.current_step - 1]
                hour = ts.hour
                dam_commitment = env.df.iloc[env.current_step - 1].get('dam_commitment', 0)
            else:
                hour = 0
                dam_commitment = 0
        except (AttributeError, IndexError, KeyError) as e:
            # I log the error instead of silently swallowing it
            logger.debug(f"Could not extract hour/DAM from env: {type(e).__name__}: {e}")
            hour = 0
            dam_commitment = 0

        # Calculate profit
        profit = dam_rev + balancing_rev - degradation - shortfall_penalty
        self.current_day_profit += profit

        # Track hourly
        self.hourly_profits[hour].append(profit)

        # DAM Compliance
        if abs(dam_commitment) > 0.1:
            self.dam_total += 1
            if (dam_commitment > 0 and power > 0) or (dam_commitment < 0 and power < 0):
                self.dam_met += 1

        # Peak hours (17-21)
        if 17 <= hour <= 21:
            self.peak_total += 1
            if power > 0.1:
                self.peak_discharge += 1

        # Trade vs Idle
        if abs(power) > 0.1:
            self.trade_count += 1
            self.total_energy += abs(power)
        else:
            self.idle_count += 1

        # Price-action correlation (I limit the buffer to prevent memory growth)
        self.price_action_pairs.append((dam_price, power))
        if len(self.price_action_pairs) > 10000:
            # I keep only the most recent entries to bound memory usage
            self.price_action_pairs = self.price_action_pairs[-5000:]

        # I detect day boundary from timestamps (resolution-independent)
        try:
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                if hasattr(env, 'current_step'):
                    break
                env = env.env
            if hasattr(env, 'df') and hasattr(env, 'current_step') and env.current_step > 1:
                prev_ts = env.df.index[env.current_step - 2]
                curr_ts = env.df.index[env.current_step - 1]
                if curr_ts.date() != prev_ts.date():
                    self.daily_profits.append(self.current_day_profit)
                    self.current_day_profit = 0
        except (AttributeError, IndexError, KeyError):
            pass

        self.steps_in_window += 1

        # Log metrics every eval_freq steps
        if self.num_timesteps % self.eval_freq == 0 and self.steps_in_window > 100:
            self._log_metrics()
            self.reset_trackers()

        return True

    def _log_metrics(self):
        """Log production metrics to TensorBoard."""
        # DAM Compliance
        dam_compliance = self.dam_met / self.dam_total * 100 if self.dam_total > 0 else 0
        self.logger.record("prod/dam_compliance_pct", dam_compliance)

        # Peak Discharge Rate
        peak_rate = self.peak_discharge / self.peak_total * 100 if self.peak_total > 0 else 0
        self.logger.record("prod/peak_discharge_pct", peak_rate)

        # Trading Activity
        total_actions = self.trade_count + self.idle_count
        trade_rate = self.trade_count / total_actions * 100 if total_actions > 0 else 0
        idle_rate = self.idle_count / total_actions * 100 if total_actions > 0 else 0
        self.logger.record("prod/trade_rate_pct", trade_rate)
        self.logger.record("prod/idle_rate_pct", idle_rate)

        # Price Correlation
        if len(self.price_action_pairs) > 10:
            prices, powers = zip(*self.price_action_pairs)
            prices = np.array(prices)
            powers = np.array(powers)
            # I avoid division by zero
            if np.std(prices) > 0 and np.std(powers) > 0:
                corr = np.corrcoef(prices, powers)[0, 1]
                if not np.isnan(corr):
                    self.logger.record("prod/price_correlation", corr)

        # Daily Profit Stats
        if len(self.daily_profits) > 0:
            daily_arr = np.array(self.daily_profits)
            self.logger.record("prod/avg_daily_profit", daily_arr.mean())

            # Win Rate
            win_rate = (daily_arr > 0).sum() / len(daily_arr) * 100
            self.logger.record("prod/win_rate_pct", win_rate)

            # Sharpe (simplified)
            if np.std(daily_arr) > 0:
                sharpe = np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)
                self.logger.record("prod/sharpe_ratio", sharpe)

        # Energy efficiency
        if self.total_energy > 0:
            total_profit = sum(self.daily_profits) if self.daily_profits else 0
            profit_per_mwh = total_profit / self.total_energy
            self.logger.record("prod/profit_per_mwh", profit_per_mwh)

        # Best/Worst hours
        best_hour_profit = -float('inf')
        worst_hour_profit = float('inf')
        for hour, profits in self.hourly_profits.items():
            if profits:
                avg = np.mean(profits)
                if avg > best_hour_profit:
                    best_hour_profit = avg
                if avg < worst_hour_profit:
                    worst_hour_profit = avg

        if best_hour_profit > -float('inf'):
            self.logger.record("prod/best_hour_profit", best_hour_profit)
        if worst_hour_profit < float('inf'):
            self.logger.record("prod/worst_hour_profit", worst_hour_profit)

        if self.verbose > 0:
            print(f"[Prod Metrics] DAM: {dam_compliance:.1f}% | Peak: {peak_rate:.1f}% | Trade: {trade_rate:.1f}%")
