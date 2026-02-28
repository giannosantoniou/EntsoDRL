"""
DAM (Day-Ahead Market) Battery Trading Environment

I implement the DAM model that decides 24 hourly commitments at D-1 12:00.
This is the FIRST model in the cascade: DAM -> aFRR -> IntraDay -> mFRR.

Key Design Decisions:
1. Decision moment: D-1 12:00, agent plans 24 hourly commitments sequentially
2. Episode: 24 steps (hour 0 -> 23), one commitment per step
3. Information: ONLY price FORECASTS (price_lag_24h as naive forecast), NO actuals
4. Action space: Discrete(11) = [-30, -24, ..., 0, ..., +24, +30] MW
5. After episode: the 24 commitments become dam_schedule for downstream models
6. Forward feasibility checking via SoC trajectory simulation

The agent sees ONLY information available at D-1 12:00:
- Historical price patterns (yesterday same hour as forecast proxy)
- Calendar features (hour, day-of-week, month)
- Battery state (SoC, cycles, remaining capacity)
- Partial schedule built so far (hours 0..t-1)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

from gym_envs.shared_battery_state import SharedBatteryState
from gym_envs.decomposed_rewards import DAMRewardCalculator

warnings.filterwarnings('ignore')


# I define 11 DAM commitment levels in MW (~6 MW granularity)
DAM_LEVELS = np.array([-30.0, -24.0, -18.0, -12.0, -6.0, 0.0,
                        6.0, 12.0, 18.0, 24.0, 30.0], dtype=np.float32)


class BatteryEnvDAM(gym.Env):
    """
    DAM Environment: 24-step episodes for hourly day-ahead commitments.

    I decide the full 24-hour DAM schedule using only D-1 information.
    Each step t (0-23) commits MW for hour t of the delivery day.
    """

    metadata = {'render_modes': ['human']}
    N_OBS = 20
    N_ACTIONS = 11  # len(DAM_LEVELS)

    def __init__(
        self,
        df: pd.DataFrame,
        capacity_mwh: float = 146.0,
        max_power_mw: float = 30.0,
        efficiency: float = 0.94,
        min_soc: float = 0.05,
        max_soc: float = 0.95,
        max_daily_cycles: float = 2.0,
        degradation_cost: float = 15.0,
        random_start: bool = True,
        reward_config: Optional[Dict] = None,
    ):
        super().__init__()

        # I store the data and configuration
        self.df = df.copy()
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.eff_sqrt = np.sqrt(efficiency)
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.max_daily_cycles = max_daily_cycles
        self.degradation_cost = degradation_cost
        self.random_start = random_start

        # I prepare the data
        self._prepare_data()

        # I define spaces
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.N_OBS,), dtype=np.float32
        )

        # I create battery state and reward calculator
        self.battery = SharedBatteryState(
            capacity_mwh=capacity_mwh,
            max_power_mw=max_power_mw,
            efficiency=efficiency,
            min_soc=min_soc,
            max_soc=max_soc,
            max_daily_cycles=max_daily_cycles,
        )

        rc = reward_config or {}
        self.reward_calculator = DAMRewardCalculator(
            degradation_cost_per_mwh=rc.get('degradation_cost', degradation_cost),
            reward_scale=rc.get('reward_scale', 0.001),
        )

        # I initialize episode state
        self.current_hour = 0
        self.day_idx = 0
        self.schedule = np.zeros(24, dtype=np.float32)
        self.day_forecast = np.zeros(24, dtype=np.float32)

    def _prepare_data(self) -> None:
        """I prepare and validate the training data."""
        df = self.df

        # I parse datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='mixed', utc=True)
                df.index = df.index.tz_convert('Europe/Athens')
            except Exception:
                df.index = pd.to_datetime(df.index, format='mixed')

        # I validate required columns
        if 'price' not in df.columns:
            raise ValueError("Missing required column: 'price'")

        # I detect time resolution
        if len(df) > 1:
            time_diff = (df.index[1] - df.index[0]).total_seconds() / 3600
            self._sph = max(1, int(round(1.0 / time_diff)))
        else:
            self._sph = 4  # default 15-min

        # I add time features
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek
        if 'month' not in df.columns:
            df['month'] = df.index.month

        # I create naive forecast: yesterday same hour as price proxy
        # This is the ONLY price information available at D-1 12:00
        w24 = 24 * self._sph
        if 'price_lag_24h' not in df.columns:
            df['price_lag_24h'] = df['price'].shift(w24)

        # LEAKAGE FIX: I mark the first w24 rows as INVALID for forecasting.
        # shift(w24) produces NaN for the first w24 rows. Using bfill() would
        # contaminate these rows with FUTURE prices. Instead, I fill them with
        # the dataset-wide mean (a non-informative prior) and exclude them
        # from episode starts below.
        price_mean = df['price'].mean()
        df['price_lag_24h'] = df['price_lag_24h'].fillna(price_mean)

        # I create rolling statistics for forecast context (backward-looking)
        if 'price_mean_7d' not in df.columns:
            w7d = 7 * 24 * self._sph
            df['price_mean_7d'] = df['price'].rolling(w7d, min_periods=1).mean()
        if 'price_std_7d' not in df.columns:
            w7d = 7 * 24 * self._sph
            df['price_std_7d'] = df['price'].rolling(w7d, min_periods=1).std().fillna(10.0)

        # I fill remaining NaN values (NOT price_lag_24h, already handled above)
        df = df.ffill().bfill()
        self.df = df

        # I identify valid day start indices (midnight boundaries)
        # LEAKAGE FIX: I skip the first w24 rows where price_lag_24h was
        # filled with non-informative mean instead of actual lagged prices
        warmup = w24  # first 24h have no valid lag
        self.day_starts = []
        for i in range(warmup, len(df) - 24 * self._sph, self._sph):
            if df.index[i].hour == 0 and df.index[i].minute == 0:
                self.day_starts.append(i)

        # I fallback: if no midnight boundaries, use any 24h window after warmup
        if len(self.day_starts) == 0:
            step = 24 * self._sph
            self.day_starts = list(range(warmup, len(df) - step, step))

        if len(self.day_starts) == 0:
            self.day_starts = [warmup]

    def reset(self, seed=None, options=None):
        """I reset the environment for a new 24-step DAM episode."""
        super().reset(seed=seed)

        # I pick a random day (or sequential)
        if self.random_start:
            self.day_idx = self.np_random.integers(0, len(self.day_starts))
        else:
            self.day_idx = (self.day_idx + 1) % len(self.day_starts)

        day_start = self.day_starts[self.day_idx]

        # I randomize initial SoC for exploration
        if self.random_start:
            initial_soc = self.np_random.uniform(0.2, 0.8)
        else:
            initial_soc = 0.5

        self.battery.reset(soc=initial_soc)
        self.current_hour = 0
        self.schedule = np.zeros(24, dtype=np.float32)
        self._day_start = day_start

        # I extract 24-hour forecast (D-1 prices = yesterday same hour)
        # LEAKAGE FIX: I NEVER fall back to actual 'price' column.
        # If price_lag_24h is unavailable, I use the dataset-wide mean
        # (non-informative prior) to prevent the agent from seeing actuals.
        self.day_forecast = np.zeros(24, dtype=np.float32)
        price_mean = float(self.df['price'].mean()) if 'price' in self.df.columns else 80.0
        for h in range(24):
            row_idx = day_start + h * self._sph
            if row_idx < len(self.df):
                self.day_forecast[h] = self.df.iloc[row_idx].get(
                    'price_lag_24h', price_mean
                )
            else:
                self.day_forecast[h] = price_mean

        obs = self._build_observation()
        return obs, {}

    def step(self, action: int):
        """
        I execute one step: commit MW for hour self.current_hour.

        Args:
            action: Index into DAM_LEVELS (0-10)

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        h = self.current_hour
        action_mw = DAM_LEVELS[action]

        # I store the commitment
        self.schedule[h] = action_mw

        # I simulate the SoC change from this commitment (1 hour duration)
        self.battery.apply_energy(action_mw, 1.0)

        # I check if this is the last step
        is_last = (h == 23)

        # I calculate planned revenue and flexibility for end-of-episode bonus
        total_planned_revenue = 0.0
        remaining_capacity_frac = 1.0
        if is_last:
            total_planned_revenue = float(np.sum(self.schedule * self.day_forecast))
            max_commitment = np.max(np.abs(self.schedule))
            remaining_capacity_frac = max(0.0, 1.0 - max_commitment / self.max_power_mw)

        # I calculate reward
        reward, components = self.reward_calculator.calculate(
            action_mw=action_mw,
            forecast_price=self.day_forecast[h],
            soc=self.battery.soc,
            capacity_mwh=self.capacity_mwh,
            hour=h,
            is_last_step=is_last,
            total_planned_revenue=total_planned_revenue,
            remaining_capacity_frac=remaining_capacity_frac,
            daily_cycles=self.battery.daily_cycles,
            max_daily_cycles=self.max_daily_cycles,
        )

        # I advance to next hour
        self.current_hour += 1
        terminated = (self.current_hour >= 24)
        truncated = False

        info = {
            'hour': h,
            'action_mw': action_mw,
            'forecast_price': self.day_forecast[h],
            'soc': self.battery.soc,
            'daily_cycles': self.battery.daily_cycles,
            **{f'r_{k}': v for k, v in components.items()},
        }

        if terminated:
            info['dam_schedule'] = self.schedule.copy()
            info['total_planned_revenue'] = total_planned_revenue

        obs = self._build_observation() if not terminated else np.zeros(self.N_OBS, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """
        I build the 20-feature observation for the current hour.

        Features are ONLY those available at D-1 12:00 (no actual prices).
        """
        h = self.current_hour
        features = []
        row_idx = self._day_start + h * self._sph
        row_idx = min(row_idx, len(self.df) - 1)
        row = self.df.iloc[row_idx]

        # [0] SoC (accounting for steps 0..h-1)
        features.append(self.battery.soc)

        # [1-2] max discharge/charge remaining (normalized by max_power)
        max_d, max_c = self.battery.remaining_capacity('dam', hour=h)
        features.append(max_d / self.max_power_mw)
        features.append(max_c / self.max_power_mw)

        # [3-4] hour sin/cos encoding
        features.append(np.sin(2 * np.pi * h / 24))
        features.append(np.cos(2 * np.pi * h / 24))

        # [5-6] day-of-week sin/cos
        dow = row.get('day_of_week', 0)
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # [7-10] DAM forecast prices (h, h+1, h+2, h+3)
        # I use price_lag_24h as naive forecast (NOT actual)
        for offset in range(4):
            fh = h + offset
            if fh < 24:
                features.append(self.day_forecast[fh] / 100.0)
            else:
                features.append(self.day_forecast[23] / 100.0)

        # [11-12] forecast daily mean and std (from rolling 7d same-dow)
        features.append(row.get('price_mean_7d', 80.0) / 100.0)
        features.append(row.get('price_std_7d', 20.0) / 100.0)

        # [13] is_peak_hour (17:00-21:00)
        features.append(1.0 if 17 <= h <= 21 else 0.0)

        # [14] is_solar_hour (10:00-15:00)
        features.append(1.0 if 10 <= h <= 15 else 0.0)

        # [15] net committed energy so far / capacity
        committed_energy = np.sum(self.schedule[:h]) * 1.0  # MWh (1h per step)
        features.append(committed_energy / self.capacity_mwh)

        # [16] SoC headroom (min of charge/discharge room)
        charge_room = self.max_soc - self.battery.soc
        discharge_room = self.battery.soc - self.min_soc
        features.append(min(charge_room, discharge_room))

        # [17] daily cycles / max daily cycles
        features.append(self.battery.daily_cycles / self.max_daily_cycles)

        # [18] month sin encoding
        month = row.get('month', 1)
        features.append(np.sin(2 * np.pi * month / 12))

        # [19] schedule feasibility flag (forward simulation)
        _, is_feasible = self.battery.simulate_soc_trajectory(
            self.schedule[:h + 1], self.battery.soc, 1.0
        )
        features.append(1.0 if is_feasible else 0.0)

        obs = np.array(features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        assert len(obs) == self.N_OBS, f"Expected {self.N_OBS} features, got {len(obs)}"
        return obs

    def action_masks(self) -> np.ndarray:
        """
        I return action masks ensuring forward SoC feasibility.

        For each DAM level, I check if committing that MW for this hour
        keeps the SoC trajectory feasible for the remaining hours.
        """
        mask = np.ones(self.N_ACTIONS, dtype=bool)
        h = self.current_hour

        for i, level in enumerate(DAM_LEVELS):
            if abs(level) < 0.01:
                continue  # idle always valid

            # I build a test schedule: committed hours + this action
            test_schedule = self.schedule[:h].copy()
            test_schedule = np.append(test_schedule, level)

            # I simulate SoC trajectory
            _, is_feasible = self.battery.simulate_soc_trajectory(
                test_schedule, self.battery.soc, 1.0
            )

            if not is_feasible:
                mask[i] = False

            # I also check basic power/SoC constraints
            if level > 0:  # discharge
                max_d, _ = self.battery.remaining_capacity('dam', hour=h)
                if level > max_d + 0.1:
                    mask[i] = False
            elif level < 0:  # charge
                _, max_c = self.battery.remaining_capacity('dam', hour=h)
                if abs(level) > max_c + 0.1:
                    mask[i] = False

        # I always allow idle
        mask[5] = True  # index 5 = 0 MW
        return mask

    def get_dam_schedule(self) -> np.ndarray:
        """I return the completed 24-hour schedule after episode ends."""
        return self.schedule.copy()
