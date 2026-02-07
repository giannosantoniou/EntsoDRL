"""
Realistic Battery Trading Environment - NO DATA LEAKAGE

Key differences from original:
1. NO future price lookahead (only lagged prices)
2. NO max_future_price in rewards (unknown at decision time)
3. Time-based features only (hour, day_of_week)
4. Lagged price features (1h, 4h, 24h ago)

This environment trains agents that can actually work in production!
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BatteryEnvRealistic(gym.Env):
    """
    Realistic battery trading environment for mFRR market.

    Observation space (13 features):
    0. soc - State of charge [0, 1]
    1. max_discharge_norm - Max discharge power normalized
    2. max_charge_norm - Max charge power normalized
    3. hour_sin - Sine of hour (cyclical)
    4. hour_cos - Cosine of hour (cyclical)
    5. dow_sin - Sine of day of week
    6. dow_cos - Cosine of day of week
    7. is_weekend - Weekend indicator
    8. price_lag_1h - Price 1 hour ago (normalized)
    9. price_lag_4h - Price 4 hours ago (normalized)
    10. price_lag_24h - Price 24 hours ago (normalized)
    11. price_rolling_mean_24h - 24h rolling mean (backward-looking)
    12. dam_commitment - Current DAM commitment (normalized)

    Action space: Discrete(21)
    - 0-9: Charge at 100% to 10% of max power
    - 10: Idle
    - 11-20: Discharge at 10% to 100% of max power
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        capacity_mwh: float = 146.0,
        max_power_mw: float = 30.0,
        efficiency: float = 0.94,
        degradation_cost: float = 15.0,
        min_soc: float = 0.05,
        max_soc: float = 0.95,
        time_step_hours: float = 1.0,
        n_actions: int = 21,
        reward_scale: float = 0.001,
    ):
        super().__init__()

        self.df = df.copy()
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.eff_sqrt = np.sqrt(efficiency)
        self.degradation_cost = degradation_cost
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.time_step_hours = time_step_hours
        self.n_actions = n_actions
        self.reward_scale = reward_scale

        # Precompute lagged features if not present
        self._prepare_features()

        # Action levels: -1.0 (full charge) to +1.0 (full discharge)
        self.action_levels = np.linspace(-1.0, 1.0, n_actions)

        # Skip first 24 rows (need lag_24h)
        self.start_step = 24
        self.max_steps = len(self.df) - 1
        self.current_step = self.start_step

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

        # State
        self.soc = 0.5
        self.total_profit = 0.0
        self.total_cycles = 0.0

        print(f"Realistic Environment: {n_actions} actions, {len(df)} rows")
        print(f"  NO future price lookahead (realistic)")
        print(f"  Uses only lagged prices and time features")

    def _prepare_features(self):
        """I precompute lagged features for realistic training."""
        df = self.df

        # Ensure we have a price column
        if 'price' not in df.columns:
            if 'mfrr_price_up' in df.columns:
                df['price'] = df['mfrr_price_up']
            else:
                raise ValueError("No price column found")

        # Lagged prices (BACKWARD-LOOKING ONLY)
        if 'price_lag_1h' not in df.columns:
            df['price_lag_1h'] = df['price'].shift(1)
        if 'price_lag_4h' not in df.columns:
            df['price_lag_4h'] = df['price'].shift(4)
        if 'price_lag_24h' not in df.columns:
            df['price_lag_24h'] = df['price'].shift(24)

        # Rolling mean (BACKWARD-LOOKING: shift then roll)
        if 'price_rolling_mean_24h' not in df.columns:
            df['price_rolling_mean_24h'] = df['price'].shift(1).rolling(24).mean()

        # Fill NaN with forward fill then backward fill
        for col in ['price_lag_1h', 'price_lag_4h', 'price_lag_24h', 'price_rolling_mean_24h']:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        self.df = df

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Random starting position (after lag period)
        self.current_step = self.np_random.integers(
            self.start_step,
            self.max_steps - 100  # Leave room for episode
        )

        # Random starting SoC
        self.soc = self.np_random.uniform(0.3, 0.7)

        # Reset tracking
        self.total_profit = 0.0
        self.total_cycles = 0.0

        obs = self._get_observation()
        info = {'soc': self.soc, 'step': self.current_step}

        return obs, info

    def _get_observation(self) -> np.ndarray:
        """
        Build observation using ONLY information available at decision time.
        NO FUTURE DATA!
        """
        row = self.df.iloc[self.current_step]
        ts = self.df.index[self.current_step]

        # Calculate max discharge/charge based on SoC
        available_energy = (self.soc - self.min_soc) * self.capacity_mwh
        available_headroom = (self.max_soc - self.soc) * self.capacity_mwh

        max_discharge = min(
            self.max_power_mw,
            available_energy * self.eff_sqrt / self.time_step_hours
        )
        max_charge = min(
            self.max_power_mw,
            available_headroom / self.eff_sqrt / self.time_step_hours
        )

        # Time features (always known)
        hour = ts.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        dow = ts.dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        is_weekend = 1.0 if dow >= 5 else 0.0

        # Lagged prices (PAST ONLY - no leakage)
        price_lag_1h = row.get('price_lag_1h', 100.0) / 100.0
        price_lag_4h = row.get('price_lag_4h', 100.0) / 100.0
        price_lag_24h = row.get('price_lag_24h', 100.0) / 100.0
        price_rolling = row.get('price_rolling_mean_24h', 100.0) / 100.0

        # DAM commitment (known from day-ahead)
        dam_commitment = row.get('dam_commitment', 0.0) / self.max_power_mw

        obs = np.array([
            self.soc,
            max_discharge / self.max_power_mw,
            max_charge / self.max_power_mw,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            is_weekend,
            price_lag_1h,
            price_lag_4h,
            price_lag_24h,
            price_rolling,
            dam_commitment,
        ], dtype=np.float32)

        return obs

    def action_masks(self) -> np.ndarray:
        """Return valid actions based on SoC constraints."""
        mask = np.zeros(self.n_actions, dtype=bool)

        available_energy = (self.soc - self.min_soc) * self.capacity_mwh
        available_headroom = (self.max_soc - self.soc) * self.capacity_mwh

        max_discharge_frac = min(1.0, available_energy * self.eff_sqrt /
                                  (self.max_power_mw * self.time_step_hours))
        max_charge_frac = min(1.0, available_headroom / self.eff_sqrt /
                               (self.max_power_mw * self.time_step_hours))

        for i, level in enumerate(self.action_levels):
            if level >= 0:  # Discharge
                mask[i] = level <= max_discharge_frac + 0.01
            else:  # Charge
                mask[i] = abs(level) <= max_charge_frac + 0.01

        # Always allow idle
        mask[self.n_actions // 2] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        row = self.df.iloc[self.current_step]

        # Get action level
        action_level = self.action_levels[action]
        power_mw = action_level * self.max_power_mw

        # Get CURRENT price (revealed at settlement, not before!)
        # In mFRR market, you bid before knowing the settlement price
        # But settlement happens at this price
        if power_mw > 0:  # Discharge
            price = row.get('mfrr_price_up', row.get('price', 100.0))
        elif power_mw < 0:  # Charge
            price = row.get('mfrr_price_down', row.get('price', 100.0))
        else:
            price = 0.0

        # Calculate energy and update SoC
        old_soc = self.soc

        if power_mw > 0:  # Discharge
            energy_mwh = power_mw * self.time_step_hours
            soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
            self.soc = max(self.min_soc, self.soc - soc_delta)
            actual_energy = (old_soc - self.soc) * self.capacity_mwh * self.eff_sqrt
            revenue = actual_energy * price
        elif power_mw < 0:  # Charge
            energy_mwh = abs(power_mw) * self.time_step_hours
            soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
            self.soc = min(self.max_soc, self.soc + soc_delta)
            actual_energy = (self.soc - old_soc) * self.capacity_mwh / self.eff_sqrt
            revenue = -actual_energy * price  # Cost
        else:
            actual_energy = 0.0
            revenue = 0.0

        # Calculate degradation cost
        cycle_fraction = abs(actual_energy) / self.capacity_mwh
        degradation = cycle_fraction * self.capacity_mwh * self.degradation_cost
        self.total_cycles += cycle_fraction

        # Net profit
        profit = revenue - degradation
        self.total_profit += profit

        # Reward (scaled for RL)
        reward = profit * self.reward_scale

        # Add small bonus for smart timing based on PAST data only
        hour = self.df.index[self.current_step].hour
        if power_mw > 0 and 17 <= hour <= 21:  # Discharge during evening peak
            reward += 0.1 * abs(action_level)
        elif power_mw < 0 and 9 <= hour <= 15:  # Charge during solar hours
            reward += 0.1 * abs(action_level)

        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation()
        info = {
            'soc': self.soc,
            'power_mw': power_mw,
            'price': price,
            'profit': profit,
            'total_profit': self.total_profit,
            'cycles': self.total_cycles,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render current state."""
        print(f"Step {self.current_step}: SoC={self.soc:.2%}, "
              f"Profit={self.total_profit:,.0f} EUR, Cycles={self.total_cycles:.1f}")
