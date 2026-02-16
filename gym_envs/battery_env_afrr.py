"""
aFRR (automatic Frequency Restoration Reserve) Battery Trading Environment

I implement the aFRR model that bids capacity for 4-hour blocks.
This is the SECOND model in the cascade: DAM -> aFRR -> IntraDay -> mFRR.

Key Design Decisions:
1. Decision moment: Before each 4h block (6x/day)
2. Episode: 42 steps (7 days x 6 blocks/day)
3. Information: DAM schedule + actual prices KNOWN, lagged aFRR prices
4. Action space: MultiDiscrete([5, 5]) = 25 combos (capacity + price tier)
5. Internal simulation: 16 ISPs per block for stochastic activation
6. The agent commits capacity and a price tier, TSO decides activation

Information Barriers:
- SEES: DAM schedule (known from D-1 14:00), DAM clearing prices, lagged aFRR,
        SoC after DAM execution, system imbalance trend
- DOESN'T SEE: Future aFRR activations, IntraDay/mFRR prices for this block
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

from gym_envs.shared_battery_state import SharedBatteryState, UpstreamSimulator
from gym_envs.decomposed_rewards import AFRRRewardCalculator

warnings.filterwarnings('ignore')


# I define aFRR commitment levels (fraction of remaining capacity after DAM)
AFRR_CAP_LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

# I define aFRR price tier multipliers (bid aggressiveness)
# Lower = more aggressive (higher selection probability)
AFRR_PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3], dtype=np.float32)


class BatteryEnvAFRR(gym.Env):
    """
    aFRR Environment: 42-step episodes for 4-hour block capacity bids.

    I decide aFRR capacity commitment and price tier for each 4h block.
    A 7-day episode has 6 blocks/day = 42 steps total.
    """

    metadata = {'render_modes': ['human']}
    N_OBS = 20
    BLOCKS_PER_DAY = 6
    HOURS_PER_BLOCK = 4
    EPISODE_DAYS = 7
    EPISODE_LENGTH = BLOCKS_PER_DAY * EPISODE_DAYS  # 42

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
        # aFRR simulation parameters
        base_selection_prob: float = 0.80,
        base_activation_rate: float = 0.15,
        peak_activation_boost: float = 1.3,
        high_imbalance_boost: float = 1.5,
        # Upstream source
        upstream_simulator: Optional[UpstreamSimulator] = None,
        reward_config: Optional[Dict] = None,
    ):
        super().__init__()

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

        # aFRR simulation parameters
        self.base_selection_prob = base_selection_prob
        self.base_activation_rate = base_activation_rate
        self.peak_activation_boost = peak_activation_boost
        self.high_imbalance_boost = high_imbalance_boost

        # I use upstream simulator for DAM schedule
        self.upstream = upstream_simulator or UpstreamSimulator(use_data_fallback=True)

        # I prepare data
        self._prepare_data()

        # I define spaces
        self.action_space = spaces.MultiDiscrete([
            len(AFRR_CAP_LEVELS),   # 5 capacity levels
            len(AFRR_PRICE_TIERS),  # 5 price tiers
        ])
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
        self.reward_calculator = AFRRRewardCalculator(
            degradation_cost_per_mwh=rc.get('degradation_cost', degradation_cost),
            nonresponse_penalty=rc.get('nonresponse_penalty', 2000.0),
            opportunity_cost_weight=rc.get('opportunity_cost_weight', 0.3),
            reward_scale=rc.get('reward_scale', 0.001),
        )

        # I initialize episode state
        self.current_step = 0
        self._episode_start_idx = 0

    def _prepare_data(self) -> None:
        """I prepare and validate training data for aFRR environment."""
        df = self.df

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='mixed', utc=True)
                df.index = df.index.tz_convert('Europe/Athens')
            except Exception:
                df.index = pd.to_datetime(df.index, format='mixed')

        if 'price' not in df.columns:
            raise ValueError("Missing required column: 'price'")

        # I detect time resolution
        if len(df) > 1:
            time_diff = (df.index[1] - df.index[0]).total_seconds() / 3600
            self._sph = max(1, int(round(1.0 / time_diff)))
        else:
            self._sph = 4

        self._time_step_hours = 1.0 / self._sph

        # I add defaults for missing columns
        defaults = {
            'afrr_cap_up_price': 15.0,
            'afrr_cap_down_price': 12.0,
            'afrr_up': 80.0,
            'afrr_down': 80.0,
            'net_imbalance_mw': 0.0,
            'dam_commitment': 0.0,
            'hour': df.index.hour if hasattr(df.index, 'hour') else 0,
            'day_of_week': df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        # I create lagged features (backward-looking only)
        w_block = self.HOURS_PER_BLOCK * self._sph  # 16 ISPs per block
        for col in ['afrr_cap_up_price', 'afrr_cap_down_price', 'afrr_up']:
            lag_col = f'{col}_lag_1block'
            if lag_col not in df.columns:
                df[lag_col] = df[col].shift(w_block)

        # I create rolling statistics
        w24 = 24 * self._sph
        if 'afrr_cap_percentile' not in df.columns:
            w7d = 7 * 24 * self._sph
            rolling_mean = df['afrr_cap_up_price'].rolling(w7d, min_periods=1).mean()
            rolling_std = df['afrr_cap_up_price'].rolling(w7d, min_periods=1).std().fillna(1.0)
            df['afrr_cap_percentile'] = np.clip(
                (df['afrr_cap_up_price'] - rolling_mean) / (rolling_std + 1.0) * 0.5 + 0.5,
                0.0, 1.0
            )

        if 'price_mean_24h' not in df.columns:
            df['price_mean_24h'] = df['price'].rolling(w24, min_periods=1).mean()

        # I fill NaN
        df = df.ffill().bfill()
        self.df = df

        # I calculate valid episode start indices
        # Each episode = 7 days = 7*24*sph rows
        episode_rows = self.EPISODE_DAYS * 24 * self._sph
        warmup = 48 * self._sph
        self._max_start = len(df) - episode_rows - 1
        self._warmup = warmup

    def reset(self, seed=None, options=None):
        """I reset for a new 42-step aFRR episode (7 days)."""
        super().reset(seed=seed)

        # I pick episode start
        if self.random_start:
            self._episode_start_idx = self.np_random.integers(
                self._warmup, max(self._warmup + 1, self._max_start)
            )
        else:
            self._episode_start_idx = self._warmup

        # I randomize initial SoC
        if self.random_start:
            initial_soc = self.np_random.uniform(0.2, 0.8)
        else:
            initial_soc = 0.5

        self.battery.reset(soc=initial_soc)
        self.current_step = 0

        # I load DAM schedule for the first day
        self._load_dam_schedule_for_day(0)

        # I simulate DAM execution up to the first block
        self._execute_dam_to_block(0)

        obs = self._build_observation()
        return obs, {}

    def _load_dam_schedule_for_day(self, day_offset: int) -> None:
        """I load DAM schedule for a specific day in the episode."""
        day_start = self._episode_start_idx + day_offset * 24 * self._sph
        self.battery.dam_schedule = self.upstream.get_dam_schedule(
            self.df, day_start, self._sph
        )

    def _execute_dam_to_block(self, block_in_day: int) -> None:
        """
        I simulate DAM execution from midnight up to the start of this block.
        This updates SoC to reflect mandatory DAM commitments already executed.
        """
        end_hour = block_in_day * self.HOURS_PER_BLOCK
        for h in range(end_hour):
            dam_mw = self.battery.dam_schedule[h]
            if abs(dam_mw) > 0.01:
                self.battery.apply_energy(dam_mw, 1.0)

    def _get_block_data_idx(self) -> int:
        """I return the DataFrame index for the current block's start."""
        day_in_episode = self.current_step // self.BLOCKS_PER_DAY
        block_in_day = self.current_step % self.BLOCKS_PER_DAY
        block_start_hour = block_in_day * self.HOURS_PER_BLOCK
        return (self._episode_start_idx +
                day_in_episode * 24 * self._sph +
                block_start_hour * self._sph)

    def step(self, action):
        """
        I execute one step: commit aFRR capacity for a 4h block.

        Args:
            action: [capacity_idx, price_tier_idx]

        Returns:
            (obs, reward, terminated, truncated, info)
        """
        cap_idx, price_idx = action
        block_idx = self._get_block_data_idx()
        block_in_day = self.current_step % self.BLOCKS_PER_DAY
        block_start_hour = block_in_day * self.HOURS_PER_BLOCK

        # I calculate remaining capacity after DAM
        remaining_d, remaining_c = self.battery.remaining_capacity(
            'afrr', hour=block_start_hour
        )
        remaining = min(remaining_d, remaining_c)

        # I determine commitment MW
        commitment_mw = AFRR_CAP_LEVELS[cap_idx] * remaining
        price_tier = AFRR_PRICE_TIERS[price_idx]

        # I store commitment in battery state
        self.battery.afrr_commitments[block_in_day] = commitment_mw
        self.battery.afrr_price_tiers[block_in_day] = price_idx

        # I simulate the 4h block (selection + activation)
        was_selected, was_activated, activation_energy, energy_price, shortfall = \
            self._simulate_block(commitment_mw, price_tier, block_idx,
                                 block_start_hour)

        # I get capacity clearing price
        if block_idx < len(self.df):
            cap_price = self.df.iloc[min(block_idx, len(self.df) - 1)].get(
                'afrr_cap_up_price', 15.0
            )
        else:
            cap_price = 15.0

        # I get lagged IntraDay spread for opportunity cost
        lagged_id_spread = 4.0  # default
        if block_idx < len(self.df):
            bid = self.df.iloc[min(block_idx, len(self.df) - 1)].get('intraday_bid', 98.0)
            ask = self.df.iloc[min(block_idx, len(self.df) - 1)].get('intraday_ask', 102.0)
            lagged_id_spread = max(0.0, ask - bid)

        # I calculate reward
        reward, components = self.reward_calculator.calculate(
            commitment_mw=commitment_mw,
            cap_clearing_price=cap_price,
            was_selected=was_selected,
            was_activated=was_activated,
            activation_energy_mwh=activation_energy,
            energy_price=energy_price,
            shortfall_mw=shortfall,
            capacity_mwh=self.capacity_mwh,
            lagged_id_spread=lagged_id_spread,
            block_hours=float(self.HOURS_PER_BLOCK),
        )

        # I advance to next block
        self.current_step += 1

        # I handle day transitions
        new_block_in_day = self.current_step % self.BLOCKS_PER_DAY
        new_day = self.current_step // self.BLOCKS_PER_DAY
        if new_block_in_day == 0 and self.current_step < self.EPISODE_LENGTH:
            # I crossed midnight - load new DAM schedule
            self.battery.check_day_boundary(0, 23)
            self._load_dam_schedule_for_day(new_day)
        elif self.current_step < self.EPISODE_LENGTH:
            # I execute DAM for the hours in the just-completed block
            for h in range(block_start_hour, block_start_hour + self.HOURS_PER_BLOCK):
                if h < 24:
                    dam_mw = self.battery.dam_schedule[h]
                    if abs(dam_mw) > 0.01:
                        self.battery.apply_energy(dam_mw, 1.0)

        terminated = (self.current_step >= self.EPISODE_LENGTH)
        truncated = False

        info = {
            'block': self.current_step - 1,
            'commitment_mw': commitment_mw,
            'price_tier': float(price_tier),
            'was_selected': was_selected,
            'was_activated': was_activated,
            'activation_energy': activation_energy,
            'shortfall': shortfall,
            'soc': self.battery.soc,
            **{f'r_{k}': v for k, v in components.items()},
        }

        obs = self._build_observation() if not terminated else np.zeros(self.N_OBS, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def _simulate_block(
        self, commitment_mw: float, price_tier: float,
        block_start_idx: int, block_start_hour: int
    ) -> Tuple[bool, bool, float, float, float]:
        """
        I simulate a 4h aFRR block with stochastic selection and activation.

        Returns:
            (was_selected, was_activated, activation_energy_mwh, energy_price, shortfall_mw)
        """
        if commitment_mw < 0.1:
            return False, False, 0.0, 0.0, 0.0

        # I calculate selection probability (aggressive bid = higher selection)
        selection_prob = self.base_selection_prob * (1.3 - price_tier) / 0.6
        selection_prob = np.clip(selection_prob, 0.3, 0.95)
        was_selected = self.np_random.random() < selection_prob

        if not was_selected:
            return False, False, 0.0, 0.0, 0.0

        # I simulate activation across 16 ISPs in the block
        isps_per_block = self.HOURS_PER_BLOCK * self._sph
        total_activation_energy = 0.0
        total_shortfall = 0.0
        activated_any = False
        energy_price_sum = 0.0
        activation_count = 0

        for isp in range(isps_per_block):
            idx = block_start_idx + isp
            if idx >= len(self.df):
                break

            row = self.df.iloc[idx]

            # I determine activation probability
            activation_prob = self.base_activation_rate
            imbalance = abs(row.get('net_imbalance_mw', 0.0))
            hour = block_start_hour + isp // self._sph

            if imbalance > 500:
                activation_prob *= self.high_imbalance_boost
            if 17 <= hour <= 21:
                activation_prob *= self.peak_activation_boost

            if self.np_random.random() < activation_prob:
                activated_any = True
                activation_count += 1

                # I determine direction from imbalance
                net_imb = row.get('net_imbalance_mw', 0.0)
                if net_imb > 0:
                    direction = 'up'  # discharge
                else:
                    direction = 'down'  # charge

                # I execute activation (15-min ISP)
                isp_duration = self._time_step_hours
                if direction == 'up':
                    actual = self.battery.apply_energy(commitment_mw, isp_duration)
                    total_activation_energy += actual
                    e_price = row.get('afrr_up', 80.0)
                else:
                    actual = self.battery.apply_energy(-commitment_mw, isp_duration)
                    total_activation_energy += actual
                    e_price = row.get('afrr_down', 80.0)

                energy_price_sum += e_price

                # I check shortfall
                expected = commitment_mw * isp_duration
                if abs(actual) < expected - 0.01:
                    total_shortfall += expected - abs(actual)

        avg_energy_price = energy_price_sum / max(activation_count, 1)

        self.battery.afrr_is_selected = was_selected
        self.battery.afrr_is_activated = activated_any
        self.battery.afrr_activated_mw = commitment_mw if activated_any else 0.0

        return was_selected, activated_any, total_activation_energy, avg_energy_price, total_shortfall

    def _build_observation(self) -> np.ndarray:
        """I build the 20-feature observation for the current block."""
        features = []
        block_idx = self._get_block_data_idx()
        block_idx = min(block_idx, len(self.df) - 1)
        row = self.df.iloc[block_idx]
        block_in_day = self.current_step % self.BLOCKS_PER_DAY
        block_start_hour = block_in_day * self.HOURS_PER_BLOCK

        # [0] SoC after DAM execution up to this block
        features.append(self.battery.soc)

        # [1] remaining capacity after DAM / max_power
        remaining_d, remaining_c = self.battery.remaining_capacity(
            'afrr', hour=block_start_hour
        )
        features.append(min(remaining_d, remaining_c) / self.max_power_mw)

        # [2-3] max discharge, max charge after DAM
        features.append(remaining_d / self.max_power_mw)
        features.append(remaining_c / self.max_power_mw)

        # [4-5] block sin/cos (block_in_day / 6)
        features.append(np.sin(2 * np.pi * block_in_day / self.BLOCKS_PER_DAY))
        features.append(np.cos(2 * np.pi * block_in_day / self.BLOCKS_PER_DAY))

        # [6-7] hour sin/cos (block start hour)
        features.append(np.sin(2 * np.pi * block_start_hour / 24))
        features.append(np.cos(2 * np.pi * block_start_hour / 24))

        # [8-9] day-of-week sin/cos
        dow = row.get('day_of_week', 0)
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # [10-11] aFRR capacity prices (LAGGED by 1 block)
        features.append(row.get('afrr_cap_up_price_lag_1block',
                                row.get('afrr_cap_up_price', 15.0)) / 50.0)
        features.append(row.get('afrr_cap_down_price_lag_1block',
                                row.get('afrr_cap_down_price', 12.0)) / 50.0)

        # [12] aFRR capacity price percentile (rolling)
        features.append(row.get('afrr_cap_percentile', 0.5))

        # [13] DAM commitment block average (known from D-1)
        dam_block_avg = 0.0
        for h in range(block_start_hour, min(block_start_hour + 4, 24)):
            dam_block_avg += self.battery.dam_schedule[h]
        dam_block_avg /= self.HOURS_PER_BLOCK
        features.append(dam_block_avg / self.max_power_mw)

        # [14] DAM absolute energy in this block
        dam_energy = 0.0
        for h in range(block_start_hour, min(block_start_hour + 4, 24)):
            dam_energy += abs(self.battery.dam_schedule[h])
        features.append(dam_energy / (self.max_power_mw * self.HOURS_PER_BLOCK))

        # [15] DAM price block average (actual, known post D-1 14:00)
        features.append(row.get('price', 80.0) / 100.0)

        # [16-17] net imbalance lag, system stress
        imbalance = row.get('net_imbalance_mw', 0.0)
        features.append(imbalance / 1000.0)
        features.append(min(abs(imbalance) / 500.0, 1.0))  # system stress

        # [18] is peak block (block 4 = 16:00-20:00)
        features.append(1.0 if block_in_day == 4 else 0.0)

        # [19] daily cycles / max
        features.append(self.battery.daily_cycles / self.max_daily_cycles)

        obs = np.array(features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        assert len(obs) == self.N_OBS, f"Expected {self.N_OBS} features, got {len(obs)}"
        return obs

    def action_masks(self) -> np.ndarray:
        """
        I return action masks for [capacity, price_tier].

        I mask out capacity levels that can't sustain FULL 4h activation
        in BOTH directions (worst case). Price tiers are always valid.
        """
        block_in_day = self.current_step % self.BLOCKS_PER_DAY
        block_start_hour = block_in_day * self.HOURS_PER_BLOCK

        remaining_d, remaining_c = self.battery.remaining_capacity(
            'afrr', hour=block_start_hour
        )
        remaining = min(remaining_d, remaining_c)

        cap_mask = np.ones(len(AFRR_CAP_LEVELS), dtype=bool)
        for i, level in enumerate(AFRR_CAP_LEVELS):
            if level < 0.01:
                continue  # 0% always valid

            commitment_mw = level * remaining

            # I must sustain FULL 4h activation in BOTH directions (worst case)
            # Worst case discharge: commitment * 4h / eff_sqrt
            discharge_energy = commitment_mw * self.HOURS_PER_BLOCK / self.eff_sqrt
            # Worst case charge: commitment * 4h * eff_sqrt
            charge_energy = commitment_mw * self.HOURS_PER_BLOCK * self.eff_sqrt

            # I also account for DAM energy in this block
            dam_energy_block = 0.0
            for h in range(block_start_hour, min(block_start_hour + 4, 24)):
                dam_energy_block += self.battery.dam_schedule[h] * 1.0  # MWh

            soc_after_discharge = (
                self.battery.soc -
                (discharge_energy + max(dam_energy_block, 0)) / self.capacity_mwh
            )
            soc_after_charge = (
                self.battery.soc +
                (charge_energy - min(dam_energy_block, 0)) / self.capacity_mwh
            )

            if soc_after_discharge < self.min_soc - 0.01:
                cap_mask[i] = False
            if soc_after_charge > self.max_soc + 0.01:
                cap_mask[i] = False

        cap_mask[0] = True  # 0% always valid

        # I allow all price tiers (no physical constraint)
        price_mask = np.ones(len(AFRR_PRICE_TIERS), dtype=bool)

        return np.concatenate([cap_mask, price_mask])
