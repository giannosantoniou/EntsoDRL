"""
mFRR (manual Frequency Restoration Reserve) Battery Trading Environment

I implement the mFRR model that submits Free Bids every ISP.
This is the FOURTH and LAST model in the cascade: DAM -> aFRR -> IntraDay -> mFRR.

Key Design Decisions:
1. Decision moment: Every ISP, H-25min before delivery (Free Bids)
2. Episode: 672 steps (7 days x 96 ISPs/day)
3. Direction constraint: TSO imbalance determines allowed direction
4. Activation: Probabilistic, merit-order based (lower price -> higher prob)
5. Action space: MultiDiscrete([11, 5]) = 55 combos (quantity + price tier)
6. Uses LAGGED imbalance to prevent real-time information leakage

Information Barriers:
- SEES: All upstream positions (DAM+aFRR+IntraDay), lagged mFRR prices,
        system imbalance (LAGGED), direction signal
- DOESN'T SEE: Actual mFRR clearing price (ex-post), real-time imbalance
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

from gym_envs.shared_battery_state import SharedBatteryState, UpstreamSimulator
from gym_envs.decomposed_rewards import MFRRRewardCalculator

warnings.filterwarnings('ignore')


# I define mFRR quantity levels (fraction of remaining capacity)
MFRR_LEVELS = np.linspace(-1.0, 1.0, 11, dtype=np.float32)

# I define mFRR price tier multipliers (relative to reference price)
MFRR_PRICE_TIERS = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)

# I define minimum |imbalance| for mFRR market to be open
MFRR_IMBALANCE_THRESHOLD = 30.0  # MW


class BatteryEnvMFRR(gym.Env):
    """
    mFRR Environment: 672-step episodes for per-ISP Free Bid submission.

    I submit mFRR Free Bids constrained by TSO system imbalance direction.
    The bid may or may not be activated (probabilistic, merit-order based).
    """

    metadata = {'render_modes': ['human']}
    N_OBS = 18
    N_QTY_ACTIONS = 11   # len(MFRR_LEVELS)
    N_PRICE_ACTIONS = 5  # len(MFRR_PRICE_TIERS)
    ISPS_PER_DAY = 96
    EPISODE_DAYS = 7
    EPISODE_LENGTH = ISPS_PER_DAY * EPISODE_DAYS  # 672

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
        time_step_hours: float = 0.25,
        random_start: bool = True,
        # mFRR parameters
        imbalance_threshold: float = 30.0,
        base_activation_prob: float = 0.35,
        price_cap: float = 200.0,
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
        self.time_step_hours = time_step_hours
        self.random_start = random_start
        self.imbalance_threshold = imbalance_threshold
        self.base_activation_prob = base_activation_prob
        self.price_cap = price_cap

        self.upstream = upstream_simulator or UpstreamSimulator(use_data_fallback=True)

        self._prepare_data()

        self.action_space = spaces.MultiDiscrete([
            self.N_QTY_ACTIONS,    # 11 quantity levels
            self.N_PRICE_ACTIONS,  # 5 price tiers
        ])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.N_OBS,), dtype=np.float32
        )

        self.battery = SharedBatteryState(
            capacity_mwh=capacity_mwh,
            max_power_mw=max_power_mw,
            efficiency=efficiency,
            min_soc=min_soc,
            max_soc=max_soc,
            max_daily_cycles=max_daily_cycles,
        )

        rc = reward_config or {}
        self.reward_calculator = MFRRRewardCalculator(
            degradation_cost_per_mwh=rc.get('degradation_cost', degradation_cost),
            price_cap=rc.get('price_cap', price_cap),
            reward_scale=rc.get('reward_scale', 0.001),
        )

        self.current_step = 0
        self._episode_start_idx = 0

    def _prepare_data(self) -> None:
        """I prepare and validate training data for mFRR environment."""
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

        # I add defaults for missing columns
        defaults = {
            'mfrr_price_up': 100.0,
            'mfrr_price_down': 80.0,
            'net_imbalance_mw': 0.0,
            'dam_commitment': 0.0,
            'afrr_up': 80.0,
            'intraday_bid': 98.0,
            'intraday_ask': 102.0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        if 'hour' not in df.columns:
            df['hour'] = df.index.hour + df.index.minute / 60.0
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek

        # I create lagged features (LAGGED to prevent information leakage)
        for col in ['mfrr_price_up', 'mfrr_price_down', 'net_imbalance_mw']:
            lag_col = f'{col}_lag'
            if lag_col not in df.columns:
                df[lag_col] = df[col].shift(1)

        # I create mFRR spread
        if 'mfrr_spread_lag' not in df.columns:
            df['mfrr_spread_lag'] = (
                df.get('mfrr_price_up', pd.Series(100.0, index=df.index)).shift(1) -
                df.get('mfrr_price_down', pd.Series(80.0, index=df.index)).shift(1)
            )

        # I create system stress indicator
        if 'system_stress' not in df.columns:
            df['system_stress'] = np.clip(
                abs(df['net_imbalance_mw']) / 500.0, 0.0, 1.0
            )

        df = df.ffill().bfill()
        self.df = df

        episode_rows = self.EPISODE_LENGTH
        warmup = 48 * self._sph
        self._max_start = len(df) - episode_rows - 1
        self._warmup = warmup

    def reset(self, seed=None, options=None):
        """I reset for a new 672-step mFRR episode (7 days)."""
        super().reset(seed=seed)

        if self.random_start:
            self._episode_start_idx = self.np_random.integers(
                self._warmup, max(self._warmup + 1, self._max_start)
            )
        else:
            self._episode_start_idx = self._warmup

        if self.random_start:
            initial_soc = self.np_random.uniform(0.2, 0.8)
        else:
            initial_soc = 0.5

        self.battery.reset(soc=initial_soc)
        self.current_step = 0

        # I load upstream commitments for first day
        self._load_upstream_for_day(0)

        obs = self._build_observation()
        return obs, {}

    def _load_upstream_for_day(self, day_offset: int) -> None:
        """I load all upstream commitments (DAM + aFRR + IntraDay)."""
        day_start = self._episode_start_idx + day_offset * self.ISPS_PER_DAY

        # I load DAM schedule
        self.battery.dam_schedule = self.upstream.get_dam_schedule(
            self.df, day_start, self._sph
        )

        # I load aFRR commitments
        for block in range(6):
            block_start = day_start + block * 4 * self._sph
            cap_mw, price_tier = self.upstream.get_afrr_commitment(
                self.battery, self.df, block_start, block
            )
            self.battery.afrr_commitments[block] = cap_mw
            self.battery.afrr_price_tiers[block] = price_tier

        # I load IntraDay positions
        for isp in range(self.ISPS_PER_DAY):
            id_pos = self.upstream.get_intraday_position(
                self.battery, self.df, isp
            )
            self.battery.intraday_positions[isp] = id_pos

    def _get_current_idx(self) -> int:
        """I return the DataFrame index for the current ISP."""
        return min(self._episode_start_idx + self.current_step, len(self.df) - 1)

    def _get_isp_in_day(self) -> int:
        return self.current_step % self.ISPS_PER_DAY

    def _get_hour(self) -> int:
        return self._get_isp_in_day() // self._sph

    def _get_block(self) -> int:
        return self._get_hour() // 4

    def step(self, action):
        """
        I execute one ISP step: execute upstream + submit mFRR Free Bid.

        Stage 1: Execute DAM commitment (mandatory)
        Stage 2: Execute aFRR if activated (mandatory)
        Stage 3: Execute IntraDay position (from upstream)
        Stage 4: Submit and simulate mFRR Free Bid (agent's decision)
        """
        qty_idx, price_idx = action
        idx = self._get_current_idx()
        row = self.df.iloc[idx]
        isp_in_day = self._get_isp_in_day()
        hour = self._get_hour()
        block = self._get_block()

        # STAGE 1: Execute DAM (mandatory)
        dam_mw = self.battery.dam_schedule[hour]
        dam_fraction = dam_mw / self._sph
        if abs(dam_fraction) > 0.01:
            self.battery.apply_energy(dam_fraction, self.time_step_hours)

        # STAGE 2: Execute aFRR if activated (stochastic)
        afrr_commitment = self.battery.afrr_commitments[block]
        afrr_activated = False
        if afrr_commitment > 0.1:
            activation_prob = 0.15
            if abs(row.get('net_imbalance_mw', 0.0)) > 500:
                activation_prob *= 1.5
            if 17 <= hour <= 21:
                activation_prob *= 1.3
            afrr_activated = self.np_random.random() < activation_prob

            if afrr_activated:
                imb = row.get('net_imbalance_mw', 0.0)
                if imb > 0:
                    self.battery.apply_energy(afrr_commitment, self.time_step_hours)
                else:
                    self.battery.apply_energy(-afrr_commitment, self.time_step_hours)

        # STAGE 3: Execute IntraDay position (from upstream)
        id_pos = self.battery.intraday_positions[isp_in_day]
        if abs(id_pos) > 0.01:
            self.battery.apply_energy(id_pos, self.time_step_hours)

        # STAGE 4: mFRR Free Bid
        remaining_d, remaining_c = self.battery.remaining_capacity(
            'mfrr', isp=isp_in_day, hour=hour
        )

        # I get LAGGED imbalance for direction constraint
        imbalance_lag = row.get('net_imbalance_mw_lag',
                                row.get('net_imbalance_mw', 0.0))

        # I determine bid quantity
        qty_level = MFRR_LEVELS[qty_idx]
        price_tier = MFRR_PRICE_TIERS[price_idx]

        # I apply direction constraint
        bid_mw = 0.0
        if abs(imbalance_lag) <= self.imbalance_threshold:
            bid_mw = 0.0  # market closed
        elif imbalance_lag < -self.imbalance_threshold:
            # deficit -> only discharge (UP regulation, positive qty)
            if qty_level > 0:
                bid_mw = qty_level * remaining_d
            else:
                bid_mw = 0.0  # wrong direction
        elif imbalance_lag > self.imbalance_threshold:
            # surplus -> only charge (DOWN regulation, negative qty)
            if qty_level < 0:
                bid_mw = qty_level * remaining_c
            else:
                bid_mw = 0.0  # wrong direction

        # I calculate bid price
        if abs(bid_mw) > 0.01:
            ref_price = row.get('mfrr_price_up_lag',
                                row.get('mfrr_price_up', 100.0))
            if bid_mw < 0:  # down regulation
                ref_price = row.get('mfrr_price_down_lag',
                                    row.get('mfrr_price_down', 80.0))
            bid_price = ref_price * price_tier
        else:
            bid_price = 0.0

        # I simulate activation (probabilistic, merit-order based)
        was_activated = False
        if abs(bid_mw) > 0.01:
            activation_prob = self.base_activation_prob
            if abs(imbalance_lag) > 100:
                activation_prob += 0.20
            if price_tier < 1.0:
                activation_prob += 0.15  # aggressive bid -> more likely
            elif price_tier > 1.0:
                activation_prob -= 0.10  # conservative bid -> less likely
            activation_prob = np.clip(activation_prob, 0.05, 0.90)
            was_activated = self.np_random.random() < activation_prob

        # I execute if activated
        actual_energy = 0.0
        if was_activated and abs(bid_mw) > 0.01:
            actual_energy = self.battery.apply_energy(bid_mw, self.time_step_hours)

        # I calculate reward
        reward, components = self.reward_calculator.calculate(
            bid_mw=abs(bid_mw),
            bid_price=bid_price,
            was_activated=was_activated,
            capacity_mwh=self.capacity_mwh,
            time_step_hours=self.time_step_hours,
        )

        # I advance
        self.current_step += 1

        # I handle day boundary
        if self._get_isp_in_day() == 0 and self.current_step < self.EPISODE_LENGTH:
            new_day = self.current_step // self.ISPS_PER_DAY
            self.battery.check_day_boundary(0, 23)
            self._load_upstream_for_day(new_day)

        terminated = (self.current_step >= self.EPISODE_LENGTH)
        truncated = False

        info = {
            'isp': self.current_step - 1,
            'bid_mw': bid_mw,
            'bid_price': bid_price,
            'was_activated': was_activated,
            'imbalance_lag': imbalance_lag,
            'soc': self.battery.soc,
            **{f'r_{k}': v for k, v in components.items()},
        }

        obs = self._build_observation() if not terminated else np.zeros(self.N_OBS, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """I build the 18-feature observation for the current ISP."""
        features = []
        idx = self._get_current_idx()
        row = self.df.iloc[idx]
        isp_in_day = self._get_isp_in_day()
        hour = self._get_hour()
        block = self._get_block()

        # [0] SoC
        features.append(self.battery.soc)

        # [1-2] max discharge/charge after all upstream
        max_d, max_c = self.battery.remaining_capacity(
            'mfrr', isp=isp_in_day, hour=hour
        )
        features.append(max_d / self.max_power_mw)
        features.append(max_c / self.max_power_mw)

        # [3-4] hour sin/cos
        fractional_hour = row.get('hour', float(hour))
        if isinstance(fractional_hour, (int, np.integer)):
            fractional_hour = float(fractional_hour) + (isp_in_day % self._sph) * (60.0 / self._sph) / 60.0
        features.append(np.sin(2 * np.pi * fractional_hour / 24))
        features.append(np.cos(2 * np.pi * fractional_hour / 24))

        # [5-6] day-of-week sin/cos
        dow = row.get('day_of_week', 0)
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # [7-8] net imbalance lag, mFRR direction
        imbalance_lag = row.get('net_imbalance_mw_lag',
                                row.get('net_imbalance_mw', 0.0))
        features.append(imbalance_lag / 1000.0)

        # I encode direction as {-1, 0, +1}
        if imbalance_lag < -self.imbalance_threshold:
            direction = -1.0  # deficit -> UP
        elif imbalance_lag > self.imbalance_threshold:
            direction = 1.0   # surplus -> DOWN
        else:
            direction = 0.0   # market closed
        features.append(direction)

        # [9-11] mFRR prices lag + spread
        features.append(row.get('mfrr_price_up_lag',
                                row.get('mfrr_price_up', 100.0)) / 100.0)
        features.append(row.get('mfrr_price_down_lag',
                                row.get('mfrr_price_down', 80.0)) / 100.0)
        features.append(row.get('mfrr_spread_lag', 20.0) / 100.0)

        # [12] system stress
        features.append(row.get('system_stress', 0.0))

        # [13-15] upstream positions
        dam_mw = self.battery.dam_schedule[hour] if hour < 24 else 0.0
        afrr_mw = self.battery.afrr_commitments[block] if block < 6 else 0.0
        id_mw = self.battery.intraday_positions[isp_in_day] if isp_in_day < 96 else 0.0
        features.append(dam_mw / self.max_power_mw)
        features.append(afrr_mw / self.max_power_mw)
        features.append(id_mw / self.max_power_mw)

        # [16] is peak hour
        features.append(1.0 if 17 <= hour <= 21 else 0.0)

        # [17] daily cycles / max
        features.append(self.battery.daily_cycles / self.max_daily_cycles)

        obs = np.array(features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        assert len(obs) == self.N_OBS, f"Expected {self.N_OBS} features, got {len(obs)}"
        return obs

    def action_masks(self) -> np.ndarray:
        """
        I return action masks with direction constraint from system imbalance.

        If |imbalance| <= threshold: market CLOSED, only idle allowed.
        If imbalance < -threshold: deficit, only UP (discharge, positive qty).
        If imbalance > threshold: surplus, only DOWN (charge, negative qty).
        """
        idx = self._get_current_idx()
        row = self.df.iloc[idx]
        isp_in_day = self._get_isp_in_day()
        hour = self._get_hour()

        remaining_d, remaining_c = self.battery.remaining_capacity(
            'mfrr', isp=isp_in_day, hour=hour
        )

        imbalance_lag = row.get('net_imbalance_mw_lag',
                                row.get('net_imbalance_mw', 0.0))

        qty_mask = np.zeros(self.N_QTY_ACTIONS, dtype=bool)
        price_mask = np.ones(self.N_PRICE_ACTIONS, dtype=bool)

        if abs(imbalance_lag) <= self.imbalance_threshold:
            # I close the market - only idle allowed
            qty_mask[5] = True  # center = idle
            return np.concatenate([qty_mask, price_mask])

        if imbalance_lag < -self.imbalance_threshold:
            # I allow only UP regulation (discharge, positive actions)
            for i in range(6, self.N_QTY_ACTIONS):
                level = MFRR_LEVELS[i]
                power = level * remaining_d
                if power <= remaining_d + 0.01:
                    qty_mask[i] = True
        elif imbalance_lag > self.imbalance_threshold:
            # I allow only DOWN regulation (charge, negative actions)
            for i in range(0, 5):
                level = MFRR_LEVELS[i]
                power = abs(level) * remaining_c
                if power <= remaining_c + 0.01:
                    qty_mask[i] = True

        qty_mask[5] = True  # idle always valid
        return np.concatenate([qty_mask, price_mask])
