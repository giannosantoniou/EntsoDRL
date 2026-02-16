"""
IntraDay (XBID) Battery Trading Environment

I implement the IntraDay model that trades every ISP (15 min) on the
continuous XBID market. This is the THIRD model in the cascade:
DAM -> aFRR -> IntraDay -> mFRR.

Key Design Decisions:
1. Decision moment: Every ISP (15 min), XBID closes 60 min before delivery
2. Episode: 672 steps (7 days x 96 ISPs/day)
3. Blocking: aFRR activation -> forced idle (can't trade)
4. Action space: Discrete(11) = [-1.0 .. +1.0] x remaining_after_afrr
5. Executes DAM commitment + aFRR activation before own trade each ISP

Information Barriers:
- SEES: DAM+aFRR commitments (known), lagged XBID prices, DAM actual prices,
        SoC after DAM+aFRR execution, aFRR activation status
- DOESN'T SEE: Future XBID prices, mFRR activations
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings

from gym_envs.shared_battery_state import SharedBatteryState, UpstreamSimulator
from gym_envs.decomposed_rewards import IntraDayRewardCalculator

warnings.filterwarnings('ignore')


# I define IntraDay action levels (fraction of remaining capacity)
ID_LEVELS = np.linspace(-1.0, 1.0, 11, dtype=np.float32)


class BatteryEnvIntraDay(gym.Env):
    """
    IntraDay Environment: 672-step episodes for per-ISP XBID trading.

    I trade on the continuous IntraDay market, blocked during aFRR activation.
    Each step is one 15-minute ISP.
    """

    metadata = {'render_modes': ['human']}
    N_OBS = 22
    N_ACTIONS = 11  # len(ID_LEVELS)
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

        self.upstream = upstream_simulator or UpstreamSimulator(use_data_fallback=True)

        self._prepare_data()

        self.action_space = spaces.Discrete(self.N_ACTIONS)
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
        self.reward_calculator = IntraDayRewardCalculator(
            degradation_cost_per_mwh=rc.get('degradation_cost', degradation_cost),
            spread_bonus_weight=rc.get('spread_bonus_weight', 0.05),
            reward_scale=rc.get('reward_scale', 0.001),
        )

        self.current_step = 0
        self._episode_start_idx = 0
        self._daily_net_position = 0.0

    def _prepare_data(self) -> None:
        """I prepare and validate training data for IntraDay environment."""
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
            'intraday_bid': 98.0,
            'intraday_ask': 102.0,
            'intraday_spread': 4.0,
            'intraday_volume': 0.5,
            'dam_commitment': 0.0,
            'net_imbalance_mw': 0.0,
            'afrr_up': 80.0,
            'afrr_down': 80.0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        if 'hour' not in df.columns:
            df['hour'] = df.index.hour + df.index.minute / 60.0
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek

        # I create lagged features (backward-looking only)
        lag_1isp = 1
        for col in ['intraday_bid', 'intraday_ask', 'intraday_spread', 'intraday_volume']:
            lag_col = f'{col}_lag'
            if lag_col not in df.columns:
                df[lag_col] = df[col].shift(lag_1isp)

        # I create price momentum (4-ISP lookback)
        if 'price_momentum_4isp' not in df.columns:
            price_4_ago = df['price'].shift(4)
            df['price_momentum_4isp'] = np.where(
                price_4_ago > 1.0,
                (df['price'] - price_4_ago) / price_4_ago,
                0.0
            )

        # I create rolling stats
        w24 = 24 * self._sph
        if 'price_mean_24h' not in df.columns:
            df['price_mean_24h'] = df['price'].rolling(w24, min_periods=1).mean()

        df = df.ffill().bfill()
        self.df = df

        episode_rows = self.EPISODE_LENGTH
        warmup = 48 * self._sph
        self._max_start = len(df) - episode_rows - 1
        self._warmup = warmup

    def reset(self, seed=None, options=None):
        """I reset for a new 672-step IntraDay episode (7 days)."""
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
        self._daily_net_position = 0.0

        # I load upstream commitments for first day
        self._load_upstream_for_day(0)

        obs = self._build_observation()
        return obs, {}

    def _load_upstream_for_day(self, day_offset: int) -> None:
        """I load DAM schedule and aFRR commitments for a day."""
        day_start = self._episode_start_idx + day_offset * self.ISPS_PER_DAY
        self.battery.dam_schedule = self.upstream.get_dam_schedule(
            self.df, day_start, self._sph
        )
        # I load aFRR commitments for all 6 blocks
        for block in range(6):
            block_start = day_start + block * 4 * self._sph
            cap_mw, price_tier = self.upstream.get_afrr_commitment(
                self.battery, self.df, block_start, block
            )
            self.battery.afrr_commitments[block] = cap_mw
            self.battery.afrr_price_tiers[block] = price_tier

    def _get_current_idx(self) -> int:
        """I return the DataFrame index for the current ISP."""
        return min(self._episode_start_idx + self.current_step, len(self.df) - 1)

    def _get_isp_in_day(self) -> int:
        """I return current ISP index within the day (0-95)."""
        return self.current_step % self.ISPS_PER_DAY

    def _get_hour(self) -> int:
        """I return current hour (0-23)."""
        return self._get_isp_in_day() // self._sph

    def _get_block(self) -> int:
        """I return current 4h block index (0-5)."""
        return self._get_hour() // 4

    def _check_afrr_activation(self) -> bool:
        """
        I check if aFRR is activated for the current ISP.
        This is determined by the UpstreamSimulator's stochastic simulation.
        """
        block = self._get_block()
        commitment = self.battery.afrr_commitments[block]
        if commitment < 0.1:
            self.battery.afrr_is_activated = False
            return False

        idx = self._get_current_idx()
        row = self.df.iloc[idx]

        # I simulate activation probability
        activation_prob = 0.15
        imbalance = abs(row.get('net_imbalance_mw', 0.0))
        hour = self._get_hour()
        if imbalance > 500:
            activation_prob *= 1.5
        if 17 <= hour <= 21:
            activation_prob *= 1.3

        activated = self.np_random.random() < activation_prob
        self.battery.afrr_is_activated = activated

        if activated:
            imb = row.get('net_imbalance_mw', 0.0)
            self.battery.afrr_activation_direction = 'up' if imb > 0 else 'down'
            self.battery.afrr_activated_mw = commitment
        else:
            self.battery.afrr_activation_direction = 'none'
            self.battery.afrr_activated_mw = 0.0

        return activated

    def step(self, action: int):
        """
        I execute one ISP step: mandatory DAM + optional aFRR + IntraDay trade.

        Stage 1: Execute DAM commitment (mandatory)
        Stage 2: Check + execute aFRR activation (mandatory if activated)
        Stage 3: Execute IntraDay trade (voluntary, blocked during aFRR)
        """
        idx = self._get_current_idx()
        row = self.df.iloc[idx]
        isp_in_day = self._get_isp_in_day()
        hour = self._get_hour()
        block = self._get_block()

        # STAGE 1: Execute DAM commitment (mandatory, 1/sph hourly rate)
        dam_mw = self.battery.dam_schedule[hour]
        dam_fraction = dam_mw / self._sph  # split hourly into ISPs
        if abs(dam_fraction) > 0.01:
            self.battery.apply_energy(dam_fraction, self.time_step_hours)

        # STAGE 2: Check aFRR activation
        afrr_activated = self._check_afrr_activation()
        if afrr_activated:
            commitment = self.battery.afrr_commitments[block]
            if self.battery.afrr_activation_direction == 'up':
                self.battery.apply_energy(commitment, self.time_step_hours)
            else:
                self.battery.apply_energy(-commitment, self.time_step_hours)

        # STAGE 3: Execute IntraDay trade (blocked during aFRR activation)
        remaining_d, remaining_c = self.battery.remaining_capacity(
            'intraday', isp=isp_in_day, hour=hour
        )
        remaining = min(remaining_d, remaining_c)

        action_level = ID_LEVELS[action]
        if afrr_activated:
            action_mw = 0.0  # I block trading during aFRR activation
        else:
            if action_level >= 0:
                action_mw = action_level * remaining_d
            else:
                action_mw = action_level * remaining_c

        actual_energy = 0.0
        if abs(action_mw) > 0.01 and not afrr_activated:
            actual_energy = self.battery.apply_energy(action_mw, self.time_step_hours)
            self.battery.intraday_positions[isp_in_day] = action_mw
            self._daily_net_position += action_mw

        # I get prices (lagged for observation, current for reward)
        xbid_bid = row.get('intraday_bid', 98.0)
        xbid_ask = row.get('intraday_ask', 102.0)
        dam_price = row.get('price', 80.0)

        # I calculate shortfall risk (looking ahead 4h)
        shortfall_risk = self._calculate_shortfall_risk()

        # I calculate reward
        reward, components = self.reward_calculator.calculate(
            action_mw=action_mw,
            xbid_bid_price=xbid_bid,
            xbid_ask_price=xbid_ask,
            dam_price=dam_price,
            capacity_mwh=self.capacity_mwh,
            time_step_hours=self.time_step_hours,
            soc=self.battery.soc,
            shortfall_risk=shortfall_risk,
        )

        # I advance
        self.current_step += 1

        # I handle day boundary
        if self._get_isp_in_day() == 0 and self.current_step < self.EPISODE_LENGTH:
            new_day = self.current_step // self.ISPS_PER_DAY
            self.battery.check_day_boundary(0, 23)
            self._daily_net_position = 0.0
            self._load_upstream_for_day(new_day)

        terminated = (self.current_step >= self.EPISODE_LENGTH)
        truncated = False

        info = {
            'isp': self.current_step - 1,
            'action_mw': action_mw,
            'afrr_activated': afrr_activated,
            'soc': self.battery.soc,
            'xbid_bid': xbid_bid,
            'xbid_ask': xbid_ask,
            **{f'r_{k}': v for k, v in components.items()},
        }

        obs = self._build_observation() if not terminated else np.zeros(self.N_OBS, dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def _calculate_shortfall_risk(self) -> float:
        """I calculate risk of not meeting future upstream commitments."""
        hour = self._get_hour()
        lookahead = min(4 * self._sph, self.EPISODE_LENGTH - self.current_step)

        total_sell = 0.0
        total_buy = 0.0

        for i in range(1, lookahead + 1):
            future_step = self.current_step + i
            future_isp = future_step % self.ISPS_PER_DAY
            future_hour = future_isp // self._sph
            if future_hour < 24:
                dam = self.battery.dam_schedule[future_hour]
                if dam > 0.1:
                    total_sell += dam / self._sph
                elif dam < -0.1:
                    total_buy += abs(dam) / self._sph

        if total_sell > 0:
            available = (self.battery.soc - self.min_soc) * self.capacity_mwh
            needed = total_sell * self.time_step_hours / self.eff_sqrt
            if available < needed:
                return min(1.0, (needed - available) / max(needed, 0.01))

        return 0.0

    def _build_observation(self) -> np.ndarray:
        """I build the 22-feature observation for the current ISP."""
        features = []
        idx = self._get_current_idx()
        row = self.df.iloc[idx]
        isp_in_day = self._get_isp_in_day()
        hour = self._get_hour()
        block = self._get_block()

        # [0] SoC
        features.append(self.battery.soc)

        # [1-2] max discharge/charge after DAM+aFRR
        max_d, max_c = self.battery.remaining_capacity(
            'intraday', isp=isp_in_day, hour=hour
        )
        features.append(max_d / self.max_power_mw)
        features.append(max_c / self.max_power_mw)

        # [3-4] hour sin/cos (fractional for 15-min)
        fractional_hour = row.get('hour', hour)
        if isinstance(fractional_hour, (int, np.integer)):
            fractional_hour = float(fractional_hour) + (isp_in_day % self._sph) * (60.0 / self._sph) / 60.0
        features.append(np.sin(2 * np.pi * fractional_hour / 24))
        features.append(np.cos(2 * np.pi * fractional_hour / 24))

        # [5-6] day-of-week sin/cos
        dow = row.get('day_of_week', 0)
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # [7-8] DAM commitment this ISP and next hour
        dam_this = self.battery.dam_schedule[hour] / self.max_power_mw
        next_hour = min(hour + 1, 23)
        dam_next = self.battery.dam_schedule[next_hour] / self.max_power_mw
        features.append(dam_this)
        features.append(dam_next)

        # [9-10] aFRR committed MW and activation status
        afrr_mw = self.battery.afrr_commitments[block]
        features.append(afrr_mw / self.max_power_mw)
        features.append(1.0 if self.battery.afrr_is_activated else 0.0)

        # [11-14] XBID prices (LAGGED by 1 ISP)
        features.append(row.get('intraday_bid_lag', row.get('intraday_bid', 98.0)) / 100.0)
        features.append(row.get('intraday_ask_lag', row.get('intraday_ask', 102.0)) / 100.0)
        features.append(row.get('intraday_spread_lag', row.get('intraday_spread', 4.0)) / 100.0)
        features.append(row.get('intraday_volume_lag', row.get('intraday_volume', 0.5)))

        # [15-16] DAM price, XBID vs DAM spread
        dam_price = row.get('price', 80.0)
        xbid_mid = (row.get('intraday_bid', 98.0) + row.get('intraday_ask', 102.0)) / 2
        features.append(dam_price / 100.0)
        features.append((xbid_mid - dam_price) / 100.0)

        # [17-18] price momentum (4 ISPs), shortfall risk
        features.append(np.clip(row.get('price_momentum_4isp', 0.0), -1.0, 1.0))
        features.append(self._calculate_shortfall_risk())

        # [19] net imbalance lag
        features.append(row.get('net_imbalance_mw', 0.0) / 1000.0)

        # [20] daily net IntraDay position / max_power
        features.append(np.clip(self._daily_net_position / self.max_power_mw, -1.0, 1.0))

        # [21] daily cycles / max
        features.append(self.battery.daily_cycles / self.max_daily_cycles)

        obs = np.array(features, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        assert len(obs) == self.N_OBS, f"Expected {self.N_OBS} features, got {len(obs)}"
        return obs

    def action_masks(self) -> np.ndarray:
        """
        I return action masks. aFRR activation blocks ALL trading.
        Otherwise, I mask by power/SoC limits.
        """
        mask = np.zeros(self.N_ACTIONS, dtype=bool)
        isp_in_day = self._get_isp_in_day()
        hour = self._get_hour()

        # I block all trading during aFRR activation
        if self.battery.afrr_is_activated:
            mask[5] = True  # idle only
            return mask

        remaining_d, remaining_c = self.battery.remaining_capacity(
            'intraday', isp=isp_in_day, hour=hour
        )

        for i, level in enumerate(ID_LEVELS):
            if abs(level) < 0.01:
                mask[i] = True  # idle
                continue

            if level > 0:  # sell (discharge)
                power = level * remaining_d
                if power <= remaining_d + 0.01:
                    mask[i] = True
            else:  # buy (charge)
                power = abs(level) * remaining_c
                if power <= remaining_c + 0.01:
                    mask[i] = True

        mask[5] = True  # idle always valid
        return mask
