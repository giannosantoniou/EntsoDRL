"""
Unified Multi-Market Battery Trading Environment

I implement a single unified environment that handles ALL markets:
- DAM (Day-Ahead Market) commitment tracking
- IntraDay (XBID/HEnEx) energy trading — voluntary arbitrage
- aFRR (automatic Frequency Restoration Reserve) capacity + energy
- mFRR (manual Frequency Restoration Reserve) energy trading — balancing market

Key Design Decisions:
1. Single model for all markets (learns market interactions)
2. MultiDiscrete action space: [aFRR_commitment, aFRR_price_tier, intraday_action, mfrr_action]
3. Hierarchical capacity cascading: DAM -> aFRR -> IntraDay -> mFRR
4. aFRR activation simulation with MANDATORY response requirement
5. HEnEx-compliant gate closure awareness

Action Space: MultiDiscrete([5, 5, 11, 11]) = 3,025 combinations
- [0] aFRR Commitment: 0%, 25%, 50%, 75%, 100% of remaining capacity
- [1] aFRR Price Tier: 5 levels from aggressive to conservative bidding
- [2] IntraDay Action: 11 levels (-1.0 to +1.0) × remaining capacity after aFRR
- [3] mFRR Action: 11 levels (-1.0 to +1.0) × remaining capacity after IntraDay

Observation Space: 63 features (see _build_observation for details)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState


class BatteryEnvUnified(gym.Env):
    """
    Unified Multi-Market Battery Trading Environment.

    I integrate DAM, IntraDay, aFRR, and mFRR markets into a single training
    environment with hierarchical action masking and capacity cascading.
    """

    metadata = {'render_modes': ['human']}

    # I define aFRR commitment levels (fraction of remaining capacity)
    AFRR_LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])  # 5 levels

    # I define aFRR price tier multipliers (bid aggressiveness)
    # Lower = more aggressive (higher chance of selection, less revenue)
    # Higher = more conservative (lower chance, more revenue if selected)
    AFRR_PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])  # 5 tiers

    # I define separate IntraDay and mFRR action levels (~6 MW granularity per level)
    INTRADAY_LEVELS = np.linspace(-1.0, 1.0, 11)  # 11 levels
    MFRR_LEVELS = np.linspace(-1.0, 1.0, 11)  # 11 levels
    N_INTRADAY_ACTIONS = 11
    N_MFRR_ACTIONS = 11

    def __init__(
        self,
        df: pd.DataFrame,
        # Battery parameters
        capacity_mwh: float = 146.0,
        max_power_mw: float = 30.0,
        efficiency: float = 0.94,
        min_soc: float = 0.05,
        max_soc: float = 0.95,

        # Time configuration
        time_step_hours: float = 1.0,
        warmup_steps: int = 48,  # 48h warmup for rolling statistics

        # aFRR simulation parameters
        selection_probability: float = 0.80,  # 80% chance of capacity bid selection
        afrr_activation_rate: float = 0.15,  # 15% activation rate when selected
        peak_activation_boost: float = 1.3,  # Higher activation during peak hours
        high_imbalance_boost: float = 1.5,  # Higher activation during system stress

        # Cycling limits
        max_daily_cycles: float = 2.0,
        degradation_cost: float = 15.0,

        # Reward configuration
        reward_config: Optional[Dict] = None,

        # Feature configuration
        lookahead_hours: int = 12,  # Price and DAM lookahead window

        # Training configuration
        episode_length: Optional[int] = None,  # None = full dataset
        random_start: bool = True
    ):
        super().__init__()

        # I store configuration
        self.df = df.copy()
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.eff_sqrt = np.sqrt(efficiency)
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.time_step_hours = time_step_hours
        self.warmup_steps = warmup_steps

        # aFRR parameters
        self.selection_probability = selection_probability
        self.afrr_activation_rate = afrr_activation_rate
        self.peak_activation_boost = peak_activation_boost
        self.high_imbalance_boost = high_imbalance_boost

        # Cycling
        self.max_daily_cycles = max_daily_cycles
        self.degradation_cost = degradation_cost

        # Reward calculator
        self.reward_config = reward_config or {}
        self.reward_calculator = UnifiedRewardCalculator(
            degradation_cost_per_mwh=self.reward_config.get('degradation_cost', 15.0),
            dam_violation_penalty=self.reward_config.get('dam_violation_penalty', 1500.0),
            afrr_nonresponse_penalty=self.reward_config.get('afrr_nonresponse_penalty', 2000.0),
            reward_scale=self.reward_config.get('reward_scale', 0.001)
        )

        # Feature configuration
        self.lookahead_hours = lookahead_hours
        self.lookahead_steps = int(lookahead_hours / time_step_hours)

        # Episode configuration
        self.episode_length = episode_length
        self.random_start = random_start

        # I prepare the data
        self._prepare_data()

        # I define action and observation spaces
        self._setup_spaces()

        # I initialize state
        self._reset_state()

        print(f"Unified Multi-Market Environment Initialized (IntraDay/mFRR separated)")
        print(f"  Battery: {capacity_mwh} MWh, {max_power_mw} MW")
        print(f"  Action Space: MultiDiscrete({list(self.action_space.nvec)}) = {np.prod(self.action_space.nvec)} combos")
        print(f"  Observation Space: {self.observation_space.shape[0]} features")
        print(f"  aFRR: selection={selection_probability:.0%}, activation={afrr_activation_rate:.0%}")
        print(f"  Data: {len(self.df)} hours ({len(self.df)/24:.0f} days)")

    def _prepare_data(self):
        """I prepare and validate the training data."""
        df = self.df

        # I ensure datetime index (handle various formats including timezone-aware)
        if not isinstance(df.index, pd.DatetimeIndex):
            # I parse the index, handling timezone info in the strings
            # Format: "2021-01-01 00:00:00+02:00"
            try:
                df.index = pd.to_datetime(df.index, format='mixed', utc=True)
                df.index = df.index.tz_convert('Europe/Athens')
            except (ValueError, TypeError):
                df.index = pd.to_datetime(df.index)
        # I leave naive DatetimeIndex as-is — hour/dayofweek work without tz

        # I ensure required columns exist
        required = ['price', 'dam_commitment']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # I add aFRR/mFRR/IntraDay columns if missing (with safe constant defaults
        # that do NOT depend on DAM price to avoid data leakage)
        defaults = {
            'afrr_up': 80.0,
            'afrr_down': 80.0,
            'afrr_cap_up_price': 20.0,
            'afrr_cap_down_price': 30.0,
            'intraday_bid': 98.0,   # Fixed default, NOT derived from DAM
            'intraday_ask': 102.0,  # Fixed default, NOT derived from DAM
            'intraday_spread': 4.0,
            'intraday_volume': 0.5,
            'mfrr_price_up': 120.0,
            'mfrr_price_down': 60.0,
            'mfrr_spread': 60.0,
            'net_imbalance_mw': 0.0
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default() if callable(default) else default

        # I add time features if missing
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df.index.dayofweek

        # I add lagged features (backward-looking only!)
        lag_cols = ['price', 'afrr_up', 'afrr_down', 'mfrr_price_up', 'mfrr_price_down',
                    'intraday_bid', 'intraday_ask', 'intraday_spread', 'intraday_volume']
        for col in lag_cols:
            if col in df.columns and f'{col}_lag_1h' not in df.columns:
                df[f'{col}_lag_1h'] = df[col].shift(1)

        # I add rolling statistics (backward-looking only!)
        if 'price_mean_24h' not in df.columns:
            df['price_mean_24h'] = df['price'].rolling(24, min_periods=1).mean()
        if 'price_std_24h' not in df.columns:
            df['price_std_24h'] = df['price'].rolling(24, min_periods=1).std().fillna(10)
        if 'price_min_24h' not in df.columns:
            df['price_min_24h'] = df['price'].rolling(24, min_periods=1).min()
        if 'price_max_24h' not in df.columns:
            df['price_max_24h'] = df['price'].rolling(24, min_periods=1).max()

        # I compute price momentum (backward-looking)
        if 'price_momentum' not in df.columns:
            price_6h_ago = df['price'].shift(6)
            df['price_momentum'] = (df['price'] - price_6h_ago) / (price_6h_ago + 1)
            df['price_momentum'] = df['price_momentum'].fillna(0).clip(-1, 1)

        # I compute aFRR capacity price percentile
        if 'afrr_cap_percentile' not in df.columns:
            df['afrr_cap_percentile'] = df['afrr_cap_up_price'].rolling(168, min_periods=24).rank(pct=True).fillna(0.5)

        # I fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        self.df = df

        # I calculate step limits
        self.start_step = self.warmup_steps
        self.max_steps = len(self.df) - 1 - self.lookahead_steps

    def _setup_spaces(self):
        """I define action and observation spaces."""
        # Action space: MultiDiscrete([aFRR_commitment, aFRR_price_tier, intraday_action, mfrr_action])
        self.action_space = spaces.MultiDiscrete([
            len(self.AFRR_LEVELS),      # 5 aFRR commitment levels
            len(self.AFRR_PRICE_TIERS), # 5 aFRR price tiers
            self.N_INTRADAY_ACTIONS,    # 11 IntraDay actions
            self.N_MFRR_ACTIONS         # 11 mFRR actions
        ])

        # Observation space: 63 features (was 58, +5 for separate IntraDay features)
        # See _build_observation for detailed breakdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(63,), dtype=np.float32
        )

    def _reset_state(self):
        """I initialize/reset the environment state."""
        self.current_step = self.start_step
        self.soc = 0.5
        self.afrr_commitment_mw = 0.0
        self.afrr_commitment_level = 0  # Index into AFRR_LEVELS
        self.is_selected_for_afrr = False
        self.hours_since_afrr_activation = 24  # Start with no recent activation

        # I track profits by market
        self.total_profit = 0.0
        self.dam_profit = 0.0
        self.intraday_profit = 0.0
        self.afrr_capacity_profit = 0.0
        self.afrr_energy_profit = 0.0
        self.mfrr_profit = 0.0

        # I track cycling
        self.total_cycles = 0.0
        self.daily_cycles = 0.0
        self.current_day = None

        # I track episode
        self.episode_start_step = self.current_step
        self.episode_profit = 0.0

    def reset(self, seed=None, options=None):
        """I reset the environment for a new episode."""
        super().reset(seed=seed)

        # I determine starting position
        if self.random_start:
            max_start = self.max_steps - (self.episode_length or 168)  # Default 1 week
            max_start = max(self.start_step + 1, max_start)
            self.current_step = self.np_random.integers(self.start_step, max_start)
        else:
            self.current_step = self.start_step

        # I randomize initial SoC
        self.soc = self.np_random.uniform(0.3, 0.7)

        # I reset tracking variables
        self._reset_state()
        self.episode_start_step = self.current_step
        self.current_day = self._get_current_day()

        obs = self._build_observation()
        info = self._get_info()

        return obs, info

    def _get_current_day(self) -> int:
        """I return a unique identifier for the current day."""
        ts = self.df.index[self.current_step]
        return ts.dayofyear + ts.year * 1000

    def _check_day_reset(self):
        """I reset daily cycling counter at day boundaries."""
        current_day = self._get_current_day()
        if self.current_day is None or current_day != self.current_day:
            self.current_day = current_day
            self.daily_cycles = 0.0

    def action_masks(self) -> np.ndarray:
        """
        I return hierarchical action masks for the MultiDiscrete space.

        Returns a flattened mask for sb3_contrib.MaskablePPO compatibility.
        Total length = 5 + 5 + 11 + 11 = 32
        """
        row = self.df.iloc[self.current_step]

        # I get current DAM commitment (consumes capacity)
        dam_commitment = row.get('dam_commitment', 0.0)
        dam_commitment = np.clip(dam_commitment, -self.max_power_mw, self.max_power_mw)

        # I calculate remaining capacity after DAM
        remaining_capacity = self.max_power_mw - abs(dam_commitment)

        # I calculate SoC-based limits using DAM-aware reserves
        dam_min_soc, dam_max_soc = self._calculate_dam_soc_reserve()
        available_discharge_mwh = max(0.0, (self.soc - dam_min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (dam_max_soc - self.soc) * self.capacity_mwh)

        max_discharge_mw = min(
            remaining_capacity,
            available_discharge_mwh * self.eff_sqrt / self.time_step_hours
        )
        max_charge_mw = min(
            remaining_capacity,
            available_charge_mwh / self.eff_sqrt / self.time_step_hours
        )

        # I apply daily cycle limits
        cycles_remaining = max(0, self.max_daily_cycles - self.daily_cycles)
        max_cycle_power = cycles_remaining * self.capacity_mwh / self.time_step_hours
        max_discharge_mw = min(max_discharge_mw, max_cycle_power)
        max_charge_mw = min(max_charge_mw, max_cycle_power)

        # Mask 1: aFRR commitment levels (5 options)
        # I allow all levels, but mask out those exceeding remaining capacity
        afrr_mask = np.ones(len(self.AFRR_LEVELS), dtype=bool)
        for i, level in enumerate(self.AFRR_LEVELS):
            required_capacity = level * remaining_capacity
            # I need at least this much capacity available in BOTH directions
            if required_capacity > max_discharge_mw or required_capacity > max_charge_mw:
                afrr_mask[i] = False
        afrr_mask[0] = True  # Always allow zero commitment

        # Mask 2: aFRR price tiers (5 options)
        # I always allow all price tiers (no physical constraints)
        price_mask = np.ones(len(self.AFRR_PRICE_TIERS), dtype=bool)

        # Mask 3: IntraDay actions (11 options)
        # I mask based on available power after aFRR reservation
        intraday_mask = np.zeros(self.N_INTRADAY_ACTIONS, dtype=bool)
        for i, level in enumerate(self.INTRADAY_LEVELS):
            power_mw = level * remaining_capacity
            if level >= 0:  # Discharge (sell)
                if power_mw <= max_discharge_mw + 0.01:
                    intraday_mask[i] = True
            else:  # Charge (buy)
                if abs(power_mw) <= max_charge_mw + 0.01:
                    intraday_mask[i] = True
        # I always allow idle (center index = 5)
        intraday_mask[self.N_INTRADAY_ACTIONS // 2] = True

        # Mask 4: mFRR actions (11 options)
        # I apply independent mask (step() clips if combined > capacity)
        mfrr_mask = np.zeros(self.N_MFRR_ACTIONS, dtype=bool)
        for i, level in enumerate(self.MFRR_LEVELS):
            power_mw = level * remaining_capacity
            if level >= 0:  # Discharge (sell)
                if power_mw <= max_discharge_mw + 0.01:
                    mfrr_mask[i] = True
            else:  # Charge (buy)
                if abs(power_mw) <= max_charge_mw + 0.01:
                    mfrr_mask[i] = True
        # I always allow idle (center index = 5)
        mfrr_mask[self.N_MFRR_ACTIONS // 2] = True

        # I concatenate masks for MultiDiscrete format
        full_mask = np.concatenate([afrr_mask, price_mask, intraday_mask, mfrr_mask])

        return full_mask

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """I execute one step in the unified environment."""
        self._check_day_reset()

        # =====================================================================
        # STAGE 1: UNPACK ACTION [afrr, price, intraday, mfrr]
        # =====================================================================
        afrr_action = action[0]
        price_tier_action = action[1]
        intraday_action = action[2]
        mfrr_action = action[3]

        # I get current market data
        row = self.df.iloc[self.current_step]
        dam_commitment = np.clip(row.get('dam_commitment', 0.0),
                                  -self.max_power_mw, self.max_power_mw)

        # I calculate remaining capacity after DAM
        remaining_capacity = self.max_power_mw - abs(dam_commitment)

        # =====================================================================
        # STAGE 2: aFRR COMMITMENT & SELECTION
        # =====================================================================
        new_afrr_level = self.AFRR_LEVELS[afrr_action]
        self.afrr_commitment_level = afrr_action
        self.afrr_commitment_mw = new_afrr_level * remaining_capacity

        # I determine if we're selected for aFRR
        price_tier = self.AFRR_PRICE_TIERS[price_tier_action]
        adjusted_selection_prob = self.selection_probability * (1.5 - price_tier * 0.5)
        adjusted_selection_prob = np.clip(adjusted_selection_prob, 0.1, 0.95)

        self.is_selected_for_afrr = (
            self.afrr_commitment_mw > 0.1 and
            np.random.random() < adjusted_selection_prob
        )

        # aFRR capacity revenue
        afrr_capacity_revenue = 0.0
        if self.is_selected_for_afrr:
            cap_price = row.get('afrr_cap_up_price', 20.0) * price_tier
            afrr_capacity_revenue = self.afrr_commitment_mw * cap_price * self.time_step_hours
            self.afrr_capacity_profit += afrr_capacity_revenue

        # =====================================================================
        # STAGE 3: aFRR ACTIVATION CHECK
        # =====================================================================
        afrr_activated = False
        afrr_direction = None
        afrr_energy_revenue = 0.0
        afrr_energy_delivered = 0.0

        if self.is_selected_for_afrr and self.afrr_commitment_mw > 0.1:
            afrr_activated, afrr_direction = self._check_afrr_activation(row)

        # =====================================================================
        # STAGE 4: EXECUTE DAM COMMITMENT (MANDATORY)
        # =====================================================================
        actual_energy_mw = 0.0
        dam_executed_mw = 0.0

        if abs(dam_commitment) > 0.1:
            available_discharge_mwh = (self.soc - self.min_soc) * self.capacity_mwh
            available_charge_mwh = (self.max_soc - self.soc) * self.capacity_mwh

            if dam_commitment > 0:  # Must discharge
                max_dam_discharge = min(
                    abs(dam_commitment),
                    available_discharge_mwh * self.eff_sqrt / self.time_step_hours
                )
                dam_executed_mw = max_dam_discharge

                energy_mwh = dam_executed_mw * self.time_step_hours
                soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                self.soc = max(self.min_soc, self.soc - soc_delta)

                cycle_fraction = energy_mwh / self.capacity_mwh
                self.total_cycles += cycle_fraction
                self.daily_cycles += cycle_fraction

            else:  # Must charge (dam_commitment < 0)
                max_dam_charge = min(
                    abs(dam_commitment),
                    available_charge_mwh / self.eff_sqrt / self.time_step_hours
                )
                dam_executed_mw = -max_dam_charge

                energy_mwh = max_dam_charge * self.time_step_hours
                soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                self.soc = min(self.max_soc, self.soc + soc_delta)

                cycle_fraction = energy_mwh / self.capacity_mwh
                self.total_cycles += cycle_fraction
                self.daily_cycles += cycle_fraction

            actual_energy_mw = dam_executed_mw
            self.dam_profit += dam_executed_mw * row.get('price', 100.0) * self.time_step_hours

        # =====================================================================
        # STAGE 5: EXECUTE aFRR IF ACTIVATED (takes priority)
        # =====================================================================
        available_discharge_mwh = (self.soc - self.min_soc) * self.capacity_mwh
        available_charge_mwh = (self.max_soc - self.soc) * self.capacity_mwh

        max_discharge = min(
            remaining_capacity,
            available_discharge_mwh * self.eff_sqrt / self.time_step_hours
        )
        max_charge = min(
            remaining_capacity,
            available_charge_mwh / self.eff_sqrt / self.time_step_hours
        )

        if afrr_activated:
            self.hours_since_afrr_activation = 0

            if afrr_direction == 'up':
                afrr_energy_delivered = min(self.afrr_commitment_mw, max_discharge)
                if afrr_energy_delivered > 0.1:
                    energy_mwh = afrr_energy_delivered * self.time_step_hours
                    soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                    old_soc = self.soc
                    self.soc = max(self.min_soc, self.soc - soc_delta)
                    actual_energy = (old_soc - self.soc) * self.capacity_mwh * self.eff_sqrt

                    afrr_price = row.get('afrr_up', 80.0)
                    afrr_energy_revenue = actual_energy * afrr_price

                    cycle_fraction = actual_energy / self.capacity_mwh
                    self.total_cycles += cycle_fraction
                    self.daily_cycles += cycle_fraction

                    afrr_energy_delivered = actual_energy / self.time_step_hours

            else:  # 'down'
                afrr_energy_delivered = min(self.afrr_commitment_mw, max_charge)
                if afrr_energy_delivered > 0.1:
                    energy_mwh = afrr_energy_delivered * self.time_step_hours
                    soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                    old_soc = self.soc
                    self.soc = min(self.max_soc, self.soc + soc_delta)
                    actual_energy = (self.soc - old_soc) * self.capacity_mwh / self.eff_sqrt

                    afrr_price = row.get('afrr_down', 80.0)
                    afrr_energy_revenue = -actual_energy * afrr_price

                    cycle_fraction = actual_energy / self.capacity_mwh
                    self.total_cycles += cycle_fraction
                    self.daily_cycles += cycle_fraction

                    afrr_energy_delivered = -actual_energy / self.time_step_hours

            self.afrr_energy_profit += afrr_energy_revenue
            actual_energy_mw += afrr_energy_delivered

            remaining_after_afrr = remaining_capacity - abs(afrr_energy_delivered)
        else:
            self.hours_since_afrr_activation += 1
            remaining_after_afrr = remaining_capacity

        # =====================================================================
        # STAGE 6: EXECUTE INTRADAY TRADE (voluntary arbitrage)
        # =====================================================================
        intraday_energy_mw = 0.0
        intraday_revenue = 0.0

        if not afrr_activated:
            intraday_level = self.INTRADAY_LEVELS[intraday_action]
            requested_intraday = intraday_level * remaining_after_afrr

            if abs(requested_intraday) > 0.1:
                # I recalculate limits after previous stages
                available_discharge_mwh = (self.soc - self.min_soc) * self.capacity_mwh
                available_charge_mwh = (self.max_soc - self.soc) * self.capacity_mwh

                max_discharge = min(
                    remaining_after_afrr,
                    available_discharge_mwh * self.eff_sqrt / self.time_step_hours
                )
                max_charge = min(
                    remaining_after_afrr,
                    available_charge_mwh / self.eff_sqrt / self.time_step_hours
                )

                if requested_intraday > 0:
                    actual_intraday = min(requested_intraday, max_discharge)
                else:
                    actual_intraday = max(requested_intraday, -max_charge)

                if abs(actual_intraday) > 0.1:
                    if actual_intraday > 0:  # Sell (discharge) at bid
                        energy_mwh = actual_intraday * self.time_step_hours
                        soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = max(self.min_soc, self.soc - soc_delta)
                        actual_energy = (old_soc - self.soc) * self.capacity_mwh * self.eff_sqrt

                        id_price = row.get('intraday_bid', row.get('price', 100.0) - 2.0)
                        intraday_revenue = actual_energy * id_price

                    else:  # Buy (charge) at ask
                        energy_mwh = abs(actual_intraday) * self.time_step_hours
                        soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = min(self.max_soc, self.soc + soc_delta)
                        actual_energy = (self.soc - old_soc) * self.capacity_mwh / self.eff_sqrt

                        id_price = row.get('intraday_ask', row.get('price', 100.0) + 2.0)
                        intraday_revenue = -actual_energy * id_price

                    cycle_fraction = actual_energy / self.capacity_mwh
                    self.total_cycles += cycle_fraction
                    self.daily_cycles += cycle_fraction

                    intraday_energy_mw = actual_intraday
                    self.intraday_profit += intraday_revenue
                    actual_energy_mw += actual_intraday

        # I calculate remaining after IntraDay for mFRR
        remaining_after_intraday = remaining_after_afrr - abs(intraday_energy_mw)

        # =====================================================================
        # STAGE 7: EXECUTE mFRR TRADE (uses remaining after IntraDay)
        # =====================================================================
        mfrr_energy_mw = 0.0
        mfrr_revenue = 0.0

        if not afrr_activated:
            mfrr_level = self.MFRR_LEVELS[mfrr_action]
            requested_mfrr = mfrr_level * remaining_after_intraday

            if abs(requested_mfrr) > 0.1:
                # I recalculate limits after IntraDay
                available_discharge_mwh = (self.soc - self.min_soc) * self.capacity_mwh
                available_charge_mwh = (self.max_soc - self.soc) * self.capacity_mwh

                max_discharge = min(
                    remaining_after_intraday,
                    available_discharge_mwh * self.eff_sqrt / self.time_step_hours
                )
                max_charge = min(
                    remaining_after_intraday,
                    available_charge_mwh / self.eff_sqrt / self.time_step_hours
                )

                if requested_mfrr > 0:
                    actual_mfrr = min(requested_mfrr, max_discharge)
                else:
                    actual_mfrr = max(requested_mfrr, -max_charge)

                if abs(actual_mfrr) > 0.1:
                    if actual_mfrr > 0:  # Sell (discharge) at mfrr_up
                        energy_mwh = actual_mfrr * self.time_step_hours
                        soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = max(self.min_soc, self.soc - soc_delta)
                        actual_energy = (old_soc - self.soc) * self.capacity_mwh * self.eff_sqrt

                        mfrr_price = row.get('mfrr_price_up', 120.0)
                        mfrr_revenue = actual_energy * mfrr_price

                    else:  # Buy (charge) at mfrr_down
                        energy_mwh = abs(actual_mfrr) * self.time_step_hours
                        soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = min(self.max_soc, self.soc + soc_delta)
                        actual_energy = (self.soc - old_soc) * self.capacity_mwh / self.eff_sqrt

                        mfrr_price = row.get('mfrr_price_down', 60.0)
                        mfrr_revenue = -actual_energy * mfrr_price

                    cycle_fraction = actual_energy / self.capacity_mwh
                    self.total_cycles += cycle_fraction
                    self.daily_cycles += cycle_fraction

                    mfrr_energy_mw = actual_mfrr
                    self.mfrr_profit += mfrr_revenue
                    actual_energy_mw += actual_mfrr

        # =====================================================================
        # STAGE 8: CALCULATE REWARD
        # =====================================================================
        market_state = UnifiedMarketState(
            dam_price=row.get('price', 100.0),
            dam_commitment=dam_commitment,
            intraday_bid=row.get('intraday_bid', row.get('price', 100.0) - 2.0),
            intraday_ask=row.get('intraday_ask', row.get('price', 100.0) + 2.0),
            intraday_spread=row.get('intraday_spread', 4.0),
            afrr_cap_up_price=row.get('afrr_cap_up_price', 20.0),
            afrr_cap_down_price=row.get('afrr_cap_down_price', 30.0),
            afrr_energy_up_price=row.get('afrr_up', 80.0),
            afrr_energy_down_price=row.get('afrr_down', 80.0),
            afrr_activated=afrr_activated,
            afrr_activation_direction=afrr_direction,
            mfrr_price_up=row.get('mfrr_price_up', 120.0),
            mfrr_price_down=row.get('mfrr_price_down', 60.0)
        )

        shortfall_risk = self._calculate_shortfall_risk()
        price_momentum = row.get('price_momentum', 0.0)

        reward_info = self.reward_calculator.calculate(
            market=market_state,
            actual_energy_mw=actual_energy_mw,
            afrr_capacity_committed_mw=self.afrr_commitment_mw if self.is_selected_for_afrr else 0.0,
            afrr_energy_delivered_mw=afrr_energy_delivered,
            intraday_energy_mw=intraday_energy_mw,
            mfrr_energy_mw=mfrr_energy_mw,
            current_soc=self.soc,
            capacity_mwh=self.capacity_mwh,
            time_step_hours=self.time_step_hours,
            is_physical_violation=False,
            shortfall_risk=shortfall_risk,
            price_momentum=price_momentum,
            is_selected_for_afrr=self.is_selected_for_afrr
        )

        reward = reward_info['reward']
        self.total_profit += reward_info['components'].get('net_profit', 0)
        self.episode_profit += reward_info['components'].get('net_profit', 0)

        # =====================================================================
        # STAGE 9: ADVANCE STEP
        # =====================================================================
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.episode_length is not None:
            steps_in_episode = self.current_step - self.episode_start_step
            if steps_in_episode >= self.episode_length:
                truncated = True

        obs = self._build_observation()
        info = self._get_info()
        info.update({
            'afrr_activated': afrr_activated,
            'afrr_direction': afrr_direction,
            'is_selected': self.is_selected_for_afrr,
            'actual_energy_mw': actual_energy_mw,
            'intraday_energy_mw': intraday_energy_mw,
            'intraday_profit': intraday_revenue,
            'intraday_revenue': intraday_revenue,
            'mfrr_energy_mw': mfrr_energy_mw,
            'mfrr_revenue': mfrr_revenue,
            **reward_info['components']
        })

        return obs, reward, terminated, truncated, info

    def _check_afrr_activation(self, row: pd.Series) -> Tuple[bool, Optional[str]]:
        """I simulate aFRR activation based on system conditions."""
        # I start with base activation rate
        activation_prob = self.afrr_activation_rate

        # I increase during high imbalance
        imbalance = abs(row.get('net_imbalance_mw', 0))
        if imbalance > 500:
            activation_prob *= self.high_imbalance_boost

        # I increase during peak hours
        hour = self.df.index[self.current_step].hour
        if 17 <= hour <= 21:
            activation_prob *= self.peak_activation_boost

        # I roll the dice
        if np.random.random() < activation_prob:
            # I determine direction based on system need
            afrr_up = row.get('afrr_up', 80)
            afrr_down = row.get('afrr_down', 80)

            # I use imbalance direction if available
            imbalance_mw = row.get('net_imbalance_mw', 0)
            if imbalance_mw > 100:
                return True, 'up'  # System short - need discharge
            elif imbalance_mw < -100:
                return True, 'down'  # System long - need charge
            else:
                # I use price signal
                if afrr_up > afrr_down:
                    return True, 'up'
                else:
                    return True, 'down'

        return False, None

    def _calculate_shortfall_risk(self) -> float:
        """
        I calculate the risk of not meeting future DAM commitments.

        I check both sell-side (discharge) and buy-side (charge) risks over
        a 4-hour lookahead window, consistent with the mask reserve horizon.
        """
        sell_risk = 0.0
        buy_risk = 0.0
        total_sell_commitment = 0.0
        total_buy_commitment = 0.0

        lookahead = min(4, self.max_steps - self.current_step)

        for i in range(1, lookahead + 1):
            future_idx = self.current_step + i
            if future_idx >= len(self.df):
                break
            future_row = self.df.iloc[future_idx]
            dam = np.clip(future_row.get('dam_commitment', 0.0),
                          -self.max_power_mw, self.max_power_mw)
            if dam > 0.1:
                total_sell_commitment += dam
            elif dam < -0.1:
                total_buy_commitment += abs(dam)

        # I assess sell-side risk (not enough energy to discharge)
        if total_sell_commitment > 0:
            available_energy = (self.soc - self.min_soc) * self.capacity_mwh
            needed_energy = total_sell_commitment * self.time_step_hours / self.eff_sqrt
            if available_energy < needed_energy:
                sell_risk = (needed_energy - available_energy) / needed_energy
                sell_risk = np.clip(sell_risk, 0, 1)

        # I assess buy-side risk (not enough headroom to charge)
        if total_buy_commitment > 0:
            available_headroom = (self.max_soc - self.soc) * self.capacity_mwh
            needed_headroom = total_buy_commitment * self.time_step_hours * self.eff_sqrt
            if available_headroom < needed_headroom:
                buy_risk = (needed_headroom - available_headroom) / needed_headroom
                buy_risk = np.clip(buy_risk, 0, 1)

        return max(sell_risk, buy_risk)

    def _calculate_dam_soc_reserve(self) -> Tuple[float, float]:
        """
        I calculate dynamic SoC reserves to guarantee future DAM commitment execution.

        I look 4 hours ahead and compute:
        - min_soc_reserve: raised above self.min_soc when future SELL commitments exist
          (I must keep enough energy to discharge later)
        - max_soc_reserve: lowered below self.max_soc when future BUY commitments exist
          (I must keep enough headroom to charge later)

        I also account for intermediate charge/discharge potential (50% discount)
        so the reserve is not overly conservative when the agent has time to prepare.
        """
        min_soc_reserve = self.min_soc
        max_soc_reserve = self.max_soc

        lookahead = min(4, self.max_steps - self.current_step)
        if lookahead <= 0:
            return min_soc_reserve, max_soc_reserve

        # I first scan to find the last sell/buy commitment positions
        # so I only credit idle hours that occur BEFORE a commitment
        commitments = []
        for i in range(1, lookahead + 1):
            future_idx = self.current_step + i
            if future_idx >= len(self.df):
                break
            future_row = self.df.iloc[future_idx]
            dam = np.clip(future_row.get('dam_commitment', 0.0),
                          -self.max_power_mw, self.max_power_mw)
            commitments.append((i, dam))

        # I find last sell and last buy positions in the lookahead window
        last_sell_pos = 0
        last_buy_pos = 0
        for pos, dam in commitments:
            if dam > 0.1:
                last_sell_pos = pos
            elif dam < -0.1:
                last_buy_pos = pos

        # I accumulate energy needs and intermediate potential
        sell_energy_needed = 0.0
        buy_headroom_needed = 0.0
        intermediate_charge_potential = 0.0
        intermediate_discharge_potential = 0.0

        for pos, dam in commitments:
            if dam > 0.1:
                # I have a future sell commitment — need energy to discharge
                energy_needed = dam * self.time_step_hours / self.eff_sqrt
                sell_energy_needed += energy_needed
            elif dam < -0.1:
                # I have a future buy commitment — need headroom to charge
                energy_needed = abs(dam) * self.time_step_hours * self.eff_sqrt
                buy_headroom_needed += energy_needed
            else:
                # I have an idle hour — I can only use it to prepare if it occurs
                # BEFORE a future commitment (can't prepare after the fact)
                potential = self.max_power_mw * self.time_step_hours * 0.5
                if pos < last_sell_pos:
                    intermediate_charge_potential += potential * self.eff_sqrt
                if pos < last_buy_pos:
                    intermediate_discharge_potential += potential / self.eff_sqrt

        # I subtract intermediate potential from needs (agent can prepare)
        net_sell_energy = max(0.0, sell_energy_needed - intermediate_charge_potential)
        net_buy_headroom = max(0.0, buy_headroom_needed - intermediate_discharge_potential)

        # I convert energy needs to SoC reserves
        sell_soc_reserve = net_sell_energy / self.capacity_mwh
        buy_soc_reserve = net_buy_headroom / self.capacity_mwh

        min_soc_reserve = min(self.max_soc, self.min_soc + sell_soc_reserve)
        max_soc_reserve = max(self.min_soc, self.max_soc - buy_soc_reserve)

        # I ensure min <= max (edge case with very large commitments)
        if min_soc_reserve > max_soc_reserve:
            midpoint = (min_soc_reserve + max_soc_reserve) / 2.0
            min_soc_reserve = midpoint
            max_soc_reserve = midpoint

        return min_soc_reserve, max_soc_reserve

    def _build_observation(self) -> np.ndarray:
        """
        I build the 63-feature observation vector.

        Feature Groups:
        1. Battery State (3): soc, max_discharge, max_charge
        2. Market Prices (11): dam, id_bid, id_ask, id_spread, id_volume, mfrr_up, mfrr_down, mfrr_spread, id_dam_spread, mfrr_dam_premium, mfrr_id_spread
        3. Time Encoding (4): hour_sin/cos, dow_sin/cos
        4. DAM Lookahead (13): current + 12h commitments
        5. Price Lookahead (12): 12h price forecasts
        6. aFRR State (8): cap_prices, activation, commitment, selection, signal, hours_since
        7. Risk/Imbalance (4): dam_discharge_reserve, shortfall_risk, momentum, worthiness
        8. Gate Closure (4): dam_open, intraday_hours, afrr_hours, mfrr_open
        9. Timing Signals (4): price_vs_typical, is_peak, is_solar, hours_to_max
        """
        row = self.df.iloc[self.current_step]
        ts = self.df.index[self.current_step]

        features = []

        # =====================================================================
        # 1. BATTERY STATE (3 features)
        # =====================================================================
        features.append(self.soc)

        dam_commitment = np.clip(row.get('dam_commitment', 0.0),
                                  -self.max_power_mw, self.max_power_mw)
        remaining_capacity = self.max_power_mw - abs(dam_commitment)

        available_discharge = (self.soc - self.min_soc) * self.capacity_mwh
        available_charge = (self.max_soc - self.soc) * self.capacity_mwh

        max_discharge = min(remaining_capacity,
                           available_discharge * self.eff_sqrt / self.time_step_hours)
        max_charge = min(remaining_capacity,
                        available_charge / self.eff_sqrt / self.time_step_hours)

        features.append(max_discharge / self.max_power_mw)
        features.append(max_charge / self.max_power_mw)

        # =====================================================================
        # 2. MARKET PRICES (11 features — was 6)
        # =====================================================================
        dam_price = row.get('price', 100.0)
        features.append(dam_price / 100.0)  # [3] DAM price

        # IntraDay prices (lagged to avoid lookahead — NO fallback to current-step)
        id_bid = row.get('intraday_bid_lag_1h', 0.0)
        id_ask = row.get('intraday_ask_lag_1h', 0.0)
        id_spread = row.get('intraday_spread_lag_1h', 0.0)
        id_volume = row.get('intraday_volume_lag_1h', 0.5)
        features.append(id_bid / 100.0)      # [4] IntraDay bid
        features.append(id_ask / 100.0)      # [5] IntraDay ask
        features.append(id_spread / 100.0)   # [6] IntraDay spread
        features.append(id_volume)            # [7] IntraDay volume (already 0-1)

        # mFRR prices (lagged — NO fallback to current-step)
        mfrr_up = row.get('mfrr_price_up_lag_1h', 0.0)
        mfrr_down = row.get('mfrr_price_down_lag_1h', 0.0)
        features.append(mfrr_up / 100.0)     # [8] mFRR up
        features.append(mfrr_down / 100.0)   # [9] mFRR down

        # mFRR spread
        features.append((mfrr_up - mfrr_down) / 100.0)  # [10] mFRR spread

        # Cross-market signals (help agent pick the best market)
        features.append((id_bid - dam_price) / 100.0)    # [11] IntraDay-DAM spread
        features.append((mfrr_up - dam_price) / 100.0)   # [12] mFRR premium over DAM
        features.append((mfrr_up - id_ask) / 100.0)      # [13] mFRR-IntraDay spread

        # =====================================================================
        # 3. TIME ENCODING (4 features)
        # =====================================================================
        hour = ts.hour
        dow = ts.dayofweek

        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # =====================================================================
        # 4. DAM LOOKAHEAD (13 features: current + 12h)
        # =====================================================================
        features.append(dam_commitment / self.max_power_mw)  # Current

        # I only show same-day commitments (HEnEx compliant)
        current_day = ts.date()
        for i in range(1, 13):
            target = min(self.current_step + i, len(self.df) - 1)
            target_ts = self.df.index[target]
            if target_ts.date() == current_day:
                future_dam = self.df.iloc[target].get('dam_commitment', 0.0)
                future_dam = np.clip(future_dam, -self.max_power_mw, self.max_power_mw)
            else:
                future_dam = 0.0  # Unknown (not yet submitted)
            features.append(future_dam / self.max_power_mw)

        # =====================================================================
        # 5. PRICE LOOKAHEAD (12 features) - HEnEx compliant
        # =====================================================================
        # DAM prices are published at 14:00 for the NEXT day:
        # - If hour >= 14:00: I know today + tomorrow prices
        # - If hour < 14:00: I only know today prices
        # For unknown prices, I use persistence forecast (last known price)
        current_hour = hour
        from datetime import timedelta

        # I determine the last day for which DAM prices are known
        if current_hour >= 14:
            # After 14:00: prices for tomorrow are published
            last_known_day = (ts + timedelta(days=1)).date()
        else:
            # Before 14:00: only today's prices known
            last_known_day = current_day

        for i in range(1, 13):
            target = min(self.current_step + i, len(self.df) - 1)
            target_ts = self.df.index[target]
            target_day = target_ts.date()

            if target_day <= last_known_day:
                # Price is known (DAM already published)
                future_price = self.df.iloc[target].get('price', dam_price)
            else:
                # Price NOT known yet - use persistence forecast
                # I use same hour from last known day as forecast
                future_price = dam_price  # Simple persistence: use current price

            features.append(future_price / 100.0)

        # =====================================================================
        # 6. aFRR STATE (8 features)
        # =====================================================================
        # Capacity prices (lagged)
        cap_up = row.get('afrr_cap_up_price', 20.0)
        cap_down = row.get('afrr_cap_down_price', 30.0)
        features.append(cap_up / 50.0)
        features.append(cap_down / 50.0)

        # Activation signal (from lagged imbalance)
        imbalance = row.get('net_imbalance_mw', 0)
        features.append(np.clip(imbalance / 1000.0, -1, 1))

        # Our commitment
        features.append(self.afrr_commitment_mw / self.max_power_mw)

        # Selection probability estimate
        cap_percentile = row.get('afrr_cap_percentile', 0.5)
        features.append(cap_percentile)

        # Activation signal (binary indicator based on recent activations)
        features.append(1.0 if self.hours_since_afrr_activation < 4 else 0.0)

        # Hours since last activation (normalized)
        features.append(min(self.hours_since_afrr_activation, 24) / 24.0)

        # Is currently selected
        features.append(1.0 if self.is_selected_for_afrr else 0.0)

        # =====================================================================
        # 7. RISK/IMBALANCE (4 features)
        # =====================================================================
        # DAM discharge reserve — how much SoC the mask is reserving for future sells
        dam_min_soc, dam_max_soc = self._calculate_dam_soc_reserve()
        dam_discharge_reserve = (dam_min_soc - self.min_soc) / (self.max_soc - self.min_soc)
        features.append(np.clip(dam_discharge_reserve, 0, 1))

        # Shortfall risk
        shortfall_risk = self._calculate_shortfall_risk()
        features.append(shortfall_risk)

        # Price momentum
        momentum = row.get('price_momentum', 0.0)
        features.append(np.clip(momentum, -1, 1))

        # Trade worthiness (volatility indicator)
        price_std = row.get('price_std_24h', 20.0)
        worthiness = min(price_std / 50.0, 1.0)
        features.append(worthiness)

        # =====================================================================
        # 8. GATE CLOSURE (4 features)
        # =====================================================================
        # DAM gate (always closed for current day, open for D+1 until 14:00)
        dam_gate_open = 1.0 if hour < 14 else 0.0
        features.append(dam_gate_open)

        # IntraDay gate hours remaining (H-1 before delivery)
        intraday_hours_remaining = max(0, 24 - hour) / 24.0
        features.append(intraday_hours_remaining)

        # aFRR gate hours (H-0.25 before delivery)
        afrr_hours_remaining = max(0, 23.75 - hour) / 24.0
        features.append(afrr_hours_remaining)

        # mFRR gate open
        mfrr_gate_open = 1.0 if hour < 23 else 0.0
        features.append(mfrr_gate_open)

        # =====================================================================
        # 9. TIMING SIGNALS (4 features)
        # =====================================================================
        # Price vs typical hour
        mean_24h = row.get('price_mean_24h', 100.0)
        if mean_24h > 1:
            price_vs_typical = (dam_price - mean_24h) / mean_24h
        else:
            price_vs_typical = 0.0
        features.append(np.clip(price_vs_typical, -1, 1))

        # Is peak hour
        is_peak = 1.0 if 17 <= hour <= 21 else 0.0
        features.append(is_peak)

        # Is solar hour
        is_solar = 1.0 if 9 <= hour <= 15 else 0.0
        features.append(is_solar)

        # Hours to daily max price (estimated from pattern)
        # Peak typically at hour 19-20
        hours_to_peak = min(abs(19 - hour), abs(19 - hour + 24)) / 12.0
        features.append(hours_to_peak)

        # =====================================================================
        # VALIDATION
        # =====================================================================
        obs = np.array(features, dtype=np.float32)

        assert len(obs) == 63, f"Expected 63 features, got {len(obs)}"

        # I replace any NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    def _get_info(self) -> Dict:
        """I return step information."""
        return {
            'soc': self.soc,
            'step': self.current_step,
            'afrr_commitment': self.afrr_commitment_mw,
            'is_selected': self.is_selected_for_afrr,
            'total_profit': self.total_profit,
            'dam_profit': self.dam_profit,
            'intraday_profit': self.intraday_profit,
            'afrr_capacity_profit': self.afrr_capacity_profit,
            'afrr_energy_profit': self.afrr_energy_profit,
            'mfrr_profit': self.mfrr_profit,
            'total_cycles': self.total_cycles,
            'daily_cycles': self.daily_cycles
        }

    def render(self, mode='human'):
        """I render the environment state."""
        row = self.df.iloc[self.current_step]
        print(f"Step {self.current_step}: "
              f"SoC={self.soc:.1%}, "
              f"aFRR={self.afrr_commitment_mw:.1f}MW, "
              f"Selected={self.is_selected_for_afrr}, "
              f"Price={row.get('price', 0):.1f}EUR, "
              f"Profit={self.total_profit:,.0f}EUR")


def create_unified_env(
    data_path: str = "data/unified_multimarket_training.csv",
    **kwargs
) -> BatteryEnvUnified:
    """Factory function to create the unified environment."""
    import pandas as pd
    from pathlib import Path

    # I load the data
    path = Path(data_path)
    if not path.exists():
        # Try relative to project root
        path = Path(__file__).parent.parent / data_path

    print(f"Loading unified training data from {path}...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    return BatteryEnvUnified(df, **kwargs)
