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
3. Hierarchical capacity cascading: DAM -> aFRR -> mFRR -> IntraDay
4. aFRR activation simulation with MANDATORY response requirement
5. HEnEx-compliant gate closure awareness

Action Space: MultiDiscrete([5, 5, 11, 11]) = 3,025 combinations (legacy)
             MultiDiscrete([5, 5, 11, 11, 11, 5]) = 166,375 (full market)
- [0] aFRR Commitment: 0%, 25%, 50%, 75%, 100% of remaining capacity
- [1] aFRR Price Tier: 5 levels from aggressive to conservative bidding
- [2] IntraDay Action: 11 levels (-1.0 to +1.0) × remaining capacity after aFRR
- [3] mFRR Qty: 11 levels — TSO-activated, pay-as-cleared, mandatory
- [4] Free Bid Qty: 11 levels — agent-submitted, pay-as-bid (full market only)
- [5] Free Bid Price Tier: 5 levels (full market only)

Observation Space: 64-100 features (see _build_observation for details)
  Base: 64 | +forecast: 9 | +market_forecast: 20 | +full_market: 12
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
    # Lower = more aggressive bid (higher chance of selection)
    # Higher = more conservative bid (lower chance of selection)
    # NOTE: Payment is MARGINAL (market clearing price), NOT bid price.
    # Price tier only affects selection probability, not revenue per MW.
    AFRR_PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])  # 5 tiers

    # I define separate IntraDay and mFRR action levels (~6 MW granularity per level)
    # I widen the no-trade dead zone: 3 center actions (4,5,6) all map to 0.0
    # This gives 27% base no-trade probability instead of 9%, helping the agent
    # learn that IntraDay trades are only profitable when volatility > spread
    INTRADAY_LEVELS = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0])
    MFRR_LEVELS = np.linspace(-1.0, 1.0, 11)  # 11 levels
    N_INTRADAY_ACTIONS = 11
    N_MFRR_ACTIONS = 11

    # I define Free Bid quantity levels (full market mode, same granularity as mFRR)
    FREEBID_QTY_LEVELS = np.linspace(-1.0, 1.0, 11)  # 11 levels
    N_FREEBID_QTY_ACTIONS = 11

    # I define Free Bid price tier multipliers (full market mode)
    FREEBID_PRICE_TIERS = np.array([0.8, 0.9, 1.0, 1.1, 1.2])  # 5 tiers
    N_FREEBID_PRICE_TIERS = 5

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

        # mFRR parameters
        mfrr_imbalance_threshold: float = 10.0,  # MW minimum |imbalance| for mFRR
        mfrr_activation_rate: float = 0.35,  # Probability of mFRR bid acceptance (35%)
        mfrr_price_cap: float = 1000.0,  # EUR/MWh cap — real prices reach 1190 EUR

        # Cycling limits — 2.5 gives breathing room for mandatory DAM + aFRR
        # while the cycle_target (2.0) still penalizes excess via reward shaping
        max_daily_cycles: float = 2.5,
        degradation_cost: float = 15.0,

        # Reward configuration
        reward_config: Optional[Dict] = None,

        # Feature configuration
        lookahead_hours: int = 12,  # Price and DAM lookahead window

        # Training configuration
        episode_length: Optional[int] = None,  # None = full dataset
        random_start: bool = True,

        # Full market mode (IDA1/2/3/XBID + Free Bids)
        enable_full_market: bool = False,

        # IntraDay forecast features
        enable_forecast: bool = False,
        forecaster_path: str = "models/intraday_forecaster.pkl",
        forecast_noise: bool = True,

        # Market forecast features (Phase 3: DAM+ID+Imbalance)
        enable_market_forecast: bool = False,
        market_forecaster_path: str = "models/market_forecaster.pkl",

        # Endogenous DAM (Phase 4: agent-decided commitments)
        enable_endogenous_dam: bool = False,
        dam_bidder_min_spread: float = 30.0,
    ):
        super().__init__()

        # I store full market mode flag
        self.enable_full_market = enable_full_market

        # I store forecast configuration
        self.enable_forecast = enable_forecast
        self.forecast_noise = forecast_noise
        self.price_forecaster = None
        self._res_mean_24h = None
        if enable_forecast:
            try:
                from models.intraday_forecaster import IntraDayForecaster
                self.price_forecaster = IntraDayForecaster.load(forecaster_path)
            except Exception as e:
                print(f"  Warning: Forecaster not loaded: {e}. Using persistence.")

        # I store market forecast configuration (Phase 3)
        self.enable_market_forecast = enable_market_forecast
        self.market_forecaster = None
        if enable_market_forecast:
            try:
                from models.market_forecaster import MarketForecaster
                self.market_forecaster = MarketForecaster.load(market_forecaster_path)
                print(f"  MarketForecaster loaded from {market_forecaster_path}")
            except Exception as e:
                print(f"  Warning: MarketForecaster not loaded: {e}. Using pre-computed columns.")

        # I store endogenous DAM configuration (Phase 4)
        self.enable_endogenous_dam = enable_endogenous_dam
        self.dam_bidder_min_spread = dam_bidder_min_spread
        self.dam_schedule = None  # I store the 24h schedule generated at episode reset

        # I store configuration
        self.df = df.copy()
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        self.efficiency = efficiency
        self.eff_sqrt = np.sqrt(efficiency)
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.time_step_hours = time_step_hours
        # I compute steps-per-hour once at init for parameterized time resolution
        self._sph = int(round(1.0 / self.time_step_hours))  # 4 for 15-min, 1 for 1-hour
        self.warmup_steps = warmup_steps

        # aFRR parameters
        self.selection_probability = selection_probability
        self.afrr_activation_rate = afrr_activation_rate
        self.peak_activation_boost = peak_activation_boost
        self.high_imbalance_boost = high_imbalance_boost

        # mFRR parameters
        self.mfrr_imbalance_threshold = mfrr_imbalance_threshold
        self.mfrr_activation_rate = mfrr_activation_rate
        self.mfrr_price_cap = mfrr_price_cap

        # Cycling
        self.max_daily_cycles = max_daily_cycles
        self.degradation_cost = degradation_cost

        # Reward calculator
        self.reward_config = reward_config or {}
        self.reward_calculator = UnifiedRewardCalculator(
            degradation_cost_per_mwh=self.reward_config.get('degradation_cost', 5.0),
            dam_violation_penalty=self.reward_config.get('dam_violation_penalty', 800.0),
            afrr_nonresponse_penalty=self.reward_config.get('afrr_nonresponse_penalty', 500.0),
            cycle_target=self.reward_config.get('cycle_target', 2.0),
            cycle_excess_penalty=self.reward_config.get('cycle_excess_penalty', 3000.0),
            calendar_aging_coefficient=self.reward_config.get('calendar_aging_coefficient', 50.0),
            reward_scale=self.reward_config.get('reward_scale', 0.01)
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

        mode_str = "FULL MARKET (IDA1/2/3/XBID/FreeBids)" if enable_full_market else "IntraDay/mFRR separated"
        print(f"Unified Multi-Market Environment Initialized ({mode_str})")
        print(f"  Battery: {capacity_mwh} MWh, {max_power_mw} MW")
        print(f"  Action Space: MultiDiscrete({list(self.action_space.nvec)}) = {np.prod(self.action_space.nvec)} combos")
        print(f"  Observation Space: {self.observation_space.shape[0]} features")
        print(f"  aFRR: selection={selection_probability:.0%}, activation={afrr_activation_rate:.0%}")
        print(f"  mFRR: threshold={mfrr_imbalance_threshold:.0f} MW, activation={mfrr_activation_rate:.0%}, price_cap={mfrr_price_cap:.0f} EUR/MWh")
        if enable_forecast:
            print(f"  Forecast: IntraDay LightGBM ({forecaster_path})")
        if enable_market_forecast:
            print(f"  MarketForecast: DAM+ID+Imbalance (+15 obs features)")
        if enable_endogenous_dam:
            print(f"  Endogenous DAM: agent-decided commitments (min_spread={dam_bidder_min_spread} EUR)")
        days = len(self.df) * self.time_step_hours / 24
        print(f"  Data: {len(self.df)} steps ({days:.0f} days, {self.time_step_hours*60:.0f}-min resolution)")

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

        # I reconstruct net_imbalance_mw from activated balancing energy when
        # the column exists but is all-zero (data pipeline gap before Sep 2025).
        # net_balancing_energy_mwh is the net activated energy (up - down) published
        # by ADMIE ex-post; it correlates perfectly with real-time system imbalance
        # and is available for every ISP in the training period.
        imb = df['net_imbalance_mw']
        if imb.abs().max() < 1.0 and 'net_balancing_energy_mwh' in df.columns:
            df['net_imbalance_mw'] = df['net_balancing_energy_mwh']
            n_active = (df['net_imbalance_mw'].abs() > self.mfrr_imbalance_threshold).sum()
            print(f"  Reconstructed net_imbalance_mw from net_balancing_energy_mwh "
                  f"({n_active}/{len(df)} rows exceed {self.mfrr_imbalance_threshold} MW threshold)")

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

        # I scale rolling windows by steps-per-hour for resolution independence
        w24 = 24 * self._sph      # 96 steps for 24h at 15-min, 24 at 1h
        w168 = 168 * self._sph    # 672 steps for 1 week at 15-min, 168 at 1h
        lag6 = 6 * self._sph      # 24 steps for 6h momentum at 15-min, 6 at 1h

        # I add rolling statistics (backward-looking only!)
        if 'price_mean_24h' not in df.columns:
            df['price_mean_24h'] = df['price'].rolling(w24, min_periods=1).mean()
        if 'price_std_24h' not in df.columns:
            df['price_std_24h'] = df['price'].rolling(w24, min_periods=1).std().fillna(10)
        if 'price_min_24h' not in df.columns:
            df['price_min_24h'] = df['price'].rolling(w24, min_periods=1).min()
        if 'price_max_24h' not in df.columns:
            df['price_max_24h'] = df['price'].rolling(w24, min_periods=1).max()

        # I compute price momentum (backward-looking)
        if 'price_momentum' not in df.columns:
            price_6h_ago = df['price'].shift(lag6)
            df['price_momentum'] = (df['price'] - price_6h_ago) / (price_6h_ago + 1)
            df['price_momentum'] = df['price_momentum'].fillna(0).clip(-1, 1)

        # I compute aFRR capacity price percentile
        if 'afrr_cap_percentile' not in df.columns:
            df['afrr_cap_percentile'] = df['afrr_cap_up_price'].rolling(w168, min_periods=w24).rank(pct=True).fillna(0.5)

        # I pre-compute RES 24h rolling mean for forecast deviation feature
        if self.enable_forecast:
            w24 = 24 * self._sph
            if 'res_total_mw' in df.columns:
                self._res_mean_24h = df['res_total_mw'].rolling(w24, min_periods=1).mean().values
            elif 'solar' in df.columns and 'wind_onshore' in df.columns:
                res = df['solar'].fillna(0) + df['wind_onshore'].fillna(0)
                self._res_mean_24h = res.rolling(w24, min_periods=1).mean().values
            else:
                self._res_mean_24h = None

        # I fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        self.df = df

        # I calculate step limits
        self.start_step = self.warmup_steps
        self.max_steps = len(self.df) - 1 - self.lookahead_steps

    def _setup_spaces(self):
        """I define action and observation spaces."""
        if self.enable_full_market:
            # I use 6-dim action space: mFRR and Free Bids are independent markets
            # mFRR = TSO-activated, pay-as-cleared, mandatory response
            # Free Bids = agent-submitted to balancing auction, pay-as-bid, merit-order
            self.action_space = spaces.MultiDiscrete([
                len(self.AFRR_LEVELS),          # 5 aFRR commitment levels
                len(self.AFRR_PRICE_TIERS),     # 5 aFRR price tiers
                self.N_INTRADAY_ACTIONS,        # 11 XBID actions
                self.N_MFRR_ACTIONS,            # 11 mFRR qty (TSO-activated)
                self.N_FREEBID_QTY_ACTIONS,     # 11 Free Bid qty (agent-submitted)
                self.N_FREEBID_PRICE_TIERS      # 5 Free Bid price tiers
            ])
        else:
            # I use legacy 4-dim action space
            self.action_space = spaces.MultiDiscrete([
                len(self.AFRR_LEVELS),      # 5 aFRR commitment levels
                len(self.AFRR_PRICE_TIERS), # 5 aFRR price tiers
                self.N_INTRADAY_ACTIONS,    # 11 IntraDay actions
                self.N_MFRR_ACTIONS         # 11 mFRR actions
            ])

        # I compute dynamic observation size based on enabled feature groups
        n_obs = 64  # Base features (Groups 1-9) + cycles_remaining
        if self.enable_forecast:
            n_obs += 9   # IntraDay forecast + RES fundamentals
        if self.enable_market_forecast:
            n_obs += 20  # DAM+ID+Imbalance forecast + ISP cascade features (Group 10b)
        if self.enable_full_market:
            n_obs += 12  # IDA/XBID/FreeBid features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

    def _reset_state(self):
        """I initialize/reset the environment state."""
        self.current_step = self.start_step
        self.soc = 0.5
        self.afrr_commitment_mw = 0.0
        self.afrr_commitment_level = 0  # Index into AFRR_LEVELS
        self.is_selected_for_afrr = False
        self.steps_since_afrr_activation = 24 * self._sph  # Start with no recent activation

        # I lock aFRR commitment for 4-hour blocks (6 blocks/day) per real market rules.
        # The agent can only change commitment at block boundaries (00:00, 04:00, 08:00, 12:00, 16:00, 20:00).
        self._afrr_block_steps = 4 * self._sph  # 4 hours * steps_per_hour
        self._afrr_steps_remaining = 0  # Steps until next re-commitment allowed

        # I track profits by market
        self.total_profit = 0.0
        self.dam_profit = 0.0
        self.intraday_profit = 0.0
        self.afrr_capacity_profit = 0.0
        self.afrr_energy_profit = 0.0
        self.mfrr_profit = 0.0

        # I track full market mode profits
        self.ida_profit = 0.0
        self.xbid_profit = 0.0
        self.free_bid_profit = 0.0
        self.free_bid_activations = 0
        self.free_bid_submissions = 0

        # I track cycling
        self.total_cycles = 0.0
        self.daily_cycles = 0.0
        self.current_day = None

        # I track episode
        self.episode_start_step = self.current_step
        self.episode_profit = 0.0

        # I track SoC extremes over the episode
        self.episode_min_soc = 1.0
        self.episode_max_soc = 0.0

    def reset(self, seed=None, options=None):
        """I reset the environment for a new episode."""
        super().reset(seed=seed)

        # I determine starting position
        if self.random_start:
            max_start = self.max_steps - (self.episode_length or 168 * self._sph)  # Default 1 week
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

        # I generate endogenous DAM schedule if enabled (Phase 4)
        if self.enable_endogenous_dam:
            self.dam_schedule = self._generate_endogenous_dam_schedule()

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
        Legacy: 5 + 5 + 11 + 11 = 32
        Full market: 5 + 5 + 11 + 11 + 11 + 5 = 48
        """
        row = self.df.iloc[self.current_step]

        # I get current DAM commitment (consumes capacity)
        # I use endogenous schedule when available (Phase 4)
        dam_commitment = self._get_dam_commitment(self.current_step)
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

        # I apply daily cycle limits, pre-subtracting the current step's mandatory
        # DAM cycle cost and a buffer for possible aFRR activation. Without this,
        # the mask underestimates cycle consumption because DAM executes AFTER
        # masking (Stage 4) and aFRR activation is stochastic (Stage 5).
        dam_cycle_cost = abs(dam_commitment) * self.time_step_hours / self.capacity_mwh
        afrr_buffer = self.afrr_commitment_mw * self.time_step_hours / self.capacity_mwh * 0.15
        cycles_remaining = max(0, self.max_daily_cycles - self.daily_cycles
                               - dam_cycle_cost - afrr_buffer)
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

        # I subtract aFRR commitment to prevent cross-market over-commitment.
        # Without this, the agent can commit 100% aFRR AND max IntraDay AND max
        # mFRR simultaneously — all pass masking individually but together exceed
        # physical limits, draining the battery in 3 hours.
        remaining_for_trading = max(0, remaining_capacity - self.afrr_commitment_mw)
        # I also reduce SoC-based limits to reserve energy for aFRR activation
        afrr_energy_reserve = self.afrr_commitment_mw * self.time_step_hours
        adjusted_max_discharge = min(
            remaining_for_trading,
            max(0, available_discharge_mwh - afrr_energy_reserve / self.eff_sqrt)
            * self.eff_sqrt / self.time_step_hours
        )
        adjusted_max_charge = min(
            remaining_for_trading,
            max(0, available_charge_mwh - afrr_energy_reserve * self.eff_sqrt)
            / self.eff_sqrt / self.time_step_hours
        )

        # Mask 2: aFRR price tiers (5 options)
        # I always allow all price tiers (no physical constraints)
        price_mask = np.ones(len(self.AFRR_PRICE_TIERS), dtype=bool)

        # Mask 3: IntraDay actions (11 options)
        # I enforce IntraDay gate: open 00:00-22:59, closed 23:00+ and during aFRR
        # I use adjusted capacity/limits that account for aFRR commitment
        intraday_mask = np.zeros(self.N_INTRADAY_ACTIONS, dtype=bool)

        intraday_open = self._is_intraday_open(row)
        if intraday_open:
            for i, level in enumerate(self.INTRADAY_LEVELS):
                power_mw = level * remaining_for_trading
                if level >= 0:  # Discharge (sell)
                    if power_mw <= adjusted_max_discharge + 0.01:
                        intraday_mask[i] = True
                else:  # Charge (buy)
                    if abs(power_mw) <= adjusted_max_charge + 0.01:
                        intraday_mask[i] = True
        # I always allow idle (center index = 5)
        intraday_mask[self.N_INTRADAY_ACTIONS // 2] = True

        # Mask 4: mFRR actions (11 options) — constrained by real activation data
        # I use historical activation volumes from ADMIE data to determine mFRR
        # availability. Real data shows 84% UP and 96% DOWN activation rates
        # (both directions can be active simultaneously).
        # When activation columns are missing, I fall back to the legacy
        # imbalance-based heuristic for backward compatibility.
        # I use adjusted capacity/limits that account for aFRR commitment
        mfrr_mask = np.zeros(self.N_MFRR_ACTIONS, dtype=bool)

        # I pre-define fallback variables so they're available for Free Bid mask too
        mfrr_activated_up = False
        mfrr_activated_down = False

        # I check for real activation data first (preferred)
        has_activation_data = ('mfrr_activated_up_mwh' in row.index
                               if hasattr(row, 'index') else
                               'mfrr_activated_up_mwh' in row)
        if has_activation_data:
            mfrr_activated_up = row.get('mfrr_activated_up_mwh', 0.0) > 0
            mfrr_activated_down = row.get('mfrr_activated_down_mwh', 0.0) > 0

            if mfrr_activated_up:
                # I allow discharge (sell/UP) levels
                for i, level in enumerate(self.MFRR_LEVELS):
                    if level > 0:
                        power_mw = level * remaining_for_trading
                        if power_mw <= adjusted_max_discharge + 0.01:
                            mfrr_mask[i] = True

            if mfrr_activated_down:
                # I allow charge (buy/DOWN) levels — note: NOT elif, both can be active
                for i, level in enumerate(self.MFRR_LEVELS):
                    if level < 0:
                        power_mw = abs(level) * remaining_for_trading
                        if power_mw <= adjusted_max_charge + 0.01:
                            mfrr_mask[i] = True
        else:
            # I fall back to legacy imbalance-based heuristic
            imbalance = row.get('net_imbalance_mw_lag_1h',
                                row.get('system_deviation_mwh_lag_1h',
                                        row.get('net_imbalance_mw', 0.0)))
            mfrr_up_price = row.get('mfrr_price_up_lag_1h',
                                    row.get('mfrr_price_up', 0.0))
            mfrr_down_price = row.get('mfrr_price_down_lag_1h',
                                      row.get('mfrr_price_down', 0.0))

            if imbalance < -self.mfrr_imbalance_threshold and mfrr_up_price > 1.0:
                for i, level in enumerate(self.MFRR_LEVELS):
                    if level > 0:
                        power_mw = level * remaining_for_trading
                        if power_mw <= adjusted_max_discharge + 0.01:
                            mfrr_mask[i] = True
            elif imbalance > self.mfrr_imbalance_threshold and mfrr_down_price > 1.0:
                for i, level in enumerate(self.MFRR_LEVELS):
                    if level < 0:
                        power_mw = abs(level) * remaining_for_trading
                        if power_mw <= adjusted_max_charge + 0.01:
                            mfrr_mask[i] = True

        mfrr_mask[self.N_MFRR_ACTIONS // 2] = True  # I always allow idle

        # I concatenate masks for MultiDiscrete format
        if self.enable_full_market:
            # Mask 5: Free Bid qty (11 options) — independent from mFRR
            # I use the same capacity pool and direction logic as mFRR but they're
            # separate markets: mFRR is TSO-activated, Free Bids are agent-submitted.
            freebid_qty_mask = np.zeros(self.N_FREEBID_QTY_ACTIONS, dtype=bool)

            if has_activation_data:
                # I allow Free Bid directions based on TSO activation data
                if mfrr_activated_up:
                    for i, level in enumerate(self.FREEBID_QTY_LEVELS):
                        if level > 0:
                            power_mw = level * remaining_for_trading
                            if power_mw <= adjusted_max_discharge + 0.01:
                                freebid_qty_mask[i] = True
                if mfrr_activated_down:
                    for i, level in enumerate(self.FREEBID_QTY_LEVELS):
                        if level < 0:
                            power_mw = abs(level) * remaining_for_trading
                            if power_mw <= adjusted_max_charge + 0.01:
                                freebid_qty_mask[i] = True
            else:
                # I fall back to legacy imbalance-based heuristic for Free Bids
                if imbalance < -self.mfrr_imbalance_threshold and mfrr_up_price > 1.0:
                    for i, level in enumerate(self.FREEBID_QTY_LEVELS):
                        if level > 0:
                            power_mw = level * remaining_for_trading
                            if power_mw <= adjusted_max_discharge + 0.01:
                                freebid_qty_mask[i] = True
                elif imbalance > self.mfrr_imbalance_threshold and mfrr_down_price > 1.0:
                    for i, level in enumerate(self.FREEBID_QTY_LEVELS):
                        if level < 0:
                            power_mw = abs(level) * remaining_for_trading
                            if power_mw <= adjusted_max_charge + 0.01:
                                freebid_qty_mask[i] = True

            freebid_qty_mask[self.N_FREEBID_QTY_ACTIONS // 2] = True  # I always allow idle

            # Mask 6: Free Bid price tier (5 options)
            price_tier_mask = np.ones(self.N_FREEBID_PRICE_TIERS, dtype=bool)

            # I only allow price setting when Free Bid qty has a non-idle direction
            freebid_has_direction = any(
                freebid_qty_mask[i] for i in range(self.N_FREEBID_QTY_ACTIONS)
                if i != self.N_FREEBID_QTY_ACTIONS // 2
            )
            if not freebid_has_direction:
                # No Free Bid available -> force neutral price (index 2 = 1.0x)
                price_tier_mask = np.zeros(self.N_FREEBID_PRICE_TIERS, dtype=bool)
                price_tier_mask[2] = True

            full_mask = np.concatenate([
                afrr_mask, price_mask, intraday_mask, mfrr_mask,
                freebid_qty_mask, price_tier_mask
            ])
        else:
            full_mask = np.concatenate([afrr_mask, price_mask, intraday_mask, mfrr_mask])

        return full_mask

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """I execute one step in the unified environment."""
        self._check_day_reset()

        # =====================================================================
        # STAGE 1: UNPACK ACTION
        # =====================================================================
        afrr_action = action[0]
        price_tier_action = action[1]
        xbid_action = action[2]                    # Renamed from intraday_action
        mfrr_qty_action = action[3]                # mFRR qty (TSO-activated)
        freebid_qty_action = action[4] if self.enable_full_market else 5  # FreeBid qty (idle=5)
        freebid_price_action = action[5] if self.enable_full_market else 2  # FreeBid price (neutral=2)

        # I get current market data
        row = self.df.iloc[self.current_step]
        # I use endogenous DAM commitment when available (Phase 4)
        dam_commitment = np.clip(self._get_dam_commitment(self.current_step),
                                  -self.max_power_mw, self.max_power_mw)

        # I calculate remaining capacity after DAM
        remaining_capacity = self.max_power_mw - abs(dam_commitment)

        # =====================================================================
        # STAGE 2: aFRR COMMITMENT & SELECTION
        # I only allow re-commitment at 4-hour block boundaries per real
        # Greek aFRR procurement rules. During a block, the committed MW
        # stays locked.
        # =====================================================================
        if self._afrr_steps_remaining <= 0:
            # I allow the agent to set a new commitment at block boundary
            new_afrr_level = self.AFRR_LEVELS[afrr_action]
            self.afrr_commitment_level = afrr_action
            self.afrr_commitment_mw = new_afrr_level * remaining_capacity
            self._afrr_steps_remaining = self._afrr_block_steps

            # I determine if we're selected for this block
            price_tier = self.AFRR_PRICE_TIERS[price_tier_action]
            adjusted_selection_prob = self.selection_probability * (1.5 - price_tier * 0.5)
            adjusted_selection_prob = np.clip(adjusted_selection_prob, 0.1, 0.95)

            self.is_selected_for_afrr = (
                self.afrr_commitment_mw > 0.1 and
                np.random.random() < adjusted_selection_prob
            )
        # else: I keep the existing commitment (locked within 4h block)
        self._afrr_steps_remaining -= 1

        # aFRR capacity revenue
        afrr_capacity_revenue = 0.0
        if self.is_selected_for_afrr:
            # I use marginal pricing: ALL selected providers receive the market
            # clearing price, regardless of their individual bid. The price_tier
            # only affects selection probability (line 444), NOT the payment.
            cap_price = row.get('afrr_cap_up_price', 20.0)
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

        # I compute the max power actually deliverable for aFRR given post-DAM SoC.
        # I pass this to the reward calculator so it can distinguish between
        # agent fault (chose wrong action) and SoC starvation (DAM consumed energy).
        if afrr_direction == 'up':
            afrr_max_deliverable_mw = max_discharge
        elif afrr_direction == 'down':
            afrr_max_deliverable_mw = max_charge
        else:
            afrr_max_deliverable_mw = None

        if afrr_activated:
            self.steps_since_afrr_activation = 0

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
            self.steps_since_afrr_activation += 1
            remaining_after_afrr = remaining_capacity

        # =====================================================================
        # STAGE 6: EXECUTE IDA POSITIONS (locked, mandatory like DAM)
        # I time-gate IDA positions per real market rules:
        #   IDA1: delivers all ISPs (00:00-24:00), results from D-1 15:00
        #   IDA2: delivers all ISPs (00:00-24:00), results from D-1 22:00
        #   IDA3: delivers only 14:00-24:00, results from D+0 10:00
        # =====================================================================
        ida_energy_mw = 0.0
        if self.enable_full_market:
            ts = self.df.index[self.current_step]
            hour = ts.hour

            # I compute time-gated IDA position
            ida1_pos = row.get('ida1_position', 0.0)
            ida2_pos = row.get('ida2_position', 0.0)
            ida3_pos = row.get('ida3_position', 0.0)

            # IDA3 only delivers for ISPs >= 14:00
            if hour < 14:
                ida3_pos = 0.0

            net_ida = ida1_pos + ida2_pos + ida3_pos
            net_ida = np.clip(net_ida, -remaining_after_afrr, remaining_after_afrr)

            if abs(net_ida) > 0.1:
                # I recalculate limits for IDA execution
                available_discharge_mwh_ida = (self.soc - self.min_soc) * self.capacity_mwh
                available_charge_mwh_ida = (self.max_soc - self.soc) * self.capacity_mwh

                if net_ida > 0:  # Discharge
                    max_ida = min(net_ida,
                                  available_discharge_mwh_ida * self.eff_sqrt / self.time_step_hours)
                    if max_ida > 0.1:
                        energy_mwh = max_ida * self.time_step_hours
                        soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                        self.soc = max(self.min_soc, self.soc - soc_delta)
                        ida_energy_mw = max_ida
                        cycle_fraction = energy_mwh / self.capacity_mwh
                        self.total_cycles += cycle_fraction
                        self.daily_cycles += cycle_fraction
                else:  # Charge
                    max_ida = min(abs(net_ida),
                                  available_charge_mwh_ida / self.eff_sqrt / self.time_step_hours)
                    if max_ida > 0.1:
                        energy_mwh = max_ida * self.time_step_hours
                        soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                        self.soc = min(self.max_soc, self.soc + soc_delta)
                        ida_energy_mw = -max_ida
                        cycle_fraction = energy_mwh / self.capacity_mwh
                        self.total_cycles += cycle_fraction
                        self.daily_cycles += cycle_fraction

                actual_energy_mw += ida_energy_mw
                remaining_after_afrr -= abs(ida_energy_mw)

            # I compute IDA revenue from individual auction clearing prices
            ida_revenue = 0.0
            if abs(ida1_pos) > 0.01:
                p1 = row.get('ida1_clearing_price', row.get('price', 100.0))
                ida_revenue += ida1_pos * p1 * self.time_step_hours
            if abs(ida2_pos) > 0.01:
                p2 = row.get('ida2_clearing_price', row.get('price', 100.0))
                ida_revenue += ida2_pos * p2 * self.time_step_hours
            if abs(ida3_pos) > 0.01:
                p3 = row.get('ida3_clearing_price', row.get('price', 100.0))
                ida_revenue += ida3_pos * p3 * self.time_step_hours

            # I scale proportionally if execution was clipped by SoC/capacity limits
            if abs(net_ida) > 0.1 and abs(ida_energy_mw) > 0.01:
                execution_ratio = abs(ida_energy_mw) / abs(net_ida)
                ida_revenue *= execution_ratio

            self.ida_profit += ida_revenue

        remaining_after_ida = remaining_after_afrr

        # =====================================================================
        # STAGE 7: EXECUTE mFRR (TSO-activated, pay-as-cleared, mandatory)
        # I promote mFRR before IntraDay because it's TSO-mandated and should
        # get priority access to capacity. Cascade: DAM → aFRR → IDA → mFRR → IntraDay → FreeBid
        # =====================================================================
        mfrr_energy_mw = 0.0
        mfrr_revenue = 0.0

        if not afrr_activated:
            mfrr_level = self.MFRR_LEVELS[mfrr_qty_action]
            requested_mfrr = mfrr_level * remaining_after_ida

            # I determine mFRR activation from real market data (ADMIE volumes).
            # Real data shows 84% UP and 96% DOWN system-wide activation rates,
            # but individual BSP selection is much rarer. I apply a probabilistic
            # gate (mfrr_activation_rate=35%) on top of the system-wide check.
            # When activation columns are missing, I fall back to the legacy
            # imbalance-based heuristic.
            has_activation_data_exec = 'mfrr_activated_up_mwh' in row.index if hasattr(row, 'index') else 'mfrr_activated_up_mwh' in row
            if has_activation_data_exec:
                # I check two gates: (1) TSO activated mFRR in this direction,
                # (2) our BSP was selected.
                total_up = row.get('mfrr_activated_up_mwh', 0.0)
                total_down = row.get('mfrr_activated_down_mwh', 0.0)

                if requested_mfrr > 0:
                    if total_up <= 0:
                        requested_mfrr = 0.0  # TSO didn't activate UP this period
                    elif np.random.random() > self.mfrr_activation_rate:
                        requested_mfrr = 0.0  # Our BSP not selected
                elif requested_mfrr < 0:
                    if total_down <= 0:
                        requested_mfrr = 0.0  # TSO didn't activate DOWN this period
                    elif np.random.random() > self.mfrr_activation_rate:
                        requested_mfrr = 0.0  # Our BSP not selected
            else:
                # I fall back to legacy imbalance-based constraint
                imbalance = row.get('net_imbalance_mw_lag_1h',
                                    row.get('system_deviation_mwh_lag_1h',
                                            row.get('net_imbalance_mw', 0.0)))
                if abs(imbalance) <= self.mfrr_imbalance_threshold:
                    requested_mfrr = 0.0  # Balanced -> mFRR closed
                elif imbalance < -self.mfrr_imbalance_threshold and requested_mfrr < 0:
                    requested_mfrr = 0.0  # Can't charge during deficit
                elif imbalance > self.mfrr_imbalance_threshold and requested_mfrr > 0:
                    requested_mfrr = 0.0  # Can't discharge during surplus

                # I apply probabilistic activation
                if abs(requested_mfrr) > 0.1:
                    activation_prob = self.mfrr_activation_rate
                    abs_imbalance = abs(imbalance)
                    if abs_imbalance > 500:
                        activation_prob = min(0.9, activation_prob * 2.0)
                    elif abs_imbalance > 200:
                        activation_prob = min(0.8, activation_prob * 1.5)
                    if np.random.random() > activation_prob:
                        requested_mfrr = 0.0

            if abs(requested_mfrr) > 0.1:
                # I recalculate limits after IDA
                available_discharge_mwh = (self.soc - self.min_soc) * self.capacity_mwh
                available_charge_mwh = (self.max_soc - self.soc) * self.capacity_mwh

                max_discharge = min(
                    remaining_after_ida,
                    available_discharge_mwh * self.eff_sqrt / self.time_step_hours
                )
                max_charge = min(
                    remaining_after_ida,
                    available_charge_mwh / self.eff_sqrt / self.time_step_hours
                )

                if requested_mfrr > 0:
                    actual_mfrr = min(requested_mfrr, max_discharge)
                else:
                    actual_mfrr = max(requested_mfrr, -max_charge)

                if abs(actual_mfrr) > 0.1:
                    if actual_mfrr > 0:  # Sell (discharge) — pay-as-cleared
                        energy_mwh = actual_mfrr * self.time_step_hours
                        soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = max(self.min_soc, self.soc - soc_delta)
                        actual_energy = (old_soc - self.soc) * self.capacity_mwh * self.eff_sqrt

                        # I use LAGGED mFRR price — the settlement price is ex-post,
                        # so I use the best estimate available at decision time (1h lag).
                        mfrr_price = min(
                            row.get('mfrr_price_up_lag_1h',
                                    row.get('mfrr_price_up', 120.0)),
                            self.mfrr_price_cap
                        )
                        mfrr_revenue = actual_energy * mfrr_price

                    else:  # Provide DOWN regulation (charge) — pay-as-cleared
                        energy_mwh = abs(actual_mfrr) * self.time_step_hours
                        soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = min(self.max_soc, self.soc + soc_delta)
                        actual_energy = (self.soc - old_soc) * self.capacity_mwh / self.eff_sqrt

                        mfrr_price = min(
                            row.get('mfrr_price_down_lag_1h',
                                    row.get('mfrr_price_down', 60.0)),
                            self.mfrr_price_cap
                        )
                        mfrr_revenue = actual_energy * mfrr_price

                    cycle_fraction = actual_energy / self.capacity_mwh
                    self.total_cycles += cycle_fraction
                    # I do NOT increment daily_cycles for mFRR — it's TSO-mandated,
                    # the agent can't avoid it, so it shouldn't be penalized.

                    mfrr_energy_mw = actual_mfrr
                    self.mfrr_profit += mfrr_revenue
                    actual_energy_mw += actual_mfrr

        # I calculate remaining capacity after mFRR for IntraDay
        remaining_after_mfrr = remaining_after_ida - abs(mfrr_energy_mw)

        # =====================================================================
        # STAGE 8: EXECUTE INTRADAY CORRECTION (IDA auctions + ISP continuous)
        # I route the unified IntraDay action to the appropriate market
        # based on time-of-day: IDA1/2/3 auctions or ISP cascade.
        # IntraDay now uses remaining capacity AFTER mFRR.
        # =====================================================================
        intraday_energy_mw = 0.0
        xbid_energy_mw = 0.0
        intraday_revenue = 0.0

        active_market = self._get_intraday_market(row)
        intraday_open = self._is_intraday_open(row)
        if not afrr_activated and intraday_open and active_market != 'closed':
            intraday_level = self.INTRADAY_LEVELS[xbid_action]
            requested_intraday = intraday_level * remaining_after_mfrr

            if abs(requested_intraday) > 0.1:
                # I recalculate limits after mFRR
                available_discharge_mwh = (self.soc - self.min_soc) * self.capacity_mwh
                available_charge_mwh = (self.max_soc - self.soc) * self.capacity_mwh

                max_discharge = min(
                    remaining_after_mfrr,
                    available_discharge_mwh * self.eff_sqrt / self.time_step_hours
                )
                max_charge = min(
                    remaining_after_mfrr,
                    available_charge_mwh / self.eff_sqrt / self.time_step_hours
                )

                if requested_intraday > 0:
                    actual_intraday = min(requested_intraday, max_discharge)
                else:
                    actual_intraday = max(requested_intraday, -max_charge)

                if abs(actual_intraday) > 0.1:
                    if actual_intraday > 0:  # Sell (discharge)
                        energy_mwh = actual_intraday * self.time_step_hours
                        soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = max(self.min_soc, self.soc - soc_delta)
                        actual_energy = (old_soc - self.soc) * self.capacity_mwh * self.eff_sqrt

                        # I route to the correct market for pricing
                        id_price = self._get_intraday_sell_price(active_market, row)
                        intraday_revenue = actual_energy * id_price

                    else:  # Buy (charge)
                        energy_mwh = abs(actual_intraday) * self.time_step_hours
                        soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                        old_soc = self.soc
                        self.soc = min(self.max_soc, self.soc + soc_delta)
                        actual_energy = (self.soc - old_soc) * self.capacity_mwh / self.eff_sqrt

                        # I route to the correct market for pricing
                        id_price = self._get_intraday_buy_price(active_market, row)
                        intraday_revenue = -actual_energy * id_price

                    cycle_fraction = actual_energy / self.capacity_mwh
                    self.total_cycles += cycle_fraction
                    self.daily_cycles += cycle_fraction

                    intraday_energy_mw = actual_intraday
                    xbid_energy_mw = 0.0  # I no longer distinguish XBID
                    self.intraday_profit += intraday_revenue
                    actual_energy_mw += actual_intraday

        # I calculate remaining after IntraDay for Free Bids
        remaining_after_intraday = remaining_after_mfrr - abs(intraday_energy_mw)

        # =====================================================================
        # STAGE 9: EXECUTE FREE BID (agent-submitted, pay-as-bid, merit-order)
        # I only execute Free Bids in full-market mode. The agent sets both
        # quantity and price; activation depends on merit-order simulation.
        # Free Bids use remaining capacity after IntraDay.
        # =====================================================================
        free_bid_energy_mw = 0.0
        free_bid_revenue = 0.0
        free_bid_activated = False
        agent_bid_price = 0.0

        if self.enable_full_market and not afrr_activated:
            freebid_level = self.FREEBID_QTY_LEVELS[freebid_qty_action]
            requested_freebid = freebid_level * remaining_after_intraday

            if abs(requested_freebid) > 0.1:
                # I recalculate limits after IntraDay (SoC may have changed)
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

                if requested_freebid > 0:
                    actual_freebid = min(requested_freebid, max_discharge)
                else:
                    actual_freebid = max(requested_freebid, -max_charge)

                if abs(actual_freebid) > 0.1:
                    # I set the agent's bid price using the selected price tier
                    imbalance_fb = row.get('net_imbalance_mw_lag_1h',
                                           row.get('net_imbalance_mw', 0.0))
                    price_tier = self.FREEBID_PRICE_TIERS[freebid_price_action]
                    ref_price = row.get('free_bid_reference_price',
                                        row.get('mfrr_price_up', 100.0))
                    agent_bid_price = ref_price * price_tier
                    self.free_bid_submissions += 1

                    # I simulate merit-order activation
                    free_bid_activated = self._simulate_free_bid_activation(
                        agent_bid_price, ref_price, imbalance_fb, row
                    )

                    if free_bid_activated:
                        if actual_freebid > 0:  # Sell (discharge) — pay-as-bid
                            energy_mwh = actual_freebid * self.time_step_hours
                            soc_delta = energy_mwh / self.eff_sqrt / self.capacity_mwh
                            old_soc = self.soc
                            self.soc = max(self.min_soc, self.soc - soc_delta)
                            actual_energy = (old_soc - self.soc) * self.capacity_mwh * self.eff_sqrt

                            free_bid_revenue = actual_energy * agent_bid_price

                        else:  # Buy (charge) — pay-as-bid
                            energy_mwh = abs(actual_freebid) * self.time_step_hours
                            soc_delta = energy_mwh * self.eff_sqrt / self.capacity_mwh
                            old_soc = self.soc
                            self.soc = min(self.max_soc, self.soc + soc_delta)
                            actual_energy = (self.soc - old_soc) * self.capacity_mwh / self.eff_sqrt

                            free_bid_revenue = actual_energy * agent_bid_price

                        cycle_fraction = actual_energy / self.capacity_mwh
                        self.total_cycles += cycle_fraction
                        self.daily_cycles += cycle_fraction

                        free_bid_energy_mw = actual_freebid
                        self.free_bid_profit += free_bid_revenue
                        self.free_bid_activations += 1
                        actual_energy_mw += actual_freebid

        # =====================================================================
        # STAGE 10: CALCULATE REWARD
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
            mfrr_price_down=row.get('mfrr_price_down', 60.0),
            # IDA sub-market fields (full market mode) — I pass ISP prices
            ida1_clearing_price=row.get('ida1_clearing_price', 0.0),
            ida2_clearing_price=row.get('ida2_clearing_price', 0.0),
            ida3_clearing_price=row.get('ida3_clearing_price', 0.0) if not np.isnan(row.get('ida3_clearing_price', 0.0)) else 0.0,
            net_ida_position=row.get('net_ida_position', 0.0),
            xbid_bid=self._get_intraday_sell_price(None, row),
            xbid_ask=self._get_intraday_buy_price(None, row),
            free_bid_activated=free_bid_activated,
            free_bid_agent_price=agent_bid_price if self.enable_full_market else 0.0,
        )

        shortfall_risk = self._calculate_shortfall_risk()
        price_momentum = row.get('price_momentum', 0.0)

        reward_info = self.reward_calculator.calculate(
            market=market_state,
            actual_energy_mw=actual_energy_mw,
            afrr_capacity_committed_mw=self.afrr_commitment_mw if self.is_selected_for_afrr else 0.0,
            afrr_energy_delivered_mw=afrr_energy_delivered,
            intraday_energy_mw=intraday_energy_mw if not self.enable_full_market else 0.0,
            mfrr_energy_mw=mfrr_energy_mw,
            current_soc=self.soc,
            capacity_mwh=self.capacity_mwh,
            time_step_hours=self.time_step_hours,
            is_physical_violation=False,
            shortfall_risk=shortfall_risk,
            price_momentum=price_momentum,
            is_selected_for_afrr=self.is_selected_for_afrr,
            # I pass DAM-only energy so violation check doesn't penalize
            # IntraDay/aFRR/mFRR trades in the opposite direction
            dam_executed_mw=dam_executed_mw,
            # I pass max deliverable power so reward calculator can distinguish
            # agent fault from SoC starvation on aFRR non-response
            afrr_max_deliverable_mw=afrr_max_deliverable_mw,
            # Full market mode params
            ida_energy_mw=ida_energy_mw,
            xbid_energy_mw=xbid_energy_mw,
            free_bid_energy_mw=free_bid_energy_mw,
            # I pass daily_cycles for super-linear cycle excess penalty
            daily_cycles=self.daily_cycles,
        )

        reward = reward_info['reward']
        self.total_profit += reward_info['components'].get('net_profit', 0)
        self.episode_profit += reward_info['components'].get('net_profit', 0)

        # =====================================================================
        # STAGE 11: ADVANCE STEP
        # =====================================================================
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        if self.episode_length is not None:
            steps_in_episode = self.current_step - self.episode_start_step
            if steps_in_episode >= self.episode_length:
                truncated = True

        # I track SoC extremes after all market actions have executed
        self.episode_min_soc = min(self.episode_min_soc, self.soc)
        self.episode_max_soc = max(self.episode_max_soc, self.soc)

        obs = self._build_observation()
        info = self._get_info()
        # I spread reward components first, then override with authoritative
        # step-level values (the reward calculator uses simplified pricing
        # that doesn't know about IDA/XBID routing).
        info.update(reward_info['components'])
        info.update({
            'afrr_activated': afrr_activated,
            'afrr_direction': afrr_direction,
            'is_selected': self.is_selected_for_afrr,
            'actual_energy_mw': actual_energy_mw,
            'dam_commitment_mw': dam_commitment,
            'intraday_energy_mw': intraday_energy_mw,
            'intraday_revenue': intraday_revenue,
            'mfrr_energy_mw': mfrr_energy_mw,
            'mfrr_revenue': mfrr_revenue,
            # Full market mode fields
            'ida_energy_mw': ida_energy_mw,
            'xbid_energy_mw': xbid_energy_mw,
            'free_bid_energy_mw': free_bid_energy_mw,
            'free_bid_activated': free_bid_activated,
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

        lookahead = min(4 * self._sph, self.max_steps - self.current_step)

        for i in range(1, lookahead + 1):
            future_idx = self.current_step + i
            if future_idx >= len(self.df):
                break
            # I use endogenous DAM commitment when available (Phase 4)
            dam = np.clip(self._get_dam_commitment(future_idx),
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

        I also account for intermediate charge/discharge potential (25% discount)
        so the reserve is not overly conservative when the agent has time to prepare.
        The discount is conservative because the agent often trades IntraDay
        instead of actually charging during idle hours.
        """
        min_soc_reserve = self.min_soc
        max_soc_reserve = self.max_soc

        lookahead = min(4 * self._sph, self.max_steps - self.current_step)
        if lookahead <= 0:
            return min_soc_reserve, max_soc_reserve

        # I first scan to find the last sell/buy commitment positions
        # so I only credit idle hours that occur BEFORE a commitment
        commitments = []
        for i in range(1, lookahead + 1):
            future_idx = self.current_step + i
            if future_idx >= len(self.df):
                break
            # I use endogenous DAM commitment when available (Phase 4)
            dam = np.clip(self._get_dam_commitment(future_idx),
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
                potential = self.max_power_mw * self.time_step_hours * 0.25
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

        # I add a safety buffer of one max-power step to guard against
        # cross-market SoC drift between mask time and DAM execution.
        # Only applied when there are actual commitments to protect.
        buffer = 0.0
        if sell_soc_reserve > 0 or buy_soc_reserve > 0:
            buffer = self.max_power_mw * self.time_step_hours / self.capacity_mwh

        min_soc_reserve = min(self.max_soc, self.min_soc + sell_soc_reserve + buffer)
        max_soc_reserve = max(self.min_soc, self.max_soc - buy_soc_reserve - buffer)

        # I ensure min <= max (edge case with very large commitments)
        if min_soc_reserve > max_soc_reserve:
            midpoint = (min_soc_reserve + max_soc_reserve) / 2.0
            min_soc_reserve = midpoint
            max_soc_reserve = midpoint

        return min_soc_reserve, max_soc_reserve

    def _generate_endogenous_dam_schedule(self) -> np.ndarray:
        """
        I generate a DAM commitment schedule based on price forecasts.

        This replaces the CSV-based exogenous DAM commitment with a
        forecast-powered rule-based bidder. The agent then executes
        this schedule (DAM is mandatory once committed).

        Strategy:
        - Buy when forecast < P25(day), sell when forecast > P75(day)
        - Respect SoC limits and cycle constraints
        - Minimum spread threshold (default 30 EUR) to avoid low-margin trades
        """
        # I get the forecast for the current day
        ts = self.df.index[self.current_step]
        current_day = ts.date()

        # I collect DAM prices for today (known if published) or forecast
        day_mask = self.df.index.date == current_day
        day_indices = np.where(day_mask)[0]

        if len(day_indices) < 12 * self._sph:
            # I fall back to CSV commitments for short days
            return None

        # I get price forecasts (prefer MarketForecaster, fall back to actual)
        if self.market_forecaster is not None:
            forecast = self.market_forecaster.dam_forecaster.predict(
                self.df, self.current_step, 24
            )
        else:
            # I use actual prices as proxy (simulating a "perfect" forecast
            # with noise to avoid data leakage)
            hourly_indices = day_indices[::self._sph][:24]
            forecast = np.array([self.df.iloc[i].get('price', 80.0) for i in hourly_indices])

            # I add forecast noise (MAE ~18 EUR) to avoid oracle effect
            noise = np.random.normal(0, 18.0, len(forecast))
            forecast = forecast + noise

        if len(forecast) < 12:
            return None

        # I compute thresholds
        p25 = np.percentile(forecast, 25)
        p75 = np.percentile(forecast, 75)
        daily_spread = p75 - p25

        # I skip low-spread days (not worth the cycling cost)
        if daily_spread < self.dam_bidder_min_spread:
            # I return zero commitments (no DAM trading today)
            schedule = np.zeros(len(day_indices))
            return schedule

        # I size commitments based on spread magnitude
        spread_factor = np.clip(daily_spread / 80.0, 0.4, 1.0)
        base_power = self.max_power_mw * spread_factor

        # I generate per-hour commitments then expand to sub-hourly
        hourly_schedule = np.zeros(len(forecast))
        for h in range(len(forecast)):
            if forecast[h] <= p25:
                hourly_schedule[h] = -base_power * np.random.uniform(0.6, 1.0)
            elif forecast[h] >= p75:
                hourly_schedule[h] = base_power * np.random.uniform(0.6, 1.0)

        # I ensure energy balance (charge >= discharge / efficiency)
        total_charge = abs(hourly_schedule[hourly_schedule < 0].sum())
        total_discharge = hourly_schedule[hourly_schedule > 0].sum()
        if total_charge < total_discharge * 1.1 and total_discharge > 0:
            scale = total_charge / (total_discharge * 1.1 + 1e-6)
            hourly_schedule[hourly_schedule > 0] *= scale

        # I expand hourly schedule to sub-hourly resolution (same value per quarter-hour)
        schedule = np.repeat(hourly_schedule, self._sph)[:len(day_indices)]

        # I clip to inverter limits
        schedule = np.clip(schedule, -self.max_power_mw, self.max_power_mw)

        return schedule

    def _get_dam_commitment(self, step_idx: int) -> float:
        """
        I return the DAM commitment for a given step, using endogenous
        schedule if available, otherwise falling back to CSV data.
        """
        if self.enable_endogenous_dam and self.dam_schedule is not None:
            # I compute the offset within the current day
            ts = self.df.index[step_idx]
            current_day = ts.date()
            day_mask = self.df.index.date == current_day
            day_indices = np.where(day_mask)[0]

            if len(day_indices) > 0 and step_idx in day_indices:
                local_idx = np.searchsorted(day_indices, step_idx)
                if local_idx < len(self.dam_schedule):
                    return float(self.dam_schedule[local_idx])

        # I fall back to CSV data
        return float(self.df.iloc[step_idx].get('dam_commitment', 0.0))

    def _build_observation(self) -> np.ndarray:
        """
        I build the observation vector (64-85 features depending on flags).

        Feature Groups:
        1. Battery State (4): soc, max_discharge, max_charge, cycles_remaining
        2. Market Prices (11): dam, id_bid, id_ask, id_spread, id_volume, mfrr_up, mfrr_down, mfrr_spread, id_dam_spread, mfrr_dam_premium, mfrr_id_spread
        3. Time Encoding (4): hour_sin/cos, dow_sin/cos
        4. DAM Lookahead (13): current + 12h commitments
        5. Price Lookahead (12): 12h price forecasts
        6. aFRR State (8): cap_prices, activation, commitment, selection, signal, hours_since
        7. Risk/Imbalance (4): dam_discharge_reserve, shortfall_risk, momentum, worthiness
        8. Market Phase (4): ida3_correction, system_stress, res_penetration, mfrr_direction
        9. Timing Signals (4): price_vs_typical, is_peak, is_solar, hours_to_max
        10. IntraDay Forecast + RES (9, enable_forecast): fc_1h-8h, correction, spread, solar, wind, res_dev
        10b. Market Forecasts (20, enable_market_forecast): dam_fc_4h, id_fc, imbalance, serbia, confidence, timing, isp_cascade
        11-12. Full Market (12, enable_full_market): IDA/XBID/FreeBid state
        """
        row = self.df.iloc[self.current_step]
        ts = self.df.index[self.current_step]

        features = []

        # =====================================================================
        # 1. BATTERY STATE (4 features — was 3, added cycles_remaining)
        # =====================================================================
        features.append(self.soc)

        # I use endogenous DAM commitment when available (Phase 4)
        dam_commitment = np.clip(self._get_dam_commitment(self.current_step),
                                  -self.max_power_mw, self.max_power_mw)
        # I subtract aFRR commitment so the agent sees capacity available for
        # IntraDay/mFRR, not just post-DAM capacity (which was misleading)
        remaining_capacity = max(0, self.max_power_mw - abs(dam_commitment) - self.afrr_commitment_mw)

        available_discharge = (self.soc - self.min_soc) * self.capacity_mwh
        available_charge = (self.max_soc - self.soc) * self.capacity_mwh

        max_discharge = min(remaining_capacity,
                           available_discharge * self.eff_sqrt / self.time_step_hours)
        max_charge = min(remaining_capacity,
                        available_charge / self.eff_sqrt / self.time_step_hours)

        features.append(max_discharge / self.max_power_mw)
        features.append(max_charge / self.max_power_mw)

        # I add daily cycle budget remaining so the agent can pace its trading
        cycles_remaining = max(0, self.max_daily_cycles - self.daily_cycles) / self.max_daily_cycles
        features.append(cycles_remaining)

        # =====================================================================
        # 2. MARKET PRICES (11 features — was 6)
        # =====================================================================
        dam_price = row.get('price', 100.0)
        features.append(dam_price / 100.0)  # [3] DAM price

        # I use XBID lag prices for accurate IntraDay signal (preferred over
        # ISP1 settlement prices which have balancing-market convention mismatch).
        # With battery convention fix, intraday_bid = sell/UP price, intraday_ask = buy/DOWN price.
        id_bid = row.get('xbid_bid_lag_1h', row.get('intraday_bid_lag_1h', 0.0))
        id_ask = row.get('xbid_ask_lag_1h', row.get('intraday_ask_lag_1h', 0.0))
        id_spread = abs(id_ask - id_bid) if (id_bid > 0 and id_ask > 0) else row.get('intraday_spread_lag_1h', 0.0)
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
        minute = ts.minute
        # I use fractional hour for sub-hourly discrimination (14.0, 14.25, 14.5, 14.75)
        fractional_hour = hour + minute / 60.0
        dow = ts.dayofweek

        features.append(np.sin(2 * np.pi * fractional_hour / 24))
        features.append(np.cos(2 * np.pi * fractional_hour / 24))
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # =====================================================================
        # 4. DAM LOOKAHEAD (13 features: current + 12h)
        # =====================================================================
        features.append(dam_commitment / self.max_power_mw)  # Current

        # I only show same-day commitments (HEnEx compliant)
        # I sample every _sph steps so that 12 values cover 12 hours (DAM products are hourly)
        current_day = ts.date()
        for i in range(1, 13):
            target = min(self.current_step + i * self._sph, len(self.df) - 1)
            target_ts = self.df.index[target]
            if target_ts.date() == current_day:
                # I use _get_dam_commitment for endogenous DAM support
                future_dam = np.clip(self._get_dam_commitment(target),
                                     -self.max_power_mw, self.max_power_mw)
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

        # I sample every _sph steps so that 12 values cover 12 hours (DAM prices are hourly)
        for i in range(1, 13):
            target = min(self.current_step + i * self._sph, len(self.df) - 1)
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
        features.append(1.0 if self.steps_since_afrr_activation < 4 * self._sph else 0.0)

        # Steps since last activation (normalized to 24h equivalent)
        features.append(min(self.steps_since_afrr_activation, 24 * self._sph) / (24 * self._sph))

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
        # 8. MARKET PHASE & SYSTEM STATE (4 features)
        # =====================================================================
        # [55] IDA phase indicator (full market) or IDA3 correction (legacy)
        if self.enable_full_market:
            features.append(self._get_ida_phase(row))
        else:
            ida3_applies = 1.0 if hour >= 12 else 0.0
            features.append(ida3_applies)

        # [56] System stress magnitude — |imbalance| normalized (0=balanced, 1=severe)
        # I use this to help predict aFRR activation and mFRR availability
        imbalance_mag = row.get('net_imbalance_mw', 0.0)
        system_stress = min(abs(imbalance_mag) / 300.0, 1.0)
        features.append(system_stress)

        # [57] RES penetration — renewable share of load
        # High RES → more price volatility, more downward balancing likely
        res_total = row.get('res_total_mw', 0.0)
        load_total = row.get('load_mw', 1.0)
        res_penetration = np.clip(res_total / max(load_total, 1.0), 0.0, 1.5) / 1.5
        features.append(res_penetration)

        # mFRR direction signal — I use activation volumes for a richer signal
        # than the old binary imbalance threshold.
        # +1.0 = UP only, -1.0 = DOWN only, 0.0 = both active or neither
        if 'mfrr_activated_up_mwh' in row.index if hasattr(row, 'index') else 'mfrr_activated_up_mwh' in row:
            mfrr_up_active = 1.0 if row.get('mfrr_activated_up_mwh', 0) > 0 else 0.0
            mfrr_down_active = 1.0 if row.get('mfrr_activated_down_mwh', 0) > 0 else 0.0
            mfrr_direction = mfrr_up_active - mfrr_down_active
        else:
            # I fall back to legacy imbalance-based signal
            imbalance = row.get('net_imbalance_mw', 0.0)
            if imbalance < -self.mfrr_imbalance_threshold:
                mfrr_direction = 1.0   # UP available (discharge/sell)
            elif imbalance > self.mfrr_imbalance_threshold:
                mfrr_direction = -1.0  # DOWN available (charge/buy)
            else:
                mfrr_direction = 0.0   # Closed
        features.append(mfrr_direction)

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
        # 10. INTRADAY FORECAST & RES FUNDAMENTALS (9 features, enable_forecast)
        # =====================================================================
        if self.enable_forecast:
            # I get 4-horizon IntraDay price forecasts from LightGBM model
            if self.price_forecaster is not None:
                all_fc = self.price_forecaster.predict(
                    self.df, self.current_step, add_noise=self.forecast_noise
                )
                # I select horizons 1h, 2h, 4h, 8h from the 12-element forecast
                fc_1h = all_fc[0]
                fc_2h = all_fc[1]
                fc_4h = all_fc[3] if len(all_fc) > 3 else all_fc[-1]
                fc_8h = all_fc[7] if len(all_fc) > 7 else all_fc[-1]
            else:
                # I use persistence fallback (DAM price + optional noise)
                fc_1h = fc_2h = fc_4h = fc_8h = dam_price
                if self.forecast_noise:
                    for i, h in enumerate([1, 2, 4, 8]):
                        err = 0.08 + 0.12 * (h - 1) / 11
                        val = max(0.1, dam_price + np.random.normal(0, abs(dam_price) * err))
                        if i == 0: fc_1h = val
                        elif i == 1: fc_2h = val
                        elif i == 2: fc_4h = val
                        elif i == 3: fc_8h = val

            features.append(fc_1h / 100.0)                              # id_forecast_1h
            features.append(fc_2h / 100.0)                              # id_forecast_2h
            features.append(fc_4h / 100.0)                              # id_forecast_4h
            features.append(fc_8h / 100.0)                              # id_forecast_8h
            features.append((fc_1h - dam_price) / 100.0)                # forecast_vs_dam (correction signal)
            fc_list = [fc_1h, fc_2h, fc_4h, fc_8h]
            features.append((max(fc_list) - min(fc_list)) / 100.0)      # forecast_spread (volatility)
            features.append(row.get('solar', 0.0) / 3000.0)             # solar_norm
            features.append(row.get('wind_onshore', 0.0) / 3000.0)      # wind_norm
            res = row.get('res_total_mw', row.get('solar', 0) + row.get('wind_onshore', 0))
            res_mean = self._res_mean_24h[self.current_step] if self._res_mean_24h is not None else res
            features.append((res - res_mean) / 1000.0)                  # res_deviation

        # =====================================================================
        # 10b. MARKET FORECASTS (15 features, enable_market_forecast)
        # =====================================================================
        if self.enable_market_forecast:
            # I prefer pre-computed columns (faster) over live forecaster calls
            if 'dam_forecast_next_4h_mean' in self.df.columns:
                # I read pre-computed forecast columns from the DataFrame
                features.append(row.get('dam_forecast_next_4h_mean', dam_price) / 100.0)
                features.append(row.get('dam_forecast_next_4h_max', dam_price) / 100.0)
                features.append(row.get('dam_forecast_next_4h_min', dam_price) / 100.0)
                fc_spread = row.get('dam_forecast_spread', 0.0)
                features.append(fc_spread / 100.0)
                features.append(row.get('dam_forecast_vs_current', 0.0) / 100.0)
                features.append(row.get('id_forecast_1h', dam_price) / 100.0)
                features.append(row.get('id_forecast_4h', dam_price) / 100.0)
                id_fc_1h = row.get('id_forecast_1h', dam_price)
                dam_fc_mean = row.get('dam_forecast_next_4h_mean', dam_price)
                features.append((id_fc_1h - dam_fc_mean) / 100.0)
                features.append(row.get('imbalance_forecast_dir', 0.0))
                features.append(row.get('imbalance_forecast_mag', 0.0) / 500.0)
                features.append(row.get('mfrr_opportunity_score', 0.0))
                serbia = row.get('serbia_dam_price', np.nan)
                if not np.isnan(serbia):
                    features.append((serbia - dam_price) / 100.0)
                else:
                    features.append(0.0)
                features.append(row.get('forecast_confidence', 0.5))
                # I compute dynamic hours to peak/trough respecting DAM publication window.
                # DAM prices published at 14:00 for next day — same logic as price lookahead (Group 5).
                # Beyond the publication horizon I use persistence (current price) as fallback.
                from datetime import timedelta
                current_hour = ts.hour
                current_day = ts.date()
                if current_hour >= 14:
                    last_known_day = (ts + timedelta(days=1)).date()
                else:
                    last_known_day = current_day

                known_prices = []
                for i in range(96):  # 24h window
                    target = min(self.current_step + i, len(self.df) - 1)
                    target_day = self.df.index[target].date()
                    if target_day <= last_known_day:
                        known_prices.append(self.df.iloc[target].get('price', dam_price))
                    else:
                        known_prices.append(dam_price)  # persistence fallback

                if len(known_prices) > 4:
                    hours_to_peak = np.argmax(known_prices) * self.time_step_hours
                    hours_to_trough = np.argmin(known_prices) * self.time_step_hours
                else:
                    hours_to_peak, hours_to_trough = 12.0, 6.0
                features.append(hours_to_peak / 24.0)
                features.append(hours_to_trough / 24.0)

                # I add ISP features (5 features) — ISP1 only to avoid data leakage
                # ISP1 up/down prices (known ~18h before delivery)
                isp1_up = row.get('isp1_price_up', dam_price)
                isp1_down = row.get('isp1_price_down', dam_price)
                features.append(isp1_up / 100.0)
                features.append(isp1_down / 100.0)
                # ISP1 vs DAM spread (no future info)
                features.append((isp1_up - dam_price) / 100.0)
                # ISP1 bid-ask spread (volatility signal without ISP3)
                features.append((isp1_up - isp1_down) / 100.0)
                # Historical cascade spread (lagged — yesterday's ISP3-ISP1)
                features.append(row.get('isp_cascade_spread_up_lag_24h', 0.0) / 100.0)

            elif self.market_forecaster is not None:
                # I call the live forecaster (slower but works without pre-computed columns)
                try:
                    fc = self.market_forecaster.predict_all(self.df, self.current_step)
                    dam_4h = fc['dam_forecast_24h'][:4]
                    features.append(np.mean(dam_4h) / 100.0)
                    features.append(np.max(dam_4h) / 100.0)
                    features.append(np.min(dam_4h) / 100.0)
                    features.append((np.max(dam_4h) - np.min(dam_4h)) / 100.0)
                    features.append((np.mean(dam_4h) - dam_price) / 100.0)
                    id_fc = fc['id_forecast']
                    features.append(id_fc.get(1, dam_price) / 100.0)
                    features.append(id_fc.get(4, dam_price) / 100.0)
                    features.append((id_fc.get(1, dam_price) - np.mean(dam_4h)) / 100.0)
                    imb = fc['imbalance']
                    features.append(imb['direction'])
                    features.append(imb['magnitude'] / 500.0)
                    features.append(imb['opportunity_score'])
                    serbia = row.get('serbia_dam_price', np.nan)
                    features.append((serbia - dam_price) / 100.0 if not np.isnan(serbia) else 0.0)
                    avg_std = np.mean(fc['dam_forecast_std'][:4])
                    features.append(1.0 / (1.0 + avg_std / 20.0))
                    dam_24h = fc['dam_forecast_24h']
                    features.append(float(np.argmax(dam_24h) + 1) / 24.0)
                    features.append(float(np.argmin(dam_24h) + 1) / 24.0)

                    # I add ISP features — ISP1 only to avoid data leakage
                    isp1_up = row.get('isp1_price_up', dam_price)
                    isp1_down = row.get('isp1_price_down', dam_price)
                    features.append(isp1_up / 100.0)
                    features.append(isp1_down / 100.0)
                    features.append((isp1_up - dam_price) / 100.0)
                    features.append((isp1_up - isp1_down) / 100.0)
                    features.append(row.get('isp_cascade_spread_up_lag_24h', 0.0) / 100.0)
                except Exception:
                    # I fall back to zeros on any forecaster error (15 base + 5 ISP = 20)
                    features.extend([dam_price / 100.0] * 2 + [dam_price / 100.0] +
                                    [0.0] * 2 + [dam_price / 100.0] * 2 + [0.0] * 8 +
                                    [dam_price / 100.0] * 2 + [0.0] * 3)
            else:
                # I use zeros as placeholder when no forecaster and no pre-computed columns (20 features)
                features.extend([dam_price / 100.0] * 3 + [0.0] * 2 +
                                [dam_price / 100.0] * 2 + [0.0] * 8 +
                                [dam_price / 100.0] * 2 + [0.0] * 3)

        # =====================================================================
        # 11. FULL MARKET FEATURES (12 features, enable_full_market)
        # =====================================================================
        if self.enable_full_market:
            imbalance_fm = row.get('net_imbalance_mw', 0.0)

            # Group 10: IDA State (63-69)
            features.append(row.get('ida1_position', 0.0) / self.max_power_mw)    # [63]
            features.append(row.get('ida2_position', 0.0) / self.max_power_mw)    # [64]
            features.append(row.get('ida3_position', 0.0) / self.max_power_mw)    # [65]
            # [66]: ISP1 sell price — no ISP3 leakage (uses fixed _get_intraday_sell_price)
            features.append(self._get_intraday_sell_price(None, row) / 100.0)     # [66]
            # [67]: ISP1 buy price — no ISP3 leakage (uses fixed _get_intraday_buy_price)
            features.append(self._get_intraday_buy_price(None, row) / 100.0)      # [67]
            features.append(row.get('net_ida_position', 0.0) / self.max_power_mw) # [68]
            features.append(1.0 if self._is_intraday_open(row) else 0.0)          # [69]

            # Group 11: Free Bid State (70-72)
            features.append(1.0 if abs(imbalance_fm) > self.mfrr_imbalance_threshold else 0.0)  # [70]
            features.append(row.get('free_bid_activation_base', 0.3))              # [71]
            features.append(row.get('free_bid_reference_price', 100.0) / 100.0)   # [72]

            # Group 12: ISP Cascade Quality (73-74)
            # [73]: Lagged cascade spread (yesterday's ISP3-ISP1) — no future info
            features.append(row.get('isp_cascade_spread_up_lag_24h', 0.0) / 100.0)  # [73]
            features.append(self._get_time_to_delivery(row) / 24.0)               # [74]

        # =====================================================================
        # VALIDATION
        # =====================================================================
        obs = np.array(features, dtype=np.float32)

        expected_features = 64  # Base features (Groups 1-9) including cycles_remaining
        if self.enable_forecast:
            expected_features += 9
        if self.enable_market_forecast:
            expected_features += 20
        if self.enable_full_market:
            expected_features += 12
        assert len(obs) == expected_features, f"Expected {expected_features} features, got {len(obs)}"

        # I replace any NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs

    def _get_intraday_market(self, row):
        """I determine which ISP session price to use. No XBID.

        I route the unified IntraDay action [2] to the appropriate market:
        - IDA1 auction at hour 15:00 (gate closure ~15:00, results ~15:30)
        - IDA2 auction at hour 22:00 (gate closure ~22:00)
        - IDA3 auction at hour 10:00 for ISPs >= 14:00
        - ISP1 at all other hours 00:00-22:59 (ISP1 is known ~18h before delivery;
          ISP2/ISP3 are future information at decision time)
        - Closed at 23:00-23:59 (no tradeable ISPs remaining)
        """
        ts = self.df.index[self.current_step]
        hour = ts.hour
        minute = ts.minute

        # I check IDA auction windows (15-min step containing gate closure)
        if hour == 15 and minute < 15:
            return 'ida1'
        elif hour == 22 and minute < 15:
            return 'ida2'
        elif hour == 10 and minute < 15:
            return 'ida3'
        elif hour < 23:
            # I use ISP1 only — ISP2/ISP3 are future information at decision time
            return 'isp1'
        else:
            return 'closed'

    def _is_intraday_open(self, row):
        """I check if any IntraDay market (IDA or XBID) is open for trading.

        I open IntraDay for all hours 00:00-22:59 (covers IDA1/2/3 + XBID).
        Only blocked during aFRR activation and at 23:00-23:59.
        """
        if self.steps_since_afrr_activation <= 0:
            return False

        ts = self.df.index[self.current_step]
        if ts.hour >= 23:
            return False

        return True

    def _is_xbid_gate_open(self, row):
        """I delegate to _is_intraday_open for backward compatibility."""
        return self._is_intraday_open(row)

    def _get_ida_phase(self, row):
        """
        I return IDA phase indicator based on real HEnEx gate closure schedule.
        0.0  = Before any IDA results (hour < 16, IDA1 results ~15:30)
        0.25 = After IDA1 results (16 <= hour < 23)
        0.50 = After IDA2 results (hour >= 23 on D-1 or hour < 10 on D+0)
        0.75 = After IDA3 results (hour >= 10, delivery 14:00-24:00)
        1.0  = In IDA3 delivery window (hour >= 14)
        """
        hour = self.df.index[self.current_step].hour
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

    def _get_intraday_sell_price(self, active_market, row):
        """I return the sell (discharge) price using real ISP clearing data.

        I settle at ISP1 (known ~18h before delivery).
        ISP2/ISP3 are future information at decision time — using them is data leakage.
        ISP UP prices = what the battery receives for selling regulation energy.
        For IDA auctions, I still use the IDA clearing price.
        """
        if active_market in ('ida1', 'ida2', 'ida3'):
            # IDA auctions: single clearing price (same for buy/sell)
            price_col = f'{active_market}_clearing_price'
            id_price = row.get(price_col, row.get('price', 100.0))
            if active_market == 'ida3':
                ts = self.df.index[self.current_step]
                if ts.hour < 14:
                    id_price = row.get('ida2_clearing_price', row.get('price', 100.0))
            return id_price
        # I use ISP1 only — ISP2/ISP3 are future information at decision time
        p1 = row.get('isp1_price_up', np.nan)
        if not np.isnan(p1) and p1 > 0:
            return p1
        return row.get('price', 100.0)

    def _get_intraday_buy_price(self, active_market, row):
        """I return the buy (charge) price using real ISP clearing data.

        I settle at ISP1 (known ~18h before delivery).
        ISP2/ISP3 are future information at decision time — using them is data leakage.
        ISP DOWN prices = what the battery pays for buying regulation energy.
        For IDA auctions, I still use the IDA clearing price.
        """
        if active_market in ('ida1', 'ida2', 'ida3'):
            # IDA auctions: single clearing price (same for buy/sell)
            price_col = f'{active_market}_clearing_price'
            id_price = row.get(price_col, row.get('price', 100.0))
            if active_market == 'ida3':
                ts = self.df.index[self.current_step]
                if ts.hour < 14:
                    id_price = row.get('ida2_clearing_price', row.get('price', 100.0))
            return id_price
        # I use ISP1 only — ISP2/ISP3 are future information at decision time
        p1 = row.get('isp1_price_down', np.nan)
        if not np.isnan(p1) and p1 > 0:
            return p1
        return row.get('price', 100.0)

    def _simulate_free_bid_activation(self, bid_price, ref_price, imbalance, row):
        """I simulate merit-order based Free Bid activation."""
        # I check activation using real ADMIE data when available, because
        # net_imbalance_mw is often ~0 even when TSO activates mFRR volumes
        mfrr_up = row.get('mfrr_activated_up_mwh', 0.0)
        mfrr_down = row.get('mfrr_activated_down_mwh', 0.0)
        if mfrr_up > 0 or mfrr_down > 0:
            # I use activation data — TSO did activate mFRR this period
            pass
        elif abs(imbalance) < 10.0:
            # I fall back to imbalance check only when activation data unavailable
            return False

        base_prob = row.get('free_bid_activation_base', 0.3)

        # Lower bid price -> higher activation probability
        if ref_price > 0:
            price_ratio = bid_price / ref_price
        else:
            price_ratio = 1.0

        if price_ratio <= 1.0:
            price_factor = 1.0 + (1.0 - price_ratio) * 2.5  # Aggressive = more activation
        else:
            price_factor = max(0.05, 1.0 - (price_ratio - 1.0) * 3.5)  # Conservative = less

        activation_prob = np.clip(base_prob * price_factor, 0.0, 0.95)
        return np.random.random() < activation_prob

    def _get_time_to_delivery(self, row):
        """
        I compute hours remaining until the end of the current delivery day.
        For XBID purposes, this indicates how many tradeable ISPs remain.
        At 23:00 there's 1 hour left, at 14:00 there's 10 hours left.
        """
        ts = self.df.index[self.current_step]
        hours_in_day = ts.hour + ts.minute / 60.0
        remaining = 24.0 - hours_in_day
        return max(0.25, remaining)

    def _get_info(self) -> Dict:
        """I return step information."""
        info = {
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
            'daily_cycles': self.daily_cycles,
            'episode_min_soc': self.episode_min_soc,
            'episode_max_soc': self.episode_max_soc,
        }
        if self.enable_full_market:
            info.update({
                'ida_profit': self.ida_profit,
                'xbid_profit': self.xbid_profit,
                'free_bid_profit': self.free_bid_profit,
                'free_bid_activation_rate': (
                    self.free_bid_activations / max(1, self.free_bid_submissions)
                ),
            })
        return info

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
