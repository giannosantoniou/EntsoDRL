"""
Unified Multi-Market Battery Trading Environment

I implement a single unified environment that handles ALL markets:
- DAM (Day-Ahead Market) commitment tracking
- XBID continuous energy trading (ISP1 UP/DOWN bid-ask spread) — voluntary arbitrage
- IDA1/IDA2/IDA3 auctions (rule-based, NN-forecast driven) — locked commitments
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

        # Cycling — degradation_cost already penalizes each cycle proportionally.
        # I removed the cycle_target/excess penalty — if profit/cycle is high, cycle freely.
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

        # IDA schedule configuration
        ida_min_spread: float = 5.0,  # Minimum IDA-DAM spread to act (EUR/MWh)
        ida_forecaster: object = None,  # Optional IDASubForecaster instance
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

        # I store IDA schedule configuration
        self.ida_min_spread = ida_min_spread
        self.ida_forecaster = ida_forecaster

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
            degradation_cost_per_mwh=self.reward_config.get('degradation_cost', 12.0),
            dam_violation_penalty=self.reward_config.get('dam_violation_penalty', 800.0),
            afrr_nonresponse_penalty=self.reward_config.get('afrr_nonresponse_penalty', 500.0),
            soc_penalty_coeff=self.reward_config.get('soc_penalty_coeff', 0.005),
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

        # I precompute date-based mappings to avoid O(n) scans in _get_dam_commitment.
        # Previously, each call did `self.df.index.date == current_day` which converted
        # all 57k timestamps to dates — this was 90% of the step time.
        dates = self.df.index.date
        self._step_day_offset = np.zeros(len(self.df), dtype=np.int32)
        self._step_day_id = np.zeros(len(self.df), dtype=np.int32)
        current_date = None
        day_id = -1
        day_start = 0
        day_starts = []   # I track where each day starts
        for i, d in enumerate(dates):
            if d != current_date:
                current_date = d
                day_id += 1
                day_start = i
                day_starts.append(i)
            self._step_day_offset[i] = i - day_start
            self._step_day_id[i] = day_id

        # I precompute day start/length arrays for O(1) day_indices lookup
        n_days = day_id + 1
        self._day_start = np.array(day_starts, dtype=np.int32)
        self._day_length = np.zeros(n_days, dtype=np.int32)
        for d_id in range(n_days - 1):
            self._day_length[d_id] = self._day_start[d_id + 1] - self._day_start[d_id]
        self._day_length[n_days - 1] = len(self.df) - self._day_start[n_days - 1]

        # I precompute dam_commitment as numpy view to avoid pandas iloc overhead.
        # Previously, _get_dam_commitment() called self.df.iloc[step_idx].get('dam_commitment')
        # which creates a full pandas Series per call — 80% of step time at 73k calls/500 steps.
        # Using .values (not .astype) preserves the view — runtime df mutations auto-reflect.
        self._dam_commitment_values = self.df['dam_commitment'].values

        # I calculate step limits
        self.start_step = self.warmup_steps
        self.max_steps = len(self.df) - 1 - self.lookahead_steps

    def _get_day_indices_for_step(self, step_idx: int) -> np.ndarray:
        """I return all step indices for the same day as step_idx in O(1).

        Uses precomputed _day_start and _day_length arrays instead of the
        slow `self.df.index.date == current_day` scan that converts all timestamps.
        """
        day_id = self._step_day_id[step_idx]
        start = self._day_start[day_id]
        length = self._day_length[day_id]
        return np.arange(start, start + length)

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
        n_obs = 68  # Base features (Groups 1-9) + predicted_soc_eod + dam_net_energy_ratio
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
        self.ida1_profit = 0.0
        self.ida2_profit = 0.0
        self.ida3_profit = 0.0
        self.xbid_profit = 0.0
        self.free_bid_profit = 0.0
        self.free_bid_activations = 0
        self.free_bid_submissions = 0

        # I track IDA schedules (per-QH MW arrays, locked at gate closure)
        self.ida1_schedule = None  # Per-QH MW locked at IDA1 gate (D-1 15:00)
        self.ida2_schedule = None  # Per-QH MW locked at IDA2 gate (D-1 22:00)
        self.ida3_schedule = None  # Per-QH MW locked at IDA3 gate (D 10:00)
        self.ida1_locked_today = False
        self.ida2_locked_today = False
        self.ida3_locked_today = False

        # I track IDA violations (like DAM violations)
        self.episode_ida_violation_cost = 0.0
        self.episode_ida_violation_steps = 0

        # I track cycling
        self.total_cycles = 0.0
        self.daily_cycles = 0.0
        self.current_day = None

        # I track episode
        self.episode_start_step = self.current_step
        self.episode_profit = 0.0
        self.episode_shaping_penalty = 0.0
        self.episode_net_profit = 0.0
        self.episode_dam_violation_cost = 0.0
        self.episode_afrr_nonresponse_cost = 0.0
        self.episode_dam_violation_count = 0

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
            self._dam_schedule_day_id = self._step_day_id[self.current_step]

        obs = self._build_observation()
        info = self._get_info()

        return obs, info

    def _get_current_day(self) -> int:
        """I return a unique identifier for the current day."""
        ts = self.df.index[self.current_step]
        return ts.dayofyear + ts.year * 1000

    def _check_day_reset(self):
        """I reset daily cycling counter, IDA schedules, and regenerate DAM schedule at day boundaries."""
        current_day = self._get_current_day()
        if self.current_day is None or current_day != self.current_day:
            self.current_day = current_day
            self.daily_cycles = 0.0
            # I reset IDA schedules for the new day
            self.ida1_schedule = None
            self.ida2_schedule = None
            self.ida3_schedule = None
            self.ida1_locked_today = False
            self.ida2_locked_today = False
            self.ida3_locked_today = False
            # I regenerate endogenous DAM schedule for the new day
            if self.enable_endogenous_dam:
                self.dam_schedule = self._generate_endogenous_dam_schedule()
                self._dam_schedule_day_id = self._step_day_id[self.current_step]

    def set_degradation_cost(self, cost: float):
        """I allow the degradation curriculum callback to update the cost mid-training."""
        if hasattr(self, 'reward_calculator'):
            self.reward_calculator.degradation_cost = cost

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
        # I enforce IntraDay gate: open 00:00-22:59, closed 23:00+
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
        # STAGE 0: IDA GATE TRIGGERS (rule-based, NN-forecast driven)
        # I check if we're at an IDA gate closure timestamp. If so, I generate
        # and lock the IDA schedule before processing the agent's action.
        # The agent's action dim[2] controls participation level during gate steps.
        # =====================================================================
        if self.enable_full_market:
            ts_gate = self.df.index[self.current_step]
            gate_hour = ts_gate.hour
            gate_minute = ts_gate.minute

            # I compute participation level from the IntraDay action (reused for IDA)
            # At gate steps: 11 levels → [-100%, ..., 0, 0, 0, ..., +100%]
            # I use abs() since direction is determined by the schedule generator
            ida_action_raw = action[2]
            ida_level = abs(self.INTRADAY_LEVELS[ida_action_raw])  # 0.0 to 1.0

            if gate_hour == 15 and gate_minute == 0 and not self.ida1_locked_today:
                self._trigger_ida_gate(ida_number=1, participation_level=ida_level)
            elif gate_hour == 22 and gate_minute == 0 and not self.ida2_locked_today:
                self._trigger_ida_gate(ida_number=2, participation_level=ida_level)
            elif gate_hour == 10 and gate_minute == 0 and not self.ida3_locked_today:
                self._trigger_ida_gate(ida_number=3, participation_level=ida_level)

        # =====================================================================
        # STAGE 1: UNPACK ACTION
        # =====================================================================
        afrr_action = action[0]
        price_tier_action = action[1]
        xbid_action = action[2]                    # XBID continuous / IDA participation
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

                    # I apply ADMIE "highest value rule": max(aFRR, mFRR)
                    afrr_price = max(row.get('afrr_up', 80.0), row.get('mfrr_price_up', 0.0))
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

                    # I apply ADMIE "highest value rule": max(aFRR, mFRR)
                    afrr_price = max(row.get('afrr_down', 80.0), row.get('mfrr_price_down', 0.0))
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
            # I subtract aFRR committed MW even when not activated — the capacity
            # is reserved for potential activation and must not be used for other
            # markets. This matches the action mask logic (line 557).
            remaining_after_afrr = max(0, remaining_capacity - self.afrr_commitment_mw)

        # =====================================================================
        # STAGE 6: EXECUTE IDA POSITIONS (locked, mandatory like DAM)
        # I execute locked IDA schedules generated at gate closure.
        # IDA positions are mandatory — the agent must deliver them.
        # IDA1/IDA2: deliver 00:00-24:00, IDA3: deliver 12:00-24:00
        # =====================================================================
        ida_energy_mw = 0.0
        ida_revenue = 0.0
        ida_shortfall_mw = 0.0
        if self.enable_full_market:
            # I get the locked IDA commitment for this step
            net_ida = self._get_ida_commitment(self.current_step)
            net_ida = np.clip(net_ida, -remaining_after_afrr, remaining_after_afrr)

            if abs(net_ida) > 0.1:
                # I recalculate limits for IDA execution
                available_discharge_mwh_ida = (self.soc - self.min_soc) * self.capacity_mwh
                available_charge_mwh_ida = (self.max_soc - self.soc) * self.capacity_mwh

                if net_ida > 0:  # Discharge (sell)
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
                    # I check for shortfall (committed but couldn't deliver)
                    ida_shortfall_mw = max(0, net_ida - max_ida) if max_ida > 0.1 else net_ida
                else:  # Charge (buy)
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
                    # I check for shortfall
                    ida_shortfall_mw = max(0, abs(net_ida) - max_ida) if max_ida > 0.1 else abs(net_ida)

                actual_energy_mw += ida_energy_mw
                remaining_after_afrr -= abs(ida_energy_mw)

            # I compute IDA revenue from individual schedule positions
            day_offset_ida = self._step_day_offset[self.current_step]

            ida1_pos = 0.0
            if self.ida1_schedule is not None and 0 <= day_offset_ida < len(self.ida1_schedule):
                ida1_pos = self.ida1_schedule[day_offset_ida]
            ida2_pos = 0.0
            if self.ida2_schedule is not None and 0 <= day_offset_ida < len(self.ida2_schedule):
                ida2_pos = self.ida2_schedule[day_offset_ida]
            ida3_pos = 0.0
            if (self.ida3_schedule is not None and self.df.index[self.current_step].hour >= 12
                    and 0 <= day_offset_ida < len(self.ida3_schedule)):
                ida3_pos = self.ida3_schedule[day_offset_ida]

            # I compute revenue per IDA at its clearing price
            if abs(ida1_pos) > 0.01:
                p1 = row.get('ida1_clearing_price', row.get('price', 100.0))
                r1 = ida1_pos * p1 * self.time_step_hours
                ida_revenue += r1
                self.ida1_profit += r1
            if abs(ida2_pos) > 0.01:
                p2 = row.get('ida2_clearing_price', row.get('price', 100.0))
                r2 = ida2_pos * p2 * self.time_step_hours
                ida_revenue += r2
                self.ida2_profit += r2
            if abs(ida3_pos) > 0.01:
                p3 = row.get('ida3_clearing_price', row.get('price', 100.0))
                r3 = ida3_pos * p3 * self.time_step_hours
                ida_revenue += r3
                self.ida3_profit += r3

            # I scale proportionally if execution was clipped by SoC/capacity limits
            net_ida_committed = abs(ida1_pos) + abs(ida2_pos) + abs(ida3_pos)
            if net_ida_committed > 0.1 and abs(ida_energy_mw) > 0.01:
                execution_ratio = min(1.0, abs(ida_energy_mw) / net_ida_committed)
                ida_revenue *= execution_ratio

            self.ida_profit += ida_revenue

            # I track IDA violations
            if ida_shortfall_mw > 0.1:
                self.episode_ida_violation_steps += 1
                # I compute violation cost (1.5x clearing price, lighter than DAM's 2x)
                avg_ida_price = row.get('ida1_clearing_price', row.get('price', 100.0))
                ida_violation_cost = ida_shortfall_mw * avg_ida_price * 1.5 * self.time_step_hours
                self.episode_ida_violation_cost += ida_violation_cost

        remaining_after_ida = remaining_after_afrr

        # =====================================================================
        # STAGE 7: EXECUTE mFRR (TSO-activated, pay-as-cleared, mandatory)
        # I promote mFRR before IntraDay because it's TSO-mandated and should
        # get priority access to capacity. Cascade: DAM → aFRR → IDA → mFRR → IntraDay → FreeBid
        # =====================================================================
        mfrr_energy_mw = 0.0
        mfrr_revenue = 0.0

        # I removed the `not afrr_activated` guard. In production, aFRR
        # activates in real-time (AGC signal) while mFRR positions are
        # pre-committed. Both execute simultaneously — the only constraint
        # is physical MW headroom, already handled by remaining_after_ida.
        if True:
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

                        # I use CURRENT mFRR clearing price for settlement.
                        # Same as IntraDay: agent decides without knowing the price,
                        # but settlement uses the realized clearing price (production-like).
                        mfrr_price = min(
                            row.get('mfrr_price_up', 120.0),
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
                            row.get('mfrr_price_down', 60.0),
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
        # I removed the `not afrr_activated` guard. aFRR and IntraDay
        # execute in parallel — capacity already reduced via cascade.
        if intraday_open and active_market != 'closed':
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
                    # I track XBID vs IDA profit separately for diagnostics
                    if active_market == 'isp1':
                        xbid_energy_mw = actual_intraday
                        self.xbid_profit += intraday_revenue
                    else:
                        xbid_energy_mw = 0.0
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

        # I removed the `not afrr_activated` guard. Free Bids execute
        # in parallel with aFRR — capacity cascade handles the constraint.
        if self.enable_full_market:
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
        # I use ISP1 UP/DOWN for XBID settlement (real bid-ask spread)
        isp1_up = row.get('isp1_price_up', np.nan)
        isp1_down = row.get('isp1_price_down', np.nan)
        imbalance_price = row.get('imbalance_price', row.get('price', 100.0))
        if np.isnan(imbalance_price) or imbalance_price <= 0:
            imbalance_price = row.get('price', 100.0)
        # I prefer ISP1 UP for sell, ISP1 DOWN for buy; fall back to imbalance_price
        xbid_sell = isp1_up if (not np.isnan(isp1_up) and isp1_up > 0) else imbalance_price
        xbid_buy = isp1_down if (not np.isnan(isp1_down) and isp1_down > 0) else imbalance_price
        market_state = UnifiedMarketState(
            dam_price=row.get('price', 100.0),
            dam_commitment=dam_commitment,
            intraday_bid=xbid_sell,    # ISP1 UP for sell-side
            intraday_ask=xbid_buy,     # ISP1 DOWN for buy-side
            intraday_spread=max(0.0, xbid_sell - xbid_buy),  # Real bid-ask spread
            afrr_cap_up_price=row.get('afrr_cap_up_price', 20.0),
            afrr_cap_down_price=row.get('afrr_cap_down_price', 30.0),
            afrr_energy_up_price=row.get('afrr_up', 80.0),
            afrr_energy_down_price=row.get('afrr_down', 80.0),
            afrr_activated=afrr_activated,
            afrr_activation_direction=afrr_direction,
            mfrr_price_up=row.get('mfrr_price_up', 120.0),
            mfrr_price_down=row.get('mfrr_price_down', 60.0),
            # IDA sub-market fields (full market mode) — I pass clearing prices
            ida1_clearing_price=row.get('ida1_clearing_price', 0.0),
            ida2_clearing_price=row.get('ida2_clearing_price', 0.0),
            ida3_clearing_price=row.get('ida3_clearing_price', 0.0) if not np.isnan(row.get('ida3_clearing_price', 0.0)) else 0.0,
            net_ida_position=row.get('net_ida_position', 0.0),
            xbid_bid=xbid_sell,
            xbid_ask=xbid_buy,
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
            # I pass IDA shortfall for violation penalty
            ida_shortfall_mw=ida_shortfall_mw if self.enable_full_market else 0.0,
            # I pass daily_cycles for super-linear cycle excess penalty
            daily_cycles=self.daily_cycles,
        )

        reward = reward_info['reward']
        # I track trading P&L (real economic profit) separately from shaped reward
        trading_pnl = reward_info['components'].get('trading_pnl', 0)
        self.total_profit += trading_pnl
        self.episode_profit += trading_pnl
        self.episode_shaping_penalty += reward_info['components'].get('shaping_penalty', 0)
        self.episode_net_profit += reward_info['components'].get('net_profit', 0)
        dam_viol = reward_info['components'].get('dam_violation_cost', 0)
        self.episode_dam_violation_cost += dam_viol
        if dam_viol > 0:
            self.episode_dam_violation_count += 1
        self.episode_afrr_nonresponse_cost += reward_info['components'].get('afrr_nonresponse_cost', 0)

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

        # I collect DAM prices for today using O(1) precomputed lookup
        day_indices = self._get_day_indices_for_step(self.current_step)

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

        # I simulate SoC trajectory and cap commitments that exceed physical limits.
        # This runs AFTER energy balance so both buys and sells are finalized
        # before I check against SoC headroom.
        simulated_soc = self.soc
        for h in range(len(hourly_schedule)):
            commitment = hourly_schedule[h]
            if commitment < -0.1:  # Buy/charge
                soc_delta = abs(commitment) * 1.0 * self.eff_sqrt / self.capacity_mwh
                new_soc = simulated_soc + soc_delta
                if new_soc > self.max_soc:
                    # I cap: only charge what headroom allows
                    headroom = (self.max_soc - simulated_soc) * self.capacity_mwh
                    max_power = headroom / (1.0 * self.eff_sqrt)
                    hourly_schedule[h] = -max(max_power, 0.0)
                    simulated_soc = self.max_soc
                else:
                    simulated_soc = new_soc
            elif commitment > 0.1:  # Sell/discharge
                soc_delta = commitment * 1.0 / self.eff_sqrt / self.capacity_mwh
                new_soc = simulated_soc - soc_delta
                if new_soc < self.min_soc:
                    # I cap: only discharge what stored energy allows
                    available = (simulated_soc - self.min_soc) * self.capacity_mwh
                    max_power = available * self.eff_sqrt / 1.0
                    hourly_schedule[h] = max(max_power, 0.0)
                    simulated_soc = self.min_soc
                else:
                    simulated_soc = new_soc

        # I expand hourly schedule to sub-hourly resolution (same value per quarter-hour)
        schedule = np.repeat(hourly_schedule, self._sph)[:len(day_indices)]

        # I clip to inverter limits
        schedule = np.clip(schedule, -self.max_power_mw, self.max_power_mw)

        return schedule

    def _generate_ida_schedule(self, ida_number: int, participation_level: float = 1.0) -> np.ndarray:
        """
        I generate an IDA correction schedule based on NN price forecasts.

        At gate closure, I compare forecasted IDA clearing prices vs existing
        DAM+IDA commitments. I bid where the spread is favorable or where
        correction is needed for SoC feasibility.

        Args:
            ida_number: 1, 2, or 3
            participation_level: 0.0-1.0 scaling factor (from agent action)

        Returns:
            Per-QH MW schedule for the delivery window (24*sph or 12*sph values).
            Positive = sell/discharge, Negative = buy/charge.
        """
        ts = self.df.index[self.current_step]
        current_day = ts.date()

        # I determine delivery window for this IDA
        if ida_number == 3:
            delivery_hours = list(range(12, 24))
        else:
            delivery_hours = list(range(0, 24))
        n_hours = len(delivery_hours)

        # I get day indices using O(1) precomputed lookup
        day_indices = self._get_day_indices_for_step(self.current_step)

        if len(day_indices) < n_hours * self._sph:
            return np.zeros(n_hours * self._sph)

        # I get NN forecast for IDA clearing prices
        if self.ida_forecaster is not None and hasattr(self.ida_forecaster, 'predict'):
            forecast_prices = self.ida_forecaster.predict(
                self.df, self.current_step, ida_number
            )
        elif self.market_forecaster is not None and hasattr(self.market_forecaster, 'ida_forecaster'):
            forecast_prices = self.market_forecaster.ida_forecaster.predict(
                self.df, self.current_step, ida_number
            )
        else:
            # I use DAM prices with noise as fallback
            forecast_prices = np.zeros(n_hours)
            for h_idx, hour in enumerate(delivery_hours):
                qh_idx = day_indices[0] + hour * self._sph
                if qh_idx < len(self.df):
                    forecast_prices[h_idx] = self.df.iloc[qh_idx].get('price', 80.0)
                else:
                    forecast_prices[h_idx] = 80.0
            # I add noise to avoid oracle effect
            noise_std = {1: 4.0, 2: 8.0, 3: 12.0}.get(ida_number, 8.0)
            forecast_prices += np.random.normal(0, noise_std, n_hours)
            forecast_prices = np.maximum(forecast_prices, 0.0)

        # I get existing committed schedule (DAM + prior IDAs)
        existing_schedule = self._get_total_committed_schedule(delivery_hours, day_indices)

        # I simulate SoC trajectory under existing commitments
        soc_trajectory = self._simulate_soc_trajectory_for_ida(existing_schedule, delivery_hours, day_indices)

        # I compute remaining capacity for each QH
        ida_schedule = np.zeros(n_hours * self._sph)

        for h_idx, hour in enumerate(delivery_hours):
            # I get DAM price for this hour (known at gate closure)
            qh_idx = day_indices[0] + hour * self._sph
            if qh_idx < len(self.df):
                dam_price = self.df.iloc[qh_idx].get('price', 80.0)
            else:
                dam_price = 80.0

            ida_price = forecast_prices[h_idx]
            committed_mw = existing_schedule[h_idx * self._sph] if h_idx * self._sph < len(existing_schedule) else 0.0
            predicted_soc = soc_trajectory[h_idx] if h_idx < len(soc_trajectory) else 0.5

            # I compute remaining capacity after existing commitments
            remaining = self.max_power_mw - abs(committed_mw) - self.afrr_commitment_mw
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
            elif ida_price > dam_price + self.ida_min_spread:
                # I sell on IDA (higher price than DAM commitment)
                power = remaining * min(1.0, (ida_price - dam_price) / 50.0)
                correction_mw = power * np.random.uniform(0.5, 1.0)

            elif ida_price < dam_price - self.ida_min_spread:
                # I buy on IDA (lower price than DAM commitment)
                power = remaining * min(1.0, (dam_price - ida_price) / 50.0)
                correction_mw = -power * np.random.uniform(0.5, 1.0)

            # I apply to all QHs within this hour
            for qh in range(self._sph):
                slot = h_idx * self._sph + qh
                if slot < len(ida_schedule):
                    ida_schedule[slot] = correction_mw

        # I apply SoC trajectory simulation to cap infeasible positions
        ida_schedule = self._soc_cap_ida_schedule(ida_schedule, existing_schedule,
                                                   delivery_hours, day_indices)

        # I scale by agent's participation level
        ida_schedule *= participation_level

        # I clip to inverter limits
        ida_schedule = np.clip(ida_schedule, -self.max_power_mw, self.max_power_mw)

        return ida_schedule

    def _get_total_committed_schedule(self, delivery_hours: list,
                                       day_indices: np.ndarray) -> np.ndarray:
        """
        I return the total committed MW schedule (DAM + prior IDAs) for the delivery window.

        Returns per-QH MW array (positive = sell, negative = buy).
        """
        n_qh = len(delivery_hours) * self._sph
        schedule = np.zeros(n_qh)

        for h_idx, hour in enumerate(delivery_hours):
            for qh in range(self._sph):
                slot = h_idx * self._sph + qh
                qh_idx = day_indices[0] + hour * self._sph + qh
                if qh_idx < len(self.df):
                    # I get DAM commitment
                    dam = self._get_dam_commitment(qh_idx)
                    schedule[slot] = dam

                    # I add prior IDA commitments
                    if self.ida1_schedule is not None and qh_idx - day_indices[0] < len(self.ida1_schedule):
                        schedule[slot] += self.ida1_schedule[qh_idx - day_indices[0]]
                    if self.ida2_schedule is not None and qh_idx - day_indices[0] < len(self.ida2_schedule):
                        schedule[slot] += self.ida2_schedule[qh_idx - day_indices[0]]
                    # I don't add IDA3 here because IDA3 is the current one being generated
                    # (or not yet generated)

        return schedule

    def _simulate_soc_trajectory_for_ida(self, committed_schedule: np.ndarray,
                                          delivery_hours: list,
                                          day_indices: np.ndarray) -> np.ndarray:
        """
        I simulate SoC trajectory under existing commitments for the delivery window.

        Returns per-hour SoC array (one value per delivery hour).
        """
        soc = self.soc
        trajectory = np.zeros(len(delivery_hours))

        for h_idx, hour in enumerate(delivery_hours):
            # I sum MW across all QHs in this hour
            total_mw = 0.0
            for qh in range(self._sph):
                slot = h_idx * self._sph + qh
                if slot < len(committed_schedule):
                    total_mw += committed_schedule[slot]
            avg_mw = total_mw / self._sph if self._sph > 0 else 0.0

            # I update SoC
            if avg_mw > 0.1:  # Sell/discharge
                energy = avg_mw * 1.0 / self.eff_sqrt
                soc = max(self.min_soc, soc - energy / self.capacity_mwh)
            elif avg_mw < -0.1:  # Buy/charge
                energy = abs(avg_mw) * 1.0 * self.eff_sqrt
                soc = min(self.max_soc, soc + energy / self.capacity_mwh)

            trajectory[h_idx] = soc

        return trajectory

    def _soc_cap_ida_schedule(self, ida_schedule: np.ndarray,
                               existing_schedule: np.ndarray,
                               delivery_hours: list,
                               day_indices: np.ndarray) -> np.ndarray:
        """
        I cap IDA schedule positions that would violate SoC limits.

        I simulate the combined trajectory (existing + IDA) and reduce
        IDA positions where SoC would breach min/max limits.
        """
        soc = self.soc
        capped = ida_schedule.copy()

        for h_idx, hour in enumerate(delivery_hours):
            for qh in range(self._sph):
                slot = h_idx * self._sph + qh
                if slot >= len(capped):
                    break

                # I combine existing + proposed IDA
                existing_mw = existing_schedule[slot] if slot < len(existing_schedule) else 0.0
                total_mw = existing_mw + capped[slot]

                if total_mw > 0.1:  # Net sell/discharge
                    energy = total_mw * self.time_step_hours / self.eff_sqrt
                    new_soc = soc - energy / self.capacity_mwh
                    if new_soc < self.min_soc:
                        # I reduce the IDA sell position
                        available = max(0, (soc - self.min_soc) * self.capacity_mwh * self.eff_sqrt / self.time_step_hours)
                        max_ida_sell = max(0, available - existing_mw) if existing_mw > 0 else available
                        capped[slot] = min(capped[slot], max_ida_sell)
                        total_mw = existing_mw + capped[slot]
                    soc = max(self.min_soc, soc - total_mw * self.time_step_hours / self.eff_sqrt / self.capacity_mwh)

                elif total_mw < -0.1:  # Net buy/charge
                    energy = abs(total_mw) * self.time_step_hours * self.eff_sqrt
                    new_soc = soc + energy / self.capacity_mwh
                    if new_soc > self.max_soc:
                        # I reduce the IDA buy position
                        headroom = max(0, (self.max_soc - soc) * self.capacity_mwh / self.eff_sqrt / self.time_step_hours)
                        max_ida_buy = max(0, headroom - abs(existing_mw)) if existing_mw < 0 else headroom
                        capped[slot] = max(capped[slot], -max_ida_buy)
                        total_mw = existing_mw + capped[slot]
                    soc = min(self.max_soc, soc + abs(total_mw) * self.time_step_hours * self.eff_sqrt / self.capacity_mwh)

        return capped

    def _trigger_ida_gate(self, ida_number: int, participation_level: float):
        """
        I trigger IDA gate closure: generate and lock the IDA schedule.

        This is called at the gate closure timestamp for each IDA auction.
        The generated schedule becomes a mandatory commitment (like DAM).
        """
        schedule = self._generate_ida_schedule(ida_number, participation_level)

        # I use O(1) precomputed lookup for day indices
        day_indices = self._get_day_indices_for_step(self.current_step)
        n_qh_day = len(day_indices)

        if ida_number == 1:
            # IDA1 delivers all QHs (00:00-24:00)
            self.ida1_schedule = np.zeros(n_qh_day)
            # I map schedule to full-day indices
            for h_idx, hour in enumerate(range(0, 24)):
                for qh in range(self._sph):
                    slot = h_idx * self._sph + qh
                    day_slot = hour * self._sph + qh
                    if slot < len(schedule) and day_slot < n_qh_day:
                        self.ida1_schedule[day_slot] = schedule[slot]
            self.ida1_locked_today = True
        elif ida_number == 2:
            self.ida2_schedule = np.zeros(n_qh_day)
            for h_idx, hour in enumerate(range(0, 24)):
                for qh in range(self._sph):
                    slot = h_idx * self._sph + qh
                    day_slot = hour * self._sph + qh
                    if slot < len(schedule) and day_slot < n_qh_day:
                        self.ida2_schedule[day_slot] = schedule[slot]
            self.ida2_locked_today = True
        elif ida_number == 3:
            self.ida3_schedule = np.zeros(n_qh_day)
            # IDA3 delivers only 12:00-24:00
            for h_idx, hour in enumerate(range(12, 24)):
                for qh in range(self._sph):
                    slot = h_idx * self._sph + qh
                    day_slot = hour * self._sph + qh
                    if slot < len(schedule) and day_slot < n_qh_day:
                        self.ida3_schedule[day_slot] = schedule[slot]
            self.ida3_locked_today = True

    def _get_ida_commitment(self, step_idx: int) -> float:
        """
        I return the total locked IDA commitment for a given step.

        I sum IDA1 + IDA2 + IDA3 positions, respecting delivery windows.
        """
        total = 0.0
        ts = self.df.index[step_idx]
        hour = ts.hour

        # I use O(1) precomputed day offset (avoids O(n) date scan)
        day_offset = self._step_day_offset[step_idx]

        # I add IDA1 (delivers 00:00-24:00)
        if self.ida1_schedule is not None and 0 <= day_offset < len(self.ida1_schedule):
            total += self.ida1_schedule[day_offset]

        # I add IDA2 (delivers 00:00-24:00)
        if self.ida2_schedule is not None and 0 <= day_offset < len(self.ida2_schedule):
            total += self.ida2_schedule[day_offset]

        # I add IDA3 (delivers 12:00-24:00 only)
        if self.ida3_schedule is not None and hour >= 12 and 0 <= day_offset < len(self.ida3_schedule):
            total += self.ida3_schedule[day_offset]

        return total

    def _get_dam_commitment(self, step_idx: int) -> float:
        """
        I return the DAM commitment for a given step, using endogenous
        schedule if available, otherwise falling back to CSV data.

        I use precomputed _step_day_id and _step_day_offset arrays for O(1)
        lookup instead of the old O(n) `self.df.index.date == current_day` scan
        that was converting 57k timestamps to dates on every call.
        """
        if self.enable_endogenous_dam and self.dam_schedule is not None:
            # I check if step_idx is in the same day as the current schedule
            if (step_idx < len(self._step_day_id)
                    and self._step_day_id[step_idx] == self._dam_schedule_day_id):
                day_offset = self._step_day_offset[step_idx]
                if day_offset < len(self.dam_schedule):
                    return float(self.dam_schedule[day_offset])

        # I fall back to precomputed numpy array (avoids pandas iloc overhead)
        return float(self._dam_commitment_values[step_idx])

    def _predict_dam_soc(self) -> tuple:
        """
        I predict end-of-day SoC by simulating remaining DAM commitments.

        I walk through remaining DAM commitments from current_step to end of
        day, applying efficiency to compute SoC trajectory. This helps the
        agent plan IntraDay/mFRR trades around the mandatory DAM schedule.

        Returns:
            (predicted_soc_eod, net_energy_mwh): Predicted SoC at end of day
            and net energy direction of remaining schedule.
        """
        simulated_soc = self.soc
        net_energy_mwh = 0.0

        # I use precomputed day boundaries instead of per-step .date() calls
        current_day_id = self._step_day_id[self.current_step]
        day_start = self._day_start[current_day_id]
        day_end = day_start + self._day_length[current_day_id]

        # I walk from current step to end of day
        for step in range(self.current_step, min(day_end, len(self.df))):
            commitment = self._get_dam_commitment(step)
            if commitment < -0.1:  # Buy/charge
                energy = abs(commitment) * self.time_step_hours * self.eff_sqrt
                simulated_soc = min(self.max_soc, simulated_soc + energy / self.capacity_mwh)
                net_energy_mwh -= abs(commitment) * self.time_step_hours  # Negative = buying
            elif commitment > 0.1:  # Sell/discharge
                energy = commitment * self.time_step_hours / self.eff_sqrt
                simulated_soc = max(self.min_soc, simulated_soc - energy / self.capacity_mwh)
                net_energy_mwh += commitment * self.time_step_hours  # Positive = selling

        return simulated_soc, net_energy_mwh

    def _build_observation(self) -> np.ndarray:
        """
        I build the observation vector (68-89 features depending on flags).

        Feature Groups:
        1. Battery State (4): soc, max_discharge, max_charge, cycles_remaining
        2. Market Prices (11): dam, id_bid, id_ask, id_spread, id_volume, mfrr_up, mfrr_down, mfrr_spread, id_dam_spread, mfrr_dam_premium, mfrr_id_spread
        3. Time Encoding (4): hour_sin/cos, dow_sin/cos
        4. DAM Lookahead (13): current + 12h commitments
        5. Price Lookahead (12): 12h price forecasts
        6. aFRR State (8): cap_prices, activation, commitment, selection, signal, hours_since
        7. Risk/Imbalance (6): dam_discharge_reserve, shortfall_risk, momentum, worthiness, predicted_soc_eod, dam_net_energy_ratio
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
        # 2. MARKET PRICES (13 features — was 11)
        # =====================================================================
        dam_price = row.get('price', 100.0)
        features.append(dam_price / 100.0)  # [3] DAM price

        # I use lagged imbalance pricing for the observation signal.
        # XBID settlement uses ISP1 UP/DOWN (real bid-ask spread) but obs shows
        # the lagged imbalance price as a market condition indicator.
        imb_price_lag = row.get('imbalance_price_lag_1h', dam_price)
        net_imb_lag = row.get('net_imbalance_mw_lag_1h',
                              row.get('net_imbalance_mw', 0.0))
        features.append(imb_price_lag / 100.0)                    # [4] Imbalance settlement price (1h lag)
        features.append(np.sign(net_imb_lag))                      # [5] System direction (-1/0/+1)
        features.append(min(abs(net_imb_lag) / 500.0, 1.0))       # [6] System imbalance magnitude (norm)
        features.append(row.get('intraday_volume_lag_1h', 0.5))   # [7] Volume

        # mFRR prices (lagged — NO fallback to current-step)
        mfrr_up = row.get('mfrr_price_up_lag_1h', 0.0)
        mfrr_down = row.get('mfrr_price_down_lag_1h', 0.0)
        features.append(mfrr_up / 100.0)     # [8] mFRR up
        features.append(mfrr_down / 100.0)   # [9] mFRR down
        features.append((mfrr_up - mfrr_down) / 100.0)  # [10] mFRR spread

        # Cross-market signals (help agent pick the best market)
        features.append((imb_price_lag - dam_price) / 100.0)      # [11] Imbalance-DAM spread (key signal)
        features.append((mfrr_up - dam_price) / 100.0)            # [12] mFRR premium over DAM
        features.append((mfrr_up - imb_price_lag) / 100.0)        # [13] mFRR-Imbalance spread

        # I add HVR (Highest Value Rule) spreads so the agent sees when aFRR
        # commitment is extra profitable (mFRR settles higher → aFRR gets uplift)
        afrr_up_lag = row.get('afrr_up_lag_1h', row.get('afrr_up', 80.0))
        afrr_down_lag = row.get('afrr_down_lag_1h', row.get('afrr_down', 80.0))
        features.append((mfrr_up - afrr_up_lag) / 100.0)        # [14] HVR spread UP
        features.append((mfrr_down - afrr_down_lag) / 100.0)    # [15] HVR spread DOWN

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
        # 7. RISK/IMBALANCE (6 features)
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

        # Predicted SoC at end of day — I simulate remaining DAM commitments
        # so the agent knows its SoC trajectory and can plan counter-trades
        predicted_soc_eod, net_energy_mwh = self._predict_dam_soc()
        features.append(np.clip(predicted_soc_eod, 0.0, 1.0))

        # DAM net energy ratio — direction of remaining schedule
        # Positive = sell-heavy (SoC will fall), negative = buy-heavy (SoC will rise)
        dam_net_energy_ratio = np.clip(net_energy_mwh / self.capacity_mwh, -1.0, 1.0)
        features.append(dam_net_energy_ratio)

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

            # Group 13: IDA State (8 features) — replaces old Groups 10-12
            # [0] IDA phase (0.0=pre-IDA1, 0.25=post-IDA1, 0.5=post-IDA2, 0.75=post-IDA3, 1.0=delivery)
            features.append(self._get_ida_phase(row))

            # [1] Hours to next IDA gate closure (normalized 0-1)
            features.append(self._get_hours_to_next_ida_gate() / 24.0)

            # [2] IDA net commitment (total locked IDA MW / max_power)
            ida_net = self._get_ida_commitment(self.current_step)
            features.append(np.clip(ida_net / self.max_power_mw, -1.0, 1.0))

            # [3] Total net commitment (DAM + IDA net) / max_power
            total_net = dam_commitment + ida_net
            features.append(np.clip(total_net / self.max_power_mw, -1.0, 1.0))

            # [4] IDA forecast vs DAM spread (next IDA, clipped -1 to 1)
            ida_fc_spread = self._get_ida_forecast_vs_dam_spread(row)
            features.append(np.clip(ida_fc_spread / 50.0, -1.0, 1.0))

            # [5] XBID spread (ISP1 UP - ISP1 DOWN, normalized)
            isp1_up_obs = row.get('isp1_price_up', dam_price)
            isp1_down_obs = row.get('isp1_price_down', dam_price)
            xbid_spread = max(0, isp1_up_obs - isp1_down_obs)
            features.append(min(xbid_spread / 50.0, 1.0))

            # [6] SoC trajectory feasibility (1=feasible, 0=violation expected)
            soc_feasible = 1.0 if self._check_soc_trajectory_feasible() else 0.0
            features.append(soc_feasible)

            # [7] IDA correction needed (signed magnitude)
            correction_needed = self._get_ida_correction_signal()
            features.append(np.clip(correction_needed, -1.0, 1.0))

            # Free Bid State (3 features — preserved from old Group 11)
            features.append(1.0 if abs(imbalance_fm) > self.mfrr_imbalance_threshold else 0.0)
            features.append(row.get('free_bid_activation_base', 0.3))
            features.append(row.get('free_bid_reference_price', 100.0) / 100.0)

            # ISP Cascade Quality (1 feature — compressed from old Group 12)
            features.append(row.get('isp_cascade_spread_up_lag_24h', 0.0) / 100.0)

        # =====================================================================
        # VALIDATION
        # =====================================================================
        obs = np.array(features, dtype=np.float32)

        expected_features = 68  # Base features (Groups 1-9) + predicted_soc_eod + dam_net_energy_ratio
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

    def _get_hours_to_next_ida_gate(self) -> float:
        """I return hours until the next IDA gate closure."""
        ts = self.df.index[self.current_step]
        hour = ts.hour + ts.minute / 60.0

        # IDA gate closures: D-1 15:00 (IDA1), D-1 22:00 (IDA2), D+0 10:00 (IDA3)
        # I compute distance to the next gate within the current day
        if hour < 10:
            return 10.0 - hour  # Next: IDA3 at 10:00
        elif hour < 15:
            return 15.0 - hour  # Next: IDA1 at 15:00 (for next day)
        elif hour < 22:
            return 22.0 - hour  # Next: IDA2 at 22:00
        else:
            return 24.0 - hour + 10.0  # Next: IDA3 at 10:00 tomorrow

    def _get_ida_forecast_vs_dam_spread(self, row) -> float:
        """I return the mean forecast IDA-DAM spread for the next upcoming IDA."""
        ts = self.df.index[self.current_step]
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
        if self.ida_forecaster is not None and hasattr(self.ida_forecaster, 'predict_mean_spread'):
            try:
                return self.ida_forecaster.predict_mean_spread(
                    self.df, self.current_step, next_ida
                )
            except Exception:
                pass

        if (self.market_forecaster is not None
                and hasattr(self.market_forecaster, 'ida_forecaster')):
            try:
                return self.market_forecaster.ida_forecaster.predict_mean_spread(
                    self.df, self.current_step, next_ida
                )
            except Exception:
                pass

        # I fall back to pre-computed column or zero
        return row.get(f'ida{next_ida}_forecast_vs_dam', 0.0)

    def _check_soc_trajectory_feasible(self) -> bool:
        """
        I check if the current DAM+IDA schedule can execute without SoC violation.

        I simulate the combined trajectory through end of day and check
        if SoC stays within [min_soc, max_soc] bounds.
        """
        soc = self.soc
        ts = self.df.index[self.current_step]
        current_day = ts.date()

        step = self.current_step
        while step < len(self.df) and self.df.index[step].date() == current_day:
            dam = self._get_dam_commitment(step)
            ida = self._get_ida_commitment(step)
            total = dam + ida

            if total > 0.1:  # Sell/discharge
                energy = total * self.time_step_hours / self.eff_sqrt
                soc -= energy / self.capacity_mwh
                if soc < self.min_soc - 0.01:
                    return False
            elif total < -0.1:  # Buy/charge
                energy = abs(total) * self.time_step_hours * self.eff_sqrt
                soc += energy / self.capacity_mwh
                if soc > self.max_soc + 0.01:
                    return False
            step += 1

        return True

    def _get_ida_correction_signal(self) -> float:
        """
        I compute the signed magnitude of SoC correction needed.

        Positive = I need to reduce discharge (buy more / sell less)
        Negative = I need to reduce charge (sell more / buy less)
        """
        predicted_soc, _ = self._predict_dam_soc()

        # I also account for IDA commitments
        ida_total = self._get_ida_commitment(self.current_step)

        # I compute how far the predicted SoC is from the safe zone
        if predicted_soc < 0.15:
            # I am going too low — need positive correction (buy more)
            return (0.15 - predicted_soc) / 0.15  # 0 to 1.0
        elif predicted_soc > 0.85:
            # I am going too high — need negative correction (sell more)
            return -(predicted_soc - 0.85) / 0.15  # -1.0 to 0
        else:
            return 0.0  # No correction needed

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

        I close XBID at hour >= 23 (gate closure ~1h before delivery end).
        IDA auctions have their own gate closures handled by _get_intraday_market().
        aFRR and IntraDay execute in parallel — no blocking needed.
        """
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
        """I return the sell (discharge) price for IntraDay/XBID settlement.

        For IDA auctions: I use the IDA clearing price (pay-as-cleared).
        For XBID continuous: I use ISP1 UP price (sell-side of the bid-ask spread).
        ISP1 UP > ISP1 DOWN creates a real spread — profit comes from both
        timing AND the buy/sell differential, not just imbalance deviations.
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
        # I use ISP1 UP for selling (XBID continuous) — this is the sell-side
        # clearing price from the first ISP session, available ~18h before delivery
        isp1_up = row.get('isp1_price_up', np.nan)
        if not np.isnan(isp1_up) and isp1_up > 0:
            return isp1_up
        # I fall back to imbalance_price, then DAM
        imb = row.get('imbalance_price', np.nan)
        if not np.isnan(imb) and imb > 0:
            return imb
        return row.get('price', 100.0)

    def _get_intraday_buy_price(self, active_market, row):
        """I return the buy (charge) price for IntraDay/XBID settlement.

        For IDA auctions: I use the IDA clearing price (pay-as-cleared).
        For XBID continuous: I use ISP1 DOWN price (buy-side of the bid-ask spread).
        ISP1 DOWN < ISP1 UP — buying at the lower price and selling at the
        higher price creates arbitrage opportunity when spread > degradation cost.
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
        # I use ISP1 DOWN for buying (XBID continuous) — this is the buy-side
        # clearing price from the first ISP session
        isp1_down = row.get('isp1_price_down', np.nan)
        if not np.isnan(isp1_down) and isp1_down > 0:
            return isp1_down
        # I fall back to imbalance_price, then DAM
        imb = row.get('imbalance_price', np.nan)
        if not np.isnan(imb) and imb > 0:
            return imb
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
            'episode_shaping_penalty': self.episode_shaping_penalty,
            'episode_net_profit': self.episode_net_profit,
            'episode_dam_violation_cost': self.episode_dam_violation_cost,
            'episode_afrr_nonresponse_cost': self.episode_afrr_nonresponse_cost,
            'episode_dam_violation_count': self.episode_dam_violation_count,
            'episode_step': self.current_step - self.episode_start_step,
        }
        if self.enable_full_market:
            info.update({
                'ida_profit': self.ida_profit,
                'ida1_profit': self.ida1_profit,
                'ida2_profit': self.ida2_profit,
                'ida3_profit': self.ida3_profit,
                'xbid_profit': self.xbid_profit,
                'free_bid_profit': self.free_bid_profit,
                'free_bid_activation_rate': (
                    self.free_bid_activations / max(1, self.free_bid_submissions)
                ),
                'episode_ida_violation_cost': self.episode_ida_violation_cost,
                'episode_ida_violation_steps': self.episode_ida_violation_steps,
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
