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

Observation Space: 70-102 features (see _build_observation for details)
  Base: 70 | +forecast: 9 | +market_forecast: 20 | +full_market: 13
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from gym_envs.unified_reward_calculator import UnifiedRewardCalculator, UnifiedMarketState
from gym_envs.data_preparer import DataPreparer
from gym_envs.market_executors import PrecomputedArrays, BatterySnapshot, MarketResult
from gym_envs.capacity_manager import CapacityManager
from gym_envs.mask_builder import MaskBuilder
from gym_envs.observation_builder import ObservationBuilder
from gym_envs.market_executors.dam_executor import DamExecutor
from gym_envs.market_executors.afrr_executor import AfrrExecutor
from gym_envs.market_executors.ida_executor import IdaExecutor
from gym_envs.market_executors.trading_executor import TradingExecutor


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

        # I create modular components (Strangler Fig pattern)
        self._capacity_manager = CapacityManager(
            capacity_mwh=capacity_mwh,
            max_power_mw=max_power_mw,
            efficiency=efficiency,
            eff_sqrt=self.eff_sqrt,
            min_soc=min_soc,
            max_soc=max_soc,
            time_step_hours=time_step_hours,
            max_daily_cycles=max_daily_cycles,
        )
        self._mask_builder = MaskBuilder(self)
        self._obs_builder = ObservationBuilder(self)
        self._dam_executor = DamExecutor(self)
        self._afrr_executor = AfrrExecutor(self)
        self._ida_executor = IdaExecutor(self)
        self._trading_executor = TradingExecutor(self)

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
        """I delegate data preparation to the DataPreparer module.

        This preserves all the same instance attributes (_step_day_offset,
        _step_day_id, _day_start, _day_length, _dam_commitment_values, etc.)
        for backward compatibility with the rest of the env.
        """
        preparer = DataPreparer(
            sph=self._sph,
            mfrr_imbalance_threshold=self.mfrr_imbalance_threshold,
            enable_forecast=self.enable_forecast,
            warmup_steps=self.warmup_steps,
            lookahead_hours=self.lookahead_hours,
            time_step_hours=self.time_step_hours,
        )
        self.df, self._precomputed, self._res_mean_24h, self.start_step, self.max_steps = \
            preparer.prepare(self.df)

        # I expose precomputed arrays as instance attributes for backward compat
        self._step_day_offset = self._precomputed.step_day_offset
        self._step_day_id = self._precomputed.step_day_id
        self._day_start = self._precomputed.day_start
        self._day_length = self._precomputed.day_length
        self._dam_commitment_values = self._precomputed.dam_commitment_values

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
        n_obs = 70  # Base features (Groups 1-9) + situational awareness
        if self.enable_forecast:
            n_obs += 9   # IntraDay forecast + RES fundamentals
        if self.enable_market_forecast:
            n_obs += 20  # DAM+ID+Imbalance forecast + ISP cascade features (Group 10b)
        if self.enable_full_market:
            n_obs += 13  # IDA/XBID/FreeBid features + ISP1-DAM spread

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

        # I track per-market energy volumes (MWh bought/sold)
        self.dam_mwh_sold = 0.0
        self.dam_mwh_bought = 0.0
        self.afrr_mwh_sold = 0.0
        self.afrr_mwh_bought = 0.0
        self.mfrr_mwh_sold = 0.0
        self.mfrr_mwh_bought = 0.0
        self.intraday_mwh_sold = 0.0
        self.intraday_mwh_bought = 0.0
        self.ida_mwh_sold = 0.0
        self.ida_mwh_bought = 0.0
        self.xbid_mwh_sold = 0.0
        self.xbid_mwh_bought = 0.0
        self.free_bid_mwh_sold = 0.0
        self.free_bid_mwh_bought = 0.0

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

        # I sample per-episode systematic bias for DAM forecast noise.
        # Real forecasters have persistent biases during weather regime changes
        # (e.g., consistently over-forecasting during cloudy weeks). This bias
        # ensures the agent doesn't rely on a "noisy oracle" DAM schedule.
        self._dam_forecast_bias = np.random.uniform(-12, 12)

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
        """I delegate to MaskBuilder for hierarchical action masks."""
        return self._mask_builder.build()

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
            # Positive = follow generator direction, Negative = reverse direction
            ida_action_raw = action[2]
            ida_level = self.INTRADAY_LEVELS[ida_action_raw]  # -1.0 to +1.0

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

            # I track DAM energy volumes
            dam_mwh = abs(dam_executed_mw) * self.time_step_hours
            if dam_executed_mw > 0:
                self.dam_mwh_sold += dam_mwh
            elif dam_executed_mw < 0:
                self.dam_mwh_bought += dam_mwh

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
                    afrr_energy_revenue = actual_energy * afrr_price

                    cycle_fraction = actual_energy / self.capacity_mwh
                    self.total_cycles += cycle_fraction
                    self.daily_cycles += cycle_fraction

                    afrr_energy_delivered = -actual_energy / self.time_step_hours

            self.afrr_energy_profit += afrr_energy_revenue
            actual_energy_mw += afrr_energy_delivered

            # I track aFRR energy volumes (up = sold/discharge, down = bought/charge)
            afrr_mwh = abs(afrr_energy_delivered) * self.time_step_hours
            if afrr_energy_delivered > 0:
                self.afrr_mwh_sold += afrr_mwh
            elif afrr_energy_delivered < 0:
                self.afrr_mwh_bought += afrr_mwh

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

            # I track IDA energy volumes
            ida_mwh_step = abs(ida_energy_mw) * self.time_step_hours
            if ida_energy_mw > 0:
                self.ida_mwh_sold += ida_mwh_step
            elif ida_energy_mw < 0:
                self.ida_mwh_bought += ida_mwh_step

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

                    # I track mFRR energy volumes
                    mfrr_mwh = abs(actual_mfrr) * self.time_step_hours
                    if actual_mfrr > 0:
                        self.mfrr_mwh_sold += mfrr_mwh
                    else:
                        self.mfrr_mwh_bought += mfrr_mwh

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

                    # I track IntraDay and XBID energy volumes
                    id_mwh = abs(actual_intraday) * self.time_step_hours
                    if actual_intraday > 0:
                        self.intraday_mwh_sold += id_mwh
                        if active_market == 'isp1':
                            self.xbid_mwh_sold += id_mwh
                    else:
                        self.intraday_mwh_bought += id_mwh
                        if active_market == 'isp1':
                            self.xbid_mwh_bought += id_mwh

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

                        # I track Free Bid energy volumes
                        fb_mwh = abs(actual_freebid) * self.time_step_hours
                        if actual_freebid > 0:
                            self.free_bid_mwh_sold += fb_mwh
                        else:
                            self.free_bid_mwh_bought += fb_mwh

        # =====================================================================
        # STAGE 10: CALCULATE REWARD
        # =====================================================================
        # I use realistic XBID continuous market prices for settlement
        # ISP1 up/down remain as observation features (balancing signal)
        dam_price = row.get('price', 100.0)
        xbid_bid = row.get('xbid_price_bid', np.nan)
        xbid_ask = row.get('xbid_price_ask', np.nan)
        # I fall back to DAM ± small spread when XBID columns missing
        if np.isnan(xbid_bid) or xbid_bid <= 0:
            xbid_bid = dam_price - 1.5
        if np.isnan(xbid_ask) or xbid_ask <= 0:
            xbid_ask = dam_price + 1.5
        xbid_sell = xbid_bid   # sell at bid (lower)
        xbid_buy = xbid_ask    # buy at ask (higher)
        market_state = UnifiedMarketState(
            dam_price=dam_price,
            dam_commitment=dam_commitment,
            intraday_bid=xbid_sell,    # XBID bid for sell-side
            intraday_ask=xbid_buy,     # XBID ask for buy-side
            intraday_spread=max(0.0, xbid_buy - xbid_sell),  # Correct: ask - bid
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
        """I delegate to AfrrExecutor for activation simulation."""
        return self._afrr_executor.check_activation(row)

    def _calculate_shortfall_risk(self) -> float:
        """I delegate to DamExecutor for shortfall risk calculation."""
        return self._dam_executor.calculate_shortfall_risk()

    def _calculate_dam_soc_reserve(self) -> Tuple[float, float]:
        """I delegate to DamExecutor for SoC reserve calculation."""
        return self._dam_executor.calculate_soc_reserve()

    def _generate_endogenous_dam_schedule(self) -> np.ndarray:
        """I delegate to DamExecutor for endogenous DAM schedule generation."""
        return self._dam_executor.generate_endogenous_schedule()

    def _generate_ida_schedule(self, ida_number: int, participation_level: float = 1.0) -> np.ndarray:
        """I delegate to IdaExecutor for IDA schedule generation."""
        return self._ida_executor.generate_schedule(ida_number, participation_level)

    def _get_total_committed_schedule(self, delivery_hours: list,
                                       day_indices: np.ndarray) -> np.ndarray:
        """I delegate to IdaExecutor for total committed schedule."""
        return self._ida_executor._get_total_committed_schedule(delivery_hours, day_indices)

    def _simulate_soc_trajectory_for_ida(self, committed_schedule: np.ndarray,
                                          delivery_hours: list,
                                          day_indices: np.ndarray) -> np.ndarray:
        """I delegate to IdaExecutor for SoC trajectory simulation."""
        return self._ida_executor._simulate_soc_trajectory(committed_schedule, delivery_hours, day_indices)

    def _soc_cap_ida_schedule(self, ida_schedule: np.ndarray,
                               existing_schedule: np.ndarray,
                               delivery_hours: list,
                               day_indices: np.ndarray) -> np.ndarray:
        """I delegate to IdaExecutor for SoC-capped IDA schedule."""
        return self._ida_executor._soc_cap_schedule(ida_schedule, existing_schedule, delivery_hours, day_indices)

    def _trigger_ida_gate(self, ida_number: int, participation_level: float):
        """I delegate to IdaExecutor for IDA gate triggering."""
        self._ida_executor._trigger_gate(ida_number, participation_level)

    def _get_ida_commitment(self, step_idx: int) -> float:
        """I delegate to IdaExecutor for IDA commitment lookup."""
        return self._ida_executor.get_commitment(step_idx)

    def _get_dam_commitment(self, step_idx: int) -> float:
        """I delegate to DamExecutor for O(1) DAM commitment lookup."""
        return self._dam_executor.get_commitment(step_idx)

    def _predict_dam_soc(self) -> tuple:
        """I delegate to DamExecutor for end-of-day SoC prediction."""
        return self._dam_executor.predict_soc()

    def _build_observation(self) -> np.ndarray:
        """I delegate to ObservationBuilder for the observation vector."""
        return self._obs_builder.build()

    def _get_hours_to_next_ida_gate(self) -> float:
        """I delegate to IdaExecutor for hours to next gate."""
        return self._ida_executor.get_hours_to_next_gate()

    def _get_ida_forecast_vs_dam_spread(self, row) -> float:
        """I delegate to IdaExecutor for IDA-DAM forecast spread."""
        return self._ida_executor.get_forecast_vs_dam_spread(row)

    def _check_soc_trajectory_feasible(self) -> bool:
        """I delegate to IdaExecutor for SoC trajectory feasibility check."""
        return self._ida_executor.check_soc_trajectory_feasible()

    def _get_ida_correction_signal(self) -> float:
        """I delegate to IdaExecutor for SoC correction signal."""
        return self._ida_executor.get_correction_signal()

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
        """I delegate to IdaExecutor for IDA phase indicator."""
        return self._ida_executor.get_phase()

    def _get_intraday_sell_price(self, active_market, row):
        """I return the sell (discharge) price for IntraDay/XBID settlement.

        For IDA auctions: I use the IDA clearing price (pay-as-cleared).
        For XBID continuous: I use xbid_price_bid (realistic continuous market
        sell price). This is close to DAM with tight spread, NOT the ISP1
        balancing up-regulation price (which requires TSO activation).
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
        # I use XBID bid price for selling (continuous market)
        xbid_bid = row.get('xbid_price_bid', np.nan)
        if not np.isnan(xbid_bid) and xbid_bid > 0:
            return xbid_bid
        # I fall back to DAM price minus small spread
        return row.get('price', 100.0) - 1.5

    def _get_intraday_buy_price(self, active_market, row):
        """I return the buy (charge) price for IntraDay/XBID settlement.

        For IDA auctions: I use the IDA clearing price (pay-as-cleared).
        For XBID continuous: I use xbid_price_ask (realistic continuous market
        buy price). This is close to DAM with tight spread, NOT the ISP1
        balancing down-regulation price.
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
        # I use XBID ask price for buying (continuous market)
        xbid_ask = row.get('xbid_price_ask', np.nan)
        if not np.isnan(xbid_ask) and xbid_ask > 0:
            return xbid_ask
        # I fall back to DAM price plus small spread
        return row.get('price', 100.0) + 1.5

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
            # I expose per-market energy volumes (MWh) for cross-market analysis
            'dam_mwh_sold': self.dam_mwh_sold,
            'dam_mwh_bought': self.dam_mwh_bought,
            'afrr_mwh_sold': self.afrr_mwh_sold,
            'afrr_mwh_bought': self.afrr_mwh_bought,
            'mfrr_mwh_sold': self.mfrr_mwh_sold,
            'mfrr_mwh_bought': self.mfrr_mwh_bought,
            'intraday_mwh_sold': self.intraday_mwh_sold,
            'intraday_mwh_bought': self.intraday_mwh_bought,
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
                'ida_mwh_sold': self.ida_mwh_sold,
                'ida_mwh_bought': self.ida_mwh_bought,
                'xbid_mwh_sold': self.xbid_mwh_sold,
                'xbid_mwh_bought': self.xbid_mwh_bought,
                'free_bid_mwh_sold': self.free_bid_mwh_sold,
                'free_bid_mwh_bought': self.free_bid_mwh_bought,
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
