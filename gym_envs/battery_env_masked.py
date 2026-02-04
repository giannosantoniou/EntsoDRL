"""
Battery Environment with Action Masking and Market Depth.
Uses Bid/Ask prices for realistic trading simulation.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class SimpleMarketState:
    """Lightweight market state for reward calculation."""
    dam_price: float
    best_bid: float
    best_ask: float
    spread: float


class BatteryEnvMasked(gym.Env):
    """
    Battery Trading Environment with Action Masking and Market Depth.
    
    Features:
    - Action masking prevents physically impossible actions
    - Bid/Ask spread for realistic execution costs
    - Counter-Trading support (strategic deviation from DAM)
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self, 
        df: pd.DataFrame, 
        battery_params: Dict[str, float], 
        n_actions: int = 21,
        initial_soc: float = 0.5,
        base_spread: float = 4.0,      # Realistic: 3-5€
        noise_std: float = 10.0,       # Realistic IntraDay volatility
        price_cap: float = 200.0,      # Max price (0.05% exception)
        extreme_spike_prob: float = 0.0005,  # 0.05% chance of extreme prices
        # Time Step Configuration
        time_step_hours: float = 1.0,  # 1.0 for hourly, 0.25 for 15-min
        # Forecast Uncertainty Parameters (for realistic training)
        forecast_err_1h: float = 0.08,       # 8% error at 1h horizon
        forecast_err_24h: float = 0.20,      # 20% error at 24h horizon
        forecast_bias_prob: float = 0.05,    # 5% chance of systematic bias
        forecast_miss_prob: float = 0.005,   # 0.5% chance of complete miss
        use_perfect_forecast: bool = False,   # For debugging/comparison
        # Reward Configuration (v12: configurable penalties)
        reward_config: dict = None,
        # v14: Use ML forecaster for price lookahead (realistic training)
        use_ml_forecaster: bool = False,
        forecaster_path: str = "models/intraday_forecaster.pkl"
    ):
        super().__init__()

        # I store reward config for use in step() - allows tuning without code changes
        self.reward_config = reward_config or {}

        # v14: Load ML price forecaster if specified
        self.use_ml_forecaster = use_ml_forecaster
        self.price_forecaster = None
        if use_ml_forecaster:
            try:
                from models.intraday_forecaster import IntraDayForecaster
                self.price_forecaster = IntraDayForecaster.load(forecaster_path)
                print(f"  Loaded ML price forecaster from {forecaster_path}")
            except Exception as e:
                print(f"  Warning: Could not load forecaster: {e}. Using standard noise.")

        self.df = df.copy()
        self.df.columns = [c.lower() for c in self.df.columns]
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        self.n_actions = n_actions
        self.base_spread = base_spread
        self.noise_std = noise_std
        self.price_cap = price_cap
        self.extreme_spike_prob = extreme_spike_prob
        
        # Forecast Uncertainty Settings
        self.forecast_err_1h = forecast_err_1h
        self.forecast_err_24h = forecast_err_24h
        self.forecast_bias_prob = forecast_bias_prob
        self.forecast_miss_prob = forecast_miss_prob
        self.use_perfect_forecast = use_perfect_forecast
        
        self.capacity_mwh = battery_params.get('capacity_mwh', 146.0)
        self.max_power_mw = battery_params.get('max_discharge_mw', 30.0)
        self.efficiency = battery_params.get('efficiency', 0.94)
        
        # Time step configuration
        self.time_step_hours = time_step_hours
        self.steps_per_hour = int(1.0 / time_step_hours)
        
        # Pre-compute volatility for spread simulation (window = 24 hours in steps)
        volatility_window = int(24 / self.time_step_hours)  # 24 for hourly, 96 for 15-min
        prices = self.df['price'].values if 'price' in self.df.columns else np.ones(len(self.df)) * 100
        self.price_volatility = pd.Series(prices).rolling(volatility_window, min_periods=1).std().fillna(5.0).values
        
        # Action space
        self.action_space = spaces.Discrete(self.n_actions)
        self.action_levels = np.linspace(-1.0, 1.0, self.n_actions)
        
        # Observation space (38 features):
        # Battery(3) + Market(3) + Time(2) + Price Lookahead(12) + DAM Lookahead(12) + Imbalance Risk(3) + Shortfall Risk(1) + aFRR(2)
        self.lookahead_hours = 12  # Extended lookahead for better planning
        self.strategic_lookahead_hours = 24  # Strategic price horizon in hours
        self.lookahead_steps = int(self.lookahead_hours / self.time_step_hours)  # 12 for hourly
        self.strategic_lookahead = int(self.strategic_lookahead_hours / self.time_step_hours)  # 24 for hourly
        feature_count = 42  # v13: 8 base + 12 price + 12 DAM + 6 risk/aFRR + 4 timing
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        # Feature Engineer
        from agent.feature_engineer import FeatureEngineer
        self.feature_engine = FeatureEngineer(config={
            'capacity_mwh': self.capacity_mwh,
            'max_power_mw': self.max_power_mw,
            'efficiency': self.efficiency,
            'time_step_hours': self.time_step_hours
        })
        
        # Pre-compute statistics for each step
        self._precompute_24h_stats()
        
        # Pre-compute imbalance data for FeatureEngineer
        self._prepare_imbalance_data()
        
        # State
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.max_steps = len(self.df) - 1 - max(self.lookahead_steps, self.strategic_lookahead)
        
        # SoC limits
        self.min_soc = 0.05
        self.max_soc = 0.95
        
        time_res = f"{int(self.time_step_hours * 60)}min" if self.time_step_hours < 1 else "hourly"
        if self.use_ml_forecaster and self.price_forecaster is not None:
            forecast_mode = "ML FORECASTER (trained model)"
        elif self.use_perfect_forecast:
            forecast_mode = "PERFECT FORESIGHT"
        else:
            forecast_mode = f"REALISTIC (err_1h={self.forecast_err_1h:.0%}, err_24h={self.forecast_err_24h:.0%})"
        print(f"Masked Environment Initialized: {self.n_actions} actions | {time_res} | lookahead={self.lookahead_steps} steps | Forecast: {forecast_mode}")

    def _precompute_24h_stats(self):
        """Pre-compute 24h rolling statistics for strategic lookahead."""
        prices = self.df['price'].values if 'price' in self.df.columns else np.ones(len(self.df)) * 100
        n = len(prices)
        
        self.price_max_24h = np.zeros(n)
        self.price_min_24h = np.zeros(n)
        self.price_mean_24h = np.zeros(n)
        
        for i in range(n):
            end_idx = min(i + self.strategic_lookahead, n)
            future_prices = prices[i+1:end_idx] if i+1 < end_idx else [prices[i]]
            if len(future_prices) == 0:
                future_prices = [prices[i]]
            self.price_max_24h[i] = np.max(future_prices)
            self.price_min_24h[i] = np.min(future_prices)
            self.price_mean_24h[i] = np.mean(future_prices)

    def _prepare_imbalance_data(self):
        """Pre-load data arrays for FeatureEngineer."""
        n = len(self.df)
        self.imbalance_prices = self.df['price'].values if 'price' in self.df.columns else np.ones(n) * 100
        
        solar = self.df['solar'].values if 'solar' in self.df.columns else np.zeros(n)
        wind = self.df['wind_onshore'].values if 'wind_onshore' in self.df.columns else np.zeros(n)
        self.imbalance_res = solar + wind


    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_soc = self.initial_soc
        return self._get_observation(), {}

    def action_masks(self) -> np.ndarray:
        """Returns boolean mask of valid actions."""
        eff_sqrt = np.sqrt(self.efficiency)
        
        # Available energy in MWh
        available_discharge_mwh = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh)
        
        # Max power considering what can be delivered in this time step
        # Energy per step = Power × time_step_hours, so max_power = energy / time_step_hours
        max_discharge_mw = min(self.max_power_mw, (available_discharge_mwh * eff_sqrt) / self.time_step_hours)
        max_charge_mw = min(self.max_power_mw, (available_charge_mwh / eff_sqrt) / self.time_step_hours)
        
        max_discharge_action = max_discharge_mw / self.max_power_mw
        max_charge_action = max_charge_mw / self.max_power_mw
        
        mask = np.zeros(self.n_actions, dtype=bool)
        for i, action_level in enumerate(self.action_levels):
            if action_level >= 0:
                mask[i] = action_level <= max_discharge_action + 0.01
            else:
                mask[i] = abs(action_level) <= max_charge_action + 0.01
        
        # Always allow idle
        mask[self.n_actions // 2] = True
        return mask

    def _get_market_state(self) -> SimpleMarketState:
        """
        Generate market state with HYBRID volatility filtering.
        
        DAM: Filtered (20% chance of keeping spikes >200€)
        Intraday: UNRESTRICTED (raw + noise) - let the agent see real spikes!
        """
        row = self.df.iloc[self.current_step]
        raw_price = row.get('price', 100.0)
        
        # 1. INTRADAY PRICE: Unrestricted (let spikes through!)
        noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0
        intraday_price = max(0.1, raw_price + noise)
        intraday_price = min(intraday_price, 1000.0)  # Safety cap only
        
        # 2. DAM PRICE: Filtered (20% chance of keeping spikes >200€)
        dam_price = raw_price
        if dam_price > 200.0:
            # Roll the dice: Only keep the spike 20% of the time
            if np.random.random() > 0.20:  # 80% suppress
                dam_price = 200.0
        dam_price = min(dam_price, 1000.0)  # Safety cap
        
        # 3. Imbalance settlement prices - I use real Long/Short prices for HEnEx
        # Long = price received when you have positive imbalance (delivered MORE than committed)
        # Short = price paid when you have negative imbalance (delivered LESS than committed)
        row = self.df.iloc[self.current_step]

        # I check for real Long/Short imbalance settlement prices
        if 'long' in self.df.columns and 'short' in self.df.columns:
            long_price = row.get('long', dam_price)
            short_price = row.get('short', dam_price)

            # I check if Long/Short are actually differentiated (useful data)
            # If Long = Short (or placeholder 163.8), I create synthetic spread from mFRR
            needs_synthetic = (
                abs(long_price - short_price) < 0.1 or  # Long = Short (useless)
                (abs(long_price - 163.8) < 0.1 and abs(short_price - 163.8) < 0.1)  # 2021-2024 placeholder
            )

            if needs_synthetic:
                # I create realistic synthetic spread based on mFRR activation prices
                # mFRR_Up = cost to activate upward reserves (system short)
                # mFRR_Down = cost to activate downward reserves (system long)
                mfrr_up = row.get('mfrr_up', dam_price * 1.3)
                mfrr_down = row.get('mfrr_down', dam_price * 0.7)

                # Synthetic spread: 30% of the distance to mFRR prices
                # Short penalty: move toward mFRR_Up (you pay more when system needs power)
                # Long penalty: move toward mFRR_Down (you receive less when system has excess)
                spread_factor = 0.30
                short_price = dam_price + spread_factor * abs(mfrr_up - dam_price)
                long_price = dam_price - spread_factor * abs(dam_price - mfrr_down)

            # I ensure reasonable bounds
            # Short price should be >= DAM (penalty for being short)
            short_price = max(short_price, dam_price)
            # Long price should be <= DAM (less favorable for surplus)
            long_price = min(long_price, dam_price)
            # No negative prices
            long_price = max(0.1, long_price)

            best_bid = long_price   # What you receive for positive imbalance
            best_ask = short_price  # What you pay for negative imbalance
            spread = best_ask - best_bid
        else:
            # Fallback to synthetic spread based on volatility
            volatility = self.price_volatility[self.current_step]
            spread = self.base_spread * (1 + volatility / 30)
            spread = min(spread, 15.0)  # Max 15 EUR spread

            best_bid = max(0.1, intraday_price - spread / 2)
            best_ask = intraday_price + spread / 2

        return SimpleMarketState(
            dam_price=dam_price,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread
        )

    def step(self, action: int):
        # Convert action
        action_level = self.action_levels[action]
        req_phy_mw = action_level * self.max_power_mw
        
        # Physical limits - scale by time_step_hours for energy constraints
        eff_sqrt = np.sqrt(self.efficiency)
        
        # Max energy in/out this step = Power × time_step_hours
        energy_in_max_step = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh / eff_sqrt)
        energy_out_max_step = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh * eff_sqrt)
        
        # Convert energy limits to power limits for this time step
        power_in_max = energy_in_max_step / self.time_step_hours
        power_out_max = energy_out_max_step / self.time_step_hours
        
        if req_phy_mw < 0:  # Charging
            actual_phy_mw = max(req_phy_mw, -self.max_power_mw, -power_in_max)
        else:  # Discharging
            actual_phy_mw = min(req_phy_mw, self.max_power_mw, power_out_max)
        
        # Update SoC - Energy = Power × time_step_hours
        energy_mwh = abs(actual_phy_mw) * self.time_step_hours
        if actual_phy_mw < 0:  # Charging
            self.current_soc += (energy_mwh * eff_sqrt) / self.capacity_mwh
        else:  # Discharging
            self.current_soc -= (energy_mwh / eff_sqrt) / self.capacity_mwh
        # I clip to physical limits, but action masking should prevent reaching these
        self.current_soc = np.clip(self.current_soc, self.min_soc, self.max_soc)
        
        # Get market state and calculate reward
        row = self.df.iloc[self.current_step]
        dam_comm_mw = row.get('dam_commitment', 0.0)
        # I clip DAM commitment to battery's physical limits (data may have impossible values)
        dam_comm_mw = np.clip(dam_comm_mw, -self.max_power_mw, self.max_power_mw)
        market = self._get_market_state()

        violation_mw = abs(req_phy_mw - actual_phy_mw)
        is_violation = violation_mw > 0.01

        # I calculate shortfall risk for proactive penalty
        # Get upcoming DAM commitments for risk calculation (12 hours to match observation!)
        dam_commitments = []
        for i in range(1, 13):  # I fixed: was range(1,5), now matches 12h observation lookahead
            step_offset = int(i / self.time_step_hours)
            target = min(self.current_step + step_offset, len(self.df) - 1)
            future_dam = self.df.iloc[target].get('dam_commitment', 0.0)
            # I clip future DAM to battery limits too
            future_dam = np.clip(future_dam, -self.max_power_mw, self.max_power_mw)
            dam_commitments.append(future_dam)

        shortfall_info = self.feature_engine.calculate_shortfall_risk(
            battery_soc=self.current_soc,
            dam_commitments=dam_commitments,
            current_dam_commitment=dam_comm_mw  # I added current commitment
        )

        # Calculate reward with Market Depth and Shortfall Penalties
        # v13: Configurable penalties via reward_config + lookahead timing
        from gym_envs.reward_calculator import RewardCalculator
        rc = self.reward_config  # I use stored config for flexibility
        reward_calc = RewardCalculator(
            deg_cost_per_mwh=rc.get('deg_cost_per_mwh', 0.5),
            violation_penalty=rc.get('violation_penalty', 5.0),
            imbalance_penalty_multiplier=rc.get('imbalance_penalty_multiplier', 20.0),
            fixed_shortfall_penalty=rc.get('fixed_shortfall_penalty', 800.0),
            proactive_risk_penalty=rc.get('proactive_risk_penalty', 50.0),
            soc_buffer_bonus=rc.get('soc_buffer_bonus', 2.0),
            min_soc_threshold=rc.get('min_soc_threshold', 0.25),
            price_timing_bonus=rc.get('price_timing_bonus', 0.1),
            # v13 timing rewards
            peak_sell_bonus=rc.get('peak_sell_bonus', 10.0),
            early_sell_penalty=rc.get('early_sell_penalty', 15.0),
            early_sell_threshold=rc.get('early_sell_threshold', 1.3)
        )

        # I get mean price for timing bonus calculation
        mean_24h = self.price_mean_24h[self.current_step] if self.current_step < len(self.price_mean_24h) else 100.0

        # v13: Calculate max future price for lookahead timing reward
        max_future_price = row.get('price', 100.0)  # Default to current
        for i in range(1, 13):
            future_step = min(self.current_step + i, len(self.df) - 1)
            future_price = self.df.iloc[future_step].get('price', max_future_price)
            max_future_price = max(max_future_price, future_price)

        # v13: Check if current hour has DAM commitment (agent must respect it)
        has_dam_now = abs(dam_comm_mw) > 0.1

        reward_info = reward_calc.calculate(
            dam_commitment=dam_comm_mw,
            actual_physical=actual_phy_mw,
            market_state=market,
            is_violation=is_violation,
            shortfall_risk=shortfall_info['shortfall_risk'],
            current_soc=self.current_soc,
            price_mean_24h=mean_24h,
            # v13 lookahead timing
            max_future_price=max_future_price,
            has_dam_now=has_dam_now
        )
        
        reward = reward_info['reward']
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'req_mw': req_phy_mw,
            'actual_mw': actual_phy_mw,
            'violation': violation_mw,
            'soc': self.current_soc,
            'dam_price': market.dam_price,
            'best_bid': market.best_bid,
            'best_ask': market.best_ask,
            'spread': market.spread,
            **reward_info['components']
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        """
        Observation Construction via FeatureEngineer.
        This ensures parity with Production environment.
        """
        row = self.df.iloc[self.current_step]
        
        # 1. Prepare MarketStateDTO
        from agent.feature_engineer import MarketStateDTO
        
        # Forecast components (Apply noise if needed)
        current_price = row.get('price', 100.0)
        
        if self.use_perfect_forecast:
            # PERFECT MODE
            next_price = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)].get('price', current_price)
            max_24h = self.price_max_24h[self.current_step]
            min_24h = self.price_min_24h[self.current_step]
            mean_24h = self.price_mean_24h[self.current_step]
        else:
            # NOISY MODE (Simplified for readability, logic matches previous implementation)
            # Re-implementing simplified noise injection here or moving to FeatureEngineer?
            # Ideally, Env handles "Sensors" (adding noise), FeatureEngineer handles "Processing".
            # So we keep noise injection HERE.
            
            # ... (Noise injection logic omitted for brevity, assuming perfect for now or copying) ...
            # For this refactor, let's keep it simple and assume the noise logic stays here 
            # OR we pass noise parameters to FeatureEngineer?
            # Better: Keep raw noisy values here.
            
            # --- Quick Noisy Logic Re-implementation ---
            next_price_raw = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)].get('price', current_price)
            vol = self.price_volatility[self.current_step]
            err = self.forecast_err_1h * (1 + vol/50.0)
            next_price = next_price_raw * (1 + np.random.normal(0, err))
            
            # 24h stats noise
            max_24h = self.price_max_24h[self.current_step] * (1 + np.random.normal(0, self.forecast_err_24h))
            min_24h = self.price_min_24h[self.current_step] * (1 + np.random.normal(0, self.forecast_err_24h))
            mean_24h = self.price_mean_24h[self.current_step] * (1 + np.random.normal(0, self.forecast_err_24h))
        
        # DAM Commitments - Extended to 12 hours for better planning
        # I clip all DAM commitments to battery's physical limits
        dam_commitments = []
        for i in range(1, 13):  # 12 steps ahead
            step_offset = int(i / self.time_step_hours)
            target = min(self.current_step + step_offset, len(self.df)-1)
            future_dam = self.df.iloc[target].get('dam_commitment', 0.0)
            future_dam = np.clip(future_dam, -self.max_power_mw, self.max_power_mw)
            dam_commitments.append(future_dam)

        # Price Lookahead - 12 hours of future prices for strategic planning
        # v14: Use ML forecaster if available, otherwise use standard method
        if self.use_ml_forecaster and self.price_forecaster is not None:
            # Use trained ML model to predict future prices (realistic - no cheating)
            price_lookahead = self.price_forecaster.predict(
                self.df, self.current_step, add_noise=True
            )
        else:
            # Fallback: use actual prices with noise (less realistic but works)
            price_lookahead = []
            for i in range(1, 13):  # 12 steps ahead
                step_offset = int(i / self.time_step_hours)
                target = min(self.current_step + step_offset, len(self.df)-1)
                future_price = self.df.iloc[target].get('price', current_price)
                # Apply forecast noise if not using perfect forecast
                if not self.use_perfect_forecast:
                    # Error increases with horizon
                    horizon_factor = i / 12.0  # 0 to 1
                    err = self.forecast_err_1h + (self.forecast_err_24h - self.forecast_err_1h) * horizon_factor
                    future_price = future_price * (1 + np.random.normal(0, err))
                price_lookahead.append(future_price)

        # Get aFRR data if available (for market stress indicators)
        afrr_up = row.get('afrr_up_mw', 600.0)
        afrr_down = row.get('afrr_down_mw', 130.0)

        # I clip current DAM commitment to battery limits
        current_dam = np.clip(row.get('dam_commitment', 0.0), -self.max_power_mw, self.max_power_mw)

        market_state = MarketStateDTO(
            timestamp=pd.Timestamp(row.name),
            price=current_price,
            spread=self.base_spread,
            dam_commitment_now=current_dam,
            next_price=next_price,
            price_max_24h=max_24h,
            price_min_24h=min_24h,
            price_mean_24h=mean_24h,
            dam_commitments=dam_commitments,
            price_lookahead=price_lookahead,
            volatility=self.price_volatility[self.current_step],
            afrr_up_mw=afrr_up,
            afrr_down_mw=afrr_down
        )
        
        # 2. Get History Slices for Imbalance Risk
        # Need past 2 hours of data
        history_window = int(2 / self.time_step_hours) + 1
        start_idx = max(0, self.current_step - history_window)
        end_idx = self.current_step + 1
        
        price_history = self.imbalance_prices[start_idx:end_idx]
        res_history = self.imbalance_res[start_idx:end_idx]
        
        # Get hour for time-of-day risk (I use step index assuming hourly data)
        current_hour_approx = (self.current_step % 24) 
        
        imbalance_risk = self.feature_engine.calculate_imbalance_risk(
            price_history=price_history, 
            res_history=res_history,
            current_hour=current_hour_approx
        )
        
        # 3. Build Observation
        obs = self.feature_engine.build_observation(
            market=market_state,
            battery_soc=self.current_soc,
            imbalance_info=imbalance_risk
        )
        
        return obs
