"""
Battery Environment for RecurrentPPO with Soft Action Masking.

Based on BatteryEnvMasked but uses penalty-based soft masking
instead of hard action masking (since MaskableRecurrentPPO doesn't exist).
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


class BatteryEnvRecurrent(gym.Env):
    """
    Battery Trading Environment for RecurrentPPO.
    
    Key differences from BatteryEnvMasked:
    - No action_masks() method (RecurrentPPO doesn't support it)
    - Invalid actions receive a penalty instead of being blocked
    - LSTM-friendly: consistent observation structure
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self, 
        df: pd.DataFrame, 
        battery_params: Dict[str, float], 
        n_actions: int = 21,
        initial_soc: float = 0.5,
        base_spread: float = 4.0,
        noise_std: float = 10.0,
        price_cap: float = 200.0,
        extreme_spike_prob: float = 0.0005,
        time_step_hours: float = 1.0,
        # Soft Masking Config
        invalid_action_penalty: float = 50.0,  # Penalty for invalid actions
        # Forecast Uncertainty Parameters
        forecast_err_1h: float = 0.08,
        forecast_err_24h: float = 0.20,
        forecast_bias_prob: float = 0.05,
        forecast_miss_prob: float = 0.005,
        use_perfect_forecast: bool = False
    ):
        super().__init__()
        
        self.df = df.copy()
        self.df.columns = [c.lower() for c in self.df.columns]
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        self.n_actions = n_actions
        self.base_spread = base_spread
        self.noise_std = noise_std
        self.price_cap = price_cap
        self.extreme_spike_prob = extreme_spike_prob
        self.invalid_action_penalty = invalid_action_penalty
        
        # Forecast Uncertainty Settings
        self.forecast_err_1h = forecast_err_1h
        self.forecast_err_24h = forecast_err_24h
        self.forecast_bias_prob = forecast_bias_prob
        self.forecast_miss_prob = forecast_miss_prob
        self.use_perfect_forecast = use_perfect_forecast
        
        self.capacity_mwh = battery_params.get('capacity_mwh', 50.0)
        self.max_power_mw = battery_params.get('max_discharge_mw', 50.0)
        self.efficiency = battery_params.get('efficiency', 0.94)
        
        # Time step configuration
        self.time_step_hours = time_step_hours
        self.steps_per_hour = int(1.0 / time_step_hours)
        
        # Pre-compute volatility for spread simulation
        volatility_window = int(24 / self.time_step_hours)
        prices = self.df['price'].values if 'price' in self.df.columns else np.ones(len(self.df)) * 100
        self.price_volatility = pd.Series(prices).rolling(volatility_window, min_periods=1).std().fillna(5.0).values
        
        # Action space
        self.action_space = spaces.Discrete(self.n_actions)
        self.action_levels = np.linspace(-1.0, 1.0, self.n_actions)
        
        # Observation space (19 features - same as BatteryEnvMasked for compatibility)
        self.lookahead_hours = 4
        self.strategic_lookahead_hours = 24
        self.lookahead_steps = int(self.lookahead_hours / self.time_step_hours)
        self.strategic_lookahead = int(self.strategic_lookahead_hours / self.time_step_hours)
        feature_count = 19
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        # Feature Engineer
        from agent.feature_engineer import FeatureEngineer
        self.feature_engine = FeatureEngineer(config={
            'capacity_mwh': self.capacity_mwh,
            'max_power_mw': self.max_power_mw,
            'efficiency': self.efficiency,
            'time_step_hours': self.time_step_hours
        })
        
        # Pre-compute statistics
        self._precompute_24h_stats()
        self._prepare_imbalance_data()
        
        # State
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.max_steps = len(self.df) - 1 - max(self.lookahead_steps, self.strategic_lookahead)
        
        # SoC limits
        self.min_soc = 0.05
        self.max_soc = 0.95
        
        time_res = f"{int(self.time_step_hours * 60)}min" if self.time_step_hours < 1 else "hourly"
        print(f"Recurrent Environment Initialized: {self.n_actions} actions | {time_res} | Soft Masking (penalty={self.invalid_action_penalty})")

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

    def _get_valid_action_limits(self):
        """Calculate valid action limits based on current SoC."""
        eff_sqrt = np.sqrt(self.efficiency)
        
        # Available energy in MWh
        available_discharge_mwh = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh)
        
        # Max power considering what can be delivered in this time step
        max_discharge_mw = min(self.max_power_mw, (available_discharge_mwh * eff_sqrt) / self.time_step_hours)
        max_charge_mw = min(self.max_power_mw, (available_charge_mwh / eff_sqrt) / self.time_step_hours)
        
        max_discharge_action = max_discharge_mw / self.max_power_mw
        max_charge_action = max_charge_mw / self.max_power_mw
        
        return max_charge_action, max_discharge_action

    def _is_action_valid(self, action: int) -> bool:
        """Check if action is valid given current SoC."""
        max_charge_action, max_discharge_action = self._get_valid_action_limits()
        action_level = self.action_levels[action]
        
        if action_level >= 0:  # Discharge
            return action_level <= max_discharge_action + 0.01
        else:  # Charge
            return abs(action_level) <= max_charge_action + 0.01

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean mask of valid actions.
        Note: This is NOT used by RecurrentPPO, but kept for compatibility/debugging.
        """
        max_charge_action, max_discharge_action = self._get_valid_action_limits()
        
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
        """Generate market state with hybrid volatility filtering."""
        row = self.df.iloc[self.current_step]
        raw_price = row.get('price', 100.0)
        
        # INTRADAY PRICE: Unrestricted
        noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0
        intraday_price = max(0.1, raw_price + noise)
        intraday_price = min(intraday_price, 1000.0)
        
        # DAM PRICE: Filtered (20% chance of keeping spikes >200€)
        dam_price = raw_price
        if dam_price > 200.0:
            if np.random.random() > 0.20:
                dam_price = 200.0
        dam_price = min(dam_price, 1000.0)
        
        # Bid-Ask spread
        volatility = self.price_volatility[self.current_step]
        spread = self.base_spread * (1 + volatility / 30)
        spread = min(spread, 15.0)
        
        best_bid = max(0.1, intraday_price - spread / 2)
        best_ask = intraday_price + spread / 2
        
        return SimpleMarketState(
            dam_price=dam_price,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread
        )

    def step(self, action: int):
        # Check if action is valid BEFORE execution
        is_invalid_action = not self._is_action_valid(action)
        
        # Convert action
        action_level = self.action_levels[action]
        req_phy_mw = action_level * self.max_power_mw
        
        # Physical limits
        eff_sqrt = np.sqrt(self.efficiency)
        
        energy_in_max_step = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh / eff_sqrt)
        energy_out_max_step = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh * eff_sqrt)
        
        power_in_max = energy_in_max_step / self.time_step_hours
        power_out_max = energy_out_max_step / self.time_step_hours
        
        if req_phy_mw < 0:  # Charging
            actual_phy_mw = max(req_phy_mw, -self.max_power_mw, -power_in_max)
        else:  # Discharging
            actual_phy_mw = min(req_phy_mw, self.max_power_mw, power_out_max)
        
        # Update SoC
        energy_mwh = abs(actual_phy_mw) * self.time_step_hours
        if actual_phy_mw < 0:  # Charging
            self.current_soc += (energy_mwh * eff_sqrt) / self.capacity_mwh
        else:  # Discharging
            self.current_soc -= (energy_mwh / eff_sqrt) / self.capacity_mwh
        self.current_soc = np.clip(self.current_soc, 0.0, 1.0)
        
        # Get market state and calculate reward
        row = self.df.iloc[self.current_step]
        dam_comm_mw = row.get('dam_commitment', 0.0)
        market = self._get_market_state()
        
        violation_mw = abs(req_phy_mw - actual_phy_mw)
        is_violation = violation_mw > 0.01
        
        # Calculate reward with Market Depth
        from gym_envs.reward_calculator import RewardCalculator
        reward_calc = RewardCalculator(deg_cost_per_mwh=10.0, violation_penalty=5.0)
        
        reward_info = reward_calc.calculate(
            dam_commitment=dam_comm_mw,
            actual_physical=actual_phy_mw,
            market_state=market,
            is_violation=is_violation
        )
        
        reward = reward_info['reward']
        
        # SOFT ACTION MASKING: Apply penalty for invalid actions
        if is_invalid_action:
            reward -= self.invalid_action_penalty
        
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
            'invalid_action': is_invalid_action,
            **reward_info['components']
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        """Observation Construction via FeatureEngineer."""
        row = self.df.iloc[self.current_step]
        
        from agent.feature_engineer import MarketStateDTO
        
        current_price = row.get('price', 100.0)
        
        if self.use_perfect_forecast:
            next_price = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)].get('price', current_price)
            max_24h = self.price_max_24h[self.current_step]
            min_24h = self.price_min_24h[self.current_step]
            mean_24h = self.price_mean_24h[self.current_step]
        else:
            next_price_raw = self.df.iloc[min(self.current_step + 1, len(self.df) - 1)].get('price', current_price)
            vol = self.price_volatility[self.current_step]
            err = self.forecast_err_1h * (1 + vol/50.0)
            next_price = next_price_raw * (1 + np.random.normal(0, err))
            
            max_24h = self.price_max_24h[self.current_step] * (1 + np.random.normal(0, self.forecast_err_24h))
            min_24h = self.price_min_24h[self.current_step] * (1 + np.random.normal(0, self.forecast_err_24h))
            mean_24h = self.price_mean_24h[self.current_step] * (1 + np.random.normal(0, self.forecast_err_24h))
        
        # DAM Commitments
        dam_commitments = []
        for i in range(1, 5):
            step_offset = int(i / self.time_step_hours)
            target = min(self.current_step + step_offset, len(self.df)-1)
            dam_commitments.append(self.df.iloc[target].get('dam_commitment', 0.0))
            
        market_state = MarketStateDTO(
            timestamp=pd.Timestamp(row.name),
            price=current_price,
            spread=self.base_spread,
            dam_commitment_now=row.get('dam_commitment', 0.0),
            next_price=next_price,
            price_max_24h=max_24h,
            price_min_24h=min_24h,
            price_mean_24h=mean_24h,
            dam_commitments=dam_commitments,
            volatility=self.price_volatility[self.current_step]
        )
        
        # Imbalance Risk
        history_window = int(2 / self.time_step_hours) + 1
        start_idx = max(0, self.current_step - history_window)
        end_idx = self.current_step + 1
        
        price_history = self.imbalance_prices[start_idx:end_idx]
        res_history = self.imbalance_res[start_idx:end_idx]
        
        current_hour_approx = (self.current_step % 24)
        
        imbalance_risk = self.feature_engine.calculate_imbalance_risk(
            price_history=price_history, 
            res_history=res_history,
            current_hour=current_hour_approx
        )
        
        obs = self.feature_engine.build_observation(
            market=market_state,
            battery_soc=self.current_soc,
            imbalance_info=imbalance_risk
        )
        
        return obs
