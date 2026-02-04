import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class DayAheadBatchEnv(gym.Env):
    """
    Day-Ahead Batch Environment (Phase 10).
    Simulates the planning of a Full 24-hour Schedule in advance.
    
    Logic:
    1. Agent observes:
       - Current SoC (at start of Day)
       - Forecasts for 24h (Price, Solar, Wind)
    2. Agent Acts:
       - 24 Float values (MW Commitments for each hour).
    3. Step:
       - Simulates the 24 hours internally.
       - Uses ACTUAL prices/generation (Action vs Reality).
       - Calculates Imbalance and Penalties.
       - Updates SoC for the next Day.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, battery_params: Dict[str, float], initial_soc: float = 0.5):
        super().__init__()
        
        self.df = df
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        
        # Physics
        self.capacity = battery_params.get('capacity_mwh', 50.0)
        self.max_mw = battery_params.get('max_discharge_mw', 50.0)
        self.efficiency = battery_params.get('efficiency', 0.9)
        self.cost_deg = battery_params.get('cost_per_mwh_throughput', 15.0)
        
        # Action Space: 24 floats [-1, 1]
        # -1 = Max Charge, +1 = Max Discharge
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        
        # Observation Space:
        # 1 (SoC) + 24 (Price) + 24 (Solar) + 24 (Wind) = 73 floats
        # We perform simple scaling (Price/100, MW/1000)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(73,), dtype=np.float32)
        
        # Data Splitting
        self.total_days = len(df) // 24
        self.current_day = 0
        self.current_soc = initial_soc
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_day = 0
        self.current_soc = self.initial_soc
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Action is (24,) array of DAM Commitments
        
        # Get Actual Data for this Day (24 rows)
        start_idx = self.current_day * 24
        day_data = self.df.iloc[start_idx : start_idx + 24]
        
        total_reward = 0.0
        info_tracker = {'imbalance_mwh': 0.0, 'revenue': 0.0, 'penalties': 0.0}
        
        # ---------------------------------------------------------
        # SYNTHETIC "EASY MODE" (User Request)
        # Force Prices to be deterministic: 12h Low, 12h High.
        # This overrides the historical DataFrame for purely RL training purposes.
        # ---------------------------------------------------------
        
        # Pattern: Hours 0-11 = 50.0 EUR, Hours 12-23 = 150.0 EUR
        synthetic_prices = np.array([50.0] * 12 + [150.0] * 12, dtype=np.float32)
        
        for h in range(24):
            # 1. Parse Hourly Command
            # Scale to MW
            target_mw = float(action[h]) * self.max_mw
            
            # 2. Physics Execution (Internal Simulation)
            # 5-95% Hard Limits logic
            soc_min_mwh = 0.05 * self.capacity
            soc_max_mwh = 0.95 * self.capacity
            current_energy = self.current_soc * self.capacity
            
            # Calc Available Power
            max_in = (soc_max_mwh - current_energy) / np.sqrt(self.efficiency)
            max_out = (current_energy - soc_min_mwh) * np.sqrt(self.efficiency)
            
            # Clip Physical Flow
            if target_mw > 0: # Discharge
                physical_mw = min(target_mw, max_out, self.max_mw)
            else: # Charge
                # target is neg, max_in is pos.
                physical_mw = max(target_mw, -max_in, -self.max_mw)
                
            # Update SoC
            if physical_mw > 0:
                energy_delta = -physical_mw / np.sqrt(self.efficiency)
            else:
                energy_delta = abs(physical_mw) * np.sqrt(self.efficiency)
            
            current_energy += energy_delta
            self.current_soc = np.clip(current_energy / self.capacity, 0.0, 1.0)
            
            # 3. Financial Settlement (Using Synthetic Price)
            price = synthetic_prices[h]
            
            # DAM Revenue
            dam_rev = target_mw * price
            
            # Imbalance
            imbalance = physical_mw - target_mw
            
            # Penalty (Still strict to force compliance)
            if imbalance < 0: # Short
                penalty = abs(imbalance) * (price * 1.5) 
                imbalance_rev = -penalty
            else: # Long
                imbalance_rev = 0.0 
                
            # Degradation
            deg_cost = abs(physical_mw) * self.cost_deg
            
            # Hourly Reward
            reward_h = dam_rev + imbalance_rev - deg_cost
            total_reward += reward_h
            
            # Trackers
            info_tracker['imbalance_mwh'] += abs(imbalance)
            info_tracker['revenue'] += dam_rev
            info_tracker['penalties'] += imbalance_rev
            
        # End of Day
        self.current_day += 1
        terminated = self.current_day >= self.total_days
        truncated = False
        
        return self._get_obs(), total_reward, terminated, truncated, info_tracker

    def _get_obs(self):
        if self.current_day >= self.total_days:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        start_idx = self.current_day * 24
        day_data = self.df.iloc[start_idx : start_idx + 24]
        
        # SYNTHETIC "EASY MODE" OBSERVATION
        # Agent must SEE the synthetic prices (50 vs 150) to learn.
        synthetic_prices = np.array([50.0] * 12 + [150.0] * 12, dtype=np.float32)
        prices = synthetic_prices / 100.0 # Scale
        # No Noise for Easy Mode
        
        solars = day_data.get('solar', np.zeros(24)).values / 1000.0
        winds = day_data.get('wind_onshore', np.zeros(24)).values / 1000.0
        
        # Flatten
        obs = np.concatenate(([self.current_soc], prices, solars, winds)).astype(np.float32)
        return obs
