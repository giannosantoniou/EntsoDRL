
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class BatteryTradingEnvV4(gym.Env):
    """
    V4: The "Reliable" Intraday Agent.
    Improvements from V3:
    1. Observation Space includes 24-hour LOOKAHEAD for DAM Commitments and Prices.
       This allows the agent to plan SoC usage based on future obligations.
    2. Reward Function includes a "Stinging Rules Penalty" for missing obligations.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame, battery_params: Dict[str, float], initial_soc: float = 0.5):
        super().__init__()
        
        self.df = df
        self.df.columns = [c.lower() for c in self.df.columns]
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        
        self.capacity_mwh = battery_params.get('capacity_mwh', 50.0)
        self.max_power_mw = battery_params.get('max_discharge_mw', 50.0)
        self.efficiency = battery_params.get('efficiency', 0.94)
        
        # Action: Physical Dispatch (-1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Features
        # 1. Current State Features (7): SoC, Price, Load, Solar, Wind, HourSin, HourCos
        # 2. Lookahead Features (48): 
        #    - Next 24h DAM Commitment (relative to current step)
        #    - Next 24h Price Forecast (relative to current step)
        # Total = 1 + 6 + 24 + 24 = 55 features.
        
        self.base_features = ['price', 'load_forecast', 'solar', 'wind_onshore', 'dam_commitment', 'hour_sin', 'hour_cos']
        self.lookahead_steps = 24
        
        feature_count = 1 + len(self.base_features) + (2 * self.lookahead_steps) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.max_steps = len(self.df) - 1 - self.lookahead_steps # Avoid bounds error
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_soc = self.initial_soc
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Physical Logic (Same as V3)
        req_phy_mw = float(action[0]) * self.max_power_mw
        
        min_soc = 0.05
        max_soc = 0.95
        
        # Calculate Constraints
        energy_in_max = max(0.0, (max_soc - self.current_soc) * self.capacity_mwh / np.sqrt(self.efficiency))
        energy_out_max = max(0.0, (self.current_soc - min_soc) * self.capacity_mwh * np.sqrt(self.efficiency))
        
        if req_phy_mw < 0: # Charge
            actual_phy_mw = max(req_phy_mw, -self.max_power_mw, -energy_in_max)
        else: # Discharge
            actual_phy_mw = min(req_phy_mw, self.max_power_mw, energy_out_max)
            
        # Update SoC
        eff_sqrt = np.sqrt(self.efficiency)
        if actual_phy_mw < 0:
            self.current_soc += (abs(actual_phy_mw) * eff_sqrt) / self.capacity_mwh
        else:
            self.current_soc -= (actual_phy_mw / eff_sqrt) / self.capacity_mwh
        self.current_soc = np.clip(self.current_soc, 0.0, 1.0)
        
        # 2. Financials & Penalty
        row = self.df.iloc[self.current_step]
        dam_comm_mw = row.get('dam_commitment', 0.0)
        price = row.get('price', 0.0)
        
        imbalance_mw = actual_phy_mw - dam_comm_mw
        
        # Revenue Streams
        dam_rev = dam_comm_mw * price
        
        # Imbalance Settlement (Simplified to pure cost for now to emphasize penalty)
        # If Imbalance != 0, we pay price * penalty_factor?
        # Let's keep realistic settlement:
        long_price = price * 0.5
        short_price = price * 2.0 # More punitive short price
        
        bal_rev = 0.0
        if imbalance_mw > 0:
            bal_rev = imbalance_mw * long_price
        elif imbalance_mw < 0:
            bal_rev = imbalance_mw * short_price
            
        deg_cost = abs(actual_phy_mw) * 10.0 # Fixed deg cost 10 EUR
        
        # 3. THE USER'S "STINGING PENALTY" 🦂
        # If Agent fails to deliver commitment (within 1MW tolerance), apply HUGE penalty.
        # This acts as a Barrier Function.
        
        reliability_penalty = 0.0
        if abs(imbalance_mw) > 1.0: # Tolerance 1 MW
            # Penalty is proportional to violation MW, but SCALED UP massively.
            # Example: 100 EUR per MW of violation on top of market cost.
            reliability_penalty = abs(imbalance_mw) * 100.0 
            
            # Plus a fixed "Shame" penalty for generating any imbalance
            reliability_penalty += 500.0 
            
        reward = dam_rev + bal_rev - deg_cost - reliability_penalty
        
        # Normalize
        reward /= 1000.0
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'imbalance': imbalance_mw,
            'penalty': reliability_penalty,
            'reward': reward
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        # Flattened Vector construction
        row = self.df.iloc[self.current_step]
        
        # 1. Base Features ([SoC, Price, Load...])
        obs = [self.current_soc]
        for f in self.base_features:
            val = row.get(f, 0.0)
            if 'price' in f: val /= 100.0
            if 'load' in f or 'solar' in f or 'wind' in f: val /= 10000.0
            if 'dam_commitment' in f: val /= 50.0
            obs.append(val)
            
        # 2. Lookahead (Next 24 steps)
        # We need prices and commitments
        # Slice the dataframe
        future_slice = self.df.iloc[self.current_step + 1 : self.current_step + 1 + self.lookahead_steps]
        
        # If nearing end of DF, pad with last known value or zeros
        pad_len = self.lookahead_steps - len(future_slice)
        
        # DAM Commitments
        dam_future = future_slice['dam_commitment'].values / 50.0
        if pad_len > 0:
            dam_future = np.pad(dam_future, (0, pad_len), 'edge')
        obs.extend(dam_future)
        
        # Prices
        price_future = future_slice['price'].values / 100.0
        if pad_len > 0:
            price_future = np.pad(price_future, (0, pad_len), 'edge')
        obs.extend(price_future)
            
        return np.array(obs, dtype=np.float32)
