
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class BatteryTradingEnvV5(gym.Env):
    """
    V5: The "Smart" Reliable Intraday Agent.
    
    KEY DIFFERENCE FROM V4:
    Distinguishes between "Intraday Trading" (Financial Choice) and "Constraint Failure" (Physical Failure).
    
    Logic:
    1. Agent Requests X MW (Physical Dispatch).
    2. Environment checks constraints (SoC, MaxPower).
    3. Actual Y MW is delivered.
    4. If X != Y (Constraint Violation), Penalty is massive.
    5. Imbalance (Y - DAM) is settled financially at Spot Price.
       This allows Agent to BUY energy to fill capacity (Y < DAM) without Penalty, just Cost.
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
        
        # Action: Desired Physical Dispatch (-1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.base_features = ['price', 'load_forecast', 'solar', 'wind_onshore', 'dam_commitment', 'hour_sin', 'hour_cos']
        self.lookahead_steps = 24
        
        feature_count = 1 + len(self.base_features) + (2 * self.lookahead_steps) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.max_steps = len(self.df) - 1 - self.lookahead_steps
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_soc = self.initial_soc
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Parse Request
        req_phy_mw = float(action[0]) * self.max_power_mw
        
        # 2. Physical Limits
        min_soc = 0.05
        max_soc = 0.95
        
        energy_in_max = max(0.0, (max_soc - self.current_soc) * self.capacity_mwh / np.sqrt(self.efficiency))
        energy_out_max = max(0.0, (self.current_soc - min_soc) * self.capacity_mwh * np.sqrt(self.efficiency))
        
        # Determine Actual
        if req_phy_mw < 0: # Charge
            actual_phy_mw = max(req_phy_mw, -self.max_power_mw, -energy_in_max)
        else: # Discharge
            actual_phy_mw = min(req_phy_mw, self.max_power_mw, energy_out_max)
            
        # 3. Update SoC
        eff_sqrt = np.sqrt(self.efficiency)
        if actual_phy_mw < 0:
            self.current_soc += (abs(actual_phy_mw) * eff_sqrt) / self.capacity_mwh
        else:
            self.current_soc -= (actual_phy_mw / eff_sqrt) / self.capacity_mwh
        self.current_soc = np.clip(self.current_soc, 0.0, 1.0)
        
        # 4. Financials
        row = self.df.iloc[self.current_step]
        dam_comm_mw = row.get('dam_commitment', 0.0)
        price = row.get('price', 0.0)
        
        # DAM Revenue (Fixed)
        dam_rev = dam_comm_mw * price
        
        # Intraday Trading
        # If Actual != DAM, we traded the difference.
        # Net Trade = Actual - DAM
        trade_mw = actual_phy_mw - dam_comm_mw
        
        # Trading Settlement (Spot Price +/- Spread)
        # Buying Replacement Energy is costly (Price + Spread).
        # Selling Surplus Energy is cheap (Price - Spread).
        
        trade_rev = 0.0
        if trade_mw > 0: # Selling Extra
            trade_rev = trade_mw * (price * 0.9) # Sell Low
        elif trade_mw < 0: # Buying Missing
            trade_rev = trade_mw * (price * 1.1) # Buy High (Cost)
            
        deg_cost = abs(actual_phy_mw) * 10.0
        
        # 5. CONSTRAINT VIOLATION PENALTY 🦂 (STRENGTHENED)
        # Did the Agent REQUEST something he couldn't do?
        # Violation = |Requested - Actual|
        
        violation_mw = abs(req_phy_mw - actual_phy_mw)
        
        reliability_penalty = 0.0
        if violation_mw > 0.5:  # Lower tolerance (was 1.0)
            # Stronger penalty: 2.5x increase
            reliability_penalty = violation_mw * 500.0 + 1000.0
        
        # 6. IDLE PENALTY 💤 (NEW)
        # Penalize agent for being idle when there's a DAM commitment
        idle_penalty = 0.0
        if abs(actual_phy_mw) < 1.0 and abs(dam_comm_mw) > 1.0:
            # Agent is idle but has commitment to fulfill
            idle_penalty = abs(dam_comm_mw) * 50.0
        
        # 7. OPPORTUNITY COST 📉 (NEW)
        # Encourage action during high prices
        opportunity_cost = 0.0
        if abs(actual_phy_mw) < 5.0 and price > 100.0:
            # Missing opportunity during high price
            opportunity_cost = (price - 100.0) * 0.5
            
        reward = dam_rev + trade_rev - deg_cost - reliability_penalty - idle_penalty - opportunity_cost
        
        # Normalize
        reward /= 1000.0
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'req_mw': req_phy_mw,
            'actual_mw': actual_phy_mw,
            'violation': violation_mw,
            'penalty': reliability_penalty,
            'idle_penalty': idle_penalty,
            'opportunity_cost': opportunity_cost,
            'reward': reward
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        # Flattened Vector (Same as V4)
        row = self.df.iloc[self.current_step]
        obs = [self.current_soc]
        for f in self.base_features:
            val = row.get(f, 0.0)
            if 'price' in f: val /= 100.0
            if 'load' in f or 'solar' in f or 'wind' in f: val /= 10000.0
            if 'dam_commitment' in f: val /= 50.0
            obs.append(val)
        
        future_slice = self.df.iloc[self.current_step + 1 : self.current_step + 1 + self.lookahead_steps]
        pad_len = self.lookahead_steps - len(future_slice)
        
        dam_future = future_slice['dam_commitment'].values / 50.0
        if pad_len > 0: dam_future = np.pad(dam_future, (0, pad_len), 'edge')
        obs.extend(dam_future)
        
        price_future = future_slice['price'].values / 100.0
        if pad_len > 0: price_future = np.pad(price_future, (0, pad_len), 'edge')
        obs.extend(price_future)
            
        return np.array(obs, dtype=np.float32)
