
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

class BatteryTradingEnvV3(gym.Env):
    """
    V3: True Intraday/Balancing Environment.
    The Day-Ahead (DAM) commitment is now an INPUT (Observation) derived from an external process/simulation.
    The Agent's sole job is to decide the Physical Dispatch (Real-Time), managing imbalances.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df: pd.DataFrame, battery_params: Dict[str, float], initial_soc: float = 0.5):
        super().__init__()
        
        self.df = df
        self.df.columns = [c.lower() for c in self.df.columns]
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        
        # Battery Specs
        self.capacity_mwh = battery_params.get('capacity_mwh', 50.0)
        self.max_power_mw = battery_params.get('max_discharge_mw', 50.0)
        self.efficiency = battery_params.get('efficiency', 0.94)
        self.cost_per_mwh_throughput = battery_params.get('cost_per_mwh_throughput', 0.0)
        self.transaction_fee = battery_params.get('transaction_fee', 0.5)
        
        # Action Space: 1 Continuous Value (Net Physical Dispatch)
        # Range: [-1, 1]. 1 = Max Discharge (50MW), -1 = Max Charge (50MW).
        # OR: Should action be "Adjustment to DAM"?
        # If DAM says 50MW, and I want 0MW, adjustment is -50.
        # It's cleaner if Action = Desired Physical Setpoint. 
        # The Environment calculates Imbalance = Setpoint - DAM.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Features including DAM Commitment
        self.features = [
            'price', 'load_forecast', 'solar', 'wind_onshore', 
            'dam_commitment', # NEW: The obligation we must respect or pay for
            'hour_sin', 'hour_cos'
        ]
        
        feature_count = 1 + len(self.features) # SoC + features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.max_steps = len(self.df) - 1
        
    def set_cost(self, cost: float):
        self.cost_per_mwh_throughput = cost

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_soc = self.initial_soc
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        # 1. Parse Action (Desired Physical Flow)
        req_phy_mw = float(action[0]) * self.max_power_mw
        
        # 2. Apply Physical Constraints (Battery Physics)
        # SOC Constraints
        # Max Charge (MW) constrained by Space Left (MWh)
        # Max Discharge (MW) constrained by Energy Left (MWh)
        # Assuming 1 Hour Step
        
        # Limits (0.05 - 0.95)
        min_soc = 0.05
        max_soc = 0.95
        
        energy_in_max = max(0.0, (max_soc - self.current_soc) * self.capacity_mwh / np.sqrt(self.efficiency))
        energy_out_max = max(0.0, (self.current_soc - min_soc) * self.capacity_mwh * np.sqrt(self.efficiency))
        
        # Clip Request
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
        
        # 4. Financial Calculation
        # Load Data
        row = self.df.iloc[self.current_step]
        
        # Inputs
        dam_comm_mw = row.get('dam_commitment', 0.0)
        dam_price = row.get('price', 0.0)
        
        # Imbalance
        # Imbalance = Actual - Scheduled
        imbalance_mw = actual_phy_mw - dam_comm_mw
        
        # Imbalance Prices (Mocking if missing)
        # Usually: Short Price > Spot > Long Price
        # Fallback logic:
        long_price = row.get('long', dam_price * 0.5) 
        short_price = row.get('short', dam_price * 1.5)
        
        # Revenue 1: DAM Settlement (Sunk/Fixed)
        # We assume we ALREADY got paid for DAM Commitment (previous day).
        # BUT, to learn "Total Value", we usually include it.
        # The Agent calculates: Total Profit = DAM_Rev + Imbalance_Rev - Costs.
        # Since DAM_Rev is derived from inputs, it's a constant bias.
        # Agent maximizes (Imbalance_Rev - Costs).
        dam_revenue = dam_comm_mw * dam_price
        
        # Revenue 2: Balancing
        bal_revenue = 0.0
        if imbalance_mw > 0: # Long (Surplus) -> Sell at Long Price
            bal_revenue = imbalance_mw * long_price
        elif imbalance_mw < 0: # Short (Deficit) -> Buy at Short Price
            bal_revenue = imbalance_mw * short_price
            
        # Costs
        deg_cost = abs(actual_phy_mw) * self.cost_per_mwh_throughput
        
        # Reward
        reward = dam_revenue + bal_revenue - deg_cost
        
        # Normalize Reward for PPO stability (optional but reccomended)
        reward /= 1000.0 
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'dam_comm': dam_comm_mw,
            'actual_phy': actual_phy_mw,
            'imbalance': imbalance_mw,
            'dam_rev': dam_revenue,
            'bal_rev': bal_revenue,
            'reward': reward
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        row = self.df.iloc[self.current_step]
        obs = [self.current_soc]
        for f in self.features:
            val = row.get(f, 0.0)
            # Manual Scaling as per our lesson learned
            if 'price' in f: val /= 100.0
            if 'load' in f or 'solar' in f or 'wind' in f: val /= 10000.0
            if 'dam_commitment' in f: val /= 50.0 # Scale MW to 0-1 (approx)
            obs.append(val)
            
        return np.array(obs, dtype=np.float32)
