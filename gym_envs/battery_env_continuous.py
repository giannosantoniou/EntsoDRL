"""
Battery Environment with Continuous Action Space (PPO Compatible)
Based on user specifications for energy trading.

Key Design:
- Action: Continuous Net_Position_Change (-1 to 1) → clamped to valid range
- State: SoC, DAM Obligation, Prices (Spot+Long+Short), Forecast, Time
- Reward: Revenue - Cost - Penalties - Degradation
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class BatteryEnvContinuous(gym.Env):
    """
    Battery Trading Environment with Continuous Action Space.
    
    Designed for PPO algorithm with action clamping to prevent violations.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self, 
        df: pd.DataFrame, 
        battery_params: Dict[str, float], 
        initial_soc: float = 0.5
    ):
        super().__init__()
        
        self.df = df.copy()
        self.df.columns = [c.lower() for c in self.df.columns]
        self.battery_params = battery_params
        self.initial_soc = initial_soc
        
        self.capacity_mwh = battery_params.get('capacity_mwh', 50.0)
        self.max_power_mw = battery_params.get('max_discharge_mw', 50.0)
        self.efficiency = battery_params.get('efficiency', 0.94)
        
        # Continuous Action Space: Net Position Change [-1, 1]
        # Maps to [-max_power_mw, +max_power_mw]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # State Space Features:
        # 1. SoC (1)
        # 2. Available Discharge/Charge Capacity (2)
        # 3. DAM Obligation (1)
        # 4. Current Prices: Spot, Long, Short (3)
        # 5. Time: hour_sin, hour_cos, day_sin, day_cos (4)
        # 6. Load/Solar/Wind (3)
        # 7. Price Forecast 4h (4)
        # Total: 14 base + 4 forecast = 18
        
        self.forecast_hours = 4
        feature_count = 14 + self.forecast_hours  # 18 total
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feature_count,), dtype=np.float32)
        
        self.current_step = 0
        self.current_soc = self.initial_soc
        self.max_steps = len(self.df) - 1 - self.forecast_hours
        
        # SoC limits
        self.min_soc = 0.05
        self.max_soc = 0.95
        
        print(f"Continuous Environment Initialized")
        print(f"  Action: Continuous [-1, 1] → MW")
        print(f"  Features: {feature_count}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_soc = self.initial_soc
        return self._get_observation(), {}

    def _get_valid_action_range(self):
        """Calculate valid action range based on current SoC."""
        eff_sqrt = np.sqrt(self.efficiency)
        
        # Available energy
        available_discharge_mwh = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh)
        
        # Convert to MW (1 hour intervals)
        max_discharge_mw = min(self.max_power_mw, available_discharge_mwh * eff_sqrt)
        max_charge_mw = min(self.max_power_mw, available_charge_mwh / eff_sqrt)
        
        # Normalize to [-1, 1]
        max_discharge_action = max_discharge_mw / self.max_power_mw
        max_charge_action = max_charge_mw / self.max_power_mw
        
        return -max_charge_action, max_discharge_action

    def step(self, action: np.ndarray):
        # 1. Get raw action and clamp to valid range
        raw_action = float(action[0])
        min_action, max_action = self._get_valid_action_range()
        clamped_action = np.clip(raw_action, min_action, max_action)
        
        # Convert to MW
        actual_phy_mw = clamped_action * self.max_power_mw
        
        # 2. Update SoC
        eff_sqrt = np.sqrt(self.efficiency)
        if actual_phy_mw < 0:  # Charging
            energy_in = abs(actual_phy_mw) * eff_sqrt
            self.current_soc += energy_in / self.capacity_mwh
        else:  # Discharging
            energy_out = actual_phy_mw / eff_sqrt
            self.current_soc -= energy_out / self.capacity_mwh
        self.current_soc = np.clip(self.current_soc, 0.0, 1.0)
        
        # 3. Get market data
        row = self.df.iloc[self.current_step]
        dam_comm_mw = row.get('dam_commitment', 0.0)
        spot_price = row.get('price', 0.0)
        long_price = row.get('long', spot_price * 0.5)   # Imbalance buy price
        short_price = row.get('short', spot_price * 2.0)  # Imbalance sell price
        
        # 4. Calculate Financials
        # DAM Revenue (fixed commitment)
        dam_rev = dam_comm_mw * spot_price
        
        # Net position = what we actually delivered vs what we promised
        net_position = actual_phy_mw - dam_comm_mw
        
        # IntraDay/Imbalance Settlement
        imbalance_rev = 0.0
        if net_position > 0:  # Delivered MORE than promised (long position)
            imbalance_rev = net_position * long_price  # Sell excess at lower price
        elif net_position < 0:  # Delivered LESS than promised (short position)
            imbalance_rev = net_position * short_price  # Buy shortage at higher price
        
        # Degradation Cost (per cycle)
        deg_cost = abs(actual_phy_mw) * 10.0
        
        # 5. Penalties
        # Violation penalty (should be ~0 with clamping, but just in case)
        violation = abs(raw_action - clamped_action) * self.max_power_mw
        violation_penalty = 0.0
        if violation > 0.5:
            violation_penalty = violation * 500.0 + 1000.0
        
        # Idle penalty (not acting when there's commitment)
        idle_penalty = 0.0
        if abs(actual_phy_mw) < 1.0 and abs(dam_comm_mw) > 1.0:
            idle_penalty = abs(dam_comm_mw) * 50.0
        
        # Opportunity cost (missing high price)
        opportunity_cost = 0.0
        if abs(actual_phy_mw) < 5.0 and spot_price > 100.0:
            opportunity_cost = (spot_price - 100.0) * 0.5
        
        # 6. Total Reward
        reward = dam_rev + imbalance_rev - deg_cost - violation_penalty - idle_penalty - opportunity_cost
        reward /= 1000.0  # Normalize
        
        # 7. Step forward
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        info = {
            'actual_mw': actual_phy_mw,
            'dam_commitment': dam_comm_mw,
            'net_position': net_position,
            'violation': violation,
            'soc': self.current_soc,
            'dam_rev': dam_rev,
            'imbalance_rev': imbalance_rev,
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        eff_sqrt = np.sqrt(self.efficiency)
        
        # Available capacity
        available_discharge_mwh = max(0.0, (self.current_soc - self.min_soc) * self.capacity_mwh)
        available_charge_mwh = max(0.0, (self.max_soc - self.current_soc) * self.capacity_mwh)
        max_discharge_mw = min(self.max_power_mw, available_discharge_mwh * eff_sqrt)
        max_charge_mw = min(self.max_power_mw, available_charge_mwh / eff_sqrt)
        
        obs = []
        
        # 1. SoC
        obs.append(self.current_soc)
        
        # 2. Available Capacity (normalized)
        obs.append(max_discharge_mw / self.max_power_mw)
        obs.append(max_charge_mw / self.max_power_mw)
        
        # 3. DAM Obligation (normalized)
        dam = row.get('dam_commitment', 0.0)
        obs.append(dam / self.max_power_mw)
        
        # 4. Current Prices (normalized by typical price ~100)
        spot = row.get('price', 100.0)
        long_p = row.get('long', spot * 0.5)
        short_p = row.get('short', spot * 2.0)
        obs.append(spot / 100.0)
        obs.append(long_p / 100.0)
        obs.append(short_p / 100.0)
        
        # 5. Time features
        obs.append(row.get('hour_sin', 0.0))
        obs.append(row.get('hour_cos', 0.0))
        obs.append(row.get('day_sin', 0.0))
        obs.append(row.get('day_cos', 0.0))
        
        # 6. Load/Generation context
        load = row.get('load_forecast', 0.0) / 10000.0
        solar = row.get('solar', 0.0) / 10000.0
        wind = row.get('wind_onshore', 0.0) / 10000.0
        obs.append(load)
        obs.append(solar)
        obs.append(wind)
        
        # 7. Price Forecast (next 4 hours)
        for i in range(1, self.forecast_hours + 1):
            if self.current_step + i < len(self.df):
                future_price = self.df.iloc[self.current_step + i].get('price', spot)
            else:
                future_price = spot
            obs.append(future_price / 100.0)
        
        return np.array(obs, dtype=np.float32)
