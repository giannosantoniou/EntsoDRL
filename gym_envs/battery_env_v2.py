"""
Battery Environment V2 - Uses Data Provider Pattern.
Decoupled from data source - works with both Training and Production.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any

from data.data_provider import IDataProvider
from gym_envs.reward_calculator import RewardCalculator


class BatteryEnvV2(gym.Env):
    """
    Battery Trading Environment with Data Provider Pattern.
    
    This environment is DECOUPLED from the data source.
    It works identically whether data comes from CSV or Live API.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(
        self, 
        data_provider: IDataProvider,
        n_actions: int = 21,
        lookahead_hours: int = 4
    ):
        super().__init__()
        
        self.data_provider = data_provider
        self.n_actions = n_actions
        self.lookahead_hours = lookahead_hours
        
        # Get normalization config from provider (NO magic numbers!)
        self.norm_config = data_provider.get_normalization_config()
        
        # Battery params from normalization config
        self.max_power_mw = self.norm_config.power_scale
        self.capacity_mwh = self.max_power_mw  # Assume 1C battery
        self.efficiency = 0.94
        
        # Action space: discrete actions [-1.0 to 1.0]
        self.action_space = spaces.Discrete(n_actions)
        self.action_levels = np.linspace(-1.0, 1.0, n_actions)
        
        # Observation space (15 features)
        # Battery(3) + Market(2) + Time(2) + Price Lookahead(4) + DAM Lookahead(4)
        feature_count = 3 + 2 + 2 + lookahead_hours + lookahead_hours
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_count,), 
            dtype=np.float32
        )
        
        # Reward calculator
        self.reward_calc = RewardCalculator(deg_cost_per_mwh=10.0, violation_penalty=500.0)
        
        self.max_steps = data_provider.get_max_steps()
        self.current_step = 0
        
        print(f"BatteryEnvV2 Initialized with {n_actions} actions, {feature_count} features.")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.data_provider.reset()
        self.current_step = 0
        return self._get_observation(), {}

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean mask for valid actions.
        Used by MaskablePPO to prevent invalid actions.
        """
        telemetry = self.data_provider.get_battery_telemetry()
        
        max_discharge_action = telemetry.max_discharge_mw / self.max_power_mw
        max_charge_action = telemetry.max_charge_mw / self.max_power_mw
        
        mask = np.zeros(self.n_actions, dtype=bool)
        
        for i, action_level in enumerate(self.action_levels):
            if action_level >= 0:  # Discharge
                mask[i] = action_level <= max_discharge_action + 0.01
            else:  # Charge
                mask[i] = abs(action_level) <= max_charge_action + 0.01
        
        # Always allow idle
        center_idx = self.n_actions // 2
        mask[center_idx] = True
        
        return mask

    def step(self, action: int):
        # Get current state from provider
        market_state = self.data_provider.get_current_market_state()
        telemetry = self.data_provider.get_battery_telemetry()
        
        # Convert action to MW
        action_level = self.action_levels[action]
        req_phy_mw = action_level * self.max_power_mw
        
        # Apply physical constraints
        eff_sqrt = np.sqrt(self.efficiency)
        
        if req_phy_mw < 0:  # Charge
            actual_phy_mw = max(req_phy_mw, -telemetry.max_charge_mw)
        else:  # Discharge
            actual_phy_mw = min(req_phy_mw, telemetry.max_discharge_mw)
        
        # Dispatch to battery (DECOUPLED - works for both Sim and Live)
        self.data_provider.dispatch_battery(actual_phy_mw)
        
        # Calculate reward using market data
        violation_mw = abs(req_phy_mw - actual_phy_mw)
        is_violation = violation_mw > 0.01
        
        reward_info = self.reward_calc.calculate(
            dam_commitment=market_state.dam_commitment_mw,
            actual_physical=actual_phy_mw,
            price=market_state.intraday_price,
            is_violation=is_violation
        )
        
        reward = reward_info['reward']
        
        # Advance to next step
        has_more = self.data_provider.step()
        self.current_step += 1
        terminated = not has_more or self.current_step >= self.max_steps
        
        info = {
            'req_mw': req_phy_mw,
            'actual_mw': actual_phy_mw,
            'violation': violation_mw,
            'soc': telemetry.soc,
            **reward_info['components']
        }
        
        return self._get_observation(), reward, terminated, False, info

    def _get_observation(self):
        """
        Build observation from Data Provider.
        
        Features (15 total):
        1. SoC
        2. max_discharge (normalized)
        3. max_charge (normalized)
        4. current_price (normalized)
        5. dam_commitment (normalized)
        6. hour_sin
        7. hour_cos
        8-11. price_lookahead (4h)
        12-15. dam_lookahead (4h)
        """
        market = self.data_provider.get_current_market_state()
        telemetry = self.data_provider.get_battery_telemetry()
        forecast = self.data_provider.get_forecast(self.lookahead_hours)
        
        obs = [
            telemetry.soc,
            telemetry.max_discharge_mw / self.norm_config.power_scale,
            telemetry.max_charge_mw / self.norm_config.power_scale,
            market.intraday_price / self.norm_config.price_scale,
            market.dam_commitment_mw / self.norm_config.power_scale,
            market.hour_sin,
            market.hour_cos,
        ]
        
        # Price lookahead (normalized)
        for price in forecast.prices:
            obs.append(price / self.norm_config.price_scale)
        
        # DAM commitment lookahead (normalized)
        for dam in forecast.dam_commitments:
            obs.append(dam / self.norm_config.power_scale)
        
        return np.array(obs, dtype=np.float32)
