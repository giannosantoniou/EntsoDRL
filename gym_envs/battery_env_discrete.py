
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .battery_env_v5 import BatteryTradingEnvV5

class BatteryTradingEnvDiscrete(BatteryTradingEnvV5):
    """
    Discrete Action Wrapper for DQN Compatibility.
    Discretizes the continuous action space [-1, 1] into N discrete levels.
    """
    def __init__(self, data, battery_params, n_actions=21):
        super().__init__(data, battery_params)
        
        self.n_actions = n_actions
        # Action space: 0 to n_actions-1
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Pre-compute action mapping
        # e.g. if n=5: [0, 1, 2, 3, 4] -> [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.action_levels = np.linspace(-1.0, 1.0, self.n_actions)
        
        print(f"Discrete Environment Initialized with {self.n_actions} power levels.")
        print(f"Levels: {self.action_levels}")

    def step(self, action_idx):
        """
        Convert Discrete Index (0..N) to Continuous Power (-1..1)
        """
        # DQN passes a scalar integer usually
        continuous_action = self.action_levels[action_idx]
        
        # Wrap as list/array because V5 expects array inputs for PPO usually?
        # V5 step(): `float(action[0])`. It expects list/array.
        # But wait, if I assume it expects list, I must pass list.
        # Let's verify V5 code.
        # V5: `req_phy_mw = float(action[0]) * self.max_power_mw`
        # So yes, it expects an iterable.
        
        # So we pass [continuous_action] to super().step()
        # BUT super().step() does the logic.
        # Wait, if I change the logic here, I can just copy the logic or call super with prepared action.
        
        # NOTE: V5 step() signature is step(self, action).
        # We can just pass the array `[continuous_action]`.
        
        # However, V5 action_space is Box. We overrode it to Discrete.
        # This is fine for the Agent check.
        
        return super().step(np.array([continuous_action]))
