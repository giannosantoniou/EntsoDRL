import gymnasium as gym
import numpy as np

class ActionMasker(gym.ActionWrapper):
    """
    Hard-coded safety layer to prevent physical violations.
    If SoC is full, preventing Charging.
    If SoC is empty, prevent Discharging.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def action(self, action):
        # We need access to current SoC from the environment
        # Assuming env.unwrapped gives access to 'current_soc'
        current_soc = self.env.unwrapped.current_soc
        
        # Check if action is scalar (1D) or vector (2D)
        # 1D Case: Old logic (Direct Physical Control)
        if action.shape == (1,):
            raw_action = float(action[0])
            safe_action = self._mask_physical(raw_action, current_soc)
            return np.array([safe_action], dtype=np.float32)
            
        # 2D Case: Multi-Market (DAM, Physical)
        # Action[0] = DAM Commitment (Financial - Unconstrained)
        # Action[1] = Physical Flow (Subject to Physics)
        elif action.shape == (2,):
            dam_action = float(action[0])
            phy_action = float(action[1])
            
            # Mask Physical Action
            safe_phy_section = self._mask_physical(phy_action, current_soc)
            
            # NO Financial Masking. Let the agent fail and learn.
            return np.array([dam_action, safe_phy_section], dtype=np.float32)
            
        else:
            # Fallback for unexpected shapes
            return action

    def _mask_physical(self, raw_action, current_soc):
        safe_action = raw_action
        
        # Hard Constraints (Updated for 5-95%)
        # Block Charging if SoC >= 0.95
        if current_soc >= 0.95 and raw_action < 0:
            safe_action = 0.0
            
        # Block Discharging if SoC <= 0.05
        if current_soc <= 0.05 and raw_action > 0:
            safe_action = 0.0
            
        return safe_action
