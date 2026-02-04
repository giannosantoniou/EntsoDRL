"""
Decision Strategy Pattern - Decoupling Brain from Body.

This module defines the interface and implementations for decision-making.
The Controller doesn't know if it's talking to AI, Rules, or Random.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


# ============================================================================
# 1. THE CONTRACT (INTERFACE)
# ============================================================================

class IDecisionStrategy(ABC):
    """
    Interface for any 'brain' that makes decisions.
    Doesn't care if it's Neural Network, IF/ELSE, or Random.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""
        pass
    
    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        """Load model if needed. Path can be None for rule-based strategies."""
        pass

    @abstractmethod
    def predict_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Make a decision.
        
        Args:
            observation: What the agent sees (normalized features)
            action_mask: Boolean array of valid actions
            
        Returns:
            Action index (0 to n_actions-1)
        """
        pass


# ============================================================================
# 2. IMPLEMENTATION A: THE AI TRADER (PPO/MaskablePPO)
# ============================================================================

class RLAgentStrategy(IDecisionStrategy):
    """
    Strategy using trained RL model (PPO or MaskablePPO).
    """
    
    def __init__(self, use_maskable: bool = True):
        self.model = None
        self.use_maskable = use_maskable
        self._name = "MaskablePPO" if use_maskable else "PPO"
    
    @property
    def name(self) -> str:
        return self._name
    
    def load(self, path: Optional[str] = None) -> None:
        if path is None:
            raise ValueError("RLAgentStrategy requires a model path")
        
        print(f"🧠 Loading AI Model from {path}...")
        
        if self.use_maskable:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(path)
        else:
            from stable_baselines3 import PPO
            self.model = PPO.load(path)
        
        print(f"✅ {self._name} Model Loaded Successfully.")

    def predict_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if self.model is None:
            raise RuntimeError("Model not loaded! Call load() first.")
        
        # Ensure observation is 2D (batch dimension)
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        # MaskablePPO requires action_masks parameter
        if self.use_maskable:
            action, _states = self.model.predict(
                observation, 
                deterministic=True, 
                action_masks=action_mask.reshape(1, -1)
            )
        else:
            action, _states = self.model.predict(observation, deterministic=True)
        
        return int(action[0])


# ============================================================================
# 3. IMPLEMENTATION B: THE SAFEGUARD (RULE-BASED)
# ============================================================================

class RuleBasedStrategy(IDecisionStrategy):
    """
    Simple rule-based logic for fallback or safety.
    Example: If price > 200€ sell, if price < 50€ buy, else idle.
    """
    
    def __init__(
        self, 
        sell_threshold: float = 150.0,
        buy_threshold: float = 50.0,
        n_actions: int = 21
    ):
        self.sell_threshold = sell_threshold
        self.buy_threshold = buy_threshold
        self.n_actions = n_actions
        self.idle_action = n_actions // 2  # Middle = idle
    
    @property
    def name(self) -> str:
        return f"RuleBased(sell>{self.sell_threshold}, buy<{self.buy_threshold})"
    
    def load(self, path: Optional[str] = None) -> None:
        # No loading needed for rule-based
        print(f"📋 {self.name} initialized (no model to load).")

    def predict_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        # Price is at index 3 (normalized by 100)
        # Features: [soc, max_discharge, max_charge, price, dam_commitment, ...]
        price_normalized = observation[3] if observation.ndim == 1 else observation[0, 3]
        price = price_normalized * 100.0  # Denormalize
        
        # Determine action based on rules
        if price > self.sell_threshold:
            action = self.n_actions - 1  # Max discharge (sell)
        elif price < self.buy_threshold:
            action = 0  # Max charge (buy)
        else:
            action = self.idle_action  # Idle
        
        # Safety: Check if action is allowed by mask
        if not action_mask[action]:
            print(f"⚠️ Rule-based action {action} blocked by mask! Falling back to IDLE.")
            # Find closest valid action to idle
            if action_mask[self.idle_action]:
                return self.idle_action
            else:
                # Find any valid action
                valid_actions = np.where(action_mask)[0]
                return int(valid_actions[len(valid_actions) // 2])
        
        return action


# ============================================================================
# 4. IMPLEMENTATION C: THE DUMMY (FOR DEBUGGING/TESTING)
# ============================================================================

class RandomStrategy(IDecisionStrategy):
    """
    Randomly selects a VALID action.
    Useful for testing PLC connection without worrying about profit.
    """
    
    @property
    def name(self) -> str:
        return "Random"
    
    def load(self, path: Optional[str] = None) -> None:
        print("🎲 Random Strategy initialized (no model to load).")

    def predict_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            raise RuntimeError("No valid actions available!")
        return int(np.random.choice(valid_actions))


# ============================================================================
# 5. IMPLEMENTATION D: CONSERVATIVE (IDLE ONLY)
# ============================================================================

class ConservativeStrategy(IDecisionStrategy):
    """
    Always returns IDLE action.
    Useful for safety mode or when unsure.
    """
    
    def __init__(self, n_actions: int = 21):
        self.idle_action = n_actions // 2
    
    @property
    def name(self) -> str:
        return "Conservative(IDLE)"
    
    def load(self, path: Optional[str] = None) -> None:
        print("🛡️ Conservative Strategy initialized (always IDLE).")

    def predict_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        return self.idle_action


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_strategy(strategy_type: str, model_path: Optional[str] = None, **kwargs) -> IDecisionStrategy:
    """
    Factory function to create the appropriate strategy.
    
    Args:
        strategy_type: "AI", "RULE_BASED", "RANDOM", or "CONSERVATIVE"
        model_path: Path to model file (required for AI)
        **kwargs: Additional arguments for specific strategies
        
    Returns:
        IDecisionStrategy instance
    """
    strategy_type = strategy_type.upper()
    
    if strategy_type == "AI":
        strategy = RLAgentStrategy(use_maskable=kwargs.get('use_maskable', True))
        strategy.load(model_path)
        return strategy
    
    elif strategy_type == "RULE_BASED":
        strategy = RuleBasedStrategy(
            sell_threshold=kwargs.get('sell_threshold', 150.0),
            buy_threshold=kwargs.get('buy_threshold', 50.0),
            n_actions=kwargs.get('n_actions', 21)
        )
        strategy.load()
        return strategy
    
    elif strategy_type == "RANDOM":
        strategy = RandomStrategy()
        strategy.load()
        return strategy
    
    elif strategy_type == "CONSERVATIVE":
        strategy = ConservativeStrategy(n_actions=kwargs.get('n_actions', 21))
        strategy.load()
        return strategy
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
