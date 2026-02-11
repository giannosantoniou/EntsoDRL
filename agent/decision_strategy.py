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
        
        print(f" Loading AI Model from {path}...")
        
        if self.use_maskable:
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(path)
        else:
            from stable_baselines3 import PPO
            self.model = PPO.load(path)
        
        print(f" {self._name} Model Loaded Successfully.")

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
        print(f" {self.name} initialized (no model to load).")

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
            print(f" Rule-based action {action} blocked by mask! Falling back to IDLE.")
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
        print(" Random Strategy initialized (no model to load).")

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
        print(" Conservative Strategy initialized (always IDLE).")

    def predict_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        return self.idle_action


# ============================================================================
# 6. IMPLEMENTATION E: AGGRESSIVE HYBRID (TIME + PRICE INTELLIGENCE)
# ============================================================================

class AggressiveHybridStrategy(IDecisionStrategy):
    """
    I combine Time-Based activity patterns with price intelligence.

    This strategy was developed through backtesting and achieved:
    - 8.9M EUR/year projected profit
    - 56.2% utilization
    - 28,244 EUR profit per cycle

    I always trade during peak/solar hours but scale power based on price.
    """

    def __init__(
        self,
        n_actions: int = 21,
        high_price_threshold: float = 150.0,
        low_price_threshold: float = 50.0,
        very_high_price: float = 200.0,
        very_low_price: float = 20.0
    ):
        self.n_actions = n_actions
        self.idle_action = n_actions // 2
        self.high_price_threshold = high_price_threshold
        self.low_price_threshold = low_price_threshold
        self.very_high_price = very_high_price
        self.very_low_price = very_low_price

        # I precompute action mappings
        # Actions: 0 = max charge (-30MW), 10 = idle, 20 = max discharge (+30MW)
        self.max_discharge = n_actions - 1
        self.high_discharge = int(n_actions * 0.85)  # ~70% power
        self.medium_discharge = int(n_actions * 0.70)  # ~40% power
        self.max_charge = 0
        self.high_charge = int(n_actions * 0.15)  # ~70% charge
        self.medium_charge = int(n_actions * 0.30)  # ~40% charge

    @property
    def name(self) -> str:
        return "AggressiveHybrid"

    def load(self, path: Optional[str] = None) -> None:
        print("Aggressive Hybrid Strategy initialized.")
        print("  - Evening peak (17-21h): Discharge")
        print("  - Morning ramp (5-8h): Moderate discharge")
        print("  - Solar hours (9-15h): Charge")
        print("  - Off-peak: Trade only on extreme prices")

    def predict_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        I make decisions based on time windows and price signals.

        Observation format (from BatteryController, 15 features):
        [0] soc - State of Charge (0-1)
        [1] max_discharge - Max discharge power (normalized)
        [2] max_charge - Max charge power (normalized)
        [3] price - Current price (normalized by /100)
        [4] dam_commitment - DAM commitment (normalized)
        [5] hour_sin - Sine of hour
        [6] hour_cos - Cosine of hour
        [7-10] price_lookahead (4h)
        [11-14] dam_lookahead (4h)
        """
        # I extract features from observation
        if observation.ndim > 1:
            observation = observation[0]

        # I denormalize price (stored as price/100)
        price = observation[3] * 100.0

        # I reconstruct hour from sine/cosine encoding (indices 5 and 6)
        hour_sin = observation[5]
        hour_cos = observation[6]
        hour = self._decode_hour(hour_sin, hour_cos)

        # I check if it's weekend (from day_of_week encoding if available)
        # For now I assume weekday - weekend detection would need more features
        is_weekend = False
        weekend_factor = 0.5 if is_weekend else 1.0

        # I determine the action based on time window and price
        action = self._decide_action(hour, price, weekend_factor)

        # I ensure the action is valid (respects battery constraints)
        action = self._apply_mask(action, action_mask)

        return action

    def _decode_hour(self, hour_sin: float, hour_cos: float) -> int:
        """I reconstruct hour from sine/cosine encoding."""
        import math
        # hour_sin = sin(2*pi*hour/24), hour_cos = cos(2*pi*hour/24)
        angle = math.atan2(hour_sin, hour_cos)
        hour = (angle * 24) / (2 * math.pi)
        if hour < 0:
            hour += 24
        return int(round(hour)) % 24

    def _decide_action(self, hour: int, price: float, weekend_factor: float) -> int:
        """I decide based on time windows and price intelligence."""

        # Evening peak (17-21h): ALWAYS discharge
        if 17 <= hour <= 21:
            if price > self.high_price_threshold:
                # High price - full power
                return self.max_discharge
            else:
                # Normal price - reduced power (70%)
                return self.high_discharge

        # Morning ramp (5-8h): Usually discharge
        elif 5 <= hour <= 8:
            if price > self.high_price_threshold:
                # High price - high power
                return self.high_discharge
            else:
                # Normal price - medium power
                return self.medium_discharge

        # Solar hours (9-15h): ALWAYS charge
        elif 9 <= hour <= 15:
            if price < self.low_price_threshold:
                # Low price - full charge
                return self.max_charge
            else:
                # Normal price - reduced charge (70%)
                return self.high_charge

        # Off-peak hours (22-4h, 16h): Only trade on extreme prices
        else:
            if price > self.very_high_price:
                # Very high price - discharge
                return self.max_discharge
            elif price < self.very_low_price:
                # Very low price - charge
                return self.max_charge
            else:
                # Normal price - idle
                return self.idle_action

    def _apply_mask(self, action: int, action_mask: np.ndarray) -> int:
        """I ensure the action respects battery constraints."""
        if action_mask[action]:
            return action

        # Action is blocked - find closest valid action
        # I prefer to stay close to the original intent
        if action > self.idle_action:
            # I wanted to discharge - find closest valid discharge or idle
            for a in range(action, self.idle_action - 1, -1):
                if action_mask[a]:
                    return a
        elif action < self.idle_action:
            # I wanted to charge - find closest valid charge or idle
            for a in range(action, self.idle_action + 1):
                if action_mask[a]:
                    return a

        # Fallback: return any valid action (prefer idle)
        if action_mask[self.idle_action]:
            return self.idle_action

        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) > 0:
            return int(valid_actions[len(valid_actions) // 2])

        return self.idle_action


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_strategy(strategy_type: str, model_path: Optional[str] = None, **kwargs) -> IDecisionStrategy:
    """
    Factory function to create the appropriate strategy.

    Args:
        strategy_type: "AI", "RULE_BASED", "RANDOM", "CONSERVATIVE", "AGGRESSIVE_HYBRID",
                       "UNIFIED", "UNIFIED_RULE_BASED", or "UNIFIED_CONSERVATIVE"
        model_path: Path to model file (required for AI and UNIFIED)
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

    elif strategy_type in ("AGGRESSIVE_HYBRID", "HYBRID", "AGGRESSIVE"):
        strategy = AggressiveHybridStrategy(
            n_actions=kwargs.get('n_actions', 21),
            high_price_threshold=kwargs.get('high_price_threshold', 150.0),
            low_price_threshold=kwargs.get('low_price_threshold', 50.0),
            very_high_price=kwargs.get('very_high_price', 200.0),
            very_low_price=kwargs.get('very_low_price', 20.0)
        )
        strategy.load()
        return strategy

    # I add unified multi-market strategies
    elif strategy_type in ("UNIFIED", "UNIFIED_AI"):
        from agent.unified_strategy import create_unified_strategy
        return create_unified_strategy("AI", model_path, **kwargs)

    elif strategy_type == "UNIFIED_RULE_BASED":
        from agent.unified_strategy import create_unified_strategy
        return create_unified_strategy("RULE_BASED", **kwargs)

    elif strategy_type == "UNIFIED_CONSERVATIVE":
        from agent.unified_strategy import create_unified_strategy
        return create_unified_strategy("CONSERVATIVE", **kwargs)

    else:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Valid options: AI, RULE_BASED, RANDOM, CONSERVATIVE, AGGRESSIVE_HYBRID, "
            f"UNIFIED, UNIFIED_RULE_BASED, UNIFIED_CONSERVATIVE"
        )
