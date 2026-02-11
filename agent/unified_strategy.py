"""
Unified Multi-Market Decision Strategy

I implement the decision strategy for the unified multi-market model.
This wraps the trained MaskablePPO model and provides the IDecisionStrategy
interface for integration with the production system.

Key Features:
1. Handles MultiDiscrete action space [5, 5, 21]
2. Applies action masking for capacity cascading
3. Provides unified action interpretation
4. Supports VecNormalize for observation normalization
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

from agent.decision_strategy import IDecisionStrategy


class UnifiedStrategy(IDecisionStrategy):
    """
    Decision strategy for the unified multi-market model.

    I wrap the trained MaskablePPO model with MultiDiscrete action space
    and provide proper action interpretation for production use.
    """

    # I define the action space structure
    N_AFRR_LEVELS = 5      # 0%, 25%, 50%, 75%, 100%
    N_PRICE_TIERS = 5      # Aggressive to conservative
    N_ENERGY_ACTIONS = 21  # -30 to +30 MW

    # I define action interpretation
    AFRR_LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    ENERGY_LEVELS = np.linspace(-1.0, 1.0, N_ENERGY_ACTIONS)

    def __init__(
        self,
        max_power_mw: float = 30.0,
        n_obs_features: int = 58
    ):
        self.model = None
        self.vec_normalize = None
        self.max_power_mw = max_power_mw
        self.n_obs_features = n_obs_features
        self._name = "UnifiedMultiMarket"

    @property
    def name(self) -> str:
        return self._name

    def load(self, path: Optional[str] = None) -> None:
        """I load the trained unified model and normalizer."""
        if path is None:
            raise ValueError("UnifiedStrategy requires a model path")

        print(f"  Loading Unified Multi-Market Model from {path}...")

        # I load the MaskablePPO model
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(path)

        # I try to load the VecNormalize if it exists
        normalizer_path = Path(path).parent / "vec_normalize_unified.pkl"
        if normalizer_path.exists():
            import pickle
            with open(normalizer_path, 'rb') as f:
                self.vec_normalize = pickle.load(f)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            print(f"  Loaded VecNormalize from {normalizer_path}")
        else:
            print(f"  Warning: No VecNormalize found at {normalizer_path}")

        print(f"  {self._name} Model Loaded Successfully.")

    def predict_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray
    ) -> np.ndarray:
        """
        I predict the unified action given observation and mask.

        Args:
            observation: 58-feature observation vector
            action_mask: Flattened boolean mask (31 elements = 5 + 5 + 21)

        Returns:
            Action array [afrr_level, price_tier, energy_action]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded! Call load() first.")

        # I ensure observation is 2D
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # I normalize observation if VecNormalize is available
        if self.vec_normalize is not None:
            observation = self.vec_normalize.normalize_obs(observation)

        # I reshape action mask for MultiDiscrete format
        # MaskablePPO expects mask as list of arrays for MultiDiscrete
        afrr_mask = action_mask[:self.N_AFRR_LEVELS]
        price_mask = action_mask[self.N_AFRR_LEVELS:self.N_AFRR_LEVELS + self.N_PRICE_TIERS]
        energy_mask = action_mask[self.N_AFRR_LEVELS + self.N_PRICE_TIERS:]

        # I stack masks for batch dimension
        action_masks = np.array([
            np.concatenate([afrr_mask, price_mask, energy_mask])
        ])

        # I predict with action masking
        action, _states = self.model.predict(
            observation,
            deterministic=True,
            action_masks=action_masks
        )

        return action[0] if action.ndim > 1 else action

    def predict_with_interpretation(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        remaining_capacity_mw: float
    ) -> dict:
        """
        I predict action and provide full interpretation.

        Args:
            observation: 58-feature observation vector
            action_mask: Flattened boolean mask
            remaining_capacity_mw: Capacity after DAM commitment

        Returns:
            Dictionary with action indices and physical values
        """
        action = self.predict_action(observation, action_mask)

        # I unpack the action
        afrr_action = int(action[0])
        price_action = int(action[1])
        energy_action = int(action[2])

        # I interpret the action
        afrr_level = self.AFRR_LEVELS[afrr_action]
        afrr_commitment_mw = afrr_level * remaining_capacity_mw

        price_tier = self.PRICE_TIERS[price_action]

        energy_level = self.ENERGY_LEVELS[energy_action]
        energy_power_mw = energy_level * remaining_capacity_mw

        return {
            # Action indices
            'afrr_action': afrr_action,
            'price_action': price_action,
            'energy_action': energy_action,

            # Physical values
            'afrr_level': afrr_level,
            'afrr_commitment_mw': afrr_commitment_mw,
            'price_tier': price_tier,
            'energy_level': energy_level,
            'energy_power_mw': energy_power_mw,
            'remaining_capacity_mw': remaining_capacity_mw
        }


class UnifiedRuleBasedStrategy(IDecisionStrategy):
    """
    Rule-based fallback strategy for unified multi-market trading.

    I provide a deterministic strategy that can be used when the AI model
    is not available or during safety mode.
    """

    # I use the same action structure as UnifiedStrategy
    N_AFRR_LEVELS = 5
    N_PRICE_TIERS = 5
    N_ENERGY_ACTIONS = 21

    AFRR_LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    ENERGY_LEVELS = np.linspace(-1.0, 1.0, N_ENERGY_ACTIONS)

    def __init__(
        self,
        max_power_mw: float = 30.0,
        high_price_threshold: float = 150.0,
        low_price_threshold: float = 50.0,
        high_cap_percentile: float = 0.7
    ):
        self.max_power_mw = max_power_mw
        self.high_price_threshold = high_price_threshold
        self.low_price_threshold = low_price_threshold
        self.high_cap_percentile = high_cap_percentile
        self._name = "UnifiedRuleBased"

    @property
    def name(self) -> str:
        return self._name

    def load(self, path: Optional[str] = None) -> None:
        """I don't need to load anything for rule-based strategy."""
        print(f"  {self.name} initialized (no model to load).")

    def predict_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray
    ) -> np.ndarray:
        """
        I decide based on rules extracted from observation features.

        Observation structure (key features):
        [0] soc
        [3] dam_price / 100
        [10-12] time encoding
        [26] afrr_cap_price
        [34] is_peak
        [35] is_solar
        """
        if observation.ndim > 1:
            observation = observation[0]

        # I extract key features
        soc = observation[0]
        dam_price = observation[3] * 100  # Denormalize
        hour_sin = observation[10]
        hour_cos = observation[11]
        afrr_cap_price_norm = observation[26] if len(observation) > 26 else 0.5
        is_peak = observation[34] if len(observation) > 34 else 0.0
        is_solar = observation[35] if len(observation) > 35 else 0.0

        # I decode hour
        import math
        angle = math.atan2(hour_sin, hour_cos)
        hour = (angle * 24) / (2 * math.pi)
        if hour < 0:
            hour += 24
        hour = int(round(hour)) % 24

        # I decide aFRR commitment level
        # High commitment when capacity prices are high
        if afrr_cap_price_norm > 0.7:
            afrr_action = 4  # 100% commitment
        elif afrr_cap_price_norm > 0.5:
            afrr_action = 2  # 50% commitment
        else:
            afrr_action = 1  # 25% commitment

        # I decide price tier
        # More aggressive during peak hours (higher selection chance)
        if is_peak > 0.5:
            price_action = 1  # Aggressive
        else:
            price_action = 2  # Neutral

        # I decide energy action
        # Similar to AggressiveHybridStrategy but for unified format
        if is_peak > 0.5 or dam_price > self.high_price_threshold:
            # Peak or high price: discharge
            if soc > 0.4:
                energy_action = 18  # ~80% discharge
            else:
                energy_action = 14  # ~40% discharge
        elif is_solar > 0.5 or dam_price < self.low_price_threshold:
            # Solar hours or low price: charge
            if soc < 0.6:
                energy_action = 2  # ~80% charge
            else:
                energy_action = 6  # ~40% charge
        else:
            # Off-peak: idle or minimal trading
            energy_action = 10  # Idle

        # I apply action mask
        afrr_mask = action_mask[:self.N_AFRR_LEVELS]
        price_mask = action_mask[self.N_AFRR_LEVELS:self.N_AFRR_LEVELS + self.N_PRICE_TIERS]
        energy_mask = action_mask[self.N_AFRR_LEVELS + self.N_PRICE_TIERS:]

        # I find valid actions
        if not afrr_mask[afrr_action]:
            valid_afrr = np.where(afrr_mask)[0]
            afrr_action = int(valid_afrr[len(valid_afrr) // 2]) if len(valid_afrr) > 0 else 0

        if not price_mask[price_action]:
            valid_price = np.where(price_mask)[0]
            price_action = int(valid_price[len(valid_price) // 2]) if len(valid_price) > 0 else 2

        if not energy_mask[energy_action]:
            valid_energy = np.where(energy_mask)[0]
            energy_action = int(valid_energy[len(valid_energy) // 2]) if len(valid_energy) > 0 else 10

        return np.array([afrr_action, price_action, energy_action])


class UnifiedConservativeStrategy(IDecisionStrategy):
    """
    Conservative fallback strategy that minimizes risk.

    I commit minimal aFRR capacity and stay mostly idle for energy trading.
    This is useful during system issues or when the market is uncertain.
    """

    N_AFRR_LEVELS = 5
    N_PRICE_TIERS = 5
    N_ENERGY_ACTIONS = 21

    def __init__(self):
        self._name = "UnifiedConservative"

    @property
    def name(self) -> str:
        return self._name

    def load(self, path: Optional[str] = None) -> None:
        print(f"  {self.name} initialized (always minimal action).")

    def predict_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray
    ) -> np.ndarray:
        """I return minimal action: low aFRR commitment, neutral price, idle."""
        # I parse masks
        afrr_mask = action_mask[:self.N_AFRR_LEVELS]
        price_mask = action_mask[self.N_PRICE_TIERS:self.N_AFRR_LEVELS + self.N_PRICE_TIERS]
        energy_mask = action_mask[self.N_AFRR_LEVELS + self.N_PRICE_TIERS:]

        # I prefer lowest valid aFRR commitment
        afrr_action = 0 if afrr_mask[0] else int(np.where(afrr_mask)[0][0])

        # I prefer neutral price tier
        price_action = 2 if price_mask[2] else int(np.where(price_mask)[0][len(np.where(price_mask)[0]) // 2])

        # I prefer idle
        idle_action = self.N_ENERGY_ACTIONS // 2
        energy_action = idle_action if energy_mask[idle_action] else int(np.where(energy_mask)[0][len(np.where(energy_mask)[0]) // 2])

        return np.array([afrr_action, price_action, energy_action])


def create_unified_strategy(
    strategy_type: str,
    model_path: Optional[str] = None,
    **kwargs
) -> IDecisionStrategy:
    """
    Factory function to create unified strategy instances.

    Args:
        strategy_type: "AI", "RULE_BASED", or "CONSERVATIVE"
        model_path: Path to model file (required for AI)
        **kwargs: Additional arguments for specific strategies

    Returns:
        IDecisionStrategy instance for unified multi-market trading
    """
    strategy_type = strategy_type.upper()

    if strategy_type in ("AI", "UNIFIED", "UNIFIED_AI"):
        strategy = UnifiedStrategy(
            max_power_mw=kwargs.get('max_power_mw', 30.0),
            n_obs_features=kwargs.get('n_obs_features', 58)
        )
        strategy.load(model_path)
        return strategy

    elif strategy_type in ("RULE_BASED", "UNIFIED_RULE_BASED"):
        strategy = UnifiedRuleBasedStrategy(
            max_power_mw=kwargs.get('max_power_mw', 30.0),
            high_price_threshold=kwargs.get('high_price_threshold', 150.0),
            low_price_threshold=kwargs.get('low_price_threshold', 50.0)
        )
        strategy.load()
        return strategy

    elif strategy_type in ("CONSERVATIVE", "UNIFIED_CONSERVATIVE"):
        strategy = UnifiedConservativeStrategy()
        strategy.load()
        return strategy

    else:
        raise ValueError(
            f"Unknown unified strategy type: {strategy_type}. "
            f"Valid options: AI, RULE_BASED, CONSERVATIVE"
        )
