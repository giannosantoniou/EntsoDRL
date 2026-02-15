"""
Unified Multi-Market Decision Strategy

I implement the decision strategy for the unified multi-market model.
This wraps the trained MaskablePPO model and provides the IDecisionStrategy
interface for integration with the production system.

Key Features:
1. Handles MultiDiscrete action space [5, 5, 11, 11]
2. Applies action masking for capacity cascading: DAM -> aFRR -> IntraDay -> mFRR
3. Provides unified action interpretation
4. Supports VecNormalize for observation normalization
5. Resolution-independent: works with both 15-min and 1-hour timesteps.
   Observation indices (63 features) stay the same regardless of resolution —
   DAM/price lookahead samples at hourly intervals via _sph parameter.
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
    N_AFRR_LEVELS = 5       # 0%, 25%, 50%, 75%, 100%
    N_PRICE_TIERS = 5       # Aggressive to conservative
    N_INTRADAY_ACTIONS = 11  # -1.0 to +1.0 (XBID/IntraDay)
    N_MFRR_ACTIONS = 11     # -1.0 to +1.0 (Balancing/mFRR)
    N_FREEBID_PRICE_TIERS = 5  # Free Bid price tiers

    # I define action interpretation
    AFRR_LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    INTRADAY_LEVELS = np.linspace(-1.0, 1.0, N_INTRADAY_ACTIONS)
    MFRR_LEVELS = np.linspace(-1.0, 1.0, N_MFRR_ACTIONS)
    FREEBID_PRICE_TIERS = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

    def __init__(
        self,
        max_power_mw: float = 30.0,
        n_obs_features: int = 63,
        enable_full_market: bool = False
    ):
        self.model = None
        self.vec_normalize = None
        self.max_power_mw = max_power_mw
        self.enable_full_market = enable_full_market
        self.n_obs_features = 75 if enable_full_market else n_obs_features
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
            observation: 63-feature observation vector
            action_mask: Flattened boolean mask (32 elements = 5 + 5 + 11 + 11)

        Returns:
            Action array [afrr_level, price_tier, intraday_action, mfrr_action]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded! Call load() first.")

        # I ensure observation is 2D
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # I normalize observation if VecNormalize is available
        if self.vec_normalize is not None:
            observation = self.vec_normalize.normalize_obs(observation)

        # I parse the flattened mask into sub-masks
        offset = 0
        afrr_mask = action_mask[offset:offset + self.N_AFRR_LEVELS]; offset += self.N_AFRR_LEVELS
        price_mask = action_mask[offset:offset + self.N_PRICE_TIERS]; offset += self.N_PRICE_TIERS
        intraday_mask = action_mask[offset:offset + self.N_INTRADAY_ACTIONS]; offset += self.N_INTRADAY_ACTIONS
        mfrr_mask = action_mask[offset:offset + self.N_MFRR_ACTIONS]; offset += self.N_MFRR_ACTIONS

        if self.enable_full_market and offset < len(action_mask):
            freebid_mask = action_mask[offset:offset + self.N_FREEBID_PRICE_TIERS]
            combined_mask = np.concatenate([afrr_mask, price_mask, intraday_mask, mfrr_mask, freebid_mask])
        else:
            combined_mask = np.concatenate([afrr_mask, price_mask, intraday_mask, mfrr_mask])

        # I stack masks for batch dimension
        action_masks = np.array([combined_mask])

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
            observation: 63-feature observation vector
            action_mask: Flattened boolean mask
            remaining_capacity_mw: Capacity after DAM commitment

        Returns:
            Dictionary with action indices and physical values
        """
        action = self.predict_action(observation, action_mask)

        # I unpack the action
        afrr_action = int(action[0])
        price_action = int(action[1])
        intraday_action = int(action[2])
        mfrr_action = int(action[3])

        # I interpret the action
        afrr_level = self.AFRR_LEVELS[afrr_action]
        afrr_commitment_mw = afrr_level * remaining_capacity_mw

        price_tier = self.PRICE_TIERS[price_action]

        intraday_level = self.INTRADAY_LEVELS[intraday_action]
        intraday_power_mw = intraday_level * remaining_capacity_mw

        mfrr_level = self.MFRR_LEVELS[mfrr_action]
        mfrr_power_mw = mfrr_level * remaining_capacity_mw

        result = {
            # Action indices
            'afrr_action': afrr_action,
            'price_action': price_action,
            'intraday_action': intraday_action,
            'mfrr_action': mfrr_action,

            # Physical values
            'afrr_level': afrr_level,
            'afrr_commitment_mw': afrr_commitment_mw,
            'price_tier': price_tier,
            'intraday_level': intraday_level,
            'intraday_power_mw': intraday_power_mw,
            'mfrr_level': mfrr_level,
            'mfrr_power_mw': mfrr_power_mw,
            'remaining_capacity_mw': remaining_capacity_mw
        }

        if self.enable_full_market and len(action) > 4:
            freebid_action = int(action[4])
            result['freebid_action'] = freebid_action
            result['balancing_price_tier'] = self.FREEBID_PRICE_TIERS[freebid_action]

        return result


class UnifiedRuleBasedStrategy(IDecisionStrategy):
    """
    Rule-based fallback strategy for unified multi-market trading.

    I provide a deterministic strategy that can be used when the AI model
    is not available or during safety mode. IntraDay is used for arbitrage
    (buy solar/sell peak), mFRR for spread capture.
    """

    # I use the same action structure as UnifiedStrategy
    N_AFRR_LEVELS = 5
    N_PRICE_TIERS = 5
    N_INTRADAY_ACTIONS = 11
    N_MFRR_ACTIONS = 11
    N_FREEBID_PRICE_TIERS = 5

    AFRR_LEVELS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    PRICE_TIERS = np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    INTRADAY_LEVELS = np.linspace(-1.0, 1.0, N_INTRADAY_ACTIONS)
    MFRR_LEVELS = np.linspace(-1.0, 1.0, N_MFRR_ACTIONS)
    FREEBID_PRICE_TIERS = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

    def __init__(
        self,
        max_power_mw: float = 30.0,
        high_price_threshold: float = 150.0,
        low_price_threshold: float = 50.0,
        high_cap_percentile: float = 0.7,
        enable_full_market: bool = False
    ):
        self.max_power_mw = max_power_mw
        self.high_price_threshold = high_price_threshold
        self.low_price_threshold = low_price_threshold
        self.high_cap_percentile = high_cap_percentile
        self.enable_full_market = enable_full_market
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

        Observation structure (key features for 63-feature obs):
        [0] soc
        [3] dam_price / 100
        [4] intraday_bid / 100
        [5] intraday_ask / 100
        [14-17] time encoding (sin/cos)
        [43] afrr_cap_price
        [55] ida3_correction (1.0 if hour >= 12, IDA3 prices available)
        [56] system_stress (|imbalance|/300, capped at 1.0)
        [57] res_penetration (res_total/load, normalized 0-1)
        [58] mfrr_direction (+1 UP, -1 DOWN, 0 closed)
        [60] is_peak
        [61] is_solar
        """
        if observation.ndim > 1:
            observation = observation[0]

        # I extract key features (63-feature observation layout)
        soc = observation[0]
        dam_price = observation[3] * 100  # Denormalize
        id_bid = observation[4] * 100 if len(observation) > 4 else dam_price - 2
        id_ask = observation[5] * 100 if len(observation) > 5 else dam_price + 2
        hour_sin = observation[14] if len(observation) > 14 else 0.0
        hour_cos = observation[15] if len(observation) > 15 else 1.0
        afrr_cap_price_norm = observation[43] if len(observation) > 43 else 0.5
        is_peak = observation[60] if len(observation) > 60 else 0.0
        is_solar = observation[61] if len(observation) > 61 else 0.0

        # I decode hour
        import math
        angle = math.atan2(hour_sin, hour_cos)
        hour = (angle * 24) / (2 * math.pi)
        if hour < 0:
            hour += 24
        hour = int(round(hour)) % 24

        # I decide aFRR commitment level
        if afrr_cap_price_norm > 0.7:
            afrr_action = 4  # 100% commitment
        elif afrr_cap_price_norm > 0.5:
            afrr_action = 2  # 50% commitment
        else:
            afrr_action = 1  # 25% commitment

        # I decide price tier
        if is_peak > 0.5:
            price_action = 1  # Aggressive
        else:
            price_action = 2  # Neutral

        # I decide IntraDay action — buy during solar, sell during peak
        id_idle = self.N_INTRADAY_ACTIONS // 2  # 5
        if is_solar > 0.5 and dam_price < self.low_price_threshold:
            # Solar hours + low price: buy on IntraDay
            if soc < 0.6:
                id_action = 1  # ~80% charge
            else:
                id_action = 3  # ~40% charge
        elif is_peak > 0.5 or dam_price > self.high_price_threshold:
            # Peak or high price: sell on IntraDay
            if soc > 0.4:
                id_action = 9  # ~80% discharge
            else:
                id_action = 7  # ~40% discharge
        else:
            id_action = id_idle  # Idle

        # I decide mFRR action — constrained by TSO direction signal
        # observation[58] encodes mFRR direction: +1 (UP/discharge), -1 (DOWN/charge), 0 (closed)
        mfrr_idle = self.N_MFRR_ACTIONS // 2  # 5
        mfrr_direction = observation[58] if len(observation) > 58 else 0.0
        mfrr_up = observation[8] * 100 if len(observation) > 8 else 120.0
        mfrr_down = observation[9] * 100 if len(observation) > 9 else 60.0

        if mfrr_direction > 0.5 and soc > 0.3:
            # UP available (deficit) — I can discharge/sell
            if mfrr_up > dam_price * 1.2:
                mfrr_action = 8  # Moderate discharge
            else:
                mfrr_action = mfrr_idle
        elif mfrr_direction < -0.5 and soc < 0.7:
            # DOWN available (surplus) — I can charge/buy (usually loss, be cautious)
            if mfrr_down < dam_price * 0.5:
                mfrr_action = 2  # Moderate charge at very low price
            else:
                mfrr_action = mfrr_idle  # Not worth it
        else:
            mfrr_action = mfrr_idle  # Closed or uncertain

        # I apply action masks
        offset = 0
        afrr_mask = action_mask[offset:offset + self.N_AFRR_LEVELS]; offset += self.N_AFRR_LEVELS
        price_mask = action_mask[offset:offset + self.N_PRICE_TIERS]; offset += self.N_PRICE_TIERS
        id_mask = action_mask[offset:offset + self.N_INTRADAY_ACTIONS]; offset += self.N_INTRADAY_ACTIONS
        mfrr_mask = action_mask[offset:offset + self.N_MFRR_ACTIONS]

        # I find valid actions (fallback to valid mid-point)
        if not afrr_mask[afrr_action]:
            valid = np.where(afrr_mask)[0]
            afrr_action = int(valid[len(valid) // 2]) if len(valid) > 0 else 0

        if not price_mask[price_action]:
            valid = np.where(price_mask)[0]
            price_action = int(valid[len(valid) // 2]) if len(valid) > 0 else 2

        if not id_mask[id_action]:
            valid = np.where(id_mask)[0]
            id_action = int(valid[len(valid) // 2]) if len(valid) > 0 else id_idle

        if not mfrr_mask[mfrr_action]:
            valid = np.where(mfrr_mask)[0]
            mfrr_action = int(valid[len(valid) // 2]) if len(valid) > 0 else mfrr_idle

        if self.enable_full_market:
            # I default to neutral price tier (1.0x = market price)
            freebid_action = 2
            if offset < len(action_mask):
                freebid_mask = action_mask[offset:offset + self.N_FREEBID_PRICE_TIERS]
                if not freebid_mask[freebid_action]:
                    valid = np.where(freebid_mask)[0]
                    freebid_action = int(valid[len(valid) // 2]) if len(valid) > 0 else 2
            return np.array([afrr_action, price_action, id_action, mfrr_action, freebid_action])

        return np.array([afrr_action, price_action, id_action, mfrr_action])


class UnifiedConservativeStrategy(IDecisionStrategy):
    """
    Conservative fallback strategy that minimizes risk.

    I commit minimal aFRR capacity and stay idle for both IntraDay and mFRR.
    This is useful during system issues or when the market is uncertain.
    """

    N_AFRR_LEVELS = 5
    N_PRICE_TIERS = 5
    N_INTRADAY_ACTIONS = 11
    N_MFRR_ACTIONS = 11
    N_FREEBID_PRICE_TIERS = 5

    def __init__(self, enable_full_market: bool = False):
        self.enable_full_market = enable_full_market
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
        """I return minimal action: [0 aFRR, neutral price, idle IntraDay, idle mFRR, (neutral price)]."""
        # I parse masks
        offset = 0
        afrr_mask = action_mask[offset:offset + self.N_AFRR_LEVELS]; offset += self.N_AFRR_LEVELS
        price_mask = action_mask[offset:offset + self.N_PRICE_TIERS]; offset += self.N_PRICE_TIERS
        id_mask = action_mask[offset:offset + self.N_INTRADAY_ACTIONS]; offset += self.N_INTRADAY_ACTIONS
        mfrr_mask = action_mask[offset:offset + self.N_MFRR_ACTIONS]; offset += self.N_MFRR_ACTIONS

        # I prefer lowest valid aFRR commitment
        afrr_action = 0 if afrr_mask[0] else int(np.where(afrr_mask)[0][0])

        # I prefer neutral price tier
        price_action = 2 if price_mask[2] else int(np.where(price_mask)[0][len(np.where(price_mask)[0]) // 2])

        # I prefer idle for IntraDay (center = 5)
        id_idle = self.N_INTRADAY_ACTIONS // 2
        id_action = id_idle if id_mask[id_idle] else int(np.where(id_mask)[0][len(np.where(id_mask)[0]) // 2])

        # I prefer idle for mFRR (center = 5)
        mfrr_idle = self.N_MFRR_ACTIONS // 2
        mfrr_action = mfrr_idle if mfrr_mask[mfrr_idle] else int(np.where(mfrr_mask)[0][len(np.where(mfrr_mask)[0]) // 2])

        if self.enable_full_market:
            # I default to neutral Free Bid price tier (1.0x)
            freebid_action = 2
            if offset < len(action_mask):
                freebid_mask = action_mask[offset:offset + self.N_FREEBID_PRICE_TIERS]
                if not freebid_mask[freebid_action]:
                    valid = np.where(freebid_mask)[0]
                    freebid_action = int(valid[len(valid) // 2]) if len(valid) > 0 else 2
            return np.array([afrr_action, price_action, id_action, mfrr_action, freebid_action])

        return np.array([afrr_action, price_action, id_action, mfrr_action])


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

    enable_full_market = kwargs.get('enable_full_market', False)

    if strategy_type in ("AI", "UNIFIED", "UNIFIED_AI"):
        strategy = UnifiedStrategy(
            max_power_mw=kwargs.get('max_power_mw', 30.0),
            n_obs_features=kwargs.get('n_obs_features', 63),
            enable_full_market=enable_full_market
        )
        strategy.load(model_path)
        return strategy

    elif strategy_type in ("RULE_BASED", "UNIFIED_RULE_BASED"):
        strategy = UnifiedRuleBasedStrategy(
            max_power_mw=kwargs.get('max_power_mw', 30.0),
            high_price_threshold=kwargs.get('high_price_threshold', 150.0),
            low_price_threshold=kwargs.get('low_price_threshold', 50.0),
            enable_full_market=enable_full_market
        )
        strategy.load()
        return strategy

    elif strategy_type in ("CONSERVATIVE", "UNIFIED_CONSERVATIVE"):
        strategy = UnifiedConservativeStrategy(enable_full_market=enable_full_market)
        strategy.load()
        return strategy

    else:
        raise ValueError(
            f"Unknown unified strategy type: {strategy_type}. "
            f"Valid options: AI, RULE_BASED, CONSERVATIVE"
        )
