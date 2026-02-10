"""
Balancing Trader - aFRR + mFRR Trading System

I handle real-time balancing market participation:
- aFRR: Capacity bidding (reserve provision)
- mFRR: Energy trading based on activation signals

I use the trained DRL model for mFRR trading decisions,
while considering DAM commitments and available capacity.

The DRL model was trained on mFRR price data and learns to:
- Buy energy when prices are low (charge)
- Sell energy when prices are high (discharge)
- Respect battery constraints via action masking
"""

import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from .interfaces import IBalancingTrader, BatteryState


@dataclass
class BalancingDecision:
    """Decision for balancing market participation."""
    timestamp: datetime

    # aFRR (capacity)
    afrr_up_capacity_mw: float = 0.0    # MW reserved for up-regulation
    afrr_down_capacity_mw: float = 0.0  # MW reserved for down-regulation
    afrr_bid_price: float = 0.0         # Capacity bid price (EUR/MW/h)

    # mFRR (energy)
    mfrr_action_mw: float = 0.0         # MW to trade (positive=sell, negative=buy)

    # Combined
    total_power_mw: float = 0.0         # Net power after all markets

    # Debug info
    strategy_used: str = ""
    observation_used: Optional[np.ndarray] = None


class BalancingTrader(IBalancingTrader):
    """Balancing Market Trader.

    I manage participation in aFRR and mFRR markets using:
    - DRL model for mFRR energy trading
    - Rule-based strategy for aFRR capacity bidding

    The trader respects DAM commitments - it only uses
    remaining capacity for balancing markets.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        strategy_type: str = "AI",  # AI, RULE_BASED, AGGRESSIVE_HYBRID
        n_actions: int = 21,
        max_power_mw: float = 30.0,
        capacity_mwh: float = 146.0,
        # aFRR settings
        afrr_enabled: bool = True,
        afrr_reserve_fraction: float = 0.3,  # Reserve 30% for aFRR
        afrr_min_price: float = 5.0,         # Minimum capacity price EUR/MW/h
        # mFRR settings
        mfrr_enabled: bool = True,
        mfrr_price_scale: float = 100.0,     # Price normalization
    ):
        """Initialize the balancing trader.

        Args:
            model_path: Path to trained DRL model (.zip)
            strategy_type: Strategy to use (AI, RULE_BASED, AGGRESSIVE_HYBRID)
            n_actions: Number of discrete actions
            max_power_mw: Maximum battery power
            capacity_mwh: Battery capacity
            afrr_enabled: Whether to participate in aFRR
            afrr_reserve_fraction: Fraction of capacity to reserve for aFRR
            afrr_min_price: Minimum aFRR capacity bid price
            mfrr_enabled: Whether to participate in mFRR
            mfrr_price_scale: Scale for price normalization
        """
        self.model_path = model_path
        self.strategy_type = strategy_type
        self.n_actions = n_actions
        self.max_power_mw = max_power_mw
        self.capacity_mwh = capacity_mwh

        # aFRR settings
        self.afrr_enabled = afrr_enabled
        self.afrr_reserve_fraction = afrr_reserve_fraction
        self.afrr_min_price = afrr_min_price

        # mFRR settings
        self.mfrr_enabled = mfrr_enabled
        self.mfrr_price_scale = mfrr_price_scale

        # Action mapping: action_idx -> power_fraction (-1 to +1)
        self.action_levels = np.linspace(-1.0, 1.0, n_actions)
        self.idle_action = n_actions // 2

        # Strategy (loaded lazily)
        self._strategy = None

        # VecNormalize for observation normalization (if using AI)
        self._vec_normalize = None

        print(f"Balancing Trader initialized")
        print(f"  Strategy: {strategy_type}")
        print(f"  aFRR: {'enabled' if afrr_enabled else 'disabled'}")
        print(f"  mFRR: {'enabled' if mfrr_enabled else 'disabled'}")

    def _load_strategy(self):
        """Lazy load the decision strategy."""
        if self._strategy is not None:
            return

        # I import here to avoid circular dependencies
        from agent.decision_strategy import create_strategy

        self._strategy = create_strategy(
            strategy_type=self.strategy_type,
            model_path=self.model_path,
            n_actions=self.n_actions,
        )

        # Try to load VecNormalize if it exists
        if self.model_path and self.strategy_type == "AI":
            vec_norm_path = self.model_path.replace('.zip', '_vec_normalize.pkl')
            if Path(vec_norm_path).exists():
                try:
                    from stable_baselines3.common.vec_env import VecNormalize
                    import pickle
                    with open(vec_norm_path, 'rb') as f:
                        self._vec_normalize = pickle.load(f)
                    self._vec_normalize.training = False
                    self._vec_normalize.norm_reward = False
                    print(f"  Loaded VecNormalize from {vec_norm_path}")
                except Exception as e:
                    print(f"  Warning: Could not load VecNormalize: {e}")

    def decide_action(
        self,
        battery_state: BatteryState,
        available_capacity_mw: float,
        market_data: Dict[str, Any],
    ) -> float:
        """Decide balancing market action.

        I return the total power to add to the DAM commitment.
        Positive = discharge/sell, Negative = charge/buy.

        Args:
            battery_state: Current battery state (SoC, etc.)
            available_capacity_mw: Capacity remaining after DAM commitment
            market_data: Real-time market data (prices, signals)

        Returns:
            Power in MW to trade in balancing markets
        """
        self._load_strategy()

        # If no capacity available, return 0
        if available_capacity_mw < 0.5:
            return 0.0

        # Allocate capacity between aFRR and mFRR
        if self.afrr_enabled:
            afrr_capacity = available_capacity_mw * self.afrr_reserve_fraction
            mfrr_capacity = available_capacity_mw - afrr_capacity
        else:
            afrr_capacity = 0.0
            mfrr_capacity = available_capacity_mw

        # mFRR decision
        mfrr_action = 0.0
        if self.mfrr_enabled and mfrr_capacity > 0.5:
            mfrr_action = self._decide_mfrr(
                battery_state=battery_state,
                available_mw=mfrr_capacity,
                market_data=market_data,
            )

        return mfrr_action

    def decide_full(
        self,
        battery_state: BatteryState,
        available_capacity_mw: float,
        market_data: Dict[str, Any],
        dam_commitment_mw: float = 0.0,
    ) -> BalancingDecision:
        """Full balancing decision with aFRR + mFRR details.

        Args:
            battery_state: Current battery state
            available_capacity_mw: Capacity after DAM
            market_data: Real-time market data
            dam_commitment_mw: Current DAM commitment (for context)

        Returns:
            BalancingDecision with full details
        """
        self._load_strategy()

        timestamp = battery_state.timestamp

        # Initialize decision
        decision = BalancingDecision(
            timestamp=timestamp,
            strategy_used=self._strategy.name if self._strategy else "None",
        )

        if available_capacity_mw < 0.5:
            return decision

        # aFRR capacity bidding
        if self.afrr_enabled:
            afrr_up, afrr_down, afrr_price = self._decide_afrr(
                battery_state=battery_state,
                available_mw=available_capacity_mw,
                market_data=market_data,
            )
            decision.afrr_up_capacity_mw = afrr_up
            decision.afrr_down_capacity_mw = afrr_down
            decision.afrr_bid_price = afrr_price

            # Reduce available capacity for mFRR
            afrr_reserved = max(afrr_up, afrr_down)
            mfrr_capacity = available_capacity_mw - afrr_reserved
        else:
            mfrr_capacity = available_capacity_mw

        # mFRR energy trading
        if self.mfrr_enabled and mfrr_capacity > 0.5:
            decision.mfrr_action_mw = self._decide_mfrr(
                battery_state=battery_state,
                available_mw=mfrr_capacity,
                market_data=market_data,
            )

        # Calculate total power
        decision.total_power_mw = decision.mfrr_action_mw

        return decision

    def _decide_afrr(
        self,
        battery_state: BatteryState,
        available_mw: float,
        market_data: Dict[str, Any],
    ) -> tuple:
        """Decide aFRR capacity bid.

        I use a simple rule-based strategy for aFRR:
        - Bid symmetric capacity (same up and down)
        - Price based on current market conditions

        Returns:
            (up_capacity_mw, down_capacity_mw, bid_price_eur)
        """
        # Reserve fraction of available capacity
        reserve_mw = available_mw * self.afrr_reserve_fraction

        # Check SoC constraints
        soc = battery_state.soc

        # I can only provide up-regulation if I have energy to discharge
        if soc < 0.15:
            up_capacity = 0.0
        else:
            up_capacity = reserve_mw

        # I can only provide down-regulation if I have room to charge
        if soc > 0.85:
            down_capacity = 0.0
        else:
            down_capacity = reserve_mw

        # Calculate bid price based on market conditions
        afrr_up_price = market_data.get('afrr_up_price', 10.0)
        afrr_down_price = market_data.get('afrr_down_price', 10.0)

        # I bid slightly below market price to ensure acceptance
        bid_price = min(afrr_up_price, afrr_down_price) * 0.95
        bid_price = max(bid_price, self.afrr_min_price)

        return up_capacity, down_capacity, bid_price

    def _decide_mfrr(
        self,
        battery_state: BatteryState,
        available_mw: float,
        market_data: Dict[str, Any],
    ) -> float:
        """Decide mFRR trading action using DRL model.

        I build the observation and use the trained strategy
        to decide how much energy to trade.

        Returns:
            Power in MW (positive=discharge/sell, negative=charge/buy)
        """
        # Build observation for the DRL model
        obs = self._build_observation(battery_state, market_data)

        # Build action mask based on SoC
        action_mask = self._build_action_mask(battery_state, available_mw)

        # Normalize observation if VecNormalize is available
        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        # Get action from strategy
        action_idx = self._strategy.predict_action(obs, action_mask)

        # Convert action index to power
        power_fraction = self.action_levels[action_idx]
        power_mw = power_fraction * available_mw

        return power_mw

    def _build_observation(
        self,
        battery_state: BatteryState,
        market_data: Dict[str, Any],
    ) -> np.ndarray:
        """Build observation for the DRL model.

        I create a simplified observation matching what the
        model was trained on. The full 44-feature observation
        would need more market data.
        """
        # Extract values with defaults
        soc = battery_state.soc
        price = market_data.get('mfrr_price_up', 100.0)
        price_down = market_data.get('mfrr_price_down', 80.0)

        # Time features (from timestamp)
        hour = battery_state.timestamp.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        dow = battery_state.timestamp.weekday()
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # Normalized features
        obs = np.array([
            soc,                                    # 0: SoC
            1.0,                                    # 1: max_discharge (normalized)
            1.0,                                    # 2: max_charge (normalized)
            price / self.mfrr_price_scale,          # 3: price (normalized)
            0.0,                                    # 4: dam_commitment
            hour_sin,                               # 5: hour_sin
            hour_cos,                               # 6: hour_cos
            # Price lookahead (4 values)
            price / self.mfrr_price_scale,          # 7-10: price lookahead
            price / self.mfrr_price_scale,
            price / self.mfrr_price_scale,
            price / self.mfrr_price_scale,
            # DAM lookahead (4 values)
            0.0, 0.0, 0.0, 0.0,                     # 11-14: dam lookahead
            # Extended features
            dow_sin,                                # 15: dow_sin
            dow_cos,                                # 16: dow_cos
            (price - price_down) / self.mfrr_price_scale,  # 17: spread
            0.0,                                    # 18: imbalance
            0.0,                                    # 19: volatility
        ], dtype=np.float32)

        return obs

    def _build_action_mask(
        self,
        battery_state: BatteryState,
        available_mw: float,
    ) -> np.ndarray:
        """Build action mask based on SoC constraints.

        I mask out actions that would violate SoC limits.
        """
        soc = battery_state.soc
        min_soc = 0.05
        max_soc = 0.95

        # Create mask (all True initially)
        mask = np.ones(self.n_actions, dtype=bool)

        # Calculate max charge/discharge in this interval (15 min = 0.25h)
        energy_per_step = available_mw * 0.25  # MWh

        # Max discharge before hitting min_soc
        max_discharge_energy = (soc - min_soc) * self.capacity_mwh
        max_discharge_fraction = max_discharge_energy / energy_per_step if energy_per_step > 0 else 0

        # Max charge before hitting max_soc
        max_charge_energy = (max_soc - soc) * self.capacity_mwh
        max_charge_fraction = max_charge_energy / energy_per_step if energy_per_step > 0 else 0

        # Mask invalid actions
        for i, level in enumerate(self.action_levels):
            if level > 0 and level > max_discharge_fraction:
                mask[i] = False  # Can't discharge this much
            elif level < 0 and abs(level) > max_charge_fraction:
                mask[i] = False  # Can't charge this much

        # Ensure at least idle is always valid
        mask[self.idle_action] = True

        return mask

    def get_afrr_bid(
        self,
        battery_state: BatteryState,
        available_capacity_mw: float,
        market_data: Dict[str, Any],
    ) -> Dict:
        """Get aFRR capacity bid for the next hour.

        This is called before gate closure for aFRR bidding.

        Returns:
            Dict with up_mw, down_mw, price
        """
        up, down, price = self._decide_afrr(
            battery_state, available_capacity_mw, market_data
        )

        return {
            'up_capacity_mw': up,
            'down_capacity_mw': down,
            'bid_price_eur': price,
            'timestamp': battery_state.timestamp,
        }


# Factory function
def create_balancing_trader(
    model_path: Optional[str] = None,
    strategy_type: str = "AI",
    **kwargs,
) -> BalancingTrader:
    """Create a balancing trader.

    Args:
        model_path: Path to DRL model (required for AI strategy)
        strategy_type: AI, RULE_BASED, or AGGRESSIVE_HYBRID
        **kwargs: Additional parameters

    Returns:
        Configured BalancingTrader
    """
    return BalancingTrader(
        model_path=model_path,
        strategy_type=strategy_type,
        **kwargs,
    )


# Test
if __name__ == "__main__":
    from datetime import datetime

    # Test with rule-based strategy (no model needed)
    trader = BalancingTrader(
        strategy_type="RULE_BASED",
        afrr_enabled=True,
        mfrr_enabled=True,
    )

    # Create test state
    battery = BatteryState(
        timestamp=datetime.now(),
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    # Test market data
    market_data = {
        'mfrr_price_up': 150.0,
        'mfrr_price_down': 80.0,
        'afrr_up_price': 12.0,
        'afrr_down_price': 10.0,
    }

    # Test simple decision
    print("\n=== Simple Decision Test ===")
    action = trader.decide_action(
        battery_state=battery,
        available_capacity_mw=20.0,
        market_data=market_data,
    )
    print(f"mFRR action: {action:+.1f} MW")

    # Test full decision
    print("\n=== Full Decision Test ===")
    decision = trader.decide_full(
        battery_state=battery,
        available_capacity_mw=20.0,
        market_data=market_data,
        dam_commitment_mw=10.0,
    )
    print(f"Strategy: {decision.strategy_used}")
    print(f"aFRR UP: {decision.afrr_up_capacity_mw:.1f} MW")
    print(f"aFRR DOWN: {decision.afrr_down_capacity_mw:.1f} MW")
    print(f"aFRR Price: {decision.afrr_bid_price:.2f} EUR/MW/h")
    print(f"mFRR Action: {decision.mfrr_action_mw:+.1f} MW")
    print(f"Total: {decision.total_power_mw:+.1f} MW")

    # Test at different SoC levels
    print("\n=== SoC Sensitivity Test ===")
    for soc in [0.10, 0.30, 0.50, 0.70, 0.90]:
        battery.soc = soc
        action = trader.decide_action(
            battery_state=battery,
            available_capacity_mw=20.0,
            market_data=market_data,
        )
        print(f"SoC={soc:.0%}: action={action:+.1f} MW")

    # Test aFRR bid
    print("\n=== aFRR Bid Test ===")
    battery.soc = 0.5
    bid = trader.get_afrr_bid(
        battery_state=battery,
        available_capacity_mw=15.0,
        market_data=market_data,
    )
    print(f"aFRR Bid: UP={bid['up_capacity_mw']:.1f} MW, "
          f"DOWN={bid['down_capacity_mw']:.1f} MW, "
          f"Price={bid['bid_price_eur']:.2f} EUR")
