"""
Bidding Strategies for mFRR Market

I implement different strategies to compare their performance:
1. BaselineStrategy: Always idle (benchmark)
2. SimpleThresholdStrategy: Bid based on price thresholds
3. TimeBasedStrategy: Bid based on time-of-day patterns
4. DataDrivenStrategy: Use features to make decisions
5. OptimalStrategy: Perfect foresight (theoretical maximum)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class MarketState:
    """Current market state for strategy decisions."""
    timestamp: pd.Timestamp
    hour: int
    day_of_week: int
    is_weekend: bool

    # Prices (may be None if not available)
    mfrr_price_up: Optional[float] = None
    mfrr_price_down: Optional[float] = None
    mfrr_spread: Optional[float] = None

    # Lagged prices (for prediction)
    price_up_lag_1h: Optional[float] = None
    price_up_lag_24h: Optional[float] = None
    price_up_mean_24h: Optional[float] = None

    # System state
    net_imbalance_mw: Optional[float] = None
    load_mw: Optional[float] = None
    res_total_mw: Optional[float] = None
    net_cross_border_mw: Optional[float] = None

    # Battery state
    soc: float = 0.5
    can_discharge: bool = True
    can_charge: bool = True


class Strategy(ABC):
    """Base class for bidding strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def decide(self, state: MarketState) -> Tuple[str, float]:
        """
        Make a bidding decision.

        Args:
            state: Current market state

        Returns:
            Tuple of (action, power_mw)
            action: 'discharge', 'charge', or 'idle'
            power_mw: Power level (positive for discharge, negative for charge)
        """
        pass


class BaselineStrategy(Strategy):
    """
    I do nothing - always idle.
    This is the benchmark to compare against.
    """

    def __init__(self):
        super().__init__("Baseline (Idle)")

    def decide(self, state: MarketState) -> Tuple[str, float]:
        return 'idle', 0.0


class SimpleThresholdStrategy(Strategy):
    """
    I bid based on simple price thresholds.

    - Discharge when mFRR Up price > threshold_high
    - Charge when mFRR Up price < threshold_low (or negative)
    - Idle otherwise
    """

    def __init__(self, threshold_high: float = 150.0, threshold_low: float = 30.0,
                 max_power_mw: float = 30.0):
        super().__init__(f"Threshold (>{threshold_high}, <{threshold_low})")
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.max_power_mw = max_power_mw

    def decide(self, state: MarketState) -> Tuple[str, float]:
        price = state.mfrr_price_up

        if price is None or np.isnan(price):
            return 'idle', 0.0

        if price > self.threshold_high and state.can_discharge:
            return 'discharge', self.max_power_mw
        elif price < self.threshold_low and state.can_charge:
            return 'charge', -self.max_power_mw
        else:
            return 'idle', 0.0


class TimeBasedStrategy(Strategy):
    """
    I bid based on time-of-day patterns discovered in EDA.

    - Discharge during evening peak (17-21h)
    - Charge during solar hours (9-15h)
    - Reduce activity on weekends
    """

    def __init__(self, max_power_mw: float = 30.0, weekend_factor: float = 0.5):
        super().__init__("Time-Based")
        self.max_power_mw = max_power_mw
        self.weekend_factor = weekend_factor

    def decide(self, state: MarketState) -> Tuple[str, float]:
        hour = state.hour
        power = self.max_power_mw

        # Reduce power on weekends
        if state.is_weekend:
            power *= self.weekend_factor

        # Evening peak: discharge
        if 17 <= hour <= 21 and state.can_discharge:
            return 'discharge', power

        # Solar hours: charge
        elif 9 <= hour <= 15 and state.can_charge:
            return 'charge', -power

        # Morning ramp: discharge
        elif 5 <= hour <= 8 and state.can_discharge:
            return 'discharge', power * 0.5

        else:
            return 'idle', 0.0


class SpreadStrategy(Strategy):
    """
    I bid based on the mFRR spread (Up - Down).

    - Discharge when spread > threshold (Up pays more than Down)
    - Charge when spread < -threshold (Down pays more)
    - Scale power based on spread magnitude
    """

    def __init__(self, threshold: float = 50.0, max_power_mw: float = 30.0):
        super().__init__(f"Spread (>{threshold})")
        self.threshold = threshold
        self.max_power_mw = max_power_mw

    def decide(self, state: MarketState) -> Tuple[str, float]:
        spread = state.mfrr_spread

        if spread is None or np.isnan(spread):
            return 'idle', 0.0

        if spread > self.threshold and state.can_discharge:
            # Scale power by spread magnitude (up to 2x threshold)
            scale = min(spread / (2 * self.threshold), 1.0)
            return 'discharge', self.max_power_mw * scale

        elif spread < -self.threshold and state.can_charge:
            scale = min(abs(spread) / (2 * self.threshold), 1.0)
            return 'charge', -self.max_power_mw * scale

        else:
            return 'idle', 0.0


class DataDrivenStrategy(Strategy):
    """
    I use multiple features to make decisions.

    Combines:
    - Price signals (current, lagged, rolling)
    - Time patterns
    - System state (if available)
    """

    def __init__(self, max_power_mw: float = 30.0):
        super().__init__("Data-Driven")
        self.max_power_mw = max_power_mw

        # Thresholds based on EDA findings
        self.price_high = 150.0
        self.price_low = 30.0
        self.spread_threshold = 40.0

    def decide(self, state: MarketState) -> Tuple[str, float]:
        score = 0.0  # Positive = discharge, Negative = charge

        # Price signal
        if state.mfrr_price_up is not None and not np.isnan(state.mfrr_price_up):
            if state.mfrr_price_up > self.price_high:
                score += 2.0
            elif state.mfrr_price_up < self.price_low:
                score -= 2.0

        # Spread signal
        if state.mfrr_spread is not None and not np.isnan(state.mfrr_spread):
            if state.mfrr_spread > self.spread_threshold:
                score += 1.5
            elif state.mfrr_spread < -self.spread_threshold:
                score -= 1.5

        # Time signal
        if 17 <= state.hour <= 21:  # Evening peak
            score += 1.0
        elif 9 <= state.hour <= 15:  # Solar hours
            score -= 1.0

        # Weekend penalty
        if state.is_weekend:
            score *= 0.7

        # Price momentum (if available)
        if state.price_up_lag_1h is not None and state.mfrr_price_up is not None:
            if not np.isnan(state.price_up_lag_1h) and not np.isnan(state.mfrr_price_up):
                momentum = state.mfrr_price_up - state.price_up_lag_1h
                if momentum > 50:
                    score += 0.5
                elif momentum < -50:
                    score -= 0.5

        # SoC management
        if state.soc > 0.8:  # High SoC, prefer discharge
            score += 0.5
        elif state.soc < 0.3:  # Low SoC, prefer charge
            score -= 0.5

        # Make decision
        if score >= 2.0 and state.can_discharge:
            power_scale = min(score / 4.0, 1.0)
            return 'discharge', self.max_power_mw * power_scale
        elif score <= -2.0 and state.can_charge:
            power_scale = min(abs(score) / 4.0, 1.0)
            return 'charge', -self.max_power_mw * power_scale
        else:
            return 'idle', 0.0


class OptimalStrategy(Strategy):
    """
    I use perfect foresight (cheating) to show theoretical maximum profit.
    This is only for comparison - not achievable in practice.

    I simulate an optimal trading schedule using dynamic programming,
    considering battery constraints (SoC, efficiency, degradation).
    """

    def __init__(self, prices: pd.Series, max_power_mw: float = 30.0,
                 capacity_mwh: float = 146.0, efficiency: float = 0.94,
                 degradation_cost: float = 15.0):
        super().__init__("Optimal (Perfect Foresight)")
        self.prices = prices
        self.max_power_mw = max_power_mw
        self.capacity_mwh = capacity_mwh
        self.efficiency = efficiency
        self.degradation_cost = degradation_cost

        # Pre-calculate optimal actions
        self._precompute_optimal()

    def _precompute_optimal(self):
        """
        I pre-compute optimal actions using a greedy approach with lookahead.

        Strategy: For each cycle, find the best charge/discharge pair within
        a lookahead window that maximizes profit (spread - degradation).
        """
        self.optimal_actions = {}

        prices = self.prices.dropna()
        if len(prices) < 48:
            return

        # I calculate the minimum spread needed to be profitable
        # Profit = discharge_price - charge_price/efficiency - 2*degradation_cost
        # For profit > 0: spread > 2*degradation + charge*(1/eff - 1)
        min_profitable_spread = 2 * self.degradation_cost + 50 * (1/self.efficiency - 1)

        # I find profitable trading opportunities
        # Look for pairs of hours where: high_price - low_price > min_profitable_spread
        price_arr = prices.values
        timestamps = prices.index
        n = len(prices)

        # I mark hours as discharge (1), charge (-1), or idle (0)
        actions = np.zeros(n)

        # I use a sliding window approach
        # For each day, find the best charge and discharge hours
        window_size = 24  # One day lookahead

        for start in range(0, n - window_size, window_size // 2):
            end = min(start + window_size, n)
            window_prices = price_arr[start:end]

            # I find the maximum and minimum prices in the window
            max_idx = np.argmax(window_prices)
            min_idx = np.argmin(window_prices)

            max_price = window_prices[max_idx]
            min_price = window_prices[min_idx]

            spread = max_price - min_price / self.efficiency
            net_profit = spread - 2 * self.degradation_cost

            # Only trade if profitable and min comes before max
            if net_profit > 0 and min_idx < max_idx:
                # Mark charge and discharge hours
                if actions[start + min_idx] == 0:
                    actions[start + min_idx] = -1  # Charge
                if actions[start + max_idx] == 0:
                    actions[start + max_idx] = 1   # Discharge

        # I also check for high-value individual hours
        price_percentile_high = np.percentile(price_arr, 90)
        price_percentile_low = np.percentile(price_arr, 10)

        for i, (ts, price) in enumerate(zip(timestamps, price_arr)):
            # I discharge at very high prices regardless
            if price > price_percentile_high and actions[i] == 0:
                actions[i] = 1  # Discharge
            # I charge at very low prices regardless
            elif price < price_percentile_low and actions[i] == 0:
                actions[i] = -1  # Charge

            # Store action
            if actions[i] == 1:
                self.optimal_actions[ts] = ('discharge', self.max_power_mw)
            elif actions[i] == -1:
                self.optimal_actions[ts] = ('charge', -self.max_power_mw)
            else:
                self.optimal_actions[ts] = ('idle', 0.0)

    def decide(self, state: MarketState) -> Tuple[str, float]:
        action, power = self.optimal_actions.get(state.timestamp, ('idle', 0.0))

        # I check battery constraints
        if action == 'discharge' and not state.can_discharge:
            return 'idle', 0.0
        if action == 'charge' and not state.can_charge:
            return 'idle', 0.0

        return action, power


class HybridMLTimeStrategy(Strategy):
    """
    I combine Time-Based activity with ML price intelligence.

    - I use time windows from Time-Based (when to be active)
    - I use ML predictions to filter for good prices (quality control)
    - This gives high utilization with smart price selection
    """

    def __init__(self, ml_predictor, max_power_mw: float = 30.0,
                 min_discharge_price: float = 100.0,
                 max_charge_price: float = 80.0):
        super().__init__("Hybrid (Time + ML)")
        self.ml_predictor = ml_predictor
        self.max_power_mw = max_power_mw
        self.min_discharge_price = min_discharge_price
        self.max_charge_price = max_charge_price

    def decide(self, state: MarketState) -> Tuple[str, float]:
        hour = state.hour

        # I check if we're in a Time-Based active window
        is_discharge_window = (17 <= hour <= 21) or (5 <= hour <= 8)
        is_charge_window = (9 <= hour <= 15)

        # I get ML prediction for price quality check
        try:
            ml_action, ml_power = self.ml_predictor.decide(state)
        except Exception:
            ml_action, ml_power = 'idle', 0.0

        # I combine time windows with ML quality filter
        if is_discharge_window and state.can_discharge:
            # Discharge during peak hours, but check if ML agrees or if price is good
            if ml_action == 'discharge':
                # ML agrees - full power
                return 'discharge', self.max_power_mw
            elif state.mfrr_price_up and state.mfrr_price_up > self.min_discharge_price:
                # Price is above minimum threshold - trade anyway
                return 'discharge', self.max_power_mw * 0.8
            elif hour >= 17:  # Evening peak - always trade
                return 'discharge', self.max_power_mw * 0.6
            else:
                return 'idle', 0.0

        elif is_charge_window and state.can_charge:
            # Charge during solar hours, but check if ML agrees or if price is good
            if ml_action == 'charge':
                # ML agrees - full power
                return 'charge', -self.max_power_mw
            elif state.mfrr_price_up and state.mfrr_price_up < self.max_charge_price:
                # Price is below maximum threshold - trade anyway
                return 'charge', -self.max_power_mw * 0.8
            else:
                return 'idle', 0.0

        else:
            # Outside time windows - only trade if ML is very confident
            if ml_action == 'discharge' and state.can_discharge:
                if state.mfrr_price_up and state.mfrr_price_up > 200:
                    return 'discharge', self.max_power_mw
            elif ml_action == 'charge' and state.can_charge:
                if state.mfrr_price_up and state.mfrr_price_up < 20:
                    return 'charge', -self.max_power_mw

            return 'idle', 0.0


class AggressiveHybridStrategy(Strategy):
    """
    I'm an aggressive hybrid that always trades during time windows,
    but uses ML to optimize power level.

    - Always trade during peak/solar hours (like Time-Based)
    - But scale power based on ML confidence and price quality
    - This maximizes utilization while still being smart about sizing
    """

    def __init__(self, ml_predictor, max_power_mw: float = 30.0):
        super().__init__("Aggressive Hybrid")
        self.ml_predictor = ml_predictor
        self.max_power_mw = max_power_mw

    def decide(self, state: MarketState) -> Tuple[str, float]:
        hour = state.hour
        is_weekend = state.is_weekend

        # I get ML prediction
        try:
            ml_action, ml_power = self.ml_predictor.decide(state)
            ml_agrees = True
        except Exception:
            ml_action, ml_power = 'idle', 0.0
            ml_agrees = False

        # Weekend adjustment factor
        weekend_factor = 0.5 if is_weekend else 1.0

        # Evening peak (17-21h): ALWAYS discharge
        if 17 <= hour <= 21 and state.can_discharge:
            if ml_action == 'discharge':
                power = self.max_power_mw  # Full power if ML agrees
            else:
                power = self.max_power_mw * 0.7  # Reduced if ML disagrees
            return 'discharge', power * weekend_factor

        # Morning ramp (5-8h): Usually discharge
        elif 5 <= hour <= 8 and state.can_discharge:
            if ml_action == 'discharge':
                power = self.max_power_mw * 0.8
            else:
                power = self.max_power_mw * 0.4
            return 'discharge', power * weekend_factor

        # Solar hours (9-15h): ALWAYS charge
        elif 9 <= hour <= 15 and state.can_charge:
            if ml_action == 'charge':
                power = self.max_power_mw  # Full power if ML agrees
            else:
                power = self.max_power_mw * 0.7  # Reduced if ML disagrees
            return 'charge', -power * weekend_factor

        # Off-peak hours: Only trade if ML strongly recommends
        else:
            if ml_action == 'discharge' and state.can_discharge:
                if state.mfrr_price_up and state.mfrr_price_up > 150:
                    return 'discharge', self.max_power_mw * weekend_factor
            elif ml_action == 'charge' and state.can_charge:
                if state.mfrr_price_up and state.mfrr_price_up < 30:
                    return 'charge', -self.max_power_mw * weekend_factor

            return 'idle', 0.0
