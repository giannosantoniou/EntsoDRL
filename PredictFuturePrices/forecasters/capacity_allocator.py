"""
Dynamic Capacity Allocator — I decide how to split battery MW between aFRR and DAM.

I monitor aFRR market conditions and automatically adjust allocation:
- High aFRR prices (>20 EUR) → commit more to aFRR
- Low aFRR prices (<10 EUR) → free capacity for DAM arbitrage
- Falling aFRR trend → gradually shift to DAM

This makes the system future-proof: as more BESS enter the market and
aFRR prices drop, I automatically pivot to DAM arbitrage.

Usage:
    allocator = CapacityAllocator(max_power_mw=30)
    afrr_mw, dam_mw = allocator.decide(afrr_price=25, dam_spread=80)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AllocationDecision:
    """I hold the capacity split decision."""
    afrr_mw: float          # MW committed to aFRR
    dam_mw: float           # MW available for DAM arbitrage
    afrr_fraction: float    # 0-1, fraction of capacity for aFRR
    reason: str


class CapacityAllocator:
    """I dynamically split battery capacity between aFRR and DAM.

    My strategy adapts to market conditions:
    1. I track rolling aFRR prices (7-day window)
    2. I compare aFRR expected revenue vs DAM expected revenue
    3. I allocate more MW to whichever pays more
    4. I always keep minimum 5MW for each market (diversification)

    As BESS deployment grows (4.7 GW planned in Greece), aFRR prices
    will fall. I automatically detect this via declining rolling average
    and shift capacity to DAM.
    """

    MIN_AFRR_MW = 5.0       # I always keep minimum for aFRR
    MIN_DAM_MW = 5.0         # I always keep minimum for DAM
    AFRR_SELECTION_RATE = 0.65

    # I define aFRR price tiers for allocation
    # These thresholds auto-calibrate via rolling percentiles
    DEFAULT_HIGH_PRICE = 30.0     # EUR/MW/h — commit aggressively
    DEFAULT_MED_PRICE = 15.0      # EUR/MW/h — balanced split
    DEFAULT_LOW_PRICE = 8.0       # EUR/MW/h — favor DAM

    def __init__(
        self,
        max_power_mw: float = 30.0,
        history_days: int = 7,
        sph: int = 4,
    ):
        self._max_power = max_power_mw
        self._history_window = history_days * 24 * sph
        self._sph = sph

        # I track rolling aFRR price history for adaptive thresholds
        self._afrr_price_history = []
        self._dam_spread_history = []

    def update_history(self, afrr_price: float, dam_spread: float):
        """I add a new observation to my rolling history."""
        self._afrr_price_history.append(afrr_price)
        self._dam_spread_history.append(dam_spread)

        # I keep only the last N observations
        if len(self._afrr_price_history) > self._history_window:
            self._afrr_price_history.pop(0)
        if len(self._dam_spread_history) > self._history_window:
            self._dam_spread_history.pop(0)

    def decide(
        self,
        afrr_price: float,
        dam_spread: float,
        soc: float = 0.5,
        dam_schedule_mw: float = 0.0,
    ) -> AllocationDecision:
        """I decide the optimal aFRR/DAM capacity split for one 4h block.

        Args:
            afrr_price: Current aFRR capacity price (EUR/MW/h)
            dam_spread: Expected DAM price spread in this block (EUR/MWh)
            soc: Current battery SoC (0-1)
            dam_schedule_mw: Already committed DAM MW (absolute)

        Returns:
            AllocationDecision with afrr_mw and dam_mw
        """
        available = max(0, self._max_power - abs(dam_schedule_mw))

        if available < self.MIN_AFRR_MW + self.MIN_DAM_MW:
            # I don't have enough capacity to split — give it all to higher value
            if afrr_price > 10:
                return AllocationDecision(available, 0, 1.0, "Low capacity, aFRR priority")
            else:
                return AllocationDecision(0, available, 0.0, "Low capacity, DAM priority")

        # I compute adaptive thresholds from recent history
        if len(self._afrr_price_history) > 100:
            p75 = np.percentile(self._afrr_price_history, 75)
            p25 = np.percentile(self._afrr_price_history, 25)
            high_threshold = p75
            low_threshold = p25
        else:
            high_threshold = self.DEFAULT_HIGH_PRICE
            low_threshold = self.DEFAULT_LOW_PRICE

        # I compute expected revenue per MW for each market
        afrr_rev_per_mw = afrr_price * 4 * self.AFRR_SELECTION_RATE  # EUR per MW per block
        dam_rev_per_mw = dam_spread * 0.40 * 2 * 0.94 if dam_spread > 20 else 0  # Conservative

        # I decide allocation based on relative value
        if afrr_price >= high_threshold:
            # I commit heavily to aFRR — high prices, don't miss out
            afrr_fraction = 0.85
            reason = f"High aFRR ({afrr_price:.0f}>{high_threshold:.0f})"
        elif afrr_price >= low_threshold:
            # I do balanced split — both markets have value
            # I compute proportional allocation based on expected revenue
            total_rev = afrr_rev_per_mw + max(dam_rev_per_mw, 1)
            afrr_fraction = afrr_rev_per_mw / total_rev
            afrr_fraction = np.clip(afrr_fraction, 0.3, 0.8)
            reason = f"Balanced (aFRR={afrr_rev_per_mw:.0f} vs DAM={dam_rev_per_mw:.0f} EUR/MW)"
        else:
            # I favor DAM — aFRR prices too low
            afrr_fraction = 0.20
            reason = f"Low aFRR ({afrr_price:.0f}<{low_threshold:.0f}), DAM priority"

        # I adjust for SoC constraints
        if soc < 0.20:
            # I reduce aFRR commitment — risk of non-delivery
            afrr_fraction *= 0.5
            reason += " [SoC low]"

        # I compute MW allocation
        afrr_mw = round(available * afrr_fraction / 5) * 5  # I round to 5MW
        afrr_mw = max(self.MIN_AFRR_MW, min(available - self.MIN_DAM_MW, afrr_mw))
        dam_mw = available - afrr_mw

        # I ensure minimums
        if afrr_mw < self.MIN_AFRR_MW:
            afrr_mw = self.MIN_AFRR_MW
            dam_mw = available - afrr_mw
        if dam_mw < self.MIN_DAM_MW and available > self.MIN_AFRR_MW + self.MIN_DAM_MW:
            dam_mw = self.MIN_DAM_MW
            afrr_mw = available - dam_mw

        return AllocationDecision(
            afrr_mw=afrr_mw,
            dam_mw=dam_mw,
            afrr_fraction=afrr_mw / max(available, 1),
            reason=reason,
        )

    def simulate_future(
        self,
        current_afrr_prices: np.ndarray,
        price_decline_pct: float = 0.0,
    ) -> dict:
        """I simulate what happens if aFRR prices decline by X%.

        Useful for scenario analysis: what if 4.7 GW BESS enters
        and aFRR prices drop 50%?
        """
        adjusted = current_afrr_prices * (1 - price_decline_pct / 100)

        # I simulate allocation decisions across all blocks
        blocks = len(adjusted) // 16
        total_afrr_rev = 0
        total_dam_rev = 0
        afrr_fractions = []

        for b in range(blocks):
            start = b * 16
            end = start + 16
            avg_price = np.mean(adjusted[start:end])
            self.update_history(avg_price, 50)  # I assume 50 EUR DAM spread
            decision = self.decide(avg_price, 50)
            afrr_fractions.append(decision.afrr_fraction)

            afrr_rev = decision.afrr_mw * avg_price * 4 * self.AFRR_SELECTION_RATE
            dam_rev = decision.dam_mw * 50 * 0.40 * 2 * 0.94 if 50 > 20 else 0
            total_afrr_rev += afrr_rev
            total_dam_rev += dam_rev

        daily = (total_afrr_rev + total_dam_rev) / max(blocks / 6, 1)
        return {
            'daily_revenue': daily,
            'mean_afrr_fraction': np.mean(afrr_fractions),
            'afrr_daily': total_afrr_rev / max(blocks / 6, 1),
            'dam_daily': total_dam_rev / max(blocks / 6, 1),
            'price_decline': price_decline_pct,
        }
