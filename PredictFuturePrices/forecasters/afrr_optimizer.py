"""
aFRR Capacity Optimizer — I decide optimal aFRR commitment per 4h block.

I am a rule-based optimizer that maximizes aFRR capacity payments while
ensuring the battery can deliver if activated by the TSO.

Key principles:
1. Higher aFRR price → commit more MW
2. Always maintain SoC buffer for delivery
3. Don't conflict with DAM schedule capacity
4. Prefer peak hours (higher activation probability)

Revenue model:
  capacity_payment = committed_MW × afrr_cap_price × hours
  Expected: 15MW × 27 EUR/MW/h × 24h × 65% selection = ~6,300 EUR/day

Usage:
    optimizer = AfrrOptimizer(max_power_mw=30, capacity_mwh=146)
    commitment = optimizer.decide(block_hour=16, soc=0.5,
                                   dam_avg_mw=10, afrr_cap_price=25)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class AfrrDecision:
    """I hold the aFRR commitment decision for one 4h block."""
    commitment_mw: float      # MW to commit (0..30)
    block_start_hour: int     # 0, 4, 8, 12, 16, 20
    price_tier: float         # 0.7..1.3 (merit order position)
    reason: str               # Why this decision
    expected_revenue: float   # EUR for this 4h block (if selected)


class AfrrOptimizer:
    """I optimize aFRR capacity commitment using rule-based logic.

    I decide for each 4h block:
    - How many MW to commit (0, 5, 10, 15, 20, 25, 30)
    - At what price tier (affects selection probability)

    My strategy:
    - I commit MORE when aFRR prices are HIGH (profit opportunity)
    - I commit LESS when SoC is LOW (delivery risk)
    - I respect DAM schedule capacity limits
    - I bid COMPETITIVELY (price_tier ~0.85) to maximize selection
    """

    # I define price thresholds for commitment levels
    PRICE_THRESHOLDS = [
        (40, 1.0),   # Price > 40 EUR → commit 100% available
        (25, 0.75),  # Price > 25 EUR → commit 75%
        (15, 0.50),  # Price > 15 EUR → commit 50%
        (8,  0.30),  # Price > 8 EUR → commit 30%
    ]

    # I define minimum SoC for commitment
    MIN_SOC_FOR_COMMIT = 0.20       # I need at least 20% SoC to deliver
    COMFORTABLE_SOC = 0.40          # Above 40%, I commit freely
    MIN_COMMITMENT_MW = 5.0         # Minimum practical commitment

    # I set competitive price tier (lower = cheaper bid = more likely selected)
    DEFAULT_PRICE_TIER = 0.85       # 15% below clearing → ~90% selection prob

    def __init__(
        self,
        max_power_mw: float = 30.0,
        capacity_mwh: float = 146.0,
        min_soc: float = 0.05,
        selection_rate: float = 0.65,
    ):
        self._max_power = max_power_mw
        self._capacity = capacity_mwh
        self._min_soc = min_soc
        self._selection_rate = selection_rate

    def decide(
        self,
        block_start_hour: int,
        soc: float,
        dam_avg_mw: float,
        afrr_cap_price: float,
        afrr_cap_price_history: Optional[np.ndarray] = None,
    ) -> AfrrDecision:
        """I decide aFRR commitment for one 4h block.

        Args:
            block_start_hour: 0, 4, 8, 12, 16, 20
            soc: Current SoC (0-1)
            dam_avg_mw: Average absolute DAM commitment in this block (MW)
            afrr_cap_price: Current aFRR capacity price (EUR/MW/h)
            afrr_cap_price_history: Optional recent price history for context

        Returns:
            AfrrDecision with commitment_mw and metadata
        """
        # I compute available capacity after DAM
        available = max(0, self._max_power - abs(dam_avg_mw))

        # I check SoC constraint: can I deliver if activated?
        if soc < self.MIN_SOC_FOR_COMMIT:
            return AfrrDecision(
                commitment_mw=0, block_start_hour=block_start_hour,
                price_tier=1.0, reason=f"SoC too low ({soc:.0%})",
                expected_revenue=0
            )

        if available < self.MIN_COMMITMENT_MW:
            return AfrrDecision(
                commitment_mw=0, block_start_hour=block_start_hour,
                price_tier=1.0, reason=f"No capacity (DAM uses {dam_avg_mw:.0f}MW)",
                expected_revenue=0
            )

        # I determine commitment level based on price
        commitment_fraction = 0.0
        for threshold, fraction in self.PRICE_THRESHOLDS:
            if afrr_cap_price >= threshold:
                commitment_fraction = fraction
                break

        if commitment_fraction == 0:
            return AfrrDecision(
                commitment_mw=0, block_start_hour=block_start_hour,
                price_tier=1.0, reason=f"Price too low ({afrr_cap_price:.0f} EUR)",
                expected_revenue=0
            )

        # I scale commitment by SoC confidence
        if soc < self.COMFORTABLE_SOC:
            # I reduce commitment proportionally when SoC is low
            soc_scale = (soc - self.MIN_SOC_FOR_COMMIT) / (self.COMFORTABLE_SOC - self.MIN_SOC_FOR_COMMIT)
            soc_scale = np.clip(soc_scale, 0.3, 1.0)
            commitment_fraction *= soc_scale

        # I compute final MW
        commitment_mw = min(available, self._max_power * commitment_fraction)
        commitment_mw = max(0, round(commitment_mw / 5) * 5)  # I round to nearest 5 MW

        if commitment_mw < self.MIN_COMMITMENT_MW:
            commitment_mw = 0

        # I compute expected revenue
        expected_revenue = (commitment_mw * afrr_cap_price * 4.0 *
                           self._selection_rate)  # 4h block × selection probability

        return AfrrDecision(
            commitment_mw=commitment_mw,
            block_start_hour=block_start_hour,
            price_tier=self.DEFAULT_PRICE_TIER,
            reason=f"Price {afrr_cap_price:.0f}EUR → {commitment_mw:.0f}MW ({commitment_fraction:.0%})",
            expected_revenue=expected_revenue,
        )

    def optimize_day(
        self,
        soc_initial: float,
        dam_schedule: np.ndarray,
        afrr_prices: np.ndarray,
        sph: int = 4,
    ) -> List[AfrrDecision]:
        """I optimize aFRR commitment for all 6 blocks of a day.

        Args:
            soc_initial: SoC at start of day
            dam_schedule: DAM MW schedule (96 slots for 15-min)
            afrr_prices: aFRR capacity prices (96 slots or 24 hourly)
            sph: Steps per hour (4 for 15-min)

        Returns:
            List of 6 AfrrDecision (one per 4h block)
        """
        decisions = []
        soc = soc_initial
        block_hours = [0, 4, 8, 12, 16, 20]

        for block_start in block_hours:
            # I compute average DAM commitment in this 4h block
            slot_start = block_start * sph
            slot_end = min(slot_start + 4 * sph, len(dam_schedule))
            if slot_end > slot_start:
                dam_block = np.abs(dam_schedule[slot_start:slot_end])
                dam_avg = np.mean(dam_block)
            else:
                dam_avg = 0

            # I get aFRR price for this block
            if len(afrr_prices) >= 96:
                # I use quarter-hourly prices
                price_block = afrr_prices[slot_start:slot_end]
                afrr_price = np.mean(price_block) if len(price_block) > 0 else 20
            elif len(afrr_prices) >= 24:
                # I use hourly prices
                price_block = afrr_prices[block_start:block_start+4]
                afrr_price = np.mean(price_block) if len(price_block) > 0 else 20
            else:
                afrr_price = 20  # Default

            # I decide commitment
            decision = self.decide(
                block_start_hour=block_start,
                soc=soc,
                dam_avg_mw=dam_avg,
                afrr_cap_price=afrr_price,
            )
            decisions.append(decision)

            # I simulate SoC change from DAM (rough estimate)
            if slot_end > slot_start:
                dam_energy = np.sum(dam_schedule[slot_start:slot_end]) * (1.0 / sph) / 4
                soc -= dam_energy / self._capacity * 0.1  # Rough approximation
                soc = np.clip(soc, self._min_soc, 0.95)

        return decisions

    def backtest(
        self,
        df,
        n_days: int = 365,
        sph: int = 4,
    ) -> dict:
        """I backtest the aFRR optimizer on historical data.

        Returns:
            dict with daily_revenues, total_revenue, mean_commitment, etc.
        """
        day_steps = 24 * sph
        daily_revenues = []
        daily_commitments = []

        for d in range(min(n_days, len(df) // day_steps)):
            start = d * day_steps
            end = start + day_steps
            if end > len(df):
                break

            day_data = df.iloc[start:end]

            # I get aFRR prices
            if 'afrr_cap_up_price' in df.columns:
                afrr_prices = day_data['afrr_cap_up_price'].values
            else:
                afrr_prices = np.full(day_steps, 20.0)

            # I get DAM schedule (use price as proxy for schedule direction)
            dam_schedule = np.zeros(day_steps)  # I assume no DAM for simplicity

            # I optimize the day
            decisions = self.optimize_day(
                soc_initial=0.5,
                dam_schedule=dam_schedule,
                afrr_prices=afrr_prices,
                sph=sph,
            )

            # I compute daily revenue
            day_revenue = sum(d.expected_revenue for d in decisions)
            day_commitment = np.mean([d.commitment_mw for d in decisions])

            daily_revenues.append(day_revenue)
            daily_commitments.append(day_commitment)

        daily_revenues = np.array(daily_revenues)
        return {
            'daily_revenues': daily_revenues,
            'total_revenue': np.sum(daily_revenues),
            'mean_daily_revenue': np.mean(daily_revenues),
            'mean_commitment_mw': np.mean(daily_commitments),
            'positive_days': np.sum(daily_revenues > 0),
            'total_days': len(daily_revenues),
        }
