"""
Market Executor DTOs and Shared Types

I define the data transfer objects shared across all market executors.
Each executor reads from BatterySnapshot (immutable state) and returns
MarketResult (what happened). Only the env's _apply_result() mutates state.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PrecomputedArrays:
    """I hold all precomputed O(1) lookup arrays built from the DataFrame.

    These replace the slow O(n) `self.df.index.date == current_day` scans
    that were converting 57k timestamps per call.
    """
    # I map each step index to its offset within the day (0, 1, 2, ...)
    step_day_offset: np.ndarray
    # I map each step index to a unique day identifier (0, 1, 2, ...)
    step_day_id: np.ndarray
    # I map each day_id to the starting step index of that day
    day_start: np.ndarray
    # I map each day_id to the number of steps in that day
    day_length: np.ndarray
    # I precompute dam_commitment as numpy array for O(1) access
    dam_commitment_values: np.ndarray

    def get_day_indices_for_step(self, step_idx: int) -> np.ndarray:
        """I return all step indices for the same day as step_idx in O(1)."""
        day_id = self.step_day_id[step_idx]
        start = self.day_start[day_id]
        length = self.day_length[day_id]
        return np.arange(start, start + length)


@dataclass
class BatterySnapshot:
    """I hold an immutable snapshot of battery + env state for executor reads.

    Executors receive this instead of accessing the env directly, ensuring
    they cannot mutate state. Only the env's _apply_result() updates state.
    """
    soc: float
    capacity_mwh: float
    max_power_mw: float
    efficiency: float
    eff_sqrt: float
    min_soc: float
    max_soc: float
    time_step_hours: float
    sph: int  # steps-per-hour

    # I include capacity/constraint state
    afrr_commitment_mw: float
    is_selected_for_afrr: bool
    remaining_capacity: float  # max_power_mw - |dam_commitment|
    dam_commitment: float

    # I include cycling state
    daily_cycles: float
    total_cycles: float
    max_daily_cycles: float

    # I include market parameters
    mfrr_imbalance_threshold: float
    mfrr_activation_rate: float
    mfrr_price_cap: float


@dataclass
class MarketResult:
    """I hold the result of a single market execution step.

    Executors return this to the env, which applies it via _apply_result().
    This ensures cascade ordering (DAM → aFRR → IDA → mFRR → XBID → FreeBid)
    because each executor's result updates state before the next runs.
    """
    energy_mw: float = 0.0        # Net energy (positive=discharge, negative=charge)
    revenue: float = 0.0          # Revenue in EUR (before reward scaling)
    capacity_used_mw: float = 0.0 # Capacity consumed for cascade tracking
    cycle_fraction: float = 0.0   # Fraction of capacity cycled
    soc_delta: float = 0.0        # Direct SoC change (signed, post-efficiency)
    is_activated: bool = False     # Whether the market actually executed
    direction: Optional[str] = None  # 'up', 'down', or None
    mwh_sold: float = 0.0         # MWh sold (always positive)
    mwh_bought: float = 0.0       # MWh bought (always positive)
    metadata: Dict = field(default_factory=dict)  # Market-specific extra data
