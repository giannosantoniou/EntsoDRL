"""
Step Context DTO for the Pipeline Pattern

I hold all cross-stage state during a single step() execution.
Stages mutate this context in place, avoiding complex return types.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional

from gym_envs.market_executors import MarketResult


@dataclass
class StepContext:
    """I hold the mutable state that flows through the market pipeline.

    Each stage reads what it needs, executes its market logic, and updates
    me in place. This replaces the 20+ local variables in the old step().
    """

    # I hold inputs that don't change during the step
    action: np.ndarray
    row: pd.Series
    step_idx: int
    dam_commitment: float

    # I track remaining inverter capacity through the cascade
    remaining_capacity: float

    # I store each stage's result for reward calculation
    results: Dict[str, MarketResult] = field(default_factory=dict)

    # I accumulate total energy across all stages
    actual_energy_mw: float = 0.0
    dam_executed_mw: float = 0.0

    # I hold aFRR cross-stage state (set by AfrrStage, read by reward)
    afrr_activated: bool = False
    afrr_direction: Optional[str] = None
    afrr_energy_delivered: float = 0.0
    afrr_capacity_revenue: float = 0.0
    afrr_max_deliverable_mw: Optional[float] = None

    # I hold IDA/FreeBid cross-stage state (set by stages, read by reward)
    ida_shortfall_mw: float = 0.0
    xbid_energy_mw: float = 0.0
    free_bid_activated: bool = False
    agent_bid_price: float = 0.0

    # I track per-stage energy for the info dict
    intraday_energy_mw: float = 0.0
    intraday_revenue: float = 0.0
    mfrr_energy_mw: float = 0.0
    mfrr_revenue: float = 0.0
    ida_energy_mw: float = 0.0
    free_bid_energy_mw: float = 0.0
