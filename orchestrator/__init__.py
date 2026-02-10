# Battery Trading Orchestrator
# Phase 1: DAM + Commitment System

"""
Battery Trading Orchestrator Package

I provide a complete orchestration layer for battery trading:

Subsystems:
- DAMForecaster: Price forecasting using EntsoE3 models
- DAMBidder: Threshold-based bidding strategy
- CommitmentExecutor: DAM commitment tracking and execution
- BatteryOrchestrator: Main controller

Usage:
    from orchestrator import BatteryOrchestrator

    # Initialize
    orch = BatteryOrchestrator()

    # Day-ahead (run at 10:00 D-1)
    commitment = orch.run_day_ahead(target_date)

    # Real-time (run every 15 min)
    dispatch = orch.run_realtime(current_time)
"""

from .interfaces import (
    # Data classes
    PriceForecast,
    DAMCommitment,
    BatteryState,
    # Interfaces
    IDAMForecaster,
    IDAMBidder,
    ICommitmentExecutor,
    IBalancingTrader,
    IOrchestrator,
)

from .dam_forecaster import DAMForecaster
from .dam_bidder import DAMBidder
from .commitment_executor import CommitmentExecutor
from .balancing_trader import BalancingTrader, BalancingDecision, create_balancing_trader
from .intraday_trader import IntraDayTrader, IntraDayOrder, IntraDayPosition, create_intraday_trader
from .orchestrator import BatteryOrchestrator

__all__ = [
    # Data classes
    'PriceForecast',
    'DAMCommitment',
    'BatteryState',
    'BalancingDecision',
    'IntraDayOrder',
    'IntraDayPosition',
    # Interfaces
    'IDAMForecaster',
    'IDAMBidder',
    'ICommitmentExecutor',
    'IBalancingTrader',
    'IOrchestrator',
    # Implementations
    'DAMForecaster',
    'DAMBidder',
    'CommitmentExecutor',
    'BalancingTrader',
    'IntraDayTrader',
    'BatteryOrchestrator',
    # Factories
    'create_balancing_trader',
    'create_intraday_trader',
]
