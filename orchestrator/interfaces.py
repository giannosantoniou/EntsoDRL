"""
Interfaces for the Battery Trading Orchestrator

I define the contracts that each subsystem must implement.
This ensures loose coupling and easy testing/replacement of components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional
import numpy as np


@dataclass
class PriceForecast:
    """DAM price forecast for a single day.

    Contains 96 values (24h × 4 quarters) for both buy and sell prices.
    """
    target_date: date
    timestamps: List[datetime]  # 96 timestamps (15-min intervals)
    buy_prices: np.ndarray      # 96 values - price to BUY energy
    sell_prices: np.ndarray     # 96 values - price to SELL energy
    confidence: Optional[np.ndarray] = None  # Optional confidence intervals

    @property
    def n_intervals(self) -> int:
        return len(self.timestamps)

    def get_hourly_prices(self) -> tuple:
        """Aggregate to hourly prices (mean of 4 quarters)."""
        buy_hourly = self.buy_prices.reshape(24, 4).mean(axis=1)
        sell_hourly = self.sell_prices.reshape(24, 4).mean(axis=1)
        return buy_hourly, sell_hourly


@dataclass
class DAMCommitment:
    """DAM commitment schedule for a single day.

    Positive values = SELL (discharge)
    Negative values = BUY (charge)
    """
    target_date: date
    timestamps: List[datetime]  # 96 timestamps
    power_mw: np.ndarray        # 96 values - committed power in MW

    @property
    def total_sell_mwh(self) -> float:
        """Total energy to sell (discharge)."""
        return float(np.sum(self.power_mw[self.power_mw > 0]) / 4)  # /4 for 15-min

    @property
    def total_buy_mwh(self) -> float:
        """Total energy to buy (charge)."""
        return float(np.sum(np.abs(self.power_mw[self.power_mw < 0])) / 4)


@dataclass
class BatteryState:
    """Current state of the battery."""
    timestamp: datetime
    soc: float                  # State of charge [0, 1]
    available_power_mw: float   # Currently available power
    capacity_mwh: float         # Total capacity

    @property
    def available_energy_mwh(self) -> float:
        return self.soc * self.capacity_mwh

    @property
    def available_headroom_mwh(self) -> float:
        return (1 - self.soc) * self.capacity_mwh


class IDAMForecaster(ABC):
    """Interface for DAM price forecasting system."""

    @abstractmethod
    def forecast(self, target_date: date) -> PriceForecast:
        """Generate price forecast for target date.

        Args:
            target_date: The date to forecast prices for

        Returns:
            PriceForecast with 96 buy/sell prices
        """
        pass

    @abstractmethod
    def save_forecast(self, forecast: PriceForecast, path: str) -> None:
        """Save forecast to CSV file."""
        pass


class IDAMBidder(ABC):
    """Interface for DAM bidding strategy."""

    @abstractmethod
    def generate_bids(
        self,
        forecast: PriceForecast,
        battery_state: BatteryState,
        constraints: dict
    ) -> DAMCommitment:
        """Generate DAM bids based on forecast and battery state.

        Args:
            forecast: Price forecast for target date
            battery_state: Current battery state
            constraints: Operational constraints (max cycles, etc.)

        Returns:
            DAMCommitment with 96 power values
        """
        pass


class ICommitmentExecutor(ABC):
    """Interface for DAM commitment execution."""

    @abstractmethod
    def get_current_commitment(self, timestamp: datetime) -> float:
        """Get committed power for current timestamp.

        Args:
            timestamp: Current time

        Returns:
            Committed power in MW (positive=sell, negative=buy)
        """
        pass

    @abstractmethod
    def load_commitment(self, commitment: DAMCommitment) -> None:
        """Load a commitment schedule."""
        pass

    @abstractmethod
    def get_remaining_capacity(self, timestamp: datetime) -> float:
        """Get capacity remaining after DAM commitment.

        Returns:
            Available power in MW for other markets
        """
        pass


class IBalancingTrader(ABC):
    """Interface for balancing market (aFRR + mFRR) trading."""

    @abstractmethod
    def decide_action(
        self,
        battery_state: BatteryState,
        available_capacity_mw: float,
        market_data: dict
    ) -> float:
        """Decide balancing market action.

        Args:
            battery_state: Current battery state
            available_capacity_mw: Capacity after DAM commitment
            market_data: Current aFRR/mFRR prices and signals

        Returns:
            Power in MW (positive=discharge/sell, negative=charge/buy)
        """
        pass


class IOrchestrator(ABC):
    """Interface for the main orchestrator."""

    @abstractmethod
    def run_day_ahead(self, target_date: date) -> DAMCommitment:
        """Run day-ahead process: forecast -> bid -> commitment."""
        pass

    @abstractmethod
    def run_realtime(self, timestamp: datetime) -> dict:
        """Run real-time process: execute commitment + balancing."""
        pass
