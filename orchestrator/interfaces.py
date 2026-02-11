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


@dataclass
class BalancingDecision:
    """Decision for balancing market (aFRR + mFRR) trading.

    Positive values = discharge/sell
    Negative values = charge/buy
    """
    timestamp: datetime
    afrr_commitment_mw: float  # aFRR capacity commitment
    afrr_price_tier: int       # Bid aggressiveness (0-4)
    mfrr_power_mw: float       # mFRR energy trading


@dataclass
class UnifiedDecision:
    """
    Decision for the unified multi-market trading model.

    I combine DAM execution, aFRR commitment, and mFRR/IntraDay trading
    into a single decision structure. This extends BalancingDecision
    to include DAM compliance tracking.

    Action Interpretation:
    - afrr_commitment_level: 0-4 (0%, 25%, 50%, 75%, 100% of remaining capacity)
    - afrr_price_tier: 0-4 (aggressive to conservative bidding)
    - energy_action: 0-20 (-30 to +30 MW energy trading)
    """
    timestamp: datetime

    # aFRR decisions
    afrr_commitment_level: int  # Index 0-4
    afrr_commitment_mw: float   # Actual MW committed
    afrr_price_tier: int        # Index 0-4

    # Energy trading
    energy_action: int          # Index 0-20
    energy_power_mw: float      # Actual MW traded

    # DAM context (for compliance tracking)
    dam_commitment_mw: float    # Current DAM commitment
    remaining_capacity_mw: float  # Capacity after DAM

    # Execution results
    is_selected_for_afrr: bool = False
    afrr_activated: bool = False
    afrr_activation_direction: Optional[str] = None
    actual_energy_mw: float = 0.0

    @property
    def total_power_mw(self) -> float:
        """I return the total power including DAM and trading."""
        return self.dam_commitment_mw + self.actual_energy_mw


class IUnifiedTrader(ABC):
    """Interface for unified multi-market trading."""

    @abstractmethod
    def decide_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        battery_state: BatteryState,
        dam_commitment_mw: float,
        market_data: dict
    ) -> UnifiedDecision:
        """
        I decide on unified multi-market action.

        Args:
            observation: 58-feature observation vector
            action_mask: Boolean mask for valid actions
            battery_state: Current battery state
            dam_commitment_mw: Current DAM commitment
            market_data: Current market prices and signals

        Returns:
            UnifiedDecision with aFRR commitment and energy trading
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str, normalizer_path: Optional[str] = None) -> None:
        """Load the trained unified model."""
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


class IUnifiedOrchestrator(IOrchestrator):
    """Interface for the unified multi-market orchestrator."""

    @abstractmethod
    def run_unified_step(self, timestamp: datetime) -> UnifiedDecision:
        """
        I execute a single unified trading step.

        This combines DAM execution, aFRR commitment, and mFRR trading
        into a single coordinated decision.

        Returns:
            UnifiedDecision with all market actions
        """
        pass

    @abstractmethod
    def get_current_state(self) -> dict:
        """I return the current state of all markets."""
        pass
