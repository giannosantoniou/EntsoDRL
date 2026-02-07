"""
Backtesting Engine for mFRR Bidding Strategies

I simulate battery operation over historical data and calculate profits.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .battery import Battery, BatteryConfig
from .strategies import Strategy, MarketState


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: pd.Timestamp
    action: str  # 'discharge', 'charge', 'idle'
    power_mw: float
    price_eur_mwh: float
    revenue_eur: float
    degradation_eur: float
    net_profit_eur: float
    soc_before: float
    soc_after: float


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    strategy_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_hours: int

    # Financial metrics
    total_revenue: float = 0.0
    total_degradation_cost: float = 0.0
    total_net_profit: float = 0.0

    # Operational metrics
    total_discharge_mwh: float = 0.0
    total_charge_mwh: float = 0.0
    total_cycles: float = 0.0
    discharge_hours: int = 0
    charge_hours: int = 0
    idle_hours: int = 0

    # Performance metrics
    avg_discharge_price: float = 0.0
    avg_charge_price: float = 0.0
    profit_per_cycle: float = 0.0
    capacity_utilization: float = 0.0

    # Trade history
    trades: List[TradeRecord] = field(default_factory=list)

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 60,
            f"Strategy: {self.strategy_name}",
            "=" * 60,
            f"Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Total hours: {self.total_hours:,}",
            "",
            "FINANCIAL RESULTS",
            "-" * 40,
            f"Total Revenue:        {self.total_revenue:>12,.0f} EUR",
            f"Degradation Cost:     {self.total_degradation_cost:>12,.0f} EUR",
            f"NET PROFIT:           {self.total_net_profit:>12,.0f} EUR",
            "",
            f"Profit/hour:          {self.total_net_profit/max(self.total_hours,1):>12,.2f} EUR",
            f"Profit/cycle:         {self.profit_per_cycle:>12,.2f} EUR",
            "",
            "OPERATIONAL METRICS",
            "-" * 40,
            f"Discharge hours:      {self.discharge_hours:>12,} ({100*self.discharge_hours/max(self.total_hours,1):.1f}%)",
            f"Charge hours:         {self.charge_hours:>12,} ({100*self.charge_hours/max(self.total_hours,1):.1f}%)",
            f"Idle hours:           {self.idle_hours:>12,} ({100*self.idle_hours/max(self.total_hours,1):.1f}%)",
            "",
            f"Total discharge:      {self.total_discharge_mwh:>12,.0f} MWh",
            f"Total charge:         {self.total_charge_mwh:>12,.0f} MWh",
            f"Full cycles:          {self.total_cycles:>12,.1f}",
            "",
            f"Avg discharge price:  {self.avg_discharge_price:>12,.2f} EUR/MWh",
            f"Avg charge price:     {self.avg_charge_price:>12,.2f} EUR/MWh",
            f"Capacity utilization: {self.capacity_utilization*100:>12,.1f}%",
            "=" * 60,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    I run backtests of bidding strategies on historical data.
    """

    def __init__(self, data: pd.DataFrame, battery_config: BatteryConfig = None):
        """
        Initialize backtester.

        Args:
            data: DataFrame with mFRR prices and features
            battery_config: Battery configuration
        """
        self.data = data.copy()
        self.battery_config = battery_config or BatteryConfig()

        # Ensure datetime index
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        # Sort by time
        self.data = self.data.sort_index()

    def run(self, strategy: Strategy, start_date: str = None,
            end_date: str = None, verbose: bool = False) -> BacktestResult:
        """
        Run backtest for a strategy.

        Args:
            strategy: Bidding strategy to test
            start_date: Start date (None = use all data)
            end_date: End date (None = use all data)
            verbose: Print progress

        Returns:
            BacktestResult with performance metrics
        """
        # Filter data by date
        data = self.data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        if len(data) == 0:
            raise ValueError("No data in specified date range")

        # Initialize battery
        battery = Battery(self.battery_config)

        # Initialize result
        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=data.index.min(),
            end_date=data.index.max(),
            total_hours=len(data)
        )

        # Track for averages
        discharge_prices = []
        charge_prices = []

        # Run simulation
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Build market state
            state = self._build_state(timestamp, row, battery)

            # Get strategy decision
            action, power = strategy.decide(state)

            # Get price for this action
            if action == 'discharge':
                price = row.get('mfrr_price_up', 0)
            elif action == 'charge':
                price = row.get('mfrr_price_down', 0)
            else:
                price = 0

            if pd.isna(price):
                price = 0

            # Execute action
            soc_before = battery.soc
            actual_power, degradation = battery.step(power, duration_hours=1.0)
            soc_after = battery.soc

            # Calculate revenue
            # Discharge: we sell power, receive money (positive revenue)
            # Charge: we buy power, pay money (negative revenue, but we use down price)
            if action == 'discharge':
                revenue = actual_power * price  # positive
                result.total_discharge_mwh += actual_power
                result.discharge_hours += 1
                if actual_power > 0:
                    discharge_prices.append(price)
            elif action == 'charge':
                # When charging, we're providing down-regulation (absorbing excess)
                # We receive the down-regulation price
                revenue = abs(actual_power) * price  # down-reg pays us
                result.total_charge_mwh += abs(actual_power)
                result.charge_hours += 1
                if actual_power != 0:
                    charge_prices.append(price)
            else:
                revenue = 0
                result.idle_hours += 1

            net_profit = revenue - degradation

            # Record trade
            trade = TradeRecord(
                timestamp=timestamp,
                action=action,
                power_mw=actual_power,
                price_eur_mwh=price,
                revenue_eur=revenue,
                degradation_eur=degradation,
                net_profit_eur=net_profit,
                soc_before=soc_before,
                soc_after=soc_after
            )
            result.trades.append(trade)

            # Update totals
            result.total_revenue += revenue
            result.total_degradation_cost += degradation
            result.total_net_profit += net_profit

            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(data)} hours...")

        # Calculate summary metrics
        result.total_cycles = result.total_discharge_mwh / self.battery_config.capacity_mwh
        result.avg_discharge_price = np.mean(discharge_prices) if discharge_prices else 0
        result.avg_charge_price = np.mean(charge_prices) if charge_prices else 0
        result.profit_per_cycle = result.total_net_profit / max(result.total_cycles, 0.01)
        result.capacity_utilization = (result.discharge_hours + result.charge_hours) / max(result.total_hours, 1)

        return result

    def _build_state(self, timestamp: pd.Timestamp, row: pd.Series,
                     battery: Battery) -> MarketState:
        """Build MarketState from data row."""
        can_discharge, can_charge = battery.get_action_mask()

        return MarketState(
            timestamp=timestamp,
            hour=timestamp.hour,
            day_of_week=timestamp.dayofweek,
            is_weekend=timestamp.dayofweek >= 5,
            mfrr_price_up=row.get('mfrr_price_up'),
            mfrr_price_down=row.get('mfrr_price_down'),
            mfrr_spread=row.get('mfrr_spread'),
            price_up_lag_1h=row.get('price_up_lag_1h'),
            price_up_lag_24h=row.get('price_up_lag_24h'),
            price_up_mean_24h=row.get('price_up_mean_24h'),
            net_imbalance_mw=row.get('net_imbalance_mw'),
            load_mw=row.get('load_mw'),
            res_total_mw=row.get('res_total_mw'),
            net_cross_border_mw=row.get('net_cross_border_mw'),
            soc=battery.soc,
            can_discharge=can_discharge,
            can_charge=can_charge
        )

    def compare_strategies(self, strategies: List[Strategy],
                           start_date: str = None, end_date: str = None,
                           verbose: bool = True) -> pd.DataFrame:
        """
        Compare multiple strategies.

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for strategy in strategies:
            if verbose:
                print(f"\nRunning {strategy.name}...")
            result = self.run(strategy, start_date, end_date, verbose=False)
            results.append(result)
            if verbose:
                print(f"  Net Profit: {result.total_net_profit:,.0f} EUR")

        # Build comparison DataFrame
        comparison = pd.DataFrame([
            {
                'Strategy': r.strategy_name,
                'Net Profit (EUR)': r.total_net_profit,
                'Revenue (EUR)': r.total_revenue,
                'Degradation (EUR)': r.total_degradation_cost,
                'Cycles': r.total_cycles,
                'Profit/Cycle': r.profit_per_cycle,
                'Discharge Hours': r.discharge_hours,
                'Charge Hours': r.charge_hours,
                'Avg Discharge Price': r.avg_discharge_price,
                'Avg Charge Price': r.avg_charge_price,
                'Utilization %': r.capacity_utilization * 100
            }
            for r in results
        ])

        return comparison.sort_values('Net Profit (EUR)', ascending=False)
