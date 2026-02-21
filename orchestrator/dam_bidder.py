"""
DAM Bidder - Strategy for DAM Bidding

I decide how to bid in the Day-Ahead Market based on:
- Price forecasts
- Battery state
- Operational constraints

Strategy: Buy at low prices, sell at high prices
while respecting cycle limits and SoC constraints.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional
from pathlib import Path

from .interfaces import (
    IDAMBidder,
    PriceForecast,
    DAMCommitment,
    BatteryState,
)


class DAMBidder(IDAMBidder):
    """DAM Bidding Strategy.

    I implement a threshold-based strategy:
    1. Identify low-price periods (buy/charge)
    2. Identify high-price periods (sell/discharge)
    3. Optimize schedule to maximize profit while respecting constraints
    """

    def __init__(
        self,
        max_power_mw: float = 30.0,
        capacity_mwh: float = 146.0,
        efficiency: float = 0.94,
        min_soc: float = 0.05,
        max_soc: float = 0.95,
        max_daily_cycles: float = 2.0,
        buy_threshold_percentile: float = 25,   # Buy when price < 25th percentile
        sell_threshold_percentile: float = 75,  # Sell when price > 75th percentile
        min_spread_eur: float = 30.0,           # Minimum spread to trade
        market_forecaster_path: Optional[str] = None,
        day_ahead_forecaster: Optional[object] = None,
    ):
        self.max_power_mw = max_power_mw
        self.capacity_mwh = capacity_mwh
        self.efficiency = efficiency
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.max_daily_cycles = max_daily_cycles
        self.buy_threshold_pct = buy_threshold_percentile
        self.sell_threshold_pct = sell_threshold_percentile
        self.min_spread = min_spread_eur

        # I store the DAMDayAheadForecaster for forecast_and_bid()
        self.day_ahead_forecaster = day_ahead_forecaster

        # I load MarketForecaster if path provided (Phase 4: enhanced DAM bidding)
        self.market_forecaster = None
        if market_forecaster_path is not None:
            fc_path = Path(market_forecaster_path)
            if fc_path.exists():
                try:
                    from models.market_forecaster import MarketForecaster
                    self.market_forecaster = MarketForecaster.load(str(fc_path))
                    print(f"  DAM Bidder: MarketForecaster loaded from {fc_path}")
                except Exception as e:
                    print(f"  DAM Bidder: Could not load MarketForecaster: {e}")

        print(f"DAM Bidder initialized")
        print(f"  Max power: {max_power_mw} MW, Capacity: {capacity_mwh} MWh")
        print(f"  Max daily cycles: {max_daily_cycles}")
        print(f"  Buy threshold: {buy_threshold_percentile}th percentile")
        print(f"  Sell threshold: {sell_threshold_percentile}th percentile")
        if self.day_ahead_forecaster:
            print(f"  Using DAMDayAheadForecaster for D+1 prediction")
        if self.market_forecaster:
            print(f"  Using MarketForecaster for enhanced price prediction")

    def generate_bids(
        self,
        forecast: PriceForecast,
        battery_state: BatteryState,
        constraints: Optional[dict] = None,
    ) -> DAMCommitment:
        """Generate DAM bids based on forecast.

        Strategy:
        1. Calculate price thresholds
        2. Identify buy/sell windows
        3. Schedule power to maximize profit
        4. Respect SoC and cycle constraints
        """
        constraints = constraints or {}

        # Get prices (use mid-price for decision, actual buy/sell for execution)
        mid_prices = (forecast.buy_prices + forecast.sell_prices) / 2

        # Calculate thresholds
        buy_threshold = np.percentile(mid_prices, self.buy_threshold_pct)
        sell_threshold = np.percentile(mid_prices, self.sell_threshold_pct)

        print(f"\nGenerating DAM bids for {forecast.target_date}")
        print(f"  Price range: {mid_prices.min():.1f} - {mid_prices.max():.1f} EUR")
        print(f"  Buy threshold: {buy_threshold:.1f} EUR")
        print(f"  Sell threshold: {sell_threshold:.1f} EUR")
        print(f"  Spread: {sell_threshold - buy_threshold:.1f} EUR")

        # Check if spread is sufficient
        if sell_threshold - buy_threshold < self.min_spread:
            print(f"  WARNING: Spread too small, reducing trading")
            # Tighten thresholds
            buy_threshold = np.percentile(mid_prices, 10)
            sell_threshold = np.percentile(mid_prices, 90)

        # Initialize commitment (0 = no action)
        power_mw = np.zeros(96)

        # Simulate SoC through the day
        soc = battery_state.soc
        cycles_used = 0.0

        # Energy per interval at max power (15 min = 0.25 hour)
        energy_per_interval = self.max_power_mw * 0.25  # MWh

        # Two-pass approach:
        # Pass 1: Identify best buy/sell opportunities
        # Pass 2: Execute while respecting constraints

        # Calculate profit potential for each interval
        profit_potential = np.zeros(96)
        for i in range(96):
            if mid_prices[i] <= buy_threshold:
                # Buy opportunity - profit when we sell later
                # Expected profit = (avg_sell_price - buy_price) * energy
                avg_sell = np.mean(mid_prices[mid_prices >= sell_threshold])
                profit_potential[i] = -(avg_sell - forecast.buy_prices[i])  # Negative = buy
            elif mid_prices[i] >= sell_threshold:
                # Sell opportunity - profit from selling
                # Actual profit = sell_price - avg_buy_price
                avg_buy = np.mean(mid_prices[mid_prices <= buy_threshold])
                profit_potential[i] = forecast.sell_prices[i] - avg_buy

        # Sort intervals by profitability
        buy_intervals = np.where(profit_potential < 0)[0]
        sell_intervals = np.where(profit_potential > 0)[0]

        # Sort by absolute profit potential
        buy_intervals = buy_intervals[np.argsort(profit_potential[buy_intervals])]
        sell_intervals = sell_intervals[np.argsort(-profit_potential[sell_intervals])]

        print(f"  Buy opportunities: {len(buy_intervals)} intervals")
        print(f"  Sell opportunities: {len(sell_intervals)} intervals")

        # Execute schedule
        # First pass: Schedule buys (charging)
        for i in buy_intervals:
            if cycles_used >= self.max_daily_cycles:
                break

            # Check SoC headroom
            available_headroom = (self.max_soc - soc) * self.capacity_mwh
            charge_energy = min(energy_per_interval, available_headroom)

            if charge_energy > 0.1:  # Minimum threshold
                charge_power = charge_energy / 0.25  # Convert back to MW
                power_mw[i] = -charge_power  # Negative = buy/charge

                # Update SoC (simplified, actual would track per interval)
                soc += charge_energy * self.efficiency / self.capacity_mwh
                cycles_used += charge_energy / self.capacity_mwh

        # Reset SoC for sell scheduling (we'll do proper simulation later)
        soc = battery_state.soc + np.sum(power_mw[power_mw < 0]) * -0.25 * self.efficiency / self.capacity_mwh

        # Second pass: Schedule sells (discharging)
        for i in sell_intervals:
            if cycles_used >= self.max_daily_cycles:
                break

            # Check SoC availability
            available_energy = (soc - self.min_soc) * self.capacity_mwh
            discharge_energy = min(energy_per_interval, available_energy)

            if discharge_energy > 0.1:
                discharge_power = discharge_energy / 0.25
                power_mw[i] = discharge_power  # Positive = sell/discharge

                soc -= discharge_energy / self.efficiency / self.capacity_mwh
                cycles_used += discharge_energy / self.capacity_mwh

        # Final validation: Simulate full day
        power_mw = self._validate_and_adjust(power_mw, battery_state.soc)

        # Summary
        total_buy = np.sum(power_mw[power_mw < 0]) * -0.25
        total_sell = np.sum(power_mw[power_mw > 0]) * 0.25
        print(f"\nCommitment summary:")
        print(f"  Total buy: {total_buy:.1f} MWh")
        print(f"  Total sell: {total_sell:.1f} MWh")
        print(f"  Cycles: {(total_buy + total_sell) / self.capacity_mwh:.2f}")

        return DAMCommitment(
            target_date=forecast.target_date,
            timestamps=forecast.timestamps,
            power_mw=power_mw,
        )

    def forecast_and_bid(
        self,
        target_date: date,
        df: pd.DataFrame,
        serbia_24h: np.ndarray,
        battery_state: BatteryState,
        sph: int = 1,
    ) -> tuple:
        """I forecast D+1 prices and generate bids in one step.

        Args:
            target_date: The delivery date (D+1)
            df: Historical Greek DAM DataFrame for feature computation
            serbia_24h: 24 hourly Serbia D+1 prices
            battery_state: Current battery state
            sph: Steps per hour (1=hourly, 4=15-min)

        Returns:
            Tuple of (PriceForecast, DAMCommitment)
        """
        from .dam_forecaster import HOURLY_GR_RS_PARAMS

        timestamps = []
        base_dt = datetime.combine(target_date, datetime.min.time())
        for i in range(96):
            timestamps.append(base_dt + timedelta(minutes=15 * i))

        # I find the day index in the historical data
        day_mask = df.index.normalize() == pd.Timestamp(target_date)
        day_indices = np.where(day_mask.values)[0] if hasattr(day_mask, 'values') else np.where(day_mask)[0]
        day_idx = day_indices[0] if len(day_indices) > 0 else len(df) - 24 * sph

        if (
            self.day_ahead_forecaster is not None
            and self.day_ahead_forecaster.is_trained
        ):
            result = self.day_ahead_forecaster.predict_96qh(df, day_idx, serbia_24h, sph)
            prices_15min = result['qh']
        else:
            # I use Serbia regression fallback
            hourly = np.zeros(24)
            for h in range(24):
                params = HOURLY_GR_RS_PARAMS.get(h, {'slope': 0.7, 'intercept': 20.0})
                hourly[h] = max(0.0, params['slope'] * serbia_24h[h] + params['intercept'])
            prices_15min = np.repeat(hourly, 4)

        spread_pct = 0.02
        half_spread = prices_15min * spread_pct / 2
        buy_prices = prices_15min + half_spread
        sell_prices = prices_15min - half_spread

        forecast = PriceForecast(
            target_date=target_date,
            timestamps=timestamps,
            buy_prices=buy_prices,
            sell_prices=sell_prices,
        )

        commitment = self.generate_bids(forecast, battery_state)
        return forecast, commitment

    def _validate_and_adjust(
        self,
        power_mw: np.ndarray,
        initial_soc: float,
    ) -> np.ndarray:
        """Validate schedule against SoC constraints and adjust if needed."""
        soc = initial_soc
        adjusted = power_mw.copy()

        for i in range(96):
            power = adjusted[i]
            energy = abs(power) * 0.25  # MWh

            if power < 0:  # Charging
                # Check headroom
                headroom = (self.max_soc - soc) * self.capacity_mwh
                if energy > headroom:
                    # Reduce charging
                    adjusted[i] = -headroom / 0.25 if headroom > 0.1 else 0
                    energy = headroom
                soc += energy * self.efficiency / self.capacity_mwh

            elif power > 0:  # Discharging
                # Check available energy
                available = (soc - self.min_soc) * self.capacity_mwh
                if energy > available:
                    # Reduce discharging
                    adjusted[i] = available / 0.25 if available > 0.1 else 0
                    energy = available
                soc -= energy / self.efficiency / self.capacity_mwh

            # Clamp SoC
            soc = np.clip(soc, self.min_soc, self.max_soc)

        return adjusted


# Test
if __name__ == "__main__":
    from .dam_forecaster import DAMForecaster
    from datetime import timedelta

    # Create forecaster and bidder
    forecaster = DAMForecaster()
    bidder = DAMBidder()

    # Get forecast
    tomorrow = date.today() + timedelta(days=1)
    forecast = forecaster.forecast(tomorrow)

    # Current battery state
    battery = BatteryState(
        timestamp=datetime.now(),
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    # Generate bids
    commitment = bidder.generate_bids(forecast, battery)

    print(f"\n15-min Power Schedule (non-zero):")
    for i, (ts, power) in enumerate(zip(commitment.timestamps, commitment.power_mw)):
        if abs(power) > 0.1:
            action = "SELL" if power > 0 else "BUY"
            print(f"  {ts.strftime('%H:%M')}: {action} {abs(power):.1f} MW")
