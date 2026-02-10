"""
Battery Trading Orchestrator - Main Controller

I orchestrate the complete battery trading workflow:

Day-Ahead (D-1):
1. Get price forecast from EntsoE3
2. Generate DAM bids based on forecast
3. Store commitment for execution

Real-Time (D-0):
1. Execute DAM commitment (binding)
2. Calculate remaining capacity
3. Participate in balancing with available capacity

This is the main entry point for the trading system.
"""

import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict
from pathlib import Path

from .interfaces import (
    IOrchestrator,
    DAMCommitment,
    BatteryState,
)
from .dam_forecaster import DAMForecaster
from .dam_bidder import DAMBidder
from .commitment_executor import CommitmentExecutor


class BatteryOrchestrator(IOrchestrator):
    """Main orchestrator for battery trading.

    I coordinate all subsystems:
    - DAM forecasting and bidding
    - Commitment execution
    - Balancing market participation (future)

    Usage:
        orchestrator = BatteryOrchestrator()

        # Day-ahead process (run at 10:00 D-1)
        commitment = orchestrator.run_day_ahead(target_date)

        # Real-time loop (run every 15 min on D-0)
        result = orchestrator.run_realtime(current_time)
    """

    def __init__(
        self,
        max_power_mw: float = 30.0,
        capacity_mwh: float = 146.0,
        efficiency: float = 0.94,
        commitment_dir: Optional[str] = None,
        # Phase 2: Balancing configuration
        balancing_enabled: bool = False,
        balancing_model_path: Optional[str] = None,
        balancing_strategy: str = "RULE_BASED",
        afrr_enabled: bool = True,
        mfrr_enabled: bool = True,
        # Phase 3: IntraDay configuration
        intraday_enabled: bool = False,
        intraday_min_spread: float = 5.0,
    ):
        """Initialize the orchestrator.

        Args:
            max_power_mw: Maximum battery power
            capacity_mwh: Battery capacity
            efficiency: Round-trip efficiency
            commitment_dir: Directory to store commitment files
            balancing_enabled: Enable balancing market participation
            balancing_model_path: Path to DRL model for mFRR (if AI strategy)
            balancing_strategy: Strategy type (AI, RULE_BASED, AGGRESSIVE_HYBRID)
            afrr_enabled: Enable aFRR capacity bidding
            mfrr_enabled: Enable mFRR energy trading
            intraday_enabled: Enable IntraDay trading
            intraday_min_spread: Minimum spread for IntraDay trades
        """
        self.max_power_mw = max_power_mw
        self.capacity_mwh = capacity_mwh
        self.efficiency = efficiency
        self.commitment_dir = Path(commitment_dir or "commitments")
        self.commitment_dir.mkdir(exist_ok=True)

        # Initialize subsystems
        self.forecaster = DAMForecaster()
        self.bidder = DAMBidder(
            max_power_mw=max_power_mw,
            capacity_mwh=capacity_mwh,
            efficiency=efficiency,
        )
        self.executor = CommitmentExecutor(
            max_power_mw=max_power_mw,
            capacity_mwh=capacity_mwh,
        )

        # Current battery state (should be updated from SCADA)
        self._battery_state = BatteryState(
            timestamp=datetime.now(),
            soc=0.5,  # Default 50%
            available_power_mw=max_power_mw,
            capacity_mwh=capacity_mwh,
        )

        # Balancing trader (Phase 2)
        self._balancing_trader = None
        self._market_data = {}  # Real-time market data cache

        if balancing_enabled:
            self._setup_balancing(
                model_path=balancing_model_path,
                strategy=balancing_strategy,
                afrr_enabled=afrr_enabled,
                mfrr_enabled=mfrr_enabled,
            )

        # IntraDay trader (Phase 3)
        self._intraday_trader = None
        self._intraday_prices = None  # Current IntraDay prices

        if intraday_enabled:
            self._setup_intraday(min_spread=intraday_min_spread)

        print(f"\n{'='*60}")
        print(f"Battery Trading Orchestrator Initialized")
        print(f"{'='*60}")
        print(f"  Battery: {capacity_mwh} MWh / {max_power_mw} MW")
        print(f"  Efficiency: {efficiency:.0%}")
        print(f"  Balancing: {'ENABLED' if self._balancing_trader else 'DISABLED'}")
        print(f"  IntraDay: {'ENABLED' if self._intraday_trader else 'DISABLED'}")
        print(f"  Commitment storage: {self.commitment_dir}")

    def _setup_balancing(
        self,
        model_path: Optional[str],
        strategy: str,
        afrr_enabled: bool,
        mfrr_enabled: bool,
    ) -> None:
        """Setup balancing trader (Phase 2).

        I initialize the balancing trader with the specified
        strategy and market participation settings.
        """
        from .balancing_trader import BalancingTrader

        self._balancing_trader = BalancingTrader(
            model_path=model_path,
            strategy_type=strategy,
            n_actions=21,
            max_power_mw=self.max_power_mw,
            capacity_mwh=self.capacity_mwh,
            afrr_enabled=afrr_enabled,
            mfrr_enabled=mfrr_enabled,
        )

    def _setup_intraday(self, min_spread: float) -> None:
        """Setup IntraDay trader (Phase 3).

        I initialize the IntraDay trader for position adjustments.
        """
        from .intraday_trader import IntraDayTrader

        self._intraday_trader = IntraDayTrader(
            max_power_mw=self.max_power_mw,
            capacity_mwh=self.capacity_mwh,
            min_spread_eur=min_spread,
        )

    def update_intraday_prices(self, prices: np.ndarray) -> None:
        """Update current IntraDay market prices.

        I should be called with live IntraDay price data.

        Args:
            prices: 96 IntraDay prices (15-min resolution)
        """
        self._intraday_prices = prices

    def run_intraday(
        self,
        current_time: datetime,
        auto_execute: bool = True,
        min_profit_eur: float = 10.0,
    ) -> Dict:
        """Run IntraDay position optimization and execution.

        I evaluate adjustment opportunities and optionally execute them.
        This should be called periodically during IntraDay trading window
        (typically D-1 14:00 to D-0 H-1).

        Args:
            current_time: Current timestamp
            auto_execute: Whether to automatically execute orders
            min_profit_eur: Minimum profit threshold for execution

        Returns:
            Dict with opportunities, executed orders, and P&L
        """
        if self._intraday_trader is None:
            return {'enabled': False, 'opportunities': []}

        if self._intraday_prices is None:
            return {'enabled': True, 'opportunities': [], 'error': 'no_prices'}

        # Find opportunities
        opportunities = self._intraday_trader.get_adjustment_opportunities(
            intraday_prices=self._intraday_prices,
            battery_state=self._battery_state,
        )

        # Execute if enabled
        executed_orders = []
        if auto_execute and opportunities:
            executed_orders = self._intraday_trader.execute_opportunities(
                intraday_prices=self._intraday_prices,
                battery_state=self._battery_state,
                current_time=current_time,
                min_profit_eur=min_profit_eur,
            )

        # Get session summary
        session_summary = self._intraday_trader.get_session_summary()

        return {
            'enabled': True,
            'timestamp': current_time,
            'total_opportunities': len(opportunities),
            'top_opportunities': opportunities[:5],
            'executed_orders': len(executed_orders),
            'executed_details': [
                {
                    'delivery': o.delivery_start.strftime('%H:%M'),
                    'power_mw': o.power_mw,
                    'fill_price': o.fill_price,
                    'pnl_eur': o.pnl_eur,
                }
                for o in executed_orders
            ],
            'session_pnl_eur': session_summary.get('total_pnl_eur', 0),
            'position_summary': self._intraday_trader.get_position_summary(
                current_time.date() if hasattr(current_time, 'date') else current_time
            ),
        }

    def start_intraday_session(self, delivery_date: date) -> Dict:
        """Start IntraDay trading session for a delivery date.

        Call this after DAM results are known (typically D-1 14:00).

        Args:
            delivery_date: The delivery date

        Returns:
            Session info
        """
        if self._intraday_trader is None:
            return {'enabled': False}

        session = self._intraday_trader.start_session(delivery_date)
        return {
            'enabled': True,
            'delivery_date': session.delivery_date,
            'session_start': session.session_start,
        }

    def close_intraday_session(self) -> Dict:
        """Close IntraDay session and get final P&L.

        Returns:
            Session summary with P&L
        """
        if self._intraday_trader is None:
            return {'enabled': False}

        return self._intraday_trader.close_session()

    def get_net_position(self, delivery_time: datetime) -> float:
        """Get net position for delivery period (DAM + IntraDay).

        Args:
            delivery_time: Delivery period start

        Returns:
            Net position in MW
        """
        if self._intraday_trader is None:
            # No IntraDay, return DAM commitment
            return self.executor.get_commitment(delivery_time)

        return self._intraday_trader.get_net_position(delivery_time)

    def update_market_data(self, market_data: Dict) -> None:
        """Update real-time market data for balancing decisions.

        I should be called with live market data feeds.

        Args:
            market_data: Dict with mfrr_price_up, mfrr_price_down,
                        afrr_up_price, afrr_down_price, etc.
        """
        self._market_data.update(market_data)

    def update_battery_state(self, state: BatteryState) -> None:
        """Update current battery state.

        I should be called regularly with SCADA data.
        """
        self._battery_state = state

    def run_day_ahead(self, target_date: date) -> DAMCommitment:
        """Run the day-ahead process.

        I execute the complete D-1 workflow:
        1. Generate price forecast
        2. Create optimal bids
        3. Store commitment

        Args:
            target_date: The delivery date (D-0)

        Returns:
            DAMCommitment for the target date
        """
        print(f"\n{'='*60}")
        print(f"DAY-AHEAD PROCESS for {target_date}")
        print(f"{'='*60}")

        # Step 1: Get price forecast
        print(f"\n[1/3] Generating price forecast...")
        forecast = self.forecaster.forecast(target_date)

        # Save forecast for audit trail
        forecast_path = self.commitment_dir / f"forecast_{target_date}.csv"
        self.forecaster.save_forecast(forecast, str(forecast_path))

        # Step 2: Generate bids
        print(f"\n[2/3] Generating DAM bids...")
        commitment = self.bidder.generate_bids(
            forecast=forecast,
            battery_state=self._battery_state,
        )

        # Step 3: Store commitment
        print(f"\n[3/3] Storing commitment...")
        self._save_commitment(commitment)
        self.executor.load_commitment(commitment)

        # Initialize IntraDay trader with commitment (Phase 3)
        if self._intraday_trader is not None:
            # Use mid-price for IntraDay reference
            dam_prices = (forecast.buy_prices + forecast.sell_prices) / 2
            self._intraday_trader.load_dam_commitment(commitment, dam_prices)
            print(f"  IntraDay trader initialized with DAM positions")

        # Summary
        print(f"\n{'='*60}")
        print(f"DAY-AHEAD COMPLETE")
        print(f"{'='*60}")
        print(f"  Buy:  {commitment.total_buy_mwh:6.1f} MWh")
        print(f"  Sell: {commitment.total_sell_mwh:6.1f} MWh")
        print(f"  Net:  {commitment.total_sell_mwh - commitment.total_buy_mwh:+6.1f} MWh")

        # Estimate profit (rough)
        buy_cost = np.sum(
            forecast.buy_prices[commitment.power_mw < 0] *
            np.abs(commitment.power_mw[commitment.power_mw < 0]) * 0.25
        )
        sell_revenue = np.sum(
            forecast.sell_prices[commitment.power_mw > 0] *
            commitment.power_mw[commitment.power_mw > 0] * 0.25
        )
        estimated_profit = sell_revenue - buy_cost
        print(f"  Estimated profit: {estimated_profit:+.0f} EUR")

        return commitment

    def run_realtime(self, timestamp: datetime) -> Dict:
        """Run the real-time process.

        I execute the D-0 workflow for the current interval:
        1. Get DAM commitment for this interval
        2. Calculate available capacity for balancing
        3. Decide balancing action (if trader available)
        4. Return dispatch command

        Args:
            timestamp: Current time

        Returns:
            Dict with dispatch commands and status
        """
        # Get current commitment
        dam_commitment = self.executor.get_current_commitment(timestamp)
        direction = self.executor.get_commitment_direction(timestamp)
        remaining_capacity = self.executor.get_remaining_capacity(timestamp)

        # Check balancing availability
        balance_cap = self.executor.can_participate_in_balancing(
            timestamp, self._battery_state
        )

        # Decide balancing action (Phase 2)
        balancing_action = 0.0
        afrr_up = 0.0
        afrr_down = 0.0
        if self._balancing_trader is not None:
            # Use balancing trader for full decision
            decision = self._balancing_trader.decide_full(
                battery_state=self._battery_state,
                available_capacity_mw=remaining_capacity,
                market_data=self._market_data,
                dam_commitment_mw=dam_commitment,
            )
            balancing_action = decision.mfrr_action_mw
            afrr_up = decision.afrr_up_capacity_mw
            afrr_down = decision.afrr_down_capacity_mw

        # Calculate total dispatch
        total_dispatch = dam_commitment + balancing_action

        # Build result
        result = {
            'timestamp': timestamp,
            'interval': timestamp.strftime('%H:%M'),

            # DAM execution
            'dam_commitment_mw': dam_commitment,
            'dam_direction': direction,

            # Balancing - capacity
            'remaining_capacity_mw': remaining_capacity,
            'up_capacity_mw': balance_cap['up_capacity_mw'],
            'down_capacity_mw': balance_cap['down_capacity_mw'],

            # aFRR (Phase 2)
            'afrr_up_reserved_mw': afrr_up,
            'afrr_down_reserved_mw': afrr_down,

            # mFRR (Phase 2)
            'mfrr_action_mw': balancing_action,

            # Total dispatch
            'total_dispatch_mw': total_dispatch,
            'dispatch_direction': 'discharge' if total_dispatch > 0 else 'charge' if total_dispatch < 0 else 'idle',

            # State
            'soc': self._battery_state.soc,
        }

        return result

    def execute_interval(
        self,
        timestamp: datetime,
        actual_power_mw: float,
    ) -> Dict:
        """Record execution of an interval.

        I should be called after actual dispatch to record
        what was really delivered (for settlement and audit).

        Args:
            timestamp: Interval timestamp
            actual_power_mw: Actual power delivered

        Returns:
            Dict with execution status
        """
        # Record execution
        status = self.executor.record_execution(timestamp, actual_power_mw)

        # Update SoC estimate
        energy_mwh = actual_power_mw * 0.25
        if actual_power_mw > 0:  # Discharging
            new_soc = self._battery_state.soc - energy_mwh / self.capacity_mwh
        else:  # Charging
            new_soc = self._battery_state.soc + abs(energy_mwh) * self.efficiency / self.capacity_mwh

        new_soc = np.clip(new_soc, 0.05, 0.95)

        self._battery_state = BatteryState(
            timestamp=timestamp,
            soc=new_soc,
            available_power_mw=self.max_power_mw,
            capacity_mwh=self.capacity_mwh,
        )

        return {
            'timestamp': timestamp,
            'committed_mw': status.committed_mw,
            'executed_mw': status.executed_mw,
            'deviation_mw': status.deviation_mw,
            'new_soc': new_soc,
        }

    def get_daily_summary(self, target_date: date) -> Optional[Dict]:
        """Get summary of execution for a day."""
        return self.executor.get_execution_summary(target_date)

    def load_commitment_from_file(self, target_date: date) -> bool:
        """Load commitment from file (for recovery/restart).

        Returns:
            True if loaded successfully
        """
        path = self.commitment_dir / f"commitment_{target_date}.npz"
        if not path.exists():
            print(f"No commitment file found for {target_date}")
            return False

        data = np.load(path)

        # Reconstruct timestamps
        base_dt = datetime.combine(target_date, datetime.min.time())
        timestamps = [base_dt + timedelta(minutes=15 * i) for i in range(96)]

        commitment = DAMCommitment(
            target_date=target_date,
            timestamps=timestamps,
            power_mw=data['power_mw'],
        )

        self.executor.load_commitment(commitment)
        print(f"Loaded commitment for {target_date}")
        return True

    def _save_commitment(self, commitment: DAMCommitment) -> None:
        """Save commitment to file."""
        path = self.commitment_dir / f"commitment_{commitment.target_date}.npz"
        np.savez(
            path,
            target_date=str(commitment.target_date),
            power_mw=commitment.power_mw,
        )
        print(f"  Saved: {path}")

    def set_balancing_trader(self, trader) -> None:
        """Set the balancing market trader (Phase 2).

        Args:
            trader: Instance implementing IBalancingTrader
        """
        self._balancing_trader = trader
        print(f"Balancing trader set: {type(trader).__name__}")


# Test
if __name__ == "__main__":
    # Create orchestrator
    orchestrator = BatteryOrchestrator()

    # Simulate day-ahead process
    tomorrow = date.today() + timedelta(days=1)
    commitment = orchestrator.run_day_ahead(tomorrow)

    # Simulate real-time execution
    print(f"\n{'='*60}")
    print(f"REAL-TIME SIMULATION")
    print(f"{'='*60}")

    # Run through a few hours
    for hour in range(6, 22, 2):
        for quarter in [0, 30]:
            ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=hour, minutes=quarter)

            # Get dispatch command
            result = orchestrator.run_realtime(ts)

            # Simulate execution (with small random deviation)
            actual = result['total_dispatch_mw'] + np.random.uniform(-0.2, 0.2)

            # Record execution
            exec_result = orchestrator.execute_interval(ts, actual)

            if abs(result['dam_commitment_mw']) > 0.1:
                print(f"{ts.strftime('%H:%M')}: "
                      f"DAM={result['dam_commitment_mw']:+5.1f} MW ({result['dam_direction']}) | "
                      f"SoC={exec_result['new_soc']:.1%}")

    # Get daily summary
    print(f"\n{'='*60}")
    print(f"DAILY SUMMARY")
    print(f"{'='*60}")
    summary = orchestrator.get_daily_summary(tomorrow)
    if summary:
        print(f"  Intervals executed: {summary['intervals_executed']}")
        print(f"  Total deviation: {summary['total_deviation_mwh']:.2f} MWh")
        print(f"  Max deviation: {summary['max_deviation_mw']:.2f} MW")
        print(f"  Execution accuracy: {summary['execution_accuracy']:.1%}")
