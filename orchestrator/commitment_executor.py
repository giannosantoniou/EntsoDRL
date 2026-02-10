"""
Commitment Executor - DAM Commitment Tracking and Execution

I manage the execution of DAM commitments:
- Load and store daily commitment schedules
- Track execution status per interval
- Calculate remaining capacity for balancing markets
- Handle deviations and penalties

The battery MUST honor DAM commitments - they are binding contracts.
"""

import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from .interfaces import (
    ICommitmentExecutor,
    DAMCommitment,
    BatteryState,
)


@dataclass
class ExecutionStatus:
    """Status of commitment execution for a single interval."""
    timestamp: datetime
    dam_committed_mw: float   # DAM commitment (binding contract)
    mfrr_planned_mw: float    # mFRR action (intentional deviation)
    total_planned_mw: float   # Combined plan (DAM + mFRR)
    executed_mw: float        # What we actually did
    deviation_mw: float       # Difference (executed - total_planned)
    is_executed: bool         # Whether this interval is complete

    # Legacy alias for backwards compatibility
    @property
    def committed_mw(self) -> float:
        return self.dam_committed_mw


@dataclass
class DailyExecutionLog:
    """Log of execution for a full day."""
    target_date: date
    intervals: List[ExecutionStatus] = field(default_factory=list)

    @property
    def total_deviation_mwh(self) -> float:
        """Total energy deviation from plan (positive = over-delivered)."""
        return sum(s.deviation_mw for s in self.intervals) * 0.25

    @property
    def execution_accuracy(self) -> float:
        """Percentage of intervals executed within tolerance of TOTAL PLAN (DAM + mFRR).

        This measures how accurately the battery followed the combined dispatch plan,
        NOT how closely it followed DAM alone (which would be wrong since mFRR is intentional).
        """
        if not self.intervals:
            return 0.0
        within_tolerance = sum(1 for s in self.intervals if abs(s.deviation_mw) < 0.5)
        return within_tolerance / len(self.intervals)

    @property
    def dam_deviation_mwh(self) -> float:
        """Total energy deviation from DAM commitment (for settlement purposes)."""
        return sum(s.executed_mw - s.dam_committed_mw for s in self.intervals) * 0.25

    @property
    def mfrr_total_mwh(self) -> float:
        """Total mFRR energy traded (absolute value)."""
        return sum(abs(s.mfrr_planned_mw) for s in self.intervals) * 0.25

    @property
    def active_mfrr_intervals(self) -> int:
        """Number of intervals with mFRR activity."""
        return sum(1 for s in self.intervals if abs(s.mfrr_planned_mw) > 0.1)


class CommitmentExecutor(ICommitmentExecutor):
    """DAM Commitment Executor.

    I track DAM commitments and ensure the battery honors them.
    Any remaining capacity can be used for balancing markets.

    Key responsibilities:
    1. Store daily commitment schedule
    2. Provide current commitment at any timestamp
    3. Calculate available capacity for other markets
    4. Log execution and track deviations
    """

    def __init__(
        self,
        max_power_mw: float = 30.0,
        capacity_mwh: float = 146.0,
        deviation_tolerance_mw: float = 0.5,  # Acceptable deviation
    ):
        self.max_power_mw = max_power_mw
        self.capacity_mwh = capacity_mwh
        self.deviation_tolerance = deviation_tolerance_mw

        # Current commitment (loaded daily)
        self._commitment: Optional[DAMCommitment] = None

        # Execution log
        self._execution_log: Dict[date, DailyExecutionLog] = {}

        # Current day tracking
        self._current_interval_idx: int = 0

        print(f"Commitment Executor initialized")
        print(f"  Max power: {max_power_mw} MW")
        print(f"  Deviation tolerance: {deviation_tolerance_mw} MW")

    def load_commitment(self, commitment: DAMCommitment) -> None:
        """Load a new commitment schedule.

        I store the commitment and prepare the execution log.
        This should be called after DAM results are published (D-1 afternoon).
        """
        self._commitment = commitment
        self._current_interval_idx = 0

        # Initialize execution log for this day
        self._execution_log[commitment.target_date] = DailyExecutionLog(
            target_date=commitment.target_date,
            intervals=[]
        )

        # Summary
        buy_intervals = np.sum(commitment.power_mw < -0.1)
        sell_intervals = np.sum(commitment.power_mw > 0.1)
        idle_intervals = 96 - buy_intervals - sell_intervals

        print(f"\nLoaded commitment for {commitment.target_date}")
        print(f"  Buy intervals: {buy_intervals}")
        print(f"  Sell intervals: {sell_intervals}")
        print(f"  Idle intervals: {idle_intervals}")
        print(f"  Total buy: {commitment.total_buy_mwh:.1f} MWh")
        print(f"  Total sell: {commitment.total_sell_mwh:.1f} MWh")

    def get_current_commitment(self, timestamp: datetime) -> float:
        """Get committed power for current timestamp.

        Args:
            timestamp: Current time

        Returns:
            Committed power in MW (positive=sell, negative=buy)
            Returns 0 if no commitment loaded or outside schedule.
        """
        if self._commitment is None:
            return 0.0

        # Check if timestamp is for the committed day
        if timestamp.date() != self._commitment.target_date:
            return 0.0

        # Find the interval index
        idx = self._get_interval_index(timestamp)
        if idx is None or idx >= 96:
            return 0.0

        return float(self._commitment.power_mw[idx])

    def get_remaining_capacity(self, timestamp: datetime) -> float:
        """Get capacity remaining after DAM commitment.

        I calculate how much power is available for balancing markets
        after accounting for the DAM commitment.

        Returns:
            Available power in MW (always positive)
        """
        committed = abs(self.get_current_commitment(timestamp))
        remaining = self.max_power_mw - committed
        return max(0.0, remaining)

    def get_commitment_direction(self, timestamp: datetime) -> str:
        """Get the direction of current commitment.

        Returns:
            'buy', 'sell', or 'idle'
        """
        committed = self.get_current_commitment(timestamp)
        if committed > 0.1:
            return 'sell'
        elif committed < -0.1:
            return 'buy'
        else:
            return 'idle'

    def record_execution(
        self,
        timestamp: datetime,
        actual_power_mw: float,
        mfrr_action_mw: float = 0.0,
    ) -> ExecutionStatus:
        """Record actual execution for an interval.

        I compare what we planned (DAM + mFRR) vs what we actually did.
        The mFRR action is an INTENTIONAL deviation from DAM, so execution
        accuracy measures adherence to the total plan, not DAM alone.

        Args:
            timestamp: Interval timestamp
            actual_power_mw: Actual power dispatched
            mfrr_action_mw: Planned mFRR action (0 if no mFRR trading)

        Returns:
            ExecutionStatus with deviation info
        """
        dam_committed = self.get_current_commitment(timestamp)
        total_planned = dam_committed + mfrr_action_mw
        deviation = actual_power_mw - total_planned

        status = ExecutionStatus(
            timestamp=timestamp,
            dam_committed_mw=dam_committed,
            mfrr_planned_mw=mfrr_action_mw,
            total_planned_mw=total_planned,
            executed_mw=actual_power_mw,
            deviation_mw=deviation,
            is_executed=True,
        )

        # Log it
        target_date = timestamp.date()
        if target_date in self._execution_log:
            self._execution_log[target_date].intervals.append(status)

        # Warn if significant deviation from TOTAL PLAN (not DAM alone)
        if abs(deviation) > self.deviation_tolerance:
            print(f"  WARNING: Execution deviation at {timestamp.strftime('%H:%M')}: "
                  f"planned={total_planned:.1f} (DAM={dam_committed:.1f}, mFRR={mfrr_action_mw:.1f}), "
                  f"actual={actual_power_mw:.1f}, deviation={deviation:.1f} MW")

        return status

    def get_lookahead(
        self,
        timestamp: datetime,
        hours: int = 4,
    ) -> np.ndarray:
        """Get commitment lookahead for the next N hours.

        I return the committed power for the next intervals.
        This is useful for planning balancing market participation.

        Args:
            timestamp: Current time
            hours: Hours to look ahead

        Returns:
            Array of committed power values (length = hours * 4)
        """
        if self._commitment is None:
            return np.zeros(hours * 4)

        idx = self._get_interval_index(timestamp)
        if idx is None:
            return np.zeros(hours * 4)

        # Get next N intervals
        n_intervals = hours * 4
        end_idx = min(idx + n_intervals, 96)

        lookahead = self._commitment.power_mw[idx:end_idx]

        # Pad with zeros if needed
        if len(lookahead) < n_intervals:
            lookahead = np.pad(lookahead, (0, n_intervals - len(lookahead)))

        return lookahead

    def get_remaining_commitment_today(self, timestamp: datetime) -> Dict:
        """Get summary of remaining commitment for today.

        Returns:
            Dict with remaining buy/sell MWh and intervals
        """
        if self._commitment is None:
            return {'buy_mwh': 0, 'sell_mwh': 0, 'intervals': 0}

        idx = self._get_interval_index(timestamp)
        if idx is None:
            idx = 0

        remaining = self._commitment.power_mw[idx:]

        return {
            'buy_mwh': float(np.sum(np.abs(remaining[remaining < 0])) * 0.25),
            'sell_mwh': float(np.sum(remaining[remaining > 0]) * 0.25),
            'intervals': 96 - idx,
        }

    def get_execution_summary(self, target_date: date) -> Optional[Dict]:
        """Get execution summary for a day.

        Returns:
            Dict with execution metrics or None if no data
        """
        if target_date not in self._execution_log:
            return None

        log = self._execution_log[target_date]

        if not log.intervals:
            return None

        deviations = [s.deviation_mw for s in log.intervals]

        return {
            'date': target_date,
            'intervals_executed': len(log.intervals),
            # Execution accuracy (vs total plan = DAM + mFRR)
            'execution_accuracy': log.execution_accuracy,
            'total_deviation_mwh': log.total_deviation_mwh,
            'max_deviation_mw': max(abs(d) for d in deviations),
            'mean_deviation_mw': np.mean(np.abs(deviations)),
            # mFRR activity
            'mfrr_intervals': log.active_mfrr_intervals,
            'mfrr_total_mwh': log.mfrr_total_mwh,
            # DAM settlement info
            'dam_deviation_mwh': log.dam_deviation_mwh,
        }

    def _get_interval_index(self, timestamp: datetime) -> Optional[int]:
        """Convert timestamp to interval index (0-95)."""
        if self._commitment is None:
            return None

        # Calculate minutes since midnight
        minutes = timestamp.hour * 60 + timestamp.minute

        # Each interval is 15 minutes
        idx = minutes // 15

        return min(idx, 95)

    def can_participate_in_balancing(
        self,
        timestamp: datetime,
        battery_state: BatteryState,
    ) -> Dict:
        """Check if battery can participate in balancing markets.

        I determine available capacity considering:
        1. DAM commitment (must be honored)
        2. Current SoC
        3. Power limits

        Args:
            timestamp: Current time
            battery_state: Current battery state

        Returns:
            Dict with available up/down capacity
        """
        commitment = self.get_current_commitment(timestamp)
        direction = self.get_commitment_direction(timestamp)

        # Calculate available power after commitment
        remaining_power = self.max_power_mw - abs(commitment)

        # Calculate energy constraints
        soc = battery_state.soc
        min_soc = 0.05
        max_soc = 0.95

        # Available for discharge (sell/up-regulation)
        discharge_headroom = (soc - min_soc) * self.capacity_mwh / 0.25  # MW for 15min

        # Available for charge (buy/down-regulation)
        charge_headroom = (max_soc - soc) * self.capacity_mwh / 0.25  # MW for 15min

        # Combine power and energy constraints
        if direction == 'sell':
            # Already discharging, can add more discharge or offset with charge
            up_capacity = min(remaining_power, discharge_headroom)
            down_capacity = min(self.max_power_mw, charge_headroom)  # Can reverse
        elif direction == 'buy':
            # Already charging, can add more charge or offset with discharge
            up_capacity = min(self.max_power_mw, discharge_headroom)  # Can reverse
            down_capacity = min(remaining_power, charge_headroom)
        else:
            # Idle - full flexibility
            up_capacity = min(self.max_power_mw, discharge_headroom)
            down_capacity = min(self.max_power_mw, charge_headroom)

        return {
            'up_capacity_mw': max(0, up_capacity),    # Can discharge
            'down_capacity_mw': max(0, down_capacity), # Can charge
            'commitment_direction': direction,
            'commitment_power_mw': commitment,
            'remaining_power_mw': remaining_power,
        }


# Test
if __name__ == "__main__":
    from .dam_forecaster import DAMForecaster
    from .dam_bidder import DAMBidder
    from datetime import timedelta

    # Create components
    forecaster = DAMForecaster()
    bidder = DAMBidder()
    executor = CommitmentExecutor()

    # Generate forecast and bids for tomorrow
    tomorrow = date.today() + timedelta(days=1)
    forecast = forecaster.forecast(tomorrow)

    # Current battery state
    battery = BatteryState(
        timestamp=datetime.now(),
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    # Generate commitment
    commitment = bidder.generate_bids(forecast, battery)

    # Load into executor
    executor.load_commitment(commitment)

    # Simulate execution at various times
    print(f"\n=== Execution Simulation ===")
    for hour in [6, 12, 18, 22]:
        ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=hour)

        committed = executor.get_current_commitment(ts)
        remaining = executor.get_remaining_capacity(ts)
        direction = executor.get_commitment_direction(ts)

        print(f"\n{ts.strftime('%H:%M')}:")
        print(f"  Commitment: {committed:+.1f} MW ({direction})")
        print(f"  Remaining capacity: {remaining:.1f} MW")

        # Simulate actual execution (with small deviation)
        actual = committed + np.random.uniform(-0.3, 0.3)
        status = executor.record_execution(ts, actual)
        print(f"  Executed: {actual:+.1f} MW (deviation: {status.deviation_mw:+.2f} MW)")

        # Check balancing capability
        balance_cap = executor.can_participate_in_balancing(ts, battery)
        print(f"  Balancing: UP={balance_cap['up_capacity_mw']:.1f} MW, "
              f"DOWN={balance_cap['down_capacity_mw']:.1f} MW")

    # Get lookahead
    ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=10)
    lookahead = executor.get_lookahead(ts, hours=4)
    print(f"\n4-hour lookahead from 10:00:")
    print(f"  Values: {lookahead[:8]}...")  # First 8 intervals (2 hours)
