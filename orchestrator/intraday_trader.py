"""
IntraDay Trader - Position Adjustment System

I handle IntraDay market participation:
- Monitor DAM commitment vs actual needs
- Adjust positions when profitable
- React to forecast updates and market changes

IntraDay trading happens AFTER DAM results (D-1 14:00)
and BEFORE real-time (D-0). It's a continuous market
where we can buy/sell to optimize our position.

Use cases:
1. Forecast error correction: DAM bid was based on D-2 forecast,
   now we have better info
2. Arbitrage: IntraDay price different from DAM price
3. Balancing preparation: Adjust position before activation
"""

import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .interfaces import BatteryState, DAMCommitment


@dataclass
class IntraDayOrder:
    """An IntraDay market order."""
    timestamp: datetime          # Order creation time
    delivery_start: datetime     # Delivery period start
    delivery_end: datetime       # Delivery period end
    power_mw: float             # Power (positive=sell, negative=buy)
    price_limit: float          # Price limit (max buy, min sell)
    order_type: str = "limit"   # "limit" or "market"
    status: str = "pending"     # pending, filled, partial, cancelled, expired
    fill_price: float = 0.0     # Actual fill price
    fill_time: Optional[datetime] = None
    pnl_eur: float = 0.0        # Profit/loss from this order


@dataclass
class IntraDaySession:
    """IntraDay trading session for a delivery day."""
    delivery_date: date
    session_start: datetime      # When ID trading opens (D-1 14:00)
    gate_closure_hours: float = 1.0  # Hours before delivery
    total_orders: int = 0
    filled_orders: int = 0
    total_pnl_eur: float = 0.0
    is_active: bool = True


@dataclass
class IntraDayPosition:
    """Current IntraDay position for a delivery period."""
    delivery_start: datetime
    delivery_end: datetime
    dam_commitment_mw: float    # Original DAM position
    intraday_adjustment_mw: float = 0.0  # Net IntraDay trades
    orders: List[IntraDayOrder] = field(default_factory=list)

    @property
    def net_position_mw(self) -> float:
        """Net position after IntraDay adjustments."""
        return self.dam_commitment_mw + self.intraday_adjustment_mw


class IntraDayTrader:
    """IntraDay Market Trader.

    I manage IntraDay position adjustments to optimize
    the battery's trading position between DAM and real-time.

    Strategy:
    1. Compare DAM commitment with updated forecasts
    2. Identify profitable adjustment opportunities
    3. Place orders when spread is sufficient
    4. Track position changes

    Gate closures (typical):
    - IntraDay opens: D-1 14:00 (after DAM results)
    - IntraDay closes: D-0 H-1 (1 hour before delivery)
    """

    def __init__(
        self,
        max_power_mw: float = 30.0,
        capacity_mwh: float = 146.0,
        min_spread_eur: float = 5.0,      # Minimum spread to trade
        max_position_change: float = 0.5,  # Max fraction of DAM to adjust
        price_forecast_weight: float = 0.7,  # Weight for forecast vs current
    ):
        """Initialize IntraDay trader.

        Args:
            max_power_mw: Maximum battery power
            capacity_mwh: Battery capacity
            min_spread_eur: Minimum price spread to consider trading
            max_position_change: Maximum position change as fraction of DAM
            price_forecast_weight: Weight for price forecast in decisions
        """
        self.max_power_mw = max_power_mw
        self.capacity_mwh = capacity_mwh
        self.min_spread = min_spread_eur
        self.max_position_change = max_position_change
        self.price_forecast_weight = price_forecast_weight

        # Position tracking
        self._positions: Dict[datetime, IntraDayPosition] = {}
        self._orders: List[IntraDayOrder] = []

        # DAM prices (from commitment)
        self._dam_prices: Dict[datetime, float] = {}

        # Session tracking
        self._session: Optional[IntraDaySession] = None

        # Execution settings
        self.auto_execute = True
        self.gate_closure_hours = 1.0  # Close 1h before delivery
        self.max_orders_per_interval = 3  # Max orders per delivery period
        self.execution_probability = 0.85  # Simulated fill probability

        print(f"IntraDay Trader initialized")
        print(f"  Min spread: {min_spread_eur} EUR")
        print(f"  Max position change: {max_position_change:.0%}")
        print(f"  Gate closure: {self.gate_closure_hours}h before delivery")

    def load_dam_commitment(
        self,
        commitment: DAMCommitment,
        dam_prices: np.ndarray,
    ) -> None:
        """Load DAM commitment and prices for IntraDay trading.

        I need to know the original DAM position and prices
        to calculate adjustment opportunities.

        Args:
            commitment: DAM commitment schedule
            dam_prices: DAM prices (96 values)
        """
        self._positions.clear()
        self._dam_prices.clear()

        for i, (ts, power) in enumerate(zip(commitment.timestamps, commitment.power_mw)):
            self._positions[ts] = IntraDayPosition(
                delivery_start=ts,
                delivery_end=ts + timedelta(minutes=15),
                dam_commitment_mw=power,
            )
            self._dam_prices[ts] = float(dam_prices[i])

        print(f"Loaded DAM commitment for {commitment.target_date}")
        print(f"  Intervals: {len(self._positions)}")

    def evaluate_adjustment(
        self,
        delivery_time: datetime,
        current_intraday_price: float,
        price_forecast: Optional[float] = None,
        battery_state: Optional[BatteryState] = None,
    ) -> Dict:
        """Evaluate whether to adjust position for a delivery period.

        I compare the DAM price with IntraDay price and forecast
        to decide if adjustment is profitable.

        Args:
            delivery_time: Delivery period start
            current_intraday_price: Current IntraDay market price
            price_forecast: Optional updated price forecast
            battery_state: Current battery state (for constraints)

        Returns:
            Dict with recommendation and details
        """
        if delivery_time not in self._positions:
            return {'action': 'none', 'reason': 'no_position'}

        position = self._positions[delivery_time]
        dam_price = self._dam_prices.get(delivery_time, current_intraday_price)

        # Calculate weighted price expectation
        if price_forecast is not None:
            expected_price = (
                self.price_forecast_weight * price_forecast +
                (1 - self.price_forecast_weight) * current_intraday_price
            )
        else:
            expected_price = current_intraday_price

        # Calculate spread
        spread = current_intraday_price - dam_price

        result = {
            'delivery_time': delivery_time,
            'dam_commitment_mw': position.dam_commitment_mw,
            'current_position_mw': position.net_position_mw,
            'dam_price': dam_price,
            'intraday_price': current_intraday_price,
            'expected_price': expected_price,
            'spread': spread,
            'action': 'none',
            'recommended_mw': 0.0,
            'reason': '',
        }

        # Check if spread is sufficient
        if abs(spread) < self.min_spread:
            result['reason'] = f'spread_too_small ({spread:.1f} EUR)'
            return result

        # Determine direction
        if spread > self.min_spread:
            # IntraDay price higher than DAM - we should have bought more in DAM
            # If we're selling (discharging), we got a bad deal - could buy back
            # If we're buying (charging), we got a good deal - could sell
            if position.dam_commitment_mw > 0:
                # We're selling - price went up, we're doing well
                # Consider selling more if we have capacity
                result['action'] = 'sell_more'
                result['recommended_mw'] = min(
                    self.max_power_mw - position.dam_commitment_mw,
                    position.dam_commitment_mw * self.max_position_change
                )
                result['reason'] = f'price_up +{spread:.1f} EUR'
            else:
                # We're buying - price went up, bad deal
                # Consider reducing buy (selling some)
                result['action'] = 'reduce_buy'
                result['recommended_mw'] = min(
                    abs(position.dam_commitment_mw) * self.max_position_change,
                    self.max_power_mw
                )
                result['reason'] = f'price_up +{spread:.1f} EUR, reduce exposure'

        elif spread < -self.min_spread:
            # IntraDay price lower than DAM - we should have sold more in DAM
            if position.dam_commitment_mw > 0:
                # We're selling - price went down, bad deal
                # Consider buying back some
                result['action'] = 'reduce_sell'
                result['recommended_mw'] = min(
                    position.dam_commitment_mw * self.max_position_change,
                    self.max_power_mw
                )
                result['reason'] = f'price_down {spread:.1f} EUR, reduce exposure'
            else:
                # We're buying - price went down, even better
                # Consider buying more
                result['action'] = 'buy_more'
                result['recommended_mw'] = min(
                    self.max_power_mw - abs(position.dam_commitment_mw),
                    abs(position.dam_commitment_mw) * self.max_position_change
                )
                result['reason'] = f'price_down {spread:.1f} EUR'

        # Apply battery constraints if available
        if battery_state is not None:
            result = self._apply_constraints(result, battery_state)

        return result

    def _apply_constraints(
        self,
        recommendation: Dict,
        battery_state: BatteryState,
    ) -> Dict:
        """Apply battery constraints to recommendation."""
        soc = battery_state.soc
        min_soc = 0.05
        max_soc = 0.95

        if recommendation['action'] in ('sell_more', 'reduce_buy'):
            # Need energy to sell/discharge
            available_energy = (soc - min_soc) * self.capacity_mwh
            max_power = available_energy / 0.25  # 15 min interval
            recommendation['recommended_mw'] = min(
                recommendation['recommended_mw'],
                max_power
            )

        elif recommendation['action'] in ('buy_more', 'reduce_sell'):
            # Need headroom to buy/charge
            available_headroom = (max_soc - soc) * self.capacity_mwh
            max_power = available_headroom / 0.25
            recommendation['recommended_mw'] = min(
                recommendation['recommended_mw'],
                max_power
            )

        # If constrained to 0, change action to none
        if recommendation['recommended_mw'] < 0.5:
            recommendation['action'] = 'none'
            recommendation['reason'] += ' (battery_constrained)'

        return recommendation

    def create_order(
        self,
        delivery_time: datetime,
        power_mw: float,
        price_limit: float,
        order_type: str = "limit",
    ) -> IntraDayOrder:
        """Create an IntraDay order.

        Args:
            delivery_time: Delivery period start
            power_mw: Power to trade (positive=sell)
            price_limit: Price limit
            order_type: "limit" or "market"

        Returns:
            Created order
        """
        order = IntraDayOrder(
            timestamp=datetime.now(),
            delivery_start=delivery_time,
            delivery_end=delivery_time + timedelta(minutes=15),
            power_mw=power_mw,
            price_limit=price_limit,
            order_type=order_type,
        )

        self._orders.append(order)

        # Update position
        if delivery_time in self._positions:
            self._positions[delivery_time].orders.append(order)

        print(f"Created IntraDay order:")
        print(f"  Delivery: {delivery_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Power: {power_mw:+.1f} MW")
        print(f"  Price limit: {price_limit:.2f} EUR")

        return order

    def fill_order(
        self,
        order: IntraDayOrder,
        fill_price: float,
        fill_quantity: Optional[float] = None,
    ) -> None:
        """Record order fill.

        Args:
            order: Order to fill
            fill_price: Execution price
            fill_quantity: Quantity filled (None = full fill)
        """
        fill_mw = fill_quantity or order.power_mw

        if abs(fill_mw - order.power_mw) < 0.1:
            order.status = "filled"
        else:
            order.status = "partial"

        # Update position
        if order.delivery_start in self._positions:
            self._positions[order.delivery_start].intraday_adjustment_mw += fill_mw

        print(f"Order filled: {fill_mw:+.1f} MW @ {fill_price:.2f} EUR")

    def get_position_summary(self, target_date: date) -> Dict:
        """Get summary of IntraDay positions for a date.

        Returns:
            Dict with position summary
        """
        positions = [p for ts, p in self._positions.items()
                    if ts.date() == target_date]

        if not positions:
            return {'date': target_date, 'intervals': 0}

        dam_buy = sum(p.dam_commitment_mw for p in positions if p.dam_commitment_mw < 0)
        dam_sell = sum(p.dam_commitment_mw for p in positions if p.dam_commitment_mw > 0)
        id_adjustment = sum(p.intraday_adjustment_mw for p in positions)
        net_buy = sum(p.net_position_mw for p in positions if p.net_position_mw < 0)
        net_sell = sum(p.net_position_mw for p in positions if p.net_position_mw > 0)

        return {
            'date': target_date,
            'intervals': len(positions),
            'dam_buy_mwh': abs(dam_buy) * 0.25,
            'dam_sell_mwh': dam_sell * 0.25,
            'intraday_adjustment_mwh': id_adjustment * 0.25,
            'net_buy_mwh': abs(net_buy) * 0.25,
            'net_sell_mwh': net_sell * 0.25,
            'orders_total': sum(len(p.orders) for p in positions),
            'orders_filled': sum(1 for p in positions for o in p.orders if o.status == 'filled'),
        }

    def get_adjustment_opportunities(
        self,
        intraday_prices: np.ndarray,
        battery_state: BatteryState,
        min_spread: Optional[float] = None,
    ) -> List[Dict]:
        """Find all adjustment opportunities above threshold.

        Args:
            intraday_prices: Current IntraDay prices (96 values)
            battery_state: Current battery state
            min_spread: Minimum spread (default: self.min_spread)

        Returns:
            List of opportunities sorted by potential profit
        """
        min_spread = min_spread or self.min_spread
        opportunities = []

        for i, (ts, position) in enumerate(self._positions.items()):
            if i >= len(intraday_prices):
                break

            result = self.evaluate_adjustment(
                delivery_time=ts,
                current_intraday_price=intraday_prices[i],
                battery_state=battery_state,
            )

            if result['action'] != 'none':
                # Calculate potential profit
                spread = abs(result['spread'])
                energy_mwh = result['recommended_mw'] * 0.25
                potential_profit = spread * energy_mwh
                result['potential_profit_eur'] = potential_profit
                opportunities.append(result)

        # Sort by potential profit
        opportunities.sort(key=lambda x: x.get('potential_profit_eur', 0), reverse=True)

        return opportunities


    def start_session(self, delivery_date: date) -> IntraDaySession:
        """Start IntraDay trading session for a delivery date.

        IntraDay trading typically opens D-1 14:00 after DAM results.

        Args:
            delivery_date: The delivery date

        Returns:
            New IntraDaySession
        """
        # Session starts D-1 14:00 (after DAM results)
        session_start = datetime.combine(
            delivery_date - timedelta(days=1),
            datetime.min.time().replace(hour=14)
        )

        self._session = IntraDaySession(
            delivery_date=delivery_date,
            session_start=session_start,
            gate_closure_hours=self.gate_closure_hours,
        )

        print(f"IntraDay session started for {delivery_date}")
        print(f"  Session start: {session_start}")
        print(f"  Gate closure: {self.gate_closure_hours}h before each delivery")

        return self._session

    def is_gate_open(self, delivery_time: datetime, current_time: datetime) -> bool:
        """Check if IntraDay gate is still open for a delivery period.

        Args:
            delivery_time: Delivery period start
            current_time: Current time

        Returns:
            True if gate is open (can still trade)
        """
        gate_closure = delivery_time - timedelta(hours=self.gate_closure_hours)
        return current_time < gate_closure

    def execute_opportunities(
        self,
        intraday_prices: np.ndarray,
        battery_state: BatteryState,
        current_time: datetime,
        max_orders: int = 5,
        min_profit_eur: float = 10.0,
    ) -> List[IntraDayOrder]:
        """Automatically execute IntraDay trading opportunities.

        I find the best opportunities and execute them automatically,
        respecting gate closure times and position limits.

        Args:
            intraday_prices: Current IntraDay prices (96 values)
            battery_state: Current battery state
            current_time: Current time
            max_orders: Maximum orders to place
            min_profit_eur: Minimum potential profit to execute

        Returns:
            List of executed orders
        """
        if not self.auto_execute:
            return []

        # Find opportunities
        opportunities = self.get_adjustment_opportunities(
            intraday_prices=intraday_prices,
            battery_state=battery_state,
        )

        executed_orders = []

        for opp in opportunities[:max_orders]:
            # Check minimum profit
            if opp.get('potential_profit_eur', 0) < min_profit_eur:
                continue

            # Check gate closure
            if not self.is_gate_open(opp['delivery_time'], current_time):
                continue

            # Check if already have too many orders for this interval
            existing_orders = len(self._positions[opp['delivery_time']].orders)
            if existing_orders >= self.max_orders_per_interval:
                continue

            # Determine power and price
            if opp['action'] in ('sell_more', 'reduce_buy'):
                power_mw = opp['recommended_mw']
                # For selling, set price slightly below market to ensure fill
                price_limit = opp['intraday_price'] * 0.99
            else:  # buy_more, reduce_sell
                power_mw = -opp['recommended_mw']
                # For buying, set price slightly above market
                price_limit = opp['intraday_price'] * 1.01

            # Create and execute order
            order = self.create_order(
                delivery_time=opp['delivery_time'],
                power_mw=power_mw,
                price_limit=price_limit,
                order_type="limit",
            )

            # Simulate execution (in production, this would go to exchange)
            fill_success = self._simulate_execution(order, opp['intraday_price'])

            if fill_success:
                # Calculate P&L
                dam_price = self._dam_prices.get(opp['delivery_time'], opp['intraday_price'])
                pnl = self._calculate_order_pnl(order, dam_price)
                order.pnl_eur = pnl

                executed_orders.append(order)

                if self._session:
                    self._session.total_orders += 1
                    self._session.filled_orders += 1
                    self._session.total_pnl_eur += pnl

        return executed_orders

    def _simulate_execution(
        self,
        order: IntraDayOrder,
        market_price: float,
    ) -> bool:
        """Simulate order execution (for backtesting/paper trading).

        In production, this would connect to the actual exchange.

        Args:
            order: Order to execute
            market_price: Current market price

        Returns:
            True if order was filled
        """
        import random

        # Check if price is acceptable
        if order.power_mw > 0:  # Selling
            if market_price < order.price_limit:
                order.status = "cancelled"
                return False
        else:  # Buying
            if market_price > order.price_limit:
                order.status = "cancelled"
                return False

        # Simulate fill probability
        if random.random() < self.execution_probability:
            # Add some slippage
            slippage = random.uniform(-0.5, 0.5)
            fill_price = market_price + slippage

            self.fill_order(order, fill_price)
            return True
        else:
            order.status = "partial"
            return False

    def _calculate_order_pnl(
        self,
        order: IntraDayOrder,
        dam_price: float,
    ) -> float:
        """Calculate P&L for an IntraDay order.

        P&L = (ID_price - DAM_price) * energy * direction

        For selling in ID at higher price than DAM: profit
        For buying in ID at lower price than DAM: profit

        Args:
            order: Filled order
            dam_price: Original DAM price

        Returns:
            P&L in EUR
        """
        energy_mwh = abs(order.power_mw) * 0.25  # 15-min interval
        price_diff = order.fill_price - dam_price

        if order.power_mw > 0:  # Selling
            # Selling at higher ID price than DAM = profit
            pnl = price_diff * energy_mwh
        else:  # Buying
            # Buying at lower ID price than DAM = profit (we pay less)
            pnl = -price_diff * energy_mwh

        return pnl

    def get_session_summary(self) -> Dict:
        """Get summary of current IntraDay session.

        Returns:
            Dict with session statistics
        """
        if not self._session:
            return {'status': 'no_active_session'}

        return {
            'delivery_date': self._session.delivery_date,
            'session_start': self._session.session_start,
            'total_orders': self._session.total_orders,
            'filled_orders': self._session.filled_orders,
            'fill_rate': (
                self._session.filled_orders / self._session.total_orders
                if self._session.total_orders > 0 else 0
            ),
            'total_pnl_eur': self._session.total_pnl_eur,
            'is_active': self._session.is_active,
        }

    def close_session(self) -> Dict:
        """Close IntraDay session and return final summary.

        Returns:
            Final session summary
        """
        if not self._session:
            return {'status': 'no_session'}

        self._session.is_active = False

        summary = self.get_session_summary()
        summary['status'] = 'closed'

        print(f"\nIntraDay Session Closed")
        print(f"  Orders: {self._session.filled_orders}/{self._session.total_orders} filled")
        print(f"  P&L: {self._session.total_pnl_eur:+.0f} EUR")

        return summary

    def get_net_position(self, delivery_time: datetime) -> float:
        """Get net position for a delivery period after IntraDay adjustments.

        Args:
            delivery_time: Delivery period start

        Returns:
            Net position in MW (DAM + IntraDay adjustments)
        """
        if delivery_time not in self._positions:
            return 0.0

        return self._positions[delivery_time].net_position_mw


# Factory function
def create_intraday_trader(**kwargs) -> IntraDayTrader:
    """Create an IntraDay trader with specified settings."""
    return IntraDayTrader(**kwargs)


# Test
if __name__ == "__main__":
    from datetime import datetime, date, timedelta
    import numpy as np

    # Create trader
    trader = IntraDayTrader()

    # Create mock DAM commitment
    tomorrow = date.today() + timedelta(days=1)
    base_dt = datetime.combine(tomorrow, datetime.min.time())
    timestamps = [base_dt + timedelta(minutes=15*i) for i in range(96)]

    # Mock commitment: charge at night, discharge in evening
    power_mw = np.zeros(96)
    power_mw[8:20] = -30.0   # Charge 02:00-05:00
    power_mw[68:80] = 30.0   # Discharge 17:00-20:00

    # Mock DAM prices
    dam_prices = 80 + 40 * np.sin(np.linspace(0, 2*np.pi, 96))

    # Create mock commitment object
    class MockCommitment:
        def __init__(self):
            self.target_date = tomorrow
            self.timestamps = timestamps
            self.power_mw = power_mw

    trader.load_dam_commitment(MockCommitment(), dam_prices)

    # Simulate IntraDay prices (slightly different from DAM)
    intraday_prices = dam_prices + np.random.uniform(-10, 10, 96)

    # Create battery state
    battery = BatteryState(
        timestamp=datetime.now(),
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    # Find opportunities
    print("\n=== IntraDay Opportunities ===")
    opportunities = trader.get_adjustment_opportunities(intraday_prices, battery)

    for i, opp in enumerate(opportunities[:5]):
        print(f"\n{i+1}. {opp['delivery_time'].strftime('%H:%M')}")
        print(f"   Action: {opp['action']}")
        print(f"   Spread: {opp['spread']:+.1f} EUR")
        print(f"   Recommended: {opp['recommended_mw']:.1f} MW")
        print(f"   Potential profit: {opp.get('potential_profit_eur', 0):.0f} EUR")

    # Test order creation
    print("\n=== Order Creation ===")
    if opportunities:
        best = opportunities[0]
        power = best['recommended_mw'] if 'sell' in best['action'] else -best['recommended_mw']
        order = trader.create_order(
            delivery_time=best['delivery_time'],
            power_mw=power,
            price_limit=best['intraday_price'],
        )

        # Simulate fill
        trader.fill_order(order, fill_price=best['intraday_price'])

    # Position summary
    print("\n=== Position Summary ===")
    summary = trader.get_position_summary(tomorrow)
    for key, value in summary.items():
        print(f"  {key}: {value}")
