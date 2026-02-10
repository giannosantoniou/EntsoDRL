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
    status: str = "pending"     # pending, filled, partial, cancelled


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

        print(f"IntraDay Trader initialized")
        print(f"  Min spread: {min_spread_eur} EUR")
        print(f"  Max position change: {max_position_change:.0%}")

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
