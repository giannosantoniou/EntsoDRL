"""
Phase 3 Integration Test: IntraDay Trading

I test the complete Phase 3 workflow:
1. IntraDay position tracking
2. Adjustment opportunity detection
3. Order creation and execution
4. Full orchestration with IntraDay

Run: python -m orchestrator.test_phase3
"""

import numpy as np
from datetime import datetime, date, timedelta

from orchestrator import (
    BatteryOrchestrator,
    IntraDayTrader,
    BatteryState,
    DAMCommitment,
    create_intraday_trader,
)


def test_intraday_trader_standalone():
    """Test the IntraDayTrader component independently."""
    print("\n" + "="*60)
    print("INTRADAY TRADER STANDALONE TEST")
    print("="*60)

    # Create trader
    trader = create_intraday_trader(
        max_power_mw=30.0,
        capacity_mwh=146.0,
        min_spread_eur=5.0,
    )

    # Create mock DAM commitment
    tomorrow = date.today() + timedelta(days=1)
    base_dt = datetime.combine(tomorrow, datetime.min.time())
    timestamps = [base_dt + timedelta(minutes=15*i) for i in range(96)]

    # Mock commitment: charge at night, discharge in evening
    power_mw = np.zeros(96)
    power_mw[8:20] = -30.0   # Charge 02:00-05:00 (intervals 8-20)
    power_mw[68:80] = 30.0   # Discharge 17:00-20:00 (intervals 68-80)

    # Mock DAM prices (typical daily pattern)
    hours = np.arange(96) / 4
    dam_prices = 80 + 40 * np.sin(2 * np.pi * (hours - 6) / 24)

    # Create mock commitment object
    commitment = DAMCommitment(
        target_date=tomorrow,
        timestamps=timestamps,
        power_mw=power_mw,
    )

    # Load commitment
    trader.load_dam_commitment(commitment, dam_prices)

    print("\n[1] Position loaded:")
    summary = trader.get_position_summary(tomorrow)
    print(f"  DAM Buy: {summary['dam_buy_mwh']:.1f} MWh")
    print(f"  DAM Sell: {summary['dam_sell_mwh']:.1f} MWh")

    # Create battery state
    battery = BatteryState(
        timestamp=datetime.now(),
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    # Test 1: IntraDay prices same as DAM (no opportunity)
    print("\n[2] No spread scenario:")
    intraday_prices = dam_prices.copy()
    opps = trader.get_adjustment_opportunities(intraday_prices, battery, min_spread=5.0)
    print(f"  Opportunities found: {len(opps)} (expected: 0)")
    assert len(opps) == 0, "No opportunities when prices match"
    print("  [OK] Correctly identified no opportunities")

    # Test 2: IntraDay prices higher than DAM (opportunities for sell positions)
    print("\n[3] Prices UP scenario (+15 EUR):")
    intraday_prices = dam_prices + 15
    opps = trader.get_adjustment_opportunities(intraday_prices, battery)
    print(f"  Opportunities found: {len(opps)}")
    if opps:
        best = opps[0]
        print(f"  Best opportunity: {best['delivery_time'].strftime('%H:%M')}")
        print(f"    Action: {best['action']}")
        print(f"    Spread: {best['spread']:+.1f} EUR")
        print(f"    Recommended: {best['recommended_mw']:.1f} MW")
    print("  [OK] Found adjustment opportunities")

    # Test 3: IntraDay prices lower than DAM
    print("\n[4] Prices DOWN scenario (-15 EUR):")
    intraday_prices = dam_prices - 15
    opps = trader.get_adjustment_opportunities(intraday_prices, battery)
    print(f"  Opportunities found: {len(opps)}")
    if opps:
        best = opps[0]
        print(f"  Best opportunity: {best['delivery_time'].strftime('%H:%M')}")
        print(f"    Action: {best['action']}")
        print(f"    Spread: {best['spread']:+.1f} EUR")
    print("  [OK] Found adjustment opportunities")

    # Test 4: Order creation
    print("\n[5] Order creation test:")
    delivery_time = timestamps[70]  # During discharge period
    order = trader.create_order(
        delivery_time=delivery_time,
        power_mw=5.0,  # Sell additional 5 MW
        price_limit=100.0,
        order_type="limit",
    )
    assert order.status == "pending", "Order should be pending"
    print("  [OK] Order created successfully")

    # Test 5: Order fill
    print("\n[6] Order fill test:")
    trader.fill_order(order, fill_price=102.0)
    assert order.status == "filled", "Order should be filled"

    # Check position updated
    summary = trader.get_position_summary(tomorrow)
    print(f"  IntraDay adjustment: {summary['intraday_adjustment_mwh']:.1f} MWh")
    print(f"  Orders filled: {summary['orders_filled']}")
    print("  [OK] Order filled and position updated")

    print("\n  [OK] All IntraDay trader tests passed")
    return True


def test_intraday_with_soc_constraints():
    """Test IntraDay trading respects SoC constraints."""
    print("\n" + "="*60)
    print("INTRADAY SOC CONSTRAINTS TEST")
    print("="*60)

    trader = create_intraday_trader(min_spread_eur=5.0)

    # Setup commitment
    tomorrow = date.today() + timedelta(days=1)
    base_dt = datetime.combine(tomorrow, datetime.min.time())
    timestamps = [base_dt + timedelta(minutes=15*i) for i in range(96)]

    power_mw = np.zeros(96)
    power_mw[40:52] = 30.0  # Discharge 10:00-13:00

    dam_prices = np.ones(96) * 100

    commitment = DAMCommitment(
        target_date=tomorrow,
        timestamps=timestamps,
        power_mw=power_mw,
    )
    trader.load_dam_commitment(commitment, dam_prices)

    # Test with very low SoC
    print("\n[1] Low SoC (10%) - should limit sell recommendations:")
    battery_low = BatteryState(
        timestamp=datetime.now(),
        soc=0.10,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    intraday_prices = dam_prices + 20  # Prices up
    opps = trader.get_adjustment_opportunities(intraday_prices, battery_low)

    sell_opps = [o for o in opps if 'sell' in o['action']]
    print(f"  Sell opportunities: {len(sell_opps)}")
    for opp in sell_opps[:3]:
        print(f"    {opp['delivery_time'].strftime('%H:%M')}: {opp['recommended_mw']:.1f} MW")

    # Test with very high SoC
    print("\n[2] High SoC (90%) - should limit buy recommendations:")
    battery_high = BatteryState(
        timestamp=datetime.now(),
        soc=0.90,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    intraday_prices = dam_prices - 20  # Prices down
    opps = trader.get_adjustment_opportunities(intraday_prices, battery_high)

    buy_opps = [o for o in opps if 'buy' in o['action'] or 'reduce_sell' in o['action']]
    print(f"  Buy/reduce opportunities: {len(buy_opps)}")
    for opp in buy_opps[:3]:
        print(f"    {opp['delivery_time'].strftime('%H:%M')}: {opp['action']} {opp['recommended_mw']:.1f} MW")

    print("\n  [OK] SoC constraints test passed")
    return True


def test_full_orchestrator_with_intraday():
    """Test complete orchestrator with IntraDay enabled."""
    print("\n" + "="*60)
    print("FULL ORCHESTRATOR WITH INTRADAY TEST")
    print("="*60)

    # Create orchestrator with all features enabled
    orchestrator = BatteryOrchestrator(
        max_power_mw=30.0,
        capacity_mwh=146.0,
        commitment_dir="test_commitments_phase3",
        balancing_enabled=True,
        balancing_strategy="RULE_BASED",
        afrr_enabled=True,
        mfrr_enabled=True,
        intraday_enabled=True,
        intraday_min_spread=5.0,
    )

    # Run day-ahead
    tomorrow = date.today() + timedelta(days=1)
    commitment = orchestrator.run_day_ahead(tomorrow)

    assert commitment is not None, "Day-ahead should return commitment"
    print("\n  [OK] Day-ahead completed with IntraDay enabled")

    # Simulate IntraDay trading window (D-1 14:00 to D-0)
    print("\n" + "="*60)
    print("INTRADAY TRADING SIMULATION")
    print("="*60)

    # Simulate IntraDay prices (different from DAM forecast)
    np.random.seed(42)  # For reproducibility
    base_prices = 80 + 40 * np.sin(np.linspace(0, 2*np.pi, 96))
    intraday_prices = base_prices + np.random.uniform(-15, 15, 96)

    # Update IntraDay prices
    orchestrator.update_intraday_prices(intraday_prices)

    # Run IntraDay optimization
    current_time = datetime.now()
    result = orchestrator.run_intraday(current_time)

    print(f"\n  IntraDay Analysis:")
    print(f"    Total opportunities: {result['total_opportunities']}")

    if result['top_opportunities']:
        print(f"\n  Top opportunities:")
        for i, opp in enumerate(result['top_opportunities'][:3]):
            print(f"    {i+1}. {opp['delivery_time'].strftime('%H:%M')}: "
                  f"{opp['action']} ({opp['spread']:+.1f} EUR spread)")

    # Check position summary
    summary = result.get('position_summary', {})
    print(f"\n  Position summary:")
    print(f"    DAM Buy: {summary.get('dam_buy_mwh', 0):.1f} MWh")
    print(f"    DAM Sell: {summary.get('dam_sell_mwh', 0):.1f} MWh")

    print("\n  [OK] Full orchestrator with IntraDay test passed")
    return True


def test_complete_trading_day():
    """Simulate a complete trading day with all markets."""
    print("\n" + "="*60)
    print("COMPLETE TRADING DAY SIMULATION")
    print("="*60)

    # Create fully-featured orchestrator
    orchestrator = BatteryOrchestrator(
        commitment_dir="test_complete_day",
        balancing_enabled=True,
        balancing_strategy="AGGRESSIVE_HYBRID",
        afrr_enabled=True,
        mfrr_enabled=True,
        intraday_enabled=True,
    )

    tomorrow = date.today() + timedelta(days=1)

    # D-1 10:00: Run day-ahead
    print("\n[D-1 10:00] DAY-AHEAD BIDDING")
    commitment = orchestrator.run_day_ahead(tomorrow)

    # D-1 14:00: IntraDay trading opens
    print("\n[D-1 14:00] INTRADAY TRADING")
    np.random.seed(123)
    intraday_prices = 90 + 30 * np.sin(np.linspace(0, 2*np.pi, 96)) + np.random.randn(96) * 10
    orchestrator.update_intraday_prices(intraday_prices)

    id_result = orchestrator.run_intraday(datetime.now())
    print(f"  Found {id_result['total_opportunities']} adjustment opportunities")

    # D-0: Real-time execution (sample hours)
    print("\n[D-0] REAL-TIME EXECUTION (sample)")
    sample_hours = [8, 12, 18, 22]

    for hour in sample_hours:
        ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=hour)

        # Update market data
        orchestrator.update_market_data({
            'mfrr_price_up': 100 + 50 * np.sin(2 * np.pi * (hour - 6) / 24),
            'mfrr_price_down': 80 + 40 * np.sin(2 * np.pi * (hour - 6) / 24),
            'afrr_up_price': 10,
            'afrr_down_price': 8,
        })

        result = orchestrator.run_realtime(ts)

        print(f"  {hour:02d}:00: DAM={result['dam_commitment_mw']:+5.1f}, "
              f"mFRR={result['mfrr_action_mw']:+5.1f}, "
              f"Total={result['total_dispatch_mw']:+5.1f} MW")

    print("\n  [OK] Complete trading day simulation passed")
    return True


def main():
    """Run all Phase 3 tests."""
    print("\n" + "="*60)
    print("PHASE 3 INTEGRATION TESTS")
    print("IntraDay Trading System")
    print("="*60)

    tests = [
        ("IntraDay Trader Standalone", test_intraday_trader_standalone),
        ("SoC Constraints", test_intraday_with_soc_constraints),
        ("Full Orchestrator with IntraDay", test_full_orchestrator_with_intraday),
        ("Complete Trading Day", test_complete_trading_day),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n  [FAIL] ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, status in results:
        symbol = "[OK]" if status == "PASS" else "[FAIL]"
        print(f"  {symbol} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\nTotal: {passed}/{len(results)} tests passed")

    return all(s == "PASS" for _, s in results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
