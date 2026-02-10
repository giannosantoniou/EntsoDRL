"""
Phase 1 Integration Test: DAM + Commitment System

I test the complete Phase 1 workflow:
1. DAM price forecasting
2. Bid generation
3. Commitment storage
4. Real-time execution
5. Deviation tracking

Run: python -m orchestrator.test_phase1
"""

import numpy as np
from datetime import datetime, date, timedelta

from orchestrator import (
    BatteryOrchestrator,
    DAMForecaster,
    DAMBidder,
    CommitmentExecutor,
    BatteryState,
)


def test_individual_components():
    """Test each component separately."""
    print("\n" + "="*60)
    print("COMPONENT TESTS")
    print("="*60)

    tomorrow = date.today() + timedelta(days=1)

    # Test 1: DAM Forecaster
    print("\n[1] Testing DAMForecaster...")
    forecaster = DAMForecaster()
    forecast = forecaster.forecast(tomorrow)

    assert forecast.n_intervals == 96, f"Expected 96 intervals, got {forecast.n_intervals}"
    assert len(forecast.buy_prices) == 96, "Buy prices should have 96 values"
    assert len(forecast.sell_prices) == 96, "Sell prices should have 96 values"
    assert np.all(forecast.buy_prices >= forecast.sell_prices), "Buy price should be >= sell price"

    print(f"  [OK] Forecast generated: {forecast.buy_prices.min():.1f}-{forecast.buy_prices.max():.1f} EUR")

    # Test 2: DAM Bidder
    print("\n[2] Testing DAMBidder...")
    bidder = DAMBidder()
    battery_state = BatteryState(
        timestamp=datetime.now(),
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    commitment = bidder.generate_bids(forecast, battery_state)

    assert commitment.target_date == tomorrow, "Commitment date mismatch"
    assert len(commitment.power_mw) == 96, "Commitment should have 96 values"
    assert np.all(np.abs(commitment.power_mw) <= 30.0), "Power should be within limits"

    print(f"  [OK] Bids generated: Buy={commitment.total_buy_mwh:.1f} MWh, Sell={commitment.total_sell_mwh:.1f} MWh")

    # Test 3: Commitment Executor
    print("\n[3] Testing CommitmentExecutor...")
    executor = CommitmentExecutor()
    executor.load_commitment(commitment)

    # Test at a few timestamps
    for hour in [0, 8, 14, 20]:
        ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=hour)
        power = executor.get_current_commitment(ts)
        remaining = executor.get_remaining_capacity(ts)
        direction = executor.get_commitment_direction(ts)

        assert remaining >= 0, "Remaining capacity should be non-negative"
        assert remaining <= 30.0, "Remaining capacity should be <= max power"
        print(f"  {ts.strftime('%H:%M')}: {power:+5.1f} MW ({direction}), remaining: {remaining:.1f} MW")

    print(f"  [OK] Commitment executor working correctly")

    return True


def test_full_pipeline():
    """Test the complete orchestrator pipeline."""
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)

    # Create orchestrator
    orchestrator = BatteryOrchestrator(
        max_power_mw=30.0,
        capacity_mwh=146.0,
        commitment_dir="test_commitments",
    )

    # Day-ahead process
    tomorrow = date.today() + timedelta(days=1)
    commitment = orchestrator.run_day_ahead(tomorrow)

    assert commitment is not None, "Day-ahead should return commitment"

    # Real-time simulation (every 15 minutes for 4 hours)
    print("\n" + "="*60)
    print("REAL-TIME SIMULATION (4 hours)")
    print("="*60)

    results = []
    start_hour = 8
    for i in range(16):  # 4 hours = 16 intervals
        ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=start_hour, minutes=15*i)

        # Get dispatch command
        result = orchestrator.run_realtime(ts)
        results.append(result)

        # Simulate actual execution
        actual = result['dam_commitment_mw'] + np.random.uniform(-0.1, 0.1)
        orchestrator.execute_interval(ts, actual)

        # Print non-idle intervals
        if abs(result['dam_commitment_mw']) > 0.1:
            print(f"  {result['interval']}: DAM={result['dam_commitment_mw']:+5.1f} MW, "
                  f"Balancing capacity: UP={result['up_capacity_mw']:.1f} DOWN={result['down_capacity_mw']:.1f}")

    # Verify results
    assert len(results) == 16, "Should have 16 results"
    assert all('dam_commitment_mw' in r for r in results), "All results should have DAM commitment"
    assert all('remaining_capacity_mw' in r for r in results), "All results should have remaining capacity"

    print(f"\n  [OK] Real-time simulation completed successfully")

    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("EDGE CASE TESTS")
    print("="*60)

    # Test 1: Low SoC scenario
    print("\n[1] Low SoC scenario...")
    forecaster = DAMForecaster()
    bidder = DAMBidder()

    tomorrow = date.today() + timedelta(days=1)
    forecast = forecaster.forecast(tomorrow)

    # Low SoC - should limit sell commitments
    battery_low = BatteryState(
        timestamp=datetime.now(),
        soc=0.10,  # Only 10%
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    commitment_low = bidder.generate_bids(forecast, battery_low)
    print(f"  Low SoC (10%): Buy={commitment_low.total_buy_mwh:.1f} MWh, Sell={commitment_low.total_sell_mwh:.1f} MWh")

    # High SoC - should limit buy commitments
    battery_high = BatteryState(
        timestamp=datetime.now(),
        soc=0.90,  # 90% full
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    commitment_high = bidder.generate_bids(forecast, battery_high)
    print(f"  High SoC (90%): Buy={commitment_high.total_buy_mwh:.1f} MWh, Sell={commitment_high.total_sell_mwh:.1f} MWh")

    # Test 2: Commitment recovery
    print("\n[2] Commitment recovery...")
    orchestrator = BatteryOrchestrator(commitment_dir="test_commitments")

    # Run day-ahead
    commitment = orchestrator.run_day_ahead(tomorrow)

    # Create new orchestrator (simulating restart)
    orchestrator2 = BatteryOrchestrator(commitment_dir="test_commitments")

    # Try to load commitment
    loaded = orchestrator2.load_commitment_from_file(tomorrow)
    assert loaded, "Should be able to load commitment from file"

    # Verify it works
    ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=12)
    result = orchestrator2.run_realtime(ts)
    assert 'dam_commitment_mw' in result, "Should have DAM commitment after recovery"

    print(f"  [OK] Commitment recovery working")

    # Test 3: No commitment loaded
    print("\n[3] No commitment loaded...")
    executor = CommitmentExecutor()

    # Without loading commitment, should return 0
    ts = datetime.now()
    power = executor.get_current_commitment(ts)
    assert power == 0.0, "Should return 0 when no commitment loaded"

    remaining = executor.get_remaining_capacity(ts)
    assert remaining == 30.0, "Should return max power when no commitment"

    print(f"  [OK] No commitment scenario handled correctly")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PHASE 1 INTEGRATION TESTS")
    print("DAM + Commitment System")
    print("="*60)

    tests = [
        ("Component Tests", test_individual_components),
        ("Full Pipeline", test_full_pipeline),
        ("Edge Cases", test_edge_cases),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n  [FAIL] ERROR: {e}")
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
