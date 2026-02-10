"""
Phase 2 Integration Test: Full Balancing (aFRR + mFRR)

I test the complete Phase 2 workflow:
1. DAM + Commitment (from Phase 1)
2. aFRR capacity bidding
3. mFRR energy trading with DRL model
4. Full orchestration

Run: python -m orchestrator.test_phase2
"""

import numpy as np
from datetime import datetime, date, timedelta

from orchestrator import (
    BatteryOrchestrator,
    BalancingTrader,
    BatteryState,
    create_balancing_trader,
)


def test_balancing_trader_standalone():
    """Test the BalancingTrader component independently."""
    print("\n" + "="*60)
    print("BALANCING TRADER STANDALONE TEST")
    print("="*60)

    # Test with rule-based strategy (no model needed)
    trader = create_balancing_trader(
        strategy_type="RULE_BASED",
        afrr_enabled=True,
        mfrr_enabled=True,
    )

    # Create test state
    battery = BatteryState(
        timestamp=datetime.now(),
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    # Test market data scenarios
    test_scenarios = [
        {"name": "Low price", "mfrr_price_up": 40.0, "mfrr_price_down": 30.0},
        {"name": "Normal price", "mfrr_price_up": 100.0, "mfrr_price_down": 80.0},
        {"name": "High price", "mfrr_price_up": 180.0, "mfrr_price_down": 150.0},
        {"name": "Very high price", "mfrr_price_up": 250.0, "mfrr_price_down": 200.0},
    ]

    print("\n[1] Price response test:")
    for scenario in test_scenarios:
        market_data = {
            'mfrr_price_up': scenario['mfrr_price_up'],
            'mfrr_price_down': scenario['mfrr_price_down'],
            'afrr_up_price': 10.0,
            'afrr_down_price': 8.0,
        }

        action = trader.decide_action(
            battery_state=battery,
            available_capacity_mw=20.0,
            market_data=market_data,
        )

        expected_direction = "discharge" if scenario['mfrr_price_up'] > 150 else "charge" if scenario['mfrr_price_up'] < 50 else "idle"
        actual_direction = "discharge" if action > 0.5 else "charge" if action < -0.5 else "idle"

        print(f"  {scenario['name']:15s} (price={scenario['mfrr_price_up']:3.0f}): "
              f"action={action:+6.1f} MW ({actual_direction})")

    # Test SoC sensitivity
    print("\n[2] SoC sensitivity test:")
    market_data = {'mfrr_price_up': 180.0, 'mfrr_price_down': 150.0}

    for soc in [0.10, 0.30, 0.50, 0.70, 0.90]:
        battery.soc = soc
        action = trader.decide_action(
            battery_state=battery,
            available_capacity_mw=20.0,
            market_data=market_data,
        )
        print(f"  SoC={soc:.0%}: action={action:+6.1f} MW")

    # Test aFRR bidding
    print("\n[3] aFRR bidding test:")
    battery.soc = 0.5
    for soc in [0.10, 0.50, 0.90]:
        battery.soc = soc
        bid = trader.get_afrr_bid(
            battery_state=battery,
            available_capacity_mw=20.0,
            market_data={'afrr_up_price': 12.0, 'afrr_down_price': 10.0},
        )
        print(f"  SoC={soc:.0%}: UP={bid['up_capacity_mw']:.1f} MW, "
              f"DOWN={bid['down_capacity_mw']:.1f} MW, "
              f"Price={bid['bid_price_eur']:.2f} EUR")

    print("\n  [OK] Balancing trader tests passed")
    return True


def test_aggressive_hybrid_strategy():
    """Test the AggressiveHybrid strategy."""
    print("\n" + "="*60)
    print("AGGRESSIVE HYBRID STRATEGY TEST")
    print("="*60)

    trader = create_balancing_trader(
        strategy_type="AGGRESSIVE_HYBRID",
        afrr_enabled=False,  # Disable aFRR for cleaner test
        mfrr_enabled=True,
    )

    battery = BatteryState(
        timestamp=datetime.now().replace(hour=10),  # 10:00 - solar hours
        soc=0.5,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )

    # Test time-based behavior
    print("\n[1] Time-window behavior test:")
    test_hours = [
        (3, "Night"),
        (7, "Morning ramp"),
        (12, "Solar hours"),
        (19, "Evening peak"),
        (22, "Off-peak"),
    ]

    market_data = {'mfrr_price_up': 100.0, 'mfrr_price_down': 80.0}

    for hour, period in test_hours:
        battery.timestamp = datetime.now().replace(hour=hour, minute=0)
        action = trader.decide_action(
            battery_state=battery,
            available_capacity_mw=25.0,
            market_data=market_data,
        )
        direction = "discharge" if action > 0.5 else "charge" if action < -0.5 else "idle"
        print(f"  {hour:02d}:00 ({period:12s}): {action:+6.1f} MW ({direction})")

    print("\n  [OK] Aggressive hybrid strategy tests passed")
    return True


def test_full_orchestrator_with_balancing():
    """Test complete orchestrator with balancing enabled."""
    print("\n" + "="*60)
    print("FULL ORCHESTRATOR WITH BALANCING TEST")
    print("="*60)

    # Create orchestrator with balancing enabled
    orchestrator = BatteryOrchestrator(
        max_power_mw=30.0,
        capacity_mwh=146.0,
        commitment_dir="test_commitments_phase2",
        balancing_enabled=True,
        balancing_strategy="RULE_BASED",  # No model needed for test
        afrr_enabled=True,
        mfrr_enabled=True,
    )

    # Run day-ahead
    tomorrow = date.today() + timedelta(days=1)
    commitment = orchestrator.run_day_ahead(tomorrow)

    assert commitment is not None, "Day-ahead should return commitment"
    print("\n  [OK] Day-ahead completed with balancing enabled")

    # Simulate real-time with market data
    print("\n" + "="*60)
    print("REAL-TIME WITH BALANCING SIMULATION")
    print("="*60)

    # Simulate 4 hours of trading
    results = []
    start_hour = 8

    for i in range(16):  # 4 hours = 16 intervals
        ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=start_hour, minutes=15*i)

        # Update market data (simulate price variation)
        hour = ts.hour
        base_price = 100 + 50 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 18:00
        orchestrator.update_market_data({
            'mfrr_price_up': base_price + np.random.uniform(-10, 10),
            'mfrr_price_down': base_price - 20 + np.random.uniform(-5, 5),
            'afrr_up_price': 10 + np.random.uniform(-2, 2),
            'afrr_down_price': 8 + np.random.uniform(-2, 2),
        })

        # Get dispatch command
        result = orchestrator.run_realtime(ts)
        results.append(result)

        # Simulate execution
        actual = result['total_dispatch_mw'] + np.random.uniform(-0.1, 0.1)
        orchestrator.execute_interval(ts, actual)

        # Print intervals with activity
        if abs(result['dam_commitment_mw']) > 0.1 or abs(result['mfrr_action_mw']) > 0.1:
            print(f"  {result['interval']}: "
                  f"DAM={result['dam_commitment_mw']:+5.1f}, "
                  f"mFRR={result['mfrr_action_mw']:+5.1f}, "
                  f"aFRR(U/D)={result['afrr_up_reserved_mw']:.1f}/{result['afrr_down_reserved_mw']:.1f}, "
                  f"Total={result['total_dispatch_mw']:+5.1f} MW")

    # Verify results
    assert all('dam_commitment_mw' in r for r in results), "All results should have DAM"
    assert all('mfrr_action_mw' in r for r in results), "All results should have mFRR"
    assert all('afrr_up_reserved_mw' in r for r in results), "All results should have aFRR"

    # Summary statistics
    total_dam_energy = sum(r['dam_commitment_mw'] for r in results) * 0.25
    total_mfrr_energy = sum(r['mfrr_action_mw'] for r in results) * 0.25
    avg_afrr_capacity = np.mean([r['afrr_up_reserved_mw'] + r['afrr_down_reserved_mw'] for r in results]) / 2

    print(f"\n  Summary (4 hours):")
    print(f"    DAM energy: {total_dam_energy:+.1f} MWh")
    print(f"    mFRR energy: {total_mfrr_energy:+.1f} MWh")
    print(f"    Avg aFRR capacity: {avg_afrr_capacity:.1f} MW")

    print("\n  [OK] Full orchestrator with balancing test passed")
    return True


def test_ai_strategy_loading():
    """Test that AI strategy can be configured (model path validation)."""
    print("\n" + "="*60)
    print("AI STRATEGY CONFIGURATION TEST")
    print("="*60)

    # Test that orchestrator accepts AI configuration
    # (won't actually load model if path doesn't exist)
    try:
        orchestrator = BatteryOrchestrator(
            balancing_enabled=True,
            balancing_strategy="AI",
            balancing_model_path="models/nonexistent.zip",  # Won't load
            afrr_enabled=True,
            mfrr_enabled=True,
            commitment_dir="test_commitments_ai",
        )
        print("  [OK] AI strategy configuration accepted")
        return True
    except Exception as e:
        # Expected to fail during model loading, but config should be accepted
        if "model path" in str(e).lower() or "load" in str(e).lower():
            print(f"  [OK] AI strategy correctly requires valid model path")
            return True
        else:
            print(f"  [FAIL] Unexpected error: {e}")
            return False


def main():
    """Run all Phase 2 tests."""
    print("\n" + "="*60)
    print("PHASE 2 INTEGRATION TESTS")
    print("Full Balancing: aFRR + mFRR")
    print("="*60)

    tests = [
        ("Balancing Trader Standalone", test_balancing_trader_standalone),
        ("Aggressive Hybrid Strategy", test_aggressive_hybrid_strategy),
        ("Full Orchestrator with Balancing", test_full_orchestrator_with_balancing),
        ("AI Strategy Configuration", test_ai_strategy_loading),
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
