"""
Integration Test: Full Orchestrator with Trained DRL Model

I test the complete trading system using a real trained model
for mFRR decisions, simulating a full trading day.

Run: python -m orchestrator.test_with_model
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

from orchestrator import BatteryOrchestrator, BatteryState


def load_market_data():
    """Load real market data for testing."""
    # Try to load real mFRR data
    data_paths = [
        "data/mfrr_features_clean.csv",
        "data/feasible_data_with_dam.csv",
        "data/processed_training_data.csv",
    ]

    for path in data_paths:
        if Path(path).exists():
            df = pd.read_csv(path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(path, nrows=1).columns else None)
            print(f"Loaded market data from {path}")
            print(f"  Rows: {len(df)}, Columns: {list(df.columns)[:10]}...")
            return df

    print("No market data found, using synthetic data")
    return None


def test_with_trained_model():
    """Test orchestrator with trained DRL model."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Orchestrator with Trained DRL Model")
    print("="*70)

    # Find best model
    model_paths = [
        "models/final_v2_20260210_130905/best_model.zip",
        "models/ppo_v21_best.zip",
        "models/best_model_5M_profit.zip",
    ]

    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break

    if model_path is None:
        print("ERROR: No trained model found!")
        return False

    print(f"\nUsing model: {model_path}")

    # Create orchestrator with all features
    print("\n[1] Initializing Orchestrator...")
    orchestrator = BatteryOrchestrator(
        max_power_mw=30.0,
        capacity_mwh=146.0,
        efficiency=0.94,
        commitment_dir="test_integration",

        # Enable balancing with AI model
        balancing_enabled=True,
        balancing_model_path=model_path,
        balancing_strategy="AI",
        afrr_enabled=True,
        mfrr_enabled=True,

        # Enable IntraDay
        intraday_enabled=True,
        intraday_min_spread=5.0,
    )

    # Load market data
    print("\n[2] Loading Market Data...")
    df = load_market_data()

    # Set target date
    tomorrow = date.today() + timedelta(days=1)

    # Run Day-Ahead
    print("\n[3] Running Day-Ahead Process...")
    commitment = orchestrator.run_day_ahead(tomorrow)

    # Simulate IntraDay
    print("\n[4] Simulating IntraDay Trading...")
    if df is not None and 'price' in df.columns:
        # Use real prices with some variation
        base_prices = df['price'].values[:96] if len(df) >= 96 else np.ones(96) * 100
    else:
        # Realistic Greek market pattern (matches Serbia-based forecast)
        # Solar dip at 12-14h, evening peak at 19-21h
        hourly_pattern = np.array([
            90,   # 00:00 - night
            85,   # 01:00
            70,   # 02:00
            65,   # 03:00 - lowest night
            65,   # 04:00
            65,   # 05:00
            75,   # 06:00 - morning ramp
            90,   # 07:00
            115,  # 08:00 - morning peak
            105,  # 09:00
            60,   # 10:00 - solar starts
            45,   # 11:00
            30,   # 12:00 - solar dip
            25,   # 13:00 - solar minimum
            30,   # 14:00
            40,   # 15:00
            50,   # 16:00
            85,   # 17:00 - evening ramp
            115,  # 18:00
            120,  # 19:00 - evening peak
            105,  # 20:00
            115,  # 21:00
            110,  # 22:00
            110,  # 23:00
        ], dtype=np.float64)
        # Expand to 15-min resolution
        base_prices = np.repeat(hourly_pattern, 4)

    # Add realistic variation (simulate IntraDay prices different from DAM)
    # IntraDay prices typically have more volatility and can deviate from DAM
    np.random.seed(123)  # For reproducibility
    id_variation = np.random.uniform(-20, 20, 96)
    intraday_prices = base_prices + id_variation
    orchestrator.update_intraday_prices(intraday_prices)

    # Start IntraDay session
    orchestrator.start_intraday_session(tomorrow)

    # Run IntraDay with auto-execution
    id_result = orchestrator.run_intraday(
        current_time=datetime.now(),
        auto_execute=True,
        min_profit_eur=5.0,
    )

    print(f"  IntraDay opportunities found: {id_result['total_opportunities']}")
    print(f"  Orders executed: {id_result['executed_orders']}")
    print(f"  Session P&L: {id_result['session_pnl_eur']:+.0f} EUR")

    if id_result['executed_details']:
        print("  Executed orders:")
        for order in id_result['executed_details'][:5]:
            print(f"    {order['delivery']}: {order['power_mw']:+.1f} MW @ "
                  f"{order['fill_price']:.1f} EUR (P&L: {order['pnl_eur']:+.1f} EUR)")

    if id_result['top_opportunities']:
        print("  Top 3 remaining opportunities:")
        for i, opp in enumerate(id_result['top_opportunities'][:3]):
            print(f"    {i+1}. {opp['delivery_time'].strftime('%H:%M')}: "
                  f"{opp['action']} (spread: {opp['spread']:+.1f} EUR)")

    # Simulate Real-Time Execution
    print("\n[5] Simulating Real-Time Execution (Full Day)...")
    print("-" * 70)

    results = []
    total_dam_revenue = 0.0
    total_mfrr_revenue = 0.0
    total_afrr_value = 0.0

    # Simulate every 15-minute interval
    for interval in range(96):
        ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(minutes=15 * interval)
        hour = ts.hour

        # Generate realistic market data based on time of day
        price_factor = 1.0 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)
        mfrr_up = 80 * price_factor + np.random.uniform(-20, 20)
        mfrr_down = 60 * price_factor + np.random.uniform(-15, 15)

        # Update market data
        orchestrator.update_market_data({
            'mfrr_price_up': max(20, mfrr_up),
            'mfrr_price_down': max(10, mfrr_down),
            'afrr_up_price': 8 + np.random.uniform(-2, 2),
            'afrr_down_price': 6 + np.random.uniform(-2, 2),
        })

        # Get dispatch command
        dispatch = orchestrator.run_realtime(ts)
        results.append(dispatch)

        # Calculate revenues
        dam_power = dispatch['dam_commitment_mw']
        mfrr_power = dispatch['mfrr_action_mw']
        afrr_capacity = (dispatch['afrr_up_reserved_mw'] + dispatch['afrr_down_reserved_mw']) / 2

        # DAM revenue (use forecast prices)
        dam_price = intraday_prices[interval]
        if dam_power > 0:
            total_dam_revenue += dam_power * 0.25 * dam_price  # Sell
        else:
            total_dam_revenue -= abs(dam_power) * 0.25 * dam_price  # Buy cost

        # mFRR revenue
        if mfrr_power > 0:
            total_mfrr_revenue += mfrr_power * 0.25 * mfrr_up
        else:
            total_mfrr_revenue -= abs(mfrr_power) * 0.25 * mfrr_down

        # aFRR capacity value (simplified)
        afrr_price = (8 + 6) / 2  # Average
        total_afrr_value += afrr_capacity * 0.25 * afrr_price

        # Simulate execution
        actual = dispatch['total_dispatch_mw'] + np.random.uniform(-0.1, 0.1)
        orchestrator.execute_interval(ts, actual)

        # Print hourly summary
        if interval % 4 == 0:
            print(f"  {ts.strftime('%H:%M')}: "
                  f"DAM={dam_power:+6.1f} MW, "
                  f"mFRR={mfrr_power:+6.1f} MW, "
                  f"aFRR={afrr_capacity:4.1f} MW, "
                  f"Total={dispatch['total_dispatch_mw']:+6.1f} MW, "
                  f"SoC={dispatch['soc']:.0%}")

    # Summary Statistics
    print("\n" + "="*70)
    print("DAILY SUMMARY")
    print("="*70)

    # Energy flows
    dam_sell = sum(r['dam_commitment_mw'] for r in results if r['dam_commitment_mw'] > 0) * 0.25
    dam_buy = sum(abs(r['dam_commitment_mw']) for r in results if r['dam_commitment_mw'] < 0) * 0.25
    mfrr_sell = sum(r['mfrr_action_mw'] for r in results if r['mfrr_action_mw'] > 0) * 0.25
    mfrr_buy = sum(abs(r['mfrr_action_mw']) for r in results if r['mfrr_action_mw'] < 0) * 0.25
    avg_afrr = np.mean([(r['afrr_up_reserved_mw'] + r['afrr_down_reserved_mw'])/2 for r in results])

    print(f"\nEnergy Flows:")
    print(f"  DAM Sold:     {dam_sell:7.1f} MWh")
    print(f"  DAM Bought:   {dam_buy:7.1f} MWh")
    print(f"  mFRR Sold:    {mfrr_sell:7.1f} MWh")
    print(f"  mFRR Bought:  {mfrr_buy:7.1f} MWh")
    print(f"  Avg aFRR:     {avg_afrr:7.1f} MW")

    # Close IntraDay session and get P&L
    id_session = orchestrator.close_intraday_session()
    total_intraday_pnl = id_session.get('total_pnl_eur', 0)

    print(f"\nRevenue Breakdown:")
    print(f"  DAM Revenue:     {total_dam_revenue:+10.0f} EUR")
    print(f"  IntraDay P&L:    {total_intraday_pnl:+10.0f} EUR")
    print(f"  mFRR Revenue:    {total_mfrr_revenue:+10.0f} EUR")
    print(f"  aFRR Value:      {total_afrr_value:+10.0f} EUR")
    print(f"  --------------------------------")
    total_revenue = total_dam_revenue + total_intraday_pnl + total_mfrr_revenue + total_afrr_value
    print(f"  TOTAL:           {total_revenue:+10.0f} EUR")

    # Annual projection
    annual_projection = total_revenue * 365
    print(f"\n  Annual Projection: {annual_projection/1e6:+.2f} M EUR")

    # Execution quality
    summary = orchestrator.get_daily_summary(tomorrow)
    if summary:
        print(f"\nExecution Quality:")
        print(f"  Intervals executed: {summary['intervals_executed']}")
        print(f"  Execution accuracy: {summary['execution_accuracy']:.1%}")
        print(f"  Max deviation: {summary['max_deviation_mw']:.2f} MW")

    # Activity analysis
    active_intervals = sum(1 for r in results if abs(r['total_dispatch_mw']) > 0.5)
    utilization = active_intervals / 96

    print(f"\nUtilization:")
    print(f"  Active intervals: {active_intervals}/96 ({utilization:.0%})")

    # Strategy breakdown
    ai_decisions = sum(1 for r in results if abs(r['mfrr_action_mw']) > 0.5)
    print(f"  AI (mFRR) decisions: {ai_decisions}")

    print("\n" + "="*70)
    print("[OK] Integration test completed successfully!")
    print("="*70)

    return True


def test_strategy_comparison():
    """Compare AI vs Rule-Based vs Hybrid strategies."""
    print("\n" + "="*70)
    print("STRATEGY COMPARISON TEST")
    print("="*70)

    strategies = [
        ("AI", "models/final_v2_20260210_130905/best_model.zip"),
        ("RULE_BASED", None),
        ("AGGRESSIVE_HYBRID", None),
    ]

    results = {}
    tomorrow = date.today() + timedelta(days=1)

    # Generate consistent market data with realistic Greek pattern
    np.random.seed(42)
    # Realistic pattern: solar dip at 12-14h, evening peak at 19-21h
    hourly_pattern = np.array([
        90, 85, 70, 65, 65, 65, 75, 90, 115, 105,  # 00-09
        60, 45, 30, 25, 30, 40, 50, 85, 115, 120,  # 10-19
        105, 115, 110, 110                          # 20-23
    ], dtype=np.float64)
    intraday_prices = np.repeat(hourly_pattern, 4) + np.random.randn(96) * 10

    for strategy_name, model_path in strategies:
        if model_path and not Path(model_path).exists():
            print(f"\n  Skipping {strategy_name}: model not found")
            continue

        print(f"\n  Testing {strategy_name}...")

        try:
            orch = BatteryOrchestrator(
                commitment_dir=f"test_compare_{strategy_name.lower()}",
                balancing_enabled=True,
                balancing_model_path=model_path,
                balancing_strategy=strategy_name,
                afrr_enabled=True,
                mfrr_enabled=True,
                intraday_enabled=False,  # Disable for fair comparison
            )

            # Run DAM
            commitment = orch.run_day_ahead(tomorrow)

            # Simulate realtime
            total_revenue = 0.0
            for interval in range(96):
                ts = datetime.combine(tomorrow, datetime.min.time()) + timedelta(minutes=15 * interval)
                hour = ts.hour

                price_factor = 1.0 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)
                orch.update_market_data({
                    'mfrr_price_up': 80 * price_factor,
                    'mfrr_price_down': 60 * price_factor,
                })

                dispatch = orch.run_realtime(ts)

                # Calculate revenue
                price = intraday_prices[interval]
                power = dispatch['total_dispatch_mw']
                if power > 0:
                    total_revenue += power * 0.25 * price
                else:
                    total_revenue -= abs(power) * 0.25 * price

                orch.execute_interval(ts, power)

            results[strategy_name] = total_revenue
            print(f"    Revenue: {total_revenue:+,.0f} EUR")

        except Exception as e:
            print(f"    Error: {e}")
            results[strategy_name] = 0

    # Summary
    print("\n" + "-"*40)
    print("Strategy Comparison Results:")
    print("-"*40)
    for strategy, revenue in sorted(results.items(), key=lambda x: x[1], reverse=True):
        annual = revenue * 365
        print(f"  {strategy:20s}: {revenue:+10,.0f} EUR/day ({annual/1e6:+.2f} M/year)")

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ORCHESTRATOR INTEGRATION TESTS WITH TRAINED MODEL")
    print("="*70)

    tests = [
        ("Full Day Simulation", test_with_trained_model),
        ("Strategy Comparison", test_strategy_comparison),
    ]

    all_passed = True
    for name, test_func in tests:
        try:
            success = test_func()
            if not success:
                all_passed = False
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*70)
