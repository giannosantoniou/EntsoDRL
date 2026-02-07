"""
Run REALISTIC Backtests - No Data Leakage

I compare strategies using only information available at bid time:
- NO mfrr_spread (contains target)
- NO current settlement price (not known before auction)
- ONLY lagged prices and external features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from .battery import BatteryConfig
from .engine import BacktestEngine
from .strategies import (
    BaselineStrategy,
    SimpleThresholdStrategy,
    TimeBasedStrategy,
    SpreadStrategy,
    DataDrivenStrategy,
)
from .ml_strategy_realistic import create_realistic_strategies


def run_realistic_backtest(data_path: str = None, output_dir: str = None):
    """
    Run backtest with REALISTIC constraints (no data leakage).
    """
    # Paths
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("REALISTIC mFRR BIDDING STRATEGY BACKTEST")
    print("(No Data Leakage - Only information available at bid time)")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)
    df = df.sort_index()
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Add lagged spread (this is SAFE - it's from the past)
    if 'spread_lag_1h' not in df.columns and 'mfrr_spread' in df.columns:
        df['spread_lag_1h'] = df['mfrr_spread'].shift(1)

    # Split data: 70% train, 30% test (TEMPORAL split)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\n2. Train/Test Split (Temporal):")
    print(f"   Training: {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"   Testing: {len(test_df)} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")

    # Verify no overlap
    assert train_df.index.max() < test_df.index.min(), "Train/test overlap detected!"
    print("   [OK] No temporal overlap between train and test")

    # Battery configuration
    battery_config = BatteryConfig(
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        degradation_cost=15.0,
    )

    # Train REALISTIC ML models (no leakage)
    print("\n3. Training REALISTIC ML Models...")
    ml_strategies = create_realistic_strategies(
        train_df, test_df,
        max_power_mw=battery_config.max_power_mw
    )

    # Rule-based strategies (for comparison)
    # NOTE: I modify these to NOT use current price (realistic)
    rule_strategies = [
        BaselineStrategy(),
        TimeBasedStrategy(),  # Pure time-based, no price info
    ]

    all_strategies = rule_strategies + ml_strategies

    # Run backtest
    print("\n" + "=" * 70)
    print("4. RUNNING REALISTIC BACKTEST")
    print("=" * 70)

    engine = BacktestEngine(test_df, battery_config)

    results = []
    for strategy in all_strategies:
        print(f"\n   Running {strategy.name}...")
        try:
            result = engine.run(strategy, verbose=False)
            results.append({
                'Strategy': strategy.name,
                'Net Profit (EUR)': result.total_net_profit,
                'Revenue (EUR)': result.total_revenue,
                'Degradation (EUR)': result.total_degradation_cost,
                'Cycles': result.total_cycles,
                'Profit/Cycle': result.profit_per_cycle,
                'Discharge Hours': result.discharge_hours,
                'Charge Hours': result.charge_hours,
                'Avg Discharge Price': result.avg_discharge_price,
                'Avg Charge Price': result.avg_charge_price,
                'Utilization %': result.capacity_utilization * 100,
            })
            print(f"      Net Profit: {result.total_net_profit:,.0f} EUR")
            print(f"      Cycles: {result.total_cycles:.1f}")
        except Exception as e:
            print(f"      Error: {e}")
            import traceback
            traceback.print_exc()

    # Create comparison
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values('Net Profit (EUR)', ascending=False)

    # Print results
    print("\n" + "=" * 70)
    print("5. REALISTIC RESULTS COMPARISON")
    print("=" * 70)

    print(f"\nTest Period: {test_df.index.min().date()} to {test_df.index.max().date()}")
    print(f"Total Hours: {len(test_df)}\n")

    # Format for display
    display_df = comparison[[
        'Strategy', 'Net Profit (EUR)', 'Cycles', 'Profit/Cycle',
        'Utilization %', 'Avg Discharge Price', 'Avg Charge Price'
    ]].copy()

    display_df['Net Profit (EUR)'] = display_df['Net Profit (EUR)'].apply(lambda x: f"{x:,.0f}")
    display_df['Profit/Cycle'] = display_df['Profit/Cycle'].apply(lambda x: f"{x:,.0f}")
    display_df['Cycles'] = display_df['Cycles'].apply(lambda x: f"{x:.1f}")
    display_df['Utilization %'] = display_df['Utilization %'].apply(lambda x: f"{x:.1f}%")
    display_df['Avg Discharge Price'] = display_df['Avg Discharge Price'].apply(lambda x: f"{x:.1f}")
    display_df['Avg Charge Price'] = display_df['Avg Charge Price'].apply(lambda x: f"{x:.1f}")

    print(display_df.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"realistic_comparison_{timestamp}.csv"
    comparison.to_csv(results_path, index=False)

    # Annualized projections
    print("\n" + "=" * 70)
    print("6. REALISTIC ANNUALIZED PROJECTIONS")
    print("=" * 70)

    hours_year = 8760
    test_hours = len(test_df)

    print(f"\nBased on {test_hours} hours of test data:\n")
    print(f"{'Strategy':<45} {'Test Profit':>15} {'Annual Est.':>15}")
    print("-" * 75)

    for _, row in comparison.iterrows():
        annual = row['Net Profit (EUR)'] * hours_year / test_hours
        print(f"{row['Strategy']:<45} {row['Net Profit (EUR)']:>15,.0f} {annual:>15,.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("REALISTIC EXECUTIVE SUMMARY")
    print("=" * 70)

    best = comparison.iloc[0]
    annual = best['Net Profit (EUR)'] * hours_year / test_hours

    print(f"""
WINNER: {best['Strategy']}

Test Period Performance ({test_hours} hours):
  - Net Profit:       {best['Net Profit (EUR)']:>12,.0f} EUR
  - Full Cycles:      {best['Cycles']:>12.1f}
  - Profit per Cycle: {best['Profit/Cycle']:>12,.0f} EUR
  - Utilization:      {best['Utilization %']:>12.1f}%

REALISTIC Annualized Projection:
  - Annual Profit:    {annual:>12,.0f} EUR
  - Monthly Average:  {annual/12:>12,.0f} EUR
  - Daily Average:    {annual/365:>12,.0f} EUR
""")

    # Compare to leaky version
    print("\n" + "=" * 70)
    print("COMPARISON: LEAKY vs REALISTIC")
    print("=" * 70)

    print("""
Previous (LEAKY) Results:
  - Aggressive Hybrid: ~8.9M EUR/year
  - ML had R2 = 0.98 (using mfrr_spread)

Current (REALISTIC) Results:
  - ML has R2 ~ 0.49 (without mfrr_spread)
  - See above for realistic projections

Key Insight:
  The high profits were INFLATED by data leakage.
  Realistic ML prediction is much harder!
""")

    return comparison


if __name__ == "__main__":
    run_realistic_backtest()
