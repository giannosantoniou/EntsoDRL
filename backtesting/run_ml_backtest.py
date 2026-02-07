"""
Run ML-Enhanced Backtests

I train ML models on historical data and compare their performance
against rule-based strategies.

IMPORTANT: To avoid look-ahead bias, I use walk-forward validation:
- Train on first 70% of data
- Test on remaining 30%
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
from .ml_strategy import (
    MLTrainer,
    PricePredictorStrategy,
    ActionClassifierStrategy,
    QLearningStrategy,
    create_ml_strategies
)


def run_ml_backtest(data_path: str = None, output_dir: str = None):
    """
    Run backtest with ML strategies using proper train/test split.
    """
    # Paths
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ML-ENHANCED mFRR BIDDING STRATEGY BACKTEST")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)
    df = df.sort_index()
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Split data: 70% train, 30% test
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\n2. Train/Test Split:")
    print(f"   Training: {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"   Testing: {len(test_df)} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")

    # Battery configuration
    battery_config = BatteryConfig(
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        degradation_cost=15.0,
    )

    print(f"\n3. Battery Configuration:")
    print(f"   Capacity: {battery_config.capacity_mwh} MWh")
    print(f"   Max Power: {battery_config.max_power_mw} MW")

    # Train ML models on training data
    print("\n" + "=" * 70)
    print("4. TRAINING ML MODELS")
    print("=" * 70)

    ml_strategies = create_ml_strategies(train_df, max_power_mw=battery_config.max_power_mw)

    # Define all strategies to compare
    all_strategies = [
        BaselineStrategy(),
        SimpleThresholdStrategy(threshold_high=150, threshold_low=30),
        TimeBasedStrategy(),
        SpreadStrategy(threshold=30),
        DataDrivenStrategy(),
    ] + ml_strategies

    # Run backtest on TEST data only (to avoid look-ahead bias)
    print("\n" + "=" * 70)
    print("5. RUNNING BACKTEST ON TEST DATA")
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
                'Utilization %': result.capacity_utilization * 100,
            })
            print(f"      Net Profit: {result.total_net_profit:,.0f} EUR")
        except Exception as e:
            print(f"      Error: {e}")

    # Create comparison DataFrame
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values('Net Profit (EUR)', ascending=False)

    # Print results
    print("\n" + "=" * 70)
    print("6. RESULTS COMPARISON (TEST PERIOD)")
    print("=" * 70)

    print(f"\nTest Period: {test_df.index.min().date()} to {test_df.index.max().date()}")
    print(f"Total Hours: {len(test_df)}")
    print("\n" + comparison.to_string(index=False))

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"ml_comparison_{timestamp}.csv"
    comparison.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Detailed analysis of top ML strategy vs best rule-based
    print("\n" + "=" * 70)
    print("7. ML vs RULE-BASED ANALYSIS")
    print("=" * 70)

    # Find best ML and best rule-based
    ml_names = [s.name for s in ml_strategies]
    ml_results = comparison[comparison['Strategy'].isin(ml_names)]
    rule_results = comparison[~comparison['Strategy'].isin(ml_names)]

    if len(ml_results) > 0 and len(rule_results) > 0:
        best_ml = ml_results.iloc[0]
        best_rule = rule_results.iloc[0]

        print(f"\nBest ML Strategy: {best_ml['Strategy']}")
        print(f"  Net Profit: {best_ml['Net Profit (EUR)']:,.0f} EUR")
        print(f"  Profit/Cycle: {best_ml['Profit/Cycle']:,.0f} EUR")
        print(f"  Utilization: {best_ml['Utilization %']:.1f}%")

        print(f"\nBest Rule-Based: {best_rule['Strategy']}")
        print(f"  Net Profit: {best_rule['Net Profit (EUR)']:,.0f} EUR")
        print(f"  Profit/Cycle: {best_rule['Profit/Cycle']:,.0f} EUR")
        print(f"  Utilization: {best_rule['Utilization %']:.1f}%")

        improvement = best_ml['Net Profit (EUR)'] - best_rule['Net Profit (EUR)']
        pct_improvement = improvement / max(abs(best_rule['Net Profit (EUR)']), 1) * 100

        print(f"\nML Improvement: {improvement:+,.0f} EUR ({pct_improvement:+.1f}%)")

    # Annualized projections
    print("\n" + "=" * 70)
    print("8. ANNUALIZED PROJECTIONS")
    print("=" * 70)

    hours_in_year = 8760
    test_hours = len(test_df)

    print(f"\nBased on {test_hours} hours of test data:")
    print("\n{:<40} {:>15} {:>15}".format("Strategy", "Test Profit", "Annual Est."))
    print("-" * 70)

    for _, row in comparison.head(5).iterrows():
        annual = row['Net Profit (EUR)'] * hours_in_year / test_hours
        print("{:<40} {:>15,.0f} {:>15,.0f}".format(
            row['Strategy'][:40], row['Net Profit (EUR)'], annual
        ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best = comparison.iloc[0]
    annual_profit = best['Net Profit (EUR)'] * hours_in_year / test_hours

    print(f"""
Winner: {best['Strategy']}

Test Period Performance:
  - Net Profit: {best['Net Profit (EUR)']:,.0f} EUR
  - Cycles: {best['Cycles']:.1f}
  - Profit/Cycle: {best['Profit/Cycle']:,.0f} EUR
  - Avg Discharge Price: {best['Avg Discharge Price']:.1f} EUR/MWh

Annualized Estimate:
  - Annual Profit: {annual_profit:,.0f} EUR
  - Monthly Average: {annual_profit/12:,.0f} EUR
""")

    return comparison


if __name__ == "__main__":
    run_ml_backtest()
