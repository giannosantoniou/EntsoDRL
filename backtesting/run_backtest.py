"""
Run Backtests and Generate Report

I compare different mFRR bidding strategies on historical data
and generate a comprehensive performance report.
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
    OptimalStrategy
)


def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare feature data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    return df


def run_full_backtest(data_path: str = None, output_dir: str = None):
    """
    Run comprehensive backtest with all strategies.

    Args:
        data_path: Path to feature CSV (default: data/mfrr_features.csv)
        output_dir: Directory for output files (default: backtesting/results)
    """
    # Paths
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("=" * 70)
    print("mFRR BIDDING STRATEGY BACKTEST")
    print("=" * 70)

    df = load_data(str(data_path))

    # Battery configuration (from CLAUDE.md)
    battery_config = BatteryConfig(
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        soc_min=0.05,
        soc_max=0.95,
        degradation_cost=15.0,  # EUR/MWh
        initial_soc=0.50
    )

    print(f"\nBattery Configuration:")
    print(f"  Capacity: {battery_config.capacity_mwh} MWh")
    print(f"  Max Power: {battery_config.max_power_mw} MW")
    print(f"  Efficiency: {battery_config.efficiency*100}%")
    print(f"  Degradation: {battery_config.degradation_cost} EUR/MWh")

    # Initialize engine
    engine = BacktestEngine(df, battery_config)

    # Define strategies to test
    strategies = [
        BaselineStrategy(),
        SimpleThresholdStrategy(threshold_high=150, threshold_low=30),
        SimpleThresholdStrategy(threshold_high=200, threshold_low=0),
        TimeBasedStrategy(),
        SpreadStrategy(threshold=50),
        SpreadStrategy(threshold=30),
        DataDrivenStrategy(),
        OptimalStrategy(df['mfrr_price_up']),
    ]

    # Run comparison
    print("\n" + "=" * 70)
    print("RUNNING STRATEGY COMPARISON")
    print("=" * 70)

    comparison = engine.compare_strategies(strategies, verbose=True)

    # Print comparison table
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON RESULTS")
    print("=" * 70)

    # Format for display
    display_cols = ['Strategy', 'Net Profit (EUR)', 'Cycles', 'Profit/Cycle',
                    'Utilization %', 'Avg Discharge Price']
    print("\n" + comparison[display_cols].to_string(index=False))

    # Detailed results for top strategies
    print("\n" + "=" * 70)
    print("DETAILED RESULTS - TOP 3 STRATEGIES")
    print("=" * 70)

    top_strategies = comparison.head(3)['Strategy'].tolist()

    for strat_name in top_strategies:
        for strat in strategies:
            if strat.name == strat_name:
                result = engine.run(strat)
                print("\n" + result.summary())
                break

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"comparison_{timestamp}.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nResults saved to: {comparison_path}")

    # Generate monthly breakdown for best strategy
    print("\n" + "=" * 70)
    print("MONTHLY PERFORMANCE - BEST STRATEGY")
    print("=" * 70)

    best_strategy_name = comparison.iloc[0]['Strategy']
    best_strategy = None
    for s in strategies:
        if s.name == best_strategy_name:
            best_strategy = s
            break

    if best_strategy:
        result = engine.run(best_strategy)

        # Build monthly summary
        trades_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'revenue': t.revenue_eur,
                'degradation': t.degradation_eur,
                'net_profit': t.net_profit_eur,
                'action': t.action
            }
            for t in result.trades
        ])
        trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')

        monthly = trades_df.groupby('month').agg({
            'revenue': 'sum',
            'degradation': 'sum',
            'net_profit': 'sum',
            'action': lambda x: (x == 'discharge').sum()
        }).rename(columns={'action': 'discharge_hours'})

        monthly['profit_per_hour'] = monthly['net_profit'] / (
            trades_df.groupby('month').size()
        )

        print(f"\nMonthly Performance for '{best_strategy_name}':")
        print(monthly.round(0).to_string())

        # Save monthly report
        monthly_path = output_dir / f"monthly_{timestamp}.csv"
        monthly.to_csv(monthly_path)

    # Final summary
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    baseline_profit = comparison[comparison['Strategy'] == 'Baseline (Idle)']['Net Profit (EUR)'].values[0]
    best_profit = comparison.iloc[0]['Net Profit (EUR)']
    best_name = comparison.iloc[0]['Strategy']

    print(f"""
Best Strategy: {best_name}
Net Profit: {best_profit:,.0f} EUR

Compared to Baseline:
  - Incremental Profit: {best_profit - baseline_profit:,.0f} EUR
  - Improvement: {(best_profit - baseline_profit) / max(abs(baseline_profit), 1) * 100:,.1f}%

Annualized Estimates (based on {len(df):,} hours of data):
  - Annual Profit: {best_profit * 8760 / len(df):,.0f} EUR
  - Monthly Average: {best_profit * 730 / len(df):,.0f} EUR
""")

    return comparison


def quick_test():
    """Quick test with limited data."""
    data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"

    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)

    # Use last 3 months only
    df = df.tail(3 * 30 * 24)

    battery_config = BatteryConfig()
    engine = BacktestEngine(df, battery_config)

    strategies = [
        BaselineStrategy(),
        TimeBasedStrategy(),
        DataDrivenStrategy(),
    ]

    comparison = engine.compare_strategies(strategies)
    print("\nQuick Test Results:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    run_full_backtest()
