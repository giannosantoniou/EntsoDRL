"""
Run ML-Enhanced Backtests v2 - Fixed Feature Handling
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
    HybridMLTimeStrategy,
    AggressiveHybridStrategy,
)
from .ml_strategy_v2 import create_ml_strategies_v2


def run_ml_backtest_v2(data_path: str = None, output_dir: str = None):
    """
    Run backtest with ML strategies v2.
    """
    # Paths
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ML-ENHANCED mFRR BIDDING STRATEGY BACKTEST v2")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)
    df = df.sort_index()
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Split data: 70% train, 30% test
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

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

    # Train ML models
    print("\n3. Training ML Models...")
    ml_strategies = create_ml_strategies_v2(
        train_df, test_df,
        max_power_mw=battery_config.max_power_mw
    )

    # Create hybrid strategies using the ML Ensemble predictor
    hybrid_strategies = []
    for ml_strat in ml_strategies:
        if 'Ensemble' in ml_strat.name:
            # Conservative hybrid
            hybrid = HybridMLTimeStrategy(
                ml_predictor=ml_strat,
                max_power_mw=battery_config.max_power_mw,
                min_discharge_price=100.0,
                max_charge_price=80.0
            )
            hybrid_strategies.append(hybrid)

            # Aggressive hybrid
            aggressive = AggressiveHybridStrategy(
                ml_predictor=ml_strat,
                max_power_mw=battery_config.max_power_mw
            )
            hybrid_strategies.append(aggressive)
            break

    # All strategies
    all_strategies = [
        BaselineStrategy(),
        SimpleThresholdStrategy(threshold_high=150, threshold_low=30),
        TimeBasedStrategy(),
        SpreadStrategy(threshold=30),
        DataDrivenStrategy(),
    ] + ml_strategies + hybrid_strategies

    # Run backtest
    print("\n" + "=" * 70)
    print("4. RUNNING BACKTEST ON TEST DATA")
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
    print("5. RESULTS COMPARISON")
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
    results_path = output_dir / f"ml_v2_comparison_{timestamp}.csv"
    comparison.to_csv(results_path, index=False)

    # ML vs Rule-based analysis
    print("\n" + "=" * 70)
    print("6. ML vs RULE-BASED COMPARISON")
    print("=" * 70)

    ml_names = [s.name for s in ml_strategies]
    ml_results = comparison[comparison['Strategy'].isin(ml_names)]
    rule_results = comparison[~comparison['Strategy'].isin(ml_names)]
    rule_results = rule_results[rule_results['Strategy'] != 'Baseline (Idle)']

    if len(ml_results) > 0 and len(rule_results) > 0:
        best_ml = ml_results.iloc[0]
        best_rule = rule_results.iloc[0]

        print(f"\nBest ML Strategy: {best_ml['Strategy']}")
        print(f"  Net Profit:     {best_ml['Net Profit (EUR)']:>12,.0f} EUR")
        print(f"  Cycles:         {best_ml['Cycles']:>12.1f}")
        print(f"  Profit/Cycle:   {best_ml['Profit/Cycle']:>12,.0f} EUR")
        print(f"  Discharge Avg:  {best_ml['Avg Discharge Price']:>12.1f} EUR/MWh")

        print(f"\nBest Rule-Based: {best_rule['Strategy']}")
        print(f"  Net Profit:     {best_rule['Net Profit (EUR)']:>12,.0f} EUR")
        print(f"  Cycles:         {best_rule['Cycles']:>12.1f}")
        print(f"  Profit/Cycle:   {best_rule['Profit/Cycle']:>12,.0f} EUR")
        print(f"  Discharge Avg:  {best_rule['Avg Discharge Price']:>12.1f} EUR/MWh")

        improvement = best_ml['Net Profit (EUR)'] - best_rule['Net Profit (EUR)']
        pct = improvement / max(abs(best_rule['Net Profit (EUR)']), 1) * 100

        print(f"\n{'ML WINS!' if improvement > 0 else 'Rule-Based Wins'}")
        print(f"  Difference: {improvement:+,.0f} EUR ({pct:+.1f}%)")

    # Annualized projections
    print("\n" + "=" * 70)
    print("7. ANNUALIZED PROJECTIONS (Top 5)")
    print("=" * 70)

    hours_year = 8760
    test_hours = len(test_df)

    print(f"\nBased on {test_hours} hours of test data:\n")
    print(f"{'Strategy':<35} {'Test Profit':>15} {'Annual Est.':>15}")
    print("-" * 65)

    for _, row in comparison.head(5).iterrows():
        annual = row['Net Profit (EUR)'] * hours_year / test_hours
        print(f"{row['Strategy']:<35} {row['Net Profit (EUR)']:>15,.0f} {annual:>15,.0f}")

    # Summary
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    best = comparison.iloc[0]
    annual = best['Net Profit (EUR)'] * hours_year / test_hours

    print(f"""
WINNER: {best['Strategy']}

Test Period Performance (7 months):
  - Net Profit:       {best['Net Profit (EUR)']:>12,.0f} EUR
  - Full Cycles:      {best['Cycles']:>12.1f}
  - Profit per Cycle: {best['Profit/Cycle']:>12,.0f} EUR
  - Utilization:      {best['Utilization %']:>12.1f}%

Annualized Projection:
  - Annual Profit:    {annual:>12,.0f} EUR
  - Monthly Average:  {annual/12:>12,.0f} EUR
  - Daily Average:    {annual/365:>12,.0f} EUR
""")

    return comparison


if __name__ == "__main__":
    run_ml_backtest_v2()
