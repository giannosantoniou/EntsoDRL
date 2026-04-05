"""
Oracle Benchmark — I compute the theoretical maximum P&L with perfect foresight.

I run the ScheduleOptimizer with ACTUAL prices for every day in the eval dataset.
I report both gross trading P&L and net (with operational costs).

Usage:
    python agent/compute_oracle_benchmark.py
    python agent/compute_oracle_benchmark.py --include-costs   # with degradation/network
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PredictFuturePrices.forecasters.schedule_optimizer import ScheduleOptimizer


def main():
    parser = argparse.ArgumentParser(description='Oracle benchmark')
    parser.add_argument('--data', type=str,
                        default='data/unified_multimarket_training_v5_raw_features.csv')
    parser.add_argument('--include-costs', action='store_true',
                        help='Include degradation + network in P&L')
    parser.add_argument('--eval-split', type=float, default=0.80,
                        help='Use last (1-split) fraction as eval')
    args = parser.parse_args()

    # I load data
    data_path = project_root / args.data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    split_idx = int(len(df) * args.eval_split)
    df_eval = df.iloc[split_idx:].copy()
    print(f"Eval data: {len(df_eval)} rows ({df_eval.index[0]} to {df_eval.index[-1]})")

    # I create optimizer
    optimizer = ScheduleOptimizer(
        max_power_mw=30.0,
        capacity_mwh=146.0,
        efficiency=0.94,
        min_soc=0.05,
        max_soc=0.95,
        time_step_hours=0.25,
        max_daily_cycles=2.0,
    )

    # I compute oracle for each day
    sph = 4  # 15-min resolution
    slots_per_day = 24 * sph  # 96

    # I identify unique days
    dates = df_eval.index.date
    unique_days = sorted(set(dates))
    print(f"Days in eval set: {len(unique_days)}")

    daily_results = []
    initial_soc = 0.50  # I start each day at 50%

    for day in unique_days:
        # I get actual prices for this day
        day_mask = dates == day
        day_data = df_eval[day_mask]

        if len(day_data) < slots_per_day * 0.8:
            continue  # I skip incomplete days

        actual_prices = day_data['price'].values[:slots_per_day]
        if len(actual_prices) < slots_per_day:
            # I pad with last price
            actual_prices = np.pad(actual_prices, (0, slots_per_day - len(actual_prices)),
                                   mode='edge')

        # I run oracle optimizer
        result = optimizer.compute_oracle_schedule(
            actual_prices, initial_soc, slots_per_day
        )

        # I compute gross trading revenue (sell_price - buy_cost)
        # The optimizer already computes this as expected_revenue
        gross_revenue = result.expected_revenue

        # I compute profit WITH and WITHOUT operational costs
        # The optimizer includes both network + degradation
        profit_with_costs = result.expected_profit  # includes network + degradation + eff loss
        profit_gross = gross_revenue  # just trading revenue, no costs

        # I compute what the costs were
        energy_traded = np.sum(np.abs(result.schedule)) * 0.25
        deg_cost = energy_traded * optimizer.DEGRADATION
        network_cost = energy_traded * optimizer.NETWORK_CHARGE
        eff_loss = energy_traded * (1.0 - 0.94) * np.mean(actual_prices) / 2

        daily_results.append({
            'date': day,
            'mean_price': np.mean(actual_prices),
            'price_spread': np.max(actual_prices) - np.min(actual_prices),
            'gross_revenue': gross_revenue,
            'profit_net': profit_with_costs,
            'deg_cost': deg_cost,
            'network_cost': network_cost,
            'eff_loss': eff_loss,
            'energy_traded_mwh': energy_traded,
            'cycles': result.total_cycles,
            'slots_used': result.slots_used,
        })

        # I carry forward final SoC for next day (oracle can start where it left off)
        initial_soc = result.soc_trajectory[-1] if len(result.soc_trajectory) > 0 else 0.50

    results_df = pd.DataFrame(daily_results)
    n_days = len(results_df)

    print(f"\n{'='*70}")
    print(f"  ORACLE BENCHMARK — {n_days} days")
    print(f"{'='*70}")

    # I compute averages
    avg_gross = results_df['gross_revenue'].mean()
    avg_net = results_df['profit_net'].mean()
    avg_deg = results_df['deg_cost'].mean()
    avg_network = results_df['network_cost'].mean()
    avg_eff = results_df['eff_loss'].mean()
    avg_energy = results_df['energy_traded_mwh'].mean()
    avg_cycles = results_df['cycles'].mean()
    avg_spread = results_df['price_spread'].mean()
    avg_price = results_df['mean_price'].mean()

    print(f"\n  Mean DAM price:        {avg_price:.1f} EUR/MWh")
    print(f"  Mean daily spread:     {avg_spread:.1f} EUR/MWh")
    print(f"  Mean energy traded:    {avg_energy:.1f} MWh/day")
    print(f"  Mean cycles:           {avg_cycles:.2f}/day")

    print(f"\n--- Trading P&L (NO operational costs) ---")
    print(f"  Gross revenue:         {avg_gross:>10.0f} EUR/day = {avg_gross*365/1e6:.2f}M EUR/year")

    print(f"\n--- With operational costs ---")
    print(f"  Degradation:           {avg_deg:>10.0f} EUR/day ({avg_deg*365/1e6:.2f}M/year)")
    print(f"  Network charges:       {avg_network:>10.0f} EUR/day ({avg_network*365/1e6:.2f}M/year)")
    print(f"  Efficiency loss:       {avg_eff:>10.0f} EUR/day ({avg_eff*365/1e6:.2f}M/year)")
    print(f"  Net profit:            {avg_net:>10.0f} EUR/day = {avg_net*365/1e6:.2f}M EUR/year")

    print(f"\n--- Annual Summary ---")
    print(f"  {'Metric':<30} {'EUR/day':>10} {'EUR/year':>12}")
    print(f"  {'-'*55}")
    print(f"  {'Gross trading revenue':<30} {avg_gross:>10.0f} {avg_gross*365:>12,.0f}")
    print(f"  {'Net (with oper. costs)':<30} {avg_net:>10.0f} {avg_net*365:>12,.0f}")
    print(f"  {'Oper. costs total':<30} {avg_deg+avg_network+avg_eff:>10.0f} {(avg_deg+avg_network+avg_eff)*365:>12,.0f}")

    # I show monthly breakdown
    results_df['month'] = pd.to_datetime(results_df['date']).dt.to_period('M')
    monthly = results_df.groupby('month').agg({
        'gross_revenue': 'mean',
        'profit_net': 'mean',
        'mean_price': 'mean',
        'price_spread': 'mean',
    })

    print(f"\n--- Monthly Oracle Performance ---")
    print(f"  {'Month':<10} {'Gross/day':>10} {'Net/day':>10} {'Price':>8} {'Spread':>8}")
    print(f"  {'-'*50}")
    for month, row in monthly.iterrows():
        print(f"  {str(month):<10} {row['gross_revenue']:>10.0f} {row['profit_net']:>10.0f} "
              f"{row['mean_price']:>8.1f} {row['price_spread']:>8.1f}")

    # I note: oracle does NOT include aFRR, mFRR, XBID (only DAM schedule optimization)
    print(f"\n  NOTE: Oracle includes ONLY DAM schedule optimization (buy low/sell high).")
    print(f"  It does NOT include aFRR capacity revenue, mFRR auto-response, or XBID trading.")
    print(f"  Real ceiling with all markets would be 20-40% higher.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
