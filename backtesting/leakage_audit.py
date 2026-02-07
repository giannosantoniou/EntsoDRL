"""
Data Leakage Audit Script

I check for various forms of data leakage in the backtesting framework.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def audit_feature_leakage():
    """
    I check if mfrr_spread feature causes target leakage.

    mfrr_spread = mfrr_price_up - mfrr_price_down

    If we use mfrr_spread to predict mfrr_price_up, we're essentially
    using the answer to predict the answer!
    """
    print("=" * 70)
    print("AUDIT 1: FEATURE LEAKAGE (mfrr_spread)")
    print("=" * 70)

    # Load features
    data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)

    # Check correlation between mfrr_spread and mfrr_price_up
    corr = df['mfrr_spread'].corr(df['mfrr_price_up'])
    print(f"\nCorrelation between mfrr_spread and mfrr_price_up: {corr:.4f}")

    # The relationship: mfrr_price_up = mfrr_spread + mfrr_price_down
    # Let's verify
    df['reconstructed_up'] = df['mfrr_spread'] + df['mfrr_price_down']
    reconstruction_error = (df['reconstructed_up'] - df['mfrr_price_up']).abs().mean()
    print(f"Reconstruction error (spread + down = up): {reconstruction_error:.6f}")

    # Test 1: Predict price_up WITH mfrr_spread (LEAKY)
    print("\n--- Test 1: With mfrr_spread (LEAKY) ---")
    features_leaky = ['mfrr_spread', 'hour_sin', 'hour_cos', 'is_weekend']
    X = df[features_leaky].dropna()
    y = df.loc[X.index, 'mfrr_price_up']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae_leaky = mean_absolute_error(y_test, y_pred)
    r2_leaky = r2_score(y_test, y_pred)

    print(f"  MAE: {mae_leaky:.2f} EUR/MWh")
    print(f"  R²: {r2_leaky:.4f}")
    print(f"  Feature importances:")
    for f, imp in zip(features_leaky, model.feature_importances_):
        print(f"    {f}: {imp:.4f}")

    # Test 2: Predict price_up WITHOUT mfrr_spread (CLEAN)
    print("\n--- Test 2: Without mfrr_spread (CLEAN) ---")
    features_clean = [
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'is_peak_hour', 'is_solar_hour',
        'price_up_lag_1h', 'price_up_lag_24h',
        'net_cross_border_mw', 'hydro_level_normalized'
    ]
    features_clean = [f for f in features_clean if f in df.columns]

    X = df[features_clean].dropna()
    y = df.loc[X.index, 'mfrr_price_up']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae_clean = mean_absolute_error(y_test, y_pred)
    r2_clean = r2_score(y_test, y_pred)

    print(f"  MAE: {mae_clean:.2f} EUR/MWh")
    print(f"  R²: {r2_clean:.4f}")

    print("\n--- CONCLUSION ---")
    print(f"MAE increase when removing leaky feature: {mae_clean - mae_leaky:.2f} EUR/MWh")
    print(f"R² drop when removing leaky feature: {r2_leaky - r2_clean:.4f}")

    if r2_leaky > 0.9 and r2_clean < 0.5:
        print("\n⚠️  SEVERE LEAKAGE: mfrr_spread is essentially the target variable!")

    return mae_leaky, r2_leaky, mae_clean, r2_clean


def audit_temporal_split():
    """
    I check if the train/test split respects temporal ordering.
    """
    print("\n" + "=" * 70)
    print("AUDIT 2: TEMPORAL SPLIT")
    print("=" * 70)

    data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)
    df = df.sort_index()

    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\nTrain period: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test period:  {test_df.index.min()} to {test_df.index.max()}")

    # Check for overlap
    train_max = train_df.index.max()
    test_min = test_df.index.min()

    if test_min <= train_max:
        print("\n⚠️  WARNING: Train and test periods OVERLAP!")
        return False
    else:
        print("\n✓ CLEAN: Train period ends before test period starts")
        gap = (test_min - train_max).total_seconds() / 3600
        print(f"  Gap between train and test: {gap:.1f} hours")
        return True


def audit_strategy_access():
    """
    I check what data the strategy has access to at decision time.
    """
    print("\n" + "=" * 70)
    print("AUDIT 3: STRATEGY DATA ACCESS")
    print("=" * 70)

    from .strategies import MarketState

    print("\nMarketState fields available at decision time:")
    print("  - timestamp (current)")
    print("  - hour, day_of_week, is_weekend (current)")
    print("  - mfrr_price_up (CURRENT - this is the settlement price)")
    print("  - mfrr_price_down (CURRENT)")
    print("  - mfrr_spread (CURRENT)")
    print("  - price_up_lag_1h, price_up_lag_24h (PAST)")
    print("  - price_up_mean_24h (PAST)")
    print("  - soc, can_discharge, can_charge (CURRENT battery state)")

    print("\n--- ANALYSIS ---")
    print("The strategy sees mfrr_price_up at decision time.")
    print("In REAL trading, you would NOT know the settlement price before bidding!")
    print("")
    print("In mFRR market:")
    print("  1. You submit bid BEFORE the settlement period")
    print("  2. Settlement price is determined AFTER all bids are in")
    print("  3. You cannot know the exact price when making the bid")
    print("")
    print("⚠️  Using current mfrr_price_up for decisions is UNREALISTIC")
    print("   This gives the strategy perfect knowledge of settlement prices!")


def audit_time_based_vs_hybrid():
    """
    I compare Time-Based (no ML) vs AggressiveHybrid to see if ML adds value.
    """
    print("\n" + "=" * 70)
    print("AUDIT 4: TIME-BASED vs AGGRESSIVE HYBRID")
    print("=" * 70)

    from .battery import BatteryConfig
    from .engine import BacktestEngine
    from .strategies import TimeBasedStrategy

    data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)
    df = df.sort_index()

    # Use test split only
    split_idx = int(len(df) * 0.7)
    test_df = df.iloc[split_idx:].copy()

    battery_config = BatteryConfig(
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        degradation_cost=15.0,
    )

    engine = BacktestEngine(test_df, battery_config)

    # Run Time-Based only (no ML)
    time_based = TimeBasedStrategy()
    result = engine.run(time_based, verbose=False)

    print(f"\nTime-Based Strategy (Pure rule-based, NO ML):")
    print(f"  Net Profit:    {result.total_net_profit:>12,.0f} EUR")
    print(f"  Cycles:        {result.total_cycles:>12.1f}")
    print(f"  Profit/Cycle:  {result.profit_per_cycle:>12,.0f} EUR")
    print(f"  Utilization:   {result.capacity_utilization*100:>12.1f}%")

    print("\n--- CONCLUSION ---")
    print("If Time-Based alone achieves similar profit to AggressiveHybrid,")
    print("then ML contribution is minimal and high performance comes from")
    print("simple time-of-day patterns, not sophisticated ML predictions.")


def run_full_audit():
    """Run all audits."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DATA LEAKAGE AUDIT")
    print("=" * 70)

    # Audit 1: Feature leakage
    mae_leaky, r2_leaky, mae_clean, r2_clean = audit_feature_leakage()

    # Audit 2: Temporal split
    temporal_clean = audit_temporal_split()

    # Audit 3: Strategy access
    audit_strategy_access()

    # Audit 4: Time-based comparison
    audit_time_based_vs_hybrid()

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    issues = []

    if r2_leaky > 0.9:
        issues.append("CRITICAL: mfrr_spread feature causes severe target leakage")

    if not temporal_clean:
        issues.append("CRITICAL: Train/test periods overlap")

    issues.append("WARNING: Strategy sees settlement price before bidding (unrealistic)")

    if issues:
        print("\nISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo critical issues found.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_full_audit()
