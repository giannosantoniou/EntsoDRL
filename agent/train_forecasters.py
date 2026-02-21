"""
Train Market Forecasters

I train the unified MarketForecaster (DAM + IntraDay + Imbalance) on
historical data and evaluate per-horizon accuracy.

Usage:
    python agent/train_forecasters.py --data data/unified_multimarket_training_v2.csv
    python agent/train_forecasters.py --eval  # Evaluate only (load existing model)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# I add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.market_forecaster import MarketForecaster, DAMDayAheadForecaster


def train_forecasters(
    data_path: str = "data/unified_multimarket_training_v2.csv",
    output_path: str = "models/market_forecaster.pkl",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> MarketForecaster:
    """
    I train all sub-forecasters with temporal train/val/test split.

    Split: 70% train, 15% validation, 15% test (held out for evaluation).
    """
    print("=" * 60)
    print("MARKET FORECASTER TRAINING")
    print("=" * 60)

    # I resolve data path
    resolved = Path(data_path)
    if not resolved.exists():
        resolved = project_root / data_path
    if not resolved.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    print(f"Loading data from {resolved}...")
    df = pd.read_csv(resolved, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows, {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {len(df.columns)}")

    # I compute split indices
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    print(f"\nTemporal split:")
    print(f"  Train: {train_end} rows ({df.index[0]} to {df.index[train_end - 1]})")
    print(f"  Val:   {val_end - train_end} rows ({df.index[train_end]} to {df.index[val_end - 1]})")
    print(f"  Test:  {n - val_end} rows ({df.index[val_end]} to {df.index[-1]})")

    # I train the forecaster
    forecaster = MarketForecaster()
    forecaster.fit(df, train_end, val_end)

    # I save the trained model
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(str(output))

    # I evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    evaluate_forecaster(forecaster, df, val_end)

    return forecaster


def evaluate_forecaster(
    forecaster: MarketForecaster,
    df: pd.DataFrame,
    test_start_idx: int,
    n_samples: int = 500
) -> dict:
    """I evaluate all sub-forecasters on the test set."""
    sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

    # I sample test indices evenly
    test_range = range(test_start_idx, len(df) - 24 * sph)
    step = max(1, len(test_range) // n_samples)
    test_indices = list(test_range)[::step][:n_samples]

    results = {
        'dam_mae': {}, 'dam_mape': {},
        'id_mae': {}, 'id_mape': {},
        'imb_accuracy': 0.0, 'imb_mae': 0.0
    }

    # I evaluate DAM forecaster
    print("\nDAM Forecaster Evaluation:")
    for horizon_h in [1, 4, 8, 12, 24]:
        errors = []
        pct_errors = []
        target_steps = horizon_h * sph

        for idx in test_indices:
            target_idx = idx + target_steps
            if target_idx >= len(df):
                continue

            pred = forecaster.dam_forecaster.predict(df, idx, horizon_h)
            actual = df.iloc[target_idx]['price']

            if not np.isnan(actual) and abs(actual) > 1.0:
                errors.append(abs(pred[-1] - actual))
                pct_errors.append(abs(pred[-1] - actual) / abs(actual) * 100)

        if errors:
            mae = np.mean(errors)
            mape = np.mean(pct_errors)
            results['dam_mae'][horizon_h] = mae
            results['dam_mape'][horizon_h] = mape
            print(f"  {horizon_h:2d}h ahead: MAE={mae:.2f} EUR/MWh, MAPE={mape:.1f}%")

    # I evaluate IntraDay forecaster
    print("\nIntraDay Forecaster Evaluation:")
    for horizon_h in [1, 2, 4, 8]:
        errors = []
        pct_errors = []
        target_steps = horizon_h * sph

        # I determine the target column
        target_col = 'isp1_price_up' if 'isp1_price_up' in df.columns else 'price'

        for idx in test_indices:
            target_idx = idx + target_steps
            if target_idx >= len(df):
                continue

            preds = forecaster.intraday_forecaster.predict(df, idx)
            pred = preds.get(horizon_h, 80.0)
            actual = df.iloc[target_idx][target_col]

            if not np.isnan(actual) and abs(actual) > 1.0:
                errors.append(abs(pred - actual))
                pct_errors.append(abs(pred - actual) / abs(actual) * 100)

        if errors:
            mae = np.mean(errors)
            mape = np.mean(pct_errors)
            results['id_mae'][horizon_h] = mae
            results['id_mape'][horizon_h] = mape
            print(f"  {horizon_h:2d}h ahead: MAE={mae:.2f} EUR/MWh, MAPE={mape:.1f}%")

    # I evaluate Imbalance forecaster
    print("\nImbalance Forecaster Evaluation:")
    correct_dir = 0
    mag_errors = []
    total_eval = 0

    for idx in test_indices:
        target_end = idx + 4 * sph
        if target_end >= len(df) or 'net_imbalance_mw' not in df.columns:
            continue

        pred = forecaster.imbalance_forecaster.predict(df, idx)
        actual_avg = df.iloc[idx + 1:target_end + 1]['net_imbalance_mw'].mean()

        if np.isnan(actual_avg):
            continue

        actual_dir = 1.0 if actual_avg > 30 else (-1.0 if actual_avg < -30 else 0.0)
        if pred['direction'] == actual_dir:
            correct_dir += 1
        mag_errors.append(abs(pred['magnitude'] - abs(actual_avg)))
        total_eval += 1

    if total_eval > 0:
        accuracy = correct_dir / total_eval
        mag_mae = np.mean(mag_errors)
        results['imb_accuracy'] = accuracy
        results['imb_mae'] = mag_mae
        print(f"  Direction accuracy: {accuracy:.1%} ({correct_dir}/{total_eval})")
        print(f"  Magnitude MAE: {mag_mae:.1f} MW")
    else:
        print("  No test samples available")

    return results


def train_day_ahead_forecaster(
    data_path: str = "data/unified_multimarket_training_v2.csv",
    serbia_path: str = "data/serbia_dam_historical.csv",
    dam_path: str = "data/dam_prices_2021_2026.csv",
    output_path: str = "models/day_ahead_forecaster.pkl",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> DAMDayAheadForecaster:
    """
    I train the D+1 Day-Ahead forecaster on historical Greek + Serbia DAM data.

    Split: 70% train, 15% validation, 15% test (held out for evaluation).
    """
    print("=" * 60)
    print("D+1 DAY-AHEAD FORECASTER TRAINING")
    print("=" * 60)

    # I load Greek DAM prices
    dam_resolved = Path(dam_path)
    if not dam_resolved.exists():
        dam_resolved = project_root / dam_path
    if not dam_resolved.exists():
        # I try the training data as fallback
        dam_resolved = Path(data_path)
        if not dam_resolved.exists():
            dam_resolved = project_root / data_path
    if not dam_resolved.exists():
        raise FileNotFoundError(f"DAM data not found at {dam_path} or {data_path}")

    print(f"Loading Greek DAM data from {dam_resolved}...")
    df = pd.read_csv(dam_resolved, index_col=0, parse_dates=True)
    print(f"  Loaded {len(df)} rows, {df.index.min()} to {df.index.max()}")

    # I load Serbia DAM prices
    serbia_resolved = Path(serbia_path)
    if not serbia_resolved.exists():
        serbia_resolved = project_root / serbia_path
    if not serbia_resolved.exists():
        raise FileNotFoundError(f"Serbia DAM data not found at {serbia_path}")

    print(f"Loading Serbia DAM data from {serbia_resolved}...")
    serbia_df = pd.read_csv(serbia_resolved, index_col=0, parse_dates=True)
    print(f"  Loaded {len(serbia_df)} rows, {serbia_df.index.min()} to {serbia_df.index.max()}")

    # I ensure both have 'price' column
    if 'price' not in df.columns:
        price_cols = [c for c in df.columns if 'price' in c.lower() or 'dam' in c.lower()]
        if price_cols:
            df['price'] = df[price_cols[0]]
        else:
            raise ValueError(f"No 'price' column found in {dam_resolved}")

    if 'price' not in serbia_df.columns:
        price_cols = [c for c in serbia_df.columns if 'price' in c.lower()]
        if price_cols:
            serbia_df['price'] = serbia_df[price_cols[0]]
        else:
            raise ValueError(f"No 'price' column found in {serbia_resolved}")

    # I find overlapping date range
    gr_dates = set(df.index.normalize().unique())
    rs_dates = set(serbia_df.index.normalize().unique())
    overlap_dates = sorted(gr_dates & rs_dates)
    print(f"\n  Overlapping dates: {len(overlap_dates)} days "
          f"({overlap_dates[0]} to {overlap_dates[-1]})")

    # I compute temporal split based on overlapping dates
    n_days = len(overlap_dates)
    train_end_day = int(n_days * train_fraction)
    val_end_day = int(n_days * (train_fraction + val_fraction))

    train_end = pd.Timestamp(overlap_dates[train_end_day])
    val_end = pd.Timestamp(overlap_dates[val_end_day])

    print(f"\nTemporal split:")
    print(f"  Train: {overlap_dates[0]} to {overlap_dates[train_end_day - 1]} ({train_end_day} days)")
    print(f"  Val:   {overlap_dates[train_end_day]} to {overlap_dates[val_end_day - 1]} ({val_end_day - train_end_day} days)")
    print(f"  Test:  {overlap_dates[val_end_day]} to {overlap_dates[-1]} ({n_days - val_end_day} days)")

    # I also add serbia_dam_price column to df for feature computation
    # (used by build_day_features for spread history)
    if 'serbia_dam_price' not in df.columns:
        serbia_hourly = serbia_df['price'].resample('h').first()
        df = df.copy()
        df['serbia_dam_price'] = serbia_hourly.reindex(df.index, method='ffill')

    # I train the forecaster
    forecaster = DAMDayAheadForecaster()
    forecaster.fit(df, serbia_df, train_end, val_end)

    # I save the trained model
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(str(output))

    # I evaluate on test set
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    _evaluate_day_ahead(forecaster, df, serbia_df, overlap_dates, val_end_day)

    return forecaster


def _evaluate_day_ahead(
    forecaster: DAMDayAheadForecaster,
    df: pd.DataFrame,
    serbia_df: pd.DataFrame,
    overlap_dates: list,
    test_start_day: int,
    max_test_days: int = 100,
) -> dict:
    """I evaluate the D+1 forecaster on the test set."""
    sph = max(1, int(round(
        1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600
    ))) if len(df) > 1 else 1

    test_dates = overlap_dates[test_start_day:]
    if len(test_dates) > max_test_days:
        step = len(test_dates) // max_test_days
        test_dates = test_dates[::step][:max_test_days]

    per_hour_errors = {h: [] for h in range(24)}
    per_hour_pct_errors = {h: [] for h in range(24)}
    daily_profile_rmse = []
    p10_coverage = 0
    p90_coverage = 0
    total_hours = 0

    for day_date in test_dates:
        day_mask = df.index.normalize() == day_date
        day_indices = np.where(day_mask)[0]
        if len(day_indices) < 24 * sph:
            continue

        day_idx = day_indices[0]

        # I get Serbia D+1 prices
        serbia_day = serbia_df.loc[serbia_df.index.normalize() == day_date, 'price']
        if len(serbia_day) < 24:
            continue

        serbia_24h = np.zeros(24)
        for h in range(24):
            hour_mask = serbia_day.index.hour == h
            hour_vals = serbia_day[hour_mask].values
            if len(hour_vals) > 0:
                serbia_24h[h] = hour_vals[0]
            else:
                serbia_24h[h] = serbia_day.mean()

        if day_idx < 7 * 24 * sph:
            continue

        # I predict
        result = forecaster.predict_day(df, day_idx, serbia_24h, sph)

        # I compare with actuals
        actuals = np.zeros(24)
        valid = True
        for h in range(24):
            target_idx = day_idx + h * sph
            if target_idx >= len(df):
                valid = False
                break
            actuals[h] = df.iloc[target_idx]['price']
            if np.isnan(actuals[h]):
                valid = False
                break

        if not valid:
            continue

        for h in range(24):
            err = abs(result['hourly'][h] - actuals[h])
            per_hour_errors[h].append(err)
            if abs(actuals[h]) > 1.0:
                per_hour_pct_errors[h].append(err / abs(actuals[h]) * 100)

            # I check quantile calibration
            total_hours += 1
            if actuals[h] >= result['p10'][h]:
                p10_coverage += 1
            if actuals[h] <= result['p90'][h]:
                p90_coverage += 1

        # I compute daily profile RMSE
        profile_rmse = np.sqrt(np.mean((result['hourly'] - actuals) ** 2))
        daily_profile_rmse.append(profile_rmse)

    # I print results
    print(f"\nPer-hour MAE (EUR/MWh):")
    total_mae = []
    total_mape = []
    for h in range(24):
        if per_hour_errors[h]:
            mae = np.mean(per_hour_errors[h])
            mape = np.mean(per_hour_pct_errors[h]) if per_hour_pct_errors[h] else 0
            total_mae.append(mae)
            total_mape.append(mape)
            print(f"  {h:02d}:00  MAE={mae:6.2f}  MAPE={mape:5.1f}%")

    if total_mae:
        print(f"\nOverall MAE: {np.mean(total_mae):.2f} EUR/MWh")
        print(f"Overall MAPE: {np.mean(total_mape):.1f}%")
    if daily_profile_rmse:
        print(f"Daily profile RMSE: {np.mean(daily_profile_rmse):.2f} EUR/MWh")
    if total_hours > 0:
        print(f"\nQuantile calibration ({total_hours} test hours):")
        print(f"  P10 coverage: {p10_coverage / total_hours:.1%} (target: 90%)")
        print(f"  P90 coverage: {p90_coverage / total_hours:.1%} (target: 90%)")

    return {
        'per_hour_mae': {h: np.mean(per_hour_errors[h]) for h in range(24) if per_hour_errors[h]},
        'overall_mae': np.mean(total_mae) if total_mae else None,
        'daily_profile_rmse': np.mean(daily_profile_rmse) if daily_profile_rmse else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Market Forecasters")
    parser.add_argument("--data", type=str,
                        default="data/unified_multimarket_training_v2.csv",
                        help="Path to training data")
    parser.add_argument("--output", type=str,
                        default="models/market_forecaster.pkl",
                        help="Output path for trained model")
    parser.add_argument("--eval", type=str, default=None,
                        help="Evaluate existing model (skip training)")
    parser.add_argument("--day_ahead", action="store_true",
                        help="Train only the D+1 day-ahead forecaster")
    parser.add_argument("--serbia_data", type=str,
                        default="data/serbia_dam_historical.csv",
                        help="Path to Serbia DAM historical data")
    parser.add_argument("--dam_data", type=str,
                        default="data/dam_prices_2021_2026.csv",
                        help="Path to Greek DAM prices")

    args = parser.parse_args()

    if args.day_ahead:
        train_day_ahead_forecaster(
            data_path=args.data,
            serbia_path=args.serbia_data,
            dam_path=args.dam_data,
            output_path=args.output if args.output != "models/market_forecaster.pkl"
                else "models/day_ahead_forecaster.pkl",
        )
    elif args.eval:
        print(f"Loading existing forecaster from {args.eval}...")
        forecaster = MarketForecaster.load(args.eval)

        resolved = Path(args.data)
        if not resolved.exists():
            resolved = project_root / args.data
        df = pd.read_csv(resolved, index_col=0, parse_dates=True)

        n = len(df)
        val_end = int(n * 0.85)
        evaluate_forecaster(forecaster, df, val_end)
    else:
        train_forecasters(
            data_path=args.data,
            output_path=args.output
        )
