"""
Create Unified Multi-Market Training Dataset

I merge DAM data with balancing market data to create a single unified training
dataset for the multi-market environment.

Two modes:
1. ADMIE-based (preferred): Real ADMIE balancing data + DAM prices
   - Uses admie_market_data_combined.csv + dam_prices_2021_2026.csv
   - Real mFRR energy/aFRR capacity prices, system load, RES, imbalance
   - Only IntraDay is synthetic (no XBID data available)

2. Synthetic fallback: feasible_data_with_dam.csv + afrr_mfrr_combined_v2.csv
   - All balancing data synthesized via OU processes
   - Used when ADMIE data is not available

Output: data/unified_multimarket_training.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def load_dam_data(path: str = "feasible_data_with_dam.csv") -> pd.DataFrame:
    """Load DAM training data with commitments."""
    data_dir = Path(__file__).parent
    filepath = data_dir / path

    print(f"Loading DAM data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # I ensure the index is a DatetimeIndex and timezone-aware
    if not isinstance(df.index, pd.DatetimeIndex):
        # I handle mixed timezone formats by converting to UTC first
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Athens')
    elif df.index.tz is None:
        try:
            df.index = df.index.tz_localize('Europe/Athens', ambiguous='infer')
        except Exception:
            df.index = df.index.tz_localize('UTC').tz_convert('Europe/Athens')

    print(f"  Loaded {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def load_afrr_mfrr_data(path: str = "afrr_mfrr_combined_v2.csv") -> pd.DataFrame:
    """Load aFRR/mFRR balancing market data."""
    data_dir = Path(__file__).parent
    filepath = data_dir / path

    print(f"Loading aFRR/mFRR data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # I ensure the index is a DatetimeIndex and timezone-aware
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('Europe/Athens')
    elif df.index.tz is None:
        try:
            df.index = df.index.tz_localize('Europe/Athens', ambiguous='infer')
        except Exception:
            df.index = df.index.tz_localize('UTC').tz_convert('Europe/Athens')

    print(f"  Loaded {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")

    return df


def create_unified_dataset_from_admie(
    dam_path: str = 'dam_prices_2021_2026.csv',
    admie_path: str = 'admie_market_data_combined.csv',
    output_path: str = 'unified_multimarket_training.csv'
) -> pd.DataFrame:
    """
    I create a unified training dataset from real ADMIE balancing data + DAM prices.

    This replaces synthetic balancing data with actual market data from ADMIE,
    which has realistic price levels (mFRR spread ~380 EUR vs synthetic ~80 EUR)
    and real system features (load, RES production, imbalance).

    Only IntraDay (XBID) prices remain synthetic since no real data is available.
    """
    data_dir = Path(__file__).parent

    # Step 1: I load DAM prices and normalize to hourly timezone-naive index
    print("=" * 60)
    print("CREATING UNIFIED DATASET FROM REAL ADMIE DATA")
    print("=" * 60)

    dam_filepath = data_dir / dam_path
    print(f"\nLoading DAM prices from {dam_filepath}...")
    dam_df = pd.read_csv(dam_filepath, index_col=0)

    # I parse timestamps as UTC (handles mixed +02:00/+03:00 DST offsets),
    # convert to Athens local time, then strip tz to get naive local timestamps
    dam_df.index = pd.to_datetime(dam_df.index, utc=True)
    dam_df.index = dam_df.index.tz_convert('Europe/Athens')
    dam_df.index = dam_df.index.tz_localize(None)

    # I resample to hourly (DAM data may have sub-hourly resolution at the tail)
    dam_df = dam_df.resample('h').mean().dropna()
    dam_df.columns = ['price']  # I ensure consistent column name
    print(f"  DAM: {len(dam_df)} hourly rows, {dam_df.index.min()} to {dam_df.index.max()}")
    print(f"  DAM price: mean={dam_df['price'].mean():.2f}, std={dam_df['price'].std():.2f}")

    # Step 2: I load ADMIE balancing market data
    admie_filepath = data_dir / admie_path
    print(f"\nLoading ADMIE data from {admie_filepath}...")
    admie_df = pd.read_csv(admie_filepath, parse_dates=['timestamp'])
    admie_df = admie_df.set_index('timestamp')

    # I strip timezone if present (ADMIE is typically naive)
    if admie_df.index.tz is not None:
        admie_df.index = admie_df.index.tz_localize(None)

    print(f"  ADMIE: {len(admie_df)} rows, {admie_df.index.min()} to {admie_df.index.max()}")

    # Step 3: I inner join on timestamp (overlap: 2024-01-01 → 2026-01-17)
    print("\nJoining DAM + ADMIE on timestamp...")
    unified = dam_df.join(admie_df, how='inner')
    print(f"  Overlap: {len(unified)} rows, {unified.index.min()} to {unified.index.max()}")

    # Step 4: I map ADMIE columns → environment column names
    # I rename mFRR energy activation prices (the high-value spreads the agent trades)
    # Note: ADMIE's 'mfrr_price_up/down' are *capacity* prices (~0.65 EUR),
    #        while 'mfrr_energy_price_up/down' are *energy activation* prices (~586/206 EUR)
    column_mapping = {
        'mfrr_energy_price_up': 'mfrr_price_up',
        'mfrr_energy_price_down': 'mfrr_price_down',
        'afrr_price_up': 'afrr_cap_up_price',
        'afrr_price_down': 'afrr_cap_down_price',
        'afrr_requirements_up': 'afrr_cap_up_qty',
        'afrr_requirements_down': 'afrr_cap_down_qty',
        'system_load_mw': 'load_mw',
        'res_production_mw': 'res_total_mw',
        'system_deviation_mwh': 'net_imbalance_mw',
    }

    # I drop ADMIE's own mfrr_price_up/down (capacity) before renaming
    # to avoid collision with the energy prices I'm mapping in
    admie_cap_cols = ['mfrr_price_up', 'mfrr_price_down']
    for col in admie_cap_cols:
        if col in unified.columns:
            unified = unified.rename(columns={col: f'mfrr_cap_{col.split("_", 1)[1]}'})

    unified = unified.rename(columns=column_mapping)

    # Step 5: I derive aFRR energy prices from imbalance_price
    # I use imbalance_price as aFRR energy up proxy, 30% as down proxy
    if 'imbalance_price' in unified.columns:
        unified['afrr_up'] = unified['imbalance_price']
        unified['afrr_down'] = unified['imbalance_price'] * 0.3
        unified['afrr_down'] = unified['afrr_down'].clip(lower=0)
    else:
        unified['afrr_up'] = 80.0
        unified['afrr_down'] = 24.0

    # I compute mFRR spread from real energy prices
    unified['mfrr_spread'] = unified['mfrr_price_up'] - unified['mfrr_price_down']

    # Step 6: I estimate solar/wind split from total RES + time-of-day
    # ADMIE provides total RES but not the breakdown
    if 'res_total_mw' in unified.columns:
        hours = unified.index.hour.values
        # I estimate solar with a bell curve centered at noon (hours 6-18)
        solar_fraction = np.maximum(0, np.sin(np.pi * (hours - 5) / 13))
        # I adjust the fraction: solar is ~60% of RES during daylight, 0% at night
        solar_fraction = solar_fraction * 0.6
        solar_fraction[hours < 6] = 0.0
        solar_fraction[hours > 19] = 0.0

        unified['solar'] = unified['res_total_mw'] * solar_fraction
        unified['wind_onshore'] = unified['res_total_mw'] * (1 - solar_fraction)
    else:
        unified['solar'] = 0.0
        unified['wind_onshore'] = 0.0

    # Step 7: I generate DAM commitments from EntsoE3 D-1 price forecasts
    # In production, the EntsoE3 forecaster predicts DAM prices at 12:55 D-1,
    # bids are submitted before 13:00, and accepted positions become mandatory.
    # I use retrospective forecasts (imperfect, MAE ~18 EUR) — NOT actual prices.
    forecast_path = data_dir / "dam_forecasts_2024_2026.csv"
    if forecast_path.exists():
        print("  Using EntsoE3 D-1 forecasts for DAM commitments...")
        unified['dam_commitment'] = _generate_dam_commitments_from_forecasts(
            unified, forecast_path
        )
    else:
        print("  WARNING: EntsoE3 forecasts not found, using heuristic DAM commitments")
        unified['dam_commitment'] = _generate_dam_commitments(unified)

    # Step 8: I add rolling statistics BEFORE synthesizing IntraDay
    # (IntraDay synthesis uses price_std_24h if available)
    unified['price_mean_24h'] = unified['price'].rolling(24, min_periods=1).mean()
    unified['price_std_24h'] = unified['price'].rolling(24, min_periods=1).std().fillna(10)
    unified['price_min_24h'] = unified['price'].rolling(24, min_periods=1).min()
    unified['price_max_24h'] = unified['price'].rolling(24, min_periods=1).max()

    # Step 9: I synthesize IntraDay data (no real XBID data available)
    # I use the existing synthesis function calibrated to real DAM prices
    all_mask = pd.Series(True, index=unified.index)
    unified = _synthesize_intraday_data(unified, all_mask)

    # Step 10: I add all derived features (lags, rolling stats, time features)
    unified = _add_derived_features(unified)

    # Step 11: I validate the final dataset
    _validate_dataset(unified)

    # I print comparison statistics for verification
    _print_admie_vs_synthetic_stats(unified)

    # I save the output
    output_filepath = data_dir / output_path
    unified.to_csv(output_filepath)
    print(f"\nSaved unified ADMIE-based dataset to {output_filepath}")
    print(f"Dataset shape: {unified.shape}")

    # I save a sample for inspection
    sample_path = data_dir / "unified_multimarket_sample.csv"
    unified.head(100).to_csv(sample_path)
    print(f"Saved sample (100 rows) to {sample_path}")

    return unified


def create_unified_dataset_from_admie_15min(
    dam_path: str = 'dam_prices_2021_2026.csv',
    admie_path: str = 'admie_market_data_combined_15min.csv',
    forecast_path_name: str = 'dam_forecasts_2024_2026.csv',
    output_path: str = 'unified_multimarket_training_15min.csv'
) -> pd.DataFrame:
    """
    I create a native 15-min unified training dataset from real ADMIE data.

    Unlike the Phase 1 approach (forward-fill from hourly), this uses native
    15-min resolution data from ADMIE for mFRR, aFRR, imbalance, and system
    deviation. Only DAM prices and load/RES (which are truly hourly) get
    forward-filled to 15-min.

    This means the agent sees real intra-hour price variation in balancing
    markets, enabling it to learn 15-min trading strategies.
    """
    data_dir = Path(__file__).parent
    steps_per_hour = 4  # 15-min resolution

    print("=" * 60)
    print("CREATING NATIVE 15-MIN UNIFIED DATASET FROM ADMIE DATA")
    print("=" * 60)

    # Step 1: I load DAM prices (hourly by market design) and upsample to 15-min
    dam_filepath = data_dir / dam_path
    print(f"\nLoading DAM prices from {dam_filepath}...")
    dam_df = pd.read_csv(dam_filepath, index_col=0)

    # I parse timestamps as UTC, convert to Athens local, strip timezone
    dam_df.index = pd.to_datetime(dam_df.index, utc=True)
    dam_df.index = dam_df.index.tz_convert('Europe/Athens')
    dam_df.index = dam_df.index.tz_localize(None)

    dam_df = dam_df.resample('h').mean().dropna()
    dam_df.columns = ['price']

    # I upsample DAM prices to 15-min via forward-fill (DAM is hourly by design)
    dam_df = dam_df.resample('15min').ffill()
    print(f"  DAM: {len(dam_df)} rows at 15-min (forward-filled from hourly)")
    print(f"  DAM price: mean={dam_df['price'].mean():.2f}, std={dam_df['price'].std():.2f}")

    # Step 2: I load native 15-min ADMIE balancing data
    admie_filepath = data_dir / admie_path
    if not admie_filepath.exists():
        # I fall back to the hourly file if 15-min doesn't exist yet
        admie_filepath_fallback = data_dir / 'admie_market_data_combined.csv'
        if admie_filepath_fallback.exists():
            print(f"\n  WARNING: {admie_path} not found, falling back to hourly ADMIE data")
            print("  Run: python data/admie_data_fetcher.py --start 2024-01-01 --end 2026-01-31 --resolution 15min")
            admie_filepath = admie_filepath_fallback
        else:
            raise FileNotFoundError(
                f"Neither {admie_path} nor admie_market_data_combined.csv found in {data_dir}. "
                "Run the ADMIE fetcher first."
            )

    print(f"\nLoading ADMIE data from {admie_filepath}...")
    admie_df = pd.read_csv(admie_filepath, parse_dates=['timestamp'])
    admie_df = admie_df.set_index('timestamp')

    if admie_df.index.tz is not None:
        admie_df.index = admie_df.index.tz_localize(None)

    # I check if this is truly 15-min data by examining the index frequency
    if len(admie_df) > 2:
        median_diff = admie_df.index.to_series().diff().median()
        is_native_15min = median_diff <= pd.Timedelta(minutes=20)
        print(f"  ADMIE: {len(admie_df)} rows, median step={median_diff}")
        if is_native_15min:
            print("  -> Native 15-min resolution confirmed")
        else:
            print("  -> WARNING: Data appears hourly, will forward-fill to 15-min")
            admie_df = admie_df.resample('15min').ffill()
            print(f"  -> After ffill: {len(admie_df)} rows")

    print(f"  Date range: {admie_df.index.min()} to {admie_df.index.max()}")

    # Step 3: I inner join on timestamp
    print("\nJoining DAM (15-min ffill) + ADMIE (native 15-min)...")
    unified = dam_df.join(admie_df, how='inner')
    print(f"  Overlap: {len(unified)} rows, {unified.index.min()} to {unified.index.max()}")

    if len(unified) == 0:
        raise ValueError("No overlapping timestamps between DAM and ADMIE data!")

    # Step 4: I map ADMIE columns -> environment column names
    # I rename mFRR energy activation prices (the high-value spreads the agent trades)
    column_mapping = {
        'mfrr_energy_price_up': 'mfrr_price_up',
        'mfrr_energy_price_down': 'mfrr_price_down',
        'afrr_price_up': 'afrr_cap_up_price',
        'afrr_price_down': 'afrr_cap_down_price',
        'afrr_requirements_up': 'afrr_cap_up_qty',
        'afrr_requirements_down': 'afrr_cap_down_qty',
        'system_load_mw': 'load_mw',
        'res_production_mw': 'res_total_mw',
        'system_deviation_mwh': 'net_imbalance_mw',
    }

    # I drop ADMIE's mfrr_price_up/down (capacity) before renaming to avoid collision
    admie_cap_cols = ['mfrr_price_up', 'mfrr_price_down']
    for col in admie_cap_cols:
        if col in unified.columns:
            unified = unified.rename(columns={col: f'mfrr_cap_{col.split("_", 1)[1]}'})

    unified = unified.rename(columns=column_mapping)

    # Step 5: I derive aFRR energy prices from imbalance_price
    if 'imbalance_price' in unified.columns:
        unified['afrr_up'] = unified['imbalance_price']
        unified['afrr_down'] = unified['imbalance_price'] * 0.3
        unified['afrr_down'] = unified['afrr_down'].clip(lower=0)
    else:
        unified['afrr_up'] = 80.0
        unified['afrr_down'] = 24.0

    # I compute mFRR spread from real energy prices
    unified['mfrr_spread'] = unified['mfrr_price_up'] - unified['mfrr_price_down']

    # Step 6: I estimate solar/wind split from total RES + time-of-day
    if 'res_total_mw' in unified.columns:
        hours = unified.index.hour.values
        solar_fraction = np.maximum(0, np.sin(np.pi * (hours - 5) / 13))
        solar_fraction = solar_fraction * 0.6
        solar_fraction[hours < 6] = 0.0
        solar_fraction[hours > 19] = 0.0
        unified['solar'] = unified['res_total_mw'] * solar_fraction
        unified['wind_onshore'] = unified['res_total_mw'] * (1 - solar_fraction)
    else:
        unified['solar'] = 0.0
        unified['wind_onshore'] = 0.0

    # Step 7: I generate DAM commitments from forecasts, then forward-fill to 15-min
    # DAM commitments are per-hour (same MW value for all 4 quarter-hours)
    forecast_path = data_dir / forecast_path_name
    if forecast_path.exists():
        print("  Using EntsoE3 D-1 forecasts for DAM commitments...")
        # I need hourly index for the commitment generator, then ffill to 15-min
        hourly_index = unified.index[unified.index.minute == 0]
        hourly_proxy = unified.loc[hourly_index].copy()
        commitments_hourly = _generate_dam_commitments_from_forecasts(
            hourly_proxy, forecast_path
        )
        # I create an hourly series and reindex to 15-min with ffill
        commit_series = pd.Series(
            commitments_hourly, index=hourly_index, name='dam_commitment'
        )
        unified['dam_commitment'] = commit_series.reindex(unified.index).ffill().bfill()
    else:
        print("  WARNING: EntsoE3 forecasts not found, using heuristic DAM commitments")
        hourly_index = unified.index[unified.index.minute == 0]
        hourly_proxy = unified.loc[hourly_index].copy()
        commitments_hourly = _generate_dam_commitments(hourly_proxy)
        commit_series = pd.Series(
            commitments_hourly, index=hourly_index, name='dam_commitment'
        )
        unified['dam_commitment'] = commit_series.reindex(unified.index).ffill().bfill()

    # Step 8: I add rolling statistics BEFORE synthesizing IntraDay
    window_24h = 24 * steps_per_hour  # 96 steps at 15-min
    unified['price_mean_24h'] = unified['price'].rolling(window_24h, min_periods=1).mean()
    unified['price_std_24h'] = unified['price'].rolling(window_24h, min_periods=1).std().fillna(10)
    unified['price_min_24h'] = unified['price'].rolling(window_24h, min_periods=1).min()
    unified['price_max_24h'] = unified['price'].rolling(window_24h, min_periods=1).max()

    # Step 9: I synthesize IntraDay data at 15-min resolution
    # OU processes use sqrt(0.25) scaling -> finer intra-hour variation
    all_mask = pd.Series(True, index=unified.index)
    unified = _synthesize_intraday_data(unified, all_mask, time_step_hours=0.25)

    # Step 10: I add all derived features with 15-min-aware window sizes
    unified = _add_derived_features(unified, steps_per_hour=steps_per_hour)

    # Step 11: I validate the final dataset
    _validate_dataset(unified)
    _print_admie_vs_synthetic_stats(unified)

    # I verify native 15-min variability
    _verify_15min_variability(unified)

    # I save the output
    output_filepath = data_dir / output_path
    unified.to_csv(output_filepath)
    print(f"\nSaved native 15-min unified dataset to {output_filepath}")
    print(f"Dataset shape: {unified.shape}")

    sample_path = data_dir / "unified_multimarket_sample.csv"
    unified.head(100).to_csv(sample_path)
    print(f"Saved sample (100 rows) to {sample_path}")

    return unified


def _verify_15min_variability(df: pd.DataFrame) -> None:
    """
    I verify that native 15-min columns actually vary within hours,
    as opposed to being flat (forward-filled from hourly).
    """
    print("\n" + "=" * 60)
    print("15-MIN VARIABILITY CHECK")
    print("=" * 60)

    # I check columns that should have native 15-min variation
    native_15min_cols = ['mfrr_price_up', 'mfrr_price_down', 'imbalance_price',
                         'net_imbalance_mw', 'afrr_cap_up_price', 'afrr_cap_down_price']
    # I check columns that should be flat within hour (forward-filled)
    hourly_cols = ['price']

    # I compute within-hour standard deviation
    df_check = df.copy()
    df_check['hour_group'] = df_check.index.floor('h')

    for col in native_15min_cols:
        if col not in df_check.columns:
            continue
        within_hour_std = df_check.groupby('hour_group')[col].std().mean()
        overall_std = df_check[col].std()
        ratio = within_hour_std / overall_std if overall_std > 0 else 0
        is_native = within_hour_std > 0.001
        status = "NATIVE 15-min" if is_native else "FLAT (hourly ffill)"
        print(f"  {col:30s}: within-hour std={within_hour_std:8.3f}, "
              f"overall std={overall_std:8.2f}, ratio={ratio:.3f} -> {status}")

    for col in hourly_cols:
        if col not in df_check.columns:
            continue
        within_hour_std = df_check.groupby('hour_group')[col].std().mean()
        is_flat = within_hour_std < 0.001
        status = "FLAT (correct)" if is_flat else "UNEXPECTED variation"
        print(f"  {col:30s}: within-hour std={within_hour_std:8.3f} -> {status}")


def _print_admie_vs_synthetic_stats(df: pd.DataFrame) -> None:
    """I print key statistics to verify real data quality vs old synthetic levels."""
    print("\n" + "=" * 60)
    print("REAL ADMIE DATA STATISTICS (compare vs old synthetic)")
    print("=" * 60)

    stats = {
        'DAM price': ('price', None),
        'mFRR up (energy)': ('mfrr_price_up', 'was ~145 synthetic'),
        'mFRR down (energy)': ('mfrr_price_down', 'was ~65 synthetic'),
        'mFRR spread': ('mfrr_spread', 'was ~80 synthetic'),
        'aFRR cap up': ('afrr_cap_up_price', 'was ~22 synthetic'),
        'aFRR cap down': ('afrr_cap_down_price', 'was ~15 synthetic'),
        'aFRR up (energy)': ('afrr_up', 'was ~105 synthetic'),
        'System load': ('load_mw', 'was ~5000 synthetic'),
        'RES total': ('res_total_mw', 'was missing'),
        'Net imbalance': ('net_imbalance_mw', 'was simulated'),
    }

    for label, (col, note) in stats.items():
        if col in df.columns:
            note_str = f"  ({note})" if note else ""
            print(f"  {label}: mean={df[col].mean():.1f}, "
                  f"std={df[col].std():.1f}, "
                  f"min={df[col].min():.1f}, "
                  f"max={df[col].max():.1f}{note_str}")


def create_unified_dataset(dam_df: pd.DataFrame, afrr_df: pd.DataFrame) -> pd.DataFrame:
    """
    I merge DAM and aFRR/mFRR datasets into a unified training dataset.

    Strategy:
    1. Find overlapping date range
    2. For DAM data: use directly (older data has DAM prices/commitments)
    3. For aFRR/mFRR: merge where available, synthesize for older periods
    """
    # I find the common time range
    dam_start, dam_end = dam_df.index.min(), dam_df.index.max()
    afrr_start, afrr_end = afrr_df.index.min(), afrr_df.index.max()

    print(f"\nDate ranges:")
    print(f"  DAM data: {dam_start} to {dam_end}")
    print(f"  aFRR/mFRR data: {afrr_start} to {afrr_end}")

    # I start with DAM data as the base
    unified = dam_df.copy()

    # I add aFRR/mFRR columns, filling with synthetic data where not available
    afrr_columns = [
        'afrr_up', 'afrr_down',
        'mfrr_price_up', 'mfrr_price_down', 'mfrr_spread',
        'afrr_cap_up_price', 'afrr_cap_up_qty',
        'afrr_cap_down_price', 'afrr_cap_down_qty',
        'net_imbalance_mw', 'load_mw', 'res_total_mw'
    ]

    # I initialize columns with NaN
    for col in afrr_columns:
        unified[col] = np.nan

    # I merge actual aFRR/mFRR data where available
    overlap_mask = unified.index.isin(afrr_df.index)
    print(f"  Overlap: {overlap_mask.sum()} rows have real aFRR/mFRR data")

    for col in afrr_columns:
        if col in afrr_df.columns:
            unified.loc[overlap_mask, col] = afrr_df.loc[unified.index[overlap_mask], col].values

    # I synthesize missing aFRR/mFRR data based on DAM prices
    # This creates realistic training data for periods without actual balancing data
    missing_mask = ~overlap_mask
    print(f"  Synthesizing aFRR/mFRR for {missing_mask.sum()} rows...")

    if missing_mask.sum() > 0:
        unified = _synthesize_balancing_data(unified, missing_mask)

    # I synthesize IntraDay data for ALL rows (no real IntraDay data available)
    all_mask = pd.Series(True, index=unified.index)
    unified = _synthesize_intraday_data(unified, all_mask)

    # I add derived features for training
    unified = _add_derived_features(unified)

    # I validate the dataset
    _validate_dataset(unified)

    return unified


def _synthesize_balancing_data(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """
    I synthesize realistic aFRR/mFRR data with independent dynamics.

    Key design: mFRR/aFRR prices are partially correlated with DAM (~0.5-0.6)
    but have significant independent variation from:
    - Ornstein-Uhlenbeck mean-reverting spread process
    - Regime shifts (scarcity events, low-demand periods)
    - RES forecast error driving imbalance prices
    - Temporal autocorrelation (prices don't jump randomly hour-to-hour)
    """
    dam_prices = df.loc[mask, 'price'].values
    hours = df.loc[mask].index.hour.values if hasattr(df.index, 'hour') else np.zeros(mask.sum())
    n = len(dam_prices)

    # I generate an independent Ornstein-Uhlenbeck spread process
    # This ensures mFRR spread has temporal autocorrelation but is NOT
    # deterministically derived from DAM
    ou_theta = 0.15  # Mean reversion speed
    ou_mu_up = 45.0  # Long-run mean spread (upward)
    ou_mu_down = 30.0  # Long-run mean spread (downward)
    ou_sigma = 15.0  # Volatility of spread

    spread_up = np.zeros(n)
    spread_down = np.zeros(n)
    spread_up[0] = ou_mu_up + np.random.normal(0, ou_sigma)
    spread_down[0] = ou_mu_down + np.random.normal(0, ou_sigma)

    for i in range(1, n):
        # I add mean-reverting dynamics with regime shifts
        spread_up[i] = spread_up[i-1] + ou_theta * (ou_mu_up - spread_up[i-1]) + ou_sigma * np.random.normal() * np.sqrt(1.0)
        spread_down[i] = spread_down[i-1] + ou_theta * (ou_mu_down - spread_down[i-1]) + ou_sigma * np.random.normal() * np.sqrt(1.0)

    # I add regime shifts — scarcity events where spreads spike
    # These happen ~5% of hours and are unpredictable from DAM alone
    scarcity_events = np.random.random(n) < 0.05
    spread_up[scarcity_events] *= np.random.uniform(2.0, 4.0, scarcity_events.sum())

    # I add time-of-day modulation (smaller effect than before)
    tod_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hours / 24)

    # I clip spreads to realistic range
    spread_up = np.clip(spread_up * tod_factor, 5, 200)
    spread_down = np.clip(spread_down * tod_factor, 3, 150)

    # I compute mFRR prices: DAM correlation ~0.5 + independent spread
    # I add an independent price component that doesn't track DAM
    independent_level = np.cumsum(np.random.normal(0, 2.0, n))  # Random walk
    independent_level -= np.convolve(independent_level, np.ones(168)/168, mode='same')  # I detrend weekly
    independent_component = np.clip(independent_level, -40, 40)

    df.loc[mask, 'mfrr_price_up'] = dam_prices + spread_up + independent_component
    df.loc[mask, 'mfrr_price_down'] = np.clip(dam_prices - spread_down + independent_component * 0.5, 0, None)
    df.loc[mask, 'mfrr_spread'] = df.loc[mask, 'mfrr_price_up'] - df.loc[mask, 'mfrr_price_down']

    # aFRR prices: partial DAM correlation + independent dynamics
    afrr_independent = np.cumsum(np.random.normal(0, 1.5, n))
    afrr_independent -= np.convolve(afrr_independent, np.ones(168)/168, mode='same')
    afrr_independent = np.clip(afrr_independent, -20, 20)

    df.loc[mask, 'afrr_up'] = dam_prices * (1.0 + np.random.uniform(0.02, 0.15, n)) + afrr_independent
    df.loc[mask, 'afrr_down'] = np.clip(dam_prices * (1.0 - np.random.uniform(0.05, 0.20, n)) + afrr_independent * 0.5, 0, None)

    # aFRR capacity prices: independent OU process (not DAM-derived)
    cap_ou = np.zeros(n)
    cap_mu = 22.0
    cap_ou[0] = cap_mu
    for i in range(1, n):
        cap_ou[i] = cap_ou[i-1] + 0.1 * (cap_mu - cap_ou[i-1]) + 5.0 * np.random.normal()

    peak_hours = (hours >= 17) & (hours <= 21)
    cap_peak_boost = np.where(peak_hours, 10.0, 0.0)
    df.loc[mask, 'afrr_cap_up_price'] = np.clip(cap_ou + cap_peak_boost + np.random.normal(0, 3, n), 2, 80)
    df.loc[mask, 'afrr_cap_down_price'] = np.clip(cap_ou * 0.7 + np.random.normal(0, 3, n), 1, 60)

    # Capacity quantities: typical reserve needs
    df.loc[mask, 'afrr_cap_up_qty'] = np.random.uniform(200, 800, n)
    df.loc[mask, 'afrr_cap_down_qty'] = np.random.uniform(100, 300, n)

    # System state: net imbalance correlates with RES variability
    solar = df.loc[mask, 'solar'].values if 'solar' in df.columns else np.zeros(n)
    wind = df.loc[mask, 'wind_onshore'].values if 'wind_onshore' in df.columns else np.zeros(n)
    res_total = solar + wind

    # I create imbalance based on RES forecast error simulation
    forecast_error = np.random.normal(0, 0.1, n) * (res_total + 100)
    df.loc[mask, 'net_imbalance_mw'] = forecast_error
    df.loc[mask, 'res_total_mw'] = res_total
    df.loc[mask, 'load_mw'] = 5000 + 2000 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 200, n)

    return df


def _generate_dam_commitments_from_forecasts(
    df: pd.DataFrame,
    forecast_path: Path
) -> np.ndarray:
    """
    I generate DAM commitments using D-1 price forecasts from EntsoE3.

    This is realistic because:
    1. Forecasts have imperfect accuracy (MAE ~18 EUR/MWh) — not oracle
    2. The operator bids based on forecasted prices, not actual prices
    3. Commitments are sized based on forecasted price spreads within each day

    Strategy: I charge during forecasted cheap hours and discharge during
    forecasted expensive hours, with size proportional to the daily spread.
    """
    max_power_mw = 30.0
    commitments = np.zeros(len(df))

    # I load the D-1 forecasts
    forecasts_df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
    print(f"    Loaded {len(forecasts_df)} hourly forecasts")

    # I match forecasts to training data timestamps
    dates = df.index.normalize().unique()

    n_active_days = 0
    for date in dates:
        day_mask = df.index.normalize() == date
        day_idx = df.index[day_mask]
        n_hours = len(day_idx)

        if n_hours < 12:
            continue

        # I get forecasted prices for this day
        day_forecasts = forecasts_df.loc[
            forecasts_df.index.normalize() == date, 'forecast_price'
        ]

        if len(day_forecasts) < 12:
            continue

        # I align forecast hours with data hours
        forecast_prices = np.full(n_hours, np.nan)
        for i, ts in enumerate(day_idx):
            if ts in day_forecasts.index:
                forecast_prices[i] = day_forecasts.loc[ts]

        # I skip days where forecasts are mostly missing
        valid = ~np.isnan(forecast_prices)
        if valid.sum() < 12:
            continue

        # I fill any missing hours with interpolation
        forecast_prices = pd.Series(forecast_prices).interpolate().fillna(method='bfill').fillna(method='ffill').values

        # I skip ~10% of days randomly (maintenance, operator caution)
        if np.random.random() < 0.10:
            continue

        # I compute the daily forecasted price spread
        daily_spread = np.max(forecast_prices) - np.min(forecast_prices)

        # I skip days with very flat forecasted prices (spread < 15 EUR)
        # — not worth the cycling cost
        if daily_spread < 15.0:
            continue

        # I rank hours by forecasted price
        price_rank = np.argsort(np.argsort(forecast_prices))

        # I select charge hours (cheapest 4-5) and discharge hours (most expensive 4-5)
        n_trade = min(5, n_hours // 4)
        charge_threshold = n_trade
        discharge_threshold = n_hours - n_trade

        # I size commitments based on daily spread (higher spread → more aggressive)
        # Spread 15-30 EUR: 40-60% of max power
        # Spread 30-60 EUR: 60-80%
        # Spread > 60 EUR: 80-100%
        spread_factor = np.clip(daily_spread / 80.0, 0.4, 1.0)
        base_power = max_power_mw * spread_factor

        day_commitments = np.zeros(n_hours)

        for i in range(n_hours):
            if price_rank[i] < charge_threshold:
                day_commitments[i] = -base_power * np.random.uniform(0.6, 1.0)
            elif price_rank[i] >= discharge_threshold:
                day_commitments[i] = base_power * np.random.uniform(0.6, 1.0)

        # I ensure energy balance feasibility (charge ≥ discharge / efficiency)
        total_charge = abs(day_commitments[day_commitments < 0].sum())
        total_discharge = day_commitments[day_commitments > 0].sum()

        if total_charge < total_discharge * 1.1:
            scale = total_charge / (total_discharge * 1.1 + 1e-6)
            day_commitments[day_commitments > 0] *= scale

        commitments[day_mask] = day_commitments
        n_active_days += 1

    commitments = np.clip(commitments, -max_power_mw, max_power_mw)

    n_active = np.sum(np.abs(commitments) > 0.1)
    n_charge = np.sum(commitments < -0.1)
    n_discharge = np.sum(commitments > 0.1)
    print(f"    DAM commitments (forecast-based): {n_active} active hours "
          f"({n_charge} charge, {n_discharge} discharge) across {n_active_days} days, "
          f"{len(commitments) - n_active} idle")

    return commitments


def _generate_dam_commitments(df: pd.DataFrame) -> np.ndarray:
    """
    I generate realistic DAM commitments based on daily price profiles.

    In the real market, a battery operator would:
    1. Forecast next-day DAM prices (submitted by 12:00 D-1)
    2. Bid to charge during the cheapest hours (solar midday, night)
    3. Bid to discharge during the most expensive hours (evening peak)
    4. Not every day has commitments (sometimes prices are flat, or operator is cautious)

    I use actual DAM prices as a proxy for "perfect forecast" — this represents
    what a well-informed operator would have committed. Randomization adds
    training diversity and prevents the agent from overfitting to a fixed pattern.
    """
    max_power_mw = 30.0  # Battery inverter rating
    commitments = np.zeros(len(df))

    # I group by date and generate commitments per day
    dates = df.index.normalize().unique()

    for date in dates:
        day_mask = df.index.normalize() == date
        day_prices = df.loc[day_mask, 'price'].values
        day_hours = df.loc[day_mask].index.hour.values
        n_hours = len(day_prices)

        if n_hours < 12:
            continue

        # I skip ~15% of days randomly (operator caution, maintenance, etc.)
        if np.random.random() < 0.15:
            continue

        # I rank hours by price within the day
        price_rank = np.argsort(np.argsort(day_prices))  # 0=cheapest, n-1=most expensive

        # I select charge hours (cheapest 4-6 hours) and discharge hours (most expensive 4-6 hours)
        n_charge = np.random.randint(3, 6)
        n_discharge = np.random.randint(3, 6)

        charge_threshold = n_charge
        discharge_threshold = n_hours - n_discharge

        # I determine commitment size with randomization (50-100% of max power)
        charge_power = max_power_mw * np.random.uniform(0.5, 1.0)
        discharge_power = max_power_mw * np.random.uniform(0.5, 1.0)

        day_commitments = np.zeros(n_hours)

        for i in range(n_hours):
            if price_rank[i] < charge_threshold:
                # I commit to charge (buy) — negative = charge
                day_commitments[i] = -charge_power * np.random.uniform(0.6, 1.0)
            elif price_rank[i] >= discharge_threshold:
                # I commit to discharge (sell) — positive = discharge
                day_commitments[i] = discharge_power * np.random.uniform(0.6, 1.0)

        # I ensure net energy balance is roughly feasible within the day
        # (total charge ≥ total discharge / efficiency)
        total_charge = abs(day_commitments[day_commitments < 0].sum())
        total_discharge = day_commitments[day_commitments > 0].sum()

        if total_charge < total_discharge * 1.1:
            # I scale down discharge to maintain feasibility
            scale = total_charge / (total_discharge * 1.1 + 1e-6)
            day_commitments[day_commitments > 0] *= scale

        commitments[day_mask] = day_commitments

    # I clip to inverter limits
    commitments = np.clip(commitments, -max_power_mw, max_power_mw)

    n_active = np.sum(np.abs(commitments) > 0.1)
    n_charge = np.sum(commitments < -0.1)
    n_discharge = np.sum(commitments > 0.1)
    print(f"  DAM commitments: {n_active} active hours "
          f"({n_charge} charge, {n_discharge} discharge), "
          f"{len(commitments) - n_active} idle")

    return commitments


def _synthesize_intraday_data(
    df: pd.DataFrame,
    mask: pd.Series,
    time_step_hours: float = 1.0
) -> pd.DataFrame:
    """
    I synthesize realistic IntraDay (XBID/HEnEx) market data with independent dynamics.

    Key design: IntraDay prices are partially correlated with DAM (~0.6-0.7)
    but have significant independent variation from:
    - Ornstein-Uhlenbeck deviation process (autocorrelated, not white noise)
    - RES forecast updates between DAM and IntraDay gate
    - Liquidity-driven spread dynamics
    - Regime shifts (e.g., unexpected weather changes)

    Args:
        time_step_hours: Time step in hours (1.0 for hourly, 0.25 for 15-min).
            OU processes scale with sqrt(dt) for per-step noise, producing
            finer-grained variation at 15-min while preserving hourly statistics.
    """
    dt = time_step_hours
    dam_prices = df.loc[mask, 'price'].values
    hours = df.loc[mask].index.hour.values if hasattr(df.index, 'hour') else np.zeros(mask.sum())
    n = len(dam_prices)

    # I generate an independent OU deviation process
    # This represents market information arriving AFTER DAM clearing
    # (RES forecast updates, cross-border flow changes, outages)
    # I scale sigma proportionally to DAM price level for realistic deviation
    ou_theta = 0.05  # Slow mean reversion - deviations persist for many hours
    ou_sigma_base = 8.0  # Base EUR volatility per sqrt(hour)
    deviation = np.zeros(n)
    deviation[0] = np.random.normal(0, ou_sigma_base)

    for i in range(1, n):
        # I scale volatility by price level - higher prices have larger deviations
        price_scale = max(dam_prices[i] / 100.0, 0.3)
        sigma_i = ou_sigma_base * price_scale
        # I use sqrt(dt) scaling so the OU process produces smaller per-step
        # jumps at 15-min but 4x more steps per hour -> same hourly variance
        deviation[i] = (
            deviation[i-1]
            + ou_theta * dt * (0 - deviation[i-1])
            + sigma_i * np.random.normal() * np.sqrt(dt)
        )

    # I add regime shifts - sudden forecast changes (~5% of hours, scaled by dt)
    # At 15-min, each step has 1.25% shock probability -> ~5% per hour
    shock_prob = 0.05 * dt
    forecast_shocks = np.random.random(n) < shock_prob
    if forecast_shocks.sum() > 0:
        shock_scale = np.abs(dam_prices[forecast_shocks]) * 0.15
        deviation[forecast_shocks] += np.random.normal(0, 1.0, forecast_shocks.sum()) * shock_scale

    # I add time-of-day effects (smaller than before)
    peak_premium = np.where((hours >= 17) & (hours <= 21), 2.0, 0.0)
    solar_discount = np.where((hours >= 9) & (hours <= 15), -1.5, 0.0)

    # I compute IntraDay mid: DAM + independent_deviation + small_tod_effect
    intraday_mid = dam_prices + deviation + peak_premium + solar_discount

    # I generate spread with independent OU dynamics
    spread_ou = np.zeros(n)
    spread_mu = 3.0  # Mean spread
    spread_ou[0] = spread_mu
    for i in range(1, n):
        spread_ou[i] = (
            spread_ou[i-1]
            + 0.12 * dt * (spread_mu - spread_ou[i-1])
            + 1.0 * abs(np.random.normal()) * np.sqrt(dt)
        )

    # I modulate spread by liquidity hours
    is_liquid = ((hours >= 8) & (hours <= 20)).astype(float)
    liquidity_factor = np.where(is_liquid, 0.7, 1.5)

    # I add volatility-based spread widening
    price_std = df.loc[mask, 'price_std_24h'].values if 'price_std_24h' in df.columns else np.full(n, 15.0)
    volatility_factor = np.clip(price_std / 30.0, 0.5, 2.0)

    spread = spread_ou * liquidity_factor * volatility_factor
    spread = np.clip(spread, 0.5, 15.0)

    # I calculate bid and ask from mid and spread
    df.loc[mask, 'intraday_bid'] = intraday_mid - spread / 2.0
    df.loc[mask, 'intraday_ask'] = intraday_mid + spread / 2.0
    df.loc[mask, 'intraday_spread'] = spread

    # I calculate volume with independent dynamics
    vol_ou = np.zeros(n)
    vol_mu = 0.55
    vol_ou[0] = vol_mu
    for i in range(1, n):
        vol_ou[i] = (
            vol_ou[i-1]
            + 0.1 * dt * (vol_mu - vol_ou[i-1])
            + 0.08 * np.random.normal() * np.sqrt(dt)
        )

    peak_boost = np.where((hours >= 17) & (hours <= 21), 0.15, 0.0)
    liquid_boost = np.where(is_liquid, 0.1, -0.1)
    volume = vol_ou + peak_boost + liquid_boost
    df.loc[mask, 'intraday_volume'] = np.clip(volume, 0.1, 1.0)

    return df


def _add_derived_features(df: pd.DataFrame, steps_per_hour: int = 1) -> pd.DataFrame:
    """
    I add derived features useful for training.

    Args:
        steps_per_hour: Number of time steps per hour (1 for hourly, 4 for 15-min).
            All lag and rolling window sizes are scaled accordingly so that
            e.g. "1h lag" = shift(4) at 15-min resolution.
    """
    sph = steps_per_hour

    # I extract hour and day of week if not present
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['is_solar_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)

    # I add lagged features for backward-looking observations
    # At 15-min: shift(4)=1h lag, shift(16)=4h lag
    for col in ['price', 'afrr_up', 'afrr_down', 'mfrr_price_up', 'mfrr_price_down']:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1 * sph)
            df[f'{col}_lag_4h'] = df[col].shift(4 * sph)

    # I add IntraDay lagged features (avoid lookahead)
    for col in ['intraday_bid', 'intraday_ask', 'intraday_spread', 'intraday_volume']:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1 * sph)

    # I add rolling statistics (backward-looking only!)
    # At 15-min: 24h window = 96 steps, at hourly: 24 steps
    window_24h = 24 * sph
    df['price_mean_24h'] = df['price'].rolling(window_24h, min_periods=1).mean()
    df['price_std_24h'] = df['price'].rolling(window_24h, min_periods=1).std().fillna(10)
    df['price_min_24h'] = df['price'].rolling(window_24h, min_periods=1).min()
    df['price_max_24h'] = df['price'].rolling(window_24h, min_periods=1).max()

    # mFRR spread statistics
    if 'mfrr_spread' in df.columns:
        df['mfrr_spread_mean_24h'] = df['mfrr_spread'].rolling(window_24h, min_periods=1).mean()

    # aFRR capacity price percentile (for bid optimization)
    # Weekly window = 168h, minimum = 24h
    if 'afrr_cap_up_price' in df.columns:
        df['afrr_cap_percentile'] = df['afrr_cap_up_price'].rolling(
            168 * sph, min_periods=24 * sph
        ).rank(pct=True)

    # I forward-fill then back-fill any remaining NaN values
    df = df.ffill().bfill()

    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    """I validate the unified dataset."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # I check for required columns
    required_columns = [
        'price', 'dam_commitment',  # DAM
        'intraday_bid', 'intraday_ask', 'intraday_spread',  # IntraDay
        'afrr_up', 'afrr_down',  # aFRR energy
        'afrr_cap_up_price', 'afrr_cap_down_price',  # aFRR capacity
        'mfrr_price_up', 'mfrr_price_down', 'mfrr_spread',  # mFRR
        'hour', 'is_peak_hour', 'is_solar_hour'  # Time features
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
    else:
        print("All required columns present")

    # I check for NaN values
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.sum() > 0:
        print(f"\nWARNING: NaN values found:")
        print(nan_counts[nan_counts > 0])
    else:
        print("No NaN values in required columns")

    # I print summary statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total hours: {len(df)}")
    print(f"Total days: {len(df) / 24:.0f}")

    print("\nPrice statistics:")
    print(f"  DAM price: mean={df['price'].mean():.2f}, std={df['price'].std():.2f}")
    print(f"  aFRR up: mean={df['afrr_up'].mean():.2f}")
    print(f"  aFRR down: mean={df['afrr_down'].mean():.2f}")
    print(f"  mFRR spread: mean={df['mfrr_spread'].mean():.2f}")

    if 'intraday_bid' in df.columns:
        print(f"  IntraDay bid: mean={df['intraday_bid'].mean():.2f}")
        print(f"  IntraDay ask: mean={df['intraday_ask'].mean():.2f}")
        print(f"  IntraDay spread: mean={df['intraday_spread'].mean():.2f}")

    if 'afrr_cap_up_price' in df.columns:
        print(f"  aFRR cap price: mean={df['afrr_cap_up_price'].mean():.2f}")


def main():
    """
    I create and save the unified training dataset.

    I default to ADMIE-based creation (real market data) when the ADMIE CSV
    exists, falling back to synthetic generation otherwise.

    Supports --resolution 1h (default) or 15min for native 15-min pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Create unified multi-market training dataset'
    )
    parser.add_argument(
        '--resolution', type=str, default='1h',
        choices=['1h', '15min'],
        help='Time resolution: 1h (default) or 15min (native ADMIE)'
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent
    dam_path = data_dir / "dam_prices_2021_2026.csv"

    if args.resolution == '15min':
        # I use the native 15-min pipeline
        print("Resolution: 15-min (native ADMIE)")
        unified_df = create_unified_dataset_from_admie_15min()
    else:
        # I use the standard hourly pipeline
        admie_path = data_dir / "admie_market_data_combined.csv"
        if admie_path.exists() and dam_path.exists():
            print("ADMIE data found - using real market data")
            unified_df = create_unified_dataset_from_admie()
        else:
            print("ADMIE data not found - falling back to synthetic generation")
            print("=" * 60)
            print("CREATING UNIFIED MULTI-MARKET TRAINING DATASET (SYNTHETIC)")
            print("=" * 60)

            dam_df = load_dam_data()
            afrr_df = load_afrr_mfrr_data()
            unified_df = create_unified_dataset(dam_df, afrr_df)

            output_path = data_dir / "unified_multimarket_training.csv"
            unified_df.to_csv(output_path)
            print(f"\nSaved unified dataset to {output_path}")
            print(f"Dataset shape: {unified_df.shape}")

            sample_path = data_dir / "unified_multimarket_sample.csv"
            unified_df.head(100).to_csv(sample_path)
            print(f"Saved sample (100 rows) to {sample_path}")

    return unified_df


if __name__ == "__main__":
    unified_df = main()
