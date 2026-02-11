"""
Create Unified Multi-Market Training Dataset

I merge DAM data (feasible_data_with_dam.csv) with aFRR/mFRR data (afrr_mfrr_combined_v2.csv)
to create a single unified training dataset for the multi-market environment.

The unified dataset contains all features needed for:
- DAM commitment tracking
- IntraDay price trading
- aFRR capacity bidding and activation
- mFRR energy trading

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

    # I add derived features for training
    unified = _add_derived_features(unified)

    # I validate the dataset
    _validate_dataset(unified)

    return unified


def _synthesize_balancing_data(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """
    I synthesize realistic aFRR/mFRR data based on DAM prices.

    This is based on observed correlations:
    - aFRR prices correlate with DAM prices (~0.7-0.8)
    - mFRR spreads are typically 30-80 EUR
    - Capacity prices are 10-50 EUR/MW/h
    - Activation correlates with price volatility
    """
    dam_prices = df.loc[mask, 'price'].values
    hours = df.loc[mask].index.hour.values if hasattr(df.index, 'hour') else np.zeros(mask.sum())

    # I add noise proportional to price level
    noise_factor = 0.15

    # aFRR prices: typically slightly above DAM
    df.loc[mask, 'afrr_up'] = dam_prices * (1.0 + np.random.uniform(0.02, 0.15, len(dam_prices)))
    df.loc[mask, 'afrr_down'] = dam_prices * (1.0 - np.random.uniform(0.05, 0.20, len(dam_prices)))

    # mFRR prices: wider spreads than DAM
    base_spread = 40 + 30 * np.sin(2 * np.pi * hours / 24)  # I vary by time of day
    df.loc[mask, 'mfrr_price_up'] = dam_prices + base_spread * np.random.uniform(0.8, 1.5, len(dam_prices))
    df.loc[mask, 'mfrr_price_down'] = dam_prices - base_spread * np.random.uniform(0.5, 1.0, len(dam_prices))
    df.loc[mask, 'mfrr_price_down'] = df.loc[mask, 'mfrr_price_down'].clip(lower=0)  # No negative prices
    df.loc[mask, 'mfrr_spread'] = df.loc[mask, 'mfrr_price_up'] - df.loc[mask, 'mfrr_price_down']

    # aFRR capacity prices: typically 10-50 EUR/MW/h
    peak_hours = (hours >= 17) & (hours <= 21)
    base_cap_price = np.where(peak_hours, 30, 15)
    df.loc[mask, 'afrr_cap_up_price'] = base_cap_price * np.random.uniform(0.5, 2.0, len(dam_prices))
    df.loc[mask, 'afrr_cap_down_price'] = base_cap_price * np.random.uniform(0.3, 1.5, len(dam_prices))

    # Capacity quantities: typical reserve needs
    df.loc[mask, 'afrr_cap_up_qty'] = np.random.uniform(200, 800, len(dam_prices))
    df.loc[mask, 'afrr_cap_down_qty'] = np.random.uniform(100, 300, len(dam_prices))

    # System state: net imbalance correlates with RES variability
    solar = df.loc[mask, 'solar'].values if 'solar' in df.columns else np.zeros(len(dam_prices))
    wind = df.loc[mask, 'wind_onshore'].values if 'wind_onshore' in df.columns else np.zeros(len(dam_prices))
    res_total = solar + wind

    # I create imbalance based on RES forecast error simulation
    forecast_error = np.random.normal(0, 0.1, len(dam_prices)) * (res_total + 100)
    df.loc[mask, 'net_imbalance_mw'] = forecast_error
    df.loc[mask, 'res_total_mw'] = res_total
    df.loc[mask, 'load_mw'] = 5000 + 2000 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 200, len(dam_prices))

    return df


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """I add derived features useful for training."""

    # I extract hour and day of week if not present
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['is_solar_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)

    # I add lagged features for backward-looking observations
    for col in ['price', 'afrr_up', 'afrr_down', 'mfrr_price_up', 'mfrr_price_down']:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_4h'] = df[col].shift(4)

    # I add rolling statistics (backward-looking only!)
    df['price_mean_24h'] = df['price'].rolling(24, min_periods=1).mean()
    df['price_std_24h'] = df['price'].rolling(24, min_periods=1).std().fillna(10)
    df['price_min_24h'] = df['price'].rolling(24, min_periods=1).min()
    df['price_max_24h'] = df['price'].rolling(24, min_periods=1).max()

    # mFRR spread statistics
    if 'mfrr_spread' in df.columns:
        df['mfrr_spread_mean_24h'] = df['mfrr_spread'].rolling(24, min_periods=1).mean()

    # aFRR capacity price percentile (for bid optimization)
    if 'afrr_cap_up_price' in df.columns:
        df['afrr_cap_percentile'] = df['afrr_cap_up_price'].rolling(168, min_periods=24).rank(pct=True)  # Weekly window

    # I forward-fill then back-fill any remaining NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df


def _validate_dataset(df: pd.DataFrame) -> None:
    """I validate the unified dataset."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # I check for required columns
    required_columns = [
        'price', 'dam_commitment',  # DAM
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

    if 'afrr_cap_up_price' in df.columns:
        print(f"  aFRR cap price: mean={df['afrr_cap_up_price'].mean():.2f}")


def main():
    """I create and save the unified training dataset."""
    print("=" * 60)
    print("CREATING UNIFIED MULTI-MARKET TRAINING DATASET")
    print("=" * 60)

    # I load both datasets
    dam_df = load_dam_data()
    afrr_df = load_afrr_mfrr_data()

    # I create the unified dataset
    unified_df = create_unified_dataset(dam_df, afrr_df)

    # I save the unified dataset
    output_path = Path(__file__).parent / "unified_multimarket_training.csv"
    unified_df.to_csv(output_path)
    print(f"\nSaved unified dataset to {output_path}")
    print(f"Dataset shape: {unified_df.shape}")

    # I also save a sample for inspection
    sample_path = Path(__file__).parent / "unified_multimarket_sample.csv"
    unified_df.head(100).to_csv(sample_path)
    print(f"Saved sample (100 rows) to {sample_path}")

    return unified_df


if __name__ == "__main__":
    unified_df = main()
