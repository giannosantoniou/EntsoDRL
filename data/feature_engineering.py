"""
Feature Engineering for mFRR Bidding DRL Agent

I create a comprehensive feature set based on exploratory data analysis:

Key Findings from EDA:
1. mFRR Up prices: highly variable, negative during solar hours, peak in evening
2. Greece is ALWAYS system short (99.7%) - structural upward regulation need
3. Spread (Up - Down): positive 81% of time, +42 EUR/MWh mean
4. Weekend effect: 42% lower prices than weekdays
5. Seasonal: Winter/summer highest, spring lowest (solar effect)
6. Cross-border: Net importer from IT, balanced with BG

Feature Categories:
- Time features (cyclical encoding)
- Price features (current, lagged, rolling stats)
- System state features (load, RES, imbalance)
- Cross-border features (flows, net position)
- Derived/interaction features
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clip_extreme_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    I clip extreme placeholder values (-99999, 99999) that indicate
    no activation or missing data. Based on EDA, reasonable ranges are:
    - mFRR Up: -500 to 1000 EUR/MWh
    - mFRR Down: -100 to 500 EUR/MWh
    """
    df = df.copy()

    if 'mfrr_price_up' in df.columns:
        df['mfrr_price_up'] = df['mfrr_price_up'].clip(-500, 1000)

    if 'mfrr_price_down' in df.columns:
        df['mfrr_price_down'] = df['mfrr_price_down'].clip(-100, 500)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add cyclical time encodings and binary time indicators.
    Cyclical encoding preserves the circular nature of time (23:00 is close to 00:00).
    """
    df = df.copy()

    # Ensure we have datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Cyclical day of week encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Cyclical month encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Binary indicators for key periods
    df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['is_solar_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 15)).astype(int)
    df['is_morning_ramp'] = ((df['hour'] >= 5) & (df['hour'] <= 8)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add price-derived features including spreads, lags, and rolling statistics.
    These capture price dynamics and momentum.
    """
    df = df.copy()

    # mFRR spread (Up - Down) - key feature for bidding strategy
    df['mfrr_spread'] = df['mfrr_price_up'] - df['mfrr_price_down']

    # Lagged prices (1h, 4h, 24h)
    for lag in [1, 4, 24]:
        df[f'price_up_lag_{lag}h'] = df['mfrr_price_up'].shift(lag)
        if lag <= 4:  # Only short lags for down price
            df[f'price_down_lag_{lag}h'] = df['mfrr_price_down'].shift(lag)

    # Price changes (momentum)
    df['price_up_change_1h'] = df['mfrr_price_up'].diff(1)
    df['price_up_change_4h'] = df['mfrr_price_up'].diff(4)
    df['price_up_change_24h'] = df['mfrr_price_up'].diff(24)

    # Rolling statistics
    for window in [4, 24]:
        df[f'price_up_mean_{window}h'] = df['mfrr_price_up'].rolling(window).mean()
        df[f'price_up_std_{window}h'] = df['mfrr_price_up'].rolling(window).std()
        df[f'price_up_min_{window}h'] = df['mfrr_price_up'].rolling(window).min()
        df[f'price_up_max_{window}h'] = df['mfrr_price_up'].rolling(window).max()

    # Price relative to rolling mean (normalized deviation)
    df['price_up_zscore_24h'] = (
        (df['mfrr_price_up'] - df['price_up_mean_24h']) /
        df['price_up_std_24h'].replace(0, np.nan)
    )

    # Spread rolling mean
    df['spread_mean_24h'] = df['mfrr_spread'].rolling(24).mean()

    return df


def add_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add features related to system imbalance.
    Greece is structurally short, so imbalance magnitude matters.
    """
    df = df.copy()

    # Lagged imbalance
    if 'net_imbalance_mw' in df.columns:
        df['imbalance_lag_1h'] = df['net_imbalance_mw'].shift(1)
        df['imbalance_lag_4h'] = df['net_imbalance_mw'].shift(4)

        # Rolling imbalance
        df['imbalance_mean_4h'] = df['net_imbalance_mw'].rolling(4).mean()
        df['imbalance_mean_24h'] = df['net_imbalance_mw'].rolling(24).mean()

    # Imbalance price features
    if 'imbalance_price_long' in df.columns:
        df['imbalance_price_lag_1h'] = df['imbalance_price_long'].shift(1)
        df['imbalance_price_mean_24h'] = df['imbalance_price_long'].rolling(24).mean()

    return df


def add_load_res_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add features for load and renewable energy sources.
    RES penetration is key driver of price volatility.
    """
    df = df.copy()

    # Load features
    if 'load_mw' in df.columns:
        df['load_lag_1h'] = df['load_mw'].shift(1)
        df['load_lag_24h'] = df['load_mw'].shift(24)
        df['load_change_24h'] = df['load_mw'].diff(24)
        df['load_mean_24h'] = df['load_mw'].rolling(24).mean()

    # RES features
    if 'solar_forecast_mw' in df.columns:
        df['solar_lag_1h'] = df['solar_forecast_mw'].shift(1)

    if 'wind_forecast_mw' in df.columns:
        df['wind_lag_1h'] = df['wind_forecast_mw'].shift(1)
        df['wind_mean_24h'] = df['wind_forecast_mw'].rolling(24).mean()

    # RES penetration features (already have res_total_mw)
    if 'res_penetration' in df.columns:
        df['res_penetration_lag_1h'] = df['res_penetration'].shift(1)
        df['res_penetration_mean_24h'] = df['res_penetration'].rolling(24).mean()

    # Net residual load (load - RES)
    if 'load_mw' in df.columns and 'res_total_mw' in df.columns:
        df['residual_load_mw'] = df['load_mw'] - df['res_total_mw']
        df['residual_load_lag_1h'] = df['residual_load_mw'].shift(1)

    return df


def add_cross_border_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add features for cross-border electricity flows.
    Net position affects balancing needs.
    """
    df = df.copy()

    if 'net_cross_border_mw' in df.columns:
        # Binary indicator
        df['is_net_importer'] = (df['net_cross_border_mw'] > 0).astype(int)

        # Lagged flows
        df['net_flow_lag_1h'] = df['net_cross_border_mw'].shift(1)
        df['net_flow_lag_24h'] = df['net_cross_border_mw'].shift(24)

        # Rolling mean
        df['net_flow_mean_24h'] = df['net_cross_border_mw'].rolling(24).mean()

    # Per-neighbor features (only for major neighbors with good data)
    for neighbor in ['BG', 'MK', 'TR']:
        col = f'flow_{neighbor}_mw'
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)

    return df


def add_hydro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add features for hydro reservoir levels.
    Higher hydro = more flexibility = lower balancing prices.
    """
    df = df.copy()

    if 'hydro_reservoir_gwh' in df.columns:
        # Weekly change (since data is weekly)
        df['hydro_change_week'] = df['hydro_reservoir_gwh'].diff(168)  # 168 hours = 1 week

        # Normalized level (0-1 scale based on observed range)
        hydro_min = df['hydro_reservoir_gwh'].min()
        hydro_max = df['hydro_reservoir_gwh'].max()
        df['hydro_level_normalized'] = (
            (df['hydro_reservoir_gwh'] - hydro_min) / (hydro_max - hydro_min)
        )

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I add interaction features that combine multiple signals.
    These capture complex market dynamics.
    """
    df = df.copy()

    # Peak hour + high load = stress
    if 'load_mw' in df.columns:
        load_median = df['load_mw'].median()
        df['peak_high_load'] = (
            df['is_peak_hour'] * (df['load_mw'] > load_median).astype(int)
        )

    # Solar hour + high solar = potential negative prices
    if 'solar_forecast_mw' in df.columns:
        solar_p75 = df['solar_forecast_mw'].quantile(0.75)
        df['solar_hour_high_solar'] = (
            df['is_solar_hour'] * (df['solar_forecast_mw'] > solar_p75).astype(int)
        )

    # Weekend + night = very low demand
    df['weekend_night'] = df['is_weekend'] * df['is_night']

    # High RES + low load = oversupply
    if 'res_penetration' in df.columns:
        df['high_res_penetration'] = (df['res_penetration'] > 0.5).astype(int)

    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    I select the final feature set, dropping intermediate columns
    and columns with too many missing values.
    """
    # Columns to definitely keep (targets and key features)
    keep_cols = [
        # Target variables
        'mfrr_price_up', 'mfrr_price_down', 'mfrr_spread',

        # Time features
        'hour', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos',
        'is_weekend', 'is_peak_hour', 'is_solar_hour', 'is_morning_ramp', 'is_night',

        # Price features
        'price_up_lag_1h', 'price_up_lag_4h', 'price_up_lag_24h',
        'price_down_lag_1h', 'price_down_lag_4h',
        'price_up_change_1h', 'price_up_change_24h',
        'price_up_mean_4h', 'price_up_mean_24h',
        'price_up_std_24h',
        'price_up_zscore_24h',
        'spread_mean_24h',

        # System state
        'load_mw', 'load_lag_24h', 'load_mean_24h',
        'res_total_mw', 'solar_forecast_mw', 'wind_forecast_mw',
        'res_penetration', 'residual_load_mw',

        # Imbalance
        'net_imbalance_mw', 'imbalance_mean_24h',
        'imbalance_price_long',

        # Cross-border
        'net_cross_border_mw', 'is_net_importer', 'net_flow_mean_24h',
        'flow_BG_mw', 'flow_MK_mw', 'flow_TR_mw',

        # Hydro
        'hydro_reservoir_gwh', 'hydro_level_normalized',

        # Interaction features
        'peak_high_load', 'solar_hour_high_solar', 'weekend_night', 'high_res_penetration',
    ]

    # Filter to columns that exist
    available_cols = [c for c in keep_cols if c in df.columns]

    return df[available_cols]


def create_training_dataset(input_path: str, output_path: str,
                            min_coverage: float = 0.3) -> pd.DataFrame:
    """
    I create the final training dataset with all engineered features.

    Args:
        input_path: Path to merged dataset
        output_path: Path to save feature-engineered dataset
        min_coverage: Minimum fraction of non-null values required per column

    Returns:
        Feature-engineered DataFrame
    """
    print("=" * 70)
    print("Feature Engineering for mFRR Bidding")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    # Clip extreme prices
    print("\n2. Clipping extreme prices...")
    df = clip_extreme_prices(df)

    # Add features
    print("\n3. Adding time features...")
    df = add_time_features(df)

    print("4. Adding price features...")
    df = add_price_features(df)

    print("5. Adding imbalance features...")
    df = add_imbalance_features(df)

    print("6. Adding load/RES features...")
    df = add_load_res_features(df)

    print("7. Adding cross-border features...")
    df = add_cross_border_features(df)

    print("8. Adding hydro features...")
    df = add_hydro_features(df)

    print("9. Adding interaction features...")
    df = add_interaction_features(df)

    # Select final features
    print("\n10. Selecting final features...")
    df = select_final_features(df)

    # Drop columns with too few values
    print(f"\n11. Filtering columns (min coverage: {min_coverage*100:.0f}%)...")
    coverage = df.notna().mean()
    keep_cols = coverage[coverage >= min_coverage].index.tolist()
    df = df[keep_cols]

    # Drop rows with missing target
    print("\n12. Dropping rows with missing target...")
    initial_rows = len(df)
    df = df.dropna(subset=['mfrr_price_up'])
    print(f"   Dropped {initial_rows - len(df)} rows")

    # Sort by index
    df = df.sort_index()

    # Save
    print(f"\n13. Saving to {output_path}...")
    df.to_csv(output_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final dataset: {len(df)} rows x {len(df.columns)} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nFeatures ({len(df.columns)}):")
    for col in df.columns:
        coverage_pct = 100 * df[col].notna().mean()
        print(f"   {col}: {coverage_pct:.1f}% complete")

    print("\n" + "=" * 70)
    print(f"SAVED: {output_path}")
    print("=" * 70)

    return df


if __name__ == "__main__":
    input_path = Path(__file__).parent / "mfrr_training_data.csv"
    output_path = Path(__file__).parent / "mfrr_features.csv"

    df = create_training_dataset(str(input_path), str(output_path))
