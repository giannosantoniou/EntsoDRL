"""
Data Upsampling Utility - Convert Hourly Data to 15-Minute Resolution.

Supports:
- Step interpolation (forward fill) - for DAM prices (constant per hour)
- Linear interpolation - for smoother intraday/balancing simulation
"""
import numpy as np
import pandas as pd
from typing import Literal


def upsample_hourly_to_15min(
    df: pd.DataFrame,
    method: Literal['step', 'linear'] = 'step',
    energy_columns: list = None,
    divide_energy_by_4: bool = True
) -> pd.DataFrame:
    """
    Convert hourly DataFrame to 15-minute resolution.
    
    Args:
        df: Input DataFrame with DatetimeIndex at hourly frequency
        method: 'step' for forward fill (DAM prices), 'linear' for interpolation
        energy_columns: Columns representing energy (e.g., 'dam_commitment') 
                       that should be divided by 4 (or repeated as-is)
        divide_energy_by_4: If True, divide energy columns by 4; if False, repeat
    
    Returns:
        DataFrame with 15-minute resolution (4× more rows)
    """
    if energy_columns is None:
        energy_columns = ['dam_commitment']
    
    # Validate input
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Store original length for verification
    original_len = len(df)
    
    # Create 15-minute index spanning the same time period
    start = df.index[0]
    end = df.index[-1]
    new_index = pd.date_range(start=start, end=end, freq='15T')
    
    # Resample based on method
    if method == 'step':
        # Forward fill - same value for all 4 slots per hour
        df_15min = df.resample('15T').ffill()
    elif method == 'linear':
        # Linear interpolation for smoother transitions
        df_15min = df.resample('15T').interpolate(method='linear')
    else:
        raise ValueError(f"Unknown method: {method}. Use 'step' or 'linear'")
    
    # Handle energy columns (divide by 4 if requested)
    if divide_energy_by_4:
        for col in energy_columns:
            if col in df_15min.columns:
                df_15min[col] = df_15min[col] / 4.0
    
    # Recalculate time features for 96 slots per day
    df_15min = _recalculate_time_features(df_15min)
    
    # Verification
    expected_len = (original_len - 1) * 4 + 1  # Ffill doesn't extend beyond last hour
    actual_len = len(df_15min)
    
    print(f"Upsampling complete: {original_len} hours → {actual_len} 15-min slots")
    print(f"  Method: {method}")
    print(f"  Energy columns ({'divided by 4' if divide_energy_by_4 else 'repeated'}): {energy_columns}")
    
    return df_15min


def _recalculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate time-based features for 15-minute resolution.
    
    For 15-min data, there are 96 slots per day (vs 24 for hourly).
    """
    # Calculate slot index: (hour * 4) + (minute // 15)
    slot_of_day = df.index.hour * 4 + df.index.minute // 15  # 0-95
    
    # Recalculate hour_sin and hour_cos for 96 slots
    df['hour_sin'] = np.sin(2 * np.pi * slot_of_day / 96)
    df['hour_cos'] = np.cos(2 * np.pi * slot_of_day / 96)
    
    # Recalculate day features if present
    if 'day_sin' in df.columns:
        day_of_week = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    return df


def verify_upsampling(df_hourly: pd.DataFrame, df_15min: pd.DataFrame) -> dict:
    """
    Verify that upsampling was performed correctly.
    
    Returns dict with verification results.
    """
    results = {
        'passed': True,
        'checks': []
    }
    
    # Check 1: Row count
    expected_rows = (len(df_hourly) - 1) * 4 + 1
    actual_rows = len(df_15min)
    row_check = actual_rows >= expected_rows - 1  # Allow slight tolerance
    results['checks'].append({
        'name': 'row_count',
        'passed': row_check,
        'expected': expected_rows,
        'actual': actual_rows
    })
    if not row_check:
        results['passed'] = False
    
    # Check 2: Price preservation (for step method, first 4 slots should match first hour)
    if 'price' in df_hourly.columns and 'price' in df_15min.columns:
        first_hour_price = df_hourly['price'].iloc[0]
        first_4_slots = df_15min['price'].iloc[:4]
        price_check = np.allclose(first_4_slots, first_hour_price, rtol=0.01)
        results['checks'].append({
            'name': 'price_preservation',
            'passed': price_check,
            'expected': first_hour_price,
            'actual': first_4_slots.tolist()
        })
        if not price_check:
            results['passed'] = False
    
    # Check 3: Time features range
    if 'hour_sin' in df_15min.columns:
        sin_range = df_15min['hour_sin'].min() >= -1.01 and df_15min['hour_sin'].max() <= 1.01
        cos_range = df_15min['hour_cos'].min() >= -1.01 and df_15min['hour_cos'].max() <= 1.01
        time_check = sin_range and cos_range
        results['checks'].append({
            'name': 'time_features_range',
            'passed': time_check,
            'sin_range': [df_15min['hour_sin'].min(), df_15min['hour_sin'].max()],
            'cos_range': [df_15min['hour_cos'].min(), df_15min['hour_cos'].max()]
        })
        if not time_check:
            results['passed'] = False
    
    # Check 4: Index frequency
    inferred_freq = pd.infer_freq(df_15min.index[:10])
    freq_check = inferred_freq in ['15T', '15min']
    results['checks'].append({
        'name': 'index_frequency',
        'passed': freq_check,
        'expected': '15T',
        'actual': inferred_freq
    })
    if not freq_check:
        results['passed'] = False
    
    return results


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing upsample_to_15min utility...\n")
    
    # Create sample hourly data
    dates = pd.date_range(start='2024-01-01', periods=48, freq='H')
    df_hourly = pd.DataFrame({
        'price': np.sin(np.linspace(0, 4*np.pi, 48)) * 30 + 70,  # Sinusoidal prices 40-100
        'dam_commitment': np.random.choice([10, 20, -10, -20, 0], size=48),
        'solar': np.maximum(0, np.sin(np.linspace(0, 4*np.pi, 48)) * 500),
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dates.hour / 24),
    }, index=dates)
    
    print("Original hourly data:")
    print(f"  Shape: {df_hourly.shape}")
    print(f"  Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")
    print()
    
    # Test step interpolation
    print("=" * 50)
    df_15min_step = upsample_hourly_to_15min(df_hourly, method='step')
    print()
    
    # Verify
    verification = verify_upsampling(df_hourly, df_15min_step)
    print(f"Verification: {'PASSED ✓' if verification['passed'] else 'FAILED ✗'}")
    for check in verification['checks']:
        status = '✓' if check['passed'] else '✗'
        print(f"  {status} {check['name']}")
    
    print()
    print("Sample 15-min data (first 8 rows):")
    print(df_15min_step.head(8).to_string())
