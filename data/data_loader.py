import numpy as np
import pandas as pd
from typing import Optional

def generate_sine_wave_data(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic sine wave price data for Pattern Recognition training.
    """
    np.random.seed(seed)
    hours = days * 24
    time_index = pd.date_range(start='2024-01-01', periods=hours, freq='H')
    
    # Base Sine Wave: Low at night, High at noon
    # 24 hour cycle
    # x goes from 0 to 2pi * days
    x = np.linspace(0, 2 * np.pi * days, hours)
    
    # Sine wave between -1 and 1 -> Shift to 50 +/- 30
    price_base = 50 + 30 * np.sin(x - np.pi/2) # Shift peak to ~noon
    
    # Add some noise
    noise = np.random.normal(0, 5, hours)
    price = price_base + noise
    
    # Ensure positive prices for now
    price = np.maximum(price, 0.0)
    
    # Create Forecast (perfect knowledge + noise)
    forecast_noise = np.random.normal(0, 2, hours)
    forecast_price = price + forecast_noise
    
    df = pd.DataFrame(index=time_index)
    df['price'] = price
    df['forecast_price'] = forecast_price
    df['hour'] = df.index.hour / 23.0 # Normalize
    df['day_of_week'] = df.index.dayofweek / 6.0 # Normalize
    
    return df

def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Load real historical data from CSV.
    Assumes first column is timestamp/index.
    """
    try:
        # Try reading with 'timestamp' column first
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    except ValueError:
        # Fallback: Read first column as index/timestamp
        df = pd.read_csv(filepath, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Ensure index is datetime (redundant but safe)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    # Convert timezone if needed (e.g. to Athens) or keep UTC.
    # Battery Env uses local time features? 
    # prepare_data.py generated features based on Athens time.
    # So hour_sin/cos are correct relative to functionality.
    # We just need df.index.hour to work.
    
    # Create features if missing
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour / 23.0
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df.index.dayofweek / 6.0
        
    return df
