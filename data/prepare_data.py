import os
import pandas as pd
import numpy as np
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.entsoe_connector import EntsoeConnector

def prepare_data():
    load_dotenv()
    api_key = os.getenv('ENTSOE_API_KEY')
    if not api_key:
        print("Error: ENTSOE_API_KEY not found in .env")
        return

    connector = EntsoeConnector(api_key=api_key)
    
    # Range: 2021-2026
    start_str = '2021-01-01'
    # Fetch up to now
    end = pd.Timestamp.now(tz='Europe/Athens')
    
    print(f"Starting Data Preparation Pipeline ({start_str} to {end})...")
    
    # 1. Fetch Day-Ahead Prices
    # We might already have it, but let's fetch/merge to be sure or load from cache if implemented
    # For this script, we'll fetch everything fresh or check if file exists to save time?
    # Let's fetch clean chunks.
    
    # Helper to fetch by year to avoid timeouts
    years = range(2021, end.year + 1)
    
    dfs_price = []
    dfs_load = []
    dfs_res = []
    dfs_imb = []
    
    for year in years:
        y_start = pd.Timestamp(f'{year}-01-01', tz='Europe/Athens')
        y_end = pd.Timestamp(f'{year}-12-31 23:00', tz='Europe/Athens')
        if y_end > end: y_end = end
        
        print(f"--- Processing {year} ---")
        
        # Prices
        try:
            p = connector.fetch_day_ahead_prices(y_start, y_end)
            if not p.empty: dfs_price.append(p)
        except: pass
        
        # Load Forecast
        try:
            l = connector.fetch_load_forecast(y_start, y_end)
            if not l.empty: dfs_load.append(l)
        except: pass
        
        # RES Forecast
        try:
            r = connector.fetch_wind_solar_forecast(y_start, y_end)
            if not r.empty: dfs_res.append(r)
        except: pass
        
        # Imbalance Prices (ISP)
        try:
            i = connector.fetch_imbalance_prices(y_start, y_end)
            if not i.empty: dfs_imb.append(i)
        except: pass

    # 2. Merge Dataframes
    print("Merging datasets...")
    if not dfs_price:
        print("No price data found!")
        return
        
    df_price = pd.concat(dfs_price)
    # Remove duplicates
    df_price = df_price[~df_price.index.duplicated(keep='first')]
    
    final_df = df_price.copy()
    
    if dfs_load:
        df_load = pd.concat(dfs_load)
        df_load = df_load[~df_load.index.duplicated(keep='first')]
        final_df = final_df.join(df_load, how='left')
        
    if dfs_res:
        df_res = pd.concat(dfs_res)
        df_res = df_res[~df_res.index.duplicated(keep='first')]
        final_df = final_df.join(df_res, how='left')
        
    if dfs_imb:
        df_imb = pd.concat(dfs_imb)
        df_imb = df_imb[~df_imb.index.duplicated(keep='first')]
        # Rename columns to avoid collision if any, but ISP usually 'Long', 'Short' or 'Price'
        # entsoe-py returns 'Long', 'Short' usually
        final_df = final_df.join(df_imb, how='left')
        
    # Fill missing values (forward fill then backward fill)
    final_df = final_df.ffill().bfill()
    
    # 3. Feature Engineering
    print("Engineering features...")
    
    # Temporal Features
    # Sine/Cosine encoding for cyclic features
    final_df['hour_sin'] = np.sin(2 * np.pi * final_df.index.hour / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * final_df.index.hour / 24)
    
    final_df['day_sin'] = np.sin(2 * np.pi * final_df.index.dayofweek / 7)
    final_df['day_cos'] = np.cos(2 * np.pi * final_df.index.dayofweek / 7)
    
    final_df['month_sin'] = np.sin(2 * np.pi * final_df.index.month / 12)
    final_df['month_cos'] = np.cos(2 * np.pi * final_df.index.month / 12)
    
    # Residual Load Calculation (if available)
    # Load - (Wind + Solar)
    # Check column names from ENTSO-E (usually 'Solar', 'Wind Onshore')
    # We lowercased them in connector: 'solar', 'wind_onshore'
    
    res_cols = [c for c in final_df.columns if 'solar' in c or 'wind' in c]
    if 'load_forecast' in final_df.columns and res_cols:
        total_res = final_df[res_cols].sum(axis=1)
        final_df['residual_load_forecast'] = final_df['load_forecast'] - total_res
    
    # Price Lags (Autoregression)
    # Use Past 24h prices as features?
    # No, that explodes dimensions. Maybe just previous hour or 24h ago.
    # The gym env handles state construction usually.
    # But we can pre-calculate some rolling stats.
    
    # 4. Save
    output_path = "data/processed_training_data.csv"
    final_df.to_csv(output_path)
    print(f"Preparation Complete. Saved {len(final_df)} rows to {output_path}")
    print("Columns:", final_df.columns.tolist())

if __name__ == "__main__":
    prepare_data()
