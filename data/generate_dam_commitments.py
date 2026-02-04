
import pandas as pd
import numpy as np
import os

def generate_dam_commitments(input_path="data/processed_training_data.csv", output_path="data/processed_data_with_dam.csv"):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Ensure datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize DAM Commitment Column
    df['dam_commitment'] = 0.0
    
    # Simulation Params
    BATTERY_CAPACITY = 50.0 # MWh
    MAX_POWER = 50.0 # MW
    EFFICIENCY = 0.94
    CYCLES_PER_DAY = 1.0 # Standard strategy
    
    # We process day by day (chunks of 24)
    # Assuming data is hourly and contiguous
    
    n_days = len(df) // 24
    print(f"Processing {n_days} days of DAM simulation...")
    
    for day in range(n_days):
        start_idx = day * 24
        end_idx = start_idx + 24
        day_slice = df.iloc[start_idx:end_idx].copy()
        
        # 1. Get Prices
        prices = day_slice['price'].values
        
        # 2. Simulate Forecast Error (The DAM trader doesn't see actual prices perfectly)
        # Add 10% Noise to prices before deciding
        forecast_noise = np.random.normal(0, 0.1 * np.mean(np.abs(prices)), size=24)
        forecast_prices = prices + forecast_noise
        
        # 3. Simple Optimization Strategy (Price Spread)
        # Find lowest price hours to Charge, highest to Discharge
        # Sort hours by forecast price
        sorted_indices = np.argsort(forecast_prices)
        
        # Charge in cheapest N hours (e.g. 4 hours to fill 50MWh at 12.5MW? No, usually operate at full power)
        # Full Cycle: Charge 1 hour at 50MW = 50MWh. Discharge 1 hour.
        # Let's say we do 2 cycles per day aggressive, or 1 cycle conservative.
        # Let's randomize strategy per day!
        strategy = np.random.choice(['aggressive', 'conservative', 'broken'])
        
        dam_actions = np.zeros(24)
        
        if strategy == 'aggressive':
            # 2 Cycles. Top 4 hours Sell, Bottom 4 hours Buy.
            charge_indices = sorted_indices[:4]
            discharge_indices = sorted_indices[-4:]
            dam_actions[charge_indices] = -MAX_POWER # Charge
            dam_actions[discharge_indices] = MAX_POWER # Discharge
            
        elif strategy == 'conservative':
            # 1 Cycle. Top 2 hours Sell, Bottom 2 hours Buy.
            charge_indices = sorted_indices[:2]
            discharge_indices = sorted_indices[-2:]
            dam_actions[charge_indices] = -MAX_POWER
            dam_actions[discharge_indices] = MAX_POWER
            
        elif strategy == 'broken':
            # Simulates a "Force Majeure" or "Dunkelflaute" plan.
            # Maybe we committed to sell but have no energy?
            # Or constant output (e.g. Base load contract simulation)
            dam_actions[:] = 10.0 # Constant 10MW sell
            
        # 4. Apply "Operational Noise" (The user's request for variety)
        # Randomly drop a commitment (Outage)
        if np.random.random() < 0.1: # 10% of days have a trip
             trip_hour = np.random.randint(0, 24)
             dam_actions[trip_hour] = 0.0
             
        # Randomly Spike a commitment (Fat Finger)
        if np.random.random() < 0.01: # 1% of days
             fat_finger_hour = np.random.randint(0, 24)
             dam_actions[fat_finger_hour] *= 2.0 # Committed 100MW! Problem.
             
        # Interactive Learning: Agent must handle 'dam_actions' constraint.
        
        df.loc[start_idx:end_idx-1, 'dam_commitment'] = dam_actions
        
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows with realistic DAM commitments to {output_path}")

if __name__ == "__main__":
    np.random.seed(42)
    generate_dam_commitments()
