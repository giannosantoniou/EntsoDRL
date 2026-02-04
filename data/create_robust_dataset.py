
import pandas as pd
import numpy as np
import os

def create_robust_dataset(input_path="data/processed_training_data.csv", output_path="data/robust_training_data.csv"):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check if 'price' column exists (case sensitive)
    # Our data likely has lowercase columns from previous steps
    if 'price' not in df.columns and 'Price' in df.columns:
        df = df.rename(columns={'Price': 'price'})
        
    original_len = len(df)
    print(f"Original Data Length: {original_len}")
    
    # 1. Create a Copy for Augmentation
    df_aug = df.copy()
    
    # 2. Inject Extreme High Prices (Crisis Scenarios)
    # Target: 5% of data.
    # Logic: Pick random indices, set Price to Uniform(1000, 4000)
    n_high = int(len(df_aug) * 0.05)
    high_idx = np.random.choice(df_aug.index, n_high, replace=False)
    
    print(f"Injecting {n_high} Extreme High Price events (1000-4000 EUR)...")
    df_aug.loc[high_idx, 'price'] = np.random.uniform(1000, 4000, size=n_high)
    
    # 3. Inject Extreme Low/Negative Prices (Renewable Glut)
    # Target: 5% of data.
    n_low = int(len(df_aug) * 0.05)
    low_idx = np.random.choice(df_aug.index, n_low, replace=False) # Independent from high
    
    print(f"Injecting {n_low} Negative Price events (-500 to -50 EUR)...")
    df_aug.loc[low_idx, 'price'] = np.random.uniform(-500, -50, size=n_low)
    
    # 4. Inject "Dunkelflaute" (Dark Wind Lull)
    # Solar = 0, Wind = 0, Load = High, Price = High
    # Pick contiguous blocks
    n_blocks = 20
    block_size = 48 # 2 days
    
    print(f"Injecting {n_blocks} Dunkelflaute events (Zero RES, High Load)...")
    for _ in range(n_blocks):
        start = np.random.randint(0, len(df_aug) - block_size)
        indices = range(start, start + block_size)
        
        df_aug.loc[indices, 'solar'] = 0.0
        df_aug.loc[indices, 'wind_onshore'] = 0.0
        # Boost Load by 20%
        if 'load' in df_aug.columns:
            df_aug.loc[indices, 'load'] *= 1.2
            
        # Price usually spikes here too
        df_aug.loc[indices, 'price'] = np.maximum(df_aug.loc[indices, 'price'], np.random.uniform(500, 1500))

    # 5. Concatenate Original + Augmented
    # We keep original to preserve ground truth realism, add augmented for robustness.
    df_final = pd.concat([df, df_aug], ignore_index=True)
    
    # Shuffle? No, LSTM/PPO likes temporal consistency.
    # But concatenating breaks temporal link at the seam.
    # PPO treats episodes. If we don't reset index properly, it's one big episode?
    # BatteryEnv resets at end of DF.
    # So we effectively have 2 episodes: Normal History -> Robust History.
    
    print(f"Final Dataset Length: {len(df_final)}")
    df_final.to_csv(output_path, index=False)
    print(f"Saved robust data to {output_path}")

if __name__ == "__main__":
    np.random.seed(42) # Reproducibility
    create_robust_dataset()
