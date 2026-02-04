
import pandas as pd
import numpy as np
import os

def generate_feasible_dam():
    input_path = "data/processed_training_data.csv"
    output_path = "data/feasible_data_with_dam.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("Loading base data...")
    df = pd.read_csv(input_path)
    
    # Battery Physics for Simulation
    capacity = 50.0
    max_power = 50.0
    efficiency = 0.94
    soc = 0.5
    
    dam_commitments = []
    
    # Rolling window stats for "Smart" simulation
    window = 24
    
    print("Simulating Ideal Battery Strategy to create Feasible DAM Commitments...")
    
    for i in range(len(df)):
        price = df.iloc[i]['price']
        
        # Look at surrounding prices (cheat a bit to be smart)
        start = max(0, i - window)
        end = min(len(df), i + window)
        local_avg = df.iloc[start:end]['price'].mean()
        
        # Simple Logic: Sell High, Buy Low
        # But respect SOC Constraints strictly
        
        action_mw = 0.0
        
        # DECISION
        if price > local_avg * 1.05: # High Price -> Discharge
            # How much can we really discharge?
            # Max due to Power: 50
            # Max due to Energy: (SoC - 0.05) * Cap * sqrt(eff)
            max_energy_out = (soc - 0.05) * capacity * np.sqrt(efficiency)
            feasible_mw = min(max_power, max_energy_out)
            action_mw = feasible_mw # Commit to this
            
        elif price < local_avg * 0.95: # Low Price -> Charge
            # How much can we really charge?
            # Max due to Power: -50
            # Max due to Energy: (0.95 - SoC) * Cap / sqrt(eff)
            max_energy_in = (0.95 - soc) * capacity / np.sqrt(efficiency)
            feasible_mw = -min(max_power, max_energy_in)
            action_mw = feasible_mw
            
        # UPDATE SOC (Simulation)
        if action_mw < 0:
            soc += (abs(action_mw) * np.sqrt(efficiency)) / capacity
        else:
            soc -= (action_mw / np.sqrt(efficiency)) / capacity
            
        soc = np.clip(soc, 0.0, 1.0)
        
        # Store Commitment
        # Normalize to -1..1 for consistency with other columns if needed, 
        # but here we store MW.
        dam_commitments.append(action_mw)
        
    df['dam_commitment'] = dam_commitments
    
    # Check stats
    feasible_count = len([x for x in dam_commitments if abs(x) > 0.1])
    print(f"Generated {len(df)} steps. Non-zero commitments: {feasible_count}")
    
    df.to_csv(output_path, index=False)
    print(f"Saved feasible data to {output_path}")

if __name__ == "__main__":
    generate_feasible_dam()
