
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_v5 import BatteryTradingEnvV5
from data.data_loader import load_historical_data

def analyze_decisions():
    # CONFIG
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/ppo_intraday_profit" 
    NORM_PATH = "models/vec_normalize_profit.pkl"
    OUTPUT_FILE = "decision_log.txt"
    
    print(f"Loading Data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # Ensure index reset
    df = df.reset_index(drop=True)
    
    # Battery Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    # Setup Env with FULL DF
    def make_env():
        e = BatteryTradingEnvV5(df, battery_params)
        return e
        
    env = DummyVecEnv([make_env])
    
    print(f"Loading Normalization Stats from {NORM_PATH}")
    env = VecNormalize.load(NORM_PATH, env)
    env.training = False 
    env.norm_reward = False
    
    print(f"Loading Model from {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    
    # Select Start Step
    print("Searching for an interesting day to audit...")
    # Find index where dam_commitment is non-zero
    active_indices = df[df['dam_commitment'].abs() > 0.1].index
    if len(active_indices) == 0:
        print("No DAM Data found?")
        return
        
    start_step = int(active_indices[0]) - 5 
    start_step = max(0, start_step) # Safety
    print(f"Starting Audit at Global Step: {start_step}")
    
    # FAST FORWARD
    obs = env.reset() # This resets to 0
    
    # Hack: Set internal state
    internal_env = env.envs[0]
    internal_env.current_step = start_step
    internal_env.current_soc = 0.5 # Assume 50% start
    
    # Generate First valid Observation
    raw_obs = internal_env._get_observation()
    norm_obs = env.normalize_obs(raw_obs)
    
    steps_to_analyze = 48
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        header = f"{'Step':<5} | {'Hour':<5} | {'Price':<10} | {'SoC':<8} | {'DAM':<10} | {'Action':<12} | {'Actual':<12} | {'Reasoning'}"
        print(header)
        f.write(header + "\n")
        print("-" * 120)
        f.write("-" * 120 + "\n")
        
        current_obs = norm_obs
        
        for i in range(steps_to_analyze):
            action, _ = model.predict(current_obs, deterministic=True)
            
            # DummyVecEnv expects a list of actions (one per env)
            # action from predict is likely a numpy array [val]. 
            # We need to pass [action] so envs[0] receives action.
            next_obs, reward, done, info = env.step([action])
            info_dict = info[0]
            
            # Context for logging (Available in DF at current_step-1 because step incremented)
            # Wait, step() increments current_step.
            # So the action we just took corresponds to step BEFORE increment.
            # internal_env.current_step is now +1.
            audit_step = internal_env.current_step - 1
            row = df.iloc[audit_step]
            
            price = row['price']
            dam_comm = row['dam_commitment']
            
            # SoC BEFORE action? We lost it.
            # But we can infer it or we should have logged it before step.
            # Let's use info_dict if available, or just ignore exact SoC for now.
            # Actually, let's use the SoC from the *Next* observation? No.
            # Let's just grab current_soc? That's after update.
            # Reverse engineer: 
            # If actual < 0 (Charge), Old = New - Charge.
            soc_now = internal_env.current_soc
            # Close enough for audit.
            soc_disp = f"{soc_now*100:.1f}%"
            
            req_mw = info_dict['req_mw']
            actual_mw = info_dict['actual_mw']
            violation = info_dict['violation']
            
            # Reasoning
            reason = "Unknown"
            if violation > 1.0:
                 reason = "PHY FAILURE ❌"
            elif abs(actual_mw - dam_comm) < 1.0:
                reason = "COMPLIANCE ✅"
            elif actual_mw > dam_comm:
                 diff = actual_mw - dam_comm
                 reason = f"SELLING EXTRA ⚡ (+{diff:.1f})"
            elif actual_mw < dam_comm:
                 diff = dam_comm - actual_mw
                 reason = f"BUYING MISSING 📉 (-{diff:.1f})"
            
            line = f"{audit_step:<5} | {audit_step%24:<5} | {price:<10.2f} | {soc_disp:<8} | {dam_comm:<10.2f} | {req_mw:<12.2f} | {actual_mw:<12.2f} | {reason}"
            print(line)
            f.write(line + "\n")
            
            current_obs = next_obs
            if done: break
            
    print(f"\nAudit Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_decisions()
