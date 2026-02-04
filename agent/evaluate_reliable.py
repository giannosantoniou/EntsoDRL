
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_v4 import BatteryTradingEnvV4
from data.data_loader import load_historical_data

def evaluate_reliable_agent():
    # CONFIG
    DATA_PATH = "data/processed_data_with_dam.csv"
    MODEL_PATH = "models/ppo_intraday_reliable"
    NORM_PATH = "models/vec_normalize_reliable.pkl"
    REPORT_PATH = "reliability_report.csv"
    
    print(f"Loading DATA: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # Use last 2000 steps for validation (Holdout)
    if len(df) > 5000:
        val_df = df.iloc[-5000:].reset_index(drop=True)
    else:
        val_df = df
        
    # Battery Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
        'cost_per_mwh_throughput': 0.0, 
        'transaction_fee': 0.5,
    }
    
    def make_env():
        e = BatteryTradingEnvV4(val_df, battery_params)
        return e
        
    env = DummyVecEnv([make_env])
    
    # Load Normalization Stats
    if os.path.exists(NORM_PATH):
        print(f"Loading Normalization Stats from {NORM_PATH}")
        env = VecNormalize.load(NORM_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: No Normalization Stats found! Model performance may be poor.")
        
    # Load Model
    print(f"Loading Model from {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    
    obs = env.reset()
    
    results = []
    total_violations = 0
    total_steps = 0
    
    print("Starting Evaluation Loop...")
    
    # Run one full episode
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Extract info (VectorEnv returns list of infos)
        info_dict = info[0]
        
        imbalance = info_dict['imbalance']
        penalty = info_dict['penalty']
        reward_val = reward[0]
        
        # Check Violation
        is_violation = abs(imbalance) > 1.0
        if is_violation:
            total_violations += 1
            
        results.append({
            'step': total_steps,
            'dam_commitment': val_df.iloc[total_steps]['dam_commitment'],
            'actual_dispatch': val_df.iloc[total_steps]['dam_commitment'] + imbalance,
            'imbalance': imbalance,
            'penalty': penalty,
            'reward': reward_val,
            'is_violation': is_violation
        })
        
        total_steps += 1
        if done:
            break
            
    # Save Report
    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORT_PATH, index=False)
    
    violation_rate = (total_violations / total_steps) * 100.0
    avg_penalty = res_df['penalty'].mean()
    
    print("="*40)
    print("RELIABILITY REPORT")
    print("="*40)
    print(f"Total Steps Evaluated: {total_steps}")
    print(f"Violations (>1MW): {total_violations} steps")
    print(f"Violation Rate: {violation_rate:.2f}%")
    print(f"Average Penalty Cost: {avg_penalty:.2f} EUR/step")
    print("="*40)
    
    # Analyze Specific Scenario: High Price + High Commitment
    # Did it deliver?
    print("Deep Dive Analysis saved to reliability_report.csv")

if __name__ == "__main__":
    evaluate_reliable_agent()
