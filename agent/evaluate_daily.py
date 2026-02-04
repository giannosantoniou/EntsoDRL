import argparse
import sys
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.daily_env import DayAheadBatchEnv
from data.data_loader import load_historical_data

def evaluate_daily(args):
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    df_full = load_historical_data(args.data_path)
    df_test = df_full[df_full.index.year == args.year]
    
    if df_test.empty:
        print(f"No data for year {args.year}")
        return

    # 2. Setup Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
        'cost_per_mwh_throughput': 15.0
    }
    
    # 3. Create Env
    def make_env():
        return DayAheadBatchEnv(df_test, battery_params)
        
    env = DummyVecEnv([make_env])
    
    # Load Norm Stats
    norm_path = "models/vec_normalize_daily.pkl"
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    
    # 4. Load Model
    model = PPO.load("models/ppo_daily_planner")
    
    # 5. Run Loop
    obs = env.reset()
    done = False
    total_reward = 0
    
    hourly_logs = []
    
    day_idx = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Action is (1, 24)
        flat_action = action[0] 
        
        # Step env
        obs, reward, terminated, info = env.step(action)
        done = terminated[0]
        
        total_reward += reward[0]
        
        # Log Logic (Reconstruct Hourly from Info not possible easily? 
        # Actually daily_env doesn't return hourly traces in info.
        # We need to simulate locally or hack the env to return traces.
        # For now, let's just log the 'Action Plan' vs 'Outcome' summary.
        
        # Or better: Extract the actions (Plan)
        # We know the day index.
        start_idx = day_idx * 24
        if start_idx + 24 <= len(df_test):
            day_dates = df_test.index[start_idx : start_idx + 24]
            
            for h in range(24):
                log_entry = {
                    'Date': day_dates[h],
                    'Planned_DAM_MW': flat_action[h] * 50.0, # Approximate scaling
                    'Daily_Reward_Share': reward[0] / 24.0 # Crude approximation
                }
                hourly_logs.append(log_entry)
        
        day_idx += 1
        
    print(f"Total Daily Planner Profit: {total_reward:.2f} EUR")
    
    # Save Log
    if hourly_logs:
        pd.DataFrame(hourly_logs).to_csv("trade_log_daily.csv", index=False)
        print("Saved trade_log_daily.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--data_path", type=str, default="data/processed_training_data.csv")
    args = parser.parse_args()
    evaluate_daily(args)
