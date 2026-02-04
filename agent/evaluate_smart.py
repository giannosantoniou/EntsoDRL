
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_v5 import BatteryTradingEnvV5
from data.data_loader import load_historical_data

def evaluate_smart_agent():
    # CONFIG
    DATA_PATH = "data/processed_data_with_dam.csv"
    MODEL_PATH = "models/ppo_intraday_smart"
    NORM_PATH = "models/vec_normalize_smart.pkl"
    REPORT_PATH = "smart_report.csv"
    
    print(f"Loading DATA: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # Use last 5000 steps for validation (Holdout)
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
    }
    
    def make_env():
        e = BatteryTradingEnvV5(val_df, battery_params)
        return e
        
    env = DummyVecEnv([make_env])
    
    if os.path.exists(NORM_PATH):
        print(f"Loading Normalization Stats from {NORM_PATH}")
        env = VecNormalize.load(NORM_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: No Normalization Stats found!")
        
    print(f"Loading Model from {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)
    
    obs = env.reset()
    
    results = []
    total_violations = 0
    total_trades = 0
    total_steps = 0
    
    print("Starting Evaluation Loop...")
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info_dict = info[0]
        
        violation_mw = info_dict['violation']
        req_mw = info_dict['req_mw']
        actual_mw = info_dict['actual_mw']
        
        dam_comm = val_df.iloc[total_steps]['dam_commitment']
        
        # Intraday Trade = Actual - DAM
        trade_mw = actual_mw - dam_comm
        
        is_violation = violation_mw > 1.0
        if is_violation: total_violations += 1
        
        if abs(trade_mw) > 1.0: total_trades += 1
            
        results.append({
            'step': total_steps,
            'dam_commitment': dam_comm,
            'req_dispatch': req_mw,
            'actual_dispatch': actual_mw,
            'violation_mw': violation_mw,
            'trade_mw': trade_mw,
            'reward': reward[0],
            'is_violation': is_violation
        })
        
        total_steps += 1
        if done: break
            
    # Save Report
    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORT_PATH, index=False)
    
    violation_rate = (total_violations / total_steps) * 100.0
    active_trading_rate = (total_trades / total_steps) * 100.0
    
    print("="*40)
    print("SMART AGENT (PPO_17b) REPORT")
    print("="*40)
    print(f"Total Steps: {total_steps}")
    print(f"Violations (>1MW): {total_violations} ({violation_rate:.2f}%)")
    print(f"Active Trades (>1MW): {total_trades} ({active_trading_rate:.2f}%)")
    print(f"Comparison: V4 had ~57% Violation Rate")
    print("="*40)

if __name__ == "__main__":
    evaluate_smart_agent()
