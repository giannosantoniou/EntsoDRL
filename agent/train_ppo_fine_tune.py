
import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_v5 import BatteryTradingEnvV5
from data.data_loader import load_historical_data

def train_fine_tune():
    # CONFIG
    # Continue from where PPO_17c left off
    DATA_PATH = "data/feasible_data_with_dam.csv" 
    LOAD_MODEL_PATH = "models/ppo_intraday_profit"
    LOAD_NORM_PATH = "models/vec_normalize_profit.pkl"
    
    NEW_MODEL_PATH = "models/ppo_intraday_finetuned"
    NEW_NORM_PATH = "models/vec_normalize_finetuned.pkl"
    
    TIMESTEPS = 1000000  # 1 Million Steps (The Marathon)
    NEW_LEARNING_RATE = 0.00005 # 5e-5 (Very slow, precise refinement)
    
    print(f"Loading FEASIBLE DATA: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # Battery Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    def make_env():
        e = BatteryTradingEnvV5(df, battery_params)
        e = Monitor(e)
        return e
        
    env = DummyVecEnv([make_env])
    
    # Load Normalization Stats (Crucial! Otherwise agent is blind)
    print(f"Loading Normalization Stats from {LOAD_NORM_PATH}")
    env = VecNormalize.load(LOAD_NORM_PATH, env)
    env.training = True # We ARE training, so update stats? 
    # Decision: Yes, keep updating stats as behavior changes, 
    # OR No, freeze stats to force agent to adapt to fixed view?
    # Standard: Keep training = True.
    env.norm_reward = True
    
    print(f"Loading Pre-Trained Model from {LOAD_MODEL_PATH}")
    # Enable Tensorboard Logging for the fine-tuning session
    model = PPO.load(LOAD_MODEL_PATH, env=env, tensorboard_log="./logs/fine_tune")
    
    # CRITICAL FIX: Increase Gamma for Long-Term Planning
    # Default 0.99 is too short for 24h cycle (0.99^24 = 0.78)
    # 0.999 extends horizon (0.999^24 = 0.97)
    model.gamma = 0.999
    print(f"Updated Discount Factor (Gamma) to {model.gamma}")

    # MANUAL LR UPDATE
    model.learning_rate = NEW_LEARNING_RATE
    # Note: SB3 updates the optimizer learning rate at the start of learning if it's a float.
    
    print(f"Starting Fine-Tuning (LR={NEW_LEARNING_RATE})...")
    print(f"Target: {TIMESTEPS} steps.")
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    
    model.save(NEW_MODEL_PATH)
    env.save(NEW_NORM_PATH)
    print("Fine-Tuning Complete. Model Saved.")

if __name__ == "__main__":
    train_fine_tune()
