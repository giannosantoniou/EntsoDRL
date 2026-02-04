import argparse
import sys
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.daily_env import DayAheadBatchEnv
from data.data_loader import load_historical_data

def train_daily(args):
    # 1. Load Data
    print(f"Loading data from {args.data_path}...")
    df = load_historical_data(args.data_path)
    
    # 2. Setup Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
        'cost_per_mwh_throughput': 15.0
    }
    
    # 3. Create Env
    def make_env():
        return DayAheadBatchEnv(df, battery_params)
        
    env = DummyVecEnv([make_env])
    # Normalize inputs/rewards is crucial for PPO
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 4. Model (Larger Net for 24h planning)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    model = PPO("MlpPolicy", env, verbose=1, 
                tensorboard_log="./logs_daily/", 
                learning_rate=0.0003,
                n_steps=365, # Update once per "Year" (365 steps)? No, 365 days is small batch.
                batch_size=64,
                policy_kwargs=policy_kwargs)
                
    print("Starting Daily PPO Training...")
    model.learn(total_timesteps=args.timesteps)
    
    model.save("models/ppo_daily_planner")
    env.save("models/vec_normalize_daily.pkl")
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50000) # 50k Days = 136 years
    parser.add_argument("--data_path", type=str, default="data/processed_training_data.csv")
    args = parser.parse_args()
    train_daily(args)
