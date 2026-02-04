
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

def train_ppo_17b():
    # CONFIG
    DATA_PATH = "data/processed_data_with_dam.csv"
    MODEL_PATH = "models/ppo_intraday_smart"
    NORM_PATH = "models/vec_normalize_smart.pkl"
    TIMESTEPS = 500000 
    
    print(f"Loading DATA: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # Battery Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
        'cost_per_mwh_throughput': 0.0, 
    }
    
    def make_env():
        e = BatteryTradingEnvV5(df, battery_params)
        e = Monitor(e)
        return e
        
    env = DummyVecEnv([make_env])
    # Clip 100 to handle price spikes
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.)
    
    print(f"Env Created (V5 - Smart Penalties). Obs Size: {env.observation_space.shape}")
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/smart/")
    
    print("Starting Training PPO_17b (500k Steps)...")
    
    model.learn(total_timesteps=TIMESTEPS)
    
    model.save(MODEL_PATH)
    env.save(NORM_PATH)
    print("Training PPO_17b Complete. Model Saved.")

if __name__ == "__main__":
    train_ppo_17b()
