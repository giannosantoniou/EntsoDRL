
import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_v3 import BatteryTradingEnvV3
from data.data_loader import load_historical_data

class DegradationCurriculumCallback(BaseCallback):
    def __init__(self, target_cost: float, warmup_steps: int, verbose=0):
        super().__init__(verbose)
        self.target_cost = target_cost
        self.warmup_steps = warmup_steps
        
    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        current_cost = progress * self.target_cost
        self.training_env.env_method("set_cost", current_cost)
        if self.num_timesteps % 50000 == 0:
            print(f"Curriculum: Step {self.num_timesteps}, Cost set to {current_cost:.2f} EUR/MWh")
        return True

def train_ppo_16():
    # CONFIG
    DATA_PATH = "data/processed_data_with_dam.csv"
    MODEL_PATH = "models/ppo_intraday_v1"
    NORM_PATH = "models/vec_normalize_intraday.pkl"
    TIMESTEPS = 1000000 
    
    print(f"Loading DATA with DAM Commitments: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # Battery Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
        'cost_per_mwh_throughput': 0.0, # Start 0
        'transaction_fee': 0.5,
        'market_impact': 0.0, # Simplification for V1
        'liquidity_risk': 0.0 # Simplification
    }
    
    def make_env():
        e = BatteryTradingEnvV3(df, battery_params)
        e = Monitor(e)
        return e
        
    env = DummyVecEnv([make_env])
    # Clip 100 to handle price spikes if present
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.)
    
    print(f"Env Created. Obs Size: {env.observation_space.shape}")
    
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/intraday/")
    
    print("Starting Training PPO_16...")
    
    callback = DegradationCurriculumCallback(target_cost=15.0, warmup_steps=500000)
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    
    model.save(MODEL_PATH)
    env.save(NORM_PATH)
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    train_ppo_16()
