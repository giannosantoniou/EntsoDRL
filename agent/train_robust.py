
import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env import BatteryTradingEnvV2
from gym_envs.action_mask import ActionMasker
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
        if self.num_timesteps % 10000 == 0:
            print(f"Curriculum: Step {self.num_timesteps}, Cost set to {current_cost:.2f} EUR/MWh")
        return True

def train_robust():
    # PATHS
    DATA_PATH = "data/robust_training_data.csv"
    MODEL_PATH = "models/ppo_battery_agent_robust"
    NORM_PATH = "models/vec_normalize_robust.pkl"
    TIMESTEPS = 2000000 # 2M Steps for profound learning
    
    # 1. Load Data
    print(f"Loading ROBUST Data from {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # 2. Battery Params (Identical to Production)
    TARGET_DEGRADATION_COST = 15.0
    battery_params = {
        'capacity_mwh': 50.0,
        'max_charge_mw': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
        'cost_per_mwh_throughput': 0.0, 
        'transaction_fee': 0.5,
        'market_impact': 0.01,
        'liquidity_risk': 0.05,
        'observation_noise': 0.05 
    }
    
    # 3. Env
    def make_env():
        e = BatteryTradingEnvV2(df, battery_params)
        e = ActionMasker(e) 
        e = Monitor(e)
        return e
        
    env = DummyVecEnv([make_env])
    # IMPORTANT: clip_obs=100.0 instead of 10.0 because we have Spikes (4000 EUR > 10 Sigma?)
    # If Norm Mean is 1.4, 4000 is 40.0. So 10 clip is too small.
    # We set clip to 100.
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100.)
    
    print(f"DEBUG: Env Created. Obs Space: {env.observation_space.shape}")
    
    # 4. Agent
    # MlpPolicy is fine.
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/robust/")
    
    # 5. Train
    print(f"Starting ROBUST training for {TIMESTEPS} timesteps...")
    
    curriculum = DegradationCurriculumCallback(
        target_cost=TARGET_DEGRADATION_COST, 
        warmup_steps=int(TIMESTEPS * 0.3) # Faster warmup
    )
    
    model.learn(total_timesteps=TIMESTEPS, callback=curriculum)
    
    # 6. Save (Distinct Files)
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    env.save(NORM_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Normalization stats saved to {NORM_PATH}")

if __name__ == "__main__":
    train_robust()
