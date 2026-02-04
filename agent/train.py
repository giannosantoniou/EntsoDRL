import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env import BatteryTradingEnvV2
from gym_envs.action_mask import ActionMasker
from data.data_loader import generate_sine_wave_data, load_historical_data

class DegradationCurriculumCallback(BaseCallback):
    """
    Linearly increases degradation cost from 0 to TARGET_COST over warm-up period.
    """
    def __init__(self, target_cost: float, warmup_steps: int, verbose=0):
        super().__init__(verbose)
        self.target_cost = target_cost
        self.warmup_steps = warmup_steps
        
    def _on_step(self) -> bool:
        # Calculate current cost
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        current_cost = progress * self.target_cost
        
        # Update env(s)
        # Note: self.training_env is a VecEnv. Accessing inner envs is tricky if SubprocVecEnv, 
        # but DummyVecEnv is easy.
        # methods: get_attr, set_attr.
        
        # We can just set it directly for DummyVecEnv
        self.training_env.env_method("set_cost", current_cost)
        
        if self.num_timesteps % 10000 == 0:
            print(f"Curriculum: Step {self.num_timesteps}, Cost set to {current_cost:.2f} EUR/MWh")
            
        return True

def train(args):
    # 1. Load Data
    if args.mode == 'sine_wave':
        df = generate_sine_wave_data(days=365)
        print("Generated Sine Wave Data for Training")
    elif args.mode == 'historical':
        df = load_historical_data(args.data_path)
        print(f"Loaded Historical Data from {args.data_path}")
    else:
        raise ValueError("Invalid mode")

    # 2. Define Battery Parameters (Sungrow Lithium-ion LFP Profile - 50MW Scale)
    # Ref: Standard LFP BESS ~94% RTE, 0.5C-1C, ~6000 cycles
    TARGET_DEGRADATION_COST = 15.0
    battery_params = {
        'capacity_mwh': 50.0,          # 50MWh Capacity
        'max_charge_mw': 50.0,         # 50MW Charge
        'max_discharge_mw': 50.0,      # 50MW Discharge
        'efficiency': 0.94,            # High AC-AC efficiency
        'cost_per_mwh_throughput': 0.0, # Start at 0 for curriculum
        'transaction_fee': 0.5,         # 0.5 EUR/MWh Exchange Fee
        
        # Adversarial (Phase 6)
        'market_impact': 0.01,          # 1% Price Move per 50MW (Huge impact!)
        'liquidity_risk': 0.05,         # 5% chance TSO rejects you
        'observation_noise': 0.05       # 5% Noise on Forecasts
    }
    
    # 3. Create Environment
    # Note: ActionMasker is a Gym wrapper, usually applied inside the lambda
    # Create Env
    def make_env():
        # Use V2
        e = BatteryTradingEnvV2(df, battery_params)
        e = ActionMasker(e) 
        e = Monitor(e)
        return e
        
    env = DummyVecEnv([make_env])
    # Apply VecNormalize regarding observations and rewards
    # clip_obs=10.0 is standard. gamma should match PPO gamma (0.99 default).
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    print(f"DEBUG: Env Created (V2 + Norm). Obs Space: {env.observation_space.shape}")
    
    # Verify environment
    print("Verifying environment with check_env...")
    test_env = BatteryTradingEnvV2(df, battery_params)
    check_env(test_env, warn=True)
    print("Environment check passed.")
    
    # 4. Initialize Agent
    # 4. Initialize Agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    
    # 5. Train with Curriculum
    print(f"Starting training for {args.timesteps} timesteps with Curriculum...")
    
    # Warmup over 50% of training steps (e.g. 500k steps)
    curriculum_callback = DegradationCurriculumCallback(
        target_cost=TARGET_DEGRADATION_COST, 
        warmup_steps=int(args.timesteps * 0.5)
    )
    
    model.learn(total_timesteps=args.timesteps, callback=curriculum_callback)
    
    # 6. Save Model and Normalization Stats
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_battery_agent_v2")
    env.save("models/vec_normalize.pkl")
    print("Model saved to models/ppo_battery_agent_v2")
    print("Normalization stats saved to models/vec_normalize.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sine_wave", choices=["sine_wave", "historical"])
    parser.add_argument("--data_path", type=str, default="data/processed_training_data.csv")
    parser.add_argument("--timesteps", type=int, default=1000000)
    args = parser.parse_args()
    
    train(args)
