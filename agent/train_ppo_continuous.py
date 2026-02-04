"""
PPO Training with Continuous Action Space
Uses standard PPO with action clamping for constraint satisfaction.
"""
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_continuous import BatteryEnvContinuous
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments


def train_ppo_continuous():
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/ppo_continuous"
    VECNORM_PATH = "models/vec_normalize_continuous.pkl"
    CHECKPOINT_DIR = "models/checkpoints/ppo_continuous/"
    
    # === HYPERPARAMETERS ===
    TIMESTEPS = 5_000_000
    CHECKPOINT_FREQ = 200_000
    
    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    # === LOAD DATA ===
    print(f"Loading data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    print("Regenerating DAM commitments...")
    df = regenerate_dam_commitments(df, battery_params=battery_params)
    
    # === CREATE ENVIRONMENT ===
    def make_env():
        env = BatteryEnvContinuous(df, battery_params)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # === CREATE MODEL ===
    print("\nInitializing PPO Agent (Continuous Action)...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,  # Long-term planning
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])),
        verbose=1,
        tensorboard_log="./logs/ppo_continuous/"
    )
    
    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_continuous",
        save_vecnormalize=True
    )
    
    # === TRAIN ===
    print(f"\nStarting Training PPO (Continuous Action)...")
    print(f"  - Action Space: Continuous [-1, 1] with clamping")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    print(f"  - TensorBoard: ./logs/ppo_continuous/")
    print(f"  - Total Steps: {TIMESTEPS:,}")
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # === SAVE ===
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    
    print("=" * 50)
    print("Training Complete!")
    print(f"  Model: {MODEL_PATH}")
    print(f"  VecNormalize: {VECNORM_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    train_ppo_continuous()
