"""
MaskablePPO Training with 15-Minute Data Resolution.

Uses upsampled 15-minute data for finer-grained trading decisions.
Agent makes 4× more decisions per hour compared to hourly training.
"""
import os
import sys
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments
from data.upsample_to_15min import upsample_hourly_to_15min


def mask_fn(env):
    """Get action mask from environment."""
    return env.action_masks()


def train_15min():
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/ppo_15min"
    VECNORM_PATH = "models/vec_normalize_15min.pkl"
    CHECKPOINT_DIR = "models/checkpoints/ppo_15min/"
    
    # === HYPERPARAMETERS ===
    # With 15-min data: 4× more steps per day, so adjust total timesteps
    TIMESTEPS = 5_000_000  # Same total training, more granular episodes
    CHECKPOINT_FREQ = 200_000
    
    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 146.0,      # Updated: 146 MWh capacity (~5h duration)
        'max_discharge_mw': 30.0,   # Inverter rating: 30 MW (unchanged)
        'efficiency': 0.94,
    }
    
    # === LOAD AND UPSAMPLE DATA ===
    print(f"Loading data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    original_len = len(df)
    
    print("Regenerating DAM commitments (hourly)...")
    df = regenerate_dam_commitments(df, battery_params=battery_params)
    
    print(f"\nUpsampling to 15-minute resolution...")
    df_15min = upsample_hourly_to_15min(df, method='step', divide_energy_by_4=True)
    print(f"  Hourly: {original_len:,} rows")
    print(f"  15-min: {len(df_15min):,} rows (4× more decisions)")
    
    # === CREATE ENVIRONMENT ===
    def make_env():
        env = BatteryEnvMasked(
            df_15min, 
            battery_params, 
            n_actions=21,
            time_step_hours=0.25  # 15-minute intervals
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=10.)
    
    # === CREATE MODEL ===
    print("\nInitializing MaskablePPO Agent (15-min)...")
    
    from typing import Callable
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func
    
    # Hyperparameters optimized for 15-min granularity
    initial_lr = 3e-4
    n_steps = 4096         # Collect more steps before update
    batch_size = 256
    
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=linear_schedule(initial_lr),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,           # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,        # Exploration
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./logs/ppo_15min/"
    )
    
    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_15min",
        save_vecnormalize=True
    )
    
    # === TRAIN ===
    print(f"\n{'='*60}")
    print("Starting 15-Minute Resolution Training")
    print(f"{'='*60}")
    print(f"  Resolution: 15-minute (4 decisions/hour)")
    print(f"  Total Timesteps: {TIMESTEPS:,}")
    print(f"  Checkpoint Frequency: {CHECKPOINT_FREQ:,}")
    print(f"  TensorBoard: ./logs/ppo_15min/")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # === SAVE ===
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Model: {MODEL_PATH}")
    print(f"  VecNormalize: {VECNORM_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_15min()
