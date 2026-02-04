"""
DQN Training with Action Masking
Uses MaskableDQN to prevent agent from selecting physically impossible actions.
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


def mask_fn(env):
    """Get action mask from environment."""
    return env.action_masks()


def train_masked():
    # === PATHS ===
    # v9: 12-hour lookahead + Critical Bug Fixes (38 features)
    # - Symmetric penalty for BUY commitment violations (was missing!)
    # - SoC clipping to [min_soc, max_soc] instead of [0, 1]
    # - Shortfall lookahead extended to 12h (matches observation)
    # - Removed dead code (hardcoded hour=18)
    DATA_PATH = "data/feasible_data_with_balancing.csv"
    MODEL_PATH = "models/ppo_12h_lookahead_v9"
    VECNORM_PATH = "models/vec_normalize_12h_v9.pkl"
    CHECKPOINT_DIR = "models/checkpoints/ppo_12h_v9/"
    LOG_DIR = "./logs/ppo_12h_v9/"

    # === HYPERPARAMETERS ===
    TIMESTEPS = 10_000_000  # Extended training for 5h battery patterns
    CHECKPOINT_FREQ = 200_000
    
    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 146.0,      # Updated: 146 MWh capacity (~5h duration)
        'max_discharge_mw': 30.0,   # Inverter rating: 30 MW (unchanged)
        'efficiency': 0.94,
    }
    
    # === LOAD DATA ===
    print(f"Loading data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    print("Regenerating DAM commitments...")
    df = regenerate_dam_commitments(df, battery_params=battery_params)
    
    # === CREATE ENVIRONMENT ===
    def make_env():
        env = BatteryEnvMasked(df, battery_params, n_actions=21)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=10.)
    
    # === CREATE MODEL ===
    # Using MaskablePPO with expert optimizations
    print("\nInitializing MaskablePPO Agent (Optimized)...")
    
    # Linear learning rate schedule (decay to 0)
    from typing import Callable
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func
    
    # Expert configuration
    initial_lr = 3e-4      # Start LR
    n_steps = 4096         # Larger horizon (was 2048)
    batch_size = 256       # Larger batch (was 64)
    
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=linear_schedule(initial_lr),  # Dynamic LR decay
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,           # Slightly lower for trading (was 0.999)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,        # I increased to 0.05 for better exploration with new rewards
        policy_kwargs=dict(net_arch=[512, 256, 128]),  # 3 layers for 38 features (12h lookahead)
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    
    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_masked_imbal",
        save_vecnormalize=True
    )
    
    # === TRAIN ===
    print(f"\nStarting Training MaskablePPO (Imbalance Features)...")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    print(f"  - TensorBoard: ./logs/ppo_masked_imbal/")
    print(f"  - Total Steps: {TIMESTEPS:,}")
    print(f"  - Action Masking: ENABLED")
    
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
    train_masked()
