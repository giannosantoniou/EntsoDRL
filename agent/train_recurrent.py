"""
RecurrentPPO Training with LSTM Memory.

Uses LSTM for temporal pattern recognition in battery trading.
Implements soft action masking via reward penalty since MaskableRecurrentPPO
is not available in sb3-contrib.
"""
import os
import sys
import argparse
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_recurrent import BatteryEnvRecurrent
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments
from data.upsample_to_15min import upsample_hourly_to_15min


class LSTMStateCallback(BaseCallback):
    """Callback to log LSTM-specific metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if self.num_timesteps % 10000 == 0:
            # Log episode statistics
            if len(self.model.ep_info_buffer) > 0:
                avg_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                avg_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                print(f"Step {self.num_timesteps}: avg_reward={avg_reward:.2f}, avg_ep_len={avg_length:.0f}")
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train_recurrent(timesteps: int = 2_000_000, use_15min: bool = True):
    """
    Train RecurrentPPO agent with LSTM memory.
    
    Args:
        timesteps: Total training timesteps
        use_15min: If True, use 15-minute intervals; else hourly
    """
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_dam.csv"
    time_suffix = "15min" if use_15min else "hourly"
    MODEL_PATH = f"models/ppo_recurrent_{time_suffix}"
    VECNORM_PATH = f"models/vec_normalize_recurrent_{time_suffix}.pkl"
    CHECKPOINT_DIR = f"models/checkpoints/ppo_recurrent_{time_suffix}/"
    LOG_DIR = f"./logs/ppo_recurrent_{time_suffix}/"
    
    # === HYPERPARAMETERS ===
    CHECKPOINT_FREQ = 100_000
    
    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 30.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }
    
    # === LOAD DATA ===
    print(f"Loading data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    original_len = len(df)
    
    print("Regenerating DAM commitments...")
    df = regenerate_dam_commitments(df, battery_params=battery_params)
    
    # Optionally upsample to 15-minute
    time_step_hours = 0.25 if use_15min else 1.0
    if use_15min:
        print("Upsampling to 15-minute resolution...")
        df = upsample_hourly_to_15min(df, method='step', divide_energy_by_4=True)
        print(f"  Hourly: {original_len:,} rows → 15-min: {len(df):,} rows")
    
    # === CREATE ENVIRONMENT ===
    def make_env():
        env = BatteryEnvRecurrent(
            df, 
            battery_params, 
            n_actions=21,
            time_step_hours=time_step_hours,
            invalid_action_penalty=50.0  # Soft masking penalty
        )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=10.)
    
    # === CREATE MODEL ===
    print("\nInitializing RecurrentPPO Agent (LSTM)...")
    
    initial_lr = 3e-4
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=linear_schedule(initial_lr),
        n_steps=2048,          # Steps per rollout
        batch_size=128,        # Must be divisible by sequence length
        n_epochs=10,
        gamma=0.99,            # Discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,         # Exploration
        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=True,   # Share LSTM between actor and critic
            enable_critic_lstm=True,
        ),
        verbose=1,
        tensorboard_log=LOG_DIR
    )
    
    print(f"  Policy: MlpLstmPolicy")
    print(f"  LSTM Hidden Size: 256")
    print(f"  LSTM Layers: 1")
    print(f"  Time Resolution: {time_suffix}")
    
    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_recurrent",
        save_vecnormalize=True
    )
    
    lstm_callback = LSTMStateCallback(verbose=1)
    
    # === TRAIN ===
    print(f"\n{'='*60}")
    print("Starting RecurrentPPO Training (LSTM)")
    print(f"{'='*60}")
    print(f"  Algorithm: RecurrentPPO (PPO + LSTM)")
    print(f"  Resolution: {time_suffix}")
    print(f"  Total Timesteps: {timesteps:,}")
    print(f"  Soft Action Masking: Enabled (penalty=50.0)")
    print(f"  TensorBoard: {LOG_DIR}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=timesteps, 
        callback=[checkpoint_callback, lstm_callback]
    )
    
    # === SAVE ===
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Model: {MODEL_PATH}")
    print(f"  VecNormalize: {VECNORM_PATH}")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RecurrentPPO battery agent")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    parser.add_argument("--hourly", action="store_true", help="Use hourly instead of 15-min")
    
    args = parser.parse_args()
    
    train_recurrent(timesteps=args.timesteps, use_15min=not args.hourly)
