"""
Train Realistic DRL Agent - NO DATA LEAKAGE

This trains an agent that:
1. Uses only past price data (no lookahead)
2. Makes decisions based on time patterns and lagged prices
3. Can actually work in production!
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from gym_envs.battery_env_realistic import BatteryEnvRealistic


def mask_fn(env):
    """Return action mask for MaskablePPO."""
    return env.action_masks()


def load_training_data(data_path: str = None) -> pd.DataFrame:
    """Load and prepare training data."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)
    df = df.sort_index()

    # Ensure price column exists
    if 'price' not in df.columns and 'mfrr_price_up' in df.columns:
        df['price'] = df['mfrr_price_up']

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def create_env(df: pd.DataFrame, is_eval: bool = False):
    """Create realistic training environment."""
    env = BatteryEnvRealistic(
        df=df,
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        degradation_cost=15.0,
        time_step_hours=1.0,
        n_actions=21,
        reward_scale=0.001,
    )

    # Wrap with action masker for MaskablePPO
    env = ActionMasker(env, mask_fn)

    return env


def train_realistic_agent(
    total_timesteps: int = 500_000,
    save_path: str = None,
    eval_freq: int = 10_000,
):
    """Train a realistic DRL agent."""

    print("=" * 70)
    print("REALISTIC DRL TRAINING (NO DATA LEAKAGE)")
    print("=" * 70)

    # Load data
    df = load_training_data()

    # Split: 80% train, 20% eval
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    eval_df = df.iloc[split_idx:].copy()

    print(f"\nData Split:")
    print(f"  Train: {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  Eval:  {len(eval_df)} rows ({eval_df.index.min().date()} to {eval_df.index.max().date()})")

    # Create environments
    print("\nCreating environments...")

    def make_train_env():
        return create_env(train_df, is_eval=False)

    def make_eval_env():
        return create_env(eval_df, is_eval=True)

    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Sync normalization stats
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    # Create model
    print("\nCreating MaskablePPO model...")
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs/realistic_ppo/",
    )

    # Callbacks
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/realistic_ppo_{timestamp}"

    Path(save_path).parent.mkdir(exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=save_path,
        name_prefix="realistic_ppo",
    )

    # Train
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    print(f"  Checkpoints: {save_path}")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = f"{save_path}/realistic_ppo_final"
    model.save(final_path)
    train_env.save(f"{save_path}/vec_normalize.pkl")

    print(f"\nModel saved to {final_path}")

    # Quick evaluation
    print("\n" + "=" * 70)
    print("QUICK EVALUATION")
    print("=" * 70)

    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    total_reward = 0
    total_profit = 0
    steps = 0

    for _ in range(1000):
        masks = np.array([eval_env.envs[0].action_masks()])
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total_reward += reward[0]
        if 'total_profit' in info[0]:
            total_profit = info[0]['total_profit']
        steps += 1
        if done[0]:
            break

    print(f"  Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Total Profit: {total_profit:,.0f} EUR")

    return model, train_env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Realistic DRL Agent')
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Total training timesteps')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save model')
    args = parser.parse_args()

    train_realistic_agent(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
    )
