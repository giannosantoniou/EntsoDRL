"""
Training Script for PPO Discrete Intraday Battery Trading

I train MaskablePPO with 5 discrete actions (IDLE, BUY, SELL, aFRR, aFRR+BUY).
DAM is handled by LP Optimizer. Agent focuses on WHEN to trade.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from gym_envs.battery_env_ppo_discrete import BatteryEnvPPODiscrete
from gym_envs.battery_env_unified import BatteryEnvUnified


def mask_fn(env):
    """I return the action mask from the environment."""
    return env.action_masks()


def make_env(df, rank, **kwargs):
    """I create a single environment with action masking."""
    def _init():
        env = BatteryEnvPPODiscrete(df=df, **kwargs)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def train_ppo_discrete(
    data_path: str,
    output_dir: str = "models/ppo_discrete",
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    seed: int = 42,
):
    """I train MaskablePPO on discrete intraday actions."""

    # I load data
    resolved = Path(data_path)
    if not resolved.exists():
        resolved = project_root / data_path
    df = pd.read_csv(resolved, index_col=0, parse_dates=True)
    print(f"Loaded: {len(df)} rows, {df.shape[1]} columns")

    # I split train/eval (80/20 temporal)
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split].copy()
    df_eval = df.iloc[split:].copy()
    print(f"Train: {len(df_train)} rows, Eval: {len(df_eval)} rows")

    # I create common kwargs
    env_kwargs = dict(
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        episode_length=672,
        random_start=True,
        time_step_hours=0.25,     # 15-min resolution! Was missing → defaulted to 1.0h
        enable_endogenous_dam=True,
        dam_bidder_min_spread=5.0,
        mfrr_activation_rate=0.05,
        mfrr_price_cap=1000.0,
    )

    # I create vectorized training environments
    train_env = DummyVecEnv([make_env(df_train, i, **env_kwargs) for i in range(n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # I create eval environment
    eval_env = DummyVecEnv([make_env(df_eval, 0, **{**env_kwargs, 'random_start': False})])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # I setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"ppo_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {run_dir}")
    print(f"Action space: Discrete(5) — IDLE, BUY, SELL, aFRR, aFRR+BUY")
    print(f"Observation: 20 focused features")
    print(f"Training: {total_timesteps:,} steps, {n_envs} envs")

    # I create MaskablePPO model
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,       # Lowered from 3e-4 to prevent catastrophic forgetting
        n_steps=4096,             # Doubled from 2048 for more diverse batches
        batch_size=256,
        n_epochs=5,               # Reduced from 10 to limit overfitting per batch
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.15,          # Tighter from 0.2 for more conservative updates
        ent_coef=0.02,            # Doubled from 0.01 for sustained exploration
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=seed,
    )

    # I setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(50000 // n_envs, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(200000 // n_envs, 5000),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_discrete",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # I train
    print(f"\nStarting MaskablePPO training...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)

    # I save final model
    model.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "vecnormalize.pkl"))
    print(f"\nTraining complete! Model saved to {run_dir}")

    return str(run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO Discrete Intraday")
    parser.add_argument("--data", type=str,
                        default="data/unified_multimarket_training_v5_raw_features.csv")
    parser.add_argument("--output", type=str, default="models/ppo_discrete")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_ppo_discrete(
        data_path=args.data,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
    )
