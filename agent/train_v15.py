"""
v15 Training: Realistic Imbalance Settlement with Synthetic Long/Short Spread

Key improvements over v14:
1. Synthetic Long/Short spread based on mFRR (30% of distance to activation prices)
2. ML price forecaster for realistic price predictions
3. Proper imbalance penalties with meaningful spread

The model will learn to:
- Minimize imbalance (bid/ask spread is costly)
- Time trades better to avoid spread costs
- Balance DAM compliance with IntraDay opportunities
"""
import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import torch

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data


class ValidationCallback(BaseCallback):
    """
    I run validation every N steps and track best model.
    Early stopping if no improvement for 'patience' validations.
    """
    def __init__(self, val_env, eval_freq=50000, patience=8, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_score = float('-inf')
        self.no_improve_count = 0
        self.validation_scores = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            score = self._validate()
            self.validation_scores.append(score)

            if score > self.best_score:
                self.best_score = score
                self.no_improve_count = 0
                self.model.save("models/ppo_v15_best")
                if self.verbose:
                    print(f"  [VAL] New best: {score:.2f} - Model saved!")
            else:
                self.no_improve_count += 1
                if self.verbose:
                    print(f"  [VAL] Score: {score:.2f} (best: {self.best_score:.2f}, no improve: {self.no_improve_count}/{self.patience})")

            if self.no_improve_count >= self.patience:
                print(f"  [EARLY STOP] No improvement for {self.patience} validations. Stopping.")
                return False

        return True

    def _validate(self, n_episodes=3):
        """Run validation episodes and return mean reward."""
        total_rewards = []
        for _ in range(n_episodes):
            obs = self.val_env.reset()
            episode_reward = 0
            done = False
            steps = 0
            max_steps = 2000

            while not done and steps < max_steps:
                inner_env = self.val_env.envs[0]
                while hasattr(inner_env, 'env'):
                    if hasattr(inner_env, 'action_masks'):
                        break
                    inner_env = inner_env.env
                mask = inner_env.action_masks() if hasattr(inner_env, 'action_masks') else None

                action, _ = self.model.predict(
                    obs, deterministic=True,
                    action_masks=np.array([mask]) if mask is not None else None
                )
                obs, reward, done, info = self.val_env.step(action)
                episode_reward += reward[0]
                steps += 1
                done = done[0]

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)


def main():
    print("=" * 70)
    print("v15 TRAINING: Realistic Imbalance Settlement")
    print("=" * 70)

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    df = load_historical_data('data/feasible_data_with_balancing.csv')
    print(f"Total data: {len(df):,} rows")

    # Split: 70% train, 15% val, 15% test
    train_end = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)

    print(f"Train: {len(train_df):,} rows")
    print(f"Val: {len(val_df):,} rows")
    print(f"Test: {len(df) - val_end:,} rows (held out)")

    battery_params = {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94
    }

    # Reward config for v15: Focus on spread-aware trading
    reward_config = {
        'deg_cost_per_mwh': 0.5,
        'violation_penalty': 5.0,
        'imbalance_penalty_multiplier': 25.0,  # Increased (spread matters more now!)
        'fixed_shortfall_penalty': 800.0,
        'proactive_risk_penalty': 50.0,
        'soc_buffer_bonus': 2.0,
        'min_soc_threshold': 0.25,
        'price_timing_bonus': 0.1,
        'peak_sell_bonus': 10.0,
        'early_sell_penalty': 15.0,
        'early_sell_threshold': 1.3
    }

    # Training environment
    def make_train_env():
        env = BatteryEnvMasked(
            train_df, battery_params, n_actions=21,
            use_ml_forecaster=True,
            forecaster_path="models/intraday_forecaster.pkl",
            reward_config=reward_config
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    # Validation environment
    def make_val_env():
        env = BatteryEnvMasked(
            val_df, battery_params, n_actions=21,
            use_ml_forecaster=True,
            forecaster_path="models/intraday_forecaster.pkl",
            reward_config=reward_config
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=10.0)

    val_env = DummyVecEnv([make_val_env])
    val_env = VecNormalize.load("models/vec_normalize_v15.pkl", val_env) if os.path.exists("models/vec_normalize_v15.pkl") else VecNormalize(val_env, norm_obs=True, norm_reward=False)
    val_env.training = False
    val_env.norm_reward = False

    # Model
    print("\nCreating MaskablePPO model...")
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256, 128]},
        verbose=1,
        device=device
    )

    # Callbacks
    val_callback = ValidationCallback(
        val_env,
        eval_freq=50000,
        patience=8,  # More patience for spread learning
        verbose=1
    )

    # Training
    total_timesteps = 5_000_000
    print(f"\nStarting training: {total_timesteps:,} timesteps")
    print("Key changes in v15:")
    print("  - Synthetic Long/Short spread (30% of mFRR distance)")
    print("  - ML price forecaster")
    print("  - Higher imbalance penalty (25x)")
    print("-" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=val_callback,
        progress_bar=True
    )

    # Save final artifacts
    train_env.save("models/vec_normalize_v15.pkl")
    model.save("models/ppo_v15_final")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation score: {val_callback.best_score:.2f}")
    print("Models saved:")
    print("  - models/ppo_v15_best.zip (best validation)")
    print("  - models/ppo_v15_final.zip (final)")
    print("  - models/vec_normalize_v15.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
