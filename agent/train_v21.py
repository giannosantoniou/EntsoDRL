"""
v21 Training: Fix Data Leakage + Trader-Inspired Signals

Key fixes vs v20:
1. _precompute_24h_stats() replaced with _precompute_backward_stats() (PAST-ONLY)
2. Feature 43: price_vs_typical_hour (current vs 30-day mean at this hour)
3. Feature 44: trade_worthiness (30-day avg daily spread, normalized)

The v20 features (price_percentile, price_vs_mean) used FUTURE min/max/mean prices
in the 24h window — classic data leakage. The agent appeared smart in training
but would fail in production where future prices are unknown.

v21 features use ONLY past data:
- price_vs_typical_hour: "Is this price unusual for this hour?" (past 30 days)
- trade_worthiness: "Is this a volatile (profitable) period?" (past 30 days)

Reward config: SAME AS v15/v20 (proven baseline, no quartile penalties).
ML forecaster: mandatory for price lookahead (no raw DataFrame peek).
"""
import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import torch

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data

# I set the number of parallel environments for CPU-bound speedup
# Each env loads its own forecaster + DataFrame copy, so memory scales linearly.
# With 52 cores and ~100MB per env, 24 envs should work well.
N_ENVS = 24


class ValidationCallback(BaseCallback):
    """
    I run validation every N steps and track best model.
    I also save VecNormalize stats alongside the model and sync
    normalization stats from the training env to the validation env
    before each validation run.
    Early stopping if no improvement for 'patience' validations.
    """
    def __init__(self, val_env, train_env=None, eval_freq=50000, patience=15, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_score = float('-inf')
        self.no_improve_count = 0
        self.validation_scores = []

    def _sync_normalization(self):
        """I sync VecNormalize stats from train_env to val_env so
        the validation observations are on the same scale as training."""
        if self.train_env is not None:
            self.val_env.obs_rms = self.train_env.obs_rms
            self.val_env.ret_rms = self.train_env.ret_rms

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # I sync normalization stats before each validation
            self._sync_normalization()
            score = self._validate()
            self.validation_scores.append(score)

            if score > self.best_score:
                self.best_score = score
                self.no_improve_count = 0
                self.model.save("models/ppo_v21_best")
                # I save VecNormalize stats alongside the model
                if self.train_env is not None:
                    self.train_env.save("models/vec_normalize_v21.pkl")
                if self.verbose:
                    print(f"  [VAL] New best: {score:.2f} - Model + VecNormalize saved!")
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
    print("v21 TRAINING: Fix Data Leakage + Trader-Inspired Signals")
    print("=" * 70)

    # I check CUDA availability for GPU training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # I load historical data with balancing prices
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

    # v21 Reward config: SAME AS v15/v20 (proven baseline, no quartile penalties!)
    reward_config = {
        'deg_cost_per_mwh': 0.5,
        'violation_penalty': 5.0,
        'imbalance_penalty_multiplier': 25.0,
        'fixed_shortfall_penalty': 800.0,
        'proactive_risk_penalty': 50.0,
        'soc_buffer_bonus': 2.0,
        'min_soc_threshold': 0.25,
        'price_timing_bonus': 0.1,
        # v13 timing
        'peak_sell_bonus': 10.0,
        'early_sell_penalty': 15.0,
        'early_sell_threshold': 1.3,
        # ALL quartile penalties DISABLED (v15 baseline)
        'quartile_sell_cheap_penalty': 0.0,
        'quartile_buy_expensive_penalty': 0.0,
        'quartile_good_trade_bonus': 0.0,
        'use_multiplicative_quartile': False,
        'use_charge_cost_scaling': False,
    }

    # I verify ML forecaster exists before training
    forecaster_path = "models/intraday_forecaster.pkl"
    if not os.path.exists(forecaster_path):
        print(f"\n  WARNING: ML forecaster not found at {forecaster_path}")
        print(f"  Training will fall back to noisy DataFrame peek for price lookahead.")
        print(f"  For best results, train the forecaster first:")
        print(f"    python models/intraday_forecaster.py")
        use_ml = False
    else:
        use_ml = True
        print(f"\n  ML forecaster: {forecaster_path}")

    # I use a factory that returns a closure — required for SubprocVecEnv
    # Each subprocess gets its own copy of the environment
    def make_train_env(rank):
        def _init():
            env = BatteryEnvMasked(
                train_df, battery_params, n_actions=21,
                use_ml_forecaster=use_ml,
                forecaster_path=forecaster_path,
                reward_config=reward_config,
                include_price_awareness=True  # v21 uses trader-inspired features
            )
            env = ActionMasker(env, lambda e: e.action_masks())
            return env
        return _init

    # Validation environment (single, DummyVecEnv is fine)
    def make_val_env():
        env = BatteryEnvMasked(
            val_df, battery_params, n_actions=21,
            use_ml_forecaster=use_ml,
            forecaster_path=forecaster_path,
            reward_config=reward_config,
            include_price_awareness=True
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    # I check for an existing checkpoint to resume training
    checkpoint_path = "models/ppo_v21_best.zip"
    vecnorm_path = "models/vec_normalize_v21.pkl"
    resume = os.path.exists(checkpoint_path)

    # I use SubprocVecEnv for N_ENVS parallel environments — each runs in its own process
    # This is the key speedup: CPU-bound env stepping happens in parallel
    print(f"\nCreating {N_ENVS} parallel training environments (SubprocVecEnv)...")
    train_env = SubprocVecEnv([make_train_env(i) for i in range(N_ENVS)])

    if resume and os.path.exists(vecnorm_path):
        # I load saved VecNormalize stats for seamless resume
        print(f"  Loading VecNormalize stats from {vecnorm_path}")
        train_env = VecNormalize.load(vecnorm_path, train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        # I increased clip_reward from 10 to 100 — with 24 envs the reward variance
        # was being compressed to near-zero, causing value_loss collapse
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=100.0, clip_reward=100.0)

    val_env = DummyVecEnv([make_val_env])
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False)
    val_env.training = False
    val_env.norm_reward = False

    if resume:
        # I load the best checkpoint and continue training
        print(f"\n  Resuming from checkpoint: {checkpoint_path}")
        model = MaskablePPO.load(
            checkpoint_path,
            env=train_env,
            device=device
        )
        print(f"  Model loaded successfully.")
        if not os.path.exists(vecnorm_path):
            # I skip explicit warm-up — VecNormalize learns stats during the first
            # few training iterations. PPO is robust to non-stationary obs distributions.
            print(f"  No VecNormalize stats found — stats will converge during first iterations.")
    else:
        # I create a fresh model — same architecture as v15/v20
        # n_steps is per-env, so total batch = n_steps × N_ENVS = 2048 × 24 = 49152
        print("\nCreating MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=1024,  # I scale with N_ENVS: rollout=2048×24=49152, ~48 minibatches
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
    # I pass train_env so the callback can save VecNormalize stats and sync to val_env
    # I reduced eval_freq from 50000 to 10000 so we get validation every ~5 min
    # instead of every ~2 hours. Increased patience to 30 to compensate.
    val_callback = ValidationCallback(
        val_env,
        train_env=train_env,
        eval_freq=10000,
        patience=30,
        verbose=1
    )

    # Training
    total_timesteps = 5_000_000
    print(f"\nStarting training: {total_timesteps:,} timesteps")
    if resume:
        print(f"  (Resuming from checkpoint — model weights preserved)")
    print("v21 key changes (Data Leakage Fix):")
    print(f"  - FIXED: _precompute_backward_stats() replaces forward-looking stats")
    print(f"  - Feature 43: price_vs_typical_hour (current vs 30-day hourly mean)")
    print(f"  - Feature 44: trade_worthiness (30-day avg daily spread)")
    print(f"  - Observation space: 44 features (same size as v20)")
    print(f"  - Reward config: SAME AS v15/v20 (no quartile penalties)")
    print(f"  - ML forecaster: {'ENABLED' if use_ml else 'DISABLED (fallback to noisy peek)'}")
    print("-" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=val_callback,
        progress_bar=True
    )

    # I save final artifacts
    train_env.save("models/vec_normalize_v21.pkl")
    model.save("models/ppo_v21_final")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation score: {val_callback.best_score:.2f}")
    print("Models saved:")
    print("  - models/ppo_v21_best.zip (best validation)")
    print("  - models/ppo_v21_final.zip (final)")
    print("  - models/vec_normalize_v21.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
