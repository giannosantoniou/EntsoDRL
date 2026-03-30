"""
v23 Training: Production-Ready (No Oracle Features, Randomized Starts)

Key fixes vs v22:
1. REMOVED oracle timing features 23-26 (replaced with backward-looking alternatives)
2. NEW: Randomized episode start positions (prevents memorization/overfitting)
3. NEW: Randomized initial SoC (±10% for robustness)
4. PRESERVED: All v22 HEnEx compliance (gate closure, fixed penalties, DAM boundaries)

Oracle features removed:
  - price_ratio (used max of future prices — ORACLE)
  - hours_to_peak (revealed exact peak timing — ORACLE)
  - is_at_peak (compared current to future max — ORACLE)
  - dam_slot_ratio (compared current to future DAM slot — ORACLE)

Replaced with backward-looking alternatives:
  - price_vs_24h_range: current price position in past 24h range [0,1]
  - price_momentum_6h: 6-hour backward price momentum [-1,1]
  - is_near_daily_high: binary, current >= 95% of past 24h max
  - price_vs_mean_24h: current vs past 24h mean [-1,1]

These features give the agent the same INTENT (timing signals) but using
ONLY past data — exactly what a human trader would have access to.
"""
import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data

# I use 10 parallel envs for better sample diversity and faster collection
N_ENVS = 10
MODEL_DIR = "models/ppo_v23_production"
VECNORM_PATH = f"{MODEL_DIR}/vec_normalize.pkl"
BEST_MODEL_PATH = f"{MODEL_DIR}/best_model"
FINAL_MODEL_PATH = f"{MODEL_DIR}/final_model"
CHECKPOINT_DIR = f"{MODEL_DIR}/checkpoints"


class ValidationCallback(BaseCallback):
    """
    I run validation every N steps and track the best model.
    I sync VecNormalize stats from training to validation before each run.
    Early stopping if no improvement for 'patience' validations.
    """
    def __init__(self, val_env, train_env=None, eval_freq=50000, patience=20, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.train_env = train_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_score = float('-inf')
        self.no_improve_count = 0
        self.validation_scores = []

    def _sync_normalization(self):
        """I sync VecNormalize stats from train_env to val_env."""
        if self.train_env is not None:
            self.val_env.obs_rms = self.train_env.obs_rms
            self.val_env.ret_rms = self.train_env.ret_rms

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._sync_normalization()
            score = self._validate()
            self.validation_scores.append(score)

            if score > self.best_score:
                self.best_score = score
                self.no_improve_count = 0
                self.model.save(BEST_MODEL_PATH)
                if self.train_env is not None:
                    self.train_env.save(VECNORM_PATH)
                if self.verbose:
                    print(f"  [VAL] New best: {score:.2f} — Model + VecNormalize saved!")
            else:
                self.no_improve_count += 1
                if self.verbose:
                    print(f"  [VAL] Score: {score:.2f} (best: {self.best_score:.2f}, "
                          f"no improve: {self.no_improve_count}/{self.patience})")

            if self.no_improve_count >= self.patience:
                print(f"  [EARLY STOP] No improvement for {self.patience} validations.")
                return False

        return True

    def _validate(self, n_episodes=3):
        """I run validation episodes and return mean reward."""
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
    print("v23 TRAINING: Production-Ready (No Oracle, Randomized Starts)")
    print("=" * 70)

    # I check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # I create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # I load 15-min resolution data (upsampled from hourly where needed)
    df = load_historical_data('data/feasible_data_with_dam_15min.csv')
    print(f"Total data: {len(df):,} rows")

    # I verify the data is uniform 15-min resolution
    time_step_hours = 0.25  # 15-minute intervals
    print(f"Time step: {time_step_hours}h (15-min resolution)")

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

    # v22 reward config preserved (no changes needed — leakage was in features, not rewards)
    reward_config = {
        'deg_cost_per_mwh': 0.5,
        'violation_penalty': 5.0,
        'commitment_violation_penalty': 1500.0,
        'proactive_risk_penalty': 50.0,
        'soc_buffer_bonus': 2.0,
        'min_soc_threshold': 0.25,
        'price_timing_bonus': 0.1,
        'momentum_bonus': 8.0,
        'momentum_threshold': 0.05,
        'quartile_sell_cheap_penalty': 0.0,
        'quartile_buy_expensive_penalty': 0.0,
        'quartile_good_trade_bonus': 0.0,
        'use_multiplicative_quartile': False,
        'use_charge_cost_scaling': False,
    }

    # I check ML forecaster availability
    forecaster_path = "models/intraday_forecaster.pkl"
    use_ml = os.path.exists(forecaster_path)
    if use_ml:
        print(f"  ML forecaster: {forecaster_path}")
    else:
        print(f"  ML forecaster not found — using noisy peek fallback")

    # I create training envs with randomize_start=True (v23 key change!)
    def make_train_env(rank):
        def _init():
            env = BatteryEnvMasked(
                train_df, battery_params, n_actions=21,
                time_step_hours=time_step_hours,  # 15-min resolution
                use_ml_forecaster=use_ml,
                forecaster_path=forecaster_path,
                reward_config=reward_config,
                include_price_awareness=True,
                gate_closure_hours=1.0,
                enable_gate_closure=True,
                # v23: Randomized starts for generalization
                randomize_start=True,
                min_episode_length=672,  # 168h × 4 steps/h = 1 week in 15-min steps
            )
            env = ActionMasker(env, lambda e: e.action_masks())
            return env
        return _init

    # I create validation env with randomize_start=False for deterministic evaluation
    def make_val_env():
        env = BatteryEnvMasked(
            val_df, battery_params, n_actions=21,
            time_step_hours=time_step_hours,  # 15-min resolution
            use_ml_forecaster=use_ml,
            forecaster_path=forecaster_path,
            reward_config=reward_config,
            include_price_awareness=True,
            gate_closure_hours=1.0,
            enable_gate_closure=True,
            randomize_start=False,  # Deterministic for reproducible validation
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    # I create parallel training environments
    print(f"\nCreating {N_ENVS} parallel training environments (SubprocVecEnv)...")
    train_env = SubprocVecEnv([make_train_env(i) for i in range(N_ENVS)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True,
        clip_obs=100.0, clip_reward=100.0
    )

    val_env = DummyVecEnv([make_val_env])
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False)
    val_env.training = False
    val_env.norm_reward = False

    # I create the MaskablePPO model
    print("\nCreating MaskablePPO model...")
    # I tuned hyperparameters based on v23 run #1 diagnostics:
    # - ent_coef 0.01→0.05: Entropy collapsed from 2.9 to 0.9 in 2M steps.
    #   Higher entropy bonus keeps exploration alive longer.
    # - n_steps 2048→4096: With 10 envs, buffer = 40,960 transitions per rollout.
    #   Larger buffer captures more diverse market regimes per update.
    # - n_epochs 10→5: Fewer passes over each buffer reduces overfitting per rollout
    #   and prevents the high KL divergence (was 0.06, target <0.03).
    # - batch_size 256→512: Larger batches give more stable gradient estimates,
    #   reducing the clip fraction (was 0.32, target <0.25).
    # - learning_rate 3e-4→2e-4 with linear decay: Gentler updates reduce
    #   the aggressive policy changes that caused high KL and clip warnings.
    # - clip_range 0.2→0.25: Slightly wider clip allows healthy policy updates
    #   without triggering excessive clipping at the 0.2 boundary.
    # - net_arch [256,256,128]→[512,256,128]: Wider first layer for better
    #   market regime capture, same as v22 proven architecture.
    def linear_schedule(initial_value):
        def func(progress_remaining):
            return progress_remaining * initial_value
        return func

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=linear_schedule(1.5e-4),  # I reduced from 2e-4 for gentler updates (lower KL)
        n_steps=4096,
        batch_size=512,
        n_epochs=4,           # I reduced from 5 — fewer passes = less policy drift per rollout
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=0.08,        # I raised from 0.05 — more exploration, slower entropy decay
        target_kl=0.03,       # I added — SB3 stops epoch early if KL exceeds this
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [512, 256, 128]},
        verbose=1,
        device=device,
        tensorboard_log=f"{MODEL_DIR}/logs/"
    )

    # I set up callbacks
    val_callback = ValidationCallback(
        val_env,
        train_env=train_env,
        eval_freq=10000,
        patience=30,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_v23"
    )

    # Training
    total_timesteps = 3_000_000
    print(f"\nStarting training: {total_timesteps:,} timesteps")
    print("v23 key changes (Production-Ready):")
    print(f"  - REMOVED: Oracle features 23-26 (price_ratio, hours_to_peak, is_at_peak, dam_slot_ratio)")
    print(f"  - ADDED: Backward alternatives (price_vs_24h_range, momentum_6h, is_near_daily_high, price_vs_mean)")
    print(f"  - ADDED: Randomized episode starts (min_episode_length=168)")
    print(f"  - ADDED: Randomized initial SoC (±10%)")
    print(f"  - PRESERVED: v22 HEnEx compliance (gate closure, fixed penalties)")
    print(f"  - Observation space: 45 features (same count, different content)")
    print(f"  - ML forecaster: {'ENABLED' if use_ml else 'DISABLED'}")
    print("-" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[val_callback, checkpoint_callback],
        progress_bar=False
    )

    # I save final artifacts
    train_env.save(VECNORM_PATH)
    model.save(FINAL_MODEL_PATH)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best validation score: {val_callback.best_score:.2f}")
    print("Models saved:")
    print(f"  - {BEST_MODEL_PATH}.zip (best validation)")
    print(f"  - {FINAL_MODEL_PATH}.zip (final)")
    print(f"  - {VECNORM_PATH}")
    print(f"  - {CHECKPOINT_DIR}/ (checkpoints every 200K steps)")
    print("=" * 70)


if __name__ == "__main__":
    main()
