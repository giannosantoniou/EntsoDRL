"""
Training v12: Reduced Penalties for Profitability
- v11 learned correct price-following behavior (+29 EUR spread)
- But still losing money due to high penalties
- v12: I reduce penalties to see if agent can become profitable

Changes from v11:
- deg_cost_per_mwh: 0.5 -> 0.2 (less cycling penalty)
- fixed_shortfall_penalty: 800 -> 300 (less harsh DAM penalty)
- proactive_risk_penalty: 50 -> 20 (allow more risk-taking)
- Keep same network architecture [256, 128] and entropy 0.05
"""
import os
import sys
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from agent.callbacks import ProductionMetricsCallback


def mask_fn(env):
    return env.action_masks()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class EarlyStoppingCallback(BaseCallback):
    """
    I stop training if validation performance doesn't improve for N evaluations.
    """
    def __init__(
        self,
        eval_env,
        eval_freq: int = 50000,
        patience: int = 5,
        min_improvement: float = 0.01,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_improvement = min_improvement

        self.best_score = -float('inf')
        self.no_improvement_count = 0
        self.best_model_path = "models/ppo_v12_best.zip"
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        score = self._evaluate()
        self.eval_results.append((self.num_timesteps, score))

        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.no_improvement_count = 0
            self.model.save(self.best_model_path)
            if self.verbose > 0:
                print(f"[EarlyStopping] New best score: {score:.2f} - Model saved!")
        else:
            self.no_improvement_count += 1
            if self.verbose > 0:
                print(f"[EarlyStopping] No improvement ({self.no_improvement_count}/{self.patience})")

        if self.no_improvement_count >= self.patience:
            if self.verbose > 0:
                print(f"[EarlyStopping] Stopping training - no improvement for {self.patience} evaluations")
            return False

        return True

    def _evaluate(self) -> float:
        """
        I evaluate the model on validation data.
        v12: I weight profit more heavily since we want profitability.
        """
        obs = self.eval_env.reset()
        done = False

        total_profit = 0
        dam_met = 0
        dam_total = 0
        steps = 0

        while not done and steps < 5000:
            env = self.eval_env.envs[0]
            while hasattr(env, 'env'):
                if hasattr(env, 'action_masks'):
                    break
                env = env.env
            mask = env.action_masks() if hasattr(env, 'action_masks') else None

            action, _ = self.model.predict(obs, deterministic=True, action_masks=np.array([mask]) if mask is not None else None)
            obs, reward, done, info = self.eval_env.step(action)

            info_dict = info[0]
            profit = info_dict.get('net_profit', 0)
            power = info_dict.get('actual_mw', 0)
            dam_commitment = info_dict.get('dam_commitment', 0)

            total_profit += profit

            if abs(dam_commitment) > 0.1:
                dam_total += 1
                if (dam_commitment > 0 and power > 0) or (dam_commitment < 0 and power < 0):
                    dam_met += 1

            steps += 1

        dam_compliance = dam_met / dam_total if dam_total > 0 else 0
        profit_score = total_profit / 5000  # Normalize by steps

        # v12: I weight profit more (50%) since that's our goal now
        score = 0.3 * dam_compliance * 100 + 0.5 * profit_score + 0.2 * 50  # Base score

        if self.verbose > 0:
            print(f"[Validation] DAM: {dam_compliance*100:.1f}% | Profit: {total_profit:.0f} | Score: {score:.2f}")

        return score


def train_v12():
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_balancing.csv"
    MODEL_PATH = "models/ppo_v12_final"
    VECNORM_PATH = "models/vec_normalize_v12.pkl"
    CHECKPOINT_DIR = "models/checkpoints/ppo_v12/"
    LOG_DIR = "./logs/ppo_v12/"

    # === HYPERPARAMETERS ===
    TIMESTEPS = 5_000_000
    CHECKPOINT_FREQ = 200_000

    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }

    # === v12 REWARD CONFIG: REDUCED PENALTIES ===
    # I reduce penalties to allow agent to be more profitable
    reward_config = {
        'deg_cost_per_mwh': 0.2,           # REDUCED from 0.5 - allow more cycling
        'violation_penalty': 5.0,           # Same
        'imbalance_penalty_multiplier': 15.0,  # REDUCED from 20.0
        'fixed_shortfall_penalty': 300.0,   # REDUCED from 800.0 - less harsh
        'proactive_risk_penalty': 20.0,     # REDUCED from 50.0 - allow risk
        'soc_buffer_bonus': 2.0,            # Same
        'min_soc_threshold': 0.25,          # Same
        'price_timing_bonus': 0.15          # INCREASED from 0.1 - reward good trades
    }

    # === LOAD DATA ===
    print(f"Loading data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)

    # === TRAIN/VALIDATION SPLIT (80/20) ===
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"Train data: {len(train_df):,} hours ({len(train_df)/24:.0f} days)")
    print(f"Validation data: {len(val_df):,} hours ({len(val_df)/24:.0f} days)")

    # === CREATE TRAINING ENVIRONMENT ===
    def make_train_env():
        env = BatteryEnvMasked(train_df, battery_params, n_actions=21, reward_config=reward_config)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=10.)

    # === CREATE VALIDATION ENVIRONMENT ===
    def make_val_env():
        env = BatteryEnvMasked(val_df, battery_params, n_actions=21, reward_config=reward_config)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    val_env = DummyVecEnv([make_val_env])
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, training=False)

    # === CREATE MODEL ===
    print("\n" + "=" * 60)
    print("TRAINING v12: REDUCED PENALTIES FOR PROFITABILITY")
    print("=" * 60)
    print("Changes from v11:")
    print("  - deg_cost_per_mwh: 0.5 -> 0.2 (less cycling penalty)")
    print("  - fixed_shortfall_penalty: 800 -> 300 (less harsh)")
    print("  - proactive_risk_penalty: 50 -> 20 (allow risk)")
    print("  - price_timing_bonus: 0.1 -> 0.15 (reward good trades)")
    print("  - Same network [256, 128], entropy 0.05")
    print("=" * 60)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        train_env,
        learning_rate=linear_schedule(2e-4),
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # Same as v11
        policy_kwargs=dict(
            net_arch=[256, 128],  # Same as v11
        ),
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_v12",
        save_vecnormalize=True
    )

    def sync_val_env():
        val_env.obs_rms = train_env.obs_rms
        val_env.ret_rms = train_env.ret_rms

    class SyncNormCallback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 10000 == 0:
                sync_val_env()
            return True

    sync_callback = SyncNormCallback()

    early_stopping_callback = EarlyStoppingCallback(
        eval_env=val_env,
        eval_freq=100000,
        patience=5,
        verbose=1
    )

    prod_metrics_callback = ProductionMetricsCallback(
        eval_freq=4096,
        verbose=0
    )

    callbacks = CallbackList([
        checkpoint_callback,
        sync_callback,
        early_stopping_callback,
        prod_metrics_callback
    ])

    # === TRAIN ===
    print(f"\nStarting Training v12...")
    print(f"  - Reward Config: {reward_config}")
    print(f"  - Early Stopping: patience=5")
    print(f"  - Max Steps: {TIMESTEPS:,}")

    try:
        model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # === SAVE FINAL MODEL ===
    model.save(MODEL_PATH)
    train_env.save(VECNORM_PATH)

    # === REPORT ===
    print("\n" + "=" * 60)
    print("Training v12 Complete!")
    print(f"  Final Model: {MODEL_PATH}")
    print(f"  Best Model: models/ppo_v12_best.zip")
    print(f"  VecNormalize: {VECNORM_PATH}")

    if early_stopping_callback.eval_results:
        print(f"\n  Validation History:")
        for steps, score in early_stopping_callback.eval_results[-5:]:
            print(f"    {steps:>8,} steps: score {score:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    train_v12()
