"""
Training v10: Tuned Reward Function
- Lower degradation cost (0.5 vs 10.0)
- Higher DAM penalties (20x + 800 EUR vs 2x + 50 EUR)
- Price timing bonus for buy-low-sell-high
"""
import os
import sys
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agent.callbacks import ProductionMetricsCallback

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments


def mask_fn(env):
    return env.action_masks()


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train_v10():
    # === PATHS (v10) ===
    DATA_PATH = "data/feasible_data_with_balancing.csv"
    MODEL_PATH = "models/ppo_v10_tuned"
    VECNORM_PATH = "models/vec_normalize_v10.pkl"
    CHECKPOINT_DIR = "models/checkpoints/ppo_v10/"
    LOG_DIR = "./logs/ppo_v10/"

    # === HYPERPARAMETERS ===
    # I reduced timesteps for faster iteration - can increase later
    TIMESTEPS = 5_000_000
    CHECKPOINT_FREQ = 200_000

    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
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
    print("\n" + "=" * 60)
    print("TRAINING v10: TUNED REWARD FUNCTION")
    print("=" * 60)
    print("Changes from v9:")
    print("  - deg_cost_per_mwh: 10.0 -> 0.5 (encourage trading)")
    print("  - imbalance_penalty: 2x -> 20x (stricter DAM)")
    print("  - fixed_shortfall: 50 -> 800 EUR (stricter DAM)")
    print("  - NEW: price_timing_bonus for buy-low-sell-high")
    print("=" * 60)

    # I use slightly lower LR for more stable learning with new rewards
    initial_lr = 2e-4  # Reduced from 3e-4
    n_steps = 4096
    batch_size = 256

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=linear_schedule(initial_lr),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # REDUCED from 0.05 - less random exploration
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        verbose=1,
        tensorboard_log=LOG_DIR
    )

    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_v10",
        save_vecnormalize=True
    )

    # Production metrics callback for TensorBoard
    prod_metrics_callback = ProductionMetricsCallback(
        eval_freq=4096,  # Log every rollout (n_steps)
        verbose=1
    )

    callbacks = CallbackList([checkpoint_callback, prod_metrics_callback])

    # === TRAIN ===
    print(f"\nStarting Training v10 with Production Metrics...")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    print(f"  - TensorBoard: {LOG_DIR}")
    print(f"  - Production Metrics: ENABLED (every 4096 steps)")
    print(f"  - Total Steps: {TIMESTEPS:,}")

    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

    # === SAVE ===
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)

    print("=" * 60)
    print("Training v10 Complete!")
    print(f"  Model: {MODEL_PATH}")
    print(f"  VecNormalize: {VECNORM_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_v10()
