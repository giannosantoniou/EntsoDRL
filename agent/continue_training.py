"""
Continue Training from Checkpoint
Loads the latest checkpoint and resumes training.
"""
import os
import sys
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments


def mask_fn(env):
    """Get action mask from environment."""
    return env.action_masks()


def find_latest_checkpoint(checkpoint_dir: str, prefix: str = "ppo_masked_imbal") -> tuple:
    """
    I find the latest checkpoint by parsing step numbers from filenames.
    Returns (model_path, vecnormalize_path, steps).
    """
    if not os.path.exists(checkpoint_dir):
        return None, None, 0

    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith(prefix) and f.endswith('_steps.zip')]

    if not checkpoints:
        return None, None, 0

    # I extract step numbers and find the maximum
    latest_steps = 0
    latest_file = None
    for ckpt in checkpoints:
        try:
            # Format: ppo_masked_imbal_1000000_steps.zip
            parts = ckpt.replace('.zip', '').split('_')
            steps = int(parts[-2])
            if steps > latest_steps:
                latest_steps = steps
                latest_file = ckpt
        except (ValueError, IndexError):
            continue

    if latest_file is None:
        return None, None, 0

    model_path = os.path.join(checkpoint_dir, latest_file)
    vecnorm_file = latest_file.replace('_steps.zip', '_vecnormalize_steps.pkl').replace(
        f'{prefix}_', f'{prefix}_vecnormalize_'
    )
    # I handle the actual naming pattern: ppo_masked_imbal_vecnormalize_1000000_steps.pkl
    vecnorm_file = f"{prefix}_vecnormalize_{latest_steps}_steps.pkl"
    vecnorm_path = os.path.join(checkpoint_dir, vecnorm_file)

    if not os.path.exists(vecnorm_path):
        print(f"Warning: VecNormalize not found at {vecnorm_path}")
        vecnorm_path = None

    return model_path, vecnorm_path, latest_steps


def continue_training():
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_balancing.csv"
    MODEL_PATH = "models/ppo_12h_lookahead_v9"
    VECNORM_PATH = "models/vec_normalize_12h_v9.pkl"
    CHECKPOINT_DIR = "models/checkpoints/ppo_12h_v9/"
    LOG_DIR = "./logs/ppo_12h_v9/"

    # === HYPERPARAMETERS ===
    TOTAL_TIMESTEPS = 10_000_000
    CHECKPOINT_FREQ = 200_000

    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }

    # === FIND LATEST CHECKPOINT ===
    print("Searching for latest checkpoint...")
    model_path, vecnorm_path, completed_steps = find_latest_checkpoint(CHECKPOINT_DIR)

    if model_path is None:
        print("ERROR: No checkpoint found! Run train_masked.py first.")
        return

    remaining_steps = TOTAL_TIMESTEPS - completed_steps

    print(f"=" * 50)
    print(f"Found checkpoint:")
    print(f"  Model: {model_path}")
    print(f"  VecNormalize: {vecnorm_path}")
    print(f"  Completed: {completed_steps:,} steps")
    print(f"  Remaining: {remaining_steps:,} steps")
    print(f"=" * 50)

    if remaining_steps <= 0:
        print("Training already complete!")
        return

    # === LOAD DATA ===
    print(f"\nLoading data: {DATA_PATH}")
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

    # I load the VecNormalize stats from checkpoint (critical for continuity!)
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from checkpoint: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True  # I ensure we continue updating stats
        env.norm_reward = True
    else:
        print("WARNING: Creating new VecNormalize (stats will be reset!)")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=10.)

    # === LOAD MODEL ===
    print(f"\nLoading model from checkpoint: {model_path}")
    model = MaskablePPO.load(model_path, env=env, tensorboard_log=LOG_DIR)

    # I reset the number of timesteps so learning rate schedule continues correctly
    model.num_timesteps = completed_steps
    model._num_timesteps_at_start = completed_steps

    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_masked_imbal",
        save_vecnormalize=True
    )

    # === CONTINUE TRAINING ===
    print(f"\nResuming Training from {completed_steps:,} steps...")
    print(f"  - Target: {TOTAL_TIMESTEPS:,} total steps")
    print(f"  - Remaining: {remaining_steps:,} steps")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    print(f"  - TensorBoard: {LOG_DIR}")

    model.learn(
        total_timesteps=remaining_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=False  # I continue from current timestep count
    )

    # === SAVE FINAL MODEL ===
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)

    print("=" * 50)
    print("Training Complete!")
    print(f"  Final Model: {MODEL_PATH}")
    print(f"  VecNormalize: {VECNORM_PATH}")
    print(f"  Total Steps: {model.num_timesteps:,}")
    print("=" * 50)


if __name__ == "__main__":
    continue_training()
