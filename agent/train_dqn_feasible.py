"""
DQN Training with Feasible DAM Commitments
Uses the FeasibleDAMGenerator for realistic training data.
"""
import os
import sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_discrete import BatteryTradingEnvDiscrete
from data.data_loader import load_historical_data
from data.dam_commitment_generator import FeasibleDAMGenerator, regenerate_dam_commitments


def train_dqn_feasible():
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/dqn_feasible"
    VECNORM_PATH = "models/vec_normalize_dqn_feasible.pkl"
    CHECKPOINT_DIR = "models/checkpoints/dqn_feasible/"
    
    # === HYPERPARAMETERS ===
    TIMESTEPS = 5_000_000
    CHECKPOINT_FREQ = 200_000
    
    # === BATTERY PARAMS ===
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    # === LOAD DATA WITH FEASIBLE DAM COMMITMENTS ===
    print(f"Loading data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    print("Regenerating DAM commitments with FeasibleDAMGenerator...")
    df = regenerate_dam_commitments(df, battery_params=battery_params)
    
    print(f"DAM Commitment Stats:")
    print(f"  Range: [{df['dam_commitment'].min():.1f}, {df['dam_commitment'].max():.1f}] MW")
    print(f"  Discharging: {(df['dam_commitment'] > 0).mean()*100:.1f}%")
    print(f"  Charging: {(df['dam_commitment'] < 0).mean()*100:.1f}%")
    
    # === CREATE ENVIRONMENT ===
    def make_env():
        e = BatteryTradingEnvDiscrete(df, battery_params, n_actions=21)
        e = Monitor(e)
        return e
        
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=10.)
    
    # === CREATE MODEL ===
    print("\nInitializing DQN Agent...")
    policy_kwargs = dict(net_arch=[256, 256, 256])
    
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-4, 
        buffer_size=100_000, 
        learning_starts=10_000, 
        batch_size=128, 
        gamma=0.999,
        exploration_fraction=0.5,  # Faster exploration decay for feasible data
        exploration_final_eps=0.05,
        target_update_interval=5_000,
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log="./logs/dqn_feasible/"
    )
    
    # === CALLBACKS ===
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_feasible",
        save_vecnormalize=True
    )
    
    # === TRAIN ===
    print(f"\nStarting Training DQN (Feasible DAM)...")
    print(f"  - Checkpoints: {CHECKPOINT_DIR}")
    print(f"  - TensorBoard: ./logs/dqn_feasible/")
    print(f"  - Total Steps: {TIMESTEPS:,}")
    
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
    train_dqn_feasible()
