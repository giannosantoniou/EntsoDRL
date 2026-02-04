
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

def train_dqn():
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/dqn_intraday"
    VECNORM_PATH = "models/vec_normalize_dqn.pkl"
    CHECKPOINT_DIR = "models/checkpoints/dqn/"
    
    # === HYPERPARAMETERS ===
    TIMESTEPS = 5_000_000  # 5 Million Steps (The Real Marathon)
    CHECKPOINT_FREQ = 200_000  # Save every 200k steps
    
    print(f"Loading FEASIBLE DATA: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    def make_env():
        # Use DISCRETE wrapper with 21 levels (5% steps)
        e = BatteryTradingEnvDiscrete(df, battery_params, n_actions=21)
        e = Monitor(e)
        return e
        
    env = DummyVecEnv([make_env])
    
    # CRITICAL: Reward Normalization
    # With Gamma=0.999, unscaled rewards (EUR) lead to Q-Values > 100,000.
    # Neural Nets fail to learn such large magnitudes.
    # VecNormalize keeps Q-values in stable range (~1-100).
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100., clip_reward=10.)
    
    print("Initializing DQN Agent (Super Big Brain Mode - 3 Layers)...")
    
    # Network Architecture: 3 hidden layers of 256 neurons each
    policy_kwargs = dict(net_arch=[256, 256, 256])
    
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-4, 
        buffer_size=100_000, 
        learning_starts=10_000, 
        batch_size=128, 
        gamma=0.999,  # Long-term planning
        exploration_fraction=0.9,  # Slow decay until 90% of steps
        exploration_final_eps=0.05,
        target_update_interval=5_000,  # Sync target network every 5k steps
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        tensorboard_log="./logs/dqn/"
    )
    
    # === CALLBACKS ===
    # Checkpoint callback for crash recovery
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_checkpoint",
        save_vecnormalize=True  # Also save VecNormalize with checkpoints
    )
    
    print(f"Starting Training DQN (Optimized)...")
    print(f"  - Checkpoints every {CHECKPOINT_FREQ:,} steps in {CHECKPOINT_DIR}")
    print(f"  - TensorBoard logs in ./logs/dqn/")
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    # === SAVE FINAL MODEL ===
    model.save(MODEL_PATH)
    env.save(VECNORM_PATH)  # CRITICAL: Save normalizer for evaluation!
    
    print("=" * 50)
    print("DQN Training Complete!")
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  VecNormalize saved to: {VECNORM_PATH}")
    print("=" * 50)

if __name__ == "__main__":
    train_dqn()
