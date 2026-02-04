"""
DQN Evaluation Script
Properly loads the trained model WITH VecNormalize for accurate evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_discrete import BatteryTradingEnvDiscrete
from data.data_loader import load_historical_data


def evaluate_dqn(n_episodes: int = 5):
    """Evaluate trained DQN agent."""
    
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/dqn_intraday.zip"
    VECNORM_PATH = "models/vec_normalize_dqn.pkl"
    
    # === LOAD DATA ===
    print(f"Loading data from: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    # === CREATE ENVIRONMENT ===
    def make_env():
        e = BatteryTradingEnvDiscrete(df, battery_params, n_actions=21)
        e = Monitor(e)
        return e
    
    env = DummyVecEnv([make_env])
    
    # CRITICAL: Load VecNormalize from training!
    if os.path.exists(VECNORM_PATH):
        print(f"Loading VecNormalize from: {VECNORM_PATH}")
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False  # Don't update stats during evaluation
        env.norm_reward = False  # Return raw rewards for interpretability
    else:
        print(f"WARNING: VecNormalize not found at {VECNORM_PATH}")
        print("Evaluation may be inaccurate without proper normalization!")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # === LOAD MODEL ===
    print(f"Loading model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH, env=env)
    
    # === EVALUATE ===
    print(f"\nRunning {n_episodes} evaluation episodes...")
    print("=" * 60)
    
    all_rewards = []
    all_violations = []
    all_penalties = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_violations = 0
        episode_penalty = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            if info[0].get('violation', 0) > 1.0:
                episode_violations += 1
            episode_penalty += info[0].get('penalty', 0)
            steps += 1
        
        all_rewards.append(episode_reward)
        all_violations.append(episode_violations)
        all_penalties.append(episode_penalty)
        
        print(f"Episode {ep+1}: Reward = {episode_reward:,.2f}, "
              f"Violations = {episode_violations}, "
              f"Penalty = {episode_penalty:,.2f}, "
              f"Steps = {steps:,}")
    
    # === SUMMARY ===
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Mean Reward:     {np.mean(all_rewards):>15,.2f}")
    print(f"  Std Reward:      {np.std(all_rewards):>15,.2f}")
    print(f"  Mean Violations: {np.mean(all_violations):>15,.1f}")
    print(f"  Mean Penalty:    {np.mean(all_penalties):>15,.2f}")
    print("=" * 60)
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_violations': np.mean(all_violations),
        'mean_penalty': np.mean(all_penalties),
    }


if __name__ == "__main__":
    evaluate_dqn(n_episodes=5)
