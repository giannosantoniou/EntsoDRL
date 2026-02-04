"""
DQN Evaluation on UNSEEN Data (Train/Test Split)
Tests the trained model on data it has never seen during training.
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
from data.dam_commitment_generator import FeasibleDAMGenerator


def evaluate_on_unseen(
    model_path: str = None,
    vecnorm_path: str = None,
    test_ratio: float = 0.2,
    n_episodes: int = 1,
):
    """
    Evaluate trained DQN on unseen test data.
    
    Args:
        model_path: Path to model .zip file
        vecnorm_path: Path to VecNormalize .pkl file  
        test_ratio: Fraction of data to use for testing (last X%)
        n_episodes: Number of evaluation episodes
    """
    
    # === PATHS ===
    DATA_PATH = "data/feasible_data_with_dam.csv"
    
    # Find latest checkpoint if not specified
    if model_path is None:
        checkpoint_dir = "models/checkpoints/dqn_feasible/"
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')],
            key=lambda x: int(x.split('_')[-2]),  # Sort by step number
            reverse=True
        )
        if checkpoints:
            model_path = os.path.join(checkpoint_dir, checkpoints[0])
            # VecNormalize is saved alongside
            vecnorm_path = model_path.replace('.zip', '_vecnormalize.pkl')
            if not os.path.exists(vecnorm_path):
                vecnorm_path = None
        else:
            raise FileNotFoundError("No checkpoints found!")
    
    print("=" * 60)
    print("DQN EVALUATION ON UNSEEN DATA")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Test ratio: {test_ratio*100:.0f}% of data")
    print("=" * 60)
    
    # === LOAD DATA ===
    print("\nLoading data...")
    df = load_historical_data(DATA_PATH)
    
    # Split: Use LAST X% for testing (unseen future data)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Total data: {len(df):,} hours ({len(df)/8760:.1f} years)")
    print(f"Training data: {len(train_df):,} hours ({len(train_df)/8760:.1f} years)")
    print(f"Test data: {len(test_df):,} hours ({len(test_df)/8760:.1f} years) [UNSEEN]")
    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    
    # === GENERATE FEASIBLE DAM COMMITMENTS FOR TEST DATA ===
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    print("\nGenerating DAM commitments for test data...")
    generator = FeasibleDAMGenerator(**{
        'capacity_mwh': battery_params['capacity_mwh'],
        'max_power_mw': battery_params['max_discharge_mw'],
        'efficiency': battery_params['efficiency'],
    })
    test_df = test_df.copy()
    test_df['dam_commitment'] = generator.generate(test_df)
    
    # === CREATE TEST ENVIRONMENT ===
    def make_test_env():
        e = BatteryTradingEnvDiscrete(test_df, battery_params, n_actions=21)
        e = Monitor(e)
        return e
    
    env = DummyVecEnv([make_test_env])
    
    # Load VecNormalize if available
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from training...")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: No VecNormalize found, creating new one (may affect results)")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # === LOAD MODEL ===
    print(f"\nLoading model...")
    model = DQN.load(model_path, env=env)
    
    # === EVALUATE ===
    print(f"\nRunning {n_episodes} evaluation episode(s) on UNSEEN data...")
    print("-" * 60)
    
    all_rewards = []
    all_violations = []
    all_penalties = []
    all_actions = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_violations = 0
        episode_penalty = 0
        action_counts = {i: 0 for i in range(21)}  # Track action distribution
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            action_counts[int(action[0])] += 1
            
            if info[0].get('violation', 0) > 1.0:
                episode_violations += 1
            episode_penalty += info[0].get('penalty', 0)
            steps += 1
        
        all_rewards.append(episode_reward)
        all_violations.append(episode_violations)
        all_penalties.append(episode_penalty)
        all_actions.append(action_counts)
        
        print(f"Episode {ep+1}:")
        print(f"  Total Reward: {episode_reward:>15,.2f}")
        print(f"  Violations:   {episode_violations:>15,}")
        print(f"  Total Penalty:{episode_penalty:>15,.2f}")
        print(f"  Steps:        {steps:>15,}")
    
    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (UNSEEN DATA)")
    print("=" * 60)
    print(f"  Mean Reward:     {np.mean(all_rewards):>15,.2f}")
    print(f"  Std Reward:      {np.std(all_rewards):>15,.2f}")
    print(f"  Mean Violations: {np.mean(all_violations):>15,.1f}")
    print(f"  Mean Penalty:    {np.mean(all_penalties):>15,.2f}")
    
    # Action distribution analysis
    if all_actions:
        total_actions = sum(all_actions[0].values())
        print(f"\n  Action Distribution (Episode 1):")
        # Group actions: Discharge (11-20), Idle (10), Charge (0-9)
        discharge = sum(all_actions[0][i] for i in range(11, 21))
        idle = all_actions[0][10]
        charge = sum(all_actions[0][i] for i in range(0, 10))
        print(f"    Discharge (>0):  {discharge/total_actions*100:>6.1f}%")
        print(f"    Idle (=0):       {idle/total_actions*100:>6.1f}%")
        print(f"    Charge (<0):     {charge/total_actions*100:>6.1f}%")
    
    print("=" * 60)
    
    # Interpretation
    avg_reward = np.mean(all_rewards)
    if avg_reward > 0:
        print("✅ PROFITABLE on unseen data!")
    elif avg_reward > -50000:
        print("📈 Making progress - losses reduced significantly")
    else:
        print("⚠️  Still learning - needs more training")
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_violations': np.mean(all_violations),
        'mean_penalty': np.mean(all_penalties),
    }


if __name__ == "__main__":
    evaluate_on_unseen()
