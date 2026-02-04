"""
Decision Logger - Evaluates trained model and logs every decision.
Creates detailed text file for analysis.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.wrappers import ActionMasker

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments


def create_decision_log(model_path: str, vecnorm_path: str, output_file: str, n_steps: int = 500):
    """
    Run model and log every decision to text file.
    """
    # Load data
    print("Loading data...")
    df = load_historical_data("data/feasible_data_with_dam.csv")
    
    # Use last 20% for evaluation
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    
    battery_params = {
        'capacity_mwh': 30.0,       # Updated: was 50.0
        'max_discharge_mw': 30.0,   # Updated: was 50.0
        'efficiency': 0.94,
    }
    
    # Regenerate DAM
    df_test = regenerate_dam_commitments(df_test, battery_params=battery_params)
    
    # Create environment
    def mask_fn(env):
        return env.action_masks()
    
    def make_env():
        env = BatteryEnvMasked(df_test, battery_params, n_actions=21)
        env = ActionMasker(env, mask_fn)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False
    
    # Load model
    print(f"Loading model: {model_path}")
    model = MaskablePPO.load(model_path, env=env)
    
    # Run evaluation and log
    print(f"Running {n_steps} steps and logging...")
    
    obs = env.reset()
    
    lines = []
    lines.append("=" * 100)
    lines.append(f"DECISION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {model_path}")
    lines.append(f"Steps: {n_steps}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Step':>5} | {'Action':>6} | {'MW':>7} | {'SoC':>5} | {'Bid':>6} | {'Ask':>6} | {'DAM Commit':>10} | {'Imbalance':>9} | {'Reward':>8} | {'Net Profit':>10}")
    lines.append("-" * 100)
    
    total_profit = 0
    total_reward = 0
    action_counts = {}
    
    for step in range(n_steps):
        # Get action masks
        base_env = env.envs[0]
        while hasattr(base_env, 'env'):
            if hasattr(base_env, 'action_masks'):
                break
            base_env = base_env.env
        
        action_masks = base_env.action_masks() if hasattr(base_env, 'action_masks') else None
        
        # Predict
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        
        # Step
        obs, reward, done, info = env.step(action)
        
        # Extract info
        i = info[0]
        action_val = action[0]
        action_mw = i.get('actual_mw', 0)
        soc = i.get('soc', 0)
        bid = i.get('best_bid', 0)
        ask = i.get('best_ask', 0)
        dam_commit = i.get('dam_rev', 0) / max(1, bid)  # Estimate from dam_rev
        imbalance = i.get('imbalance_mw', 0)
        net_profit = i.get('net_profit', 0)
        
        # Track metrics
        total_profit += net_profit
        total_reward += reward[0]
        action_counts[action_val] = action_counts.get(action_val, 0) + 1
        
        # Log line
        lines.append(f"{step+1:>5} | {action_val:>6} | {action_mw:>7.1f} | {soc:>5.2f} | {bid:>6.1f} | {ask:>6.1f} | {dam_commit:>10.1f} | {imbalance:>9.1f} | {reward[0]:>8.2f} | {net_profit:>10.1f}")
        
        if done[0]:
            lines.append("--- EPISODE END ---")
            obs = env.reset()
    
    # Summary
    lines.append("")
    lines.append("=" * 100)
    lines.append("SUMMARY")
    lines.append("=" * 100)
    lines.append(f"Total Steps: {n_steps}")
    lines.append(f"Total Reward: {total_reward:.2f}")
    lines.append(f"Total Profit: €{total_profit:,.2f}")
    lines.append(f"Avg Profit/Step: €{total_profit/n_steps:.2f}")
    lines.append("")
    lines.append("Action Distribution:")
    for act, count in sorted(action_counts.items()):
        pct = count / n_steps * 100
        action_levels = np.linspace(-1.0, 1.0, 21)
        mw = action_levels[act] * 30  # 30MW battery
        lines.append(f"  Action {act:>2} ({mw:>+6.1f} MW): {count:>4} times ({pct:>5.1f}%)")
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nLog saved to: {output_file}")
    print(f"Total Profit: €{total_profit:,.2f}")


if __name__ == "__main__":
    import glob
    
    # Find latest checkpoint
    checkpoints = sorted(glob.glob("models/checkpoints/ppo_masked/ppo_masked_*_steps.zip"),
                         key=lambda x: int(x.split("_")[-2]))
    if checkpoints:
        latest = checkpoints[-1]
        # VecNormalize pattern: ppo_masked_vecnormalize_STEPS.pkl
        steps = latest.split("_")[-2]  # Extract steps number
        vecnorm = f"models/checkpoints/ppo_masked/ppo_masked_vecnormalize_{steps}_steps.pkl"
        print(f"Model: {latest}")
        print(f"VecNorm: {vecnorm}")
        if os.path.exists(vecnorm):
            create_decision_log(latest, vecnorm, "decision_analysis.txt", n_steps=2000)
        else:
            print(f"VecNormalize not found: {vecnorm}")
            print("Available vecnorm files:")
            for f in glob.glob("models/checkpoints/ppo_masked/*vecnormalize*.pkl"):
                print(f"  {f}")
    else:
        print("No checkpoints found!")
