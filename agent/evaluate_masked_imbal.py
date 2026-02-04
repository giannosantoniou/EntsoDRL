"""
Evaluation for MaskablePPO with Imbalance Risk Features
Generates decision analysis log.
"""
import os
import sys
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments


def get_action_mask(vec_env):
    """Traverse wrappers to get action_masks from the base environment."""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        if hasattr(env, 'action_masks'):
            return env.action_masks()
        env = env.env
    if hasattr(env, 'action_masks'):
        return env.action_masks()
    raise RuntimeError("Could not find action_masks method!")


def evaluate_masked_imbal(model_path=None, test_ratio=0.2):
    DATA_PATH = "data/feasible_data_with_dam.csv"
    VECNORM_PATH = "models/vec_normalize_masked_imbal.pkl"
    LOG_FILE = "decision_analysis_imbal.txt"
    
    if model_path is None:
        checkpoint_dir = "models/checkpoints/ppo_masked_imbal/"
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')],
                               key=lambda x: int(x.split('_')[-2]), reverse=True)
            model_path = os.path.join(checkpoint_dir, checkpoints[0])
        else:
            model_path = "models/ppo_masked_imbal"
    
    print("=" * 60)
    print("MaskablePPO EVALUATION (IMBALANCE RISK)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Log:   {LOG_FILE}")
    
    # Load data
    df = load_historical_data(DATA_PATH)
    split_idx = int(len(df) * (1 - test_ratio))
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Test size: {len(test_df):,} hours")
    
    # Battery parameters (must match training)
    battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}
    
    # Regenerate DAM commitments for test consistency
    # Pass battery_params as a keyword argument to avoid position mismatch (arg2 is 'generator')
    test_df = regenerate_dam_commitments(test_df, battery_params=battery_params)
    
    # Create env
    def make_env():
        env = BatteryEnvMasked(test_df, battery_params, n_actions=21)
        env = Monitor(env)
        env = ActionMasker(env, lambda e: e.env.action_masks())
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if exists
    if os.path.exists(VECNORM_PATH):
        print(f"Loading VecNormalize stats from {VECNORM_PATH}")
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: VecNormalize stats not found. Using new normalization (suboptimal).")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # Load model
    model = MaskablePPO.load(model_path, env=env)
    
    # Open log file
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"DECISION ANALYSIS LOG - IMBALANCE RISK AGENT\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write("="*120 + "\n")
        f.write(f"{'Step':<6} | {'Time':<19} | {'DAM':<6} | {'SoC%':<5} | {'Risk':<5} | {'Action':<15} | {'MW':<6} | {'Price':<6} | {'Profit':<8}\n")
        f.write("-" * 120 + "\n")
        
        obs = env.reset()
        done = False
        total_profit = 0.0
        steps = 0
        
        # Real battery physics for calculation
        cap_mwh = battery_params['capacity_mwh']
        max_mw = battery_params['max_discharge_mw']
        eff = battery_params['efficiency']
        eff_sqrt = np.sqrt(eff)
        time_step = 1.0  # Hourly
        
        actions_count = {i: 0 for i in range(21)}
        
        while not done:
            mask = get_action_mask(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=np.array([mask]))
            action_idx = int(action[0])
            actions_count[action_idx] += 1
            
            # Access unnormalized environment
            real_env = env.envs[0].env.env
            current_idx = real_env.current_step
            row = real_env.df.iloc[current_idx]
            
            # Get market data
            price = row.get('price', 0.0)
            
            # Calculate REAL profit manually (bypassing gym reward normalization)
            # Action levels: -1.0 to 1.0
            level = real_env.action_levels[action_idx]
            mw = 0.0
            step_profit = 0.0
            
            if level > 0:  # Discharge
                mw = level * max_mw
                # Available energy check handled by action mask, but good to be safe
                step_profit = mw * time_step * price
            elif level < 0:  # Charge
                mw = level * max_mw  # Negative
                step_profit = mw * time_step * price # Negative cost (buy)
            
            # Observation for next step
            obs, _, done, _ = env.step(action)
            
            total_profit += step_profit
            steps += 1
            
            # Risk info
            # Calculate imbalance risk using FeatureEngineer
            history_window = int(2 / real_env.time_step_hours) + 1
            start_idx = max(0, current_idx - history_window)
            end_idx = current_idx + 1
            p_hist = real_env.imbalance_prices[start_idx:end_idx]
            r_hist = real_env.imbalance_res[start_idx:end_idx]
            c_hour = (current_idx % 24)
            risk_info = real_env.feature_engine.calculate_imbalance_risk(p_hist, r_hist, c_hour)
            
            # Log action string
            action_name = "IDLE"
            if level < 0: action_name = f"CHARGE {abs(level):.0%}"
            elif level > 0: action_name = f"DISCHG {level:.0%}"
            
            timestamp = row.name if hasattr(row, 'name') else f"Step {steps}"
            
            f.write(f"{steps:<6} | {str(timestamp):<19} | {price:>6.1f} | {real_env.current_soc:>5.2f} | {risk_info['risk_score']:>5.2f} | {action_name:<15} | {mw:>6.1f} | {price:>6.1f} | {step_profit:>8.2f}\n")
            
            if steps % 1000 == 0:
                print(f"Processed {steps} steps. Current Profit: {total_profit:,.2f} EUR")

        # SUMMARY
        avg_profit = total_profit / steps if steps > 0 else 0
        hours = steps  # Assuming hourly resolution
        
        f.write("="*120 + "\n")
        f.write("SUMMARY\n")
        f.write("="*120 + "\n")
        f.write(f"Total Steps:       {steps:,}\n")
        f.write(f"Total Profit:      {total_profit:,.2f} EUR\n")
        f.write(f"Avg Profit/Step:   {avg_profit:,.2f} EUR\n")
        f.write(f"Avg Profit/Day:    {avg_profit*24:,.2f} EUR\n")
        f.write("-" * 40 + "\n")
        f.write("Action Distribution:\n")
        f.write(f"  Discharge: {sum(actions_count[i] for i in range(11,21))/steps*100:.1f}%\n")
        f.write(f"  Charge:    {sum(actions_count[i] for i in range(0,10))/steps*100:.1f}%\n")
        f.write(f"  Idle:      {actions_count[10]/steps*100:.1f}%\n")
        f.write("="*120 + "\n")

    print(f"\nEvaluation Complete!")
    print(f"Total Profit:    {total_profit:,.2f} EUR")
    print(f"Avg Profit/Step: {avg_profit:,.2f} EUR")
    print(f"Log saved to: {os.path.abspath(LOG_FILE)}")
    return total_profit

if __name__ == "__main__":
    evaluate_masked_imbal()
