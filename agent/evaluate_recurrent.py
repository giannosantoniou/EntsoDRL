"""
Evaluation for RecurrentPPO Model with LSTM State Management.
Generates detailed text file with step-by-step trading decisions.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from gym_envs.battery_env_recurrent import BatteryEnvRecurrent
from data.data_loader import load_historical_data
from data.upsample_to_15min import upsample_hourly_to_15min
from data.dam_commitment_generator import regenerate_dam_commitments


def evaluate_recurrent(
    model_path: str = None, 
    vecnorm_path: str = None, 
    output_file: str = "decision_analysis_recurrent.txt", 
    n_steps: int = 2000, 
    test_ratio: float = 0.2,
    use_15min: bool = True
):
    """
    Evaluate RecurrentPPO model with proper LSTM state handling.
    
    Args:
        model_path: Path to trained model (.zip)
        vecnorm_path: Path to VecNormalize stats (.pkl)
        output_file: Output text file path
        n_steps: Number of steps to evaluate
        test_ratio: Fraction of data to use for testing (from end)
        use_15min: If True, use 15-min intervals; else hourly
    """
    
    # Find latest checkpoint if not specified
    time_suffix = "15min" if use_15min else "hourly"
    if model_path is None:
        checkpoint_dir = f"models/checkpoints/ppo_recurrent_{time_suffix}/"
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')],
                key=lambda x: int(x.split('_')[-2]) if x.split('_')[-2].isdigit() else 0, 
                reverse=True
            )
            if checkpoints:
                model_path = os.path.join(checkpoint_dir, checkpoints[0])
                steps = checkpoints[0].split('_')[-2]
                vecnorm_path = os.path.join(checkpoint_dir, f"ppo_recurrent_vecnormalize_{steps}_steps.pkl")
        
        # Fallback to main model
        if model_path is None or not os.path.exists(model_path):
            model_path = f"models/ppo_recurrent_{time_suffix}.zip"
            vecnorm_path = f"models/vec_normalize_recurrent_{time_suffix}.pkl"
    
    print("=" * 80)
    print("RECURRENTPPO (LSTM) EVALUATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    
    # Load data
    print("\nLoading data...")
    df = load_historical_data("data/feasible_data_with_dam.csv")
    original_len = len(df)
    
    time_step_hours = 0.25 if use_15min else 1.0
    if use_15min:
        df = upsample_hourly_to_15min(df)
        print(f"Original data: {original_len:,} hours → Upsampled: {len(df):,} 15-min intervals")
    
    # Use last portion for evaluation
    split_idx = int(len(df) * (1 - test_ratio))
    df_test = df.iloc[split_idx:].copy()
    time_periods = len(df_test)/4 if use_15min else len(df_test)
    print(f"Test period: {len(df_test):,} intervals ({time_periods:.0f} hours)")
    
    # Battery parameters
    battery_params = {
        'capacity_mwh': 30.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }
    
    # Regenerate DAM commitments
    df_test = regenerate_dam_commitments(df_test, battery_params=battery_params)
    
    # Create environment
    def make_env():
        env = BatteryEnvRecurrent(
            df_test, 
            battery_params, 
            n_actions=21, 
            time_step_hours=time_step_hours,
            invalid_action_penalty=50.0
        )
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize stats...")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: VecNormalize not found, using fresh normalization")
        env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # Load model
    print(f"Loading model...")
    model = RecurrentPPO.load(model_path, env=env)
    
    # Prepare logging
    lines = []
    lines.append("=" * 120)
    lines.append(f"DECISION ANALYSIS - RECURRENTPPO (LSTM)")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model: {model_path}")
    lines.append(f"Steps Analyzed: {n_steps}")
    lines.append(f"Time Resolution: {'15 minutes' if use_15min else 'hourly'}")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    header = (f"{'Step':>6} | {'Time':>5} | {'Action':>6} | {'MW':>8} | {'SoC':>6} | "
              f"{'Bid':>7} | {'Ask':>7} | {'DAM':>8} | "
              f"{'Reward':>9} | {'Profit':>10} | {'Invalid':>7}")
    lines.append(header)
    lines.append("-" * 120)
    
    # Run evaluation with LSTM state tracking
    print(f"\nRunning {n_steps} steps evaluation with LSTM states...")
    obs = env.reset()
    
    # CRITICAL: Initialize LSTM states
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    
    total_profit = 0.0
    total_reward = 0.0
    violations = 0
    invalid_actions = 0
    action_counts = {}
    hourly_profits = []
    current_hour_profit = 0.0
    steps_in_hour = 0
    
    # Trading statistics
    discharge_energy = 0.0
    charge_energy = 0.0
    
    for step in range(n_steps):
        # LSTM-aware prediction
        action, lstm_states = model.predict(
            obs, 
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        
        # Execute step
        obs, reward, done, info = env.step(action)
        
        # CRITICAL: Update episode starts for LSTM
        episode_starts = done
        
        # Extract info
        i = info[0]
        action_val = int(action[0])
        action_mw = i.get('actual_mw', 0)
        soc = i.get('soc', 0.5)
        bid = i.get('best_bid', 0)
        ask = i.get('best_ask', 0)
        dam_commit = i.get('dam_commitment', 0)
        net_profit = i.get('net_profit', 0)
        is_invalid = i.get('invalid_action', False)
        reward_val = reward[0]
        
        # Track violations and invalid actions
        if i.get('violation', 0) > 0.5:
            violations += 1
        if is_invalid:
            invalid_actions += 1
        
        # Track energy flows
        if action_mw > 0:
            discharge_energy += action_mw * time_step_hours
        else:
            charge_energy += abs(action_mw) * time_step_hours
        
        # Track metrics
        total_profit += net_profit
        total_reward += reward_val
        action_counts[action_val] = action_counts.get(action_val, 0) + 1
        
        # Hourly profit aggregation
        steps_per_hour = 4 if use_15min else 1
        current_hour_profit += net_profit
        steps_in_hour += 1
        if steps_in_hour == steps_per_hour:
            hourly_profits.append(current_hour_profit)
            current_hour_profit = 0.0
            steps_in_hour = 0
        
        # Time of day
        intervals_per_day = 96 if use_15min else 24
        interval_in_day = step % intervals_per_day
        if use_15min:
            hour = interval_in_day // 4
            minute = (interval_in_day % 4) * 15
            time_str = f"{hour:02d}:{minute:02d}"
        else:
            time_str = f"{interval_in_day:02d}:00"
        
        # Log line
        invalid_str = "⚠️" if is_invalid else ""
        line = (f"{step+1:>6} | {time_str:>5} | {action_val:>6} | {action_mw:>+8.1f} | "
                f"{soc:>6.2f} | {bid:>7.1f} | {ask:>7.1f} | {dam_commit:>+8.1f} | "
                f"{reward_val:>9.2f} | {net_profit:>+10.1f} | {invalid_str:>7}")
        lines.append(line)
        
        if done[0]:
            lines.append("-" * 120)
            lines.append(">>> EPISODE END - LSTM states reset <<<")
            lines.append("-" * 120)
            obs = env.reset()
            # Reset LSTM states on episode end
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
    
    # Calculate statistics
    avg_hourly_profit = np.mean(hourly_profits) if hourly_profits else 0
    std_hourly_profit = np.std(hourly_profits) if hourly_profits else 0
    
    # Summary section
    lines.append("")
    lines.append("=" * 120)
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 120)
    lines.append("")
    
    # Performance metrics
    lines.append("=== PERFORMANCE METRICS ===")
    lines.append(f"Total Steps:          {n_steps:>12,}")
    lines.append(f"Total Hours:          {n_steps * time_step_hours:>12.1f}")
    lines.append(f"Total Days:           {n_steps * time_step_hours / 24:>12.2f}")
    lines.append(f"Total Reward:         {total_reward:>12,.2f}")
    lines.append(f"Total Profit:         €{total_profit:>11,.2f}")
    lines.append(f"Avg Profit/Hour:      €{avg_hourly_profit:>11.2f}")
    lines.append(f"Std Profit/Hour:      €{std_hourly_profit:>11.2f}")
    lines.append(f"Violations:           {violations:>12} ({violations/n_steps*100:.1f}%)")
    lines.append(f"Invalid Actions:      {invalid_actions:>12} ({invalid_actions/n_steps*100:.1f}%)")
    lines.append("")
    
    # Energy statistics
    lines.append("=== ENERGY STATISTICS ===")
    lines.append(f"Total Discharged:     {discharge_energy:>12.1f} MWh")
    lines.append(f"Total Charged:        {charge_energy:>12.1f} MWh")
    lines.append(f"Net Energy Flow:      {discharge_energy - charge_energy:>+12.1f} MWh")
    lines.append(f"Round-Trip Cycles:    {min(discharge_energy, charge_energy) / battery_params['capacity_mwh']:>12.1f}")
    lines.append("")
    
    # Action distribution
    lines.append("=== ACTION DISTRIBUTION ===")
    action_levels = np.linspace(-1.0, 1.0, 21)
    max_power = battery_params['max_discharge_mw']
    
    charge_count = sum(action_counts.get(i, 0) for i in range(0, 10))
    idle_count = action_counts.get(10, 0)
    discharge_count = sum(action_counts.get(i, 0) for i in range(11, 21))
    
    lines.append(f"Charge Actions:       {charge_count:>8} ({charge_count/n_steps*100:>5.1f}%)")
    lines.append(f"Idle Actions:         {idle_count:>8} ({idle_count/n_steps*100:>5.1f}%)")
    lines.append(f"Discharge Actions:    {discharge_count:>8} ({discharge_count/n_steps*100:>5.1f}%)")
    lines.append("")
    
    lines.append("Detailed Action Breakdown:")
    for act in range(21):
        count = action_counts.get(act, 0)
        if count > 0:
            pct = count / n_steps * 100
            mw = action_levels[act] * max_power
            bar = "█" * int(pct / 2)
            lines.append(f"  Action {act:>2} ({mw:>+6.1f} MW): {count:>5} ({pct:>5.1f}%) {bar}")
    
    lines.append("")
    lines.append("=" * 120)
    lines.append("END OF REPORT")
    lines.append("=" * 120)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Total Reward:      {total_reward:>12,.2f}")
    print(f"  Total Profit:      €{total_profit:>11,.2f}")
    print(f"  Avg Profit/Hour:   €{avg_hourly_profit:>11.2f}")
    print(f"  Violations:        {violations:>12} ({violations/n_steps*100:.1f}%)")
    print(f"  Invalid Actions:   {invalid_actions:>12} ({invalid_actions/n_steps*100:.1f}%)")
    print(f"  Energy Traded:     {discharge_energy + charge_energy:>12.1f} MWh")
    print("=" * 60)
    print(f"\n✅ Decision log saved to: {output_file}")
    
    return total_reward, total_profit, violations, invalid_actions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RecurrentPPO battery model")
    parser.add_argument("--model", type=str, default=None, help="Model path (.zip)")
    parser.add_argument("--vecnorm", type=str, default=None, help="VecNormalize path (.pkl)")
    parser.add_argument("--output", type=str, default="decision_analysis_recurrent.txt", help="Output file")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps to evaluate")
    parser.add_argument("--hourly", action="store_true", help="Use hourly instead of 15-min")
    
    args = parser.parse_args()
    
    evaluate_recurrent(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        output_file=args.output,
        n_steps=args.steps,
        use_15min=not args.hourly
    )
