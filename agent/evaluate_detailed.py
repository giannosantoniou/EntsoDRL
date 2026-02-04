"""
Detailed Evaluation for MaskablePPO - Trade Analysis
Shows every trade with prices, P&L breakdown, and statistics.
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
from data.dam_commitment_generator import FeasibleDAMGenerator


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


def evaluate_detailed(model_path=None, test_ratio=0.2, deterministic=True):
    DATA_PATH = "data/feasible_data_with_dam.csv"
    
    if model_path is None:
        checkpoint_dir = "models/checkpoints/ppo_masked/"
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')],
                           key=lambda x: int(x.split('_')[-2]), reverse=True)
        model_path = os.path.join(checkpoint_dir, checkpoints[0])
    
    print("=" * 70)
    print("MaskablePPO DETAILED TRADE ANALYSIS")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Deterministic: {deterministic}")
    
    # Load data
    df = load_historical_data(DATA_PATH)
    split_idx = int(len(df) * (1 - test_ratio))
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Test size: {len(test_df):,} hours")
    
    # Generate DAM commitments
    battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}
    gen = FeasibleDAMGenerator(
        capacity_mwh=battery_params['capacity_mwh'], 
        max_power_mw=battery_params['max_discharge_mw'],
        efficiency=battery_params['efficiency']
    )
    test_df['dam_commitment'] = gen.generate(test_df)
    
    # Create env
    def make_env():
        env = BatteryEnvMasked(test_df, battery_params, n_actions=21)
        env = Monitor(env)
        env = ActionMasker(env, lambda e: e.env.action_masks())
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)
    
    # Load model
    model = MaskablePPO.load(model_path, env=env)
    
    # Evaluate with detailed logging
    print("\n" + "=" * 70)
    print("RUNNING DETAILED EVALUATION...")
    print("=" * 70)
    
    obs = env.reset()
    done = False
    
    # Track everything
    trades = []
    total_reward = 0
    total_dam_rev = 0
    total_balancing_rev = 0
    total_degradation = 0
    violations = 0
    steps = 0
    
    action_levels = np.linspace(-1.0, 1.0, 21)
    
    while not done:
        mask = get_action_mask(env)
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=np.array([mask]))
        
        obs, reward, done, info = env.step(action)
        info = info[0]
        
        action_idx = int(action[0])
        action_level = action_levels[action_idx]
        actual_mw = info.get('actual_mw', action_level * 30.0)
        
        # Track P&L components
        dam_rev = info.get('dam_rev', 0)
        balancing_rev = info.get('balancing_rev', 0)
        degradation = info.get('degradation', 0)
        net_profit = info.get('net_profit', 0)
        
        total_reward += reward[0]
        total_dam_rev += dam_rev
        total_balancing_rev += balancing_rev
        total_degradation += degradation
        
        if info.get('violation', 0) > 0.5:
            violations += 1
        
        # Log trades (non-idle actions)
        if abs(action_level) > 0.05:  # Not idle
            trades.append({
                'step': steps,
                'action': action_idx,
                'action_level': action_level,
                'actual_mw': actual_mw,
                'soc': info.get('soc', 0),
                'dam_price': info.get('dam_price', 0),
                'best_bid': info.get('best_bid', 0),
                'best_ask': info.get('best_ask', 0),
                'spread': info.get('spread', 0),
                'dam_rev': dam_rev,
                'balancing_rev': balancing_rev,
                'degradation': degradation,
                'net_profit': net_profit,
                'reward': reward[0]
            })
        
        steps += 1
    
    # Summary Statistics
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"  Total Reward:         {total_reward:>15,.2f}")
    print(f"  Total DAM Revenue:    {total_dam_rev:>15,.2f} €")
    print(f"  Total Balancing Rev:  {total_balancing_rev:>15,.2f} €")
    print(f"  Total Degradation:    {total_degradation:>15,.2f} €")
    print(f"  Violations:           {violations:>15,} ({violations/steps*100:.2f}%)")
    print(f"  Steps:                {steps:>15,}")
    print(f"  Total Trades:         {len(trades):>15,} ({len(trades)/steps*100:.2f}%)")
    
    # Trade Analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Separate buys and sells
        sells = trades_df[trades_df['action_level'] > 0]
        buys = trades_df[trades_df['action_level'] < 0]
        
        print("\n" + "-" * 70)
        print("TRADE BREAKDOWN")
        print("-" * 70)
        print(f"\n  SELL Trades (Discharge):")
        print(f"    Count:              {len(sells):>10,}")
        if len(sells) > 0:
            print(f"    Total MW:           {sells['actual_mw'].sum():>10,.1f}")
            print(f"    Avg Price:          {sells['dam_price'].mean():>10,.2f} €/MWh")
            print(f"    Max Price:          {sells['dam_price'].max():>10,.2f} €/MWh")
            print(f"    Avg Action Level:   {sells['action_level'].mean():>10,.2f}")
        
        print(f"\n  BUY Trades (Charge):")
        print(f"    Count:              {len(buys):>10,}")
        if len(buys) > 0:
            print(f"    Total MW:           {buys['actual_mw'].sum():>10,.1f}")
            print(f"    Avg Price:          {buys['dam_price'].mean():>10,.2f} €/MWh")
            print(f"    Min Price:          {buys['dam_price'].min():>10,.2f} €/MWh")
            print(f"    Avg Action Level:   {buys['action_level'].mean():>10,.2f}")
        
        # Show top 10 best trades
        print("\n" + "-" * 70)
        print("TOP 10 MOST PROFITABLE TRADES")
        print("-" * 70)
        top_trades = trades_df.nlargest(10, 'net_profit')
        for _, t in top_trades.iterrows():
            action_type = "SELL" if t['action_level'] > 0 else "BUY"
            print(f"  Step {t['step']:>5}: {action_type:>4} {abs(t['actual_mw']):>5.1f}MW @ {t['dam_price']:>6.1f}€ "
                  f"→ Net: {t['net_profit']:>7.2f}€  (SoC: {t['soc']*100:.1f}%)")
        
        # Show worst trades
        print("\n" + "-" * 70)
        print("TOP 10 LEAST PROFITABLE TRADES")
        print("-" * 70)
        bottom_trades = trades_df.nsmallest(10, 'net_profit')
        for _, t in bottom_trades.iterrows():
            action_type = "SELL" if t['action_level'] > 0 else "BUY"
            print(f"  Step {t['step']:>5}: {action_type:>4} {abs(t['actual_mw']):>5.1f}MW @ {t['dam_price']:>6.1f}€ "
                  f"→ Net: {t['net_profit']:>7.2f}€  (SoC: {t['soc']*100:.1f}%)")
        
        # Save to CSV
        trades_df.to_csv('detailed_trades.csv', index=False)
        print(f"\n✅ Detailed trades saved to: detailed_trades.csv")
    
    print("=" * 70)
    
    return total_reward, trades


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
    args = parser.parse_args()
    
    evaluate_detailed(deterministic=not args.stochastic)
