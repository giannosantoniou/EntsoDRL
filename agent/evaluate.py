import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env import BatteryTradingEnvV2
from gym_envs.action_mask import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data.data_loader import generate_sine_wave_data, load_historical_data

def simple_lp_solver(df, battery_params):
    """
    Simplified Perfect Foresight Solver (Greedy Strategy with Lookahead or simple Heuristic).
    A full LP solver would use PuLP/Gurobi.
    Here we implement a 'Buy Low, Sell High' logic based on perfect knowledge of the day.
    """
    # Simply: If price < daily_avg, Charge. If price > daily_avg, Discharge.
    # This is not optimal LP but a good baseline heuristic.
    
    prices = df['price'].values
    avg_price = np.mean(prices)
    
    actions = []
    
    for p in prices:
        if p < avg_price:
            actions.append(-1.0) # Charge
        elif p > avg_price:
            actions.append(1.0) # Discharge
        else:
            actions.append(0.0)
            
    return actions
    
def simple_multi_market_solver(df, battery_params):
    """
    Benchmark updated for Multi-Market.
    Basic Strategy: 
    1. DAM: Buy if Price < Avg, Sell if Price > Avg (Classic Arbitrage).
    2. Balancing: Passive (0.0).
    """
    prices = df['price'].values
    avg_price = np.mean(prices)
    
    actions = [] # List of [dam, phy]
    
    for p in prices:
        dam_action = 0.0
        phy_action = 0.0
        
        # Simple Logic
        if p < avg_price:
            phy_action = -1.0 # Charge physically
            dam_action = -1.0 # Buy in DAM to cover charging
        elif p > avg_price:
            phy_action = 1.0 # Discharge physically
            dam_action = 1.0 # Sell in DAM
            
        actions.append([dam_action, phy_action])
            
    return actions

def evaluate(args):
    # 1. Load Test Data
    if args.year:
        print(f"Loading Historical Data for {args.year} from {args.data_path}...")
        df_full = load_historical_data(args.data_path)
        # Filter by year
        df_test = df_full[df_full.index.year == args.year]
        if df_test.empty:
            print(f"No data found for year {args.year}")
            return
        print(f"Loaded {len(df_test)} hours for evaluation.")
    else:
        # Synthetic
        df_test = generate_sine_wave_data(days=30, seed=999)
        print("Generated Test Data (30 days)")

    battery_params = {
        'capacity_mwh': 50.0,
        'max_charge_mw': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,            # Sungrow LFP Efficiency
        'cost_per_mwh_throughput': 15.0, # ~Degradation Cost
        'transaction_fee': 0.5,
        # Adversarial (Phase 6)
        'market_impact': 0.01,
        'liquidity_risk': 0.05,
        'observation_noise': 0.05
    }

    # 2. Load Agent
    # Update model path to v2
    model_path = "models/ppo_battery_agent_v2"
    if not os.path.exists(model_path + ".zip"):
        print("Model not found. Run train.py first.")
        return

    model = PPO.load(model_path)

    # 3. Inference Loop
    # 3. Inference Loop
    # Must match training env structure (VecEnv + Normalize)
    def make_test_env():
        e = BatteryTradingEnvV2(df_test, battery_params)
        e = ActionMasker(e)
        return e
        
    env = DummyVecEnv([make_test_env])
    
    # Load Normalization Stats
    norm_path = "models/vec_normalize.pkl"
    if os.path.exists(norm_path):
        print(f"Loading Normalization Stats from {norm_path}...")
        env = VecNormalize.load(norm_path, env)
        # Critical: Turn off training and reward normalization for test
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: Normalization stats not found! Agent might fail.")
    
    # 3. Simulate Agent
    obs = env.reset() # VecEnv reset returns only obs
    done = False
    
    drl_rewards = []
    drl_socs = []
    audit_logs = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # VecEnv step returns (obs, reward, done, info) - array format
        obs, reward, terminated, info = env.step(action)
        done = terminated[0] # VecEnv returns list of dones
        
        
        # Audit Log Collection
        if args.audit:
             # Get original row from dataframe (unnormalized)
             # Training env is DummyVecEnv([make_env]). Dataframe is in env.envs[0].df
             # However, current_step in env is now 1 step ahead (because we called step()).
             # We should look at step-1 or grab it from info if possible.
             # Actually, simpler: The BatteryEnv updates current_step AFTER calculating reward.
             # So info['step'] is the step index just executed. Perfect.
             
             step_idx = info[0]['step'] - 1 # info returns current_step, which was incremented
             if step_idx < len(df_test):
                 row = df_test.iloc[step_idx]
                 
                 log_entry = {
                     'Step': step_idx,
                     'Date': row.name if hasattr(row, 'name') else step_idx, # Timestamp index
                     'Price_DAM': row.get('price', 0),
                     'Load_Forecast': row.get('load_forecast', row.get('load', 0)), # Fallback or 0
                     'Solar_Forecast': row.get('solar_forecast', row.get('solar', 0)),
                     'Wind_Forecast': row.get('wind_onshore_forecast', row.get('wind_onshore', 0)),
                     # Agent State
                     'SoC': info[0]['soc'],
                     # Agent Action
                     'Action_DAM_MW': info[0]['dam_mw'],
                     'Action_Phy_MW': info[0]['phy_mw'],
                     # Result
                     'Imbalance_MW': info[0]['imbalance_mw'],
                     'Revenue_DAM': info[0]['dam_rev'],
                     'Revenue_Bal': info[0]['bal_rev'],
                     'Total_Reward': reward[0]
                 }
                 audit_logs.append(log_entry)
        
        drl_rewards.append(reward[0]) # Reward is list
        drl_socs.append(info[0]['soc']) # Info is list
        
    if args.audit and audit_logs:
        audit_df = pd.DataFrame(audit_logs)
        audit_filename = "trade_log.csv"
        audit_df.to_csv(audit_filename, index=False)
        print(f"Audit Log saved to {audit_filename} ({len(audit_df)} records)")
        
    total_drl_profit = sum(drl_rewards)
    print(f"DRL Agent Total Profit: {total_drl_profit:.2f} EUR")

    # 4. Compare with Benchmark (LP Solver / Heuristic)
    print("Running Benchmark (Simple Multi-Market Solver)...")
    env_bench = BatteryTradingEnvV2(df_test, battery_params)
    
    # Use new solver
    lp_actions = simple_multi_market_solver(df_test, battery_params)
    
    obs, _ = env_bench.reset()
    done = False
    idx = 0
    lp_rewards = []
    
    while not done and idx < len(lp_actions):
        # Action is already 2D list [dam, phy]
        action = np.array(lp_actions[idx], dtype=np.float32)
        
        # We manually mask here just in case our heuristic is dumb
        # Or better, use ActionMasker wrapper on bench too
        # But let's just run it roughly
        
        obs, reward, terminated, truncated, _ = env_bench.step(action)
        done = terminated or truncated
        lp_rewards.append(reward)
        idx += 1
        
    total_lp_profit = sum(lp_rewards)
    print(f"Benchmark (Heuristic) Total Profit: {total_lp_profit:.2f} EUR")
    
    # 5. Report
    performance_ratio = (total_drl_profit / total_lp_profit) * 100 if total_lp_profit != 0 else 0
    print(f"Performance Ratio (DRL / Benchmark): {performance_ratio:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed_training_data.csv")
    parser.add_argument("--year", type=int, default=None, help="Year to evaluate on (e.g. 2024). If None, uses synthetic data.")
    parser.add_argument("--audit", action="store_true", help="Save detailed trade log to trade_log.csv")
    args = parser.parse_args()
    evaluate(args)
