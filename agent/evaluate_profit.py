
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_v5 import BatteryTradingEnvV5
from data.data_loader import load_historical_data

def evaluate_profit_breakdown():
    # CONFIG
    # Use the same feasible data
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/ppo_intraday_profit" # This might be incomplete, we need to save intermediate?
    # Actually, we can load the checkpoint if available, or just use the model in RAM if we were inside the script.
    # Since we are separate, we hope the training script saves eventually.
    # But training is still running.
    # We can try to load the "best_model.zip" if Monitor saved it? No, we didn't use EvalCallback.
    # We have to wait for training to finish OR stop it.
    pass

    # Alternative: Use PPO_17c (Profit Agent) on Feasible Data.
    PROFIT_MODEL = "models/ppo_intraday_profit"
    NORM_PATH = "models/vec_normalize_profit.pkl"
    
    print(f"Loading FEASIBLE DATA: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    
    # Battery Params
    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    def make_env():
        # Feasible Data
        e = BatteryTradingEnvV5(df, battery_params)
        return e
        
    env = DummyVecEnv([make_env])
    
    if os.path.exists(NORM_PATH):
        print(f"Loading Normalization Stats from {NORM_PATH}")
        env = VecNormalize.load(NORM_PATH, env)
        env.training = False
        env.norm_reward = False
    
    print(f"Loading Model from {PROFIT_MODEL}")
    model = PPO.load(PROFIT_MODEL)
    
    obs = env.reset()
    
    results = []
    
    print("Starting Profit Evaluation Loop...")
    
    # Run full episode
    done = False
    total_steps = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        info_dict = info[0]
        
        # Breakdown from Environment Info?
        # Env V5 info: {'req_mw', 'actual_mw', 'violation', 'penalty', 'reward'}
        # It doesn't give discrete financial breakdown (DAM Rev, Trade Rev).
        # We must recalculate or infer.
        
        row = df.iloc[total_steps]
        dam_comm = row['dam_commitment']
        price = row['price']
        
        actual_mw = info_dict['actual_mw']
        req_mw = info_dict['req_mw']
        violation = info_dict['violation']
        penalty = info_dict['penalty']
        
        # Recalculate Financials
        dam_rev = dam_comm * price
        
        trade_mw = actual_mw - dam_comm
        trade_rev = 0.0
        if trade_mw > 0: trade_rev = trade_mw * (price * 0.9)
        elif trade_mw < 0: trade_rev = trade_mw * (price * 1.1)
            
        total_pnl = dam_rev + trade_rev - penalty
        
        results.append({
            'step': total_steps,
            'price': price,
            'dam_comm': dam_comm,
            'actual_mw': actual_mw,
            'dam_revenue': dam_rev,
            'trade_revenue': trade_rev,
            'penalty': penalty,
            'total_pnl': total_pnl
        })
        
        total_steps += 1
        if done: break
            
    res_df = pd.DataFrame(results)
    
    total_dam_rev = res_df['dam_revenue'].sum()
    total_trade_rev = res_df['trade_revenue'].sum()
    total_penalty = res_df['penalty'].sum()
    net_profit = res_df['total_pnl'].sum()
    
    print("="*40)
    print("PROFITABILITY REPORT (Smart Agent on Feasible Data)")
    print("="*40)
    print(f"Total Steps: {total_steps}")
    print(f"Total DAM Revenue: {total_dam_rev:,.2f} EUR")
    print(f"Total Trading PnL: {total_trade_rev:,.2f} EUR")
    print(f"Total Penalties:   {total_penalty:,.2f} EUR")
    print("-" * 40)
    print(f"NET PROFIT:        {net_profit:,.2f} EUR")
    print("="*40)
    
    if net_profit > 0:
        print("SUCCESS: System is Profitable on Feasible Scenarios! 🚀")
    else:
        print("FAILURE: Still losing money. Check Penalties/Costs.")

if __name__ == "__main__":
    evaluate_profit_breakdown()
