
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_v5 import BatteryTradingEnvV5
from data.data_loader import load_historical_data

def audit_learning():
    # CONFIG
    DATA_PATH = "data/feasible_data_with_dam.csv"
    MODEL_PATH = "models/ppo_intraday_profit" 
    NORM_PATH = "models/vec_normalize_profit.pkl"
    OUTPUT_FILE = "learning_audit_stats.txt"
    
    print(f"Loading Data: {DATA_PATH}")
    df = load_historical_data(DATA_PATH)
    df = df.reset_index(drop=True)
    
    # Calculate Price Quantiles for "Cheap" and "Expensive" definitions
    price_low_threshold = df['price'].quantile(0.25)
    price_high_threshold = df['price'].quantile(0.75)
    print(f"Price Thresholds: Cheap < {price_low_threshold:.2f} | Expensive > {price_high_threshold:.2f}")

    battery_params = {
        'capacity_mwh': 50.0,
        'max_discharge_mw': 50.0,
        'efficiency': 0.94,
    }
    
    def make_env():
        return BatteryTradingEnvV5(df, battery_params)
        
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(NORM_PATH, env)
    env.training = False 
    env.norm_reward = False
    
    model = PPO.load(MODEL_PATH)
    
    # Run a full episode (or large chunk)
    steps_to_audit = min(len(df) - 50, 10000) # Audit 10k steps for speed
    
    obs = env.reset()
    
    # METRICS
    laziness_count = 0 # Didn't charge when cheap
    cheap_opportunities = 0
    
    panic_count = 0 # Bought when expensive
    expensive_moments = 0
    
    soc_history = []
    price_history = []
    
    print(f"Auditing {steps_to_audit} steps...")
    
    internal_env = env.envs[0]
    
    for i in range(steps_to_audit):
        current_soc = internal_env.current_soc
        current_step = internal_env.current_step
        row = df.iloc[current_step]
        price = row['price']
        dam = row['dam_commitment']
        
        # Action
        raw_obs = internal_env._get_observation()
        norm_obs = env.normalize_obs(raw_obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        
        # Step
        # Wrap action for DummyVecEnv
        _, _, done, info = env.step([action])
        
        # Analyze Decision (using info which reflects result of action)
        actual_mw = info[0]['actual_mw']
        
        # 1. LAZINESS (Timidity)
        # Condition: Price is Low AND SoC < 90% AND Battery is NOT Charging
        if price < price_low_threshold:
            cheap_opportunities += 1
            if current_soc < 0.90 and actual_mw >= 0: # >0 means Discharging or Idle. <0 means Charging.
                 laziness_count += 1
                 
        # 2. PANIC (Bad Planning)
        # Condition: Price is High AND Buying from Grid (actual_mw < 0)
        # Note: Buying from Grid is actual_mw < 0.
        # Wait, usually Dispatch > 0 is Discharge (Sell), Dispatch < 0 is Charge (Buy).
        # In this Env, action > 0 is Discharge.
        # Panic Buying = Buying when Price is High.
        if price > price_high_threshold:
            expensive_moments += 1
            if actual_mw < -0.1: # Significant buying
                # Check if it was FORCED by DAM?
                # Panic is buying *regardless* of reason, but if DAM forced it, it's "Forced Panic".
                # If DAM was 0 and we bought, that's "Stupid Panic".
                # Let's count all High Price Buying as "Panic/Forced"
                panic_count += 1

        soc_history.append(current_soc)
        price_history.append(price)
        
        if done: break
        
    # STATS
    laziness_rate = (laziness_count / cheap_opportunities * 100) if cheap_opportunities > 0 else 0
    panic_rate = (panic_count / expensive_moments * 100) if expensive_moments > 0 else 0
    
    # Inventory Correlation
    # We want correlation between SoC and Price.
    # Ideally: Negative? Charge(High SoC) when Price Low?
    # No, we want High SoC *when* Price is High (to sell).
    # So we want positive correlation between SoC and Price.
    soc_arr = np.array(soc_history)
    price_arr = np.array(price_history)
    corr, _ = pearsonr(soc_arr, price_arr)
    
    report = f"""
AUDIT RESULTS (Steps: {steps_to_audit})
---------------------------------------
1. TIMIDITY (Laziness Score): {laziness_rate:.1f}%
   - Defined as: Opportunities to charge (Price < {price_low_threshold:.1f}€) where agent sat idle or discharged.
   - Interpretation: High % means agent is "Afraid to Charge".

2. POOR PLANNING (Panic Score): {panic_rate:.1f}%
   - Defined as: Moments of High Price (> {price_high_threshold:.1f}€) where agent was BUYING.
   - Interpretation: High % means agent ran out of energy and was forced to buy at peak.

3. INVENTORY IQ (SoC-Price Correlation): {corr:.2f}
   - Goal: Positive (High SoC during High Prices).
   - Interpretation: If near 0 or negative, agent ignores market timing.
    """
    
    print(report)
    with open(OUTPUT_FILE, "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    audit_learning()
