"""
Detailed Loss Analysis for Unified Model

I analyze exactly where money is being lost:
- mFRR trading losses (buy high, sell low)
- Degradation costs
- aFRR penalties
- DAM violations
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO
from gym_envs.battery_env_unified import BatteryEnvUnified


def find_latest_model():
    """I find the latest unified model."""
    models_dir = project_root / "models" / "unified"
    model_dirs = sorted(models_dir.glob("unified_*"), reverse=True)

    for model_dir in model_dirs:
        best_model = model_dir / "best" / "best_model.zip"
        if best_model.exists():
            return best_model
    return None


def analyze_losses(n_episodes: int = 10):
    """I run detailed analysis of where money is lost."""

    model_path = find_latest_model()
    if not model_path:
        print("No model found!")
        return

    print(f"Loading model: {model_path}")

    # I load data and create environment
    data_path = project_root / "data" / "unified_multimarket_training.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    env = BatteryEnvUnified(
        df=df,
        episode_length=168,  # 1 week
        random_start=True
    )

    vec_env = DummyVecEnv([lambda: env])
    model = MaskablePPO.load(str(model_path), env=vec_env)

    # I track detailed statistics
    stats = defaultdict(list)

    # I track per-step details
    all_trades = []

    print(f"\nRunning {n_episodes} episodes with detailed tracking...\n")

    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        step = 0

        # I reset episode tracking
        ep_mfrr_profit = 0
        ep_mfrr_buy_cost = 0
        ep_mfrr_sell_revenue = 0
        ep_afrr_cap_revenue = 0
        ep_afrr_energy_revenue = 0
        ep_degradation_cost = 0
        ep_cycles = 0

        mfrr_buys = []
        mfrr_sells = []

        while not done:
            action_masks = vec_env.env_method("action_masks")[0]
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

            # I get pre-step state
            pre_soc = env.soc

            obs, reward, done, info = vec_env.step(action)
            info = info[0]

            post_soc = env.soc
            soc_change = post_soc - pre_soc

            # I track mFRR trades
            mfrr_mw = info.get('mfrr_energy_mw', 0)
            if abs(mfrr_mw) > 0.1:
                # I get prices from the environment's current row
                row = env.df.iloc[env.current_step - 1]

                if mfrr_mw > 0:  # Selling (discharging)
                    price = row.get('mfrr_price_up', 120)
                    revenue = abs(soc_change) * env.capacity_mwh * env.eff_sqrt * price
                    ep_mfrr_sell_revenue += revenue
                    mfrr_sells.append({'mw': mfrr_mw, 'price': price, 'revenue': revenue})
                else:  # Buying (charging)
                    price = row.get('mfrr_price_down', 60)
                    cost = abs(soc_change) * env.capacity_mwh / env.eff_sqrt * price
                    ep_mfrr_buy_cost += cost
                    mfrr_buys.append({'mw': mfrr_mw, 'price': price, 'cost': cost})

            # I track aFRR
            if info.get('is_selected', False):
                row = env.df.iloc[env.current_step - 1]
                cap_price = row.get('afrr_cap_up_price', 20)
                ep_afrr_cap_revenue += env.afrr_commitment_mw * cap_price

            if info.get('afrr_activated', False):
                ep_afrr_energy_revenue += info.get('afrr_energy_profit', 0)

            step += 1

        # I get final episode stats
        final_info = info
        ep_cycles = final_info.get('total_cycles', 0)
        ep_degradation_cost = ep_cycles * env.capacity_mwh * env.reward_calculator.degradation_cost

        ep_mfrr_profit = ep_mfrr_sell_revenue - ep_mfrr_buy_cost

        # I store episode stats
        stats['mfrr_sell_revenue'].append(ep_mfrr_sell_revenue)
        stats['mfrr_buy_cost'].append(ep_mfrr_buy_cost)
        stats['mfrr_net'].append(ep_mfrr_profit)
        stats['afrr_cap_revenue'].append(ep_afrr_cap_revenue)
        stats['afrr_energy_revenue'].append(ep_afrr_energy_revenue)
        stats['degradation_cost'].append(ep_degradation_cost)
        stats['cycles'].append(ep_cycles)
        stats['total_profit'].append(final_info.get('total_profit', 0))

        # I calculate average buy/sell prices
        if mfrr_buys:
            avg_buy_price = np.mean([t['price'] for t in mfrr_buys])
            stats['avg_buy_price'].append(avg_buy_price)
        if mfrr_sells:
            avg_sell_price = np.mean([t['price'] for t in mfrr_sells])
            stats['avg_sell_price'].append(avg_sell_price)

        print(f"Episode {ep+1}: mFRR net={ep_mfrr_profit:,.0f}, "
              f"Degradation={ep_degradation_cost:,.0f}, Cycles={ep_cycles:.2f}")

    vec_env.close()

    # I print detailed analysis
    print("\n" + "=" * 70)
    print("DETAILED LOSS ANALYSIS")
    print("=" * 70)

    print("\n[mFRR TRADING]")
    print("-" * 50)
    print(f"  Sell Revenue (discharge): {np.mean(stats['mfrr_sell_revenue']):>12,.0f} EUR")
    print(f"  Buy Cost (charge):        {np.mean(stats['mfrr_buy_cost']):>12,.0f} EUR")
    print(f"  Net mFRR Profit:          {np.mean(stats['mfrr_net']):>12,.0f} EUR")
    if stats['avg_buy_price'] and stats['avg_sell_price']:
        print(f"\n  Avg Buy Price:            {np.mean(stats['avg_buy_price']):>12,.1f} EUR/MWh")
        print(f"  Avg Sell Price:           {np.mean(stats['avg_sell_price']):>12,.1f} EUR/MWh")
        spread = np.mean(stats['avg_sell_price']) - np.mean(stats['avg_buy_price'])
        print(f"  Price Spread:             {spread:>12,.1f} EUR/MWh")
        if spread < 0:
            print(f"  [PROBLEM] Buying HIGH, selling LOW!")

    print("\n[aFRR REVENUE]")
    print("-" * 50)
    print(f"  Capacity Revenue:         {np.mean(stats['afrr_cap_revenue']):>12,.0f} EUR")
    print(f"  Energy Revenue:           {np.mean(stats['afrr_energy_revenue']):>12,.0f} EUR")

    print("\n[COSTS]")
    print("-" * 50)
    print(f"  Degradation Cost:         {np.mean(stats['degradation_cost']):>12,.0f} EUR")
    print(f"  Avg Cycles:               {np.mean(stats['cycles']):>12.2f}")
    print(f"  Degradation Rate:         {env.reward_calculator.degradation_cost:>12.1f} EUR/MWh")

    print("\n[SUMMARY]")
    print("-" * 50)
    total_revenue = (np.mean(stats['mfrr_net']) +
                    np.mean(stats['afrr_cap_revenue']) +
                    np.mean(stats['afrr_energy_revenue']))
    total_cost = np.mean(stats['degradation_cost'])
    calculated_profit = total_revenue - total_cost
    reported_profit = np.mean(stats['total_profit'])

    print(f"  Total Revenue:            {total_revenue:>12,.0f} EUR")
    print(f"  Total Cost:               {total_cost:>12,.0f} EUR")
    print(f"  Calculated Profit:        {calculated_profit:>12,.0f} EUR")
    print(f"  Reported Profit:          {reported_profit:>12,.0f} EUR")
    print(f"  Discrepancy:              {reported_profit - calculated_profit:>12,.0f} EUR")

    if reported_profit < calculated_profit:
        print(f"\n  [INFO] Discrepancy likely from reward scaling or penalties")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_losses(n_episodes=10)
