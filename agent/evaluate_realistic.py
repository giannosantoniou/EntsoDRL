"""
Evaluate Realistic DRL Agent - Compare with Time-Based Strategy

I compare the trained realistic agent (no data leakage) against
rule-based strategies on unseen test data.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_envs.battery_env_realistic import BatteryEnvRealistic
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env):
    """Return action mask for MaskablePPO."""
    return env.action_masks()


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load and prepare data."""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "mfrr_features.csv"

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(str(data_path), index_col=0, parse_dates=True)
    df = df.sort_index()

    if 'price' not in df.columns and 'mfrr_price_up' in df.columns:
        df['price'] = df['mfrr_price_up']

    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    return df


def create_test_env(df: pd.DataFrame):
    """Create test environment."""
    env = BatteryEnvRealistic(
        df=df,
        capacity_mwh=146.0,
        max_power_mw=30.0,
        efficiency=0.94,
        degradation_cost=15.0,
        time_step_hours=1.0,
        n_actions=21,
        reward_scale=0.001,
    )
    env = ActionMasker(env, mask_fn)
    return env


def evaluate_drl_agent(model, vec_normalize, test_df: pd.DataFrame,
                       max_steps: int = None) -> dict:
    """Evaluate DRL agent on test data."""
    print("\nEvaluating DRL Agent...")

    # I create a fresh test environment
    env = create_test_env(test_df)
    vec_env = DummyVecEnv([lambda: env])

    # I wrap with the trained VecNormalize (inference mode)
    vec_env = VecNormalize.load(vec_normalize, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()

    total_profit = 0.0
    total_cycles = 0.0
    discharge_hours = 0
    charge_hours = 0
    discharge_revenue = 0.0
    charge_cost = 0.0
    discharge_prices = []
    charge_prices = []
    actions_taken = []

    steps = 0
    max_steps = max_steps or len(test_df) - 25

    while steps < max_steps:
        masks = np.array([vec_env.envs[0].action_masks()])
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        info = info[0]
        power_mw = info.get('power_mw', 0.0)
        price = info.get('price', 0.0)
        profit = info.get('profit', 0.0)

        actions_taken.append(action[0])

        if power_mw > 0.5:  # Discharge
            discharge_hours += 1
            discharge_revenue += profit
            discharge_prices.append(price)
        elif power_mw < -0.5:  # Charge
            charge_hours += 1
            charge_cost += abs(profit)
            charge_prices.append(price)

        total_profit = info.get('total_profit', total_profit + profit)
        total_cycles = info.get('cycles', total_cycles)

        steps += 1

        if done[0]:
            break

    # I calculate statistics
    avg_discharge_price = np.mean(discharge_prices) if discharge_prices else 0
    avg_charge_price = np.mean(charge_prices) if charge_prices else 0

    results = {
        'strategy': 'Realistic DRL Agent',
        'steps': steps,
        'total_profit': total_profit,
        'total_cycles': total_cycles,
        'profit_per_cycle': total_profit / total_cycles if total_cycles > 0 else 0,
        'discharge_hours': discharge_hours,
        'charge_hours': charge_hours,
        'idle_hours': steps - discharge_hours - charge_hours,
        'avg_discharge_price': avg_discharge_price,
        'avg_charge_price': avg_charge_price,
        'price_spread': avg_discharge_price - avg_charge_price,
        'action_distribution': pd.Series(actions_taken).value_counts().to_dict(),
    }

    return results


def evaluate_time_based(test_df: pd.DataFrame, max_steps: int = None) -> dict:
    """Evaluate time-based strategy (no future info)."""
    print("\nEvaluating Time-Based Strategy...")

    # I create environment
    env = create_test_env(test_df)
    env = env.env  # I unwrap ActionMasker to get base env

    obs, info = env.reset()

    # I define action mapping (21 actions: 0=full charge, 10=idle, 20=full discharge)
    ACTION_IDLE = 10
    ACTION_DISCHARGE_FULL = 20
    ACTION_DISCHARGE_HALF = 15
    ACTION_CHARGE_FULL = 0
    ACTION_CHARGE_HALF = 5

    total_profit = 0.0
    total_cycles = 0.0
    discharge_hours = 0
    charge_hours = 0
    discharge_prices = []
    charge_prices = []

    steps = 0
    max_steps = max_steps or len(test_df) - 25

    while steps < max_steps:
        ts = test_df.index[env.current_step]
        hour = ts.hour
        dow = ts.dayofweek
        is_weekend = dow >= 5

        mask = env.action_masks()

        # I apply time-based logic
        weekend_factor = 0.6 if is_weekend else 1.0

        if 17 <= hour <= 21:  # Evening peak - discharge
            if is_weekend:
                action = ACTION_DISCHARGE_HALF if mask[ACTION_DISCHARGE_HALF] else ACTION_IDLE
            else:
                action = ACTION_DISCHARGE_FULL if mask[ACTION_DISCHARGE_FULL] else ACTION_IDLE
        elif 6 <= hour <= 8:  # Morning ramp - moderate discharge
            action = ACTION_DISCHARGE_HALF if mask[ACTION_DISCHARGE_HALF] else ACTION_IDLE
        elif 10 <= hour <= 14:  # Solar hours - charge
            if is_weekend:
                action = ACTION_CHARGE_HALF if mask[ACTION_CHARGE_HALF] else ACTION_IDLE
            else:
                action = ACTION_CHARGE_FULL if mask[ACTION_CHARGE_FULL] else ACTION_IDLE
        else:
            action = ACTION_IDLE

        obs, reward, terminated, truncated, info = env.step(action)

        power_mw = info.get('power_mw', 0.0)
        price = info.get('price', 0.0)
        profit = info.get('profit', 0.0)

        if power_mw > 0.5:
            discharge_hours += 1
            discharge_prices.append(price)
        elif power_mw < -0.5:
            charge_hours += 1
            charge_prices.append(price)

        total_profit = info.get('total_profit', total_profit + profit)
        total_cycles = info.get('cycles', total_cycles)

        steps += 1

        if terminated or truncated:
            break

    avg_discharge_price = np.mean(discharge_prices) if discharge_prices else 0
    avg_charge_price = np.mean(charge_prices) if charge_prices else 0

    results = {
        'strategy': 'Time-Based',
        'steps': steps,
        'total_profit': total_profit,
        'total_cycles': total_cycles,
        'profit_per_cycle': total_profit / total_cycles if total_cycles > 0 else 0,
        'discharge_hours': discharge_hours,
        'charge_hours': charge_hours,
        'idle_hours': steps - discharge_hours - charge_hours,
        'avg_discharge_price': avg_discharge_price,
        'avg_charge_price': avg_charge_price,
        'price_spread': avg_discharge_price - avg_charge_price,
    }

    return results


def evaluate_hold_baseline(test_df: pd.DataFrame, max_steps: int = None) -> dict:
    """Evaluate hold (idle) baseline."""
    print("\nEvaluating Hold Baseline...")

    env = create_test_env(test_df)
    env = env.env

    obs, info = env.reset()

    ACTION_IDLE = 10
    total_profit = 0.0

    steps = 0
    max_steps = max_steps or len(test_df) - 25

    while steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(ACTION_IDLE)
        total_profit = info.get('total_profit', 0.0)
        steps += 1
        if terminated or truncated:
            break

    return {
        'strategy': 'Hold (Baseline)',
        'steps': steps,
        'total_profit': total_profit,
        'total_cycles': 0.0,
        'profit_per_cycle': 0.0,
        'discharge_hours': 0,
        'charge_hours': 0,
        'idle_hours': steps,
        'avg_discharge_price': 0,
        'avg_charge_price': 0,
        'price_spread': 0,
    }


def main():
    """Run comprehensive evaluation."""
    print("=" * 70)
    print("REALISTIC DRL EVALUATION (NO DATA LEAKAGE)")
    print("=" * 70)

    # I load data
    df = load_data()

    # I use 80/20 split (same as training)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    print(f"\nTest Period: {test_df.index.min().date()} to {test_df.index.max().date()}")
    print(f"Test Hours: {len(test_df)}")

    # I load trained model
    model_path = Path(__file__).parent.parent / "models" / "realistic_ppo_20260207_081917"
    model_file = model_path / "realistic_ppo_final.zip"
    vec_normalize_file = model_path / "vec_normalize.pkl"

    if not model_file.exists():
        print(f"\nError: Model not found at {model_file}")
        return

    print(f"\nLoading model from {model_file}...")
    model = MaskablePPO.load(str(model_file))

    # I evaluate all strategies
    results = []

    # 1. Baseline (Hold)
    results.append(evaluate_hold_baseline(test_df))

    # 2. Time-Based
    results.append(evaluate_time_based(test_df))

    # 3. DRL Agent
    results.append(evaluate_drl_agent(model, str(vec_normalize_file), test_df))

    # I create comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    comparison = pd.DataFrame(results)

    # I display key metrics
    display_cols = ['strategy', 'total_profit', 'total_cycles', 'profit_per_cycle',
                    'discharge_hours', 'charge_hours', 'avg_discharge_price',
                    'avg_charge_price', 'price_spread']

    print(f"\nTest Period: {len(test_df)} hours\n")

    for _, row in comparison.iterrows():
        print(f"\n{row['strategy']}")
        print("-" * 40)
        print(f"  Total Profit:       {row['total_profit']:>12,.0f} EUR")
        print(f"  Total Cycles:       {row['total_cycles']:>12.1f}")
        print(f"  Profit/Cycle:       {row['profit_per_cycle']:>12,.0f} EUR")
        print(f"  Discharge Hours:    {row['discharge_hours']:>12}")
        print(f"  Charge Hours:       {row['charge_hours']:>12}")
        print(f"  Avg Discharge Price:{row['avg_discharge_price']:>12.1f} EUR/MWh")
        print(f"  Avg Charge Price:   {row['avg_charge_price']:>12.1f} EUR/MWh")
        print(f"  Price Spread:       {row['price_spread']:>12.1f} EUR/MWh")

    # I calculate annualized projections
    print("\n" + "=" * 70)
    print("ANNUALIZED PROJECTIONS (Realistic)")
    print("=" * 70)

    hours_year = 8760
    test_hours = len(test_df)

    print(f"\nBased on {test_hours} hours of test data:\n")
    print(f"{'Strategy':<25} {'Test Profit':>15} {'Annual Est.':>15}")
    print("-" * 55)

    for _, row in comparison.iterrows():
        annual = row['total_profit'] * hours_year / row['steps']
        print(f"{row['strategy']:<25} {row['total_profit']:>15,.0f} {annual:>15,.0f}")

    # I find winner
    best = comparison.loc[comparison['total_profit'].idxmax()]
    annual = best['total_profit'] * hours_year / best['steps']

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
WINNER: {best['strategy']}

Performance ({best['steps']} hours):
  - Net Profit:       {best['total_profit']:>12,.0f} EUR
  - Full Cycles:      {best['total_cycles']:>12.1f}
  - Profit per Cycle: {best['profit_per_cycle']:>12,.0f} EUR

Realistic Annualized:
  - Annual Profit:    {annual:>12,.0f} EUR
  - Monthly Average:  {annual/12:>12,.0f} EUR
  - Daily Average:    {annual/365:>12,.0f} EUR

NOTE: These are REALISTIC estimates without data leakage.
      Previous "8.9M EUR/year" was inflated by lookahead bias.
""")

    # I check DRL action distribution
    drl_result = [r for r in results if 'DRL' in r['strategy']][0]
    if 'action_distribution' in drl_result:
        print("\nDRL Agent Action Distribution:")
        dist = drl_result['action_distribution']
        for action, count in sorted(dist.items()):
            pct = count / sum(dist.values()) * 100
            action_name = 'IDLE' if action == 10 else ('CHARGE' if action < 10 else 'DISCHARGE')
            print(f"  Action {action:2d} ({action_name}): {count:>5} ({pct:>5.1f}%)")

    return comparison


if __name__ == "__main__":
    main()
