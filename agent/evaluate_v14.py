"""
v14 Evaluation: Realistic Price Forecasts

I compare v14 (realistic forecasts) with v13 (cheating forecasts).
Key metrics:
- Profit performance
- DAM compliance
- IntraDay trading behavior
- Timing quality
"""
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import os
sys.path.append('.')
from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data


def evaluate_model(model_path, vecnorm_path, test_df, battery_params, use_ml_forecaster=False, model_name="Model"):
    """
    I evaluate a model on test data and return detailed metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model = MaskablePPO.load(model_path)

    def make_env():
        env = BatteryEnvMasked(
            test_df, battery_params, n_actions=21,
            use_ml_forecaster=use_ml_forecaster,
            forecaster_path="models/intraday_forecaster.pkl"
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    # Collect data
    obs = env.reset()
    data = []

    for step in range(min(8000, len(test_df) - 50)):
        inner_env = env.envs[0]
        while hasattr(inner_env, 'env'):
            if hasattr(inner_env, 'action_masks'):
                break
            inner_env = inner_env.env
        mask = inner_env.action_masks() if hasattr(inner_env, 'action_masks') else None

        action, _ = model.predict(obs, deterministic=True, action_masks=np.array([mask]) if mask is not None else None)
        obs, reward, done, info = env.step(action)

        info_dict = info[0]
        current_price = test_df.iloc[step].get('price', 0)
        dam_commitment = test_df.iloc[step].get('dam_commitment', 0)

        # Get future info for timing analysis
        future_prices = []
        for i in range(1, 13):
            if step + i < len(test_df):
                future_prices.append(test_df.iloc[step + i].get('price', current_price))

        max_future = max(future_prices) if future_prices else current_price
        min_future = min(future_prices) if future_prices else current_price

        data.append({
            'step': step,
            'hour': step % 24,
            'day': step // 24,
            'price': current_price,
            'max_future_12h': max_future,
            'min_future_12h': min_future,
            'price_upside': max_future - current_price,
            'price_downside': current_price - min_future,
            'power': info_dict.get('actual_mw', 0),
            'soc': info_dict.get('soc', 0.5),
            'dam_commitment': dam_commitment,
            'profit': info_dict.get('net_profit', 0),
            'action': action[0],
        })

        if done:
            obs = env.reset()

    df_data = pd.DataFrame(data)
    df_data['is_discharge'] = df_data['power'] > 0.1
    df_data['is_charge'] = df_data['power'] < -0.1
    df_data['is_idle'] = ~df_data['is_discharge'] & ~df_data['is_charge']
    df_data['has_dam'] = abs(df_data['dam_commitment']) > 0.1

    # Calculate metrics
    metrics = {}

    # 1. Overall metrics
    metrics['total_profit'] = df_data['profit'].sum()
    metrics['avg_daily_profit'] = df_data.groupby('day')['profit'].sum().mean()
    metrics['discharge_pct'] = df_data['is_discharge'].mean() * 100
    metrics['charge_pct'] = df_data['is_charge'].mean() * 100
    metrics['idle_pct'] = df_data['is_idle'].mean() * 100

    # 2. DAM compliance
    dam_hours = df_data[df_data['has_dam']]
    if len(dam_hours) > 0:
        discharge_commits = dam_hours[dam_hours['dam_commitment'] > 0]
        charge_commits = dam_hours[dam_hours['dam_commitment'] < 0]

        discharge_met = (discharge_commits['power'] > 0).mean() * 100 if len(discharge_commits) > 0 else 100
        charge_met = (charge_commits['power'] < 0).mean() * 100 if len(charge_commits) > 0 else 100

        metrics['dam_discharge_compliance'] = discharge_met
        metrics['dam_charge_compliance'] = charge_met
        metrics['dam_total_compliance'] = (len(discharge_commits) * discharge_met + len(charge_commits) * charge_met) / max(len(dam_hours), 1)
    else:
        metrics['dam_discharge_compliance'] = 100
        metrics['dam_charge_compliance'] = 100
        metrics['dam_total_compliance'] = 100

    # 3. Timing analysis (IntraDay)
    free_sells = df_data[(df_data['is_discharge']) & (~df_data['has_dam'])]
    free_buys = df_data[(df_data['is_charge']) & (~df_data['has_dam'])]

    if len(free_sells) > 0:
        good_timing_sells = (free_sells['price_upside'] <= 10).mean() * 100
        bad_timing_sells = (free_sells['price_upside'] > 40).mean() * 100
        metrics['sell_good_timing'] = good_timing_sells
        metrics['sell_bad_timing'] = bad_timing_sells
    else:
        metrics['sell_good_timing'] = 0
        metrics['sell_bad_timing'] = 0

    if len(free_buys) > 0:
        good_timing_buys = (free_buys['price_downside'] <= 10).mean() * 100
        bad_timing_buys = (free_buys['price_downside'] > 40).mean() * 100
        metrics['buy_good_timing'] = good_timing_buys
        metrics['buy_bad_timing'] = bad_timing_buys
    else:
        metrics['buy_good_timing'] = 0
        metrics['buy_bad_timing'] = 0

    # 4. Price quartile behavior
    df_valid = df_data[df_data['price'] > 1].copy()
    q1, q2, q3 = df_valid['price'].quantile([0.25, 0.5, 0.75])

    q4_data = df_valid[df_valid['price'] > q3]
    q1_data = df_valid[df_valid['price'] <= q1]

    metrics['q4_discharge'] = q4_data['is_discharge'].mean() * 100 if len(q4_data) > 0 else 0
    metrics['q1_charge'] = q1_data['is_charge'].mean() * 100 if len(q1_data) > 0 else 0

    # Print results
    print(f"\n1. OVERALL PERFORMANCE")
    print(f"   Total Profit: {metrics['total_profit']:,.0f} EUR")
    print(f"   Avg Daily: {metrics['avg_daily_profit']:,.0f} EUR")
    print(f"   Discharge: {metrics['discharge_pct']:.1f}% | Charge: {metrics['charge_pct']:.1f}% | Idle: {metrics['idle_pct']:.1f}%")

    print(f"\n2. DAM COMPLIANCE")
    print(f"   Discharge: {metrics['dam_discharge_compliance']:.1f}%")
    print(f"   Charge: {metrics['dam_charge_compliance']:.1f}%")
    print(f"   Overall: {metrics['dam_total_compliance']:.1f}%")

    print(f"\n3. TIMING QUALITY (IntraDay)")
    print(f"   Sells - Good: {metrics['sell_good_timing']:.1f}% | Bad: {metrics['sell_bad_timing']:.1f}%")
    print(f"   Buys  - Good: {metrics['buy_good_timing']:.1f}% | Bad: {metrics['buy_bad_timing']:.1f}%")

    print(f"\n4. PRICE SENSITIVITY")
    print(f"   Q4 (expensive) Discharge: {metrics['q4_discharge']:.1f}%")
    print(f"   Q1 (cheap) Charge: {metrics['q1_charge']:.1f}%")

    return metrics


def main():
    print("="*70)
    print("v14 vs v13 COMPARISON")
    print("="*70)

    # Load test data
    df = load_historical_data('data/feasible_data_with_balancing.csv')
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}

    results = {}

    # Evaluate v14 (if exists)
    if os.path.exists('models/ppo_v14_best.zip'):
        results['v14'] = evaluate_model(
            'models/ppo_v14_best.zip',
            'models/vec_normalize_v14.pkl',
            test_df, battery_params,
            use_ml_forecaster=True,
            model_name="v14 (ML Forecaster)"
        )

    # Evaluate v13 (if exists)
    if os.path.exists('models/ppo_v13_best.zip'):
        results['v13'] = evaluate_model(
            'models/ppo_v13_best.zip',
            'models/vec_normalize_v13.pkl',
            test_df, battery_params,
            use_ml_forecaster=False,  # v13 was trained without ML forecaster
            model_name="v13 (Cheating Forecast)"
        )

    # Comparison
    if len(results) >= 2:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Metric':<30} {'v14 (Realistic)':<20} {'v13 (Cheating)':<20} {'Δ':<15}")
        print("-"*85)

        for metric in ['total_profit', 'avg_daily_profit', 'dam_total_compliance',
                       'sell_bad_timing', 'buy_bad_timing', 'q4_discharge', 'q1_charge']:
            v14_val = results['v14'].get(metric, 0)
            v13_val = results['v13'].get(metric, 0)
            delta = v14_val - v13_val
            delta_str = f"{delta:+.1f}"
            print(f"{metric:<30} {v14_val:<20.1f} {v13_val:<20.1f} {delta_str:<15}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
