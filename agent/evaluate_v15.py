"""
v15 Evaluation: Realistic Imbalance Settlement

Compares v15 (trained with synthetic spread) vs v14 (trained without spread awareness).
Key metrics:
- Profit with realistic spread
- Spread costs (imbalance penalty)
- DAM compliance
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


def evaluate_model(model_path, vecnorm_path, test_df, battery_params, model_name="Model"):
    """Evaluate a model on test data with realistic spread."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        return None

    model = MaskablePPO.load(model_path)

    def make_env():
        env = BatteryEnvMasked(
            test_df, battery_params, n_actions=21,
            use_ml_forecaster=True,
            forecaster_path="models/intraday_forecaster.pkl"
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        return env

    env = DummyVecEnv([make_env])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    env.training = False
    env.norm_reward = False

    # Collect data
    obs = env.reset()
    data = []

    for step in range(min(5000, len(test_df) - 50)):
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

        data.append({
            'step': step,
            'hour': step % 24,
            'day': step // 24,
            'price': current_price,
            'power': info_dict.get('actual_mw', 0),
            'soc': info_dict.get('soc', 0.5),
            'dam_commitment': dam_commitment,
            'profit': info_dict.get('net_profit', 0),
            'spread': info_dict.get('spread', 0),
            'action': action[0],
        })

        if done:
            obs = env.reset()

    df_data = pd.DataFrame(data)
    df_data['is_discharge'] = df_data['power'] > 0.1
    df_data['is_charge'] = df_data['power'] < -0.1
    df_data['is_idle'] = ~df_data['is_discharge'] & ~df_data['is_charge']
    df_data['has_dam'] = abs(df_data['dam_commitment']) > 0.1

    # Metrics
    metrics = {}

    # 1. Overall
    metrics['total_profit'] = df_data['profit'].sum()
    metrics['avg_daily_profit'] = df_data.groupby('day')['profit'].sum().mean()
    metrics['avg_spread'] = df_data['spread'].mean()
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
        metrics['dam_compliance'] = (discharge_met + charge_met) / 2
    else:
        metrics['dam_compliance'] = 100

    # 3. Spread efficiency (lower spread when trading = better)
    trading_hours = df_data[~df_data['is_idle']]
    metrics['avg_spread_when_trading'] = trading_hours['spread'].mean() if len(trading_hours) > 0 else 0

    # Print
    print(f"\n1. PROFIT")
    print(f"   Total: {metrics['total_profit']:,.0f} EUR")
    print(f"   Daily avg: {metrics['avg_daily_profit']:,.0f} EUR")

    print(f"\n2. ACTIVITY")
    print(f"   Discharge: {metrics['discharge_pct']:.1f}% | Charge: {metrics['charge_pct']:.1f}% | Idle: {metrics['idle_pct']:.1f}%")

    print(f"\n3. DAM COMPLIANCE: {metrics['dam_compliance']:.1f}%")

    print(f"\n4. SPREAD COSTS")
    print(f"   Avg spread: {metrics['avg_spread']:.1f} EUR")
    print(f"   Avg spread when trading: {metrics['avg_spread_when_trading']:.1f} EUR")

    return metrics


def main():
    print("=" * 70)
    print("v15 vs v14 COMPARISON (Realistic Spread)")
    print("=" * 70)

    # Load test data
    df = load_historical_data('data/feasible_data_with_balancing.csv')
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    print(f"Test data: {len(test_df):,} rows")

    battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}

    results = {}

    # Evaluate v15
    results['v15'] = evaluate_model(
        'models/ppo_v15_best.zip',
        'models/vec_normalize_v15.pkl',
        test_df, battery_params,
        model_name="v15 (Spread-aware)"
    )

    # Evaluate v14
    results['v14'] = evaluate_model(
        'models/ppo_v14_best.zip',
        'models/vec_normalize_v14.pkl',
        test_df, battery_params,
        model_name="v14 (No spread training)"
    )

    # Comparison
    if results['v15'] and results['v14']:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<30} {'v15 (Spread)':<18} {'v14 (No Spread)':<18} {'Delta':<12}")
        print("-" * 78)

        for metric in ['total_profit', 'avg_daily_profit', 'dam_compliance', 'avg_spread_when_trading']:
            v15_val = results['v15'].get(metric, 0)
            v14_val = results['v14'].get(metric, 0)
            delta = v15_val - v14_val
            print(f"{metric:<30} {v15_val:<18,.1f} {v14_val:<18,.1f} {delta:+,.1f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
