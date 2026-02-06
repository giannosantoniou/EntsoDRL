"""
v21 Evaluation: Fix Data Leakage + Trader-Inspired Signals
Compare v21 with v15 (baseline)

I handle the observation space:
- v21: 44 features (include_price_awareness=True, backward-only stats)
- v15: 42 features (include_price_awareness=False)

Key difference from v20 eval: v21 uses backward-looking stats so the
environment no longer peeks at future prices. This makes evaluation
results more representative of real production performance.
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


def evaluate_model(model_path, vecnorm_path, test_df, battery_params,
                   model_name="Model", include_price_awareness=True):
    """
    Evaluate model with comprehensive metrics.

    I use include_price_awareness to match the observation space
    that was used during training for each model version.
    """
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
            forecaster_path="models/intraday_forecaster.pkl",
            include_price_awareness=include_price_awareness
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
        row = test_df.iloc[step]

        data.append({
            'step': step,
            'price': row.get('price', 0),
            'dam_commitment': row.get('dam_commitment', 0),
            'power': info_dict.get('actual_mw', 0),
            'profit': info_dict.get('net_profit', 0),
        })

        if done:
            obs = env.reset()

    df_data = pd.DataFrame(data)

    df_data['is_discharge'] = df_data['power'] > 0.5
    df_data['is_charge'] = df_data['power'] < -0.5
    df_data['is_idle'] = (~df_data['is_discharge']) & (~df_data['is_charge'])
    df_data['has_dam'] = abs(df_data['dam_commitment']) > 0.5
    df_data['is_intraday'] = ~df_data['has_dam']
    df_data['price_percentile'] = df_data['price'].rank(pct=True)

    metrics = {}
    metrics['total_profit'] = df_data['profit'].sum()
    metrics['avg_daily_profit'] = df_data.groupby(df_data.index // 24)['profit'].sum().mean()

    id_data = df_data[df_data['is_intraday']]
    id_discharge = id_data[id_data['is_discharge']]
    id_charge = id_data[id_data['is_charge']]

    metrics['discharge_profit'] = id_discharge['profit'].sum()
    metrics['charge_profit'] = id_charge['profit'].sum()

    q1_data = id_data[id_data['price_percentile'] < 0.25]
    q4_data = id_data[id_data['price_percentile'] > 0.75]

    metrics['q1_charge_rate'] = q1_data['is_charge'].mean() * 100 if len(q1_data) > 0 else 0
    metrics['q4_discharge_rate'] = q4_data['is_discharge'].mean() * 100 if len(q4_data) > 0 else 0
    metrics['q4_charge_rate'] = q4_data['is_charge'].mean() * 100 if len(q4_data) > 0 else 0

    id_buys = id_data[id_data['is_charge']]
    id_sells = id_data[id_data['is_discharge']]

    metrics['buys_in_q4_pct'] = (id_buys['price_percentile'] > 0.75).mean() * 100 if len(id_buys) > 0 else 0
    metrics['sells_in_q1_pct'] = (id_sells['price_percentile'] < 0.25).mean() * 100 if len(id_sells) > 0 else 0

    # I compute action distribution across all trades
    total = len(df_data)
    metrics['discharge_pct'] = df_data['is_discharge'].mean() * 100
    metrics['charge_pct'] = df_data['is_charge'].mean() * 100
    metrics['idle_pct'] = df_data['is_idle'].mean() * 100

    print(f"\n1. PROFIT")
    print(f"   Total: {metrics['total_profit']:,.0f} EUR")
    print(f"   Daily avg: {metrics['avg_daily_profit']:,.0f} EUR")
    print(f"   Discharge profit: {metrics['discharge_profit']:,.0f} EUR")
    print(f"   Charge profit: {metrics['charge_profit']:,.0f} EUR")

    print(f"\n2. ACTION DISTRIBUTION")
    print(f"   Discharge: {metrics['discharge_pct']:.1f}%")
    print(f"   Charge: {metrics['charge_pct']:.1f}%")
    print(f"   Idle: {metrics['idle_pct']:.1f}%")

    print(f"\n3. QUARTILE BEHAVIOR (IntraDay)")
    print(f"   Q1 (cheap): Charge {metrics['q1_charge_rate']:.1f}%")
    print(f"   Q4 (expensive): Discharge {metrics['q4_discharge_rate']:.1f}% | Charge {metrics['q4_charge_rate']:.1f}%")

    print(f"\n4. BAD TRADES")
    print(f"   Buys in Q4: {metrics['buys_in_q4_pct']:.1f}%")
    print(f"   Sells in Q1: {metrics['sells_in_q1_pct']:.1f}%")

    return metrics


def main():
    print("=" * 70)
    print("v21 vs v15 COMPARISON")
    print("  v21: Backward-only stats (no data leakage)")
    print("  v15: Baseline (42 features, no price awareness)")
    print("=" * 70)

    df = load_historical_data('data/feasible_data_with_balancing.csv')
    split_idx = int(len(df) * 0.85)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    print(f"Test data: {len(test_df):,} rows")

    battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}

    r = {}
    # v21: 44 features (include_price_awareness=True, backward stats)
    r['v21'] = evaluate_model(
        'models/ppo_v21_best.zip', 'models/vec_normalize_v21.pkl',
        test_df, battery_params, "v21 (Backward Stats + Trader Signals)",
        include_price_awareness=True
    )
    # v15: 42 features (include_price_awareness=False)
    r['v15'] = evaluate_model(
        'models/ppo_v15_best.zip', 'models/vec_normalize_v15.pkl',
        test_df, battery_params, "v15 (Baseline)",
        include_price_awareness=False
    )

    valid = {k: v for k, v in r.items() if v is not None}
    if len(valid) >= 2:
        print("\n" + "=" * 70)
        print("COMPARISON: v21 (no leakage) vs v15 (baseline)")
        print("=" * 70)
        print(f"\n{'Metric':<25} {'v21':>15} {'v15':>15} {'Delta':>15}")
        print("-" * 70)
        for metric, label in [
            ('total_profit', 'Total Profit'),
            ('avg_daily_profit', 'Daily Avg Profit'),
            ('discharge_profit', 'Discharge Profit'),
            ('charge_profit', 'Charge Profit'),
            ('q1_charge_rate', 'Q1 Charge %'),
            ('q4_discharge_rate', 'Q4 Discharge %'),
            ('buys_in_q4_pct', 'Buys in Q4 %'),
            ('sells_in_q1_pct', 'Sells in Q1 %'),
        ]:
            v21_val = valid.get('v21', {}).get(metric, 0)
            v15_val = valid.get('v15', {}).get(metric, 0)
            print(f"{label:<25} {v21_val:>15,.1f} {v15_val:>15,.1f} {v21_val-v15_val:>+15,.1f}")

    print("\n" + "=" * 70)
    print("NOTE: v21 profits may be lower than v20 because the agent")
    print("no longer has access to future prices. This is EXPECTED.")
    print("Production performance should be CLOSER to these numbers.")
    print("=" * 70)


if __name__ == "__main__":
    main()
