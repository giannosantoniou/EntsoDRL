"""
Detailed SAC Evaluation Script — Per-Step Decision Logging

I run a trained SAC model for N episodes and produce:
1. Per-step CSV with ALL decision variables, market conditions, and financial results
2. Summary report with per-market P&L, SoC trajectory, action distribution
3. Top-10 best/worst steps for pattern diagnosis

Usage:
    python agent/evaluate_sac_detailed.py \
        --model models/unified_sac_v7_imbalance/sac_*/best/best_model.zip \
        --data data/unified_multimarket_training_v5_raw_features.csv \
        --episodes 4 --output evaluation_results/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gym_envs.battery_env_unified import BatteryEnvUnified
from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC


def create_eval_env(df, time_step_hours=0.25):
    """I create a single SAC env for evaluation (deterministic, no noise)."""
    inner = BatteryEnvUnified(
        df=df,
        episode_length=672,  # 1 week at 15-min
        random_start=False,
        forecast_noise=False,
        time_step_hours=time_step_hours,
        enable_endogenous_dam=True,
        dam_bidder_min_spread=5.0,
        mfrr_activation_rate=0.05,
    )
    penalties = {'soc_soft_margin': 60.0, 'cycle_excess': 2000.0}
    env = BatteryEnvUnifiedSAC(inner_env=inner, penalties=penalties)
    return env


def run_episode(env, model=None, use_vecnorm=False, vec_env=None, episode_idx=0):
    """I run one episode and collect per-step info dicts.

    If model is None, I run IDLE strategy (baseline).
    """
    if use_vecnorm and vec_env is not None:
        return _run_episode_vecnorm(vec_env, model, episode_idx)

    obs, _ = env.reset(seed=42 + episode_idx)
    infos = []

    for step in range(672):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, 0.0, 0.0, 0.0])  # IDLE

        obs, reward, terminated, truncated, info = env.step(action)
        info['episode'] = episode_idx
        info['step_in_episode'] = step
        infos.append(info)

        if terminated or truncated:
            break

    return infos


def _run_episode_vecnorm(vec_env, model, episode_idx):
    """I run episode through VecNormalize for proper observation scaling."""
    obs = vec_env.reset()
    infos = []

    for step in range(672):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info_list = vec_env.step(action)
        info = info_list[0]
        info['episode'] = episode_idx
        info['step_in_episode'] = step
        infos.append(info)

        if dones[0]:
            break

    return infos


def infos_to_dataframe(all_infos):
    """I convert list of info dicts to DataFrame, flattening nested dicts."""
    rows = []
    for info in all_infos:
        row = {}
        for k, v in info.items():
            if isinstance(v, dict):
                # I flatten nested dicts with prefix
                for k2, v2 in v.items():
                    row[f'{k}_{k2}'] = v2
            elif isinstance(v, (np.floating, np.integer)):
                row[k] = float(v)
            elif isinstance(v, np.ndarray):
                for i, val in enumerate(v):
                    row[f'{k}_{i}'] = float(val)
            else:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary_report(df, output_dir=None):
    """I print a comprehensive summary of the evaluation."""
    n_episodes = df['episode'].nunique()
    n_steps = len(df)

    print("\n" + "=" * 80)
    print("  DETAILED EVALUATION REPORT")
    print("=" * 80)

    # 1. Overview
    total_reward = df['reward'].sum()
    total_profit = df['step_profit'].sum()
    print(f"\n  Episodes: {n_episodes}")
    print(f"  Steps: {n_steps} ({n_steps/96:.1f} days)")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"  Total profit: {total_profit:.0f} EUR")
    print(f"  EUR/day: {total_profit / (n_steps / 96):.0f}")

    # 2. Per-market P&L
    print(f"\n--- Per-Market P&L ---")
    markets = {
        'DAM': 'step_dam_revenue',
        'aFRR cap': 'step_afrr_cap_revenue',
        'aFRR energy': 'step_afrr_energy_revenue',
        'XBID': 'step_xbid_revenue',
        'mFRR': 'step_mfrr_revenue',
    }
    costs = {
        'Degradation': 'step_degradation',
        'Network': 'step_network_cost',
        'Calendar aging': 'step_calendar_aging',
        'Inventory MTM': 'step_inventory_cost',
        'Imbalance': 'step_imbalance_cost',
    }

    print(f"\n  {'Market':<18} {'Total EUR':>12} {'EUR/day':>10} {'EUR/step':>10}")
    print("  " + "-" * 52)
    gross = 0
    for name, col in markets.items():
        if col in df.columns:
            total = df[col].sum()
            gross += total
            per_day = total / (n_steps / 96)
            print(f"  {name:<18} {total:>12.0f} {per_day:>10.0f} {total/n_steps:>10.2f}")
    print(f"  {'GROSS REVENUE':<18} {gross:>12.0f} {gross/(n_steps/96):>10.0f}")

    print()
    total_costs = 0
    for name, col in costs.items():
        if col in df.columns:
            total = df[col].sum()
            total_costs += total
            per_day = total / (n_steps / 96)
            print(f"  {name:<18} {total:>12.0f} {per_day:>10.0f} {total/n_steps:>10.2f}")
    print(f"  {'TOTAL COSTS':<18} {total_costs:>12.0f} {total_costs/(n_steps/96):>10.0f}")
    print(f"  {'NET PROFIT':<18} {gross-total_costs:>12.0f} {(gross-total_costs)/(n_steps/96):>10.0f}")

    # 3. SoC statistics
    print(f"\n--- SoC Statistics ---")
    print(f"  Mean: {df['soc'].mean():.3f}")
    print(f"  Min:  {df['soc'].min():.3f}")
    print(f"  Max:  {df['soc'].max():.3f}")
    print(f"  Std:  {df['soc'].std():.3f}")

    # 4. Imbalance stats
    if 'shortfall_mwh' in df.columns:
        shortfall_steps = (df['shortfall_mwh'] > 0.01).sum()
        print(f"\n--- Imbalance Stats ---")
        print(f"  Steps with shortfall: {shortfall_steps} ({shortfall_steps/n_steps*100:.1f}%)")
        print(f"  Total shortfall: {df['shortfall_mwh'].sum():.1f} MWh")
        print(f"  Total imbalance cost: {df['step_imbalance_cost'].sum():.0f} EUR")
        print(f"  Mean delivery ratio: {df['delivery_ratio'].mean():.4f}")

    # 5. Action distribution
    print(f"\n--- Action Distribution ---")
    if 'raw_action_0_afrr_commit' in df.columns:
        for i, name in enumerate(['afrr_commit', 'afrr_price', 'xbid_qty', 'xbid_price']):
            col = f'raw_action_{i}_{name}'
            if col in df.columns:
                a = df[col]
                print(f"  {name:<16}: mean={a.mean():+.3f}  std={a.std():.3f}  "
                      f"[{a.min():+.3f}, {a.max():+.3f}]")

    # I compute participation rates
    if 'afrr_commitment_mw' in df.columns:
        afrr_active = (df['afrr_commitment_mw'] > 0.5).mean()
        print(f"\n  aFRR participation: {afrr_active*100:.1f}%")
    if 'afrr_is_selected' in df.columns:
        afrr_sel = df['afrr_is_selected'].mean()
        print(f"  aFRR selection rate: {afrr_sel*100:.1f}%")
    if 'xbid_mw' in df.columns:
        xbid_sell = (df['xbid_mw'] > 0.5).mean()
        xbid_buy = (df['xbid_mw'] < -0.5).mean()
        xbid_idle = 1 - xbid_sell - xbid_buy
        print(f"  XBID sell: {xbid_sell*100:.1f}%  buy: {xbid_buy*100:.1f}%  idle: {xbid_idle*100:.1f}%")

    # 6. Top-10 best/worst steps
    print(f"\n--- Top-10 Best Steps ---")
    print(f"  {'Step':>5} {'Episode':>4} {'SoC':>6} {'Net_MW':>8} {'Profit':>10} {'Timestamp'}")
    top = df.nlargest(10, 'step_profit')
    for _, r in top.iterrows():
        print(f"  {r.get('step_in_episode',0):>5.0f} {r.get('episode',0):>4.0f} "
              f"{r['soc']:>6.3f} {r.get('net_mw',0):>8.1f} "
              f"{r['step_profit']:>10.1f} {r.get('timestamp','')}")

    print(f"\n--- Top-10 Worst Steps ---")
    bottom = df.nsmallest(10, 'step_profit')
    for _, r in bottom.iterrows():
        print(f"  {r.get('step_in_episode',0):>5.0f} {r.get('episode',0):>4.0f} "
              f"{r['soc']:>6.3f} {r.get('net_mw',0):>8.1f} "
              f"{r['step_profit']:>10.1f} {r.get('timestamp','')}")

    # 7. Hourly SoC profile
    if 'timestamp' in df.columns:
        try:
            df_ts = df.copy()
            df_ts['hour'] = pd.to_datetime(df_ts['timestamp']).dt.hour
            hourly_soc = df_ts.groupby('hour')['soc'].mean()
            hourly_profit = df_ts.groupby('hour')['step_profit'].mean()
            print(f"\n--- Hourly Profile ---")
            print(f"  {'Hour':>4} {'SoC':>6} {'Profit/step':>12}")
            for h in range(24):
                if h in hourly_soc.index:
                    print(f"  {h:>4} {hourly_soc[h]:>6.3f} {hourly_profit.get(h,0):>12.1f}")
        except Exception:
            pass

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Detailed SAC Evaluation')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model .zip')
    parser.add_argument('--data', type=str,
                        default='data/unified_multimarket_training_v5_raw_features.csv')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--output', type=str, default='evaluation_results')
    parser.add_argument('--time-step', type=float, default=0.25)
    parser.add_argument('--baseline', action='store_true',
                        help='Run IDLE baseline instead of model')
    args = parser.parse_args()

    # I resolve paths
    data_path = project_root / args.data
    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        return

    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # I load data (eval split = last 20%)
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    split_idx = int(len(df) * 0.80)
    df_eval = df.iloc[split_idx:].copy()
    print(f"Eval data: {len(df_eval)} rows ({df_eval.index[0]} to {df_eval.index[-1]})")

    # I load model if specified
    model = None
    vec_env = None
    use_vecnorm = False

    if args.model and not args.baseline:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model_path = Path(args.model)
        if not model_path.exists():
            # I try glob
            import glob
            candidates = glob.glob(str(model_path))
            if candidates:
                model_path = Path(candidates[0])
            else:
                print(f"ERROR: Model not found at {args.model}")
                return

        print(f"Loading model: {model_path}")
        model = SAC.load(model_path)

        # I check for VecNormalize stats
        norm_path = model_path.parent.parent / "vecnormalize.pkl"
        if not norm_path.exists():
            norm_path = model_path.parent.parent / "vec_normalize_sac.pkl"

        if norm_path.exists():
            print(f"Loading VecNormalize: {norm_path}")
            env = create_eval_env(df_eval, args.time_step)
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecNormalize.load(str(norm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            use_vecnorm = True
        else:
            print("WARNING: No VecNormalize stats found, running without normalization")

    # I run episodes
    all_infos = []
    for ep in range(args.episodes):
        print(f"\nRunning episode {ep+1}/{args.episodes}...")
        env = create_eval_env(df_eval, args.time_step)

        if use_vecnorm:
            infos = run_episode(env, model, use_vecnorm=True,
                                vec_env=vec_env, episode_idx=ep)
        else:
            infos = run_episode(env, model, episode_idx=ep)

        all_infos.extend(infos)
        ep_profit = sum(i.get('step_profit', 0) for i in infos)
        ep_reward = sum(i.get('reward', 0) for i in infos)
        print(f"  Steps: {len(infos)}, Profit: {ep_profit:.0f} EUR, "
              f"Reward: {ep_reward:.0f}")

    # I build DataFrame
    result_df = infos_to_dataframe(all_infos)

    # I save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "baseline" if args.baseline else Path(args.model).parent.parent.name
    csv_path = output_dir / f"eval_{model_name}_{timestamp}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\nSaved per-step CSV: {csv_path} ({len(result_df)} rows, "
          f"{len(result_df.columns)} columns)")

    # I print summary report
    print_summary_report(result_df, output_dir)

    # I also save summary to text file
    summary_path = output_dir / f"summary_{model_name}_{timestamp}.txt"
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        print_summary_report(result_df, output_dir)
    summary_path.write_text(f.getvalue(), encoding='utf-8')
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
