"""
Evaluation Visualizer — Matplotlib-based debugging charts.

I generate comprehensive charts showing agent behavior during evaluation:
- Battery SoC over time with charge/discharge annotations
- DAM prices with buy/sell markers
- Per-market revenue breakdown
- Hourly trading pattern heatmap
- Daily P&L waterfall

Usage:
    python tools/evaluation_visualizer.py --model models/unified_td3_v6/sac_*/final_model
    python tools/evaluation_visualizer.py --data evaluation_results/eval_data.csv

Output:
    evaluation_results/eval_charts.png (multi-panel figure)
    evaluation_results/eval_data.csv (raw step-by-step data)
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # I use non-interactive backend for server compatibility
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import json
import glob
import warnings
warnings.filterwarnings('ignore')


def run_evaluation(model_path: str, n_episodes: int = 1) -> pd.DataFrame:
    """I run evaluation and collect step-by-step data."""
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gym_envs.battery_env_unified import BatteryEnvUnified
    from gym_envs.battery_env_unified_sac import BatteryEnvUnifiedSAC

    # I find the model directory
    model_dir = Path(model_path).parent
    config_path = model_dir / 'training_config.json'

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        time_step = config.get('time_step_hours', 0.25)
        data_path = config.get('data_path', 'data/unified_multimarket_training_v2.csv')
        algorithm = config.get('algorithm', 'SAC')
    else:
        time_step = 0.25
        data_path = 'data/unified_multimarket_training_v2.csv'
        algorithm = 'TD3'

    # I load data (eval split = last 20%)
    df = pd.read_csv(data_path)
    n = len(df)
    eval_df = df.iloc[int(n * 0.8):]
    print(f"Eval data: {len(eval_df)} rows, time_step={time_step}h")

    # I determine episode length (7 days)
    steps_per_day = int(24 / time_step)
    episode_length = steps_per_day * 7

    # I create environment
    inner = BatteryEnvUnified(eval_df, time_step_hours=time_step, episode_length=episode_length)
    sac_env = BatteryEnvUnifiedSAC(inner_env=inner)
    vec_env = DummyVecEnv([lambda: sac_env])

    vec_norm_path = model_dir / 'vec_normalize_sac.pkl'
    if vec_norm_path.exists():
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # I load model
    ModelClass = SAC if algorithm == 'SAC' else TD3
    model = ModelClass.load(str(model_path))
    print(f"Model loaded: {algorithm}")

    all_steps = []

    for episode in range(n_episodes):
        obs = vec_env.reset()

        for step_i in range(episode_length):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            inf = info[0]
            d = inf.get('decoded_mw', {})

            # I get timestamp from data
            step_idx = inner.current_step - 1
            if step_idx >= 0 and step_idx < len(eval_df):
                row = eval_df.iloc[step_idx]
                ts_col = eval_df.columns[0]
                timestamp = row[ts_col] if 'Unnamed' not in ts_col else row.name
                dam_price = row.get('price', 0)
                xbid_bid = row.get('xbid_price_bid', dam_price - 1.5)
                xbid_ask = row.get('xbid_price_ask', dam_price + 1.5)
                mfrr_up = row.get('mfrr_price_up', 0)
                mfrr_down = row.get('mfrr_price_down', 0)
            else:
                timestamp = f"step_{step_i}"
                dam_price = xbid_bid = xbid_ask = mfrr_up = mfrr_down = 0

            all_steps.append({
                'episode': episode,
                'step': step_i,
                'timestamp': timestamp,
                'hour': (step_i * time_step) % 24,
                'day': int(step_i * time_step / 24),
                'soc': inf.get('soc', 0.5),
                'reward': reward[0],
                'dam_price': dam_price,
                'xbid_bid': xbid_bid,
                'xbid_ask': xbid_ask,
                'mfrr_price_up': mfrr_up,
                'mfrr_price_down': mfrr_down,
                'dam_mw': inf.get('dam_commitment_mw', 0),
                'xbid_mw': d.get('xbid', 0),
                'mfrr_mw': d.get('mfrr', 0),
                'afrr_mw': d.get('afrr', 0),
                'xbid_price_offset': d.get('xbid_price_offset', 0),
                'freebid_mw': d.get('freebid', 0),
                'dam_profit': inf.get('dam_profit', 0),
                'intraday_profit': inf.get('intraday_profit', 0),
                'afrr_cap_profit': inf.get('afrr_capacity_profit', 0),
                'afrr_energy_profit': inf.get('afrr_energy_profit', 0),
                'mfrr_profit': inf.get('mfrr_profit', 0),
                'total_profit': inf.get('total_profit', 0),
                'daily_cycles': inf.get('daily_cycles', 0),
                'total_cycles': inf.get('total_cycles', 0),
                'net_profit_step': inf.get('net_profit', 0),
            })

            if done[0]:
                break

    return pd.DataFrame(all_steps)


def create_debug_charts(df: pd.DataFrame, output_path: str = 'evaluation_results/eval_debug.png',
                        time_step: float = 0.25):
    """I create comprehensive debugging charts."""
    fig, axes = plt.subplots(5, 1, figsize=(20, 28), gridspec_kw={'height_ratios': [2, 2, 1.5, 1.5, 1.5]})
    fig.suptitle('EntsoDRL Agent Evaluation — Debug View', fontsize=16, fontweight='bold', y=0.98)

    hours = df['step'] * time_step
    n_days = int(hours.max() / 24) + 1

    # I add alternating day background shading
    for ax in axes:
        for d in range(n_days):
            if d % 2 == 0:
                ax.axvspan(d * 24, (d + 1) * 24, alpha=0.05, color='blue')

    # ─── Panel 1: SoC + Actions ─────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(hours, df['soc'] * 100, alpha=0.3, color='steelblue', label='SoC')
    ax1.plot(hours, df['soc'] * 100, color='steelblue', linewidth=1.5)
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Min SoC (5%)')
    ax1.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='Max SoC (95%)')
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.3)

    # I annotate charge/discharge actions on SoC chart
    sells = df[df['xbid_mw'] > 1.0]
    buys = df[df['xbid_mw'] < -1.0]
    if len(sells) > 0:
        ax1.scatter(sells['step'] * time_step, sells['soc'] * 100,
                   marker='v', color='red', s=sells['xbid_mw'].abs() * 3, alpha=0.5, label='XBID Sell')
    if len(buys) > 0:
        ax1.scatter(buys['step'] * time_step, buys['soc'] * 100,
                   marker='^', color='green', s=buys['xbid_mw'].abs() * 3, alpha=0.5, label='XBID Buy')

    ax1.set_ylabel('State of Charge (%)', fontsize=12)
    ax1.set_ylim(-2, 102)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('Battery SoC with Trading Actions', fontsize=13)

    # ─── Panel 2: Prices + Trading Decisions ─────────────────────
    ax2 = axes[1]
    ax2.plot(hours, df['dam_price'], color='navy', linewidth=1, alpha=0.8, label='DAM Price')
    ax2.fill_between(hours, df['xbid_bid'], df['xbid_ask'], alpha=0.15, color='orange', label='XBID Spread')

    # I mark buy/sell actions with price
    if len(sells) > 0:
        ax2.scatter(sells['step'] * time_step, sells['dam_price'],
                   marker='v', color='red', s=60, zorder=5, label=f'Sell ({len(sells)}x)')
    if len(buys) > 0:
        ax2.scatter(buys['step'] * time_step, buys['dam_price'],
                   marker='^', color='green', s=60, zorder=5, label=f'Buy ({len(buys)}x)')

    ax2.set_ylabel('Price (EUR/MWh)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_title('Market Prices with Agent Decisions', fontsize=13)

    # ─── Panel 3: MW Dispatch per Market ─────────────────────────
    ax3 = axes[2]
    ax3.bar(hours, df['dam_mw'], width=time_step * 0.8, alpha=0.6, color='navy', label='DAM')
    ax3.bar(hours, df['xbid_mw'], width=time_step * 0.6, alpha=0.6, color='orange', label='XBID')
    ax3.bar(hours, df['mfrr_mw'], width=time_step * 0.4, alpha=0.6, color='purple', label='mFRR')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Power (MW)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_title('Dispatch per Market (positive=sell, negative=buy)', fontsize=13)

    # ─── Panel 4: Cumulative Revenue per Market ──────────────────
    ax4 = axes[3]
    # I compute per-step revenue (diff of cumulative)
    for col, label, color in [
        ('dam_profit', 'DAM', 'navy'),
        ('intraday_profit', 'IntraDay', 'orange'),
        ('afrr_cap_profit', 'aFRR Cap', 'teal'),
        ('afrr_energy_profit', 'aFRR Energy', 'cyan'),
        ('mfrr_profit', 'mFRR', 'purple'),
    ]:
        if col in df.columns:
            ax4.plot(hours, df[col], linewidth=1.5, label=label, color=color)

    ax4.plot(hours, df['total_profit'], linewidth=2.5, label='TOTAL', color='black', linestyle='--')
    ax4.axhline(y=0, color='red', linewidth=0.5)
    ax4.set_ylabel('Cumulative EUR', fontsize=12)
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_title('Cumulative Revenue per Market', fontsize=13)

    # ─── Panel 5: Cycles + Reward per Step ───────────────────────
    ax5 = axes[4]
    ax5_twin = ax5.twinx()
    ax5.bar(hours, df['reward'], width=time_step * 0.8, alpha=0.4,
            color=np.where(df['reward'] > 0, 'green', 'red'), label='Reward')
    ax5_twin.plot(hours, df['total_cycles'], color='black', linewidth=1.5, label='Total Cycles')
    ax5.set_ylabel('Reward (scaled)', fontsize=12, color='gray')
    ax5_twin.set_ylabel('Total Cycles', fontsize=12)
    ax5.set_xlabel('Hours', fontsize=12)
    ax5.set_title('Per-Step Reward + Cycle Count', fontsize=13)

    # I add legends
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # I add summary text box
    total_profit = df['total_profit'].iloc[-1]
    total_days = len(df) * time_step / 24
    total_cycles = df['total_cycles'].iloc[-1]
    mean_soc = df['soc'].mean()

    summary = (
        f"Days: {total_days:.1f} | "
        f"Profit: {total_profit:,.0f} EUR ({total_profit/total_days:,.0f}/day) | "
        f"Cycles: {total_cycles:.1f} ({total_cycles/total_days:.1f}/day) | "
        f"Mean SoC: {mean_soc:.0%}"
    )
    fig.text(0.5, 0.005, summary, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Debug charts saved: {output_path}")


def create_investor_charts(df: pd.DataFrame, output_path: str = 'evaluation_results/eval_investor.png',
                           time_step: float = 0.25):
    """I create clean, professional charts for investor presentations."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('EntsoDRL Battery Trading System — Performance Report',
                fontsize=16, fontweight='bold', y=0.98)

    total_days = len(df) * time_step / 24
    total_profit = df['total_profit'].iloc[-1]
    daily_profit = total_profit / total_days

    # ─── Panel 1: Daily P&L ──────────────────────────────────────
    ax1 = axes[0, 0]
    steps_per_day = int(24 / time_step)

    daily_pnl = []
    for d in range(int(total_days)):
        day_slice = df.iloc[d * steps_per_day:(d + 1) * steps_per_day]
        if len(day_slice) > 0:
            day_profit = day_slice['net_profit_step'].sum()
            daily_pnl.append(day_profit)

    if daily_pnl:
        colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in daily_pnl]
        ax1.bar(range(len(daily_pnl)), daily_pnl, color=colors, alpha=0.8)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.axhline(y=np.mean(daily_pnl), color='navy', linewidth=1.5, linestyle='--',
                   label=f'Average: {np.mean(daily_pnl):,.0f} EUR/day')
        win_days = sum(1 for p in daily_pnl if p > 0)
        ax1.set_title(f'Daily P&L — Win Rate: {win_days}/{len(daily_pnl)} ({win_days/len(daily_pnl)*100:.0f}%)',
                     fontsize=13)
        ax1.set_xlabel('Day')
        ax1.set_ylabel('EUR')
        ax1.legend(fontsize=10)

    # ─── Panel 2: Revenue by Market (pie) ────────────────────────
    ax2 = axes[0, 1]
    market_totals = {
        'DAM': df['dam_profit'].iloc[-1],
        'IntraDay': df['intraday_profit'].iloc[-1],
        'aFRR Cap': df['afrr_cap_profit'].iloc[-1],
        'aFRR Energy': df['afrr_energy_profit'].iloc[-1],
        'mFRR': df['mfrr_profit'].iloc[-1],
    }

    # I separate positive and negative
    positive = {k: v for k, v in market_totals.items() if v > 0}
    negative = {k: v for k, v in market_totals.items() if v < 0}

    if positive:
        labels = [f"{k}\n{v:,.0f} EUR" for k, v in positive.items()]
        sizes = list(positive.values())
        colors_pie = ['#3498db', '#e67e22', '#1abc9c', '#2ecc71', '#9b59b6']
        ax2.pie(sizes, labels=labels, colors=colors_pie[:len(sizes)],
               autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9})

    neg_text = '\n'.join([f"{k}: {v:,.0f} EUR" for k, v in negative.items()])
    if neg_text:
        ax2.text(0.95, 0.05, f"Costs:\n{neg_text}", transform=ax2.transAxes,
                fontsize=9, va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax2.set_title(f'Revenue by Market (Total: {total_profit:,.0f} EUR)', fontsize=13)

    # ─── Panel 3: SoC Management ─────────────────────────────────
    ax3 = axes[1, 0]
    hours = df['step'] * time_step
    ax3.fill_between(hours, df['soc'] * 100, alpha=0.3, color='steelblue')
    ax3.plot(hours, df['soc'] * 100, color='steelblue', linewidth=1)
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(y=95, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylim(-2, 102)
    ax3.set_xlabel('Hours')
    ax3.set_ylabel('SoC (%)')
    ax3.set_title(f'Battery Management (Mean SoC: {df["soc"].mean():.0%})', fontsize=13)

    # ─── Panel 4: Key Metrics Summary ────────────────────────────
    ax4 = axes[1, 1]
    ax4.axis('off')

    total_cycles = df['total_cycles'].iloc[-1]
    daily_cycles = total_cycles / total_days

    metrics = [
        ('Total Profit', f'{total_profit:,.0f} EUR'),
        ('Daily Average', f'{daily_profit:,.0f} EUR/day'),
        ('Annualized', f'{daily_profit * 365:,.0f} EUR/year'),
        ('', ''),
        ('Total Cycles', f'{total_cycles:.1f}'),
        ('Cycles/Day', f'{daily_cycles:.2f}'),
        ('Mean SoC', f'{df["soc"].mean():.0%}'),
        ('SoC Range', f'{df["soc"].min():.0%} - {df["soc"].max():.0%}'),
        ('', ''),
        ('Evaluation Period', f'{total_days:.1f} days'),
        ('Win Rate', f'{sum(1 for p in daily_pnl if p > 0)}/{len(daily_pnl)} days' if daily_pnl else 'N/A'),
    ]

    if daily_pnl and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365)
        metrics.append(('Sharpe Ratio', f'{sharpe:.2f}'))

    y_pos = 0.95
    for label, value in metrics:
        if label == '':
            y_pos -= 0.03
            continue
        ax4.text(0.1, y_pos, label, fontsize=12, fontweight='bold',
                transform=ax4.transAxes, va='top')
        ax4.text(0.7, y_pos, value, fontsize=12,
                transform=ax4.transAxes, va='top', ha='right')
        y_pos -= 0.08

    ax4.set_title('Key Performance Metrics', fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Investor charts saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluation Visualizer')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--data', type=str, help='Path to pre-computed eval CSV')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    parser.add_argument('--output-dir', type=str, default='evaluation_results')
    parser.add_argument('--time-step', type=float, default=0.25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # I load or generate evaluation data
    if args.data:
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} steps from {args.data}")
    elif args.model:
        # I find the model (support glob patterns)
        model_paths = glob.glob(args.model)
        if not model_paths:
            print(f"No model found at {args.model}")
            return
        model_path = sorted(model_paths)[-1]
        print(f"Using model: {model_path}")
        df = run_evaluation(model_path, n_episodes=args.episodes)
        # I save raw data
        csv_path = output_dir / 'eval_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"Raw data saved: {csv_path}")
    else:
        print("Provide --model or --data")
        return

    # I generate both chart types
    create_debug_charts(df, str(output_dir / 'eval_debug.png'), args.time_step)
    create_investor_charts(df, str(output_dir / 'eval_investor.png'), args.time_step)

    print(f"\nAll charts saved to {output_dir}/")


if __name__ == '__main__':
    main()
