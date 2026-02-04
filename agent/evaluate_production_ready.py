"""
Production-Ready Evaluation
Metrics that matter for real-world deployment, not just training performance.
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data
from data.dam_commitment_generator import regenerate_dam_commitments


def get_action_mask(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        if hasattr(env, 'action_masks'):
            return env.action_masks()
        env = env.env
    if hasattr(env, 'action_masks'):
        return env.action_masks()
    raise RuntimeError("Could not find action_masks!")


def mask_fn(env):
    return env.action_masks()


def get_base_env(vec_env):
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        if isinstance(env, BatteryEnvMasked):
            return env
        env = env.env
    return env


def evaluate_production_ready(model_path: str, vecnorm_path: str, version: str = ""):
    """
    I evaluate the model with production-relevant metrics.
    These metrics indicate whether the agent will perform well in live trading.
    """
    DATA_PATH = "data/feasible_data_with_balancing.csv"

    print("=" * 70)
    print(f"PRODUCTION-READY EVALUATION {version}")
    print("=" * 70)

    # === LOAD DATA ===
    battery_params = {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }

    df = load_historical_data(DATA_PATH)
    df = regenerate_dam_commitments(df, battery_params=battery_params)

    # I use last 20% as test (unseen data)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"Test period: {len(test_df):,} hours ({len(test_df)/24:.0f} days)")
    print(f"Model: {model_path}")

    # === CREATE ENV ===
    def make_env():
        env = BatteryEnvMasked(test_df, battery_params, n_actions=21)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    model = MaskablePPO.load(model_path, env=env)

    # === TRACKING ===
    daily_profits = []
    daily_revenues = []
    hourly_profits = defaultdict(list)
    trade_profits = []  # Individual trade P&L

    peak_hours = set(range(17, 22))  # 17:00-21:00
    peak_discharge_count = 0
    peak_total_count = 0

    dam_met = 0
    dam_total = 0

    total_energy_traded = 0
    cycle_count = 0
    last_soc = 0.5

    current_day_profit = 0
    current_day_revenue = 0
    current_hour = 0
    steps_in_day = 0

    # === RUN EVALUATION ===
    print("\nRunning evaluation...")
    obs = env.reset()
    done = False
    total_steps = 0

    while not done:
        mask = get_action_mask(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=np.array([mask]))
        action_int = int(action[0])

        base_env = get_base_env(env)
        current_soc = base_env.current_soc
        hour = base_env.current_step % 24
        price = base_env.df.iloc[base_env.current_step]['price'] if base_env.current_step < len(base_env.df) else 100

        obs, reward, done, info = env.step(action)
        info_dict = info[0]

        # I extract metrics
        power = info_dict.get('actual_mw', 0)
        dam_rev = info_dict.get('dam_rev', 0)
        balancing_rev = info_dict.get('balancing_rev', 0)
        degradation = info_dict.get('degradation', 0)
        shortfall_penalty = info_dict.get('shortfall_penalty', 0)
        dam_commitment = base_env.df.iloc[base_env.current_step - 1].get('dam_commitment', 0) if base_env.current_step > 0 else 0

        # I calculate trade profit (excluding penalties for clean comparison)
        trade_profit = dam_rev + balancing_rev - degradation
        net_profit = trade_profit - shortfall_penalty

        # Track individual trades
        if abs(power) > 0.1:
            trade_profits.append(trade_profit)
            total_energy_traded += abs(power)

        # Daily tracking
        current_day_profit += net_profit
        current_day_revenue += dam_rev + balancing_rev
        steps_in_day += 1

        if hour == 23 and steps_in_day >= 20:  # End of day
            daily_profits.append(current_day_profit)
            daily_revenues.append(current_day_revenue)
            current_day_profit = 0
            current_day_revenue = 0
            steps_in_day = 0

        # Hourly profit tracking
        hourly_profits[hour].append(net_profit)

        # Peak hour analysis
        if hour in peak_hours:
            peak_total_count += 1
            if power > 0.1:  # Discharging
                peak_discharge_count += 1

        # DAM compliance
        if abs(dam_commitment) > 0.1:
            dam_total += 1
            if dam_commitment > 0 and power > 0:  # SELL met
                dam_met += 1
            elif dam_commitment < 0 and power < 0:  # BUY met
                dam_met += 1

        # Cycle counting (full cycle = SoC goes from low to high or vice versa)
        soc_change = abs(current_soc - last_soc)
        if soc_change > 0.5:  # Significant SoC swing
            cycle_count += 0.5
        last_soc = current_soc

        total_steps += 1

    # === CALCULATE PRODUCTION METRICS ===
    print("\n" + "=" * 70)
    print("PRODUCTION READINESS METRICS")
    print("=" * 70)

    # 1. Profitability Metrics
    daily_profits = np.array(daily_profits)
    total_profit = daily_profits.sum()
    avg_daily_profit = daily_profits.mean() if len(daily_profits) > 0 else 0

    print(f"\n[1] PROFITABILITY")
    print(f"    Total P&L:           {total_profit:>12,.0f} EUR")
    print(f"    Avg Daily P&L:       {avg_daily_profit:>12,.0f} EUR/day")
    print(f"    Days Evaluated:      {len(daily_profits):>12}")

    # 2. Risk Metrics
    if len(daily_profits) > 1:
        returns = daily_profits / (battery_params['capacity_mwh'] * 100)  # Normalize
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max Drawdown
        cumulative = np.cumsum(daily_profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max()

        # Win Rate
        winning_days = (daily_profits > 0).sum()
        win_rate = winning_days / len(daily_profits) * 100

        # Profit Factor
        gross_profit = daily_profits[daily_profits > 0].sum()
        gross_loss = abs(daily_profits[daily_profits < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        sharpe = 0
        max_drawdown = 0
        win_rate = 0
        profit_factor = 0

    print(f"\n[2] RISK METRICS")
    print(f"    Sharpe Ratio:        {sharpe:>12.2f}")
    print(f"    Max Drawdown:        {max_drawdown:>12,.0f} EUR")
    print(f"    Win Rate:            {win_rate:>12.1f}%")
    print(f"    Profit Factor:       {profit_factor:>12.2f}")

    # 3. Operational Metrics
    dam_compliance = dam_met / dam_total * 100 if dam_total > 0 else 0
    peak_discharge_rate = peak_discharge_count / peak_total_count * 100 if peak_total_count > 0 else 0
    cycles_per_day = cycle_count / (len(daily_profits) if len(daily_profits) > 0 else 1)
    revenue_per_mwh = sum(daily_revenues) / total_energy_traded if total_energy_traded > 0 else 0

    print(f"\n[3] OPERATIONAL METRICS")
    print(f"    DAM Compliance:      {dam_compliance:>12.1f}%")
    print(f"    Peak Discharge Rate: {peak_discharge_rate:>12.1f}%  (target: >50%)")
    print(f"    Cycles/Day:          {cycles_per_day:>12.2f}")
    print(f"    Revenue/MWh:         {revenue_per_mwh:>12.2f} EUR")

    # 4. Timing Analysis
    print(f"\n[4] HOURLY PROFIT PATTERN (EUR/hour avg)")
    best_hours = []
    worst_hours = []
    for hour in range(24):
        if hourly_profits[hour]:
            avg = np.mean(hourly_profits[hour])
            best_hours.append((hour, avg))

    best_hours.sort(key=lambda x: x[1], reverse=True)

    print(f"    Best Hours:  ", end="")
    for h, p in best_hours[:3]:
        print(f"{h:02d}:00 ({p:+.0f})", end="  ")
    print()

    print(f"    Worst Hours: ", end="")
    for h, p in best_hours[-3:]:
        print(f"{h:02d}:00 ({p:+.0f})", end="  ")
    print()

    # === PRODUCTION READINESS SCORE ===
    print("\n" + "=" * 70)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 70)

    score = 0
    max_score = 100
    issues = []

    # DAM Compliance (25 points)
    if dam_compliance >= 95:
        score += 25
        print(f"    [OK] DAM Compliance >= 95%: +25 pts")
    elif dam_compliance >= 85:
        score += 15
        print(f"    [PARTIAL] DAM Compliance 85-95%: +15 pts")
        issues.append("Improve DAM compliance to >95%")
    else:
        score += 5
        print(f"    [FAIL] DAM Compliance < 85%: +5 pts")
        issues.append("CRITICAL: DAM compliance too low")

    # Peak Hour Discharge (25 points)
    if peak_discharge_rate >= 50:
        score += 25
        print(f"    [OK] Peak discharge >= 50%: +25 pts")
    elif peak_discharge_rate >= 35:
        score += 15
        print(f"    [PARTIAL] Peak discharge 35-50%: +15 pts")
        issues.append("Increase discharge during peak hours (17-21)")
    else:
        score += 5
        print(f"    [FAIL] Peak discharge < 35%: +5 pts")
        issues.append("CRITICAL: Not discharging during peak hours")

    # Sharpe Ratio (25 points)
    if sharpe >= 1.0:
        score += 25
        print(f"    [OK] Sharpe >= 1.0: +25 pts")
    elif sharpe >= 0.5:
        score += 15
        print(f"    [PARTIAL] Sharpe 0.5-1.0: +15 pts")
        issues.append("Risk-adjusted returns could improve")
    elif sharpe >= 0:
        score += 10
        print(f"    [WEAK] Sharpe 0-0.5: +10 pts")
        issues.append("Low risk-adjusted returns")
    else:
        score += 0
        print(f"    [FAIL] Sharpe < 0: +0 pts")
        issues.append("CRITICAL: Negative risk-adjusted returns")

    # Win Rate (25 points)
    if win_rate >= 55:
        score += 25
        print(f"    [OK] Win rate >= 55%: +25 pts")
    elif win_rate >= 45:
        score += 15
        print(f"    [PARTIAL] Win rate 45-55%: +15 pts")
        issues.append("Win rate is marginal")
    else:
        score += 5
        print(f"    [FAIL] Win rate < 45%: +5 pts")
        issues.append("CRITICAL: Losing more days than winning")

    print(f"\n    TOTAL SCORE: {score}/{max_score}")

    if score >= 80:
        print(f"\n    STATUS: [READY] Safe to deploy to paper trading")
    elif score >= 60:
        print(f"\n    STATUS: [CAUTION] Consider more training before deployment")
    else:
        print(f"\n    STATUS: [NOT READY] Needs significant improvement")

    if issues:
        print(f"\n    Issues to address:")
        for issue in issues:
            print(f"      - {issue}")

    print("\n" + "=" * 70)

    return {
        'score': score,
        'dam_compliance': dam_compliance,
        'peak_discharge_rate': peak_discharge_rate,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'issues': issues
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--vecnorm', type=str, required=True)
    parser.add_argument('--version', type=str, default="")
    args = parser.parse_args()

    evaluate_production_ready(args.model, args.vecnorm, args.version)
