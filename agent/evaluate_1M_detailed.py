"""
Detailed Evaluation of 1M Checkpoint
Comprehensive analysis to identify improvement opportunities.
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
    """Traverse wrappers to get action_masks from base env."""
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
    """Get the base BatteryEnvMasked from wrapper stack."""
    env = vec_env.envs[0]
    while hasattr(env, 'env'):
        if isinstance(env, BatteryEnvMasked):
            return env
        env = env.env
    return env


def evaluate_detailed():
    # === PATHS ===
    MODEL_PATH = "models/checkpoints/ppo_12h_v9/ppo_masked_imbal_1800000_steps.zip"
    VECNORM_PATH = "models/checkpoints/ppo_12h_v9/ppo_masked_imbal_vecnormalize_1800000_steps.pkl"
    DATA_PATH = "data/feasible_data_with_balancing.csv"

    print("=" * 70)
    print("DETAILED EVALUATION - 1M CHECKPOINT")
    print("=" * 70)

    # === LOAD DATA ===
    battery_params = {
        'capacity_mwh': 146.0,
        'max_discharge_mw': 30.0,
        'efficiency': 0.94,
    }

    df = load_historical_data(DATA_PATH)
    df = regenerate_dam_commitments(df, battery_params=battery_params)

    # I split: 80% train, 20% test (unseen data)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    print(f"Test period: {len(test_df):,} hours")
    print(f"Model: {MODEL_PATH}")
    print(f"VecNorm: {VECNORM_PATH}")

    # === CREATE ENV ===
    def make_env():
        env = BatteryEnvMasked(test_df, battery_params, n_actions=21)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])

    # I load VecNormalize with training=False for evaluation
    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False
    env.norm_reward = False

    # === LOAD MODEL ===
    model = MaskablePPO.load(MODEL_PATH, env=env)

    # === TRACKING VARIABLES ===
    episode_rewards = []
    episode_profits = []
    episode_violations = []
    episode_degradation = []

    hourly_actions = defaultdict(lambda: defaultdict(int))  # hour -> action -> count
    hourly_rewards = defaultdict(list)
    price_action_corr = []  # (price, action_power)
    soc_trajectory = []

    dam_fulfillment = {'met': 0, 'missed_buy': 0, 'missed_sell': 0}

    action_counts = defaultdict(int)
    total_steps = 0
    total_reward = 0
    total_profit = 0
    total_degradation = 0
    total_violations = 0

    # === RUN EVALUATION ===
    print("\nRunning evaluation...")
    obs = env.reset()
    done = False

    step_data = []

    while not done:
        # I get action mask and predict
        mask = get_action_mask(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=np.array([mask]))
        action_int = int(action[0])

        # I get base env state before step
        base_env = get_base_env(env)
        current_soc = base_env.current_soc
        current_hour = base_env.current_step % 24
        current_price = base_env.df.iloc[base_env.current_step]['price'] if base_env.current_step < len(base_env.df) else 0
        dam_commitment = base_env.df.iloc[base_env.current_step].get('dam_commitment', 0) if base_env.current_step < len(base_env.df) else 0

        # Step
        obs, reward, done, info = env.step(action)
        reward_val = reward[0]
        info_dict = info[0]

        # I extract detailed metrics from info (correct keys from reward_calculator)
        profit = info_dict.get('net_profit', 0)
        degradation = info_dict.get('degradation', 0)
        violation = info_dict.get('violation', 0)
        power = info_dict.get('actual_mw', 0)  # actual physical power in MW

        # Track everything
        total_reward += reward_val
        total_profit += profit
        total_degradation += degradation
        total_violations += 1 if violation > 0.5 else 0
        total_steps += 1

        action_counts[action_int] += 1
        hourly_actions[current_hour][action_int] += 1
        hourly_rewards[current_hour].append(reward_val)
        price_action_corr.append((current_price, power))
        soc_trajectory.append(current_soc)

        # DAM fulfillment tracking
        if dam_commitment > 0.1:  # SELL commitment
            if power > 0:  # Discharging = selling
                dam_fulfillment['met'] += 1
            else:
                dam_fulfillment['missed_sell'] += 1
        elif dam_commitment < -0.1:  # BUY commitment
            if power < 0:  # Charging = buying
                dam_fulfillment['met'] += 1
            else:
                dam_fulfillment['missed_buy'] += 1

        step_data.append({
            'step': total_steps,
            'hour': current_hour,
            'price': current_price,
            'action': action_int,
            'power': power,
            'soc': current_soc,
            'reward': reward_val,
            'profit': profit,
            'degradation': degradation,
            'violation': violation,
            'dam_commitment': dam_commitment
        })

    # === ANALYSIS ===
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n[OVERALL METRICS]")
    print(f"  Total Steps:        {total_steps:>12,}")
    print(f"  Total Reward:       {total_reward:>12,.2f}")
    print(f"  Total Profit:       {total_profit:>12,.2f} EUR")
    print(f"  Total Degradation:  {total_degradation:>12,.2f} EUR")
    print(f"  Net Profit:         {total_profit - total_degradation:>12,.2f} EUR")
    print(f"  Violations:         {total_violations:>12,} ({total_violations/total_steps*100:.2f}%)")

    # Action distribution
    print(f"\n[ACTION DISTRIBUTION]")
    discharge = sum(action_counts[i] for i in range(11, 21))
    idle = action_counts[10]
    charge = sum(action_counts[i] for i in range(0, 10))
    print(f"  Charge (0-9):       {charge:>8,} ({charge/total_steps*100:.1f}%)")
    print(f"  Idle (10):          {idle:>8,} ({idle/total_steps*100:.1f}%)")
    print(f"  Discharge (11-20):  {discharge:>8,} ({discharge/total_steps*100:.1f}%)")

    # Top 5 actions
    print(f"\n  Top 5 Actions:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for action, count in sorted_actions:
        power_mw = (action - 10) * (battery_params['max_discharge_mw'] / 10)
        print(f"    Action {action:2d} ({power_mw:+6.1f} MW): {count:>6,} ({count/total_steps*100:.1f}%)")

    # DAM commitment fulfillment
    print(f"\n[TARGET] DAM COMMITMENT FULFILLMENT:")
    total_commitments = dam_fulfillment['met'] + dam_fulfillment['missed_buy'] + dam_fulfillment['missed_sell']
    if total_commitments > 0:
        print(f"  Met:          {dam_fulfillment['met']:>8,} ({dam_fulfillment['met']/total_commitments*100:.1f}%)")
        print(f"  Missed SELL:  {dam_fulfillment['missed_sell']:>8,} ({dam_fulfillment['missed_sell']/total_commitments*100:.1f}%)")
        print(f"  Missed BUY:   {dam_fulfillment['missed_buy']:>8,} ({dam_fulfillment['missed_buy']/total_commitments*100:.1f}%)")

    # Hourly analysis
    print(f"\n[HOURLY] HOURLY BEHAVIOR:")
    print(f"  {'Hour':<6} {'Avg Reward':>12} {'Main Action':>12} {'Discharge%':>12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")
    for hour in range(24):
        if hourly_rewards[hour]:
            avg_rew = np.mean(hourly_rewards[hour])
            total_hour = sum(hourly_actions[hour].values())
            discharge_pct = sum(hourly_actions[hour][i] for i in range(11, 21)) / total_hour * 100 if total_hour > 0 else 0
            main_action = max(hourly_actions[hour], key=hourly_actions[hour].get) if hourly_actions[hour] else 10
            print(f"  {hour:02d}:00  {avg_rew:>12.2f} {main_action:>12d} {discharge_pct:>11.1f}%")

    # Price correlation
    print(f"\n[PRICE] PRICE-ACTION CORRELATION:")
    prices, powers = zip(*price_action_corr)
    corr = np.corrcoef(prices, powers)[0, 1]
    print(f"  Correlation: {corr:.4f}")
    print(f"  (Ideal: >0.5 = sell when price high, buy when low)")

    # I calculate price quintiles behavior
    price_arr = np.array(prices)
    power_arr = np.array(powers)
    quintiles = np.percentile(price_arr, [20, 40, 60, 80])
    print(f"\n  Price Quintile Behavior:")
    print(f"    Very Low  (< {quintiles[0]:.1f} EUR):  Avg Power = {power_arr[price_arr < quintiles[0]].mean():+.2f} MW")
    print(f"    Low       (< {quintiles[1]:.1f} EUR):  Avg Power = {power_arr[(price_arr >= quintiles[0]) & (price_arr < quintiles[1])].mean():+.2f} MW")
    print(f"    Medium    (< {quintiles[2]:.1f} EUR):  Avg Power = {power_arr[(price_arr >= quintiles[1]) & (price_arr < quintiles[2])].mean():+.2f} MW")
    print(f"    High      (< {quintiles[3]:.1f} EUR):  Avg Power = {power_arr[(price_arr >= quintiles[2]) & (price_arr < quintiles[3])].mean():+.2f} MW")
    print(f"    Very High (>={quintiles[3]:.1f} EUR):  Avg Power = {power_arr[price_arr >= quintiles[3]].mean():+.2f} MW")

    # SoC analysis
    print(f"\n[SOC] SOC TRAJECTORY:")
    soc_arr = np.array(soc_trajectory)
    print(f"  Min SoC:  {soc_arr.min()*100:.1f}%")
    print(f"  Max SoC:  {soc_arr.max()*100:.1f}%")
    print(f"  Mean SoC: {soc_arr.mean()*100:.1f}%")
    print(f"  Std SoC:  {soc_arr.std()*100:.1f}%")

    # I identify problematic patterns
    print("\n" + "=" * 70)
    print("[ANALYSIS] IDENTIFIED ISSUES & IMPROVEMENT OPPORTUNITIES")
    print("=" * 70)

    issues = []

    # Issue 1: Negative profit
    if total_profit < 0:
        issues.append(f"[X] CRITICAL: Negative profit ({total_profit:,.0f} EUR). Agent is losing money!")

    # Issue 2: Poor price correlation
    if corr < 0.3:
        issues.append(f"[!] LOW PRICE CORRELATION ({corr:.3f}). Agent doesn't follow buy-low-sell-high strategy.")

    # Issue 3: Too much idle
    if idle / total_steps > 0.4:
        issues.append(f"[!] EXCESSIVE IDLE ({idle/total_steps*100:.1f}%). Agent is too conservative.")

    # Issue 4: DAM violations
    if total_commitments > 0:
        miss_rate = (dam_fulfillment['missed_buy'] + dam_fulfillment['missed_sell']) / total_commitments
        if miss_rate > 0.1:
            issues.append(f"[!] HIGH DAM MISS RATE ({miss_rate*100:.1f}%). Penalties hurt profit.")

    # Issue 5: SoC not cycling
    if soc_arr.std() < 0.1:
        issues.append(f"[!] LOW SOC VARIANCE ({soc_arr.std()*100:.1f}%). Battery not being utilized.")

    # Issue 6: Wrong direction in extreme prices
    low_price_power = power_arr[price_arr < quintiles[0]].mean()
    high_price_power = power_arr[price_arr >= quintiles[3]].mean()
    if low_price_power > 0:
        issues.append(f"[X] SELLING at LOW prices (avg {low_price_power:+.2f} MW)! Should be CHARGING.")
    if high_price_power < 0:
        issues.append(f"[X] BUYING at HIGH prices (avg {high_price_power:+.2f} MW)! Should be DISCHARGING.")

    for issue in issues:
        print(f"  {issue}")

    if not issues:
        print("  [OK] No major issues detected!")

    # Recommendations
    print("\n" + "=" * 70)
    print("[IDEAS] RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    if corr < 0.3:
        recommendations.append("1. INCREASE PRICE SIGNAL: Add price to reward shaping or increase its weight.")

    if idle / total_steps > 0.3:
        recommendations.append("2. REDUCE IDLE PENALTY: Lower entropy coefficient or add trading incentive.")

    if total_degradation > abs(total_profit) * 0.5:
        recommendations.append("3. REDUCE DEGRADATION COST: Current setting may be too aggressive.")

    if miss_rate > 0.1:
        recommendations.append("4. INCREASE DAM PENALTY: Agent not respecting commitments enough.")

    if soc_arr.std() < 0.15:
        recommendations.append("5. ENCOURAGE CYCLING: Add reward bonus for full charge/discharge cycles.")

    # Always recommend
    recommendations.append("6. CONTINUE TRAINING: 1M steps is early. Metrics should improve by 3-5M.")

    for rec in recommendations:
        print(f"  {rec}")

    print("\n" + "=" * 70)

    # Save detailed data
    results_df = pd.DataFrame(step_data)
    results_df.to_csv("models/checkpoints/ppo_12h_v9/eval_1M_detailed.csv", index=False)
    print(f"Detailed results saved to: models/checkpoints/ppo_12h_v9/eval_1M_detailed.csv")

    return {
        'total_reward': total_reward,
        'total_profit': total_profit,
        'total_degradation': total_degradation,
        'violations': total_violations,
        'price_correlation': corr,
        'issues': issues
    }


if __name__ == "__main__":
    evaluate_detailed()
