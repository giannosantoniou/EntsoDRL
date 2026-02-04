"""
Comprehensive v12 Analysis
"""
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append('.')
from gym_envs.battery_env_masked import BatteryEnvMasked
from data.data_loader import load_historical_data

print('='*70)
print('COMPREHENSIVE v12 ANALYSIS')
print('='*70)

# Load model and data
model = MaskablePPO.load('models/ppo_v12_best.zip')
df = load_historical_data('data/feasible_data_with_balancing.csv')
battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}

# Test set
split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

# Create env
def make_env():
    env = BatteryEnvMasked(test_df, battery_params, n_actions=21)
    env = ActionMasker(env, lambda e: e.action_masks())
    return env

env = DummyVecEnv([make_env])
env = VecNormalize.load('models/vec_normalize_v12.pkl', env)
env.training = False
env.norm_reward = False

# Collect detailed data
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
    data.append({
        'step': step,
        'hour': step % 24,
        'day': step // 24,
        'price': info_dict.get('dam_price', 0) or info_dict.get('best_bid', 0),
        'power': info_dict.get('actual_mw', 0),
        'soc': info_dict.get('soc', 0.5),
        'dam_commitment': info_dict.get('dam_commitment', 0),
        'profit': info_dict.get('net_profit', 0),
        'action': action[0]
    })

    if done:
        obs = env.reset()

df_data = pd.DataFrame(data)
df_data['is_discharge'] = df_data['power'] > 0.1
df_data['is_charge'] = df_data['power'] < -0.1
df_data['is_idle'] = ~df_data['is_discharge'] & ~df_data['is_charge']

# ========================================
# 1. PRICE TIMING ANALYSIS
# ========================================
print('\n' + '='*70)
print('1. PRICE TIMING ANALYSIS')
print('='*70)

df_valid = df_data[df_data['price'] > 1].copy()
q1, q2, q3 = df_valid['price'].quantile([0.25, 0.5, 0.75])

print(f'Price quartiles: Q1<{q1:.0f}, Q2<{q2:.0f}, Q3<{q3:.0f}')

def classify(p):
    if p <= q1: return 'Q1 (Cheap)'
    elif p <= q2: return 'Q2'
    elif p <= q3: return 'Q3'
    else: return 'Q4 (Expensive)'

df_valid['quartile'] = df_valid['price'].apply(classify)

print('\nBehavior by Price Quartile:')
print('-'*60)
for q in ['Q1 (Cheap)', 'Q2', 'Q3', 'Q4 (Expensive)']:
    subset = df_valid[df_valid['quartile'] == q]
    if len(subset) > 0:
        discharge = subset['is_discharge'].mean() * 100
        charge = subset['is_charge'].mean() * 100
        idle = subset['is_idle'].mean() * 100
        avg_price = subset['price'].mean()
        print(f'{q:15s}: D:{discharge:5.1f}% | C:{charge:5.1f}% | I:{idle:5.1f}% | Price:{avg_price:6.0f}')

# Ideal behavior check
q4_discharge = df_valid[df_valid['quartile'] == 'Q4 (Expensive)']['is_discharge'].mean() * 100
q1_charge = df_valid[df_valid['quartile'] == 'Q1 (Cheap)']['is_charge'].mean() * 100
q4_charge = df_valid[df_valid['quartile'] == 'Q4 (Expensive)']['is_charge'].mean() * 100
q1_discharge = df_valid[df_valid['quartile'] == 'Q1 (Cheap)']['is_discharge'].mean() * 100

print(f'\n[CHECK] Q4 Discharge (want HIGH): {q4_discharge:.1f}%', '[OK]' if q4_discharge > 50 else '[NEEDS WORK]')
print(f'[CHECK] Q1 Charge (want HIGH): {q1_charge:.1f}%', '[OK]' if q1_charge > 40 else '[NEEDS WORK]')
print(f'[CHECK] Q4 Charge (want LOW): {q4_charge:.1f}%', '[OK]' if q4_charge < 20 else '[PROBLEM!]')
print(f'[CHECK] Q1 Discharge (want LOW): {q1_discharge:.1f}%', '[OK]' if q1_discharge < 30 else '[PROBLEM!]')

# ========================================
# 2. HOURLY PATTERN ANALYSIS
# ========================================
print('\n' + '='*70)
print('2. HOURLY PATTERN ANALYSIS')
print('='*70)

hourly = df_data.groupby('hour').agg({
    'is_discharge': 'mean',
    'is_charge': 'mean',
    'price': 'mean',
    'power': 'mean'
}).reset_index()

print('\nHour  Discharge%  Charge%  AvgPrice  AvgPower  Pattern')
print('-'*60)
for _, row in hourly.iterrows():
    h = int(row['hour'])
    d = row['is_discharge'] * 100
    c = row['is_charge'] * 100
    p = row['price']
    pwr = row['power']

    # Determine pattern
    if d > 50:
        pattern = 'SELL'
    elif c > 40:
        pattern = 'BUY'
    else:
        pattern = 'MIXED'

    marker = ' <-- PEAK' if 17 <= h <= 21 else ''
    print(f'{h:02d}:00  {d:6.1f}%    {c:5.1f}%   {p:7.1f}   {pwr:+6.1f}    {pattern}{marker}')

# Check peak hour behavior
peak_hours = df_data[df_data['hour'].isin([17, 18, 19, 20, 21])]
peak_discharge = peak_hours['is_discharge'].mean() * 100
off_peak = df_data[df_data['hour'].isin([2, 3, 4, 5, 6])]
off_peak_charge = off_peak['is_charge'].mean() * 100

print(f'\n[CHECK] Peak (17-21) discharge: {peak_discharge:.1f}%', '[OK]' if peak_discharge > 40 else '[NEEDS WORK]')
print(f'[CHECK] Off-peak (02-06) charge: {off_peak_charge:.1f}%', '[OK]' if off_peak_charge > 30 else '[NEEDS WORK]')

# ========================================
# 3. DAM COMMITMENT ANALYSIS
# ========================================
print('\n' + '='*70)
print('3. DAM COMMITMENT ANALYSIS')
print('='*70)

dam_hours = df_data[abs(df_data['dam_commitment']) > 0.1].copy()
print(f'Hours with DAM commitment: {len(dam_hours)} / {len(df_data)} ({len(dam_hours)/len(df_data)*100:.1f}%)')

if len(dam_hours) > 0:
    # Discharge commitments (positive = must sell)
    discharge_commits = dam_hours[dam_hours['dam_commitment'] > 0]
    if len(discharge_commits) > 0:
        met_discharge = (discharge_commits['power'] > 0).mean() * 100
        partial = ((discharge_commits['power'] > 0) & (discharge_commits['power'] < discharge_commits['dam_commitment'])).mean() * 100
        print(f'\nDischarge commitments: {len(discharge_commits)}')
        print(f'  Met (any discharge): {met_discharge:.1f}%')
        print(f'  Partial delivery: {partial:.1f}%')

    # Charge commitments (negative = must buy)
    charge_commits = dam_hours[dam_hours['dam_commitment'] < 0]
    if len(charge_commits) > 0:
        met_charge = (charge_commits['power'] < 0).mean() * 100
        print(f'\nCharge commitments: {len(charge_commits)}')
        print(f'  Met (any charge): {met_charge:.1f}%')

# ========================================
# 4. SoC MANAGEMENT ANALYSIS
# ========================================
print('\n' + '='*70)
print('4. SoC MANAGEMENT ANALYSIS')
print('='*70)

soc_stats = df_data['soc'].describe()
print(f'SoC Statistics:')
print(f'  Mean: {soc_stats["mean"]:.2f}')
print(f'  Std: {soc_stats["std"]:.2f}')
print(f'  Min: {soc_stats["min"]:.2f}')
print(f'  Max: {soc_stats["max"]:.2f}')

# SoC by hour
soc_by_hour = df_data.groupby('hour')['soc'].mean()
morning_soc = soc_by_hour[6:12].mean()
evening_soc = soc_by_hour[17:22].mean()
night_soc = soc_by_hour[0:6].mean()

print(f'\nSoC by time of day:')
print(f'  Night (00-06): {night_soc:.2f}')
print(f'  Morning (06-12): {morning_soc:.2f}')
print(f'  Evening (17-22): {evening_soc:.2f}')

if evening_soc < morning_soc:
    print('  [OK] Evening SoC lower than morning (discharging during day)')
else:
    print('  [ISSUE] Evening SoC not lower - not discharging enough during peak')

# Check for SoC extremes
low_soc_hours = (df_data['soc'] < 0.1).sum()
high_soc_hours = (df_data['soc'] > 0.9).sum()
print(f'\nExtreme SoC events:')
print(f'  Very low (<10%): {low_soc_hours} hours ({low_soc_hours/len(df_data)*100:.1f}%)')
print(f'  Very high (>90%): {high_soc_hours} hours ({high_soc_hours/len(df_data)*100:.1f}%)')

# ========================================
# 5. EXTREME PRICE BEHAVIOR
# ========================================
print('\n' + '='*70)
print('5. EXTREME PRICE BEHAVIOR')
print('='*70)

# Very high prices (top 5%)
p95 = df_valid['price'].quantile(0.95)
extreme_high = df_valid[df_valid['price'] >= p95]
if len(extreme_high) > 0:
    high_discharge = extreme_high['is_discharge'].mean() * 100
    high_charge = extreme_high['is_charge'].mean() * 100
    print(f'Very high prices (>{p95:.0f} EUR, top 5%):')
    print(f'  Discharge: {high_discharge:.1f}%', '[OK]' if high_discharge > 60 else '[SHOULD BE HIGHER!]')
    print(f'  Charge: {high_charge:.1f}%', '[OK]' if high_charge < 10 else '[PROBLEM - buying expensive!]')

# Very low prices (bottom 5%)
p5 = df_valid['price'].quantile(0.05)
extreme_low = df_valid[df_valid['price'] <= p5]
if len(extreme_low) > 0:
    low_discharge = extreme_low['is_discharge'].mean() * 100
    low_charge = extreme_low['is_charge'].mean() * 100
    print(f'\nVery low prices (<{p5:.0f} EUR, bottom 5%):')
    print(f'  Charge: {low_charge:.1f}%', '[OK]' if low_charge > 50 else '[SHOULD BE HIGHER!]')
    print(f'  Discharge: {low_discharge:.1f}%', '[OK]' if low_discharge < 20 else '[PROBLEM - selling cheap!]')

# Negative prices
negative = df_data[df_data['price'] < 0]
if len(negative) > 0:
    neg_charge = negative['is_charge'].mean() * 100
    print(f'\nNegative prices ({len(negative)} hours):')
    print(f'  Charge: {neg_charge:.1f}%', '[EXCELLENT]' if neg_charge > 80 else '[MISSED OPPORTUNITY!]')

# ========================================
# 6. ACTION DISTRIBUTION
# ========================================
print('\n' + '='*70)
print('6. ACTION DISTRIBUTION')
print('='*70)

action_counts = df_data['action'].value_counts().sort_index()
print('Action histogram (0=full charge, 10=idle, 20=full discharge):')
for action, count in action_counts.items():
    pct = count / len(df_data) * 100
    bar = '#' * int(pct / 2)
    print(f'  {action:2d}: {bar:25s} {pct:5.1f}%')

idle_rate = (df_data['action'] == 10).mean() * 100
print(f'\nIdle rate (action=10): {idle_rate:.1f}%')

# ========================================
# 7. SUMMARY & RECOMMENDATIONS
# ========================================
print('\n' + '='*70)
print('7. SUMMARY & RECOMMENDATIONS')
print('='*70)

issues = []
strengths = []

# Check each metric
if q4_discharge > 50:
    strengths.append(f'Good Q4 discharge rate ({q4_discharge:.1f}%)')
else:
    issues.append(f'Q4 discharge too low ({q4_discharge:.1f}%) - should sell more when expensive')

if q1_charge > 40:
    strengths.append(f'Good Q1 charge rate ({q1_charge:.1f}%)')
else:
    issues.append(f'Q1 charge too low ({q1_charge:.1f}%) - should buy more when cheap')

if q4_charge < 20:
    strengths.append(f'Avoids buying expensive ({q4_charge:.1f}%)')
else:
    issues.append(f'Q4 charge too high ({q4_charge:.1f}%) - buying when expensive!')

if q1_discharge < 30:
    strengths.append(f'Avoids selling cheap ({q1_discharge:.1f}%)')
else:
    issues.append(f'Q1 discharge too high ({q1_discharge:.1f}%) - selling when cheap!')

if peak_discharge > 40:
    strengths.append(f'Good peak hour discharge ({peak_discharge:.1f}%)')
else:
    issues.append(f'Peak discharge low ({peak_discharge:.1f}%) - missing peak hours')

if len(extreme_high) > 0 and high_discharge > 60:
    strengths.append(f'Exploits extreme high prices ({high_discharge:.1f}%)')
else:
    issues.append(f'Not aggressive enough on extreme high prices ({high_discharge:.1f}%)')

if len(extreme_low) > 0 and low_charge > 50:
    strengths.append(f'Exploits extreme low prices ({low_charge:.1f}%)')
else:
    issues.append(f'Not aggressive enough on extreme low prices ({low_charge:.1f}%)')

print('\n[STRENGTHS]')
for s in strengths:
    print(f'  + {s}')

print('\n[NEEDS IMPROVEMENT]')
for i in issues:
    print(f'  - {i}')

print('\n[RECOMMENDATIONS]')
recommendations = []
if any('Q4 discharge' in i for i in issues):
    recommendations.append('Increase price_timing_bonus for high prices')
if any('Q1 charge' in i for i in issues):
    recommendations.append('Increase price_timing_bonus for low prices')
if any('extreme' in i.lower() for i in issues):
    recommendations.append('Add explicit bonus for extreme price exploitation')
if any('peak' in i.lower() for i in issues):
    recommendations.append('Consider longer training or different architecture')

for i, r in enumerate(recommendations, 1):
    print(f'  {i}. {r}')

if not recommendations:
    print('  Model is performing well! Consider production testing.')
