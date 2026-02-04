"""
Comprehensive v13 Evaluation
Compares timing behavior vs v12
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
print('v13 EVALUATION: TIMING AWARENESS')
print('='*70)

# Load model and data
model = MaskablePPO.load('models/ppo_v13_best.zip')
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
env = VecNormalize.load('models/vec_normalize_v13.pkl', env)
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
    current_price = info_dict.get('dam_price', 0) or test_df.iloc[step].get('price', 0)

    # Get DAM commitment from raw data
    dam_commitment = test_df.iloc[step].get('dam_commitment', 0)

    # Get future prices for timing analysis
    future_prices = []
    for i in range(1, 13):
        if step + i < len(test_df):
            fp = test_df.iloc[step + i].get('price', current_price)
            future_prices.append(fp)

    max_future = max(future_prices) if future_prices else current_price

    data.append({
        'step': step,
        'hour': step % 24,
        'day': step // 24,
        'price': current_price,
        'max_future_12h': max_future,
        'price_upside': max_future - current_price,
        'power': info_dict.get('actual_mw', 0),
        'soc': info_dict.get('soc', 0.5),
        'dam_commitment': dam_commitment,
        'profit': info_dict.get('net_profit', 0),
        'action': action[0],
        'lookahead_timing': info_dict.get('lookahead_timing', 0)
    })

    if done:
        obs = env.reset()

df_data = pd.DataFrame(data)
df_data['is_discharge'] = df_data['power'] > 0.1
df_data['is_charge'] = df_data['power'] < -0.1
df_data['is_idle'] = ~df_data['is_discharge'] & ~df_data['is_charge']
df_data['has_dam'] = abs(df_data['dam_commitment']) > 0.1

# ========================================
# 1. OVERALL METRICS
# ========================================
print('\n' + '='*70)
print('1. OVERALL METRICS')
print('='*70)

total_profit = df_data['profit'].sum()
avg_daily = df_data.groupby('day')['profit'].sum().mean()
discharge_hours = df_data['is_discharge'].sum()
charge_hours = df_data['is_charge'].sum()
idle_hours = df_data['is_idle'].sum()

print(f'Total Profit: {total_profit:,.0f} EUR')
print(f'Avg Daily Profit: {avg_daily:,.0f} EUR')
print(f'Discharge hours: {discharge_hours} ({discharge_hours/len(df_data)*100:.1f}%)')
print(f'Charge hours: {charge_hours} ({charge_hours/len(df_data)*100:.1f}%)')
print(f'Idle hours: {idle_hours} ({idle_hours/len(df_data)*100:.1f}%)')

# ========================================
# 2. DAM COMPLIANCE
# ========================================
print('\n' + '='*70)
print('2. DAM COMPLIANCE')
print('='*70)

dam_hours = df_data[df_data['has_dam']]
print(f'Hours with DAM commitment: {len(dam_hours)}')

if len(dam_hours) > 0:
    # Discharge commitments
    discharge_commits = dam_hours[dam_hours['dam_commitment'] > 0]
    if len(discharge_commits) > 0:
        met = (discharge_commits['power'] > 0).mean() * 100
        print(f'  Discharge commitments: {len(discharge_commits)} | Met: {met:.1f}%')

    # Charge commitments
    charge_commits = dam_hours[dam_hours['dam_commitment'] < 0]
    if len(charge_commits) > 0:
        met = (charge_commits['power'] < 0).mean() * 100
        print(f'  Charge commitments: {len(charge_commits)} | Met: {met:.1f}%')

# ========================================
# 3. TIMING ANALYSIS (KEY v13 METRIC)
# ========================================
print('\n' + '='*70)
print('3. TIMING ANALYSIS (v13 Key Metric)')
print('='*70)

# Free trading sells (no DAM obligation)
free_sells = df_data[(df_data['is_discharge']) & (~df_data['has_dam'])]
print(f'Free trading sells (no DAM): {len(free_sells)}')

if len(free_sells) > 0:
    # Analyze by price upside
    good_timing = free_sells[free_sells['price_upside'] <= 10]
    ok_timing = free_sells[(free_sells['price_upside'] > 10) & (free_sells['price_upside'] <= 40)]
    bad_timing = free_sells[free_sells['price_upside'] > 40]

    print(f'\nTiming Quality:')
    print(f'  Good (<=10 EUR upside): {len(good_timing)} ({len(good_timing)/len(free_sells)*100:.1f}%)')
    print(f'  OK (10-40 EUR upside): {len(ok_timing)} ({len(ok_timing)/len(free_sells)*100:.1f}%)')
    print(f'  Bad (>40 EUR upside): {len(bad_timing)} ({len(bad_timing)/len(free_sells)*100:.1f}%)')

    if len(bad_timing) > 0:
        print(f'\n  Bad timing sells:')
        print(f'    Avg current price: {bad_timing["price"].mean():.0f} EUR')
        print(f'    Avg max future: {bad_timing["max_future_12h"].mean():.0f} EUR')
        print(f'    Avg SoC: {bad_timing["soc"].mean():.2f}')
        print(f'    Avg missed upside: {bad_timing["price_upside"].mean():.0f} EUR')

# ========================================
# 4. PRICE QUARTILE BEHAVIOR
# ========================================
print('\n' + '='*70)
print('4. PRICE QUARTILE BEHAVIOR')
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
        print(f'{q:15s}: D:{discharge:5.1f}% | C:{charge:5.1f}% | I:{idle:5.1f}%')

# Key metrics
q4_discharge = df_valid[df_valid['quartile'] == 'Q4 (Expensive)']['is_discharge'].mean() * 100
q1_charge = df_valid[df_valid['quartile'] == 'Q1 (Cheap)']['is_charge'].mean() * 100

print(f'\n[CHECK] Q4 Discharge: {q4_discharge:.1f}%', '[GOOD]' if q4_discharge > 50 else '[NEEDS WORK]')
print(f'[CHECK] Q1 Charge: {q1_charge:.1f}%', '[GOOD]' if q1_charge > 40 else '[NEEDS WORK]')

# ========================================
# 5. HOURLY PATTERN
# ========================================
print('\n' + '='*70)
print('5. HOURLY PATTERN')
print('='*70)

hourly = df_data.groupby('hour').agg({
    'is_discharge': 'mean',
    'is_charge': 'mean',
    'price': 'mean'
}).reset_index()

peak_discharge = df_data[df_data['hour'].isin([17, 18, 19, 20, 21])]['is_discharge'].mean() * 100
off_peak_charge = df_data[df_data['hour'].isin([2, 3, 4, 5, 6])]['is_charge'].mean() * 100

print(f'Peak hours (17-21) discharge: {peak_discharge:.1f}%')
print(f'Off-peak (02-06) charge: {off_peak_charge:.1f}%')

# ========================================
# 6. SUMMARY
# ========================================
print('\n' + '='*70)
print('SUMMARY')
print('='*70)

bad_timing_pct = len(bad_timing) / max(len(free_sells), 1) * 100 if 'bad_timing' in dir() else 0

print(f'Total Profit: {total_profit:,.0f} EUR')
print(f'DAM Compliance: Check above')
print(f'Bad Timing Sells: {bad_timing_pct:.1f}%')
print(f'Q4 Discharge: {q4_discharge:.1f}%')
print(f'Q1 Charge: {q1_charge:.1f}%')
print(f'Peak Discharge: {peak_discharge:.1f}%')
