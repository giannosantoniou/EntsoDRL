"""
Analysis: Are expensive IntraDay buys justified by upcoming DAM obligations?
If agent has upcoming DAM DISCHARGE, it NEEDS to charge now to fulfill it.
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
print('BUY JUSTIFICATION ANALYSIS')
print('Are expensive buys needed for upcoming DAM discharge?')
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

# Collect data
obs = env.reset()
data = []

print("Collecting data...")
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
    soc = info_dict.get('soc', 0.5)

    # Get future info (12h lookahead)
    future_prices = []
    future_dam = []
    for i in range(1, 13):
        if step + i < len(test_df):
            fp = test_df.iloc[step + i].get('price', current_price)
            fd = test_df.iloc[step + i].get('dam_commitment', 0)
            future_prices.append(fp)
            future_dam.append(fd)

    min_future = min(future_prices) if future_prices else current_price

    # Calculate upcoming DAM discharge obligations
    upcoming_dam_discharge_mw = sum(d for d in future_dam if d > 0.1)
    upcoming_dam_discharge_count = sum(1 for d in future_dam if d > 0.1)

    # Calculate energy needed for DAM discharge
    # If I have upcoming discharge of X MW for Y hours, I need X*Y MWh
    # Account for efficiency
    eff = 0.94
    energy_needed_for_dam = upcoming_dam_discharge_mw * 1.0 / np.sqrt(eff)  # MWh needed

    # Available energy above min SoC
    capacity = 146.0
    min_soc = 0.05
    available_energy = max(0, (soc - min_soc)) * capacity

    # Energy deficit for DAM
    energy_deficit = max(0, energy_needed_for_dam - available_energy)

    # Is charging justified?
    needs_charge_for_dam = energy_deficit > 5  # More than 5 MWh deficit

    data.append({
        'step': step,
        'hour': step % 24,
        'price': current_price,
        'min_future_12h': min_future,
        'price_downside': current_price - min_future,
        'power': info_dict.get('actual_mw', 0),
        'soc': soc,
        'dam_commitment': dam_commitment,
        'upcoming_dam_discharge_mw': upcoming_dam_discharge_mw,
        'upcoming_dam_discharge_count': upcoming_dam_discharge_count,
        'energy_needed_for_dam': energy_needed_for_dam,
        'available_energy': available_energy,
        'energy_deficit': energy_deficit,
        'needs_charge_for_dam': needs_charge_for_dam,
    })

    if done:
        obs = env.reset()

df_data = pd.DataFrame(data)
df_data['is_charge'] = df_data['power'] < -0.1
df_data['has_dam'] = abs(df_data['dam_commitment']) > 0.1
df_data['is_intraday'] = ~df_data['has_dam']

# Focus on IntraDay buys
intraday_buys = df_data[(df_data['is_charge']) & (df_data['is_intraday'])].copy()

print(f'\nTotal IntraDay buys: {len(intraday_buys)}')

# ========================================
# 1. EXPENSIVE BUYS ANALYSIS
# ========================================
print('\n' + '='*70)
print('1. EXPENSIVE BUYS (Price > 100 EUR)')
print('='*70)

expensive_buys = intraday_buys[intraday_buys['price'] > 100]
print(f'Total expensive buys: {len(expensive_buys)}')

if len(expensive_buys) > 0:
    justified = expensive_buys[expensive_buys['needs_charge_for_dam']]
    unjustified = expensive_buys[~expensive_buys['needs_charge_for_dam']]

    print(f'\n  Justified (need charge for DAM): {len(justified)} ({len(justified)/len(expensive_buys)*100:.1f}%)')
    print(f'  Unjustified (no DAM need): {len(unjustified)} ({len(unjustified)/len(expensive_buys)*100:.1f}%)')

    if len(justified) > 0:
        print(f'\n  Justified expensive buys:')
        print(f'    Avg price: {justified["price"].mean():.0f} EUR')
        print(f'    Avg SoC: {justified["soc"].mean():.2f}')
        print(f'    Avg upcoming DAM discharge: {justified["upcoming_dam_discharge_mw"].mean():.1f} MW')
        print(f'    Avg energy deficit: {justified["energy_deficit"].mean():.1f} MWh')

    if len(unjustified) > 0:
        print(f'\n  Unjustified expensive buys:')
        print(f'    Avg price: {unjustified["price"].mean():.0f} EUR')
        print(f'    Avg SoC: {unjustified["soc"].mean():.2f}')
        print(f'    Avg upcoming DAM discharge: {unjustified["upcoming_dam_discharge_mw"].mean():.1f} MW')

# ========================================
# 2. BAD TIMING BUYS ANALYSIS
# ========================================
print('\n' + '='*70)
print('2. BAD TIMING BUYS (>40 EUR downside available)')
print('='*70)

bad_timing_buys = intraday_buys[intraday_buys['price_downside'] > 40]
print(f'Total bad timing buys: {len(bad_timing_buys)}')

if len(bad_timing_buys) > 0:
    justified = bad_timing_buys[bad_timing_buys['needs_charge_for_dam']]
    unjustified = bad_timing_buys[~bad_timing_buys['needs_charge_for_dam']]

    print(f'\n  Justified (need charge for DAM): {len(justified)} ({len(justified)/len(bad_timing_buys)*100:.1f}%)')
    print(f'  Unjustified (no DAM need): {len(unjustified)} ({len(unjustified)/len(bad_timing_buys)*100:.1f}%)')

    if len(justified) > 0:
        print(f'\n  Justified bad timing buys:')
        print(f'    Avg current price: {justified["price"].mean():.0f} EUR')
        print(f'    Avg min future: {justified["min_future_12h"].mean():.0f} EUR')
        print(f'    Avg SoC: {justified["soc"].mean():.2f}')
        print(f'    Avg upcoming DAM: {justified["upcoming_dam_discharge_mw"].mean():.1f} MW')
        print(f'    Avg energy deficit: {justified["energy_deficit"].mean():.1f} MWh')

    if len(unjustified) > 0:
        print(f'\n  Unjustified bad timing buys:')
        print(f'    Avg current price: {unjustified["price"].mean():.0f} EUR')
        print(f'    Avg min future: {unjustified["min_future_12h"].mean():.0f} EUR')
        print(f'    Avg SoC: {unjustified["soc"].mean():.2f}')
        print(f'    Avg upcoming DAM: {unjustified["upcoming_dam_discharge_mw"].mean():.1f} MW')

# ========================================
# 3. PRICE BUCKET ANALYSIS
# ========================================
print('\n' + '='*70)
print('3. BY PRICE BUCKET')
print('='*70)

price_buckets = [
    (0, 50, 'Cheap (<50)'),
    (50, 80, 'Low (50-80)'),
    (80, 100, 'Medium (80-100)'),
    (100, 130, 'Above avg (100-130)'),
    (130, 180, 'High (130-180)'),
    (180, 10000, 'Very high (>180)')
]

print('Price Bucket         Total   Justified   Unjust.   Just%')
print('-'*60)
for low, high, label in price_buckets:
    bucket = intraday_buys[(intraday_buys['price'] >= low) & (intraday_buys['price'] < high)]
    if len(bucket) > 0:
        justified = bucket[bucket['needs_charge_for_dam']]
        unjustified = bucket[~bucket['needs_charge_for_dam']]
        just_pct = len(justified) / len(bucket) * 100
        print(f'{label:20s}  {len(bucket):5d}   {len(justified):5d}       {len(unjustified):5d}    {just_pct:5.1f}%')

# ========================================
# 4. LOW SOC ANALYSIS
# ========================================
print('\n' + '='*70)
print('4. LOW SoC BUYS (SoC < 30%)')
print('='*70)

low_soc_buys = intraday_buys[intraday_buys['soc'] < 0.30]
print(f'Total low SoC buys: {len(low_soc_buys)}')

if len(low_soc_buys) > 0:
    justified = low_soc_buys[low_soc_buys['needs_charge_for_dam']]
    unjustified = low_soc_buys[~low_soc_buys['needs_charge_for_dam']]

    print(f'  Justified (DAM need): {len(justified)} ({len(justified)/len(low_soc_buys)*100:.1f}%)')
    print(f'  Unjustified: {len(unjustified)} ({len(unjustified)/len(low_soc_buys)*100:.1f}%)')

    # Low SoC buys might also be justified by safety (even without DAM)
    # If SoC < 15%, buying is reasonable even without DAM
    very_low_soc = unjustified[unjustified['soc'] < 0.15]
    print(f'\n  Very low SoC (<15%) - safety justified: {len(very_low_soc)}')

# ========================================
# 5. SAMPLE ANALYSIS
# ========================================
print('\n' + '='*70)
print('5. SAMPLE: Unjustified Expensive Buys')
print('='*70)

unjust_expensive = intraday_buys[(intraday_buys['price'] > 120) & (~intraday_buys['needs_charge_for_dam'])].head(15)
if len(unjust_expensive) > 0:
    print('Step   Hour  Price  MinFuture  SoC   DAM_MW  Deficit')
    print('-'*60)
    for _, row in unjust_expensive.iterrows():
        print(f"{row['step']:5d}  {row['hour']:02d}:00  {row['price']:5.0f}  {row['min_future_12h']:9.0f}  {row['soc']:.2f}  {row['upcoming_dam_discharge_mw']:6.1f}  {row['energy_deficit']:.1f}")

# ========================================
# 6. SUMMARY
# ========================================
print('\n' + '='*70)
print('SUMMARY')
print('='*70)

total_buys = len(intraday_buys)
justified_total = intraday_buys['needs_charge_for_dam'].sum()
unjustified_total = total_buys - justified_total

print(f'Total IntraDay buys: {total_buys}')
print(f'  Justified by DAM needs: {justified_total} ({justified_total/total_buys*100:.1f}%)')
print(f'  Unjustified: {unjustified_total} ({unjustified_total/total_buys*100:.1f}%)')

# Focus on expensive unjustified
expensive_unjust = len(intraday_buys[(intraday_buys['price'] > 100) & (~intraday_buys['needs_charge_for_dam'])])
print(f'\nExpensive (>100 EUR) & Unjustified: {expensive_unjust} ({expensive_unjust/total_buys*100:.1f}%)')

# The real problem
bad_timing_unjust = len(intraday_buys[(intraday_buys['price_downside'] > 40) & (~intraday_buys['needs_charge_for_dam'])])
print(f'Bad timing (>40 downside) & Unjustified: {bad_timing_unjust} ({bad_timing_unjust/total_buys*100:.1f}%)')

print('\n' + '='*70)
print('CONCLUSION')
print('='*70)
if justified_total / total_buys > 0.5:
    print('Most expensive buys ARE justified by upcoming DAM obligations!')
    print('The agent is buying to prepare for DAM discharge.')
else:
    print('Many expensive buys are NOT justified by DAM.')
    print('The agent needs better price awareness for buying.')
