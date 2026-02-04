"""
Strategic Analysis: Why does agent sell early?
Check if there are upcoming DAM commitments that justify early selling.
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

# Load
model = MaskablePPO.load('models/ppo_v12_best.zip')
df = load_historical_data('data/feasible_data_with_balancing.csv')
battery_params = {'capacity_mwh': 146.0, 'max_discharge_mw': 30.0, 'efficiency': 0.94}

split_idx = int(len(df) * 0.8)
test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

def make_env():
    env = BatteryEnvMasked(test_df, battery_params, n_actions=21)
    env = ActionMasker(env, lambda e: e.action_masks())
    return env

env = DummyVecEnv([make_env])
env = VecNormalize.load('models/vec_normalize_v12.pkl', env)
env.training = False
env.norm_reward = False

obs = env.reset()
data = []

for step in range(min(7000, len(test_df) - 50)):
    inner_env = env.envs[0]
    while hasattr(inner_env, 'env'):
        if hasattr(inner_env, 'action_masks'):
            break
        inner_env = inner_env.env
    mask = inner_env.action_masks() if hasattr(inner_env, 'action_masks') else None

    action, _ = model.predict(obs, deterministic=True, action_masks=np.array([mask]) if mask is not None else None)
    obs, reward, done, info = env.step(action)

    info_dict = info[0]
    current_price = info_dict.get('dam_price', 0) or info_dict.get('best_bid', 0)

    # Current DAM commitment
    dam_now = test_df.iloc[step].get('dam_commitment', 0) if step < len(test_df) else 0

    # Future prices and DAM commitments (next 12 hours)
    future_prices = []
    future_dam_discharge = []  # Positive DAM = sell obligation
    future_dam_charge = []     # Negative DAM = buy obligation

    for i in range(1, 13):
        if step + i < len(test_df):
            fp = test_df.iloc[step + i].get('price', current_price)
            fd = test_df.iloc[step + i].get('dam_commitment', 0)
            future_prices.append(fp)
            if fd > 0.1:
                future_dam_discharge.append((i, fd))
            elif fd < -0.1:
                future_dam_charge.append((i, fd))

    max_future = max(future_prices) if future_prices else current_price

    # Total upcoming DAM obligations
    total_dam_discharge = sum([d[1] for d in future_dam_discharge])
    total_dam_charge = sum([abs(d[1]) for d in future_dam_charge])
    has_upcoming_dam_discharge = len(future_dam_discharge) > 0
    has_upcoming_dam_charge = len(future_dam_charge) > 0

    data.append({
        'step': step,
        'current_price': current_price,
        'max_future_12h': max_future,
        'price_upside': max_future - current_price,
        'power': info_dict.get('actual_mw', 0),
        'soc': info_dict.get('soc', 0.5),
        'is_discharge': info_dict.get('actual_mw', 0) > 0.1,
        'dam_now': dam_now,
        'has_dam_now': dam_now > 0.1,
        'has_upcoming_dam_discharge': has_upcoming_dam_discharge,
        'has_upcoming_dam_charge': has_upcoming_dam_charge,
        'total_upcoming_dam_discharge': total_dam_discharge,
        'total_upcoming_dam_charge': total_dam_charge,
    })

    if done:
        obs = env.reset()

df_data = pd.DataFrame(data)

print('='*70)
print('STRATEGIC ANALYSIS: WHY SELL EARLY?')
print('='*70)

# Focus on free choice "bad" decisions
bad_decisions = df_data[(df_data['price_upside'] > 40) &
                         (df_data['is_discharge']) &
                         (df_data['soc'] > 0.3) &
                         (~df_data['has_dam_now'])]  # No current DAM

total_bad = len(bad_decisions)
print(f'Total "early sells" (free choice, >40 EUR upside, SoC>30%): {total_bad}')

# Check upcoming DAM
with_upcoming_discharge = bad_decisions['has_upcoming_dam_discharge'].sum()
with_upcoming_charge = bad_decisions['has_upcoming_dam_charge'].sum()
has_any_upcoming = (bad_decisions['has_upcoming_dam_discharge'] | bad_decisions['has_upcoming_dam_charge']).sum()
no_upcoming_dam = total_bad - has_any_upcoming

print(f'\nBreakdown by upcoming DAM commitments (next 12h):')
print(f'  Has upcoming DAM DISCHARGE: {with_upcoming_discharge} ({with_upcoming_discharge/total_bad*100:.1f}%)')
print(f'  Has upcoming DAM CHARGE: {with_upcoming_charge} ({with_upcoming_charge/total_bad*100:.1f}%)')
print(f'  No upcoming DAM at all: {no_upcoming_dam} ({no_upcoming_dam/total_bad*100:.1f}%)')

# CASE 1: Upcoming DAM CHARGE - might justify early discharge
print('\n' + '='*70)
print('CASE 1: Has upcoming DAM CHARGE (need to make room)')
print('='*70)

upcoming_charge = bad_decisions[bad_decisions['has_upcoming_dam_charge']]
if len(upcoming_charge) > 0:
    print(f'Count: {len(upcoming_charge)}')
    print(f'Avg SoC: {upcoming_charge["soc"].mean():.2f}')
    print(f'Avg upcoming DAM charge: {upcoming_charge["total_upcoming_dam_charge"].mean():.1f} MW')

    # If high SoC and DAM charge coming, JUSTIFIED to discharge
    high_soc = upcoming_charge[upcoming_charge['soc'] > 0.6]
    print(f'\nWith SoC > 60%: {len(high_soc)}')
    print('  --> JUSTIFIED: Must discharge to make room for DAM charge!')

    low_soc = upcoming_charge[upcoming_charge['soc'] <= 0.6]
    print(f'With SoC <= 60%: {len(low_soc)}')
    print('  --> Less justified: has room, could wait')

# CASE 2: Upcoming DAM DISCHARGE
print('\n' + '='*70)
print('CASE 2: Has upcoming DAM DISCHARGE')
print('='*70)

upcoming_discharge = bad_decisions[bad_decisions['has_upcoming_dam_discharge'] & ~bad_decisions['has_upcoming_dam_charge']]
if len(upcoming_discharge) > 0:
    print(f'Count: {len(upcoming_discharge)}')
    print(f'Avg SoC: {upcoming_discharge["soc"].mean():.2f}')
    print('  --> Could wait: DAM discharge means sell later anyway')

# CASE 3: NO upcoming DAM
print('\n' + '='*70)
print('CASE 3: NO upcoming DAM (pure free trading)')
print('='*70)

no_dam = bad_decisions[~bad_decisions['has_upcoming_dam_discharge'] & ~bad_decisions['has_upcoming_dam_charge']]
if len(no_dam) > 0:
    print(f'Count: {len(no_dam)}')
    print(f'Avg SoC: {no_dam["soc"].mean():.2f}')
    print(f'Avg current price: {no_dam["current_price"].mean():.0f} EUR')
    print(f'Avg max future: {no_dam["max_future_12h"].mean():.0f} EUR')
    print(f'Avg missed upside: {no_dam["price_upside"].mean():.0f} EUR')
    print()
    print('  --> NO JUSTIFICATION: Should have waited!')

# SUMMARY
print('\n' + '='*70)
print('SUMMARY')
print('='*70)

justified = 0
if len(upcoming_charge) > 0:
    justified = len(upcoming_charge[upcoming_charge['soc'] > 0.6])

unjustified = len(no_dam) + len(upcoming_discharge)

print(f'Potentially justified (make room for DAM charge): {justified}')
print(f'Unjustified (no strategic reason): {unjustified}')
print(f'\nReal problem: {unjustified}/{total_bad} = {unjustified/total_bad*100:.1f}% of early sells')
