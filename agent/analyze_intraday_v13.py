"""
Detailed IntraDay Analysis for v13
Separates DAM-obligated actions from free IntraDay trading decisions.
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
print('v13 DETAILED INTRADAY ANALYSIS')
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

    # Get future prices (12h lookahead)
    future_prices = []
    future_dam = []
    for i in range(1, 13):
        if step + i < len(test_df):
            fp = test_df.iloc[step + i].get('price', current_price)
            fd = test_df.iloc[step + i].get('dam_commitment', 0)
            future_prices.append(fp)
            future_dam.append(fd)

    max_future = max(future_prices) if future_prices else current_price
    min_future = min(future_prices) if future_prices else current_price

    # Find hours to max/min price
    hours_to_max = future_prices.index(max_future) + 1 if future_prices and max_future > current_price else 0
    hours_to_min = future_prices.index(min_future) + 1 if future_prices and min_future < current_price else 0

    # Check upcoming DAM obligations
    upcoming_dam_discharge = sum(1 for d in future_dam if d > 0.1)
    upcoming_dam_charge = sum(1 for d in future_dam if d < -0.1)
    upcoming_dam_discharge_mw = sum(d for d in future_dam if d > 0.1)
    upcoming_dam_charge_mw = sum(abs(d) for d in future_dam if d < -0.1)

    data.append({
        'step': step,
        'hour': step % 24,
        'day': step // 24,
        'price': current_price,
        'max_future_12h': max_future,
        'min_future_12h': min_future,
        'price_upside': max_future - current_price,
        'price_downside': current_price - min_future,
        'hours_to_max': hours_to_max,
        'hours_to_min': hours_to_min,
        'power': info_dict.get('actual_mw', 0),
        'soc': info_dict.get('soc', 0.5),
        'dam_commitment': dam_commitment,
        'profit': info_dict.get('net_profit', 0),
        'action': action[0],
        'upcoming_dam_discharge': upcoming_dam_discharge,
        'upcoming_dam_charge': upcoming_dam_charge,
        'upcoming_dam_discharge_mw': upcoming_dam_discharge_mw,
        'upcoming_dam_charge_mw': upcoming_dam_charge_mw,
    })

    if done:
        obs = env.reset()

df_data = pd.DataFrame(data)
df_data['is_discharge'] = df_data['power'] > 0.1
df_data['is_charge'] = df_data['power'] < -0.1
df_data['is_idle'] = ~df_data['is_discharge'] & ~df_data['is_charge']
df_data['has_dam'] = abs(df_data['dam_commitment']) > 0.1
df_data['is_intraday'] = ~df_data['has_dam']

# ========================================
# 1. OVERVIEW: DAM vs IntraDay
# ========================================
print('\n' + '='*70)
print('1. OVERVIEW: DAM vs INTRADAY')
print('='*70)

dam_hours = df_data[df_data['has_dam']]
intraday_hours = df_data[df_data['is_intraday']]

print(f'Total hours analyzed: {len(df_data)}')
print(f'  DAM hours: {len(dam_hours)} ({len(dam_hours)/len(df_data)*100:.1f}%)')
print(f'  IntraDay hours: {len(intraday_hours)} ({len(intraday_hours)/len(df_data)*100:.1f}%)')

# ========================================
# 2. INTRADAY SELLS ANALYSIS
# ========================================
print('\n' + '='*70)
print('2. INTRADAY SELLS (Free Trading Discharges)')
print('='*70)

intraday_sells = df_data[(df_data['is_discharge']) & (df_data['is_intraday'])]
print(f'Total IntraDay sells: {len(intraday_sells)}')

if len(intraday_sells) > 0:
    # By timing quality
    print('\n--- Timing Quality ---')
    excellent = intraday_sells[intraday_sells['price_upside'] <= 5]
    good = intraday_sells[(intraday_sells['price_upside'] > 5) & (intraday_sells['price_upside'] <= 20)]
    ok = intraday_sells[(intraday_sells['price_upside'] > 20) & (intraday_sells['price_upside'] <= 40)]
    bad = intraday_sells[(intraday_sells['price_upside'] > 40) & (intraday_sells['price_upside'] <= 80)]
    terrible = intraday_sells[intraday_sells['price_upside'] > 80]

    print(f'  Excellent (<=5 EUR upside): {len(excellent)} ({len(excellent)/len(intraday_sells)*100:.1f}%)')
    print(f'  Good (5-20 EUR upside): {len(good)} ({len(good)/len(intraday_sells)*100:.1f}%)')
    print(f'  OK (20-40 EUR upside): {len(ok)} ({len(ok)/len(intraday_sells)*100:.1f}%)')
    print(f'  Bad (40-80 EUR upside): {len(bad)} ({len(bad)/len(intraday_sells)*100:.1f}%)')
    print(f'  Terrible (>80 EUR upside): {len(terrible)} ({len(terrible)/len(intraday_sells)*100:.1f}%)')

    # Price distribution
    print('\n--- Price Distribution of IntraDay Sells ---')
    q1, median, q3 = intraday_sells['price'].quantile([0.25, 0.5, 0.75])
    print(f'  Min: {intraday_sells["price"].min():.0f} EUR')
    print(f'  Q1: {q1:.0f} EUR')
    print(f'  Median: {median:.0f} EUR')
    print(f'  Q3: {q3:.0f} EUR')
    print(f'  Max: {intraday_sells["price"].max():.0f} EUR')
    print(f'  Mean: {intraday_sells["price"].mean():.0f} EUR')

    # SoC distribution
    print('\n--- SoC Distribution at IntraDay Sells ---')
    print(f'  Min: {intraday_sells["soc"].min():.2f}')
    print(f'  Mean: {intraday_sells["soc"].mean():.2f}')
    print(f'  Max: {intraday_sells["soc"].max():.2f}')

    # Bad timing analysis
    if len(bad) + len(terrible) > 0:
        bad_sells = pd.concat([bad, terrible])
        print('\n--- Analysis of Bad Timing Sells ---')
        print(f'  Count: {len(bad_sells)}')
        print(f'  Avg current price: {bad_sells["price"].mean():.0f} EUR')
        print(f'  Avg max future: {bad_sells["max_future_12h"].mean():.0f} EUR')
        print(f'  Avg missed upside: {bad_sells["price_upside"].mean():.0f} EUR')
        print(f'  Avg SoC: {bad_sells["soc"].mean():.2f}')
        print(f'  Avg hours to max: {bad_sells["hours_to_max"].mean():.1f}')

        # Check if they have upcoming DAM obligations
        with_upcoming_discharge = bad_sells[bad_sells['upcoming_dam_discharge'] > 0]
        with_upcoming_charge = bad_sells[bad_sells['upcoming_dam_charge'] > 0]
        print(f'\n  With upcoming DAM discharge: {len(with_upcoming_discharge)} ({len(with_upcoming_discharge)/len(bad_sells)*100:.1f}%)')
        print(f'  With upcoming DAM charge: {len(with_upcoming_charge)} ({len(with_upcoming_charge)/len(bad_sells)*100:.1f}%)')

        # High SoC that need to discharge for upcoming charge
        high_soc_with_charge = bad_sells[(bad_sells['soc'] > 0.6) & (bad_sells['upcoming_dam_charge'] > 0)]
        print(f'  High SoC (>60%) with upcoming charge: {len(high_soc_with_charge)} ({len(high_soc_with_charge)/len(bad_sells)*100:.1f}%) - JUSTIFIED')

        # Truly unjustified
        unjustified = bad_sells[(bad_sells['upcoming_dam_charge'] == 0) | (bad_sells['soc'] <= 0.6)]
        print(f'  Unjustified bad sells: {len(unjustified)} ({len(unjustified)/len(bad_sells)*100:.1f}%)')

# ========================================
# 3. INTRADAY BUYS ANALYSIS
# ========================================
print('\n' + '='*70)
print('3. INTRADAY BUYS (Free Trading Charges)')
print('='*70)

intraday_buys = df_data[(df_data['is_charge']) & (df_data['is_intraday'])]
print(f'Total IntraDay buys: {len(intraday_buys)}')

if len(intraday_buys) > 0:
    # By timing quality (for buys, downside means price will go down - bad to buy now)
    print('\n--- Timing Quality ---')
    excellent = intraday_buys[intraday_buys['price_downside'] <= 5]
    good = intraday_buys[(intraday_buys['price_downside'] > 5) & (intraday_buys['price_downside'] <= 20)]
    ok = intraday_buys[(intraday_buys['price_downside'] > 20) & (intraday_buys['price_downside'] <= 40)]
    bad = intraday_buys[(intraday_buys['price_downside'] > 40) & (intraday_buys['price_downside'] <= 80)]
    terrible = intraday_buys[intraday_buys['price_downside'] > 80]

    print(f'  Excellent (<=5 EUR downside): {len(excellent)} ({len(excellent)/len(intraday_buys)*100:.1f}%)')
    print(f'  Good (5-20 EUR downside): {len(good)} ({len(good)/len(intraday_buys)*100:.1f}%)')
    print(f'  OK (20-40 EUR downside): {len(ok)} ({len(ok)/len(intraday_buys)*100:.1f}%)')
    print(f'  Bad (40-80 EUR downside): {len(bad)} ({len(bad)/len(intraday_buys)*100:.1f}%)')
    print(f'  Terrible (>80 EUR downside): {len(terrible)} ({len(terrible)/len(intraday_buys)*100:.1f}%)')

    # Price distribution
    print('\n--- Price Distribution of IntraDay Buys ---')
    q1, median, q3 = intraday_buys['price'].quantile([0.25, 0.5, 0.75])
    print(f'  Min: {intraday_buys["price"].min():.0f} EUR')
    print(f'  Q1: {q1:.0f} EUR')
    print(f'  Median: {median:.0f} EUR')
    print(f'  Q3: {q3:.0f} EUR')
    print(f'  Max: {intraday_buys["price"].max():.0f} EUR')
    print(f'  Mean: {intraday_buys["price"].mean():.0f} EUR')

    # SoC distribution
    print('\n--- SoC Distribution at IntraDay Buys ---')
    print(f'  Min: {intraday_buys["soc"].min():.2f}')
    print(f'  Mean: {intraday_buys["soc"].mean():.2f}')
    print(f'  Max: {intraday_buys["soc"].max():.2f}')

# ========================================
# 4. INTRADAY IDLE ANALYSIS
# ========================================
print('\n' + '='*70)
print('4. INTRADAY IDLE (Free Trading - No Action)')
print('='*70)

intraday_idle = df_data[(df_data['is_idle']) & (df_data['is_intraday'])]
print(f'Total IntraDay idle: {len(intraday_idle)}')

if len(intraday_idle) > 0:
    # Check for missed opportunities
    could_sell = intraday_idle[(intraday_idle['soc'] > 0.2) & (intraday_idle['price'] > 150)]
    could_buy = intraday_idle[(intraday_idle['soc'] < 0.8) & (intraday_idle['price'] < 50)]

    print(f'\n--- Missed Opportunities ---')
    print(f'  Could have sold (SoC>20%, Price>150): {len(could_sell)}')
    print(f'  Could have bought (SoC<80%, Price<50): {len(could_buy)}')

# ========================================
# 5. HOURLY INTRADAY PATTERN
# ========================================
print('\n' + '='*70)
print('5. HOURLY INTRADAY PATTERN')
print('='*70)

hourly = intraday_hours.groupby('hour').agg({
    'is_discharge': 'mean',
    'is_charge': 'mean',
    'is_idle': 'mean',
    'price': 'mean',
    'step': 'count'
}).rename(columns={'step': 'count'}).reset_index()

print('Hour   Sell%  Buy%  Idle%  AvgPrice  Count')
print('-'*50)
for _, row in hourly.iterrows():
    h = int(row['hour'])
    sell = row['is_discharge'] * 100
    buy = row['is_charge'] * 100
    idle = row['is_idle'] * 100
    price = row['price']
    count = int(row['count'])
    marker = ' <-- PEAK' if 17 <= h <= 21 else (' <-- OFF-PEAK' if 2 <= h <= 6 else '')
    print(f'{h:02d}:00  {sell:5.1f}  {buy:5.1f}  {idle:5.1f}  {price:7.1f}  {count:5d}{marker}')

# ========================================
# 6. PRICE-BASED INTRADAY BEHAVIOR
# ========================================
print('\n' + '='*70)
print('6. PRICE-BASED INTRADAY BEHAVIOR')
print('='*70)

price_bins = [0, 50, 80, 100, 120, 150, 200, 500, 10000]
price_labels = ['<50', '50-80', '80-100', '100-120', '120-150', '150-200', '200-500', '>500']
intraday_hours['price_bin'] = pd.cut(intraday_hours['price'], bins=price_bins, labels=price_labels)

print('Price Range    Sell%  Buy%  Idle%  Count  Optimal')
print('-'*60)
for label in price_labels:
    subset = intraday_hours[intraday_hours['price_bin'] == label]
    if len(subset) > 0:
        sell = subset['is_discharge'].mean() * 100
        buy = subset['is_charge'].mean() * 100
        idle = subset['is_idle'].mean() * 100
        count = len(subset)

        # Determine optimal behavior
        if label in ['<50', '50-80']:
            optimal = 'BUY'
            status = '[OK]' if buy > sell else '[WRONG]'
        elif label in ['150-200', '200-500', '>500']:
            optimal = 'SELL'
            status = '[OK]' if sell > buy else '[WRONG]'
        else:
            optimal = 'MIXED'
            status = ''

        print(f'{label:12s}  {sell:5.1f}  {buy:5.1f}  {idle:5.1f}  {count:5d}  {optimal:5s} {status}')

# ========================================
# 7. PROFITABILITY BY INTRADAY ACTION
# ========================================
print('\n' + '='*70)
print('7. PROFITABILITY BY INTRADAY ACTION')
print('='*70)

intraday_sell_profit = intraday_sells['profit'].sum() if len(intraday_sells) > 0 else 0
intraday_buy_profit = intraday_buys['profit'].sum() if len(intraday_buys) > 0 else 0
intraday_idle_profit = intraday_idle['profit'].sum() if len(intraday_idle) > 0 else 0
dam_profit = dam_hours['profit'].sum() if len(dam_hours) > 0 else 0

print(f'IntraDay Sell profit: {intraday_sell_profit:,.0f} EUR ({len(intraday_sells)} actions)')
print(f'IntraDay Buy profit: {intraday_buy_profit:,.0f} EUR ({len(intraday_buys)} actions)')
print(f'IntraDay Idle profit: {intraday_idle_profit:,.0f} EUR ({len(intraday_idle)} actions)')
print(f'DAM Total profit: {dam_profit:,.0f} EUR ({len(dam_hours)} hours)')

print(f'\nTotal IntraDay: {intraday_sell_profit + intraday_buy_profit + intraday_idle_profit:,.0f} EUR')
print(f'Total DAM: {dam_profit:,.0f} EUR')

# ========================================
# 8. SAMPLE BAD TIMING SELLS
# ========================================
print('\n' + '='*70)
print('8. SAMPLE BAD TIMING SELLS (>40 EUR upside)')
print('='*70)

bad_timing_sells = intraday_sells[intraday_sells['price_upside'] > 40].head(20)
if len(bad_timing_sells) > 0:
    print('Step   Hour  Price  MaxFuture  Upside  SoC   UpcomingDAM')
    print('-'*65)
    for _, row in bad_timing_sells.iterrows():
        upcoming = f"D:{row['upcoming_dam_discharge']}/C:{row['upcoming_dam_charge']}"
        print(f"{row['step']:5d}  {row['hour']:02d}:00  {row['price']:5.0f}  {row['max_future_12h']:9.0f}  {row['price_upside']:6.0f}  {row['soc']:.2f}  {upcoming}")

# ========================================
# 9. SUMMARY
# ========================================
print('\n' + '='*70)
print('SUMMARY')
print('='*70)

total_intraday_trades = len(intraday_sells) + len(intraday_buys)
good_timing_sells = len(intraday_sells[intraday_sells['price_upside'] <= 20]) if len(intraday_sells) > 0 else 0
good_timing_buys = len(intraday_buys[intraday_buys['price_downside'] <= 20]) if len(intraday_buys) > 0 else 0

print(f'Total IntraDay trades: {total_intraday_trades}')
print(f'  Good timing sells: {good_timing_sells}/{len(intraday_sells)} ({good_timing_sells/max(len(intraday_sells),1)*100:.1f}%)')
print(f'  Good timing buys: {good_timing_buys}/{len(intraday_buys)} ({good_timing_buys/max(len(intraday_buys),1)*100:.1f}%)')
print(f'  Idle hours: {len(intraday_idle)}')

print('\nKey Observations:')
if len(intraday_sells) > 0:
    bad_sell_pct = len(intraday_sells[intraday_sells['price_upside'] > 40]) / len(intraday_sells) * 100
    print(f'  - {bad_sell_pct:.1f}% of IntraDay sells have >40 EUR upside (BAD)')
if len(intraday_buys) > 0:
    bad_buy_pct = len(intraday_buys[intraday_buys['price_downside'] > 40]) / len(intraday_buys) * 100
    print(f'  - {bad_buy_pct:.1f}% of IntraDay buys have >40 EUR downside (BAD)')
