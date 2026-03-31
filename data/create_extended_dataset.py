"""
I create an extended 6-year unified training dataset (2021-2026)
by mapping feasible_data (2021-2023) columns to unified format
and concatenating with unified_multimarket_v2 (2024-2026).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

print('=== Creating 6-year unified dataset ===')
print()

# Step 1: Load datasets
feas = pd.read_csv('data/feasible_data_with_dam.csv', index_col='timestamp', parse_dates=True)
feas.index = pd.to_datetime(feas.index, utc=True).tz_localize(None)
old = feas[feas.index < '2024-01-01'].copy()
print(f'Feasible 2021-2023: {len(old)} rows ({old.index[0].date()} to {old.index[-1].date()})')

uni = pd.read_csv('data/unified_multimarket_training_v2.csv', index_col=0, parse_dates=True)
uni.index = pd.to_datetime(uni.index, utc=True).tz_localize(None)
print(f'Unified 2024-2026: {len(uni)} rows ({uni.index[0].date()} to {uni.index[-1].date()})')
print()

# Step 2: Map feasible columns to unified format
print('Mapping feasible columns...')
mapped = pd.DataFrame(index=old.index)

# I map direct columns
mapped['price'] = old['price']
mapped['solar'] = old.get('solar', 0)
mapped['wind_onshore'] = old.get('wind_onshore', 0)
mapped['dam_commitment'] = old.get('dam_commitment', 0)

# I map aFRR
mapped['afrr_up'] = old.get('aFRR_Up_Price', 100.0)
mapped['afrr_down'] = old.get('aFRR_Down_Price', 80.0)
mapped['afrr_cap_up_price'] = old.get('aFRR_Up_Price', 20.0)
mapped['afrr_cap_down_price'] = old.get('aFRR_Down_Price', 25.0)
mapped['afrr_activated_up_mwh'] = old.get('aFRR_Up_MW', 0).clip(lower=0)
mapped['afrr_activated_down_mwh'] = old.get('aFRR_Down_MW', 0).clip(lower=0)
mapped['afrr_cap_up_qty'] = 600.0
mapped['afrr_cap_down_qty'] = 120.0

# I map mFRR
mapped['mfrr_price_up'] = old.get('mFRR_Up', 100.0)
mapped['mfrr_price_down'] = old.get('mFRR_Down', 80.0)

# I estimate activation from imbalance spread
long_p = old.get('Long', old['price'])
short_p = old.get('Short', old['price'])
spread = (short_p - long_p).abs()
mapped['mfrr_activated_up_mwh'] = np.where(short_p > long_p, spread * 2, 0)
mapped['mfrr_activated_down_mwh'] = np.where(long_p > short_p, spread * 2, 0)
mapped['mfrr_requirements_up'] = 600.0
mapped['mfrr_requirements_down'] = 200.0

# I add imbalance and system data
mapped['imbalance_price'] = (long_p + short_p) / 2
mapped['net_imbalance_mw'] = np.where(short_p > long_p, 50, -50)
mapped['load_mw'] = 4000.0
mapped['res_total_mw'] = mapped['solar'] + mapped['wind_onshore']

# I add time features
mapped['hour'] = mapped.index.hour
mapped['day_of_week'] = mapped.index.dayofweek
mapped['month'] = mapped.index.month
mapped['is_weekend'] = (mapped.index.dayofweek >= 5).astype(int)

# I add derived features
mapped['mfrr_energy_spread'] = mapped['mfrr_price_up'] - mapped['mfrr_price_down']
mapped['afrr_capacity_spread'] = mapped['afrr_cap_up_price'] - mapped['afrr_cap_down_price']
mapped['mfrr_capacity_spread'] = 0.0
mapped['net_balancing_energy_mwh'] = mapped['mfrr_activated_up_mwh'] - mapped['mfrr_activated_down_mwh']
mapped['activated_energy_up_mwh'] = mapped['afrr_activated_up_mwh'] + mapped['mfrr_activated_up_mwh']
mapped['activated_energy_down_mwh'] = mapped['afrr_activated_down_mwh'] + mapped['mfrr_activated_down_mwh']
mapped['mfrr_spread'] = mapped['mfrr_energy_spread']

# I add rolling price stats
mapped['price_mean_24h'] = mapped['price'].rolling(24, min_periods=1).mean()
mapped['price_std_24h'] = mapped['price'].rolling(24, min_periods=1).std().fillna(10)
mapped['price_std_6h'] = mapped['price'].rolling(6, min_periods=1).std().fillna(10)
mapped['price_min_24h'] = mapped['price'].rolling(24, min_periods=1).min()
mapped['price_max_24h'] = mapped['price'].rolling(24, min_periods=1).max()

# I add synthetic XBID/IDA
mapped['intraday_bid'] = mapped['price'] - 2.0
mapped['intraday_ask'] = mapped['price'] + 2.0
mapped['intraday_spread'] = 4.0
mapped['intraday_volume'] = 50.0
mapped['xbid_price_bid'] = mapped['price'] - 1.5
mapped['xbid_price_ask'] = mapped['price'] + 1.5
mapped['xbid_spread'] = 3.0
mapped['xbid_bid'] = mapped['xbid_price_bid']
mapped['xbid_ask'] = mapped['xbid_price_ask']

for ida in ['ida1', 'ida2', 'ida3']:
    alpha = {'ida1': 0.85, 'ida2': 0.60, 'ida3': 0.30}[ida]
    mapped[f'{ida}_clearing_price'] = alpha * mapped['price'] + (1-alpha) * mapped['imbalance_price']
    mapped[f'{ida}_position'] = 0.0
mapped['net_ida_position'] = 0.0

# I add ISP prices
for isp in ['isp1', 'isp2', 'isp3']:
    mapped[f'{isp}_price_up'] = mapped['mfrr_price_up'] * 1.1
    mapped[f'{isp}_price_down'] = mapped['mfrr_price_down'] * 0.9

mapped['isp_cascade_spread_up'] = mapped['isp1_price_up'] - mapped['price']
mapped['isp_cascade_spread_down'] = mapped['price'] - mapped['isp1_price_down']
for s in ['isp12_spread_up', 'isp12_spread_down', 'isp23_spread_up', 'isp23_spread_down']:
    mapped[s] = 0.0
mapped['isp_avg_price_up'] = mapped['mfrr_price_up']
mapped['isp_avg_price_down'] = mapped['mfrr_price_down']
mapped['isp_cascade_spread_up_lag_24h'] = mapped['isp_cascade_spread_up'].shift(24)

# I add misc columns
mapped['free_bid_activation_base'] = 0.3
mapped['free_bid_reference_price'] = mapped['price']
mapped['mfrr_spread_mean_24h'] = mapped['mfrr_energy_spread'].rolling(24, min_periods=1).mean()
mapped['afrr_cap_percentile'] = 0.5
mapped['serbia_dam_price'] = mapped['price'] * 0.95

# I add forecast columns
mapped['dam_forecast_error'] = mapped['price'].rolling(24, min_periods=1).std().fillna(20)
mapped['dam_forecast_next_4h_mean'] = mapped['price'].rolling(4, min_periods=1).mean()
mapped['dam_forecast_next_4h_max'] = mapped['price'].rolling(4, min_periods=1).max()
mapped['dam_forecast_next_4h_min'] = mapped['price'].rolling(4, min_periods=1).min()
mapped['dam_forecast_spread'] = mapped['dam_forecast_next_4h_max'] - mapped['dam_forecast_next_4h_min']
mapped['dam_forecast_vs_current'] = mapped['dam_forecast_next_4h_mean'] - mapped['price']
mapped['forecast_confidence'] = 0.5
mapped['id_forecast_1h'] = mapped['price'] * 0.98
mapped['id_forecast_2h'] = mapped['price'] * 0.97
mapped['id_forecast_4h'] = mapped['price'] * 0.95
mapped['id_forecast_8h'] = mapped['price'] * 0.93
mapped['imbalance_forecast_dir'] = np.sign(mapped['net_imbalance_mw'])
mapped['imbalance_forecast_mag'] = mapped['net_imbalance_mw'].abs() / 100
mapped['mfrr_opportunity_score'] = mapped['mfrr_energy_spread'] / 100

mapped['hours_to_dam_peak'] = 6.0
mapped['hours_to_dam_trough'] = 12.0
mapped['is_peak_hour'] = ((mapped.index.hour >= 17) & (mapped.index.hour <= 21)).astype(float)
mapped['is_solar_hour'] = ((mapped.index.hour >= 9) & (mapped.index.hour <= 15)).astype(float)

# I add lag features
mapped['price_lag_1h'] = mapped['price'].shift(1)
mapped['price_lag_4h'] = mapped['price'].shift(4)
mapped['afrr_up_lag_1h'] = mapped['afrr_up'].shift(1)
mapped['afrr_up_lag_4h'] = mapped['afrr_up'].shift(4)
mapped['afrr_down_lag_1h'] = mapped['afrr_down'].shift(1)
mapped['afrr_down_lag_4h'] = mapped['afrr_down'].shift(4)
mapped['mfrr_price_up_lag_1h'] = mapped['mfrr_price_up'].shift(1)
mapped['mfrr_price_up_lag_4h'] = mapped['mfrr_price_up'].shift(4)
mapped['mfrr_price_down_lag_1h'] = mapped['mfrr_price_down'].shift(1)
mapped['mfrr_price_down_lag_4h'] = mapped['mfrr_price_down'].shift(4)
mapped['mfrr_energy_price_up_lag_1h'] = mapped['mfrr_price_up'].shift(1)
mapped['mfrr_energy_price_down_lag_1h'] = mapped['mfrr_price_down'].shift(1)
mapped['imbalance_price_lag_1h'] = mapped['imbalance_price'].shift(1)
mapped['mfrr_energy_spread_lag_1h'] = mapped['mfrr_energy_spread'].shift(1)
mapped['afrr_price_up_lag_1h'] = mapped['afrr_up'].shift(1)
mapped['afrr_price_down_lag_1h'] = mapped['afrr_down'].shift(1)
mapped['system_deviation_mwh_lag_1h'] = mapped['net_balancing_energy_mwh'].shift(1)
mapped['intraday_bid_lag_1h'] = mapped['intraday_bid'].shift(1)
mapped['intraday_ask_lag_1h'] = mapped['intraday_ask'].shift(1)
mapped['intraday_spread_lag_1h'] = 4.0
mapped['intraday_volume_lag_1h'] = 50.0
mapped['ida1_clearing_price_lag_1h'] = mapped['ida1_clearing_price'].shift(1)
mapped['ida2_clearing_price_lag_1h'] = mapped['ida2_clearing_price'].shift(1)
mapped['xbid_bid_lag_1h'] = mapped['xbid_price_bid'].shift(1)
mapped['xbid_ask_lag_1h'] = mapped['xbid_price_ask'].shift(1)

mapped = mapped.ffill().bfill().fillna(0)

# Step 3: Align columns
print('Aligning columns...')
missing = set(uni.columns) - set(mapped.columns)
for c in missing:
    mapped[c] = 0.0
mapped = mapped.reindex(columns=uni.columns, fill_value=0)
print(f'Mapped: {len(mapped)} rows, {len(mapped.columns)} columns')

# Step 4: Upsample hourly to 15-min
print('Upsampling 2021-2023 to 15-min...')
mapped_15 = mapped.resample('15min').asfreq()
for col in mapped_15.columns:
    if mapped_15[col].dtype in ['float64', 'float32', 'int64']:
        mapped_15[col] = mapped_15[col].interpolate(method='linear')
mapped_15 = mapped_15.ffill().bfill().fillna(0)
print(f'Upsampled: {len(mapped_15)} rows')

# Step 5: Concatenate
print('Concatenating...')
combined = pd.concat([mapped_15, uni], axis=0).sort_index()
combined = combined[~combined.index.duplicated(keep='last')]

# Step 6: Save
output = 'data/unified_multimarket_training_v3_6years.csv'
combined.to_csv(output)
years = (combined.index[-1] - combined.index[0]).days / 365
print()
print(f'=== DONE ===')
print(f'Output: {output}')
print(f'Rows: {len(combined):,}')
print(f'Range: {combined.index[0].date()} to {combined.index[-1].date()} ({years:.1f} years)')
print(f'Columns: {len(combined.columns)}')
