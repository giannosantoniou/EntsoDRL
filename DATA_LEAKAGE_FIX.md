# Data Leakage Analysis and Fix

## Issue Discovered

The original ML models and DRL training had **severe data leakage**, resulting in inflated profit estimates (8.9M EUR/year).

## Root Causes

### 1. Feature Leakage in ML Models

The following features contained information about the target variable (`mfrr_price_up`):

| Feature | Problem | Correlation |
|---------|---------|-------------|
| `mfrr_spread` | = `mfrr_price_up` - `mfrr_price_down` (contains target!) | 0.947 |
| `price_up_change_Xh` | = current - lag (contains current price) | High |
| `price_up_mean_Xh` | Rolling mean includes current row | High |
| `price_up_std_Xh` | Rolling std includes current row | High |
| `price_up_zscore_Xh` | Uses current price | High |

### 2. DRL Environment Leakage

In `gym_envs/battery_env_masked.py`:
- Lines 549-560: `max_future_price` computed from future data
- Reward calculator used `max_future_price` for timing bonus
- Agent could "see" future prices in observation space

### 3. Random Train/Test Split

- Used `sklearn.train_test_split()` with random sampling
- Time series requires **temporal** (chronological) split
- Random split allows model to learn from "future" data

## Evidence

| Metric | With Leakage | Without Leakage |
|--------|--------------|-----------------|
| ML R^2 | 0.98 | 0.40 |
| ML MAE | 7 EUR/MWh | 154 EUR/MWh |
| Feature Importance (mfrr_spread) | 99.52% | N/A (excluded) |
| Annual Profit Estimate | 8.9M EUR | 4.8M EUR |

## Fix Applied

### 1. Created Realistic ML Trainer (`backtesting/ml_strategy_realistic.py`)

Safe features (no leakage):
```python
SAFE_FEATURES = [
    # Time features (always known)
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_peak_hour', 'is_solar_hour', 'is_morning_ramp',

    # ONLY shifted prices (truly from the past)
    'price_up_lag_1h', 'price_up_lag_4h', 'price_up_lag_24h',

    # Lagged spread (from past only)
    'spread_lag_1h',

    # External features
    'net_cross_border_mw', 'hydro_level_normalized',
]
```

### 2. Created Realistic DRL Environment (`gym_envs/battery_env_realistic.py`)

Observation space (13 features, no future data):
- SoC, max_discharge, max_charge
- hour_sin, hour_cos, dow_sin, dow_cos, is_weekend
- price_lag_1h, price_lag_4h, price_lag_24h
- price_rolling_mean_24h (backward-looking only)
- dam_commitment

### 3. Temporal Train/Test Split

- 80% training, 20% test (chronological)
- No overlap between train and test periods

## Results Comparison

### Old (Leaky) Results
- Aggressive Hybrid: 8.9M EUR/year
- ML R^2: 0.98 (too good to be true!)

### New (Realistic) Results

| Strategy | Annual Profit | Cycles | Profit/Cycle |
|----------|---------------|--------|--------------|
| **Realistic DRL Agent** | **4,817,473 EUR** | 264.8 | 3,910 EUR |
| Time-Based | 3,734,309 EUR | 55.6 | 7,056 EUR |

The realistic DRL agent still outperforms Time-Based by **29%** without any data leakage.

## Files Created/Modified

- `backtesting/leakage_audit.py` - Comprehensive leakage audit
- `backtesting/ml_strategy_realistic.py` - Realistic ML strategies
- `backtesting/run_realistic_backtest.py` - Realistic backtest runner
- `gym_envs/battery_env_realistic.py` - Realistic DRL environment
- `agent/train_realistic.py` - Realistic training script
- `agent/evaluate_realistic.py` - Realistic evaluation script

## Conclusion

The 4.8M EUR/year is a **realistic, achievable** estimate based on:
- Only past price information
- Proper temporal splits
- Conservative assumptions

The previous 8.9M EUR estimate was inflated by ~45% due to data leakage.
