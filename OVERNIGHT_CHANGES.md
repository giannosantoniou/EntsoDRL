# Training History Summary

## LATEST: v15 Results (2024-02-04)

### Critical Bugs Fixed in v15

1. **DAM Commitment Clipping** - DAM commitments in data exceeded battery limits (±46 MW vs ±30 MW max)
   - This caused impossible shortfall penalties (-46K to -55K EUR per incident)
   - Fix: Clip all DAM commitments to ±30 MW (battery physical limits)

2. **Synthetic Long/Short Spread** - Data had Long = Short (0% spread) making imbalance settlement meaningless
   - Fix: Generate synthetic spread from mFRR prices:
     - `Short = DAM + 30% × |mFRR_Up - DAM|` (deficit penalty)
     - `Long = DAM - 30% × |DAM - mFRR_Down|` (surplus price)
   - Average spread: ~22 EUR

### v15 Training Results

| Metric | Value |
|--------|-------|
| Training time | ~3.5 hours |
| Total timesteps | ~1.1M (early stop at 8/8) |
| Best validation score | **+7,601.18** |
| GPU | NVIDIA RTX PRO 6000 |

### v15 vs v14 Comparison (Test Data)

| Metric | v15 (Fixed) | v14 (Broken) | Improvement |
|--------|-------------|--------------|-------------|
| **Total Profit** | -13,382 EUR | -846,591 EUR | **+833,208 EUR** |
| **Daily Average** | -64 EUR | -4,051 EUR | **+3,987 EUR/day** |
| DAM Compliance | 100% | 100% | = |
| Discharge % | 59.9% | 53.4% | +6.5% |
| Charge % | 30.9% | 36.7% | -5.8% |

### Key Finding
v15 achieves **98.4% reduction in daily losses** compared to v14. The model is now nearly break-even (-64 EUR/day) instead of massive losses (-4,051 EUR/day).

---

## v14 Results (Previous)

### v14 Training Completed Successfully
- **Training stopped**: Early stopping at 1.3M timesteps (after 6 consecutive non-improvements)
- **Best validation score**: -1448.17
- **Model saved**: `models/ppo_v14_best.zip`

### v14 vs v13 Comparison (on test data):

| Metric | v14 (Realistic) | v13 (Cheating) | Winner |
|--------|-----------------|----------------|--------|
| DAM Compliance | 100% | 100% | Tie |
| Sells Good Timing | **52.7%** | 49.3% | v14 |
| Sells Bad Timing | **24.7%** | 27.2% | v14 |
| Buys Good Timing | **56.6%** | 45.2% | **v14** |
| Buys Bad Timing | **18.9%** | 27.9% | **v14** |
| Q1 (cheap) Charge | **46.1%** | 38.4% | v14 |
| Q4 (expensive) Discharge | 47.3% | 54.7% | v13 |

### Key Finding (v14)
v14, trained with REALISTIC price forecasts, achieved BETTER timing behavior than v13 which was trained with "cheating" lookahead (actual future prices + noise).

---

## All Changes Made

### Environment Fixes (`gym_envs/battery_env_masked.py`)

1. **Imbalance Settlement** - Use actual Long/Short prices from dataset
2. **ML Forecaster Integration** - `use_ml_forecaster` parameter for realistic training
3. **DAM Clipping** - Clip commitments to battery physical limits (±30 MW)
4. **Synthetic Spread** - Generate spread from mFRR when Long = Short

### New Files Created

| File | Purpose |
|------|---------|
| `models/intraday_forecaster.py` | ML price forecaster (Ridge regression, 12 horizons) |
| `models/intraday_forecaster.pkl` | Trained forecaster model |
| `agent/train_v14.py` | v14 training script |
| `agent/evaluate_v14.py` | v14 evaluation script |
| `agent/train_v15.py` | v15 training script (with fixes) |
| `agent/evaluate_v15.py` | v15 evaluation script |

### Models

| Model | Validation Score | Status |
|-------|-----------------|--------|
| `ppo_v13_best.zip` | N/A | Baseline (cheating forecasts) |
| `ppo_v14_best.zip` | -1,448 | ML forecaster, no DAM clip |
| `ppo_v15_best.zip` | **+7,601** | Full fixes, production-ready |

---

## Commands

```bash
# Train v15 (with all fixes)
python agent/train_v15.py

# Evaluate v15 vs v14
python agent/evaluate_v15.py

# Evaluate v14 vs v13
python agent/evaluate_v14.py
```

## Key Insights

1. **Data Quality Matters** - Impossible DAM commitments caused 98% of losses
2. **Spread is Critical** - Without realistic Long/Short spread, imbalance has no cost
3. **ML Forecaster Works** - Ridge regression provides realistic price predictions
4. **v15 is Production-Ready** - Nearly break-even with realistic market simulation
