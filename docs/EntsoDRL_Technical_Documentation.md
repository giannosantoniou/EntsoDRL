# EntsoDRL — Technical & Business Documentation

> **Version**: 2.0 (v23 Production-Ready)
> **Date**: March 2026
> **System**: Deep Reinforcement Learning for Battery Energy Storage Trading
> **Target Market**: HEnEx (Hellenic Energy Exchange) — Adaptable to EU markets

---

# PART A: BUSINESS OVERVIEW

---

## 1. Executive Summary

EntsoDRL is an autonomous battery energy trading system that uses Deep Reinforcement Learning (DRL) to maximize revenue from a Battery Energy Storage System (BESS) participating in the Greek wholesale electricity market.

### What It Does

The system observes market prices, battery state, and forecasts, then decides — every 15 minutes — whether to charge, discharge, or hold idle. It participates across multiple energy markets simultaneously:

```
Market Data (prices, forecasts, system state)
        |
        v
+------------------+
|  DRL Agent       |  Observation: 45-112 features
|  (MaskablePPO)   |  Action: charge / discharge / idle (21 levels)
|  Neural Network  |  Constraints: physical limits enforced via action masking
+------------------+
        |
        v
Battery Dispatch Command (MW setpoint)
```

### Key Metrics

| Metric | Value |
|--------|-------|
| Battery Capacity | 146 MWh (30 MW inverter) |
| Markets Traded | DAM, aFRR, IDA1/2/3, XBID, mFRR |
| Decision Frequency | Every 15 minutes (96 decisions/day) |
| Training Data | 5+ years of historical Greek market data |
| Model Type | MaskablePPO (Proximal Policy Optimization with action masking) |
| Observation Features | 45 (legacy) to 112 (full-market mode) |
| Action Space | 21 discrete power levels (-30MW to +30MW) |

### Value Proposition

1. **Automated Trading**: No human intervention required for real-time dispatch
2. **Multi-Market Optimization**: Simultaneously optimizes across 6+ revenue streams
3. **Physical Safety**: Action masking guarantees physically valid operations
4. **Adaptable**: Architecture supports any EU energy market with minimal changes

---

## 2. Market Structure — HEnEx (Hellenic Energy Exchange)

### 2.1 Market Overview

The Greek electricity market operates under the EU Target Model framework. BESS operators can participate in multiple markets:

```
Timeline (Day D):

D-2        D-1                      D (Delivery Day)                D+1
 |          |                        |                               |
 |   12:00  |  15:00  22:00         00:00              23:00         |
 |   DAM    |  IDA1   IDA2          |  10:00            |    Settlement
 |  closes  |  gate   gate    Delivery IDA3             |    published
 |          |  close  close   starts  gate              |
 |          |                        close              |
 |          |        XBID continuous trading ---------->|
 |          |        aFRR activation (real-time) ------>|
 |          |        mFRR activation (TSO-controlled) ->|
```

### 2.2 Markets Detail

| Market | Type | Gate Closure | Resolution | Settlement | Commitment |
|--------|------|-------------|-----------|------------|------------|
| **DAM** | Auction | D-1 12:00 EET | Hourly | Pay-as-cleared | Mandatory |
| **IDA1** | Auction | D-1 15:00 EET | 15-min | Pay-as-cleared | Mandatory (locked) |
| **IDA2** | Auction | D-1 22:00 EET | 15-min | Pay-as-cleared | Mandatory (locked) |
| **IDA3** | Auction | D+0 10:00 EET | 15-min | Pay-as-cleared | Mandatory (locked) |
| **XBID** | Continuous | ~H-1 | 15-min | Pay-as-bid | Voluntary |
| **aFRR** | Capacity + Energy | 4h blocks | Real-time | max(aFRR, mFRR) | If selected |
| **mFRR** | TSO-activated | N/A | Per-event | Pay-as-cleared | Mandatory if accepted |
| **Free Bid** | Agent-submitted | N/A | Per-event | Pay-as-bid | If accepted |

### 2.3 Penalty Structure

| Violation | Penalty | Description |
|-----------|---------|-------------|
| DAM shortfall | 800 EUR/MWh | Failure to deliver committed energy |
| aFRR non-response | 500 EUR/MWh | Failure to activate when TSO calls |
| IDA violation | 1.5x clearing price | Deviation from locked IDA schedule |
| Physical violation | 100 EUR/event | Attempting impossible action (caught by masking) |

### 2.4 Capacity Cascade

When the battery participates in multiple markets, capacity is allocated in priority order:

```
Total Power (30 MW)
  |
  +-- DAM commitment (e.g., 10 MW sell) --> 20 MW remaining
       |
       +-- aFRR commitment (e.g., 5 MW up) --> 15 MW remaining
            |
            +-- IDA commitment (e.g., 3 MW) --> 12 MW remaining
                 |
                 +-- mFRR (TSO calls 4 MW) --> 8 MW remaining
                      |
                      +-- XBID (agent trades 5 MW) --> 3 MW remaining
                           |
                           +-- Free Bid (3 MW available)
```

> **Source**: `gym_envs/capacity_manager.py`, `gym_envs/market_executors/`

---

## 3. Battery Asset Specification

### 3.1 Physical Parameters

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| **Capacity** | 146.0 | MWh | Total energy storage |
| **Max Power** | 30.0 | MW | Bidirectional inverter rating |
| **Duration** | ~4.9 | hours | At rated power (146/30) |
| **Round-trip Efficiency** | 94% | - | LFP chemistry |
| **Charge Efficiency** | 96.95% | - | sqrt(0.94) per direction |
| **Discharge Efficiency** | 96.95% | - | sqrt(0.94) per direction |
| **Min SoC** | 5% | - | Deep discharge protection |
| **Max SoC** | 95% | - | Overcharge protection |
| **Usable Capacity** | 131.4 | MWh | 90% of 146 MWh |

### 3.2 SoC Update Equations

```
Charging:
  delta_soc = (power_mw * time_step_hours * sqrt(efficiency)) / capacity_mwh
  Example: 30MW * 0.25h * 0.9695 / 146 = +0.0498 (+4.98%)

Discharging:
  delta_soc = -(power_mw * time_step_hours / sqrt(efficiency)) / capacity_mwh
  Example: 30MW * 0.25h / 0.9695 / 146 = -0.0531 (-5.31%)

Note: Asymmetric — discharging costs more SoC than charging gains,
due to efficiency losses in both directions.
```

### 3.3 Degradation Model

| Parameter | Value | Description |
|-----------|-------|-------------|
| Cost per MWh | 12-15 EUR | Linear cost per MWh cycled |
| Daily cycle limit | 2.5 | Soft limit (penalty increases above) |
| Calendar aging | 0.005 * (SoC%-50)^2 | Quadratic SoC-dependent |
| Chemistry | LFP | ~8,000+ cycles expected lifetime |

> **Source**: `gym_envs/battery_env_unified.py`, `gym_envs/unified_reward_calculator.py`

---

## 4. Revenue Streams

### 4.1 Per-Market Revenue Model

| Market | Revenue Type | Typical Range | Formula |
|--------|-------------|---------------|---------|
| **DAM** | Energy arbitrage | 40-200 EUR/MWh | commitment * dam_price |
| **aFRR Capacity** | Availability payment | 15-25 EUR/MW/h | Selected capacity * cap_price |
| **aFRR Energy** | Activation payment | 80-150 EUR/MWh | Activated MW * max(aFRR_price, mFRR_price) |
| **IDA** | Spread arbitrage | 2-30 EUR/MWh | Position * clearing_price |
| **XBID** | Continuous arbitrage | 1-15 EUR/MWh | Volume * (ask - bid) / 2 |
| **mFRR** | TSO activation | 50-1000 EUR/MWh | Activated MW * mFRR_price |

### 4.2 Example Daily P&L

```
DAM Revenue:        2 cycles * 130 MWh/cycle * 40 EUR spread  = +10,400 EUR
aFRR Capacity:      30 MW * 8h selected * 20 EUR/MW/h         = +4,800 EUR
aFRR Energy:        10 MW * 2h activated * 120 EUR/MWh         = +2,400 EUR
XBID Arbitrage:     5 trades * 20 MWh avg * 5 EUR spread      = +500 EUR
Degradation:        260 MWh cycled * 12 EUR/MWh                = -3,120 EUR
                                                          Total: +14,980 EUR/day
                                                    Annualized: ~5.5M EUR/year
```

> Note: Actual results depend on market conditions. Paper trading recommended before production.

---

## 5. Data Sources

### 5.1 Overview

| Source | API | Frequency | Data | Timezone |
|--------|-----|-----------|------|----------|
| **ENTSO-E** | REST (entsoe-py) | Daily | DAM prices (GR, RS), load, RES | UTC → Athens |
| **ADMIE** | HTTP/Excel | Daily | ISP1/2/3, aFRR, mFRR, imbalance | Europe/Athens |
| **Open-Meteo** | REST (free) | Hourly | Wind, solar, temperature, clouds | UTC |
| **Yahoo Finance** | yfinance | Daily | TTF gas, EUA carbon | UTC |

### 5.2 ENTSO-E Transparency Platform

```python
# Connection (from data/entsoe_connector.py)
from entsoe import EntsoePandasClient
client = EntsoePandasClient(api_key=os.getenv('ENTSOE_API_KEY'))

# Day-Ahead Prices
prices_gr = client.query_day_ahead_prices('GR', start=start, end=end)
prices_rs = client.query_day_ahead_prices('RS', start=start, end=end)

# Load Forecast
load = client.query_load_forecast('GR', start=start, end=end)

# RES Generation
wind = client.query_wind_and_solar_forecast('GR', start=start, end=end)
```

### 5.3 ADMIE (Greek TSO)

```
Base URL: https://www.admie.gr/getOperationMarketFile

Files downloaded:
  ISP1 Results: YYYYMMDD_ISP1ISPResults_01.xlsx
  ISP2 Results: YYYYMMDD_ISP2ISPResults_01.xlsx
  ISP3 Results: YYYYMMDD_ISP3ISPResults_01.xlsx
  IMBABE:       YYYYMMDD_ImbOffsettingYYYYMMDD01v01.xlsx

Key rows (search by label in column A):
  "Greece Price Energy Up"   → Sell-side balancing price
  "Greece Price Energy Down" → Buy-side balancing price

Resolution: 30-min (2024) → 15-min (Oct 2025+)
```

### 5.4 Weather (Open-Meteo)

```
Endpoint: https://api.open-meteo.com/v1/forecast
Parameters: latitude=37.98, longitude=23.73 (Athens)
Variables: temperature_2m, wind_speed_100m, direct_radiation, cloud_cover
Resolution: Hourly, free API
```

### 5.5 Commodities (Yahoo Finance)

```python
import yfinance as yf
ttf = yf.download('TTF=F', start=start, end=end)   # Dutch TTF Natural Gas
eua = yf.download('CKZ24.F', start=start, end=end)  # EU Carbon Allowances
```

### 5.6 Data Column Schema

| Column | Source | Unit | Description |
|--------|--------|------|-------------|
| `price` | ENTSO-E | EUR/MWh | GR DAM clearing price |
| `rs_dam_price` | ENTSO-E | EUR/MWh | Serbia DAM price |
| `solar` | ENTSO-E | MW | Solar generation forecast |
| `wind_onshore` | ENTSO-E | MW | Wind generation forecast |
| `Long` | ADMIE | EUR/MWh | Imbalance price (positive imbalance) |
| `Short` | ADMIE | EUR/MWh | Imbalance price (negative imbalance) |
| `mFRR_Up` | ADMIE | EUR/MWh | mFRR upward activation price |
| `mFRR_Down` | ADMIE | EUR/MWh | mFRR downward activation price |
| `aFRR_Up_MW` | ADMIE | MW | aFRR upward activated volume |
| `aFRR_Down_MW` | ADMIE | MW | aFRR downward activated volume |
| `dam_commitment` | Generated | MW | DAM commitment for this period |
| `hour_sin` | Derived | [-1,1] | sin(2*pi*hour/24) |
| `hour_cos` | Derived | [-1,1] | cos(2*pi*hour/24) |

> **Source**: `PredictFuturePrices/config.py`, `PredictFuturePrices/data_merger.py`

---

# PART B: TECHNICAL ARCHITECTURE

---

## 6. System Architecture

### 6.1 Three-Layer Dependency-Inverted Design

```
                    +------------------------+
                    |   BatteryController    |  CONTROL LAYER
                    |   (Orchestrator)       |  Builds obs, calls strategy,
                    +---+----------------+---+  executes dispatch
                        |                |
              +---------v------+  +------v----------+
              | IDataProvider  |  | IDecisionStrategy|  INTERFACES
              | (abstract)     |  | (abstract)       |  (Dependency Inversion)
              +--------+-------+  +------+-----------+
                       |                 |
            +----------+----+    +-------+--------+
            |               |    |                |
  +---------v--+  +---------v+  +v-----------+  +v-----------+
  |Simulated   |  |Live      |  |RLAgent     |  |RuleBased   |  IMPLEMENTATIONS
  |DataProvider|  |DataProvider| |Strategy    |  |Strategy    |
  |(CSV/train) |  |(APIs/prod)|  |(model.zip) |  |(fallback)  |
  +------------+  +-----------+  +------------+  +------------+
```

### 6.2 SOLID Principles Applied

| Principle | Implementation |
|-----------|---------------|
| **S**ingle Responsibility | Each market executor handles one market only |
| **O**pen/Closed | New market = create MarketStage + register in pipeline |
| **L**iskov Substitution | All strategies implement IDecisionStrategy |
| **I**nterface Segregation | IDataProvider has focused methods |
| **D**ependency Inversion | Controller depends on abstractions, not implementations |

### 6.3 Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Strategy** | `decision_strategy.py` | Swap AI/Rule/Random without code change |
| **Factory** | `config.py` | `create_data_provider()` based on mode |
| **Pipeline** | `market_stages.py` | step() delegates to 7 sequential stages |
| **DTO** | `step_context.py` | Cross-stage data flow without coupling |
| **Strangler Fig** | `battery_env_unified.py` | Gradual refactoring of god object |

### 6.4 File Map

```
EntsoDRL/
+-- config.py                              Mode switching, factories
+-- main.py                                Production entry point
+-- agent/
|   +-- train_v23_production_ready.py      Training script (current)
|   +-- decision_strategy.py               IDecisionStrategy + implementations
|   +-- battery_controller.py              Production control loop
|   +-- feature_engineer.py                Observation construction (legacy env)
+-- gym_envs/
|   +-- battery_env_unified.py             Main training environment (984 lines)
|   +-- battery_env_masked.py              Legacy environment (v22)
|   +-- unified_reward_calculator.py       Multi-market reward (unified)
|   +-- reward_calculator.py               Single-market reward (legacy)
|   +-- data_preparer.py                   Data validation, precomputed arrays
|   +-- capacity_manager.py                Cascade capacity limits
|   +-- observation_builder.py             Feature vector (70-112 features)
|   +-- mask_builder.py                    Action mask generation
|   +-- step_context.py                    Pipeline DTO
|   +-- market_stage.py                    Stage Protocol (interface)
|   +-- market_stages.py                   7 stage implementations
|   +-- market_executors/
|       +-- __init__.py                    DTOs
|       +-- dam_executor.py                DAM settlement
|       +-- afrr_executor.py               aFRR commitment/activation
|       +-- ida_executor.py                IDA gates, schedules
|       +-- trading_executor.py            mFRR, XBID, FreeBid
+-- data/
|   +-- data_provider.py                   IDataProvider interface
|   +-- entsoe_connector.py                ENTSO-E API client
|   +-- admie_data_fetcher.py              ADMIE API client
|   +-- dam_commitment_generator.py        Synthetic DAM commitments
|   +-- feasible_data_with_dam_15min.csv   Training data (15-min)
+-- PredictFuturePrices/
|   +-- config.py                          Forecaster configuration
|   +-- forecasters/                       LightGBM models
|   +-- collectors/                        Data collection automation
|   +-- data/                              Collected raw data
+-- models/
|   +-- ppo_v23_production/                Current production model
+-- production/
|   +-- paper_trading_v2.py                Paper trading simulator
|   +-- monitor_dashboard.py               Real-time monitoring
+-- tests/
    +-- test_unified_env.py                205 tests
    +-- test_unified_env_sac.py            43 tests
    +-- test_components.py                 50 component tests
```

---

## 7. Environment Design

### 7.1 Observation Space

The agent observes 70 to 112 features depending on enabled feature groups:

| Group | Features | Count | Enabled By |
|-------|----------|-------|------------|
| 1. Battery State | SoC, max_discharge, max_charge, cycles_remaining | 4 | Always |
| 2. Market Prices | DAM, imbalance, mFRR, spreads, HVR | 13 | Always |
| 3. Time Encoding | Hour sin/cos, day-of-week sin/cos | 4 | Always |
| 4. DAM Lookahead | Current + 12h future commitments | 13 | Always |
| 5. Price Lookahead | 12-hour price forecasts | 12 | Always |
| 6. aFRR State | Capacity prices, activation, commitment | 8 | Always |
| 7. Risk/Imbalance | Reserve, shortfall, momentum, volatility | 8 | Always |
| 8. Market Phase | IDA, stress, RES, mFRR signals | 4 | Always |
| 9. Timing Signals | Price vs typical, peak flags | 4 | Always |
| **Base Total** | | **70** | |
| 10a. Forecasts | IntraDay 1h/4h/12h, load, RES | 9 | `enable_forecast` |
| 10b. Market Forecasts | DAM/ID/Imbalance/IDA/ISP1 features | 20 | `enable_market_forecast` |
| 11-12. Full Market | IDA/XBID/FreeBid features | 13 | `enable_full_market` |
| **Max Total** | | **112** | All flags on |

All features are:
- **Backward-looking** (no future data leakage)
- **NaN-protected** (NaN -> 0, Inf -> clip)
- **Normalized** (divided by constants or clipped to [-1, 1])

> **Source**: `gym_envs/observation_builder.py`

### 7.2 Action Space

**Legacy mode** — `MultiDiscrete([5, 5, 11, 11])` = 3,025 combinations:

| Dimension | Values | Meaning |
|-----------|--------|---------|
| [0] aFRR Commitment | [0%, 25%, 50%, 75%, 100%] | Capacity offered to aFRR |
| [1] aFRR Price Tier | [0.7, 0.85, 1.0, 1.15, 1.3] | Bid aggressiveness |
| [2] IntraDay Action | 11 levels: -1.0 to +1.0 | XBID/IDA trading |
| [3] mFRR Quantity | 11 levels: -1.0 to +1.0 | mFRR participation |

**Simplified mode** (v22/v23 masked env) — `Discrete(21)`:

```
Action  0 → -30.0 MW (full charge / buy)
Action  5 → -15.0 MW
Action 10 →   0.0 MW (idle)
Action 15 → +15.0 MW
Action 20 → +30.0 MW (full discharge / sell)
```

### 7.3 Time Resolution

| Version | time_step_hours | Steps/day | Lookahead |
|---------|----------------|-----------|-----------|
| v22 | 1.0 (hourly) | 24 | 12 steps = 12h |
| v23 | 0.25 (15-min) | 96 | 48 steps = 12h |

> **Source**: `gym_envs/battery_env_masked.py`, `gym_envs/battery_env_unified.py`

---

## 8. Reward Function

### 8.1 Components

```
Reward = (Revenue - Penalties + Bonuses) * reward_scale

Revenue:
  + DAM Revenue:          dam_commitment * dam_price
  + Balancing Revenue:    imbalance_mw * (bid or ask price)
  + aFRR Capacity:        capacity_mw * capacity_price
  + aFRR Energy:          activation_mw * max(afrr_price, mfrr_price)
  + mFRR Revenue:         activation_mw * mfrr_price
  + XBID Revenue:         volume * (sell_price or buy_price)

Penalties:
  - Degradation:          |actual_mw| * deg_cost_per_mwh
  - DAM Shortfall:        shortfall_mw * 800 EUR/MWh
  - aFRR Non-Response:    undelivered_mw * 500 EUR/MWh
  - IDA Violation:        deviation_mw * 1.5 * clearing_price
  - SoC Aging:            0.005 * (|SoC% - 50|)^2

Bonuses:
  + DAM Compliance:       5 EUR/MWh (if perfect delivery)
  + aFRR Response:        10 EUR/MWh (if fast activation)
  + Price Timing:         bonus for sell-high / buy-low

reward_scale = 0.01 (normalizes for neural network stability)
```

### 8.2 Imbalance Settlement

```python
# From reward_calculator.py
imbalance_mw = actual_physical - dam_commitment

if imbalance_mw > 0:  # Delivered more than committed
    balancing_revenue = imbalance_mw * best_bid     # Lower price
elif imbalance_mw < 0:  # Delivered less than committed
    balancing_revenue = imbalance_mw * best_ask     # Higher price (penalty)
```

### 8.3 Shortfall Penalty (Data Leakage Prevention)

The shortfall penalty uses a **fixed value** (800 EUR/MWh) instead of actual imbalance prices, because imbalance prices are published ex-post and would constitute data leakage if used during training.

> **Source**: `gym_envs/reward_calculator.py`, `gym_envs/unified_reward_calculator.py`

---

## 9. Action Masking

### 9.1 Physical Constraints

Action masking prevents the agent from selecting physically impossible actions:

```python
# Available energy for discharge
available_discharge_mwh = (current_soc - min_soc) * capacity_mwh

# Available energy for charge
available_charge_mwh = (max_soc - current_soc) * capacity_mwh

# Convert to power limits for this time step
max_discharge_mw = min(max_power, available_discharge_mwh * sqrt(eff) / time_step_hours)
max_charge_mw = min(max_power, available_charge_mwh / sqrt(eff) / time_step_hours)

# Mask: only actions within these limits are allowed
for action in range(n_actions):
    power = action_to_mw(action)
    if power > max_discharge_mw:  mask[action] = False
    if power < -max_charge_mw:    mask[action] = False

# Idle is ALWAYS allowed
mask[idle_action] = True
```

### 9.2 Three Safety Layers

```
Layer 1: MaskablePPO (model level)
  Masked logits → probability = 0 for invalid actions
  100% prevention at selection time

Layer 2: ActionMasker wrapper
  If invalid action somehow passes, converts to idle
  Secondary safety net

Layer 3: Physics clamp
  np.clip(soc, min_soc, max_soc)
  Last resort — prevents hardware damage
```

> **Source**: `gym_envs/battery_env_masked.py`, `gym_envs/mask_builder.py`

---

## 10. Neural Network & PPO

### 10.1 Architecture

```
Observation (45-112 features)
        |
   +----v----+        +----------+
   |  ACTOR  |        |  CRITIC  |
   | [512]   |        | [512]    |
   | [256]   |        | [256]    |
   | [128]   |        | [128]    |
   +----+----+        +-----+----+
        |                   |
   Action Probs (21)    Value (1)
   (after masking)      (expected return)
```

### 10.2 Hyperparameters (v23 Production)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `learning_rate` | 1.5e-4 (linear decay) | Gentle updates, low KL |
| `gamma` | 0.995 | ~200 step horizon (50h at 15-min) |
| `gae_lambda` | 0.95 | Balance bias/variance in advantages |
| `n_steps` | 4,096 | Buffer = 40,960 with 10 envs |
| `batch_size` | 512 | Stable gradient estimates |
| `n_epochs` | 4 | Minimal policy drift per rollout |
| `clip_range` | 0.25 | Slightly wider for healthy updates |
| `ent_coef` | 0.08 | Sustained exploration |
| `target_kl` | 0.03 | Auto-stop if KL exceeds threshold |
| `net_arch` | [512, 256, 128] | Wide first layer for market regimes |
| `device` | CUDA | GPU acceleration |

### 10.3 Training Diagnostics

| Metric | Healthy Range | Warning |
|--------|--------------|---------|
| `entropy_loss` | > -1.5 (abs > 1.5) | < -0.8 → collapse risk |
| `clip_fraction` | < 0.25 | > 0.30 → aggressive updates |
| `approx_kl` | < 0.03 | > 0.05 → policy instability |
| `explained_variance` | > 0.5 | < 0.2 → critic not learning |
| `value_loss` | Decreasing | Increasing → divergence |

> **Source**: `agent/train_v23_production_ready.py`

---

## 11. Training Pipeline

### 11.1 Pipeline Flow

```
CSV Data (15-min resolution)
    |
    v
+-- Train/Val/Test Split (70/15/15%) --+
|                                       |
v                                       v
BatteryEnvMasked (x10 parallel)    BatteryEnvMasked (x1)
+ ActionMasker wrapper              (validation, deterministic)
+ SubprocVecEnv (10 envs)
+ VecNormalize
    |
    v
MaskablePPO.learn(3M-5M steps)
    |
    +-- CheckpointCallback (every 200K steps)
    +-- ValidationCallback (every 10K steps, patience=30)
    |
    v
best_model.zip + vec_normalize.pkl
```

### 11.2 Key Training Features

- **Randomized Episode Starts**: Each episode begins at a random position in the dataset (`min_episode_length=672` = 1 week at 15-min)
- **Randomized Initial SoC**: 50% +/- 10% for robustness
- **ML Forecaster**: Uses trained LightGBM model for price lookahead (not oracle)
- **VecNormalize**: Online normalization of observations (mean/std tracking)
- **Early Stopping**: Stops if no validation improvement for 30 evaluations

### 11.3 Curriculum Learning (Unified Env)

```
Steps 0 - 500K:    degradation = 0 EUR/MWh  (learn basic trading)
Steps 500K - 1M:   degradation ramps 0→15   (learn efficiency)
Steps 1M+:         degradation = 15 EUR/MWh (full cost)
```

> **Source**: `agent/train_v23_production_ready.py`

---

# PART C: FORECASTING

---

## 12. Forecasting System Overview

The `PredictFuturePrices` package provides market forecasts used as observation features during training and production.

### 12.1 Model Inventory

| Forecaster | Models | Resolution | Horizons | Algorithm |
|-----------|--------|-----------|----------|-----------|
| **DAM** | 288 | 15-min | 96 periods (24h) | LightGBM |
| **Load** | 96 | 15-min | 96 periods (24h) | LightGBM |
| **IDA** | 3 | 15-min | Per auction (1/2/3) | LightGBM |
| **XBID** | 4 | 15-min | 1h, 2h, 4h, 8h | LightGBM |
| **Total** | **391** | | | |

### 12.2 Data Flow

```
Collectors (daily cron)
  +-- ENTSO-E → GR/RS DAM prices, load, RES
  +-- ADMIE → ISP1/2/3, mFRR, imbalance
  +-- Open-Meteo → wind, solar, temperature
  +-- yfinance → TTF gas, EUA carbon
      |
      v
Data Merger → merged_training.csv (30+ columns)
      |
      v
Train Script → 391 LightGBM models (.pkl files)
      |
      v
Predict Script → JSON/CSV forecasts for next 24h
      |
      v
Accuracy Tracker → Rolling 7d/30d MAPE metrics
```

> **Source**: `PredictFuturePrices/config.py`, `PredictFuturePrices/train.py`

---

## 13. DAM Forecaster

### 13.1 Architecture

- **288 models** = 96 periods/day x 3 quantiles (P10, P50, P90)
- Model key format: `p{period}_q{quantile}` (e.g., `p0_q50`, `p95_q90`)
- **LightGBM** with 500 estimators, max_depth=7, learning_rate=0.05

### 13.2 Feature Set (29 features)

| Category | Features | Count |
|----------|----------|-------|
| Serbia D+1 | mean, max, min, spread | 4 |
| GR DAM lags | Same period D-1/D-2/D-3/D-7 | 4 |
| Rolling stats | 24h mean, std, min, max | 4 |
| Calendar | Period sin/cos, DoW sin/cos, month sin/cos | 6 |
| Load/RES | Load forecast, RES forecast (normalized) | 2 |
| Cross-border | RS-GR spread | 1 |
| Weather | Temperature, wind_100m, solar, cloud_cover | 4 |
| Commodities | TTF gas, EUA carbon | 2 |
| Error lags | Load error lag, RES error lag | 2 |

### 13.3 Backward Compatibility

Models trained with hourly resolution (key: `h{N}_q{Q}`) are detected and a warning is issued. The system supports both formats.

> **Source**: `PredictFuturePrices/forecasters/dam_forecaster.py`

---

## 14. IDA / XBID / Load Forecasters

### 14.1 IDA Forecaster

| Auction | Alpha (DAM weight) | Noise sigma | Features |
|---------|-------------------|-------------|----------|
| IDA1 | 0.85 | 3 EUR | 30 |
| IDA2 | 0.60 | 3.5 EUR | 30 |
| IDA3 | 0.30 | 4 EUR | 30 |

**Blending formula**: `ida_price = alpha * DAM + (1-alpha) * ISP1 + OU_noise(sigma)`

### 14.2 XBID Forecaster

- **4 models**: 1h, 2h, 4h, 8h horizons
- **37 features**: DAM, lagged ISP, order book dynamics
- **Price model**: `mid = DAM + 10% ISP1_signal + OU(sigma=5)`, spread 1-8 EUR

### 14.3 Load Forecaster

- **96 models**: One per 15-min period
- **20 features**: Calendar, lagged load, weather, RES
- **No quantiles**: Load is more predictable (MAPE < 3%)

> **Source**: `PredictFuturePrices/forecasters/`

---

# PART D: PRODUCTION & DEPLOYMENT

---

## 15. Production Architecture

### 15.1 BatteryController (Production Loop)

```python
# Simplified production loop (from agent/battery_controller.py)
while True:
    market = provider.get_current_market_state()      # Live API data
    battery = provider.get_battery_telemetry()         # SCADA/BMS
    forecast = provider.get_forecast(12)               # ML forecasts

    observation = build_observation(market, battery, forecast)  # 45 features
    mask = calculate_action_mask(battery)                       # Physical limits

    action = strategy.predict_action(observation, mask)         # RL decision
    power_mw = action_to_mw(action)                            # Convert

    provider.dispatch_battery(power_mw)                        # Execute!
    sleep(interval_seconds)                                     # Wait 15 min
```

### 15.2 Paper Trading

```python
# From production/paper_trading_v2.py
class PaperTrader:
    # Uses REAL live data
    # Simulates battery state internally (no actual dispatch)
    # Logs all decisions and simulated P&L
    # Recommended: 2-4 weeks before production
```

### 15.3 Strategy Fallback

```python
try:
    strategy = RLAgentStrategy(model_path='models/best_model.zip')
except Exception:
    strategy = RuleBasedStrategy()  # IF price > 150: sell, IF < 50: buy
    logger.warning("Fell back to rule-based strategy")
```

> **Source**: `agent/battery_controller.py`, `agent/decision_strategy.py`

---

## 16. Configuration

### 16.1 Mode Switching

```python
# config.py
class Mode(Enum):
    TRAINING = "training"      # SimulatedDataProvider (CSV)
    PRODUCTION = "production"  # LiveDataProvider (APIs)

class TradingMode(Enum):
    DAM_ONLY = "dam_only"      # DAM + IntraDay
    BALANCING = "balancing"    # aFRR + mFRR
    UNIFIED = "unified"        # All markets
```

### 16.2 Feature Flags

| Flag | Effect | Default |
|------|--------|---------|
| `enable_forecast` | +9 observation features | False |
| `enable_market_forecast` | +20 observation features | False |
| `enable_full_market` | +13 features, 6D action space | False |
| `enable_endogenous_dam` | Agent generates DAM schedule | False |
| `enable_gate_closure` | +1 feature (hours until closure) | True |
| `randomize_start` | Random episode start positions | True |
| `include_price_awareness` | +2 backward-looking features | True |

---

## 17. Docker & Deployment

### 17.1 Docker

```yaml
# docker-compose.yml
services:
  training:
    build: .
    command: python agent/train_v23_production_ready.py
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  evaluation:
    build: .
    command: python agent/evaluate_masked.py

  tests:
    build: .
    command: pytest tests/ -v
```

### 17.2 Scheduled Tasks

```
Data Collection:    Every day at 14:00 (after DAM publication)
Forecast Update:    Every day at 14:30
Model Evaluation:   Weekly (compare vs rule-based)
```

---

## 18. Evaluation Metrics

### 18.1 Key Performance Indicators

| Metric | Target | Description |
|--------|--------|-------------|
| Win Rate | > 65% | % of profitable days |
| Sharpe Ratio | > 1.5 | Risk-adjusted return (annualized) |
| Violation Rate | 0% | Physical constraint violations |
| DAM Compliance | > 90% | Commitment delivery rate |
| Trade Rate | 30-50% | % of active (non-idle) steps |
| Mean Daily P&L | > 0 | Average daily profit |

### 18.2 Action Distribution

| Action | Healthy Range | Warning |
|--------|--------------|---------|
| Discharge (sell) | 15-25% | > 40% → over-trading |
| Charge (buy) | 15-25% | > 40% → over-trading |
| Idle | 50-70% | > 90% → too passive |

---

# PART E: REPLICATION GUIDE

---

## 19. Country Adaptation Checklist

### 19.1 Market Parameters

- [ ] **Timezone**: `Europe/Athens` → your timezone (e.g., `Europe/Nicosia`)
- [ ] **Currency**: EUR/MWh (same across EU, but verify)
- [ ] **DAM Gate Closure**: 12:00 → verify with your exchange
- [ ] **IDA Gate Closures**: IDA1 15:00, IDA2 22:00, IDA3 10:00 → verify
- [ ] **XBID Continuous**: Verify cross-border XBID participation
- [ ] **aFRR Requirements**: Activation speed, minimum capacity, block duration
- [ ] **mFRR Price Cap**: 1000 EUR/MWh → verify your market's cap

### 19.2 TSO API

- [ ] **DAM Prices**: ENTSO-E country code (`GR` → `CY`, `BG`, etc.)
- [ ] **Balancing Prices**: Replace ADMIE with your TSO API
- [ ] **aFRR Data**: Capacity/energy prices from your TSO
- [ ] **Load Forecast**: Your TSO or grid operator API
- [ ] **RES Forecast**: Solar/wind from your TSO or weather API

### 19.3 Battery Parameters

- [ ] **Capacity (MWh)**: Adjust to your asset
- [ ] **Power (MW)**: Adjust inverter rating
- [ ] **Efficiency**: LFP ~94%, NMC ~90%, verify
- [ ] **Degradation Cost**: Recalculate for your chemistry + system cost
- [ ] **Cycle Limits**: Warranty-dependent

### 19.4 Penalty Coefficients

- [ ] **DAM Violation**: Recalibrate to your market's imbalance regime
- [ ] **aFRR Penalty**: Based on your TSO's non-compliance fee structure
- [ ] **IDA Penalty**: Based on your market's rules

### 19.5 Software Changes

- [ ] `data/entsoe_connector.py`: Change country code
- [ ] `data/admie_data_fetcher.py`: Replace with your TSO fetcher
- [ ] `PredictFuturePrices/collectors/`: Add your TSO collector
- [ ] `PredictFuturePrices/config.py`: Update paths, columns, timezone
- [ ] `gym_envs/battery_env_*.py`: Update default parameters
- [ ] `config.py`: Update battery params

---

## 20. Cyprus Market Notes

### 20.1 Market Structure

- **TSO**: TSOC (Transmission System Operator Cyprus)
- **Market Operator**: KAPE (Cyprus Energy Regulatory Authority oversees)
- **DAM**: Operates via HEnEx (coupled with Greek market since ~2023)
- **Balancing**: TSOC manages balancing mechanism
- **Interconnection**: EuroAsia Interconnector (under construction) will link CY-GR-IL

### 20.2 Key Differences from Greece

| Aspect | Greece | Cyprus |
|--------|--------|--------|
| Market size | ~55 GW peak | ~1.5 GW peak |
| Interconnections | Multiple (BG, IT, TR, AL) | Limited (planned) |
| RES penetration | ~40% | ~20% (growing rapidly) |
| Price volatility | Moderate (coupled market) | Higher (island system) |
| aFRR market | Mature | Developing |
| XBID | Active | Limited |

### 20.3 Adaptation Priority

1. **DAM**: Already coupled — ENTSO-E API works with `CY` country code
2. **Balancing**: Replace ADMIE with TSOC data feeds
3. **aFRR**: May have different capacity product structure
4. **Forecasting**: Retrain models on CY-specific data (island dynamics)

---

## 21. Column Mapping Template

When connecting a new TSO API, map their columns to the internal schema:

```python
def map_your_api_to_entsodrl(your_df: pd.DataFrame) -> pd.DataFrame:
    """
    Template: Map your TSO's column names to EntsoDRL's internal schema.
    Adjust column names to match your API's output.
    """
    return pd.DataFrame({
        # Required columns
        'price':          your_df['your_dam_price_col'],          # EUR/MWh
        'dam_commitment': your_df['your_commitment_col'],         # MW
        'solar':          your_df['your_solar_forecast_col'],     # MW
        'wind_onshore':   your_df['your_wind_forecast_col'],      # MW

        # Balancing (from your TSO)
        'Long':           your_df['your_positive_imbalance_price'],  # EUR/MWh
        'Short':          your_df['your_negative_imbalance_price'],  # EUR/MWh
        'mFRR_Up':        your_df['your_mfrr_up_price'],            # EUR/MWh
        'mFRR_Down':      your_df['your_mfrr_down_price'],          # EUR/MWh

        # aFRR (from your TSO)
        'aFRR_Up_MW':     your_df['your_afrr_activated_up'],        # MW
        'aFRR_Down_MW':   your_df['your_afrr_activated_down'],      # MW

        # Time encoding (computed)
        'hour_sin':       np.sin(2 * np.pi * your_df.index.hour / 24),
        'hour_cos':       np.cos(2 * np.pi * your_df.index.hour / 24),
    })
```

---

## 22. Appendices

### 22.1 Glossary

| Term | Definition |
|------|-----------|
| **BESS** | Battery Energy Storage System |
| **DAM** | Day-Ahead Market (hourly auction, D-1) |
| **HEnEx** | Hellenic Energy Exchange |
| **SoC** | State of Charge (battery level, 0-100%) |
| **aFRR** | Automatic Frequency Restoration Reserve |
| **mFRR** | Manual Frequency Restoration Reserve |
| **IDA** | Intra-Day Auction (IDA1/2/3) |
| **XBID** | Cross-border Intra-Day continuous market |
| **ISP** | Imbalance Settlement Period (15-min) |
| **HVR** | Highest Value Rule (max of aFRR/mFRR price) |
| **TSO** | Transmission System Operator (ADMIE in GR) |
| **PPO** | Proximal Policy Optimization (RL algorithm) |
| **DRL** | Deep Reinforcement Learning |
| **VecNormalize** | Running normalization of observations |
| **Action Masking** | Preventing invalid actions at selection time |
| **Curriculum** | Gradually increasing training difficulty |
| **Entropy** | Measure of policy randomness (exploration) |
| **KL Divergence** | Measure of policy change between updates |

### 22.2 File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `battery_env_unified.py` | 984 | Main training environment |
| `battery_env_masked.py` | ~700 | Legacy environment (v22/v23) |
| `unified_reward_calculator.py` | ~400 | Multi-market reward |
| `reward_calculator.py` | ~400 | Single-market reward |
| `observation_builder.py` | 486 | Feature vector construction |
| `mask_builder.py` | 184 | Action mask generation |
| `capacity_manager.py` | 135 | Cascade capacity limits |
| `dam_executor.py` | 319 | DAM settlement logic |
| `afrr_executor.py` | 197 | aFRR logic |
| `ida_executor.py` | 603 | IDA gate/schedule logic |
| `trading_executor.py` | 359 | mFRR/XBID/FreeBid |
| `train_v23_production_ready.py` | ~320 | Training script |
| `decision_strategy.py` | ~300 | Strategy pattern |
| `battery_controller.py` | ~200 | Production controller |
| `data_provider.py` | ~400 | Data abstraction |
| `feature_engineer.py` | ~340 | Observation construction |

---

> **Document maintained by**: EntsoDRL Development Team
> **Last updated**: March 2026
> **Repository**: https://github.com/giannosantoniou/EntsoDRL
