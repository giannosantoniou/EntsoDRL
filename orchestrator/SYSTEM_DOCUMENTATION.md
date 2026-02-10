# EntsoDRL Battery Trading System

## Complete Technical Documentation

**Version:** 2.0
**Last Updated:** February 2026
**Author:** EntsoDRL Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Market Integration](#3-market-integration)
4. [Components Reference](#4-components-reference)
5. [Data Flows](#5-data-flows)
6. [Configuration](#6-configuration)
7. [AI/ML Models](#7-aiml-models)
8. [Performance Metrics](#8-performance-metrics)
9. [API Reference](#9-api-reference)
10. [Deployment Guide](#10-deployment-guide)

---

## 1. Executive Summary

### 1.1 What is EntsoDRL?

EntsoDRL is an **AI-powered battery energy trading system** designed for the Greek electricity market (HEnEx). It uses Deep Reinforcement Learning (DRL) to optimize battery charging/discharging decisions across multiple energy markets.

### 1.2 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Market Trading** | DAM, IntraDay, aFRR, mFRR |
| **AI-Powered Decisions** | MaskablePPO deep reinforcement learning |
| **Serbia-Based Forecasting** | Uses Serbia DAM (published earlier) for Greek price prediction |
| **Real-Time Execution** | 15-minute dispatch resolution |
| **Risk Management** | Action masking prevents physically impossible trades |

### 1.3 Performance Summary

```
Daily Revenue Breakdown:
├── DAM Revenue:      +12,144 EUR
├── IntraDay P&L:      +1,772 EUR
├── mFRR Revenue:      +2,427 EUR
└── aFRR Value:          +859 EUR
────────────────────────────────
TOTAL:                +17,202 EUR/day
ANNUAL:                +6.28 M EUR/year
```

### 1.4 Battery Specifications

| Parameter | Value |
|-----------|-------|
| Capacity | 146 MWh |
| Max Power | 30 MW |
| Duration | ~5 hours |
| Efficiency | 94% round-trip |
| Operating SoC | 5% - 95% |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL DATA SOURCES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ENTSO-E API          │  Market Feeds        │  Battery SCADA              │
│  ├─ Serbia DAM        │  ├─ mFRR prices      │  ├─ SoC                     │
│  ├─ Greek DAM         │  ├─ IntraDay prices  │  ├─ Power                   │
│  └─ Load forecasts    │  └─ aFRR prices      │  └─ Temperature             │
└───────────┬───────────┴──────────┬───────────┴──────────┬───────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BATTERY ORCHESTRATOR                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  DAM Forecaster │  │   DAM Bidder    │  │   Commitment    │             │
│  │                 │  │                 │  │    Executor     │             │
│  │ Serbia→Greek    │  │ Buy/Sell        │  │                 │             │
│  │ Hourly Regr.    │  │ Optimization    │  │ Track & Execute │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           └────────────────────┼────────────────────┘                       │
│                                │                                            │
│  ┌─────────────────┐  ┌───────┴───────┐  ┌─────────────────┐               │
│  │ IntraDay Trader │  │    Battery    │  │   Balancing     │               │
│  │                 │  │     State     │  │    Trader       │               │
│  │ Position Adjust │  │               │  │                 │               │
│  │ Auto-Execute    │  │ SoC, Power    │  │ aFRR + mFRR     │               │
│  │ P&L Tracking    │  │ Constraints   │  │ AI Model        │               │
│  └─────────────────┘  └───────────────┘  └─────────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MARKET INTERFACES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  HEnEx DAM API        │  HEnEx IntraDay      │  IPTO/ADMIE                 │
│  (Order Submission)   │  (Continuous Market) │  (Balancing Market)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Directory Structure

```
EntsoDRL/
├── orchestrator/                 # Main trading system
│   ├── __init__.py
│   ├── orchestrator.py          # Main controller
│   ├── interfaces.py            # Abstract interfaces
│   ├── dam_forecaster.py        # Price forecasting
│   ├── dam_bidder.py            # Bid generation
│   ├── commitment_executor.py   # DAM execution
│   ├── intraday_trader.py       # IntraDay trading
│   ├── balancing_trader.py      # aFRR + mFRR
│   └── test_with_model.py       # Integration tests
│
├── data/
│   ├── entsoe_connector.py      # ENTSO-E API client
│   ├── mfrr_features_clean.csv  # Historical mFRR data
│   ├── dam_prices_2021_2026.csv # Historical DAM data
│   └── gr_rs_dam_comparison.csv # GR-RS correlation data
│
├── agent/
│   ├── decision_strategy.py     # Strategy implementations
│   └── train_*.py               # Training scripts
│
├── gym_envs/
│   ├── battery_env_masked.py    # Main training environment
│   └── battery_env_threshold_v2.py  # 23-feature environment
│
└── models/
    └── final_v2_*/              # Trained models
        ├── best_model.zip
        └── vec_normalize.pkl
```

### 2.3 Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| ML Framework | PyTorch, Stable Baselines 3 |
| RL Algorithm | MaskablePPO (sb3-contrib) |
| Data Processing | pandas, numpy |
| API Client | entsoe-py |
| Environment | Gymnasium |

---

## 3. Market Integration

### 3.1 Market Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DAILY TRADING TIMELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  D-1 (Day Before Delivery)                                                  │
│  ─────────────────────────                                                  │
│  ~12:00  Serbia DAM results published                                       │
│          └─ fetch_serbia_dam_d1() retrieves prices                         │
│                                                                             │
│  12:00-  DAM Forecasting                                                    │
│  12:30   └─ Generate Greek price forecast using Serbia + hourly regression │
│                                                                             │
│  12:30-  DAM Bidding                                                        │
│  12:55   └─ Generate optimal buy/sell schedule                             │
│          └─ Save commitment file                                           │
│                                                                             │
│  13:00   ═══ DAM GATE CLOSURE ═══                                          │
│                                                                             │
│  14:00   IntraDay Session Opens                                             │
│          └─ start_intraday_session()                                       │
│          └─ Monitor prices, execute adjustments                            │
│                                                                             │
│  D-0 (Delivery Day)                                                         │
│  ─────────────────                                                          │
│  00:00-  Real-Time Execution (every 15 minutes)                            │
│  23:45   ├─ Execute DAM commitment (mandatory)                             │
│          ├─ IntraDay adjustments (until H-1)                               │
│          ├─ aFRR capacity reservation                                       │
│          └─ mFRR AI trading decisions                                       │
│                                                                             │
│  H-1     IntraDay gate closes for hour H                                    │
│                                                                             │
│  23:45   Last interval executed                                             │
│          └─ close_intraday_session()                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 DAM (Day-Ahead Market)

#### Purpose
Commit to buy/sell energy for next day at fixed prices.

#### Process
1. **Forecast**: Use Serbia DAM prices (available ~12:00) to predict Greek prices
2. **Optimize**: Find hours to buy (low price) and sell (high price)
3. **Submit**: Send bids before 13:00 gate closure
4. **Execute**: Mandatory execution next day

#### Forecasting Model

```python
# Hourly regression: Greek = slope[h] * Serbia + intercept[h]
# Based on 6-month correlation analysis

HOURLY_GR_RS_PARAMS = {
    0:  {'slope': 0.8649, 'intercept':  21.93},  # Night - good correlation
    ...
    12: {'slope': 0.5603, 'intercept':   5.24},  # Solar dip - GR much cheaper
    13: {'slope': 0.5545, 'intercept':   2.45},  # Solar peak
    ...
    20: {'slope': 1.3837, 'intercept': -25.64},  # Evening peak - GR spikes
    21: {'slope': 1.2570, 'intercept':   5.70},  # Evening peak
    ...
}
```

#### Key Insights
- **Solar hours (12-14h)**: Greek prices 20-60 EUR lower than Serbia
- **Evening peak (20-21h)**: Greek prices spike higher than Serbia
- **Overall correlation**: 0.65 (R² = 0.42)

### 3.3 IntraDay Market

#### Purpose
Adjust DAM positions based on updated information.

#### Process
1. **Session Start**: D-1 14:00 (after DAM results)
2. **Monitor**: Compare IntraDay prices with DAM prices
3. **Execute**: Auto-trade when spread > 5 EUR
4. **Gate Closure**: 1 hour before delivery

#### Execution Logic

```python
# Find opportunities
opportunities = trader.get_adjustment_opportunities(
    intraday_prices=current_prices,
    battery_state=battery_state,
)

# Auto-execute profitable ones
for opp in opportunities:
    if opp['potential_profit_eur'] > min_profit_eur:
        if is_gate_open(opp['delivery_time'], current_time):
            execute_order(opp)
```

#### Actions
| Action | Trigger | Result |
|--------|---------|--------|
| `sell_more` | ID price > DAM price, we're selling | Increase sell position |
| `reduce_sell` | ID price < DAM price, we're selling | Buy back some |
| `buy_more` | ID price < DAM price, we're buying | Increase buy position |
| `reduce_buy` | ID price > DAM price, we're buying | Sell some back |

### 3.4 aFRR (Automatic Frequency Restoration Reserve)

#### Purpose
Provide capacity for automatic frequency control.

#### Process
1. **Reserve**: Allocate 30% of available capacity
2. **Bid**: Submit capacity price (simplified - not fully implemented)
3. **Activate**: Respond to TSO frequency signals (not implemented)

#### Current Implementation
- Capacity reservation only
- No actual TSO signal handling
- Estimated value based on capacity prices

### 3.5 mFRR (Manual Frequency Restoration Reserve)

#### Purpose
Energy trading in real-time balancing market.

#### Process
1. **Observe**: Current mFRR prices, battery state, time features
2. **Predict**: AI model selects optimal action
3. **Execute**: Charge/discharge/idle
4. **Learn**: Model trained on historical mFRR data

#### AI Model
- **Algorithm**: MaskablePPO (Proximal Policy Optimization with action masking)
- **Observation**: 23 features (see Section 7)
- **Actions**: 21 discrete levels (-30 MW to +30 MW)
- **Training**: 5M timesteps on historical data

---

## 4. Components Reference

### 4.1 BatteryOrchestrator

Main controller that coordinates all subsystems.

```python
from orchestrator import BatteryOrchestrator

orchestrator = BatteryOrchestrator(
    max_power_mw=30.0,
    capacity_mwh=146.0,
    efficiency=0.94,

    # Balancing (mFRR + aFRR)
    balancing_enabled=True,
    balancing_model_path="models/final_v2/best_model.zip",
    balancing_strategy="AI",  # or "RULE_BASED", "AGGRESSIVE_HYBRID"

    # IntraDay
    intraday_enabled=True,
    intraday_min_spread=5.0,
)
```

#### Key Methods

| Method | When to Call | Returns |
|--------|--------------|---------|
| `run_day_ahead(date)` | D-1 before 13:00 | DAMCommitment |
| `start_intraday_session(date)` | D-1 14:00 | Session info |
| `run_intraday(time)` | D-1 14:00 to D-0 | Executed orders |
| `run_realtime(time)` | D-0 every 15 min | Dispatch command |
| `execute_interval(time, power)` | After dispatch | Execution record |
| `close_intraday_session()` | D-0 end of day | Session P&L |

### 4.2 DAMForecaster

Generates price forecasts using Serbia DAM data.

```python
from orchestrator.dam_forecaster import DAMForecaster

forecaster = DAMForecaster(
    model_path="path/to/model",  # Optional EntsoE3 model
    spread_percent=0.02,         # 2% bid-ask spread
)

forecast = forecaster.forecast(target_date)
# Returns: PriceForecast with 96 buy/sell prices
```

#### Forecast Process
1. Fetch Serbia DAM prices via ENTSO-E API
2. Apply hourly regression model
3. Add small noise (±3%)
4. Return buy/sell prices with spread

### 4.3 DAMBidder

Generates optimal buy/sell schedule.

```python
from orchestrator.dam_bidder import DAMBidder

bidder = DAMBidder(
    max_power_mw=30.0,
    capacity_mwh=146.0,
    efficiency=0.94,
    max_daily_cycles=2.0,
    buy_percentile=25,   # Buy below 25th percentile
    sell_percentile=75,  # Sell above 75th percentile
)

commitment = bidder.generate_bids(forecast)
# Returns: DAMCommitment with 96 power values
```

### 4.4 IntraDayTrader

Manages IntraDay position adjustments.

```python
from orchestrator.intraday_trader import IntraDayTrader

trader = IntraDayTrader(
    max_power_mw=30.0,
    capacity_mwh=146.0,
    min_spread_eur=5.0,       # Minimum spread to trade
    max_position_change=0.5,  # Max 50% of DAM position
)

# Start session
trader.start_session(delivery_date)

# Find and execute opportunities
orders = trader.execute_opportunities(
    intraday_prices=prices,
    battery_state=state,
    current_time=now,
    min_profit_eur=10.0,
)

# Get P&L
summary = trader.close_session()
```

### 4.5 BalancingTrader

Handles aFRR and mFRR trading with AI model.

```python
from orchestrator.balancing_trader import BalancingTrader

trader = BalancingTrader(
    model_path="models/final_v2/best_model.zip",
    strategy_type="AI",  # Uses trained DRL model
    afrr_enabled=True,
    mfrr_enabled=True,
    afrr_reserve_fraction=0.3,
)

decision = trader.decide_full(
    battery_state=state,
    available_capacity_mw=20.0,
    market_data={'mfrr_price_up': 150, 'mfrr_price_down': 80},
)

# decision.mfrr_action_mw = power to trade
# decision.afrr_up_capacity_mw = reserved up capacity
# decision.afrr_down_capacity_mw = reserved down capacity
```

### 4.6 EntsoeConnector

ENTSO-E Transparency Platform API client.

```python
from data.entsoe_connector import EntsoeConnector

connector = EntsoeConnector()  # Uses ENTSOE_API_KEY from .env

# Fetch Serbia DAM for tomorrow
serbia_prices = connector.fetch_serbia_dam_d1(tomorrow)

# Fetch any neighbor's DAM
bulgaria_prices = connector.fetch_neighbor_dam_prices('BG', start, end)

# Available neighbors: RS (Serbia), BG (Bulgaria), MK, AL, TR, IT
```

---

## 5. Data Flows

### 5.1 Day-Ahead Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  ENTSO-E API │────▶│ DAMForecaster│────▶│  DAMBidder   │
│              │     │              │     │              │
│ Serbia DAM   │     │ Hourly Regr. │     │ Optimize     │
│ prices D+1   │     │ → Greek fcst │     │ buy/sell     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Storage    │◀────│ Commitment   │
                     │              │     │   Executor   │
                     │ .npz file    │     │              │
                     └──────────────┘     └──────────────┘
```

### 5.2 Real-Time Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Market Feed  │────▶│  Balancing   │────▶│  Dispatch    │
│              │     │   Trader     │     │  Command     │
│ mFRR prices  │     │              │     │              │
│ aFRR prices  │     │ AI Model     │     │ Total MW     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐            │
│   Battery    │────▶│ Orchestrator │◀───────────┘
│    SCADA     │     │              │
│              │     │ Combine:     │
│ SoC, Power   │     │ DAM + mFRR   │
└──────────────┘     └──────────────┘
```

### 5.3 Observation Building (for AI)

```
Battery State                Market Data               Time
────────────────            ────────────────         ────────
SoC (0-1)                   mFRR price up            Hour (0-23)
Max discharge MW            mFRR price down          │
Max charge MW               Spread                   ▼
      │                           │                Hour sin/cos
      │                           │
      ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    OBSERVATION (23 features)                │
├─────────────────────────────────────────────────────────────┤
│ [soc, max_discharge, max_charge,                            │
│  price_norm, price_very_low, price_very_high,              │
│  negative_price_flag, price_magnitude, price_percentile,   │
│  spread_norm, spread_high,                                  │
│  soc_low, soc_high,                                         │
│  hour_sin, hour_cos,                                        │
│  price_lag_1h, price_lag_4h, price_lag_24h,                │
│  price_rolling_mean, price_momentum, price_volatility,      │
│  spread_lag_1h, spread_lag_4h]                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  VecNormalize   │
                     │                 │
                     │ Standardize to  │
                     │ zero mean,      │
                     │ unit variance   │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │  MaskablePPO    │
                     │                 │
                     │ Select action   │
                     │ (0-20)          │
                     └────────┬────────┘
                              │
                              ▼
                     Action → Power MW
                     (-30 to +30 MW)
```

---

## 6. Configuration

### 6.1 Environment Variables

Create `.env` file:

```bash
# ENTSO-E API Key (required for Serbia DAM fetch)
ENTSOE_API_KEY=your-api-key-here
```

### 6.2 Battery Parameters

```python
# In orchestrator initialization
BATTERY_CONFIG = {
    'capacity_mwh': 146.0,       # Total energy capacity
    'max_power_mw': 30.0,        # Max charge/discharge power
    'efficiency': 0.94,          # Round-trip efficiency
    'min_soc': 0.05,             # Minimum SoC (5%)
    'max_soc': 0.95,             # Maximum SoC (95%)
}
```

### 6.3 Trading Parameters

```python
# DAM Bidding
DAM_CONFIG = {
    'max_daily_cycles': 2.0,     # Max full cycles per day
    'buy_percentile': 25,        # Buy below this percentile
    'sell_percentile': 75,       # Sell above this percentile
    'spread_percent': 0.02,      # Bid-ask spread
}

# IntraDay Trading
INTRADAY_CONFIG = {
    'min_spread_eur': 5.0,       # Min spread to trade
    'max_position_change': 0.5,  # Max 50% of DAM
    'gate_closure_hours': 1.0,   # Close 1h before delivery
    'execution_probability': 0.85,  # Simulated fill rate
}

# Balancing (mFRR)
BALANCING_CONFIG = {
    'strategy': 'AI',            # AI, RULE_BASED, AGGRESSIVE_HYBRID
    'afrr_reserve_fraction': 0.3,  # 30% for aFRR
    'n_actions': 21,             # Discrete action levels
}
```

### 6.4 Price Thresholds (matching training)

```python
# Must match training environment for AI to work correctly
PRICE_THRESHOLDS = {
    'price_low': 40.0,           # EUR - below = "very low"
    'price_high': 140.0,         # EUR - above = "very high"
    'spread_threshold': 50.0,    # EUR - above = "high spread"
    'soc_low': 0.27,             # Below = "low SoC"
    'soc_high': 0.79,            # Above = "high SoC"
}
```

---

## 7. AI/ML Models

### 7.1 Model Architecture

| Component | Specification |
|-----------|---------------|
| Algorithm | MaskablePPO |
| Policy Network | MLP [256, 256] |
| Value Network | MLP [256, 256] |
| Activation | ReLU |
| Observation Space | Box(23,) |
| Action Space | Discrete(21) |

### 7.2 Observation Features (23 total)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 0 | soc_norm | State of charge | [0, 1] |
| 1 | max_discharge_norm | Max discharge / max_power | [0, 1] |
| 2 | max_charge_norm | Max charge / max_power | [0, 1] |
| 3 | price_normalized | (price - mean) / std | [-5, 5] |
| 4 | price_very_low | price < 40 EUR | {0, 1} |
| 5 | price_very_high | price > 140 EUR | {0, 1} |
| 6 | negative_price_flag | price < 0 | {0, 1} |
| 7 | price_magnitude | How extreme the price is | [-5, 5] |
| 8 | price_percentile | Percentile vs 7-day history | [0, 1] |
| 9 | spread_normalized | (spread - mean) / std | [-3, 3] |
| 10 | spread_high | spread > 50 EUR | {0, 1} |
| 11 | soc_low | SoC < 27% | {0, 1} |
| 12 | soc_high | SoC > 79% | {0, 1} |
| 13 | hour_sin | sin(2π × hour / 24) | [-1, 1] |
| 14 | hour_cos | cos(2π × hour / 24) | [-1, 1] |
| 15 | price_lag_1h | Price 1 hour ago (norm) | [-5, 5] |
| 16 | price_lag_4h | Price 4 hours ago (norm) | [-5, 5] |
| 17 | price_lag_24h | Price 24 hours ago (norm) | [-5, 5] |
| 18 | price_rolling_mean | 24h rolling mean (norm) | [-5, 5] |
| 19 | price_momentum | Price change 1h→4h | [-3, 3] |
| 20 | price_volatility | 24h rolling std | [0, 3] |
| 21 | spread_lag_1h | Spread 1 hour ago | [-3, 3] |
| 22 | spread_lag_4h | Spread 4 hours ago | [-3, 3] |

### 7.3 Action Space

```
Action Index    Power (MW)      Meaning
──────────────────────────────────────────
0               -30.0           Full discharge
1               -27.0
...             ...
10               0.0            Idle
...             ...
19              +27.0
20              +30.0           Full charge
```

### 7.4 Action Masking

Actions are masked based on physical constraints:

```python
# Cannot discharge more than available energy
if soc < min_soc + required_energy:
    mask[discharge_actions] = False

# Cannot charge more than available headroom
if soc > max_soc - required_energy:
    mask[charge_actions] = False

# Idle is always valid
mask[10] = True
```

### 7.5 Training Details

| Parameter | Value |
|-----------|-------|
| Total Timesteps | 5,000,000 |
| Learning Rate | 3e-4 |
| Batch Size | 256 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| VecNormalize | obs=True, reward=True |

### 7.6 VecNormalize

The model uses VecNormalize for observation standardization:

```python
# Loading for inference
vec_normalize = pickle.load(open('vec_normalize.pkl', 'rb'))
vec_normalize.training = False
vec_normalize.norm_reward = False

# Apply to observations
obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))
```

---

## 8. Performance Metrics

### 8.1 Strategy Comparison

| Strategy | Daily Revenue | Annual |
|----------|---------------|--------|
| **AI (MaskablePPO)** | +14,398 EUR | +5.26 M EUR |
| RULE_BASED | +2,100 EUR | +0.77 M EUR |
| AGGRESSIVE_HYBRID | +1,687 EUR | +0.62 M EUR |

### 8.2 Revenue by Market

| Market | Revenue | % of Total |
|--------|---------|------------|
| DAM | +12,144 EUR | 70.6% |
| mFRR | +2,427 EUR | 14.1% |
| IntraDay | +1,772 EUR | 10.3% |
| aFRR | +859 EUR | 5.0% |
| **Total** | **+17,202 EUR** | 100% |

### 8.3 Key Performance Indicators

| KPI | Value |
|-----|-------|
| Daily Profit | +17,202 EUR |
| Annual Projection | +6.28 M EUR |
| Profit per MWh capacity | 118 EUR/day |
| Profit per MW power | 573 EUR/day |
| Daily Cycles | 1.3 |
| Battery Utilization | 79% |

### 8.4 GR-RS Correlation Analysis

| Hour | Correlation | R² | GR-RS Diff |
|------|-------------|-----|------------|
| 02:00-04:00 | 0.71-0.77 | 0.50-0.60 | +5 to +7 EUR |
| 12:00-14:00 | 0.56-0.63 | 0.31-0.40 | -28 to -39 EUR |
| 20:00-21:00 | 0.70-0.78 | 0.49-0.60 | +33 to +38 EUR |

---

## 9. API Reference

### 9.1 Orchestrator API

```python
class BatteryOrchestrator:
    def __init__(self, max_power_mw, capacity_mwh, efficiency, ...)

    # Day-Ahead
    def run_day_ahead(self, target_date: date) -> DAMCommitment

    # IntraDay
    def start_intraday_session(self, delivery_date: date) -> dict
    def update_intraday_prices(self, prices: np.ndarray) -> None
    def run_intraday(self, current_time: datetime, auto_execute: bool) -> dict
    def close_intraday_session(self) -> dict

    # Real-Time
    def update_battery_state(self, state: BatteryState) -> None
    def update_market_data(self, market_data: dict) -> None
    def run_realtime(self, timestamp: datetime) -> dict
    def execute_interval(self, timestamp: datetime, actual_power: float) -> None

    # Queries
    def get_net_position(self, delivery_time: datetime) -> float
    def get_daily_summary(self, target_date: date) -> dict
```

### 9.2 Data Classes

```python
@dataclass
class BatteryState:
    timestamp: datetime
    soc: float                    # 0-1
    available_power_mw: float
    capacity_mwh: float

@dataclass
class DAMCommitment:
    target_date: date
    timestamps: List[datetime]    # 96 values
    power_mw: np.ndarray          # 96 values

@dataclass
class PriceForecast:
    target_date: date
    timestamps: List[datetime]
    buy_prices: np.ndarray
    sell_prices: np.ndarray

@dataclass
class IntraDayOrder:
    timestamp: datetime
    delivery_start: datetime
    power_mw: float               # +sell, -buy
    price_limit: float
    status: str                   # pending, filled, cancelled
    fill_price: float
    pnl_eur: float
```

### 9.3 ENTSO-E Connector API

```python
class EntsoeConnector:
    def __init__(self, api_key: str = None)

    # Greek data
    def fetch_day_ahead_prices(self, start, end) -> pd.DataFrame
    def fetch_load_forecast(self, start, end) -> pd.DataFrame
    def fetch_imbalance_prices(self, start, end) -> pd.DataFrame

    # Neighbor data
    def fetch_neighbor_dam_prices(self, country_code, start, end) -> pd.DataFrame
    def fetch_serbia_dam_d1(self, target_date) -> pd.DataFrame

    # Country codes: RS, BG, MK, AL, TR, IT
```

---

## 10. Deployment Guide

### 10.1 Installation

```bash
# Clone repository
git clone https://github.com/giannosantoniou/EntsoDRL.git
cd EntsoDRL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 10.2 Configuration

```bash
# Create .env file
echo "ENTSOE_API_KEY=your-key-here" > .env

# Verify API key works
python -c "from data.entsoe_connector import EntsoeConnector; EntsoeConnector()"
```

### 10.3 Running Tests

```bash
# Full integration test
python -m orchestrator.test_with_model

# Expected output:
# - DAM forecast generation
# - IntraDay session with executed orders
# - Real-time simulation for 96 intervals
# - Revenue breakdown
# - Strategy comparison
```

### 10.4 Typical Usage

```python
from orchestrator import BatteryOrchestrator
from datetime import date, datetime, timedelta

# Initialize
orchestrator = BatteryOrchestrator(
    balancing_enabled=True,
    balancing_model_path="models/final_v2_20260210_130905/best_model.zip",
    balancing_strategy="AI",
    intraday_enabled=True,
)

# Day-Ahead (run before 13:00)
tomorrow = date.today() + timedelta(days=1)
commitment = orchestrator.run_day_ahead(tomorrow)

# Start IntraDay session (run at 14:00)
orchestrator.start_intraday_session(tomorrow)

# IntraDay loop (run periodically until delivery)
while in_intraday_window:
    orchestrator.update_intraday_prices(get_current_prices())
    result = orchestrator.run_intraday(datetime.now())

# Real-time loop (run every 15 min on delivery day)
for interval in range(96):
    ts = get_interval_timestamp(interval)

    # Update state
    orchestrator.update_battery_state(get_scada_data())
    orchestrator.update_market_data(get_market_data())

    # Get dispatch
    dispatch = orchestrator.run_realtime(ts)

    # Send to battery
    send_dispatch_command(dispatch['total_dispatch_mw'])

    # Confirm execution
    actual = get_actual_power()
    orchestrator.execute_interval(ts, actual)

# End of day
summary = orchestrator.close_intraday_session()
print(f"Daily P&L: {summary['total_pnl_eur']}")
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **DAM** | Day-Ahead Market - Energy traded day before delivery |
| **IntraDay** | Continuous market for position adjustments |
| **aFRR** | Automatic Frequency Restoration Reserve (capacity) |
| **mFRR** | Manual Frequency Restoration Reserve (energy) |
| **SoC** | State of Charge (0-100%) |
| **Gate Closure** | Deadline for submitting orders |
| **HEnEx** | Hellenic Energy Exchange |
| **TSO** | Transmission System Operator (IPTO/ADMIE) |
| **MaskablePPO** | RL algorithm with invalid action masking |
| **VecNormalize** | Observation normalization wrapper |

---

## Appendix B: File Checksums

```
orchestrator/orchestrator.py      - Main controller
orchestrator/dam_forecaster.py    - Serbia-based forecasting
orchestrator/dam_bidder.py        - Bid optimization
orchestrator/intraday_trader.py   - IntraDay execution
orchestrator/balancing_trader.py  - AI trading
data/entsoe_connector.py          - ENTSO-E API
models/final_v2_*/best_model.zip  - Trained AI model
models/final_v2_*/vec_normalize.pkl - Normalization stats
```

---

## Appendix C: Contact & Support

For questions or issues:
- GitHub: https://github.com/giannosantoniou/EntsoDRL
- Issues: https://github.com/giannosantoniou/EntsoDRL/issues

---

*End of Documentation*
