# Battery Trading Orchestrator - Quick Start Guide

## Installation

```bash
cd D:\WSLUbuntu\EntsoDRL
pip install -r requirements.txt
```

## Basic Usage

### Minimal Setup (DAM Only)
```python
from orchestrator import BatteryOrchestrator
from datetime import date, timedelta

# Create orchestrator
orch = BatteryOrchestrator()

# Run day-ahead bidding
tomorrow = date.today() + timedelta(days=1)
commitment = orch.run_day_ahead(tomorrow)

print(f"Buy: {commitment.total_buy_mwh:.1f} MWh")
print(f"Sell: {commitment.total_sell_mwh:.1f} MWh")
```

### Full Setup (All Markets)
```python
from orchestrator import BatteryOrchestrator
from datetime import datetime, date, timedelta

# Create with all features
orch = BatteryOrchestrator(
    max_power_mw=30.0,
    capacity_mwh=146.0,
    balancing_enabled=True,
    balancing_strategy="AGGRESSIVE_HYBRID",  # or "AI" with model_path
    afrr_enabled=True,
    mfrr_enabled=True,
    intraday_enabled=True,
)

# 1. Day-ahead (D-1 10:00)
commitment = orch.run_day_ahead(target_date)

# 2. IntraDay (D-1 14:00 onwards)
orch.update_intraday_prices(intraday_prices)  # 96 values
opportunities = orch.run_intraday(datetime.now())

# 3. Real-time (D-0 every 15 min)
orch.update_market_data({
    'mfrr_price_up': 150.0,
    'mfrr_price_down': 120.0,
})
dispatch = orch.run_realtime(datetime.now())
print(f"Dispatch: {dispatch['total_dispatch_mw']} MW")
```

## Sign Convention

| Value | Meaning | Action |
|-------|---------|--------|
| `+30 MW` | SELL | Discharge (inject to grid) |
| `-30 MW` | BUY | Charge (withdraw from grid) |
| `0 MW` | IDLE | No action |

## Key Components

```
BatteryOrchestrator
├── DAMForecaster      # Price forecasting
├── DAMBidder          # Bid generation
├── CommitmentExecutor # DAM tracking
├── IntraDayTrader     # Position adjustment
└── BalancingTrader    # aFRR + mFRR
```

## Strategy Options

| Strategy | Description | Requires Model |
|----------|-------------|----------------|
| `AI` | DRL trained model | Yes |
| `RULE_BASED` | Price thresholds | No |
| `AGGRESSIVE_HYBRID` | Time + Price | No |

## Running Tests

```bash
# Phase 1: DAM
python -m orchestrator.test_phase1

# Phase 2: Balancing
python -m orchestrator.test_phase2

# Phase 3: IntraDay
python -m orchestrator.test_phase3
```

## Output Files

```
commitments/
├── forecast_2026-02-11.csv     # Price forecast
└── commitment_2026-02-11.npz   # Power schedule
```

## Market Timeline

```
D-1 10:00  → run_day_ahead()      # Generate bids
D-1 12:00  → DAM gate closes      # Submit to HEnEx
D-1 14:00  → run_intraday()       # Start adjustments
D-0        → run_realtime()       # Execute every 15 min
```

## Emergency: Stop Trading

```python
# Set battery to IDLE
orch.update_battery_state(BatteryState(
    timestamp=datetime.now(),
    soc=current_soc,
    available_power_mw=0.0,  # Disable trading
    capacity_mwh=146.0,
))
```

## Documentation

Full architecture documentation: `orchestrator/ARCHITECTURE.md`
