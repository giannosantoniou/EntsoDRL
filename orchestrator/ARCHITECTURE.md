# Battery Trading Orchestrator - Technical Architecture

**Version**: 1.0.0
**Last Updated**: 2026-02-10
**Authors**: EntsoDRL Team
**Classification**: Confidential - Internal Use Only

---

## Executive Summary

The Battery Trading Orchestrator is a multi-market trading system designed for utility-scale battery energy storage systems (BESS) participating in the Greek electricity market (HEnEx). The system coordinates participation across three market segments:

1. **Day-Ahead Market (DAM)**: Price forecasting, bidding, and commitment management
2. **IntraDay Market**: Position adjustment and arbitrage opportunities
3. **Balancing Markets**: aFRR capacity provision and mFRR energy trading

The architecture follows SOLID principles with clear interface contracts, enabling component replacement, testing, and evolution without system-wide changes.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Principles](#2-architecture-principles)
3. [Component Details](#3-component-details)
4. [Data Structures](#4-data-structures)
5. [Market Timeline & Workflows](#5-market-timeline--workflows)
6. [Configuration Reference](#6-configuration-reference)
7. [Integration Points](#7-integration-points)
8. [Error Handling](#8-error-handling)
9. [Testing Strategy](#9-testing-strategy)
10. [Production Deployment](#10-production-deployment)
11. [Performance Considerations](#11-performance-considerations)
12. [Security Considerations](#12-security-considerations)

---

## 1. System Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SYSTEMS                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  HEnEx   │  │  IPTO    │  │  SCADA   │  │ EntsoE3  │  │  ADMIE   │  │
│  │  (DAM)   │  │(Balancing│  │ (Battery)│  │(Forecast)│  │  (TSO)   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │             │             │             │         │
└───────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────┘
        │             │             │             │             │
        ▼             ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BATTERY TRADING ORCHESTRATOR                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BatteryOrchestrator                          │   │
│  │                    (Main Controller)                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                    │                    │                   │
│           ▼                    ▼                    ▼                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   DAM Layer     │  │  IntraDay Layer │  │ Balancing Layer │        │
│  │   (Phase 1)     │  │   (Phase 3)     │  │   (Phase 2)     │        │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤        │
│  │ DAMForecaster   │  │ IntraDayTrader  │  │ BalancingTrader │        │
│  │ DAMBidder       │  │                 │  │  - aFRR         │        │
│  │ CommitmentExec  │  │                 │  │  - mFRR (DRL)   │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           BATTERY SYSTEM                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  146 MWh Capacity  │  30 MW Power  │  94% Efficiency  │  2 Cycles │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| `BatteryOrchestrator` | Main controller, coordinates all subsystems | Market data, battery state | Dispatch commands |
| `DAMForecaster` | Price prediction for D+1 | Historical data, weather | 96 buy/sell prices |
| `DAMBidder` | Optimal bid generation | Forecast, battery state | DAM commitment |
| `CommitmentExecutor` | Track and enforce DAM commitments | Commitment, timestamp | Power setpoint |
| `IntraDayTrader` | Position adjustment optimization | DAM position, ID prices | Adjustment orders |
| `BalancingTrader` | Real-time balancing decisions | Market signals, capacity | Power action |

### 1.3 File Structure

```
orchestrator/
├── __init__.py              # Package exports
├── interfaces.py            # Abstract base classes and data structures
├── dam_forecaster.py        # DAM price forecasting (EntsoE3 wrapper)
├── dam_bidder.py            # DAM bidding strategy
├── commitment_executor.py   # DAM commitment tracking
├── intraday_trader.py       # IntraDay position management
├── balancing_trader.py      # aFRR + mFRR trading
├── orchestrator.py          # Main controller
├── test_phase1.py           # DAM tests
├── test_phase2.py           # Balancing tests
├── test_phase3.py           # IntraDay tests
└── ARCHITECTURE.md          # This document
```

---

## 2. Architecture Principles

### 2.1 SOLID Principles Applied

#### Single Responsibility Principle (SRP)
Each component has one clear responsibility:
- `DAMForecaster`: Only forecasts prices
- `DAMBidder`: Only generates bids
- `CommitmentExecutor`: Only tracks commitments

#### Open/Closed Principle (OCP)
New strategies can be added without modifying existing code:
```python
# Adding a new bidding strategy
class MLBasedBidder(IDAMBidder):
    def generate_bids(self, forecast, battery_state, constraints):
        # New ML-based implementation
        pass
```

#### Liskov Substitution Principle (LSP)
All implementations are interchangeable through interfaces:
```python
# Any IDAMForecaster implementation works
forecaster: IDAMForecaster = DAMForecaster()  # or MLForecaster() or PerfectForecaster()
forecast = forecaster.forecast(target_date)
```

#### Interface Segregation Principle (ISP)
Interfaces are focused and minimal:
- `IDAMForecaster`: Only `forecast()` and `save_forecast()`
- `IDAMBidder`: Only `generate_bids()`

#### Dependency Inversion Principle (DIP)
High-level modules depend on abstractions:
```python
class BatteryOrchestrator:
    # Depends on interfaces, not concrete implementations
    forecaster: IDAMForecaster
    bidder: IDAMBidder
    executor: ICommitmentExecutor
```

### 2.2 Design Patterns Used

| Pattern | Usage | Location |
|---------|-------|----------|
| **Strategy** | Interchangeable trading strategies | `BalancingTrader` with AI/Rule/Hybrid |
| **Factory** | Component creation | `create_balancing_trader()`, `create_intraday_trader()` |
| **Repository** | Commitment storage | `CommitmentExecutor._save/load_commitment()` |
| **Observer** | Market data updates | `update_market_data()`, `update_intraday_prices()` |
| **Template Method** | Trading workflow | `run_day_ahead()`, `run_realtime()` |

### 2.3 Separation of Concerns

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION                             │
│  (Logging, Monitoring, Dashboard)                               │
├─────────────────────────────────────────────────────────────────┤
│                        ORCHESTRATION                            │
│  BatteryOrchestrator - Coordinates workflows                    │
├─────────────────────────────────────────────────────────────────┤
│                        BUSINESS LOGIC                           │
│  DAMBidder, IntraDayTrader, BalancingTrader                    │
├─────────────────────────────────────────────────────────────────┤
│                        DATA ACCESS                              │
│  DAMForecaster (EntsoE3), CommitmentExecutor (Storage)         │
├─────────────────────────────────────────────────────────────────┤
│                        EXTERNAL INTERFACES                      │
│  SCADA, HEnEx API, IPTO API, Market Data Feeds                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 DAMForecaster (`dam_forecaster.py`)

#### Purpose
Generates 24-hour ahead DAM price forecasts at 15-minute resolution.

#### Interface
```python
class IDAMForecaster(ABC):
    @abstractmethod
    def forecast(self, target_date: date) -> PriceForecast:
        """Generate price forecast for target date."""
        pass

    @abstractmethod
    def save_forecast(self, forecast: PriceForecast, path: str) -> None:
        """Save forecast to CSV file."""
        pass
```

#### Implementation Details

**Initialization:**
```python
DAMForecaster(
    model_path: str = None,      # Path to EntsoE3 TFT model
    spread_percent: float = 0.02  # 2% bid-ask spread
)
```

**Forecast Generation:**
1. Load EntsoE3 TFT model (lazy loading)
2. Generate 24 hourly price predictions
3. Expand to 96 values (15-min resolution)
4. Apply bid-ask spread:
   - `buy_price = mid_price * (1 + spread/2)`
   - `sell_price = mid_price * (1 - spread/2)`

**Fallback Mechanism:**
If EntsoE3 model unavailable, uses pattern-based forecast:
```python
def _generate_fallback_forecast(self, target_date):
    # Typical Greek market daily pattern
    hourly_pattern = [
        85,   # 00:00 - night
        80,   # 01:00
        75,   # 02:00 - low
        72,   # 03:00 - lowest
        # ... (full 24-hour pattern)
        155,  # 18:00 - evening peak
        160,  # 19:00 - peak
        # ...
    ]

    # Weekend adjustment
    if target_date.weekday() >= 5:
        hourly_pattern *= 0.85

    # Random noise (±10%)
    noise = np.random.uniform(0.9, 1.1, 24)
    return hourly_pattern * noise
```

#### Output Format
```python
PriceForecast(
    target_date=date(2026, 2, 11),
    timestamps=[datetime(2026, 2, 11, 0, 0), ...],  # 96 timestamps
    buy_prices=np.array([...]),   # 96 values (EUR/MWh)
    sell_prices=np.array([...]),  # 96 values (EUR/MWh)
)
```

---

### 3.2 DAMBidder (`dam_bidder.py`)

#### Purpose
Generates optimal DAM bids based on price forecast and battery constraints.

#### Interface
```python
class IDAMBidder(ABC):
    @abstractmethod
    def generate_bids(
        self,
        forecast: PriceForecast,
        battery_state: BatteryState,
        constraints: dict = None
    ) -> DAMCommitment:
        """Generate DAM bids based on forecast and battery state."""
        pass
```

#### Implementation Details

**Initialization:**
```python
DAMBidder(
    max_power_mw: float = 30.0,
    capacity_mwh: float = 146.0,
    efficiency: float = 0.94,
    min_soc: float = 0.05,
    max_soc: float = 0.95,
    max_daily_cycles: float = 2.0,
    buy_threshold_percentile: float = 25,   # Buy when price < 25th percentile
    sell_threshold_percentile: float = 75,  # Sell when price > 75th percentile
    min_spread_eur: float = 30.0            # Minimum spread to trade
)
```

**Bidding Algorithm:**

```
Step 1: Calculate Price Thresholds
├── buy_threshold = percentile(prices, 25)
├── sell_threshold = percentile(prices, 75)
└── IF spread < min_spread THEN tighten to 10th/90th percentile

Step 2: Calculate Profit Potential
FOR each interval i:
├── IF price[i] <= buy_threshold:
│   └── profit_potential[i] = -(expected_sell_price - buy_price[i])
├── ELIF price[i] >= sell_threshold:
│   └── profit_potential[i] = sell_price[i] - expected_buy_price

Step 3: Sort and Select
├── buy_intervals = sorted by profit potential (most negative first)
├── sell_intervals = sorted by profit potential (most positive first)

Step 4: Schedule with Constraints
FOR each buy_interval (sorted):
├── Check cycle limit
├── Check SoC headroom
├── IF valid: schedule charge at max power
└── Update SoC and cycle count

FOR each sell_interval (sorted):
├── Check cycle limit
├── Check SoC availability
├── IF valid: schedule discharge at max power
└── Update SoC and cycle count

Step 5: Validate
└── Run simulation to verify SoC stays within [5%, 95%]
```

**Key Constraints:**
- Maximum 2 cycles per day (to limit degradation)
- SoC must stay within 5%-95% at all times
- Power limited to ±30 MW

#### Output Format
```python
DAMCommitment(
    target_date=date(2026, 2, 11),
    timestamps=[...],  # 96 timestamps
    power_mw=np.array([  # 96 values
        -30.0,  # Negative = BUY (charge)
        0.0,    # Zero = IDLE
        30.0,   # Positive = SELL (discharge)
        ...
    ])
)
```

---

### 3.3 CommitmentExecutor (`commitment_executor.py`)

#### Purpose
Tracks DAM commitments and calculates remaining capacity for other markets.

#### Interface
```python
class ICommitmentExecutor(ABC):
    @abstractmethod
    def load_commitment(self, commitment: DAMCommitment) -> None:
        """Load a commitment schedule."""
        pass

    @abstractmethod
    def get_current_commitment(self, timestamp: datetime) -> float:
        """Get committed power for current timestamp."""
        pass

    @abstractmethod
    def get_remaining_capacity(self, timestamp: datetime) -> float:
        """Get capacity remaining after DAM commitment."""
        pass
```

#### Implementation Details

**Key Methods:**

```python
def get_current_commitment(self, timestamp: datetime) -> float:
    """
    Returns committed power in MW.
    Positive = SELL/discharge, Negative = BUY/charge
    """
    idx = self._get_interval_index(timestamp)
    return self._commitment.power_mw[idx]

def get_remaining_capacity(self, timestamp: datetime) -> float:
    """
    Returns MW available for balancing markets.
    Example: If DAM commitment is 20 MW discharge,
             remaining = 30 - 20 = 10 MW
    """
    committed = abs(self.get_current_commitment(timestamp))
    return self.max_power_mw - committed

def can_participate_in_balancing(
    self,
    timestamp: datetime,
    battery_state: BatteryState
) -> Dict:
    """
    Returns available up/down regulation capacity
    considering both power limits and SoC constraints.
    """
    commitment = self.get_current_commitment(timestamp)
    remaining_power = self.max_power_mw - abs(commitment)

    # Energy constraints
    discharge_headroom = (soc - min_soc) * capacity / 0.25
    charge_headroom = (max_soc - soc) * capacity / 0.25

    return {
        'up_capacity_mw': min(remaining_power, discharge_headroom),
        'down_capacity_mw': min(remaining_power, charge_headroom),
    }
```

**Execution Recording:**
```python
def record_execution(self, timestamp: datetime, actual_power_mw: float):
    """Record actual execution for settlement and audit."""
    committed = self.get_current_commitment(timestamp)
    deviation = actual_power_mw - committed

    # Log deviation for settlement
    status = ExecutionStatus(
        timestamp=timestamp,
        committed_mw=committed,
        executed_mw=actual_power_mw,
        deviation_mw=deviation,
    )

    # Warning if significant deviation
    if abs(deviation) > tolerance:
        log.warning(f"Deviation: {deviation} MW at {timestamp}")
```

---

### 3.4 IntraDayTrader (`intraday_trader.py`)

#### Purpose
Identifies and executes position adjustments between DAM and real-time.

#### Key Concepts

**When IntraDay Trading Occurs:**
```
D-1 12:00  │  DAM gate closes
D-1 14:00  │  DAM results published ← IntraDay window OPENS
   ...     │  Continuous trading
D-0 H-1    │  IntraDay closes (1 hour before delivery)
D-0        │  Real-time execution
```

**Why Adjust Positions:**
1. **Forecast Error**: DAM bid was based on D-2 forecast, now we have better info
2. **Arbitrage**: IntraDay price different from DAM clearing price
3. **Position Optimization**: Rebalance ahead of real-time

#### Implementation Details

**Position Tracking:**
```python
@dataclass
class IntraDayPosition:
    delivery_start: datetime
    delivery_end: datetime
    dam_commitment_mw: float        # Original DAM position
    intraday_adjustment_mw: float   # Net IntraDay trades
    orders: List[IntraDayOrder]     # Order history

    @property
    def net_position_mw(self) -> float:
        return self.dam_commitment_mw + self.intraday_adjustment_mw
```

**Opportunity Evaluation:**
```python
def evaluate_adjustment(
    self,
    delivery_time: datetime,
    current_intraday_price: float,
    battery_state: BatteryState
) -> Dict:
    """
    Decision logic:

    IF spread > min_spread:
        IF we're SELLING in DAM:
            → Price went UP, we're doing well
            → Consider SELLING MORE
        IF we're BUYING in DAM:
            → Price went UP, bad deal
            → Consider REDUCING BUY

    IF spread < -min_spread:
        IF we're SELLING in DAM:
            → Price went DOWN, bad deal
            → Consider REDUCING SELL
        IF we're BUYING in DAM:
            → Price went DOWN, even better
            → Consider BUYING MORE
    """
```

**Order Types:**
```python
@dataclass
class IntraDayOrder:
    timestamp: datetime          # Order creation
    delivery_start: datetime     # Delivery period
    delivery_end: datetime
    power_mw: float             # Positive=sell, Negative=buy
    price_limit: float          # Max buy / Min sell price
    order_type: str             # "limit" or "market"
    status: str                 # pending/filled/partial/cancelled
```

---

### 3.5 BalancingTrader (`balancing_trader.py`)

#### Purpose
Manages real-time participation in aFRR and mFRR markets using DRL or rule-based strategies.

#### Key Concepts

**aFRR (Automatic Frequency Restoration Reserve):**
- Capacity product: "I promise to be available"
- Bid: MW capacity + price (EUR/MW/h)
- Activated automatically by frequency deviation
- Symmetric bidding typical (same up/down capacity)

**mFRR (Manual Frequency Restoration Reserve):**
- Energy product: "I will deliver if activated"
- Decision: Buy/sell based on price signals
- Uses trained DRL model for optimal decisions

#### Implementation Details

**Strategy Selection:**
```python
class BalancingTrader:
    def __init__(
        self,
        strategy_type: str = "AI",  # AI, RULE_BASED, AGGRESSIVE_HYBRID
        model_path: str = None,     # Required for AI
        ...
    ):
        # AI: Uses MaskablePPO trained model
        # RULE_BASED: Simple threshold logic
        # AGGRESSIVE_HYBRID: Time-window + price intelligence
```

**aFRR Bidding:**
```python
def _decide_afrr(self, battery_state, available_mw, market_data):
    """
    Simple rule-based aFRR strategy:
    1. Reserve fraction of available capacity (default 30%)
    2. Check SoC constraints:
       - Low SoC (<15%): Can't provide UP regulation
       - High SoC (>85%): Can't provide DOWN regulation
    3. Price: Slightly below market to ensure acceptance
    """
    reserve_mw = available_mw * self.afrr_reserve_fraction

    up_capacity = reserve_mw if soc > 0.15 else 0.0
    down_capacity = reserve_mw if soc < 0.85 else 0.0

    bid_price = min(afrr_up_price, afrr_down_price) * 0.95

    return up_capacity, down_capacity, bid_price
```

**mFRR Decision (DRL):**
```python
def _decide_mfrr(self, battery_state, available_mw, market_data):
    """
    Uses trained strategy to decide mFRR action.

    1. Build observation vector (20 features)
    2. Build action mask based on SoC
    3. Query strategy for action
    4. Convert action index to power MW
    """
    obs = self._build_observation(battery_state, market_data)
    mask = self._build_action_mask(battery_state, available_mw)

    action_idx = self._strategy.predict_action(obs, mask)
    power_fraction = self.action_levels[action_idx]  # -1 to +1

    return power_fraction * available_mw
```

**Action Masking:**
```python
def _build_action_mask(self, battery_state, available_mw):
    """
    Mask invalid actions based on SoC constraints.

    Example with SoC=10%:
    - Cannot discharge much (low energy)
    - Discharge actions masked
    - Charge and idle allowed

    Actions: [max_charge, ..., idle, ..., max_discharge]
    Indices: [0, 1, 2, ..., 10, ..., 18, 19, 20]
    """
    soc = battery_state.soc
    mask = np.ones(21, dtype=bool)

    # Calculate limits
    max_discharge = (soc - 0.05) * capacity / 0.25
    max_charge = (0.95 - soc) * capacity / 0.25

    for i, level in enumerate(action_levels):
        if level > 0 and level * available_mw > max_discharge:
            mask[i] = False  # Can't discharge this much
        if level < 0 and abs(level) * available_mw > max_charge:
            mask[i] = False  # Can't charge this much

    mask[10] = True  # Idle always valid
    return mask
```

---

### 3.6 BatteryOrchestrator (`orchestrator.py`)

#### Purpose
Main controller that coordinates all subsystems and manages the complete trading workflow.

#### Initialization
```python
BatteryOrchestrator(
    # Battery parameters
    max_power_mw: float = 30.0,
    capacity_mwh: float = 146.0,
    efficiency: float = 0.94,

    # Storage
    commitment_dir: str = "commitments",

    # Phase 2: Balancing
    balancing_enabled: bool = False,
    balancing_model_path: str = None,
    balancing_strategy: str = "RULE_BASED",
    afrr_enabled: bool = True,
    mfrr_enabled: bool = True,

    # Phase 3: IntraDay
    intraday_enabled: bool = False,
    intraday_min_spread: float = 5.0,
)
```

#### Workflow Methods

**Day-Ahead Process (`run_day_ahead`):**
```python
def run_day_ahead(self, target_date: date) -> DAMCommitment:
    """
    Executed: D-1 10:00 (before DAM gate closure)

    Flow:
    1. Generate price forecast
    2. Create optimal bids
    3. Store commitment
    4. Initialize IntraDay positions

    Returns: DAMCommitment for execution
    """
```

**IntraDay Process (`run_intraday`):**
```python
def run_intraday(self, current_time: datetime) -> Dict:
    """
    Executed: D-1 14:00 to D-0 H-1 (periodically)

    Flow:
    1. Compare DAM positions with current IntraDay prices
    2. Identify adjustment opportunities
    3. Return recommendations

    Returns: {opportunities, position_summary}
    """
```

**Real-Time Process (`run_realtime`):**
```python
def run_realtime(self, timestamp: datetime) -> Dict:
    """
    Executed: D-0 every 15 minutes

    Flow:
    1. Get DAM commitment for this interval
    2. Calculate remaining capacity
    3. Decide aFRR reservation
    4. Decide mFRR action
    5. Combine into dispatch command

    Returns: {
        dam_commitment_mw,
        afrr_up_reserved_mw,
        afrr_down_reserved_mw,
        mfrr_action_mw,
        total_dispatch_mw,
        soc,
    }
    """
```

---

## 4. Data Structures

### 4.1 Core Data Classes

```python
@dataclass
class PriceForecast:
    """DAM price forecast for a single day."""
    target_date: date
    timestamps: List[datetime]  # 96 timestamps (15-min intervals)
    buy_prices: np.ndarray      # 96 values - price to BUY energy
    sell_prices: np.ndarray     # 96 values - price to SELL energy
    confidence: Optional[np.ndarray] = None  # Optional confidence intervals

    @property
    def n_intervals(self) -> int:
        return len(self.timestamps)  # Always 96

    def get_hourly_prices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate to 24 hourly prices."""
        buy_hourly = self.buy_prices.reshape(24, 4).mean(axis=1)
        sell_hourly = self.sell_prices.reshape(24, 4).mean(axis=1)
        return buy_hourly, sell_hourly


@dataclass
class DAMCommitment:
    """DAM commitment schedule for a single day."""
    target_date: date
    timestamps: List[datetime]  # 96 timestamps
    power_mw: np.ndarray        # 96 values
    # Positive = SELL (discharge)
    # Negative = BUY (charge)
    # Zero = IDLE

    @property
    def total_sell_mwh(self) -> float:
        return float(np.sum(self.power_mw[self.power_mw > 0]) / 4)

    @property
    def total_buy_mwh(self) -> float:
        return float(np.sum(np.abs(self.power_mw[self.power_mw < 0])) / 4)


@dataclass
class BatteryState:
    """Current state of the battery."""
    timestamp: datetime
    soc: float                  # State of charge [0, 1]
    available_power_mw: float   # Currently available power
    capacity_mwh: float         # Total capacity

    @property
    def available_energy_mwh(self) -> float:
        return self.soc * self.capacity_mwh

    @property
    def available_headroom_mwh(self) -> float:
        return (1 - self.soc) * self.capacity_mwh


@dataclass
class BalancingDecision:
    """Decision for balancing market participation."""
    timestamp: datetime
    afrr_up_capacity_mw: float = 0.0
    afrr_down_capacity_mw: float = 0.0
    afrr_bid_price: float = 0.0
    mfrr_action_mw: float = 0.0
    total_power_mw: float = 0.0
    strategy_used: str = ""


@dataclass
class IntraDayOrder:
    """An IntraDay market order."""
    timestamp: datetime
    delivery_start: datetime
    delivery_end: datetime
    power_mw: float
    price_limit: float
    order_type: str = "limit"
    status: str = "pending"


@dataclass
class IntraDayPosition:
    """Current position for a delivery period."""
    delivery_start: datetime
    delivery_end: datetime
    dam_commitment_mw: float
    intraday_adjustment_mw: float = 0.0
    orders: List[IntraDayOrder] = field(default_factory=list)

    @property
    def net_position_mw(self) -> float:
        return self.dam_commitment_mw + self.intraday_adjustment_mw
```

### 4.2 Sign Conventions

**CRITICAL: Consistent sign conventions throughout the system**

| Value | Meaning | Battery Action | Grid Perspective |
|-------|---------|---------------|------------------|
| `+30 MW` | SELL | Discharge | Injection |
| `-30 MW` | BUY | Charge | Withdrawal |
| `0 MW` | IDLE | No action | No flow |

```python
# Example: DAM Commitment
power_mw = [
    -30.0,  # 00:00 - BUY (charging at 30 MW)
    -30.0,  # 00:15 - BUY
    0.0,    # 00:30 - IDLE
    30.0,   # 00:45 - SELL (discharging at 30 MW)
]

# Energy calculation (15-min interval = 0.25 hour)
energy_mwh = power_mw * 0.25
# -30 MW * 0.25h = -7.5 MWh (bought)
# +30 MW * 0.25h = +7.5 MWh (sold)
```

---

## 5. Market Timeline & Workflows

### 5.1 Complete Trading Day Timeline

```
═══════════════════════════════════════════════════════════════════════════
                           TRADING DAY TIMELINE
═══════════════════════════════════════════════════════════════════════════

D-2                  D-1                           D-0
 │                    │                             │
 │    ┌───────────────┼───────────────┐             │
 │    │               │               │             │
 │    ▼               ▼               ▼             │
 │  10:00           12:00           14:00           │
 │  DAM             DAM             DAM             │
 │  Bidding         Gate            Results         │
 │  Starts          Closes          Published       │
 │    │               │               │             │
 │    │   PHASE 1     │               │             │
 │    │   ──────────► │               │             │
 │    │   Forecast    │               │             │
 │    │   + Bid       │               │             │
 │                                    │             │
 │                                    │   PHASE 3   │
 │                                    │   ────────► │
 │                                    │   IntraDay  │
 │                                    │   Trading   │
 │                                    │             │
 │                                                  │
 │                                         ┌────────┴────────┐
 │                                         │   D-0 H-1       │
 │                                         │   IntraDay      │
 │                                         │   Closes        │
 │                                         └────────┬────────┘
 │                                                  │
 │                                                  ▼
 │                                         ┌────────────────┐
 │                                         │ REAL-TIME      │
 │                                         │ Every 15 min   │
 │                                         │                │
 │                                         │ PHASE 2        │
 │                                         │ DAM Execute    │
 │                                         │ + Balancing    │
 │                                         │ (aFRR + mFRR)  │
 │                                         └────────────────┘
 │
═══════════════════════════════════════════════════════════════════════════
```

### 5.2 Phase 1: Day-Ahead Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: DAY-AHEAD                              │
│                         Timing: D-1 10:00                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐          │
│   │  EntsoE3    │     │  DAMBidder  │     │  Commitment     │          │
│   │  Forecaster │────►│  Strategy   │────►│  Storage        │          │
│   └─────────────┘     └─────────────┘     └─────────────────┘          │
│         │                   │                     │                     │
│         │                   │                     │                     │
│         ▼                   ▼                     ▼                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐          │
│   │ PriceForecast│    │DAMCommitment│     │ commitment.npz  │          │
│   │ 96 buy/sell │     │ 96 power_mw │     │ forecast.csv    │          │
│   │ prices      │     │ values      │     │                 │          │
│   └─────────────┘     └─────────────┘     └─────────────────┘          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  INPUT:                                                                 │
│  - target_date: date                                                   │
│  - battery_state: BatteryState (current SoC)                           │
│                                                                         │
│  OUTPUT:                                                                │
│  - DAMCommitment: 96 power values                                      │
│  - Saved: forecast_{date}.csv, commitment_{date}.npz                   │
│                                                                         │
│  CONSTRAINTS APPLIED:                                                   │
│  - Max 2 daily cycles                                                  │
│  - SoC within [5%, 95%]                                                │
│  - Power within [-30, +30] MW                                          │
│  - Minimum spread requirement (30 EUR default)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Phase 3: IntraDay Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 3: INTRADAY                               │
│                         Timing: D-1 14:00 to D-0 H-1                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐          │
│   │ DAM Position│     │  IntraDay   │     │  Opportunity    │          │
│   │ (from Ph.1) │────►│  Prices     │────►│  Evaluation     │          │
│   └─────────────┘     └─────────────┘     └─────────────────┘          │
│         │                   │                     │                     │
│         │                   │                     ▼                     │
│         │                   │             ┌─────────────────┐          │
│         │                   │             │  IF spread >    │          │
│         │                   │             │  min_spread     │          │
│         │                   │             └────────┬────────┘          │
│         │                   │                      │                    │
│         │                   │                      ▼                    │
│         │                   │             ┌─────────────────┐          │
│         │                   │             │  Create Order   │          │
│         │                   │             │  (limit/market) │          │
│         │                   │             └────────┬────────┘          │
│         │                   │                      │                    │
│         │                   │                      ▼                    │
│         │                   │             ┌─────────────────┐          │
│         │                   │             │  Update Position│          │
│         │                   │             │  Tracking       │          │
│         │                   │             └─────────────────┘          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  DECISION LOGIC:                                                        │
│                                                                         │
│  spread = intraday_price - dam_price                                   │
│                                                                         │
│  IF spread > +5 EUR (prices UP):                                       │
│    IF DAM was SELL: prices up = good → consider SELL MORE              │
│    IF DAM was BUY:  prices up = bad  → consider REDUCE BUY             │
│                                                                         │
│  IF spread < -5 EUR (prices DOWN):                                     │
│    IF DAM was SELL: prices down = bad  → consider REDUCE SELL          │
│    IF DAM was BUY:  prices down = good → consider BUY MORE             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Phase 2: Real-Time Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 2: REAL-TIME                              │
│                         Timing: D-0, every 15 minutes                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    INPUT GATHERING                            │     │
│   │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │     │
│   │  │   DAM     │  │  Battery  │  │  aFRR     │  │  mFRR     │ │     │
│   │  │Commitment │  │   State   │  │  Prices   │  │  Prices   │ │     │
│   │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘ │     │
│   └────────┼──────────────┼──────────────┼──────────────┼────────┘     │
│            │              │              │              │               │
│            ▼              ▼              ▼              ▼               │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                 CAPACITY CALCULATION                          │     │
│   │                                                                │     │
│   │  dam_commitment = executor.get_current_commitment(timestamp)  │     │
│   │  remaining = max_power - abs(dam_commitment)                  │     │
│   │                                                                │     │
│   │  Example:                                                      │     │
│   │  - DAM commitment: +20 MW (selling)                           │     │
│   │  - Max power: 30 MW                                           │     │
│   │  - Remaining for balancing: 10 MW                             │     │
│   └──────────────────────────┬───────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    aFRR DECISION                              │     │
│   │                                                                │     │
│   │  afrr_reserve = remaining * 0.3  (30% for aFRR)              │     │
│   │                                                                │     │
│   │  IF soc > 15%: can provide UP regulation                      │     │
│   │  IF soc < 85%: can provide DOWN regulation                    │     │
│   │                                                                │     │
│   │  bid_price = market_price * 0.95  (ensure acceptance)         │     │
│   └──────────────────────────┬───────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    mFRR DECISION (DRL)                        │     │
│   │                                                                │     │
│   │  mfrr_capacity = remaining - afrr_reserve                     │     │
│   │                                                                │     │
│   │  observation = build_obs(soc, prices, time, ...)             │     │
│   │  action_mask = build_mask(soc, mfrr_capacity)                │     │
│   │  action_idx = strategy.predict(observation, action_mask)      │     │
│   │  mfrr_action = action_to_power(action_idx, mfrr_capacity)    │     │
│   └──────────────────────────┬───────────────────────────────────┘     │
│                              │                                          │
│                              ▼                                          │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                 OUTPUT: DISPATCH COMMAND                       │     │
│   │                                                                │     │
│   │  total_dispatch = dam_commitment + mfrr_action                │     │
│   │                                                                │     │
│   │  Example:                                                      │     │
│   │  - DAM: +20 MW (sell)                                         │     │
│   │  - mFRR: +5 MW (additional sell)                              │     │
│   │  - Total: +25 MW → Send to SCADA                              │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration Reference

### 6.1 Full Configuration Example

```python
from orchestrator import BatteryOrchestrator

orchestrator = BatteryOrchestrator(
    # ═══════════════════════════════════════════════════════════════
    # BATTERY PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    max_power_mw=30.0,          # Maximum charge/discharge power
    capacity_mwh=146.0,         # Total energy capacity
    efficiency=0.94,            # Round-trip efficiency (94%)

    # ═══════════════════════════════════════════════════════════════
    # STORAGE
    # ═══════════════════════════════════════════════════════════════
    commitment_dir="commitments",  # Directory for commitment files

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: BALANCING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    balancing_enabled=True,     # Enable aFRR + mFRR trading

    balancing_model_path="models/best_model.zip",  # DRL model path
    # Options: None (for rule-based), or path to .zip file

    balancing_strategy="AI",    # Strategy type
    # Options: "AI" (DRL model), "RULE_BASED", "AGGRESSIVE_HYBRID"

    afrr_enabled=True,          # Enable aFRR capacity bidding
    mfrr_enabled=True,          # Enable mFRR energy trading

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: INTRADAY CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    intraday_enabled=True,      # Enable IntraDay trading
    intraday_min_spread=5.0,    # Minimum spread to consider (EUR)
)
```

### 6.2 Component-Level Configuration

**DAMBidder:**
```python
DAMBidder(
    max_power_mw=30.0,
    capacity_mwh=146.0,
    efficiency=0.94,
    min_soc=0.05,               # Minimum SoC (5%)
    max_soc=0.95,               # Maximum SoC (95%)
    max_daily_cycles=2.0,       # Cycle limit (degradation control)
    buy_threshold_percentile=25,   # Buy below 25th percentile
    sell_threshold_percentile=75,  # Sell above 75th percentile
    min_spread_eur=30.0,        # Minimum spread to trade
)
```

**BalancingTrader:**
```python
BalancingTrader(
    model_path="models/best_model.zip",
    strategy_type="AI",
    n_actions=21,               # Discrete action space size
    max_power_mw=30.0,
    capacity_mwh=146.0,

    # aFRR settings
    afrr_enabled=True,
    afrr_reserve_fraction=0.3,  # Reserve 30% for aFRR
    afrr_min_price=5.0,         # Minimum capacity price

    # mFRR settings
    mfrr_enabled=True,
    mfrr_price_scale=100.0,     # Price normalization factor
)
```

**IntraDayTrader:**
```python
IntraDayTrader(
    max_power_mw=30.0,
    capacity_mwh=146.0,
    min_spread_eur=5.0,         # Minimum spread to trade
    max_position_change=0.5,    # Max 50% adjustment of DAM position
    price_forecast_weight=0.7,  # Weight for forecast vs current price
)
```

### 6.3 Environment Variables

```bash
# EntsoE3 model path
ENTSOE3_MODEL_PATH=/path/to/tft_v2_best.ckpt

# API keys (for production)
HENEX_API_KEY=your_key_here
IPTO_API_KEY=your_key_here
SCADA_ENDPOINT=https://scada.example.com

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 7. Integration Points

### 7.1 External System Interfaces

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SYSTEM INTERFACES                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                        MARKET DATA (INPUT)                      │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  HEnEx API:                                                      │    │
│  │  ├── DAM prices (hourly, D-1 14:00)                             │    │
│  │  ├── IntraDay prices (continuous)                               │    │
│  │  └── Market depth/orderbook                                     │    │
│  │                                                                  │    │
│  │  IPTO/ADMIE API:                                                 │    │
│  │  ├── aFRR capacity prices                                       │    │
│  │  ├── mFRR energy prices                                         │    │
│  │  ├── System imbalance                                           │    │
│  │  └── Activation signals                                         │    │
│  │                                                                  │    │
│  │  Weather API:                                                    │    │
│  │  ├── Solar irradiance forecast                                  │    │
│  │  ├── Wind speed forecast                                        │    │
│  │  └── Temperature forecast                                       │    │
│  │                                                                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                        BATTERY SYSTEM (I/O)                     │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  SCADA Interface:                                                │    │
│  │  ├── INPUT: SoC, power, temperature, alarms                     │    │
│  │  └── OUTPUT: Power setpoint                                     │    │
│  │                                                                  │    │
│  │  Protocol Options:                                               │    │
│  │  ├── Modbus TCP/RTU                                             │    │
│  │  ├── OPC-UA                                                     │    │
│  │  └── REST API                                                   │    │
│  │                                                                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                        TRADING PLATFORM (OUTPUT)                │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  DAM Bidding:                                                    │    │
│  │  ├── Submit bids (D-1 before 12:00)                             │    │
│  │  └── Format: CSV or API                                         │    │
│  │                                                                  │    │
│  │  IntraDay Orders:                                                │    │
│  │  ├── Limit/market orders                                        │    │
│  │  └── Order status callbacks                                     │    │
│  │                                                                  │    │
│  │  Balancing Bids:                                                 │    │
│  │  ├── aFRR capacity bids                                         │    │
│  │  └── mFRR energy bids                                           │    │
│  │                                                                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Data Flow Diagram

```
                            ┌─────────────────┐
                            │  MARKET DATA    │
                            │  FEEDS          │
                            └────────┬────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   update_market_data()      update_intraday_prices()                   │
│         │                           │                                   │
│         ▼                           ▼                                   │
│   ┌───────────┐               ┌───────────┐                            │
│   │_market_   │               │_intraday_ │                            │
│   │  _data    │               │  _prices  │                            │
│   └─────┬─────┘               └─────┬─────┘                            │
│         │                           │                                   │
│         └───────────┬───────────────┘                                   │
│                     │                                                   │
│                     ▼                                                   │
│            ┌────────────────┐                                          │
│            │ ORCHESTRATOR   │                                          │
│            │    CORE        │                                          │
│            └───────┬────────┘                                          │
│                    │                                                    │
│         ┌──────────┼──────────┐                                        │
│         │          │          │                                        │
│         ▼          ▼          ▼                                        │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                            │
│  │run_day_   │ │run_intra  │ │run_real   │                            │
│  │  ahead()  │ │   day()   │ │  time()   │                            │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                            │
│        │             │             │                                    │
│        ▼             ▼             ▼                                    │
│  DAMCommitment  ID_Opps     DispatchCmd                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │     SCADA       │
                            │  (Power Setpt)  │
                            └─────────────────┘
```

### 7.3 API Methods for Integration

```python
# ═══════════════════════════════════════════════════════════════════════
# INPUT METHODS (Call these with external data)
# ═══════════════════════════════════════════════════════════════════════

# Update battery telemetry (from SCADA)
orchestrator.update_battery_state(BatteryState(
    timestamp=datetime.now(),
    soc=0.65,
    available_power_mw=30.0,
    capacity_mwh=146.0,
))

# Update market data (from price feeds)
orchestrator.update_market_data({
    'mfrr_price_up': 150.0,
    'mfrr_price_down': 120.0,
    'afrr_up_price': 12.0,
    'afrr_down_price': 10.0,
    'system_imbalance': -50.0,  # MW
})

# Update IntraDay prices (from exchange)
orchestrator.update_intraday_prices(np.array([...]))  # 96 values


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT METHODS (Use these for trading actions)
# ═══════════════════════════════════════════════════════════════════════

# Day-ahead bidding
commitment = orchestrator.run_day_ahead(target_date)
# → Submit commitment.power_mw to HEnEx

# IntraDay recommendations
result = orchestrator.run_intraday(datetime.now())
# → result['top_opportunities'] contains adjustment suggestions

# Real-time dispatch
dispatch = orchestrator.run_realtime(datetime.now())
# → Send dispatch['total_dispatch_mw'] to SCADA
```

---

## 8. Error Handling

### 8.1 Error Categories

```python
# ═══════════════════════════════════════════════════════════════════════
# ERROR HIERARCHY
# ═══════════════════════════════════════════════════════════════════════

class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass

class ForecastError(OrchestratorError):
    """Price forecast failed."""
    pass

class BiddingError(OrchestratorError):
    """Bid generation failed."""
    pass

class CommitmentError(OrchestratorError):
    """Commitment tracking/execution error."""
    pass

class BalancingError(OrchestratorError):
    """Balancing market error."""
    pass

class ConstraintViolation(OrchestratorError):
    """Battery constraint violated."""
    pass
```

### 8.2 Fallback Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FALLBACK STRATEGIES                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FORECAST FAILURE:                                                      │
│  ├── Primary: EntsoE3 TFT model                                        │
│  ├── Fallback 1: XGBoost model                                         │
│  ├── Fallback 2: Pattern-based forecast                                │
│  └── Fallback 3: Last known prices + volatility                        │
│                                                                         │
│  DRL MODEL FAILURE:                                                     │
│  ├── Primary: MaskablePPO trained model                                │
│  ├── Fallback 1: AggressiveHybrid strategy                             │
│  ├── Fallback 2: Rule-based strategy                                   │
│  └── Fallback 3: Conservative (IDLE only)                              │
│                                                                         │
│  COMMUNICATION FAILURE:                                                 │
│  ├── SCADA timeout: Hold last setpoint                                 │
│  ├── Market data gap: Use last known + extrapolation                   │
│  └── API error: Retry with exponential backoff                         │
│                                                                         │
│  CONSTRAINT VIOLATION:                                                  │
│  ├── SoC limit: Clamp power to safe level                              │
│  ├── Power limit: Scale down proportionally                            │
│  └── Cycle limit: Reduce trading, prioritize DAM                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Deviation Handling

```python
def handle_dam_deviation(committed_mw: float, actual_mw: float):
    """
    Handle deviations from DAM commitment.

    Deviations incur imbalance charges:
    - Over-delivery: Paid at lower price
    - Under-delivery: Charged at higher price

    Strategy: Minimize deviations, log for settlement
    """
    deviation = actual_mw - committed_mw

    if abs(deviation) < 0.5:  # MW
        # Within tolerance
        return

    if deviation > 0:
        # Over-delivered (sold more than committed)
        log.warning(f"Over-delivery: {deviation} MW")
        # Will be settled at system sell price
    else:
        # Under-delivered (sold less than committed)
        log.warning(f"Under-delivery: {abs(deviation)} MW")
        # Will be charged at system buy price (penalty)

    # Record for settlement
    settlement_log.record(committed_mw, actual_mw, deviation)
```

---

## 9. Testing Strategy

### 9.1 Test Structure

```
orchestrator/
├── test_phase1.py      # DAM + Commitment tests
├── test_phase2.py      # Balancing (aFRR + mFRR) tests
└── test_phase3.py      # IntraDay tests

# Run all tests:
python -m orchestrator.test_phase1
python -m orchestrator.test_phase2
python -m orchestrator.test_phase3
```

### 9.2 Test Categories

**Unit Tests:**
```python
def test_dam_forecaster_output_shape():
    """Verify forecast has 96 intervals."""
    forecaster = DAMForecaster()
    forecast = forecaster.forecast(date.today() + timedelta(days=1))
    assert forecast.n_intervals == 96

def test_commitment_sign_convention():
    """Verify positive = sell, negative = buy."""
    executor = CommitmentExecutor()
    # ... test sign conventions
```

**Integration Tests:**
```python
def test_full_pipeline():
    """Test DAM → IntraDay → Balancing flow."""
    orch = BatteryOrchestrator(
        balancing_enabled=True,
        intraday_enabled=True,
    )

    # Day-ahead
    commitment = orch.run_day_ahead(tomorrow)
    assert commitment.total_sell_mwh > 0

    # IntraDay
    orch.update_intraday_prices(prices)
    id_result = orch.run_intraday(now)
    assert 'opportunities' in id_result

    # Real-time
    dispatch = orch.run_realtime(now)
    assert 'total_dispatch_mw' in dispatch
```

**Edge Case Tests:**
```python
def test_low_soc_constraints():
    """Verify system respects low SoC limits."""
    battery = BatteryState(soc=0.06)  # Near minimum
    # Should limit sell/discharge actions

def test_high_soc_constraints():
    """Verify system respects high SoC limits."""
    battery = BatteryState(soc=0.94)  # Near maximum
    # Should limit buy/charge actions

def test_no_spread_scenario():
    """Verify no trades when spread insufficient."""
    # All prices identical → no opportunities
```

### 9.3 Backtesting Framework

```python
def run_backtest(
    start_date: date,
    end_date: date,
    historical_prices: pd.DataFrame,
) -> BacktestResult:
    """
    Simulate trading over historical period.

    Returns:
    - Total profit/loss
    - Sharpe ratio
    - Max drawdown
    - Win rate
    - Average trade profit
    """
    orchestrator = BatteryOrchestrator(...)

    results = []
    for day in date_range(start_date, end_date):
        # Simulate day-ahead
        commitment = orchestrator.run_day_ahead(day)

        # Simulate real-time execution
        for interval in range(96):
            dispatch = orchestrator.run_realtime(...)
            pnl = calculate_pnl(dispatch, actual_prices)
            results.append(pnl)

    return BacktestResult(results)
```

---

## 10. Production Deployment

### 10.1 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRODUCTION DEPLOYMENT                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    KUBERNETES CLUSTER                            │   │
│  │                                                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │ Orchestrator │  │   Market     │  │   SCADA      │           │   │
│  │  │   Service    │  │   Connector  │  │   Gateway    │           │   │
│  │  │              │  │              │  │              │           │   │
│  │  │ - DAM bidding│  │ - HEnEx API  │  │ - Modbus/TCP │           │   │
│  │  │ - IntraDay   │  │ - IPTO API   │  │ - OPC-UA     │           │   │
│  │  │ - Balancing  │  │ - Price feed │  │ - REST API   │           │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │   │
│  │         │                 │                 │                    │   │
│  │         └─────────────────┼─────────────────┘                    │   │
│  │                           │                                      │   │
│  │                           ▼                                      │   │
│  │                  ┌──────────────────┐                           │   │
│  │                  │    Message       │                           │   │
│  │                  │    Queue         │                           │   │
│  │                  │  (Redis/Kafka)   │                           │   │
│  │                  └──────────────────┘                           │   │
│  │                           │                                      │   │
│  │         ┌─────────────────┼─────────────────┐                    │   │
│  │         │                 │                 │                    │   │
│  │         ▼                 ▼                 ▼                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │  PostgreSQL  │  │   Grafana    │  │   Alerting   │           │   │
│  │  │  (History)   │  │  (Dashboard) │  │  (PagerDuty) │           │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Operational Runbook

**Daily Operations:**
```
06:00  System health check
09:00  Review previous day performance
10:00  Generate DAM forecast and bids
11:30  Final review of bids before gate closure
12:00  DAM gate closure (bids submitted)
14:00  DAM results published - verify positions
14:15  IntraDay trading window opens
       Continuous: Monitor and adjust positions
00:00  Day end - generate settlement report
```

**Critical Alerts:**
```
PRIORITY 1 (Immediate response):
- SCADA communication failure
- SoC out of bounds (<3% or >97%)
- Power limit exceeded

PRIORITY 2 (15-minute response):
- Large DAM deviation (>5 MW)
- DRL model failure (fallback active)
- Price feed stale (>5 min)

PRIORITY 3 (1-hour response):
- Minor forecast deviation
- IntraDay order timeout
- Settlement discrepancy
```

### 10.3 Rollback Procedures

```python
def emergency_shutdown():
    """
    Emergency procedure: Stop all trading, go to safe state.

    Triggers:
    - Critical system failure
    - Regulatory instruction
    - Major market anomaly
    """
    # 1. Stop all new orders
    trading_enabled = False

    # 2. Cancel pending orders
    cancel_all_pending_orders()

    # 3. Set battery to IDLE
    scada.set_power(0.0)

    # 4. Notify operators
    alert.send("EMERGENCY SHUTDOWN ACTIVATED")

    # 5. Preserve state for analysis
    save_state_snapshot()
```

---

## 11. Performance Considerations

### 11.1 Timing Requirements

| Operation | Max Latency | Typical | Critical |
|-----------|------------|---------|----------|
| DAM forecast | 30s | 5s | No |
| DAM bid generation | 10s | 2s | No |
| IntraDay evaluation | 5s | 500ms | No |
| Real-time dispatch | 500ms | 50ms | **YES** |
| SCADA setpoint | 100ms | 20ms | **YES** |

### 11.2 Optimization Strategies

**Model Loading:**
```python
# Lazy loading - only load when needed
def _load_model(self):
    if self._model is not None:
        return  # Already loaded
    self._model = load_model(path)

# Preload for real-time critical paths
def warmup():
    """Call during startup to preload models."""
    forecaster._load_model()
    balancing_trader._load_strategy()
```

**Caching:**
```python
# Cache forecast during IntraDay window
@lru_cache(maxsize=1)
def get_cached_forecast(target_date: date):
    return forecaster.forecast(target_date)

# Cache action masks (SoC-dependent)
def get_action_mask(soc_bucket: int):
    """SoC bucketed to nearest 5% for caching."""
    return cached_masks[soc_bucket]
```

### 11.3 Resource Requirements

```
Minimum Production Requirements:
├── CPU: 4 cores (8 recommended)
├── RAM: 8 GB (16 GB recommended)
├── Storage: 50 GB SSD
├── Network: 100 Mbps (1 Gbps recommended)
└── Redundancy: Active-passive failover

GPU (Optional, for DRL training):
├── NVIDIA GPU with CUDA support
├── 8 GB VRAM minimum
└── Used only for model training, not inference
```

---

## 12. Security Considerations

### 12.1 Access Control

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ACCESS CONTROL MATRIX                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Role              │ DAM Bid │ IntraDay │ Real-time │ Config │ Admin   │
│  ──────────────────┼─────────┼──────────┼───────────┼────────┼──────── │
│  Operator          │  View   │   View   │   View    │  None  │  None   │
│  Trader            │  Edit   │   Edit   │   View    │  None  │  None   │
│  Senior Trader     │  Edit   │   Edit   │   Edit    │  View  │  None   │
│  System Admin      │  View   │   View   │   View    │  Edit  │  Edit   │
│  Risk Manager      │  View   │   View   │   View    │  Edit  │  None   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Data Protection

```python
# API keys in environment variables, not code
api_key = os.environ.get('HENEX_API_KEY')

# Encrypt sensitive data at rest
def encrypt_commitment(commitment):
    return fernet.encrypt(pickle.dumps(commitment))

# Audit logging for all trading actions
def log_trading_action(action, user, timestamp):
    audit_log.info(
        action=action,
        user=user,
        timestamp=timestamp,
        ip_address=get_client_ip(),
    )
```

### 12.3 Network Security

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NETWORK TOPOLOGY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INTERNET                                                               │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────┐                                                        │
│  │  Firewall  │  (Only HTTPS 443 inbound)                              │
│  └─────┬──────┘                                                        │
│        │                                                                │
│        ▼                                                                │
│  ┌────────────┐                                                        │
│  │  WAF/LB    │  (Rate limiting, DDoS protection)                      │
│  └─────┬──────┘                                                        │
│        │                                                                │
│  ══════╪══════════════════════════════════════════ DMZ ═══════════     │
│        │                                                                │
│        ▼                                                                │
│  ┌────────────────────────────────────────┐                            │
│  │           INTERNAL NETWORK              │                            │
│  │                                          │                            │
│  │  ┌──────────┐  ┌──────────┐  ┌───────┐ │                            │
│  │  │Orchestr. │──│ Database │──│ SCADA │ │                            │
│  │  └──────────┘  └──────────┘  └───────┘ │                            │
│  │                                          │                            │
│  └────────────────────────────────────────┘                            │
│                                                                         │
│  SCADA NETWORK (Isolated)                                               │
│  ┌────────────────────────────────────────┐                            │
│  │  Battery Management System              │                            │
│  │  (Air-gapped or hardware firewall)      │                            │
│  └────────────────────────────────────────┘                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **DAM** | Day-Ahead Market - Electricity traded for next-day delivery |
| **IntraDay** | Continuous market for same-day position adjustments |
| **aFRR** | Automatic Frequency Restoration Reserve - Capacity product |
| **mFRR** | Manual Frequency Restoration Reserve - Energy product |
| **SoC** | State of Charge - Battery energy level (0-100%) |
| **HEnEx** | Hellenic Energy Exchange - Greek electricity market operator |
| **IPTO/ADMIE** | Independent Power Transmission Operator - Greek TSO |
| **Commitment** | Binding obligation to buy/sell energy at specific times |
| **Spread** | Price difference between buy and sell (or between markets) |
| **Cycle** | One full charge + discharge of battery capacity |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-10 | Initial documentation |

---

## Appendix C: Contact Information

**Technical Lead**: [Name]
**Operations**: [Contact]
**Emergency**: [24/7 Number]

---

*This document is confidential and intended for internal use only.*
