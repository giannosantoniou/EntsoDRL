# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
# AI Persona & Rules

## Role
- Act as a **Senior Software Developer**.
- Tone: Professional, authoritative, yet collaborative.

## Coding Standards (Non-negotiable)
- **SOLID Principles**: Must be applied in every approach.
- **Testing**: All code must include associated tests.
- **Comments**: Write comments in a first-person style (e.g., "I checked this edge case...").

## Output Formats
- **Funding Proposals**: When asked, provide text in a format allowed for immediate copy-pasting.
- **Language**: Use **Greek** for explanations, but keep technical terms in English.

## Project Overview

EntsoDRL is a Deep Reinforcement Learning system for automated battery energy trading in the Greek Energy Market (HEnEx). It implements intelligent battery charging/discharging strategies to maximize profit while minimizing degradation.

**Tech Stack**: Python 3.10+, Stable Baselines 3 (PPO/MaskablePPO), Gymnasium, PyTorch, pandas

## Common Commands

### Training
```bash
# Sine wave curriculum (synthetic data)
python agent/train.py --mode sine_wave --timesteps 1000000

# Historical data training
python agent/train.py --mode historical --data_path data/feasible_data_with_dam.csv --timesteps 1000000

# MaskablePPO with action masking (recommended)
python agent/train_masked.py
```

### Evaluation
```bash
# Benchmark vs heuristic solver
python agent/evaluate.py

# MaskablePPO evaluation
python agent/evaluate_masked.py

# Detailed trade analysis
python agent/evaluate_detailed.py
```

### Running the System
```bash
# Training mode with AI strategy
python main.py

# With different strategies
python main.py --strategy ai          # AI (default)
python main.py --strategy rule        # Rule-based fallback
python main.py --strategy random      # Random (testing)
python main.py --strategy conservative # Always idle (safety)

# Custom model
python main.py --model models/custom_model.zip --cycles 1000
```

### Paper Trading
```bash
cd production
python paper_trading.py
python monitor_dashboard.py
```

### Testing
```bash
pytest tests/
pytest tests/test_battery.py -v
```

### Docker
```bash
docker-compose build
docker-compose up training
docker-compose up evaluation
docker-compose up tests
```

## Architecture

The system uses a three-layer dependency-inverted architecture:

### Data Layer (`data/data_provider.py`)
- **Interface**: `IDataProvider` (abstract base class)
- **Implementations**:
  - `SimulatedDataProvider`: Training mode, uses CSV historical data
  - `LiveDataProvider`: Production mode, connects to real APIs
- **DTOs**: `MarketState`, `BatteryTelemetry`, `PriceForecast`, `NormalizationConfig`

### Logic Layer (`agent/decision_strategy.py`)
- **Interface**: `IDecisionStrategy` (abstract base class)
- **Implementations**:
  - `RLAgentStrategy`: Trained MaskablePPO/PPO models
  - `RuleBasedStrategy`: Price threshold fallback
  - `RandomStrategy`: Testing only
  - `ConservativeStrategy`: Safety mode (always idle)
- **Factory**: `create_strategy()` for instantiation

### Control Layer (`agent/battery_controller.py`)
- `BatteryController`: Orchestrates data + decisions
- Builds observations, calculates action masks, executes dispatch
- `run_continuous()` for main control loop

### Environments (`gym_envs/`)
- `BatteryEnvMasked`: Main environment with action masking (Discrete(21) actions: -50MW to +50MW)
- `BatteryEnvRecurrent`: LSTM-compatible variant
- `BatteryEnvV2`: Multi-market (DAM + Intraday)
- `action_mask.py`: Safety wrapper preventing physically impossible actions

## Key Patterns

**Action Masking**: All strategies must respect action masks based on SoC constraints (5%-95% operating range). Use `sb3_contrib.MaskablePPO` and `ActionMasker` wrapper.

**VecNormalize**: Training uses `VecNormalize(norm_obs=True, norm_reward=True)`. During inference, set `training=False` and `norm_reward=False`.

**Curriculum Learning**: Degradation cost ramps from 0 to target (15 EUR/MWh) over warmup period via `DegradationCurriculumCallback`.

**Observation Features** (16-19 features):
- SoC, max_discharge, max_charge (normalized)
- Current price, DAM commitment (normalized)
- Hour sine/cosine encoding
- Price lookahead (4h), DAM commitment lookahead (4h)
- Optional: bid/ask spread, imbalance risk

## Configuration

**Environment Variables** (`.env`):
```
ENTSOE_API_KEY=your_key_here
```

**Battery Parameters** (current hardware):
```python
{
    'capacity_mwh': 146.0,       # 146 MWh capacity (~5h duration)
    'max_discharge_mw': 30.0,   # 30 MW inverter rating (unchanged)
    'efficiency': 0.94,
}
```

**Mode Switching**: `config.py` contains `Mode.TRAINING` / `Mode.PRODUCTION` enum and `create_data_provider()` factory.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Production entry point |
| `config.py` | Mode switching & factories |
| `agent/decision_strategy.py` | Strategy interface & implementations |
| `agent/battery_controller.py` | Main control loop |
| `agent/feature_engineer.py` | Observation construction |
| `data/data_provider.py` | Data abstraction layer |
| `gym_envs/battery_env_masked.py` | Main training environment |
| `production/paper_trading.py` | Live paper trading |

## Models

Production models stored in `models/`:
- `best_model_5M_profit.zip` - Main production model
- `ppo_masked.zip` - MaskablePPO variant
- Corresponding `vec_normalize_*.pkl` files for observation normalization
