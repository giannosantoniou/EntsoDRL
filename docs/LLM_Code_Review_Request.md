# EntsoDRL — Expert Code Review Request

> **Purpose**: Review this DRL battery trading system for production readiness.
> **Goal**: Identify bugs, unrealistic assumptions, missing market conditions, and improvements.
> **Target Market**: HEnEx (Greek Energy Exchange), adaptable to EU markets.

---

## System Overview

EntsoDRL uses Deep Reinforcement Learning (SAC/TD3) to control a 146 MWh / 30 MW BESS (Battery Energy Storage System) trading across multiple energy markets in Greece.

**Architecture**: SAC continuous wrapper (`BatteryEnvUnifiedSAC`) around a discrete multi-market environment (`BatteryEnvUnified`). The wrapper bypasses the inner env's discrete cascade and executes trades directly in absolute MW with a single inverter (NET physical execution).

**Training**: Stable Baselines 3 SAC/TD3, 10M timesteps, 15-min resolution, 7-day episodes.

---

## 1. Action Space (Box(7,) continuous [-1, +1])

```
[0] aFRR Commitment:    (a[0]+1)/2 * 30 → [0, 30] MW
[1] aFRR Price Tier:    0.7 + (a[1]+1)/2 * 0.6 → [0.7, 1.3] multiplier
[2] XBID Quantity:      a[2] * 30 → [-30, +30] MW (neg=buy, pos=sell)
[3] XBID Price Offset:  a[3] * 10 → [-10, +10] EUR/MWh from mid-price
[4] mFRR Quantity:      a[4] * 30 → [-30, +30] MW
[5] FreeBid Quantity:   a[5] * 30 → [-30, +30] MW (full-market only, else 0)
[6] DAM Aggressiveness: a[6] raw → [-1, +1] (controls bid threshold)
```

### Questions for Review:
- Is 7 dimensions sufficient? Should we add IDA quantity/price?
- The aFRR commitment is always positive (0-30 MW). Should it be bidirectional?
- XBID price offset ±10 EUR — is this range realistic?

---

## 2. Execution Model (NET Physical)

The battery has ONE inverter. It either charges or discharges — never both simultaneously.

```python
# Phase 1: Determine all market positions (NO SoC changes)
dam_mw = get_dam_commitment()           # mandatory, from CSV or EUPHEMIA
afrr_energy_mw = determine_afrr()       # 0 if not activated
xbid_mw = decoded['xbid_mw']           # agent-controlled
mfrr_mw = determine_mfrr()             # 0 if not activated

# Phase 2: NET physical MW (single inverter setpoint)
net_mw = dam_mw + afrr_energy_mw + xbid_mw + mfrr_mw
net_mw = clip(net_mw, -30, +30)

# Phase 3: SINGLE SoC update
actual_energy, soc_delta = execute_energy(net_mw)

# Phase 4: Financial settlement (per market, independent of physical)
dam_revenue = dam_mw * dam_price * time_step
xbid_revenue = xbid_mw * xbid_price * time_step
mfrr_revenue = mfrr_mw * mfrr_price * time_step  # only if activated
afrr_revenue = afrr_mw * afrr_price * time_step  # only if activated
```

### Questions for Review:
- Is financial settlement per-market × price × time_step correct? Or should it use actual_energy (efficiency-adjusted)?
- When net_mw is clipped to ±30 MW, which market gets priority? Currently no priority — all scaled proportionally.
- In reality, DAM and aFRR are mandatory. Should we prioritize them over XBID/mFRR when capacity is exceeded?

---

## 3. SoC Update Formulas

```python
# Battery: 146 MWh, 30 MW, efficiency 94%, sqrt(0.94) = 0.9695
# Operating range: 5% - 95% (usable: 131.4 MWh)
# Time step: 0.25h (15 minutes)

# DISCHARGE (sell, power > 0):
energy_mwh = power_mw * time_step_hours            # 30 * 0.25 = 7.5 MWh
soc_delta = energy_mwh / sqrt(eff) / capacity_mwh  # 7.5 / 0.9695 / 146 = 0.0530
new_soc = max(min_soc, old_soc - soc_delta)
actual_energy = (old_soc - new_soc) * capacity * sqrt(eff)  # MWh to grid

# CHARGE (buy, power < 0):
energy_mwh = abs(power_mw) * time_step_hours
soc_delta = energy_mwh * sqrt(eff) / capacity_mwh  # 7.5 * 0.9695 / 146 = 0.0498
new_soc = min(max_soc, old_soc + soc_delta)
actual_energy = -((new_soc - old_soc) * capacity / sqrt(eff))  # MWh from grid (negative)
```

**Physical constraints**:
- Full charge (5%→95%) at 30 MW: 4.52 hours (271 min)
- Full discharge (95%→5%) at 30 MW: 4.25 hours (255 min)
- Max cycles/day: 2.74 (theoretical), target: 2.5

### Questions for Review:
- Is sqrt(efficiency) per direction correct? Some models use efficiency only on discharge.
- Should we model depth-of-discharge effects? (deeper cycles = more degradation)
- Should ramp rate limits be modeled? (e.g., max 10 MW/min change)

---

## 4. Market Models

### 4.1 DAM (Day-Ahead Market)
- **Bidding**: Agent controls aggressiveness [-1,+1] → affects spread threshold
- **EUPHEMIA simulation**: Sell accepted if limit_price <= clearing_price; Buy if limit >= clearing
- **Partial acceptance**: 50-100% for bids within 2 EUR of clearing
- **Settlement**: At clearing price (marginal pricing), NOT bid price
- **Acceptance rate**: ~53-85% depending on aggressiveness
- **Penalty for non-delivery**: 800 EUR/MWh (fixed, no data leakage)

**Questions**: Is 800 EUR/MWh realistic? Real imbalance prices can be 100-1200 EUR/MWh.

### 4.2 aFRR (Automatic Frequency Restoration Reserve)
- **Commitment**: 4-hour blocks, agent sets MW and price tier
- **Selection**: Probability based on capacity offered vs TSO requirement
  - `base_prob = min(0.85, commitment/estimated_pool + 0.3)`
  - `selection_prob = clip(base_prob * (1.4 - 0.4*price_tier), 0.05, 0.90)`
- **Activation**: Based on real ADMIE `afrr_activated_up/down_mwh` data
  - `prob = min(0.70, commitment / max(activated_vol, 1.0))`
- **Settlement**: Highest-Value-Rule: `max(afrr_price, mfrr_price)` per ISP
- **Non-response penalty**: 500 EUR/MWh (skipped if SoC-starved)

**Questions**: Is the selection probability formula realistic? Real ADMIE selection depends on merit order, not random probability. Is 4-hour block correct? Some markets use 1-hour or variable blocks.

### 4.3 mFRR (Manual Frequency Restoration Reserve)
- **Activation**: TSO-controlled, based on real ADMIE `mfrr_activated_up/down_mwh`
  - `prob = min(0.60, abs(mfrr_mw) / max(real_activated, 1.0))`
- **Settlement**: Pay-as-cleared at `mfrr_price_up` or `mfrr_price_down`
- **Price cap**: 1000 EUR/MWh
- **Daily cycles**: mFRR trades NOT counted toward daily_cycles (TSO-mandated)

**Questions**: Is the activation probability formula correct? Real activation is deterministic (based on merit order + system need), not probabilistic.

### 4.4 XBID (Continuous Intraday)
- **Agent control**: MW quantity + price offset from mid-price
- **Prices**: From data columns `xbid_price_bid`, `xbid_price_ask`
- **Execution**: Always executed (no rejection simulation)
- **Settlement**: At agent's price (mid + offset)

**Questions**: Should XBID have execution probability? In reality, continuous orders may not fill. Should there be a minimum trade duration?

### 4.5 IDA (Intra-Day Auctions)
- **Gate closures**: IDA1 (D-1 15:00), IDA2 (D-1 22:00), IDA3 (D 10:00)
- **Currently**: Rule-based schedule generation in inner env
- **Agent control**: Not directly controlled in SAC wrapper

**Questions**: Should the agent control IDA bids? Currently only available in the discrete inner env.

---

## 5. Reward Function

```
Reward = (trading_pnl + shaping_bonus - shaping_penalty) * reward_scale

trading_pnl = market_revenue - real_costs
  market_revenue = DAM + aFRR_cap + aFRR_energy + IntraDay + mFRR + IDA + XBID + FreeBid
  real_costs = DAM_violation + aFRR_nonresponse + IDA_violation

shaping_bonus = dam_compliance_bonus + price_timing_bonus
shaping_penalty = physical_violation + calendar_aging + degradation

reward_scale = 0.01
```

### Component Details

| Component | Formula | Typical Range |
|-----------|---------|---------------|
| DAM revenue | `dam_mw * price * time_step` | -50 to +50 EUR/step |
| aFRR capacity | `commitment * cap_price * time_step` | 0 to 5 EUR/step |
| aFRR energy | `energy * max(afrr, mfrr_price) * time_step` | 0 to 30 EUR/step |
| IntraDay | `xbid_mw * xbid_price * time_step` | -30 to +30 EUR/step |
| mFRR | `mfrr_mw * mfrr_price * time_step` | -50 to +50 EUR/step |
| DAM violation | `shortfall * 800 * time_step` | 0 to -200 EUR/step |
| aFRR non-response | `required * 500 * time_step` | 0 to -100 EUR/step |
| Degradation | `abs(energy) * 12 EUR/MWh` | 0 to 90 EUR/step |
| Network charges | `abs(energy) * 4 EUR/MWh` | 0 to 30 EUR/step |
| Calendar aging | `0.5 + 0.02*(SoC%-50)^2` | 0.5 to 50 EUR/step |
| Availability contract | `30MW * 13.15 EUR/MW/h * 0.25h` | +99 EUR/step (fixed) |
| Price timing bonus | `energy * (price_dev/mean) * 5 * time_step` | -10 to +10 EUR/step |

### SAC Wrapper Additional Penalties

| Penalty | Formula | Purpose |
|---------|---------|---------|
| SoC soft margin | `60 * proximity * intensity` (near 5% or 95%) | Smooth SoC limits |
| Cycle excess | `2000 * (cycles - 5.0)^1.5` (if > 5 cycles/day) | Prevent over-cycling |

### Questions for Review:
- Is 12 EUR/MWh degradation realistic for LFP? Literature suggests 5-15 EUR/MWh.
- Calendar aging formula `0.5 + 0.02*(SoC%-50)^2` — is this calibrated? At SoC=95%: 0.5 + 0.02*2025 = 41 EUR/step = 3,936 EUR/day. Is this too high?
- Availability contract 115K EUR/MW/year = 13.15 EUR/MW/h. Should this be conditional on actually being available (not in maintenance)?
- Network charges 4 EUR/MWh per direction. Real ADMIE/HEDNO charges: are they per-direction or per-round-trip?

---

## 6. Observation Space (72 base features)

### Feature Groups

| Group | Features | Indices | Description |
|-------|----------|---------|-------------|
| 1. Battery | 4 | 0-3 | SoC, max_discharge, max_charge, cycles_remaining |
| 2. Market Prices | 13 | 4-16 | DAM, imbalance, mFRR, spreads, HVR |
| 3. Time | 4 | 17-20 | Hour sin/cos, DoW sin/cos |
| 4. DAM Lookahead | 13 | 21-33 | Current + 12h commitments |
| 5. Price Lookahead | 12 | 34-45 | 12h price forecasts |
| 6. aFRR State | 8 | 46-53 | Cap prices, activation, commitment |
| 7. Risk/Imbalance | 12 | 54-65 | Reserve, shortfall, momentum, DAM acceptance |
| 8. Market Phase | 4 | 66-69 | IDA, stress, RES, mFRR direction |
| 9. Timing | 4 | 70-73 | Price vs typical, peak flags |

**Optional**: +9 (forecast), +20 (market forecast), +13 (full market)

### Questions for Review:
- 72 features for SAC — is this too many? Should we use feature selection?
- Price lookahead (Group 5) uses ML forecaster. Is 12-hour horizon sufficient?
- DAM lookahead (Group 4) — beyond current day, values are 0 (unknown). Is this realistic? DAM results for D+1 are published at D 12:00-13:00.

---

## 7. Training Configuration

```python
# SAC Hyperparameters
algorithm = "SAC"  # or "TD3"
learning_rate = 3e-4
buffer_size = 2,000,000
learning_starts = 10,000
batch_size = 256
tau = 0.005          # Soft Q update rate
gamma = 0.99         # Discount factor (~100 step horizon)
ent_coef = "auto"    # Automatic entropy tuning (SAC only)
policy = "MlpPolicy"
net_arch = [256, 256]

# Environment
time_step_hours = 0.25  # 15-min resolution
episode_length = 672    # 672 * 0.25h = 168h = 7 days
n_envs = 8             # Parallel environments (SubprocVecEnv)
VecNormalize(norm_obs=True, norm_reward=False, clip_obs=10.0)

# Curriculum
degradation_cost: 0 → 5 EUR/MWh over first 20% of training
```

### Questions for Review:
- `gamma=0.99` → effective horizon ~100 steps = 25 hours. Is this enough for weekly patterns?
- `net_arch=[256,256]` — should we try deeper/wider for 72 features?
- `norm_reward=False` — should reward normalization be enabled?
- Episode length 7 days — should it be longer (e.g., 30 days) to learn monthly patterns?

---

## 8. Data Pipeline

**Training data**: `unified_multimarket_training_v2.csv` (71,475 rows)
- **Period**: ~2021-2026 (mixed hourly + 15-min resolution)
- **Columns**: 122 (prices, volumes, activations, forecasts, lags, rolling stats)
- **Split**: 80% train / 20% eval (temporal, no shuffle)

**Key columns used**:
- `price` — DAM clearing price
- `dam_commitment` — Pre-generated (or agent-endogenous)
- `mfrr_activated_up/down_mwh` — Real ADMIE activation volumes
- `afrr_activated_up/down_mwh` — Real ADMIE activation volumes
- `xbid_price_bid/ask` — Synthetic XBID prices
- `imbalance_price` — Single imbalance price (ADMIE)

**Data leakage prevention**:
- All lags are backward-looking (shift(-N), N > 0)
- ISP cascade spread uses 24h lag
- DAM lookahead respects 14:00 publication cutoff
- Forecast columns use ML predictions, not actuals

### Questions for Review:
- `xbid_price_bid/ask` are synthetic (DAM ± spread). Should we use real XBID data?
- `dam_commitment` is pre-generated. The EUPHEMIA simulation uses forecast vs actual — does the forecast quality affect training too much?
- Should we add more data sources? (e.g., gas prices, carbon prices, interconnector flows)

---

## 9. Known Limitations

1. **IDA auctions not agent-controlled** in SAC wrapper — rule-based only
2. **No order book simulation** — XBID fills are guaranteed (no partial fills)
3. **No market impact** — 30 MW trades don't affect prices
4. **Fixed battery parameters** — no temperature effects, no SoH degradation over time
5. **No maintenance windows** — battery assumed always available
6. **No portfolio effects** — single battery, no cross-asset optimization
7. **Synthetic XBID prices** — not from real continuous market data
8. **No regulatory regime changes** — rules are static during training
9. **Single imbalance price** — implemented but `imbalance_price` column may not always be populated
10. **No cross-border flows** — Serbia DAM as proxy only

---

## 10. Recent Bug Fixes (Chronological)

1. **Phantom energy** — Revenue calculated per-market while SoC updated per-market independently. Fixed: single NET SoC update.
2. **Revenue sign convention** — Buy was generating positive revenue. Fixed: signed actual_energy.
3. **Double-counting cycles** — Each market added to daily_cycles independently. Fixed: single cycle count from net SoC delta.
4. **Episode length mismatch** — 168 steps at 0.25h = 1.75 days, not 7 days. Fixed: 672 steps.
5. **DAM 100% acceptance** — All CSV commitments were "won". Fixed: EUPHEMIA simulation.
6. **Oracle features** — Price lookahead used actual future prices. Fixed: ML forecaster.

---

## 11. What I Need From You

Please review the above and identify:

1. **Physics bugs** — Are the SoC formulas correct? Energy balance?
2. **Market realism** — Does the simulation match real HEnEx/ADMIE rules?
3. **Reward design** — Are the incentives aligned? Will the agent learn optimal trading?
4. **Missing features** — What real-world conditions should we add?
5. **Architectural issues** — Is the NET execution model correct? Financial settlement?
6. **Training concerns** — Hyperparameters, observation space, action space?
7. **Production readiness** — What would break in production vs simulation?

Be specific: cite formulas, suggest values, reference EU electricity market regulations where applicable.
