# EntsoDRL: Deep Learning Architecture & Logic

## 1. Overview

Το EntsoDRL χρησιμοποιεί **Deep Reinforcement Learning (DRL)** για αυτόματη διαχείριση μπαταρίας αποθήκευσης ενέργειας (146 MWh / 30 MW) στο Ελληνικό Χρηματιστήριο Ενέργειας (HEnEx). Ο agent μαθαίνει πότε να φορτίζει (buy), αποφορτίζει (sell) ή να μένει αδρανής (idle) ώστε να μεγιστοποιεί το κέρδος και να ελαχιστοποιεί τη φθορά της μπαταρίας.

### Neural Networks στο σύστημα

Το σύστημα χρησιμοποιεί **2 ξεχωριστά ML μοντέλα**:

| # | Μοντέλο | Τύπος | Σκοπός |
|---|---------|-------|--------|
| 1 | **PPO Agent** (MaskablePPO) | Feed-Forward Neural Network (MLP) | Λήψη αποφάσεων trading (ο "εγκέφαλος") |
| 2 | **Price Forecaster** | LightGBM Gradient Boosting (12 μοντέλα) | Πρόβλεψη τιμών ενέργειας 1-12 ώρες μπροστά |

---

## 2. Neural Network #1: PPO Agent (MaskablePPO)

### 2.1 Αρχιτεκτονική

Ο PPO Agent αποτελείται από **2 νευρωνικά δίκτυα** που μοιράζονται τα ίδια hidden layers:

```
                    ┌─────────────────────────────────────────┐
                    │          SHARED TRUNK (MLP)              │
                    │                                         │
  44 features ──►   │  Input(44) → Dense(256) → ReLU          │
  (observation)     │           → Dense(256) → ReLU           │
                    │           → Dense(128) → ReLU           │
                    │                                         │
                    └──────────┬──────────────┬───────────────┘
                               │              │
                    ┌──────────▼──────┐  ┌────▼────────────┐
                    │  ACTOR (Policy)  │  │  CRITIC (Value)  │
                    │                  │  │                  │
                    │  Dense(128→21)   │  │  Dense(128→1)    │
                    │  + Softmax       │  │                  │
                    │  + Action Mask   │  │                  │
                    │                  │  │                  │
                    │  Output: 21      │  │  Output: 1       │
                    │  probabilities   │  │  (state value)   │
                    └──────────────────┘  └──────────────────┘
```

**Actor (Policy Network):** Παράγει probability distribution πάνω σε 21 δυνατές ενέργειες. Η action masking μηδενίζει τις πιθανότητες φυσικά αδύνατων ενεργειών (π.χ. discharge σε άδεια μπαταρία).

**Critic (Value Network):** Εκτιμά πόσο "καλή" είναι η τρέχουσα κατάσταση (expected future reward). Χρησιμοποιείται εσωτερικά από τον PPO αλγόριθμο για υπολογισμό advantage — δεν παράγει output προς τον χρήστη.

### 2.2 PPO Hyperparameters

| Παράμετρος | Τιμή | Εξήγηση |
|-----------|-------|---------|
| `learning_rate` | 3×10⁻⁴ | Ρυθμός μάθησης για gradient descent |
| `n_steps` | 2048 | Steps ανά environment πριν κάθε update |
| `batch_size` | 1024 | Minibatch size κατά το PPO update |
| `n_epochs` | 10 | Πόσες φορές ξαναδιαβάζει κάθε rollout |
| `gamma` (γ) | 0.995 | Discount factor — πόσο μετράει το μέλλον |
| `gae_lambda` (λ) | 0.95 | Generalized Advantage Estimation smoothing |
| `clip_range` (ε) | 0.2 | PPO clipping — αποτρέπει μεγάλα policy updates |
| `ent_coef` | 0.01 | Entropy bonus — ενθαρρύνει exploration |
| `vf_coef` | 0.5 | Value function loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `N_ENVS` | 24 | Παράλληλα environments (SubprocVecEnv) |

**Συνολικό batch ανά iteration:** 2048 × 24 = **49,152 samples**
**Minibatches ανά epoch:** 49,152 / 1024 ≈ **48 minibatches**

### 2.3 VecNormalize

Τα observations κανονικοποιούνται αυτόματα με running mean/variance:

```
normalized_obs = (obs - running_mean) / sqrt(running_var + ε)
```

| Ρύθμιση | Training | Inference |
|---------|----------|-----------|
| `norm_obs` | True | True |
| `norm_reward` | True | False |
| `clip_obs` | 100.0 | 100.0 |
| `clip_reward` | 100.0 | — |
| `training` | True (stats update) | False (frozen stats) |

---

## 3. Observation Space: 44 Features (Input στο Neural Network)

### 3.1 Πλήρης Λίστα Features

#### Battery State (Features 1-3)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 1 | `soc` | [0, 1] | Battery telemetry | Τρέχουσα φόρτιση / χωρητικότητα |
| 2 | `max_discharge_mw` | [0, 1] | Υπολογισμένο | Μέγιστη δυνατή αποφόρτιση MW / 30 |
| 3 | `max_charge_mw` | [0, 1] | Υπολογισμένο | Μέγιστη δυνατή φόρτιση MW / 30 |

#### Market Prices (Features 4-5)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 4 | `best_bid` | [0, 1] | Long price ή synthetic | Τιμή πώλησης / 100 |
| 5 | `best_ask` | [0, 1] | Short price ή synthetic | Τιμή αγοράς / 100 |

#### DAM & Time (Features 6-8)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 6 | `dam_commitment` | [-1, 1] | CSV `dam_commitment` | Υποχρέωση DAM MW / 30 |
| 7 | `hour_sin` | [-1, 1] | Timestamp | sin(2π × hour / 24) |
| 8 | `hour_cos` | [-1, 1] | Timestamp | cos(2π × hour / 24) |

#### Price Lookahead (Features 9-20)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 9-20 | `price_lookahead[0:12]` | [0, ∞] | **ML Forecaster** | Πρόβλεψη τιμών 1-12h ahead / 100 |

Αυτά τα features παράγονται από το **2ο ML μοντέλο** (LightGBM Forecaster, βλ. Ενότητα 5).

#### DAM Lookahead (Features 21-32)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 21-32 | `dam_lookahead[0:12]` | [-1, 1] | CSV μελλοντικές σειρές | DAM obligations 1-12h ahead / 30 |

Οι DAM commitments είναι γνωστές εκ των προτέρων (κλειδώνονται στο Day-Ahead Market), οπότε δεν αποτελούν data leakage.

#### Market Intelligence (Features 33-36)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 33 | `gradient` | [-∞, ∞] | Rolling window | (price_now − price_2h_ago) / 500 |
| 34 | `res_deviation` | [-∞, ∞] | Solar + Wind | (current_RES − recent_avg) / 200 |
| 35 | `imbalance_risk` | [0, 1] | Composite | Βλ. υπολογισμό παρακάτω |
| 36 | `shortfall_risk` | [0, 1] | SoC vs DAM obligations | Βλ. υπολογισμό παρακάτω |

#### Balancing Market (Features 37-38)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 37 | `afrr_up_mw` | [0, 1] | CSV `aFRR_Up_MW` | Activated upward reserve / 1000 |
| 38 | `afrr_down_mw` | [0, 1] | CSV `aFRR_Down_MW` | Activated downward reserve / 200 |

#### Strategic Signals (Features 39-42)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 39 | `price_ratio` | [0, 1] | Forecaster | max(future_prices) / current_price / 3 |
| 40 | `hours_to_peak` | [0, 1] | Forecaster | Ώρες μέχρι max μελλοντική τιμή / 12 |
| 41 | `is_at_peak` | {0, 1} | Forecaster | 1 αν current ≥ 0.95 × max_future |
| 42 | `dam_slot_ratio` | [0, 1] | DAM + Price | Τιμή στο επόμενο DAM discharge slot / current / 2 |

#### v21 Trader-Inspired Features (Features 43-44)

| # | Feature | Range | Πηγή | Υπολογισμός |
|---|---------|-------|------|------------|
| 43 | `price_vs_typical_hour` | [-1, 1] | 30-day rolling per hour | (current − typical_this_hour) / typical |
| 44 | `trade_worthiness` | [0, 1] | 30-day daily spreads | Μέσο ημερήσιο spread / 100 |

Τα features 43-44 χρησιμοποιούν **αποκλειστικά παρελθοντικά δεδομένα** (backward-looking). Αντικατέστησαν τα αντίστοιχα features του v20 που χρησιμοποιούσαν μελλοντικά δεδομένα (data leakage).

### 3.2 Υπολογισμός `imbalance_risk`

```
risk = 0.0

1. Price gradient > 20 EUR:     risk += min(0.4, gradient / 50)
2. Peak hours (17:00-20:00):    risk += 0.3
3. High price (> 150 EUR):      risk += min(0.3, (price − 150) / 100)
4. RES underperformance:        risk += min(0.2, |res_deviation| / 200)

Final: risk = min(risk, 1.0)
```

### 3.3 Υπολογισμός `shortfall_risk`

```
available = (SoC − 0.05 − 0.05) × 146 × √0.94
upcoming_discharge = Σ max(0, DAM_commitment[t+1..t+12])  [MWh]
coverage = available / upcoming_discharge

coverage ≥ 1.5:    risk = 0.00
coverage ≥ 1.2:    risk = 0.00 → 0.15  (linear)
coverage ≥ 1.0:    risk = 0.15 → 0.40  (linear)
coverage ≥ 0.7:    risk = 0.40 → 0.75  (linear)
coverage < 0.7:    risk = 0.75 → 1.00  (linear)
```

---

## 4. Action Space: 21 Discrete Actions

Ο agent επιλέγει μία από **21 ενέργειες** σε κάθε ωριαίο βήμα:

```
Action Index:   0     1     2     3     4     5    ...  10   ...  15    16   ...  20
Action Level: -1.0  -0.9  -0.8  -0.7  -0.6  -0.5  ... 0.0  ... +0.5  +0.6  ... +1.0
Power (MW):   -30   -27   -24   -21   -18   -15   ...  0   ... +15   +18   ... +30
Meaning:      MAX                CHARGE               IDLE            DISCHARGE      MAX
              CHARGE                                                               DISCHARGE
```

### 4.1 Action Masking

Σε κάθε step, ο agent βλέπει μόνο τις **φυσικά εφικτές** ενέργειες:

```python
available_discharge = max(0, (SoC − 0.05) × 146.0) × √0.94    # MWh available
available_charge    = max(0, (0.95 − SoC) × 146.0) / √0.94    # MWh headroom

max_discharge_mw = min(30.0, available_discharge / 1.0)  # per-hour limit
max_charge_mw    = min(30.0, available_charge / 1.0)

# Mask: action allowed only if within physical limits
# Action 10 (Idle) is ALWAYS allowed
```

### 4.2 SoC Update

```
Αν discharge (power > 0):   new_SoC = SoC − (|power| / √0.94) / 146.0
Αν charge (power < 0):      new_SoC = SoC + (|power| × √0.94) / 146.0

Limits: SoC ∈ [0.05, 0.95]
```

---

## 5. Neural Network #2: Price Forecaster (LightGBM)

### 5.1 Αρχιτεκτονική

**Δεν** είναι neural network — είναι **gradient boosted decision trees** (LightGBM). Αποτελείται από **12 ξεχωριστά μοντέλα**, ένα για κάθε ορίζοντα πρόβλεψης:

```
                    ┌────────────────────────────┐
                    │     41 Input Features       │
                    │  (market data + price lags) │
                    └─────────────┬──────────────┘
                                  │
           ┌──────────┬───────────┼───────────┬──────────┐
           │          │           │           │          │
      ┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐
      │Model 1h│ │Model 2h│ │Model 3h│ │  ...   │ │Model12h│
      │ LGBMs  │ │ LGBMs  │ │ LGBMs  │ │        │ │ LGBMs  │
      │ 500est │ │ 500est │ │ 500est │ │        │ │ 800est │
      │depth=6 │ │depth=6 │ │depth=6 │ │        │ │depth=7 │
      └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
          │          │           │          │          │
          ▼          ▼           ▼          ▼          ▼
     price_1h   price_2h    price_3h   ...       price_12h
                    (EUR/MWh predictions)
```

### 5.2 Input Features (41)

| Κατηγορία | Features | Πηγή |
|-----------|----------|------|
| Time encoding | hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos, hour_of_day | Timestamp |
| RES | solar, wind_onshore, res_total | CSV |
| aFRR volumes | afrr_up_mw, afrr_down_mw, afrr_mw_ratio | CSV |
| aFRR prices | afrr_up_price, afrr_down_price | CSV |
| mFRR | mfrr_up, mfrr_down, mfrr_spread | CSV |
| Imbalance | long, short, imbalance_spread | CSV |
| Current price | price_norm | CSV |
| Price lags | lag_1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h | Υπολογισμένο |
| Rolling stats | mean_6h, mean_12h, mean_24h, std_6h, std_12h, std_24h | Υπολογισμένο |
| Momentum | price_change_1h, price_change_3h | Υπολογισμένο |
| Relative | price_vs_mean_24h, price_percentile_24h | Υπολογισμένο |
| RES deviation | res_vs_mean | Υπολογισμένο |

### 5.3 Output

12 predicted prices (EUR/MWh), ένα ανά ώρα μπροστά. Κατά το training, προστίθεται noise:

```
error_pct = 0.08 + (0.20 − 0.08) × (horizon − 1) / 11
noise = Normal(0, |prediction| × error_pct)

Ορίζοντας 1h:  ±8% error
Ορίζοντας 12h: ±20% error
```

### 5.4 Πώς χρησιμοποιείται

Η έξοδος του Forecaster τροφοδοτεί **απευθείας** τα features 9-20 του PPO Agent (price lookahead). Επίσης υπολογίζονται features 39-42 (price_ratio, hours_to_peak, is_at_peak, dam_slot_ratio) από αυτές τις προβλέψεις.

---

## 6. Reward Function: Η Εξίσωση

### 6.1 Συνολική Εξίσωση

```
                    net_profit + violation_penalty
                  − proactive_risk_penalty
                  + soc_buffer_bonus
reward_raw =      + price_timing_bonus
                  + lookahead_timing_bonus
                  + quartile_adjustment          (disabled στο v21)

reward = reward_raw / 100.0                      (scaling for neural network)
```

### 6.2 Αναλυτικοί Υπολογισμοί

#### A. Net Profit (EUR)

```
net_profit = dam_revenue + balancing_revenue − degradation − shortfall_penalty
```

**DAM Revenue:**
```
dam_revenue = dam_commitment × dam_price
```

**Balancing Revenue (imbalance settlement):**
```
imbalance = actual_power − dam_commitment

Αν imbalance > 0 (over-delivery):    balancing = imbalance × best_bid
Αν imbalance < 0 (under-delivery):   balancing = imbalance × best_ask
Αν imbalance = 0:                    balancing = 0
```

**Degradation Cost:**
```
degradation = |actual_power| × 0.5 EUR/MWh
```

**Shortfall Penalty (αστοχία DAM commitment):**
```
Αν dam > 0 ΚΑΙ actual < dam:  (δεν παρέδωσε αρκετή ενέργεια)
    shortfall = (dam − actual) × (best_ask × 25 + 800)

Αν dam < 0 ΚΑΙ actual > dam:  (δεν απορρόφησε αρκετή ενέργεια)
    shortfall = (actual − dam) × (best_ask × 25 + 800)
```

#### B. Violation Penalty

```
Αν |requested − actual| > 0.01:    penalty = −5.0
Αλλιώς:                            penalty = 0.0
```

#### C. Proactive Risk Penalty

```
threshold = 0.20

Αν shortfall_risk > 0.20:
    risk_factor = (shortfall_risk − 0.20) / 0.80
    penalty = 50.0 × risk_factor^1.5
Αλλιώς:
    penalty = 0.0
```

#### D. SoC Buffer Bonus

```
Αν SoC ≥ 0.25:
    buffer = (SoC − 0.25) / 0.75
    bonus = 2.0 × buffer

Αν SoC < 0.25:
    deficit = (0.25 − SoC) / 0.25
    bonus = −2.0 × deficit × 2.0    (penalty)
```

#### E. Price Timing Bonus

```
deviation = (price − mean_24h) / mean_24h

Αν discharge (power > 0.1):  bonus = |power| × deviation × 0.1 × 100
Αν charge (power < −0.1):    bonus = |power| × (−deviation) × 0.1 × 100
Αλλιώς:                      bonus = 0.0
```

Ο agent επιβραβεύεται όταν πουλάει πάνω από τον μέσο όρο ή αγοράζει κάτω.

#### F. Lookahead Timing Bonus

```
price_ratio = max_future_price / current_price

CASE A: Πουλάει κοντά στο peak (current ≥ 0.95 × max_future)
    bonus = 10.0 × (power / 30)

CASE B: Πουλάει ενώ υπάρχει upside > 30% (ratio > 1.3, χωρίς DAM)
    upside = ratio − 1.0
    penalty = −15.0 × upside × (power / 30)
```

### 6.3 Reward Config (v21)

```python
{
    'deg_cost_per_mwh':             0.5,    # EUR/MWh degradation cost
    'violation_penalty':            5.0,    # EUR penalty for constraint breach
    'imbalance_penalty_multiplier': 25.0,   # ×best_ask for DAM shortfall
    'fixed_shortfall_penalty':      800.0,  # Fixed EUR per MWh shortfall
    'proactive_risk_penalty':       50.0,   # Max penalty for delivery risk
    'soc_buffer_bonus':             2.0,    # Bonus for maintaining SoC ≥ 25%
    'min_soc_threshold':            0.25,   # SoC target for buffer bonus
    'price_timing_bonus':           0.1,    # Scaling for price deviation bonus
    'peak_sell_bonus':              10.0,   # Bonus for selling at peak
    'early_sell_penalty':           15.0,   # Penalty for selling before peak
    'early_sell_threshold':         1.3,    # Price ratio trigger (30% upside)
    'quartile_*':                   0.0,    # ALL disabled in v21
}
```

---

## 7. Data Pipeline

### 7.1 Δεδομένα Εισόδου

**Αρχείο:** `data/feasible_data_with_balancing.csv`

| Column | Τύπος | Μονάδα | Πηγή |
|--------|-------|--------|------|
| `price` | float | EUR/MWh | HEnEx IntraDay Spot |
| `solar` | float | MW | ENTSO-E Transparency |
| `wind_onshore` | float | MW | ENTSO-E Transparency |
| `Long` | float | EUR/MWh | Imbalance Settlement (surplus) |
| `Short` | float | EUR/MWh | Imbalance Settlement (deficit) |
| `dam_commitment` | float | MW | Day-Ahead Market |
| `hour_sin/cos` | float | — | Time encoding |
| `day_sin/cos` | float | — | Time encoding |
| `month_sin/cos` | float | — | Time encoding |
| `mFRR_Up/Down` | float | EUR/MWh | Manual reserves |
| `aFRR_Up/Down_MW` | float | MW | Automatic reserve volumes |
| `aFRR_Up/Down_Price` | float | EUR/MWh | Automatic reserve prices |

### 7.2 Data Split

```
├── Train (70%): 36,428 rows ─── Εκπαίδευση agent
├── Val (15%):    7,806 rows ─── Early stopping & model selection
└── Test (15%):   7,806 rows ─── Τελική αξιολόγηση (held out)
```

---

## 8. Ροή Δεδομένων (End-to-End)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                 │
│                                                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐ │
│  │   CSV Data   │───►│  BatteryEnvMasked │───►│  FeatureEngineer   │ │
│  │ (52,040 rows)│    │  (Gymnasium Env)  │    │  (44 features)     │ │
│  └─────────────┘    └────────┬─────────┘    └─────────┬───────────┘ │
│                              │                        │              │
│                     ┌────────▼─────────┐     ┌───────▼────────┐    │
│                     │  LightGBM        │     │  VecNormalize   │    │
│                     │  Forecaster      │────►│  (obs scaling)  │    │
│                     │  (12 models)     │     └───────┬────────┘    │
│                     │  price_1h..12h   │             │              │
│                     └──────────────────┘    ┌───────▼────────┐    │
│                                             │  PPO Agent      │    │
│                                             │  (MLP 256×256   │    │
│         ┌───────────────────────────────────│   ×128)         │    │
│         │                                   │                 │    │
│         │  action (0-20)                    │  Actor: 21 acts │    │
│         │                                   │  Critic: V(s)   │    │
│         │                                   └─────────────────┘    │
│         ▼                                                          │
│  ┌──────────────────┐    ┌────────────────────┐                   │
│  │  Action Masking   │───►│  Environment Step   │                   │
│  │  (SoC constraints)│    │  power → SoC update │                   │
│  └──────────────────┘    │  → market settlement│                   │
│                          │  → reward calc      │                   │
│                          └────────┬───────────┘                   │
│                                   │                                │
│                          ┌────────▼───────────┐                   │
│                          │  RewardCalculator   │                   │
│                          │  (reward_raw / 100) │                   │
│                          └────────┬───────────┘                   │
│                                   │                                │
│                                   ▼                                │
│                          PPO Policy Update                         │
│                          (gradient descent)                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. Battery Parameters

| Παράμετρος | Τιμή | Μονάδα |
|-----------|-------|--------|
| Χωρητικότητα | 146.0 | MWh |
| Max Power (charge/discharge) | 30.0 | MW |
| Round-trip Efficiency | 0.94 | — |
| Duration (full power) | ~5 | hours |
| SoC Operating Range | 5% — 95% | — |
| Usable Capacity | 131.4 | MWh |
| Per-direction Efficiency | √0.94 ≈ 0.9695 | — |

---

## 10. Training Infrastructure

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q |
| CPU Cores | 52 |
| Parallel Envs | 24 (SubprocVecEnv) |
| FPS | ~165 |
| Total Timesteps | 5,000,000 |
| Steps per Iteration | 49,152 (2048 × 24 envs) |
| Validation | Κάθε ~5 iterations (eval_freq=10000) |
| Early Stopping | Patience = 30 validations χωρίς βελτίωση |

---

*Τελευταία ενημέρωση: v21 (Backward Stats + Trader-Inspired Signals)*
