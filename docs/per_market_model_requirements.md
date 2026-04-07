# Προδιαγραφές Μοντέλων ανά Αγορά — Per-Market Price Forecasting + Optimizer

## Μεταδεδομένα

| Πεδίο | Τιμή |
|-------|------|
| **Έκδοση** | 2.0.0 |
| **Ημερομηνία** | 2026-04-04 |
| **Σύστημα** | EntsoDRL — 30MW / 146MWh BESS, Ελληνική αγορά (HEnEx) |
| **Αρχιτεκτονική** | Per-Market Price Forecasting + Common Optimizer + SAC RL Agent |
| **Αρχή** | Κάθε μοντέλο προβλέπει **ΤΙΜΕΣ** → ο **ίδιος optimizer** μετατρέπει σε schedule |
| **Σύνολο μοντέλων** | 4 LightGBM (1 ανά market) + 1 Optimizer + 1 SAC Agent |

---

## 1. Αρχιτεκτονική — Information Cascade

Κάθε αγορά κλείνει σε διαφορετικό χρόνο. Κάθε μεταγενέστερη αγορά έχει πρόσβαση σε **περισσότερη πληροφορία** από την προηγούμενη.

**Αρχή σχεδίασης**: Κάθε μοντέλο προβλέπει **ΤΙΜΕΣ** (EUR/MWh). Ένας κοινός optimizer μετατρέπει τιμές → schedule. Αυτό απλοποιεί: 84 μοντέλα αντί 336, 1 optimizer αντί 4.

```
Χρονολογία              Μοντέλο              Output              Optimizer
────────────────────────────────────────────────────────────────────────────
D-1 12:00  ──→  DAM Forecaster      ──→  24h τιμές     ──→  DAM Schedule
                24 LightGBM, 42 features                     [96 slots]
                ΔΕΝ βλέπει: Serbia D+1, GR DAM results

D-1 15:00  ──→  IDA1 Forecaster     ──→  24h τιμές     ──→  IDA1 Delta
                24 LightGBM, 58 features                     [96 slots]
                ΝΕΟ: GR DAM results, Serbia D+1 hourly

D-1 22:00  ──→  IDA2 Forecaster     ──→  24h τιμές     ──→  IDA2 Delta
                24 LightGBM, 72 features                     [96 slots]
                ΝΕΟ: ISP1, IDA1 results, evening actuals

D+0 10:00  ──→  IDA3 Forecaster     ──→  12h τιμές     ──→  IDA3 Delta
                12 LightGBM, 86 features                     [48 slots]
                ΝΕΟ: Morning actuals, ISP2, πραγματικό SoC

Κάθε 15'   ──→  SAC Agent           ──→  aFRR + XBID [real-time]
                1 neural network, 20 features
```

### Cascade Εκτέλεσης

```
Σε κάθε gate closure:
  1. Forecaster προβλέπει ΤΙΜΕΣ (με νεότερα δεδομένα)
  2. Optimizer υπολογίζει ΒΕΛΤΙΣΤΟ schedule (δεδομένων τιμών + SoC)
  3. Delta = νέο_schedule - ήδη_κλειδωμένο
  4. SoC feasibility validation
  5. Lock delta → mandatory execution

cumulative_schedule[t] = DAM[t] + IDA1_delta[t] + IDA2_delta[t] + IDA3_delta[t]

Constraints:
  - SoC μένει [5%, 95%] σε κάθε slot
  - Ισχύς ≤ 30 MW σε κάθε slot (σωρευτικά)
  - Round-trip efficiency 94% (sqrt per direction)
```

---

## 2. Μοντέλο #1: DAM Schedule Forecaster

### 2.1 Ταυτότητα

| Πεδίο | Τιμή |
|-------|------|
| **Σκοπός** | Πρόβλεψη 24 ωριαίων τιμών DAM D+1 → optimizer δημιουργεί schedule |
| **Gate Closure** | D-1 12:00 CET |
| **Decision Deadline** | D-1 11:30 (30' buffer) |
| **Output** | `EUR/MWh[24]` — ωριαίες τιμές, εύρος [0, 500] |
| **Μετά-επεξεργασία** | Optimizer μετατρέπει τιμές → `MW[96]` schedule (15-min) |
| **Δέσμευση** | Υποχρεωτική εκτέλεση μετά τη δημοπρασία |
| **Αλγόριθμος** | LightGBM, 1 μοντέλο (hour ως input feature) |

### 2.2 Input Features — 42 features

**ΚΡΙΣΙΜΟ**: Μόνο δεδομένα διαθέσιμα **πριν** τις 12:00 D-1.

| # | Ομάδα | Feature | Columns στο Dataset | Σημειώσεις |
|---|-------|---------|---------------------|------------|
| 1-4 | **GR DAM Lags** | `dam_lag_1d`, `_2d`, `_3d`, `_7d` | Derived: `price.shift(96/192/288/672)` | Ίδιο slot πριν 1/2/3/7 ημέρες |
| 5-8 | **Rolling Stats** | `dam_mean_24h`, `_std_24h`, `_min_24h`, `_max_24h` | `price_mean_24h`, `price_std_24h`, `price_min_24h`, `price_max_24h` | Backward-looking 24h rolling |
| 9 | **Volatility** | `price_std_6h` | `price_std_6h` | Πρόσφατη μεταβλητότητα |
| 10-15 | **Calendar** | `period_sin/cos`, `dow_sin/cos`, `month_sin/cos` | Derived: `hour`, `day_of_week`, `month` | Cyclical encoding |
| 16-17 | **Peak Indicators** | `is_peak_hour`, `is_solar_hour` | Direct columns | 17:00-21:00 peak, 09:00-15:00 solar |
| 18-19 | **Load/RES Forecast** | `load_forecast_norm`, `res_forecast_norm` | `load_mw` (D+1 forecast), `res_total_mw` | TSO πρόβλεψη D+1, normalized |
| 20-23 | **Weather D+1** | `temperature`, `wind_100m`, `solar_radiation`, `cloud_cover` | Open-Meteo forecast | Πρόβλεψη καιρού D+1 |
| 24-25 | **Commodities** | `ttf_gas`, `eua_carbon` | `ttf_gas_price`, CO2 ticker | Marginal cost drivers |
| 26-27 | **Gas Momentum** | `ttf_gas_change_24h`, `_7d` | Direct columns | Τάση τιμών gas |
| 28-31 | **Neighbor DAM D-0** | `dam_bg`, `_ro`, `_hu`, `_it_south` | `dam_price_bg/ro/hu/it_south` (ημερήσιες) | Γνωστές D-0 τιμές γειτόνων |
| 32-35 | **Neighbor D+1** | `dam_bg_d1`, `_ro_d1`, `_hu_d1`, `_it_south_d1` | `dam_price_bg_d1/...` | Αν δημοσιευμένα πριν 12:00 |
| 36 | **RS-GR Spread** | `rs_gr_spread` | `serbia_dam_price - price` | Cross-border signal (D-0 μόνο) |
| 37-40 | **Serbia D-0 Stats** | `rs_d0_mean`, `_max`, `_min`, `_spread` | Aggregated `serbia_dam_price` | ΟΧΙ D+1 — δημοσιεύεται 12:50! |
| 41-42 | **Forecast Errors** | `load_fcst_error_lag`, `res_fcst_error_lag` | Derived | Bias correction — πόσο πήγε λάθος χθες |

**LOOK-AHEAD CHECK**:
- ❌ Serbia D+1 hourly (`serbia_d1_h00..h23`): Δημοσιεύονται ~12:50 → **ΜΕΤΑ** τη gate 12:00
- ❌ GR DAM D+1 results: Δημοσιεύονται ~14:00 → δεν υπάρχουν ακόμα
- ❌ ISP1/2/3 prices: Μελλοντικά δεδομένα
- ✅ Neighbor D-0 prices: Γνωστές (χθεσινές)
- ✅ Neighbor D+1 (BG, RO, HU, IT): Μερικές δημοσιεύονται πριν τις 12:00

### 2.3 Training Target

**Actual DAM prices** (supervised learning):

```
Target: actual_price[h]  για h = 0..23
Loss: Huber loss (ανθεκτικό σε extreme τιμές)
```

Κάθε ώρα γίνεται ένα regression target: `features[42] → EUR/MWh[1]`

Μετά, ο **Optimizer** παίρνει τις 24 predicted τιμές και βγάζει schedule:
```
maximize: Σ(predicted_price[t] × power[t] × 0.25h)  for t = 0..95
subject to:
  -30 ≤ power[t] ≤ 30 MW                              ∀t
  0.05 ≤ SoC[t] ≤ 0.95                                ∀t
  SoC[t+1] = SoC[t] ± power[t]×0.25 / (146×√0.94)
  SoC[0] = current_soc
```

### 2.4 Μετρική Αξιολόγησης

```
Forecaster: MAE (EUR/MWh) — στόχος ≤ 15 EUR (comparable to EntsoE3)
Pipeline:   capture_ratio = realized_pnl / oracle_pnl — στόχος ≥ 0.80
```

---

## 3. Μοντέλο #2: IDA1 Schedule Corrector

### 3.1 Ταυτότητα

| Πεδίο | Τιμή |
|-------|------|
| **Σκοπός** | Πρόβλεψη ενημερωμένων τιμών → optimizer υπολογίζει delta vs DAM |
| **Gate Closure** | D-1 15:00 CET |
| **Decision Deadline** | D-1 14:45 |
| **Output** | `EUR/MWh[24]` — ενημερωμένες ωριαίες τιμές |
| **Μετά-επεξεργασία** | Optimizer → νέο schedule → delta = νέο - DAM_locked |
| **Δέσμευση** | Υποχρεωτική εκτέλεση μετά το gate closure |
| **Αλγόριθμος** | LightGBM, 1 μοντέλο (hour ως input feature) |

### 3.2 ΝΕΑ Πληροφορία vs DAM

| # | Νέο Δεδομένο | Ώρα Δημοσίευσης | Γιατί Σημαντικό |
|---|-------------|-----------------|-----------------|
| 1 | **GR DAM D+1 Τιμές** (96 τιμές) | HEnEx ~14:00 | Οι ΠΡΑΓΜΑΤΙΚΕΣ τιμές DAM — η πιο κρίσιμη πληροφορία |
| 2 | **Serbia D+1 Hourly** (24 τιμές) | ENTSO-E ~12:50 | `serbia_d1_h00..h23` — τώρα πλέον διαθέσιμα |
| 3 | **Updated Weather Forecast** | Open-Meteo | 2-3h πιο πρόσφατη πρόβλεψη |
| 4 | **Updated Load Forecast** | ENTSO-E | Αναθεωρημένη πρόβλεψη φορτίου |

### 3.3 Input Features — 58 features

| # | Ομάδα | Features | Πλήθος | Σημειώσεις |
|---|-------|----------|--------|------------|
| 1-42 | **Βάση DAM** | (ίδια με Section 2.2) | 42 | Πλήρες baseline context |
| 43-46 | **GR DAM Actual D+1** | `dam_actual_mean`, `_max`, `_min`, `_spread` | 4 | **ΝΕΟ**: Stats 96 actual DAM prices D+1 |
| 47-50 | **Locked DAM Schedule** | `sched_mean_mw`, `sched_buy_mwh`, `sched_sell_mwh`, `sched_net_mwh` | 4 | **ΝΕΟ**: Χαρακτηριστικά locked DAM schedule |
| 51-54 | **Serbia D+1 Hourly** | `rs_d1_mean`, `_max`, `_min`, `_corr_gr` | 4 | **ΝΕΟ**: `serbia_d1_h00..h23` — τώρα γνωστά |
| 55-56 | **Price Deviation** | `dam_actual_vs_forecast`, `dam_actual_vs_lag1d` | 2 | **ΝΕΟ**: Πόσο απέχουν actual από αναμενόμενα |
| 57-58 | **Schedule Error** | `sched_overcharged`, `sched_underdischarged` | 2 | **ΝΕΟ**: Αν DAM αγόρασε σε ακριβές ώρες |

### 3.4 Training Target

```
Target: actual_price[h]  για h = 0..23  (ίδιες πραγματικές τιμές με DAM)
```

Τα IDA1 μοντέλα μαθαίνουν: `features[58] → EUR/MWh[1]`

Η βελτίωση προέρχεται από τα **νέα features** (GR DAM actual, Serbia D+1) — ο forecaster πρέπει να κάνει μικρότερο MAE από τον DAM forecaster.

Μετά: Optimizer(νέες τιμές) → νέο schedule → delta = νέο - DAM_locked

### 3.5 SoC Validation

Μετά τον optimizer: `cumulative = DAM + IDA1_delta` → SoC trajectory check [5%, 95%]. Αν παραβιάζει, clip to feasible.

---

## 4. Μοντέλο #3: IDA2 Schedule Corrector

### 4.1 Ταυτότητα

| Πεδίο | Τιμή |
|-------|------|
| **Σκοπός** | Πρόβλεψη ενημερωμένων τιμών → optimizer υπολογίζει delta vs cumulative |
| **Gate Closure** | D-1 22:00 CET |
| **Decision Deadline** | D-1 21:45 |
| **Output** | `EUR/MWh[24]` — ενημερωμένες ωριαίες τιμές |
| **Μετά-επεξεργασία** | Optimizer → νέο schedule → delta = νέο - (DAM+IDA1) |
| **Αλγόριθμος** | LightGBM, 1 μοντέλο (hour ως input feature) |

### 4.2 ΝΕΑ Πληροφορία vs IDA1

| # | Νέο Δεδομένο | Ώρα | Column στο Dataset |
|---|-------------|-----|---------------------|
| 1 | **ISP1 Clearing** | ~18:00 | `isp1_price_up`, `isp1_price_down` |
| 2 | **IDA1 Results** | ~15:30 | `ida1_clearing_price`, `ida1_position` |
| 3 | **XBID Activity** | 15:00-22:00 | `xbid_price_bid`, `xbid_price_ask`, `xbid_spread` |
| 4 | **Evening Demand** | 18:00-22:00 | `load_mw` (actual evening) |
| 5 | **RES Actual vs Forecast** | Real-time | `solar`, `wind_onshore` vs forecasts |
| 6 | **Imbalance Data** | Real-time | `net_imbalance_mw`, `imbalance_price` |

### 4.3 Input Features — 72 features

| # | Ομάδα | Features | Πλήθος | Σημειώσεις |
|---|-------|----------|--------|------------|
| 1-58 | **Βάση IDA1** | (ίδια με Section 3.3) | 58 | Full IDA1 cascade |
| 59-61 | **ISP1 Signal** | `isp1_price_up`, `_down`, `_spread` | 3 | **ΝΕΟ**: Πρώτο balancing signal |
| 62-63 | **IDA1 Results** | `ida1_clearing_price`, `ida1_position` | 2 | **ΝΕΟ**: Τιμή/θέση IDA1 |
| 64-66 | **XBID Signal** | `xbid_bid`, `xbid_ask`, `xbid_spread` | 3 | **ΝΕΟ**: Continuous market activity |
| 67-68 | **RES Deviation** | `wind_actual_deviation`, `solar_actual_deviation` | 2 | **ΝΕΟ**: Actual vs forecast |
| 69-70 | **Evening Load** | `evening_load_actual`, `evening_load_deviation` | 2 | **ΝΕΟ**: Πραγματική βραδινή ζήτηση |
| 71-72 | **Imbalance** | `net_imbalance_mw`, `imbalance_price` | 2 | **ΝΕΟ**: Σύστημα deviation |

### 4.4 Training Target

```
Target: actual_price[h]  για h = 0..23
```

Βελτίωση μέσω ISP1, evening actuals → μικρότερο MAE από IDA1 forecaster.

---

## 5. Μοντέλο #4: IDA3 Schedule Corrector

### 5.1 Ταυτότητα

| Πεδίο | Τιμή |
|-------|------|
| **Σκοπός** | Πρόβλεψη τιμών 12:00-24:00 → optimizer υπολογίζει τελικό delta |
| **Gate Closure** | D+0 10:00 CET |
| **Decision Deadline** | D+0 09:45 |
| **Output** | `EUR/MWh[12]` — ωριαίες τιμές μόνο 12:00-24:00 |
| **Μετά-επεξεργασία** | Optimizer → schedule[48:96] → delta = νέο - cumulative |
| **ΚΡΙΣΙΜΟ** | Slots 0-47 (00:00-12:00) ΗΔΗ εκτελούνται — δεν τα αγγίζουμε |
| **Αλγόριθμος** | LightGBM, 1 μοντέλο (hour ως feature, μόνο h=12..23) |

### 5.2 ΝΕΑ Πληροφορία vs IDA2

| # | Νέο Δεδομένο | Ώρα | Column στο Dataset |
|---|-------------|-----|---------------------|
| 1 | **Morning Actual Prices** | 00:00-10:00 | `price` (πρωινά actual slots) |
| 2 | **Morning RES** | 06:00-10:00 | `solar`, `wind_onshore` (πρωινά actual) |
| 3 | **ISP2 Results** | ~07:00 | `isp2_price_up`, `isp2_price_down` |
| 4 | **IDA2 Results** | ~22:30 D-1 | `ida2_clearing_price`, `ida2_position` |
| 5 | **Overnight System** | 00:00-06:00 | `net_imbalance_mw`, `load_mw` overnight |
| 6 | **Near-RT Weather** | 09:00 | Open-Meteo (<12h horizon) |
| 7 | **Πραγματικό SoC** | Real-time | SCADA — η πιο ακριβής πληροφορία |

### 5.3 Input Features — 86 features

| # | Ομάδα | Features | Πλήθος | Σημειώσεις |
|---|-------|----------|--------|------------|
| 1-72 | **Βάση IDA2** | (ίδια με Section 4.3) | 72 | Full IDA2 cascade |
| 73-76 | **Morning Actual** | `morning_price_mean`, `_max`, `_min`, `_trend` | 4 | **ΝΕΟ**: Actual prices slots 0-39 |
| 77-78 | **ISP2 Signal** | `isp2_price_up`, `isp2_price_down` | 2 | **ΝΕΟ**: Δεύτερο balancing signal |
| 79-80 | **IDA2 Results** | `ida2_clearing_price`, `ida2_position` | 2 | **ΝΕΟ**: Τιμή/θέση IDA2 |
| 81-82 | **Morning RES** | `morning_solar_actual`, `morning_wind_actual` | 2 | **ΝΕΟ**: Πραγματική πρωινή παραγωγή |
| 83 | **Πραγματικό SoC** | `current_soc` | 1 | **ΝΕΟ**: Actual SoC στις 10:00 |
| 84-85 | **Execution Deviation** | `morning_sched_deviation_mwh`, `morning_soc_vs_expected` | 2 | **ΝΕΟ**: Απόκλιση πρωινής εκτέλεσης |
| 86 | **Overnight Imbalance** | `overnight_avg_imbalance` | 1 | **ΝΕΟ**: Μέσο imbalance 00:00-06:00 |

### 5.4 Training Target

```
Target: actual_price[h]  για h = 12..23  (12 ώρες μόνο)
```

Βελτίωση μέσω morning actuals, ISP2 → η πιο ακριβής πρόβλεψη (πλησιέστερα σε real-time).

---

## 6. SAC Agent — aFRR + XBID Only

### 6.1 Ταυτότητα

| Πεδίο | Τιμή |
|-------|------|
| **Σκοπός** | Real-time αποφάσεις αποκλειστικά aFRR + XBID |
| **Συχνότητα** | Κάθε 15 λεπτά |
| **Αλγόριθμος** | SAC (Soft Actor-Critic) |
| **Action Space** | 4D continuous [-1, +1] |
| **ΔΕΝ ελέγχει** | DAM, IDA1/2/3, mFRR (αυτόματα/κλειδωμένα) |

### 6.2 Action Space — 4 dimensions

| # | Action | Εύρος | Σημειώσεις |
|---|--------|-------|------------|
| 0 | aFRR Commitment MW | 0..30 | action<0 → 0 MW (πρέπει ενεργά να επιλέξει) |
| 1 | aFRR Price Tier | 0.7..1.3 | Πολλαπλασιαστής reference price |
| 2 | XBID Quantity MW | -30..+30 | Αρνητικό = αγορά, Θετικό = πώληση |
| 3 | XBID Price Offset | -50..+50 EUR | Απόσταση από mid-price |

### 6.3 Observation Space — 20 features

| # | Feature | Πηγή | Normalization |
|---|---------|------|---------------|
| 1 | `current_soc` | Battery SCADA | /1.0 |
| 2 | `current_power_mw` | Inverter | /30.0 |
| 3-6 | `locked_schedule_next_1h` (4 slots) | Cascade result | /30.0 |
| 7-10 | `locked_schedule_4h_stats` (mean, max, min, net) | Cascade result | /30.0 |
| 11 | `remaining_discharge_mwh` | Computed: (SoC - 0.05) × 146 | /146.0 |
| 12 | `remaining_charge_mwh` | Computed: (0.95 - SoC) × 146 | /146.0 |
| 13-14 | `afrr_cap_up_price`, `afrr_cap_down_price` | IPTO | /200.0 |
| 15-16 | `xbid_bid`, `xbid_ask` | XBID platform | /500.0 |
| 17 | `xbid_spread` | Computed: ask - bid | /100.0 |
| 18-19 | `hour_sin`, `hour_cos` | Clock | [-1, 1] |
| 20 | `net_imbalance_mw` | ADMIE | /500.0 |

### 6.4 Αλληλεπίδραση με Schedules

Ο SAC agent **δεν τροποποιεί** τα locked schedules. Η διαθέσιμη χωρητικότητα:

```
available_capacity = max_power_mw - abs(cumulative_schedule[current_slot])
```

Ο agent μπορεί να χρησιμοποιήσει μόνο τη **μη δεσμευμένη** χωρητικότητα.

---

## 7. Feature Availability Matrix

| Feature Category | DAM (12:00) | IDA1 (15:00) | IDA2 (22:00) | IDA3 (10:00) | SAC (RT) |
|-----------------|:-----------:|:------------:|:------------:|:------------:|:--------:|
| GR DAM Lags D-1/2/3/7 | ✅ | ✅ | ✅ | ✅ | ✅ |
| GR DAM Rolling 24h | ✅ | ✅ | ✅ | ✅ | ✅ |
| Weather Forecast D+1 | ✅ | ✅ (updated) | ✅ (updated) | ✅ (near-RT) | — |
| TTF Gas / Carbon | ✅ | ✅ | ✅ | ✅ | — |
| Neighbor DAM D-0 | ✅ | ✅ | ✅ | ✅ | — |
| Neighbor DAM D+1 | Μερικά | ✅ | ✅ | ✅ | — |
| **Serbia D+1 Hourly** | **❌ (12:50!)** | **✅ ΝΕΟ** | ✅ | ✅ | — |
| **GR DAM D+1 Actual** | **❌ (14:00!)** | **✅ ΝΕΟ** | ✅ | ✅ | — |
| **ISP1 Clearing** | ❌ | ❌ | **✅ ΝΕΟ** | ✅ | ✅ |
| **IDA1 Results** | ❌ | ❌ | **✅ ΝΕΟ** | ✅ | — |
| **XBID Activity** | ❌ | ❌ | **✅ ΝΕΟ** | ✅ | ✅ |
| **Evening Load Actual** | ❌ | ❌ | **✅ ΝΕΟ** | ✅ | — |
| **ISP2 Results** | ❌ | ❌ | ❌ | **✅ ΝΕΟ** | ✅ |
| **IDA2 Results** | ❌ | ❌ | ❌ | **✅ ΝΕΟ** | — |
| **Morning Prices** | ❌ | ❌ | ❌ | **✅ ΝΕΟ** | ✅ |
| **Morning RES** | ❌ | ❌ | ❌ | **✅ ΝΕΟ** | — |
| **Πραγματικό SoC** | ❌ | ❌ | ❌ | **✅ ΝΕΟ** | ✅ |
| Locked Schedules | — | ✅ (DAM) | ✅ (DAM+1) | ✅ (DAM+1+2) | ✅ (all) |

---

## 8. Training Pipeline

### 8.1 Κοινός Optimizer (LP Solver)

Ένας optimizer χρησιμοποιείται από ΟΛΑ τα μοντέλα:

```python
# Input: predicted_prices[24], current_soc, locked_schedule (αν υπάρχει)
# Output: optimal MW[96] schedule (15-min resolution)
# Method: scipy.optimize.linprog ή greedy spread-maximization (ήδη υπάρχει)
```

Σε κάθε gate closure:
1. Forecaster → 24h predicted prices
2. Optimizer(predicted_prices, current_soc, locked_commitments) → new_schedule
3. delta = new_schedule - locked_cumulative
4. SoC validation → clip infeasible
5. Lock delta

### 8.2 Training Pipeline

```
Για κάθε ημέρα D στο ιστορικό (2021-01-01 ÷ 2026-01-17, ~1800 ημέρες):

  Step 1: Collect features at D-1 12:00 → X_dam[42]
          Target: actual_price[h] for h=0..23
          → Train DAM forecaster

  Step 2: Collect features at D-1 15:00 → X_ida1[58]
          (περιλαμβάνει GR DAM actual, Serbia D+1, DAM schedule stats)
          Target: actual_price[h] for h=0..23
          → Train IDA1 forecaster

  Step 3: Collect features at D-1 22:00 → X_ida2[72]
          (περιλαμβάνει ISP1, IDA1 results, evening actuals)
          Target: actual_price[h] for h=0..23
          → Train IDA2 forecaster

  Step 4: Collect features at D+0 10:00 → X_ida3[86]
          (περιλαμβάνει morning actuals, ISP2, πραγματικό SoC)
          Target: actual_price[h] for h=12..23
          → Train IDA3 forecaster
```

Κάθε forecaster εκπαιδεύεται **ανεξάρτητα** σε actual τιμές. Η βελτίωση προέρχεται ΜΟΝΟ από τα νεότερα features.

### 8.3 Model Summary

| Μοντέλο | Αλγόριθμος | # Models | # Features | Output | Post-processing |
|---------|------------|----------|------------|--------|-----------------|
| DAM Forecaster | LightGBM | **1** | 42 + hour | EUR/MWh × 24 | → Optimizer → MW[96] |
| IDA1 Forecaster | LightGBM | **1** | 58 + hour | EUR/MWh × 24 | → Optimizer → delta MW[96] |
| IDA2 Forecaster | LightGBM | **1** | 72 + hour | EUR/MWh × 24 | → Optimizer → delta MW[96] |
| IDA3 Forecaster | LightGBM | **1** | 86 + hour | EUR/MWh × 12 | → Optimizer → delta MW[48] |
| **Σύνολο LightGBM** | | **4** | | | |
| Optimizer | LP/Greedy | 1 | — | MW[96] | Κοινός για όλους |
| SAC Agent | PyTorch | 1 NN | 20 | 4D continuous | Real-time only |

---

## 9. Αξιολόγηση

### 9.1 Μετρικές (P&L-based, ΟΧΙ MAE/RMSE)

```
capture_ratio = realized_pnl / oracle_pnl

Στόχοι:
  DAM μόνο:           capture_ratio ≥ 0.80
  DAM + IDA1:         capture_ratio ≥ 0.85
  DAM + IDA1 + IDA2:  capture_ratio ≥ 0.87
  DAM + all IDAs:     capture_ratio ≥ 0.90
  DAM + IDAs + SAC:   capture_ratio ≥ 0.92

Κάθε IDA πρέπει να προσθέτει ≥ 2-3% incremental P&L
```

### 9.2 Backtesting Protocol

- Walk-forward validation: 30-day rolling window
- Out-of-sample: 2025-01-15 ÷ 2026-01-17 (1 year held-out)
- Per-day P&L tracking
- Worst-case analysis: 5th percentile daily P&L

---

## 10. Σειρά Υλοποίησης

| Phase | Task | Εκτιμώμενος Χρόνος |
|-------|------|---------------------|
| 1 | Oracle LP Schedule Generator | 2-3 ημέρες |
| 2 | DAM Schedule Forecaster (96 models) | 3-4 ημέρες |
| 3 | IDA1 Corrector (96 models) | 2-3 ημέρες |
| 4 | IDA2 Corrector (96 models) | 2-3 ημέρες |
| 5 | IDA3 Corrector (48 models) | 2 ημέρες |
| 6 | SAC Agent restructure (20 obs, 4D) | 2-3 ημέρες |
| 7 | End-to-end integration & testing | 3-4 ημέρες |
| 8 | Backtesting & evaluation | 2-3 ημέρες |
| **Σύνολο** | | **~20-25 ημέρες** |

---

## 11. Σχετικά Αρχεία Κώδικα

| Αρχείο | Σκοπός | Κατάσταση |
|--------|--------|-----------|
| `gym_envs/entsoe3_price_forecaster.py` | 24 XGBoost models, 57 features | Υπάρχει — reference |
| `PredictFuturePrices/forecasters/dam_forecaster.py` | 96×3 LightGBM, 29 features | Υπάρχει — base pattern |
| `PredictFuturePrices/forecasters/ida_forecaster.py` | 3 LightGBM, 30 features, blending | Υπάρχει — αντικαθίσταται |
| `PredictFuturePrices/forecasters/base_forecaster.py` | Base class forecasters | Υπάρχει — reuse |
| `PredictFuturePrices/config.py` | Configurations & market events | Τροποποιείται |
| `gym_envs/market_executors/dam_executor.py` | DamOptimizer, SoC capping | Υπάρχει — reuse LP logic |
| `gym_envs/market_executors/ida_executor.py` | Gate closures, schedule generation | Τροποποιείται |
| `gym_envs/battery_env_unified_sac.py` | SAC wrapper, 4D actions | Τροποποιείται (→20 obs) |
| `data/unified_multimarket_training_v5_raw_features.csv` | 189 columns, 176K rows | Input data |
