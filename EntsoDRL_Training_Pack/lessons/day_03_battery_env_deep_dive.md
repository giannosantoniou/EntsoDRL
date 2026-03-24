# Ημέρα 3: BatteryEnvMasked — Deep Dive στο Environment

> **Στόχος**: Να καταλάβεις πώς η φυσική της μπαταρίας και η αγορά ενέργειας
> κωδικοποιούνται μέσα στο `gym_envs/battery_env_masked.py`.

---

## 1. Constructor — Παράμετροι Μπαταρίας

```python
# Από gym_envs/battery_env_masked.py:
class BatteryEnvMasked(gym.Env):
    def __init__(self, df, capacity_mwh=146.0, max_discharge_mw=30.0,
                 efficiency=0.94, n_actions=21, **kwargs):
        super().__init__()

        self.df = df                          # Historical market data
        self.capacity_mwh = capacity_mwh      # 146 MWh
        self.max_discharge_mw = max_discharge_mw  # 30 MW
        self.efficiency = efficiency          # 94% round-trip
        self.n_actions = n_actions            # 21 discrete actions
```

**Τι σημαίνουν φυσικά:**

| Παράμετρος | Τιμή | Σημασία |
|-----------|-------|---------|
| `capacity_mwh` | 146 MWh | Αν η μπαταρία είναι 100% γεμάτη, χωράει 146 MWh ενέργεια |
| `max_discharge_mw` | 30 MW | Μέγιστη ισχύς εξόδου (inverter limit) |
| `efficiency` | 0.94 | Σε κάθε κύκλο χάνεται 6% ενέργεια (θερμότητα) |
| `n_actions` | 21 | Βήματα: -30MW, -27MW, ..., 0, ..., +27MW, +30MW |

**Ανάλογη σκέψη:** Η μπαταρία είναι σαν ένα δοχείο νερού (146 λίτρα), με μία βρύση που βγάζει max 30 λίτρα/ώρα, αλλά κάθε φορά που "μεταγγίζεις" χάνεις 6%.

---

## 2. Φυσική Μπαταρίας — Η Εξίσωση SoC

Η πιο σημαντική εξίσωση στο project:

```python
# Charging (αγοράζω ενέργεια, γεμίζω μπαταρία):
delta_soc = (power_mw * time_hours * sqrt(efficiency)) / capacity_mwh

# Discharging (πουλάω ενέργεια, αδειάζω μπαταρία):
delta_soc = -(power_mw * time_hours / sqrt(efficiency)) / capacity_mwh
```

**Γιατί √efficiency και όχι efficiency;**

Η efficiency 94% είναι **round-trip** (charge → store → discharge). Σπάμε σε δύο μέρη:
- Charging loss: √0.94 ≈ 96.95% (χάνεις 3.05% κατά τη φόρτιση)
- Discharging loss: √0.94 ≈ 96.95% (χάνεις 3.05% κατά την εκφόρτιση)
- Συνολικά: 0.9695 × 0.9695 ≈ 0.94 (6% συνολική απώλεια)

**Παράδειγμα:**
```
Charge 30MW για 1 ώρα:
  energy_in = 30 × 1 × √0.94 = 30 × 0.9695 = 29.09 MWh αποθηκευμένα
  ΔSoC = 29.09 / 146 = +0.199 (≈ 20% increase)

Discharge 30MW για 1 ώρα:
  energy_out = 30 × 1 / √0.94 = 30 / 0.9695 = 30.94 MWh αφαιρούνται
  ΔSoC = -30.94 / 146 = -0.212 (≈ 21% decrease)

Παρατήρηση: χάνεις περισσότερο SoC στο discharge (30.94) απ' ό,τι κερδίζεις
στο charge (29.09) — αυτό είναι η efficiency loss.
```

---

## 3. Operating Range — Γιατί 5%-95%

```python
# Στο battery_env_masked.py:
self.min_soc = 0.05    # 5% — ποτέ κάτω από αυτό
self.max_soc = 0.95    # 95% — ποτέ πάνω από αυτό
```

**Γιατί;**
- Κάτω από 5%: Η μπαταρία υφίσταται μόνιμη βλάβη (deep discharge)
- Πάνω από 95%: Κίνδυνος overcharge, θερμική διαφυγή
- Αυτοί οι περιορισμοί εφαρμόζονται μέσω **action masking** (Ημέρα 5)

---

## 4. Η Αγορά Ενέργειας μέσα στο Environment

Κάθε βήμα (step) αντιστοιχεί σε 1 ώρα στην αγορά. Τα δεδομένα φορτώνονται από CSV:

```python
# Σε κάθε step:
current_row = self.df.iloc[self.current_step]

dam_price = current_row['dam_price']           # Day-Ahead Market price
best_bid = current_row.get('best_bid', dam_price * 0.98)  # IntraDay bid
best_ask = current_row.get('best_ask', dam_price * 1.02)  # IntraDay ask
dam_commitment = current_row.get('dam_commitment', 0.0)    # MW committed
```

**Τι σημαίνει κάθε τιμή:**

```
DAM Price: Η τιμή που κλείδωσε χθες για αυτή την ώρα (Day-Ahead Market)
Best Bid: Η καλύτερη τιμή αγοράς τώρα στο IntraDay market
Best Ask: Η καλύτερη τιμή πώλησης τώρα στο IntraDay market
Spread: best_ask - best_bid (κόστος "εμπορίας")
DAM Commitment: Πόσα MW δεσμεύτηκε η μπαταρία στο DAM (χθες)
```

**Η σχέση μεταξύ τους:**
```
Χθες (D-1):
  → DAM: "Αύριο 14:00 θα πουλήσω 10MW στα 85€/MWh"
  → dam_commitment = +10 MW
  → dam_price = 85€/MWh

Σήμερα (D, real-time):
  → Η πραγματική τιμή μπορεί να είναι 92€/MWh
  → best_bid = 90€, best_ask = 94€
  → Αν πουλήσω 15MW (αντί 10MW committed):
    - 10MW × 85€ = 850€ (DAM revenue, σίγουρο)
    - 5MW × 90€ = 450€ (Balancing revenue, bid price)
    - Σύνολο: 1300€
```

---

## 5. Η Μέθοδος `step()` — Βήμα-βήμα

Ακολουθεί η λογική ροή (simplified) του `step()`:

```
step(action=3)
│
├── 1. Action → MW
│   action=3 → normalized=(3-10)/10=-0.7 → -0.7 × 30 = -21MW (discharge)
│
├── 2. Physics Update
│   SoC_old = 0.60
│   ΔSoC = -(21 × 1 / √0.94) / 146 = -0.148
│   SoC_new = 0.60 - 0.148 = 0.452
│   SoC_clamped = max(0.05, min(0.95, 0.452)) = 0.452 ✓
│
├── 3. Market Settlement
│   dam_commitment = +10MW (sell)
│   actual_physical = -21MW (discharge = sell 21MW)
│   imbalance = actual - commitment = -21 - 10 = -31 ... ← λάθος
│   (στην πραγματικότητα: imbalance = |actual_sell| - commitment)
│   extra_sold = 21 - 10 = 11MW (πούλησε 11MW extra)
│   DAM revenue: 10 × dam_price
│   Balancing revenue: 11 × best_bid (lower price for extra)
│
├── 4. Reward Calculation
│   reward = DAM_rev + Balancing_rev - Degradation - Penalties + Bonuses
│
├── 5. Advance Step
│   current_step += 1
│   terminated = (current_step >= len(df) - 1)
│
└── 6. Return
    → (observation, reward, terminated, False, info)
```

---

## 6. Time Encoding — Γιατί sin/cos

Η ώρα 23 και η ώρα 0 είναι "δίπλα" χρονικά, αλλά αν απλά βάλεις hour=23 και hour=0, το neural network δεν ξέρει ότι είναι κοντά. Λύση: **κυκλική κωδικοποίηση**:

```python
import numpy as np

hour = 14
hour_sin = np.sin(2 * np.pi * hour / 24)  # = sin(π × 14/12) ≈ -0.866
hour_cos = np.cos(2 * np.pi * hour / 24)  # = cos(π × 14/12) ≈ -0.500
```

**Γιατί δουλεύει:**

```
Hour 0:   sin=0.000, cos=1.000  ← μεσάνυχτα
Hour 6:   sin=1.000, cos=0.000  ← πρωί
Hour 12:  sin=0.000, cos=-1.000 ← μεσημέρι
Hour 18:  sin=-1.000, cos=0.000 ← βράδυ
Hour 23:  sin=-0.259, cos=0.966 ← κοντά στο 0!

Απόσταση 23↔0: √((−0.259−0)² + (0.966−1)²) = 0.26 ← μικρή!
Απόσταση 0↔12: √((0−0)² + (1−(−1))²) = 2.0 ← μεγάλη!
```

Τώρα το neural network "ξέρει" ότι 23:00 και 00:00 είναι κοντά, ενώ 00:00 και 12:00 είναι μακριά.

---

## 7. Lookahead Features — Τι ξέρει ο Agent για το μέλλον

Ο agent βλέπει τις **επόμενες 12 ώρες** τιμών και DAM commitments:

```python
# Price lookahead (12 features):
for i in range(12):
    future_step = self.current_step + i + 1
    if future_step < len(self.df):
        obs[8 + i] = self.df.iloc[future_step]['dam_price'] / 100.0
    else:
        obs[8 + i] = 0.0  # Padding αν κοντά στο τέλος

# DAM commitment lookahead (12 features):
for i in range(12):
    future_step = self.current_step + i + 1
    if future_step < len(self.df):
        obs[20 + i] = self.df.iloc[future_step].get('dam_commitment', 0.0) / 30.0
    else:
        obs[20 + i] = 0.0
```

**Γιατί 12 ώρες;**
- Η μπαταρία έχει ~5h duration (146MWh / 30MW)
- 12 ώρες δίνουν αρκετό ορίζοντα για planning
- Π.χ. "Βλέπω peak prices σε 4 ώρες → charge τώρα, discharge τότε"

**Είναι data leakage;**
- **Τιμές DAM**: ΟΧΙ — στην πραγματικότητα οι DAM τιμές ανακοινώνονται D-1 (χθες), οπότε ξέρεις τις τιμές ολόκληρης της σημερινής ημέρας
- **Backward-looking stats**: Χρησιμοποιούν ΜΟΝΟ παρελθοντικά δεδομένα
- **IntraDay τιμές**: Μόνο η τρέχουσα ώρα (δεν ξέρεις τις μελλοντικές)

---

## 8. Η `info` Dictionary — Debug Πληροφορίες

Κάθε `step()` επιστρέφει ένα `info` dictionary:

```python
info = {
    'soc': self.current_soc,              # Τρέχον State of Charge
    'actual_mw': power_mw,               # Πραγματικά MW (μετά masking)
    'violation': 1 if violated else 0,    # Παραβίαση constraints;
    'dam_price': dam_price,               # Τιμή DAM
    'best_bid': best_bid,                 # IntraDay bid
    'best_ask': best_ask,                 # IntraDay ask
    'dam_commitment': dam_commitment,     # Committed MW
    'reward_breakdown': {                 # Ανάλυση reward
        'dam_revenue': ...,
        'balancing_revenue': ...,
        'degradation_cost': ...,
        'shortfall_penalty': ...,
    }
}
```

**Χρησιμότητα:**
- Δεν χρησιμοποιείται στο training (ο agent ΔΕΝ βλέπει το info)
- Χρησιμοποιείται στο evaluation/debugging (`evaluate_detailed.py`)
- Σε logging/monitoring (callbacks)

---

## 9. Episode Structure

Ένα "episode" = μία πλήρης διάσχιση των δεδομένων:

```
Episode 1:
  Step 0: 2024-01-01 00:00 → SoC=0.50
  Step 1: 2024-01-01 01:00 → SoC=0.55
  ...
  Step 8759: 2024-12-31 23:00 → terminated=True
  Total reward = 45,230 (€ equivalent)

Episode 2:
  reset() → SoC=0.50
  Step 0: 2024-01-01 00:00 → ...
  (ξανά από την αρχή, αλλά τώρα ο agent ξέρει περισσότερα)
```

**Πόσα episodes γίνονται;**
Αν `timesteps=10_000_000` και τα data είναι 8760 ώρες (1 έτος):
- ~1142 episodes (10M / 8760)
- Ο agent "ζει" 1142 φορές το ίδιο έτος
- Κάθε φορά γίνεται λίγο καλύτερος

---

## 10. Πρακτική Άσκηση

### Άσκηση Α: Υπολογισμός SoC

Ξεκίνα με SoC = 0.50 και efficiency = 0.94:

1. Charge 20MW για 1 ώρα. Τι γίνεται το SoC;
   - ΔSoC = (20 × 1 × √0.94) / 146 = ?
   - Νέο SoC = 0.50 + ΔSoC = ?

2. Μετά, discharge 30MW για 1 ώρα. Τι γίνεται;
   - ΔSoC = -(30 × 1 / √0.94) / 146 = ?
   - Νέο SoC = ? (ελέγχεται αν < 0.05)

### Άσκηση Β: Κατανόηση action space

Αν `n_actions=21` και `max_discharge_mw=30`:

- Ποια MW αντιστοιχούν στο action 5;
  - normalized = (5 - 10) / 10 = -0.5
  - MW = -0.5 × 30 = ?

- Ποιο action αντιστοιχεί σε discharge 15MW;
  - Λύσε: (action - 10) / 10 × 30 = -15
  - action = ?

### Άσκηση Γ: Ανάγνωση κώδικα

Άνοιξε `gym_envs/battery_env_masked.py` και βρες:

1. Σε ποια γραμμή γίνεται η ενημέρωση SoC;
2. Πού ελέγχεται αν τελείωσε το episode;
3. Πού καλείται ο `reward_calculator`;

---

## Σύνοψη Ημέρας

```
BatteryEnvMasked
├── Battery Physics
│   ├── SoC update: ΔSoC = ±(P × t × √η) / C
│   ├── Operating range: [5%, 95%]
│   └── Round-trip efficiency: 94%
├── Market Data
│   ├── DAM price (from day-ahead, known)
│   ├── Best bid/ask (intraday, current)
│   └── DAM commitment (obligation from yesterday)
├── Time Features
│   ├── sin/cos encoding (cyclic hours)
│   └── 12h price + commitment lookahead
└── Step Output
    ├── observation (45 floats)
    ├── reward (profit - costs)
    ├── terminated (end of data?)
    └── info (debug dict)
```

> **Αύριο (Ημέρα 4)**: Θα κάνουμε deep dive στο reward function — πώς ο agent μαθαίνει τι είναι "καλό" και τι "κακό" μέσα από 11 reward components.
