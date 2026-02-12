# Deep Reinforcement Learning για Battery Energy Trading

## Οδηγός Κατανόησης του EntsoDRL System

> Αυτό το αρχείο εξηγεί τη φιλοσοφία και τη λογική του DRL συστήματος που εκπαιδεύουμε,
> ώστε να χρησιμεύσει ως εκπαιδευτικός οδηγός για όποιον θέλει να μάθει Deep Reinforcement Learning
> μέσα από ένα πραγματικό παράδειγμα.

---

## 1. Τι Είναι το Reinforcement Learning (RL)

### 1.1 Η Βασική Ιδέα

Φαντάσου ένα παιδί που μαθαίνει να παίζει ένα παιχνίδι. Κανείς δεν του λέει "κάνε αυτό στο βήμα 3
και εκείνο στο βήμα 7". Αντίθετα:

1. **Παρατηρεί** την κατάσταση (observation)
2. **Κάνει μια ενέργεια** (action)
3. **Λαμβάνει ανταμοιβή ή τιμωρία** (reward)
4. **Προσαρμόζεται** σταδιακά

Αυτό ακριβώς κάνει και ο agent μας: παρατηρεί την αγορά ενέργειας, αποφασίζει τι θα κάνει
με την μπαταρία, και μαθαίνει από τα κέρδη ή τις ζημιές.

### 1.2 Η Διαφορά από Supervised Learning

| Supervised Learning | Reinforcement Learning |
|---|---|
| Έχεις σωστές απαντήσεις (labels) | Δεν υπάρχουν σωστές απαντήσεις |
| Μαθαίνει από παραδείγματα | Μαθαίνει από εμπειρία |
| "Αυτή η εικόνα είναι γάτα" | "Αυτή η ενέργεια έφερε κέρδος" |
| Εκπαίδευση σε dataset | Εκπαίδευση μέσω αλληλεπίδρασης |

Στην αγορά ενέργειας **δεν υπάρχει σωστή απάντηση** — κανείς δεν ξέρει τη βέλτιστη
στρατηγική a priori. Γι' αυτό χρησιμοποιούμε RL.

### 1.3 Τα 5 Βασικά Στοιχεία

```
┌─────────────────────────────────────────────────────┐
│                  RL Framework                       │
│                                                     │
│   Agent ──(action)──> Environment                   │
│     ^                      │                        │
│     │                      │                        │
│     └──(observation,       │                        │
│         reward)────────────┘                        │
│                                                     │
│   1. Agent:       Ο "παίκτης" (νευρωνικό δίκτυο)   │
│   2. Environment: Ο "κόσμος" (αγορά + μπαταρία)    │
│   3. State:       Τι βλέπει ο agent                 │
│   4. Action:      Τι αποφασίζει                     │
│   5. Reward:      Πόσο καλά τα πήγε                 │
└─────────────────────────────────────────────────────┘
```

---

## 2. Πώς Εφαρμόζεται στο Σύστημά Μας

### 2.1 Ο Agent: MaskablePPO

Ο agent μας είναι ένα **νευρωνικό δίκτυο** που εκπαιδεύεται με τον αλγόριθμο
**Proximal Policy Optimization (PPO)** — έναν από τους πιο σταθερούς αλγόριθμους RL.

**Γιατί MaskablePPO;**
- **PPO**: Σταθερή εκπαίδευση, δεν κάνει τεράστια "άλματα" στη συμπεριφορά
- **Maskable**: Μπορεί να αποκλείσει **φυσικά αδύνατες** ενέργειες (π.χ. εκφόρτιση με άδεια μπαταρία)

Η αρχιτεκτονική του δικτύου:

```
Observation (58 features)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
 Policy    Value
  Head      Head
   │         │
 [256]     [256]    ← Hidden layer 1
   │         │
 [256]     [256]    ← Hidden layer 2
   │         │
   ▼         ▼
Actions   V(s)
(τι κάνω) (πόσο καλά είμαι)
```

- **Policy Head** (pi): Αποφασίζει **τι** να κάνει — ποια ενέργεια να εκτελέσει
- **Value Head** (vf): Εκτιμά **πόσο καλή** είναι η τρέχουσα κατάσταση — "αξίζει να είμαι εδώ;"

### 2.2 Το Environment: Η Προσομοίωση της Αγοράς

Το environment (`battery_env_unified.py`) προσομοιώνει:
- Την μπαταρία 146 MWh / 30 MW
- 4 αγορές ενέργειας ταυτόχρονα (DAM, IntraDay, aFRR, mFRR)
- Κανόνες αγοράς HEnEx (gate closures, penalties, settlement)
- Φυσική μπαταρίας (efficiency, degradation, SoC limits)

```
Αρχείο: gym_envs/battery_env_unified.py

Κάθε "βήμα" = 1 ώρα πραγματικού χρόνου στην αγορά
Κάθε "επεισόδιο" = 168 ώρες (1 εβδομάδα)
Εκπαίδευση = 3.000.000 βήματα = ~17.800 εβδομάδες εμπειρίας
```

---

## 3. Τι Βλέπει ο Agent (Observation Space — 58 Features)

Κάθε ώρα, ο agent λαμβάνει 58 αριθμούς που περιγράφουν τα πάντα:

### 3.1 Κατάσταση Μπαταρίας (3 features)

```python
# Αρχείο: gym_envs/battery_env_unified.py

soc                  # State of Charge: 0.05 - 0.95 (πόσο γεμάτη)
max_discharge_power  # Πόσα MW μπορεί να εκφορτίσει τώρα
max_charge_power     # Πόσα MW μπορεί να φορτίσει τώρα
```

**Γιατί είναι σημαντικά:** Ο agent πρέπει να ξέρει τι μπορεί να κάνει φυσικά.
Αν το SoC είναι στο 10%, δεν μπορεί να πουλήσει πολύ.

### 3.2 Τιμές Αγοράς (6 features)

```python
dam_price           # Τιμή Day-Ahead Market (EUR/MWh)
intraday_price      # Τιμή IntraDay (lagged — χωρίς data leakage!)
bid_ask_spread      # Διαφορά bid/ask
mfrr_up_price       # Τιμή mFRR για discharge
mfrr_down_price     # Τιμή mFRR για charge
mfrr_spread         # Διαφορά mFRR up/down
```

**Γιατί lagged τιμές;** Δεν θέλουμε ο agent να "βλέπει το μέλλον" — στην πραγματικότητα
γνωρίζεις μόνο τιμές που έχουν ήδη δημοσιευτεί.

### 3.3 Κωδικοποίηση Χρόνου (4 features)

```python
sin(2π × hour/24)    # "Πού βρισκόμαστε στη μέρα" (κυκλικά)
cos(2π × hour/24)
sin(2π × day/7)      # "Πού βρισκόμαστε στην εβδομάδα"
cos(2π × day/7)
```

**Γιατί sin/cos;** Η ώρα 23:00 και η 00:00 είναι **κοντά** χρονικά, αλλά αν χρησιμοποιήσουμε
απλά τον αριθμό (23 vs 0), φαίνονται μακριά. Η κυκλική κωδικοποίηση λύνει αυτό:

```
Ώρα    sin     cos     Απόσταση 23→0
23:    -0.26   0.97
 0:     0.00   1.00    ← Πολύ κοντά στο sin/cos space!
 1:     0.26   0.97
```

### 3.4 DAM Lookahead (13 features)

```python
current_dam_commitment        # Τρέχουσα δέσμευση DAM (MW)
next_12h_dam_commitments[12]  # Επόμενες 12 ώρες δεσμεύσεων
```

**Γιατί;** Αν ξέρω ότι σε 3 ώρες πρέπει να εκφορτίσω 20 MW (DAM commitment),
πρέπει να κρατήσω αρκετή ενέργεια. Αυτό βοηθά τον agent να σχεδιάσει μπροστά.

**HEnEx κανόνας:** Οι commitments μετά τα μεσάνυχτα φαίνονται ως 0 (δεν τα γνωρίζουμε ακόμα).

### 3.5 Price Lookahead (12 features)

```python
next_12h_prices[12]  # Τιμές DAM επόμενων 12 ωρών (αν είναι γνωστές)
```

**HEnEx κανόνας — Gate Closure:**
- Πριν τις 14:00: Βλέπεις μόνο τις τιμές **σήμερα**
- Μετά τις 14:00: Βλέπεις και τις τιμές **αύριο** (δημοσιεύονται στις 14:00)
- Άγνωστες τιμές γεμίζουν με persistence forecast (τελευταία γνωστή τιμή)

### 3.6 aFRR State (8 features)

```python
afrr_cap_up_price       # Τιμή aFRR capacity (up direction)
afrr_cap_down_price     # Τιμή aFRR capacity (down direction)
imbalance_signal        # Κατεύθυνση ανισορροπίας συστήματος
our_afrr_commitment     # Τρέχουσα δέσμευσή μας (MW)
selection_probability   # Εκτίμηση πιθανότητας επιλογής
recent_activation       # Ενεργοποιηθήκαμε πρόσφατα;
hours_since_activation  # Ώρες από τελευταία ενεργοποίηση
is_currently_selected   # Είμαστε τώρα επιλεγμένοι;
```

### 3.7 Risk/Imbalance (4 features)

```python
net_system_imbalance    # Ανισορροπία δικτύου (MW)
shortfall_risk          # Κίνδυνος αδυναμίας εκπλήρωσης DAM (0-1)
price_momentum          # Τάση τιμής (ανεβαίνει/πέφτει)
trade_worthiness        # Αξίζει να κάνω trade; (volatility)
```

### 3.8 Gate Closure Signals (4 features)

```python
dam_gate_open           # Μπορώ ακόμα να κάνω DAM bid; (1 πριν 14:00)
intraday_hours_left     # Πόσες ώρες μέχρι κλείσιμο IntraDay
afrr_hours_left         # Πόσες ώρες μέχρι κλείσιμο aFRR
mfrr_gate_open          # Είναι ανοιχτή η αγορά mFRR;
```

### 3.9 Timing Signals (4 features)

```python
price_vs_typical        # Η τρέχουσα τιμή vs μέσος 24ώρου
is_peak_hour            # Peak ώρα; (17:00-21:00)
is_solar_hour           # Ώρα ηλιακής παραγωγής; (09:00-15:00)
hours_to_daily_max      # Ώρες μέχρι αναμενόμενη μέγιστη τιμή
```

---

## 4. Τι Αποφασίζει ο Agent (Action Space)

### 4.1 MultiDiscrete([5, 5, 21]) = 525 Πιθανοί Συνδυασμοί

Κάθε ώρα, ο agent παίρνει **3 αποφάσεις ταυτόχρονα**:

```
Απόφαση 1: Πόση χωρητικότητα aFRR προσφέρω;
┌──────────────────────────────────────┐
│ 0: Τίποτα (0%)                       │
│ 1: Λίγο (25% διαθέσιμης ισχύος)     │
│ 2: Μέτρια (50%)                      │
│ 3: Πολύ (75%)                        │
│ 4: Όλα (100%)                        │
└──────────────────────────────────────┘

Απόφαση 2: Σε τι τιμή κάνω bid στο aFRR;
┌──────────────────────────────────────┐
│ 0: Πολύ φθηνά (0.7x — μεγάλη        │
│    πιθανότητα επιλογής, χαμηλό κέρδος)│
│ 1: Φθηνά (0.85x)                    │
│ 2: Στην αγορά (1.0x)                │
│ 3: Ακριβά (1.15x)                   │
│ 4: Πολύ ακριβά (1.3x — μικρή        │
│    πιθανότητα, μεγάλο κέρδος)        │
└──────────────────────────────────────┘

Απόφαση 3: Τι κάνω στο mFRR/IntraDay;
┌──────────────────────────────────────┐
│  0: Φόρτιση  -30 MW (μέγιστη αγορά) │
│  1: Φόρτιση  -27 MW                 │
│  ...                                 │
│  9: Φόρτιση   -3 MW                 │
│ 10: Αδράνεια   0 MW (idle)          │
│ 11: Εκφόρτιση  +3 MW                │
│  ...                                 │
│ 19: Εκφόρτιση +27 MW                │
│ 20: Εκφόρτιση +30 MW (μέγιστη πώληση)│
└──────────────────────────────────────┘
```

### 4.2 Γιατί MultiDiscrete και Όχι Continuous;

Θα μπορούσαμε να δώσουμε στον agent **συνεχείς τιμές** (π.χ. "εκφόρτισε 17.3 MW").
Αλλά τα discrete actions έχουν πλεονεκτήματα:

1. **Ευκολότερη εκμάθηση**: "Ψήφισε 1 από 21" είναι πιο εύκολο από "βρες τον σωστό αριθμό"
2. **Action masking**: Μπορούμε να αποκλείσουμε μεμονωμένες επιλογές
3. **Αγορές ενέργειας**: Στην πραγματικότητα κάνεις bids σε στρογγυλοποιημένα MW

---

## 5. Action Masking — Η Ασπίδα Ασφαλείας

### 5.1 Η Ιδέα

Ο agent δεν πρέπει ποτέ να επιλέξει κάτι **φυσικά αδύνατο**. Αντί να τον αφήσουμε
να κάνει λάθος και να τον τιμωρήσουμε, **αφαιρούμε τελείως** τις αδύνατες επιλογές.

```
Αρχείο: gym_envs/action_mask.py

Χωρίς masking:                    Με masking:
┌───────────────────┐             ┌───────────────────┐
│ -30 MW  ← επέλεξε│             │ -30 MW  ✗ blocked │
│ -27 MW            │             │ -27 MW  ✗ blocked │
│ -24 MW            │             │ -24 MW  ✗ blocked │
│  ...              │             │  ...              │
│  -6 MW            │             │  -6 MW  ✓         │
│  -3 MW            │             │  -3 MW  ✓         │
│   0 MW            │             │   0 MW  ✓ (πάντα) │
│  +3 MW            │             │  +3 MW  ✓         │
│  ...              │             │  ...              │
│ +30 MW            │             │ +30 MW  ✗ blocked │
└───────────────────┘             └───────────────────┘
  Μπορεί να "σπάσει"               Μόνο ασφαλείς επιλογές
  τη μπαταρία                       είναι διαθέσιμες
```

### 5.2 Τι Ελέγχεται

```python
def action_masks():
    # I calculate remaining capacity after DAM commitment
    remaining_capacity = max_power_mw - abs(dam_commitment)

    # I calculate available energy from battery state
    available_discharge = (soc - min_soc) × capacity / sqrt(efficiency)
    available_charge = (max_soc - soc) × capacity × sqrt(efficiency)

    # aFRR masks: Μπορώ να δεσμεύσω αυτή τη χωρητικότητα;
    for level in [0%, 25%, 50%, 75%, 100%]:
        required = level × remaining_capacity
        if required > available_discharge OR required > available_charge:
            mask[level] = False  # Δεν μπορώ να ανταποκριθώ αν ενεργοποιηθώ

    # Energy masks: Μπορώ να εκτελέσω αυτή τη φόρτιση/εκφόρτιση;
    for power_mw in [-30, -27, ..., +27, +30]:
        if power_mw > 0 and power_mw > available_discharge:
            mask[action] = False  # Δεν έχω αρκετή ενέργεια
        if power_mw < 0 and abs(power_mw) > available_charge:
            mask[action] = False  # Δεν χωράει η μπαταρία
```

### 5.3 Παράδειγμα

**Κατάσταση:** SoC = 8%, DAM commitment = 5 MW discharge

```
Διαθέσιμη εκφόρτιση: (0.08 - 0.05) × 146 / √0.94 = 4.5 MWh
Διαθέσιμη φόρτιση:   (0.95 - 0.08) × 146 × √0.94 = 123 MWh
Remaining capacity:   30 - 5 = 25 MW

aFRR masks: [✓, ✗, ✗, ✗, ✗]
  — Μόνο 0% επιτρέπεται (4.5 MWh < 6.25 MW × 1h για 25%)

Energy masks: [✓✓✓...✓, ✓(idle), ✓(+3), ✗(+6), ✗..✗]
  — Charge: Σχεδόν όλα ΟΚ (123 MWh χωρητικότητα)
  — Discharge: Μόνο +3 MW (4.5 - DAM overhead)
```

### 5.4 Masking vs Penalties — Η Φιλοσοφική Διαφορά

| | Action Masking | Penalties |
|---|---|---|
| **Σκοπός** | Αποτρέπει **φυσικά αδύνατα** | Αποθαρρύνει **οικονομικά κακά** |
| **Παράδειγμα** | Εκφόρτιση με SoC 5% | Overbid στο aFRR χωρίς ενέργεια |
| **Μηχανισμός** | Αφαίρεση επιλογής | Αρνητικό reward |
| **Ο agent** | Δεν μπορεί καν να το δοκιμάσει | Μπορεί, αλλά θα "πονέσει" |

Αυτός ο διαχωρισμός είναι κρίσιμος: ο agent **μαθαίνει** από τα οικονομικά λάθη,
αλλά **ποτέ** δεν κάνει φυσικά αδύνατες ενέργειες.

---

## 6. Το Reward System — Πώς Μαθαίνει

### 6.1 Η Γενική Φόρμουλα

```
Αρχείο: gym_envs/unified_reward_calculator.py

Reward = (Έσοδα) - (Κόστη)

Έσοδα = DAM Revenue
       + aFRR Capacity Revenue
       + aFRR Energy Revenue
       + mFRR Revenue
       + Bonuses (compliance, SoC buffer)

Κόστη = Degradation Cost
      + DAM Violation Penalty
      + aFRR Non-Response Penalty
      + Shortfall Risk Penalty
```

### 6.2 Πηγές Εσόδων

#### A. DAM Revenue (Day-Ahead Market)

```python
# Η μπαταρία εκτελεί δεσμεύσεις που έγιναν χθες
dam_revenue = dam_commitment_mw × dam_price × 1_hour

# Αν εκτελέσει ακριβώς τη δέσμευση → bonus
if actual >= dam_commitment:
    dam_bonus = 5 EUR/MWh × actual_delivered

# Παράδειγμα: 10 MW × 80 EUR/MWh = 800 EUR + 50 EUR bonus
```

#### B. aFRR Capacity Revenue (Πληρώνεσαι για Διαθεσιμότητα)

```python
# Αν επιλεγείς, πληρώνεσαι ΜΟΝΟ ΚΑΙ ΜΟΝΟ που υπάρχεις
afrr_capacity_revenue = commitment_mw × cap_price × 1_hour

# Παράδειγμα: 10 MW × 20 EUR/MW/h = 200 EUR
# Αυτά τα παίρνεις είτε ενεργοποιηθείς είτε όχι!
```

Αυτό είναι **πολύ ελκυστικό** — χρήμα χωρίς να κάνεις τίποτα. Αλλά... αν ενεργοποιηθείς
και δεν μπορείς να ανταποκριθείς, η τιμωρία είναι **καταστροφική** (βλ. penalties).

#### C. aFRR Energy Revenue (Αν Ενεργοποιηθείς)

```python
# Ο TSO σε καλεί να βοηθήσεις το σύστημα
if direction == 'up':  # Το σύστημα χρειάζεται ενέργεια
    revenue = actual_mw × afrr_up_price × 1_hour
    # Bonus αν ανταποκριθείς > 90%
    if actual >= 0.9 × committed:
        bonus = 10 EUR/MWh × committed
```

#### D. mFRR Revenue (Manual Frequency Restoration)

```python
# Trading ενέργειας στην αγορά mFRR
if energy_mw > 0:  # Εκφόρτιση (πώληση)
    revenue = energy_mw × mfrr_up_price
else:  # Φόρτιση (αγορά)
    cost = abs(energy_mw) × mfrr_down_price
```

### 6.3 Κόστη και Penalties

#### A. Degradation — Φθορά Μπαταρίας

```python
# Κάθε κύκλος φόρτισης/εκφόρτισης φθείρει τη μπαταρία
energy_cycled = abs(power_mw) × 1_hour
cycle_fraction = energy_cycled / 146  # (% πλήρους κύκλου)
degradation_cost = cycle_fraction × 146 × 15  # EUR

# Παράδειγμα: 30 MW × 1h = 30 MWh
#   cycle_fraction = 30/146 = 0.205
#   cost = 0.205 × 146 × 15 = 449 EUR
```

**Αυτό σημαίνει:** Κάθε φορά που κάνεις trade, "ξοδεύεις" λίγο τη μπαταρία.
Ο agent πρέπει να μάθει: "αξίζει αυτό το trade σε σχέση με τη φθορά;"

#### B. DAM Violation Penalty — 1.500 EUR/MWh

```python
# Αν δεν εκπληρώσεις τη DAM δέσμευση → ΒΑΡΙΑ τιμωρία
if actual < dam_commitment:
    shortfall = dam_commitment - actual
    penalty = shortfall × 1500  # EUR

# Παράδειγμα: Δέσμευση 10 MW, παρέδωσες 5 MW
#   penalty = 5 × 1500 = 7,500 EUR!
```

#### C. aFRR Non-Response Penalty — 2.000 EUR/MWh

```python
# Η χειρότερη τιμωρία: δεν ανταποκρίθηκες στο TSO
if activated AND delivered < 0.9 × committed:
    penalty = committed × 2000  # EUR

# Παράδειγμα: Δέσμευση 10 MW, δεν μπορείς
#   penalty = 10 × 2000 = 20,000 EUR!
```

### 6.4 Πλήρες Παράδειγμα — Ένα Βήμα

```
Σενάριο: Ώρα 18:00 (peak), SoC 65%
  DAM commitment: +12 MW (εκφόρτιση) @ 95 EUR/MWh
  aFRR: Επιλεγμένος 13.5 MW, ΕΝΕΡΓΟΠΟΙΗΘΗΚΕ (up direction)
  Remaining after DAM: 6 MW μόνο

ΕΚΤΕΛΕΣΗ:
  1. DAM: Εκφόρτισε 12 MW ✓ → 12 × 95 = 1,140 EUR
  2. aFRR: Χρειάζεται 13.5 MW, αλλά μόνο 6 MW available!
     → Energy revenue: 6 × 85 = 510 EUR
     → PENALTY: 13.5 × 2000 = 27,000 EUR!
  3. mFRR: Ακυρώθηκε (aFRR έχει προτεραιότητα)

ΛΟΓΑΡΙΑΣΜΟΣ:
  Έσοδα:  1,140 + 337 (capacity) + 510 + 60 (bonus) = 2,047 EUR
  Κόστη:  270 (degradation) + 27,000 (penalty) = 27,270 EUR
  ─────────────────────────────────────────────────────────
  ΖΗΜΙΑ:  -25,223 EUR  ← Ο agent θα "θυμάται" αυτό!
```

**Τι μαθαίνει:** "Μην δεσμεύεις 75% στο aFRR όταν έχεις μεγάλο DAM commitment
σε peak hour — ο κίνδυνος ενεργοποίησης είναι υψηλός!"

### 6.5 Reward Scaling

```python
# Τα raw rewards (χιλιάδες EUR) θα "τρέλαιναν" το neural network
# Γι' αυτό πολλαπλασιάζουμε × 0.001
final_reward = net_profit × 0.001

# 1,000 EUR profit → reward = 1.0
# -5,000 EUR loss → reward = -5.0
```

Επίσης, το `VecNormalize` κανονικοποιεί περαιτέρω τα rewards ώστε να έχουν
μέσο ~0 και τυπική απόκλιση ~1. Αυτό βοηθά πάρα πολύ τη σύγκλιση.

---

## 7. Η Ροή Εκτέλεσης — Τι Γίνεται Κάθε Ώρα

```
Αρχείο: gym_envs/battery_env_unified.py — step()

Agent παίρνει action = [afrr_level, price_tier, energy_action]
                │
                ▼
┌─── ΦΑΣΗ 1: aFRR COMMITMENT ───────────────────────┐
│  Μετατροπή afrr_level σε MW                        │
│  Υπολογισμός πιθανότητας επιλογής βάσει price_tier │
│  Κλήρωση: Επιλέχθηκε; (random < probability)      │
│  Αν ΝΑΙ → capacity revenue                         │
└────────────────────────────────────────────────────┘
                │
                ▼
┌─── ΦΑΣΗ 2: ΕΛΕΓΧΟΣ ΕΝΕΡΓΟΠΟΙΗΣΗΣ aFRR ───────────┐
│  Base rate: 15%                                     │
│  + 30% boost σε peak hours (17:00-21:00)           │
│  + 50% boost σε high imbalance (>500 MW)           │
│  Κλήρωση: Ενεργοποιήθηκε;                          │
│  Κατεύθυνση: up (discharge) ή down (charge)        │
└────────────────────────────────────────────────────┘
                │
                ▼
┌─── ΦΑΣΗ 3: ΕΚΤΕΛΕΣΗ DAM (ΥΠΟΧΡΕΩΤΙΚΗ) ───────────┐
│  dam_commitment > 0 → discharge (πώληση)            │
│  dam_commitment < 0 → charge (αγορά)               │
│  ΠΡΕΠΕΙ να εκτελεστεί — αλλιώς penalty!            │
│  Ενημέρωση SoC                                      │
└────────────────────────────────────────────────────┘
                │
                ▼
┌─── ΦΑΣΗ 4: ΕΚΤΕΛΕΣΗ aFRR ENERGY (αν activated) ──┐
│  Παίρνει ΠΡΟΤΕΡΑΙΟΤΗΤΑ πάνω από mFRR              │
│  Ανταπόκριση με ό,τι capacity απομένει             │
│  Αν < 90% committed → PENALTY 2000 EUR/MWh!       │
│  Ενημέρωση SoC                                      │
└────────────────────────────────────────────────────┘
                │
                ▼
┌─── ΦΑΣΗ 5: ΕΚΤΕΛΕΣΗ mFRR/IntraDay ───────────────┐
│  ΜΟΝΟ αν aFRR ΔΕΝ ενεργοποιήθηκε                  │
│  Χρησιμοποιεί remaining capacity                   │
│  Positive → discharge, Negative → charge           │
│  Ενημέρωση SoC                                      │
└────────────────────────────────────────────────────┘
                │
                ▼
┌─── ΦΑΣΗ 6: ΥΠΟΛΟΓΙΣΜΟΣ REWARD ───────────────────┐
│  Σύνοψη εσόδων - κοστών (βλ. §6)                  │
│  Scaling × 0.001                                    │
└────────────────────────────────────────────────────┘
                │
                ▼
        Επιστροφή (observation, reward, done, info)
```

---

## 8. Πώς Εκπαιδεύεται (Training Process)

### 8.1 PPO — Η Βασική Ιδέα

Ο PPO λειτουργεί σε **κύκλους**:

```
Κύκλος N:
  1. ROLLOUT: Ο agent παίζει 2048 βήματα σε κάθε ένα
     από τα 8 παράλληλα environments = 16,384 samples

  2. ADVANTAGE: Υπολογίζει "πόσο καλύτερα/χειρότερα
     πήγε από ό,τι περίμενε" (advantage estimation)

  3. UPDATE: Ενημερώνει το neural network 10 φορές
     (n_epochs) σε batches των 256 samples

  4. CLIP: Αν η αλλαγή στη policy είναι > 20% (clip_range),
     κόβεται — αυτό προστατεύει από ασταθείς αλλαγές

  → Επόμενος κύκλος
```

### 8.2 Τα Hyperparameters Εξηγημένα

```python
# Αρχείο: agent/train_unified.py

learning_rate = 1e-4    # Πόσο γρήγορα μαθαίνει
# ↑ Μικρό = σταθερή μάθηση, Μεγάλο = γρήγορη αλλά ασταθής
# I μείωσα από 3e-4 σε 1e-4 γιατί η clip fraction ήταν 35%

ent_coef = 0.03         # Πόσο ενθαρρύνουμε την "περιέργεια"
# ↑ Μικρό = ο agent κολλάει σε μία στρατηγική
# Μεγάλο = εξερευνεί περισσότερο
# I αύξησα από 0.01 σε 0.03 γιατί η entropy έπεφτε γρήγορα

gamma = 0.99            # Discount factor
# ↑ Πόσο σημαντικό είναι το μέλλον vs το τώρα
# 0.99 = 99% βάρος στο μέλλον = σκέφτεται μακροπρόθεσμα
# (σημαντικό γιατί οι DAM commitments είναι δεσμευτικές)

clip_range = 0.2        # Μέγιστη αλλαγή policy ανά update
# ↑ Αν ο agent θέλει να αλλάξει πολύ τη στρατηγική,
# κόβεται στο 20% — αυτό σταθεροποιεί τη μάθηση

n_steps = 2048          # Βήματα πριν κάνει update
# ↑ Μεγαλύτερο = καλύτερη εκτίμηση, αλλά πιο αργό
# 2048 × 8 envs = 16,384 samples ανά update

batch_size = 256        # Μέγεθος batch για SGD
# ↑ 16,384 / 256 = 64 mini-batches × 10 epochs = 640 gradient steps
```

### 8.3 Curriculum Learning — Σταδιακή Δυσκολία

```
Αρχείο: agent/train_unified.py — DegradationCurriculumCallback

Φάση 1 (0 → 500K steps):
  degradation_cost = 0 → 15 EUR/MWh (γραμμικά)

  Βήμα      0:  cost = 0   EUR/MWh  "Η μπαταρία είναι δωρεάν!"
  Βήμα 100K:  cost = 3   EUR/MWh  "Λίγο κόστος..."
  Βήμα 250K:  cost = 7.5 EUR/MWh  "Αρχίζει να μετράει"
  Βήμα 500K:  cost = 15  EUR/MWh  "Πλήρες κόστος"

Φάση 2 (500K+):
  degradation_cost = 15 EUR/MWh (σταθερό)
```

**Γιατί;** Αν από την αρχή βάλεις πλήρες degradation cost, ο agent μαθαίνει:
"η καλύτερη στρατηγική είναι να μην κάνω τίποτα" (idle forever).

Με curriculum:
1. Πρώτα μαθαίνει **ΠΩΣ** να κάνει trade (χωρίς κόστος φθοράς)
2. Μετά μαθαίνει **ΠΟΤΕ** αξίζει να κάνει trade (με κόστος φθοράς)

### 8.4 VecNormalize — Κανονικοποίηση

```python
# Αρχείο: agent/train_unified.py

train_env = VecNormalize(
    train_env,
    norm_obs=True,      # Κανονικοποίηση observations
    norm_reward=True,    # Κανονικοποίηση rewards
    clip_obs=10.0,      # Αποκοπή ακραίων τιμών
    clip_reward=10.0
)
```

**Γιατί;** Τα 58 features έχουν πολύ διαφορετικές κλίμακες:

```
Πριν κανονικοποίηση:           Μετά κανονικοποίηση:
  SoC: 0.05 - 0.95              SoC: -1.5 ... +1.5
  DAM price: 20 - 300            DAM price: -2.0 ... +2.0
  Imbalance: -1000 ... +1000     Imbalance: -1.8 ... +1.8
```

Τα neural networks δουλεύουν πολύ καλύτερα όταν όλα τα inputs είναι στην ίδια κλίμακα (~0, ±2).

### 8.5 Parallel Environments — 8× Ταχύτητα

```python
# 8 environments τρέχουν ΤΑΥΤΟΧΡΟΝΑ
train_env = MaskableVecEnv([make_env(i) for i in range(8)])
```

```
Env 0: Εβδομάδα 14 Μαρτίου 2023    ─┐
Env 1: Εβδομάδα 5 Ιουνίου 2024      │  Τρέχουν
Env 2: Εβδομάδα 21 Δεκεμβρίου 2022  │  παράλληλα
Env 3: Εβδομάδα 8 Σεπτεμβρίου 2023  ├─ σε κάθε
Env 4: Εβδομάδα 3 Φεβρουαρίου 2024  │  GPU batch
Env 5: Εβδομάδα 17 Ιουλίου 2022     │
Env 6: Εβδομάδα 29 Απριλίου 2023    │
Env 7: Εβδομάδα 12 Νοεμβρίου 2024   ─┘
```

**Πλεονέκτημα:** Κάθε update χρησιμοποιεί δεδομένα από 8 **διαφορετικές** χρονικές
περιόδους. Αυτό αποτρέπει τον agent από το να "αποστηθίσει" μια συγκεκριμένη εβδομάδα.

---

## 9. Τι Παρακολουθούμε (TensorBoard Metrics)

### 9.1 Μετρικές Σταθερότητας Εκπαίδευσης

```
clip_fraction:
  Ιδανικό: < 15%  ← Οι περισσότεροι updates εφαρμόζονται κανονικά
  Κακό: > 30%     ← Πάρα πολλά updates κόβονται (LR πολύ μεγάλο)

approx_kl (KL divergence):
  Ιδανικό: < 0.015 ← Η policy αλλάζει ομαλά
  Κακό: > 0.03     ← Η policy αλλάζει απότομα (ασταθές)

entropy_loss:
  Αρχή: ~-5.0     ← Πολύ random (εξερεύνηση)
  Μέση: ~-3.5      ← Μέτρια εξειδίκευση
  Κακό: > -2.0     ← Πολύ "σίγουρος" (premature convergence)

explained_variance:
  Ιδανικό: > 0.7   ← Ο Value Head εκτιμά σωστά
  Κακό: < 0.0      ← Δεν ξέρει τι γίνεται
```

### 9.2 Μετρικές Performance

```
avg_profit:           Μέσο κέρδος ανά εβδομάδα (EUR)
avg_cycles:           Μέσοι κύκλοι ανά εβδομάδα
profit_per_cycle:     Κέρδος ανά κύκλο (βασική μετρική αποδοτικότητας)

Per-market breakdown:
  dam_profit:             Κέρδος DAM
  afrr_capacity_profit:   Έσοδα από aFRR capacity
  afrr_energy_profit:     Έσοδα/κόστη aFRR energy
  mfrr_profit:            Κέρδος mFRR
```

### 9.3 Πώς Διαβάζουμε τα Αποτελέσματα

Ένα τυπικό log entry:

```
iteration 22, total_timesteps 360,448:
  avg_profit        = 89,400 EUR    ← Κερδοφόρο!
  profit_per_cycle  = 6,960 EUR     ← Υγιές
  clip_fraction     = 0.107         ← Εξαιρετικό (< 15%)
  approx_kl         = 0.0105        ← Εξαιρετικό (< 0.015)
  entropy           = -3.93         ← Καλό (εξερευνεί ακόμα)
  explained_var     = 0.761         ← Πολύ καλό (> 0.7)
```

**Ερμηνεία:** Ο agent μαθαίνει σταθερά, βγάζει κέρδος, και συνεχίζει να εξερευνεί
νέες στρατηγικές χωρίς να "κολλήσει" πρόωρα.

---

## 10. Σύνοψη Φιλοσοφίας

### 10.1 Γιατί DRL και Όχι Rule-Based;

| Rule-Based | DRL Agent |
|---|---|
| "Αν τιμή > 100, πούλα" | Μαθαίνει πολύπλοκα patterns |
| Δεν προσαρμόζεται | Προσαρμόζεται σε νέες συνθήκες |
| Ανθρώπινη εμπειρία | Υπερ-ανθρώπινη βελτιστοποίηση |
| 1 αγορά τη φορά | 4 αγορές ταυτόχρονα |
| Δεν σκέφτεται μέλλον | gamma=0.99 → μακροπρόθεσμα |

### 10.2 Τα 4 Βασικά Σχεδιαστικά Μοτίβα

1. **Action Masking**: Η φυσική δεν παραβιάζεται ποτέ
2. **Curriculum Learning**: Από απλά σε σύνθετα (σαν σχολείο)
3. **Hierarchical Decisions**: aFRR strategy + energy trading ξεχωριστά
4. **Reward Shaping**: Οι penalties αντικατοπτρίζουν τους πραγματικούς κανόνες αγοράς

### 10.3 Η Αλυσίδα Μάθησης

```
Δεδομένα αγοράς (52,040 ώρες)
        │
        ▼
Simulated Environment (8 παράλληλα)
        │
        ▼
Agent παρατηρεί → αποφασίζει → εκτελεί
        │
        ▼
Reward signal (+κέρδος / -ζημία)
        │
        ▼
PPO update (gradient descent στο neural network)
        │
        ▼
Νέα, καλύτερη στρατηγική
        │
        ▼
... επανάληψη 3,000,000 φορές ...
        │
        ▼
Τελικός agent που ξέρει πότε/πόσο/πού
να κάνει trade σε 4 αγορές ταυτόχρονα
```

---

## Παράρτημα: Χρήσιμοι Πόροι για Εκμάθηση

1. **PPO Paper**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
2. **Stable Baselines 3 Docs**: https://stable-baselines3.readthedocs.io/
3. **Gymnasium Docs**: https://gymnasium.farama.org/
4. **Spinning Up in Deep RL**: https://spinningup.openai.com/ (OpenAI tutorial)
5. **SB3-Contrib (MaskablePPO)**: https://sb3-contrib.readthedocs.io/

---

*Τελευταία ενημέρωση: 12 Φεβρουαρίου 2026*
*Training run: `unified_20260212_045603` (LR=1e-4, ent_coef=0.03)*
