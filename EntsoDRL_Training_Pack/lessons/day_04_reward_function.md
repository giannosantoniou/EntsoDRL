# Ημέρα 4: Reward Function — Πώς ο Agent Μαθαίνει τι είναι "Καλό"

> **Στόχος**: Να καταλάβεις πώς δουλεύει η reward function στο `gym_envs/reward_calculator.py`.
> Αυτό είναι το πιο κρίσιμο κομμάτι — ό,τι ορίσεις εδώ, αυτό μαθαίνει ο agent.

---

## 1. Η Μεγάλη Αρχή: Reward = "Βαθμολογία Δασκάλου"

Φαντάσου ότι εκπαιδεύεις ένα σκυλί:
- Κάνει κάτι καλό → λιχουδιά (+reward)
- Κάνει κάτι κακό → αγνόηση ή ήχος (-reward)
- Με τον καιρό, μαθαίνει τι πρέπει να κάνει

Το ίδιο ακριβώς κάνει η reward function: **βαθμολογεί κάθε action** ώστε ο agent να μάθει ποιες αποφάσεις είναι κερδοφόρες.

**Κρίσιμη αρχή**: Ο agent θα κάνει ΑΚΡΙΒΩΣ αυτό που τον reward-άρεις. Αν η reward function έχει bugs ή κακά κίνητρα, ο agent θα μάθει κακή συμπεριφορά — τέλεια.

---

## 2. Η Φόρμουλα — Επισκόπηση

```python
# Από gym_envs/reward_calculator.py:
Total Reward = (
    DAM Revenue                    # Κέρδος από Day-Ahead Market
  + Balancing Revenue              # Κέρδος/κόστος από IntraDay
  - Degradation Cost               # Κόστος φθοράς μπαταρίας
  - Shortfall Penalty              # Πρόστιμο αν δεν εκπλήρωσε commitment
  - Proactive Risk Penalty         # Πρόστιμο για ρίσκο shortfall
  + SoC Buffer Bonus               # Μπόνους για υγιές SoC level
  + Price Timing Bonus             # Μπόνους αν πούλησε ακριβά / αγόρασε φθηνά
  + Momentum Timing Bonus          # Μπόνους αν ακολουθεί price trends
  + Quartile Adjustment            # Penalty αν αγοράζει/πουλάει σε κακές τιμές
  + DAM Compliance Bonus           # Μπόνους αν τηρεί DAM commitments
) / 100                            # Κλιμάκωση για το neural network
```

Ας δούμε κάθε component ξεχωριστά:

---

## 3. DAM Revenue — Το "σίγουρο" κέρδος

```python
# I calculate revenue from the Day-Ahead Market commitment
dam_revenue = dam_commitment * dam_price
```

**Παράδειγμα:**
```
Χθες δεσμεύτηκες: "Αύριο 14:00 θα πουλήσω 10MW"
dam_commitment = +10 MW
dam_price = 85 €/MWh

dam_revenue = 10 × 85 = 850€

Αν δεσμεύτηκες να αγοράσεις:
dam_commitment = -10 MW → dam_revenue = -10 × 85 = -850€ (κόστος)
```

**Σημαντικό**: Αυτό είναι κλειδωμένο (contracted) — ανεξάρτητα τι κάνεις φυσικά, αυτά τα χρήματα τα πληρώνεις/εισπράττεις.

---

## 4. Balancing Revenue — Τι γίνεται με τη "διαφορά"

```python
# I settle the imbalance between physical action and DAM commitment
imbalance_mw = actual_physical_mw - dam_commitment

if imbalance_mw > 0:
    # Πούλησα ΠΕΡΙΣΣΟΤΕΡΑ από ό,τι committed → εισπράττω bid price
    balancing_revenue = imbalance_mw * best_bid
elif imbalance_mw < 0:
    # Πούλησα ΛΙΓΟΤΕΡΑ (ή αγόρασα) → πληρώνω ask price
    balancing_revenue = imbalance_mw * best_ask  # αρνητικό!
```

**Παράδειγμα:**
```
Commitment: πώληση 10MW (dam_commitment = +10)
Πραγματική εκφόρτιση: 25MW (actual = -25, αλλά σε sell terms = +25)

Imbalance = 25 - 10 = +15MW extra sold
Balancing = 15 × best_bid(90€) = +1350€

ΑΛΛΑ αν πούλησες μόνο 5MW:
Imbalance = 5 - 10 = -5MW shortfall
Balancing = -5 × best_ask(94€) = -470€ (πληρώνεις ακριβότερα!)
```

---

## 5. Degradation Cost — "Φθαρμένη μπαταρία"

```python
# I penalize battery usage to encourage selective trading
degradation_cost = abs(actual_mw) * degradation_cost_per_mwh  # default: 0.5 €/MWh
```

**Γιατί υπάρχει:**
- Κάθε κύκλος charge/discharge φθείρει τη μπαταρία
- Χωρίς αυτό, ο agent θα κάνει trade σε ΚΑΘΕ ώρα
- Με αυτό, μαθαίνει: "κάνε trade ΜΟΝΟ αν η τιμή αξίζει"

**Παράδειγμα:**
```
Discharge 30MW → degradation = 30 × 0.5 = 15€
Idle 0MW → degradation = 0€
Charge 20MW → degradation = 20 × 0.5 = 10€
```

---

## 6. Shortfall Penalty — Η πιο σοβαρή ποινή

```python
# I apply a STEEP penalty for failing to deliver on DAM commitments
PENALTY_RATE = 1500  # €/MWh — fixed to avoid data leakage

if committed_to_sell and delivered_less:
    shortfall = commitment - actual_delivery
    penalty = shortfall * PENALTY_RATE

elif committed_to_buy and received_less:
    shortfall = actual_reception - commitment
    penalty = shortfall * PENALTY_RATE
```

**Γιατί 1500 €/MWh;**
- Στην πραγματικότητα, η ποινή εξαρτάται από imbalance prices (μεταβλητές)
- Αν χρησιμοποιούσαμε πραγματικές imbalance prices → **data leakage** (ο agent μαθαίνει τις μελλοντικές τιμές)
- Fixed 1500 €/MWh = ασφαλές, σταθερό penalty

**Παράδειγμα:**
```
Commitment: sell 10MW
Actual: sell 7MW (μόνο)
Shortfall: 10 - 7 = 3MW

Penalty: 3 × 1500 = 4,500€ ← ΠΟΛΥ ΑΚΡΙΒΟ!

Μάθημα για τον agent: "Αν δεσμεύτηκες, ΠΑΡΑΔΩΣΕ."
```

---

## 7. Proactive Risk Penalty — "Πρόληψη"

```python
# I penalize the agent BEFORE a shortfall happens,
# when SoC is dangerously low relative to commitments
if shortfall_risk > 0.20:
    penalty = 50.0 * (shortfall_risk - 0.20) ** 1.5
```

**Τι μετράει:**
- `shortfall_risk`: πόσο κοντά είναι η μπαταρία στο να μην μπορεί να εκπληρώσει τα commitments
- 0.0 = πλήρως ασφαλής
- 1.0 = σίγουρο shortfall

**Γιατί υπάρχει:**
Χωρίς αυτό, ο agent μαθαίνει: "Πούλα μέχρι να αδειάσεις, πλήρωσε penalty μία φορά."
Με αυτό, μαθαίνει: "Κράτα αρκετή ενέργεια για τα commitments σου."

---

## 8. SoC Buffer Bonus — "Υγιής μπαταρία"

```python
# I reward maintaining a healthy SoC level
if soc >= 0.25:
    bonus = 2.0 * (soc - 0.25) / 0.70     # Max 2.0 at SoC=0.95
elif soc < 0.25:
    penalty = -2.0 * (0.25 - soc) / 0.25 * 2  # Double penalty below 25%
```

**Διάγραμμα:**
```
Bonus
  +2 |                              ___________
     |                         ____/
  +1 |                    ____/
     |               ____/
   0 |──────────────/────────────────────────── SoC
     |            /
  -1 |          /
     |        /
  -4 |──────/
     0%   5%   25%                95%   100%
         ↑              ↑
     DANGER ZONE    OPTIMAL ZONE
```

**Ο agent μαθαίνει**: "Μην αφήσεις τη μπαταρία κάτω από 25%. Ιδανικά 40-60%."

---

## 9. Price Timing Bonus — "Buy Low, Sell High"

```python
# I reward the agent for trading at good prices
mean_24h = np.mean(prices_last_24h)

if discharging and current_price > mean_24h:
    # Πουλάω ΠΑΝΩ από τον μέσο → μπόνους
    bonus = price_timing_factor * (current_price - mean_24h) / mean_24h

elif charging and current_price < mean_24h:
    # Αγοράζω ΚΑΤΩ από τον μέσο → μπόνους
    bonus = price_timing_factor * (mean_24h - current_price) / mean_24h
```

**Παράδειγμα:**
```
mean_24h = 80€
Current price = 120€, discharge 20MW → bonus! (πουλάω ακριβά)
Current price = 45€, charge 20MW → bonus! (αγοράζω φθηνά)
Current price = 75€, discharge 20MW → no bonus (κοντά στον μέσο)
```

---

## 10. Momentum Timing Bonus — "Follow the Trend"

```python
# I use BACKWARD-LOOKING momentum (no data leakage!)
price_momentum = (current_price - price_6h_ago) / price_6h_ago

if price_momentum > 0.05 and discharging:
    # Τιμές ανεβαίνουν + πουλάω → μπόνους
    bonus = 8.0 * (power / max_power) * min(abs(momentum), 0.3)

elif price_momentum < -0.05 and charging:
    # Τιμές πέφτουν + αγοράζω → μπόνους
    bonus = 8.0 * (power / max_power) * min(abs(momentum), 0.3)
```

**Κρίσιμο — Backward-looking**: Το momentum υπολογίζεται από **τιμές 6 ωρών πριν** (γνωστές), όχι μελλοντικές. Αυτό αποφεύγει data leakage.

---

## 11. Quartile Penalties — "Μην πουλάς φθηνά, μην αγοράζεις ακριβά"

```python
# I penalize selling at bottom-25% prices and buying at top-25% prices
percentile = calculate_24h_percentile(current_price)

if discharging and percentile < 25:
    penalty = -15.0  # Πουλάω σε χαμηλή τιμή — κακό!

elif charging and percentile > 75:
    penalty = -15.0  # Αγοράζω σε υψηλή τιμή — κακό!
```

**Παράδειγμα:**
```
Τελευταίες 24 ώρες: τιμές 40-150€
Percentile 25% = 67€
Percentile 75% = 123€

Discharge at 55€ → percentile=15 (Q1) → -15€ penalty!
Charge at 140€ → percentile=90 (Q4) → -15€ penalty!
Discharge at 130€ → percentile=82 (Q4) → μπόνους! (sell high)
```

---

## 12. DAM Compliance Bonus

```python
# I reward correct direction matching with DAM commitments
if dam_commitment > 0 and discharging:  # committed sell + actually selling
    bonus = compliance_bonus * coverage_ratio

elif dam_commitment < 0 and charging:   # committed buy + actually buying
    bonus = compliance_bonus * coverage_ratio
```

---

## 13. Κλιμάκωση: / 100

```python
# I scale the total reward for neural network stability
total_reward = raw_reward / 100.0
```

**Γιατί;**
- Τα neural networks δουλεύουν καλύτερα με rewards κοντά στο [-1, +1]
- Χωρίς κλιμάκωση, μία shortfall penalty (4500€) θα "εξαφανίσει" μικρά bonuses (2€)
- Η VecNormalize (Ημέρα 9) κάνει επιπλέον normalization

---

## 14. Ολοκληρωμένο Παράδειγμα

```
Ώρα: 14:00
SoC: 0.60
DAM commitment: +10MW (πώληση)
DAM price: 85€
Best bid: 83€
Best ask: 87€
mean_24h: 80€
momentum: +0.08 (ανεβαίνουν)
Action: discharge 20MW

1. DAM Revenue:      10 × 85 = +850€
2. Balancing:        10 × 83 = +830€ (extra 10MW @ bid)
3. Degradation:      20 × 0.5 = -10€
4. Shortfall:        0€ (delivered 20 > committed 10) ✓
5. Proactive Risk:   0€ (SoC=0.60, risk low) ✓
6. SoC Buffer:       +1.0€ (SoC=0.60, healthy)
7. Price Timing:     +5€ (85 > 80 mean, selling high)
8. Momentum:         +5.3€ (momentum +8%, discharging)
9. Quartile:         0€ (percentile ~60%, neutral)
10. DAM Compliance:  +3€ (correct direction)

Raw reward = 850 + 830 - 10 - 0 - 0 + 1 + 5 + 5.3 + 0 + 3 = 1684.3€
Scaled reward = 1684.3 / 100 = 16.84
```

---

## 15. Reward Shaping — Η Τέχνη

Η σχεδίαση reward function είναι **τέχνη, όχι επιστήμη**. Μερικά μαθήματα:

| Πρόβλημα | Λύση στο project |
|----------|-----------------|
| Agent αδράνεια (idle παντού) | Price timing bonus: "δες, αν πουλούσες ΤΩΡΑ, θα κέρδιζες" |
| Agent υπερδραστήριος | Degradation cost: "κάθε trade κοστίζει" |
| Αδειάζει μπαταρία χωρίς λόγο | Shortfall penalty + SoC buffer bonus |
| Αγοράζει ακριβά | Quartile penalty |
| Δεν ακολουθεί trends | Momentum bonus |
| Παραβιάζει commitments | 1500€/MWh shortfall penalty |

**Design principle: Dense vs Sparse rewards**
- **Sparse**: Μόνο κέρδος στο τέλος → δύσκολο να μάθει (πολύ μακριά feedback)
- **Dense**: Μικρά bonuses/penalties σε κάθε step → μαθαίνει γρήγορα ποιες μικρές αποφάσεις είναι καλές
- Το EntsoDRL χρησιμοποιεί **dense rewards** (11 components σε κάθε step)

---

## 16. Πρακτική Άσκηση

### Άσκηση Α: Υπολογισμός reward

Υπολόγισε τη reward με αυτά τα δεδομένα:
- Action: charge 15MW
- DAM commitment: -5MW (committed to buy)
- DAM price: 45€
- Best bid: 43€, Best ask: 47€
- mean_24h: 80€
- SoC: 0.30
- Shortfall risk: 0.10

Hints:
1. DAM Revenue = -5 × 45 = ?
2. Balancing = ? (actual buy = 15MW, committed = 5MW, excess = 10MW)
3. Degradation = 15 × 0.5 = ?
4. Price Timing: 45 < 80, charging → bonus or not?
5. SoC Buffer: 0.30 > 0.25 → small bonus

### Άσκηση Β: Σχεδίαση κινήτρων

Αν ο agent μαθαίνει να κάνει ΜΟΝΟ idle (action=10), ποιο reward component πρέπει να αυξήσεις;

Αν ο agent αγοράζει σε ακριβές τιμές, ποιο component βοηθάει;

### Άσκηση Γ: Ανάγνωση κώδικα

Άνοιξε `gym_envs/reward_calculator.py` και βρες:
1. Πού υπολογίζεται η shortfall penalty;
2. Πού γίνεται η κλιμάκωση / 100;
3. Ποιο component έχει τη μεγαλύτερη πιθανή τιμή;

---

## Σύνοψη Ημέρας

```
Reward Function = "Τι μαθαίνει ο agent"
├── Revenue (+)
│   ├── DAM Revenue: commitment × price
│   └── Balancing Revenue: imbalance × bid/ask
├── Costs (-)
│   ├── Degradation: |MW| × 0.5€
│   ├── Shortfall: gap × 1500€ (σοβαρή!)
│   └── Proactive Risk: penalty for low SoC
├── Shaping Bonuses
│   ├── SoC Buffer: healthy range bonus
│   ├── Price Timing: sell high, buy low
│   ├── Momentum: follow trends
│   ├── Quartile: avoid worst prices
│   └── DAM Compliance: honor commitments
└── Scale: / 100 for neural network
```

> **Αύριο (Ημέρα 5)**: Action Masking — πώς αποτρέπουμε τον agent από φυσικά αδύνατες ενέργειες. Αυτό είναι το pattern που κάνει το MaskablePPO ξεχωριστό.
