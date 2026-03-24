# Ημέρα 9: Evaluation — Πώς Ξέρουμε ότι ο Agent Δουλεύει

> **Στόχος**: Να καταλάβεις πώς αξιολογούμε τον trained agent,
> τα pitfalls του VecNormalize, και τις μετρικές που μετράνε.

---

## 1. Η Μεγάλη Ερώτηση: "Δουλεύει;"

Ένα υψηλό training reward ΔΕΝ σημαίνει ότι ο agent είναι καλός. Πρέπει:

```
Training Reward ↑     →  Μαθαίνει? Ίσως.
Eval Reward ↑         →  Γενικεύει? Πιθανώς.
Beats Benchmark?      →  Καλύτερος από rule-based? Ναι!
Real-world Profit ↑   →  Production-ready? Ναι!
```

---

## 2. Train/Test Split

```python
# ΠΟΤΕ δεν αξιολογούμε στα ίδια δεδομένα που εκπαιδεύτηκε!

df = pd.read_csv('data/feasible_data_with_dam.csv')

# 80% training / 20% testing
split = int(len(df) * 0.8)
train_df = df.iloc[:split]     # Jan 2024 - Oct 2024
test_df = df.iloc[split:]      # Nov 2024 - Dec 2024

# Ο agent ΠΟΤΕ δεν είδε τα test data κατά τη διάρκεια training
```

**Αντιστοιχία C#:**
```csharp
// Σαν unit testing: δεν δοκιμάζεις με τα ίδια data που χρησιμοποίησες
// για development — χρησιμοποιείς ξεχωριστό test set
```

---

## 3. Evaluation Script — `evaluate_masked.py`

```python
# Από agent/evaluate_masked.py:

# 1. Φόρτωσε model + VecNormalize
model = MaskablePPO.load("models/best_model.zip")
eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize.load("models/vec_normalize.pkl", eval_env)

# ΚΡΙΣΙΜΟ: Σωστές ρυθμίσεις!
eval_env.training = False       # ΜΗΝ ενημερώνεις τα stats
eval_env.norm_reward = False    # ΜΗΝ normalize τα rewards

# 2. Τρέξε evaluation loop
obs = eval_env.reset()
done = False
total_reward = 0
step_count = 0
violations = 0

while not done:
    # Πάρε action mask
    mask = eval_env.envs[0].action_masks()

    # Deterministic prediction (πάντα η "καλύτερη" action)
    action, _ = model.predict(
        obs,
        deterministic=True,
        action_masks=np.array([mask])
    )

    obs, reward, done, info = eval_env.step(action)
    total_reward += reward[0]
    step_count += 1

    if info[0].get('violation', 0) > 0.5:
        violations += 1

# 3. Αποτελέσματα
print(f"Total Reward: {total_reward:.2f}")
print(f"Steps: {step_count}")
print(f"Violations: {violations} ({violations/step_count*100:.1f}%)")
```

---

## 4. VecNormalize Pitfalls — Τα ΚΛΑΣΙΚΑ λάθη

### Pitfall 1: Ξέχασες να φορτώσεις VecNormalize

```python
# ΛΑΘΟΣ — Observations χωρίς normalization:
model = MaskablePPO.load("model.zip")
obs = raw_env.reset()
action = model.predict(obs)  # Ο agent βλέπει ΜΗ normalized data!
                              # Τα weights εκπαιδεύτηκαν σε normalized data!
                              # → Τυχαίες αποφάσεις!

# ΣΩΣΤΟ:
env = VecNormalize.load("vec_normalize.pkl", raw_env)
env.training = False
obs = env.reset()
action = model.predict(obs)  # Τώρα βλέπει normalized data ✓
```

### Pitfall 2: Αφήνεις training=True κατά evaluation

```python
# ΛΑΘΟΣ:
env = VecNormalize.load("vec_normalize.pkl", raw_env)
# env.training = True (default!)
# → Τα running mean/std αλλάζουν κατά evaluation!
# → Τα πρώτα steps είναι OK, αλλά σταδιακά χαλάνε!

# ΣΩΣΤΟ:
env.training = False   # "Πάγωσε" τα statistics
env.norm_reward = False # Θέλουμε raw rewards για μέτρηση
```

### Pitfall 3: Διαφορετικό observation space

```python
# ΛΑΘΟΣ — Training με 45 features, evaluation με 16:
train_env = BatteryEnvMasked(train_df, features=45)  # v22
eval_env = BatteryEnvMasked(eval_df, features=16)    # v15 (!)
# → Crash ή σιωπηλά λάθος αποτελέσματα!

# ΣΩΣΤΟ — Ίδιο configuration:
eval_env = BatteryEnvMasked(eval_df, features=45)  # v22 ✓
```

---

## 5. Μετρικές Αξιολόγησης

### 5.1 Βασικές Μετρικές

```python
# Από evaluate_detailed.py:

# Total Reward — Αθροιστικό reward (δεν είναι €, είναι scaled!)
total_reward = sum(rewards)

# Total Profit — Πραγματικό κέρδος σε €
total_profit = sum(info['dam_revenue'] + info['balancing_revenue']
                   - info['degradation_cost'])

# Violations — Φυσικές παραβιάσεις
violation_rate = violations / total_steps * 100  # θέλουμε 0%

# Trade Rate — Πόσο συχνά κάνει trade
trades = sum(1 for a in actions if abs(a) > 0.1)
trade_rate = trades / total_steps * 100  # 30-50% ιδανικό
```

### 5.2 Action Distribution

```python
# Ανάλυση τι κάνει ο agent:
discharge_pct = sum(1 for a in actions if a < -0.1) / len(actions)
idle_pct = sum(1 for a in actions if abs(a) <= 0.1) / len(actions)
charge_pct = sum(1 for a in actions if a > 0.1) / len(actions)

print(f"Discharge: {discharge_pct:.1%}")   # ~15-25%
print(f"Idle: {idle_pct:.1%}")             # ~50-70%
print(f"Charge: {charge_pct:.1%}")         # ~15-25%
```

**Τι σημαίνει:**
```
Healthy agent:     Discharge 20% | Idle 60% | Charge 20%
Too aggressive:    Discharge 40% | Idle 20% | Charge 40%  ← πολύ trading!
Too passive:       Discharge 5%  | Idle 90% | Charge 5%   ← δεν κάνει τίποτα!
Biased:           Discharge 35% | Idle 50% | Charge 15%  ← πουλάει μόνο!
```

### 5.3 Benchmark Comparison

```python
# Σύγκριση με Rule-Based Strategy:
rule_based_profit = evaluate_strategy(RuleBasedStrategy(), eval_df)
ai_profit = evaluate_strategy(RLAgentStrategy(model), eval_df)

improvement = (ai_profit - rule_based_profit) / abs(rule_based_profit) * 100
print(f"AI vs Rule-Based: +{improvement:.1f}%")
```

**Benchmarks στο project:**
```
Strategy            | Annual Profit (est.) | Notes
--------------------|---------------------|------
ConservativeStrategy| 0€                  | Always idle
RandomStrategy      | -50K€               | Loses money
RuleBasedStrategy   | +2-3M€              | Price thresholds
RLAgentStrategy     | +5-9M€              | Learned patterns
AggressiveHybrid    | +8.9M€              | Best rule-based
```

---

## 6. Detailed Trade Analysis

```python
# Από evaluate_detailed.py:

trades_log = []
for step in range(len(eval_df)):
    if abs(actual_mw) > 0.1:  # Non-idle action
        trades_log.append({
            'step': step,
            'hour': current_hour,
            'action_mw': actual_mw,
            'soc': current_soc,
            'dam_price': dam_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'pnl': dam_revenue + balancing_revenue - degradation,
        })

# Top 10 profitable trades:
top_trades = sorted(trades_log, key=lambda t: t['pnl'], reverse=True)[:10]

# Bottom 10 losing trades:
worst_trades = sorted(trades_log, key=lambda t: t['pnl'])[:10]
```

**Παράδειγμα output:**
```
Top 10 Profitable Trades:
  Step 1420 | Hour 18 | Discharge -30MW | SoC 0.62 | Price 185€ | PnL +5,550€
  Step 1421 | Hour 19 | Discharge -30MW | SoC 0.41 | Price 172€ | PnL +5,160€
  ...

Bottom 10 Losing Trades:
  Step 3201 | Hour 03 | Discharge -15MW | SoC 0.15 | Price 32€  | PnL -480€
  Step 5002 | Hour 14 | Charge +20MW    | SoC 0.88 | Price 140€ | PnL -2,800€
```

---

## 7. Daily Profit Analysis

```python
# Πόσα κερδίζει/χάνει ανά ημέρα:
daily_profits = {}
for trade in trades_log:
    day = trade['step'] // 24
    if day not in daily_profits:
        daily_profits[day] = 0
    daily_profits[day] += trade['pnl']

# Στατιστικά:
profits = list(daily_profits.values())
win_days = sum(1 for p in profits if p > 0)
loss_days = sum(1 for p in profits if p <= 0)
win_rate = win_days / len(profits) * 100

avg_win = np.mean([p for p in profits if p > 0])
avg_loss = np.mean([p for p in profits if p <= 0])
sharpe = np.mean(profits) / np.std(profits) * np.sqrt(252)  # Annualized

print(f"Win Rate: {win_rate:.1f}%")           # >65% good
print(f"Avg Win: {avg_win:.0f}€")             # Should be > |Avg Loss|
print(f"Avg Loss: {avg_loss:.0f}€")
print(f"Sharpe Ratio: {sharpe:.2f}")           # >1.5 good, >2.0 excellent
```

---

## 8. Overfitting Detection

**Πώς ξέρεις αν ο agent "αποστήθισε" τα training data:**

```
Training Reward: 500    ← Υψηλό
Eval Reward: 500        ← Επίσης υψηλό → OK, γενικεύει ✓

Training Reward: 800    ← Πολύ υψηλό
Eval Reward: 200        ← Χαμηλό → OVERFITTING! ✗
```

**Λύσεις overfitting:**
1. Μείωσε network size: `[256, 128]` αντί `[512, 256, 128]`
2. Αύξησε data: Χρησιμοποίησε περισσότερα χρόνια δεδομένων
3. Μείωσε n_epochs: 5 αντί 10
4. Αύξησε entropy: `ent_coef=0.1` αντί `0.05`

---

## 9. Deterministic vs Stochastic Evaluation

```python
# Deterministic — ΠΑΝΤΑ η πιο πιθανή action:
action, _ = model.predict(obs, deterministic=True)
# Χρήση: Production, evaluation
# Πλεονέκτημα: Αναπαραγώγιμα αποτελέσματα

# Stochastic — Sample βάσει πιθανοτήτων:
action, _ = model.predict(obs, deterministic=False)
# Χρήση: Training (exploration), stress testing
# Πλεονέκτημα: Εξερεύνηση, ποικιλία

# Για σωστή αξιολόγηση: ΠΑΝΤΑ deterministic=True
```

---

## 10. Production Metrics Callback

```python
# Από agent/callbacks/production_metrics_callback.py:

class ProductionMetricsCallback(BaseCallback):
    """I track real production metrics during training."""

    def _evaluate(self):
        metrics = {
            'dam_compliance': self._calc_dam_compliance(),    # % commitment met
            'peak_discharge': self._calc_peak_discharge(),    # % peak hours active
            'trade_rate': self._calc_trade_rate(),             # % active steps
            'price_correlation': self._calc_price_corr(),      # price-action corr
            'avg_daily_profit': self._calc_daily_profit(),     # € per day
            'win_rate': self._calc_win_rate(),                 # % profitable days
            'sharpe_ratio': self._calc_sharpe(),               # Risk-adjusted
            'profit_per_mwh': self._calc_efficiency(),         # €/MWh traded
        }

        for key, value in metrics.items():
            self.logger.record(f'production/{key}', value)
```

**Κρίσιμες μετρικές:**

| Μετρική | Target | Σημασία |
|---------|--------|---------|
| DAM Compliance | >90% | Τηρεί τα commitments; |
| Peak Discharge | >60% | Πουλάει σε peak hours; |
| Trade Rate | 30-50% | Ούτε πολύ, ούτε λίγο |
| Price Correlation | >0.3 | Πουλάει ακριβά, αγοράζει φθηνά; |
| Win Rate | >65% | Κερδοφόρες μέρες; |
| Sharpe Ratio | >1.5 | Risk-adjusted return; |

---

## 11. Πρακτική Άσκηση

### Άσκηση Α: VecNormalize

Ποιο λάθος κρύβεται εδώ;
```python
model = MaskablePPO.load("model.zip")
env = DummyVecEnv([make_eval_env])
# env = VecNormalize.load(...)  ← σχολιασμένο!
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

### Άσκηση Β: Ερμηνεία αποτελεσμάτων

```
Results:
  Total Reward: 12,500
  Violations: 0 (0.0%)
  Action Distribution: Discharge 5% | Idle 90% | Charge 5%
  AI Profit: €1.2M
  Rule-Based Profit: €2.5M
```

Ο agent είναι "καλός" ή "κακός"; Τι πρέπει να αλλάξεις;

### Άσκηση Γ: Ανάγνωση κώδικα

1. Στο `agent/evaluate_masked.py`: πού γίνεται `training=False`;
2. Στο `agent/evaluate_detailed.py`: ποια στοιχεία αποθηκεύονται ανά trade;
3. Πού γίνεται η σύγκριση με rule-based;

---

## Σύνοψη Ημέρας

```
Evaluation = "Δοκιμή στο πεδίο"
├── Setup:
│   ├── Test data (unseen during training!)
│   ├── VecNormalize.load() with training=False
│   └── deterministic=True prediction
├── Metrics:
│   ├── Total Reward / Profit
│   ├── Violation Rate (→ 0%)
│   ├── Action Distribution (balanced?)
│   ├── Win Rate (→ >65%)
│   ├── Sharpe Ratio (→ >1.5)
│   └── Benchmark comparison (vs Rule-Based)
├── Pitfalls:
│   ├── Missing VecNormalize → garbage predictions
│   ├── training=True → shifting statistics
│   └── Mismatched observation space → crash/wrong
└── Analysis:
    ├── Trade-by-trade P&L breakdown
    ├── Daily profit distribution
    └── Overfitting detection (train vs eval gap)
```

> **Αύριο (Ημέρα 10)**: Production & Full Picture — πώς πάμε από trained model σε real-world trading, paper trading, και η αρχιτεκτονική από αρχή μέχρι τέλος.
