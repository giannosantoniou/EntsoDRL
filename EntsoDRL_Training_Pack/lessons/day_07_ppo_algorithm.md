# Ημέρα 7: PPO Algorithm — Πώς Μαθαίνει ο Agent

> **Στόχος**: Να καταλάβεις τον PPO αλγόριθμο σε intuitive επίπεδο,
> και πώς κάθε hyperparameter στο `train_masked.py` επηρεάζει τη μάθηση.

---

## 1. Η Μεγάλη Ιδέα: Trial & Error + Memory

Φαντάσου ότι μαθαίνεις να παίζεις darts:

```
Βολή 1: Στόχος αριστερά → "Πρέπει να ρίξω πιο δεξιά"
Βολή 2: Πιο δεξιά → Bullseye! → "Αυτό ήταν σωστό!"
Βολή 3: Ακόμα πιο δεξιά → Έξω → "Πήγα πολύ, πίσω λίγο"
```

Ο PPO κάνει ακριβώς αυτό:
1. **Δοκιμάζει** ενέργειες (rollout)
2. **Αξιολογεί** τα αποτελέσματα (reward)
3. **Ενημερώνει** τα weights (gradient update)
4. Επαναλαμβάνει

---

## 2. Τα Τρία Βήματα του PPO

```
┌────────────────────────────────────────────────────────┐
│                    PPO Training Loop                    │
│                                                        │
│  Step 1: COLLECT (Rollout)                             │
│  ├── Παίξε n_steps=4096 steps στο environment          │
│  ├── Κάθε step: obs → action → reward → next_obs       │
│  └── Αποθήκευσε τα πάντα σε buffer                     │
│                                                        │
│  Step 2: COMPUTE (Advantages)                          │
│  ├── Για κάθε step: πόσο ΚΑΛΥΤΕΡΟ ήταν                 │
│  │   το reward σε σχέση με το αναμενόμενο;             │
│  └── advantage = actual_return - predicted_value        │
│                                                        │
│  Step 3: UPDATE (Optimization)                         │
│  ├── Χρησιμοποίησε τα advantages                       │
│  ├── Ενημέρωσε Actor (policy) weights                  │
│  ├── Ενημέρωσε Critic (value) weights                  │
│  └── Κάνε n_epochs=10 passes πάνω στα δεδομένα         │
│                                                        │
│  Repeat until total_timesteps reached                  │
└────────────────────────────────────────────────────────┘
```

---

## 3. Step 1: Rollout — Συλλογή Εμπειρίας

```python
# Αυτό γίνεται αυτόματα από τον SB3:
# n_steps = 4096 σημαίνει: "Παίξε 4096 steps πριν κάνεις update"

buffer = []
for step in range(4096):
    action = actor.predict(obs)          # Τι λέει το δίκτυο
    next_obs, reward, done, info = env.step(action)
    value = critic.predict(obs)          # Πόσο αξίζει αυτή η κατάσταση;
    buffer.append((obs, action, reward, value, done))
    obs = next_obs
```

**Γιατί 4096 steps;**
- Πολύ μικρό (64): Ο agent βλέπει μόνο λίγες ώρες → δεν πιάνει daily patterns
- Πολύ μεγάλο (65536): Πολύ αργές updates → αργή μάθηση
- 4096 ≈ 170 ημέρες αγοράς → πιάνει εβδομαδιαία + daily patterns

---

## 4. Step 2: Advantage — "Πόσο καλύτερα πήγε;"

Η βασική ερώτηση: "Αυτό που έκανε ο agent ήταν ΚΑΛΥΤΕΡΟ ή ΧΕΙΡΟΤΕΡΟ από το μέσο;"

```
Advantage = "Τι πήρα" - "Τι ΠΕΡΙΜΕΝΑ να πάρω"
          = Actual Return - Predicted Value

Παράδειγμα:
  Step 100: Ο critic λέει "αυτή η κατάσταση αξίζει +50"
            Ο agent κάνει discharge, παίρνει actual return = +80

  Advantage = 80 - 50 = +30
  Σημασία: "Αυτό που έκανα ήταν 30 καλύτερο από το μέσο!"
  → Ενίσχυσε αυτή τη συμπεριφορά

  Αλλιώς:
  Ο agent κάνει idle, παίρνει actual return = +20
  Advantage = 20 - 50 = -30
  → "Αυτό ήταν 30 χειρότερο" → Αποδυνάμωσε αυτή τη συμπεριφορά
```

### GAE (Generalized Advantage Estimation)

```python
# gae_lambda = 0.95 στο project
```

Η GAE κάνει smoothing στα advantages — ισορροπεί μεταξύ:
- **Άμεσο reward** (τι πήρα τώρα): low variance, high bias
- **Μακροπρόθεσμο return** (τι θα πάρω συνολικά): high variance, low bias

`gae_lambda = 0.95` → δίνει βάρος στο μακροπρόθεσμο (ταιριάζει στο trading)

---

## 5. Step 3: Policy Update — Η "Μαγεία" του PPO

Εδώ είναι η καρδιά του PPO. Η βασική ιδέα:

```
ΑΝ advantage > 0: Αύξησε την πιθανότητα αυτής της action
ΑΝ advantage < 0: Μείωσε την πιθανότητα αυτής της action
ΑΛΛΑ μην αλλάξεις τα weights ΠΟΛΥ γρήγορα (clipping)
```

### Clipping — Το μυστικό του PPO

```python
# clip_range = 0.2 στο project
```

Χωρίς clipping (vanilla Policy Gradient):
```
"Αυτή η action ήταν πολύ καλή (advantage=+100)!"
→ Αλλάζω τα weights δραστικά
→ Ο agent "ξεχνά" τα πάντα
→ Αστάθεια, collapse
```

Με PPO clipping:
```
"Αυτή η action ήταν πολύ καλή (advantage=+100)!"
→ Θέλω να αλλάξω τα weights πολύ
→ ΑΛΛΑ: ratio = new_prob / old_prob ≤ 1.2 (clipped!)
→ Μόνο 20% αλλαγή max ανά update
→ Σταθερή, αξιόπιστη μάθηση
```

**Αντιστοιχία**: Σαν ένα speed limiter σε αυτοκίνητο — θέλεις να πας γρήγορα, αλλά δεν σε αφήνει να ξεπεράσεις τα 120km/h.

---

## 6. Hyperparameters — Τι Σημαίνει Κάθε Ένα

```python
# Από train_masked.py:
model = MaskablePPO(
    "MlpPolicy",
    env,
    learning_rate=linear_schedule(3e-4),  # Πόσο γρήγορα μαθαίνει
    n_steps=4096,                         # Steps πριν κάθε update
    batch_size=256,                       # Μέγεθος mini-batch
    n_epochs=10,                          # Πόσα passes στα δεδομένα
    gamma=0.99,                           # Discount factor
    gae_lambda=0.95,                      # GAE smoothing
    clip_range=0.2,                       # PPO clipping
    ent_coef=0.05,                        # Entropy bonus
    policy_kwargs=dict(
        net_arch=[512, 256, 128]          # Network architecture
    ),
)
```

### 6.1 `learning_rate = 3e-4` (0.0003)

```
Πόσο αλλάζουν τα weights σε κάθε update step.

Πολύ μεγάλο (0.01): Μαθαίνει γρήγορα αλλά ασταθώς (overshoot)
Σωστό (0.0003): Σταθερή πρόοδος
Πολύ μικρό (0.000001): Δεν μαθαίνει ποτέ (πολύ αργό)

linear_schedule: Ξεκινά 3e-4, τελειώνει 0 (σταδιακά μειώνεται)
Ιδέα: Αρχικά μάθε γρήγορα, στο τέλος "fine-tune"
```

### 6.2 `gamma = 0.99` (Discount Factor)

```
Πόσο νοιάζεται ο agent για ΜΕΛΛΟΝΤΙΚΑ rewards.

gamma = 0.0:  Μόνο τώρα μετράει (myopic)
gamma = 0.99: Μετράει μέχρι ~100 steps μπροστά
gamma = 1.0:  Όλο το μέλλον μετράει εξίσου

Effective horizon ≈ 1 / (1 - gamma) = 1 / 0.01 = 100 steps = 100 ώρες ≈ 4 ημέρες

Γιατί 0.99: Η μπαταρία χρειάζεται ~5h για full cycle.
Πρέπει να "βλέπει" αρκετές ώρες μπροστά για planning.
```

**Παράδειγμα:**
```
Step 0: reward = +10
Step 1: reward = +20
Step 2: reward = -5

Return from Step 0 = 10 + 0.99×20 + 0.99²×(-5)
                    = 10 + 19.8 + (-4.9)
                    = 24.9

Χωρίς discounting (gamma=1.0):
Return = 10 + 20 + (-5) = 25

Με πολύ discounting (gamma=0.5):
Return = 10 + 0.5×20 + 0.25×(-5) = 10 + 10 - 1.25 = 18.75
```

### 6.3 `n_epochs = 10`

```
Πόσες φορές χρησιμοποιούμε τα ίδια δεδομένα πριν τα πετάξουμε.

n_epochs=1: Χρήση μία φορά, πέταξε (sample inefficient)
n_epochs=10: Μάθε 10 φορές από κάθε batch (data efficient)
n_epochs=50: Overfitting σε αυτό το batch (κακό)
```

### 6.4 `ent_coef = 0.05` (Entropy Coefficient)

```
Πόσο "τυχαίος" πρέπει να είναι ο agent.

ent_coef = 0.0:  Ο agent κάνει πάντα το ίδιο (exploitation only)
ent_coef = 0.05: Λίγη τυχαιότητα (balanced)
ent_coef = 1.0:  Σχεδόν τυχαίος (exploration only)

Entropy = μέτρο τυχαιότητας
  - Αν probs = [0.0, 0.0, ..., 1.0, 0.0]: Entropy = 0 (σίγουρος)
  - Αν probs = [1/21, 1/21, ..., 1/21]:   Entropy = max (τυχαίος)

ent_coef bonus: reward += ent_coef × entropy
→ Ο agent παίρνει μπόνους για να "δοκιμάζει" νέα πράγματα
→ Αποτρέπει πρόωρο "κόλλημα" σε μία στρατηγική
```

---

## 7. Linear Schedule — Μείωση Learning Rate

```python
# Από train_masked.py:
def linear_schedule(initial_value):
    """I linearly decrease the learning rate from initial to 0."""
    def func(progress_remaining):
        # progress_remaining: 1.0 → 0.0 κατά τη διάρκεια training
        return progress_remaining * initial_value
    return func

# Χρήση:
learning_rate = linear_schedule(3e-4)
# Step 0:         lr = 3e-4 (fast learning)
# Step 5M:        lr = 1.5e-4 (moderate)
# Step 10M:       lr = 0 (fine-tuned, σταμάτα)
```

**Αναλογία**: Σαν να μαθαίνεις ζωγραφική — αρχικά χρησιμοποιείς μεγάλο πινέλο (γενικές μορφές), μετά μικρό (λεπτομέρειες).

---

## 8. Batch Size vs n_steps — Η Σχέση

```
n_steps = 4096: Συλλέγω 4096 transitions
batch_size = 256: Χωρίζω σε mini-batches των 256

4096 / 256 = 16 mini-batches per epoch
n_epochs = 10
→ 16 × 10 = 160 gradient updates per rollout
```

```
Rollout (4096 steps)
├── Epoch 1
│   ├── Mini-batch 1 (256 samples) → gradient update
│   ├── Mini-batch 2 (256 samples) → gradient update
│   ├── ...
│   └── Mini-batch 16 (256 samples) → gradient update
├── Epoch 2 (ίδια data, shuffled)
│   ├── ...
├── ...
└── Epoch 10
    └── ... → gradient update

Total: 160 weight updates per 4096 environment steps
```

---

## 9. Training Timeline — Τι Γίνεται σε 10M Steps

```
Steps 0 - 500K:      Random exploration
                      Agent κάνει τυχαίες ενέργειες
                      Reward: ~0 (ούτε κέρδος ούτε ζημιά)

Steps 500K - 2M:      Βασική μάθηση
                      Μαθαίνει: "idle > random trading"
                      Μαθαίνει: "μην αδειάζεις τη μπαταρία εντελώς"
                      Reward: αρχίζει να ανεβαίνει

Steps 2M - 5M:        Pattern recognition
                      Μαθαίνει: "sell high, buy low"
                      Μαθαίνει: daily price patterns
                      Reward: σταθερά ανεβαίνει

Steps 5M - 8M:        Fine-tuning
                      Μαθαίνει: momentum signals
                      Μαθαίνει: DAM compliance
                      Μαθαίνει: optimal SoC management
                      Reward: plateau με μικρές βελτιώσεις

Steps 8M - 10M:       Convergence
                      Σταθεροποίηση στρατηγικής
                      learning_rate → 0
                      Reward: σταθερό
```

---

## 10. On-Policy vs Off-Policy

Ο PPO είναι **on-policy** αλγόριθμος:

```
On-Policy (PPO):
  1. Μάζεψε εμπειρία με ΤΡΕΧΟΥΣΑ πολιτική
  2. Μάθε από αυτή
  3. ΠΕΤΑΞΕ τα δεδομένα
  4. Μάζεψε νέα εμπειρία με ΕΝΗΜΕΡΩΜΕΝΗ πολιτική
  → Χρειάζεται ΠΟΛΛΑ δεδομένα (sample inefficient)
  → Αλλά: σταθερό, αξιόπιστο

Off-Policy (DQN, SAC):
  1. Μάζεψε εμπειρία, βάλε σε Replay Buffer
  2. Μάθε από τυχαία δείγματα του buffer
  3. ΚΡΑΤΑ τα παλιά δεδομένα
  → Χρειάζεται ΛΙΓΟΤΕΡΑ δεδομένα (sample efficient)
  → Αλλά: πιο ασταθές, πιο δύσκολο tuning
```

**Γιατί PPO στο EntsoDRL:** Σταθερότητα > Sample Efficiency. Τα data (historical prices) είναι δωρεάν, η σταθερότητα μετράει.

---

## 11. Πρακτική Άσκηση

### Άσκηση Α: Hyperparameter Intuition

Αν ο agent μαθαίνει "αργά" (reward ανεβαίνει αργά μετά 5M steps):
- Τι θα αλλάξεις; (Hint: learning_rate, n_steps)

Αν ο agent "κολλάει" σε idle (κάνει πάντα action=10):
- Τι θα αυξήσεις; (Hint: ent_coef)
- Τι θα ελέγξεις στη reward function; (Hint: price timing bonus)

### Άσκηση Β: Discount Factor

Υπολόγισε τον discounted return:

```
gamma = 0.99
Rewards: step0=+10, step1=-5, step2=+20, step3=+15

Return(step0) = 10 + 0.99×(-5) + 0.99²×20 + 0.99³×15 = ?
```

### Άσκηση Γ: Ανάγνωση κώδικα

Άνοιξε `agent/train_masked.py`:
1. Βρες όλα τα hyperparameters (learning_rate, gamma, κλπ.)
2. Βρες πού ορίζεται η `linear_schedule`
3. Πόσα total_timesteps τρέχει;
4. Κάθε πόσα steps γίνεται checkpoint;

---

## Σύνοψη Ημέρας

```
PPO = "Safe Trial & Error Learning"
├── Rollout: Παίξε n_steps=4096 steps
├── Compute Advantages: "Πόσο καλύτερα πήγε;"
├── Update Weights:
│   ├── Actor: ↑ πιθανότητα καλών actions
│   ├── Critic: Βελτίωσε value estimates
│   └── Clipping: Μην αλλάζεις πολύ (±20%)
├── Key Hyperparameters:
│   ├── learning_rate: 3e-4 → 0 (linear schedule)
│   ├── gamma: 0.99 (~100h horizon)
│   ├── n_steps: 4096 (rollout size)
│   ├── batch_size: 256 (mini-batch)
│   ├── n_epochs: 10 (passes per rollout)
│   ├── clip_range: 0.2 (stability)
│   └── ent_coef: 0.05 (exploration)
└── On-Policy: Μάζεψε → Μάθε → Πέταξε → Repeat
```

> **Αύριο (Ημέρα 8)**: Training Pipeline — Callbacks, Curriculum Learning, και πώς ελέγχουμε τη διαδικασία εκπαίδευσης.
