# Ημέρα 5: Action Masking — Φυσικοί Περιορισμοί στο RL

> **Στόχος**: Να καταλάβεις πώς και γιατί χρησιμοποιούμε action masking
> στο `MaskablePPO` — η κρίσιμη τεχνική που κάνει τον agent "ασφαλή".

---

## 1. Το Πρόβλημα: Αδύνατες Ενέργειες

Σκέψου αυτό το σενάριο:

```
SoC = 0.06 (σχεδόν άδεια μπαταρία)
Ο agent θέλει: action=0 → discharge 30MW (πούλα τα πάντα!)

Πρόβλημα: Η μπαταρία δεν έχει αρκετή ενέργεια!
Διαθέσιμη ενέργεια = (0.06 - 0.05) × 146 × √0.94 = 1.42 MWh
Max discharge = 1.42 MW (όχι 30!)
```

**Χωρίς action masking**, ο agent μπορεί να επιλέξει αυτό, να πάρει penalty, και να "μάθει" αργά να μην το κάνει. Αυτό είναι **αργό και ριψοκίνδυνο**.

**Με action masking**, αυτές οι επιλογές **δεν υπάρχουν καν** — ο agent δεν μπορεί καν να τις σκεφτεί.

---

## 2. Η Αναλογία: Σκάκι

Σκέψου το σκάκι:
- **Χωρίς masking**: Ο agent μπορεί να μετακινήσει τον πύργο διαγώνια, πληρώνει -1000 penalty, και σιγά-σιγά μαθαίνει ότι "αυτό δεν γίνεται"
- **Με masking**: Ο agent βλέπει ΜΟΝΟ τις νόμιμες κινήσεις. Ο πύργος δεν έχει καν διαγώνια option.

Το EntsoDRL κάνει ακριβώς αυτό: αν η μπαταρία είναι σχεδόν άδεια, τα discharge actions **εξαφανίζονται** από τις επιλογές.

---

## 3. Πώς Δουλεύει — Η Μάσκα

Η action mask είναι ένα binary array μεγέθους 21 (όσα τα actions):

```python
# Mask = array of 0s and 1s
# 1 = valid action, 0 = forbidden action

# Παράδειγμα: SoC = 0.90 (σχεδόν γεμάτη)
mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#       ^discharge actions^  ^idle^  ^forbidden charge actions^

# Action 0-9: discharge (OK, μπορεί να αδειάσει)
# Action 10: idle (ΠΑΝΤΑ OK)
# Action 11-20: charge (FORBIDDEN, θα ξεπεράσει 95%)
```

---

## 4. Ο Κώδικας — `action_masks()` method

```python
# Από gym_envs/battery_env_masked.py:
def action_masks(self) -> np.ndarray:
    """I compute which actions are physically valid right now."""
    mask = np.ones(self.n_actions, dtype=np.int8)  # Ξεκίνα: όλα valid

    # Πόση ενέργεια μπορώ να εκφορτίσω;
    available_discharge = (self.current_soc - self.min_soc) * self.capacity_mwh * np.sqrt(self.efficiency)

    # Πόση ενέργεια μπορώ να φορτίσω?
    available_charge = (self.max_soc - self.current_soc) * self.capacity_mwh / np.sqrt(self.efficiency)

    for i in range(self.n_actions):
        power_mw = self._action_to_mw(i)

        if power_mw < 0:  # Discharge
            # Χρειάζεται |power_mw| × 1h / √η MWh
            energy_needed = abs(power_mw) * 1.0 / np.sqrt(self.efficiency)
            if energy_needed > available_discharge:
                mask[i] = 0  # FORBIDDEN

        elif power_mw > 0:  # Charge
            # Χρειάζεται power_mw × 1h × √η MWh
            energy_needed = power_mw * 1.0 * np.sqrt(self.efficiency)
            if energy_needed > available_charge:
                mask[i] = 0  # FORBIDDEN

    # Idle (action=10) is ALWAYS valid
    mask[self.n_actions // 2] = 1

    return mask
```

---

## 5. Παραδείγματα Μασκών

### Παράδειγμα Α: Μπαταρία στο 50% (ελεύθερη)

```
SoC = 0.50
Available discharge = (0.50 - 0.05) × 146 × √0.94 = 63.7 MWh → 30MW OK
Available charge = (0.95 - 0.50) × 146 / √0.94 = 67.8 MWh → 30MW OK

Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                       ^ idle
Όλα τα actions είναι valid!
```

### Παράδειγμα Β: Μπαταρία στο 7% (σχεδόν άδεια)

```
SoC = 0.07
Available discharge = (0.07 - 0.05) × 146 × √0.94 = 2.83 MWh
Max discharge power = 2.83 MW (πολύ λίγο)

Actions σε MW:  [-30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0, 3, ...]
Mask:           [ 0,   0,   0,   0,   0,   0,   0,  0,  0,  1, 1, 1, ...]
                                                          ^  ^ idle
                                                     -3MW OK (< 2.83)

Μόνο: idle, charge, και discharge μέχρι ~3MW
```

### Παράδειγμα Γ: Μπαταρία στο 93% (σχεδόν γεμάτη)

```
SoC = 0.93
Available charge = (0.95 - 0.93) × 146 / √0.94 = 3.01 MWh
Max charge power = 3.01 MW

Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
       ^all discharge OK^           ^idle ^ ^+3 ^forbidden (>3MW)

Μπορεί: discharge τα πάντα, idle, ή charge μέχρι ~3MW
```

---

## 6. Πώς το MaskablePPO Χρησιμοποιεί τη Μάσκα

Κανονικά, ένα PPO model επιλέγει actions ως εξής:

```
1. Neural Network → 21 logits (π.χ. [2.1, 1.5, 0.3, ..., -0.8])
2. Softmax → 21 probabilities (π.χ. [0.15, 0.12, 0.08, ..., 0.01])
3. Sample → Επέλεξε action βάσει πιθανοτήτων
```

Με **MaskablePPO**, μετά το βήμα 1:

```
1. Neural Network → 21 logits: [2.1, 1.5, 0.3, ..., -0.8]
2. Apply mask:
   logits = [2.1, 1.5, 0.3, ..., -0.8]
   mask   = [0,   0,   0,   ...,  1  ]  (forbidden actions)

   masked_logits = [2.1, 1.5, 0.3, ..., -0.8]
   masked_logits[mask == 0] = -∞        ← Η μαγεία!

   masked_logits = [-∞, -∞, -∞, ..., -0.8]
3. Softmax → [-∞ γίνεται 0.0 probability]
   probs = [0.0, 0.0, 0.0, ..., 0.05]
4. Sample → Μόνο valid actions μπορούν να επιλεγούν!
```

**Κρίσιμο**: Τα masked actions παίρνουν πιθανότητα **ακριβώς 0** — δεν γίνεται καν κατά λάθος να επιλεγούν.

---

## 7. Action Masking vs Penalty — Γιατί όχι απλά penalty;

| Approach | Πλεονεκτήματα | Μειονεκτήματα |
|----------|-------------|-------------|
| **Penalty only** | Απλό | Αργή μάθηση, ριψοκίνδυνο, πιθανές παραβιάσεις |
| **Masking only** | Ασφαλές, γρήγορη μάθηση | Χρειάζεται special algorithm |
| **Masking + Penalty** ← EntsoDRL | Ασφαλές + ισχυρά κίνητρα | Πιο σύνθετο |

Το EntsoDRL χρησιμοποιεί **και τα δύο**:
- **Masking**: Αποτρέπει φυσικά αδύνατες ενέργειες (hard constraint)
- **Reward penalty**: Αποθαρρύνει κακές αλλά δυνατές ενέργειες (soft guidance)

---

## 8. Ο Safety Wrapper — `action_mask.py`

Εκτός από τη μάσκα στο environment, υπάρχει ένα **δεύτερο δίχτυ ασφαλείας**:

```python
# Από gym_envs/action_mask.py:
class ActionMasker(gym.ActionWrapper):
    """I provide a second safety layer on top of the environment mask."""

    def action(self, action):
        current_soc = self.env.unwrapped.current_soc

        if action.shape == (1,):
            raw_action = action[0]
            safe_action = self._mask_physical(raw_action, current_soc)
            return np.array([safe_action])

    def _mask_physical(self, raw_action, current_soc):
        safe_action = raw_action

        if current_soc >= 0.95 and raw_action < 0:
            safe_action = 0.0   # Block charging if full

        if current_soc <= 0.05 and raw_action > 0:
            safe_action = 0.0   # Block discharging if empty

        return safe_action
```

**Αντιστοιχία C#:**
```csharp
// Σαν Decorator pattern:
public class ActionMasker : IEnvironment
{
    private IEnvironment _inner;

    public StepResult Step(int action)
    {
        int safeAction = ApplyPhysicsConstraints(action);
        return _inner.Step(safeAction);
    }
}
```

---

## 9. Τρία Επίπεδα Ασφάλειας

Το project χρησιμοποιεί **defense in depth** — τρία επίπεδα:

```
Layer 1: MaskablePPO (model level)
  → Ο agent δεν μπορεί ΚΑΝ να επιλέξει invalid action
  → action_masks() method

Layer 2: ActionMasker (wrapper level)
  → Αν κάπως περάσει invalid action, γίνεται idle
  → _mask_physical() method

Layer 3: Physics Clamp (environment level)
  → SoC κόβεται στο [0.05, 0.95] με np.clip
  → Αν κάτι πάει στραβά, η μπαταρία δεν καταστρέφεται

Reward Penalty (learning level)
  → -5€ per violation
  → Ο agent μαθαίνει να αποφεύγει "οριακές" καταστάσεις
```

---

## 10. Στον Κώδικα Training — Πώς Συνδέονται

```python
# Από agent/train_masked.py:

# 1. Δημιουργία environment
env = BatteryEnvMasked(df, battery_params, n_actions=21)

# 2. Wrapper για masking (sb3_contrib pattern)
from sb3_contrib.common.wrappers import ActionMasker as SB3ActionMasker

def mask_fn(env):
    """I extract the action mask from the environment."""
    return env.action_masks()

env = SB3ActionMasker(env, mask_fn)

# 3. MaskablePPO αντί κανονικού PPO
from sb3_contrib import MaskablePPO

model = MaskablePPO(
    "MlpPolicy",
    env,
    # ... hyperparameters
)

# 4. Κατά το training, το model αυτόματα:
#    - Καλεί mask_fn(env) σε κάθε step
#    - Μασκάρει τα logits πριν το sampling
#    - Δεν υπολογίζει gradient για masked actions
```

**Κατά το evaluation:**

```python
# Από agent/evaluate_masked.py:
obs = env.reset()
done = False

while not done:
    # Παίρνω τη μάσκα
    mask = env.action_masks()  # [1, 0, 0, ..., 1, 1, 1]

    # Ο agent αποφασίζει ΜΟΝΟ ανάμεσα σε valid actions
    action, _ = model.predict(
        obs,
        deterministic=True,        # Πάντα η "καλύτερη" επιλογή
        action_masks=np.array([mask])  # Πέρνα τη μάσκα
    )

    obs, reward, done, info = env.step(action)
```

---

## 11. Γιατί `MaskablePPO` (sb3-contrib) και όχι κανονικό PPO

| Feature | PPO (Stable Baselines 3) | MaskablePPO (sb3-contrib) |
|---------|--------------------------|---------------------------|
| Invalid actions | Μαθαίνει μέσω penalty (αργό) | Αποκλείονται εντελώς (γρήγορο) |
| Exploration | Μπορεί να "χαθεί" σε invalid actions | Εξερευνεί μόνο νόμιμο χώρο |
| Safety | Δεν εγγυάται τίποτα | 100% valid actions |
| Training speed | Αργό (σπαταλά samples σε invalid) | Γρήγορο (κάθε sample μετράει) |
| Import | `from stable_baselines3 import PPO` | `from sb3_contrib import MaskablePPO` |

---

## 12. Πρακτική Άσκηση

### Άσκηση Α: Υπολόγισε τη μάσκα

Δεδομένα: SoC=0.12, capacity=146MWh, efficiency=0.94, max_power=30MW, n_actions=21

1. Available discharge energy = (0.12 - 0.05) × 146 × √0.94 = ?
2. Max discharge power (1h) = ? MW
3. Ποια discharge actions (0-9) πρέπει να γίνουν 0 στη μάσκα;

Hints:
- Action i → MW = (i - 10) / 10 × 30
- Action 9 → -3MW, Action 8 → -6MW, Action 7 → -9MW, ...

### Άσκηση Β: Edge case

Τι μάσκα παράγεται αν SoC = 0.05 (ακριβώς στο minimum);
- Available discharge = (0.05 - 0.05) × ... = 0
- Ποια actions είναι valid;

### Άσκηση Γ: Ανάγνωση κώδικα

Άνοιξε `gym_envs/battery_env_masked.py` και βρες τη μέθοδο `action_masks()`:
1. Πώς υπολογίζεται η `available_discharge`;
2. Τι γίνεται αν `n_actions` αλλάξει σε 41 αντί 21;
3. Πού διασφαλίζεται ότι idle είναι ΠΑΝΤΑ valid;

---

## Σύνοψη Ημέρας

```
Action Masking = "Φυσικοί Νόμοι στο RL"
├── Problem: Agent μπορεί να επιλέξει αδύνατες ενέργειες
├── Solution: Binary mask [1,0,0,...,1,1] per step
├── Implementation:
│   ├── action_masks() → Υπολογισμός valid actions
│   ├── MaskablePPO → Logits masking (-∞ for invalid)
│   └── ActionMasker wrapper → Secondary safety net
├── Three safety layers:
│   ├── Layer 1: Model-level masking (probability = 0)
│   ├── Layer 2: Wrapper clipping (force idle)
│   └── Layer 3: Physics clamping (np.clip SoC)
└── Result: 100% physically valid actions, faster training
```

> **Αύριο (Ημέρα 6)**: Neural Networks & PyTorch — πώς ένα νευρωνικό δίκτυο "βλέπει" 45 αριθμούς και αποφασίζει τι να κάνει.
