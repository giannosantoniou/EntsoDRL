# Ημέρα 8: Training Pipeline — Callbacks, Curriculum, και Monitoring

> **Στόχος**: Να καταλάβεις πώς ελέγχεται η εκπαίδευση του agent —
> callbacks, curriculum learning, checkpoints, και TensorBoard monitoring.

---

## 1. Η Μεγάλη Εικόνα: Training Pipeline

```
Data (CSV) → Environment → VecNormalize → MaskablePPO → Callbacks
                                                            │
                                         ┌──────────────────┼─────────────────┐
                                         │                  │                 │
                                    Checkpoint       Curriculum        TensorBoard
                                    Callback          Callback          Logging
                                    (save model)     (adjust reward)   (metrics)
```

```python
# Από train_masked.py — η πλήρης ροή:

# 1. Φόρτωσε δεδομένα
df = pd.read_csv('data/feasible_data_with_balancing.csv', parse_dates=['timestamp'])

# 2. Δημιούργησε environment
env = make_env(df, battery_params)
env = DummyVecEnv([lambda: env])            # Vectorized wrapper
env = VecNormalize(env, norm_obs=True, norm_reward=True)  # Normalize

# 3. Δημιούργησε model
model = MaskablePPO("MlpPolicy", env, **hyperparams)

# 4. Πρόσθεσε callbacks
callbacks = [
    CheckpointCallback(save_freq=200_000, save_path='./models/checkpoints/'),
    DegradationCurriculumCallback(warmup_steps=500_000, target_deg=15.0),
    # ProductionMetricsCallback(eval_freq=50_000, ...),
]

# 5. ΤΡΕΞΕ!
model.learn(total_timesteps=10_000_000, callback=callbacks)

# 6. Αποθήκευσε
model.save("models/best_model")
env.save("models/vec_normalize.pkl")
```

---

## 2. Callbacks — "Hooks" κατά το Training

Τα callbacks είναι σαν **event handlers** στο C#: κώδικας που εκτελείται σε κάθε step ή σε intervals.

**Αντιστοιχία C#:**
```csharp
// C# events:
model.OnStep += (sender, args) => { SaveCheckpoint(); };
model.OnRolloutEnd += (sender, args) => { LogMetrics(); };

// Python callbacks:
class MyCallback(BaseCallback):
    def _on_step(self):        # Κάθε step
        pass
    def _on_rollout_end(self): # Μετά κάθε rollout
        pass
```

### 2.1 CheckpointCallback — Αποθήκευση Ενδιαμέσων

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint = CheckpointCallback(
    save_freq=200_000,                         # Κάθε 200K steps
    save_path='./models/checkpoints/',         # Πού
    name_prefix='ppo_masked'                   # Πρόθεμα
)

# Δημιουργεί αρχεία:
# models/checkpoints/ppo_masked_200000_steps.zip
# models/checkpoints/ppo_masked_400000_steps.zip
# models/checkpoints/ppo_masked_600000_steps.zip
# ...
```

**Γιατί χρειάζεται:**
- Αν κρασάρει το training, δεν χάνεις τα πάντα
- Μπορείς να δοκιμάσεις διάφορα checkpoints (ποιο είναι καλύτερο)
- Μερικές φορές ένα νωρίτερο checkpoint είναι καλύτερο (overfitting αργότερα)

### 2.2 EvalCallback — Αξιολόγηση κατά τη Μάθηση

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,                              # Ξεχωριστό environment (test data!)
    best_model_save_path='./models/best/',
    log_path='./logs/',
    eval_freq=50_000,                      # Κάθε 50K steps
    n_eval_episodes=5,                     # 5 episodes evaluation
    deterministic=True,                    # Χωρίς randomness
)
```

**Τι κάνει:**
```
Training Step 50K:
  → Pause training
  → Τρέξε 5 episodes στο eval_env
  → Μέτρα mean reward
  → Αν mean_reward > best_so_far:
    → Αποθήκευσε "best_model.zip"
  → Resume training
```

---

## 3. Curriculum Learning — Σταδιακή Αύξηση Δυσκολίας

Η ιδέα: Μην δώσεις στον agent **όλα τα challenges ταυτόχρονα**. Ξεκίνα εύκολα, δυσκόλεψε σταδιακά.

```python
# I implement curriculum learning for degradation cost:
class DegradationCurriculumCallback(BaseCallback):
    """I gradually increase the degradation cost during training,
    so the agent first learns WHAT to do, then learns to be efficient."""

    def __init__(self, warmup_steps=500_000, target_deg=15.0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.target_deg = target_deg

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.warmup_steps)
        current_deg = progress * self.target_deg

        # Ενημέρωσε το environment
        self.training_env.envs[0].degradation_cost = current_deg
        return True
```

**Timeline:**
```
Steps 0 - 100K:     degradation = 0 €/MWh
                     Agent: "Trade πάρα πολύ! Δεν κοστίζει!"
                     ΜΑΘΑΙΝΕΙ: "buy low, sell high"

Steps 100K - 300K:  degradation = 3-9 €/MWh
                     Agent: "Hmm, κάθε trade κοστίζει λίγο..."
                     ΜΑΘΑΙΝΕΙ: "trade μόνο αν αξίζει"

Steps 300K - 500K:  degradation = 9-15 €/MWh
                     Agent: "Κάθε MW κοστίζει 15€ degradation!"
                     ΜΑΘΑΙΝΕΙ: "trade μόνο σε μεγάλα spreads"

Steps 500K+:        degradation = 15 €/MWh (σταθερό)
                     Agent: Τελική πολιτική, production-ready
```

**Αναλογία:** Σαν να μαθαίνεις κολύμβηση:
1. Πρώτα μαθαίνεις τις κινήσεις στο ρηχό (εύκολο)
2. Μετά πηγαίνεις στο βαθύ (δυσκολότερο)
3. Τέλος, μαθαίνεις ρυθμό αναπνοής (τελειοποίηση)

---

## 4. Custom Callback — Παράδειγμα

```python
from stable_baselines3.common.callbacks import BaseCallback

class TradingMetricsCallback(BaseCallback):
    """I log trading-specific metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.trades = 0
        self.idles = 0

    def _on_step(self):
        """I run on every single step."""
        # Πάρε info από environment
        info = self.locals.get('infos', [{}])[0]
        actual_mw = info.get('actual_mw', 0)

        if abs(actual_mw) > 0.1:
            self.trades += 1
        else:
            self.idles += 1

        # Log κάθε 10000 steps
        if self.num_timesteps % 10000 == 0:
            trade_rate = self.trades / max(1, self.trades + self.idles)
            self.logger.record('trading/trade_rate', trade_rate)
            self.logger.record('trading/total_trades', self.trades)
            self.trades = 0
            self.idles = 0

        return True  # True = συνέχισε training, False = σταμάτα
```

**Αντιστοιχία C#:**
```csharp
public class TradingMetricsCallback : ITrainingCallback
{
    public bool OnStep(StepInfo info)
    {
        if (Math.Abs(info.ActualMw) > 0.1)
            _trades++;
        // Log every 10K steps...
        return true; // continue training
    }
}
```

---

## 5. VecNormalize — Online Normalization

```python
from stable_baselines3.common.vec_env import VecNormalize

env = DummyVecEnv([make_env])
env = VecNormalize(
    env,
    norm_obs=True,       # Normalize observations
    norm_reward=True,     # Normalize rewards
    clip_obs=100.0,       # Clip normalized obs to [-100, 100]
    clip_reward=10.0,     # Clip normalized reward to [-10, 10]
)
```

**Τι κάνει:**
```
Raw observation: [0.50, 85.0, 10.0, 0.866, -0.5, ...]
Running mean:    [0.48, 82.3, 5.2,  0.0,   0.0,  ...]
Running std:     [0.15, 25.1, 8.7,  0.71,  0.71, ...]

Normalized: (raw - mean) / std = [0.13, 0.11, 0.55, 1.22, -0.70, ...]
```

**Κρίσιμο**: Πρέπει να ΑΠΟΘΗΚΕΥΣΕΙΣ τα running stats:
```python
# Save μαζί με model:
env.save("models/vec_normalize.pkl")

# Load στο evaluation:
env = VecNormalize.load("models/vec_normalize.pkl", eval_env)
env.training = False     # Μην ενημερώνεις τα stats!
env.norm_reward = False  # Μην normalize-άρεις τα rewards
```

---

## 6. DummyVecEnv vs SubprocVecEnv

```python
# Single process (απλό, για development):
env = DummyVecEnv([make_env])

# Multiple processes (γρήγορο, για production training):
env = SubprocVecEnv([make_env for _ in range(4)])
# Τρέχει 4 environments παράλληλα σε ξεχωριστά processes
# 4× γρηγορότερη συλλογή δεδομένων
```

**Στο project** χρησιμοποιούμε `DummyVecEnv` (1 environment) γιατί:
- Η CPU bottleneck είναι στο PPO update, όχι στο environment
- Απλούστερο debugging
- Deterministic αποτελέσματα

---

## 7. TensorBoard — Monitoring Training

```python
model = MaskablePPO(
    "MlpPolicy", env,
    tensorboard_log="./logs/"   # ← Εδώ αποθηκεύονται τα logs
)
```

**Τρέξε TensorBoard:**
```bash
tensorboard --logdir ./logs/
# Άνοιξε http://localhost:6006
```

**Τι βλέπεις:**

```
Charts:
├── rollout/ep_rew_mean          # Μέσο reward ανά episode (ΚΥΡΙΟ μετρικό)
├── rollout/ep_len_mean          # Μέσο μήκος episode
├── train/loss                   # Total loss (actor + critic)
├── train/policy_gradient_loss   # Actor loss (θέλεις αρνητικό)
├── train/value_loss             # Critic loss (θέλεις μικρό)
├── train/entropy_loss           # Exploration (θέλεις μέτριο)
├── train/clip_fraction          # % updates που κόπηκαν (θέλεις <0.2)
├── train/approx_kl              # Πόσο άλλαξε η πολιτική (θέλεις μικρό)
└── train/explained_variance     # Critic accuracy (θέλεις κοντά στο 1.0)
```

**Τι ψάχνεις:**

```
ΚΑΛΟ training:
  ep_rew_mean: ↑ σταθερά ανεβαίνει
  value_loss: ↓ σταθερά πέφτει
  entropy_loss: Σταδιακά μειώνεται (αλλά δεν πέφτει στο 0)
  explained_variance: → 0.8-0.95

ΚΑΚΟ training (collapse):
  ep_rew_mean: ↑↑↑ απότομα, μετά ↓↓↓ πέφτει
  entropy_loss: → 0 (ο agent "κόλλησε")
  value_loss: ↑↑↑ εκρήγνυται

ΑΡΓΟ training:
  ep_rew_mean: → σχεδόν σταθερό
  Λύση: Αύξησε learning_rate ή ent_coef
```

---

## 8. Seeding — Αναπαραγωγιμότητα

```python
import torch
import numpy as np

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

model = MaskablePPO(
    "MlpPolicy", env,
    seed=SEED,        # RL algorithm seed
)
```

**Γιατί:** Αν δύο runs με ίδιο seed δίνουν ίδια αποτελέσματα, μπορείς να συγκρίνεις hyperparameter changes.

**Στην πράξη:** Στο project τρέχουμε πολλά seeds (0, 1, 2) για robustness:
```python
for seed in [0, 1, 2]:
    model = MaskablePPO(..., seed=seed)
    model.learn(...)
    model.save(f"models/production_seed_{seed}/model")
```

---

## 9. Ολοκληρωμένο Training Script

```python
"""I structured the complete training pipeline with all components."""

import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib import MaskablePPO

# ──── 1. Data ────
df = pd.read_csv('data/feasible_data_with_balancing.csv', parse_dates=['timestamp'])
train_df = df.iloc[:int(len(df)*0.8)]   # 80% training
eval_df = df.iloc[int(len(df)*0.8):]    # 20% evaluation

# ──── 2. Environments ────
def make_train_env():
    env = BatteryEnvMasked(train_df, n_actions=21, **battery_params)
    env = ActionMasker(env, mask_fn)
    return Monitor(env)

def make_eval_env():
    env = BatteryEnvMasked(eval_df, n_actions=21, **battery_params)
    env = ActionMasker(env, mask_fn)
    return Monitor(env)

train_env = DummyVecEnv([make_train_env])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

# ──── 3. Model ────
model = MaskablePPO(
    "MlpPolicy",
    train_env,
    learning_rate=linear_schedule(3e-4),
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    policy_kwargs=dict(net_arch=[512, 256, 128]),
    tensorboard_log="./logs/",
    seed=42,
    verbose=1,
)

# ──── 4. Callbacks ────
callbacks = [
    CheckpointCallback(save_freq=200_000, save_path='./models/checkpoints/'),
    EvalCallback(eval_env, best_model_save_path='./models/best/',
                 eval_freq=50_000, n_eval_episodes=5),
]

# ──── 5. Train! ────
model.learn(total_timesteps=10_000_000, callback=callbacks)

# ──── 6. Save ────
model.save("models/final_model")
train_env.save("models/vec_normalize.pkl")
```

---

## 10. Πρακτική Άσκηση

### Άσκηση Α: Callback Design

Σχεδίασε (σε ψευδοκώδικα) ένα callback που:
- Κάθε 100K steps, ελέγχει το μέσο SoC
- Αν μέσο SoC < 0.20, εκτυπώνει warning
- Αν μέσο SoC > 0.80, εκτυπώνει warning

### Άσκηση Β: TensorBoard ερμηνεία

Αν δεις αυτά τα charts:
```
ep_rew_mean: 100 → 500 → 450 → 400 → 350
entropy_loss: 2.0 → 1.5 → 0.5 → 0.1 → 0.01
```
Τι συμβαίνει; (Hint: entropy collapse → ο agent κόλλησε)

### Άσκηση Γ: Ανάγνωση κώδικα

Στο `agent/train_masked.py`:
1. Ποιο CSV φορτώνει;
2. Τι callbacks χρησιμοποιεί;
3. Πού αποθηκεύεται η VecNormalize;
4. Πόσα total_timesteps τρέχει;

---

## Σύνοψη Ημέρας

```
Training Pipeline
├── Data Loading: CSV → DataFrame → train/eval split
├── Environment: BatteryEnvMasked → ActionMasker → Monitor → DummyVecEnv → VecNormalize
├── Model: MaskablePPO with hyperparams
├── Callbacks:
│   ├── CheckpointCallback: Αποθήκευση κάθε 200K steps
│   ├── EvalCallback: Αξιολόγηση + best model save
│   ├── CurriculumCallback: Σταδιακή αύξηση degradation
│   └── Custom: Trading metrics logging
├── Training: model.learn(10M steps)
├── Monitoring: TensorBoard (ep_rew_mean, losses, entropy)
└── Output: model.zip + vec_normalize.pkl
```

> **Αύριο (Ημέρα 9)**: Evaluation — πώς αξιολογούμε αν ο agent είναι πραγματικά καλός, VecNormalize pitfalls, και production readiness.
