# Ημέρα 2: Gymnasium — Πώς "βλέπει" τον κόσμο ένα RL Agent

> **Στόχος**: Να καταλάβεις τη δομή ενός Gymnasium environment.
> Αυτή είναι η βάση πάνω στην οποία χτίζεται ολόκληρο το EntsoDRL.

---

## 1. Η Μεγάλη Ιδέα: RL = Video Game

Σκέψου το Reinforcement Learning σαν ένα video game:

```
┌─────────────┐         action          ┌─────────────────┐
│             │ ───────────────────────► │                 │
│    AGENT    │                         │   ENVIRONMENT   │
│  (παίκτης)  │ ◄─────────────────────  │    (παιχνίδι)   │
│             │   observation, reward   │                 │
└─────────────┘                         └─────────────────┘
```

| Video Game | EntsoDRL |
|-----------|----------|
| Παίκτης | PPO Neural Network |
| Παιχνίδι | Αγορά ενέργειας + μπαταρία |
| Οθόνη (τι βλέπει) | Observation: τιμές, SoC, ώρα, κλπ (45 αριθμοί) |
| Κουμπιά (τι πατάει) | Action: charge/discharge/idle (21 επιλογές) |
| Σκορ | Reward: κέρδος - κόστος degradation - penalties |
| Game Over | Τέλος dataset (π.χ. 8760 ώρες = 1 έτος) |

---

## 2. Το Gymnasium Interface

Κάθε RL environment στην Python υλοποιεί αυτό το interface:

```python
import gymnasium as gym

class MyEnvironment(gym.Env):
    """I implement the standard Gymnasium interface."""

    def __init__(self):
        # Ορισμός observation space (τι βλέπει ο agent)
        # Ορισμός action space (τι μπορεί να κάνει)
        pass

    def reset(self):
        # Επαναφορά στην αρχή — σαν "New Game"
        return observation, info

    def step(self, action):
        # Εκτέλεση μιας ενέργειας — σαν ένα frame του game
        return observation, reward, terminated, truncated, info
```

**Αντιστοιχία C#:**
```csharp
// Αν ήταν C# interface:
public interface IEnvironment
{
    (float[] Observation, Dictionary Info) Reset();
    (float[] Observation, float Reward, bool Terminated, bool Truncated, Dictionary Info) Step(int action);
}
```

---

## 3. Ο Κύκλος Ζωής — The Game Loop

Αυτός είναι ο βασικός βρόχος που τρέχει σε ΚΑΘΕ RL σύστημα:

```python
# Αυτό γίνεται στο agent/evaluate_masked.py:
env = BatteryEnvMasked(df, battery_params)
obs, info = env.reset()           # 1. Start new game

done = False
total_reward = 0

while not done:                    # 2. Game loop
    action = agent.predict(obs)    # 3. Agent αποφασίζει
    obs, reward, terminated, truncated, info = env.step(action)  # 4. Environment εκτελεί
    total_reward += reward
    done = terminated or truncated

print(f"Final score: {total_reward}")
```

**Βήμα-βήμα τι γίνεται σε κάθε iteration:**

```
Step 1: Agent βλέπει → obs = [SoC=0.5, price=85€, hour=14, ...]
Step 2: Agent αποφασίζει → action = 15 (discharge 16.5MW)
Step 3: Environment εκτελεί:
    - Φυσική: SoC μειώνεται (0.5 → 0.48)
    - Αγορά: κερδίζει 16.5MW × 85€ = 1402€
    - Degradation: κόστος χρήσης μπαταρίας
    - Reward = κέρδος - κόστη
Step 4: Environment επιστρέφει:
    - Νέο obs: [SoC=0.48, price=92€, hour=15, ...]
    - reward: 12.5
    - terminated: False (δεν τελείωσε)
Step 5: Επόμενο step → πίσω στο 1
```

---

## 4. Observation Space — "Τι βλέπει ο Agent"

Ο observation space ορίζει **τι μπορεί να δει** ο agent. Στο project μας:

```python
# Από gym_envs/battery_env_masked.py:
self.observation_space = gym.spaces.Box(
    low=-np.inf,               # Ελάχιστη τιμή: -∞
    high=np.inf,               # Μέγιστη τιμή: +∞
    shape=(45,),               # 45 αριθμοί (features)
    dtype=np.float32           # 32-bit floating point
)
```

**Αντιστοιχία C#:**
```csharp
// Σαν να λέμε:
float[] observation = new float[45];
// Κάθε θέση έχει συγκεκριμένη σημασία:
// [0] = SoC
// [1] = max_discharge_normalized
// [2] = max_charge_normalized
// [3] = best_bid / 100
// [4] = best_ask / 100
// ... κ.ο.κ.
```

**Γιατί `Box`?** Γιατί τα observations είναι συνεχείς αριθμοί (float). Αν ήταν κατηγορικά (π.χ. "buy"/"sell"), θα χρησιμοποιούσαμε `Discrete`.

Τα 45 features ομαδοποιούνται:

```
Observation Vector (45 floats):
├── Battery State [0-2]
│   ├── [0] SoC (0.0-1.0)
│   ├── [1] max_discharge_mw / 30.0
│   └── [2] max_charge_mw / 30.0
├── Market State [3-5]
│   ├── [3] best_bid / 100.0
│   ├── [4] best_ask / 100.0
│   └── [5] dam_commitment / 30.0
├── Time [6-7]
│   ├── [6] sin(2π × hour / 24)
│   └── [7] cos(2π × hour / 24)
├── Price Lookahead [8-19]
│   └── Επόμενες 12 ώρες τιμών / 100.0
├── DAM Lookahead [20-31]
│   └── Επόμενες 12 ώρες commitments / 30.0
├── Risk Features [32-35]
│   ├── Imbalance gradient
│   ├── RES deviation
│   ├── Imbalance risk score
│   └── Shortfall risk
├── aFRR Indicators [36-37]
├── Timing Features [38-41]
├── Price Awareness [42-43]
│   ├── price_vs_typical_hour (backward-looking!)
│   └── trade_worthiness (backward-looking!)
└── Gate Closure [44]
    └── hours_until_gate_closure / 24.0
```

---

## 5. Action Space — "Τι μπορεί να κάνει ο Agent"

```python
# Από gym_envs/battery_env_masked.py:
self.action_space = gym.spaces.Discrete(21)
# 21 δυνατές ενέργειες: 0, 1, 2, ..., 20
```

Κάθε αριθμός αντιστοιχεί σε MW:

```
Action 0  → -30.0 MW (full discharge — πούλα 30MW)
Action 1  → -27.0 MW
Action 2  → -24.0 MW
...
Action 10 → 0.0 MW   (idle — μην κάνεις τίποτα)
...
Action 18 → +24.0 MW
Action 19 → +27.0 MW
Action 20 → +30.0 MW (full charge — αγόρασε 30MW)
```

**Η μετατροπή:**
```python
# Από battery_env_masked.py:
def _action_to_mw(self, action: int) -> float:
    """I convert discrete action to MW power."""
    # action ∈ [0, 20]
    # Normalize to [-1, 1]
    normalized = (action - 10) / 10.0
    # Scale to [-max_power, +max_power]
    return normalized * self.max_discharge_mw

# Παραδείγματα:
# action=0  → (0-10)/10 = -1.0 → -1.0 × 30 = -30 MW (discharge)
# action=10 → (10-10)/10 = 0.0 → 0.0 × 30 = 0 MW (idle)
# action=20 → (20-10)/10 = 1.0 → 1.0 × 30 = +30 MW (charge)
```

**Γιατί Discrete(21) και όχι Continuous?**
- Discrete actions + action masking = μπορούμε να **αποκλείσουμε** φυσικά αδύνατες ενέργειες
- Αν η μπαταρία είναι γεμάτη (SoC=95%), τα actions 11-20 (charge) γίνονται `masked` (απαγορευμένα)
- Αυτό είναι πολύ πιο δύσκολο με continuous actions

---

## 6. `reset()` — New Game

```python
# Από battery_env_masked.py:
def reset(self, seed=None, options=None):
    """I reset the environment to initial state."""
    super().reset(seed=seed)

    self.current_step = 0
    self.current_soc = 0.5    # Ξεκίνα με μπαταρία στο 50%
    self.total_reward = 0.0
    self.episode_trades = []

    observation = self._get_observation()
    info = {}

    return observation, info
```

**Πότε καλείται:**
- Στην αρχή του training
- Όταν τελειώσει ένα episode (φτάσαμε στο τέλος του dataset)
- Ο trainer (SB3) το καλεί αυτόματα

---

## 7. `step(action)` — Ένα "frame" του game

Αυτή είναι η **καρδιά** του environment. Ας τη δούμε σε ψευδοκώδικα:

```python
def step(self, action):
    """I execute one timestep of the environment."""

    # 1. Μετατροπή action → MW
    power_mw = self._action_to_mw(action)

    # 2. Φυσική μπαταρίας — ενημέρωση SoC
    if power_mw > 0:   # Charging
        energy = power_mw * dt * sqrt(efficiency)
        self.current_soc += energy / capacity
    elif power_mw < 0: # Discharging
        energy = abs(power_mw) * dt / sqrt(efficiency)
        self.current_soc -= energy / capacity

    # 3. Clamp SoC στο [0.05, 0.95]
    self.current_soc = np.clip(self.current_soc, 0.05, 0.95)

    # 4. Υπολογισμός reward (κέρδος - κόστη)
    reward = self.reward_calculator.calculate(
        action_mw=power_mw,
        soc=self.current_soc,
        dam_price=current_dam_price,
        # ... πολλά ακόμα
    )

    # 5. Πάμε στο επόμενο βήμα
    self.current_step += 1

    # 6. Τέλος episode?
    terminated = self.current_step >= len(self.df) - 1

    # 7. Νέο observation
    observation = self._get_observation()

    # 8. Debug info
    info = {
        'soc': self.current_soc,
        'actual_mw': power_mw,
        'reward': reward,
        'dam_price': current_dam_price,
    }

    return observation, reward, terminated, False, info
```

**Τα 5 return values:**

| Return | Τύπος | Σημασία |
|--------|-------|---------|
| `observation` | `np.array(45,)` | Τι βλέπει ο agent μετά |
| `reward` | `float` | Πόντοι αυτού του step |
| `terminated` | `bool` | Τέλος episode (π.χ. τέλος δεδομένων) |
| `truncated` | `bool` | Κόπηκε πρόωρα (π.χ. max steps) |
| `info` | `dict` | Extra debug πληροφορίες |

---

## 8. Spaces — Τύποι "Χώρων"

Τα Gymnasium spaces ορίζουν τι τιμές είναι **νόμιμες**:

```python
import gymnasium as gym
import numpy as np

# Discrete — ακέραιοι επιλογές
gym.spaces.Discrete(21)           # {0, 1, 2, ..., 20}
# Χρήση: action space

# Box — συνεχή floats σε εύρος
gym.spaces.Box(
    low=-np.inf, high=np.inf,
    shape=(45,), dtype=np.float32
)
# Χρήση: observation space

# MultiBinary — binary vector
gym.spaces.MultiBinary(21)         # [0,1,1,0,...] — 21 bits
# Χρήση: action masks (ποιες actions είναι valid)

# MultiDiscrete — πολλαπλά discrete
gym.spaces.MultiDiscrete([3, 21])  # Πρώτο: {0,1,2}, Δεύτερο: {0..20}
# Χρήση: multi-market environments (DAM action + physical action)
```

---

## 9. Ένα Minimal Example

Για να δεις πώς δουλεύει, φαντάσου ένα υπέρ-απλό battery environment:

```python
import gymnasium as gym
import numpy as np

class SimpleBatteryEnv(gym.Env):
    """I created this minimal example to teach the core concepts."""

    def __init__(self):
        super().__init__()
        # Agent βλέπει: [SoC, price]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 200.0]),
            dtype=np.float32
        )
        # Agent κάνει: 0=discharge, 1=idle, 2=charge
        self.action_space = gym.spaces.Discrete(3)

        self.prices = [50, 60, 120, 130, 80, 70, 40, 110]
        self.soc = 0.5
        self.step_idx = 0

    def reset(self, seed=None, options=None):
        self.soc = 0.5
        self.step_idx = 0
        return np.array([self.soc, self.prices[0]], dtype=np.float32), {}

    def step(self, action):
        price = self.prices[self.step_idx]

        if action == 0:   # Discharge (sell)
            power = 10.0  # MW
            self.soc -= 0.1
            reward = power * price / 1000   # Revenue
        elif action == 2: # Charge (buy)
            power = 10.0
            self.soc += 0.1
            reward = -power * price / 1000  # Cost
        else:             # Idle
            reward = 0.0

        self.soc = np.clip(self.soc, 0.0, 1.0)
        self.step_idx += 1
        terminated = self.step_idx >= len(self.prices)

        obs = np.array([self.soc, self.prices[min(self.step_idx, len(self.prices)-1)]], dtype=np.float32)
        return obs, reward, terminated, False, {'soc': self.soc}
```

**Η βέλτιστη στρατηγική:**
- Charge (buy) στις τιμές 40, 50, 60 (φθηνά)
- Discharge (sell) στις τιμές 120, 130 (ακριβά)
- Idle στις ενδιάμεσες

Αυτό ακριβώς μαθαίνει το RL agent!

---

## 10. Πρακτική Άσκηση

### Άσκηση Α: Ανάγνωση κώδικα

Άνοιξε το `gym_envs/battery_env_masked.py` και βρες:

1. Πού ορίζεται το `observation_space`; (στο `__init__`)
2. Πού ορίζεται το `action_space`; (στο `__init__`)
3. Ποιο είναι το αρχικό SoC στο `reset()`; (0.5 = 50%)
4. Πώς γίνεται η μετατροπή action → MW;

### Άσκηση Β: Νοητικό trace

Περπάτα νοητικά ένα episode 3 steps:

```
Step 0: SoC=0.50, price=40€
  → Agent επιλέγει action=20 (charge +30MW)
  → SoC αυξάνεται: 0.50 + (30 × 1 × √0.94) / 146 ≈ 0.70
  → Reward: αρνητικό (αγοράζει, πληρώνει)

Step 1: SoC=0.70, price=130€
  → Agent επιλέγει action=0 (discharge -30MW)
  → SoC μειώνεται: 0.70 - (30 × 1 / √0.94) / 146 ≈ 0.49
  → Reward: θετικό (πουλάει, κερδίζει)

Step 2: SoC=0.49, price=85€
  → Agent επιλέγει action=10 (idle)
  → SoC παραμένει: 0.49
  → Reward: ~0 (μικρό SoC buffer bonus)
```

---

## Σύνοψη Ημέρας

```
Gymnasium Environment = Video Game
├── observation_space → Τι βλέπει ο agent (Box: 45 floats)
├── action_space      → Τι μπορεί να κάνει (Discrete: 21 actions)
├── reset()           → New Game → returns observation
├── step(action)      → One frame → returns (obs, reward, done, truncated, info)
└── Game Loop         → while not done: action → step → repeat
```

> **Αύριο (Ημέρα 3)**: Θα κάνουμε deep dive στο `BatteryEnvMasked` — τη φυσική της μπαταρίας, τα market mechanics, και πώς δουλεύει πραγματικά η αγορά ενέργειας μέσα στο environment.
