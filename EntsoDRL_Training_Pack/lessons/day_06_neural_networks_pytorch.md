# Ημέρα 6: Neural Networks & PyTorch — Ο "Εγκέφαλος" του Agent

> **Στόχος**: Να καταλάβεις τι ΕΙΝΑΙ ένα neural network, πώς μετατρέπει
> observations σε actions, και πώς συνδέεται με τον κώδικα στο `train_masked.py`.

---

## 1. Η Μεγάλη Ιδέα: Function Approximation

Ένα neural network είναι απλά μια **μαθηματική συνάρτηση** που μαθαίνει:

```
f(observation) → action probabilities

Είσοδος: 45 αριθμοί (SoC, prices, hour, ...)
Έξοδος: 21 πιθανότητες (μία για κάθε action)
```

**Αντιστοιχία C#:**
```csharp
// Σαν ένα εξαιρετικά σύνθετο:
public float[] Predict(float[] observation)  // 45 in → 21 out
{
    // Αλλά αντί για if/else κανόνες,
    // χρησιμοποιεί εκατομμύρια weights
    // που μάθηκαν από εκατομμύρια παραδείγματα
}
```

---

## 2. Layers — Τα "Στρώματα"

Ένα neural network αποτελείται από layers (στρώματα). Στο project:

```python
# Από train_masked.py:
policy_kwargs = dict(
    net_arch=[512, 256, 128]  # 3 hidden layers
)
```

Αυτό σημαίνει:

```
Input Layer:  45 neurons  (observation)
    ↓ (45 × 512 = 23,040 weights)
Hidden Layer 1: 512 neurons
    ↓ (512 × 256 = 131,072 weights)
Hidden Layer 2: 256 neurons
    ↓ (256 × 128 = 32,768 weights)
Hidden Layer 3: 128 neurons
    ↓ (128 × 21 = 2,688 weights)
Output Layer: 21 neurons  (action probabilities)

Σύνολο: ~189,568 trainable parameters (weights)
```

---

## 3. Πώς "Σκέφτεται" — Forward Pass

Ας δούμε τι γίνεται βήμα-βήμα:

```
Observation: [0.50, 0.85, 0.92, 0.83, 0.87, 0.33, -0.87, 0.50, ...]
                                                              45 αριθμοί

Step 1: Linear Transform (Matrix Multiply)
  z₁ = W₁ × input + b₁
  45 inputs × [45×512] matrix = 512 outputs

Step 2: Activation Function (ReLU)
  h₁ = ReLU(z₁) = max(0, z₁)
  Αν z₁[i] < 0 → h₁[i] = 0
  Αν z₁[i] > 0 → h₁[i] = z₁[i]

Step 3: Repeat for each layer
  z₂ = W₂ × h₁ + b₂  →  h₂ = ReLU(z₂)   # 256 outputs
  z₃ = W₃ × h₂ + b₃  →  h₃ = ReLU(z₃)   # 128 outputs
  z₄ = W₄ × h₃ + b₄                       # 21 outputs (logits)

Step 4: Softmax (για action probabilities)
  probs = softmax(z₄)
  probs = [0.02, 0.01, ..., 0.15, 0.30, 0.20, ...]
                              ^          ^
                          action 8   action 10 (idle)
```

---

## 4. Activation Functions — Γιατί ReLU

Χωρίς activation functions, πολλά linear layers = ένα linear layer (μαθηματικά).
Η activation function εισάγει **μη-γραμμικότητα** — αυτό δίνει στο δίκτυο δύναμη:

```
ReLU(x) = max(0, x)

x = -3 → ReLU(-3) = 0
x = 0  → ReLU(0)  = 0
x = 5  → ReLU(5)  = 5

Διάγραμμα:
output
  5 |          /
  4 |         /
  3 |        /
  2 |       /
  1 |      /
  0 |─────/──────── input
    -3  -2  -1  0  1  2  3  4  5
```

**Γιατί ReLU και όχι κάτι άλλο;**
- Απλό, γρήγορο στον υπολογισμό
- Δεν έχει vanishing gradient πρόβλημα (σε αντίθεση με sigmoid)
- "Just works" — standard επιλογή

---

## 5. Actor-Critic Architecture

Στο EntsoDRL, δεν έχουμε ΕΝΑ network, αλλά **ΔΥΟ**:

```
                    Observation (45 features)
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼─────┐ ┌────▼────┐
              │   ACTOR   │ │  CRITIC │
              │ (Policy)  │ │ (Value) │
              │ [512,256, │ │ [512,256│
              │  128]     │ │  ,128]  │
              └─────┬─────┘ └────┬────┘
                    │            │
              ┌─────▼─────┐ ┌───▼────┐
              │ 21 action  │ │ 1 value│
              │ probs      │ │ (€)    │
              └────────────┘ └────────┘
```

**Actor (Policy Network):**
- Input: observation (45 features)
- Output: 21 πιθανότητες — μία για κάθε action
- Ρόλος: "Τι πρέπει να κάνω;"

**Critic (Value Network):**
- Input: observation (45 features)
- Output: 1 αριθμός — estimated future reward
- Ρόλος: "Πόσο καλή είναι αυτή η κατάσταση;"

**Αντιστοιχία C#:**
```csharp
// Actor:
float[] GetActionProbabilities(float[] observation);  // 45 → 21

// Critic:
float EstimateValue(float[] observation);              // 45 → 1
```

---

## 6. Weights — Τα "Βάρη" του Δικτύου

Τα weights είναι αριθμοί που **μαθαίνονται** κατά το training. Αρχικά είναι τυχαίοι:

```
Πριν training (random weights):
  observation [0.50, 85€, ...] → action probs [0.048, 0.047, 0.049, ...]
                                                 ≈ ομοιόμορφα τυχαίες
  Ο agent κάνει τυχαίες ενέργειες

Μετά 10M steps training (learned weights):
  observation [0.50, 85€, ...] → action probs [0.01, 0.01, ..., 0.35, 0.20, ...]
                                                                  ^      ^
                                                           idle  discharge
  Ο agent κάνει "έξυπνες" ενέργειες
```

**Τι αλλάζει;** Μόνο τα weights (W₁, W₂, W₃, W₄ + biases). Η αρχιτεκτονική μένει ίδια.

---

## 7. PyTorch — Η Βιβλιοθήκη

Το PyTorch (by Meta/Facebook) κάνει δύο πράγματα:
1. **Tensor operations**: Πράξεις σε μεγάλους πίνακες αριθμών (σαν NumPy, αλλά σε GPU)
2. **Automatic differentiation**: Υπολογίζει αυτόματα πόσο πρέπει να αλλάξει κάθε weight

```python
import torch

# Tensor — σαν np.array αλλά υποστηρίζει GPU + autograd
x = torch.tensor([0.5, 0.85, 0.92])

# Neural network layer
layer = torch.nn.Linear(45, 512)  # 45 inputs → 512 outputs

# Forward pass
output = layer(x)  # Matrix multiply + bias

# Activation
output = torch.nn.functional.relu(output)  # ReLU
```

**Δεν χρειάζεται να γράψεις PyTorch** — το Stable Baselines 3 το κάνει για σένα. Αλλά βοηθά να ξέρεις τι γίνεται "κάτω από το καπό".

---

## 8. Στο Project — Πού Βλέπεις Neural Networks

### 8.1 Δημιουργία Model

```python
# Από train_masked.py:
from sb3_contrib import MaskablePPO

model = MaskablePPO(
    "MlpPolicy",                          # MLP = Multi-Layer Perceptron
    env,
    policy_kwargs=dict(
        net_arch=[512, 256, 128]           # Hidden layers
    ),
    # ... άλλα hyperparameters
)
```

`"MlpPolicy"` = `MaskableActorCriticPolicy` = Actor + Critic, κάθε ένα με [512, 256, 128] layers.

### 8.2 Inference (Πρόβλεψη)

```python
# Από evaluate_masked.py:
action, _ = model.predict(
    observation,                # 45 floats
    deterministic=True,         # Πάντα η πιο πιθανή action
    action_masks=mask           # Binary mask
)
```

Τι γίνεται εσωτερικά:
```
1. observation → Actor Network → 21 logits
2. Mask logits (invalid = -∞)
3. Softmax → 21 probabilities
4. deterministic=True → argmax → action με μεγαλύτερη probability
   deterministic=False → sample → τυχαία action βάσει probabilities
```

### 8.3 Αποθήκευση / Φόρτωση

```python
# Save (μετά training):
model.save("models/best_model")
# Δημιουργεί: models/best_model.zip (weights + config)

# Load (για inference):
model = MaskablePPO.load("models/best_model")
```

Τα `.zip` files στον φάκελο `models/` περιέχουν τα trained weights.

---

## 9. Normalization — Γιατί χρειάζεται

Χωρίς normalization, τα inputs μπορεί να είναι:
```
SoC = 0.50          (εύρος 0-1)
price = 85.0        (εύρος 0-500)
commitment = 10.0   (εύρος -30 to 30)
```

Πρόβλημα: Η price (85) "κυριαρχεί" πάνω στο SoC (0.50). Το δίκτυο δυσκολεύεται.

**Λύση 1: Manual scaling** (στο feature_engineer.py)
```python
obs[0] = soc                    # Ήδη [0, 1]
obs[3] = best_bid / 100.0       # [0, 5] → ~[0, 1]
obs[5] = dam_commitment / 30.0  # [-1, 1]
```

**Λύση 2: VecNormalize** (στο train_masked.py)
```python
from stable_baselines3.common.vec_env import VecNormalize
env = VecNormalize(env, norm_obs=True, norm_reward=True)
# Αυτόματα κάνει: obs = (obs - mean) / std
# Ενημερώνεται online κατά τη διάρκεια training
```

---

## 10. GPU vs CPU

```python
# Training: GPU γρηγορότερο (matrix multiplications)
model = MaskablePPO(
    "MlpPolicy", env,
    device="cuda"    # Χρήση GPU (NVIDIA)
    # device="cpu"   # Fallback σε CPU
)

# Inference: CPU αρκετό (μόνο ένα forward pass)
model = MaskablePPO.load("model.zip", device="cpu")
```

Στο project μας, training 10M steps:
- **CPU**: ~4-6 ώρες
- **GPU (CUDA)**: ~1-2 ώρες

---

## 11. Τι ΑΚΡΙΒΩΣ μαθαίνει το δίκτυο;

Μετά το training, τα weights κωδικοποιούν "κανόνες" σαν:

```
ΑΝ τιμή > μέσος_24h ΚΑΙ SoC > 30% ΚΑΙ momentum > 0
ΤΟΤΕ discharge (sell) με μεγάλο power

ΑΝ τιμή < μέσος_24h ΚΑΙ SoC < 70% ΚΑΙ off-peak
ΤΟΤΕ charge (buy)

ΑΝ SoC < 20% ΚΑΙ dam_commitment > 0
ΤΟΤΕ idle (κράτα ενέργεια για commitment)
```

Αλλά δεν είναι if/else — είναι **smooth, continuous functions** που μπορούν να χειριστούν αμέτρητους συνδυασμούς. Η δύναμη του neural network είναι ότι βρίσκει patterns που ένας άνθρωπος δεν θα σκεφτόταν.

---

## 12. Πρακτική Άσκηση

### Άσκηση Α: Μέγεθος δικτύου

Αν `net_arch = [512, 256, 128]` και input=45, output=21:
1. Πόσα weights στο Layer 1; (45 × 512 + 512 biases = ?)
2. Πόσα weights στο Layer 2; (512 × 256 + 256 = ?)
3. Πόσα weights στο Layer 3; (256 × 128 + 128 = ?)
4. Πόσα weights στο Output? (128 × 21 + 21 = ?)
5. Σύνολο weights Actor = ?

Δεν ξεχνάμε: Critic έχει ΤΟ ΙΔΙΟ μέγεθος, αλλά output = 1 αντί 21.

### Άσκηση Β: Forward pass νοητικά

Είσοδος: [SoC=0.50, price_norm=0.85, ...]
- Layer 1: 512 neurons → κάθε neuron κάνει weighted sum + ReLU
- Αν ένα neuron πάρει αρνητική τιμή, τι βγάζει μετά το ReLU;
- Αν η τελική έξοδος είναι logits = [2.0, -1.0, 0.5], ποιο action επιλέγεται με `deterministic=True`;

### Άσκηση Γ: Ανάγνωση κώδικα

Στο `agent/train_masked.py`:
1. Βρες τη γραμμή `policy_kwargs` — τι αρχιτεκτονική χρησιμοποιεί;
2. Βρες τη γραμμή `model.save()` — πού αποθηκεύεται;
3. Τι device χρησιμοποιεί (GPU ή CPU);

---

## Σύνοψη Ημέρας

```
Neural Network = "Μαθηματική Συνάρτηση που μαθαίνει"
├── Architecture
│   ├── Actor: observation(45) → [512, 256, 128] → action_probs(21)
│   └── Critic: observation(45) → [512, 256, 128] → value(1)
├── Components
│   ├── Weights: Αριθμοί που μαθαίνονται (≈200K parameters)
│   ├── ReLU: Activation function (max(0, x))
│   └── Softmax: Μετατροπή logits → probabilities
├── Training
│   ├── Random weights → Τυχαίες αποφάσεις
│   ├── 10M steps later → Έξυπνες αποφάσεις
│   └── Saved as .zip (weights + config)
└── Inference
    └── observation → forward pass → action (in milliseconds)
```

> **Αύριο (Ημέρα 7)**: PPO Algorithm — ΠΩΣ ακριβώς αλλάζουν τα weights. Ο αλγόριθμος πίσω από τη μάθηση.
