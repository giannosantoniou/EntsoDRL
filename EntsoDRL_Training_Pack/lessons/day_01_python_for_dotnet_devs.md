# Ημέρα 1: Python για .NET Developers — NumPy & Pandas

> **Στόχος**: Να καταλάβεις τα Python patterns που χρησιμοποιεί ο κώδικας του EntsoDRL.
> Δεν θα μάθεις "Python from scratch" — θα μάθεις μόνο ό,τι χρειάζεσαι για να διαβάζεις τον κώδικα.

---

## 1. Αντιστοιχίσεις .NET → Python

Αν έρχεσαι από C#, αυτός ο πίνακας θα σε βοηθήσει να "μεταφράζεις" νοητικά:

| C# / .NET | Python | Παράδειγμα στο project |
|-----------|--------|----------------------|
| `interface IFoo` | `class IFoo(ABC)` | `IDataProvider` στο `data/data_provider.py` |
| `abstract void Method()` | `@abstractmethod` | `def get_market_state(self)` |
| `class Foo : IFoo` | `class Foo(IFoo)` | `SimulatedDataProvider(IDataProvider)` |
| `Dictionary<K,V>` | `dict` | `battery_params = {'capacity_mwh': 146.0, ...}` |
| `List<T>` | `list` | `price_lookahead = [85.0, 92.0, ...]` |
| `enum Mode { Training, Production }` | `class Mode(Enum)` | `config.py` |
| `record / struct` | `@dataclass` | `MarketState`, `BatteryTelemetry` |
| `Nullable<float>` | `Optional[float]` | Type hints σε function signatures |
| `using / IDisposable` | Context managers (`with`) | `with open(path) as f:` |
| `LINQ .Where().Select()` | List comprehensions | `[p/100 for p in prices]` |
| `NuGet packages` | `pip install` / `requirements.txt` | `requirements.txt` |

---

## 2. Dataclasses — Τα "Records" της Python

Στο C# έχεις `record` ή `struct`. Στην Python χρησιμοποιούμε `@dataclass`:

```python
# Από data/data_provider.py
from dataclasses import dataclass

@dataclass
class MarketState:
    current_price: float
    best_bid: float
    best_ask: float
    spread: float
    imbalance_price: float
    dam_commitment: float
    hour: int
    hour_sin: float
    hour_cos: float
```

**Αντιστοιχία C#:**
```csharp
public record MarketState(
    double CurrentPrice,
    double BestBid,
    double BestAsk,
    double Spread,
    // ...
);
```

Η διαφορά: στην Python, τα `@dataclass` δίνουν αυτόματα `__init__`, `__repr__`, `__eq__` — ακριβώς όπως τα C# records.

---

## 3. Abstract Base Classes — Τα "Interfaces" της Python

Στο project χρησιμοποιούμε ABC (Abstract Base Class) αντί για C# `interface`:

```python
# Από data/data_provider.py
from abc import ABC, abstractmethod

class IDataProvider(ABC):
    """I defined this interface for dependency inversion —
    training and production use different implementations."""

    @abstractmethod
    def get_market_state(self) -> MarketState:
        pass

    @abstractmethod
    def get_battery_telemetry(self) -> BatteryTelemetry:
        pass
```

**Αντιστοιχία C#:**
```csharp
public interface IDataProvider
{
    MarketState GetMarketState();
    BatteryTelemetry GetBatteryTelemetry();
}
```

**Σημαντικό**: Το `pass` στην Python σημαίνει "κενό body" — όπως το `;` μετά από ένα abstract method στο C#.

---

## 4. NumPy — Τα βασικά που χρειάζεσαι

Το NumPy είναι η βιβλιοθήκη αριθμητικών υπολογισμών. Σκέψου το ως "vectorized math arrays". Ο κώδικας του project το χρησιμοποιεί παντού.

### 4.1 Arrays

```python
import numpy as np

# 1D array — σαν float[] στο C#
prices = np.array([85.0, 92.0, 78.0, 110.0])

# Τύπος
print(prices.dtype)   # float64
print(prices.shape)   # (4,) — 4 στοιχεία, 1 διάσταση

# Indexing — ίδιο με C#
print(prices[0])      # 85.0
print(prices[-1])     # 110.0 (τελευταίο στοιχείο)
print(prices[1:3])    # [92.0, 78.0] (slice, αποκλείει το τέλος)
```

### 4.2 Vectorized Operations — Η μεγάλη διαφορά

Στο C# γράφεις loops. Στο NumPy, κάνεις πράξεις σε ολόκληρα arrays:

```python
# C# τρόπος (ΜΗΝ το κάνεις αυτό σε Python):
# for i in range(len(prices)):
#     normalized[i] = prices[i] / 100.0

# Python/NumPy τρόπος:
normalized = prices / 100.0   # [0.85, 0.92, 0.78, 1.10]

# Πού το βλέπεις στο project (feature_engineer.py):
# price_lookahead = [p / 100.0 for p in raw_prices]
```

### 4.3 Βασικές συναρτήσεις που θα δεις στον κώδικα

```python
# Στο reward_calculator.py:
np.clip(soc, 0.05, 0.95)       # Κόβει τιμές στο εύρος [0.05, 0.95]
np.abs(-15.5)                   # 15.5 — απόλυτη τιμή
np.sqrt(0.94)                   # √0.94 ≈ 0.9695 — τετραγωνική ρίζα
np.mean(prices)                 # Μέσος όρος
np.std(prices)                  # Τυπική απόκλιση

# Στο feature_engineer.py (κωδικοποίηση ώρας):
np.sin(2 * np.pi * hour / 24)  # Κυκλική κωδικοποίηση ώρας (sin)
np.cos(2 * np.pi * hour / 24)  # Κυκλική κωδικοποίηση ώρας (cos)

# Στο battery_env_masked.py (action masks):
np.ones(21, dtype=np.int8)     # [1,1,1,...,1] — 21 στοιχεία, όλα 1
mask[0:5] = 0                   # Μηδένισε τα πρώτα 5 στοιχεία

# Zeros — κενό observation vector:
obs = np.zeros(45, dtype=np.float32)   # 45 μηδενικά
```

### 4.4 Boolean Masking

Αυτό θα το δεις πολύ στο action masking:

```python
actions = np.array([-30, -20, -10, 0, 10, 20, 30])
mask = actions <= 10   # [True, True, True, True, True, False, False]

# Φιλτράρισμα:
valid = actions[mask]  # [-30, -20, -10, 0, 10]
```

---

## 5. Pandas — Τα βασικά που χρειάζεσαι

Το Pandas χειρίζεται tabular data — σαν ένα in-memory DataTable / SQL table.

### 5.1 DataFrame — Ο κεντρικός τύπος

```python
import pandas as pd

# Φόρτωση CSV — αυτό γίνεται στο train_masked.py
df = pd.read_csv('data/feasible_data_with_dam.csv', parse_dates=['timestamp'])

# Τι είναι ένα DataFrame:
# timestamp          | price  | dam_commitment | soc  | ...
# 2024-01-01 00:00   | 85.0   | 10.0           | 0.50 | ...
# 2024-01-01 01:00   | 78.0   | 10.0           | 0.55 | ...
# 2024-01-01 02:00   | 72.0   | 0.0            | 0.60 | ...
```

### 5.2 Πρόσβαση σε δεδομένα

```python
# Μία στήλη (σαν SELECT price FROM df):
prices = df['price']                    # Series (1D)
prices = df.price                       # Ίδιο πράγμα

# Πολλές στήλες:
subset = df[['price', 'dam_commitment']] # DataFrame (2D)

# Μία γραμμή (by index):
row = df.iloc[0]                        # Πρώτη γραμμή
row = df.iloc[-1]                       # Τελευταία γραμμή

# Slice γραμμών:
last_24h = df.iloc[-24:]               # Τελευταίες 24 γραμμές

# Τιμή σε συγκεκριμένη θέση:
price_at_step = df.iloc[42]['price']    # Γραμμή 42, στήλη 'price'
```

### 5.3 Φιλτράρισμα — Σαν LINQ .Where()

```python
# C#:  df.Where(row => row.Price > 100).ToList()
# Python:
expensive = df[df['price'] > 100]

# Σύνθετο φίλτρο:
peak_hours = df[(df['hour'] >= 17) & (df['hour'] <= 21)]
#                                    ^ ΠΡΟΣΟΧΗ: & αντί για &&
#              ( ) υποχρεωτικές γύρω από κάθε condition
```

### 5.4 Στο project — Πώς χρησιμοποιείται

```python
# Στο battery_env_masked.py — step():
current_row = self.df.iloc[self.current_step]
dam_price = current_row['dam_price']
best_bid = current_row.get('best_bid', dam_price * 0.98)

# Στο evaluate_detailed.py — rolling statistics:
df['rolling_mean_24h'] = df['price'].rolling(24).mean()

# Στο feature_engineer.py — lookahead:
lookahead_prices = self.df.iloc[step:step+12]['price'].values
# .values μετατρέπει Series → NumPy array
```

### 5.5 `.rolling()` — Κυλιόμενα στατιστικά

Αυτό θα το δεις στα backward-looking features:

```python
# Μέσος όρος τελευταίων 24 ωρών (χωρίς data leakage):
df['mean_24h'] = df['price'].rolling(window=24, min_periods=1).mean()

# min_periods=1: Αν δεν υπάρχουν 24 τιμές ακόμα,
#                χρησιμοποίησε όσες υπάρχουν (αντί για NaN)
```

---

## 6. List Comprehensions — Η Python "LINQ"

Θα τις δεις παντού:

```python
# C#:  prices.Select(p => p / 100.0).ToList()
# Python:
normalized = [p / 100.0 for p in prices]

# C#:  prices.Where(p => p > 100).Select(p => p / 100.0).ToList()
# Python:
expensive_norm = [p / 100.0 for p in prices if p > 100]

# Nested — π.χ. flatten a list of lists:
all_features = [f for group in feature_groups for f in group]
```

---

## 7. Dictionary Unpacking — Το `**kwargs` pattern

Θα το δεις στο configuration:

```python
# Στο main.py / train_masked.py:
battery_params = {
    'capacity_mwh': 146.0,
    'max_discharge_mw': 30.0,
    'efficiency': 0.94,
}

# Unpacking — "σπάει" το dict σε keyword arguments:
env = BatteryEnvMasked(df, **battery_params)
# Ισοδύναμο με:
# env = BatteryEnvMasked(df, capacity_mwh=146.0, max_discharge_mw=30.0, efficiency=0.94)
```

**Αντιστοιχία C#:** Δεν υπάρχει ακριβές αντίστοιχο. Το πιο κοντινό είναι ένα `params` ή named arguments.

---

## 8. Type Hints — Τα "Type Annotations" της Python

Η Python είναι dynamically typed, αλλά χρησιμοποιούμε type hints για documentation:

```python
from typing import Optional, List, Tuple, Dict

def calculate_reward(
    action_mw: float,              # float
    price: float,
    soc: float,
    dam_commitment: float = 0.0,   # default value
) -> Tuple[float, Dict[str, float]]:  # return type
    """I return (total_reward, breakdown_dict)."""
    ...
```

**Σημαντικό**: Τα type hints στην Python ΔΕΝ εφαρμόζονται (δεν κάνουν compile-time check). Είναι μόνο για documentation και IDE support.

---

## 9. `__init__` και `self` — Constructor & Instance

```python
class BatteryEnvMasked:
    def __init__(self, df, capacity_mwh=146.0, max_discharge_mw=30.0, efficiency=0.94):
        # self = this στο C#
        self.df = df
        self.capacity_mwh = capacity_mwh
        self.max_discharge_mw = max_discharge_mw
        self.efficiency = efficiency
        self.current_soc = 0.5   # Αρχική κατάσταση
        self.current_step = 0
```

**Αντιστοιχία C#:**
```csharp
public class BatteryEnvMasked {
    public BatteryEnvMasked(DataFrame df, double capacityMwh = 146.0, ...) {
        this.df = df;
        this.capacityMwh = capacityMwh;
        // ...
    }
}
```

**Βασική διαφορά**: Στην Python, το `self` πρέπει ΠΑΝΤΑ να είναι πρώτο argument σε instance methods. Στο C#, το `this` είναι implicit.

---

## 10. Πρακτική Άσκηση

Διάβασε τα αρχεία με αυτή τη σειρά και προσπάθησε να "μεταφράσεις" νοητικά σε C#:

1. **`config.py`** (~100 γραμμές) — Enums, dataclasses, factory functions
2. **`data/data_provider.py`** (πρώτες 50 γραμμές) — Interface + dataclasses
3. **`agent/decision_strategy.py`** (πρώτες 50 γραμμές) — Interface + strategy pattern

**Ερωτήσεις για self-check:**
- Πού είναι τα interfaces; (Ψάξε `class I....(ABC)`)
- Πού είναι τα DTOs; (Ψάξε `@dataclass`)
- Πού είναι τα factory methods; (Ψάξε `def create_...()`)
- Ποιο design pattern χρησιμοποιείται στο `decision_strategy.py`; (Strategy Pattern)

---

## Σύνοψη Ημέρας

| Θέμα | C# | Python | Αρχείο στο project |
|------|-----|--------|-------------------|
| Interfaces | `interface` | `ABC` + `@abstractmethod` | `data_provider.py` |
| DTOs | `record` | `@dataclass` | `data_provider.py` |
| Arrays | `float[]` | `np.array()` | `battery_env_masked.py` |
| Tables | `DataTable` | `pd.DataFrame` | `train_masked.py` |
| LINQ | `.Where().Select()` | List comprehensions | παντού |
| Config | `appsettings.json` | `dict` / `dataclass` | `config.py` |

> **Αύριο (Ημέρα 2)**: Θα μπούμε στο Gymnasium — πώς ορίζεται ένα RL environment και γιατί μοιάζει με video game.
