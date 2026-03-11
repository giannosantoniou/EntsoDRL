# Data Sources & Training Pipeline

Τεκμηρίωση όλων των δεδομένων που συλλέγουμε, από ποιες πηγές, και πώς τα χρησιμοποιούμε
για την εκπαίδευση των μοντέλων πρόβλεψης τιμών (κυρίως DAM).

---

## 1. Πηγές Δεδομένων (Data Sources)

### 1.1 ENTSO-E Transparency Platform

**API**: `entsoe-py` Python library
**API Key**: Δωρεάν εγγραφή στο https://transparency.entsoe.eu/
**Connector**: `data/entsoe_connector.py`
**Collector**: `PredictFuturePrices/collectors/entsoe_collector.py`

#### 1.1.1 Τιμές DAM Ελλάδας (GR DAM Prices)

| Πεδίο | Τιμή |
|-------|------|
| **Κλάση** | `EntsoeDAMCollector` |
| **API Method** | `client.query_day_ahead_prices("GR", start, end)` |
| **Column** | `price` → μετονομάζεται σε `dam_price_gr` στο merge |
| **Μονάδα** | EUR/MWh |
| **Ανάλυση** | Ωριαία (hourly) |
| **Εύρος** | 2021-01-01 → σήμερα |
| **CSV** | `data/entsoe/gr_dam_prices.csv` |
| **Validation** | Εύρος -500 έως 5000 EUR/MWh, NaN check >50% |

**Γιατί το χρειαζόμαστε**: Είναι ο **κύριος στόχος πρόβλεψης** (target variable) του DAM forecaster. Αποτελεί τον πρωτεύοντα χρονικό άξονα (primary time axis) στο merge — όλα τα υπόλοιπα δεδομένα γίνονται left-join πάνω σε αυτό.

#### 1.1.2 Τιμές DAM Σερβίας (RS DAM Prices)

| Πεδίο | Τιμή |
|-------|------|
| **Κλάση** | `EntsoeSerbiaDAMCollector` |
| **API Method** | `client.query_day_ahead_prices("RS", start, end)` |
| **Column** | `rs_dam_price` |
| **Μονάδα** | EUR/MWh |
| **Ανάλυση** | Ωριαία (hourly) |
| **Εύρος** | 2021-01-01 → σήμερα |
| **CSV** | `data/entsoe/rs_dam_prices.csv` |

**Γιατί το χρειαζόμαστε**: Η δημοπρασία DAM της Σερβίας κλείνει **πριν** από την ελληνική (~12:00 vs 13:00). Οι τιμές της Σερβίας είναι διαθέσιμες πριν τη δική μας δημοπρασία και αποτελούν **leading indicator** για τις ελληνικές τιμές. Derived feature: `rs_gr_spread = rs_dam_price - dam_price_gr`.

#### 1.1.3 Πρόβλεψη Φορτίου (Load Forecast)

| Πεδίο | Τιμή |
|-------|------|
| **Κλάση** | `EntsoeLoadForecastCollector` |
| **API Method** | `client.query_load_forecast("GR", start, end)` |
| **Column** | `load_forecast` |
| **Μονάδα** | MWh |
| **Ανάλυση** | Ωριαία (hourly) |
| **Εύρος** | 2023-01-01 → σήμερα |
| **CSV** | `data/entsoe/load_forecast.csv` |

**Γιατί το χρειαζόμαστε**: Η πρόβλεψη φορτίου από τον TSO είναι **βασικός οδηγός τιμών**. Υψηλό φορτίο → υψηλή ζήτηση → υψηλότερες τιμές DAM. Χρησιμοποιείται και ως input στον Load Forecaster (σύγκριση πρόβλεψης vs πραγματικών).

#### 1.1.4 Πρόβλεψη ΑΠΕ (RES Forecast — Wind + Solar)

| Πεδίο | Τιμή |
|-------|------|
| **Κλάση** | `EntsoeRESForecastCollector` |
| **API Method** | `client.query_wind_and_solar_forecast("GR", start, end)` |
| **Columns** | `solar`, `wind_onshore` → παράγωγο `res_forecast_mw = solar + wind_onshore` |
| **Μονάδα** | MW |
| **Ανάλυση** | Ωριαία (hourly) |
| **Εύρος** | 2023-01-01 → σήμερα |
| **CSV** | `data/entsoe/res_forecast.csv` |

**Γιατί το χρειαζόμαστε**: Η παραγωγή ΑΠΕ (ηλιακά + αιολικά) πιέζει τις τιμές **προς τα κάτω** (merit order effect). Σε ώρες υψηλής ΑΠΕ παραγωγής, η DAM τιμή πέφτει σημαντικά — μερικές φορές ακόμα και σε αρνητικές τιμές.

---

### 1.2 ADMIE (Ελληνικός Διαχειριστής Μεταφοράς — TSO)

**API**: REST API `https://www.admie.gr/getOperationMarketFilewRange`
**Fetcher**: `data/admie_data_fetcher.py`
**Collector**: `PredictFuturePrices/collectors/admie_collector.py`
**Format**: Excel (.xlsx) αρχεία
**Rate limit**: 0.5s μεταξύ requests, fetch σε chunks των 30 ημερών

#### 1.2.1 ISP Clearing Prices (ISP1/ISP2/ISP3)

| Πεδίο | Τιμή |
|-------|------|
| **Κλάση** | `AdmieISPCollector` |
| **File Pattern** | `ISP{N}ISPResults_*.xlsx` |
| **Ανάλυση πηγής** | 15-min (15λεπτα) |
| **Resample** | Mean → ωριαία (για merge στο training) |
| **Εύρος** | 2024-01-01 → σήμερα |
| **CSV** | `data/admie/isp_clearing_prices.csv` |

**Columns ανά ISP session** (π.χ. ISP1):

| Column | Περιγραφή | Μονάδα |
|--------|-----------|--------|
| `isp1_afrr_price_up` | Τιμή aFRR up-regulation | EUR/MW/h |
| `isp1_afrr_price_down` | Τιμή aFRR down-regulation | EUR/MW/h |
| `isp1_afrr_requirements_up` | Απαίτηση aFRR up | MW |
| `isp1_afrr_requirements_down` | Απαίτηση aFRR down | MW |
| `isp1_mfrr_price_up` | Τιμή mFRR up-regulation | EUR/MW/h |
| `isp1_mfrr_price_down` | Τιμή mFRR down-regulation | EUR/MW/h |
| `isp1_mfrr_requirements_up` | Απαίτηση mFRR up | MW |
| `isp1_mfrr_requirements_down` | Απαίτηση mFRR down | MW |

> **Σημαντικό (Data Leakage Prevention)**: Χρησιμοποιούμε **μόνο ISP1** στο training. Τα ISP2/ISP3 δημοσιεύονται πιο κοντά στην ώρα παράδοσης (delivery time) και θα αποτελούσαν data leakage αν τα χρησιμοποιούσαμε για πρόβλεψη τιμών DAM.

**Γιατί το χρειαζόμαστε**: Οι τιμές ISP1 αντικατοπτρίζουν τη **στενότητα του Balancing Market** (balancing stress). Υψηλό ISP1 spread (`isp1_price_up - isp1_price_down`) σημαίνει αυξημένη αβεβαιότητα στο σύστημα, που συνήθως σημαίνει υψηλότερες ενδοημερήσιες τιμές. Derived feature: `isp1_spread`.

#### 1.2.2 IMBABE (Imbalance & Balancing Energy)

| Πεδίο | Τιμή |
|-------|------|
| **Κλάση** | `AdmieBalancingCollector` (μέρος του `fetch_all()`) |
| **File Pattern** | `IMBABE_*.xlsx` |
| **Ανάλυση πηγής** | 15-min |
| **Resample** | Mean → ωριαία |
| **Εύρος** | 2024-01-01 → σήμερα |
| **CSV** | `data/admie/admie_balancing.csv` (merged) |

**Columns**:

| Column | Περιγραφή | Μονάδα |
|--------|-----------|--------|
| `activated_energy_up_mwh` | Ενεργοποιημένη ενέργεια ανόδου | MWh |
| `activated_energy_down_mwh` | Ενεργοποιημένη ενέργεια καθόδου | MWh |
| `system_deviation_mwh` | Απόκλιση συστήματος | MWh |
| `imbalance_price` | Τιμή αδιάθετης ενέργειας | EUR/MWh |
| `mfrr_energy_price_up` | Τιμή ενέργειας mFRR up | EUR/MWh |
| `mfrr_energy_price_down` | Τιμή ενέργειας mFRR down | EUR/MWh |
| `uplift_1`, `uplift_2` | Λογαριασμοί uplift | EUR/MWh |

**Γιατί το χρειαζόμαστε**: Τα imbalance data δείχνουν πόσο **αποκλίνει** το σύστημα από τις DAM προβλέψεις. Μεγάλη system deviation σημαίνει ευκαιρίες στο balancing market. Derived feature: `net_imbalance_mw`, `mfrr_spread`.

#### 1.2.3 Balancing Energy (Ενεργοποιήσεις mFRR/aFRR)

| Πεδίο | Τιμή |
|-------|------|
| **File Pattern** | `BalancingEnergyProduct_*.xlsx` |
| **Ανάλυση πηγής** | 15-min |
| **Resample** | Mean → ωριαία |

**Columns**:

| Column | Περιγραφή | Μονάδα |
|--------|-----------|--------|
| `mfrr_activated_up_mwh` | mFRR ενεργοποίηση ανόδου | MWh |
| `mfrr_activated_down_mwh` | mFRR ενεργοποίηση καθόδου | MWh |
| `afrr_activated_up_mwh` | aFRR ενεργοποίηση ανόδου | MWh |
| `afrr_activated_down_mwh` | aFRR ενεργοποίηση καθόδου | MWh |

**Γιατί το χρειαζόμαστε**: Ο όγκος ενεργοποιήσεων δείχνει πόσο «δραστήριο» ήταν το balancing market. Χρησιμοποιείται ως `afrr_stress` feature στον IDA forecaster.

#### 1.2.4 System Load (Πραγματικό Φορτίο SCADA)

| Πεδίο | Τιμή |
|-------|------|
| **File Pattern** | `RealTimeSCADASystemLoad_*.xlsx` |
| **Column** | `system_load_mw` |
| **Μονάδα** | MW |
| **Ανάλυση** | Ωριαία |
| **CSV** | `data/admie/system_load.csv` |

**Γιατί το χρειαζόμαστε**: Σε σύγκριση με το `load_forecast`, δίνει το **σφάλμα πρόβλεψης φορτίου** (`load_forecast_error = system_load_mw - load_forecast`). Αυτό το σφάλμα (lagged κατά 24h) είναι feature στον DAM forecaster.

#### 1.2.5 RES Production (Πραγματική Παραγωγή ΑΠΕ SCADA)

| Πεδίο | Τιμή |
|-------|------|
| **File Pattern** | `RealTimeSCADARES_*.xlsx` |
| **Column** | `res_production_mw` |
| **Μονάδα** | MW |
| **Ανάλυση** | Ωριαία |
| **CSV** | `data/admie/res_actual.csv` |

**Γιατί το χρειαζόμαστε**: Σε σύγκριση με `res_forecast_mw`, δίνει `res_forecast_error`. Η αναπάντεχη παραγωγή ΑΠΕ (θετική ή αρνητική) επηρεάζει τις ενδοημερήσιες τιμές.

---

### 1.3 Open-Meteo Weather API

**API**: https://api.open-meteo.com (δωρεάν, χωρίς API key)
**Archive**: https://archive-api.open-meteo.com (ιστορικά >5 ημερών)
**Collector**: `PredictFuturePrices/collectors/weather_collector.py`
**Τοποθεσία**: Αθήνα (37.98°N, 23.73°E) — αντιπροσωπευτική για ελληνικό δίκτυο

| Column | API Variable | Περιγραφή | Μονάδα |
|--------|-------------|-----------|--------|
| `temperature` | `temperature_2m` | Θερμοκρασία στα 2m | °C |
| `wind_100m` | `wind_speed_100m` | Ταχύτητα ανέμου στα 100m | m/s |
| `solar_radiation` | `direct_radiation` | Άμεση ηλιακή ακτινοβολία | W/m² |
| `cloud_cover` | `cloud_cover` | Νεφοκάλυψη | % |

| Πεδίο | Τιμή |
|-------|------|
| **Ανάλυση** | Ωριαία |
| **Εύρος** | 2023-01-01 → σήμερα + 7 ημέρες forecast |
| **CSV** | `data/weather/weather_hourly.csv` |

**Γιατί τα χρειαζόμαστε**:
- **Θερμοκρασία** → Ψύξη/θέρμανση → Αυξημένη ζήτηση → Υψηλότερες τιμές
- **Ταχύτητα ανέμου** → Αιολική παραγωγή → Πίεση τιμών προς τα κάτω
- **Ηλιακή ακτινοβολία** → Φωτοβολταϊκή παραγωγή → Χαμηλές μεσημεριανές τιμές (duck curve)
- **Νεφοκάλυψη** → Μειωμένη ηλιακή παραγωγή → Πιο ακριβή ενέργεια

---

### 1.4 Yahoo Finance Commodities

**Library**: `yfinance`
**Collector**: `PredictFuturePrices/collectors/commodity_collector.py`

| Column | Ticker | Περιγραφή | Μονάδα |
|--------|--------|-----------|--------|
| `ttf_gas` | `TTF=F` | TTF Natural Gas Front-Month Futures | EUR/MWh |
| `eua_carbon` | `CO2.L` | EU Carbon Allowances (EUA) | EUR/tCO₂ |

| Πεδίο | Τιμή |
|-------|------|
| **Ανάλυση πηγής** | Ημερήσια (Daily Close) |
| **Resample** | Forward-fill → Ωριαία (ίδια τιμή για 24 ώρες) |
| **Εύρος** | 2023-01-01 → σήμερα |
| **CSV** | `data/commodities/commodity_prices.csv` |

**Γιατί τα χρειαζόμαστε**:
- **TTF Gas**: Το φυσικό αέριο είναι ο **marginal fuel** στην Ελλάδα. Η τιμή gas καθορίζει σε μεγάλο βαθμό τη DAM τιμή (cost of generation).
- **EUA Carbon**: Τα δικαιώματα εκπομπών CO₂ αυξάνουν το κόστος παραγωγής από fossil fuels, επηρεάζοντας τη DAM τιμή.

---

## 2. Data Merge Pipeline

**Module**: `PredictFuturePrices/data_merger.py`
**Output**: `data/merged_training.csv`

### 2.1 Στρατηγική Merge

```
GR DAM Prices (primary axis, hourly)
  ├── LEFT JOIN Serbia DAM (hourly)
  ├── LEFT JOIN ADMIE Balancing (15min → hourly mean)
  ├── LEFT JOIN ISP1 Clearing (15min → hourly mean, μόνο ISP1)
  ├── LEFT JOIN Load Forecast (hourly)
  ├── LEFT JOIN RES Forecast (hourly)
  ├── LEFT JOIN Weather (hourly)
  └── LEFT JOIN Commodities (daily → hourly ffill)
```

- **Trimming**: Περιορισμός σε `2023-01-01+` (πριν δεν υπάρχουν weather/commodities)
- **Forward-fill**: Μέχρι 3 ώρες κενά (short gaps)
- **Timezone**: Naive local time (Europe/Athens χωρίς tzinfo)

### 2.2 Derived Features (Παράγωγα Χαρακτηριστικά)

Τα παρακάτω features υπολογίζονται αυτόματα στο merge:

#### Calendar Features (7)
| Feature | Υπολογισμός | Σκοπός |
|---------|-------------|--------|
| `hour_sin` | sin(2π × hour / 24) | Κυκλική κωδικοποίηση ώρας |
| `hour_cos` | cos(2π × hour / 24) | Κυκλική κωδικοποίηση ώρας |
| `dow_sin` | sin(2π × day_of_week / 7) | Κυκλική κωδικοποίηση ημέρας |
| `dow_cos` | cos(2π × day_of_week / 7) | Κυκλική κωδικοποίηση ημέρας |
| `month_sin` | sin(2π × month / 12) | Κυκλική κωδικοποίηση μήνα |
| `month_cos` | cos(2π × month / 12) | Κυκλική κωδικοποίηση μήνα |
| `is_weekend` | dow >= 5 | Flag Σαββατοκύριακου |

#### DAM Rolling Stats (5)
| Feature | Υπολογισμός | Σκοπός |
|---------|-------------|--------|
| `dam_price_gr_mean_24h` | Rolling mean (24h window) | Μέσο επίπεδο τιμής |
| `dam_price_gr_std_24h` | Rolling std (24h window) | Μεταβλητότητα τιμής |
| `dam_price_gr_min_24h` | Rolling min (24h window) | Κατώτατο 24h |
| `dam_price_gr_max_24h` | Rolling max (24h window) | Ανώτατο 24h |
| `dam_price_gr_std_6h` | Rolling std (6h window) | Βραχυπρόθεσμη μεταβλητότητα |

#### DAM Price Lags (4)
| Feature | Υπολογισμός | Σκοπός |
|---------|-------------|--------|
| `dam_price_gr_lag_1d` | shift(24h) | Ίδια ώρα χθες |
| `dam_price_gr_lag_2d` | shift(48h) | Ίδια ώρα D-2 |
| `dam_price_gr_lag_3d` | shift(72h) | Ίδια ώρα D-3 |
| `dam_price_gr_lag_7d` | shift(168h) | Ίδια ώρα πριν 1 εβδομάδα |

#### Market Spreads (3)
| Feature | Υπολογισμός | Σκοπός |
|---------|-------------|--------|
| `rs_gr_spread` | rs_dam_price − dam_price_gr | Cross-border arbitrage signal |
| `isp1_spread` | isp1_price_up − isp1_price_down | Balancing market stress |
| `mfrr_spread` | mfrr_price_up − mfrr_price_down | mFRR market stress |

#### Forecast Errors (4)
| Feature | Υπολογισμός | Σκοπός |
|---------|-------------|--------|
| `load_forecast_error` | system_load_mw − load_forecast | Σφάλμα πρόβλεψης φορτίου |
| `load_forecast_error_lag` | load_forecast_error.shift(24h) | Lagged (χωρίς data leakage) |
| `res_forecast_error` | res_production_mw − res_forecast_mw | Σφάλμα πρόβλεψης ΑΠΕ |
| `res_forecast_error_lag` | res_forecast_error.shift(24h) | Lagged (χωρίς data leakage) |

---

## 3. Forecasters (Μοντέλα Πρόβλεψης)

### 3.1 DAM Forecaster (Πρόβλεψη Τιμών DAM)

| Πεδίο | Τιμή |
|-------|------|
| **Module** | `forecasters/dam_forecaster.py` |
| **Αλγόριθμος** | LightGBM |
| **Μοντέλα** | 288 (= 96 periods × 3 quantiles) |
| **Ανάλυση** | 15λεπτα (96 periods/day) |
| **Target** | Τιμή DAM (EUR/MWh) |
| **Quantiles** | P10, P50, P90 |

**Features (29)**:

| # | Feature | Normalization | Πηγή |
|---|---------|--------------|-------|
| 1-4 | `rs_dam_mean/max/min/spread` | 24h rolling stats Σερβίας | ENTSO-E RS |
| 5-8 | `dam_lag_1d/2d/3d/7d` | DAM τιμή ίδιας ώρας πριν 1/2/3/7 μέρες | ENTSO-E GR |
| 9-12 | `dam_mean/std/min/max_24h` | Rolling stats 24h DAM | ENTSO-E GR |
| 13-18 | `period_sin/cos, dow_sin/cos, month_sin/cos` | Calendar encoding | Υπολογίζονται |
| 19 | `load_forecast_norm` | ÷10000 | ENTSO-E |
| 20 | `res_forecast_norm` | ÷5000 | ENTSO-E |
| 21 | `rs_gr_spread` | ÷50 | Derived |
| 22-25 | `temperature, wind_100m, solar_radiation, cloud_cover` | ÷40, ÷25, ÷1000, ÷100 | Open-Meteo |
| 26-27 | `ttf_gas, eua_carbon` | ÷50, ÷100 | Yahoo Finance |
| 28-29 | `load_forecast_error_lag, res_forecast_error_lag` | ÷500 | Derived (lagged) |

**LightGBM Hyperparameters**:
- `n_estimators=500`, `max_depth=7`, `learning_rate=0.05`
- `num_leaves=63`, `min_child_samples=30`
- `subsample=0.8`, `colsample_bytree=0.8`
- `reg_alpha=0.1`, `reg_lambda=0.1`

### 3.2 Load Forecaster (Πρόβλεψη Φορτίου)

| Πεδίο | Τιμή |
|-------|------|
| **Module** | `forecasters/load_forecaster.py` |
| **Μοντέλα** | 96 (= 96 periods × 1 quantile) |
| **Target** | System load (MW) |

**Features (20)**:

| # | Feature | Normalization | Πηγή |
|---|---------|--------------|-------|
| 1-4 | `load_lag_1d/2d/3d/7d` | ÷10000 | ADMIE SCADA |
| 5-8 | `load_mean/std/min/max_24h` | Rolling stats | ADMIE SCADA |
| 9-10 | `temperature, temperature_delta_24h` | ÷40, ÷20 | Open-Meteo |
| 11-16 | `period_sin/cos, dow_sin/cos, month_sin/cos` | Calendar | Υπολογίζονται |
| 17 | `is_weekend` | Binary | Υπολογίζεται |
| 18 | `res_forecast_norm` | ÷5000 | ENTSO-E |
| 19-20 | `wind_100m, solar_radiation` | ÷25, ÷1000 | Open-Meteo |

### 3.3 IDA Forecaster (Πρόβλεψη Ενδοημερήσιων Δημοπρασιών)

| Πεδίο | Τιμή |
|-------|------|
| **Module** | `forecasters/ida_forecaster.py` |
| **Μοντέλα** | 3 (IDA1, IDA2, IDA3) |
| **Target** | IDA clearing prices (EUR/MWh) |

**Gate Closures**:
- IDA1: D-1 15:00 (σταθεροποίηση schedule)
- IDA2: D-1 22:00 (βραδινή αναθεώρηση)
- IDA3: D+0 10:00 (πρωινή αναθεώρηση, ώρες 12-23)

**Features (30)**:

| # | Feature | Πηγή |
|---|---------|-------|
| 1-5 | DAM stats (current, mean/std/max/min 24h) | ENTSO-E GR |
| 6-8 | ISP1 signal (up, down, bid-ask spread) | ADMIE ISP1 |
| 9-11 | Imbalance (direction, magnitude, vs DAM) | ADMIE IMBABE |
| 12-13 | Balancing stress (aFRR stress, mFRR spread) | ADMIE |
| 14-17 | IDA lags (1d, 2d, 3d, 7d) | Historical IDA |
| 18-23 | Calendar (hour/dow/month sin/cos) | Υπολογίζονται |
| 24 | Cross-border spread (RS-GR) | Derived |
| 25 | IDA number encoding (1/2/3) | Gate identity |
| 26-28 | Revisions (wind, solar, load) | Forecasts |
| 29-30 | Commodities (TTF gas, DAM forecast error) | Yahoo Finance |

### 3.4 XBID Forecaster (Πρόβλεψη Continuous Intraday)

| Πεδίο | Τιμή |
|-------|------|
| **Module** | `forecasters/xbid_forecaster.py` |
| **Μοντέλα** | 4 (horizons: 1h, 2h, 4h, 8h) |
| **Target** | XBID mid-price (EUR/MWh) |

**Features (37)** — βλ. `xbid_forecaster.py` για πλήρη λίστα. Περιλαμβάνουν time encoding, RES, ISP1 signal, DAM level, imbalance, price lags (1h-168h), rolling mean/std (6h/12h/24h), weather, commodity.

---

## 4. Δομή Αρχείων (Directory Structure)

```
PredictFuturePrices/
├── data/
│   ├── entsoe/
│   │   ├── gr_dam_prices.csv          ← Τιμές DAM Ελλάδας (ωριαίες)
│   │   ├── rs_dam_prices.csv          ← Τιμές DAM Σερβίας (ωριαίες)
│   │   ├── load_forecast.csv          ← Πρόβλεψη φορτίου ENTSO-E (ωριαία)
│   │   └── res_forecast.csv           ← Πρόβλεψη ΑΠΕ ENTSO-E (ωριαία)
│   ├── admie/
│   │   ├── admie_balancing.csv        ← ISP+IMBABE+activations (ωριαίο merged)
│   │   ├── isp_clearing_prices.csv    ← ISP1/2/3 (15-min resolution)
│   │   ├── system_load.csv            ← SCADA φορτίο (ωριαίο)
│   │   ├── res_actual.csv             ← SCADA ΑΠΕ (ωριαίο)
│   │   └── cache/                     ← Cached ADMIE Excel αρχεία
│   │       ├── isp1/                  ← ISP1ISPResults_*.xlsx
│   │       ├── isp2/                  ← ISP2ISPResults_*.xlsx
│   │       ├── isp3/                  ← ISP3ISPResults_*.xlsx
│   │       └── imbabe/                ← IMBABE_*.xlsx
│   ├── weather/
│   │   └── weather_hourly.csv         ← Μετεωρολογικά (ωριαία)
│   ├── commodities/
│   │   └── commodity_prices.csv       ← TTF Gas + EUA Carbon (ωριαία)
│   └── merged_training.csv            ← ΤΕΛΙΚΟ MERGED DATASET
├── models/
│   ├── dam_forecaster.pkl             ← 288 LightGBM μοντέλα DAM
│   ├── load_forecaster.pkl            ← 96 LightGBM μοντέλα φορτίου
│   ├── ida_forecaster.pkl             ← 3 LightGBM μοντέλα IDA
│   └── xbid_forecaster.pkl            ← 4 LightGBM μοντέλα XBID
└── accuracy/reports/
    └── {forecaster}_accuracy.csv      ← Rolling accuracy metrics
```

---

## 5. Εντολές Εκτέλεσης (Commands)

### Βήμα 1: Συλλογή Δεδομένων
```bash
# Πρέπει να υπάρχει ENTSOE_API_KEY στο .env
python -m PredictFuturePrices.collect_all
```

### Βήμα 2: Merge σε ενιαίο dataset
```bash
python -m PredictFuturePrices.data_merger
# Output: data/merged_training.csv
```

### Βήμα 3: Εκπαίδευση μοντέλων
```bash
python -m PredictFuturePrices.train --forecasters dam,load,ida,xbid
# Train split: 70%, Validation: 15%, Test: 15% (χρονολογική σειρά)
```

### Βήμα 4: Πρόβλεψη (Production)
```bash
python -m PredictFuturePrices.predict --fill-actuals --summary
```

---

## 6. Σημαντικές Σημειώσεις

### Data Leakage Prevention
1. **ISP1 only**: Χρησιμοποιούμε μόνο ISP1 (όχι ISP2/ISP3) γιατί ISP2/3 δημοσιεύονται πιο κοντά στο delivery
2. **Lagged errors**: Τα `load_forecast_error` και `res_forecast_error` χρησιμοποιούνται μόνο με lag 24h (shift)
3. **No future data**: Όλα τα rolling stats υπολογίζονται με `min_periods=1` (backward-looking only)

### Ανάλυση Δεδομένων (Resolution)
- **Πηγές 15-min** (ISP, IMBABE): Resample σε ωριαία με `mean()` πριν το merge
- **Πηγές daily** (commodities): Forward-fill σε ωριαία (ίδια τιμή για 24h)
- **Merged dataset**: Ωριαίο (hourly)
- **Forecasters**: Εσωτερικά δουλεύουν σε 15-min (96 periods/day) μέσω interpolation

### Minimum History
- Όλοι οι forecasters χρειάζονται **τουλάχιστον 7 ημέρες** ιστορικών δεδομένων (168 ωριαίες εγγραφές) για να δημιουργήσουν lag features (7-day lag).

### Κανονικοποίηση (Normalization)
- Κάθε feature κανονικοποιείται με **domain-specific factors** (π.χ. load ÷ 10000, temperature ÷ 40) αντί για z-score, ώστε τα μοντέλα να λειτουργούν σταθερά χωρίς να χρειάζεται επανεκπαίδευση scaler.
