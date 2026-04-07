"""
DAM Schedule Forecaster — I predict 24 hourly DAM prices using 1 LightGBM model.

Unlike the original DAMForecaster (288 models, 1 per period×quantile), I use
a SINGLE model with hour as an input feature. This gives:
- 43,200 training samples per model (vs 1,800 with per-hour models)
- Cross-hour pattern learning ("when morning is expensive, evening tends too")
- Simpler maintenance (1 model vs 288)

Gate: D-1 12:00 CET
Output: 24 hourly prices (EUR/MWh)
Post-processing: ScheduleOptimizer converts prices → MW[96] schedule

CRITICAL: I only use data available BEFORE 12:00 D-1.
- NO Serbia D+1 (published 12:50)
- NO GR DAM D+1 results (published ~14:00)
- NO ISP/IDA prices (future)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class DAMScheduleForecaster:
    """I predict 24 hourly DAM prices with a single LightGBM model.

    Feature vector: 42 base features + 1 target_hour = 43 total.
    I build features for ALL 24 hours at once, creating 24 rows per day.
    """

    FEATURE_NAMES = [
        # GR DAM lags (4) — same hour at D-1, D-2, D-3, D-7
        "dam_lag_1d", "dam_lag_2d", "dam_lag_3d", "dam_lag_7d",
        # Rolling stats (5) — backward-looking 24h
        "dam_mean_24h", "dam_std_24h", "dam_min_24h", "dam_max_24h", "dam_std_6h",
        # Calendar (6) — cyclical encoding
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Peak indicators (2)
        "is_peak_hour", "is_solar_hour",
        # Load/RES forecast (2) — TSO D+1 forecast, normalized
        "load_forecast_norm", "res_forecast_norm",
        # Weather D+1 forecast (4)
        "temperature", "wind_100m", "solar_radiation", "cloud_cover",
        # Commodities (2)
        "ttf_gas", "eua_carbon",
        # Gas momentum (2)
        "ttf_gas_change_24h", "ttf_gas_change_7d",
        # Neighbor DAM D-0 (4) — yesterday's prices
        "dam_bg", "dam_ro", "dam_hu", "dam_it_south",
        # Neighbor D+1 partial (4) — published before 12:00 for some countries
        "dam_bg_d1", "dam_ro_d1", "dam_hu_d1", "dam_it_south_d1",
        # Serbia D+1 target hour (1) — 0.90 correlation, strongest single predictor
        "rs_d1_target_hour",
        # Serbia D+1 stats (4) — aggregated from 24 hourly prices
        "rs_d1_mean", "rs_d1_max", "rs_d1_min", "rs_d1_spread",
        # RS-GR spread (1)
        "rs_gr_spread",
        # Forecast errors (2)
        "load_fcst_error_lag", "res_fcst_error_lag",
        # Peak-phase correction (8) — fixes 1-3h forecast phase shift
        "hours_to_load_peak_sin", "hours_to_load_peak_cos",
        "hours_to_serbia_peak_sin", "hours_to_serbia_peak_cos",
        "solar_descent_rate",
        "residual_load_at_hour",
        "position_in_load_curve",
        "temperature_peak_shift",
        # Target hour (2)
        "target_hour_sin", "target_hour_cos",
    ]

    N_FEATURES = 53  # 51 base + 2 target_hour encoding

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.train_metrics: Dict[str, float] = {}
        self.feature_names = self.FEATURE_NAMES

    def build_features_for_day(
        self, df: pd.DataFrame, decision_idx: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """I build feature matrix for ALL 24 target hours from one decision point.

        Args:
            df: DataFrame with price/market data, datetime index
            decision_idx: Index at D-1 ~12:00 (decision time)

        Returns:
            (X, hours) where X.shape = (24, N_FEATURES) and hours = [0..23]
            Returns None if insufficient history.
        """
        sph = self._detect_sph(df)
        min_history = 7 * 24 * sph
        if decision_idx < min_history:
            return None

        row = df.iloc[decision_idx]
        ts = df.index[decision_idx]
        w24 = 24 * sph
        w6 = 6 * sph

        # I extract price column
        pcol = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        prices = df[pcol].values if pcol in df.columns else None

        # I precompute peak timing data BEFORE the per-hour loop
        # CRITICAL: I use YESTERDAY's load/solar as proxy for tomorrow (no look-ahead!)
        # Yesterday's patterns correlate 0.83-0.88 with today's → excellent proxy
        load_24h = np.zeros(24)
        solar_24h = np.zeros(24)
        res_24h = np.zeros(24)
        for h in range(24):
            # I look at YESTERDAY same hour (24h ago from decision point)
            idx = decision_idx - w24 + h * sph
            if 0 <= idx < len(df):
                load_24h[h] = df['load_mw'].iloc[idx] if 'load_mw' in df.columns else 5000
                solar_24h[h] = df['solar'].iloc[idx] if 'solar' in df.columns else 0
                res_24h[h] = df['res_total_mw'].iloc[idx] if 'res_total_mw' in df.columns else 0

        # I find peak indices from YESTERDAY's pattern (no leakage)
        peak_load_idx = int(np.argmax(load_24h))
        load_min = np.min(load_24h) if np.max(load_24h) > 0 else 0
        load_max = np.max(load_24h) if np.max(load_24h) > 0 else 10000

        # I find Serbia D+1 peak hour
        rs_peak_idx = 18  # Default evening peak
        rs_prices_24h = np.zeros(24)
        for h in range(24):
            col = f"serbia_d1_h{h:02d}"
            if col in df.columns:
                v = row.get(col, 0)
                rs_prices_24h[h] = v if v > 1.0 else 0
        if np.max(rs_prices_24h) > 0:
            rs_peak_idx = int(np.argmax(rs_prices_24h))

        # I compute solar descent rate (steepness of solar drop 15h-19h)
        solar_15 = solar_24h[15] if len(solar_24h) > 15 else 0
        solar_19 = solar_24h[19] if len(solar_24h) > 19 else 0
        solar_descent = (solar_15 - solar_19) / max(solar_15, 1.0)  # 0-1 range

        # I precompute shared features (same for all 24 target hours)
        # 51 base features, then 2 target_hour appended per-row = 53 total
        shared = np.zeros(51)
        fi = 0

        # I skip first 4 features (lags) — they depend on target hour
        fi = 4

        # Rolling stats (5)
        if prices is not None:
            window = prices[max(0, decision_idx - w24):decision_idx]
            if len(window) > 0:
                shared[fi] = np.mean(window)
                shared[fi+1] = np.std(window) if len(window) > 1 else 0
                shared[fi+2] = np.min(window)
                shared[fi+3] = np.max(window)
            w6_data = prices[max(0, decision_idx - w6):decision_idx]
            if len(w6_data) > 1:
                shared[fi+4] = np.std(w6_data)
        fi += 5

        # Calendar (6)
        # I skip hour_sin/cos — they depend on target hour
        shared[fi+2] = np.sin(2 * np.pi * ts.dayofweek / 7)
        shared[fi+3] = np.cos(2 * np.pi * ts.dayofweek / 7)
        shared[fi+4] = np.sin(2 * np.pi * ts.month / 12)
        shared[fi+5] = np.cos(2 * np.pi * ts.month / 12)
        fi += 6

        # Peak indicators (2) — I skip, depend on target hour
        fi += 2

        # Load/RES forecast (2)
        for col, norm in [("load_mw", 10000.0), ("res_total_mw", 5000.0)]:
            if col in df.columns:
                shared[fi] = row.get(col, 0) / norm
            fi += 1

        # Weather D+1 (4)
        for col, norm in [("temperature", 40.0), ("wind_100m", 25.0),
                          ("solar_radiation", 1000.0), ("cloud_cover", 100.0)]:
            if col in df.columns:
                shared[fi] = row.get(col, 0) / norm
            fi += 1

        # Commodities (2)
        gas = row.get("ttf_gas_price", 0)
        shared[fi] = gas / 50.0 if gas > 0 else 0.7
        fi += 1
        shared[fi] = row.get("eua_carbon", 0) / 100.0
        fi += 1

        # Gas momentum (2)
        shared[fi] = row.get("ttf_gas_change_24h", 0) / 50.0
        fi += 1
        shared[fi] = row.get("ttf_gas_change_7d", 0) / 50.0
        fi += 1

        # Neighbor D-0 (4)
        for col in ["dam_price_bg", "dam_price_ro", "dam_price_hu", "dam_price_it_south"]:
            alt = col.replace("dam_price_", "")
            if col in df.columns:
                shared[fi] = row.get(col, 0) / 200.0
            fi += 1

        # Neighbor D+1 partial (4)
        for col in ["dam_price_bg_d1", "dam_price_ro_d1", "dam_price_hu_d1", "dam_price_it_south_d1"]:
            if col in df.columns:
                shared[fi] = row.get(col, 0) / 200.0
            fi += 1

        # Serbia D+1 target hour — I skip here, filled per-hour below
        fi += 1

        # Serbia D+1 stats (4) — aggregated from serbia_d1_h00..h23
        rs_d1_vals = []
        for h in range(24):
            col = f"serbia_d1_h{h:02d}"
            if col in df.columns:
                v = row.get(col, 0)
                if v > 1.0:
                    rs_d1_vals.append(v)
        if rs_d1_vals:
            rs_arr = np.array(rs_d1_vals)
            shared[fi] = np.mean(rs_arr) / 200.0
            shared[fi+1] = np.max(rs_arr) / 200.0
            shared[fi+2] = np.min(rs_arr) / 200.0
            shared[fi+3] = (np.max(rs_arr) - np.min(rs_arr)) / 200.0
        fi += 4

        # RS-GR spread (1) — I use Serbia D+1 mean vs GR recent mean
        rs_mean_d1 = shared[fi-4] * 200.0 if shared[fi-4] > 0 else 0  # rs_d1_mean (denormalized)
        gr_mean = shared[4] if shared[4] > 0 else 0  # dam_mean_24h (already normalized)
        shared[fi] = (rs_mean_d1 / 200.0 - gr_mean) / 100.0 if rs_mean_d1 > 0 else 0
        fi += 1

        # Forecast errors (2)
        shared[fi] = row.get("load_forecast_error_lag", 0) / 500.0
        fi += 1
        shared[fi] = row.get("res_forecast_error_lag", 0) / 500.0
        fi += 1

        # Peak-phase correction: I skip 8 features here (filled per-hour below)
        fi += 8

        # I build 24 rows — one per target hour
        X = np.zeros((24, self.N_FEATURES))
        target_hours = np.arange(24)

        for h in range(24):
            x = shared.copy()

            # I fill hour-dependent features
            # Lags (4): same hour at D-1, D-2, D-3, D-7
            if prices is not None:
                for li, lag_days in enumerate([1, 2, 3, 7]):
                    lag_idx = decision_idx - lag_days * w24 + h * sph
                    if 0 <= lag_idx < len(prices):
                        x[li] = prices[lag_idx] / 200.0

            # Hour sin/cos (indices 9, 10 in the shared array)
            x[9] = np.sin(2 * np.pi * h / 24)
            x[10] = np.cos(2 * np.pi * h / 24)

            # Peak indicators (indices 15, 16)
            x[15] = 1.0 if 17 <= h <= 21 else 0.0  # is_peak_hour
            x[16] = 1.0 if 9 <= h <= 15 else 0.0   # is_solar_hour

            # I fill Serbia D+1 target hour price (index 35)
            rs_d1_col = f"serbia_d1_h{h:02d}"
            if rs_d1_col in df.columns:
                rs_val = row.get(rs_d1_col, 0)
                x[35] = rs_val / 200.0 if rs_val > 1.0 else 0.0

            # Peak-phase correction features (indices 43-50)
            pfi = 43  # Start after forecast_errors (42)

            # Distance to load forecast peak (2): cyclical encoding
            hours_to_load_peak = (peak_load_idx - h) % 24
            x[pfi] = np.sin(2 * np.pi * hours_to_load_peak / 24)
            x[pfi+1] = np.cos(2 * np.pi * hours_to_load_peak / 24)

            # Distance to Serbia D+1 peak (2): cyclical encoding
            hours_to_rs_peak = (rs_peak_idx - h) % 24
            x[pfi+2] = np.sin(2 * np.pi * hours_to_rs_peak / 24)
            x[pfi+3] = np.cos(2 * np.pi * hours_to_rs_peak / 24)

            # Solar descent rate (1): how fast solar drops → predicts evening peak timing
            x[pfi+4] = solar_descent

            # Residual load at this hour (1): load - RES → net demand
            residual = load_24h[h] - res_24h[h]
            x[pfi+5] = residual / 10000.0

            # Position in daily load curve (1): 0=trough, 1=peak
            load_range = max(load_max - load_min, 1.0)
            x[pfi+6] = (load_24h[h] - load_min) / load_range

            # Temperature peak shift (1): cold→early peak, warm→late peak
            temp = row.get("temperature", 15) if "temperature" in df.columns else 15
            peak_shift = (15.0 - temp) * 0.1  # ±1.5 range
            x[pfi+7] = np.clip(peak_shift, -2.0, 2.0) / 2.0

            # I append target_hour encoding (features 51, 52)
            X[h, :51] = x
            X[h, 51] = np.sin(2 * np.pi * h / 24)  # target_hour_sin
            X[h, 52] = np.cos(2 * np.pi * h / 24)  # target_hour_cos

        return X, target_hours

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
    ) -> "DAMScheduleForecaster":
        """I train a single LightGBM model on all hours combined.

        I iterate over training days, build 24 rows per day, stack everything,
        and fit one regression model.
        """
        if lgb is None:
            raise ImportError("LightGBM required: pip install lightgbm")

        sph = self._detect_sph(df)
        day_steps = 24 * sph
        pcol = "dam_price_gr" if "dam_price_gr" in df.columns else "price"

        logger.info(f"[DAM-Schedule] Building training data (sph={sph})...")

        X_all, y_all = [], []
        min_history = 7 * day_steps

        # I iterate over days in training period
        for day_start in range(min_history, train_end_idx, day_steps):
            # I use the start of the day as decision point (simulating D-1 12:00)
            decision_idx = day_start
            result = self.build_features_for_day(df, decision_idx)
            if result is None:
                continue

            X_day, hours = result

            # I extract target prices (actual prices for this day)
            targets = np.full(24, np.nan)
            for h in range(24):
                # I target is the NEXT day's price at hour h
                target_idx = day_start + day_steps + h * sph
                if target_idx < len(df):
                    targets[h] = df[pcol].iloc[target_idx]

            valid = ~np.isnan(targets)
            if valid.sum() > 0:
                X_all.append(X_day[valid])
                y_all.append(targets[valid])

        if not X_all:
            logger.warning("[DAM-Schedule] No valid training data!")
            return self

        X_train = np.vstack(X_all)
        y_train = np.concatenate(y_all)

        # I split off validation
        val_X_all, val_y_all = [], []
        for day_start in range(train_end_idx, val_end_idx, day_steps):
            decision_idx = day_start
            result = self.build_features_for_day(df, decision_idx)
            if result is None:
                continue
            X_day, _ = result
            targets = np.full(24, np.nan)
            for h in range(24):
                target_idx = day_start + day_steps + h * sph
                if target_idx < len(df):
                    targets[h] = df[pcol].iloc[target_idx]
            valid = ~np.isnan(targets)
            if valid.sum() > 0:
                val_X_all.append(X_day[valid])
                val_y_all.append(targets[valid])

        X_val = np.vstack(val_X_all) if val_X_all else X_train[-500:]
        y_val = np.concatenate(val_y_all) if val_y_all else y_train[-500:]

        logger.info(f"[DAM-Schedule] Training: {len(X_train)} samples, "
                     f"Validation: {len(X_val)} samples")

        # I train LightGBM
        train_set = lgb.Dataset(X_train, label=y_train,
                                feature_name=self.FEATURE_NAMES)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "huber",
            "metric": "mae",
            "n_estimators": 800,
            "max_depth": 8,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
        }

        callbacks = [lgb.log_evaluation(period=100)]
        self.model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=params.pop("n_estimators"),
            callbacks=callbacks,
        )

        # I compute metrics
        val_pred = self.model.predict(X_val)
        mae = np.mean(np.abs(val_pred - y_val))
        rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        self.train_metrics = {"val_mae": mae, "val_rmse": rmse, "n_train": len(X_train)}
        self.is_fitted = True

        logger.info(f"[DAM-Schedule] Trained: MAE={mae:.2f} EUR/MWh, "
                     f"RMSE={rmse:.2f}, samples={len(X_train)}")

        return self

    def predict(
        self, df: pd.DataFrame, decision_idx: int
    ) -> Optional[np.ndarray]:
        """I predict 24 hourly prices for D+1.

        Args:
            df: DataFrame with current market data
            decision_idx: Index at D-1 ~12:00

        Returns:
            24-element array of predicted prices (EUR/MWh), or None if not fitted
        """
        if not self.is_fitted or self.model is None:
            return None

        result = self.build_features_for_day(df, decision_idx)
        if result is None:
            return None

        X, _ = result
        prices = self.model.predict(X)  # 24 predictions
        prices = np.maximum(prices, 0)  # I floor at 0 (no negative prices)
        return prices

    def save(self, path: str):
        """I save the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "metrics": self.train_metrics,
                "feature_names": self.feature_names,
            }, f)
        logger.info(f"[DAM-Schedule] Saved to {path}")

    def load(self, path: str) -> "DAMScheduleForecaster":
        """I load a previously trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.train_metrics = data.get("metrics", {})
        self.is_fitted = True
        logger.info(f"[DAM-Schedule] Loaded from {path}")
        return self

    def _detect_sph(self, df: pd.DataFrame) -> int:
        """I detect steps-per-hour from the DataFrame index."""
        if len(df) < 10:
            return 1
        diffs = pd.Series(df.index[:100]).diff().dropna()
        median_minutes = diffs.dt.total_seconds().median() / 60
        if median_minutes <= 20:
            return 4  # 15-min
        elif median_minutes <= 45:
            return 2  # 30-min
        return 1  # hourly
