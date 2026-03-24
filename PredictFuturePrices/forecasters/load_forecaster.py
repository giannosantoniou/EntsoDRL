"""Load forecaster — predicts 96 quarter-hourly system load values.

I train 96 LightGBM models, one per period, using weather, calendar,
and historical load patterns.

Feature set (20 features):
- Load lags D-1/2/3/7 (4): same period historical load
- Rolling stats (4): 24h mean, std, min, max
- Temperature (2): current + 24h delta
- Calendar (6): period sin/cos, DoW sin/cos, month sin/cos
- Holiday flag (1): is_weekend
- RES forecast (1): renewable generation forecast
- Wind (1): wind speed at hub height
- Solar (1): direct radiation
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from PredictFuturePrices.config import LOAD_FORECASTER
from PredictFuturePrices.forecasters.base_forecaster import BaseForecaster

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class LoadForecaster(BaseForecaster):
    """I predict 96 quarter-hourly load values for the next day."""

    FEATURE_NAMES = [
        # Load lags (4)
        "load_lag_1d", "load_lag_2d", "load_lag_3d", "load_lag_7d",
        # Rolling stats (4)
        "load_mean_24h", "load_std_24h", "load_min_24h", "load_max_24h",
        # Temperature (2)
        "temperature", "temperature_delta_24h",
        # Calendar (6)
        "period_sin", "period_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Holiday (1)
        "is_weekend",
        # RES (1)
        "res_forecast_norm",
        # Weather (2)
        "wind_100m", "solar_radiation",
    ]

    def __init__(self):
        super().__init__(name="load", n_features=LOAD_FORECASTER.n_features)
        self.feature_names = self.FEATURE_NAMES
        self.n_periods = LOAD_FORECASTER.n_periods
        self.sph = 60 // LOAD_FORECASTER.resolution_minutes

    def _get_load_col(self, df: pd.DataFrame) -> Optional[str]:
        """I find the load column in the DataFrame."""
        for col in ["system_load_mw", "load_mw", "load_forecast"]:
            if col in df.columns:
                return col
        return None

    def build_features(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[np.ndarray]:
        """I build a 20-feature vector for load prediction."""
        period = kwargs.get("period", self.n_periods // 2)

        min_history = 7 * 24 * self.sph
        if idx < min_history:
            return None

        row = df.iloc[idx]
        features = np.zeros(self.n_features)
        fi = 0
        window_24h = 24 * self.sph

        load_col = self._get_load_col(df)
        norm = 10000.0  # I normalize load by typical Greek system load

        # I extract load lags (4 features)
        for lag_days in [1, 2, 3, 7]:
            lag_idx = idx - lag_days * 24 * self.sph
            if lag_idx >= 0 and load_col and load_col in df.columns:
                features[fi] = df[load_col].iloc[lag_idx] / norm
            fi += 1

        # I extract load rolling stats (4 features)
        if load_col and load_col in df.columns:
            window = df[load_col].iloc[max(0, idx - window_24h):idx]
            if len(window) > 0:
                features[fi] = window.mean() / norm
                features[fi + 1] = window.std() / norm if len(window) > 1 else 0
                features[fi + 2] = window.min() / norm
                features[fi + 3] = window.max() / norm
        fi += 4

        # I extract temperature features (2 features)
        if "temperature" in df.columns:
            features[fi] = row.get("temperature", 15) / 40.0
            if idx >= window_24h:
                temp_24h_ago = df["temperature"].iloc[idx - window_24h]
                features[fi + 1] = (row.get("temperature", 15) - temp_24h_ago) / 20.0
        fi += 2

        # I extract calendar features (6 features)
        ts = df.index[idx]
        features[fi] = np.sin(2 * np.pi * period / self.n_periods)
        features[fi + 1] = np.cos(2 * np.pi * period / self.n_periods)
        features[fi + 2] = np.sin(2 * np.pi * ts.dayofweek / 7)
        features[fi + 3] = np.cos(2 * np.pi * ts.dayofweek / 7)
        features[fi + 4] = np.sin(2 * np.pi * ts.month / 12)
        features[fi + 5] = np.cos(2 * np.pi * ts.month / 12)
        fi += 6

        # I extract weekend flag (1 feature)
        features[fi] = 1.0 if ts.dayofweek >= 5 else 0.0
        fi += 1

        # I extract RES forecast (1 feature)
        for col in ["res_forecast_mw", "res_forecast"]:
            if col in df.columns:
                features[fi] = row.get(col, 0) / 5000.0
                break
        fi += 1

        # I extract weather (2 features)
        if "wind_100m" in df.columns:
            features[fi] = row.get("wind_100m", 0) / 25.0
        fi += 1
        if "solar_radiation" in df.columns:
            features[fi] = row.get("solar_radiation", 0) / 1000.0
        fi += 1

        return features

    def build_features_matrix(
        self, df: pd.DataFrame, start_idx: int, end_idx: int, **kwargs
    ) -> Optional[tuple]:
        """I build all load features vectorized for speed."""
        period = kwargs.get("period", self.n_periods // 2)
        n = end_idx - start_idx
        if n <= 0:
            return np.empty((0, self.n_features)), np.empty(0), np.zeros(0, dtype=bool)

        X = np.zeros((n, self.n_features), dtype=np.float64)
        valid = np.ones(n, dtype=bool)

        min_history = 7 * 24 * self.sph
        for i in range(n):
            if start_idx + i < min_history:
                valid[i] = False

        sl = slice(start_idx, end_idx)
        fi = 0
        norm = 10000.0
        window_24h = 24 * self.sph

        load_col = self._get_load_col(df)

        # I compute load lags (4 features)
        if load_col and load_col in df.columns:
            for lag_i, lag_days in enumerate([1, 2, 3, 7]):
                shifted = df[load_col].shift(lag_days * 24 * self.sph).values
                X[:, fi + lag_i] = np.nan_to_num(shifted[sl], nan=0.0) / norm
        fi += 4

        # I compute load rolling stats (4 features)
        if load_col and load_col in df.columns:
            load_series = df[load_col]
            lm = load_series.rolling(window_24h, min_periods=1).mean().shift(1).bfill().values
            ls = load_series.rolling(window_24h, min_periods=1).std().shift(1).fillna(0).values
            ln = load_series.rolling(window_24h, min_periods=1).min().shift(1).bfill().values
            lx = load_series.rolling(window_24h, min_periods=1).max().shift(1).bfill().values
            X[:, fi] = lm[sl] / norm
            X[:, fi + 1] = ls[sl] / norm
            X[:, fi + 2] = ln[sl] / norm
            X[:, fi + 3] = lx[sl] / norm
        fi += 4

        # I compute temperature features (2 features)
        if "temperature" in df.columns:
            temp = df["temperature"].values
            X[:, fi] = np.nan_to_num(temp[sl], nan=15.0) / 40.0
            temp_24h = df["temperature"].shift(window_24h).values
            delta = (temp[sl] - np.nan_to_num(temp_24h[sl], nan=temp[sl])) / 20.0
            X[:, fi + 1] = np.nan_to_num(delta, nan=0.0)
        fi += 2

        # I compute calendar features (6 features)
        period_sin = np.sin(2 * np.pi * period / self.n_periods)
        period_cos = np.cos(2 * np.pi * period / self.n_periods)
        X[:, fi] = period_sin
        X[:, fi + 1] = period_cos
        ts_index = df.index[start_idx:end_idx]
        X[:, fi + 2] = np.sin(2 * np.pi * ts_index.dayofweek.values / 7)
        X[:, fi + 3] = np.cos(2 * np.pi * ts_index.dayofweek.values / 7)
        X[:, fi + 4] = np.sin(2 * np.pi * ts_index.month.values / 12)
        X[:, fi + 5] = np.cos(2 * np.pi * ts_index.month.values / 12)
        fi += 6

        # I compute weekend flag (1 feature)
        X[:, fi] = (ts_index.dayofweek.values >= 5).astype(float)
        fi += 1

        # I compute RES forecast (1 feature)
        for col in ["res_forecast_mw", "res_forecast"]:
            if col in df.columns:
                X[:, fi] = np.nan_to_num(df[col].values[sl], nan=0.0) / 5000.0
                break
        fi += 1

        # I compute weather (2 features)
        if "wind_100m" in df.columns:
            X[:, fi] = np.nan_to_num(df["wind_100m"].values[sl], nan=0.0) / 25.0
        fi += 1
        if "solar_radiation" in df.columns:
            X[:, fi] = np.nan_to_num(df["solar_radiation"].values[sl], nan=0.0) / 1000.0
        fi += 1

        # I compute targets — load at idx + period + 1
        y = np.full(n, np.nan)
        if load_col and load_col in df.columns:
            load_vals = df[load_col].values
            for i in range(n):
                target_idx = start_idx + i + period + 1
                if target_idx < len(df):
                    y[i] = load_vals[target_idx]

        valid &= ~np.any(np.isnan(X), axis=1)
        valid &= ~np.isnan(y)

        return X, y, valid

    def _extract_target(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[float]:
        """I extract the load value for the target period."""
        period = kwargs.get("period", self.n_periods // 2)
        target_idx = idx + period + 1
        load_col = self._get_load_col(df)

        if target_idx >= len(df) or load_col is None:
            return None
        return float(df[load_col].iloc[target_idx])

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
    ) -> "LoadForecaster":
        """I train n_periods LightGBM models, one per period.

        I auto-detect data resolution so lags and model count match the data.
        """
        if lgb is None:
            raise ImportError("LightGBM is required.")

        # I detect resolution from actual data, overriding config
        self.sph = self._detect_resolution(df)
        self.n_periods = 24 * self.sph

        logger.info(f"[Load] Training {self.n_periods} models (sph={self.sph})")

        params = dict(LOAD_FORECASTER.lgbm_params)
        min_history = 7 * 24 * self.sph

        for period in range(self.n_periods):
            X_train, y_train = self.build_dataset(df, min_history, train_end_idx, period=period)
            X_val, y_val = self.build_dataset(df, train_end_idx, val_end_idx, period=period)

            if len(X_train) < 50:
                logger.warning(f"[Load] Period {period}: too few samples ({len(X_train)})")
                continue

            model = lgb.LGBMRegressor(**params)
            callbacks = []
            eval_set = None
            if len(X_val) > 0:
                callbacks = [lgb.early_stopping(50, verbose=False)]
                eval_set = [(X_val, y_val)]

            model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
            self.models[f"p{period}"] = model

            # I free the datasets immediately after training this period
            del X_train, y_train, X_val, y_val

            if period % 24 == 0:
                logger.info(f"[Load] Trained periods 0-{period}")

        self.is_fitted = True

        if val_end_idx > train_end_idx:
            eval_period = self.n_periods // 2  # I use noon: hour 12 for hourly, period 48 for 15-min
            metrics = self.evaluate(df, train_end_idx, min(val_end_idx, len(df) - self.n_periods - 1), period=eval_period)
            self.train_metrics = metrics
            logger.info(f"[Load] Val MAE={metrics['mae']:.1f} MW (period {eval_period})")

        return self

    def predict(self, df: pd.DataFrame, idx: int, **kwargs) -> Dict[str, np.ndarray]:
        """I predict 96 quarter-hourly load values.

        Returns:
            Dict with key "point" → (96,) array of MW values.
        """
        if not self.is_fitted:
            raise RuntimeError("[Load] Not fitted yet.")

        point = np.zeros(self.n_periods)

        for period in range(self.n_periods):
            features = self.build_features(df, idx, period=period)
            if features is None:
                continue

            model_key = f"p{period}"
            if model_key in self.models:
                point[period] = self.models[model_key].predict(features.reshape(1, -1))[0]

        return {"point": point}
