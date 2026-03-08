"""Load forecaster — predicts 24 hourly system load values.

I train 24 LightGBM models, one per hour, using weather, calendar,
and historical load patterns.

Feature set (20 features):
- Load lags D-1/2/3/7 (4): same hour historical load
- Rolling stats (4): 24h mean, std, min, max
- Temperature (2): current + 24h delta
- Calendar (6): hour sin/cos, DoW sin/cos, month sin/cos
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
    """I predict 24 hourly load values for the next day."""

    FEATURE_NAMES = [
        # Load lags (4)
        "load_lag_1d", "load_lag_2d", "load_lag_3d", "load_lag_7d",
        # Rolling stats (4)
        "load_mean_24h", "load_std_24h", "load_min_24h", "load_max_24h",
        # Temperature (2)
        "temperature", "temperature_delta_24h",
        # Calendar (6)
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
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
        self.n_hours = LOAD_FORECASTER.n_hours

    def _get_load_col(self, df: pd.DataFrame) -> Optional[str]:
        """I find the load column in the DataFrame."""
        for col in ["system_load_mw", "load_mw", "load_forecast"]:
            if col in df.columns:
                return col
        return None

    def build_features(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[np.ndarray]:
        """I build a 20-feature vector for load prediction."""
        hour = kwargs.get("hour", 12)

        if idx < 168:
            return None

        row = df.iloc[idx]
        features = np.zeros(self.n_features)
        fi = 0

        load_col = self._get_load_col(df)
        norm = 10000.0  # I normalize load by typical Greek system load

        # I extract load lags (4 features)
        for lag_days in [1, 2, 3, 7]:
            lag_idx = idx - lag_days * 24
            if lag_idx >= 0 and load_col and load_col in df.columns:
                features[fi] = df[load_col].iloc[lag_idx] / norm
            fi += 1

        # I extract load rolling stats (4 features)
        if load_col and load_col in df.columns:
            window = df[load_col].iloc[max(0, idx - 24):idx]
            if len(window) > 0:
                features[fi] = window.mean() / norm
                features[fi + 1] = window.std() / norm if len(window) > 1 else 0
                features[fi + 2] = window.min() / norm
                features[fi + 3] = window.max() / norm
        fi += 4

        # I extract temperature features (2 features)
        if "temperature" in df.columns:
            features[fi] = row.get("temperature", 15) / 40.0
            if idx >= 24:
                temp_24h_ago = df["temperature"].iloc[idx - 24]
                features[fi + 1] = (row.get("temperature", 15) - temp_24h_ago) / 20.0
        fi += 2

        # I extract calendar features (6 features)
        ts = df.index[idx]
        features[fi] = np.sin(2 * np.pi * hour / 24)
        features[fi + 1] = np.cos(2 * np.pi * hour / 24)
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

    def _extract_target(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[float]:
        """I extract the load value for the target hour."""
        hour = kwargs.get("hour", 12)
        target_idx = idx + hour + 1
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
        """I train 24 LightGBM models, one per hour."""
        if lgb is None:
            raise ImportError("LightGBM is required.")

        params = dict(LOAD_FORECASTER.lgbm_params)

        for hour in range(self.n_hours):
            X_train, y_train = self.build_dataset(df, 168, train_end_idx, hour=hour)
            X_val, y_val = self.build_dataset(df, train_end_idx, val_end_idx, hour=hour)

            if len(X_train) < 50:
                logger.warning(f"[Load] Hour {hour}: too few samples ({len(X_train)})")
                continue

            model = lgb.LGBMRegressor(**params)
            callbacks = []
            eval_set = None
            if len(X_val) > 0:
                callbacks = [lgb.early_stopping(50, verbose=False)]
                eval_set = [(X_val, y_val)]

            model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
            self.models[f"h{hour}"] = model

            if hour % 6 == 0:
                logger.info(f"[Load] Trained hours 0-{hour}")

        self.is_fitted = True

        if val_end_idx > train_end_idx:
            metrics = self.evaluate(df, train_end_idx, min(val_end_idx, len(df) - 25), hour=12)
            self.train_metrics = metrics
            logger.info(f"[Load] Val MAE={metrics['mae']:.1f} MW")

        return self

    def predict(self, df: pd.DataFrame, idx: int, **kwargs) -> Dict[str, np.ndarray]:
        """I predict 24 hourly load values.

        Returns:
            Dict with key "point" → (24,) array of MW values.
        """
        if not self.is_fitted:
            raise RuntimeError("[Load] Not fitted yet.")

        point = np.zeros(self.n_hours)

        for hour in range(self.n_hours):
            features = self.build_features(df, idx, hour=hour)
            if features is None:
                continue

            model_key = f"h{hour}"
            if model_key in self.models:
                point[hour] = self.models[model_key].predict(features.reshape(1, -1))[0]

        return {"point": point}
