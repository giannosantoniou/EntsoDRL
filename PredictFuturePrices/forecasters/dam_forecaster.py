"""DAM (Day-Ahead Market) price forecaster.

I predict 24 hourly prices for the next day using LightGBM.
For each hour, I train 3 models: P10 (low), P50 (point), P90 (high).
Total: 72 LightGBM models.

Feature set (29 features):
- Serbia D+1 prices (4): mean, max, min, spread
- GR DAM lags (4): same hour at D-1, D-2, D-3, D-7
- Rolling stats (4): 24h mean, std, min, max
- Calendar (6): hour sin/cos, DoW sin/cos, month sin/cos
- Load/RES (2): load_forecast, res_forecast (normalized)
- RS-GR spread (1): Serbia minus Greece
- Weather (4): temperature, wind_100m, solar_radiation, cloud_cover
- Commodities (2): TTF gas, EUA carbon
- Forecast errors (2): load_forecast_error_lag, res_forecast_error_lag
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PredictFuturePrices.config import DAM_FORECASTER
from PredictFuturePrices.forecasters.base_forecaster import BaseForecaster

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class DAMForecaster(BaseForecaster):
    """I predict 24 hourly DAM prices with quantile confidence bands."""

    FEATURE_NAMES = [
        # Serbia D+1 (4)
        "rs_dam_mean", "rs_dam_max", "rs_dam_min", "rs_dam_spread",
        # GR DAM lags (4)
        "dam_lag_1d", "dam_lag_2d", "dam_lag_3d", "dam_lag_7d",
        # Rolling stats (4)
        "dam_mean_24h", "dam_std_24h", "dam_min_24h", "dam_max_24h",
        # Calendar (6)
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Load/RES (2)
        "load_forecast_norm", "res_forecast_norm",
        # Spread (1)
        "rs_gr_spread",
        # Weather (4)
        "temperature", "wind_100m", "solar_radiation", "cloud_cover",
        # Commodities (2)
        "ttf_gas", "eua_carbon",
        # Forecast errors (2)
        "load_forecast_error_lag", "res_forecast_error_lag",
    ]

    def __init__(self):
        super().__init__(name="dam", n_features=DAM_FORECASTER.n_features)
        self.feature_names = self.FEATURE_NAMES
        self.n_hours = DAM_FORECASTER.n_hours
        self.quantiles = DAM_FORECASTER.quantiles

    def build_features(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[np.ndarray]:
        """I build a 29-feature vector for DAM prediction.

        Args:
            df: Merged training DataFrame.
            idx: Current position (the row from which I predict tomorrow).
            **kwargs: 'hour' (int) — target delivery hour (0-23).

        Returns:
            1D numpy array of shape (29,) or None if insufficient data.
        """
        hour = kwargs.get("hour", 12)

        if idx < 168:  # I need at least 7 days of history
            return None

        row = df.iloc[idx]
        features = np.zeros(self.n_features)
        fi = 0

        # I extract Serbia D+1 stats (4 features)
        # I look at the Serbia prices from 24h before this index
        rs_col = "rs_dam_price"
        if rs_col in df.columns:
            rs_window = df[rs_col].iloc[max(0, idx - 24):idx]
            if len(rs_window) > 0:
                features[fi] = rs_window.mean()
                features[fi + 1] = rs_window.max()
                features[fi + 2] = rs_window.min()
                features[fi + 3] = rs_window.max() - rs_window.min()
        fi += 4

        # I extract GR DAM lags (4 features) — same hour at D-1, D-2, D-3, D-7
        dam_col = "dam_price_gr"
        if dam_col not in df.columns:
            dam_col = "price"

        for lag_days in [1, 2, 3, 7]:
            lag_idx = idx - lag_days * 24
            if lag_idx >= 0 and dam_col in df.columns:
                features[fi] = df[dam_col].iloc[lag_idx]
            fi += 1

        # I extract rolling stats (4 features)
        if dam_col in df.columns:
            window = df[dam_col].iloc[max(0, idx - 24):idx]
            if len(window) > 0:
                features[fi] = window.mean()
                features[fi + 1] = window.std() if len(window) > 1 else 0
                features[fi + 2] = window.min()
                features[fi + 3] = window.max()
        fi += 4

        # I extract calendar features (6 features)
        ts = df.index[idx]
        features[fi] = np.sin(2 * np.pi * hour / 24)
        features[fi + 1] = np.cos(2 * np.pi * hour / 24)
        features[fi + 2] = np.sin(2 * np.pi * ts.dayofweek / 7)
        features[fi + 3] = np.cos(2 * np.pi * ts.dayofweek / 7)
        features[fi + 4] = np.sin(2 * np.pi * ts.month / 12)
        features[fi + 5] = np.cos(2 * np.pi * ts.month / 12)
        fi += 6

        # I extract load/RES forecasts (2 features, normalized)
        for col, norm in [("load_forecast", 10000.0), ("res_forecast_mw", 5000.0)]:
            alt_col = col.replace("_mw", "")
            actual_col = col if col in df.columns else (alt_col if alt_col in df.columns else None)
            if actual_col and actual_col in df.columns:
                features[fi] = row.get(actual_col, 0) / norm
            fi += 1

        # I extract RS-GR spread (1 feature)
        spread_col = "rs_gr_spread"
        if spread_col in df.columns:
            features[fi] = row.get(spread_col, 0) / 50.0
        fi += 1

        # I extract weather features (4 features)
        for col, norm in [
            ("temperature", 40.0), ("wind_100m", 25.0),
            ("solar_radiation", 1000.0), ("cloud_cover", 100.0),
        ]:
            if col in df.columns:
                features[fi] = row.get(col, 0) / norm
            fi += 1

        # I extract commodity features (2 features)
        for col, norm in [("ttf_gas", 50.0), ("eua_carbon", 100.0)]:
            if col in df.columns:
                features[fi] = row.get(col, 0) / norm
            fi += 1

        # I extract forecast error lags (2 features)
        for col in ["load_forecast_error_lag", "res_forecast_error_lag"]:
            if col in df.columns:
                features[fi] = row.get(col, 0) / 500.0
            fi += 1

        return features

    def _extract_target(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[float]:
        """I extract the DAM price for the target hour."""
        hour = kwargs.get("hour", 12)
        target_idx = idx + hour + 1  # I predict next-day hours
        dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"

        if target_idx >= len(df) or dam_col not in df.columns:
            return None
        return float(df[dam_col].iloc[target_idx])

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
    ) -> "DAMForecaster":
        """I train 72 LightGBM models (24 hours × 3 quantiles).

        I use early stopping on the validation set.
        """
        if lgb is None:
            raise ImportError("LightGBM is required. Install: pip install lightgbm")

        logger.info(f"[DAM] Training on {train_end_idx} samples, validating on "
                     f"{val_end_idx - train_end_idx}")

        params = dict(DAM_FORECASTER.lgbm_params)

        for hour in range(self.n_hours):
            # I build datasets for this hour
            X_train, y_train = self.build_dataset(
                df, 168, train_end_idx, hour=hour
            )
            X_val, y_val = self.build_dataset(
                df, train_end_idx, val_end_idx, hour=hour
            )

            if len(X_train) < 50:
                logger.warning(f"[DAM] Hour {hour}: too few samples ({len(X_train)})")
                continue

            # I train quantile models
            for q in self.quantiles:
                model_key = f"h{hour}_q{q}"

                if q == 0.5:
                    # I use regression for the point estimate
                    model = lgb.LGBMRegressor(**params)
                else:
                    # I use quantile regression for P10/P90
                    q_params = {**params, "objective": "quantile", "alpha": q}
                    model = lgb.LGBMRegressor(**q_params)

                callbacks = []
                if len(X_val) > 0:
                    callbacks = [lgb.early_stopping(50, verbose=False)]
                    eval_set = [(X_val, y_val)]
                else:
                    eval_set = None

                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=callbacks,
                )
                self.models[model_key] = model

            if hour % 6 == 0:
                logger.info(f"[DAM] Trained hours 0-{hour}")

        self.is_fitted = True

        # I compute validation metrics
        if val_end_idx > train_end_idx:
            metrics = self.evaluate(df, train_end_idx, min(val_end_idx, len(df) - 25), hour=12)
            self.train_metrics = metrics
            logger.info(f"[DAM] Val metrics (hour 12): MAE={metrics['mae']:.2f}, "
                        f"RMSE={metrics['rmse']:.2f}")

        return self

    def predict(self, df: pd.DataFrame, idx: int, **kwargs) -> Dict[str, np.ndarray]:
        """I predict 24 hourly DAM prices with confidence bands.

        Returns:
            Dict with keys: "point" (24,), "p10" (24,), "p90" (24,)
        """
        if not self.is_fitted:
            raise RuntimeError("[DAM] Not fitted yet.")

        point = np.zeros(self.n_hours)
        p10 = np.zeros(self.n_hours)
        p90 = np.zeros(self.n_hours)

        for hour in range(self.n_hours):
            features = self.build_features(df, idx, hour=hour)
            if features is None:
                continue

            X = features.reshape(1, -1)

            for q, arr in [(0.1, p10), (0.5, point), (0.9, p90)]:
                model_key = f"h{hour}_q{q}"
                if model_key in self.models:
                    arr[hour] = self.models[model_key].predict(X)[0]

        return {"point": point, "p10": p10, "p90": p90}
