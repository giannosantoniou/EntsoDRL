"""DAM (Day-Ahead Market) price forecaster.

I predict 96 quarter-hourly prices for the next day using LightGBM.
For each period, I train 3 models: P10 (low), P50 (point), P90 (high).
Total: 288 LightGBM models (96 periods × 3 quantiles).

Feature set (29 features):
- Serbia D+1 prices (4): mean, max, min, spread
- GR DAM lags (4): same period at D-1, D-2, D-3, D-7
- Rolling stats (4): 24h mean, std, min, max
- Calendar (6): period sin/cos, DoW sin/cos, month sin/cos
- Load/RES (2): load_forecast, res_forecast (normalized)
- RS-GR spread (1): Serbia minus Greece
- Weather (4): temperature, wind_100m, solar_radiation, cloud_cover
- Commodities (2): TTF gas, EUA carbon
- Forecast errors (2): load_forecast_error_lag, res_forecast_error_lag
"""

import gc
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
    """I predict 96 quarter-hourly DAM prices with quantile confidence bands."""

    FEATURE_NAMES = [
        # Serbia D+1 (4)
        "rs_dam_mean", "rs_dam_max", "rs_dam_min", "rs_dam_spread",
        # GR DAM lags (4)
        "dam_lag_1d", "dam_lag_2d", "dam_lag_3d", "dam_lag_7d",
        # Rolling stats (4)
        "dam_mean_24h", "dam_std_24h", "dam_min_24h", "dam_max_24h",
        # Calendar (6)
        "period_sin", "period_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
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
        self.n_periods = DAM_FORECASTER.n_periods
        self.sph = 60 // DAM_FORECASTER.resolution_minutes  # steps per hour
        self.quantiles = DAM_FORECASTER.quantiles

    def build_features(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[np.ndarray]:
        """I build a 29-feature vector for DAM prediction.

        Args:
            df: Merged training DataFrame.
            idx: Current position (the row from which I predict tomorrow).
            **kwargs: 'period' (int) — target delivery period (0-95).

        Returns:
            1D numpy array of shape (29,) or None if insufficient data.
        """
        period = kwargs.get("period", self.n_periods // 2)

        min_history = 7 * 24 * self.sph  # I need at least 7 days of history
        if idx < min_history:
            return None

        row = df.iloc[idx]
        features = np.zeros(self.n_features)
        fi = 0
        window_24h = 24 * self.sph

        # I extract Serbia D+1 stats (4 features)
        # I look at the Serbia prices from 24h before this index
        rs_col = "rs_dam_price"
        if rs_col in df.columns:
            rs_window = df[rs_col].iloc[max(0, idx - window_24h):idx]
            if len(rs_window) > 0:
                features[fi] = rs_window.mean()
                features[fi + 1] = rs_window.max()
                features[fi + 2] = rs_window.min()
                features[fi + 3] = rs_window.max() - rs_window.min()
        fi += 4

        # I extract GR DAM lags (4 features) — same period at D-1, D-2, D-3, D-7
        dam_col = "dam_price_gr"
        if dam_col not in df.columns:
            dam_col = "price"

        for lag_days in [1, 2, 3, 7]:
            lag_idx = idx - lag_days * 24 * self.sph
            if lag_idx >= 0 and dam_col in df.columns:
                features[fi] = df[dam_col].iloc[lag_idx]
            fi += 1

        # I extract rolling stats (4 features)
        if dam_col in df.columns:
            window = df[dam_col].iloc[max(0, idx - window_24h):idx]
            if len(window) > 0:
                features[fi] = window.mean()
                features[fi + 1] = window.std() if len(window) > 1 else 0
                features[fi + 2] = window.min()
                features[fi + 3] = window.max()
        fi += 4

        # I extract calendar features (6 features)
        # I use sin(2π * period / n_periods) — mathematically equivalent to hour encoding
        ts = df.index[idx]
        features[fi] = np.sin(2 * np.pi * period / self.n_periods)
        features[fi + 1] = np.cos(2 * np.pi * period / self.n_periods)
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

    def build_features_matrix(
        self, df: pd.DataFrame, start_idx: int, end_idx: int, **kwargs
    ) -> Optional[tuple]:
        """I build all features vectorized — 100x faster than row-by-row.

        I pre-compute rolling stats, lags, and calendar features using
        pandas/numpy vectorized operations, then slice the range.
        """
        period = kwargs.get("period", self.n_periods // 2)
        n = end_idx - start_idx
        if n <= 0:
            return np.empty((0, self.n_features)), np.empty(0), np.zeros(0, dtype=bool)

        X = np.zeros((n, self.n_features), dtype=np.float64)
        valid = np.ones(n, dtype=bool)

        min_history = 7 * 24 * self.sph
        # I mark rows with insufficient history as invalid
        for i in range(n):
            if start_idx + i < min_history:
                valid[i] = False

        sl = slice(start_idx, end_idx)
        fi = 0
        window_24h = 24 * self.sph

        # I compute Serbia stats (4 features) — rolling 24h window
        rs_col = "rs_dam_price"
        if rs_col in df.columns:
            # I use shift(1) so rolling excludes current row (matching build_features)
            rs_series = df[rs_col]
            rs_mean = rs_series.rolling(window_24h, min_periods=1).mean().shift(1).bfill().values
            rs_max = rs_series.rolling(window_24h, min_periods=1).max().shift(1).bfill().values
            rs_min = rs_series.rolling(window_24h, min_periods=1).min().shift(1).bfill().values
            X[:, fi] = rs_mean[sl]
            X[:, fi + 1] = rs_max[sl]
            X[:, fi + 2] = rs_min[sl]
            X[:, fi + 3] = rs_max[sl] - rs_min[sl]
        fi += 4

        # I compute GR DAM lags (4 features) — shift by D-1/2/3/7
        dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        if dam_col in df.columns:
            for lag_i, lag_days in enumerate([1, 2, 3, 7]):
                lag = lag_days * 24 * self.sph
                shifted = df[dam_col].shift(lag).values
                X[:, fi + lag_i] = np.nan_to_num(shifted[sl], nan=0.0)
        fi += 4

        # I compute rolling stats (4 features) — 24h window excluding current
        if dam_col in df.columns:
            dam_series = df[dam_col]
            d_mean = dam_series.rolling(window_24h, min_periods=1).mean().shift(1).bfill().values
            d_std = dam_series.rolling(window_24h, min_periods=1).std().shift(1).fillna(0).values
            d_min = dam_series.rolling(window_24h, min_periods=1).min().shift(1).bfill().values
            d_max = dam_series.rolling(window_24h, min_periods=1).max().shift(1).bfill().values
            X[:, fi] = d_mean[sl]
            X[:, fi + 1] = d_std[sl]
            X[:, fi + 2] = d_min[sl]
            X[:, fi + 3] = d_max[sl]
        fi += 4

        # I compute calendar features (6 features) — period depends on kwarg
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

        # I compute load/RES (2 features, normalized)
        for col, norm in [("load_forecast", 10000.0), ("res_forecast_mw", 5000.0)]:
            alt_col = col.replace("_mw", "")
            actual_col = col if col in df.columns else (alt_col if alt_col in df.columns else None)
            if actual_col:
                X[:, fi] = np.nan_to_num(df[actual_col].values[sl], nan=0.0) / norm
            fi += 1

        # I compute RS-GR spread (1 feature)
        spread_col = "rs_gr_spread"
        if spread_col in df.columns:
            X[:, fi] = np.nan_to_num(df[spread_col].values[sl], nan=0.0) / 50.0
        fi += 1

        # I compute weather features (4 features)
        for col, norm in [
            ("temperature", 40.0), ("wind_100m", 25.0),
            ("solar_radiation", 1000.0), ("cloud_cover", 100.0),
        ]:
            if col in df.columns:
                X[:, fi] = np.nan_to_num(df[col].values[sl], nan=0.0) / norm
            fi += 1

        # I compute commodities (2 features)
        for col, norm in [("ttf_gas", 50.0), ("eua_carbon", 100.0)]:
            if col in df.columns:
                X[:, fi] = np.nan_to_num(df[col].values[sl], nan=0.0) / norm
            fi += 1

        # I compute forecast error lags (2 features)
        for col in ["load_forecast_error_lag", "res_forecast_error_lag"]:
            if col in df.columns:
                X[:, fi] = np.nan_to_num(df[col].values[sl], nan=0.0) / 500.0
            fi += 1

        # I compute targets — DAM price at idx + period + 1
        y = np.full(n, np.nan)
        if dam_col in df.columns:
            dam_vals = df[dam_col].values
            for i in range(n):
                target_idx = start_idx + i + period + 1
                if target_idx < len(df):
                    y[i] = dam_vals[target_idx]

        # I invalidate rows with NaN features or NaN targets
        valid &= ~np.any(np.isnan(X), axis=1)
        valid &= ~np.isnan(y)

        return X, y, valid

    def _extract_target(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[float]:
        """I extract the DAM price for the target period."""
        period = kwargs.get("period", self.n_periods // 2)
        target_idx = idx + period + 1  # I predict next-day periods
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
        """I train n_periods × 3 quantile LightGBM models.

        I auto-detect data resolution so lags and model count match the data.
        """
        if lgb is None:
            raise ImportError("LightGBM is required. Install: pip install lightgbm")

        # I detect resolution from actual data, overriding config
        self.sph = self._detect_resolution(df)
        self.n_periods = 24 * self.sph

        logger.info(f"[DAM] Training {self.n_periods} periods × {len(self.quantiles)} quantiles "
                     f"= {self.n_periods * len(self.quantiles)} models")
        logger.info(f"[DAM] Training on {train_end_idx} samples, validating on "
                     f"{val_end_idx - train_end_idx}")

        params = dict(DAM_FORECASTER.lgbm_params)
        min_history = 7 * 24 * self.sph

        for period in range(self.n_periods):
            # I build datasets for this period
            X_train, y_train = self.build_dataset(
                df, min_history, train_end_idx, period=period
            )
            X_val, y_val = self.build_dataset(
                df, train_end_idx, val_end_idx, period=period
            )

            if len(X_train) < 50:
                logger.warning(f"[DAM] Period {period}: too few samples ({len(X_train)})")
                continue

            # I train quantile models
            for q in self.quantiles:
                model_key = f"p{period}_q{q}"

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

            # I free the datasets immediately after training this period
            del X_train, y_train, X_val, y_val

            if period % 24 == 0:
                logger.info(f"[DAM] Trained periods 0-{period}")

        # I trigger garbage collection after all 288 models are trained
        gc.collect()

        self.is_fitted = True

        # I compute validation metrics using midday period (noon)
        if val_end_idx > train_end_idx:
            eval_period = self.n_periods // 2  # I use noon: hour 12 for hourly, period 48 for 15-min
            metrics = self.evaluate(df, train_end_idx, min(val_end_idx, len(df) - self.n_periods - 1), period=eval_period)
            self.train_metrics = metrics
            logger.info(f"[DAM] Val metrics (period {eval_period}): MAE={metrics['mae']:.2f}, "
                        f"RMSE={metrics['rmse']:.2f}")

        return self

    def predict(self, df: pd.DataFrame, idx: int, **kwargs) -> Dict[str, np.ndarray]:
        """I predict 96 quarter-hourly DAM prices with confidence bands.

        Returns:
            Dict with keys: "point" (96,), "p10" (96,), "p90" (96,)
        """
        if not self.is_fitted:
            raise RuntimeError("[DAM] Not fitted yet.")

        point = np.zeros(self.n_periods)
        p10 = np.zeros(self.n_periods)
        p90 = np.zeros(self.n_periods)

        for period in range(self.n_periods):
            features = self.build_features(df, idx, period=period)
            if features is None:
                continue

            X = features.reshape(1, -1)

            for q, arr in [(0.1, p10), (0.5, point), (0.9, p90)]:
                model_key = f"p{period}_q{q}"
                if model_key in self.models:
                    arr[period] = self.models[model_key].predict(X)[0]

        return {"point": point, "p10": p10, "p90": p90}
