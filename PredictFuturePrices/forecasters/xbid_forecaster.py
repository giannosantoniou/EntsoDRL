"""XBID (Cross-border Intraday) forecaster — predicts continuous market prices.

I train 4 LightGBM models for different horizons: 1h, 2h, 4h, 8h ahead.
XBID prices represent the continuous intraday market where the agent
can trade discretionary volumes (unlike ISP balancing which is TSO-activated).

Feature set (37 features):
- Time features (7): hour sin/cos, DoW sin/cos, month sin/cos, hour/24
- RES (3): solar, wind, total normalized
- ISP1 signal (4): up/down prices, spread, lagged
- DAM (2): current price, DAM vs 24h mean
- Imbalance (2): direction, magnitude
- Price lags (8): 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
- Rolling mean (3): 6h/12h/24h mean of XBID
- Rolling std (3): 6h/12h/24h std of XBID
- Weather (3): wind, temperature, cloud cover
- Commodities (1): TTF gas
- System deviation (1): trend (3h slope)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from PredictFuturePrices.config import XBID_FORECASTER
from PredictFuturePrices.forecasters.base_forecaster import BaseForecaster

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class XBIDForecaster(BaseForecaster):
    """I predict XBID continuous market mid-prices at multiple horizons."""

    FEATURE_NAMES = [
        # Time (7)
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "hour_frac",
        # RES (3)
        "solar_norm", "wind_norm", "res_total_norm",
        # ISP1 (4)
        "isp1_up", "isp1_down", "isp1_spread", "isp1_up_lag",
        # DAM (2)
        "dam_price_norm", "dam_vs_24h_mean",
        # Imbalance (2)
        "imbalance_dir", "imbalance_mag",
        # Price lags (8)
        "lag_1h", "lag_2h", "lag_3h", "lag_6h",
        "lag_12h", "lag_24h", "lag_48h", "lag_168h",
        # Rolling stats (3) — 6h/12h/24h mean
        "xbid_mean_6h", "xbid_mean_12h", "xbid_mean_24h",
        # Rolling std (3) — 6h/12h/24h std
        "xbid_std_6h", "xbid_std_12h", "xbid_std_24h",
        # Weather (3)
        "wind_wx", "temperature_wx", "cloud_cover_wx",
        # Commodity (1)
        "ttf_gas_norm",
        # System deviation trend (1)
        "deviation_trend",
    ]

    def __init__(self):
        super().__init__(name="xbid", n_features=XBID_FORECASTER.n_features)
        self.feature_names = self.FEATURE_NAMES
        self.horizons = XBID_FORECASTER.horizons_hours
        self.sph = 4  # I use 15-min resolution (steps per hour)

    def _get_xbid_col(self, df: pd.DataFrame) -> str:
        """I find the best XBID price column."""
        for col in ["xbid_price_bid", "xbid_price_mid", "isp1_price_up"]:
            if col in df.columns:
                return col
        # I fall back to DAM price as proxy
        return "dam_price_gr" if "dam_price_gr" in df.columns else "price"

    def build_features(
        self, df: pd.DataFrame, idx: int, **kwargs
    ) -> Optional[np.ndarray]:
        """I build a 37-feature vector for XBID prediction."""
        min_history = 7 * 24 * self.sph
        if idx < min_history:
            return None

        row = df.iloc[idx]
        features = np.zeros(self.n_features)
        fi = 0

        ts = df.index[idx]

        # I extract time features (7)
        features[fi] = np.sin(2 * np.pi * ts.hour / 24)
        features[fi + 1] = np.cos(2 * np.pi * ts.hour / 24)
        features[fi + 2] = np.sin(2 * np.pi * ts.dayofweek / 7)
        features[fi + 3] = np.cos(2 * np.pi * ts.dayofweek / 7)
        features[fi + 4] = np.sin(2 * np.pi * ts.month / 12)
        features[fi + 5] = np.cos(2 * np.pi * ts.month / 12)
        features[fi + 6] = ts.hour / 24.0
        fi += 7

        # I extract RES features (3)
        for col, norm in [
            ("solar_radiation", 1000.0), ("wind_100m", 25.0), ("res_forecast_mw", 5000.0)
        ]:
            alt = col.replace("_mw", "")
            actual_col = col if col in df.columns else (alt if alt in df.columns else None)
            if actual_col:
                features[fi] = row.get(actual_col, 0) / norm
            fi += 1

        # I extract ISP1 signal (4)
        if "isp1_price_up" in df.columns:
            features[fi] = row.get("isp1_price_up", 0) / 200.0
        fi += 1
        if "isp1_price_down" in df.columns:
            features[fi] = row.get("isp1_price_down", 0) / 200.0
        fi += 1
        if "isp1_price_up" in df.columns and "isp1_price_down" in df.columns:
            features[fi] = (row.get("isp1_price_up", 0) - row.get("isp1_price_down", 0)) / 200.0
        fi += 1
        if "isp1_price_up" in df.columns and idx >= 1:
            features[fi] = df["isp1_price_up"].iloc[idx - 1] / 200.0
        fi += 1

        # I extract DAM features (2)
        dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        window_24h = 24 * self.sph
        if dam_col in df.columns:
            dam_val = row.get(dam_col, 0)
            features[fi] = dam_val / 200.0
            dam_mean = df[dam_col].iloc[max(0, idx - window_24h):idx].mean()
            features[fi + 1] = (dam_val - dam_mean) / 100.0 if not np.isnan(dam_mean) else 0
        fi += 2

        # I extract imbalance features (2)
        if "net_imbalance_mw" in df.columns:
            imb = row.get("net_imbalance_mw", 0)
            features[fi] = np.sign(imb)
            features[fi + 1] = abs(imb) / 500.0
        fi += 2

        # I extract price lags (8)
        xbid_col = self._get_xbid_col(df)
        for lag_hours in [1, 2, 3, 6, 12, 24, 48, 168]:
            lag_idx = idx - lag_hours * self.sph
            if lag_idx >= 0:
                features[fi] = df[xbid_col].iloc[lag_idx] / 200.0
            fi += 1

        # I extract rolling mean stats (3)
        for window_hours in [6, 12, 24]:
            w_size = window_hours * self.sph
            w = df[xbid_col].iloc[max(0, idx - w_size):idx]
            if len(w) > 0:
                features[fi] = w.mean() / 200.0
            fi += 1

        # I extract rolling std stats (3)
        for window_hours in [6, 12, 24]:
            w_size = window_hours * self.sph
            w = df[xbid_col].iloc[max(0, idx - w_size):idx]
            if len(w) > 1:
                features[fi] = w.std() / 100.0
            fi += 1

        # I extract weather (3)
        for col, norm in [("wind_100m", 25.0), ("temperature", 40.0), ("cloud_cover", 100.0)]:
            if col in df.columns:
                features[fi] = row.get(col, 0) / norm
            fi += 1

        # I extract TTF gas (1)
        if "ttf_gas" in df.columns:
            features[fi] = row.get("ttf_gas", 0) / 50.0
        fi += 1

        # I extract system deviation trend (1) — 3h slope
        dev_lag = 3 * self.sph
        if "system_deviation_mwh" in df.columns and idx >= dev_lag:
            devs = df["system_deviation_mwh"].iloc[idx - dev_lag:idx + 1].values
            if len(devs) > 1 and not np.any(np.isnan(devs)):
                features[fi] = (devs[-1] - devs[0]) / 500.0
        fi += 1

        return features

    def build_features_matrix(
        self, df: pd.DataFrame, start_idx: int, end_idx: int, **kwargs
    ) -> Optional[tuple]:
        """I build all XBID features vectorized for speed."""
        horizon = kwargs.get("horizon", 1)
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

        # I compute time features (7)
        ts_index = df.index[start_idx:end_idx]
        hours = ts_index.hour.values
        X[:, fi] = np.sin(2 * np.pi * hours / 24)
        X[:, fi + 1] = np.cos(2 * np.pi * hours / 24)
        X[:, fi + 2] = np.sin(2 * np.pi * ts_index.dayofweek.values / 7)
        X[:, fi + 3] = np.cos(2 * np.pi * ts_index.dayofweek.values / 7)
        X[:, fi + 4] = np.sin(2 * np.pi * ts_index.month.values / 12)
        X[:, fi + 5] = np.cos(2 * np.pi * ts_index.month.values / 12)
        X[:, fi + 6] = hours / 24.0
        fi += 7

        # I compute RES features (3)
        for col, norm_val in [
            ("solar_radiation", 1000.0), ("wind_100m", 25.0), ("res_forecast_mw", 5000.0)
        ]:
            alt = col.replace("_mw", "")
            actual_col = col if col in df.columns else (alt if alt in df.columns else None)
            if actual_col:
                X[:, fi] = np.nan_to_num(df[actual_col].values[sl], nan=0.0) / norm_val
            fi += 1

        # I compute ISP1 signal (4)
        if "isp1_price_up" in df.columns:
            X[:, fi] = np.nan_to_num(df["isp1_price_up"].values[sl], nan=0.0) / 200.0
        fi += 1
        if "isp1_price_down" in df.columns:
            X[:, fi] = np.nan_to_num(df["isp1_price_down"].values[sl], nan=0.0) / 200.0
        fi += 1
        if "isp1_price_up" in df.columns and "isp1_price_down" in df.columns:
            up = df["isp1_price_up"].values[sl]
            dn = df["isp1_price_down"].values[sl]
            X[:, fi] = np.nan_to_num((up - dn) / 200.0, nan=0.0)
        fi += 1
        if "isp1_price_up" in df.columns:
            X[:, fi] = np.nan_to_num(df["isp1_price_up"].shift(1).values[sl], nan=0.0) / 200.0
        fi += 1

        # I compute DAM features (2)
        dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        window_24h = 24 * self.sph
        if dam_col in df.columns:
            dam_v = df[dam_col].values[sl]
            X[:, fi] = np.nan_to_num(dam_v, nan=0.0) / 200.0
            dam_mean24 = df[dam_col].rolling(window_24h, min_periods=1).mean().shift(1).bfill().values
            X[:, fi + 1] = np.nan_to_num((dam_v - dam_mean24[sl]) / 100.0, nan=0.0)
        fi += 2

        # I compute imbalance (2)
        if "net_imbalance_mw" in df.columns:
            imb = df["net_imbalance_mw"].values[sl]
            X[:, fi] = np.sign(np.nan_to_num(imb, nan=0.0))
            X[:, fi + 1] = np.abs(np.nan_to_num(imb, nan=0.0)) / 500.0
        fi += 2

        # I compute price lags (8)
        xbid_col = self._get_xbid_col(df)
        for lag_hours in [1, 2, 3, 6, 12, 24, 48, 168]:
            shifted = df[xbid_col].shift(lag_hours * self.sph).values
            X[:, fi] = np.nan_to_num(shifted[sl], nan=0.0) / 200.0
            fi += 1

        # I compute rolling mean stats (3)
        xbid_series = df[xbid_col]
        for window_hours in [6, 12, 24]:
            w_size = window_hours * self.sph
            rm = xbid_series.rolling(w_size, min_periods=1).mean().shift(1).bfill().values
            X[:, fi] = rm[sl] / 200.0
            fi += 1

        # I compute rolling std stats (3)
        for window_hours in [6, 12, 24]:
            w_size = window_hours * self.sph
            rs = xbid_series.rolling(w_size, min_periods=1).std().shift(1).fillna(0).values
            X[:, fi] = rs[sl] / 100.0
            fi += 1

        # I compute weather (3)
        for col, norm_val in [("wind_100m", 25.0), ("temperature", 40.0), ("cloud_cover", 100.0)]:
            if col in df.columns:
                X[:, fi] = np.nan_to_num(df[col].values[sl], nan=0.0) / norm_val
            fi += 1

        # I compute TTF gas (1)
        if "ttf_gas" in df.columns:
            X[:, fi] = np.nan_to_num(df["ttf_gas"].values[sl], nan=0.0) / 50.0
        fi += 1

        # I compute system deviation trend (1) — 3h slope
        dev_lag = 3 * self.sph
        if "system_deviation_mwh" in df.columns:
            dev = df["system_deviation_mwh"].values
            dev_current = dev[sl]
            dev_3h_ago = df["system_deviation_mwh"].shift(dev_lag).values[sl]
            X[:, fi] = np.nan_to_num((dev_current - np.nan_to_num(dev_3h_ago, nan=dev_current)) / 500.0, nan=0.0)
        fi += 1

        # I compute targets — XBID at idx + horizon (in steps, not hours)
        horizon_steps = horizon * self.sph
        y = np.full(n, np.nan)
        xbid_vals = df[xbid_col].values
        for i in range(n):
            t_idx = start_idx + i + horizon_steps
            if t_idx < len(df):
                y[i] = xbid_vals[t_idx]

        valid &= ~np.any(np.isnan(X), axis=1)
        valid &= ~np.isnan(y)

        return X, y, valid

    def _extract_target(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[float]:
        """I extract the XBID price at the target horizon."""
        horizon = kwargs.get("horizon", 1)
        target_idx = idx + horizon * self.sph

        xbid_col = self._get_xbid_col(df)
        if target_idx >= len(df):
            return None
        return float(df[xbid_col].iloc[target_idx])

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
    ) -> "XBIDForecaster":
        """I train 4 LightGBM models (one per horizon).

        I auto-detect data resolution so lags match the data.
        """
        if lgb is None:
            raise ImportError("LightGBM is required.")

        # I detect resolution from actual data, overriding config
        self.sph = self._detect_resolution(df)

        params = dict(XBID_FORECASTER.lgbm_params)

        min_history = 7 * 24 * self.sph
        for horizon in self.horizons:
            X_train, y_train = self.build_dataset(
                df, min_history, train_end_idx, horizon=horizon
            )
            X_val, y_val = self.build_dataset(
                df, train_end_idx, val_end_idx, horizon=horizon
            )

            if len(X_train) < 50:
                logger.warning(f"[XBID] Horizon {horizon}h: too few samples ({len(X_train)})")
                continue

            model = lgb.LGBMRegressor(**params)
            callbacks = []
            eval_set = None
            if len(X_val) > 0:
                callbacks = [lgb.early_stopping(50, verbose=False)]
                eval_set = [(X_val, y_val)]

            n_samples = len(X_train)
            model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
            self.models[f"h{horizon}"] = model

            # I free the datasets immediately after training
            del X_train, y_train, X_val, y_val

            logger.info(f"[XBID] Horizon {horizon}h trained ({n_samples} samples)")

        self.is_fitted = True

        if val_end_idx > train_end_idx:
            metrics = self.evaluate(df, train_end_idx, val_end_idx, horizon=1)
            self.train_metrics = metrics
            logger.info(f"[XBID] Val MAE={metrics['mae']:.2f}")

        return self

    def predict(self, df: pd.DataFrame, idx: int, **kwargs) -> Dict[str, np.ndarray]:
        """I predict XBID prices at all horizons.

        Returns:
            Dict with keys: "1h", "2h", "4h", "8h" → single price values.
        """
        if not self.is_fitted:
            raise RuntimeError("[XBID] Not fitted yet.")

        result = {}
        features = self.build_features(df, idx)
        if features is None:
            return {f"{h}h": np.array([0.0]) for h in self.horizons}

        X = features.reshape(1, -1)
        for horizon in self.horizons:
            model_key = f"h{horizon}"
            if model_key in self.models:
                result[f"{horizon}h"] = np.array([self.models[model_key].predict(X)[0]])
            else:
                result[f"{horizon}h"] = np.array([0.0])

        return result
