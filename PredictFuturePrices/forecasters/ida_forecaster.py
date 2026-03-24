"""IDA (Intraday Auction) forecaster — predicts clearing prices for IDA1/2/3.

I train 3 LightGBM models (one per auction) to predict the IDA clearing
price relative to DAM. Feature engineering follows the existing
IDASubForecaster in models/market_forecaster.py.

IDA auctions:
- IDA1: gate D-1 15:00, delivery hours 0-23
- IDA2: gate D-1 22:00, delivery hours 0-23
- IDA3: gate D+0 10:00, delivery hours 12-23

Feature set (30 features per auction):
- DAM stats (5): current, 24h mean/std/max/min
- ISP1 signal (3): up price, down price, bid-ask spread
- Imbalance (3): direction, magnitude, spread vs DAM
- aFRR/mFRR (2): aFRR capacity stress, mFRR energy spread
- Historical lags (4): same hour D-1, D-2, D-3, D-7
- Calendar (6): hour sin/cos, DoW sin/cos, month sin/cos
- Serbia spread (1): RS-GR DAM spread
- IDA number (1): 1/2/3 encoded
- Weather revision (3): wind, solar, load forecast revisions
- Commodities (2): TTF gas, DAM forecast error
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from PredictFuturePrices.config import IDA_FORECASTER
from PredictFuturePrices.forecasters.base_forecaster import BaseForecaster

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


# I define IDA gate configurations (same as existing IDASubForecaster)
IDA_CONFIG = {
    1: {"gate_hour": 15, "delivery_hours": list(range(24)), "name": "IDA1"},
    2: {"gate_hour": 22, "delivery_hours": list(range(24)), "name": "IDA2"},
    3: {"gate_hour": 10, "delivery_hours": list(range(12, 24)), "name": "IDA3"},
}


class IDAForecaster(BaseForecaster):
    """I predict IDA clearing prices for all 3 auctions."""

    FEATURE_NAMES = [
        # DAM stats (5)
        "dam_current", "dam_mean_24h", "dam_std_24h", "dam_max_24h", "dam_min_24h",
        # ISP1 signal (3)
        "isp1_price_up", "isp1_price_down", "isp1_bid_ask_spread",
        # Imbalance (3)
        "imbalance_direction", "imbalance_magnitude", "imbalance_vs_dam",
        # Balancing capacity (2)
        "afrr_stress", "mfrr_spread",
        # Lags (4)
        "ida_lag_1d", "ida_lag_2d", "ida_lag_3d", "ida_lag_7d",
        # Calendar (6)
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Cross-border (1)
        "rs_gr_spread",
        # IDA number (1)
        "ida_number_enc",
        # Revisions (3)
        "wind_revision", "solar_revision", "load_revision",
        # Commodities (2)
        "ttf_gas", "dam_forecast_error",
    ]

    def __init__(self):
        super().__init__(name="ida", n_features=IDA_FORECASTER.n_features)
        self.feature_names = self.FEATURE_NAMES
        self.sph = 4  # I use 15-min resolution (steps per hour)

    def build_features(
        self, df: pd.DataFrame, idx: int, **kwargs
    ) -> Optional[np.ndarray]:
        """I build a 30-feature vector for IDA prediction."""
        ida_number = kwargs.get("ida_number", 1)

        min_history = 7 * 24 * self.sph
        if idx < min_history:
            return None

        row = df.iloc[idx]
        features = np.zeros(self.n_features)
        fi = 0
        window_24h = 24 * self.sph

        # I extract DAM stats (5 features)
        dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        if dam_col in df.columns:
            features[fi] = row.get(dam_col, 0) / 200.0
            window = df[dam_col].iloc[max(0, idx - window_24h):idx]
            if len(window) > 0:
                features[fi + 1] = window.mean() / 200.0
                features[fi + 2] = window.std() / 100.0 if len(window) > 1 else 0
                features[fi + 3] = window.max() / 200.0
                features[fi + 4] = window.min() / 200.0
        fi += 5

        # I extract ISP1 signal (3 features)
        for i, col in enumerate(["isp1_price_up", "isp1_price_down"]):
            if col in df.columns:
                features[fi + i] = row.get(col, 0) / 200.0
        if "isp1_price_up" in df.columns and "isp1_price_down" in df.columns:
            features[fi + 2] = (row.get("isp1_price_up", 0) - row.get("isp1_price_down", 0)) / 200.0
        fi += 3

        # I extract imbalance (3 features)
        if "net_imbalance_mw" in df.columns:
            imb = row.get("net_imbalance_mw", 0)
            features[fi] = np.sign(imb)
            features[fi + 1] = abs(imb) / 500.0
        if "imbalance_price" in df.columns and dam_col in df.columns:
            features[fi + 2] = (row.get("imbalance_price", 0) - row.get(dam_col, 0)) / 100.0
        fi += 3

        # I extract balancing capacity (2 features)
        if "afrr_price_up" in df.columns and "afrr_price_down" in df.columns:
            features[fi] = (row.get("afrr_price_up", 0) + row.get("afrr_price_down", 0)) / 200.0
        if "mfrr_spread" in df.columns:
            features[fi + 1] = row.get("mfrr_spread", 0) / 200.0
        elif "mfrr_price_up" in df.columns and "mfrr_price_down" in df.columns:
            features[fi + 1] = (row.get("mfrr_price_up", 0) - row.get("mfrr_price_down", 0)) / 200.0
        fi += 2

        # I extract lags (4 features) — I use DAM as proxy for IDA lags
        for lag_days in [1, 2, 3, 7]:
            lag_idx = idx - lag_days * 24 * self.sph
            if lag_idx >= 0 and dam_col in df.columns:
                features[fi] = df[dam_col].iloc[lag_idx] / 200.0
            fi += 1

        # I extract calendar (6 features)
        ts = df.index[idx]
        features[fi] = np.sin(2 * np.pi * ts.hour / 24)
        features[fi + 1] = np.cos(2 * np.pi * ts.hour / 24)
        features[fi + 2] = np.sin(2 * np.pi * ts.dayofweek / 7)
        features[fi + 3] = np.cos(2 * np.pi * ts.dayofweek / 7)
        features[fi + 4] = np.sin(2 * np.pi * ts.month / 12)
        features[fi + 5] = np.cos(2 * np.pi * ts.month / 12)
        fi += 6

        # I extract RS-GR spread (1 feature)
        if "rs_gr_spread" in df.columns:
            features[fi] = row.get("rs_gr_spread", 0) / 50.0
        fi += 1

        # I encode the IDA number (1 feature)
        features[fi] = ida_number / 3.0
        fi += 1

        # I extract forecast revisions (3 features)
        # I compute as difference between current and 24h-ago forecast
        for col, norm in [
            ("wind_100m", 25.0), ("solar_radiation", 1000.0), ("load_forecast", 10000.0)
        ]:
            if col in df.columns and idx >= window_24h:
                features[fi] = (row.get(col, 0) - df[col].iloc[idx - window_24h]) / norm
            fi += 1

        # I extract commodities + DAM error (2 features)
        if "ttf_gas" in df.columns:
            features[fi] = row.get("ttf_gas", 0) / 50.0
        fi += 1
        if "load_forecast_error_lag" in df.columns:
            features[fi] = row.get("load_forecast_error_lag", 0) / 1000.0
        fi += 1

        return features

    def build_features_matrix(
        self, df: pd.DataFrame, start_idx: int, end_idx: int, **kwargs
    ) -> Optional[tuple]:
        """I build all IDA features vectorized for speed."""
        ida_number = kwargs.get("ida_number", 1)
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
        window_24h = 24 * self.sph

        # I compute DAM stats (5 features)
        dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"
        if dam_col in df.columns:
            X[:, fi] = np.nan_to_num(df[dam_col].values[sl], nan=0.0) / 200.0
            ds = df[dam_col]
            dm = ds.rolling(window_24h, min_periods=1).mean().shift(1).bfill().values
            dd = ds.rolling(window_24h, min_periods=1).std().shift(1).fillna(0).values
            dx = ds.rolling(window_24h, min_periods=1).max().shift(1).bfill().values
            dn = ds.rolling(window_24h, min_periods=1).min().shift(1).bfill().values
            X[:, fi + 1] = dm[sl] / 200.0
            X[:, fi + 2] = dd[sl] / 100.0
            X[:, fi + 3] = dx[sl] / 200.0
            X[:, fi + 4] = dn[sl] / 200.0
        fi += 5

        # I compute ISP1 signal (3 features)
        if "isp1_price_up" in df.columns:
            X[:, fi] = np.nan_to_num(df["isp1_price_up"].values[sl], nan=0.0) / 200.0
        if "isp1_price_down" in df.columns:
            X[:, fi + 1] = np.nan_to_num(df["isp1_price_down"].values[sl], nan=0.0) / 200.0
        if "isp1_price_up" in df.columns and "isp1_price_down" in df.columns:
            up = df["isp1_price_up"].values[sl]
            down = df["isp1_price_down"].values[sl]
            X[:, fi + 2] = np.nan_to_num((up - down) / 200.0, nan=0.0)
        fi += 3

        # I compute imbalance (3 features)
        if "net_imbalance_mw" in df.columns:
            imb = df["net_imbalance_mw"].values[sl]
            X[:, fi] = np.sign(np.nan_to_num(imb, nan=0.0))
            X[:, fi + 1] = np.abs(np.nan_to_num(imb, nan=0.0)) / 500.0
        if "imbalance_price" in df.columns and dam_col in df.columns:
            imp = df["imbalance_price"].values[sl]
            dmp = df[dam_col].values[sl]
            X[:, fi + 2] = np.nan_to_num((imp - dmp) / 100.0, nan=0.0)
        fi += 3

        # I compute balancing capacity (2 features)
        if "afrr_price_up" in df.columns and "afrr_price_down" in df.columns:
            aup = df["afrr_price_up"].values[sl]
            adn = df["afrr_price_down"].values[sl]
            X[:, fi] = np.nan_to_num((aup + adn) / 200.0, nan=0.0)
        if "mfrr_spread" in df.columns:
            X[:, fi + 1] = np.nan_to_num(df["mfrr_spread"].values[sl], nan=0.0) / 200.0
        elif "mfrr_price_up" in df.columns and "mfrr_price_down" in df.columns:
            mu = df["mfrr_price_up"].values[sl]
            md = df["mfrr_price_down"].values[sl]
            X[:, fi + 1] = np.nan_to_num((mu - md) / 200.0, nan=0.0)
        fi += 2

        # I compute lags (4 features) — DAM as proxy
        if dam_col in df.columns:
            for lag_i, lag_days in enumerate([1, 2, 3, 7]):
                shifted = df[dam_col].shift(lag_days * 24 * self.sph).values
                X[:, fi + lag_i] = np.nan_to_num(shifted[sl], nan=0.0) / 200.0
        fi += 4

        # I compute calendar (6 features)
        ts_index = df.index[start_idx:end_idx]
        X[:, fi] = np.sin(2 * np.pi * ts_index.hour.values / 24)
        X[:, fi + 1] = np.cos(2 * np.pi * ts_index.hour.values / 24)
        X[:, fi + 2] = np.sin(2 * np.pi * ts_index.dayofweek.values / 7)
        X[:, fi + 3] = np.cos(2 * np.pi * ts_index.dayofweek.values / 7)
        X[:, fi + 4] = np.sin(2 * np.pi * ts_index.month.values / 12)
        X[:, fi + 5] = np.cos(2 * np.pi * ts_index.month.values / 12)
        fi += 6

        # I compute RS-GR spread (1 feature)
        if "rs_gr_spread" in df.columns:
            X[:, fi] = np.nan_to_num(df["rs_gr_spread"].values[sl], nan=0.0) / 50.0
        fi += 1

        # I encode the IDA number (1 feature) — constant for entire batch
        X[:, fi] = ida_number / 3.0
        fi += 1

        # I compute forecast revisions (3 features)
        for col, norm_val in [
            ("wind_100m", 25.0), ("solar_radiation", 1000.0), ("load_forecast", 10000.0)
        ]:
            if col in df.columns:
                current = df[col].values[sl]
                lagged = df[col].shift(window_24h).values[sl]
                X[:, fi] = np.nan_to_num((current - np.nan_to_num(lagged, nan=current)) / norm_val, nan=0.0)
            fi += 1

        # I compute commodities + DAM error (2 features)
        if "ttf_gas" in df.columns:
            X[:, fi] = np.nan_to_num(df["ttf_gas"].values[sl], nan=0.0) / 50.0
        fi += 1
        if "load_forecast_error_lag" in df.columns:
            X[:, fi] = np.nan_to_num(df["load_forecast_error_lag"].values[sl], nan=0.0) / 1000.0
        fi += 1

        # I compute targets — blended DAM + ISP1
        alpha = IDA_FORECASTER.blending_alphas.get(ida_number, 0.7)
        y = np.full(n, np.nan)
        if dam_col in df.columns:
            dam_vals = df[dam_col].values
            isp1_up = df["isp1_price_up"].values if "isp1_price_up" in df.columns else None
            isp1_dn = df["isp1_price_down"].values if "isp1_price_down" in df.columns else None

            for i in range(n):
                t_idx = start_idx + i + 1
                if t_idx < len(df):
                    dp = dam_vals[t_idx]
                    isp1_mid = dp
                    if isp1_up is not None and isp1_dn is not None:
                        up_v, dn_v = isp1_up[t_idx], isp1_dn[t_idx]
                        if not (np.isnan(up_v) or np.isnan(dn_v)):
                            isp1_mid = (up_v + dn_v) / 2
                    y[i] = alpha * dp + (1 - alpha) * isp1_mid

        valid &= ~np.any(np.isnan(X), axis=1)
        valid &= ~np.isnan(y)

        return X, y, valid

    def _extract_target(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[float]:
        """I extract the target IDA price.

        Since real IDA prices may not be in the data, I use a synthetic
        target: blended DAM + ISP1 (same pattern as IDASubForecaster).
        """
        ida_number = kwargs.get("ida_number", 1)
        dam_col = "dam_price_gr" if "dam_price_gr" in df.columns else "price"

        if idx + 1 >= len(df) or dam_col not in df.columns:
            return None

        dam_price = df[dam_col].iloc[idx + 1]
        alpha = IDA_FORECASTER.blending_alphas.get(ida_number, 0.7)

        # I blend with ISP1 if available (same as real IDA correlation)
        isp1_mid = dam_price
        if "isp1_price_up" in df.columns and "isp1_price_down" in df.columns:
            up = df["isp1_price_up"].iloc[idx + 1]
            down = df["isp1_price_down"].iloc[idx + 1]
            if not (np.isnan(up) or np.isnan(down)):
                isp1_mid = (up + down) / 2

        return float(alpha * dam_price + (1 - alpha) * isp1_mid)

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
    ) -> "IDAForecaster":
        """I train 3 LightGBM models (one per IDA auction).

        I auto-detect data resolution so lags match the data.
        """
        if lgb is None:
            raise ImportError("LightGBM is required.")

        # I detect resolution from actual data, overriding config
        self.sph = self._detect_resolution(df)

        params = dict(IDA_FORECASTER.lgbm_params)

        min_history = 7 * 24 * self.sph
        for ida_num in IDA_FORECASTER.ida_numbers:
            X_train, y_train = self.build_dataset(
                df, min_history, train_end_idx, ida_number=ida_num
            )
            X_val, y_val = self.build_dataset(
                df, train_end_idx, val_end_idx, ida_number=ida_num
            )

            if len(X_train) < 50:
                logger.warning(f"[IDA{ida_num}] Too few samples ({len(X_train)})")
                continue

            model = lgb.LGBMRegressor(**params)
            callbacks = []
            eval_set = None
            if len(X_val) > 0:
                callbacks = [lgb.early_stopping(50, verbose=False)]
                eval_set = [(X_val, y_val)]

            n_samples = len(X_train)
            model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
            self.models[f"ida{ida_num}"] = model

            # I free the datasets immediately after training
            del X_train, y_train, X_val, y_val

            logger.info(f"[IDA{ida_num}] Trained ({n_samples} samples)")

        self.is_fitted = True

        if val_end_idx > train_end_idx:
            metrics = self.evaluate(df, train_end_idx, val_end_idx, ida_number=1)
            self.train_metrics = metrics
            logger.info(f"[IDA] Val MAE={metrics['mae']:.2f}")

        return self

    def predict(self, df: pd.DataFrame, idx: int, **kwargs) -> Dict[str, np.ndarray]:
        """I predict IDA clearing prices for all 3 auctions.

        Returns:
            Dict with keys: "ida1", "ida2", "ida3" → single price values.
        """
        if not self.is_fitted:
            raise RuntimeError("[IDA] Not fitted yet.")

        result = {}
        for ida_num in IDA_FORECASTER.ida_numbers:
            features = self.build_features(df, idx, ida_number=ida_num)
            if features is None:
                result[f"ida{ida_num}"] = np.array([0.0])
                continue

            model_key = f"ida{ida_num}"
            if model_key in self.models:
                pred = self.models[model_key].predict(features.reshape(1, -1))[0]
                result[f"ida{ida_num}"] = np.array([pred])
            else:
                result[f"ida{ida_num}"] = np.array([0.0])

        return result
