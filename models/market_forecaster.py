"""
Unified Market Forecaster

I provide price forecasts for DAM, IntraDay, and Imbalance markets.
This module contains three sub-forecasters that can be trained independently
and used together to enhance the RL agent's observation space.

Architecture:
- DAMForecaster: 24h ahead, hourly — uses Serbia DAM as leading indicator
- IntraDayForecaster: 4 horizons (1h, 2h, 4h, 8h) — LightGBM on real ISP1 data
- ImbalanceForecaster: 4h ahead — predicts imbalance direction and magnitude

All forecasters follow the same API:
    .fit(X_train, y_train, X_val, y_val) -> self
    .predict(features) -> np.ndarray
    .save(path) / .load(path)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pickle
import warnings

warnings.filterwarnings('ignore')

# I use the hour-specific regression params from the existing DAM forecaster
# as a baseline before the LightGBM model is trained
HOURLY_GR_RS_PARAMS = {
    0: {'slope': 0.8635, 'intercept': 11.24},
    1: {'slope': 0.6393, 'intercept': 20.51},
    2: {'slope': 0.7116, 'intercept': 16.77},
    3: {'slope': 0.7785, 'intercept': 13.21},
    4: {'slope': 0.7523, 'intercept': 15.93},
    5: {'slope': 0.3039, 'intercept': 42.04},
    6: {'slope': 0.2107, 'intercept': 49.07},
    7: {'slope': 0.2952, 'intercept': 44.65},
    8: {'slope': 0.9399, 'intercept': 8.45},
    9: {'slope': 0.8768, 'intercept': 5.98},
    10: {'slope': 0.8409, 'intercept': 2.97},
    11: {'slope': 0.5596, 'intercept': 9.10},
    12: {'slope': 0.5505, 'intercept': 8.28},
    13: {'slope': 0.5192, 'intercept': 12.91},
    14: {'slope': 0.5612, 'intercept': 13.37},
    15: {'slope': 0.3808, 'intercept': 32.02},
    16: {'slope': 0.4458, 'intercept': 32.16},
    17: {'slope': 0.4838, 'intercept': 38.16},
    18: {'slope': 0.4360, 'intercept': 47.74},
    19: {'slope': 0.4741, 'intercept': 49.33},
    20: {'slope': 1.2802, 'intercept': -5.12},
    21: {'slope': 1.3832, 'intercept': -20.05},
    22: {'slope': 0.9260, 'intercept': 0.41},
    23: {'slope': 0.7632, 'intercept': 12.73},
}


class DAMSubForecaster:
    """
    I forecast DAM prices 24h ahead (hourly resolution).

    Primary signal: Serbia DAM D+1 prices (available before GR gate closure).
    Enhanced with LightGBM trained on:
    - Serbia DAM prices (per-hour)
    - Historical Greek DAM (7-day lookback)
    - Load forecast
    - RES forecast (solar + wind)
    - Calendar features (hour, DoW, month)
    """

    def __init__(self):
        self.models = {}  # I store one model per hour (0-23)
        self.is_trained = False
        self.feature_names = None

    def build_features(self, df: pd.DataFrame, idx: int, hour: int) -> np.ndarray:
        """I build feature vector for a single hour prediction."""
        features = []

        # I use backward-looking data only (available at forecast time = D-1 12:00)
        # Serbia DAM D+1 price for this hour
        serbia_col = 'serbia_dam_price'
        if serbia_col in df.columns:
            features.append(df.iloc[idx].get(serbia_col, 80.0))
        else:
            features.append(80.0)

        # I add historical Greek DAM features (7-day lookback, same hour)
        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1
        for lag_days in [1, 2, 3, 7]:
            lag_idx = idx - lag_days * 24 * sph
            if lag_idx >= 0:
                features.append(df.iloc[lag_idx].get('price', 80.0))
            else:
                features.append(80.0)

        # I add 24h rolling stats of Greek DAM
        w24 = 24 * sph
        start = max(0, idx - w24)
        recent_prices = df.iloc[start:idx]['price'].values
        if len(recent_prices) > 0:
            features.append(np.mean(recent_prices))
            features.append(np.std(recent_prices) if len(recent_prices) > 1 else 10.0)
            features.append(np.min(recent_prices))
            features.append(np.max(recent_prices))
        else:
            features.extend([80.0, 10.0, 60.0, 120.0])

        # I add load and RES forecasts
        features.append(df.iloc[idx].get('load_mw', 5000.0) / 10000.0)
        features.append(df.iloc[idx].get('res_total_mw', 2000.0) / 5000.0)
        features.append(df.iloc[idx].get('solar', 0.0) / 3000.0)
        features.append(df.iloc[idx].get('wind_onshore', 0.0) / 3000.0)

        # I add calendar features
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        dow = df.index[idx].dayofweek
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))
        month = df.index[idx].month
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))

        return np.array(features, dtype=np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int
    ) -> 'DAMSubForecaster':
        """I train 24 per-hour LightGBM models."""
        try:
            import lightgbm as lgb
        except ImportError:
            print("  WARNING: LightGBM not available, DAM forecaster will use regression fallback")
            return self

        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        print("  Training DAM sub-forecaster (24 hourly models)...")
        for hour in range(24):
            # I collect training samples for this hour
            X_train, y_train = [], []
            X_val, y_val = [], []

            for idx in range(48 * sph, len(df) - 24 * sph):
                ts = df.index[idx]
                if ts.hour != hour or (ts.minute != 0 and sph > 1):
                    continue

                # I predict 24h ahead
                target_idx = idx + 24 * sph
                if target_idx >= len(df):
                    continue

                features = self.build_features(df, idx, hour)
                target = df.iloc[target_idx]['price']

                if idx < train_end_idx:
                    X_train.append(features)
                    y_train.append(target)
                elif idx < val_end_idx:
                    X_val.append(features)
                    y_val.append(target)

            if len(X_train) < 50:
                continue

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            params = {
                'objective': 'regression',
                'metric': 'mae',
                'max_depth': 6,
                'n_estimators': 500,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
            }

            model = lgb.LGBMRegressor(**params)

            if len(X_val) > 10:
                X_v = np.array(X_val)
                y_v = np.array(y_val)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_v, y_v)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                model.fit(X_train, y_train)

            self.models[hour] = model

        self.is_trained = len(self.models) > 0
        print(f"  DAM: trained {len(self.models)}/24 hourly models")
        return self

    def predict(self, df: pd.DataFrame, idx: int, n_hours: int = 24) -> np.ndarray:
        """I predict DAM prices for the next n_hours."""
        predictions = np.zeros(n_hours)
        current_hour = df.index[idx].hour

        for h in range(n_hours):
            target_hour = (current_hour + h + 1) % 24

            if self.is_trained and target_hour in self.models:
                features = self.build_features(df, idx, target_hour).reshape(1, -1)
                predictions[h] = self.models[target_hour].predict(features)[0]
            else:
                # I use Serbia regression fallback
                serbia_price = df.iloc[idx].get('serbia_dam_price', None)
                if serbia_price is not None and not np.isnan(serbia_price):
                    params = HOURLY_GR_RS_PARAMS.get(target_hour, {'slope': 0.7, 'intercept': 20.0})
                    predictions[h] = params['slope'] * serbia_price + params['intercept']
                else:
                    # I use persistence (last known price)
                    predictions[h] = df.iloc[idx].get('price', 80.0)

        return predictions

    def predict_with_confidence(self, df: pd.DataFrame, idx: int, n_hours: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """I predict DAM prices with confidence (std) estimate."""
        predictions = self.predict(df, idx, n_hours)

        # I estimate confidence as increasing with horizon
        base_std = df.iloc[max(0, idx - 96):idx]['price'].std() if idx > 96 else 15.0
        if np.isnan(base_std) or base_std < 1.0:
            base_std = 15.0

        stds = np.array([base_std * (1.0 + 0.1 * h) for h in range(n_hours)])
        return predictions, stds


class IntraDaySubForecaster:
    """
    I forecast IntraDay clearing prices at 4 horizons: 1h, 2h, 4h, 8h.

    I extend the existing IntraDayForecaster with real ISP1 training data
    and additional features (Serbia DAM spread, DAM forecast deviation).
    """

    HORIZONS = [1, 2, 4, 8]  # I predict at these hour horizons

    def __init__(self):
        self.models = {}  # I store one model per horizon
        self.is_trained = False

    def build_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """I build 45-feature vector for IntraDay prediction."""
        row = df.iloc[idx]
        ts = df.index[idx]
        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        features = []

        # I add time features (7)
        hour = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
        month = ts.month
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))
        features.append(hour / 24.0)

        # I add RES features (3)
        features.append(row.get('solar', 0.0) / 3000.0)
        features.append(row.get('wind_onshore', 0.0) / 3000.0)
        features.append(row.get('res_total_mw', 0.0) / 5000.0)

        # I add balancing market features (6)
        features.append(row.get('afrr_up', 80.0) / 200.0)
        features.append(row.get('afrr_down', 24.0) / 200.0)
        features.append(row.get('mfrr_price_up', 120.0) / 500.0)
        features.append(row.get('mfrr_price_down', 60.0) / 500.0)
        features.append(row.get('mfrr_spread', 60.0) / 500.0)
        features.append(row.get('net_imbalance_mw', 0.0) / 1000.0)

        # I add current DAM price (1)
        dam_price = row.get('price', 80.0)
        features.append(dam_price / 200.0)

        # I add price lags (8)
        lag_steps = [1 * sph, 2 * sph, 3 * sph, 6 * sph,
                     12 * sph, 24 * sph, 48 * sph, 168 * sph]
        for lag in lag_steps:
            lag_idx = idx - lag
            if lag_idx >= 0:
                features.append(df.iloc[lag_idx].get('price', dam_price) / 200.0)
            else:
                features.append(dam_price / 200.0)

        # I add rolling statistics (6)
        for window in [6, 12, 24]:
            w = window * sph
            start = max(0, idx - w)
            prices = df.iloc[start:idx]['price'].values
            if len(prices) > 0:
                features.append(np.mean(prices) / 200.0)
                features.append(np.std(prices) / 100.0 if len(prices) > 1 else 0.1)
            else:
                features.append(dam_price / 200.0)
                features.append(0.1)

        # I add momentum features (2)
        for lag in [1 * sph, 3 * sph]:
            lag_idx = idx - lag
            if lag_idx >= 0:
                old_price = df.iloc[lag_idx].get('price', dam_price)
                features.append((dam_price - old_price) / max(abs(old_price), 1.0))
            else:
                features.append(0.0)

        # I add relative price features (2)
        mean_24h = row.get('price_mean_24h', dam_price)
        features.append((dam_price - mean_24h) / max(abs(mean_24h), 1.0))
        features.append(0.5)  # I use placeholder for percentile (computed in training)

        # I add ISP1-specific features (4) - real IntraDay price lags
        for col in ['isp1_price_up', 'isp1_price_down']:
            if col in df.columns:
                features.append(row.get(col, dam_price) / 200.0)
                lag_idx = idx - 1 * sph
                if lag_idx >= 0 and col in df.columns:
                    features.append(df.iloc[lag_idx].get(col, dam_price) / 200.0)
                else:
                    features.append(dam_price / 200.0)
            else:
                features.append(dam_price / 200.0)
                features.append(dam_price / 200.0)

        # I add Serbia DAM spread (1)
        serbia = row.get('serbia_dam_price', np.nan)
        if not np.isnan(serbia):
            features.append((serbia - dam_price) / 100.0)
        else:
            features.append(0.0)

        # I add DAM forecast deviation (1) - how wrong was the forecast?
        dam_fc = row.get('dam_forecast_error', 0.0)
        features.append(np.clip(dam_fc / 50.0, -1.0, 1.0))

        # I add system load (1)
        features.append(row.get('load_mw', 5000.0) / 10000.0)

        return np.array(features, dtype=np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
        target_col: str = 'isp1_price_up'
    ) -> 'IntraDaySubForecaster':
        """I train LightGBM models for each horizon."""
        try:
            import lightgbm as lgb
        except ImportError:
            print("  WARNING: LightGBM not available for IntraDay forecaster")
            return self

        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        # I check if target column exists
        if target_col not in df.columns:
            print(f"  WARNING: Target column '{target_col}' not found, using 'price'")
            target_col = 'price'

        print(f"  Training IntraDay sub-forecaster ({len(self.HORIZONS)} horizons)...")

        for horizon_h in self.HORIZONS:
            X_train, y_train = [], []
            X_val, y_val = [], []

            horizon_steps = horizon_h * sph

            for idx in range(168 * sph, len(df) - horizon_steps):
                features = self.build_features(df, idx)
                target_idx = idx + horizon_steps
                target = df.iloc[target_idx][target_col]

                if np.isnan(target):
                    continue

                if idx < train_end_idx:
                    X_train.append(features)
                    y_train.append(target)
                elif idx < val_end_idx:
                    X_val.append(features)
                    y_val.append(target)

            if len(X_train) < 100:
                print(f"    Horizon {horizon_h}h: insufficient training data ({len(X_train)} samples)")
                continue

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # I use deeper models for longer horizons
            if horizon_h <= 2:
                params = {'max_depth': 6, 'n_estimators': 500, 'learning_rate': 0.05}
            else:
                params = {'max_depth': 7, 'n_estimators': 800, 'learning_rate': 0.03}

            params.update({
                'objective': 'regression',
                'metric': 'mae',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
            })

            model = lgb.LGBMRegressor(**params)

            if len(X_val) > 20:
                X_v = np.array(X_val)
                y_v = np.array(y_val)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_v, y_v)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                val_pred = model.predict(X_v)
                mae = np.mean(np.abs(val_pred - y_v))
                print(f"    Horizon {horizon_h}h: {len(X_train)} train, val MAE={mae:.2f} EUR/MWh")
            else:
                model.fit(X_train, y_train)
                print(f"    Horizon {horizon_h}h: {len(X_train)} train (no val)")

            self.models[horizon_h] = model

        self.is_trained = len(self.models) > 0
        return self

    def predict(self, df: pd.DataFrame, idx: int, add_noise: bool = False) -> Dict[int, float]:
        """I predict IntraDay prices at each horizon."""
        features = self.build_features(df, idx).reshape(1, -1)
        predictions = {}

        for horizon_h in self.HORIZONS:
            if self.is_trained and horizon_h in self.models:
                pred = float(self.models[horizon_h].predict(features)[0])
            else:
                # I use persistence fallback
                pred = df.iloc[idx].get('price', 80.0)

            if add_noise:
                error_pct = 0.08 + 0.04 * (horizon_h - 1)
                pred += np.random.normal(0, abs(pred) * error_pct)

            predictions[horizon_h] = max(0.0, pred)

        return predictions


class IDASubForecaster:
    """
    I forecast IDA clearing prices for each IntraDay Auction (IDA1/IDA2/IDA3).

    IDA clearing prices are correlated with DAM but deviate due to updated
    information (wind/solar forecast updates, demand changes, cross-border flows).
    I predict the IDA-DAM spread rather than absolute prices, since the spread
    is more stationary and easier to learn.

    Architecture: One LightGBM model per IDA auction (3 total).
    Each predicts hourly clearing prices for the delivery window:
      - IDA1: 24 hours (00:00-24:00), gate D-1 15:00
      - IDA2: 24 hours (00:00-24:00), gate D-1 22:00
      - IDA3: 12 hours (12:00-24:00), gate D+0 10:00

    Training target: synthetic IDA clearing prices (OU noise on DAM), since
    real IDA clearing data is not available in sufficient quantity.
    """

    # I define IDA delivery windows and horizons
    IDA_CONFIG = {
        1: {'delivery_hours': list(range(24)), 'gate_hour': 15, 'name': 'IDA1'},
        2: {'delivery_hours': list(range(24)), 'gate_hour': 22, 'name': 'IDA2'},
        3: {'delivery_hours': list(range(12, 24)), 'gate_hour': 10, 'name': 'IDA3'},
    }

    def __init__(self):
        self.models = {}  # I store one model per IDA number {1: model, 2: model, 3: model}
        self.is_trained = False

    def _synthesize_ida_prices(self, dam_prices: np.ndarray, ida_number: int,
                                seed: int = None) -> np.ndarray:
        """
        I generate synthetic IDA clearing prices using Ornstein-Uhlenbeck noise on DAM.

        IDA prices deviate from DAM due to updated forecasts:
        - IDA1 (9-33h before delivery): σ=4 EUR/MWh (closest to DAM, least new info)
        - IDA2 (2-26h before delivery): σ=8 EUR/MWh (more volatile, more forecast updates)
        - IDA3 (2-14h before delivery): σ=12 EUR/MWh (most volatile, last correction)

        I use OU process with mean-reversion to DAM (θ=0.3) so spreads don't drift.
        """
        rng = np.random.RandomState(seed)
        n = len(dam_prices)

        # I set volatility by IDA auction (later = more volatile)
        sigma_map = {1: 4.0, 2: 8.0, 3: 12.0}
        sigma = sigma_map.get(ida_number, 8.0)
        theta = 0.3  # Mean-reversion speed to DAM

        # I simulate OU process for spread
        spread = np.zeros(n)
        dt = 1.0  # hourly
        for i in range(1, n):
            spread[i] = (spread[i-1]
                         - theta * spread[i-1] * dt
                         + sigma * np.sqrt(dt) * rng.normal())

        ida_prices = dam_prices + spread
        # I ensure non-negative prices
        ida_prices = np.maximum(ida_prices, 0.0)
        return ida_prices

    def build_features(self, df: pd.DataFrame, idx: int, ida_number: int) -> np.ndarray:
        """
        I build feature vector for IDA price prediction (~25 features).

        Features are available at each IDA gate closure:
        - DAM prices for delivery day (known at D-1 12:00)
        - Historical IDA-DAM spread patterns (7-day rolling)
        - Current ISP1 UP/DOWN prices (real market signal)
        - Recent imbalance direction/magnitude
        - aFRR capacity price (system stress indicator)
        - Calendar features (hour, DoW, month)
        """
        row = df.iloc[idx]
        ts = df.index[idx]
        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        features = []

        # I add DAM price statistics for the delivery day (5 features)
        dam_price = row.get('price', 80.0)
        features.append(dam_price / 100.0)

        # I compute 24h DAM stats (known at gate closure)
        w24 = 24 * sph
        start = max(0, idx - w24)
        recent_prices = df.iloc[start:idx]['price'].values
        if len(recent_prices) > 0:
            features.append(np.mean(recent_prices) / 100.0)
            features.append(np.std(recent_prices) / 50.0 if len(recent_prices) > 1 else 0.2)
            features.append(np.max(recent_prices) / 100.0)
            features.append(np.min(recent_prices) / 100.0)
        else:
            features.extend([0.8, 0.2, 1.2, 0.6])

        # I add ISP1 UP/DOWN prices (real, available at gate time) (3 features)
        isp1_up = row.get('isp1_price_up', dam_price)
        isp1_down = row.get('isp1_price_down', dam_price)
        features.append(isp1_up / 100.0)
        features.append(isp1_down / 100.0)
        features.append((isp1_up - isp1_down) / 100.0)  # ISP1 bid-ask spread

        # I add imbalance features (3 features)
        net_imb = row.get('net_imbalance_mw_lag_1h', row.get('net_imbalance_mw', 0.0))
        features.append(np.sign(net_imb))
        features.append(min(abs(net_imb) / 500.0, 1.0))
        imb_price = row.get('imbalance_price_lag_1h', dam_price)
        features.append((imb_price - dam_price) / 100.0)

        # I add aFRR capacity price (system stress indicator) (1 feature)
        features.append(row.get('afrr_cap_up_price', 20.0) / 50.0)

        # I add mFRR spread (1 feature)
        mfrr_up = row.get('mfrr_price_up_lag_1h', row.get('mfrr_price_up', 120.0))
        mfrr_down = row.get('mfrr_price_down_lag_1h', row.get('mfrr_price_down', 60.0))
        features.append((mfrr_up - mfrr_down) / 200.0)

        # I add price lags for same hour (4 features)
        for lag_days in [1, 2, 3, 7]:
            lag_idx = idx - lag_days * 24 * sph
            if 0 <= lag_idx < len(df):
                features.append(df.iloc[lag_idx].get('price', dam_price) / 100.0)
            else:
                features.append(dam_price / 100.0)

        # I add calendar features (6 features)
        hour = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
        month = ts.month
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))

        # I add IDA-specific signal: which IDA this is (1 feature)
        features.append(ida_number / 3.0)

        # I add Serbia DAM spread (1 feature)
        serbia = row.get('serbia_dam_price', np.nan)
        if not np.isnan(serbia):
            features.append((serbia - dam_price) / 100.0)
        else:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int
    ) -> 'IDASubForecaster':
        """I train one LightGBM model per IDA auction."""
        try:
            import lightgbm as lgb
        except ImportError:
            print("  WARNING: LightGBM not available, IDA forecaster will use DAM fallback")
            return self

        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        print("  Training IDA sub-forecaster (3 auction models)...")

        for ida_num in [1, 2, 3]:
            X_train, y_train = [], []
            X_val, y_val = [], []

            # I generate synthetic IDA prices for training targets
            dam_prices = df['price'].values
            ida_prices = self._synthesize_ida_prices(dam_prices, ida_num, seed=42 + ida_num)

            for idx in range(48 * sph, len(df)):
                features = self.build_features(df, idx, ida_num)
                # I predict the IDA clearing price (target = synthetic IDA price)
                target = ida_prices[idx]

                if idx < train_end_idx:
                    X_train.append(features)
                    y_train.append(target)
                elif idx < val_end_idx:
                    X_val.append(features)
                    y_val.append(target)

            if len(X_train) < 100:
                print(f"    IDA{ida_num}: insufficient training data ({len(X_train)} samples)")
                continue

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            params = {
                'objective': 'regression',
                'metric': 'mae',
                'max_depth': 6,
                'n_estimators': 500,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
            }

            model = lgb.LGBMRegressor(**params)

            if len(X_val) > 10:
                X_v = np.array(X_val)
                y_v = np.array(y_val)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_v, y_v)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                model.fit(X_train, y_train)

            self.models[ida_num] = model
            print(f"    IDA{ida_num}: trained on {len(X_train)} samples")

        self.is_trained = len(self.models) > 0
        print(f"  IDA: trained {len(self.models)}/3 auction models")
        return self

    def predict(self, df: pd.DataFrame, idx: int, ida_number: int) -> np.ndarray:
        """
        I predict IDA clearing prices for the delivery window.

        Returns:
            Array of hourly clearing price predictions for the delivery period.
            IDA1/IDA2: 24 values (hours 0-23)
            IDA3: 12 values (hours 12-23)
        """
        config = self.IDA_CONFIG[ida_number]
        delivery_hours = config['delivery_hours']
        n_hours = len(delivery_hours)
        predictions = np.zeros(n_hours)

        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        if self.is_trained and ida_number in self.models:
            # I predict for each hour in the delivery window
            for h_idx, hour in enumerate(delivery_hours):
                # I build features at the gate closure time (current idx)
                features = self.build_features(df, idx, ida_number).reshape(1, -1)
                predictions[h_idx] = self.models[ida_number].predict(features)[0]
        else:
            # I fall back to DAM prices with OU-noise for uncertainty
            for h_idx, hour in enumerate(delivery_hours):
                target_idx = idx + hour * sph
                if 0 <= target_idx < len(df):
                    dam_price = df.iloc[target_idx].get('price', 80.0)
                else:
                    dam_price = df.iloc[idx].get('price', 80.0)
                # I add small noise to avoid oracle effect
                noise_std = {1: 4.0, 2: 8.0, 3: 12.0}.get(ida_number, 8.0)
                predictions[h_idx] = dam_price + np.random.normal(0, noise_std)

        # I ensure non-negative prices
        predictions = np.maximum(predictions, 0.0)
        return predictions

    def predict_mean_spread(self, df: pd.DataFrame, idx: int, ida_number: int) -> float:
        """
        I predict the mean IDA-DAM spread for the delivery window.

        Positive spread means IDA > DAM (sell on IDA is profitable).
        Negative spread means IDA < DAM (buy on IDA is profitable).
        """
        ida_prices = self.predict(df, idx, ida_number)
        config = self.IDA_CONFIG[ida_number]
        delivery_hours = config['delivery_hours']
        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        dam_prices = []
        for hour in delivery_hours:
            target_idx = idx + hour * sph
            if 0 <= target_idx < len(df):
                dam_prices.append(df.iloc[target_idx].get('price', 80.0))
            else:
                dam_prices.append(df.iloc[idx].get('price', 80.0))

        dam_prices = np.array(dam_prices)
        spread = np.mean(ida_prices - dam_prices)
        return float(spread)


class ImbalanceSubForecaster:
    """
    I forecast system imbalance direction and magnitude for the next 4 hours.

    Features:
    - Load gradient (rate of change)
    - RES deviation from expected (solar/wind forecast error proxy)
    - Hour-of-day (imbalance patterns are time-dependent)
    - Recent imbalance history (autocorrelation)
    """

    def __init__(self):
        self.direction_model = None  # I use classification for direction
        self.magnitude_model = None  # I use regression for magnitude
        self.is_trained = False

    def build_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """I build features for imbalance prediction."""
        row = df.iloc[idx]
        ts = df.index[idx]
        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        features = []

        # I add time features (4)
        hour = ts.hour + ts.minute / 60.0
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        dow = ts.dayofweek
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))

        # I add load gradient (2) - rate of change over 1h and 3h
        load = row.get('load_mw', 5000.0)
        for lag in [1 * sph, 3 * sph]:
            lag_idx = idx - lag
            if lag_idx >= 0:
                old_load = df.iloc[lag_idx].get('load_mw', load)
                features.append((load - old_load) / max(abs(old_load), 1.0))
            else:
                features.append(0.0)

        # I add RES deviation (2)
        res = row.get('res_total_mw', 0.0)
        w24 = 24 * sph
        start = max(0, idx - w24)
        res_mean = df.iloc[start:idx]['res_total_mw'].mean() if 'res_total_mw' in df.columns and idx > 0 else res
        features.append((res - res_mean) / max(abs(res_mean), 1.0))
        features.append(res / 5000.0)

        # I add recent imbalance history (6) - last 6 values
        for lag in range(1, 7):
            lag_idx = idx - lag * sph
            if lag_idx >= 0 and 'net_imbalance_mw' in df.columns:
                features.append(df.iloc[lag_idx].get('net_imbalance_mw', 0.0) / 500.0)
            else:
                features.append(0.0)

        # I add imbalance rolling statistics (3)
        w = 6 * sph
        start = max(0, idx - w)
        if 'net_imbalance_mw' in df.columns:
            recent_imb = df.iloc[start:idx]['net_imbalance_mw'].values
            if len(recent_imb) > 0:
                features.append(np.mean(recent_imb) / 500.0)
                features.append(np.std(recent_imb) / 500.0 if len(recent_imb) > 1 else 0.0)
                features.append(np.sign(np.mean(recent_imb)))
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])

        # I add price features (3) - price deviations signal imbalance
        dam_price = row.get('price', 80.0)
        mean_24h = row.get('price_mean_24h', dam_price)
        features.append((dam_price - mean_24h) / max(abs(mean_24h), 1.0))
        features.append(row.get('mfrr_spread', 60.0) / 500.0)
        features.append(row.get('afrr_up', 80.0) / 200.0)

        return np.array(features, dtype=np.float32)

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
        horizon_steps: int = None
    ) -> 'ImbalanceSubForecaster':
        """I train imbalance direction and magnitude models."""
        try:
            import lightgbm as lgb
        except ImportError:
            print("  WARNING: LightGBM not available for Imbalance forecaster")
            return self

        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1
        if horizon_steps is None:
            horizon_steps = 4 * sph  # 4 hours ahead

        if 'net_imbalance_mw' not in df.columns:
            print("  WARNING: net_imbalance_mw not in data, skipping imbalance forecaster")
            return self

        print(f"  Training Imbalance sub-forecaster ({horizon_steps} steps ahead)...")

        X_train, y_dir_train, y_mag_train = [], [], []
        X_val, y_dir_val, y_mag_val = [], [], []

        for idx in range(48 * sph, len(df) - horizon_steps):
            features = self.build_features(df, idx)

            # I predict average imbalance over the next 4 hours
            future_slice = df.iloc[idx + 1: idx + 1 + horizon_steps]
            if 'net_imbalance_mw' in future_slice.columns:
                avg_imb = future_slice['net_imbalance_mw'].mean()
            else:
                continue

            if np.isnan(avg_imb):
                continue

            direction = 1.0 if avg_imb > 30.0 else (-1.0 if avg_imb < -30.0 else 0.0)
            magnitude = abs(avg_imb)

            if idx < train_end_idx:
                X_train.append(features)
                y_dir_train.append(direction)
                y_mag_train.append(magnitude)
            elif idx < val_end_idx:
                X_val.append(features)
                y_dir_val.append(direction)
                y_mag_val.append(magnitude)

        if len(X_train) < 100:
            print(f"  Imbalance: insufficient training data ({len(X_train)} samples)")
            return self

        X_train = np.array(X_train)
        y_mag_train = np.array(y_mag_train)
        y_dir_train_num = np.array(y_dir_train)
        # I map direction {-1, 0, 1} to {0, 1, 2} for classification
        y_dir_cls = (y_dir_train_num + 1).astype(int)

        # I train magnitude model (regression)
        self.magnitude_model = lgb.LGBMRegressor(
            max_depth=5, n_estimators=300, learning_rate=0.05,
            objective='regression', metric='mae', verbose=-1
        )

        if len(X_val) > 20:
            X_v = np.array(X_val)
            self.magnitude_model.fit(
                X_train, y_mag_train,
                eval_set=[(X_v, np.array(y_mag_val))],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
        else:
            self.magnitude_model.fit(X_train, y_mag_train)

        # I train direction model (classification)
        self.direction_model = lgb.LGBMClassifier(
            max_depth=5, n_estimators=300, learning_rate=0.05,
            objective='multiclass', num_class=3, metric='multi_logloss',
            verbose=-1
        )

        if len(X_val) > 20:
            X_v = np.array(X_val)
            y_dir_val_cls = (np.array(y_dir_val) + 1).astype(int)
            self.direction_model.fit(
                X_train, y_dir_cls,
                eval_set=[(X_v, y_dir_val_cls)],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
        else:
            self.direction_model.fit(X_train, y_dir_cls)

        self.is_trained = True
        print(f"    Imbalance: {len(X_train)} train samples, trained OK")
        return self

    def predict(self, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        I predict imbalance direction and magnitude.

        Returns:
            dict with keys: direction (-1/0/+1), magnitude (MW), opportunity_score (0-1)
        """
        if not self.is_trained:
            return {'direction': 0.0, 'magnitude': 0.0, 'opportunity_score': 0.0}

        features = self.build_features(df, idx).reshape(1, -1)

        # I predict direction
        dir_probs = self.direction_model.predict_proba(features)[0]
        dir_class = np.argmax(dir_probs)
        direction = dir_class - 1  # I map {0,1,2} back to {-1,0,+1}

        # I predict magnitude
        magnitude = float(self.magnitude_model.predict(features)[0])
        magnitude = max(0.0, magnitude)

        # I compute opportunity score: higher when imbalance is large and confident
        confidence = max(dir_probs) - 1.0 / 3.0  # I subtract uniform baseline
        opportunity = np.clip(confidence * 2.0 * min(magnitude / 300.0, 1.0), 0.0, 1.0)

        return {
            'direction': float(direction),
            'magnitude': magnitude,
            'opportunity_score': float(opportunity)
        }


class DAMDayAheadForecaster:
    """
    I forecast 24 hourly Greek DAM prices for D+1 using Serbia DAM as primary signal.

    Key features per hour (21 total):
    1. Serbia D+1 price for this hour (available D-1 ~12:00)
    2. Serbia D+1 daily mean
    3. Serbia D+1 daily spread (max - min)
    4. Serbia D+1 relative position (price_h - mean) / spread
    5-8. Greek DAM same-hour lags (D-1, D-2, D-3, D-7)
    9-12. Greek DAM 7-day rolling stats (mean, std, min, max)
    13-14. Hour sin/cos encoding
    15-16. DoW sin/cos encoding
    17-18. Month sin/cos encoding
    19. Load forecast (normalized)
    20. RES forecast (normalized)
    21. Historical Serbia-Greece spread for this hour (7-day avg)

    Output: 24 hourly point predictions + P10/P90 quantile estimates
    Expansion: 96 QH via cubic spline interpolation
    """

    N_FEATURES = 21

    def __init__(self):
        self.point_models = {}   # I store one model per hour (0-23)
        self.p10_models = {}     # I store P10 quantile model per hour
        self.p90_models = {}     # I store P90 quantile model per hour
        self.is_trained = False

    def build_day_features(
        self,
        df: pd.DataFrame,
        day_idx: int,
        serbia_24h: np.ndarray,
        sph: int = 1,
    ) -> Dict[int, np.ndarray]:
        """
        I build feature vectors for all 24 hours of a target day.

        Args:
            df: DataFrame with 'price' column and DatetimeIndex
            day_idx: Index of the first row of the target day in df
            serbia_24h: 24 hourly Serbia D+1 prices
            sph: Steps per hour (1 for hourly, 4 for 15-min)

        Returns:
            Dict mapping hour (0-23) to feature arrays of shape (N_FEATURES,)
        """
        serbia_mean = np.mean(serbia_24h)
        serbia_spread = np.max(serbia_24h) - np.min(serbia_24h)
        if serbia_spread < 1e-6:
            serbia_spread = 1.0

        features_by_hour = {}

        for hour in range(24):
            features = []

            # 1. Serbia D+1 price for this hour
            features.append(serbia_24h[hour])

            # 2. Serbia D+1 daily mean
            features.append(serbia_mean)

            # 3. Serbia D+1 daily spread
            features.append(serbia_spread)

            # 4. Serbia D+1 relative position
            features.append((serbia_24h[hour] - serbia_mean) / serbia_spread)

            # 5-8. Greek DAM same-hour lags (D-1, D-2, D-3, D-7)
            for lag_days in [1, 2, 3, 7]:
                lag_idx = day_idx + hour * sph - lag_days * 24 * sph
                if 0 <= lag_idx < len(df):
                    features.append(df.iloc[lag_idx].get('price', 80.0))
                else:
                    features.append(80.0)

            # 9-12. Greek DAM 7-day rolling stats (168h lookback from day start)
            lookback_start = max(0, day_idx - 7 * 24 * sph)
            lookback_end = day_idx
            if lookback_end > lookback_start:
                recent_prices = df.iloc[lookback_start:lookback_end]['price'].values
                valid = recent_prices[~np.isnan(recent_prices)]
                if len(valid) > 0:
                    features.append(np.mean(valid))
                    features.append(np.std(valid) if len(valid) > 1 else 10.0)
                    features.append(np.min(valid))
                    features.append(np.max(valid))
                else:
                    features.extend([80.0, 10.0, 60.0, 120.0])
            else:
                features.extend([80.0, 10.0, 60.0, 120.0])

            # 13-14. Hour sin/cos
            features.append(np.sin(2 * np.pi * hour / 24))
            features.append(np.cos(2 * np.pi * hour / 24))

            # 15-16. DoW sin/cos
            ts = df.index[min(day_idx, len(df) - 1)]
            dow = ts.dayofweek
            features.append(np.sin(2 * np.pi * dow / 7))
            features.append(np.cos(2 * np.pi * dow / 7))

            # 17-18. Month sin/cos
            month = ts.month
            features.append(np.sin(2 * np.pi * month / 12))
            features.append(np.cos(2 * np.pi * month / 12))

            # 19. Load forecast (normalized)
            row_idx = min(day_idx + hour * sph, len(df) - 1)
            features.append(df.iloc[row_idx].get('load_mw', 5000.0) / 10000.0)

            # 20. RES forecast (normalized)
            features.append(df.iloc[row_idx].get('res_total_mw', 2000.0) / 5000.0)

            # 21. Historical Serbia-Greece spread for this hour (7-day avg)
            spread_vals = []
            for lag_days in range(1, 8):
                lag_idx = day_idx + hour * sph - lag_days * 24 * sph
                if 0 <= lag_idx < len(df):
                    gr_price = df.iloc[lag_idx].get('price', np.nan)
                    sr_price = df.iloc[lag_idx].get('serbia_dam_price', np.nan)
                    if not np.isnan(gr_price) and not np.isnan(sr_price):
                        spread_vals.append(sr_price - gr_price)
            features.append(np.mean(spread_vals) if spread_vals else 0.0)

            features_by_hour[hour] = np.array(features, dtype=np.float32)

        return features_by_hour

    def fit(
        self,
        df: pd.DataFrame,
        serbia_df: pd.DataFrame,
        train_end: pd.Timestamp,
        val_end: pd.Timestamp,
    ) -> 'DAMDayAheadForecaster':
        """
        I train 24×3 LightGBM models (point + P10 + P90) for D+1 forecasting.

        Args:
            df: Greek DAM prices DataFrame with DatetimeIndex and 'price' column
            serbia_df: Serbia DAM prices with DatetimeIndex and 'price' column
            train_end: End of training period (exclusive)
            val_end: End of validation period (exclusive)
        """
        try:
            import lightgbm as lgb
        except ImportError:
            print("  WARNING: LightGBM not available, D+1 forecaster will use regression fallback")
            return self

        sph = max(1, int(round(
            1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600
        ))) if len(df) > 1 else 1

        print("  Training D+1 Day-Ahead forecaster (24 hourly × 3 objectives)...")

        # I build per-hour training data by iterating over days
        X_by_hour = {h: {'train': [], 'val': []} for h in range(24)}
        y_by_hour = {h: {'train': [], 'val': []} for h in range(24)}

        dates = df.index.normalize().unique()

        for day_date in dates:
            day_mask = df.index.normalize() == day_date
            day_indices = np.where(day_mask)[0]
            if len(day_indices) < 24 * sph:
                continue

            day_idx = day_indices[0]

            # I need Serbia D+1 prices for this day
            serbia_day = serbia_df.loc[
                serbia_df.index.normalize() == day_date, 'price'
            ]
            if len(serbia_day) < 24:
                continue

            # I resample Serbia to exactly 24 hourly values
            serbia_24h = np.zeros(24)
            for h in range(24):
                hour_mask = serbia_day.index.hour == h
                hour_vals = serbia_day[hour_mask].values
                if len(hour_vals) > 0:
                    serbia_24h[h] = hour_vals[0]
                else:
                    serbia_24h[h] = serbia_day.mean()

            # I need at least 7 days of history
            if day_idx < 7 * 24 * sph:
                continue

            # I build features for all 24 hours
            features_by_hour = self.build_day_features(df, day_idx, serbia_24h, sph)

            # I collect targets (actual Greek prices for this day)
            for hour in range(24):
                target_idx = day_idx + hour * sph
                if target_idx >= len(df):
                    continue
                target = df.iloc[target_idx]['price']
                if np.isnan(target):
                    continue

                if day_date < train_end:
                    X_by_hour[hour]['train'].append(features_by_hour[hour])
                    y_by_hour[hour]['train'].append(target)
                elif day_date < val_end:
                    X_by_hour[hour]['val'].append(features_by_hour[hour])
                    y_by_hour[hour]['val'].append(target)

        # I train models for each hour
        for hour in range(24):
            X_train = np.array(X_by_hour[hour]['train']) if X_by_hour[hour]['train'] else np.empty((0, self.N_FEATURES))
            y_train = np.array(y_by_hour[hour]['train']) if y_by_hour[hour]['train'] else np.empty(0)

            if len(X_train) < 30:
                continue

            X_val = np.array(X_by_hour[hour]['val']) if X_by_hour[hour]['val'] else None
            y_val = np.array(y_by_hour[hour]['val']) if X_by_hour[hour]['val'] else None

            base_params = {
                'max_depth': 6,
                'n_estimators': 500,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbose': -1,
            }

            # I train point model (L2 regression)
            point_model = lgb.LGBMRegressor(
                objective='regression', metric='mae', **base_params
            )
            if X_val is not None and len(X_val) > 5:
                point_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                point_model.fit(X_train, y_train)
            self.point_models[hour] = point_model

            # I train P10 quantile model
            p10_model = lgb.LGBMRegressor(
                objective='quantile', alpha=0.10, metric='quantile', **base_params
            )
            if X_val is not None and len(X_val) > 5:
                p10_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                p10_model.fit(X_train, y_train)
            self.p10_models[hour] = p10_model

            # I train P90 quantile model
            p90_model = lgb.LGBMRegressor(
                objective='quantile', alpha=0.90, metric='quantile', **base_params
            )
            if X_val is not None and len(X_val) > 5:
                p90_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                p90_model.fit(X_train, y_train)
            self.p90_models[hour] = p90_model

        self.is_trained = len(self.point_models) > 0
        print(f"  D+1 forecaster: trained {len(self.point_models)}/24 hourly models × 3 objectives")
        return self

    def predict_day(
        self,
        df: pd.DataFrame,
        day_idx: int,
        serbia_24h: np.ndarray,
        sph: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        I predict 24 hourly Greek DAM prices for D+1.

        Returns:
            Dict with 'hourly' (24,), 'p10' (24,), 'p90' (24,) arrays
        """
        features_by_hour = self.build_day_features(df, day_idx, serbia_24h, sph)

        hourly = np.zeros(24)
        p10 = np.zeros(24)
        p90 = np.zeros(24)

        for hour in range(24):
            feat = features_by_hour[hour].reshape(1, -1)

            if self.is_trained and hour in self.point_models:
                hourly[hour] = self.point_models[hour].predict(feat)[0]
                p10[hour] = self.p10_models[hour].predict(feat)[0]
                p90[hour] = self.p90_models[hour].predict(feat)[0]
            else:
                # I use Serbia regression fallback
                params = HOURLY_GR_RS_PARAMS.get(hour, {'slope': 0.7, 'intercept': 20.0})
                pred = params['slope'] * serbia_24h[hour] + params['intercept']
                hourly[hour] = pred
                p10[hour] = pred * 0.85
                p90[hour] = pred * 1.15

            # I enforce quantile ordering: P10 <= point <= P90
            if p10[hour] > hourly[hour]:
                p10[hour] = hourly[hour] * 0.90
            if p90[hour] < hourly[hour]:
                p90[hour] = hourly[hour] * 1.10

        return {'hourly': hourly, 'p10': p10, 'p90': p90}

    def _expand_to_96qh(self, hourly_prices: np.ndarray) -> np.ndarray:
        """I expand 24 hourly prices to 96 QH using cubic spline interpolation."""
        from scipy.interpolate import CubicSpline

        # I place hourly values at the midpoint of each hour in QH indices
        hour_midpoints = np.arange(24) * 4 + 1.5

        # I add ghost points at boundaries for smooth extrapolation to QH 0 and 95
        # Ghost point before hour 0: mirror hour 23 price (wrapping)
        # Ghost point after hour 23: mirror hour 0 price (wrapping)
        extended_x = np.concatenate([[-2.5], hour_midpoints, [97.5]])
        extended_y = np.concatenate([[hourly_prices[-1]], hourly_prices, [hourly_prices[0]]])

        cs = CubicSpline(extended_x, extended_y, bc_type='not-a-knot')
        qh_indices = np.arange(96)
        return cs(qh_indices)

    def predict_96qh(
        self,
        df: pd.DataFrame,
        day_idx: int,
        serbia_24h: np.ndarray,
        sph: int = 1,
    ) -> Dict[str, np.ndarray]:
        """
        I predict 96 quarter-hourly Greek DAM prices for D+1.

        Returns:
            Dict with 'qh' (96,), 'p10' (96,), 'p90' (96,), 'hourly' (24,) arrays
        """
        day_pred = self.predict_day(df, day_idx, serbia_24h, sph)

        qh = self._expand_to_96qh(day_pred['hourly'])
        qh_p10 = self._expand_to_96qh(day_pred['p10'])
        qh_p90 = self._expand_to_96qh(day_pred['p90'])

        # I enforce non-negativity and quantile ordering on QH level
        qh = np.maximum(qh, 0.0)
        qh_p10 = np.maximum(qh_p10, 0.0)
        qh_p90 = np.maximum(qh_p90, 0.0)
        qh_p10 = np.minimum(qh_p10, qh)
        qh_p90 = np.maximum(qh_p90, qh)

        return {
            'qh': qh,
            'p10': qh_p10,
            'p90': qh_p90,
            'hourly': day_pred['hourly'],
        }

    def save(self, path: str):
        """I save all models to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'point_models': self.point_models,
                'p10_models': self.p10_models,
                'p90_models': self.p90_models,
                'is_trained': self.is_trained,
            }, f)
        print(f"DAMDayAheadForecaster saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'DAMDayAheadForecaster':
        """I load models from a pickle file."""
        forecaster = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        forecaster.point_models = data['point_models']
        forecaster.p10_models = data['p10_models']
        forecaster.p90_models = data['p90_models']
        forecaster.is_trained = data['is_trained']
        return forecaster


class MarketForecaster:
    """
    I orchestrate all three sub-forecasters into a unified interface.

    Usage:
        forecaster = MarketForecaster()
        forecaster.fit(df, train_end_idx, val_end_idx)
        forecaster.save("models/market_forecaster.pkl")

        # At inference time:
        forecaster = MarketForecaster.load("models/market_forecaster.pkl")
        forecast = forecaster.predict_all(df, idx)
    """

    def __init__(self):
        self.dam_forecaster = DAMSubForecaster()
        self.intraday_forecaster = IntraDaySubForecaster()
        self.imbalance_forecaster = ImbalanceSubForecaster()
        self.ida_forecaster = IDASubForecaster()

    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int
    ) -> 'MarketForecaster':
        """I train all sub-forecasters on the same data with temporal split."""
        print("=" * 50)
        print("TRAINING MARKET FORECASTER")
        print("=" * 50)

        self.dam_forecaster.fit(df, train_end_idx, val_end_idx)
        self.intraday_forecaster.fit(df, train_end_idx, val_end_idx)
        self.imbalance_forecaster.fit(df, train_end_idx, val_end_idx)
        self.ida_forecaster.fit(df, train_end_idx, val_end_idx)

        print("  All sub-forecasters trained.")
        return self

    def predict_all(self, df: pd.DataFrame, idx: int) -> Dict[str, np.ndarray]:
        """
        I return all forecasts needed by the enhanced observation space.

        Returns dict with:
            'dam_forecast_24h': array[24] — hourly DAM forecasts
            'dam_forecast_std': array[24] — confidence (std)
            'id_forecast': dict {1: price, 2: price, 4: price, 8: price}
            'imbalance': dict {direction, magnitude, opportunity_score}
            'ida_forecasts': dict {1: array[24], 2: array[24], 3: array[12]}
            'ida_spreads': dict {1: float, 2: float, 3: float} — mean IDA-DAM spread
        """
        dam_preds, dam_stds = self.dam_forecaster.predict_with_confidence(df, idx, 24)
        id_preds = self.intraday_forecaster.predict(df, idx)
        imb_preds = self.imbalance_forecaster.predict(df, idx)

        # I add IDA price forecasts for each auction
        ida_forecasts = {}
        ida_spreads = {}
        for ida_num in [1, 2, 3]:
            ida_forecasts[ida_num] = self.ida_forecaster.predict(df, idx, ida_num)
            ida_spreads[ida_num] = self.ida_forecaster.predict_mean_spread(df, idx, ida_num)

        return {
            'dam_forecast_24h': dam_preds,
            'dam_forecast_std': dam_stds,
            'id_forecast': id_preds,
            'imbalance': imb_preds,
            'ida_forecasts': ida_forecasts,
            'ida_spreads': ida_spreads,
        }

    def batch_predict(
        self,
        df: pd.DataFrame,
        start_idx: int = 0,
        end_idx: int = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        I run forecasts over the entire dataset and return a DataFrame
        with forecast columns aligned to the original index.

        This is used to pre-compute forecast columns for training data
        so the environment doesn't need to run the forecaster at each step.
        """
        if end_idx is None:
            end_idx = len(df)

        sph = max(1, int(round(1.0 / (df.index[1] - df.index[0]).total_seconds() * 3600))) if len(df) > 1 else 1

        # I initialize output columns
        n = end_idx - start_idx
        results = {
            'dam_forecast_next_4h_mean': np.full(n, np.nan),
            'dam_forecast_next_4h_max': np.full(n, np.nan),
            'dam_forecast_next_4h_min': np.full(n, np.nan),
            'dam_forecast_spread': np.full(n, np.nan),
            'dam_forecast_vs_current': np.full(n, np.nan),
            'id_forecast_1h': np.full(n, np.nan),
            'id_forecast_2h': np.full(n, np.nan),
            'id_forecast_4h': np.full(n, np.nan),
            'id_forecast_8h': np.full(n, np.nan),
            'imbalance_forecast_dir': np.full(n, np.nan),
            'imbalance_forecast_mag': np.full(n, np.nan),
            'mfrr_opportunity_score': np.full(n, np.nan),
            'forecast_confidence': np.full(n, np.nan),
            'hours_to_dam_peak': np.full(n, np.nan),
            'hours_to_dam_trough': np.full(n, np.nan),
        }

        for i in range(n):
            idx = start_idx + i

            if verbose and (i + 1) % 5000 == 0:
                print(f"  Batch forecast: {i + 1}/{n}...")

            try:
                forecast = self.predict_all(df, idx)

                dam_4h = forecast['dam_forecast_24h'][:4]
                results['dam_forecast_next_4h_mean'][i] = np.mean(dam_4h)
                results['dam_forecast_next_4h_max'][i] = np.max(dam_4h)
                results['dam_forecast_next_4h_min'][i] = np.min(dam_4h)
                results['dam_forecast_spread'][i] = np.max(dam_4h) - np.min(dam_4h)
                results['dam_forecast_vs_current'][i] = np.mean(dam_4h) - df.iloc[idx].get('price', 80.0)

                id_fc = forecast['id_forecast']
                results['id_forecast_1h'][i] = id_fc.get(1, 80.0)
                results['id_forecast_2h'][i] = id_fc.get(2, 80.0)
                results['id_forecast_4h'][i] = id_fc.get(4, 80.0)
                results['id_forecast_8h'][i] = id_fc.get(8, 80.0)

                imb = forecast['imbalance']
                results['imbalance_forecast_dir'][i] = imb['direction']
                results['imbalance_forecast_mag'][i] = imb['magnitude']
                results['mfrr_opportunity_score'][i] = imb['opportunity_score']

                # I compute confidence from DAM forecast std
                avg_std = np.mean(forecast['dam_forecast_std'][:4])
                results['forecast_confidence'][i] = 1.0 / (1.0 + avg_std / 20.0)

                # I compute timing signals
                dam_24h = forecast['dam_forecast_24h']
                peak_idx = np.argmax(dam_24h)
                trough_idx = np.argmin(dam_24h)
                results['hours_to_dam_peak'][i] = float(peak_idx + 1)
                results['hours_to_dam_trough'][i] = float(trough_idx + 1)

            except Exception:
                # I leave NaN for failed predictions
                pass

        # I build a DataFrame aligned to the original index
        forecast_df = pd.DataFrame(results, index=df.index[start_idx:end_idx])

        # I forward-fill any gaps
        forecast_df = forecast_df.ffill().bfill()

        return forecast_df

    def save(self, path: str):
        """I save all sub-forecasters to a single pickle file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'dam': self.dam_forecaster,
                'intraday': self.intraday_forecaster,
                'imbalance': self.imbalance_forecaster,
                'ida': self.ida_forecaster,
            }, f)
        print(f"MarketForecaster saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'MarketForecaster':
        """I load all sub-forecasters from a pickle file."""
        forecaster = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)

        forecaster.dam_forecaster = data['dam']
        forecaster.intraday_forecaster = data['intraday']
        forecaster.imbalance_forecaster = data['imbalance']
        # I handle backward compatibility for older saved files without IDA
        forecaster.ida_forecaster = data.get('ida', IDASubForecaster())
        return forecaster
