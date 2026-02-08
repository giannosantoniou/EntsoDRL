"""
IntraDay Price Forecaster for Battery Trading (LightGBM Version).

I predict IntraDay prices for the next 12 hours using ~42 engineered features.
I use LightGBM instead of Ridge to capture non-linear price dynamics (spikes,
regime changes) and improve accuracy from ~38% MAPE to target <15%.

Backward compatible: I can load old Ridge pickles transparently.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# I define feature names once so batch and single-row creation stay in sync
FEATURE_NAMES = [
    # Time features (0-6)
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
    'hour_of_day',
    # RES features (7-9)
    'solar', 'wind_onshore', 'res_total',
    # aFRR MW features (10-12)
    'afrr_up_mw', 'afrr_down_mw', 'afrr_mw_ratio',
    # aFRR price features (13-14)
    'afrr_up_price', 'afrr_down_price',
    # mFRR features (15-17)
    'mfrr_up', 'mfrr_down', 'mfrr_spread',
    # Imbalance features (18-20)
    'long', 'short', 'imbalance_spread',
    # Current price (21)
    'price_norm',
    # Price lags (22-29)
    'price_lag_1h', 'price_lag_2h', 'price_lag_3h', 'price_lag_6h',
    'price_lag_12h', 'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
    # Rolling means (30-32)
    'rolling_mean_6h', 'rolling_mean_12h', 'rolling_mean_24h',
    # Rolling std / volatility (33-35)
    'rolling_std_6h', 'rolling_std_12h', 'rolling_std_24h',
    # Momentum features (36-37)
    'price_change_1h', 'price_change_3h',
    # Relative price features (38-39)
    'price_vs_mean_24h', 'price_percentile_24h',
    # RES deviation (40)
    'res_vs_mean',
]

NUM_FEATURES = len(FEATURE_NAMES)  # 41


class IntraDayForecaster:
    """
    LightGBM-based IntraDay price forecaster with 41 features.

    I train 12 separate models (one per horizon) with horizon-dependent
    hyperparameters and early stopping on a chronological validation set.
    """

    def __init__(self, max_horizon: int = 12):
        self.max_horizon = max_horizon
        self.models: Dict[int, Any] = {}
        self.scalers: Dict[int, StandardScaler] = {}
        self.is_trained = False
        self.model_type = 'lightgbm'  # 'lightgbm' or 'ridge'
        self.feature_names = FEATURE_NAMES

        # Forecast error parameters for noise injection during RL training
        self.base_error_1h = 0.08
        self.base_error_12h = 0.20

    # ------------------------------------------------------------------
    # Feature engineering (batch — vectorized)
    # ------------------------------------------------------------------

    def _create_features_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        I create feature matrix for all timesteps using vectorized operations.
        Returns shape (n, NUM_FEATURES).
        """
        n = len(df)
        features = np.full((n, NUM_FEATURES), np.nan)

        price = df['price'].values.astype(np.float64)
        price_series = pd.Series(price)

        # --- Time features (0-6) ---
        features[:, 0] = df['hour_sin'].values
        features[:, 1] = df['hour_cos'].values
        features[:, 2] = df['day_sin'].values
        features[:, 3] = df['day_cos'].values
        features[:, 4] = df['month_sin'].values
        features[:, 5] = df['month_cos'].values
        # I derive hour_of_day from sin/cos encoding (atan2)
        hour_raw = np.arctan2(df['hour_sin'].values, df['hour_cos'].values)
        features[:, 6] = np.round((hour_raw * 12 / np.pi) % 24).astype(np.float64)

        # --- RES features (7-9) ---
        solar = df['solar'].fillna(0).values.astype(np.float64)
        wind = df['wind_onshore'].fillna(0).values.astype(np.float64)
        features[:, 7] = solar
        features[:, 8] = wind
        features[:, 9] = solar + wind  # res_total

        # --- aFRR MW features (10-12) ---
        afrr_up = df['afrr_up_mw'].fillna(600).values.astype(np.float64)
        afrr_down = df['afrr_down_mw'].fillna(130).values.astype(np.float64)
        features[:, 10] = afrr_up
        features[:, 11] = afrr_down
        # I use safe division for the ratio
        features[:, 12] = np.where(afrr_down > 0, afrr_up / afrr_down, 1.0)

        # --- aFRR price features (13-14) ---
        features[:, 13] = df['afrr_up_price'].fillna(0).values.astype(np.float64)
        features[:, 14] = df['afrr_down_price'].fillna(0).values.astype(np.float64)

        # --- mFRR features (15-17) ---
        mfrr_up = df['mfrr_up'].fillna(100).values.astype(np.float64)
        mfrr_down = df['mfrr_down'].fillna(80).values.astype(np.float64)
        features[:, 15] = mfrr_up
        features[:, 16] = mfrr_down
        features[:, 17] = mfrr_up - mfrr_down  # mfrr_spread

        # --- Imbalance features (18-20) ---
        long_val = df['long'].fillna(0).values.astype(np.float64)
        short_val = df['short'].fillna(0).values.astype(np.float64)
        features[:, 18] = long_val
        features[:, 19] = short_val
        features[:, 20] = short_val - long_val  # imbalance_spread

        # --- Current price (21) ---
        features[:, 21] = price

        # --- Price lags (22-29) ---
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
        for i, lag in enumerate(lags):
            features[:, 22 + i] = price_series.shift(lag).values

        # --- Rolling means (30-32) ---
        features[:, 30] = price_series.rolling(6, min_periods=1).mean().values
        features[:, 31] = price_series.rolling(12, min_periods=1).mean().values
        features[:, 32] = price_series.rolling(24, min_periods=1).mean().values

        # --- Rolling std / volatility (33-35) ---
        features[:, 33] = price_series.rolling(6, min_periods=2).std().values
        features[:, 34] = price_series.rolling(12, min_periods=2).std().values
        features[:, 35] = price_series.rolling(24, min_periods=2).std().values

        # --- Momentum (36-37) ---
        features[:, 36] = price - price_series.shift(1).values   # price_change_1h
        features[:, 37] = price - price_series.shift(3).values   # price_change_3h

        # --- Relative price features (38-39) ---
        mean_24h = price_series.rolling(24, min_periods=1).mean().values
        features[:, 38] = price - mean_24h  # price_vs_mean_24h

        # I compute a rolling percentile using rank-based approach
        percentile_vals = (
            price_series
            .rolling(24, min_periods=1)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            .values
        )
        features[:, 39] = percentile_vals

        # --- RES deviation (40) ---
        res_total = features[:, 9]
        res_mean = pd.Series(res_total).rolling(24, min_periods=1).mean().values
        features[:, 40] = res_total - res_mean

        return features

    # ------------------------------------------------------------------
    # Feature engineering (single row — for predict)
    # ------------------------------------------------------------------

    def _create_single_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """I create feature vector for a single timestep (used during inference)."""
        row = df.iloc[idx]
        price_col = df['price'].values.astype(np.float64)
        current_price = float(row.get('price', 100))

        features = np.full(NUM_FEATURES, np.nan)

        # --- Time features (0-6) ---
        features[0] = row.get('hour_sin', 0)
        features[1] = row.get('hour_cos', 1)
        features[2] = row.get('day_sin', 0)
        features[3] = row.get('day_cos', 1)
        features[4] = row.get('month_sin', 0)
        features[5] = row.get('month_cos', 1)
        hour_raw = np.arctan2(features[0], features[1])
        features[6] = round((hour_raw * 12 / np.pi) % 24)

        # --- RES features (7-9) ---
        solar = float(row.get('solar', 0) or 0)
        wind = float(row.get('wind_onshore', 0) or 0)
        features[7] = solar
        features[8] = wind
        features[9] = solar + wind

        # --- aFRR MW features (10-12) ---
        afrr_up = float(row.get('afrr_up_mw', 600) or 600)
        afrr_down = float(row.get('afrr_down_mw', 130) or 130)
        features[10] = afrr_up
        features[11] = afrr_down
        features[12] = afrr_up / afrr_down if afrr_down > 0 else 1.0

        # --- aFRR price features (13-14) ---
        features[13] = float(row.get('afrr_up_price', 0) or 0)
        features[14] = float(row.get('afrr_down_price', 0) or 0)

        # --- mFRR features (15-17) ---
        mfrr_up = float(row.get('mfrr_up', 100) or 100)
        mfrr_down = float(row.get('mfrr_down', 80) or 80)
        features[15] = mfrr_up
        features[16] = mfrr_down
        features[17] = mfrr_up - mfrr_down

        # --- Imbalance features (18-20) ---
        long_val = float(row.get('long', 0) or 0)
        short_val = float(row.get('short', 0) or 0)
        features[18] = long_val
        features[19] = short_val
        features[20] = short_val - long_val

        # --- Current price (21) ---
        features[21] = current_price

        # --- Price lags (22-29) ---
        lags = [1, 2, 3, 6, 12, 24, 48, 168]
        for i, lag in enumerate(lags):
            if idx >= lag:
                features[22 + i] = price_col[idx - lag]
            # else stays NaN — LightGBM handles it natively

        # --- Rolling means (30-32) ---
        for i, window in enumerate([6, 12, 24]):
            start = max(0, idx - window + 1)
            features[30 + i] = np.mean(price_col[start:idx + 1])

        # --- Rolling std (33-35) — I use ddof=1 to match pandas .std() ---
        for i, window in enumerate([6, 12, 24]):
            start = max(0, idx - window + 1)
            segment = price_col[start:idx + 1]
            features[33 + i] = np.std(segment, ddof=1) if len(segment) > 1 else 0.0

        # --- Momentum (36-37) ---
        features[36] = (current_price - price_col[idx - 1]) if idx >= 1 else np.nan
        features[37] = (current_price - price_col[idx - 3]) if idx >= 3 else np.nan

        # --- Relative price features (38-39) ---
        start_24 = max(0, idx - 23)
        mean_24h = np.mean(price_col[start_24:idx + 1])
        features[38] = current_price - mean_24h

        window_24 = price_col[start_24:idx + 1]
        rank = np.sum(window_24 <= current_price) / len(window_24)
        features[39] = rank

        # --- RES deviation (40) ---
        # I approximate using a simple lookback on the available data
        res_total = features[9]
        if idx >= 24 and 'solar' in df.columns and 'wind_onshore' in df.columns:
            res_past = (
                df['solar'].fillna(0).values[max(0, idx-23):idx+1]
                + df['wind_onshore'].fillna(0).values[max(0, idx-23):idx+1]
            )
            features[40] = res_total - np.mean(res_past)
        else:
            features[40] = 0.0

        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_lgb_params(self, horizon: int) -> dict:
        """I return horizon-dependent LightGBM hyperparameters."""
        if horizon <= 3:
            return {
                'objective': 'regression',
                'metric': 'mae',
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'num_leaves': 40,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 30,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'verbose': -1,
                'n_jobs': -1,
            }
        else:
            return {
                'objective': 'regression',
                'metric': 'mae',
                'n_estimators': 800,
                'max_depth': 7,
                'learning_rate': 0.03,
                'num_leaves': 60,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 30,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'verbose': -1,
                'n_jobs': -1,
            }

    def train(self, df: pd.DataFrame, verbose: bool = True):
        """
        I train 12 LightGBM models (one per horizon) with early stopping.
        Chronological split: 70% train, 15% validation, 15% test (held out).
        """
        if not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is required for training. "
                "Install with: pip install lightgbm>=4.0.0"
            )

        # I lowercase columns for consistency
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if verbose:
            print("Training IntraDay Forecaster (LightGBM, 41 features)...")
            print(f"  Data: {len(df)} hours")
            print(f"  Features: {NUM_FEATURES}")

        # I create all features at once (vectorized)
        if verbose:
            print("  Creating features...")
        X_all = self._create_features_batch(df)
        y_all = df['price'].values.astype(np.float64)

        # I need at least 168h (1 week) of history for weekly lag
        valid_start = 168
        X_all = X_all[valid_start:]
        y_all = y_all[valid_start:]

        # Chronological split for validation
        n = len(X_all)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        if verbose:
            print(f"  Split: train={train_end:,}, val={val_end - train_end:,}, "
                  f"test={n - val_end:,} (held out)")

        self.model_type = 'lightgbm'
        self.models = {}
        self.scalers = {}

        for horizon in range(1, self.max_horizon + 1):
            if horizon >= len(y_all):
                continue

            # I shift target by horizon
            X = X_all[:-horizon]
            y = y_all[horizon:]

            if len(X) < 500:
                if verbose:
                    print(f"  Horizon {horizon}h: Not enough data")
                continue

            # I split chronologically (accounting for the shift)
            t_end = min(train_end, len(X))
            v_end = min(val_end, len(X))

            X_train, y_train = X[:t_end], y[:t_end]
            X_val, y_val = X[t_end:v_end], y[t_end:v_end]

            if len(X_val) < 50:
                X_val, y_val = X_train[-500:], y_train[-500:]

            params = self._get_lgb_params(horizon)
            n_estimators = params.pop('n_estimators')

            model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)

            # I use early stopping on the validation set
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            self.models[horizon] = model

            if verbose:
                # I report on validation set
                y_pred_val = model.predict(X_val)
                mae = np.mean(np.abs(y_val - y_pred_val))
                mask = np.abs(y_val) > 1.0
                if mask.sum() > 0:
                    mape = np.mean(
                        np.abs((y_val[mask] - y_pred_val[mask]) / y_val[mask])
                    ) * 100
                else:
                    mape = float('nan')
                best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else n_estimators
                print(f"  Horizon {horizon:2d}h: MAE={mae:5.1f} EUR, "
                      f"MAPE={mape:4.1f}%, trees={best_iter}")

        self.is_trained = True
        if verbose:
            print("Training complete!")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        idx: int,
        add_noise: bool = True
    ) -> List[float]:
        """
        I predict prices for the next 12 hours from position idx.
        Interface is identical to the old Ridge version.
        """
        # I lowercase columns
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]

        current_price = float(df_lower.iloc[idx].get('price', 100))

        if not self.is_trained:
            # Fallback: persistence with noise
            predictions = []
            for h in range(1, self.max_horizon + 1):
                error_pct = self.base_error_1h + \
                    (self.base_error_12h - self.base_error_1h) * (h - 1) / 11
                if add_noise:
                    noise = np.random.normal(0, abs(current_price) * error_pct)
                else:
                    noise = 0
                predictions.append(max(0.1, current_price + noise))
            return predictions

        # I create features for this timestep
        features = self._create_single_features(df_lower, idx)

        predictions = []
        for horizon in range(1, self.max_horizon + 1):
            if horizon in self.models:
                model = self.models[horizon]

                if self.model_type == 'lightgbm':
                    # LightGBM handles NaN natively — no scaler needed
                    pred = model.predict(features.reshape(1, -1))[0]
                else:
                    # Ridge path (backward compat) — uses only first 22 features
                    feat_ridge = features[:22]
                    scaler = self.scalers.get(horizon)
                    if scaler is not None:
                        X_scaled = scaler.transform(feat_ridge.reshape(1, -1))
                    else:
                        X_scaled = feat_ridge.reshape(1, -1)
                    pred = model.predict(X_scaled)[0]

                # I add realistic noise for RL training
                if add_noise:
                    error_pct = self.base_error_1h + \
                        (self.base_error_12h - self.base_error_1h) * (horizon - 1) / 11
                    noise = np.random.normal(0, abs(pred) * error_pct)
                    pred = pred + noise

                predictions.append(max(0.1, float(pred)))
            else:
                predictions.append(current_price)

        return predictions

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self, horizon: int = 1) -> Optional[pd.DataFrame]:
        """I return feature importance for a given horizon's model."""
        if horizon not in self.models or self.model_type != 'lightgbm':
            return None

        model = self.models[horizon]
        importance = model.feature_importances_
        fi = pd.DataFrame({
            'feature': self.feature_names[:len(importance)],
            'importance': importance,
        }).sort_values('importance', ascending=False)
        fi['pct'] = fi['importance'] / fi['importance'].sum() * 100
        return fi

    # ------------------------------------------------------------------
    # Save / Load (version-aware)
    # ------------------------------------------------------------------

    def save(self, path: str):
        """I save the trained forecaster with model_type metadata."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'models': self.models,
                'scalers': self.scalers,
                'max_horizon': self.max_horizon,
                'base_error_1h': self.base_error_1h,
                'base_error_12h': self.base_error_12h,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
            }, f)
        print(f"Forecaster saved to {path} (type={self.model_type})")

    @classmethod
    def load(cls, path: str) -> 'IntraDayForecaster':
        """
        I load a trained forecaster.
        Backward compatible: detects Ridge (no model_type key) vs LightGBM.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(max_horizon=data['max_horizon'])
        forecaster.models = data['models']
        forecaster.scalers = data.get('scalers', {})
        forecaster.base_error_1h = data['base_error_1h']
        forecaster.base_error_12h = data['base_error_12h']
        forecaster.is_trained = data['is_trained']

        # I detect model type for backward compatibility
        forecaster.model_type = data.get('model_type', 'ridge')
        if 'feature_names' in data:
            forecaster.feature_names = data['feature_names']

        return forecaster


# ======================================================================
# Standalone training script
# ======================================================================

def train_and_save_forecaster(
    data_path: str = "data/feasible_data_with_balancing.csv",
    model_path: str = "models/intraday_forecaster.pkl"
):
    """I train and save the LightGBM forecaster."""
    print("=" * 60)
    print("TRAINING INTRADAY PRICE FORECASTER (LightGBM)")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    forecaster = IntraDayForecaster(max_horizon=12)
    forecaster.train(df, verbose=True)
    forecaster.save(model_path)

    # I run a quick test prediction
    print("\nTest prediction at idx=40000:")
    predictions = forecaster.predict(df, idx=40000, add_noise=False)
    actual_prices = []
    for h in range(1, 13):
        if 40000 + h < len(df):
            actual_prices.append(
                df.iloc[40000 + h].get('price', df.iloc[40000 + h].get('Price', 0))
            )
        else:
            actual_prices.append(np.nan)

    print(f"{'Hour':>5}  {'Predicted':>10}  {'Actual':>10}  {'Error%':>8}")
    print("-" * 40)
    for h, (pred, actual) in enumerate(zip(predictions, actual_prices), 1):
        if not np.isnan(actual) and actual > 0:
            error_pct = abs(pred - actual) / max(actual, 1) * 100
        else:
            error_pct = float('nan')
        print(f" {h:2d}h    {pred:8.1f}    {actual:8.1f}    {error_pct:6.1f}%")

    # I show top features for 1h and 6h horizons
    for h in [1, 6]:
        fi = forecaster.get_feature_importance(horizon=h)
        if fi is not None:
            print(f"\nTop 10 features (horizon {h}h):")
            for _, row in fi.head(10).iterrows():
                print(f"  {row['feature']:<25s} {row['pct']:5.1f}%")

    return forecaster


if __name__ == "__main__":
    train_and_save_forecaster()
