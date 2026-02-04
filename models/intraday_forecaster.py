"""
IntraDay Price Forecaster for Battery Trading (Fast Vectorized Version).

I predict IntraDay prices for the next 12 hours using available features.
This replaces the "cheating" lookahead that sees actual future prices.

This version uses vectorized operations for fast training on 50k+ rows.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
import pickle
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class IntraDayForecaster:
    """
    Fast ML-based IntraDay price forecaster using vectorized operations.

    I use Ridge regression for each forecast horizon (1-12h ahead).
    Training is fast due to vectorized feature creation.
    """

    def __init__(self, max_horizon: int = 12):
        self.max_horizon = max_horizon
        self.models = {}
        self.scalers = {}
        self.is_trained = False

        # Forecast error parameters (realistic calibration)
        self.base_error_1h = 0.08   # 8% at 1h
        self.base_error_12h = 0.20  # 20% at 12h

    def _create_features_vectorized(self, df: pd.DataFrame) -> np.ndarray:
        """
        I create feature matrix for all timesteps (vectorized).

        Features (22 total):
        0-5: Time (hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos)
        6-7: RES (solar, wind_onshore) - normalized
        8-9: aFRR (up_mw, down_mw) - normalized
        10-11: mFRR (up, down) - normalized
        12: Current price - normalized
        13-18: Price lags (1, 2, 3, 6, 12, 24h)
        19-21: Rolling stats (mean_6h, mean_12h, mean_24h)
        """
        n = len(df)
        features = np.zeros((n, 22))

        # Time features (0-5)
        features[:, 0] = df['hour_sin'].values
        features[:, 1] = df['hour_cos'].values
        features[:, 2] = df['day_sin'].values
        features[:, 3] = df['day_cos'].values
        features[:, 4] = df['month_sin'].values
        features[:, 5] = df['month_cos'].values

        # RES (6-7) - normalized
        features[:, 6] = df['solar'].fillna(0).values / 5000
        features[:, 7] = df['wind_onshore'].fillna(0).values / 5000

        # aFRR (8-9) - normalized
        features[:, 8] = df['afrr_up_mw'].fillna(600).values / 1000
        features[:, 9] = df['afrr_down_mw'].fillna(130).values / 500

        # mFRR (10-11) - normalized
        features[:, 10] = df['mfrr_up'].fillna(100).values / 200
        features[:, 11] = df['mfrr_down'].fillna(80).values / 200

        # Current price (12)
        price = df['price'].fillna(100).values
        features[:, 12] = price / 200

        # Price lags (13-18) - shift and fill with method='bfill'
        lags = [1, 2, 3, 6, 12, 24]
        price_series = pd.Series(price)
        for i, lag in enumerate(lags):
            shifted = price_series.shift(lag).bfill().values
            features[:, 13 + i] = shifted / 200

        # Rolling means (19-21)
        price_series = pd.Series(price)
        features[:, 19] = price_series.rolling(6, min_periods=1).mean().values / 200
        features[:, 20] = price_series.rolling(12, min_periods=1).mean().values / 200
        features[:, 21] = price_series.rolling(24, min_periods=1).mean().values / 200

        return features

    def train(self, df: pd.DataFrame, verbose: bool = True):
        """
        I train 12 Ridge models (one per horizon) using vectorized operations.
        Much faster than iterating row by row.
        """
        # I lowercase columns for consistency
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if verbose:
            print("Training IntraDay Forecaster (vectorized)...")
            print(f"  Data: {len(df)} hours")

        # I create all features at once
        if verbose:
            print("  Creating features...")
        X_all = self._create_features_vectorized(df)
        y_all = df['price'].fillna(100).values

        # I need at least 24h of history
        valid_start = 24
        X_all = X_all[valid_start:]
        y_all = y_all[valid_start:]

        for horizon in range(1, self.max_horizon + 1):
            # Target is price at t+horizon
            if horizon >= len(y_all):
                continue

            # I shift target by horizon
            X = X_all[:-horizon]
            y = y_all[horizon:]

            if len(X) < 100:
                if verbose:
                    print(f"  Horizon {horizon}h: Not enough data")
                continue

            # I scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # I train Ridge model (fast)
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)

            self.models[horizon] = model
            self.scalers[horizon] = scaler

            if verbose:
                # Quick validation on last 10%
                split = int(len(X) * 0.9)
                y_pred = model.predict(X_scaled[split:])
                y_val = y[split:]
                mae = np.mean(np.abs(y_val - y_pred))
                mape = np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1))) * 100
                print(f"  Horizon {horizon:2d}h: MAE={mae:5.1f} EUR, MAPE={mape:4.1f}%")

        self.is_trained = True
        if verbose:
            print("Training complete!")

    def predict(
        self,
        df: pd.DataFrame,
        idx: int,
        add_noise: bool = True
    ) -> List[float]:
        """
        I predict prices for the next 12 hours from position idx.
        """
        # I lowercase columns
        df_lower = df.copy()
        df_lower.columns = [c.lower() for c in df_lower.columns]

        current_price = df_lower.iloc[idx].get('price', 100)

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
                scaler = self.scalers[horizon]

                X_scaled = scaler.transform(features.reshape(1, -1))
                pred = model.predict(X_scaled)[0]

                # I add realistic noise
                if add_noise:
                    error_pct = self.base_error_1h + \
                        (self.base_error_12h - self.base_error_1h) * (horizon - 1) / 11
                    noise = np.random.normal(0, abs(pred) * error_pct)
                    pred = pred + noise

                predictions.append(max(0.1, pred))
            else:
                predictions.append(current_price)

        return predictions

    def _create_single_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """I create feature vector for a single timestep."""
        row = df.iloc[idx]
        price_col = df['price'].values
        current_price = row.get('price', 100)

        features = np.zeros(22)

        # Time features (0-5)
        features[0] = row.get('hour_sin', 0)
        features[1] = row.get('hour_cos', 1)
        features[2] = row.get('day_sin', 0)
        features[3] = row.get('day_cos', 1)
        features[4] = row.get('month_sin', 0)
        features[5] = row.get('month_cos', 1)

        # RES (6-7)
        features[6] = row.get('solar', 0) / 5000
        features[7] = row.get('wind_onshore', 0) / 5000

        # aFRR (8-9)
        features[8] = row.get('afrr_up_mw', 600) / 1000
        features[9] = row.get('afrr_down_mw', 130) / 500

        # mFRR (10-11)
        features[10] = row.get('mfrr_up', 100) / 200
        features[11] = row.get('mfrr_down', 80) / 200

        # Current price (12)
        features[12] = current_price / 200

        # Price lags (13-18)
        lags = [1, 2, 3, 6, 12, 24]
        for i, lag in enumerate(lags):
            if idx >= lag:
                features[13 + i] = price_col[idx - lag] / 200
            else:
                features[13 + i] = current_price / 200

        # Rolling means (19-21)
        for i, window in enumerate([6, 12, 24]):
            start = max(0, idx - window + 1)
            features[19 + i] = np.mean(price_col[start:idx + 1]) / 200

        return features

    def save(self, path: str):
        """I save the trained forecaster."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'max_horizon': self.max_horizon,
                'base_error_1h': self.base_error_1h,
                'base_error_12h': self.base_error_12h,
                'is_trained': self.is_trained
            }, f)
        print(f"Forecaster saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'IntraDayForecaster':
        """I load a trained forecaster."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(max_horizon=data['max_horizon'])
        forecaster.models = data['models']
        forecaster.scalers = data['scalers']
        forecaster.base_error_1h = data['base_error_1h']
        forecaster.base_error_12h = data['base_error_12h']
        forecaster.is_trained = data['is_trained']

        return forecaster


def train_and_save_forecaster(
    data_path: str = "data/feasible_data_with_balancing.csv",
    model_path: str = "models/intraday_forecaster.pkl"
):
    """I train and save the forecaster."""
    print("="*60)
    print("TRAINING INTRADAY PRICE FORECASTER")
    print("="*60)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    forecaster = IntraDayForecaster(max_horizon=12)
    forecaster.train(df, verbose=True)
    forecaster.save(model_path)

    # Test
    print("\nTest prediction at idx=40000:")
    predictions = forecaster.predict(df, idx=40000, add_noise=True)
    actual_prices = [df.iloc[40000 + h].get('price', 0) for h in range(1, 13)]

    print("Hour  Predicted  Actual   Error%")
    print("-"*40)
    for h, (pred, actual) in enumerate(zip(predictions, actual_prices), 1):
        error_pct = abs(pred - actual) / max(actual, 1) * 100
        print(f" {h:2d}h    {pred:6.1f}    {actual:6.1f}    {error_pct:5.1f}%")

    return forecaster


if __name__ == "__main__":
    train_and_save_forecaster()
