"""
ML-Based Bidding Strategy - REALISTIC Version (No Data Leakage)

I fixed the data leakage issues by:
1. Removing mfrr_spread from features (it contains the target)
2. Using only lagged prices (not current price)
3. Proper temporal train/test split within ML training
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

from .strategies import Strategy, MarketState


class RealisticMLPredictor(Strategy):
    """
    I predict mFRR prices using ONLY information available BEFORE bidding.

    Key differences from leaky version:
    - NO mfrr_spread (contains target)
    - NO current price (not known at bid time)
    - ONLY lagged prices and external features
    """

    def __init__(self, model, scaler, feature_df: pd.DataFrame,
                 feature_cols: List[str],
                 high_threshold: float = 150.0,
                 low_threshold: float = 50.0,
                 max_power_mw: float = 30.0):
        super().__init__("Realistic ML Predictor")
        self.model = model
        self.scaler = scaler
        self.feature_df = feature_df
        self.feature_cols = feature_cols
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_power_mw = max_power_mw

    def decide(self, state: MarketState) -> Tuple[str, float]:
        """
        I make decisions based on PREDICTED price, not actual price.
        """
        try:
            row = self.feature_df.loc[state.timestamp]
            features = row[self.feature_cols].values.reshape(1, -1)

            if np.any(np.isnan(features)):
                return self._fallback_decision(state)

            features_scaled = self.scaler.transform(features)
            predicted_price = self.model.predict(features_scaled)[0]

        except (KeyError, Exception):
            return self._fallback_decision(state)

        # Decision based on PREDICTED price (not actual!)
        if predicted_price > self.high_threshold and state.can_discharge:
            confidence = min((predicted_price - self.high_threshold) / 100, 1.0)
            return 'discharge', self.max_power_mw * (0.5 + 0.5 * confidence)

        elif predicted_price < self.low_threshold and state.can_charge:
            confidence = min((self.low_threshold - predicted_price) / 50, 1.0)
            return 'charge', -self.max_power_mw * (0.5 + 0.5 * confidence)

        else:
            return 'idle', 0.0

    def _fallback_decision(self, state: MarketState) -> Tuple[str, float]:
        """Fallback to time-based logic when prediction fails."""
        if 17 <= state.hour <= 21 and state.can_discharge:
            return 'discharge', self.max_power_mw * 0.5
        elif 9 <= state.hour <= 15 and state.can_charge:
            return 'charge', -self.max_power_mw * 0.5
        return 'idle', 0.0


class RealisticHybridStrategy(Strategy):
    """
    I combine Time-Based windows with REALISTIC ML predictions.

    Key difference: I use PREDICTED prices, not actual settlement prices.
    """

    def __init__(self, ml_predictor, max_power_mw: float = 30.0,
                 pred_high_threshold: float = 120.0,
                 pred_low_threshold: float = 60.0):
        super().__init__("Realistic Hybrid")
        self.ml_predictor = ml_predictor
        self.max_power_mw = max_power_mw
        self.pred_high = pred_high_threshold
        self.pred_low = pred_low_threshold

    def decide(self, state: MarketState) -> Tuple[str, float]:
        hour = state.hour
        is_weekend = state.is_weekend

        # I get ML predicted price (NOT actual price!)
        try:
            row = self.ml_predictor.feature_df.loc[state.timestamp]
            features = row[self.ml_predictor.feature_cols].values.reshape(1, -1)
            features_scaled = self.ml_predictor.scaler.transform(features)
            predicted_price = self.ml_predictor.model.predict(features_scaled)[0]
        except Exception:
            predicted_price = 100.0  # Default to neutral

        weekend_factor = 0.6 if is_weekend else 1.0

        # Evening peak (17-21h): Discharge if prediction supports it
        if 17 <= hour <= 21 and state.can_discharge:
            if predicted_price > self.pred_high:
                return 'discharge', self.max_power_mw * weekend_factor
            elif predicted_price > 80:
                return 'discharge', self.max_power_mw * 0.6 * weekend_factor
            else:
                return 'idle', 0.0

        # Morning ramp (6-8h): Moderate discharge if prediction supports
        elif 6 <= hour <= 8 and state.can_discharge:
            if predicted_price > self.pred_high:
                return 'discharge', self.max_power_mw * 0.7 * weekend_factor
            else:
                return 'idle', 0.0

        # Solar hours (10-14h): Charge if prediction supports it
        elif 10 <= hour <= 14 and state.can_charge:
            if predicted_price < self.pred_low:
                return 'charge', -self.max_power_mw * weekend_factor
            elif predicted_price < 80:
                return 'charge', -self.max_power_mw * 0.6 * weekend_factor
            else:
                return 'idle', 0.0

        # Off-peak: Only trade on strong predictions
        else:
            if predicted_price > 180 and state.can_discharge:
                return 'discharge', self.max_power_mw * 0.8 * weekend_factor
            elif predicted_price < 30 and state.can_charge:
                return 'charge', -self.max_power_mw * 0.8 * weekend_factor
            else:
                return 'idle', 0.0


class RealisticMLTrainer:
    """
    I train ML models using ONLY realistic features (no leakage).
    """

    # Features that are STRICTLY SAFE (absolutely no current price info)
    # IMPORTANT: rolling().mean() includes current row (LEAKY!)
    # We only use .shift(N) features
    SAFE_FEATURES = [
        # Time features (always known)
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'is_peak_hour', 'is_solar_hour', 'is_morning_ramp',

        # ONLY shifted prices (truly from the past)
        'price_up_lag_1h',   # shift(1) - SAFE
        'price_up_lag_4h',   # shift(4) - SAFE
        'price_up_lag_24h',  # shift(24) - SAFE

        # Lagged spread (from past only)
        'spread_lag_1h',

        # External features (known in advance or from day-ahead)
        'net_cross_border_mw',
        'hydro_level_normalized',
    ]

    # Features to EXCLUDE (cause leakage - contain current price!)
    LEAKY_FEATURES = [
        'mfrr_spread',          # Contains target!
        'mfrr_price_up',        # IS the target!
        'mfrr_price_down',      # Combined with spread = target
        'price_up_change_1h',   # = current - lag_1h (LEAKY!)
        'price_up_change_4h',   # = current - lag_4h (LEAKY!)
        'price_up_change_24h',  # = current - lag_24h (LEAKY!)
        'price_up_mean_4h',     # rolling includes current (LEAKY!)
        'price_up_mean_24h',    # rolling includes current (LEAKY!)
        'price_up_std_24h',     # rolling includes current (LEAKY!)
        'price_up_zscore_24h',  # uses current price (LEAKY!)
        'spread_mean_24h',      # rolling includes current (LEAKY!)
    ]

    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        I prepare features, ensuring no leakage.
        """
        # Add lagged spread if not present
        if 'spread_lag_1h' not in df.columns and 'mfrr_spread' in df.columns:
            df = df.copy()
            df['spread_lag_1h'] = df['mfrr_spread'].shift(1)

        # Filter to safe features only
        available = [f for f in self.SAFE_FEATURES
                     if f in df.columns and df[f].notna().mean() > 0.5]

        print(f"  Safe features available: {len(available)}")

        # Double-check no leaky features snuck in
        for feat in available:
            if feat in self.LEAKY_FEATURES:
                raise ValueError(f"Leaky feature detected: {feat}")

        return available, df

    def train_price_predictor(self, df: pd.DataFrame) -> Tuple:
        """
        I train a price predictor using ONLY safe features.
        Uses proper TEMPORAL split (not random!).
        """
        print("Training REALISTIC Price Predictor...")
        print("  (No mfrr_spread, no current price)")

        feature_cols, df = self.prepare_features(df)

        # Prepare data
        target_col = 'mfrr_price_up'
        data = df[[target_col] + feature_cols].dropna()

        print(f"  Training samples: {len(data)}")
        print(f"  Features: {len(feature_cols)}")

        X = data[feature_cols].values
        y = data[target_col].values

        # TEMPORAL split (not random!) - critical for time series
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Scale
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  MAE: {mae:.2f} EUR/MWh")
        print(f"  R2: {r2:.4f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n  Top 5 Features:")
        for _, row in importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

        return model, self.scaler, feature_cols, {'mae': mae, 'r2': r2}


def create_realistic_strategies(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 max_power_mw: float = 30.0) -> List[Strategy]:
    """
    I create ML strategies with NO data leakage.
    """
    trainer = RealisticMLTrainer()
    strategies = []

    print("\n" + "=" * 60)
    print("TRAINING REALISTIC ML MODELS (NO LEAKAGE)")
    print("=" * 60)

    # 1. Realistic Price Predictor
    try:
        model, scaler, feature_cols, metrics = trainer.train_price_predictor(train_df)

        # Prepare test_df with same features
        _, test_df_prepared = trainer.prepare_features(test_df.copy())

        strategy = RealisticMLPredictor(
            model=model,
            scaler=scaler,
            feature_df=test_df_prepared,
            feature_cols=feature_cols,
            high_threshold=140.0,  # More conservative thresholds
            low_threshold=60.0,
            max_power_mw=max_power_mw
        )
        strategy.name = f"Realistic ML (MAE={metrics['mae']:.0f}, R2={metrics['r2']:.2f})"
        strategies.append(strategy)

        # 2. Realistic Hybrid
        hybrid = RealisticHybridStrategy(
            ml_predictor=strategy,
            max_power_mw=max_power_mw,
            pred_high_threshold=120.0,
            pred_low_threshold=60.0
        )
        strategies.append(hybrid)

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    return strategies
