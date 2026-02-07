"""
ML-Based Bidding Strategy v2 - Fixed Feature Handling

I fixed the feature mismatch issue by:
1. Storing feature data from the DataFrame directly
2. Looking up features by timestamp during backtesting
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score

from .strategies import Strategy, MarketState


class MLPricePredictorV2(Strategy):
    """
    I predict mFRR prices using ML and make decisions.

    Fixed version: I store the full feature DataFrame and look up by timestamp.
    """

    def __init__(self, model, scaler, feature_df: pd.DataFrame,
                 feature_cols: List[str],
                 high_threshold: float = 150.0,
                 low_threshold: float = 30.0,
                 max_power_mw: float = 30.0):
        super().__init__("ML Price Predictor v2")
        self.model = model
        self.scaler = scaler
        self.feature_df = feature_df
        self.feature_cols = feature_cols
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_power_mw = max_power_mw

    def decide(self, state: MarketState) -> Tuple[str, float]:
        # Look up features by timestamp
        try:
            row = self.feature_df.loc[state.timestamp]
            features = row[self.feature_cols].values.reshape(1, -1)

            if np.any(np.isnan(features)):
                return 'idle', 0.0

            features_scaled = self.scaler.transform(features)
            predicted_price = self.model.predict(features_scaled)[0]

        except (KeyError, Exception):
            # Fallback to time-based if no data
            return self._fallback_decision(state)

        # Decision based on prediction
        if predicted_price > self.high_threshold and state.can_discharge:
            confidence = min((predicted_price - self.high_threshold) / 100, 1.0)
            return 'discharge', self.max_power_mw * (0.6 + 0.4 * confidence)

        elif predicted_price < self.low_threshold and state.can_charge:
            confidence = min((self.low_threshold - predicted_price) / 50, 1.0)
            return 'charge', -self.max_power_mw * (0.6 + 0.4 * confidence)

        else:
            return 'idle', 0.0

    def _fallback_decision(self, state: MarketState) -> Tuple[str, float]:
        """Fallback to simple time-based logic."""
        if 17 <= state.hour <= 21 and state.can_discharge:
            return 'discharge', self.max_power_mw * 0.5
        elif 9 <= state.hour <= 15 and state.can_charge:
            return 'charge', -self.max_power_mw * 0.5
        return 'idle', 0.0


class MLActionClassifierV2(Strategy):
    """
    I classify optimal actions using ML.

    Fixed version with proper feature lookup.
    """

    def __init__(self, model, scaler, feature_df: pd.DataFrame,
                 feature_cols: List[str],
                 max_power_mw: float = 30.0,
                 confidence_threshold: float = 0.55):
        super().__init__("ML Action Classifier v2")
        self.model = model
        self.scaler = scaler
        self.feature_df = feature_df
        self.feature_cols = feature_cols
        self.max_power_mw = max_power_mw
        self.confidence_threshold = confidence_threshold
        self.actions = ['idle', 'discharge', 'charge']

    def decide(self, state: MarketState) -> Tuple[str, float]:
        try:
            row = self.feature_df.loc[state.timestamp]
            features = row[self.feature_cols].values.reshape(1, -1)

            if np.any(np.isnan(features)):
                return self._fallback_decision(state)

            features_scaled = self.scaler.transform(features)
            probs = self.model.predict_proba(features_scaled)[0]
            action_idx = np.argmax(probs)
            confidence = probs[action_idx]

        except (KeyError, Exception):
            return self._fallback_decision(state)

        # Only act if confident enough
        if confidence < self.confidence_threshold:
            return 'idle', 0.0

        action = self.actions[action_idx]

        # Check constraints
        if action == 'discharge' and not state.can_discharge:
            return 'idle', 0.0
        if action == 'charge' and not state.can_charge:
            return 'idle', 0.0

        # Scale power by confidence
        power_scale = 0.5 + 0.5 * confidence
        power = self.max_power_mw * power_scale

        if action == 'discharge':
            return 'discharge', power
        elif action == 'charge':
            return 'charge', -power
        else:
            return 'idle', 0.0

    def _fallback_decision(self, state: MarketState) -> Tuple[str, float]:
        """Fallback to time-based logic."""
        if 17 <= state.hour <= 21 and state.can_discharge:
            return 'discharge', self.max_power_mw * 0.5
        elif 9 <= state.hour <= 15 and state.can_charge:
            return 'charge', -self.max_power_mw * 0.5
        return 'idle', 0.0


class MLEnsembleStrategy(Strategy):
    """
    I combine multiple signals for robust decisions:
    1. ML price prediction
    2. Time-of-day patterns
    3. Price momentum
    4. Spread analysis
    """

    def __init__(self, price_model, scaler, feature_df: pd.DataFrame,
                 feature_cols: List[str], max_power_mw: float = 30.0):
        super().__init__("ML Ensemble")
        self.price_model = price_model
        self.scaler = scaler
        self.feature_df = feature_df
        self.feature_cols = feature_cols
        self.max_power_mw = max_power_mw

    def decide(self, state: MarketState) -> Tuple[str, float]:
        score = 0.0  # Positive = discharge, Negative = charge

        # 1. ML Price Prediction Signal
        ml_score = self._get_ml_score(state)
        score += ml_score * 2.0  # Weight: 2x

        # 2. Time-of-day Signal
        time_score = self._get_time_score(state)
        score += time_score * 1.5  # Weight: 1.5x

        # 3. Spread Signal (if available)
        spread_score = self._get_spread_score(state)
        score += spread_score * 1.0

        # 4. SoC Management
        soc_score = self._get_soc_score(state)
        score += soc_score * 0.5

        # Weekend adjustment
        if state.is_weekend:
            score *= 0.7

        # Decision
        if score >= 2.5 and state.can_discharge:
            power_scale = min(score / 5.0, 1.0)
            return 'discharge', self.max_power_mw * power_scale
        elif score <= -2.5 and state.can_charge:
            power_scale = min(abs(score) / 5.0, 1.0)
            return 'charge', -self.max_power_mw * power_scale
        else:
            return 'idle', 0.0

    def _get_ml_score(self, state: MarketState) -> float:
        """Get score from ML price prediction."""
        try:
            row = self.feature_df.loc[state.timestamp]
            features = row[self.feature_cols].values.reshape(1, -1)

            if np.any(np.isnan(features)):
                return 0.0

            features_scaled = self.scaler.transform(features)
            pred_price = self.price_model.predict(features_scaled)[0]

            if pred_price > 200:
                return 2.0
            elif pred_price > 150:
                return 1.0
            elif pred_price < 0:
                return -2.0
            elif pred_price < 30:
                return -1.0
            else:
                return 0.0

        except Exception:
            return 0.0

    def _get_time_score(self, state: MarketState) -> float:
        """Get score from time-of-day patterns."""
        hour = state.hour

        if 17 <= hour <= 21:  # Evening peak
            return 1.5
        elif 5 <= hour <= 8:  # Morning ramp
            return 0.8
        elif 9 <= hour <= 15:  # Solar hours
            return -1.5
        elif 22 <= hour or hour <= 4:  # Night
            return -0.3
        else:
            return 0.0

    def _get_spread_score(self, state: MarketState) -> float:
        """Get score from mFRR spread."""
        try:
            row = self.feature_df.loc[state.timestamp]
            spread = row.get('mfrr_spread', 0)

            if pd.isna(spread):
                return 0.0

            if spread > 100:
                return 1.5
            elif spread > 50:
                return 1.0
            elif spread < -50:
                return -1.0
            elif spread < -100:
                return -1.5
            else:
                return 0.0
        except Exception:
            return 0.0

    def _get_soc_score(self, state: MarketState) -> float:
        """Adjust based on battery SoC."""
        if state.soc > 0.85:
            return 1.0  # High SoC, prefer discharge
        elif state.soc < 0.25:
            return -1.0  # Low SoC, prefer charge
        else:
            return 0.0


class MLTrainerV2:
    """
    Train ML models with proper feature handling.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def train_price_predictor(self, df: pd.DataFrame) -> Tuple:
        """Train price prediction model."""
        print("Training Price Predictor Model...")

        # Select features that have good coverage
        feature_cols = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_peak_hour', 'is_solar_hour', 'is_morning_ramp',
            'price_up_lag_1h', 'price_up_lag_4h', 'price_up_lag_24h',
            'price_up_mean_4h', 'price_up_mean_24h',
            'price_up_change_1h',
            'mfrr_spread', 'spread_mean_24h',
            'net_cross_border_mw',
            'hydro_level_normalized',
        ]

        # Filter to available columns with good coverage
        available_cols = []
        for col in feature_cols:
            if col in df.columns:
                coverage = df[col].notna().mean()
                if coverage > 0.7:
                    available_cols.append(col)

        feature_cols = available_cols
        print(f"  Using {len(feature_cols)} features")

        # Prepare data
        target_col = 'mfrr_price_up'
        data = df[[target_col] + feature_cols].dropna()

        print(f"  Training samples: {len(data)}")

        X = data[feature_cols].values
        y = data[target_col].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  MAE: {mae:.2f} EUR/MWh")
        print(f"  R²: {r2:.3f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n  Top 5 Features:")
        for _, row in importance.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")

        return model, self.scaler, feature_cols, {'mae': mae, 'r2': r2}

    def train_action_classifier(self, df: pd.DataFrame,
                                  high_thresh: float = 150,
                                  low_thresh: float = 30) -> Tuple:
        """Train action classification model."""
        print("\nTraining Action Classifier Model...")

        # Create labels
        df = df.copy()
        df['action_label'] = 0
        df.loc[df['mfrr_price_up'] > high_thresh, 'action_label'] = 1
        df.loc[df['mfrr_price_up'] < low_thresh, 'action_label'] = 2

        # Features
        feature_cols = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_peak_hour', 'is_solar_hour',
            'price_up_lag_1h', 'price_up_mean_24h',
            'price_up_change_1h',
            'mfrr_spread',
            'hydro_level_normalized',
        ]

        available_cols = [c for c in feature_cols if c in df.columns and df[c].notna().mean() > 0.7]
        feature_cols = available_cols

        print(f"  Using {len(feature_cols)} features")

        # Prepare data
        data = df[['action_label'] + feature_cols].dropna()
        print(f"  Training samples: {len(data)}")

        X = data[feature_cols].values
        y = data['action_label'].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        print(f"  Accuracy: {acc:.3f}")
        print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return model, scaler, feature_cols, {'accuracy': acc}


def create_ml_strategies_v2(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             max_power_mw: float = 30.0) -> List[Strategy]:
    """
    Train ML models and create strategy instances.

    Args:
        train_df: Training data
        test_df: Test data (for feature lookup during backtest)
    """
    trainer = MLTrainerV2()
    strategies = []

    print("\n" + "=" * 60)
    print("TRAINING ML MODELS (v2)")
    print("=" * 60)

    # 1. Price Predictor
    try:
        model, scaler, feature_cols, metrics = trainer.train_price_predictor(train_df)

        strategy = MLPricePredictorV2(
            model=model,
            scaler=scaler,
            feature_df=test_df,
            feature_cols=feature_cols,
            high_threshold=150.0,
            low_threshold=30.0,
            max_power_mw=max_power_mw
        )
        strategy.name = f"ML Price Predictor (MAE={metrics['mae']:.0f})"
        strategies.append(strategy)

    except Exception as e:
        print(f"  Price Predictor Error: {e}")

    # 2. Action Classifier
    try:
        model, scaler, feature_cols, metrics = trainer.train_action_classifier(train_df)

        strategy = MLActionClassifierV2(
            model=model,
            scaler=scaler,
            feature_df=test_df,
            feature_cols=feature_cols,
            max_power_mw=max_power_mw,
            confidence_threshold=0.55
        )
        strategy.name = f"ML Action Classifier (Acc={metrics['accuracy']:.2f})"
        strategies.append(strategy)

    except Exception as e:
        print(f"  Action Classifier Error: {e}")

    # 3. Ensemble Strategy
    try:
        # Reuse price predictor model
        model, scaler, feature_cols, _ = trainer.train_price_predictor(train_df)

        strategy = MLEnsembleStrategy(
            price_model=model,
            scaler=scaler,
            feature_df=test_df,
            feature_cols=feature_cols,
            max_power_mw=max_power_mw
        )
        strategy.name = "ML Ensemble"
        strategies.append(strategy)

    except Exception as e:
        print(f"  Ensemble Error: {e}")

    return strategies
