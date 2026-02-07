"""
ML-Based Bidding Strategy for mFRR Market

I train machine learning models to predict optimal bidding actions
based on the features we engineered.

Approaches:
1. PricePredictorStrategy: Predict future prices, bid when favorable
2. ActionClassifierStrategy: Directly classify optimal action
3. ReinforcementStrategy: Use Q-learning for action selection
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score

from .strategies import Strategy, MarketState


@dataclass
class MLModelConfig:
    """Configuration for ML models."""
    model_type: str = 'random_forest'  # 'random_forest', 'gradient_boosting'
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_leaf: int = 10
    test_size: float = 0.2
    random_state: int = 42


class PricePredictorStrategy(Strategy):
    """
    I predict future mFRR Up prices and make decisions based on predictions.

    Logic:
    - If predicted price > high_threshold: discharge
    - If predicted price < low_threshold: charge
    - Otherwise: idle

    The model predicts price change rather than absolute price
    to handle non-stationarity.
    """

    def __init__(self, model=None, scaler=None, feature_cols: List[str] = None,
                 high_threshold: float = 150.0, low_threshold: float = 30.0,
                 max_power_mw: float = 30.0):
        super().__init__("ML Price Predictor")
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols or []
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_power_mw = max_power_mw

        # For online feature building
        self._last_features = None

    def decide(self, state: MarketState) -> Tuple[str, float]:
        if self.model is None:
            return 'idle', 0.0

        # Build feature vector from state
        features = self._build_features(state)

        if features is None:
            return 'idle', 0.0

        # Predict
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            predicted_price = self.model.predict(features_scaled)[0]
        except Exception:
            return 'idle', 0.0

        # Make decision
        if predicted_price > self.high_threshold and state.can_discharge:
            # Scale power by confidence (distance from threshold)
            confidence = min((predicted_price - self.high_threshold) / 100, 1.0)
            return 'discharge', self.max_power_mw * (0.5 + 0.5 * confidence)

        elif predicted_price < self.low_threshold and state.can_charge:
            confidence = min((self.low_threshold - predicted_price) / 50, 1.0)
            return 'charge', -self.max_power_mw * (0.5 + 0.5 * confidence)

        else:
            return 'idle', 0.0

    def _build_features(self, state: MarketState) -> Optional[np.ndarray]:
        """Build feature vector from market state."""
        features = []

        for col in self.feature_cols:
            val = getattr(state, col, None)
            if val is None:
                # Try to get from common mappings
                val = self._get_feature_value(state, col)

            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None

            features.append(val)

        return np.array(features)

    def _get_feature_value(self, state: MarketState, col: str) -> Optional[float]:
        """Get feature value with fallbacks."""
        mappings = {
            'hour': state.hour,
            'hour_sin': np.sin(2 * np.pi * state.hour / 24),
            'hour_cos': np.cos(2 * np.pi * state.hour / 24),
            'dow_sin': np.sin(2 * np.pi * state.day_of_week / 7),
            'dow_cos': np.cos(2 * np.pi * state.day_of_week / 7),
            'is_weekend': 1 if state.is_weekend else 0,
            'is_peak_hour': 1 if 17 <= state.hour <= 21 else 0,
            'is_solar_hour': 1 if 9 <= state.hour <= 15 else 0,
            'soc': state.soc,
        }
        return mappings.get(col)


class ActionClassifierStrategy(Strategy):
    """
    I directly classify the optimal action (discharge/charge/idle)
    based on features.

    The model is trained on labeled data where labels are determined
    by realized profits.
    """

    def __init__(self, model=None, scaler=None, feature_cols: List[str] = None,
                 max_power_mw: float = 30.0, confidence_threshold: float = 0.6):
        super().__init__("ML Action Classifier")
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols or []
        self.max_power_mw = max_power_mw
        self.confidence_threshold = confidence_threshold

        # Action mapping
        self.actions = ['idle', 'discharge', 'charge']

    def decide(self, state: MarketState) -> Tuple[str, float]:
        if self.model is None:
            return 'idle', 0.0

        # Build feature vector
        features = self._build_features(state)

        if features is None:
            return 'idle', 0.0

        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Get probabilities
            probs = self.model.predict_proba(features_scaled)[0]
            action_idx = np.argmax(probs)
            confidence = probs[action_idx]

            # Only act if confident
            if confidence < self.confidence_threshold:
                return 'idle', 0.0

            action = self.actions[action_idx]

            # Check constraints
            if action == 'discharge' and not state.can_discharge:
                return 'idle', 0.0
            if action == 'charge' and not state.can_charge:
                return 'idle', 0.0

            # Scale power by confidence
            power_scale = (confidence - self.confidence_threshold) / (1 - self.confidence_threshold)
            power = self.max_power_mw * (0.5 + 0.5 * power_scale)

            if action == 'discharge':
                return 'discharge', power
            elif action == 'charge':
                return 'charge', -power
            else:
                return 'idle', 0.0

        except Exception:
            return 'idle', 0.0

    def _build_features(self, state: MarketState) -> Optional[np.ndarray]:
        """Build feature vector from market state."""
        features = []

        for col in self.feature_cols:
            val = getattr(state, col, None)
            if val is None:
                val = self._get_feature_value(state, col)

            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None

            features.append(val)

        return np.array(features)

    def _get_feature_value(self, state: MarketState, col: str) -> Optional[float]:
        """Get feature value with fallbacks."""
        mappings = {
            'hour': state.hour,
            'hour_sin': np.sin(2 * np.pi * state.hour / 24),
            'hour_cos': np.cos(2 * np.pi * state.hour / 24),
            'dow_sin': np.sin(2 * np.pi * state.day_of_week / 7),
            'dow_cos': np.cos(2 * np.pi * state.day_of_week / 7),
            'is_weekend': 1 if state.is_weekend else 0,
            'is_peak_hour': 1 if 17 <= state.hour <= 21 else 0,
            'is_solar_hour': 1 if 9 <= state.hour <= 15 else 0,
            'soc': state.soc,
        }
        return mappings.get(col)


class QLearningStrategy(Strategy):
    """
    I use a simple Q-learning approach with discretized states.

    State space: (hour_bucket, soc_bucket, price_bucket)
    Action space: [idle, discharge, charge]
    """

    def __init__(self, q_table: dict = None, max_power_mw: float = 30.0,
                 epsilon: float = 0.1):
        super().__init__("Q-Learning")
        self.q_table = q_table or {}
        self.max_power_mw = max_power_mw
        self.epsilon = epsilon

        self.actions = ['idle', 'discharge', 'charge']

    def _discretize_state(self, state: MarketState) -> Tuple:
        """Discretize continuous state to buckets."""
        # Hour buckets: 0-5, 6-11, 12-17, 18-23
        hour_bucket = state.hour // 6

        # SoC buckets: 0-0.33, 0.33-0.66, 0.66-1.0
        soc_bucket = int(state.soc * 3) if state.soc < 1 else 2

        # Price bucket based on current or lagged price
        price = state.mfrr_price_up or state.price_up_lag_1h or 100
        if np.isnan(price):
            price = 100

        if price < 50:
            price_bucket = 0
        elif price < 100:
            price_bucket = 1
        elif price < 150:
            price_bucket = 2
        elif price < 200:
            price_bucket = 3
        else:
            price_bucket = 4

        # Weekend indicator
        weekend = 1 if state.is_weekend else 0

        return (hour_bucket, soc_bucket, price_bucket, weekend)

    def decide(self, state: MarketState) -> Tuple[str, float]:
        discrete_state = self._discretize_state(state)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(3)
        else:
            # Get Q-values for this state
            q_values = self.q_table.get(discrete_state, [0, 0, 0])
            action_idx = np.argmax(q_values)

        action = self.actions[action_idx]

        # Check constraints
        if action == 'discharge' and not state.can_discharge:
            action = 'idle'
        if action == 'charge' and not state.can_charge:
            action = 'idle'

        if action == 'discharge':
            return 'discharge', self.max_power_mw
        elif action == 'charge':
            return 'charge', -self.max_power_mw
        else:
            return 'idle', 0.0


class MLTrainer:
    """
    I train ML models for mFRR bidding strategies.
    """

    def __init__(self, config: MLModelConfig = None):
        self.config = config or MLModelConfig()
        self.scaler = StandardScaler()

    def train_price_predictor(self, df: pd.DataFrame,
                               target_col: str = 'mfrr_price_up') -> Tuple:
        """
        Train a model to predict mFRR prices.

        Returns:
            Tuple of (model, scaler, feature_cols, metrics)
        """
        print("Training Price Predictor Model...")

        # Select features
        feature_cols = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_peak_hour', 'is_solar_hour',
            'price_up_lag_1h', 'price_up_lag_4h', 'price_up_lag_24h',
            'price_up_mean_24h', 'price_up_std_24h',
            'mfrr_spread', 'spread_mean_24h',
            'net_cross_border_mw', 'hydro_level_normalized',
        ]

        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]

        # Prepare data
        data = df[[target_col] + feature_cols].dropna()

        if len(data) < 100:
            raise ValueError("Not enough data for training")

        X = data[feature_cols].values
        y = data[target_col].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if self.config.model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state
            )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        print(f"  MAE: {metrics['mae']:.2f} EUR/MWh")
        print(f"  R²: {metrics['r2']:.3f}")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\n  Top Features:")
            for _, row in importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.3f}")

        return model, self.scaler, feature_cols, metrics

    def train_action_classifier(self, df: pd.DataFrame,
                                 price_col: str = 'mfrr_price_up',
                                 high_threshold: float = 150,
                                 low_threshold: float = 30) -> Tuple:
        """
        Train a classifier to predict optimal action.

        Labels are created based on price thresholds:
        - price > high_threshold: discharge (1)
        - price < low_threshold: charge (2)
        - otherwise: idle (0)
        """
        print("Training Action Classifier Model...")

        # Create labels
        df = df.copy()
        df['action_label'] = 0  # idle
        df.loc[df[price_col] > high_threshold, 'action_label'] = 1  # discharge
        df.loc[df[price_col] < low_threshold, 'action_label'] = 2  # charge

        # Select features
        feature_cols = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'is_peak_hour', 'is_solar_hour', 'is_morning_ramp',
            'price_up_lag_1h', 'price_up_lag_4h',
            'price_up_mean_24h', 'price_up_change_1h',
            'spread_mean_24h',
            'net_cross_border_mw', 'is_net_importer',
            'hydro_level_normalized',
        ]

        feature_cols = [c for c in feature_cols if c in df.columns]

        # Prepare data
        data = df[['action_label'] + feature_cols].dropna()

        X = data[feature_cols].values
        y = data['action_label'].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1
        )

        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
        }

        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Class distribution: {metrics['class_distribution']}")

        return model, scaler, feature_cols, metrics

    def train_q_learning(self, df: pd.DataFrame, episodes: int = 10,
                          learning_rate: float = 0.1,
                          discount_factor: float = 0.95,
                          price_col: str = 'mfrr_price_up') -> dict:
        """
        Train Q-table using historical data.

        This is offline Q-learning where we use realized prices
        as rewards.
        """
        print("Training Q-Learning Model...")

        q_table = {}

        # Discretization parameters
        def discretize(row):
            hour_bucket = row.name.hour // 6 if hasattr(row.name, 'hour') else 0

            # Estimate SoC bucket (assume we don't know true SoC)
            soc_bucket = 1  # middle bucket

            price = row.get(price_col, 100)
            if np.isnan(price):
                price = 100

            if price < 50:
                price_bucket = 0
            elif price < 100:
                price_bucket = 1
            elif price < 150:
                price_bucket = 2
            elif price < 200:
                price_bucket = 3
            else:
                price_bucket = 4

            weekend = 1 if row.name.dayofweek >= 5 else 0

            return (hour_bucket, soc_bucket, price_bucket, weekend)

        # Actions: 0=idle, 1=discharge, 2=charge
        n_actions = 3

        for episode in range(episodes):
            # Shuffle data for each episode
            data = df.sample(frac=1, random_state=episode)

            for idx, (timestamp, row) in enumerate(data.iterrows()):
                state = discretize(row)

                # Get current Q-values
                if state not in q_table:
                    q_table[state] = [0.0, 0.0, 0.0]

                # Calculate rewards for each action
                price_up = row.get(price_col, 0)
                price_down = row.get('mfrr_price_down', 0)

                if np.isnan(price_up):
                    price_up = 0
                if np.isnan(price_down):
                    price_down = 0

                degradation = 15.0  # EUR/MWh
                power = 30.0

                rewards = [
                    0,  # idle
                    (price_up - degradation) * power,  # discharge
                    (price_down - degradation) * power,  # charge (down-reg payment)
                ]

                # Q-learning update
                for action in range(n_actions):
                    old_q = q_table[state][action]
                    new_q = old_q + learning_rate * (rewards[action] - old_q)
                    q_table[state][action] = new_q

            if (episode + 1) % 2 == 0:
                print(f"  Episode {episode + 1}/{episodes} complete")

        print(f"  Q-table size: {len(q_table)} states")

        return q_table

    def save_models(self, models_dir: str, **models):
        """Save trained models to disk."""
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)

        for name, model_data in models.items():
            filepath = models_dir / f"{name}.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Saved {name} to {filepath}")

    def load_models(self, models_dir: str) -> dict:
        """Load trained models from disk."""
        models_dir = Path(models_dir)
        models = {}

        for filepath in models_dir.glob("*.pkl"):
            with open(filepath, 'rb') as f:
                models[filepath.stem] = pickle.load(f)
            print(f"Loaded {filepath.stem}")

        return models


def create_ml_strategies(df: pd.DataFrame,
                          max_power_mw: float = 30.0) -> List[Strategy]:
    """
    Train all ML models and return strategy instances.
    """
    trainer = MLTrainer()
    strategies = []

    # 1. Price Predictor Strategy
    print("\n" + "=" * 60)
    print("1. PRICE PREDICTOR STRATEGY")
    print("=" * 60)

    try:
        model, scaler, feature_cols, metrics = trainer.train_price_predictor(df)

        strategy = PricePredictorStrategy(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            high_threshold=150.0,
            low_threshold=30.0,
            max_power_mw=max_power_mw
        )
        strategy.name = f"ML Price Predictor (MAE={metrics['mae']:.0f})"
        strategies.append(strategy)

    except Exception as e:
        print(f"  Error: {e}")

    # 2. Action Classifier Strategy
    print("\n" + "=" * 60)
    print("2. ACTION CLASSIFIER STRATEGY")
    print("=" * 60)

    try:
        model, scaler, feature_cols, metrics = trainer.train_action_classifier(df)

        strategy = ActionClassifierStrategy(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            max_power_mw=max_power_mw,
            confidence_threshold=0.6
        )
        strategy.name = f"ML Action Classifier (Acc={metrics['accuracy']:.2f})"
        strategies.append(strategy)

    except Exception as e:
        print(f"  Error: {e}")

    # 3. Q-Learning Strategy
    print("\n" + "=" * 60)
    print("3. Q-LEARNING STRATEGY")
    print("=" * 60)

    try:
        q_table = trainer.train_q_learning(df, episodes=10)

        strategy = QLearningStrategy(
            q_table=q_table,
            max_power_mw=max_power_mw,
            epsilon=0.0  # No exploration during backtest
        )
        strategy.name = f"Q-Learning ({len(q_table)} states)"
        strategies.append(strategy)

    except Exception as e:
        print(f"  Error: {e}")

    return strategies
