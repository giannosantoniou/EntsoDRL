"""Abstract base class for all forecasters.

I define the contract that every forecaster must follow:
- build_features(): construct feature vector from merged data
- fit(): train on historical data
- predict(): generate predictions for a given timestamp
- save()/load(): persist and restore trained models
"""

import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base for all LightGBM forecasters.

    I enforce a consistent pattern: build_features extracts a fixed-size
    feature vector, fit trains one or more LightGBM models, predict returns
    forecasts, and save/load handle persistence.
    """

    def __init__(self, name: str, n_features: int):
        self.name = name
        self.n_features = n_features
        self.models: Dict[str, Any] = {}
        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.train_metrics: Dict[str, float] = {}

    @abstractmethod
    def build_features(self, df: pd.DataFrame, idx: int, **kwargs) -> np.ndarray:
        """Build feature vector for a single sample.

        Args:
            df: Merged training DataFrame with DatetimeIndex.
            idx: Integer position index into df.
            **kwargs: Forecaster-specific arguments (e.g., hour, ida_number).

        Returns:
            1D numpy array of shape (n_features,).
        """
        ...

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        val_end_idx: int,
    ) -> "BaseForecaster":
        """Train all models on the given data.

        Args:
            df: Merged training DataFrame.
            train_end_idx: Last index (exclusive) of training split.
            val_end_idx: Last index (exclusive) of validation split.

        Returns:
            self (for chaining).
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, idx: int, **kwargs) -> Dict[str, np.ndarray]:
        """Generate predictions for a given position.

        Args:
            df: Merged DataFrame (must contain features up to idx).
            idx: Position to predict from.
            **kwargs: Forecaster-specific arguments.

        Returns:
            Dict mapping prediction keys to arrays.
            E.g., {"point": [24 values], "p10": [24 values], "p90": [24 values]}
        """
        ...

    def build_dataset(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and target vector y for training.

        I iterate over the data range, calling build_features for each valid
        sample. Subclasses can override to customize target extraction.

        Args:
            df: Merged DataFrame.
            start_idx: First index (inclusive).
            end_idx: Last index (exclusive).
            **kwargs: Passed to build_features.

        Returns:
            Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,).
        """
        X_list = []
        y_list = []

        for idx in range(start_idx, end_idx):
            try:
                features = self.build_features(df, idx, **kwargs)
                target = self._extract_target(df, idx, **kwargs)

                if features is not None and target is not None:
                    if not np.any(np.isnan(features)):
                        X_list.append(features)
                        y_list.append(target)
            except (IndexError, KeyError):
                continue

        if not X_list:
            return np.empty((0, self.n_features)), np.empty(0)

        return np.array(X_list), np.array(y_list)

    @abstractmethod
    def _extract_target(self, df: pd.DataFrame, idx: int, **kwargs) -> Optional[float]:
        """Extract the target value for a single sample.

        Args:
            df: Merged DataFrame.
            idx: Position in DataFrame.
            **kwargs: Forecaster-specific arguments.

        Returns:
            Target value, or None if unavailable.
        """
        ...

    def save(self, path: Path) -> None:
        """Persist trained models and metadata to disk.

        I save everything needed to restore the forecaster: models, feature
        names, training metrics, and fitted flag.

        Args:
            path: File path (typically .pkl).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "name": self.name,
            "n_features": self.n_features,
            "models": self.models,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "train_metrics": self.train_metrics,
            "saved_at": datetime.now().isoformat(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"[{self.name}] Saved to {path}")

    def load(self, path: Path) -> "BaseForecaster":
        """Load trained models from disk.

        Args:
            path: File path to load from.

        Returns:
            self (for chaining).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.models = state["models"]
        self.is_fitted = state["is_fitted"]
        self.feature_names = state.get("feature_names", [])
        self.train_metrics = state.get("train_metrics", {})

        logger.info(f"[{self.name}] Loaded from {path}")
        return self

    def evaluate(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        **kwargs,
    ) -> Dict[str, float]:
        """Evaluate forecaster on a data range.

        I compute MAE, RMSE, MAPE, and bias on the given range.

        Args:
            df: Merged DataFrame.
            start_idx: First index (inclusive).
            end_idx: Last index (exclusive).
            **kwargs: Passed to predict and _extract_target.

        Returns:
            Dict with metric names → values.
        """
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] Not fitted yet.")

        errors = []
        abs_errors = []
        pct_errors = []

        for idx in range(start_idx, end_idx):
            try:
                preds = self.predict(df, idx, **kwargs)
                target = self._extract_target(df, idx, **kwargs)

                if target is None or np.isnan(target):
                    continue

                # I use the "point" prediction (or first key)
                pred_key = "point" if "point" in preds else list(preds.keys())[0]
                pred_val = preds[pred_key]
                if isinstance(pred_val, np.ndarray):
                    pred_val = pred_val[0] if len(pred_val) > 0 else np.nan

                if np.isnan(pred_val):
                    continue

                error = pred_val - target
                errors.append(error)
                abs_errors.append(abs(error))
                if abs(target) > 1.0:
                    pct_errors.append(abs(error) / abs(target) * 100)
            except (IndexError, KeyError):
                continue

        if not errors:
            return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "bias": np.nan}

        errors_arr = np.array(errors)
        return {
            "mae": np.mean(abs_errors),
            "rmse": np.sqrt(np.mean(errors_arr ** 2)),
            "mape": np.mean(pct_errors) if pct_errors else np.nan,
            "bias": np.mean(errors_arr),
            "n_samples": len(errors),
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"
