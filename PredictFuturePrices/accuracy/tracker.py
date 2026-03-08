"""Accuracy tracking system for all forecasters.

I implement a two-phase approach:
1. record_prediction(): writes a row with actual_value=NaN at prediction time
2. fill_actuals(): joins with real data when delivery period has passed

I track per-forecaster CSVs and compute rolling metrics (MAE, MAPE, RMSE, bias).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from PredictFuturePrices.config import ACCURACY, PATHS

logger = logging.getLogger(__name__)

# I define the CSV schema for prediction records
PREDICTION_COLUMNS = [
    "prediction_timestamp",
    "target_timestamp",
    "predicted_value",
    "actual_value",
    "error",
    "abs_error",
    "pct_error",
    "horizon_hours",
    "model_version",
]


class AccuracyTracker:
    """Records predictions and computes accuracy metrics.

    I maintain one CSV per forecaster in the reports directory.
    Each row represents a single (timestamp, horizon) prediction.
    """

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = Path(reports_dir) if reports_dir else PATHS.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _get_report_path(self, forecaster_name: str) -> Path:
        """Return the CSV path for a given forecaster."""
        return self.reports_dir / f"{forecaster_name}_predictions.csv"

    def _load_report(self, forecaster_name: str) -> pd.DataFrame:
        """Load existing prediction records for a forecaster."""
        path = self._get_report_path(forecaster_name)
        if not path.exists():
            return pd.DataFrame(columns=PREDICTION_COLUMNS)

        try:
            df = pd.read_csv(path, parse_dates=["prediction_timestamp", "target_timestamp"])
            return df
        except Exception as e:
            logger.warning(f"Could not load report for {forecaster_name}: {e}")
            return pd.DataFrame(columns=PREDICTION_COLUMNS)

    def _save_report(self, forecaster_name: str, df: pd.DataFrame) -> None:
        """Save prediction records to CSV."""
        path = self._get_report_path(forecaster_name)
        df.to_csv(path, index=False)
        logger.debug(f"Saved {len(df)} records to {path.name}")

    def record_prediction(
        self,
        forecaster_name: str,
        prediction_timestamp: datetime,
        target_timestamp: datetime,
        predicted_value: float,
        horizon_hours: float,
        model_version: str = "v1",
    ) -> None:
        """Record a new prediction (Phase 1 — actual_value is NaN).

        I append a single row with predicted_value set and actual_value=NaN.
        Error metrics will be filled later by fill_actuals().

        Args:
            forecaster_name: Which forecaster made this prediction (e.g., "dam").
            prediction_timestamp: When the prediction was made.
            target_timestamp: The delivery period being predicted.
            predicted_value: The forecasted value.
            horizon_hours: Hours between prediction and delivery.
            model_version: Model version string for tracking.
        """
        report = self._load_report(forecaster_name)

        new_row = pd.DataFrame([{
            "prediction_timestamp": prediction_timestamp,
            "target_timestamp": target_timestamp,
            "predicted_value": predicted_value,
            "actual_value": np.nan,
            "error": np.nan,
            "abs_error": np.nan,
            "pct_error": np.nan,
            "horizon_hours": horizon_hours,
            "model_version": model_version,
        }])

        report = pd.concat([report, new_row], ignore_index=True)
        self._save_report(forecaster_name, report)

    def record_batch(
        self,
        forecaster_name: str,
        prediction_timestamp: datetime,
        target_timestamps: List[datetime],
        predicted_values: Union[np.ndarray, List[float]],
        horizon_hours: Union[float, List[float]],
        model_version: str = "v1",
    ) -> None:
        """Record multiple predictions at once (e.g., 24h DAM forecast).

        I batch-write for efficiency instead of one-by-one.

        Args:
            forecaster_name: Which forecaster.
            prediction_timestamp: When predictions were made.
            target_timestamps: List of delivery timestamps.
            predicted_values: Array of predicted values.
            horizon_hours: Single value or list of horizons.
            model_version: Model version string.
        """
        report = self._load_report(forecaster_name)

        n = len(target_timestamps)
        if isinstance(horizon_hours, (int, float)):
            horizons = [horizon_hours] * n
        else:
            horizons = list(horizon_hours)

        rows = []
        for i in range(n):
            rows.append({
                "prediction_timestamp": prediction_timestamp,
                "target_timestamp": target_timestamps[i],
                "predicted_value": float(predicted_values[i]),
                "actual_value": np.nan,
                "error": np.nan,
                "abs_error": np.nan,
                "pct_error": np.nan,
                "horizon_hours": horizons[i],
                "model_version": model_version,
            })

        new_rows = pd.DataFrame(rows)
        report = pd.concat([report, new_rows], ignore_index=True)
        self._save_report(forecaster_name, report)

    def fill_actuals(
        self,
        forecaster_name: str,
        actuals: Dict[datetime, float],
    ) -> int:
        """Fill actual values for past predictions (Phase 2).

        I match predictions by target_timestamp and compute error metrics.

        Args:
            forecaster_name: Which forecaster.
            actuals: Dict mapping delivery timestamps to actual values.

        Returns:
            Number of rows updated.
        """
        report = self._load_report(forecaster_name)
        if report.empty:
            return 0

        updated = 0
        for i, row in report.iterrows():
            if not pd.isna(row["actual_value"]):
                continue

            target_ts = pd.Timestamp(row["target_timestamp"])
            # I try exact match first, then round to nearest hour
            actual = actuals.get(target_ts.to_pydatetime())
            if actual is None:
                target_rounded = target_ts.floor("h").to_pydatetime()
                actual = actuals.get(target_rounded)

            if actual is not None and not np.isnan(actual):
                predicted = row["predicted_value"]
                error = predicted - actual
                abs_error = abs(error)
                pct_error = (
                    abs_error / abs(actual) * 100
                    if abs(actual) > ACCURACY.mape_floor
                    else np.nan
                )

                report.at[i, "actual_value"] = actual
                report.at[i, "error"] = error
                report.at[i, "abs_error"] = abs_error
                report.at[i, "pct_error"] = pct_error
                updated += 1

        if updated > 0:
            self._save_report(forecaster_name, report)
            logger.info(
                f"[{forecaster_name}] Filled {updated} actuals"
            )

        return updated

    def compute_metrics(
        self,
        forecaster_name: str,
        window_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute accuracy metrics for a forecaster.

        I calculate MAE, RMSE, MAPE, and bias on filled predictions.

        Args:
            forecaster_name: Which forecaster.
            window_days: If set, only use predictions from the last N days.

        Returns:
            Dict with metric names → values.
        """
        report = self._load_report(forecaster_name)
        if report.empty:
            return {m: np.nan for m in ACCURACY.metrics}

        # I filter to rows that have actuals filled
        filled = report.dropna(subset=["actual_value"])
        if filled.empty:
            return {m: np.nan for m in ACCURACY.metrics}

        # I apply time window if specified
        if window_days is not None:
            cutoff = datetime.now() - pd.Timedelta(days=window_days)
            filled = filled[
                pd.to_datetime(filled["prediction_timestamp"]) >= cutoff
            ]
            if filled.empty:
                return {m: np.nan for m in ACCURACY.metrics}

        errors = filled["error"].values
        abs_errors = filled["abs_error"].values
        pct_errors = filled["pct_error"].dropna().values

        return {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mape": float(np.mean(pct_errors)) if len(pct_errors) > 0 else np.nan,
            "bias": float(np.mean(errors)),
            "n_filled": int(len(filled)),
            "n_total": int(len(report)),
            "fill_rate": float(len(filled) / len(report)) if len(report) > 0 else 0.0,
        }

    def generate_summary(self) -> pd.DataFrame:
        """Generate accuracy summary across all forecasters.

        I scan the reports directory for all prediction CSVs and compute
        metrics with rolling windows.

        Returns:
            DataFrame with one row per (forecaster, window) combination.
        """
        rows = []
        # I discover all forecaster reports
        for report_path in sorted(self.reports_dir.glob("*_predictions.csv")):
            forecaster_name = report_path.stem.replace("_predictions", "")

            # I compute overall metrics
            overall = self.compute_metrics(forecaster_name)
            overall["forecaster"] = forecaster_name
            overall["window"] = "all"
            rows.append(overall)

            # I compute rolling window metrics
            for window_days in ACCURACY.rolling_windows_days:
                windowed = self.compute_metrics(forecaster_name, window_days)
                windowed["forecaster"] = forecaster_name
                windowed["window"] = f"{window_days}d"
                rows.append(windowed)

        if not rows:
            return pd.DataFrame()

        summary = pd.DataFrame(rows)
        # I reorder columns for readability
        cols = ["forecaster", "window"] + list(ACCURACY.metrics) + [
            "n_filled", "n_total", "fill_rate",
        ]
        cols = [c for c in cols if c in summary.columns]
        summary = summary[cols]

        # I save the summary
        summary_path = self.reports_dir / "accuracy_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Accuracy summary saved to {summary_path}")

        return summary
