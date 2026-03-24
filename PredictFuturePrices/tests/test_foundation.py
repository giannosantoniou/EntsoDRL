"""Tests for Phase 1 foundation: config, base_collector, base_forecaster, accuracy tracker."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from PredictFuturePrices.config import (
    PATHS, ENTSOE, ADMIE, WEATHER, COMMODITY,
    DAM_FORECASTER, IDA_FORECASTER, LOAD_FORECASTER, XBID_FORECASTER,
    ACCURACY, PACKAGE_ROOT, PathConfig,
)
from PredictFuturePrices.collectors.base_collector import BaseCollector
from PredictFuturePrices.forecasters.base_forecaster import BaseForecaster
from PredictFuturePrices.accuracy.tracker import AccuracyTracker, PREDICTION_COLUMNS


# ─── Config Tests ───────────────────────────────────────────────────────


class TestConfig:
    """I verify that all config dataclasses have sane defaults."""

    def test_paths_exist_as_path_objects(self):
        assert isinstance(PATHS.data_dir, Path)
        assert isinstance(PATHS.models_dir, Path)
        assert isinstance(PATHS.reports_dir, Path)

    def test_package_root_is_correct(self):
        assert PACKAGE_ROOT.name == "PredictFuturePrices"

    def test_entsoe_defaults(self):
        assert ENTSOE.country_code_gr == "GR"
        assert ENTSOE.country_code_rs == "RS"
        assert ENTSOE.request_delay_s > 0

    def test_admie_defaults(self):
        assert "isp1" in ADMIE.categories
        assert ADMIE.chunk_days == 30

    def test_weather_defaults(self):
        assert WEATHER.latitude == pytest.approx(37.98)
        assert len(WEATHER.hourly_variables) == 4

    def test_commodity_defaults(self):
        assert COMMODITY.ttf_ticker == "TTF=F"

    def test_dam_forecaster_config(self):
        assert DAM_FORECASTER.n_periods == 96
        assert DAM_FORECASTER.resolution_minutes == 15
        assert len(DAM_FORECASTER.quantiles) == 3
        assert DAM_FORECASTER.lgbm_params["n_estimators"] == 500

    def test_ida_forecaster_config(self):
        assert IDA_FORECASTER.ida_numbers == (1, 2, 3)
        assert IDA_FORECASTER.gate_hours[1] == 15
        assert IDA_FORECASTER.blending_alphas[1] == pytest.approx(0.85)

    def test_load_forecaster_config(self):
        assert LOAD_FORECASTER.n_periods == 96
        assert LOAD_FORECASTER.resolution_minutes == 15

    def test_xbid_forecaster_config(self):
        assert XBID_FORECASTER.horizons_hours == (1, 2, 4, 8)

    def test_accuracy_config(self):
        assert 7 in ACCURACY.rolling_windows_days
        assert 30 in ACCURACY.rolling_windows_days
        assert "mae" in ACCURACY.metrics

    def test_frozen_dataclass(self):
        """I verify configs are immutable."""
        with pytest.raises(AttributeError):
            PATHS.data_dir = Path("/tmp")


# ─── BaseCollector Tests ────────────────────────────────────────────────


class ConcreteCollector(BaseCollector):
    """I create a concrete implementation for testing the ABC."""

    def __init__(self, output_path: Path):
        super().__init__("test_collector", output_path)
        self._fetch_calls = []

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        self._fetch_calls.append((start, end))
        # I return hourly data for the requested range
        idx = pd.date_range(start, end, freq="h")
        return pd.DataFrame({"value": np.random.randn(len(idx))}, index=idx)

    def get_default_start(self) -> datetime:
        return datetime(2024, 1, 1)


class TestBaseCollector:
    """I test the incremental fetch/save pattern."""

    def test_fetch_incremental_new_data(self, tmp_path):
        output = tmp_path / "test.csv"
        collector = ConcreteCollector(output)

        end = datetime(2024, 1, 3)
        data = collector.fetch_incremental(end=end)

        assert not data.empty
        assert output.exists()
        assert len(collector._fetch_calls) == 1

    def test_fetch_incremental_no_duplicate(self, tmp_path):
        output = tmp_path / "test.csv"
        collector = ConcreteCollector(output)

        end1 = datetime(2024, 1, 2)
        data1 = collector.fetch_incremental(end=end1)

        end2 = datetime(2024, 1, 4)
        data2 = collector.fetch_incremental(end=end2)

        # I verify the second fetch starts after the first data
        assert len(collector._fetch_calls) == 2
        second_start = collector._fetch_calls[1][0]
        assert second_start > datetime(2024, 1, 1)

    def test_save_incremental_deduplicates(self, tmp_path):
        output = tmp_path / "test.csv"
        collector = ConcreteCollector(output)

        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        df1 = pd.DataFrame({"value": np.ones(10)}, index=idx)
        collector.save_incremental(df1)

        # I save overlapping data with different values
        df2 = pd.DataFrame({"value": np.ones(10) * 2}, index=idx)
        added = collector.save_incremental(df2)

        # I verify deduplication kept latest values
        loaded = collector.load_existing()
        assert len(loaded) == 10
        assert loaded["value"].iloc[0] == pytest.approx(2.0)
        assert added == 0  # No new rows since same timestamps

    def test_load_existing_empty(self, tmp_path):
        output = tmp_path / "nonexistent.csv"
        collector = ConcreteCollector(output)
        assert collector.load_existing().empty

    def test_get_last_timestamp_none(self, tmp_path):
        output = tmp_path / "nonexistent.csv"
        collector = ConcreteCollector(output)
        assert collector.get_last_timestamp() is None

    def test_get_last_timestamp_after_save(self, tmp_path):
        output = tmp_path / "test.csv"
        collector = ConcreteCollector(output)

        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({"value": range(5)}, index=idx)
        collector.save_incremental(df)

        last = collector.get_last_timestamp()
        assert last is not None
        assert last == datetime(2024, 1, 1, 4)

    def test_fetch_incremental_already_up_to_date(self, tmp_path):
        output = tmp_path / "test.csv"
        collector = ConcreteCollector(output)

        # I save data up to now
        end = datetime.now()
        idx = pd.date_range(end - timedelta(hours=5), end, freq="h")
        df = pd.DataFrame({"value": range(len(idx))}, index=idx)
        collector.save_incremental(df)

        # I try fetching but we're already up to date
        data = collector.fetch_incremental(end=end)
        assert data.empty

    def test_repr(self, tmp_path):
        collector = ConcreteCollector(tmp_path / "test.csv")
        assert "ConcreteCollector" in repr(collector)
        assert "test_collector" in repr(collector)


# ─── BaseForecaster Tests ───────────────────────────────────────────────


class ConcreteForecaster(BaseForecaster):
    """I create a concrete implementation for testing the ABC."""

    def __init__(self):
        super().__init__("test_forecaster", n_features=3)
        self.feature_names = ["f1", "f2", "f3"]

    def build_features(self, df, idx, **kwargs):
        if idx < 1:
            return None
        row = df.iloc[idx]
        return np.array([row["price"], row.get("load", 0), idx / len(df)])

    def _extract_target(self, df, idx, **kwargs):
        if idx + 1 >= len(df):
            return None
        return df.iloc[idx + 1]["price"]

    def fit(self, df, train_end_idx, val_end_idx):
        # I just store mean as a trivial "model"
        X_train, y_train = self.build_dataset(df, 1, train_end_idx)
        if len(y_train) > 0:
            self.models["mean"] = float(np.mean(y_train))
        self.is_fitted = True
        return self

    def predict(self, df, idx, **kwargs):
        mean_val = self.models.get("mean", 0.0)
        return {"point": np.array([mean_val])}


class TestBaseForecaster:
    """I test the forecaster ABC functionality."""

    def _make_df(self, n=100):
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "price": np.sin(np.arange(n) * 0.1) * 50 + 100,
            "load": np.random.randn(n) * 1000 + 5000,
        }, index=idx)

    def test_build_features(self):
        fc = ConcreteForecaster()
        df = self._make_df()
        features = fc.build_features(df, 5)
        assert features.shape == (3,)
        assert features[0] == pytest.approx(df.iloc[5]["price"])

    def test_build_features_returns_none_for_first(self):
        fc = ConcreteForecaster()
        df = self._make_df()
        assert fc.build_features(df, 0) is None

    def test_build_dataset(self):
        fc = ConcreteForecaster()
        df = self._make_df()
        X, y = fc.build_dataset(df, 1, 50)
        assert X.shape[1] == 3
        assert len(X) == len(y)
        assert len(X) > 0

    def test_fit_and_predict(self):
        fc = ConcreteForecaster()
        df = self._make_df()
        fc.fit(df, 70, 90)
        assert fc.is_fitted
        preds = fc.predict(df, 50)
        assert "point" in preds
        assert len(preds["point"]) == 1

    def test_evaluate(self):
        fc = ConcreteForecaster()
        df = self._make_df()
        fc.fit(df, 70, 90)
        metrics = fc.evaluate(df, 1, 20)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["n_samples"] > 0

    def test_evaluate_unfitted_raises(self):
        fc = ConcreteForecaster()
        df = self._make_df()
        with pytest.raises(RuntimeError):
            fc.evaluate(df, 1, 20)

    def test_save_and_load(self, tmp_path):
        fc = ConcreteForecaster()
        df = self._make_df()
        fc.fit(df, 70, 90)

        model_path = tmp_path / "test_model.pkl"
        fc.save(model_path)
        assert model_path.exists()

        fc2 = ConcreteForecaster()
        fc2.load(model_path)
        assert fc2.is_fitted
        assert fc2.models["mean"] == pytest.approx(fc.models["mean"])

    def test_load_nonexistent_raises(self, tmp_path):
        fc = ConcreteForecaster()
        with pytest.raises(FileNotFoundError):
            fc.load(tmp_path / "nonexistent.pkl")

    def test_repr(self):
        fc = ConcreteForecaster()
        assert "unfitted" in repr(fc)
        fc.is_fitted = True
        assert "fitted" in repr(fc)


# ─── AccuracyTracker Tests ─────────────────────────────────────────────


class TestAccuracyTracker:
    """I test the two-phase accuracy tracking system."""

    def test_record_single_prediction(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)
        tracker.record_prediction(
            forecaster_name="dam",
            prediction_timestamp=datetime(2024, 1, 1, 12),
            target_timestamp=datetime(2024, 1, 2, 0),
            predicted_value=85.5,
            horizon_hours=12.0,
        )

        report = tracker._load_report("dam")
        assert len(report) == 1
        assert report.iloc[0]["predicted_value"] == pytest.approx(85.5)
        assert pd.isna(report.iloc[0]["actual_value"])

    def test_record_batch(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)
        targets = [datetime(2024, 1, 2, h) for h in range(24)]
        values = np.random.randn(24) * 20 + 80

        tracker.record_batch(
            forecaster_name="dam",
            prediction_timestamp=datetime(2024, 1, 1, 12),
            target_timestamps=targets,
            predicted_values=values,
            horizon_hours=[12 + h for h in range(24)],
        )

        report = tracker._load_report("dam")
        assert len(report) == 24

    def test_fill_actuals(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)

        # I record a prediction first
        target_ts = datetime(2024, 1, 2, 0)
        tracker.record_prediction(
            forecaster_name="dam",
            prediction_timestamp=datetime(2024, 1, 1, 12),
            target_timestamp=target_ts,
            predicted_value=85.0,
            horizon_hours=12.0,
        )

        # I fill the actual value
        updated = tracker.fill_actuals("dam", {target_ts: 90.0})
        assert updated == 1

        report = tracker._load_report("dam")
        assert report.iloc[0]["actual_value"] == pytest.approx(90.0)
        assert report.iloc[0]["error"] == pytest.approx(-5.0)
        assert report.iloc[0]["abs_error"] == pytest.approx(5.0)

    def test_fill_actuals_skips_already_filled(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)
        target_ts = datetime(2024, 1, 2, 0)

        tracker.record_prediction(
            forecaster_name="dam",
            prediction_timestamp=datetime(2024, 1, 1, 12),
            target_timestamp=target_ts,
            predicted_value=85.0,
            horizon_hours=12.0,
        )

        tracker.fill_actuals("dam", {target_ts: 90.0})
        # I try filling again — should not change anything
        updated = tracker.fill_actuals("dam", {target_ts: 95.0})
        assert updated == 0

    def test_compute_metrics(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)

        # I record and fill 10 predictions
        for h in range(10):
            target_ts = datetime(2024, 1, 2, h)
            tracker.record_prediction(
                forecaster_name="dam",
                prediction_timestamp=datetime(2024, 1, 1, 12),
                target_timestamp=target_ts,
                predicted_value=80.0 + h,
                horizon_hours=12.0 + h,
            )

        actuals = {datetime(2024, 1, 2, h): 85.0 for h in range(10)}
        tracker.fill_actuals("dam", actuals)

        metrics = tracker.compute_metrics("dam")
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["n_filled"] == 10
        assert metrics["fill_rate"] == pytest.approx(1.0)

    def test_compute_metrics_empty(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)
        metrics = tracker.compute_metrics("nonexistent")
        assert np.isnan(metrics["mae"])

    def test_compute_metrics_with_window(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)

        # I record a prediction with a recent timestamp
        now = datetime.now()
        target_ts = now + timedelta(hours=12)
        tracker.record_prediction(
            forecaster_name="dam",
            prediction_timestamp=now,
            target_timestamp=target_ts,
            predicted_value=85.0,
            horizon_hours=12.0,
        )
        tracker.fill_actuals("dam", {target_ts: 90.0})

        # I compute with 7-day window — should include our prediction
        metrics_7d = tracker.compute_metrics("dam", window_days=7)
        assert metrics_7d["n_filled"] == 1

    def test_generate_summary(self, tmp_path):
        tracker = AccuracyTracker(reports_dir=tmp_path)

        # I set up two forecasters with data
        for name in ["dam", "ida1"]:
            target_ts = datetime(2024, 1, 2, 0)
            tracker.record_prediction(
                forecaster_name=name,
                prediction_timestamp=datetime(2024, 1, 1, 12),
                target_timestamp=target_ts,
                predicted_value=85.0,
                horizon_hours=12.0,
            )
            tracker.fill_actuals(name, {target_ts: 90.0})

        summary = tracker.generate_summary()
        assert not summary.empty
        assert "dam" in summary["forecaster"].values
        assert "ida1" in summary["forecaster"].values

        # I verify summary CSV was written
        summary_path = tmp_path / "accuracy_summary.csv"
        assert summary_path.exists()

    def test_pct_error_near_zero_actual(self, tmp_path):
        """I verify MAPE is NaN when actual is near zero (below floor)."""
        tracker = AccuracyTracker(reports_dir=tmp_path)
        target_ts = datetime(2024, 1, 2, 0)

        tracker.record_prediction(
            forecaster_name="test",
            prediction_timestamp=datetime(2024, 1, 1, 12),
            target_timestamp=target_ts,
            predicted_value=5.0,
            horizon_hours=12.0,
        )
        tracker.fill_actuals("test", {target_ts: 0.5})

        report = tracker._load_report("test")
        assert pd.isna(report.iloc[0]["pct_error"])

    def test_record_batch_scalar_horizon(self, tmp_path):
        """I verify batch recording works with a single horizon value."""
        tracker = AccuracyTracker(reports_dir=tmp_path)
        targets = [datetime(2024, 1, 2, h) for h in range(3)]
        values = [80.0, 85.0, 90.0]

        tracker.record_batch(
            forecaster_name="test",
            prediction_timestamp=datetime(2024, 1, 1, 12),
            target_timestamps=targets,
            predicted_values=values,
            horizon_hours=12.0,
        )

        report = tracker._load_report("test")
        assert len(report) == 3
        assert all(report["horizon_hours"] == 12.0)
