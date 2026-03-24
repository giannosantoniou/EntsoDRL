"""Integration tests for the full PredictFuturePrices pipeline.

I test the end-to-end flow: data → train → predict → accuracy tracking.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def _create_full_dataset(n_periods=3200):
    """I create a complete dataset mimicking the merged output at 15-min resolution."""
    sph = 4
    idx = pd.date_range("2024-01-01", periods=n_periods, freq="15min")
    np.random.seed(42)

    dam = np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph)) * 30 + 80
    load = np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph) + 1) * 2000 + 6000

    df = pd.DataFrame({
        "dam_price_gr": dam + np.random.randn(n_periods) * 5,
        "price": dam + np.random.randn(n_periods) * 5,
        "rs_dam_price": dam + np.random.randn(n_periods) * 8 + 3,
        "rs_gr_spread": np.random.randn(n_periods) * 8 + 3,
        "system_load_mw": load + np.random.randn(n_periods) * 200,
        "load_forecast": load + np.random.randn(n_periods) * 300,
        "load_forecast_error_lag": np.random.randn(n_periods) * 200,
        "res_forecast_mw": np.random.rand(n_periods) * 2000 + 500,
        "res_production_mw": np.random.rand(n_periods) * 2000 + 500,
        "res_forecast_error_lag": np.random.randn(n_periods) * 300,
        "isp1_price_up": dam + np.random.randn(n_periods) * 15 + 10,
        "isp1_price_down": dam + np.random.randn(n_periods) * 15 - 10,
        "net_imbalance_mw": np.random.randn(n_periods) * 100,
        "imbalance_price": dam + np.random.randn(n_periods) * 20,
        "system_deviation_mwh": np.random.randn(n_periods) * 50,
        "xbid_price_bid": dam + np.random.randn(n_periods) * 5 - 1.5,
        "xbid_price_mid": dam + np.random.randn(n_periods) * 5,
        "afrr_price_up": np.random.rand(n_periods) * 20 + 5,
        "afrr_price_down": np.random.rand(n_periods) * 15 + 3,
        "mfrr_price_up": np.random.rand(n_periods) * 100 + 50,
        "mfrr_price_down": np.random.rand(n_periods) * 80 + 30,
        "mfrr_spread": np.random.randn(n_periods) * 30,
        "temperature": np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph) * 365) * 10 + 18,
        "wind_100m": np.random.rand(n_periods) * 20,
        "solar_radiation": np.maximum(0, np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph)) * 800),
        "cloud_cover": np.random.rand(n_periods) * 100,
        "ttf_gas": np.random.rand(n_periods) * 10 + 30,
        "eua_carbon": np.random.rand(n_periods) * 15 + 65,
    }, index=idx)

    return df


class TestTrainModule:
    """I test the train.py module functions."""

    def test_split_data(self):
        from PredictFuturePrices.train import split_data
        df = _create_full_dataset()
        train_end, val_end = split_data(df)

        assert train_end < val_end < len(df)
        assert train_end == int(len(df) * 0.7)
        assert val_end == int(len(df) * 0.85)

    def test_load_training_data_fallback(self, tmp_path, monkeypatch):
        """I verify load_training_data raises when no data is found."""
        from PredictFuturePrices.train import load_training_data
        import PredictFuturePrices.train as train_module
        # I override PATHS so fallbacks also point to nonexistent dir
        monkeypatch.setattr(train_module, "PATHS", type(train_module.PATHS)(
            data_dir=tmp_path,
            merged_csv=tmp_path / "nonexistent.csv",
        ))
        with pytest.raises(FileNotFoundError):
            load_training_data(tmp_path / "nonexistent.csv")


class TestPredictModule:
    """I test the predict.py module functions."""

    def test_load_models_empty_dir(self, tmp_path):
        from PredictFuturePrices.predict import load_models
        models = load_models(tmp_path)
        assert len(models) == 0

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_load_models_with_saved(self, tmp_path):
        """I verify models can be saved and loaded back."""
        from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster
        from PredictFuturePrices.predict import load_models

        df = _create_full_dataset()
        fc = DAMForecaster()
        fc.fit(df, 2000, 2800)
        fc.save(tmp_path / "dam_forecaster.pkl")

        models = load_models(tmp_path)
        assert "dam" in models
        assert models["dam"].is_fitted


class TestAccuracyIntegration:
    """I test the accuracy tracking integration."""

    def test_record_and_fill_cycle(self, tmp_path):
        """I test the full record → fill → metrics cycle."""
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker

        tracker = AccuracyTracker(reports_dir=tmp_path)

        # I record DAM predictions
        now = datetime(2024, 1, 1, 12)
        targets = [datetime(2024, 1, 2, h) for h in range(24)]
        values = np.random.randn(24) * 20 + 80
        tracker.record_batch("dam", now, targets, values, horizon_hours=12.0)

        # I verify predictions were recorded (report has 24 rows)
        report = tracker._load_report("dam")
        assert len(report) == 24
        assert report["actual_value"].isna().all()

        # I fill actuals
        actuals = {datetime(2024, 1, 2, h): 85.0 + h * 0.5 for h in range(24)}
        updated = tracker.fill_actuals("dam", actuals)
        assert updated == 24

        # I verify metrics after filling
        metrics_after = tracker.compute_metrics("dam")
        assert metrics_after["n_filled"] == 24
        assert metrics_after["fill_rate"] == pytest.approx(1.0)
        assert metrics_after["mae"] > 0


@pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
class TestEndToEnd:
    """I test the full pipeline: train → predict → accuracy."""

    def test_dam_pipeline(self, tmp_path):
        """I test DAM forecaster end-to-end."""
        from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import run_predictions, fill_actuals_from_data

        df = _create_full_dataset()

        # I train
        fc = DAMForecaster()
        fc.fit(df, 2000, 2800)
        assert fc.is_fitted

        # I predict
        tracker = AccuracyTracker(reports_dir=tmp_path)
        forecasters = {"dam": fc}
        results = run_predictions(df, forecasters, tracker, predict_idx=1200)
        assert "dam" in results
        assert "point" in results["dam"]

        # I fill actuals from data
        fill_actuals_from_data(df, tracker)

        # I check accuracy
        metrics = tracker.compute_metrics("dam")
        assert metrics["n_filled"] > 0

    def test_all_forecasters_pipeline(self, tmp_path):
        """I test all 4 forecasters in a mini pipeline."""
        from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster
        from PredictFuturePrices.forecasters.load_forecaster import LoadForecaster
        from PredictFuturePrices.forecasters.ida_forecaster import IDAForecaster
        from PredictFuturePrices.forecasters.xbid_forecaster import XBIDForecaster
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import run_predictions

        df = _create_full_dataset()
        train_end, val_end = 2000, 2800

        forecasters = {}
        for name, cls in [
            ("dam", DAMForecaster),
            ("load", LoadForecaster),
            ("ida", IDAForecaster),
            ("xbid", XBIDForecaster),
        ]:
            fc = cls()
            fc.fit(df, train_end, val_end)
            forecasters[name] = fc

        # I run predictions
        tracker = AccuracyTracker(reports_dir=tmp_path)
        results = run_predictions(df, forecasters, tracker, predict_idx=1600)

        assert len(results) == 4
        assert "dam" in results
        assert "load" in results
        assert "ida" in results
        assert "xbid" in results

        # I verify DAM has 96 quarter-hourly predictions
        assert results["dam"]["point"].shape == (96,)
        # I verify XBID has 4 horizon predictions
        assert "1h" in results["xbid"]
        assert "8h" in results["xbid"]

    def test_save_load_predict_roundtrip(self, tmp_path):
        """I test that saved models produce the same predictions."""
        from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster

        df = _create_full_dataset()
        fc = DAMForecaster()
        fc.fit(df, 2000, 2800)

        # I predict before saving
        preds_before = fc.predict(df, 1200)

        # I save and reload
        model_path = tmp_path / "dam_model.pkl"
        fc.save(model_path)

        fc2 = DAMForecaster()
        fc2.load(model_path)
        preds_after = fc2.predict(df, 1200)

        # I verify predictions are identical
        np.testing.assert_array_almost_equal(
            preds_before["point"], preds_after["point"]
        )

    def test_accuracy_summary_generation(self, tmp_path):
        """I test that accuracy summary includes all forecasters."""
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker

        tracker = AccuracyTracker(reports_dir=tmp_path)

        # I record predictions for multiple forecasters
        now = datetime(2024, 1, 1, 12)
        for name in ["dam", "load", "ida1", "xbid"]:
            target = datetime(2024, 1, 2, 12)
            tracker.record_prediction(name, now, target, 85.0, 24.0)
            tracker.fill_actuals(name, {target: 90.0})

        summary = tracker.generate_summary()
        assert not summary.empty
        assert len(summary["forecaster"].unique()) == 4

        # I verify summary CSV was written
        assert (tmp_path / "accuracy_summary.csv").exists()


class TestDataMergerIntegration:
    """I test the data merger integration."""

    def test_merged_data_is_trainable(self, tmp_path):
        """I verify merged data can be used directly for training."""
        from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster

        df = _create_full_dataset()

        # I save as CSV and reload (simulating merger output)
        csv_path = tmp_path / "merged.csv"
        df.to_csv(csv_path)
        reloaded = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # I verify feature building works on reloaded data
        fc = DAMForecaster()
        features = fc.build_features(reloaded, 800, period=48)
        assert features is not None
        assert features.shape == (29,)


class TestResolutionAwarePredictions:
    """I test that _record_predictions adapts to the loaded model's resolution."""

    def _make_mock_forecaster(self, sph, n_periods, name="dam"):
        """I create a minimal mock forecaster with given resolution metadata."""
        class MockForecaster:
            def __init__(self, sph, n_periods, name):
                self.sph = sph
                self.n_periods = n_periods
                self.name = name
                self.is_fitted = True
                self.models = {f"p{p}_q0.5": None for p in range(n_periods)}

            def predict(self, df, idx, **kwargs):
                point = np.random.randn(self.n_periods) * 10 + 80
                return {"point": point}

        return MockForecaster(sph, n_periods, name)

    def test_record_predictions_hourly_model(self, tmp_path):
        """I verify that an hourly model (sph=1) records 24 predictions, not 96."""
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import _record_predictions

        tracker = AccuracyTracker(reports_dir=tmp_path)
        fc = self._make_mock_forecaster(sph=1, n_periods=24)
        preds = {"point": np.random.randn(24) * 10 + 80}
        now = datetime(2024, 6, 1, 13, 0)
        predict_ts = pd.Timestamp("2024-06-01 12:00")

        _record_predictions("dam", fc, preds, tracker, now, predict_ts)

        report = tracker._load_report("dam")
        assert len(report) == 24, f"Expected 24 rows for hourly model, got {len(report)}"

    def test_record_predictions_15min_model(self, tmp_path):
        """I verify that a 15-min model (sph=4) records 96 predictions."""
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import _record_predictions

        tracker = AccuracyTracker(reports_dir=tmp_path)
        fc = self._make_mock_forecaster(sph=4, n_periods=96)
        preds = {"point": np.random.randn(96) * 10 + 80}
        now = datetime(2024, 6, 1, 13, 0)
        predict_ts = pd.Timestamp("2024-06-01 12:00")

        _record_predictions("dam", fc, preds, tracker, now, predict_ts)

        report = tracker._load_report("dam")
        assert len(report) == 96, f"Expected 96 rows for 15-min model, got {len(report)}"

    def test_record_predictions_load_hourly(self, tmp_path):
        """I verify that an hourly load model records 24 predictions."""
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import _record_predictions

        tracker = AccuracyTracker(reports_dir=tmp_path)
        fc = self._make_mock_forecaster(sph=1, n_periods=24, name="load")
        preds = {"point": np.random.randn(24) * 500 + 6000}
        now = datetime(2024, 6, 1, 13, 0)
        predict_ts = pd.Timestamp("2024-06-01 12:00")

        _record_predictions("load", fc, preds, tracker, now, predict_ts)

        report = tracker._load_report("load")
        assert len(report) == 24, f"Expected 24 rows for hourly load model, got {len(report)}"

    def test_predict_idx_uses_model_periods(self):
        """I verify predict_idx = len(df) - n_periods - 1 for hourly models."""
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import run_predictions

        # I create a dataset long enough for hourly models (7*24 + 24 + some margin)
        n_rows = 250
        idx = pd.date_range("2024-01-01", periods=n_rows, freq="h")
        df = pd.DataFrame({"dam_price_gr": np.random.randn(n_rows) * 10 + 80}, index=idx)

        fc = self._make_mock_forecaster(sph=1, n_periods=24)
        forecasters = {"dam": fc}
        tracker = AccuracyTracker()

        # I call run_predictions without explicit predict_idx to trigger auto-calc
        # With hourly models: predict_idx = 250 - 24 - 1 = 225
        # With old code (96 from config): predict_idx = 250 - 96 - 1 = 153
        results = run_predictions(df, forecasters, tracker, predict_idx=None)

        # I verify it didn't error — with the old code, predict_idx=153 < min_history=672
        # would fail. With the fix, predict_idx=225 > min_history=168, so it succeeds.
        assert "dam" in results

    def test_hourly_horizon_spacing(self, tmp_path):
        """I verify hourly models produce horizons spaced 1h apart (not 0.25h)."""
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import _record_predictions

        tracker = AccuracyTracker(reports_dir=tmp_path)
        fc = self._make_mock_forecaster(sph=1, n_periods=24)
        preds = {"point": np.ones(24) * 80}
        now = datetime(2024, 6, 1, 13, 0)
        predict_ts = pd.Timestamp("2024-06-01 12:00")

        _record_predictions("dam", fc, preds, tracker, now, predict_ts)

        report = tracker._load_report("dam")
        # I verify first horizon is 1.0h (60 min), not 0.25h (15 min)
        assert report.iloc[0]["horizon_hours"] == pytest.approx(1.0)
        # I verify last horizon is 24.0h
        assert report.iloc[-1]["horizon_hours"] == pytest.approx(24.0)


class TestIDABlendedActuals:
    """I test that IDA fill_actuals uses blended target, not pure DAM."""

    def test_ida_actuals_equal_dam_when_isp1_equals_dam(self, tmp_path):
        """I verify when ISP1 mid == DAM, blended IDA actuals == DAM.

        This is the sanity check: α×DAM + (1-α)×DAM = DAM for any α.
        """
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import fill_actuals_from_data

        idx = pd.date_range("2024-06-01", periods=48, freq="15min")
        dam_prices = np.full(48, 85.0)
        df = pd.DataFrame({
            "dam_price_gr": dam_prices,
            "isp1_price_up": dam_prices,  # ISP1 mid = DAM
            "isp1_price_down": dam_prices,
        }, index=idx)

        tracker = AccuracyTracker(reports_dir=tmp_path)

        # I record a dummy IDA1 prediction so fill_actuals has something to fill
        target_ts = datetime(2024, 6, 1, 3, 0)
        tracker.record_prediction("ida1", datetime(2024, 6, 1, 0, 0), target_ts, 80.0, 3.0)

        fill_actuals_from_data(df, tracker)

        # I verify the actual value matches DAM (since ISP1 mid == DAM)
        report = tracker._load_report("ida1")
        filled = report.dropna(subset=["actual_value"])
        assert len(filled) == 1
        assert filled.iloc[0]["actual_value"] == pytest.approx(85.0)

    def test_ida_actuals_blend_with_isp1_spread(self, tmp_path):
        """I verify IDA actuals use blended formula when ISP1 != DAM.

        IDA1 α=0.85: actual = 0.85×100 + 0.15×120 = 85 + 18 = 103.0
        IDA3 α=0.30: actual = 0.30×100 + 0.70×120 = 30 + 84 = 114.0
        """
        from PredictFuturePrices.accuracy.tracker import AccuracyTracker
        from PredictFuturePrices.predict import fill_actuals_from_data

        idx = pd.date_range("2024-06-01", periods=48, freq="15min")
        df = pd.DataFrame({
            "dam_price_gr": np.full(48, 100.0),
            "isp1_price_up": np.full(48, 130.0),   # ISP1 mid = (130+110)/2 = 120
            "isp1_price_down": np.full(48, 110.0),
        }, index=idx)

        tracker = AccuracyTracker(reports_dir=tmp_path)

        target_ts = datetime(2024, 6, 1, 3, 0)
        tracker.record_prediction("ida1", datetime(2024, 6, 1, 0, 0), target_ts, 90.0, 3.0)
        tracker.record_prediction("ida3", datetime(2024, 6, 1, 0, 0), target_ts, 90.0, 3.0)

        fill_actuals_from_data(df, tracker)

        # I verify IDA1 blended: 0.85×100 + 0.15×120 = 103.0
        report1 = tracker._load_report("ida1")
        filled1 = report1.dropna(subset=["actual_value"])
        assert len(filled1) == 1
        assert filled1.iloc[0]["actual_value"] == pytest.approx(103.0)

        # I verify IDA3 blended: 0.30×100 + 0.70×120 = 114.0
        report3 = tracker._load_report("ida3")
        filled3 = report3.dropna(subset=["actual_value"])
        assert len(filled3) == 1
        assert filled3.iloc[0]["actual_value"] == pytest.approx(114.0)
