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


def _create_full_dataset(n_hours=800):
    """I create a complete dataset mimicking the merged output."""
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    np.random.seed(42)

    dam = np.sin(np.arange(n_hours) * 2 * np.pi / 24) * 30 + 80
    load = np.sin(np.arange(n_hours) * 2 * np.pi / 24 + 1) * 2000 + 6000

    df = pd.DataFrame({
        "dam_price_gr": dam + np.random.randn(n_hours) * 5,
        "price": dam + np.random.randn(n_hours) * 5,
        "rs_dam_price": dam + np.random.randn(n_hours) * 8 + 3,
        "rs_gr_spread": np.random.randn(n_hours) * 8 + 3,
        "system_load_mw": load + np.random.randn(n_hours) * 200,
        "load_forecast": load + np.random.randn(n_hours) * 300,
        "load_forecast_error_lag": np.random.randn(n_hours) * 200,
        "res_forecast_mw": np.random.rand(n_hours) * 2000 + 500,
        "res_production_mw": np.random.rand(n_hours) * 2000 + 500,
        "res_forecast_error_lag": np.random.randn(n_hours) * 300,
        "isp1_price_up": dam + np.random.randn(n_hours) * 15 + 10,
        "isp1_price_down": dam + np.random.randn(n_hours) * 15 - 10,
        "net_imbalance_mw": np.random.randn(n_hours) * 100,
        "imbalance_price": dam + np.random.randn(n_hours) * 20,
        "system_deviation_mwh": np.random.randn(n_hours) * 50,
        "xbid_price_bid": dam + np.random.randn(n_hours) * 5 - 1.5,
        "xbid_price_mid": dam + np.random.randn(n_hours) * 5,
        "afrr_price_up": np.random.rand(n_hours) * 20 + 5,
        "afrr_price_down": np.random.rand(n_hours) * 15 + 3,
        "mfrr_price_up": np.random.rand(n_hours) * 100 + 50,
        "mfrr_price_down": np.random.rand(n_hours) * 80 + 30,
        "mfrr_spread": np.random.randn(n_hours) * 30,
        "temperature": np.sin(np.arange(n_hours) * 2 * np.pi / 24 * 365) * 10 + 18,
        "wind_100m": np.random.rand(n_hours) * 20,
        "solar_radiation": np.maximum(0, np.sin(np.arange(n_hours) * 2 * np.pi / 24) * 800),
        "cloud_cover": np.random.rand(n_hours) * 100,
        "ttf_gas": np.random.rand(n_hours) * 10 + 30,
        "eua_carbon": np.random.rand(n_hours) * 15 + 65,
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
        fc.fit(df, 500, 700)
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
        fc.fit(df, 500, 700)
        assert fc.is_fitted

        # I predict
        tracker = AccuracyTracker(reports_dir=tmp_path)
        forecasters = {"dam": fc}
        results = run_predictions(df, forecasters, tracker, predict_idx=300)
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
        train_end, val_end = 500, 700

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
        results = run_predictions(df, forecasters, tracker, predict_idx=400)

        assert len(results) == 4
        assert "dam" in results
        assert "load" in results
        assert "ida" in results
        assert "xbid" in results

        # I verify DAM has 24 hourly predictions
        assert results["dam"]["point"].shape == (24,)
        # I verify XBID has 4 horizon predictions
        assert "1h" in results["xbid"]
        assert "8h" in results["xbid"]

    def test_save_load_predict_roundtrip(self, tmp_path):
        """I test that saved models produce the same predictions."""
        from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster

        df = _create_full_dataset()
        fc = DAMForecaster()
        fc.fit(df, 500, 700)

        # I predict before saving
        preds_before = fc.predict(df, 300)

        # I save and reload
        model_path = tmp_path / "dam_model.pkl"
        fc.save(model_path)

        fc2 = DAMForecaster()
        fc2.load(model_path)
        preds_after = fc2.predict(df, 300)

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
        features = fc.build_features(reloaded, 200, hour=12)
        assert features is not None
        assert features.shape == (29,)
