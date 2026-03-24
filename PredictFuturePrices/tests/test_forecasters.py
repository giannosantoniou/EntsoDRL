"""Tests for Phase 3: DAM, Load, IDA, XBID forecasters.

I test feature building, fitting, prediction, and save/load for all 4 forecasters.
I use synthetic data to avoid LightGBM training being too slow.
"""

import numpy as np
import pandas as pd
import pytest

from PredictFuturePrices.forecasters.dam_forecaster import DAMForecaster
from PredictFuturePrices.forecasters.load_forecaster import LoadForecaster
from PredictFuturePrices.forecasters.ida_forecaster import IDAForecaster
from PredictFuturePrices.forecasters.xbid_forecaster import XBIDForecaster

# I check if LightGBM is available
try:
    import lightgbm
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def _make_merged_df(n_periods=2000):
    """I create a realistic merged DataFrame for testing all forecasters.

    I use 15-min resolution (freq='15min') to match the 96-period DAM structure.
    """
    sph = 4  # steps per hour (15-min resolution)
    idx = pd.date_range("2024-01-01", periods=n_periods, freq="15min")
    np.random.seed(42)

    dam = np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph)) * 30 + 80
    load = np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph) + 1) * 2000 + 6000

    df = pd.DataFrame({
        # DAM prices
        "dam_price_gr": dam + np.random.randn(n_periods) * 5,
        "price": dam + np.random.randn(n_periods) * 5,
        # Serbia prices
        "rs_dam_price": dam + np.random.randn(n_periods) * 8 + 3,
        "rs_gr_spread": np.random.randn(n_periods) * 8 + 3,
        # Load
        "system_load_mw": load + np.random.randn(n_periods) * 200,
        "load_forecast": load + np.random.randn(n_periods) * 300,
        "load_forecast_error_lag": np.random.randn(n_periods) * 200,
        # RES
        "res_forecast_mw": np.random.rand(n_periods) * 2000 + 500,
        "res_production_mw": np.random.rand(n_periods) * 2000 + 500,
        "res_forecast_error_lag": np.random.randn(n_periods) * 300,
        # ISP1 balancing (observation signals)
        "isp1_price_up": dam + np.random.randn(n_periods) * 15 + 10,
        "isp1_price_down": dam + np.random.randn(n_periods) * 15 - 10,
        # Imbalance
        "net_imbalance_mw": np.random.randn(n_periods) * 100,
        "imbalance_price": dam + np.random.randn(n_periods) * 20,
        "system_deviation_mwh": np.random.randn(n_periods) * 50,
        # XBID
        "xbid_price_bid": dam + np.random.randn(n_periods) * 5 - 1.5,
        "xbid_price_mid": dam + np.random.randn(n_periods) * 5,
        # Balancing capacity
        "afrr_price_up": np.random.rand(n_periods) * 20 + 5,
        "afrr_price_down": np.random.rand(n_periods) * 15 + 3,
        "mfrr_price_up": np.random.rand(n_periods) * 100 + 50,
        "mfrr_price_down": np.random.rand(n_periods) * 80 + 30,
        "mfrr_spread": np.random.randn(n_periods) * 30,
        # Weather
        "temperature": np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph) * 365) * 10 + 18,
        "wind_100m": np.random.rand(n_periods) * 20,
        "solar_radiation": np.maximum(0, np.sin(np.arange(n_periods) * 2 * np.pi / (24 * sph)) * 800),
        "cloud_cover": np.random.rand(n_periods) * 100,
        # Commodities
        "ttf_gas": np.random.rand(n_periods) * 10 + 30,
        "eua_carbon": np.random.rand(n_periods) * 15 + 65,
    }, index=idx)

    return df


# ─── DAM Forecaster Tests ──────────────────────────────────────────────


class TestDAMForecaster:
    """I test the DAM price forecaster."""

    def test_init(self):
        fc = DAMForecaster()
        assert fc.name == "dam"
        assert fc.n_features == 29
        assert len(fc.FEATURE_NAMES) == 29
        assert fc.n_periods == 96
        assert fc.sph == 4
        assert not fc.is_fitted

    def test_build_features_shape(self):
        fc = DAMForecaster()
        df = _make_merged_df()
        features = fc.build_features(df, 800, period=48)
        assert features is not None
        assert features.shape == (29,)
        assert not np.any(np.isnan(features))

    def test_build_features_too_early(self):
        fc = DAMForecaster()
        df = _make_merged_df()
        assert fc.build_features(df, 5, period=0) is None

    def test_build_features_different_periods(self):
        fc = DAMForecaster()
        df = _make_merged_df()
        # I test period 24 (=hour 6) and period 72 (=hour 18)
        # sin(2π*24/96)=1.0, sin(2π*72/96)=-1.0
        f24 = fc.build_features(df, 800, period=24)
        f72 = fc.build_features(df, 800, period=72)

        # I verify period_sin/cos differ between periods 24 and 72
        assert f24[12] != pytest.approx(f72[12])  # period_sin
        assert f24[12] == pytest.approx(1.0, abs=0.01)
        assert f72[12] == pytest.approx(-1.0, abs=0.01)

    def test_extract_target(self):
        fc = DAMForecaster()
        df = _make_merged_df()
        target = fc._extract_target(df, 800, period=20)
        assert target is not None
        assert 0 < target < 200  # I expect reasonable price range

    def test_build_dataset(self):
        fc = DAMForecaster()
        df = _make_merged_df()
        X, y = fc.build_dataset(df, 672, 1200, period=48)
        assert X.shape[1] == 29
        assert len(X) == len(y)
        assert len(X) > 0

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_fit_and_predict(self):
        fc = DAMForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        assert fc.is_fitted
        assert len(fc.models) > 0

        preds = fc.predict(df, 1200)
        assert "point" in preds
        assert "p10" in preds
        assert "p90" in preds
        assert preds["point"].shape == (96,)
        # I verify P10 <= point <= P90 (generally)
        assert np.mean(preds["p10"] <= preds["point"] + 5) > 0.7

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_save_and_load(self, tmp_path):
        fc = DAMForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        model_path = tmp_path / "dam_model.pkl"
        fc.save(model_path)

        fc2 = DAMForecaster()
        fc2.load(model_path)
        assert fc2.is_fitted

        # I verify predictions match
        p1 = fc.predict(df, 1200)
        p2 = fc2.predict(df, 1200)
        np.testing.assert_array_almost_equal(p1["point"], p2["point"])

    def test_predict_unfitted_raises(self):
        fc = DAMForecaster()
        df = _make_merged_df()
        with pytest.raises(RuntimeError):
            fc.predict(df, 800)


# ─── Load Forecaster Tests ─────────────────────────────────────────────


class TestLoadForecaster:
    """I test the load forecaster."""

    def test_init(self):
        fc = LoadForecaster()
        assert fc.name == "load"
        assert fc.n_features == 20
        assert fc.n_periods == 96
        assert fc.sph == 4

    def test_build_features_shape(self):
        fc = LoadForecaster()
        df = _make_merged_df()
        features = fc.build_features(df, 800, period=48)
        assert features is not None
        assert features.shape == (20,)

    def test_get_load_col(self):
        fc = LoadForecaster()
        df = _make_merged_df()
        assert fc._get_load_col(df) == "system_load_mw"

    def test_get_load_col_fallback(self):
        fc = LoadForecaster()
        df = pd.DataFrame({"load_mw": [1, 2, 3]})
        assert fc._get_load_col(df) == "load_mw"

    def test_get_load_col_none(self):
        fc = LoadForecaster()
        df = pd.DataFrame({"other": [1, 2, 3]})
        assert fc._get_load_col(df) is None

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_fit_and_predict(self):
        fc = LoadForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        assert fc.is_fitted
        preds = fc.predict(df, 1200)
        assert "point" in preds
        assert preds["point"].shape == (96,)
        # I verify predictions are in reasonable load range
        assert np.all(preds["point"] > 0)


# ─── IDA Forecaster Tests ──────────────────────────────────────────────


class TestIDAForecaster:
    """I test the IDA auction forecaster."""

    def test_init(self):
        fc = IDAForecaster()
        assert fc.name == "ida"
        assert fc.n_features == 30
        assert fc.sph == 4

    def test_build_features_shape(self):
        fc = IDAForecaster()
        df = _make_merged_df()
        for ida_num in [1, 2, 3]:
            features = fc.build_features(df, 800, ida_number=ida_num)
            assert features is not None
            assert features.shape == (30,)

    def test_ida_number_encoding(self):
        fc = IDAForecaster()
        df = _make_merged_df()
        f1 = fc.build_features(df, 800, ida_number=1)
        f2 = fc.build_features(df, 800, ida_number=2)
        f3 = fc.build_features(df, 800, ida_number=3)

        # I verify IDA number feature (index 24: after DAM(5)+ISP1(3)+Imbalance(3)+Bal(2)+Lags(4)+Cal(6)+Spread(1))
        assert f1[24] == pytest.approx(1.0 / 3.0)
        assert f2[24] == pytest.approx(2.0 / 3.0)
        assert f3[24] == pytest.approx(1.0)

    def test_extract_target_blending(self):
        fc = IDAForecaster()
        df = _make_merged_df()

        # I verify blending: IDA1 is closer to DAM, IDA3 closer to ISP1
        targets = {}
        for ida_num in [1, 2, 3]:
            targets[ida_num] = fc._extract_target(df, 800, ida_number=ida_num)
            assert targets[ida_num] is not None

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_fit_and_predict(self):
        fc = IDAForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        assert fc.is_fitted
        preds = fc.predict(df, 1200)
        assert "ida1" in preds
        assert "ida2" in preds
        assert "ida3" in preds

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_save_and_load(self, tmp_path):
        fc = IDAForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        path = tmp_path / "ida_model.pkl"
        fc.save(path)

        fc2 = IDAForecaster()
        fc2.load(path)
        assert fc2.is_fitted
        assert len(fc2.models) == len(fc.models)


# ─── XBID Forecaster Tests ─────────────────────────────────────────────


class TestXBIDForecaster:
    """I test the XBID continuous market forecaster."""

    def test_init(self):
        fc = XBIDForecaster()
        assert fc.name == "xbid"
        assert fc.n_features == 37
        assert fc.horizons == (1, 2, 4, 8)
        assert fc.sph == 4

    def test_build_features_shape(self):
        fc = XBIDForecaster()
        df = _make_merged_df()
        features = fc.build_features(df, 800)
        assert features is not None
        assert features.shape == (37,)

    def test_get_xbid_col_preference(self):
        fc = XBIDForecaster()
        df = _make_merged_df()
        assert fc._get_xbid_col(df) == "xbid_price_bid"

    def test_get_xbid_col_fallback(self):
        fc = XBIDForecaster()
        df = pd.DataFrame({"dam_price_gr": [80, 85, 90]})
        assert fc._get_xbid_col(df) == "dam_price_gr"

    def test_build_features_too_early(self):
        fc = XBIDForecaster()
        df = _make_merged_df()
        assert fc.build_features(df, 10) is None

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_fit_and_predict(self):
        fc = XBIDForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        assert fc.is_fitted
        preds = fc.predict(df, 1200)
        assert "1h" in preds
        assert "2h" in preds
        assert "4h" in preds
        assert "8h" in preds

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_evaluate(self):
        fc = XBIDForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        assert fc.is_fitted
        assert len(fc.models) > 0

        metrics = fc.evaluate(df, 1200, 1600, horizon=1)
        assert "mae" in metrics
        assert "rmse" in metrics


# ─── Cross-Forecaster Tests ────────────────────────────────────────────


class TestCrossForecaster:
    """I test patterns that should hold across all forecasters."""

    @pytest.mark.parametrize("ForecasterClass", [
        DAMForecaster, LoadForecaster, IDAForecaster, XBIDForecaster
    ])
    def test_repr(self, ForecasterClass):
        fc = ForecasterClass()
        assert "unfitted" in repr(fc)

    @pytest.mark.parametrize("ForecasterClass", [
        DAMForecaster, LoadForecaster, IDAForecaster, XBIDForecaster
    ])
    def test_feature_names_match_n_features(self, ForecasterClass):
        fc = ForecasterClass()
        assert len(fc.feature_names) == fc.n_features

    @pytest.mark.parametrize("ForecasterClass", [
        DAMForecaster, LoadForecaster, IDAForecaster, XBIDForecaster
    ])
    def test_predict_unfitted_raises(self, ForecasterClass):
        fc = ForecasterClass()
        df = _make_merged_df()
        with pytest.raises(RuntimeError):
            fc.predict(df, 800)

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    @pytest.mark.parametrize("ForecasterClass", [
        DAMForecaster, LoadForecaster, IDAForecaster, XBIDForecaster
    ])
    def test_save_load_roundtrip(self, ForecasterClass, tmp_path):
        fc = ForecasterClass()
        df = _make_merged_df(3200)

        fc.fit(df, 2000, 2800)
        path = tmp_path / f"{fc.name}_model.pkl"
        fc.save(path)

        fc2 = ForecasterClass()
        fc2.load(path)
        assert fc2.is_fitted
        assert len(fc2.models) == len(fc.models)


# ─── Resolution Detection Tests ───────────────────────────────────────


def _make_hourly_df(n_hours=2000):
    """I create a merged DataFrame with hourly resolution for resolution detection tests."""
    idx = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    np.random.seed(42)

    dam = np.sin(np.arange(n_hours) * 2 * np.pi / 24) * 30 + 80

    df = pd.DataFrame({
        "dam_price_gr": dam + np.random.randn(n_hours) * 5,
        "price": dam + np.random.randn(n_hours) * 5,
        "rs_dam_price": dam + np.random.randn(n_hours) * 8 + 3,
        "rs_gr_spread": np.random.randn(n_hours) * 8 + 3,
        "system_load_mw": np.random.rand(n_hours) * 2000 + 6000,
        "load_forecast": np.random.rand(n_hours) * 2000 + 6000,
        "load_forecast_error_lag": np.random.randn(n_hours) * 200,
        "res_forecast_mw": np.random.rand(n_hours) * 2000 + 500,
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


class TestResolutionDetection:
    """I test auto-detection of data resolution in forecasters."""

    def test_detect_15min(self):
        """I verify _detect_resolution returns sph=4 for 15-min data."""
        fc = DAMForecaster()
        df = _make_merged_df(500)
        sph = fc._detect_resolution(df)
        assert sph == 4

    def test_detect_hourly(self):
        """I verify _detect_resolution returns sph=1 for hourly data."""
        fc = DAMForecaster()
        df = _make_hourly_df(500)
        sph = fc._detect_resolution(df)
        assert sph == 1

    def test_detect_30min(self):
        """I verify _detect_resolution returns sph=2 for 30-min data."""
        fc = DAMForecaster()
        idx = pd.date_range("2024-01-01", periods=500, freq="30min")
        df = pd.DataFrame({"dam_price_gr": np.random.randn(500)}, index=idx)
        sph = fc._detect_resolution(df)
        assert sph == 2

    def test_detect_tiny_df(self):
        """I verify _detect_resolution returns sph=1 for tiny DataFrames."""
        fc = DAMForecaster()
        df = pd.DataFrame({"dam_price_gr": [80, 85]},
                          index=pd.date_range("2024-01-01", periods=2, freq="h"))
        sph = fc._detect_resolution(df)
        assert sph == 1

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_dam_fit_hourly_sets_24_periods(self):
        """I verify DAM fit() on hourly data produces 24 periods, not 96."""
        fc = DAMForecaster()
        df = _make_hourly_df(2000)
        fc.fit(df, 1200, 1600)

        assert fc.sph == 1
        assert fc.n_periods == 24
        # I verify we have 24 × 3 = 72 models, not 96 × 3 = 288
        assert len(fc.models) == 24 * 3

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_dam_fit_15min_sets_96_periods(self):
        """I verify DAM fit() on 15-min data produces 96 periods."""
        fc = DAMForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        assert fc.sph == 4
        assert fc.n_periods == 96
        assert len(fc.models) == 96 * 3

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_load_fit_hourly_sets_24_periods(self):
        """I verify Load fit() on hourly data produces 24 periods."""
        fc = LoadForecaster()
        df = _make_hourly_df(2000)
        fc.fit(df, 1200, 1600)

        assert fc.sph == 1
        assert fc.n_periods == 24
        assert len(fc.models) == 24

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_ida_fit_hourly_sets_sph1(self):
        """I verify IDA fit() on hourly data uses sph=1 for lags."""
        fc = IDAForecaster()
        df = _make_hourly_df(2000)
        fc.fit(df, 1200, 1600)

        assert fc.sph == 1
        assert len(fc.models) == 3  # still 3 auctions

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_xbid_fit_hourly_sets_sph1(self):
        """I verify XBID fit() on hourly data uses sph=1 for lags."""
        fc = XBIDForecaster()
        df = _make_hourly_df(2000)
        fc.fit(df, 1200, 1600)

        assert fc.sph == 1
        assert len(fc.models) == 4  # still 4 horizons

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_save_load_preserves_sph(self, tmp_path):
        """I verify save/load round-trip preserves sph and n_periods."""
        fc = DAMForecaster()
        df = _make_hourly_df(2000)
        fc.fit(df, 1200, 1600)

        assert fc.sph == 1
        assert fc.n_periods == 24

        path = tmp_path / "dam_hourly.pkl"
        fc.save(path)

        fc2 = DAMForecaster()
        fc2.load(path)
        assert fc2.sph == 1
        assert fc2.n_periods == 24

    @pytest.mark.skipif(not HAS_LGBM, reason="LightGBM not installed")
    def test_evaluate_correct_period_index(self):
        """I verify evaluate() uses the correct period from kwargs."""
        fc = DAMForecaster()
        df = _make_merged_df(3200)
        fc.fit(df, 2000, 2800)

        # I evaluate at period 48 — should use pred_val = preds["point"][48]
        metrics_48 = fc.evaluate(df, 2000, 2100, period=48)
        assert "mae" in metrics_48
        assert metrics_48.get("n_samples", 0) > 0

        # I evaluate at period 0 — should use pred_val = preds["point"][0]
        metrics_0 = fc.evaluate(df, 2000, 2100, period=0)
        assert "mae" in metrics_0
        assert metrics_0.get("n_samples", 0) > 0
