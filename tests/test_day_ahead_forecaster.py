"""
Tests for DAMDayAheadForecaster

I verify the D+1 day-ahead forecaster's feature construction,
prediction shapes, quantile ordering, spline expansion, and persistence.
"""

import sys
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, date, timedelta

# I add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.market_forecaster import DAMDayAheadForecaster, HOURLY_GR_RS_PARAMS


# --- Fixtures ---

def _make_test_df(n_days: int = 30, sph: int = 1) -> pd.DataFrame:
    """I create a synthetic DataFrame with realistic price patterns."""
    n = n_days * 24 * sph
    idx = pd.date_range("2024-06-01", periods=n, freq="h" if sph == 1 else "15min")

    np.random.seed(42)

    # I generate a realistic daily price profile
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in idx])
    base_price = (
        80.0
        + 30 * np.sin(2 * np.pi * (hours - 6) / 24)   # daily cycle
        + 20 * np.sin(2 * np.pi * (hours - 18) / 24)   # evening peak
        + np.random.normal(0, 8, n)                      # noise
    )
    base_price = np.maximum(base_price, 5.0)

    serbia_price = base_price * 0.85 + np.random.normal(5, 3, n)

    df = pd.DataFrame({
        'price': base_price,
        'serbia_dam_price': serbia_price,
        'load_mw': 5000 + 1500 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 200, n),
        'res_total_mw': 2000 + 1000 * np.maximum(np.sin(2 * np.pi * (hours - 6) / 12), 0) + np.random.normal(0, 100, n),
    }, index=idx)

    return df


def _make_serbia_df(n_days: int = 30) -> pd.DataFrame:
    """I create a synthetic Serbia DAM DataFrame."""
    n = n_days * 24
    idx = pd.date_range("2024-06-01", periods=n, freq="h")

    np.random.seed(123)
    hours = np.array([ts.hour for ts in idx])
    prices = (
        75.0
        + 25 * np.sin(2 * np.pi * (hours - 6) / 24)
        + 15 * np.sin(2 * np.pi * (hours - 18) / 24)
        + np.random.normal(0, 5, n)
    )
    prices = np.maximum(prices, 5.0)

    return pd.DataFrame({'price': prices}, index=idx)


def _make_serbia_24h() -> np.ndarray:
    """I create a 24-hour Serbia DAM price curve."""
    np.random.seed(99)
    hours = np.arange(24)
    return (
        75.0
        + 25 * np.sin(2 * np.pi * (hours - 6) / 24)
        + 15 * np.sin(2 * np.pi * (hours - 18) / 24)
        + np.random.normal(0, 3, 24)
    )


# --- Tests ---

class TestBuildFeatures:
    """I test feature construction."""

    def test_build_features_shape(self):
        """I verify that build_day_features returns 21 features per hour."""
        df = _make_test_df(n_days=15)
        serbia_24h = _make_serbia_24h()
        forecaster = DAMDayAheadForecaster()

        # I use day_idx at day 10 so there's enough history
        day_idx = 10 * 24
        features = forecaster.build_day_features(df, day_idx, serbia_24h, sph=1)

        assert len(features) == 24, f"Expected 24 hours, got {len(features)}"
        for hour in range(24):
            assert hour in features, f"Missing hour {hour}"
            assert features[hour].shape == (DAMDayAheadForecaster.N_FEATURES,), \
                f"Hour {hour}: expected {DAMDayAheadForecaster.N_FEATURES} features, got {features[hour].shape}"
            assert features[hour].dtype == np.float32

    def test_features_use_serbia_data(self):
        """I verify that Serbia data appears in features."""
        df = _make_test_df(n_days=15)
        serbia_24h = np.full(24, 100.0)  # constant serbia prices
        forecaster = DAMDayAheadForecaster()

        day_idx = 10 * 24
        features = forecaster.build_day_features(df, day_idx, serbia_24h, sph=1)

        # Feature 0 is Serbia D+1 price for the hour
        for hour in range(24):
            assert features[hour][0] == 100.0, \
                f"Hour {hour}: Serbia price should be 100.0, got {features[hour][0]}"

        # Feature 1 is Serbia daily mean (also 100.0 since all same)
        assert features[0][1] == 100.0

        # Feature 2 is Serbia daily spread (should be 0, clamped to 1.0)
        assert features[0][2] == 1.0  # I clamp spread to avoid div-by-zero


class TestPredictDay:
    """I test daily prediction output."""

    def test_predict_day_output_shape(self):
        """I verify predict_day returns dict with 24-element arrays."""
        df = _make_test_df(n_days=15)
        serbia_24h = _make_serbia_24h()
        forecaster = DAMDayAheadForecaster()  # untrained => regression fallback

        day_idx = 10 * 24
        result = forecaster.predict_day(df, day_idx, serbia_24h, sph=1)

        assert 'hourly' in result
        assert 'p10' in result
        assert 'p90' in result
        assert result['hourly'].shape == (24,)
        assert result['p10'].shape == (24,)
        assert result['p90'].shape == (24,)

    def test_predict_day_uses_regression_fallback(self):
        """I verify untrained forecaster uses Serbia regression."""
        df = _make_test_df(n_days=15)
        serbia_24h = np.full(24, 80.0)
        forecaster = DAMDayAheadForecaster()

        day_idx = 10 * 24
        result = forecaster.predict_day(df, day_idx, serbia_24h, sph=1)

        # I check that regression produces expected values
        for hour in range(24):
            params = HOURLY_GR_RS_PARAMS.get(hour, {'slope': 0.7, 'intercept': 20.0})
            expected = params['slope'] * 80.0 + params['intercept']
            assert abs(result['hourly'][hour] - expected) < 0.01, \
                f"Hour {hour}: expected {expected:.2f}, got {result['hourly'][hour]:.2f}"


class TestPredict96QH:
    """I test 96 quarter-hourly prediction."""

    def test_predict_96qh_output_shape(self):
        """I verify 96 QH outputs."""
        df = _make_test_df(n_days=15)
        serbia_24h = _make_serbia_24h()
        forecaster = DAMDayAheadForecaster()

        day_idx = 10 * 24
        result = forecaster.predict_96qh(df, day_idx, serbia_24h, sph=1)

        assert 'qh' in result
        assert 'p10' in result
        assert 'p90' in result
        assert 'hourly' in result
        assert result['qh'].shape == (96,)
        assert result['p10'].shape == (96,)
        assert result['p90'].shape == (96,)
        assert result['hourly'].shape == (24,)

    def test_96qh_non_negative(self):
        """I verify all 96 QH prices are non-negative."""
        df = _make_test_df(n_days=15)
        serbia_24h = _make_serbia_24h()
        forecaster = DAMDayAheadForecaster()

        day_idx = 10 * 24
        result = forecaster.predict_96qh(df, day_idx, serbia_24h, sph=1)

        assert np.all(result['qh'] >= 0), "Some QH prices are negative"
        assert np.all(result['p10'] >= 0), "Some P10 prices are negative"
        assert np.all(result['p90'] >= 0), "Some P90 prices are negative"


class TestQuantileOrdering:
    """I test quantile ordering guarantees."""

    def test_quantile_ordering_hourly(self):
        """I verify P10 <= point <= P90 at hourly level."""
        df = _make_test_df(n_days=15)
        serbia_24h = _make_serbia_24h()
        forecaster = DAMDayAheadForecaster()

        day_idx = 10 * 24
        result = forecaster.predict_day(df, day_idx, serbia_24h, sph=1)

        for h in range(24):
            assert result['p10'][h] <= result['hourly'][h], \
                f"Hour {h}: P10 ({result['p10'][h]:.2f}) > point ({result['hourly'][h]:.2f})"
            assert result['p90'][h] >= result['hourly'][h], \
                f"Hour {h}: P90 ({result['p90'][h]:.2f}) < point ({result['hourly'][h]:.2f})"

    def test_quantile_ordering_96qh(self):
        """I verify P10 <= point <= P90 at QH level."""
        df = _make_test_df(n_days=15)
        serbia_24h = _make_serbia_24h()
        forecaster = DAMDayAheadForecaster()

        day_idx = 10 * 24
        result = forecaster.predict_96qh(df, day_idx, serbia_24h, sph=1)

        assert np.all(result['p10'] <= result['qh'] + 1e-6), \
            "P10 exceeds point forecast at some QH"
        assert np.all(result['p90'] >= result['qh'] - 1e-6), \
            "P90 below point forecast at some QH"


class TestCubicSpline:
    """I test the cubic spline expansion."""

    def test_cubic_spline_continuity(self):
        """I verify the 96 QH curve is smooth (no large jumps)."""
        forecaster = DAMDayAheadForecaster()

        # I create a smooth hourly curve
        hours = np.arange(24)
        hourly = 80 + 30 * np.sin(2 * np.pi * (hours - 6) / 24)

        qh = forecaster._expand_to_96qh(hourly)

        # I check that consecutive QH values don't jump more than expected
        # Max hourly change is ~15 EUR, so QH change should be ~4 EUR max
        diffs = np.abs(np.diff(qh))
        max_diff = np.max(diffs)
        assert max_diff < 10.0, f"QH curve has jump of {max_diff:.2f} EUR (too large)"

    def test_periodic_boundary(self):
        """I verify hour 23 -> hour 0 transition is smooth."""
        forecaster = DAMDayAheadForecaster()

        hours = np.arange(24)
        hourly = 80 + 10 * np.sin(2 * np.pi * hours / 24)  # smooth periodic

        qh = forecaster._expand_to_96qh(hourly)

        # I check that the boundary (QH 95 -> QH 0) is reasonably smooth
        boundary_jump = abs(qh[0] - qh[95])
        # For a periodic spline with periodic boundary, this should be small
        assert boundary_jump < 5.0, \
            f"Boundary jump = {boundary_jump:.2f} EUR (expected < 5.0)"

    def test_spline_preserves_mean(self):
        """I verify that mean of 96 QH ~ mean of 24 hourly."""
        forecaster = DAMDayAheadForecaster()

        np.random.seed(42)
        hourly = np.random.uniform(50, 150, 24)

        qh = forecaster._expand_to_96qh(hourly)

        # I allow some deviation due to interpolation, but mean should be close
        mean_hourly = np.mean(hourly)
        mean_qh = np.mean(qh)
        assert abs(mean_hourly - mean_qh) < 5.0, \
            f"Mean mismatch: hourly={mean_hourly:.2f}, QH={mean_qh:.2f}"


class TestFitWithMinimalData:
    """I test graceful handling of small datasets."""

    def test_fit_with_minimal_data(self):
        """I verify fit works with minimal data (< 30 days per hour)."""
        df = _make_test_df(n_days=10)  # very small dataset
        serbia_df = _make_serbia_df(n_days=10)

        forecaster = DAMDayAheadForecaster()

        # I use very tight train/val split
        train_end = pd.Timestamp("2024-06-07")
        val_end = pd.Timestamp("2024-06-09")

        # I should not crash, even if no models get trained
        forecaster.fit(df, serbia_df, train_end, val_end)

        # I predict should still work (regression fallback)
        serbia_24h = _make_serbia_24h()
        day_idx = 8 * 24
        result = forecaster.predict_day(df, day_idx, serbia_24h, sph=1)
        assert result['hourly'].shape == (24,)


class TestSaveLoad:
    """I test persistence."""

    def test_save_load_roundtrip(self):
        """I verify save/load preserves forecaster state."""
        forecaster = DAMDayAheadForecaster()

        # I set up a minimal trained state (mock)
        forecaster.is_trained = False  # untrained is fine for this test

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            tmp_path = f.name

        try:
            forecaster.save(tmp_path)
            loaded = DAMDayAheadForecaster.load(tmp_path)

            assert loaded.is_trained == forecaster.is_trained
            assert len(loaded.point_models) == len(forecaster.point_models)
            assert len(loaded.p10_models) == len(forecaster.p10_models)
            assert len(loaded.p90_models) == len(forecaster.p90_models)
        finally:
            import os
            os.unlink(tmp_path)


class TestIntegrationWithDAMBidder:
    """I test integration with the DAMBidder."""

    def test_integration_with_dam_bidder(self):
        """I verify PriceForecast from forecaster -> DAMBidder.generate_bids()."""
        from orchestrator.dam_bidder import DAMBidder
        from orchestrator.interfaces import PriceForecast, BatteryState

        df = _make_test_df(n_days=15)
        serbia_24h = _make_serbia_24h()
        forecaster = DAMDayAheadForecaster()

        # I predict 96 QH
        day_idx = 10 * 24
        result = forecaster.predict_96qh(df, day_idx, serbia_24h, sph=1)

        # I build a PriceForecast from the result
        target_date = df.index[day_idx].date()
        timestamps = []
        base_dt = datetime.combine(target_date, datetime.min.time())
        for i in range(96):
            timestamps.append(base_dt + timedelta(minutes=15 * i))

        spread_pct = 0.02
        mid = result['qh']
        half_spread = mid * spread_pct / 2

        price_forecast = PriceForecast(
            target_date=target_date,
            timestamps=timestamps,
            buy_prices=mid + half_spread,
            sell_prices=mid - half_spread,
        )

        # I verify PriceForecast is valid
        assert price_forecast.n_intervals == 96

        # I create a battery state and generate bids
        battery = BatteryState(
            timestamp=datetime.now(),
            soc=0.5,
            available_power_mw=30.0,
            capacity_mwh=146.0,
        )

        bidder = DAMBidder()
        commitment = bidder.generate_bids(price_forecast, battery)

        # I verify commitment structure
        assert len(commitment.power_mw) == 96
        assert commitment.target_date == target_date
        # I verify power is within bounds
        assert np.all(np.abs(commitment.power_mw) <= 30.0 + 0.01)
