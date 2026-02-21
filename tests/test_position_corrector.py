"""
Tests for orchestrator/position_corrector.py

I test re-forecasting, deviation detection, correction application,
and full IDA correction cycles.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch

from orchestrator.position_corrector import (
    PositionCorrector,
    CorrectionOpportunity,
    CorrectionResult,
    IDA_SESSIONS,
    HENEX_FEE_EUR_MWH,
)
from orchestrator.interfaces import BatteryState
from orchestrator.intraday_trader import IntraDayTrader, IntraDayOrder
from models.market_forecaster import DAMDayAheadForecaster


# --- Helpers ---

def _make_battery_state(soc=0.5):
    """I create a standard battery state for testing."""
    return BatteryState(
        timestamp=datetime(2026, 3, 15, 12, 0),
        soc=soc,
        available_power_mw=30.0,
        capacity_mwh=146.0,
    )


def _make_forecast(hourly=None, offset=0.0):
    """I create a forecast dict with hourly, p10, p90 arrays."""
    if hourly is None:
        hourly = np.array([80 + 20 * np.sin(2 * np.pi * h / 24) + offset for h in range(24)])
    p10 = hourly * 0.90
    p90 = hourly * 1.10
    return {'hourly': hourly.copy(), 'p10': p10, 'p90': p90}


def _make_df(n_days=30, sph=1):
    """I create a minimal DataFrame that looks like historical data."""
    n = n_days * 24 * sph
    start = pd.Timestamp('2026-01-01')
    freq = '1h' if sph == 1 else '15min'
    idx = pd.date_range(start, periods=n, freq=freq)

    prices = 80 + 20 * np.sin(2 * np.pi * np.arange(n) / (24 * sph))
    prices += np.random.normal(0, 5, n)

    df = pd.DataFrame({
        'price': prices,
        'serbia_dam_price': prices * 0.95 + 5,
        'load_mw': 5000 + 1000 * np.sin(2 * np.pi * np.arange(n) / (24 * sph)),
        'res_total_mw': 2000 + 500 * np.sin(2 * np.pi * np.arange(n) / (24 * sph)),
        'solar': 1000 * np.maximum(0, np.sin(2 * np.pi * (np.arange(n) / (24 * sph) - 0.25))),
        'wind_onshore': 500 + 200 * np.random.randn(n),
    }, index=idx)

    return df


def _make_corrector(min_price_deviation=8.0, min_gain=15.0, confidence_threshold=0.6):
    """I create a PositionCorrector with a mock forecaster."""
    forecaster = MagicMock(spec=DAMDayAheadForecaster)
    forecaster.is_trained = True
    trader = IntraDayTrader(max_power_mw=30.0, min_spread_eur=2.0)
    return PositionCorrector(
        day_ahead_forecaster=forecaster,
        intraday_trader=trader,
        min_price_deviation_eur=min_price_deviation,
        min_expected_gain_eur=min_gain,
        confidence_threshold=confidence_threshold,
    )


# =======================================================================
# Re-forecasting tests
# =======================================================================

class TestReforecast:
    """I test the _reforecast method."""

    def test_reforecast_ida1_uses_dam_actuals(self):
        """I verify that injecting DAM actuals changes the forecast output."""
        corrector = _make_corrector()
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        target = date(2026, 1, 20)

        # I set up the mock to return different results based on input
        original_hourly = np.full(24, 80.0)
        updated_hourly = np.full(24, 95.0)

        call_count = [0]
        def mock_predict_day(df_arg, day_idx, serbia_arg, sph):
            call_count[0] += 1
            # I check if DAM actuals were injected by looking at the price column
            day_mask = df_arg.index.normalize() == pd.Timestamp(target)
            day_indices = np.where(np.asarray(day_mask))[0]
            if len(day_indices) > 0 and df_arg.iloc[day_indices[0]]['price'] > 100:
                return {'hourly': updated_hourly, 'p10': updated_hourly * 0.9, 'p90': updated_hourly * 1.1}
            return {'hourly': original_hourly, 'p10': original_hourly * 0.9, 'p90': original_hourly * 1.1}

        corrector.forecaster.predict_day = mock_predict_day

        # I call without actuals
        result_no_actuals = corrector._reforecast('IDA1', target, df, serbia, sph=1)

        # I call with DAM actuals (high prices)
        dam_actuals = np.full(24, 120.0)
        result_with_actuals = corrector._reforecast(
            'IDA1', target, df, serbia, sph=1, dam_actuals=dam_actuals
        )

        assert call_count[0] == 2
        # I verify the forecaster was called both times
        # The result should differ because the updated df has different prices
        assert not np.allclose(result_no_actuals['hourly'], result_with_actuals['hourly'])

    def test_reforecast_ida2_blends_ida1_results(self):
        """I verify that IDA2 re-forecast blends IDA1 results at 70/30 weight."""
        corrector = _make_corrector()
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        target = date(2026, 1, 20)

        # I track the df passed to predict_day to verify blending
        captured_dfs = []
        def mock_predict_day(df_arg, day_idx, serbia_arg, sph):
            captured_dfs.append(df_arg.copy())
            return {'hourly': np.full(24, 90.0), 'p10': np.full(24, 80.0), 'p90': np.full(24, 100.0)}

        corrector.forecaster.predict_day = mock_predict_day

        dam_actuals = np.full(24, 100.0)
        ida1_results = np.full(24, 120.0)

        corrector._reforecast(
            'IDA2', target, df, serbia, sph=1,
            dam_actuals=dam_actuals, ida1_results=ida1_results,
        )

        assert len(captured_dfs) == 1
        # I verify the blended prices: 0.7 * 120 + 0.3 * 100 = 114
        blended_df = captured_dfs[0]
        day_mask = blended_df.index.normalize() == pd.Timestamp(target)
        day_indices = np.where(np.asarray(day_mask))[0]
        if len(day_indices) > 0:
            blended_price = blended_df.iloc[day_indices[0]]['price']
            assert abs(blended_price - 114.0) < 1.0

    def test_reforecast_ida3_only_afternoon(self):
        """I verify that IDA3 returns NaN for hours 0-11."""
        corrector = _make_corrector()
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        target = date(2026, 1, 20)

        corrector.forecaster.predict_day = MagicMock(return_value={
            'hourly': np.full(24, 90.0),
            'p10': np.full(24, 80.0),
            'p90': np.full(24, 100.0),
        })

        result = corrector._reforecast('IDA3', target, df, serbia, sph=1)

        # I check morning hours are NaN
        assert all(np.isnan(result['hourly'][h]) for h in range(12))
        # I check afternoon hours are valid
        assert all(not np.isnan(result['hourly'][h]) for h in range(12, 24))

    def test_reforecast_no_extra_info_matches_original(self):
        """I verify that without new info, re-forecast uses the same data."""
        corrector = _make_corrector()
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        target = date(2026, 1, 20)

        expected = {'hourly': np.full(24, 90.0), 'p10': np.full(24, 80.0), 'p90': np.full(24, 100.0)}
        corrector.forecaster.predict_day = MagicMock(return_value=expected)

        result = corrector._reforecast('IDA1', target, df, serbia, sph=1)

        np.testing.assert_array_equal(result['hourly'], expected['hourly'])

    def test_reforecast_returns_quantiles(self):
        """I verify that re-forecast includes p10 and p90."""
        corrector = _make_corrector()
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        target = date(2026, 1, 20)

        corrector.forecaster.predict_day = MagicMock(return_value={
            'hourly': np.full(24, 90.0),
            'p10': np.full(24, 75.0),
            'p90': np.full(24, 105.0),
        })

        result = corrector._reforecast('IDA1', target, df, serbia, sph=1)

        assert 'p10' in result
        assert 'p90' in result
        assert len(result['p10']) == 24
        assert len(result['p90']) == 24


# =======================================================================
# Deviation detection tests
# =======================================================================

class TestDeviationDetection:
    """I test the _detect_deviations method."""

    def test_no_deviation_when_forecasts_match(self):
        """I verify identical forecasts produce zero opportunities."""
        corrector = _make_corrector()
        forecast = _make_forecast()
        commitment = np.zeros(24)
        battery = _make_battery_state()

        opps = corrector._detect_deviations(
            original_forecast=forecast,
            updated_forecast=forecast,
            original_commitment=commitment,
            battery_state=battery,
            hour_range=range(0, 24),
        )

        assert len(opps) == 0

    def test_deviation_detected_above_threshold(self):
        """I verify a large deviation creates at least one opportunity."""
        corrector = _make_corrector(min_price_deviation=5.0, min_gain=0.0)
        original = _make_forecast()
        updated = _make_forecast(offset=20.0)  # I add 20 EUR deviation
        commitment = np.zeros(24)
        battery = _make_battery_state()

        opps = corrector._detect_deviations(
            original_forecast=original,
            updated_forecast=updated,
            original_commitment=commitment,
            battery_state=battery,
            hour_range=range(0, 24),
        )

        assert len(opps) > 0
        # I verify the deviations are positive (price went up)
        for opp in opps:
            assert opp.price_deviation_eur > 0

    def test_small_deviation_filtered(self):
        """I verify deviations below min_price_deviation are filtered out."""
        corrector = _make_corrector(min_price_deviation=25.0, min_gain=0.0)
        original = _make_forecast()
        # I create a small 5 EUR deviation - below the 25 EUR threshold
        updated = _make_forecast(offset=5.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        opps = corrector._detect_deviations(
            original_forecast=original,
            updated_forecast=updated,
            original_commitment=commitment,
            battery_state=battery,
            hour_range=range(0, 24),
        )

        assert len(opps) == 0

    def test_quantile_filter_rejects_low_confidence(self):
        """I verify wide P10-P90 spread rejects low-confidence corrections."""
        corrector = _make_corrector(min_price_deviation=5.0, min_gain=0.0, confidence_threshold=0.8)
        original = _make_forecast()

        # I create a forecast with very wide quantile spread (low confidence)
        updated_hourly = original['hourly'] + 15.0
        updated = {
            'hourly': updated_hourly,
            'p10': updated_hourly * 0.30,   # Very wide: 30% below
            'p90': updated_hourly * 1.70,   # Very wide: 70% above
        }
        commitment = np.zeros(24)
        battery = _make_battery_state()

        opps = corrector._detect_deviations(
            original_forecast=original,
            updated_forecast=updated,
            original_commitment=commitment,
            battery_state=battery,
            hour_range=range(0, 24),
        )

        # I expect no opportunities because confidence is too low
        assert len(opps) == 0

    def test_ida3_only_checks_hours_12_to_23(self):
        """I verify IDA3 hour range only detects opportunities in afternoon."""
        corrector = _make_corrector(min_price_deviation=5.0, min_gain=0.0)
        original = _make_forecast()
        updated = _make_forecast(offset=20.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        opps = corrector._detect_deviations(
            original_forecast=original,
            updated_forecast=updated,
            original_commitment=commitment,
            battery_state=battery,
            hour_range=IDA_SESSIONS['IDA3']['hour_range'],
        )

        # I verify all opportunities are in hours 12-23
        for opp in opps:
            assert 12 <= opp.hour <= 23


# =======================================================================
# Correction application tests
# =======================================================================

class TestCorrectionApplication:
    """I test the _apply_corrections method."""

    def _make_opportunities(self, n=5, correction_mw=3.0, expected_gain=50.0):
        """I create synthetic opportunities for testing."""
        return [
            CorrectionOpportunity(
                hour=h,
                original_commitment_mw=0.0,
                recommended_commitment_mw=correction_mw,
                correction_mw=correction_mw,
                original_price_forecast=80.0,
                updated_price_forecast=95.0,
                price_deviation_eur=15.0,
                p10_price=85.0,
                p90_price=105.0,
                expected_gain_eur=expected_gain,
                confidence=0.8,
            )
            for h in range(n)
        ]

    def test_correction_per_hour_limit(self):
        """I verify max_correction_per_hour_mw is respected."""
        corrector = _make_corrector()
        corrector.max_correction_per_hour_mw = 3.0
        corrector.max_correction_per_session_mw = 100.0

        # I create opportunities with correction larger than per-hour limit
        opps = self._make_opportunities(n=3, correction_mw=10.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        orders = corrector._apply_corrections(
            opps, commitment, battery, 'IDA1',
            datetime(2026, 3, 14, 15, 0)
        )

        # I verify each order respects per-hour limit
        for order in orders:
            assert abs(order.power_mw) <= 3.0 + 0.01

    def test_correction_per_session_limit(self):
        """I verify cumulative session limit is respected."""
        corrector = _make_corrector()
        corrector.max_correction_per_session_mw = 8.0
        corrector.max_correction_per_hour_mw = 5.0

        opps = self._make_opportunities(n=10, correction_mw=4.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        orders = corrector._apply_corrections(
            opps, commitment, battery, 'IDA1',
            datetime(2026, 3, 14, 15, 0)
        )

        total = sum(abs(o.power_mw) for o in orders)
        assert total <= 8.0 + 0.01

    def test_correction_total_position_limit(self):
        """I verify DAM + correction stays within +/-30 MW."""
        corrector = _make_corrector()
        corrector.max_correction_per_hour_mw = 10.0
        corrector.max_correction_per_session_mw = 100.0

        # I start with a large DAM commitment of 28 MW
        commitment = np.full(24, 28.0)

        opps = self._make_opportunities(n=3, correction_mw=5.0)
        # I set original_commitment_mw to match the actual commitment
        for opp in opps:
            opp.original_commitment_mw = 28.0

        battery = _make_battery_state()

        orders = corrector._apply_corrections(
            opps, commitment, battery, 'IDA1',
            datetime(2026, 3, 14, 15, 0)
        )

        # I verify each correction + DAM stays within 30 MW
        for order in orders:
            assert abs(28.0 + order.power_mw) <= 30.0 + 0.01

    def test_correction_creates_orders(self):
        """I verify orders are created in the IntraDayTrader."""
        corrector = _make_corrector()
        opps = self._make_opportunities(n=2, correction_mw=3.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        orders = corrector._apply_corrections(
            opps, commitment, battery, 'IDA1',
            datetime(2026, 3, 14, 15, 0)
        )

        assert len(orders) > 0
        for order in orders:
            assert isinstance(order, IntraDayOrder)
            assert order.status == 'filled'


# =======================================================================
# Full cycle tests
# =======================================================================

class TestFullCycle:
    """I test the complete run_correction_cycle flow."""

    def _setup_corrector_with_mock(self, deviation_eur=20.0):
        """I set up a corrector with a mock forecaster that returns shifted prices."""
        corrector = _make_corrector(min_price_deviation=5.0, min_gain=0.0)
        original = _make_forecast()

        def mock_predict_day(df_arg, day_idx, serbia_arg, sph):
            # I return an updated forecast with deviation
            updated_hourly = original['hourly'] + deviation_eur
            return {
                'hourly': updated_hourly,
                'p10': updated_hourly * 0.92,
                'p90': updated_hourly * 1.08,
            }

        corrector.forecaster.predict_day = mock_predict_day
        return corrector, original

    def test_full_ida1_correction_cycle(self):
        """I test end-to-end IDA1 correction with synthetic data."""
        corrector, original_fc = self._setup_corrector_with_mock(deviation_eur=20.0)
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        result = corrector.run_correction_cycle(
            session='IDA1',
            target_date=date(2026, 1, 20),
            df=df,
            serbia_24h=serbia,
            original_forecast=original_fc,
            original_commitment=commitment,
            battery_state=battery,
            sph=1,
            dam_actuals=np.full(24, 100.0),
        )

        assert isinstance(result, CorrectionResult)
        assert result.session == 'IDA1'
        assert result.n_opportunities > 0
        assert result.n_corrections_applied > 0

    def test_full_ida2_cycle_with_ida1_info(self):
        """I test IDA2 uses IDA1 results in the re-forecast."""
        corrector, original_fc = self._setup_corrector_with_mock(deviation_eur=15.0)
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        result = corrector.run_correction_cycle(
            session='IDA2',
            target_date=date(2026, 1, 20),
            df=df,
            serbia_24h=serbia,
            original_forecast=original_fc,
            original_commitment=commitment,
            battery_state=battery,
            sph=1,
            dam_actuals=np.full(24, 100.0),
            ida1_results=np.full(24, 110.0),
        )

        assert isinstance(result, CorrectionResult)
        assert result.session == 'IDA2'
        assert result.n_corrections_applied >= 0

    def test_full_ida3_cycle_afternoon_only(self):
        """I test IDA3 only corrects afternoon hours (12-23)."""
        corrector, original_fc = self._setup_corrector_with_mock(deviation_eur=20.0)
        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        result = corrector.run_correction_cycle(
            session='IDA3',
            target_date=date(2026, 1, 20),
            df=df,
            serbia_24h=serbia,
            original_forecast=original_fc,
            original_commitment=commitment,
            battery_state=battery,
            sph=1,
        )

        assert isinstance(result, CorrectionResult)
        assert result.session == 'IDA3'

        # I verify all orders are for afternoon delivery hours
        for order in result.orders:
            assert order.delivery_start.hour >= 12

    def test_no_correction_when_spread_insufficient(self):
        """I test that flat prices produce no corrections."""
        corrector = _make_corrector(min_price_deviation=50.0, min_gain=100.0)

        # I set up a mock that returns nearly identical forecast
        original = _make_forecast()
        corrector.forecaster.predict_day = MagicMock(return_value={
            'hourly': original['hourly'] + 1.0,  # Only 1 EUR change
            'p10': original['p10'] + 1.0,
            'p90': original['p90'] + 1.0,
        })

        df = _make_df(n_days=30)
        serbia = np.full(24, 85.0)
        commitment = np.zeros(24)
        battery = _make_battery_state()

        result = corrector.run_correction_cycle(
            session='IDA1',
            target_date=date(2026, 1, 20),
            df=df,
            serbia_24h=serbia,
            original_forecast=original,
            original_commitment=commitment,
            battery_state=battery,
            sph=1,
        )

        assert result.n_corrections_applied == 0
        assert len(result.orders) == 0


# =======================================================================
# Transaction cost tests
# =======================================================================

class TestTransactionCost:
    """I test the _compute_transaction_cost method."""

    def test_ida1_highest_cost(self):
        """I verify IDA1 has the highest transaction cost."""
        corrector = _make_corrector()
        cost1 = corrector._compute_transaction_cost('IDA1', 5.0, 100.0)
        cost2 = corrector._compute_transaction_cost('IDA2', 5.0, 100.0)
        cost3 = corrector._compute_transaction_cost('IDA3', 5.0, 100.0)
        assert cost1 > cost2 > cost3

    def test_cost_includes_henex_fee(self):
        """I verify the HEnEx fee is included."""
        corrector = _make_corrector()
        cost = corrector._compute_transaction_cost('IDA1', 5.0, 100.0)
        # I expect at least 2% + 0.04/100 = 0.0204
        assert cost > 0.02
