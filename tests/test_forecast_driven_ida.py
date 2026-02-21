"""
Tests for _generate_forecast_driven_ida_positions in data/create_unified_training_data.py

I test that forecast-error-driven IDA positions are correlated with price errors,
respect magnitude ranges, and maintain backward compatibility.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from data.create_unified_training_data import (
    _generate_forecast_driven_ida_positions,
    _generate_ida_positions,
)


# --- Helpers ---

def _make_test_df(n=2400, sph=1, seed=42):
    """
    I create a test DataFrame with realistic price dynamics.

    Args:
        n: Number of rows (default 2400 = 100 days hourly)
        sph: Steps per hour
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    start = pd.Timestamp('2025-01-01')
    freq = '1h' if sph == 1 else '15min'
    idx = pd.date_range(start, periods=n, freq=freq)

    # I create a realistic sinusoidal + trend + noise price pattern
    base = 80 + 20 * np.sin(2 * np.pi * np.arange(n) / (24 * sph))
    # I add day-to-day variation (autocorrelated shocks)
    daily_shock = np.zeros(n)
    for i in range(1, n):
        daily_shock[i] = 0.95 * daily_shock[i - 1] + np.random.normal(0, 3)
    prices = base + daily_shock + np.random.normal(0, 2, n)

    # I create a DAM commitment with charge/discharge cycles
    dam = np.zeros(n)
    hours = idx.hour
    dam[hours < 6] = -15.0    # Charge at night
    dam[(hours >= 17) & (hours <= 21)] = 20.0  # Discharge in evening

    df = pd.DataFrame({
        'price': prices,
        'dam_commitment': dam,
    }, index=idx)

    return df


# =======================================================================
# Tests
# =======================================================================

class TestForecastDrivenIDA:
    """I test the forecast-driven IDA position generator."""

    def test_positions_correlated_with_forecast_error(self):
        """I verify the sign of IDA1 correction matches forecast error sign."""
        df = _make_test_df(n=2400, seed=42)
        result = _generate_forecast_driven_ida_positions(df.copy(), max_power_mw=30.0)

        # I check that the forecast error proxy column exists
        assert 'dam_forecast_error_proxy' in result.columns

        error = result['dam_forecast_error_proxy'].values
        ida1 = result['ida1_position'].values

        # I only compare non-zero positions where error is significant
        mask = (np.abs(error) > 8.0) & (np.abs(ida1) > 0.1)
        if mask.sum() > 10:
            # I compute sign correlation: positive error -> positive correction
            signs_match = np.sign(error[mask]) == np.sign(ida1[mask])
            match_rate = signs_match.mean()
            # I expect at least 50% sign agreement (noise may reduce it)
            assert match_rate > 0.5, f"Sign match rate {match_rate:.2f} too low"

    def test_ida1_range_within_5mw(self):
        """I verify IDA1 positions are clipped to [-5, 5] MW."""
        df = _make_test_df(n=2400, seed=42)
        result = _generate_forecast_driven_ida_positions(df.copy(), max_power_mw=30.0)

        assert result['ida1_position'].max() <= 5.0 + 0.01
        assert result['ida1_position'].min() >= -5.0 - 0.01

    def test_ida2_range_within_3mw(self):
        """I verify IDA2 positions are clipped to [-3, 3] MW."""
        df = _make_test_df(n=2400, seed=42)
        result = _generate_forecast_driven_ida_positions(df.copy(), max_power_mw=30.0)

        assert result['ida2_position'].max() <= 3.0 + 0.01
        assert result['ida2_position'].min() >= -3.0 - 0.01

    def test_ida3_only_afternoon(self):
        """I verify IDA3 positions are zero for hours < 12."""
        df = _make_test_df(n=2400, seed=42)
        result = _generate_forecast_driven_ida_positions(df.copy(), max_power_mw=30.0)

        morning_mask = result.index.hour < 12
        morning_ida3 = result.loc[morning_mask, 'ida3_position']
        assert (morning_ida3 == 0.0).all(), "IDA3 must be zero before 12:00"

    def test_total_position_within_limits(self):
        """I verify total (DAM + IDA) position stays within +/-30 MW."""
        df = _make_test_df(n=2400, seed=42)
        result = _generate_forecast_driven_ida_positions(df.copy(), max_power_mw=30.0)

        total = (result['dam_commitment'] + result['ida1_position']
                 + result['ida2_position'] + result['ida3_position'])

        # I allow a small floating-point tolerance
        assert total.max() <= 30.0 + 0.1
        assert total.min() >= -30.0 - 0.1

    def test_zero_correction_below_threshold(self):
        """I verify corrections are zero when forecast error is small."""
        # I create a DataFrame with near-constant prices (low forecast error)
        n = 720
        idx = pd.date_range('2025-01-01', periods=n, freq='1h')
        df = pd.DataFrame({
            'price': 80.0 + np.random.normal(0, 0.5, n),  # Very stable prices
            'dam_commitment': np.zeros(n),
        }, index=idx)

        result = _generate_forecast_driven_ida_positions(
            df.copy(), max_power_mw=30.0, min_deviation_eur=8.0
        )

        # I expect most IDA1 positions to be near zero (only noise component)
        # The raw correction should be zero; only noise remains
        assert result['ida1_position'].std() < 2.0, "IDA1 should be mostly noise for stable prices"

    def test_forecast_error_proxy_column_exists(self):
        """I verify the dam_forecast_error_proxy column is added."""
        df = _make_test_df(n=720, seed=42)
        result = _generate_forecast_driven_ida_positions(df.copy(), max_power_mw=30.0)

        assert 'dam_forecast_error_proxy' in result.columns
        assert not result['dam_forecast_error_proxy'].isna().all()

    def test_backward_compat_old_function(self):
        """I verify the old random function still works and produces expected columns."""
        df = _make_test_df(n=720, seed=42)
        result = _generate_ida_positions(df.copy(), max_power_mw=30.0)

        assert 'ida1_position' in result.columns
        assert 'ida2_position' in result.columns
        assert 'ida3_position' in result.columns
        assert 'net_ida_position' in result.columns

        # I verify IDA3 is zero for morning hours
        morning_mask = result.index.hour < 12
        assert (result.loc[morning_mask, 'ida3_position'] == 0.0).all()
