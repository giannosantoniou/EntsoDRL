"""Tests for Phase 2: data collectors, scheduler, and data merger.

I test collectors using mocked APIs to avoid real network calls.
I test the merger with synthetic CSVs.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from PredictFuturePrices.collectors.base_collector import BaseCollector
from PredictFuturePrices.collectors.weather_collector import WeatherCollector
from PredictFuturePrices.collectors.commodity_collector import CommodityCollector
from PredictFuturePrices.data_merger import DataMerger


# ─── Weather Collector Tests ────────────────────────────────────────────


class TestWeatherCollector:
    """I test the Open-Meteo weather data collector."""

    def _mock_response(self, start_date="2024-01-01", n_hours=48):
        """I create a realistic Open-Meteo JSON response."""
        timestamps = pd.date_range(start_date, periods=n_hours, freq="h")
        return {
            "hourly": {
                "time": [t.isoformat() for t in timestamps],
                "temperature_2m": list(np.random.randn(n_hours) * 5 + 15),
                "wind_speed_100m": list(np.random.rand(n_hours) * 20),
                "direct_radiation": list(np.random.rand(n_hours) * 800),
                "cloud_cover": list(np.random.rand(n_hours) * 100),
            }
        }

    def test_parse_response(self, tmp_path):
        collector = WeatherCollector(output_path=tmp_path / "weather.csv")
        data = self._mock_response()
        df = collector._parse_response(data)

        assert not df.empty
        assert "temperature" in df.columns
        assert "wind_100m" in df.columns
        assert "solar_radiation" in df.columns
        assert "cloud_cover" in df.columns
        assert len(df) == 48

    def test_parse_empty_response(self, tmp_path):
        collector = WeatherCollector(output_path=tmp_path / "weather.csv")
        df = collector._parse_response({})
        assert df.empty

    def test_parse_response_missing_variables(self, tmp_path):
        collector = WeatherCollector(output_path=tmp_path / "weather.csv")
        data = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                "temperature_2m": [15.0, 16.0],
                # I intentionally omit other variables
            }
        }
        df = collector._parse_response(data)
        assert not df.empty
        assert "temperature" in df.columns
        assert len(df) == 2

    @patch("requests.Session.get")
    def test_fetch_archive(self, mock_get, tmp_path):
        collector = WeatherCollector(output_path=tmp_path / "weather.csv")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self._mock_response()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        df = collector._fetch_archive(
            datetime(2024, 1, 1),
            datetime(2024, 1, 3),
        )
        assert not df.empty
        assert mock_get.called

    @patch("requests.Session.get")
    def test_fetch_archive_failure(self, mock_get, tmp_path):
        collector = WeatherCollector(output_path=tmp_path / "weather.csv")
        mock_get.side_effect = Exception("Network error")

        df = collector._fetch_archive(
            datetime(2024, 1, 1),
            datetime(2024, 1, 3),
        )
        assert df.empty

    def test_default_start(self, tmp_path):
        collector = WeatherCollector(output_path=tmp_path / "weather.csv")
        assert collector.get_default_start() == datetime(2023, 1, 1)

    def test_incremental_save_and_load(self, tmp_path):
        collector = WeatherCollector(output_path=tmp_path / "weather.csv")

        # I create synthetic weather data
        idx = pd.date_range("2024-01-01", periods=24, freq="h")
        df1 = pd.DataFrame({
            "temperature": np.random.randn(24) * 5 + 15,
            "wind_100m": np.random.rand(24) * 20,
        }, index=idx)

        collector.save_incremental(df1)
        loaded = collector.load_existing()
        assert len(loaded) == 24

        # I add more data
        idx2 = pd.date_range("2024-01-02", periods=24, freq="h")
        df2 = pd.DataFrame({
            "temperature": np.random.randn(24) * 5 + 15,
            "wind_100m": np.random.rand(24) * 20,
        }, index=idx2)

        collector.save_incremental(df2)
        loaded = collector.load_existing()
        assert len(loaded) == 48


# ─── Commodity Collector Tests ──────────────────────────────────────────


class TestCommodityCollector:
    """I test the Yahoo Finance commodity collector."""

    def test_default_start(self, tmp_path):
        collector = CommodityCollector(output_path=tmp_path / "commodity.csv")
        assert collector.get_default_start() == datetime(2023, 1, 1)

    @patch("PredictFuturePrices.collectors.commodity_collector._import_yfinance")
    def test_fetch_range_with_mock(self, mock_yf_import, tmp_path):
        """I test commodity fetching with mocked yfinance."""
        collector = CommodityCollector(output_path=tmp_path / "commodity.csv")

        # I mock the yfinance module
        mock_yf = MagicMock()
        mock_yf_import.return_value = mock_yf

        # I create mock download results
        daily_idx = pd.date_range("2024-01-01", periods=30, freq="B")
        mock_ttf = pd.DataFrame(
            {"Close": np.random.rand(30) * 30 + 25},
            index=daily_idx,
        )
        mock_eua = pd.DataFrame(
            {"Close": np.random.rand(30) * 20 + 60},
            index=daily_idx,
        )

        mock_yf.download.side_effect = [mock_ttf, mock_eua]

        df = collector.fetch_range(
            datetime(2024, 1, 1),
            datetime(2024, 2, 1),
        )

        assert not df.empty
        # I verify hourly resolution
        assert df.index.freq == "h" or len(df) > 30

    def test_incremental_pattern(self, tmp_path):
        """I test the incremental save/load cycle."""
        collector = CommodityCollector(output_path=tmp_path / "commodity.csv")

        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        df = pd.DataFrame({
            "ttf_gas": np.random.rand(48) * 30 + 25,
            "eua_carbon": np.random.rand(48) * 20 + 60,
        }, index=idx)

        collector.save_incremental(df)
        loaded = collector.load_existing()
        assert len(loaded) == 48
        assert "ttf_gas" in loaded.columns


# ─── Data Merger Tests ──────────────────────────────────────────────────


class TestDataMerger:
    """I test the CSV merge logic."""

    def _create_test_csvs(self, data_dir: Path, n_hours: int = 100):
        """I create minimal test CSVs for merge testing."""
        idx = pd.date_range("2024-01-01", periods=n_hours, freq="h")

        # I create GR DAM prices (primary)
        dam = pd.DataFrame({"price": np.sin(np.arange(n_hours) * 0.1) * 30 + 80}, index=idx)
        dam.index.name = "timestamp"
        dam_path = data_dir / "entsoe" / "gr_dam_prices.csv"
        dam_path.parent.mkdir(parents=True, exist_ok=True)
        dam.to_csv(dam_path)

        # I create RS DAM prices
        rs = pd.DataFrame({"rs_dam_price": dam["price"] + np.random.randn(n_hours) * 5}, index=idx)
        rs.index.name = "timestamp"
        rs_path = data_dir / "entsoe" / "rs_dam_prices.csv"
        rs.to_csv(rs_path)

        # I create weather data
        wx = pd.DataFrame({
            "temperature": np.random.randn(n_hours) * 5 + 15,
            "wind_100m": np.random.rand(n_hours) * 20,
            "solar_radiation": np.random.rand(n_hours) * 800,
            "cloud_cover": np.random.rand(n_hours) * 100,
        }, index=idx)
        wx.index.name = "timestamp"
        wx_path = data_dir / "weather" / "weather_hourly.csv"
        wx_path.parent.mkdir(parents=True, exist_ok=True)
        wx.to_csv(wx_path)

        # I create commodity data
        cm = pd.DataFrame({
            "ttf_gas": np.random.rand(n_hours) * 30 + 25,
            "eua_carbon": np.random.rand(n_hours) * 20 + 60,
        }, index=idx)
        cm.index.name = "timestamp"
        cm_path = data_dir / "commodities" / "commodity_prices.csv"
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        cm.to_csv(cm_path)

        return dam_path, rs_path, wx_path, cm_path

    def test_merge_all(self, tmp_path):
        """I test the full merge pipeline."""
        data_dir = tmp_path / "data"
        self._create_test_csvs(data_dir)

        # I override PATHS for the merger
        from PredictFuturePrices import config
        original_paths = config.PATHS

        merger = DataMerger(data_dir=data_dir)
        # I manually set paths to test directory
        merger_paths = type(original_paths)(
            data_dir=data_dir,
            gr_dam_csv=data_dir / "entsoe" / "gr_dam_prices.csv",
            rs_dam_csv=data_dir / "entsoe" / "rs_dam_prices.csv",
            weather_csv=data_dir / "weather" / "weather_hourly.csv",
            commodity_csv=data_dir / "commodities" / "commodity_prices.csv",
            load_forecast_csv=data_dir / "entsoe" / "load_forecast.csv",
            res_forecast_csv=data_dir / "entsoe" / "res_forecast.csv",
            admie_balancing_csv=data_dir / "admie" / "admie_balancing.csv",
            merged_csv=data_dir / "merged.csv",
        )

        # I monkey-patch the merger to use test paths
        import PredictFuturePrices.data_merger as merger_module
        old_paths = merger_module.PATHS
        merger_module.PATHS = merger_paths
        try:
            merged = merger.merge_all(output_path=data_dir / "merged.csv")
        finally:
            merger_module.PATHS = old_paths

        assert not merged.empty
        assert "dam_price_gr" in merged.columns
        assert "rs_dam_price" in merged.columns
        assert "temperature" in merged.columns
        assert "ttf_gas" in merged.columns

        # I verify derived features were added
        assert "hour_sin" in merged.columns
        assert "dam_price_gr_mean_24h" in merged.columns
        assert "rs_gr_spread" in merged.columns

        # I verify the merged CSV was saved
        assert (data_dir / "merged.csv").exists()

    def test_merge_dam_only(self, tmp_path):
        """I test merge works with only DAM prices (minimum viable)."""
        data_dir = tmp_path / "data"
        idx = pd.date_range("2024-01-01", periods=50, freq="h")
        dam = pd.DataFrame({"price": np.random.rand(50) * 50 + 60}, index=idx)
        dam.index.name = "timestamp"

        dam_path = data_dir / "entsoe" / "gr_dam_prices.csv"
        dam_path.parent.mkdir(parents=True, exist_ok=True)
        dam.to_csv(dam_path)

        from PredictFuturePrices import config
        merger = DataMerger(data_dir=data_dir)

        merger_paths = type(config.PATHS)(
            data_dir=data_dir,
            gr_dam_csv=dam_path,
            rs_dam_csv=data_dir / "nonexistent.csv",
            weather_csv=data_dir / "nonexistent.csv",
            commodity_csv=data_dir / "nonexistent.csv",
            load_forecast_csv=data_dir / "nonexistent.csv",
            res_forecast_csv=data_dir / "nonexistent.csv",
            admie_balancing_csv=data_dir / "nonexistent.csv",
            merged_csv=data_dir / "merged.csv",
        )

        import PredictFuturePrices.data_merger as merger_module
        old_paths = merger_module.PATHS
        merger_module.PATHS = merger_paths
        try:
            merged = merger.merge_all(output_path=data_dir / "merged.csv")
        finally:
            merger_module.PATHS = old_paths

        assert not merged.empty
        assert "dam_price_gr" in merged.columns
        assert len(merged) == 50

    def test_derived_features_calendar(self, tmp_path):
        """I verify calendar features are correct."""
        data_dir = tmp_path / "data"
        idx = pd.date_range("2024-06-15 12:00", periods=1, freq="h")
        dam = pd.DataFrame({"price": [80.0]}, index=idx)
        dam.index.name = "timestamp"

        merger = DataMerger(data_dir=data_dir)
        result = merger._add_derived_features(dam.rename(columns={"price": "dam_price_gr"}))

        # I verify hour_sin for noon (12:00)
        expected_sin = np.sin(2 * np.pi * 12 / 24)
        assert result["hour_sin"].iloc[0] == pytest.approx(expected_sin)

        # I verify June 15, 2024 is Saturday → is_weekend=1
        assert result["is_weekend"].iloc[0] == 1.0

    def test_load_csv_missing_file(self, tmp_path):
        merger = DataMerger(data_dir=tmp_path)
        df = merger._load_csv(tmp_path / "nonexistent.csv", "test")
        assert df.empty

    def test_rolling_stats(self, tmp_path):
        """I verify rolling statistics are computed correctly."""
        data_dir = tmp_path / "data"
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        df = pd.DataFrame({
            "dam_price_gr": np.ones(48) * 100,
        }, index=idx)

        merger = DataMerger(data_dir=data_dir)
        result = merger._add_derived_features(df)

        # I verify mean of constant series is the constant itself
        assert result["dam_price_gr_mean_24h"].iloc[-1] == pytest.approx(100.0)
        # I verify std of constant series is ~0
        assert result["dam_price_gr_std_24h"].iloc[-1] == pytest.approx(0.0, abs=1e-10)

    def test_lag_features(self, tmp_path):
        """I verify lag features shift correctly."""
        data_dir = tmp_path / "data"
        idx = pd.date_range("2024-01-01", periods=200, freq="h")
        prices = np.arange(200, dtype=float)
        df = pd.DataFrame({"dam_price_gr": prices}, index=idx)

        merger = DataMerger(data_dir=data_dir)
        result = merger._add_derived_features(df)

        # I verify 1-day lag = value 24 hours ago
        assert result["dam_price_gr_lag_1d"].iloc[48] == pytest.approx(24.0)
        # I verify 7-day lag
        assert result["dam_price_gr_lag_7d"].iloc[200 - 1] == pytest.approx(200 - 1 - 168)
