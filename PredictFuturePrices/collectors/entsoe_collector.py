"""ENTSO-E data collector wrapping the existing EntsoeConnector.

I collect DAM prices (Greece + Serbia), load forecasts, and RES forecasts
from ENTSO-E Transparency Platform. All data is stored as incremental CSVs.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from PredictFuturePrices.config import ENTSOE, PATHS

# I add the project root to sys.path so I can import the existing connector
sys.path.insert(0, str(PATHS.data_dir.parent.parent))

logger = logging.getLogger(__name__)


def _get_connector():
    """Lazy import of EntsoeConnector to avoid hard dependency at module load."""
    from data.entsoe_connector import EntsoeConnector
    return EntsoeConnector(api_key=ENTSOE.api_key or None)


class EntsoeDAMCollector:
    """I collect Greek DAM prices from ENTSO-E."""

    def __init__(self, output_path: Optional[Path] = None):
        self.name = "entsoe_gr_dam"
        self.output_path = output_path or PATHS.gr_dam_csv
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        connector = _get_connector()
        start_ts = pd.Timestamp(start, tz="Europe/Athens")
        end_ts = pd.Timestamp(end, tz="Europe/Athens")

        df = connector.fetch_day_ahead_prices(start_ts, end_ts)

        # I normalize to naive local time
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Europe/Athens").tz_localize(None)
        df.index.name = "timestamp"
        return df

    def get_default_start(self) -> datetime:
        return datetime(2021, 1, 1)

    def fetch_incremental(self, end: Optional[datetime] = None) -> pd.DataFrame:
        from PredictFuturePrices.collectors.base_collector import BaseCollector

        # I create a temporary wrapper to reuse BaseCollector logic
        collector = _IncrementalWrapper(self)
        return collector.fetch_incremental(end)


class EntsoeSerbiaDAMCollector:
    """I collect Serbian DAM prices from ENTSO-E."""

    def __init__(self, output_path: Optional[Path] = None):
        self.name = "entsoe_rs_dam"
        self.output_path = output_path or PATHS.rs_dam_csv
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        connector = _get_connector()
        start_ts = pd.Timestamp(start, tz="Europe/Athens")
        end_ts = pd.Timestamp(end, tz="Europe/Athens")

        df = connector.fetch_neighbor_dam_prices("RS", start_ts, end_ts)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Europe/Athens").tz_localize(None)
        df.index.name = "timestamp"
        # I rename to distinguish from GR prices
        df.columns = ["rs_dam_price"]
        return df

    def get_default_start(self) -> datetime:
        return datetime(2021, 1, 1)


class EntsoeLoadForecastCollector:
    """I collect load forecast data from ENTSO-E."""

    def __init__(self, output_path: Optional[Path] = None):
        self.name = "entsoe_load_forecast"
        self.output_path = output_path or PATHS.load_forecast_csv
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        connector = _get_connector()
        start_ts = pd.Timestamp(start, tz="Europe/Athens")
        end_ts = pd.Timestamp(end, tz="Europe/Athens")

        df = connector.fetch_load_forecast(start_ts, end_ts)
        if df.empty:
            return df
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Europe/Athens").tz_localize(None)
        df.index.name = "timestamp"
        return df

    def get_default_start(self) -> datetime:
        return datetime(2023, 1, 1)


class EntsoeRESForecastCollector:
    """I collect wind/solar forecast data from ENTSO-E."""

    def __init__(self, output_path: Optional[Path] = None):
        self.name = "entsoe_res_forecast"
        self.output_path = output_path or PATHS.res_forecast_csv
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        connector = _get_connector()
        start_ts = pd.Timestamp(start, tz="Europe/Athens")
        end_ts = pd.Timestamp(end, tz="Europe/Athens")

        df = connector.fetch_wind_solar_forecast(start_ts, end_ts)
        if df.empty:
            return df
        if df.index.tz is not None:
            df.index = df.index.tz_convert("Europe/Athens").tz_localize(None)
        df.index.name = "timestamp"
        return df

    def get_default_start(self) -> datetime:
        return datetime(2023, 1, 1)


class _IncrementalWrapper:
    """I wrap simple collector classes to use BaseCollector incremental logic."""

    def __init__(self, collector):
        self.collector = collector

    def fetch_incremental(self, end: Optional[datetime] = None) -> pd.DataFrame:
        from PredictFuturePrices.collectors.base_collector import BaseCollector

        if end is None:
            end = datetime.now()

        # I check last timestamp on disk
        last = self._get_last_timestamp()
        if last is not None:
            from datetime import timedelta
            start = last + timedelta(hours=1)
        else:
            start = self.collector.get_default_start()

        if start >= end:
            logger.info(f"[{self.collector.name}] Already up to date.")
            return pd.DataFrame()

        logger.info(f"[{self.collector.name}] Fetching {start.date()} → {end.date()}")

        try:
            new_data = self.collector.fetch_range(start, end)
        except Exception as e:
            logger.error(f"[{self.collector.name}] Fetch failed: {e}")
            return pd.DataFrame()

        if not new_data.empty:
            self._save_incremental(new_data)

        return new_data

    def _get_last_timestamp(self) -> Optional[datetime]:
        path = self.collector.output_path
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty:
                return None
            last = pd.Timestamp(df.index.max())
            if last.tzinfo is not None:
                last = last.tz_localize(None)
            return last.to_pydatetime()
        except Exception:
            return None

    def _save_incremental(self, new_data: pd.DataFrame) -> None:
        path = self.collector.output_path
        new_data.index.name = "timestamp"
        if new_data.index.tz is not None:
            new_data.index = new_data.index.tz_localize(None)

        if path.exists():
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            if existing.index.tz is not None:
                existing.index = existing.index.tz_localize(None)
            merged = pd.concat([existing, new_data])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
        else:
            merged = new_data.sort_index()

        merged.to_csv(path)
        logger.info(f"[{self.collector.name}] Saved {len(merged)} rows to {path.name}")
