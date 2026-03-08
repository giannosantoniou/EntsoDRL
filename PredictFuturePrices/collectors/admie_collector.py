"""ADMIE data collector wrapping the existing ADMIEDataFetcher.

I collect ISP clearing prices, balancing market data, system load,
and RES production from ADMIE (Greek TSO). Data is stored incrementally.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from PredictFuturePrices.config import ADMIE, PATHS

sys.path.insert(0, str(PATHS.data_dir.parent.parent))

logger = logging.getLogger(__name__)


def _get_fetcher(cache_dir: Optional[Path] = None):
    """Lazy import of ADMIEDataFetcher."""
    from data.admie_data_fetcher import ADMIEDataFetcher
    cache = str(cache_dir) if cache_dir else str(PATHS.admie_cache_dir)
    return ADMIEDataFetcher(cache_dir=cache)


class AdmieBalancingCollector:
    """I collect ADMIE balancing market data (ISP, IMBABE, activations)."""

    def __init__(self, output_path: Optional[Path] = None, cache_dir: Optional[Path] = None):
        self.name = "admie_balancing"
        self.output_path = output_path or PATHS.admie_balancing_csv
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir or PATHS.admie_cache_dir

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """I fetch all ADMIE balancing data for the given range.

        I use the existing ADMIEDataFetcher.fetch_all() which handles:
        - ISP1/2/3 capacity prices
        - IMBABE (mFRR energy prices, imbalance, system deviation)
        - Balancing energy activations (mFRR/aFRR volumes)
        - System load and RES production
        - Load and RES forecasts
        """
        fetcher = _get_fetcher(self.cache_dir)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        df = fetcher.fetch_all(start_str, end_str, resample_to="1h")
        if df.empty:
            return df

        # I normalize index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "timestamp"
        return df

    def get_default_start(self) -> datetime:
        return datetime(2024, 1, 1)

    def fetch_incremental(self, end: Optional[datetime] = None) -> pd.DataFrame:
        """I fetch only new data since last collection."""
        if end is None:
            end = datetime.now()

        last = self._get_last_timestamp()
        if last is not None:
            from datetime import timedelta
            start = last + timedelta(hours=1)
        else:
            start = self.get_default_start()

        if start >= end:
            logger.info(f"[{self.name}] Already up to date.")
            return pd.DataFrame()

        logger.info(f"[{self.name}] Fetching {start.date()} → {end.date()}")

        try:
            new_data = self.fetch_range(start, end)
        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return pd.DataFrame()

        if not new_data.empty:
            self._save_incremental(new_data)

        return new_data

    def _get_last_timestamp(self) -> Optional[datetime]:
        if not self.output_path.exists():
            return None
        try:
            df = pd.read_csv(self.output_path, index_col=0, parse_dates=True)
            if df.empty:
                return None
            last = pd.Timestamp(df.index.max())
            if last.tzinfo is not None:
                last = last.tz_localize(None)
            return last.to_pydatetime()
        except Exception:
            return None

    def _save_incremental(self, new_data: pd.DataFrame) -> None:
        new_data.index.name = "timestamp"
        if new_data.index.tz is not None:
            new_data.index = new_data.index.tz_localize(None)

        if self.output_path.exists():
            existing = pd.read_csv(self.output_path, index_col=0, parse_dates=True)
            if existing.index.tz is not None:
                existing.index = existing.index.tz_localize(None)
            merged = pd.concat([existing, new_data])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index()
        else:
            merged = new_data.sort_index()

        merged.to_csv(self.output_path)
        logger.info(f"[{self.name}] Saved {len(merged)} rows to {self.output_path.name}")


class AdmieISPCollector:
    """I collect ISP clearing prices separately for detailed analysis."""

    def __init__(self, output_path: Optional[Path] = None, cache_dir: Optional[Path] = None):
        self.name = "admie_isp"
        self.output_path = output_path or PATHS.isp_clearing_csv
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir or PATHS.admie_cache_dir

    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """I fetch only ISP1/2/3 clearing prices (15-min resolution)."""
        fetcher = _get_fetcher(self.cache_dir)

        # I use the ISP parser directly for 15-min granularity
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        dfs = []
        for isp_num in [1, 2, 3]:
            category = f"isp{isp_num}"
            try:
                isp_df = fetcher._fetch_and_parse_category(
                    category,
                    start_ts, end_ts,
                    fetcher._parse_isp_file,
                )
                if isp_df is not None and not isp_df.empty:
                    # I prefix columns with isp number
                    isp_df.columns = [f"isp{isp_num}_{c}" for c in isp_df.columns]
                    dfs.append(isp_df)
            except Exception as e:
                logger.warning(f"[{self.name}] ISP{isp_num} fetch failed: {e}")

        if not dfs:
            return pd.DataFrame()

        merged = pd.concat(dfs, axis=1)
        if merged.index.tz is not None:
            merged.index = merged.index.tz_localize(None)
        merged.index.name = "timestamp"
        return merged

    def get_default_start(self) -> datetime:
        return datetime(2024, 1, 1)
