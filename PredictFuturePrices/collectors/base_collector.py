"""Abstract base class for all data collectors.

I define the contract that every collector must follow:
- fetch_incremental(): fetch only new data since last collection
- get_last_timestamp(): check what we already have
- save_incremental(): append new rows without duplicates
- load_existing(): load previously collected data
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base for all data collectors.

    I enforce a consistent pattern: check what's on disk, fetch only the delta,
    and merge without duplicates. Subclasses implement the actual API calls.
    """

    def __init__(self, name: str, output_path: Path):
        self.name = name
        self.output_path = output_path
        # I ensure the output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch data for a specific date range from the external source.

        Subclasses implement the actual API call here.

        Args:
            start: Start datetime (inclusive).
            end: End datetime (inclusive).

        Returns:
            DataFrame with DatetimeIndex, columns depend on source.
        """
        ...

    @abstractmethod
    def get_default_start(self) -> datetime:
        """Return the earliest date this source supports.

        I use this when no existing data is found on disk.
        """
        ...

    def get_last_timestamp(self) -> Optional[datetime]:
        """Check the most recent timestamp in existing data on disk.

        Returns:
            Last timestamp as naive datetime, or None if no data exists.
        """
        if not self.output_path.exists():
            return None

        try:
            df = pd.read_csv(
                self.output_path,
                index_col=0,
                parse_dates=True,
                nrows=0,  # I only need the schema to validate
            )
            # I read just the tail for efficiency on large files
            df_tail = pd.read_csv(
                self.output_path,
                index_col=0,
                parse_dates=True,
            )
            if df_tail.empty:
                return None
            last = pd.Timestamp(df_tail.index.max())
            # I strip timezone info to keep everything naive
            if last.tzinfo is not None:
                last = last.tz_localize(None)
            return last.to_pydatetime()
        except Exception as e:
            logger.warning(f"[{self.name}] Could not read last timestamp: {e}")
            return None

    def load_existing(self) -> pd.DataFrame:
        """Load the full existing dataset from disk.

        Returns:
            DataFrame with DatetimeIndex, or empty DataFrame if file missing.
        """
        if not self.output_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.output_path, index_col=0, parse_dates=True)
            # I normalize to naive timestamps
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "timestamp"
            return df
        except Exception as e:
            logger.warning(f"[{self.name}] Could not load existing data: {e}")
            return pd.DataFrame()

    def save_incremental(self, new_data: pd.DataFrame) -> int:
        """Merge new data with existing and save without duplicates.

        I use a simple strategy: load existing, concat, drop duplicate indices
        keeping the latest values, sort, and write back.

        Args:
            new_data: New rows to append.

        Returns:
            Number of new rows actually added.
        """
        if new_data.empty:
            logger.info(f"[{self.name}] No new data to save.")
            return 0

        # I normalize the new data index
        new_data = new_data.copy()
        new_data.index.name = "timestamp"
        if new_data.index.tz is not None:
            new_data.index = new_data.index.tz_localize(None)

        existing = self.load_existing()
        rows_before = len(existing)

        if existing.empty:
            merged = new_data
        else:
            # I concat and drop duplicates, keeping new values (last)
            merged = pd.concat([existing, new_data])
            merged = merged[~merged.index.duplicated(keep="last")]

        merged = merged.sort_index()
        rows_added = len(merged) - rows_before

        merged.to_csv(self.output_path)
        logger.info(
            f"[{self.name}] Saved {len(merged)} rows "
            f"({rows_added} new) to {self.output_path.name}"
        )
        return rows_added

    def fetch_incremental(self, end: Optional[datetime] = None) -> pd.DataFrame:
        """Fetch only data newer than what's already on disk.

        I check the last timestamp on disk, then fetch from there to now.
        If no data exists, I start from get_default_start().

        Args:
            end: End datetime. Defaults to now.

        Returns:
            The newly fetched DataFrame (before merging with existing).
        """
        if end is None:
            end = datetime.now()

        last = self.get_last_timestamp()
        if last is not None:
            # I start 1 hour after last known point to avoid re-fetching
            start = last + timedelta(hours=1)
        else:
            start = self.get_default_start()

        if start >= end:
            logger.info(f"[{self.name}] Already up to date (last={last}).")
            return pd.DataFrame()

        logger.info(
            f"[{self.name}] Fetching {start.date()} → {end.date()}"
        )

        try:
            new_data = self.fetch_range(start, end)
        except Exception as e:
            logger.error(f"[{self.name}] Fetch failed: {e}")
            return pd.DataFrame()

        if not new_data.empty:
            self.save_incremental(new_data)

        return new_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, path={self.output_path})"
